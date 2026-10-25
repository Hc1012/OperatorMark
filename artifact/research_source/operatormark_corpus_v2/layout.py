"""corpus_v2 tensor inventory, sample/direction indexing and file layout.

Revision 3 D7 as corrected by Addendum A.  Pure arithmetic: no solves, no seeds.

* Splits, counts, offsets and the 64-sample shard follow v1 (§§6, 8, 9); the
  directory names follow v1 (`test.sealed`).
* Per-sample families are the seven v1 families with `u_tpfa64` renamed `u_f64`.
* Per-direction families: `dir`, `du_f64`, `du32_areaavg`, `du16_areaavg`.
  One direction per training sample, two per validation/test sample, so
  training shards hold 64 directions and validation/test shards hold 128.
* Native T_{N_ref} arrays `u_tpfa{N}` (per sample) and `du_tpfa{N}` (per
  direction) are stored only under retention option S.  Under E they are
  hash-only.  kappa_N, the P1-256 nodal solution and its 64-grid projection
  are hash-only under both options (Addendum A3, v1 §8).

A `Layout` is the production layout (4096/256/512, shard 64), the expressly
labelled real-development layout (4096/256/0, shard 64), or a synthetic one.
Synthetic layouts exist for engineering tests,
are labelled in every record, and are refused by production generation.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterator

import numpy as np

BENCHMARK = "darcy_branch_c_v2"
SPLITS = ("training", "validation", "test")
SPLIT_DIRECTORIES = {"training": "training", "validation": "validation", "test": "test.sealed"}
PRODUCTION_COUNTS = (4096, 256, 512)
PRODUCTION_SHARD_SIZE = 64
DIRECTIONS_PER_SAMPLE = {"training": 1, "validation": 2, "test": 2}
N_REF_OPTIONS = (128, 256)
RETENTION_OPTIONS = ("E", "S")
STAGING = "corpus_staging"
F8 = 8

# (family, side) in protocol order; position in a shard manifest is a protocol fact.
SAMPLE_FAMILIES = (("rho", 8), ("kappa64", 64), ("u_f64", 64), ("kappa32_areaavg", 32), ("u32_areaavg", 32),
                   ("kappa16_areaavg", 16), ("u16_areaavg", 16))
DIRECTION_FAMILIES = (("dir", 8), ("du_f64", 64), ("du32_areaavg", 32), ("du16_areaavg", 16))
V1_RENAMES = {"u_f64": "u_tpfa64"}


def native_forward_family(n_ref: int) -> str:
    return f"u_tpfa{n_ref}"


def native_tangent_family(n_ref: int) -> str:
    return f"du_tpfa{n_ref}"


def file_name(family: str) -> str:
    return f"{family}.float64.npy"


def check_n_ref(n_ref: object) -> int:
    if type(n_ref) is not int or n_ref not in N_REF_OPTIONS:
        raise ValueError(f"N_ref must be explicitly 128 or 256, got {n_ref!r}")
    return n_ref


def check_retention(option: object) -> str:
    if type(option) is not str or option not in RETENTION_OPTIONS:
        raise ValueError(f"retention must be explicitly 'E' or 'S', got {option!r}")
    return option


def retention_classes(n_ref: int, option: str) -> dict[str, str]:
    """What is stored and what is hash-only, by array class."""

    check_n_ref(n_ref)
    stored = check_retention(option) == "S"
    return {
        native_forward_family(n_ref): "stored" if stored else "hash_only",
        native_tangent_family(n_ref): "stored" if stored else "hash_only",
        f"kappa{n_ref}": "hash_only",
        "p1_256_nodal": "hash_only",
        "p1_256_projected64": "hash_only",
    }


@dataclass(frozen=True)
class Layout:
    """Split sizes and shard size.  `synthetic` must be True unless this is the production layout."""

    counts: tuple[int, int, int]
    shard_size: int
    synthetic: bool
    real_development: bool = False

    def __post_init__(self):
        counts = tuple(self.counts)
        if len(counts) != 3 or any(type(c) is not int or c < 0 for c in counts):
            raise ValueError("counts must be three non-negative ints (training, validation, test)")
        if type(self.shard_size) is not int or self.shard_size < 1:
            raise ValueError("shard_size must be a positive int")
        if any(c % self.shard_size for c in counts):
            raise ValueError("every split count must be a multiple of the shard size")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be a bool")
        production = counts == PRODUCTION_COUNTS and self.shard_size == PRODUCTION_SHARD_SIZE
        if self.real_development and (self.synthetic or counts != (4096, 256, 0) or self.shard_size != 64):
            raise ValueError('real development is exactly 4096/256/0, shard 64, non-synthetic')
        if not production and not self.synthetic and not self.real_development:
            raise ValueError("a non-production layout must be declared synthetic")
        object.__setattr__(self, "counts", counts)

    @property
    def is_production(self) -> bool:
        return not self.synthetic and self.counts == PRODUCTION_COUNTS and self.shard_size == PRODUCTION_SHARD_SIZE

    # ---------------------------------------------------------------- counts
    def count(self, split: str) -> int:
        return self.counts[SPLITS.index(split)]

    def shards(self, split: str) -> int:
        return self.count(split) // self.shard_size

    def sample_offset(self, split: str) -> int:
        return sum(self.counts[:SPLITS.index(split)])

    def direction_offset(self, split: str) -> int:
        index = SPLITS.index(split)
        return sum(self.counts[i] * DIRECTIONS_PER_SAMPLE[SPLITS[i]] for i in range(index))

    @property
    def total_samples(self) -> int:
        return sum(self.counts)

    @property
    def total_directions(self) -> int:
        return sum(self.count(s) * DIRECTIONS_PER_SAMPLE[s] for s in SPLITS)

    def all_shards(self) -> list[tuple[str, int]]:
        return [(split, shard) for split in SPLITS for shard in range(self.shards(split))]

    def record(self) -> dict:
        return {**({'real_development': True} if self.real_development else {}),
                "counts": {s: self.count(s) for s in SPLITS}, "shard_size": self.shard_size,
                "directions_per_sample": dict(DIRECTIONS_PER_SAMPLE), "split_directories": dict(SPLIT_DIRECTORIES),
                "synthetic": self.synthetic, "production_layout": self.is_production}

    # ---------------------------------------------------------------- identities
    def sample_identity(self, split: str, local: int) -> dict:
        if split not in SPLITS or type(local) is not int or not 0 <= local < self.count(split):
            raise ValueError(f"no sample {split}/{local}")
        shard, offset = divmod(local, self.shard_size)
        return {"benchmark": BENCHMARK, "split": split, "shard_index": shard, "shard_offset": offset,
                "local_index": local, "global_index": self.sample_offset(split) + local,
                "sample_id": f"{BENCHMARK}/{split}/{local:06d}",
                "stratum": "relaxed" if local % 2 == 0 else "binary_32_of_64"}

    def direction_identity(self, split: str, local: int, k: int) -> dict:
        sample = self.sample_identity(split, local)
        d = DIRECTIONS_PER_SAMPLE[split]
        if type(k) is not int or not 0 <= k < d:
            raise ValueError(f"direction index {k} out of range for {split}")
        direction_local = local * d + k
        return {"benchmark": BENCHMARK, "split": split, "shard_index": sample["shard_index"],
                "shard_position": sample["shard_offset"] * d + k, "sample_local_index": local,
                "sample_global_index": sample["global_index"], "sample_id": sample["sample_id"], "direction_k": k,
                "direction_local_index": direction_local,
                "direction_global_index": self.direction_offset(split) + direction_local,
                "direction_id": f"{sample['sample_id']}/d{k}"}

    def shard_samples(self, split: str, shard: int) -> range:
        if not 0 <= shard < self.shards(split):
            raise ValueError(f"no shard {split}/{shard}")
        return range(shard * self.shard_size, (shard + 1) * self.shard_size)

    def iter_shard_directions(self, split: str, shard: int) -> Iterator[tuple[int, int]]:
        for local in self.shard_samples(split, shard):
            for k in range(DIRECTIONS_PER_SAMPLE[split]):
                yield local, k

    # ---------------------------------------------------------------- files
    def shard_dir(self, split: str, shard: int) -> str:
        return f"{STAGING}/{SPLIT_DIRECTORIES[split]}/shard_{shard:04d}"

    def shard_families(self, split: str, n_ref: int, option: str) -> list[tuple[str, str, tuple[int, ...]]]:
        """[(family, kind, shard array shape)] in manifest order for this retention option."""

        check_n_ref(n_ref)
        stored = check_retention(option) == "S"
        b = self.shard_size
        d = DIRECTIONS_PER_SAMPLE[split]
        out = [(name, "sample", (b, 1, side, side)) for name, side in SAMPLE_FAMILIES]
        out += [(name, "direction", (b * d, 1, side, side)) for name, side in DIRECTION_FAMILIES]
        if stored:
            out.append((native_forward_family(n_ref), "native_sample", (b, 1, n_ref, n_ref)))
            out.append((native_tangent_family(n_ref), "native_direction", (b * d, 1, n_ref, n_ref)))
        return out


PRODUCTION_LAYOUT = Layout(PRODUCTION_COUNTS, PRODUCTION_SHARD_SIZE, synthetic=False)


def npy_header_bytes(shape: tuple[int, ...]) -> int:
    """Length of the .npy v2.0 header NumPy's writer emits for a C-order <f8 array of this shape."""

    buffer = io.BytesIO()
    np.lib.format.write_array_header_2_0(buffer, {"descr": "<f8", "fortran_order": False, "shape": tuple(shape)})
    return len(buffer.getvalue())


@dataclass(frozen=True)
class FamilyInventory:
    family: str
    kind: str                 # sample | direction | native_sample | native_direction | hash_only
    item_shape: tuple[int, ...]
    item_bytes: int
    items: int
    raw_bytes: int            # 0 for hash-only classes
    files: int
    header_bytes: int
    retention: str            # stored | hash_only


def tensor_inventory(layout: Layout, n_ref: int, option: str) -> list[FamilyInventory]:
    """Exact per-family inventory (stored families and hash-only classes)."""

    check_n_ref(n_ref)
    check_retention(option)
    rows: list[FamilyInventory] = []
    samples, directions = layout.total_samples, layout.total_directions
    files = sum(layout.shards(s) for s in SPLITS)

    def headers(kind: str, side: int) -> int:
        total = 0
        for split in SPLITS:
            per = layout.shard_size * (DIRECTIONS_PER_SAMPLE[split] if "direction" in kind else 1)
            total += layout.shards(split) * npy_header_bytes((per, 1, side, side))
        return total

    for name, side in SAMPLE_FAMILIES:
        item = side * side * F8
        rows.append(FamilyInventory(name, "sample", (1, side, side), item, samples, item * samples, files,
                                    headers("sample", side), "stored"))
    for name, side in DIRECTION_FAMILIES:
        item = side * side * F8
        rows.append(FamilyInventory(name, "direction", (1, side, side), item, directions, item * directions, files,
                                    headers("direction", side), "stored"))
    item = n_ref * n_ref * F8
    stored = option == "S"
    for name, kind, count in ((native_forward_family(n_ref), "native_sample", samples),
                              (native_tangent_family(n_ref), "native_direction", directions)):
        rows.append(FamilyInventory(name, kind, (1, n_ref, n_ref), item, count, item * count if stored else 0,
                                    files if stored else 0, headers(kind, n_ref) if stored else 0,
                                    "stored" if stored else "hash_only"))
    rows.append(FamilyInventory(f"kappa{n_ref}", "hash_only", (1, n_ref, n_ref), item, samples, 0, 0, 0, "hash_only"))
    rows.append(FamilyInventory("p1_256_nodal", "hash_only", (257, 257), 257 * 257 * F8, samples, 0, 0, 0, "hash_only"))
    rows.append(FamilyInventory("p1_256_projected64", "hash_only", (64, 64), 64 * 64 * F8, samples, 0, 0, 0, "hash_only"))
    return rows
