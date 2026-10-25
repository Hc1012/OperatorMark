"""D4 lock builder: designed cases, the §6 sampler, direction normals, starts,
128-bit seed rules, a public-material deny-list, the private manifest and its
shareable variant.

Nothing here has been run with real seeds.  Building the real lock is a
separate approval and happens on the owner's machine only.  Test-mode seed
files are accepted for engineering tests; every runner rejects test-mode
material unless it is itself in test mode.

What this module does NOT establish: freshness or disjointness in general.
Seed entropy plus a commitment show that material was fixed before use and is
reproducible from the committed seeds.  The deny-list only excludes *public*
material (the development seed, its 32 controls, the v1 fixtures and the
designed cases).  Disjointness from private v1/pilot seeds rests on the range
rule (formal seeds >= 2**64) plus the owner's confirmation that those seeds are
below 2**64; this module never reads them.

Seed roles: six streams (QS/CS x controls, directions, starts) rather than the
spec's four, so that CS material is independent of QS counts and can be opened
once after N_ref is frozen.  Recorded as a deliberate deviation for the owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from operatormark.corpus import canonical

from . import directions as _directions

SCHEMA = "operatormark_numerics_v2.lock_manifest.v1"
SHAREABLE_SCHEMA = "operatormark_numerics_v2.lock_manifest_shareable.v1"
SEED_DOMAIN = "operatormark_numerics_v2/lock/seed-commitment/v1"
FORMAL_ROLES = ("qs_controls", "qs_directions", "qs_starts", "cs_controls", "cs_directions", "cs_starts")
FORMAL_SEED_MIN = 2**64
FORMAL_SEED_LIMIT = 2**128
REQUIRED_NUMPY = "2.5.2"
DIRECTIONS_PER_CONTROL = 4
QS_RANDOM_COUNT = 16
CS_COUNT = 16
STARTS_LOW, STARTS_HIGH = 0.2, 0.8   # [C] pilot start procedure
PUBLIC_DEVELOPMENT_SEED = 2026083101  # darcy_spec.DEVELOPMENT_SEED (public in the frozen source)
PUBLIC_DEVELOPMENT_COUNT = 32


def _grid(fn) -> np.ndarray:
    out = np.empty((8, 8), dtype=np.float64)
    for j in range(8):
        for i in range(8):
            out[j, i] = fn(j, i)
    return out


# Exposed v1 fixtures (development evidence; never relabelled as fresh).
FIXTURES: dict[str, np.ndarray] = {
    "constant_half": _grid(lambda j, i: 0.5),
    "vertical_jump": _grid(lambda j, i: float(i >= 4)),
    "checkerboard": _grid(lambda j, i: float((i + j) % 2)),
}

# [C] Proposed designed binary cases ([y,x] = (j,i), u[j,i] = u(x_i, y_j)).
DESIGNED_CASES: dict[str, np.ndarray] = {
    "corner_high": _grid(lambda j, i: 1.0 if (j, i) == (0, 0) else 0.0),
    "interior_low": _grid(lambda j, i: 0.0 if (j, i) == (3, 3) else 1.0),
    "stripes": _grid(lambda j, i: float(i % 2)),
    "staircase": _grid(lambda j, i: 1.0 if i > j else 0.0),
}


class LockError(ValueError):
    pass


@dataclass(frozen=True)
class QualificationSeed:
    """The only way to hand a seed to the generic samplers in this module.

    mode 'formal': value must be a 128-bit draw >= 2**64 and the role a formal role.
    mode 'test'  : any value < 2**128 (engineering tests; runners reject test material outside test mode).
    """

    role: str
    value: int
    mode: str

    def __post_init__(self):
        if self.mode not in ("formal", "test"):
            raise LockError("seed mode must be 'formal' or 'test'")
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            raise LockError("seed value must be an int")
        if self.mode == "formal":
            if self.role not in FORMAL_ROLES:
                raise LockError(f"unknown formal role {self.role}")
            if not FORMAL_SEED_MIN <= self.value < FORMAL_SEED_LIMIT:
                raise LockError(f"formal seed for {self.role} must be a 128-bit draw >= 2**64")
        elif not 0 <= self.value < FORMAL_SEED_LIMIT:
            raise LockError(f"test seed for {self.role} out of range")

    def generator(self) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(self.value))


def stratum_of(rho: np.ndarray) -> str:
    """[C] binary if every value is exactly 0 or 1, otherwise relaxed."""

    values = np.asarray(rho, dtype=np.float64)
    return "binary" if bool(np.all((values == 0.0) | (values == 1.0))) else "relaxed"


def _require_seed(seed) -> QualificationSeed:
    if not isinstance(seed, QualificationSeed):
        raise LockError("samplers accept only QualificationSeed values")
    return seed


def sample_section6(seed: QualificationSeed, count: int) -> np.ndarray:
    """v1 §6: one PCG64 generator; even indices Uniform[0,1) 8x8; odd indices exact 32/32 binary."""

    generator = _require_seed(seed).generator()
    controls = np.empty((count, 8, 8), dtype=np.float64, order="C")
    for index in range(count):
        if index % 2 == 0:
            controls[index] = generator.uniform(0.0, 1.0, size=(8, 8))
        else:
            sample = np.zeros(64, dtype=np.float64)
            permutation = generator.permutation(64)
            sample[permutation[:32]] = 1.0
            controls[index] = sample.reshape((8, 8), order="C")
    return controls


def sample_starts(seed: QualificationSeed, count: int) -> np.ndarray:
    """Pilot start procedure: one PCG64 stream, Uniform(0.2, 0.8) per component."""

    generator = _require_seed(seed).generator()
    return np.ascontiguousarray(generator.uniform(STARTS_LOW, STARTS_HIGH, size=(count, 64)).reshape(count, 8, 8))


def direction_normals(seed: QualificationSeed, controls: int, per_control: int) -> np.ndarray:
    generator = _require_seed(seed).generator()
    normals = np.empty((controls, per_control, 64), dtype=np.float64)
    for c in range(controls):
        for k in range(per_control):
            normals[c, k] = generator.standard_normal(64)
    return normals


def feasible_directions(controls: np.ndarray, normals: np.ndarray) -> np.ndarray:
    out = np.empty(normals.shape[:2] + (8, 8), dtype=np.float64)
    for c in range(normals.shape[0]):
        for k in range(normals.shape[1]):
            out[c, k] = _directions.feasible_direction(controls[c], normals[c, k])
    return out


def seed_commitment(role: str, seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(SEED_DOMAIN.encode("utf-8") + b"\x00" + role.encode("utf-8") + b"\x00")
    digest.update(int(seed).to_bytes(16, "big"))
    return digest.hexdigest()


def public_deny_list() -> dict[str, set]:
    """Public material fresh controls must not reproduce (hashes only)."""

    development = sample_section6(QualificationSeed("development", PUBLIC_DEVELOPMENT_SEED, "test"), PUBLIC_DEVELOPMENT_COUNT)
    controls = {canonical.tensor_sha256(c) for c in development}
    controls |= {canonical.tensor_sha256(c) for c in FIXTURES.values()}
    controls |= {canonical.tensor_sha256(c) for c in DESIGNED_CASES.values()}
    return {"seeds": {PUBLIC_DEVELOPMENT_SEED}, "control_sha256": controls}


@dataclass(frozen=True)
class SeedSet:
    mode: str                       # "formal" | "test"
    seeds: dict[str, int]

    def seed(self, role: str) -> QualificationSeed:
        return QualificationSeed(role, self.seeds[role], self.mode)


def _decimal(text) -> int:
    if not isinstance(text, str) or not text or not text.isascii() or not text.isdigit():
        raise LockError("seeds must be ASCII decimal strings")
    return int(text)


def parse_seed_file(data: bytes) -> SeedSet:
    try:
        record = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockError("seed file is not UTF-8 JSON") from exc
    if not isinstance(record, dict) or set(record) != {"mode", "seeds"} or record["mode"] not in ("formal", "test"):
        raise LockError("seed file must contain exactly 'mode' (formal|test) and 'seeds'")
    if not isinstance(record["seeds"], dict) or set(record["seeds"]) != set(FORMAL_ROLES):
        raise LockError(f"seed roles must be exactly {FORMAL_ROLES}")
    seeds = {role: _decimal(text) for role, text in record["seeds"].items()}
    seed_set = SeedSet(record["mode"], seeds)
    for role in FORMAL_ROLES:
        seed_set.seed(role)                                  # range rules
    if record["mode"] == "formal":
        if len(set(seeds.values())) != len(seeds):
            raise LockError("formal seeds must be distinct")
        if set(seeds.values()) & public_deny_list()["seeds"]:
            raise LockError("formal seed is on the public deny-list")
    return seed_set


QS_ROLES = ("qs_controls", "qs_directions", "qs_starts")
CS_ROLES = ("cs_controls", "cs_directions", "cs_starts")
CS_SEALED_SCHEMA = "operatormark_numerics_v2.lock_cs_sealed.v1"
CS_SEALED_FILE = "cs_sealed.private.json"


def _check_deny_list(random_hashes: list[str], *, others: list[str] = ()) -> None:
    """Fresh (random) controls must avoid public material, each other and `others`."""

    deny = public_deny_list()["control_sha256"]
    if set(random_hashes) & deny:
        raise LockError("fresh control reproduces public material (deny-list hit)")
    if len(set(random_hashes)) != len(random_hashes) or set(random_hashes) & set(others):
        raise LockError("duplicate fresh control within or across QS and CS")


def build_qs(seed_set: SeedSet, *, qs_random: int = QS_RANDOM_COUNT, per_control: int = DIRECTIONS_PER_CONTROL,
             include_fixed: bool = True) -> dict:
    """QS material only: uses the three QS seed roles and nothing else."""

    qs_index, qs_controls = [], []
    if include_fixed:
        for name, control in FIXTURES.items():
            qs_index.append({"kind": "fixture", "name": name})
            qs_controls.append(control)
        for name, control in DESIGNED_CASES.items():
            qs_index.append({"kind": "designed", "name": name})
            qs_controls.append(control)
    random_qs = sample_section6(seed_set.seed("qs_controls"), qs_random)
    for k in range(qs_random):
        qs_index.append({"kind": "random", "name": f"qs_random_{k:02d}"})
        qs_controls.append(random_qs[k])
    qs = np.ascontiguousarray(np.stack(qs_controls))
    _check_deny_list([canonical.tensor_sha256(qs[i]) for i, row in enumerate(qs_index) if row["kind"] == "random"])
    qs_dirs = feasible_directions(qs, direction_normals(seed_set.seed("qs_directions"), qs.shape[0], per_control))
    qs_starts = sample_starts(seed_set.seed("qs_starts"), qs_random)
    for i, row in enumerate(qs_index):
        row.update(id=f"qs{i:02d}", stratum=stratum_of(qs[i]), control_sha256=canonical.tensor_sha256(qs[i]),
                   direction_sha256=[canonical.tensor_sha256(qs_dirs[i, k]) for k in range(per_control)])
    randoms = [row for row in qs_index if row["kind"] == "random"]
    for k, row in enumerate(randoms):
        row["start_sha256"] = canonical.tensor_sha256(qs_starts[k])
    return {"arrays": {"qs_controls": qs, "qs_directions": qs_dirs, "qs_starts": qs_starts}, "index": qs_index}


def build_cs(seed_set: SeedSet, *, cs_count: int = CS_COUNT, per_control: int = DIRECTIONS_PER_CONTROL,
             qs_random_hashes: list[str] = ()) -> dict:
    """CS (confirmation holdout) material only: uses the three CS seed roles."""

    cs = sample_section6(seed_set.seed("cs_controls"), cs_count)
    _check_deny_list([canonical.tensor_sha256(c) for c in cs], others=list(qs_random_hashes))
    cs_dirs = feasible_directions(cs, direction_normals(seed_set.seed("cs_directions"), cs.shape[0], per_control))
    cs_starts = sample_starts(seed_set.seed("cs_starts"), cs_count)
    cs_index = [{"id": f"cs{i:02d}", "kind": "random", "name": f"cs_random_{i:02d}", "stratum": stratum_of(cs[i]),
                 "control_sha256": canonical.tensor_sha256(cs[i]),
                 "direction_sha256": [canonical.tensor_sha256(cs_dirs[i, k]) for k in range(per_control)],
                 "start_sha256": canonical.tensor_sha256(cs_starts[i])} for i in range(cs_count)]
    return {"arrays": {"cs_controls": cs, "cs_directions": cs_dirs, "cs_starts": cs_starts}, "index": cs_index}


def _random_hashes(index: list) -> list[str]:
    return [row["control_sha256"] for row in index if row["kind"] == "random"]


def build_material(seed_set: SeedSet, *, qs_random: int = QS_RANDOM_COUNT, cs_count: int = CS_COUNT,
                   per_control: int = DIRECTIONS_PER_CONTROL, include_fixed: bool = True) -> dict:
    """Both partitions (used at lock creation only)."""

    if seed_set.mode == "formal" and (qs_random, cs_count, per_control, include_fixed) != (QS_RANDOM_COUNT, CS_COUNT, DIRECTIONS_PER_CONTROL, True):
        raise LockError("formal material uses the revision 3 counts only")
    qs = build_qs(seed_set, qs_random=qs_random, per_control=per_control, include_fixed=include_fixed)
    cs = build_cs(seed_set, cs_count=cs_count, per_control=per_control, qs_random_hashes=_random_hashes(qs["index"]))
    return {"arrays": dict(qs["arrays"], **cs["arrays"]), "qs_index": qs["index"], "cs_index": cs["index"],
            "counts": {"qs_random": qs_random, "cs": cs_count, "directions_per_control": per_control,
                       "include_fixed": include_fixed}}


def _file_entries(out_dir: Path, arrays: dict) -> dict:
    files = {}
    for name, array in arrays.items():
        data = canonical.write_npy_v2_exclusive(out_dir / "material" / f"{name}.npy", array)
        files[name] = {"file": f"material/{name}.npy", "file_sha256": canonical.sha256_bytes(data),
                       "tensor_sha256": canonical.tensor_sha256(array), "shape": list(array.shape)}
    return files


def _manifest_body(seed_set: SeedSet, material: dict, files: dict, cs_sealed_sha256: str) -> dict:
    return {
        "schema": SCHEMA,
        "mode": seed_set.mode,
        "numpy_version": np.__version__,
        "generator": "numpy.random.Generator(PCG64(seed)); one stream per role",
        "procedures": {
            "qs_controls": "v1 §6 sampler (even: Uniform[0,1) 8x8; odd: permutation(64), first 32 set to 1)",
            "cs_controls": "v1 §6 sampler (sealed)",
            "qs_directions": "standard_normal(64) per (control, direction) in canonical order -> D6 feasible direction",
            "cs_directions": "as qs_directions (sealed)",
            "qs_starts": "pilot procedure: uniform(0.2, 0.8, (count, 64)) -> [count,8,8]  [C]",
            "cs_starts": "as qs_starts (sealed)",
        },
        "seeds": {role: {"value": str(seed_set.seeds[role])} for role in QS_ROLES},
        "counts": material["counts"],
        "qs_index": material["qs_index"],
        "files": {k: v for k, v in files.items() if k.startswith("qs_")},
        "cs_sealed": {"file": CS_SEALED_FILE, "file_sha256": cs_sealed_sha256,
                      "note": "confirmation holdout: committed here, opened only by Q4 after the Q3 bindings are validated"},
        "deny_list": {"public_development_seed": True, "public_development_controls": PUBLIC_DEVELOPMENT_COUNT,
                      "v1_fixtures": True, "designed_cases": True, "qs_cs_disjoint": True},
        "disjointness_rule": "formal seeds are 128-bit draws >= 2**64; v1 corpus and pilot seeds are not read; "
                             "disjointness from them holds only if all of those are < 2**64 (owner confirmation required)",
        "freshness_note": "a commitment fixes material before use; it does not by itself prove freshness or disjointness",
    }


def _cs_sealed_body(seed_set: SeedSet, material: dict, files: dict) -> dict:
    return {"schema": CS_SEALED_SCHEMA, "mode": seed_set.mode,
            "seeds": {role: {"value": str(seed_set.seeds[role])} for role in CS_ROLES},
            "cs_index": material["cs_index"],
            "files": {k: v for k, v in files.items() if k.startswith("cs_")},
            "counts": {"cs": material["counts"]["cs"], "directions_per_control": material["counts"]["directions_per_control"]}}


def shareable_variant(private: dict, cs_seed_commitments: dict) -> dict:
    body = {k: v for k, v in private.items() if k not in ("seeds", "manifest_self_sha256")}
    body["schema"] = SHAREABLE_SCHEMA
    body["seed_commitments"] = {role: seed_commitment(role, int(private["seeds"][role]["value"])) for role in QS_ROLES}
    body["cs_seed_commitments"] = dict(cs_seed_commitments)
    body["private_manifest_self_sha256"] = private["manifest_self_sha256"]
    return canonical.with_self_hash(body, "manifest_self_sha256")


def write_lock(seed_set: SeedSet, out_dir: Path, **counts) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise LockError("lock output directory must not exist yet")
    material = build_material(seed_set, **counts)
    files = _file_entries(out_dir, material["arrays"])
    sealed = canonical.with_self_hash(_cs_sealed_body(seed_set, material, files), "sealed_self_sha256")
    sealed_bytes = canonical.write_canonical_json_exclusive(out_dir / CS_SEALED_FILE, sealed)
    private = canonical.with_self_hash(_manifest_body(seed_set, material, files, canonical.sha256_bytes(sealed_bytes)),
                                       "manifest_self_sha256")
    private_bytes = canonical.write_canonical_json_exclusive(out_dir / "lock_manifest.private.json", private)
    cs_commitments = {role: seed_commitment(role, seed_set.seeds[role]) for role in CS_ROLES}
    canonical.write_canonical_json_exclusive(out_dir / "lock_manifest.shareable.json", shareable_variant(private, cs_commitments))
    return {"commitment_sha256": canonical.sha256_bytes(private_bytes), "out_dir": str(out_dir)}


@dataclass(frozen=True)
class LockedMaterial:
    mode: str
    commitment_sha256: str
    manifest: dict                 # QS private manifest (never contains CS seeds or CS arrays)
    arrays: dict                   # QS arrays; CS arrays only after open_cs
    lock_dir: str
    cs: dict | None = None         # sealed CS record after open_cs

    @property
    def cs_opened(self) -> bool:
        return self.cs is not None


def _read_arrays(lock_dir: Path, files: dict, expected: set) -> dict:
    if set(files) != expected:
        raise LockError("material file inventory mismatch")
    arrays = {}
    for name, info in files.items():
        data = (lock_dir / info["file"]).read_bytes()
        if canonical.sha256_bytes(data) != info["file_sha256"]:
            raise LockError(f"material file hash mismatch: {name}")
        array = canonical.validate_npy_v2_bytes(data, expected_shape=tuple(info["shape"]))
        if canonical.tensor_sha256(array) != info["tensor_sha256"]:
            raise LockError(f"material tensor hash mismatch: {name}")
        arrays[name] = array
    return arrays


def _seed_set(mode: str, seeds: dict, roles: tuple) -> SeedSet:
    values = {role: _decimal(seeds[role]["value"]) for role in roles}
    seed_set = SeedSet(mode, values)
    for role in roles:
        seed_set.seed(role)                          # range rules
    if mode == "formal" and set(values.values()) & public_deny_list()["seeds"]:
        raise LockError("formal seed is on the public deny-list")
    return seed_set


def load_qs(lock_dir: Path, *, allow_test: bool) -> LockedMaterial:
    """Qualification-stage loading (Q1-Q3): QS seeds, QS files and QS regeneration only.

    The sealed CS file and the CS material files are not opened, read or regenerated.
    """

    lock_dir = Path(lock_dir)
    try:
        raw = (lock_dir / "lock_manifest.private.json").read_bytes()
        manifest = canonical.strict_canonical_json_loads(raw)
        canonical.validate_self_hash(manifest, "manifest_self_sha256")
        if manifest.get("schema") != SCHEMA:
            raise LockError("unexpected lock schema")
        if manifest["mode"] == "test" and not allow_test:
            raise LockError("test-mode lock material is rejected outside test mode")
        if manifest["mode"] == "formal" and allow_test:
            raise LockError("formal lock material is rejected in test mode")
        if set(manifest["seeds"]) != set(QS_ROLES):
            raise LockError("the private manifest must hold exactly the QS seed roles")
        seed_set = _seed_set(manifest["mode"], manifest["seeds"], QS_ROLES)
        arrays = _read_arrays(lock_dir, manifest["files"], {"qs_controls", "qs_directions", "qs_starts"})
        counts = manifest["counts"]
        if manifest["mode"] == "formal" and (counts["qs_random"], counts["directions_per_control"], counts["include_fixed"]) != (QS_RANDOM_COUNT, DIRECTIONS_PER_CONTROL, True):
            raise LockError("formal material uses the revision 3 counts only")
        regenerated = build_qs(seed_set, qs_random=counts["qs_random"], per_control=counts["directions_per_control"],
                               include_fixed=counts["include_fixed"])
        for name, array in regenerated["arrays"].items():
            if array.tobytes() != arrays[name].tobytes():
                raise LockError(f"material does not regenerate from the committed seeds: {name}")
        if regenerated["index"] != manifest["qs_index"]:
            raise LockError("material index does not regenerate")
        shareable_path = lock_dir / "lock_manifest.shareable.json"
        if shareable_path.exists():
            verify_shareable(manifest, shareable_path.read_bytes(), cs_seed_commitments=None)
    except (KeyError, TypeError, canonical.CanonicalRecordError, OSError) as exc:
        raise LockError(f"invalid lock material: {type(exc).__name__}: {exc}") from exc
    return LockedMaterial(manifest["mode"], canonical.sha256_bytes(raw), manifest, arrays, str(lock_dir))


def open_cs(locked: LockedMaterial) -> LockedMaterial:
    """Q4 only: open the sealed confirmation holdout and verify it against the commitment."""

    lock_dir = Path(locked.lock_dir)
    try:
        sealed_bytes = (lock_dir / locked.manifest["cs_sealed"]["file"]).read_bytes()
        if canonical.sha256_bytes(sealed_bytes) != locked.manifest["cs_sealed"]["file_sha256"]:
            raise LockError("sealed CS file does not match the committed hash")
        sealed = canonical.strict_canonical_json_loads(sealed_bytes)
        canonical.validate_self_hash(sealed, "sealed_self_sha256")
        if sealed.get("schema") != CS_SEALED_SCHEMA or sealed["mode"] != locked.mode:
            raise LockError("sealed CS record schema/mode mismatch")
        if set(sealed["seeds"]) != set(CS_ROLES):
            raise LockError("the sealed CS record must hold exactly the CS seed roles")
        seed_set = _seed_set(locked.mode, sealed["seeds"], CS_ROLES)
        if sealed["counts"]["directions_per_control"] != locked.manifest["counts"]["directions_per_control"] or \
                sealed["counts"]["cs"] != locked.manifest["counts"]["cs"]:
            raise LockError("sealed CS counts disagree with the manifest")
        arrays = _read_arrays(lock_dir, sealed["files"], {"cs_controls", "cs_directions", "cs_starts"})
        regenerated = build_cs(seed_set, cs_count=sealed["counts"]["cs"], per_control=sealed["counts"]["directions_per_control"],
                               qs_random_hashes=_random_hashes(locked.manifest["qs_index"]))
        for name, array in regenerated["arrays"].items():
            if array.tobytes() != arrays[name].tobytes():
                raise LockError(f"material does not regenerate from the committed seeds: {name}")
        if regenerated["index"] != sealed["cs_index"]:
            raise LockError("material index does not regenerate")
        shareable_path = lock_dir / "lock_manifest.shareable.json"
        if shareable_path.exists():
            commitments = {role: seed_commitment(role, seed_set.seeds[role]) for role in CS_ROLES}
            verify_shareable(locked.manifest, shareable_path.read_bytes(), cs_seed_commitments=commitments)
    except (KeyError, TypeError, canonical.CanonicalRecordError, OSError) as exc:
        raise LockError(f"invalid sealed CS material: {type(exc).__name__}: {exc}") from exc
    return LockedMaterial(locked.mode, locked.commitment_sha256, locked.manifest, dict(locked.arrays, **arrays),
                          locked.lock_dir, cs=sealed)


def verify_shareable(private: dict, shareable_bytes: bytes, *, cs_seed_commitments: dict | None) -> None:
    """QS stages check the QS part; Q4 (with the CS commitments recomputed from the sealed seeds) checks everything."""

    shared = canonical.strict_canonical_json_loads(shareable_bytes)
    canonical.validate_self_hash(shared, "manifest_self_sha256")
    claimed = shared.get("cs_seed_commitments", {})
    expected = shareable_variant(private, cs_seed_commitments if cs_seed_commitments is not None else claimed)
    if canonical.canonical_json_bytes(shared) != canonical.canonical_json_bytes(expected):
        raise LockError("shareable manifest does not derive from the private manifest")


def load_lock(lock_dir: Path, *, allow_test: bool) -> LockedMaterial:
    """Full verification of both partitions (lock-builder checks and tests; runners use load_qs / open_cs)."""

    return open_cs(load_qs(lock_dir, allow_test=allow_test))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="operatormark_numerics_v2.lock")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build lock material and manifests from a local seed file")
    build.add_argument("--seed-file", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if np.__version__ != REQUIRED_NUMPY:
        print(f"refusing: NumPy {REQUIRED_NUMPY} required, found {np.__version__}", file=sys.stderr)
        return 2
    seed_set = parse_seed_file(args.seed_file.read_bytes())
    result = write_lock(seed_set, args.out)
    print(json.dumps({"commitment_sha256": result["commitment_sha256"], "mode": seed_set.mode}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("use: python -m operatormark_numerics_v2.lock_cli build ... (checks frozen pins before import)")
