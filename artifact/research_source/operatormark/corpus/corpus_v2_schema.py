"""Addendum-A inventory, independent of Drop 2's eventual Python API.

Metadata only: no arrays, random draws, imports of numerical packages or State
execution. This is not a replacement for the frozen v1 format or qualification.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import Literal

from .corpus_format import SHARD_SIZE, SPLITS, SPLIT_SAMPLE_COUNTS, SPLIT_SHARD_COUNTS

ADDENDUM_SPEC_ZIP_SHA256 = '5d230a7423d641e85060c9d24cf4691846a6a58cd488873b0cb25a7255d9194a'


@dataclass(frozen=True, slots=True)
class V2Layout:
    n_ref: Literal[128, 256]
    retention: Literal['E', 'S']

    def __post_init__(self):
        if type(self.n_ref) is not int or self.n_ref not in (128, 256):
            raise ValueError('N_ref must be exactly 128 or 256')
        if type(self.retention) is not str or self.retention not in ('E', 'S'):
            raise ValueError('Addendum A requires E or S; mixed native retention is obsolete')


@dataclass(frozen=True, slots=True)
class V2TensorFamily:
    name: str
    side: int
    population: Literal['sample', 'direction']
    stored: bool

    def items_per_sample(self, split: str) -> int:
        if split not in SPLITS:
            raise ValueError('unknown split')
        return 2 if self.population == 'direction' and split != 'training' else 1

    def shard_shape(self, split: str) -> tuple[int, int, int, int]:
        return (SHARD_SIZE * self.items_per_sample(split), 1, self.side, self.side)

    def population_count(self, split: str) -> int:
        return SPLIT_SAMPLE_COUNTS[split] * self.items_per_sample(split)

    def raw_bytes(self, split: str) -> int:
        return self.population_count(split) * self.side * self.side * 8


def tensor_inventory(layout: V2Layout) -> tuple[V2TensorFamily, ...]:
    if type(layout) is not V2Layout:
        raise TypeError('exact validated V2Layout required')
    sample = [('rho', 8), ('kappa64', 64), ('u_f64', 64),
              ('kappa32_areaavg', 32), ('u32_areaavg', 32),
              ('kappa16_areaavg', 16), ('u16_areaavg', 16)]
    direction = [('dir', 8), ('du_f64', 64), ('du32_areaavg', 32), ('du16_areaavg', 16)]
    return tuple(V2TensorFamily(name, side, 'sample', True) for name, side in sample) + tuple(
        V2TensorFamily(name, side, 'direction', True) for name, side in direction) + (
        V2TensorFamily(f'u_tpfa{layout.n_ref}', layout.n_ref, 'sample', layout.retention == 'S'),
        V2TensorFamily(f'du_tpfa{layout.n_ref}', layout.n_ref, 'direction', layout.retention == 'S'),
        V2TensorFamily(f'kappa{layout.n_ref}', layout.n_ref, 'sample', False),
        V2TensorFamily('P256', 257, 'sample', False),
        V2TensorFamily('bar_P256', 64, 'sample', False))


def direction_keys(split: str, shard: int) -> tuple[tuple[int, int], ...]:
    """Local sample-major direction ordering; no private seeds or data."""
    if split not in SPLITS or type(shard) is not int or not 0 <= shard < SPLIT_SHARD_COUNTS[split]:
        raise ValueError('split/shard outside frozen population')
    return tuple((sample, direction) for sample in range(SHARD_SIZE)
                 for direction in range(1 if split == 'training' else 2))


def inventory_record(layout: V2Layout) -> dict:
    """A binding input, not a Drop 2 on-disk manifest/schema assignment."""
    return dict(schema='operatormark.corpus_v2.addendum_a.inventory.v1',
        specification_zip_sha256=ADDENDUM_SPEC_ZIP_SHA256,
        n_ref=layout.n_ref, retention=layout.retention, dtype='<f8', order='C', npy_version=[2, 0],
        splits={s: dict(samples=SPLIT_SAMPLE_COUNTS[s], shards=SPLIT_SHARD_COUNTS[s],
                       directions_per_sample=1 if s == 'training' else 2) for s in SPLITS},
        families=[dict(name=f.name, side=f.side, population=f.population, stored=f.stored,
                       shard_shapes={s: list(f.shard_shape(s)) for s in SPLITS})
                  for f in tensor_inventory(layout)])


def inventory_sha256(layout: V2Layout) -> str:
    return hashlib.sha256(json.dumps(inventory_record(layout), sort_keys=True,
        separators=(',', ':'), ensure_ascii=True).encode('ascii')).hexdigest()
