"""v1 §8.1 derived views and the TPFA restriction R (N -> 64).

Every function here is elementwise IEEE arithmetic in an explicit order, so
results are platform-independent (equality level L2):

* view32: first-term initialisation, then + (0,1), + (1,0), + (1,1), times 0.25
  (identical to final_scientific_verify.area_view / final_verify._derive_area_view,
  including signed zero);
* view16: start from +0.0, add the 16 terms in a-major, b-minor order, times 0.0625;
* restrict_to_64: the view32 rule applied once (N=128) or twice (N=256).

darcy_spec.restrict_tpfa64 is deliberately NOT used: its reshape-sum is
pairwise and differs from §8.1 at the ulp level.
"""

from __future__ import annotations

import math

import numpy as np

_EPS = float(np.finfo(np.float64).eps)


def _as_field(z: np.ndarray) -> np.ndarray:
    if not isinstance(z, np.ndarray) or z.dtype != np.float64:
        raise TypeError("field must be a float64 numpy array")
    if z.ndim < 2 or z.shape[-1] != z.shape[-2]:
        raise ValueError("field must have square trailing axes")
    if not np.isfinite(z).all():
        raise ValueError("field must be finite")
    return np.ascontiguousarray(z)


def _halve(z: np.ndarray) -> np.ndarray:
    """One application of the 2x2 rule on the trailing axes."""

    if z.shape[-1] % 2:
        raise ValueError("trailing size must be even")
    total = z[..., 0::2, 0::2].copy()
    total = total + z[..., 0::2, 1::2]
    total = total + z[..., 1::2, 0::2]
    total = total + z[..., 1::2, 1::2]
    return np.ascontiguousarray(total * np.float64(0.25))


def view32(z64: np.ndarray) -> np.ndarray:
    z = _as_field(z64)
    if z.shape[-1] != 64:
        raise ValueError("view32 needs a 64x64 field")
    return _halve(z)


def view16(z64: np.ndarray) -> np.ndarray:
    z = _as_field(z64)
    if z.shape[-1] != 64:
        raise ValueError("view16 needs a 64x64 field")
    total = np.zeros(z.shape[:-2] + (16, 16), dtype=np.float64)
    for a in range(4):
        for b in range(4):
            total = total + z[..., a::4, b::4]
    return np.ascontiguousarray(total * np.float64(0.0625))


def restrict_to_64(z: np.ndarray, n: int) -> np.ndarray:
    """D2's R: exact cell average from an N-grid (N in 64, 128, 256) to the 64-grid."""

    field = _as_field(z)
    if field.shape[-1] != n or n not in (64, 128, 256):
        raise ValueError("restrict_to_64 supports N = 64, 128, 256 with matching shape")
    out = field.copy()
    while out.shape[-1] > 64:
        out = _halve(out)
    return np.ascontiguousarray(out)


def restrict_transpose(w64: np.ndarray, n: int) -> np.ndarray:
    """Exact transpose of the linear map R (N -> 64): spread w with weight 1/(N/64)^2."""

    w = _as_field(w64)
    if w.shape[-1] != 64 or n not in (64, 128, 256):
        raise ValueError("restrict_transpose supports a 64-grid input and N = 64, 128, 256")
    block = n // 64
    weight = np.float64(1.0) if block == 1 else (np.float64(0.25) if block == 2 else np.float64(0.0625))
    spread = np.repeat(np.repeat(w, block, axis=-2), block, axis=-1)
    return np.ascontiguousarray(spread * weight)


def transfer_integral_error(native: np.ndarray, n: int, field64: np.ndarray) -> float:
    """|integral(field64) - integral(native)| relative to max(1, integral scale)."""

    original = float(np.sum(native, dtype=np.float64)) / float(n * n)
    projected = float(np.sum(field64, dtype=np.float64)) / 4096.0
    scale = float(np.sum(np.abs(native), dtype=np.float64)) / float(n * n)
    return abs(projected - original) / max(1.0, scale)


TRANSFER_TOLERANCE = 64.0 * _EPS


def checked_restrict_to_64(z: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    field = restrict_to_64(z, n)
    error = transfer_integral_error(z, n, field)
    if not math.isfinite(error) or error > TRANSFER_TOLERANCE:
        raise ValueError("restriction failed its integral-preservation check")
    return field, error
