"""D6 feasible directions, finite-difference stencil choice and FD checks.

Shared consumer: works only through the FiniteDifferenceOracle protocol and
64-grid arrays.  Direction construction and normalisation use correctly
rounded operations (abs, sign, math.fsum, math.sqrt, division), so a
direction is a platform-independent function of its Gaussian row (level L2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .api import FiniteDifferenceOracle

NEAR_BOUND = 0.02          # [C] read inclusively: rho <= 0.02 -> v >= 0; rho >= 0.98 -> v <= 0
STENCIL_STEP = 1.0e-2      # stencil chosen once per sample at the largest h
FD_STEPS = (1.0e-2, 1.0e-3, 1.0e-4)
SLOPE_WINDOW = (1.5, 2.5)


def feasible_direction(rho: np.ndarray, gaussian: np.ndarray) -> np.ndarray:
    r = np.asarray(rho, dtype=np.float64).reshape(64)
    g = np.asarray(gaussian, dtype=np.float64).reshape(64)
    if not (np.isfinite(r).all() and np.isfinite(g).all()):
        raise ValueError("inputs must be finite")
    lower = r <= NEAR_BOUND
    upper = r >= 1.0 - NEAR_BOUND
    v = g.copy()
    v[lower] = np.abs(g[lower])
    v[upper] = -np.abs(g[upper])
    norm = math.sqrt(math.fsum(float(x) * float(x) for x in v))
    if not norm > 0.0:
        raise ValueError("degenerate direction")
    out = v / np.float64(norm)
    return np.ascontiguousarray(out.reshape(8, 8))


def perturbed(rho: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(rho, dtype=np.float64) + np.float64(t) * np.asarray(v, dtype=np.float64))


def _inside(x: np.ndarray) -> bool:
    return bool(np.all(x >= 0.0) and np.all(x <= 1.0))


def choose_stencil(rho: np.ndarray, v: np.ndarray) -> str:
    h = STENCIL_STEP
    if _inside(perturbed(rho, v, h)) and _inside(perturbed(rho, v, -h)):
        return "central"
    return "forward3"


def stencil_points(stencil: str, h: float) -> tuple[tuple[float, float], ...]:
    """(offset t, weight) pairs; estimate = sum(weight * F(rho + t v)) / (2h)."""

    if stencil == "central":
        return ((h, 1.0), (-h, -1.0))
    if stencil == "forward3":
        return ((0.0, -3.0), (h, 4.0), (2.0 * h, -1.0))
    raise ValueError(stencil)


def fd_estimate(oracle: FiniteDifferenceOracle, rho: np.ndarray, v: np.ndarray, level: int, h: float, stencil: str,
                cache: dict | None = None) -> np.ndarray:
    total = np.zeros((64, 64), dtype=np.float64)
    for offset, weight in stencil_points(stencil, h):
        key = float(offset)
        if cache is not None and key in cache:
            value = cache[key]
        else:
            value = oracle.field64(perturbed(rho, v, offset), level)
            if cache is not None:
                cache[key] = value
        total = total + np.float64(weight) * value
    return total / np.float64(2.0 * h)


def norm64(y: np.ndarray) -> float:
    return math.sqrt(float(np.sum(y * y, dtype=np.float64) / np.float64(4096)))


@dataclass(frozen=True)
class FDCheck:
    stencil: str
    steps: tuple[float, ...]
    raw_errors: tuple[float, ...]       # norm64(FD_h - tangent), unnormalised
    slope: float                        # log10(e(1e-2)/e(1e-3))
    passes: bool


def fd_check(oracle: FiniteDifferenceOracle, rho: np.ndarray, v: np.ndarray, level: int,
             tangent64: np.ndarray, steps: tuple[float, ...] = FD_STEPS) -> tuple[FDCheck, dict[float, np.ndarray]]:
    stencil = choose_stencil(rho, v)
    estimates: dict[float, np.ndarray] = {}
    errors = []
    base_cache: dict = {}
    for h in steps:
        # f(rho) is shared between steps for the one-sided stencil; other points are not.
        cache = {0.0: base_cache[0.0]} if 0.0 in base_cache else {}
        est = fd_estimate(oracle, rho, v, level, h, stencil, cache)
        if 0.0 in cache:
            base_cache[0.0] = cache[0.0]
        estimates[h] = est
        errors.append(norm64(est - tangent64))
    e1, e2 = errors[0], errors[1]
    slope = math.log10(e1 / e2) if e1 > 0.0 and e2 > 0.0 else float("nan")
    passes = math.isfinite(slope) and SLOPE_WINDOW[0] <= slope <= SLOPE_WINDOW[1]
    return FDCheck(stencil, tuple(steps), tuple(errors), slope, passes), estimates
