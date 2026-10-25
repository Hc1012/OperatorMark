"""D5 estimator: normalisation, ladders, V1-V4, estimates, statuses, N_ref rule,
confirmation and the anomaly bound.  Pure functions of 64-grid arrays/scalars.

Literal implementation of revision 3 D5 (as corrected by Addendum A; no rule
changes).  Budgets are inputs (D3 allocation confirmed before Q1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

QUANTITIES = ("field", "compliance", "derivative", "task")
VECTOR_QUANTITIES = ("field", "derivative")
PROPOSED_BUDGETS = {"field": 0.005, "compliance": 0.005, "derivative": 0.025, "task": 0.01}
V2_ORDER = (0.5, 3.0)
V3_FLOOR = 1.0e-11
V3_ETA_FACTOR = 100.0
V4_COS = 0.9
UNCERTAINTY_FACTOR = 1.25
FLOOR_FRACTION = 0.01


def norm64(y: np.ndarray) -> float:
    """Cell-area-weighted L2 on the 64 grid (v1 verifier form)."""

    return math.sqrt(float(np.sum(y * y, dtype=np.float64) / np.float64(4096)))


def inner64(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b, dtype=np.float64) / np.float64(4096))


def compliance(field64: np.ndarray) -> float:
    return float(np.sum(field64, dtype=np.float64) / np.float64(4096))


def derivative_denominator(du_t256_norm: float, m_s: float) -> tuple[float, bool]:
    floor = FLOOR_FRACTION * m_s
    return (max(du_t256_norm, floor), floor > du_t256_norm)


def measure(quantity: str, y) -> float:
    """Unnormalised size of a difference/correction for this quantity."""

    if quantity in VECTOR_QUANTITIES:
        return norm64(y)
    return abs(float(y))


@dataclass(frozen=True)
class LadderResult:
    quantity: str
    valid: bool
    failed: tuple[str, ...]
    d1: float               # normalised |y2 - y1|
    d2: float               # normalised |y3 - y2|
    r: float
    p: float
    alignment: float        # cos for vectors; +1/-1/0 sign agreement for scalars
    eta: float              # normalised, max of levels 2 and 3
    y_inf: object           # extrapolated limit (raw units); None if r outside (0, 1)
    e2: float
    e3: float
    U2: float
    U3: float


def ladder(quantity: str, y1, y2, y3, eta2: float, eta3: float, denom: float) -> LadderResult:
    if quantity not in QUANTITIES:
        raise ValueError(quantity)
    if not (denom > 0.0 and math.isfinite(denom)):
        raise ValueError("denominator must be positive and finite")
    vector = quantity in VECTOR_QUANTITIES
    if vector:
        d1v = np.asarray(y2, dtype=np.float64) - np.asarray(y1, dtype=np.float64)
        d2v = np.asarray(y3, dtype=np.float64) - np.asarray(y2, dtype=np.float64)
        n1, n2 = norm64(d1v) / denom, norm64(d2v) / denom
        alignment = inner64(d1v, d2v) / (norm64(d1v) * norm64(d2v)) if n1 > 0 and n2 > 0 else 0.0
    else:
        d1s, d2s = float(y2) - float(y1), float(y3) - float(y2)
        n1, n2 = abs(d1s) / denom, abs(d2s) / denom
        alignment = 1.0 if (d1s > 0 and d2s > 0) or (d1s < 0 and d2s < 0) else (-1.0 if d1s * d2s < 0 else 0.0)
    eta = max(eta2, eta3) / denom
    r = n2 / n1 if n1 > 0.0 else math.inf
    p = -math.log2(r) if 0.0 < r < math.inf else (math.inf if r == 0.0 else -math.inf)
    failed = []
    if not (0.0 < r < 1.0):
        failed.append("V1")
    if not (V2_ORDER[0] <= p <= V2_ORDER[1]):
        failed.append("V2")
    if not (n2 >= max(V3_FLOOR, V3_ETA_FACTOR * eta)):
        failed.append("V3")
    if vector:
        if not (alignment >= V4_COS):
            failed.append("V4")
    elif alignment != 1.0:
        failed.append("V4")
    if 0.0 < r < 1.0:
        factor = r / (1.0 - r)
        y_inf = (np.asarray(y3, dtype=np.float64) + d2v * np.float64(factor)) if vector else float(y3) + d2s * factor
        e2 = n2 / (1.0 - r)
        e3 = n2 * r / (1.0 - r)
    else:
        y_inf, e2, e3 = None, math.inf, math.inf
    return LadderResult(quantity, not failed, tuple(failed), n1, n2, r, p, alignment, eta, y_inf,
                        e2, e3, UNCERTAINTY_FACTOR * e2, UNCERTAINTY_FACTOR * e3)


@dataclass(frozen=True)
class CrossResult:
    complete: bool          # both ladders valid and limits consistent
    consistent: bool
    limit_gap: float        # normalised |y_inf^T - y_inf^P|
    e_hat: Mapping[int, float]   # reference error at levels 2 and 3 (TPFA levels T128, T256)


def cross(quantity: str, tpfa: LadderResult, p1: LadderResult, tpfa_levels: Sequence, denom: float) -> CrossResult:
    """tpfa_levels = (y1, y2, y3) raw TPFA values, for e_hat at levels 2 and 3."""

    if not (tpfa.valid and p1.valid):
        return CrossResult(False, False, math.inf, {2: math.inf, 3: math.inf})
    gap = measure(quantity, _diff(tpfa.y_inf, p1.y_inf)) / denom
    consistent = gap <= tpfa.U3 + p1.U3
    e_hat = {}
    for level, u_t in ((2, tpfa.U2), (3, tpfa.U3)):
        distance = measure(quantity, _diff(tpfa_levels[level - 1], p1.y_inf)) / denom
        e_hat[level] = max(u_t, distance + p1.U3)
    return CrossResult(consistent, consistent, gap, e_hat)


def _diff(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(a) - float(b)


@dataclass(frozen=True)
class QuantityStatus:
    quantity: str
    complete: bool
    status_by_level: Mapping[int, str]    # "qualified" | "over_budget" | "incomplete"
    failing_instances: tuple[str, ...]

    def qualified_at(self, level: int) -> bool:
        return self.status_by_level.get(level) == "qualified"


def quantity_status(quantity: str, instances: Mapping[str, CrossResult], budget: float) -> QuantityStatus:
    incomplete = tuple(sorted(name for name, c in instances.items() if not c.complete))
    if incomplete or not instances:
        return QuantityStatus(quantity, False, {2: "incomplete", 3: "incomplete"}, incomplete)
    by_level = {}
    for level in (2, 3):
        by_level[level] = "qualified" if all(c.e_hat[level] <= budget for c in instances.values()) else "over_budget"
    return QuantityStatus(quantity, True, by_level, ())


def choose_n_ref(statuses: Mapping[str, QuantityStatus]) -> int:
    q2 = sum(1 for s in statuses.values() if s.qualified_at(2))
    q3 = sum(1 for s in statuses.values() if s.qualified_at(3))
    return 256 if q3 > q2 else 128


def level_of(n_ref: int) -> int:
    return {128: 2, 256: 3}[n_ref]


def confirm(quantity: str, qs_status: QuantityStatus, cs_instances: Mapping[str, CrossResult], n_ref: int,
            budget: float) -> tuple[str, tuple[str, ...]]:
    """Q4: 'confirmed' only if qualified at N_ref on QS and every CS instance passes. No retries."""

    level = level_of(n_ref)
    if not qs_status.qualified_at(level):
        return "not_applicable", ()
    failures = tuple(sorted(name for name, c in cs_instances.items() if not c.complete or not (c.e_hat[level] <= budget)))
    if failures or not cs_instances:
        return "not_confirmed", failures
    return "confirmed", ()


def median_by_stratum(values: Iterable[tuple[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for stratum, value in values:
        grouped.setdefault(stratum, []).append(float(value))
    return {stratum: float(np.median(np.asarray(v, dtype=np.float64))) for stratum, v in sorted(grouped.items())}


def anomaly_bound(field_ladders: Iterable[tuple[str, LadderResult, LadderResult]], n_ref: int) -> dict[str, float]:
    """A_s = max over field instances in stratum s of (U_lref^T + U_2^P)."""

    return anomaly_bound_from_u(((s, t.U2, t.U3, p.U2) for s, t, p in field_ladders), n_ref)


def anomaly_bound_from_u(rows: Iterable[tuple[str, float, float, float]], n_ref: int) -> dict[str, float]:
    """Same bound from (stratum, U2^T, U3^T, U2^P) rows (used by Q4 on recorded QS summaries)."""

    level = level_of(n_ref)
    out: dict[str, float] = {}
    for stratum, u2_t, u3_t, u2_p in rows:
        u_t = u2_t if level == 2 else u3_t
        out[stratum] = max(out.get(stratum, 0.0), u_t + u2_p)
    return out
