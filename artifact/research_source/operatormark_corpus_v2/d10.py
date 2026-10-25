"""D10 generation integrity at N_ref (revision 3 D10; v1 §§11.1-11.3 gates).

Hard gates per sample (all recorded; a failure is kept and flagged, never retried):
  structural        control matches its stratum (v1 rule); kappa64 and views are built, not checked, here
  tpfa_native       frozen ``compute_tpfa_diagnostics(...).passes`` on T_{N_ref}
  p1_256            frozen ``compute_p1_diagnostics(...).passes`` on P1-256
  r_integral        R(N_ref -> 64) preserves the integral within 64 eps, forward and every tangent
  tangent_residual  frozen algebraic backward error of every tangent <= 1e-10
  directions        every direction finite, unit (fsum), sign-feasible at the 0.02 band, distinct

Per-sample discrepancy, recorded for every sample:
  delta = ||F64 - Pi P256|| / ||F64||, ||y|| = sqrt(sum(y*y) / 4096) (float64 sum).

Anomaly: if the procedure is active, delta > A_s (A_s by stratum) flags the
sample.  It stays in the corpus.  Anomaly ladders (full forward ladders with
eta, at most 64) and the field downgrade rule are in ``anomaly_ladders`` and
``downgrade``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from operatormark.corpus import canonical
from operatormark_numerics_v2 import views

from .config import MAX_ANOMALY_LADDERS, STRATUM_KEYS, AnomalyPolicy

TANGENT_GATE = 1.0e-10
R_INTEGRAL_TOLERANCE = views.TRANSFER_TOLERANCE      # 64 eps
NEAR_BOUND = 0.02
UNIT_NORM_TOLERANCE = 1.0e-14
GATE_NAMES = ("structural", "tpfa_native", "p1_256", "r_integral", "tangent_residual", "directions")


def closed_ladder_failure_code(code) -> bool:
    """Drop 1 codes ("tpfa_superlu_failure", "p1_frozen_error:<Type>") or a bare type name; <=105 ASCII."""
    if type(code) is not str or not code.isascii():
        return False
    prefix, colon, name = code.rpartition(":")
    ident = len(name) <= 64 and name.isidentifier()
    return ident and (not colon or (0 < len(prefix) <= 40 and prefix[0].islower() and
                                    all(c.islower() or c.isdigit() or c == "_" for c in prefix)))


def norm64(y: np.ndarray) -> float:
    return math.sqrt(float(np.sum(y * y, dtype=np.float64) / np.float64(4096)))


def compliance64(y: np.ndarray) -> float:
    return float(np.sum(y, dtype=np.float64) / np.float64(4096))


def delta_field(f64: np.ndarray, projected: np.ndarray) -> float:
    return norm64(f64 - projected) / norm64(f64)


def control_matches_stratum(rho: np.ndarray, stratum: str) -> bool:
    """v1 rule (final_scientific_verify): relaxed in [0,1) without negative zero; binary exactly 32 zeros, 32 ones."""

    control = np.ascontiguousarray(rho, dtype="<f8").reshape(8, 8)
    if not np.isfinite(control).all():
        return False
    if stratum == "relaxed":
        return not bool(np.signbit(control).any()) and bool(((control >= 0) & (control < 1)).all())
    bits = control.view("<u8")
    return bool(((bits == 0) | (bits == 0x3FF0000000000000)).all()) and int((bits == 0).sum()) == 32


def direction_facts(rho: np.ndarray, v: np.ndarray) -> dict:
    r = np.asarray(rho, dtype=np.float64).reshape(64)
    x = np.asarray(v, dtype=np.float64).reshape(64)
    finite = bool(np.isfinite(x).all())
    norm_sq = math.fsum(float(a) * float(a) for a in x) if finite else math.inf
    sign = finite and bool(np.all(x[r <= NEAR_BOUND] >= 0.0)) and bool(np.all(x[r >= 1.0 - NEAR_BOUND] <= 0.0))
    unit = finite and abs(norm_sq - 1.0) <= UNIT_NORM_TOLERANCE
    return {"finite": finite, "sign_rule": sign, "unit_norm": unit, "norm_squared": norm_sq if finite else None}


@dataclass(frozen=True)
class SampleVerdict:
    gates: dict
    delta: float | None
    anomaly_flagged: bool | None
    direction_gates: list[dict]
    direction_facts: list[dict]


def evaluate(stratum: str, rho: np.ndarray, directions: list[np.ndarray], solution, policy: AnomalyPolicy) -> SampleVerdict:
    dfacts = [direction_facts(rho, v) for v in directions]
    hashes = [canonical.tensor_sha256(np.ascontiguousarray(v, dtype=np.float64)) for v in directions]
    distinct = len(set(hashes)) == len(hashes)
    dgates = []
    for facts, t in zip(dfacts, solution.tangents):
        feasible = facts["finite"] and facts["sign_rule"] and facts["unit_norm"] and distinct
        residual = math.isfinite(t.backward_error) and 0.0 <= t.backward_error <= TANGENT_GATE
        r_ok = math.isfinite(t.r_integral_error) and t.r_integral_error <= R_INTEGRAL_TOLERANCE
        dgates.append({"feasible": feasible, "tangent_residual": residual, "r_integral": r_ok,
                       "overall": feasible and residual and r_ok})
    forward_r = math.isfinite(solution.r_integral_error) and solution.r_integral_error <= R_INTEGRAL_TOLERANCE
    gates = {
        "structural": control_matches_stratum(rho, stratum),
        "tpfa_native": bool(solution.tpfa_passes),
        "p1_256": bool(solution.p1.passes),
        "r_integral": forward_r and all(g["r_integral"] for g in dgates),
        "tangent_residual": all(g["tangent_residual"] for g in dgates),
        "directions": all(g["feasible"] for g in dgates) and len(dgates) == len(directions),
    }
    gates["overall"] = all(gates[name] for name in GATE_NAMES)
    if solution.p1.projected is None:          # verifier failed: no delta, no anomaly decision (recorded as such)
        return SampleVerdict(gates, None, None, dgates, dfacts)
    delta = delta_field(solution.f64, solution.p1.projected)
    flagged = None
    if policy.state == "active":
        flagged = bool(delta > float(policy.bounds[STRATUM_KEYS[stratum]]))
    return SampleVerdict(gates, delta, flagged, dgates, dfacts)


# ---------------------------------------------------------------------------- anomaly ladders and downgrade
def anomaly_ladder(rho: np.ndarray, n_ref: int, eps_field: float) -> dict:
    """Full D5 forward (field) ladder with eta for one anomalous sample, from the unchanged Drop 1 code."""

    from operatormark_numerics_v2 import ladder as L
    from operatormark_numerics_v2.api import NumericsError
    from operatormark_numerics_v2.p1_family import P1Family
    from operatormark_numerics_v2.tpfa_family import TPFAFamily

    fields, etas, gates = {"TPFA": [], "P1": []}, {"TPFA": [], "P1": []}, []
    try:
        for method, family, levels in (("TPFA", TPFAFamily(), (64, 128, 256)), ("P1", P1Family(), (128, 256, 512))):
            for level in levels:
                with family.open(rho, level) as session:
                    f = session.forward
                fields[method].append(f.field64)
                etas[method].append(L.norm64(f.correction64))
                gates.append(bool(f.gates_pass))
    except (NumericsError, ValueError, ArithmeticError) as exc:
        code = None
        try:           # an attribute/metaclass hook must not replace the original error
            code = getattr(exc, "code", type(exc).__name__)
        except BaseException:
            pass
        # Closed form keeps the row inside sizing._ladder_bound; anything else is
        # re-raised whole (never truncated or replaced) to the stage failure path,
        # which records type, message and traceback; a separate .code joins as a note.
        if not closed_ladder_failure_code(code):
            try:
                if code is not None and hasattr(exc, "code") and code != str(exc):
                    exc.add_note("anomaly ladder failure code outside closed form: " + repr(code))
            except BaseException:
                pass           # the note is lost; the original exception still propagates whole
            raise
        return {"valid": False, "reason": "solve_failure:" + code, "e_hat": None, "within_budget": False}
    denom = L.norm64(fields["TPFA"][2])
    lt = L.ladder("field", *fields["TPFA"], etas["TPFA"][1], etas["TPFA"][2], denom)
    lp = L.ladder("field", *fields["P1"], etas["P1"][1], etas["P1"][2], denom)
    c = L.cross("field", lt, lp, fields["TPFA"], denom)
    level = L.level_of(n_ref)
    valid = bool(c.complete and all(gates))
    e_hat = c.e_hat[level]
    reason = None if valid else ("gate_failure" if not all(gates) else "ladder_invalid_or_inconsistent")
    return {"valid": valid, "reason": reason, "tpfa_failed": list(lt.failed), "p1_failed": list(lp.failed),
            "consistent": bool(c.consistent), "e_hat": e_hat if math.isfinite(e_hat) else None,
            "within_budget": bool(valid and math.isfinite(e_hat) and e_hat <= eps_field)}


def downgrade(anomalies_by_stratum: dict[str, list[str]], ladders: dict[str, dict], policy: AnomalyPolicy) -> dict:
    """D10 field downgrade per stratum.  `ladders` maps sample_id -> anomaly_ladder() result.

    not_confirmed if more than 64 anomalies, or any anomaly ladder is invalid
    or has e_hat > eps_F.  A missing ladder leaves the stratum `incomplete`;
    nothing is assumed.

    Interpretation (open for the owner): revision 3 caps ladders at 64 in
    total but states the downgrade "for that stratum".  Implemented: when the
    total exceeds 64, every stratum with at least one anomaly is
    not_confirmed; a stratum with none is unchanged.
    """

    if policy.state != "active":
        return {stratum: {"field": "no_anomaly_procedure", "reason": "field incomplete"} for stratum in ("relaxed", "binary")}
    total = sum(len(v) for v in anomalies_by_stratum.values())
    out = {}
    for stratum in ("relaxed", "binary"):
        ids = anomalies_by_stratum.get(stratum, [])
        if total > MAX_ANOMALY_LADDERS and ids:
            out[stratum] = {"field": "not_confirmed", "reason": f"{total} anomalies > {MAX_ANOMALY_LADDERS}", "anomalies": len(ids)}
            continue
        missing = [i for i in ids if i not in ladders]
        bad = [i for i in ids if i in ladders and not ladders[i]["within_budget"]]
        if bad:
            out[stratum] = {"field": "not_confirmed", "reason": "anomaly ladder invalid or over budget", "samples": bad}
        elif missing:
            out[stratum] = {"field": "incomplete", "reason": "anomaly ladders missing", "samples": missing}
        else:
            out[stratum] = {"field": "unchanged", "anomalies": len(ids)}
    return out
