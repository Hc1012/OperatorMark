"""Q1-Q4 plans (instance lists in canonical order) and their derived results.

Q1  forward ladders on QS                      -> field and compliance statuses
Q2  derivative ladders + FD checks on QS       -> m_s (frozen), derivative status
Q3  T64 designs + score ladders on QS          -> task status, N_ref, frozen QS statuses
Q4  CS confirmation at N_ref                    -> confirmed / not confirmed per quantity, A_s
"""

from __future__ import annotations

from operatormark.corpus import canonical

from .. import ladder as L
from .common import Stage, cross_summary, hx, ladder_summary, lvl, status_from_summary, status_summary, unhex

METHODS = ("P1", "TPFA")      # canonical (sorted) order


# ---------------------------------------------------------------- plans
def forward_plan(st: Stage, prefix: str):
    plan = []
    for cid, rho, _ in st.controls(prefix):
        for method in METHODS:
            for level in st.levels(method):
                plan.append((f"f/{cid}/{method}/{lvl(level)}",
                             lambda cid=cid, rho=rho, method=method, level=level: st.forward_instance(cid, rho, method, level)))
    return plan


def tangent_plan(st: Stage, prefix: str, fd_levels: dict[str, tuple[int, ...]]):
    plan = []
    dirs_by = st.directions_of(prefix)
    for cid, rho, _ in st.controls(prefix):
        dirs = dirs_by[cid]
        for method in METHODS:
            for level in st.levels(method):
                for k in range(len(dirs)):
                    plan.append((f"t/{cid}/{method}/{lvl(level)}/d{k}",
                                 lambda cid=cid, rho=rho, method=method, level=level, dirs=dirs, k=k:
                                 st.tangent_instance(cid, rho, method, level, dirs, k)))
                if level in fd_levels.get(method, ()):
                    for k in range(len(dirs)):
                        plan.append((f"x/{cid}/{method}/{lvl(level)}/d{k}",
                                     lambda cid=cid, rho=rho, method=method, level=level, dirs=dirs, k=k:
                                     st.fd_instance(cid, rho, method, level, dirs, k)))
    return plan


def task_plan(st: Stage, prefix: str):
    plan = []
    targets = st.task_targets(prefix)
    batch = len(targets)
    for cid, rho, start in targets:
        plan.append((f"o/{cid}", lambda cid=cid, rho=rho, start=start: st.optimize_instance(cid, rho, start, batch)))
        for method in METHODS:
            for level in st.levels(method):
                plan.append((f"s/{cid}/{method}/{lvl(level)}",
                             lambda cid=cid, rho=rho, start=start, method=method, level=level:
                             st.score_instance(cid, rho, start, batch, method, level)))
    return plan


def sorted_plan(plan):
    return sorted(plan, key=lambda item: item[0])


# ---------------------------------------------------------------- derived pieces
INCOMPLETE = L.CrossResult(False, False, float("inf"), {2: float("inf"), 3: float("inf")})


def gates_ok(record: dict) -> bool:
    """Every recorded gate passed.  Gate failures keep their numbers; they only block verdicts."""

    gates = record.get("gates_pass", True)
    if isinstance(gates, dict):
        gates = all(gates.values())
    return bool(gates) and record.get("tangent_gate_pass", True) is not False and record.get("forward_gates_pass", True) is not False


def series(st: Stage, prefix: str, method: str, suffix: str, kind_field: str | None = "field64"):
    """(values by level, records by level, outcome) with outcome ok | gate_failure | solver_failure:<code> | missing."""

    values, records, outcome = [], [], "ok"
    for level in st.levels(method):
        record, arrays = st.rec(f"{prefix}/{method}/{lvl(level)}{suffix}")
        if record is None:
            return None, None, "missing"
        if record.get("status") != "ok":
            return None, None, "solver_failure:" + str(record.get("status"))
        values.append(arrays.get(kind_field) if kind_field else None)
        records.append(record)
        if not gates_ok(record):
            outcome = "gate_failure"
    return values, records, outcome


def _worst(*outcomes: str) -> str:
    for o in outcomes:
        if o != "ok":
            return o
    return "ok"


def blocked_status(status: L.QuantityStatus, extra_failures: list[str]) -> L.QuantityStatus:
    """A required check failed: the quantity cannot be qualified at any level."""

    if not extra_failures:
        return status
    return L.QuantityStatus(status.quantity, False, {2: "incomplete", 3: "incomplete"},
                            tuple(list(status.failing_instances) + list(extra_failures)))


def derive_forward(st: Stage, prefix: str):
    budgets = st.budgets
    per, crosses = {}, {"field": {}, "compliance": {}}
    for cid, _, stratum in st.controls(prefix):
        tv, tr, out_t = series(st, f"f/{cid}", "TPFA", "")
        pv, pr, out_p = series(st, f"f/{cid}", "P1", "")
        outcome = _worst(out_t, out_p)
        if tv is None or pv is None:
            per[cid] = {"stratum": stratum, "status": outcome}
            for q in crosses:
                crosses[q][cid] = INCOMPLETE
            continue
        denom_f = L.norm64(tv[2])
        denom_j = abs(L.compliance(tv[2]))
        entry = {"stratum": stratum, "status": outcome, "denominators": {"field": hx(denom_f), "compliance": hx(denom_j)}}
        for quantity, values_t, values_p, eta_key, denom in (
                ("field", tv, pv, "eta_field_raw", denom_f),
                ("compliance", [L.compliance(v) for v in tv], [L.compliance(v) for v in pv], "eta_compliance_raw", denom_j)):
            et = [unhex(r[eta_key]) for r in tr]
            ep = [unhex(r[eta_key]) for r in pr]
            lt = L.ladder(quantity, *values_t, et[1], et[2], denom)
            lp = L.ladder(quantity, *values_p, ep[1], ep[2], denom)
            c = L.cross(quantity, lt, lp, values_t, denom)
            crosses[quantity][cid] = c if outcome == "ok" else INCOMPLETE      # gate failures never support a verdict
            entry[quantity] = {"tpfa": ladder_summary(lt), "p1": ladder_summary(lp), "cross": cross_summary(c)}
        per[cid] = entry
    statuses = {q: L.quantity_status(q, crosses[q], budgets[q]) for q in crosses}
    return {"instances": per, "status": {q: status_summary(s) for q, s in statuses.items()}}, crosses


def fd_coverage(st: Stage, prefix: str, fd_required: dict[str, tuple[int, ...]]) -> dict:
    """D6: every required finite-difference check must be present, numerically complete, gated and passing."""

    dirs_by = st.directions_of(prefix)
    expected = sorted(f"x/{cid}/{method}/{lvl(level)}/d{k}"
                      for cid, _, _ in st.controls(prefix) for method, levels in fd_required.items()
                      for level in levels for k in range(len(dirs_by[cid])))
    failures, rows = [], []
    for instance_id in expected:
        record, _ = st.rec(instance_id)
        if record is None:
            failures.append(f"{instance_id}:missing")
            continue
        rows.append({"instance_id": instance_id, "status": record.get("status"), "passes": record.get("passes"),
                     "slope": record.get("slope"), "stencil": record.get("stencil"),
                     "forward_gates_pass": record.get("forward_gates_pass"), "tangent_gate_pass": record.get("tangent_gate_pass")})
        if record.get("status") != "ok":
            failures.append(f"{instance_id}:{record.get('status')}")
        elif record.get("passes") is not True:
            failures.append(f"{instance_id}:slope_outside_window")
        elif not gates_ok(record):
            failures.append(f"{instance_id}:gate_failure")
    unexpected = sorted(i for i in st.results if i.startswith("x/") and i.split("/")[1].startswith(prefix) and i not in expected)
    failures += [f"{i}:unexpected" for i in unexpected]
    return {"required": {m: list(v) for m, v in sorted(fd_required.items())}, "expected": len(expected),
            "present": len(rows), "checks": rows, "failures": failures, "required_checks_pass": not failures,
            "note": None if expected else "no finite-difference checks required in this stage"}


def derive_derivative(st: Stage, prefix: str, m_s: dict | None, fd_required: dict[str, tuple[int, ...]]):
    budgets = st.budgets
    dirs_by = st.directions_of(prefix)
    controls = st.controls(prefix)
    t256 = max(st.levels("TPFA"))
    norms = {}
    for cid, _, stratum in controls:
        for k in range(len(dirs_by[cid])):
            record, arrays = st.rec(f"t/{cid}/TPFA/{lvl(t256)}/d{k}")
            if record is not None and record.get("status") == "ok":
                norms[(cid, k)] = (stratum, L.norm64(arrays["field64"]))
    frozen_now = m_s is None
    if m_s is None:
        m_s = L.median_by_stratum(norms.values())
    per, crosses, floor_bound = {}, {}, []
    for cid, _, stratum in controls:
        for k in range(len(dirs_by[cid])):
            name = f"{cid}/d{k}"
            tv, tr, out_t = series(st, f"t/{cid}", "TPFA", f"/d{k}")
            pv, pr, out_p = series(st, f"t/{cid}", "P1", f"/d{k}")
            outcome = _worst(out_t, out_p)
            if tv is None or pv is None or (cid, k) not in norms:
                per[name] = {"stratum": stratum, "status": outcome}
                crosses[name] = INCOMPLETE
                continue
            denom, bound = L.derivative_denominator(norms[(cid, k)][1], m_s.get(stratum, 0.0))
            if bound:
                floor_bound.append(name)
            et = [unhex(r["eta_raw"]) for r in tr]
            ep = [unhex(r["eta_raw"]) for r in pr]
            lt = L.ladder("derivative", *tv, et[1], et[2], denom)
            lp = L.ladder("derivative", *pv, ep[1], ep[2], denom)
            c = L.cross("derivative", lt, lp, tv, denom)
            crosses[name] = c if outcome == "ok" else INCOMPLETE
            per[name] = {"stratum": stratum, "status": outcome, "denominator": hx(denom), "floor_bound": bound,
                         "tpfa": ladder_summary(lt), "p1": ladder_summary(lp), "cross": cross_summary(c)}
    fd = fd_coverage(st, prefix, fd_required)
    ladder_status = L.quantity_status("derivative", crosses, budgets["derivative"])
    status = blocked_status(ladder_status, [] if fd["required_checks_pass"] else ["fd:" + f for f in fd["failures"]])
    out = {"m_s": {s: hx(v) for s, v in sorted(m_s.items())}, "m_s_frozen_in_this_stage": frozen_now,
           "floor_bound_instances": floor_bound, "instances": per, "ladder_status": status_summary(ladder_status),
           "status": status_summary(status), "fd": fd}
    return out, crosses, fd


def target_field_agreement(st: Stage, prefix: str) -> dict:
    """Score instances re-solve target fields; every expected comparison must exist and agree bitwise."""

    forward = {}
    if prefix == "qs":
        for record in st.priors["q1"]["instances"]:
            if record["kind"] == "forward" and record.get("status") == "ok":
                forward[(record["control_id"], record["method"], record["level"])] = record["arrays"]["field64"]["tensor_sha256"]
    else:
        for _, (record, arrays) in st.results.items():
            if record["kind"] == "forward" and record.get("status") == "ok":
                forward[(record["control_id"], record["method"], record["level"])] = canonical.tensor_sha256(arrays["field64"])
    expected, checked, mismatches, missing_scores, missing_counterparts = 0, 0, [], [], []
    for cid, _, _ in st.task_targets(prefix):
        for method in METHODS:
            for level in st.levels(method):
                expected += 1
                instance_id = f"s/{cid}/{method}/{lvl(level)}"
                record, _ = st.rec(instance_id)
                if record is None or record.get("status") != "ok":
                    missing_scores.append(instance_id)
                    continue
                key = (cid, method, level)
                if key not in forward:
                    missing_counterparts.append(instance_id)
                    continue
                checked += 1
                if forward[key] != record["target_field_sha256"]:
                    mismatches.append(instance_id)
    affected = sorted({i.split("/")[1] for i in mismatches + missing_scores + missing_counterparts})
    return {"expected": expected, "checked": checked, "mismatches": mismatches, "missing_scores": missing_scores,
            "missing_counterparts": missing_counterparts, "affected_targets": affected,
            "complete_and_consistent": expected > 0 and checked == expected and not mismatches}


def derive_task(st: Stage, prefix: str, agreement: dict):
    budgets = st.budgets
    per, crosses = {}, {}
    blocked = set(agreement["affected_targets"])
    for cid, _, _ in st.task_targets(prefix):
        tv, tr, out_t = series(st, f"s/{cid}", "TPFA", "", kind_field=None)
        pv, pr, out_p = series(st, f"s/{cid}", "P1", "", kind_field=None)
        outcome = _worst(out_t, out_p)
        if tv is None or pv is None:
            per[cid] = {"status": outcome}
            crosses[cid] = INCOMPLETE
            continue
        if cid in blocked and outcome == "ok":
            outcome = "target_field_disagreement"
        st_ = [unhex(r["score"]) for r in tr]
        sp = [unhex(r["score"]) for r in pr]
        et = [unhex(r["eta_score_raw"]) for r in tr]
        ep = [unhex(r["eta_score_raw"]) for r in pr]
        lt = L.ladder("task", *st_, et[1], et[2], 1.0)
        lp = L.ladder("task", *sp, ep[1], ep[2], 1.0)
        c = L.cross("task", lt, lp, st_, 1.0)
        crosses[cid] = c if outcome == "ok" else INCOMPLETE
        per[cid] = {"status": outcome, "scores_tpfa": [hx(x) for x in st_], "scores_p1": [hx(x) for x in sp],
                    "tpfa": ladder_summary(lt), "p1": ladder_summary(lp), "cross": cross_summary(c)}
    ladder_status = L.quantity_status("task", crosses, budgets["task"])
    status = blocked_status(ladder_status, [] if agreement["complete_and_consistent"] else ["target_field_agreement"])
    return {"instances": per, "ladder_status": status_summary(ladder_status), "status": status_summary(status),
            "target_field_agreement": agreement}, crosses


def q2_fd_required_levels(levels: dict) -> dict[str, tuple[int, ...]]:
    """D6 in Q2: every QS (control, direction) at the second TPFA level (T128) and second P1 level (P256; test: P128)."""

    return {"TPFA": (levels["TPFA"][1],), "P1": (levels["P1"][1],)}


def q2_fd_required(st: Stage) -> dict[str, tuple[int, ...]]:
    return q2_fd_required_levels({"TPFA": st.tpfa_levels, "P1": st.p1_levels})


def q4_fd_required(n_ref: int) -> dict[str, tuple[int, ...]]:
    return {"TPFA": (256,)} if n_ref == 256 else {}                     # D6: at N_ref on CS only if N_ref = 256


def confirm_all(qs_status: dict, cs_cross: dict, n_ref: int, budgets: dict, extra_failures: dict) -> dict:
    confirmation = {}
    for quantity in L.QUANTITIES:
        verdict, failures = L.confirm(quantity, qs_status[quantity], cs_cross[quantity], n_ref, budgets[quantity])
        failures = list(failures)
        if extra_failures.get(quantity) and verdict != "not_applicable":
            verdict = "not_confirmed"
            failures += list(extra_failures[quantity])
        confirmation[quantity] = {"verdict": verdict, "failing_cs_instances": failures}
    return confirmation


# ---------------------------------------------------------------- stages
def build_q1(st: Stage):
    plan = sorted_plan(forward_plan(st, "qs"))

    def derive():
        out, _ = derive_forward(st, "qs")
        return out
    return plan, derive


def build_q2(st: Stage):
    fd_required = q2_fd_required(st)
    plan = sorted_plan(tangent_plan(st, "qs", fd_required))

    def derive():
        out, _, _ = derive_derivative(st, "qs", None, fd_required)
        return out
    return plan, derive


def build_q3(st: Stage):
    plan = sorted_plan(task_plan(st, "qs"))

    def derive():
        task_out, _ = derive_task(st, "qs", target_field_agreement(st, "qs"))
        q1 = st.priors["q1"]["footer"]["derived"]
        q2 = st.priors["q2"]["footer"]["derived"]
        statuses = {
            "field": status_from_summary("field", q1["status"]["field"]),
            "compliance": status_from_summary("compliance", q1["status"]["compliance"]),
            "derivative": status_from_summary("derivative", q2["status"]),
            "task": status_from_summary("task", task_out["status"]),
        }
        n_ref = L.choose_n_ref(statuses)
        task_out.update(qs_statuses_frozen={q: status_summary(s) for q, s in statuses.items()}, n_ref=n_ref)
        return task_out
    return plan, derive


def build_q4(st: Stage):
    q3 = st.priors["q3"]["footer"]["derived"]
    n_ref = int(q3["n_ref"])
    fd_required = q4_fd_required(n_ref)
    plan = sorted_plan(forward_plan(st, "cs") + tangent_plan(st, "cs", fd_required) + task_plan(st, "cs"))

    def derive():
        budgets = st.budgets
        m_s = {s: unhex(v) for s, v in st.priors["q2"]["footer"]["derived"]["m_s"].items()}
        fwd, fwd_cross = derive_forward(st, "cs")
        der, der_cross, fd = derive_derivative(st, "cs", m_s, fd_required)
        tsk, tsk_cross = derive_task(st, "cs", target_field_agreement(st, "cs"))
        qs_status = {q: status_from_summary(q, s) for q, s in q3["qs_statuses_frozen"].items()}
        cs_cross = {"field": fwd_cross["field"], "compliance": fwd_cross["compliance"], "derivative": der_cross, "task": tsk_cross}
        extra = {"derivative": ["fd:" + f for f in fd["failures"]],
                 "task": [] if tsk["target_field_agreement"]["complete_and_consistent"] else ["target_field_agreement"]}
        confirmation = confirm_all(qs_status, cs_cross, n_ref, budgets, extra)
        q1_instances = st.priors["q1"]["footer"]["derived"]["instances"]
        rows = [(entry["stratum"], unhex(entry["field"]["tpfa"]["U2"]), unhex(entry["field"]["tpfa"]["U3"]),
                 unhex(entry["field"]["p1"]["U2"]))
                for source in (q1_instances, fwd["instances"]) for entry in source.values() if entry.get("status") == "ok"]
        anomaly = L.anomaly_bound_from_u(rows, n_ref)
        # [C] A_s is defined only if the field is complete; this code also requires every CS field ladder to be complete.
        field_complete = qs_status["field"].complete and all(c.complete for c in fwd_cross["field"].values())
        return {"n_ref": n_ref, "forward": fwd, "derivative": der, "task": tsk, "confirmation": confirmation,
                "anomaly_bound": {s: hx(v) for s, v in sorted(anomaly.items())} if field_complete else None,
                "anomaly_bound_note": None if field_complete else "field incomplete: no anomaly procedure (D10)"}
    return plan, derive


BUILDERS = {"q1": build_q1, "q2": build_q2, "q3": build_q3, "q4": build_q4}


def run_stage(config) -> dict:
    stage = Stage(config)
    try:
        plan, derive = BUILDERS[config.stage](stage)
        complete, reason = stage.execute(plan)
        return stage.finish(complete, reason, derive)
    except BaseException as exc:      # interruption, OS error, unexpected failure: never leave a stage unrecorded
        stage.abort(exc)
        raise
