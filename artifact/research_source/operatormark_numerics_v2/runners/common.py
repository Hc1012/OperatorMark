"""Shared stage machinery for the Q1-Q4 runners.

A stage is an ordered list of instances (canonical id order).  Each instance
record depends only on its own inputs, settings, code identity and scope
(instance purity, test H4).  Caps are soft and are checked after every native
operation (solve, tangent, oracle evaluation), so the stop happens at the
first check over the limit.  A cap stop or any process-level failure yields an
*incomplete* payload plus a run record with the stop reason; there is no resume.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from operatormark.corpus import canonical

from .. import directions as D
from .. import ladder as L
from .. import lock as K
from .. import payload as P
from .. import pins, resources, scope
from .. import task as T
from ..api import P1_LEVELS, TEST_P1_LEVELS, TEST_TPFA_LEVELS, TPFA_LEVELS, NumericsError
from ..p1_family import ETA_RTOL as P1_ETA_RTOL
from ..p1_family import P1DirectOracle, P1Family
from ..tpfa_family import TPFAFamily, TPFAFiniteDifferenceOracle, task_field_and_vjp

STAGES = ("q1", "q2", "q3", "q4")
PRIORS = {"q1": (), "q2": ("q1",), "q3": ("q1", "q2"), "q4": ("q1", "q2", "q3")}
TASK_LEVEL = 64       # D8: Q3/Q4 designs use exact T64 gradients
# Revision 3 proposed caps (to be confirmed at each stage approval).
STAGE_CAPS = {"q1": {"time_cap_s": 1800.0, "mem_cap_bytes": 8 * 10**9},
              "q2": {"time_cap_s": 9000.0, "mem_cap_bytes": 8 * 10**9},
              "q3": {"time_cap_s": 2700.0, "mem_cap_bytes": 8 * 10**9},
              "q4": {"time_cap_s": 10800.0, "mem_cap_bytes": 8 * 10**9}}
TANGENT_GATE = 1.0e-10


class CapStop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def hx(value: float | None) -> str | None:
    return P.hexf(value)


def unhex(text: str | None) -> float | None:
    if text is None:
        return None
    if text in ("inf", "-inf", "nan"):
        return float(text)
    return float.fromhex(text)


def code_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root.parent)).replace("\\", "/"): pins.file_sha256(p)
            for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts}


def lvl(level: int) -> str:
    return f"{level:04d}"


@dataclass
class StageConfig:
    stage: str
    mode: str                       # "formal" | "test"
    lock_dir: Path
    out_dir: Path
    priors: dict[str, Path] = field(default_factory=dict)
    time_cap_s: float | None = None
    mem_cap_bytes: int | None = None
    budgets: dict[str, float] | None = None       # required (confirmed D3 budgets) in formal mode
    test_delay_s: float = 0.0
    only_instance: str | None = None


class CheckedOracle:
    """Wraps a finite-difference oracle so caps are checked after every evaluation."""

    def __init__(self, oracle, check):
        self._oracle, self._check = oracle, check

    def field64(self, rho, level):
        value = self._oracle.field64(rho, level)
        self._check()
        return value


class Stage:
    def __init__(self, config: StageConfig):
        if config.stage not in STAGES:
            raise ValueError(config.stage)
        self.cfg = config
        self.test = config.mode == "test"
        if not self.test and (config.test_delay_s or config.only_instance):
            raise ValueError("test hooks are refused outside test mode")
        unknown = set(config.priors) - set(PRIORS[config.stage])
        if unknown:
            raise ValueError(f"unexpected prior payloads for {config.stage}: {sorted(unknown)}")
        caps = STAGE_CAPS[config.stage]
        self.time_cap_s = caps["time_cap_s"] if config.time_cap_s is None else float(config.time_cap_s)
        self.mem_cap_bytes = caps["mem_cap_bytes"] if config.mem_cap_bytes is None else int(config.mem_cap_bytes)
        if config.budgets is None:
            if not self.test:
                raise ValueError("formal stages require the confirmed D3 budgets")
            self.budgets, self.budget_status = dict(L.PROPOSED_BUDGETS), "proposed (test mode)"
        else:
            if set(config.budgets) != set(L.QUANTITIES):
                raise ValueError(f"budgets must cover exactly {L.QUANTITIES}")
            self.budgets, self.budget_status = {k: float(v) for k, v in config.budgets.items()}, "confirmed"
        self.frozen_identity = pins.verify_frozen_sources()
        pins.verify_loaded_modules()
        # Confirmation holdout: Q1-Q3 load the QS partition only; the sealed CS partition is
        # opened by Q4 after the Q3 reference/status bindings have been validated (below).
        self.locked = K.load_qs(config.lock_dir, allow_test=self.test)
        self.tpfa_levels = TEST_TPFA_LEVELS if self.test else TPFA_LEVELS
        self.p1_levels = TEST_P1_LEVELS if self.test else P1_LEVELS
        self.families = {"TPFA": TPFAFamily(self.tpfa_levels), "P1": P1Family(self.p1_levels, allow_test_levels=self.test)}
        self.oracles = {"TPFA": CheckedOracle(TPFAFiniteDifferenceOracle(), self.check_caps),
                        "P1": CheckedOracle(P1DirectOracle(allow_test_levels=self.test), self.check_caps)}
        self.scope = scope.determinism_scope()
        self.code = code_identity()
        self.priors = {}
        for name in PRIORS[config.stage]:
            if name not in config.priors:
                raise ValueError(f"{config.stage} requires the {name} payload")
            self.priors[name] = P.read_payload(config.priors[name], name)
        self._check_priors()
        if config.stage == "q4":
            self._check_q3_reference()
            self.locked = K.open_cs(self.locked)
        self.results: dict[str, tuple[dict, dict]] = {}
        self.cache: dict[str, Any] = {}
        header = {
            "mode": config.mode,
            "code_identity": self.code,
            "frozen_identity": self.frozen_identity,
            "determinism_scope": self.scope,
            "determinism_scope_sha256": scope.scope_sha256(self.scope),
            "lock_commitment_sha256": self.locked.commitment_sha256,
            "holdout": {"cs_opened": self.locked.cs_opened},
            "prior_payload_sha256": {n: d["footer"]["payload_sha256"] for n, d in sorted(self.priors.items())},
            "settings": self.settings(),
        }
        self.writer = P.PayloadWriter(config.out_dir, config.stage, header)
        self.run = P.RunRecord(config.out_dir, config.stage,
                               {"time_cap_s": self.time_cap_s, "mem_cap_bytes": self.mem_cap_bytes,
                                "soft_caps": "checked after every native operation"})
        self.t0 = time.monotonic()

    # ------------------------------------------------------------------ prior binding (no mixing of runs)
    def _check_priors(self) -> None:
        settings = self.settings()
        scope_sha = scope.scope_sha256(self.scope)
        for name, data in self.priors.items():
            header, footer = data["header"], data["footer"]
            if footer["completion"] != "complete":
                raise ValueError(f"{name} payload is incomplete")
            if header["lock_commitment_sha256"] != self.locked.commitment_sha256:
                raise ValueError(f"{name} payload was produced from different lock material")
            for bound_name, bound_sha in header["prior_payload_sha256"].items():
                if bound_name not in self.priors or self.priors[bound_name]["footer"]["payload_sha256"] != bound_sha:
                    raise ValueError(f"{name} was produced from a different {bound_name} payload than the one supplied")
            same = {"code_identity": header["code_identity"] == self.code,
                    "frozen_identity": header["frozen_identity"] == self.frozen_identity,
                    "determinism_scope_sha256": header["determinism_scope_sha256"] == scope_sha,
                    "settings": header["settings"] == settings}
            if not all(same.values()):
                raise ValueError(f"{name} payload differs from this stage in {[k for k, v in same.items() if not v]}")

    def _check_q3_reference(self) -> None:
        """Q4 precondition: a complete Q3 with a frozen N_ref and frozen QS statuses for every quantity."""

        derived = self.priors["q3"]["footer"]["derived"]
        if derived is None or derived.get("n_ref") not in (128, 256):
            raise ValueError("Q3 payload has no frozen N_ref")
        statuses = derived.get("qs_statuses_frozen") or {}
        if set(statuses) != set(L.QUANTITIES):
            raise ValueError("Q3 payload does not freeze a QS status for every quantity")
        bound = self.priors["q3"]["header"]["prior_payload_sha256"]
        for name in ("q1", "q2"):
            if bound.get(name) != self.priors[name]["footer"]["payload_sha256"]:
                raise ValueError(f"Q3 was not produced from the supplied {name} payload")

    # ------------------------------------------------------------------ settings
    def settings(self) -> dict:
        return {
            "tpfa_levels": list(self.tpfa_levels),
            "p1_levels": list(self.p1_levels),
            "budgets": {k: hx(v) for k, v in sorted(self.budgets.items())},
            "budget_status": self.budget_status,
            "task": {"scale": hx(T.SCALE), "steps": T.STEPS, "lr": hx(T.LEARNING_RATE),
                     "betas": [hx(b) for b in T.BETAS], "eps": hx(T.EPS), "box": [hx(b) for b in T.BOX],
                     "task_level": TASK_LEVEL},
            "fd_steps": [hx(h) for h in D.FD_STEPS],
            "fd_slope_window": [hx(x) for x in D.SLOPE_WINDOW],
            "near_bound": hx(D.NEAR_BOUND),
            "p1_eta_rtol": hx(P1_ETA_RTOL),
            "tangent_gate": hx(TANGENT_GATE),
            "v_checks": {"order": [hx(x) for x in L.V2_ORDER], "floor": hx(L.V3_FLOOR),
                         "eta_factor": hx(L.V3_ETA_FACTOR), "cos": hx(L.V4_COS)},
        }

    # ------------------------------------------------------------------ caps
    def check_caps(self) -> None:
        if time.monotonic() - self.t0 > self.time_cap_s:
            raise CapStop("time_cap")
        peak = resources.peak_rss_bytes()
        if peak is not None and peak > self.mem_cap_bytes:
            raise CapStop("memory_cap")

    # ------------------------------------------------------------------ material
    def _index(self, prefix: str) -> list:
        if prefix == "qs":
            return self.locked.manifest["qs_index"]
        if prefix == "cs" and self.locked.cs_opened:
            return self.locked.cs["cs_index"]
        raise ValueError("the CS holdout is not open in this stage")

    def controls(self, prefix: str) -> list[tuple[str, np.ndarray, str]]:
        index = self._index(prefix)
        arrays = self.locked.arrays[f"{prefix}_controls"]
        return [(row["id"], arrays[i], row["stratum"]) for i, row in enumerate(index)]

    def directions_of(self, prefix: str) -> dict[str, list[np.ndarray]]:
        index = self._index(prefix)
        arrays = self.locked.arrays[f"{prefix}_directions"]
        return {row["id"]: [arrays[i, k] for k in range(arrays.shape[1])] for i, row in enumerate(index)}

    def task_targets(self, prefix: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
        index = self._index(prefix)
        controls = self.locked.arrays[f"{prefix}_controls"]
        starts = self.locked.arrays[f"{prefix}_starts"]
        random_rows = [(i, row) for i, row in enumerate(index) if row["kind"] == "random"]
        if len(random_rows) != starts.shape[0]:
            raise ValueError("starts do not match the random targets")
        return [(row["id"], controls[i], starts[k]) for k, (i, row) in enumerate(random_rows)]

    def levels(self, method: str) -> tuple[int, ...]:
        return self.tpfa_levels if method == "TPFA" else self.p1_levels

    # ------------------------------------------------------------------ instance kinds
    def forward_instance(self, cid: str, rho: np.ndarray, method: str, level: int):
        base = {"kind": "forward", "control_id": cid, "control_sha256": canonical.tensor_sha256(rho),
                "method": method, "level": level}
        try:
            with self.families[method].open(rho, level) as session:
                f = session.forward
        except NumericsError as exc:
            return dict(base, status=exc.code), {"control": np.ascontiguousarray(rho)}
        record = dict(base, status="ok", native_sha256=f.native_sha256, coefficient_sha256=f.coefficient_sha256,
                      gates_pass=f.gates_pass, diagnostics={k: hx(v) for k, v in sorted(f.diagnostics.items())},
                      transfer_integral_error=hx(f.transfer_integral_error), iterations=f.iterations,
                      eta_iterations=f.eta_iterations, compliance=hx(L.compliance(f.field64)),
                      eta_field_raw=hx(L.norm64(f.correction64)), eta_compliance_raw=hx(abs(L.compliance(f.correction64))))
        return record, {"field64": f.field64, "control": np.ascontiguousarray(rho)}

    def tangent_group(self, cid: str, rho: np.ndarray, method: str, level: int, dirs: list[np.ndarray]):
        """One session, every direction's tangent; errors are recorded per direction (cached by group)."""

        key = f"tg/{cid}/{method}/{lvl(level)}"
        if key in self.cache:
            return self.cache[key]
        out: dict[Any, Any] = {}
        try:
            session = self.families[method].open(rho, level, compute_eta=False)
        except NumericsError as exc:
            out = {k: exc.code for k in range(len(dirs))}
            out["forward_gates_pass"] = None
        else:
            with session:
                out["forward_gates_pass"] = bool(session.forward.gates_pass)
                self.check_caps()
                for k, v in enumerate(dirs):
                    try:
                        out[k] = session.tangent(v)
                    except NumericsError as exc:
                        out[k] = exc.code
                    self.check_caps()
        self.cache[key] = out
        return out

    def tangent_instance(self, cid: str, rho: np.ndarray, method: str, level: int, dirs: list[np.ndarray], k: int):
        group = self.tangent_group(cid, rho, method, level, dirs)
        base = {"kind": "tangent", "control_id": cid, "direction_id": f"d{k}", "direction_sha256": canonical.tensor_sha256(dirs[k]),
                "method": method, "level": level}
        arrays = {"direction": np.ascontiguousarray(dirs[k])}
        if isinstance(group[k], str):
            return dict(base, status=group[k]), arrays
        t = group[k]
        # Negative results are preserved: gate failures keep their numbers and are judged in the derivation.
        record = dict(base, status="ok", native_sha256=t.native_sha256, tangent_backward_error=hx(t.tangent_backward_error),
                      tangent_gate_pass=bool(t.tangent_backward_error <= TANGENT_GATE),
                      forward_gates_pass=group["forward_gates_pass"], iterations=t.iterations,
                      eta_iterations=t.eta_iterations, eta_raw=hx(L.norm64(t.correction64)))
        arrays["field64"] = t.field64
        return record, arrays

    def fd_instance(self, cid: str, rho: np.ndarray, method: str, level: int, dirs: list[np.ndarray], k: int):
        group = self.tangent_group(cid, rho, method, level, dirs)
        base = {"kind": "fd", "control_id": cid, "direction_id": f"d{k}", "method": method, "level": level,
                "oracle": "frozen TPFA direct solve" if method == "TPFA" else "P1 family direct sparse LU"}
        if isinstance(group[k], str):
            return dict(base, status=group[k]), {}
        tangent64 = group[k].field64
        try:
            check, estimates = D.fd_check(self.oracles[method], rho, dirs[k], level, tangent64)
        except NumericsError as exc:
            return dict(base, status=exc.code), {}
        record = dict(base, status="ok", stencil=check.stencil, steps=[hx(h) for h in check.steps],
                      raw_errors=[hx(e) for e in check.raw_errors], slope=hx(check.slope), passes=check.passes,
                      forward_gates_pass=group["forward_gates_pass"],
                      tangent_gate_pass=bool(group[k].tangent_backward_error <= TANGENT_GATE),
                      tangent_field_sha256=canonical.tensor_sha256(tangent64))
        arrays = {f"fd_h{i}": estimates[h] for i, h in enumerate(check.steps)}
        return record, arrays

    def design_of(self, cid: str, target_rho: np.ndarray, start: np.ndarray, batch: int):
        key = f"design/{cid}"
        if key not in self.cache:
            fv = task_field_and_vjp(TASK_LEVEL)

            def checked(x):
                result = fv(x)
                self.check_caps()
                return result

            target_field = fv(target_rho)[0]
            self.cache[key] = T.optimize(checked, start[None], target_field[None], batch_size=batch)
        return self.cache[key]

    def optimize_instance(self, cid: str, target_rho: np.ndarray, start: np.ndarray, batch: int):
        base = {"kind": "optimize", "target_control_id": cid, "target_sha256": canonical.tensor_sha256(target_rho),
                "start_sha256": canonical.tensor_sha256(start), "nominal_batch": batch, "task_level": TASK_LEVEL}
        try:
            result = self.design_of(cid, target_rho, start, batch)
        except NumericsError as exc:
            return dict(base, status=exc.code), {}
        record = dict(base, status="ok", design_sha256=canonical.tensor_sha256(result.designs[0]),
                      steps=len(result.loss_trace), loss_trace=[hx(x) for x in result.loss_trace],
                      final_loss=hx(result.final_loss))
        return record, {"design": result.designs[0], "start": np.ascontiguousarray(start),
                        "target_control": np.ascontiguousarray(target_rho)}

    def score_instance(self, cid: str, target_rho: np.ndarray, start: np.ndarray, batch: int, method: str, level: int):
        base = {"kind": "score", "target_control_id": cid, "method": method, "level": level}
        try:
            design = self.design_of(cid, target_rho, start, batch).designs[0]
            base["design_sha256"] = canonical.tensor_sha256(design)
            with self.families[method].open(design, level) as sd:
                f_design = sd.forward
            self.check_caps()
            with self.families[method].open(target_rho, level) as st:
                f_target = st.forward
        except NumericsError as exc:
            return dict(base, status=exc.code), {}
        gates = {"design": f_design.gates_pass, "target": f_target.gates_pass}
        s0 = T.score(f_design.field64, f_target.field64)
        s1 = T.score(f_design.field64 + f_design.correction64, f_target.field64 + f_target.correction64)
        record = dict(base, status="ok", gates_pass=gates,
                      score=hx(s0), eta_score_raw=hx(abs(s1 - s0)),
                      design_field_sha256=canonical.tensor_sha256(f_design.field64),
                      target_field_sha256=canonical.tensor_sha256(f_target.field64))
        return record, {"design_field64": f_design.field64}

    # ------------------------------------------------------------------ execution
    def execute(self, plan: list[tuple[str, Callable[[], tuple[dict, dict]]]]) -> tuple[bool, str]:
        ids = [i for i, _ in plan]
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise ValueError("plan must be in strictly increasing canonical id order")
        selected = [(i, fn) for i, fn in plan if self.cfg.only_instance in (None, i)]
        if self.cfg.only_instance is not None and not selected:
            raise ValueError("selected instance not in plan")
        try:
            for instance_id, fn in selected:
                self.check_caps()
                started = time.perf_counter()
                record, arrays = fn()
                seconds = time.perf_counter() - started
                self.writer.add_instance(instance_id, record, arrays)
                self.results[instance_id] = (record, arrays)
                self.run.add_region(record["kind"], seconds)
                self.run.log(instance_id=instance_id, seconds=seconds, peak_rss_bytes=resources.peak_rss_bytes())
                if self.cfg.test_delay_s:
                    time.sleep(self.cfg.test_delay_s)
                self.check_caps()
        except CapStop as stop:
            return False, stop.reason
        if self.cfg.only_instance is not None:
            return False, "single_instance_selection"
        return True, "completed"

    def finish(self, complete: bool, reason: str, derive: Callable[[], dict]) -> dict:
        derived = derive() if complete else None
        footer = self.writer.finalize(complete=complete, derived=derived)
        self.run.finish(stop_reason=reason, payload_sha256=footer["payload_sha256"],
                        peak_rss_bytes=resources.peak_rss_bytes())
        return footer

    def abort(self, exc: BaseException) -> None:
        """Process-level failure: finalize an incomplete payload and a run record, then let the error propagate."""

        try:
            footer = None if self.writer.finalized else self.writer.finalize(complete=False, derived=None)
        except BaseException:      # the original error matters more than a secondary write failure
            footer = None
        try:
            self.run.finish(stop_reason=f"error:{type(exc).__name__}",
                            payload_sha256=None if footer is None else footer["payload_sha256"],
                            peak_rss_bytes=resources.peak_rss_bytes())
        except BaseException:
            pass

    # ------------------------------------------------------------------ derived helpers
    def rec(self, instance_id: str):
        return self.results.get(instance_id, (None, None))


def ladder_summary(result: L.LadderResult) -> dict:
    return {"valid": result.valid, "failed": list(result.failed), "d1": hx(result.d1), "d2": hx(result.d2),
            "r": hx(result.r), "p": hx(result.p), "alignment": hx(result.alignment), "eta": hx(result.eta),
            "e2": hx(result.e2), "e3": hx(result.e3), "U2": hx(result.U2), "U3": hx(result.U3)}


def cross_summary(result: L.CrossResult) -> dict:
    return {"complete": result.complete, "consistent": result.consistent, "limit_gap": hx(result.limit_gap),
            "e_hat": {str(k): hx(v) for k, v in sorted(result.e_hat.items())}}


def status_summary(status: L.QuantityStatus) -> dict:
    return {"complete": status.complete, "status_by_level": {str(k): v for k, v in sorted(status.status_by_level.items())},
            "failing_instances": list(status.failing_instances)}


def status_from_summary(quantity: str, summary: dict) -> L.QuantityStatus:
    return L.QuantityStatus(quantity, summary["complete"], {int(k): v for k, v in summary["status_by_level"].items()},
                            tuple(summary["failing_instances"]))


def method_series(stage: Stage, prefix: str, method: str, suffix: str, kind_field: str | None = "field64"):
    """(values by level, records by level, ok) for one quantity series."""

    values, records = [], []
    for level in stage.levels(method):
        record, arrays = stage.rec(f"{prefix}/{method}/{lvl(level)}{suffix}")
        if record is None or record.get("status") != "ok":
            return None, None, False
        values.append(arrays.get(kind_field) if kind_field else None)
        records.append(record)
    return values, records, True
