"""Development generator and v2 serializer.

``generate(config, controls, directions, out_dir)`` writes a corpus_v2 tree
(layout in ``records``).  Behaviour that matters:

* N_ref and the retention option come only from the explicit config.
* Production refuses to start without its confirmed inputs (``config``).
* Each shard is computed in memory, then written into ``shard_NNNN.partial/``
  with exclusive, read-back-verified writes (frozen ``canonical`` writers), and
  published by one directory rename after its manifest is written.  A shard
  directory without the ``.partial`` suffix is therefore complete as written.
* Gate failures are measurements: the sample is stored with its gates false.
* A solve that yields no arrays is a retained failure record in
  ``failures.jsonl``; the shard holding it is not written (no array can stand
  in for it), the other samples of that shard are still computed so every
  failure is recorded, and generation continues with the next shard unless
  ``on_failure="stop"``.  Nothing is ever resampled or retried.
* Anomalies (delta > A_s) stay in place, flagged.
* ``generation_summary.json`` is always written (also on interruption or a
  cap), with ``completion: incomplete`` unless every selected shard was
  written.  It is a claim; ``readback.validate_corpus`` re-derives it.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from operatormark.corpus import canonical
from operatormark_numerics_v2 import scope as drop1_scope
from operatormark_numerics_v2 import views
from operatormark_numerics_v2.resources import peak_rss_bytes

from . import d10, pins, records as R
from . import layout as LY
from .config import STRATUM_KEYS, GenerationConfig, ProductionRefused
from .native import NativeSolver, SampleFailure, p1_settings_record, tpfa_settings


def d10_settings() -> dict:
    return {"tangent_gate": R.hx(d10.TANGENT_GATE), "r_integral_tolerance": R.hx(d10.R_INTEGRAL_TOLERANCE),
            "near_bound": R.hx(d10.NEAR_BOUND), "unit_norm_tolerance": R.hx(d10.UNIT_NORM_TOLERANCE),
            "tpfa_gate": "frozen compute_tpfa_diagnostics(...).passes", "p1_gate": "frozen compute_p1_diagnostics(...).passes",
            "delta": "sqrt(sum((F64-PiP256)^2)/4096)/sqrt(sum(F64^2)/4096), float64 sums",
            "anomaly": "delta > A_s(stratum); retained and flagged; ladders for at most 64"}


def macro_coefficient(rho: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.float64(9.0) * np.ascontiguousarray(rho, dtype="<f8") + np.float64(3.0))


def kappa64_from_rho(rho: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.repeat(np.repeat(macro_coefficient(rho), 8, axis=0), 8, axis=1))


def _append_line(path: Path, record: dict) -> None:
    with path.open("ab") as stream:
        stream.write(canonical.canonical_json_bytes(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


class _Stop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# Storage-bound domain restrictions, enforced before the value is serialized.
# An exception class name enters the summary/run stop_reason only as a bounded
# ASCII identifier; any other name maps to one fixed token that no identifier can
# spell. Nothing is discarded: the full name stays in the re-raised exception and
# in run/unexpected_failure.json, which belong to the exception domain.
EXCEPTION_TYPE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
UNREPRESENTABLE_EXCEPTION_TYPE = "exception:?type_name_outside_closed_form"
MAX_TIME_CAP_S = 10**9


def exception_stop_reason(exc: BaseException) -> str:
    try:
        name = type(exc).__name__
    except BaseException:       # a metaclass hook must not replace the in-flight primary
        return UNREPRESENTABLE_EXCEPTION_TYPE
    if type(name) is str and EXCEPTION_TYPE_NAME.fullmatch(name):
        return "exception:" + name
    return UNREPRESENTABLE_EXCEPTION_TYPE


def _class_name(obj) -> str:
    """Class qualname without metaclass descriptors or user hooks."""
    cls = type(obj)
    try:
        name = type.__getattribute__(cls, "__qualname__") if type(cls) is type else None
    except BaseException:
        name = None
    return name if type(name) is str else "?unavailable"


def _note(primary, text) -> None:
    """add_note that can never replace the in-flight primary (a hostile hook only loses the note)."""
    try:
        primary.add_note(text)
    except BaseException:
        pass


def _describe(exc) -> str:
    """repr() for an evidence note; a raising __repr__ must not replace the primary."""
    try:
        return repr(exc)
    except BaseException:
        return "<" + _class_name(exc) + "; repr unavailable>"


def check_run_parameters(time_cap_s, mem_cap_bytes) -> None:
    """Closed run-record scalars; refused before any output exists."""

    # Exact builtin types: a subclass could override comparison or repr.
    if time_cap_s is not None and (type(time_cap_s) not in (int, float) or not math.isfinite(time_cap_s)
                                   or not 0 <= time_cap_s <= MAX_TIME_CAP_S):
        raise ValueError("time_cap_s must be None or a finite number in [0, 1e9]")
    if mem_cap_bytes is not None and (type(mem_cap_bytes) is not int or not 0 <= mem_cap_bytes < 2**63):
        raise ValueError("mem_cap_bytes must be None or an int in [0, 2**63)")


def configuration_record(cfg, controls, directions, solver, on_failure, *, identity: dict, scope_sha256: str) -> dict:
    """The only configuration builder; storage templates call it with placeholder hashes."""

    return R.configuration(cfg, controls_binding=controls.binding(), directions_binding=directions.binding(),
                           solver_kind=solver.kind, identity=identity, scope_sha256=scope_sha256,
                           d10_settings=d10_settings(), failure_policy=on_failure)


def run_record(*, started, finished, wall_seconds, peak_rss, time_cap_s, mem_cap_bytes, timings, stop_reason) -> dict:
    return {"started_utc": started, "finished_utc": finished, "wall_seconds": wall_seconds,
            "peak_rss_bytes": peak_rss, "time_cap_s": time_cap_s, "mem_cap_bytes": mem_cap_bytes,
            "shard_timings": timings, "stop_reason": stop_reason,
            "note": "variable run record; not part of the deterministic corpus bytes and never validated"}


class _Sink:
    """Where retained evidence goes the moment it exists (survives a cap or an exception later in the shard)."""

    def __init__(self, out: Path, cfg_sha: str):
        self.failures = out / R.FAILURES_FILE
        self.measurements = out / R.MEASUREMENTS_FILE
        self.cfg_sha = cfg_sha
        self.failed_shards: set[tuple[str, int]] = set()
        self.failure_count = 0
        self.measurement_ids: list[str] = []
        self.unwritten_anomalies: dict[str, list[str]] = {"relaxed": [], "binary": []}
        self.pending = []
        self.persistence_failure = None

    def failure(self, split, shard, record):
        _append_line(self.failures, record)
        self.failed_shards.add((split, shard))
        self.failure_count += 1

    def measurement(self, rho, dirs, row, drows, stratum_key, flagged, reason="shard_failed"):
        if self.persistence_failure is not None:
            raise RuntimeError('prior measurement persistence failed; no blind append retry') from self.persistence_failure
        try:
            _append_line(self.measurements, R.measurement_record(configuration_sha256=self.cfg_sha, rho=rho, directions=dirs,
                                                                 sample_record=row, direction_records=drows, reason=reason))
        except BaseException as exc:
            self.persistence_failure = exc
            raise
        self.measurement_ids.append(row["sample_id"])
        if flagged:
            self.unwritten_anomalies[stratum_key].append(row["sample_id"])

    def flush_pending(self, reason):
        # Remove only after durable append succeeds. Never regenerate a sample.
        while self.pending:
            self.measurement(*self.pending[0], reason=reason)
            self.pending.pop(0)


def _sample_records(cfg, split, local, offset, rho, dirs, solution, verdict, arrays, cfg_sha):
    layout, n, option = cfg.layout, cfg.n_ref, cfg.retention.option
    d = LY.DIRECTIONS_PER_SAMPLE[split]
    ident = layout.sample_identity(split, local)
    stored_native = option == "S"
    kappa64 = kappa64_from_rho(rho)
    items = {"rho": rho, "kappa64": kappa64, "u_f64": solution.f64, "kappa32_areaavg": views.view32(kappa64),
             "u32_areaavg": views.view32(solution.f64), "kappa16_areaavg": views.view16(kappa64),
             "u16_areaavg": views.view16(solution.f64)}
    if stored_native:
        items[LY.native_forward_family(n)] = solution.u_native
    hashes = {}
    for name, value in items.items():
        arrays[name][offset, 0] = value
        hashes[name] = canonical.tensor_sha256(arrays[name][offset])
    drows = []
    for k, (v, t) in enumerate(zip(dirs, solution.tangents)):
        pos = offset * d + k
        ditems = {"dir": v, "du_f64": t.du64, "du32_areaavg": views.view32(t.du64), "du16_areaavg": views.view16(t.du64)}
        if stored_native:
            ditems[LY.native_tangent_family(n)] = t.du_native
        dhashes = {}
        for name, value in ditems.items():
            arrays[name][pos, 0] = value
            dhashes[name] = canonical.tensor_sha256(arrays[name][pos])
        drows.append(R.direction_record(
            configuration_sha256=cfg_sha, identity=layout.direction_identity(split, local, k),
            control_sha256=canonical.tensor_sha256(rho), direction_sha256=canonical.tensor_sha256(v),
            tensor_sha256=dhashes, du_native_sha256=t.du_native_sha256, backward_error=t.backward_error,
            r_integral_error=t.r_integral_error, facts=verdict.direction_facts[k], gates=verdict.direction_gates[k]))
    p1 = solution.p1
    stratum_key = STRATUM_KEYS[ident["stratum"]]
    bound = None if cfg.anomaly.state != "active" else cfg.anomaly.bounds[stratum_key]
    has_p1 = p1.projected is not None
    row = R.sample_record(
        configuration_sha256=cfg_sha, identity=ident, control_sha256=canonical.tensor_sha256(rho),
        rho_storage_sha256=hashes["rho"], macro_sha256=canonical.tensor_sha256(macro_coefficient(rho)),
        tensor_sha256=hashes, direction_records=drows,
        native={"n_ref": n, "retention": option, "settings": tpfa_settings(n),
                "coefficient_tensor_sha256": solution.kappa_native_sha256,
                "solution_tensor_sha256": solution.u_native_sha256,
                "diagnostics": {k: R.hx(v) for k, v in sorted(solution.tpfa_diagnostics.items())},
                "passes": solution.tpfa_passes, "r_integral_error": R.hx(solution.r_integral_error)},
        p1={"settings": p1_settings_record(), "outcome": "completed" if has_p1 else "failed:" + str(p1.failure),
            "nodal_tensor_sha256": p1.nodal_sha256, "projected_tensor_sha256": p1.projected_sha256,
            "diagnostics": None if p1.diagnostics is None else {k: R.hx(v) for k, v in sorted(p1.diagnostics.items())},
            "iterations": p1.iterations, "passes": p1.passes},
        metrics={"delta_field": R.hx(verdict.delta), "norm_f64": R.hx(d10.norm64(solution.f64)),
                 "norm_projected_p256": R.hx(d10.norm64(p1.projected)) if has_p1 else None,
                 "compliance_f64": R.hx(d10.compliance64(solution.f64)),
                 "compliance_projected_p256": R.hx(d10.compliance64(p1.projected)) if has_p1 else None},
        anomaly={"procedure": cfg.anomaly.state, "bound": R.hx(bound),
                 "flagged": verdict.anomaly_flagged if has_p1 else None,
                 "decision": ("not_applicable" if cfg.anomaly.state != "active" else
                              "decided" if has_p1 else "undecidable_p1_failed")},
        gates=verdict.gates)
    return ident, stratum_key, row, drows


def _build_shard(cfg: GenerationConfig, split: str, shard: int, controls, directions, solver, cfg_sha: str, check,
                 sink: _Sink):
    layout = cfg.layout
    admitted = None
    if cfg.mode in ('production','scientific'):
        admitted = {(s, i): h for s, i, h in controls.admission_expectations.controls}
    d = LY.DIRECTIONS_PER_SAMPLE[split]
    families = layout.shard_families(split, cfg.n_ref, cfg.retention.option)
    arrays = {name: np.empty(shape, dtype="<f8") for name, _, shape in families}
    sample_rows, direction_rows, verdicts = [], [], []
    if sink.pending:
        raise RuntimeError('previous shard measurements not disposed')
    failed = False
    def retain_development_prefix(count):
        if cfg.mode != 'real_development' or count == 0:
            return
        target = sink.measurements.parent / 'run' / f'partial_{split}_{shard:04d}'
        target.mkdir(exist_ok=False)
        for name, kind, _ in families:
            stop = count * d if 'direction' in kind else count
            canonical.write_npy_v2_exclusive(target / LY.file_name(name), arrays[name][:stop].copy())
    for offset, local in enumerate(layout.shard_samples(split, shard)):
        try:
            check()
        except _Stop:
            retain_development_prefix(offset)
            if not failed:                # generation stops here: keep what this shard already measured
                sink.flush_pending("generation_stopped")
            raise
        ident = layout.sample_identity(split, local)
        try:
            try:
                rho = np.ascontiguousarray(controls.control(split, local), dtype="<f8")
            except KeyError:
                raise SampleFailure("controls", "control_unavailable") from None
            if rho.shape != (8, 8):
                raise SampleFailure("controls", "control_shape")
            if admitted is not None and canonical.tensor_sha256(rho) != admitted.get((split, local)):
                raise ProductionRefused('acquired control differs from externally admitted State 4 content')
            try:
                dirs = [np.ascontiguousarray(directions.direction(split, local, k, rho), dtype="<f8") for k in range(d)]
            except KeyError:
                raise SampleFailure("directions", "direction_unavailable") from None
            except ValueError:
                raise SampleFailure("directions", "direction_invalid") from None
            if any(v.shape != (8, 8) for v in dirs):
                raise SampleFailure("directions", "direction_shape")
            solution = solver.solve(rho, dirs)
        except SampleFailure as failure:
            sink.failure(split, shard, R.failure_record(configuration_sha256=cfg_sha, identity=ident, stage=failure.stage,
                                                        code=failure.code, partial=failure.partial))
            if not failed:                # this shard will not be written: keep what was already measured in it
                failed = True
                sink.flush_pending("shard_failed")
            if cfg.mode == 'real_development':
                retain_development_prefix(offset)
                raise _Stop('sample_failure')
            continue
        verdict = d10.evaluate(ident["stratum"], rho, dirs, solution, cfg.anomaly)
        ident, stratum_key, row, drows = _sample_records(cfg, split, local, offset, rho, dirs, solution, verdict, arrays,
                                                         cfg_sha)
        item = (rho, dirs, row, drows, stratum_key, verdict.anomaly_flagged)
        if failed:
            sink.measurement(*item)
            continue
        sink.pending.append(item)
        if cfg.mode == 'real_development' and not all(verdict.gates.values()):
            retain_development_prefix(offset + 1)
            sink.flush_pending('hard_gate_failure')
            raise _Stop('hard_gate_failure')
        sample_rows.append(row)
        direction_rows.extend(drows)
        verdicts.append((ident, verdict))
    return arrays, families, sample_rows, direction_rows, failed, verdicts


def _write_shard(out: Path, cfg: GenerationConfig, split: str, shard: int, cfg_sha: str, arrays, families,
                 sample_rows, direction_rows) -> str:
    rel = cfg.layout.shard_dir(split, shard)
    final = out / rel
    partial = final.with_name(final.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    members = []
    for name, _, shape in families:
        data = canonical.write_npy_v2_exclusive(partial / LY.file_name(name), arrays[name])
        members.append(R.npy_member(f"{rel}/{LY.file_name(name)}", name, data, shape, canonical.tensor_sha256(arrays[name])))
    for filename, schema, rows in ((R.VERIFICATION_FILE, R.SAMPLE_SCHEMA, sample_rows),
                                   (R.DIRECTIONS_FILE, R.DIRECTION_SCHEMA, direction_rows)):
        data = canonical.write_canonical_jsonl_exclusive(partial / filename, rows)
        members.append(R.jsonl_member(f"{rel}/{filename}", schema, data, rows))
    manifest = R.shard_manifest(configuration_sha256=cfg_sha, layout=cfg.layout, split=split, shard=shard, members=members)
    data = canonical.write_canonical_json_exclusive(partial / R.SHARD_MANIFEST_FILE, manifest)
    if final.exists():
        raise FileExistsError(final)
    os.rename(partial, final)
    return canonical.sha256_bytes(data)


def generate(cfg: GenerationConfig, controls, directions, out_dir, *, solver=None, time_cap_s: float | None = None,
             mem_cap_bytes: int | None = None, on_failure: str | None = None,
             max_configuration_bytes: int | None = None) -> dict:
    identity = pins.verify_dependencies()
    solver = solver if solver is not None else NativeSolver(cfg.n_ref)
    if solver.n_ref != cfg.n_ref:
        raise ValueError("solver N_ref differs from the configuration")
    cfg.require_production_inputs(controls, directions, solver, on_failure)     # no-op outside production
    if on_failure is None:
        on_failure = "continue"                                                 # synthetic_test default only
    if on_failure not in ("continue", "stop"):
        raise ValueError("on_failure must be 'continue' or 'stop'")
    check_run_parameters(time_cap_s, mem_cap_bytes)
    if max_configuration_bytes is not None and (type(max_configuration_bytes) is not int or max_configuration_bytes < 0):
        raise ValueError("max_configuration_bytes must be None or a non-negative int")
    out = Path(out_dir)
    scope = drop1_scope.determinism_scope()
    config_record = configuration_record(cfg, controls, directions, solver, on_failure,
                                         identity={"drop2": pins.drop2_identity(), "drop1": identity["drop1"],
                                                   "frozen": identity["frozen"]},
                                         scope_sha256=drop1_scope.scope_sha256(scope))
    # The admitted storage term is the exact length of this record over the
    # admitted inputs; a drifted (larger) record is refused before any output.
    if (max_configuration_bytes is not None and
            len(canonical.canonical_json_bytes(config_record)) > max_configuration_bytes):
        raise ProductionRefused("configuration record exceeds its admitted storage template; nothing was written")
    out.mkdir(parents=True, exist_ok=False)
    cfg_bytes = canonical.write_canonical_json_exclusive(out / R.CONFIGURATION_FILE, config_record)
    cfg_sha = canonical.sha256_bytes(cfg_bytes)
    failures_path = out / R.FAILURES_FILE
    failures_path.open("xb").close()
    (out / R.MEASUREMENTS_FILE).open("xb").close()
    sink = _Sink(out, cfg_sha)
    (out / "run").mkdir()
    t0, started = time.monotonic(), datetime.now(timezone.utc).isoformat()
    shard_status = {key: {"split": key[0], "shard": key[1], "status": "not_reached", "manifest_sha256": None}
                    for key in cfg.shards()}
    anomalies = {"relaxed": [], "binary": []}
    gate_failures = {name: [] for name in d10.GATE_NAMES}
    delta_max: dict[str, float] = {}
    counts = {"samples_written": 0, "directions_written": 0, "failure_records": 0, "unwritten_measurements": 0}
    timings, stop_reason = [], None
    primary = None
    evidence_errors = []

    def check():
        if time_cap_s is not None and time.monotonic() - t0 > time_cap_s:
            raise _Stop("time_cap")
        if mem_cap_bytes is not None and (peak_rss_bytes() or 0) > mem_cap_bytes:
            raise _Stop("memory_cap")

    try:
        for split, shard in cfg.shards():
            ts = time.monotonic()
            try:
                arrays, families, srows, drows, failed, verdicts = _build_shard(cfg, split, shard, controls, directions,
                                                                                solver, cfg_sha, check, sink)
            finally:
                if (split, shard) in sink.failed_shards:
                    shard_status[(split, shard)]["status"] = "failed"
            if failed:
                timings.append({"split": split, "shard": shard, "seconds": time.monotonic() - ts, "status": "failed"})
                if on_failure == "stop":
                    stop_reason = "sample_failure"
                    break
                continue
            sha = _write_shard(out, cfg, split, shard, cfg_sha, arrays, families, srows, drows)
            sink.pending.clear()  # Published bytes now retain these measurements.
            shard_status[(split, shard)].update(status="written", manifest_sha256=sha)
            counts["samples_written"] += len(srows)
            counts["directions_written"] += len(drows)
            for ident, verdict in verdicts:
                key = STRATUM_KEYS[ident["stratum"]]
                if verdict.anomaly_flagged:
                    anomalies[key].append(ident["sample_id"])
                for name in d10.GATE_NAMES:
                    if not verdict.gates[name]:
                        gate_failures[name].append(ident["sample_id"])
                if verdict.delta is not None:
                    delta_max[key] = max(delta_max.get(key, 0.0), verdict.delta)
            timings.append({"split": split, "shard": shard, "seconds": time.monotonic() - ts, "status": "written"})
    except _Stop as stop:
        stop_reason = stop.reason
    except BaseException as exc:          # always leave a summary, then re-raise
        stop_reason = exception_stop_reason(exc)
        primary = exc
    finally:
        try:
            sink.flush_pending("generation_stopped")
        except BaseException as exc:
            evidence_errors.append('unwritten measurements unavailable: '+_describe(exc))
        if primary is not None:
            try:
                # Adopted A: one bounded exception record, canonical bytes,
                # rendered before the file is created.
                from operatormark.corpus.bounded_exception import bounded_exception_record
                data = canonical.canonical_json_bytes({'exception': bounded_exception_record(primary)})
                with (out / 'run' / 'unexpected_failure.json').open('xb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException as exc:
                evidence_errors.append('original failure record unavailable: ' + _describe(exc))
        try:
            _finish_generation(out, cfg_sha, shard_status, stop_reason, sink, counts, anomalies,
                               gate_failures, delta_max, failures_path, started, t0, time_cap_s,
                               mem_cap_bytes, timings)
        except BaseException as exc:
            evidence_errors.append('summary/run evidence unavailable: '+_describe(exc))
        if evidence_errors:
            message='; '.join(evidence_errors)
            if primary is None:primary=RuntimeError(message)
            else:_note(primary,message)
            try:
                sys.stderr.write('EVIDENCE_PERSISTENCE_FAILURE '+message+'\n');sys.stderr.flush()
            except BaseException as reporting:   # a reporting failure must not replace the primary
                _note(primary,'evidence persistence reporting unavailable: '+_class_name(reporting))
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    summary=canonical.strict_canonical_json_loads((out / R.SUMMARY_FILE).read_bytes())
    return {"configuration_sha256": cfg_sha, "summary": summary}


def _finish_generation(out, cfg_sha, shard_status, stop_reason, sink, counts, anomalies,
                       gate_failures, delta_max, failures_path, started, t0, time_cap_s,
                       mem_cap_bytes, timings):
        complete = all(s["status"] == "written" for s in shard_status.values()) and stop_reason is None
        counts["failure_records"] = sink.failure_count
        counts["unwritten_measurements"] = len(sink.measurement_ids)
        summary = R.summary(
            configuration_sha256=cfg_sha, completion="complete" if complete else "incomplete", stop_reason=stop_reason,
            shards=list(shard_status.values()), counts=counts, anomalies=anomalies, gate_failures=gate_failures,
            delta_max={k: R.hx(v) for k, v in sorted(delta_max.items())},
            failures_sha256=canonical.file_sha256(failures_path),
            measurements_sha256=canonical.file_sha256(out / R.MEASUREMENTS_FILE),
            unwritten_anomalies=sink.unwritten_anomalies)
        canonical.write_canonical_json_exclusive(out / R.SUMMARY_FILE, summary)
        run = run_record(started=started, finished=datetime.now(timezone.utc).isoformat(),
                         wall_seconds=time.monotonic() - t0, peak_rss=peak_rss_bytes(), time_cap_s=time_cap_s,
                         mem_cap_bytes=mem_cap_bytes, timings=timings, stop_reason=stop_reason)
        (out / "run" / "generation_run.json").write_text(json.dumps(run, indent=1) + "\n")
