"""Bounded CPU adapter for the already frozen shared F64/P256 task schedule.

This does not issue admission or promote the R3 diagnostics runtime gate. The
caller supplies a separately reviewed native policy, fresh runtime receipt,
activation and external process supervision. Numerical task/solver modules are
loaded from the pinned source tree unchanged. No retry or resume entry exists.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import traceback

SCOPE = "native_F64_P256_optimisation_only"
TARGETS = 64
STEPS = 120
DIVISOR = 16
INPUT_SHAPES = {"final_starts": (64, 8, 8), "final_targets": (64, 64, 64),
                "final_targets_p256": (64, 64, 64)}
REQUIRED_SOURCES = {"operatormark_numerics_v2/task.py", "tools/q6_common.py",
                    "tools/msfinal/p256_adjoint.py", "tools/msfinal/p256_tasks.py",
                    "tools/msfinal/attacks.py"}
THREAD_ENV = dict(PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                 MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                 BLIS_NUM_THREADS="1", GOTO_NUM_THREADS="1", OMP_THREAD_LIMIT="1",
                 OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", PYTHONDONTWRITEBYTECODE="1")
OPERATIONS = ("optimization_forward", "adjoint", "scoring_t128", "scoring_p256")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_bytes(), parse_constant=invalid)


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def beneath(root, name):
    need(isinstance(name, str) and "\\" not in name and ":" not in name,
         "Expected a portable relative path")
    relative = PurePosixPath(name)
    need(relative.parts and not relative.is_absolute() and ".." not in relative.parts,
         "Unsafe relative path")
    path = root.joinpath(*relative.parts)
    need(not any(part.is_symlink() for part in (path, *path.parents) if part != root.parent),
         "Symbolic paths are not admitted")
    need(path.resolve().is_relative_to(root.resolve()), "Path escaped package")
    return path


def hash_identity(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def workload():
    return {arm: dict(optimization_forward=TARGETS * (STEPS + 1), adjoint=TARGETS * STEPS,
                      scoring_t128=TARGETS, scoring_p256=TARGETS)
            for arm in ("F64", "P256")}


def verify_files(root, policy):
    source_root = beneath(root, policy["source_root"])
    pins = policy["source_pins"]
    need(isinstance(pins, dict) and REQUIRED_SOURCES <= set(pins), "Native scientific pins incomplete")
    for name, expected in pins.items():
        path = beneath(source_root, name)
        need(hash_identity(expected) and path.is_file() and sha(path) == expected,
             "Frozen scientific source differs: " + name)
    need(set(policy["inputs"]) == set(INPUT_SHAPES), "Exact three native input arrays required")
    for key, expected in policy["inputs"].items():
        path = beneath(root, expected["path"])
        need(type(expected["bytes"]) is int and expected["bytes"] > 0
             and hash_identity(expected["sha256"]) and path.is_file()
             and path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"],
             "Frozen native input differs: " + key)
    need(hash_identity(policy["worker_sha256"]) and sha(__file__) == policy["worker_sha256"],
         "Native adapter source differs")
    return source_root


def admission(root, policy_path, deadline, expected_policy_sha=None):
    policy_path = Path(policy_path)
    if not policy_path.is_absolute():
        policy_path = beneath(root, policy_path.as_posix())
    need(policy_path.resolve().is_relative_to(root) and not policy_path.is_symlink(),
         "Policy must be within this package")
    policy_sha = sha(policy_path)
    if expected_policy_sha is not None:
        need(hash_identity(expected_policy_sha) and policy_sha == expected_policy_sha,
             "Frozen native policy differs")
    policy = read(policy_path)
    need(policy["scope"] == SCOPE and isinstance(policy["study_id"], str) and policy["study_id"]
         and hash_identity(policy["opening_sha256"]), "Native scientific scope differs")
    need(policy.get("operation_limits") == workload(), "Frozen native operation limits differ")
    act_path = root / "ACTIVATION.json"
    act = read(act_path)
    now = time.time()
    need(act["policy_sha256"] == policy_sha and isinstance(act["attempt"], str)
         and math.isfinite(act["start"]) and act["start"] <= now
         and math.isfinite(act["deadline"]) and now < deadline <= act["deadline"],
         "Fixed native activation/deadline binding differs")
    need(act.get("previous_clocks_reused", False) is False and act.get("retries", 0) == 0,
         "Retries and reused clocks are not admitted")
    gate = read(root / "NATIVE_RUNTIME_GATE.json")
    need(gate.get("passed") is True and gate.get("scope") == SCOPE
         and gate.get("policy_sha256") == policy_sha
         and gate.get("activation_sha256") == sha(act_path)
         and gate.get("deadline") == act["deadline"], "A bound native runtime gate is required")
    observed = gate.get("observed_at")
    need(type(observed) in (int, float) and math.isfinite(observed)
         and act["start"] <= observed <= now and now - observed <= 300,
         "Fresh native runtime observation required")
    source_root = verify_files(root, policy)
    need(time.time() < deadline, "Deadline elapsed during immutable input verification")
    need(all(os.environ.get(k) == v for k, v in THREAD_ENV.items()),
         "Native single-thread environment differs")
    return policy, policy_sha, act, gate, source_root


def load_kernels(source_root):
    """Only the pinned modules; no permissive fallback to installed science."""
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(source_root / "tools"), str(source_root)]
    import numpy as np
    task = importlib.import_module("operatormark_numerics_v2.task")
    common = importlib.import_module("q6_common")
    adjoint = importlib.import_module("msfinal.p256_adjoint")
    for module, relative in ((task, "operatormark_numerics_v2/task.py"),
                             (common, "tools/q6_common.py"),
                             (adjoint, "tools/msfinal/p256_adjoint.py")):
        need(Path(module.__file__).resolve() == (source_root / relative).resolve(),
             "A native module was imported from outside the pinned tree")
    return np, task, common.true_field, common.true_field_vjp, adjoint.Session


class Recorder:
    def __init__(self, out, deadline):
        self.out, self.deadline = out, deadline
        self.started = time.monotonic()
        self.last_progress = -float("inf")
        self.current = "input readback"
        self.sequence = 0
        self.completed_targets = {"F64": 0, "P256": 0}
        self.attempts = {a: dict.fromkeys(OPERATIONS, 0) for a in ("F64", "P256")}
        self.completions = {a: dict.fromkeys(OPERATIONS, 0) for a in ("F64", "P256")}
        self.pending = {}
        self.journal = (out / "native_calls.jsonl").open("x", encoding="utf-8", newline="\n")

    def close(self):
        self.journal.close()

    def check(self):
        if time.time() >= self.deadline:
            raise TimeoutError("Fixed native work deadline reached; no retry or clock reset")
        if shutil.disk_usage(self.out).free < 128 * 1024**2:
            raise RuntimeError("Less than 128 MiB local free space remains")
        if time.monotonic() - self.last_progress >= 20:
            self.last_progress = time.monotonic()
            row = dict(current=self.current, completed_targets=self.completed_targets.copy(),
                       total_targets=128, seconds=round(self.last_progress - self.started, 1))
            with (self.out / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
            print(json.dumps(row, sort_keys=True), flush=True)

    def event(self, value):
        self.sequence += 1
        row = dict(sequence=self.sequence, time=time.time(), **value)
        line = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        need(len(line.encode()) <= 4096 and self.journal.tell() + len(line.encode()) <= 128 * 1024**2,
             "Native event journal bound exceeded")
        self.journal.write(line)
        self.journal.flush()
        os.fsync(self.journal.fileno())

    def invoke(self, arm, target, operation, function):
        self.check()
        self.attempts[arm][operation] += 1
        call_id = sum(sum(v.values()) for v in self.attempts.values())
        self.pending[call_id] = dict(arm=arm, target=target, operation=operation)
        self.event(dict(event="intent", call_id=call_id, **self.pending[call_id]))
        value = function()
        self.completions[arm][operation] += 1
        self.event(dict(event="completion", call_id=call_id, **self.pending[call_id]))
        del self.pending[call_id]
        # A call completing after deadline is retained, but no further call starts.
        return value


def execute(root, policy, policy_sha, activation, gate, source_root, deadline):
    out = root / "output"
    out.mkdir(exist_ok=False)
    recorder = Recorder(out, deadline)
    try:
        put(out / "START.json", dict(scope=SCOPE, policy_sha256=policy_sha,
            activation_sha256=sha(root / "ACTIVATION.json"),
            native_runtime_gate_sha256=sha(root / "NATIVE_RUNTIME_GATE.json"),
            opening_sha256=policy["opening_sha256"], source_pins=policy["source_pins"],
            inputs=policy["inputs"], task_targets=64, steps=120, batch_loss_divisor=16,
            physical_qualification=False, scientific_acceptance=False,
            formal_custody_assurance=False, automatic_retries=0, deadline=deadline))
        np, task, true_field, true_field_vjp, Session = load_kernels(source_root)
        arrays = {}
        for key, shape in INPUT_SHAPES.items():
            recorder.check()
            arrays[key] = np.load(beneath(root, policy["inputs"][key]["path"]), allow_pickle=False)
            need(arrays[key].shape == shape and arrays[key].dtype == np.float64
                 and np.isfinite(arrays[key]).all(), "Native input shape/dtype/finiteness differs: " + key)
        starts, targets, ptargets = (arrays[n] for n in INPUT_SHAPES)
        need(np.all((starts >= 0) & (starts <= 1)), "Frozen starts outside the task box")
        need(all(np.sum(t * t, dtype=np.float64) > 0 for ts in (targets, ptargets) for t in ts),
             "Native targets must have positive norms")
        results = {}
        for arm in ("F64", "P256"):
            arm_dir = out / arm
            arm_dir.mkdir(exist_ok=False)
            rows = []
            for i in range(TARGETS):
                recorder.current = f"{arm}/target_{i:02d}"
                recorder.check()
                calls = 0
                active = []

                def oracle(rho):
                    nonlocal calls
                    calls += 1
                    need(calls <= STEPS + 1, "Frozen optimizer forward population differs")
                    final = calls == STEPS + 1
                    if arm == "F64":
                        if final:
                            field = recorder.invoke(arm, i, "optimization_forward", lambda: true_field(rho))
                            def unused(_):
                                raise ValueError("Unexpected final-field derivative")
                            return field, unused
                        field, original_vjp = recorder.invoke(arm, i, "optimization_forward",
                                                              lambda: true_field_vjp(rho))
                        consumed = False
                        def derivative(weight):
                            nonlocal consumed
                            need(not consumed, "Duplicate F64 adjoint call")
                            consumed = True
                            return recorder.invoke(arm, i, "adjoint", lambda: original_vjp(weight))
                        return field, derivative
                    session = recorder.invoke(arm, i, "optimization_forward", lambda: Session(rho))
                    active.append(session)
                    field = session.field64
                    if final:
                        session.close()
                        def unused(_):
                            raise ValueError("Unexpected final-field derivative")
                        return field, unused
                    consumed = False
                    def derivative(weight):
                        nonlocal consumed
                        need(not consumed, "Duplicate P256 adjoint call")
                        consumed = True
                        try:
                            return recorder.invoke(arm, i, "adjoint", lambda: session.vjp64(weight))
                        finally:
                            session.close()
                    return field, derivative

                try:
                    result = task.optimize(oracle, starts[i:i+1], targets[i:i+1],
                                           steps=STEPS, batch_size=DIVISOR)
                finally:
                    for session in active:
                        if not session.closed:
                            session.close()
                need(calls == STEPS + 1 and len(result.loss_trace) == STEPS,
                     "Incomplete native optimization trace")
                design = result.designs[0]
                need(design.shape == (8, 8) and design.dtype == np.float64 and np.isfinite(design).all(),
                     "Native design shape/dtype/finiteness differs")
                # Preserve the optimized design before either scoring call.
                with (arm_dir / f"design_{i:02d}.npy").open("xb") as stream:
                    np.save(stream, design, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                put(arm_dir / f"optimization_{i:02d}.json", dict(target=i,
                    trace=list(result.loss_trace), final_loss=result.final_loss,
                    steps=120, batch_loss_divisor=16, scoring_pending=True))
                t = recorder.invoke(arm, i, "scoring_t128", lambda: true_field(design))
                p = recorder.invoke(arm, i, "scoring_p256", lambda: true_field(design, True))
                row = dict(target=i, T=task.score(t, targets[i]), P256=task.score(p, ptargets[i]),
                           trace=list(result.loss_trace), batch_loss_divisor=16)
                if arm == "F64":
                    row["final_loss"] = result.final_loss
                else:
                    row.update(objective_target_family="T128", steps=120)
                put(arm_dir / f"task_{i:02d}.json", row)
                rows.append(row)
                recorder.completed_targets[arm] += 1
                print(f"{arm}: {i+1}/64 targets complete and saved locally", flush=True)
            need(recorder.attempts[arm] == recorder.completions[arm] == workload()[arm],
                 "Native operation population differs")
            counts = (dict(forward=7744, adjoint=7680, scoring_t128=64, scoring_p256=64, completed_targets=64)
                      if arm == "F64" else dict(P256_optimization_forward=7744, P256_adjoint=7680,
                          T128_scoring=64, P256_scoring=64, per_target_inverse_updates=7680))
            results[arm] = dict(complete=True, scope=SCOPE, counts=counts, rows=rows,
                physical_qualification=False, scientific_acceptance=False,
                production_acceptance=False, formal_custody_assurance=False)
            put(arm_dir / "RESULT.json", results[arm])
        recorder.check()
        verify_files(root, policy)
        need(not recorder.pending, "Native call completion missing")
        final = dict(complete=True, scope=SCOPE, policy_sha256=policy_sha,
            opening_sha256=policy["opening_sha256"], completed_targets=recorder.completed_targets,
            native_calls_attempted=recorder.attempts, native_calls_completed=recorder.completions,
            per_target_inverse_updates=15360, results={a:sha(out/a/"RESULT.json") for a in results},
            elapsed_seconds=time.monotonic()-recorder.started, scientific_acceptance=False,
            physical_qualification=False, formal_custody_assurance=False,
            producer_closure_pending=True, task_selection_pending=True,
            automatic_retries=0, previous_completed_diagnostics_repeated=0)
        put(out / "RESULT.json", final)
        return final
    except BaseException:
        put(out / "FAILURE.json", dict(complete=False, scope=SCOPE, policy_sha256=policy_sha,
            traceback=traceback.format_exc(), current=recorder.current,
            attempted=recorder.attempts, completed=recorder.completions,
            pending_calls=recorder.pending, completed_targets=recorder.completed_targets,
            automatic_retry=False, clock_reset=False, scientific_acceptance=False,
            physical_qualification=False, formal_custody_assurance=False))
        raise
    finally:
        recorder.close()


def run(root, policy_path, deadline, *, expected_policy_sha=None):
    root = Path(root).resolve()
    need(type(deadline) in (int, float) and math.isfinite(deadline), "Finite external deadline required")
    need(not (root / "output").exists(), "An output already exists; never rerun or overwrite this attempt")
    policy, policy_sha, act, gate, source = admission(root, policy_path, deadline, expected_policy_sha)
    return execute(root, policy, policy_sha, act, gate, source, deadline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--policy-sha", required=True)
    parser.add_argument("--deadline", required=True, type=float)
    args = parser.parse_args()
    run(args.root, "NATIVE_POLICY.json", args.deadline, expected_policy_sha=args.policy_sha)
