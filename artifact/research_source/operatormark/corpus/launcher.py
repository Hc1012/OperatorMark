"""Parent-only launcher and fresh-child pre-import environment gate.

The launcher uses a short, isolated invocation of ``child_controller.py``.
That stdlib-only script performs the complete pre-project bootstrap before it
imports the installed OperatorMark package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final


PROTOCOL_SHA256: Final = "a38ef79b6c4bdccf53f599fa08f77e48a111afbf0b2783ad5733bda99996599a"
BASELINE_COMMIT: Final = "e16456395cfe67b73b9d298fd0491cf9524d798a"
BASELINE_TREE: Final = "81f40e24102e51e0da3ce65dafd769a7678565fd"

MANDATORY_PARENT_ENVIRONMENT: Final[dict[str, str]] = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}

# State 2 enumerates these additional thread controls. State 3 must bind an
# explicit approved value for each variable shown applicable by its passive
# runtime inspection. A review-only candidate never satisfies that approval.
ADDITIONAL_THREAD_CONTROL_VARIABLES: Final[dict[str, tuple[str, ...]]] = {
    "openblas": ("OPENBLAS_DEFAULT_NUM_THREADS",),
    "mkl": ("MKL_DOMAIN_NUM_THREADS", "MKL_THREADING_LAYER"),
    "openmp": (
        "OMP_MAX_ACTIVE_LEVELS",
        "OMP_NESTED",
        "OMP_PLACES",
        "OMP_PROC_BIND",
        "OMP_WAIT_POLICY",
    ),
    "intel_openmp": ("KMP_AFFINITY", "KMP_BLOCKTIME", "KMP_HW_SUBSET"),
    "tbb": ("TBB_NUM_THREADS",),
}
ALL_RECOGNIZED_THREAD_VARIABLES: Final = frozenset(MANDATORY_PARENT_ENVIRONMENT) | {
    value
    for values in ADDITIONAL_THREAD_CONTROL_VARIABLES.values()
    for value in values
}
ALLOWED_ACTIONS: Final = {
    "state2-environment-probe": "state2_probe",
    "state2-rng-adversarial-probe": "state2_probe",
    "state3-environment-probe": "state3_environment_probe",
    "state3-microbenchmark": "state3_coordinator",
    "state3-microbenchmark-worker": "state3_worker",
    "state4-prepare": "state4",
    "state5-solve": "state5",
    "state6-seal": "state6",
}
ACTION_STATE: Final = {
    "state2-environment-probe": "state2",
    "state2-rng-adversarial-probe": "state2",
    "state3-environment-probe": "state3",
    "state3-microbenchmark": "state3",
    "state3-microbenchmark-worker": "state3",
    "state4-prepare": "state4",
    "state5-solve": "state5",
    "state6-seal": "state6",
}
PUBLIC_ACTIONS: Final = frozenset(ALLOWED_ACTIONS) - {"state3-microbenchmark-worker"}
THREAD_POLICY_KEYS: Final = {
    "schema",
    "state",
    "approval_status",
    "state3_authorization_file_sha256",
    "implementation_commit",
    "implementation_tree",
    "applicable_additional_thread_variables",
    "values",
    "policy_self_sha256",
}
CHILD_REAP_TIMEOUT_SECONDS: Final = 5.0


class LaunchGuardError(RuntimeError):
    """Raised when a parent or child launch requirement fails closed."""


@dataclass(frozen=True, slots=True)
class ParentLaunchSpec:
    state: str
    action: str
    authorization_file_sha256: str
    implementation_commit: str
    implementation_tree: str
    additional_environment: tuple[tuple[str, str], ...] = ()
    internal_worker: bool = False


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LaunchGuardError(f"{name} must be lowercase SHA-256 hexadecimal")
    return value


def _git_oid(value: object, *, name: str) -> str:
    """Validate this repository's full lowercase SHA-1 object name."""

    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LaunchGuardError(f"{name} must be a lowercase 40-hex Git object ID")
    return value


def required_environment(spec: ParentLaunchSpec) -> dict[str, str]:
    if spec.action not in ALLOWED_ACTIONS or ACTION_STATE[spec.action] != spec.state:
        raise LaunchGuardError("action is not authorized for the named state")
    _sha256(spec.authorization_file_sha256, name="authorization_file_sha256")
    _git_oid(spec.implementation_commit, name="implementation_commit")
    _git_oid(spec.implementation_tree, name="implementation_tree")
    if spec.action == "state3-microbenchmark-worker" and not spec.internal_worker:
        raise LaunchGuardError("State 3 worker action is coordinator-only")
    if spec.action != "state3-microbenchmark-worker" and spec.internal_worker:
        raise LaunchGuardError("internal worker binding is invalid for this action")
    required = dict(MANDATORY_PARENT_ENVIRONMENT)
    seen: set[str] = set()
    for name, value in spec.additional_environment:
        if name not in ALL_RECOGNIZED_THREAD_VARIABLES - set(MANDATORY_PARENT_ENVIRONMENT):
            raise LaunchGuardError(f"unrecognized additional thread variable: {name}")
        if not isinstance(value, str) or not value:
            raise LaunchGuardError(f"empty additional thread value: {name}")
        if name in required or name in seen:
            raise LaunchGuardError(f"duplicate thread variable: {name}")
        seen.add(name)
        required[name] = value
    return required


def build_child_environment(
    spec: ParentLaunchSpec,
    *,
    parent_environment: Mapping[str, str] | None = None,
    source_root: str | None = None,
) -> dict[str, str]:
    if source_root is not None and (type(source_root) is not str or not source_root):
        raise LaunchGuardError("bound source root must be explicit nonempty text")
    source = dict(os.environ if parent_environment is None else parent_environment)
    required = required_environment(spec)
    for name in ALL_RECOGNIZED_THREAD_VARIABLES:
        if name in source and name not in required:
            raise LaunchGuardError(
                f"recognized thread variable {name} is set but absent from the frozen state policy"
            )
    child = source
    child.update(required)
    child.update(
        {
            "OPERATORMARK_ACTION": spec.action,
            "OPERATORMARK_STATE": spec.state,
            "OPERATORMARK_PROTOCOL_SHA256": PROTOCOL_SHA256,
            "OPERATORMARK_AUTHORIZATION_FILE_SHA256": spec.authorization_file_sha256,
            "OPERATORMARK_IMPLEMENTATION_COMMIT": spec.implementation_commit,
            "OPERATORMARK_IMPLEMENTATION_TREE": spec.implementation_tree,
            "OPERATORMARK_SOURCE_ROOT": str(Path(__file__).resolve().parents[2]) if source_root is None else source_root,
            "OPERATORMARK_REQUIRED_ENVIRONMENT_JSON": json.dumps(
                required, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            ),
        }
    )
    if spec.internal_worker:
        child["OPERATORMARK_INTERNAL_WORKER"] = "1"
    else:
        child.pop("OPERATORMARK_INTERNAL_WORKER", None)
    return child


def verify_child_preimport_environment(
    *,
    environ: Mapping[str, str] | None = None,
    loaded_modules: Mapping[str, object] | None = None,
) -> dict[str, str]:
    actual_environment = os.environ if environ is None else environ
    modules = sys.modules if loaded_modules is None else loaded_modules
    forbidden_imports = sorted(
        name
        for name in modules
        if name == "numpy"
        or name.startswith("numpy.")
        or name == "scipy"
        or name.startswith("scipy.")
    )
    if forbidden_imports:
        raise LaunchGuardError("numerical module imported before the child environment gate")
    try:
        required = json.loads(actual_environment["OPERATORMARK_REQUIRED_ENVIRONMENT_JSON"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise LaunchGuardError("missing or invalid frozen environment record") from exc
    if not isinstance(required, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in required.items()
    ):
        raise LaunchGuardError("frozen environment record must map strings to strings")
    if set(MANDATORY_PARENT_ENVIRONMENT) - set(required):
        raise LaunchGuardError("frozen environment record omits mandatory variables")
    for name, value in required.items():
        if actual_environment.get(name) != value:
            raise LaunchGuardError(f"child environment mismatch for {name}")
    for name in ALL_RECOGNIZED_THREAD_VARIABLES:
        if name in actual_environment and name not in required:
            raise LaunchGuardError(f"conflicting unbound thread variable {name}")
    if actual_environment.get("OPERATORMARK_PROTOCOL_SHA256") != PROTOCOL_SHA256:
        raise LaunchGuardError("protocol commitment mismatch")
    action = actual_environment.get("OPERATORMARK_ACTION")
    state = actual_environment.get("OPERATORMARK_STATE")
    if action not in ACTION_STATE or ACTION_STATE[action] != state:
        raise LaunchGuardError("child action/state mismatch")
    return dict(required)


def environment_commitment(environment: Mapping[str, str]) -> str:
    payload = json.dumps(
        dict(environment), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(b"operatormark-parent-environment-v1\n" + payload).hexdigest()


def _kill_and_reap(process: subprocess.Popen[str]) -> None:
    kill_error: BaseException | None = None
    try:
        process.kill()
    except BaseException as exc:
        kill_error = exc
    try:
        process.wait(timeout=CHILD_REAP_TIMEOUT_SECONDS)
    except BaseException as wait_error:
        if kill_error is not None:
            raise wait_error from kill_error
        raise
    if process.poll() is None:
        if kill_error is not None:
            raise LaunchGuardError(
                "child kill failed and the child was not reaped"
            ) from kill_error
        raise LaunchGuardError("killed child was not reaped")


def _terminate_and_reap_after_callback_failure(process: subprocess.Popen[str]) -> None:
    """Terminate and reap a child whose post-creation callback failed."""

    try:
        running = process.poll() is None
    except BaseException:
        running = True
    if running:
        try:
            process.terminate()
        except BaseException:
            _kill_and_reap(process)
            return
    try:
        process.wait(timeout=CHILD_REAP_TIMEOUT_SECONDS)
    except BaseException:
        _kill_and_reap(process)
    if process.poll() is None:
        raise LaunchGuardError("callback-failed child was not reaped")


def launch_fresh_child(
    spec: ParentLaunchSpec,
    child_arguments: Sequence[str] = (),
    *,
    check: bool = False,
    on_child_started: Callable[[], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    if spec.state in {"state4", "state5", "state6"}:
        raise LaunchGuardError("production later-state launcher is disabled")
    environment = build_child_environment(spec)
    controller = Path(__file__).with_name("child_controller.py").resolve(strict=True)
    if spec.state == "state3":
        expected_interpreter = (
            Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe"
        ).resolve(strict=True)
        if Path(sys.executable).resolve(strict=True) != expected_interpreter:
            raise LaunchGuardError(
                "State 3 requires the exact repository virtual-environment interpreter"
            )
    command = [sys.executable, "-I", "-B", str(controller), *child_arguments]
    process = subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        text=True,
        encoding="utf-8",
    )
    try:
        if on_child_started is not None:
            on_child_started()
        stdout, stderr = process.communicate()
    except BaseException as callback_error:
        try:
            _terminate_and_reap_after_callback_failure(process)
        except BaseException as cleanup_error:
            callback_error.add_note(
                f"child cleanup also failed: {type(cleanup_error).__name__}"
            )
            raise callback_error from cleanup_error
        raise
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


@dataclass(frozen=True, slots=True)
class BoundBinaryLaunch:
    """Independently pinned synthetic process transport; not state authority."""
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    cwd: str
    capsule_length: int
    capsule_sha256: str
    command_binding_sha256: str
    inherited_descriptors: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BinaryLaunchResult:
    pid: int
    direct_exit_code: int
    stdout: bytes
    stderr: bytes
    reap_status: str


def bound_command_commitment(*, command, environment, cwd, capsule_length,
                             capsule_sha256, inherited_descriptors):
    projection = dict(command=list(command), environment=[list(x) for x in environment], cwd=cwd,
                      capsule_length=capsule_length, capsule_sha256=capsule_sha256,
                      inherited_descriptors=list(inherited_descriptors))
    data = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(b"operatormark-launch-command-v1\n"+data).hexdigest()


def launch_bound_child(*, expected, capsule, process_factory=None, mode=None,
                       on_child_started=None):
    """Actual binary pipe path, enabled only with explicit synthetic injection.

The trusted caller binds helper execution before invoking this function and
supplies independently accepted command/capsule pins. No source path is opened.
No production caller obtains authority by constructing this private value.
"""
    if mode != "synthetic" or process_factory is None:
        raise LaunchGuardError("bound production process provider is disabled")
    if type(expected) is not BoundBinaryLaunch or type(capsule) is not bytes:
        raise LaunchGuardError("exact independent binary launch pins required")
    if (type(expected.command) is not tuple or len(expected.command) < 4
        or not all(type(x) is str and x and "\x00" not in x for x in expected.command)
        or expected.command[1:3] != ("-I", "-B")
        or type(expected.cwd) is not str or not expected.cwd
        or type(expected.environment) is not tuple
        or not all(type(x) is tuple and len(x) == 2 and all(type(v) is str for v in x) for x in expected.environment)
        or tuple(sorted(dict(expected.environment).items())) != expected.environment
        or type(expected.capsule_length) is not int or expected.capsule_length != len(capsule)
        or hashlib.sha256(capsule).hexdigest() != expected.capsule_sha256
        or expected.inherited_descriptors != (0, 1, 2)):
        raise LaunchGuardError("binary launch command/environment/transport mismatch")
    commitment = bound_command_commitment(command=expected.command, environment=expected.environment,
        cwd=expected.cwd, capsule_length=expected.capsule_length, capsule_sha256=expected.capsule_sha256,
        inherited_descriptors=expected.inherited_descriptors)
    if expected.command_binding_sha256 != commitment:
        raise LaunchGuardError("independently fixed command commitment mismatch")
    process = process_factory(list(expected.command), cwd=expected.cwd, env=dict(expected.environment),
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              shell=False, text=False, close_fds=True)
    try:
        if type(process.pid) is not int or process.pid <= 0:
            raise LaunchGuardError("direct child identity unavailable")
        # Creation has already happened. A callback exception never unconsumes it.
        if on_child_started is not None:
            on_child_started(process)
        stdout, stderr = process.communicate(input=capsule)
        code = process.wait()
        if type(stdout) is not bytes or type(stderr) is not bytes or type(code) is not int:
            raise LaunchGuardError("binary capture/direct exit unavailable")
        return BinaryLaunchResult(process.pid, code, stdout, stderr, "observed_reaped")
    except BaseException as primary:
        cleanup = []
        direct_exit = None
        try:
            process.terminate()
        except BaseException as exc:
            cleanup.append(("terminate", exc))
        try:
            direct_exit = process.wait(timeout=CHILD_REAP_TIMEOUT_SECONDS)
        except BaseException as exc:
            cleanup.append(("reap_after_terminate", exc))
            try:
                process.kill()
            except BaseException as kill_error:
                cleanup.append(("kill", kill_error))
            try:
                direct_exit = process.wait(timeout=CHILD_REAP_TIMEOUT_SECONDS)
            except BaseException as reap_error:
                cleanup.append(("reap_after_kill", reap_error))
        output = error = None
        try:
            # No new input/write or new child: drain surviving owned pipes.
            # This observation stays distinct from the failed original capture.
            if type(direct_exit) is int:
                output, error = process.communicate()
                if type(output) is not bytes or type(error) is not bytes:
                    output = error = None
                    raise LaunchGuardError("cleanup stream observation unavailable")
        except BaseException as exc:
            cleanup.append(("cleanup_stream_capture", exc))
        # Do not replace the primary cause/context/traceback with cleanup errors.
        dictionary = BaseException.__dict__["__dict__"].__get__(primary, type(primary))
        dictionary["_state_execution_cleanup_failures"] = tuple(cleanup)
        dictionary["_state_execution_process_facts"] = dict(
            direct_exit_code=direct_exit if type(direct_exit) is int else None,
            cleanup_stdout=output, cleanup_stderr=error,
            cleanup_streams="observed_after_failure" if output is not None and error is not None else "unavailable")
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded OperatorMark fresh-child launcher")
    parser.add_argument("--state", required=True)
    parser.add_argument("--action", required=True, choices=sorted(PUBLIC_ACTIONS))
    parser.add_argument("--authorization-file-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--implementation-tree", required=True)
    parser.add_argument(
        "--additional-thread-policy-file",
        help="canonical user-approved additional backend thread-policy JSON",
    )
    parser.add_argument(
        "--bindings-file",
        help="canonical metadata-only pre-measurement feasibility bindings JSON",
    )
    parser.add_argument("child_arguments", nargs="*")
    return parser


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LaunchGuardError(f"duplicate thread-policy JSON key: {key}")
        result[key] = value
    return result


def load_approved_additional_thread_policy(
    path: str | Path,
    *,
    authorization_file_sha256: str,
    implementation_commit: str,
    implementation_tree: str,
) -> tuple[tuple[str, str], ...]:
    """Load a canonical authorization-bound policy; candidates are rejected."""

    try:
        data = Path(path).read_bytes()
        record = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                LaunchGuardError(f"non-finite thread-policy token: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LaunchGuardError("additional thread-policy file is unavailable or invalid") from exc
    if not isinstance(record, dict) or set(record) != THREAD_POLICY_KEYS:
        raise LaunchGuardError("additional thread-policy keys are incomplete or unexpected")
    canonical = json.dumps(
        record, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    if data != canonical:
        raise LaunchGuardError("additional thread-policy bytes are not canonical JSON")
    claimed_self_hash = record["policy_self_sha256"]
    _sha256(claimed_self_hash, name="thread_policy.policy_self_sha256")
    payload = dict(record)
    payload.pop("policy_self_sha256")
    actual_self_hash = hashlib.sha256(
        json.dumps(
            payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    if claimed_self_hash != actual_self_hash:
        raise LaunchGuardError("additional thread-policy self-hash mismatch")
    _sha256(
        record["state3_authorization_file_sha256"],
        name="thread_policy.state3_authorization_file_sha256",
    )
    _git_oid(record["implementation_commit"], name="thread_policy.implementation_commit")
    _git_oid(record["implementation_tree"], name="thread_policy.implementation_tree")
    if (
        record["schema"]
        != "operatormark.darcy_branch_c_v1.additional_thread_policy.v1"
        or record["state"] != "state3"
        or record["approval_status"] != "explicitly_approved_for_state3"
    ):
        raise LaunchGuardError("review-only or non-State-3 thread policy is not executable")
    if (
        record["state3_authorization_file_sha256"] != authorization_file_sha256
        or record["implementation_commit"] != implementation_commit
        or record["implementation_tree"] != implementation_tree
    ):
        raise LaunchGuardError("additional thread policy does not match State 3 bindings")
    names = record["applicable_additional_thread_variables"]
    values = record["values"]
    if (
        not isinstance(names, list)
        or not all(isinstance(name, str) for name in names)
        or names != sorted(set(names))
        or not isinstance(values, dict)
        or set(values) != set(names)
    ):
        raise LaunchGuardError("additional thread-policy names and values disagree")
    permitted = ALL_RECOGNIZED_THREAD_VARIABLES - set(MANDATORY_PARENT_ENVIRONMENT)
    if any(name not in permitted for name in names):
        raise LaunchGuardError("additional thread policy contains an unrecognized variable")
    if not all(isinstance(value, str) and value for value in values.values()):
        raise LaunchGuardError("additional thread policy contains an empty or non-string value")
    return tuple((name, values[name]) for name in names)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    later_state = arguments.state in {"state3", "state4", "state5", "state6"}
    if later_state != bool(arguments.additional_thread_policy_file):
        raise LaunchGuardError(
            "later states require exactly one approved additional thread-policy file"
        )
    additional_environment = ()
    if arguments.additional_thread_policy_file:
        additional_environment = load_approved_additional_thread_policy(
            arguments.additional_thread_policy_file,
            authorization_file_sha256=arguments.authorization_file_sha256,
            implementation_commit=arguments.implementation_commit,
            implementation_tree=arguments.implementation_tree,
        )
    child_arguments = tuple(arguments.child_arguments)
    if arguments.action == "state3-microbenchmark":
        if not arguments.bindings_file or child_arguments:
            raise LaunchGuardError(
                "State 3 coordinator requires exactly one --bindings-file and no other arguments"
            )
        child_arguments = (arguments.bindings_file,)
    elif arguments.bindings_file:
        raise LaunchGuardError("compute-feasibility bindings are valid only for State 3")
    spec = ParentLaunchSpec(
        state=arguments.state,
        action=arguments.action,
        authorization_file_sha256=arguments.authorization_file_sha256,
        implementation_commit=arguments.implementation_commit,
        implementation_tree=arguments.implementation_tree,
        additional_environment=additional_environment,
    )
    result = launch_fresh_child(spec, child_arguments)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
