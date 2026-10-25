"""Committed two-phase State 3 workflow with a metadata-only parent process.

The administrative phase is repeatable and cannot launch a child.  The execute
phase reaches scientific code only through :func:`launcher.launch_fresh_child`.
Nothing in this module imports a numerical, solver, fixture, RNG, or corpus-data
module.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from . import launcher, state3_audit, storage
from .canonical import (
    canonical_json_bytes,
    file_sha256,
    strict_canonical_json_loads,
    validate_self_hash,
    with_self_hash,
    write_canonical_json_exclusive,
)


PROTOCOL_SHA256: Final = state3_audit.PROTOCOL_SHA256
GATE9_AUDIT_SHA256: Final = (
    "c86b5d5b124609dce13ee2b5ed74ae140c8f980ae6fb8053ca6c8cda31938450"
)
STATE2_CORRECTIVE_AUDIT_SIZE: Final = 233_984
STATE2_CORRECTIVE_AUDIT_SHA256: Final = (
    "104efd5e764f7a2615f914b23f63f11d7a4cf2585823d7de5530935f0c0c6e21"
)
STATE2_ENTRYPOINTS_AUDIT_SIZE: Final = 29_631
STATE2_ENTRYPOINTS_AUDIT_SHA256: Final = (
    "e6f8ce0905d3c018c090d01ea5de8a8a02f4d8fb657ea464314393dc20e236b3"
)
REVOKED_STATE3_V1_4_SIZE: Final = 24_101
REVOKED_STATE3_V1_4_SHA256: Final = (
    "4ada7254e920b802a8f17ee2ede9223e1af76b34de0a2029483a859e44184243"
)
OPENBLAS_DEFAULT_POLICY: Final = (("OPENBLAS_DEFAULT_NUM_THREADS", "1"),)
ADMINISTRATIVE_RECEIPT_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.state3_administrative_preflight_receipt.v1"
)
PRE_LATCH_STATE_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.state3_pre_latch_execution_state.v1"
)
LATCH_SCHEMA: Final = "operatormark.darcy_branch_c_v1.state3_execution_latch.v1"
COMPLETION_SCHEMA: Final = "operatormark.darcy_branch_c_v1.state3_workflow_completion.v1"

STATE3_IMPLEMENTATION_PATHS: Final = (
    "operatormark/corpus/canonical.py",
    "operatormark/corpus/child_controller.py",
    "operatormark/corpus/launcher.py",
    "operatormark/corpus/microbenchmark.py",
    "operatormark/corpus/rng_guard.py",
    "operatormark/corpus/runtime.py",
    "operatormark/corpus/state3_audit.py",
    "operatormark/corpus/state3_audit_verify.py",
    "operatormark/corpus/state3_workflow.py",
    "operatormark/corpus/storage.py",
)

_FORBIDDEN_PARENT_IMPORT_PREFIXES: Final = (
    "numpy",
    "scipy",
    "operatormark.benchmarks",
    "operatormark.evaluation",
    "operatormark.solvers",
    "operatormark.corpus.base_imports",
    "operatormark.corpus.child_controller",
    "operatormark.corpus.microbenchmark",
    "operatormark.corpus.rng_guard",
    "operatormark.corpus.runtime",
    "operatormark.corpus.state4",
)


class State3WorkflowError(RuntimeError):
    """Raised when the committed workflow cannot prove an exact safe transition."""


FileBindingReader = Callable[[Path], tuple[int, str]]
RecordPublisher = Callable[[Path, Mapping[str, Any]], bytes]
ProcessRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]
ChildLauncher = Callable[..., subprocess.CompletedProcess[str]]
GitRecordReader = Callable[[Path, str, str, str], Mapping[str, Any]]
SourceRecordBuilder = Callable[..., Mapping[str, Any]]
FailureBindingValidator = Callable[[Path, Mapping[str, Any]], None]


class _UniqueValue(argparse.Action):
    """Reject repeated security-sensitive options instead of accepting the last."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may be supplied exactly once")
        setattr(namespace, self.dest, values)


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise State3WorkflowError(f"{name} must be lowercase SHA-256 hexadecimal")
    return value


def _git_oid(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise State3WorkflowError(f"{name} must be a lowercase 40-hex Git object ID")
    return value


def _nonnegative(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise State3WorkflowError(f"{name} must be a non-negative integer")
    return value


def _assert_parent_import_boundary() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _FORBIDDEN_PARENT_IMPORT_PREFIXES
        )
    )
    if loaded:
        raise State3WorkflowError(
            "State 3 parent workflow has a forbidden import closure: " + ",".join(loaded)
        )


def _default_file_binding(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise State3WorkflowError("external binding is not one ordinary file")
    return path.stat().st_size, file_sha256(path)


def _literal_path_commitment(value: str) -> str:
    return hashlib.sha256(
        b"operatormark-literal-path-v1\n" + value.encode("utf-8")
    ).hexdigest()


def _external_binding(
    literal_path: str,
    *,
    expected_sha256: str,
    expected_size: int | None,
    reader: FileBindingReader,
) -> dict[str, Any]:
    _sha256(expected_sha256, name="expected external SHA-256")
    observed_size, observed_sha256 = reader(Path(literal_path))
    if observed_sha256 != expected_sha256 or (
        expected_size is not None and observed_size != expected_size
    ):
        raise State3WorkflowError("external file size or SHA-256 binding mismatch")
    return {
        "filename": Path(literal_path).name,
        "literal_path_sha256": _literal_path_commitment(literal_path),
        "sha256": observed_sha256,
        "size": observed_size,
    }


def _relative_artifact_path(value: str, *, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise State3WorkflowError(f"{name} must be canonical repository-relative POSIX text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or path.parts[0] != "artifacts"
        or any(part in {"", ".", ".."} for part in path.parts)
        or "private_corpus_v1" in {part.casefold() for part in path.parts}
    ):
        raise State3WorkflowError(f"{name} must be safely contained under artifacts")
    return path


def _resolved_artifact_path(root: Path, value: str, *, name: str) -> Path:
    relative = _relative_artifact_path(value, name=name)
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise State3WorkflowError(f"{name} parent cannot be a symbolic link")
    return root.joinpath(*relative.parts)


def _default_process_runner(
    command: Sequence[str], root: Path,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        list(command),
        cwd=root,
        env=environment,
        check=False,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )


def _git_result(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return _default_process_runner(("git", *arguments), root)


def _git_output(root: Path, *arguments: str) -> str:
    result = _git_result(root, *arguments)
    if result.returncode != 0 or result.stderr:
        raise State3WorkflowError("Git metadata query failed: " + " ".join(arguments))
    return result.stdout.rstrip("\r\n")


def _capture_git_record(
    root: Path,
    implementation_commit: str,
    implementation_tree: str,
    expected_parent_commit: str,
) -> dict[str, Any]:
    branch = _git_output(root, "branch", "--show-current")
    head = _git_output(root, "rev-parse", "HEAD")
    tree = _git_output(root, "show", "-s", "--format=%T", "HEAD")
    parents = _git_output(root, "show", "-s", "--format=%P", "HEAD").split()
    tracked = _git_result(root, "diff", "--no-ext-diff", "--binary", "--")
    staged = _git_result(root, "diff", "--cached", "--no-ext-diff", "--binary", "--")
    status = _git_output(root, "status", "--short", "--untracked-files=all")
    if any(result.returncode != 0 or result.stderr for result in (tracked, staged)):
        raise State3WorkflowError("Git diff metadata query failed")
    tracked_bytes = tracked.stdout.encode("utf-8")
    staged_bytes = staged.stdout.encode("utf-8")
    record = {
        "branch": branch,
        "head": head,
        "tree": tree,
        "parents": parents,
        "status_short_untracked_all": status,
        "tracked_diff_sha256": hashlib.sha256(tracked_bytes).hexdigest(),
        "tracked_diff_size_bytes": len(tracked_bytes),
        "staged_diff_sha256": hashlib.sha256(staged_bytes).hexdigest(),
        "staged_diff_size_bytes": len(staged_bytes),
        "tracked_changes_empty": tracked_bytes == b"",
        "staged_changes_empty": staged_bytes == b"",
        "untracked_inclusive_status_empty": status == "",
    }
    if branch != "main" or head != implementation_commit or tree != implementation_tree:
        raise State3WorkflowError("live Git identity differs from the bound implementation")
    state3_audit.validate_git_record(
        record,
        implementation_commit=implementation_commit,
        implementation_tree=implementation_tree,
        parent_commit=expected_parent_commit,
        name="state3_workflow.git",
    )
    return record


def _require_exact_interpreter(root: Path) -> None:
    expected = (root / ".venv" / "Scripts" / "python.exe").resolve(strict=True)
    if Path(sys.executable).resolve(strict=True) != expected:
        raise State3WorkflowError(
            "State 3 workflow requires the exact repository virtual-environment interpreter"
        )


def _git_blob_bytes(root: Path, object_id: str) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", object_id],
        cwd=root,
        check=False,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0 or result.stderr:
        raise State3WorkflowError("Git blob read failed")
    return result.stdout


def _git_blob_id(root: Path, commit: str, path: str) -> str:
    return _git_output(root, "rev-parse", f"{commit}:{path}")


def _build_source_record(
    *,
    root: Path,
    authorization_file_sha256: str,
    implementation_commit: str,
    implementation_tree: str,
    implementation_parent_commit: str,
    git_before: Mapping[str, Any],
    git_after: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_rows: list[dict[str, Any]] = []
    for path in sorted(state3_audit.FROZEN_BASELINE_REQUIRED_PATHS):
        baseline_blob = _git_blob_id(root, state3_audit.SOLVER_SPEC_BASELINE_COMMIT, path)
        implementation_blob = _git_blob_id(root, implementation_commit, path)
        baseline_sha = hashlib.sha256(_git_blob_bytes(root, baseline_blob)).hexdigest()
        implementation_path = root.joinpath(*PurePosixPath(path).parts)
        implementation_sha = file_sha256(implementation_path)
        baseline_rows.append(
            {
                "path": path,
                "baseline_git_blob": baseline_blob,
                "implementation_git_blob": implementation_blob,
                "baseline_file_sha256": baseline_sha,
                "implementation_file_sha256": implementation_sha,
                "byte_identical": (
                    baseline_blob == implementation_blob and baseline_sha == implementation_sha
                ),
            }
        )
    implementation_rows: list[dict[str, Any]] = []
    for path in STATE3_IMPLEMENTATION_PATHS:
        blob = _git_blob_id(root, implementation_commit, path)
        current = root.joinpath(*PurePosixPath(path).parts)
        current_sha = file_sha256(current)
        blob_sha = hashlib.sha256(_git_blob_bytes(root, blob)).hexdigest()
        implementation_rows.append(
            {
                "path": path,
                "git_blob": blob,
                "file_sha256": current_sha,
                "matches_git_blob": current_sha == blob_sha,
            }
        )
    lock_path = "requirements-lock-cpu.txt"
    lock_blob = _git_blob_id(root, implementation_commit, lock_path)
    return state3_audit.build_source_blob_verification_record(
        protocol_file_sha256=PROTOCOL_SHA256,
        state3_authorization_file_sha256=authorization_file_sha256,
        implementation_commit=implementation_commit,
        implementation_tree=implementation_tree,
        implementation_parent_commit=implementation_parent_commit,
        requirements_lock_git_blob=lock_blob,
        requirements_lock_file_sha256=file_sha256(root / lock_path),
        frozen_baseline_blobs=baseline_rows,
        implementation_files=implementation_rows,
        git_before=git_before,
        git_after=git_after,
    )


def _thread_policy_record(
    *, authorization_file_sha256: str, implementation_commit: str,
    implementation_tree: str, openblas_default_num_threads: str,
) -> dict[str, Any]:
    if openblas_default_num_threads != "1":
        raise State3WorkflowError(
            "OPENBLAS_DEFAULT_NUM_THREADS must be the exact frozen string 1"
        )
    return with_self_hash(
        {
            "applicable_additional_thread_variables": [
                "OPENBLAS_DEFAULT_NUM_THREADS"
            ],
            "approval_status": "explicitly_approved_for_state3",
            "implementation_commit": implementation_commit,
            "implementation_tree": implementation_tree,
            "schema": "operatormark.darcy_branch_c_v1.additional_thread_policy.v1",
            "state": "state3",
            "state3_authorization_file_sha256": authorization_file_sha256,
            "values": {"OPENBLAS_DEFAULT_NUM_THREADS": "1"},
        },
        "policy_self_sha256",
    )


def _validate_thread_policy(
    record: Mapping[str, Any], *, authorization_file_sha256: str,
    implementation_commit: str, implementation_tree: str,
) -> tuple[tuple[str, str], ...]:
    expected = _thread_policy_record(
        authorization_file_sha256=authorization_file_sha256,
        implementation_commit=implementation_commit,
        implementation_tree=implementation_tree,
        openblas_default_num_threads="1",
    )
    if canonical_json_bytes(record) != canonical_json_bytes(expected):
        raise State3WorkflowError("additional thread policy is not the exact frozen policy")
    try:
        validate_self_hash(record, "policy_self_sha256")
    except ValueError as exc:
        raise State3WorkflowError("additional thread policy self-hash mismatch") from exc
    # The launcher validator is exercised against a file by the execute phase.  This
    # in-memory equality is the deliberately narrower workflow-level contract.
    return OPENBLAS_DEFAULT_POLICY


def _load_canonical_record(path: Path, *, self_hash_field: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise State3WorkflowError("required workflow record is not one ordinary file")
    try:
        record = strict_canonical_json_loads(path.read_bytes())
        if self_hash_field is not None:
            validate_self_hash(record, self_hash_field)
    except (OSError, ValueError) as exc:
        raise State3WorkflowError("required workflow record is invalid") from exc
    return record


def _publish_record_default(path: Path, record: Mapping[str, Any]) -> bytes:
    return write_canonical_json_exclusive(path, record)


def _validate_existing_expected(path: Path, expected: bytes) -> bool:
    if path.is_symlink():
        raise State3WorkflowError("existing deterministic artifact is a symbolic link")
    if not path.exists():
        return False
    if not path.is_file():
        raise State3WorkflowError("existing deterministic artifact is not an ordinary file")
    try:
        actual = path.read_bytes()
        strict_canonical_json_loads(actual)
    except (OSError, ValueError) as exc:
        raise State3WorkflowError("existing deterministic artifact is malformed") from exc
    if actual != expected:
        raise State3WorkflowError("existing deterministic artifact differs byte-for-byte")
    return True


def _publish_deterministic_pair(
    first_path: Path,
    first_record: Mapping[str, Any],
    second_path: Path,
    second_record: Mapping[str, Any],
    *,
    publisher: RecordPublisher,
) -> tuple[bytes, bytes]:
    """Validate both destinations before exclusively publishing either missing file."""

    first_data = canonical_json_bytes(first_record)
    second_data = canonical_json_bytes(second_record)
    first_exists = _validate_existing_expected(first_path, first_data)
    second_exists = _validate_existing_expected(second_path, second_data)
    if not first_exists:
        observed = publisher(first_path, first_record)
        if observed != first_data:
            raise State3WorkflowError("policy publisher returned unexpected bytes")
        _validate_existing_expected(first_path, first_data)
    if not second_exists:
        observed = publisher(second_path, second_record)
        if observed != second_data:
            raise State3WorkflowError("receipt publisher returned unexpected bytes")
        _validate_existing_expected(second_path, second_data)
    return first_data, second_data


def _output_plan(run_directory: str, archive_filename: str) -> dict[str, Any]:
    run = _relative_artifact_path(run_directory, name="run_directory")
    if run.suffix:
        raise State3WorkflowError("run_directory must name a directory")
    if (
        not archive_filename
        or PurePosixPath(archive_filename).name != archive_filename
        or not archive_filename.endswith(".zip")
    ):
        raise State3WorkflowError("archive_filename must be one local ZIP filename")
    prefix = run.as_posix()
    return {
        "additional_thread_policy": f"{prefix}/additional_thread_policy.json",
        "administrative_preflight_receipt": (
            f"{prefix}/administrative_preflight_receipt.json"
        ),
        "archive": f"{prefix}/{archive_filename}",
        "audit_evidence_directory": f"{prefix}/audit_evidence",
        "compute_feasibility_preflight_bindings": (
            f"{prefix}/compute_feasibility_preflight_bindings.json"
        ),
        "environment_runtime": f"{prefix}/environment_runtime.json",
        "failure_inventory": f"{prefix}/failure_inventory.json",
        "independent_verification": f"{prefix}/independent_verification.json",
        "measurement_bundle": f"{prefix}/measurement_bundle.json",
        "pre_latch_state_pattern": f"{prefix}/pre_latch_attempt_{{attempt:04d}}.json",
        "prohibited_action_counts": f"{prefix}/prohibited_action_counts.json",
        "source_blob_verification": f"{prefix}/source_blob_verification.json",
        "state3_execution_latch": f"{prefix}/state3_execution_latch.json",
        "storage_preflight_after": f"{prefix}/storage_preflight_after.json",
        "storage_preflight_before_pattern": (
            f"{prefix}/storage_attempt_{{attempt:04d}}/storage_preflight_before.json"
        ),
        "workflow_completion": f"{prefix}/workflow_completion.json",
    }


def _validate_output_plan(record: object) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise State3WorkflowError("output plan must be a mapping")
    expected_keys = set(_output_plan("artifacts/x", "x.zip"))
    if set(record) != expected_keys or not all(
        isinstance(value, str) for value in record.values()
    ):
        raise State3WorkflowError("output plan keys are incomplete or unexpected")
    for key, value in record.items():
        if "{attempt:04d}" in value:
            value = value.replace("{attempt:04d}", "0001")
        _relative_artifact_path(value, name=f"output_plan.{key}")
    return dict(record)


def _disk_estimate(arguments: argparse.Namespace) -> dict[str, int]:
    return state3_audit.build_writer_disk_estimate_metadata(
        projected_npy_header_bytes=arguments.projected_npy_header_bytes,
        projected_jsonl_bytes=arguments.projected_jsonl_bytes,
        projected_manifest_and_audit_bytes=(
            arguments.projected_manifest_and_audit_bytes
        ),
        controls_only_bytes=arguments.controls_only_bytes,
        projected_audit_bytes=arguments.projected_audit_bytes,
    )


def _external_bindings(
    arguments: argparse.Namespace, *, reader: FileBindingReader,
) -> dict[str, dict[str, Any]]:
    return {
        "authorization": _external_binding(
            arguments.authorization_file,
            expected_sha256=arguments.authorization_file_sha256,
            expected_size=arguments.authorization_file_size,
            reader=reader,
        ),
        "frozen_protocol": _external_binding(
            arguments.protocol_file,
            expected_sha256=PROTOCOL_SHA256,
            expected_size=arguments.protocol_file_size,
            reader=reader,
        ),
        "gate9_metadata_only_audit": _external_binding(
            arguments.gate9_audit_file,
            expected_sha256=GATE9_AUDIT_SHA256,
            expected_size=arguments.gate9_audit_file_size,
            reader=reader,
        ),
        "state2_corrective_audit": _external_binding(
            arguments.state2_corrective_audit_file,
            expected_sha256=STATE2_CORRECTIVE_AUDIT_SHA256,
            expected_size=STATE2_CORRECTIVE_AUDIT_SIZE,
            reader=reader,
        ),
        "state2_entrypoints_audit": _external_binding(
            arguments.state2_entrypoints_audit_file,
            expected_sha256=STATE2_ENTRYPOINTS_AUDIT_SHA256,
            expected_size=STATE2_ENTRYPOINTS_AUDIT_SIZE,
            reader=reader,
        ),
        "revoked_state3_v1_4_authorization": _external_binding(
            arguments.revoked_state3_authorization_file,
            expected_sha256=REVOKED_STATE3_V1_4_SHA256,
            expected_size=REVOKED_STATE3_V1_4_SIZE,
            reader=reader,
        ),
    }


def _validate_receipt(
    record: Mapping[str, Any], *, authorization_file_sha256: str,
    implementation_commit: str, implementation_tree: str,
    implementation_parent_commit: str,
) -> None:
    expected_keys = {
        "disk_estimate", "external_bindings", "failure_inventory_commitment",
        "implementation_commit", "implementation_parent_commit",
        "implementation_tree", "interpreter", "output_plan", "policy_commitment",
        "receipt_self_sha256", "schema", "source_blob_commitment",
        "state3_authority_consumed", "state3_authorization_file_sha256",
        "state3_execution_latch_reached", "storage_attempt_relative_paths",
    }
    if set(record) != expected_keys:
        raise State3WorkflowError("administrative receipt keys are incomplete or unexpected")
    try:
        validate_self_hash(record, "receipt_self_sha256")
    except ValueError as exc:
        raise State3WorkflowError("administrative receipt self-hash mismatch") from exc
    if (
        record["schema"] != ADMINISTRATIVE_RECEIPT_SCHEMA
        or record["state3_authorization_file_sha256"] != authorization_file_sha256
        or record["implementation_commit"] != implementation_commit
        or record["implementation_tree"] != implementation_tree
        or record["implementation_parent_commit"] != implementation_parent_commit
        or record["state3_execution_latch_reached"] is not False
        or record["state3_authority_consumed"] is not False
    ):
        raise State3WorkflowError("administrative receipt identity or authority flags mismatch")
    if not isinstance(record["interpreter"], Mapping) or set(record["interpreter"]) != {
        "filename", "sha256", "size"
    }:
        raise State3WorkflowError("administrative receipt interpreter binding is invalid")
    if record["interpreter"]["filename"] != "python.exe":
        raise State3WorkflowError("administrative receipt interpreter filename is invalid")
    _sha256(record["interpreter"]["sha256"], name="interpreter.sha256")
    _nonnegative(record["interpreter"]["size"], name="interpreter.size")
    state3_audit._validate_disk_estimate_metadata(record["disk_estimate"])
    for name in (
        "failure_inventory_commitment", "policy_commitment", "source_blob_commitment"
    ):
        binding = record[name]
        if not isinstance(binding, Mapping) or set(binding) != {"sha256", "self_sha256"}:
            raise State3WorkflowError(f"{name} is invalid")
        _sha256(binding["sha256"], name=f"{name}.sha256")
        _sha256(binding["self_sha256"], name=f"{name}.self_sha256")
    _validate_output_plan(record["output_plan"])
    attempt_paths = record["storage_attempt_relative_paths"]
    if not isinstance(attempt_paths, list) or not all(
        isinstance(value, str) for value in attempt_paths
    ):
        raise State3WorkflowError("storage attempt paths are not a string list")
    try:
        normalized_attempts = storage._validate_attempt_paths(
            attempt_paths, PurePosixPath(storage.DESTINATION_RELATIVE.rstrip("/"))
        )
    except storage.StoragePreflightError as exc:
        raise State3WorkflowError("storage attempt paths are invalid") from exc
    if [path.as_posix() for path in normalized_attempts] != attempt_paths:
        raise State3WorkflowError("storage attempt paths are not canonical")


def administrative_preflight(
    arguments: argparse.Namespace,
    *,
    repository_root: str | Path | None = None,
    file_binding_reader: FileBindingReader = _default_file_binding,
    git_record_reader: GitRecordReader = _capture_git_record,
    source_record_builder: SourceRecordBuilder = _build_source_record,
    failure_binding_validator: FailureBindingValidator = (
        state3_audit.verify_failure_inventory_physical_bindings
    ),
    publisher: RecordPublisher = _publish_record_default,
    interpreter_validator: Callable[[Path], None] = _require_exact_interpreter,
    enforce_import_boundary: bool = True,
) -> dict[str, Any]:
    """Build the two deterministic, metadata-only administrative artifacts."""

    if enforce_import_boundary:
        _assert_parent_import_boundary()
    root = Path.cwd().resolve(strict=True) if repository_root is None else Path(
        repository_root
    ).resolve(strict=True)
    _sha256(arguments.authorization_file_sha256, name="authorization_file_sha256")
    _git_oid(arguments.implementation_commit, name="implementation_commit")
    _git_oid(arguments.implementation_tree, name="implementation_tree")
    _git_oid(arguments.expected_parent_commit, name="expected_parent_commit")
    interpreter_validator(root)
    git_record = dict(git_record_reader(
        root, arguments.implementation_commit, arguments.implementation_tree,
        arguments.expected_parent_commit,
    ))
    external = _external_bindings(arguments, reader=file_binding_reader)
    disk = _disk_estimate(arguments)
    policy = _thread_policy_record(
        authorization_file_sha256=arguments.authorization_file_sha256,
        implementation_commit=arguments.implementation_commit,
        implementation_tree=arguments.implementation_tree,
        openblas_default_num_threads=arguments.openblas_default_num_threads,
    )
    failure_inventory = state3_audit.build_failure_inventory_record(
        state3_audit.expected_failure_inventory_rows()
    )
    failure_binding_validator(root, failure_inventory)
    source = dict(source_record_builder(
        root=root,
        authorization_file_sha256=arguments.authorization_file_sha256,
        implementation_commit=arguments.implementation_commit,
        implementation_tree=arguments.implementation_tree,
        implementation_parent_commit=arguments.expected_parent_commit,
        git_before=git_record,
        git_after=git_record,
    ))
    state3_audit.validate_source_blob_verification_record(source)
    plan = _output_plan(arguments.run_directory, arguments.archive_filename)
    try:
        attempt_paths = [
            path.as_posix() for path in storage._validate_attempt_paths(
                arguments.attempt_relative_path,
                PurePosixPath(storage.DESTINATION_RELATIVE.rstrip("/")),
            )
        ]
    except storage.StoragePreflightError as exc:
        raise State3WorkflowError("storage attempt paths are invalid") from exc
    interpreter = root / ".venv" / "Scripts" / "python.exe"
    interpreter_size, interpreter_sha256 = file_binding_reader(interpreter)
    receipt = with_self_hash(
        {
            "disk_estimate": disk,
            "external_bindings": external,
            "failure_inventory_commitment": {
                "self_sha256": failure_inventory["record_self_sha256"],
                "sha256": hashlib.sha256(canonical_json_bytes(failure_inventory)).hexdigest(),
            },
            "implementation_commit": arguments.implementation_commit,
            "implementation_parent_commit": arguments.expected_parent_commit,
            "implementation_tree": arguments.implementation_tree,
            "interpreter": {
                "filename": interpreter.name,
                "sha256": interpreter_sha256,
                "size": interpreter_size,
            },
            "output_plan": plan,
            "policy_commitment": {
                "self_sha256": policy["policy_self_sha256"],
                "sha256": hashlib.sha256(canonical_json_bytes(policy)).hexdigest(),
            },
            "schema": ADMINISTRATIVE_RECEIPT_SCHEMA,
            "source_blob_commitment": {
                "self_sha256": source["record_self_sha256"],
                "sha256": hashlib.sha256(canonical_json_bytes(source)).hexdigest(),
            },
            "state3_authority_consumed": False,
            "state3_authorization_file_sha256": arguments.authorization_file_sha256,
            "state3_execution_latch_reached": False,
            "storage_attempt_relative_paths": attempt_paths,
        },
        "receipt_self_sha256",
    )
    _validate_receipt(
        receipt,
        authorization_file_sha256=arguments.authorization_file_sha256,
        implementation_commit=arguments.implementation_commit,
        implementation_tree=arguments.implementation_tree,
        implementation_parent_commit=arguments.expected_parent_commit,
    )
    policy_path = _resolved_artifact_path(
        root, plan["additional_thread_policy"], name="additional_thread_policy"
    )
    receipt_path = _resolved_artifact_path(
        root, plan["administrative_preflight_receipt"],
        name="administrative_preflight_receipt",
    )
    policy_data, receipt_data = _publish_deterministic_pair(
        policy_path, policy, receipt_path, receipt, publisher=publisher
    )
    if enforce_import_boundary:
        _assert_parent_import_boundary()
    return {
        "administrative_preflight_receipt_path": plan[
            "administrative_preflight_receipt"
        ],
        "administrative_preflight_receipt_sha256": hashlib.sha256(
            receipt_data
        ).hexdigest(),
        "additional_thread_policy_path": plan["additional_thread_policy"],
        "additional_thread_policy_sha256": hashlib.sha256(policy_data).hexdigest(),
        "state3_authority_consumed": False,
        "state3_execution_latch_reached": False,
    }


def _strict_completed_json(
    result: subprocess.CompletedProcess[str], *, name: str,
    validator: Callable[[Mapping[str, Any]], None] | None = None,
    self_hash_field: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    if result.returncode != 0:
        raise State3WorkflowError(f"{name} process returned nonzero")
    if result.stderr != "":
        raise State3WorkflowError(f"{name} process wrote stderr")
    if not isinstance(result.stdout, str):
        raise State3WorkflowError(f"{name} stdout was not strict UTF-8 text")
    if not result.stdout or result.stdout.endswith(("\r", "\n")):
        raise State3WorkflowError(f"{name} stdout is empty or newline-terminated")
    try:
        data = result.stdout.encode("utf-8", errors="strict")
        record = strict_canonical_json_loads(data)
        if self_hash_field is not None:
            validate_self_hash(record, self_hash_field)
        if validator is not None:
            validator(record)
    except (UnicodeError, ValueError) as exc:
        raise State3WorkflowError(f"{name} stdout is not one valid canonical record") from exc
    if canonical_json_bytes(record) != data:
        raise State3WorkflowError(f"{name} stdout differs from canonical bytes")
    return record, data


def _publish_checked_record(
    root: Path, relative: str, record: Mapping[str, Any], *,
    publisher: RecordPublisher,
) -> bytes:
    path = _resolved_artifact_path(root, relative, name="workflow output")
    if path.is_symlink() or path.exists():
        raise State3WorkflowError("workflow output already exists")
    expected = canonical_json_bytes(record)
    observed = publisher(path, record)
    if observed != expected:
        raise State3WorkflowError("workflow publisher returned unexpected bytes")
    _validate_existing_expected(path, expected)
    return expected


def _require_receipt_external_bindings(
    arguments: argparse.Namespace, receipt: Mapping[str, Any], *,
    reader: FileBindingReader,
) -> dict[str, dict[str, Any]]:
    live = _external_bindings(arguments, reader=reader)
    if canonical_json_bytes(live) != canonical_json_bytes(receipt["external_bindings"]):
        raise State3WorkflowError("live external bindings differ from administrative receipt")
    return live


def _storage_command(
    *, authorization_file_sha256: str, implementation_commit: str,
    implementation_tree: str, expected_parent_commit: str,
    disk_estimate: Mapping[str, Any], output: str,
    attempt_relative_paths: Sequence[str],
) -> tuple[str, ...]:
    command = [
        sys.executable, "-B", "-m", "operatormark.corpus.storage",
        "--protocol-file-sha256", PROTOCOL_SHA256,
        "--authorization-file-sha256", authorization_file_sha256,
        "--implementation-commit", implementation_commit,
        "--implementation-tree", implementation_tree,
        "--expected-parent-commit", expected_parent_commit,
        "--estimated-accepted-bytes", str(disk_estimate["estimated_accepted_bytes"]),
        "--projected-audit-bytes", str(disk_estimate["projected_audit_bytes"]),
    ]
    for relative in attempt_relative_paths:
        command.extend(("--attempt-relative-path", relative))
    command.extend(("--output", output))
    return tuple(command)


def _run_storage_preflight(
    *, root: Path, relative_output: str, arguments: argparse.Namespace,
    receipt: Mapping[str, Any], attempt_relative_paths: Sequence[str],
    process_runner: ProcessRunner,
) -> dict[str, Any]:
    result = process_runner(
        _storage_command(
            authorization_file_sha256=arguments.authorization_file_sha256,
            implementation_commit=arguments.implementation_commit,
            implementation_tree=arguments.implementation_tree,
            expected_parent_commit=arguments.expected_parent_commit,
            disk_estimate=receipt["disk_estimate"], output=relative_output,
            attempt_relative_paths=attempt_relative_paths,
        ),
        root,
    )
    record, _ = _strict_completed_json(result, name="storage preflight receipt")
    expected_receipt_keys = {
        "authorization_file_sha256", "implementation_commit", "implementation_tree",
        "output_path", "output_sha256", "output_size", "record_self_sha256", "schema",
    }
    if (
        set(record) != expected_receipt_keys
        or record["output_path"] != relative_output
        or record["authorization_file_sha256"] != arguments.authorization_file_sha256
        or record["implementation_commit"] != arguments.implementation_commit
        or record["implementation_tree"] != arguments.implementation_tree
    ):
        raise State3WorkflowError("storage preflight receipt binding mismatch")
    output_path = _resolved_artifact_path(root, relative_output, name="storage output")
    storage_record = _load_canonical_record(output_path, self_hash_field="record_self_sha256")
    storage.validate_storage_preflight_record(storage_record)
    data = canonical_json_bytes(storage_record)
    if (
        hashlib.sha256(data).hexdigest() != record["output_sha256"]
        or len(data) != record["output_size"]
        or storage_record["record_self_sha256"] != record["record_self_sha256"]
    ):
        raise State3WorkflowError("storage output differs from its process receipt")
    return storage_record


def _validate_environment_bindings(
    record: Mapping[str, Any], *, authorization_file_sha256: str,
    implementation_commit: str, implementation_tree: str,
) -> None:
    state3_audit.validate_environment_runtime_record(record)
    if (
        record["state3_authorization_file_sha256"] != authorization_file_sha256
        or record["implementation_commit"] != implementation_commit
        or record["implementation_tree"] != implementation_tree
        or tuple(sorted(record["parent_environment"].items()))
        != tuple(sorted(dict(launcher.MANDATORY_PARENT_ENVIRONMENT | dict(
            OPENBLAS_DEFAULT_POLICY
        )).items()))
    ):
        raise State3WorkflowError("environment/runtime child binding mismatch")


def _validate_measurement_bindings(
    record: Mapping[str, Any], *, authorization_file_sha256: str,
    implementation_commit: str, implementation_tree: str,
) -> None:
    state3_audit.validate_measurement_bundle_metadata(record)
    if (
        record["state3_authorization_file_sha256"] != authorization_file_sha256
        or record["implementation_commit"] != implementation_commit
        or record["implementation_tree"] != implementation_tree
    ):
        raise State3WorkflowError("measurement child binding mismatch")


def _validate_resume(
    *, root: Path, plan: Mapping[str, str], attempt_number: int,
    resume_path: str | None, resume_sha256: str | None,
    receipt_sha256: str, explicit_continuation: bool,
) -> None:
    if attempt_number < 1:
        raise State3WorkflowError("attempt_number must be positive")
    if attempt_number == 1:
        if resume_path is not None or resume_sha256 is not None or explicit_continuation:
            raise State3WorkflowError("first execution attempt cannot carry resume arguments")
        return
    if not explicit_continuation or resume_path is None or resume_sha256 is None:
        raise State3WorkflowError(
            "pre-latch continuation requires explicit instruction and path/hash binding"
        )
    expected_path = plan["pre_latch_state_pattern"].format(attempt=attempt_number - 1)
    if resume_path != expected_path:
        raise State3WorkflowError("resume record is not the immediately prior attempt")
    _sha256(resume_sha256, name="resume_pre_latch_state_sha256")
    record_path = _resolved_artifact_path(root, resume_path, name="resume record")
    if record_path.is_symlink() or not record_path.is_file():
        raise State3WorkflowError("resume record is not one ordinary file")
    if file_sha256(record_path) != resume_sha256:
        raise State3WorkflowError("resume record ordinary SHA-256 mismatch")
    record = _load_canonical_record(record_path, self_hash_field="record_self_sha256")
    if set(record) != {
        "administrative_preflight_receipt_sha256", "attempt_number", "error_class",
        "failure_stage", "raw_process_output_embedded", "record_self_sha256", "schema",
        "state3_authority_consumed", "state3_execution_latch_reached",
    }:
        raise State3WorkflowError("resume record keys are incomplete or unexpected")
    if (
        record.get("schema") != PRE_LATCH_STATE_SCHEMA
        or record.get("attempt_number") != attempt_number - 1
        or record.get("administrative_preflight_receipt_sha256") != receipt_sha256
        or record.get("state3_execution_latch_reached") is not False
        or record.get("state3_authority_consumed") is not False
        or record.get("raw_process_output_embedded") is not False
        or not isinstance(record.get("failure_stage"), str)
        or not record.get("failure_stage")
        or not isinstance(record.get("error_class"), str)
        or not record.get("error_class")
    ):
        raise State3WorkflowError("resume record does not prove a pre-latch stop")


def _pre_latch_record(
    *, attempt_number: int, receipt_sha256: str, stage: str,
    error: BaseException,
) -> dict[str, Any]:
    return with_self_hash(
        {
            "administrative_preflight_receipt_sha256": receipt_sha256,
            "attempt_number": attempt_number,
            "error_class": type(error).__name__,
            "failure_stage": stage,
            "raw_process_output_embedded": False,
            "schema": PRE_LATCH_STATE_SCHEMA,
            "state3_authority_consumed": False,
            "state3_execution_latch_reached": False,
        },
        "record_self_sha256",
    )


def _finalizer_command(
    *, arguments: argparse.Namespace, plan: Mapping[str, str],
    storage_before: str,
) -> tuple[str, ...]:
    return (
        sys.executable, "-B", "-m", "operatormark.corpus.state3_audit",
        "--authorization-file-sha256", arguments.authorization_file_sha256,
        "--implementation-commit", arguments.implementation_commit,
        "--implementation-tree", arguments.implementation_tree,
        "--preflight-bindings", plan["compute_feasibility_preflight_bindings"],
        "--measurement-bundle", plan["measurement_bundle"],
        "--storage-before", storage_before,
        "--storage-after", plan["storage_preflight_after"],
        "--environment-runtime", plan["environment_runtime"],
        "--source-blob-verification", plan["source_blob_verification"],
        "--failure-inventory", plan["failure_inventory"],
        "--prohibited-action-counts", plan["prohibited_action_counts"],
        "--evidence-directory", plan["audit_evidence_directory"],
        "--destination-zip", plan["archive"],
    )


def _verifier_command(
    *, arguments: argparse.Namespace, plan: Mapping[str, str],
) -> tuple[str, ...]:
    return (
        sys.executable, "-B", "-m", "operatormark.corpus.state3_audit_verify",
        plan["archive"], "--expected-archive-filename", Path(plan["archive"]).name,
        "--authorization-file-sha256", arguments.authorization_file_sha256,
        "--implementation-commit", arguments.implementation_commit,
        "--implementation-tree", arguments.implementation_tree,
    )


def execute(
    arguments: argparse.Namespace,
    *,
    repository_root: str | Path | None = None,
    file_binding_reader: FileBindingReader = _default_file_binding,
    git_record_reader: GitRecordReader = _capture_git_record,
    source_record_builder: SourceRecordBuilder = _build_source_record,
    failure_binding_validator: FailureBindingValidator = (
        state3_audit.verify_failure_inventory_physical_bindings
    ),
    process_runner: ProcessRunner = _default_process_runner,
    child_launcher: ChildLauncher = launcher.launch_fresh_child,
    publisher: RecordPublisher = _publish_record_default,
    interpreter_validator: Callable[[Path], None] = _require_exact_interpreter,
    enforce_import_boundary: bool = True,
) -> dict[str, Any]:
    """Run the one-shot phase; tests must inject every worker and publisher."""

    if enforce_import_boundary:
        _assert_parent_import_boundary()
    root = Path.cwd().resolve(strict=True) if repository_root is None else Path(
        repository_root
    ).resolve(strict=True)
    _sha256(arguments.authorization_file_sha256, name="authorization_file_sha256")
    _git_oid(arguments.implementation_commit, name="implementation_commit")
    _git_oid(arguments.implementation_tree, name="implementation_tree")
    _git_oid(arguments.expected_parent_commit, name="expected_parent_commit")
    receipt_path = _resolved_artifact_path(
        root, arguments.administrative_preflight_receipt,
        name="administrative_preflight_receipt",
    )
    _sha256(
        arguments.administrative_preflight_receipt_sha256,
        name="administrative_preflight_receipt_sha256",
    )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise State3WorkflowError("administrative receipt is not one ordinary file")
    if file_sha256(receipt_path) != arguments.administrative_preflight_receipt_sha256:
        raise State3WorkflowError("administrative receipt ordinary SHA-256 mismatch")
    receipt = _load_canonical_record(receipt_path, self_hash_field="receipt_self_sha256")
    _validate_receipt(
        receipt,
        authorization_file_sha256=arguments.authorization_file_sha256,
        implementation_commit=arguments.implementation_commit,
        implementation_tree=arguments.implementation_tree,
        implementation_parent_commit=arguments.expected_parent_commit,
    )
    plan = _validate_output_plan(receipt["output_plan"])
    if arguments.administrative_preflight_receipt != plan[
        "administrative_preflight_receipt"
    ]:
        raise State3WorkflowError("receipt path differs from its fixed output plan")
    latch_path = _resolved_artifact_path(
        root, plan["state3_execution_latch"], name="state3_execution_latch"
    )
    completion_path = _resolved_artifact_path(
        root, plan["workflow_completion"], name="workflow_completion"
    )
    if (
        latch_path.is_symlink() or latch_path.exists()
        or completion_path.is_symlink() or completion_path.exists()
    ):
        raise State3WorkflowError("a latch or terminal completion already prohibits execution")
    _validate_resume(
        root=root, plan=plan, attempt_number=arguments.attempt_number,
        resume_path=arguments.resume_pre_latch_state,
        resume_sha256=arguments.resume_pre_latch_state_sha256,
        receipt_sha256=arguments.administrative_preflight_receipt_sha256,
        explicit_continuation=arguments.explicit_pre_latch_continuation,
    )
    storage_before_relative = plan["storage_preflight_before_pattern"].format(
        attempt=arguments.attempt_number
    )
    pre_latch_relative = plan["pre_latch_state_pattern"].format(
        attempt=arguments.attempt_number
    )
    storage_before_path = _resolved_artifact_path(
        root, storage_before_relative, name="storage_preflight_before"
    )
    pre_latch_path = _resolved_artifact_path(
        root, pre_latch_relative, name="pre_latch_state"
    )
    if (
        storage_before_path.is_symlink() or storage_before_path.exists()
        or pre_latch_path.is_symlink() or pre_latch_path.exists()
    ):
        raise State3WorkflowError("attempt-specific output path already exists")

    latch_reached = False
    stage = "pre_latch_live_binding_validation"
    try:
        interpreter_validator(root)
        live_interpreter_size, live_interpreter_sha256 = file_binding_reader(
            root / ".venv" / "Scripts" / "python.exe"
        )
        if (
            live_interpreter_size != receipt["interpreter"]["size"]
            or live_interpreter_sha256 != receipt["interpreter"]["sha256"]
        ):
            raise State3WorkflowError(
                "live interpreter differs from administrative receipt"
            )
        _require_receipt_external_bindings(
            arguments, receipt, reader=file_binding_reader
        )
        git_before = dict(git_record_reader(
            root, arguments.implementation_commit, arguments.implementation_tree,
            arguments.expected_parent_commit,
        ))
        policy_path = _resolved_artifact_path(
            root, plan["additional_thread_policy"], name="additional_thread_policy"
        )
        policy = _load_canonical_record(
            policy_path, self_hash_field="policy_self_sha256"
        )
        normalized = launcher.load_approved_additional_thread_policy(
            policy_path,
            authorization_file_sha256=arguments.authorization_file_sha256,
            implementation_commit=arguments.implementation_commit,
            implementation_tree=arguments.implementation_tree,
        )
        if normalized != OPENBLAS_DEFAULT_POLICY:
            raise State3WorkflowError(
                "launcher normalized a non-frozen additional policy"
            )
        _validate_thread_policy(
            policy,
            authorization_file_sha256=arguments.authorization_file_sha256,
            implementation_commit=arguments.implementation_commit,
            implementation_tree=arguments.implementation_tree,
        )
        policy_data = canonical_json_bytes(policy)
        if hashlib.sha256(policy_data).hexdigest() != receipt[
            "policy_commitment"
        ]["sha256"]:
            raise State3WorkflowError(
                "additional policy differs from receipt commitment"
            )
        stage = "storage_preflight_before"
        storage_before = _run_storage_preflight(
            root=root, relative_output=storage_before_relative,
            arguments=arguments, receipt=receipt,
            attempt_relative_paths=tuple(receipt["storage_attempt_relative_paths"]),
            process_runner=process_runner,
        )
        stage = "environment_process_creation"
        environment_spec = launcher.ParentLaunchSpec(
            state="state3", action="state3-environment-probe",
            authorization_file_sha256=arguments.authorization_file_sha256,
            implementation_commit=arguments.implementation_commit,
            implementation_tree=arguments.implementation_tree,
            additional_environment=OPENBLAS_DEFAULT_POLICY,
        )

        def publish_latch() -> None:
            nonlocal latch_reached
            latch_reached = True
            latch = with_self_hash(
                {
                    "action": "state3-environment-probe",
                    "administrative_preflight_receipt_sha256": (
                        arguments.administrative_preflight_receipt_sha256
                    ),
                    "attempt_number": arguments.attempt_number,
                    "schema": LATCH_SCHEMA,
                    "state3_authority_consumed": True,
                    "state3_execution_latch_reached": True,
                },
                "record_self_sha256",
            )
            _publish_checked_record(
                root, plan["state3_execution_latch"], latch, publisher=publisher
            )

        environment_result = child_launcher(
            environment_spec, (), check=False, on_child_started=publish_latch
        )
        if not latch_reached:
            raise State3WorkflowError(
                "environment launcher returned without reaching the execution latch"
            )
    except BaseException as exc:
        if not latch_reached:
            pre_latch = _pre_latch_record(
                attempt_number=arguments.attempt_number,
                receipt_sha256=arguments.administrative_preflight_receipt_sha256,
                stage=stage, error=exc,
            )
            _publish_checked_record(
                root, pre_latch_relative, pre_latch, publisher=publisher
            )
        raise

    stage = "environment_runtime_publication"
    environment, _ = _strict_completed_json(
        environment_result, name="environment/runtime child",
        validator=lambda record: _validate_environment_bindings(
            record,
            authorization_file_sha256=arguments.authorization_file_sha256,
            implementation_commit=arguments.implementation_commit,
            implementation_tree=arguments.implementation_tree,
        ),
        self_hash_field="record_self_sha256",
    )
    environment_data = _publish_checked_record(
        root, plan["environment_runtime"], environment, publisher=publisher
    )
    preflight = state3_audit.build_compute_feasibility_preflight_bindings_record(
        protocol_file_sha256=PROTOCOL_SHA256,
        state3_authorization_file_sha256=arguments.authorization_file_sha256,
        implementation_commit=arguments.implementation_commit,
        implementation_tree=arguments.implementation_tree,
        launcher_file_sha256=environment["launcher_file_sha256"],
        runtime_inspector_file_sha256=environment["runtime_inspector_file_sha256"],
        environment_record_sha256=hashlib.sha256(environment_data).hexdigest(),
        storage_preflight_before_file_sha256=hashlib.sha256(
            canonical_json_bytes(storage_before)
        ).hexdigest(),
        disk_estimate=receipt["disk_estimate"], git_before=git_before,
    )
    _publish_checked_record(
        root, plan["compute_feasibility_preflight_bindings"], preflight,
        publisher=publisher,
    )
    measurement_spec = launcher.ParentLaunchSpec(
        state="state3", action="state3-microbenchmark",
        authorization_file_sha256=arguments.authorization_file_sha256,
        implementation_commit=arguments.implementation_commit,
        implementation_tree=arguments.implementation_tree,
        additional_environment=OPENBLAS_DEFAULT_POLICY,
    )
    measurement_result = child_launcher(
        measurement_spec, (plan["compute_feasibility_preflight_bindings"],),
        check=False,
    )
    measurement, _ = _strict_completed_json(
        measurement_result, name="State 3 microbenchmark child",
        validator=lambda record: _validate_measurement_bindings(
            record,
            authorization_file_sha256=arguments.authorization_file_sha256,
            implementation_commit=arguments.implementation_commit,
            implementation_tree=arguments.implementation_tree,
        ),
        self_hash_field="measurement_bundle_self_sha256",
    )
    _publish_checked_record(
        root, plan["measurement_bundle"], measurement, publisher=publisher
    )
    _run_storage_preflight(
        root=root, relative_output=plan["storage_preflight_after"],
        arguments=arguments, receipt=receipt,
        attempt_relative_paths=tuple(receipt["storage_attempt_relative_paths"]),
        process_runner=process_runner,
    )
    git_after = dict(git_record_reader(
        root, arguments.implementation_commit, arguments.implementation_tree,
        arguments.expected_parent_commit,
    ))
    source = dict(source_record_builder(
        root=root,
        authorization_file_sha256=arguments.authorization_file_sha256,
        implementation_commit=arguments.implementation_commit,
        implementation_tree=arguments.implementation_tree,
        implementation_parent_commit=arguments.expected_parent_commit,
        git_before=git_before, git_after=git_after,
    ))
    state3_audit.validate_source_blob_verification_record(source)
    source_data = canonical_json_bytes(source)
    if hashlib.sha256(source_data).hexdigest() != receipt[
        "source_blob_commitment"
    ]["sha256"]:
        raise State3WorkflowError("final source binding differs from preflight commitment")
    _publish_checked_record(
        root, plan["source_blob_verification"], source, publisher=publisher
    )
    failure_inventory = state3_audit.build_failure_inventory_record(
        state3_audit.expected_failure_inventory_rows()
    )
    failure_binding_validator(root, failure_inventory)
    failure_data = canonical_json_bytes(failure_inventory)
    if hashlib.sha256(failure_data).hexdigest() != receipt[
        "failure_inventory_commitment"
    ]["sha256"]:
        raise State3WorkflowError("final failure inventory differs from preflight commitment")
    _publish_checked_record(
        root, plan["failure_inventory"], failure_inventory, publisher=publisher
    )
    prohibited = state3_audit.build_prohibited_action_counts_record(
        {name: 0 for name in state3_audit.PROHIBITED_COUNT_NAMES}
    )
    _publish_checked_record(
        root, plan["prohibited_action_counts"], prohibited, publisher=publisher
    )
    finalizer_result = process_runner(
        _finalizer_command(
            arguments=arguments, plan=plan, storage_before=storage_before_relative
        ),
        root,
    )
    finalizer_receipt, _ = _strict_completed_json(
        finalizer_result, name="State 3 audit finalizer"
    )
    if (
        finalizer_receipt.get("destination_zip") != plan["archive"]
        or finalizer_receipt.get("authorization_file_sha256")
        != arguments.authorization_file_sha256
        or finalizer_receipt.get("implementation_commit")
        != arguments.implementation_commit
        or finalizer_receipt.get("implementation_tree") != arguments.implementation_tree
        or finalizer_receipt.get("member_order")
        != list(state3_audit.STATE3_AUDIT_MEMBER_ORDER)
    ):
        raise State3WorkflowError("audit-finalizer receipt binding mismatch")
    verifier_result = process_runner(
        _verifier_command(arguments=arguments, plan=plan), root
    )
    verifier_receipt, _ = _strict_completed_json(
        verifier_result, name="independent State 3 verifier"
    )
    if (
        verifier_receipt.get("outcome") != "verified"
        or verifier_receipt.get("archive_sha256")
        != finalizer_receipt.get("archive_sha256")
        or verifier_receipt.get("member_order")
        != list(state3_audit.STATE3_AUDIT_MEMBER_ORDER)
    ):
        raise State3WorkflowError("independent verification receipt mismatch")
    verified = with_self_hash(
        {
            "archive_sha256": verifier_receipt["archive_sha256"],
            "archive_size": verifier_receipt["archive_size"],
            "authorization_file_sha256": arguments.authorization_file_sha256,
            "implementation_commit": arguments.implementation_commit,
            "implementation_tree": arguments.implementation_tree,
            "member_order": verifier_receipt["member_order"],
            "outcome": "verified",
            "schema": "operatormark.darcy_branch_c_v1.state3_independent_verification.v1",
        },
        "record_self_sha256",
    )
    verified_data = _publish_checked_record(
        root, plan["independent_verification"], verified, publisher=publisher
    )
    completion = with_self_hash(
        {
            "archive_sha256": verifier_receipt["archive_sha256"],
            "attempt_number": arguments.attempt_number,
            "independent_verification_sha256": hashlib.sha256(
                verified_data
            ).hexdigest(),
            "schema": COMPLETION_SCHEMA,
            "state3_authority_consumed": True,
            "state3_execution_latch_reached": True,
            "state4_authorized": False,
        },
        "record_self_sha256",
    )
    _publish_checked_record(
        root, plan["workflow_completion"], completion, publisher=publisher
    )
    if enforce_import_boundary:
        _assert_parent_import_boundary()
    return completion


def _add_common_binding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--authorization-file", required=True)
    parser.add_argument("--authorization-file-size", required=True, type=int)
    parser.add_argument("--authorization-file-sha256", required=True)
    parser.add_argument("--protocol-file", required=True)
    parser.add_argument("--protocol-file-size", required=True, type=int)
    parser.add_argument("--gate9-audit-file", required=True)
    parser.add_argument("--gate9-audit-file-size", required=True, type=int)
    parser.add_argument("--state2-corrective-audit-file", required=True)
    parser.add_argument("--state2-entrypoints-audit-file", required=True)
    parser.add_argument("--revoked-state3-authorization-file", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--implementation-tree", required=True)
    parser.add_argument("--expected-parent-commit", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Committed two-phase OperatorMark State 3 workflow"
    )
    subparsers = parser.add_subparsers(dest="workflow_phase", required=True)
    administrative = subparsers.add_parser("administrative-preflight")
    _add_common_binding_arguments(administrative)
    administrative.add_argument("--run-directory", required=True)
    administrative.add_argument("--archive-filename", required=True)
    administrative.add_argument(
        "--openblas-default-num-threads", required=True, default=None,
        action=_UniqueValue,
    )
    administrative.add_argument("--projected-npy-header-bytes", required=True, type=int)
    administrative.add_argument("--projected-jsonl-bytes", required=True, type=int)
    administrative.add_argument(
        "--projected-manifest-and-audit-bytes", required=True, type=int
    )
    administrative.add_argument("--controls-only-bytes", required=True, type=int)
    administrative.add_argument("--projected-audit-bytes", required=True, type=int)
    administrative.add_argument("--attempt-relative-path", action="append", default=[])

    execution = subparsers.add_parser("execute")
    _add_common_binding_arguments(execution)
    execution.add_argument("--administrative-preflight-receipt", required=True)
    execution.add_argument("--administrative-preflight-receipt-sha256", required=True)
    execution.add_argument("--attempt-number", required=True, type=int)
    execution.add_argument("--resume-pre-latch-state")
    execution.add_argument("--resume-pre-latch-state-sha256")
    execution.add_argument("--explicit-pre-latch-continuation", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    administrative_runner: Callable[[argparse.Namespace], Mapping[str, Any]] | None = None,
    execution_runner: Callable[[argparse.Namespace], Mapping[str, Any]] | None = None,
) -> int:
    """Dispatch exactly one of the two committed workflow phases."""

    arguments = _parser().parse_args(argv)
    if arguments.workflow_phase == "administrative-preflight":
        result = (
            administrative_preflight(arguments)
            if administrative_runner is None else administrative_runner(arguments)
        )
    elif arguments.workflow_phase == "execute":
        result = execute(arguments) if execution_runner is None else execution_runner(arguments)
    else:  # pragma: no cover - argparse makes this unreachable
        raise State3WorkflowError("unknown workflow phase")
    sys.stdout.write(canonical_json_bytes(dict(result)).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
