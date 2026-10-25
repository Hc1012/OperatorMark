"""Strict metadata-only State 3 audit records and deterministic packaging.

The command-line entry point consumes already-created canonical metadata.  It
cannot launch a numerical child, fixture, microbenchmark, or solver.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .canonical import (
    canonical_json_bytes,
    file_sha256,
    strict_canonical_json_loads,
    validate_self_hash,
    with_self_hash,
    write_canonical_json_exclusive,
)


PROTOCOL_SHA256: Final = "a38ef79b6c4bdccf53f599fa08f77e48a111afbf0b2783ad5733bda99996599a"
SOLVER_SPEC_BASELINE_COMMIT: Final = "e16456395cfe67b73b9d298fd0491cf9524d798a"
SOLVER_SPEC_BASELINE_TREE: Final = "81f40e24102e51e0da3ce65dafd769a7678565fd"
EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()
STATE3_STATE: Final = "state3_pre_generation_audit_no_corpus_controls"
STATE3_OUTCOME: Final = "complete_pending_independent_review_and_separate_state4_authorization"
STATE3_CLASSIFICATION: Final = "private_confidential_metadata_only_external_sharing_prohibited"

STATE3_AUDIT_MEMBER_ORDER: Final = (
    "state3_audit.json",
    "environment_runtime.json",
    "storage_preflight_before.json",
    "compute_feasibility_report.json",
    "storage_preflight_after.json",
    "source_blob_verification.json",
    "failure_inventory.json",
    "prohibited_action_counts.json",
    "member_hash_manifest.json",
)
STATE3_MEMBER_SCHEMAS: Final = {
    "state3_audit.json": "operatormark.darcy_branch_c_v1.state3_audit.v1",
    "environment_runtime.json": "operatormark.darcy_branch_c_v1.state3_environment_runtime.v1",
    "storage_preflight_before.json": "operatormark.darcy_branch_c_v1.private_storage_preflight.v1",
    "compute_feasibility_report.json": "operatormark.darcy_branch_c_v1.compute_feasibility.v2",
    "storage_preflight_after.json": "operatormark.darcy_branch_c_v1.private_storage_preflight.v1",
    "source_blob_verification.json": "operatormark.darcy_branch_c_v1.state3_source_blob_verification.v1",
    "failure_inventory.json": "operatormark.darcy_branch_c_v1.state3_failure_inventory.v2",
    "prohibited_action_counts.json": "operatormark.darcy_branch_c_v1.state3_prohibited_action_counts.v1",
}
MANIFEST_SCHEMA: Final = "operatormark.darcy_branch_c_v1.state3_audit_member_manifest.v1"
FIXED_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
FIXED_ZIP_EXTERNAL_ATTR: Final = 0x20

MANDATORY_PARENT_ENVIRONMENT: Final = {
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

GIT_RECORD_KEYS: Final = {
    "branch", "head", "tree", "parents",
    "status_short_untracked_all",
    "tracked_diff_sha256", "tracked_diff_size_bytes",
    "staged_diff_sha256", "staged_diff_size_bytes",
    "tracked_changes_empty", "staged_changes_empty", "untracked_inclusive_status_empty",
}
PREFLIGHT_BINDING_KEYS: Final = {
    "schema", "protocol_file_sha256", "state3_authorization_file_sha256",
    "solver_spec_baseline_commit", "solver_spec_baseline_tree", "implementation_commit",
    "implementation_tree", "launcher_file_sha256", "runtime_inspector_file_sha256",
    "environment_record_sha256", "storage_preflight_before_file_sha256", "disk_estimate",
    "git_before", "preflight_bindings_self_sha256",
}
MEASUREMENT_BUNDLE_KEYS: Final = {
    "schema", "state", "protocol_file_sha256", "state3_authorization_file_sha256",
    "implementation_commit", "implementation_tree", "warmup_count_per_fixture",
    "measured_count_per_fixture", "measurement_order", "raw_measurements",
    "all_fixture_gates_passed", "all_rng_state_equalities_passed",
    "forbidden_rng_call_count", "forbidden_corpus_seed_access_count",
    "measurement_bundle_self_sha256",
}
FIXTURE_ORDER: Final = ("constant_half", "vertical_jump", "checkerboard")
MEASURED_COUNT_PER_FIXTURE: Final = 5
CORPUS_SAMPLE_COUNT: Final = 4864
FROZEN_SAFETY_FACTOR: Final = 2.0
RAW_SEVEN_TENSOR_PAYLOAD_BYTES: Final = 420_872_192
TIMING_FIELDS: Final = (
    "tpfa_assembly_ns", "tpfa_solve_ns", "tpfa_diagnostics_ns", "p1_assembly_ns",
    "p1_solve_ns", "p1_diagnostics_ns", "p1_projection_ns", "paired_solver_ns",
    "child_wall_ns",
)
MEMORY_FIELDS: Final = (
    "baseline_rss_bytes", "peak_rss_bytes", "incremental_peak_rss_bytes",
)
SUMMARY_FIELDS: Final = TIMING_FIELDS + MEMORY_FIELDS
FIXTURE_DEFINITIONS: Final = {
    "constant_half": {
        "construction": "deterministic_indexed_assignment_no_rng", "dtype": "<f8",
        "formula": "rho[j,i]=0.5", "shape": [8, 8],
    },
    "vertical_jump": {
        "construction": "deterministic_indexed_assignment_no_rng", "dtype": "<f8",
        "formula": "rho[j,i]=0.0 if i<4 else 1.0", "shape": [8, 8],
    },
    "checkerboard": {
        "construction": "deterministic_indexed_assignment_no_rng", "dtype": "<f8",
        "formula": "rho[j,i]=float((i+j) mod 2)", "shape": [8, 8],
    },
}
RNG_STATE_COMMITMENT_SPECIFICATION: Final = {
    "numpy_legacy_domain_separator": "operatormark-numpy-legacy-rng-state-v1\n",
    "numpy_legacy_canonical_fields": [
        "bit_generator", "key_dtype", "key_shape", "key_c_order_hex", "position",
        "has_gauss", "cached_gaussian_hex",
    ],
    "python_random_domain_separator": "operatormark-python-random-state-v1\n",
    "python_random_canonical_fields": ["version", "internal_state", "gauss_next"],
    "canonical_json_encoding": "utf8_compact_sorted_keys_no_trailing_newline",
    "ordered_checkpoints": [
        "after_base_import", "after_guard_install", "after_project_import", "after_execution",
    ],
    "passive_module_import_increments_forbidden_call_count": False,
}
ENVIRONMENT_RUNTIME_KEYS: Final = {
    "schema", "state", "protocol_file_sha256", "state3_authorization_file_sha256",
    "implementation_commit", "implementation_tree", "python_version", "numpy_version",
    "scipy_version", "operating_system", "architecture", "locale", "parent_environment",
    "parent_environment_commitment", "applicable_additional_thread_variables",
    "launcher_file_sha256", "runtime_inspector_file_sha256", "requirements_lock_path",
    "requirements_lock_git_blob", "requirements_lock_file_sha256", "base_import_closure",
    "base_import_closure_sha256", "numerical_runtime", "numerical_runtime_sha256",
    "all_environment_checks_passed", "all_runtime_checks_passed", "record_self_sha256",
}

FROZEN_BASELINE_REQUIRED_PATHS: Final = (
    "configs/darcy_branch_c_v1.yaml",
    "operatormark/benchmarks/darcy_spec.py",
    "operatormark/evaluation/darcy_cross_solver_tolerance.py",
    "operatormark/evaluation/darcy_solver_heldout.py",
    "operatormark/solvers/darcy_p1.py",
    "operatormark/solvers/darcy_tpfa.py",
    "reports/darcy_branch_c_v1_preregistration.md",
)
FROZEN_REVIEW_COPY_SHA256: Final = {
    "configs/darcy_branch_c_v1.yaml": "b24da08f028c52c2f275a49a07a971efcf42ad63962412ddda81ce81fa6e242d",
    "reports/darcy_branch_c_v1_preregistration.md": "0ae56c37e4f158e6038b80fa2bdbebfb0f72f8ba6911ec51889b6d3321b0ff17",
}
BASELINE_BLOB_KEYS: Final = {
    "path", "baseline_git_blob", "implementation_git_blob", "baseline_file_sha256",
    "implementation_file_sha256", "byte_identical",
}
IMPLEMENTATION_FILE_KEYS: Final = {"path", "git_blob", "file_sha256", "matches_git_blob"}
SOURCE_BLOB_KEYS: Final = {
    "schema", "state", "protocol_file_sha256", "state3_authorization_file_sha256",
    "solver_spec_baseline_commit", "solver_spec_baseline_tree", "implementation_commit",
    "implementation_tree", "implementation_parent_commit", "requirements_lock_path",
    "requirements_lock_git_blob", "requirements_lock_file_sha256", "frozen_baseline_blobs",
    "implementation_files", "all_frozen_baseline_blobs_byte_identical",
    "all_implementation_files_match_git", "git_before", "git_after", "record_self_sha256",
}

FAILURE_ORIGINAL_PREFIX: Final = "artifacts/darcy_branch_c_v1/private_corpus_v1/state3_audits"
FAILURE_DESTINATION_PREFIX: Final = "artifacts/darcy_branch_c_v1/state3_preflight_failures_20260903"
FAILURE_SPECS: Final = (
    ("state3_eighth_retry_preflight_failure_20260903_markdown_path_misinterpretation.json", 3333, "41301182f49b4ef905f9b70889e77e845fb1790cbb14e84cf35691450f4d103f"),
    ("state3_elevation_retry_preflight_failure_20260903_invalid_rg_glob.json", 2581, "5494eff4f7858a50bf67766ab24e31805fcb140b957c321555ed6928f4d9ab11"),
    ("state3_eleventh_retry_preflight_failure_20260903_encoded_command_line_too_long.json", 5538, "5de16b7c9fc8d972cc6729dbc97a9cd59278ef58b18bc140367de5e20be0800f"),
    ("state3_fourth_retry_preflight_failure_20260903_base_interpreter_access_denied.json", 2548, "9810ffe80eda781f8e06d62e0f406fe4f267bf0413b17fdd96671e8a592de367"),
    ("state3_further_retry_preflight_failure_20260903_base_interpreter_access_denied.json", 2367, "95055a44bf2368df2602906c87048556ff06fc67e96dbed90565011497a6e8b6"),
    ("state3_ninth_retry_preflight_failure_20260903_exact_authorization_path_mismatch.json", 3729, "870637ac62c89e75485351bc01de64168febed66f30ebef51cb4093e40d3c175"),
    ("state3_preflight_failure_20260903_python_command_not_found.json", 1472, "77dadacc8e9bae38ef79f27b54feffed2c1422f3df9c4d11b3b7c0284ad95f22"),
    ("state3_retry_preflight_failure_20260903_cim_access_denied.json", 1783, "178cde3681cc1b3863b43eb001fcc519bd108c1a662c242b74741ecf9afbfa46"),
    ("state3_retry_preflight_failure_20260903_unreachable_state3_launcher_action.json", 3278, "172aeab723954aacc68b7a515ee5f958807f20553336350ab6e592e23fea7e72"),
    ("state3_seventh_retry_preflight_failure_20260903_authorization_file_missing.json", 3569, "bbfb9108ad07970caf90d05307f4608eac3c5ed654e50c78ec54034a2123c8be"),
    ("state3_sixth_retry_preflight_failure_20260903_powershell_parser_error.json", 3139, "f3bfc67dfea046b9e69ffe6acb768248db3911c09cf2a6a559e23c480506f2ed"),
    ("state3_tenth_retry_preflight_failure_20260903_psreadline_pty_input_corruption.json", 3944, "979885cb1331f87af3e94f784b68ed8a68169b7b49ec479c9cd7e48c8bfaf0f9"),
)
IN_PLACE_FAILURE_SPECS: Final = (
    (
        "artifacts/state2-corrective-entrypoints-20260904/"
        "state2_corrective_entrypoints_stopped_unauthorized_full_suite.json",
        3755,
        "1d8a8b877e343783324e5b1801f5217831ce7f88aa0e6e5dd244eb69222a51b2",
    ),
    (
        "artifacts/state3-v1.1-20260904/"
        "state3_preflight_failure_unreachable_committed_audit_orchestration.json",
        2965,
        "c319988f00d4f8ca34a1e6440ff61d2399064bf677cd465dc54c45c96aaa0d7e",
    ),
    (
        "artifacts/state3-v1.2-20260904/"
        "state3_preflight_failure_missing_committed_entrypoints.json",
        3664,
        "0573022fe0d0d1603f8bb9f6dd1120748407ec4c8618a6f45f0c0195ec581554",
    ),
    (
        "artifacts/state3-v1.3-20260904/"
        "state3_preflight_failure_literal_binding_paths_missing.json",
        1445,
        "87a5c8695fba047c25a72acc3a2786f5fa29c18c3e431e5844c3d18e85e433a8",
    ),
    (
        "artifacts/state3-v1.4-20260904/"
        "state3_preflight_failure_windows_host_context_transport_unavailable.json",
        908,
        "bbd48b9e3b6f1f7cd1b6dfa377a5c82dda3951309529915ec2bc407e491bab52",
    ),
)
FAILURE_ROW_KEYS: Final = {
    "filename", "original_relative_path", "relocated_relative_path", "size", "sha256",
    "byte_identical", "preserved",
}
DRIVER_PATH: Final = "artifacts/state3-recovery-20260903/state3-driver.ps1"
DRIVER_SHA256: Final = "07596fd9a0f845e1f70710521f5e6bf6ddaf603c9832c83bae5eed9be48abde5"
DRIVER_RECORD_KEYS: Final = {"path", "sha256", "executed", "byte_identical"}
FAILURE_INVENTORY_KEYS: Final = {
    "schema", "state", "expected_record_count", "records", "relocation_method",
    "original_container_removed_only_after_empty", "all_records_preserved_byte_identical",
    "original_relocated_record_count", "in_place_preserved_record_count",
    "failure_record_contents_embedded",
    "historical_overbroad_pytest_prohibited_action_accounting",
    "review_only_driver", "record_self_sha256",
}

PROHIBITED_COUNT_NAMES: Final = (
    "legacy_or_prohibited_data_access_count",
    "development_or_solver_held_out_data_access_count",
    "corpus_split_seed_access_count", "corpus_control_generation_count",
    "dataset_generation_count", "model_checkpoint_prediction_or_watermark_access_count",
    "training_count", "dependency_change_count", "network_access_count",
    "publication_count", "push_count", "external_sharing_count",
    "repository_visibility_change_count", "review_only_driver_execution_count",
    "state4_or_later_activity_count", "forbidden_rng_call_count",
)
PROHIBITED_ACTION_KEYS: Final = {
    "schema", "state", "counts", "all_counts_zero", "record_self_sha256",
}
STATE3_AUDIT_KEYS: Final = {
    "schema", "state", "outcome", "classification", "protocol_file_sha256",
    "state3_authorization_file_sha256", "solver_spec_baseline_commit",
    "solver_spec_baseline_tree", "implementation_commit", "implementation_tree",
    "measurement_bundle_self_sha256", "environment_runtime_file_sha256",
    "storage_preflight_before_file_sha256", "compute_feasibility_report_file_sha256",
    "storage_preflight_after_file_sha256", "source_blob_verification_file_sha256",
    "failure_inventory_file_sha256", "prohibited_action_counts_file_sha256", "git_before",
    "git_after", "storage_preflight_records_match", "contains_corpus_controls_or_solutions",
    "contains_scientific_arrays", "external_sharing_authorized", "state4_or_later_authorized",
    "all_checks_passed", "record_self_sha256",
}

FORBIDDEN_MEMBER_SUFFIXES: Final = (
    ".npy", ".npz", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".py", ".pyc",
    ".zip", ".tar", ".tgz", ".gz", ".7z",
)
FORBIDDEN_PAYLOAD_KEYS: Final = {
    "control", "controls", "control_values", "corpus_seed", "development_seed",
    "held_out_seed", "rho", "solution", "solution_array", "tensor_values",
    "training_seed", "validation_seed", "test_seed", "username", "sid", "credential",
    "credentials", "account_identifier",
}
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
ABSOLUTE_PATH_PATTERN: Final = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|//|/)")


class State3AuditError(RuntimeError):
    """Raised when State 3 evidence is incomplete, unsafe, or inconsistent."""


def _require_exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise State3AuditError(f"{name} keys mismatch")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise State3AuditError(f"{name} must be lowercase SHA-256 hexadecimal")
    return value


def _require_git_object(value: object, *, name: str) -> str:
    if not isinstance(value, str) or GIT_OBJECT_PATTERN.fullmatch(value) is None:
        raise State3AuditError(f"{name} must be a 40-hex Git object identifier")
    return value


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise State3AuditError(f"{name} must be a non-negative integer")
    return value


def _require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise State3AuditError(f"{name} must be non-empty text")
    return value


def _require_relative_path(value: object, *, name: str) -> str:
    text = _require_nonempty_string(value, name=name)
    path = PurePosixPath(text)
    if ("\\" in text or path.is_absolute() or ".." in path.parts or ":" in text or path.as_posix() != text):
        raise State3AuditError(f"{name} must be a canonical repository-relative POSIX path")
    return text


def _safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name or "\\" in name or path.is_absolute() or ".." in path.parts
        or len(path.parts) != 1 or ":" in name or not name.endswith(".json")
        or name.casefold().endswith(FORBIDDEN_MEMBER_SUFFIXES)
    ):
        raise State3AuditError(f"unsafe or prohibited audit member path: {name}")


def _assert_metadata_only(value: object, *, trail: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise State3AuditError("audit JSON keys must be strings")
            if key.casefold() in FORBIDDEN_PAYLOAD_KEYS:
                raise State3AuditError("prohibited payload field: " + ".".join((*trail, key)))
            _assert_metadata_only(child, trail=(*trail, key))
    elif isinstance(value, list):
        if len(value) > 64 and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        ):
            raise State3AuditError("raw numerical array is prohibited in metadata-only audit")
        for index, child in enumerate(value):
            _assert_metadata_only(child, trail=(*trail, str(index)))
    elif isinstance(value, float) and not (value == value and abs(value) != float("inf")):
        raise State3AuditError("non-finite JSON number in audit evidence")
    elif isinstance(value, str) and ABSOLUTE_PATH_PATTERN.match(value):
        raise State3AuditError("absolute path text is prohibited in metadata-only audit")


def validate_git_record(
    record: object, *, implementation_commit: str, implementation_tree: str,
    parent_commit: str | None = None, name: str = "Git record",
) -> Mapping[str, Any]:
    value = _require_exact_keys(record, GIT_RECORD_KEYS, name=name)
    _require_git_object(value["head"], name=f"{name}.head")
    _require_git_object(value["tree"], name=f"{name}.tree")
    parents = value["parents"]
    if not isinstance(parents, list) or not parents:
        raise State3AuditError(f"{name}.parents must be a non-empty list")
    for index, parent in enumerate(parents):
        _require_git_object(parent, name=f"{name}.parents[{index}]")
    if len(set(parents)) != len(parents):
        raise State3AuditError(f"{name}.parents must be unique")
    if parent_commit is not None and parents != [parent_commit]:
        raise State3AuditError(f"{name} parent mismatch")
    for field in ("tracked_diff_sha256", "staged_diff_sha256"):
        _require_sha256(value[field], name=f"{name}.{field}")
    for field in ("tracked_diff_size_bytes", "staged_diff_size_bytes"):
        if _require_nonnegative_integer(value[field], name=f"{name}.{field}") != 0:
            raise State3AuditError(f"{name} contains a Git diff")
    if (
        value["branch"] != "main" or value["head"] != implementation_commit
        or value["tree"] != implementation_tree or value["status_short_untracked_all"] != ""
        or value["tracked_diff_sha256"] != EMPTY_SHA256
        or value["staged_diff_sha256"] != EMPTY_SHA256
        or value["tracked_changes_empty"] is not True or value["staged_changes_empty"] is not True
        or value["untracked_inclusive_status_empty"] is not True
    ):
        raise State3AuditError(f"{name} is not the exact clean implementation state")
    return value


def _validate_disk_estimate_metadata(record: object) -> Mapping[str, Any]:
    from .state3_audit_verify import (
        State3AuditVerificationError,
        _validate_disk,
    )

    try:
        return _validate_disk(record)
    except State3AuditVerificationError as exc:
        raise State3AuditError("invalid writer-derived disk estimate") from exc


def build_v2_disk_planning_metadata(layout, **costs):
    """Non-production v2 planning record, deliberately outside the v1 schema.

    Preserves unknown costs and header/writer prerequisites. It cannot pass the
    frozen v1 disk-record validator or substitute for native storage preflight.
    """
    from dataclasses import asdict
    from .writer_sizing import derive_v2_disk_projection
    projection = derive_v2_disk_projection(layout, **costs)
    return dict(schema='operatormark.corpus_v2.disk_planning.v1',
                native_readiness=False, writer_binding_complete=False,
                accounting='E=R+H+J+M; C subset E; max(5GiB,4E+B+1GiB)',
                **asdict(projection))


def build_writer_disk_estimate_metadata(
    *,
    projected_npy_header_bytes: int,
    projected_jsonl_bytes: int,
    projected_manifest_and_audit_bytes: int,
    controls_only_bytes: int,
    projected_audit_bytes: int,
) -> dict[str, int]:
    """Build the frozen writer-derived storage estimate without scientific imports."""

    inputs = (
        projected_npy_header_bytes,
        projected_jsonl_bytes,
        projected_manifest_and_audit_bytes,
        controls_only_bytes,
        projected_audit_bytes,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in inputs
    ):
        raise State3AuditError("writer disk-estimate inputs must be non-negative integers")
    accepted = (
        RAW_SEVEN_TENSOR_PAYLOAD_BYTES
        + projected_npy_header_bytes
        + projected_jsonl_bytes
        + projected_manifest_and_audit_bytes
    )
    if controls_only_bytes > accepted:
        raise State3AuditError("controls-only writer estimate exceeds accepted corpus estimate")
    four_copy_formula = 4 * accepted + projected_audit_bytes + 2**30
    result = {
        "raw_seven_tensor_payload_bytes": RAW_SEVEN_TENSOR_PAYLOAD_BYTES,
        "projected_npy_header_bytes": projected_npy_header_bytes,
        "projected_jsonl_bytes": projected_jsonl_bytes,
        "projected_manifest_and_audit_bytes": projected_manifest_and_audit_bytes,
        "controls_only_bytes": controls_only_bytes,
        "estimated_accepted_bytes": accepted,
        "active_staging_bytes": accepted,
        "readback_atomic_publication_bytes": accepted,
        "preserved_failed_attempt_allowance_bytes": accepted,
        "projected_audit_bytes": projected_audit_bytes,
        "fixed_headroom_bytes": 2**30,
        "minimum_floor_bytes": 5 * 2**30,
        "four_copy_formula_bytes": four_copy_formula,
        "total_required_free_space_bytes": max(5 * 2**30, four_copy_formula),
    }
    _validate_disk_estimate_metadata(result)
    return result


def build_compute_feasibility_preflight_bindings_record(
    *,
    protocol_file_sha256: str,
    state3_authorization_file_sha256: str,
    implementation_commit: str,
    implementation_tree: str,
    launcher_file_sha256: str,
    runtime_inspector_file_sha256: str,
    environment_record_sha256: str,
    storage_preflight_before_file_sha256: str,
    disk_estimate: Mapping[str, Any],
    git_before: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical pre-measurement bindings through a metadata-only API."""

    payload = {
        "schema": (
            "operatormark.darcy_branch_c_v1."
            "compute_feasibility_preflight_bindings.v1"
        ),
        "protocol_file_sha256": protocol_file_sha256,
        "state3_authorization_file_sha256": state3_authorization_file_sha256,
        "solver_spec_baseline_commit": SOLVER_SPEC_BASELINE_COMMIT,
        "solver_spec_baseline_tree": SOLVER_SPEC_BASELINE_TREE,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "launcher_file_sha256": launcher_file_sha256,
        "runtime_inspector_file_sha256": runtime_inspector_file_sha256,
        "environment_record_sha256": environment_record_sha256,
        "storage_preflight_before_file_sha256": storage_preflight_before_file_sha256,
        "disk_estimate": dict(disk_estimate),
        "git_before": dict(git_before),
    }
    result = with_self_hash(payload, "preflight_bindings_self_sha256")
    validate_compute_feasibility_preflight_bindings_metadata(result)
    return result


def _validate_raw_measurement_metadata(record: object) -> Mapping[str, Any]:
    from .state3_audit_verify import (
        RAW_KEYS,
        State3AuditVerificationError,
        _validate_raw_measurement,
    )

    value = _require_exact_keys(record, set(RAW_KEYS), name="raw measurement")
    try:
        _validate_raw_measurement(value)
    except State3AuditVerificationError as exc:
        raise State3AuditError("invalid fixed-fixture raw measurement") from exc
    return value


def validate_compute_feasibility_preflight_bindings_metadata(
    record: Mapping[str, Any],
) -> None:
    value = _require_exact_keys(
        record, PREFLIGHT_BINDING_KEYS, name="compute-feasibility preflight bindings"
    )
    if (
        value["schema"]
        != "operatormark.darcy_branch_c_v1.compute_feasibility_preflight_bindings.v1"
        or value["protocol_file_sha256"] != PROTOCOL_SHA256
        or value["solver_spec_baseline_commit"] != SOLVER_SPEC_BASELINE_COMMIT
        or value["solver_spec_baseline_tree"] != SOLVER_SPEC_BASELINE_TREE
    ):
        raise State3AuditError("preflight binding frozen identity mismatch")
    for name in (
        "protocol_file_sha256", "state3_authorization_file_sha256",
        "launcher_file_sha256", "runtime_inspector_file_sha256",
        "environment_record_sha256", "storage_preflight_before_file_sha256",
        "preflight_bindings_self_sha256",
    ):
        _require_sha256(value[name], name=f"preflight_bindings.{name}")
    for name in (
        "solver_spec_baseline_commit", "solver_spec_baseline_tree",
        "implementation_commit", "implementation_tree",
    ):
        _require_git_object(value[name], name=f"preflight_bindings.{name}")
    _validate_disk_estimate_metadata(value["disk_estimate"])
    validate_git_record(
        value["git_before"], implementation_commit=value["implementation_commit"],
        implementation_tree=value["implementation_tree"], name="preflight_bindings.git_before",
    )
    try:
        validate_self_hash(value, "preflight_bindings_self_sha256")
    except ValueError as exc:
        raise State3AuditError("preflight bindings self-hash mismatch") from exc
    _assert_metadata_only(value)


def validate_measurement_bundle_metadata(record: Mapping[str, Any]) -> None:
    value = _require_exact_keys(record, MEASUREMENT_BUNDLE_KEYS, name="measurement bundle")
    if (
        value["schema"]
        != "operatormark.darcy_branch_c_v1.compute_feasibility_measurements.v1"
        or value["state"]
        != "state3_feasibility_measurements_complete_pending_post_storage"
        or value["protocol_file_sha256"] != PROTOCOL_SHA256
        or value["warmup_count_per_fixture"] != 0
        or value["measured_count_per_fixture"] != MEASURED_COUNT_PER_FIXTURE
        or value["measurement_order"] != list(FIXTURE_ORDER)
    ):
        raise State3AuditError("measurement-bundle identity, count, or order mismatch")
    for name in (
        "protocol_file_sha256", "state3_authorization_file_sha256",
        "measurement_bundle_self_sha256",
    ):
        _require_sha256(value[name], name=f"measurement_bundle.{name}")
    for name in ("implementation_commit", "implementation_tree"):
        _require_git_object(value[name], name=f"measurement_bundle.{name}")
    raw = value["raw_measurements"]
    if not isinstance(raw, list) or len(raw) != len(FIXTURE_ORDER) * MEASURED_COUNT_PER_FIXTURE:
        raise State3AuditError("measurement bundle must contain exactly fifteen records")
    expected = [
        (fixture, repeat_index) for fixture in FIXTURE_ORDER
        for repeat_index in range(MEASURED_COUNT_PER_FIXTURE)
    ]
    if [
        (item.get("fixture"), item.get("repeat_index"))
        if isinstance(item, Mapping) else (None, None)
        for item in raw
    ] != expected:
        raise State3AuditError("measurement-bundle fixture-major coverage mismatch")
    validated = [_validate_raw_measurement_metadata(item) for item in raw]
    expected_fixture = all(item["gates_passed"] for item in validated)
    expected_rng = all(all(item["rng_state_equalities"].values()) for item in validated)
    expected_rng_count = sum(item["forbidden_rng_call_count"] for item in validated)
    expected_seed_count = sum(item["forbidden_corpus_seed_access_count"] for item in validated)
    if (
        value["all_fixture_gates_passed"] is not expected_fixture
        or value["all_rng_state_equalities_passed"] is not expected_rng
        or value["forbidden_rng_call_count"] != expected_rng_count
        or value["forbidden_corpus_seed_access_count"] != expected_seed_count
        or expected_fixture is not True or expected_rng is not True
        or expected_rng_count != 0 or expected_seed_count != 0
    ):
        raise State3AuditError("measurement-bundle aggregate mismatch")
    try:
        validate_self_hash(value, "measurement_bundle_self_sha256")
    except ValueError as exc:
        raise State3AuditError("measurement-bundle self-hash mismatch") from exc
    _assert_metadata_only(value)


def _build_summary_metadata(raw: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            field: {
                "maximum": max(int(record[field]) for record in records),
                "median": statistics.median(int(record[field]) for record in records),
            }
            for field in SUMMARY_FIELDS
        }

    return {
        "fields": list(SUMMARY_FIELDS),
        "global": group(raw),
        "per_fixture": {
            fixture: group([record for record in raw if record["fixture"] == fixture])
            for fixture in FIXTURE_ORDER
        },
    }


def _build_extrapolation_metadata(
    raw: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    paired = [int(record["paired_solver_ns"]) for record in raw]
    peaks = [int(record["peak_rss_bytes"]) for record in raw]
    median_pair_seconds = float(statistics.median(paired)) / 1e9
    max_pair_seconds = float(max(paired)) / 1e9
    median_extrapolated_seconds = CORPUS_SAMPLE_COUNT * median_pair_seconds
    max_extrapolated_seconds = CORPUS_SAMPLE_COUNT * max_pair_seconds
    budget_seconds = FROZEN_SAFETY_FACTOR * max_extrapolated_seconds
    return {
        "median_pair_seconds": median_pair_seconds,
        "max_pair_seconds": max_pair_seconds,
        "median_extrapolated_seconds": median_extrapolated_seconds,
        "max_extrapolated_seconds": max_extrapolated_seconds,
        "frozen_safety_factor": FROZEN_SAFETY_FACTOR,
        "safety_factored_compute_budget_seconds": budget_seconds,
        "safety_factored_peak_memory_budget_bytes": 2 * max(peaks),
        "median_extrapolated_hours": median_extrapolated_seconds / 3600.0,
        "max_extrapolated_hours": max_extrapolated_seconds / 3600.0,
        "safety_factored_compute_budget_hours": budget_seconds / 3600.0,
    }


def validate_compute_feasibility_report_metadata(record: Mapping[str, Any]) -> None:
    from .state3_audit_verify import State3AuditVerificationError, _validate_report

    try:
        _validate_report(record)
    except State3AuditVerificationError as exc:
        raise State3AuditError("invalid compute-feasibility report metadata") from exc
    _assert_metadata_only(record)


def build_compute_feasibility_report_metadata(
    bindings: Mapping[str, Any], raw_measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = PREFLIGHT_BINDING_KEYS - {"schema", "preflight_bindings_self_sha256"} | {
        "storage_preflight_after_file_sha256", "git_after"
    }
    value = _require_exact_keys(bindings, required, name="final compute-feasibility bindings")
    if (
        value["protocol_file_sha256"] != PROTOCOL_SHA256
        or value["solver_spec_baseline_commit"] != SOLVER_SPEC_BASELINE_COMMIT
        or value["solver_spec_baseline_tree"] != SOLVER_SPEC_BASELINE_TREE
    ):
        raise State3AuditError("final report binding frozen identity mismatch")
    _validate_disk_estimate_metadata(value["disk_estimate"])
    before = validate_git_record(
        value["git_before"], implementation_commit=value["implementation_commit"],
        implementation_tree=value["implementation_tree"], name="report.git_before",
    )
    after = validate_git_record(
        value["git_after"], implementation_commit=value["implementation_commit"],
        implementation_tree=value["implementation_tree"], name="report.git_after",
    )
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        raise State3AuditError("Git state changed across compute-feasibility metadata")
    raw = [dict(_validate_raw_measurement_metadata(item)) for item in raw_measurements]
    expected = [
        (fixture, repeat_index) for fixture in FIXTURE_ORDER
        for repeat_index in range(MEASURED_COUNT_PER_FIXTURE)
    ]
    if [(item["fixture"], item["repeat_index"]) for item in raw] != expected:
        raise State3AuditError("raw measurements are incomplete, duplicated, or reordered")
    counter_names = {item["rss_counter_name"] for item in raw}
    if len(counter_names) != 1:
        raise State3AuditError("workers used different process-memory counters")
    counter_name = next(iter(counter_names))
    payload = {
        "schema": "operatormark.darcy_branch_c_v1.compute_feasibility.v2",
        "state": "state3_feasibility_only_no_corpus_controls",
        "protocol_file_sha256": value["protocol_file_sha256"],
        "state3_authorization_file_sha256": value["state3_authorization_file_sha256"],
        "solver_spec_baseline_commit": value["solver_spec_baseline_commit"],
        "solver_spec_baseline_tree": value["solver_spec_baseline_tree"],
        "implementation_commit": value["implementation_commit"],
        "implementation_tree": value["implementation_tree"],
        "launcher_file_sha256": value["launcher_file_sha256"],
        "runtime_inspector_file_sha256": value["runtime_inspector_file_sha256"],
        "environment_record_sha256": value["environment_record_sha256"],
        "storage_preflight_before_file_sha256": value[
            "storage_preflight_before_file_sha256"
        ],
        "storage_preflight_after_file_sha256": value[
            "storage_preflight_after_file_sha256"
        ],
        "fixture_definitions": FIXTURE_DEFINITIONS,
        "warmup_count_per_fixture": 0,
        "measured_count_per_fixture": MEASURED_COUNT_PER_FIXTURE,
        "measurement_order": list(FIXTURE_ORDER),
        "timing_clock": {
            "name": "time.perf_counter_ns", "unit": "nanoseconds", "monotonic": True,
            "interpretation": "same_platform_feasibility_estimates_not_performance_guarantees",
        },
        "memory_counter": {
            "name": counter_name, "unit": "bytes",
            "definition": (
                "absolute_peak_working_set"
                if counter_name == "windows_process_peak_working_set_bytes"
                else "platform_ru_maxrss_normalized_to_bytes"
            ),
            "cross_platform_equivalence_claimed": False,
        },
        "rng_state_commitment_specification": RNG_STATE_COMMITMENT_SPECIFICATION,
        "raw_measurements": raw,
        "summary": _build_summary_metadata(raw),
        "corpus_sample_count": CORPUS_SAMPLE_COUNT,
        "frozen_safety_factor": FROZEN_SAFETY_FACTOR,
        "extrapolation": _build_extrapolation_metadata(raw),
        "disk_estimate": dict(value["disk_estimate"]),
        "all_fixture_gates_passed": all(item["gates_passed"] for item in raw),
        "all_rng_state_equalities_passed": all(
            all(item["rng_state_equalities"].values()) for item in raw
        ),
        "forbidden_rng_call_count": sum(item["forbidden_rng_call_count"] for item in raw),
        "forbidden_corpus_seed_access_count": sum(
            item["forbidden_corpus_seed_access_count"] for item in raw
        ),
        "git_before": dict(value["git_before"]),
        "git_after": dict(value["git_after"]),
    }
    result = with_self_hash(payload, "report_self_sha256")
    validate_compute_feasibility_report_metadata(result)
    return result


def _environment_commitment(environment: Mapping[str, str]) -> str:
    return hashlib.sha256(
        b"operatormark-parent-environment-v1\n" + canonical_json_bytes(dict(environment))
    ).hexdigest()


def _validate_base_import_closure_record(record: Mapping[str, Any]) -> None:
    from .base_imports import validate_base_import_closure
    validate_base_import_closure(record)


def _validate_runtime_record(record: Mapping[str, Any]) -> None:
    from .runtime import validate_numerical_runtime_record
    validate_numerical_runtime_record(record)


def build_environment_runtime_record(
    *, protocol_file_sha256: str, state3_authorization_file_sha256: str,
    implementation_commit: str, implementation_tree: str,
    parent_environment: Mapping[str, str], launcher_file_sha256: str,
    runtime_inspector_file_sha256: str, requirements_lock_git_blob: str,
    requirements_lock_file_sha256: str, base_import_closure: Mapping[str, Any],
    numerical_runtime: Mapping[str, Any], python_version: str, numpy_version: str,
    scipy_version: str, operating_system: str, architecture: str, locale: str,
) -> dict[str, Any]:
    environment = dict(parent_environment)
    payload = {
        "schema": STATE3_MEMBER_SCHEMAS["environment_runtime.json"], "state": STATE3_STATE,
        "protocol_file_sha256": protocol_file_sha256,
        "state3_authorization_file_sha256": state3_authorization_file_sha256,
        "implementation_commit": implementation_commit, "implementation_tree": implementation_tree,
        "python_version": python_version, "numpy_version": numpy_version,
        "scipy_version": scipy_version, "operating_system": operating_system,
        "architecture": architecture, "locale": locale, "parent_environment": environment,
        "parent_environment_commitment": _environment_commitment(environment),
        "applicable_additional_thread_variables": sorted(set(environment) - set(MANDATORY_PARENT_ENVIRONMENT)),
        "launcher_file_sha256": launcher_file_sha256,
        "runtime_inspector_file_sha256": runtime_inspector_file_sha256,
        "requirements_lock_path": "requirements-lock-cpu.txt",
        "requirements_lock_git_blob": requirements_lock_git_blob,
        "requirements_lock_file_sha256": requirements_lock_file_sha256,
        "base_import_closure": dict(base_import_closure),
        "base_import_closure_sha256": base_import_closure.get("closure_sha256"),
        "numerical_runtime": dict(numerical_runtime),
        "numerical_runtime_sha256": numerical_runtime.get("runtime_record_sha256"),
        "all_environment_checks_passed": True, "all_runtime_checks_passed": True,
    }
    result = with_self_hash(payload, "record_self_sha256")
    validate_environment_runtime_record(result)
    return result


def validate_environment_runtime_record(record: Mapping[str, Any]) -> None:
    value = _require_exact_keys(record, ENVIRONMENT_RUNTIME_KEYS, name="environment/runtime record")
    if (
        value["schema"] != STATE3_MEMBER_SCHEMAS["environment_runtime.json"]
        or value["state"] != STATE3_STATE or value["protocol_file_sha256"] != PROTOCOL_SHA256
    ):
        raise State3AuditError("environment/runtime frozen identity mismatch")
    for name in (
        "protocol_file_sha256", "state3_authorization_file_sha256", "launcher_file_sha256",
        "runtime_inspector_file_sha256", "requirements_lock_file_sha256",
        "base_import_closure_sha256", "numerical_runtime_sha256",
        "parent_environment_commitment", "record_self_sha256",
    ):
        _require_sha256(value[name], name=f"environment_runtime.{name}")
    for name in ("implementation_commit", "implementation_tree", "requirements_lock_git_blob"):
        _require_git_object(value[name], name=f"environment_runtime.{name}")
    if value["requirements_lock_path"] != "requirements-lock-cpu.txt":
        raise State3AuditError("dependency-lock path mismatch")
    if (value["python_version"], value["numpy_version"], value["scipy_version"]) != (
        "3.13.5", "2.5.2", "1.18.1"
    ):
        raise State3AuditError("frozen interpreter/dependency versions changed")
    for name in ("operating_system", "architecture", "locale"):
        _require_nonempty_string(value[name], name=f"environment_runtime.{name}")
    environment = value["parent_environment"]
    if not isinstance(environment, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) and item for key, item in environment.items()
    ):
        raise State3AuditError("parent environment must map names to non-empty strings")
    if any(environment.get(key) != item for key, item in MANDATORY_PARENT_ENVIRONMENT.items()):
        raise State3AuditError("mandatory parent environment mismatch")
    additional = value["applicable_additional_thread_variables"]
    if (
        not isinstance(additional, list) or additional != sorted(set(additional))
        or set(additional) != set(environment) - set(MANDATORY_PARENT_ENVIRONMENT)
    ):
        raise State3AuditError("additional parent thread-policy binding mismatch")
    if value["parent_environment_commitment"] != _environment_commitment(environment):
        raise State3AuditError("parent environment commitment mismatch")
    closure, runtime = value["base_import_closure"], value["numerical_runtime"]
    if not isinstance(closure, Mapping) or not isinstance(runtime, Mapping):
        raise State3AuditError("closure/runtime must be objects")
    _validate_base_import_closure_record(closure)
    _validate_runtime_record(runtime)
    if closure.get("closure_sha256") != value["base_import_closure_sha256"]:
        raise State3AuditError("base-import closure binding mismatch")
    if runtime.get("runtime_record_sha256") != value["numerical_runtime_sha256"]:
        raise State3AuditError("numerical-runtime binding mismatch")
    if runtime.get("applicable_additional_thread_variables") != additional:
        raise State3AuditError("runtime/additional-policy set mismatch")
    if runtime.get("bound_applicable_thread_policy") != {name: environment[name] for name in additional}:
        raise State3AuditError("runtime/additional-policy values mismatch")
    if (
        value["all_environment_checks_passed"] is not True
        or value["all_runtime_checks_passed"] is not True
        or runtime.get("all_runtime_requirements_satisfied") is not True
    ):
        raise State3AuditError("environment/runtime checks did not all pass")
    try:
        validate_self_hash(value, "record_self_sha256")
    except ValueError as exc:
        raise State3AuditError("environment/runtime self-hash mismatch") from exc
    _assert_metadata_only(value)


def build_source_blob_verification_record(
    *, protocol_file_sha256: str, state3_authorization_file_sha256: str,
    implementation_commit: str, implementation_tree: str, implementation_parent_commit: str,
    requirements_lock_git_blob: str, requirements_lock_file_sha256: str,
    frozen_baseline_blobs: Sequence[Mapping[str, Any]],
    implementation_files: Sequence[Mapping[str, Any]],
    git_before: Mapping[str, Any], git_after: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": STATE3_MEMBER_SCHEMAS["source_blob_verification.json"], "state": STATE3_STATE,
        "protocol_file_sha256": protocol_file_sha256,
        "state3_authorization_file_sha256": state3_authorization_file_sha256,
        "solver_spec_baseline_commit": SOLVER_SPEC_BASELINE_COMMIT,
        "solver_spec_baseline_tree": SOLVER_SPEC_BASELINE_TREE,
        "implementation_commit": implementation_commit, "implementation_tree": implementation_tree,
        "implementation_parent_commit": implementation_parent_commit,
        "requirements_lock_path": "requirements-lock-cpu.txt",
        "requirements_lock_git_blob": requirements_lock_git_blob,
        "requirements_lock_file_sha256": requirements_lock_file_sha256,
        "frozen_baseline_blobs": [dict(item) for item in frozen_baseline_blobs],
        "implementation_files": [dict(item) for item in implementation_files],
        "all_frozen_baseline_blobs_byte_identical": True,
        "all_implementation_files_match_git": True,
        "git_before": dict(git_before), "git_after": dict(git_after),
    }
    result = with_self_hash(payload, "record_self_sha256")
    validate_source_blob_verification_record(result)
    return result


def validate_source_blob_verification_record(record: Mapping[str, Any]) -> None:
    value = _require_exact_keys(record, SOURCE_BLOB_KEYS, name="source/blob verification")
    if (
        value["schema"] != STATE3_MEMBER_SCHEMAS["source_blob_verification.json"]
        or value["state"] != STATE3_STATE or value["protocol_file_sha256"] != PROTOCOL_SHA256
        or value["solver_spec_baseline_commit"] != SOLVER_SPEC_BASELINE_COMMIT
        or value["solver_spec_baseline_tree"] != SOLVER_SPEC_BASELINE_TREE
        or value["requirements_lock_path"] != "requirements-lock-cpu.txt"
    ):
        raise State3AuditError("source/blob frozen identity mismatch")
    for name in ("protocol_file_sha256", "state3_authorization_file_sha256",
                 "requirements_lock_file_sha256", "record_self_sha256"):
        _require_sha256(value[name], name=f"source_blob.{name}")
    for name in (
        "solver_spec_baseline_commit", "solver_spec_baseline_tree", "implementation_commit",
        "implementation_tree", "implementation_parent_commit", "requirements_lock_git_blob",
    ):
        _require_git_object(value[name], name=f"source_blob.{name}")

    baseline = value["frozen_baseline_blobs"]
    if not isinstance(baseline, list):
        raise State3AuditError("frozen baseline blobs must be a list")
    paths: list[str] = []
    for index, item in enumerate(baseline):
        row = _require_exact_keys(item, BASELINE_BLOB_KEYS, name=f"baseline blob {index}")
        path = _require_relative_path(row["path"], name=f"baseline blob {index} path")
        paths.append(path)
        for name in ("baseline_git_blob", "implementation_git_blob"):
            _require_git_object(row[name], name=f"{path}.{name}")
        for name in ("baseline_file_sha256", "implementation_file_sha256"):
            _require_sha256(row[name], name=f"{path}.{name}")
        if (
            row["byte_identical"] is not True
            or row["baseline_git_blob"] != row["implementation_git_blob"]
            or row["baseline_file_sha256"] != row["implementation_file_sha256"]
        ):
            raise State3AuditError(f"frozen baseline blob changed: {path}")
        expected_review_hash = FROZEN_REVIEW_COPY_SHA256.get(path)
        if expected_review_hash is not None and row["implementation_file_sha256"] != expected_review_hash:
            raise State3AuditError(f"review-copy commitment mismatch: {path}")
    if paths != sorted(set(paths)) or not set(FROZEN_BASELINE_REQUIRED_PATHS).issubset(paths):
        raise State3AuditError("frozen baseline coverage or ordering mismatch")

    implementation = value["implementation_files"]
    if not isinstance(implementation, list) or not implementation:
        raise State3AuditError("implementation source inventory must be non-empty")
    implementation_paths: list[str] = []
    for index, item in enumerate(implementation):
        row = _require_exact_keys(item, IMPLEMENTATION_FILE_KEYS, name=f"implementation file {index}")
        path = _require_relative_path(row["path"], name=f"implementation file {index} path")
        implementation_paths.append(path)
        _require_git_object(row["git_blob"], name=f"{path}.git_blob")
        _require_sha256(row["file_sha256"], name=f"{path}.file_sha256")
        if row["matches_git_blob"] is not True:
            raise State3AuditError(f"implementation file does not match Git: {path}")
    if implementation_paths != sorted(set(implementation_paths)):
        raise State3AuditError("implementation source inventory is not sorted and unique")
    if (
        value["all_frozen_baseline_blobs_byte_identical"] is not True
        or value["all_implementation_files_match_git"] is not True
    ):
        raise State3AuditError("source/blob aggregate did not pass")
    before = validate_git_record(
        value["git_before"], implementation_commit=value["implementation_commit"],
        implementation_tree=value["implementation_tree"],
        parent_commit=value["implementation_parent_commit"], name="source_blob.git_before",
    )
    after = validate_git_record(
        value["git_after"], implementation_commit=value["implementation_commit"],
        implementation_tree=value["implementation_tree"],
        parent_commit=value["implementation_parent_commit"], name="source_blob.git_after",
    )
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        raise State3AuditError("Git state changed across State 3")
    try:
        validate_self_hash(value, "record_self_sha256")
    except ValueError as exc:
        raise State3AuditError("source/blob self-hash mismatch") from exc
    _assert_metadata_only(value)


def expected_original_failure_inventory_rows() -> list[dict[str, Any]]:
    return [
        {
            "filename": filename,
            "original_relative_path": f"{FAILURE_ORIGINAL_PREFIX}/{filename}",
            "relocated_relative_path": f"{FAILURE_DESTINATION_PREFIX}/{filename}",
            "size": size, "sha256": sha256, "byte_identical": True, "preserved": True,
        }
        for filename, size, sha256 in FAILURE_SPECS
    ]


def expected_failure_inventory_rows() -> list[dict[str, Any]]:
    rows = expected_original_failure_inventory_rows()
    rows.extend(
        {
            "filename": PurePosixPath(path).name,
            "original_relative_path": path,
            "relocated_relative_path": path,
            "size": size,
            "sha256": sha256,
            "byte_identical": True,
            "preserved": True,
        }
        for path, size, sha256 in IN_PLACE_FAILURE_SPECS
    )
    return rows


def build_failure_inventory_record(
    records: Sequence[Mapping[str, Any]], *, review_only_driver_sha256: str = DRIVER_SHA256,
) -> dict[str, Any]:
    payload = {
        "schema": STATE3_MEMBER_SCHEMAS["failure_inventory.json"], "state": STATE3_STATE,
        "expected_record_count": 17,
        "original_relocated_record_count": 12,
        "in_place_preserved_record_count": 5,
        "records": [dict(item) for item in records],
        "relocation_method": (
            "original_twelve_same_volume_atomic_directory_rename_"
            "plus_five_preserved_in_place"
        ),
        "original_container_removed_only_after_empty": True,
        "all_records_preserved_byte_identical": True,
        "failure_record_contents_embedded": False,
        "historical_overbroad_pytest_prohibited_action_accounting": "not_provably_zero",
        "review_only_driver": {
            "path": DRIVER_PATH, "sha256": review_only_driver_sha256,
            "executed": False, "byte_identical": True,
        },
    }
    result = with_self_hash(payload, "record_self_sha256")
    validate_failure_inventory_record(result)
    return result


def validate_failure_inventory_record(record: Mapping[str, Any]) -> None:
    value = _require_exact_keys(record, FAILURE_INVENTORY_KEYS, name="failure inventory")
    if (
        value["schema"] != STATE3_MEMBER_SCHEMAS["failure_inventory.json"]
        or value["state"] != STATE3_STATE or value["expected_record_count"] != 17
        or value["original_relocated_record_count"] != 12
        or value["in_place_preserved_record_count"] != 5
        or value["records"] != expected_failure_inventory_rows()
        or value["relocation_method"]
        != "original_twelve_same_volume_atomic_directory_rename_plus_five_preserved_in_place"
        or value["original_container_removed_only_after_empty"] is not True
        or value["all_records_preserved_byte_identical"] is not True
        or value["failure_record_contents_embedded"] is not False
        or value["historical_overbroad_pytest_prohibited_action_accounting"]
        != "not_provably_zero"
    ):
        raise State3AuditError("failure evidence inventory changed or is incomplete")
    filenames: list[str] = []
    current_paths: list[str] = []
    for index, item in enumerate(value["records"]):
        row = _require_exact_keys(item, FAILURE_ROW_KEYS, name="failure inventory row")
        filenames.append(_require_nonempty_string(row["filename"], name="failure filename"))
        current_paths.append(
            _require_relative_path(
                row["relocated_relative_path"], name="failure current path"
            )
        )
        _require_relative_path(row["original_relative_path"], name="failure original path")
        _require_nonnegative_integer(row["size"], name="failure size")
        _require_sha256(row["sha256"], name="failure sha256")
        if row["preserved"] is not True or row["byte_identical"] is not True:
            raise State3AuditError(f"failure inventory row {index} is not preserved")
    if len(set(filenames)) != 17 or len(set(current_paths)) != 17:
        raise State3AuditError("failure inventory filenames and paths must be unique")
    driver = _require_exact_keys(value["review_only_driver"], DRIVER_RECORD_KEYS,
                                 name="review-only driver")
    if (
        driver["path"] != DRIVER_PATH or driver["sha256"] != DRIVER_SHA256
        or driver["executed"] is not False or driver["byte_identical"] is not True
    ):
        raise State3AuditError("review-only driver was changed or executed")
    try:
        validate_self_hash(value, "record_self_sha256")
    except ValueError as exc:
        raise State3AuditError("failure inventory self-hash mismatch") from exc
    _assert_metadata_only(value)


def verify_failure_inventory_physical_bindings(
    repository_root: str | Path,
    record: Mapping[str, Any],
    *,
    binding_reader: Callable[[Path], tuple[int, str]] | None = None,
) -> None:
    """Verify only path, size, and streaming SHA-256 for the exact 17 files."""

    validate_failure_inventory_record(record)
    root = Path(repository_root).resolve(strict=True)

    def read_binding(path: Path) -> tuple[int, str]:
        if path.is_symlink() or not path.is_file():
            raise State3AuditError("failure binding is not one ordinary file")
        return path.stat().st_size, file_sha256(path)

    reader = read_binding if binding_reader is None else binding_reader
    for row in record["records"]:
        relative = PurePosixPath(row["relocated_relative_path"])
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise State3AuditError("failure binding path traverses a symbolic link")
        observed_size, observed_sha256 = reader(current)
        if observed_size != row["size"] or observed_sha256 != row["sha256"]:
            raise State3AuditError(
                f"failure binding size or SHA-256 mismatch: {row['filename']}"
            )


def build_prohibited_action_counts_record(counts: Mapping[str, int]) -> dict[str, Any]:
    result = with_self_hash(
        {"schema": STATE3_MEMBER_SCHEMAS["prohibited_action_counts.json"],
         "state": STATE3_STATE, "counts": dict(counts), "all_counts_zero": True},
        "record_self_sha256",
    )
    validate_prohibited_action_counts_record(result)
    return result


def validate_prohibited_action_counts_record(record: Mapping[str, Any]) -> None:
    value = _require_exact_keys(record, PROHIBITED_ACTION_KEYS, name="prohibited-action record")
    if (
        value["schema"] != STATE3_MEMBER_SCHEMAS["prohibited_action_counts.json"]
        or value["state"] != STATE3_STATE or value["all_counts_zero"] is not True
    ):
        raise State3AuditError("prohibited-action record identity mismatch")
    counts = _require_exact_keys(value["counts"], set(PROHIBITED_COUNT_NAMES),
                                 name="prohibited-action counts")
    if any(_require_nonnegative_integer(item, name=name) != 0 for name, item in counts.items()):
        raise State3AuditError("a prohibited-action count is nonzero")
    try:
        validate_self_hash(value, "record_self_sha256")
    except ValueError as exc:
        raise State3AuditError("prohibited-action self-hash mismatch") from exc
    _assert_metadata_only(value)


def build_state3_audit_record(
    *, protocol_file_sha256: str, state3_authorization_file_sha256: str,
    implementation_commit: str, implementation_tree: str,
    measurement_bundle_self_sha256: str, evidence_file_sha256: Mapping[str, str],
    git_before: Mapping[str, Any], git_after: Mapping[str, Any],
) -> dict[str, Any]:
    if set(evidence_file_sha256) != set(STATE3_AUDIT_MEMBER_ORDER[1:-1]):
        raise State3AuditError("State 3 root evidence-file set mismatch")
    payload = {
        "schema": STATE3_MEMBER_SCHEMAS["state3_audit.json"], "state": STATE3_STATE,
        "outcome": STATE3_OUTCOME, "classification": STATE3_CLASSIFICATION,
        "protocol_file_sha256": protocol_file_sha256,
        "state3_authorization_file_sha256": state3_authorization_file_sha256,
        "solver_spec_baseline_commit": SOLVER_SPEC_BASELINE_COMMIT,
        "solver_spec_baseline_tree": SOLVER_SPEC_BASELINE_TREE,
        "implementation_commit": implementation_commit, "implementation_tree": implementation_tree,
        "measurement_bundle_self_sha256": measurement_bundle_self_sha256,
        "environment_runtime_file_sha256": evidence_file_sha256["environment_runtime.json"],
        "storage_preflight_before_file_sha256": evidence_file_sha256["storage_preflight_before.json"],
        "compute_feasibility_report_file_sha256": evidence_file_sha256["compute_feasibility_report.json"],
        "storage_preflight_after_file_sha256": evidence_file_sha256["storage_preflight_after.json"],
        "source_blob_verification_file_sha256": evidence_file_sha256["source_blob_verification.json"],
        "failure_inventory_file_sha256": evidence_file_sha256["failure_inventory.json"],
        "prohibited_action_counts_file_sha256": evidence_file_sha256["prohibited_action_counts.json"],
        "git_before": dict(git_before), "git_after": dict(git_after),
        "storage_preflight_records_match": True,
        "contains_corpus_controls_or_solutions": False, "contains_scientific_arrays": False,
        "external_sharing_authorized": False, "state4_or_later_authorized": False,
        "all_checks_passed": True,
    }
    result = with_self_hash(payload, "record_self_sha256")
    validate_state3_audit_record(result)
    return result


def validate_state3_audit_record(record: Mapping[str, Any]) -> None:
    value = _require_exact_keys(record, STATE3_AUDIT_KEYS, name="State 3 audit root")
    if (
        value["schema"] != STATE3_MEMBER_SCHEMAS["state3_audit.json"]
        or value["state"] != STATE3_STATE or value["outcome"] != STATE3_OUTCOME
        or value["classification"] != STATE3_CLASSIFICATION
        or value["protocol_file_sha256"] != PROTOCOL_SHA256
        or value["solver_spec_baseline_commit"] != SOLVER_SPEC_BASELINE_COMMIT
        or value["solver_spec_baseline_tree"] != SOLVER_SPEC_BASELINE_TREE
    ):
        raise State3AuditError("State 3 audit frozen identity mismatch")
    for name in (
        "protocol_file_sha256", "state3_authorization_file_sha256",
        "measurement_bundle_self_sha256", "environment_runtime_file_sha256",
        "storage_preflight_before_file_sha256", "compute_feasibility_report_file_sha256",
        "storage_preflight_after_file_sha256", "source_blob_verification_file_sha256",
        "failure_inventory_file_sha256", "prohibited_action_counts_file_sha256",
        "record_self_sha256",
    ):
        _require_sha256(value[name], name=f"state3_audit.{name}")
    for name in ("solver_spec_baseline_commit", "solver_spec_baseline_tree",
                 "implementation_commit", "implementation_tree"):
        _require_git_object(value[name], name=f"state3_audit.{name}")
    before = validate_git_record(value["git_before"], implementation_commit=value["implementation_commit"],
                                 implementation_tree=value["implementation_tree"], name="state3_audit.git_before")
    after = validate_git_record(value["git_after"], implementation_commit=value["implementation_commit"],
                                implementation_tree=value["implementation_tree"], name="state3_audit.git_after")
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        raise State3AuditError("Git state changed across State 3 audit")
    if (
        value["storage_preflight_records_match"] is not True
        or value["contains_corpus_controls_or_solutions"] is not False
        or value["contains_scientific_arrays"] is not False
        or value["external_sharing_authorized"] is not False
        or value["state4_or_later_authorized"] is not False
        or value["all_checks_passed"] is not True
    ):
        raise State3AuditError("State 3 audit safety flags mismatch")
    try:
        validate_self_hash(value, "record_self_sha256")
    except ValueError as exc:
        raise State3AuditError("State 3 audit self-hash mismatch") from exc
    _assert_metadata_only(value)


def _validate_storage_record(record: Mapping[str, Any]) -> None:
    from .storage import validate_storage_preflight_record
    validate_storage_preflight_record(record)


def _compare_storage_records(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    from .storage import compare_storage_preflight_records
    comparison = compare_storage_preflight_records(before, after)
    if comparison.get("matched") is not True:
        raise State3AuditError("repeated storage-preflight records differ")


def validate_state3_evidence_member(name: str, data: bytes) -> dict[str, Any]:
    """Validate one canonical metadata-only member and its exact nested schema."""
    if name not in STATE3_MEMBER_SCHEMAS:
        raise State3AuditError(f"unexpected State 3 evidence member: {name}")
    try:
        record = strict_canonical_json_loads(data)
    except (UnicodeError, ValueError) as exc:
        raise State3AuditError(f"invalid canonical JSON member: {name}") from exc
    if record.get("schema") != STATE3_MEMBER_SCHEMAS[name]:
        raise State3AuditError(f"schema mismatch for {name}")
    _assert_metadata_only(record)
    if name == "state3_audit.json":
        validate_state3_audit_record(record)
    elif name == "environment_runtime.json":
        validate_environment_runtime_record(record)
    elif name.startswith("storage_preflight_"):
        _validate_storage_record(record)
    elif name == "compute_feasibility_report.json":
        validate_compute_feasibility_report_metadata(record)
    elif name == "source_blob_verification.json":
        validate_source_blob_verification_record(record)
    elif name == "failure_inventory.json":
        validate_failure_inventory_record(record)
    elif name == "prohibited_action_counts.json":
        validate_prohibited_action_counts_record(record)
    return record


def _common_identity(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    authorization_key = (
        "authorization_file_sha256" if "authorization_file_sha256" in record
        else "state3_authorization_file_sha256"
    )
    return (
        record["protocol_file_sha256"], record[authorization_key],
        record["implementation_commit"], record["implementation_tree"],
    )


def _validate_cross_bindings(
    records: Mapping[str, Mapping[str, Any]], member_bytes: Mapping[str, bytes]
) -> None:
    root = records["state3_audit.json"]
    environment = records["environment_runtime.json"]
    before = records["storage_preflight_before.json"]
    report = records["compute_feasibility_report.json"]
    after = records["storage_preflight_after.json"]
    source = records["source_blob_verification.json"]
    failure = records["failure_inventory.json"]
    prohibited = records["prohibited_action_counts.json"]
    expected_identity = _common_identity(root)
    for name in (
        "environment_runtime.json", "storage_preflight_before.json",
        "compute_feasibility_report.json", "storage_preflight_after.json",
        "source_blob_verification.json",
    ):
        if _common_identity(records[name]) != expected_identity:
            raise State3AuditError(f"common identity binding mismatch: {name}")
    if (
        source["solver_spec_baseline_commit"] != report["solver_spec_baseline_commit"]
        or source["solver_spec_baseline_tree"] != report["solver_spec_baseline_tree"]
    ):
        raise State3AuditError("report/source solver baseline binding mismatch")
    links = {
        "environment_runtime.json": "environment_runtime_file_sha256",
        "storage_preflight_before.json": "storage_preflight_before_file_sha256",
        "compute_feasibility_report.json": "compute_feasibility_report_file_sha256",
        "storage_preflight_after.json": "storage_preflight_after_file_sha256",
        "source_blob_verification.json": "source_blob_verification_file_sha256",
        "failure_inventory.json": "failure_inventory_file_sha256",
        "prohibited_action_counts.json": "prohibited_action_counts_file_sha256",
    }
    observed = {name: hashlib.sha256(member_bytes[name]).hexdigest() for name in links}
    for name, field in links.items():
        if root[field] != observed[name]:
            raise State3AuditError(f"root file-hash binding mismatch: {name}")
    if (
        report["environment_record_sha256"] != observed["environment_runtime.json"]
        or report["storage_preflight_before_file_sha256"] != observed["storage_preflight_before.json"]
        or report["storage_preflight_after_file_sha256"] != observed["storage_preflight_after.json"]
    ):
        raise State3AuditError("report/member file-hash binding mismatch")
    if (
        report["launcher_file_sha256"] != environment["launcher_file_sha256"]
        or report["runtime_inspector_file_sha256"] != environment["runtime_inspector_file_sha256"]
        or environment["requirements_lock_git_blob"] != source["requirements_lock_git_blob"]
        or environment["requirements_lock_file_sha256"] != source["requirements_lock_file_sha256"]
    ):
        raise State3AuditError("environment/source/report code binding mismatch")
    _compare_storage_records(before, after)
    disk = report["disk_estimate"]
    for label, storage in (("before", before), ("after", after)):
        if (
            storage["estimated_accepted_bytes"] != disk["estimated_accepted_bytes"]
            or storage["projected_audit_bytes"] != disk["projected_audit_bytes"]
            or storage["required_free_bytes"] != disk["total_required_free_space_bytes"]
        ):
            raise State3AuditError(f"{label} storage/report disk estimate mismatch")
    if (
        root["git_before"] != source["git_before"] or root["git_after"] != source["git_after"]
        or report["git_before"] != source["git_before"] or report["git_after"] != source["git_after"]
    ):
        raise State3AuditError("audit/source/report Git binding mismatch")
    counts = prohibited["counts"]
    if (
        counts["forbidden_rng_call_count"] != report["forbidden_rng_call_count"]
        or counts["corpus_split_seed_access_count"] != report["forbidden_corpus_seed_access_count"]
    ):
        raise State3AuditError("prohibited-action/report count binding mismatch")
    if (
        failure["expected_record_count"] != 17
        or failure["original_relocated_record_count"] != 12
        or failure["in_place_preserved_record_count"] != 5
        or failure["historical_overbroad_pytest_prohibited_action_accounting"]
        != "not_provably_zero"
    ):
        raise State3AuditError("failure inventory count changed")


def build_member_hash_manifest(
    *, archive_filename: str, ordered_members: Sequence[tuple[str, bytes]],
) -> dict[str, Any]:
    if tuple(name for name, _ in ordered_members) != STATE3_AUDIT_MEMBER_ORDER[:-1]:
        raise State3AuditError("State 3 evidence member order is not frozen")
    if PurePosixPath(archive_filename).name != archive_filename or not archive_filename.endswith(".zip"):
        raise State3AuditError("archive filename binding must be one plain ZIP filename")
    return with_self_hash(
        {
            "archive_filename": archive_filename,
            "member_order": list(STATE3_AUDIT_MEMBER_ORDER),
            "members_excluding_manifest": [
                {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                for name, data in ordered_members
            ],
            "schema": MANIFEST_SCHEMA,
        },
        "manifest_self_sha256",
    )


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 0
    info.create_version = 20
    info.extract_version = 20
    info.flag_bits = 0
    info.volume = 0
    info.internal_attr = 0
    info.external_attr = FIXED_ZIP_EXTERNAL_ATTR
    info.extra = b""
    info.comment = b""
    return info


def _deterministic_zip_bytes(ordered_members: Sequence[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        archive.comment = b""
        for name, data in ordered_members:
            archive.writestr(_zip_info(name), data)
    return buffer.getvalue()


def _outside_private_corpus(path: Path, *, name: str) -> None:
    if "private_corpus_v1" in {part.casefold() for part in path.resolve(strict=False).parts}:
        raise State3AuditError(f"{name} must remain outside private_corpus_v1")


def package_state3_audit(source_directory: str | Path, destination_zip: str | Path) -> str:
    """Validate, cross-bind, and exclusively publish one deterministic ZIP."""
    source = Path(source_directory).resolve(strict=True)
    destination = Path(destination_zip).resolve(strict=False)
    _outside_private_corpus(source, name="State 3 evidence")
    _outside_private_corpus(destination, name="State 3 archive")
    if not source.is_dir():
        raise State3AuditError("evidence source must be one directory")
    if destination.suffix.casefold() != ".zip" or destination.exists():
        raise State3AuditError("destination must be one new ZIP path")
    if sorted(path.name for path in source.iterdir()) != sorted(STATE3_AUDIT_MEMBER_ORDER[:-1]):
        raise State3AuditError("source directory must contain exactly the frozen JSON members")
    ordered: list[tuple[str, bytes]] = []
    records: dict[str, Mapping[str, Any]] = {}
    member_bytes: dict[str, bytes] = {}
    for name in STATE3_AUDIT_MEMBER_ORDER[:-1]:
        _safe_member_name(name)
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise State3AuditError(f"audit member is not one regular file: {name}")
        data = path.read_bytes()
        records[name] = validate_state3_evidence_member(name, data)
        member_bytes[name] = data
        ordered.append((name, data))
    _validate_cross_bindings(records, member_bytes)
    manifest = build_member_hash_manifest(archive_filename=destination.name, ordered_members=ordered)
    archive_bytes = _deterministic_zip_bytes(
        (*ordered, (STATE3_AUDIT_MEMBER_ORDER[-1], canonical_json_bytes(manifest)))
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".__creating__")
    if temporary.exists():
        raise State3AuditError("archive staging path already exists")
    try:
        with temporary.open("xb") as stream:
            stream.write(archive_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != archive_bytes:
            raise State3AuditError("archive staging read-back mismatch")
        from .state3_audit_verify import verify_state3_audit
        verify_state3_audit(temporary, expected_archive_filename=destination.name)
        if destination.exists():
            raise State3AuditError("destination appeared before publication")
        if os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
        verify_state3_audit(destination, expected_archive_filename=destination.name)
    except BaseException:
        # Preserve the exclusive staging bytes as failed-attempt evidence.  A
        # later review decides their disposition; this writer never erases it.
        raise
    return file_sha256(destination)


def _binding_value(bindings: object, name: str) -> Any:
    if isinstance(bindings, Mapping):
        if name not in bindings:
            raise State3AuditError(f"preflight binding missing: {name}")
        return bindings[name]
    try:
        return getattr(bindings, name)
    except AttributeError as exc:
        raise State3AuditError(f"preflight binding missing: {name}") from exc


def finalize_state3_audit(
    *, evidence_directory: str | Path, destination_zip: str | Path,
    preflight_bindings: Mapping[str, Any], measurement_bundle: Mapping[str, Any],
    storage_preflight_before: Mapping[str, Any], storage_preflight_after: Mapping[str, Any],
    environment_runtime: Mapping[str, Any], source_blob_verification: Mapping[str, Any],
    failure_inventory: Mapping[str, Any], prohibited_action_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Finalize the audit after the measured bundle and post-storage record.

    This function has no launcher or worker callback and therefore cannot run
    the microbenchmark.  Its required order is encoded by accepting a complete
    measurement bundle plus the later storage record, then constructing the
    final report and archive from those already-existing metadata objects.
    """
    validate_compute_feasibility_preflight_bindings_metadata(preflight_bindings)
    validate_measurement_bundle_metadata(measurement_bundle)
    validate_environment_runtime_record(environment_runtime)
    _validate_storage_record(storage_preflight_before)
    _validate_storage_record(storage_preflight_after)
    _compare_storage_records(storage_preflight_before, storage_preflight_after)
    validate_source_blob_verification_record(source_blob_verification)
    validate_failure_inventory_record(failure_inventory)
    validate_prohibited_action_counts_record(prohibited_action_counts)

    names = (
        "protocol_file_sha256", "state3_authorization_file_sha256",
        "implementation_commit", "implementation_tree",
    )
    identity = tuple(_binding_value(preflight_bindings, name) for name in names)
    for label, record in (
        ("measurement bundle", measurement_bundle),
        ("environment/runtime", environment_runtime),
        ("source/blob", source_blob_verification),
    ):
        if tuple(record[name] for name in names) != identity:
            raise State3AuditError(f"{label} differs from preflight identity")
    for label, record in (
        ("before storage", storage_preflight_before),
        ("after storage", storage_preflight_after),
    ):
        observed = (
            record["protocol_file_sha256"], record["authorization_file_sha256"],
            record["implementation_commit"], record["implementation_tree"],
        )
        if observed != identity:
            raise State3AuditError(f"{label} differs from preflight identity")

    environment_bytes = canonical_json_bytes(dict(environment_runtime))
    before_bytes = canonical_json_bytes(dict(storage_preflight_before))
    after_bytes = canonical_json_bytes(dict(storage_preflight_after))
    if hashlib.sha256(environment_bytes).hexdigest() != _binding_value(
        preflight_bindings, "environment_record_sha256"
    ):
        raise State3AuditError("preflight environment file hash mismatch")
    if hashlib.sha256(before_bytes).hexdigest() != _binding_value(
        preflight_bindings, "storage_preflight_before_file_sha256"
    ):
        raise State3AuditError("preflight before-storage file hash mismatch")
    if _binding_value(preflight_bindings, "git_before") != source_blob_verification["git_before"]:
        raise State3AuditError("preflight Git-before binding mismatch")

    final_bindings = {
        "protocol_file_sha256": identity[0],
        "state3_authorization_file_sha256": identity[1],
        "solver_spec_baseline_commit": _binding_value(
            preflight_bindings, "solver_spec_baseline_commit"
        ),
        "solver_spec_baseline_tree": _binding_value(
            preflight_bindings, "solver_spec_baseline_tree"
        ),
        "implementation_commit": identity[2],
        "implementation_tree": identity[3],
        "launcher_file_sha256": _binding_value(preflight_bindings, "launcher_file_sha256"),
        "runtime_inspector_file_sha256": _binding_value(
            preflight_bindings, "runtime_inspector_file_sha256"
        ),
        "environment_record_sha256": hashlib.sha256(environment_bytes).hexdigest(),
        "storage_preflight_before_file_sha256": hashlib.sha256(before_bytes).hexdigest(),
        "storage_preflight_after_file_sha256": hashlib.sha256(after_bytes).hexdigest(),
        "disk_estimate": _binding_value(preflight_bindings, "disk_estimate"),
        "git_before": source_blob_verification["git_before"],
        "git_after": source_blob_verification["git_after"],
    }
    report = build_compute_feasibility_report_metadata(
        final_bindings, measurement_bundle["raw_measurements"]
    )
    records: dict[str, Mapping[str, Any]] = {
        "environment_runtime.json": environment_runtime,
        "storage_preflight_before.json": storage_preflight_before,
        "compute_feasibility_report.json": report,
        "storage_preflight_after.json": storage_preflight_after,
        "source_blob_verification.json": source_blob_verification,
        "failure_inventory.json": failure_inventory,
        "prohibited_action_counts.json": prohibited_action_counts,
    }
    member_bytes = {name: canonical_json_bytes(dict(record)) for name, record in records.items()}
    root = build_state3_audit_record(
        protocol_file_sha256=identity[0], state3_authorization_file_sha256=identity[1],
        implementation_commit=identity[2], implementation_tree=identity[3],
        measurement_bundle_self_sha256=measurement_bundle["measurement_bundle_self_sha256"],
        evidence_file_sha256={
            name: hashlib.sha256(data).hexdigest() for name, data in member_bytes.items()
        },
        git_before=source_blob_verification["git_before"],
        git_after=source_blob_verification["git_after"],
    )
    records["state3_audit.json"] = root
    source = Path(evidence_directory).resolve(strict=False)
    _outside_private_corpus(source, name="State 3 evidence")
    if source.exists():
        raise State3AuditError("State 3 evidence directory already exists")
    source.mkdir(parents=True, exist_ok=False)
    for name in STATE3_AUDIT_MEMBER_ORDER[:-1]:
        write_canonical_json_exclusive(source / name, records[name])
    archive_sha256 = package_state3_audit(source, destination_zip)
    from .state3_audit_verify import verify_state3_audit
    verification = verify_state3_audit(
        destination_zip, expected_archive_filename=Path(destination_zip).name
    )
    if verification["archive_sha256"] != archive_sha256:
        raise State3AuditError("independent archive hash result changed")
    return {
        "archive_sha256": archive_sha256,
        "archive_size": Path(destination_zip).stat().st_size,
        "member_order": list(STATE3_AUDIT_MEMBER_ORDER),
        "verification": verification,
    }


def canonical_evidence_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create a generic metadata-only self-hashed object for administrative use."""
    _assert_metadata_only(payload)
    return with_self_hash(payload, "record_self_sha256")


_METADATA_FORBIDDEN_MODULE_PREFIXES: Final = (
    "numpy", "scipy", "operatormark.corpus.microbenchmark",
    "operatormark.corpus.rng_guard", "operatormark.corpus.state4",
    "operatormark.benchmarks", "operatormark.solvers", "operatormark.evaluation",
)


def _assert_metadata_import_boundary() -> None:
    loaded = sorted(
        name for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".")
               for prefix in _METADATA_FORBIDDEN_MODULE_PREFIXES)
    )
    if loaded:
        raise State3AuditError(
            "metadata-only audit finalizer has a forbidden import closure: " + ",".join(loaded)
        )


def _administrative_relative_path(value: str, *, name: str) -> PurePosixPath:
    text = _require_relative_path(value, name=name)
    path = PurePosixPath(text)
    if not path.parts or path.parts[0] != "artifacts":
        raise State3AuditError(f"{name} must be under the ignored artifacts directory")
    if "private_corpus_v1" in {part.casefold() for part in path.parts}:
        raise State3AuditError(f"{name} must remain outside private_corpus_v1")
    return path


def _resolved_administrative_path(
    root: Path, value: str, *, name: str, expected_suffix: str | None,
) -> Path:
    relative = _administrative_relative_path(value, name=name)
    if expected_suffix is not None and relative.suffix.casefold() != expected_suffix:
        raise State3AuditError(f"{name} has the wrong suffix")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise State3AuditError(f"{name} parent cannot be a symbolic link")
    return root.joinpath(*relative.parts)


def _load_cli_record(root: Path, value: str, *, name: str) -> dict[str, Any]:
    path = _resolved_administrative_path(root, value, name=name, expected_suffix=".json")
    if path.is_symlink() or not path.is_file():
        raise State3AuditError(f"{name} is not one existing regular JSON file")
    try:
        return strict_canonical_json_loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise State3AuditError(f"{name} is malformed or noncanonical") from exc


def _git_process(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="strict",
    )


def _git_output(root: Path, *arguments: str) -> str:
    result = _git_process(root, *arguments)
    if result.returncode != 0 or result.stderr:
        raise State3AuditError("audit-finalizer Git query failed: " + " ".join(arguments))
    return result.stdout.rstrip("\r\n")


def _validate_live_repository_state(
    root: Path, implementation_commit: str, implementation_tree: str,
) -> None:
    if _git_output(root, "symbolic-ref", "--quiet", "--short", "HEAD") != "main":
        raise State3AuditError("audit-finalizer branch is not main")
    if _git_output(root, "rev-parse", "HEAD") != implementation_commit:
        raise State3AuditError("audit-finalizer implementation commit mismatch")
    if _git_output(root, "rev-parse", "HEAD^{tree}") != implementation_tree:
        raise State3AuditError("audit-finalizer implementation tree mismatch")
    for arguments in (("diff", "--quiet", "--"), ("diff", "--cached", "--quiet", "--")):
        result = _git_process(root, *arguments)
        if result.returncode != 0 or result.stdout or result.stderr:
            raise State3AuditError("audit-finalizer found a tracked Git change")
    if _git_output(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise State3AuditError("audit-finalizer found a non-clean Git status")


def _require_ignored_path(root: Path, relative: str) -> None:
    result = _git_process(root, "check-ignore", "--no-index", "--quiet", "--", relative)
    if result.returncode != 0 or result.stdout or result.stderr:
        raise State3AuditError("audit-finalizer output is not covered by a repository ignore rule")


def _audit_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Metadata-only OperatorMark State 3 audit-finalizer entry point"
    )
    parser.add_argument("--authorization-file-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--implementation-tree", required=True)
    parser.add_argument("--preflight-bindings", required=True)
    parser.add_argument("--measurement-bundle", required=True)
    parser.add_argument("--storage-before", required=True)
    parser.add_argument("--storage-after", required=True)
    parser.add_argument("--environment-runtime", required=True)
    parser.add_argument("--source-blob-verification", required=True)
    parser.add_argument("--failure-inventory", required=True)
    parser.add_argument("--prohibited-action-counts", required=True)
    parser.add_argument("--evidence-directory", required=True)
    parser.add_argument("--destination-zip", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    finalizer: Callable[..., Mapping[str, Any]] | None = None,
    repository_root: str | Path | None = None,
    enforce_import_boundary: bool = True,
    git_validator: Callable[[Path, str, str], None] | None = None,
    ignore_validator: Callable[[Path, str], None] | None = None,
    failure_binding_validator: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> int:
    """Finalize already-created metadata through the committed administrative route.

    Injectable callbacks are available only to State 2 fixed-fixture tests.  A
    command-line invocation always uses live read-only Git validation and the
    committed finalizer.
    """

    if enforce_import_boundary:
        _assert_metadata_import_boundary()
    arguments = _audit_cli_parser().parse_args(argv)
    _require_sha256(arguments.authorization_file_sha256, name="authorization_file_sha256")
    _require_git_object(arguments.implementation_commit, name="implementation_commit")
    _require_git_object(arguments.implementation_tree, name="implementation_tree")
    root = (
        Path.cwd().resolve(strict=True)
        if repository_root is None else Path(repository_root).resolve(strict=True)
    )
    if not (root / ".git").exists():
        raise State3AuditError("audit-finalizer working directory is not the repository root")
    validate_git = _validate_live_repository_state if git_validator is None else git_validator
    validate_git(root, arguments.implementation_commit, arguments.implementation_tree)

    paths = {
        "preflight_bindings": arguments.preflight_bindings,
        "measurement_bundle": arguments.measurement_bundle,
        "storage_preflight_before": arguments.storage_before,
        "storage_preflight_after": arguments.storage_after,
        "environment_runtime": arguments.environment_runtime,
        "source_blob_verification": arguments.source_blob_verification,
        "failure_inventory": arguments.failure_inventory,
        "prohibited_action_counts": arguments.prohibited_action_counts,
    }
    records = {
        name: _load_cli_record(root, value, name=name) for name, value in paths.items()
    }
    validate_failures = (
        verify_failure_inventory_physical_bindings
        if failure_binding_validator is None
        else failure_binding_validator
    )
    validate_failures(root, records["failure_inventory"])
    preflight = records["preflight_bindings"]
    if (
        preflight.get("state3_authorization_file_sha256")
        != arguments.authorization_file_sha256
        or preflight.get("implementation_commit") != arguments.implementation_commit
        or preflight.get("implementation_tree") != arguments.implementation_tree
    ):
        raise State3AuditError("audit-finalizer runtime binding differs from preflight")
    evidence_relative = _administrative_relative_path(
        arguments.evidence_directory, name="evidence_directory"
    ).as_posix()
    archive_relative = _administrative_relative_path(
        arguments.destination_zip, name="destination_zip"
    ).as_posix()
    evidence_directory = _resolved_administrative_path(
        root, evidence_relative, name="evidence_directory", expected_suffix=None
    )
    destination_zip = _resolved_administrative_path(
        root, archive_relative, name="destination_zip", expected_suffix=".zip"
    )
    validate_ignore = _require_ignored_path if ignore_validator is None else ignore_validator
    validate_ignore(root, evidence_relative)
    validate_ignore(root, archive_relative)
    run_finalizer = finalize_state3_audit if finalizer is None else finalizer
    result = dict(run_finalizer(
        evidence_directory=evidence_directory,
        destination_zip=destination_zip,
        **records,
    ))
    validate_git(root, arguments.implementation_commit, arguments.implementation_tree)
    if enforce_import_boundary:
        _assert_metadata_import_boundary()
    receipt = {
        "archive_sha256": result.get("archive_sha256"),
        "archive_size": result.get("archive_size"),
        "authorization_file_sha256": arguments.authorization_file_sha256,
        "destination_zip": archive_relative,
        "implementation_commit": arguments.implementation_commit,
        "implementation_tree": arguments.implementation_tree,
        "member_order": result.get("member_order"),
        "schema": "operatormark.darcy_branch_c_v1.state3_audit_finalizer_receipt.v1",
        "verification": result.get("verification"),
    }
    _assert_metadata_only(receipt)
    sys.stdout.write(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
