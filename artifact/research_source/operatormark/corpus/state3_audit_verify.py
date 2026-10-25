"""Independent byte-level verifier for metadata-only State 3 audit ZIPs.

The verifier intentionally duplicates the archive, record, report, storage,
failure-inventory, and cross-binding rules.  It does not import the writer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import statistics
import struct
import sys
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final


PROTOCOL_SHA256: Final = "a38ef79b6c4bdccf53f599fa08f77e48a111afbf0b2783ad5733bda99996599a"
BASELINE_COMMIT: Final = "e16456395cfe67b73b9d298fd0491cf9524d798a"
BASELINE_TREE: Final = "81f40e24102e51e0da3ce65dafd769a7678565fd"
STATE: Final = "state3_pre_generation_audit_no_corpus_controls"
OUTCOME: Final = "complete_pending_independent_review_and_separate_state4_authorization"
CLASSIFICATION: Final = "private_confidential_metadata_only_external_sharing_prohibited"
EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()
MEMBER_ORDER: Final = (
    "state3_audit.json", "environment_runtime.json", "storage_preflight_before.json",
    "compute_feasibility_report.json", "storage_preflight_after.json",
    "source_blob_verification.json", "failure_inventory.json",
    "prohibited_action_counts.json", "member_hash_manifest.json",
)
MEMBER_SCHEMAS: Final = {
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
FIXED_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
FIXED_EXTERNAL_ATTR: Final = 0x20
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE: Final = re.compile(r"^[0-9a-f]{40}$")
ABSOLUTE_PATH_RE: Final = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|//|/)")

GIT_KEYS: Final = {
    "branch", "head", "tree", "parents", "status_short_untracked_all",
    "tracked_diff_sha256", "tracked_diff_size_bytes", "staged_diff_sha256",
    "staged_diff_size_bytes", "tracked_changes_empty", "staged_changes_empty",
    "untracked_inclusive_status_empty",
}
REPORT_KEYS: Final = {
    "schema", "state", "protocol_file_sha256", "state3_authorization_file_sha256",
    "solver_spec_baseline_commit", "solver_spec_baseline_tree", "implementation_commit",
    "implementation_tree", "launcher_file_sha256", "runtime_inspector_file_sha256",
    "environment_record_sha256", "storage_preflight_before_file_sha256",
    "storage_preflight_after_file_sha256", "fixture_definitions", "warmup_count_per_fixture",
    "measured_count_per_fixture", "measurement_order", "timing_clock", "memory_counter",
    "rng_state_commitment_specification", "raw_measurements", "summary",
    "corpus_sample_count", "frozen_safety_factor", "extrapolation", "disk_estimate",
    "all_fixture_gates_passed", "all_rng_state_equalities_passed",
    "forbidden_rng_call_count", "forbidden_corpus_seed_access_count", "git_before",
    "git_after", "report_self_sha256",
}
RAW_KEYS: Final = {
    "fixture", "repeat_index", "child_exit_code", "tpfa_assembly_ns", "tpfa_solve_ns",
    "tpfa_diagnostics_ns", "p1_assembly_ns", "p1_solve_ns", "p1_diagnostics_ns",
    "p1_projection_ns", "paired_solver_ns", "child_wall_ns", "rss_counter_name",
    "baseline_rss_bytes", "peak_rss_bytes", "incremental_peak_rss_bytes",
    "rng_state_commitments", "rng_state_equalities", "forbidden_rng_call_count",
    "forbidden_corpus_seed_access_count", "tpfa_diagnostics", "p1_diagnostics",
    "cross_solver_metrics", "gates_passed",
}
TIMING_FIELDS: Final = (
    "tpfa_assembly_ns", "tpfa_solve_ns", "tpfa_diagnostics_ns", "p1_assembly_ns",
    "p1_solve_ns", "p1_diagnostics_ns", "p1_projection_ns", "paired_solver_ns", "child_wall_ns",
)
MEMORY_FIELDS: Final = ("baseline_rss_bytes", "peak_rss_bytes", "incremental_peak_rss_bytes")
SUMMARY_FIELDS: Final = TIMING_FIELDS + MEMORY_FIELDS
FIXTURES: Final = ("constant_half", "vertical_jump", "checkerboard")
RNG_COMMITMENT_KEYS: Final = {
    "numpy_legacy_after_base_import_sha256", "python_random_after_base_import_sha256",
    "numpy_legacy_after_guard_install_sha256", "python_random_after_guard_install_sha256",
    "numpy_legacy_after_project_import_sha256", "python_random_after_project_import_sha256",
    "numpy_legacy_after_execution_sha256", "python_random_after_execution_sha256",
}
RNG_EQUALITY_KEYS: Final = {
    "numpy_legacy_after_guard_install_equals_baseline",
    "python_random_after_guard_install_equals_baseline",
    "numpy_legacy_after_project_import_equals_baseline",
    "python_random_after_project_import_equals_baseline",
    "numpy_legacy_after_execution_equals_baseline",
    "python_random_after_execution_equals_baseline",
}
TPFA_KEYS: Final = {
    "algebraic_backward_error", "boundary_transmissibility_error",
    "maximum_local_conservation_ratio", "maximum_interior_face_antisymmetry",
    "global_balance_ratio", "rhs_relative_l2_residual", "passes",
}
P1_KEYS: Final = {
    "algebraic_backward_error", "boundary_reconstruction_error", "reaction_denominator",
    "reaction_balance_ratio", "outward_boundary_flux", "integrated_source",
    "rhs_relative_l2_residual", "iterations", "passes",
}
DISK_KEYS: Final = {
    "raw_seven_tensor_payload_bytes", "projected_npy_header_bytes", "projected_jsonl_bytes",
    "projected_manifest_and_audit_bytes", "controls_only_bytes", "estimated_accepted_bytes",
    "active_staging_bytes", "readback_atomic_publication_bytes",
    "preserved_failed_attempt_allowance_bytes", "projected_audit_bytes",
    "fixed_headroom_bytes", "minimum_floor_bytes", "four_copy_formula_bytes",
    "total_required_free_space_bytes",
}
ROOT_KEYS: Final = {
    "schema", "state", "outcome", "classification", "protocol_file_sha256",
    "state3_authorization_file_sha256", "solver_spec_baseline_commit", "solver_spec_baseline_tree",
    "implementation_commit", "implementation_tree", "measurement_bundle_self_sha256",
    "environment_runtime_file_sha256", "storage_preflight_before_file_sha256",
    "compute_feasibility_report_file_sha256", "storage_preflight_after_file_sha256",
    "source_blob_verification_file_sha256", "failure_inventory_file_sha256",
    "prohibited_action_counts_file_sha256", "git_before", "git_after",
    "storage_preflight_records_match", "contains_corpus_controls_or_solutions",
    "contains_scientific_arrays", "external_sharing_authorized", "state4_or_later_authorized",
    "all_checks_passed", "record_self_sha256",
}
ENV_KEYS: Final = {
    "schema", "state", "protocol_file_sha256", "state3_authorization_file_sha256",
    "implementation_commit", "implementation_tree", "python_version", "numpy_version",
    "scipy_version", "operating_system", "architecture", "locale", "parent_environment",
    "parent_environment_commitment", "applicable_additional_thread_variables",
    "launcher_file_sha256", "runtime_inspector_file_sha256", "requirements_lock_path",
    "requirements_lock_git_blob", "requirements_lock_file_sha256", "base_import_closure",
    "base_import_closure_sha256", "numerical_runtime", "numerical_runtime_sha256",
    "all_environment_checks_passed", "all_runtime_checks_passed", "record_self_sha256",
}
SOURCE_KEYS: Final = {
    "schema", "state", "protocol_file_sha256", "state3_authorization_file_sha256",
    "solver_spec_baseline_commit", "solver_spec_baseline_tree", "implementation_commit",
    "implementation_tree", "implementation_parent_commit", "requirements_lock_path",
    "requirements_lock_git_blob", "requirements_lock_file_sha256", "frozen_baseline_blobs",
    "implementation_files", "all_frozen_baseline_blobs_byte_identical",
    "all_implementation_files_match_git", "git_before", "git_after", "record_self_sha256",
}
STORAGE_KEYS: Final = {
    "schema", "state", "protocol_file_sha256", "authorization_file_sha256",
    "implementation_commit", "implementation_tree",
    "repository_root_commitment",
    "destination_relative_path", "destination_absolute_path_commitment",
    "canonical_resolution_method", "path_component_evidence", "volume_evidence",
    "sync_detection_evidence", "git_ignore_evidence", "git_tracking_evidence",
    "permission_evidence", "encryption_evidence", "backup_and_indexing_evidence",
    "probe_evidence", "estimated_accepted_bytes", "projected_audit_bytes",
    "required_free_bytes", "observed_free_bytes", "residual_risks",
    "all_mandatory_checks_passed", "record_self_sha256",
}
PATH_COMPONENT_KEYS: Final = {
    "relative_component", "exists", "is_directory", "file_attributes_hex",
    "is_reparse_point", "reparse_tag_hex", "canonical_path_commitment",
    "volume_identifier_commitment", "file_identity_commitment", "canonical_name_exact",
    "unicode_nfc_exact", "same_volume_as_repository_root", "stable_before_after",
}
VOLUME_KEYS: Final = {
    "allocation_unit_bytes", "capacity_bytes", "free_bytes", "drive_type", "filesystem_type",
    "filesystem_flags_hex", "is_fixed_local", "maximum_component_length",
    "path_length_capability", "case_sensitivity_behavior", "volume_identifier_commitment",
    "volume_root_commitment",
}
SYNC_KEYS: Final = {
    "inspection_complete", "environment_names_checked", "registry_sources_checked",
    "configured_sync_roots", "active_sync_clients", "destination_in_detected_sync_root",
    "cloud_attribute_components",
}
GIT_IGNORE_KEYS: Final = {
    "candidate_count", "matched", "intended_rule", "all_candidates_matched_intended_rule",
    "candidate_path_commitments", "check_ignore_output_sha256",
}
GIT_TRACKING_KEYS: Final = {
    "tracked_path_count", "submodule_path_count", "nested_worktree_count",
    "nested_repository_marker_count", "candidate_collision_count", "worktree_evidence_sha256",
    "branch", "head_commit", "head_tree", "parent_commit", "tracked_status_clean",
    "staged_status_clean", "untracked_inclusive_status_clean", "status_porcelain_sha256",
}
PERMISSION_KEYS: Final = {
    "platform", "inspection_method", "security_descriptor_commitment",
    "owning_principal_commitment", "dacl_present", "inheritance_enabled",
    "current_process_effective_rights", "broad_principal_effective_access",
    "additional_allow_principal_effective_access", "additional_allow_principal_count",
    "blocking_broad_access_count", "broader_local_user_residual_count",
}
ENCRYPTION_KEYS: Final = {
    "filesystem_supports_encryption", "encrypted_ancestor_count",
    "device_encryption_query_outcome", "device_encryption_interpreted_status",
    "device_encryption_evidence_commitment",
    "encryption_presence_establishes_legal_or_cryptographic_protection", "status",
}
BACKUP_KEYS: Final = {
    "administrator_access_may_expose_copies", "backup_software_may_expose_copies",
    "indexing_may_expose_copies", "antivirus_may_expose_copies",
    "crash_dumps_may_expose_copies", "paging_may_expose_copies",
    "local_sync_clients_may_expose_copies", "service_query_evidence_commitments",
    "service_query_outcomes", "registry_query_evidence_commitment", "registry_query_outcome",
    "administrator_backup_malware_physical_access_and_remanence_eliminated", "status",
}
PROBE_KEYS: Final = {
    "fixed_literal_sha256", "operations", "probe_file_preserved", "probe_directory_preserved",
    "same_volume_verified", "atomic_directory_rename_verified",
    "source_volume_identifier_commitment", "renamed_volume_identifier_commitment", "passed",
}
SYNC_ROOT_KEYS: Final = {
    "provider", "source", "exists", "handle_canonicalized",
    "lexical_root_commitment", "canonical_root_commitment",
}
SYNC_CLIENT_KEYS: Final = {"provider", "process_name_commitment", "configured_root_count"}
RIGHTS_KEYS: Final = {"read", "write", "create", "rename", "delete"}
PRINCIPAL_KEYS: Final = {
    "principal_class", "principal_commitment", "effective_mask_hex", "effective_read",
    "effective_write", "blocking", "residual_risk",
}
BROAD_PRINCIPALS: Final = (
    ("everyone", "S-1-1-0", True), ("anonymous", "S-1-5-7", True),
    ("network", "S-1-5-2", True), ("service", "S-1-5-6", True),
    ("local_service", "S-1-5-19", True), ("network_service", "S-1-5-20", True),
    ("guests", "S-1-5-32-546", True), ("authenticated_users", "S-1-5-11", False),
    ("builtin_users", "S-1-5-32-545", False),
)
ACL_READ_MASK: Final = 0x1 | 0x80 | 0x20000 | 0x80000000 | 0x10000000
ACL_WRITE_MASK: Final = (
    0x2 | 0x4 | 0x10 | 0x40 | 0x10000 | 0x40000 | 0x80000 | 0x40000000 | 0x10000000
)
REPARSE_ATTRIBUTE: Final = 0x00000400
ENCRYPTED_ATTRIBUTE: Final = 0x00004000
NOT_CONTENT_INDEXED_ATTRIBUTE: Final = 0x00002000
CLOUD_ATTRIBUTES: Final = (
    0x00001000 | 0x00040000 | 0x00080000 | 0x00100000
    | 0x00400000 | 0x00800000 | 0x01000000
)
DESTINATION_RELATIVE: Final = "artifacts/darcy_branch_c_v1/private_corpus_v1/"
PROBE_LITERAL_SHA256: Final = hashlib.sha256(
    b"OperatorMark private storage preflight v1\r\n"
).hexdigest()
PROBE_OPERATIONS: Final = [
    "create_directory", "create_file", "write", "flush", "fsync", "read",
    "verify_rename_target_absent", "atomic_directory_rename_no_replace", "read",
    "delete_file", "delete_directories",
]
SYNC_ENVIRONMENT_NAMES: Final = [
    "OneDrive", "OneDriveCommercial", "OneDriveConsumer", "Dropbox",
    "GoogleDrive", "GoogleDriveFS", "iCloudDrive", "Box", "SharePoint",
]

MANDATORY_PARENT_ENVIRONMENT: Final = {
    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "GOTO_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1", "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE",
}
ADDITIONAL_THREAD_VARIABLES: Final = {
    "openblas": ("OPENBLAS_DEFAULT_NUM_THREADS",),
    "mkl": ("MKL_DOMAIN_NUM_THREADS", "MKL_THREADING_LAYER"),
    "openmp": ("OMP_MAX_ACTIVE_LEVELS", "OMP_NESTED", "OMP_PLACES", "OMP_PROC_BIND", "OMP_WAIT_POLICY"),
    "intel_openmp": ("KMP_AFFINITY", "KMP_BLOCKTIME", "KMP_HW_SUBSET"),
    "tbb": ("TBB_NUM_THREADS",),
}
BASE_CLOSURE_KEYS: Final = {
    "closure_sha256", "numpy_version", "ordered_modules", "root_modules", "schema",
    "scipy_version",
}
BASE_MODULE_KEYS: Final = {"file_sha256", "kind", "module", "suffix"}
BASE_IMPORT_ROOTS: Final = [
    "numpy", "numpy.typing", "scipy", "scipy.sparse", "scipy.sparse.linalg",
]
RUNTIME_KEYS: Final = {
    "active_backend_kinds", "active_backends", "all_active_backends_inspectable",
    "all_effective_thread_counts_one", "all_parent_thread_policy_bound",
    "all_runtime_requirements_satisfied", "applicable_additional_thread_variables",
    "bound_applicable_thread_policy", "missing_applicable_thread_policy_variables",
    "non_single_thread_backend_binary_sha256", "numerical_wrappers", "runtime_record_sha256",
    "schema", "unexpected_additional_thread_policy_variables",
    "uninspectable_backend_binary_sha256",
}
ACTIVE_BACKEND_KEYS: Final = {
    "architecture", "backend", "binary_sha256", "effective_thread_count",
    "inspection_complete", "inspection_symbols", "path_commitment", "role",
    "runtime_configuration", "threading_layer", "version",
}
WRAPPER_KEYS: Final = {"architecture", "binary_sha256", "interface", "path_commitment", "role"}
INSPECTION_SYMBOL_KEYS: Final = {
    "effective_thread_count", "runtime_configuration", "threading_layer",
}
FROZEN_BASELINE_REQUIRED_PATHS: Final = {
    "configs/darcy_branch_c_v1.yaml", "operatormark/benchmarks/darcy_spec.py",
    "operatormark/evaluation/darcy_cross_solver_tolerance.py",
    "operatormark/evaluation/darcy_solver_heldout.py", "operatormark/solvers/darcy_p1.py",
    "operatormark/solvers/darcy_tpfa.py", "reports/darcy_branch_c_v1_preregistration.md",
}
FROZEN_REVIEW_HASHES: Final = {
    "configs/darcy_branch_c_v1.yaml": "b24da08f028c52c2f275a49a07a971efcf42ad63962412ddda81ce81fa6e242d",
    "reports/darcy_branch_c_v1_preregistration.md": "0ae56c37e4f158e6038b80fa2bdbebfb0f72f8ba6911ec51889b6d3321b0ff17",
}
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
FAILURE_ORIGINAL_PREFIX: Final = "artifacts/darcy_branch_c_v1/private_corpus_v1/state3_audits"
FAILURE_DESTINATION_PREFIX: Final = "artifacts/darcy_branch_c_v1/state3_preflight_failures_20260903"
DRIVER_PATH: Final = "artifacts/state3-recovery-20260903/state3-driver.ps1"
DRIVER_SHA256: Final = "07596fd9a0f845e1f70710521f5e6bf6ddaf603c9832c83bae5eed9be48abde5"
MANIFEST_KEYS: Final = {
    "schema", "archive_filename", "member_order", "members_excluding_manifest",
    "manifest_self_sha256",
}
MANIFEST_ROW_KEYS: Final = {"path", "size", "sha256"}
PROHIBITED_COUNT_NAMES: Final = {
    "legacy_or_prohibited_data_access_count", "development_or_solver_held_out_data_access_count",
    "corpus_split_seed_access_count", "corpus_control_generation_count", "dataset_generation_count",
    "model_checkpoint_prediction_or_watermark_access_count", "training_count",
    "dependency_change_count", "network_access_count", "publication_count", "push_count",
    "external_sharing_count", "repository_visibility_change_count",
    "review_only_driver_execution_count", "state4_or_later_activity_count",
    "forbidden_rng_call_count",
}
FORBIDDEN_KEYS: Final = {
    "control", "controls", "control_values", "corpus_seed", "development_seed",
    "held_out_seed", "rho", "solution", "solution_array", "tensor_values", "training_seed",
    "validation_seed", "test_seed", "username", "sid", "credential", "credentials",
    "account_identifier",
}


class State3AuditVerificationError(RuntimeError):
    """Raised when independently inspected evidence is not exact."""


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise State3AuditVerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=False,
                      separators=(",", ":"), sort_keys=True).encode("utf-8")


def _load_canonical(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                State3AuditVerificationError(f"non-finite JSON token: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise State3AuditVerificationError("invalid UTF-8 JSON member") from exc
    if not isinstance(value, dict) or data != _canonical(value):
        raise State3AuditVerificationError("archive JSON is not canonical")
    return value


def _exact(value: object, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise State3AuditVerificationError(f"{name} keys mismatch")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise State3AuditVerificationError(f"invalid SHA-256: {name}")
    return value


def _oid(value: object, name: str) -> str:
    if not isinstance(value, str) or GIT_OID_RE.fullmatch(value) is None:
        raise State3AuditVerificationError(f"invalid Git object ID: {name}")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise State3AuditVerificationError(f"invalid non-negative integer: {name}")
    return value


def _validate_self_hash(record: Mapping[str, Any], field: str) -> None:
    claimed = _sha(record.get(field), field)
    payload = dict(record)
    del payload[field]
    if hashlib.sha256(_canonical(payload)).hexdigest() != claimed:
        raise State3AuditVerificationError(f"self-hash mismatch: {field}")


def _metadata_only(value: object, trail: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key.casefold() in FORBIDDEN_KEYS:
                raise State3AuditVerificationError("prohibited payload field: " + ".".join((*trail, key)))
            _metadata_only(child, (*trail, key))
    elif isinstance(value, list):
        if len(value) > 64 and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        ):
            raise State3AuditVerificationError("raw numerical array in metadata-only evidence")
        for index, child in enumerate(value):
            _metadata_only(child, (*trail, str(index)))
    elif isinstance(value, str) and ABSOLUTE_PATH_RE.match(value):
        raise State3AuditVerificationError("absolute path in metadata-only evidence")


def _validate_git(record: object, commit: str, tree: str) -> Mapping[str, Any]:
    value = _exact(record, GIT_KEYS, "Git record")
    _oid(value["head"], "Git head")
    _oid(value["tree"], "Git tree")
    parents = value["parents"]
    if not isinstance(parents, list) or not parents or len(set(parents)) != len(parents):
        raise State3AuditVerificationError("Git parents invalid")
    for parent in parents:
        _oid(parent, "Git parent")
    for name in ("tracked_diff_sha256", "staged_diff_sha256"):
        _sha(value[name], name)
    if (
        value["branch"] != "main" or value["head"] != commit or value["tree"] != tree
        or value["status_short_untracked_all"] != ""
        or value["tracked_diff_sha256"] != EMPTY_SHA256
        or value["staged_diff_sha256"] != EMPTY_SHA256
        or value["tracked_diff_size_bytes"] != 0 or value["staged_diff_size_bytes"] != 0
        or value["tracked_changes_empty"] is not True or value["staged_changes_empty"] is not True
        or value["untracked_inclusive_status_empty"] is not True
    ):
        raise State3AuditVerificationError("Git record is not clean/frozen")
    return value


def _commit(domain: str, authorization_hash: str, value: str) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\n" + authorization_hash.encode("ascii")
        + b"\n" + value.encode("utf-8")
    ).hexdigest()


def _relative_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise State3AuditVerificationError(f"invalid relative path: {name}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise State3AuditVerificationError(f"noncanonical relative path: {name}")
    return value


def _validate_base_closure(record: object) -> Mapping[str, Any]:
    value = _exact(record, BASE_CLOSURE_KEYS, "base-import closure")
    if (
        value["schema"] != "operatormark.darcy_branch_c_v1.base_import_closure.v1"
        or value["numpy_version"] != "2.5.2" or value["scipy_version"] != "1.18.1"
        or value["root_modules"] != BASE_IMPORT_ROOTS
    ):
        raise State3AuditVerificationError("base-import closure identity changed")
    modules = value["ordered_modules"]
    if not isinstance(modules, list) or not modules:
        raise State3AuditVerificationError("base-import closure is empty")
    names: list[str] = []
    for item in modules:
        row = _exact(item, BASE_MODULE_KEYS, "base-import module")
        name = row["module"]
        if not isinstance(name, str) or not (
            name in {"numpy", "scipy"} or name.startswith(("numpy.", "scipy."))
        ):
            raise State3AuditVerificationError("non-numerical base-import module")
        names.append(name)
        suffix = row["suffix"]
        if not isinstance(suffix, str) or suffix != suffix.lower():
            raise State3AuditVerificationError("noncanonical base-import suffix")
        if row["kind"] == "source_module":
            if not any(suffix.endswith(expected) for expected in {".py", ".pyi"}):
                raise State3AuditVerificationError("invalid numerical source suffix")
            _sha(row["file_sha256"], "base-import source")
        elif row["kind"] == "extension_binary":
            if not any(
                suffix.endswith(expected) for expected in {".pyd", ".so", ".dll", ".dylib"}
            ):
                raise State3AuditVerificationError("invalid numerical extension suffix")
            _sha(row["file_sha256"], "base-import extension")
        elif row["kind"] == "namespace_package_no_module_bytes":
            if row["file_sha256"] is not None or suffix != "":
                raise State3AuditVerificationError("namespace closure row has byte identity")
        else:
            raise State3AuditVerificationError("invalid base-import module kind")
    if names != sorted(set(names)) or not set(BASE_IMPORT_ROOTS).issubset(names):
        raise State3AuditVerificationError("base-import closure coverage/order mismatch")
    payload = {key: value[key] for key in BASE_CLOSURE_KEYS - {"closure_sha256"}}
    if _sha(value["closure_sha256"], "base-import closure") != hashlib.sha256(
        _canonical(payload)
    ).hexdigest():
        raise State3AuditVerificationError("base-import closure self-commitment mismatch")
    return value


def _recognized_thread_variables(
    backends: Sequence[str], threading_layers: Sequence[str],
) -> list[str]:
    result: set[str] = set()
    for backend in backends:
        result.update(ADDITIONAL_THREAD_VARIABLES.get(backend, ()))
        if backend == "intel_openmp":
            result.update(ADDITIONAL_THREAD_VARIABLES["openmp"])
    if "openmp" in threading_layers:
        result.update(ADDITIONAL_THREAD_VARIABLES["openmp"])
    return sorted(result)


def _validate_runtime(record: object) -> Mapping[str, Any]:
    value = _exact(record, RUNTIME_KEYS, "numerical runtime")
    if value["schema"] != "operatormark.darcy_branch_c_v1.numerical_runtime.v2":
        raise State3AuditVerificationError("numerical runtime schema mismatch")
    active = value["active_backends"]
    wrappers = value["numerical_wrappers"]
    if not isinstance(active, list) or not isinstance(wrappers, list):
        raise State3AuditVerificationError("runtime libraries are not arrays")
    backend_names = {"openblas", "mkl", "intel_openmp", "openmp", "blis", "tbb", "lapack", "blas"}
    for item in active:
        row = _exact(item, ACTIVE_BACKEND_KEYS, "active backend")
        if row["role"] != "active_thread_pool" or row["backend"] not in backend_names:
            raise State3AuditVerificationError("invalid active-backend identity")
        _sha(row["binary_sha256"], "active backend binary")
        _sha(row["path_commitment"], "active backend path")
        if not isinstance(row["architecture"], str) or not row["architecture"]:
            raise State3AuditVerificationError("active backend architecture missing")
        threads = row["effective_thread_count"]
        if threads is not None and (
            isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0
        ):
            raise State3AuditVerificationError("active backend thread count invalid")
        symbols = _exact(row["inspection_symbols"], INSPECTION_SYMBOL_KEYS, "inspection symbols")
        if not all(item is None or isinstance(item, str) and item for item in symbols.values()):
            raise State3AuditVerificationError("runtime inspection symbol invalid")
        for key in ("runtime_configuration", "threading_layer", "version"):
            if row[key] is not None and (not isinstance(row[key], str) or not row[key]):
                raise State3AuditVerificationError("runtime inspection text invalid")
        complete = all(
            item is not None
            for item in (
                threads, symbols["effective_thread_count"], row["threading_layer"], row["version"]
            )
        )
        if row["inspection_complete"] is not complete:
            raise State3AuditVerificationError("runtime inspection-completeness mismatch")
    if active != sorted(
        active, key=lambda item: (item["backend"], item["binary_sha256"], item["path_commitment"])
    ):
        raise State3AuditVerificationError("active backend order mismatch")
    for item in wrappers:
        row = _exact(item, WRAPPER_KEYS, "numerical wrapper")
        if row["role"] != "numerical_interface_wrapper" or row["interface"] not in {"blas", "lapack"}:
            raise State3AuditVerificationError("invalid numerical wrapper")
        _sha(row["binary_sha256"], "wrapper binary")
        _sha(row["path_commitment"], "wrapper path")
        if not isinstance(row["architecture"], str) or not row["architecture"]:
            raise State3AuditVerificationError("wrapper architecture missing")
    if wrappers != sorted(
        wrappers, key=lambda item: (item["interface"], item["binary_sha256"], item["path_commitment"])
    ):
        raise State3AuditVerificationError("wrapper order mismatch")
    kinds = sorted({item["backend"] for item in active})
    layers = sorted(
        {item["threading_layer"] for item in active if isinstance(item["threading_layer"], str)}
    )
    applicable = _recognized_thread_variables(kinds, layers)
    if value["active_backend_kinds"] != kinds or value["applicable_additional_thread_variables"] != applicable:
        raise State3AuditVerificationError("runtime backend/policy summary mismatch")
    bound = value["bound_applicable_thread_policy"]
    if not isinstance(bound, Mapping) or any(
        name not in applicable or not isinstance(item, str) or not item
        for name, item in bound.items()
    ):
        raise State3AuditVerificationError("bound runtime policy invalid")
    missing = sorted(set(applicable) - set(bound))
    recognized = {name for names in ADDITIONAL_THREAD_VARIABLES.values() for name in names}
    unexpected = value["unexpected_additional_thread_policy_variables"]
    if (
        value["missing_applicable_thread_policy_variables"] != missing
        or not isinstance(unexpected, list) or unexpected != sorted(set(unexpected))
        or any(name in applicable or name not in recognized for name in unexpected)
    ):
        raise State3AuditVerificationError("runtime missing/unexpected policy mismatch")
    uninspectable = sorted(item["binary_sha256"] for item in active if not item["inspection_complete"])
    non_single = sorted(
        item["binary_sha256"] for item in active
        if item["effective_thread_count"] is not None and item["effective_thread_count"] != 1
    )
    expected_flags = {
        "all_active_backends_inspectable": bool(active) and not uninspectable,
        "all_effective_thread_counts_one": bool(active) and not non_single
        and all(item["effective_thread_count"] == 1 for item in active),
        "all_parent_thread_policy_bound": not missing and not unexpected,
    }
    expected_flags["all_runtime_requirements_satisfied"] = all(expected_flags.values())
    if (
        value["uninspectable_backend_binary_sha256"] != uninspectable
        or value["non_single_thread_backend_binary_sha256"] != non_single
        or any(value[name] is not expected for name, expected in expected_flags.items())
    ):
        raise State3AuditVerificationError("runtime aggregate mismatch")
    payload = {key: value[key] for key in RUNTIME_KEYS - {"runtime_record_sha256"}}
    expected_hash = hashlib.sha256(
        b"operatormark-numerical-runtime-record-v2\n" + _canonical(payload)
    ).hexdigest()
    if _sha(value["runtime_record_sha256"], "runtime record") != expected_hash:
        raise State3AuditVerificationError("runtime self-commitment mismatch")
    return value


def _validate_environment(record: Mapping[str, Any]) -> None:
    value = _exact(record, ENV_KEYS, "environment/runtime record")
    if (
        value["schema"] != MEMBER_SCHEMAS["environment_runtime.json"]
        or value["state"] != STATE or value["protocol_file_sha256"] != PROTOCOL_SHA256
        or (value["python_version"], value["numpy_version"], value["scipy_version"])
        != ("3.13.5", "2.5.2", "1.18.1")
        or value["requirements_lock_path"] != "requirements-lock-cpu.txt"
    ):
        raise State3AuditVerificationError("environment/runtime identity mismatch")
    for name in (
        "protocol_file_sha256", "state3_authorization_file_sha256", "launcher_file_sha256",
        "runtime_inspector_file_sha256", "requirements_lock_file_sha256",
        "base_import_closure_sha256", "numerical_runtime_sha256",
        "parent_environment_commitment", "record_self_sha256",
    ):
        _sha(value[name], name)
    for name in ("implementation_commit", "implementation_tree", "requirements_lock_git_blob"):
        _oid(value[name], name)
    for name in ("operating_system", "architecture", "locale"):
        if not isinstance(value[name], str) or not value[name]:
            raise State3AuditVerificationError("environment platform identity missing")
    environment = value["parent_environment"]
    if not isinstance(environment, Mapping) or not all(
        isinstance(name, str) and isinstance(item, str) and item
        for name, item in environment.items()
    ):
        raise State3AuditVerificationError("parent environment invalid")
    if any(environment.get(name) != item for name, item in MANDATORY_PARENT_ENVIRONMENT.items()):
        raise State3AuditVerificationError("mandatory parent environment mismatch")
    additional = value["applicable_additional_thread_variables"]
    if (
        not isinstance(additional, list) or additional != sorted(set(additional))
        or set(additional) != set(environment) - set(MANDATORY_PARENT_ENVIRONMENT)
    ):
        raise State3AuditVerificationError("additional parent policy mismatch")
    expected_environment_commitment = hashlib.sha256(
        b"operatormark-parent-environment-v1\n" + _canonical(dict(environment))
    ).hexdigest()
    if value["parent_environment_commitment"] != expected_environment_commitment:
        raise State3AuditVerificationError("parent environment commitment mismatch")
    closure = _validate_base_closure(value["base_import_closure"])
    runtime = _validate_runtime(value["numerical_runtime"])
    if (
        value["base_import_closure_sha256"] != closure["closure_sha256"]
        or value["numerical_runtime_sha256"] != runtime["runtime_record_sha256"]
        or runtime["applicable_additional_thread_variables"] != additional
        or runtime["bound_applicable_thread_policy"] != {
            name: environment[name] for name in additional
        }
        or value["all_environment_checks_passed"] is not True
        or value["all_runtime_checks_passed"] is not True
        or runtime["all_runtime_requirements_satisfied"] is not True
    ):
        raise State3AuditVerificationError("environment/runtime cross-binding mismatch")
    _validate_self_hash(value, "record_self_sha256")


def _validate_storage(record: Mapping[str, Any]) -> None:
    _validate_storage_for_protocol(record, expected_protocol_sha256=PROTOCOL_SHA256)


def _validate_storage_for_protocol(
    record: Mapping[str, Any], *, expected_protocol_sha256: str,
) -> None:
    value = _exact(record, STORAGE_KEYS, "storage preflight")
    if (
        value["schema"] != MEMBER_SCHEMAS["storage_preflight_before.json"]
        or value["protocol_file_sha256"] != expected_protocol_sha256
        or value["destination_relative_path"] != DESTINATION_RELATIVE
        or value["all_mandatory_checks_passed"] is not True
        or not isinstance(value["state"], str) or not value["state"]
    ):
        raise State3AuditVerificationError("storage-preflight identity mismatch")
    for name in (
        "protocol_file_sha256", "authorization_file_sha256", "repository_root_commitment",
        "destination_absolute_path_commitment", "record_self_sha256",
    ):
        _sha(value[name], name)
    for name in ("implementation_commit", "implementation_tree"):
        _oid(value[name], name)
    estimated = _nonnegative_integer(value["estimated_accepted_bytes"], "estimated bytes")
    projected = _nonnegative_integer(value["projected_audit_bytes"], "projected audit bytes")
    required = _nonnegative_integer(value["required_free_bytes"], "required free bytes")
    observed = _nonnegative_integer(value["observed_free_bytes"], "observed free bytes")
    if required != max(5 * 2**30, 4 * estimated + projected + 2**30) or observed < required:
        raise State3AuditVerificationError("storage free-space formula mismatch")
    method = value["canonical_resolution_method"]
    windows_method = (
        "GetFinalPathNameByHandleW(FILE_NAME_NORMALIZED) and GetFileInformationByHandle "
        "for every existing component; append only the validated fixed non-existing suffix"
    )
    posix_method = (
        "open(O_NOFOLLOW) plus fstat and realpath for every existing component; append only "
        "the validated fixed non-existing suffix"
    )
    if method not in {windows_method, posix_method}:
        raise State3AuditVerificationError("storage canonical-resolution method invalid")
    components = value["path_component_evidence"]
    expected_relatives = [".", "artifacts", "artifacts/darcy_branch_c_v1"]
    if not isinstance(components, list) or len(components) != len(expected_relatives):
        raise State3AuditVerificationError("storage component coverage incomplete")
    volume_commitment: str | None = None
    file_identities: set[str] = set()
    attributes: list[int] = []
    for component, expected_relative in zip(components, expected_relatives, strict=True):
        row = _exact(component, PATH_COMPONENT_KEYS, "storage path component")
        if row["relative_component"] != expected_relative or any(
            row[name] is not True
            for name in (
                "exists", "is_directory", "canonical_name_exact", "unicode_nfc_exact",
                "same_volume_as_repository_root", "stable_before_after",
            )
        ):
            raise State3AuditVerificationError("storage path-component check failed")
        if row["is_reparse_point"] is not False or row["reparse_tag_hex"] != "0x0":
            raise State3AuditVerificationError("storage path component is a reparse point")
        try:
            attribute = int(row["file_attributes_hex"], 16)
        except (TypeError, ValueError) as exc:
            raise State3AuditVerificationError("invalid storage file attributes") from exc
        attributes.append(attribute)
        if attribute & REPARSE_ATTRIBUTE:
            raise State3AuditVerificationError("storage attributes contain reparse flag")
        if method == windows_method and (
            not attribute & 0x10 or attribute & CLOUD_ATTRIBUTES
        ):
            raise State3AuditVerificationError("Windows storage component attributes unsafe")
        for name in (
            "canonical_path_commitment", "volume_identifier_commitment", "file_identity_commitment"
        ):
            _sha(row[name], name)
        if volume_commitment is None:
            volume_commitment = row["volume_identifier_commitment"]
        elif row["volume_identifier_commitment"] != volume_commitment:
            raise State3AuditVerificationError("storage component volume changed")
        if row["file_identity_commitment"] in file_identities:
            raise State3AuditVerificationError("storage component file identity duplicated")
        file_identities.add(row["file_identity_commitment"])
    volume = _exact(value["volume_evidence"], VOLUME_KEYS, "storage volume")
    for name in ("allocation_unit_bytes", "capacity_bytes", "free_bytes", "maximum_component_length"):
        if _nonnegative_integer(volume[name], name) == 0:
            raise State3AuditVerificationError("storage volume integer is zero")
    if (
        volume["is_fixed_local"] is not True or volume["free_bytes"] != observed
        or volume["capacity_bytes"] < observed
    ):
        raise State3AuditVerificationError("storage volume locality/capacity mismatch")
    if method == windows_method:
        if volume["drive_type"] != 3 or volume["path_length_capability"] != (
            "not_tested_no_extended_length_path_claim"
        ):
            raise State3AuditVerificationError("Windows storage volume claim invalid")
        try:
            filesystem_flags = int(volume["filesystem_flags_hex"], 16)
        except (TypeError, ValueError) as exc:
            raise State3AuditVerificationError("Windows filesystem flags invalid") from exc
        if not filesystem_flags & 0x2 or volume["case_sensitivity_behavior"] not in {
            "enabled", "disabled", "query_unsupported_exact_directory_entry_scan_enforced",
        }:
            raise State3AuditVerificationError("Windows filesystem capabilities invalid")
    else:
        filesystem_flags = 0
    for name in ("volume_identifier_commitment", "volume_root_commitment"):
        _sha(volume[name], name)
    if volume["volume_identifier_commitment"] != volume_commitment:
        raise State3AuditVerificationError("storage component/volume commitment mismatch")
    sync = _exact(value["sync_detection_evidence"], SYNC_KEYS, "storage sync evidence")
    if (
        sync["inspection_complete"] is not True
        or sync["destination_in_detected_sync_root"] is not False
        or sync["cloud_attribute_components"] != []
        or sync["environment_names_checked"] != SYNC_ENVIRONMENT_NAMES
    ):
        raise State3AuditVerificationError("storage synchronization exclusion failed")
    registry_sources = sync["registry_sources_checked"]
    if (
        not isinstance(registry_sources, list) or registry_sources != sorted(set(registry_sources))
        or not {"HKCU:OneDrive.Accounts", "HKCU:SyncRootManager", "HKCU:UserShellFolders"}
        .issubset(registry_sources)
    ):
        raise State3AuditVerificationError("storage sync registry coverage incomplete")
    roots = sync["configured_sync_roots"]
    clients = sync["active_sync_clients"]
    if not isinstance(roots, list) or not isinstance(clients, list):
        raise State3AuditVerificationError("storage sync roots/clients invalid")
    for root in roots:
        row = _exact(root, SYNC_ROOT_KEYS, "sync root")
        if (
            not isinstance(row["provider"], str) or not row["provider"]
            or row["source"] not in {
                "environment", "registry", "sync_root_manager",
                "sync_root_manager_user_roots", "user_shell_folders",
            }
            or not isinstance(row["exists"], bool)
            or row["handle_canonicalized"] is not row["exists"]
        ):
            raise State3AuditVerificationError("sync-root evidence invalid")
        _sha(row["lexical_root_commitment"], "sync root lexical commitment")
        if row["exists"]:
            _sha(row["canonical_root_commitment"], "sync root canonical commitment")
        elif row["canonical_root_commitment"] is not None:
            raise State3AuditVerificationError("nonexistent sync root has canonical commitment")
    if roots != sorted(
        roots, key=lambda item: (item["provider"], item["source"], item["lexical_root_commitment"])
    ):
        raise State3AuditVerificationError("sync-root order invalid")
    for client in clients:
        row = _exact(client, SYNC_CLIENT_KEYS, "sync client")
        _sha(row["process_name_commitment"], "sync process commitment")
        count = _nonnegative_integer(row["configured_root_count"], "sync root count")
        matching = sum(
            1 for root in roots
            if root["provider"] == row["provider"] and root["exists"] is True
            and root["handle_canonicalized"] is True
        )
        if count == 0 or count != matching:
            raise State3AuditVerificationError("active sync client lacks bound roots")
    if clients != sorted(clients, key=lambda item: (item["provider"], item["process_name_commitment"])):
        raise State3AuditVerificationError("sync-client order invalid")
    ignore = _exact(value["git_ignore_evidence"], GIT_IGNORE_KEYS, "storage Git ignore")
    count = _nonnegative_integer(ignore["candidate_count"], "Git candidate count")
    commitments = ignore["candidate_path_commitments"]
    if (
        ignore["matched"] is not True
        or ignore["all_candidates_matched_intended_rule"] is not True
        or ignore["intended_rule"] != ".gitignore:artifacts/" or count < 1
        or not isinstance(commitments, list) or len(commitments) != count
    ):
        raise State3AuditVerificationError("storage Git-ignore proof failed")
    for commitment in commitments:
        _sha(commitment, "Git candidate commitment")
    if len(set(commitments)) != count or commitments[0] != _commit(
        "operatormark-storage-git-candidate-path-v1",
        value["authorization_file_sha256"], DESTINATION_RELATIVE,
    ):
        raise State3AuditVerificationError("storage Git candidate binding failed")
    _sha(ignore["check_ignore_output_sha256"], "check-ignore output")
    tracking = _exact(value["git_tracking_evidence"], GIT_TRACKING_KEYS, "storage Git tracking")
    for name in (
        "tracked_path_count", "submodule_path_count", "nested_worktree_count",
        "nested_repository_marker_count", "candidate_collision_count",
    ):
        if _nonnegative_integer(tracking[name], name) != 0:
            raise State3AuditVerificationError("storage Git collision/tracking count nonzero")
    _sha(tracking["worktree_evidence_sha256"], "worktree evidence")
    _oid(tracking["head_commit"], "storage head")
    _oid(tracking["head_tree"], "storage tree")
    _oid(tracking["parent_commit"], "storage parent")
    if (
        tracking["branch"] != "main"
        or tracking["head_commit"] != value["implementation_commit"]
        or tracking["head_tree"] != value["implementation_tree"]
        or tracking["tracked_status_clean"] is not True
        or tracking["staged_status_clean"] is not True
        or tracking["untracked_inclusive_status_clean"] is not True
        or tracking["status_porcelain_sha256"] != EMPTY_SHA256
    ):
        raise State3AuditVerificationError("storage Git binding is not clean/exact")
    probe = _exact(value["probe_evidence"], PROBE_KEYS, "storage probe")
    if (
        probe["fixed_literal_sha256"] != PROBE_LITERAL_SHA256
        or probe["operations"] != PROBE_OPERATIONS or probe["passed"] is not True
        or probe["probe_file_preserved"] is not False
        or probe["probe_directory_preserved"] is not False
        or probe["same_volume_verified"] is not True
        or probe["atomic_directory_rename_verified"] is not True
    ):
        raise State3AuditVerificationError("storage operation probe failed")
    for name in ("source_volume_identifier_commitment", "renamed_volume_identifier_commitment"):
        _sha(probe[name], name)
    if (
        probe["source_volume_identifier_commitment"]
        != probe["renamed_volume_identifier_commitment"]
        or probe["source_volume_identifier_commitment"] != volume_commitment
    ):
        raise State3AuditVerificationError("storage probe volume binding mismatch")
    permission = _exact(value["permission_evidence"], PERMISSION_KEYS, "storage permission")
    rights = _exact(permission["current_process_effective_rights"], RIGHTS_KEYS, "storage rights")
    blocking_count = _nonnegative_integer(
        permission["blocking_broad_access_count"], "blocking broad-access count"
    )
    residual_count = _nonnegative_integer(
        permission["broader_local_user_residual_count"], "permission residual count"
    )
    if (
        permission["platform"] != "nt"
        or permission["inspection_method"]
        != "GetNamedSecurityInfoW plus GetEffectiveRightsFromAclW plus operation probe"
        or permission["dacl_present"] is not True or blocking_count != 0
        or not isinstance(permission["inheritance_enabled"], bool)
        or any(item is not True for item in rights.values())
    ):
        raise State3AuditVerificationError("storage effective-permission proof failed")
    _sha(permission["security_descriptor_commitment"], "security descriptor")
    _sha(permission["owning_principal_commitment"], "owning principal")
    broad = permission["broad_principal_effective_access"]
    if not isinstance(broad, list) or len(broad) != len(BROAD_PRINCIPALS):
        raise State3AuditVerificationError("broad-principal coverage incomplete")
    calculated_blocking = 0
    calculated_residual = 0
    for item, (expected_class, sid, blocks) in zip(broad, BROAD_PRINCIPALS, strict=True):
        row = _exact(item, PRINCIPAL_KEYS, "broad principal")
        if row["principal_class"] != expected_class or row["principal_commitment"] != _commit(
            "operatormark-storage-principal-v1", value["authorization_file_sha256"], sid
        ):
            raise State3AuditVerificationError("fixed broad-principal binding mismatch")
        try:
            mask = int(row["effective_mask_hex"], 16)
        except (TypeError, ValueError) as exc:
            raise State3AuditVerificationError("broad-principal mask invalid") from exc
        read = bool(mask & ACL_READ_MASK)
        write = bool(mask & ACL_WRITE_MASK)
        blocking = blocks and (read or write)
        residual = not blocks and (read or write)
        if (
            row["effective_read"] is not read or row["effective_write"] is not write
            or row["blocking"] is not blocking or row["residual_risk"] is not residual
        ):
            raise State3AuditVerificationError("broad-principal interpretation mismatch")
        calculated_blocking += int(blocking)
        calculated_residual += int(residual)
    additional = permission["additional_allow_principal_effective_access"]
    additional_count = _nonnegative_integer(
        permission["additional_allow_principal_count"], "additional principal count"
    )
    if (
        not isinstance(additional, list) or len(additional) != additional_count
        or not all(isinstance(item, Mapping) for item in additional)
        or additional != sorted(additional, key=lambda item: item["principal_commitment"])
    ):
        raise State3AuditVerificationError("additional allow-principal coverage invalid")
    seen_additional: set[str] = set()
    for item in additional:
        row = _exact(item, PRINCIPAL_KEYS, "additional principal")
        commitment = _sha(row["principal_commitment"], "additional principal")
        if commitment in seen_additional or row["principal_class"] != "additional_allow_ace":
            raise State3AuditVerificationError("additional principal binding invalid")
        seen_additional.add(commitment)
        try:
            mask = int(row["effective_mask_hex"], 16)
        except (TypeError, ValueError) as exc:
            raise State3AuditVerificationError("additional-principal mask invalid") from exc
        read = bool(mask & ACL_READ_MASK)
        write = bool(mask & ACL_WRITE_MASK)
        residual = read or write
        if (
            row["effective_read"] is not read or row["effective_write"] is not write
            or row["blocking"] is not False or row["residual_risk"] is not residual
        ):
            raise State3AuditVerificationError("additional-principal interpretation mismatch")
        calculated_residual += int(residual)
    if calculated_blocking != blocking_count or calculated_residual != residual_count:
        raise State3AuditVerificationError("permission aggregate count mismatch")
    encryption = _exact(value["encryption_evidence"], ENCRYPTION_KEYS, "storage encryption")
    if (
        not isinstance(encryption["filesystem_supports_encryption"], bool)
        or encryption["encryption_presence_establishes_legal_or_cryptographic_protection"] is not False
        or encryption["status"] != "evidence_only_with_key_management_and_access_residuals"
    ):
        raise State3AuditVerificationError("storage encryption claim overstated")
    encrypted_count = _nonnegative_integer(
        encryption["encrypted_ancestor_count"], "encrypted ancestor count"
    )
    if encrypted_count != sum(bool(item & ENCRYPTED_ATTRIBUTE) for item in attributes):
        raise State3AuditVerificationError("encrypted-ancestor count mismatch")
    if method == windows_method and encryption["filesystem_supports_encryption"] is not bool(
        filesystem_flags & 0x00020000
    ):
        raise State3AuditVerificationError("filesystem encryption flag mismatch")
    outcomes = {
        "observed": {"reported_protected_not_independently_proven", "reported_not_protected", "observed_but_unclassified"},
        "query_failed": {"unknown_query_failed"},
        "unavailable": {"unknown_query_unavailable"},
    }
    query = encryption["device_encryption_query_outcome"]
    if query not in outcomes or encryption["device_encryption_interpreted_status"] not in outcomes[query]:
        raise State3AuditVerificationError("device-encryption interpretation invalid")
    if query == "unavailable":
        if encryption["device_encryption_evidence_commitment"] is not None:
            raise State3AuditVerificationError("unavailable encryption query has evidence")
    else:
        _sha(encryption["device_encryption_evidence_commitment"], "encryption evidence")
    backup = _exact(value["backup_and_indexing_evidence"], BACKUP_KEYS, "storage exposure")
    if (
        backup["administrator_backup_malware_physical_access_and_remanence_eliminated"] is not False
        or backup["status"] != (
            "administrator_backup_indexing_antivirus_crash_dump_paging_malware_"
            "physical_access_key_management_and_deletion_remanence_risks_remain"
        )
    ):
        raise State3AuditVerificationError("storage exposure claim overstated")
    exposure_names = (
        "administrator_access_may_expose_copies", "backup_software_may_expose_copies",
        "indexing_may_expose_copies", "antivirus_may_expose_copies",
        "crash_dumps_may_expose_copies", "paging_may_expose_copies",
        "local_sync_clients_may_expose_copies",
    )
    if any(not isinstance(backup[name], bool) for name in exposure_names) or any(
        backup[name] is not True
        for name in (
            "administrator_access_may_expose_copies", "backup_software_may_expose_copies",
            "antivirus_may_expose_copies", "crash_dumps_may_expose_copies",
            "paging_may_expose_copies",
        )
    ):
        raise State3AuditVerificationError("storage exposure booleans invalid")
    expected_indexing = any(not item & NOT_CONTENT_INDEXED_ATTRIBUTE for item in attributes)
    if (
        backup["indexing_may_expose_copies"] is not expected_indexing
        or backup["local_sync_clients_may_expose_copies"] is not bool(clients)
    ):
        raise State3AuditVerificationError("storage exposure interpretation mismatch")
    service_commitments = backup["service_query_evidence_commitments"]
    service_outcomes = backup["service_query_outcomes"]
    services = {"WSearch", "wbengine", "VSS"}
    if (
        not isinstance(service_commitments, Mapping) or set(service_commitments) != services
        or not isinstance(service_outcomes, Mapping) or set(service_outcomes) != services
    ):
        raise State3AuditVerificationError("storage service evidence invalid")
    for service in services:
        outcome = service_outcomes[service]
        if outcome not in {"observed", "query_failed", "unavailable"}:
            raise State3AuditVerificationError("storage service outcome invalid")
        if outcome == "unavailable":
            if service_commitments[service] is not None:
                raise State3AuditVerificationError("unavailable service has evidence")
        else:
            _sha(service_commitments[service], "storage service evidence")
    if backup["registry_query_outcome"] not in {"observed", "partially_observed", "unavailable"}:
        raise State3AuditVerificationError("storage registry evidence outcome invalid")
    _sha(backup["registry_query_evidence_commitment"], "storage registry evidence")
    expected_risks = [
        "administrator backup malware physical-access key-management and deletion-remanence risks remain"
    ]
    if residual_count:
        expected_risks.append(
            "broader local-user effective access remains and requires explicit State 4 acceptance"
        )
    if value["residual_risks"] != expected_risks:
        raise State3AuditVerificationError("storage residual-risk list mismatch")
    _validate_self_hash(value, "record_self_sha256")


def _float_hex(value: object, *, nonnegative: bool = True) -> float:
    if not isinstance(value, str):
        raise State3AuditVerificationError("scientific scalar is not float.hex text")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise State3AuditVerificationError("invalid float.hex text") from exc
    if not math.isfinite(parsed) or parsed.hex().lower() != value or (nonnegative and parsed < 0):
        raise State3AuditVerificationError("noncanonical/non-finite scientific scalar")
    return parsed


def _validate_raw_measurement(record: object) -> None:
    value = _exact(record, RAW_KEYS, "raw measurement")
    if value["fixture"] not in FIXTURES or value["repeat_index"] not in range(5):
        raise State3AuditVerificationError("raw fixture/repeat invalid")
    if isinstance(value["repeat_index"], bool) or value["child_exit_code"] != 0:
        raise State3AuditVerificationError("raw child status invalid")
    for name in SUMMARY_FIELDS:
        _nonnegative_integer(value[name], name)
    if value["paired_solver_ns"] != sum(value[name] for name in TIMING_FIELDS[:7]):
        raise State3AuditVerificationError("paired timing formula mismatch")
    if value["child_wall_ns"] < value["paired_solver_ns"]:
        raise State3AuditVerificationError("child wall shorter than measured regions")
    if value["incremental_peak_rss_bytes"] != max(
        0, value["peak_rss_bytes"] - value["baseline_rss_bytes"]
    ):
        raise State3AuditVerificationError("incremental peak RSS mismatch")
    if not isinstance(value["rss_counter_name"], str) or not value["rss_counter_name"]:
        raise State3AuditVerificationError("missing RSS counter identity")
    commitments = _exact(value["rng_state_commitments"], RNG_COMMITMENT_KEYS, "RNG commitments")
    equalities = _exact(value["rng_state_equalities"], RNG_EQUALITY_KEYS, "RNG equalities")
    for name, commitment in commitments.items():
        _sha(commitment, name)
    if any(item is not True for item in equalities.values()):
        raise State3AuditVerificationError("RNG state equality failed")
    if not (
        commitments["numpy_legacy_after_base_import_sha256"]
        == commitments["numpy_legacy_after_guard_install_sha256"]
        == commitments["numpy_legacy_after_project_import_sha256"]
        == commitments["numpy_legacy_after_execution_sha256"]
        and commitments["python_random_after_base_import_sha256"]
        == commitments["python_random_after_guard_install_sha256"]
        == commitments["python_random_after_project_import_sha256"]
        == commitments["python_random_after_execution_sha256"]
    ):
        raise State3AuditVerificationError("RNG commitments differ across checkpoints")
    if value["forbidden_rng_call_count"] != 0 or value["forbidden_corpus_seed_access_count"] != 0:
        raise State3AuditVerificationError("forbidden worker count nonzero")
    tpfa = _exact(value["tpfa_diagnostics"], TPFA_KEYS, "TPFA diagnostics")
    p1 = _exact(value["p1_diagnostics"], P1_KEYS, "P1 diagnostics")
    if tpfa["passes"] is not True or p1["passes"] is not True:
        raise State3AuditVerificationError("solver diagnostics did not pass")
    for name in TPFA_KEYS - {"passes"}:
        _float_hex(tpfa[name])
    for name in P1_KEYS - {"passes", "iterations", "outward_boundary_flux", "integrated_source"}:
        _float_hex(p1[name])
    for name in ("outward_boundary_flux", "integrated_source"):
        _float_hex(p1[name], nonnegative=False)
    if isinstance(p1["iterations"], bool) or not isinstance(p1["iterations"], int) or p1["iterations"] <= 0:
        raise State3AuditVerificationError("P1 iteration count invalid")
    cross = _exact(value["cross_solver_metrics"], {"scales", "compliances", "delta_L2", "delta_J"},
                   "cross-solver metrics")
    scales = _exact(cross["scales"], {"S_L2", "S_J"}, "cross-solver scales")
    compliances = _exact(cross["compliances"], {"T64", "bar_P256"}, "compliances")
    for item in scales.values():
        _float_hex(item)
    for item in compliances.values():
        _float_hex(item, nonnegative=False)
    decoded_tpfa = {name: _float_hex(tpfa[name]) for name in TPFA_KEYS - {"passes"}}
    decoded_p1 = {
        name: _float_hex(p1[name], nonnegative=name not in {"outward_boundary_flux", "integrated_source"})
        for name in P1_KEYS - {"passes", "iterations"}
    }
    if any(
        decoded_tpfa[name] > limit
        for name, limit in {
            "algebraic_backward_error": 1.0e-10,
            "boundary_transmissibility_error": 100.0 * sys.float_info.epsilon,
            "maximum_local_conservation_ratio": 1.0e-10,
            "maximum_interior_face_antisymmetry": 1.0e-10,
            "global_balance_ratio": 1.0e-10,
        }.items()
    ) or any(
        decoded_p1[name] > limit
        for name, limit in {
            "algebraic_backward_error": 1.0e-10,
            "boundary_reconstruction_error": 100.0 * sys.float_info.epsilon,
            "reaction_balance_ratio": 1.0e-10,
        }.items()
    ):
        raise State3AuditVerificationError("solver diagnostic threshold failed")
    decoded_scales = {name: _float_hex(item) for name, item in scales.items()}
    decoded_compliances = {
        name: _float_hex(item, nonnegative=False) for name, item in compliances.items()
    }
    if decoded_scales["S_J"] != abs(decoded_compliances["bar_P256"]):
        raise State3AuditVerificationError("cross-solver S_J formula mismatch")
    denominator = decoded_scales["S_J"]
    numerator = abs(decoded_compliances["T64"] - decoded_compliances["bar_P256"])
    expected_delta_j = numerator / denominator if denominator > 0 else 0.0 if numerator == 0 else math.inf
    delta_j = _float_hex(cross["delta_J"])
    if not math.isfinite(expected_delta_j) or delta_j != expected_delta_j:
        raise State3AuditVerificationError("cross-solver delta_J formula mismatch")
    if _float_hex(cross["delta_L2"]) > 0.02543117962436324:
        raise State3AuditVerificationError("delta_L2 gate failed")
    if delta_j > 0.022582030716092032:
        raise State3AuditVerificationError("delta_J gate failed")
    if value["gates_passed"] is not True:
        raise State3AuditVerificationError("raw aggregate gate failed")


def _summary(values: Sequence[int]) -> dict[str, int | float]:
    return {"maximum": max(values), "median": statistics.median(values)}


def _expected_summary(raw: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def group(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {field: _summary([item[field] for item in items]) for field in SUMMARY_FIELDS}
    return {
        "fields": list(SUMMARY_FIELDS), "global": group(raw),
        "per_fixture": {
            fixture: group([item for item in raw if item["fixture"] == fixture])
            for fixture in FIXTURES
        },
    }


def _expected_extrapolation(raw: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pair = [item["paired_solver_ns"] for item in raw]
    peaks = [item["peak_rss_bytes"] for item in raw]
    median_pair_seconds = float(statistics.median(pair)) / 1e9
    max_pair_seconds = float(max(pair)) / 1e9
    median_seconds = 4864 * median_pair_seconds
    max_seconds = 4864 * max_pair_seconds
    budget = 2.0 * max_seconds
    return {
        "median_pair_seconds": median_pair_seconds, "max_pair_seconds": max_pair_seconds,
        "median_extrapolated_seconds": median_seconds, "max_extrapolated_seconds": max_seconds,
        "frozen_safety_factor": 2.0, "safety_factored_compute_budget_seconds": budget,
        "safety_factored_peak_memory_budget_bytes": 2 * max(peaks),
        "median_extrapolated_hours": median_seconds / 3600.0,
        "max_extrapolated_hours": max_seconds / 3600.0,
        "safety_factored_compute_budget_hours": budget / 3600.0,
    }


def _validate_disk(record: object) -> Mapping[str, Any]:
    value = _exact(record, DISK_KEYS, "disk estimate")
    for name, item in value.items():
        _nonnegative_integer(item, name)
    if value["raw_seven_tensor_payload_bytes"] != 420_872_192:
        raise State3AuditVerificationError("raw tensor payload estimate changed")
    accepted = (
        value["raw_seven_tensor_payload_bytes"] + value["projected_npy_header_bytes"]
        + value["projected_jsonl_bytes"] + value["projected_manifest_and_audit_bytes"]
    )
    formula = 4 * accepted + value["projected_audit_bytes"] + 2**30
    if (
        value["estimated_accepted_bytes"] != accepted or value["controls_only_bytes"] > accepted
        or value["active_staging_bytes"] != accepted
        or value["readback_atomic_publication_bytes"] != accepted
        or value["preserved_failed_attempt_allowance_bytes"] != accepted
        or value["fixed_headroom_bytes"] != 2**30 or value["minimum_floor_bytes"] != 5 * 2**30
        or value["four_copy_formula_bytes"] != formula
        or value["total_required_free_space_bytes"] != max(5 * 2**30, formula)
    ):
        raise State3AuditVerificationError("writer-derived disk formula mismatch")
    return value


def _validate_report(record: Mapping[str, Any]) -> None:
    value = _exact(record, REPORT_KEYS, "compute-feasibility report")
    if (
        value["schema"] != MEMBER_SCHEMAS["compute_feasibility_report.json"]
        or value["state"] != "state3_feasibility_only_no_corpus_controls"
        or value["protocol_file_sha256"] != PROTOCOL_SHA256
        or value["solver_spec_baseline_commit"] != BASELINE_COMMIT
        or value["solver_spec_baseline_tree"] != BASELINE_TREE
        or value["warmup_count_per_fixture"] != 0 or value["measured_count_per_fixture"] != 5
        or value["measurement_order"] != list(FIXTURES) or value["corpus_sample_count"] != 4864
        or value["frozen_safety_factor"] != 2.0
    ):
        raise State3AuditVerificationError("compute report frozen identity/count mismatch")
    for name in (
        "protocol_file_sha256", "state3_authorization_file_sha256", "launcher_file_sha256",
        "runtime_inspector_file_sha256", "environment_record_sha256",
        "storage_preflight_before_file_sha256", "storage_preflight_after_file_sha256",
        "report_self_sha256",
    ):
        _sha(value[name], name)
    for name in ("solver_spec_baseline_commit", "solver_spec_baseline_tree",
                 "implementation_commit", "implementation_tree"):
        _oid(value[name], name)
    expected_definitions = {
        "constant_half": {"construction": "deterministic_indexed_assignment_no_rng", "dtype": "<f8",
                          "formula": "rho[j,i]=0.5", "shape": [8, 8]},
        "vertical_jump": {"construction": "deterministic_indexed_assignment_no_rng", "dtype": "<f8",
                          "formula": "rho[j,i]=0.0 if i<4 else 1.0", "shape": [8, 8]},
        "checkerboard": {"construction": "deterministic_indexed_assignment_no_rng", "dtype": "<f8",
                         "formula": "rho[j,i]=float((i+j) mod 2)", "shape": [8, 8]},
    }
    if value["fixture_definitions"] != expected_definitions:
        raise State3AuditVerificationError("fixture definitions changed")
    expected_rng_spec = {
        "numpy_legacy_domain_separator": "operatormark-numpy-legacy-rng-state-v1\n",
        "numpy_legacy_canonical_fields": ["bit_generator", "key_dtype", "key_shape",
            "key_c_order_hex", "position", "has_gauss", "cached_gaussian_hex"],
        "python_random_domain_separator": "operatormark-python-random-state-v1\n",
        "python_random_canonical_fields": ["version", "internal_state", "gauss_next"],
        "canonical_json_encoding": "utf8_compact_sorted_keys_no_trailing_newline",
        "ordered_checkpoints": ["after_base_import", "after_guard_install",
                                "after_project_import", "after_execution"],
        "passive_module_import_increments_forbidden_call_count": False,
    }
    if value["rng_state_commitment_specification"] != expected_rng_spec:
        raise State3AuditVerificationError("RNG commitment specification changed")
    if value["timing_clock"] != {
        "name": "time.perf_counter_ns", "unit": "nanoseconds", "monotonic": True,
        "interpretation": "same_platform_feasibility_estimates_not_performance_guarantees",
    }:
        raise State3AuditVerificationError("timing clock changed")
    memory = _exact(value["memory_counter"],
                    {"name", "unit", "definition", "cross_platform_equivalence_claimed"},
                    "memory counter")
    expected_memory_definition = (
        "absolute_peak_working_set"
        if memory["name"] == "windows_process_peak_working_set_bytes"
        else "platform_ru_maxrss_normalized_to_bytes"
    )
    if (
        not isinstance(memory["name"], str) or not memory["name"]
        or memory["unit"] != "bytes" or memory["definition"] != expected_memory_definition
        or memory["cross_platform_equivalence_claimed"] is not False
    ):
        raise State3AuditVerificationError("memory counter invalid")
    raw = value["raw_measurements"]
    if not isinstance(raw, list) or len(raw) != 15:
        raise State3AuditVerificationError("report must contain 15 raw measurements")
    expected_coverage = [(fixture, repeat) for fixture in FIXTURES for repeat in range(5)]
    if [(item.get("fixture"), item.get("repeat_index")) for item in raw] != expected_coverage:
        raise State3AuditVerificationError("report measurement coverage/order mismatch")
    for item in raw:
        _validate_raw_measurement(item)
        if item["rss_counter_name"] != memory["name"]:
            raise State3AuditVerificationError("worker/report memory counter mismatch")
    if value["summary"] != _expected_summary(raw):
        raise State3AuditVerificationError("summary formula mismatch")
    if value["extrapolation"] != _expected_extrapolation(raw):
        raise State3AuditVerificationError("extrapolation formula mismatch")
    _validate_disk(value["disk_estimate"])
    if (
        value["all_fixture_gates_passed"] is not True
        or value["all_rng_state_equalities_passed"] is not True
        or value["forbidden_rng_call_count"] != 0
        or value["forbidden_corpus_seed_access_count"] != 0
    ):
        raise State3AuditVerificationError("report aggregate gate/count mismatch")
    before = _validate_git(value["git_before"], value["implementation_commit"], value["implementation_tree"])
    after = _validate_git(value["git_after"], value["implementation_commit"], value["implementation_tree"])
    if _canonical(before) != _canonical(after):
        raise State3AuditVerificationError("Git state changed across report")
    _validate_self_hash(value, "report_self_sha256")


def _validate_source(record: Mapping[str, Any]) -> None:
    value = _exact(record, SOURCE_KEYS, "source/blob verification")
    if (
        value["schema"] != MEMBER_SCHEMAS["source_blob_verification.json"]
        or value["state"] != STATE or value["protocol_file_sha256"] != PROTOCOL_SHA256
        or value["solver_spec_baseline_commit"] != BASELINE_COMMIT
        or value["solver_spec_baseline_tree"] != BASELINE_TREE
        or value["requirements_lock_path"] != "requirements-lock-cpu.txt"
    ):
        raise State3AuditVerificationError("source/blob identity mismatch")
    for name in (
        "protocol_file_sha256", "state3_authorization_file_sha256",
        "requirements_lock_file_sha256", "record_self_sha256",
    ):
        _sha(value[name], name)
    for name in (
        "solver_spec_baseline_commit", "solver_spec_baseline_tree", "implementation_commit",
        "implementation_tree", "implementation_parent_commit", "requirements_lock_git_blob",
    ):
        _oid(value[name], name)
    baseline = value["frozen_baseline_blobs"]
    if not isinstance(baseline, list):
        raise State3AuditVerificationError("baseline blob inventory invalid")
    baseline_paths: list[str] = []
    baseline_keys = {
        "path", "baseline_git_blob", "implementation_git_blob", "baseline_file_sha256",
        "implementation_file_sha256", "byte_identical",
    }
    for item in baseline:
        row = _exact(item, baseline_keys, "baseline blob row")
        path = _relative_path(row["path"], "baseline blob path")
        baseline_paths.append(path)
        for name in ("baseline_git_blob", "implementation_git_blob"):
            _oid(row[name], name)
        for name in ("baseline_file_sha256", "implementation_file_sha256"):
            _sha(row[name], name)
        if (
            row["byte_identical"] is not True
            or row["baseline_git_blob"] != row["implementation_git_blob"]
            or row["baseline_file_sha256"] != row["implementation_file_sha256"]
            or (
                path in FROZEN_REVIEW_HASHES
                and row["implementation_file_sha256"] != FROZEN_REVIEW_HASHES[path]
            )
        ):
            raise State3AuditVerificationError("frozen baseline blob changed")
    if baseline_paths != sorted(set(baseline_paths)) or not FROZEN_BASELINE_REQUIRED_PATHS.issubset(
        baseline_paths
    ):
        raise State3AuditVerificationError("baseline blob coverage/order mismatch")
    implementation = value["implementation_files"]
    if not isinstance(implementation, list) or not implementation:
        raise State3AuditVerificationError("implementation file inventory invalid")
    implementation_paths: list[str] = []
    for item in implementation:
        row = _exact(item, {"path", "git_blob", "file_sha256", "matches_git_blob"},
                     "implementation file row")
        implementation_paths.append(_relative_path(row["path"], "implementation file path"))
        _oid(row["git_blob"], "implementation Git blob")
        _sha(row["file_sha256"], "implementation file hash")
        if row["matches_git_blob"] is not True:
            raise State3AuditVerificationError("implementation file does not match Git")
    if implementation_paths != sorted(set(implementation_paths)):
        raise State3AuditVerificationError("implementation inventory order/uniqueness mismatch")
    if (
        value["all_frozen_baseline_blobs_byte_identical"] is not True
        or value["all_implementation_files_match_git"] is not True
    ):
        raise State3AuditVerificationError("source/blob aggregate failed")
    before = _validate_git(value["git_before"], value["implementation_commit"], value["implementation_tree"])
    after = _validate_git(value["git_after"], value["implementation_commit"], value["implementation_tree"])
    if before["parents"] != [value["implementation_parent_commit"]] or _canonical(before) != _canonical(after):
        raise State3AuditVerificationError("source/blob Git parent or equality mismatch")
    _validate_self_hash(value, "record_self_sha256")


def _expected_original_failure_rows() -> list[dict[str, Any]]:
    return [
        {
            "filename": filename,
            "original_relative_path": f"{FAILURE_ORIGINAL_PREFIX}/{filename}",
            "relocated_relative_path": f"{FAILURE_DESTINATION_PREFIX}/{filename}",
            "size": size,
            "sha256": digest,
            "byte_identical": True,
            "preserved": True,
        }
        for filename, size, digest in FAILURE_SPECS
    ]


def _expected_failure_rows() -> list[dict[str, Any]]:
    rows = _expected_original_failure_rows()
    rows.extend(
        {
            "filename": PurePosixPath(path).name,
            "original_relative_path": path,
            "relocated_relative_path": path,
            "size": size,
            "sha256": digest,
            "byte_identical": True,
            "preserved": True,
        }
        for path, size, digest in IN_PLACE_FAILURE_SPECS
    )
    return rows


def _validate_failure_inventory(record: Mapping[str, Any]) -> None:
    keys = {
        "schema", "state", "expected_record_count", "records", "relocation_method",
        "original_container_removed_only_after_empty", "all_records_preserved_byte_identical",
        "original_relocated_record_count", "in_place_preserved_record_count",
        "failure_record_contents_embedded",
        "historical_overbroad_pytest_prohibited_action_accounting",
        "review_only_driver", "record_self_sha256",
    }
    value = _exact(record, keys, "failure inventory")
    if (
        value["schema"] != MEMBER_SCHEMAS["failure_inventory.json"]
        or value["state"] != STATE or value["expected_record_count"] != 17
        or value["original_relocated_record_count"] != 12
        or value["in_place_preserved_record_count"] != 5
        or value["records"] != _expected_failure_rows()
        or value["relocation_method"]
        != "original_twelve_same_volume_atomic_directory_rename_plus_five_preserved_in_place"
        or value["original_container_removed_only_after_empty"] is not True
        or value["all_records_preserved_byte_identical"] is not True
        or value["failure_record_contents_embedded"] is not False
        or value["historical_overbroad_pytest_prohibited_action_accounting"]
        != "not_provably_zero"
    ):
        raise State3AuditVerificationError("failure inventory changed")
    row_keys = {
        "filename", "original_relative_path", "relocated_relative_path", "size", "sha256",
        "byte_identical", "preserved",
    }
    filenames: list[str] = []
    current_paths: list[str] = []
    for item in value["records"]:
        row = _exact(item, row_keys, "failure row")
        if not isinstance(row["filename"], str) or not row["filename"]:
            raise State3AuditVerificationError("failure filename is invalid")
        filenames.append(row["filename"])
        current_paths.append(_relative_path(row["relocated_relative_path"], "failure path"))
        _relative_path(row["original_relative_path"], "failure original path")
        _nonnegative_integer(row["size"], "failure size")
        _sha(row["sha256"], "failure hash")
        if row["byte_identical"] is not True or row["preserved"] is not True:
            raise State3AuditVerificationError("failure row is not preserved")
    if len(set(filenames)) != 17 or len(set(current_paths)) != 17:
        raise State3AuditVerificationError("failure paths or filenames are not unique")
    driver = _exact(
        value["review_only_driver"], {"path", "sha256", "executed", "byte_identical"},
        "review-only driver",
    )
    if (
        driver["path"] != DRIVER_PATH or driver["sha256"] != DRIVER_SHA256
        or driver["executed"] is not False or driver["byte_identical"] is not True
    ):
        raise State3AuditVerificationError("review-only driver changed or executed")
    _validate_self_hash(value, "record_self_sha256")


def _validate_prohibited_counts(record: Mapping[str, Any]) -> None:
    value = _exact(
        record, {"schema", "state", "counts", "all_counts_zero", "record_self_sha256"},
        "prohibited-action counts",
    )
    if (
        value["schema"] != MEMBER_SCHEMAS["prohibited_action_counts.json"]
        or value["state"] != STATE or value["all_counts_zero"] is not True
    ):
        raise State3AuditVerificationError("prohibited-action identity mismatch")
    counts = _exact(value["counts"], PROHIBITED_COUNT_NAMES, "prohibited-action map")
    if any(_nonnegative_integer(item, name) != 0 for name, item in counts.items()):
        raise State3AuditVerificationError("prohibited-action count nonzero")
    _validate_self_hash(value, "record_self_sha256")


def _validate_root(record: Mapping[str, Any]) -> None:
    value = _exact(record, ROOT_KEYS, "State 3 audit root")
    if (
        value["schema"] != MEMBER_SCHEMAS["state3_audit.json"]
        or value["state"] != STATE or value["outcome"] != OUTCOME
        or value["classification"] != CLASSIFICATION
        or value["protocol_file_sha256"] != PROTOCOL_SHA256
        or value["solver_spec_baseline_commit"] != BASELINE_COMMIT
        or value["solver_spec_baseline_tree"] != BASELINE_TREE
    ):
        raise State3AuditVerificationError("State 3 audit root identity mismatch")
    for name in (
        "protocol_file_sha256", "state3_authorization_file_sha256",
        "measurement_bundle_self_sha256", "environment_runtime_file_sha256",
        "storage_preflight_before_file_sha256", "compute_feasibility_report_file_sha256",
        "storage_preflight_after_file_sha256", "source_blob_verification_file_sha256",
        "failure_inventory_file_sha256", "prohibited_action_counts_file_sha256",
        "record_self_sha256",
    ):
        _sha(value[name], name)
    for name in (
        "solver_spec_baseline_commit", "solver_spec_baseline_tree",
        "implementation_commit", "implementation_tree",
    ):
        _oid(value[name], name)
    before = _validate_git(value["git_before"], value["implementation_commit"], value["implementation_tree"])
    after = _validate_git(value["git_after"], value["implementation_commit"], value["implementation_tree"])
    if _canonical(before) != _canonical(after):
        raise State3AuditVerificationError("State 3 root Git state changed")
    expected_flags = {
        "storage_preflight_records_match": True,
        "contains_corpus_controls_or_solutions": False,
        "contains_scientific_arrays": False,
        "external_sharing_authorized": False,
        "state4_or_later_authorized": False,
        "all_checks_passed": True,
    }
    if any(value[name] is not expected for name, expected in expected_flags.items()):
        raise State3AuditVerificationError("State 3 audit safety flag mismatch")
    _validate_self_hash(value, "record_self_sha256")


def _storage_records_match(
    before: Mapping[str, Any], after: Mapping[str, Any], *, expected_parent_commit: str,
) -> None:
    left = json.loads(_canonical(before))
    right = json.loads(_canonical(after))
    left.pop("record_self_sha256")
    right.pop("record_self_sha256")
    left["observed_free_bytes"] = "<VOLATILE_FREE_SPACE>"
    right["observed_free_bytes"] = "<VOLATILE_FREE_SPACE>"
    left["volume_evidence"]["free_bytes"] = "<VOLATILE_FREE_SPACE>"
    right["volume_evidence"]["free_bytes"] = "<VOLATILE_FREE_SPACE>"
    if left != right:
        raise State3AuditVerificationError("storage preflight changed outside free-space fields")
    if any(
        record["git_tracking_evidence"]["parent_commit"] != expected_parent_commit
        for record in (before, after)
    ):
        raise State3AuditVerificationError(
            "storage parent differs from independently validated source parent"
        )


def _validate_complete_records(
    records: Mapping[str, Mapping[str, Any]], member_bytes: Mapping[str, bytes],
) -> None:
    root = records["state3_audit.json"]
    environment = records["environment_runtime.json"]
    before = records["storage_preflight_before.json"]
    report = records["compute_feasibility_report.json"]
    after = records["storage_preflight_after.json"]
    source = records["source_blob_verification.json"]
    failure = records["failure_inventory.json"]
    prohibited = records["prohibited_action_counts.json"]
    _validate_root(root)
    _validate_environment(environment)
    _validate_storage(before)
    _validate_report(report)
    _validate_storage(after)
    _validate_source(source)
    _validate_failure_inventory(failure)
    _validate_prohibited_counts(prohibited)
    for name, record in records.items():
        if record.get("schema") != MEMBER_SCHEMAS[name]:
            raise State3AuditVerificationError(f"member schema mismatch: {name}")
        _metadata_only(record)
    authorization = root["state3_authorization_file_sha256"]
    identity = (
        root["protocol_file_sha256"], authorization,
        root["implementation_commit"], root["implementation_tree"],
    )
    for name in (
        "environment_runtime.json", "compute_feasibility_report.json",
        "source_blob_verification.json",
    ):
        record = records[name]
        if (
            record["protocol_file_sha256"], record["state3_authorization_file_sha256"],
            record["implementation_commit"], record["implementation_tree"],
        ) != identity:
            raise State3AuditVerificationError(f"common identity mismatch: {name}")
    for name in ("storage_preflight_before.json", "storage_preflight_after.json"):
        record = records[name]
        if (
            record["protocol_file_sha256"], record["authorization_file_sha256"],
            record["implementation_commit"], record["implementation_tree"],
        ) != identity:
            raise State3AuditVerificationError(f"storage identity mismatch: {name}")
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
    if any(root[field] != observed[name] for name, field in links.items()):
        raise State3AuditVerificationError("root member-hash binding mismatch")
    if (
        report["environment_record_sha256"] != observed["environment_runtime.json"]
        or report["storage_preflight_before_file_sha256"]
        != observed["storage_preflight_before.json"]
        or report["storage_preflight_after_file_sha256"]
        != observed["storage_preflight_after.json"]
        or report["launcher_file_sha256"] != environment["launcher_file_sha256"]
        or report["runtime_inspector_file_sha256"] != environment["runtime_inspector_file_sha256"]
        or environment["requirements_lock_git_blob"] != source["requirements_lock_git_blob"]
        or environment["requirements_lock_file_sha256"] != source["requirements_lock_file_sha256"]
        or source["solver_spec_baseline_commit"] != report["solver_spec_baseline_commit"]
        or source["solver_spec_baseline_tree"] != report["solver_spec_baseline_tree"]
    ):
        raise State3AuditVerificationError("report/environment/source binding mismatch")
    _storage_records_match(
        before, after, expected_parent_commit=source["implementation_parent_commit"]
    )
    disk = report["disk_estimate"]
    for storage in (before, after):
        if (
            storage["estimated_accepted_bytes"] != disk["estimated_accepted_bytes"]
            or storage["projected_audit_bytes"] != disk["projected_audit_bytes"]
            or storage["required_free_bytes"] != disk["total_required_free_space_bytes"]
        ):
            raise State3AuditVerificationError("storage/report/source binding mismatch")
    if (
        root["git_before"] != source["git_before"] or root["git_after"] != source["git_after"]
        or report["git_before"] != source["git_before"] or report["git_after"] != source["git_after"]
    ):
        raise State3AuditVerificationError("audit/report/source Git binding mismatch")
    counts = prohibited["counts"]
    if (
        counts["forbidden_rng_call_count"] != report["forbidden_rng_call_count"]
        or counts["corpus_split_seed_access_count"] != report["forbidden_corpus_seed_access_count"]
        or failure["expected_record_count"] != 17
        or failure["original_relocated_record_count"] != 12
        or failure["in_place_preserved_record_count"] != 5
        or failure["historical_overbroad_pytest_prohibited_action_accounting"]
        != "not_provably_zero"
    ):
        raise State3AuditVerificationError("count/failure cross-binding mismatch")


def _validate_manifest(
    manifest: Mapping[str, Any], *, expected_archive_filename: str,
    member_bytes: Mapping[str, bytes],
) -> None:
    value = _exact(manifest, MANIFEST_KEYS, "member-hash manifest")
    if (
        value["schema"] != MANIFEST_SCHEMA
        or value["archive_filename"] != expected_archive_filename
        or value["member_order"] != list(MEMBER_ORDER)
    ):
        raise State3AuditVerificationError("member manifest identity/order mismatch")
    if PurePosixPath(expected_archive_filename).name != expected_archive_filename or not (
        expected_archive_filename.casefold().endswith(".zip")
    ):
        raise State3AuditVerificationError("expected archive filename is not one plain ZIP name")
    rows = value["members_excluding_manifest"]
    if not isinstance(rows, list) or len(rows) != len(MEMBER_ORDER) - 1:
        raise State3AuditVerificationError("manifest member rows incomplete")
    for row, name in zip(rows, MEMBER_ORDER[:-1], strict=True):
        item = _exact(row, MANIFEST_ROW_KEYS, "manifest member row")
        if (
            item["path"] != name
            or _nonnegative_integer(item["size"], "manifest member size")
            != len(member_bytes[name])
            or _sha(item["sha256"], "manifest member hash")
            != hashlib.sha256(member_bytes[name]).hexdigest()
        ):
            raise State3AuditVerificationError("manifest member size/hash mismatch")
    _validate_self_hash(value, "manifest_self_sha256")


def _validate_raw_zip32_structure(archive_bytes: bytes) -> None:
    """Consume the entire frozen stored-ZIP32 layout, independently of zipfile.

    Signatures inside payloads are ordinary data. Only length-delimited record
    boundaries are interpreted; no signature search or producer reserialization
    can hide a prefix, suffix, gap, overlapping record or local/central mismatch.
    This proves container structure only. All existing semantic checks still run.
    """
    if type(archive_bytes) is not bytes or len(archive_bytes) < 22:
        raise State3AuditVerificationError("raw ZIP32: missing immutable archive bytes/EOCD")
    end_offset = len(archive_bytes) - 22
    end = struct.unpack_from("<4s4H2IH", archive_bytes, end_offset)
    if (
        end[0] != b"PK\x05\x06" or end[1:3] != (0, 0)
        or end[3:5] != (len(MEMBER_ORDER), len(MEMBER_ORDER)) or end[7] != 0
    ):
        raise State3AuditVerificationError("raw ZIP32: noncanonical final EOCD")
    directory_size, directory_offset = end[5:7]
    if (
        directory_size == 0xFFFFFFFF or directory_offset == 0xFFFFFFFF
        or directory_offset > end_offset
        or directory_size != end_offset - directory_offset
    ):
        raise State3AuditVerificationError("raw ZIP32: central directory extent mismatch")

    offset = 0
    payload_total = 0
    # (encoded name, actual local offset, CRC, payload size) from local bytes.
    local_records: list[tuple[bytes, int, int, int]] = []
    for name in MEMBER_ORDER:
        encoded_name = name.encode("ascii")
        if offset + 30 > directory_offset:
            raise State3AuditVerificationError("raw ZIP32: truncated local header")
        local = struct.unpack_from("<4s5H3I2H", archive_bytes, offset)
        if (
            local[0] != b"PK\x03\x04"
            or local[1:6] != (20, 0, 0, 0, 33)
            or local[7] == 0xFFFFFFFF or local[8] == 0xFFFFFFFF
            or local[7] != local[8]
            or local[9] != len(encoded_name) or local[10] != 0
        ):
            raise State3AuditVerificationError("raw ZIP32: noncanonical local header: " + name)
        payload_start = offset + 30 + len(encoded_name)
        payload_end = payload_start + local[7]
        if payload_end > directory_offset:
            raise State3AuditVerificationError("raw ZIP32: local payload exceeds its region")
        if archive_bytes[offset + 30:payload_start] != encoded_name:
            raise State3AuditVerificationError("raw ZIP32: local member name/order mismatch")
        actual_crc = zlib.crc32(memoryview(archive_bytes)[payload_start:payload_end]) & 0xFFFFFFFF
        if actual_crc != local[6]:
            raise State3AuditVerificationError("raw ZIP32: local payload CRC mismatch")
        local_records.append((encoded_name, offset, local[6], local[7]))
        payload_total += local[7]
        offset = payload_end
    if offset != directory_offset:
        raise State3AuditVerificationError("raw ZIP32: unclaimed bytes before central directory")

    for encoded_name, local_offset, crc, size in local_records:
        if offset + 46 > end_offset:
            raise State3AuditVerificationError("raw ZIP32: truncated central header")
        central = struct.unpack_from("<4s6H3I5H2I", archive_bytes, offset)
        if (
            central[0] != b"PK\x01\x02"
            or central[1:7] != (20, 20, 0, 0, 0, 33)
            or central[7:10] != (crc, size, size)
            or central[10:15] != (len(encoded_name), 0, 0, 0, 0)
            or central[15] != FIXED_EXTERNAL_ATTR or central[16] != local_offset
            or central[16] == 0xFFFFFFFF
        ):
            raise State3AuditVerificationError("raw ZIP32: central/local header mismatch")
        next_offset = offset + 46 + len(encoded_name)
        if next_offset > end_offset or archive_bytes[offset + 46:next_offset] != encoded_name:
            raise State3AuditVerificationError("raw ZIP32: central member name/extent mismatch")
        offset = next_offset
    if offset != end_offset:
        raise State3AuditVerificationError("raw ZIP32: unclaimed central-directory bytes")
    overhead = 22 + sum(30 + 46 + 2 * len(name) for name, _, _, _ in local_records)
    if overhead != 1174 or len(archive_bytes) != payload_total + overhead:
        raise State3AuditVerificationError("raw ZIP32: frozen structural byte equation mismatch")


def verify_state3_audit(
    archive_path: str | Path, *, expected_archive_filename: str | None = None,
    expected_authorization_file_sha256: str | None = None,
    expected_implementation_commit: str | None = None,
    expected_implementation_tree: str | None = None,
) -> dict[str, Any]:
    """Independently verify one complete deterministic metadata-only State 3 archive."""
    path = Path(archive_path)
    if path.is_symlink() or not path.is_file():
        raise State3AuditVerificationError("State 3 archive is not one regular file")
    if "private_corpus_v1" in {part.casefold() for part in path.resolve().parts}:
        raise State3AuditVerificationError("State 3 archive is inside the private corpus")
    expected_name = path.name if expected_archive_filename is None else expected_archive_filename
    archive_bytes = path.read_bytes()
    if not archive_bytes:
        raise State3AuditVerificationError("State 3 archive is empty")
    _validate_raw_zip32_structure(archive_bytes)
    member_bytes: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if names != list(MEMBER_ORDER) or len(set(names)) != len(names):
                raise State3AuditVerificationError("archive member order/set mismatch")
            if archive.comment != b"" or archive.testzip() is not None:
                raise State3AuditVerificationError("archive comment or CRC integrity invalid")
            for info, name in zip(infos, MEMBER_ORDER, strict=True):
                member_path = PurePosixPath(name)
                if (
                    info.filename != name or member_path.name != name or len(member_path.parts) != 1
                    or not name.endswith(".json") or info.is_dir()
                    or info.date_time != FIXED_TIMESTAMP or info.compress_type != zipfile.ZIP_STORED
                    or info.create_system != 0 or info.create_version != 20
                    or info.extract_version != 20 or info.flag_bits != 0 or info.volume != 0
                    or info.internal_attr != 0 or info.external_attr != FIXED_EXTERNAL_ATTR
                    or info.extra != b"" or info.comment != b""
                ):
                    raise State3AuditVerificationError(
                        f"archive member metadata is non-deterministic: {name}"
                    )
                data = archive.read(info)
                if (
                    len(data) != info.file_size or info.compress_size != info.file_size
                    or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC
                ):
                    raise State3AuditVerificationError(f"archive member integrity mismatch: {name}")
                member_bytes[name] = data
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, State3AuditVerificationError):
            raise
        raise State3AuditVerificationError("invalid State 3 ZIP container") from exc
    records: dict[str, Mapping[str, Any]] = {}
    for name in MEMBER_ORDER[:-1]:
        record = _load_canonical(member_bytes[name])
        if record.get("schema") != MEMBER_SCHEMAS[name]:
            raise State3AuditVerificationError(f"archive member schema mismatch: {name}")
        _metadata_only(record)
        records[name] = record
    manifest = _load_canonical(member_bytes[MEMBER_ORDER[-1]])
    _metadata_only(manifest)
    _validate_manifest(
        manifest,
        expected_archive_filename=expected_name,
        member_bytes=member_bytes,
    )
    _validate_complete_records(records, member_bytes)
    root = records["state3_audit.json"]
    for expected, field, label, validator in (
        (
            expected_authorization_file_sha256,
            "state3_authorization_file_sha256",
            "authorization-file SHA-256",
            _sha,
        ),
        (
            expected_implementation_commit,
            "implementation_commit",
            "implementation commit",
            _oid,
        ),
        (
            expected_implementation_tree,
            "implementation_tree",
            "implementation tree",
            _oid,
        ),
    ):
        if expected is not None:
            validator(expected, f"expected {label}")
            observed = root[field]
            if expected != observed:
                raise State3AuditVerificationError(f"runtime {label} binding mismatch")
    return {
        "archive_filename": expected_name,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_size": len(archive_bytes),
        "canonical_json_valid": True,
        "crc_integrity": True,
        "member_count": len(MEMBER_ORDER),
        "member_hash_manifest_verified": True,
        "member_order": list(MEMBER_ORDER),
        "outcome": "verified",
        "semantic_cross_bindings_verified": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify a metadata-only OperatorMark State 3 audit ZIP"
    )
    parser.add_argument("archive")
    parser.add_argument("--expected-archive-filename")
    parser.add_argument("--authorization-file-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--implementation-tree", required=True)
    arguments = parser.parse_args(argv)
    result = verify_state3_audit(
        arguments.archive,
        expected_archive_filename=arguments.expected_archive_filename,
        expected_authorization_file_sha256=arguments.authorization_file_sha256,
        expected_implementation_commit=arguments.implementation_commit,
        expected_implementation_tree=arguments.implementation_tree,
    )
    sys.stdout.write(_canonical(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
