"""Frozen corpus-family facts and adopted final-record schema vocabulary.

This module is intentionally standard-library-only.  Constants labelled as
frozen facts come directly from the frozen corpus protocol; the final context,
manifest, and rights schema identifiers are the delegated representation
choices adopted for the Section 9 format package.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping


# Frozen protocol facts.
BENCHMARK: Final = "darcy_branch_c_v1"
PROTOCOL_SHA256: Final = (
    "a38ef79b6c4bdccf53f599fa08f77e48a111afbf0b2783ad5733bda99996599a"
)
SOLVER_SPEC_BASELINE_COMMIT: Final = "e16456395cfe67b73b9d298fd0491cf9524d798a"
SOLVER_SPEC_BASELINE_TREE: Final = "81f40e24102e51e0da3ce65dafd769a7678565fd"

# Delegated Section 9 representation choice.  The value names the frozen
# destination-directory version component; the protocol declares the
# ``corpus_version`` field but does not separately state its value literal.
CORPUS_VERSION: Final = "private_corpus_v1"

SHARD_SIZE: Final = 64
SPLITS: Final = ("training", "validation", "test")
SPLIT_SAMPLE_COUNTS: Final[Mapping[str, int]] = MappingProxyType(
    {"training": 4096, "validation": 256, "test": 512}
)
SPLIT_SHARD_COUNTS: Final[Mapping[str, int]] = MappingProxyType(
    {"training": 64, "validation": 4, "test": 8}
)
SPLIT_GLOBAL_RANGES: Final[Mapping[str, tuple[int, int]]] = MappingProxyType(
    {"training": (0, 4095), "validation": (4096, 4351), "test": (4352, 4863)}
)
SPLIT_STORAGE_DIRECTORIES: Final[Mapping[str, str]] = MappingProxyType(
    {"training": "training", "validation": "validation", "test": "test.sealed"}
)

CANONICAL_TENSOR_DTYPE: Final = "<f8"
CANONICAL_TENSOR_ORDER: Final = "C"
CANONICAL_NPY_VERSION: Final = (2, 0)
VERIFICATION_FILENAME: Final = "verification.jsonl"
SHARD_MANIFEST_FILENAME: Final = "shard_manifest.json"


@dataclass(frozen=True, slots=True)
class TensorFamilySpec:
    """One frozen stored tensor family, including its physical shard shape."""

    name: str
    filename: str
    per_sample_shape: tuple[int, int, int]
    shard_shape: tuple[int, int, int, int]


TENSOR_FAMILY_ORDER: Final = (
    "rho",
    "kappa64",
    "u_tpfa64",
    "kappa32_areaavg",
    "u32_areaavg",
    "kappa16_areaavg",
    "u16_areaavg",
)
_TENSOR_FAMILY_ROWS = (
    TensorFamilySpec("rho", "rho.float64.npy", (1, 8, 8), (64, 1, 8, 8)),
    TensorFamilySpec("kappa64", "kappa64.float64.npy", (1, 64, 64), (64, 1, 64, 64)),
    TensorFamilySpec("u_tpfa64", "u_tpfa64.float64.npy", (1, 64, 64), (64, 1, 64, 64)),
    TensorFamilySpec(
        "kappa32_areaavg", "kappa32_areaavg.float64.npy", (1, 32, 32), (64, 1, 32, 32)
    ),
    TensorFamilySpec("u32_areaavg", "u32_areaavg.float64.npy", (1, 32, 32), (64, 1, 32, 32)),
    TensorFamilySpec(
        "kappa16_areaavg", "kappa16_areaavg.float64.npy", (1, 16, 16), (64, 1, 16, 16)
    ),
    TensorFamilySpec("u16_areaavg", "u16_areaavg.float64.npy", (1, 16, 16), (64, 1, 16, 16)),
)
TENSOR_FAMILY_SPECS: Final[Mapping[str, TensorFamilySpec]] = MappingProxyType(
    {row.name: row for row in _TENSOR_FAMILY_ROWS}
)
FINAL_SHARD_MEMBER_ORDER: Final = tuple(
    row.filename for row in _TENSOR_FAMILY_ROWS
) + (VERIFICATION_FILENAME,)

# Frozen verification schema and nested fields.
FINAL_VERIFICATION_SCHEMA: Final = "operatormark.darcy_branch_c_v1.sample_verification.v1"
FINAL_VERIFICATION_KEYS: Final = frozenset(
    {
        "schema",
        "benchmark",
        "split",
        "shard_index",
        "local_index",
        "global_index",
        "sample_id",
        "stratum",
        "control_tensor_sha256",
        "rho_storage_tensor_sha256",
        "macro_coefficient_tensor_sha256",
        "tensor_sha256",
        "verifier_tensor_sha256",
        "solver_diagnostics",
        "metrics",
        "gates",
        "record_self_sha256",
    }
)
FINAL_TENSOR_SHA256_KEYS: Final = frozenset(TENSOR_FAMILY_ORDER)
VERIFIER_TENSOR_SHA256_KEYS: Final = frozenset({"P256", "bar_P256"})
SOLVER_DIAGNOSTIC_KEYS: Final = frozenset({"T64", "P256"})
T64_DIAGNOSTIC_KEYS: Final = frozenset(
    {
        "algebraic_backward_error",
        "boundary_transmissibility_error",
        "maximum_local_conservation_ratio",
        "maximum_interior_face_antisymmetry",
        "global_balance_ratio",
        "rhs_relative_l2_residual",
        "passes",
    }
)
P256_DIAGNOSTIC_KEYS: Final = frozenset(
    {
        "algebraic_backward_error",
        "boundary_reconstruction_error",
        "reaction_denominator",
        "reaction_balance_ratio",
        "outward_boundary_flux",
        "integrated_source",
        "rhs_relative_l2_residual",
        "iterations",
        "passes",
    }
)
METRICS_KEYS: Final = frozenset({"scales", "compliances", "delta_L2", "delta_J"})
METRIC_SCALE_KEYS: Final = frozenset({"S_L2", "S_J"})
METRIC_COMPLIANCE_KEYS: Final = frozenset({"T64", "bar_P256"})
FINAL_GATE_KEYS: Final = frozenset(
    {"structural", "TPFA64", "P1_256", "cross_solver", "overall"}
)

ALGEBRAIC_ERROR_GATE: Final = 1.0e-10
BOUNDARY_ERROR_GATE: Final = 100.0 * 2.220446049250313e-16
CONSERVATION_ERROR_GATE: Final = 1.0e-10
TAU_L2: Final = 0.02543117962436324
TAU_J: Final = 0.022582030716092032
P256_MAX_ITERATIONS: Final = 10000

# Reference schemas used by the delegated final records.
LOCAL_JSON_REFERENCE_KEYS: Final = frozenset({"relative_path", "size", "sha256", "schema"})
LOCAL_OPAQUE_REFERENCE_KEYS: Final = frozenset({"relative_path", "size", "sha256"})
EXTERNAL_REFERENCE_KEYS: Final = frozenset(
    {"filename", "literal_path_sha256", "sha256", "size"}
)

# Delegated final-record identifiers and exact key sets.
FINAL_CONTEXT_SCHEMA: Final = "operatormark.darcy_branch_c_v1.final_context.v1"
FINAL_CONTEXT_KEYS: Final = frozenset(
    {
        "schema",
        "benchmark",
        "corpus_version",
        "attempt_id",
        "protocol_file_sha256",
        "solver_spec_baseline_commit",
        "solver_spec_baseline_tree",
        "implementation_commit",
        "implementation_tree",
        "state4_authorization_file_sha256",
        "state5_authorization_file_sha256",
        "state6_authorization_file_sha256",
        "pre_solve_manifest",
        "controls_completion",
        "split_order",
        "record_self_sha256",
    }
)

FINAL_SHARD_MANIFEST_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.final_shard_manifest.v1"
)
FINAL_SHARD_MANIFEST_KEYS: Final = frozenset(
    {
        "schema",
        "benchmark",
        "split",
        "shard_index",
        "sample_count",
        "first_local_index",
        "last_local_index",
        "first_global_index",
        "last_global_index",
        "ordered_sample_ids",
        "parent_context",
        "pre_solve_manifest",
        "state4_controls_shard_manifest",
        "members",
        "manifest_self_sha256",
    }
)
FINAL_NPY_MEMBER_KEYS: Final = frozenset(
    {
        "relative_path",
        "size",
        "sha256",
        "tensor_name",
        "dtype",
        "shape",
        "order",
        "npy_version",
        "tensor_sha256",
    }
)
FINAL_JSONL_MEMBER_KEYS: Final = frozenset(
    {
        "relative_path",
        "size",
        "sha256",
        "schema",
        "record_count",
        "ordered_record_self_sha256",
    }
)

FINAL_SPLIT_MANIFEST_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.final_split_manifest.v1"
)
FINAL_SPLIT_MANIFEST_KEYS: Final = frozenset(
    {
        "schema",
        "benchmark",
        "split",
        "sealed",
        "sample_count",
        "shard_count",
        "parent_context",
        "state4_split_manifest",
        "seed_binding",
        "ordered_shard_manifests",
        "split_tensor_sha256",
        "access_state",
        "manifest_self_sha256",
    }
)
SEED_BINDING_KEYS: Final = frozenset(
    {"state4_split_manifest_file_sha256", "json_pointer"}
)
SPLIT_ACCESS_STATES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "training": "accepted_training_authorization_required",
        "validation": "accepted_validation_authorization_required",
        "test": "sealed_test_opening_authorization_required",
    }
)

GENERATED_DATA_RIGHTS_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.generated_data_rights.v1"
)
GENERATED_DATA_RIGHTS_KEYS: Final = frozenset(
    {
        "schema",
        "benchmark",
        "identifier",
        "gate9_record_file_sha256",
        "rights_text_json_pointer",
        "rights_text",
        "rights_text_sha256",
        "third_party_inventory_file_sha256",
        "record_self_sha256",
    }
)
GENERATED_DATA_RIGHTS_IDENTIFIER: Final = (
    "LicenseRef-OperatorMark-Private-Generated-Data-v1"
)
GENERATED_DATA_RIGHTS_JSON_POINTER: Final = (
    "#/license_refs/LicenseRef-OperatorMark-Private-Generated-Data-v1/"
    "full_internal_rights_position_text"
)
GENERATED_DATA_RIGHTS_TEXT: Final = (
    "LicenseRef-OperatorMark-Private-Generated-Data-v1 is an internal prospective rights "
    "and confidentiality position for future OperatorMark-generated data, if any is later "
    "generated under separate authorization. No such dataset is authorized or established "
    "by Gate 9. Subject to unresolved source-data terms, database rights, chain-of-title "
    "facts, and third-party rights, the user-asserted owner provisionally intends proprietary "
    "and confidential treatment, all rights reserved, with no third-party use, copying, "
    "publication, disclosure, or distribution right granted. This is not a standard SPDX "
    "licence, is not a complete legal licence, has not been drafted or reviewed by a lawyer, "
    "does not override any third-party licence or pre-existing right, and does not guarantee "
    "ownership, database rights, copyright, confidentiality, trade-secret status, validity, "
    "or enforceability."
)
GENERATED_DATA_RIGHTS_TEXT_SHA256: Final = (
    "7fa6d35e24e7d810b4cbd407272550536ec4d7610a65f144c3c41b43bd8bc5bf"
)
GATE9_RECORD_FILE_SHA256: Final = (
    "655096db303f39587b35f43442debe9c19380895d9978418e1aec7c6b22c0093"
)
THIRD_PARTY_INVENTORY_FILE_SHA256: Final = (
    "1fff86a00cb8728e89ee9fafb140979f63e1d65a7fc9ebd1edd7201925f3ffee"
)

# Existing State 4 schemas referenced by the final records.
STATE4_PRE_SOLVE_MANIFEST_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.state4_pre_solve_manifest.v2"
)
STATE4_CONTROLS_COMPLETE_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.state4_controls_complete.v1"
)
STATE4_CONTROLS_SHARD_MANIFEST_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.state4_controls_shard_manifest.v1"
)
