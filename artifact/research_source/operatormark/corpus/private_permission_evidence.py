"""Inactive private permission-evidence codecs and injected retention protocol.

This module deliberately has no concrete filesystem, Windows, authority, or corpus
integration.  Every effectful operation is supplied by a caller-owned protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
import re
import unicodedata
from typing import Literal, Protocol, TypeAlias

from operatormark.corpus.canonical import (
    canonical_json_bytes,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    strict_canonical_json_loads,
    validate_self_hash,
    with_self_hash,
)
from operatormark.corpus.permission_observation_windows import (
    AuthorityAcquireOutcome,
    AuthorityAccounting,
    AuthorityCheckOutcome,
    AuthorityFinishOutcome,
    CapturedPermissionObservation,
    ComponentFailureItem,
    FailedPermissionObservation,
    PermissionAccessExpectation,
    PermissionAuthorityCheckpoint,
    PermissionObservationAuthority,
    PermissionObservationRequest,
    PermissionOperationScope,
    validate_permission_access_expectation,
    validate_permission_observation_request,
)
from operatormark.corpus import permission_authz_records as authz_records


OBSERVATION_SCHEMA = (
    "operatormark.darcy_branch_c_v1.private_file_permission_observation.v2"
)
OBSERVATION_FAILURE_SCHEMA = (
    "operatormark.darcy_branch_c_v1.private_file_permission_observation_failure.v2"
)
RETENTION_FAILURE_SCHEMA = (
    "operatormark.darcy_branch_c_v1.private_permission_evidence_retention_failure.v2"
)
RETENTION_FAILURE_RESULT_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_evidence_retention_failure_result.v2"
)
AUTHZ_DESTINATION_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_authz_evidence_destination.v1"
)
AUTHZ_RETENTION_FAILURE_SCHEMA = (
    "operatormark.darcy_branch_c_v1.private_permission_authz_retention_failure.v1"
)
AUTHZ_RETENTION_FAILURE_RESULT_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_authz_retention_failure_result.v1"
)
RETAINED_AUTHZ_EVIDENCE_SCHEMA = (
    "operatormark.darcy_branch_c_v1.retained_private_permission_authz_evidence.v1"
)
STORE_EXPECTATION_SCHEMA = (
    "operatormark.darcy_branch_c_v1.confidential_permission_store_expectation.v1"
)
DESTINATION_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_evidence_destination.v2"
)
SCOPE_SCHEMA = "operatormark.darcy_branch_c_v1.permission_operation_scope.v1"
OBSERVATION_DOMAIN = "operatormark-state4-private-file-permission-observation-v1"
OBSERVATION_FAILURE_DOMAIN = (
    "operatormark-state4-private-file-permission-observation-failure-v2"
)
RETENTION_FAILURE_DOMAIN = (
    "operatormark-state4-private-permission-evidence-retention-failure-v2"
)
AUTHZ_RETENTION_FAILURE_DOMAIN = (
    "operatormark-state4-private-permission-authz-retention-failure-v1"
)
PRIVATE_ROOT = "artifacts/darcy_branch_c_v1/private_corpus_v1/permission_evidence"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DWORD_RE = re.compile(r"^0x(?:0|[1-9a-f][0-9a-f]*)$")
_BINARY_HEX_RE = re.compile(r"^(?:[0-9a-f]{2})*$")


class PrivatePermissionEvidenceContractError(RuntimeError):
    """Raised before effects when a value violates the private contract."""


class PrivatePermissionEvidenceProtocolError(RuntimeError):
    """Raised only for impossible local construction/programmer errors."""


@dataclass(frozen=True, slots=True)
class ConfidentialStoreExpectation:
    schema: str
    request_self_sha256: str
    attempt_id: str
    namespace_relative_path: str
    namespace_object_identity: str
    records_parent_relative_path: str
    records_parent_object_identity: str
    volume_identifier_commitment: str
    confidential_store_profile_sha256: str
    records_parent_security_descriptor_sha256: str
    required_created_object_security_descriptor_sha256: str
    creation_security_policy: str
    publication_policy: str
    expectation_self_sha256: str


@dataclass(frozen=True, slots=True)
class PermissionEvidenceDestination:
    schema: str
    payload_kind: Literal["observation", "observation_failure"]
    attempt_id: str
    request_self_sha256: str
    observation_sequence: int
    observation_phase: str
    namespace_relative_path: str
    namespace_object_identity: str
    records_parent_relative_path: str
    records_parent_object_identity: str
    volume_identifier_commitment: str
    confidential_store_profile_sha256: str
    required_created_object_security_descriptor_sha256: str
    primary_creating_relative_path: str
    primary_accepted_relative_path: str
    retention_failure_creating_relative_path: str
    retention_failure_accepted_relative_path: str
    store_expectation_self_sha256: str
    destination_self_sha256: str


@dataclass(frozen=True, slots=True)
class PermissionAuthzEvidenceDestination:
    schema: str
    payload_kind: Literal["authz_evidence"]
    attempt_id: str
    request_self_sha256: str
    observation_sequence: int
    observation_phase: str
    namespace_relative_path: str
    namespace_object_identity: str
    records_parent_relative_path: str
    records_parent_object_identity: str
    volume_identifier_commitment: str
    confidential_store_profile_sha256: str
    required_created_object_security_descriptor_sha256: str
    primary_creating_relative_path: str
    primary_accepted_relative_path: str
    retention_failure_creating_relative_path: str
    retention_failure_accepted_relative_path: str
    store_expectation_self_sha256: str
    authz_binding_self_sha256: str
    destination_self_sha256: str


@dataclass(frozen=True, slots=True)
class StoreParentSnapshot:
    relative_path: str
    object_identity: str
    namespace_object_identity: str
    volume_identifier_commitment: str
    security_descriptor_sha256: str
    confidential_store_profile_sha256: str


@dataclass(frozen=True, slots=True)
class ConfidentialNamespaceSnapshot:
    namespace_relative_path: str
    namespace_object_identity: str
    volume_identifier_commitment: str
    confidential_store_profile_sha256: str
    records_parent: StoreParentSnapshot


@dataclass(frozen=True, slots=True)
class StoreCreationEvidence:
    creation_disposition: str
    prewrite_size_bytes: int
    namespace_object_identity: str
    parent_relative_path: str
    parent_object_identity: str
    object_identity: str
    volume_identifier_commitment: str
    creation_security_descriptor_sha256: str
    confidential_store_profile_sha256: str
    security_applied_in_create: bool


@dataclass(frozen=True, slots=True)
class StoreObjectSnapshot:
    relative_path: str
    namespace_object_identity: str
    parent_relative_path: str
    parent_object_identity: str
    object_identity: str
    volume_identifier_commitment: str
    size_bytes: int
    file_sha256: str
    security_descriptor_sha256: str
    confidential_store_profile_sha256: str
    publication_state: Literal["creating", "published"]


@dataclass(frozen=True, slots=True)
class StoreFailureEffect:
    operation: str
    completion_status: Literal["succeeded", "collision", "failed"]
    effect_certainty: Literal["no_effect", "reported_effect", "unknown"]
    before_snapshot: StoreObjectSnapshot | None
    after_snapshot: StoreObjectSnapshot | None
    collision_relative_path: str | None
    bytes_requested: int | None
    bytes_reported: int | None
    backend_error_hex: str | None


@dataclass(frozen=True, slots=True)
class StoreNamespaceEffect:
    operation: Literal["open_namespace", "release_namespace"]
    completion_status: Literal["succeeded", "failed"]
    effect_certainty: Literal["no_effect", "reported_effect", "unknown"]
    before_snapshot: ConfidentialNamespaceSnapshot | None
    after_snapshot: ConfidentialNamespaceSnapshot | None
    backend_error_hex: str | None


@dataclass(frozen=True, slots=True)
class StoreObjectTerminalState:
    slot_kind: Literal[
        "primary", "retention_failure", "authz_primary", "authz_retention_failure"
    ]
    payload_kind: Literal[
        "observation",
        "observation_failure",
        "retention_failure",
        "authz_evidence",
        "authz_retention_failure",
    ]
    ownership: Literal[
        "none", "unowned_collision", "owned_by_this_invocation", "unknown"
    ]
    effect_certainty: Literal["no_effect", "reported_effect", "unknown"]
    creating_relative_path: str
    accepted_relative_path: str
    last_confirmed_relative_path: str | None
    reported_relative_path: str | None
    object_identity: str | None
    volume_identifier_commitment: str | None
    last_confirmed_size_bytes: int | None
    last_confirmed_file_sha256: str | None
    publication_state: str
    handle_state: Literal["not_acquired", "released", "release_failed", "unknown"]


@dataclass(frozen=True, slots=True)
class NamespaceOpenOutcome:
    status: Literal["opened", "failed"]
    lease: ConfidentialNamespaceLease | None
    effect: StoreNamespaceEffect
    failure: ComponentFailureItem | None


@dataclass(frozen=True, slots=True)
class StoreCreateOutcome:
    status: Literal["created", "collision", "failed"]
    object_handle: ConfidentialStoreObject | None
    creation_evidence: StoreCreationEvidence | None
    effect: StoreFailureEffect
    failure: ComponentFailureItem | None


@dataclass(frozen=True, slots=True)
class StoreIoOutcome:
    status: Literal["succeeded", "failed"]
    effect: StoreFailureEffect
    failure: ComponentFailureItem | None


@dataclass(frozen=True, slots=True)
class StoreBytesReadOutcome:
    status: Literal["succeeded", "failed"]
    data: bytes | None
    effect: StoreFailureEffect
    failure: ComponentFailureItem | None


@dataclass(frozen=True, slots=True)
class StoreSnapshotOutcome:
    status: Literal["succeeded", "failed"]
    snapshot: StoreObjectSnapshot | None
    effect: StoreFailureEffect
    failure: ComponentFailureItem | None


@dataclass(frozen=True, slots=True)
class StorePromoteOutcome:
    status: Literal["published", "collision", "failed"]
    effect: StoreFailureEffect
    failure: ComponentFailureItem | None


@dataclass(frozen=True, slots=True)
class StoreReleaseOutcome:
    resource_kind: Literal["store_object_handle", "namespace_handle"]
    status: Literal["released", "failed"]
    object_effect: StoreFailureEffect | None
    namespace_effect: StoreNamespaceEffect | None
    failure: ComponentFailureItem | None


@dataclass(frozen=True, slots=True)
class RetainedPermissionEvidence:
    schema: str
    payload_kind: Literal["observation"]
    request_self_sha256: str
    access_expectation_self_sha256: str
    destination_self_sha256: str
    store_expectation_self_sha256: str
    relative_path: str
    file_bytes: int
    file_sha256: str
    commitment_domain: str
    authorization_file_sha256: str
    private_observation_commitment: str
    namespace_object_identity: str
    records_parent_object_identity: str
    store_object_identity: str
    creation_security_descriptor_sha256: str
    confidential_store_profile_sha256: str
    held_object_readback_sha256: str
    publication_state: str
    receipt_self_sha256: str


@dataclass(frozen=True, slots=True)
class RetainedPermissionObservationFailure:
    schema: str
    payload_kind: Literal["observation_failure"]
    request_self_sha256: str
    access_expectation_self_sha256: str
    destination_self_sha256: str
    store_expectation_self_sha256: str
    relative_path: str
    file_bytes: int
    file_sha256: str
    commitment_domain: str
    authorization_file_sha256: str
    private_observation_commitment: str
    namespace_object_identity: str
    records_parent_object_identity: str
    store_object_identity: str
    creation_security_descriptor_sha256: str
    confidential_store_profile_sha256: str
    held_object_readback_sha256: str
    publication_state: str
    receipt_self_sha256: str


@dataclass(frozen=True, slots=True)
class RetainedPermissionAuthzEvidence:
    schema: str
    payload_kind: Literal["authz_evidence"]
    request_self_sha256: str
    access_expectation_self_sha256: str
    destination_self_sha256: str
    store_expectation_self_sha256: str
    relative_path: str
    file_bytes: int
    file_sha256: str
    commitment_domain: str
    authorization_file_sha256: str
    private_authz_commitment: str
    namespace_object_identity: str
    records_parent_object_identity: str
    store_object_identity: str
    creation_security_descriptor_sha256: str
    confidential_store_profile_sha256: str
    held_object_readback_sha256: str
    publication_state: str
    receipt_self_sha256: str


@dataclass(frozen=True, slots=True)
class PermissionEvidenceRetentionFailureDraft:
    failed_primary_payload_kind: Literal["observation", "observation_failure"]
    request_self_sha256: str
    access_expectation_self_sha256: str
    destination_self_sha256: str
    store_expectation_self_sha256: str
    primary_failure: ComponentFailureItem
    ordered_follow_on_failures: tuple[ComponentFailureItem, ...]
    main_authority_accounting: AuthorityAccounting
    payload_effects: tuple[StoreNamespaceEffect | StoreFailureEffect, ...]
    payload_object_state: StoreObjectTerminalState
    acceptance_emitted: Literal[False]


@dataclass(frozen=True, slots=True)
class PermissionEvidenceRetentionFailure:
    schema: str
    payload_kind: Literal["retention_failure"]
    failed_primary_payload_kind: Literal["observation", "observation_failure"]
    request_self_sha256: str
    access_expectation_self_sha256: str
    destination_self_sha256: str
    store_expectation_self_sha256: str
    primary_failure: ComponentFailureItem
    ordered_follow_on_failures: tuple[ComponentFailureItem, ...]
    main_authority_accounting: AuthorityAccounting
    payload_effects: tuple[StoreNamespaceEffect | StoreFailureEffect, ...]
    payload_object_state: StoreObjectTerminalState
    failure_record_attempted: bool
    failure_record_primary_failure: ComponentFailureItem | None
    failure_record_ordered_follow_on_failures: tuple[ComponentFailureItem, ...]
    failure_record_effects: tuple[StoreNamespaceEffect | StoreFailureEffect, ...] | None
    failure_record_object_state: StoreObjectTerminalState | None
    failure_record_authority_accounting: AuthorityAccounting | None
    retained_failure_record_relative_path: str | None
    retained_failure_record_file_bytes: int | None
    retained_failure_record_file_sha256: str | None
    retained_failure_record_commitment_domain: str | None
    retained_failure_record_authorization_file_sha256: str | None
    retained_failure_record_commitment: str | None
    acceptance_emitted: Literal[False]
    result_self_sha256: str


@dataclass(frozen=True, slots=True)
class PermissionAuthzEvidenceRetentionFailureDraft:
    failed_primary_payload_kind: Literal["authz_evidence"]
    request_self_sha256: str
    access_expectation_self_sha256: str
    destination_self_sha256: str
    store_expectation_self_sha256: str
    primary_failure: ComponentFailureItem
    ordered_follow_on_failures: tuple[ComponentFailureItem, ...]
    main_authority_accounting: AuthorityAccounting
    payload_effects: tuple[StoreNamespaceEffect | StoreFailureEffect, ...]
    payload_object_state: StoreObjectTerminalState
    acceptance_emitted: Literal[False]


@dataclass(frozen=True, slots=True)
class PermissionAuthzEvidenceRetentionFailure:
    schema: str
    payload_kind: Literal["authz_retention_failure"]
    failed_primary_payload_kind: Literal["authz_evidence"]
    request_self_sha256: str
    access_expectation_self_sha256: str
    destination_self_sha256: str
    store_expectation_self_sha256: str
    primary_failure: ComponentFailureItem
    ordered_follow_on_failures: tuple[ComponentFailureItem, ...]
    main_authority_accounting: AuthorityAccounting
    payload_effects: tuple[StoreNamespaceEffect | StoreFailureEffect, ...]
    payload_object_state: StoreObjectTerminalState
    failure_record_attempted: bool
    failure_record_primary_failure: ComponentFailureItem | None
    failure_record_ordered_follow_on_failures: tuple[ComponentFailureItem, ...]
    failure_record_effects: tuple[StoreNamespaceEffect | StoreFailureEffect, ...] | None
    failure_record_object_state: StoreObjectTerminalState | None
    failure_record_authority_accounting: AuthorityAccounting | None
    retained_failure_record_relative_path: str | None
    retained_failure_record_file_bytes: int | None
    retained_failure_record_file_sha256: str | None
    retained_failure_record_commitment_domain: str | None
    retained_failure_record_authorization_file_sha256: str | None
    retained_failure_record_commitment: str | None
    acceptance_emitted: Literal[False]
    result_self_sha256: str


class ConfidentialEvidenceStore(Protocol):
    def open_namespace(self, namespace_relative_path: str) -> NamespaceOpenOutcome: ...


class ConfidentialNamespaceLease(Protocol):
    @property
    def snapshot(self) -> ConfidentialNamespaceSnapshot: ...

    def create_exclusive(
        self,
        relative_path: str,
        *,
        confidential_store_profile_sha256: str,
        required_security_descriptor_sha256: str,
    ) -> StoreCreateOutcome: ...

    def release(self) -> StoreReleaseOutcome: ...


class ConfidentialStoreObject(Protocol):
    def write_once(self, data: bytes) -> StoreIoOutcome: ...

    def flush_data_and_metadata(self) -> StoreIoOutcome: ...

    def rewind(self) -> StoreIoOutcome: ...

    def read(self, max_bytes: int) -> StoreBytesReadOutcome: ...

    def snapshot(self) -> StoreSnapshotOutcome: ...

    def publish_same_object_exclusive(
        self, final_relative_path: str
    ) -> StorePromoteOutcome: ...

    def release(self) -> StoreReleaseOutcome: ...


PayloadKind: TypeAlias = Literal[
    "observation",
    "observation_failure",
    "retention_failure",
    "authz_evidence",
    "authz_retention_failure",
]
RetentionDestination: TypeAlias = (
    PermissionEvidenceDestination | PermissionAuthzEvidenceDestination
)
RetentionFailureDraft: TypeAlias = (
    PermissionEvidenceRetentionFailureDraft
    | PermissionAuthzEvidenceRetentionFailureDraft
)
RetentionSlotKind: TypeAlias = Literal[
    "primary", "retention_failure", "authz_primary", "authz_retention_failure"
]


def _primary_slot(slot_kind: str) -> bool:
    return slot_kind in {"primary", "authz_primary"}


def _sidecar_slot(slot_kind: str) -> bool:
    return slot_kind in {"retention_failure", "authz_retention_failure"}


def _slot_resource_role(slot_kind: str) -> str:
    return {
        "primary": "primary_payload",
        "retention_failure": "retention_failure_payload",
        "authz_primary": "authz_primary_payload",
        "authz_retention_failure": "authz_retention_failure_payload",
    }.get(slot_kind, "invalid_retention_payload")


_OBSERVATION_KEYS = {
    "schema", "request", "access_expectation", "outcome", "ancestry_before",
    "subject_before", "stream_inventory", "content_evidence",
    "subject_security_before", "ancestry_security_before",
    "effective_context_before", "access_check_token_before", "access_checks",
    "subject_security_after", "ancestry_security_after", "access_check_token_after",
    "effective_context_after", "subject_after", "ancestry_after",
    "resource_events", "capture_status", "record_self_sha256",
}
_OBSERVATION_FAILURE_KEYS = {
    "schema", "request", "access_expectation", "outcome", "primary_failure",
    "partial_capture", "ordered_follow_on_failures", "authority_accounting",
    "acceptance_emitted", "record_self_sha256",
}
_FAILURE_ITEM_KEYS = {
    "failure_kind", "stage", "component_operation", "resource_kind",
    "resource_role", "resource_ordinal", "operation_ordinal", "backend_error_hex",
    "reason_code",
}
_AUTHORITY_ACCOUNTING_KEYS = {
    "acquire_status", "lease_candidate_returned", "lease_acquired",
    "lease_binding_status", "preterminal_authority_reason_code", "finish_attempted",
    "requested_outcome", "finish_status", "effect_certainty", "recorded_outcome",
    "finish_reason_code",
}
_OBJECT_SNAPSHOT_KEYS = {
    "object_role", "ancestor_index", "is_repository_root", "is_immediate_parent",
    "requested_canonical_volume_guid_path", "facts",
}
_OBJECT_FACT_KEYS = {
    "canonical_volume_guid_path", "canonical_dos_path", "canonical_name_exact",
    "unicode_nfc_exact", "identity", "file_attributes_hex", "reparse_tag_hex",
    "end_of_file_bytes", "allocation_size_bytes", "number_of_links",
    "delete_pending", "directory",
}
_IDENTITY_KEYS = {
    "volume_serial_number_hex", "file_id_128_hex",
    "legacy_volume_serial_number_hex", "legacy_file_index_hex",
}
_STREAM_KEYS = {
    "stream_index", "stream_name", "stream_name_utf16le_hex", "stream_size_bytes",
    "stream_allocation_size_bytes",
}
_SID_KEYS = {"sid_string", "sid_bytes_hex"}
_ACE_KEYS = {
    "ace_index", "ace_kind", "ace_type_hex", "ace_flags_hex", "access_mask_hex",
    "trustee_sid", "raw_ace_hex",
}
_SECURITY_DESCRIPTOR_KEYS = {
    "self_relative_descriptor_hex", "self_relative_descriptor_sha256", "revision",
    "control_hex", "owner_sid", "group_sid", "dacl_present", "dacl_defaulted",
    "dacl_revision", "dacl_protected", "dacl_auto_inherited", "ordered_dacl_aces",
}
_BOUND_SECURITY_KEYS = {
    "object_role", "ancestor_index", "object_identity",
    "canonical_volume_guid_path", "descriptor",
}
_TOKEN_SNAPSHOT_KEYS = {
    "token_role", "selection_source", "duplicated_from_token_id_luid_hex", "facts",
}
_TOKEN_FACT_KEYS = {
    "token_type", "impersonation_level", "token_id_luid_hex",
    "authentication_id_luid_hex", "modified_id_luid_hex", "session_id", "user",
    "owner", "primary_group", "ordered_groups", "ordered_restricted_sids",
    "ordered_privileges", "integrity_level_sid", "elevation_type", "is_elevated",
    "has_restrictions", "is_app_container", "ordered_capabilities",
    "virtualization_allowed", "virtualization_enabled", "ui_access",
}
_GROUP_KEYS = {"sid", "attributes_hex"}
_PRIVILEGE_KEYS = {"luid_hex", "name", "attributes_hex"}
_ACCESS_OUTCOME_KEYS = {
    "object_binding", "access_token_role", "access_token_id_luid_hex",
    "access_token_modified_id_luid_hex", "desired_access_hex", "generic_mapping",
    "raw_result",
}
_ACCESS_BINDING_KEYS = {
    "object_role", "object_kind", "canonical_volume_guid_path", "identity",
    "descriptor_sha256",
}
_GENERIC_MAPPING_KEYS = {
    "generic_read_hex", "generic_write_hex", "generic_execute_hex", "generic_all_hex",
}
_RAW_ACCESS_KEYS = {
    "api_call_status", "access_status", "granted_access_mask_hex",
    "access_denial_error_hex", "api_failure_error_hex", "ordered_privileges_used",
}
_RESOURCE_EVENT_KEYS = {
    "resource_kind", "resource_role", "resource_ordinal", "acquisition_sequence",
    "acquired", "release_attempted", "released", "release_failure_index",
    "object_identity",
}
_PARTIAL_CAPTURE_KEYS = {
    "ancestry_before", "subject_before", "stream_inventory", "content_progress",
    "subject_security_before", "ancestry_security_before", "effective_context_before",
    "access_check_token_before", "access_checks", "subject_security_after",
    "ancestry_security_after", "access_check_token_after", "effective_context_after",
    "subject_after", "ancestry_after", "opened_resource_identities", "raw_prefixes",
    "resource_events",
}
_PARTIAL_SCALAR_KEYS = {"state", "value", "last_completed_operation", "raw_prefix_indices"}
_PARTIAL_ARRAY_KEYS = {
    "state", "value", "last_completed_operation", "raw_prefix_indices",
    "expected_length",
}
_RAW_PREFIX_KEYS = {
    "capture_sequence", "source_operation", "resource_kind", "resource_role",
    "resource_ordinal", "encoding", "declared_size_bytes", "captured_size_bytes",
    "complete", "data_hex", "captured_bytes_sha256",
}
_OPENED_RESOURCE_KEYS = {
    "resource_kind", "resource_role", "resource_ordinal", "acquisition_sequence",
    "object_identity",
}
_RETENTION_RECORD_KEYS = {
    "schema", "payload_kind", "failed_primary_payload_kind", "request_self_sha256",
    "access_expectation_self_sha256", "destination_self_sha256",
    "store_expectation_self_sha256", "primary_failure",
    "ordered_follow_on_failures", "main_authority_accounting", "payload_effects",
    "payload_object_state", "acceptance_emitted",
    "failure_sidecar_recursion_permitted", "record_self_sha256",
}
_AUTHZ_RETENTION_RECORD_KEYS = _RETENTION_RECORD_KEYS | {"intended_authz_record"}
_INTENDED_AUTHZ_RECORD_KEYS = {"record_hex", "file_bytes", "file_sha256", "commitment"}
_FAILURE_KINDS = {
    "authority_refusal", "authority_failure", "lease_binding_failure",
    "observation_refusal", "native_api_failure", "native_protocol_failure",
    "validation_failure", "store_collision", "store_operation_failure",
    "cleanup_failure", "terminal_accounting_failure",
}
_LEASE_REASONS = {
    "lease_malformed", "lease_request_mismatch", "lease_operation_mismatch",
    "lease_access_expectation_mismatch", "lease_destination_mismatch",
    "lease_store_expectation_mismatch", "lease_scope_mismatch",
}
_PRETERMINAL_REASONS = {
    "none", "authority_acquire_refused", "authority_acquire_failed", *_LEASE_REASONS,
    "authority_inactive", "authority_check_failed",
}
_PUBLICATION_STATES = {
    "not_created", "creating_empty", "creating_partial",
    "creating_complete_unverified", "creating_verified", "published_unverified",
    "published_verified_unaccepted", "collision_unowned", "effect_unknown",
}

_STAGES = {
    "request_validation", "access_expectation_validation", "store_expectation_validation",
    "authority_acquisition", "lease_binding", "ancestor_before", "subject_before",
    "stream_inventory", "content_read", "security_before", "effective_context_before",
    "access_token", "access_check", "security_after", "access_token_after",
    "effective_context_after", "object_after", "capture_assembly", "capture_validation",
    "private_record_validation", "destination_validation", "namespace_open",
    "namespace_validation", "exclusive_create", "creation_validation", "payload_write",
    "payload_flush", "payload_rewind", "payload_readback", "readback_validation",
    "prepublication_snapshot", "publication", "postpublication_snapshot",
    "postpublication_readback", "receipt_validation", "retention_failure_build",
    "retention_failure_retention", "resource_cleanup", "final_authority_check",
    "terminal_accounting",
}
_OPERATIONS = {
    "validate_request", "validate_access_expectation", "validate_store_expectation",
    "acquire_authority_lease", "validate_authority_lease", "require_active",
    "open_existing_directory", "open_existing_subject", "query_object", "query_streams",
    "read_unnamed_at", "acquire_security_descriptor", "parse_security_descriptor",
    "open_current_effective_token", "duplicate_for_access_check", "query_token",
    "access_check", "validate_capture", "release_native_resource",
    "finish_authority_lease", "validate_private_record", "canonicalize_private_record",
    "build_destination", "validate_destination", "open_namespace", "validate_namespace",
    "create_exclusive", "validate_creation_evidence", "write_once",
    "flush_data_and_metadata", "rewind", "read", "validate_readback", "snapshot",
    "validate_snapshot", "publish_same_object_exclusive", "release_store_object",
    "release_namespace", "build_retention_failure", "validate_retention_failure",
}
_REASONS = {
    "request_malformed", "request_self_hash_mismatch", "request_context_mismatch",
    "attempt_mismatch", "subject_mapping_mismatch", "access_expectation_mismatch",
    "access_expectation_malformed", "store_expectation_malformed",
    "private_record_malformed", "destination_malformed", "authority_acquire_refused",
    "authority_inactive", "authority_acquire_failed", "authority_check_failed",
    *_LEASE_REASONS,
    "canonical_path_mismatch", "volume_mismatch", "identity_changed", "size_changed",
    "attributes_changed", "descriptor_changed", "token_changed",
    "current_effective_context_changed", "reparse_unsupported",
    "cloud_attribute_unsupported", "named_stream_unsupported", "hard_link_unsupported",
    "ace_unsupported", "token_unsupported", "descriptor_incomplete",
    "raw_prefix_malformed", "short_read", "content_grew", "content_hash_mismatch",
    "access_binding_mismatch", "access_token_binding_mismatch", "generic_mapping_mismatch",
    "access_result_inconsistent", "native_api_failure", "native_protocol_failure",
    "capture_validation_failed", "store_profile_mismatch", "namespace_identity_mismatch",
    "namespace_volume_mismatch", "records_parent_mismatch", "namespace_open_failed",
    "namespace_validation_failed", "creation_evidence_malformed",
    "creation_descriptor_mismatch", "security_not_applied_at_create",
    "prewrite_size_nonzero", "collision", "create_failed", "effect_unknown",
    "write_failed", "short_write", "flush_failed", "rewind_failed", "read_failed",
    "extra_byte", "readback_mismatch", "semantic_validation_failed", "snapshot_failed",
    "snapshot_mismatch", "publish_failed", "postpublish_mismatch",
    "store_protocol_failure", "resource_release_failed", "namespace_release_failed",
    "terminal_finish_failed", "failure_record_build_failed",
    "retention_failure_record_malformed",
}

_RESOURCE_BINDINGS: dict[str, frozenset[str]] = {
    "directory_handle": frozenset({"ancestor"}),
    "file_handle": frozenset({"subject"}),
    "security_descriptor_buffer": frozenset({"subject_security", "ancestor_security"}),
    "effective_token_handle": frozenset({"effective_context_before", "effective_context_after"}),
    "access_check_token_handle": frozenset({"access_check_duplicate"}),
    "namespace_handle": frozenset({"records_namespace"}),
    "store_object_handle": frozenset({
        "primary_payload", "retention_failure_payload",
        "authz_primary_payload", "authz_retention_failure_payload",
    }),
}

_STORE_BINDING_REASONS = frozenset({
    "store_profile_mismatch", "namespace_identity_mismatch", "namespace_volume_mismatch",
    "records_parent_mismatch", "creation_descriptor_mismatch",
})


def _rule(
    stages: set[str], kinds: set[str], reasons: set[str]
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    return frozenset(stages), frozenset(kinds), frozenset(reasons)


_FAILURE_RULES: dict[
    str, tuple[tuple[frozenset[str], frozenset[str], frozenset[str]], ...]
] = {
    "validate_request": (_rule(
        {"request_validation"}, {"validation_failure"},
        {"request_malformed", "request_self_hash_mismatch", "request_context_mismatch",
         "attempt_mismatch", "subject_mapping_mismatch"},
    ),),
    "validate_access_expectation": (_rule(
        {"access_expectation_validation"}, {"validation_failure"},
        {"access_expectation_malformed", "access_expectation_mismatch",
         "generic_mapping_mismatch"},
    ),),
    "validate_store_expectation": (_rule(
        {"store_expectation_validation"}, {"validation_failure"},
        {"store_expectation_malformed", *_STORE_BINDING_REASONS},
    ),),
    "acquire_authority_lease": (
        _rule({"authority_acquisition"}, {"authority_refusal"}, {"authority_acquire_refused"}),
        _rule({"authority_acquisition"}, {"authority_failure"}, {"authority_acquire_failed"}),
    ),
    "validate_authority_lease": (_rule(
        {"lease_binding"}, {"lease_binding_failure"}, set(_LEASE_REASONS),
    ),),
    "require_active": (
        _rule(_STAGES - {"request_validation", "access_expectation_validation",
                         "store_expectation_validation", "authority_acquisition",
                         "lease_binding", "resource_cleanup", "terminal_accounting"},
              {"authority_refusal"}, {"authority_inactive"}),
        _rule(_STAGES - {"request_validation", "access_expectation_validation",
                         "store_expectation_validation", "authority_acquisition",
                         "lease_binding", "resource_cleanup", "terminal_accounting"},
              {"authority_failure"}, {"authority_check_failed"}),
    ),
    "open_existing_directory": (
        _rule({"ancestor_before"}, {"native_api_failure"}, {"native_api_failure"}),
        _rule({"ancestor_before"}, {"observation_refusal"},
              {"canonical_path_mismatch", "volume_mismatch", "reparse_unsupported",
               "cloud_attribute_unsupported"}),
        _rule({"ancestor_before"}, {"native_protocol_failure"}, {"native_protocol_failure"}),
    ),
    "open_existing_subject": (
        _rule({"subject_before"}, {"native_api_failure"}, {"native_api_failure"}),
        _rule({"subject_before"}, {"observation_refusal"},
              {"canonical_path_mismatch", "volume_mismatch", "reparse_unsupported",
               "cloud_attribute_unsupported"}),
        _rule({"subject_before"}, {"native_protocol_failure"}, {"native_protocol_failure"}),
    ),
    "query_object": (
        _rule({"ancestor_before", "subject_before", "object_after"},
              {"native_api_failure"}, {"native_api_failure"}),
        _rule({"ancestor_before", "subject_before", "object_after"},
              {"observation_refusal", "validation_failure"},
              {"canonical_path_mismatch", "volume_mismatch", "identity_changed",
               "size_changed", "attributes_changed", "current_effective_context_changed",
               "reparse_unsupported", "cloud_attribute_unsupported",
               "hard_link_unsupported"}),
        _rule({"ancestor_before", "subject_before", "object_after"},
              {"native_protocol_failure"}, {"native_protocol_failure"}),
    ),
    "query_streams": (
        _rule({"stream_inventory"}, {"native_api_failure"}, {"native_api_failure"}),
        _rule({"stream_inventory"}, {"observation_refusal"}, {"named_stream_unsupported"}),
        _rule({"stream_inventory"}, {"native_protocol_failure"}, {"native_protocol_failure"}),
    ),
    "read_unnamed_at": (
        _rule({"content_read"}, {"native_api_failure"}, {"native_api_failure"}),
        _rule({"content_read"}, {"validation_failure"},
              {"short_read", "content_grew", "content_hash_mismatch"}),
        _rule({"content_read"}, {"native_protocol_failure"}, {"native_protocol_failure"}),
    ),
    "acquire_security_descriptor": (
        _rule({"security_before", "security_after"}, {"native_api_failure"}, {"native_api_failure"}),
        _rule({"security_before", "security_after"}, {"observation_refusal"}, {"descriptor_incomplete"}),
        _rule({"security_before", "security_after"}, {"native_protocol_failure"}, {"native_protocol_failure"}),
    ),
    "parse_security_descriptor": (
        _rule({"security_before", "security_after"}, {"observation_refusal", "validation_failure"},
              {"descriptor_incomplete", "ace_unsupported", "raw_prefix_malformed"}),
        _rule({"security_before", "security_after"}, {"native_protocol_failure"}, {"native_protocol_failure"}),
    ),
    "open_current_effective_token": (
        _rule({"effective_context_before", "effective_context_after"},
              {"native_api_failure"}, {"native_api_failure"}),
        _rule({"effective_context_before", "effective_context_after"},
              {"observation_refusal"}, {"token_unsupported"}),
        _rule({"effective_context_before", "effective_context_after"},
              {"validation_failure"}, {"token_changed", "current_effective_context_changed"}),
        _rule({"effective_context_before", "effective_context_after"},
              {"native_protocol_failure"}, {"native_protocol_failure"}),
    ),
    "duplicate_for_access_check": (
        _rule({"access_token"}, {"native_api_failure"}, {"native_api_failure"}),
        _rule({"access_token"}, {"observation_refusal"}, {"token_unsupported"}),
        _rule({"access_token"}, {"validation_failure"}, {"token_changed"}),
        _rule({"access_token"}, {"native_protocol_failure"}, {"native_protocol_failure"}),
    ),
    "query_token": (
        _rule({"effective_context_before", "access_token", "access_token_after",
               "effective_context_after"}, {"native_api_failure"}, {"native_api_failure"}),
        _rule({"effective_context_before", "access_token", "access_token_after",
               "effective_context_after"}, {"observation_refusal"}, {"token_unsupported"}),
        _rule({"effective_context_before", "access_token", "access_token_after",
               "effective_context_after"}, {"validation_failure"},
              {"token_changed", "current_effective_context_changed"}),
        _rule({"effective_context_before", "access_token", "access_token_after",
               "effective_context_after"}, {"native_protocol_failure"},
              {"native_protocol_failure"}),
    ),
    "access_check": (
        _rule({"access_check"}, {"native_api_failure"}, {"native_api_failure"}),
        _rule({"access_check"}, {"validation_failure"},
              {"access_binding_mismatch",
               "access_token_binding_mismatch", "generic_mapping_mismatch",
               "access_result_inconsistent"}),
        _rule({"access_check"}, {"native_protocol_failure"}, {"native_protocol_failure"}),
    ),
    "validate_capture": (_rule(
        {"capture_assembly", "capture_validation"}, {"validation_failure"},
        {"capture_validation_failed", "canonical_path_mismatch", "volume_mismatch",
         "identity_changed", "size_changed", "attributes_changed", "descriptor_changed",
         "token_changed", "current_effective_context_changed", "access_binding_mismatch",
         "access_token_binding_mismatch", "generic_mapping_mismatch",
         "access_result_inconsistent", "content_hash_mismatch"},
    ),),
    "release_native_resource": (_rule(
        {"resource_cleanup"}, {"cleanup_failure"}, {"resource_release_failed"},
    ),),
    "finish_authority_lease": (_rule(
        {"terminal_accounting"}, {"terminal_accounting_failure"},
        {"terminal_finish_failed"},
    ),),
    "validate_private_record": (_rule(
        {"private_record_validation"}, {"validation_failure"},
        {"private_record_malformed", "semantic_validation_failed", "request_malformed",
         "request_self_hash_mismatch", "request_context_mismatch", "attempt_mismatch",
         "subject_mapping_mismatch", "access_expectation_mismatch",
         "access_expectation_malformed", "generic_mapping_mismatch"},
    ),),
    "canonicalize_private_record": (_rule(
        {"private_record_validation"}, {"validation_failure"},
        {"private_record_malformed", "semantic_validation_failed", "request_self_hash_mismatch",
         "access_expectation_mismatch"},
    ),),
    "build_destination": (_rule(
        {"destination_validation"}, {"validation_failure"},
        {"destination_malformed", *_STORE_BINDING_REASONS},
    ),),
    "validate_destination": (_rule(
        {"destination_validation"}, {"validation_failure"},
        {"destination_malformed", *_STORE_BINDING_REASONS},
    ),),
    "open_namespace": (_rule(
        {"namespace_open"}, {"store_operation_failure"},
        {"namespace_open_failed", "effect_unknown", "store_protocol_failure"},
    ),),
    "validate_namespace": (_rule(
        {"namespace_validation"}, {"validation_failure"},
        {"namespace_validation_failed", "store_profile_mismatch", "namespace_identity_mismatch",
         "namespace_volume_mismatch", "records_parent_mismatch"},
    ),),
    "create_exclusive": (
        _rule({"exclusive_create"}, {"store_collision"}, {"collision"}),
        _rule({"exclusive_create"}, {"store_operation_failure"},
              {"create_failed", "effect_unknown", "store_protocol_failure"}),
    ),
    "validate_creation_evidence": (_rule(
        {"creation_validation"}, {"validation_failure"},
        {"creation_evidence_malformed", "creation_descriptor_mismatch",
         "security_not_applied_at_create", "prewrite_size_nonzero", *_STORE_BINDING_REASONS},
    ),),
    "write_once": (_rule(
        {"payload_write"}, {"store_operation_failure"},
        {"write_failed", "short_write", "effect_unknown", "store_protocol_failure"},
    ),),
    "flush_data_and_metadata": (_rule(
        {"payload_flush"}, {"store_operation_failure"},
        {"flush_failed", "effect_unknown", "store_protocol_failure"},
    ),),
    "rewind": (_rule(
        {"payload_rewind", "postpublication_readback"}, {"store_operation_failure"},
        {"rewind_failed", "effect_unknown", "store_protocol_failure"},
    ),),
    "read": (_rule(
        {"payload_readback", "postpublication_readback"}, {"store_operation_failure"},
        {"read_failed", "short_read", "extra_byte", "effect_unknown", "store_protocol_failure"},
    ),),
    "validate_readback": (_rule(
        {"readback_validation", "postpublication_readback"}, {"validation_failure"},
        {"short_read", "extra_byte", "readback_mismatch", "semantic_validation_failed"},
    ),),
    "snapshot": (_rule(
        {"prepublication_snapshot", "postpublication_snapshot"}, {"store_operation_failure"},
        {"snapshot_failed", "effect_unknown", "store_protocol_failure"},
    ),),
    "validate_snapshot": (_rule(
        {"prepublication_snapshot", "postpublication_snapshot"}, {"validation_failure"},
        {"snapshot_mismatch", "postpublish_mismatch"},
    ),),
    "publish_same_object_exclusive": (
        _rule({"publication"}, {"store_collision"}, {"collision"}),
        _rule({"publication"}, {"store_operation_failure"},
              {"publish_failed", "effect_unknown", "store_protocol_failure"}),
    ),
    "release_store_object": (_rule(
        {"resource_cleanup"}, {"cleanup_failure"}, {"resource_release_failed"},
    ),),
    "release_namespace": (_rule(
        {"resource_cleanup"}, {"cleanup_failure"}, {"namespace_release_failed"},
    ),),
    "build_retention_failure": (_rule(
        {"retention_failure_build"}, {"validation_failure"}, {"failure_record_build_failed"},
    ),),
    "validate_retention_failure": (_rule(
        {"retention_failure_build", "retention_failure_retention"}, {"validation_failure"},
        {"retention_failure_record_malformed", "semantic_validation_failed",
         "request_self_hash_mismatch", "access_expectation_mismatch",
         "destination_malformed", "store_expectation_malformed", *_STORE_BINDING_REASONS},
    ),),
}


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    plain = _plain(value)
    if not isinstance(plain, Mapping):
        raise PrivatePermissionEvidenceContractError(f"{name} must be a mapping")
    return plain


def _exact(value: object, keys: set[str], *, name: str) -> Mapping[str, object]:
    record = _mapping(value, name=name)
    try:
        require_exact_keys(record, keys, name=name)
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    return record


def _sha(value: object, *, name: str) -> str:
    try:
        return require_sha256(value, name=name)
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PrivatePermissionEvidenceContractError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _nfc(value: object, *, name: str, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise PrivatePermissionEvidenceContractError(f"{name} must be text")
    if unicodedata.normalize("NFC", value) != value:
        raise PrivatePermissionEvidenceContractError(f"{name} must be NFC")
    return value


def _relative_path(value: object, *, name: str) -> str:
    text = _nfc(value, name=name)
    if (
        "\\" in text
        or text.startswith("/")
        or text.endswith("/")
        or "//" in text
        or any(part in {"", ".", ".."} for part in text.split("/"))
    ):
        raise PrivatePermissionEvidenceContractError(f"{name} is not canonical relative POSIX")
    return text


def _get(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _request_hash(request: PermissionObservationRequest) -> str:
    return _sha(_get(request, "request_self_sha256"), name="request_self_sha256")


def _access_hash(access: PermissionAccessExpectation) -> str:
    return _sha(
        _get(access, "expectation_self_sha256"),
        name="access_expectation.expectation_self_sha256",
    )


def _authorization_hash(request: PermissionObservationRequest) -> str:
    return _sha(
        _get(_get(request, "context"), "state4_authorization_file_sha256"),
        name="request.context.state4_authorization_file_sha256",
    )


def _validate_expected_inputs(
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
) -> None:
    try:
        validate_permission_observation_request(
            expected_request, expected_request=expected_request
        )
        validate_permission_access_expectation(
            expected_access, expected_request=expected_request
        )
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc


def _validate_hex_shape(value: object, *, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise PrivatePermissionEvidenceContractError(f"{name} must be hexadecimal text")
    if value.startswith("0x"):
        if not _DWORD_RE.fullmatch(value):
            raise PrivatePermissionEvidenceContractError(f"{name} is not canonical hexadecimal")
    elif not _BINARY_HEX_RE.fullmatch(value):
        raise PrivatePermissionEvidenceContractError(f"{name} is not lowercase binary hex")


def _dword_hex(value: object, *, name: str) -> int:
    _validate_hex_shape(value, name=name)
    if not isinstance(value, str) or not value.startswith("0x"):
        raise PrivatePermissionEvidenceContractError(f"{name} must be a canonical DWORD")
    number = int(value, 16)
    if number > 0xFFFFFFFF:
        raise PrivatePermissionEvidenceContractError(f"{name} exceeds DWORD width")
    return number


def _uint64_hex(value: object, *, name: str) -> int:
    _validate_hex_shape(value, name=name)
    if not isinstance(value, str) or not value.startswith("0x"):
        raise PrivatePermissionEvidenceContractError(
            f"{name} must be canonical unsigned-64 hexadecimal"
        )
    number = int(value, 16)
    if number > 0xFFFFFFFFFFFFFFFF:
        raise PrivatePermissionEvidenceContractError(f"{name} exceeds unsigned-64 width")
    return number


def _walk_scalar_syntax(value: object, *, name: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PrivatePermissionEvidenceContractError(f"{name} key must be text")
            child = f"{name}.{key}"
            if key.endswith("_sha256") and item is not None:
                _sha(item, name=child)
            elif key.endswith("_hex"):
                _validate_hex_shape(item, name=child)
            elif isinstance(item, str) and unicodedata.normalize("NFC", item) != item:
                raise PrivatePermissionEvidenceContractError(f"{child} must be NFC")
            _walk_scalar_syntax(item, name=child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_scalar_syntax(item, name=f"{name}[{index}]")


def _self_hash(record: Mapping[str, object], *, field: str) -> None:
    try:
        validate_self_hash(record, field)
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc


def _validate_identity(value: object, *, name: str) -> Mapping[str, object]:
    identity = _exact(value, _IDENTITY_KEYS, name=name)
    _uint64_hex(identity["volume_serial_number_hex"], name=f"{name}.volume")
    file_id = identity["file_id_128_hex"]
    legacy_volume = identity["legacy_volume_serial_number_hex"]
    legacy_index = identity["legacy_file_index_hex"]
    if not isinstance(file_id, str) or not re.fullmatch(r"[0-9a-f]{32}", file_id):
        raise PrivatePermissionEvidenceContractError(f"{name}.file_id_128_hex mismatch")
    if not isinstance(legacy_volume, str) or not re.fullmatch(r"[0-9a-f]{8}", legacy_volume):
        raise PrivatePermissionEvidenceContractError(f"{name}.legacy volume mismatch")
    if not isinstance(legacy_index, str) or not re.fullmatch(r"[0-9a-f]{16}", legacy_index):
        raise PrivatePermissionEvidenceContractError(f"{name}.legacy index mismatch")
    return identity


def _validate_object_snapshot_mapping(
    value: object,
    *,
    name: str,
    role: str | None = None,
    ancestor_index: int | None = None,
) -> Mapping[str, object]:
    snapshot = _exact(value, _OBJECT_SNAPSHOT_KEYS, name=name)
    if snapshot["object_role"] not in {"ancestor", "subject"}:
        raise PrivatePermissionEvidenceContractError(f"{name}.object_role mismatch")
    if role is not None and snapshot["object_role"] != role:
        raise PrivatePermissionEvidenceContractError(f"{name}.bound role mismatch")
    if snapshot["object_role"] == "subject":
        if (
            snapshot["ancestor_index"] is not None
            or snapshot["is_repository_root"] is not False
            or snapshot["is_immediate_parent"] is not False
        ):
            raise PrivatePermissionEvidenceContractError(f"{name}.subject binding mismatch")
    else:
        index = _integer(snapshot["ancestor_index"], name=f"{name}.ancestor_index")
        if ancestor_index is not None and index != ancestor_index:
            raise PrivatePermissionEvidenceContractError(f"{name}.ancestor index mismatch")
        if not isinstance(snapshot["is_repository_root"], bool) or not isinstance(
            snapshot["is_immediate_parent"], bool
        ):
            raise PrivatePermissionEvidenceContractError(f"{name}.ancestor flags mismatch")
    _nfc(snapshot["requested_canonical_volume_guid_path"], name=f"{name}.requested_path")
    facts = _exact(snapshot["facts"], _OBJECT_FACT_KEYS, name=f"{name}.facts")
    for key in (
        "canonical_volume_guid_path", "canonical_dos_path", "canonical_name_exact"
    ):
        _nfc(facts[key], name=f"{name}.facts.{key}")
    if facts["unicode_nfc_exact"] is not True:
        raise PrivatePermissionEvidenceContractError(f"{name}.NFC flag mismatch")
    _validate_identity(facts["identity"], name=f"{name}.facts.identity")
    attributes = _dword_hex(facts["file_attributes_hex"], name=f"{name}.attributes")
    reparse_tag = _dword_hex(facts["reparse_tag_hex"], name=f"{name}.reparse_tag")
    for key in ("end_of_file_bytes", "allocation_size_bytes", "number_of_links"):
        _integer(facts[key], name=f"{name}.facts.{key}")
    if not isinstance(facts["delete_pending"], bool) or not isinstance(facts["directory"], bool):
        raise PrivatePermissionEvidenceContractError(f"{name}.object Boolean mismatch")
    if (
        facts["canonical_volume_guid_path"]
        != snapshot["requested_canonical_volume_guid_path"]
        or facts["allocation_size_bytes"] < facts["end_of_file_bytes"]
        or facts["delete_pending"] is not False
        or reparse_tag != 0
        or attributes & (0x400 | 0x1000 | 0x40000 | 0x400000)
    ):
        raise PrivatePermissionEvidenceContractError(f"{name}.object policy mismatch")
    if snapshot["object_role"] == "subject":
        if facts["directory"] is not False or facts["number_of_links"] != 1:
            raise PrivatePermissionEvidenceContractError(f"{name}.subject type/link mismatch")
    elif facts["directory"] is not True:
        raise PrivatePermissionEvidenceContractError(f"{name}.ancestor is not a directory")
    return snapshot


def _validate_stream(value: object, *, name: str) -> None:
    stream = _exact(value, _STREAM_KEYS, name=name)
    _integer(stream["stream_index"], name=f"{name}.stream_index")
    text = _nfc(stream["stream_name"], name=f"{name}.stream_name")
    raw = stream["stream_name_utf16le_hex"]
    _validate_hex_shape(raw, name=f"{name}.stream_name_utf16le_hex")
    if bytes.fromhex(raw).decode("utf-16le") != text:
        raise PrivatePermissionEvidenceContractError(f"{name}.stream encoding mismatch")
    _integer(stream["stream_size_bytes"], name=f"{name}.stream_size_bytes")
    _integer(stream["stream_allocation_size_bytes"], name=f"{name}.allocation")
    if stream["stream_allocation_size_bytes"] < stream["stream_size_bytes"]:
        raise PrivatePermissionEvidenceContractError(f"{name}.allocation is smaller than size")


def _sid_text(data: bytes, *, name: str) -> str:
    if len(data) < 8:
        raise PrivatePermissionEvidenceContractError(f"{name} SID is truncated")
    count = data[1]
    if data[0] != 1 or count > 15 or len(data) != 8 + 4 * count:
        raise PrivatePermissionEvidenceContractError(f"{name} SID length mismatch")
    authority = int.from_bytes(data[2:8], "big")
    subauthorities = [
        str(int.from_bytes(data[8 + 4 * index : 12 + 4 * index], "little"))
        for index in range(count)
    ]
    return "-".join([f"S-{data[0]}", str(authority), *subauthorities])


def _sid_wire_length(data: bytes, offset: int, *, name: str) -> int:
    if offset < 0 or offset + 8 > len(data):
        raise PrivatePermissionEvidenceContractError(f"{name} SID offset is outside bytes")
    length = 8 + 4 * data[offset + 1]
    if offset + length > len(data):
        raise PrivatePermissionEvidenceContractError(f"{name} SID is truncated")
    return length


def _validate_sid(value: object, *, name: str) -> None:
    sid = _exact(value, _SID_KEYS, name=name)
    text = _nfc(sid["sid_string"], name=f"{name}.sid_string")
    raw_hex = sid["sid_bytes_hex"]
    _validate_hex_shape(raw_hex, name=f"{name}.sid_bytes_hex")
    raw = bytes.fromhex(raw_hex)
    if not raw or _sid_text(raw, name=name) != text:
        raise PrivatePermissionEvidenceContractError(
            f"{name}.SID text does not reproduce its binary value"
        )


def _validate_security(value: object, *, name: str) -> Mapping[str, object]:
    bound = _exact(value, _BOUND_SECURITY_KEYS, name=name)
    if bound["object_role"] not in {"subject", "ancestor"}:
        raise PrivatePermissionEvidenceContractError(f"{name}.role mismatch")
    if bound["object_role"] == "subject" and bound["ancestor_index"] is not None:
        raise PrivatePermissionEvidenceContractError(f"{name}.subject ordinal mismatch")
    if bound["object_role"] == "ancestor":
        _integer(bound["ancestor_index"], name=f"{name}.ancestor_index")
    _validate_identity(bound["object_identity"], name=f"{name}.object_identity")
    _nfc(bound["canonical_volume_guid_path"], name=f"{name}.path")
    descriptor = _exact(
        bound["descriptor"], _SECURITY_DESCRIPTOR_KEYS, name=f"{name}.descriptor"
    )
    raw = descriptor["self_relative_descriptor_hex"]
    _validate_hex_shape(raw, name=f"{name}.descriptor.raw")
    descriptor_bytes = bytes.fromhex(raw)
    if descriptor["self_relative_descriptor_sha256"] != sha256_bytes(descriptor_bytes):
        raise PrivatePermissionEvidenceContractError(f"{name}.descriptor hash mismatch")
    if _integer(descriptor["revision"], name=f"{name}.descriptor.revision") != 1:
        raise PrivatePermissionEvidenceContractError(f"{name}.descriptor revision mismatch")
    _dword_hex(descriptor["control_hex"], name=f"{name}.descriptor.control")
    _validate_sid(descriptor["owner_sid"], name=f"{name}.descriptor.owner")
    _validate_sid(descriptor["group_sid"], name=f"{name}.descriptor.group")
    for key in (
        "dacl_present", "dacl_defaulted", "dacl_protected", "dacl_auto_inherited"
    ):
        if not isinstance(descriptor[key], bool):
            raise PrivatePermissionEvidenceContractError(f"{name}.descriptor.{key} mismatch")
    if _integer(descriptor["dacl_revision"], name=f"{name}.descriptor.dacl_revision") not in {2, 4}:
        raise PrivatePermissionEvidenceContractError(f"{name}.descriptor ACL revision mismatch")
    aces = descriptor["ordered_dacl_aces"]
    if not isinstance(aces, list):
        raise PrivatePermissionEvidenceContractError(f"{name}.descriptor ACEs must be array")
    for index, ace_value in enumerate(aces):
        ace = _exact(ace_value, _ACE_KEYS, name=f"{name}.ace[{index}]")
        if ace["ace_index"] != index:
            raise PrivatePermissionEvidenceContractError(f"{name}.ACE order mismatch")
        if ace["ace_kind"] not in {"allow", "deny"}:
            raise PrivatePermissionEvidenceContractError(f"{name}.ACE kind is unsupported")
        for key in ("ace_type_hex", "ace_flags_hex", "access_mask_hex"):
            _dword_hex(ace[key], name=f"{name}.ace.{key}")
        _validate_sid(ace["trustee_sid"], name=f"{name}.ace.trustee")
        _validate_hex_shape(ace["raw_ace_hex"], name=f"{name}.ace.raw")
        ace_bytes = bytes.fromhex(ace["raw_ace_hex"])
        ace_type = int(ace["ace_type_hex"], 16)
        ace_flags = int(ace["ace_flags_hex"], 16)
        access_mask = int(ace["access_mask_hex"], 16)
        trustee = _mapping(ace["trustee_sid"], name=f"{name}.ace.trustee")
        if (
            len(ace_bytes) < 8
            or ace_type != (0 if ace["ace_kind"] == "allow" else 1)
            or ace_type > 0xff
            or ace_flags > 0xff
            or ace_bytes[0] != ace_type
            or ace_bytes[1] != ace_flags
            or int.from_bytes(ace_bytes[2:4], "little") != len(ace_bytes)
            or int.from_bytes(ace_bytes[4:8], "little") != access_mask
            or ace_bytes[8:].hex() != trustee["sid_bytes_hex"]
        ):
            raise PrivatePermissionEvidenceContractError(
                f"{name}.ACE raw bytes do not reproduce typed fields"
            )
    if (
        len(descriptor_bytes) < 20
        or descriptor_bytes[0] != descriptor["revision"]
        or descriptor_bytes[1] != 0
    ):
        raise PrivatePermissionEvidenceContractError(f"{name}.descriptor header mismatch")
    control = int(descriptor["control_hex"], 16)
    if int.from_bytes(descriptor_bytes[2:4], "little") != control or not control & 0x8000:
        raise PrivatePermissionEvidenceContractError(f"{name}.descriptor control mismatch")
    owner_offset = int.from_bytes(descriptor_bytes[4:8], "little")
    group_offset = int.from_bytes(descriptor_bytes[8:12], "little")
    sacl_offset = int.from_bytes(descriptor_bytes[12:16], "little")
    dacl_offset = int.from_bytes(descriptor_bytes[16:20], "little")
    owner_length = _sid_wire_length(descriptor_bytes, owner_offset, name=f"{name}.owner")
    group_length = _sid_wire_length(descriptor_bytes, group_offset, name=f"{name}.group")
    owner = _mapping(descriptor["owner_sid"], name=f"{name}.owner")
    group = _mapping(descriptor["group_sid"], name=f"{name}.group")
    if (
        descriptor_bytes[owner_offset : owner_offset + owner_length].hex()
        != owner["sid_bytes_hex"]
        or descriptor_bytes[group_offset : group_offset + group_length].hex()
        != group["sid_bytes_hex"]
        or sacl_offset != 0
    ):
        raise PrivatePermissionEvidenceContractError(f"{name}.descriptor SID/SACL mismatch")
    if (
        descriptor["dacl_present"] != bool(control & 0x0004)
        or descriptor["dacl_defaulted"] != bool(control & 0x0008)
        or descriptor["dacl_protected"] != bool(control & 0x1000)
        or descriptor["dacl_auto_inherited"] != bool(control & 0x0400)
        or not descriptor["dacl_present"]
        or dacl_offset == 0
        or dacl_offset + 8 > len(descriptor_bytes)
    ):
        raise PrivatePermissionEvidenceContractError(f"{name}.descriptor DACL mismatch")
    acl_size = int.from_bytes(descriptor_bytes[dacl_offset + 2 : dacl_offset + 4], "little")
    ace_count = int.from_bytes(descriptor_bytes[dacl_offset + 4 : dacl_offset + 6], "little")
    if (
        descriptor_bytes[dacl_offset] != descriptor["dacl_revision"]
        or descriptor_bytes[dacl_offset + 1] != 0
        or descriptor_bytes[dacl_offset + 6 : dacl_offset + 8] != b"\x00\x00"
        or acl_size < 8
        or dacl_offset + acl_size > len(descriptor_bytes)
        or ace_count != len(aces)
    ):
        raise PrivatePermissionEvidenceContractError(f"{name}.descriptor ACL header mismatch")
    intervals = sorted(
        (
            (owner_offset, owner_offset + owner_length),
            (group_offset, group_offset + group_length),
            (dacl_offset, dacl_offset + acl_size),
        )
    )
    if (
        any(start < 20 or start % 4 for start, _ in intervals)
        or any(left[1] > right[0] for left, right in zip(intervals, intervals[1:]))
    ):
        raise PrivatePermissionEvidenceContractError(f"{name}.descriptor component layout mismatch")
    cursor = dacl_offset + 8
    for ace_value in aces:
        ace = _mapping(ace_value, name=f"{name}.ACE")
        ace_bytes = bytes.fromhex(ace["raw_ace_hex"])
        if descriptor_bytes[cursor : cursor + len(ace_bytes)] != ace_bytes:
            raise PrivatePermissionEvidenceContractError(
                f"{name}.descriptor does not reproduce ordered ACE bytes"
            )
        cursor += len(ace_bytes)
    if cursor != dacl_offset + acl_size:
        raise PrivatePermissionEvidenceContractError(f"{name}.descriptor ACE length mismatch")
    return bound


def _validate_token(value: object, *, name: str) -> Mapping[str, object]:
    token = _exact(value, _TOKEN_SNAPSHOT_KEYS, name=name)
    if token["token_role"] not in {"effective_context", "access_check_duplicate"}:
        raise PrivatePermissionEvidenceContractError(f"{name}.role mismatch")
    if token["selection_source"] not in {
        "thread_impersonation", "process_primary", "duplicated_effective_context"
    }:
        raise PrivatePermissionEvidenceContractError(f"{name}.selection source mismatch")
    duplicate_source = token["duplicated_from_token_id_luid_hex"]
    if duplicate_source is not None:
        if not isinstance(duplicate_source, str) or not re.fullmatch(r"[0-9a-f]{16}", duplicate_source):
            raise PrivatePermissionEvidenceContractError(f"{name}.duplicate source mismatch")
    facts = _exact(token["facts"], _TOKEN_FACT_KEYS, name=f"{name}.facts")
    if facts["token_type"] not in {"primary", "impersonation"}:
        raise PrivatePermissionEvidenceContractError(f"{name}.token type mismatch")
    levels = {
        None, "anonymous", "identification", "security_impersonation",
        "security_delegation",
    }
    if facts["impersonation_level"] not in levels:
        raise PrivatePermissionEvidenceContractError(f"{name}.impersonation level mismatch")
    if (
        (facts["token_type"] == "primary" and facts["impersonation_level"] is not None)
        or (facts["token_type"] == "impersonation" and facts["impersonation_level"] is None)
    ):
        raise PrivatePermissionEvidenceContractError(f"{name}.token/level equation mismatch")
    for key in ("token_id_luid_hex", "authentication_id_luid_hex", "modified_id_luid_hex"):
        if not isinstance(facts[key], str) or not re.fullmatch(r"[0-9a-f]{16}", facts[key]):
            raise PrivatePermissionEvidenceContractError(f"{name}.{key} mismatch")
    _integer(facts["session_id"], name=f"{name}.session_id")
    for key in ("user", "owner", "primary_group", "integrity_level_sid"):
        _validate_sid(facts[key], name=f"{name}.{key}")
    for group_key in ("ordered_groups", "ordered_restricted_sids", "ordered_capabilities"):
        groups = facts[group_key]
        if not isinstance(groups, list):
            raise PrivatePermissionEvidenceContractError(f"{name}.{group_key} must be array")
        for index, group_value in enumerate(groups):
            group = _exact(group_value, _GROUP_KEYS, name=f"{name}.{group_key}[{index}]")
            _validate_sid(group["sid"], name=f"{name}.{group_key}[{index}].sid")
            _dword_hex(group["attributes_hex"], name=f"{name}.group.attributes")
    privileges = facts["ordered_privileges"]
    if not isinstance(privileges, list):
        raise PrivatePermissionEvidenceContractError(f"{name}.privileges must be array")
    for index, privilege_value in enumerate(privileges):
        privilege = _exact(
            privilege_value, _PRIVILEGE_KEYS, name=f"{name}.privilege[{index}]"
        )
        if not isinstance(privilege["luid_hex"], str) or not re.fullmatch(
            r"[0-9a-f]{16}", privilege["luid_hex"]
        ):
            raise PrivatePermissionEvidenceContractError(f"{name}.privilege LUID mismatch")
        _nfc(privilege["name"], name=f"{name}.privilege.name")
        _dword_hex(privilege["attributes_hex"], name=f"{name}.privilege.attributes")
    if facts["elevation_type"] not in {"default", "full", "limited"}:
        raise PrivatePermissionEvidenceContractError(f"{name}.elevation mismatch")
    for key in (
        "is_elevated", "has_restrictions", "is_app_container",
        "virtualization_allowed", "virtualization_enabled", "ui_access"
    ):
        if not isinstance(facts[key], bool):
            raise PrivatePermissionEvidenceContractError(f"{name}.{key} mismatch")
    if (
        (facts["elevation_type"] == "full" and facts["is_elevated"] is not True)
        or (facts["elevation_type"] == "limited" and facts["is_elevated"] is not False)
        or facts["is_app_container"] is not False
        or facts["ordered_capabilities"] != []
    ):
        raise PrivatePermissionEvidenceContractError(f"{name}.unsupported token policy")
    if token["token_role"] == "effective_context":
        if token["selection_source"] not in {"thread_impersonation", "process_primary"}:
            raise PrivatePermissionEvidenceContractError(f"{name}.effective source mismatch")
        if token["duplicated_from_token_id_luid_hex"] is not None:
            raise PrivatePermissionEvidenceContractError(f"{name}.effective duplicate binding mismatch")
        if (
            (token["selection_source"] == "process_primary" and facts["token_type"] != "primary")
            or (
                token["selection_source"] == "thread_impersonation"
                and (
                    facts["token_type"] != "impersonation"
                    or facts["impersonation_level"]
                    not in {"security_impersonation", "security_delegation"}
                )
            )
        ):
            raise PrivatePermissionEvidenceContractError(f"{name}.effective token equation mismatch")
    else:
        if (
            token["selection_source"] != "duplicated_effective_context"
            or token["duplicated_from_token_id_luid_hex"] is None
            or facts["token_type"] != "impersonation"
            or facts["impersonation_level"]
            not in {"security_impersonation", "security_delegation"}
        ):
            raise PrivatePermissionEvidenceContractError(f"{name}.duplicate token equation mismatch")
    return token


def _validate_access_outcome(value: object, *, name: str) -> Mapping[str, object]:
    outcome = _exact(value, _ACCESS_OUTCOME_KEYS, name=name)
    binding = _exact(outcome["object_binding"], _ACCESS_BINDING_KEYS, name=f"{name}.binding")
    if binding["object_role"] not in {"subject", "immediate_parent"}:
        raise PrivatePermissionEvidenceContractError(f"{name}.binding role mismatch")
    if binding["object_kind"] not in {"file", "directory"}:
        raise PrivatePermissionEvidenceContractError(f"{name}.binding kind mismatch")
    _nfc(binding["canonical_volume_guid_path"], name=f"{name}.binding.path")
    _validate_identity(binding["identity"], name=f"{name}.binding.identity")
    _sha(binding["descriptor_sha256"], name=f"{name}.binding.descriptor_sha256")
    if outcome["access_token_role"] != "access_check_duplicate":
        raise PrivatePermissionEvidenceContractError(f"{name}.token role mismatch")
    for key in ("access_token_id_luid_hex", "access_token_modified_id_luid_hex"):
        if not isinstance(outcome[key], str) or not re.fullmatch(r"[0-9a-f]{16}", outcome[key]):
            raise PrivatePermissionEvidenceContractError(f"{name}.{key} mismatch")
    _dword_hex(outcome["desired_access_hex"], name=f"{name}.desired_access")
    mapping = _exact(outcome["generic_mapping"], _GENERIC_MAPPING_KEYS, name=f"{name}.mapping")
    for key in mapping:
        mapped = _dword_hex(mapping[key], name=f"{name}.mapping.{key}")
        if mapped & 0xF0000000:
            raise PrivatePermissionEvidenceContractError(f"{name}.mapping retains generic bits")
    raw = _exact(outcome["raw_result"], _RAW_ACCESS_KEYS, name=f"{name}.raw")
    status = (raw["api_call_status"], raw["access_status"])
    if status == ("succeeded", "denied"):
        if (
            raw["granted_access_mask_hex"] != "0x0"
            or raw["access_denial_error_hex"] != "0x5"
            or raw["api_failure_error_hex"] is not None
            or raw["ordered_privileges_used"] != []
        ):
            raise PrivatePermissionEvidenceContractError(f"{name}.denial equation mismatch")
    elif status == ("succeeded", "granted"):
        _dword_hex(raw["granted_access_mask_hex"], name=f"{name}.granted mask")
        privileges = raw["ordered_privileges_used"]
        if (
            raw["access_denial_error_hex"] is not None
            or raw["api_failure_error_hex"] is not None
            or not isinstance(privileges, list)
            or int(raw["granted_access_mask_hex"], 16) & 0xF0000000
        ):
            raise PrivatePermissionEvidenceContractError(f"{name}.grant error mismatch")
        for index, value in enumerate(privileges):
            privilege = _exact(
                value, _PRIVILEGE_KEYS, name=f"{name}.privilege[{index}]"
            )
            if not isinstance(privilege["luid_hex"], str) or not re.fullmatch(
                r"[0-9a-f]{16}", privilege["luid_hex"]
            ):
                raise PrivatePermissionEvidenceContractError(f"{name}.privilege LUID mismatch")
            _nfc(privilege["name"], name=f"{name}.privilege.name")
            _dword_hex(
                privilege["attributes_hex"], name=f"{name}.privilege.attributes"
            )
    elif status == ("failed", "not_evaluated"):
        if (
            raw["granted_access_mask_hex"] is not None
            or raw["access_denial_error_hex"] is not None
            or raw["ordered_privileges_used"] is not None
        ):
            raise PrivatePermissionEvidenceContractError(f"{name}.API failure equation mismatch")
        _dword_hex(raw["api_failure_error_hex"], name=f"{name}.API error")
        if raw["api_failure_error_hex"] == "0x0":
            raise PrivatePermissionEvidenceContractError(f"{name}.API error cannot be zero")
    else:
        raise PrivatePermissionEvidenceContractError(f"{name}.access status mismatch")
    return outcome


def _validate_resource_event(value: object, *, name: str, success: bool) -> Mapping[str, object]:
    event = _exact(value, _RESOURCE_EVENT_KEYS, name=name)
    kind = event["resource_kind"]
    role = event["resource_role"]
    if (
        kind not in _RESOURCE_BINDINGS
        or kind in {"namespace_handle", "store_object_handle"}
        or role not in _RESOURCE_BINDINGS[kind]
    ):
        raise PrivatePermissionEvidenceContractError(f"{name}.resource binding mismatch")
    if event["acquired"] is not True or event["release_attempted"] is not True:
        raise PrivatePermissionEvidenceContractError(f"{name}.resource lifecycle mismatch")
    _integer(event["acquisition_sequence"], name=f"{name}.acquisition_sequence")
    if role in {"ancestor", "ancestor_security"}:
        _integer(event["resource_ordinal"], name=f"{name}.resource_ordinal")
    elif event["resource_ordinal"] is not None:
        raise PrivatePermissionEvidenceContractError(f"{name}.resource ordinal mismatch")
    if kind in {"directory_handle", "file_handle"} and event["object_identity"] is not None:
        _validate_identity(event["object_identity"], name=f"{name}.object_identity")
    elif kind not in {"directory_handle", "file_handle"} and event["object_identity"] is not None:
        raise PrivatePermissionEvidenceContractError(f"{name}.non-object resource has identity")
    if type(event["released"]) is not bool:
        raise PrivatePermissionEvidenceContractError(f"{name}.released must be Boolean")
    if success and (event["released"] is not True or event["release_failure_index"] is not None):
        raise PrivatePermissionEvidenceContractError(f"{name}.success release mismatch")
    return event


def _validate_partial_slot(
    value: object,
    *,
    name: str,
    array: bool,
) -> Mapping[str, object]:
    slot = _exact(
        value, _PARTIAL_ARRAY_KEYS if array else _PARTIAL_SCALAR_KEYS, name=name
    )
    legal = {"not_reached", "prefix", "complete"} if array else {"not_reached", "complete", "raw_only"}
    if slot["state"] not in legal:
        raise PrivatePermissionEvidenceContractError(f"{name}.state mismatch")
    if not isinstance(slot["raw_prefix_indices"], list):
        raise PrivatePermissionEvidenceContractError(f"{name}.raw references mismatch")
    for index in slot["raw_prefix_indices"]:
        _integer(index, name=f"{name}.raw index")
    if array:
        if slot["expected_length"] is not None:
            _integer(slot["expected_length"], name=f"{name}.expected_length")
        if not isinstance(slot["value"], list):
            raise PrivatePermissionEvidenceContractError(f"{name}.value must be array")
    if slot["state"] == "not_reached":
        expected_value: object = None
        if array:
            length = slot["expected_length"]
            expected_value = [] if length is None else [None] * length
        if (
            slot["value"] != expected_value
            or slot["last_completed_operation"] is not None
            or slot["raw_prefix_indices"] != []
        ):
            raise PrivatePermissionEvidenceContractError(f"{name}.not-reached equation mismatch")
    elif array:
        operation = _nfc(
            slot["last_completed_operation"], name=f"{name}.last_completed_operation"
        )
        if not operation:
            raise PrivatePermissionEvidenceContractError(f"{name}.operation mismatch")
        values = slot["value"]
        seen_null = False
        for item in values:
            if item is None:
                seen_null = True
            elif seen_null:
                raise PrivatePermissionEvidenceContractError(f"{name}.typed prefix is not contiguous")
        if slot["expected_length"] is not None and len(values) != slot["expected_length"]:
            raise PrivatePermissionEvidenceContractError(f"{name}.expected length mismatch")
        if slot["state"] == "complete" and (
            slot["expected_length"] is None
            or any(item is None for item in values)
            or slot["raw_prefix_indices"] != []
        ):
            raise PrivatePermissionEvidenceContractError(f"{name}.complete equation mismatch")
    elif slot["state"] == "complete":
        _nfc(slot["last_completed_operation"], name=f"{name}.last_completed_operation")
        if slot["value"] is None or slot["raw_prefix_indices"] != []:
            raise PrivatePermissionEvidenceContractError(f"{name}.complete equation mismatch")
    else:
        _nfc(slot["last_completed_operation"], name=f"{name}.last_completed_operation")
        if slot["value"] is not None or not slot["raw_prefix_indices"]:
            raise PrivatePermissionEvidenceContractError(f"{name}.raw-only equation mismatch")
    return slot


def _validate_partial_capture(
    value: object,
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
) -> Mapping[str, object]:
    partial = _exact(value, _PARTIAL_CAPTURE_KEYS, name="partial_capture")
    array_names = {
        "ancestry_before", "ancestry_security_before",
        "access_checks", "ancestry_security_after", "ancestry_after",
    }
    scalar_names = {
        "subject_before", "stream_inventory", "subject_security_before", "effective_context_before",
        "access_check_token_before", "subject_security_after", "access_check_token_after",
        "effective_context_after", "subject_after",
    }
    slots: dict[str, Mapping[str, object]] = {}
    ancestor_paths = tuple(
        _get(_get(expected_request, "subject"), "required_ancestor_canonical_volume_guid_paths")
    )
    exact_lengths = {
        "ancestry_before": len(ancestor_paths),
        "ancestry_security_before": len(ancestor_paths),
        "access_checks": 2,
        "ancestry_security_after": len(ancestor_paths),
        "ancestry_after": len(ancestor_paths),
    }
    for slot_name in array_names:
        slot = _validate_partial_slot(
            partial[slot_name], name=f"partial_capture.{slot_name}", array=True
        )
        slots[slot_name] = slot
        if slot_name in exact_lengths and slot["expected_length"] != exact_lengths[slot_name]:
            raise PrivatePermissionEvidenceContractError(
                f"partial_capture.{slot_name}.expected_length mismatch"
            )
    for slot_name in scalar_names:
        slots[slot_name] = _validate_partial_slot(
            partial[slot_name], name=f"partial_capture.{slot_name}", array=False
        )
    slot_operations: dict[str, set[str]] = {
        "ancestry_before": {"open_existing_directory", "query_object"},
        "subject_before": {"open_existing_subject", "query_object"},
        "stream_inventory": {"query_streams"},
        "subject_security_before": {"acquire_security_descriptor", "parse_security_descriptor"},
        "ancestry_security_before": {"acquire_security_descriptor", "parse_security_descriptor"},
        "effective_context_before": {"open_current_effective_token", "query_token"},
        "access_check_token_before": {"duplicate_for_access_check", "query_token"},
        "access_checks": {"access_check"},
        "subject_security_after": {"acquire_security_descriptor", "parse_security_descriptor"},
        "ancestry_security_after": {"acquire_security_descriptor", "parse_security_descriptor"},
        "access_check_token_after": {"query_token"},
        "effective_context_after": {"open_current_effective_token", "query_token"},
        "subject_after": {"query_object"},
        "ancestry_after": {"query_object"},
    }
    for slot_name, slot in slots.items():
        operation = slot["last_completed_operation"]
        if operation is not None and operation not in slot_operations[slot_name]:
            raise PrivatePermissionEvidenceContractError(
                f"partial_capture.{slot_name} operation mismatch"
            )

    def validate_typed(slot_name: str, item: object, index: int | None = None) -> None:
        label = f"partial_capture.{slot_name}"
        if index is not None:
            label += f"[{index}]"
        if slot_name in {"ancestry_before", "ancestry_after"}:
            if index is None or index >= len(ancestor_paths):
                raise PrivatePermissionEvidenceContractError(f"{label} has no requested ancestor")
            snapshot = _validate_object_snapshot_mapping(
                item, name=label, role="ancestor", ancestor_index=index
            )
            if (
                snapshot["requested_canonical_volume_guid_path"] != ancestor_paths[index]
                or snapshot["is_repository_root"] is not (index == 0)
                or snapshot["is_immediate_parent"] is not (index == len(ancestor_paths) - 1)
            ):
                raise PrivatePermissionEvidenceContractError(f"{label} positional binding mismatch")
        elif slot_name in {"subject_before", "subject_after"}:
            snapshot = _validate_object_snapshot_mapping(item, name=label, role="subject")
            if snapshot["requested_canonical_volume_guid_path"] != _get(
                _get(expected_request, "subject"), "canonical_volume_guid_path"
            ):
                raise PrivatePermissionEvidenceContractError(f"{label} subject binding mismatch")
        elif slot_name == "stream_inventory":
            if not isinstance(item, (list, tuple)):
                raise PrivatePermissionEvidenceContractError(f"{label} must be a stream array")
            for stream_index, stream_value in enumerate(item):
                _validate_stream(stream_value, name=f"{label}[{stream_index}]")
                stream = _mapping(stream_value, name=f"{label}[{stream_index}]")
                if stream["stream_index"] != stream_index:
                    raise PrivatePermissionEvidenceContractError(f"{label} stream index mismatch")
            if slots[slot_name]["state"] == "complete":
                if len(item) != 1:
                    raise PrivatePermissionEvidenceContractError(
                        f"{label} complete inventory is not unnamed-stream-only"
                    )
                stream = _mapping(item[0], name=f"{label}[0]")
                if (
                    stream["stream_name"] != "::$DATA"
                    or stream["stream_size_bytes"]
                    != _get(_get(expected_request, "subject"), "expected_file_bytes")
                ):
                    raise PrivatePermissionEvidenceContractError(
                        f"{label} complete unnamed stream binding mismatch"
                    )
        elif slot_name in {"subject_security_before", "subject_security_after"}:
            bound = _validate_security(item, name=label)
            if (
                bound["object_role"] != "subject"
                or bound["ancestor_index"] is not None
                or bound["canonical_volume_guid_path"]
                != _get(_get(expected_request, "subject"), "canonical_volume_guid_path")
            ):
                raise PrivatePermissionEvidenceContractError(f"{label} subject security mismatch")
        elif slot_name in {"ancestry_security_before", "ancestry_security_after"}:
            bound = _validate_security(item, name=label)
            if (
                index is None
                or bound["object_role"] != "ancestor"
                or bound["ancestor_index"] != index
                or bound["canonical_volume_guid_path"] != ancestor_paths[index]
            ):
                raise PrivatePermissionEvidenceContractError(f"{label} ancestor security mismatch")
        elif slot_name in {"effective_context_before", "effective_context_after"}:
            token = _validate_token(item, name=label)
            if token["token_role"] != "effective_context":
                raise PrivatePermissionEvidenceContractError(f"{label} token role mismatch")
        elif slot_name in {"access_check_token_before", "access_check_token_after"}:
            token = _validate_token(item, name=label)
            if token["token_role"] != "access_check_duplicate":
                raise PrivatePermissionEvidenceContractError(f"{label} token role mismatch")
        elif slot_name == "access_checks":
            checked = _validate_access_outcome(item, name=label)
            expected_mapping = _plain(
                _get(
                    expected_access,
                    "file_generic_mapping" if index == 0 else "directory_generic_mapping",
                )
            )
            binding = _mapping(checked["object_binding"], name=f"{label}.object_binding")
            expected_path = (
                _get(_get(expected_request, "subject"), "canonical_volume_guid_path")
                if index == 0
                else ancestor_paths[-1]
            )
            if (
                checked["desired_access_hex"] != _get(expected_access, "desired_access_hex")
                or checked["generic_mapping"] != expected_mapping
                or binding["object_role"] != ("subject" if index == 0 else "immediate_parent")
                or binding["object_kind"] != ("file" if index == 0 else "directory")
                or binding["canonical_volume_guid_path"] != expected_path
            ):
                raise PrivatePermissionEvidenceContractError(f"{label} access expectation mismatch")
            object_slot = slots["subject_before"] if index == 0 else slots["ancestry_before"]
            object_value = (
                object_slot["value"]
                if index == 0
                else object_slot["value"][len(ancestor_paths) - 1]
            )
            if object_value is not None:
                object_facts = _mapping(
                    _mapping(object_value, name=f"{label}.bound_object")["facts"],
                    name=f"{label}.bound_object.facts",
                )
                if binding["identity"] != object_facts["identity"]:
                    raise PrivatePermissionEvidenceContractError(
                        f"{label} object identity binding mismatch"
                    )
            security_slot = (
                slots["subject_security_before"]
                if index == 0
                else slots["ancestry_security_before"]
            )
            security_value = (
                security_slot["value"]
                if index == 0
                else security_slot["value"][len(ancestor_paths) - 1]
            )
            if security_value is not None:
                descriptor = _mapping(
                    _mapping(security_value, name=f"{label}.bound_security")["descriptor"],
                    name=f"{label}.bound_security.descriptor",
                )
                if binding["descriptor_sha256"] != descriptor["self_relative_descriptor_sha256"]:
                    raise PrivatePermissionEvidenceContractError(
                        f"{label} descriptor binding mismatch"
                    )
            duplicate_value = slots["access_check_token_before"]["value"]
            if duplicate_value is not None:
                duplicate_facts = _mapping(
                    _mapping(duplicate_value, name=f"{label}.access_token")["facts"],
                    name=f"{label}.access_token.facts",
                )
                if (
                    checked["access_token_id_luid_hex"] != duplicate_facts["token_id_luid_hex"]
                    or checked["access_token_modified_id_luid_hex"]
                    != duplicate_facts["modified_id_luid_hex"]
                ):
                    raise PrivatePermissionEvidenceContractError(
                        f"{label} access-token binding mismatch"
                    )

    for slot_name, slot in slots.items():
        slot_value = slot["value"]
        if slot_name in array_names:
            for index, item in enumerate(slot_value):
                if item is not None:
                    validate_typed(slot_name, item, index)
        elif slot_value is not None:
            validate_typed(slot_name, slot_value)
    effective_before_value = slots["effective_context_before"]["value"]
    effective_after_value = slots["effective_context_after"]["value"]
    duplicate_before_value = slots["access_check_token_before"]["value"]
    duplicate_after_value = slots["access_check_token_after"]["value"]
    if effective_before_value is not None and effective_after_value is not None:
        if effective_before_value != effective_after_value:
            raise PrivatePermissionEvidenceContractError(
                "partial effective context before/after mismatch"
            )
    if duplicate_before_value is not None and duplicate_after_value is not None:
        if duplicate_before_value != duplicate_after_value:
            raise PrivatePermissionEvidenceContractError(
                "partial access token before/after mismatch"
            )
    if effective_before_value is not None and duplicate_before_value is not None:
        effective = _mapping(effective_before_value, name="partial effective token")
        duplicate = _mapping(duplicate_before_value, name="partial duplicate token")
        effective_facts = _mapping(effective["facts"], name="partial effective token facts")
        duplicate_facts = _mapping(duplicate["facts"], name="partial duplicate token facts")
        if duplicate["duplicated_from_token_id_luid_hex"] != effective_facts["token_id_luid_hex"]:
            raise PrivatePermissionEvidenceContractError(
                "partial duplicate source token binding mismatch"
            )
        authorization_projection = {
            "authentication_id_luid_hex", "modified_id_luid_hex", "session_id", "user",
            "owner", "primary_group", "ordered_groups", "ordered_restricted_sids",
            "ordered_privileges", "integrity_level_sid", "elevation_type", "is_elevated",
            "has_restrictions", "is_app_container", "ordered_capabilities",
            "virtualization_allowed", "virtualization_enabled", "ui_access",
        }
        if any(
            effective_facts[field] != duplicate_facts[field]
            for field in authorization_projection
        ):
            raise PrivatePermissionEvidenceContractError(
                "partial effective/duplicate authorization projection mismatch"
            )
    progress = _exact(
        partial["content_progress"],
        {"algorithm", "bytes_hashed", "prefix_sha256", "eof_observed"},
        name="partial_capture.content_progress",
    )
    if progress["algorithm"] != "sha256" or not isinstance(progress["eof_observed"], bool):
        raise PrivatePermissionEvidenceContractError("content progress envelope mismatch")
    bytes_hashed = _integer(progress["bytes_hashed"], name="content_progress.bytes_hashed")
    _sha(progress["prefix_sha256"], name="content_progress.prefix_sha256")
    if bytes_hashed == 0 and progress["prefix_sha256"] != sha256_bytes(b""):
        raise PrivatePermissionEvidenceContractError("empty content progress hash mismatch")
    opened = partial["opened_resource_identities"]
    prefixes = partial["raw_prefixes"]
    events = partial["resource_events"]
    if not all(isinstance(items, list) for items in (opened, prefixes, events)):
        raise PrivatePermissionEvidenceContractError("partial ledgers must be arrays")
    opened_sequences: set[int] = set()
    opened_rows: list[Mapping[str, object]] = []
    for index, item in enumerate(opened):
        row = _exact(item, _OPENED_RESOURCE_KEYS, name=f"opened_resource[{index}]")
        kind = row["resource_kind"]
        role = row["resource_role"]
        if (kind, role) not in {("directory_handle", "ancestor"), ("file_handle", "subject")}:
            raise PrivatePermissionEvidenceContractError("opened resource kind/role mismatch")
        if role == "ancestor":
            ordinal = _integer(row["resource_ordinal"], name="opened resource ordinal")
            if ordinal >= len(ancestor_paths):
                raise PrivatePermissionEvidenceContractError("opened ancestor ordinal mismatch")
        elif row["resource_ordinal"] is not None:
            raise PrivatePermissionEvidenceContractError("opened subject ordinal mismatch")
        sequence = _integer(row["acquisition_sequence"], name="opened acquisition sequence")
        if sequence in opened_sequences:
            raise PrivatePermissionEvidenceContractError("duplicate opened acquisition sequence")
        opened_sequences.add(sequence)
        _validate_identity(row["object_identity"], name="opened object identity")
        typed_object = (
            slots["ancestry_before"]["value"][row["resource_ordinal"]]
            if role == "ancestor"
            else slots["subject_before"]["value"]
        )
        if typed_object is not None:
            typed_facts = _mapping(
                _mapping(typed_object, name="opened bound typed object")["facts"],
                name="opened bound typed object facts",
            )
            if row["object_identity"] != typed_facts["identity"]:
                raise PrivatePermissionEvidenceContractError(
                    "opened resource identity differs from its completed typed slot"
                )
        opened_rows.append(row)
    prefix_sequences: set[int] = set()
    for index, item in enumerate(prefixes):
        row = _exact(item, _RAW_PREFIX_KEYS, name=f"raw_prefix[{index}]")
        sequence = _integer(row["capture_sequence"], name="raw capture sequence")
        if sequence in prefix_sequences:
            raise PrivatePermissionEvidenceContractError("duplicate raw capture sequence")
        prefix_sequences.add(sequence)
        if row["source_operation"] not in {
            "query_object", "query_streams", "acquire_security_descriptor",
            "parse_security_descriptor", "query_token",
        }:
            raise PrivatePermissionEvidenceContractError("raw prefix source operation mismatch")
        if row["encoding"] not in {
            "security_descriptor", "ace", "sid", "token_information",
            "stream_information", "object_information",
        }:
            raise PrivatePermissionEvidenceContractError("raw prefix encoding mismatch")
        sources_for_encoding = {
            "object_information": {"query_object"},
            "stream_information": {"query_streams"},
            "security_descriptor": {"acquire_security_descriptor", "parse_security_descriptor"},
            "ace": {"parse_security_descriptor"},
            "sid": {"parse_security_descriptor", "query_token"},
            "token_information": {"query_token"},
        }
        if row["source_operation"] not in sources_for_encoding[row["encoding"]]:
            raise PrivatePermissionEvidenceContractError("raw prefix source/encoding mismatch")
        kind = row["resource_kind"]
        role = row["resource_role"]
        if kind not in _RESOURCE_BINDINGS or kind in {"namespace_handle", "store_object_handle"}:
            raise PrivatePermissionEvidenceContractError("raw prefix resource kind mismatch")
        if role not in _RESOURCE_BINDINGS[kind]:
            raise PrivatePermissionEvidenceContractError("raw prefix resource role mismatch")
        if role in {"ancestor", "ancestor_security"}:
            _integer(row["resource_ordinal"], name="raw resource ordinal")
        elif row["resource_ordinal"] is not None:
            raise PrivatePermissionEvidenceContractError("raw prefix resource ordinal mismatch")
        if row["declared_size_bytes"] is not None:
            declared = _integer(row["declared_size_bytes"], name="raw declared size")
        else:
            declared = None
        captured = _integer(row["captured_size_bytes"], name="raw captured size")
        if type(row["complete"]) is not bool:
            raise PrivatePermissionEvidenceContractError("raw prefix complete must be Boolean")
        _validate_hex_shape(row["data_hex"], name="raw data")
        if (
            (declared is not None and declared < captured)
            or (row["complete"] is True and declared is not None and declared != captured)
            or captured != len(row["data_hex"]) // 2
            or row["captured_bytes_sha256"] != sha256_bytes(bytes.fromhex(row["data_hex"]))
        ):
            raise PrivatePermissionEvidenceContractError("raw prefix byte/hash mismatch")
    if prefix_sequences != set(range(len(prefixes))):
        raise PrivatePermissionEvidenceContractError("raw capture sequence is not contiguous")
    referenced: list[int] = []
    for slot_name, slot in slots.items():
        indices = slot["raw_prefix_indices"]
        if len(indices) != len(set(indices)):
            raise PrivatePermissionEvidenceContractError(
                f"partial_capture.{slot_name} repeats a raw prefix reference"
            )
        for index in indices:
            if index >= len(prefixes):
                raise PrivatePermissionEvidenceContractError(
                    f"partial_capture.{slot_name} raw prefix reference is out of range"
                )
            referenced.append(index)
    if sorted(referenced) != list(range(len(prefixes))):
        raise PrivatePermissionEvidenceContractError("raw prefixes are not referenced exactly once")
    event_rows: list[Mapping[str, object]] = []
    event_sequences: set[int] = set()
    for index, item in enumerate(events):
        event = _validate_resource_event(
            item, name=f"partial resource event[{index}]", success=False
        )
        sequence = event["acquisition_sequence"]
        if sequence in event_sequences:
            raise PrivatePermissionEvidenceContractError("duplicate resource acquisition sequence")
        event_sequences.add(sequence)
        event_rows.append(event)
    if event_sequences != set(range(len(events))):
        raise PrivatePermissionEvidenceContractError("resource acquisition sequence is not contiguous")
    for row in opened_rows:
        matches = [
            event for event in event_rows
            if event["resource_kind"] == row["resource_kind"]
            and event["resource_role"] == row["resource_role"]
            and event["resource_ordinal"] == row["resource_ordinal"]
            and event["acquisition_sequence"] == row["acquisition_sequence"]
            and event["object_identity"] == row["object_identity"]
        ]
        if len(matches) != 1:
            raise PrivatePermissionEvidenceContractError("opened resource/event binding mismatch")
    for event in event_rows:
        if event["object_identity"] is None:
            continue
        matches = [
            row for row in opened_rows
            if row["resource_kind"] == event["resource_kind"]
            and row["resource_role"] == event["resource_role"]
            and row["resource_ordinal"] == event["resource_ordinal"]
            and row["acquisition_sequence"] == event["acquisition_sequence"]
            and row["object_identity"] == event["object_identity"]
        ]
        if len(matches) != 1:
            raise PrivatePermissionEvidenceContractError(
                "identity-bearing resource event lacks its opened-resource row"
            )
    return partial


def _validate_ancestor_stability(record: Mapping[str, object]) -> None:
    before = record["ancestry_before"]
    after = record["ancestry_after"]
    if not isinstance(before, list) or not isinstance(after, list) or len(before) != len(after):
        raise PrivatePermissionEvidenceContractError("ancestry before/after cardinality mismatch")
    ignored_facts = {"end_of_file_bytes", "allocation_size_bytes", "number_of_links"}
    for index, (left_value, right_value) in enumerate(zip(before, after, strict=True)):
        left = _mapping(left_value, name=f"ancestry_before[{index}]")
        right = _mapping(right_value, name=f"ancestry_after[{index}]")
        if set(left) != set(right):
            raise PrivatePermissionEvidenceContractError("ancestry row keys changed")
        for key in left:
            if key != "facts" and left[key] != right[key]:
                raise PrivatePermissionEvidenceContractError(
                    f"ancestry binding changed at index {index}: {key}"
                )
        left_facts = _mapping(left["facts"], name="ancestor facts before")
        right_facts = _mapping(right["facts"], name="ancestor facts after")
        if set(left_facts) != set(right_facts):
            raise PrivatePermissionEvidenceContractError("ancestor fact keys changed")
        for key in left_facts:
            if key not in ignored_facts and left_facts[key] != right_facts[key]:
                raise PrivatePermissionEvidenceContractError(
                    f"ancestor stable fact changed at index {index}: {key}"
                )


def _validate_failure_item(value: object, *, name: str) -> Mapping[str, object]:
    item = _exact(value, _FAILURE_ITEM_KEYS, name=name)
    kind = item["failure_kind"]
    stage = item["stage"]
    operation = item["component_operation"]
    reason = item["reason_code"]
    if kind not in _FAILURE_KINDS or stage not in _STAGES or operation not in _OPERATIONS:
        raise PrivatePermissionEvidenceContractError(f"{name} uses a non-closed failure literal")
    if reason not in _REASONS:
        raise PrivatePermissionEvidenceContractError(f"{name}.reason_code is not closed")
    rules = _FAILURE_RULES.get(operation, ())
    if not any(
        stage in stages and kind in kinds and reason in reasons
        for stages, kinds, reasons in rules
    ):
        raise PrivatePermissionEvidenceContractError(f"{name} failure matrix mismatch")
    if reason == "native_protocol_failure" and (
        kind != "native_protocol_failure" or item["backend_error_hex"] is not None
    ):
        raise PrivatePermissionEvidenceContractError("native protocol failure equation mismatch")
    if reason == "lease_scope_mismatch" and kind != "lease_binding_failure":
        raise PrivatePermissionEvidenceContractError("lease scope failure equation mismatch")
    if item["backend_error_hex"] is not None:
        _dword_hex(item["backend_error_hex"], name=f"{name}.backend_error_hex")
        if item["backend_error_hex"] == "0x0":
            raise PrivatePermissionEvidenceContractError("backend error cannot be zero")
        if kind not in {"native_api_failure", "store_operation_failure", "cleanup_failure"}:
            raise PrivatePermissionEvidenceContractError(
                f"{name} pure failure cannot carry a backend error"
            )
        if reason in {"native_protocol_failure", "store_protocol_failure", "collision"}:
            raise PrivatePermissionEvidenceContractError(
                f"{name} protocol/collision failure cannot invent a backend error"
            )
    for integer_name in ("resource_ordinal", "operation_ordinal"):
        if item[integer_name] is not None:
            _integer(item[integer_name], name=f"{name}.{integer_name}")
    resource_kind = item["resource_kind"]
    resource_role = item["resource_role"]
    if (resource_kind is None) is not (resource_role is None):
        raise PrivatePermissionEvidenceContractError(f"{name} has partial resource binding")
    pure_operations = {
        "validate_request", "validate_access_expectation", "validate_store_expectation",
        "acquire_authority_lease", "validate_authority_lease", "require_active",
        "finish_authority_lease", "validate_private_record",
        "canonicalize_private_record", "build_destination", "validate_destination",
        "build_retention_failure", "validate_retention_failure",
    }
    if operation in pure_operations and resource_kind is not None:
        raise PrivatePermissionEvidenceContractError(f"{name} pure failure has a resource")
    if resource_kind is not None:
        if (
            resource_kind not in _RESOURCE_BINDINGS
            or resource_role not in _RESOURCE_BINDINGS[resource_kind]
        ):
            raise PrivatePermissionEvidenceContractError(f"{name} resource kind/role mismatch")
        if resource_role in {"ancestor", "ancestor_security"}:
            _integer(item["resource_ordinal"], name=f"{name}.resource_ordinal")
        elif item["resource_ordinal"] is not None:
            raise PrivatePermissionEvidenceContractError(f"{name} unexpected resource ordinal")
        compatible: dict[str, set[str]] = {
            "directory_handle": {
                "open_existing_directory", "query_object", "release_native_resource",
                "acquire_security_descriptor", "validate_capture",
            },
            "file_handle": {
                "open_existing_subject", "query_object", "query_streams", "read_unnamed_at",
                "acquire_security_descriptor", "release_native_resource", "validate_capture",
            },
            "security_descriptor_buffer": {
                "acquire_security_descriptor", "parse_security_descriptor",
                "release_native_resource", "validate_capture",
            },
            "effective_token_handle": {
                "open_current_effective_token", "query_token", "duplicate_for_access_check",
                "release_native_resource", "validate_capture",
            },
            "access_check_token_handle": {
                "duplicate_for_access_check", "query_token", "access_check",
                "release_native_resource", "validate_capture",
            },
            "namespace_handle": {
                "open_namespace", "validate_namespace", "release_namespace",
            },
            "store_object_handle": {
                "create_exclusive", "validate_creation_evidence", "write_once",
                "flush_data_and_metadata", "rewind", "read", "validate_readback",
                "snapshot", "validate_snapshot", "publish_same_object_exclusive",
                "release_store_object",
            },
        }
        if operation not in compatible[resource_kind]:
            raise PrivatePermissionEvidenceContractError(f"{name} operation/resource mismatch")
        if (
            resource_role in {"ancestor", "ancestor_security"}
            and operation not in {"release_native_resource", "validate_capture"}
        ):
            expected_operation_ordinal = item["resource_ordinal"]
            if stage in {"security_before", "security_after", "object_after"}:
                expected_operation_ordinal += 1
            if item["operation_ordinal"] != expected_operation_ordinal:
                raise PrivatePermissionEvidenceContractError(
                    f"{name} ancestor operation/resource ordinal mismatch"
                )
        if (
            resource_role in {"subject", "subject_security"}
            and stage in {"security_before", "security_after", "object_after"}
            and operation not in {"release_native_resource", "validate_capture"}
            and item["operation_ordinal"] != 0
        ):
            raise PrivatePermissionEvidenceContractError(
                f"{name} subject operation ordinal mismatch"
            )
    elif operation in {
        "flush_data_and_metadata", "rewind", "read", "write_once", "snapshot",
        "validate_snapshot", "validate_readback", "release_store_object",
        "publish_same_object_exclusive", "validate_creation_evidence",
        "validate_namespace", "release_namespace", "release_native_resource",
    }:
        raise PrivatePermissionEvidenceContractError(f"{name} resource-specific failure is unbound")
    return item


def _require_failure(
    value: object,
    *,
    name: str,
    kinds: set[str],
    stages: set[str],
    operations: set[str],
    reasons: set[str],
) -> ComponentFailureItem:
    item = _validate_failure_item(value, name=name)
    if (
        item["failure_kind"] not in kinds
        or item["stage"] not in stages
        or item["component_operation"] not in operations
        or item["reason_code"] not in reasons
    ):
        raise PrivatePermissionEvidenceContractError(f"{name} failure matrix mismatch")
    if not isinstance(value, ComponentFailureItem):
        try:
            return ComponentFailureItem(**item)  # type: ignore[arg-type]
        except TypeError as exc:
            raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    return value


def _validate_authority_accounting(
    value: object,
    *,
    name: str,
    allow_prefinish: bool = False,
) -> Mapping[str, object]:
    accounting = _exact(value, _AUTHORITY_ACCOUNTING_KEYS, name=name)
    acquire = accounting["acquire_status"]
    candidate = accounting["lease_candidate_returned"]
    acquired = accounting["lease_acquired"]
    binding = accounting["lease_binding_status"]
    preterminal = accounting["preterminal_authority_reason_code"]
    attempted = accounting["finish_attempted"]
    requested = accounting["requested_outcome"]
    finish_status = accounting["finish_status"]
    certainty = accounting["effect_certainty"]
    recorded = accounting["recorded_outcome"]
    finish_reason = accounting["finish_reason_code"]
    if acquire not in {"granted", "refused", "failed"}:
        raise PrivatePermissionEvidenceContractError(f"{name} unknown acquire status")
    if type(candidate) is not bool or type(acquired) is not bool or type(attempted) is not bool:
        raise PrivatePermissionEvidenceContractError(f"{name} accounting flags must be Boolean")
    if binding not in {"not_applicable", "valid", "invalid"}:
        raise PrivatePermissionEvidenceContractError(f"{name} unknown binding status")
    if accounting["preterminal_authority_reason_code"] not in _PRETERMINAL_REASONS:
        raise PrivatePermissionEvidenceContractError("unknown preterminal authority reason")
    if finish_reason not in {"none", "terminal_finish_failed"}:
        raise PrivatePermissionEvidenceContractError("unknown finish reason")
    if requested not in {None, "succeeded", "refused", "failed"}:
        raise PrivatePermissionEvidenceContractError(f"{name} unknown requested outcome")
    if recorded not in {None, "succeeded", "refused", "failed"}:
        raise PrivatePermissionEvidenceContractError(f"{name} unknown recorded outcome")
    if finish_status not in {"not_applicable", "succeeded", "failed"}:
        raise PrivatePermissionEvidenceContractError(f"{name} unknown finish status")
    if certainty not in {"not_applicable", "confirmed", "not_applied", "unknown"}:
        raise PrivatePermissionEvidenceContractError(f"{name} unknown finish certainty")
    no_finish = (
        attempted is False
        and requested is None
        and finish_status == "not_applicable"
        and certainty == "not_applicable"
        and recorded is None
        and finish_reason == "none"
    )
    if acquire in {"refused", "failed"}:
        expected_reason = (
            "authority_acquire_refused" if acquire == "refused" else "authority_acquire_failed"
        )
        if (
            candidate is not False
            or acquired is not False
            or binding != "not_applicable"
            or preterminal != expected_reason
            or not no_finish
        ):
            raise PrivatePermissionEvidenceContractError(f"{name} acquisition branch mismatch")
        return accounting
    if candidate is not True:
        raise PrivatePermissionEvidenceContractError(f"{name} granted acquisition lacks candidate")
    if acquired is False:
        if binding != "invalid" or preterminal != "lease_malformed" or not no_finish:
            raise PrivatePermissionEvidenceContractError(f"{name} malformed lease branch mismatch")
        return accounting
    if binding == "invalid":
        if preterminal not in (_LEASE_REASONS - {"lease_malformed"}):
            raise PrivatePermissionEvidenceContractError(f"{name} lease mismatch reason mismatch")
        if not attempted and allow_prefinish:
            if not no_finish:
                raise PrivatePermissionEvidenceContractError(f"{name} malformed pre-finish tuple")
            return accounting
        if requested != "failed":
            raise PrivatePermissionEvidenceContractError(f"{name} mismatched lease must finish failed")
    elif binding == "valid":
        if preterminal not in {"none", "authority_inactive", "authority_check_failed"}:
            raise PrivatePermissionEvidenceContractError(f"{name} valid lease reason mismatch")
        if not attempted and allow_prefinish:
            if not no_finish:
                raise PrivatePermissionEvidenceContractError(f"{name} malformed pre-finish tuple")
            return accounting
    else:
        raise PrivatePermissionEvidenceContractError(f"{name} granted candidate binding mismatch")
    if attempted is not True or requested is None:
        raise PrivatePermissionEvidenceContractError(f"{name} structurally valid lease was not finished")
    if finish_status == "succeeded":
        if certainty != "confirmed" or recorded != requested or finish_reason != "none":
            raise PrivatePermissionEvidenceContractError(f"{name} successful finish equation mismatch")
    elif finish_status == "failed":
        if finish_reason != "terminal_finish_failed":
            raise PrivatePermissionEvidenceContractError(f"{name} failed finish reason mismatch")
        if certainty == "confirmed":
            if recorded != requested:
                raise PrivatePermissionEvidenceContractError(f"{name} confirmed failed finish mismatch")
        elif certainty in {"not_applied", "unknown"}:
            if recorded is not None:
                raise PrivatePermissionEvidenceContractError(f"{name} uncertain finish invented outcome")
        else:
            raise PrivatePermissionEvidenceContractError(f"{name} failed finish certainty mismatch")
    else:
        raise PrivatePermissionEvidenceContractError(f"{name} attempted finish is not terminal")
    return accounting


def _validate_failure_accounting_equations(
    primary_value: object,
    follow_values: object,
    accounting_value: object,
    *,
    name: str,
    allow_prefinish: bool = False,
    expected_requested_outcome: str | None = None,
) -> tuple[Mapping[str, object], list[Mapping[str, object]], Mapping[str, object]]:
    primary = _validate_failure_item(primary_value, name=f"{name}.primary_failure")
    if not isinstance(follow_values, (list, tuple)):
        raise PrivatePermissionEvidenceContractError(f"{name}.follow-ons must be an array")
    follows = [
        _validate_failure_item(item, name=f"{name}.follow_on[{index}]")
        for index, item in enumerate(follow_values)
    ]
    accounting = _validate_authority_accounting(
        accounting_value, name=f"{name}.authority_accounting",
        allow_prefinish=allow_prefinish,
    )
    vector = [primary, *follows]
    terminal_indices = [
        index for index, item in enumerate(vector)
        if item["failure_kind"] == "terminal_accounting_failure"
    ]
    if terminal_indices and terminal_indices != [len(vector) - 1]:
        raise PrivatePermissionEvidenceContractError(
            f"{name} terminal-accounting failure is not last/unique"
        )
    cleanup_follows = follows[:-1] if terminal_indices else follows
    for item in cleanup_follows:
        if item["failure_kind"] != "cleanup_failure":
            raise PrivatePermissionEvidenceContractError(
                f"{name} non-cleanup follow-on precedes terminal accounting"
            )
    if terminal_indices and follows and follows[-1]["failure_kind"] != "terminal_accounting_failure":
        raise PrivatePermissionEvidenceContractError(f"{name} terminal failure ordering mismatch")
    finish_failed = accounting["finish_reason_code"] == "terminal_finish_failed"
    if allow_prefinish:
        if finish_failed or terminal_indices:
            raise PrivatePermissionEvidenceContractError(
                f"{name} pre-finish draft contains terminal accounting"
            )
    elif finish_failed != bool(terminal_indices):
        raise PrivatePermissionEvidenceContractError(
            f"{name} finish accounting/failure vector mismatch"
        )
    preterminal = accounting["preterminal_authority_reason_code"]
    authority_equations = {
        "authority_acquire_refused": (
            "authority_refusal", "authority_acquisition", "acquire_authority_lease",
            "authority_acquire_refused",
        ),
        "authority_acquire_failed": (
            "authority_failure", "authority_acquisition", "acquire_authority_lease",
            "authority_acquire_failed",
        ),
        **{
            reason: ("lease_binding_failure", "lease_binding", "validate_authority_lease", reason)
            for reason in _LEASE_REASONS
        },
        "authority_inactive": (
            "authority_refusal", None, "require_active", "authority_inactive",
        ),
        "authority_check_failed": (
            "authority_failure", None, "require_active", "authority_check_failed",
        ),
    }
    if preterminal != "none":
        expected = authority_equations[preterminal]
        actual = (
            primary["failure_kind"], primary["stage"],
            primary["component_operation"], primary["reason_code"],
        )
        if (
            actual[0] != expected[0]
            or (expected[1] is not None and actual[1] != expected[1])
            or actual[2:] != expected[2:]
        ):
            raise PrivatePermissionEvidenceContractError(
                f"{name} preterminal authority cause does not match primary failure"
            )
    elif primary["failure_kind"] in {
        "authority_refusal", "authority_failure", "lease_binding_failure"
    }:
        raise PrivatePermissionEvidenceContractError(
            f"{name} authority primary lacks preterminal accounting cause"
        )
    requested = accounting["requested_outcome"]
    if accounting["finish_attempted"] is True:
        derived = (
            "succeeded"
            if primary["failure_kind"] == "terminal_accounting_failure"
            else "refused"
            if primary["failure_kind"] in {"authority_refusal", "observation_refusal", "store_collision"}
            else "failed"
        )
        wanted = expected_requested_outcome or derived
        if requested != wanted:
            raise PrivatePermissionEvidenceContractError(
                f"{name} requested outcome does not match failure branch"
            )
    return primary, follows, accounting


def _validate_resource_failure_indices(record: Mapping[str, object]) -> None:
    primary = _validate_failure_item(record["primary_failure"], name="primary_failure")
    follow_values = record["ordered_follow_on_failures"]
    if not isinstance(follow_values, list):
        raise PrivatePermissionEvidenceContractError("ordered_follow_on_failures must be an array")
    follows = [
        _validate_failure_item(value, name=f"ordered_follow_on_failures[{index}]")
        for index, value in enumerate(follow_values)
    ]
    vector = [primary, *follows]
    partial = _mapping(record["partial_capture"], name="partial_capture")
    events = partial.get("resource_events")
    if not isinstance(events, list):
        raise PrivatePermissionEvidenceContractError("partial resource events must be an array")
    targets: set[int] = set()
    for index, event_value in enumerate(events):
        event = _mapping(event_value, name=f"resource_events[{index}]")
        if event.get("acquired") is not True or event.get("release_attempted") is not True:
            raise PrivatePermissionEvidenceContractError("every acquired native resource releases once")
        released = event.get("released")
        target = event.get("release_failure_index")
        if released is True:
            if target is not None:
                raise PrivatePermissionEvidenceContractError("released resource has failure index")
            continue
        if released is not False:
            raise PrivatePermissionEvidenceContractError("released must be Boolean")
        target_index = _integer(target, name="release_failure_index")
        if target_index >= len(vector) or target_index in targets:
            raise PrivatePermissionEvidenceContractError("release failure index is invalid or shared")
        targets.add(target_index)
        failure = vector[target_index]
        if (
            failure["failure_kind"] != "cleanup_failure"
            or failure["stage"] != "resource_cleanup"
            or failure["component_operation"] != "release_native_resource"
            or failure["reason_code"] != "resource_release_failed"
            or failure["resource_kind"] != event.get("resource_kind")
            or failure["resource_role"] != event.get("resource_role")
            or failure["resource_ordinal"] != event.get("resource_ordinal")
        ):
            raise PrivatePermissionEvidenceContractError("release failure target mismatch")
    cleanup_targets = {
        index for index, failure in enumerate(vector)
        if failure["failure_kind"] == "cleanup_failure"
        and failure["component_operation"] == "release_native_resource"
    }
    if targets != cleanup_targets:
        raise PrivatePermissionEvidenceContractError(
            "native cleanup failures and resource-event references differ"
        )


def validate_private_permission_observation(
    record: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
) -> None:
    _validate_expected_inputs(expected_request, expected_access)
    candidate = _exact(record, _OBSERVATION_KEYS, name="private permission observation")
    if candidate["schema"] != OBSERVATION_SCHEMA or candidate["outcome"] != "captured":
        raise PrivatePermissionEvidenceContractError("captured observation envelope mismatch")
    if candidate["capture_status"] != "capture_valid_policy_not_accepted":
        raise PrivatePermissionEvidenceContractError("capture status mismatch")
    if candidate["request"] != _plain(expected_request):
        raise PrivatePermissionEvidenceContractError("independent request mismatch")
    if candidate["access_expectation"] != _plain(expected_access):
        raise PrivatePermissionEvidenceContractError("independent access expectation mismatch")
    ancestry_before = candidate["ancestry_before"]
    ancestry_after = candidate["ancestry_after"]
    requested_ancestors = tuple(
        _get(
            _get(expected_request, "subject"),
            "required_ancestor_canonical_volume_guid_paths",
        )
    )
    if (
        not isinstance(ancestry_before, list)
        or not isinstance(ancestry_after, list)
        or len(ancestry_before) != len(requested_ancestors)
        or len(ancestry_after) != len(requested_ancestors)
    ):
        raise PrivatePermissionEvidenceContractError("ancestry cardinality mismatch")
    for index, (before_value, after_value, requested_path) in enumerate(
        zip(ancestry_before, ancestry_after, requested_ancestors, strict=True)
    ):
        before = _validate_object_snapshot_mapping(
            before_value, name=f"ancestry_before[{index}]", role="ancestor",
            ancestor_index=index,
        )
        after = _validate_object_snapshot_mapping(
            after_value, name=f"ancestry_after[{index}]", role="ancestor",
            ancestor_index=index,
        )
        if (
            before["requested_canonical_volume_guid_path"] != requested_path
            or after["requested_canonical_volume_guid_path"] != requested_path
            or before["is_repository_root"] is not (index == 0)
            or after["is_repository_root"] is not (index == 0)
            or before["is_immediate_parent"] is not (index == len(requested_ancestors) - 1)
            or after["is_immediate_parent"] is not (index == len(requested_ancestors) - 1)
        ):
            raise PrivatePermissionEvidenceContractError("ancestor positional binding mismatch")
    subject_before = _validate_object_snapshot_mapping(
        candidate["subject_before"], name="subject_before", role="subject"
    )
    subject_after = _validate_object_snapshot_mapping(
        candidate["subject_after"], name="subject_after", role="subject"
    )
    expected_subject_path = _get(
        _get(expected_request, "subject"), "canonical_volume_guid_path"
    )
    if (
        subject_before["requested_canonical_volume_guid_path"] != expected_subject_path
        or subject_after["requested_canonical_volume_guid_path"] != expected_subject_path
    ):
        raise PrivatePermissionEvidenceContractError("subject path binding mismatch")
    streams = candidate["stream_inventory"]
    if not isinstance(streams, list) or len(streams) != 1:
        raise PrivatePermissionEvidenceContractError(
            "stream inventory must contain only the unnamed data stream"
        )
    for index, stream in enumerate(streams):
        _validate_stream(stream, name=f"stream_inventory[{index}]")
        row = _mapping(stream, name=f"stream_inventory[{index}]")
        subject_facts_for_stream = _mapping(subject_before["facts"], name="subject facts")
        if (
            row["stream_index"] != index
            or row["stream_name"] != "::$DATA"
            or row["stream_size_bytes"] != subject_facts_for_stream["end_of_file_bytes"]
        ):
            raise PrivatePermissionEvidenceContractError("unnamed stream binding mismatch")
    content = _exact(
        candidate["content_evidence"],
        {"algorithm", "file_bytes", "file_sha256", "content_interpretation", "eof_observed"},
        name="content_evidence",
    )
    expected_subject = _get(expected_request, "subject")
    before_facts = _mapping(subject_before["facts"], name="subject before facts")
    after_facts = _mapping(subject_after["facts"], name="subject after facts")
    if (
        content["algorithm"] != "sha256"
        or content["file_bytes"] != _get(expected_subject, "expected_file_bytes")
        or content["file_sha256"] != _get(expected_subject, "expected_file_sha256")
        or content["content_interpretation"] != "opaque_bytes_not_deserialized"
        or content["eof_observed"] is not True
        or before_facts["end_of_file_bytes"] != content["file_bytes"]
        or after_facts["end_of_file_bytes"] != content["file_bytes"]
    ):
        raise PrivatePermissionEvidenceContractError("content evidence mismatch")
    subject_security_before = _validate_security(
        candidate["subject_security_before"], name="subject_security_before"
    )
    subject_security_after = _validate_security(
        candidate["subject_security_after"], name="subject_security_after"
    )
    subject_facts = _mapping(subject_before["facts"], name="subject facts")
    for bound in (subject_security_before, subject_security_after):
        if (
            bound["object_role"] != "subject"
            or bound["ancestor_index"] is not None
            or bound["object_identity"] != subject_facts["identity"]
            or bound["canonical_volume_guid_path"]
            != subject_facts["canonical_volume_guid_path"]
        ):
            raise PrivatePermissionEvidenceContractError("subject security binding mismatch")
    for key in ("ancestry_security_before", "ancestry_security_after"):
        values = candidate[key]
        if not isinstance(values, list) or len(values) != len(requested_ancestors):
            raise PrivatePermissionEvidenceContractError(f"{key} cardinality mismatch")
        for index, value in enumerate(values):
            bound = _validate_security(value, name=f"{key}[{index}]")
            ancestor_facts = _mapping(
                (ancestry_before if key.endswith("before") else ancestry_after)[index]["facts"],
                name="bound ancestor facts",
            )
            if (
                bound["object_role"] != "ancestor"
                or bound["ancestor_index"] != index
                or bound["object_identity"] != ancestor_facts["identity"]
                or bound["canonical_volume_guid_path"]
                != ancestor_facts["canonical_volume_guid_path"]
            ):
                raise PrivatePermissionEvidenceContractError(f"{key} positional mismatch")
    for key in (
        "effective_context_before", "access_check_token_before",
        "access_check_token_after", "effective_context_after",
    ):
        _validate_token(candidate[key], name=key)
    effective_before = _mapping(
        candidate["effective_context_before"], name="effective_context_before"
    )
    effective_facts = _mapping(effective_before["facts"], name="effective token facts")
    duplicate_binding = _mapping(
        candidate["access_check_token_before"], name="access duplicate"
    )
    if (
        effective_before["token_role"] != "effective_context"
        or duplicate_binding["token_role"] != "access_check_duplicate"
        or duplicate_binding["selection_source"] != "duplicated_effective_context"
        or duplicate_binding["duplicated_from_token_id_luid_hex"]
        != effective_facts["token_id_luid_hex"]
    ):
        raise PrivatePermissionEvidenceContractError("effective/duplicate token binding mismatch")
    duplicate_facts = _mapping(duplicate_binding["facts"], name="duplicate token facts")
    authorization_projection = {
        "authentication_id_luid_hex", "modified_id_luid_hex", "session_id", "user",
        "owner", "primary_group", "ordered_groups", "ordered_restricted_sids",
        "ordered_privileges", "integrity_level_sid", "elevation_type", "is_elevated",
        "has_restrictions", "is_app_container", "ordered_capabilities",
        "virtualization_allowed", "virtualization_enabled", "ui_access",
    }
    for key in authorization_projection:
        if effective_facts[key] != duplicate_facts[key]:
            raise PrivatePermissionEvidenceContractError(
                f"effective/access-check authorization projection mismatch: {key}"
            )
    access_checks = candidate["access_checks"]
    if not isinstance(access_checks, list) or len(access_checks) != 2:
        raise PrivatePermissionEvidenceContractError("access checks must contain exactly two rows")
    duplicate_before = _mapping(
        candidate["access_check_token_before"], name="access_check_token_before"
    )
    duplicate_facts = _mapping(duplicate_before["facts"], name="duplicate token facts")
    for index, value in enumerate(access_checks):
        checked = _validate_access_outcome(value, name=f"access_checks[{index}]")
        expected_object = subject_before if index == 0 else ancestry_before[-1]
        expected_security = _mapping(
            candidate["subject_security_before"]
            if index == 0
            else candidate["ancestry_security_before"][-1],
            name="expected access security",
        )
        expected_descriptor = _mapping(
            expected_security["descriptor"], name="expected access descriptor"
        )["self_relative_descriptor_sha256"]
        expected_mapping = _plain(
            _get(
                expected_access,
                "file_generic_mapping" if index == 0 else "directory_generic_mapping",
            )
        )
        binding = _mapping(checked["object_binding"], name="access object binding")
        object_facts = _mapping(expected_object["facts"], name="access object facts")
        if (
            checked["desired_access_hex"] != _get(expected_access, "desired_access_hex")
            or checked["generic_mapping"] != expected_mapping
            or checked["access_token_id_luid_hex"] != duplicate_facts["token_id_luid_hex"]
            or checked["access_token_modified_id_luid_hex"]
            != duplicate_facts["modified_id_luid_hex"]
            or binding["object_role"] != ("subject" if index == 0 else "immediate_parent")
            or binding["object_kind"] != ("file" if index == 0 else "directory")
            or binding["canonical_volume_guid_path"] != object_facts["canonical_volume_guid_path"]
            or binding["identity"] != object_facts["identity"]
            or binding["descriptor_sha256"] != expected_descriptor
        ):
            raise PrivatePermissionEvidenceContractError("access outcome binding mismatch")
    if candidate["subject_before"] != candidate["subject_after"]:
        raise PrivatePermissionEvidenceContractError("subject before/after mismatch")
    if candidate["subject_security_before"] != candidate["subject_security_after"]:
        raise PrivatePermissionEvidenceContractError("subject security changed")
    if candidate["ancestry_security_before"] != candidate["ancestry_security_after"]:
        raise PrivatePermissionEvidenceContractError("ancestor security changed")
    if candidate["access_check_token_before"] != candidate["access_check_token_after"]:
        raise PrivatePermissionEvidenceContractError("access token changed")
    if candidate["effective_context_before"] != candidate["effective_context_after"]:
        raise PrivatePermissionEvidenceContractError("effective context changed")
    _validate_ancestor_stability(candidate)
    events = candidate["resource_events"]
    if not isinstance(events, list):
        raise PrivatePermissionEvidenceContractError("resource_events must be an array")
    acquisition_sequences: set[int] = set()
    resource_multiset: dict[tuple[str, str, int | None], int] = {}
    for index, value in enumerate(events):
        event = _validate_resource_event(
            value, name=f"resource_events[{index}]", success=True
        )
        if (
            event.get("acquired") is not True
            or event.get("release_attempted") is not True
            or event.get("released") is not True
            or event.get("release_failure_index") is not None
        ):
            raise PrivatePermissionEvidenceContractError("success resource ledger is incomplete")
        sequence = event["acquisition_sequence"]
        if sequence in acquisition_sequences:
            raise PrivatePermissionEvidenceContractError("duplicate resource acquisition sequence")
        acquisition_sequences.add(sequence)
        binding = (
            event["resource_kind"], event["resource_role"], event["resource_ordinal"]
        )
        resource_multiset[binding] = resource_multiset.get(binding, 0) + 1
        if event["resource_kind"] == "directory_handle":
            ordinal = event["resource_ordinal"]
            if ordinal >= len(ancestry_before) or event["object_identity"] != _mapping(
                ancestry_before[ordinal]["facts"], name="ancestor resource facts"
            )["identity"]:
                raise PrivatePermissionEvidenceContractError("ancestor resource identity mismatch")
        elif event["resource_kind"] == "file_handle" and event["object_identity"] != subject_facts["identity"]:
            raise PrivatePermissionEvidenceContractError("subject resource identity mismatch")
    if acquisition_sequences != set(range(len(events))):
        raise PrivatePermissionEvidenceContractError("resource acquisition sequence is not contiguous")
    expected_resources: dict[tuple[str, str, int | None], int] = {
        ("file_handle", "subject", None): 1,
        ("security_descriptor_buffer", "subject_security", None): 2,
        ("effective_token_handle", "effective_context_before", None): 1,
        ("access_check_token_handle", "access_check_duplicate", None): 1,
        ("effective_token_handle", "effective_context_after", None): 1,
    }
    for index in range(len(ancestry_before)):
        expected_resources[("directory_handle", "ancestor", index)] = 1
        expected_resources[("security_descriptor_buffer", "ancestor_security", index)] = 2
    if resource_multiset != expected_resources:
        raise PrivatePermissionEvidenceContractError("successful resource ledger multiset mismatch")
    _walk_scalar_syntax(candidate)
    _self_hash(candidate, field="record_self_sha256")


def _validate_complete_observation_failure_projection(
    partial: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
) -> None:
    slot_names = (
        "ancestry_before", "subject_before", "stream_inventory",
        "subject_security_before", "ancestry_security_before",
        "effective_context_before", "access_check_token_before", "access_checks",
        "subject_security_after", "ancestry_security_after",
        "access_check_token_after", "effective_context_after", "subject_after",
        "ancestry_after",
    )
    completion_operations = {
        "ancestry_before": "query_object",
        "subject_before": "query_object",
        "stream_inventory": "query_streams",
        "subject_security_before": "parse_security_descriptor",
        "ancestry_security_before": "parse_security_descriptor",
        "effective_context_before": "query_token",
        "access_check_token_before": "query_token",
        "access_checks": "access_check",
        "subject_security_after": "parse_security_descriptor",
        "ancestry_security_after": "parse_security_descriptor",
        "access_check_token_after": "query_token",
        "effective_context_after": "query_token",
        "subject_after": "query_object",
        "ancestry_after": "query_object",
    }
    slots = {
        name: _mapping(partial[name], name=f"complete partial {name}")
        for name in slot_names
    }
    for name, slot in slots.items():
        if (
            slot["state"] != "complete"
            or slot["last_completed_operation"] != completion_operations[name]
            or slot["raw_prefix_indices"] != []
        ):
            raise PrivatePermissionEvidenceContractError(
                f"complete failure projection {name} mismatch"
            )
    if partial["raw_prefixes"] != []:
        raise PrivatePermissionEvidenceContractError(
            "complete failure projection cannot contain raw prefixes"
        )
    progress = _mapping(
        partial["content_progress"], name="complete partial content_progress"
    )
    subject = _get(expected_request, "subject")
    if (
        progress["algorithm"] != "sha256"
        or progress["bytes_hashed"] != _get(subject, "expected_file_bytes")
        or progress["prefix_sha256"] != _get(subject, "expected_file_sha256")
        or progress["eof_observed"] is not True
    ):
        raise PrivatePermissionEvidenceContractError(
            "complete failure projection content mismatch"
        )
    unsigned = {
        "schema": OBSERVATION_SCHEMA,
        "request": _plain(expected_request),
        "access_expectation": _plain(expected_access),
        "outcome": "captured",
        "ancestry_before": slots["ancestry_before"]["value"],
        "subject_before": slots["subject_before"]["value"],
        "stream_inventory": slots["stream_inventory"]["value"],
        "content_evidence": {
            "algorithm": "sha256",
            "file_bytes": progress["bytes_hashed"],
            "file_sha256": progress["prefix_sha256"],
            "content_interpretation": "opaque_bytes_not_deserialized",
            "eof_observed": True,
        },
        "subject_security_before": slots["subject_security_before"]["value"],
        "ancestry_security_before": slots["ancestry_security_before"]["value"],
        "effective_context_before": slots["effective_context_before"]["value"],
        "access_check_token_before": slots["access_check_token_before"]["value"],
        "access_checks": slots["access_checks"]["value"],
        "subject_security_after": slots["subject_security_after"]["value"],
        "ancestry_security_after": slots["ancestry_security_after"]["value"],
        "access_check_token_after": slots["access_check_token_after"]["value"],
        "effective_context_after": slots["effective_context_after"]["value"],
        "subject_after": slots["subject_after"]["value"],
        "ancestry_after": slots["ancestry_after"]["value"],
        "resource_events": partial["resource_events"],
        "capture_status": "capture_valid_policy_not_accepted",
    }
    try:
        projected = with_self_hash(unsigned, "record_self_sha256")
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    validate_private_permission_observation(
        projected, expected_request=expected_request, expected_access=expected_access
    )


def _validate_observation_failure_chronology(
    partial: Mapping[str, object],
    primary: Mapping[str, object],
    follows: list[Mapping[str, object]],
    accounting: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
) -> None:
    completion_operations = {
        "ancestry_before": "query_object",
        "subject_before": "query_object",
        "stream_inventory": "query_streams",
        "subject_security_before": "parse_security_descriptor",
        "ancestry_security_before": "parse_security_descriptor",
        "effective_context_before": "query_token",
        "access_check_token_before": "query_token",
        "access_checks": "access_check",
        "subject_security_after": "parse_security_descriptor",
        "ancestry_security_after": "parse_security_descriptor",
        "access_check_token_after": "query_token",
        "effective_context_after": "query_token",
        "subject_after": "query_object",
        "ancestry_after": "query_object",
    }
    slots = {
        name: _mapping(partial[name], name=f"chronology partial {name}")
        for name in completion_operations
    }
    for name, slot in slots.items():
        if (
            slot["state"] in {"prefix", "complete"}
            and slot["last_completed_operation"] != completion_operations[name]
        ):
            raise PrivatePermissionEvidenceContractError(
                f"partial chronology {name} completion operation mismatch"
            )

    vector = [primary, *follows]
    for item in vector:
        if (
            item["failure_kind"] == "terminal_accounting_failure"
            and item["operation_ordinal"] is not None
        ):
            raise PrivatePermissionEvidenceContractError(
                "terminal observation accounting has an operation ordinal"
            )
        if item["failure_kind"] == "cleanup_failure" and (
            item["component_operation"] != "release_native_resource"
            or item["operation_ordinal"] is not None
        ):
            raise PrivatePermissionEvidenceContractError(
                "observation failure contains non-native cleanup chronology"
            )

    progress = _mapping(partial["content_progress"], name="chronology content_progress")
    empty_content = {
        "algorithm": "sha256",
        "bytes_hashed": 0,
        "prefix_sha256": sha256_bytes(b""),
        "eof_observed": False,
    }

    def require_no_native_evidence() -> None:
        if any(slot["state"] != "not_reached" for slot in slots.values()):
            raise PrivatePermissionEvidenceContractError(
                "pre-native failure contains completed capture evidence"
            )
        if progress != empty_content or any(
            partial[name] != []
            for name in ("opened_resource_identities", "raw_prefixes", "resource_events")
        ):
            raise PrivatePermissionEvidenceContractError(
                "pre-native failure contains native evidence"
            )

    def require_complete_capture_evidence() -> None:
        for name, slot in slots.items():
            if (
                slot["state"] != "complete"
                or slot["last_completed_operation"] != completion_operations[name]
                or slot["raw_prefix_indices"] != []
            ):
                raise PrivatePermissionEvidenceContractError(
                    f"cleanup failure lacks completed {name} evidence"
                )
        subject = _get(expected_request, "subject")
        if (
            partial["raw_prefixes"] != []
            or progress["algorithm"] != "sha256"
            or progress["bytes_hashed"] != _get(subject, "expected_file_bytes")
            or progress["prefix_sha256"] != _get(subject, "expected_file_sha256")
            or progress["eof_observed"] is not True
        ):
            raise PrivatePermissionEvidenceContractError(
                "cleanup failure lacks complete content evidence"
            )

    pre_native = (
        accounting["acquire_status"] in {"refused", "failed"}
        or accounting["lease_acquired"] is False
        or accounting["lease_binding_status"] == "invalid"
    )
    initial_active_failure = (
        primary["component_operation"] == "require_active"
        and primary["stage"] == "ancestor_before"
        and primary["operation_ordinal"] is None
    )
    if pre_native or initial_active_failure:
        if primary["operation_ordinal"] is not None:
            raise PrivatePermissionEvidenceContractError(
                "pre-native primary failure has an operation ordinal"
            )
        require_no_native_evidence()
        return

    if primary["failure_kind"] == "terminal_accounting_failure":
        if (
            follows
            or primary["operation_ordinal"] is not None
            or accounting["requested_outcome"] != "succeeded"
        ):
            raise PrivatePermissionEvidenceContractError(
                "post-finish observation failure chronology mismatch"
            )
        _validate_complete_observation_failure_projection(
            partial, expected_request=expected_request, expected_access=expected_access
        )
        return

    ranked_groups = (
        (0, ("ancestry_before",)),
        (1, ("subject_before",)),
        (2, ("stream_inventory",)),
        (4, ("subject_security_before", "ancestry_security_before")),
        (5, ("effective_context_before",)),
        (6, ("access_check_token_before",)),
        (7, ("access_checks",)),
        (8, ("subject_security_after", "ancestry_security_after")),
        (9, ("access_check_token_after",)),
        (10, ("subject_after", "ancestry_after")),
        (11, ("effective_context_after",)),
    )
    stage_ranks = {
        "ancestor_before": 0,
        "subject_before": 1,
        "stream_inventory": 2,
        "content_read": 3,
        "security_before": 4,
        "effective_context_before": 5,
        "access_token": 6,
        "access_check": 7,
        "security_after": 8,
        "access_token_after": 9,
        "object_after": 10,
        "effective_context_after": 11,
        "capture_assembly": 12,
        "capture_validation": 12,
        "final_authority_check": 12,
    }
    stage = primary["stage"]
    if stage == "resource_cleanup":
        if primary["component_operation"] != "release_native_resource":
            raise PrivatePermissionEvidenceContractError(
                "observation failure contains non-native cleanup primary"
            )
        require_complete_capture_evidence()
        return
    if stage not in stage_ranks:
        raise PrivatePermissionEvidenceContractError(
            "observation failure stage has no capture chronology"
        )
    stage_rank = stage_ranks[stage]
    repeated_ordinal_stages = {
        "ancestor_before", "content_read", "security_before", "access_check",
        "security_after", "object_after",
    }
    if stage not in repeated_ordinal_stages and primary["operation_ordinal"] is not None:
        raise PrivatePermissionEvidenceContractError(
            "observation failure has an unexpected operation ordinal"
        )

    def require_complete(names: tuple[str, ...]) -> None:
        for name in names:
            if slots[name]["state"] != "complete":
                raise PrivatePermissionEvidenceContractError(
                    f"partial chronology requires completed {name}"
                )

    def require_not_reached(names: tuple[str, ...]) -> None:
        for name in names:
            if slots[name]["state"] != "not_reached":
                raise PrivatePermissionEvidenceContractError(
                    f"partial chronology reached future slot {name}"
                )

    for rank, names in ranked_groups:
        if rank < stage_rank:
            require_complete(names)
        elif rank > stage_rank:
            require_not_reached(names)

    subject = _get(expected_request, "subject")
    expected_bytes = _get(subject, "expected_file_bytes")
    expected_sha256 = _get(subject, "expected_file_sha256")
    if stage_rank < 3:
        if progress != empty_content:
            raise PrivatePermissionEvidenceContractError(
                "partial chronology reached content too early"
            )
    elif stage_rank > 3:
        if (
            progress["bytes_hashed"] != expected_bytes
            or progress["prefix_sha256"] != expected_sha256
            or progress["eof_observed"] is not True
        ):
            raise PrivatePermissionEvidenceContractError(
                "partial chronology lacks completed content"
            )

    def completed_count(name: str) -> int:
        value = slots[name]["value"]
        if not isinstance(value, list):
            raise PrivatePermissionEvidenceContractError(
                f"partial chronology {name} is not an array"
            )
        count = sum(item is not None for item in value)
        expected_state = (
            "complete"
            if count == len(value)
            else "not_reached"
            if count == 0 and slots[name]["raw_prefix_indices"] == []
            else "prefix"
        )
        if slots[name]["state"] != expected_state:
            raise PrivatePermissionEvidenceContractError(
                f"partial chronology {name} state/prefix mismatch"
            )
        return count

    ancestor_count = len(_get(subject, "required_ancestor_canonical_volume_guid_paths"))
    if stage == "ancestor_before":
        ordinal = _integer(primary["operation_ordinal"], name="ancestor failure ordinal")
        if ordinal >= ancestor_count or completed_count("ancestry_before") != ordinal:
            raise PrivatePermissionEvidenceContractError(
                "ancestor failure/partial prefix chronology mismatch"
            )
    elif stage in {"subject_before", "stream_inventory"}:
        current = "subject_before" if stage == "subject_before" else "stream_inventory"
        if slots[current]["state"] == "complete":
            raise PrivatePermissionEvidenceContractError(
                f"{stage} failure contains a completed current slot"
            )
    elif stage == "content_read":
        ordinal = _integer(primary["operation_ordinal"], name="content failure ordinal")
        chunk_bytes = 1_048_576
        chunk_count = (expected_bytes + chunk_bytes - 1) // chunk_bytes
        if ordinal > chunk_count:
            raise PrivatePermissionEvidenceContractError("content failure ordinal is out of range")
        expected_prefix_bytes = min(ordinal * chunk_bytes, expected_bytes)
        if progress["bytes_hashed"] != expected_prefix_bytes:
            raise PrivatePermissionEvidenceContractError(
                "content failure/partial byte chronology mismatch"
            )
        if primary["reason_code"] == "content_hash_mismatch":
            if (
                ordinal != chunk_count
                or progress["eof_observed"] is not True
                or progress["prefix_sha256"] == expected_sha256
            ):
                raise PrivatePermissionEvidenceContractError(
                    "content hash failure evidence mismatch"
                )
        elif progress["eof_observed"] is not False:
            raise PrivatePermissionEvidenceContractError(
                "content failure claims an impossible EOF chronology"
            )
    elif stage in {"security_before", "security_after"}:
        ordinal = _integer(primary["operation_ordinal"], name="security failure ordinal")
        if ordinal > ancestor_count:
            raise PrivatePermissionEvidenceContractError("security failure ordinal is out of range")
        subject_name = (
            "subject_security_before" if stage == "security_before"
            else "subject_security_after"
        )
        ancestry_name = (
            "ancestry_security_before" if stage == "security_before"
            else "ancestry_security_after"
        )
        if ordinal == 0:
            if (
                slots[subject_name]["state"] == "complete"
                or slots[ancestry_name]["state"] != "not_reached"
            ):
                raise PrivatePermissionEvidenceContractError(
                    "subject security failure chronology mismatch"
                )
        elif (
            slots[subject_name]["state"] != "complete"
            or completed_count(ancestry_name) != ordinal - 1
        ):
            raise PrivatePermissionEvidenceContractError(
                "ancestor security failure chronology mismatch"
            )
    elif stage == "access_check":
        ordinal = _integer(primary["operation_ordinal"], name="access-check failure ordinal")
        if ordinal >= 2 or completed_count("access_checks") != ordinal:
            raise PrivatePermissionEvidenceContractError(
                "access-check failure/partial prefix chronology mismatch"
            )
    elif stage in {
        "effective_context_before", "access_token", "access_token_after",
        "effective_context_after",
    }:
        current = {
            "effective_context_before": "effective_context_before",
            "access_token": "access_check_token_before",
            "access_token_after": "access_check_token_after",
            "effective_context_after": "effective_context_after",
        }[stage]
        if slots[current]["state"] == "complete" and not (
            primary["failure_kind"] == "observation_refusal"
            and primary["component_operation"] == "query_token"
            and primary["reason_code"] == "token_unsupported"
        ):
            raise PrivatePermissionEvidenceContractError(
                f"{stage} failure contains an impossible completed current token"
            )
    elif stage == "object_after":
        ordinal = _integer(primary["operation_ordinal"], name="object-after failure ordinal")
        if ordinal > ancestor_count:
            raise PrivatePermissionEvidenceContractError("object-after failure ordinal is out of range")
        if ordinal == 0:
            if (
                slots["subject_after"]["state"] == "complete"
                or slots["ancestry_after"]["state"] != "not_reached"
            ):
                raise PrivatePermissionEvidenceContractError(
                    "subject-after failure chronology mismatch"
                )
        elif (
            slots["subject_after"]["state"] != "complete"
            or completed_count("ancestry_after") != ordinal - 1
        ):
            raise PrivatePermissionEvidenceContractError(
                "ancestor-after failure chronology mismatch"
            )
    elif stage == "final_authority_check":
        _validate_complete_observation_failure_projection(
            partial, expected_request=expected_request, expected_access=expected_access
        )


def validate_private_permission_observation_failure(
    record: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
) -> None:
    _validate_expected_inputs(expected_request, expected_access)
    candidate = _exact(
        record, _OBSERVATION_FAILURE_KEYS, name="private permission observation failure"
    )
    if candidate["schema"] != OBSERVATION_FAILURE_SCHEMA:
        raise PrivatePermissionEvidenceContractError("observation failure schema mismatch")
    if candidate["outcome"] not in {"refused", "failed"}:
        raise PrivatePermissionEvidenceContractError("observation failure outcome mismatch")
    if candidate["acceptance_emitted"] is not False:
        raise PrivatePermissionEvidenceContractError("failure cannot emit acceptance")
    if candidate["request"] != _plain(expected_request):
        raise PrivatePermissionEvidenceContractError("independent request mismatch")
    if candidate["access_expectation"] != _plain(expected_access):
        raise PrivatePermissionEvidenceContractError("independent access expectation mismatch")
    partial = _validate_partial_capture(
        candidate["partial_capture"],
        expected_request=expected_request,
        expected_access=expected_access,
    )
    _validate_resource_failure_indices(candidate)
    primary, follows, accounting = _validate_failure_accounting_equations(
        candidate["primary_failure"], candidate["ordered_follow_on_failures"],
        candidate["authority_accounting"], name="observation failure",
    )
    _validate_observation_failure_chronology(
        partial, primary, follows, accounting,
        expected_request=expected_request, expected_access=expected_access,
    )
    expected_outcome = (
        "refused"
        if primary["failure_kind"] in {
            "authority_refusal", "observation_refusal", "store_collision"
        }
        else "failed"
    )
    if candidate["outcome"] != expected_outcome:
        raise PrivatePermissionEvidenceContractError("failure outcome/primary kind mismatch")
    _walk_scalar_syntax(candidate)
    _self_hash(candidate, field="record_self_sha256")


def build_private_permission_observation_record(
    capture: CapturedPermissionObservation,
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
) -> dict[str, object]:
    record = dict(_mapping(capture, name="capture"))
    validate_private_permission_observation(
        record, expected_request=expected_request, expected_access=expected_access
    )
    return record


def build_private_permission_observation_failure_record(
    failure: FailedPermissionObservation,
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
) -> dict[str, object]:
    record = dict(_mapping(failure, name="failure"))
    validate_private_permission_observation_failure(
        record, expected_request=expected_request, expected_access=expected_access
    )
    return record


def canonical_private_permission_observation_bytes(
    record: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
) -> bytes:
    validate_private_permission_observation(
        record, expected_request=expected_request, expected_access=expected_access
    )
    try:
        return canonical_json_bytes(dict(record))
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc


def canonical_private_permission_observation_failure_bytes(
    record: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
) -> bytes:
    validate_private_permission_observation_failure(
        record, expected_request=expected_request, expected_access=expected_access
    )
    try:
        return canonical_json_bytes(dict(record))
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc


def strict_private_permission_observation_loads(
    data: bytes,
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
) -> dict[str, object]:
    try:
        record = strict_canonical_json_loads(data)
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    validate_private_permission_observation(
        record, expected_request=expected_request, expected_access=expected_access
    )
    return record


def strict_private_permission_observation_failure_loads(
    data: bytes,
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
) -> dict[str, object]:
    try:
        record = strict_canonical_json_loads(data)
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    validate_private_permission_observation_failure(
        record, expected_request=expected_request, expected_access=expected_access
    )
    return record


def validate_confidential_store_expectation(
    expectation: ConfidentialStoreExpectation,
    *,
    expected_request: PermissionObservationRequest,
) -> None:
    record = _exact(
        expectation,
        {
            "schema", "request_self_sha256", "attempt_id", "namespace_relative_path",
            "namespace_object_identity", "records_parent_relative_path",
            "records_parent_object_identity", "volume_identifier_commitment",
            "confidential_store_profile_sha256",
            "records_parent_security_descriptor_sha256",
            "required_created_object_security_descriptor_sha256",
            "creation_security_policy", "publication_policy", "expectation_self_sha256",
        },
        name="confidential store expectation",
    )
    if record["schema"] != STORE_EXPECTATION_SCHEMA:
        raise PrivatePermissionEvidenceContractError("store expectation schema mismatch")
    if record["request_self_sha256"] != _request_hash(expected_request):
        raise PrivatePermissionEvidenceContractError("store request binding mismatch")
    attempt = _nfc(_get(expected_request, "attempt_id"), name="request.attempt_id")
    if record["attempt_id"] != attempt:
        raise PrivatePermissionEvidenceContractError("store attempt mismatch")
    namespace = f"{PRIVATE_ROOT}/{attempt}"
    if record["namespace_relative_path"] != namespace:
        raise PrivatePermissionEvidenceContractError("namespace path mismatch")
    if record["records_parent_relative_path"] != f"{namespace}/records":
        raise PrivatePermissionEvidenceContractError("records parent mismatch")
    for key in (
        "namespace_relative_path", "records_parent_relative_path",
    ):
        _relative_path(record[key], name=key)
    for key in ("namespace_object_identity", "records_parent_object_identity"):
        _nfc(record[key], name=key)
    context = _get(expected_request, "context")
    if record["volume_identifier_commitment"] != _get(
        context, "volume_identifier_commitment"
    ):
        raise PrivatePermissionEvidenceContractError("store volume mismatch")
    for key in (
        "volume_identifier_commitment", "confidential_store_profile_sha256",
        "records_parent_security_descriptor_sha256",
        "required_created_object_security_descriptor_sha256",
        "expectation_self_sha256",
    ):
        _sha(record[key], name=key)
    if record["creation_security_policy"] != "security_descriptor_applied_atomically_at_create":
        raise PrivatePermissionEvidenceContractError("creation security policy mismatch")
    if record["publication_policy"] != (
        "same_parent_same_volume_same_object_no_replace_preserve_security"
    ):
        raise PrivatePermissionEvidenceContractError("publication policy mismatch")
    _self_hash(record, field="expectation_self_sha256")


def build_permission_evidence_destination(
    request: PermissionObservationRequest,
    *,
    payload_kind: Literal["observation", "observation_failure"],
    expected_store: ConfidentialStoreExpectation,
) -> PermissionEvidenceDestination:
    if payload_kind not in {"observation", "observation_failure"}:
        raise PrivatePermissionEvidenceContractError("invalid primary payload kind")
    validate_confidential_store_expectation(expected_store, expected_request=request)
    sequence = _integer(_get(request, "observation_sequence"), name="observation_sequence")
    phase = _nfc(_get(request, "observation_phase"), name="observation_phase")
    stem = f"{sequence:03d}-{phase}"
    parent = expected_store.records_parent_relative_path
    unsigned: dict[str, object] = {
        "schema": DESTINATION_SCHEMA,
        "payload_kind": payload_kind,
        "attempt_id": expected_store.attempt_id,
        "request_self_sha256": _request_hash(request),
        "observation_sequence": sequence,
        "observation_phase": phase,
        "namespace_relative_path": expected_store.namespace_relative_path,
        "namespace_object_identity": expected_store.namespace_object_identity,
        "records_parent_relative_path": parent,
        "records_parent_object_identity": expected_store.records_parent_object_identity,
        "volume_identifier_commitment": expected_store.volume_identifier_commitment,
        "confidential_store_profile_sha256": expected_store.confidential_store_profile_sha256,
        "required_created_object_security_descriptor_sha256": (
            expected_store.required_created_object_security_descriptor_sha256
        ),
        "primary_creating_relative_path": f"{parent}/{stem}.permission-evidence.json.__creating__",
        "primary_accepted_relative_path": f"{parent}/{stem}.permission-evidence.json",
        "retention_failure_creating_relative_path": f"{parent}/{stem}.retention-failure.json.__creating__",
        "retention_failure_accepted_relative_path": f"{parent}/{stem}.retention-failure.json",
        "store_expectation_self_sha256": expected_store.expectation_self_sha256,
    }
    try:
        signed = with_self_hash(unsigned, "destination_self_sha256")
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    destination = PermissionEvidenceDestination(**signed)  # type: ignore[arg-type]
    validate_permission_evidence_destination(
        destination, expected_request=request, expected_store=expected_store
    )
    return destination


def validate_permission_evidence_destination(
    destination: PermissionEvidenceDestination,
    *,
    expected_request: PermissionObservationRequest,
    expected_store: ConfidentialStoreExpectation,
) -> None:
    validate_confidential_store_expectation(expected_store, expected_request=expected_request)
    record = _exact(
        destination,
        {
            "schema", "payload_kind", "attempt_id", "request_self_sha256",
            "observation_sequence", "observation_phase", "namespace_relative_path",
            "namespace_object_identity", "records_parent_relative_path",
            "records_parent_object_identity", "volume_identifier_commitment",
            "confidential_store_profile_sha256",
            "required_created_object_security_descriptor_sha256",
            "primary_creating_relative_path", "primary_accepted_relative_path",
            "retention_failure_creating_relative_path",
            "retention_failure_accepted_relative_path", "store_expectation_self_sha256",
            "destination_self_sha256",
        },
        name="permission evidence destination",
    )
    if record["schema"] != DESTINATION_SCHEMA:
        raise PrivatePermissionEvidenceContractError("destination schema mismatch")
    if record["payload_kind"] not in {"observation", "observation_failure"}:
        raise PrivatePermissionEvidenceContractError("destination payload kind mismatch")
    if record["request_self_sha256"] != _request_hash(expected_request):
        raise PrivatePermissionEvidenceContractError("destination request mismatch")
    if record["store_expectation_self_sha256"] != expected_store.expectation_self_sha256:
        raise PrivatePermissionEvidenceContractError("destination store mismatch")
    sequence = _integer(_get(expected_request, "observation_sequence"), name="observation_sequence")
    phase = _nfc(_get(expected_request, "observation_phase"), name="observation_phase")
    if record["observation_sequence"] != sequence or record["observation_phase"] != phase:
        raise PrivatePermissionEvidenceContractError("destination subject/phase mismatch")
    equality = {
        "attempt_id": expected_store.attempt_id,
        "namespace_relative_path": expected_store.namespace_relative_path,
        "namespace_object_identity": expected_store.namespace_object_identity,
        "records_parent_relative_path": expected_store.records_parent_relative_path,
        "records_parent_object_identity": expected_store.records_parent_object_identity,
        "volume_identifier_commitment": expected_store.volume_identifier_commitment,
        "confidential_store_profile_sha256": expected_store.confidential_store_profile_sha256,
        "required_created_object_security_descriptor_sha256": (
            expected_store.required_created_object_security_descriptor_sha256
        ),
    }
    for key, expected in equality.items():
        if record[key] != expected:
            raise PrivatePermissionEvidenceContractError(f"destination {key} mismatch")
    stem = f"{sequence:03d}-{phase}"
    parent = expected_store.records_parent_relative_path
    paths = {
        "primary_creating_relative_path": f"{parent}/{stem}.permission-evidence.json.__creating__",
        "primary_accepted_relative_path": f"{parent}/{stem}.permission-evidence.json",
        "retention_failure_creating_relative_path": f"{parent}/{stem}.retention-failure.json.__creating__",
        "retention_failure_accepted_relative_path": f"{parent}/{stem}.retention-failure.json",
    }
    if len(set(paths.values())) != 4:
        raise PrivatePermissionEvidenceContractError("destination paths are not distinct")
    for key, expected in paths.items():
        _relative_path(record[key], name=key)
        if record[key] != expected:
            raise PrivatePermissionEvidenceContractError(f"destination {key} mismatch")
    _sha(record["destination_self_sha256"], name="destination_self_sha256")
    _self_hash(record, field="destination_self_sha256")


def build_permission_authz_evidence_destination(
    request: PermissionObservationRequest,
    *,
    expected_store: ConfidentialStoreExpectation,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding,
) -> PermissionAuthzEvidenceDestination:
    """Build the separately bound two-slot destination for durable Authz bytes."""

    validate_confidential_store_expectation(expected_store, expected_request=request)
    try:
        # The access object is already identified by the binding; its full value is
        # validated by the preparation boundary before this builder is called.
        if expected_authz_binding.request_self_sha256 != _request_hash(request):
            raise PrivatePermissionEvidenceContractError(
                "Authz destination binding request mismatch"
            )
        _sha(
            expected_authz_binding.binding_self_sha256,
            name="authz_binding_self_sha256",
        )
    except Exception as exc:
        if isinstance(exc, PrivatePermissionEvidenceContractError):
            raise
        raise PrivatePermissionEvidenceContractError(
            "Authz destination binding is malformed"
        ) from exc
    sequence = _integer(_get(request, "observation_sequence"), name="observation_sequence")
    phase = _nfc(_get(request, "observation_phase"), name="observation_phase")
    stem = f"{sequence:03d}-{phase}"
    parent = expected_store.records_parent_relative_path
    unsigned: dict[str, object] = {
        "schema": AUTHZ_DESTINATION_SCHEMA,
        "payload_kind": "authz_evidence",
        "attempt_id": expected_store.attempt_id,
        "request_self_sha256": _request_hash(request),
        "observation_sequence": sequence,
        "observation_phase": phase,
        "namespace_relative_path": expected_store.namespace_relative_path,
        "namespace_object_identity": expected_store.namespace_object_identity,
        "records_parent_relative_path": parent,
        "records_parent_object_identity": expected_store.records_parent_object_identity,
        "volume_identifier_commitment": expected_store.volume_identifier_commitment,
        "confidential_store_profile_sha256": expected_store.confidential_store_profile_sha256,
        "required_created_object_security_descriptor_sha256": (
            expected_store.required_created_object_security_descriptor_sha256
        ),
        "primary_creating_relative_path": (
            f"{parent}/{stem}.authz-evidence.json.__creating__"
        ),
        "primary_accepted_relative_path": f"{parent}/{stem}.authz-evidence.json",
        "retention_failure_creating_relative_path": (
            f"{parent}/{stem}.authz-retention-failure.json.__creating__"
        ),
        "retention_failure_accepted_relative_path": (
            f"{parent}/{stem}.authz-retention-failure.json"
        ),
        "store_expectation_self_sha256": expected_store.expectation_self_sha256,
        "authz_binding_self_sha256": expected_authz_binding.binding_self_sha256,
    }
    try:
        signed = with_self_hash(unsigned, "destination_self_sha256")
        destination = PermissionAuthzEvidenceDestination(**signed)  # type: ignore[arg-type]
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    validate_permission_authz_evidence_destination(
        destination,
        expected_request=request,
        expected_store=expected_store,
        expected_authz_binding=expected_authz_binding,
    )
    return destination


def validate_permission_authz_evidence_destination(
    destination: PermissionAuthzEvidenceDestination,
    *,
    expected_request: PermissionObservationRequest,
    expected_store: ConfidentialStoreExpectation,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding | None = None,
) -> None:
    validate_confidential_store_expectation(
        expected_store, expected_request=expected_request
    )
    record = _exact(
        destination,
        {
            "schema", "payload_kind", "attempt_id", "request_self_sha256",
            "observation_sequence", "observation_phase", "namespace_relative_path",
            "namespace_object_identity", "records_parent_relative_path",
            "records_parent_object_identity", "volume_identifier_commitment",
            "confidential_store_profile_sha256",
            "required_created_object_security_descriptor_sha256",
            "primary_creating_relative_path", "primary_accepted_relative_path",
            "retention_failure_creating_relative_path",
            "retention_failure_accepted_relative_path", "store_expectation_self_sha256",
            "authz_binding_self_sha256", "destination_self_sha256",
        },
        name="permission Authz evidence destination",
    )
    if type(destination) is not PermissionAuthzEvidenceDestination:
        raise PrivatePermissionEvidenceContractError(
            "Authz destination has the wrong exact type"
        )
    if (
        record["schema"] != AUTHZ_DESTINATION_SCHEMA
        or record["payload_kind"] != "authz_evidence"
    ):
        raise PrivatePermissionEvidenceContractError("Authz destination envelope mismatch")
    if record["request_self_sha256"] != _request_hash(expected_request):
        raise PrivatePermissionEvidenceContractError("Authz destination request mismatch")
    if record["store_expectation_self_sha256"] != expected_store.expectation_self_sha256:
        raise PrivatePermissionEvidenceContractError("Authz destination store mismatch")
    _sha(record["authz_binding_self_sha256"], name="authz_binding_self_sha256")
    if expected_authz_binding is not None:
        if type(expected_authz_binding) is not authz_records.PermissionAuthzEvidenceBinding:
            raise PrivatePermissionEvidenceContractError(
                "Authz destination expected binding has the wrong exact type"
            )
        if (
            expected_authz_binding.request_self_sha256 != _request_hash(expected_request)
            or record["authz_binding_self_sha256"]
            != expected_authz_binding.binding_self_sha256
        ):
            raise PrivatePermissionEvidenceContractError("Authz destination binding mismatch")
    sequence = _integer(
        _get(expected_request, "observation_sequence"), name="observation_sequence"
    )
    phase = _nfc(_get(expected_request, "observation_phase"), name="observation_phase")
    if type(record["observation_sequence"]) is not int or record["observation_sequence"] != sequence or record["observation_phase"] != phase:
        raise PrivatePermissionEvidenceContractError("Authz destination subject/phase mismatch")
    equality = {
        "attempt_id": expected_store.attempt_id,
        "namespace_relative_path": expected_store.namespace_relative_path,
        "namespace_object_identity": expected_store.namespace_object_identity,
        "records_parent_relative_path": expected_store.records_parent_relative_path,
        "records_parent_object_identity": expected_store.records_parent_object_identity,
        "volume_identifier_commitment": expected_store.volume_identifier_commitment,
        "confidential_store_profile_sha256": expected_store.confidential_store_profile_sha256,
        "required_created_object_security_descriptor_sha256": (
            expected_store.required_created_object_security_descriptor_sha256
        ),
    }
    for key, wanted in equality.items():
        if record[key] != wanted:
            raise PrivatePermissionEvidenceContractError(
                f"Authz destination {key} mismatch"
            )
    parent = expected_store.records_parent_relative_path
    stem = f"{sequence:03d}-{phase}"
    paths = {
        "primary_creating_relative_path": (
            f"{parent}/{stem}.authz-evidence.json.__creating__"
        ),
        "primary_accepted_relative_path": f"{parent}/{stem}.authz-evidence.json",
        "retention_failure_creating_relative_path": (
            f"{parent}/{stem}.authz-retention-failure.json.__creating__"
        ),
        "retention_failure_accepted_relative_path": (
            f"{parent}/{stem}.authz-retention-failure.json"
        ),
    }
    if len(set(paths.values())) != 4:
        raise PrivatePermissionEvidenceContractError("Authz destination paths are not distinct")
    for key, wanted in paths.items():
        _relative_path(record[key], name=key)
        if record[key] != wanted:
            raise PrivatePermissionEvidenceContractError(
                f"Authz destination {key} mismatch"
            )
    _sha(record["destination_self_sha256"], name="destination_self_sha256")
    _self_hash(record, field="destination_self_sha256")


def _validate_retention_destination(
    destination: PermissionEvidenceDestination | PermissionAuthzEvidenceDestination,
    *,
    expected_request: PermissionObservationRequest,
    expected_store: ConfidentialStoreExpectation,
) -> None:
    if type(destination) is PermissionEvidenceDestination:
        validate_permission_evidence_destination(
            destination,
            expected_request=expected_request,
            expected_store=expected_store,
        )
        return
    if type(destination) is PermissionAuthzEvidenceDestination:
        validate_permission_authz_evidence_destination(
            destination,
            expected_request=expected_request,
            expected_store=expected_store,
        )
        return
    raise PrivatePermissionEvidenceContractError(
        "retention destination has no closed validator"
    )


def build_private_permission_retention_failure_record(
    failure: PermissionEvidenceRetentionFailureDraft,
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: PermissionEvidenceDestination,
    include_only_pre_sidecar_state: Literal[True],
) -> dict[str, object]:
    if include_only_pre_sidecar_state is not True:
        raise PrivatePermissionEvidenceContractError("sidecar projection must be pre-sidecar only")
    _validate_failure_draft(
        failure,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
    )
    draft = _mapping(failure, name="retention failure draft")
    unsigned = {
        "schema": RETENTION_FAILURE_SCHEMA,
        "payload_kind": "retention_failure",
        "failed_primary_payload_kind": draft["failed_primary_payload_kind"],
        "request_self_sha256": draft["request_self_sha256"],
        "access_expectation_self_sha256": draft["access_expectation_self_sha256"],
        "destination_self_sha256": draft["destination_self_sha256"],
        "store_expectation_self_sha256": draft["store_expectation_self_sha256"],
        "primary_failure": draft["primary_failure"],
        "ordered_follow_on_failures": draft["ordered_follow_on_failures"],
        "main_authority_accounting": draft["main_authority_accounting"],
        "payload_effects": draft["payload_effects"],
        "payload_object_state": draft["payload_object_state"],
        "acceptance_emitted": False,
        "failure_sidecar_recursion_permitted": False,
    }
    try:
        record = with_self_hash(unsigned, "record_self_sha256")
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    validate_private_permission_retention_failure(
        record,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
    )
    return record


def validate_private_permission_retention_failure(
    record: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: PermissionEvidenceDestination,
) -> None:
    _validate_expected_inputs(expected_request, expected_access)
    validate_confidential_store_expectation(expected_store, expected_request=expected_request)
    validate_permission_evidence_destination(
        expected_destination,
        expected_request=expected_request,
        expected_store=expected_store,
    )
    candidate = _exact(record, _RETENTION_RECORD_KEYS, name="retention failure sidecar")
    if candidate["schema"] != RETENTION_FAILURE_SCHEMA or candidate["payload_kind"] != "retention_failure":
        raise PrivatePermissionEvidenceContractError("retention failure envelope mismatch")
    if candidate["failed_primary_payload_kind"] != expected_destination.payload_kind:
        raise PrivatePermissionEvidenceContractError("failed primary payload kind mismatch")
    expected_bindings = {
        "request_self_sha256": _request_hash(expected_request),
        "access_expectation_self_sha256": _access_hash(expected_access),
        "destination_self_sha256": expected_destination.destination_self_sha256,
        "store_expectation_self_sha256": expected_store.expectation_self_sha256,
    }
    for key, expected in expected_bindings.items():
        if candidate[key] != expected:
            raise PrivatePermissionEvidenceContractError(f"retention failure {key} mismatch")
    if candidate["acceptance_emitted"] is not False or candidate["failure_sidecar_recursion_permitted"] is not False:
        raise PrivatePermissionEvidenceContractError("retention failure cannot accept or recurse")
    primary, follows, _ = _validate_failure_accounting_equations(
        candidate["primary_failure"], candidate["ordered_follow_on_failures"],
        candidate["main_authority_accounting"], name="retention failure sidecar",
    )
    state = _validate_effect_terminal_fold(
        candidate["payload_effects"], candidate["payload_object_state"],
        destination=expected_destination, expected_store=expected_store,
    )
    if state.slot_kind != "primary" or state.payload_kind != expected_destination.payload_kind:
        raise PrivatePermissionEvidenceContractError("retention failure main state slot mismatch")
    _validate_effect_failure_equations(
        candidate["payload_effects"], primary, follows,
        destination=expected_destination, expected_store=expected_store,
    )
    _walk_scalar_syntax(candidate)
    _self_hash(candidate, field="record_self_sha256")


def canonical_private_permission_retention_failure_bytes(
    record: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: PermissionEvidenceDestination,
) -> bytes:
    validate_private_permission_retention_failure(
        record,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
    )
    try:
        return canonical_json_bytes(dict(record))
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc


def strict_private_permission_retention_failure_loads(
    data: bytes,
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: PermissionEvidenceDestination,
) -> dict[str, object]:
    try:
        record = strict_canonical_json_loads(data)
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    validate_private_permission_retention_failure(
        record,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
    )
    return record


def _intended_authz_projection(
    data: bytes,
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: PermissionAuthzEvidenceDestination,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding,
) -> dict[str, object]:
    if type(data) is not bytes:
        raise PrivatePermissionEvidenceContractError(
            "intended Authz record must be exact immutable bytes"
        )
    authz_records.strict_private_permission_authz_loads(
        data,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_binding=expected_authz_binding,
        expected_destination_self_sha256=expected_destination.destination_self_sha256,
        expected_store_expectation_self_sha256=expected_store.expectation_self_sha256,
    )
    commitment = authz_records.private_permission_authz_commitment(
        data,
        authorization_file_sha256=_authorization_hash(expected_request),
        expected_request=expected_request,
        expected_access=expected_access,
        expected_binding=expected_authz_binding,
        expected_destination_self_sha256=expected_destination.destination_self_sha256,
        expected_store_expectation_self_sha256=expected_store.expectation_self_sha256,
    )
    return {
        "record_hex": data.hex(),
        "file_bytes": len(data),
        "file_sha256": sha256_bytes(data),
        "commitment": commitment,
    }


def build_private_permission_authz_retention_failure_record(
    failure: PermissionAuthzEvidenceRetentionFailureDraft,
    *,
    intended_authz_record: bytes,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: PermissionAuthzEvidenceDestination,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding,
    include_only_pre_sidecar_state: Literal[True],
) -> dict[str, object]:
    if include_only_pre_sidecar_state is not True:
        raise PrivatePermissionEvidenceContractError(
            "Authz sidecar projection must be pre-sidecar only"
        )
    _validate_failure_draft(
        failure,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
        expected_payload_bytes=len(intended_authz_record),
        expected_payload_sha256=sha256_bytes(intended_authz_record),
        expected_payload_data=intended_authz_record,
    )
    intended = _intended_authz_projection(
        intended_authz_record,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
        expected_authz_binding=expected_authz_binding,
    )
    draft = _mapping(failure, name="Authz retention failure draft")
    unsigned = {
        "schema": AUTHZ_RETENTION_FAILURE_SCHEMA,
        "payload_kind": "authz_retention_failure",
        "failed_primary_payload_kind": "authz_evidence",
        "request_self_sha256": draft["request_self_sha256"],
        "access_expectation_self_sha256": draft[
            "access_expectation_self_sha256"
        ],
        "destination_self_sha256": draft["destination_self_sha256"],
        "store_expectation_self_sha256": draft[
            "store_expectation_self_sha256"
        ],
        "primary_failure": draft["primary_failure"],
        "ordered_follow_on_failures": draft["ordered_follow_on_failures"],
        "main_authority_accounting": draft["main_authority_accounting"],
        "payload_effects": draft["payload_effects"],
        "payload_object_state": draft["payload_object_state"],
        "acceptance_emitted": False,
        "failure_sidecar_recursion_permitted": False,
        "intended_authz_record": intended,
    }
    try:
        record = with_self_hash(unsigned, "record_self_sha256")
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    validate_private_permission_authz_retention_failure(
        record,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
        expected_authz_binding=expected_authz_binding,
    )
    return record


def validate_private_permission_authz_retention_failure(
    record: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: PermissionAuthzEvidenceDestination,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding,
) -> None:
    _validate_expected_inputs(expected_request, expected_access)
    authz_records.validate_permission_authz_evidence_binding(
        expected_authz_binding,
        expected_request=expected_request,
        expected_access=expected_access,
    )
    validate_permission_authz_evidence_destination(
        expected_destination,
        expected_request=expected_request,
        expected_store=expected_store,
        expected_authz_binding=expected_authz_binding,
    )
    candidate = _exact(
        record,
        _AUTHZ_RETENTION_RECORD_KEYS,
        name="Authz retention failure sidecar",
    )
    if (
        candidate["schema"] != AUTHZ_RETENTION_FAILURE_SCHEMA
        or candidate["payload_kind"] != "authz_retention_failure"
        or candidate["failed_primary_payload_kind"] != "authz_evidence"
    ):
        raise PrivatePermissionEvidenceContractError(
            "Authz retention failure envelope mismatch"
        )
    expected_bindings = {
        "request_self_sha256": _request_hash(expected_request),
        "access_expectation_self_sha256": _access_hash(expected_access),
        "destination_self_sha256": expected_destination.destination_self_sha256,
        "store_expectation_self_sha256": expected_store.expectation_self_sha256,
    }
    for key, wanted in expected_bindings.items():
        if candidate[key] != wanted:
            raise PrivatePermissionEvidenceContractError(
                f"Authz retention failure {key} mismatch"
            )
    if (
        candidate["acceptance_emitted"] is not False
        or candidate["failure_sidecar_recursion_permitted"] is not False
    ):
        raise PrivatePermissionEvidenceContractError(
            "Authz retention failure cannot accept or recurse"
        )
    intended = _exact(
        candidate["intended_authz_record"],
        _INTENDED_AUTHZ_RECORD_KEYS,
        name="intended Authz record",
    )
    _validate_hex_shape(intended["record_hex"], name="intended_authz_record.record_hex")
    try:
        intended_bytes = bytes.fromhex(intended["record_hex"])
    except (TypeError, ValueError) as exc:
        raise PrivatePermissionEvidenceContractError(
            "intended Authz record hex is malformed"
        ) from exc
    projection = _intended_authz_projection(
        intended_bytes,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
        expected_authz_binding=expected_authz_binding,
    )
    if dict(intended) != projection:
        raise PrivatePermissionEvidenceContractError(
            "intended Authz record bytes/hash/commitment mismatch"
        )
    primary, follows, accounting = _validate_failure_accounting_equations(
        candidate["primary_failure"],
        candidate["ordered_follow_on_failures"],
        candidate["main_authority_accounting"],
        name="Authz retention failure sidecar",
    )
    draft = PermissionAuthzEvidenceRetentionFailureDraft(
        failed_primary_payload_kind="authz_evidence",
        request_self_sha256=_request_hash(expected_request),
        access_expectation_self_sha256=_access_hash(expected_access),
        destination_self_sha256=expected_destination.destination_self_sha256,
        store_expectation_self_sha256=expected_store.expectation_self_sha256,
        primary_failure=primary,
        ordered_follow_on_failures=follows,
        main_authority_accounting=accounting,
        payload_effects=_validate_effect_array(
            candidate["payload_effects"], expected_destination, expected_store
        ),
        payload_object_state=_validate_terminal_state(
            candidate["payload_object_state"],
            expected_destination,
            expected_payload_data=intended_bytes,
        ),
        acceptance_emitted=False,
    )
    _validate_failure_draft(
        draft,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
        expected_payload_data=intended_bytes,
    )
    _walk_scalar_syntax(candidate)
    _self_hash(candidate, field="record_self_sha256")


def canonical_private_permission_authz_retention_failure_bytes(
    record: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: PermissionAuthzEvidenceDestination,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding,
) -> bytes:
    validate_private_permission_authz_retention_failure(
        record,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
        expected_authz_binding=expected_authz_binding,
    )
    try:
        return canonical_json_bytes(dict(record))
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc


def strict_private_permission_authz_retention_failure_loads(
    data: bytes,
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: PermissionAuthzEvidenceDestination,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding,
) -> dict[str, object]:
    try:
        record = strict_canonical_json_loads(data)
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    validate_private_permission_authz_retention_failure(
        record,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
        expected_authz_binding=expected_authz_binding,
    )
    return record


def private_permission_observation_commitment(
    data: bytes,
    *,
    authorization_file_sha256: str,
    payload_kind: PayloadKind,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation | None,
    expected_destination: PermissionEvidenceDestination | None,
) -> str:
    authorization = _sha(authorization_file_sha256, name="authorization_file_sha256")
    if authorization != _authorization_hash(expected_request):
        raise PrivatePermissionEvidenceContractError(
            "authorization commitment does not match the request-bound full A4"
        )
    if payload_kind == "observation":
        if expected_store is not None or expected_destination is not None:
            raise PrivatePermissionEvidenceContractError("observation commitment has no store binding")
        strict_private_permission_observation_loads(
            data, expected_request=expected_request, expected_access=expected_access
        )
        domain = OBSERVATION_DOMAIN
    elif payload_kind == "observation_failure":
        if expected_store is not None or expected_destination is not None:
            raise PrivatePermissionEvidenceContractError("observation failure commitment has no store binding")
        strict_private_permission_observation_failure_loads(
            data, expected_request=expected_request, expected_access=expected_access
        )
        domain = OBSERVATION_FAILURE_DOMAIN
    elif payload_kind == "retention_failure":
        if expected_store is None or expected_destination is None:
            raise PrivatePermissionEvidenceContractError("retention failure requires store binding")
        strict_private_permission_retention_failure_loads(
            data,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=expected_store,
            expected_destination=expected_destination,
        )
        domain = RETENTION_FAILURE_DOMAIN
    else:
        raise PrivatePermissionEvidenceContractError("unknown commitment payload kind")
    return sha256_bytes(
        domain.encode("utf-8") + b"\n" + authorization.encode("ascii") + b"\n" + data
    )


def _retention_payload_contract(
    payload_kind: PayloadKind,
) -> tuple[str, RetentionSlotKind, str]:
    """Closed dispatch for the five permitted retention payloads."""

    contracts: dict[PayloadKind, tuple[str, RetentionSlotKind, str]] = {
        "observation": (
            "retain_observation", "primary", OBSERVATION_DOMAIN
        ),
        "observation_failure": (
            "retain_observation_failure", "primary", OBSERVATION_FAILURE_DOMAIN
        ),
        "retention_failure": (
            "retain_retention_failure", "retention_failure", RETENTION_FAILURE_DOMAIN
        ),
        "authz_evidence": (
            "retain_authz_evidence",
            "authz_primary",
            authz_records.PRIVATE_PERMISSION_AUTHZ_DOMAIN,
        ),
        "authz_retention_failure": (
            "retain_authz_retention_failure",
            "authz_retention_failure",
            AUTHZ_RETENTION_FAILURE_DOMAIN,
        ),
    }
    try:
        return contracts[payload_kind]
    except (KeyError, TypeError) as exc:
        raise PrivatePermissionEvidenceContractError(
            "payload kind has no closed retention contract"
        ) from exc


def _validate_embedded_authz_base(
    base: authz_records.PermissionAuthzBaseObservation,
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
) -> None:
    """Validate intrinsic base claims; this supplies no independent corroboration."""
    data = None if base.record_hex is None else bytes.fromhex(base.record_hex)
    kind = {"captured": "observation", "failed": "observation_failure"}.get(base.outcome)
    if data is not None:
        loader = (
            strict_private_permission_observation_loads
            if kind == "observation" else strict_private_permission_observation_failure_loads
        )
        loader(data, expected_request=expected_request, expected_access=expected_access)
    retained = base.retention_result
    if retained is None:
        return
    if kind is None or data is None:
        raise PrivatePermissionEvidenceContractError("base retention result lacks its intended payload")
    destination = build_permission_evidence_destination(
        expected_request, payload_kind=kind, expected_store=expected_store,
    )
    if base.retention_result_kind == "retention_failure":
        draft_names = PermissionEvidenceRetentionFailureDraft.__dataclass_fields__
        draft = PermissionEvidenceRetentionFailureDraft(**{key: retained[key] for key in draft_names})
        result = PermissionEvidenceRetentionFailure(**retained)
        _validate_failure_draft(
            draft, expected_request=expected_request, expected_access=expected_access,
            expected_store=expected_store, expected_destination=destination,
            expected_payload_bytes=len(data), expected_payload_sha256=sha256_bytes(data),
            expected_payload_data=data,
        )
        _validate_retention_failure_result(
            result, draft=draft, expected_request=expected_request, expected_access=expected_access,
            expected_store=expected_store, expected_destination=destination,
        )
        return
    expected_tag = "captured_receipt" if kind == "observation" else "failure_receipt"
    if base.retention_result_kind != expected_tag:
        raise PrivatePermissionEvidenceContractError("base outcome/receipt kind mismatch")
    cls = RetainedPermissionEvidence if kind == "observation" else RetainedPermissionObservationFailure
    _exact(retained, set(cls.__dataclass_fields__), name="embedded base receipt")
    commitment = private_permission_observation_commitment(
        data, authorization_file_sha256=_authorization_hash(expected_request),
        payload_kind=kind, expected_request=expected_request, expected_access=expected_access,
        expected_store=None, expected_destination=None,
    )
    expected = {
        "schema": "operatormark.darcy_branch_c_v1." + (
            "retained_private_permission_evidence.v2" if kind == "observation"
            else "retained_private_permission_observation_failure.v2"
        ),
        "payload_kind": kind,
        "request_self_sha256": _request_hash(expected_request),
        "access_expectation_self_sha256": _access_hash(expected_access),
        "destination_self_sha256": destination.destination_self_sha256,
        "store_expectation_self_sha256": expected_store.expectation_self_sha256,
        "relative_path": destination.primary_accepted_relative_path,
        "file_bytes": len(data), "file_sha256": sha256_bytes(data),
        "commitment_domain": OBSERVATION_DOMAIN if kind == "observation" else OBSERVATION_FAILURE_DOMAIN,
        "authorization_file_sha256": _authorization_hash(expected_request),
        "private_observation_commitment": commitment,
        "namespace_object_identity": expected_store.namespace_object_identity,
        "records_parent_object_identity": expected_store.records_parent_object_identity,
        "creation_security_descriptor_sha256": expected_store.required_created_object_security_descriptor_sha256,
        "confidential_store_profile_sha256": expected_store.confidential_store_profile_sha256,
        "held_object_readback_sha256": sha256_bytes(data),
        "publication_state": (
            "retained_capture_only_not_permission_accepted" if kind == "observation"
            else "retained_observation_failure_not_permission_accepted"
        ),
    }
    if any(type(retained[key]) is not type(value) or retained[key] != value for key, value in expected.items()):
        raise PrivatePermissionEvidenceContractError("embedded base receipt equations differ")
    _nfc(retained["store_object_identity"], name="embedded base store object identity")
    validate_self_hash(dict(retained), "receipt_self_sha256")


def _validate_authz_record_for_retention(
    data: bytes,
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: PermissionAuthzEvidenceDestination,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding,
) -> dict[str, object]:
    record = authz_records.strict_private_permission_authz_loads(
        data,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_binding=expected_authz_binding,
        expected_destination_self_sha256=expected_destination.destination_self_sha256,
        expected_store_expectation_self_sha256=expected_store.expectation_self_sha256,
    )
    _validate_embedded_authz_base(
        authz_records._decode_base_observation(record["base_observation"]),
        expected_request=expected_request, expected_access=expected_access,
        expected_store=expected_store,
    )
    return record


def _validate_retention_payload_bytes(
    data: bytes,
    *,
    payload_kind: PayloadKind,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: RetentionDestination,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding | None,
) -> None:
    _retention_payload_contract(payload_kind)
    if payload_kind == "observation":
        if type(expected_destination) is not PermissionEvidenceDestination:
            raise PrivatePermissionEvidenceContractError("observation destination type mismatch")
        strict_private_permission_observation_loads(
            data, expected_request=expected_request, expected_access=expected_access
        )
    elif payload_kind == "observation_failure":
        if type(expected_destination) is not PermissionEvidenceDestination:
            raise PrivatePermissionEvidenceContractError(
                "observation failure destination type mismatch"
            )
        strict_private_permission_observation_failure_loads(
            data, expected_request=expected_request, expected_access=expected_access
        )
    elif payload_kind == "retention_failure":
        if type(expected_destination) is not PermissionEvidenceDestination:
            raise PrivatePermissionEvidenceContractError("sidecar destination type mismatch")
        strict_private_permission_retention_failure_loads(
            data,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=expected_store,
            expected_destination=expected_destination,
        )
    elif payload_kind == "authz_evidence":
        if (
            type(expected_destination) is not PermissionAuthzEvidenceDestination
            or type(expected_authz_binding)
            is not authz_records.PermissionAuthzEvidenceBinding
        ):
            raise PrivatePermissionEvidenceContractError(
                "Authz payload lacks its exact destination/binding"
            )
        _validate_authz_record_for_retention(
            data,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=expected_store,
            expected_destination=expected_destination,
            expected_authz_binding=expected_authz_binding,
        )
    elif payload_kind == "authz_retention_failure":
        if (
            type(expected_destination) is not PermissionAuthzEvidenceDestination
            or type(expected_authz_binding)
            is not authz_records.PermissionAuthzEvidenceBinding
        ):
            raise PrivatePermissionEvidenceContractError(
                "Authz sidecar lacks its exact destination/binding"
            )
        strict_private_permission_authz_retention_failure_loads(
            data,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=expected_store,
            expected_destination=expected_destination,
            expected_authz_binding=expected_authz_binding,
        )


def _retention_payload_commitment(
    data: bytes,
    *,
    payload_kind: PayloadKind,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: RetentionDestination,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding | None = None,
) -> str:
    _validate_retention_payload_bytes(
        data,
        payload_kind=payload_kind,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
        expected_authz_binding=expected_authz_binding,
    )
    if payload_kind in {"observation", "observation_failure", "retention_failure"}:
        return private_permission_observation_commitment(
            data,
            authorization_file_sha256=_authorization_hash(expected_request),
            payload_kind=payload_kind,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=(expected_store if payload_kind == "retention_failure" else None),
            expected_destination=(
                expected_destination if payload_kind == "retention_failure" else None
            ),
        )
    if payload_kind == "authz_evidence":
        assert type(expected_destination) is PermissionAuthzEvidenceDestination
        assert type(expected_authz_binding) is authz_records.PermissionAuthzEvidenceBinding
        return authz_records.private_permission_authz_commitment(
            data,
            authorization_file_sha256=_authorization_hash(expected_request),
            expected_request=expected_request,
            expected_access=expected_access,
            expected_binding=expected_authz_binding,
            expected_destination_self_sha256=expected_destination.destination_self_sha256,
            expected_store_expectation_self_sha256=expected_store.expectation_self_sha256,
        )
    return sha256_bytes(
        AUTHZ_RETENTION_FAILURE_DOMAIN.encode("utf-8")
        + b"\n"
        + _authorization_hash(expected_request).encode("ascii")
        + b"\n"
        + data
    )


def _validate_parent_snapshot(
    snapshot: StoreParentSnapshot, expected: ConfidentialStoreExpectation
) -> None:
    if snapshot != StoreParentSnapshot(
        relative_path=expected.records_parent_relative_path,
        object_identity=expected.records_parent_object_identity,
        namespace_object_identity=expected.namespace_object_identity,
        volume_identifier_commitment=expected.volume_identifier_commitment,
        security_descriptor_sha256=expected.records_parent_security_descriptor_sha256,
        confidential_store_profile_sha256=expected.confidential_store_profile_sha256,
    ):
        raise PrivatePermissionEvidenceContractError("records parent snapshot mismatch")


def _validate_namespace_snapshot(
    snapshot: ConfidentialNamespaceSnapshot, expected: ConfidentialStoreExpectation
) -> None:
    if (
        snapshot.namespace_relative_path != expected.namespace_relative_path
        or snapshot.namespace_object_identity != expected.namespace_object_identity
        or snapshot.volume_identifier_commitment != expected.volume_identifier_commitment
        or snapshot.confidential_store_profile_sha256
        != expected.confidential_store_profile_sha256
    ):
        raise PrivatePermissionEvidenceContractError("namespace snapshot mismatch")
    _validate_parent_snapshot(snapshot.records_parent, expected)


def _validate_object_snapshot(
    snapshot: StoreObjectSnapshot,
    destination: RetentionDestination,
    *,
    permitted_paths: set[str],
) -> None:
    if snapshot.relative_path not in permitted_paths:
        raise PrivatePermissionEvidenceContractError("store object path mismatch")
    if snapshot.namespace_object_identity != destination.namespace_object_identity:
        raise PrivatePermissionEvidenceContractError("snapshot namespace mismatch")
    if snapshot.parent_relative_path != destination.records_parent_relative_path:
        raise PrivatePermissionEvidenceContractError("snapshot parent path mismatch")
    if snapshot.parent_object_identity != destination.records_parent_object_identity:
        raise PrivatePermissionEvidenceContractError("snapshot parent identity mismatch")
    if snapshot.volume_identifier_commitment != destination.volume_identifier_commitment:
        raise PrivatePermissionEvidenceContractError("snapshot volume mismatch")
    if snapshot.security_descriptor_sha256 != destination.required_created_object_security_descriptor_sha256:
        raise PrivatePermissionEvidenceContractError("snapshot security descriptor mismatch")
    if snapshot.confidential_store_profile_sha256 != destination.confidential_store_profile_sha256:
        raise PrivatePermissionEvidenceContractError("snapshot store profile mismatch")
    _nfc(snapshot.object_identity, name="snapshot.object_identity")
    _integer(snapshot.size_bytes, name="snapshot.size_bytes")
    _sha(snapshot.file_sha256, name="snapshot.file_sha256")
    expected_state = (
        "creating"
        if snapshot.relative_path.endswith(".__creating__")
        else "published"
    )
    if snapshot.publication_state != expected_state:
        raise PrivatePermissionEvidenceContractError("snapshot publication state mismatch")


def _validate_store_effect(
    effect: StoreFailureEffect,
    destination: RetentionDestination,
) -> None:
    permitted_paths = {
        destination.primary_creating_relative_path,
        destination.primary_accepted_relative_path,
        destination.retention_failure_creating_relative_path,
        destination.retention_failure_accepted_relative_path,
    }
    operation = effect.operation
    if operation not in {
        "create_exclusive", "write_once", "flush_data_and_metadata", "rewind",
        "read", "snapshot", "publish_same_object_exclusive", "release_store_object",
    }:
        raise PrivatePermissionEvidenceContractError("unknown store effect operation")
    if effect.completion_status not in {"succeeded", "collision", "failed"}:
        raise PrivatePermissionEvidenceContractError("unknown store completion")
    if effect.effect_certainty not in {"no_effect", "reported_effect", "unknown"}:
        raise PrivatePermissionEvidenceContractError("unknown store effect certainty")
    if effect.completion_status in {"succeeded", "collision"} and effect.backend_error_hex is not None:
        raise PrivatePermissionEvidenceContractError("successful/collision effect has backend error")
    if effect.backend_error_hex is not None:
        _dword_hex(effect.backend_error_hex, name="store effect backend_error_hex")
        if effect.backend_error_hex == "0x0":
            raise PrivatePermissionEvidenceContractError("store effect error cannot be zero")
    combination = (effect.completion_status, effect.effect_certainty)
    if operation == "release_store_object":
        if any(
            value is not None
            for value in (
                effect.before_snapshot, effect.after_snapshot,
                effect.collision_relative_path, effect.bytes_requested, effect.bytes_reported,
            )
        ):
            raise PrivatePermissionEvidenceContractError("object release cannot invent snapshots")
        legal = {
            ("succeeded", "reported_effect"),
            ("failed", "no_effect"),
            ("failed", "reported_effect"),
            ("failed", "unknown"),
        }
        if combination not in legal:
            raise PrivatePermissionEvidenceContractError("illegal object release effect")
        return
    if effect.before_snapshot is not None:
        _validate_object_snapshot(effect.before_snapshot, destination, permitted_paths=permitted_paths)
    if effect.after_snapshot is not None:
        _validate_object_snapshot(effect.after_snapshot, destination, permitted_paths=permitted_paths)
    legal_combinations = {
        "create_exclusive": {
            ("succeeded", "reported_effect"), ("collision", "no_effect"),
            ("failed", "no_effect"), ("failed", "reported_effect"),
            ("failed", "unknown"),
        },
        "write_once": {
            ("succeeded", "reported_effect"), ("failed", "no_effect"),
            ("failed", "reported_effect"), ("failed", "unknown"),
        },
        "flush_data_and_metadata": {
            ("succeeded", "reported_effect"), ("failed", "no_effect"),
            ("failed", "reported_effect"), ("failed", "unknown"),
        },
        "rewind": {
            ("succeeded", "reported_effect"), ("failed", "no_effect"),
            ("failed", "reported_effect"), ("failed", "unknown"),
        },
        "read": {
            ("succeeded", "reported_effect"), ("failed", "no_effect"),
            ("failed", "reported_effect"), ("failed", "unknown"),
        },
        "snapshot": {
            ("succeeded", "reported_effect"), ("failed", "no_effect"),
            ("failed", "reported_effect"), ("failed", "unknown"),
        },
        "publish_same_object_exclusive": {
            ("succeeded", "reported_effect"), ("collision", "no_effect"),
            ("failed", "no_effect"), ("failed", "reported_effect"),
            ("failed", "unknown"),
        },
    }
    if combination not in legal_combinations[operation]:
        raise PrivatePermissionEvidenceContractError("illegal store effect status/certainty")
    creating_paths = {
        destination.primary_creating_relative_path,
        destination.retention_failure_creating_relative_path,
    }
    accepted_paths = {
        destination.primary_accepted_relative_path,
        destination.retention_failure_accepted_relative_path,
    }
    paired_paths = {
        destination.primary_creating_relative_path: destination.primary_accepted_relative_path,
        destination.retention_failure_creating_relative_path:
            destination.retention_failure_accepted_relative_path,
    }
    before = effect.before_snapshot
    after = effect.after_snapshot

    def binding(snapshot: StoreObjectSnapshot) -> tuple[object, ...]:
        return (
            snapshot.namespace_object_identity, snapshot.parent_relative_path,
            snapshot.parent_object_identity, snapshot.object_identity,
            snapshot.volume_identifier_commitment, snapshot.security_descriptor_sha256,
            snapshot.confidential_store_profile_sha256,
        )

    if before is not None and after is not None and binding(before) != binding(after):
        raise PrivatePermissionEvidenceContractError("store effect changed immutable object binding")
    if operation != "publish_same_object_exclusive" and before is not None and after is not None:
        if (
            before.relative_path != after.relative_path
            or before.publication_state != after.publication_state
        ):
            raise PrivatePermissionEvidenceContractError("non-publication effect changed path/state")

    if effect.completion_status == "collision":
        expected_collision = (
            destination.primary_creating_relative_path
            if operation == "create_exclusive" and effect.collision_relative_path
            == destination.primary_creating_relative_path
            else destination.retention_failure_creating_relative_path
            if operation == "create_exclusive" and effect.collision_relative_path
            == destination.retention_failure_creating_relative_path
            else destination.primary_accepted_relative_path
            if operation == "publish_same_object_exclusive" and effect.collision_relative_path
            == destination.primary_accepted_relative_path
            else destination.retention_failure_accepted_relative_path
            if operation == "publish_same_object_exclusive" and effect.collision_relative_path
            == destination.retention_failure_accepted_relative_path
            else None
        )
        if expected_collision is None:
            raise PrivatePermissionEvidenceContractError("collision path does not bind its slot")
    elif effect.collision_relative_path is not None:
        raise PrivatePermissionEvidenceContractError("non-collision has collision path")
    if operation not in {"write_once", "read"}:
        if effect.bytes_requested is not None or effect.bytes_reported is not None:
            raise PrivatePermissionEvidenceContractError("non-byte operation has byte counts")
    else:
        if effect.bytes_requested is not None:
            _integer(effect.bytes_requested, name="bytes_requested")
        if effect.bytes_reported is not None:
            _integer(effect.bytes_reported, name="bytes_reported")
        if (
            effect.bytes_requested is not None
            and effect.bytes_reported is not None
            and effect.bytes_reported > effect.bytes_requested
        ):
            raise PrivatePermissionEvidenceContractError("reported bytes exceed request")
    if effect.effect_certainty == "unknown" and after is not None:
        raise PrivatePermissionEvidenceContractError("unknown effect has no after snapshot")

    empty_hash = sha256_bytes(b"")
    if operation == "create_exclusive":
        if combination in {("succeeded", "reported_effect"), ("failed", "reported_effect")}:
            if (
                before is not None or after is None
                or after.relative_path not in creating_paths
                or after.publication_state != "creating"
                or after.size_bytes != 0 or after.file_sha256 != empty_hash
            ):
                raise PrivatePermissionEvidenceContractError("reported create effect shape mismatch")
        elif before is not None or after is not None:
            raise PrivatePermissionEvidenceContractError("unreported create invented a snapshot")
        return

    if operation == "write_once":
        if (
            before is None or before.relative_path not in creating_paths
            or before.publication_state != "creating" or before.size_bytes != 0
            or before.file_sha256 != empty_hash or effect.bytes_requested is None
        ):
            raise PrivatePermissionEvidenceContractError("write before-state/count mismatch")
        requested = effect.bytes_requested
        reported = effect.bytes_reported
        if combination == ("succeeded", "reported_effect"):
            if reported != requested or after is None or after.size_bytes != reported:
                raise PrivatePermissionEvidenceContractError("successful write count/state mismatch")
        elif combination == ("failed", "no_effect"):
            if after != before or reported not in {None, 0}:
                raise PrivatePermissionEvidenceContractError("no-effect write mismatch")
        elif combination == ("failed", "reported_effect"):
            if reported is None or after is None or after.size_bytes != reported:
                raise PrivatePermissionEvidenceContractError("reported write effect mismatch")
        elif after is not None:
            raise PrivatePermissionEvidenceContractError("unknown write invented after-state")
        return

    if operation in {"flush_data_and_metadata", "rewind", "read", "snapshot"}:
        if before is None:
            raise PrivatePermissionEvidenceContractError(f"{operation} lacks last confirmed state")
        if combination in {
            ("succeeded", "reported_effect"), ("failed", "no_effect"),
            ("failed", "reported_effect"),
        } and after != before:
            raise PrivatePermissionEvidenceContractError(f"{operation} changed physical state")
        if combination == ("failed", "unknown") and after is not None:
            raise PrivatePermissionEvidenceContractError(f"unknown {operation} has after-state")
        if operation == "read":
            if effect.bytes_requested is None:
                raise PrivatePermissionEvidenceContractError("read lacks requested byte count")
            if combination == ("succeeded", "reported_effect") and effect.bytes_reported is None:
                raise PrivatePermissionEvidenceContractError("successful read lacks reported count")
            if combination == ("failed", "no_effect") and effect.bytes_reported not in {None, 0}:
                raise PrivatePermissionEvidenceContractError("no-effect read reported bytes")
        return

    if operation == "publish_same_object_exclusive":
        if before is None or before.relative_path not in creating_paths or before.publication_state != "creating":
            raise PrivatePermissionEvidenceContractError("publication lacks creating before-state")
        final_path = paired_paths[before.relative_path]
        if combination in {
            ("succeeded", "reported_effect"), ("failed", "reported_effect")
        }:
            if (
                after is None or after.relative_path != final_path
                or after.publication_state != "published"
                or before.size_bytes != after.size_bytes
                or before.file_sha256 != after.file_sha256
            ):
                raise PrivatePermissionEvidenceContractError("reported publication transition mismatch")
        elif combination in {
            ("collision", "no_effect"), ("failed", "no_effect")
        }:
            if after != before:
                raise PrivatePermissionEvidenceContractError("no-effect publication changed state")
            if combination == ("collision", "no_effect") and effect.collision_relative_path != final_path:
                raise PrivatePermissionEvidenceContractError("publication collision target mismatch")
        elif after is not None:
            raise PrivatePermissionEvidenceContractError("unknown publication invented after-state")


def _validate_namespace_effect(
    effect: StoreNamespaceEffect,
    expected_store: ConfidentialStoreExpectation,
) -> None:
    if effect.operation not in {"open_namespace", "release_namespace"}:
        raise PrivatePermissionEvidenceContractError("unknown namespace effect")
    if effect.completion_status not in {"succeeded", "failed"}:
        raise PrivatePermissionEvidenceContractError("unknown namespace completion")
    if effect.effect_certainty not in {"no_effect", "reported_effect", "unknown"}:
        raise PrivatePermissionEvidenceContractError("unknown namespace certainty")
    if effect.before_snapshot is not None:
        _validate_namespace_snapshot(effect.before_snapshot, expected_store)
    if effect.after_snapshot is not None:
        _validate_namespace_snapshot(effect.after_snapshot, expected_store)
    if effect.completion_status == "succeeded" and effect.backend_error_hex is not None:
        raise PrivatePermissionEvidenceContractError("namespace success has backend error")
    if effect.backend_error_hex is not None:
        _dword_hex(effect.backend_error_hex, name="namespace backend_error_hex")
        if effect.backend_error_hex == "0x0":
            raise PrivatePermissionEvidenceContractError("namespace error cannot be zero")
    if effect.operation == "open_namespace":
        legal = {
            ("succeeded", "reported_effect", False, True),
            ("failed", "no_effect", False, False),
            ("failed", "reported_effect", False, True),
            ("failed", "unknown", False, False),
        }
    else:
        legal = {
            ("succeeded", "reported_effect", True, False),
            ("failed", "no_effect", True, True),
            ("failed", "reported_effect", True, False),
            ("failed", "unknown", True, False),
        }
    shape = (
        effect.completion_status, effect.effect_certainty,
        effect.before_snapshot is not None, effect.after_snapshot is not None,
    )
    if shape not in legal:
        raise PrivatePermissionEvidenceContractError("illegal namespace effect shape")
    if (
        effect.operation == "release_namespace"
        and effect.effect_certainty == "no_effect"
        and effect.before_snapshot != effect.after_snapshot
    ):
        raise PrivatePermissionEvidenceContractError("no-effect namespace release changed snapshot")


def _validate_terminal_state(
    value: object,
    destination: RetentionDestination,
    *,
    expected_payload_bytes: int | None = None,
    expected_payload_sha256: str | None = None,
    expected_payload_data: bytes | None = None,
) -> StoreObjectTerminalState:
    if not isinstance(value, StoreObjectTerminalState):
        record = _mapping(value, name="store object terminal state")
        try:
            value = StoreObjectTerminalState(**record)  # type: ignore[arg-type]
        except TypeError as exc:
            raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    state = value
    if state.publication_state not in _PUBLICATION_STATES:
        raise PrivatePermissionEvidenceContractError("unknown publication state")
    if type(destination) is PermissionEvidenceDestination and state.slot_kind == "primary":
        expected_kind: str = destination.payload_kind
        creating = destination.primary_creating_relative_path
        accepted = destination.primary_accepted_relative_path
    elif (
        type(destination) is PermissionEvidenceDestination
        and state.slot_kind == "retention_failure"
    ):
        expected_kind = "retention_failure"
        creating = destination.retention_failure_creating_relative_path
        accepted = destination.retention_failure_accepted_relative_path
    elif (
        type(destination) is PermissionAuthzEvidenceDestination
        and state.slot_kind == "authz_primary"
    ):
        expected_kind = "authz_evidence"
        creating = destination.primary_creating_relative_path
        accepted = destination.primary_accepted_relative_path
    elif (
        type(destination) is PermissionAuthzEvidenceDestination
        and state.slot_kind == "authz_retention_failure"
    ):
        expected_kind = "authz_retention_failure"
        creating = destination.retention_failure_creating_relative_path
        accepted = destination.retention_failure_accepted_relative_path
    else:
        raise PrivatePermissionEvidenceContractError("unknown terminal slot kind")
    if (
        state.payload_kind != expected_kind
        or state.creating_relative_path != creating
        or state.accepted_relative_path != accepted
    ):
        raise PrivatePermissionEvidenceContractError("terminal slot binding mismatch")
    if expected_payload_bytes is not None:
        _integer(expected_payload_bytes, name="expected payload bytes")
    if expected_payload_sha256 is not None:
        _sha(expected_payload_sha256, name="expected payload sha256")
    if expected_payload_data is not None:
        if type(expected_payload_data) is not bytes:
            raise PrivatePermissionEvidenceContractError("expected payload data must be immutable bytes")
        if (
            expected_payload_bytes not in {None, len(expected_payload_data)}
            or expected_payload_sha256 not in {None, sha256_bytes(expected_payload_data)}
        ):
            raise PrivatePermissionEvidenceContractError("expected payload inputs disagree")
        expected_payload_bytes = len(expected_payload_data)
        expected_payload_sha256 = sha256_bytes(expected_payload_data)
    evidence = (
        state.object_identity,
        state.volume_identifier_commitment,
        state.last_confirmed_size_bytes,
        state.last_confirmed_file_sha256,
    )
    known_rows = {
        "creating_empty", "creating_partial", "creating_complete_unverified",
        "creating_verified", "published_unverified", "published_verified_unaccepted",
    }
    if state.publication_state in known_rows:
        if (
            state.ownership != "owned_by_this_invocation"
            or state.effect_certainty != "reported_effect"
            or any(item is None for item in evidence)
            or state.handle_state not in {"released", "release_failed", "unknown"}
        ):
            raise PrivatePermissionEvidenceContractError("known owned terminal row mismatch")
        expected_path = accepted if state.publication_state.startswith("published") else creating
        if state.last_confirmed_relative_path != expected_path or state.reported_relative_path != expected_path:
            raise PrivatePermissionEvidenceContractError("known terminal path mismatch")
        _nfc(state.object_identity, name="terminal object identity")
        if state.volume_identifier_commitment != destination.volume_identifier_commitment:
            raise PrivatePermissionEvidenceContractError("known terminal volume mismatch")
        _integer(state.last_confirmed_size_bytes, name="terminal size")
        _sha(state.last_confirmed_file_sha256, name="terminal file hash")
        if state.publication_state == "creating_empty":
            if (
                state.last_confirmed_size_bytes != 0
                or state.last_confirmed_file_sha256 != sha256_bytes(b"")
            ):
                raise PrivatePermissionEvidenceContractError("creating-empty evidence mismatch")
        elif state.publication_state == "creating_partial":
            if state.last_confirmed_size_bytes <= 0 or (
                expected_payload_bytes is not None
                and state.last_confirmed_size_bytes >= expected_payload_bytes
            ):
                raise PrivatePermissionEvidenceContractError("creating-partial size mismatch")
            if (
                expected_payload_data is not None
                and state.last_confirmed_file_sha256
                != sha256_bytes(expected_payload_data[: state.last_confirmed_size_bytes])
            ):
                raise PrivatePermissionEvidenceContractError("creating-partial hash mismatch")
        elif state.publication_state in {
            "creating_complete_unverified", "creating_verified",
            "published_unverified", "published_verified_unaccepted",
        }:
            if (
                expected_payload_bytes is not None
                and state.last_confirmed_size_bytes != expected_payload_bytes
            ) or (
                expected_payload_sha256 is not None
                and state.last_confirmed_file_sha256 != expected_payload_sha256
            ):
                raise PrivatePermissionEvidenceContractError("complete terminal payload mismatch")
    elif state.publication_state == "not_created":
        if (
            state.ownership != "none" or state.effect_certainty != "no_effect"
            or any(item is not None for item in evidence)
            or state.last_confirmed_relative_path is not None
            or state.reported_relative_path is not None
            or state.handle_state != "not_acquired"
        ):
            raise PrivatePermissionEvidenceContractError("not-created terminal row mismatch")
    elif state.publication_state == "collision_unowned":
        if (
            state.ownership != "unowned_collision" or state.effect_certainty != "no_effect"
            or any(item is not None for item in evidence)
            or state.last_confirmed_relative_path is not None
            or state.reported_relative_path is not None
            or state.handle_state != "not_acquired"
        ):
            raise PrivatePermissionEvidenceContractError("collision terminal row mismatch")
    else:
        if state.effect_certainty != "unknown":
            raise PrivatePermissionEvidenceContractError("effect-unknown certainty mismatch")
        if state.ownership == "owned_by_this_invocation":
            all_null = all(item is None for item in evidence)
            all_known = all(item is not None for item in evidence)
            if not (all_null or all_known):
                raise PrivatePermissionEvidenceContractError("unknown terminal evidence is partial")
            if all_null and state.last_confirmed_relative_path is not None:
                raise PrivatePermissionEvidenceContractError("unknown no-snapshot row has path")
            if all_known:
                if state.last_confirmed_relative_path not in {creating, accepted}:
                    raise PrivatePermissionEvidenceContractError("unknown terminal path mismatch")
                _nfc(state.object_identity, name="unknown terminal object identity")
                if state.volume_identifier_commitment != destination.volume_identifier_commitment:
                    raise PrivatePermissionEvidenceContractError("unknown terminal volume mismatch")
                _integer(state.last_confirmed_size_bytes, name="unknown terminal size")
                _sha(state.last_confirmed_file_sha256, name="unknown terminal hash")
            if state.reported_relative_path is not None:
                raise PrivatePermissionEvidenceContractError("unknown effect cannot report path")
            if state.handle_state not in {"released", "release_failed", "unknown"}:
                raise PrivatePermissionEvidenceContractError("owned unknown handle state mismatch")
        elif state.ownership == "unknown":
            if any(item is not None for item in evidence) or state.handle_state not in {"not_acquired", "unknown"}:
                raise PrivatePermissionEvidenceContractError("unknown-ownership row mismatch")
        else:
            raise PrivatePermissionEvidenceContractError("effect-unknown ownership mismatch")
    return state


def _coerce_object_snapshot(value: object) -> StoreObjectSnapshot | None:
    if value is None or isinstance(value, StoreObjectSnapshot):
        return value
    record = _mapping(value, name="store object snapshot")
    try:
        return StoreObjectSnapshot(**record)  # type: ignore[arg-type]
    except TypeError as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc


def _coerce_namespace_snapshot(value: object) -> ConfidentialNamespaceSnapshot | None:
    if value is None or isinstance(value, ConfidentialNamespaceSnapshot):
        return value
    record = dict(_mapping(value, name="namespace snapshot"))
    parent_value = record.get("records_parent")
    if isinstance(parent_value, StoreParentSnapshot):
        parent = parent_value
    else:
        try:
            parent = StoreParentSnapshot(
                **_mapping(parent_value, name="records parent snapshot")
            )  # type: ignore[arg-type]
        except TypeError as exc:
            raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    record["records_parent"] = parent
    try:
        return ConfidentialNamespaceSnapshot(**record)  # type: ignore[arg-type]
    except TypeError as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc


def _validate_effect_array(
    value: object,
    destination: RetentionDestination,
    expected_store: ConfidentialStoreExpectation,
) -> tuple[StoreNamespaceEffect | StoreFailureEffect, ...]:
    if not isinstance(value, (list, tuple)):
        raise PrivatePermissionEvidenceContractError("effects must be an ordered array")
    validated: list[StoreNamespaceEffect | StoreFailureEffect] = []
    for effect in value:
        if isinstance(effect, StoreNamespaceEffect):
            _validate_namespace_effect(effect, expected_store)
            validated.append(effect)
            continue
        if isinstance(effect, StoreFailureEffect):
            _validate_store_effect(effect, destination)
            validated.append(effect)
            continue
        record = _mapping(effect, name="effect")
        if record.get("operation") in {"open_namespace", "release_namespace"}:
            namespace_record = dict(record)
            namespace_record["before_snapshot"] = _coerce_namespace_snapshot(
                namespace_record.get("before_snapshot")
            )
            namespace_record["after_snapshot"] = _coerce_namespace_snapshot(
                namespace_record.get("after_snapshot")
            )
            try:
                namespace_effect = StoreNamespaceEffect(**namespace_record)  # type: ignore[arg-type]
            except TypeError as exc:
                raise PrivatePermissionEvidenceContractError(str(exc)) from exc
            _validate_namespace_effect(namespace_effect, expected_store)
            validated.append(namespace_effect)
            continue
        object_record = dict(record)
        object_record["before_snapshot"] = _coerce_object_snapshot(
            object_record.get("before_snapshot")
        )
        object_record["after_snapshot"] = _coerce_object_snapshot(
            object_record.get("after_snapshot")
        )
        try:
            object_effect = StoreFailureEffect(**object_record)  # type: ignore[arg-type]
        except TypeError as exc:
            raise PrivatePermissionEvidenceContractError(str(exc)) from exc
        _validate_store_effect(object_effect, destination)
        validated.append(object_effect)
    return tuple(validated)


def _validate_effect_terminal_fold(
    effects_value: object,
    state_value: object,
    *,
    destination: RetentionDestination,
    expected_store: ConfidentialStoreExpectation,
    expected_payload_bytes: int | None = None,
    expected_payload_sha256: str | None = None,
    expected_payload_data: bytes | None = None,
) -> StoreObjectTerminalState:
    effects = _validate_effect_array(effects_value, destination, expected_store)
    state = _validate_terminal_state(
        state_value, destination,
        expected_payload_bytes=expected_payload_bytes,
        expected_payload_sha256=expected_payload_sha256,
        expected_payload_data=expected_payload_data,
    )
    payload_data = expected_payload_data
    payload_length = expected_payload_bytes
    payload_sha256 = expected_payload_sha256
    if payload_data is not None:
        payload_length = len(payload_data)
        payload_sha256 = sha256_bytes(payload_data)
    namespace_open_indices = [
        index for index, effect in enumerate(effects)
        if isinstance(effect, StoreNamespaceEffect) and effect.operation == "open_namespace"
    ]
    namespace_release_indices = [
        index for index, effect in enumerate(effects)
        if isinstance(effect, StoreNamespaceEffect) and effect.operation == "release_namespace"
    ]
    if len(namespace_open_indices) > 1 or len(namespace_release_indices) > 1:
        raise PrivatePermissionEvidenceContractError("namespace effect repeated")
    if namespace_open_indices and namespace_open_indices[0] != 0:
        raise PrivatePermissionEvidenceContractError("namespace open is not first effect")
    if namespace_release_indices and namespace_release_indices[0] != len(effects) - 1:
        raise PrivatePermissionEvidenceContractError("namespace release is not final effect")
    if namespace_release_indices and not namespace_open_indices:
        raise PrivatePermissionEvidenceContractError("namespace release lacks open attempt")
    namespace_open = (
        effects[namespace_open_indices[0]] if namespace_open_indices else None
    )
    namespace_release = (
        effects[namespace_release_indices[0]] if namespace_release_indices else None
    )
    namespace_handle_reported = (
        isinstance(namespace_open, StoreNamespaceEffect)
        and namespace_open.effect_certainty == "reported_effect"
        and namespace_open.after_snapshot is not None
    )
    if namespace_handle_reported and not namespace_release_indices:
        raise PrivatePermissionEvidenceContractError(
            "reported namespace handle was not released"
        )
    if namespace_release is not None:
        if (
            not isinstance(namespace_release, StoreNamespaceEffect)
            or not isinstance(namespace_open, StoreNamespaceEffect)
            or not namespace_handle_reported
        ):
            raise PrivatePermissionEvidenceContractError(
                "namespace release follows an outcome with no returned handle"
            )
        if namespace_release.before_snapshot != namespace_open.after_snapshot:
            raise PrivatePermissionEvidenceContractError(
                "namespace release does not close the opened namespace snapshot"
            )
    object_effects = [effect for effect in effects if isinstance(effect, StoreFailureEffect)]
    if object_effects and (
        not isinstance(namespace_open, StoreNamespaceEffect)
        or namespace_open.completion_status != "succeeded"
        or namespace_open.effect_certainty != "reported_effect"
    ):
        raise PrivatePermissionEvidenceContractError(
            "object effect lacks a successfully opened namespace"
        )
    create_indices = [
        index for index, effect in enumerate(object_effects)
        if effect.operation == "create_exclusive"
    ]
    release_indices = [
        index for index, effect in enumerate(object_effects)
        if effect.operation == "release_store_object"
    ]
    if len(create_indices) > 1 or len(release_indices) > 1:
        raise PrivatePermissionEvidenceContractError("create/object release effect repeated")
    if object_effects and (not create_indices or create_indices[0] != 0):
        raise PrivatePermissionEvidenceContractError("object effects do not begin with create")
    if release_indices and release_indices[0] != len(object_effects) - 1:
        raise PrivatePermissionEvidenceContractError("object release is not last object effect")
    operational_effects = [
        effect for effect in object_effects if effect.operation != "release_store_object"
    ]
    exact_operation_order = [
        "create_exclusive", "write_once", "flush_data_and_metadata", "rewind",
        "read", "read", "snapshot", "publish_same_object_exclusive", "snapshot",
        "rewind", "read", "read",
    ]
    if [effect.operation for effect in operational_effects] != exact_operation_order[
        : len(operational_effects)
    ]:
        raise PrivatePermissionEvidenceContractError("store effects violate the exact operation order")
    terminal_effect_indices = [
        index for index, effect in enumerate(operational_effects)
        if effect.completion_status in {"failed", "collision"}
    ]
    if terminal_effect_indices and terminal_effect_indices != [len(operational_effects) - 1]:
        raise PrivatePermissionEvidenceContractError(
            "sensitive store effects continue after a failed/collision outcome"
        )
    if not object_effects:
        folded = _initial_state(
            destination, slot_kind=state.slot_kind, payload_kind=state.payload_kind
        )
    else:
        if payload_length is None:
            writes = [
                effect for effect in object_effects if effect.operation == "write_once"
            ]
            payload_length = (
                writes[0].bytes_requested
                if len(writes) == 1 and writes[0].bytes_requested is not None
                else 0
            )
        folded = _initial_state(
            destination, slot_kind=state.slot_kind, payload_kind=state.payload_kind
        )
        last_snapshot: StoreObjectSnapshot | None = None
        handle_owned = False
        conflicting_snapshot_indices: list[int] = []
        for effect_index, effect in enumerate(object_effects):
            if effect.operation == "release_store_object":
                handle_state = {
                    ("succeeded", "reported_effect"): "released",
                    ("failed", "no_effect"): "release_failed",
                    ("failed", "reported_effect"): "released",
                    ("failed", "unknown"): "unknown",
                }[(effect.completion_status, effect.effect_certainty)]
                folded = _with_handle_state(folded, handle_state)
                continue
            if effect.operation != "create_exclusive":
                if not handle_owned:
                    raise PrivatePermissionEvidenceContractError(
                        "store operation occurs without caller-owned object handle"
                    )
                if effect.operation == "write_once":
                    if (
                        payload_length is not None
                        and effect.bytes_requested != payload_length
                    ):
                        raise PrivatePermissionEvidenceContractError(
                            "write effect does not bind the expected payload length"
                        )
                    if (
                        effect.effect_certainty == "reported_effect"
                        and effect.after_snapshot is not None
                        and effect.bytes_reported is not None
                    ):
                        reported = effect.bytes_reported
                        expected_reported_hash = None
                        if payload_data is not None:
                            expected_reported_hash = sha256_bytes(payload_data[:reported])
                        elif reported == payload_length:
                            expected_reported_hash = payload_sha256
                        if (
                            expected_reported_hash is not None
                            and effect.after_snapshot.file_sha256
                            != expected_reported_hash
                        ):
                            raise PrivatePermissionEvidenceContractError(
                                "write effect does not bind the expected payload bytes"
                            )
                snapshot_payload_conflict = (
                    effect.operation == "snapshot"
                    and effect.completion_status == "succeeded"
                    and effect.effect_certainty == "reported_effect"
                    and effect.after_snapshot is not None
                    and (
                        (
                            payload_length is not None
                            and effect.after_snapshot.size_bytes != payload_length
                        )
                        or (
                            payload_sha256 is not None
                            and effect.after_snapshot.file_sha256 != payload_sha256
                        )
                    )
                )
                snapshot_conflict = (
                    effect.operation == "snapshot"
                    and effect.completion_status == "succeeded"
                    and effect.effect_certainty == "reported_effect"
                    and effect.after_snapshot is not None
                    and (
                        _snapshot_disagrees_with_owned_state(
                            folded, effect.after_snapshot
                        )
                        or snapshot_payload_conflict
                    )
                )
                if effect.before_snapshot != last_snapshot and not snapshot_conflict:
                    raise PrivatePermissionEvidenceContractError(
                        "store effect before-state is not the last confirmed snapshot"
                    )
                if snapshot_conflict:
                    conflicting_snapshot_indices.append(effect_index)
            else:
                snapshot_conflict = False
                handle_owned = (
                    effect.effect_certainty == "reported_effect"
                    or (
                        effect.effect_certainty == "unknown"
                        and state.ownership == "owned_by_this_invocation"
                    )
                )
            folded = _fold_snapshot_effect(
                folded, effect, payload_length=payload_length,
                owned_handle=handle_owned,
            )
            if snapshot_conflict and folded.effect_certainty != "unknown":
                folded = _unknown_state(folded, owned_handle=handle_owned)
            if effect.after_snapshot is not None and not snapshot_conflict:
                last_snapshot = effect.after_snapshot
            elif effect.effect_certainty == "unknown":
                last_snapshot = _last_snapshot_from_state(folded, destination)
        operational_count = len(operational_effects)
        if conflicting_snapshot_indices and conflicting_snapshot_indices != [
            operational_count - 1
        ]:
            raise PrivatePermissionEvidenceContractError(
                "conflicting snapshot is not the final operational effect"
            )
        if handle_owned and not release_indices:
            raise PrivatePermissionEvidenceContractError("owned store handle was not released")
    physical_fields = (
        "slot_kind", "payload_kind", "ownership", "effect_certainty",
        "creating_relative_path", "accepted_relative_path", "last_confirmed_relative_path",
        "reported_relative_path", "object_identity", "volume_identifier_commitment",
        "last_confirmed_size_bytes", "last_confirmed_file_sha256", "handle_state",
    )
    if any(getattr(folded, field) != getattr(state, field) for field in physical_fields):
        raise PrivatePermissionEvidenceContractError(
            "terminal object state is not the left-to-right effect fold"
        )
    folded_phase = folded.publication_state
    legal_phase = {folded_phase}
    if folded_phase == "creating_complete_unverified":
        legal_phase.add("creating_verified")
    if folded_phase == "published_unverified":
        legal_phase.add("published_verified_unaccepted")
    if state.publication_state not in legal_phase:
        raise PrivatePermissionEvidenceContractError("terminal publication phase/effect mismatch")
    operations = [effect.operation for effect in object_effects]
    if state.publication_state == "creating_verified":
        snapshot_positions = [
            index for index, effect in enumerate(object_effects)
            if effect.operation == "snapshot" and effect.completion_status == "succeeded"
            and effect.after_snapshot is not None
            and effect.after_snapshot.publication_state == "creating"
        ]
        if not snapshot_positions:
            raise PrivatePermissionEvidenceContractError("creating-verified lacks verified snapshot")
        before_snapshot = object_effects[:snapshot_positions[-1]]
        if sum(effect.operation == "read" for effect in before_snapshot) < 2:
            raise PrivatePermissionEvidenceContractError("creating-verified lacks full read/EOF")
    if state.publication_state == "published_verified_unaccepted":
        publish_positions = [
            index for index, effect in enumerate(object_effects)
            if effect.operation == "publish_same_object_exclusive"
            and effect.effect_certainty == "reported_effect"
        ]
        if not publish_positions:
            raise PrivatePermissionEvidenceContractError("published-verified lacks publication")
        after_publish = object_effects[publish_positions[-1] + 1:]
        if (
            "snapshot" not in [effect.operation for effect in after_publish]
            or "rewind" not in [effect.operation for effect in after_publish]
            or sum(effect.operation == "read" for effect in after_publish) < 2
        ):
            raise PrivatePermissionEvidenceContractError(
                "published-verified lacks final snapshot/readback/EOF"
            )
    return state


def _validate_effect_failure_equations(
    effects_value: object,
    primary_value: object,
    follow_values: object,
    *,
    destination: RetentionDestination,
    expected_store: ConfidentialStoreExpectation,
) -> None:
    effects = _validate_effect_array(effects_value, destination, expected_store)
    primary = _validate_failure_item(primary_value, name="effect failure primary")
    if not isinstance(follow_values, (list, tuple)):
        raise PrivatePermissionEvidenceContractError("effect failure follow-ons must be an array")
    follows = [
        _validate_failure_item(value, name=f"effect failure follow-on[{index}]")
        for index, value in enumerate(follow_values)
    ]
    vector = [primary, *follows]
    object_effects = [
        effect for effect in effects if isinstance(effect, StoreFailureEffect)
    ]
    last_snapshot: StoreObjectSnapshot | None = None
    conflicting_snapshots: list[
        tuple[int, StoreFailureEffect, StoreObjectSnapshot]
    ] = []
    for index, effect in enumerate(object_effects):
        if effect.operation == "release_store_object":
            continue
        if effect.operation == "create_exclusive":
            if effect.after_snapshot is not None:
                last_snapshot = effect.after_snapshot
            continue
        is_conflicting_snapshot = (
            effect.operation == "snapshot"
            and effect.completion_status == "succeeded"
            and effect.effect_certainty == "reported_effect"
            and effect.after_snapshot is not None
            and effect.before_snapshot != last_snapshot
        )
        if is_conflicting_snapshot:
            if last_snapshot is None:
                raise PrivatePermissionEvidenceContractError(
                    "conflicting snapshot lacks prior confirmed state"
                )
            conflicting_snapshots.append((index, effect, last_snapshot))
            continue
        if effect.after_snapshot is not None:
            last_snapshot = effect.after_snapshot

    logical_snapshot_failure = (
        primary["component_operation"] == "validate_snapshot"
        and primary["reason_code"] in {"snapshot_mismatch", "postpublish_mismatch"}
    )
    if conflicting_snapshots:
        if len(conflicting_snapshots) != 1:
            raise PrivatePermissionEvidenceContractError(
                "multiple conflicting snapshot outcomes"
            )
        index, conflict, prior = conflicting_snapshots[0]
        operational_count = sum(
            effect.operation != "release_store_object" for effect in object_effects
        )
        if index != operational_count - 1:
            raise PrivatePermissionEvidenceContractError(
                "operational I/O follows a conflicting snapshot"
            )
        reported = conflict.after_snapshot
        if reported is None or (
            reported.relative_path != prior.relative_path
            or reported.publication_state != prior.publication_state
        ):
            raise PrivatePermissionEvidenceContractError(
                "conflicting snapshot phase/slot mismatch"
            )
        if reported.relative_path in {
            destination.primary_creating_relative_path,
            destination.retention_failure_creating_relative_path,
        }:
            expected_stage = "prepublication_snapshot"
            expected_reason = "snapshot_mismatch"
        else:
            expected_stage = "postpublication_snapshot"
            expected_reason = "postpublish_mismatch"
        expected_role = _slot_resource_role(
            (
                "authz_primary"
                if type(destination) is PermissionAuthzEvidenceDestination
                else "primary"
            )
            if reported.relative_path in {
                destination.primary_creating_relative_path,
                destination.primary_accepted_relative_path,
            }
            else (
                "authz_retention_failure"
                if type(destination) is PermissionAuthzEvidenceDestination
                else "retention_failure"
            )
        )
        expected_conflict = {
            "failure_kind": "validation_failure",
            "stage": expected_stage,
            "component_operation": "validate_snapshot",
            "resource_kind": "store_object_handle",
            "resource_role": expected_role,
            "resource_ordinal": None,
            "operation_ordinal": None,
            "backend_error_hex": None,
            "reason_code": expected_reason,
        }
        if any(primary[key] != value for key, value in expected_conflict.items()):
            raise PrivatePermissionEvidenceContractError(
                "conflicting snapshot does not match the primary failure"
            )
        matching_failures = [
            item for item in vector
            if item["component_operation"] == "validate_snapshot"
            and item["reason_code"] in {"snapshot_mismatch", "postpublish_mismatch"}
        ]
        if len(matching_failures) != 1:
            raise PrivatePermissionEvidenceContractError(
                "conflicting snapshot failure accounting mismatch"
            )
    elif logical_snapshot_failure:
        raise PrivatePermissionEvidenceContractError(
            "snapshot mismatch failure lacks the conflicting reported snapshot"
        )

    failed_operational = [
        effect for effect in effects
        if (
            isinstance(effect, StoreFailureEffect)
            and effect.operation != "release_store_object"
            and effect.completion_status in {"failed", "collision"}
        ) or (
            isinstance(effect, StoreNamespaceEffect)
            and effect.operation == "open_namespace"
            and effect.completion_status == "failed"
        )
    ]
    if failed_operational:
        effect = failed_operational[-1]
        allowed_operations = {effect.operation}
        if primary["component_operation"] not in allowed_operations:
            raise PrivatePermissionEvidenceContractError(
                "failed operational effect does not match the primary failure"
            )
        if isinstance(effect, StoreFailureEffect) and effect.operation != "create_exclusive":
            before = effect.before_snapshot
            if before is None:
                raise PrivatePermissionEvidenceContractError(
                    "failed object operation lacks prior confirmed state"
                )
            published = before.publication_state == "published"
            expected_stage = {
                "write_once": "payload_write",
                "flush_data_and_metadata": "payload_flush",
                "rewind": (
                    "postpublication_readback" if published else "payload_rewind"
                ),
                "read": (
                    "postpublication_readback" if published else "payload_readback"
                ),
                "snapshot": (
                    "postpublication_snapshot"
                    if published else "prepublication_snapshot"
                ),
                "publish_same_object_exclusive": "publication",
            }[effect.operation]
            expected_role = _slot_resource_role(
                (
                    "authz_primary"
                    if type(destination) is PermissionAuthzEvidenceDestination
                    else "primary"
                )
                if before.relative_path in {
                    destination.primary_creating_relative_path,
                    destination.primary_accepted_relative_path,
                }
                else (
                    "authz_retention_failure"
                    if type(destination) is PermissionAuthzEvidenceDestination
                    else "retention_failure"
                )
            )
            expected_kind = (
                "store_collision"
                if effect.completion_status == "collision"
                else "store_operation_failure"
            )
            if (
                primary["failure_kind"] != expected_kind
                or primary["stage"] != expected_stage
                or primary["resource_kind"] != "store_object_handle"
                or primary["resource_role"] != expected_role
                or primary["resource_ordinal"] is not None
                or primary["operation_ordinal"] is not None
                or primary["backend_error_hex"] != effect.backend_error_hex
            ):
                raise PrivatePermissionEvidenceContractError(
                    "failed object effect/failure response binding mismatch"
                )
        if effect.completion_status == "collision" and primary["failure_kind"] != "store_collision":
            raise PrivatePermissionEvidenceContractError("collision effect lacks collision failure")
        if effect.effect_certainty == "unknown" and primary["reason_code"] not in {
            "effect_unknown", "store_protocol_failure"
        }:
            raise PrivatePermissionEvidenceContractError("unknown effect lacks uncertainty failure")
        if effect.operation == "snapshot":
            before = effect.before_snapshot
            if before is None:
                raise PrivatePermissionEvidenceContractError(
                    "failed snapshot lacks prior confirmed state"
                )
            if before.relative_path in {
                destination.primary_creating_relative_path,
                destination.retention_failure_creating_relative_path,
            }:
                expected_stage = "prepublication_snapshot"
            else:
                expected_stage = "postpublication_snapshot"
            expected_role = _slot_resource_role(
                (
                    "authz_primary"
                    if type(destination) is PermissionAuthzEvidenceDestination
                    else "primary"
                )
                if before.relative_path in {
                    destination.primary_creating_relative_path,
                    destination.primary_accepted_relative_path,
                }
                else (
                    "authz_retention_failure"
                    if type(destination) is PermissionAuthzEvidenceDestination
                    else "retention_failure"
                )
            )
            if (
                primary["failure_kind"] != "store_operation_failure"
                or primary["stage"] != expected_stage
                or primary["component_operation"] != "snapshot"
                or primary["resource_kind"] != "store_object_handle"
                or primary["resource_role"] != expected_role
                or primary["resource_ordinal"] is not None
                or primary["operation_ordinal"] is not None
                or primary["backend_error_hex"] != effect.backend_error_hex
            ):
                raise PrivatePermissionEvidenceContractError(
                    "failed snapshot effect/failure binding mismatch"
                )
            if effect.effect_certainty == "unknown":
                permitted_reasons = {"effect_unknown", "store_protocol_failure"}
            else:
                permitted_reasons = {"snapshot_failed", "store_protocol_failure"}
            if primary["reason_code"] not in permitted_reasons:
                raise PrivatePermissionEvidenceContractError(
                    "failed snapshot certainty/reason mismatch"
                )
    for operation, failure_operation in (
        ("release_store_object", "release_store_object"),
        ("release_namespace", "release_namespace"),
    ):
        matching_effects = [
            effect for effect in effects
            if effect.operation == operation and effect.completion_status == "failed"
        ]
        matching_failures = [
            item for item in vector
            if item["failure_kind"] == "cleanup_failure"
            and item["component_operation"] == failure_operation
        ]
        if len(matching_effects) != len(matching_failures):
            raise PrivatePermissionEvidenceContractError(
                f"{operation} effect/cleanup-failure accounting mismatch"
            )


def _validate_failure_draft(
    draft: RetentionFailureDraft,
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: RetentionDestination,
    allow_prefinish: bool = False,
    expected_payload_bytes: int | None = None,
    expected_payload_sha256: str | None = None,
    expected_payload_data: bytes | None = None,
) -> None:
    record = _exact(
        draft,
        {
            "failed_primary_payload_kind", "request_self_sha256",
            "access_expectation_self_sha256", "destination_self_sha256",
            "store_expectation_self_sha256", "primary_failure",
            "ordered_follow_on_failures", "main_authority_accounting",
            "payload_effects", "payload_object_state", "acceptance_emitted",
        },
        name="retention failure draft",
    )
    _validate_retention_destination(
        expected_destination,
        expected_request=expected_request,
        expected_store=expected_store,
    )
    expected = {
        "failed_primary_payload_kind": expected_destination.payload_kind,
        "request_self_sha256": _request_hash(expected_request),
        "access_expectation_self_sha256": _access_hash(expected_access),
        "destination_self_sha256": expected_destination.destination_self_sha256,
        "store_expectation_self_sha256": expected_store.expectation_self_sha256,
        "acceptance_emitted": False,
    }
    for key, wanted in expected.items():
        if record[key] != wanted:
            raise PrivatePermissionEvidenceContractError(f"failure draft {key} mismatch")
    _validate_failure_accounting_equations(
        record["primary_failure"], record["ordered_follow_on_failures"],
        record["main_authority_accounting"], name="retention failure draft",
        allow_prefinish=allow_prefinish,
    )
    state = _validate_effect_terminal_fold(
        record["payload_effects"], record["payload_object_state"],
        destination=expected_destination, expected_store=expected_store,
        expected_payload_bytes=expected_payload_bytes,
        expected_payload_sha256=expected_payload_sha256,
        expected_payload_data=expected_payload_data,
    )
    _validate_effect_failure_equations(
        record["payload_effects"], record["primary_failure"],
        record["ordered_follow_on_failures"], destination=expected_destination,
        expected_store=expected_store,
    )
    expected_slot = (
        "authz_primary"
        if type(expected_destination) is PermissionAuthzEvidenceDestination
        else "primary"
    )
    if state.slot_kind != expected_slot or state.payload_kind != expected_destination.payload_kind:
        raise PrivatePermissionEvidenceContractError("main failure draft uses the wrong slot")


def _failure(
    failure_kind: str,
    stage: str,
    component_operation: str,
    reason_code: str,
    *,
    resource_kind: str | None = None,
    resource_role: str | None = None,
    operation_ordinal: int | None = None,
    backend_error_hex: str | None = None,
) -> ComponentFailureItem:
    return ComponentFailureItem(
        failure_kind=failure_kind,
        stage=stage,
        component_operation=component_operation,
        resource_kind=resource_kind,
        resource_role=resource_role,
        resource_ordinal=None,
        operation_ordinal=operation_ordinal,
        backend_error_hex=backend_error_hex,
        reason_code=reason_code,
    )


def _accounting(
    *,
    acquire_status: str,
    candidate: bool,
    acquired: bool,
    binding: str,
    preterminal: str,
    finish_attempted: bool = False,
    requested: str | None = None,
    finish_status: str = "not_applicable",
    certainty: str = "not_applicable",
    recorded: str | None = None,
    finish_reason: str = "none",
) -> AuthorityAccounting:
    return AuthorityAccounting(
        acquire_status=acquire_status,
        lease_candidate_returned=candidate,
        lease_acquired=acquired,
        lease_binding_status=binding,
        preterminal_authority_reason_code=preterminal,
        finish_attempted=finish_attempted,
        requested_outcome=requested,
        finish_status=finish_status,
        effect_certainty=certainty,
        recorded_outcome=recorded,
        finish_reason_code=finish_reason,
    )


def _scope(
    operation: str,
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    destination: RetentionDestination,
    store: ConfidentialStoreExpectation,
) -> PermissionOperationScope:
    unsigned = {
        "schema": (
            "operatormark.darcy_branch_c_v1.permission_authz_retention_scope.v1"
            if type(destination) is PermissionAuthzEvidenceDestination
            else SCOPE_SCHEMA
        ),
        "operation": operation,
        "request_self_sha256": _request_hash(request),
        "access_expectation_self_sha256": _access_hash(access),
        "destination_self_sha256": destination.destination_self_sha256,
        "store_expectation_self_sha256": store.expectation_self_sha256,
    }
    try:
        signed = with_self_hash(unsigned, "scope_self_sha256")
    except Exception as exc:
        raise PrivatePermissionEvidenceContractError(str(exc)) from exc
    return PermissionOperationScope(**signed)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class _RetentionSuccessDraft:
    operation: str
    slot_kind: str
    payload_kind: PayloadKind
    request_self_sha256: str
    access_expectation_self_sha256: str
    destination_self_sha256: str
    store_expectation_self_sha256: str
    relative_path: str
    file_bytes: int
    file_sha256: str
    commitment_domain: str
    authorization_file_sha256: str
    payload_commitment: str
    namespace_object_identity: str
    records_parent_object_identity: str
    store_object_identity: str
    creation_security_descriptor_sha256: str
    confidential_store_profile_sha256: str
    held_object_readback_sha256: str
    publication_state: str
    authority_accounting: AuthorityAccounting
    effects: tuple[StoreNamespaceEffect | StoreFailureEffect, ...]
    object_state: StoreObjectTerminalState


@dataclass(slots=True)
class _Attempt:
    success: bool
    effects: list[StoreNamespaceEffect | StoreFailureEffect]
    state: StoreObjectTerminalState
    accounting: AuthorityAccounting
    primary_failure: ComponentFailureItem | None
    follow_on_failures: list[ComponentFailureItem]
    final_snapshot: StoreObjectSnapshot | None
    creation_evidence: StoreCreationEvidence | None
    validated_failure_draft: RetentionFailureDraft | None = None
    success_draft: _RetentionSuccessDraft | None = None


def _make_failure_draft(
    *,
    destination: RetentionDestination,
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    store: ConfidentialStoreExpectation,
    primary: ComponentFailureItem,
    follows: list[ComponentFailureItem],
    accounting: AuthorityAccounting,
    effects: list[StoreNamespaceEffect | StoreFailureEffect],
    state: StoreObjectTerminalState,
) -> RetentionFailureDraft:
    cls = (
        PermissionAuthzEvidenceRetentionFailureDraft
        if type(destination) is PermissionAuthzEvidenceDestination
        else PermissionEvidenceRetentionFailureDraft
    )
    return cls(
        failed_primary_payload_kind=destination.payload_kind,
        request_self_sha256=_request_hash(request),
        access_expectation_self_sha256=_access_hash(access),
        destination_self_sha256=destination.destination_self_sha256,
        store_expectation_self_sha256=store.expectation_self_sha256,
        primary_failure=primary,
        ordered_follow_on_failures=tuple(follows),
        main_authority_accounting=accounting,
        payload_effects=tuple(effects),
        payload_object_state=state,
        acceptance_emitted=False,
    )


def _make_success_draft(
    *,
    operation: str,
    slot_kind: RetentionSlotKind,
    payload_kind: PayloadKind,
    payload: bytes,
    payload_commitment: str,
    destination: RetentionDestination,
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    store: ConfidentialStoreExpectation,
    accounting: AuthorityAccounting,
    effects: list[StoreNamespaceEffect | StoreFailureEffect],
    state: StoreObjectTerminalState,
    final_snapshot: StoreObjectSnapshot,
    creation: StoreCreationEvidence,
) -> _RetentionSuccessDraft:
    domain = {
        "observation": OBSERVATION_DOMAIN,
        "observation_failure": OBSERVATION_FAILURE_DOMAIN,
        "retention_failure": RETENTION_FAILURE_DOMAIN,
        "authz_evidence": authz_records.PRIVATE_PERMISSION_AUTHZ_DOMAIN,
        "authz_retention_failure": AUTHZ_RETENTION_FAILURE_DOMAIN,
    }[payload_kind]
    publication_state = {
        "observation": "retained_capture_only_not_permission_accepted",
        "observation_failure": "retained_observation_failure_not_permission_accepted",
        "retention_failure": "retained_failure_record_not_permission_accepted",
        "authz_evidence": "retained_authz_evidence_only_not_permission_accepted",
        "authz_retention_failure": (
            "retained_authz_retention_failure_not_permission_accepted"
        ),
    }[payload_kind]
    return _RetentionSuccessDraft(
        operation=operation,
        slot_kind=slot_kind,
        payload_kind=payload_kind,
        request_self_sha256=_request_hash(request),
        access_expectation_self_sha256=_access_hash(access),
        destination_self_sha256=destination.destination_self_sha256,
        store_expectation_self_sha256=store.expectation_self_sha256,
        relative_path=(
            destination.primary_accepted_relative_path
            if _primary_slot(slot_kind)
            else destination.retention_failure_accepted_relative_path
        ),
        file_bytes=len(payload),
        file_sha256=sha256_bytes(payload),
        commitment_domain=domain,
        authorization_file_sha256=_authorization_hash(request),
        payload_commitment=payload_commitment,
        namespace_object_identity=store.namespace_object_identity,
        records_parent_object_identity=store.records_parent_object_identity,
        store_object_identity=final_snapshot.object_identity,
        creation_security_descriptor_sha256=creation.creation_security_descriptor_sha256,
        confidential_store_profile_sha256=store.confidential_store_profile_sha256,
        held_object_readback_sha256=sha256_bytes(payload),
        publication_state=publication_state,
        authority_accounting=accounting,
        effects=tuple(effects),
        object_state=state,
    )


def _validate_success_draft(
    draft: _RetentionSuccessDraft,
    *,
    payload: bytes,
    destination: RetentionDestination,
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    store: ConfidentialStoreExpectation,
    final_snapshot: StoreObjectSnapshot,
    creation: StoreCreationEvidence,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding | None = None,
) -> None:
    operation_for_kind = {
        "observation": "retain_observation",
        "observation_failure": "retain_observation_failure",
        "retention_failure": "retain_retention_failure",
        "authz_evidence": "retain_authz_evidence",
        "authz_retention_failure": "retain_authz_retention_failure",
    }
    slot_for_kind = {
        "observation": "primary",
        "observation_failure": "primary",
        "retention_failure": "retention_failure",
        "authz_evidence": "authz_primary",
        "authz_retention_failure": "authz_retention_failure",
    }[draft.payload_kind]
    expected_domain = {
        "observation": OBSERVATION_DOMAIN,
        "observation_failure": OBSERVATION_FAILURE_DOMAIN,
        "retention_failure": RETENTION_FAILURE_DOMAIN,
        "authz_evidence": authz_records.PRIVATE_PERMISSION_AUTHZ_DOMAIN,
        "authz_retention_failure": AUTHZ_RETENTION_FAILURE_DOMAIN,
    }[draft.payload_kind]
    expected_publication = {
        "observation": "retained_capture_only_not_permission_accepted",
        "observation_failure": "retained_observation_failure_not_permission_accepted",
        "retention_failure": "retained_failure_record_not_permission_accepted",
        "authz_evidence": "retained_authz_evidence_only_not_permission_accepted",
        "authz_retention_failure": (
            "retained_authz_retention_failure_not_permission_accepted"
        ),
    }[draft.payload_kind]
    expected_path = (
        destination.retention_failure_accepted_relative_path
        if _sidecar_slot(slot_for_kind)
        else destination.primary_accepted_relative_path
    )
    expected_values = {
        "operation": operation_for_kind[draft.payload_kind],
        "slot_kind": slot_for_kind,
        "request_self_sha256": _request_hash(request),
        "access_expectation_self_sha256": _access_hash(access),
        "destination_self_sha256": destination.destination_self_sha256,
        "store_expectation_self_sha256": store.expectation_self_sha256,
        "relative_path": expected_path,
        "file_bytes": len(payload),
        "file_sha256": sha256_bytes(payload),
        "commitment_domain": expected_domain,
        "authorization_file_sha256": _authorization_hash(request),
        "namespace_object_identity": store.namespace_object_identity,
        "records_parent_object_identity": store.records_parent_object_identity,
        "store_object_identity": final_snapshot.object_identity,
        "creation_security_descriptor_sha256":
            creation.creation_security_descriptor_sha256,
        "confidential_store_profile_sha256": store.confidential_store_profile_sha256,
        "held_object_readback_sha256": sha256_bytes(payload),
        "publication_state": expected_publication,
    }
    for field, expected in expected_values.items():
        if getattr(draft, field) != expected:
            raise PrivatePermissionEvidenceProtocolError(
                f"preterminal success draft {field} mismatch"
            )
    permitted_sidecar = (
        "authz_retention_failure"
        if type(destination) is PermissionAuthzEvidenceDestination
        else "retention_failure"
    )
    if (
        draft.payload_kind != destination.payload_kind
        and draft.payload_kind != permitted_sidecar
    ):
        raise PrivatePermissionEvidenceProtocolError("preterminal success payload kind mismatch")
    expected_commitment = _retention_payload_commitment(
        payload,
        payload_kind=draft.payload_kind,
        expected_request=request,
        expected_access=access,
        expected_store=store,
        expected_destination=destination,
        expected_authz_binding=expected_authz_binding,
    )
    if draft.payload_commitment != expected_commitment:
        raise PrivatePermissionEvidenceProtocolError("preterminal success commitment mismatch")
    accounting = _validate_authority_accounting(
        draft.authority_accounting, name="preterminal success accounting",
        allow_prefinish=True,
    )
    if (
        accounting["acquire_status"] != "granted"
        or accounting["lease_binding_status"] != "valid"
        or accounting["preterminal_authority_reason_code"] != "none"
    ):
        raise PrivatePermissionEvidenceProtocolError("preterminal success authority mismatch")
    state = _validate_effect_terminal_fold(
        draft.effects, draft.object_state,
        destination=destination, expected_store=store,
        expected_payload_bytes=len(payload),
        expected_payload_sha256=sha256_bytes(payload),
        expected_payload_data=payload,
    )
    if (
        state.slot_kind != slot_for_kind
        or state.publication_state != "published_verified_unaccepted"
        or state.handle_state != "released"
        or state.object_identity != final_snapshot.object_identity
        or state.last_confirmed_relative_path != expected_path
    ):
        raise PrivatePermissionEvidenceProtocolError("preterminal success state mismatch")
    if (
        creation.creation_disposition != "created_new_exclusive"
        or creation.prewrite_size_bytes != 0
        or creation.security_applied_in_create is not True
        or creation.object_identity != final_snapshot.object_identity
        or creation.namespace_object_identity != destination.namespace_object_identity
        or creation.parent_relative_path != destination.records_parent_relative_path
        or creation.parent_object_identity != destination.records_parent_object_identity
        or creation.volume_identifier_commitment != destination.volume_identifier_commitment
        or creation.creation_security_descriptor_sha256
        != destination.required_created_object_security_descriptor_sha256
        or creation.confidential_store_profile_sha256
        != destination.confidential_store_profile_sha256
    ):
        raise PrivatePermissionEvidenceProtocolError("preterminal success creation binding mismatch")
    _validate_object_snapshot(final_snapshot, destination, permitted_paths={expected_path})


def _initial_state(
    destination: RetentionDestination,
    *,
    slot_kind: RetentionSlotKind,
    payload_kind: PayloadKind,
) -> StoreObjectTerminalState:
    if _primary_slot(slot_kind):
        creating = destination.primary_creating_relative_path
        accepted = destination.primary_accepted_relative_path
    else:
        creating = destination.retention_failure_creating_relative_path
        accepted = destination.retention_failure_accepted_relative_path
    return StoreObjectTerminalState(
        slot_kind=slot_kind,
        payload_kind=payload_kind,
        ownership="none",
        effect_certainty="no_effect",
        creating_relative_path=creating,
        accepted_relative_path=accepted,
        last_confirmed_relative_path=None,
        reported_relative_path=None,
        object_identity=None,
        volume_identifier_commitment=None,
        last_confirmed_size_bytes=None,
        last_confirmed_file_sha256=None,
        publication_state="not_created",
        handle_state="not_acquired",
    )


def _known_state(
    old: StoreObjectTerminalState,
    snapshot: StoreObjectSnapshot,
    publication_state: str,
    *,
    handle_state: Literal["released", "release_failed", "unknown"] = "unknown",
) -> StoreObjectTerminalState:
    return StoreObjectTerminalState(
        slot_kind=old.slot_kind,
        payload_kind=old.payload_kind,
        ownership="owned_by_this_invocation",
        effect_certainty="reported_effect",
        creating_relative_path=old.creating_relative_path,
        accepted_relative_path=old.accepted_relative_path,
        last_confirmed_relative_path=snapshot.relative_path,
        reported_relative_path=snapshot.relative_path,
        object_identity=snapshot.object_identity,
        volume_identifier_commitment=snapshot.volume_identifier_commitment,
        last_confirmed_size_bytes=snapshot.size_bytes,
        last_confirmed_file_sha256=snapshot.file_sha256,
        publication_state=publication_state,
        handle_state=handle_state,
    )


def _unknown_state(
    old: StoreObjectTerminalState,
    *,
    owned_handle: bool,
) -> StoreObjectTerminalState:
    has_snapshot = old.object_identity is not None
    return StoreObjectTerminalState(
        slot_kind=old.slot_kind,
        payload_kind=old.payload_kind,
        ownership="owned_by_this_invocation" if owned_handle or has_snapshot else "unknown",
        effect_certainty="unknown",
        creating_relative_path=old.creating_relative_path,
        accepted_relative_path=old.accepted_relative_path,
        last_confirmed_relative_path=old.last_confirmed_relative_path if has_snapshot else None,
        reported_relative_path=None,
        object_identity=old.object_identity if has_snapshot else None,
        volume_identifier_commitment=(old.volume_identifier_commitment if has_snapshot else None),
        last_confirmed_size_bytes=(old.last_confirmed_size_bytes if has_snapshot else None),
        last_confirmed_file_sha256=(old.last_confirmed_file_sha256 if has_snapshot else None),
        publication_state="effect_unknown",
        handle_state="unknown" if owned_handle else "not_acquired",
    )


def _snapshot_disagrees_with_owned_state(
    state: StoreObjectTerminalState,
    snapshot: StoreObjectSnapshot,
) -> bool:
    if state.ownership != "owned_by_this_invocation":
        return False
    confirmed = (
        state.last_confirmed_relative_path,
        state.object_identity,
        state.volume_identifier_commitment,
        state.last_confirmed_size_bytes,
        state.last_confirmed_file_sha256,
    )
    if any(value is None for value in confirmed):
        return True
    return confirmed != (
        snapshot.relative_path,
        snapshot.object_identity,
        snapshot.volume_identifier_commitment,
        snapshot.size_bytes,
        snapshot.file_sha256,
    )


def _fold_snapshot_effect(
    state: StoreObjectTerminalState,
    effect: StoreFailureEffect,
    *,
    payload_length: int,
    owned_handle: bool,
) -> StoreObjectTerminalState:
    if (
        effect.operation == "snapshot"
        and effect.completion_status == "succeeded"
        and effect.effect_certainty == "reported_effect"
        and effect.after_snapshot is not None
        and _snapshot_disagrees_with_owned_state(state, effect.after_snapshot)
    ):
        return _unknown_state(state, owned_handle=owned_handle)
    if effect.effect_certainty == "unknown":
        return _unknown_state(state, owned_handle=owned_handle)
    snapshot = effect.after_snapshot
    if effect.completion_status == "collision" and effect.operation == "create_exclusive":
        return StoreObjectTerminalState(
            **{
                **asdict(state),
                "ownership": "unowned_collision",
                "publication_state": "collision_unowned",
            }
        )
    if effect.effect_certainty == "no_effect":
        return state
    if snapshot is None:
        return state
    if snapshot.publication_state == "published":
        phase = "published_unverified"
    elif snapshot.size_bytes == 0:
        phase = "creating_empty"
    elif snapshot.size_bytes < payload_length:
        phase = "creating_partial"
    else:
        phase = "creating_complete_unverified"
    return _known_state(state, snapshot, phase)


def _with_handle_state(
    state: StoreObjectTerminalState,
    handle_state: Literal["released", "release_failed", "unknown"],
) -> StoreObjectTerminalState:
    values = asdict(state)
    values["handle_state"] = handle_state
    return StoreObjectTerminalState(**values)


def _last_snapshot_from_state(
    state: StoreObjectTerminalState,
    destination: RetentionDestination,
) -> StoreObjectSnapshot | None:
    if (
        state.last_confirmed_relative_path is None
        or state.object_identity is None
        or state.volume_identifier_commitment is None
        or state.last_confirmed_size_bytes is None
        or state.last_confirmed_file_sha256 is None
    ):
        return None
    return StoreObjectSnapshot(
        relative_path=state.last_confirmed_relative_path,
        namespace_object_identity=destination.namespace_object_identity,
        parent_relative_path=destination.records_parent_relative_path,
        parent_object_identity=destination.records_parent_object_identity,
        object_identity=state.object_identity,
        volume_identifier_commitment=state.volume_identifier_commitment,
        size_bytes=state.last_confirmed_size_bytes,
        file_sha256=state.last_confirmed_file_sha256,
        security_descriptor_sha256=destination.required_created_object_security_descriptor_sha256,
        confidential_store_profile_sha256=destination.confidential_store_profile_sha256,
        publication_state=(
            "creating"
            if state.last_confirmed_relative_path == state.creating_relative_path
            else "published"
        ),
    )


def _unknown_store_effect(
    operation: str,
    state: StoreObjectTerminalState,
    destination: RetentionDestination,
    *,
    bytes_requested: int | None = None,
) -> StoreFailureEffect:
    return StoreFailureEffect(
        operation=operation,
        completion_status="failed",
        effect_certainty="unknown",
        before_snapshot=_last_snapshot_from_state(state, destination),
        after_snapshot=None,
        collision_relative_path=None,
        bytes_requested=bytes_requested,
        bytes_reported=None,
        backend_error_hex=None,
    )


def _append_failure(
    primary: ComponentFailureItem | None,
    follows: list[ComponentFailureItem],
    item: ComponentFailureItem,
) -> ComponentFailureItem:
    if primary is None:
        return item
    follows.append(item)
    return primary


def _lease_fields(lease: object) -> dict[str, object] | None:
    names = (
        "request_self_sha256", "operation", "access_expectation_self_sha256",
        "destination_self_sha256", "store_expectation_self_sha256", "scope_self_sha256",
    )
    try:
        result = {name: getattr(lease, name) for name in names}
        if not callable(getattr(lease, "require_active")) or not callable(
            getattr(lease, "finish")
        ):
            return None
        return result
    except (AttributeError, TypeError):
        return None


def _finish(
    lease: object,
    requested: str,
    accounting: AuthorityAccounting,
) -> tuple[AuthorityAccounting, ComponentFailureItem | None]:
    try:
        outcome = lease.finish(requested)
        if type(outcome) is not AuthorityFinishOutcome:
            raise TypeError("finish returned the wrong exact type")
        status = getattr(outcome, "finish_status")
        certainty = getattr(outcome, "effect_certainty")
        recorded = getattr(outcome, "recorded_outcome")
        reason = getattr(outcome, "reason_code")
    except Exception:
        status, certainty, recorded, reason = "failed", "unknown", None, "terminal_finish_failed"
    valid_success = (
        status == "succeeded"
        and certainty == "confirmed"
        and recorded == requested
        and reason == "none"
    )
    valid_failure = (
        status == "failed"
        and reason == "terminal_finish_failed"
        and (
            (certainty == "confirmed" and recorded == requested)
            or (certainty in {"not_applied", "unknown"} and recorded is None)
        )
    )
    if not valid_success and not valid_failure:
        status, certainty, recorded, reason = (
            "failed", "unknown", None, "terminal_finish_failed"
        )
    finished = _accounting(
        acquire_status=accounting.acquire_status,
        candidate=accounting.lease_candidate_returned,
        acquired=accounting.lease_acquired,
        binding=accounting.lease_binding_status,
        preterminal=accounting.preterminal_authority_reason_code,
        finish_attempted=True,
        requested=requested,
        finish_status=status if status in {"succeeded", "failed"} else "failed",
        certainty=certainty if certainty in {"confirmed", "not_applied", "unknown"} else "unknown",
        recorded=recorded if recorded in {"succeeded", "refused", "failed"} else None,
        finish_reason="none" if valid_success else "terminal_finish_failed",
    )
    if valid_success:
        return finished, None
    return finished, _failure(
        "terminal_accounting_failure",
        "terminal_accounting",
        "finish_authority_lease",
        "terminal_finish_failed",
    )


def _active_failure(
    lease: object,
    operation: str,
    step: str,
    ordinal: int | None,
) -> tuple[ComponentFailureItem | None, str]:
    checkpoint = PermissionAuthorityCheckpoint(operation=operation, step=step, ordinal=ordinal)
    try:
        result = lease.require_active(checkpoint)
        if type(result) is not AuthorityCheckOutcome:
            raise TypeError("require_active returned the wrong exact type")
        status = getattr(result, "status")
        echoed = getattr(result, "checkpoint")
        supplied = getattr(result, "failure")
    except Exception:
        return (
            _failure(
                "authority_failure", "final_authority_check" if step == "final" else _stage_for_step(step),
                "require_active", "authority_check_failed", operation_ordinal=ordinal,
            ),
            "authority_check_failed",
        )
    if status == "active" and echoed == checkpoint and supplied is None:
        return None, "none"
    if status == "refused" and echoed == checkpoint:
        if supplied is not None:
            try:
                supplied = _require_failure(
                    supplied, name="authority refusal", kinds={"authority_refusal"},
                    stages={_stage_for_step(step)}, operations={"require_active"},
                    reasons={"authority_inactive"},
                )
                if supplied.operation_ordinal != ordinal:
                    raise PrivatePermissionEvidenceContractError(
                        "authority refusal operation ordinal mismatch"
                    )
            except PrivatePermissionEvidenceContractError:
                supplied = None
        return (
            supplied if isinstance(supplied, ComponentFailureItem) else _failure(
                "authority_refusal", _stage_for_step(step), "require_active",
                "authority_inactive", operation_ordinal=ordinal,
            ),
            "authority_inactive",
        )
    if supplied is not None:
        try:
            supplied = _require_failure(
                supplied, name="authority check failure", kinds={"authority_failure"},
                stages={_stage_for_step(step)}, operations={"require_active"},
                reasons={"authority_check_failed"},
            )
            if supplied.operation_ordinal != ordinal:
                raise PrivatePermissionEvidenceContractError(
                    "authority failure operation ordinal mismatch"
                )
        except PrivatePermissionEvidenceContractError:
            supplied = None
    return (
        supplied if isinstance(supplied, ComponentFailureItem) else _failure(
            "authority_failure", _stage_for_step(step), "require_active",
            "authority_check_failed", operation_ordinal=ordinal,
        ),
        "authority_check_failed",
    )


def _stage_for_step(step: str) -> str:
    if step.startswith("sidecar_"):
        step = step[len("sidecar_"):]
    if step == "final":
        return "final_authority_check"
    return {
        "namespace_open": "namespace_open",
        "namespace_validate": "namespace_validation",
        "create": "exclusive_create",
        "primary_create": "exclusive_create",
        "write": "payload_write",
        "primary_write": "payload_write",
        "flush": "payload_flush",
        "primary_flush": "payload_flush",
        "rewind": "payload_rewind",
        "primary_rewind": "payload_rewind",
        "read_chunk": "payload_readback",
        "primary_read_chunk": "payload_readback",
        "eof": "payload_readback",
        "primary_eof": "payload_readback",
        "validate": "readback_validation",
        "primary_validate": "readback_validation",
        "snapshot": "prepublication_snapshot",
        "primary_snapshot": "prepublication_snapshot",
        "publish": "publication",
        "primary_publish": "publication",
        "postpublish_snapshot": "postpublication_snapshot",
        "primary_postpublish_snapshot": "postpublication_snapshot",
        "postpublish_rewind": "postpublication_readback",
        "primary_postpublish_rewind": "postpublication_readback",
        "postpublish_read_chunk": "postpublication_readback",
        "primary_postpublish_read_chunk": "postpublication_readback",
        "postpublish_eof": "postpublication_readback",
        "primary_postpublish_eof": "postpublication_readback",
        "postpublish_validate": "postpublication_readback",
        "primary_postpublish_validate": "postpublication_readback",
    }.get(step, "retention_failure_retention")


def _protocol_store_failure(
    *,
    stage: str,
    operation: str,
    resource_kind: str | None = None,
    resource_role: str | None = None,
) -> ComponentFailureItem:
    return _failure(
        "store_operation_failure", stage, operation, "store_protocol_failure",
        resource_kind=resource_kind, resource_role=resource_role,
    )


def _validate_creation(
    evidence: StoreCreationEvidence,
    snapshot: StoreObjectSnapshot,
    destination: RetentionDestination,
) -> None:
    if (
        evidence.creation_disposition != "created_new_exclusive"
        or isinstance(evidence.prewrite_size_bytes, bool)
        or evidence.prewrite_size_bytes != 0
        or not evidence.security_applied_in_create
        or evidence.namespace_object_identity != destination.namespace_object_identity
        or evidence.parent_relative_path != destination.records_parent_relative_path
        or evidence.parent_object_identity != destination.records_parent_object_identity
        or evidence.object_identity != snapshot.object_identity
        or evidence.volume_identifier_commitment != destination.volume_identifier_commitment
        or evidence.creation_security_descriptor_sha256
        != destination.required_created_object_security_descriptor_sha256
        or evidence.confidential_store_profile_sha256
        != destination.confidential_store_profile_sha256
        or snapshot.size_bytes != 0
    ):
        raise PrivatePermissionEvidenceContractError("creation evidence mismatch")


def _release_object(
    handle: ConfidentialStoreObject,
    state: StoreObjectTerminalState,
    destination: RetentionDestination,
) -> tuple[StoreFailureEffect, StoreObjectTerminalState, ComponentFailureItem | None]:
    try:
        outcome = handle.release()
        if type(outcome) is not StoreReleaseOutcome:
            raise ValueError("object release returned wrong exact type")
        if outcome.resource_kind != "store_object_handle" or outcome.object_effect is None or outcome.namespace_effect is not None:
            raise ValueError("invalid object release outcome")
        effect = outcome.object_effect
        _validate_store_effect(effect, destination)
        if effect.operation != "release_store_object":
            raise ValueError("wrong object release operation")
        if outcome.status == "released" and effect.completion_status == "succeeded" and outcome.failure is None:
            return effect, _with_handle_state(state, "released"), None
        if outcome.status != "failed" or not isinstance(outcome.failure, ComponentFailureItem):
            raise ValueError("invalid failed object release")
        failure = _require_failure(
            outcome.failure, name="object release", kinds={"cleanup_failure"},
            stages={"resource_cleanup"}, operations={"release_store_object"},
            reasons={"resource_release_failed"},
        )
        handle_state = {
            "no_effect": "release_failed",
            "reported_effect": "released",
            "unknown": "unknown",
        }[effect.effect_certainty]
        return effect, _with_handle_state(state, handle_state), failure
    except Exception:
        effect = StoreFailureEffect(
            operation="release_store_object", completion_status="failed",
            effect_certainty="unknown", before_snapshot=None, after_snapshot=None,
            collision_relative_path=None, bytes_requested=None, bytes_reported=None,
            backend_error_hex=None,
        )
        return effect, _with_handle_state(state, "unknown"), _failure(
            "cleanup_failure", "resource_cleanup", "release_store_object",
            "resource_release_failed", resource_kind="store_object_handle",
            resource_role=_slot_resource_role(state.slot_kind),
        )


def _release_namespace(
    lease: ConfidentialNamespaceLease,
    expected_store: ConfidentialStoreExpectation,
) -> tuple[StoreNamespaceEffect, ComponentFailureItem | None]:
    try:
        outcome = lease.release()
        if type(outcome) is not StoreReleaseOutcome:
            raise ValueError("namespace release returned wrong exact type")
        if outcome.resource_kind != "namespace_handle" or outcome.namespace_effect is None or outcome.object_effect is not None:
            raise ValueError("invalid namespace release outcome")
        effect = outcome.namespace_effect
        _validate_namespace_effect(effect, expected_store)
        if outcome.status == "released" and effect.completion_status == "succeeded" and outcome.failure is None:
            return effect, None
        if outcome.status != "failed" or not isinstance(outcome.failure, ComponentFailureItem):
            raise ValueError("invalid failed namespace release")
        failure = _require_failure(
            outcome.failure, name="namespace release", kinds={"cleanup_failure"},
            stages={"resource_cleanup"}, operations={"release_namespace"},
            reasons={"namespace_release_failed"},
        )
        return effect, failure
    except Exception:
        before = lease.snapshot
        effect = StoreNamespaceEffect(
            operation="release_namespace", completion_status="failed",
            effect_certainty="unknown", before_snapshot=before, after_snapshot=None,
            backend_error_hex=None,
        )
        return effect, _failure(
            "cleanup_failure", "resource_cleanup", "release_namespace",
            "namespace_release_failed", resource_kind="namespace_handle",
            resource_role="records_namespace",
        )


def _execute_store_attempt(
    *,
    operation: Literal[
        "retain_observation", "retain_observation_failure", "retain_retention_failure",
        "retain_authz_evidence", "retain_authz_retention_failure",
    ],
    slot_kind: RetentionSlotKind,
    payload_kind: PayloadKind,
    payload: bytes,
    payload_commitment: str,
    destination: RetentionDestination,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    authority: PermissionObservationAuthority,
    store: ConfidentialEvidenceStore,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding | None = None,
) -> _Attempt:
    scope = _scope(operation, expected_request, expected_access, destination, expected_store)
    state = _initial_state(destination, slot_kind=slot_kind, payload_kind=payload_kind)
    effects: list[StoreNamespaceEffect | StoreFailureEffect] = []
    follows: list[ComponentFailureItem] = []
    primary: ComponentFailureItem | None = None
    namespace: ConfidentialNamespaceLease | None = None
    handle: ConfidentialStoreObject | None = None
    creation: StoreCreationEvidence | None = None
    final_snapshot: StoreObjectSnapshot | None = None

    def validate_prefinish_failure(
        failure: ComponentFailureItem,
        current_accounting: AuthorityAccounting,
    ) -> RetentionFailureDraft | None:
        if _primary_slot(slot_kind):
            draft = _make_failure_draft(
                destination=destination, request=expected_request, access=expected_access,
                store=expected_store, primary=failure, follows=follows,
                accounting=current_accounting, effects=effects, state=state,
            )
            _validate_failure_draft(
                draft,
                expected_request=expected_request,
                expected_access=expected_access,
                expected_store=expected_store,
                expected_destination=destination,
                allow_prefinish=True,
                expected_payload_bytes=len(payload),
                expected_payload_sha256=sha256_bytes(payload),
                expected_payload_data=payload,
            )
            return draft
        _validate_failure_accounting_equations(
            failure, follows, current_accounting,
            name="sidecar pre-finish failure", allow_prefinish=True,
        )
        side_state = _validate_effect_terminal_fold(
            effects, state, destination=destination, expected_store=expected_store,
            expected_payload_bytes=len(payload),
            expected_payload_sha256=sha256_bytes(payload),
            expected_payload_data=payload,
        )
        _validate_effect_failure_equations(
            effects, failure, follows, destination=destination,
            expected_store=expected_store,
        )
        expected_side = (
            "authz_retention_failure"
            if type(destination) is PermissionAuthzEvidenceDestination
            else "retention_failure"
        )
        if side_state.slot_kind != expected_side or side_state.payload_kind != expected_side:
            raise PrivatePermissionEvidenceProtocolError("sidecar pre-finish state slot mismatch")
        return None
    try:
        acquired = authority.acquire(request=expected_request, scope=scope)
        if type(acquired) is not AuthorityAcquireOutcome:
            raise TypeError("authority acquire returned the wrong exact type")
        acquire_status = getattr(acquired, "status")
        lease = getattr(acquired, "lease")
        acquire_failure = getattr(acquired, "failure")
    except Exception:
        acquire_status, lease = "failed", None
        acquire_failure = _failure(
            "authority_failure", "authority_acquisition", "acquire_authority_lease",
            "authority_acquire_failed",
        )
    if acquire_status != "granted":
        reason = "authority_acquire_refused" if acquire_status == "refused" else "authority_acquire_failed"
        if acquire_failure is not None:
            try:
                primary = _require_failure(
                    acquire_failure,
                    name="authority acquisition",
                    kinds={"authority_refusal" if acquire_status == "refused" else "authority_failure"},
                    stages={"authority_acquisition"},
                    operations={"acquire_authority_lease"}, reasons={reason},
                )
            except PrivatePermissionEvidenceContractError:
                primary = None
        if primary is None:
            primary = _failure(
                "authority_refusal" if acquire_status == "refused" else "authority_failure",
                "authority_acquisition", "acquire_authority_lease", reason,
            )
        current_accounting = _accounting(
            acquire_status=acquire_status if acquire_status in {"refused", "failed"} else "failed",
            candidate=False, acquired=False, binding="not_applicable", preterminal=reason,
        )
        draft = validate_prefinish_failure(primary, current_accounting)
        return _Attempt(
            False, effects, state,
            current_accounting,
            primary, follows, None, None, draft,
        )
    if lease is None or acquire_failure is not None:
        lease = object()
    accounting = _accounting(
        acquire_status="granted", candidate=True, acquired=False,
        binding="invalid", preterminal="lease_malformed",
    )
    fields = _lease_fields(lease)
    if fields is None:
        malformed = _failure(
            "lease_binding_failure", "lease_binding", "validate_authority_lease", "lease_malformed"
        )
        draft = validate_prefinish_failure(malformed, accounting)
        return _Attempt(
            False, effects, state, accounting,
            malformed, follows, None, None, draft,
        )
    mismatches = (
        ("request_self_sha256", scope.request_self_sha256, "lease_request_mismatch"),
        ("operation", scope.operation, "lease_operation_mismatch"),
        ("access_expectation_self_sha256", scope.access_expectation_self_sha256, "lease_access_expectation_mismatch"),
        ("destination_self_sha256", scope.destination_self_sha256, "lease_destination_mismatch"),
        ("store_expectation_self_sha256", scope.store_expectation_self_sha256, "lease_store_expectation_mismatch"),
        ("scope_self_sha256", scope.scope_self_sha256, "lease_scope_mismatch"),
    )
    for field, expected, reason in mismatches:
        if fields[field] != expected:
            accounting = _accounting(
                acquire_status="granted", candidate=True, acquired=True,
                binding="invalid", preterminal=reason,
            )
            primary = _failure(
                "lease_binding_failure", "lease_binding", "validate_authority_lease", reason
            )
            draft = validate_prefinish_failure(primary, accounting)
            accounting, terminal = _finish(lease, "failed", accounting)
            if terminal is not None:
                follows.append(terminal)
            final_draft: RetentionFailureDraft | None = None
            if _primary_slot(slot_kind):
                final_draft = _make_failure_draft(
                    destination=destination, request=expected_request, access=expected_access,
                    store=expected_store, primary=primary, follows=follows,
                    accounting=accounting, effects=effects, state=state,
                )
                _validate_failure_draft(
                    final_draft,
                    expected_request=expected_request,
                    expected_access=expected_access,
                    expected_store=expected_store,
                    expected_destination=destination,
                    expected_payload_bytes=len(payload),
                    expected_payload_sha256=sha256_bytes(payload),
                    expected_payload_data=payload,
                )
            else:
                _validate_failure_accounting_equations(
                    primary, follows, accounting, name="sidecar lease-binding failure"
                )
                _validate_effect_terminal_fold(
                    effects, state, destination=destination, expected_store=expected_store,
                    expected_payload_bytes=len(payload),
                    expected_payload_sha256=sha256_bytes(payload),
                    expected_payload_data=payload,
                )
            return _Attempt(
                False, effects, state, accounting, primary, follows, None, None,
                final_draft,
            )
    accounting = _accounting(
        acquire_status="granted", candidate=True, acquired=True,
        binding="valid", preterminal="none",
    )

    prefix = "sidecar_" if _sidecar_slot(slot_kind) else "primary_"
    namespace_prefix = "sidecar_" if _sidecar_slot(slot_kind) else ""
    creating_path = (
        destination.primary_creating_relative_path
        if _primary_slot(slot_kind)
        else destination.retention_failure_creating_relative_path
    )
    final_path = (
        destination.primary_accepted_relative_path
        if _primary_slot(slot_kind)
        else destination.retention_failure_accepted_relative_path
    )
    payload_length = len(payload)
    payload_sha256 = sha256_bytes(payload)
    object_role = _slot_resource_role(slot_kind)

    def validate_effect_response(
        effect: StoreFailureEffect,
        *,
        expected_operation: str,
        bytes_requested: int | None = None,
        bind_before: bool = True,
    ) -> None:
        _validate_store_effect(effect, destination)
        if effect.operation != expected_operation:
            raise PrivatePermissionEvidenceContractError(
                f"{expected_operation} response names the wrong operation"
            )
        if bind_before and expected_operation != "create_exclusive":
            expected_before = _last_snapshot_from_state(state, destination)
            if effect.before_snapshot != expected_before:
                raise PrivatePermissionEvidenceContractError(
                    f"{expected_operation} response does not bind the current state"
                )
        if bytes_requested is not None and effect.bytes_requested != bytes_requested:
            raise PrivatePermissionEvidenceContractError(
                f"{expected_operation} response does not bind the byte request"
            )
        if (
            expected_operation == "write_once"
            and effect.effect_certainty == "reported_effect"
        ):
            reported = effect.bytes_reported
            after = effect.after_snapshot
            if (
                reported is None
                or after is None
                or reported < 0
                or reported > payload_length
                or after.size_bytes != reported
                or after.file_sha256 != sha256_bytes(payload[:reported])
            ):
                raise PrivatePermissionEvidenceContractError(
                    "write response does not bind the supplied payload bytes"
                )
        if (
            expected_operation == "publish_same_object_exclusive"
            and effect.effect_certainty == "reported_effect"
            and effect.after_snapshot is not None
            and (
                effect.after_snapshot.size_bytes != payload_length
                or effect.after_snapshot.file_sha256 != payload_sha256
            )
        ):
            raise PrivatePermissionEvidenceContractError(
                "publication response does not bind the supplied payload"
            )

    def require_bound_failure(
        value: object,
        *,
        name: str,
        kinds: set[str],
        stage: str,
        operation_name: str,
        reasons: set[str],
        effect: StoreFailureEffect,
        resource_kind: str | None = "store_object_handle",
        resource_role: str | None = object_role,
    ) -> ComponentFailureItem:
        failure = _require_failure(
            value,
            name=name,
            kinds=kinds,
            stages={stage},
            operations={operation_name},
            reasons=reasons,
        )
        if (
            failure.resource_kind != resource_kind
            or failure.resource_role != resource_role
            or failure.resource_ordinal is not None
            or failure.operation_ordinal is not None
            or failure.backend_error_hex != effect.backend_error_hex
            or (
                effect.effect_certainty == "unknown"
                and failure.reason_code
                not in {"effect_unknown", "store_protocol_failure"}
            )
        ):
            raise PrivatePermissionEvidenceContractError(
                f"{name} failure does not bind its response"
            )
        return failure

    def record_protocol_failure(
        *,
        stage: str,
        operation_name: str,
        bytes_requested: int | None = None,
        resource_kind: str | None = "store_object_handle",
        resource_role: str | None = object_role,
    ) -> None:
        nonlocal primary, state
        effect = _unknown_store_effect(
            operation_name,
            state,
            destination,
            bytes_requested=bytes_requested,
        )
        next_state = _fold_snapshot_effect(
            state,
            effect,
            payload_length=payload_length,
            owned_handle=handle is not None,
        )
        effects.append(effect)
        state = next_state
        primary = _append_failure(
            primary,
            follows,
            _protocol_store_failure(
                stage=stage,
                operation=operation_name,
                resource_kind=resource_kind,
                resource_role=resource_role,
            ),
        )

    def check(step: str, ordinal: int | None = None) -> bool:
        nonlocal primary, accounting
        failure, reason = _active_failure(lease, operation, step, ordinal)
        if failure is None:
            return True
        primary = _append_failure(primary, follows, failure)
        accounting = _accounting(
            acquire_status="granted", candidate=True, acquired=True,
            binding="valid", preterminal=reason,
        )
        return False

    if check(f"{namespace_prefix}namespace_open"):
        try:
            opened = store.open_namespace(destination.namespace_relative_path)
            returned_namespace = getattr(opened, "lease", None)
            if returned_namespace is not None:
                namespace = returned_namespace
            if type(opened) is not NamespaceOpenOutcome:
                raise TypeError("namespace open returned wrong exact type")
            if type(opened.effect) is not StoreNamespaceEffect:
                raise TypeError("namespace open effect returned wrong exact type")
            _validate_namespace_effect(opened.effect, expected_store)
            effects.append(opened.effect)
            opened_shape = (
                opened.status == "opened"
                and opened.lease is not None
                and opened.failure is None
                and opened.effect.completion_status == "succeeded"
                and opened.effect.effect_certainty == "reported_effect"
                and opened.lease.snapshot == opened.effect.after_snapshot
            )
            failed_shape = (
                opened.status == "failed"
                and isinstance(opened.failure, ComponentFailureItem)
                and opened.effect.completion_status == "failed"
                and (
                    (
                        opened.effect.effect_certainty in {"no_effect", "unknown"}
                        and opened.lease is None
                    )
                    or (
                        opened.effect.effect_certainty == "reported_effect"
                        and opened.lease is not None
                        and opened.lease.snapshot == opened.effect.after_snapshot
                    )
                )
            )
            if opened_shape:
                namespace = opened.lease
            else:
                failure = None
                if failed_shape:
                    try:
                        failure = _require_failure(
                            opened.failure, name="namespace open",
                            kinds={"store_operation_failure"}, stages={"namespace_open"},
                            operations={"open_namespace"},
                            reasons={"namespace_open_failed", "effect_unknown", "store_protocol_failure"},
                        )
                    except PrivatePermissionEvidenceContractError:
                        failure = None
                primary = _append_failure(primary, follows, failure or _protocol_store_failure(
                    stage="namespace_open", operation="open_namespace"
                ))
        except Exception:
            unknown = StoreNamespaceEffect(
                operation="open_namespace", completion_status="failed",
                effect_certainty="unknown", before_snapshot=None, after_snapshot=None,
                backend_error_hex=None,
            )
            effects.append(unknown)
            primary = _append_failure(
                primary, follows,
                _protocol_store_failure(stage="namespace_open", operation="open_namespace"),
            )
    if primary is None and namespace is not None and check(f"{namespace_prefix}namespace_validate"):
        try:
            _validate_namespace_snapshot(namespace.snapshot, expected_store)
        except Exception:
            primary = _append_failure(
                primary, follows,
                _failure(
                    "validation_failure", "namespace_validation", "validate_namespace",
                    "namespace_validation_failed", resource_kind="namespace_handle",
                    resource_role="records_namespace",
                ),
            )
    if primary is None and namespace is not None and check(f"{prefix}create"):
        try:
            created = namespace.create_exclusive(
                creating_path,
                confidential_store_profile_sha256=destination.confidential_store_profile_sha256,
                required_security_descriptor_sha256=destination.required_created_object_security_descriptor_sha256,
            )
            returned_handle = getattr(created, "object_handle", None)
            if returned_handle is not None:
                handle = returned_handle
            if type(created) is not StoreCreateOutcome:
                raise TypeError("create returned wrong exact type")
            if type(created.effect) is not StoreFailureEffect:
                raise TypeError("create effect returned wrong exact type")
            handle = created.object_handle
            creation = created.creation_evidence
            validate_effect_response(
                created.effect,
                expected_operation="create_exclusive",
                bind_before=False,
            )
            created_shape = (
                created.status == "created"
                and handle is not None
                and creation is not None
                and created.failure is None
                and created.effect.completion_status == "succeeded"
                and created.effect.effect_certainty == "reported_effect"
            )
            collision_shape = (
                created.status == "collision"
                and handle is None
                and creation is None
                and isinstance(created.failure, ComponentFailureItem)
                and created.effect.completion_status == "collision"
                and created.effect.effect_certainty == "no_effect"
            )
            failed_shape = (
                created.status == "failed"
                and isinstance(created.failure, ComponentFailureItem)
                and created.effect.completion_status == "failed"
                and (
                    (
                        created.effect.effect_certainty == "no_effect"
                        and handle is None
                        and creation is None
                    )
                    or (
                        created.effect.effect_certainty == "reported_effect"
                        and handle is not None
                        and creation is not None
                        and created.effect.after_snapshot is not None
                    )
                    or (
                        created.effect.effect_certainty == "unknown"
                        and creation is None
                    )
                )
            )
            if not (created_shape or collision_shape or failed_shape):
                raise TypeError("create returned malformed outcome")
            failure: ComponentFailureItem | None = None
            creation_failure: ComponentFailureItem | None = None
            if created_shape:
                snapshot = created.effect.after_snapshot
                if snapshot is None:
                    raise ValueError("successful create lacks snapshot")
                try:
                    _validate_creation(creation, snapshot, destination)
                except PrivatePermissionEvidenceContractError:
                    creation_failure = _failure(
                        "validation_failure", "creation_validation",
                        "validate_creation_evidence", "creation_evidence_malformed",
                        resource_kind="store_object_handle",
                        resource_role=object_role,
                    )
            else:
                if failed_shape and created.effect.effect_certainty == "reported_effect":
                    snapshot = created.effect.after_snapshot
                    if snapshot is None or creation is None:
                        raise ValueError("reported failed create lacks evidence")
                    try:
                        _validate_creation(creation, snapshot, destination)
                    except PrivatePermissionEvidenceContractError:
                        creation_failure = _failure(
                            "validation_failure", "creation_validation",
                            "validate_creation_evidence", "creation_evidence_malformed",
                            resource_kind="store_object_handle",
                            resource_role=object_role,
                        )
                failure = require_bound_failure(
                    created.failure,
                    name="exclusive create",
                    kinds={"store_collision"} if collision_shape else {"store_operation_failure"},
                    stage="exclusive_create",
                    operation_name="create_exclusive",
                    reasons=(
                        {"collision"}
                        if collision_shape
                        else {"create_failed", "effect_unknown", "store_protocol_failure"}
                    ),
                    effect=created.effect,
                    resource_kind="store_object_handle" if handle is not None else None,
                    resource_role=object_role if handle is not None else None,
                )
            next_state = _fold_snapshot_effect(
                state,
                created.effect,
                payload_length=payload_length,
                owned_handle=handle is not None,
            )
            effects.append(created.effect)
            state = next_state
            if creation_failure is not None:
                primary = _append_failure(
                    primary, follows, creation_failure,
                )
            elif failure is not None:
                primary = _append_failure(primary, follows, failure)
        except Exception:
            creation = None
            record_protocol_failure(
                stage="exclusive_create",
                operation_name="create_exclusive",
                resource_kind="store_object_handle" if handle is not None else None,
                resource_role=object_role if handle is not None else None,
            )

    def io(step: str, method: str, stage: str) -> bool:
        nonlocal primary, state
        if primary is not None or handle is None or not check(step):
            return False
        try:
            outcome = getattr(handle, method)()
            if type(outcome) is not StoreIoOutcome or type(outcome.effect) is not StoreFailureEffect:
                raise TypeError(f"{method} returned wrong exact type")
            validate_effect_response(
                outcome.effect,
                expected_operation=method,
            )
            success_shape = (
                outcome.status == "succeeded"
                and outcome.failure is None
                and outcome.effect.completion_status == "succeeded"
                and outcome.effect.effect_certainty == "reported_effect"
            )
            failure_shape = (
                outcome.status == "failed"
                and isinstance(outcome.failure, ComponentFailureItem)
                and outcome.effect.completion_status == "failed"
            )
            if not (success_shape or failure_shape):
                raise TypeError(f"{method} returned malformed outcome")
            failure = None
            if failure_shape:
                failure = require_bound_failure(
                    outcome.failure,
                    name=method,
                    kinds={"store_operation_failure"},
                    stage=stage,
                    operation_name=method,
                    reasons={
                        "flush_failed", "rewind_failed", "effect_unknown",
                        "store_protocol_failure",
                    },
                    effect=outcome.effect,
                )
            next_state = _fold_snapshot_effect(
                state,
                outcome.effect,
                payload_length=payload_length,
                owned_handle=True,
            )
            effects.append(outcome.effect)
            state = next_state
            if failure is not None:
                primary = _append_failure(primary, follows, failure)
                return False
            return True
        except Exception:
            record_protocol_failure(
                stage=stage,
                operation_name=method,
            )
            return False

    if primary is None and handle is not None and check(f"{prefix}write"):
        try:
            outcome = handle.write_once(payload)
            if type(outcome) is not StoreIoOutcome or type(outcome.effect) is not StoreFailureEffect:
                raise TypeError("write_once returned wrong exact type")
            validate_effect_response(
                outcome.effect,
                expected_operation="write_once",
                bytes_requested=payload_length,
            )
            success_shape = (
                outcome.status == "succeeded"
                and outcome.failure is None
                and outcome.effect.completion_status == "succeeded"
                and outcome.effect.effect_certainty == "reported_effect"
                and outcome.effect.bytes_requested == payload_length
                and outcome.effect.bytes_reported == payload_length
            )
            failure_shape = (
                outcome.status == "failed"
                and isinstance(outcome.failure, ComponentFailureItem)
                and outcome.effect.completion_status == "failed"
            )
            if not (success_shape or failure_shape):
                raise TypeError("write_once returned malformed outcome")
            failure = None
            if failure_shape:
                failure = require_bound_failure(
                    outcome.failure,
                    name="write_once",
                    kinds={"store_operation_failure"},
                    stage="payload_write",
                    operation_name="write_once",
                    reasons={
                        "write_failed", "short_write", "effect_unknown",
                        "store_protocol_failure",
                    },
                    effect=outcome.effect,
                )
            next_state = _fold_snapshot_effect(
                state,
                outcome.effect,
                payload_length=payload_length,
                owned_handle=True,
            )
            effects.append(outcome.effect)
            state = next_state
            if failure is not None:
                primary = _append_failure(
                    primary, follows, failure,
                )
        except Exception:
            record_protocol_failure(
                stage="payload_write",
                operation_name="write_once",
                bytes_requested=payload_length,
            )
    io(f"{prefix}flush", "flush_data_and_metadata", "payload_flush")
    io(f"{prefix}rewind", "rewind", "payload_rewind")

    def readback(*, postpublish: bool) -> bool:
        nonlocal primary, state
        if primary is not None or handle is None:
            return False
        stem = f"{prefix}{'postpublish_' if postpublish else ''}"
        chunks: list[bytes] = []
        for ordinal, maximum in ((0, len(payload)),):
            if not check(f"{stem}read_chunk", ordinal):
                return False
            try:
                outcome = handle.read(maximum)
                if type(outcome) is not StoreBytesReadOutcome or type(outcome.effect) is not StoreFailureEffect:
                    raise TypeError("read returned wrong exact type")
                validate_effect_response(
                    outcome.effect,
                    expected_operation="read",
                    bytes_requested=maximum,
                )
                success_shape = (
                    outcome.status == "succeeded"
                    and type(outcome.data) is bytes
                    and outcome.failure is None
                    and outcome.effect.completion_status == "succeeded"
                    and outcome.effect.effect_certainty == "reported_effect"
                    and outcome.effect.bytes_requested == maximum
                    and outcome.effect.bytes_reported == len(outcome.data)
                )
                failure_shape = (
                    outcome.status == "failed"
                    and outcome.data is None
                    and isinstance(outcome.failure, ComponentFailureItem)
                    and outcome.effect.completion_status == "failed"
                )
                if not (success_shape or failure_shape):
                    raise TypeError("read returned malformed outcome")
                failure = None
                if failure_shape:
                    failure = require_bound_failure(
                        outcome.failure,
                        name="read",
                        kinds={"store_operation_failure"},
                        stage=(
                            "postpublication_readback"
                            if postpublish else "payload_readback"
                        ),
                        operation_name="read",
                        reasons={
                            "read_failed", "short_read", "extra_byte",
                            "effect_unknown", "store_protocol_failure",
                        },
                        effect=outcome.effect,
                    )
                next_state = _fold_snapshot_effect(
                    state,
                    outcome.effect,
                    payload_length=payload_length,
                    owned_handle=True,
                )
                effects.append(outcome.effect)
                state = next_state
                if failure is not None:
                    primary = _append_failure(
                        primary, follows, failure,
                    )
                    return False
                chunks.append(outcome.data)
            except Exception:
                record_protocol_failure(
                    stage=(
                        "postpublication_readback"
                        if postpublish else "payload_readback"
                    ),
                    operation_name="read",
                    bytes_requested=maximum,
                )
                return False
        if not check(f"{stem}eof"):
            return False
        try:
            eof = handle.read(1)
            if type(eof) is not StoreBytesReadOutcome or type(eof.effect) is not StoreFailureEffect:
                raise TypeError("EOF read returned wrong exact type")
            validate_effect_response(
                eof.effect,
                expected_operation="read",
                bytes_requested=1,
            )
            eof_success_shape = (
                eof.status == "succeeded"
                and type(eof.data) is bytes
                and eof.failure is None
                and eof.effect.completion_status == "succeeded"
                and eof.effect.effect_certainty == "reported_effect"
                and eof.effect.bytes_requested == 1
                and eof.effect.bytes_reported == len(eof.data)
            )
            eof_failure_shape = (
                eof.status == "failed"
                and eof.data is None
                and isinstance(eof.failure, ComponentFailureItem)
                and eof.effect.completion_status == "failed"
            )
            if not (eof_success_shape or eof_failure_shape):
                raise TypeError("EOF read returned malformed outcome")
            failure = None
            if eof_failure_shape:
                failure = require_bound_failure(
                    eof.failure,
                    name="EOF read",
                    kinds={"store_operation_failure"},
                    stage=(
                        "postpublication_readback"
                        if postpublish else "payload_readback"
                    ),
                    operation_name="read",
                    reasons={
                        "read_failed", "short_read", "extra_byte",
                        "effect_unknown", "store_protocol_failure",
                    },
                    effect=eof.effect,
                )
            next_state = _fold_snapshot_effect(
                state,
                eof.effect,
                payload_length=payload_length,
                owned_handle=True,
            )
            effects.append(eof.effect)
            state = next_state
            if failure is not None:
                primary = _append_failure(primary, follows, failure)
                return False
            if eof.data != b"":
                primary = _append_failure(
                    primary, follows,
                    _failure(
                        "validation_failure", "postpublication_readback" if postpublish else "readback_validation",
                        "validate_readback", "extra_byte", resource_kind="store_object_handle",
                        resource_role=_slot_resource_role(slot_kind),
                    ),
                )
                return False
        except Exception:
            record_protocol_failure(
                stage=(
                    "postpublication_readback"
                    if postpublish else "payload_readback"
                ),
                operation_name="read",
                bytes_requested=1,
            )
            return False
        if not check(f"{stem}validate"):
            return False
        if b"".join(chunks) != payload:
            primary = _append_failure(
                primary, follows,
                _failure(
                    "validation_failure", "postpublication_readback" if postpublish else "readback_validation",
                    "validate_readback", "readback_mismatch", resource_kind="store_object_handle",
                    resource_role=_slot_resource_role(slot_kind),
                ),
            )
            return False
        try:
            _validate_retention_payload_bytes(
                payload,
                payload_kind=payload_kind,
                expected_request=expected_request,
                expected_access=expected_access,
                expected_store=expected_store,
                expected_destination=destination,
                expected_authz_binding=expected_authz_binding,
            )
        except Exception:
            primary = _append_failure(
                primary, follows,
                _failure(
                    "validation_failure", "postpublication_readback" if postpublish else "readback_validation",
                    "validate_readback", "semantic_validation_failed",
                    resource_kind="store_object_handle",
                    resource_role=_slot_resource_role(slot_kind),
                ),
            )
            return False
        return True

    def snapshot_and_validate(*, postpublish: bool) -> bool:
        nonlocal primary, state, final_snapshot
        if primary is not None or handle is None:
            return False
        step = f"{prefix}{'postpublish_' if postpublish else ''}snapshot"
        if not check(step):
            return False
        stage = (
            "postpublication_snapshot" if postpublish else "prepublication_snapshot"
        )
        mismatch_reason = "postpublish_mismatch" if postpublish else "snapshot_mismatch"
        expected_path = final_path if postpublish else creating_path
        try:
            snapshot_outcome = handle.snapshot()
            if (
                type(snapshot_outcome) is not StoreSnapshotOutcome
                or type(snapshot_outcome.effect) is not StoreFailureEffect
            ):
                raise TypeError("snapshot returned wrong exact type")
            validate_effect_response(
                snapshot_outcome.effect,
                expected_operation="snapshot",
                bind_before=False,
            )
            snapshot_success = (
                snapshot_outcome.status == "succeeded"
                and snapshot_outcome.snapshot is not None
                and snapshot_outcome.failure is None
                and snapshot_outcome.effect.completion_status == "succeeded"
                and snapshot_outcome.effect.effect_certainty == "reported_effect"
                and snapshot_outcome.effect.before_snapshot
                == snapshot_outcome.snapshot
                and snapshot_outcome.effect.after_snapshot
                == snapshot_outcome.snapshot
            )
            snapshot_failure = (
                snapshot_outcome.status == "failed"
                and snapshot_outcome.snapshot is None
                and isinstance(snapshot_outcome.failure, ComponentFailureItem)
                and snapshot_outcome.effect.completion_status == "failed"
            )
            if snapshot_success:
                snapshot = snapshot_outcome.snapshot
                if snapshot is None:
                    raise TypeError("successful snapshot lacks snapshot")
                _validate_object_snapshot(
                    snapshot,
                    destination,
                    permitted_paths={expected_path},
                )
                state_conflict = _snapshot_disagrees_with_owned_state(state, snapshot)
                payload_conflict = (
                    snapshot.size_bytes != payload_length
                    or snapshot.file_sha256 != payload_sha256
                )
                snapshot_conflict = state_conflict or payload_conflict
                if not snapshot_conflict and postpublish and creation is None:
                    raise PrivatePermissionEvidenceContractError(
                        "published snapshot lacks creation evidence"
                    )
                if snapshot_conflict:
                    next_state = _fold_snapshot_effect(
                        state,
                        snapshot_outcome.effect,
                        payload_length=payload_length,
                        owned_handle=True,
                    )
                    if next_state.effect_certainty != "unknown":
                        next_state = _unknown_state(state, owned_handle=True)
                else:
                    next_state = _known_state(
                        state,
                        snapshot,
                        "published_unverified" if postpublish else "creating_verified",
                    )
                effects.append(snapshot_outcome.effect)
                state = next_state
                if snapshot_conflict:
                    primary = _append_failure(
                        primary,
                        follows,
                        _failure(
                            "validation_failure",
                            stage,
                            "validate_snapshot",
                            mismatch_reason,
                            resource_kind="store_object_handle",
                            resource_role=object_role,
                        ),
                    )
                    return False
                if postpublish:
                    final_snapshot = snapshot
                return True
            if snapshot_failure:
                validate_effect_response(
                    snapshot_outcome.effect,
                    expected_operation="snapshot",
                )
                permitted_reasons = (
                    {"effect_unknown", "store_protocol_failure"}
                    if snapshot_outcome.effect.effect_certainty == "unknown"
                    else {"snapshot_failed", "store_protocol_failure"}
                )
                failure = require_bound_failure(
                    snapshot_outcome.failure,
                    name="postpublish snapshot" if postpublish else "snapshot",
                    kinds={"store_operation_failure"},
                    stage=stage,
                    operation_name="snapshot",
                    reasons=permitted_reasons,
                    effect=snapshot_outcome.effect,
                )
                next_state = _fold_snapshot_effect(
                    state,
                    snapshot_outcome.effect,
                    payload_length=payload_length,
                    owned_handle=True,
                )
                effects.append(snapshot_outcome.effect)
                state = next_state
                primary = _append_failure(primary, follows, failure)
                return False
            raise TypeError("snapshot returned malformed outcome")
        except Exception:
            record_protocol_failure(
                stage=stage,
                operation_name="snapshot",
            )
            return False

    if readback(postpublish=False) and handle is not None and primary is None:
        snapshot_and_validate(postpublish=False)
    if primary is None and handle is not None and check(f"{prefix}publish"):
        try:
            promoted = handle.publish_same_object_exclusive(final_path)
            if type(promoted) is not StorePromoteOutcome or type(promoted.effect) is not StoreFailureEffect:
                raise TypeError("publish returned wrong exact type")
            validate_effect_response(
                promoted.effect,
                expected_operation="publish_same_object_exclusive",
            )
            before = promoted.effect.before_snapshot
            after = promoted.effect.after_snapshot
            same_object = (
                before is not None
                and after is not None
                and before.object_identity == after.object_identity
            )
            published_shape = (
                promoted.status == "published"
                and promoted.failure is None
                and promoted.effect.completion_status == "succeeded"
                and promoted.effect.effect_certainty == "reported_effect"
                and same_object
                and before.relative_path == creating_path
                and after.relative_path == final_path
            )
            collision_shape = (
                promoted.status == "collision"
                and isinstance(promoted.failure, ComponentFailureItem)
                and promoted.effect.completion_status == "collision"
                and promoted.effect.effect_certainty == "no_effect"
                and before is not None
                and before == after
                and before.relative_path == creating_path
                and promoted.effect.collision_relative_path == final_path
            )
            failed_shape = (
                promoted.status == "failed"
                and isinstance(promoted.failure, ComponentFailureItem)
                and promoted.effect.completion_status == "failed"
                and before is not None
                and before.relative_path == creating_path
                and (
                    (
                        promoted.effect.effect_certainty == "no_effect"
                        and before == after
                    )
                    or (
                        promoted.effect.effect_certainty == "reported_effect"
                        and same_object
                        and after.relative_path == final_path
                    )
                    or (
                        promoted.effect.effect_certainty == "unknown"
                        and after is None
                    )
                )
            )
            if not (published_shape or collision_shape or failed_shape):
                raise TypeError("publish returned malformed outcome")
            failure = None
            if collision_shape or failed_shape:
                failure = require_bound_failure(
                    promoted.failure,
                    name="publish",
                    kinds=(
                        {"store_collision"}
                        if collision_shape else {"store_operation_failure"}
                    ),
                    stage="publication",
                    operation_name="publish_same_object_exclusive",
                    reasons=(
                        {"collision"}
                        if collision_shape
                        else {"publish_failed", "effect_unknown", "store_protocol_failure"}
                    ),
                    effect=promoted.effect,
                )
            next_state = _fold_snapshot_effect(
                state,
                promoted.effect,
                payload_length=payload_length,
                owned_handle=True,
            )
            effects.append(promoted.effect)
            state = next_state
            if failure is not None:
                primary = _append_failure(
                    primary, follows, failure,
                )
        except Exception:
            record_protocol_failure(
                stage="publication",
                operation_name="publish_same_object_exclusive",
            )
    if primary is None and handle is not None:
        snapshot_and_validate(postpublish=True)
    if primary is None and handle is not None:
        io(f"{prefix}postpublish_rewind", "rewind", "postpublication_readback")
        if readback(postpublish=True) and final_snapshot is not None:
            state = _known_state(state, final_snapshot, "published_verified_unaccepted")

    if handle is not None:
        effect, state, release_failure = _release_object(handle, state, destination)
        effects.append(effect)
        if release_failure is not None:
            primary = _append_failure(primary, follows, release_failure)
    if namespace is not None:
        effect, release_failure = _release_namespace(namespace, expected_store)
        effects.append(effect)
        if release_failure is not None:
            primary = _append_failure(primary, follows, release_failure)

    validated_failure_draft: RetentionFailureDraft | None = None
    success_draft: _RetentionSuccessDraft | None = None
    if primary is None:
        if final_snapshot is None or creation is None:
            raise PrivatePermissionEvidenceProtocolError(
                "successful physical path lacks final snapshot/creation evidence"
            )
        success_draft = _make_success_draft(
            operation=operation, slot_kind=slot_kind, payload_kind=payload_kind,
            payload=payload, payload_commitment=payload_commitment,
            destination=destination, request=expected_request, access=expected_access,
            store=expected_store, accounting=accounting, effects=effects, state=state,
            final_snapshot=final_snapshot, creation=creation,
        )
        _validate_success_draft(
            success_draft, payload=payload, destination=destination,
            request=expected_request, access=expected_access, store=expected_store,
            final_snapshot=final_snapshot, creation=creation,
            expected_authz_binding=expected_authz_binding,
        )
    else:
        validated_failure_draft = validate_prefinish_failure(primary, accounting)
    if primary is None and not check("final"):
        if primary is None:
            raise PrivatePermissionEvidenceProtocolError("final active failure was not retained")
        validated_failure_draft = validate_prefinish_failure(primary, accounting)
    requested = "succeeded" if primary is None else (
        "refused"
        if primary.failure_kind in {"authority_refusal", "observation_refusal", "store_collision"}
        else "failed"
    )
    accounting, terminal = _finish(lease, requested, accounting)
    if terminal is not None:
        primary = _append_failure(primary, follows, terminal)
    success = (
        primary is None
        and requested == "succeeded"
        and state.publication_state == "published_verified_unaccepted"
        and state.handle_state == "released"
        and final_snapshot is not None
        and success_draft is not None
    )
    if success:
        finished_accounting = _validate_authority_accounting(
            accounting, name="successful retention accounting"
        )
        if (
            finished_accounting["requested_outcome"] != "succeeded"
            or finished_accounting["recorded_outcome"] != "succeeded"
        ):
            raise PrivatePermissionEvidenceProtocolError("successful accounting outcome mismatch")
    elif primary is not None:
        if _primary_slot(slot_kind):
            final_draft = _make_failure_draft(
                destination=destination, request=expected_request, access=expected_access,
                store=expected_store, primary=primary, follows=follows,
                accounting=accounting, effects=effects, state=state,
            )
            _validate_failure_draft(
                final_draft,
                expected_request=expected_request,
                expected_access=expected_access,
                expected_store=expected_store,
                expected_destination=destination,
                expected_payload_bytes=len(payload),
                expected_payload_sha256=sha256_bytes(payload),
                expected_payload_data=payload,
            )
            validated_failure_draft = final_draft
        else:
            _validate_failure_accounting_equations(
                primary, follows, accounting, name="sidecar terminal failure"
            )
            _validate_effect_terminal_fold(
                effects, state, destination=destination, expected_store=expected_store,
                expected_payload_bytes=len(payload),
                expected_payload_sha256=sha256_bytes(payload),
                expected_payload_data=payload,
            )
            _validate_effect_failure_equations(
                effects, primary, follows, destination=destination,
                expected_store=expected_store,
            )
    return _Attempt(
        success, effects, state, accounting, primary, follows, final_snapshot, creation,
        validated_failure_draft, success_draft,
    )


def _receipt(
    *,
    kind: Literal["observation", "observation_failure"],
    attempt: _Attempt,
    data: bytes,
    commitment: str,
    destination: PermissionEvidenceDestination,
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    store: ConfidentialStoreExpectation,
) -> RetainedPermissionEvidence | RetainedPermissionObservationFailure:
    if (
        not attempt.success
        or attempt.final_snapshot is None
        or attempt.creation_evidence is None
        or attempt.success_draft is None
    ):
        raise PrivatePermissionEvidenceProtocolError("receipt requested for failed retention")
    draft = attempt.success_draft
    if (
        draft.payload_kind != kind
        or draft.slot_kind != "primary"
        or draft.payload_commitment != commitment
    ):
        raise PrivatePermissionEvidenceProtocolError("receipt draft payload/slot mismatch")
    _validate_success_draft(
        draft, payload=data, destination=destination, request=request, access=access,
        store=store, final_snapshot=attempt.final_snapshot,
        creation=attempt.creation_evidence,
    )
    cls = RetainedPermissionEvidence if kind == "observation" else RetainedPermissionObservationFailure
    schema = (
        "operatormark.darcy_branch_c_v1.retained_private_permission_evidence.v2"
        if kind == "observation"
        else "operatormark.darcy_branch_c_v1.retained_private_permission_observation_failure.v2"
    )
    unsigned: dict[str, object] = {
        "schema": schema,
        "payload_kind": draft.payload_kind,
        "request_self_sha256": draft.request_self_sha256,
        "access_expectation_self_sha256": draft.access_expectation_self_sha256,
        "destination_self_sha256": draft.destination_self_sha256,
        "store_expectation_self_sha256": draft.store_expectation_self_sha256,
        "relative_path": draft.relative_path,
        "file_bytes": draft.file_bytes,
        "file_sha256": draft.file_sha256,
        "commitment_domain": draft.commitment_domain,
        "authorization_file_sha256": draft.authorization_file_sha256,
        "private_observation_commitment": draft.payload_commitment,
        "namespace_object_identity": draft.namespace_object_identity,
        "records_parent_object_identity": draft.records_parent_object_identity,
        "store_object_identity": draft.store_object_identity,
        "creation_security_descriptor_sha256": draft.creation_security_descriptor_sha256,
        "confidential_store_profile_sha256": draft.confidential_store_profile_sha256,
        "held_object_readback_sha256": draft.held_object_readback_sha256,
        "publication_state": draft.publication_state,
    }
    try:
        signed = with_self_hash(unsigned, "receipt_self_sha256")
    except Exception as exc:
        raise PrivatePermissionEvidenceProtocolError(str(exc)) from exc
    return cls(**signed)  # type: ignore[arg-type,return-value]


def _retention_failure_result(
    *,
    main: _Attempt,
    kind: Literal["observation", "observation_failure"],
    destination: PermissionEvidenceDestination,
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    store_expectation: ConfidentialStoreExpectation,
    authority: PermissionObservationAuthority,
    store: ConfidentialEvidenceStore,
) -> PermissionEvidenceRetentionFailure:
    if main.primary_failure is None:
        raise PrivatePermissionEvidenceProtocolError("failure result lacks primary failure")
    draft = _make_failure_draft(
        destination=destination, request=request, access=access,
        store=store_expectation, primary=main.primary_failure,
        follows=main.follow_on_failures, accounting=main.accounting,
        effects=main.effects, state=main.state,
    )
    if main.validated_failure_draft != draft:
        raise PrivatePermissionEvidenceProtocolError(
            "terminal failure draft was not validated at its required transition"
        )
    _validate_failure_draft(
        draft,
        expected_request=request,
        expected_access=access,
        expected_store=store_expectation,
        expected_destination=destination,
    )
    base: dict[str, object] = {
        "schema": RETENTION_FAILURE_RESULT_SCHEMA,
        "payload_kind": "retention_failure",
        "failed_primary_payload_kind": draft.failed_primary_payload_kind,
        "request_self_sha256": draft.request_self_sha256,
        "access_expectation_self_sha256": draft.access_expectation_self_sha256,
        "destination_self_sha256": draft.destination_self_sha256,
        "store_expectation_self_sha256": draft.store_expectation_self_sha256,
        "primary_failure": draft.primary_failure,
        "ordered_follow_on_failures": draft.ordered_follow_on_failures,
        "main_authority_accounting": draft.main_authority_accounting,
        "payload_effects": draft.payload_effects,
        "payload_object_state": draft.payload_object_state,
        "failure_record_attempted": False,
        "failure_record_primary_failure": None,
        "failure_record_ordered_follow_on_failures": (),
        "failure_record_effects": None,
        "failure_record_object_state": None,
        "failure_record_authority_accounting": None,
        "retained_failure_record_relative_path": None,
        "retained_failure_record_file_bytes": None,
        "retained_failure_record_file_sha256": None,
        "retained_failure_record_commitment_domain": None,
        "retained_failure_record_authorization_file_sha256": None,
        "retained_failure_record_commitment": None,
        "acceptance_emitted": False,
    }
    try:
        sidecar_record = build_private_permission_retention_failure_record(
            draft,
            expected_request=request,
            expected_access=access,
            expected_store=store_expectation,
            expected_destination=destination,
            include_only_pre_sidecar_state=True,
        )
        sidecar_data = canonical_private_permission_retention_failure_bytes(
            sidecar_record,
            expected_request=request,
            expected_access=access,
            expected_store=store_expectation,
            expected_destination=destination,
        )
    except Exception:
        base["failure_record_primary_failure"] = _failure(
            "validation_failure", "retention_failure_build", "build_retention_failure",
            "failure_record_build_failed",
        )
    else:
        sidecar_commitment = private_permission_observation_commitment(
            sidecar_data,
            authorization_file_sha256=_authorization_hash(request),
            payload_kind="retention_failure",
            expected_request=request,
            expected_access=access,
            expected_store=store_expectation,
            expected_destination=destination,
        )
        sidecar = _execute_store_attempt(
            operation="retain_retention_failure",
            slot_kind="retention_failure",
            payload_kind="retention_failure",
            payload=sidecar_data,
            payload_commitment=sidecar_commitment,
            destination=destination,
            expected_request=request,
            expected_access=access,
            expected_store=store_expectation,
            authority=authority,
            store=store,
        )
        base["failure_record_attempted"] = True
        base["failure_record_primary_failure"] = sidecar.primary_failure
        base["failure_record_ordered_follow_on_failures"] = tuple(sidecar.follow_on_failures)
        base["failure_record_effects"] = tuple(sidecar.effects)
        base["failure_record_object_state"] = sidecar.state
        base["failure_record_authority_accounting"] = sidecar.accounting
        if sidecar.success:
            sidecar_hash = sha256_bytes(sidecar_data)
            base.update(
                retained_failure_record_relative_path=destination.retention_failure_accepted_relative_path,
                retained_failure_record_file_bytes=len(sidecar_data),
                retained_failure_record_file_sha256=sidecar_hash,
                retained_failure_record_commitment_domain=RETENTION_FAILURE_DOMAIN,
                retained_failure_record_authorization_file_sha256=_authorization_hash(request),
                retained_failure_record_commitment=sidecar_commitment,
            )
    try:
        signed = with_self_hash(
            _mapping(base, name="retention failure result"), "result_self_sha256"
        )
    except Exception as exc:
        raise PrivatePermissionEvidenceProtocolError(str(exc)) from exc
    result_values = dict(base)
    result_values["result_self_sha256"] = signed["result_self_sha256"]
    result = PermissionEvidenceRetentionFailure(**result_values)  # type: ignore[arg-type]
    _validate_retention_failure_result(
        result,
        draft=draft,
        expected_request=request,
        expected_access=access,
        expected_store=store_expectation,
        expected_destination=destination,
    )
    return result


def _validate_retention_failure_result(
    result: PermissionEvidenceRetentionFailure,
    *,
    draft: PermissionEvidenceRetentionFailureDraft,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: PermissionEvidenceDestination,
) -> None:
    record = _mapping(result, name="retention failure result")
    expected_keys = {
        "schema", "payload_kind", "failed_primary_payload_kind", "request_self_sha256",
        "access_expectation_self_sha256", "destination_self_sha256",
        "store_expectation_self_sha256", "primary_failure",
        "ordered_follow_on_failures", "main_authority_accounting", "payload_effects",
        "payload_object_state", "failure_record_attempted",
        "failure_record_primary_failure", "failure_record_ordered_follow_on_failures",
        "failure_record_effects", "failure_record_object_state",
        "failure_record_authority_accounting", "retained_failure_record_relative_path",
        "retained_failure_record_file_bytes", "retained_failure_record_file_sha256",
        "retained_failure_record_commitment_domain",
        "retained_failure_record_authorization_file_sha256",
        "retained_failure_record_commitment", "acceptance_emitted", "result_self_sha256",
    }
    _exact(record, expected_keys, name="retention failure result")
    if record["schema"] != RETENTION_FAILURE_RESULT_SCHEMA or record["payload_kind"] != "retention_failure":
        raise PrivatePermissionEvidenceProtocolError("retention failure result envelope mismatch")
    if record["acceptance_emitted"] is not False:
        raise PrivatePermissionEvidenceProtocolError("failure result cannot accept")
    _validate_failure_draft(
        draft,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
    )
    draft_record = _mapping(draft, name="validated main failure draft")
    for key in (
        "failed_primary_payload_kind", "request_self_sha256",
        "access_expectation_self_sha256", "destination_self_sha256",
        "store_expectation_self_sha256", "primary_failure",
        "ordered_follow_on_failures", "main_authority_accounting",
        "payload_effects", "payload_object_state", "acceptance_emitted",
    ):
        if record[key] != draft_record[key]:
            raise PrivatePermissionEvidenceProtocolError(
                f"retention failure result changed validated main draft field {key}"
            )
    retained_names = (
        "retained_failure_record_relative_path", "retained_failure_record_file_bytes",
        "retained_failure_record_file_sha256", "retained_failure_record_commitment_domain",
        "retained_failure_record_authorization_file_sha256",
        "retained_failure_record_commitment",
    )
    retained = [record[name] for name in retained_names]
    all_null = all(value is None for value in retained)
    all_non_null = all(value is not None for value in retained)
    if not (all_null or all_non_null):
        raise PrivatePermissionEvidenceProtocolError("retained sidecar fields are partial")
    if type(record["failure_record_attempted"]) is not bool:
        raise PrivatePermissionEvidenceProtocolError(
            "failure_record_attempted must be an exact Boolean"
        )
    if record["failure_record_attempted"] is False:
        if (
            record["failure_record_effects"] is not None
            or record["failure_record_object_state"] is not None
            or record["failure_record_authority_accounting"] is not None
            or not all_null
        ):
            raise PrivatePermissionEvidenceProtocolError("unattempted sidecar has effects")
        failure = _validate_failure_item(
            record["failure_record_primary_failure"],
            name="failure_record_primary_failure",
        )
        if (
            failure["failure_kind"] != "validation_failure"
            or failure["stage"] != "retention_failure_build"
            or failure["component_operation"] != "build_retention_failure"
            or failure["reason_code"] != "failure_record_build_failed"
            or record["failure_record_ordered_follow_on_failures"] != []
        ):
            raise PrivatePermissionEvidenceProtocolError(
                "unattempted sidecar failure equation mismatch"
            )
    else:
        if (
            record["failure_record_effects"] is None
            or record["failure_record_object_state"] is None
            or record["failure_record_authority_accounting"] is None
        ):
            raise PrivatePermissionEvidenceProtocolError("attempted sidecar lacks terminal evidence")
        sidecar_record = build_private_permission_retention_failure_record(
            draft,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=expected_store,
            expected_destination=expected_destination,
            include_only_pre_sidecar_state=True,
        )
        sidecar_data = canonical_private_permission_retention_failure_bytes(
            sidecar_record,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=expected_store,
            expected_destination=expected_destination,
        )
        side_state = _validate_effect_terminal_fold(
            record["failure_record_effects"], record["failure_record_object_state"],
            destination=expected_destination, expected_store=expected_store,
            expected_payload_bytes=len(sidecar_data),
            expected_payload_sha256=sha256_bytes(sidecar_data),
            expected_payload_data=sidecar_data,
        )
        side_primary = record["failure_record_primary_failure"]
        side_follows = record["failure_record_ordered_follow_on_failures"]
        if side_primary is None:
            if not isinstance(side_follows, list) or side_follows:
                raise PrivatePermissionEvidenceProtocolError(
                    "successful sidecar has follow-on failures"
                )
            side_accounting = _validate_authority_accounting(
                record["failure_record_authority_accounting"], name="sidecar accounting"
            )
        else:
            _, _, side_accounting = _validate_failure_accounting_equations(
                side_primary, side_follows,
                record["failure_record_authority_accounting"],
                name="sidecar failure",
            )
            _validate_effect_failure_equations(
                record["failure_record_effects"], side_primary, side_follows,
                destination=expected_destination, expected_store=expected_store,
            )
        if all_non_null:
            if (
                side_primary is not None
                or side_state.publication_state != "published_verified_unaccepted"
                or side_state.handle_state != "released"
                or side_state.ownership != "owned_by_this_invocation"
                or side_state.effect_certainty != "reported_effect"
                or record["retained_failure_record_relative_path"]
                != expected_destination.retention_failure_accepted_relative_path
                or record["retained_failure_record_file_bytes"]
                != side_state.last_confirmed_size_bytes
                or record["retained_failure_record_file_sha256"]
                != side_state.last_confirmed_file_sha256
                or record["retained_failure_record_commitment_domain"] != RETENTION_FAILURE_DOMAIN
                or record["retained_failure_record_authorization_file_sha256"]
                != _authorization_hash(expected_request)
                or side_accounting["acquire_status"] != "granted"
                or side_accounting["lease_candidate_returned"] is not True
                or side_accounting["lease_acquired"] is not True
                or side_accounting["lease_binding_status"] != "valid"
                or side_accounting["preterminal_authority_reason_code"] != "none"
                or side_accounting["finish_attempted"] is not True
                or side_accounting["requested_outcome"] != "succeeded"
                or side_accounting["finish_status"] != "succeeded"
                or side_accounting["effect_certainty"] != "confirmed"
                or side_accounting["recorded_outcome"] != "succeeded"
                or side_accounting["finish_reason_code"] != "none"
            ):
                raise PrivatePermissionEvidenceProtocolError("successful sidecar invariant mismatch")
            expected_commitment = private_permission_observation_commitment(
                sidecar_data,
                authorization_file_sha256=_authorization_hash(expected_request),
                payload_kind="retention_failure",
                expected_request=expected_request,
                expected_access=expected_access,
                expected_store=expected_store,
                expected_destination=expected_destination,
            )
            if (
                record["retained_failure_record_file_bytes"] != len(sidecar_data)
                or record["retained_failure_record_file_sha256"] != sha256_bytes(sidecar_data)
                or record["retained_failure_record_commitment"] != expected_commitment
            ):
                raise PrivatePermissionEvidenceProtocolError(
                    "retained sidecar bytes/hash/domain commitment mismatch"
                )
        elif side_primary is None:
            raise PrivatePermissionEvidenceProtocolError("failed sidecar lacks primary failure")
    try:
        validate_self_hash(record, "result_self_sha256")
    except Exception as exc:
        raise PrivatePermissionEvidenceProtocolError(str(exc)) from exc


def _retain(
    *,
    kind: Literal["observation", "observation_failure"],
    destination: PermissionEvidenceDestination,
    record: Mapping[str, object],
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    authority: PermissionObservationAuthority,
    store: ConfidentialEvidenceStore,
) -> RetainedPermissionEvidence | RetainedPermissionObservationFailure | PermissionEvidenceRetentionFailure:
    _validate_expected_inputs(expected_request, expected_access)
    validate_confidential_store_expectation(expected_store, expected_request=expected_request)
    validate_permission_evidence_destination(
        destination, expected_request=expected_request, expected_store=expected_store
    )
    if destination.payload_kind != kind:
        raise PrivatePermissionEvidenceContractError("destination payload kind mismatch")
    if kind == "observation":
        data = canonical_private_permission_observation_bytes(
            record, expected_request=expected_request, expected_access=expected_access
        )
        operation = "retain_observation"
    else:
        data = canonical_private_permission_observation_failure_bytes(
            record, expected_request=expected_request, expected_access=expected_access
        )
        operation = "retain_observation_failure"
    commitment = private_permission_observation_commitment(
        data,
        authorization_file_sha256=_authorization_hash(expected_request),
        payload_kind=kind,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=None,
        expected_destination=None,
    )
    attempt = _execute_store_attempt(
        operation=operation,
        slot_kind="primary",
        payload_kind=kind,
        payload=data,
        payload_commitment=commitment,
        destination=destination,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        authority=authority,
        store=store,
    )
    if attempt.success:
        return _receipt(
            kind=kind, attempt=attempt, data=data, commitment=commitment,
            destination=destination, request=expected_request, access=expected_access,
            store=expected_store,
        )
    return _retention_failure_result(
        main=attempt, kind=kind, destination=destination, request=expected_request,
        access=expected_access, store_expectation=expected_store,
        authority=authority, store=store,
    )


def retain_private_permission_observation_exclusive(
    destination: PermissionEvidenceDestination,
    record: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    authority: PermissionObservationAuthority,
    store: ConfidentialEvidenceStore,
) -> RetainedPermissionEvidence | PermissionEvidenceRetentionFailure:
    return _retain(
        kind="observation", destination=destination, record=record,
        expected_request=expected_request, expected_access=expected_access,
        expected_store=expected_store, authority=authority, store=store,
    )


def retain_private_permission_observation_failure_exclusive(
    destination: PermissionEvidenceDestination,
    record: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    authority: PermissionObservationAuthority,
    store: ConfidentialEvidenceStore,
) -> RetainedPermissionObservationFailure | PermissionEvidenceRetentionFailure:
    return _retain(
        kind="observation_failure", destination=destination, record=record,
        expected_request=expected_request, expected_access=expected_access,
        expected_store=expected_store, authority=authority, store=store,
    )


def _authz_receipt(
    *,
    attempt: _Attempt,
    data: bytes,
    commitment: str,
    destination: PermissionAuthzEvidenceDestination,
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    store: ConfidentialStoreExpectation,
    binding: authz_records.PermissionAuthzEvidenceBinding,
) -> RetainedPermissionAuthzEvidence:
    if (
        not attempt.success
        or attempt.final_snapshot is None
        or attempt.creation_evidence is None
        or attempt.success_draft is None
    ):
        raise PrivatePermissionEvidenceProtocolError(
            "Authz receipt requested for failed retention"
        )
    draft = attempt.success_draft
    if (
        draft.payload_kind != "authz_evidence"
        or draft.slot_kind != "authz_primary"
        or draft.payload_commitment != commitment
    ):
        raise PrivatePermissionEvidenceProtocolError(
            "Authz receipt draft payload/slot mismatch"
        )
    _validate_success_draft(
        draft,
        payload=data,
        destination=destination,
        request=request,
        access=access,
        store=store,
        final_snapshot=attempt.final_snapshot,
        creation=attempt.creation_evidence,
        expected_authz_binding=binding,
    )
    unsigned: dict[str, object] = {
        "schema": RETAINED_AUTHZ_EVIDENCE_SCHEMA,
        "payload_kind": "authz_evidence",
        "request_self_sha256": draft.request_self_sha256,
        "access_expectation_self_sha256": draft.access_expectation_self_sha256,
        "destination_self_sha256": draft.destination_self_sha256,
        "store_expectation_self_sha256": draft.store_expectation_self_sha256,
        "relative_path": draft.relative_path,
        "file_bytes": draft.file_bytes,
        "file_sha256": draft.file_sha256,
        "commitment_domain": draft.commitment_domain,
        "authorization_file_sha256": draft.authorization_file_sha256,
        "private_authz_commitment": draft.payload_commitment,
        "namespace_object_identity": draft.namespace_object_identity,
        "records_parent_object_identity": draft.records_parent_object_identity,
        "store_object_identity": draft.store_object_identity,
        "creation_security_descriptor_sha256": (
            draft.creation_security_descriptor_sha256
        ),
        "confidential_store_profile_sha256": draft.confidential_store_profile_sha256,
        "held_object_readback_sha256": draft.held_object_readback_sha256,
        "publication_state": draft.publication_state,
    }
    try:
        signed = with_self_hash(unsigned, "receipt_self_sha256")
        receipt = RetainedPermissionAuthzEvidence(**signed)  # type: ignore[arg-type]
    except Exception as exc:
        raise PrivatePermissionEvidenceProtocolError(str(exc)) from exc
    if (
        receipt.publication_state
        != "retained_authz_evidence_only_not_permission_accepted"
        or receipt.private_authz_commitment != commitment
    ):
        raise PrivatePermissionEvidenceProtocolError("Authz receipt invariant mismatch")
    return receipt


def _validate_authz_retention_failure_result(
    result: PermissionAuthzEvidenceRetentionFailure,
    *,
    draft: PermissionAuthzEvidenceRetentionFailureDraft,
    intended_authz_record: bytes,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_destination: PermissionAuthzEvidenceDestination,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding,
) -> None:
    record = _mapping(result, name="Authz retention failure result")
    expected_keys = {
        "schema", "payload_kind", "failed_primary_payload_kind", "request_self_sha256",
        "access_expectation_self_sha256", "destination_self_sha256",
        "store_expectation_self_sha256", "primary_failure",
        "ordered_follow_on_failures", "main_authority_accounting", "payload_effects",
        "payload_object_state", "failure_record_attempted",
        "failure_record_primary_failure", "failure_record_ordered_follow_on_failures",
        "failure_record_effects", "failure_record_object_state",
        "failure_record_authority_accounting", "retained_failure_record_relative_path",
        "retained_failure_record_file_bytes", "retained_failure_record_file_sha256",
        "retained_failure_record_commitment_domain",
        "retained_failure_record_authorization_file_sha256",
        "retained_failure_record_commitment", "acceptance_emitted", "result_self_sha256",
    }
    _exact(record, expected_keys, name="Authz retention failure result")
    if (
        record["schema"] != AUTHZ_RETENTION_FAILURE_RESULT_SCHEMA
        or record["payload_kind"] != "authz_retention_failure"
        or record["failed_primary_payload_kind"] != "authz_evidence"
        or record["acceptance_emitted"] is not False
    ):
        raise PrivatePermissionEvidenceProtocolError(
            "Authz retention failure result envelope mismatch"
        )
    _validate_failure_draft(
        draft,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
        expected_payload_data=intended_authz_record,
    )
    draft_record = _mapping(draft, name="validated Authz main failure draft")
    for key in (
        "failed_primary_payload_kind", "request_self_sha256",
        "access_expectation_self_sha256", "destination_self_sha256",
        "store_expectation_self_sha256", "primary_failure",
        "ordered_follow_on_failures", "main_authority_accounting",
        "payload_effects", "payload_object_state", "acceptance_emitted",
    ):
        if record[key] != draft_record[key]:
            raise PrivatePermissionEvidenceProtocolError(
                f"Authz failure result changed validated main draft field {key}"
            )
    retained_names = (
        "retained_failure_record_relative_path", "retained_failure_record_file_bytes",
        "retained_failure_record_file_sha256", "retained_failure_record_commitment_domain",
        "retained_failure_record_authorization_file_sha256",
        "retained_failure_record_commitment",
    )
    retained = [record[name] for name in retained_names]
    all_null = all(value is None for value in retained)
    all_non_null = all(value is not None for value in retained)
    if not (all_null or all_non_null):
        raise PrivatePermissionEvidenceProtocolError(
            "retained Authz sidecar fields are partial"
        )
    if type(record["failure_record_attempted"]) is not bool:
        raise PrivatePermissionEvidenceProtocolError(
            "Authz failure_record_attempted must be exact Boolean"
        )
    if record["failure_record_attempted"] is False:
        if (
            record["failure_record_effects"] is not None
            or record["failure_record_object_state"] is not None
            or record["failure_record_authority_accounting"] is not None
            or not all_null
        ):
            raise PrivatePermissionEvidenceProtocolError(
                "unattempted Authz sidecar has effects"
            )
        failure = _validate_failure_item(
            record["failure_record_primary_failure"],
            name="Authz failure_record_primary_failure",
        )
        if (
            failure["failure_kind"] != "validation_failure"
            or failure["stage"] != "retention_failure_build"
            or failure["component_operation"] != "build_retention_failure"
            or failure["reason_code"] != "failure_record_build_failed"
            or record["failure_record_ordered_follow_on_failures"] != []
        ):
            raise PrivatePermissionEvidenceProtocolError(
                "unattempted Authz sidecar failure equation mismatch"
            )
    else:
        if (
            record["failure_record_effects"] is None
            or record["failure_record_object_state"] is None
            or record["failure_record_authority_accounting"] is None
        ):
            raise PrivatePermissionEvidenceProtocolError(
                "attempted Authz sidecar lacks terminal evidence"
            )
        sidecar_record = build_private_permission_authz_retention_failure_record(
            draft,
            intended_authz_record=intended_authz_record,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=expected_store,
            expected_destination=expected_destination,
            expected_authz_binding=expected_authz_binding,
            include_only_pre_sidecar_state=True,
        )
        sidecar_data = canonical_private_permission_authz_retention_failure_bytes(
            sidecar_record,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=expected_store,
            expected_destination=expected_destination,
            expected_authz_binding=expected_authz_binding,
        )
        side_state = _validate_effect_terminal_fold(
            record["failure_record_effects"],
            record["failure_record_object_state"],
            destination=expected_destination,
            expected_store=expected_store,
            expected_payload_data=sidecar_data,
        )
        side_primary = record["failure_record_primary_failure"]
        side_follows = record["failure_record_ordered_follow_on_failures"]
        if side_primary is None:
            if not isinstance(side_follows, list) or side_follows:
                raise PrivatePermissionEvidenceProtocolError(
                    "successful Authz sidecar has follow-on failures"
                )
            side_accounting = _validate_authority_accounting(
                record["failure_record_authority_accounting"],
                name="Authz sidecar accounting",
            )
        else:
            _, _, side_accounting = _validate_failure_accounting_equations(
                side_primary,
                side_follows,
                record["failure_record_authority_accounting"],
                name="Authz sidecar failure",
            )
            _validate_effect_failure_equations(
                record["failure_record_effects"],
                side_primary,
                side_follows,
                destination=expected_destination,
                expected_store=expected_store,
            )
        if all_non_null:
            if (
                side_primary is not None
                or side_state.slot_kind != "authz_retention_failure"
                or side_state.publication_state != "published_verified_unaccepted"
                or side_state.handle_state != "released"
                or side_state.ownership != "owned_by_this_invocation"
                or side_state.effect_certainty != "reported_effect"
                or record["retained_failure_record_relative_path"]
                != expected_destination.retention_failure_accepted_relative_path
                or record["retained_failure_record_file_bytes"] != len(sidecar_data)
                or record["retained_failure_record_file_sha256"]
                != sha256_bytes(sidecar_data)
                or record["retained_failure_record_commitment_domain"]
                != AUTHZ_RETENTION_FAILURE_DOMAIN
                or record["retained_failure_record_authorization_file_sha256"]
                != _authorization_hash(expected_request)
                or side_accounting["acquire_status"] != "granted"
                or side_accounting["lease_candidate_returned"] is not True
                or side_accounting["lease_acquired"] is not True
                or side_accounting["lease_binding_status"] != "valid"
                or side_accounting["preterminal_authority_reason_code"] != "none"
                or side_accounting["finish_attempted"] is not True
                or side_accounting["requested_outcome"] != "succeeded"
                or side_accounting["finish_status"] != "succeeded"
                or side_accounting["effect_certainty"] != "confirmed"
                or side_accounting["recorded_outcome"] != "succeeded"
                or side_accounting["finish_reason_code"] != "none"
            ):
                raise PrivatePermissionEvidenceProtocolError(
                    "successful Authz sidecar invariant mismatch"
                )
            commitment = _retention_payload_commitment(
                sidecar_data,
                payload_kind="authz_retention_failure",
                expected_request=expected_request,
                expected_access=expected_access,
                expected_store=expected_store,
                expected_destination=expected_destination,
                expected_authz_binding=expected_authz_binding,
            )
            if record["retained_failure_record_commitment"] != commitment:
                raise PrivatePermissionEvidenceProtocolError(
                    "retained Authz sidecar commitment mismatch"
                )
        elif side_primary is None:
            raise PrivatePermissionEvidenceProtocolError(
                "failed Authz sidecar lacks primary failure"
            )
    try:
        validate_self_hash(record, "result_self_sha256")
    except Exception as exc:
        raise PrivatePermissionEvidenceProtocolError(str(exc)) from exc


def _authz_retention_failure_result(
    *,
    main: _Attempt,
    intended_authz_record: bytes,
    destination: PermissionAuthzEvidenceDestination,
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    store_expectation: ConfidentialStoreExpectation,
    binding: authz_records.PermissionAuthzEvidenceBinding,
    authority: PermissionObservationAuthority,
    store: ConfidentialEvidenceStore,
) -> PermissionAuthzEvidenceRetentionFailure:
    if main.primary_failure is None:
        raise PrivatePermissionEvidenceProtocolError(
            "Authz failure result lacks primary failure"
        )
    draft = _make_failure_draft(
        destination=destination,
        request=request,
        access=access,
        store=store_expectation,
        primary=main.primary_failure,
        follows=main.follow_on_failures,
        accounting=main.accounting,
        effects=main.effects,
        state=main.state,
    )
    if type(draft) is not PermissionAuthzEvidenceRetentionFailureDraft:
        raise PrivatePermissionEvidenceProtocolError(
            "Authz failure draft has the wrong exact type"
        )
    if main.validated_failure_draft != draft:
        raise PrivatePermissionEvidenceProtocolError(
            "Authz terminal failure draft was not validated at its transition"
        )
    _validate_failure_draft(
        draft,
        expected_request=request,
        expected_access=access,
        expected_store=store_expectation,
        expected_destination=destination,
        expected_payload_data=intended_authz_record,
    )
    base: dict[str, object] = {
        "schema": AUTHZ_RETENTION_FAILURE_RESULT_SCHEMA,
        "payload_kind": "authz_retention_failure",
        "failed_primary_payload_kind": "authz_evidence",
        "request_self_sha256": draft.request_self_sha256,
        "access_expectation_self_sha256": draft.access_expectation_self_sha256,
        "destination_self_sha256": draft.destination_self_sha256,
        "store_expectation_self_sha256": draft.store_expectation_self_sha256,
        "primary_failure": draft.primary_failure,
        "ordered_follow_on_failures": draft.ordered_follow_on_failures,
        "main_authority_accounting": draft.main_authority_accounting,
        "payload_effects": draft.payload_effects,
        "payload_object_state": draft.payload_object_state,
        "failure_record_attempted": False,
        "failure_record_primary_failure": None,
        "failure_record_ordered_follow_on_failures": (),
        "failure_record_effects": None,
        "failure_record_object_state": None,
        "failure_record_authority_accounting": None,
        "retained_failure_record_relative_path": None,
        "retained_failure_record_file_bytes": None,
        "retained_failure_record_file_sha256": None,
        "retained_failure_record_commitment_domain": None,
        "retained_failure_record_authorization_file_sha256": None,
        "retained_failure_record_commitment": None,
        "acceptance_emitted": False,
    }
    try:
        sidecar_record = build_private_permission_authz_retention_failure_record(
            draft,
            intended_authz_record=intended_authz_record,
            expected_request=request,
            expected_access=access,
            expected_store=store_expectation,
            expected_destination=destination,
            expected_authz_binding=binding,
            include_only_pre_sidecar_state=True,
        )
        sidecar_data = canonical_private_permission_authz_retention_failure_bytes(
            sidecar_record,
            expected_request=request,
            expected_access=access,
            expected_store=store_expectation,
            expected_destination=destination,
            expected_authz_binding=binding,
        )
    except Exception:
        base["failure_record_primary_failure"] = _failure(
            "validation_failure",
            "retention_failure_build",
            "build_retention_failure",
            "failure_record_build_failed",
        )
    else:
        sidecar_commitment = _retention_payload_commitment(
            sidecar_data,
            payload_kind="authz_retention_failure",
            expected_request=request,
            expected_access=access,
            expected_store=store_expectation,
            expected_destination=destination,
            expected_authz_binding=binding,
        )
        sidecar = _execute_store_attempt(
            operation="retain_authz_retention_failure",
            slot_kind="authz_retention_failure",
            payload_kind="authz_retention_failure",
            payload=sidecar_data,
            payload_commitment=sidecar_commitment,
            destination=destination,
            expected_request=request,
            expected_access=access,
            expected_store=store_expectation,
            authority=authority,
            store=store,
            expected_authz_binding=binding,
        )
        base["failure_record_attempted"] = True
        base["failure_record_primary_failure"] = sidecar.primary_failure
        base["failure_record_ordered_follow_on_failures"] = tuple(
            sidecar.follow_on_failures
        )
        base["failure_record_effects"] = tuple(sidecar.effects)
        base["failure_record_object_state"] = sidecar.state
        base["failure_record_authority_accounting"] = sidecar.accounting
        if sidecar.success:
            base.update(
                retained_failure_record_relative_path=(
                    destination.retention_failure_accepted_relative_path
                ),
                retained_failure_record_file_bytes=len(sidecar_data),
                retained_failure_record_file_sha256=sha256_bytes(sidecar_data),
                retained_failure_record_commitment_domain=(
                    AUTHZ_RETENTION_FAILURE_DOMAIN
                ),
                retained_failure_record_authorization_file_sha256=(
                    _authorization_hash(request)
                ),
                retained_failure_record_commitment=sidecar_commitment,
            )
    try:
        signed = with_self_hash(
            _mapping(base, name="Authz retention failure result"),
            "result_self_sha256",
        )
        result_values = dict(base)
        result_values["result_self_sha256"] = signed["result_self_sha256"]
        result = PermissionAuthzEvidenceRetentionFailure(
            **result_values
        )  # type: ignore[arg-type]
    except Exception as exc:
        raise PrivatePermissionEvidenceProtocolError(str(exc)) from exc
    _validate_authz_retention_failure_result(
        result,
        draft=draft,
        intended_authz_record=intended_authz_record,
        expected_request=request,
        expected_access=access,
        expected_store=store_expectation,
        expected_destination=destination,
        expected_authz_binding=binding,
    )
    return result


def retain_private_permission_authz_exclusive(
    destination: PermissionAuthzEvidenceDestination,
    record: Mapping[str, object],
    *,
    expected_request: PermissionObservationRequest,
    expected_access: PermissionAccessExpectation,
    expected_store: ConfidentialStoreExpectation,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding,
    authority: PermissionObservationAuthority,
    store: ConfidentialEvidenceStore,
) -> RetainedPermissionAuthzEvidence | PermissionAuthzEvidenceRetentionFailure:
    _validate_expected_inputs(expected_request, expected_access)
    authz_records.validate_permission_authz_evidence_binding(
        expected_authz_binding,
        expected_request=expected_request,
        expected_access=expected_access,
    )
    validate_permission_authz_evidence_destination(
        destination,
        expected_request=expected_request,
        expected_store=expected_store,
        expected_authz_binding=expected_authz_binding,
    )
    data = authz_records.canonical_private_permission_authz_bytes(
        record,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_binding=expected_authz_binding,
        expected_destination_self_sha256=destination.destination_self_sha256,
        expected_store_expectation_self_sha256=expected_store.expectation_self_sha256,
    )
    _validate_authz_record_for_retention(
        data,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=destination,
        expected_authz_binding=expected_authz_binding,
    )
    commitment = _retention_payload_commitment(
        data,
        payload_kind="authz_evidence",
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=destination,
        expected_authz_binding=expected_authz_binding,
    )
    attempt = _execute_store_attempt(
        operation="retain_authz_evidence",
        slot_kind="authz_primary",
        payload_kind="authz_evidence",
        payload=data,
        payload_commitment=commitment,
        destination=destination,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        authority=authority,
        store=store,
        expected_authz_binding=expected_authz_binding,
    )
    if attempt.success:
        return _authz_receipt(
            attempt=attempt,
            data=data,
            commitment=commitment,
            destination=destination,
            request=expected_request,
            access=expected_access,
            store=expected_store,
            binding=expected_authz_binding,
        )
    return _authz_retention_failure_result(
        main=attempt,
        intended_authz_record=data,
        destination=destination,
        request=expected_request,
        access=expected_access,
        store_expectation=expected_store,
        binding=expected_authz_binding,
        authority=authority,
        store=store,
    )
