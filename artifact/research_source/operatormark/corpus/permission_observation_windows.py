"""Pure, injected Windows permission-observation contract.

This module deliberately contains no concrete native adapter and performs no work at
import time.  All host interaction is supplied by the caller through the protocols
below; the implementation records only immutable copied values.
"""

from __future__ import annotations

import hashlib
import ntpath
import re
import unicodedata
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Literal, Protocol, cast

from operatormark.corpus.canonical import (
    canonical_json_bytes,
    require_git_oid,
    require_sha256,
    sha256_bytes,
)


PROTOCOL_FILE_SHA256 = "a38ef79b6c4bdccf53f599fa08f77e48a111afbf0b2783ad5733bda99996599a"
REQUEST_SCHEMA = "operatormark.darcy_branch_c_v1.permission_observation_request.v1"
ACCESS_EXPECTATION_SCHEMA = "operatormark.darcy_branch_c_v1.permission_access_expectation.v1"
OPERATION_SCOPE_SCHEMA = "operatormark.darcy_branch_c_v1.permission_operation_scope.v1"
CAPTURE_SCHEMA = "operatormark.darcy_branch_c_v1.private_file_permission_observation.v2"
FAILURE_SCHEMA = "operatormark.darcy_branch_c_v1.private_file_permission_observation_failure.v2"
PERMISSION_MECHANISM = "windows_handle_bound_file_permission_metadata_v1"
OBSERVER_ROLE = "independent_post_protection_observer"
MAXIMUM_ALLOWED_HEX = "0x2000000"
CONTENT_CHUNK_BYTES = 1_048_576

_DWORD_RE = re.compile(r"^0x(?:0|[1-9a-f][0-9a-f]*)$")
_HEX_RE = re.compile(r"^(?:[0-9a-f]{2})*$")
_FILE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_LEGACY_VOLUME_RE = re.compile(r"^[0-9a-f]{8}$")
_LEGACY_INDEX_RE = re.compile(r"^[0-9a-f]{16}$")
_LUID_RE = re.compile(r"^[0-9a-f]{16}$")

_NATIVE_RESOURCE_KINDS = frozenset(
    {
        "directory_handle",
        "file_handle",
        "security_descriptor_buffer",
        "effective_token_handle",
        "access_check_token_handle",
    }
)
_NATIVE_RESOURCE_ROLES = frozenset(
    {
        "ancestor",
        "subject",
        "subject_security",
        "ancestor_security",
        "effective_context_before",
        "effective_context_after",
        "access_check_duplicate",
    }
)
_NATIVE_OPERATIONS = frozenset(
    {
        "open_existing_directory",
        "open_existing_subject",
        "query_object",
        "query_streams",
        "read_unnamed_at",
        "acquire_security_descriptor",
        "parse_security_descriptor",
        "open_current_effective_token",
        "duplicate_for_access_check",
        "query_token",
        "access_check",
    }
)
_NATIVE_OPERATION_STAGES = {
    "open_existing_directory": frozenset({"ancestor_before"}),
    "open_existing_subject": frozenset({"subject_before"}),
    "query_object": frozenset({"ancestor_before", "subject_before", "object_after"}),
    "query_streams": frozenset({"stream_inventory"}),
    "read_unnamed_at": frozenset({"content_read"}),
    "acquire_security_descriptor": frozenset({"security_before", "security_after"}),
    "parse_security_descriptor": frozenset({"security_before", "security_after"}),
    "open_current_effective_token": frozenset({"effective_context_before", "effective_context_after"}),
    "duplicate_for_access_check": frozenset({"access_token"}),
    "query_token": frozenset(
        {
            "effective_context_before",
            "access_token",
            "access_token_after",
            "effective_context_after",
        }
    ),
    "access_check": frozenset({"access_check"}),
}
_OBSERVER_SENSITIVE_STAGES = frozenset(
    {
        "ancestor_before",
        "subject_before",
        "stream_inventory",
        "content_read",
        "security_before",
        "effective_context_before",
        "access_token",
        "access_check",
        "security_after",
        "access_token_after",
        "object_after",
        "effective_context_after",
        "final_authority_check",
    }
)
_REPEATED_AUTHORITY_STAGES = frozenset(
    {
        "ancestor_before",
        "content_read",
        "security_before",
        "access_check",
        "security_after",
        "object_after",
    }
)
_OBJECT_FAILURE_REASONS = frozenset(
    {
        "canonical_path_mismatch",
        "volume_mismatch",
        "identity_changed",
        "size_changed",
        "attributes_changed",
        "current_effective_context_changed",
        "reparse_unsupported",
        "cloud_attribute_unsupported",
        "hard_link_unsupported",
    }
)

Operation = Literal[
    "observe",
    "observe_authz",
    "retain_observation",
    "retain_observation_failure",
    "retain_retention_failure",
    "retain_authz_evidence",
    "retain_authz_retention_failure",
]
FailureOutcome = Literal["refused", "failed"]


class PermissionObservationContractError(RuntimeError):
    """Pure candidate data violates the closed component contract."""


class PermissionObservationProtocolError(RuntimeError):
    """An injected implementation returned a structurally invalid value."""


class PermissionNativeCallError(RuntimeError):
    """Typed native failure with copied evidence and returned resources."""

    def __init__(
        self,
        failure: ComponentFailureItem,
        *,
        raw_prefixes: tuple[NativeRawPrefix, ...] = (),
        returned_resources: tuple[tuple[object, str, str, int | None], ...] = (),
    ) -> None:
        super().__init__(failure.reason_code)
        self.failure = failure
        self.raw_prefixes = raw_prefixes
        self.returned_resources = returned_resources


@dataclass(frozen=True, slots=True)
class PermissionObservationContext:
    protocol_file_sha256: str
    state4_authorization_file_sha256: str
    implementation_commit: str
    implementation_tree: str
    storage_preflight_file_sha256: str
    permission_profile_file_sha256: str
    repository_root_commitment: str
    volume_identifier_commitment: str
    accepted_local_user_residual_principal_commitments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PermissionObservationSubject:
    kind: str
    row_index: int | None
    relative_path: str
    repository_relative_path: str
    expected_file_bytes: int
    expected_file_sha256: str
    canonical_volume_guid_path: str
    required_ancestor_canonical_volume_guid_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PermissionComponentSource:
    observer_module_relative_path: str
    observer_module_file_sha256: str
    observer_module_git_blob_sha1: str
    evidence_module_relative_path: str
    evidence_module_file_sha256: str
    evidence_module_git_blob_sha1: str


@dataclass(frozen=True, slots=True)
class PermissionObservationRequest:
    schema: str
    state: str
    action: str
    attempt_id: str
    permission_mechanism: str
    observer_role: str
    context: PermissionObservationContext
    subject: PermissionObservationSubject
    observation_phase: str
    observation_sequence: int
    source_binding: PermissionComponentSource
    request_self_sha256: str


@dataclass(frozen=True, slots=True)
class NativeOpenSpecification:
    object_kind: Literal["file", "directory"]
    existing_only: bool
    follow_reparse_points: bool
    read_data: bool
    read_attributes: bool
    read_control: bool
    write_data: bool
    append_data: bool
    write_attributes: bool
    delete: bool
    write_dac: bool
    write_owner: bool
    create: bool
    truncate: bool
    share_read: bool
    share_write: bool
    share_delete: bool


NativeResourceBinding = tuple[str, str, int | None]


@dataclass(frozen=True, slots=True)
class NativeCallContext:
    """Non-serialized provenance supplied independently to one native call."""

    request_self_sha256: str
    stage: str
    component_operation: str
    resource_binding: NativeResourceBinding | None
    allowed_returned_resource_binding: NativeResourceBinding | None
    operation_ordinal: int | None
    raw_prefix_start_sequence: int


@dataclass(frozen=True, slots=True)
class NativeGenericMapping:
    generic_read_hex: str
    generic_write_hex: str
    generic_execute_hex: str
    generic_all_hex: str


@dataclass(frozen=True, slots=True)
class PermissionAccessExpectation:
    schema: str
    request_self_sha256: str
    permission_profile_file_sha256: str
    permission_mechanism: str
    desired_access_hex: str
    file_generic_mapping: NativeGenericMapping
    directory_generic_mapping: NativeGenericMapping
    subject_open_specification: NativeOpenSpecification
    ancestor_open_specification: NativeOpenSpecification
    expectation_self_sha256: str


@dataclass(frozen=True, slots=True)
class PermissionOperationScope:
    schema: str
    operation: Operation
    request_self_sha256: str
    access_expectation_self_sha256: str
    destination_self_sha256: str | None
    store_expectation_self_sha256: str | None
    scope_self_sha256: str


@dataclass(frozen=True, slots=True)
class PermissionAuthorityCheckpoint:
    operation: Operation
    step: str
    ordinal: int | None


@dataclass(frozen=True, slots=True)
class AuthorityAcquireOutcome:
    status: Literal["granted", "refused", "failed"]
    lease: PermissionAuthorityLease | None
    failure: ComponentFailureItem | None


@dataclass(frozen=True, slots=True)
class AuthorityCheckOutcome:
    status: Literal["active", "refused", "failed"]
    checkpoint: PermissionAuthorityCheckpoint
    failure: ComponentFailureItem | None


@dataclass(frozen=True, slots=True)
class AuthorityFinishOutcome:
    finish_status: Literal["succeeded", "failed"]
    effect_certainty: Literal["confirmed", "not_applied", "unknown"]
    recorded_outcome: Literal["succeeded", "refused", "failed"] | None
    reason_code: Literal["none", "terminal_finish_failed"]


@dataclass(frozen=True, slots=True)
class AuthorityAccounting:
    acquire_status: Literal["granted", "refused", "failed"]
    lease_candidate_returned: bool
    lease_acquired: bool
    lease_binding_status: Literal["not_applicable", "valid", "invalid"]
    preterminal_authority_reason_code: str
    finish_attempted: bool
    requested_outcome: Literal["succeeded", "refused", "failed"] | None
    finish_status: Literal["not_applicable", "succeeded", "failed"]
    effect_certainty: Literal["not_applicable", "confirmed", "not_applied", "unknown"]
    recorded_outcome: Literal["succeeded", "refused", "failed"] | None
    finish_reason_code: Literal["none", "terminal_finish_failed"]


@dataclass(frozen=True, slots=True)
class NativeFileIdentity:
    volume_serial_number_hex: str
    file_id_128_hex: str
    legacy_volume_serial_number_hex: str
    legacy_file_index_hex: str


@dataclass(frozen=True, slots=True)
class RawNativeObjectSnapshot:
    canonical_volume_guid_path: str
    canonical_dos_path: str
    canonical_name_exact: str
    unicode_nfc_exact: bool
    identity: NativeFileIdentity
    file_attributes_hex: str
    reparse_tag_hex: str
    end_of_file_bytes: int
    allocation_size_bytes: int
    number_of_links: int
    delete_pending: bool
    directory: bool


@dataclass(frozen=True, slots=True)
class NativeObjectSnapshot:
    object_role: Literal["ancestor", "subject"]
    ancestor_index: int | None
    is_repository_root: bool
    is_immediate_parent: bool
    requested_canonical_volume_guid_path: str
    facts: RawNativeObjectSnapshot


@dataclass(frozen=True, slots=True)
class NativeStream:
    stream_index: int
    stream_name: str
    stream_name_utf16le_hex: str
    stream_size_bytes: int
    stream_allocation_size_bytes: int


@dataclass(frozen=True, slots=True)
class NativeRawPrefix:
    capture_sequence: int
    source_operation: str
    resource_kind: str
    resource_role: str
    resource_ordinal: int | None
    encoding: str
    declared_size_bytes: int | None
    captured_size_bytes: int
    complete: bool
    data_hex: str
    captured_bytes_sha256: str


@dataclass(frozen=True, slots=True)
class NativeSid:
    sid_string: str
    sid_bytes_hex: str


@dataclass(frozen=True, slots=True)
class NativeAce:
    ace_index: int
    ace_kind: str
    ace_type_hex: str
    ace_flags_hex: str
    access_mask_hex: str
    trustee_sid: NativeSid
    raw_ace_hex: str


@dataclass(frozen=True, slots=True)
class NativeSecurityDescriptor:
    self_relative_descriptor_hex: str
    self_relative_descriptor_sha256: str
    revision: int
    control_hex: str
    owner_sid: NativeSid
    group_sid: NativeSid
    dacl_present: bool
    dacl_defaulted: bool
    dacl_revision: int
    dacl_protected: bool
    dacl_auto_inherited: bool
    ordered_dacl_aces: tuple[NativeAce, ...]


@dataclass(frozen=True, slots=True)
class BoundSecurityDescriptor:
    object_role: Literal["ancestor", "subject"]
    ancestor_index: int | None
    object_identity: NativeFileIdentity
    canonical_volume_guid_path: str
    descriptor: NativeSecurityDescriptor


@dataclass(frozen=True, slots=True)
class NativeGroup:
    sid: NativeSid
    attributes_hex: str


@dataclass(frozen=True, slots=True)
class NativePrivilege:
    luid_hex: str
    name: str
    attributes_hex: str


@dataclass(frozen=True, slots=True)
class NativeTokenFacts:
    token_type: Literal["primary", "impersonation"]
    impersonation_level: str | None
    token_id_luid_hex: str
    authentication_id_luid_hex: str
    modified_id_luid_hex: str
    session_id: int
    user: NativeSid
    owner: NativeSid
    primary_group: NativeSid
    ordered_groups: tuple[NativeGroup, ...]
    ordered_restricted_sids: tuple[NativeGroup, ...]
    ordered_privileges: tuple[NativePrivilege, ...]
    integrity_level_sid: NativeSid
    elevation_type: Literal["default", "full", "limited"]
    is_elevated: bool
    has_restrictions: bool
    is_app_container: bool
    ordered_capabilities: tuple[NativeGroup, ...]
    virtualization_allowed: bool
    virtualization_enabled: bool
    ui_access: bool


@dataclass(frozen=True, slots=True)
class NativeTokenSnapshot:
    token_role: Literal["effective_context", "access_check_duplicate"]
    selection_source: Literal[
        "thread_impersonation", "process_primary", "duplicated_effective_context"
    ]
    duplicated_from_token_id_luid_hex: str | None
    facts: NativeTokenFacts


@dataclass(frozen=True, slots=True)
class NativeAccessObjectBinding:
    object_role: Literal["subject", "immediate_parent"]
    object_kind: Literal["file", "directory"]
    canonical_volume_guid_path: str
    identity: NativeFileIdentity
    descriptor_sha256: str


@dataclass(frozen=True, slots=True)
class NativeRawAccessCheckResult:
    api_call_status: Literal["succeeded", "failed"]
    access_status: Literal["granted", "denied", "not_evaluated"]
    granted_access_mask_hex: str | None
    access_denial_error_hex: str | None
    api_failure_error_hex: str | None
    ordered_privileges_used: tuple[NativePrivilege, ...] | None


@dataclass(frozen=True, slots=True)
class NativeAccessCheckOutcome:
    object_binding: NativeAccessObjectBinding
    access_token_role: Literal["access_check_duplicate"]
    access_token_id_luid_hex: str
    access_token_modified_id_luid_hex: str
    desired_access_hex: str
    generic_mapping: NativeGenericMapping
    raw_result: NativeRawAccessCheckResult


@dataclass(frozen=True, slots=True)
class NativeResourceEvent:
    resource_kind: str
    resource_role: str
    resource_ordinal: int | None
    acquisition_sequence: int
    acquired: bool
    release_attempted: bool
    released: bool
    release_failure_index: int | None
    object_identity: NativeFileIdentity | None


@dataclass(frozen=True, slots=True)
class ComponentFailureItem:
    failure_kind: str
    stage: str
    component_operation: str
    resource_kind: str | None
    resource_role: str | None
    resource_ordinal: int | None
    operation_ordinal: int | None
    backend_error_hex: str | None
    reason_code: str


@dataclass(frozen=True, slots=True)
class PartialScalarSlot:
    state: Literal["not_reached", "complete", "raw_only"]
    value: object | None
    last_completed_operation: str | None
    raw_prefix_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PartialArraySlot:
    state: Literal["not_reached", "prefix", "complete"]
    value: tuple[object | None, ...]
    last_completed_operation: str | None
    raw_prefix_indices: tuple[int, ...]
    expected_length: int | None


@dataclass(frozen=True, slots=True)
class ContentProgress:
    algorithm: Literal["sha256"]
    bytes_hashed: int
    prefix_sha256: str
    eof_observed: bool


@dataclass(frozen=True, slots=True)
class ContentEvidence:
    algorithm: Literal["sha256"]
    file_bytes: int
    file_sha256: str
    content_interpretation: Literal["opaque_bytes_not_deserialized"]
    eof_observed: bool


@dataclass(frozen=True, slots=True)
class OpenedResourceIdentity:
    resource_kind: Literal["directory_handle", "file_handle"]
    resource_role: Literal["ancestor", "subject"]
    resource_ordinal: int | None
    acquisition_sequence: int
    object_identity: NativeFileIdentity


@dataclass(frozen=True, slots=True)
class PartialCapturedPermissionObservation:
    ancestry_before: PartialArraySlot
    subject_before: PartialScalarSlot
    stream_inventory: PartialScalarSlot
    content_progress: ContentProgress
    subject_security_before: PartialScalarSlot
    ancestry_security_before: PartialArraySlot
    effective_context_before: PartialScalarSlot
    access_check_token_before: PartialScalarSlot
    access_checks: PartialArraySlot
    subject_security_after: PartialScalarSlot
    ancestry_security_after: PartialArraySlot
    access_check_token_after: PartialScalarSlot
    effective_context_after: PartialScalarSlot
    subject_after: PartialScalarSlot
    ancestry_after: PartialArraySlot
    opened_resource_identities: tuple[OpenedResourceIdentity, ...]
    raw_prefixes: tuple[NativeRawPrefix, ...]
    resource_events: tuple[NativeResourceEvent, ...]


@dataclass(frozen=True, slots=True)
class CapturedPermissionObservation:
    schema: str
    request: PermissionObservationRequest
    access_expectation: PermissionAccessExpectation
    outcome: Literal["captured"]
    ancestry_before: tuple[NativeObjectSnapshot, ...]
    subject_before: NativeObjectSnapshot
    stream_inventory: tuple[NativeStream, ...]
    content_evidence: ContentEvidence
    subject_security_before: BoundSecurityDescriptor
    ancestry_security_before: tuple[BoundSecurityDescriptor, ...]
    effective_context_before: NativeTokenSnapshot
    access_check_token_before: NativeTokenSnapshot
    access_checks: tuple[NativeAccessCheckOutcome, NativeAccessCheckOutcome]
    subject_security_after: BoundSecurityDescriptor
    ancestry_security_after: tuple[BoundSecurityDescriptor, ...]
    access_check_token_after: NativeTokenSnapshot
    effective_context_after: NativeTokenSnapshot
    subject_after: NativeObjectSnapshot
    ancestry_after: tuple[NativeObjectSnapshot, ...]
    resource_events: tuple[NativeResourceEvent, ...]
    capture_status: Literal["capture_valid_policy_not_accepted"]
    record_self_sha256: str


@dataclass(frozen=True, slots=True)
class FailedPermissionObservation:
    schema: str
    request: PermissionObservationRequest
    access_expectation: PermissionAccessExpectation
    outcome: FailureOutcome
    primary_failure: ComponentFailureItem
    partial_capture: PartialCapturedPermissionObservation
    ordered_follow_on_failures: tuple[ComponentFailureItem, ...]
    authority_accounting: AuthorityAccounting
    acceptance_emitted: Literal[False]
    record_self_sha256: str


class PermissionObservationAuthority(Protocol):
    def acquire(
        self,
        *,
        request: PermissionObservationRequest,
        scope: PermissionOperationScope,
    ) -> AuthorityAcquireOutcome: ...


class PermissionAuthorityLease(Protocol):
    request_self_sha256: str
    operation: Operation
    access_expectation_self_sha256: str
    destination_self_sha256: str | None
    store_expectation_self_sha256: str | None
    scope_self_sha256: str

    def require_active(
        self, checkpoint: PermissionAuthorityCheckpoint
    ) -> AuthorityCheckOutcome: ...

    def finish(
        self, requested_outcome: Literal["succeeded", "refused", "failed"]
    ) -> AuthorityFinishOutcome: ...


class _HeldTokenAuthzObserverHook(Protocol):
    def __call__(
        self,
        *,
        held_duplicate_token: NativeTokenHandle,
        duplicate_token_before_facts: NativeTokenFacts,
        descriptors: tuple[NativeSecurityDescriptor, NativeSecurityDescriptor],
        object_bindings: tuple[NativeAccessObjectBinding, NativeAccessObjectBinding],
        observation_lease: PermissionAuthorityLease,
    ) -> object: ...


@dataclass(slots=True)
class _HeldTokenAuthzHookState:
    status: Literal["not_invoked", "returned", "failed"] = "not_invoked"
    value: object | None = None
    failure: ComponentFailureItem | None = None


@dataclass(frozen=True, slots=True)
class _ObservedPermissionWithAuthzHook:
    observation_result: CapturedPermissionObservation | FailedPermissionObservation
    hook_status: Literal["not_invoked", "returned", "failed"]
    hook_value: object | None
    hook_failure: ComponentFailureItem | None
    acceptance_emitted: Literal[False] = field(init=False, default=False)


class NativeObjectHandle(Protocol):
    def release(self) -> ComponentFailureItem | None: ...


class NativeBuffer(Protocol):
    def copy_bytes(self) -> bytes: ...

    def release(self) -> ComponentFailureItem | None: ...


class NativeTokenHandle(Protocol):
    def release(self) -> ComponentFailureItem | None: ...


class PermissionNativeApi(Protocol):
    def open_existing_directory(
        self,
        path: str,
        *,
        specification: NativeOpenSpecification,
        context: NativeCallContext,
    ) -> NativeObjectHandle: ...

    def open_existing_subject(
        self,
        path: str,
        *,
        specification: NativeOpenSpecification,
        context: NativeCallContext,
    ) -> NativeObjectHandle: ...

    def query_object(
        self, handle: NativeObjectHandle, *, context: NativeCallContext
    ) -> RawNativeObjectSnapshot: ...

    def query_streams(
        self, handle: NativeObjectHandle, *, context: NativeCallContext
    ) -> tuple[NativeStream, ...]: ...

    def read_unnamed_at(
        self,
        handle: NativeObjectHandle,
        *,
        offset: int,
        requested_bytes: int,
        context: NativeCallContext,
    ) -> bytes: ...

    def acquire_security_descriptor(
        self, handle: NativeObjectHandle, *, context: NativeCallContext
    ) -> NativeBuffer: ...

    def parse_security_descriptor(
        self, data: bytes, *, context: NativeCallContext
    ) -> NativeSecurityDescriptor: ...

    def open_current_effective_token(
        self, *, context: NativeCallContext
    ) -> tuple[Literal["thread_impersonation", "process_primary"], NativeTokenHandle]: ...

    def duplicate_for_access_check(
        self, token: NativeTokenHandle, *, context: NativeCallContext
    ) -> NativeTokenHandle: ...

    def query_token(
        self, token: NativeTokenHandle, *, context: NativeCallContext
    ) -> NativeTokenFacts: ...

    def access_check(
        self,
        *,
        descriptor: bytes,
        token: NativeTokenHandle,
        object_binding: NativeAccessObjectBinding,
        desired_access_hex: str,
        generic_mapping: NativeGenericMapping,
        context: NativeCallContext,
    ) -> NativeRawAccessCheckResult: ...


def _primitive(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_primitive(item) for item in value]
    if isinstance(value, list):
        return [_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    return value


def _self_hash(value: object, field_name: str) -> str:
    mapping = cast(dict[str, object], _primitive(value))
    mapping.pop(field_name, None)
    return sha256_bytes(canonical_json_bytes(mapping))


def _raise(message: str) -> None:
    raise PermissionObservationContractError(message)


def _json_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _raise(f"{name} must be an integer >= {minimum}")
    return value


def _nfc(value: object, *, name: str, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        _raise(f"{name} must be a nonempty string")
    if unicodedata.normalize("NFC", value) != value:
        _raise(f"{name} must be NFC")
    return value


def _sha(value: object, *, name: str) -> str:
    try:
        return require_sha256(value, name=name)
    except Exception as exc:
        raise PermissionObservationContractError(str(exc)) from exc


def _git(value: object, *, name: str) -> str:
    try:
        return require_git_oid(value, name=name)
    except Exception as exc:
        raise PermissionObservationContractError(str(exc)) from exc


def _dword(value: object, *, name: str, nonzero: bool = False) -> int:
    if not isinstance(value, str) or _DWORD_RE.fullmatch(value) is None:
        _raise(f"{name} must be minimal lowercase DWORD hexadecimal")
    parsed = int(value, 16)
    if parsed > 0xFFFFFFFF or (nonzero and parsed == 0):
        _raise(f"{name} is outside its allowed DWORD range")
    return parsed


def _uint64_hex(value: object, *, name: str) -> int:
    if not isinstance(value, str) or _DWORD_RE.fullmatch(value) is None:
        _raise(f"{name} must be minimal lowercase unsigned-64 hexadecimal")
    parsed = int(value, 16)
    if parsed > 0xFFFFFFFFFFFFFFFF:
        _raise(f"{name} is outside its unsigned-64 range")
    return parsed


def _raw_hex(value: object, *, name: str) -> bytes:
    if not isinstance(value, str) or _HEX_RE.fullmatch(value) is None:
        _raise(f"{name} must be lowercase even-length hexadecimal")
    return bytes.fromhex(value)


def _direct_parent(child: str) -> str:
    return ntpath.dirname(child.rstrip("\\"))


def _join_volume(root: str, repository_relative_path: str) -> str:
    return root.rstrip("\\") + "\\" + repository_relative_path.replace("/", "\\")


def _subject_descriptor(observation_sequence: int) -> tuple[str, int | None, str, str]:
    if observation_sequence == 232:
        return (
            "read_only_evidence",
            None,
            "state4_controls/read_only_evidence.json",
            "post_read_only_evidence_protection",
        )
    if observation_sequence == 233:
        return (
            "pre_solve_manifest",
            None,
            "state4_controls/pre_solve_manifest.json",
            "post_pre_solve_manifest_protection",
        )
    if observation_sequence == 234:
        return (
            "completion_marker",
            None,
            "STATE4_CONTROLS_COMPLETE.json",
            "post_completion_marker_protection",
        )
    if 0 <= observation_sequence <= 230:
        if observation_sequence < 193:
            split, offset, groups = "training", observation_sequence, 64
        elif observation_sequence < 206:
            split, offset, groups = "validation", observation_sequence - 193, 4
        else:
            split, offset, groups = "test.sealed", observation_sequence - 206, 8
        if offset == groups * 3:
            return (
                "controls_split_manifest",
                observation_sequence,
                f"state4_controls/{split}/split_manifest.json",
                "post_protection_metadata",
            )
        shard, item = divmod(offset, 3)
        names = (
            (
                "rho",
                f"corpus_staging/{split}/shard_{shard:04d}/rho.float64.npy",
            ),
            (
                "controls_jsonl",
                f"state4_controls/{split}/shard_{shard:04d}.controls.jsonl",
            ),
            (
                "controls_shard_manifest",
                f"state4_controls/{split}/shard_{shard:04d}.manifest.json",
            ),
        )
        kind, relative = names[item]
        return kind, observation_sequence, relative, "post_protection_metadata"
    if observation_sequence == 231:
        return (
            "rng_audit",
            231,
            "state4_controls/rng_audit.json",
            "post_protection_metadata",
        )
    _raise("observation_sequence must be in 0..234")


def expected_state4_permission_subject(
    observation_sequence: int,
    *,
    context: PermissionObservationContext,
    expected_file_bytes: int,
    expected_file_sha256: str,
    repository_root_volume_guid_path: str,
) -> PermissionObservationSubject:
    """Build the one deterministic subject for a State 4 observation sequence."""

    _validate_context(context)
    _json_int(observation_sequence, name="observation_sequence")
    _json_int(expected_file_bytes, name="expected_file_bytes", minimum=1)
    _sha(expected_file_sha256, name="expected_file_sha256")
    root = _nfc(repository_root_volume_guid_path, name="repository_root_volume_guid_path")
    kind, row_index, relative, _ = _subject_descriptor(observation_sequence)
    repository_relative = (
        "artifacts/darcy_branch_c_v1/private_corpus_v1/attempts/"
        f"s4-{context.state4_authorization_file_sha256[:24]}/{relative}"
    )
    subject_path = _join_volume(root, repository_relative)
    parts = repository_relative.split("/")[:-1]
    ancestors = [root.rstrip("\\")]
    current = root.rstrip("\\")
    for part in parts:
        current += "\\" + part
        ancestors.append(current)
    return PermissionObservationSubject(
        kind=kind,
        row_index=row_index,
        relative_path=relative,
        repository_relative_path=repository_relative,
        expected_file_bytes=expected_file_bytes,
        expected_file_sha256=expected_file_sha256,
        canonical_volume_guid_path=subject_path,
        required_ancestor_canonical_volume_guid_paths=tuple(ancestors),
    )


def _validate_context(context: PermissionObservationContext) -> None:
    if type(context) is not PermissionObservationContext:
        _raise("context has the wrong type")
    for name in (
        "protocol_file_sha256",
        "state4_authorization_file_sha256",
        "storage_preflight_file_sha256",
        "permission_profile_file_sha256",
        "repository_root_commitment",
        "volume_identifier_commitment",
    ):
        _sha(getattr(context, name), name=f"context.{name}")
    if context.protocol_file_sha256 != PROTOCOL_FILE_SHA256:
        _raise("protocol_file_sha256 is not the frozen protocol")
    _git(context.implementation_commit, name="context.implementation_commit")
    _git(context.implementation_tree, name="context.implementation_tree")
    residual = context.accepted_local_user_residual_principal_commitments
    if type(residual) is not tuple:
        _raise("residual principal commitments must be a tuple")
    for index, item in enumerate(residual):
        _sha(item, name=f"residual[{index}]")
    if tuple(sorted(set(residual))) != residual:
        _raise("residual principal commitments must be sorted and unique")


def _validate_subject(
    subject: PermissionObservationSubject,
    *,
    sequence: int,
    context: PermissionObservationContext,
) -> None:
    if type(subject) is not PermissionObservationSubject:
        _raise("subject has the wrong type")
    kind, row_index, relative, _ = _subject_descriptor(sequence)
    if (subject.kind, subject.row_index, subject.relative_path) != (
        kind,
        row_index,
        relative,
    ):
        _raise("subject does not match observation sequence")
    _nfc(subject.repository_relative_path, name="subject.repository_relative_path")
    expected_repository_relative = (
        "artifacts/darcy_branch_c_v1/private_corpus_v1/attempts/"
        f"s4-{context.state4_authorization_file_sha256[:24]}/{relative}"
    )
    if subject.repository_relative_path != expected_repository_relative:
        _raise("repository-relative path does not match the frozen subject")
    _json_int(subject.expected_file_bytes, name="subject.expected_file_bytes", minimum=1)
    _sha(subject.expected_file_sha256, name="subject.expected_file_sha256")
    path = _nfc(subject.canonical_volume_guid_path, name="subject.canonical_volume_guid_path")
    ancestors = subject.required_ancestor_canonical_volume_guid_paths
    if type(ancestors) is not tuple or not ancestors:
        _raise("subject must bind at least the repository-root ancestor")
    if len(set(ancestors)) != len(ancestors):
        _raise("ancestor paths must be unique")
    for index, ancestor in enumerate(ancestors):
        _nfc(ancestor, name=f"subject.ancestors[{index}]")
        child = path if index == len(ancestors) - 1 else ancestors[index + 1]
        if _direct_parent(child).casefold() != ancestor.rstrip("\\").casefold():
            _raise("ancestor paths are not direct-parent ordered")


def _validate_source(source: PermissionComponentSource) -> None:
    if type(source) is not PermissionComponentSource:
        _raise("source binding has the wrong type")
    if source.observer_module_relative_path != "operatormark/corpus/permission_observation_windows.py":
        _raise("observer module path mismatch")
    if source.evidence_module_relative_path != "operatormark/corpus/private_permission_evidence.py":
        _raise("evidence module path mismatch")
    _sha(source.observer_module_file_sha256, name="source.observer_module_file_sha256")
    _git(source.observer_module_git_blob_sha1, name="source.observer_module_git_blob_sha1")
    _sha(source.evidence_module_file_sha256, name="source.evidence_module_file_sha256")
    _git(source.evidence_module_git_blob_sha1, name="source.evidence_module_git_blob_sha1")


def _validate_request_self(request: PermissionObservationRequest) -> None:
    if type(request) is not PermissionObservationRequest:
        _raise("request has the wrong type")
    if (
        request.schema != REQUEST_SCHEMA
        or request.state != "state4"
        or request.action != "state4-prepare"
        or request.permission_mechanism != PERMISSION_MECHANISM
        or request.observer_role != OBSERVER_ROLE
    ):
        _raise("request fixed envelope mismatch")
    _validate_context(request.context)
    _json_int(request.observation_sequence, name="request.observation_sequence")
    kind, row_index, relative, phase = _subject_descriptor(request.observation_sequence)
    del kind, row_index, relative
    if request.observation_phase != phase:
        _raise("observation phase mismatch")
    if request.attempt_id != "s4-" + request.context.state4_authorization_file_sha256[:24]:
        _raise("attempt id does not derive from full A4")
    _validate_subject(
        request.subject,
        sequence=request.observation_sequence,
        context=request.context,
    )
    _validate_source(request.source_binding)
    _sha(request.request_self_sha256, name="request.request_self_sha256")
    if request.request_self_sha256 != _self_hash(request, "request_self_sha256"):
        _raise("request self-hash mismatch")


def validate_permission_observation_request(
    request: PermissionObservationRequest,
    *,
    expected_request: PermissionObservationRequest,
) -> None:
    _validate_request_self(expected_request)
    _validate_request_self(request)
    if request != expected_request:
        _raise("request does not equal the independent expected request")


def _validate_mapping(mapping: NativeGenericMapping, *, name: str) -> None:
    if type(mapping) is not NativeGenericMapping:
        _raise(f"{name} has the wrong type")
    for field_name in ("generic_read_hex", "generic_write_hex", "generic_execute_hex", "generic_all_hex"):
        if _dword(getattr(mapping, field_name), name=f"{name}.{field_name}") & 0xF0000000:
            _raise(f"{name}.{field_name} retains generic access bits")


def _expected_open_spec(object_kind: Literal["file", "directory"]) -> NativeOpenSpecification:
    return NativeOpenSpecification(
        object_kind=object_kind,
        existing_only=True,
        follow_reparse_points=False,
        read_data=object_kind == "file",
        read_attributes=True,
        read_control=True,
        write_data=False,
        append_data=False,
        write_attributes=False,
        delete=False,
        write_dac=False,
        write_owner=False,
        create=False,
        truncate=False,
        share_read=True,
        share_write=False,
        share_delete=False,
    )


def _validate_access_self(
    expectation: PermissionAccessExpectation,
    *,
    expected_request: PermissionObservationRequest,
) -> None:
    if type(expectation) is not PermissionAccessExpectation:
        _raise("access expectation has the wrong type")
    if expectation.schema != ACCESS_EXPECTATION_SCHEMA:
        _raise("access expectation schema mismatch")
    if expectation.request_self_sha256 != expected_request.request_self_sha256:
        _raise("access expectation request binding mismatch")
    if expectation.permission_profile_file_sha256 != expected_request.context.permission_profile_file_sha256:
        _raise("access expectation profile binding mismatch")
    if expectation.permission_mechanism != PERMISSION_MECHANISM:
        _raise("access expectation mechanism mismatch")
    if expectation.desired_access_hex != MAXIMUM_ALLOWED_HEX:
        _raise("desired access is not MAXIMUM_ALLOWED")
    _validate_mapping(expectation.file_generic_mapping, name="file_generic_mapping")
    _validate_mapping(expectation.directory_generic_mapping, name="directory_generic_mapping")
    if expectation.subject_open_specification != _expected_open_spec("file"):
        _raise("subject open specification mismatch")
    if expectation.ancestor_open_specification != _expected_open_spec("directory"):
        _raise("ancestor open specification mismatch")
    _sha(expectation.expectation_self_sha256, name="expectation.expectation_self_sha256")
    if expectation.expectation_self_sha256 != _self_hash(expectation, "expectation_self_sha256"):
        _raise("access expectation self-hash mismatch")


def validate_permission_access_expectation(
    expectation: PermissionAccessExpectation,
    *,
    expected_request: PermissionObservationRequest,
) -> None:
    _validate_request_self(expected_request)
    _validate_access_self(expectation, expected_request=expected_request)


def _build_scope(
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    *,
    operation: Literal["observe", "observe_authz"] = "observe",
) -> PermissionOperationScope:
    blank = PermissionOperationScope(
        schema=OPERATION_SCOPE_SCHEMA,
        operation=operation,
        request_self_sha256=request.request_self_sha256,
        access_expectation_self_sha256=access.expectation_self_sha256,
        destination_self_sha256=None,
        store_expectation_self_sha256=None,
        scope_self_sha256="0" * 64,
    )
    return replace(blank, scope_self_sha256=_self_hash(blank, "scope_self_sha256"))


def _failure(
    failure_kind: str,
    stage: str,
    operation: str,
    reason: str,
    *,
    kind: str | None = None,
    role: str | None = None,
    resource_ordinal: int | None = None,
    operation_ordinal: int | None = None,
    backend_error_hex: str | None = None,
) -> ComponentFailureItem:
    return ComponentFailureItem(
        failure_kind=failure_kind,
        stage=stage,
        component_operation=operation,
        resource_kind=kind,
        resource_role=role,
        resource_ordinal=resource_ordinal,
        operation_ordinal=operation_ordinal,
        backend_error_hex=backend_error_hex,
        reason_code=reason,
    )


@dataclass(slots=True)
class _RegisteredResource:
    value: object
    kind: str
    role: str
    ordinal: int | None
    acquisition_sequence: int
    object_identity: NativeFileIdentity | None = None
    release_attempted: bool = False
    released: bool = False
    release_failure: ComponentFailureItem | None = None


@dataclass(slots=True)
class _CaptureState:
    ancestor_count: int
    ancestry_before: list[NativeObjectSnapshot]
    subject_before: NativeObjectSnapshot | None
    streams: tuple[NativeStream, ...] | None
    content: bytearray
    eof_observed: bool
    subject_security_before: BoundSecurityDescriptor | None
    ancestry_security_before: list[BoundSecurityDescriptor]
    effective_before: NativeTokenSnapshot | None
    access_token_before: NativeTokenSnapshot | None
    access_checks: list[NativeAccessCheckOutcome]
    subject_security_after: BoundSecurityDescriptor | None
    ancestry_security_after: list[BoundSecurityDescriptor]
    access_token_after: NativeTokenSnapshot | None
    effective_after: NativeTokenSnapshot | None
    subject_after: NativeObjectSnapshot | None
    ancestry_after: list[NativeObjectSnapshot]
    opened: list[OpenedResourceIdentity]
    raw_prefixes: list[NativeRawPrefix]
    resources: list[_RegisteredResource]


class _StopObservation(Exception):
    def __init__(self, failure: ComponentFailureItem, *, authority_reason: str = "none") -> None:
        super().__init__(failure.reason_code)
        self.failure = failure
        self.authority_reason = authority_reason


class _PolicyRefusal(PermissionObservationContractError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class _NativeReturnStructuralError(PermissionObservationContractError):
    pass


def _register(
    state: _CaptureState,
    value: object,
    kind: str,
    role: str,
    ordinal: int | None,
) -> _RegisteredResource:
    item = _RegisteredResource(
        value=value,
        kind=kind,
        role=role,
        ordinal=ordinal,
        acquisition_sequence=len(state.resources),
    )
    state.resources.append(item)
    return item


def _validate_identity(identity: NativeFileIdentity, *, name: str) -> None:
    if type(identity) is not NativeFileIdentity:
        _raise(f"{name} must be NativeFileIdentity")
    _uint64_hex(
        identity.volume_serial_number_hex,
        name=f"{name}.volume_serial_number_hex",
    )
    if not isinstance(identity.file_id_128_hex, str) or _FILE_ID_RE.fullmatch(identity.file_id_128_hex) is None:
        _raise(f"{name}.file_id_128_hex is malformed")
    if (
        not isinstance(identity.legacy_volume_serial_number_hex, str)
        or _LEGACY_VOLUME_RE.fullmatch(identity.legacy_volume_serial_number_hex) is None
    ):
        _raise(f"{name}.legacy_volume_serial_number_hex is malformed")
    if (
        not isinstance(identity.legacy_file_index_hex, str)
        or _LEGACY_INDEX_RE.fullmatch(identity.legacy_file_index_hex) is None
    ):
        _raise(f"{name}.legacy_file_index_hex is malformed")


def _has_release(value: object) -> bool:
    try:
        return callable(getattr(value, "release"))
    except Exception:
        return False


def _validate_failure_item(value: object) -> ComponentFailureItem:
    if type(value) is not ComponentFailureItem:
        raise PermissionObservationProtocolError("failure item has the wrong type")
    item = cast(ComponentFailureItem, value)
    if not all(
        isinstance(field, str) and field
        for field in (
            item.failure_kind,
            item.stage,
            item.component_operation,
            item.reason_code,
        )
    ):
        raise PermissionObservationProtocolError("failure vocabulary values must be nonempty strings")
    if (item.resource_kind is None) != (item.resource_role is None):
        raise PermissionObservationProtocolError("failure resource kind/role nullability differs")
    if item.resource_kind is not None:
        if (
            not isinstance(item.resource_kind, str)
            or not isinstance(item.resource_role, str)
            or item.resource_kind not in _NATIVE_RESOURCE_KINDS
            or item.resource_role not in _NATIVE_RESOURCE_ROLES
        ):
            raise PermissionObservationProtocolError("failure resource binding is outside the observer vocabulary")
        if (item.resource_role in {"ancestor", "ancestor_security"}) != (item.resource_ordinal is not None):
            raise PermissionObservationProtocolError("failure resource ordinal is inconsistent")
        if item.resource_ordinal is not None and (
            isinstance(item.resource_ordinal, bool)
            or not isinstance(item.resource_ordinal, int)
            or item.resource_ordinal < 0
        ):
            raise PermissionObservationProtocolError("failure resource ordinal is invalid")
    elif item.resource_ordinal is not None:
        raise PermissionObservationProtocolError("unbound failure has a resource ordinal")
    if item.operation_ordinal is not None:
        if (
            isinstance(item.operation_ordinal, bool)
            or not isinstance(item.operation_ordinal, int)
            or item.operation_ordinal < 0
        ):
            raise PermissionObservationProtocolError("failure operation ordinal is invalid")
    repeated_requires_ordinal = (
        item.component_operation
        in {
            "open_existing_directory",
            "read_unnamed_at",
            "acquire_security_descriptor",
            "parse_security_descriptor",
            "access_check",
        }
        or item.component_operation == "query_object"
        and item.stage in {"ancestor_before", "object_after"}
    )
    repeated_forbids_ordinal = item.component_operation in {
        "open_existing_subject",
        "query_streams",
        "open_current_effective_token",
        "duplicate_for_access_check",
        "query_token",
        "release_native_resource",
        "finish_authority_lease",
        "acquire_authority_lease",
        "validate_authority_lease",
        "validate_capture",
    } or item.component_operation == "query_object" and item.stage == "subject_before"
    if repeated_requires_ordinal and item.operation_ordinal is None:
        raise PermissionObservationProtocolError("repeated failure omits its operation ordinal")
    if repeated_forbids_ordinal and item.operation_ordinal is not None:
        raise PermissionObservationProtocolError("non-repeated failure invents an operation ordinal")
    if item.component_operation == "require_active":
        if item.stage not in _REPEATED_AUTHORITY_STAGES and item.operation_ordinal is not None:
            raise PermissionObservationProtocolError("authority failure invents an ordinal at a scalar stage")
    if item.component_operation == "access_check" and item.operation_ordinal not in {0, 1}:
        raise PermissionObservationProtocolError("AccessCheck failure ordinal is outside 0..1")
    if item.component_operation == "query_object" and item.resource_role is not None:
        if item.stage == "subject_before" and item.resource_role != "subject":
            raise PermissionObservationProtocolError("subject-before query failure is not subject-bound")
        if item.stage == "ancestor_before" and (
            item.resource_role != "ancestor"
            or item.operation_ordinal != item.resource_ordinal
        ):
            raise PermissionObservationProtocolError("ancestor-before query failure ordinal is inconsistent")
        if item.stage == "object_after":
            expected = (
                0
                if item.resource_role == "subject"
                else 1 + cast(int, item.resource_ordinal)
                if item.resource_role == "ancestor"
                else None
            )
            if expected is None or item.operation_ordinal != expected:
                raise PermissionObservationProtocolError("object-after query failure ordinal is inconsistent")
    if item.component_operation == "open_existing_directory" and item.resource_role is not None:
        if item.resource_role != "ancestor" or item.operation_ordinal != item.resource_ordinal:
            raise PermissionObservationProtocolError("directory-open failure ordinal is inconsistent")
    if item.component_operation in {"acquire_security_descriptor", "parse_security_descriptor"}:
        expected_security_ordinal: int | None = None
        if item.resource_role in {"subject", "subject_security"}:
            expected_security_ordinal = 0
        elif item.resource_role in {"ancestor", "ancestor_security"}:
            expected_security_ordinal = 1 + cast(int, item.resource_ordinal)
        if item.resource_role is not None and item.operation_ordinal != expected_security_ordinal:
            raise PermissionObservationProtocolError("security failure ordinal is inconsistent")
    if item.backend_error_hex is not None:
        try:
            _dword(item.backend_error_hex, name="failure.backend_error_hex", nonzero=True)
        except PermissionObservationContractError as exc:
            raise PermissionObservationProtocolError("failure backend error is invalid") from exc
    if item.backend_error_hex is not None and item.failure_kind not in {
        "native_api_failure",
        "cleanup_failure",
    }:
        raise PermissionObservationProtocolError("failure kind cannot carry a backend error")

    resource_bindings = {
        "open_existing_directory": frozenset({("directory_handle", "ancestor")}),
        "open_existing_subject": frozenset({("file_handle", "subject")}),
        "query_object": frozenset(
            {("directory_handle", "ancestor"), ("file_handle", "subject")}
        ),
        "query_streams": frozenset({("file_handle", "subject")}),
        "read_unnamed_at": frozenset({("file_handle", "subject")}),
        "acquire_security_descriptor": frozenset(
            {
                ("directory_handle", "ancestor"),
                ("file_handle", "subject"),
                ("security_descriptor_buffer", "ancestor_security"),
                ("security_descriptor_buffer", "subject_security"),
            }
        ),
        "parse_security_descriptor": frozenset(
            {
                ("security_descriptor_buffer", "ancestor_security"),
                ("security_descriptor_buffer", "subject_security"),
            }
        ),
        "open_current_effective_token": frozenset(
            {
                ("effective_token_handle", "effective_context_before"),
                ("effective_token_handle", "effective_context_after"),
            }
        ),
        "duplicate_for_access_check": frozenset(
            {("effective_token_handle", "effective_context_before")}
        ),
        "query_token": frozenset(
            {
                ("effective_token_handle", "effective_context_before"),
                ("effective_token_handle", "effective_context_after"),
                ("access_check_token_handle", "access_check_duplicate"),
            }
        ),
        "access_check": frozenset(
            {("access_check_token_handle", "access_check_duplicate")}
        ),
    }
    if item.resource_kind is not None and item.component_operation in resource_bindings:
        if (item.resource_kind, cast(str, item.resource_role)) not in resource_bindings[item.component_operation]:
            raise PermissionObservationProtocolError("failure resource is incompatible with its operation")

    if item.failure_kind == "native_protocol_failure":
        if (
            item.reason_code != "native_protocol_failure"
            or item.component_operation not in _NATIVE_OPERATIONS
            or item.backend_error_hex is not None
        ):
            raise PermissionObservationProtocolError("native protocol failure is inconsistent")
        if item.stage not in _NATIVE_OPERATION_STAGES[item.component_operation]:
            raise PermissionObservationProtocolError("native protocol failure stage is inconsistent")
    elif item.failure_kind == "native_api_failure":
        if (
            item.reason_code != "native_api_failure"
            or item.component_operation not in _NATIVE_OPERATIONS
            or item.component_operation == "parse_security_descriptor"
        ):
            raise PermissionObservationProtocolError("native API failure is inconsistent")
        if item.stage not in _NATIVE_OPERATION_STAGES[item.component_operation]:
            raise PermissionObservationProtocolError("native API failure stage is inconsistent")
    elif item.failure_kind == "cleanup_failure":
        if (
            item.stage != "resource_cleanup"
            or item.component_operation != "release_native_resource"
            or item.reason_code != "resource_release_failed"
            or item.resource_kind is None
            or item.operation_ordinal is not None
        ):
            raise PermissionObservationProtocolError("cleanup failure is inconsistent")
    elif item.failure_kind == "terminal_accounting_failure":
        if (
            item.stage != "terminal_accounting"
            or item.component_operation != "finish_authority_lease"
            or item.reason_code != "terminal_finish_failed"
            or item.resource_kind is not None
            or item.backend_error_hex is not None
        ):
            raise PermissionObservationProtocolError("terminal failure is inconsistent")
    elif item.failure_kind == "lease_binding_failure":
        if (
            item.stage != "lease_binding"
            or item.component_operation != "validate_authority_lease"
            or item.reason_code
            not in {
                "lease_malformed",
                "lease_request_mismatch",
                "lease_operation_mismatch",
                "lease_access_expectation_mismatch",
                "lease_destination_mismatch",
                "lease_store_expectation_mismatch",
                "lease_scope_mismatch",
            }
            or item.resource_kind is not None
            or item.backend_error_hex is not None
        ):
            raise PermissionObservationProtocolError("lease failure is inconsistent")
    elif item.failure_kind in {"authority_refusal", "authority_failure"}:
        acquisition = (
            "authority_refusal",
            "authority_acquisition",
            "acquire_authority_lease",
            "authority_acquire_refused",
        ) if item.failure_kind == "authority_refusal" else (
            "authority_failure",
            "authority_acquisition",
            "acquire_authority_lease",
            "authority_acquire_failed",
        )
        check_reason = (
            "authority_inactive"
            if item.failure_kind == "authority_refusal"
            else "authority_check_failed"
        )
        valid = (
            (item.failure_kind, item.stage, item.component_operation, item.reason_code)
            == acquisition
            or item.component_operation == "require_active"
            and item.stage in _OBSERVER_SENSITIVE_STAGES
            and item.reason_code == check_reason
        )
        if not valid or item.resource_kind is not None or item.backend_error_hex is not None:
            label = "authority refusal" if item.failure_kind == "authority_refusal" else "authority failure"
            raise PermissionObservationProtocolError(f"{label} matrix is inconsistent")
    elif item.failure_kind == "observation_refusal":
        allowed = (
            item.component_operation in {"open_existing_directory", "open_existing_subject"}
            and item.stage in _NATIVE_OPERATION_STAGES[item.component_operation]
            and item.reason_code
            in {
                "canonical_path_mismatch",
                "volume_mismatch",
                "reparse_unsupported",
                "cloud_attribute_unsupported",
            }
            or item.component_operation == "query_object"
            and item.stage in _NATIVE_OPERATION_STAGES["query_object"]
            and item.reason_code in _OBJECT_FAILURE_REASONS
            or item.component_operation == "query_streams"
            and item.stage == "stream_inventory"
            and item.reason_code == "named_stream_unsupported"
            or item.component_operation == "acquire_security_descriptor"
            and item.stage in _NATIVE_OPERATION_STAGES["acquire_security_descriptor"]
            and item.reason_code == "descriptor_incomplete"
            or item.component_operation == "parse_security_descriptor"
            and item.stage in _NATIVE_OPERATION_STAGES["parse_security_descriptor"]
            and item.reason_code
            in {"descriptor_incomplete", "ace_unsupported", "raw_prefix_malformed"}
            or item.component_operation
            in {
                "open_current_effective_token",
                "duplicate_for_access_check",
                "query_token",
            }
            and item.stage in _NATIVE_OPERATION_STAGES[item.component_operation]
            and item.reason_code
            in {"token_unsupported", "token_changed", "current_effective_context_changed"}
        )
        if not allowed or item.backend_error_hex is not None:
            raise PermissionObservationProtocolError("authority refusal matrix is inconsistent")
    elif item.failure_kind == "validation_failure":
        allowed = (
            item.component_operation == "query_object"
            and item.stage in _NATIVE_OPERATION_STAGES["query_object"]
            and item.reason_code in _OBJECT_FAILURE_REASONS
            or item.component_operation == "read_unnamed_at"
            and item.stage == "content_read"
            and item.reason_code in {"short_read", "content_grew", "content_hash_mismatch"}
            or item.component_operation
            in {
                "open_current_effective_token",
                "duplicate_for_access_check",
                "query_token",
            }
            and item.stage in _NATIVE_OPERATION_STAGES[item.component_operation]
            and item.reason_code in {"token_changed", "current_effective_context_changed"}
            or item.component_operation == "parse_security_descriptor"
            and item.stage in _NATIVE_OPERATION_STAGES["parse_security_descriptor"]
            and item.reason_code
            in {"descriptor_incomplete", "ace_unsupported", "raw_prefix_malformed"}
            or item.component_operation == "access_check"
            and item.stage == "access_check"
            and item.reason_code
            in {
                "access_binding_mismatch",
                "access_token_binding_mismatch",
                "generic_mapping_mismatch",
                "access_result_inconsistent",
            }
            or item.component_operation == "validate_capture"
            and item.stage in {"capture_assembly", "capture_validation"}
            and item.reason_code
            in {
                "capture_validation_failed",
                "canonical_path_mismatch",
                "volume_mismatch",
                "identity_changed",
                "size_changed",
                "attributes_changed",
                "descriptor_changed",
                "token_changed",
                "current_effective_context_changed",
                "content_hash_mismatch",
                "resource_release_failed",
            }
        )
        if not allowed or item.backend_error_hex is not None:
            raise PermissionObservationProtocolError("validation failure matrix is inconsistent")
    else:
        raise PermissionObservationProtocolError("failure kind is outside the observer vocabulary")
    return item


def _validate_prefix(prefix: object, *, expected_sequence: int) -> NativeRawPrefix:
    if type(prefix) is not NativeRawPrefix:
        raise PermissionObservationProtocolError("raw prefix has the wrong type")
    item = cast(NativeRawPrefix, prefix)
    if (
        isinstance(item.capture_sequence, bool)
        or not isinstance(item.capture_sequence, int)
        or item.capture_sequence != expected_sequence
        or not isinstance(item.source_operation, str)
        or item.source_operation not in _NATIVE_OPERATIONS
    ):
        raise PermissionObservationProtocolError("raw prefix sequence/operation is inconsistent")
    if (
        not isinstance(item.resource_kind, str)
        or not isinstance(item.resource_role, str)
        or item.resource_kind not in _NATIVE_RESOURCE_KINDS
        or item.resource_role not in _NATIVE_RESOURCE_ROLES
    ):
        raise PermissionObservationProtocolError("raw prefix resource provenance is invalid")
    if (item.resource_role in {"ancestor", "ancestor_security"}) != (item.resource_ordinal is not None):
        raise PermissionObservationProtocolError("raw prefix ordinal is inconsistent")
    if item.resource_ordinal is not None and (
        isinstance(item.resource_ordinal, bool)
        or not isinstance(item.resource_ordinal, int)
        or item.resource_ordinal < 0
    ):
        raise PermissionObservationProtocolError("raw prefix resource ordinal is invalid")
    if not isinstance(item.encoding, str) or item.encoding not in {
        "security_descriptor",
        "ace",
        "sid",
        "token_information",
        "stream_information",
        "object_information",
    }:
        raise PermissionObservationProtocolError("raw prefix encoding is invalid")
    try:
        data = _raw_hex(item.data_hex, name="raw_prefix.data_hex")
    except PermissionObservationContractError as exc:
        raise PermissionObservationProtocolError("raw prefix hex is invalid") from exc
    if (
        isinstance(item.captured_size_bytes, bool)
        or not isinstance(item.captured_size_bytes, int)
        or item.captured_size_bytes != len(data)
        or item.captured_bytes_sha256 != sha256_bytes(data)
        or type(item.complete) is not bool
    ):
        raise PermissionObservationProtocolError("raw prefix byte evidence is inconsistent")
    if item.declared_size_bytes is not None and (
        isinstance(item.declared_size_bytes, bool)
        or not isinstance(item.declared_size_bytes, int)
        or item.declared_size_bytes < 0
        or item.declared_size_bytes < item.captured_size_bytes
    ):
        raise PermissionObservationProtocolError("raw prefix declared size is invalid")
    return item


def _sid_text(data: bytes) -> str:
    if len(data) < 8:
        _raise("SID bytes are truncated")
    revision = data[0]
    count = data[1]
    if len(data) != 8 + 4 * count:
        _raise("SID byte count is inconsistent")
    authority = int.from_bytes(data[2:8], "big")
    subauthorities = [
        str(int.from_bytes(data[8 + 4 * index : 12 + 4 * index], "little"))
        for index in range(count)
    ]
    suffix = "" if not subauthorities else "-" + "-".join(subauthorities)
    return f"S-{revision}-{authority}{suffix}"


def _resource_failure(
    reason: str,
    *,
    stage: str,
    operation: str,
    resource: _RegisteredResource | None = None,
    failure_kind: str = "validation_failure",
    operation_ordinal: int | None = None,
) -> _StopObservation:
    return _StopObservation(
        _failure(
            failure_kind,
            stage,
            operation,
            reason,
            kind=None if resource is None else resource.kind,
            role=None if resource is None else resource.role,
            resource_ordinal=None if resource is None else resource.ordinal,
            operation_ordinal=operation_ordinal,
        )
    )


def _label_object(
    raw: object,
    *,
    role: Literal["ancestor", "subject"],
    ancestor_index: int | None,
    ancestor_count: int,
    expected_path: str,
    expected_bytes: int | None,
    stage: str,
    operation_ordinal: int | None,
    resource: _RegisteredResource,
) -> NativeObjectSnapshot:
    if type(raw) is not RawNativeObjectSnapshot:
        raise _resource_failure(
            "native_protocol_failure",
            stage=stage,
            operation="query_object",
            resource=resource,
            failure_kind="native_protocol_failure",
            operation_ordinal=operation_ordinal,
        )
    try:
        _nfc(raw.canonical_volume_guid_path, name="object.canonical_volume_guid_path")
        _nfc(raw.canonical_dos_path, name="object.canonical_dos_path")
        _nfc(raw.canonical_name_exact, name="object.canonical_name_exact")
        _validate_identity(raw.identity, name="object.identity")
        attributes = _dword(raw.file_attributes_hex, name="object.file_attributes_hex")
        tag = _dword(raw.reparse_tag_hex, name="object.reparse_tag_hex")
        _json_int(raw.end_of_file_bytes, name="object.end_of_file_bytes")
        _json_int(raw.allocation_size_bytes, name="object.allocation_size_bytes")
        _json_int(raw.number_of_links, name="object.number_of_links")
        for name in ("unicode_nfc_exact", "delete_pending", "directory"):
            if type(getattr(raw, name)) is not bool:
                _raise(f"object.{name} must be Boolean")
        if raw.allocation_size_bytes < raw.end_of_file_bytes:
            _raise("object allocation is smaller than end of file")
    except PermissionObservationContractError:
        raise _resource_failure(
            "native_protocol_failure",
            stage=stage,
            operation="query_object",
            resource=resource,
            failure_kind="native_protocol_failure",
            operation_ordinal=operation_ordinal,
        )
    if raw.canonical_volume_guid_path != expected_path or not raw.unicode_nfc_exact:
        raise _resource_failure(
            "canonical_path_mismatch",
            stage=stage,
            operation="query_object",
            resource=resource,
            failure_kind="observation_refusal",
            operation_ordinal=operation_ordinal,
        )
    if raw.delete_pending:
        raise _resource_failure(
            "attributes_changed",
            stage=stage,
            operation="query_object",
            resource=resource,
            failure_kind="observation_refusal",
            operation_ordinal=operation_ordinal,
        )
    if tag != 0 or attributes & (0x400 | 0x1000 | 0x40000 | 0x400000):
        reason = "reparse_unsupported" if tag != 0 or attributes & 0x400 else "cloud_attribute_unsupported"
        raise _resource_failure(
            reason,
            stage=stage,
            operation="query_object",
            resource=resource,
            failure_kind="observation_refusal",
            operation_ordinal=operation_ordinal,
        )
    if role == "subject":
        if raw.directory:
            raise _resource_failure(
                "attributes_changed",
                stage=stage,
                operation="query_object",
                resource=resource,
                failure_kind="observation_refusal",
                operation_ordinal=operation_ordinal,
            )
        if raw.number_of_links != 1:
            raise _resource_failure(
                "hard_link_unsupported",
                stage=stage,
                operation="query_object",
                resource=resource,
                failure_kind="observation_refusal",
                operation_ordinal=operation_ordinal,
            )
        if expected_bytes is not None and raw.end_of_file_bytes != expected_bytes:
            raise _resource_failure(
                "size_changed",
                stage=stage,
                operation="query_object",
                resource=resource,
                operation_ordinal=operation_ordinal,
            )
    elif not raw.directory:
        raise _resource_failure(
            "attributes_changed",
            stage=stage,
            operation="query_object",
            resource=resource,
            failure_kind="observation_refusal",
            operation_ordinal=operation_ordinal,
        )
    snapshot = NativeObjectSnapshot(
        object_role=role,
        ancestor_index=ancestor_index,
        is_repository_root=role == "ancestor" and ancestor_index == 0,
        is_immediate_parent=role == "ancestor" and ancestor_index == ancestor_count - 1,
        requested_canonical_volume_guid_path=expected_path,
        facts=raw,
    )
    resource.object_identity = raw.identity
    return snapshot


def _validate_streams(
    value: object,
    *,
    subject: NativeObjectSnapshot,
    resource: _RegisteredResource,
) -> tuple[NativeStream, ...]:
    if type(value) is not tuple or any(type(item) is not NativeStream for item in value):
        raise _resource_failure(
            "native_protocol_failure",
            stage="stream_inventory",
            operation="query_streams",
            resource=resource,
            failure_kind="native_protocol_failure",
        )
    streams = cast(tuple[NativeStream, ...], value)
    if len(streams) != 1:
        raise _resource_failure(
            "named_stream_unsupported",
            stage="stream_inventory",
            operation="query_streams",
            resource=resource,
            failure_kind="observation_refusal",
        )
    stream = streams[0]
    try:
        _nfc(stream.stream_name, name="stream.stream_name")
        encoded_name = _raw_hex(
            stream.stream_name_utf16le_hex,
            name="stream.stream_name_utf16le_hex",
        )
        _json_int(stream.stream_index, name="stream.stream_index")
        _json_int(stream.stream_size_bytes, name="stream.stream_size_bytes")
        _json_int(
            stream.stream_allocation_size_bytes,
            name="stream.stream_allocation_size_bytes",
        )
        if (
            stream.stream_index != 0
            or encoded_name != stream.stream_name.encode("utf-16le")
            or stream.stream_allocation_size_bytes < stream.stream_size_bytes
        ):
            _raise("stream return is structurally inconsistent")
    except (PermissionObservationContractError, UnicodeError):
        raise _resource_failure(
            "native_protocol_failure",
            stage="stream_inventory",
            operation="query_streams",
            resource=resource,
            failure_kind="native_protocol_failure",
        )
    if stream.stream_name != "::$DATA":
        raise _resource_failure(
            "named_stream_unsupported",
            stage="stream_inventory",
            operation="query_streams",
            resource=resource,
            failure_kind="observation_refusal",
        )
    if stream.stream_size_bytes != subject.facts.end_of_file_bytes:
        raise _resource_failure(
            "native_protocol_failure",
            stage="stream_inventory",
            operation="query_streams",
            resource=resource,
            failure_kind="native_protocol_failure",
        )
    return streams


def _validate_sid(value: NativeSid, *, name: str) -> None:
    if (
        type(value) is not NativeSid
        or not isinstance(value.sid_string, str)
        or not value.sid_string.startswith("S-")
    ):
        _raise(f"{name} is not a complete SID")
    data = _raw_hex(value.sid_bytes_hex, name=f"{name}.sid_bytes_hex")
    if not data or _sid_text(data) != value.sid_string:
        _raise(f"{name} text does not reproduce its SID bytes")


def _sid_wire_length(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 8 > len(data):
        _raise("descriptor SID offset is outside copied bytes")
    length = 8 + 4 * data[offset + 1]
    if offset + length > len(data):
        _raise("descriptor SID is truncated")
    return length


def _validate_descriptor(value: NativeSecurityDescriptor) -> None:
    if type(value) is not NativeSecurityDescriptor:
        _raise("descriptor return has the wrong type")
    raw = _raw_hex(value.self_relative_descriptor_hex, name="descriptor.self_relative_descriptor_hex")
    _sha(value.self_relative_descriptor_sha256, name="descriptor.self_relative_descriptor_sha256")
    if sha256_bytes(raw) != value.self_relative_descriptor_sha256:
        _raise("descriptor hash mismatch")
    _json_int(value.revision, name="descriptor.revision")
    control = _dword(value.control_hex, name="descriptor.control_hex")
    _validate_sid(value.owner_sid, name="descriptor.owner_sid")
    _validate_sid(value.group_sid, name="descriptor.group_sid")
    for name in (
        "dacl_present",
        "dacl_defaulted",
        "dacl_protected",
        "dacl_auto_inherited",
    ):
        if type(getattr(value, name)) is not bool:
            _raise(f"descriptor.{name} must be Boolean")
    if (
        isinstance(value.dacl_revision, bool)
        or not isinstance(value.dacl_revision, int)
        or value.dacl_revision < 0
    ):
        _raise("descriptor DACL revision is malformed")
    if type(value.ordered_dacl_aces) is not tuple:
        _raise("descriptor ACEs must be a tuple")
    if len(raw) < 20 or raw[0] != value.revision:
        _raise("self-relative descriptor header is truncated or inconsistent")
    if int.from_bytes(raw[2:4], "little") != control or control & 0x8000 == 0:
        _raise("self-relative descriptor control is inconsistent")
    owner_offset = int.from_bytes(raw[4:8], "little")
    group_offset = int.from_bytes(raw[8:12], "little")
    dacl_offset = int.from_bytes(raw[16:20], "little")
    owner_length = _sid_wire_length(raw, owner_offset)
    group_length = _sid_wire_length(raw, group_offset)
    if raw[owner_offset : owner_offset + owner_length].hex() != value.owner_sid.sid_bytes_hex:
        _raise("descriptor owner SID bytes differ from typed owner")
    if raw[group_offset : group_offset + group_length].hex() != value.group_sid.sid_bytes_hex:
        _raise("descriptor group SID bytes differ from typed group")
    if (
        value.dacl_present != bool(control & 0x0004)
        or value.dacl_defaulted != bool(control & 0x0008)
        or value.dacl_protected != bool(control & 0x1000)
        or value.dacl_auto_inherited != bool(control & 0x0400)
    ):
        _raise("descriptor DACL control flags differ from typed fields")
    if not value.dacl_present:
        if dacl_offset != 0 or value.ordered_dacl_aces:
            _raise("absent descriptor DACL contradicts copied bytes")
        raise _PolicyRefusal("descriptor_incomplete")
    if dacl_offset == 0:
        if value.ordered_dacl_aces:
            _raise("null descriptor DACL has typed ACEs")
        raise _PolicyRefusal("descriptor_incomplete")
    if dacl_offset + 8 > len(raw):
        _raise("descriptor DACL is truncated")
    acl_size = int.from_bytes(raw[dacl_offset + 2 : dacl_offset + 4], "little")
    ace_count = int.from_bytes(raw[dacl_offset + 4 : dacl_offset + 6], "little")
    if (
        raw[dacl_offset] != value.dacl_revision
        or dacl_offset + acl_size > len(raw)
        or ace_count != len(value.ordered_dacl_aces)
    ):
        _raise("descriptor DACL header is inconsistent")
    for index, ace in enumerate(value.ordered_dacl_aces):
        if (
            type(ace) is not NativeAce
            or isinstance(ace.ace_index, bool)
            or not isinstance(ace.ace_index, int)
            or ace.ace_index != index
        ):
            _raise("descriptor ACE order mismatch")
        if not isinstance(ace.ace_kind, str):
            _raise("ACE kind has the wrong type")
        if ace.ace_kind not in {"allow", "deny"}:
            raise _PolicyRefusal("ace_unsupported")
        ace_type = _dword(ace.ace_type_hex, name="ace.ace_type_hex")
        ace_flags = _dword(ace.ace_flags_hex, name="ace.ace_flags_hex")
        access_mask = _dword(ace.access_mask_hex, name="ace.access_mask_hex")
        _validate_sid(ace.trustee_sid, name="ace.trustee_sid")
        ace_raw = _raw_hex(ace.raw_ace_hex, name="ace.raw_ace_hex")
        if len(ace_raw) < 8:
            _raise("ACE bytes are empty")
        if (
            ace_type != (0 if ace.ace_kind == "allow" else 1)
            or ace_raw[0] != ace_type
            or ace_raw[1] != ace_flags
            or int.from_bytes(ace_raw[2:4], "little") != len(ace_raw)
            or int.from_bytes(ace_raw[4:8], "little") != access_mask
            or ace_raw[8:].hex() != ace.trustee_sid.sid_bytes_hex
        ):
            _raise("ACE raw bytes do not reproduce typed fields")
    cursor = dacl_offset + 8
    for ace in value.ordered_dacl_aces:
        ace_bytes = bytes.fromhex(ace.raw_ace_hex)
        if raw[cursor : cursor + len(ace_bytes)] != ace_bytes:
            _raise("descriptor DACL bytes do not reproduce ordered ACEs")
        cursor += len(ace_bytes)
    if cursor != dacl_offset + acl_size:
        _raise("descriptor DACL length does not match ordered ACEs")


def _validate_token_facts(value: NativeTokenFacts, *, duplicate: bool) -> None:
    if type(value) is not NativeTokenFacts:
        _raise("token return has the wrong type")
    if not isinstance(value.token_type, str) or value.token_type not in {"primary", "impersonation"}:
        _raise("token type is outside the closed literal set")
    allowed_levels = {
        None,
        "anonymous",
        "identification",
        "security_impersonation",
        "security_delegation",
    }
    if value.impersonation_level is not None and not isinstance(value.impersonation_level, str):
        _raise("token impersonation level has the wrong type")
    if value.impersonation_level not in allowed_levels:
        _raise("token impersonation level is outside the closed literal set")
    if value.token_type == "primary" and value.impersonation_level is not None:
        _raise("primary token has impersonation level")
    if value.token_type == "impersonation" and value.impersonation_level is None:
        _raise("impersonation token omits its impersonation level")
    for name in ("token_id_luid_hex", "authentication_id_luid_hex", "modified_id_luid_hex"):
        candidate = getattr(value, name)
        if not isinstance(candidate, str) or _LUID_RE.fullmatch(candidate) is None:
            _raise(f"token {name} is malformed")
    _json_int(value.session_id, name="token.session_id")
    _validate_sid(value.user, name="token.user")
    _validate_sid(value.owner, name="token.owner")
    _validate_sid(value.primary_group, name="token.primary_group")
    _validate_sid(value.integrity_level_sid, name="token.integrity_level_sid")
    if not isinstance(value.elevation_type, str) or value.elevation_type not in {"default", "full", "limited"}:
        _raise("token elevation type is outside the closed literal set")
    for name in (
        "is_elevated",
        "has_restrictions",
        "is_app_container",
        "virtualization_allowed",
        "virtualization_enabled",
        "ui_access",
    ):
        if type(getattr(value, name)) is not bool:
            _raise(f"token {name} must be Boolean")
    if value.elevation_type == "full" and not value.is_elevated:
        _raise("full elevation is not elevated")
    if value.elevation_type == "limited" and value.is_elevated:
        _raise("limited elevation is elevated")
    for name in (
        "ordered_groups",
        "ordered_restricted_sids",
        "ordered_privileges",
        "ordered_capabilities",
    ):
        if type(getattr(value, name)) is not tuple:
            _raise(f"token {name} must be a tuple")
    for group in (
        value.ordered_groups
        + value.ordered_restricted_sids
        + value.ordered_capabilities
    ):
        if type(group) is not NativeGroup:
            _raise("token group has wrong type")
        _validate_sid(group.sid, name="token.group.sid")
        _dword(group.attributes_hex, name="token.group.attributes_hex")
    for privilege in value.ordered_privileges:
        if (
            type(privilege) is not NativePrivilege
            or not isinstance(privilege.luid_hex, str)
            or _LUID_RE.fullmatch(privilege.luid_hex) is None
        ):
            _raise("token privilege malformed")
        _nfc(privilege.name, name="token.privilege.name")
        _dword(privilege.attributes_hex, name="token.privilege.attributes_hex")
    if duplicate and value.token_type != "impersonation":
        _raise("AccessCheck duplicate is not an impersonation token")
    if duplicate and value.impersonation_level in {"anonymous", "identification"}:
        raise _PolicyRefusal("token_unsupported")
    if not duplicate and value.impersonation_level in {"anonymous", "identification"}:
        raise _PolicyRefusal("token_unsupported")
    if value.is_app_container or value.ordered_capabilities:
        raise _PolicyRefusal("token_unsupported")


def _token_projection(value: NativeTokenFacts) -> tuple[object, ...]:
    return (
        value.authentication_id_luid_hex,
        value.modified_id_luid_hex,
        value.session_id,
        value.user,
        value.owner,
        value.primary_group,
        value.ordered_groups,
        value.ordered_restricted_sids,
        value.ordered_privileges,
        value.integrity_level_sid,
        value.elevation_type,
        value.is_elevated,
        value.has_restrictions,
        value.is_app_container,
        value.ordered_capabilities,
        value.virtualization_allowed,
        value.virtualization_enabled,
        value.ui_access,
    )


def _validate_access_result(raw: object) -> NativeRawAccessCheckResult:
    if type(raw) is not NativeRawAccessCheckResult:
        raise _NativeReturnStructuralError("access_check return has the wrong type")
    if raw.api_call_status == "succeeded" and raw.access_status == "granted":
        if (
            raw.granted_access_mask_hex is None
            or type(raw.ordered_privileges_used) is not tuple
        ):
            raise _NativeReturnStructuralError("granted AccessCheck is incomplete")
        if raw.access_denial_error_hex is not None or raw.api_failure_error_hex is not None:
            raise _NativeReturnStructuralError("granted AccessCheck has an error")
        try:
            granted_mask = _dword(
                raw.granted_access_mask_hex,
                name="granted_access_mask_hex",
            )
            for privilege in raw.ordered_privileges_used:
                if (
                    type(privilege) is not NativePrivilege
                    or not isinstance(privilege.luid_hex, str)
                    or _LUID_RE.fullmatch(privilege.luid_hex) is None
                ):
                    _raise("AccessCheck privilege is malformed")
                _nfc(privilege.name, name="access_check.privilege.name")
                _dword(
                    privilege.attributes_hex,
                    name="access_check.privilege.attributes_hex",
                )
        except PermissionObservationContractError as exc:
            raise _NativeReturnStructuralError("granted AccessCheck fields are malformed") from exc
        if granted_mask & 0xF0000000:
            _raise("granted mask retains generic bits")
    elif raw.api_call_status == "succeeded" and raw.access_status == "denied":
        if (
            raw.granted_access_mask_hex != "0x0"
            or raw.ordered_privileges_used != ()
            or raw.access_denial_error_hex != "0x5"
            or raw.api_failure_error_hex is not None
        ):
            raise _NativeReturnStructuralError("denied AccessCheck equations are inconsistent")
    elif raw.api_call_status == "failed" and raw.access_status == "not_evaluated":
        if (
            raw.granted_access_mask_hex is not None
            or raw.ordered_privileges_used is not None
            or raw.access_denial_error_hex is not None
            or raw.api_failure_error_hex is None
        ):
            raise _NativeReturnStructuralError("failed AccessCheck equations are inconsistent")
        try:
            _dword(raw.api_failure_error_hex, name="api_failure_error_hex", nonzero=True)
        except PermissionObservationContractError as exc:
            raise _NativeReturnStructuralError("failed AccessCheck error is malformed") from exc
    else:
        raise _NativeReturnStructuralError("AccessCheck status combination is invalid")
    return raw


def _empty_state(ancestor_count: int) -> _CaptureState:
    return _CaptureState(
        ancestor_count=ancestor_count,
        ancestry_before=[],
        subject_before=None,
        streams=None,
        content=bytearray(),
        eof_observed=False,
        subject_security_before=None,
        ancestry_security_before=[],
        effective_before=None,
        access_token_before=None,
        access_checks=[],
        subject_security_after=None,
        ancestry_security_after=[],
        access_token_after=None,
        effective_after=None,
        subject_after=None,
        ancestry_after=[],
        opened=[],
        raw_prefixes=[],
        resources=[],
    )


def _scalar(value: object | None, operation: str) -> PartialScalarSlot:
    if value is None:
        return PartialScalarSlot("not_reached", None, None, ())
    return PartialScalarSlot("complete", value, operation, ())


def _array(values: list[object], expected: int, operation: str) -> PartialArraySlot:
    padded = tuple(values) + (None,) * (expected - len(values))
    if not values:
        return PartialArraySlot("not_reached", padded, None, (), expected)
    return PartialArraySlot(
        "complete" if len(values) == expected else "prefix",
        padded,
        operation,
        (),
        expected,
    )


def _events(
    state: _CaptureState,
    failures: tuple[ComponentFailureItem, ...],
) -> tuple[NativeResourceEvent, ...]:
    result: list[NativeResourceEvent] = []
    used: set[int] = set()
    for resource in state.resources:
        failure_index: int | None = None
        if resource.release_failure is not None:
            matches = [index for index, item in enumerate(failures) if item is resource.release_failure]
            if len(matches) != 1 or matches[0] in used:
                raise PermissionObservationProtocolError("release failure index is not one-to-one")
            failure_index = matches[0]
            used.add(failure_index)
        result.append(
            NativeResourceEvent(
                resource_kind=resource.kind,
                resource_role=resource.role,
                resource_ordinal=resource.ordinal,
                acquisition_sequence=resource.acquisition_sequence,
                acquired=True,
                release_attempted=resource.release_attempted,
                released=resource.released,
                release_failure_index=failure_index,
                object_identity=resource.object_identity,
            )
        )
    return tuple(result)


def _partial(
    state: _CaptureState,
    failures: tuple[ComponentFailureItem, ...],
) -> PartialCapturedPermissionObservation:
    digest = hashlib.sha256(bytes(state.content)).hexdigest()
    subject_security_prefixes = tuple(
        index
        for index, prefix in enumerate(state.raw_prefixes)
        if prefix.resource_role == "subject_security"
    )
    if state.subject_security_before is None and subject_security_prefixes:
        subject_security_before = PartialScalarSlot(
            "raw_only",
            None,
            state.raw_prefixes[subject_security_prefixes[-1]].source_operation,
            subject_security_prefixes,
        )
    else:
        subject_security_before = _scalar(
            state.subject_security_before, "parse_security_descriptor"
        )
    if (
        state.subject_security_before is not None
        and state.subject_security_after is None
        and subject_security_prefixes
    ):
        subject_security_after = PartialScalarSlot(
            "raw_only",
            None,
            state.raw_prefixes[subject_security_prefixes[-1]].source_operation,
            subject_security_prefixes,
        )
    else:
        subject_security_after = _scalar(
            state.subject_security_after, "parse_security_descriptor"
        )
    return PartialCapturedPermissionObservation(
        ancestry_before=_array(cast(list[object], state.ancestry_before), state.ancestor_count, "query_object"),
        subject_before=_scalar(state.subject_before, "query_object"),
        stream_inventory=_scalar(state.streams, "query_streams"),
        content_progress=ContentProgress("sha256", len(state.content), digest, state.eof_observed),
        subject_security_before=subject_security_before,
        ancestry_security_before=_array(cast(list[object], state.ancestry_security_before), state.ancestor_count, "parse_security_descriptor"),
        effective_context_before=_scalar(state.effective_before, "query_token"),
        access_check_token_before=_scalar(state.access_token_before, "query_token"),
        access_checks=_array(cast(list[object], state.access_checks), 2, "access_check"),
        subject_security_after=subject_security_after,
        ancestry_security_after=_array(cast(list[object], state.ancestry_security_after), state.ancestor_count, "parse_security_descriptor"),
        access_check_token_after=_scalar(state.access_token_after, "query_token"),
        effective_context_after=_scalar(state.effective_after, "query_token"),
        subject_after=_scalar(state.subject_after, "query_object"),
        ancestry_after=_array(cast(list[object], state.ancestry_after), state.ancestor_count, "query_object"),
        opened_resource_identities=tuple(state.opened),
        raw_prefixes=tuple(state.raw_prefixes),
        resource_events=_events(state, failures),
    )


def _accounting_without_finish(
    *,
    acquire_status: str,
    candidate: bool,
    acquired: bool,
    binding: str,
    reason: str,
) -> AuthorityAccounting:
    return AuthorityAccounting(
        acquire_status=cast(Literal["granted", "refused", "failed"], acquire_status),
        lease_candidate_returned=candidate,
        lease_acquired=acquired,
        lease_binding_status=cast(Literal["not_applicable", "valid", "invalid"], binding),
        preterminal_authority_reason_code=reason,
        finish_attempted=False,
        requested_outcome=None,
        finish_status="not_applicable",
        effect_certainty="not_applicable",
        recorded_outcome=None,
        finish_reason_code="none",
    )


def _accounting_with_finish(
    *,
    binding: str,
    reason: str,
    requested: Literal["succeeded", "refused", "failed"],
    finish: AuthorityFinishOutcome,
) -> AuthorityAccounting:
    return AuthorityAccounting(
        acquire_status="granted",
        lease_candidate_returned=True,
        lease_acquired=True,
        lease_binding_status=cast(Literal["valid", "invalid"], binding),
        preterminal_authority_reason_code=reason,
        finish_attempted=True,
        requested_outcome=requested,
        finish_status=finish.finish_status,
        effect_certainty=finish.effect_certainty,
        recorded_outcome=finish.recorded_outcome,
        finish_reason_code="none" if finish.finish_status == "succeeded" else "terminal_finish_failed",
    )


def _final_failed(
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    *,
    outcome: FailureOutcome,
    primary: ComponentFailureItem,
    follow: tuple[ComponentFailureItem, ...],
    state: _CaptureState,
    accounting: AuthorityAccounting,
) -> FailedPermissionObservation:
    failures = (primary,) + follow
    for item in failures:
        _validate_failure_item(item)
    _validate_accounting(
        accounting,
        primary=primary,
        outcome=outcome,
        failures=failures,
    )
    blank = FailedPermissionObservation(
        schema=FAILURE_SCHEMA,
        request=request,
        access_expectation=access,
        outcome=outcome,
        primary_failure=primary,
        partial_capture=_partial(state, failures),
        ordered_follow_on_failures=follow,
        authority_accounting=accounting,
        acceptance_emitted=False,
        record_self_sha256="0" * 64,
    )
    return replace(blank, record_self_sha256=_self_hash(blank, "record_self_sha256"))


def _validate_accounting(
    accounting: AuthorityAccounting,
    *,
    primary: ComponentFailureItem,
    outcome: FailureOutcome,
    failures: tuple[ComponentFailureItem, ...],
) -> None:
    if type(accounting) is not AuthorityAccounting:
        raise PermissionObservationProtocolError("authority accounting has the wrong type")
    if accounting.acquire_status not in {"granted", "refused", "failed"}:
        raise PermissionObservationProtocolError("authority acquire status is invalid")
    if type(accounting.lease_candidate_returned) is not bool or type(accounting.lease_acquired) is not bool:
        raise PermissionObservationProtocolError("authority lease accounting flags are not Boolean")
    if accounting.lease_binding_status not in {"not_applicable", "valid", "invalid"}:
        raise PermissionObservationProtocolError("authority binding status is invalid")
    preterminal_reasons = {
        "none",
        "authority_acquire_refused",
        "authority_acquire_failed",
        "lease_malformed",
        "lease_request_mismatch",
        "lease_operation_mismatch",
        "lease_access_expectation_mismatch",
        "lease_destination_mismatch",
        "lease_store_expectation_mismatch",
        "lease_scope_mismatch",
        "authority_inactive",
        "authority_check_failed",
    }
    if accounting.preterminal_authority_reason_code not in preterminal_reasons:
        raise PermissionObservationProtocolError("preterminal authority reason is invalid")
    if type(accounting.finish_attempted) is not bool:
        raise PermissionObservationProtocolError("finish-attempted flag is not Boolean")
    if accounting.requested_outcome not in {None, "succeeded", "refused", "failed"}:
        raise PermissionObservationProtocolError("requested authority outcome is invalid")
    if accounting.finish_status not in {"not_applicable", "succeeded", "failed"}:
        raise PermissionObservationProtocolError("finish status is invalid")
    if accounting.effect_certainty not in {"not_applicable", "confirmed", "not_applied", "unknown"}:
        raise PermissionObservationProtocolError("finish effect certainty is invalid")
    if accounting.recorded_outcome not in {None, "succeeded", "refused", "failed"}:
        raise PermissionObservationProtocolError("recorded authority outcome is invalid")
    if accounting.finish_reason_code not in {"none", "terminal_finish_failed"}:
        raise PermissionObservationProtocolError("finish reason is invalid")
    expected_outcome: FailureOutcome = (
        "refused"
        if primary.failure_kind in {"authority_refusal", "observation_refusal"}
        else "failed"
    )
    if outcome != expected_outcome:
        raise PermissionObservationProtocolError("record outcome does not follow its primary failure")
    terminal_indices = [
        index
        for index, failure in enumerate(failures)
        if failure.failure_kind == "terminal_accounting_failure"
    ]

    if accounting.acquire_status in {"refused", "failed"}:
        reason = (
            "authority_acquire_refused"
            if accounting.acquire_status == "refused"
            else "authority_acquire_failed"
        )
        kind = "authority_refusal" if accounting.acquire_status == "refused" else "authority_failure"
        if (
            accounting.lease_candidate_returned
            or accounting.lease_acquired
            or accounting.lease_binding_status != "not_applicable"
            or accounting.preterminal_authority_reason_code != reason
            or accounting.finish_attempted
            or accounting.requested_outcome is not None
            or accounting.finish_status != "not_applicable"
            or accounting.effect_certainty != "not_applicable"
            or accounting.recorded_outcome is not None
            or accounting.finish_reason_code != "none"
            or primary.failure_kind != kind
            or primary.reason_code != reason
            or terminal_indices
        ):
            raise PermissionObservationProtocolError("no-lease accounting is inconsistent")
        return

    if not accounting.lease_candidate_returned:
        raise PermissionObservationProtocolError("granted acquisition omits its lease candidate")
    if not accounting.lease_acquired:
        if (
            accounting.lease_binding_status != "invalid"
            or accounting.preterminal_authority_reason_code != "lease_malformed"
            or primary.failure_kind != "lease_binding_failure"
            or primary.reason_code != "lease_malformed"
            or accounting.finish_attempted
            or accounting.requested_outcome is not None
            or accounting.finish_status != "not_applicable"
            or accounting.effect_certainty != "not_applicable"
            or accounting.recorded_outcome is not None
            or accounting.finish_reason_code != "none"
            or terminal_indices
        ):
            raise PermissionObservationProtocolError("malformed-lease accounting is inconsistent")
        return

    if accounting.lease_binding_status == "not_applicable":
        raise PermissionObservationProtocolError("acquired lease has no binding result")
    if accounting.lease_binding_status == "invalid":
        binding_reasons = {
            "lease_request_mismatch",
            "lease_operation_mismatch",
            "lease_access_expectation_mismatch",
            "lease_destination_mismatch",
            "lease_store_expectation_mismatch",
            "lease_scope_mismatch",
        }
        if (
            accounting.preterminal_authority_reason_code not in binding_reasons
            or primary.failure_kind != "lease_binding_failure"
            or primary.reason_code != accounting.preterminal_authority_reason_code
        ):
            raise PermissionObservationProtocolError("invalid-binding accounting is inconsistent")
    else:
        if primary.failure_kind == "authority_refusal":
            if (
                primary.component_operation != "require_active"
                or primary.reason_code != "authority_inactive"
            ):
                raise PermissionObservationProtocolError(
                    "granted lease carries an acquisition refusal"
                )
            expected_reason = "authority_inactive"
        elif primary.failure_kind == "authority_failure":
            if (
                primary.component_operation != "require_active"
                or primary.reason_code != "authority_check_failed"
            ):
                raise PermissionObservationProtocolError(
                    "granted lease carries an acquisition failure"
                )
            expected_reason = "authority_check_failed"
        elif primary.failure_kind == "lease_binding_failure":
            raise PermissionObservationProtocolError(
                "valid lease carries a lease-binding primary failure"
            )
        else:
            expected_reason = "none"
        if accounting.preterminal_authority_reason_code != expected_reason:
            raise PermissionObservationProtocolError("valid-binding authority reason is inconsistent")

    if not accounting.finish_attempted or accounting.requested_outcome is None:
        raise PermissionObservationProtocolError("acquired lease lacks its sole finish accounting")
    if primary.failure_kind == "terminal_accounting_failure":
        if accounting.requested_outcome != "succeeded":
            raise PermissionObservationProtocolError("post-success finish failure has wrong requested outcome")
    elif accounting.requested_outcome != outcome:
        raise PermissionObservationProtocolError("preterminal finish request does not match record outcome")
    if accounting.finish_status == "succeeded":
        if (
            accounting.effect_certainty != "confirmed"
            or accounting.recorded_outcome != accounting.requested_outcome
            or accounting.finish_reason_code != "none"
            or terminal_indices
        ):
            raise PermissionObservationProtocolError("successful finish accounting is inconsistent")
    elif accounting.finish_status == "failed":
        if (
            accounting.finish_reason_code != "terminal_finish_failed"
            or accounting.effect_certainty not in {"confirmed", "not_applied", "unknown"}
            or accounting.effect_certainty == "confirmed"
            and accounting.recorded_outcome != accounting.requested_outcome
            or accounting.effect_certainty in {"not_applied", "unknown"}
            and accounting.recorded_outcome is not None
            or terminal_indices != [len(failures) - 1]
        ):
            raise PermissionObservationProtocolError("failed finish accounting is inconsistent")
    else:
        raise PermissionObservationProtocolError("attempted finish is not applicable")


def _validate_prefinish_failure(
    state: _CaptureState,
    primary: ComponentFailureItem,
    follow: tuple[ComponentFailureItem, ...],
) -> None:
    failures = (primary,) + follow
    for item in failures:
        _validate_failure_item(item)
    partial = _partial(state, failures)
    for event in partial.resource_events:
        if not event.acquired or event.acquisition_sequence < 0:
            raise PermissionObservationProtocolError("resource ledger has an invalid acquisition")
        if event.release_attempted:
            if event.released == (event.release_failure_index is not None):
                raise PermissionObservationProtocolError("resource release result is inconsistent")
        elif event.released or event.release_failure_index is not None:
            raise PermissionObservationProtocolError("unattempted resource has a release result")
        if event.release_failure_index is not None:
            target = failures[event.release_failure_index]
            if (
                target.failure_kind != "cleanup_failure"
                or target.component_operation != "release_native_resource"
                or target.resource_kind != event.resource_kind
                or target.resource_role != event.resource_role
                or target.resource_ordinal != event.resource_ordinal
            ):
                raise PermissionObservationProtocolError("release failure target is not provenance-bound")


def _active(
    lease: PermissionAuthorityLease,
    scope: PermissionOperationScope,
    *,
    step: str,
    ordinal: int | None,
    stage: str,
) -> None:
    checkpoint = PermissionAuthorityCheckpoint(scope.operation, step, ordinal)
    try:
        result = lease.require_active(checkpoint)
    except Exception as exc:
        raise PermissionObservationProtocolError("authority require_active raised") from exc
    if type(result) is not AuthorityCheckOutcome or result.checkpoint != checkpoint:
        raise PermissionObservationProtocolError("authority returned a mismatched checkpoint")
    if not isinstance(result.status, str):
        raise PermissionObservationProtocolError("authority check status has the wrong type")
    if result.status == "active" and result.failure is None:
        return
    if result.status in {"refused", "failed"} and type(result.failure) is ComponentFailureItem:
        failure = _validate_failure_item(result.failure)
        expected_kind = "authority_refusal" if result.status == "refused" else "authority_failure"
        expected_reason = "authority_inactive" if result.status == "refused" else "authority_check_failed"
        if (
            failure.failure_kind != expected_kind
            or failure.reason_code != expected_reason
            or failure.component_operation != "require_active"
            or failure.stage != stage
            or failure.operation_ordinal != ordinal
        ):
            raise PermissionObservationProtocolError("authority check failure does not bind its checkpoint")
        raise _StopObservation(failure, authority_reason=expected_reason)
    raise PermissionObservationProtocolError("authority check outcome is malformed")


def _resource_binding(resource: _RegisteredResource) -> tuple[str, str, int | None]:
    return resource.kind, resource.role, resource.ordinal


def _validate_native_resource_binding(
    value: object,
    *,
    name: str,
) -> NativeResourceBinding:
    if type(value) is not tuple or len(value) != 3:
        raise PermissionObservationContractError(f"{name} is not an exact resource-binding tuple")
    kind, role, ordinal = value
    if (
        not isinstance(kind, str)
        or not isinstance(role, str)
        or kind not in _NATIVE_RESOURCE_KINDS
        or role not in _NATIVE_RESOURCE_ROLES
    ):
        raise PermissionObservationContractError(f"{name} is outside the native resource vocabulary")
    if (role in {"ancestor", "ancestor_security"}) != (ordinal is not None):
        raise PermissionObservationContractError(f"{name} has inconsistent ordinal presence")
    if ordinal is not None and (
        isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0
    ):
        raise PermissionObservationContractError(f"{name} has an invalid ordinal")
    return cast(NativeResourceBinding, value)


def _expected_transferred_binding_from_binding(
    operation: str,
    stage: str,
    operation_ordinal: int | None,
    resource_binding: NativeResourceBinding | None,
) -> NativeResourceBinding | None:
    if operation == "open_existing_directory":
        return "directory_handle", "ancestor", operation_ordinal
    if operation == "open_existing_subject":
        return "file_handle", "subject", None
    if operation == "acquire_security_descriptor" and resource_binding is not None:
        _, role, ordinal = resource_binding
        if role == "subject":
            return "security_descriptor_buffer", "subject_security", None
        if role == "ancestor":
            return "security_descriptor_buffer", "ancestor_security", ordinal
    if operation == "open_current_effective_token":
        role = (
            "effective_context_before"
            if stage == "effective_context_before"
            else "effective_context_after"
        )
        return "effective_token_handle", role, None
    if operation == "duplicate_for_access_check":
        return "access_check_token_handle", "access_check_duplicate", None
    return None


def validate_native_call_context(context: NativeCallContext) -> None:
    """Validate the non-serialized binding for one native boundary call."""

    if type(context) is not NativeCallContext:
        raise PermissionObservationContractError("native call context has the wrong type")
    _sha(context.request_self_sha256, name="native_call_context.request_self_sha256")
    operation = context.component_operation
    if not isinstance(operation, str) or operation not in _NATIVE_OPERATIONS:
        raise PermissionObservationContractError("native call context operation is outside the closed vocabulary")
    if (
        not isinstance(context.stage, str)
        or context.stage not in _NATIVE_OPERATION_STAGES[operation]
    ):
        raise PermissionObservationContractError("native call context stage does not bind its operation")
    if context.operation_ordinal is not None and (
        isinstance(context.operation_ordinal, bool)
        or not isinstance(context.operation_ordinal, int)
        or context.operation_ordinal < 0
    ):
        raise PermissionObservationContractError("native call context operation ordinal is invalid")
    if (
        isinstance(context.raw_prefix_start_sequence, bool)
        or not isinstance(context.raw_prefix_start_sequence, int)
        or context.raw_prefix_start_sequence < 0
    ):
        raise PermissionObservationContractError("native call context raw-prefix sequence is invalid")

    binding = (
        None
        if context.resource_binding is None
        else _validate_native_resource_binding(
            context.resource_binding,
            name="native_call_context.resource_binding",
        )
    )
    allowed = (
        None
        if context.allowed_returned_resource_binding is None
        else _validate_native_resource_binding(
            context.allowed_returned_resource_binding,
            name="native_call_context.allowed_returned_resource_binding",
        )
    )
    ordinal = context.operation_ordinal
    stage = context.stage
    if operation in {"open_existing_directory", "open_existing_subject", "open_current_effective_token"}:
        if binding is not None:
            raise PermissionObservationContractError("native opening context invents an existing resource")
    elif binding is None:
        raise PermissionObservationContractError("native call context omits its existing resource")

    if operation == "open_existing_directory":
        valid = ordinal is not None
    elif operation in {
        "open_existing_subject",
        "query_streams",
        "open_current_effective_token",
        "duplicate_for_access_check",
        "query_token",
    }:
        valid = ordinal is None
    elif operation == "query_object":
        valid = bool(
            binding is not None
            and (
                stage == "ancestor_before"
                and binding[1] == "ancestor"
                and ordinal == binding[2]
                or stage == "subject_before"
                and binding[1] == "subject"
                and ordinal is None
                or stage == "object_after"
                and (
                    binding[1] == "subject"
                    and ordinal == 0
                    or binding[1] == "ancestor"
                    and ordinal == 1 + cast(int, binding[2])
                )
            )
        )
    elif operation == "read_unnamed_at":
        valid = binding is not None and binding[:2] == ("file_handle", "subject") and ordinal is not None
    elif operation in {"acquire_security_descriptor", "parse_security_descriptor"}:
        valid_roles = (
            {"subject", "ancestor"}
            if operation == "acquire_security_descriptor"
            else {"subject_security", "ancestor_security"}
        )
        valid = bool(
            binding is not None
            and binding[1] in valid_roles
            and ordinal
            == (0 if binding[1] in {"subject", "subject_security"} else 1 + cast(int, binding[2]))
        )
    elif operation == "access_check":
        valid = (
            binding is not None
            and binding[:2] == ("access_check_token_handle", "access_check_duplicate")
            and ordinal in {0, 1}
        )
    else:
        valid = True
    if not valid:
        raise PermissionObservationContractError("native call context operation ordinal/resource binding is inconsistent")

    allowed_bindings: dict[str, frozenset[tuple[str, str]]] = {
        "query_object": frozenset({("directory_handle", "ancestor"), ("file_handle", "subject")}),
        "query_streams": frozenset({("file_handle", "subject")}),
        "read_unnamed_at": frozenset({("file_handle", "subject")}),
        "acquire_security_descriptor": frozenset(
            {("directory_handle", "ancestor"), ("file_handle", "subject")}
        ),
        "parse_security_descriptor": frozenset(
            {
                ("security_descriptor_buffer", "ancestor_security"),
                ("security_descriptor_buffer", "subject_security"),
            }
        ),
        "duplicate_for_access_check": frozenset(
            {("effective_token_handle", "effective_context_before")}
        ),
        "query_token": frozenset(
            {
                ("effective_token_handle", "effective_context_before"),
                ("effective_token_handle", "effective_context_after"),
                ("access_check_token_handle", "access_check_duplicate"),
            }
        ),
        "access_check": frozenset(
            {("access_check_token_handle", "access_check_duplicate")}
        ),
    }
    if binding is not None and binding[:2] not in allowed_bindings.get(operation, frozenset()):
        raise PermissionObservationContractError("native call context resource is incompatible with its operation")
    if operation == "query_token" and binding is not None:
        expected_role = {
            "effective_context_before": "effective_context_before",
            "access_token": "access_check_duplicate",
            "access_token_after": "access_check_duplicate",
            "effective_context_after": "effective_context_after",
        }[stage]
        if binding[1] != expected_role:
            raise PermissionObservationContractError("native token context does not bind its phase")

    expected_allowed = _expected_transferred_binding_from_binding(
        operation,
        stage,
        ordinal,
        binding,
    )
    if allowed != expected_allowed:
        raise PermissionObservationContractError("native call context transferred-resource binding is inconsistent")


def _native_call_context(
    state: _CaptureState,
    *,
    request_self_sha256: str,
    stage: str,
    operation: str,
    resource: _RegisteredResource | None = None,
    operation_ordinal: int | None = None,
    raw_prefix_sequence_advance: int = 0,
) -> NativeCallContext:
    if (
        isinstance(raw_prefix_sequence_advance, bool)
        or not isinstance(raw_prefix_sequence_advance, int)
        or raw_prefix_sequence_advance < 0
    ):
        raise PermissionObservationProtocolError("native prefix-sequence advance is invalid")
    binding = None if resource is None else _resource_binding(resource)
    context = NativeCallContext(
        request_self_sha256=request_self_sha256,
        stage=stage,
        component_operation=operation,
        resource_binding=binding,
        allowed_returned_resource_binding=_expected_transferred_binding_from_binding(
            operation,
            stage,
            operation_ordinal,
            binding,
        ),
        operation_ordinal=operation_ordinal,
        raw_prefix_start_sequence=len(state.raw_prefixes) + raw_prefix_sequence_advance,
    )
    validate_native_call_context(context)
    return context


def _expected_transferred_binding(
    operation: str,
    stage: str,
    operation_ordinal: int | None,
    resource: _RegisteredResource | None,
) -> tuple[str, str, int | None] | None:
    return _expected_transferred_binding_from_binding(
        operation,
        stage,
        operation_ordinal,
        None if resource is None else _resource_binding(resource),
    )


def _native_call(
    state: _CaptureState,
    callable_: object,
    *,
    request_self_sha256: str,
    stage: str,
    operation: str,
    resource: _RegisteredResource | None = None,
    operation_ordinal: int | None = None,
) -> object:
    context = _native_call_context(
        state,
        request_self_sha256=request_self_sha256,
        stage=stage,
        operation=operation,
        resource=resource,
        operation_ordinal=operation_ordinal,
    )
    try:
        return cast(object, callable_)(context=context)  # type: ignore[operator]
    except PermissionNativeCallError as exc:
        try:
            if type(exc.returned_resources) is not tuple or len(exc.returned_resources) > 1:
                raise PermissionObservationProtocolError("typed native error resources are not an exact tuple")
            expected_transfer = _expected_transferred_binding(
                operation,
                stage,
                operation_ordinal,
                resource,
            )
            seen = {id(registered.value) for registered in state.resources}
            transferred: list[_RegisteredResource] = []
            for returned_row in exc.returned_resources:
                if type(returned_row) is not tuple or len(returned_row) != 4:
                    raise PermissionObservationProtocolError("typed native error resource row is malformed")
                returned, kind, role, ordinal = returned_row
                if (
                    expected_transfer is None
                    or (kind, role, ordinal) != expected_transfer
                    or not _has_release(returned)
                    or id(returned) in seen
                ):
                    raise PermissionObservationProtocolError("typed native error returned resource is invalid")
                transferred.append(_register(state, returned, kind, role, ordinal))
                seen.add(id(returned))
            failure = _validate_failure_item(exc.failure)
            if (
                failure.stage != stage
                or failure.component_operation != operation
                or failure.operation_ordinal != operation_ordinal
            ):
                raise PermissionObservationProtocolError("typed native failure does not bind the invoked call")
            valid_failure_bindings = (
                ([] if resource is None else [_resource_binding(resource)])
                + [_resource_binding(registered) for registered in transferred]
            )
            failure_binding = (
                failure.resource_kind,
                failure.resource_role,
                failure.resource_ordinal,
            )
            if valid_failure_bindings:
                if failure_binding not in valid_failure_bindings:
                    raise PermissionObservationProtocolError("typed native failure has wrong resource provenance")
            elif failure_binding != (None, None, None):
                raise PermissionObservationProtocolError("typed native failure invents resource provenance")
            if type(exc.raw_prefixes) is not tuple:
                raise PermissionObservationProtocolError("typed native error prefixes are not a tuple")
            valid_prefix_bindings = set(valid_failure_bindings)
            for prefix_value in exc.raw_prefixes:
                prefix = _validate_prefix(
                    prefix_value,
                    expected_sequence=len(state.raw_prefixes),
                )
                if (
                    prefix.source_operation != operation
                    or (
                        prefix.resource_kind,
                        prefix.resource_role,
                        prefix.resource_ordinal,
                    )
                    not in valid_prefix_bindings
                ):
                    raise PermissionObservationProtocolError("typed native raw prefix has wrong provenance")
                state.raw_prefixes.append(prefix)
        except (PermissionObservationProtocolError, PermissionObservationContractError) as malformed:
            raise _resource_failure(
                "native_protocol_failure",
                stage=stage,
                operation=operation,
                resource=resource,
                failure_kind="native_protocol_failure",
                operation_ordinal=operation_ordinal,
            ) from malformed
        raise _StopObservation(failure) from exc
    except _StopObservation:
        raise
    except Exception as exc:
        raise _resource_failure(
            "native_protocol_failure",
            stage=stage,
            operation=operation,
            resource=resource,
            failure_kind="native_protocol_failure",
            operation_ordinal=operation_ordinal,
        ) from exc


def _require_returned_handle(
    value: object,
    *,
    stage: str,
    operation: str,
    operation_ordinal: int | None = None,
    require_copy: bool = False,
    resource: _RegisteredResource | None = None,
) -> None:
    valid = _has_release(value)
    if require_copy:
        try:
            valid = valid and callable(getattr(value, "copy_bytes"))
        except Exception:
            valid = False
    if not valid:
        raise _StopObservation(
            _failure(
                "native_protocol_failure",
                stage,
                operation,
                "native_protocol_failure",
                kind=None if resource is None else resource.kind,
                role=None if resource is None else resource.role,
                resource_ordinal=None if resource is None else resource.ordinal,
                operation_ordinal=operation_ordinal,
            )
        )


def _cleanup_failure_for(
    resource: _RegisteredResource,
    returned: ComponentFailureItem | None,
) -> ComponentFailureItem:
    backend = None
    if type(returned) is ComponentFailureItem:
        backend = returned.backend_error_hex
        if backend is not None:
            try:
                _dword(backend, name="release backend error", nonzero=True)
            except PermissionObservationContractError:
                backend = None
    return _failure(
        "cleanup_failure",
        "resource_cleanup",
        "release_native_resource",
        "resource_release_failed",
        kind=resource.kind,
        role=resource.role,
        resource_ordinal=resource.ordinal,
        backend_error_hex=backend,
    )


def _release_one(resource: _RegisteredResource) -> ComponentFailureItem | None:
    if resource.release_attempted:
        raise PermissionObservationProtocolError("resource release attempted twice")
    resource.release_attempted = True
    try:
        result = cast(object, resource.value).release()  # type: ignore[attr-defined]
    except Exception:
        result = None
        failure = _cleanup_failure_for(resource, None)
        resource.release_failure = failure
        return failure
    if result is None:
        resource.released = True
        return None
    failure = _cleanup_failure_for(
        resource,
        result if type(result) is ComponentFailureItem else None,
    )
    resource.release_failure = failure
    return failure


def _cleanup(state: _CaptureState) -> tuple[ComponentFailureItem, ...]:
    failures: list[ComponentFailureItem] = []
    for resource in reversed(state.resources):
        if resource.release_attempted:
            if resource.release_failure is not None:
                failures.append(resource.release_failure)
            continue
        failure = _release_one(resource)
        if failure is not None:
            failures.append(failure)
    failures.sort(
        key=lambda failure: next(
            resource.acquisition_sequence
            for resource in reversed(state.resources)
            if resource.release_failure is failure
        ),
        reverse=True,
    )
    return tuple(failures)


def _raw_prefix(
    state: _CaptureState,
    data: bytes,
    *,
    source_operation: str,
    resource: _RegisteredResource,
    encoding: str,
    complete: bool = True,
    declared_size: int | None = None,
) -> NativeRawPrefix:
    prefix = NativeRawPrefix(
        capture_sequence=len(state.raw_prefixes),
        source_operation=source_operation,
        resource_kind=resource.kind,
        resource_role=resource.role,
        resource_ordinal=resource.ordinal,
        encoding=encoding,
        declared_size_bytes=len(data) if declared_size is None and complete else declared_size,
        captured_size_bytes=len(data),
        complete=complete,
        data_hex=data.hex(),
        captured_bytes_sha256=sha256_bytes(data),
    )
    state.raw_prefixes.append(prefix)
    return prefix


def _security(
    state: _CaptureState,
    lease: PermissionAuthorityLease,
    scope: PermissionOperationScope,
    native: PermissionNativeApi,
    *,
    request_self_sha256: str,
    handle_resource: _RegisteredResource,
    snapshot: NativeObjectSnapshot,
    after: bool,
    operation_ordinal: int,
) -> BoundSecurityDescriptor:
    stage = "security_after" if after else "security_before"
    step = "security_after_query" if after else "security_before_query"
    _active(lease, scope, step=step, ordinal=operation_ordinal, stage=stage)
    buffer_value = _native_call(
        state,
        lambda *, context: native.acquire_security_descriptor(
            cast(NativeObjectHandle, handle_resource.value),
            context=context,
        ),
        request_self_sha256=request_self_sha256,
        stage=stage,
        operation="acquire_security_descriptor",
        resource=handle_resource,
        operation_ordinal=operation_ordinal,
    )
    _require_returned_handle(
        buffer_value,
        stage=stage,
        operation="acquire_security_descriptor",
        operation_ordinal=operation_ordinal,
        require_copy=True,
        resource=handle_resource,
    )
    role = "subject_security" if snapshot.object_role == "subject" else "ancestor_security"
    buffer_resource = _register(
        state,
        buffer_value,
        "security_descriptor_buffer",
        role,
        snapshot.ancestor_index,
    )
    try:
        copied = cast(NativeBuffer, buffer_value).copy_bytes()
    except Exception as exc:
        raise _resource_failure(
            "native_protocol_failure",
            stage=stage,
            operation="acquire_security_descriptor",
            resource=buffer_resource,
            failure_kind="native_protocol_failure",
            operation_ordinal=operation_ordinal,
        ) from exc
    if type(copied) is not bytes:
        raise _resource_failure(
            "native_protocol_failure",
            stage=stage,
            operation="acquire_security_descriptor",
            resource=buffer_resource,
            failure_kind="native_protocol_failure",
            operation_ordinal=operation_ordinal,
        )
    parser_context = _native_call_context(
        state,
        request_self_sha256=request_self_sha256,
        stage=stage,
        operation="parse_security_descriptor",
        resource=buffer_resource,
        operation_ordinal=operation_ordinal,
        raw_prefix_sequence_advance=1,
    )
    try:
        parsed = native.parse_security_descriptor(copied, context=parser_context)
    except PermissionNativeCallError as exc:
        _raw_prefix(
            state,
            copied,
            source_operation="parse_security_descriptor",
            resource=buffer_resource,
            encoding="security_descriptor",
        )
        try:
            failure = _validate_failure_item(exc.failure)
            if (
                failure.stage != stage
                or failure.component_operation != "parse_security_descriptor"
                or failure.operation_ordinal != operation_ordinal
                or (
                    failure.resource_kind,
                    failure.resource_role,
                    failure.resource_ordinal,
                )
                != _resource_binding(buffer_resource)
            ):
                raise PermissionObservationProtocolError(
                    "typed parser failure does not bind the invoked call"
                )
            if type(exc.raw_prefixes) is not tuple:
                raise PermissionObservationProtocolError("typed parser prefixes are not a tuple")
            for prefix_value in exc.raw_prefixes:
                prefix = _validate_prefix(
                    prefix_value,
                    expected_sequence=len(state.raw_prefixes),
                )
                if (
                    prefix.source_operation != "parse_security_descriptor"
                    or (
                        prefix.resource_kind,
                        prefix.resource_role,
                        prefix.resource_ordinal,
                    )
                    != _resource_binding(buffer_resource)
                ):
                    raise PermissionObservationProtocolError(
                        "typed parser raw prefix has wrong provenance"
                    )
                state.raw_prefixes.append(prefix)
            if type(exc.returned_resources) is not tuple or exc.returned_resources:
                raise PermissionObservationProtocolError(
                    "descriptor parser cannot transfer a caller-owned native resource"
                )
        except (PermissionObservationProtocolError, PermissionObservationContractError) as malformed:
            raise _resource_failure(
                "native_protocol_failure",
                stage=stage,
                operation="parse_security_descriptor",
                resource=buffer_resource,
                failure_kind="native_protocol_failure",
                operation_ordinal=operation_ordinal,
            ) from malformed
        raise _StopObservation(failure) from exc
    except Exception as exc:
        _raw_prefix(
            state,
            copied,
            source_operation="parse_security_descriptor",
            resource=buffer_resource,
            encoding="security_descriptor",
        )
        raise _resource_failure(
            "native_protocol_failure",
            stage=stage,
            operation="parse_security_descriptor",
            resource=buffer_resource,
            failure_kind="native_protocol_failure",
            operation_ordinal=operation_ordinal,
        ) from exc
    if type(parsed) is not NativeSecurityDescriptor:
        _raw_prefix(
            state,
            copied,
            source_operation="parse_security_descriptor",
            resource=buffer_resource,
            encoding="security_descriptor",
        )
        raise _resource_failure(
            "native_protocol_failure",
            stage=stage,
            operation="parse_security_descriptor",
            resource=buffer_resource,
            failure_kind="native_protocol_failure",
            operation_ordinal=operation_ordinal,
        )
    try:
        _validate_descriptor(parsed)
    except _PolicyRefusal as exc:
        _raw_prefix(
            state,
            copied,
            source_operation="parse_security_descriptor",
            resource=buffer_resource,
            encoding="security_descriptor",
        )
        raise _resource_failure(
            exc.reason_code,
            stage=stage,
            operation="parse_security_descriptor",
            resource=buffer_resource,
            failure_kind="observation_refusal",
            operation_ordinal=operation_ordinal,
        ) from exc
    except PermissionObservationContractError as exc:
        _raw_prefix(
            state,
            copied,
            source_operation="parse_security_descriptor",
            resource=buffer_resource,
            encoding="security_descriptor",
        )
        raise _resource_failure(
            "native_protocol_failure",
            stage=stage,
            operation="parse_security_descriptor",
            resource=buffer_resource,
            failure_kind="native_protocol_failure",
            operation_ordinal=operation_ordinal,
        ) from exc
    failure = _release_one(buffer_resource)
    if failure is not None:
        raise _StopObservation(failure)
    return BoundSecurityDescriptor(
        object_role=snapshot.object_role,
        ancestor_index=snapshot.ancestor_index,
        object_identity=snapshot.facts.identity,
        canonical_volume_guid_path=snapshot.facts.canonical_volume_guid_path,
        descriptor=parsed,
    )


def _query_token_snapshot(
    state: _CaptureState,
    lease: PermissionAuthorityLease,
    scope: PermissionOperationScope,
    native: PermissionNativeApi,
    *,
    request_self_sha256: str,
    resource: _RegisteredResource,
    step: str,
    stage: str,
    role: Literal["effective_context", "access_check_duplicate"],
    source: Literal["thread_impersonation", "process_primary", "duplicated_effective_context"],
    duplicated_from: str | None,
) -> NativeTokenSnapshot:
    _active(lease, scope, step=step, ordinal=None, stage=stage)
    facts = _native_call(
        state,
        lambda *, context: native.query_token(
            cast(NativeTokenHandle, resource.value),
            context=context,
        ),
        request_self_sha256=request_self_sha256,
        stage=stage,
        operation="query_token",
        resource=resource,
    )
    if type(facts) is not NativeTokenFacts:
        raise _resource_failure(
            "native_protocol_failure",
            stage=stage,
            operation="query_token",
            resource=resource,
            failure_kind="native_protocol_failure",
        )
    snapshot = NativeTokenSnapshot(role, source, duplicated_from, cast(NativeTokenFacts, facts))
    try:
        _validate_token_facts(cast(NativeTokenFacts, facts), duplicate=role == "access_check_duplicate")
    except _PolicyRefusal as exc:
        if stage == "effective_context_before":
            state.effective_before = snapshot
        elif stage == "access_token":
            state.access_token_before = snapshot
        elif stage == "access_token_after":
            state.access_token_after = snapshot
        elif stage == "effective_context_after":
            state.effective_after = snapshot
        else:
            raise PermissionObservationProtocolError("unsupported token has an unknown capture slot") from exc
        raise _resource_failure(
            exc.reason_code,
            stage=stage,
            operation="query_token",
            resource=resource,
            failure_kind="observation_refusal",
        ) from exc
    except PermissionObservationContractError as exc:
        raise _resource_failure(
            "native_protocol_failure",
            stage=stage,
            operation="query_token",
            resource=resource,
            failure_kind="native_protocol_failure",
        ) from exc
    return snapshot


def _lease_binding_reason(
    lease: object,
    scope: PermissionOperationScope,
) -> tuple[bool, str]:
    required = (
        "request_self_sha256",
        "operation",
        "access_expectation_self_sha256",
        "destination_self_sha256",
        "store_expectation_self_sha256",
        "scope_self_sha256",
        "require_active",
        "finish",
    )
    try:
        if any(not hasattr(lease, name) for name in required):
            return False, "lease_malformed"
        if not callable(getattr(lease, "require_active")) or not callable(getattr(lease, "finish")):
            return False, "lease_malformed"
        for name in ("request_self_sha256", "access_expectation_self_sha256", "scope_self_sha256"):
            _sha(getattr(lease, name), name=f"lease.{name}")
        for name in ("destination_self_sha256", "store_expectation_self_sha256"):
            value = getattr(lease, name)
            if value is not None:
                _sha(value, name=f"lease.{name}")
    except (PermissionObservationContractError, Exception):
        return False, "lease_malformed"
    comparisons = (
        ("request_self_sha256", "lease_request_mismatch"),
        ("operation", "lease_operation_mismatch"),
        ("access_expectation_self_sha256", "lease_access_expectation_mismatch"),
        ("destination_self_sha256", "lease_destination_mismatch"),
        ("store_expectation_self_sha256", "lease_store_expectation_mismatch"),
        ("scope_self_sha256", "lease_scope_mismatch"),
    )
    for name, reason in comparisons:
        if getattr(lease, name) != getattr(scope, name):
            return True, reason
    return True, "none"


def _finish(
    lease: PermissionAuthorityLease,
    requested: Literal["succeeded", "refused", "failed"],
) -> AuthorityFinishOutcome:
    try:
        result = lease.finish(requested)
    except Exception:
        return AuthorityFinishOutcome("failed", "unknown", None, "terminal_finish_failed")
    if type(result) is not AuthorityFinishOutcome:
        return AuthorityFinishOutcome("failed", "unknown", None, "terminal_finish_failed")
    if result.finish_status == "succeeded":
        if (
            result.effect_certainty != "confirmed"
            or result.recorded_outcome != requested
            or result.reason_code != "none"
        ):
            return AuthorityFinishOutcome("failed", "unknown", None, "terminal_finish_failed")
    elif result.finish_status == "failed":
        if (
            result.reason_code != "terminal_finish_failed"
            or not isinstance(result.effect_certainty, str)
            or result.effect_certainty not in {"confirmed", "not_applied", "unknown"}
            or result.effect_certainty == "confirmed"
            and result.recorded_outcome != requested
            or result.effect_certainty in {"not_applied", "unknown"}
            and result.recorded_outcome is not None
        ):
            return AuthorityFinishOutcome("failed", "unknown", None, "terminal_finish_failed")
    else:
        return AuthorityFinishOutcome("failed", "unknown", None, "terminal_finish_failed")
    return result


def _finish_failure_item() -> ComponentFailureItem:
    return _failure(
        "terminal_accounting_failure",
        "terminal_accounting",
        "finish_authority_lease",
        "terminal_finish_failed",
    )


def _return_failure(
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    state: _CaptureState,
    *,
    lease: PermissionAuthorityLease | None,
    primary: ComponentFailureItem,
    authority_reason: str,
    binding: Literal["valid", "invalid"],
    finish_allowed: bool,
    cleanup: tuple[ComponentFailureItem, ...] = (),
) -> FailedPermissionObservation:
    outcome: FailureOutcome = (
        "refused"
        if primary.failure_kind in {"authority_refusal", "observation_refusal"}
        else "failed"
    )
    requested: Literal["refused", "failed"] = outcome
    failures = [primary, *cleanup]
    _validate_prefinish_failure(state, primary, cleanup)
    if not finish_allowed or lease is None:
        accounting = _accounting_without_finish(
            acquire_status="granted",
            candidate=True,
            acquired=lease is not None,
            binding=binding,
            reason=authority_reason,
        )
    else:
        result = _finish(lease, requested)
        accounting = _accounting_with_finish(
            binding=binding,
            reason=authority_reason,
            requested=requested,
            finish=result,
        )
        if result.finish_status != "succeeded":
            failures.append(_finish_failure_item())
    return _final_failed(
        request,
        access,
        outcome=outcome,
        primary=failures[0],
        follow=tuple(failures[1:]),
        state=state,
        accounting=accounting,
    )


def _capture_failure(reason: str) -> _StopObservation:
    return _StopObservation(
        _failure(
            "validation_failure",
            "capture_validation",
            "validate_capture",
            reason,
        )
    )


def _stable_ancestor(before: NativeObjectSnapshot, after: NativeObjectSnapshot) -> bool:
    left = before.facts
    right = after.facts
    return (
        before.object_role,
        before.ancestor_index,
        before.is_repository_root,
        before.is_immediate_parent,
        before.requested_canonical_volume_guid_path,
        left.canonical_volume_guid_path,
        left.canonical_dos_path,
        left.canonical_name_exact,
        left.unicode_nfc_exact,
        left.identity,
        left.file_attributes_hex,
        left.reparse_tag_hex,
        left.delete_pending,
        left.directory,
    ) == (
        after.object_role,
        after.ancestor_index,
        after.is_repository_root,
        after.is_immediate_parent,
        after.requested_canonical_volume_guid_path,
        right.canonical_volume_guid_path,
        right.canonical_dos_path,
        right.canonical_name_exact,
        right.unicode_nfc_exact,
        right.identity,
        right.file_attributes_hex,
        right.reparse_tag_hex,
        right.delete_pending,
        right.directory,
    )


def _validate_complete_state(
    state: _CaptureState,
    request: PermissionObservationRequest,
) -> None:
    if (
        len(state.ancestry_before) != state.ancestor_count
        or len(state.ancestry_after) != state.ancestor_count
        or len(state.ancestry_security_before) != state.ancestor_count
        or len(state.ancestry_security_after) != state.ancestor_count
        or state.subject_before is None
        or state.subject_after is None
        or state.streams is None
        or not state.eof_observed
        or state.subject_security_before is None
        or state.subject_security_after is None
        or state.effective_before is None
        or state.effective_after is None
        or state.access_token_before is None
        or state.access_token_after is None
        or len(state.access_checks) != 2
    ):
        raise _capture_failure("capture_validation_failed")
    if sha256_bytes(bytes(state.content)) != request.subject.expected_file_sha256:
        raise _capture_failure("content_hash_mismatch")
    if len(state.content) != request.subject.expected_file_bytes:
        raise _capture_failure("size_changed")
    if state.subject_before != state.subject_after:
        before = state.subject_before
        after = state.subject_after
        reason = "identity_changed" if before.facts.identity != after.facts.identity else "attributes_changed"
        if before.facts.end_of_file_bytes != after.facts.end_of_file_bytes:
            reason = "size_changed"
        if before.facts.canonical_volume_guid_path != after.facts.canonical_volume_guid_path:
            reason = "canonical_path_mismatch"
        raise _capture_failure(reason)
    for before, after in zip(state.ancestry_before, state.ancestry_after, strict=True):
        if not _stable_ancestor(before, after):
            raise _capture_failure(
                "identity_changed" if before.facts.identity != after.facts.identity else "canonical_path_mismatch"
            )
    if state.subject_security_before != state.subject_security_after:
        raise _capture_failure("descriptor_changed")
    if state.ancestry_security_before != state.ancestry_security_after:
        raise _capture_failure("descriptor_changed")
    if state.access_token_before != state.access_token_after:
        raise _capture_failure("token_changed")
    if _token_projection(state.effective_before.facts) != _token_projection(state.effective_after.facts):
        raise _capture_failure("current_effective_context_changed")
    if state.effective_before.selection_source != state.effective_after.selection_source:
        raise _capture_failure("current_effective_context_changed")
    if any(not resource.release_attempted or not resource.released for resource in state.resources):
        raise _capture_failure("resource_release_failed")
    events = _events(state, ())
    if tuple(event.acquisition_sequence for event in events) != tuple(range(len(events))):
        raise PermissionObservationProtocolError("resource acquisition sequence is not contiguous")


def _capture_result(
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    state: _CaptureState,
) -> CapturedPermissionObservation:
    assert state.subject_before is not None
    assert state.subject_after is not None
    assert state.streams is not None
    assert state.subject_security_before is not None
    assert state.subject_security_after is not None
    assert state.effective_before is not None
    assert state.effective_after is not None
    assert state.access_token_before is not None
    assert state.access_token_after is not None
    blank = CapturedPermissionObservation(
        schema=CAPTURE_SCHEMA,
        request=request,
        access_expectation=access,
        outcome="captured",
        ancestry_before=tuple(state.ancestry_before),
        subject_before=state.subject_before,
        stream_inventory=state.streams,
        content_evidence=ContentEvidence(
            "sha256",
            len(state.content),
            sha256_bytes(bytes(state.content)),
            "opaque_bytes_not_deserialized",
            True,
        ),
        subject_security_before=state.subject_security_before,
        ancestry_security_before=tuple(state.ancestry_security_before),
        effective_context_before=state.effective_before,
        access_check_token_before=state.access_token_before,
        access_checks=cast(tuple[NativeAccessCheckOutcome, NativeAccessCheckOutcome], tuple(state.access_checks)),
        subject_security_after=state.subject_security_after,
        ancestry_security_after=tuple(state.ancestry_security_after),
        access_check_token_after=state.access_token_after,
        effective_context_after=state.effective_after,
        subject_after=state.subject_after,
        ancestry_after=tuple(state.ancestry_after),
        resource_events=_events(state, ()),
        capture_status="capture_valid_policy_not_accepted",
        record_self_sha256="0" * 64,
    )
    return replace(blank, record_self_sha256=_self_hash(blank, "record_self_sha256"))


def _no_lease_failure(
    request: PermissionObservationRequest,
    access: PermissionAccessExpectation,
    state: _CaptureState,
    acquire: AuthorityAcquireOutcome,
) -> FailedPermissionObservation:
    if type(acquire.failure) is not ComponentFailureItem:
        raise PermissionObservationProtocolError("no-lease authority outcome lacks exact failure")
    authority_failure = _validate_failure_item(acquire.failure)
    if acquire.status == "refused":
        outcome: FailureOutcome = "refused"
        reason = "authority_acquire_refused"
    elif acquire.status == "failed":
        outcome = "failed"
        reason = "authority_acquire_failed"
    else:
        raise PermissionObservationProtocolError("granted authority returned no lease")
    expected_kind = "authority_refusal" if acquire.status == "refused" else "authority_failure"
    if (
        authority_failure.failure_kind != expected_kind
        or authority_failure.reason_code != reason
        or authority_failure.stage != "authority_acquisition"
        or authority_failure.component_operation != "acquire_authority_lease"
        or authority_failure.operation_ordinal is not None
    ):
        raise PermissionObservationProtocolError("authority acquisition failure is inconsistent")
    accounting = _accounting_without_finish(
        acquire_status=acquire.status,
        candidate=False,
        acquired=False,
        binding="not_applicable",
        reason=reason,
    )
    return _final_failed(
        request,
        access,
        outcome=outcome,
        primary=authority_failure,
        follow=(),
        state=state,
        accounting=accounting,
    )


def _observe_held_file_permission_core(
    request: PermissionObservationRequest,
    *,
    expected_request: PermissionObservationRequest,
    access_expectation: PermissionAccessExpectation,
    authority: PermissionObservationAuthority,
    native: PermissionNativeApi,
    authz_hook: _HeldTokenAuthzObserverHook | None,
    authz_hook_state: _HeldTokenAuthzHookState,
) -> CapturedPermissionObservation | FailedPermissionObservation:
    """Capture one held-file permission observation with an optional trusted hook."""

    validate_permission_observation_request(request, expected_request=expected_request)
    validate_permission_access_expectation(access_expectation, expected_request=expected_request)
    scope = _build_scope(request, access_expectation)
    state = _empty_state(len(request.subject.required_ancestor_canonical_volume_guid_paths))
    try:
        acquire = authority.acquire(request=request, scope=scope)
    except Exception as exc:
        raise PermissionObservationProtocolError("authority acquire raised") from exc
    if type(acquire) is not AuthorityAcquireOutcome:
        raise PermissionObservationProtocolError("authority acquire outcome is malformed")
    if not isinstance(acquire.status, str):
        raise PermissionObservationProtocolError("authority acquire status has the wrong type")
    if acquire.status in {"refused", "failed"}:
        if acquire.lease is not None:
            raise PermissionObservationProtocolError("no-lease outcome returned a lease")
        return _no_lease_failure(request, access_expectation, state, acquire)
    if acquire.status != "granted" or acquire.lease is None or acquire.failure is not None:
        raise PermissionObservationProtocolError("granted authority outcome is malformed")
    lease = acquire.lease
    structurally_valid, binding_reason = _lease_binding_reason(lease, scope)
    if not structurally_valid:
        primary = _failure(
            "lease_binding_failure",
            "lease_binding",
            "validate_authority_lease",
            "lease_malformed",
        )
        accounting = _accounting_without_finish(
            acquire_status="granted",
            candidate=True,
            acquired=False,
            binding="invalid",
            reason="lease_malformed",
        )
        return _final_failed(
            request,
            access_expectation,
            outcome="failed",
            primary=primary,
            follow=(),
            state=state,
            accounting=accounting,
        )
    typed_lease = cast(PermissionAuthorityLease, lease)
    if binding_reason != "none":
        primary = _failure(
            "lease_binding_failure",
            "lease_binding",
            "validate_authority_lease",
            binding_reason,
        )
        return _return_failure(
            request,
            access_expectation,
            state,
            lease=typed_lease,
            primary=primary,
            authority_reason=binding_reason,
            binding="invalid",
            finish_allowed=True,
        )

    ancestor_resources: list[_RegisteredResource] = []
    subject_resource: _RegisteredResource | None = None
    effective_before_resource: _RegisteredResource | None = None
    access_resource: _RegisteredResource | None = None
    effective_after_resource: _RegisteredResource | None = None
    authority_reason = "none"
    try:
        _active(typed_lease, scope, step="initial", ordinal=None, stage="ancestor_before")
        ancestors = request.subject.required_ancestor_canonical_volume_guid_paths
        for index, path in enumerate(ancestors):
            _active(typed_lease, scope, step="ancestor_open", ordinal=index, stage="ancestor_before")
            handle = _native_call(
                state,
                lambda path=path, *, context: native.open_existing_directory(
                    path,
                    specification=access_expectation.ancestor_open_specification,
                    context=context,
                ),
                request_self_sha256=request.request_self_sha256,
                stage="ancestor_before",
                operation="open_existing_directory",
                operation_ordinal=index,
            )
            _require_returned_handle(
                handle,
                stage="ancestor_before",
                operation="open_existing_directory",
                operation_ordinal=index,
            )
            resource = _register(state, handle, "directory_handle", "ancestor", index)
            ancestor_resources.append(resource)
            _active(typed_lease, scope, step="ancestor_before_query", ordinal=index, stage="ancestor_before")
            raw = _native_call(
                state,
                lambda resource=resource, *, context: native.query_object(
                    cast(NativeObjectHandle, resource.value),
                    context=context,
                ),
                request_self_sha256=request.request_self_sha256,
                stage="ancestor_before",
                operation="query_object",
                resource=resource,
                operation_ordinal=index,
            )
            snapshot = _label_object(
                raw,
                role="ancestor",
                ancestor_index=index,
                ancestor_count=len(ancestors),
                expected_path=path,
                expected_bytes=None,
                stage="ancestor_before",
                operation_ordinal=index,
                resource=resource,
            )
            state.ancestry_before.append(snapshot)
            state.opened.append(
                OpenedResourceIdentity("directory_handle", "ancestor", index, resource.acquisition_sequence, snapshot.facts.identity)
            )

        _active(typed_lease, scope, step="subject_open", ordinal=None, stage="subject_before")
        subject_handle = _native_call(
            state,
            lambda *, context: native.open_existing_subject(
                request.subject.canonical_volume_guid_path,
                specification=access_expectation.subject_open_specification,
                context=context,
            ),
            request_self_sha256=request.request_self_sha256,
            stage="subject_before",
            operation="open_existing_subject",
        )
        _require_returned_handle(
            subject_handle,
            stage="subject_before",
            operation="open_existing_subject",
        )
        subject_resource = _register(state, subject_handle, "file_handle", "subject", None)
        _active(typed_lease, scope, step="subject_before_query", ordinal=None, stage="subject_before")
        subject_raw = _native_call(
            state,
            lambda *, context: native.query_object(
                cast(NativeObjectHandle, subject_resource.value),
                context=context,
            ),
            request_self_sha256=request.request_self_sha256,
            stage="subject_before",
            operation="query_object",
            resource=subject_resource,
        )
        state.subject_before = _label_object(
            subject_raw,
            role="subject",
            ancestor_index=None,
            ancestor_count=len(ancestors),
            expected_path=request.subject.canonical_volume_guid_path,
            expected_bytes=request.subject.expected_file_bytes,
            stage="subject_before",
            operation_ordinal=None,
            resource=subject_resource,
        )
        state.opened.append(
            OpenedResourceIdentity("file_handle", "subject", None, subject_resource.acquisition_sequence, state.subject_before.facts.identity)
        )

        _active(typed_lease, scope, step="stream_query", ordinal=None, stage="stream_inventory")
        stream_value = _native_call(
            state,
            lambda *, context: native.query_streams(
                cast(NativeObjectHandle, subject_resource.value),
                context=context,
            ),
            request_self_sha256=request.request_self_sha256,
            stage="stream_inventory",
            operation="query_streams",
            resource=subject_resource,
        )
        state.streams = _validate_streams(stream_value, subject=state.subject_before, resource=subject_resource)

        expected_bytes = request.subject.expected_file_bytes
        chunks = (expected_bytes + CONTENT_CHUNK_BYTES - 1) // CONTENT_CHUNK_BYTES
        for ordinal in range(chunks):
            _active(typed_lease, scope, step="content_chunk", ordinal=ordinal, stage="content_read")
            offset = len(state.content)
            requested = min(CONTENT_CHUNK_BYTES, expected_bytes - offset)
            part = _native_call(
                state,
                lambda offset=offset, requested=requested, *, context: native.read_unnamed_at(
                    cast(NativeObjectHandle, subject_resource.value),
                    offset=offset,
                    requested_bytes=requested,
                    context=context,
                ),
                request_self_sha256=request.request_self_sha256,
                stage="content_read",
                operation="read_unnamed_at",
                resource=subject_resource,
                operation_ordinal=ordinal,
            )
            if type(part) is not bytes:
                raise _resource_failure(
                    "native_protocol_failure",
                    stage="content_read",
                    operation="read_unnamed_at",
                    resource=subject_resource,
                    failure_kind="native_protocol_failure",
                    operation_ordinal=ordinal,
                )
            if len(part) != requested:
                raise _resource_failure(
                    "short_read",
                    stage="content_read",
                    operation="read_unnamed_at",
                    resource=subject_resource,
                    operation_ordinal=ordinal,
                )
            state.content.extend(part)
        eof_ordinal = chunks
        _active(typed_lease, scope, step="content_chunk", ordinal=eof_ordinal, stage="content_read")
        eof = _native_call(
            state,
            lambda *, context: native.read_unnamed_at(
                cast(NativeObjectHandle, subject_resource.value),
                offset=expected_bytes,
                requested_bytes=1,
                context=context,
            ),
            request_self_sha256=request.request_self_sha256,
            stage="content_read",
            operation="read_unnamed_at",
            resource=subject_resource,
            operation_ordinal=eof_ordinal,
        )
        if eof != b"":
            raise _resource_failure(
                "content_grew",
                stage="content_read",
                operation="read_unnamed_at",
                resource=subject_resource,
                operation_ordinal=eof_ordinal,
            )
        state.eof_observed = True
        if sha256_bytes(bytes(state.content)) != request.subject.expected_file_sha256:
            raise _resource_failure(
                "content_hash_mismatch",
                stage="content_read",
                operation="read_unnamed_at",
                resource=subject_resource,
                operation_ordinal=eof_ordinal,
            )

        state.subject_security_before = _security(
            state,
            typed_lease,
            scope,
            native,
            request_self_sha256=request.request_self_sha256,
            handle_resource=subject_resource,
            snapshot=state.subject_before,
            after=False,
            operation_ordinal=0,
        )
        for index, (resource, snapshot) in enumerate(zip(ancestor_resources, state.ancestry_before, strict=True)):
            state.ancestry_security_before.append(
                _security(
                    state,
                    typed_lease,
                    scope,
                    native,
                    request_self_sha256=request.request_self_sha256,
                    handle_resource=resource,
                    snapshot=snapshot,
                    after=False,
                    operation_ordinal=1 + index,
                )
            )

        _active(typed_lease, scope, step="effective_context_before_open", ordinal=None, stage="effective_context_before")
        effective_open = _native_call(
            state,
            native.open_current_effective_token,
            request_self_sha256=request.request_self_sha256,
            stage="effective_context_before",
            operation="open_current_effective_token",
        )
        if type(effective_open) is not tuple or len(effective_open) != 2 or effective_open[0] not in {"thread_impersonation", "process_primary"}:
            raise _resource_failure(
                "native_protocol_failure",
                stage="effective_context_before",
                operation="open_current_effective_token",
                failure_kind="native_protocol_failure",
            )
        selection_before, effective_handle = effective_open
        _require_returned_handle(
            effective_handle,
            stage="effective_context_before",
            operation="open_current_effective_token",
        )
        effective_before_resource = _register(
            state, effective_handle, "effective_token_handle", "effective_context_before", None
        )
        state.effective_before = _query_token_snapshot(
            state,
            typed_lease,
            scope,
            native,
            request_self_sha256=request.request_self_sha256,
            resource=effective_before_resource,
            step="effective_context_before_query",
            stage="effective_context_before",
            role="effective_context",
            source=cast(Literal["thread_impersonation", "process_primary"], selection_before),
            duplicated_from=None,
        )

        _active(typed_lease, scope, step="access_token_duplicate", ordinal=None, stage="access_token")
        duplicate = _native_call(
            state,
            lambda *, context: native.duplicate_for_access_check(
                cast(NativeTokenHandle, effective_before_resource.value),
                context=context,
            ),
            request_self_sha256=request.request_self_sha256,
            stage="access_token",
            operation="duplicate_for_access_check",
            resource=effective_before_resource,
        )
        _require_returned_handle(
            duplicate,
            stage="access_token",
            operation="duplicate_for_access_check",
            resource=effective_before_resource,
        )
        access_resource = _register(state, duplicate, "access_check_token_handle", "access_check_duplicate", None)
        state.access_token_before = _query_token_snapshot(
            state,
            typed_lease,
            scope,
            native,
            request_self_sha256=request.request_self_sha256,
            resource=access_resource,
            step="access_token_before_query",
            stage="access_token",
            role="access_check_duplicate",
            source="duplicated_effective_context",
            duplicated_from=state.effective_before.facts.token_id_luid_hex,
        )
        if _token_projection(state.effective_before.facts) != _token_projection(state.access_token_before.facts):
            raise _resource_failure(
                "token_unsupported",
                stage="access_token",
                operation="query_token",
                resource=access_resource,
                failure_kind="observation_refusal",
            )

        immediate = state.ancestry_before[-1]
        bindings = (
            NativeAccessObjectBinding(
                "subject",
                "file",
                state.subject_before.facts.canonical_volume_guid_path,
                state.subject_before.facts.identity,
                state.subject_security_before.descriptor.self_relative_descriptor_sha256,
            ),
            NativeAccessObjectBinding(
                "immediate_parent",
                "directory",
                immediate.facts.canonical_volume_guid_path,
                immediate.facts.identity,
                state.ancestry_security_before[-1].descriptor.self_relative_descriptor_sha256,
            ),
        )
        descriptor_values = (
            state.subject_security_before.descriptor,
            state.ancestry_security_before[-1].descriptor,
        )
        descriptors = (
            bytes.fromhex(state.subject_security_before.descriptor.self_relative_descriptor_hex),
            bytes.fromhex(state.ancestry_security_before[-1].descriptor.self_relative_descriptor_hex),
        )
        mappings = (access_expectation.file_generic_mapping, access_expectation.directory_generic_mapping)
        for ordinal, (binding, descriptor, mapping) in enumerate(zip(bindings, descriptors, mappings, strict=True)):
            _active(typed_lease, scope, step="access_check", ordinal=ordinal, stage="access_check")
            raw_access = _native_call(
                state,
                lambda binding=binding, descriptor=descriptor, mapping=mapping, *, context: native.access_check(
                    descriptor=descriptor,
                    token=cast(NativeTokenHandle, access_resource.value),
                    object_binding=binding,
                    desired_access_hex=access_expectation.desired_access_hex,
                    generic_mapping=mapping,
                    context=context,
                ),
                request_self_sha256=request.request_self_sha256,
                stage="access_check",
                operation="access_check",
                resource=access_resource,
                operation_ordinal=ordinal,
            )
            if type(raw_access) is not NativeRawAccessCheckResult:
                raise _resource_failure(
                    "native_protocol_failure",
                    stage="access_check",
                    operation="access_check",
                    resource=access_resource,
                    failure_kind="native_protocol_failure",
                    operation_ordinal=ordinal,
                )
            try:
                checked = _validate_access_result(raw_access)
            except _NativeReturnStructuralError as exc:
                raise _resource_failure(
                    "native_protocol_failure",
                    stage="access_check",
                    operation="access_check",
                    resource=access_resource,
                    failure_kind="native_protocol_failure",
                    operation_ordinal=ordinal,
                ) from exc
            except PermissionObservationContractError as exc:
                raise _resource_failure(
                    "access_result_inconsistent",
                    stage="access_check",
                    operation="access_check",
                    resource=access_resource,
                    operation_ordinal=ordinal,
                ) from exc
            if checked.api_call_status == "failed":
                raise _StopObservation(
                    _failure(
                        "native_api_failure",
                        "access_check",
                        "access_check",
                        "native_api_failure",
                        kind=access_resource.kind,
                        role=access_resource.role,
                        resource_ordinal=access_resource.ordinal,
                        operation_ordinal=ordinal,
                        backend_error_hex=checked.api_failure_error_hex,
                    )
                )
            state.access_checks.append(
                NativeAccessCheckOutcome(
                    binding,
                    "access_check_duplicate",
                    state.access_token_before.facts.token_id_luid_hex,
                    state.access_token_before.facts.modified_id_luid_hex,
                    access_expectation.desired_access_hex,
                    mapping,
                    checked,
                )
            )

        if authz_hook is not None:
            try:
                authz_hook_state.value = authz_hook(
                    held_duplicate_token=cast(
                        NativeTokenHandle, access_resource.value
                    ),
                    duplicate_token_before_facts=state.access_token_before.facts,
                    descriptors=descriptor_values,
                    object_bindings=bindings,
                    observation_lease=typed_lease,
                )
            except Exception:
                authz_hook_state.status = "failed"
                authz_hook_state.failure = _failure(
                    "integration_failure",
                    "authz_integration",
                    "invoke_held_token_authz_hook",
                    "authz_hook_failed",
                )
            else:
                authz_hook_state.status = "returned"

        state.subject_security_after = _security(
            state,
            typed_lease,
            scope,
            native,
            request_self_sha256=request.request_self_sha256,
            handle_resource=subject_resource,
            snapshot=state.subject_before,
            after=True,
            operation_ordinal=0,
        )
        for index, (resource, snapshot) in enumerate(zip(ancestor_resources, state.ancestry_before, strict=True)):
            state.ancestry_security_after.append(
                _security(
                    state,
                    typed_lease,
                    scope,
                    native,
                    request_self_sha256=request.request_self_sha256,
                    handle_resource=resource,
                    snapshot=snapshot,
                    after=True,
                    operation_ordinal=1 + index,
                )
            )
        state.access_token_after = _query_token_snapshot(
            state,
            typed_lease,
            scope,
            native,
            request_self_sha256=request.request_self_sha256,
            resource=access_resource,
            step="access_token_after_query",
            stage="access_token_after",
            role="access_check_duplicate",
            source="duplicated_effective_context",
            duplicated_from=state.effective_before.facts.token_id_luid_hex,
        )
        _active(typed_lease, scope, step="object_after_query", ordinal=0, stage="object_after")
        subject_after_raw = _native_call(
            state,
            lambda *, context: native.query_object(
                cast(NativeObjectHandle, subject_resource.value),
                context=context,
            ),
            request_self_sha256=request.request_self_sha256,
            stage="object_after",
            operation="query_object",
            resource=subject_resource,
            operation_ordinal=0,
        )
        state.subject_after = _label_object(
            subject_after_raw,
            role="subject",
            ancestor_index=None,
            ancestor_count=len(ancestors),
            expected_path=request.subject.canonical_volume_guid_path,
            expected_bytes=request.subject.expected_file_bytes,
            stage="object_after",
            operation_ordinal=0,
            resource=subject_resource,
        )
        for index, (path, resource) in enumerate(zip(ancestors, ancestor_resources, strict=True)):
            ordinal = 1 + index
            _active(typed_lease, scope, step="object_after_query", ordinal=ordinal, stage="object_after")
            raw = _native_call(
                state,
                lambda resource=resource, *, context: native.query_object(
                    cast(NativeObjectHandle, resource.value),
                    context=context,
                ),
                request_self_sha256=request.request_self_sha256,
                stage="object_after",
                operation="query_object",
                resource=resource,
                operation_ordinal=ordinal,
            )
            state.ancestry_after.append(
                _label_object(
                    raw,
                    role="ancestor",
                    ancestor_index=index,
                    ancestor_count=len(ancestors),
                    expected_path=path,
                    expected_bytes=None,
                    stage="object_after",
                    operation_ordinal=ordinal,
                    resource=resource,
                )
            )

        _active(typed_lease, scope, step="effective_context_after_open", ordinal=None, stage="effective_context_after")
        effective_open_after = _native_call(
            state,
            native.open_current_effective_token,
            request_self_sha256=request.request_self_sha256,
            stage="effective_context_after",
            operation="open_current_effective_token",
        )
        if type(effective_open_after) is not tuple or len(effective_open_after) != 2 or effective_open_after[0] not in {"thread_impersonation", "process_primary"}:
            raise _resource_failure(
                "native_protocol_failure",
                stage="effective_context_after",
                operation="open_current_effective_token",
                failure_kind="native_protocol_failure",
            )
        selection_after, effective_handle_after = effective_open_after
        _require_returned_handle(
            effective_handle_after,
            stage="effective_context_after",
            operation="open_current_effective_token",
        )
        effective_after_resource = _register(
            state, effective_handle_after, "effective_token_handle", "effective_context_after", None
        )
        state.effective_after = _query_token_snapshot(
            state,
            typed_lease,
            scope,
            native,
            request_self_sha256=request.request_self_sha256,
            resource=effective_after_resource,
            step="effective_context_after_query",
            stage="effective_context_after",
            role="effective_context",
            source=cast(Literal["thread_impersonation", "process_primary"], selection_after),
            duplicated_from=None,
        )
    except _StopObservation as stopped:
        authority_reason = stopped.authority_reason
        cleanup = tuple(item for item in _cleanup(state) if item is not stopped.failure)
        return _return_failure(
            request,
            access_expectation,
            state,
            lease=typed_lease,
            primary=stopped.failure,
            authority_reason=authority_reason,
            binding="valid",
            finish_allowed=True,
            cleanup=cleanup,
        )

    cleanup = _cleanup(state)
    if cleanup:
        return _return_failure(
            request,
            access_expectation,
            state,
            lease=typed_lease,
            primary=cleanup[0],
            authority_reason="none",
            binding="valid",
            finish_allowed=True,
            cleanup=cleanup[1:],
        )
    try:
        _validate_complete_state(state, request)
        _active(typed_lease, scope, step="final", ordinal=None, stage="final_authority_check")
    except _StopObservation as stopped:
        return _return_failure(
            request,
            access_expectation,
            state,
            lease=typed_lease,
            primary=stopped.failure,
            authority_reason=stopped.authority_reason,
            binding="valid",
            finish_allowed=True,
        )
    finish = _finish(typed_lease, "succeeded")
    if finish.finish_status != "succeeded":
        primary = _finish_failure_item()
        accounting = _accounting_with_finish(
            binding="valid",
            reason="none",
            requested="succeeded",
            finish=finish,
        )
        return _final_failed(
            request,
            access_expectation,
            outcome="failed",
            primary=primary,
            follow=(),
            state=state,
            accounting=accounting,
        )
    return _capture_result(request, access_expectation, state)


def observe_held_file_permission(
    request: PermissionObservationRequest,
    *,
    expected_request: PermissionObservationRequest,
    access_expectation: PermissionAccessExpectation,
    authority: PermissionObservationAuthority,
    native: PermissionNativeApi,
) -> CapturedPermissionObservation | FailedPermissionObservation:
    """Capture one held-file permission observation through injected capabilities."""

    return _observe_held_file_permission_core(
        request,
        expected_request=expected_request,
        access_expectation=access_expectation,
        authority=authority,
        native=native,
        authz_hook=None,
        authz_hook_state=_HeldTokenAuthzHookState(),
    )


def _observe_held_file_permission_with_authz_hook(
    request: PermissionObservationRequest,
    *,
    expected_request: PermissionObservationRequest,
    access_expectation: PermissionAccessExpectation,
    authority: PermissionObservationAuthority,
    native: PermissionNativeApi,
    authz_hook: _HeldTokenAuthzObserverHook,
) -> _ObservedPermissionWithAuthzHook:
    """Run the private held-token hook without widening the public observer API."""

    if authz_hook is None or not callable(authz_hook):
        raise PermissionObservationContractError(
            "an exact private held-token Authz hook is required"
        )
    hook_state = _HeldTokenAuthzHookState()
    observed = _observe_held_file_permission_core(
        request,
        expected_request=expected_request,
        access_expectation=access_expectation,
        authority=authority,
        native=native,
        authz_hook=authz_hook,
        authz_hook_state=hook_state,
    )
    return _ObservedPermissionWithAuthzHook(
        observation_result=observed,
        hook_status=hook_state.status,
        hook_value=hook_state.value,
        hook_failure=hook_state.failure,
    )
