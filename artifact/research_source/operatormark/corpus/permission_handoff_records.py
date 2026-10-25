"""Closed, effect-free records for the durable permission handoff.

This module only validates caller-supplied values.  It does not read source
files, acquire authority, touch a store, or call a native API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
import re
import types
import unicodedata
from typing import Any, Literal, Protocol, TypeVar, Union, cast, get_args, get_origin, get_type_hints

from .canonical import (
    canonical_json_bytes,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    strict_canonical_json_loads,
    validate_self_hash,
)
from . import permission_authz_evidence_verifier as authz_verifier
from . import permission_authz_records as authz_records
from . import permission_authority as permission_authority
from . import permission_observation_windows as observation
from . import permission_principal_authz_windows as supplied_authz
from . import private_permission_evidence as evidence


SCHEMA_PREFIX = "operatormark.darcy_branch_c_v1.permission_handoff_"
PLAN_SCHEMA = SCHEMA_PREFIX + "plan.v1"
RECORD_SCHEMAS = {
    "start": SCHEMA_PREFIX + "start.v1",
    "outcome": SCHEMA_PREFIX + "outcome.v1",
    "acquisition": SCHEMA_PREFIX + "acquisition.v1",
    "verification": SCHEMA_PREFIX + "verification.v1",
    "index": SCHEMA_PREFIX + "index.v1",
    "receipt": SCHEMA_PREFIX + "receipt.v1",
}
WITNESS_FILENAMES = (
    "invocation-start.json",
    "invocation-outcome.json",
    "acquisition.json",
    "verification.json",
    "index.json",
    "receipt.json",
)
GRANT_OPERATIONS = ("observe", "observe_authz", "retain_authz_evidence", "retain_authz_retention_failure")
HANDOFF_SOURCE_RELATIVE_PATHS = (
    "operatormark/__init__.py",
    "operatormark/corpus/__init__.py",
    *authz_records.AUTHZ_SOURCE_RELATIVE_PATHS,
    "operatormark/corpus/permission_handoff_records.py",
    "operatormark/corpus/permission_handoff_capture.py",
    "operatormark/corpus/permission_handoff_sources.py",
    "operatormark/corpus/permission_handoff_verify.py",
    "operatormark/corpus/permission_handoff_sizing.py",
)
INPUT_ROLES = frozenset(
    {
        "base_primary",
        "base_failure_sidecar",
        "authz_primary",
        "authz_failure_sidecar",
        "journal_activation",
        "journal_reservation",
        "journal_terminal",
        "authority_input",
    }
)
FAILURE_STAGES = frozenset(
    {
        "acquire",
        "lease_binding",
        "open",
        "identity",
        "security",
        "read",
        "create",
        "write",
        "flush",
        "publish",
        "cleanup",
        "finish",
    }
)
FAILURE_OPERATIONS = frozenset(
    {
        "read_handoff_input",
        "retain_handoff_witness",
        "invoke_durable_authz",
        "capture_token_trace",
        "encode_witness",
    }
)
FAILURE_REASONS = frozenset(
    {
        "refused",
        "unavailable",
        "not_found",
        "collision",
        "mismatch",
        "malformed_output",
        "short_io",
        "raised",
        "cleanup_failed",
        "terminal_unconfirmed",
    }
)
TOKEN_METHODS = (
    "query_token_statistics",
    "initialize_resource_manager",
    "initialize_context_from_token",
    "access_check",
    "free_context",
    "free_resource_manager",
)
TOKEN_REASON_ORDER = (
    "capture_failed",
    "token_handle_mismatch",
    "incomplete_query_pair",
    "token_identity_mismatch",
    "malformed_native_return",
    "native_call_raised",
)
VERIFICATION_REASON_ORDER = (
    "original_return_unavailable",
    "token_witness_unavailable",
    "input_read_incomplete",
    "input_cleanup_failed",
    "base_evidence_unavailable",
    "journal_evidence_unavailable",
    "original_terminal_unconfirmed",
    "retention_failure_evidence",
    "content_verification_incomplete",
    "external_anchor_unavailable",
    "historical_handoff_corroborated",
)

_HEX_RE = re.compile(r"^(?:[0-9a-f]{2})*$")
_DWORD_RE = re.compile(r"^0x(?:0|[1-9a-f][0-9a-f]*)$")


class PermissionHandoffRecordError(ValueError):
    """A handoff plan or wire value is malformed or contradictory."""


@dataclass(frozen=True, slots=True)
class HandoffSourceFile:
    relative_path: str
    file_bytes: int
    file_sha256: str


@dataclass(frozen=True, slots=True)
class FileRef:
    root_binding_id: str
    relative_path: str
    file_bytes: int
    file_sha256: str


@dataclass(frozen=True, slots=True)
class ByteRecord:
    record_hex: str
    file_bytes: int
    file_sha256: str


@dataclass(frozen=True, slots=True)
class InputSlot:
    slot_id: str
    root_binding_id: str
    relative_path: str
    role: Literal[
        "base_primary",
        "base_failure_sidecar",
        "authz_primary",
        "authz_failure_sidecar",
        "journal_activation",
        "journal_reservation",
        "journal_terminal",
        "authority_input",
    ]
    expected_file_ref: FileRef | None


@dataclass(frozen=True, slots=True)
class AuthorityBinding:
    full_state4_authorization_file_sha256: str
    controlling_authorization_file_ref: FileRef
    enrollment_file_ref: FileRef
    profile_file_ref: FileRef
    accepted_roster_file_ref: FileRef


@dataclass(frozen=True, slots=True)
class JournalBinding:
    root_binding_id: str
    attempt_relative_path: str
    expected_activation: permission_authority.PermissionAttemptActivation
    expected_root_identity: observation.NativeFileIdentity
    expected_attempt_directory_identity: observation.NativeFileIdentity | None


@dataclass(frozen=True, slots=True)
class WitnessStoreExpectation:
    root_binding_id: str
    namespace_relative_path: str
    namespace_object_identity: str
    parent_relative_path: str
    parent_object_identity: str
    volume_identifier_commitment: str
    confidential_profile_file_ref: FileRef
    parent_security_descriptor_sha256: str
    created_object_security_descriptor_sha256: str
    creation_policy: str
    publication_policy: str


@dataclass(frozen=True, slots=True)
class HandoffPlan:
    schema: str
    specification_file_sha256: str
    authority_binding: AuthorityBinding
    expected_request: observation.PermissionObservationRequest
    expected_access: observation.PermissionAccessExpectation
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding
    expected_store: evidence.ConfidentialStoreExpectation
    expected_authz_destination: evidence.PermissionAuthzEvidenceDestination
    expected_base_destinations: tuple[
        evidence.PermissionEvidenceDestination,
        evidence.PermissionEvidenceDestination,
    ]
    expected_grants: Mapping[str, permission_authority.PermissionApprovalGrant]
    journal_binding: JournalBinding
    input_slots: tuple[InputSlot, ...]
    witness_store: WitnessStoreExpectation
    source_files: tuple[HandoffSourceFile, ...]
    plan_self_sha256: str

    def __post_init__(self) -> None:
        # Copy the independently supplied map: a frozen envelope must not retain
        # a caller-mutable grants dictionary.
        if isinstance(self.expected_grants, Mapping):
            object.__setattr__(self, "expected_grants", types.MappingProxyType(dict(self.expected_grants)))


@dataclass(frozen=True, slots=True)
class HandoffLeaseScope:
    operation: Literal["read_handoff_input", "retain_handoff_witness"]
    plan_self_sha256: str
    request_self_sha256: str
    full_state4_authorization_file_sha256: str
    root_binding_id: str
    slot_id: str
    namespace_identity: str
    parent_identity: str | None
    scope_self_sha256: str


class HandoffAuthorityLease(Protocol):
    @property
    def scope(self) -> HandoffLeaseScope: ...


@dataclass(frozen=True, slots=True)
class HandoffAcquireResult:
    status: Literal["granted", "refused", "failed"]
    lease: HandoffAuthorityLease | None
    reason: Literal["refused", "expired", "scope_mismatch", "unavailable", "malformed_output"] | None


@dataclass(frozen=True, slots=True)
class HandoffCheckpointResult:
    status: Literal["active", "refused", "failed"]
    reason: Literal["refused", "expired", "scope_mismatch", "unavailable", "malformed_output"] | None


@dataclass(frozen=True, slots=True)
class HandoffFinishResult:
    status: Literal["succeeded", "failed"]
    effect_certainty: Literal["confirmed", "not_applied", "unknown"]
    recorded_outcome: Literal["succeeded", "refused", "failed"] | None


@dataclass(frozen=True, slots=True)
class HandoffCheckpoint:
    name: Literal["initial", "before_open", "before_bytes", "before_publish", "final"]
    status: Literal["active", "refused", "failed"]
    reason: Literal["refused", "expired", "scope_mismatch", "unavailable", "malformed_output"] | None


@dataclass(frozen=True, slots=True)
class HandoffAuthorityAccounting:
    acquire_status: Literal["not_attempted", "granted", "refused", "failed"]
    lease_returned: bool
    lease_binding_status: Literal["not_applicable", "valid", "invalid"]
    checkpoints: tuple[HandoffCheckpoint, ...]
    finish_attempted: bool
    requested_outcome: Literal["succeeded", "refused", "failed"] | None
    finish_status: Literal["not_attempted", "succeeded", "failed"]
    effect_certainty: Literal["confirmed", "not_applied", "unknown"] | None
    recorded_outcome: Literal["succeeded", "refused", "failed"] | None


@dataclass(frozen=True, slots=True)
class ExceptionSummary:
    module: str
    qualified_type: str
    message: str | None
    message_unavailable_reason: Literal["unavailable", "unencodable"] | None


@dataclass(frozen=True, slots=True)
class HandoffFailure:
    stage: Literal[
        "acquire", "lease_binding", "open", "identity", "security", "read",
        "create", "write", "flush", "publish", "cleanup", "finish",
    ]
    operation: Literal[
        "read_handoff_input", "retain_handoff_witness", "invoke_durable_authz",
        "capture_token_trace", "encode_witness",
    ]
    reason: Literal[
        "refused", "unavailable", "not_found", "collision", "mismatch",
        "malformed_output", "short_io", "raised", "cleanup_failed",
        "terminal_unconfirmed",
    ]
    native_error_hex: str | None
    exception: ExceptionSummary | None


@dataclass(frozen=True, slots=True)
class CaptureFailure:
    stage: Literal["delegate_call", "token_capture", "runtime_capture"]
    reason: Literal["raised", "malformed_return", "copy_failed"]
    exception: ExceptionSummary | None


@dataclass(frozen=True, slots=True)
class TokenEvent:
    ordinal: int
    method: Literal[
        "query_token_statistics", "initialize_resource_manager",
        "initialize_context_from_token", "access_check", "free_context",
        "free_resource_manager",
    ]
    outcome: Literal["returned_valid", "returned_malformed", "raised"]
    statistics: supplied_authz.NativeTokenStatistics | None
    exception: ExceptionSummary | None


@dataclass(frozen=True, slots=True)
class TokenTrace:
    capture_status: Literal["complete", "partial", "unavailable"]
    events: tuple[TokenEvent, ...]
    same_borrowed_token: Literal["confirmed", "mismatch", "unavailable"]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HandoffObjectMetadata:
    root_binding_id: str
    relative_path: str
    canonical_volume_guid_path: str
    identity: observation.NativeFileIdentity
    parent_identity: observation.NativeFileIdentity
    namespace_identity: observation.NativeFileIdentity
    size_bytes: int
    security_descriptor_sha256: str
    confidential_profile_file_sha256: str | None
    volume_identifier_commitment: str


@dataclass(frozen=True, slots=True)
class HandoffReadResult:
    slot_id: str
    status: Literal["complete", "not_found", "unavailable", "partial", "inconsistent"]
    data: ByteRecord | None
    data_completeness: Literal["complete", "prefix", "unavailable"]
    before: HandoffObjectMetadata | None
    after: HandoffObjectMetadata | None
    authority_accounting: HandoffAuthorityAccounting
    cleanup_failures: tuple[HandoffFailure, ...]
    failure: HandoffFailure | None


@dataclass(frozen=True, slots=True)
class HandoffLastKnownObject:
    metadata: HandoffObjectMetadata
    file_bytes: int
    file_sha256: str


@dataclass(frozen=True, slots=True)
class HandoffWriteResult:
    slot_id: str
    status: Literal["retained", "collision", "failed", "unknown"]
    last_known_object: HandoffLastKnownObject | None
    publication_state: Literal["not_created", "creating", "published", "unknown"]
    effect_certainty: Literal["no_effect", "confirmed", "unknown"]
    ownership: Literal["no_object", "owned_by_this_invocation", "not_owned", "unknown"]
    handle_state: Literal["not_acquired", "released", "release_failed", "unknown"]
    authority_accounting: HandoffAuthorityAccounting
    cleanup_failures: tuple[HandoffFailure, ...]
    failure: HandoffFailure | None


@dataclass(frozen=True, slots=True)
class AcceptedHandoffAnchor:
    specification_file_sha256: str
    plan_self_sha256: str
    invocation_id: str
    receipt_file_bytes: int
    receipt_file_sha256: str
    handoff_source_vector_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedAuthzVerifierInputs:
    primary_bytes: bytes | None
    primary_snapshot: evidence.StoreObjectSnapshot | None
    sidecar_bytes: bytes | None
    sidecar_snapshot: evidence.StoreObjectSnapshot | None
    base_evidence: authz_verifier.IndependentBaseEvidence
    authority_evidence: authz_verifier.IndependentAuthorityEvidence
    expected_token_identity: supplied_authz.SuppliedTokenIdentity | None
    incomplete_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HandoffPhysicalFile:
    root_binding_id: str
    relative_path: str
    representation: Literal["loose_file", "container"]
    owner: Literal["accepted_attempt_M", "separate_B", "external_preexisting", "unresolved"]
    controls_subset: bool
    byte_count: int | None
    classification: Literal["exact", "conditional_exact", "upper_bound", "unresolved"]
    basis: str
    prerequisites: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HandoffContentCheck:
    content_result: authz_verifier.RetainedAuthzEvidenceVerification | None
    original_return_consistency: Literal[
        "consistent_success_return", "consistent_failure_return", "unavailable"
    ]
    reason_codes: tuple[str, ...]


_T = TypeVar("_T")


def _raise(message: str, cause: BaseException | None = None) -> None:
    error = PermissionHandoffRecordError(message)
    if cause is None:
        raise error
    raise error from cause


def _wire(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _wire(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [_wire(item) for item in value]
    if isinstance(value, list):
        return [_wire(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _wire(item) for key, item in value.items()}
    return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _raise(f"{name} must be an exact integer >= {minimum}")
    return cast(int, value)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or unicodedata.normalize("NFC", value) != value:
        _raise(f"{name} must be non-empty NFC text")
    return cast(str, value)


def _relative_path(value: object, *, name: str) -> str:
    path = _text(value, name=name)
    if (
        "\\" in path
        or ":" in path
        or path.startswith("/")
        or path.endswith("/")
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or any(part.endswith((".", " ")) or re.fullmatch(
            r"(?i)(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part
        ) for part in path.split("/"))
        or any(ord(char) < 32 or char in '<>"|?*' for char in path)
    ):
        _raise(f"{name} is not a canonical relative path")
    try:
        path.encode("utf-8")
    except UnicodeError as exc:
        _raise(f"{name} is not UTF-8 encodable", exc)
    return path


def _decode_as(annotation: object, value: object, *, name: str) -> object:
    """Decode a known annotation without accepting candidate type names."""
    if annotation is Any or annotation is object:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        if not any(type(value) is type(item) and value == item for item in args):
            _raise(f"{name} is outside its closed literal vocabulary")
        return value
    if origin in {types.UnionType, Union}:
        errors: list[BaseException] = []
        for option in args:
            try:
                return _decode_as(option, value, name=name)
            except PermissionHandoffRecordError as exc:
                errors.append(exc)
        _raise(f"{name} does not match any allowed alternative", errors[-1] if errors else None)
    if origin in {tuple, list}:
        if type(value) is not list:
            _raise(f"{name} must be a JSON array")
        fixed = origin is tuple and args and args[-1] is not Ellipsis
        if fixed and len(value) != len(args):
            _raise(f"{name} has the wrong tuple length")
        item_type = args[0] if args else object
        decoded = tuple(
            _decode_as(args[index] if fixed else item_type, item, name=f"{name}[{index}]")
            for index, item in enumerate(cast(list[object], value))
        )
        return decoded if origin is tuple else list(decoded)
    if origin in {dict, Mapping} or str(origin).endswith("collections.abc.Mapping"):
        if type(value) is not dict:
            _raise(f"{name} must be a JSON object")
        key_type, value_type = args or (str, object)
        result: dict[object, object] = {}
        for key, item in cast(dict[object, object], value).items():
            decoded_key = _decode_as(key_type, key, name=f"{name}.key")
            result[decoded_key] = _decode_as(value_type, item, name=f"{name}[{key!r}]")
        return result
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _decode_dataclass(annotation, value, name=name)
    if annotation in {str, int, bool, bytes}:
        if type(value) is not annotation:
            _raise(f"{name} has the wrong exact scalar type")
        return value
    if annotation is type(None):
        if value is not None:
            _raise(f"{name} must be null")
        return None
    if isinstance(annotation, type):
        if type(value) is not annotation:
            _raise(f"{name} has the wrong exact type")
        return value
    _raise(f"{name} uses an unsupported closed annotation")


def _decode_dataclass(cls: type[_T], value: object, *, name: str) -> _T:
    if type(value) is not dict:
        _raise(f"{name} must be a JSON object")
    mapping = cast(dict[str, object], value)
    try:
        require_exact_keys(mapping, (item.name for item in fields(cls)), name=name)
    except Exception as exc:
        # A closed union must be allowed to try its next explicitly permitted
        # dataclass shape (namespace effects and object effects differ).
        _raise(f"{name} has the wrong closed field vector", exc)
    hints = get_type_hints(cls)
    kwargs = {
        item.name: _decode_as(hints[item.name], mapping[item.name], name=f"{name}.{item.name}")
        for item in fields(cls)
    }
    try:
        result = cls(**{item.name: kwargs[item.name] for item in fields(cls) if item.init})
        for item in fields(cls):
            if not item.init and (type(kwargs[item.name]) is not type(getattr(result, item.name))
                                 or kwargs[item.name] != getattr(result, item.name)):
                _raise(f"{name}.{item.name} contradicts its fixed value")
        return result
    except Exception as exc:
        _raise(f"{name} could not be constructed", exc)


def _validate_file_ref(value: FileRef, *, name: str) -> None:
    if type(value) is not FileRef:
        _raise(f"{name} has the wrong exact type")
    _text(value.root_binding_id, name=f"{name}.root_binding_id")
    _relative_path(value.relative_path, name=f"{name}.relative_path")
    _exact_int(value.file_bytes, name=f"{name}.file_bytes")
    require_sha256(value.file_sha256, name=f"{name}.file_sha256")


def _validate_byte_record(value: ByteRecord, *, name: str) -> bytes:
    if type(value) is not ByteRecord or type(value.record_hex) is not str:
        _raise(f"{name} has the wrong exact type")
    if _HEX_RE.fullmatch(value.record_hex) is None:
        _raise(f"{name}.record_hex must be lowercase even-length hexadecimal")
    data = bytes.fromhex(value.record_hex)
    if type(value.file_bytes) is not int or value.file_bytes != len(data):
        _raise(f"{name}.file_bytes mismatch")
    require_sha256(value.file_sha256, name=f"{name}.file_sha256")
    if value.file_sha256 != sha256_bytes(data):
        _raise(f"{name}.file_sha256 mismatch")
    return data


def _validate_native_identity(value: observation.NativeFileIdentity, *, name: str) -> None:
    try:
        observation._validate_identity(value, name=name)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)


def _plan_payload(plan: HandoffPlan) -> dict[str, object]:
    payload = cast(dict[str, object], _wire(plan))
    payload.pop("plan_self_sha256", None)
    return payload


def validate_handoff_plan(plan: HandoffPlan) -> None:
    if type(plan) is not HandoffPlan:
        _raise("handoff plan has the wrong exact type")
    if plan.schema != PLAN_SCHEMA:
        _raise("handoff plan schema mismatch")
    require_sha256(plan.specification_file_sha256, name="plan.specification_file_sha256")
    require_sha256(plan.plan_self_sha256, name="plan.plan_self_sha256")
    if plan.plan_self_sha256 != sha256_bytes(canonical_json_bytes(_plan_payload(plan))):
        _raise("handoff plan self-hash mismatch")
    try:
        observation.validate_permission_observation_request(
            plan.expected_request, expected_request=plan.expected_request
        )
        observation.validate_permission_access_expectation(
            plan.expected_access, expected_request=plan.expected_request
        )
        authz_records.validate_permission_authz_evidence_binding(
            plan.expected_authz_binding,
            expected_request=plan.expected_request,
            expected_access=plan.expected_access,
        )
        evidence.validate_confidential_store_expectation(
            plan.expected_store, expected_request=plan.expected_request
        )
        evidence.validate_permission_authz_evidence_destination(
            plan.expected_authz_destination,
            expected_request=plan.expected_request,
            expected_store=plan.expected_store,
            expected_authz_binding=plan.expected_authz_binding,
        )
    except Exception as exc:
        _raise("handoff plan contains invalid existing expectations", exc)
    if type(plan.expected_base_destinations) is not tuple or len(plan.expected_base_destinations) != 2:
        _raise("expected_base_destinations must be the captured/failed pair")
    if tuple(item.payload_kind for item in plan.expected_base_destinations) != (
        "observation", "observation_failure"
    ):
        _raise("expected_base_destinations kinds or order mismatch")
    for item in plan.expected_base_destinations:
        try:
            evidence.validate_permission_evidence_destination(
                item, expected_request=plan.expected_request, expected_store=plan.expected_store
            )
        except Exception as exc:
            _raise("base destination is invalid", exc)
    if type(plan.expected_grants) is not types.MappingProxyType or set(plan.expected_grants) != set(GRANT_OPERATIONS):
        _raise("expected_grants keys mismatch")
    for operation, grant in plan.expected_grants.items():
        if type(grant) is not permission_authority.PermissionApprovalGrant:
            _raise(f"expected_grants[{operation}] has the wrong exact type")
        if grant.expected_scope.operation != operation:
            _raise(f"expected_grants[{operation}] operation mismatch")
        if grant.expected_access_expectation != plan.expected_access:
            _raise(f"expected_grants[{operation}] access mismatch")
        if operation.startswith("retain_authz") and (
            grant.expected_destination != plan.expected_authz_destination
            or grant.expected_store_expectation != plan.expected_store
        ):
            _raise(f"expected_grants[{operation}] destination/store mismatch")
        if grant.state4_authorization_file_sha256 != plan.authority_binding.full_state4_authorization_file_sha256:
            _raise(f"expected_grants[{operation}] A4 mismatch")
        try:
            permission_authority.validate_permission_approval_grant(
                grant, request=plan.expected_request, scope=grant.expected_scope
            )
        except Exception as exc:
            _raise(f"expected_grants[{operation}] is invalid", exc)
    binding = plan.authority_binding
    if type(binding) is not AuthorityBinding:
        _raise("authority_binding has the wrong exact type")
    require_sha256(
        binding.full_state4_authorization_file_sha256,
        name="authority_binding.full_state4_authorization_file_sha256",
    )
    if binding.full_state4_authorization_file_sha256 != plan.expected_request.context.state4_authorization_file_sha256:
        _raise("authority binding A4 mismatch")
    for name in (
        "controlling_authorization_file_ref", "enrollment_file_ref",
        "profile_file_ref", "accepted_roster_file_ref",
    ):
        _validate_file_ref(getattr(binding, name), name=f"authority_binding.{name}")
    if binding.controlling_authorization_file_ref.file_sha256 != binding.full_state4_authorization_file_sha256:
        _raise("controlling authorization file reference does not bind full A4")
    journal = plan.journal_binding
    if type(journal) is not JournalBinding:
        _raise("journal_binding has the wrong exact type")
    _text(journal.root_binding_id, name="journal_binding.root_binding_id")
    _relative_path(journal.attempt_relative_path, name="journal_binding.attempt_relative_path")
    try:
        permission_authority._validate_attempt_activation(journal.expected_activation)
    except Exception as exc:
        _raise("journal activation is invalid", exc)
    if (
        journal.expected_activation.state4_authorization_file_sha256
        != binding.full_state4_authorization_file_sha256
        or journal.expected_activation.attempt_id != plan.expected_request.attempt_id
    ):
        _raise("journal activation does not bind the plan")
    if any(permission_authority._attempt_activation(grant) != journal.expected_activation
           for grant in plan.expected_grants.values()):
        _raise("journal activation does not match every independently pinned grant")
    _validate_native_identity(journal.expected_root_identity, name="journal_binding.expected_root_identity")
    if journal.expected_attempt_directory_identity is not None:
        _validate_native_identity(
            journal.expected_attempt_directory_identity,
            name="journal_binding.expected_attempt_directory_identity",
        )
    if type(plan.input_slots) is not tuple or not plan.input_slots:
        _raise("input_slots must be a non-empty tuple")
    slot_ids: set[str] = set()
    paths: set[tuple[str, str]] = set()
    for index, slot in enumerate(plan.input_slots):
        if type(slot) is not InputSlot:
            _raise(f"input_slots[{index}] has the wrong exact type")
        _text(slot.slot_id, name=f"input_slots[{index}].slot_id")
        _text(slot.root_binding_id, name=f"input_slots[{index}].root_binding_id")
        _relative_path(slot.relative_path, name=f"input_slots[{index}].relative_path")
        if slot.role not in INPUT_ROLES:
            _raise(f"input_slots[{index}].role is invalid")
        if slot.role == "authority_input" and slot.expected_file_ref is None and slot.slot_id != "handoff-plan":
            _raise("preexisting authority input lacks its expected file reference")
        if slot.slot_id in slot_ids or (slot.root_binding_id, slot.relative_path) in paths:
            _raise("input slot identifier or root/path is duplicated")
        slot_ids.add(slot.slot_id)
        paths.add((slot.root_binding_id, slot.relative_path))
        if slot.expected_file_ref is not None:
            _validate_file_ref(slot.expected_file_ref, name=f"input_slots[{index}].expected_file_ref")
            if (
                slot.expected_file_ref.root_binding_id != slot.root_binding_id
                or slot.expected_file_ref.relative_path != slot.relative_path
            ):
                _raise("input slot expected reference address mismatch")
    plan_slots = [slot for slot in plan.input_slots if slot.slot_id == "handoff-plan"]
    specification_slots = [slot for slot in plan.input_slots if slot.slot_id == "handoff-specification"]
    if (len(plan_slots) != 1 or plan_slots[0].role != "authority_input"
            or plan_slots[0].expected_file_ref is not None):
        _raise("plan output needs one independently addressed handoff-plan slot without a circular file hash")
    if (len(specification_slots) != 1 or specification_slots[0].role != "authority_input"
            or specification_slots[0].expected_file_ref is None
            or specification_slots[0].expected_file_ref.file_sha256 != plan.specification_file_sha256):
        _raise("specification input must bind the independently adopted specification bytes")
    store = plan.witness_store
    if type(store) is not WitnessStoreExpectation:
        _raise("witness_store has the wrong exact type")
    _text(store.root_binding_id, name="witness_store.root_binding_id")
    _relative_path(store.namespace_relative_path, name="witness_store.namespace_relative_path")
    _relative_path(store.parent_relative_path, name="witness_store.parent_relative_path")
    namespace = "artifacts/darcy_branch_c_v1/private_corpus_v1/permission_handoff"
    parent = (f"{namespace}/s4-{binding.full_state4_authorization_file_sha256[:24]}/"
              f"{plan.expected_request.request_self_sha256}")
    if store.namespace_relative_path != namespace or store.parent_relative_path != parent:
        _raise("witness namespace or exact request parent path mismatch")
    for name in ("namespace_object_identity", "parent_object_identity"):
        _text(getattr(store, name), name=f"witness_store.{name}")
    for name in (
        "volume_identifier_commitment", "parent_security_descriptor_sha256",
        "created_object_security_descriptor_sha256",
    ):
        require_sha256(getattr(store, name), name=f"witness_store.{name}")
    _validate_file_ref(store.confidential_profile_file_ref, name="witness_store.confidential_profile_file_ref")
    if store.creation_policy != "security_descriptor_applied_atomically_at_create":
        _raise("witness creation policy mismatch")
    if store.publication_policy != "same_parent_same_volume_same_object_no_replace_preserve_security":
        _raise("witness publication policy mismatch")
    if type(plan.source_files) is not tuple or tuple(item.relative_path for item in plan.source_files) != HANDOFF_SOURCE_RELATIVE_PATHS:
        _raise("handoff source vector paths or order mismatch")
    for index, item in enumerate(plan.source_files):
        if type(item) is not HandoffSourceFile:
            _raise(f"source_files[{index}] has the wrong exact type")
        _relative_path(item.relative_path, name=f"source_files[{index}].relative_path")
        _exact_int(item.file_bytes, name=f"source_files[{index}].file_bytes")
        require_sha256(item.file_sha256, name=f"source_files[{index}].file_sha256")
    if tuple((item.relative_path, item.file_bytes, item.file_sha256) for item in plan.source_files[2:19]) != tuple(
        (item.relative_path, item.file_bytes, item.file_sha256) for item in plan.expected_authz_binding.source_files
    ):
        _raise("handoff and existing Authz source vectors disagree")


def derive_handoff_invocation_id(plan: HandoffPlan) -> str:
    validate_handoff_plan(plan)
    return sha256_bytes(
        b"operatormark-permission-handoff-invocation-v1\n"
        + plan.authority_binding.full_state4_authorization_file_sha256.encode("ascii")
        + b"\n"
        + plan.expected_request.request_self_sha256.encode("ascii")
        + b"\n"
        + plan.plan_self_sha256.encode("ascii")
    )


def handoff_record_commitment(record: Mapping[str, object], *, kind: str, expected_plan: HandoffPlan) -> str:
    data = canonical_handoff_record_bytes(record, kind=kind, expected_plan=expected_plan)
    return sha256_bytes(
        f"operatormark-permission-handoff-{kind}-v1".encode("utf-8")
        + b"\n"
        + expected_plan.authority_binding.full_state4_authorization_file_sha256.encode("ascii")
        + b"\n"
        + data
    )


def _validate_exception(value: ExceptionSummary | None, *, name: str) -> None:
    if value is None:
        return
    if type(value) is not ExceptionSummary:
        _raise(f"{name} has the wrong exact type")
    _text(value.module, name=f"{name}.module")
    _text(value.qualified_type, name=f"{name}.qualified_type")
    if value.message is None:
        if value.message_unavailable_reason not in {"unavailable", "unencodable"}:
            _raise(f"{name} null message needs an unavailable reason")
    elif type(value.message) is not str or value.message_unavailable_reason is not None:
        _raise(f"{name} message/reason combination is invalid")


def _validate_token_trace(value: object) -> TokenTrace:
    trace = value if type(value) is TokenTrace else _decode_dataclass(TokenTrace, value, name="token_trace")
    if trace.capture_status not in {"complete", "partial", "unavailable"}:
        _raise("token trace capture status is invalid")
    if trace.same_borrowed_token not in {"confirmed", "mismatch", "unavailable"}:
        _raise("token trace handle correspondence is invalid")
    if tuple(trace.reason_codes) != tuple(code for code in TOKEN_REASON_ORDER if code in trace.reason_codes) or len(set(trace.reason_codes)) != len(trace.reason_codes):
        _raise("token trace reason codes are not a closed ordered subset")
    for index, event in enumerate(trace.events):
        if type(event) is not TokenEvent or type(event.ordinal) is not int or event.ordinal != index:
            _raise("token event ordinals are not consecutive exact integers")
        if event.method not in TOKEN_METHODS or event.outcome not in {"returned_valid", "returned_malformed", "raised"}:
            _raise("token event vocabulary is invalid")
        if event.method == "query_token_statistics" and event.outcome == "returned_valid":
            if type(event.statistics) is not supplied_authz.NativeTokenStatistics:
                _raise("valid statistics query lacks exact statistics")
            try:
                supplied_authz._validate_statistics(event.statistics)
            except Exception as exc:
                _raise("token statistics are malformed", exc)
        elif event.statistics is not None:
            _raise("non-query token event invented statistics")
        if event.outcome == "raised":
            _validate_exception(event.exception, name=f"token_trace.events[{index}].exception")
            if event.exception is None:
                _raise("raised token event lacks exception summary")
        elif event.exception is not None:
            _raise("non-raised token event has an exception")
    if trace.capture_status == "unavailable" and trace.events:
        _raise("unavailable token trace must not contain events")
    cycle = ("query_token_statistics", "initialize_resource_manager", "initialize_context_from_token",
             "access_check", "query_token_statistics", "free_context", "free_resource_manager")
    if trace.capture_status == "complete":
        if (tuple(event.method for event in trace.events) != cycle * 2
                or any(event.outcome != "returned_valid" for event in trace.events)
                or trace.reason_codes or trace.same_borrowed_token != "confirmed"):
            _raise("complete token trace lacks the exact two successful evaluator cycles")
        stats = [event.statistics for event in trace.events if event.statistics is not None]
        if len(stats) != 4 or any(item != stats[0] for item in stats):
            _raise("complete token trace has inconsistent statistics")
    return trace


def _validate_original_authz_receipt(
    value: object, *, intended_bytes: bytes, expected_plan: HandoffPlan
) -> evidence.RetainedPermissionAuthzEvidence:
    receipt = _decode_dataclass(evidence.RetainedPermissionAuthzEvidence, value, name="retention success receipt")
    expected = {
        "schema": evidence.RETAINED_AUTHZ_EVIDENCE_SCHEMA,
        "payload_kind": "authz_evidence",
        "request_self_sha256": expected_plan.expected_request.request_self_sha256,
        "access_expectation_self_sha256": expected_plan.expected_access.expectation_self_sha256,
        "destination_self_sha256": expected_plan.expected_authz_destination.destination_self_sha256,
        "store_expectation_self_sha256": expected_plan.expected_store.expectation_self_sha256,
        "relative_path": expected_plan.expected_authz_destination.primary_accepted_relative_path,
        "file_bytes": len(intended_bytes),
        "file_sha256": sha256_bytes(intended_bytes),
        "commitment_domain": authz_records.PRIVATE_PERMISSION_AUTHZ_DOMAIN,
        "authorization_file_sha256": expected_plan.authority_binding.full_state4_authorization_file_sha256,
        "namespace_object_identity": expected_plan.expected_store.namespace_object_identity,
        "records_parent_object_identity": expected_plan.expected_store.records_parent_object_identity,
        "creation_security_descriptor_sha256": expected_plan.expected_store.required_created_object_security_descriptor_sha256,
        "confidential_store_profile_sha256": expected_plan.expected_store.confidential_store_profile_sha256,
        "held_object_readback_sha256": sha256_bytes(intended_bytes),
        "publication_state": "retained_authz_evidence_only_not_permission_accepted",
    }
    for name, expected_value in expected.items():
        if type(getattr(receipt, name)) is not type(expected_value) or getattr(receipt, name) != expected_value:
            _raise(f"retention success receipt {name} mismatch")
    if type(receipt.store_object_identity) is not str or re.fullmatch(
        r"win32-file-id-v1:0x(?:0|[1-9a-f][0-9a-f]{0,15}):[0-9a-f]{32}",
        receipt.store_object_identity,
    ) is None:
        _raise("retention success receipt object identity is malformed")
    try:
        expected_commitment = authz_records.private_permission_authz_commitment(
            intended_bytes,
            authorization_file_sha256=expected_plan.authority_binding.full_state4_authorization_file_sha256,
            expected_request=expected_plan.expected_request,
            expected_access=expected_plan.expected_access,
            expected_binding=expected_plan.expected_authz_binding,
            expected_destination_self_sha256=expected_plan.expected_authz_destination.destination_self_sha256,
            expected_store_expectation_self_sha256=expected_plan.expected_store.expectation_self_sha256,
        )
        if receipt.private_authz_commitment != expected_commitment:
            _raise("retention success receipt commitment mismatch")
        validate_self_hash(asdict(receipt), "receipt_self_sha256")
    except Exception as exc:
        if isinstance(exc, PermissionHandoffRecordError):
            raise
        _raise("retention success receipt is invalid", exc)
    return receipt


def _decode_original_authz_failure(
    value: object, *, intended_bytes: bytes, expected_plan: HandoffPlan
) -> evidence.PermissionAuthzEvidenceRetentionFailure:
    result = _decode_dataclass(
        evidence.PermissionAuthzEvidenceRetentionFailure,
        value,
        name="retention failure result",
    )
    draft = evidence.PermissionAuthzEvidenceRetentionFailureDraft(
        failed_primary_payload_kind=result.failed_primary_payload_kind,
        request_self_sha256=result.request_self_sha256,
        access_expectation_self_sha256=result.access_expectation_self_sha256,
        destination_self_sha256=result.destination_self_sha256,
        store_expectation_self_sha256=result.store_expectation_self_sha256,
        primary_failure=result.primary_failure,
        ordered_follow_on_failures=result.ordered_follow_on_failures,
        main_authority_accounting=result.main_authority_accounting,
        payload_effects=result.payload_effects,
        payload_object_state=result.payload_object_state,
        acceptance_emitted=result.acceptance_emitted,
    )
    try:
        evidence._validate_authz_retention_failure_result(
            result,
            draft=draft,
            intended_authz_record=intended_bytes,
            expected_request=expected_plan.expected_request,
            expected_access=expected_plan.expected_access,
            expected_store=expected_plan.expected_store,
            expected_destination=expected_plan.expected_authz_destination,
            expected_authz_binding=expected_plan.expected_authz_binding,
        )
    except Exception as exc:
        _raise("retention failure result is invalid", exc)
    return result


def _decode_original_runtime(value: object, *, expected_plan: HandoffPlan) -> dict[str, object]:
    if type(value) is not dict:
        _raise("runtime_projection must be an exact JSON object")
    record = cast(dict[str, object], value)
    require_exact_keys(
        record,
        (
            "persistence_status", "intended_authz_record", "retention_result_kind",
            "retention_result", "failure_reason", "permission_acceptance_emitted",
        ),
        name="runtime_projection",
    )
    if record["persistence_status"] not in {"not_attempted", "not_retained", "retained", "retention_unknown"}:
        _raise("runtime persistence status is invalid")
    if record["permission_acceptance_emitted"] is not False:
        _raise("runtime projection emitted permission acceptance")
    intended_value = record["intended_authz_record"]
    intended: ByteRecord | None
    intended_bytes: bytes | None
    if intended_value is None:
        intended = None
        intended_bytes = None
    else:
        intended = _decode_dataclass(ByteRecord, intended_value, name="runtime intended Authz record")
        intended_bytes = _validate_byte_record(intended, name="runtime intended Authz record")
        try:
            evidence._validate_authz_record_for_retention(
                intended_bytes,
                expected_request=expected_plan.expected_request,
                expected_access=expected_plan.expected_access,
                expected_authz_binding=expected_plan.expected_authz_binding,
                expected_destination=expected_plan.expected_authz_destination,
                expected_store=expected_plan.expected_store,
            )
        except Exception as exc:
            _raise("runtime intended Authz record is invalid", exc)
    kind = record["retention_result_kind"]
    decoded_result: object = None
    if kind == "success_receipt":
        if intended_bytes is None:
            _raise("success receipt lacks intended Authz bytes")
        decoded_result = _validate_original_authz_receipt(
            record["retention_result"], intended_bytes=intended_bytes, expected_plan=expected_plan
        )
    elif kind == "retention_failure":
        if intended_bytes is None:
            _raise("retention failure lacks intended Authz bytes")
        decoded_result = _decode_original_authz_failure(
            record["retention_result"], intended_bytes=intended_bytes, expected_plan=expected_plan
        )
    elif kind == "not_returned":
        if record["retention_result"] is not None:
            _raise("not_returned runtime has a retention result")
    else:
        _raise("runtime retention result kind is invalid")
    failure_reason = record["failure_reason"]
    if failure_reason not in {
        None,
        "authz_bundle_build_failed",
        "authz_retention_catalog_activation_failed",
        "authz_retention_integration_failed",
    }:
        _raise("runtime failure reason is invalid")
    status = record["persistence_status"]
    if status == "retained" and (kind != "success_receipt" or failure_reason is not None):
        _raise("retained runtime lacks its exact success receipt")
    if status == "not_retained" and (kind != "retention_failure" or failure_reason is not None):
        _raise("not-retained runtime lacks its exact failure result")
    if status == "not_attempted":
        if kind != "not_returned" or not (
            (failure_reason == "authz_bundle_build_failed" and intended is None)
            or (failure_reason == "authz_retention_catalog_activation_failed" and intended is not None)
        ):
            _raise("not-attempted runtime is contradictory")
    if kind == "retention_failure":
        result = cast(evidence.PermissionAuthzEvidenceRetentionFailure, decoded_result)
        accountings = [result.main_authority_accounting]
        states = [result.payload_object_state]
        effects = list(result.payload_effects)
        if result.failure_record_authority_accounting is not None:
            accountings.append(result.failure_record_authority_accounting)
        if result.failure_record_object_state is not None:
            states.append(result.failure_record_object_state)
        if result.failure_record_effects is not None:
            effects.extend(result.failure_record_effects)
        unknown = (any(a.effect_certainty == "unknown" for a in accountings)
                   or any(s.effect_certainty == "unknown" or s.handle_state == "unknown" for s in states)
                   or any(getattr(e, "effect_certainty", None) == "unknown" for e in effects))
        if status != ("retention_unknown" if unknown else "not_retained") or failure_reason is not None:
            _raise("runtime status contradicts original effect accounting")
    if status == "retention_unknown" and kind != "retention_failure":
        if kind != "not_returned" or intended is None or failure_reason != "authz_retention_integration_failed":
            _raise("unknown runtime contradicts the integration-failure branch")
    return {**record, "intended_authz_record": intended, "retention_result": decoded_result}


def _validate_predecessor_refs(value: object, *, expected_plan: HandoffPlan) -> tuple[FileRef, ...]:
    if type(value) is not list:
        _raise("predecessor_file_refs must be an array")
    refs = tuple(
        _decode_dataclass(FileRef, item, name=f"predecessor_file_refs[{index}]")
        for index, item in enumerate(cast(list[object], value))
    )
    expected = _expected_predecessor_refs(expected_plan)
    if refs != expected:
        _raise("predecessor file references or order mismatch")
    for index, ref in enumerate(refs):
        _validate_file_ref(ref, name=f"predecessor_file_refs[{index}]")
    return refs


def _expected_predecessor_refs(expected_plan: HandoffPlan) -> tuple[FileRef, ...]:
    # The prepared plan pins its future serialization's address, not its own
    # file digest. Its bytes become an existing external input before run_once.
    # This is acyclic and does not create or authorize a plan file here.
    plan_slot = next(slot for slot in expected_plan.input_slots if slot.slot_id == "handoff-plan")
    specification_slot = next(slot for slot in expected_plan.input_slots if slot.slot_id == "handoff-specification")
    plan_bytes = canonical_json_bytes(_wire(expected_plan)) + b"\n"
    return (
        FileRef(plan_slot.root_binding_id, plan_slot.relative_path, len(plan_bytes), sha256_bytes(plan_bytes)),
        cast(FileRef, specification_slot.expected_file_ref),
        expected_plan.authority_binding.enrollment_file_ref,
        expected_plan.authority_binding.profile_file_ref,
        expected_plan.authority_binding.accepted_roster_file_ref,
    )


_COMMON_KEYS = (
    "schema", "invocation_id", "plan_self_sha256", "request_self_sha256",
    "state4_authorization_file_sha256",
)
_RECORD_KEYS = {
    "start": (*_COMMON_KEYS, "dispatch_may_follow", "predecessor_file_refs", "permission_acceptance_emitted", "record_self_sha256"),
    "outcome": (*_COMMON_KEYS, "start_file_ref", "call_outcome", "runtime_projection", "exception", "token_trace", "capture_failures", "permission_acceptance_emitted", "record_self_sha256"),
    "acquisition": (*_COMMON_KEYS, "outcome_file_ref", "acquisition_mode", "ordered_reads", "permission_acceptance_emitted", "record_self_sha256"),
    "verification": (*_COMMON_KEYS, "outcome_file_ref", "acquisition_file_ref", "content_result", "original_return_consistency", "reason_codes", "external_anchor_checked", "permission_acceptance_emitted", "record_self_sha256"),
    "index": (*_COMMON_KEYS, "ordered_witness_file_refs", "ordered_evidence_file_refs", "physical_inventory", "permission_acceptance_emitted", "record_self_sha256"),
    "receipt": (*_COMMON_KEYS, "index_file_bytes", "index_file_sha256", "index_commitment", "specification_file_sha256", "handoff_source_vector_sha256", "disposition", "permission_acceptance_emitted", "record_self_sha256"),
}


def _validate_record_kind(record: Mapping[str, object], *, kind: str, expected_plan: HandoffPlan) -> None:
    validate_handoff_plan(expected_plan)
    if kind not in RECORD_SCHEMAS:
        _raise("unknown handoff record kind")
    if type(record) is not dict:
        _raise("handoff record must be an exact dict")
    require_exact_keys(record, _RECORD_KEYS[kind], name=f"handoff {kind} record")
    expected_common = {
        "schema": RECORD_SCHEMAS[kind],
        "invocation_id": derive_handoff_invocation_id(expected_plan),
        "plan_self_sha256": expected_plan.plan_self_sha256,
        "request_self_sha256": expected_plan.expected_request.request_self_sha256,
        "state4_authorization_file_sha256": expected_plan.authority_binding.full_state4_authorization_file_sha256,
        "permission_acceptance_emitted": False,
    }
    for name, expected in expected_common.items():
        if type(record[name]) is not type(expected) or record[name] != expected:
            _raise(f"handoff {kind} {name} mismatch")
    try:
        validate_self_hash(record, "record_self_sha256")
    except Exception as exc:
        _raise(f"handoff {kind} self-hash is invalid", exc)
    if kind == "start":
        if record["dispatch_may_follow"] is not True:
            _raise("start dispatch_may_follow must be true")
        _validate_predecessor_refs(record["predecessor_file_refs"], expected_plan=expected_plan)
    elif kind == "outcome":
        _validate_file_ref(_decode_dataclass(FileRef, record["start_file_ref"], name="start_file_ref"), name="start_file_ref")
        call_outcome = record["call_outcome"]
        if call_outcome not in {"returned_valid", "returned_malformed", "returned_unavailable", "raised"}:
            _raise("outcome call_outcome is invalid")
        if call_outcome == "returned_valid":
            if record["runtime_projection"] is None:
                _raise("valid return lacks runtime projection")
            _decode_original_runtime(record["runtime_projection"], expected_plan=expected_plan)
        elif record["runtime_projection"] is not None:
            _raise("non-valid return has a runtime projection")
        exception = None if record["exception"] is None else _decode_dataclass(ExceptionSummary, record["exception"], name="outcome.exception")
        _validate_exception(exception, name="outcome.exception")
        if (call_outcome == "raised") != (exception is not None):
            _raise("outcome exception/call outcome mismatch")
        _validate_token_trace(record["token_trace"])
        if type(record["capture_failures"]) is not list:
            _raise("capture_failures must be an array")
        for index, item in enumerate(cast(list[object], record["capture_failures"])):
            failure = _decode_dataclass(CaptureFailure, item, name=f"capture_failures[{index}]")
            _validate_exception(failure.exception, name=f"capture_failures[{index}].exception")
    elif kind == "acquisition":
        _validate_file_ref(_decode_dataclass(FileRef, record["outcome_file_ref"], name="outcome_file_ref"), name="outcome_file_ref")
        if record["acquisition_mode"] != "independent_reader" or type(record["ordered_reads"]) is not list:
            _raise("acquisition envelope mismatch")
        reads = tuple(
            _decode_dataclass(HandoffReadResult, item, name=f"ordered_reads[{index}]")
            for index, item in enumerate(cast(list[object], record["ordered_reads"]))
        )
        if tuple(item.slot_id for item in reads) != tuple(slot.slot_id for slot in expected_plan.input_slots):
            _raise("acquisition read slots or order mismatch")
    elif kind == "verification":
        for name in ("outcome_file_ref", "acquisition_file_ref"):
            _validate_file_ref(_decode_dataclass(FileRef, record[name], name=name), name=name)
        if record["original_return_consistency"] not in {"consistent_success_return", "consistent_failure_return", "unavailable"}:
            _raise("verification return consistency is invalid")
        if record["external_anchor_checked"] is not False:
            _raise("pre-anchor verification claims an external anchor")
        _validate_reason_codes(record["reason_codes"])
        if record["content_result"] is not None:
            _decode_dataclass(authz_verifier.RetainedAuthzEvidenceVerification, record["content_result"], name="content_result")
    elif kind == "index":
        if type(record["ordered_witness_file_refs"]) is not list or len(record["ordered_witness_file_refs"]) != 4:
            _raise("index witness inventory must contain four predecessors")
        refs = tuple(
            _decode_dataclass(FileRef, item, name=f"ordered_witness_file_refs[{index}]")
            for index, item in enumerate(cast(list[object], record["ordered_witness_file_refs"]))
        )
        expected_names = WITNESS_FILENAMES[:4]
        if tuple(item.relative_path.rsplit("/", 1)[-1] for item in refs) != expected_names:
            _raise("index witness reference order mismatch")
        if type(record["ordered_evidence_file_refs"]) is not list or type(record["physical_inventory"]) is not list:
            _raise("index evidence inventory has wrong type")
        for index, item in enumerate(cast(list[object], record["ordered_evidence_file_refs"])):
            _validate_file_ref(_decode_dataclass(FileRef, item, name=f"ordered_evidence_file_refs[{index}]"), name=f"ordered_evidence_file_refs[{index}]")
        for index, item in enumerate(cast(list[object], record["physical_inventory"])):
            _decode_dataclass(HandoffPhysicalFile, item, name=f"physical_inventory[{index}]")
    else:
        for name in (
            "index_file_bytes", "index_file_sha256", "index_commitment",
            "specification_file_sha256", "handoff_source_vector_sha256",
        ):
            if name == "index_file_bytes":
                _exact_int(record[name], name=name)
            else:
                require_sha256(record[name], name=name)
        if record["specification_file_sha256"] != expected_plan.specification_file_sha256:
            _raise("receipt specification mismatch")
        if record["disposition"] != "permission_handoff_evidence_only":
            _raise("receipt disposition mismatch")


def _validate_reason_codes(value: object) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        _raise("verification reason_codes must be an array of strings")
    result = tuple(cast(list[str], value))
    if result != tuple(code for code in VERIFICATION_REASON_ORDER if code in result) or len(set(result)) != len(result):
        _raise("verification reason codes are not a closed ordered subset")
    return result


def canonical_handoff_record_bytes(
    record: Mapping[str, object], *, kind: str, expected_plan: HandoffPlan
) -> bytes:
    wire = _wire(record)
    if type(wire) is not dict:
        _raise("handoff record root must be an object")
    typed = cast(dict[str, object], wire)
    try:
        _validate_record_kind(typed, kind=kind, expected_plan=expected_plan)
    except PermissionHandoffRecordError:
        raise
    except Exception as exc:
        _raise(f"handoff {kind} record validation failed", exc)
    return canonical_json_bytes(typed) + b"\n"


def strict_handoff_record_loads(
    data: bytes, *, kind: str, expected_plan: HandoffPlan
) -> dict[str, object]:
    if type(data) is not bytes or not data.endswith(b"\n") or data.endswith(b"\n\n"):
        _raise("handoff record must end in exactly one LF")
    try:
        record = strict_canonical_json_loads(data[:-1])
    except Exception as exc:
        _raise("handoff record is not canonical JSON", exc)
    try:
        _validate_record_kind(record, kind=kind, expected_plan=expected_plan)
    except PermissionHandoffRecordError:
        raise
    except Exception as exc:
        _raise(f"handoff {kind} record validation failed", exc)
    if canonical_json_bytes(record) + b"\n" != data:
        _raise("handoff record did not round-trip to exact canonical bytes")
    return cast(dict[str, object], record)


def _seal_record(payload: Mapping[str, object]) -> dict[str, object]:
    if "record_self_sha256" in payload:
        _raise("record payload already has a self-hash")
    result = dict(payload)
    result["record_self_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def _common_record_fields(plan: HandoffPlan, kind: str) -> dict[str, object]:
    return {
        "schema": RECORD_SCHEMAS[kind],
        "invocation_id": derive_handoff_invocation_id(plan),
        "plan_self_sha256": plan.plan_self_sha256,
        "request_self_sha256": plan.expected_request.request_self_sha256,
        "state4_authorization_file_sha256": plan.authority_binding.full_state4_authorization_file_sha256,
    }


def _byte_record(data: bytes) -> ByteRecord:
    return ByteRecord(data.hex(), len(data), sha256_bytes(data))


def _file_ref(root_binding_id: str, relative_path: str, data: bytes) -> FileRef:
    return FileRef(root_binding_id, relative_path, len(data), sha256_bytes(data))


def handoff_source_vector_sha256(plan: HandoffPlan) -> str:
    validate_handoff_plan(plan)
    return sha256_bytes(canonical_json_bytes(_wire(plan.source_files)))
