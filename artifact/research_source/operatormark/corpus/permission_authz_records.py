"""Canonical, effect-free records for durable held-token Authz evidence.

The values in this module are wire data only.  Importing or validating them does
not acquire authority, touch a filesystem, call a native API, or inspect source
files.  In particular this module deliberately has no dependency on the
controller, issuer, private retention implementation, or retained-byte verifier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
import re
import unicodedata
from typing import Literal, TypeAlias, cast

from .canonical import (
    canonical_json_bytes,
    require_exact_keys,
    require_git_oid,
    require_sha256,
    sha256_bytes,
    strict_canonical_json_loads,
    validate_self_hash,
    with_self_hash,
)
from . import permission_observation_windows as observation
from . import permission_principal_authz_windows as supplied_authz


AUTHZ_EVIDENCE_BINDING_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_authz_evidence_binding.v1"
)
PRIVATE_PERMISSION_AUTHZ_SCHEMA = (
    "operatormark.darcy_branch_c_v1.private_permission_authz_evidence.v1"
)
PRIVATE_PERMISSION_AUTHZ_DOMAIN = (
    "operatormark-state4-private-permission-authz-evidence-v1"
)

AUTHZ_SOURCE_RELATIVE_PATHS = (
    "operatormark/corpus/canonical.py",
    "operatormark/corpus/permission_approval_resolver.py",
    "operatormark/corpus/permission_assessment.py",
    "operatormark/corpus/permission_authority.py",
    "operatormark/corpus/permission_authority_journal.py",
    "operatormark/corpus/permission_authz_evidence_verifier.py",
    "operatormark/corpus/permission_authz_records.py",
    "operatormark/corpus/permission_controller.py",
    "operatormark/corpus/permission_dacl_policy.py",
    "operatormark/corpus/permission_evidence_verifier.py",
    "operatormark/corpus/permission_native_win32.py",
    "operatormark/corpus/permission_observation_windows.py",
    "operatormark/corpus/permission_principal_authz_windows.py",
    "operatormark/corpus/permission_store_native_win32.py",
    "operatormark/corpus/permission_store_windows.py",
    "operatormark/corpus/private_permission_evidence.py",
    "operatormark/corpus/storage.py",
)


class PermissionAuthzRecordError(ValueError):
    """A proposed durable Authz wire value is malformed or contradictory."""


@dataclass(frozen=True, slots=True)
class PermissionAuthzSourceFile:
    relative_path: str
    file_bytes: int
    file_sha256: str


@dataclass(frozen=True, slots=True)
class PermissionAuthzEvidenceBinding:
    schema: str
    request_self_sha256: str
    access_expectation_self_sha256: str
    protocol_file_sha256: str
    state4_authorization_file_sha256: str
    implementation_commit: str
    implementation_tree: str
    expected_operator_sid: observation.NativeSid
    source_files: tuple[PermissionAuthzSourceFile, ...]
    binding_self_sha256: str


@dataclass(frozen=True, slots=True)
class PermissionAuthzBaseObservation:
    outcome: Literal["captured", "failed", "unavailable"]
    record_hex: str | None
    record_file_bytes: int | None
    record_file_sha256: str | None
    record_unavailable_reason: Literal[
        "not_returned", "malformed_result", "serialization_failed"
    ] | None
    retention_result_kind: Literal[
        "captured_receipt", "failure_receipt", "retention_failure", "not_returned"
    ]
    retention_result: Mapping[str, object] | None
    retention_result_unavailable_reason: Literal[
        "not_returned", "malformed_result"
    ] | None


@dataclass(frozen=True, slots=True)
class PermissionAuthzHeldInputs:
    duplicate_token_before_facts: observation.NativeTokenFacts
    descriptors: tuple[
        observation.NativeSecurityDescriptor,
        observation.NativeSecurityDescriptor,
    ]
    object_bindings: tuple[
        observation.NativeAccessObjectBinding,
        observation.NativeAccessObjectBinding,
    ]


@dataclass(frozen=True, slots=True)
class PermissionAuthzChildAccounting:
    acquire_status: Literal["not_attempted", "granted", "refused", "failed"]
    lease_candidate_returned: bool
    acquire_failure: observation.ComponentFailureItem | None
    lease_binding_status: Literal["not_applicable", "valid", "invalid"]
    checkpoint_outcomes: tuple[observation.AuthorityCheckOutcome, ...]
    finish_attempted: bool
    requested_outcome: Literal["succeeded", "refused", "failed"] | None
    finish_outcome: observation.AuthorityFinishOutcome | None


@dataclass(frozen=True, slots=True)
class PermissionAuthzWireEvidence:
    status: Literal["not_invoked", "incomplete", "completed"]
    result_prefix: tuple[supplied_authz.SuppliedTokenAccessResult, ...]
    expected_token_identity: supplied_authz.SuppliedTokenIdentity | None
    failure_reason_codes: tuple[str, ...]
    child_accounting: PermissionAuthzChildAccounting


@dataclass(frozen=True, slots=True)
class PermissionOperationEvidenceProjection:
    operation: Literal["observe", "observe_authz"]
    scope_self_sha256: str
    grant_binding_sha256: str | None
    parent_reservation_id: str | None
    reservation_call_state: Literal[
        "not_called", "returned_valid", "returned_malformed", "raised"
    ]
    reservation: Mapping[str, object] | None
    reservation_acknowledgment: Mapping[str, object] | None
    terminal_call_state: Literal[
        "not_called", "returned_valid", "returned_malformed", "raised"
    ]
    terminal: Mapping[str, object] | None
    terminal_acknowledgment: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class PermissionAuthzAuthorityOperations:
    parent: PermissionOperationEvidenceProjection
    child: PermissionOperationEvidenceProjection


BaseRetentionResult: TypeAlias = Mapping[str, object]

_BINDING_KEYS = tuple(item.name for item in fields(PermissionAuthzEvidenceBinding))
_SOURCE_FILE_KEYS = tuple(item.name for item in fields(PermissionAuthzSourceFile))
_BASE_KEYS = tuple(item.name for item in fields(PermissionAuthzBaseObservation))
_HELD_KEYS = tuple(item.name for item in fields(PermissionAuthzHeldInputs))
_CHILD_KEYS = tuple(item.name for item in fields(PermissionAuthzChildAccounting))
_AUTHZ_KEYS = tuple(item.name for item in fields(PermissionAuthzWireEvidence))
_OPERATION_KEYS = tuple(item.name for item in fields(PermissionOperationEvidenceProjection))
_ROOT_KEYS = (
    "schema",
    "binding",
    "request",
    "access_expectation",
    "retention_destination_self_sha256",
    "store_expectation_self_sha256",
    "base_observation",
    "held_inputs",
    "authz",
    "authority_operations",
    "controller_failure_reason",
    "permission_acceptance_emitted",
    "record_self_sha256",
)
_SID_KEYS = ("sid_string", "sid_bytes_hex")
_IDENTITY_KEYS = (
    "volume_serial_number_hex",
    "file_id_128_hex",
    "legacy_volume_serial_number_hex",
    "legacy_file_index_hex",
)
_GROUP_KEYS = ("sid", "attributes_hex")
_PRIVILEGE_KEYS = ("luid_hex", "name", "attributes_hex")
_TOKEN_FACT_KEYS = tuple(item.name for item in fields(observation.NativeTokenFacts))
_ACE_KEYS = tuple(item.name for item in fields(observation.NativeAce))
_DESCRIPTOR_KEYS = tuple(item.name for item in fields(observation.NativeSecurityDescriptor))
_OBJECT_BINDING_KEYS = tuple(
    item.name for item in fields(observation.NativeAccessObjectBinding)
)
_TOKEN_IDENTITY_KEYS = tuple(
    item.name for item in fields(supplied_authz.SuppliedTokenIdentity)
)
_TOKEN_STATISTICS_KEYS = tuple(
    item.name for item in fields(supplied_authz.NativeTokenStatistics)
)
_AUTHZ_FAILURE_KEYS = tuple(
    item.name for item in fields(supplied_authz.PermissionPrincipalAuthzFailure)
)
_AUTHZ_RESULT_KEYS = tuple(
    item.name for item in fields(supplied_authz.SuppliedTokenAccessResult)
)
_FAILURE_ITEM_KEYS = tuple(item.name for item in fields(observation.ComponentFailureItem))
_CHECKPOINT_KEYS = tuple(item.name for item in fields(observation.PermissionAuthorityCheckpoint))
_CHECK_OUTCOME_KEYS = tuple(item.name for item in fields(observation.AuthorityCheckOutcome))
_FINISH_KEYS = tuple(item.name for item in fields(observation.AuthorityFinishOutcome))
_RESERVATION_KEYS = (
    "schema",
    "reservation_id",
    "allowance_id",
    "allowance_kind",
    "session_id",
    "state4_authorization_file_sha256",
    "request_self_sha256",
    "operation",
    "scope_self_sha256",
    "grant_binding_sha256",
)
_RESERVATION_ACK_KEYS = ("status", "effect_certainty", "reservation")
_TERMINAL_KEYS = (
    "schema",
    "terminal_id",
    "session_id",
    "reservation_id",
    "allowance_id",
    "state4_authorization_file_sha256",
    "request_self_sha256",
    "operation",
    "scope_self_sha256",
    "requested_outcome",
)
_TERMINAL_ACK_KEYS = (
    "status",
    "effect_certainty",
    "terminal",
    "recorded_outcome",
)
_CAPTURE_RECEIPT_KEYS = (
    "schema",
    "payload_kind",
    "request_self_sha256",
    "access_expectation_self_sha256",
    "destination_self_sha256",
    "store_expectation_self_sha256",
    "relative_path",
    "file_bytes",
    "file_sha256",
    "commitment_domain",
    "authorization_file_sha256",
    "private_observation_commitment",
    "namespace_object_identity",
    "records_parent_object_identity",
    "store_object_identity",
    "creation_security_descriptor_sha256",
    "confidential_store_profile_sha256",
    "held_object_readback_sha256",
    "publication_state",
    "receipt_self_sha256",
)
_RETENTION_FAILURE_RESULT_KEYS = (
    "schema",
    "payload_kind",
    "failed_primary_payload_kind",
    "request_self_sha256",
    "access_expectation_self_sha256",
    "destination_self_sha256",
    "store_expectation_self_sha256",
    "primary_failure",
    "ordered_follow_on_failures",
    "main_authority_accounting",
    "payload_effects",
    "payload_object_state",
    "failure_record_attempted",
    "failure_record_primary_failure",
    "failure_record_ordered_follow_on_failures",
    "failure_record_effects",
    "failure_record_object_state",
    "failure_record_authority_accounting",
    "retained_failure_record_relative_path",
    "retained_failure_record_file_bytes",
    "retained_failure_record_file_sha256",
    "retained_failure_record_commitment_domain",
    "retained_failure_record_authorization_file_sha256",
    "retained_failure_record_commitment",
    "acceptance_emitted",
    "result_self_sha256",
)

_AUTHZ_FAILURE_REASONS = frozenset(
    {
        "authz_hook_not_reached",
        "authz_hook_failed",
        "authz_hook_input_rejected",
        "child_authority_acquire_exception",
        "child_authority_acquire_malformed",
        "child_authority_refused",
        "child_authority_acquire_failed",
        "child_lease_binding_failed",
        "child_authority_check_exception",
        "child_authority_check_malformed",
        "child_authority_inactive",
        "authz_adapter_exception",
        "authz_result_malformed",
        "authz_result_unavailable",
        "authz_integration_exception",
        "child_terminal_incomplete",
        "authz_hook_result_malformed",
    }
)
_CONTROLLER_FAILURE_REASONS = frozenset(
    {"selected_result_catalog_activation_failed", "retention_integration_failed"}
)
_CALL_STATES = frozenset(
    {"not_called", "returned_valid", "returned_malformed", "raised"}
)
_HEX_RE = re.compile(r"^(?:[0-9a-f]{2})*$")
_DWORD_MAX = 0xFFFFFFFF
_FAILURE_OPERATIONS = {
    ("native_call_failed", "token_before"): frozenset({"GetTokenInformation"}),
    ("native_call_failed", "resource_manager"): frozenset(
        {"AuthzInitializeResourceManager"}
    ),
    ("native_call_failed", "context"): frozenset(
        {"AuthzInitializeContextFromToken"}
    ),
    ("native_call_failed", "access_check"): frozenset({"AuthzAccessCheck"}),
    ("native_call_failed", "token_after"): frozenset({"GetTokenInformation"}),
    ("invalid_native_output", "token_before"): frozenset({"GetTokenInformation"}),
    ("invalid_native_output", "resource_manager"): frozenset(
        {"AuthzInitializeResourceManager"}
    ),
    ("invalid_native_output", "context"): frozenset(
        {"AuthzInitializeContextFromToken"}
    ),
    ("invalid_native_output", "access_check"): frozenset({"AuthzAccessCheck"}),
    ("invalid_native_output", "token_after"): frozenset({"GetTokenInformation"}),
    ("token_identity_mismatch", "token_before"): frozenset(
        {"compare_token_identity"}
    ),
    ("token_changed", "token_after"): frozenset(
        {"compare_token_identity", "compare_token_statistics"}
    ),
    ("per_result_error", "access_check"): frozenset({"AuthzAccessCheck"}),
    ("inconsistent_result", "access_check"): frozenset({"AuthzAccessCheck"}),
    ("cleanup_failed", "cleanup"): frozenset(
        {"AuthzFreeContext", "AuthzFreeResourceManager"}
    ),
}


def _raise(message: str, cause: BaseException | None = None) -> None:
    error = PermissionAuthzRecordError(message)
    if cause is None:
        raise error
    raise error from cause


def _mapping(value: object, keys: tuple[str, ...], *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        _raise(f"{name} must be a JSON object")
    try:
        require_exact_keys(cast(dict[str, object], value), keys, name=name)
    except Exception as exc:
        _raise(f"{name} has the wrong exact keys", exc)
    return cast(dict[str, object], value)


def _array(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        _raise(f"{name} must be a JSON array")
    return cast(list[object], value)


def _text(value: object, *, name: str, nonempty: bool = True) -> str:
    if (
        type(value) is not str
        or (nonempty and not value)
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        _raise(f"{name} must be exact NFC text")
    return cast(str, value)


def _sha(value: object, *, name: str) -> str:
    if type(value) is not str:
        _raise(f"{name} must be exact SHA-256 text")
    try:
        return require_sha256(value, name=name)
    except Exception as exc:
        _raise(f"{name} is not lowercase SHA-256", exc)


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _raise(f"{name} must be an exact integer >= {minimum}")
    return cast(int, value)


def _optional_dword(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    result = _integer(value, name=name)
    if result > _DWORD_MAX:
        _raise(f"{name} exceeds DWORD")
    return result


def _wire(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _wire(getattr(value, item.name)) for item in fields(value)}
    if type(value) is tuple:
        return [_wire(item) for item in cast(tuple[object, ...], value)]
    if type(value) is list:
        return [_wire(item) for item in cast(list[object], value)]
    if type(value) is dict:
        return {
            cast(str, key): _wire(item)
            for key, item in cast(dict[object, object], value).items()
        }
    return value


def _scalar_tree(value: object, *, name: str = "record") -> None:
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            _text(key, name=f"{name} key")
            _scalar_tree(item, name=f"{name}.{key}")
        return
    if type(value) is list:
        for index, item in enumerate(cast(list[object], value)):
            _scalar_tree(item, name=f"{name}[{index}]")
        return
    if type(value) in {str, int, bool, type(None)}:
        if type(value) is str:
            _text(value, name=name, nonempty=False)
        return
    _raise(f"{name} has an unsupported scalar representation")


def _decode_sid(value: object, *, name: str) -> observation.NativeSid:
    record = _mapping(value, _SID_KEYS, name=name)
    sid = observation.NativeSid(
        sid_string=record["sid_string"],
        sid_bytes_hex=record["sid_bytes_hex"],
    )
    try:
        observation._validate_sid(sid, name=name)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)
    return sid


def _decode_identity(value: object, *, name: str) -> observation.NativeFileIdentity:
    record = _mapping(value, _IDENTITY_KEYS, name=name)
    identity = observation.NativeFileIdentity(**record)  # type: ignore[arg-type]
    try:
        observation._validate_identity(identity, name=name)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)
    return identity


def _decode_group(value: object, *, name: str) -> observation.NativeGroup:
    record = _mapping(value, _GROUP_KEYS, name=name)
    group = observation.NativeGroup(
        sid=_decode_sid(record["sid"], name=f"{name}.sid"),
        attributes_hex=record["attributes_hex"],
    )
    return group


def _decode_privilege(value: object, *, name: str) -> observation.NativePrivilege:
    record = _mapping(value, _PRIVILEGE_KEYS, name=name)
    return observation.NativePrivilege(**record)  # type: ignore[arg-type]


def _decode_token_facts(value: object, *, name: str) -> observation.NativeTokenFacts:
    record = _mapping(value, _TOKEN_FACT_KEYS, name=name)
    groups = tuple(
        _decode_group(item, name=f"{name}.ordered_groups[{index}]")
        for index, item in enumerate(_array(record["ordered_groups"], name=f"{name}.ordered_groups"))
    )
    restricted = tuple(
        _decode_group(item, name=f"{name}.ordered_restricted_sids[{index}]")
        for index, item in enumerate(
            _array(record["ordered_restricted_sids"], name=f"{name}.ordered_restricted_sids")
        )
    )
    privileges = tuple(
        _decode_privilege(item, name=f"{name}.ordered_privileges[{index}]")
        for index, item in enumerate(
            _array(record["ordered_privileges"], name=f"{name}.ordered_privileges")
        )
    )
    capabilities = tuple(
        _decode_group(item, name=f"{name}.ordered_capabilities[{index}]")
        for index, item in enumerate(
            _array(record["ordered_capabilities"], name=f"{name}.ordered_capabilities")
        )
    )
    facts = observation.NativeTokenFacts(
        token_type=record["token_type"],
        impersonation_level=record["impersonation_level"],
        token_id_luid_hex=record["token_id_luid_hex"],
        authentication_id_luid_hex=record["authentication_id_luid_hex"],
        modified_id_luid_hex=record["modified_id_luid_hex"],
        session_id=record["session_id"],
        user=_decode_sid(record["user"], name=f"{name}.user"),
        owner=_decode_sid(record["owner"], name=f"{name}.owner"),
        primary_group=_decode_sid(record["primary_group"], name=f"{name}.primary_group"),
        ordered_groups=groups,
        ordered_restricted_sids=restricted,
        ordered_privileges=privileges,
        integrity_level_sid=_decode_sid(
            record["integrity_level_sid"], name=f"{name}.integrity_level_sid"
        ),
        elevation_type=record["elevation_type"],
        is_elevated=record["is_elevated"],
        has_restrictions=record["has_restrictions"],
        is_app_container=record["is_app_container"],
        ordered_capabilities=capabilities,
        virtualization_allowed=record["virtualization_allowed"],
        virtualization_enabled=record["virtualization_enabled"],
        ui_access=record["ui_access"],
    )
    try:
        observation._validate_token_facts(facts, duplicate=True)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)
    return facts


def _decode_ace(value: object, *, name: str) -> observation.NativeAce:
    record = _mapping(value, _ACE_KEYS, name=name)
    return observation.NativeAce(
        ace_index=record["ace_index"],
        ace_kind=record["ace_kind"],
        ace_type_hex=record["ace_type_hex"],
        ace_flags_hex=record["ace_flags_hex"],
        access_mask_hex=record["access_mask_hex"],
        trustee_sid=_decode_sid(record["trustee_sid"], name=f"{name}.trustee_sid"),
        raw_ace_hex=record["raw_ace_hex"],
    )


def _decode_descriptor(
    value: object, *, name: str
) -> observation.NativeSecurityDescriptor:
    record = _mapping(value, _DESCRIPTOR_KEYS, name=name)
    descriptor = observation.NativeSecurityDescriptor(
        self_relative_descriptor_hex=record["self_relative_descriptor_hex"],
        self_relative_descriptor_sha256=record["self_relative_descriptor_sha256"],
        revision=record["revision"],
        control_hex=record["control_hex"],
        owner_sid=_decode_sid(record["owner_sid"], name=f"{name}.owner_sid"),
        group_sid=_decode_sid(record["group_sid"], name=f"{name}.group_sid"),
        dacl_present=record["dacl_present"],
        dacl_defaulted=record["dacl_defaulted"],
        dacl_revision=record["dacl_revision"],
        dacl_protected=record["dacl_protected"],
        dacl_auto_inherited=record["dacl_auto_inherited"],
        ordered_dacl_aces=tuple(
            _decode_ace(item, name=f"{name}.ordered_dacl_aces[{index}]")
            for index, item in enumerate(
                _array(record["ordered_dacl_aces"], name=f"{name}.ordered_dacl_aces")
            )
        ),
    )
    try:
        return supplied_authz._validate_descriptor(descriptor)[0]
    except Exception as exc:
        _raise(f"{name} is malformed", exc)


def _decode_object_binding(
    value: object,
    *,
    name: str,
    descriptor: observation.NativeSecurityDescriptor | None = None,
) -> observation.NativeAccessObjectBinding:
    record = _mapping(value, _OBJECT_BINDING_KEYS, name=name)
    binding = observation.NativeAccessObjectBinding(
        object_role=record["object_role"],
        object_kind=record["object_kind"],
        canonical_volume_guid_path=record["canonical_volume_guid_path"],
        identity=_decode_identity(record["identity"], name=f"{name}.identity"),
        descriptor_sha256=record["descriptor_sha256"],
    )
    if descriptor is not None:
        try:
            supplied_authz._validate_binding(binding, descriptor)
        except Exception as exc:
            _raise(f"{name} is malformed", exc)
    return binding


def _decode_token_identity(
    value: object, *, name: str
) -> supplied_authz.SuppliedTokenIdentity:
    record = _mapping(value, _TOKEN_IDENTITY_KEYS, name=name)
    identity = supplied_authz.SuppliedTokenIdentity(**record)  # type: ignore[arg-type]
    try:
        return supplied_authz._validate_identity(identity)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)


def _decode_statistics(
    value: object, *, name: str
) -> supplied_authz.NativeTokenStatistics | None:
    if value is None:
        return None
    record = _mapping(value, _TOKEN_STATISTICS_KEYS, name=name)
    statistics = supplied_authz.NativeTokenStatistics(**record)  # type: ignore[arg-type]
    try:
        return supplied_authz._validate_statistics(statistics)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)


def _decode_authz_failure(
    value: object, *, name: str, cleanup: bool
) -> supplied_authz.PermissionPrincipalAuthzFailure | None:
    if value is None:
        return None
    record = _mapping(value, _AUTHZ_FAILURE_KEYS, name=name)
    failure = supplied_authz.PermissionPrincipalAuthzFailure(**record)  # type: ignore[arg-type]
    if any(
        type(getattr(failure, field_name)) is not str
        for field_name in ("reason", "stage", "operation", "message")
    ):
        _raise(f"{name} contains inexact text")
    operations = _FAILURE_OPERATIONS.get((failure.reason, failure.stage))
    if (
        operations is None
        or failure.operation not in operations
        or failure.message != supplied_authz._FAILURE_MESSAGES.get(failure.reason)
        or (cleanup and failure.reason != "cleanup_failed")
        or (not cleanup and failure.reason == "cleanup_failed")
    ):
        _raise(f"{name} has contradictory failure fields")
    if failure.native_error_code is not None and (
        type(failure.native_error_code) is not int
        or not 0 < failure.native_error_code <= _DWORD_MAX
    ):
        _raise(f"{name}.native_error_code is malformed")
    return failure


def _statistics_match(
    value: supplied_authz.NativeTokenStatistics | None,
    expected: supplied_authz.SuppliedTokenIdentity,
) -> bool:
    return value is not None and supplied_authz._statistics_match_expected(
        value, expected
    )


def _validate_unavailable_result(
    result: supplied_authz.SuppliedTokenAccessResult,
    *,
    expected_identity: supplied_authz.SuppliedTokenIdentity,
) -> None:
    before = result.token_statistics_before
    after = result.token_statistics_after
    primary = result.primary_failure
    cleanup = result.cleanup_failures
    if primary is None:
        if (
            result.access_check_api_status != "succeeded"
            or not cleanup
            or not _statistics_match(before, expected_identity)
            or not _statistics_match(after, expected_identity)
            or after != before
        ):
            _raise("Authz unavailable cleanup-only result is contradictory")
        return
    if result.access_check_api_status == "failed":
        if (
            primary.reason != "native_call_failed"
            or primary.stage != "access_check"
            or not _statistics_match(before, expected_identity)
            or after is not None
        ):
            _raise("Authz unavailable failed-call result is contradictory")
        return
    if result.access_check_api_status == "succeeded":
        if primary.stage == "access_check":
            if not _statistics_match(before, expected_identity) or after is not None:
                _raise("Authz unavailable access result is contradictory")
            return
        if primary.stage == "token_after":
            if not _statistics_match(before, expected_identity):
                _raise("Authz unavailable post-token result is contradictory")
            if primary.reason == "token_changed":
                if after is None or (
                    _statistics_match(after, expected_identity) and after == before
                ):
                    _raise("Authz token-changed result did not change")
            elif after is not None:
                _raise("Authz unavailable post-token result is contradictory")
            return
        _raise("Authz unavailable succeeded result names an invalid stage")
    if result.access_check_api_status != "not_called" or after is not None:
        _raise("Authz unavailable pre-call result is contradictory")
    if primary.stage == "token_before":
        if primary.reason == "token_identity_mismatch":
            if before is None or _statistics_match(before, expected_identity):
                _raise("Authz token identity mismatch is contradictory")
        elif before is not None:
            _raise("Authz pre-token failure unexpectedly has statistics")
        return
    if primary.stage in {"resource_manager", "context"}:
        if not _statistics_match(before, expected_identity):
            _raise("Authz pre-access failure lacks bound token statistics")
        return
    _raise("Authz unavailable result has an unsupported chronology")


def _decode_authz_result(
    value: object,
    *,
    expected_identity: supplied_authz.SuppliedTokenIdentity,
    expected_descriptor: observation.NativeSecurityDescriptor,
    expected_binding: observation.NativeAccessObjectBinding,
    name: str = "authz result",
) -> supplied_authz.SuppliedTokenAccessResult:
    record = _mapping(value, _AUTHZ_RESULT_KEYS, name=name)
    cleanup = tuple(
        cast(
            supplied_authz.PermissionPrincipalAuthzFailure,
            _decode_authz_failure(
                item, name=f"{name}.cleanup_failures[{index}]", cleanup=True
            ),
        )
        for index, item in enumerate(
            _array(record["cleanup_failures"], name=f"{name}.cleanup_failures")
        )
    )
    if any(item is None for item in cleanup):
        _raise(f"{name} cleanup failures cannot contain null")
    cleanup_order = {"AuthzFreeContext": 0, "AuthzFreeResourceManager": 1}
    ordinals = tuple(cleanup_order[item.operation] for item in cleanup)
    if ordinals != tuple(sorted(set(ordinals))):
        _raise(f"{name} cleanup failures repeat or reorder release operations")
    identity = _decode_token_identity(
        record["expected_token_identity"], name=f"{name}.expected_token_identity"
    )
    binding = _decode_object_binding(
        record["object_binding"],
        name=f"{name}.object_binding",
        descriptor=expected_descriptor,
    )
    result = supplied_authz.SuppliedTokenAccessResult(
        access_check_api_status=record["access_check_api_status"],
        access_status=record["access_status"],
        granted_access_mask=record["granted_access_mask"],
        per_result_error_code=record["per_result_error_code"],
        sacl_evaluation_results=record["sacl_evaluation_results"],
        expected_token_identity=identity,
        token_statistics_before=_decode_statistics(
            record["token_statistics_before"], name=f"{name}.token_statistics_before"
        ),
        token_statistics_after=_decode_statistics(
            record["token_statistics_after"], name=f"{name}.token_statistics_after"
        ),
        object_binding=binding,
        descriptor_sha256=record["descriptor_sha256"],
        primary_failure=_decode_authz_failure(
            record["primary_failure"], name=f"{name}.primary_failure", cleanup=False
        ),
        cleanup_failures=cleanup,
    )
    if (
        record["requested_access_mask"] != supplied_authz.MAXIMUM_ALLOWED
        or type(record["requested_access_mask"]) is not int
        or record["permission_acceptance_emitted"] is not False
        or identity != expected_identity
        or binding != expected_binding
        or result.descriptor_sha256
        != expected_descriptor.self_relative_descriptor_sha256
        or type(result.access_check_api_status) is not str
        or result.access_check_api_status not in {"not_called", "succeeded", "failed"}
        or type(result.access_status) is not str
        or result.access_status not in {"granted", "denied", "unavailable"}
    ):
        _raise(f"{name} binding or fixed fields mismatch")
    mask = _optional_dword(result.granted_access_mask, name=f"{name}.granted_access_mask")
    per_error = _optional_dword(
        result.per_result_error_code, name=f"{name}.per_result_error_code"
    )
    sacl = _optional_dword(
        result.sacl_evaluation_results, name=f"{name}.sacl_evaluation_results"
    )
    if result.access_status in {"granted", "denied"}:
        if (
            result.access_check_api_status != "succeeded"
            or result.primary_failure is not None
            or result.cleanup_failures
            or not _statistics_match(result.token_statistics_before, expected_identity)
            or not _statistics_match(result.token_statistics_after, expected_identity)
            or result.token_statistics_after != result.token_statistics_before
            or mask is None
            or sacl != 0
        ):
            _raise(f"{name} completed-result equation mismatch")
        if result.access_status == "granted":
            if per_error != supplied_authz.ERROR_SUCCESS:
                _raise(f"{name} grant/error equation mismatch")
        elif mask != 0 or per_error != supplied_authz.ERROR_ACCESS_DENIED:
            _raise(f"{name} denial/error equation mismatch")
    else:
        if mask is not None or per_error is not None or sacl is not None or (
            result.primary_failure is None and not result.cleanup_failures
        ):
            _raise(f"{name} unavailable-result equation mismatch")
        _validate_unavailable_result(result, expected_identity=expected_identity)
    return result


def _decode_failure_item(
    value: object, *, name: str
) -> observation.ComponentFailureItem | None:
    if value is None:
        return None
    record = _mapping(value, _FAILURE_ITEM_KEYS, name=name)
    item = observation.ComponentFailureItem(**record)  # type: ignore[arg-type]
    try:
        return observation._validate_failure_item(item)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)


def _decode_checkpoint(
    value: object, *, name: str
) -> observation.PermissionAuthorityCheckpoint:
    record = _mapping(value, _CHECKPOINT_KEYS, name=name)
    checkpoint = observation.PermissionAuthorityCheckpoint(**record)  # type: ignore[arg-type]
    if (
        type(checkpoint.operation) is not str
        or checkpoint.operation != "observe_authz"
        or type(checkpoint.step) is not str
        or checkpoint.step not in {"initial", "authz_access_check", "final"}
        or (
            checkpoint.step == "authz_access_check"
            and (type(checkpoint.ordinal) is not int or checkpoint.ordinal not in {0, 1})
        )
        or (checkpoint.step != "authz_access_check" and checkpoint.ordinal is not None)
    ):
        _raise(f"{name} is malformed")
    return checkpoint


def _decode_checkpoint_failure(
    value: object, *, name: str
) -> observation.ComponentFailureItem | None:
    """Decode the Authz checkpoint vocabulary, not observer acquisition failures."""
    if value is None:
        return None
    record = _mapping(value, _FAILURE_ITEM_KEYS, name=name)
    failure = observation.ComponentFailureItem(**record)  # type: ignore[arg-type]
    if (
        any(type(getattr(failure, key)) is not str for key in (
            "failure_kind", "stage", "component_operation", "reason_code"
        ))
        or any(getattr(failure, key) is not None for key in (
            "resource_kind", "resource_role", "resource_ordinal", "backend_error_hex"
        ))
        or (failure.operation_ordinal is not None and (
            type(failure.operation_ordinal) is not int
            or failure.operation_ordinal not in (0, 1)
        ))
    ):
        _raise(f"{name} is malformed")
    return failure


def _decode_check_outcome(
    value: object, *, name: str
) -> observation.AuthorityCheckOutcome:
    record = _mapping(value, _CHECK_OUTCOME_KEYS, name=name)
    checkpoint = _decode_checkpoint(record["checkpoint"], name=f"{name}.checkpoint")
    failure = _decode_checkpoint_failure(record["failure"], name=f"{name}.failure")
    outcome = observation.AuthorityCheckOutcome(
        status=record["status"], checkpoint=checkpoint, failure=failure
    )
    if type(outcome.status) is not str:
        _raise(f"{name} status must be an exact string")
    if outcome.status == "active":
        if failure is not None:
            _raise(f"{name} active checkpoint carries failure")
    elif outcome.status in {"refused", "failed"}:
        if failure is None:
            _raise(f"{name} inactive checkpoint lacks failure")
        expected_kind = (
            "authority_refusal" if outcome.status == "refused" else "authority_failure"
        )
        expected_reason = (
            "authority_inactive" if outcome.status == "refused" else "authority_check_failed"
        )
        expected_stage = (
            "final_authority_check"
            if checkpoint.step == "final"
            else "authz_observation"
        )
        if (
            failure.failure_kind != expected_kind
            or failure.stage != expected_stage
            or failure.component_operation != "require_active"
            or failure.resource_kind is not None
            or failure.resource_role is not None
            or failure.resource_ordinal is not None
            or type(failure.operation_ordinal) is not type(checkpoint.ordinal)
            or failure.operation_ordinal != checkpoint.ordinal
            or failure.backend_error_hex is not None
            or failure.reason_code != expected_reason
        ):
            _raise(f"{name} inactive checkpoint failure is contradictory")
    else:
        _raise(f"{name} status is invalid")
    return outcome


def _decode_finish(
    value: object, *, name: str
) -> observation.AuthorityFinishOutcome | None:
    if value is None:
        return None
    record = _mapping(value, _FINISH_KEYS, name=name)
    outcome = observation.AuthorityFinishOutcome(**record)  # type: ignore[arg-type]
    if any(type(getattr(outcome, key)) is not str for key in (
        "finish_status", "effect_certainty", "reason_code"
    )) or (outcome.recorded_outcome is not None and type(outcome.recorded_outcome) is not str):
        _raise(f"{name} scalar fields are malformed")
    if outcome.finish_status == "succeeded":
        if (
            outcome.effect_certainty != "confirmed"
            or outcome.recorded_outcome not in {"succeeded", "refused", "failed"}
            or outcome.reason_code != "none"
        ):
            _raise(f"{name} success is contradictory")
    elif outcome.finish_status == "failed":
        if (
            outcome.effect_certainty not in {"confirmed", "not_applied", "unknown"}
            or (outcome.effect_certainty == "confirmed" and outcome.recorded_outcome not in {"succeeded", "refused", "failed"})
            or (outcome.effect_certainty != "confirmed" and outcome.recorded_outcome is not None)
            or outcome.reason_code != "terminal_finish_failed"
        ):
            _raise(f"{name} failure is contradictory")
    else:
        _raise(f"{name} status is invalid")
    return outcome


def _decode_child_accounting(value: object) -> PermissionAuthzChildAccounting:
    record = _mapping(value, _CHILD_KEYS, name="authz.child_accounting")
    checkpoints = tuple(
        _decode_check_outcome(
            item, name=f"authz.child_accounting.checkpoint_outcomes[{index}]"
        )
        for index, item in enumerate(
            _array(
                record["checkpoint_outcomes"],
                name="authz.child_accounting.checkpoint_outcomes",
            )
        )
    )
    accounting = PermissionAuthzChildAccounting(
        acquire_status=record["acquire_status"],
        lease_candidate_returned=record["lease_candidate_returned"],
        acquire_failure=_decode_failure_item(
            record["acquire_failure"], name="authz.child_accounting.acquire_failure"
        ),
        lease_binding_status=record["lease_binding_status"],
        checkpoint_outcomes=checkpoints,
        finish_attempted=record["finish_attempted"],
        requested_outcome=record["requested_outcome"],
        finish_outcome=_decode_finish(
            record["finish_outcome"], name="authz.child_accounting.finish_outcome"
        ),
    )
    if (
        type(accounting.acquire_status) is not str
        or accounting.acquire_status not in {"not_attempted", "granted", "refused", "failed"}
        or type(accounting.lease_candidate_returned) is not bool
        or type(accounting.lease_binding_status) is not str
        or accounting.lease_binding_status not in {"not_applicable", "valid", "invalid"}
        or type(accounting.finish_attempted) is not bool
        or (accounting.requested_outcome is not None and type(accounting.requested_outcome) is not str)
        or accounting.requested_outcome
        not in {None, "succeeded", "refused", "failed"}
    ):
        _raise("authz child accounting scalar fields are malformed")
    if accounting.finish_attempted != (accounting.finish_outcome is not None):
        _raise("authz child accounting finish nullability mismatch")
    if accounting.finish_attempted != (accounting.requested_outcome is not None):
        _raise("authz child accounting requested outcome nullability mismatch")
    if accounting.finish_outcome is not None and (
        accounting.finish_outcome.effect_certainty == "confirmed"
        and accounting.finish_outcome.recorded_outcome != accounting.requested_outcome
    ):
        _raise("Authz finish acknowledgment differs from its requested outcome")
    if not accounting.lease_candidate_returned and (
        accounting.lease_binding_status != "not_applicable" or checkpoints
        or accounting.finish_attempted
    ):
        _raise("Authz child without a lease candidate has lease activity")
    if accounting.lease_candidate_returned and accounting.lease_binding_status == "not_applicable":
        _raise("Authz lease candidate lacks binding accounting")
    if accounting.acquire_status == "not_attempted" and (
        accounting.lease_candidate_returned
        or accounting.acquire_failure is not None
        or accounting.lease_binding_status != "not_applicable"
        or checkpoints
        or accounting.finish_attempted
    ):
        _raise("unattempted Authz child has accounting effects")
    if accounting.acquire_status in {"refused", "failed"} and not accounting.lease_candidate_returned:
        if accounting.lease_binding_status != "not_applicable" or checkpoints or accounting.finish_attempted:
            _raise("unacquired Authz child has lease activity")
    return accounting


def _decode_held_inputs(value: object) -> PermissionAuthzHeldInputs | None:
    if value is None:
        return None
    record = _mapping(value, _HELD_KEYS, name="held_inputs")
    descriptors_raw = _array(record["descriptors"], name="held_inputs.descriptors")
    bindings_raw = _array(record["object_bindings"], name="held_inputs.object_bindings")
    if len(descriptors_raw) != 2 or len(bindings_raw) != 2:
        _raise("held input descriptor/object vectors must each have length two")
    descriptors = cast(
        tuple[
            observation.NativeSecurityDescriptor,
            observation.NativeSecurityDescriptor,
        ],
        tuple(
            _decode_descriptor(item, name=f"held_inputs.descriptors[{index}]")
            for index, item in enumerate(descriptors_raw)
        ),
    )
    bindings = cast(
        tuple[
            observation.NativeAccessObjectBinding,
            observation.NativeAccessObjectBinding,
        ],
        tuple(
            _decode_object_binding(
                item,
                name=f"held_inputs.object_bindings[{index}]",
                descriptor=descriptors[index],
            )
            for index, item in enumerate(bindings_raw)
        ),
    )
    if tuple(item.object_role for item in bindings) != ("subject", "immediate_parent"):
        _raise("held input object bindings are out of order")
    return PermissionAuthzHeldInputs(
        duplicate_token_before_facts=_decode_token_facts(
            record["duplicate_token_before_facts"],
            name="held_inputs.duplicate_token_before_facts",
        ),
        descriptors=descriptors,
        object_bindings=bindings,
    )


def _decode_base_observation(value: object) -> PermissionAuthzBaseObservation:
    record = _mapping(value, _BASE_KEYS, name="base_observation")
    outcome = record["outcome"]
    if type(outcome) is not str or outcome not in {"captured", "failed", "unavailable"}:
        _raise("base observation outcome is invalid")
    byte_fields = (
        record["record_hex"],
        record["record_file_bytes"],
        record["record_file_sha256"],
    )
    all_null = all(item is None for item in byte_fields)
    all_present = all(item is not None for item in byte_fields)
    if not (all_null or all_present):
        _raise("base observation byte fields have partial nullability")
    if all_present:
        encoded = _text(record["record_hex"], name="base_observation.record_hex", nonempty=False)
        if _HEX_RE.fullmatch(encoded) is None:
            _raise("base observation record hex is malformed")
        data = bytes.fromhex(encoded)
        if (
            _integer(
                record["record_file_bytes"], name="base_observation.record_file_bytes"
            )
            != len(data)
            or _sha(
                record["record_file_sha256"], name="base_observation.record_file_sha256"
            )
            != sha256_bytes(data)
            or record["record_unavailable_reason"] is not None
            or outcome == "unavailable"
        ):
            _raise("base observation bytes/hash/outcome mismatch")
        try:
            embedded = strict_canonical_json_loads(data)
        except Exception as exc:
            _raise("embedded base observation is not canonical JSON", exc)
        expected_schema = (
            "operatormark.darcy_branch_c_v1.private_file_permission_observation.v2"
            if outcome == "captured"
            else "operatormark.darcy_branch_c_v1.private_file_permission_observation_failure.v2"
        )
        if embedded.get("schema") != expected_schema:
            _raise("embedded base observation schema/outcome mismatch")
    else:
        if (
            outcome != "unavailable"
            or record["record_unavailable_reason"]
            not in {"not_returned", "malformed_result", "serialization_failed"}
        ):
            _raise("unavailable base observation reason mismatch")
    kind = record["retention_result_kind"]
    if kind not in {
        "captured_receipt",
        "failure_receipt",
        "retention_failure",
        "not_returned",
    }:
        _raise("base retention result kind is invalid")
    if kind == "not_returned":
        if (
            record["retention_result"] is not None
            or record["retention_result_unavailable_reason"]
            not in {"not_returned", "malformed_result"}
        ):
            _raise("unavailable base retention result is contradictory")
    else:
        if record["retention_result_unavailable_reason"] is not None:
            _raise("available base retention result carries an unavailable reason")
        result = _mapping(
            record["retention_result"],
            _RETENTION_FAILURE_RESULT_KEYS
            if kind == "retention_failure"
            else _CAPTURE_RECEIPT_KEYS,
            name="base_observation.retention_result",
        )
        expected_kind = {
            "captured_receipt": "observation",
            "failure_receipt": "observation_failure",
            "retention_failure": "retention_failure",
        }[kind]
        if result["payload_kind"] != expected_kind:
            _raise("base retention result tag/envelope mismatch")
    return PermissionAuthzBaseObservation(
        outcome=cast(object, outcome),
        record_hex=cast(str | None, record["record_hex"]),
        record_file_bytes=cast(int | None, record["record_file_bytes"]),
        record_file_sha256=cast(str | None, record["record_file_sha256"]),
        record_unavailable_reason=cast(object, record["record_unavailable_reason"]),
        retention_result_kind=cast(object, kind),
        retention_result=cast(Mapping[str, object] | None, record["retention_result"]),
        retention_result_unavailable_reason=cast(
            object, record["retention_result_unavailable_reason"]
        ),
    )


def _decode_operation_projection(
    value: object,
    *,
    expected_operation: Literal["observe", "observe_authz"],
) -> PermissionOperationEvidenceProjection:
    name = f"authority_operations.{expected_operation}"
    record = _mapping(value, _OPERATION_KEYS, name=name)
    if record["operation"] != expected_operation:
        _raise(f"{name} operation mismatch")
    _sha(record["scope_self_sha256"], name=f"{name}.scope_self_sha256")
    for field_name in ("grant_binding_sha256", "parent_reservation_id"):
        if record[field_name] is not None:
            _sha(record[field_name], name=f"{name}.{field_name}")
    if expected_operation == "observe" and record["parent_reservation_id"] is not None:
        _raise("parent operation projection names a parent reservation")
    for prefix, record_keys, ack_keys in (
        ("reservation", _RESERVATION_KEYS, _RESERVATION_ACK_KEYS),
        ("terminal", _TERMINAL_KEYS, _TERMINAL_ACK_KEYS),
    ):
        state = record[f"{prefix}_call_state"]
        if type(state) is not str or state not in _CALL_STATES:
            _raise(f"{name}.{prefix}_call_state is invalid")
        intended = record[prefix]
        acknowledgment = record[f"{prefix}_acknowledgment"]
        if state == "not_called":
            if intended is not None or acknowledgment is not None:
                _raise(f"{name} uncalled {prefix} carries evidence")
        else:
            _mapping(intended, record_keys, name=f"{name}.{prefix}")
            if state == "returned_valid":
                _mapping(
                    acknowledgment,
                    ack_keys,
                    name=f"{name}.{prefix}_acknowledgment",
                )
            elif acknowledgment is not None:
                _raise(f"{name} unvalidated {prefix} carries an acknowledgment")
    if record["terminal_call_state"] != "not_called" and record["reservation"] is None:
        _raise(f"{name} terminal exists without an intended reservation")
    return PermissionOperationEvidenceProjection(**record)  # type: ignore[arg-type]


def _decode_authz(
    value: object,
    *,
    held: PermissionAuthzHeldInputs | None,
) -> PermissionAuthzWireEvidence:
    record = _mapping(value, _AUTHZ_KEYS, name="authz")
    identity = (
        None
        if record["expected_token_identity"] is None
        else _decode_token_identity(
            record["expected_token_identity"], name="authz.expected_token_identity"
        )
    )
    raw_results = _array(record["result_prefix"], name="authz.result_prefix")
    if len(raw_results) > 2:
        _raise("Authz result prefix is too long")
    if raw_results and (held is None or identity is None):
        _raise("Authz results lack held input or token identity evidence")
    results: list[supplied_authz.SuppliedTokenAccessResult] = []
    for index, item in enumerate(raw_results):
        assert held is not None and identity is not None
        results.append(
            _decode_authz_result(
                item,
                expected_identity=identity,
                expected_descriptor=held.descriptors[index],
                expected_binding=held.object_bindings[index],
                name=f"authz.result_prefix[{index}]",
            )
        )
    reasons_raw = _array(
        record["failure_reason_codes"], name="authz.failure_reason_codes"
    )
    reasons = tuple(
        _text(item, name=f"authz.failure_reason_codes[{index}]")
        for index, item in enumerate(reasons_raw)
    )
    if any(item not in _AUTHZ_FAILURE_REASONS for item in reasons):
        _raise("Authz failure reason is outside the closed vocabulary")
    authz = PermissionAuthzWireEvidence(
        status=record["status"],
        result_prefix=tuple(results),
        expected_token_identity=identity,
        failure_reason_codes=reasons,
        child_accounting=_decode_child_accounting(record["child_accounting"]),
    )
    _validate_authz_trace(authz, held_inputs=held)
    return authz


def _validate_authz_trace(
    authz: PermissionAuthzWireEvidence,
    *,
    held_inputs: PermissionAuthzHeldInputs | None,
) -> None:
    if type(authz.status) is not str or authz.status not in {"not_invoked", "incomplete", "completed"}:
        _raise("Authz status is invalid")
    accounting = authz.child_accounting
    checkpoints = accounting.checkpoint_outcomes
    expected_steps = (
        observation.PermissionAuthorityCheckpoint("observe_authz", "initial", None),
        observation.PermissionAuthorityCheckpoint("observe_authz", "authz_access_check", 0),
        observation.PermissionAuthorityCheckpoint("observe_authz", "authz_access_check", 1),
        observation.PermissionAuthorityCheckpoint("observe_authz", "final", None),
    )
    if len(checkpoints) > len(expected_steps):
        _raise("Authz checkpoint prefix is too long")
    for index, outcome in enumerate(checkpoints):
        if outcome.checkpoint != expected_steps[index]:
            _raise("Authz checkpoint prefix is skipped or reordered")
        if outcome.status != "active" and index != len(checkpoints) - 1:
            _raise("Authz checkpoint prefix continues after inactivity")
    active_access_checks = sum(
        outcome.status == "active" and outcome.checkpoint.step == "authz_access_check"
        for outcome in checkpoints
    )
    if len(authz.result_prefix) > active_access_checks:
        _raise("Authz result prefix exceeds active checkpoint prefix")
    if any(
        result.access_status == "unavailable"
        for result in authz.result_prefix[:-1]
    ):
        _raise("Authz result prefix continues after unavailable evidence")
    if authz.expected_token_identity is not None and held_inputs is not None:
        facts = held_inputs.duplicate_token_before_facts
        derived = supplied_authz.SuppliedTokenIdentity(
            token_id_luid_hex=facts.token_id_luid_hex,
            authentication_id_luid_hex=facts.authentication_id_luid_hex,
            modified_id_luid_hex=facts.modified_id_luid_hex,
            token_type=facts.token_type,
            impersonation_level=cast(object, facts.impersonation_level),
        )
        try:
            derived = supplied_authz._validate_identity(derived)
        except Exception as exc:
            _raise("held token identity projection is invalid", exc)
        if derived != authz.expected_token_identity:
            _raise("held token and Authz identity projections differ")
    if authz.status == "not_invoked":
        if (
            held_inputs is not None
            or authz.result_prefix
            or authz.expected_token_identity is not None
            or accounting != PermissionAuthzChildAccounting(
                "not_attempted", False, None, "not_applicable", (), False, None, None
            )
            or authz.failure_reason_codes != ("authz_hook_not_reached",)
        ):
            _raise("not-invoked Authz trace is contradictory")
        return
    completed = (
        held_inputs is not None
        and len(authz.result_prefix) == 2
        and authz.expected_token_identity is not None
        and not authz.failure_reason_codes
        and accounting.acquire_status == "granted"
        and accounting.lease_candidate_returned is True
        and accounting.acquire_failure is None
        and accounting.lease_binding_status == "valid"
        and len(checkpoints) == 4
        and all(item.status == "active" for item in checkpoints)
        and accounting.finish_attempted is True
        and accounting.requested_outcome == "succeeded"
        and accounting.finish_outcome
        == observation.AuthorityFinishOutcome(
            "succeeded", "confirmed", "succeeded", "none"
        )
    )
    if authz.status == "completed" and not completed:
        _raise("Authz completion label is not supported by its trace")
    if authz.status == "incomplete" and completed:
        _raise("complete Authz trace is mislabeled incomplete")
    if authz.status == "completed" and any(
        result.access_status == "unavailable" for result in authz.result_prefix
    ):
        _raise("completed Authz trace contains unavailable result")
    _validate_partial_authz_trace(authz)


def _validate_partial_authz_trace(authz: PermissionAuthzWireEvidence) -> None:
    """Check the exact stopping boundary, including meaningful negative outcomes."""
    child = authz.child_accounting
    checks = child.checkpoint_outcomes
    results = authz.result_prefix
    reasons = authz.failure_reason_codes
    if len(set(reasons)) != len(reasons):
        _raise("Authz trace repeats a failure reason")
    terminal_failed = child.finish_outcome is not None and child.finish_outcome.finish_status == "failed"
    if terminal_failed != (bool(reasons) and reasons[-1] == "child_terminal_incomplete"):
        _raise("Authz terminal failure and reason disagree")
    cause = reasons[:-1] if terminal_failed else reasons
    if len(cause) > 1:
        _raise("Authz trace has more than one primary stopping reason")
    reason = cause[0] if cause else None
    if reason is None:
        if not (
            len(checks) == 4 and all(item.status == "active" for item in checks)
            and len(results) == 2 and all(item.access_status != "unavailable" for item in results)
            and child.acquire_status == "granted" and child.acquire_failure is None
            and child.lease_candidate_returned and child.lease_binding_status == "valid"
            and child.finish_attempted and child.requested_outcome == "succeeded"
        ):
            _raise("Authz trace without a primary failure is not a completed computation")
        return
    if authz.status != "incomplete":
        _raise("Authz failure trace must be incomplete")
    early = {
        "authz_hook_failed", "authz_hook_input_rejected", "authz_hook_result_malformed",
        "child_authority_acquire_exception", "child_authority_acquire_malformed",
        "child_authority_refused", "child_authority_acquire_failed", "child_lease_binding_failed",
    }
    if reason in early:
        if checks or results:
            _raise("Authz acquisition/input failure carries checkpoint or result activity")
        if reason in {"child_authority_refused", "child_authority_acquire_failed"}:
            refused = reason == "child_authority_refused"
            failure = child.acquire_failure
            if not (
                child.acquire_status == ("refused" if refused else "failed")
                and not child.lease_candidate_returned and failure is not None
                and failure.failure_kind == ("authority_refusal" if refused else "authority_failure")
                and failure.stage == "authority_acquisition"
                and failure.component_operation == "acquire_authority_lease"
                and failure.operation_ordinal is None
                and failure.reason_code == ("authority_acquire_refused" if refused else "authority_acquire_failed")
            ):
                _raise("Authz acquisition failure does not correspond to its reason")
        if child.finish_attempted and child.requested_outcome != "failed":
            _raise("Authz acquisition/binding failure requested a non-failed finish")
        if reason == "child_lease_binding_failed" and not (
            child.acquire_status == "granted" and child.lease_candidate_returned
            and child.lease_binding_status == "invalid" and child.acquire_failure is None
        ):
            _raise("Authz lease binding failure has no invalid granted candidate")
        if reason in {"authz_hook_failed", "authz_hook_result_malformed", "child_authority_acquire_exception"} and (
            child.lease_candidate_returned or child.acquire_failure is not None
            or child.acquire_status not in {"not_attempted", "failed"}
        ):
            _raise("Authz fallback invents acquired lease evidence")
        if reason == "authz_hook_input_rejected" and (
            child.acquire_status != "not_attempted" or authz.expected_token_identity is not None
        ):
            _raise("Authz rejected hook inputs carry acquisition or identity evidence")
        return
    if not (
        child.acquire_status == "granted" and child.acquire_failure is None
        and child.lease_candidate_returned and child.lease_binding_status == "valid"
        and child.finish_attempted
    ):
        _raise("Authz checkpoint/result activity lacks a valid acquired and finished child")
    if reason == "child_authority_inactive":
        if not checks or checks[-1].status not in {"refused", "failed"}:
            _raise("Authz inactive reason lacks an inactive checkpoint")
        if len(results) != (0, 0, 1, 2)[len(checks) - 1] or child.requested_outcome != checks[-1].status:
            _raise("Authz inactive checkpoint has the wrong result prefix or finish")
    elif reason in {"child_authority_check_exception", "child_authority_check_malformed"}:
        if len(checks) > 3 or len(results) != (0, 0, 1, 2)[len(checks)]:
            _raise("Authz failed checkpoint has the wrong prior result prefix")
    elif reason in {"authz_adapter_exception", "authz_result_malformed"}:
        if len(checks) not in (2, 3) or len(results) != len(checks) - 2:
            _raise("Authz adapter failure has the wrong checkpoint/result boundary")
    elif reason == "authz_result_unavailable":
        if len(results) not in (1, 2) or len(checks) != len(results) + 1 or results[-1].access_status != "unavailable":
            _raise("Authz unavailable result has the wrong stopping boundary")
    elif reason == "authz_integration_exception":
        # An unexpected integration failure is negative evidence only. Keep its
        # bounded prefix, without converting it to a successful computation.
        if len(results) not in {max(0, len(checks) - 2), min(2, max(0, len(checks) - 1))}:
            _raise("Authz integration failure has an impossible partial prefix")
    else:
        _raise("Authz stopping reason does not describe a supported trace")
    if reason != "child_authority_inactive" and (
        any(item.status != "active" for item in checks) or child.requested_outcome != "failed"
    ):
        _raise("Authz failure has contradictory checkpoint status or requested finish")
    if reason != "authz_result_unavailable" and any(item.access_status == "unavailable" for item in results):
        _raise("Authz unavailable result lacks its stopping reason")


def build_permission_authz_evidence_binding(
    request: observation.PermissionObservationRequest,
    access_expectation: observation.PermissionAccessExpectation,
    *,
    expected_operator_sid: observation.NativeSid,
    source_files: tuple[PermissionAuthzSourceFile, ...],
) -> PermissionAuthzEvidenceBinding:
    try:
        observation.validate_permission_observation_request(
            request, expected_request=request
        )
        observation.validate_permission_access_expectation(
            access_expectation, expected_request=request
        )
        observation._validate_sid(expected_operator_sid, name="expected_operator_sid")
    except Exception as exc:
        _raise("Authz binding inputs are invalid", exc)
    if type(source_files) is not tuple:
        _raise("Authz source files must have the exact tuple type")
    unsigned = {
        "schema": AUTHZ_EVIDENCE_BINDING_SCHEMA,
        "request_self_sha256": request.request_self_sha256,
        "access_expectation_self_sha256": access_expectation.expectation_self_sha256,
        "protocol_file_sha256": request.context.protocol_file_sha256,
        "state4_authorization_file_sha256": (
            request.context.state4_authorization_file_sha256
        ),
        "implementation_commit": request.context.implementation_commit,
        "implementation_tree": request.context.implementation_tree,
        "expected_operator_sid": _wire(expected_operator_sid),
        "source_files": _wire(source_files),
    }
    try:
        signed = with_self_hash(unsigned, "binding_self_sha256")
    except Exception as exc:
        _raise("Authz binding could not be canonicalized", exc)
    binding = PermissionAuthzEvidenceBinding(
        schema=cast(str, signed["schema"]),
        request_self_sha256=cast(str, signed["request_self_sha256"]),
        access_expectation_self_sha256=cast(
            str, signed["access_expectation_self_sha256"]
        ),
        protocol_file_sha256=cast(str, signed["protocol_file_sha256"]),
        state4_authorization_file_sha256=cast(
            str, signed["state4_authorization_file_sha256"]
        ),
        implementation_commit=cast(str, signed["implementation_commit"]),
        implementation_tree=cast(str, signed["implementation_tree"]),
        expected_operator_sid=expected_operator_sid,
        source_files=source_files,
        binding_self_sha256=cast(str, signed["binding_self_sha256"]),
    )
    validate_permission_authz_evidence_binding(
        binding, expected_request=request, expected_access=access_expectation
    )
    return binding


def validate_permission_authz_evidence_binding(
    binding: PermissionAuthzEvidenceBinding,
    *,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
) -> None:
    if type(binding) is not PermissionAuthzEvidenceBinding:
        _raise("Authz evidence binding has the wrong exact type")
    try:
        observation.validate_permission_observation_request(
            expected_request, expected_request=expected_request
        )
        observation.validate_permission_access_expectation(
            expected_access, expected_request=expected_request
        )
    except Exception as exc:
        _raise("independent request/access binding is invalid", exc)
    record = cast(dict[str, object], _wire(binding))
    _mapping(record, _BINDING_KEYS, name="Authz evidence binding")
    if binding.schema != AUTHZ_EVIDENCE_BINDING_SCHEMA:
        _raise("Authz evidence binding schema mismatch")
    expected = {
        "request_self_sha256": expected_request.request_self_sha256,
        "access_expectation_self_sha256": expected_access.expectation_self_sha256,
        "protocol_file_sha256": expected_request.context.protocol_file_sha256,
        "state4_authorization_file_sha256": (
            expected_request.context.state4_authorization_file_sha256
        ),
        "implementation_commit": expected_request.context.implementation_commit,
        "implementation_tree": expected_request.context.implementation_tree,
    }
    for field_name, wanted in expected.items():
        if getattr(binding, field_name) != wanted:
            _raise(f"Authz evidence binding {field_name} mismatch")
    for field_name in (
        "request_self_sha256",
        "access_expectation_self_sha256",
        "protocol_file_sha256",
        "state4_authorization_file_sha256",
        "binding_self_sha256",
    ):
        _sha(getattr(binding, field_name), name=f"binding.{field_name}")
    try:
        require_git_oid(binding.implementation_commit, name="binding.implementation_commit")
        require_git_oid(binding.implementation_tree, name="binding.implementation_tree")
        observation._validate_sid(
            binding.expected_operator_sid, name="binding.expected_operator_sid"
        )
    except Exception as exc:
        _raise("Authz evidence binding identity field is malformed", exc)
    if type(binding.source_files) is not tuple or len(binding.source_files) != len(
        AUTHZ_SOURCE_RELATIVE_PATHS
    ):
        _raise("Authz source vector has the wrong exact length")
    if tuple(item.relative_path for item in binding.source_files) != AUTHZ_SOURCE_RELATIVE_PATHS:
        _raise("Authz source vector paths or order mismatch")
    for index, item in enumerate(binding.source_files):
        if type(item) is not PermissionAuthzSourceFile:
            _raise(f"binding.source_files[{index}] has the wrong exact type")
        _text(item.relative_path, name=f"binding.source_files[{index}].relative_path")
        _integer(item.file_bytes, name=f"binding.source_files[{index}].file_bytes", minimum=1)
        _sha(item.file_sha256, name=f"binding.source_files[{index}].file_sha256")
    try:
        validate_self_hash(record, "binding_self_sha256")
    except Exception as exc:
        _raise("Authz evidence binding self-hash mismatch", exc)


def _binding_from_mapping(
    value: object,
    *,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
) -> PermissionAuthzEvidenceBinding:
    record = _mapping(value, _BINDING_KEYS, name="binding")
    sid = _decode_sid(record["expected_operator_sid"], name="binding.expected_operator_sid")
    source_values = _array(record["source_files"], name="binding.source_files")
    sources = tuple(
        PermissionAuthzSourceFile(
            **_mapping(item, _SOURCE_FILE_KEYS, name=f"binding.source_files[{index}]")
        )
        for index, item in enumerate(source_values)
    )
    binding = PermissionAuthzEvidenceBinding(
        schema=record["schema"],
        request_self_sha256=record["request_self_sha256"],
        access_expectation_self_sha256=record["access_expectation_self_sha256"],
        protocol_file_sha256=record["protocol_file_sha256"],
        state4_authorization_file_sha256=record[
            "state4_authorization_file_sha256"
        ],
        implementation_commit=record["implementation_commit"],
        implementation_tree=record["implementation_tree"],
        expected_operator_sid=sid,
        source_files=sources,
        binding_self_sha256=record["binding_self_sha256"],
    )
    validate_permission_authz_evidence_binding(
        binding, expected_request=expected_request, expected_access=expected_access
    )
    return binding


def build_private_permission_authz_record(
    *,
    binding: PermissionAuthzEvidenceBinding,
    request: observation.PermissionObservationRequest,
    access_expectation: observation.PermissionAccessExpectation,
    retention_destination_self_sha256: str,
    store_expectation_self_sha256: str,
    base_observation: PermissionAuthzBaseObservation | Mapping[str, object],
    held_inputs: PermissionAuthzHeldInputs | None,
    authz: PermissionAuthzWireEvidence | Mapping[str, object],
    authority_operations: PermissionAuthzAuthorityOperations | Mapping[str, object],
    controller_failure_reason: str | None,
) -> dict[str, object]:
    validate_permission_authz_evidence_binding(
        binding, expected_request=request, expected_access=access_expectation
    )
    _sha(
        retention_destination_self_sha256,
        name="retention_destination_self_sha256",
    )
    _sha(store_expectation_self_sha256, name="store_expectation_self_sha256")
    unsigned: dict[str, object] = {
        "schema": PRIVATE_PERMISSION_AUTHZ_SCHEMA,
        "binding": _wire(binding),
        "request": _wire(request),
        "access_expectation": _wire(access_expectation),
        "retention_destination_self_sha256": retention_destination_self_sha256,
        "store_expectation_self_sha256": store_expectation_self_sha256,
        "base_observation": _wire(base_observation),
        "held_inputs": _wire(held_inputs),
        "authz": _wire(authz),
        "authority_operations": _wire(authority_operations),
        "controller_failure_reason": controller_failure_reason,
        "permission_acceptance_emitted": False,
    }
    try:
        record = with_self_hash(unsigned, "record_self_sha256")
    except Exception as exc:
        _raise("private Authz record could not be canonicalized", exc)
    validate_private_permission_authz_record(
        record,
        expected_request=request,
        expected_access=access_expectation,
        expected_binding=binding,
        expected_destination_self_sha256=retention_destination_self_sha256,
        expected_store_expectation_self_sha256=store_expectation_self_sha256,
    )
    return record


def validate_private_permission_authz_record(
    record: Mapping[str, object],
    *,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
    expected_binding: PermissionAuthzEvidenceBinding,
    expected_destination_self_sha256: str,
    expected_store_expectation_self_sha256: str,
) -> None:
    if type(record) is not dict:
        _raise("private Authz record has the wrong exact mapping type")
    candidate = _mapping(record, _ROOT_KEYS, name="private Authz record")
    try:
        observation.validate_permission_observation_request(
            expected_request, expected_request=expected_request
        )
        observation.validate_permission_access_expectation(
            expected_access, expected_request=expected_request
        )
    except Exception as exc:
        _raise("private Authz expected request/access is invalid", exc)
    validate_permission_authz_evidence_binding(
        expected_binding,
        expected_request=expected_request,
        expected_access=expected_access,
    )
    if candidate["schema"] != PRIVATE_PERMISSION_AUTHZ_SCHEMA:
        _raise("private Authz record schema mismatch")
    decoded_binding = _binding_from_mapping(
        candidate["binding"],
        expected_request=expected_request,
        expected_access=expected_access,
    )
    if decoded_binding != expected_binding:
        _raise("private Authz record binding differs from independent binding")
    if canonical_json_bytes(candidate["request"]) != canonical_json_bytes(_wire(expected_request)):
        _raise("private Authz request representation mismatch")
    if canonical_json_bytes(candidate["access_expectation"]) != canonical_json_bytes(_wire(expected_access)):
        _raise("private Authz access expectation representation mismatch")
    if candidate["retention_destination_self_sha256"] != _sha(
        expected_destination_self_sha256,
        name="expected_destination_self_sha256",
    ):
        _raise("private Authz destination binding mismatch")
    if candidate["store_expectation_self_sha256"] != _sha(
        expected_store_expectation_self_sha256,
        name="expected_store_expectation_self_sha256",
    ):
        _raise("private Authz store binding mismatch")
    _decode_base_observation(candidate["base_observation"])
    held = _decode_held_inputs(candidate["held_inputs"])
    _decode_authz(candidate["authz"], held=held)
    operations = _mapping(
        candidate["authority_operations"], ("parent", "child"), name="authority_operations"
    )
    _decode_operation_projection(
        operations["parent"], expected_operation="observe"
    )
    _decode_operation_projection(
        operations["child"], expected_operation="observe_authz"
    )
    controller_failure = candidate["controller_failure_reason"]
    if controller_failure is not None and (
        type(controller_failure) is not str
        or controller_failure not in _CONTROLLER_FAILURE_REASONS
    ):
        _raise("private Authz controller failure reason is invalid")
    if candidate["permission_acceptance_emitted"] is not False:
        _raise("private Authz record cannot emit permission acceptance")
    _scalar_tree(candidate)
    try:
        validate_self_hash(candidate, "record_self_sha256")
    except Exception as exc:
        _raise("private Authz record self-hash mismatch", exc)


def canonical_private_permission_authz_bytes(
    record: Mapping[str, object],
    *,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
    expected_binding: PermissionAuthzEvidenceBinding,
    expected_destination_self_sha256: str,
    expected_store_expectation_self_sha256: str,
) -> bytes:
    validate_private_permission_authz_record(
        record,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_binding=expected_binding,
        expected_destination_self_sha256=expected_destination_self_sha256,
        expected_store_expectation_self_sha256=(
            expected_store_expectation_self_sha256
        ),
    )
    try:
        return canonical_json_bytes(dict(record))
    except Exception as exc:
        _raise("private Authz record is not canonical-JSON encodable", exc)


def strict_private_permission_authz_loads(
    data: bytes,
    *,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
    expected_binding: PermissionAuthzEvidenceBinding,
    expected_destination_self_sha256: str,
    expected_store_expectation_self_sha256: str,
) -> dict[str, object]:
    if type(data) is not bytes:
        _raise("private Authz bytes must have the exact bytes type")
    try:
        record = strict_canonical_json_loads(data)
    except Exception as exc:
        _raise("private Authz bytes are not canonical JSON", exc)
    validate_private_permission_authz_record(
        record,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_binding=expected_binding,
        expected_destination_self_sha256=expected_destination_self_sha256,
        expected_store_expectation_self_sha256=(
            expected_store_expectation_self_sha256
        ),
    )
    if canonical_json_bytes(record) != data:
        _raise("private Authz bytes did not round-trip exactly")
    return record


def private_permission_authz_commitment(
    data: bytes,
    *,
    authorization_file_sha256: str,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
    expected_binding: PermissionAuthzEvidenceBinding,
    expected_destination_self_sha256: str,
    expected_store_expectation_self_sha256: str,
) -> str:
    authorization = _sha(
        authorization_file_sha256, name="authorization_file_sha256"
    )
    if authorization != expected_request.context.state4_authorization_file_sha256:
        _raise("Authz commitment full A4 binding mismatch")
    strict_private_permission_authz_loads(
        data,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_binding=expected_binding,
        expected_destination_self_sha256=expected_destination_self_sha256,
        expected_store_expectation_self_sha256=(
            expected_store_expectation_self_sha256
        ),
    )
    return sha256_bytes(
        PRIVATE_PERMISSION_AUTHZ_DOMAIN.encode("utf-8")
        + b"\n"
        + authorization.encode("ascii")
        + b"\n"
        + data
    )


__all__ = [
    "AUTHZ_EVIDENCE_BINDING_SCHEMA",
    "AUTHZ_SOURCE_RELATIVE_PATHS",
    "PRIVATE_PERMISSION_AUTHZ_DOMAIN",
    "PRIVATE_PERMISSION_AUTHZ_SCHEMA",
    "PermissionAuthzAuthorityOperations",
    "PermissionAuthzBaseObservation",
    "PermissionAuthzChildAccounting",
    "PermissionAuthzEvidenceBinding",
    "PermissionAuthzHeldInputs",
    "PermissionAuthzRecordError",
    "PermissionAuthzSourceFile",
    "PermissionAuthzWireEvidence",
    "PermissionOperationEvidenceProjection",
    "build_permission_authz_evidence_binding",
    "build_private_permission_authz_record",
    "canonical_private_permission_authz_bytes",
    "private_permission_authz_commitment",
    "strict_private_permission_authz_loads",
    "validate_permission_authz_evidence_binding",
    "validate_private_permission_authz_record",
]
