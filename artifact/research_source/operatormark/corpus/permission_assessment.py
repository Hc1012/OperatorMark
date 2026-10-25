"""Pure integration of supplied retained, DACL, and Authz evidence.

The assessment is deliberately inactive and nonaccepting.  It performs no
native calls, file access, account discovery, authority accounting, retention,
permission changes, or production integration.  It establishes correspondence
and a bounded set of checks for one captured subject and its immediate parent.

Even a satisfied result is not complete principal coverage, full ancestor or
tree coverage, authenticated enrollment, live freshness, host qualification,
production permission acceptance, or State 3 readiness.  The trusted caller
remains responsible for the provenance of every independently supplied value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, TypeAlias, cast

from operatormark.corpus import permission_dacl_policy as _dacl
from operatormark.corpus import permission_evidence_verifier as _verifier
from operatormark.corpus import permission_observation_windows as _observation
from operatormark.corpus import permission_principal_authz_windows as _authz
from operatormark.corpus import private_permission_evidence as _evidence
from operatormark.corpus import storage as _storage


__all__ = [
    "PermissionAssessmentValidationError",
    "PermissionBundleAssessment",
    "assess_supplied_permission_bundle",
]


PermissionBundleStatus: TypeAlias = Literal[
    "satisfied", "not_satisfied", "incomplete"
]
PermissionBundleReasonCode: TypeAlias = Literal[
    "all_named_checks_satisfied",
    "captured_read_only_attribute_missing",
    "recorded_current_token_masks_not_satisfied",
    "recorded_current_token_masks_unavailable",
    "subject_dacl_outside_operator_only_profile",
    "immediate_parent_dacl_outside_operator_only_profile",
    "subject_authz_unavailable",
    "subject_authz_read_data_missing",
    "subject_authz_prohibited_rights_present",
    "immediate_parent_authz_unavailable",
    "immediate_parent_authz_delete_child_present",
]
ValidationReasonCode: TypeAlias = Literal[
    "retained_evidence_rejected",
    "observation_decode_rejected",
    "operator_sid_rejected",
    "operator_sid_mismatch",
    "authz_identity_rejected",
    "authz_identity_mismatch",
    "observation_correspondence_rejected",
    "authz_results_rejected",
    "authz_result_rejected",
    "authz_result_mismatch",
    "dacl_assessment_rejected",
]


_FILE_PROHIBITED_MASK = (
    _storage.STATE4_FILE_PERMISSION_WRITE_DATA
    | _storage.STATE4_FILE_PERMISSION_APPEND_DATA
    | _storage.STATE4_FILE_PERMISSION_DELETE
    | _storage.STATE4_FILE_PERMISSION_WRITE_EA
    | _storage.STATE4_FILE_PERMISSION_WRITE_ATTRIBUTES
    | _storage.STATE4_FILE_PERMISSION_WRITE_DAC
    | _storage.STATE4_FILE_PERMISSION_WRITE_OWNER
)
_DWORD_MAX = 0xFFFFFFFF
_AUTHZ_API_STATUSES = frozenset({"not_called", "succeeded", "failed"})
_AUTHZ_ACCESS_STATUSES = frozenset({"granted", "denied", "unavailable"})
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
    ("invalid_native_output", "token_before"): frozenset(
        {"GetTokenInformation"}
    ),
    ("invalid_native_output", "resource_manager"): frozenset(
        {"AuthzInitializeResourceManager"}
    ),
    ("invalid_native_output", "context"): frozenset(
        {"AuthzInitializeContextFromToken"}
    ),
    ("invalid_native_output", "access_check"): frozenset({"AuthzAccessCheck"}),
    ("invalid_native_output", "token_after"): frozenset(
        {"GetTokenInformation"}
    ),
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
_ERROR_MESSAGES: dict[ValidationReasonCode, str] = {
    "retained_evidence_rejected": "retained permission evidence was rejected",
    "observation_decode_rejected": "captured permission observation was rejected",
    "operator_sid_rejected": "independent operator identity was rejected",
    "operator_sid_mismatch": "independent operator identity does not correspond",
    "authz_identity_rejected": "independent Authz token identity was rejected",
    "authz_identity_mismatch": "independent Authz token identity does not correspond",
    "observation_correspondence_rejected": "captured object correspondence was rejected",
    "authz_results_rejected": "supplied Authz result collection was rejected",
    "authz_result_rejected": "a supplied Authz result was rejected",
    "authz_result_mismatch": "a supplied Authz result does not correspond",
    "dacl_assessment_rejected": "a required DACL assessment was rejected",
}


class PermissionAssessmentValidationError(ValueError):
    """Redacted rejection of malformed or noncorresponding supplied evidence."""

    def __init__(self, reason_code: ValidationReasonCode) -> None:
        self.reason_code = reason_code
        super().__init__(_ERROR_MESSAGES[reason_code])


@dataclass(frozen=True, slots=True)
class PermissionBundleAssessment:
    """Redacted, immutable result for the bounded supplied checks.

    ``recorded_access_check_assessment`` is the existing verifier result and is
    kept distinct from both independently supplied Authz outcomes.  Its four
    commitments are the existing authorization-bound redactions; this type does
    not publish raw identities, paths, descriptors, token identifiers, or the
    DACL helper's unbound SID hash.
    """

    recorded_access_check_assessment: _verifier.PermissionEvidenceAssessment
    subject_dacl_assessment_status: _dacl.DaclAllowlistAssessmentStatus
    subject_dacl_reason_codes: tuple[_dacl.DaclAllowlistReasonCode, ...]
    immediate_parent_dacl_assessment_status: _dacl.DaclAllowlistAssessmentStatus
    immediate_parent_dacl_reason_codes: tuple[
        _dacl.DaclAllowlistReasonCode, ...
    ]
    subject_authz_access_status: Literal["granted", "denied", "unavailable"]
    subject_authz_granted_access_mask_hex: str | None
    subject_authz_primary_failure_reason: str | None
    subject_authz_cleanup_failure_reasons: tuple[str, ...]
    subject_authz_mask_conditions_satisfied: bool | None
    immediate_parent_authz_access_status: Literal[
        "granted", "denied", "unavailable"
    ]
    immediate_parent_authz_granted_access_mask_hex: str | None
    immediate_parent_authz_primary_failure_reason: str | None
    immediate_parent_authz_cleanup_failure_reasons: tuple[str, ...]
    immediate_parent_authz_delete_child_absent: bool | None
    supplied_checks_status: PermissionBundleStatus
    reason_codes: tuple[PermissionBundleReasonCode, ...]
    permission_acceptance_emitted: Literal[False] = field(
        init=False, default=False
    )


def _reject(reason_code: ValidationReasonCode) -> None:
    raise PermissionAssessmentValidationError(reason_code) from None


def _mapping(value: object) -> dict[str, object]:
    if type(value) is not dict:
        _reject("observation_correspondence_rejected")
    return cast(dict[str, object], value)


def _array(value: object) -> list[object]:
    if type(value) is not list:
        _reject("observation_correspondence_rejected")
    return cast(list[object], value)


def _sid_from_record(value: object) -> _observation.NativeSid:
    record = _mapping(value)
    try:
        sid = _observation.NativeSid(
            sid_string=record["sid_string"],
            sid_bytes_hex=record["sid_bytes_hex"],
        )
        return _dacl._validate_operator_sid(sid)
    except Exception:
        _reject("observation_correspondence_rejected")


def _identity_from_record(value: object) -> _observation.NativeFileIdentity:
    record = _mapping(value)
    try:
        identity = _observation.NativeFileIdentity(
            volume_serial_number_hex=record["volume_serial_number_hex"],
            file_id_128_hex=record["file_id_128_hex"],
            legacy_volume_serial_number_hex=record[
                "legacy_volume_serial_number_hex"
            ],
            legacy_file_index_hex=record["legacy_file_index_hex"],
        )
        _observation._validate_identity(identity, name="captured object identity")
        return identity
    except PermissionAssessmentValidationError:
        raise
    except Exception:
        _reject("observation_correspondence_rejected")


def _descriptor_from_record(value: object) -> _observation.NativeSecurityDescriptor:
    record = _mapping(value)
    try:
        aces = tuple(
            _observation.NativeAce(
                ace_index=ace["ace_index"],
                ace_kind=ace["ace_kind"],
                ace_type_hex=ace["ace_type_hex"],
                ace_flags_hex=ace["ace_flags_hex"],
                access_mask_hex=ace["access_mask_hex"],
                trustee_sid=_sid_from_record(ace["trustee_sid"]),
                raw_ace_hex=ace["raw_ace_hex"],
            )
            for ace in (_mapping(item) for item in _array(record["ordered_dacl_aces"]))
        )
        descriptor = _observation.NativeSecurityDescriptor(
            self_relative_descriptor_hex=record["self_relative_descriptor_hex"],
            self_relative_descriptor_sha256=record[
                "self_relative_descriptor_sha256"
            ],
            revision=record["revision"],
            control_hex=record["control_hex"],
            owner_sid=_sid_from_record(record["owner_sid"]),
            group_sid=_sid_from_record(record["group_sid"]),
            dacl_present=record["dacl_present"],
            dacl_defaulted=record["dacl_defaulted"],
            dacl_revision=record["dacl_revision"],
            dacl_protected=record["dacl_protected"],
            dacl_auto_inherited=record["dacl_auto_inherited"],
            ordered_dacl_aces=aces,
        )
        return _authz._validate_descriptor(descriptor)[0]
    except PermissionAssessmentValidationError:
        raise
    except Exception:
        _reject("observation_correspondence_rejected")


def _binding_from_recorded_object(
    snapshot_value: object,
    security_value: object,
    *,
    object_role: Literal["subject", "immediate_parent"],
    object_kind: Literal["file", "directory"],
    ancestor_index: int | None,
    expected_path: str,
) -> tuple[
    _observation.NativeSecurityDescriptor,
    _observation.NativeAccessObjectBinding,
]:
    snapshot = _mapping(snapshot_value)
    security = _mapping(security_value)
    facts = _mapping(snapshot.get("facts"))
    try:
        expected_snapshot_role = "subject" if object_role == "subject" else "ancestor"
        if (
            snapshot["object_role"] != expected_snapshot_role
            or snapshot["ancestor_index"] != ancestor_index
            or security["object_role"] != expected_snapshot_role
            or security["ancestor_index"] != ancestor_index
            or snapshot["requested_canonical_volume_guid_path"] != expected_path
            or facts["canonical_volume_guid_path"] != expected_path
            or security["canonical_volume_guid_path"] != expected_path
            or security["object_identity"] != facts["identity"]
            or (object_role == "subject" and snapshot["is_immediate_parent"] is not False)
            or (
                object_role == "immediate_parent"
                and snapshot["is_immediate_parent"] is not True
            )
            or (object_kind == "file" and facts["directory"] is not False)
            or (object_kind == "directory" and facts["directory"] is not True)
        ):
            _reject("observation_correspondence_rejected")
        descriptor = _descriptor_from_record(security["descriptor"])
        binding = _observation.NativeAccessObjectBinding(
            object_role=object_role,
            object_kind=object_kind,
            canonical_volume_guid_path=expected_path,
            identity=_identity_from_record(facts["identity"]),
            descriptor_sha256=descriptor.self_relative_descriptor_sha256,
        )
        _authz._validate_binding(binding, descriptor)
        return descriptor, binding
    except PermissionAssessmentValidationError:
        raise
    except Exception:
        _reject("observation_correspondence_rejected")


def _observation_objects(
    record: dict[str, object],
    *,
    expected_request: _observation.PermissionObservationRequest,
) -> tuple[
    tuple[_observation.NativeSecurityDescriptor, _observation.NativeAccessObjectBinding],
    tuple[_observation.NativeSecurityDescriptor, _observation.NativeAccessObjectBinding],
]:
    try:
        ancestor_paths = expected_request.subject.required_ancestor_canonical_volume_guid_paths
        if type(ancestor_paths) is not tuple or not ancestor_paths:
            _reject("observation_correspondence_rejected")
        parent_index = len(ancestor_paths) - 1
        ancestry_before = _array(record["ancestry_before"])
        ancestry_security_before = _array(record["ancestry_security_before"])
        ancestry_security_after = _array(record["ancestry_security_after"])
        access_checks = _array(record["access_checks"])
        if (
            len(ancestry_before) != len(ancestor_paths)
            or len(ancestry_security_before) != len(ancestor_paths)
            or ancestry_security_after != ancestry_security_before
            or record["subject_security_after"] != record["subject_security_before"]
            or len(access_checks) != 2
        ):
            _reject("observation_correspondence_rejected")
        subject = _binding_from_recorded_object(
            record["subject_before"],
            record["subject_security_before"],
            object_role="subject",
            object_kind="file",
            ancestor_index=None,
            expected_path=expected_request.subject.canonical_volume_guid_path,
        )
        parent = _binding_from_recorded_object(
            ancestry_before[parent_index],
            ancestry_security_before[parent_index],
            object_role="immediate_parent",
            object_kind="directory",
            ancestor_index=parent_index,
            expected_path=ancestor_paths[parent_index],
        )
        if (
            _mapping(access_checks[0])["object_binding"] != asdict(subject[1])
            or _mapping(access_checks[1])["object_binding"] != asdict(parent[1])
        ):
            _reject("observation_correspondence_rejected")
        return subject, parent
    except PermissionAssessmentValidationError:
        raise
    except Exception:
        _reject("observation_correspondence_rejected")


def _captured_token_identity(value: object) -> _authz.SuppliedTokenIdentity:
    snapshot = _mapping(value)
    facts = _mapping(snapshot.get("facts"))
    try:
        if (
            snapshot["token_role"] != "access_check_duplicate"
            or snapshot["selection_source"] != "duplicated_effective_context"
        ):
            _reject("authz_identity_mismatch")
        return _authz._validate_identity(
            _authz.SuppliedTokenIdentity(
                token_id_luid_hex=facts["token_id_luid_hex"],
                authentication_id_luid_hex=facts["authentication_id_luid_hex"],
                modified_id_luid_hex=facts["modified_id_luid_hex"],
                token_type=facts["token_type"],
                impersonation_level=facts["impersonation_level"],
            )
        )
    except PermissionAssessmentValidationError:
        raise
    except Exception:
        _reject("authz_identity_mismatch")


def _validate_independent_correspondence(
    record: dict[str, object],
    *,
    expected_operator_sid: _observation.NativeSid,
    expected_authz_token_identity: _authz.SuppliedTokenIdentity,
) -> None:
    try:
        operator_sid = _dacl._validate_operator_sid(expected_operator_sid)
    except Exception:
        _reject("operator_sid_rejected")
    try:
        token_identity = _authz._validate_identity(expected_authz_token_identity)
    except Exception:
        _reject("authz_identity_rejected")

    for field_name in ("effective_context_before", "effective_context_after"):
        snapshot = _mapping(record[field_name])
        facts = _mapping(snapshot.get("facts"))
        captured = _sid_from_record(facts.get("user"))
        if not _dacl._same_sid(captured, operator_sid):
            _reject("operator_sid_mismatch")
    for field_name in ("access_check_token_before", "access_check_token_after"):
        if _captured_token_identity(record[field_name]) != token_identity:
            _reject("authz_identity_mismatch")


def _exact_optional_dword(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= _DWORD_MAX:
        _reject("authz_result_rejected")
    return value


def _validate_failure(
    value: object,
    *,
    cleanup: bool,
) -> _authz.PermissionPrincipalAuthzFailure:
    if type(value) is not _authz.PermissionPrincipalAuthzFailure:
        _reject("authz_result_rejected")
    failure = cast(_authz.PermissionPrincipalAuthzFailure, value)
    if any(
        type(getattr(failure, name)) is not str
        for name in ("reason", "stage", "operation", "message")
    ):
        _reject("authz_result_rejected")
    operations = _FAILURE_OPERATIONS.get((failure.reason, failure.stage))
    if (
        operations is None
        or failure.operation not in operations
        or failure.message != _authz._FAILURE_MESSAGES.get(failure.reason)
        or (cleanup and failure.reason != "cleanup_failed")
        or (not cleanup and failure.reason == "cleanup_failed")
    ):
        _reject("authz_result_rejected")
    if failure.native_error_code is not None and (
        type(failure.native_error_code) is not int
        or not 0 < failure.native_error_code <= _DWORD_MAX
    ):
        _reject("authz_result_rejected")
    return failure


def _validate_statistics(
    value: object,
) -> _authz.NativeTokenStatistics | None:
    if value is None:
        return None
    try:
        return _authz._validate_statistics(value)
    except Exception:
        _reject("authz_result_rejected")


def _matches_expected(
    statistics: _authz.NativeTokenStatistics | None,
    expected: _authz.SuppliedTokenIdentity,
) -> bool:
    return statistics is not None and _authz._statistics_match_expected(
        statistics, expected
    )


def _validate_unavailable_statistics(
    *,
    api_status: str,
    before: _authz.NativeTokenStatistics | None,
    after: _authz.NativeTokenStatistics | None,
    expected: _authz.SuppliedTokenIdentity,
    primary: _authz.PermissionPrincipalAuthzFailure | None,
    cleanup: tuple[_authz.PermissionPrincipalAuthzFailure, ...],
) -> None:
    if primary is None:
        if (
            api_status != "succeeded"
            or not cleanup
            or not _matches_expected(before, expected)
            or not _matches_expected(after, expected)
            or after != before
        ):
            _reject("authz_result_rejected")
        return

    stage = primary.stage
    reason = primary.reason
    if api_status == "failed":
        if (
            reason != "native_call_failed"
            or stage != "access_check"
            or not _matches_expected(before, expected)
            or after is not None
        ):
            _reject("authz_result_rejected")
        return
    if api_status == "succeeded":
        if stage == "access_check":
            if not _matches_expected(before, expected) or after is not None:
                _reject("authz_result_rejected")
            return
        if stage == "token_after":
            if not _matches_expected(before, expected):
                _reject("authz_result_rejected")
            if reason == "token_changed":
                if after is None or (
                    _matches_expected(after, expected) and after == before
                ):
                    _reject("authz_result_rejected")
            elif after is not None:
                _reject("authz_result_rejected")
            return
        _reject("authz_result_rejected")
    if api_status != "not_called" or after is not None:
        _reject("authz_result_rejected")
    if stage == "token_before":
        if reason == "token_identity_mismatch":
            if before is None or _matches_expected(before, expected):
                _reject("authz_result_rejected")
        elif before is not None:
            _reject("authz_result_rejected")
        return
    if stage in {"resource_manager", "context"}:
        if not _matches_expected(before, expected):
            _reject("authz_result_rejected")
        return
    _reject("authz_result_rejected")


def _validate_authz_result(
    value: object,
    *,
    expected_identity: _authz.SuppliedTokenIdentity,
    expected_descriptor: _observation.NativeSecurityDescriptor,
    expected_binding: _observation.NativeAccessObjectBinding,
) -> _authz.SuppliedTokenAccessResult:
    if type(value) is not _authz.SuppliedTokenAccessResult:
        _reject("authz_result_rejected")
    result = cast(_authz.SuppliedTokenAccessResult, value)
    if (
        type(result.access_check_api_status) is not str
        or result.access_check_api_status not in _AUTHZ_API_STATUSES
        or type(result.access_status) is not str
        or result.access_status not in _AUTHZ_ACCESS_STATUSES
        or type(result.requested_access_mask) is not int
        or result.requested_access_mask != _authz.MAXIMUM_ALLOWED
        or type(result.permission_acceptance_emitted) is not bool
        or result.permission_acceptance_emitted is not False
        or type(result.cleanup_failures) is not tuple
    ):
        _reject("authz_result_rejected")
    mask = _exact_optional_dword(result.granted_access_mask)
    per_result_error = _exact_optional_dword(result.per_result_error_code)
    sacl_result = _exact_optional_dword(result.sacl_evaluation_results)
    try:
        result_identity = _authz._validate_identity(result.expected_token_identity)
    except Exception:
        _reject("authz_result_rejected")
    if result_identity != expected_identity:
        _reject("authz_result_mismatch")
    try:
        _authz._validate_binding(result.object_binding, expected_descriptor)
    except Exception:
        _reject("authz_result_rejected")
    if (
        result.object_binding != expected_binding
        or type(result.descriptor_sha256) is not str
        or result.descriptor_sha256
        != expected_descriptor.self_relative_descriptor_sha256
    ):
        _reject("authz_result_mismatch")

    before = _validate_statistics(result.token_statistics_before)
    after = _validate_statistics(result.token_statistics_after)
    primary = (
        None
        if result.primary_failure is None
        else _validate_failure(result.primary_failure, cleanup=False)
    )
    cleanup = tuple(
        _validate_failure(item, cleanup=True) for item in result.cleanup_failures
    )

    if result.access_status in {"granted", "denied"}:
        if (
            result.access_check_api_status != "succeeded"
            or primary is not None
            or cleanup
            or before is None
            or after is None
            or not _matches_expected(before, expected_identity)
            or not _matches_expected(after, expected_identity)
            or after != before
            or mask is None
            or sacl_result != 0
        ):
            _reject("authz_result_rejected")
        if result.access_status == "granted":
            if per_result_error != _authz.ERROR_SUCCESS:
                _reject("authz_result_rejected")
        elif mask != 0 or per_result_error != _authz.ERROR_ACCESS_DENIED:
            _reject("authz_result_rejected")
    else:
        if (
            mask is not None
            or per_result_error is not None
            or sacl_result is not None
            or (primary is None and not cleanup)
        ):
            _reject("authz_result_rejected")
        _validate_unavailable_statistics(
            api_status=result.access_check_api_status,
            before=before,
            after=after,
            expected=expected_identity,
            primary=primary,
            cleanup=cleanup,
        )
    return result


def _validate_authz_results(
    supplied_authz_results: object,
    *,
    expected_identity: _authz.SuppliedTokenIdentity,
    subject: tuple[
        _observation.NativeSecurityDescriptor,
        _observation.NativeAccessObjectBinding,
    ],
    parent: tuple[
        _observation.NativeSecurityDescriptor,
        _observation.NativeAccessObjectBinding,
    ],
) -> tuple[_authz.SuppliedTokenAccessResult, _authz.SuppliedTokenAccessResult]:
    if type(supplied_authz_results) is not tuple or len(supplied_authz_results) != 2:
        _reject("authz_results_rejected")
    values = cast(tuple[object, object], supplied_authz_results)
    return (
        _validate_authz_result(
            values[0],
            expected_identity=expected_identity,
            expected_descriptor=subject[0],
            expected_binding=subject[1],
        ),
        _validate_authz_result(
            values[1],
            expected_identity=expected_identity,
            expected_descriptor=parent[0],
            expected_binding=parent[1],
        ),
    )


def _dacl_assessment(
    descriptor: _observation.NativeSecurityDescriptor,
    binding: _observation.NativeAccessObjectBinding,
    *,
    expected_operator_sid: _observation.NativeSid,
) -> _dacl.SuppliedDaclAllowlistResult:
    result = _dacl.assess_supplied_dacl_allowlist(
        _dacl.SuppliedDaclAllowlistRequest(
            expected_operator_sid=expected_operator_sid,
            descriptor=descriptor,
            object_binding=binding,
        )
    )
    if (
        type(result) is not _dacl.SuppliedDaclAllowlistResult
        or type(result.assessment_status) is not str
        or result.assessment_status
        not in {
            "matches_operator_only_allowlist",
            "outside_operator_only_allowlist",
            "unavailable",
        }
        or type(result.reason_codes) is not tuple
        or type(result.examined_allow_ace_indices) is not tuple
        or type(result.relevant_ace_indices) is not tuple
        or any(
            type(index) is not int or index < 0
            for index in (
                result.examined_allow_ace_indices + result.relevant_ace_indices
            )
        )
        or result.descriptor_sha256 != descriptor.self_relative_descriptor_sha256
        or result.object_binding != binding
        or type(result.permission_acceptance_emitted) is not bool
        or result.permission_acceptance_emitted is not False
        or result.assessment_status == "unavailable"
    ):
        _reject("dacl_assessment_rejected")
    return result


def _authz_projection(
    result: _authz.SuppliedTokenAccessResult,
    *,
    subject: bool,
) -> tuple[str | None, str | None, tuple[str, ...], bool | None]:
    mask = result.granted_access_mask
    mask_hex = None if mask is None else hex(mask)
    primary_reason = (
        None if result.primary_failure is None else result.primary_failure.reason
    )
    cleanup_reasons = tuple(item.reason for item in result.cleanup_failures)
    if result.access_status == "unavailable":
        return mask_hex, primary_reason, cleanup_reasons, None
    assert mask is not None
    if subject:
        condition = (
            result.access_status == "granted"
            and bool(mask & _storage.STATE4_FILE_PERMISSION_READ_DATA)
            and not bool(mask & _FILE_PROHIBITED_MASK)
        )
    else:
        condition = not bool(
            mask & _storage.STATE4_PARENT_PERMISSION_DELETE_CHILD
        )
    return mask_hex, primary_reason, cleanup_reasons, condition


def assess_supplied_permission_bundle(
    observation_bytes: bytes,
    receipt: _evidence.RetainedPermissionEvidence,
    supplied_authz_results: tuple[
        _authz.SuppliedTokenAccessResult,
        _authz.SuppliedTokenAccessResult,
    ],
    *,
    expected_request: _observation.PermissionObservationRequest,
    expected_access: _observation.PermissionAccessExpectation,
    expected_store: _evidence.ConfidentialStoreExpectation,
    expected_destination: _evidence.PermissionEvidenceDestination,
    expected_retained_snapshot: _evidence.StoreObjectSnapshot,
    expected_operator_sid: _observation.NativeSid,
    expected_authz_token_identity: _authz.SuppliedTokenIdentity,
) -> PermissionBundleAssessment:
    """Assess one supplied subject/parent permission-evidence bundle in memory."""

    try:
        recorded = _verifier.verify_retained_permission_observation(
            observation_bytes,
            receipt,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=expected_store,
            expected_destination=expected_destination,
            expected_retained_snapshot=expected_retained_snapshot,
        )
    except Exception:
        _reject("retained_evidence_rejected")
    try:
        record = _evidence.strict_private_permission_observation_loads(
            observation_bytes,
            expected_request=expected_request,
            expected_access=expected_access,
        )
    except Exception:
        _reject("observation_decode_rejected")

    _validate_independent_correspondence(
        record,
        expected_operator_sid=expected_operator_sid,
        expected_authz_token_identity=expected_authz_token_identity,
    )
    subject, parent = _observation_objects(
        record, expected_request=expected_request
    )
    authz_subject, authz_parent = _validate_authz_results(
        supplied_authz_results,
        expected_identity=expected_authz_token_identity,
        subject=subject,
        parent=parent,
    )
    dacl_subject = _dacl_assessment(
        subject[0], subject[1], expected_operator_sid=expected_operator_sid
    )
    dacl_parent = _dacl_assessment(
        parent[0], parent[1], expected_operator_sid=expected_operator_sid
    )

    (
        subject_mask_hex,
        subject_primary_reason,
        subject_cleanup_reasons,
        subject_authz_satisfied,
    ) = _authz_projection(authz_subject, subject=True)
    (
        parent_mask_hex,
        parent_primary_reason,
        parent_cleanup_reasons,
        parent_authz_satisfied,
    ) = _authz_projection(authz_parent, subject=False)

    reasons: list[PermissionBundleReasonCode] = []
    if not recorded.captured_file_has_read_only_attribute:
        reasons.append("captured_read_only_attribute_missing")
    if recorded.current_token_mask_assessment_status == "not_evaluated":
        reasons.append("recorded_current_token_masks_unavailable")
    elif recorded.current_token_mask_assessment_status != "satisfied":
        reasons.append("recorded_current_token_masks_not_satisfied")
    if dacl_subject.assessment_status != "matches_operator_only_allowlist":
        reasons.append("subject_dacl_outside_operator_only_profile")
    if dacl_parent.assessment_status != "matches_operator_only_allowlist":
        reasons.append("immediate_parent_dacl_outside_operator_only_profile")
    if subject_authz_satisfied is None:
        reasons.append("subject_authz_unavailable")
    else:
        subject_mask = cast(int, authz_subject.granted_access_mask)
        if (
            authz_subject.access_status != "granted"
            or not bool(subject_mask & _storage.STATE4_FILE_PERMISSION_READ_DATA)
        ):
            reasons.append("subject_authz_read_data_missing")
        if bool(subject_mask & _FILE_PROHIBITED_MASK):
            reasons.append("subject_authz_prohibited_rights_present")
    if parent_authz_satisfied is None:
        reasons.append("immediate_parent_authz_unavailable")
    elif not parent_authz_satisfied:
        reasons.append("immediate_parent_authz_delete_child_present")

    incomplete = (
        recorded.current_token_mask_assessment_status == "not_evaluated"
        or subject_authz_satisfied is None
        or parent_authz_satisfied is None
    )
    if incomplete:
        status: PermissionBundleStatus = "incomplete"
    elif reasons:
        status = "not_satisfied"
    else:
        status = "satisfied"
        reasons.append("all_named_checks_satisfied")

    return PermissionBundleAssessment(
        recorded_access_check_assessment=recorded,
        subject_dacl_assessment_status=dacl_subject.assessment_status,
        subject_dacl_reason_codes=dacl_subject.reason_codes,
        immediate_parent_dacl_assessment_status=dacl_parent.assessment_status,
        immediate_parent_dacl_reason_codes=dacl_parent.reason_codes,
        subject_authz_access_status=authz_subject.access_status,
        subject_authz_granted_access_mask_hex=subject_mask_hex,
        subject_authz_primary_failure_reason=subject_primary_reason,
        subject_authz_cleanup_failure_reasons=subject_cleanup_reasons,
        subject_authz_mask_conditions_satisfied=subject_authz_satisfied,
        immediate_parent_authz_access_status=authz_parent.access_status,
        immediate_parent_authz_granted_access_mask_hex=parent_mask_hex,
        immediate_parent_authz_primary_failure_reason=parent_primary_reason,
        immediate_parent_authz_cleanup_failure_reasons=parent_cleanup_reasons,
        immediate_parent_authz_delete_child_absent=parent_authz_satisfied,
        supplied_checks_status=status,
        reason_codes=tuple(reasons),
    )
