"""Pure, conservative assessment of one supplied Windows DACL.

The caller supplies an independently obtained operator SID, copied security
descriptor, and object binding.  This module performs no native calls, loads no
DLLs, accesses no files, discovers no accounts, and changes no permissions.

A matching result establishes only that the supplied descriptor bytes name the
supplied operator as owner and as trustee of every ordinary allow ACE.  An empty
DACL can therefore match while providing no required read access.  The scan does
not calculate effective access, write denial, or zero broad-principal access.
Owner-implied rights, privileges, actual token access, parent-directory effects,
descriptor freshness, account classification and qualification, and production
permission acceptance remain separate checks.  The trusted caller remains
responsible for authenticating and classifying the supplied operator identity.

This conservative subset neither approves a real account or ACL nor removes the
frozen protocol's separately approved local-user residual-risk route.  Broader
account variants, including special exceptions for administrators, SYSTEM,
services, or local users, are unsupported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from operatormark.corpus.canonical import sha256_bytes
from operatormark.corpus import permission_observation_windows as _observation
from operatormark.corpus import permission_principal_authz_windows as _authz
from operatormark.corpus import storage as _storage


__all__ = [
    "DaclAllowlistAssessmentStatus",
    "DaclAllowlistReasonCode",
    "SuppliedDaclAllowlistRequest",
    "SuppliedDaclAllowlistResult",
    "assess_supplied_dacl_allowlist",
]


DaclAllowlistAssessmentStatus: TypeAlias = Literal[
    "matches_operator_only_allowlist",
    "outside_operator_only_allowlist",
    "unavailable",
]
DaclAllowlistReasonCode: TypeAlias = Literal[
    "operator_only_allowlist_match",
    "expected_operator_is_broad_principal",
    "descriptor_owner_mismatch",
    "unexpected_allow_trustee",
    "request_rejected",
    "operator_sid_rejected",
    "descriptor_rejected",
    "object_binding_rejected",
]


@dataclass(frozen=True, slots=True)
class SuppliedDaclAllowlistRequest:
    expected_operator_sid: _observation.NativeSid
    descriptor: _observation.NativeSecurityDescriptor
    object_binding: _observation.NativeAccessObjectBinding


@dataclass(frozen=True, slots=True)
class SuppliedDaclAllowlistResult:
    assessment_status: DaclAllowlistAssessmentStatus
    reason_codes: tuple[DaclAllowlistReasonCode, ...]
    examined_allow_ace_indices: tuple[int, ...]
    relevant_ace_indices: tuple[int, ...]
    expected_operator_sid_sha256: str | None
    descriptor_sha256: str | None
    object_binding: _observation.NativeAccessObjectBinding | None
    message: str
    permission_acceptance_emitted: Literal[False] = field(
        init=False,
        default=False,
    )


_BROAD_PRINCIPAL_SID_TEXTS = frozenset(
    sid_text.casefold()
    for _, sid_text, _ in _storage._BROAD_PRINCIPALS
)

_MESSAGES: dict[DaclAllowlistReasonCode, str] = {
    "operator_only_allowlist_match": (
        "the supplied bytes match the bounded owner and allow-entry correspondence"
    ),
    "expected_operator_is_broad_principal": (
        "the expected operator is outside the conservative principal profile"
    ),
    "descriptor_owner_mismatch": (
        "the supplied descriptor owner differs from the expected operator"
    ),
    "unexpected_allow_trustee": (
        "one or more allow entries name an unexpected trustee"
    ),
    "request_rejected": "the supplied assessment request is malformed",
    "operator_sid_rejected": "the independently supplied operator SID is malformed",
    "descriptor_rejected": (
        "the supplied descriptor is malformed or outside the supported descriptor form"
    ),
    "object_binding_rejected": (
        "the independent object binding is malformed or does not bind the descriptor"
    ),
}


def _result(
    status: DaclAllowlistAssessmentStatus,
    reason_codes: tuple[DaclAllowlistReasonCode, ...],
    *,
    examined_allow_ace_indices: tuple[int, ...] = (),
    relevant_ace_indices: tuple[int, ...] = (),
    operator_sid: _observation.NativeSid | None = None,
    descriptor: _observation.NativeSecurityDescriptor | None = None,
    object_binding: _observation.NativeAccessObjectBinding | None = None,
) -> SuppliedDaclAllowlistResult:
    if status == "matches_operator_only_allowlist":
        message = _MESSAGES["operator_only_allowlist_match"]
    elif status == "outside_operator_only_allowlist":
        message = "the supplied descriptor is outside the conservative operator-only profile"
    else:
        message = _MESSAGES[reason_codes[0]]
    return SuppliedDaclAllowlistResult(
        assessment_status=status,
        reason_codes=reason_codes,
        examined_allow_ace_indices=examined_allow_ace_indices,
        relevant_ace_indices=relevant_ace_indices,
        expected_operator_sid_sha256=(
            None
            if operator_sid is None
            else sha256_bytes(bytes.fromhex(operator_sid.sid_bytes_hex))
        ),
        descriptor_sha256=(
            None
            if descriptor is None
            else descriptor.self_relative_descriptor_sha256
        ),
        object_binding=object_binding,
        message=message,
    )


def _validate_operator_sid(value: object) -> _observation.NativeSid:
    if (
        type(value) is not _observation.NativeSid
        or type(value.sid_string) is not str
        or type(value.sid_bytes_hex) is not str
    ):
        raise ValueError("operator SID representation")
    try:
        _observation._validate_sid(value, name="expected_operator_sid")
    except Exception as exc:
        raise ValueError("operator SID validation") from exc
    return value


def _same_sid(
    left: _observation.NativeSid,
    right: _observation.NativeSid,
) -> bool:
    return (
        left.sid_string == right.sid_string
        and left.sid_bytes_hex == right.sid_bytes_hex
    )


def assess_supplied_dacl_allowlist(
    request: object,
) -> SuppliedDaclAllowlistResult:
    """Assess the bounded owner/allow-entry profile without operational effects.

    All validation failures become redacted ``unavailable`` results.  Syntactically
    valid descriptors that fail the profile remain distinguishable as
    ``outside_operator_only_allowlist``.  No outcome emits permission acceptance.
    """

    if type(request) is not SuppliedDaclAllowlistRequest:
        return _result("unavailable", ("request_rejected",))

    try:
        operator_sid = _validate_operator_sid(request.expected_operator_sid)
    except Exception:
        return _result("unavailable", ("operator_sid_rejected",))

    try:
        descriptor, _ = _authz._validate_descriptor(request.descriptor)
    except Exception:
        return _result(
            "unavailable",
            ("descriptor_rejected",),
            operator_sid=operator_sid,
        )

    try:
        object_binding = _authz._validate_binding(
            request.object_binding,
            descriptor,
        )
    except Exception:
        return _result(
            "unavailable",
            ("object_binding_rejected",),
            operator_sid=operator_sid,
            descriptor=descriptor,
        )

    allow_aces = tuple(
        ace for ace in descriptor.ordered_dacl_aces if ace.ace_kind == "allow"
    )
    examined_indices = tuple(ace.ace_index for ace in allow_aces)
    unexpected_indices = tuple(
        ace.ace_index
        for ace in allow_aces
        if not _same_sid(ace.trustee_sid, operator_sid)
    )
    reasons: list[DaclAllowlistReasonCode] = []
    if operator_sid.sid_string.casefold() in _BROAD_PRINCIPAL_SID_TEXTS:
        reasons.append("expected_operator_is_broad_principal")
    if not _same_sid(descriptor.owner_sid, operator_sid):
        reasons.append("descriptor_owner_mismatch")
    if unexpected_indices:
        reasons.append("unexpected_allow_trustee")

    if reasons:
        return _result(
            "outside_operator_only_allowlist",
            tuple(reasons),
            examined_allow_ace_indices=examined_indices,
            relevant_ace_indices=unexpected_indices,
            operator_sid=operator_sid,
            descriptor=descriptor,
            object_binding=object_binding,
        )
    return _result(
        "matches_operator_only_allowlist",
        ("operator_only_allowlist_match",),
        examined_allow_ace_indices=examined_indices,
        operator_sid=operator_sid,
        descriptor=descriptor,
        object_binding=object_binding,
    )
