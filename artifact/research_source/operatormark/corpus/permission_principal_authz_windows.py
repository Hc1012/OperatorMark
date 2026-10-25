"""Inactive evaluation of one supplied Windows token and copied descriptor.

The caller supplies an already-held borrowed token, its independently known
identity, copied descriptor evidence, and an independent object binding.  Importing
this module neither loads a DLL nor calls Windows.  Constructing
``CtypesAuthzBindings`` is the explicit opt-in to the real native boundary.

This narrow calculation does not establish a complete principal-category policy,
authenticated enrollment, an approved roster, or live file identity.  It does not
establish descriptor freshness, complete filesystem access, host qualification, or
frozen permission acceptance.  It never closes, duplicates, impersonates, or
adjusts the borrowed token and remains disconnected from production workflows.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import os
import re
import unicodedata
from typing import Callable, Literal, Protocol, cast

from operatormark.corpus.canonical import require_sha256, sha256_bytes
from operatormark.corpus import permission_observation_windows as _observation


__all__ = [
    "AUTHZ_RM_FLAG_NO_AUDIT",
    "ERROR_ACCESS_DENIED",
    "ERROR_SUCCESS",
    "MAXIMUM_ALLOWED",
    "AuthzNativeBindings",
    "CtypesAuthzBindings",
    "NativeAuthzAccessReply",
    "NativeTokenStatistics",
    "PermissionPrincipalAuthzFailure",
    "SuppliedTokenAccessRequest",
    "SuppliedTokenAccessResult",
    "SuppliedTokenIdentity",
    "evaluate_supplied_token_access",
]


ERROR_SUCCESS = 0
ERROR_ACCESS_DENIED = 5
TOKEN_STATISTICS_INFORMATION_CLASS = 10
AUTHZ_RM_FLAG_NO_AUDIT = 0x00000001
MAXIMUM_ALLOWED = 0x02000000
_DWORD_MAX = 0xFFFFFFFF
_GENERIC_ACCESS_MASK = 0xF0000000
_SE_SACL_PRESENT = 0x0010
_LUID_RE = re.compile(r"^[0-9a-f]{16}$")

_DWORD = ctypes.c_uint32
_LONG = ctypes.c_int32
_BOOL = ctypes.c_int32
_HANDLE = ctypes.c_void_p
_LARGE_INTEGER = ctypes.c_int64


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", _DWORD), ("HighPart", _LONG)]


class _TOKEN_STATISTICS(ctypes.Structure):
    _fields_ = [
        ("TokenId", _LUID),
        ("AuthenticationId", _LUID),
        ("ExpirationTime", _LARGE_INTEGER),
        ("TokenType", _DWORD),
        ("ImpersonationLevel", _DWORD),
        ("DynamicCharged", _DWORD),
        ("DynamicAvailable", _DWORD),
        ("GroupCount", _DWORD),
        ("PrivilegeCount", _DWORD),
        ("ModifiedId", _LUID),
    ]


class _AUTHZ_ACCESS_REQUEST(ctypes.Structure):
    _fields_ = [
        ("DesiredAccess", _DWORD),
        ("PrincipalSelfSid", ctypes.c_void_p),
        ("ObjectTypeList", ctypes.c_void_p),
        ("ObjectTypeListLength", _DWORD),
        ("OptionalArguments", ctypes.c_void_p),
    ]


class _AUTHZ_ACCESS_REPLY(ctypes.Structure):
    _fields_ = [
        ("ResultListLength", _DWORD),
        ("GrantedAccessMask", ctypes.POINTER(_DWORD)),
        ("SaclEvaluationResults", ctypes.POINTER(_DWORD)),
        ("Error", ctypes.POINTER(_DWORD)),
    ]


@dataclass(frozen=True, slots=True)
class SuppliedTokenIdentity:
    token_id_luid_hex: str
    authentication_id_luid_hex: str
    modified_id_luid_hex: str
    token_type: Literal["primary", "impersonation"]
    impersonation_level: Literal[
        "anonymous",
        "identification",
        "security_impersonation",
        "security_delegation",
    ] | None


@dataclass(frozen=True, slots=True)
class SuppliedTokenAccessRequest:
    borrowed_token_handle: int
    expected_token_identity: SuppliedTokenIdentity
    descriptor: _observation.NativeSecurityDescriptor
    object_binding: _observation.NativeAccessObjectBinding


@dataclass(frozen=True, slots=True)
class NativeTokenStatistics:
    token_id_luid_hex: str
    authentication_id_luid_hex: str
    modified_id_luid_hex: str
    token_type: Literal["primary", "impersonation"]
    impersonation_level: Literal[
        "anonymous",
        "identification",
        "security_impersonation",
        "security_delegation",
    ] | None
    expiration_time: int
    dynamic_charged: int
    dynamic_available: int
    group_count: int
    privilege_count: int


@dataclass(frozen=True, slots=True)
class NativeAuthzAccessReply:
    result_count: int
    granted_access_mask: int
    sacl_evaluation_results: int
    error_code: int


@dataclass(frozen=True, slots=True)
class PermissionPrincipalAuthzFailure:
    reason: Literal[
        "invalid_input",
        "descriptor_rejected",
        "native_call_failed",
        "invalid_native_output",
        "token_identity_mismatch",
        "token_changed",
        "per_result_error",
        "inconsistent_result",
        "cleanup_failed",
    ]
    stage: Literal[
        "input_validation",
        "token_before",
        "resource_manager",
        "context",
        "access_check",
        "token_after",
        "cleanup",
    ]
    operation: str
    native_error_code: int | None
    message: str


@dataclass(frozen=True, slots=True)
class SuppliedTokenAccessResult:
    access_check_api_status: Literal["not_called", "succeeded", "failed"]
    access_status: Literal["granted", "denied", "unavailable"]
    granted_access_mask: int | None
    per_result_error_code: int | None
    sacl_evaluation_results: int | None
    expected_token_identity: SuppliedTokenIdentity | None
    token_statistics_before: NativeTokenStatistics | None
    token_statistics_after: NativeTokenStatistics | None
    object_binding: _observation.NativeAccessObjectBinding | None
    descriptor_sha256: str | None
    primary_failure: PermissionPrincipalAuthzFailure | None
    cleanup_failures: tuple[PermissionPrincipalAuthzFailure, ...]
    requested_access_mask: int = field(init=False, default=MAXIMUM_ALLOWED)
    permission_acceptance_emitted: Literal[False] = field(init=False, default=False)


class AuthzNativeBindings(Protocol):
    def query_token_statistics(self, borrowed_token_handle: int) -> NativeTokenStatistics: ...

    def initialize_resource_manager(self) -> int: ...

    def initialize_context_from_token(
        self,
        borrowed_token_handle: int,
        resource_manager_handle: int,
    ) -> int: ...

    def access_check(
        self,
        context_handle: int,
        *,
        descriptor_bytes: bytes,
        desired_access: int,
    ) -> NativeAuthzAccessReply: ...

    def free_context(self, context_handle: int) -> None: ...

    def free_resource_manager(self, resource_manager_handle: int) -> None: ...


class _NativeCallFailure(RuntimeError):
    def __init__(self, operation: str, error_code: int | None = None) -> None:
        super().__init__(f"{operation} failed")
        self.operation = operation
        self.error_code = error_code


class _DescriptorRejected(ValueError):
    pass


def _exact_uint32(value: object, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _DWORD_MAX:
        raise ValueError(f"{name} must be an exact unsigned DWORD")
    return value


def _exact_luid(value: object, *, name: str) -> str:
    if type(value) is not str or _LUID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be exact eight-byte LUID text")
    return value


def _validate_identity(value: object) -> SuppliedTokenIdentity:
    if type(value) is not SuppliedTokenIdentity:
        raise ValueError("expected token identity has an unsupported representation")
    for name in (
        "token_id_luid_hex",
        "authentication_id_luid_hex",
        "modified_id_luid_hex",
    ):
        _exact_luid(getattr(value, name), name=name)
    if type(value.token_type) is not str or value.token_type not in {
        "primary",
        "impersonation",
    }:
        raise ValueError("token type is unsupported")
    allowed_levels = {
        "anonymous",
        "identification",
        "security_impersonation",
        "security_delegation",
    }
    if value.token_type == "primary":
        if value.impersonation_level is not None:
            raise ValueError("primary token has an invented impersonation level")
    elif (
        type(value.impersonation_level) is not str
        or value.impersonation_level not in allowed_levels
    ):
        raise ValueError("impersonation token level is unsupported")
    return value


def _validate_statistics(value: object) -> NativeTokenStatistics:
    if type(value) is not NativeTokenStatistics:
        raise ValueError("TOKEN_STATISTICS has an unsupported representation")
    _validate_identity(
        SuppliedTokenIdentity(
            token_id_luid_hex=value.token_id_luid_hex,
            authentication_id_luid_hex=value.authentication_id_luid_hex,
            modified_id_luid_hex=value.modified_id_luid_hex,
            token_type=value.token_type,
            impersonation_level=value.impersonation_level,
        )
    )
    if (
        type(value.expiration_time) is not int
        or not -(1 << 63) <= value.expiration_time < (1 << 63)
    ):
        raise ValueError("TOKEN_STATISTICS expiration is malformed")
    for name in (
        "dynamic_charged",
        "dynamic_available",
        "group_count",
        "privilege_count",
    ):
        _exact_uint32(getattr(value, name), name=name)
    return value


def _statistics_match_expected(
    statistics: NativeTokenStatistics,
    expected: SuppliedTokenIdentity,
) -> bool:
    return (
        statistics.token_id_luid_hex == expected.token_id_luid_hex
        and statistics.authentication_id_luid_hex
        == expected.authentication_id_luid_hex
        and statistics.modified_id_luid_hex == expected.modified_id_luid_hex
        and statistics.token_type == expected.token_type
        and statistics.impersonation_level == expected.impersonation_level
    )


def _require_exact_descriptor_types(
    descriptor: _observation.NativeSecurityDescriptor,
) -> None:
    if type(descriptor) is not _observation.NativeSecurityDescriptor:
        raise _DescriptorRejected("descriptor representation")
    for name in (
        "self_relative_descriptor_hex",
        "self_relative_descriptor_sha256",
        "control_hex",
    ):
        if type(getattr(descriptor, name)) is not str:
            raise _DescriptorRejected("descriptor scalar representation")
    for name in ("revision", "dacl_revision"):
        if type(getattr(descriptor, name)) is not int:
            raise _DescriptorRejected("descriptor integer representation")
    for name in (
        "dacl_present",
        "dacl_defaulted",
        "dacl_protected",
        "dacl_auto_inherited",
    ):
        if type(getattr(descriptor, name)) is not bool:
            raise _DescriptorRejected("descriptor Boolean representation")
    if type(descriptor.ordered_dacl_aces) is not tuple:
        raise _DescriptorRejected("descriptor ACE representation")
    for sid in (descriptor.owner_sid, descriptor.group_sid):
        if type(sid) is not _observation.NativeSid or any(
            type(getattr(sid, field_name)) is not str
            for field_name in ("sid_string", "sid_bytes_hex")
        ):
            raise _DescriptorRejected("descriptor SID representation")
    for ace in descriptor.ordered_dacl_aces:
        if type(ace) is not _observation.NativeAce:
            raise _DescriptorRejected("descriptor ACE representation")
        if type(ace.ace_index) is not int:
            raise _DescriptorRejected("descriptor ACE index representation")
        for name in (
            "ace_kind",
            "ace_type_hex",
            "ace_flags_hex",
            "access_mask_hex",
            "raw_ace_hex",
        ):
            if type(getattr(ace, name)) is not str:
                raise _DescriptorRejected("descriptor ACE scalar representation")
        trustee = ace.trustee_sid
        if type(trustee) is not _observation.NativeSid or any(
            type(getattr(trustee, field_name)) is not str
            for field_name in ("sid_string", "sid_bytes_hex")
        ):
            raise _DescriptorRejected("descriptor trustee representation")


def _validate_descriptor(
    descriptor: object,
) -> tuple[_observation.NativeSecurityDescriptor, bytes]:
    try:
        typed = cast(_observation.NativeSecurityDescriptor, descriptor)
        _require_exact_descriptor_types(typed)
        _observation._validate_descriptor(typed)
        raw = bytes.fromhex(typed.self_relative_descriptor_hex)
        control = int(typed.control_hex, 16)
        sacl_offset = int.from_bytes(raw[12:16], "little")
        if (
            typed.revision != 1
            or typed.dacl_revision not in {2, 4}
            or control & _SE_SACL_PRESENT
            or sacl_offset != 0
        ):
            raise _DescriptorRejected("unsupported descriptor form")
        for ace in typed.ordered_dacl_aces:
            if int(ace.access_mask_hex, 16) & _GENERIC_ACCESS_MASK:
                raise _DescriptorRejected("unmapped generic ACE mask")
    except _DescriptorRejected:
        raise
    except Exception as exc:
        raise _DescriptorRejected("descriptor validation") from exc
    return typed, raw


def _validate_binding(
    value: object,
    descriptor: _observation.NativeSecurityDescriptor,
) -> _observation.NativeAccessObjectBinding:
    if type(value) is not _observation.NativeAccessObjectBinding:
        raise ValueError("object binding has an unsupported representation")
    if type(value.object_role) is not str or type(value.object_kind) is not str:
        raise ValueError("object binding role or kind has the wrong type")
    if (value.object_role, value.object_kind) not in {
        ("subject", "file"),
        ("immediate_parent", "directory"),
    }:
        raise ValueError("object binding role and kind are inconsistent")
    if (
        type(value.canonical_volume_guid_path) is not str
        or not value.canonical_volume_guid_path
        or unicodedata.normalize("NFC", value.canonical_volume_guid_path)
        != value.canonical_volume_guid_path
    ):
        raise ValueError("object binding path is malformed")
    if type(value.identity) is not _observation.NativeFileIdentity:
        raise ValueError("object identity has an unsupported representation")
    for name in (
        "volume_serial_number_hex",
        "file_id_128_hex",
        "legacy_volume_serial_number_hex",
        "legacy_file_index_hex",
    ):
        if type(getattr(value.identity, name)) is not str:
            raise ValueError("object identity scalar has the wrong type")
    _observation._validate_identity(value.identity, name="object_binding.identity")
    if type(value.descriptor_sha256) is not str:
        raise ValueError("object binding descriptor hash has the wrong type")
    require_sha256(value.descriptor_sha256, name="object_binding.descriptor_sha256")
    if value.descriptor_sha256 != descriptor.self_relative_descriptor_sha256:
        raise ValueError("object binding does not bind the supplied descriptor")
    return value


def _validate_handle(value: object) -> int:
    pointer_max = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1
    if type(value) is not int or not 0 < value <= pointer_max:
        raise ValueError("borrowed token handle has an unsupported representation")
    return value


_BINDING_METHOD_NAMES = (
    "query_token_statistics",
    "initialize_resource_manager",
    "initialize_context_from_token",
    "access_check",
    "free_context",
    "free_resource_manager",
)


def _validate_bindings(value: object) -> dict[str, Callable[..., object]]:
    methods: dict[str, Callable[..., object]] = {}
    for name in _BINDING_METHOD_NAMES:
        candidate = getattr(value, name, None)
        if not callable(candidate):
            raise ValueError("native bindings are incomplete")
        methods[name] = candidate
    return methods


_FAILURE_MESSAGES = {
    "invalid_input": "supplied Authz input was rejected before native evaluation",
    "descriptor_rejected": "the copied descriptor is malformed or unsupported",
    "native_call_failed": "a required native operation failed",
    "invalid_native_output": "a native operation returned an unsupported result",
    "token_identity_mismatch": "the borrowed token did not match the independent identity",
    "token_changed": "the borrowed token statistics changed during evaluation",
    "per_result_error": "Authz returned a non-evidentiary per-result error",
    "inconsistent_result": "Authz returned internally inconsistent result values",
    "cleanup_failed": "an acquired Authz resource could not be released cleanly",
}


def _failure(
    reason: str,
    *,
    stage: str,
    operation: str,
    error_code: int | None = None,
) -> PermissionPrincipalAuthzFailure:
    safe_error = (
        error_code
        if type(error_code) is int and 0 < error_code <= _DWORD_MAX
        else None
    )
    return PermissionPrincipalAuthzFailure(
        reason=cast(object, reason),
        stage=cast(object, stage),
        operation=operation,
        native_error_code=safe_error,
        message=_FAILURE_MESSAGES[reason],
    )


def _safe_bindings_from_request(
    request: object,
) -> tuple[
    SuppliedTokenIdentity | None,
    _observation.NativeAccessObjectBinding | None,
    str | None,
]:
    if type(request) is not SuppliedTokenAccessRequest:
        return None, None, None
    identity = (
        request.expected_token_identity
        if type(request.expected_token_identity) is SuppliedTokenIdentity
        else None
    )
    binding = (
        request.object_binding
        if type(request.object_binding) is _observation.NativeAccessObjectBinding
        else None
    )
    descriptor_sha256 = None
    if (
        type(request.descriptor) is _observation.NativeSecurityDescriptor
        and type(request.descriptor.self_relative_descriptor_sha256) is str
        and re.fullmatch(
            r"[0-9a-f]{64}", request.descriptor.self_relative_descriptor_sha256
        )
    ):
        descriptor_sha256 = request.descriptor.self_relative_descriptor_sha256
    return identity, binding, descriptor_sha256


def _result(
    request: object,
    *,
    api_status: Literal["not_called", "succeeded", "failed"],
    access_status: Literal["granted", "denied", "unavailable"],
    before: NativeTokenStatistics | None,
    after: NativeTokenStatistics | None,
    mask: int | None = None,
    per_result_error: int | None = None,
    sacl_result: int | None = None,
    primary: PermissionPrincipalAuthzFailure | None = None,
    cleanup: tuple[PermissionPrincipalAuthzFailure, ...] = (),
) -> SuppliedTokenAccessResult:
    identity, binding, descriptor_sha256 = _safe_bindings_from_request(request)
    return SuppliedTokenAccessResult(
        access_check_api_status=api_status,
        access_status=access_status,
        granted_access_mask=mask,
        per_result_error_code=per_result_error,
        sacl_evaluation_results=sacl_result,
        expected_token_identity=identity,
        token_statistics_before=before,
        token_statistics_after=after,
        object_binding=binding,
        descriptor_sha256=descriptor_sha256,
        primary_failure=primary,
        cleanup_failures=cleanup,
    )


def _native_error(exc: Exception) -> int | None:
    return exc.error_code if type(exc) is _NativeCallFailure else None


def evaluate_supplied_token_access(
    request: SuppliedTokenAccessRequest,
    *,
    bindings: AuthzNativeBindings,
) -> SuppliedTokenAccessResult:
    """Evaluate MAXIMUM_ALLOWED without acquiring, closing, or altering the token."""

    try:
        if type(request) is not SuppliedTokenAccessRequest:
            raise ValueError("request has an unsupported representation")
        token_handle = _validate_handle(request.borrowed_token_handle)
        expected_identity = _validate_identity(request.expected_token_identity)
        descriptor, descriptor_bytes = _validate_descriptor(request.descriptor)
        _validate_binding(request.object_binding, descriptor)
        methods = _validate_bindings(bindings)
    except _DescriptorRejected:
        return _result(
            request,
            api_status="not_called",
            access_status="unavailable",
            before=None,
            after=None,
            primary=_failure(
                "descriptor_rejected",
                stage="input_validation",
                operation="validate_descriptor",
            ),
        )
    except Exception:
        return _result(
            request,
            api_status="not_called",
            access_status="unavailable",
            before=None,
            after=None,
            primary=_failure(
                "invalid_input",
                stage="input_validation",
                operation="validate_supplied_input",
            ),
        )

    before: NativeTokenStatistics | None = None
    after: NativeTokenStatistics | None = None
    primary: PermissionPrincipalAuthzFailure | None = None
    cleanup: list[PermissionPrincipalAuthzFailure] = []
    resource_manager: int | None = None
    context: int | None = None
    api_status: Literal["not_called", "succeeded", "failed"] = "not_called"
    candidate_status: Literal["granted", "denied"] | None = None
    candidate_mask: int | None = None
    candidate_error: int | None = None
    candidate_sacl: int | None = None

    try:
        try:
            raw_before = methods["query_token_statistics"](token_handle)
        except Exception as exc:
            primary = _failure(
                "native_call_failed",
                stage="token_before",
                operation="GetTokenInformation",
                error_code=_native_error(exc),
            )
        if primary is None:
            try:
                before = _validate_statistics(raw_before)
            except Exception:
                primary = _failure(
                    "invalid_native_output",
                    stage="token_before",
                    operation="GetTokenInformation",
                )
        if primary is None and not _statistics_match_expected(
            cast(NativeTokenStatistics, before), expected_identity
        ):
            primary = _failure(
                "token_identity_mismatch",
                stage="token_before",
                operation="compare_token_identity",
            )

        if primary is None:
            try:
                candidate_manager = methods["initialize_resource_manager"]()
            except Exception as exc:
                primary = _failure(
                    "native_call_failed",
                    stage="resource_manager",
                    operation="AuthzInitializeResourceManager",
                    error_code=_native_error(exc),
                )
            else:
                try:
                    resource_manager = _validate_handle(candidate_manager)
                except Exception:
                    primary = _failure(
                        "invalid_native_output",
                        stage="resource_manager",
                        operation="AuthzInitializeResourceManager",
                    )

        if primary is None:
            try:
                candidate_context = methods["initialize_context_from_token"](
                    token_handle,
                    cast(int, resource_manager),
                )
            except Exception as exc:
                primary = _failure(
                    "native_call_failed",
                    stage="context",
                    operation="AuthzInitializeContextFromToken",
                    error_code=_native_error(exc),
                )
            else:
                try:
                    context = _validate_handle(candidate_context)
                except Exception:
                    primary = _failure(
                        "invalid_native_output",
                        stage="context",
                        operation="AuthzInitializeContextFromToken",
                    )

        if primary is None:
            try:
                raw_reply = methods["access_check"](
                    cast(int, context),
                    descriptor_bytes=descriptor_bytes,
                    desired_access=MAXIMUM_ALLOWED,
                )
            except Exception as exc:
                api_status = "failed"
                primary = _failure(
                    "native_call_failed",
                    stage="access_check",
                    operation="AuthzAccessCheck",
                    error_code=_native_error(exc),
                )
            else:
                api_status = "succeeded"
                if type(raw_reply) is not NativeAuthzAccessReply:
                    primary = _failure(
                        "invalid_native_output",
                        stage="access_check",
                        operation="AuthzAccessCheck",
                    )
                else:
                    try:
                        result_count = _exact_uint32(
                            raw_reply.result_count, name="result_count"
                        )
                        mask = _exact_uint32(
                            raw_reply.granted_access_mask,
                            name="granted_access_mask",
                        )
                        sacl = _exact_uint32(
                            raw_reply.sacl_evaluation_results,
                            name="sacl_evaluation_results",
                        )
                        error = _exact_uint32(
                            raw_reply.error_code, name="error_code"
                        )
                    except Exception:
                        primary = _failure(
                            "invalid_native_output",
                            stage="access_check",
                            operation="AuthzAccessCheck",
                        )
                    else:
                        if result_count != 1 or sacl != 0:
                            primary = _failure(
                                "inconsistent_result",
                                stage="access_check",
                                operation="AuthzAccessCheck",
                            )
                        elif error == ERROR_SUCCESS:
                            candidate_status = "granted"
                            candidate_mask = mask
                            candidate_error = error
                            candidate_sacl = sacl
                        elif error == ERROR_ACCESS_DENIED and mask == 0:
                            candidate_status = "denied"
                            candidate_mask = mask
                            candidate_error = error
                            candidate_sacl = sacl
                        elif error == ERROR_ACCESS_DENIED:
                            primary = _failure(
                                "inconsistent_result",
                                stage="access_check",
                                operation="AuthzAccessCheck",
                                error_code=error,
                            )
                        else:
                            primary = _failure(
                                "per_result_error",
                                stage="access_check",
                                operation="AuthzAccessCheck",
                                error_code=error,
                            )

        if primary is None and candidate_status is not None:
            try:
                raw_after = methods["query_token_statistics"](token_handle)
            except Exception as exc:
                primary = _failure(
                    "native_call_failed",
                    stage="token_after",
                    operation="GetTokenInformation",
                    error_code=_native_error(exc),
                )
            else:
                try:
                    after = _validate_statistics(raw_after)
                except Exception:
                    primary = _failure(
                        "invalid_native_output",
                        stage="token_after",
                        operation="GetTokenInformation",
                    )
                else:
                    if not _statistics_match_expected(after, expected_identity):
                        primary = _failure(
                            "token_changed",
                            stage="token_after",
                            operation="compare_token_identity",
                        )
                    elif after != before:
                        primary = _failure(
                            "token_changed",
                            stage="token_after",
                            operation="compare_token_statistics",
                        )
    finally:
        if context is not None:
            try:
                released = methods["free_context"](context)
                if released is not None:
                    raise ValueError("unexpected cleanup return")
            except Exception as exc:
                cleanup.append(
                    _failure(
                        "cleanup_failed",
                        stage="cleanup",
                        operation="AuthzFreeContext",
                        error_code=_native_error(exc),
                    )
                )
        if resource_manager is not None:
            try:
                released = methods["free_resource_manager"](resource_manager)
                if released is not None:
                    raise ValueError("unexpected cleanup return")
            except Exception as exc:
                cleanup.append(
                    _failure(
                        "cleanup_failed",
                        stage="cleanup",
                        operation="AuthzFreeResourceManager",
                        error_code=_native_error(exc),
                    )
                )

    if primary is not None or cleanup:
        return _result(
            request,
            api_status=api_status,
            access_status="unavailable",
            before=before,
            after=after,
            primary=primary,
            cleanup=tuple(cleanup),
        )
    return _result(
        request,
        api_status=api_status,
        access_status=cast(Literal["granted", "denied"], candidate_status),
        before=before,
        after=after,
        mask=candidate_mask,
        per_result_error=candidate_error,
        sacl_result=candidate_sacl,
    )


class CtypesAuthzBindings:
    """Explicit, lazily opted-in ctypes implementation of the narrow binding."""

    def __init__(
        self,
        *,
        loader: Callable[..., object] | None = None,
    ) -> None:
        if loader is None:
            if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
                raise RuntimeError("Windows Authz bindings are unavailable")
            loader = cast(Callable[..., object], ctypes.WinDLL)
        self._advapi32 = loader("advapi32.dll", use_last_error=True)
        self._authz = loader("authz.dll", use_last_error=True)
        self._declare(self._advapi32, self._authz)

    @staticmethod
    def _declare(advapi32: object, authz: object) -> None:
        dword_pointer = ctypes.POINTER(_DWORD)
        handle_pointer = ctypes.POINTER(_HANDLE)
        declarations: tuple[tuple[object, str, list[object], object], ...] = (
            (
                advapi32,
                "GetTokenInformation",
                [_HANDLE, ctypes.c_int32, ctypes.c_void_p, _DWORD, dword_pointer],
                _BOOL,
            ),
            (
                authz,
                "AuthzInitializeResourceManager",
                [
                    _DWORD,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_wchar_p,
                    handle_pointer,
                ],
                _BOOL,
            ),
            (
                authz,
                "AuthzInitializeContextFromToken",
                [
                    _DWORD,
                    _HANDLE,
                    _HANDLE,
                    ctypes.c_void_p,
                    _LUID,
                    ctypes.c_void_p,
                    handle_pointer,
                ],
                _BOOL,
            ),
            (
                authz,
                "AuthzAccessCheck",
                [
                    _DWORD,
                    _HANDLE,
                    ctypes.POINTER(_AUTHZ_ACCESS_REQUEST),
                    _HANDLE,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    _DWORD,
                    ctypes.POINTER(_AUTHZ_ACCESS_REPLY),
                    handle_pointer,
                ],
                _BOOL,
            ),
            (authz, "AuthzFreeContext", [_HANDLE], _BOOL),
            (authz, "AuthzFreeResourceManager", [_HANDLE], _BOOL),
        )
        for library, name, argtypes, restype in declarations:
            function = getattr(library, name)
            function.argtypes = argtypes
            function.restype = restype

    @staticmethod
    def _last_error(operation: str) -> _NativeCallFailure:
        code = ctypes.get_last_error()
        if type(code) is int and 0 < code <= _DWORD_MAX:
            return _NativeCallFailure(operation, code)
        return _NativeCallFailure(operation)

    @staticmethod
    def _luid_hex(value: _LUID) -> str:
        high = ctypes.c_uint32(value.HighPart).value
        return f"{high:08x}{int(value.LowPart):08x}"

    def query_token_statistics(
        self, borrowed_token_handle: int
    ) -> NativeTokenStatistics:
        statistics = _TOKEN_STATISTICS()
        returned = _DWORD()
        ctypes.set_last_error(0)
        succeeded = self._advapi32.GetTokenInformation(
            _HANDLE(borrowed_token_handle),
            ctypes.c_int32(TOKEN_STATISTICS_INFORMATION_CLASS),
            ctypes.byref(statistics),
            _DWORD(ctypes.sizeof(statistics)),
            ctypes.byref(returned),
        )
        if not succeeded:
            raise self._last_error("GetTokenInformation")
        if int(returned.value) != ctypes.sizeof(statistics):
            raise _NativeCallFailure("GetTokenInformation")
        token_types = {1: "primary", 2: "impersonation"}
        levels = {
            0: "anonymous",
            1: "identification",
            2: "security_impersonation",
            3: "security_delegation",
        }
        token_type = token_types.get(int(statistics.TokenType))
        if token_type is None:
            raise _NativeCallFailure("GetTokenInformation")
        if token_type == "primary":
            impersonation_level = None
        else:
            impersonation_level = levels.get(int(statistics.ImpersonationLevel))
            if impersonation_level is None:
                raise _NativeCallFailure("GetTokenInformation")
        return NativeTokenStatistics(
            token_id_luid_hex=self._luid_hex(statistics.TokenId),
            authentication_id_luid_hex=self._luid_hex(
                statistics.AuthenticationId
            ),
            modified_id_luid_hex=self._luid_hex(statistics.ModifiedId),
            token_type=cast(object, token_type),
            impersonation_level=cast(object, impersonation_level),
            expiration_time=int(statistics.ExpirationTime),
            dynamic_charged=int(statistics.DynamicCharged),
            dynamic_available=int(statistics.DynamicAvailable),
            group_count=int(statistics.GroupCount),
            privilege_count=int(statistics.PrivilegeCount),
        )

    def initialize_resource_manager(self) -> int:
        resource_manager = _HANDLE()
        ctypes.set_last_error(0)
        succeeded = self._authz.AuthzInitializeResourceManager(
            _DWORD(AUTHZ_RM_FLAG_NO_AUDIT),
            None,
            None,
            None,
            None,
            ctypes.byref(resource_manager),
        )
        if not succeeded:
            raise self._last_error("AuthzInitializeResourceManager")
        if resource_manager.value is None:
            raise _NativeCallFailure("AuthzInitializeResourceManager")
        return int(resource_manager.value)

    def initialize_context_from_token(
        self,
        borrowed_token_handle: int,
        resource_manager_handle: int,
    ) -> int:
        context = _HANDLE()
        identifier = _LUID(0, 0)
        ctypes.set_last_error(0)
        succeeded = self._authz.AuthzInitializeContextFromToken(
            _DWORD(0),
            _HANDLE(borrowed_token_handle),
            _HANDLE(resource_manager_handle),
            None,
            identifier,
            None,
            ctypes.byref(context),
        )
        if not succeeded:
            raise self._last_error("AuthzInitializeContextFromToken")
        if context.value is None:
            raise _NativeCallFailure("AuthzInitializeContextFromToken")
        return int(context.value)

    def access_check(
        self,
        context_handle: int,
        *,
        descriptor_bytes: bytes,
        desired_access: int,
    ) -> NativeAuthzAccessReply:
        descriptor_buffer = (ctypes.c_ubyte * len(descriptor_bytes)).from_buffer_copy(
            descriptor_bytes
        )
        request = _AUTHZ_ACCESS_REQUEST(
            DesiredAccess=_DWORD(desired_access),
            PrincipalSelfSid=None,
            ObjectTypeList=None,
            ObjectTypeListLength=_DWORD(0),
            OptionalArguments=None,
        )
        granted = _DWORD()
        sacl = _DWORD()
        error = _DWORD()
        reply = _AUTHZ_ACCESS_REPLY(
            ResultListLength=_DWORD(1),
            GrantedAccessMask=ctypes.pointer(granted),
            SaclEvaluationResults=ctypes.pointer(sacl),
            Error=ctypes.pointer(error),
        )
        ctypes.set_last_error(0)
        succeeded = self._authz.AuthzAccessCheck(
            _DWORD(0),
            _HANDLE(context_handle),
            ctypes.byref(request),
            None,
            ctypes.cast(descriptor_buffer, ctypes.c_void_p),
            None,
            _DWORD(0),
            ctypes.byref(reply),
            None,
        )
        if not succeeded:
            raise self._last_error("AuthzAccessCheck")
        pointers_unchanged = (
            bool(reply.GrantedAccessMask)
            and bool(reply.SaclEvaluationResults)
            and bool(reply.Error)
            and ctypes.addressof(reply.GrantedAccessMask.contents)
            == ctypes.addressof(granted)
            and ctypes.addressof(reply.SaclEvaluationResults.contents)
            == ctypes.addressof(sacl)
            and ctypes.addressof(reply.Error.contents) == ctypes.addressof(error)
        )
        if not pointers_unchanged:
            raise _NativeCallFailure("AuthzAccessCheck")
        return NativeAuthzAccessReply(
            result_count=int(reply.ResultListLength),
            granted_access_mask=int(granted.value),
            sacl_evaluation_results=int(sacl.value),
            error_code=int(error.value),
        )

    def free_context(self, context_handle: int) -> None:
        ctypes.set_last_error(0)
        if not self._authz.AuthzFreeContext(_HANDLE(context_handle)):
            raise self._last_error("AuthzFreeContext")

    def free_resource_manager(self, resource_manager_handle: int) -> None:
        ctypes.set_last_error(0)
        if not self._authz.AuthzFreeResourceManager(
            _HANDLE(resource_manager_handle)
        ):
            raise self._last_error("AuthzFreeResourceManager")
