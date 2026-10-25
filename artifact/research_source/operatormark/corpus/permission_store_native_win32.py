"""Lazy Win32 primitives for a future confidential evidence store.

This module deliberately stops below ``ConfidentialEvidenceStore``.  It prepares
caller-selected security bytes and performs single, explicit native file
operations; it does not choose policy, authenticate a namespace, or turn native
success into retained evidence.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import hashlib
import re
import unicodedata
from typing import Literal, cast

from operatormark.corpus.permission_native_win32 import (
    DWORD_MAX,
    FILE_NAME_NORMALIZED,
    INVALID_HANDLE_VALUE,
    READ_CONTROL,
    SIGNED_LARGE_INTEGER_MAX,
    VOLUME_NAME_GUID,
    Win32ApiError,
    Win32LocalError,
    _CtypesWin32Bindings,
)


SDDL_REVISION_1 = 1
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
DELETE = 0x00010000
CREATE_NEW = 1
FILE_ATTRIBUTE_NORMAL = 0x00000080
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
FILE_FLAG_WRITE_THROUGH = 0x80000000
FILE_BEGIN = 0
FILE_RENAME_INFO_CLASS = 3
ERROR_FILE_EXISTS = 80
ERROR_ALREADY_EXISTS = 183

SE_DACL_PRESENT = 0x0004
SE_DACL_PROTECTED = 0x1000
SE_SELF_RELATIVE = 0x8000

_VOLUME_PATH = re.compile(
    r"^\\\\\?\\Volume\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}\\"
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_NATIVE_POINTER_MAX = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
_RESERVED_LEAVES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


class _SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", ctypes.c_uint32),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", wintypes.BOOL),
    ]


class _RENAME_CHOICE(ctypes.Union):
    _fields_ = [("ReplaceIfExists", ctypes.c_ubyte), ("Flags", ctypes.c_uint32)]


class _FILE_RENAME_INFO(ctypes.Structure):
    _anonymous_ = ("choice",)
    _fields_ = [
        ("choice", _RENAME_CHOICE),
        ("RootDirectory", wintypes.HANDLE),
        ("FileNameLength", ctypes.c_uint32),
        ("FileName", ctypes.c_uint16 * 1),
    ]


@dataclass(frozen=True, slots=True)
class PreparedCreateSecurity:
    """An owned-copy result; it contains no live native pointer."""

    sddl: str
    descriptor: bytes
    descriptor_sha256: str


@dataclass(frozen=True, slots=True)
class StoreNativeFailure:
    api_name: str
    classification: Literal["native_failure", "local_failure", "unexpected_exception"]
    backend_error_code: int | None
    reason_code: str


class Win32EvidenceStoreError(RuntimeError):
    """In-memory native failure evidence; never a serialized store schema."""

    def __init__(
        self,
        operation: str,
        phase: Literal["before_invocation", "native_failure", "after_invocation", "cleanup"],
        primary_failure: StoreNativeFailure,
        *,
        cleanup_failures: tuple[StoreNativeFailure, ...] = (),
        invoked: bool = False,
        returned: bool = False,
        last_confirmed_operation: str | None = None,
        returned_handle: int | None = None,
        handle_owned: bool = False,
        confirmed_transfer_count: int | None = None,
        unconfirmed_transfer_count: int | None = None,
        collision: bool = False,
        cleanup_invoked: bool = False,
        cleanup_returned: bool = False,
    ) -> None:
        super().__init__(f"{operation}: {primary_failure.reason_code}")
        self.operation = operation
        self.phase = phase
        self.primary_failure = primary_failure
        self.cleanup_failures = cleanup_failures
        self.invoked = invoked
        self.returned = returned
        self.last_confirmed_operation = last_confirmed_operation
        self.returned_handle = returned_handle
        self.handle_owned = handle_owned
        self.confirmed_transfer_count = confirmed_transfer_count
        self.unconfirmed_transfer_count = unconfirmed_transfer_count
        self.collision = collision
        self.cleanup_invoked = cleanup_invoked
        self.cleanup_returned = cleanup_returned

    @property
    def backend_error_code(self) -> int | None:
        return self.primary_failure.backend_error_code


def _local(api_name: str, reason: str) -> StoreNativeFailure:
    return StoreNativeFailure(api_name, "local_failure", None, reason)


def _unexpected(api_name: str) -> StoreNativeFailure:
    return StoreNativeFailure(api_name, "unexpected_exception", None, "unexpected_exception")


def _applicable_last_error(api_name: str) -> StoreNativeFailure:
    try:
        value = ctypes.get_last_error()
    except Exception:
        value = 0
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= DWORD_MAX:
        return StoreNativeFailure(api_name, "native_failure", value, "native_call_failed")
    return _local(api_name, "missing_native_error")


def _boundary_failure(error: Win32ApiError | Win32LocalError) -> StoreNativeFailure:
    if isinstance(error, Win32ApiError):
        return StoreNativeFailure(
            error.api_name, "native_failure", error.error_code, "native_call_failed"
        )
    return _local(error.api_name, "local_failure")


def _require_handle(handle: object, *, operation: str) -> int:
    if (
        isinstance(handle, bool)
        or not isinstance(handle, int)
        or handle <= 0
        or handle > _NATIVE_POINTER_MAX
        or handle == INVALID_HANDLE_VALUE
    ):
        raise Win32EvidenceStoreError(
            operation,
            "before_invocation",
            _local(operation, "invalid_handle"),
        )
    return handle


def _require_leaf(value: object) -> str:
    if type(value) is not str or not value or unicodedata.normalize("NFC", value) != value:
        raise ValueError("leaf is not a nonempty NFC string")
    if (
        "\x00" in value
        or "\\" in value
        or "/" in value
        or ":" in value
        or value in {".", ".."}
        or value.endswith((".", " "))
        or any(ord(character) < 32 for character in value)
        or value.split(".", 1)[0].upper() in _RESERVED_LEAVES
    ):
        raise ValueError("leaf can escape or ambiguously address the target namespace")
    return value


def _path_parts(value: object, *, allow_root: bool) -> tuple[str, re.Match[str]]:
    if type(value) is not str or "\x00" in value or "/" in value:
        raise ValueError("path is not a canonical volume-GUID path")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("path is not NFC")
    match = _VOLUME_PATH.match(value)
    if match is None:
        raise ValueError("path is not an absolute volume-GUID path")
    tail = value[match.end() :]
    if not tail:
        if allow_root:
            return cast(str, value), match
        raise ValueError("file path lacks a final component")
    if value.endswith("\\"):
        raise ValueError("non-root path has a trailing separator")
    for component in tail.split("\\"):
        _require_leaf(component)
    return cast(str, value), match


def _file_path(value: object) -> str:
    return _path_parts(value, allow_root=False)[0]


def _parent_path(value: object) -> str:
    return _path_parts(value, allow_root=True)[0]


def _immediate_parent(path: str) -> str:
    _, match = _path_parts(path, allow_root=False)
    tail = path[match.end() :]
    if "\\" not in tail:
        return path[: match.end()]
    return path.rsplit("\\", 1)[0]


def _joined_target(parent: str, leaf: str) -> str:
    return parent + leaf if parent.endswith("\\") else parent + "\\" + leaf


def _checked_transfer(offset: object, data: object) -> tuple[int, bytes]:
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= SIGNED_LARGE_INTEGER_MAX:
        raise ValueError("offset is outside signed LARGE_INTEGER")
    if type(data) is not bytes or len(data) > DWORD_MAX:
        raise ValueError("data is not immutable bytes within DWORD")
    if offset + len(data) > SIGNED_LARGE_INTEGER_MAX:
        raise ValueError("offset plus transfer exceeds signed LARGE_INTEGER")
    return offset, data


def _sid_extent(data: bytes, offset: int) -> int:
    if offset < 20 or offset + 8 > len(data) or data[offset] != 1:
        raise ValueError("security descriptor SID is invalid")
    length = 8 + 4 * data[offset + 1]
    if offset + length > len(data):
        raise ValueError("security descriptor SID is truncated")
    return length


def _validate_relative_descriptor(data: bytes) -> tuple[int, int, int, int]:
    if len(data) < 20 or data[0] != 1:
        raise ValueError("security descriptor header is invalid")
    control = int.from_bytes(data[2:4], "little")
    owner = int.from_bytes(data[4:8], "little")
    group = int.from_bytes(data[8:12], "little")
    dacl = int.from_bytes(data[16:20], "little")
    required = SE_SELF_RELATIVE | SE_DACL_PRESENT | SE_DACL_PROTECTED
    if control & required != required or not owner or not group or not dacl:
        raise ValueError("security descriptor lacks owner, group, or protected non-null DACL")
    _sid_extent(data, owner)
    _sid_extent(data, group)
    if dacl < 20 or dacl + 8 > len(data):
        raise ValueError("security descriptor DACL is outside the allocation")
    acl_size = int.from_bytes(data[dacl + 2 : dacl + 4], "little")
    if data[dacl] not in {2, 4} or acl_size < 8 or dacl + acl_size > len(data):
        raise ValueError("security descriptor DACL is malformed")
    return control, owner, group, dacl


class Win32EvidenceStoreBindings(_CtypesWin32Bindings):
    """Pinned read boundary plus narrowly named store-only Win32 primitives."""

    def __init__(self) -> None:
        super().__init__()
        self._store_declared = False

    @property
    def store_boundary_loaded(self) -> bool:
        return self.loaded and self._store_declared

    def _ensure_store_loaded(self) -> tuple[object, object]:
        kernel32, advapi32 = super()._ensure_loaded()
        if not self._store_declared:
            self._declare_store(kernel32, advapi32)
            self._store_declared = True
        return kernel32, advapi32

    @staticmethod
    def _declare_store(kernel32: object, advapi32: object) -> None:
        dword_p = ctypes.POINTER(ctypes.c_uint32)
        bool_p = ctypes.POINTER(wintypes.BOOL)
        void_p_p = ctypes.POINTER(ctypes.c_void_p)
        declarations: tuple[tuple[object, str, list[object], object], ...] = (
            (
                advapi32,
                "ConvertStringSecurityDescriptorToSecurityDescriptorW",
                [wintypes.LPCWSTR, ctypes.c_uint32, void_p_p, dword_p],
                wintypes.BOOL,
            ),
            (advapi32, "IsValidSecurityDescriptor", [ctypes.c_void_p], wintypes.BOOL),
            (
                advapi32,
                "GetSecurityDescriptorControl",
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16), dword_p],
                wintypes.BOOL,
            ),
            (
                advapi32,
                "GetSecurityDescriptorOwner",
                [ctypes.c_void_p, void_p_p, bool_p],
                wintypes.BOOL,
            ),
            (
                advapi32,
                "GetSecurityDescriptorGroup",
                [ctypes.c_void_p, void_p_p, bool_p],
                wintypes.BOOL,
            ),
            (
                advapi32,
                "GetSecurityDescriptorDacl",
                [ctypes.c_void_p, bool_p, void_p_p, bool_p],
                wintypes.BOOL,
            ),
            (
                kernel32,
                "WriteFile",
                [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_uint32, dword_p, ctypes.c_void_p],
                wintypes.BOOL,
            ),
            (kernel32, "FlushFileBuffers", [wintypes.HANDLE], wintypes.BOOL),
            (
                kernel32,
                "SetFileInformationByHandle",
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32],
                wintypes.BOOL,
            ),
        )
        for library, name, argtypes, restype in declarations:
            function = getattr(library, name)
            function.argtypes = argtypes
            function.restype = restype

    def prepare_create_security(
        self, sddl: object, expected_descriptor_sha256: object
    ) -> PreparedCreateSecurity:
        if type(sddl) is not str or not sddl or "\x00" in sddl:
            raise Win32EvidenceStoreError(
                "prepare_create_security", "before_invocation", _local("SDDL", "invalid_sddl")
            )
        if type(expected_descriptor_sha256) is not str or _HASH.fullmatch(expected_descriptor_sha256) is None:
            raise Win32EvidenceStoreError(
                "prepare_create_security",
                "before_invocation",
                _local("SHA256", "invalid_expected_descriptor_hash"),
            )
        try:
            kernel32, advapi32 = self._ensure_store_loaded()
        except (Win32ApiError, Win32LocalError) as exc:
            detail = (
                StoreNativeFailure(exc.api_name, "native_failure", exc.error_code, "native_call_failed")
                if isinstance(exc, Win32ApiError)
                else _local(exc.api_name, "local_failure")
            )
            raise Win32EvidenceStoreError("prepare_create_security", "before_invocation", detail) from exc
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "prepare_create_security", "before_invocation", _unexpected("WinDLL")
            ) from exc

        pointer = ctypes.c_void_p()
        size = ctypes.c_uint32()
        owned_pointer: int | None = None
        prepared: PreparedCreateSecurity | None = None
        primary: StoreNativeFailure | None = None
        cleanup_failures: tuple[StoreNativeFailure, ...] = ()
        phase: Literal["before_invocation", "native_failure", "after_invocation", "cleanup"] = "native_failure"
        failure_invoked = False
        failure_returned = False
        last_confirmed: str | None = None
        try:
            ctypes.set_last_error(0)
            try:
                converted = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(  # type: ignore[attr-defined]
                    sddl, SDDL_REVISION_1, ctypes.byref(pointer), ctypes.byref(size)
                )
            except Exception:
                primary = _unexpected("ConvertStringSecurityDescriptorToSecurityDescriptorW")
                phase = "after_invocation"
                failure_invoked = True
            else:
                if not converted:
                    primary = _applicable_last_error(
                        "ConvertStringSecurityDescriptorToSecurityDescriptorW"
                    )
                    failure_invoked = True
                    failure_returned = True
                elif pointer.value is None:
                    primary = _local(
                        "ConvertStringSecurityDescriptorToSecurityDescriptorW", "missing_allocation"
                    )
                    failure_invoked = True
                    failure_returned = True
                else:
                    owned_pointer = int(pointer.value)
                    last_confirmed = "ConvertStringSecurityDescriptorToSecurityDescriptorW"
                    if not 20 <= int(size.value) <= DWORD_MAX:
                        primary = _local(
                            "ConvertStringSecurityDescriptorToSecurityDescriptorW", "invalid_allocation_size"
                        )
                    else:
                        try:
                            copied = ctypes.string_at(owned_pointer, int(size.value))
                            control, owner_offset, group_offset, dacl_offset = _validate_relative_descriptor(copied)
                            start = owned_pointer
                            end = start + len(copied)
                            if end <= start:
                                raise ValueError("descriptor address extent overflow")
                        except ValueError as exc:
                            primary = _local("SecurityDescriptor", str(exc))
                        except Exception:
                            primary = _unexpected("SecurityDescriptor")
                            phase = "after_invocation"
                        if primary is None:
                            ctypes.set_last_error(0)
                            try:
                                valid = advapi32.IsValidSecurityDescriptor(  # type: ignore[attr-defined]
                                    ctypes.c_void_p(start)
                                )
                            except Exception:
                                primary = _unexpected("IsValidSecurityDescriptor")
                                phase = "after_invocation"
                                failure_invoked = True
                            else:
                                if not valid:
                                    primary = _local(
                                        "IsValidSecurityDescriptor", "invalid_security_descriptor"
                                    )
                                    failure_invoked = True
                                    failure_returned = True
                                else:
                                    last_confirmed = "IsValidSecurityDescriptor"
                        if primary is None:
                            native_control = ctypes.c_uint16()
                            revision = ctypes.c_uint32()
                            owner_pointer = ctypes.c_void_p()
                            group_pointer = ctypes.c_void_p()
                            dacl_pointer = ctypes.c_void_p()
                            owner_defaulted = wintypes.BOOL()
                            group_defaulted = wintypes.BOOL()
                            dacl_present = wintypes.BOOL()
                            dacl_defaulted = wintypes.BOOL()
                            queries = (
                                (
                                    "GetSecurityDescriptorControl",
                                    lambda: advapi32.GetSecurityDescriptorControl(  # type: ignore[attr-defined]
                                        ctypes.c_void_p(start), ctypes.byref(native_control), ctypes.byref(revision)
                                    ),
                                ),
                                (
                                    "GetSecurityDescriptorOwner",
                                    lambda: advapi32.GetSecurityDescriptorOwner(  # type: ignore[attr-defined]
                                        ctypes.c_void_p(start), ctypes.byref(owner_pointer), ctypes.byref(owner_defaulted)
                                    ),
                                ),
                                (
                                    "GetSecurityDescriptorGroup",
                                    lambda: advapi32.GetSecurityDescriptorGroup(  # type: ignore[attr-defined]
                                        ctypes.c_void_p(start), ctypes.byref(group_pointer), ctypes.byref(group_defaulted)
                                    ),
                                ),
                                (
                                    "GetSecurityDescriptorDacl",
                                    lambda: advapi32.GetSecurityDescriptorDacl(  # type: ignore[attr-defined]
                                        ctypes.c_void_p(start), ctypes.byref(dacl_present), ctypes.byref(dacl_pointer), ctypes.byref(dacl_defaulted)
                                    ),
                                ),
                            )
                            for api_name, query in queries:
                                ctypes.set_last_error(0)
                                try:
                                    query_result = query()
                                except Exception:
                                    primary = _unexpected(api_name)
                                    phase = "after_invocation"
                                    failure_invoked = True
                                    break
                                if not query_result:
                                    primary = _applicable_last_error(api_name)
                                    failure_invoked = True
                                    failure_returned = True
                                    break
                                last_confirmed = api_name
                        if primary is None:
                            try:
                                expected_pointers = (
                                    (owner_pointer.value, owner_offset),
                                    (group_pointer.value, group_offset),
                                    (dacl_pointer.value, dacl_offset),
                                )
                                if (
                                    native_control.value != control
                                    or revision.value != 1
                                    or not bool(dacl_present.value)
                                    or any(value is None or not start <= int(value) < end or int(value) != start + offset for value, offset in expected_pointers)
                                ):
                                    raise ValueError("native descriptor queries disagree with copied bytes")
                                digest = hashlib.sha256(copied).hexdigest()
                                if digest != expected_descriptor_sha256:
                                    raise ValueError("descriptor hash does not match independent expectation")
                                prepared = PreparedCreateSecurity(sddl, copied, digest)
                            except ValueError as exc:
                                primary = _local("SecurityDescriptor", str(exc))
                            except Exception:
                                primary = _unexpected("SecurityDescriptor")
                                phase = "after_invocation"
        finally:
            cleanup: StoreNativeFailure | None = None
            cleanup_invoked = False
            cleanup_returned = False
            if owned_pointer is not None:
                ctypes.set_last_error(0)
                cleanup_invoked = True
                try:
                    release_result = kernel32.LocalFree(ctypes.c_void_p(owned_pointer))  # type: ignore[attr-defined]
                except Exception:
                    cleanup = _unexpected("LocalFree")
                else:
                    cleanup_returned = True
                    if release_result:
                        cleanup = _applicable_last_error("LocalFree")
            if cleanup is not None:
                if primary is None:
                    primary = cleanup
                    phase = "cleanup"
                    failure_invoked = cleanup_invoked
                    failure_returned = cleanup_returned
                else:
                    cleanup_failures = (cleanup,)
        if primary is not None:
            raise Win32EvidenceStoreError(
                "prepare_create_security",
                phase,
                primary,
                cleanup_failures=cleanup_failures,
                invoked=failure_invoked,
                returned=failure_returned,
                last_confirmed_operation=last_confirmed,
                cleanup_invoked=cleanup_invoked,
                cleanup_returned=cleanup_returned,
            )
        if prepared is None:
            raise Win32EvidenceStoreError(
                "prepare_create_security", "after_invocation", _local("SecurityDescriptor", "missing_result"), invoked=True
            )
        return prepared

    def create_secured_exclusive(
        self, canonical_path: object, prepared_security: object
    ) -> int:
        try:
            path = _file_path(canonical_path)
            if type(prepared_security) is not PreparedCreateSecurity:
                raise ValueError("prepared security has the wrong type")
            prepared = cast(PreparedCreateSecurity, prepared_security)
            if hashlib.sha256(prepared.descriptor).hexdigest() != prepared.descriptor_sha256:
                raise ValueError("prepared descriptor hash changed")
            _validate_relative_descriptor(prepared.descriptor)
        except ValueError as exc:
            raise Win32EvidenceStoreError(
                "create_secured_exclusive", "before_invocation", _local("CreateFileW", str(exc))
            ) from exc
        try:
            kernel32, _ = self._ensure_store_loaded()
        except (Win32ApiError, Win32LocalError) as exc:
            raise Win32EvidenceStoreError(
                "create_secured_exclusive", "before_invocation", _boundary_failure(exc)
            ) from exc
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "create_secured_exclusive", "before_invocation", _unexpected("WinDLL")
            ) from exc
        try:
            descriptor = ctypes.create_string_buffer(prepared.descriptor, len(prepared.descriptor))
            attributes = _SECURITY_ATTRIBUTES(
                ctypes.sizeof(_SECURITY_ATTRIBUTES), ctypes.addressof(descriptor), 0
            )
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "create_secured_exclusive",
                "before_invocation",
                _local("CreateFileW", "security_buffer_allocation_failed"),
            ) from exc
        try:
            ctypes.set_last_error(0)
            raw = kernel32.CreateFileW(  # type: ignore[attr-defined]
                path,
                GENERIC_READ | GENERIC_WRITE | DELETE | READ_CONTROL,
                0,
                ctypes.byref(attributes),
                CREATE_NEW,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_WRITE_THROUGH,
                None,
            )
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "create_secured_exclusive",
                "after_invocation",
                _unexpected("CreateFileW"),
                invoked=True,
                returned=False,
            ) from exc
        try:
            value = ctypes.cast(raw, ctypes.c_void_p).value
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "create_secured_exclusive", "after_invocation", _local("CreateFileW", "invalid_return"), invoked=True, returned=True
            ) from exc
        if value is None or value == INVALID_HANDLE_VALUE:
            failure = _applicable_last_error("CreateFileW")
            raise Win32EvidenceStoreError(
                "create_secured_exclusive",
                "native_failure",
                failure,
                invoked=True,
                returned=True,
                collision=failure.backend_error_code in {ERROR_FILE_EXISTS, ERROR_ALREADY_EXISTS},
            )
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Win32EvidenceStoreError(
                "create_secured_exclusive", "after_invocation", _local("CreateFileW", "invalid_return"), invoked=True, returned=True
            )
        return value

    def write_once_at(self, handle: object, offset: object, data: object) -> int:
        raw_handle = _require_handle(handle, operation="write_once_at")
        try:
            checked_offset, payload = _checked_transfer(offset, data)
        except ValueError as exc:
            raise Win32EvidenceStoreError(
                "write_once_at", "before_invocation", _local("WriteFile", str(exc))
            ) from exc
        try:
            kernel32, _ = self._ensure_store_loaded()
        except (Win32ApiError, Win32LocalError) as exc:
            raise Win32EvidenceStoreError(
                "write_once_at", "before_invocation", _boundary_failure(exc)
            ) from exc
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "write_once_at", "before_invocation", _unexpected("WinDLL")
            ) from exc
        new_position = ctypes.c_int64()
        ctypes.set_last_error(0)
        try:
            sought = kernel32.SetFilePointerEx(  # type: ignore[attr-defined]
                wintypes.HANDLE(raw_handle), ctypes.c_int64(checked_offset), ctypes.byref(new_position), FILE_BEGIN
            )
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "write_once_at", "after_invocation", _unexpected("SetFilePointerEx"), invoked=True
            ) from exc
        if not sought:
            raise Win32EvidenceStoreError(
                "write_once_at", "native_failure", _applicable_last_error("SetFilePointerEx"), invoked=True, returned=True
            )
        if new_position.value != checked_offset:
            raise Win32EvidenceStoreError(
                "write_once_at",
                "after_invocation",
                _local("SetFilePointerEx", "seek_position_mismatch"),
                invoked=True,
                returned=True,
                last_confirmed_operation="SetFilePointerEx",
            )
        try:
            buffer = ctypes.create_string_buffer(payload, len(payload)) if payload else None
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "write_once_at",
                "after_invocation",
                _local("WriteFile", "buffer_allocation_failed"),
                invoked=False,
                returned=False,
                last_confirmed_operation="SetFilePointerEx",
            ) from exc
        transferred = ctypes.c_uint32()
        ctypes.set_last_error(0)
        try:
            wrote = kernel32.WriteFile(  # type: ignore[attr-defined]
                wintypes.HANDLE(raw_handle), buffer, len(payload), ctypes.byref(transferred), None
            )
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "write_once_at",
                "after_invocation",
                _unexpected("WriteFile"),
                invoked=True,
                last_confirmed_operation="SetFilePointerEx",
                unconfirmed_transfer_count=int(transferred.value),
            ) from exc
        count = int(transferred.value)
        if not wrote:
            raise Win32EvidenceStoreError(
                "write_once_at",
                "native_failure",
                _applicable_last_error("WriteFile"),
                invoked=True,
                returned=True,
                last_confirmed_operation="SetFilePointerEx",
                unconfirmed_transfer_count=count,
            )
        if count > len(payload):
            raise Win32EvidenceStoreError(
                "write_once_at",
                "after_invocation",
                _local("WriteFile", "invalid_transfer_count"),
                invoked=True,
                returned=True,
                last_confirmed_operation="SetFilePointerEx",
                unconfirmed_transfer_count=count,
            )
        return count

    def flush_file(self, handle: object) -> None:
        raw_handle = _require_handle(handle, operation="flush_file")
        try:
            kernel32, _ = self._ensure_store_loaded()
        except (Win32ApiError, Win32LocalError) as exc:
            raise Win32EvidenceStoreError(
                "flush_file", "before_invocation", _boundary_failure(exc)
            ) from exc
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "flush_file", "before_invocation", _unexpected("WinDLL")
            ) from exc
        try:
            ctypes.set_last_error(0)
            flushed = kernel32.FlushFileBuffers(wintypes.HANDLE(raw_handle))  # type: ignore[attr-defined]
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "flush_file", "after_invocation", _unexpected("FlushFileBuffers"), invoked=True
            ) from exc
        if not flushed:
            raise Win32EvidenceStoreError(
                "flush_file", "native_failure", _applicable_last_error("FlushFileBuffers"), invoked=True, returned=True
            )

    def rename_no_replace(
        self, handle: object, canonical_parent_path: object, final_leaf: object
    ) -> str:
        raw_handle = _require_handle(handle, operation="rename_no_replace")
        try:
            parent = _parent_path(canonical_parent_path)
            leaf = _require_leaf(final_leaf)
        except ValueError as exc:
            raise Win32EvidenceStoreError(
                "rename_no_replace", "before_invocation", _local("SetFileInformationByHandle", str(exc))
            ) from exc
        try:
            kernel32, _ = self._ensure_store_loaded()
        except (Win32ApiError, Win32LocalError) as exc:
            raise Win32EvidenceStoreError(
                "rename_no_replace", "before_invocation", _boundary_failure(exc)
            ) from exc
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "rename_no_replace", "before_invocation", _unexpected("WinDLL")
            ) from exc
        try:
            source = self.get_final_path(
                raw_handle, flags=FILE_NAME_NORMALIZED | VOLUME_NAME_GUID
            )
        except (Win32ApiError, Win32LocalError) as exc:
            detail = (
                StoreNativeFailure(exc.api_name, "native_failure", exc.error_code, "native_call_failed")
                if isinstance(exc, Win32ApiError)
                else _local(exc.api_name, "local_failure")
            )
            raise Win32EvidenceStoreError("rename_no_replace", "native_failure", detail, invoked=True) from exc
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "rename_no_replace", "after_invocation", _unexpected("GetFinalPathNameByHandleW"), invoked=True
            ) from exc
        try:
            observed_source = _file_path(source)
            if _immediate_parent(observed_source) != parent:
                raise ValueError("source parent differs from supplied canonical parent")
            target = _joined_target(parent, leaf)
            _file_path(target)
            encoded = target.encode("utf-16le")
            if not encoded or len(encoded) > DWORD_MAX:
                raise ValueError("rename target encoding exceeds DWORD")
            offset = _FILE_RENAME_INFO.FileName.offset
            total = max(ctypes.sizeof(_FILE_RENAME_INFO), offset + len(encoded))
            if total > DWORD_MAX or offset + len(encoded) > total:
                raise ValueError("rename buffer extent overflow")
            buffer = ctypes.create_string_buffer(total)
            view = _FILE_RENAME_INFO.from_buffer(buffer)
            view.Flags = 0
            view.RootDirectory = None
            view.FileNameLength = len(encoded)
            ctypes.memmove(ctypes.addressof(buffer) + offset, encoded, len(encoded))
        except ValueError as exc:
            raise Win32EvidenceStoreError(
                "rename_no_replace",
                "before_invocation",
                _local("SetFileInformationByHandle", str(exc)),
                invoked=False,
                returned=False,
                last_confirmed_operation="GetFinalPathNameByHandleW",
            ) from exc
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "rename_no_replace",
                "after_invocation",
                _local("SetFileInformationByHandle", "rename_buffer_preparation_failed"),
                invoked=False,
                returned=False,
                last_confirmed_operation="GetFinalPathNameByHandleW",
            ) from exc
        ctypes.set_last_error(0)
        try:
            renamed = kernel32.SetFileInformationByHandle(  # type: ignore[attr-defined]
                wintypes.HANDLE(raw_handle), FILE_RENAME_INFO_CLASS, buffer, total
            )
        except Exception as exc:
            raise Win32EvidenceStoreError(
                "rename_no_replace",
                "after_invocation",
                _unexpected("SetFileInformationByHandle"),
                invoked=True,
                last_confirmed_operation="GetFinalPathNameByHandleW",
            ) from exc
        if not renamed:
            failure = _applicable_last_error("SetFileInformationByHandle")
            raise Win32EvidenceStoreError(
                "rename_no_replace",
                "native_failure",
                failure,
                invoked=True,
                returned=True,
                last_confirmed_operation="GetFinalPathNameByHandleW",
                collision=failure.backend_error_code in {ERROR_FILE_EXISTS, ERROR_ALREADY_EXISTS},
            )
        return target


__all__ = [
    "PreparedCreateSecurity",
    "StoreNativeFailure",
    "Win32EvidenceStoreBindings",
    "Win32EvidenceStoreError",
]
