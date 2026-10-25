"""Concrete, injected Win32 backend for permission observation.

The adapter is inert at import time.  Its default ctypes boundary loads DLLs only
when an operation is invoked; tests can inject a byte-oriented low-level binding.
No function in this module changes permissions or enables privileges.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import hashlib
import os
import struct
import threading
import unicodedata
from typing import Literal, Protocol, cast

from operatormark.corpus import permission_principal_authz_windows as _principal_authz
from operatormark.corpus.permission_observation_windows import (
    MAXIMUM_ALLOWED_HEX,
    ComponentFailureItem,
    NativeAccessObjectBinding,
    NativeAce,
    NativeCallContext,
    NativeFileIdentity,
    NativeGenericMapping,
    NativeGroup,
    NativeOpenSpecification,
    NativePrivilege,
    NativeRawAccessCheckResult,
    NativeRawPrefix,
    NativeResourceBinding,
    NativeSecurityDescriptor,
    NativeSid,
    NativeStream,
    NativeTokenFacts,
    PermissionNativeCallError,
    PermissionObservationContractError,
    RawNativeObjectSnapshot,
    validate_native_call_context,
    _validate_token_facts,
)


# Access, share, creation and flag constants used by the frozen open profile.
FILE_READ_DATA = 0x0001
FILE_READ_ATTRIBUTES = 0x0080
READ_CONTROL = 0x00020000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000

# GetFinalPathNameByHandle flags.
FILE_NAME_NORMALIZED = 0x0
VOLUME_NAME_DOS = 0x0
VOLUME_NAME_GUID = 0x1

# FILE_INFO_BY_HANDLE_CLASS values used by this adapter.
FILE_STANDARD_INFO_CLASS = 1
FILE_NAME_INFO_CLASS = 2
FILE_STREAM_INFO_CLASS = 7
FILE_ATTRIBUTE_TAG_INFO_CLASS = 9
FILE_ID_INFO_CLASS = 18

# Security and token constants.
SE_FILE_OBJECT = 1
OWNER_SECURITY_INFORMATION = 0x00000001
GROUP_SECURITY_INFORMATION = 0x00000002
DACL_SECURITY_INFORMATION = 0x00000004
TOKEN_DUPLICATE = 0x0002
TOKEN_QUERY = 0x0008
SECURITY_IMPERSONATION = 2
TOKEN_IMPERSONATION = 2
ERROR_ACCESS_DENIED = 5
ERROR_INSUFFICIENT_BUFFER = 122
ERROR_MORE_DATA = 234
ERROR_NO_TOKEN = 1008

TOKEN_USER = 1
TOKEN_GROUPS = 2
TOKEN_PRIVILEGES = 3
TOKEN_OWNER = 4
TOKEN_PRIMARY_GROUP = 5
TOKEN_STATISTICS = 10
TOKEN_RESTRICTED_SIDS = 11
TOKEN_SESSION_ID = 12
TOKEN_ELEVATION_TYPE = 18
TOKEN_ELEVATION = 20
TOKEN_HAS_RESTRICTIONS = 21
TOKEN_VIRTUALIZATION_ALLOWED = 23
TOKEN_VIRTUALIZATION_ENABLED = 24
TOKEN_INTEGRITY_LEVEL = 25
TOKEN_UI_ACCESS = 26
TOKEN_IS_APP_CONTAINER = 29
TOKEN_CAPABILITIES = 30
TOKEN_USER_CLAIM_ATTRIBUTES = 33
TOKEN_DEVICE_CLAIM_ATTRIBUTES = 34
TOKEN_DEVICE_GROUPS = 37
TOKEN_RESTRICTED_DEVICE_GROUPS = 38

DWORD_MAX = 0xFFFFFFFF
SIGNED_LARGE_INTEGER_MAX = 0x7FFFFFFFFFFFFFFF
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class Win32ApiError(OSError):
    """Low-level Win32 failure with optional partially acquired ownership."""

    def __init__(
        self,
        api_name: str,
        error_code: int,
        *,
        returned_handle: int | None = None,
        returned_pointer: int | None = None,
        returned_size: int | None = None,
        query_buffers: tuple[Win32QueryBuffer, ...] = (),
    ) -> None:
        super().__init__(error_code, api_name)
        self.api_name = api_name
        self.error_code = error_code
        self.returned_handle = returned_handle
        self.returned_pointer = returned_pointer
        self.returned_size = returned_size
        self.query_buffers = query_buffers


class Win32LocalError(RuntimeError):
    """A local/ABI inconsistency with no applicable Win32 error code."""

    def __init__(
        self,
        api_name: str,
        *,
        returned_handle: int | None = None,
        returned_pointer: int | None = None,
        returned_size: int | None = None,
        query_buffers: tuple[Win32QueryBuffer, ...] = (),
    ) -> None:
        super().__init__(api_name)
        self.api_name = api_name
        self.returned_handle = returned_handle
        self.returned_pointer = returned_pointer
        self.returned_size = returned_size
        self.query_buffers = query_buffers


class Win32ReleaseError(RuntimeError):
    """A release failed but the API exposes no reliable DWORD error."""


class HeldTokenAuthzValidationError(ValueError):
    """Redacted rejection of a malformed held-token Authz adapter input."""


@dataclass(frozen=True, slots=True)
class Win32QueryBuffer:
    """Copied variable-length native bytes plus their former address domain."""

    data: bytes
    base_address: int
    pointer_size: int
    capacity_bytes: int | None = None
    reported_size_bytes: int | None = None
    complete: bool = True


@dataclass(frozen=True, slots=True)
class Win32SecurityAllocation:
    pointer: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Win32AccessCheckCallResult:
    api_succeeded: bool
    access_granted: bool | None
    granted_access: int | None
    error_code: int | None
    privilege_set: Win32QueryBuffer | None


class Win32LowLevelBindings(Protocol):
    """Small injectable boundary around the documented Win32 calls."""

    def create_file(
        self,
        path: str,
        *,
        desired_access: int,
        share_mode: int,
        creation_disposition: int,
        flags_and_attributes: int,
    ) -> int: ...

    def close_handle(self, handle: int) -> None: ...

    def get_final_path(self, handle: int, *, flags: int) -> str: ...

    def query_file_information(
        self,
        handle: int,
        *,
        information_class: int,
        minimum_size: int,
        variable_length: bool,
    ) -> Win32QueryBuffer: ...

    def get_legacy_file_information(self, handle: int) -> bytes: ...

    def read_file_at(self, handle: int, *, offset: int, requested_bytes: int) -> bytes: ...

    def get_security_info(self, handle: int, *, security_information: int) -> Win32SecurityAllocation: ...

    def read_memory(self, pointer: int, size_bytes: int) -> bytes: ...

    def local_free(self, pointer: int) -> None: ...

    def open_thread_token(self, *, desired_access: int, open_as_self: bool) -> int: ...

    def open_process_token(self, *, desired_access: int) -> int: ...

    def duplicate_token(
        self,
        token: int,
        *,
        desired_access: int,
        impersonation_level: int,
        token_type: int,
    ) -> int: ...

    def query_token_information(self, token: int, *, information_class: int) -> Win32QueryBuffer: ...

    def lookup_privilege_name(self, *, luid_low: int, luid_high: int) -> str: ...

    def access_check(
        self,
        *,
        descriptor: bytes,
        token: int,
        desired_access: int,
        generic_mapping: tuple[int, int, int, int],
    ) -> Win32AccessCheckCallResult: ...


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]


class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("dwFileAttributes", ctypes.c_uint32),
        ("ftCreationTime", _FILETIME),
        ("ftLastAccessTime", _FILETIME),
        ("ftLastWriteTime", _FILETIME),
        ("dwVolumeSerialNumber", ctypes.c_uint32),
        ("nFileSizeHigh", ctypes.c_uint32),
        ("nFileSizeLow", ctypes.c_uint32),
        ("nNumberOfLinks", ctypes.c_uint32),
        ("nFileIndexHigh", ctypes.c_uint32),
        ("nFileIndexLow", ctypes.c_uint32),
    ]


class _FILE_STANDARD_INFO(ctypes.Structure):
    _fields_ = [
        ("AllocationSize", ctypes.c_int64),
        ("EndOfFile", ctypes.c_int64),
        ("NumberOfLinks", ctypes.c_uint32),
        ("DeletePending", ctypes.c_ubyte),
        ("Directory", ctypes.c_ubyte),
    ]


class _GENERIC_MAPPING(ctypes.Structure):
    _fields_ = [
        ("GenericRead", ctypes.c_uint32),
        ("GenericWrite", ctypes.c_uint32),
        ("GenericExecute", ctypes.c_uint32),
        ("GenericAll", ctypes.c_uint32),
    ]


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _CtypesWin32Bindings:
    """Lazy ctypes implementation of :class:`Win32LowLevelBindings`."""

    def __init__(self) -> None:
        self._kernel32: object | None = None
        self._advapi32: object | None = None

    @property
    def loaded(self) -> bool:
        return self._kernel32 is not None

    def _ensure_loaded(self) -> tuple[object, object]:
        if self._kernel32 is not None and self._advapi32 is not None:
            return self._kernel32, self._advapi32
        if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
            raise Win32LocalError("WinDLL")
        loader = cast(object, getattr(ctypes, "WinDLL"))
        kernel32 = loader("kernel32.dll", use_last_error=True)  # type: ignore[operator]
        advapi32 = loader("advapi32.dll", use_last_error=True)  # type: ignore[operator]
        self._declare(kernel32, advapi32)
        self._kernel32 = kernel32
        self._advapi32 = advapi32
        return kernel32, advapi32

    @staticmethod
    def _declare(kernel32: object, advapi32: object) -> None:
        handle = wintypes.HANDLE
        dword_p = ctypes.POINTER(ctypes.c_uint32)
        bool_p = ctypes.POINTER(wintypes.BOOL)
        void_p_p = ctypes.POINTER(ctypes.c_void_p)

        declarations: tuple[tuple[object, str, list[object], object], ...] = (
            (
                kernel32,
                "CreateFileW",
                [wintypes.LPCWSTR, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                 ctypes.c_uint32, ctypes.c_uint32, handle],
                handle,
            ),
            (kernel32, "CloseHandle", [handle], wintypes.BOOL),
            (
                kernel32,
                "GetFinalPathNameByHandleW",
                [handle, wintypes.LPWSTR, ctypes.c_uint32, ctypes.c_uint32],
                ctypes.c_uint32,
            ),
            (
                kernel32,
                "GetFileInformationByHandleEx",
                [handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32],
                wintypes.BOOL,
            ),
            (
                kernel32,
                "GetFileInformationByHandle",
                [handle, ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION)],
                wintypes.BOOL,
            ),
            (
                kernel32,
                "SetFilePointerEx",
                [handle, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.c_uint32],
                wintypes.BOOL,
            ),
            (
                kernel32,
                "ReadFile",
                [handle, ctypes.c_void_p, ctypes.c_uint32, dword_p, ctypes.c_void_p],
                wintypes.BOOL,
            ),
            (kernel32, "GetCurrentThread", [], handle),
            (kernel32, "GetCurrentProcess", [], handle),
            (kernel32, "LocalFree", [ctypes.c_void_p], ctypes.c_void_p),
            (
                advapi32,
                "GetSecurityInfo",
                [handle, ctypes.c_int, ctypes.c_uint32, void_p_p, void_p_p, void_p_p,
                 void_p_p, void_p_p],
                ctypes.c_uint32,
            ),
            (advapi32, "GetSecurityDescriptorLength", [ctypes.c_void_p], ctypes.c_uint32),
            (
                advapi32,
                "OpenThreadToken",
                [handle, ctypes.c_uint32, wintypes.BOOL, ctypes.POINTER(handle)],
                wintypes.BOOL,
            ),
            (
                advapi32,
                "OpenProcessToken",
                [handle, ctypes.c_uint32, ctypes.POINTER(handle)],
                wintypes.BOOL,
            ),
            (
                advapi32,
                "DuplicateTokenEx",
                [handle, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                 ctypes.POINTER(handle)],
                wintypes.BOOL,
            ),
            (
                advapi32,
                "GetTokenInformation",
                [handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, dword_p],
                wintypes.BOOL,
            ),
            (
                advapi32,
                "LookupPrivilegeNameW",
                [wintypes.LPCWSTR, ctypes.POINTER(_LUID), wintypes.LPWSTR, dword_p],
                wintypes.BOOL,
            ),
            (
                advapi32,
                "AccessCheck",
                [ctypes.c_void_p, handle, ctypes.c_uint32, ctypes.POINTER(_GENERIC_MAPPING),
                 ctypes.c_void_p, dword_p, dword_p, bool_p],
                wintypes.BOOL,
            ),
        )
        for library, name, argtypes, restype in declarations:
            function = getattr(library, name)
            function.argtypes = argtypes
            function.restype = restype

    @staticmethod
    def _last_error(api_name: str) -> Win32ApiError | Win32LocalError:
        code = ctypes.get_last_error()
        if 0 < code <= DWORD_MAX:
            return Win32ApiError(api_name, code)
        return Win32LocalError(api_name)

    @staticmethod
    def _native_buffer(api_name: str, capacity: int) -> ctypes.Array[ctypes.c_char]:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or not 0 < capacity <= DWORD_MAX
        ):
            raise Win32LocalError(api_name)
        try:
            return ctypes.create_string_buffer(capacity)
        except Exception as exc:
            raise Win32LocalError(api_name) from exc

    def create_file(
        self,
        path: str,
        *,
        desired_access: int,
        share_mode: int,
        creation_disposition: int,
        flags_and_attributes: int,
    ) -> int:
        kernel32, _ = self._ensure_loaded()
        ctypes.set_last_error(0)
        raw = kernel32.CreateFileW(  # type: ignore[attr-defined]
            path,
            desired_access,
            share_mode,
            None,
            creation_disposition,
            flags_and_attributes,
            None,
        )
        value = ctypes.cast(raw, ctypes.c_void_p).value
        if value is None or value == INVALID_HANDLE_VALUE:
            raise self._last_error("CreateFileW")
        return value

    def close_handle(self, handle: int) -> None:
        kernel32, _ = self._ensure_loaded()
        ctypes.set_last_error(0)
        if not kernel32.CloseHandle(wintypes.HANDLE(handle)):  # type: ignore[attr-defined]
            raise self._last_error("CloseHandle")

    def get_final_path(self, handle: int, *, flags: int) -> str:
        kernel32, _ = self._ensure_loaded()
        capacity = 512
        while True:
            if not 0 < capacity <= DWORD_MAX:
                raise Win32LocalError("GetFinalPathNameByHandleW")
            try:
                buffer = ctypes.create_unicode_buffer(capacity)
            except Exception as exc:
                raise Win32LocalError("GetFinalPathNameByHandleW") from exc
            ctypes.set_last_error(0)
            length = kernel32.GetFinalPathNameByHandleW(  # type: ignore[attr-defined]
                wintypes.HANDLE(handle), buffer, capacity, flags
            )
            if length == 0:
                raise self._last_error("GetFinalPathNameByHandleW")
            if length < capacity:
                return buffer.value
            if length >= DWORD_MAX:
                raise Win32LocalError("GetFinalPathNameByHandleW")
            capacity = length + 1

    def query_file_information(
        self,
        handle: int,
        *,
        information_class: int,
        minimum_size: int,
        variable_length: bool,
    ) -> Win32QueryBuffer:
        kernel32, _ = self._ensure_loaded()
        if (
            isinstance(minimum_size, bool)
            or not isinstance(minimum_size, int)
            or not 0 < minimum_size <= DWORD_MAX
        ):
            raise Win32LocalError("GetFileInformationByHandleEx")
        capacity = max(minimum_size, 256 if variable_length else minimum_size)
        while True:
            buffer = self._native_buffer("GetFileInformationByHandleEx", capacity)
            ctypes.set_last_error(0)
            ok = kernel32.GetFileInformationByHandleEx(  # type: ignore[attr-defined]
                wintypes.HANDLE(handle), information_class, buffer, capacity
            )
            if ok:
                return Win32QueryBuffer(
                    ctypes.string_at(buffer, capacity),
                    ctypes.addressof(buffer),
                    ctypes.sizeof(ctypes.c_void_p),
                    capacity_bytes=capacity,
                    reported_size_bytes=None,
                    complete=True,
                )
            error = ctypes.get_last_error()
            if variable_length and error in {ERROR_INSUFFICIENT_BUFFER, ERROR_MORE_DATA}:
                if capacity == DWORD_MAX:
                    raise Win32ApiError("GetFileInformationByHandleEx", error)
                capacity = min(DWORD_MAX, capacity * 2)
                continue
            if 0 < error <= DWORD_MAX:
                raise Win32ApiError("GetFileInformationByHandleEx", error)
            raise Win32LocalError("GetFileInformationByHandleEx")

    def get_legacy_file_information(self, handle: int) -> bytes:
        kernel32, _ = self._ensure_loaded()
        value = _BY_HANDLE_FILE_INFORMATION()
        ctypes.set_last_error(0)
        if not kernel32.GetFileInformationByHandle(  # type: ignore[attr-defined]
            wintypes.HANDLE(handle), ctypes.byref(value)
        ):
            raise self._last_error("GetFileInformationByHandle")
        return ctypes.string_at(ctypes.byref(value), ctypes.sizeof(value))

    def read_file_at(self, handle: int, *, offset: int, requested_bytes: int) -> bytes:
        kernel32, _ = self._ensure_loaded()
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset <= SIGNED_LARGE_INTEGER_MAX
            or isinstance(requested_bytes, bool)
            or not isinstance(requested_bytes, int)
            or not 0 < requested_bytes <= DWORD_MAX
            or offset + requested_bytes > SIGNED_LARGE_INTEGER_MAX
        ):
            raise Win32LocalError("ReadFile")
        new_position = ctypes.c_int64()
        ctypes.set_last_error(0)
        if not kernel32.SetFilePointerEx(  # type: ignore[attr-defined]
            wintypes.HANDLE(handle), ctypes.c_int64(offset), ctypes.byref(new_position), 0
        ):
            raise self._last_error("SetFilePointerEx")
        if new_position.value != offset:
            raise Win32LocalError("SetFilePointerEx")
        buffer = self._native_buffer("ReadFile", requested_bytes)
        transferred = ctypes.c_uint32()
        ctypes.set_last_error(0)
        if not kernel32.ReadFile(  # type: ignore[attr-defined]
            wintypes.HANDLE(handle), buffer, requested_bytes, ctypes.byref(transferred), None
        ):
            raise self._last_error("ReadFile")
        if transferred.value > requested_bytes:
            raise Win32LocalError("ReadFile")
        return ctypes.string_at(buffer, transferred.value)

    def get_security_info(
        self, handle: int, *, security_information: int
    ) -> Win32SecurityAllocation:
        _, advapi32 = self._ensure_loaded()
        descriptor = ctypes.c_void_p()
        result = advapi32.GetSecurityInfo(  # type: ignore[attr-defined]
            wintypes.HANDLE(handle),
            SE_FILE_OBJECT,
            security_information,
            None,
            None,
            None,
            None,
            ctypes.byref(descriptor),
        )
        pointer = descriptor.value
        if result != 0:
            if 0 < int(result) <= DWORD_MAX:
                raise Win32ApiError("GetSecurityInfo", int(result))
            raise Win32LocalError("GetSecurityInfo")
        if not pointer:
            raise Win32LocalError("GetSecurityInfo")
        size = advapi32.GetSecurityDescriptorLength(ctypes.c_void_p(pointer))  # type: ignore[attr-defined]
        if size == 0:
            raise Win32LocalError(
                "GetSecurityDescriptorLength", returned_pointer=pointer
            )
        return Win32SecurityAllocation(pointer, size)

    def read_memory(self, pointer: int, size_bytes: int) -> bytes:
        return ctypes.string_at(pointer, size_bytes)

    def local_free(self, pointer: int) -> None:
        kernel32, _ = self._ensure_loaded()
        if kernel32.LocalFree(ctypes.c_void_p(pointer)):  # type: ignore[attr-defined]
            raise Win32ReleaseError("LocalFree returned the allocation")

    def open_thread_token(self, *, desired_access: int, open_as_self: bool) -> int:
        kernel32, advapi32 = self._ensure_loaded()
        token = wintypes.HANDLE()
        ctypes.set_last_error(0)
        ok = advapi32.OpenThreadToken(  # type: ignore[attr-defined]
            kernel32.GetCurrentThread(), desired_access, bool(open_as_self), ctypes.byref(token)  # type: ignore[attr-defined]
        )
        value = ctypes.cast(token, ctypes.c_void_p).value
        if not ok:
            raise self._last_error("OpenThreadToken")
        if not value:
            raise Win32LocalError("OpenThreadToken")
        return value

    def open_process_token(self, *, desired_access: int) -> int:
        kernel32, advapi32 = self._ensure_loaded()
        token = wintypes.HANDLE()
        ctypes.set_last_error(0)
        ok = advapi32.OpenProcessToken(  # type: ignore[attr-defined]
            kernel32.GetCurrentProcess(), desired_access, ctypes.byref(token)  # type: ignore[attr-defined]
        )
        value = ctypes.cast(token, ctypes.c_void_p).value
        if not ok:
            raise self._last_error("OpenProcessToken")
        if not value:
            raise Win32LocalError("OpenProcessToken")
        return value

    def duplicate_token(
        self,
        token: int,
        *,
        desired_access: int,
        impersonation_level: int,
        token_type: int,
    ) -> int:
        _, advapi32 = self._ensure_loaded()
        duplicate = wintypes.HANDLE()
        ctypes.set_last_error(0)
        ok = advapi32.DuplicateTokenEx(  # type: ignore[attr-defined]
            wintypes.HANDLE(token),
            desired_access,
            None,
            impersonation_level,
            token_type,
            ctypes.byref(duplicate),
        )
        value = ctypes.cast(duplicate, ctypes.c_void_p).value
        if not ok:
            raise self._last_error("DuplicateTokenEx")
        if not value:
            raise Win32LocalError("DuplicateTokenEx")
        return value

    def query_token_information(
        self, token: int, *, information_class: int
    ) -> Win32QueryBuffer:
        _, advapi32 = self._ensure_loaded()
        needed = ctypes.c_uint32()
        ctypes.set_last_error(0)
        first = advapi32.GetTokenInformation(  # type: ignore[attr-defined]
            wintypes.HANDLE(token), information_class, None, 0, ctypes.byref(needed)
        )
        first_error = ctypes.get_last_error()
        if first:
            raise Win32LocalError("GetTokenInformation")
        if first_error != ERROR_INSUFFICIENT_BUFFER:
            if 0 < first_error <= DWORD_MAX:
                raise Win32ApiError("GetTokenInformation", first_error)
            raise Win32LocalError("GetTokenInformation")
        if needed.value == 0:
            raise Win32LocalError("GetTokenInformation")
        capacity = int(needed.value)
        buffer = self._native_buffer("GetTokenInformation", capacity)
        returned = ctypes.c_uint32()
        ctypes.set_last_error(0)
        succeeded = advapi32.GetTokenInformation(  # type: ignore[attr-defined]
            wintypes.HANDLE(token),
            information_class,
            buffer,
            capacity,
            ctypes.byref(returned),
        )
        reported = int(returned.value)
        if not succeeded:
            copied: tuple[Win32QueryBuffer, ...] = ()
            if reported > 0:
                captured = min(reported, capacity)
                copied = (
                    Win32QueryBuffer(
                        ctypes.string_at(buffer, captured),
                        ctypes.addressof(buffer),
                        ctypes.sizeof(ctypes.c_void_p),
                        capacity_bytes=capacity,
                        reported_size_bytes=reported,
                        complete=False,
                    ),
                )
            error = ctypes.get_last_error()
            if 0 < error <= DWORD_MAX:
                raise Win32ApiError(
                    "GetTokenInformation", error, query_buffers=copied
                )
            raise Win32LocalError("GetTokenInformation", query_buffers=copied)
        if reported == 0:
            raise Win32LocalError("GetTokenInformation")
        if reported > capacity:
            copied = Win32QueryBuffer(
                ctypes.string_at(buffer, capacity),
                ctypes.addressof(buffer),
                ctypes.sizeof(ctypes.c_void_p),
                capacity_bytes=capacity,
                reported_size_bytes=reported,
                complete=False,
            )
            raise Win32LocalError(
                "GetTokenInformation", query_buffers=(copied,)
            )
        return Win32QueryBuffer(
            ctypes.string_at(buffer, reported),
            ctypes.addressof(buffer),
            ctypes.sizeof(ctypes.c_void_p),
            capacity_bytes=capacity,
            reported_size_bytes=reported,
            complete=True,
        )

    def lookup_privilege_name(self, *, luid_low: int, luid_high: int) -> str:
        _, advapi32 = self._ensure_loaded()
        luid = _LUID(luid_low, ctypes.c_int32(luid_high).value)
        size = ctypes.c_uint32()
        ctypes.set_last_error(0)
        advapi32.LookupPrivilegeNameW(  # type: ignore[attr-defined]
            None, ctypes.byref(luid), None, ctypes.byref(size)
        )
        error = ctypes.get_last_error()
        if error != ERROR_INSUFFICIENT_BUFFER:
            if 0 < error <= DWORD_MAX:
                raise Win32ApiError("LookupPrivilegeNameW", error)
            raise Win32LocalError("LookupPrivilegeNameW")
        if size.value == 0 or size.value == DWORD_MAX:
            raise Win32LocalError("LookupPrivilegeNameW")
        try:
            buffer = ctypes.create_unicode_buffer(size.value + 1)
        except Exception as exc:
            raise Win32LocalError("LookupPrivilegeNameW") from exc
        capacity = ctypes.c_uint32(size.value + 1)
        ctypes.set_last_error(0)
        if not advapi32.LookupPrivilegeNameW(  # type: ignore[attr-defined]
            None, ctypes.byref(luid), buffer, ctypes.byref(capacity)
        ):
            raise self._last_error("LookupPrivilegeNameW")
        return buffer.value

    def access_check(
        self,
        *,
        descriptor: bytes,
        token: int,
        desired_access: int,
        generic_mapping: tuple[int, int, int, int],
    ) -> Win32AccessCheckCallResult:
        _, advapi32 = self._ensure_loaded()
        descriptor_buffer = ctypes.create_string_buffer(descriptor, len(descriptor))
        mapping = _GENERIC_MAPPING(*generic_mapping)
        capacity = 256
        while True:
            privileges = self._native_buffer("AccessCheck", capacity)
            privilege_size = ctypes.c_uint32(capacity)
            granted = ctypes.c_uint32()
            access_status = wintypes.BOOL()
            ctypes.set_last_error(0)
            ok = advapi32.AccessCheck(  # type: ignore[attr-defined]
                descriptor_buffer,
                wintypes.HANDLE(token),
                desired_access,
                ctypes.byref(mapping),
                privileges,
                ctypes.byref(privilege_size),
                ctypes.byref(granted),
                ctypes.byref(access_status),
            )
            error = ctypes.get_last_error()
            if not ok and error == ERROR_INSUFFICIENT_BUFFER and privilege_size.value > capacity:
                capacity = int(privilege_size.value)
                continue
            reported = int(privilege_size.value)
            if reported > capacity:
                copied = Win32QueryBuffer(
                    ctypes.string_at(privileges, capacity),
                    ctypes.addressof(privileges),
                    ctypes.sizeof(ctypes.c_void_p),
                    capacity_bytes=capacity,
                    reported_size_bytes=reported,
                    complete=False,
                )
                raise Win32LocalError("AccessCheck", query_buffers=(copied,))
            if not ok:
                return Win32AccessCheckCallResult(
                    False,
                    None,
                    None,
                    error if 0 < error <= DWORD_MAX else None,
                    None,
                )
            copied = Win32QueryBuffer(
                ctypes.string_at(privileges, reported),
                ctypes.addressof(privileges),
                ctypes.sizeof(ctypes.c_void_p),
                capacity_bytes=capacity,
                reported_size_bytes=reported,
                complete=True,
            )
            return Win32AccessCheckCallResult(
                True,
                bool(access_status.value),
                granted.value,
                None if access_status.value else (
                    error if 0 < error <= DWORD_MAX else None
                ),
                copied,
            )


def _hex_dword(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= DWORD_MAX:
        raise ValueError("value is not a DWORD")
    return hex(value)


def _hex_uint64(value: int) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("value is not an unsigned 64-bit integer")
    return hex(value)


def _valid_handle(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        and value != INVALID_HANDLE_VALUE
    )


def _failure_item(
    context: NativeCallContext,
    *,
    failure_kind: str,
    reason_code: str,
    backend_error: int | None = None,
    binding: NativeResourceBinding | None = None,
) -> ComponentFailureItem:
    selected = context.resource_binding if binding is None else binding
    return ComponentFailureItem(
        failure_kind=failure_kind,
        stage=context.stage,
        component_operation=context.component_operation,
        resource_kind=None if selected is None else selected[0],
        resource_role=None if selected is None else selected[1],
        resource_ordinal=None if selected is None else selected[2],
        operation_ordinal=context.operation_ordinal,
        backend_error_hex=None if backend_error is None else _hex_dword(backend_error),
        reason_code=reason_code,
    )


def _raw_prefix(
    context: NativeCallContext,
    data: bytes,
    *,
    encoding: Literal[
        "security_descriptor",
        "ace",
        "sid",
        "token_information",
        "stream_information",
        "object_information",
    ],
    sequence_offset: int,
    binding: NativeResourceBinding | None = None,
    complete: bool = True,
    declared_size: int | None = None,
) -> NativeRawPrefix:
    selected = context.resource_binding if binding is None else binding
    if selected is None:
        raise ValueError("raw native evidence lacks resource provenance")
    return NativeRawPrefix(
        capture_sequence=context.raw_prefix_start_sequence + sequence_offset,
        source_operation=context.component_operation,
        resource_kind=selected[0],
        resource_role=selected[1],
        resource_ordinal=selected[2],
        encoding=encoding,
        declared_size_bytes=declared_size,
        captured_size_bytes=len(data),
        complete=complete,
        data_hex=data.hex(),
        captured_bytes_sha256=hashlib.sha256(data).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class _CopiedNativeEvidence:
    data: bytes
    encoding: str
    complete: bool
    declared_size: int | None


def _copied_evidence(
    value: object,
    *,
    encoding: str,
) -> _CopiedNativeEvidence | None:
    if type(value) is not Win32QueryBuffer or type(value.data) is not bytes:
        return None
    buffer = cast(Win32QueryBuffer, value)
    capacity = buffer.capacity_bytes
    capacity_valid = (
        capacity is None
        or (
            isinstance(capacity, int)
            and not isinstance(capacity, bool)
            and len(buffer.data) <= capacity <= DWORD_MAX
        )
    )
    reported = buffer.reported_size_bytes
    reported_valid = (
        reported is None
        or (
            isinstance(reported, int)
            and not isinstance(reported, bool)
            and len(buffer.data) <= reported <= DWORD_MAX
        )
    )
    complete = type(buffer.complete) is bool and buffer.complete and capacity_valid
    if complete and reported is not None:
        complete = reported_valid and reported == len(buffer.data)
    declared = reported if reported_valid else None
    return _CopiedNativeEvidence(buffer.data, encoding, complete, declared)


def _append_error_evidence(
    rows: list[_CopiedNativeEvidence],
    error: Win32ApiError | Win32LocalError,
    *,
    encoding: str,
) -> None:
    for buffer in error.query_buffers:
        row = _copied_evidence(buffer, encoding=encoding)
        if row is not None:
            rows.append(row)


def _prefixes(
    context: NativeCallContext,
    rows: list[_CopiedNativeEvidence],
) -> tuple[NativeRawPrefix, ...]:
    return tuple(
        _raw_prefix(
            context,
            row.data,
            encoding=cast(
                Literal[
                    "security_descriptor",
                    "ace",
                    "sid",
                    "token_information",
                    "stream_information",
                    "object_information",
                ],
                row.encoding,
            ),
            sequence_offset=index,
            complete=row.complete,
            declared_size=row.declared_size,
        )
        for index, row in enumerate(rows)
    )


def _raise_protocol(
    context: NativeCallContext,
    *,
    prefixes: tuple[NativeRawPrefix, ...] = (),
    returned_resources: tuple[tuple[object, str, str, int | None], ...] = (),
    binding: NativeResourceBinding | None = None,
) -> None:
    raise PermissionNativeCallError(
        _failure_item(
            context,
            failure_kind="native_protocol_failure",
            reason_code="native_protocol_failure",
            binding=binding,
        ),
        raw_prefixes=prefixes,
        returned_resources=returned_resources,
    )


def _raise_api(
    context: NativeCallContext,
    error: Win32ApiError,
    *,
    prefixes: tuple[NativeRawPrefix, ...] = (),
    returned_resources: tuple[tuple[object, str, str, int | None], ...] = (),
    binding: NativeResourceBinding | None = None,
) -> None:
    if (
        isinstance(error.error_code, bool)
        or not isinstance(error.error_code, int)
        or not 1 <= error.error_code <= DWORD_MAX
    ):
        _raise_protocol(
            context,
            prefixes=prefixes,
            returned_resources=returned_resources,
            binding=binding,
        )
    raise PermissionNativeCallError(
        _failure_item(
            context,
            failure_kind="native_api_failure",
            reason_code="native_api_failure",
            backend_error=error.error_code,
            binding=binding,
        ),
        raw_prefixes=prefixes,
        returned_resources=returned_resources,
    )


def _raise_validation(context: NativeCallContext, reason: str) -> None:
    raise PermissionNativeCallError(
        _failure_item(
            context,
            failure_kind="validation_failure",
            reason_code=reason,
        )
    )


def _raise_refusal(
    context: NativeCallContext,
    reason: str,
    *,
    prefixes: tuple[NativeRawPrefix, ...] = (),
) -> None:
    raise PermissionNativeCallError(
        _failure_item(
            context,
            failure_kind="observation_refusal",
            reason_code=reason,
        ),
        raw_prefixes=prefixes,
    )


def _cleanup_failure(binding: NativeResourceBinding, backend_error: int | None) -> ComponentFailureItem:
    return ComponentFailureItem(
        failure_kind="cleanup_failure",
        stage="resource_cleanup",
        component_operation="release_native_resource",
        resource_kind=binding[0],
        resource_role=binding[1],
        resource_ordinal=binding[2],
        operation_ordinal=None,
        backend_error_hex=None if backend_error is None else _hex_dword(backend_error),
        reason_code="resource_release_failed",
    )


class Win32OwnedObjectHandle:
    """Exactly-once owned file or directory handle."""

    def __init__(
        self,
        bindings: Win32LowLevelBindings,
        raw_handle: int,
        binding: NativeResourceBinding,
        *,
        object_kind: Literal["file", "directory"],
        requested_path: str,
    ) -> None:
        self._bindings = bindings
        self.raw_handle = raw_handle
        self.binding = binding
        self.object_kind = object_kind
        self.requested_path = requested_path
        self.release_attempted = False
        self.released = False
        self._read_lock = threading.Lock()

    def release(self) -> ComponentFailureItem | None:
        if self.release_attempted:
            return _cleanup_failure(self.binding, None)
        self.release_attempted = True
        try:
            self._bindings.close_handle(self.raw_handle)
        except Win32ApiError as exc:
            code = exc.error_code if isinstance(exc.error_code, int) and 0 < exc.error_code <= DWORD_MAX else None
            return _cleanup_failure(self.binding, code)
        except Exception:
            return _cleanup_failure(self.binding, None)
        self.released = True
        return None


class Win32OwnedTokenHandle:
    """Exactly-once owned effective or AccessCheck token handle."""

    def __init__(
        self,
        bindings: Win32LowLevelBindings,
        raw_handle: int,
        binding: NativeResourceBinding,
    ) -> None:
        self._bindings = bindings
        self.raw_handle = raw_handle
        self.binding = binding
        self.release_attempted = False
        self.released = False
        self.last_facts: NativeTokenFacts | None = None

    def release(self) -> ComponentFailureItem | None:
        if self.release_attempted:
            return _cleanup_failure(self.binding, None)
        self.release_attempted = True
        try:
            self._bindings.close_handle(self.raw_handle)
        except Win32ApiError as exc:
            code = exc.error_code if isinstance(exc.error_code, int) and 0 < exc.error_code <= DWORD_MAX else None
            return _cleanup_failure(self.binding, code)
        except Exception:
            return _cleanup_failure(self.binding, None)
        self.released = True
        return None


class Win32OwnedSecurityBuffer:
    """Exactly-once LocalFree allocation with explicit copied-byte access."""

    def __init__(
        self,
        bindings: Win32LowLevelBindings,
        pointer: int,
        size_bytes: int,
        binding: NativeResourceBinding,
    ) -> None:
        self._bindings = bindings
        self.pointer = pointer
        self.size_bytes = size_bytes
        self.binding = binding
        self.release_attempted = False
        self.released = False

    def copy_bytes(self) -> bytes:
        if self.release_attempted:
            raise RuntimeError("security descriptor buffer was already released")
        if (
            not _valid_handle(self.pointer)
            or isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or not 0 < self.size_bytes <= DWORD_MAX
        ):
            raise RuntimeError("security descriptor allocation is not safely readable")
        copied = self._bindings.read_memory(self.pointer, self.size_bytes)
        if type(copied) is not bytes or len(copied) != self.size_bytes:
            raise RuntimeError("security descriptor copy differs from its declared allocation")
        return copied

    def release(self) -> ComponentFailureItem | None:
        if self.release_attempted:
            return _cleanup_failure(self.binding, None)
        self.release_attempted = True
        try:
            self._bindings.local_free(self.pointer)
        except Win32ApiError as exc:
            code = exc.error_code if isinstance(exc.error_code, int) and 0 < exc.error_code <= DWORD_MAX else None
            return _cleanup_failure(self.binding, code)
        except Exception:
            return _cleanup_failure(self.binding, None)
        self.released = True
        return None


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _require_query_buffer(value: object, *, minimum_size: int) -> Win32QueryBuffer:
    if type(value) is not Win32QueryBuffer:
        raise ValueError("native query did not return Win32QueryBuffer")
    result = cast(Win32QueryBuffer, value)
    if (
        type(result.data) is not bytes
        or len(result.data) < minimum_size
        or len(result.data) > DWORD_MAX
        or isinstance(result.base_address, bool)
        or not isinstance(result.base_address, int)
        or result.base_address <= 0
        or result.pointer_size not in {4, 8}
    ):
        raise ValueError("native query buffer metadata is invalid")
    pointer_limit = 1 << (result.pointer_size * 8)
    capacity = len(result.data) if result.capacity_bytes is None else result.capacity_bytes
    reported = result.reported_size_bytes
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or not len(result.data) <= capacity <= DWORD_MAX
        or result.base_address >= pointer_limit
        or result.base_address + capacity > pointer_limit
        or type(result.complete) is not bool
        or result.complete is not True
        or reported is not None
        and (
            isinstance(reported, bool)
            or not isinstance(reported, int)
            or not 0 <= reported <= DWORD_MAX
            or result.complete and reported != len(result.data)
            or not result.complete and reported < len(result.data)
        )
    ):
        raise ValueError("native query buffer extent metadata is invalid")
    return result


def _sid_bytes_at(buffer: Win32QueryBuffer, pointer: int) -> bytes:
    if isinstance(pointer, bool) or not isinstance(pointer, int):
        raise ValueError("SID pointer has the wrong type")
    offset = pointer - buffer.base_address
    if offset < 0 or offset + 8 > len(buffer.data):
        raise ValueError("SID pointer is outside its query buffer")
    length = 8 + 4 * buffer.data[offset + 1]
    if offset + length > len(buffer.data):
        raise ValueError("SID is truncated in its query buffer")
    return buffer.data[offset : offset + length]


def _native_sid(data: bytes) -> NativeSid:
    if len(data) < 8 or len(data) != 8 + 4 * data[1]:
        raise ValueError("SID bytes are malformed")
    authority = int.from_bytes(data[2:8], "big")
    subauthorities = [
        str(int.from_bytes(data[8 + 4 * index : 12 + 4 * index], "little"))
        for index in range(data[1])
    ]
    suffix = "" if not subauthorities else "-" + "-".join(subauthorities)
    return NativeSid(f"S-{data[0]}-{authority}{suffix}", data.hex())


def _pointer_at(buffer: Win32QueryBuffer, offset: int) -> int:
    if offset < 0 or offset + buffer.pointer_size > len(buffer.data):
        raise ValueError("native pointer field is truncated")
    return int.from_bytes(buffer.data[offset : offset + buffer.pointer_size], "little")


def _parse_sid_and_attributes(
    buffer: Win32QueryBuffer, offset: int
) -> tuple[NativeSid, int]:
    entry_size = _align(buffer.pointer_size + 4, buffer.pointer_size)
    if offset < 0 or offset + entry_size > len(buffer.data):
        raise ValueError("SID_AND_ATTRIBUTES is truncated")
    pointer = _pointer_at(buffer, offset)
    attributes = int.from_bytes(
        buffer.data[offset + buffer.pointer_size : offset + buffer.pointer_size + 4],
        "little",
    )
    return _native_sid(_sid_bytes_at(buffer, pointer)), attributes


def _parse_groups(buffer: Win32QueryBuffer) -> tuple[NativeGroup, ...]:
    if len(buffer.data) < 4:
        raise ValueError("TOKEN_GROUPS header is truncated")
    count = int.from_bytes(buffer.data[:4], "little")
    if count == 0:
        return ()
    start = _align(4, buffer.pointer_size)
    entry_size = _align(buffer.pointer_size + 4, buffer.pointer_size)
    if count > (len(buffer.data) - start) // entry_size:
        raise ValueError("TOKEN_GROUPS count exceeds copied entries")
    return tuple(
        NativeGroup(sid, _hex_dword(attributes))
        for sid, attributes in (
            _parse_sid_and_attributes(buffer, start + index * entry_size)
            for index in range(count)
        )
    )


def _parse_single_sid(buffer: Win32QueryBuffer) -> NativeSid:
    pointer = _pointer_at(buffer, 0)
    return _native_sid(_sid_bytes_at(buffer, pointer))


def _parse_luid(data: bytes, offset: int) -> tuple[int, int, str]:
    if offset < 0 or offset + 8 > len(data):
        raise ValueError("LUID is truncated")
    low, high = struct.unpack_from("<II", data, offset)
    return low, high, f"{high:08x}{low:08x}"


def _parse_dword_boolean(buffer: Win32QueryBuffer) -> bool:
    if len(buffer.data) < 4:
        raise ValueError("Boolean token information is truncated")
    value = int.from_bytes(buffer.data[:4], "little")
    if value not in {0, 1}:
        raise ValueError("Boolean token information is not BOOL-normalized")
    return bool(value)


class _UnsupportedTokenFeature(ValueError):
    pass


def _require_empty_claim_attributes(buffer: Win32QueryBuffer) -> None:
    header_size = 8 + buffer.pointer_size
    if len(buffer.data) < header_size:
        raise ValueError("claim security attributes header is truncated")
    version = int.from_bytes(buffer.data[0:2], "little")
    if version != 1:
        raise ValueError("claim security attributes version is unsupported")
    count = int.from_bytes(buffer.data[4:8], "little")
    if count != 0:
        raise _UnsupportedTokenFeature("claim security attributes are nonempty")


def _require_empty_device_groups(buffer: Win32QueryBuffer) -> None:
    if len(buffer.data) < 4:
        raise ValueError("device group header is truncated")
    count = int.from_bytes(buffer.data[:4], "little")
    if count == 0:
        return
    start = _align(4, buffer.pointer_size)
    entry_size = _align(buffer.pointer_size + 4, buffer.pointer_size)
    if count > (len(buffer.data) - start) // entry_size:
        raise ValueError("device group count exceeds copied entries")
    raise _UnsupportedTokenFeature("device groups are nonempty")


def _validate_held_token_authz_inputs(
    adapter: Win32PermissionNativeApi,
    token: object,
    expected_token_identity: object,
    descriptor: object,
    object_binding: object,
    authz_bindings: object,
) -> tuple[
    Win32OwnedTokenHandle,
    _principal_authz.SuppliedTokenIdentity,
    NativeSecurityDescriptor,
    NativeAccessObjectBinding,
]:
    try:
        existing_bindings = adapter._injected_bindings
        if existing_bindings is None:
            existing_bindings = adapter._default_bindings
        if (
            type(token) is not Win32OwnedTokenHandle
            or existing_bindings is None
            or token._bindings is not existing_bindings
            or type(token.release_attempted) is not bool
            or token.release_attempted is not False
            or type(token.released) is not bool
            or token.released is not False
            or type(token.binding) is not tuple
            or token.binding
            != ("access_check_token_handle", "access_check_duplicate", None)
            or not _valid_handle(token.raw_handle)
        ):
            raise ValueError("held token wrapper is invalid")
        _principal_authz._validate_handle(token.raw_handle)

        facts = token.last_facts
        _validate_token_facts(facts, duplicate=True)
        facts_identity = _principal_authz.SuppliedTokenIdentity(
            token_id_luid_hex=facts.token_id_luid_hex,
            authentication_id_luid_hex=facts.authentication_id_luid_hex,
            modified_id_luid_hex=facts.modified_id_luid_hex,
            token_type=facts.token_type,
            impersonation_level=cast(object, facts.impersonation_level),
        )
        _principal_authz._validate_identity(facts_identity)
        checked_identity = _principal_authz._validate_identity(expected_token_identity)
        if (
            facts.token_id_luid_hex != checked_identity.token_id_luid_hex
            or facts.authentication_id_luid_hex
            != checked_identity.authentication_id_luid_hex
            or facts.modified_id_luid_hex != checked_identity.modified_id_luid_hex
            or facts.token_type != checked_identity.token_type
            or facts.impersonation_level != checked_identity.impersonation_level
        ):
            raise ValueError("held token identity differs")

        checked_descriptor, _ = _principal_authz._validate_descriptor(descriptor)
        checked_binding = _principal_authz._validate_binding(
            object_binding,
            checked_descriptor,
        )
        _principal_authz._validate_bindings(authz_bindings)
    except Exception:
        validation_failed = True
    else:
        validation_failed = False
    if validation_failed:
        raise HeldTokenAuthzValidationError(
            "held-token Authz input validation failed"
        )
    return (
        token,
        checked_identity,
        checked_descriptor,
        checked_binding,
    )


class Win32PermissionNativeApi:
    """Concrete PermissionNativeApi over a lazy or injected Win32 boundary."""

    def __init__(self, bindings: Win32LowLevelBindings | None = None) -> None:
        self._injected_bindings = bindings
        self._default_bindings: _CtypesWin32Bindings | None = None

    @property
    def default_boundary_loaded(self) -> bool:
        return bool(self._default_bindings and self._default_bindings.loaded)

    def _bindings(self) -> Win32LowLevelBindings:
        if self._injected_bindings is not None:
            return self._injected_bindings
        if self._default_bindings is None:
            self._default_bindings = _CtypesWin32Bindings()
        return self._default_bindings

    @staticmethod
    def _context(context: NativeCallContext, operation: str) -> NativeCallContext:
        validate_native_call_context(context)
        if context.component_operation != operation:
            raise PermissionObservationContractError(
                "native adapter method and explicit call context differ"
            )
        return context

    @staticmethod
    def _open_specification(
        specification: NativeOpenSpecification,
        object_kind: Literal["file", "directory"],
    ) -> None:
        expected = NativeOpenSpecification(
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
        if type(specification) is not NativeOpenSpecification or specification != expected:
            raise PermissionObservationContractError(
                "Win32 adapter rejects an open outside the frozen read-only profile"
            )

    @staticmethod
    def _path(path: str) -> None:
        if (
            not isinstance(path, str)
            or not path
            or "\x00" in path
            or not path.startswith("\\\\?\\Volume{")
            or "}\\" not in path
            or unicodedata.normalize("NFC", path) != path
        ):
            raise PermissionObservationContractError(
                "Win32 adapter requires an NFC canonical volume-GUID path"
            )

    def _open(
        self,
        path: str,
        *,
        specification: NativeOpenSpecification,
        context: NativeCallContext,
        object_kind: Literal["file", "directory"],
    ) -> Win32OwnedObjectHandle:
        operation = "open_existing_subject" if object_kind == "file" else "open_existing_directory"
        self._context(context, operation)
        self._open_specification(specification, object_kind)
        self._path(path)
        binding = context.allowed_returned_resource_binding
        if binding is None:
            raise PermissionObservationContractError("open context omits transferred handle binding")
        bindings = self._bindings()
        desired_access = FILE_READ_ATTRIBUTES | READ_CONTROL
        if object_kind == "file":
            desired_access |= FILE_READ_DATA
        flags = FILE_FLAG_OPEN_REPARSE_POINT
        if object_kind == "directory":
            flags |= FILE_FLAG_BACKUP_SEMANTICS
        try:
            raw = bindings.create_file(
                path,
                desired_access=desired_access,
                share_mode=FILE_SHARE_READ,
                creation_disposition=OPEN_EXISTING,
                flags_and_attributes=flags,
            )
        except Win32ApiError as exc:
            returned: tuple[tuple[object, str, str, int | None], ...] = ()
            failure_binding: NativeResourceBinding | None = None
            if _valid_handle(exc.returned_handle):
                partial = Win32OwnedObjectHandle(
                    bindings,
                    cast(int, exc.returned_handle),
                    binding,
                    object_kind=object_kind,
                    requested_path=path,
                )
                returned = ((partial, binding[0], binding[1], binding[2]),)
                failure_binding = binding
            _raise_api(
                context,
                exc,
                returned_resources=returned,
                binding=failure_binding,
            )
        except Win32LocalError as exc:
            returned = ()
            failure_binding = None
            if _valid_handle(exc.returned_handle):
                partial = Win32OwnedObjectHandle(
                    bindings,
                    cast(int, exc.returned_handle),
                    binding,
                    object_kind=object_kind,
                    requested_path=path,
                )
                returned = ((partial, binding[0], binding[1], binding[2]),)
                failure_binding = binding
            _raise_protocol(
                context,
                returned_resources=returned,
                binding=failure_binding,
            )
        except Exception:
            _raise_protocol(context)
        if not _valid_handle(raw):
            _raise_protocol(context)
        return Win32OwnedObjectHandle(
            bindings,
            cast(int, raw),
            binding,
            object_kind=object_kind,
            requested_path=path,
        )

    def open_existing_directory(
        self,
        path: str,
        *,
        specification: NativeOpenSpecification,
        context: NativeCallContext,
    ) -> Win32OwnedObjectHandle:
        return self._open(
            path,
            specification=specification,
            context=context,
            object_kind="directory",
        )

    def open_existing_subject(
        self,
        path: str,
        *,
        specification: NativeOpenSpecification,
        context: NativeCallContext,
    ) -> Win32OwnedObjectHandle:
        return self._open(
            path,
            specification=specification,
            context=context,
            object_kind="file",
        )

    def _object_handle(
        self,
        handle: object,
        context: NativeCallContext,
        *,
        subject_only: bool = False,
    ) -> Win32OwnedObjectHandle:
        if (
            type(handle) is not Win32OwnedObjectHandle
            or handle._bindings is not self._bindings()
            or handle.release_attempted
            or context.resource_binding != handle.binding
            or subject_only and handle.object_kind != "file"
        ):
            _raise_protocol(context)
        return cast(Win32OwnedObjectHandle, handle)

    @staticmethod
    def _query_payload(
        bindings: Win32LowLevelBindings,
        handle: Win32OwnedObjectHandle,
        *,
        information_class: int,
        minimum_size: int,
        variable_length: bool,
        evidence: list[_CopiedNativeEvidence],
        encoding: str,
    ) -> Win32QueryBuffer:
        raw = bindings.query_file_information(
            handle.raw_handle,
            information_class=information_class,
            minimum_size=minimum_size,
            variable_length=variable_length,
        )
        copied = _copied_evidence(raw, encoding=encoding)
        if copied is not None:
            evidence.append(copied)
        return _require_query_buffer(raw, minimum_size=minimum_size)

    def query_object(
        self,
        handle: Win32OwnedObjectHandle,
        *,
        context: NativeCallContext,
    ) -> RawNativeObjectSnapshot:
        self._context(context, "query_object")
        owned = self._object_handle(handle, context)
        bindings = self._bindings()
        evidence: list[_CopiedNativeEvidence] = []
        try:
            guid_path = bindings.get_final_path(
                owned.raw_handle, flags=FILE_NAME_NORMALIZED | VOLUME_NAME_GUID
            )
            if not isinstance(guid_path, str) or not guid_path:
                raise ValueError("volume-GUID path is not a nonempty string")
            guid_bytes = guid_path.encode("utf-16le")
            evidence.append(
                _CopiedNativeEvidence(guid_bytes, "object_information", True, len(guid_bytes))
            )
            dos_path = bindings.get_final_path(
                owned.raw_handle, flags=FILE_NAME_NORMALIZED | VOLUME_NAME_DOS
            )
            if not isinstance(dos_path, str) or not dos_path:
                raise ValueError("DOS path is not a nonempty string")
            dos_bytes = dos_path.encode("utf-16le")
            evidence.append(
                _CopiedNativeEvidence(dos_bytes, "object_information", True, len(dos_bytes))
            )

            name_info = self._query_payload(
                bindings,
                owned,
                information_class=FILE_NAME_INFO_CLASS,
                minimum_size=4,
                variable_length=True,
                evidence=evidence,
                encoding="object_information",
            )
            name_length = int.from_bytes(name_info.data[:4], "little")
            if name_length == 0 or name_length % 2 or 4 + name_length > len(name_info.data):
                raise ValueError("FILE_NAME_INFO length is invalid")
            name_bytes = name_info.data[4 : 4 + name_length]
            canonical_name = name_bytes.decode("utf-16le", errors="strict")

            file_id_info = self._query_payload(
                bindings,
                owned,
                information_class=FILE_ID_INFO_CLASS,
                minimum_size=24,
                variable_length=False,
                evidence=evidence,
                encoding="object_information",
            )
            file_id_bytes = file_id_info.data[:24]
            volume_serial = int.from_bytes(file_id_bytes[:8], "little")

            standard_info = self._query_payload(
                bindings,
                owned,
                information_class=FILE_STANDARD_INFO_CLASS,
                minimum_size=ctypes.sizeof(_FILE_STANDARD_INFO),
                variable_length=False,
                evidence=evidence,
                encoding="object_information",
            )
            standard_bytes = standard_info.data[:22]
            allocation_size, end_of_file, links = struct.unpack_from("<qqI", standard_bytes, 0)
            delete_pending = standard_bytes[20]
            directory = standard_bytes[21]
            if (
                allocation_size < 0
                or end_of_file < 0
                or allocation_size < end_of_file
                or delete_pending not in {0, 1}
                or directory not in {0, 1}
            ):
                raise ValueError("FILE_STANDARD_INFO fields are invalid")

            attribute_info = self._query_payload(
                bindings,
                owned,
                information_class=FILE_ATTRIBUTE_TAG_INFO_CLASS,
                minimum_size=8,
                variable_length=False,
                evidence=evidence,
                encoding="object_information",
            )
            attribute_bytes = attribute_info.data[:8]
            attributes, reparse_tag = struct.unpack_from("<II", attribute_bytes)

            legacy = bindings.get_legacy_file_information(owned.raw_handle)
            if type(legacy) is bytes:
                evidence.append(
                    _CopiedNativeEvidence(legacy, "object_information", True, len(legacy))
                )
            if type(legacy) is not bytes or len(legacy) < 52:
                raise ValueError("BY_HANDLE_FILE_INFORMATION bytes are truncated")
            legacy = legacy[:52]
            legacy_attributes = int.from_bytes(legacy[0:4], "little")
            legacy_volume = int.from_bytes(legacy[28:32], "little")
            legacy_size = (
                int.from_bytes(legacy[32:36], "little") << 32
            ) | int.from_bytes(legacy[36:40], "little")
            legacy_links = int.from_bytes(legacy[40:44], "little")
            legacy_index = (
                int.from_bytes(legacy[44:48], "little") << 32
            ) | int.from_bytes(legacy[48:52], "little")
            if (
                legacy_attributes != attributes
                or legacy_size != end_of_file
                or legacy_links != links
            ):
                raise ValueError("strong and legacy file information disagree")
            if bool(directory) != (owned.object_kind == "directory"):
                raise ValueError("held handle kind differs from FILE_STANDARD_INFO")
            if any(
                unicodedata.normalize("NFC", value) != value
                for value in (guid_path, dos_path, canonical_name)
            ):
                unicode_nfc_exact = False
            else:
                unicode_nfc_exact = True
            identity = NativeFileIdentity(
                volume_serial_number_hex=_hex_uint64(volume_serial),
                file_id_128_hex=file_id_bytes[8:24].hex(),
                legacy_volume_serial_number_hex=f"{legacy_volume:08x}",
                legacy_file_index_hex=f"{legacy_index:016x}",
            )
            return RawNativeObjectSnapshot(
                canonical_volume_guid_path=guid_path,
                canonical_dos_path=dos_path,
                canonical_name_exact=canonical_name,
                unicode_nfc_exact=unicode_nfc_exact,
                identity=identity,
                file_attributes_hex=_hex_dword(attributes),
                reparse_tag_hex=_hex_dword(reparse_tag),
                end_of_file_bytes=end_of_file,
                allocation_size_bytes=allocation_size,
                number_of_links=links,
                delete_pending=bool(delete_pending),
                directory=bool(directory),
            )
        except PermissionNativeCallError:
            raise
        except Win32ApiError as exc:
            _append_error_evidence(evidence, exc, encoding="object_information")
            _raise_api(context, exc, prefixes=_prefixes(context, evidence))
        except Win32LocalError as exc:
            _append_error_evidence(evidence, exc, encoding="object_information")
            _raise_protocol(context, prefixes=_prefixes(context, evidence))
        except Exception:
            _raise_protocol(context, prefixes=_prefixes(context, evidence))
        raise AssertionError("unreachable")

    def query_streams(
        self,
        handle: Win32OwnedObjectHandle,
        *,
        context: NativeCallContext,
    ) -> tuple[NativeStream, ...]:
        self._context(context, "query_streams")
        owned = self._object_handle(handle, context, subject_only=True)
        evidence: list[_CopiedNativeEvidence] = []
        try:
            raw = self._query_payload(
                self._bindings(),
                owned,
                information_class=FILE_STREAM_INFO_CLASS,
                minimum_size=24,
                variable_length=True,
                evidence=evidence,
                encoding="stream_information",
            ).data
            streams: list[NativeStream] = []
            cursor = 0
            while True:
                if cursor < 0 or cursor + 24 > len(raw):
                    raise ValueError("FILE_STREAM_INFO header is outside the copied buffer")
                next_offset, name_length = struct.unpack_from("<II", raw, cursor)
                size, allocation = struct.unpack_from("<qq", raw, cursor + 8)
                if (
                    name_length == 0
                    or name_length % 2
                    or cursor + 24 + name_length > len(raw)
                    or size < 0
                    or allocation < size
                ):
                    raise ValueError("FILE_STREAM_INFO entry is malformed")
                name_bytes = raw[cursor + 24 : cursor + 24 + name_length]
                name = name_bytes.decode("utf-16le", errors="strict")
                streams.append(
                    NativeStream(
                        stream_index=len(streams),
                        stream_name=name,
                        stream_name_utf16le_hex=name_bytes.hex(),
                        stream_size_bytes=size,
                        stream_allocation_size_bytes=allocation,
                    )
                )
                if next_offset == 0:
                    break
                if next_offset < 24 + name_length or cursor + next_offset <= cursor:
                    raise ValueError("FILE_STREAM_INFO next offset is invalid")
                cursor += next_offset
            return tuple(streams)
        except PermissionNativeCallError:
            raise
        except Win32ApiError as exc:
            _append_error_evidence(evidence, exc, encoding="stream_information")
            _raise_api(context, exc, prefixes=_prefixes(context, evidence))
        except Win32LocalError as exc:
            _append_error_evidence(evidence, exc, encoding="stream_information")
            _raise_protocol(context, prefixes=_prefixes(context, evidence))
        except Exception:
            _raise_protocol(context, prefixes=_prefixes(context, evidence))
        raise AssertionError("unreachable")

    def read_unnamed_at(
        self,
        handle: Win32OwnedObjectHandle,
        *,
        offset: int,
        requested_bytes: int,
        context: NativeCallContext,
    ) -> bytes:
        self._context(context, "read_unnamed_at")
        owned = self._object_handle(handle, context, subject_only=True)
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset <= SIGNED_LARGE_INTEGER_MAX
            or isinstance(requested_bytes, bool)
            or not isinstance(requested_bytes, int)
            or not 0 < requested_bytes <= DWORD_MAX
            or offset + requested_bytes > SIGNED_LARGE_INTEGER_MAX
        ):
            raise PermissionObservationContractError("opaque read bounds are invalid")
        try:
            with owned._read_lock:
                returned = self._bindings().read_file_at(
                    owned.raw_handle,
                    offset=offset,
                    requested_bytes=requested_bytes,
                )
            if type(returned) is not bytes or len(returned) > requested_bytes:
                _raise_protocol(context)
            return cast(bytes, returned)
        except PermissionNativeCallError:
            raise
        except Win32ApiError as exc:
            _raise_api(context, exc)
        except Win32LocalError:
            _raise_protocol(context)
        except Exception:
            _raise_protocol(context)
        raise AssertionError("unreachable")

    def acquire_security_descriptor(
        self,
        handle: Win32OwnedObjectHandle,
        *,
        context: NativeCallContext,
    ) -> Win32OwnedSecurityBuffer:
        self._context(context, "acquire_security_descriptor")
        owned = self._object_handle(handle, context)
        binding = context.allowed_returned_resource_binding
        if binding is None:
            raise PermissionObservationContractError(
                "security acquisition context omits its transferred buffer binding"
            )
        bindings = self._bindings()
        try:
            allocation = bindings.get_security_info(
                owned.raw_handle,
                security_information=(
                    OWNER_SECURITY_INFORMATION
                    | GROUP_SECURITY_INFORMATION
                    | DACL_SECURITY_INFORMATION
                ),
            )
        except Win32ApiError as exc:
            returned: tuple[tuple[object, str, str, int | None], ...] = ()
            failure_binding: NativeResourceBinding | None = context.resource_binding
            if _valid_handle(exc.returned_pointer):
                partial = Win32OwnedSecurityBuffer(
                    bindings,
                    cast(int, exc.returned_pointer),
                    exc.returned_size if isinstance(exc.returned_size, int) else 0,
                    binding,
                )
                returned = ((partial, binding[0], binding[1], binding[2]),)
                failure_binding = binding
            _raise_api(
                context,
                exc,
                returned_resources=returned,
                binding=failure_binding,
            )
        except Win32LocalError as exc:
            returned = ()
            failure_binding = context.resource_binding
            if _valid_handle(exc.returned_pointer):
                partial = Win32OwnedSecurityBuffer(
                    bindings,
                    cast(int, exc.returned_pointer),
                    exc.returned_size if isinstance(exc.returned_size, int) else 0,
                    binding,
                )
                returned = ((partial, binding[0], binding[1], binding[2]),)
                failure_binding = binding
            _raise_protocol(
                context,
                returned_resources=returned,
                binding=failure_binding,
            )
        except Exception:
            _raise_protocol(context)
        if type(allocation) is not Win32SecurityAllocation:
            _raise_protocol(context)
        pointer = allocation.pointer
        size = allocation.size_bytes
        if not _valid_handle(pointer):
            _raise_protocol(context)
        buffer = Win32OwnedSecurityBuffer(bindings, pointer, size, binding)
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= DWORD_MAX
        ):
            returned = ((buffer, binding[0], binding[1], binding[2]),)
            _raise_protocol(
                context,
                returned_resources=returned,
                binding=binding,
            )
        return buffer

    @staticmethod
    def _descriptor_sid(data: bytes, offset: int) -> NativeSid:
        if offset <= 0 or offset + 8 > len(data):
            raise ValueError("security descriptor SID offset is invalid")
        length = 8 + 4 * data[offset + 1]
        if offset + length > len(data):
            raise ValueError("security descriptor SID is truncated")
        return _native_sid(data[offset : offset + length])

    def parse_security_descriptor(
        self,
        data: bytes,
        *,
        context: NativeCallContext,
    ) -> NativeSecurityDescriptor:
        self._context(context, "parse_security_descriptor")
        if type(data) is not bytes:
            _raise_protocol(context)
        try:
            if len(data) < 20:
                _raise_refusal(context, "descriptor_incomplete")
            revision = data[0]
            control = int.from_bytes(data[2:4], "little")
            if revision == 0 or not control & 0x8000:
                _raise_refusal(context, "descriptor_incomplete")
            owner_offset = int.from_bytes(data[4:8], "little")
            group_offset = int.from_bytes(data[8:12], "little")
            dacl_offset = int.from_bytes(data[16:20], "little")
            owner = self._descriptor_sid(data, owner_offset)
            group = self._descriptor_sid(data, group_offset)
            dacl_present = bool(control & 0x0004)
            aces: list[NativeAce] = []
            dacl_revision = 0
            if dacl_present and dacl_offset:
                if dacl_offset + 8 > len(data):
                    _raise_refusal(context, "descriptor_incomplete")
                dacl_revision = data[dacl_offset]
                acl_size = int.from_bytes(data[dacl_offset + 2 : dacl_offset + 4], "little")
                ace_count = int.from_bytes(data[dacl_offset + 4 : dacl_offset + 6], "little")
                acl_end = dacl_offset + acl_size
                if acl_size < 8 or acl_end > len(data):
                    _raise_refusal(context, "descriptor_incomplete")
                cursor = dacl_offset + 8
                for index in range(ace_count):
                    if cursor + 8 > acl_end:
                        _raise_refusal(context, "descriptor_incomplete")
                    ace_type = data[cursor]
                    ace_flags = data[cursor + 1]
                    ace_size = int.from_bytes(data[cursor + 2 : cursor + 4], "little")
                    if ace_size < 8 or cursor + ace_size > acl_end:
                        _raise_refusal(context, "descriptor_incomplete")
                    raw_ace = data[cursor : cursor + ace_size]
                    if ace_type not in {0, 1}:
                        _raise_refusal(context, "ace_unsupported")
                    trustee = _native_sid(raw_ace[8:])
                    aces.append(
                        NativeAce(
                            ace_index=index,
                            ace_kind="allow" if ace_type == 0 else "deny",
                            ace_type_hex=_hex_dword(ace_type),
                            ace_flags_hex=_hex_dword(ace_flags),
                            access_mask_hex=_hex_dword(
                                int.from_bytes(raw_ace[4:8], "little")
                            ),
                            trustee_sid=trustee,
                            raw_ace_hex=raw_ace.hex(),
                        )
                    )
                    cursor += ace_size
                if cursor != acl_end:
                    _raise_refusal(context, "descriptor_incomplete")
            elif dacl_present or dacl_offset:
                _raise_refusal(context, "descriptor_incomplete")
            return NativeSecurityDescriptor(
                self_relative_descriptor_hex=data.hex(),
                self_relative_descriptor_sha256=hashlib.sha256(data).hexdigest(),
                revision=revision,
                control_hex=_hex_dword(control),
                owner_sid=owner,
                group_sid=group,
                dacl_present=dacl_present,
                dacl_defaulted=bool(control & 0x0008),
                dacl_revision=dacl_revision,
                dacl_protected=bool(control & 0x1000),
                dacl_auto_inherited=bool(control & 0x0400),
                ordered_dacl_aces=tuple(aces),
            )
        except PermissionNativeCallError:
            raise
        except Exception:
            _raise_refusal(context, "descriptor_incomplete")
        raise AssertionError("unreachable")

    def _partial_token(
        self,
        context: NativeCallContext,
        bindings: Win32LowLevelBindings,
        error: Win32ApiError | Win32LocalError,
    ) -> tuple[
        tuple[tuple[object, str, str, int | None], ...],
        NativeResourceBinding | None,
    ]:
        binding = context.allowed_returned_resource_binding
        if binding is not None and _valid_handle(error.returned_handle):
            partial = Win32OwnedTokenHandle(
                bindings,
                cast(int, error.returned_handle),
                binding,
            )
            return ((partial, binding[0], binding[1], binding[2]),), binding
        return (), context.resource_binding

    def open_current_effective_token(
        self,
        *,
        context: NativeCallContext,
    ) -> tuple[
        Literal["thread_impersonation", "process_primary"],
        Win32OwnedTokenHandle,
    ]:
        self._context(context, "open_current_effective_token")
        binding = context.allowed_returned_resource_binding
        if binding is None:
            raise PermissionObservationContractError("effective-token context omits transfer binding")
        bindings = self._bindings()
        desired = TOKEN_QUERY | TOKEN_DUPLICATE
        try:
            try:
                raw = bindings.open_thread_token(
                    desired_access=desired,
                    open_as_self=True,
                )
                source: Literal["thread_impersonation", "process_primary"] = "thread_impersonation"
            except Win32ApiError as thread_error:
                if thread_error.error_code != ERROR_NO_TOKEN or _valid_handle(
                    thread_error.returned_handle
                ):
                    returned, failure_binding = self._partial_token(
                        context, bindings, thread_error
                    )
                    _raise_api(
                        context,
                        thread_error,
                        returned_resources=returned,
                        binding=failure_binding,
                    )
                raw = bindings.open_process_token(desired_access=desired)
                source = "process_primary"
        except PermissionNativeCallError:
            raise
        except Win32ApiError as exc:
            returned, failure_binding = self._partial_token(context, bindings, exc)
            _raise_api(
                context,
                exc,
                returned_resources=returned,
                binding=failure_binding,
            )
        except Win32LocalError as exc:
            returned, failure_binding = self._partial_token(context, bindings, exc)
            _raise_protocol(
                context,
                returned_resources=returned,
                binding=failure_binding,
            )
        except Exception:
            _raise_protocol(context)
        if not _valid_handle(raw):
            _raise_protocol(context)
        return source, Win32OwnedTokenHandle(bindings, cast(int, raw), binding)

    def _token_handle(
        self,
        token: object,
        context: NativeCallContext,
    ) -> Win32OwnedTokenHandle:
        if (
            type(token) is not Win32OwnedTokenHandle
            or token._bindings is not self._bindings()
            or token.release_attempted
            or context.resource_binding != token.binding
        ):
            _raise_protocol(context)
        return cast(Win32OwnedTokenHandle, token)

    def duplicate_for_access_check(
        self,
        token: Win32OwnedTokenHandle,
        *,
        context: NativeCallContext,
    ) -> Win32OwnedTokenHandle:
        self._context(context, "duplicate_for_access_check")
        original = self._token_handle(token, context)
        binding = context.allowed_returned_resource_binding
        if binding is None:
            raise PermissionObservationContractError("duplicate-token context omits transfer binding")
        bindings = self._bindings()
        try:
            raw = bindings.duplicate_token(
                original.raw_handle,
                desired_access=TOKEN_QUERY,
                impersonation_level=SECURITY_IMPERSONATION,
                token_type=TOKEN_IMPERSONATION,
            )
        except Win32ApiError as exc:
            returned, failure_binding = self._partial_token(context, bindings, exc)
            _raise_api(
                context,
                exc,
                returned_resources=returned,
                binding=failure_binding,
            )
        except Win32LocalError as exc:
            returned, failure_binding = self._partial_token(context, bindings, exc)
            _raise_protocol(
                context,
                returned_resources=returned,
                binding=failure_binding,
            )
        except Exception:
            _raise_protocol(context)
        if not _valid_handle(raw):
            _raise_protocol(context)
        return Win32OwnedTokenHandle(bindings, cast(int, raw), binding)

    @staticmethod
    def _statistics(buffer: Win32QueryBuffer) -> tuple[str, str | None, str, str, str]:
        if len(buffer.data) < 56:
            raise ValueError("TOKEN_STATISTICS is truncated")
        _, _, token_id = _parse_luid(buffer.data, 0)
        _, _, authentication_id = _parse_luid(buffer.data, 8)
        token_type_value = int.from_bytes(buffer.data[24:28], "little")
        impersonation_value = int.from_bytes(buffer.data[28:32], "little")
        _, _, modified_id = _parse_luid(buffer.data, 48)
        token_types = {1: "primary", 2: "impersonation"}
        levels = {
            0: "anonymous",
            1: "identification",
            2: "security_impersonation",
            3: "security_delegation",
        }
        if token_type_value not in token_types:
            raise ValueError("TOKEN_STATISTICS token type is unsupported")
        token_type = token_types[token_type_value]
        if token_type == "primary":
            level = None
        else:
            if impersonation_value not in levels:
                raise ValueError("TOKEN_STATISTICS impersonation level is unsupported")
            level = levels[impersonation_value]
        return token_type, level, token_id, authentication_id, modified_id

    def query_token(
        self,
        token: Win32OwnedTokenHandle,
        *,
        context: NativeCallContext,
    ) -> NativeTokenFacts:
        self._context(context, "query_token")
        owned = self._token_handle(token, context)
        bindings = self._bindings()
        evidence: list[_CopiedNativeEvidence] = []

        def query(information_class: int, minimum: int = 4) -> Win32QueryBuffer:
            raw = bindings.query_token_information(
                owned.raw_handle,
                information_class=information_class,
            )
            copied = _copied_evidence(raw, encoding="token_information")
            if copied is not None:
                evidence.append(copied)
            return _require_query_buffer(raw, minimum_size=minimum)

        try:
            statistics = query(TOKEN_STATISTICS, 56)
            user_buffer = query(TOKEN_USER, 1)
            owner_buffer = query(TOKEN_OWNER, 1)
            primary_group_buffer = query(TOKEN_PRIMARY_GROUP, 1)
            groups_buffer = query(TOKEN_GROUPS, 4)
            restricted_buffer = query(TOKEN_RESTRICTED_SIDS, 4)
            privileges_buffer = query(TOKEN_PRIVILEGES, 4)
            integrity_buffer = query(TOKEN_INTEGRITY_LEVEL, 1)
            session_buffer = query(TOKEN_SESSION_ID, 4)
            elevation_type_buffer = query(TOKEN_ELEVATION_TYPE, 4)
            elevation_buffer = query(TOKEN_ELEVATION, 4)
            restrictions_buffer = query(TOKEN_HAS_RESTRICTIONS, 4)
            app_container_buffer = query(TOKEN_IS_APP_CONTAINER, 4)
            capabilities_buffer = query(TOKEN_CAPABILITIES, 4)
            virtualization_allowed_buffer = query(TOKEN_VIRTUALIZATION_ALLOWED, 4)
            virtualization_enabled_buffer = query(TOKEN_VIRTUALIZATION_ENABLED, 4)
            ui_access_buffer = query(TOKEN_UI_ACCESS, 4)
            user_claims_buffer = query(TOKEN_USER_CLAIM_ATTRIBUTES, 8)
            device_claims_buffer = query(TOKEN_DEVICE_CLAIM_ATTRIBUTES, 8)
            device_groups_buffer = query(TOKEN_DEVICE_GROUPS, 4)
            restricted_device_groups_buffer = query(TOKEN_RESTRICTED_DEVICE_GROUPS, 4)

            _require_empty_claim_attributes(user_claims_buffer)
            _require_empty_claim_attributes(device_claims_buffer)
            _require_empty_device_groups(device_groups_buffer)
            _require_empty_device_groups(restricted_device_groups_buffer)

            token_type, impersonation_level, token_id, authentication_id, modified_id = self._statistics(
                statistics
            )
            user = _parse_single_sid(user_buffer)
            owner = _parse_single_sid(owner_buffer)
            primary_group = _parse_single_sid(primary_group_buffer)
            groups = _parse_groups(groups_buffer)
            restricted = _parse_groups(restricted_buffer)
            integrity, _ = _parse_sid_and_attributes(integrity_buffer, 0)
            capabilities = _parse_groups(capabilities_buffer)

            privilege_count = int.from_bytes(privileges_buffer.data[:4], "little")
            if privilege_count > (len(privileges_buffer.data) - 4) // 12:
                raise ValueError("TOKEN_PRIVILEGES count exceeds copied entries")
            privileges: list[NativePrivilege] = []
            for index in range(privilege_count):
                offset = 4 + index * 12
                low, high, luid_hex = _parse_luid(privileges_buffer.data, offset)
                attributes = int.from_bytes(
                    privileges_buffer.data[offset + 8 : offset + 12], "little"
                )
                name = bindings.lookup_privilege_name(
                    luid_low=low,
                    luid_high=high,
                )
                if (
                    not isinstance(name, str)
                    or not name
                    or unicodedata.normalize("NFC", name) != name
                ):
                    raise ValueError("LookupPrivilegeNameW returned an invalid name")
                privileges.append(
                    NativePrivilege(
                        luid_hex=luid_hex,
                        name=name,
                        attributes_hex=_hex_dword(attributes),
                    )
                )

            session_id = int.from_bytes(session_buffer.data[:4], "little")
            elevation_value = int.from_bytes(elevation_type_buffer.data[:4], "little")
            elevation_types = {1: "default", 2: "full", 3: "limited"}
            if elevation_value not in elevation_types:
                raise ValueError("TOKEN_ELEVATION_TYPE is outside the closed domain")
            facts = NativeTokenFacts(
                token_type=cast(Literal["primary", "impersonation"], token_type),
                impersonation_level=impersonation_level,
                token_id_luid_hex=token_id,
                authentication_id_luid_hex=authentication_id,
                modified_id_luid_hex=modified_id,
                session_id=session_id,
                user=user,
                owner=owner,
                primary_group=primary_group,
                ordered_groups=groups,
                ordered_restricted_sids=restricted,
                ordered_privileges=tuple(privileges),
                integrity_level_sid=integrity,
                elevation_type=cast(
                    Literal["default", "full", "limited"],
                    elevation_types[elevation_value],
                ),
                is_elevated=_parse_dword_boolean(elevation_buffer),
                has_restrictions=_parse_dword_boolean(restrictions_buffer),
                is_app_container=_parse_dword_boolean(app_container_buffer),
                ordered_capabilities=capabilities,
                virtualization_allowed=_parse_dword_boolean(
                    virtualization_allowed_buffer
                ),
                virtualization_enabled=_parse_dword_boolean(
                    virtualization_enabled_buffer
                ),
                ui_access=_parse_dword_boolean(ui_access_buffer),
            )
            owned.last_facts = facts
            return facts
        except PermissionNativeCallError:
            raise
        except _UnsupportedTokenFeature:
            _raise_refusal(
                context,
                "token_unsupported",
                prefixes=_prefixes(context, evidence),
            )
        except Win32ApiError as exc:
            _append_error_evidence(evidence, exc, encoding="token_information")
            _raise_api(context, exc, prefixes=_prefixes(context, evidence))
        except Win32LocalError as exc:
            _append_error_evidence(evidence, exc, encoding="token_information")
            _raise_protocol(context, prefixes=_prefixes(context, evidence))
        except Exception:
            _raise_protocol(context, prefixes=_prefixes(context, evidence))
        raise AssertionError("unreachable")

    @staticmethod
    def _mapping_value(value: str) -> int:
        if not isinstance(value, str):
            raise ValueError("generic mapping value is not a string")
        parsed = int(value, 16)
        if value != hex(parsed) or not 0 <= parsed <= DWORD_MAX or parsed & 0xF0000000:
            raise ValueError("generic mapping value is not minimal mapped DWORD hex")
        return parsed

    def _access_privileges(
        self,
        bindings: Win32LowLevelBindings,
        raw: Win32QueryBuffer,
    ) -> tuple[NativePrivilege, ...]:
        value = _require_query_buffer(raw, minimum_size=8)
        count = int.from_bytes(value.data[:4], "little")
        if count > (len(value.data) - 8) // 12:
            raise ValueError("PRIVILEGE_SET count exceeds copied entries")
        privileges: list[NativePrivilege] = []
        for index in range(count):
            offset = 8 + index * 12
            low, high, luid_hex = _parse_luid(value.data, offset)
            attributes = int.from_bytes(value.data[offset + 8 : offset + 12], "little")
            name = bindings.lookup_privilege_name(luid_low=low, luid_high=high)
            if (
                not isinstance(name, str)
                or not name
                or unicodedata.normalize("NFC", name) != name
            ):
                raise ValueError("AccessCheck privilege name is invalid")
            privileges.append(
                NativePrivilege(
                    luid_hex=luid_hex,
                    name=name,
                    attributes_hex=_hex_dword(attributes),
                )
            )
        return tuple(privileges)

    def access_check(
        self,
        *,
        descriptor: bytes,
        token: Win32OwnedTokenHandle,
        object_binding: NativeAccessObjectBinding,
        desired_access_hex: str,
        generic_mapping: NativeGenericMapping,
        context: NativeCallContext,
    ) -> NativeRawAccessCheckResult:
        self._context(context, "access_check")
        owned = self._token_handle(token, context)
        if type(descriptor) is not bytes or not descriptor:
            _raise_validation(context, "access_binding_mismatch")
        if type(object_binding) is not NativeAccessObjectBinding:
            _raise_validation(context, "access_binding_mismatch")
        expected = (
            ("subject", "file")
            if context.operation_ordinal == 0
            else ("immediate_parent", "directory")
        )
        if (
            (object_binding.object_role, object_binding.object_kind) != expected
            or object_binding.descriptor_sha256 != hashlib.sha256(descriptor).hexdigest()
        ):
            _raise_validation(context, "access_binding_mismatch")
        if owned.binding[:2] != (
            "access_check_token_handle",
            "access_check_duplicate",
        ) or owned.last_facts is None:
            _raise_validation(context, "access_token_binding_mismatch")
        if desired_access_hex != MAXIMUM_ALLOWED_HEX:
            _raise_validation(context, "generic_mapping_mismatch")
        if type(generic_mapping) is not NativeGenericMapping:
            _raise_validation(context, "generic_mapping_mismatch")
        try:
            mapping = (
                self._mapping_value(generic_mapping.generic_read_hex),
                self._mapping_value(generic_mapping.generic_write_hex),
                self._mapping_value(generic_mapping.generic_execute_hex),
                self._mapping_value(generic_mapping.generic_all_hex),
            )
            desired_access = self._mapping_value(desired_access_hex)
        except Exception:
            _raise_validation(context, "generic_mapping_mismatch")
        bindings = self._bindings()
        evidence: list[_CopiedNativeEvidence] = []
        try:
            result = bindings.access_check(
                descriptor=descriptor,
                token=owned.raw_handle,
                desired_access=desired_access,
                generic_mapping=mapping,
            )
            if type(result) is not Win32AccessCheckCallResult:
                _raise_protocol(context)
            copied = _copied_evidence(
                result.privilege_set,
                encoding="token_information",
            )
            if copied is not None:
                evidence.append(copied)
            if type(result.api_succeeded) is not bool:
                _raise_protocol(context, prefixes=_prefixes(context, evidence))
            if not result.api_succeeded:
                if (
                    result.access_granted is not None
                    or result.granted_access is not None
                    or isinstance(result.error_code, bool)
                    or not isinstance(result.error_code, int)
                    or not 1 <= result.error_code <= DWORD_MAX
                ):
                    _raise_protocol(context, prefixes=_prefixes(context, evidence))
                return NativeRawAccessCheckResult(
                    api_call_status="failed",
                    access_status="not_evaluated",
                    granted_access_mask_hex=None,
                    access_denial_error_hex=None,
                    api_failure_error_hex=_hex_dword(result.error_code),
                    ordered_privileges_used=None,
                )
            if (
                type(result.access_granted) is not bool
                or isinstance(result.granted_access, bool)
                or not isinstance(result.granted_access, int)
                or not 0 <= result.granted_access <= DWORD_MAX
                or type(result.privilege_set) is not Win32QueryBuffer
            ):
                _raise_protocol(context, prefixes=_prefixes(context, evidence))
            privileges = self._access_privileges(bindings, result.privilege_set)
            if result.access_granted:
                if result.error_code is not None:
                    _raise_protocol(context, prefixes=_prefixes(context, evidence))
                return NativeRawAccessCheckResult(
                    api_call_status="succeeded",
                    access_status="granted",
                    granted_access_mask_hex=_hex_dword(result.granted_access),
                    access_denial_error_hex=None,
                    api_failure_error_hex=None,
                    ordered_privileges_used=privileges,
                )
            if (
                isinstance(result.error_code, bool)
                or not isinstance(result.error_code, int)
                or not 1 <= result.error_code <= DWORD_MAX
            ):
                _raise_protocol(context, prefixes=_prefixes(context, evidence))
            return NativeRawAccessCheckResult(
                api_call_status="succeeded",
                access_status="denied",
                granted_access_mask_hex=_hex_dword(result.granted_access),
                access_denial_error_hex=_hex_dword(result.error_code),
                api_failure_error_hex=None,
                ordered_privileges_used=privileges,
            )
        except PermissionNativeCallError:
            raise
        except Win32ApiError as exc:
            _append_error_evidence(evidence, exc, encoding="token_information")
            _raise_api(context, exc, prefixes=_prefixes(context, evidence))
        except Win32LocalError as exc:
            _append_error_evidence(evidence, exc, encoding="token_information")
            _raise_protocol(context, prefixes=_prefixes(context, evidence))
        except Exception:
            _raise_protocol(context, prefixes=_prefixes(context, evidence))
        raise AssertionError("unreachable")

    def evaluate_held_token_authz(
        self,
        token: Win32OwnedTokenHandle,
        *,
        expected_token_identity: _principal_authz.SuppliedTokenIdentity,
        descriptor: NativeSecurityDescriptor,
        object_binding: NativeAccessObjectBinding,
        authz_bindings: _principal_authz.AuthzNativeBindings,
    ) -> _principal_authz.SuppliedTokenAccessResult:
        """Evaluate supplied evidence while the caller retains the token lifetime.

        This adapter does not establish caller authentication, live object-to-descriptor
        correspondence, or authorization.  A future controller must independently hold
        the authorized resource lifetime and establish object correspondence.
        """

        owned, identity, checked_descriptor, checked_binding = (
            _validate_held_token_authz_inputs(
                self,
                token,
                expected_token_identity,
                descriptor,
                object_binding,
                authz_bindings,
            )
        )
        return _principal_authz.evaluate_supplied_token_access(
            _principal_authz.SuppliedTokenAccessRequest(
                borrowed_token_handle=owned.raw_handle,
                expected_token_identity=identity,
                descriptor=checked_descriptor,
                object_binding=checked_binding,
            ),
            bindings=authz_bindings,
        )
