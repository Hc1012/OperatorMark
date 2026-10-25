"""Fail-closed private-storage preflight with a fixed non-scientific probe.

Serialized evidence contains commitments instead of absolute paths, account
names, or SIDs. Private helper keys prefixed with ``_`` are never serialized.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import ntpath
import os
import re
import shutil
import stat
import subprocess
import sys
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final


DESTINATION_RELATIVE: Final = "artifacts/darcy_branch_c_v1/private_corpus_v1/"
PROTOCOL_SHA256: Final = "a38ef79b6c4bdccf53f599fa08f77e48a111afbf0b2783ad5733bda99996599a"
STATE3_STORAGE_STATE: Final = "state3"
PROBE_BYTES: Final = b"OperatorMark private storage preflight v1\r\n"
PROBE_DIRECTORY: Final = ".operatormark_storage_preflight_probe"
REPARSE_ATTRIBUTE: Final = 0x00000400
ENCRYPTED_ATTRIBUTE: Final = 0x00004000
NOT_CONTENT_INDEXED_ATTRIBUTE: Final = 0x00002000
CLOUD_ATTRIBUTES: Final = (
    0x00001000  # FILE_ATTRIBUTE_OFFLINE
    | 0x00040000  # FILE_ATTRIBUTE_RECALL_ON_OPEN
    | 0x00080000  # FILE_ATTRIBUTE_PINNED
    | 0x00100000  # FILE_ATTRIBUTE_UNPINNED
    | 0x00400000  # FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
    | 0x00800000
    | 0x01000000
)
SYNC_ENVIRONMENT_NAMES: Final = (
    "OneDrive", "OneDriveCommercial", "OneDriveConsumer", "Dropbox",
    "GoogleDrive", "GoogleDriveFS", "iCloudDrive", "Box", "SharePoint",
)
REMOTE_FILESYSTEM_TYPES: Final = frozenset(
    {"afs", "cifs", "davfs", "nfs", "nfs4", "smb", "smb2", "sshfs", "webdav"}
)
WINDOWS_REMOTE_FILESYSTEM_TYPES: Final = frozenset(
    {"afs", "cifs", "nfs", "smb", "smb2", "webdav"}
)
WINDOWS_RESERVED_NAMES: Final = frozenset(
    {"aux", "clock$", "con", "nul", "prn", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
)
RECORD_KEYS: Final = {
    "schema", "state", "protocol_file_sha256", "authorization_file_sha256",
    "implementation_commit", "implementation_tree",
    "repository_root_commitment",
    "destination_relative_path", "destination_absolute_path_commitment",
    "canonical_resolution_method", "path_component_evidence", "volume_evidence",
    "sync_detection_evidence", "git_ignore_evidence", "git_tracking_evidence",
    "permission_evidence", "encryption_evidence", "backup_and_indexing_evidence",
    "probe_evidence", "estimated_accepted_bytes", "projected_audit_bytes",
    "required_free_bytes", "observed_free_bytes", "residual_risks",
    "all_mandatory_checks_passed", "record_self_sha256",
}
PATH_COMPONENT_KEYS: Final = {
    "relative_component", "exists", "is_directory", "file_attributes_hex",
    "is_reparse_point", "reparse_tag_hex", "canonical_path_commitment",
    "volume_identifier_commitment", "file_identity_commitment", "canonical_name_exact",
    "unicode_nfc_exact", "same_volume_as_repository_root", "stable_before_after",
}
VOLUME_KEYS: Final = {
    "allocation_unit_bytes", "capacity_bytes", "free_bytes", "drive_type",
    "filesystem_type", "filesystem_flags_hex", "is_fixed_local",
    "maximum_component_length", "path_length_capability", "case_sensitivity_behavior",
    "volume_identifier_commitment", "volume_root_commitment",
}
SYNC_KEYS: Final = {
    "inspection_complete", "environment_names_checked", "registry_sources_checked",
    "configured_sync_roots", "active_sync_clients", "destination_in_detected_sync_root",
    "cloud_attribute_components",
}
SYNC_ROOT_KEYS: Final = {
    "provider", "source", "exists", "handle_canonicalized",
    "lexical_root_commitment", "canonical_root_commitment",
}
SYNC_CLIENT_KEYS: Final = {"provider", "process_name_commitment", "configured_root_count"}
GIT_IGNORE_KEYS: Final = {
    "candidate_count", "matched", "intended_rule", "all_candidates_matched_intended_rule",
    "candidate_path_commitments", "check_ignore_output_sha256",
}
GIT_TRACKING_KEYS: Final = {
    "tracked_path_count", "submodule_path_count", "nested_worktree_count",
    "nested_repository_marker_count", "candidate_collision_count", "worktree_evidence_sha256",
    "branch", "head_commit", "head_tree", "parent_commit", "tracked_status_clean",
    "staged_status_clean", "untracked_inclusive_status_clean", "status_porcelain_sha256",
}
RIGHTS_KEYS: Final = {"read", "write", "create", "rename", "delete"}
BROAD_ACCESS_KEYS: Final = {
    "principal_class", "principal_commitment", "effective_mask_hex", "effective_read",
    "effective_write", "blocking", "residual_risk",
}
PERMISSION_KEYS: Final = {
    "platform", "inspection_method", "security_descriptor_commitment",
    "owning_principal_commitment", "dacl_present", "inheritance_enabled",
    "current_process_effective_rights", "broad_principal_effective_access",
    "additional_allow_principal_effective_access", "additional_allow_principal_count",
    "blocking_broad_access_count", "broader_local_user_residual_count",
}
ENCRYPTION_KEYS: Final = {
    "filesystem_supports_encryption", "encrypted_ancestor_count",
    "device_encryption_query_outcome", "device_encryption_interpreted_status",
    "device_encryption_evidence_commitment",
    "encryption_presence_establishes_legal_or_cryptographic_protection", "status",
}
BACKUP_KEYS: Final = {
    "administrator_access_may_expose_copies", "backup_software_may_expose_copies",
    "indexing_may_expose_copies", "antivirus_may_expose_copies",
    "crash_dumps_may_expose_copies", "paging_may_expose_copies",
    "local_sync_clients_may_expose_copies", "service_query_evidence_commitments",
    "service_query_outcomes", "registry_query_evidence_commitment", "registry_query_outcome",
    "administrator_backup_malware_physical_access_and_remanence_eliminated", "status",
}
PRIVATE_IDENTITY_KEYS: Final = {
    "schema", "disclosure_classification", "authorization_file_sha256",
    "implementation_commit", "implementation_tree", "expected_parent_commit",
    "repository_root_absolute_path", "destination_absolute_path",
    "acl_inspected_absolute_path", "owning_principal_sid", "record_self_sha256",
}
PROBE_OPERATIONS: Final = [
    "create_directory", "create_file", "write", "flush", "fsync", "read",
    "verify_rename_target_absent", "atomic_directory_rename_no_replace", "read",
    "delete_file", "delete_directories",
]
PROBE_KEYS: Final = {
    "fixed_literal_sha256", "operations", "probe_file_preserved",
    "probe_directory_preserved", "same_volume_verified", "atomic_directory_rename_verified",
    "source_volume_identifier_commitment", "renamed_volume_identifier_commitment", "passed",
}
STATE4_FILE_PERMISSION_METADATA_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.state4_file_permission_metadata.v1"
)
STATE4_FILE_PERMISSION_METADATA_PROFILE: Final = (
    "windows_handle_bound_file_permission_metadata_v1"
)
STATE4_FILE_PERMISSION_METADATA_OBSERVATION_PHASE: Final = "post_protection_metadata"
STATE4_FILE_PERMISSION_METADATA_ROW_COUNT: Final = 232
STATE4_FILE_PERMISSION_METADATA_PRIVATE_OBSERVATION_DOMAIN: Final = (
    "operatormark-state4-private-file-permission-observation-v1"
)
STATE4_FILE_PERMISSION_METADATA_CURRENT_PRINCIPAL_DOMAIN: Final = (
    "operatormark-state4-current-principal-v1"
)
STATE4_FILE_PERMISSION_METADATA_FIXED_PRINCIPAL_DOMAIN: Final = (
    "operatormark-storage-principal-v1"
)
STATE4_FILE_PERMISSION_METADATA_OWNER_DOMAIN: Final = "operatormark-storage-owner-v1"
STATE4_FILE_PERMISSION_METADATA_SECURITY_DESCRIPTOR_DOMAIN: Final = (
    "operatormark-storage-security-descriptor-v1"
)
STATE4_FILE_PERMISSION_METADATA_CONTEXT_KEYS: Final = {
    "protocol_file_sha256", "state4_authorization_file_sha256",
    "implementation_commit", "implementation_tree", "storage_preflight_file_sha256",
    "permission_profile_file_sha256", "repository_root_commitment",
    "volume_identifier_commitment", "accepted_local_user_residual_principal_commitments",
}
STATE4_FILE_PERMISSION_METADATA_EXPECTED_FILE_KEYS: Final = {
    "relative_path", "file_bytes", "file_sha256",
}
STATE4_FILE_PERMISSION_METADATA_KEYS: Final = {
    "schema", "context", "attempt_id", "row_index", "relative_path", "file_bytes",
    "file_sha256", "observation_phase", "path_evidence", "permission_evidence",
    "private_observation_commitment", "record_self_sha256",
}
STATE4_FILE_PERMISSION_METADATA_PATH_KEYS: Final = {
    "canonical_path_commitment", "volume_identifier_commitment",
    "file_identity_commitment", "file_attributes_hex", "reparse_tag_hex",
    "canonical_name_exact", "unicode_nfc_exact", "same_volume_as_storage_context",
}
STATE4_FILE_PERMISSION_METADATA_PERMISSION_KEYS: Final = PERMISSION_KEYS | {
    "current_principal_commitment", "current_process_file_access_mask_hex",
    "current_process_parent_access_mask_hex",
}
STATE4_FILE_PERMISSION_METADATA_ADDITIONAL_ACCESS_KEYS: Final = BROAD_ACCESS_KEYS | {
    "principal_scope",
}
STATE4_FILE_ATTRIBUTE_READONLY: Final = 0x00000001
STATE4_FILE_ATTRIBUTE_DIRECTORY: Final = 0x00000010
STATE4_FILE_PERMISSION_READ_DATA: Final = 0x00000001
STATE4_FILE_PERMISSION_WRITE_DATA: Final = 0x00000002
STATE4_FILE_PERMISSION_APPEND_DATA: Final = 0x00000004
STATE4_FILE_PERMISSION_WRITE_EA: Final = 0x00000010
STATE4_PARENT_PERMISSION_DELETE_CHILD: Final = 0x00000040
STATE4_FILE_PERMISSION_WRITE_ATTRIBUTES: Final = 0x00000100
STATE4_FILE_PERMISSION_DELETE: Final = 0x00010000
STATE4_FILE_PERMISSION_WRITE_DAC: Final = 0x00040000
STATE4_FILE_PERMISSION_WRITE_OWNER: Final = 0x00080000
STATE4_PERMISSION_GENERIC_RIGHTS_MASK: Final = 0xF0000000
CANONICAL_METHOD_WINDOWS: Final = (
    "GetFinalPathNameByHandleW(FILE_NAME_NORMALIZED) and GetFileInformationByHandle "
    "for every existing component; append only the validated fixed non-existing suffix"
)
CANONICAL_METHOD_POSIX: Final = (
    "open(O_NOFOLLOW) plus fstat and realpath for every existing component; append only "
    "the validated fixed non-existing suffix"
)


class StoragePreflightError(RuntimeError):
    """Raised when a mandatory private-storage check fails closed."""


@dataclass(frozen=True, slots=True)
class StoragePreflightRequest:
    state: str
    protocol_file_sha256: str
    authorization_file_sha256: str
    implementation_commit: str
    implementation_tree: str
    expected_parent_commit: str
    estimated_accepted_bytes: int
    projected_audit_bytes: int
    destination_relative_path: str = DESTINATION_RELATIVE
    attempt_relative_paths: tuple[str, ...] = ()
    expected_branch: str = "main"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise StoragePreflightError(f"{name} must be lowercase SHA-256 hexadecimal")


def _require_git_oid(value: str, *, name: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise StoragePreflightError(f"{name} must be a lowercase 40-hex Git object ID")


def _commit(domain: str, authorization_hash: str, value: str) -> str:
    return _commit_bytes(domain, authorization_hash, value.encode("utf-8"))


def _commit_bytes(domain: str, authorization_hash: str, value: bytes) -> str:
    return _sha(domain.encode() + b"\n" + authorization_hash.encode("ascii") + b"\n" + value)


def required_free_bytes(estimated_accepted_bytes: int, projected_audit_bytes: int) -> int:
    values = (estimated_accepted_bytes, projected_audit_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise StoragePreflightError("storage estimates must be non-negative integers")
    return max(5 * 2**30, 4 * estimated_accepted_bytes + projected_audit_bytes + 2**30)


def _run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="strict",
    )


def discover_repository_root(start: str | Path) -> Path:
    start_path = Path(start)
    result = _run_git(start_path, "rev-parse", "--show-toplevel")
    inside = _run_git(start_path, "rev-parse", "--is-inside-work-tree")
    if result.returncode != 0 or inside.returncode != 0 or inside.stdout.strip() != "true":
        raise StoragePreflightError("unable to discover repository root from verified Git metadata")
    root = Path(result.stdout.strip()).absolute()
    if not root.is_dir():
        raise StoragePreflightError("Git repository root is not an existing directory")
    return root


def _bind_repository_root_to_loaded_module(root: Path) -> None:
    """Require Git discovery and the imported storage module to name one checkout."""
    loaded_module = Path(__file__).resolve()
    loaded_root = loaded_module.parents[2]
    expected_module = root / "operatormark" / "corpus" / "storage.py"
    try:
        root_matches = os.path.samefile(root, loaded_root)
        module_matches = os.path.samefile(loaded_module, expected_module)
    except OSError as exc:
        raise StoragePreflightError(
            "unable to bind the discovered repository to the loaded storage module"
        ) from exc
    if not root_matches or not module_matches:
        raise StoragePreflightError(
            "discovered repository is not the checkout containing the loaded storage module"
        )
    if os.name == "nt":
        root_snapshot = _windows_handle_snapshot(root, open_reparse_point=False)
        loaded_snapshot = _windows_handle_snapshot(loaded_root, open_reparse_point=False)
        if (
            root_snapshot["_volume_identifier"] != loaded_snapshot["_volume_identifier"]
            or root_snapshot["_file_identity"] != loaded_snapshot["_file_identity"]
            or not _windows_exact_path_equal(
                root_snapshot["_canonical_path"], loaded_snapshot["_canonical_path"]
            )
        ):
            raise StoragePreflightError(
                "handle-backed loaded-module repository identity does not match Git discovery"
            )


def _validate_relative_destination(value: str) -> PurePosixPath:
    if value != DESTINATION_RELATIVE:
        raise StoragePreflightError("destination must equal the frozen repository-relative base")
    return _validate_relative_path(value, name="destination", require_trailing_slash=True)


def _validate_relative_path(
    value: str, *, name: str, require_trailing_slash: bool = False
) -> PurePosixPath:
    if not value or "\\" in value or "\x00" in value:
        raise StoragePreflightError(f"{name} contains empty, NUL, or non-canonical separator syntax")
    if require_trailing_slash and not value.endswith("/"):
        raise StoragePreflightError(f"{name} must retain its frozen trailing slash")
    if not require_trailing_slash and value.endswith("/"):
        raise StoragePreflightError(f"{name} must not have a trailing slash")
    body = value[:-1] if require_trailing_slash else value
    raw_parts = body.split("/")
    if not body or any(part in {"", ".", ".."} for part in raw_parts):
        raise StoragePreflightError(f"{name} contains non-canonical empty or traversal syntax")
    canonical_value = "/".join(raw_parts) + ("/" if require_trailing_slash else "")
    if value != canonical_value:
        raise StoragePreflightError(f"{name} is not in canonical slash-separated form")
    path = PurePosixPath(body)
    if path.is_absolute():
        raise StoragePreflightError(f"{name} contains traversal or absolute syntax")
    invalid_windows_characters = frozenset('<>:"|?*')
    for part in raw_parts:
        if (
            any(character in invalid_windows_characters for character in part)
            or part != unicodedata.normalize("NFC", part)
            or part.endswith((" ", ".")) or any(ord(character) < 32 for character in part)
        ):
            raise StoragePreflightError(f"{name} contains alternate-stream, Unicode, or alias syntax")
        if len(part.encode("utf-16-le")) // 2 > 255:
            raise StoragePreflightError(f"{name} contains a component exceeding the Windows limit")
        if part.split(".", 1)[0].casefold() in WINDOWS_RESERVED_NAMES:
            raise StoragePreflightError(f"{name} contains a reserved device name")
    return path


def _validate_attempt_paths(
    values: Sequence[str], destination: PurePosixPath
) -> tuple[PurePosixPath, ...]:
    if isinstance(values, (str, bytes)):
        raise StoragePreflightError("attempt paths must be a sequence of relative path strings")
    result: list[PurePosixPath] = []
    seen: set[str] = set()
    for value in values:
        candidate = _validate_relative_path(value, name="attempt path")
        if candidate.parts[: len(destination.parts)] != destination.parts or candidate == destination:
            raise StoragePreflightError("attempt path must be a strict child of the frozen destination")
        folded = unicodedata.normalize("NFC", candidate.as_posix()).casefold()
        if folded in seen:
            raise StoragePreflightError("attempt paths contain a case or Unicode alias collision")
        seen.add(folded)
        result.append(candidate)
    return tuple(result)


def _existing_ancestors(root: Path, relative: PurePosixPath) -> list[Path]:
    paths = [root]
    current = root
    for part in relative.parts:
        candidate = current / part
        if not os.path.lexists(candidate):
            break
        paths.append(candidate)
        current = candidate
    return paths


def _reject_windows_namespace_or_alias(path: Path) -> None:
    value = str(path)
    if value != unicodedata.normalize("NFC", value) or "\x00" in value:
        raise StoragePreflightError("absolute path has Unicode-normalization or NUL ambiguity")
    lowered = value.replace("/", "\\").casefold()
    if lowered.startswith(("\\\\", "\\?\\", "\\.\\", "\\??\\")) or "globalroot" in lowered:
        raise StoragePreflightError("UNC or Windows device namespace is prohibited")
    drive, tail = ntpath.splitdrive(value)
    if not re.fullmatch(r"[A-Za-z]:", drive) or not tail.startswith(("\\", "/")):
        raise StoragePreflightError("path is not a canonical local DOS-drive absolute path")
    if ":" in tail:
        raise StoragePreflightError("alternate data stream syntax is prohibited")
    for part in PureWindowsPath(value).parts[1:]:
        if part.endswith((" ", ".")) or part != unicodedata.normalize("NFC", part):
            raise StoragePreflightError("path component has case/Unicode alias ambiguity")
        if part.split(".", 1)[0].casefold() in WINDOWS_RESERVED_NAMES:
            raise StoragePreflightError("reserved device path component is prohibited")


def _strip_extended_windows_prefix(value: str) -> str:
    normalized = value.replace("/", "\\")
    if normalized.casefold().startswith("\\\\?\\unc\\"):
        raise StoragePreflightError("canonical handle resolved to UNC namespace")
    if normalized.startswith("\\\\?\\"):
        normalized = normalized[4:]
    if normalized.startswith(("\\\\", "\\.\\", "\\?\\")):
        raise StoragePreflightError("canonical handle resolved to a prohibited namespace")
    return ntpath.normpath(normalized)


def _windows_exact_path_equal(expected: str, observed: str) -> bool:
    expected_path = ntpath.normpath(expected.replace("/", "\\"))
    observed_path = ntpath.normpath(observed.replace("/", "\\"))
    if ntpath.normcase(expected_path) != ntpath.normcase(observed_path):
        return False
    expected_drive, expected_tail = ntpath.splitdrive(expected_path)
    observed_drive, observed_tail = ntpath.splitdrive(observed_path)
    return expected_drive.casefold() == observed_drive.casefold() and expected_tail == observed_tail


def _windows_handle_snapshot(
    path: Path, *, open_reparse_point: bool = True
) -> dict[str, Any]:
    from ctypes import wintypes

    class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD), ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME), ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD), ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD), ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD), ("nFileIndexLow", wintypes.DWORD),
        ]

    class FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
        _fields_ = [("FileAttributes", wintypes.DWORD), ("ReparseTag", wintypes.DWORD)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    get_final = kernel32.GetFinalPathNameByHandleW
    get_final.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_final.restype = wintypes.DWORD
    get_info = kernel32.GetFileInformationByHandle
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(BY_HANDLE_FILE_INFORMATION)]
    get_info.restype = wintypes.BOOL
    get_info_ex = kernel32.GetFileInformationByHandleEx
    get_info_ex.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    get_info_ex.restype = wintypes.BOOL
    flags = 0x02000000 | (0x00200000 if open_reparse_point else 0)
    handle = create_file(str(path), 0, 0x1 | 0x2 | 0x4, None, 3, flags, None)
    if handle == wintypes.HANDLE(-1).value:
        error = ctypes.get_last_error()
        raise StoragePreflightError(f"CreateFileW metadata handle failed with Windows error {error}")
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        length = int(get_final(handle, buffer, len(buffer), 0))
        if length == 0 or length >= len(buffer):
            error = ctypes.get_last_error()
            raise StoragePreflightError(
                f"GetFinalPathNameByHandleW failed or exceeded buffer with Windows error {error}"
            )
        basic = BY_HANDLE_FILE_INFORMATION()
        if not get_info(handle, ctypes.byref(basic)):
            raise StoragePreflightError(
                f"GetFileInformationByHandle failed with Windows error {ctypes.get_last_error()}"
            )
        tag = FILE_ATTRIBUTE_TAG_INFO()
        if not get_info_ex(handle, 9, ctypes.byref(tag), ctypes.sizeof(tag)):
            raise StoragePreflightError(
                f"FileAttributeTagInfo inspection failed with Windows error {ctypes.get_last_error()}"
            )
        return {
            "_canonical_path": _strip_extended_windows_prefix(buffer.value),
            "_volume_identifier": f"{int(basic.dwVolumeSerialNumber):08x}",
            "_file_identity": f"{int(basic.nFileIndexHigh):08x}{int(basic.nFileIndexLow):08x}",
            "file_attributes": int(tag.FileAttributes), "reparse_tag": int(tag.ReparseTag),
        }
    finally:
        close_handle(handle)


def _sibling_alias_is_exact(path: Path) -> bool:
    expected = path.name
    folded = unicodedata.normalize("NFC", expected).casefold()
    try:
        matches = [entry.name for entry in os.scandir(path.parent)
                   if unicodedata.normalize("NFC", entry.name).casefold() == folded]
    except OSError as exc:
        raise StoragePreflightError("unable to inspect directory entries for case/Unicode aliases") from exc
    return matches == [expected]


def _missing_component_has_no_alias(parent: Path, component: str) -> bool:
    """Prove the first absent suffix component has no case/NFC directory alias."""
    folded = unicodedata.normalize("NFC", component).casefold()
    try:
        matches = [
            entry.name
            for entry in os.scandir(parent)
            if unicodedata.normalize("NFC", entry.name).casefold() == folded
        ]
    except OSError as exc:
        raise StoragePreflightError(
            "unable to inspect the deepest existing directory for missing-suffix aliases"
        ) from exc
    return matches == []


def _inspect_path_components_private(
    root: Path, relative: PurePosixPath, authorization_hash: str
) -> tuple[list[dict[str, Any]], Path, str]:
    paths = _existing_ancestors(root, relative)
    evidence: list[dict[str, Any]] = []
    canonical_existing: Path | None = None
    root_volume: str | None = None
    root_device: int | None = None
    for index, path in enumerate(paths):
        item = path.lstat()
        attributes = int(getattr(item, "st_file_attributes", 0))
        is_reparse = bool(attributes & REPARSE_ATTRIBUTE)
        is_link = stat.S_ISLNK(item.st_mode)
        if is_link or is_reparse:
            raise StoragePreflightError("destination ancestor is a link or Windows reparse point")
        if not stat.S_ISDIR(item.st_mode):
            raise StoragePreflightError("destination ancestor is not a directory")
        relative_component = "." if index == 0 else path.relative_to(root).as_posix()
        if os.name == "nt":
            _reject_windows_namespace_or_alias(path.absolute())
            _windows_case_sensitivity(path)
            private = _windows_handle_snapshot(path)
            attributes = private["file_attributes"]
            if attributes & REPARSE_ATTRIBUTE or private["reparse_tag"] != 0:
                raise StoragePreflightError("Windows handle reports a reparse point or reparse tag")
            canonical_text = private["_canonical_path"]
            exact = _windows_exact_path_equal(str(path.absolute()), canonical_text)
            if not exact or (index > 0 and not _sibling_alias_is_exact(path)):
                raise StoragePreflightError("destination ancestor has case or Unicode alias ambiguity")
            volume = private["_volume_identifier"]
            if root_volume is None:
                root_volume = volume
            if volume != root_volume:
                raise StoragePreflightError("destination ancestor crosses a volume or mount point")
            canonical_existing = Path(canonical_text)
            file_identity = private["_file_identity"]
            reparse_tag = private["reparse_tag"]
        else:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(path, flags)
            except OSError as exc:
                raise StoragePreflightError("unable to open destination ancestor without following links") from exc
            try:
                handled = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            canonical_text = os.path.realpath(path)
            if canonical_text != str(path.absolute()):
                raise StoragePreflightError("destination ancestor has ambiguous canonical resolution")
            if root_device is None:
                root_device = int(handled.st_dev)
            if int(handled.st_dev) != root_device:
                raise StoragePreflightError("destination ancestor crosses a mount point")
            volume = str(handled.st_dev)
            root_volume = str(root_device)
            file_identity = f"{int(handled.st_dev)}:{int(handled.st_ino)}"
            reparse_tag = 0
            exact = True
            canonical_existing = Path(canonical_text)
        evidence.append({
            "canonical_name_exact": exact,
            "canonical_path_commitment": _commit("operatormark-storage-component-path-v1",
                                                   authorization_hash, os.path.normcase(canonical_text)),
            "exists": True, "file_attributes_hex": hex(attributes),
            "file_identity_commitment": _commit("operatormark-storage-component-file-id-v1",
                                                  authorization_hash, file_identity),
            "is_directory": True, "is_reparse_point": False,
            "relative_component": relative_component, "reparse_tag_hex": hex(reparse_tag),
            "same_volume_as_repository_root": True, "stable_before_after": False,
            "unicode_nfc_exact": canonical_text == unicodedata.normalize("NFC", canonical_text),
            "volume_identifier_commitment": _commit(
                "operatormark-private-volume-v1", authorization_hash, volume
            ),
        })
    if canonical_existing is None or root_volume is None:
        raise StoragePreflightError("no existing repository component could be resolved")
    missing_parts = relative.parts[len(paths) - 1:]
    if missing_parts:
        if os.name == "nt":
            _windows_case_sensitivity(canonical_existing)
        if not _missing_component_has_no_alias(canonical_existing, missing_parts[0]):
            raise StoragePreflightError(
                "missing destination suffix has a case or Unicode alias collision"
            )
    canonical_destination = canonical_existing.joinpath(*missing_parts)
    if os.name == "nt":
        _reject_windows_namespace_or_alias(canonical_destination.absolute())
    return evidence, canonical_destination, root_volume


def inspect_path_components(
    root: Path, relative: PurePosixPath, authorization_hash: str = "0" * 64
) -> list[dict[str, Any]]:
    _require_sha(authorization_hash, name="authorization_hash")
    evidence, _, _ = _inspect_path_components_private(root, relative, authorization_hash)
    return evidence


def _windows_case_sensitivity(path: Path) -> str:
    from ctypes import wintypes

    class FILE_CASE_SENSITIVE_INFO(ctypes.Structure):
        _fields_ = [("Flags", wintypes.DWORD)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    get_info_ex = kernel32.GetFileInformationByHandleEx
    get_info_ex.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    get_info_ex.restype = wintypes.BOOL
    handle = create_file(str(path), 0, 0x1 | 0x2 | 0x4, None, 3, 0x02000000 | 0x00200000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise StoragePreflightError("unable to open repository root for case-sensitivity inspection")
    try:
        info = FILE_CASE_SENSITIVE_INFO()
        if get_info_ex(handle, 23, ctypes.byref(info), ctypes.sizeof(info)):
            return "enabled" if info.Flags & 0x1 else "disabled"
        error = ctypes.get_last_error()
        if error in {1, 50, 87}:
            return "query_unsupported_exact_directory_entry_scan_enforced"
        raise StoragePreflightError(f"directory case-sensitivity inspection failed with Windows error {error}")
    finally:
        close_handle(handle)


def _windows_volume(path: Path) -> dict[str, Any]:
    from ctypes import wintypes

    if os.name != "nt":
        raise StoragePreflightError("Windows volume inspection requested on a non-Windows platform")
    _reject_windows_namespace_or_alias(path.absolute())
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    volume_path = ctypes.create_unicode_buffer(32768)
    get_volume_path = kernel32.GetVolumePathNameW
    get_volume_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_volume_path.restype = wintypes.BOOL
    if not get_volume_path(str(path), volume_path, len(volume_path)):
        raise StoragePreflightError("GetVolumePathNameW failed")
    if volume_path.value.startswith("\\\\"):
        raise StoragePreflightError("UNC volume roots are prohibited")
    get_drive_type = kernel32.GetDriveTypeW
    get_drive_type.argtypes = [wintypes.LPCWSTR]
    get_drive_type.restype = wintypes.UINT
    drive_type = int(get_drive_type(volume_path.value))
    if drive_type != 3:
        raise StoragePreflightError("destination volume is not a fixed local drive")
    name = ctypes.create_unicode_buffer(32768)
    filesystem = ctypes.create_unicode_buffer(32768)
    serial = wintypes.DWORD()
    maximum_component = wintypes.DWORD()
    flags = wintypes.DWORD()
    get_volume_information = kernel32.GetVolumeInformationW
    get_volume_information.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
        wintypes.LPWSTR, wintypes.DWORD]
    get_volume_information.restype = wintypes.BOOL
    if not get_volume_information(volume_path.value, name, len(name), ctypes.byref(serial),
                                  ctypes.byref(maximum_component), ctypes.byref(flags),
                                  filesystem, len(filesystem)):
        raise StoragePreflightError("GetVolumeInformationW failed")
    if filesystem.value.casefold() in WINDOWS_REMOTE_FILESYSTEM_TYPES:
        raise StoragePreflightError("remote filesystem type is prohibited")
    if not (int(flags.value) & 0x00000002):
        raise StoragePreflightError("filesystem does not preserve filename case")
    sectors_per_cluster = wintypes.DWORD()
    bytes_per_sector = wintypes.DWORD()
    free_clusters = wintypes.DWORD()
    total_clusters = wintypes.DWORD()
    get_disk_free_space = kernel32.GetDiskFreeSpaceW
    get_disk_free_space.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD)]
    get_disk_free_space.restype = wintypes.BOOL
    if not get_disk_free_space(volume_path.value, ctypes.byref(sectors_per_cluster),
                               ctypes.byref(bytes_per_sector), ctypes.byref(free_clusters),
                               ctypes.byref(total_clusters)):
        raise StoragePreflightError("GetDiskFreeSpaceW failed")
    free_to_caller = ctypes.c_ulonglong()
    total_bytes = ctypes.c_ulonglong()
    total_free = ctypes.c_ulonglong()
    get_disk_free_space_ex = kernel32.GetDiskFreeSpaceExW
    get_disk_free_space_ex.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong)]
    get_disk_free_space_ex.restype = wintypes.BOOL
    if not get_disk_free_space_ex(volume_path.value, ctypes.byref(free_to_caller),
                                  ctypes.byref(total_bytes), ctypes.byref(total_free)):
        raise StoragePreflightError("GetDiskFreeSpaceExW failed")
    guid = ctypes.create_unicode_buffer(32768)
    get_volume_name = kernel32.GetVolumeNameForVolumeMountPointW
    get_volume_name.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_volume_name.restype = wintypes.BOOL
    if not get_volume_name(volume_path.value, guid, len(guid)):
        raise StoragePreflightError("unable to establish canonical fixed-volume identity")
    return {
        "allocation_unit_bytes": int(sectors_per_cluster.value * bytes_per_sector.value),
        "capacity_bytes": int(total_bytes.value),
        "case_sensitivity_behavior": _windows_case_sensitivity(path), "drive_type": drive_type,
        "filesystem_flags_hex": hex(int(flags.value)), "filesystem_type": filesystem.value,
        "free_bytes": int(free_to_caller.value), "is_fixed_local": True,
        "maximum_component_length": int(maximum_component.value),
        "path_length_capability": "not_tested_no_extended_length_path_claim",
        "_volume_identifier": f"{int(serial.value):08x}",
        "_volume_guid": guid.value,
        "_volume_root": volume_path.value,
        "_filesystem_flags": int(flags.value),
    }


def _linux_volume(path: Path) -> dict[str, Any]:
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        raise StoragePreflightError("non-Windows fixed-local volume type cannot be established")
    resolved = str(path.resolve(strict=True))
    candidates: list[tuple[str, str, str]] = []
    for line in mountinfo.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError as exc:
            raise StoragePreflightError("invalid operating-system mount metadata") from exc
        mount_point = fields[4].replace("\\040", " ")
        filesystem = fields[separator + 1]
        source = fields[separator + 2]
        if resolved == mount_point or resolved.startswith(mount_point.rstrip("/") + "/"):
            candidates.append((mount_point, filesystem, source))
    if not candidates:
        raise StoragePreflightError("unable to bind destination to a mounted filesystem")
    mount_point, filesystem, source = max(candidates, key=lambda item: len(item[0]))
    if filesystem.casefold() in REMOTE_FILESYSTEM_TYPES or filesystem.casefold() in {"tmpfs", "ramfs"}:
        raise StoragePreflightError("remote, removable, or memory-backed filesystem is prohibited")
    if source in {"none", "-"}:
        raise StoragePreflightError("filesystem locality cannot be established")
    usage = shutil.disk_usage(path)
    statvfs = os.statvfs(path)
    return {
        "allocation_unit_bytes": int(statvfs.f_frsize), "capacity_bytes": int(usage.total),
        "case_sensitivity_behavior": "case_sensitive_exact_component_match_enforced",
        "drive_type": "mounted_local_block_device", "filesystem_flags_hex": "not_applicable",
        "filesystem_type": filesystem, "free_bytes": int(usage.free), "is_fixed_local": True,
        "maximum_component_length": int(os.pathconf(path, "PC_NAME_MAX")),
        "path_length_capability": f"pathconf:{int(os.pathconf(path, 'PC_PATH_MAX'))}",
        "_volume_identifier": f"{os.stat(path).st_dev}:{source}", "_volume_root": mount_point,
        "_filesystem_flags": 0,
    }


def _volume_evidence(path: Path) -> dict[str, Any]:
    return _windows_volume(path) if os.name == "nt" else _linux_volume(path)


def _path_is_within_windows(candidate: str, root: str) -> bool:
    normalized_candidate = ntpath.normcase(ntpath.normpath(candidate))
    normalized_root = ntpath.normcase(ntpath.normpath(root))
    try:
        return ntpath.commonpath((normalized_candidate, normalized_root)) == normalized_root
    except ValueError:
        return False


def _registry_sync_roots() -> tuple[list[tuple[str, str, str]], list[str]]:
    if os.name != "nt":
        return [], []
    import winreg

    checked: list[str] = []
    roots: list[tuple[str, str, str]] = []

    def open_optional(hive: int, subkey: str, evidence_label: str):
        checked.append(evidence_label)
        try:
            return winreg.OpenKey(hive, subkey)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StoragePreflightError("unable to inspect configured synchronization roots") from exc

    accounts_path = r"Software\Microsoft\OneDrive\Accounts"
    accounts = open_optional(winreg.HKEY_CURRENT_USER, accounts_path, "HKCU:OneDrive.Accounts")
    if accounts is not None:
        with accounts:
            index = 0
            while True:
                try:
                    account_name = winreg.EnumKey(accounts, index)
                except OSError as exc:
                    if getattr(exc, "winerror", None) == 259:
                        break
                    raise StoragePreflightError("unable to enumerate OneDrive accounts") from exc
                index += 1
                account = open_optional(
                    winreg.HKEY_CURRENT_USER,
                    accounts_path + "\\" + account_name,
                    "HKCU:OneDrive.Account",
                )
                if account is None:
                    continue
                with account:
                    try:
                        value, kind = winreg.QueryValueEx(account, "UserFolder")
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        raise StoragePreflightError("unable to inspect OneDrive account root") from exc
                    if kind in {winreg.REG_SZ, winreg.REG_EXPAND_SZ} and isinstance(value, str):
                        roots.append(("onedrive", "registry", os.path.expandvars(value)))
    manager_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\SyncRootManager"
    manager = open_optional(
        winreg.HKEY_CURRENT_USER, manager_path, "HKCU:SyncRootManager"
    )
    if manager is not None:
        with manager:
            provider_index = 0
            while True:
                try:
                    provider_name = winreg.EnumKey(manager, provider_index)
                except OSError as exc:
                    if getattr(exc, "winerror", None) == 259:
                        break
                    raise StoragePreflightError("unable to enumerate Windows SyncRootManager") from exc
                provider_index += 1
                provider = open_optional(
                    winreg.HKEY_CURRENT_USER,
                    manager_path + "\\" + provider_name,
                    "HKCU:SyncRootManager.Provider",
                )
                if provider is None:
                    continue
                folded_provider = provider_name.casefold()
                provider_class = next((name for name in
                    ("onedrive", "dropbox", "google", "icloud", "box", "sharepoint")
                    if name in folded_provider), "registered_sync_provider")
                provider_keys = [(provider, "sync_root_manager")]
                user_roots = open_optional(
                    winreg.HKEY_CURRENT_USER,
                    manager_path + "\\" + provider_name + "\\UserSyncRoots",
                    "HKCU:SyncRootManager.Provider.UserSyncRoots",
                )
                if user_roots is not None:
                    provider_keys.append((user_roots, "sync_root_manager_user_roots"))
                try:
                    for provider_key, source_name in provider_keys:
                        value_index = 0
                        while True:
                            try:
                                _, value, kind = winreg.EnumValue(provider_key, value_index)
                            except OSError as exc:
                                if getattr(exc, "winerror", None) == 259:
                                    break
                                raise StoragePreflightError(
                                    "unable to enumerate registered sync-root values"
                                ) from exc
                            value_index += 1
                            values = value if isinstance(value, list) else [value]
                            for candidate in values:
                                if (
                                    kind in {winreg.REG_SZ, winreg.REG_EXPAND_SZ, winreg.REG_MULTI_SZ}
                                    and isinstance(candidate, str)
                                    and ntpath.isabs(os.path.expandvars(candidate))
                                ):
                                    roots.append(
                                        (provider_class, source_name, os.path.expandvars(candidate))
                                    )
                finally:
                    if user_roots is not None:
                        user_roots.Close()
                    provider.Close()

    shell_folders_path = (
        r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
    )
    shell_folders = open_optional(
        winreg.HKEY_CURRENT_USER, shell_folders_path, "HKCU:UserShellFolders"
    )
    if shell_folders is not None:
        with shell_folders:
            value_index = 0
            while True:
                try:
                    value_name, value, kind = winreg.EnumValue(shell_folders, value_index)
                except OSError as exc:
                    if getattr(exc, "winerror", None) == 259:
                        break
                    raise StoragePreflightError("unable to enumerate synchronized shell folders") from exc
                value_index += 1
                folded_name = value_name.casefold()
                provider_class = next(
                    (
                        name
                        for name in ("onedrive", "dropbox", "google", "icloud", "box", "sharepoint")
                        if name in folded_name
                    ),
                    None,
                )
                if (
                    provider_class is not None
                    and kind in {winreg.REG_SZ, winreg.REG_EXPAND_SZ}
                    and isinstance(value, str)
                    and ntpath.isabs(os.path.expandvars(value))
                ):
                    roots.append(
                        (provider_class, "user_shell_folders", os.path.expandvars(value))
                    )
    return roots, sorted(set(checked))


def _active_sync_clients() -> list[tuple[str, str]]:
    if os.name != "nt":
        return []
    try:
        result = subprocess.run(["tasklist.exe", "/FO", "CSV", "/NH"], check=False,
            capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        raise StoragePreflightError("unable to inspect active synchronization clients") from exc
    if result.returncode != 0:
        raise StoragePreflightError("active synchronization-client inspection failed")
    process_providers = {
        "onedrive.exe": "onedrive", "dropbox.exe": "dropbox",
        "googledrivefs.exe": "google", "iclouddrive.exe": "icloud",
        "icloudservices.exe": "icloud", "box.exe": "box", "boxdrive.exe": "box",
        "syncthing.exe": "syncthing", "nextcloud.exe": "nextcloud",
        "owncloud.exe": "owncloud", "resilio-sync.exe": "resilio",
        "seafile-applet.exe": "seafile", "megasync.exe": "mega",
        "pcloud.exe": "pcloud", "insync.exe": "insync",
    }
    synchronization_markers = (
        "onedrive", "dropbox", "drivefs", "googledrive", "icloud", "boxdrive",
        "syncthing", "nextcloud", "owncloud", "resilio", "seafile", "megasync",
        "pcloud", "insync", "mobsync",
    )
    active: list[tuple[str, str]] = []
    for row in csv.reader(result.stdout.splitlines()):
        if not row:
            continue
        name = row[0].strip().casefold()
        if name in process_providers:
            active.append((process_providers[name], name))
        elif any(marker in name for marker in synchronization_markers):
            active.append(("unclassified_sync_client", name))
    return sorted(set(active))


def _sync_evidence(
    destination: Path, environ: Mapping[str, str], authorization_hash: str = "0" * 64
) -> dict[str, Any]:
    _require_sha(authorization_hash, name="authorization_hash")
    if os.name != "nt":
        raise StoragePreflightError(
            "complete configured synchronization-root inspection is not implemented off Windows"
        )
    configured: list[tuple[str, str, str]] = []
    for name in SYNC_ENVIRONMENT_NAMES:
        value = environ.get(name)
        if value:
            provider = next((candidate for candidate in
                ("onedrive", "dropbox", "google", "icloud", "box", "sharepoint")
                if candidate in name.casefold()), "configured_sync_provider")
            configured.append((provider, "environment", os.path.expandvars(value)))
    registry_roots, registry_checked = _registry_sync_roots()
    configured.extend(registry_roots)
    active = _active_sync_clients()
    active_providers = {provider for provider, _ in active}
    deduplicated: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    for provider, source, root in configured:
        normalized = ntpath.normpath(root)
        if not ntpath.isabs(normalized):
            raise StoragePreflightError("configured synchronization root is not absolute")
        deduplicated[(provider, source, ntpath.normcase(normalized))] = (
            provider,
            source,
            normalized,
        )
    destination_text = ntpath.normpath(str(destination))
    sync_records: list[dict[str, Any]] = []
    for provider, source, root in deduplicated.values():
        lowered_root = root.replace("/", "\\").casefold()
        if lowered_root.startswith(("\\\\", "\\?\\", "\\.\\", "\\??\\")):
            raise StoragePreflightError(
                "configured synchronization root uses an unresolvable or remote namespace"
            )
        if _path_is_within_windows(destination_text, root):
            raise StoragePreflightError(f"destination is inside detected synchronization root {provider}")
        exists = os.path.lexists(root)
        canonical_root: str | None = None
        if exists:
            try:
                snapshot = _windows_handle_snapshot(
                    Path(root), open_reparse_point=False
                )
            except (OSError, StoragePreflightError) as exc:
                raise StoragePreflightError(
                    f"existing configured {provider} synchronization root is not handle-resolvable"
                ) from exc
            if not (int(snapshot["file_attributes"]) & 0x00000010):
                raise StoragePreflightError("configured synchronization root is not a directory")
            canonical_root = str(snapshot["_canonical_path"])
            if _path_is_within_windows(destination_text, canonical_root):
                raise StoragePreflightError(
                    f"destination is inside handle-canonical {provider} synchronization root"
                )
        elif provider in active_providers:
            raise StoragePreflightError(
                f"active {provider} synchronization root does not exist or cannot be resolved"
            )
        sync_records.append({
            "canonical_root_commitment": (
                _commit(
                    "operatormark-storage-sync-canonical-root-v1",
                    authorization_hash,
                    ntpath.normcase(canonical_root),
                )
                if canonical_root is not None
                else None
            ),
            "exists": exists,
            "handle_canonicalized": canonical_root is not None,
            "lexical_root_commitment": _commit(
                "operatormark-storage-sync-lexical-root-v1",
                authorization_hash,
                ntpath.normcase(root),
            ),
            "provider": provider,
            "source": source,
        })
    sync_records.sort(
        key=lambda item: (
            item["provider"], item["source"], item["lexical_root_commitment"]
        )
    )
    active_records: list[dict[str, Any]] = []
    for provider, process_name in active:
        root_count = sum(
            1
            for item in sync_records
            if item["provider"] == provider
            and item["exists"] is True
            and item["handle_canonicalized"] is True
        )
        if root_count == 0:
            raise StoragePreflightError(
                f"active {provider} synchronization client has no inspectable configured root"
            )
        active_records.append({
            "configured_root_count": root_count,
            "process_name_commitment": _commit("operatormark-storage-sync-process-v1",
                                                authorization_hash, process_name),
            "provider": provider,
        })
    active_records.sort(
        key=lambda item: (item["provider"], item["process_name_commitment"])
    )
    cloud_components: list[str] = []
    root = Path(destination.anchor)
    if not root.exists():
        raise StoragePreflightError("destination drive root cannot be inspected for cloud attributes")
    current = root
    for part in destination.parts[1:]:
        candidate = current / part
        if not os.path.lexists(candidate):
            break
        attributes = int(getattr(candidate.stat(follow_symlinks=False), "st_file_attributes", 0))
        if attributes & CLOUD_ATTRIBUTES:
            cloud_components.append(_commit("operatormark-storage-cloud-component-v1",
                                            authorization_hash, ntpath.normcase(str(candidate))))
        current = candidate
    if cloud_components:
        raise StoragePreflightError("cloud placeholder, offline, or recall-on-access attributes detected")
    return {
        "active_sync_clients": active_records, "cloud_attribute_components": [],
        "configured_sync_roots": sync_records, "destination_in_detected_sync_root": False,
        "environment_names_checked": list(SYNC_ENVIRONMENT_NAMES), "inspection_complete": True,
        "registry_sources_checked": registry_checked,
    }


def _git_repository_binding(
    root: Path,
    *,
    expected_branch: str,
    expected_head: str,
    expected_tree: str,
    expected_parent: str,
) -> dict[str, Any]:
    commands = {
        "branch": ("symbolic-ref", "--quiet", "--short", "HEAD"),
        "head": ("rev-parse", "HEAD"),
        "tree": ("rev-parse", "HEAD^{tree}"),
        "parent": ("rev-parse", "HEAD^"),
    }
    observed: dict[str, str] = {}
    for name, arguments in commands.items():
        result = _run_git(root, *arguments)
        if result.returncode != 0 or not result.stdout.strip():
            raise StoragePreflightError(f"unable to establish Git {name} binding")
        observed[name] = result.stdout.strip()
    expected = {
        "branch": expected_branch,
        "head": expected_head,
        "tree": expected_tree,
        "parent": expected_parent,
    }
    if observed != expected:
        raise StoragePreflightError("repository branch, HEAD, tree, or parent binding mismatch")
    tracked = _run_git(root, "diff", "--quiet", "--")
    staged = _run_git(root, "diff", "--cached", "--quiet", "--")
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if tracked.returncode != 0 or staged.returncode != 0:
        raise StoragePreflightError("tracked or staged Git status is not clean")
    if status.returncode != 0 or status.stdout != "":
        raise StoragePreflightError("untracked-inclusive Git status is not clean")
    return {
        "branch": observed["branch"],
        "head_commit": observed["head"],
        "head_tree": observed["tree"],
        "parent_commit": observed["parent"],
        "staged_status_clean": True,
        "status_porcelain_sha256": _sha(status.stdout.encode("utf-8")),
        "tracked_status_clean": True,
        "untracked_inclusive_status_clean": True,
    }


def _git_evidence(
    root: Path,
    destination_relative: str,
    attempt_paths: Sequence[PurePosixPath] = (),
    repository_binding: Mapping[str, Any] | None = None,
    authorization_hash: str = "0" * 64,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if repository_binding is None:
        raise StoragePreflightError("verified repository binding is required before path inspection")
    _require_sha(authorization_hash, name="authorization_hash")
    candidates = [destination_relative, *(path.as_posix() for path in attempt_paths)]
    ignore_outputs: list[str] = []
    for candidate in candidates:
        ignore = _run_git(root, "check-ignore", "--no-index", "-v", "--", candidate)
        if ignore.returncode != 0 or not ignore.stdout.strip():
            raise StoragePreflightError("destination or proposed attempt path is not Git-ignored")
        details, separator, _ = ignore.stdout.strip().partition("\t")
        if not separator or not details.startswith(".gitignore:") or not details.endswith(":artifacts/"):
            raise StoragePreflightError("path is not matched by the intended repository artifacts ignore rule")
        ignore_outputs.append(ignore.stdout)
        tracked = _run_git(root, "ls-files", "--stage", "--", candidate)
        if tracked.returncode != 0 or tracked.stdout.strip():
            raise StoragePreflightError("destination or proposed attempt path contains tracked paths")
        if os.path.lexists(root.joinpath(*PurePosixPath(candidate).parts)):
            raise StoragePreflightError("destination or proposed attempt path already exists")
    worktrees = _run_git(root, "worktree", "list", "--porcelain")
    if worktrees.returncode != 0:
        raise StoragePreflightError("unable to enumerate Git worktrees")
    destination_absolute = root.joinpath(*PurePosixPath(destination_relative).parts)
    nested_worktrees = 0
    for line in worktrees.stdout.splitlines():
        if line.startswith("worktree "):
            try:
                Path(line.removeprefix("worktree ")).absolute().relative_to(destination_absolute)
            except ValueError:
                continue
            nested_worktrees += 1
    if nested_worktrees:
        raise StoragePreflightError("a Git worktree root is nested beneath the destination")
    entries = _run_git(root, "ls-files", "--stage")
    if entries.returncode != 0:
        raise StoragePreflightError("unable to enumerate tracked Git entries")
    destination_prefix = destination_relative.rstrip("/") + "/"
    nested_submodules = sum(1 for line in entries.stdout.splitlines()
        if line.partition("\t")[0].startswith("160000 ")
        and line.partition("\t")[2].startswith(destination_prefix))
    if nested_submodules:
        raise StoragePreflightError("a Git submodule is nested beneath the destination")
    nested_markers = sum(1 for ancestor in _existing_ancestors(root, PurePosixPath(destination_relative))[1:]
                         if (ancestor / ".git").exists())
    if nested_markers:
        raise StoragePreflightError("a nested repository marker exists in a destination ancestor")
    return (
        {"all_candidates_matched_intended_rule": True, "candidate_count": len(candidates),
         "candidate_path_commitments": [
             _commit(
                 "operatormark-storage-git-candidate-path-v1",
                 authorization_hash,
                 candidate,
             )
             for candidate in candidates
         ],
         "check_ignore_output_sha256": _sha("".join(ignore_outputs).encode()),
         "intended_rule": ".gitignore:artifacts/", "matched": True},
        {**repository_binding, "candidate_collision_count": 0, "nested_repository_marker_count": 0,
         "nested_worktree_count": 0, "submodule_path_count": 0, "tracked_path_count": 0,
         "worktree_evidence_sha256": _sha(worktrees.stdout.encode())},
    )


_BROAD_PRINCIPALS: Final = (
    ("everyone", "S-1-1-0", True), ("anonymous", "S-1-5-7", True),
    ("network", "S-1-5-2", True), ("service", "S-1-5-6", True),
    ("local_service", "S-1-5-19", True), ("network_service", "S-1-5-20", True),
    ("guests", "S-1-5-32-546", True), ("authenticated_users", "S-1-5-11", False),
    ("builtin_users", "S-1-5-32-545", False),
)
ACL_READ_MASK: Final = 0x1 | 0x80 | 0x20000 | 0x80000000 | 0x10000000
ACL_WRITE_MASK: Final = (
    0x2 | 0x4 | 0x10 | 0x40 | 0x10000 | 0x40000 | 0x80000 | 0x40000000 | 0x10000000
)


def _windows_acl_evidence(path: Path, authorization_hash: str) -> dict[str, Any]:
    from ctypes import wintypes

    class TRUSTEE_W(ctypes.Structure):
        _fields_ = [("pMultipleTrustee", ctypes.c_void_p), ("MultipleTrusteeOperation", ctypes.c_int),
                    ("TrusteeForm", ctypes.c_int), ("TrusteeType", ctypes.c_int),
                    ("ptstrName", ctypes.c_void_p)]

    class EXPLICIT_ACCESS_W(ctypes.Structure):
        _fields_ = [
            ("grfAccessPermissions", wintypes.DWORD),
            ("grfAccessMode", ctypes.c_int),
            ("grfInheritance", wintypes.DWORD),
            ("Trustee", TRUSTEE_W),
        ]

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    owner = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    get_named = advapi32.GetNamedSecurityInfoW
    get_named.argtypes = [wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p)]
    get_named.restype = wintypes.DWORD
    result = int(get_named(str(path), 1, 0x1 | 0x4, ctypes.byref(owner), None,
                           ctypes.byref(dacl), None, ctypes.byref(descriptor)))
    if result != 0 or not descriptor.value or not owner.value:
        raise StoragePreflightError(f"GetNamedSecurityInfoW failed with Windows error {result}")
    local_free = kernel32.LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p
    try:
        get_length = advapi32.GetSecurityDescriptorLength
        get_length.argtypes = [ctypes.c_void_p]
        get_length.restype = wintypes.DWORD
        descriptor_length = int(get_length(descriptor))
        if descriptor_length <= 0:
            raise StoragePreflightError("Windows security descriptor has invalid length")
        descriptor_bytes = ctypes.string_at(descriptor, descriptor_length)
        dacl_present = wintypes.BOOL()
        dacl_defaulted = wintypes.BOOL()
        dacl_again = ctypes.c_void_p()
        get_dacl = advapi32.GetSecurityDescriptorDacl
        get_dacl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
                             ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL)]
        get_dacl.restype = wintypes.BOOL
        if not get_dacl(descriptor, ctypes.byref(dacl_present), ctypes.byref(dacl_again),
                        ctypes.byref(dacl_defaulted)):
            raise StoragePreflightError("GetSecurityDescriptorDacl failed")
        if not dacl_present.value or not dacl_again.value:
            raise StoragePreflightError("missing or NULL DACL grants broad access")
        control = wintypes.WORD()
        revision = wintypes.DWORD()
        get_control = advapi32.GetSecurityDescriptorControl
        get_control.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.WORD),
                                ctypes.POINTER(wintypes.DWORD)]
        get_control.restype = wintypes.BOOL
        if not get_control(descriptor, ctypes.byref(control), ctypes.byref(revision)):
            raise StoragePreflightError("GetSecurityDescriptorControl failed")
        sid_text_pointer = wintypes.LPWSTR()
        convert_sid_to_string = advapi32.ConvertSidToStringSidW
        convert_sid_to_string.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
        convert_sid_to_string.restype = wintypes.BOOL
        if not convert_sid_to_string(owner, ctypes.byref(sid_text_pointer)):
            raise StoragePreflightError("unable to commit the owning Windows principal")
        try:
            owner_text = sid_text_pointer.value
        finally:
            local_free(ctypes.cast(sid_text_pointer, ctypes.c_void_p))
        convert_string_sid = advapi32.ConvertStringSidToSidW
        convert_string_sid.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
        convert_string_sid.restype = wintypes.BOOL
        get_rights = advapi32.GetEffectiveRightsFromAclW
        get_rights.argtypes = [ctypes.c_void_p, ctypes.POINTER(TRUSTEE_W),
                               ctypes.POINTER(wintypes.DWORD)]
        get_rights.restype = wintypes.DWORD
        get_explicit = advapi32.GetExplicitEntriesFromAclW
        get_explicit.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.ULONG),
            ctypes.POINTER(ctypes.POINTER(EXPLICIT_ACCESS_W)),
        ]
        get_explicit.restype = wintypes.DWORD
        broad: list[dict[str, Any]] = []
        blocking_count = 0
        residual_count = 0
        for principal_class, sid_text, blocks in _BROAD_PRINCIPALS:
            sid = ctypes.c_void_p()
            if not convert_string_sid(sid_text, ctypes.byref(sid)):
                raise StoragePreflightError("unable to construct broad-principal SID")
            try:
                trustee = TRUSTEE_W(None, 0, 0, 0, sid.value)
                mask = wintypes.DWORD()
                rights_result = int(get_rights(dacl_again, ctypes.byref(trustee), ctypes.byref(mask)))
                if rights_result != 0:
                    raise StoragePreflightError(
                        f"GetEffectiveRightsFromAclW failed with Windows error {rights_result}"
                    )
                effective_read = bool(int(mask.value) & ACL_READ_MASK)
                effective_write = bool(int(mask.value) & ACL_WRITE_MASK)
                blocking = blocks and (effective_read or effective_write)
                residual = not blocks and (effective_read or effective_write)
                blocking_count += int(blocking)
                residual_count += int(residual)
                broad.append({
                    "blocking": blocking, "effective_mask_hex": hex(int(mask.value)),
                    "effective_read": effective_read, "effective_write": effective_write,
                    "principal_class": principal_class,
                    "principal_commitment": _commit("operatormark-storage-principal-v1",
                                                     authorization_hash, sid_text),
                    "residual_risk": residual,
                })
            finally:
                local_free(sid)
        explicit_count = wintypes.ULONG()
        explicit_entries = ctypes.POINTER(EXPLICIT_ACCESS_W)()
        explicit_result = int(
            get_explicit(
                dacl_again,
                ctypes.byref(explicit_count),
                ctypes.byref(explicit_entries),
            )
        )
        if explicit_result != 0:
            raise StoragePreflightError(
                f"GetExplicitEntriesFromAclW failed with Windows error {explicit_result}"
            )
        additional_by_sid: dict[str, dict[str, Any]] = {}
        fixed_sids = {sid.casefold() for _, sid, _ in _BROAD_PRINCIPALS}
        try:
            for index in range(int(explicit_count.value)):
                entry = explicit_entries[index]
                if int(entry.grfAccessMode) not in {1, 2}:
                    continue
                trustee = entry.Trustee
                if trustee.pMultipleTrustee or int(trustee.TrusteeForm) != 0:
                    raise StoragePreflightError(
                        "allow ACE trustee could not be resolved to a canonical SID"
                    )
                sid_pointer = ctypes.c_void_p(trustee.ptstrName)
                if not sid_pointer.value:
                    raise StoragePreflightError("allow ACE contains a NULL trustee SID")
                entry_sid_pointer = wintypes.LPWSTR()
                if not convert_sid_to_string(sid_pointer, ctypes.byref(entry_sid_pointer)):
                    raise StoragePreflightError("unable to resolve allow ACE trustee SID")
                try:
                    entry_sid = str(entry_sid_pointer.value)
                finally:
                    local_free(ctypes.cast(entry_sid_pointer, ctypes.c_void_p))
                if entry_sid.casefold() in fixed_sids or entry_sid.casefold() == owner_text.casefold():
                    continue
                mask = wintypes.DWORD()
                rights_result = int(
                    get_rights(dacl_again, ctypes.byref(trustee), ctypes.byref(mask))
                )
                if rights_result != 0:
                    raise StoragePreflightError(
                        f"additional trustee rights query failed with Windows error {rights_result}"
                    )
                effective_read = bool(int(mask.value) & ACL_READ_MASK)
                effective_write = bool(int(mask.value) & ACL_WRITE_MASK)
                has_access = effective_read or effective_write
                commitment = _commit(
                    "operatormark-storage-principal-v1", authorization_hash, entry_sid
                )
                additional_by_sid[entry_sid.casefold()] = {
                    "blocking": False,
                    "effective_mask_hex": hex(int(mask.value)),
                    "effective_read": effective_read,
                    "effective_write": effective_write,
                    "principal_class": "additional_allow_ace",
                    "principal_commitment": commitment,
                    "residual_risk": has_access,
                }
        finally:
            if explicit_entries:
                local_free(ctypes.cast(explicit_entries, ctypes.c_void_p))
        additional = sorted(
            additional_by_sid.values(), key=lambda item: item["principal_commitment"]
        )
        residual_count += sum(int(item["residual_risk"]) for item in additional)
        if blocking_count:
            raise StoragePreflightError("prohibited broad-principal effective access detected")
        return {
            "_acl_inspected_absolute_path": str(path.absolute()),
            "_owning_principal_sid": owner_text,
            "additional_allow_principal_count": len(additional),
            "additional_allow_principal_effective_access": additional,
            "blocking_broad_access_count": blocking_count,
            "broad_principal_effective_access": broad,
            "broader_local_user_residual_count": residual_count,
            "current_process_effective_rights": {
                "create": False, "delete": False, "read": bool(os.access(path, os.R_OK)),
                "rename": False, "write": bool(os.access(path, os.W_OK)),
            },
            "dacl_present": True, "inheritance_enabled": not bool(int(control.value) & 0x1000),
            "inspection_method": "GetNamedSecurityInfoW plus GetEffectiveRightsFromAclW plus operation probe",
            "owning_principal_commitment": _commit("operatormark-storage-owner-v1",
                                                    authorization_hash, owner_text),
            "platform": "nt",
            "security_descriptor_commitment": _commit_bytes(
                "operatormark-storage-security-descriptor-v1", authorization_hash, descriptor_bytes),
        }
    finally:
        local_free(descriptor)


def _permission_evidence(
    path: Path, authorization_hash: str = "0" * 64,
    probe_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_sha(authorization_hash, name="authorization_hash")
    if os.name != "nt":
        raise StoragePreflightError("effective broad-principal ACL inspection is unavailable off Windows")
    evidence = _windows_acl_evidence(path, authorization_hash)
    if probe_evidence is not None and probe_evidence.get("passed") is True:
        evidence["current_process_effective_rights"] = {
            "create": True, "delete": True, "read": True, "rename": True, "write": True,
        }
    if not all(evidence["current_process_effective_rights"].values()):
        raise StoragePreflightError("current-process effective storage rights were not all demonstrated")
    return evidence


def _probe_volume_identifier(path: Path) -> str:
    if os.name == "nt":
        return _windows_handle_snapshot(path)["_volume_identifier"]
    return str(path.stat().st_dev)


def _probe(
    parent: Path, authorization_hash: str = "0" * 64,
    expected_volume_identifier: str | None = None,
) -> dict[str, Any]:
    _require_sha(authorization_hash, name="authorization_hash")
    directory = parent / PROBE_DIRECTORY
    staging = directory / "staging.literal"
    renamed = directory / "renamed.literal"
    source = staging / "probe.literal"
    renamed_source = renamed / "probe.literal"
    if os.path.lexists(directory):
        raise StoragePreflightError("preflight probe directory collision")
    directory.mkdir()
    try:
        staging.mkdir()
        with source.open("xb") as stream:
            stream.write(PROBE_BYTES)
            stream.flush()
            os.fsync(stream.fileno())
        if source.read_bytes() != PROBE_BYTES:
            raise StoragePreflightError("probe read-back mismatch")
        source_volume = _probe_volume_identifier(staging)
        if expected_volume_identifier is not None and source_volume != expected_volume_identifier:
            raise StoragePreflightError("probe staging directory is not on the expected fixed volume")
        if os.path.lexists(renamed):
            raise StoragePreflightError("preflight rename target collision")
        os.rename(staging, renamed)
        if os.path.lexists(staging) or not renamed.is_dir():
            raise StoragePreflightError("atomic directory rename postconditions failed")
        renamed_volume = _probe_volume_identifier(renamed)
        if renamed_volume != source_volume:
            raise StoragePreflightError("probe directory rename crossed a volume")
        if renamed_source.read_bytes() != PROBE_BYTES:
            raise StoragePreflightError("renamed probe read-back mismatch")
        renamed_source.unlink()
        renamed.rmdir()
        directory.rmdir()
        return {
            "atomic_directory_rename_verified": True, "fixed_literal_sha256": _sha(PROBE_BYTES),
            "operations": list(PROBE_OPERATIONS),
            "passed": True, "probe_directory_preserved": False, "probe_file_preserved": False,
            "renamed_volume_identifier_commitment": _commit(
                "operatormark-private-volume-v1", authorization_hash, renamed_volume),
            "same_volume_verified": True,
            "source_volume_identifier_commitment": _commit(
                "operatormark-private-volume-v1", authorization_hash, source_volume),
        }
    except BaseException as exc:
        failure = {"error_type": type(exc).__name__, "passed": False,
                   "probe_directory_preserved": True,
                   "schema": "operatormark.storage_preflight.failure.v1"}
        try:
            directory.mkdir(exist_ok=True)
            (directory / "FAILURE.json").write_bytes(_canonical(failure))
        except OSError:
            pass
        raise


def _command_observation(
    arguments: Sequence[str], authorization_hash: str, domain: str
) -> tuple[str, str | None, str | None]:
    try:
        result = subprocess.run(list(arguments), check=False, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
    except OSError:
        return "unavailable", None, None
    output = (result.stdout + "\n" + result.stderr).replace("\r\n", "\n")
    commitment = _commit(domain, authorization_hash, f"{result.returncode}\n{output}")
    return (
        "observed" if result.returncode == 0 else "query_failed",
        commitment,
        output,
    )


def _encryption_evidence(
    components: Sequence[Mapping[str, Any]], volume: Mapping[str, Any], authorization_hash: str
) -> dict[str, Any]:
    encrypted_count = sum(1 for component in components
        if int(str(component["file_attributes_hex"]), 16) & ENCRYPTED_ATTRIBUTE)
    query_outcome = "not_inspected_non_windows"
    interpreted_status = "unknown_not_inspected"
    commitment: str | None = None
    if os.name == "nt":
        query_outcome, commitment, query_output = _command_observation(
            ["manage-bde.exe", "-status", str(volume["_volume_root"])], authorization_hash,
            "operatormark-storage-device-encryption-query-v1")
        if query_outcome == "observed" and query_output is not None:
            lowered = query_output.casefold()
            if "protection on" in lowered and (
                "fully encrypted" in lowered or "percentage encrypted: 100" in lowered
            ):
                interpreted_status = "reported_protected_not_independently_proven"
            elif "protection off" in lowered or "fully decrypted" in lowered:
                interpreted_status = "reported_not_protected"
            else:
                interpreted_status = "observed_but_unclassified"
        elif query_outcome == "query_failed":
            interpreted_status = "unknown_query_failed"
        elif query_outcome == "unavailable":
            interpreted_status = "unknown_query_unavailable"
    return {
        "device_encryption_evidence_commitment": commitment,
        "device_encryption_interpreted_status": interpreted_status,
        "device_encryption_query_outcome": query_outcome,
        "encrypted_ancestor_count": encrypted_count,
        "encryption_presence_establishes_legal_or_cryptographic_protection": False,
        "filesystem_supports_encryption": bool(int(volume["_filesystem_flags"]) & 0x00020000),
        "status": "evidence_only_with_key_management_and_access_residuals",
    }


def _windows_registry_exposure_observation(
    authorization_hash: str,
) -> tuple[str, str | None]:
    if os.name != "nt":
        return "not_inspected_non_windows", None
    import winreg

    observations: list[dict[str, Any]] = []
    successful_queries = 0
    queries = (
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\CrashControl",
         "CrashDumpEnabled"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management", "PagingFiles"),
    )
    for hive, path, value_name in queries:
        try:
            with winreg.OpenKey(hive, path) as key:
                value, kind = winreg.QueryValueEx(key, value_name)
                observations.append({"name": value_name, "present": value not in (None, "", 0, []),
                                     "registry_type": kind})
                successful_queries += 1
        except OSError:
            observations.append({"name": value_name, "present": None, "registry_type": None})
    outcome = "observed" if successful_queries == len(queries) else (
        "partially_observed" if successful_queries else "unavailable"
    )
    return outcome, _commit_bytes(
        "operatormark-storage-exposure-registry-v1",
        authorization_hash,
        _canonical(observations),
    )


def _backup_and_indexing_evidence(
    components: Sequence[Mapping[str, Any]], sync: Mapping[str, Any], authorization_hash: str,
) -> dict[str, Any]:
    service_commitments: dict[str, str | None] = {}
    service_outcomes: dict[str, str] = {}
    if os.name == "nt":
        for service in ("WSearch", "wbengine", "VSS"):
            outcome, commitment, _ = _command_observation(
                ["sc.exe", "query", service], authorization_hash,
                f"operatormark-storage-service-query-{service.casefold()}-v1")
            service_commitments[service] = commitment
            service_outcomes[service] = outcome
    registry_outcome, registry_commitment = _windows_registry_exposure_observation(
        authorization_hash
    )
    indexed_possible = any(not (int(str(component["file_attributes_hex"]), 16)
                                  & NOT_CONTENT_INDEXED_ATTRIBUTE) for component in components)
    return {
        "administrator_access_may_expose_copies": True,
        "administrator_backup_malware_physical_access_and_remanence_eliminated": False,
        "antivirus_may_expose_copies": True, "backup_software_may_expose_copies": True,
        "crash_dumps_may_expose_copies": True, "indexing_may_expose_copies": indexed_possible,
        "local_sync_clients_may_expose_copies": bool(sync["active_sync_clients"]),
        "paging_may_expose_copies": True,
        "registry_query_evidence_commitment": registry_commitment,
        "registry_query_outcome": registry_outcome,
        "service_query_evidence_commitments": service_commitments,
        "service_query_outcomes": service_outcomes,
        "status": ("administrator_backup_indexing_antivirus_crash_dump_paging_malware_"
                   "physical_access_key_management_and_deletion_remanence_risks_remain"),
    }


def _contains_path(root: Path, candidate: Path) -> bool:
    if os.name == "nt":
        return _path_is_within_windows(str(candidate), str(root))
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _private_storage_identity_record(
    *,
    request: StoragePreflightRequest,
    root: Path,
    destination: Path,
    permission: Mapping[str, Any],
) -> dict[str, Any]:
    owner_sid = permission.get("_owning_principal_sid")
    acl_path = permission.get("_acl_inspected_absolute_path")
    if not isinstance(owner_sid, str) or not owner_sid or not isinstance(acl_path, str) or not acl_path:
        raise StoragePreflightError("private storage identity evidence is incomplete")
    record: dict[str, Any] = {
        "acl_inspected_absolute_path": acl_path,
        "authorization_file_sha256": request.authorization_file_sha256,
        "destination_absolute_path": str(destination),
        "disclosure_classification": "confidential_private_storage_identity_evidence",
        "expected_parent_commit": request.expected_parent_commit,
        "implementation_commit": request.implementation_commit,
        "implementation_tree": request.implementation_tree,
        "owning_principal_sid": owner_sid,
        "repository_root_absolute_path": str(root),
        "schema": "operatormark.darcy_branch_c_v1.private_storage_identity.v1",
    }
    record["record_self_sha256"] = _sha(_canonical(record))
    validate_private_storage_identity_record(record)
    return record


def validate_private_storage_identity_record(record: Mapping[str, Any]) -> None:
    _require_exact_keys(record, PRIVATE_IDENTITY_KEYS, name="private storage identity record")
    if record["schema"] != "operatormark.darcy_branch_c_v1.private_storage_identity.v1":
        raise StoragePreflightError("private storage identity schema mismatch")
    if record["disclosure_classification"] != "confidential_private_storage_identity_evidence":
        raise StoragePreflightError("private storage identity classification mismatch")
    for name in (
        "repository_root_absolute_path", "destination_absolute_path",
        "acl_inspected_absolute_path", "owning_principal_sid",
    ):
        if not isinstance(record[name], str) or not record[name]:
            raise StoragePreflightError(f"private storage identity field {name} is empty")
    path_fields = (
        "repository_root_absolute_path", "destination_absolute_path",
        "acl_inspected_absolute_path",
    )
    if any(not ntpath.isabs(str(record[name])) for name in path_fields):
        raise StoragePreflightError("private storage identity path is not absolute")
    if not str(record["owning_principal_sid"]).startswith("S-"):
        raise StoragePreflightError("private storage owning SID is not canonical text")
    if not _path_is_within_windows(
        str(record["destination_absolute_path"]),
        str(record["repository_root_absolute_path"]),
    ) or not _path_is_within_windows(
        str(record["acl_inspected_absolute_path"]),
        str(record["repository_root_absolute_path"]),
    ):
        raise StoragePreflightError("private storage identity paths escape the repository")
    for name in ("implementation_commit", "implementation_tree", "expected_parent_commit"):
        _require_git_oid(str(record[name]), name=name)
    _require_sha(str(record["authorization_file_sha256"]), name="authorization_file_sha256")
    _require_sha(str(record["record_self_sha256"]), name="record_self_sha256")
    payload = dict(record)
    claimed = payload.pop("record_self_sha256")
    if claimed != _sha(_canonical(payload)):
        raise StoragePreflightError("private storage identity self-hash mismatch")


def run_storage_preflight(
    request: StoragePreflightRequest, *, start: str | Path,
    environ: Mapping[str, str] | None = None,
    private_identity_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for name, value in (
        ("protocol_file_sha256", request.protocol_file_sha256),
        ("authorization_file_sha256", request.authorization_file_sha256),
    ):
        _require_sha(value, name=name)
    for name, value in (
        ("implementation_commit", request.implementation_commit),
        ("implementation_tree", request.implementation_tree),
        ("expected_parent_commit", request.expected_parent_commit),
    ):
        _require_git_oid(value, name=name)
    if not request.state or request.state != request.state.strip():
        raise StoragePreflightError("state must be a non-empty canonical string")
    if request.expected_branch != "main":
        raise StoragePreflightError("the frozen repository branch must be main")
    relative = _validate_relative_destination(request.destination_relative_path)
    attempt_paths = _validate_attempt_paths(request.attempt_relative_paths, relative)
    root = discover_repository_root(start)
    if os.name == "nt":
        _reject_windows_namespace_or_alias(root)
    _bind_repository_root_to_loaded_module(root)
    repository_binding = _git_repository_binding(
        root,
        expected_branch=request.expected_branch,
        expected_head=request.implementation_commit,
        expected_tree=request.implementation_tree,
        expected_parent=request.expected_parent_commit,
    )
    components_before, resolved_destination, root_volume_identifier = (
        _inspect_path_components_private(root, relative, request.authorization_file_sha256))
    if len(components_before) != len(relative.parts):
        raise StoragePreflightError(
            "every destination parent must exist and have handle-backed component evidence"
        )
    if not _contains_path(root, resolved_destination):
        raise StoragePreflightError("canonical destination escapes the verified repository root")
    destination = root.joinpath(*relative.parts)
    if os.path.lexists(destination):
        raise StoragePreflightError("canonical corpus destination already exists")
    for attempt in attempt_paths:
        if os.path.lexists(root.joinpath(*attempt.parts)):
            raise StoragePreflightError("proposed attempt path already exists")
    volume = _volume_evidence(root)
    maximum_component_length = int(volume["maximum_component_length"])
    for candidate in (relative, *attempt_paths):
        if any(len(part.encode("utf-16-le")) // 2 > maximum_component_length
               for part in candidate.parts):
            raise StoragePreflightError(
                "destination or attempt path exceeds the observed volume component limit"
            )
    if os.name == "nt" and (
        _probe_volume_identifier(root) != root_volume_identifier
        or volume["_volume_identifier"] != root_volume_identifier
    ):
        raise StoragePreflightError("repository fixed-volume identity changed during preflight")
    if environ is not None and any(
        environ.get(name) != os.environ.get(name) for name in SYNC_ENVIRONMENT_NAMES
    ):
        raise StoragePreflightError(
            "injected synchronization environment differs from the live process environment"
        )
    sync = _sync_evidence(
        resolved_destination,
        os.environ if environ is None else environ,
        request.authorization_file_sha256,
    )
    ignore, tracking = _git_evidence(
        root,
        request.destination_relative_path,
        attempt_paths,
        repository_binding=repository_binding,
        authorization_hash=request.authorization_file_sha256,
    )
    required = required_free_bytes(request.estimated_accepted_bytes, request.projected_audit_bytes)
    if volume["free_bytes"] < required:
        raise StoragePreflightError("insufficient fixed-volume free space")
    probe_parent = resolved_destination.parent
    if not probe_parent.is_dir():
        raise StoragePreflightError("deepest destination parent must already exist")
    probe = _probe(probe_parent, request.authorization_file_sha256,
                   expected_volume_identifier=root_volume_identifier)
    permission = _permission_evidence(probe_parent, request.authorization_file_sha256, probe)
    private_record: dict[str, Any] | None = None
    if private_identity_evidence is not None:
        if private_identity_evidence:
            raise StoragePreflightError("private identity evidence sink must initially be empty")
        private_record = _private_storage_identity_record(
            request=request,
            root=root,
            destination=resolved_destination,
            permission=permission,
        )
    encryption = _encryption_evidence(components_before, volume, request.authorization_file_sha256)
    backup = _backup_and_indexing_evidence(components_before, sync,
                                           request.authorization_file_sha256)
    components_after, destination_after, root_volume_after = _inspect_path_components_private(
        root, relative, request.authorization_file_sha256)
    if (components_before != components_after or destination_after != resolved_destination
            or root_volume_after != root_volume_identifier):
        raise StoragePreflightError("canonical path-component evidence changed during preflight")
    components = [{**component, "stable_before_after": True} for component in components_before]
    residual_risks = [
        "administrator backup malware physical-access key-management and deletion-remanence risks remain"
    ]
    if permission["broader_local_user_residual_count"]:
        residual_risks.append(
            "broader local-user effective access remains and requires explicit State 4 acceptance")
    public_volume = {key: value for key, value in volume.items() if not key.startswith("_")}
    public_permission = {
        key: value for key, value in permission.items() if not key.startswith("_")
    }
    public_volume.update({
        "volume_identifier_commitment": _commit("operatormark-private-volume-v1",
            request.authorization_file_sha256, str(volume["_volume_identifier"])),
        "volume_root_commitment": _commit("operatormark-private-volume-root-v1",
            request.authorization_file_sha256, os.path.normcase(str(volume["_volume_root"]))),
    })
    record: dict[str, Any] = {
        "all_mandatory_checks_passed": True,
        "authorization_file_sha256": request.authorization_file_sha256,
        "backup_and_indexing_evidence": backup,
        "canonical_resolution_method": (CANONICAL_METHOD_WINDOWS if os.name == "nt"
                                         else CANONICAL_METHOD_POSIX),
        "destination_absolute_path_commitment": _commit("operatormark-private-destination-v1",
            request.authorization_file_sha256, os.path.normcase(str(resolved_destination))),
        "destination_relative_path": request.destination_relative_path,
        "encryption_evidence": encryption, "estimated_accepted_bytes": request.estimated_accepted_bytes,
        "git_ignore_evidence": ignore, "git_tracking_evidence": tracking,
        "implementation_commit": request.implementation_commit,
        "implementation_tree": request.implementation_tree,
        "observed_free_bytes": volume["free_bytes"], "path_component_evidence": components,
        "permission_evidence": public_permission, "probe_evidence": probe,
        "projected_audit_bytes": request.projected_audit_bytes,
        "protocol_file_sha256": request.protocol_file_sha256,
        "repository_root_commitment": _commit("operatormark-repository-root-v1",
            request.authorization_file_sha256, os.path.normcase(str(root))),
        "required_free_bytes": required, "residual_risks": residual_risks,
        "schema": "operatormark.darcy_branch_c_v1.private_storage_preflight.v1",
        "state": request.state, "sync_detection_evidence": sync,
        "volume_evidence": public_volume,
    }
    record["record_self_sha256"] = _sha(_canonical(record))
    validate_storage_preflight_record(record)
    if private_identity_evidence is not None and private_record is not None:
        private_identity_evidence.update(private_record)
    return record


def _require_exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise StoragePreflightError(f"{name} keys mismatch")
    return value


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StoragePreflightError(f"{name} must be a non-negative integer")
    return value


def _validate_state4_permission_metadata_context(
    value: object,
) -> Mapping[str, Any]:
    context = _require_exact_keys(
        value,
        STATE4_FILE_PERMISSION_METADATA_CONTEXT_KEYS,
        name="State 4 permission metadata context",
    )
    hash_names = (
        "protocol_file_sha256",
        "state4_authorization_file_sha256",
        "storage_preflight_file_sha256",
        "permission_profile_file_sha256",
        "repository_root_commitment",
        "volume_identifier_commitment",
    )
    for name in hash_names:
        item = context[name]
        if not isinstance(item, str):
            raise StoragePreflightError(f"State 4 permission metadata context {name} is not text")
        _require_sha(item, name=name)
    if context["protocol_file_sha256"] != PROTOCOL_SHA256:
        raise StoragePreflightError("State 4 permission metadata protocol commitment mismatch")
    for name in ("implementation_commit", "implementation_tree"):
        item = context[name]
        if not isinstance(item, str):
            raise StoragePreflightError(f"State 4 permission metadata context {name} is not text")
        _require_git_oid(item, name=name)
    residuals = context["accepted_local_user_residual_principal_commitments"]
    if not isinstance(residuals, list):
        raise StoragePreflightError(
            "accepted local-user residual principal commitments must be a JSON array"
        )
    for index, commitment in enumerate(residuals):
        if not isinstance(commitment, str):
            raise StoragePreflightError(
                f"accepted local-user residual principal commitment {index} is not text"
            )
        _require_sha(commitment, name=f"accepted residual principal commitment {index}")
    if residuals != sorted(residuals) or len(set(residuals)) != len(residuals):
        raise StoragePreflightError(
            "accepted local-user residual principal commitments must be sorted and unique"
        )
    return context


def _validate_state4_permission_metadata_dword(value: object, *, name: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"0x(?:0|[1-9a-f][0-9a-f]*)", value) is None:
        raise StoragePreflightError(f"{name} must be canonical lowercase unsigned DWORD hex")
    parsed = int(value, 16)
    if parsed > 0xFFFFFFFF or hex(parsed) != value:
        raise StoragePreflightError(f"{name} must be canonical lowercase unsigned DWORD hex")
    return parsed


def _validate_state4_permission_metadata_path(
    value: object,
    *,
    expected_index: int,
    name: str = "relative_path",
) -> str:
    index = _require_nonnegative_integer(expected_index, name="expected permission row index")
    if index >= STATE4_FILE_PERMISSION_METADATA_ROW_COUNT:
        raise StoragePreflightError("expected permission row index is outside the frozen inventory")
    if index == STATE4_FILE_PERMISSION_METADATA_ROW_COUNT - 1:
        expected_path = "state4_controls/rng_audit.json"
    else:
        cursor = 0
        expected_path: str | None = None
        for split, component, shard_count in (
            ("training", "training", 64),
            ("validation", "validation", 4),
            ("test", "test.sealed", 8),
        ):
            shard_rows = shard_count * 3
            if index < cursor + shard_rows:
                shard_offset, member_offset = divmod(index - cursor, 3)
                if member_offset == 0:
                    expected_path = (
                        f"corpus_staging/{component}/shard_{shard_offset:04d}/rho.float64.npy"
                    )
                elif member_offset == 1:
                    expected_path = (
                        f"state4_controls/{component}/shard_{shard_offset:04d}.controls.jsonl"
                    )
                else:
                    expected_path = (
                        f"state4_controls/{component}/shard_{shard_offset:04d}.manifest.json"
                    )
                break
            cursor += shard_rows
            if index == cursor:
                expected_path = f"state4_controls/{component}/split_manifest.json"
                break
            cursor += 1
        if expected_path is None:
            raise StoragePreflightError("unable to derive the frozen permission metadata path")
    if not isinstance(value, str):
        raise StoragePreflightError(f"{name} must be text")
    checked = _validate_relative_path(value, name=name).as_posix()
    if checked != expected_path:
        raise StoragePreflightError(f"{name} does not match the frozen permission inventory")
    return checked


def _validate_state4_permission_metadata_access(
    value: object,
    *,
    name: str,
    expected_keys: set[str],
) -> tuple[Mapping[str, Any], bool, bool]:
    item = _require_exact_keys(value, expected_keys, name=name)
    if not isinstance(item["principal_class"], str) or not item["principal_class"]:
        raise StoragePreflightError(f"{name} principal class is invalid")
    commitment = item["principal_commitment"]
    if not isinstance(commitment, str):
        raise StoragePreflightError(f"{name} principal commitment is not text")
    _require_sha(commitment, name=f"{name} principal commitment")
    mask = _validate_state4_permission_metadata_dword(
        item["effective_mask_hex"], name=f"{name} effective mask"
    )
    if mask & STATE4_PERMISSION_GENERIC_RIGHTS_MASK:
        raise StoragePreflightError(f"{name} retains unmapped generic access rights")
    effective_read = bool(mask & ACL_READ_MASK)
    effective_write = bool(mask & ACL_WRITE_MASK)
    if (
        item["effective_read"] is not effective_read
        or item["effective_write"] is not effective_write
    ):
        raise StoragePreflightError(f"{name} effective-mask interpretation mismatch")
    if not isinstance(item["blocking"], bool) or not isinstance(item["residual_risk"], bool):
        raise StoragePreflightError(f"{name} risk flags must be JSON booleans")
    return item, effective_read, effective_write


def validate_state4_file_permission_metadata(
    record: Mapping[str, Any],
    *,
    expected_context: Mapping[str, Any],
    expected_index: int,
    expected_file: Mapping[str, Any],
) -> None:
    checked_context = _validate_state4_permission_metadata_context(expected_context)
    index = _require_nonnegative_integer(expected_index, name="expected permission row index")
    if index >= STATE4_FILE_PERMISSION_METADATA_ROW_COUNT:
        raise StoragePreflightError("expected permission row index is outside the frozen inventory")
    checked_expected_file = _require_exact_keys(
        expected_file,
        STATE4_FILE_PERMISSION_METADATA_EXPECTED_FILE_KEYS,
        name="expected permission file",
    )
    expected_path = _validate_state4_permission_metadata_path(
        checked_expected_file["relative_path"],
        expected_index=index,
        name="expected file relative_path",
    )
    expected_bytes = _require_nonnegative_integer(
        checked_expected_file["file_bytes"], name="expected file_bytes"
    )
    if expected_bytes == 0:
        raise StoragePreflightError("expected file_bytes must be positive")
    expected_file_sha256 = checked_expected_file["file_sha256"]
    if not isinstance(expected_file_sha256, str):
        raise StoragePreflightError("expected file_sha256 is not text")
    _require_sha(expected_file_sha256, name="expected file_sha256")

    row = _require_exact_keys(
        record,
        STATE4_FILE_PERMISSION_METADATA_KEYS,
        name="State 4 file permission metadata",
    )
    if row["schema"] != STATE4_FILE_PERMISSION_METADATA_SCHEMA:
        raise StoragePreflightError("State 4 file permission metadata schema mismatch")
    row_context = _validate_state4_permission_metadata_context(row["context"])
    if dict(row_context) != dict(checked_context):
        raise StoragePreflightError("State 4 file permission metadata context mismatch")
    authorization_hash = str(checked_context["state4_authorization_file_sha256"])
    expected_attempt_id = f"s4-{authorization_hash[:24]}"
    if row["attempt_id"] != expected_attempt_id:
        raise StoragePreflightError("State 4 file permission metadata attempt binding mismatch")
    row_index = _require_nonnegative_integer(row["row_index"], name="permission row_index")
    if row_index != index:
        raise StoragePreflightError("State 4 file permission metadata row index mismatch")
    row_path = _validate_state4_permission_metadata_path(
        row["relative_path"], expected_index=row_index
    )
    row_bytes = _require_nonnegative_integer(row["file_bytes"], name="file_bytes")
    if row_bytes == 0:
        raise StoragePreflightError("file_bytes must be positive")
    row_file_sha256 = row["file_sha256"]
    if not isinstance(row_file_sha256, str):
        raise StoragePreflightError("file_sha256 is not text")
    _require_sha(row_file_sha256, name="file_sha256")
    if (
        row_path != expected_path
        or row_bytes != expected_bytes
        or row_file_sha256 != expected_file_sha256
    ):
        raise StoragePreflightError("State 4 permission metadata subject mismatch")
    if row["observation_phase"] != STATE4_FILE_PERMISSION_METADATA_OBSERVATION_PHASE:
        raise StoragePreflightError("State 4 permission metadata observation phase mismatch")

    path_evidence = _require_exact_keys(
        row["path_evidence"],
        STATE4_FILE_PERMISSION_METADATA_PATH_KEYS,
        name="State 4 file path evidence",
    )
    for name in (
        "canonical_path_commitment",
        "volume_identifier_commitment",
        "file_identity_commitment",
    ):
        value = path_evidence[name]
        if not isinstance(value, str):
            raise StoragePreflightError(f"State 4 file path evidence {name} is not text")
        _require_sha(value, name=name)
    if path_evidence["volume_identifier_commitment"] != checked_context[
        "volume_identifier_commitment"
    ]:
        raise StoragePreflightError("State 4 file volume commitment mismatch")
    if any(
        path_evidence[name] is not True
        for name in (
            "canonical_name_exact",
            "unicode_nfc_exact",
            "same_volume_as_storage_context",
        )
    ):
        raise StoragePreflightError("State 4 file path predicates did not all pass")
    attributes = _validate_state4_permission_metadata_dword(
        path_evidence["file_attributes_hex"], name="file_attributes_hex"
    )
    reparse_tag = _validate_state4_permission_metadata_dword(
        path_evidence["reparse_tag_hex"], name="reparse_tag_hex"
    )
    prohibited_attributes = STATE4_FILE_ATTRIBUTE_DIRECTORY | REPARSE_ATTRIBUTE | CLOUD_ATTRIBUTES
    if not (attributes & STATE4_FILE_ATTRIBUTE_READONLY):
        raise StoragePreflightError("State 4 file does not report the read-only attribute")
    if attributes & prohibited_attributes or reparse_tag != 0:
        raise StoragePreflightError("State 4 file reports directory, reparse, or cloud metadata")

    private_commitment = row["private_observation_commitment"]
    if not isinstance(private_commitment, str):
        raise StoragePreflightError("private observation commitment is not text")
    _require_sha(private_commitment, name="private observation commitment")

    permission = _require_exact_keys(
        row["permission_evidence"],
        STATE4_FILE_PERMISSION_METADATA_PERMISSION_KEYS,
        name="State 4 file permission evidence",
    )
    if permission["platform"] != "nt":
        raise StoragePreflightError("State 4 file permission platform must be nt")
    if permission["inspection_method"] != STATE4_FILE_PERMISSION_METADATA_PROFILE:
        raise StoragePreflightError("State 4 file permission inspection profile mismatch")
    if permission["dacl_present"] is not True:
        raise StoragePreflightError("State 4 file permission evidence requires a DACL")
    if not isinstance(permission["inheritance_enabled"], bool):
        raise StoragePreflightError("State 4 file permission inheritance state is not Boolean")
    for name in (
        "security_descriptor_commitment",
        "owning_principal_commitment",
        "current_principal_commitment",
    ):
        value = permission[name]
        if not isinstance(value, str):
            raise StoragePreflightError(f"State 4 file permission {name} is not text")
        _require_sha(value, name=name)

    file_mask = _validate_state4_permission_metadata_dword(
        permission["current_process_file_access_mask_hex"],
        name="current process file access mask",
    )
    parent_mask = _validate_state4_permission_metadata_dword(
        permission["current_process_parent_access_mask_hex"],
        name="current process parent access mask",
    )
    if (file_mask | parent_mask) & STATE4_PERMISSION_GENERIC_RIGHTS_MASK:
        raise StoragePreflightError("current-process masks retain unmapped generic access rights")
    rights = _require_exact_keys(
        permission["current_process_effective_rights"],
        RIGHTS_KEYS,
        name="State 4 current-process file rights",
    )
    if (
        rights["read"] is not True
        or rights["write"] is not False
        or rights["create"] is not None
        or rights["delete"] is not False
        or rights["rename"] is not False
    ):
        raise StoragePreflightError("State 4 current-process file rights profile mismatch")
    if not (file_mask & STATE4_FILE_PERMISSION_READ_DATA):
        raise StoragePreflightError("reported file read right is absent from the file mask")
    if file_mask & (STATE4_FILE_PERMISSION_WRITE_DATA | STATE4_FILE_PERMISSION_APPEND_DATA):
        raise StoragePreflightError("reported file write denial conflicts with the file mask")
    if file_mask & STATE4_FILE_PERMISSION_DELETE:
        raise StoragePreflightError("reported file delete denial conflicts with the file mask")
    if parent_mask & STATE4_PARENT_PERMISSION_DELETE_CHILD:
        raise StoragePreflightError("reported file rename/delete denial conflicts with the parent mask")
    prohibited_file_alteration = (
        STATE4_FILE_PERMISSION_WRITE_EA
        | STATE4_FILE_PERMISSION_WRITE_ATTRIBUTES
        | STATE4_FILE_PERMISSION_WRITE_DAC
        | STATE4_FILE_PERMISSION_WRITE_OWNER
    )
    if file_mask & prohibited_file_alteration:
        raise StoragePreflightError("reported file mask retains prohibited alteration rights")

    broad = permission["broad_principal_effective_access"]
    if not isinstance(broad, list) or len(broad) != len(_BROAD_PRINCIPALS):
        raise StoragePreflightError("State 4 broad-principal coverage is incomplete")
    expected_residuals = set(
        checked_context["accepted_local_user_residual_principal_commitments"]
    )
    observed_residuals: set[str] = set()
    fixed_commitments: set[str] = set()
    calculated_blocking = 0
    calculated_residual = 0
    for value, (expected_class, expected_sid, expected_to_block) in zip(
        broad, _BROAD_PRINCIPALS, strict=True
    ):
        item, effective_read, effective_write = _validate_state4_permission_metadata_access(
            value,
            name="State 4 broad-principal evidence",
            expected_keys=BROAD_ACCESS_KEYS,
        )
        if item["principal_class"] != expected_class:
            raise StoragePreflightError("State 4 broad-principal coverage order mismatch")
        expected_commitment = _commit(
            STATE4_FILE_PERMISSION_METADATA_FIXED_PRINCIPAL_DOMAIN,
            authorization_hash,
            expected_sid,
        )
        if item["principal_commitment"] != expected_commitment:
            raise StoragePreflightError("State 4 broad-principal commitment mismatch")
        fixed_commitments.add(expected_commitment)
        has_access = effective_read or effective_write
        expected_blocking = expected_to_block and has_access
        expected_residual = not expected_to_block and has_access
        if (
            item["blocking"] is not expected_blocking
            or item["residual_risk"] is not expected_residual
        ):
            raise StoragePreflightError("State 4 broad-principal risk classification mismatch")
        calculated_blocking += int(expected_blocking)
        calculated_residual += int(expected_residual)
        if expected_blocking:
            raise StoragePreflightError("State 4 blocking broad-principal access is present")
        if expected_residual:
            observed_residuals.add(expected_commitment)

    additional_count = _require_nonnegative_integer(
        permission["additional_allow_principal_count"],
        name="State 4 additional allow-principal count",
    )
    additional = permission["additional_allow_principal_effective_access"]
    if not isinstance(additional, list) or len(additional) != additional_count:
        raise StoragePreflightError("State 4 additional allow-principal coverage is incomplete")
    additional_commitments: list[str] = []
    for value in additional:
        item, effective_read, effective_write = _validate_state4_permission_metadata_access(
            value,
            name="State 4 additional allow-principal evidence",
            expected_keys=STATE4_FILE_PERMISSION_METADATA_ADDITIONAL_ACCESS_KEYS,
        )
        if item["principal_class"] != "additional_allow_ace":
            raise StoragePreflightError("State 4 additional allow-principal class is invalid")
        scope = item["principal_scope"]
        if not isinstance(scope, str) or scope not in {
            "local_user", "untrusted_service", "unresolved",
        }:
            raise StoragePreflightError("State 4 additional principal scope is invalid")
        commitment = str(item["principal_commitment"])
        if commitment in fixed_commitments:
            raise StoragePreflightError("fixed broad principal is repeated as an additional principal")
        additional_commitments.append(commitment)
        has_access = effective_read or effective_write
        expected_blocking = scope in {"untrusted_service", "unresolved"} and has_access
        expected_residual = scope == "local_user" and has_access
        if (
            item["blocking"] is not expected_blocking
            or item["residual_risk"] is not expected_residual
        ):
            raise StoragePreflightError("State 4 additional-principal risk classification mismatch")
        calculated_blocking += int(expected_blocking)
        calculated_residual += int(expected_residual)
        if expected_blocking:
            raise StoragePreflightError("prohibited untrusted or unresolved principal access is present")
        if expected_residual:
            observed_residuals.add(commitment)
    if (
        additional_commitments != sorted(additional_commitments)
        or len(set(additional_commitments)) != len(additional_commitments)
    ):
        raise StoragePreflightError(
            "State 4 additional principal commitments must be sorted and unique"
        )

    blocking_count = _require_nonnegative_integer(
        permission["blocking_broad_access_count"],
        name="State 4 blocking broad-access count",
    )
    residual_count = _require_nonnegative_integer(
        permission["broader_local_user_residual_count"],
        name="State 4 broader local-user residual count",
    )
    if blocking_count != calculated_blocking:
        raise StoragePreflightError("State 4 blocking broad-access count mismatch")
    if residual_count != calculated_residual:
        raise StoragePreflightError("State 4 broader local-user residual count mismatch")
    if observed_residuals != expected_residuals:
        raise StoragePreflightError("State 4 accepted local-user residual set mismatch")

    claimed_self_hash = row["record_self_sha256"]
    if not isinstance(claimed_self_hash, str):
        raise StoragePreflightError("State 4 file permission metadata self-hash is not text")
    _require_sha(claimed_self_hash, name="State 4 file permission metadata self-hash")
    payload = dict(row)
    payload.pop("record_self_sha256")
    if claimed_self_hash != _sha(_canonical(payload)):
        raise StoragePreflightError("State 4 file permission metadata self-hash mismatch")


def validate_storage_preflight_record(record: Mapping[str, Any]) -> None:
    _require_exact_keys(record, RECORD_KEYS, name="storage preflight record")
    if record["schema"] != "operatormark.darcy_branch_c_v1.private_storage_preflight.v1":
        raise StoragePreflightError("storage preflight schema mismatch")
    if record["destination_relative_path"] != DESTINATION_RELATIVE:
        raise StoragePreflightError("storage destination mismatch")
    if record["canonical_resolution_method"] not in {CANONICAL_METHOD_WINDOWS, CANONICAL_METHOD_POSIX}:
        raise StoragePreflightError("canonical resolution method is unsupported or overstated")
    if record["all_mandatory_checks_passed"] is not True:
        raise StoragePreflightError("mandatory storage checks did not all pass")
    if not isinstance(record["state"], str) or not record["state"]:
        raise StoragePreflightError("state must be a non-empty string")
    if not isinstance(record["residual_risks"], list) or not all(
            isinstance(item, str) and item for item in record["residual_risks"]):
        raise StoragePreflightError("residual_risks must remain an explicit string list")
    estimated = _require_nonnegative_integer(record["estimated_accepted_bytes"],
                                             name="estimated_accepted_bytes")
    projected = _require_nonnegative_integer(record["projected_audit_bytes"],
                                             name="projected_audit_bytes")
    required = _require_nonnegative_integer(record["required_free_bytes"], name="required_free_bytes")
    observed = _require_nonnegative_integer(record["observed_free_bytes"], name="observed_free_bytes")
    if required != required_free_bytes(estimated, projected) or observed < required:
        raise StoragePreflightError("storage free-space formula or threshold mismatch")
    components = record["path_component_evidence"]
    expected_component_count = len(PurePosixPath(DESTINATION_RELATIVE).parts)
    if not isinstance(components, list) or len(components) != expected_component_count:
        raise StoragePreflightError(
            "path component evidence must cover the repository root and every destination parent"
        )
    expected_prefixes: list[str] = []
    component_volume_commitment: str | None = None
    component_file_identities: set[str] = set()
    for index, component_value in enumerate(components):
        component = _require_exact_keys(component_value, PATH_COMPONENT_KEYS,
                                        name="path component evidence")
        if index == 0:
            expected_relative = "."
        else:
            expected_prefixes.append(PurePosixPath(DESTINATION_RELATIVE).parts[index - 1])
            expected_relative = PurePosixPath(*expected_prefixes).as_posix()
        if component["relative_component"] != expected_relative:
            raise StoragePreflightError("path component ordering or identity mismatch")
        if any(component[name] is not True for name in
               ("exists", "is_directory", "canonical_name_exact", "unicode_nfc_exact",
                "same_volume_as_repository_root", "stable_before_after")):
            raise StoragePreflightError("path component did not pass all mandatory checks")
        if component["is_reparse_point"] is not False or component["reparse_tag_hex"] != "0x0":
            raise StoragePreflightError("path component reports reparse evidence")
        if not isinstance(component["file_attributes_hex"], str):
            raise StoragePreflightError("path component attributes are invalid")
        try:
            attributes = int(component["file_attributes_hex"], 16)
        except ValueError as exc:
            raise StoragePreflightError("path component attributes are not hexadecimal") from exc
        if attributes & REPARSE_ATTRIBUTE:
            raise StoragePreflightError("path component attributes contain reparse flag")
        if record["canonical_resolution_method"] == CANONICAL_METHOD_WINDOWS:
            if not (attributes & 0x00000010):
                raise StoragePreflightError("Windows path component lacks directory attributes")
            if attributes & CLOUD_ATTRIBUTES:
                raise StoragePreflightError(
                    "Windows path component has cloud, offline, pinned, unpinned, or recall attributes"
                )
        for name in ("canonical_path_commitment", "volume_identifier_commitment",
                     "file_identity_commitment"):
            _require_sha(component[name], name=name)
        if component_volume_commitment is None:
            component_volume_commitment = component["volume_identifier_commitment"]
        elif component["volume_identifier_commitment"] != component_volume_commitment:
            raise StoragePreflightError("path components do not share one volume commitment")
        if component["file_identity_commitment"] in component_file_identities:
            raise StoragePreflightError("path component file identities are not unique")
        component_file_identities.add(component["file_identity_commitment"])
    volume = _require_exact_keys(record["volume_evidence"], VOLUME_KEYS, name="volume evidence")
    if volume["is_fixed_local"] is not True or volume["free_bytes"] != observed:
        raise StoragePreflightError("volume locality or free-space equality mismatch")
    for name in ("allocation_unit_bytes", "capacity_bytes", "free_bytes", "maximum_component_length"):
        _require_nonnegative_integer(volume[name], name=f"volume.{name}")
    if volume["allocation_unit_bytes"] == 0 or volume["maximum_component_length"] == 0:
        raise StoragePreflightError("volume allocation unit or component limit is zero")
    if volume["capacity_bytes"] < volume["free_bytes"]:
        raise StoragePreflightError("volume free bytes exceed capacity")
    if record["canonical_resolution_method"] == CANONICAL_METHOD_WINDOWS:
        if volume["drive_type"] != 3:
            raise StoragePreflightError("Windows volume is not DRIVE_FIXED")
        if (
            not isinstance(volume["filesystem_type"], str)
            or not volume["filesystem_type"]
            or volume["filesystem_type"].casefold() in WINDOWS_REMOTE_FILESYSTEM_TYPES
        ):
            raise StoragePreflightError("Windows filesystem identity is missing or remote")
        if not isinstance(volume["filesystem_flags_hex"], str):
            raise StoragePreflightError("Windows filesystem flags are invalid")
        try:
            filesystem_flags = int(volume["filesystem_flags_hex"], 16)
        except ValueError as exc:
            raise StoragePreflightError("Windows filesystem flags are not hexadecimal") from exc
        if not (filesystem_flags & 0x00000002):
            raise StoragePreflightError("Windows filesystem does not report preserved filename case")
        if volume["case_sensitivity_behavior"] not in {
            "enabled",
            "disabled",
            "query_unsupported_exact_directory_entry_scan_enforced",
        }:
            raise StoragePreflightError("Windows case-sensitivity behavior is invalid")
        if volume["path_length_capability"] != "not_tested_no_extended_length_path_claim":
            raise StoragePreflightError("path-length evidence overstates tested capability")
    for name in ("volume_identifier_commitment", "volume_root_commitment"):
        _require_sha(volume[name], name=name)
    if component_volume_commitment != volume["volume_identifier_commitment"]:
        raise StoragePreflightError("path-component and volume identity commitments differ")
    sync = _require_exact_keys(record["sync_detection_evidence"], SYNC_KEYS, name="sync evidence")
    if (sync["inspection_complete"] is not True
            or sync["destination_in_detected_sync_root"] is not False
            or sync["cloud_attribute_components"] != []):
        raise StoragePreflightError("synchronization exclusion did not pass")
    if not isinstance(sync["environment_names_checked"], list) or not isinstance(
            sync["registry_sources_checked"], list):
        raise StoragePreflightError("synchronization inspection coverage is invalid")
    if not all(isinstance(item, str) and item for item in sync["registry_sources_checked"]):
        raise StoragePreflightError("synchronization registry-source evidence is invalid")
    if sync["environment_names_checked"] != list(SYNC_ENVIRONMENT_NAMES):
        raise StoragePreflightError("synchronization environment coverage or order mismatch")
    required_registry_sources = {
        "HKCU:OneDrive.Accounts",
        "HKCU:SyncRootManager",
        "HKCU:UserShellFolders",
    }
    if (
        sync["registry_sources_checked"] != sorted(set(sync["registry_sources_checked"]))
        or not required_registry_sources.issubset(set(sync["registry_sources_checked"]))
    ):
        raise StoragePreflightError("synchronization registry coverage or order mismatch")
    if not isinstance(sync["configured_sync_roots"], list) or not isinstance(
            sync["active_sync_clients"], list):
        raise StoragePreflightError("synchronization evidence lists are invalid")
    for value in sync["configured_sync_roots"]:
        item = _require_exact_keys(value, SYNC_ROOT_KEYS, name="sync root evidence")
        if not isinstance(item["provider"], str) or not item["provider"]:
            raise StoragePreflightError("sync root provider is invalid")
        if item["source"] not in {
            "environment",
            "registry",
            "sync_root_manager",
            "sync_root_manager_user_roots",
            "user_shell_folders",
        }:
            raise StoragePreflightError("sync root source is invalid")
        if not isinstance(item["exists"], bool) or not isinstance(item["handle_canonicalized"], bool):
            raise StoragePreflightError("sync root resolution flags are invalid")
        if item["handle_canonicalized"] is not item["exists"]:
            raise StoragePreflightError("existing sync root lacks handle-canonical evidence")
        _require_sha(item["lexical_root_commitment"], name="sync lexical root commitment")
        if item["exists"]:
            _require_sha(item["canonical_root_commitment"], name="sync canonical root commitment")
        elif item["canonical_root_commitment"] is not None:
            raise StoragePreflightError("non-existing sync root has false canonical evidence")
    expected_sync_order = sorted(
        sync["configured_sync_roots"],
        key=lambda item: (
            item["provider"], item["source"], item["lexical_root_commitment"]
        ),
    )
    if sync["configured_sync_roots"] != expected_sync_order:
        raise StoragePreflightError("configured sync-root order is non-canonical")
    for value in sync["active_sync_clients"]:
        item = _require_exact_keys(value, SYNC_CLIENT_KEYS, name="sync client evidence")
        _require_sha(item["process_name_commitment"], name="sync process commitment")
        if _require_nonnegative_integer(item["configured_root_count"],
                                        name="configured_root_count") == 0:
            raise StoragePreflightError("active sync client lacks a configured root")
        matching_roots = sum(
            1
            for root in sync["configured_sync_roots"]
            if root["provider"] == item["provider"]
            and root["exists"] is True
            and root["handle_canonicalized"] is True
        )
        if item["configured_root_count"] != matching_roots:
            raise StoragePreflightError("active sync-client root count mismatch")
    if sync["active_sync_clients"] != sorted(
        sync["active_sync_clients"],
        key=lambda item: (item["provider"], item["process_name_commitment"]),
    ):
        raise StoragePreflightError("active sync-client order is non-canonical")
    ignore = _require_exact_keys(record["git_ignore_evidence"], GIT_IGNORE_KEYS,
                                 name="Git ignore evidence")
    if (ignore["matched"] is not True or ignore["all_candidates_matched_intended_rule"] is not True
            or ignore["intended_rule"] != ".gitignore:artifacts/"):
        raise StoragePreflightError("intended Git ignore proof failed")
    candidate_count = _require_nonnegative_integer(
        ignore["candidate_count"], name="Git ignore candidate_count"
    )
    if candidate_count < 1:
        raise StoragePreflightError("Git ignore candidate coverage is empty")
    candidate_commitments = ignore["candidate_path_commitments"]
    if (
        not isinstance(candidate_commitments, list)
        or len(candidate_commitments) != candidate_count
    ):
        raise StoragePreflightError("Git candidate-path commitments do not match candidate count")
    for commitment in candidate_commitments:
        _require_sha(commitment, name="Git candidate-path commitment")
    if len(set(candidate_commitments)) != candidate_count:
        raise StoragePreflightError("Git candidate-path commitments are duplicated")
    expected_destination_commitment = _commit(
        "operatormark-storage-git-candidate-path-v1",
        str(record["authorization_file_sha256"]),
        DESTINATION_RELATIVE,
    )
    if candidate_commitments[0] != expected_destination_commitment:
        raise StoragePreflightError("Git candidate paths are not bound to the frozen destination")
    _require_sha(ignore["check_ignore_output_sha256"], name="check-ignore output hash")
    tracking = _require_exact_keys(record["git_tracking_evidence"], GIT_TRACKING_KEYS,
                                   name="Git tracking evidence")
    for name in (
        "tracked_path_count",
        "submodule_path_count",
        "nested_worktree_count",
        "nested_repository_marker_count",
        "candidate_collision_count",
    ):
        if _require_nonnegative_integer(tracking[name], name=f"Git tracking {name}") != 0:
            raise StoragePreflightError("Git tracking or collision evidence is non-zero")
    _require_sha(tracking["worktree_evidence_sha256"], name="worktree evidence hash")
    if tracking["branch"] != "main":
        raise StoragePreflightError("record is not bound to branch main")
    if tracking["head_commit"] != record["implementation_commit"]:
        raise StoragePreflightError("record HEAD and implementation commit differ")
    if tracking["head_tree"] != record["implementation_tree"]:
        raise StoragePreflightError("record tree and implementation tree differ")
    for name in ("head_commit", "head_tree", "parent_commit"):
        _require_git_oid(tracking[name], name=f"Git {name}")
    if any(
        tracking[name] is not True
        for name in (
            "tracked_status_clean",
            "staged_status_clean",
            "untracked_inclusive_status_clean",
        )
    ):
        raise StoragePreflightError("recorded Git status is not completely clean")
    if tracking["status_porcelain_sha256"] != _sha(b""):
        raise StoragePreflightError("clean Git porcelain commitment mismatch")
    probe = _require_exact_keys(record["probe_evidence"], PROBE_KEYS, name="probe evidence")
    if (any(probe[name] is not True for name in
            ("passed", "same_volume_verified", "atomic_directory_rename_verified"))
            or any(probe[name] is not False for name in
                   ("probe_file_preserved", "probe_directory_preserved"))):
        raise StoragePreflightError("fixed probe did not establish all required operations")
    if probe["fixed_literal_sha256"] != _sha(PROBE_BYTES):
        raise StoragePreflightError("probe literal commitment mismatch")
    if probe["operations"] != PROBE_OPERATIONS:
        raise StoragePreflightError("probe operation sequence is incomplete")
    for name in ("fixed_literal_sha256", "source_volume_identifier_commitment",
                 "renamed_volume_identifier_commitment"):
        _require_sha(probe[name], name=name)
    if probe["source_volume_identifier_commitment"] != probe["renamed_volume_identifier_commitment"]:
        raise StoragePreflightError("probe volume commitment changed across rename")
    if probe["source_volume_identifier_commitment"] != volume["volume_identifier_commitment"]:
        raise StoragePreflightError("probe and destination volume commitments differ")
    permission = _require_exact_keys(record["permission_evidence"], PERMISSION_KEYS,
                                     name="permission evidence")
    rights = _require_exact_keys(permission["current_process_effective_rights"], RIGHTS_KEYS,
                                 name="effective rights")
    if not all(value is True for value in rights.values()):
        raise StoragePreflightError("current process rights are incomplete")
    blocking_count = _require_nonnegative_integer(
        permission["blocking_broad_access_count"], name="blocking broad-access count"
    )
    if permission["dacl_present"] is not True or blocking_count != 0:
        raise StoragePreflightError("permission evidence reports unsafe broad access")
    if permission["platform"] != "nt" or permission["inspection_method"] != (
        "GetNamedSecurityInfoW plus GetEffectiveRightsFromAclW plus operation probe"
    ):
        raise StoragePreflightError("permission inspection method is invalid")
    if not isinstance(permission["inheritance_enabled"], bool):
        raise StoragePreflightError("ACL inheritance state is not Boolean")
    residual_count = _require_nonnegative_integer(
        permission["broader_local_user_residual_count"],
        name="broader local-user residual count",
    )
    for name in ("security_descriptor_commitment", "owning_principal_commitment"):
        _require_sha(permission[name], name=name)
    broad = permission["broad_principal_effective_access"]
    if not isinstance(broad, list) or len(broad) != len(_BROAD_PRINCIPALS):
        raise StoragePreflightError("broad-principal coverage is incomplete")
    calculated_blocking = 0
    calculated_residual = 0
    for value, (expected_class, expected_sid, expected_to_block) in zip(
        broad, _BROAD_PRINCIPALS, strict=True
    ):
        item = _require_exact_keys(value, BROAD_ACCESS_KEYS, name="broad-principal evidence")
        if item["principal_class"] != expected_class:
            raise StoragePreflightError("broad-principal coverage order mismatch")
        _require_sha(item["principal_commitment"], name="principal commitment")
        expected_principal_commitment = _commit(
            "operatormark-storage-principal-v1",
            str(record["authorization_file_sha256"]),
            expected_sid,
        )
        if item["principal_commitment"] != expected_principal_commitment:
            raise StoragePreflightError("broad-principal commitment does not bind the fixed SID")
        if not isinstance(item["effective_mask_hex"], str):
            raise StoragePreflightError("broad-principal effective mask is invalid")
        try:
            effective_mask = int(item["effective_mask_hex"], 16)
        except ValueError as exc:
            raise StoragePreflightError("broad-principal effective mask is invalid") from exc
        expected_read = bool(effective_mask & ACL_READ_MASK)
        expected_write = bool(effective_mask & ACL_WRITE_MASK)
        if item["effective_read"] is not expected_read or item["effective_write"] is not expected_write:
            raise StoragePreflightError("broad-principal effective-mask interpretation mismatch")
        has_access = expected_read or expected_write
        expected_blocking = expected_to_block and has_access
        expected_residual = not expected_to_block and has_access
        if item["blocking"] is not expected_blocking or item["residual_risk"] is not expected_residual:
            raise StoragePreflightError("broad-principal risk classification mismatch")
        calculated_blocking += int(expected_blocking)
        calculated_residual += int(expected_residual)
        if expected_blocking:
            raise StoragePreflightError("blocking broad-principal access is present")
    additional = permission["additional_allow_principal_effective_access"]
    additional_count = _require_nonnegative_integer(
        permission["additional_allow_principal_count"],
        name="additional allow-principal count",
    )
    if not isinstance(additional, list) or len(additional) != additional_count:
        raise StoragePreflightError("additional allow-principal coverage is incomplete")
    if not all(isinstance(item, Mapping) for item in additional):
        raise StoragePreflightError("additional allow-principal entries are invalid")
    if additional != sorted(additional, key=lambda item: item.get("principal_commitment", "")):
        raise StoragePreflightError("additional allow-principal order is non-canonical")
    additional_commitments: set[str] = set()
    for value in additional:
        item = _require_exact_keys(
            value, BROAD_ACCESS_KEYS, name="additional allow-principal evidence"
        )
        if item["principal_class"] != "additional_allow_ace":
            raise StoragePreflightError("additional allow-principal class is invalid")
        _require_sha(item["principal_commitment"], name="additional principal commitment")
        if item["principal_commitment"] in additional_commitments:
            raise StoragePreflightError("additional allow-principal commitments are duplicated")
        additional_commitments.add(item["principal_commitment"])
        if not isinstance(item["effective_mask_hex"], str):
            raise StoragePreflightError("additional principal effective mask is invalid")
        try:
            effective_mask = int(item["effective_mask_hex"], 16)
        except ValueError as exc:
            raise StoragePreflightError("additional principal effective mask is invalid") from exc
        expected_read = bool(effective_mask & ACL_READ_MASK)
        expected_write = bool(effective_mask & ACL_WRITE_MASK)
        expected_residual = expected_read or expected_write
        if (
            item["effective_read"] is not expected_read
            or item["effective_write"] is not expected_write
            or item["blocking"] is not False
            or item["residual_risk"] is not expected_residual
        ):
            raise StoragePreflightError("additional principal risk classification mismatch")
        calculated_residual += int(expected_residual)
    if calculated_blocking != blocking_count:
        raise StoragePreflightError("blocking broad-principal count mismatch")
    if calculated_residual != residual_count:
        raise StoragePreflightError("broader local-user residual count mismatch")
    encryption = _require_exact_keys(record["encryption_evidence"], ENCRYPTION_KEYS,
                                     name="encryption evidence")
    if encryption["encryption_presence_establishes_legal_or_cryptographic_protection"] is not False:
        raise StoragePreflightError("encryption evidence overstates protection")
    if not isinstance(encryption["filesystem_supports_encryption"], bool):
        raise StoragePreflightError("filesystem encryption-support evidence is not Boolean")
    encrypted_count = _require_nonnegative_integer(
        encryption["encrypted_ancestor_count"], name="encrypted ancestor count"
    )
    expected_encrypted_count = sum(
        1
        for component in components
        if int(component["file_attributes_hex"], 16) & ENCRYPTED_ATTRIBUTE
    )
    if encrypted_count != expected_encrypted_count:
        raise StoragePreflightError("encrypted ancestor count mismatch")
    if record["canonical_resolution_method"] == CANONICAL_METHOD_WINDOWS and (
        encryption["filesystem_supports_encryption"]
        is not bool(filesystem_flags & 0x00020000)
    ):
        raise StoragePreflightError("filesystem encryption-support flag mismatch")
    allowed_encryption_interpretations = {
        "observed": {
            "reported_protected_not_independently_proven",
            "reported_not_protected",
            "observed_but_unclassified",
        },
        "query_failed": {"unknown_query_failed"},
        "unavailable": {"unknown_query_unavailable"},
    }
    query_outcome = encryption["device_encryption_query_outcome"]
    if (
        query_outcome not in allowed_encryption_interpretations
        or encryption["device_encryption_interpreted_status"]
        not in allowed_encryption_interpretations[query_outcome]
    ):
        raise StoragePreflightError("device-encryption query/interpretation mismatch")
    if query_outcome == "unavailable":
        if encryption["device_encryption_evidence_commitment"] is not None:
            raise StoragePreflightError("unavailable encryption query has false evidence")
    else:
        _require_sha(encryption["device_encryption_evidence_commitment"],
                     name="device encryption evidence commitment")
    if encryption["status"] != "evidence_only_with_key_management_and_access_residuals":
        raise StoragePreflightError("encryption residual-risk status mismatch")
    backup = _require_exact_keys(record["backup_and_indexing_evidence"], BACKUP_KEYS,
                                 name="backup evidence")
    if backup["administrator_backup_malware_physical_access_and_remanence_eliminated"] is not False:
        raise StoragePreflightError("backup/residual-risk evidence overstates protection")
    for name in (
        "administrator_access_may_expose_copies",
        "backup_software_may_expose_copies",
        "indexing_may_expose_copies",
        "antivirus_may_expose_copies",
        "crash_dumps_may_expose_copies",
        "paging_may_expose_copies",
        "local_sync_clients_may_expose_copies",
    ):
        if not isinstance(backup[name], bool):
            raise StoragePreflightError(f"backup exposure field {name} is not Boolean")
    for name in (
        "administrator_access_may_expose_copies",
        "backup_software_may_expose_copies",
        "antivirus_may_expose_copies",
        "crash_dumps_may_expose_copies",
        "paging_may_expose_copies",
    ):
        if backup[name] is not True:
            raise StoragePreflightError(f"conservative exposure field {name} may not be false")
    expected_indexing_risk = any(
        not (int(component["file_attributes_hex"], 16) & NOT_CONTENT_INDEXED_ATTRIBUTE)
        for component in components
    )
    if backup["indexing_may_expose_copies"] is not expected_indexing_risk:
        raise StoragePreflightError("indexing exposure interpretation mismatch")
    if backup["local_sync_clients_may_expose_copies"] is not bool(
        sync["active_sync_clients"]
    ):
        raise StoragePreflightError("local sync-client exposure interpretation mismatch")
    commitments = backup["service_query_evidence_commitments"]
    outcomes = backup["service_query_outcomes"]
    if (
        not isinstance(commitments, Mapping)
        or not isinstance(outcomes, Mapping)
        or set(commitments) != {"WSearch", "wbengine", "VSS"}
        or set(outcomes) != {"WSearch", "wbengine", "VSS"}
    ):
        raise StoragePreflightError("service-query evidence is invalid")
    for service in ("WSearch", "wbengine", "VSS"):
        if outcomes[service] not in {"observed", "query_failed", "unavailable"}:
            raise StoragePreflightError("service-query outcome is invalid")
        if outcomes[service] == "unavailable":
            if commitments[service] is not None:
                raise StoragePreflightError("unavailable service query has false evidence")
        else:
            _require_sha(commitments[service], name="service-query commitment")
    if backup["registry_query_outcome"] not in {"observed", "partially_observed", "unavailable"}:
        raise StoragePreflightError("registry-query outcome is invalid")
    _require_sha(backup["registry_query_evidence_commitment"], name="registry-query commitment")
    if backup["status"] != (
        "administrator_backup_indexing_antivirus_crash_dump_paging_malware_"
        "physical_access_key_management_and_deletion_remanence_risks_remain"
    ):
        raise StoragePreflightError("backup/indexing residual-risk status mismatch")
    expected_residual_risks = [
        "administrator backup malware physical-access key-management and deletion-remanence risks remain"
    ]
    if residual_count:
        expected_residual_risks.append(
            "broader local-user effective access remains and requires explicit State 4 acceptance"
        )
    if record["residual_risks"] != expected_residual_risks:
        raise StoragePreflightError("residual-risk list or order mismatch")
    for name in ("protocol_file_sha256", "authorization_file_sha256", "repository_root_commitment",
                 "destination_absolute_path_commitment", "record_self_sha256"):
        _require_sha(record[name], name=name)
    for name in ("implementation_commit", "implementation_tree"):
        _require_git_oid(record[name], name=name)
    payload = dict(record)
    claimed = payload.pop("record_self_sha256")
    if claimed != _sha(_canonical(payload)):
        raise StoragePreflightError("storage preflight self-hash mismatch")


def compare_storage_preflight_records(
    frozen: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Require exact repeated evidence except free space and its derived self-hash."""
    validate_storage_preflight_record(frozen)
    validate_storage_preflight_record(current)
    frozen_payload = json.loads(_canonical(frozen))
    current_payload = json.loads(_canonical(current))
    frozen_hash = frozen_payload.pop("record_self_sha256")
    current_hash = current_payload.pop("record_self_sha256")
    differences: list[str] = []
    for top_level, nested in (("observed_free_bytes", None), ("volume_evidence", "free_bytes")):
        if nested is None:
            if frozen_payload[top_level] != current_payload[top_level]:
                differences.append(top_level)
            frozen_payload[top_level] = "<VOLATILE_FREE_SPACE>"
            current_payload[top_level] = "<VOLATILE_FREE_SPACE>"
        else:
            if frozen_payload[top_level][nested] != current_payload[top_level][nested]:
                differences.append(f"{top_level}.{nested}")
            frozen_payload[top_level][nested] = "<VOLATILE_FREE_SPACE>"
            current_payload[top_level][nested] = "<VOLATILE_FREE_SPACE>"
    if frozen_payload != current_payload:
        raise StoragePreflightError(
            "repeated storage preflight differs outside volatile free-space fields"
        )
    return {
        "matched": True, "permitted_differences": differences,
        "record_self_sha256_changed_only_due_to_permitted_fields": frozen_hash != current_hash,
    }


_METADATA_FORBIDDEN_MODULE_PREFIXES: Final = (
    "numpy", "scipy", "operatormark.corpus.microbenchmark",
    "operatormark.corpus.rng_guard", "operatormark.corpus.state4",
    "operatormark.benchmarks", "operatormark.solvers", "operatormark.evaluation",
)


def _assert_metadata_import_boundary() -> None:
    loaded = sorted(
        name for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".")
               for prefix in _METADATA_FORBIDDEN_MODULE_PREFIXES)
    )
    if loaded:
        raise StoragePreflightError(
            "metadata-only storage CLI has a forbidden import closure: " + ",".join(loaded)
        )


def _administrative_output_path(root: Path, value: str) -> Path:
    relative = _validate_relative_path(value, name="output")
    if relative.suffix.casefold() != ".json" or not relative.parts or relative.parts[0] != "artifacts":
        raise StoragePreflightError("storage CLI output must be one JSON file under artifacts")
    if "private_corpus_v1" in {part.casefold() for part in relative.parts}:
        raise StoragePreflightError("storage CLI evidence must remain outside private_corpus_v1")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise StoragePreflightError("storage CLI output parent cannot be a symbolic link")
    return candidate


def _require_ignored_output(root: Path, relative_output: str) -> None:
    result = _run_git(root, "check-ignore", "--no-index", "--quiet", "--", relative_output)
    if result.returncode != 0 or result.stdout or result.stderr:
        raise StoragePreflightError("storage CLI output is not covered by a repository ignore rule")


def _storage_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Metadata-only OperatorMark State 3 storage-preflight entry point"
    )
    parser.add_argument("--protocol-file-sha256", required=True)
    parser.add_argument("--authorization-file-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--implementation-tree", required=True)
    parser.add_argument("--expected-parent-commit", required=True)
    parser.add_argument("--estimated-accepted-bytes", required=True, type=int)
    parser.add_argument("--projected-audit-bytes", required=True, type=int)
    parser.add_argument("--attempt-relative-path", action="append", default=[])
    parser.add_argument("--output", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    preflight_runner: Callable[..., Mapping[str, Any]] | None = None,
    repository_root: str | Path | None = None,
    enforce_import_boundary: bool = True,
    ignore_validator: Callable[[Path, str], None] | None = None,
) -> int:
    """Run one bound storage preflight and exclusively publish its public record.

    Dependency-injection parameters exist solely for State 2 non-corpus tests.  A
    command-line invocation always uses the committed real preflight and Git
    ignore validator.
    """

    if enforce_import_boundary:
        _assert_metadata_import_boundary()
    arguments = _storage_cli_parser().parse_args(argv)
    if arguments.protocol_file_sha256 != PROTOCOL_SHA256:
        raise StoragePreflightError("storage CLI protocol commitment mismatch")
    _require_sha(arguments.authorization_file_sha256, name="authorization_file_sha256")
    for name, value in (
        ("implementation_commit", arguments.implementation_commit),
        ("implementation_tree", arguments.implementation_tree),
        ("expected_parent_commit", arguments.expected_parent_commit),
    ):
        _require_git_oid(value, name=name)
    root = (
        discover_repository_root(Path.cwd())
        if repository_root is None else Path(repository_root).resolve(strict=True)
    )
    relative_output = _validate_relative_path(arguments.output, name="output").as_posix()
    output = _administrative_output_path(root, relative_output)
    validator = _require_ignored_output if ignore_validator is None else ignore_validator
    validator(root, relative_output)
    request = StoragePreflightRequest(
        state=STATE3_STORAGE_STATE,
        protocol_file_sha256=arguments.protocol_file_sha256,
        authorization_file_sha256=arguments.authorization_file_sha256,
        implementation_commit=arguments.implementation_commit,
        implementation_tree=arguments.implementation_tree,
        expected_parent_commit=arguments.expected_parent_commit,
        estimated_accepted_bytes=arguments.estimated_accepted_bytes,
        projected_audit_bytes=arguments.projected_audit_bytes,
        attempt_relative_paths=tuple(arguments.attempt_relative_path),
    )
    runner = run_storage_preflight if preflight_runner is None else preflight_runner
    record = dict(runner(request, start=root))
    validate_storage_preflight_record(record)
    if (
        record["protocol_file_sha256"] != arguments.protocol_file_sha256
        or record["authorization_file_sha256"] != arguments.authorization_file_sha256
        or record["implementation_commit"] != arguments.implementation_commit
        or record["implementation_tree"] != arguments.implementation_tree
        or record["git_tracking_evidence"]["parent_commit"] != arguments.expected_parent_commit
        or record["estimated_accepted_bytes"] != arguments.estimated_accepted_bytes
        or record["projected_audit_bytes"] != arguments.projected_audit_bytes
    ):
        raise StoragePreflightError("storage CLI result differs from its runtime bindings")
    from .canonical import write_canonical_json_exclusive

    data = write_canonical_json_exclusive(output, record)
    if enforce_import_boundary:
        _assert_metadata_import_boundary()
    receipt = {
        "authorization_file_sha256": arguments.authorization_file_sha256,
        "implementation_commit": arguments.implementation_commit,
        "implementation_tree": arguments.implementation_tree,
        "output_path": relative_output,
        "output_sha256": _sha(data),
        "output_size": len(data),
        "record_self_sha256": record["record_self_sha256"],
        "schema": "operatormark.darcy_branch_c_v1.state3_storage_preflight_receipt.v1",
    }
    sys.stdout.write(_canonical(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
