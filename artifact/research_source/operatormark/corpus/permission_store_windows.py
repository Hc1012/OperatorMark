"""Handle-bound Windows adapter for confidential permission-evidence retention.

The adapter is deliberately policy-free.  Every path, identity, descriptor hash,
volume binding, and profile binding is supplied independently and validated before
the first native effect.  Construction is inert; Win32 libraries are loaded only
by an invoked boundary method.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import unicodedata
from typing import Literal, cast

from operatormark.corpus.permission_native_win32 import (
    DWORD_MAX,
    Win32ApiError,
    Win32OwnedObjectHandle,
    Win32PermissionNativeApi,
)
from operatormark.corpus.permission_observation_windows import (
    ComponentFailureItem,
    NativeCallContext,
    NativeFileIdentity,
    NativeOpenSpecification,
    NativeStream,
    PermissionNativeCallError,
    PermissionObservationRequest,
    RawNativeObjectSnapshot,
    validate_permission_observation_request,
)
from operatormark.corpus.permission_store_native_win32 import (
    Win32EvidenceStoreBindings,
    Win32EvidenceStoreError,
)
from operatormark.corpus.private_permission_evidence import (
    ConfidentialNamespaceSnapshot,
    ConfidentialStoreExpectation,
    NamespaceOpenOutcome,
    PermissionAuthzEvidenceDestination,
    PermissionEvidenceDestination,
    PrivatePermissionEvidenceContractError,
    StoreBytesReadOutcome,
    StoreCreateOutcome,
    StoreCreationEvidence,
    StoreFailureEffect,
    StoreIoOutcome,
    StoreNamespaceEffect,
    StoreObjectSnapshot,
    StorePromoteOutcome,
    StoreReleaseOutcome,
    StoreSnapshotOutcome,
    StoreParentSnapshot,
    validate_confidential_store_expectation,
    validate_permission_authz_evidence_destination,
    validate_permission_evidence_destination,
)


_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_DWORD_RE = re.compile(r"^0x(?:0|[1-9a-f][0-9a-f]*)$")
_UINT64_RE = re.compile(r"^0x(?:0|[1-9a-f][0-9a-f]{0,15})$")
_FILE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_LEGACY_VOLUME_RE = re.compile(r"^[0-9a-f]{8}$")
_LEGACY_INDEX_RE = re.compile(r"^[0-9a-f]{16}$")
_VOLUME_GUID_ROOT_RE = re.compile(
    r"^\\\\\?\\Volume\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}\\[^\\/]+(?:\\[^\\/]+)*$"
)
_UNSUPPORTED_ATTRIBUTES = 0x400 | 0x1000 | 0x40000 | 0x400000
_NATIVE_TRANSFER_BYTES = DWORD_MAX


@dataclass(frozen=True, slots=True)
class Win32StoreVolumeBinding:
    canonical_volume_guid_root: str
    canonical_dos_root: str
    canonical_name_root: str
    volume_serial_number_hex: str
    volume_identifier_commitment: str
    confidential_store_profile_sha256: str


@dataclass(frozen=True, slots=True)
class Win32StoreDirectoryBinding:
    relative_path: str
    canonical_volume_guid_path: str
    canonical_dos_path: str
    canonical_name_exact: str
    concrete_identity: NativeFileIdentity
    opaque_object_identity: str
    security_descriptor_sha256: str


class _StoreFault(RuntimeError):
    def __init__(
        self,
        *,
        backend_error_hex: str | None = None,
        cleanup_uncertain: bool = False,
    ) -> None:
        super().__init__("Windows store boundary failed")
        self.backend_error_hex = backend_error_hex
        self.cleanup_uncertain = cleanup_uncertain


@dataclass(slots=True)
class _HeldDirectory:
    binding: Win32StoreDirectoryBinding
    ordinal: int
    handle: Win32OwnedObjectHandle


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise PrivatePermissionEvidenceContractError(f"{name} is not lowercase SHA-256")
    return value


def _nfc(value: object, name: str, *, nonempty: bool = True) -> str:
    if (
        not isinstance(value, str)
        or (nonempty and not value)
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise PrivatePermissionEvidenceContractError(f"{name} is not exact NFC text")
    return value


def _identity(value: object, name: str) -> NativeFileIdentity:
    if type(value) is not NativeFileIdentity:
        raise PrivatePermissionEvidenceContractError(f"{name} has the wrong type")
    result = cast(NativeFileIdentity, value)
    if (
        _UINT64_RE.fullmatch(result.volume_serial_number_hex) is None
        or int(result.volume_serial_number_hex, 16) > 0xFFFFFFFFFFFFFFFF
        or _FILE_ID_RE.fullmatch(result.file_id_128_hex) is None
        or _LEGACY_VOLUME_RE.fullmatch(result.legacy_volume_serial_number_hex) is None
        or _LEGACY_INDEX_RE.fullmatch(result.legacy_file_index_hex) is None
    ):
        raise PrivatePermissionEvidenceContractError(f"{name} is malformed")
    return result


def _join(root: str, relative_path: str) -> str:
    if not relative_path:
        return root
    return root + "\\" + relative_path.replace("/", "\\")


def _expected_directories(records_parent: str) -> tuple[str, ...]:
    pieces = records_parent.split("/")
    result = [""]
    for index in range(len(pieces)):
        result.append("/".join(pieces[: index + 1]))
    return tuple(result)


def _backend_hex(value: object) -> str | None:
    if isinstance(value, _StoreFault):
        return value.backend_error_hex
    if isinstance(value, Win32EvidenceStoreError):
        code = value.backend_error_code
        if isinstance(code, int) and not isinstance(code, bool) and 0 < code <= DWORD_MAX:
            return f"0x{code:x}"
    if isinstance(value, PermissionNativeCallError):
        candidate = value.failure.backend_error_hex
        if isinstance(candidate, str) and _DWORD_RE.fullmatch(candidate):
            return candidate
    if isinstance(value, Win32ApiError):
        code = value.error_code
        if isinstance(code, int) and not isinstance(code, bool) and 0 < code <= DWORD_MAX:
            return f"0x{code:x}"
    return None


def _object_identity(value: NativeFileIdentity) -> str:
    """Losslessly label a newly measured strong Win32 identity."""

    return (
        "win32-file-id-v1:"
        f"{value.volume_serial_number_hex}:{value.file_id_128_hex}"
    )


def _failure(
    operation: str,
    stage: str,
    reason: str,
    *,
    role: Literal[
        "primary_payload", "retention_failure_payload",
        "authz_primary_payload", "authz_retention_failure_payload",
    ] | None = None,
    backend_error_hex: str | None = None,
    kind: Literal["store_operation_failure", "store_collision", "cleanup_failure"] = "store_operation_failure",
) -> ComponentFailureItem:
    return ComponentFailureItem(
        failure_kind=kind,
        stage=stage,
        component_operation=operation,
        resource_kind=(
            "store_object_handle"
            if role is not None
            else "namespace_handle"
            if operation == "release_namespace"
            else None
        ),
        resource_role=(
            role
            if role is not None
            else "records_namespace"
            if operation == "release_namespace"
            else None
        ),
        resource_ordinal=None,
        operation_ordinal=None,
        backend_error_hex=backend_error_hex,
        reason_code=reason,
    )


def _store_effect(
    operation: str,
    status: Literal["succeeded", "collision", "failed"],
    certainty: Literal["no_effect", "reported_effect", "unknown"],
    *,
    before: StoreObjectSnapshot | None = None,
    after: StoreObjectSnapshot | None = None,
    collision: str | None = None,
    requested: int | None = None,
    reported: int | None = None,
    backend_error_hex: str | None = None,
) -> StoreFailureEffect:
    return StoreFailureEffect(
        operation=operation,
        completion_status=status,
        effect_certainty=certainty,
        before_snapshot=before,
        after_snapshot=after,
        collision_relative_path=collision,
        bytes_requested=requested,
        bytes_reported=reported,
        backend_error_hex=backend_error_hex,
    )


class Win32ConfidentialEvidenceStore:
    """One independently bound Windows confidential-evidence destination."""

    def __init__(
        self,
        expected_request: PermissionObservationRequest,
        expected_store: ConfidentialStoreExpectation,
        expected_destination: (
            PermissionEvidenceDestination | PermissionAuthzEvidenceDestination
        ),
        *,
        volume_binding: Win32StoreVolumeBinding,
        directory_bindings: tuple[Win32StoreDirectoryBinding, ...],
        creation_sddl: str,
        bindings: object | None = None,
    ) -> None:
        if type(expected_request) is not PermissionObservationRequest:
            raise PrivatePermissionEvidenceContractError("request binding has the wrong type")
        if type(expected_store) is not ConfidentialStoreExpectation:
            raise PrivatePermissionEvidenceContractError("store binding has the wrong type")
        if type(expected_destination) not in {
            PermissionEvidenceDestination,
            PermissionAuthzEvidenceDestination,
        }:
            raise PrivatePermissionEvidenceContractError("destination binding has the wrong type")
        try:
            validate_permission_observation_request(
                expected_request, expected_request=expected_request
            )
        except Exception as exc:
            raise PrivatePermissionEvidenceContractError(
                "request binding is not a valid observation request"
            ) from exc
        validate_confidential_store_expectation(
            expected_store, expected_request=expected_request
        )
        if type(expected_destination) is PermissionEvidenceDestination:
            validate_permission_evidence_destination(
                expected_destination,
                expected_request=expected_request,
                expected_store=expected_store,
            )
        else:
            validate_permission_authz_evidence_destination(
                expected_destination,
                expected_request=expected_request,
                expected_store=expected_store,
            )
        if type(volume_binding) is not Win32StoreVolumeBinding:
            raise PrivatePermissionEvidenceContractError("volume binding has the wrong type")
        if type(directory_bindings) is not tuple or any(
            type(item) is not Win32StoreDirectoryBinding for item in directory_bindings
        ):
            raise PrivatePermissionEvidenceContractError("directory bindings are not an exact tuple")
        self._validate_bindings(
            expected_request,
            expected_store,
            expected_destination,
            volume_binding,
            directory_bindings,
            creation_sddl,
        )
        boundary = Win32EvidenceStoreBindings() if bindings is None else bindings
        for name in (
            "prepare_create_security", "create_secured_exclusive", "write_once_at",
            "flush_file", "rename_no_replace", "create_file", "close_handle",
            "get_final_path", "query_file_information", "get_legacy_file_information",
            "read_file_at", "get_security_info", "read_memory", "local_free",
        ):
            if not callable(getattr(boundary, name, None)):
                raise PrivatePermissionEvidenceContractError(
                    f"Win32 boundary omits {name}"
                )
        self.expected_request = expected_request
        self.expected_store = expected_store
        self.expected_destination = expected_destination
        self.volume_binding = volume_binding
        self.directory_bindings = directory_bindings
        self.creation_sddl = creation_sddl
        self._bindings = boundary
        self._native = Win32PermissionNativeApi(cast(object, boundary))
        self._open_count = 0

    @staticmethod
    def _validate_bindings(
        request: PermissionObservationRequest,
        store: ConfidentialStoreExpectation,
        destination: PermissionEvidenceDestination | PermissionAuthzEvidenceDestination,
        volume: Win32StoreVolumeBinding,
        directories: tuple[Win32StoreDirectoryBinding, ...],
        creation_sddl: str,
    ) -> None:
        guid_root = _nfc(volume.canonical_volume_guid_root, "canonical volume root")
        dos_root = _nfc(volume.canonical_dos_root, "canonical DOS root")
        name_root = _nfc(volume.canonical_name_root, "canonical name root")
        if (
            _VOLUME_GUID_ROOT_RE.fullmatch(guid_root) is None
            or dos_root.endswith("\\")
            or not name_root.startswith("\\")
            or name_root.endswith("\\")
        ):
            raise PrivatePermissionEvidenceContractError("canonical roots are malformed")
        root_parts = guid_root.split("\\")[4:]
        dos_parts = dos_root.split("\\")[1:]
        name_parts = name_root.split("\\")[1:]
        if (
            not re.fullmatch(r"[A-Za-z]:\\[^\\/]+(?:\\[^\\/]+)*", dos_root)
            or any(
                not part
                or part in {".", ".."}
                or part.endswith((".", " "))
                or any(ord(character) < 32 for character in part)
                for parts in (root_parts, dos_parts, name_parts)
                for part in parts
            )
        ):
            raise PrivatePermissionEvidenceContractError("canonical roots are unsafe")
        request_roots = request.subject.required_ancestor_canonical_volume_guid_paths
        if type(request_roots) is not tuple or not request_roots or request_roots[0] != guid_root:
            raise PrivatePermissionEvidenceContractError(
                "canonical repository root differs from the observation request"
            )
        if (
            _UINT64_RE.fullmatch(volume.volume_serial_number_hex) is None
            or int(volume.volume_serial_number_hex, 16) > 0xFFFFFFFFFFFFFFFF
        ):
            raise PrivatePermissionEvidenceContractError("volume serial binding is malformed")
        _sha(volume.volume_identifier_commitment, "volume commitment")
        _sha(volume.confidential_store_profile_sha256, "store profile")
        if not isinstance(creation_sddl, str) or not creation_sddl or "\x00" in creation_sddl:
            raise PrivatePermissionEvidenceContractError("creation SDDL is malformed")
        if unicodedata.normalize("NFC", creation_sddl) != creation_sddl:
            raise PrivatePermissionEvidenceContractError("creation SDDL is not NFC")
        if (
            request.context.volume_identifier_commitment
            != volume.volume_identifier_commitment
            or store.volume_identifier_commitment != volume.volume_identifier_commitment
            or destination.volume_identifier_commitment != volume.volume_identifier_commitment
        ):
            raise PrivatePermissionEvidenceContractError("independent volume bindings disagree")
        if (
            store.confidential_store_profile_sha256
            != volume.confidential_store_profile_sha256
            or destination.confidential_store_profile_sha256
            != volume.confidential_store_profile_sha256
        ):
            raise PrivatePermissionEvidenceContractError("independent profile bindings disagree")
        if (
            destination.required_created_object_security_descriptor_sha256
            != store.required_created_object_security_descriptor_sha256
        ):
            raise PrivatePermissionEvidenceContractError("creation descriptor bindings disagree")
        expected_relatives = _expected_directories(store.records_parent_relative_path)
        if tuple(item.relative_path for item in directories) != expected_relatives:
            raise PrivatePermissionEvidenceContractError("directory ancestry bindings are incomplete or reordered")
        for item in directories:
            relative = _nfc(item.relative_path, "directory relative path", nonempty=False)
            if relative and (relative.startswith("/") or relative.endswith("/") or "//" in relative or any(part in {"", ".", ".."} for part in relative.split("/"))):
                raise PrivatePermissionEvidenceContractError("directory relative path is unsafe")
            if item.canonical_volume_guid_path != _join(guid_root, relative):
                raise PrivatePermissionEvidenceContractError("directory GUID path is inconsistent")
            if item.canonical_dos_path != _join(dos_root, relative):
                raise PrivatePermissionEvidenceContractError("directory DOS path is inconsistent")
            if item.canonical_name_exact != _join(name_root, relative):
                raise PrivatePermissionEvidenceContractError("directory canonical name is inconsistent")
            identity = _identity(item.concrete_identity, "directory concrete identity")
            if identity.volume_serial_number_hex != volume.volume_serial_number_hex:
                raise PrivatePermissionEvidenceContractError("directory volume serial differs")
            _nfc(item.opaque_object_identity, "directory opaque identity")
            _sha(item.security_descriptor_sha256, "directory descriptor hash")
        namespace = next(item for item in directories if item.relative_path == store.namespace_relative_path)
        parent = next(item for item in directories if item.relative_path == store.records_parent_relative_path)
        if namespace.opaque_object_identity != store.namespace_object_identity:
            raise PrivatePermissionEvidenceContractError("namespace identity bindings disagree")
        if parent.opaque_object_identity != store.records_parent_object_identity:
            raise PrivatePermissionEvidenceContractError("records-parent identity bindings disagree")
        if parent.security_descriptor_sha256 != store.records_parent_security_descriptor_sha256:
            raise PrivatePermissionEvidenceContractError("records-parent descriptor bindings disagree")

    @property
    def native_boundary_loaded(self) -> bool:
        return bool(
            getattr(self._bindings, "loaded", False)
            or getattr(self._bindings, "store_boundary_loaded", False)
        )

    def _context(
        self,
        *,
        operation: str,
        stage: str,
        binding: tuple[str, str, int | None] | None,
        returned: tuple[str, str, int | None] | None,
        ordinal: int | None,
    ) -> NativeCallContext:
        return NativeCallContext(
            request_self_sha256=self.expected_request.request_self_sha256,
            stage=stage,
            component_operation=operation,
            resource_binding=cast(object, binding),
            allowed_returned_resource_binding=cast(object, returned),
            operation_ordinal=ordinal,
            raw_prefix_start_sequence=0,
        )

    @staticmethod
    def _release_error_resources(error: BaseException) -> bool:
        uncertain = False
        returned = getattr(error, "returned_resources", ())
        if type(returned) is not tuple:
            return True
        seen: set[int] = set()
        for item in returned:
            try:
                resource = item[0]
                if id(resource) in seen:
                    uncertain = True
                    continue
                seen.add(id(resource))
                if resource.release() is not None:
                    uncertain = True
            except Exception:
                uncertain = True
        return uncertain

    def _measure_security(
        self,
        handle: Win32OwnedObjectHandle,
        *,
        directory_ordinal: int | None,
        after: bool,
    ) -> str:
        subject = directory_ordinal is None
        resource = (
            "file_handle", "subject", None
        ) if subject else (
            "directory_handle", "ancestor", directory_ordinal
        )
        security = (
            "security_descriptor_buffer", "subject_security", None
        ) if subject else (
            "security_descriptor_buffer", "ancestor_security", directory_ordinal
        )
        ordinal = 0 if subject else 1 + cast(int, directory_ordinal)
        stage = "security_after" if after else "security_before"
        buffer = None
        result: str | None = None
        primary: _StoreFault | None = None
        try:
            buffer = self._native.acquire_security_descriptor(
                handle,
                context=self._context(
                    operation="acquire_security_descriptor", stage=stage,
                    binding=resource, returned=security, ordinal=ordinal,
                ),
            )
            data = buffer.copy_bytes()
            descriptor = self._native.parse_security_descriptor(
                data,
                context=self._context(
                    operation="parse_security_descriptor", stage=stage,
                    binding=security, returned=None, ordinal=ordinal,
                ),
            )
            if (
                descriptor.self_relative_descriptor_sha256 != hashlib.sha256(data).hexdigest()
                or not descriptor.dacl_present
                or not descriptor.dacl_protected
                or descriptor.dacl_defaulted
            ):
                raise _StoreFault()
            result = descriptor.self_relative_descriptor_sha256
        except PermissionNativeCallError as exc:
            uncertain = self._release_error_resources(exc)
            primary = _StoreFault(
                backend_error_hex=_backend_hex(exc), cleanup_uncertain=uncertain
            )
        except _StoreFault as exc:
            primary = exc
        except Exception as exc:
            primary = _StoreFault(backend_error_hex=_backend_hex(exc))
        cleanup: _StoreFault | None = None
        if buffer is not None:
            try:
                released = buffer.release()
            except Exception:
                cleanup = _StoreFault(cleanup_uncertain=True)
            else:
                if released is not None:
                    cleanup = _StoreFault(
                        backend_error_hex=released.backend_error_hex,
                        cleanup_uncertain=True,
                    )
        if primary is not None:
            if cleanup is not None:
                primary.cleanup_uncertain = True
            raise primary
        if cleanup is not None:
            raise cleanup
        if result is None:
            raise _StoreFault()
        return result

    def _validate_raw(
        self,
        raw: object,
        *,
        guid_path: str,
        dos_path: str,
        canonical_name: str,
        directory: bool,
        expected_identity: NativeFileIdentity | None,
    ) -> RawNativeObjectSnapshot:
        if type(raw) is not RawNativeObjectSnapshot:
            raise _StoreFault()
        value = cast(RawNativeObjectSnapshot, raw)
        identity = _identity(value.identity, "measured object identity")
        try:
            attributes = int(value.file_attributes_hex, 16)
            tag = int(value.reparse_tag_hex, 16)
        except Exception as exc:
            raise _StoreFault() from exc
        if (
            _DWORD_RE.fullmatch(value.file_attributes_hex) is None
            or _DWORD_RE.fullmatch(value.reparse_tag_hex) is None
            or value.canonical_volume_guid_path != guid_path
            or value.canonical_dos_path != dos_path
            or value.canonical_name_exact != canonical_name
            or not value.unicode_nfc_exact
            or identity.volume_serial_number_hex
            != self.volume_binding.volume_serial_number_hex
            or expected_identity is not None and identity != expected_identity
            or value.directory is not directory
            or value.delete_pending
            or tag != 0
            or attributes & _UNSUPPORTED_ATTRIBUTES
            or value.end_of_file_bytes < 0
            or value.allocation_size_bytes < value.end_of_file_bytes
            or value.number_of_links < 1
            or not directory and value.number_of_links != 1
        ):
            raise _StoreFault()
        return value

    def _query_directory(self, held: _HeldDirectory, *, after: bool) -> None:
        stage = "object_after" if after else "ancestor_before"
        ordinal = 1 + held.ordinal if after else held.ordinal
        resource = ("directory_handle", "ancestor", held.ordinal)
        try:
            raw = self._native.query_object(
                held.handle,
                context=self._context(
                    operation="query_object", stage=stage, binding=resource,
                    returned=None, ordinal=ordinal,
                ),
            )
            self._validate_raw(
                raw,
                guid_path=held.binding.canonical_volume_guid_path,
                dos_path=held.binding.canonical_dos_path,
                canonical_name=held.binding.canonical_name_exact,
                directory=True,
                expected_identity=held.binding.concrete_identity,
            )
            descriptor = self._measure_security(
                held.handle, directory_ordinal=held.ordinal, after=after
            )
            if descriptor != held.binding.security_descriptor_sha256:
                raise _StoreFault()
        except PermissionNativeCallError as exc:
            uncertain = self._release_error_resources(exc)
            raise _StoreFault(
                backend_error_hex=_backend_hex(exc),
                cleanup_uncertain=uncertain,
            ) from exc

    def _namespace_snapshot(self) -> ConfidentialNamespaceSnapshot:
        namespace = next(
            item for item in self.directory_bindings
            if item.relative_path == self.expected_store.namespace_relative_path
        )
        parent = next(
            item for item in self.directory_bindings
            if item.relative_path == self.expected_store.records_parent_relative_path
        )
        return ConfidentialNamespaceSnapshot(
            namespace_relative_path=self.expected_store.namespace_relative_path,
            namespace_object_identity=namespace.opaque_object_identity,
            volume_identifier_commitment=self.volume_binding.volume_identifier_commitment,
            confidential_store_profile_sha256=self.volume_binding.confidential_store_profile_sha256,
            records_parent=StoreParentSnapshot(
                relative_path=parent.relative_path,
                object_identity=parent.opaque_object_identity,
                namespace_object_identity=namespace.opaque_object_identity,
                volume_identifier_commitment=self.volume_binding.volume_identifier_commitment,
                security_descriptor_sha256=parent.security_descriptor_sha256,
                confidential_store_profile_sha256=self.volume_binding.confidential_store_profile_sha256,
            ),
        )

    def open_namespace(self, namespace_relative_path: str) -> NamespaceOpenOutcome:
        if namespace_relative_path != self.expected_store.namespace_relative_path:
            raise PrivatePermissionEvidenceContractError("namespace request differs from binding")
        held: list[_HeldDirectory] = []
        try:
            for ordinal, binding in enumerate(self.directory_bindings):
                resource = ("directory_handle", "ancestor", ordinal)
                handle = self._native.open_existing_directory(
                    binding.canonical_volume_guid_path,
                    specification=NativeOpenSpecification(
                        object_kind="directory", existing_only=True,
                        follow_reparse_points=False, read_data=False,
                        read_attributes=True, read_control=True, write_data=False,
                        append_data=False, write_attributes=False, delete=False,
                        write_dac=False, write_owner=False, create=False,
                        truncate=False, share_read=True, share_write=False,
                        share_delete=False,
                    ),
                    context=self._context(
                        operation="open_existing_directory", stage="ancestor_before",
                        binding=None, returned=resource, ordinal=ordinal,
                    ),
                )
                item = _HeldDirectory(binding, ordinal, handle)
                held.append(item)
                self._query_directory(item, after=False)
            snapshot = self._namespace_snapshot()
            lease = _Win32NamespaceLease(self, tuple(held), snapshot)
            self._open_count += 1
            return NamespaceOpenOutcome(
                status="opened", lease=lease,
                effect=StoreNamespaceEffect(
                    operation="open_namespace", completion_status="succeeded",
                    effect_certainty="reported_effect", before_snapshot=None,
                    after_snapshot=snapshot, backend_error_hex=None,
                ),
                failure=None,
            )
        except Exception as exc:
            uncertain = self._release_error_resources(exc) or (
                isinstance(exc, _StoreFault) and exc.cleanup_uncertain
            )
            for item in reversed(held):
                try:
                    if item.handle.release() is not None:
                        uncertain = True
                except Exception:
                    uncertain = True
            backend = _backend_hex(exc)
            certainty = "unknown" if uncertain else "no_effect"
            return NamespaceOpenOutcome(
                status="failed", lease=None,
                effect=StoreNamespaceEffect(
                    operation="open_namespace", completion_status="failed",
                    effect_certainty=certainty, before_snapshot=None,
                    after_snapshot=None, backend_error_hex=backend,
                ),
                failure=_failure(
                    "open_namespace", "namespace_open",
                    "store_protocol_failure" if uncertain else "namespace_open_failed",
                    backend_error_hex=backend,
                ),
            )


class _Win32NamespaceLease:
    def __init__(
        self,
        owner: Win32ConfidentialEvidenceStore,
        held: tuple[_HeldDirectory, ...],
        snapshot: ConfidentialNamespaceSnapshot,
    ) -> None:
        self._owner = owner
        self._held = held
        self._snapshot = snapshot
        self._release_attempted = False

    @property
    def snapshot(self) -> ConfidentialNamespaceSnapshot:
        return self._snapshot

    def _validate_held(self) -> None:
        if self._release_attempted:
            raise _StoreFault()
        for item in self._held:
            self._owner._query_directory(item, after=True)

    def create_exclusive(
        self,
        relative_path: str,
        *,
        confidential_store_profile_sha256: str,
        required_security_descriptor_sha256: str,
    ) -> StoreCreateOutcome:
        destination = self._owner.expected_destination
        valid_paths = {
            destination.primary_creating_relative_path,
            destination.retention_failure_creating_relative_path,
        }
        if relative_path not in valid_paths:
            raise PrivatePermissionEvidenceContractError("create path is outside destination")
        if (
            confidential_store_profile_sha256
            != self._owner.volume_binding.confidential_store_profile_sha256
            or required_security_descriptor_sha256
            != destination.required_created_object_security_descriptor_sha256
        ):
            raise PrivatePermissionEvidenceContractError("create bindings differ from destination")
        authz_destination = type(destination) is PermissionAuthzEvidenceDestination
        role: Literal[
            "primary_payload", "retention_failure_payload",
            "authz_primary_payload", "authz_retention_failure_payload",
        ] = (
            "authz_primary_payload" if authz_destination else "primary_payload"
        ) if relative_path == destination.primary_creating_relative_path else (
            "authz_retention_failure_payload"
            if authz_destination
            else "retention_failure_payload"
        )
        raw_handle: int | None = None
        store_object: _Win32StoreObject | None = None
        create_entered = False
        try:
            self._validate_held()
            prepared = self._owner._bindings.prepare_create_security(
                self._owner.creation_sddl, required_security_descriptor_sha256
            )
            canonical = _join(
                self._owner.volume_binding.canonical_volume_guid_root,
                relative_path,
            )
            create_entered = True
            raw_handle = self._owner._bindings.create_secured_exclusive(
                canonical, prepared
            )
            native_handle = Win32OwnedObjectHandle(
                cast(object, self._owner._bindings), raw_handle,
                ("file_handle", "subject", None), object_kind="file",
                requested_path=canonical,
            )
            store_object = _Win32StoreObject(
                self, native_handle, relative_path, role
            )
            self._validate_held()
            created = store_object._capture_snapshot(
                expected_size=0, expected_hash=hashlib.sha256(b"").hexdigest()
            )
            store_object._confirmed = created
            evidence = StoreCreationEvidence(
                creation_disposition="created_new_exclusive",
                prewrite_size_bytes=0,
                namespace_object_identity=destination.namespace_object_identity,
                parent_relative_path=destination.records_parent_relative_path,
                parent_object_identity=destination.records_parent_object_identity,
                object_identity=created.object_identity,
                volume_identifier_commitment=destination.volume_identifier_commitment,
                creation_security_descriptor_sha256=created.security_descriptor_sha256,
                confidential_store_profile_sha256=confidential_store_profile_sha256,
                security_applied_in_create=True,
            )
            return StoreCreateOutcome(
                status="created", object_handle=store_object,
                creation_evidence=evidence,
                effect=_store_effect(
                    "create_exclusive", "succeeded", "reported_effect",
                    after=created,
                ),
                failure=None,
            )
        except Win32EvidenceStoreError as exc:
            backend = _backend_hex(exc)
            if exc.collision:
                return StoreCreateOutcome(
                    status="collision", object_handle=None, creation_evidence=None,
                    effect=_store_effect(
                        "create_exclusive", "collision", "no_effect",
                        collision=relative_path,
                    ),
                    failure=_failure(
                        "create_exclusive", "exclusive_create", "collision",
                        kind="store_collision",
                    ),
                )
            if store_object is None and exc.handle_owned and isinstance(exc.returned_handle, int):
                native_handle = Win32OwnedObjectHandle(
                    cast(object, self._owner._bindings), exc.returned_handle,
                    ("file_handle", "subject", None), object_kind="file",
                    requested_path=_join(
                        self._owner.volume_binding.canonical_volume_guid_root,
                        relative_path,
                    ),
                )
                store_object = _Win32StoreObject(self, native_handle, relative_path, role)
            no_effect = not exc.invoked and store_object is None
            certainty = "no_effect" if no_effect else "unknown"
            if store_object is not None:
                store_object._operation_failed = True
            return StoreCreateOutcome(
                status="failed", object_handle=store_object,
                creation_evidence=None,
                effect=_store_effect(
                    "create_exclusive", "failed", certainty,
                    backend_error_hex=backend,
                ),
                failure=_failure(
                    "create_exclusive", "exclusive_create",
                    "create_failed" if no_effect else "effect_unknown",
                    role=role if store_object is not None else None,
                    backend_error_hex=backend,
                ),
            )
        except Exception as exc:
            if store_object is not None:
                store_object._operation_failed = True
            uncertain = create_entered or raw_handle is not None
            return StoreCreateOutcome(
                status="failed", object_handle=store_object,
                creation_evidence=None,
                effect=_store_effect(
                    "create_exclusive", "failed",
                    "unknown" if uncertain else "no_effect",
                    backend_error_hex=_backend_hex(exc),
                ),
                failure=_failure(
                    "create_exclusive", "exclusive_create",
                    (
                        "store_protocol_failure"
                        if isinstance(exc, _StoreFault)
                        else "effect_unknown" if uncertain
                        else "create_failed"
                    ),
                    role=role if store_object is not None else None,
                    backend_error_hex=_backend_hex(exc),
                ),
            )

    def release(self) -> StoreReleaseOutcome:
        if self._release_attempted:
            return StoreReleaseOutcome(
                resource_kind="namespace_handle", status="failed",
                object_effect=None,
                namespace_effect=StoreNamespaceEffect(
                    operation="release_namespace", completion_status="failed",
                    effect_certainty="unknown", before_snapshot=self._snapshot,
                    after_snapshot=None, backend_error_hex=None,
                ),
                failure=_failure(
                    "release_namespace", "resource_cleanup",
                    "namespace_release_failed", kind="cleanup_failure",
                ),
            )
        self._release_attempted = True
        failed: ComponentFailureItem | None = None
        for item in reversed(self._held):
            try:
                result = item.handle.release()
            except Exception:
                result = ComponentFailureItem(
                    "cleanup_failure", "resource_cleanup", "release_native_resource",
                    "directory_handle", "ancestor", item.ordinal, None, None,
                    "resource_release_failed",
                )
            if result is not None and failed is None:
                failed = result
        if failed is None:
            return StoreReleaseOutcome(
                resource_kind="namespace_handle", status="released",
                object_effect=None,
                namespace_effect=StoreNamespaceEffect(
                    operation="release_namespace", completion_status="succeeded",
                    effect_certainty="reported_effect", before_snapshot=self._snapshot,
                    after_snapshot=None, backend_error_hex=None,
                ),
                failure=None,
            )
        return StoreReleaseOutcome(
            resource_kind="namespace_handle", status="failed", object_effect=None,
            namespace_effect=StoreNamespaceEffect(
                operation="release_namespace", completion_status="failed",
                effect_certainty="unknown", before_snapshot=self._snapshot,
                after_snapshot=None, backend_error_hex=failed.backend_error_hex,
            ),
            failure=_failure(
                "release_namespace", "resource_cleanup",
                "namespace_release_failed", backend_error_hex=failed.backend_error_hex,
                kind="cleanup_failure",
            ),
        )


class _Win32StoreObject:
    def __init__(
        self,
        lease: _Win32NamespaceLease,
        handle: Win32OwnedObjectHandle,
        relative_path: str,
        role: Literal[
            "primary_payload", "retention_failure_payload",
            "authz_primary_payload", "authz_retention_failure_payload",
        ],
    ) -> None:
        self._lease = lease
        self._owner = lease._owner
        self._handle = handle
        self._relative_path = relative_path
        self._role = role
        self._identity: NativeFileIdentity | None = None
        self._confirmed: StoreObjectSnapshot | None = None
        self._cursor = 0
        self._write_attempted = False
        self._operation_failed = False
        self._release_attempted = False
        self._read_ordinal = 0

    def _paths(self) -> tuple[str, str, str]:
        volume = self._owner.volume_binding
        return (
            _join(volume.canonical_volume_guid_root, self._relative_path),
            _join(volume.canonical_dos_root, self._relative_path),
            _join(volume.canonical_name_root, self._relative_path),
        )

    def _query_raw(self, *, after: bool = False) -> RawNativeObjectSnapshot:
        guid, dos, name = self._paths()
        raw = self._owner._native.query_object(
            self._handle,
            context=self._owner._context(
                operation="query_object",
                stage="object_after" if after else "subject_before",
                binding=("file_handle", "subject", None), returned=None,
                ordinal=0 if after else None,
            ),
        )
        return self._owner._validate_raw(
            raw, guid_path=guid, dos_path=dos, canonical_name=name,
            directory=False, expected_identity=self._identity,
        )

    def _streams(self, raw: RawNativeObjectSnapshot) -> None:
        value = self._owner._native.query_streams(
            self._handle,
            context=self._owner._context(
                operation="query_streams", stage="stream_inventory",
                binding=("file_handle", "subject", None), returned=None,
                ordinal=None,
            ),
        )
        if (
            type(value) is not tuple
            or len(value) != 1
            or type(value[0]) is not NativeStream
            or value[0].stream_name != "::$DATA"
            or value[0].stream_index != 0
            or value[0].stream_size_bytes != raw.end_of_file_bytes
            or value[0].stream_allocation_size_bytes < value[0].stream_size_bytes
        ):
            raise _StoreFault()

    def _read_at(self, offset: int, requested: int) -> bytes:
        result = self._owner._native.read_unnamed_at(
            self._handle, offset=offset, requested_bytes=requested,
            context=self._owner._context(
                operation="read_unnamed_at", stage="content_read",
                binding=("file_handle", "subject", None), returned=None,
                ordinal=self._read_ordinal,
            ),
        )
        self._read_ordinal += 1
        if type(result) is not bytes or len(result) > requested:
            raise _StoreFault()
        return result

    def _capture_snapshot(
        self,
        *,
        expected_size: int | None = None,
        expected_hash: str | None = None,
    ) -> StoreObjectSnapshot:
        before = self._query_raw(after=self._identity is not None)
        descriptor_before = self._owner._measure_security(
            self._handle, directory_ordinal=None, after=self._identity is not None
        )
        if descriptor_before != self._owner.expected_destination.required_created_object_security_descriptor_sha256:
            raise _StoreFault()
        self._streams(before)
        digest = hashlib.sha256()
        remaining = before.end_of_file_bytes
        offset = 0
        while remaining:
            requested = min(remaining, DWORD_MAX)
            chunk = self._read_at(offset, requested)
            if len(chunk) != requested:
                raise _StoreFault()
            digest.update(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        if self._read_at(offset, 1) != b"":
            raise _StoreFault()
        after = self._query_raw(after=True)
        descriptor_after = self._owner._measure_security(
            self._handle, directory_ordinal=None, after=True
        )
        self._streams(after)
        if before != after or descriptor_before != descriptor_after:
            raise _StoreFault()
        file_hash = digest.hexdigest()
        if (
            expected_size is not None and before.end_of_file_bytes != expected_size
            or expected_hash is not None and file_hash != expected_hash
        ):
            raise _StoreFault()
        if self._identity is None:
            self._identity = before.identity
        identity = _object_identity(before.identity)
        destination = self._owner.expected_destination
        return StoreObjectSnapshot(
            relative_path=self._relative_path,
            namespace_object_identity=destination.namespace_object_identity,
            parent_relative_path=destination.records_parent_relative_path,
            parent_object_identity=destination.records_parent_object_identity,
            object_identity=identity,
            volume_identifier_commitment=destination.volume_identifier_commitment,
            size_bytes=before.end_of_file_bytes,
            file_sha256=file_hash,
            security_descriptor_sha256=descriptor_after,
            confidential_store_profile_sha256=destination.confidential_store_profile_sha256,
            publication_state=(
                "creating" if self._relative_path.endswith(".__creating__") else "published"
            ),
        )

    def _before(self) -> StoreObjectSnapshot:
        if self._release_attempted or self._operation_failed or self._confirmed is None:
            raise PrivatePermissionEvidenceContractError("store object is not active")
        return self._confirmed

    @staticmethod
    def _reported_prefix_snapshot(
        before: StoreObjectSnapshot,
        payload: bytes,
        count: int,
    ) -> StoreObjectSnapshot:
        return StoreObjectSnapshot(
            relative_path=before.relative_path,
            namespace_object_identity=before.namespace_object_identity,
            parent_relative_path=before.parent_relative_path,
            parent_object_identity=before.parent_object_identity,
            object_identity=before.object_identity,
            volume_identifier_commitment=before.volume_identifier_commitment,
            size_bytes=count,
            file_sha256=hashlib.sha256(payload[:count]).hexdigest(),
            security_descriptor_sha256=before.security_descriptor_sha256,
            confidential_store_profile_sha256=before.confidential_store_profile_sha256,
            publication_state=before.publication_state,
        )

    def _operation_failure(
        self,
        operation: str,
        stage: str,
        reason: str,
        *,
        before: StoreObjectSnapshot,
        certainty: Literal["no_effect", "reported_effect", "unknown"],
        after: StoreObjectSnapshot | None = None,
        requested: int | None = None,
        reported: int | None = None,
        backend: str | None = None,
    ) -> tuple[StoreFailureEffect, ComponentFailureItem]:
        effect = _store_effect(
            operation, "failed", certainty, before=before,
            after=(before if certainty == "no_effect" else after),
            requested=requested, reported=reported, backend_error_hex=backend,
        )
        failure = _failure(
            operation, stage, reason, role=self._role,
            backend_error_hex=backend,
        )
        return effect, failure

    def write_once(self, data: bytes) -> StoreIoOutcome:
        before = self._before()
        if type(data) is not bytes or self._write_attempted:
            self._operation_failed = True
            effect, failure = self._operation_failure(
                "write_once", "payload_write", "store_protocol_failure",
                before=before, certainty="unknown",
                requested=len(data) if type(data) is bytes else None,
            )
            return StoreIoOutcome("failed", effect, failure)
        self._write_attempted = True
        requested = len(data)
        try:
            self._validate_before_disclosure(before)
        except Exception as exc:
            self._operation_failed = True
            effect, failure = self._operation_failure(
                "write_once", "payload_write", "store_protocol_failure",
                before=before, certainty="unknown", requested=requested,
                backend=_backend_hex(exc),
            )
            return StoreIoOutcome("failed", effect, failure)
        reported_total = 0
        current_chunk = b""
        try:
            while reported_total < requested:
                current_chunk = data[
                    reported_total : reported_total + _NATIVE_TRANSFER_BYTES
                ]
                count = self._owner._bindings.write_once_at(
                    self._handle.raw_handle, reported_total, current_chunk
                )
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or not 0 <= count <= len(current_chunk)
                ):
                    raise _StoreFault()
                reported_total += count
                if count != len(current_chunk):
                    after = self._reported_prefix_snapshot(
                        before, data, reported_total
                    )
                    self._operation_failed = True
                    effect, failure = self._operation_failure(
                        "write_once", "payload_write", "short_write",
                        before=before, certainty="reported_effect", after=after,
                        requested=requested, reported=reported_total,
                    )
                    return StoreIoOutcome("failed", effect, failure)
            after = self._capture_snapshot(
                expected_size=requested,
                expected_hash=hashlib.sha256(data).hexdigest(),
            )
            self._confirmed = after
            return StoreIoOutcome(
                "succeeded",
                _store_effect(
                    "write_once", "succeeded", "reported_effect",
                    before=before, after=after, requested=requested,
                    reported=reported_total,
                ),
                None,
            )
        except Win32EvidenceStoreError as exc:
            backend = _backend_hex(exc)
            current_report = None
            for candidate in (
                exc.confirmed_transfer_count,
                exc.unconfirmed_transfer_count,
            ):
                if (
                    isinstance(candidate, int)
                    and not isinstance(candidate, bool)
                    and 0 <= candidate <= len(current_chunk)
                ):
                    current_report = candidate
                    break
            if exc.invoked:
                copied = (
                    reported_total + current_report
                    if current_report is not None
                    and reported_total + current_report <= requested
                    else reported_total if reported_total else None
                )
                certainty = "unknown"
                reason = "effect_unknown"
                after = None
            elif reported_total:
                copied = reported_total
                certainty = "reported_effect"
                reason = "write_failed"
                after = self._reported_prefix_snapshot(before, data, reported_total)
            else:
                copied = None
                certainty = "no_effect"
                reason = "write_failed"
                after = before
            self._operation_failed = True
            effect, failure = self._operation_failure(
                "write_once", "payload_write", reason,
                before=before, certainty=cast(
                    Literal["no_effect", "reported_effect", "unknown"], certainty
                ), after=after,
                requested=requested, reported=copied, backend=backend,
            )
            return StoreIoOutcome("failed", effect, failure)
        except Exception as exc:
            self._operation_failed = True
            effect, failure = self._operation_failure(
                "write_once", "payload_write",
                "store_protocol_failure" if isinstance(exc, _StoreFault) else "effect_unknown",
                before=before, certainty="unknown", requested=requested,
                reported=reported_total if reported_total else None,
                backend=_backend_hex(exc),
            )
            return StoreIoOutcome("failed", effect, failure)

    def flush_data_and_metadata(self) -> StoreIoOutcome:
        before = self._before()
        try:
            self._owner._bindings.flush_file(self._handle.raw_handle)
            return StoreIoOutcome(
                "succeeded",
                _store_effect(
                    "flush_data_and_metadata", "succeeded", "reported_effect",
                    before=before, after=before,
                ),
                None,
            )
        except Win32EvidenceStoreError as exc:
            backend = _backend_hex(exc)
            no_effect = not exc.invoked
            self._operation_failed = True
            effect, failure = self._operation_failure(
                "flush_data_and_metadata", "payload_flush",
                "flush_failed" if no_effect else "effect_unknown",
                before=before, certainty="no_effect" if no_effect else "unknown",
                backend=backend,
            )
            return StoreIoOutcome("failed", effect, failure)
        except Exception:
            self._operation_failed = True
            effect, failure = self._operation_failure(
                "flush_data_and_metadata", "payload_flush", "store_protocol_failure",
                before=before, certainty="unknown",
            )
            return StoreIoOutcome("failed", effect, failure)

    def rewind(self) -> StoreIoOutcome:
        before = self._before()
        self._cursor = 0
        return StoreIoOutcome(
            "succeeded",
            _store_effect(
                "rewind", "succeeded", "reported_effect",
                before=before, after=before,
            ),
            None,
        )

    def _validate_before_disclosure(self, before: StoreObjectSnapshot) -> None:
        self._lease._validate_held()
        raw = self._query_raw(after=True)
        descriptor = self._owner._measure_security(
            self._handle, directory_ordinal=None, after=True
        )
        self._streams(raw)
        if (
            raw.end_of_file_bytes != before.size_bytes
            or descriptor != before.security_descriptor_sha256
            or _object_identity(raw.identity) != before.object_identity
        ):
            raise _StoreFault()

    def read(self, max_bytes: int) -> StoreBytesReadOutcome:
        before = self._before()
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise PrivatePermissionEvidenceContractError("read bound is invalid")
        copied = 0
        try:
            self._validate_before_disclosure(before)
            remaining = max(0, before.size_bytes - self._cursor)
            expected = min(max_bytes, remaining)
            chunks: list[bytes] = []
            while copied < expected:
                requested = min(_NATIVE_TRANSFER_BYTES, expected - copied)
                chunk = self._read_at(self._cursor + copied, requested)
                chunks.append(chunk)
                copied += len(chunk)
                if len(chunk) != requested:
                    self._operation_failed = True
                    effect, failure = self._operation_failure(
                        "read", "postpublication_readback" if before.publication_state == "published" else "payload_readback",
                        "short_read", before=before, certainty="reported_effect",
                        after=before, requested=max_bytes, reported=copied,
                    )
                    return StoreBytesReadOutcome("failed", None, effect, failure)
            if expected == 0:
                chunk = self._read_at(
                    self._cursor, min(_NATIVE_TRANSFER_BYTES, max_bytes)
                )
                chunks.append(chunk)
                copied = len(chunk)
            data = b"".join(chunks)
            if len(data) > expected:
                self._operation_failed = True
                effect, failure = self._operation_failure(
                    "read", "postpublication_readback" if before.publication_state == "published" else "payload_readback",
                    "extra_byte", before=before, certainty="reported_effect",
                    after=before, requested=max_bytes, reported=len(data),
                )
                return StoreBytesReadOutcome("failed", None, effect, failure)
            self._cursor += len(data)
            return StoreBytesReadOutcome(
                "succeeded", data,
                _store_effect(
                    "read", "succeeded", "reported_effect",
                    before=before, after=before, requested=max_bytes,
                    reported=len(data),
                ),
                None,
            )
        except Exception as exc:
            backend = _backend_hex(exc)
            self._operation_failed = True
            reason = (
                "store_protocol_failure"
                if isinstance(exc, _StoreFault)
                else "effect_unknown"
            )
            effect, failure = self._operation_failure(
                "read", "postpublication_readback" if before.publication_state == "published" else "payload_readback",
                reason, before=before, certainty="unknown",
                requested=max_bytes, reported=copied if copied else None,
                backend=backend,
            )
            return StoreBytesReadOutcome("failed", None, effect, failure)

    def snapshot(self) -> StoreSnapshotOutcome:
        before = self._before()
        stage = (
            "postpublication_snapshot"
            if before.publication_state == "published"
            else "prepublication_snapshot"
        )
        try:
            self._lease._validate_held()
            observed = self._capture_snapshot(
                expected_size=before.size_bytes, expected_hash=before.file_sha256
            )
            if observed != before:
                raise _StoreFault()
            self._confirmed = observed
            return StoreSnapshotOutcome(
                "succeeded", observed,
                _store_effect(
                    "snapshot", "succeeded", "reported_effect",
                    before=observed, after=observed,
                ),
                None,
            )
        except Exception as exc:
            backend = _backend_hex(exc)
            self._operation_failed = True
            effect, failure = self._operation_failure(
                "snapshot", stage,
                "store_protocol_failure" if isinstance(exc, _StoreFault) else "effect_unknown",
                before=before, certainty="unknown", backend=backend,
            )
            return StoreSnapshotOutcome("failed", None, effect, failure)

    def publish_same_object_exclusive(
        self, final_relative_path: str
    ) -> StorePromoteOutcome:
        before = self._before()
        destination = self._owner.expected_destination
        expected_final = (
            destination.primary_accepted_relative_path
            if self._role in {"primary_payload", "authz_primary_payload"}
            else destination.retention_failure_accepted_relative_path
        )
        if final_relative_path != expected_final or before.publication_state != "creating":
            raise PrivatePermissionEvidenceContractError("publication target differs from binding")
        parent_guid = _join(
            self._owner.volume_binding.canonical_volume_guid_root,
            destination.records_parent_relative_path,
        )
        leaf = final_relative_path.rsplit("/", 1)[-1]
        try:
            self._lease._validate_held()
            fresh = self._capture_snapshot(
                expected_size=before.size_bytes, expected_hash=before.file_sha256
            )
            if fresh != before:
                raise _StoreFault()
            returned = self._owner._bindings.rename_no_replace(
                self._handle.raw_handle, parent_guid, leaf
            )
            expected_guid = _join(
                self._owner.volume_binding.canonical_volume_guid_root,
                final_relative_path,
            )
            if returned != expected_guid:
                raise _StoreFault()
            self._relative_path = final_relative_path
            first_after = self._capture_snapshot(
                expected_size=before.size_bytes, expected_hash=before.file_sha256
            )
            try:
                self._owner._bindings.flush_file(self._handle.raw_handle)
            except Win32EvidenceStoreError as exc:
                backend = _backend_hex(exc)
                self._confirmed = first_after
                self._operation_failed = True
                effect, failure = self._operation_failure(
                    "publish_same_object_exclusive", "publication", "publish_failed",
                    before=before, certainty="reported_effect", after=first_after,
                    backend=backend,
                )
                return StorePromoteOutcome("failed", effect, failure)
            self._lease._validate_held()
            second_after = self._capture_snapshot(
                expected_size=before.size_bytes, expected_hash=before.file_sha256
            )
            if second_after != first_after:
                raise _StoreFault()
            self._confirmed = second_after
            self._cursor = 0
            return StorePromoteOutcome(
                "published",
                _store_effect(
                    "publish_same_object_exclusive", "succeeded", "reported_effect",
                    before=before, after=second_after,
                ),
                None,
            )
        except Win32EvidenceStoreError as exc:
            backend = _backend_hex(exc)
            if exc.collision:
                self._operation_failed = True
                return StorePromoteOutcome(
                    "collision",
                    _store_effect(
                        "publish_same_object_exclusive", "collision", "no_effect",
                        before=before, after=before, collision=final_relative_path,
                    ),
                    _failure(
                        "publish_same_object_exclusive", "publication", "collision",
                        role=self._role, kind="store_collision",
                    ),
                )
            no_effect = not exc.invoked
            self._operation_failed = True
            effect, failure = self._operation_failure(
                "publish_same_object_exclusive", "publication",
                "publish_failed" if no_effect else "effect_unknown",
                before=before, certainty="no_effect" if no_effect else "unknown",
                backend=backend,
            )
            return StorePromoteOutcome("failed", effect, failure)
        except Exception as exc:
            self._operation_failed = True
            effect, failure = self._operation_failure(
                "publish_same_object_exclusive", "publication",
                "store_protocol_failure" if isinstance(exc, _StoreFault) else "effect_unknown",
                before=before, certainty="unknown", backend=_backend_hex(exc),
            )
            return StorePromoteOutcome("failed", effect, failure)

    def release(self) -> StoreReleaseOutcome:
        if self._release_attempted:
            return StoreReleaseOutcome(
                "store_object_handle", "failed",
                _store_effect("release_store_object", "failed", "unknown"),
                None,
                _failure(
                    "release_store_object", "resource_cleanup",
                    "resource_release_failed", role=self._role,
                    kind="cleanup_failure",
                ),
            )
        self._release_attempted = True
        try:
            failure = self._handle.release()
        except Exception:
            failure = ComponentFailureItem(
                "cleanup_failure", "resource_cleanup", "release_native_resource",
                "file_handle", "subject", None, None, None,
                "resource_release_failed",
            )
        if failure is None:
            return StoreReleaseOutcome(
                "store_object_handle", "released",
                _store_effect(
                    "release_store_object", "succeeded", "reported_effect"
                ),
                None, None,
            )
        return StoreReleaseOutcome(
            "store_object_handle", "failed",
            _store_effect(
                "release_store_object", "failed", "unknown",
                backend_error_hex=failure.backend_error_hex,
            ),
            None,
            _failure(
                "release_store_object", "resource_cleanup",
                "resource_release_failed", role=self._role,
                backend_error_hex=failure.backend_error_hex,
                kind="cleanup_failure",
            ),
        )


__all__ = [
    "Win32ConfidentialEvidenceStore",
    "Win32StoreDirectoryBinding",
    "Win32StoreVolumeBinding",
]
