"""Inert Windows handoff construction and process-local delegated ownership.

Configuration equality and object identity are not authenticated enrollment.
There is no issuer, native qualification, CLI or production registration here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
import threading
from typing import Any

from .canonical import require_sha256, sha256_bytes
from . import permission_handoff_records as records
from . import permission_handoff_sources as sources
from . import permission_native_win32 as native
from . import permission_observation_windows as observation
from . import permission_store_native_win32 as store


@dataclass(frozen=True, slots=True)
class HandoffWindowsDirectoryBinding:
    relative_path: str
    canonical_volume_guid_path: str
    canonical_dos_path: str
    canonical_name_exact: str
    identity: observation.NativeFileIdentity | None
    security_descriptor_sha256: str


@dataclass(frozen=True, slots=True)
class HandoffWindowsRootBinding:
    root_binding_id: str
    canonical_volume_guid_root: str
    canonical_dos_root: str
    canonical_name_root: str
    volume_serial_number_hex: str
    volume_identifier_commitment: str
    directories: tuple[HandoffWindowsDirectoryBinding, ...]

    def __post_init__(self):
        object.__setattr__(self, "directories", tuple(self.directories))


@dataclass(frozen=True, slots=True)
class HandoffWindowsInputPolicy:
    slot_id: str
    root_binding_id: str
    namespace_relative_path: str
    parent_relative_path: str
    expected_file_identity: observation.NativeFileIdentity | None
    security_descriptor_sha256: str
    confidential_profile_file_sha256: str | None


@dataclass(frozen=True, slots=True)
class HandoffWindowsConfiguration:
    plan_self_sha256: str
    enrollment_file_ref: records.FileRef
    roots: tuple[HandoffWindowsRootBinding, ...]
    input_policies: tuple[HandoffWindowsInputPolicy, ...]
    witness_creation_sddl: str
    transfer_chunk_bytes: int

    def __post_init__(self):
        object.__setattr__(self, "roots", tuple(self.roots))
        object.__setattr__(self, "input_policies", tuple(self.input_policies))


@dataclass(frozen=True, slots=True)
class WindowsHandoffProviders:
    authority: BoundHandoffAuthority
    reader: Any
    witness_sink: Any


class HandoffWindowsRefusal(RuntimeError):
    """A local guard rejection: no delegate call occurred for this attempt."""
    delegate_called = False


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _relative(value, *, root=False):
    _require(type(value) is str, "path must be an exact string")
    if root and value == "":
        return value
    records._relative_path(value, name="Windows handoff path")
    for part in value.split("/"):
        store._require_leaf(part)
        _require(not any(c in part for c in '<>"|?*'), "Windows wildcard/alias component")
    return value


def _join(root, relative):
    return root if not relative else root.rstrip("\\") + "\\" + relative.replace("/", "\\")


def _parents(relative):
    parts = relative.split("/") if relative else []
    return ("", *("/".join(parts[:i]) for i in range(1, len(parts) + 1)))


def _directory(root, path):
    return next(item for item in root.directories if item.relative_path == path)


def _root(configuration, root_id):
    return next(item for item in configuration.roots if item.root_binding_id == root_id)


def _validate_root_binding(root, plan):
    _require(type(root) is HandoffWindowsRootBinding, "wrong root binding type")
    records._text(root.root_binding_id, name="root id")
    store._path_parts(root.canonical_volume_guid_root, allow_root=True)
    _require(type(root.canonical_dos_root) is str and re.fullmatch(r"[A-Z]:\\(?:[^\\]+(?:\\[^\\]+)*)?", root.canonical_dos_root) is not None,
             "DOS root must have an exact drive spelling")
    _require(type(root.canonical_name_root) is str and root.canonical_name_root.startswith("\\") and not root.canonical_name_root.startswith("\\\\"), "invalid volume name root")
    for path in (root.canonical_dos_root[3:], root.canonical_name_root[1:]):
        if path:
            _relative(path.replace("\\", "/"))
    _require(re.fullmatch(r"0x(?:0|[1-9a-f][0-9a-f]{0,15})", root.volume_serial_number_hex or "") is not None, "invalid volume serial")
    require_sha256(root.volume_identifier_commitment, name="root volume commitment")
    paths = [x.relative_path for x in root.directories]
    _require(paths and paths[0] == "" and len(set(p.casefold() for p in paths)) == len(paths), "root ancestry alias/duplicate")
    _require(paths == sorted(paths, key=lambda p: (len(p.split('/')) if p else 0, p)), "ancestry order")
    for item in root.directories:
        _require(type(item) is HandoffWindowsDirectoryBinding, "wrong directory binding type")
        _relative(item.relative_path, root=True)
        _require(all(p in paths for p in _parents(item.relative_path)), "missing ancestor")
        for name, base in (("canonical_volume_guid_path", root.canonical_volume_guid_root), ("canonical_dos_path", root.canonical_dos_root), ("canonical_name_exact", root.canonical_name_root)):
            _require(getattr(item, name) == _join(base, item.relative_path), "directory canonical binding mismatch")
        require_sha256(item.security_descriptor_sha256, name="directory descriptor")
        if item.identity is None:
            _require(root.root_binding_id == plan.journal_binding.root_binding_id and item.relative_path == plan.journal_binding.attempt_relative_path and plan.journal_binding.expected_attempt_directory_identity is None, "null identity outside unestablished journal parent")
        else:
            records._validate_native_identity(item.identity, name="directory identity")
            _require(item.identity.volume_serial_number_hex == root.volume_serial_number_hex, "directory volume mismatch")


def _validate_slot_policy(policy, slot, plan, configuration):
    _require(type(policy) is HandoffWindowsInputPolicy, "wrong input policy type")
    _require((policy.slot_id, policy.root_binding_id) == (slot.slot_id, slot.root_binding_id), "policy slot/root mismatch")
    _relative(slot.relative_path)
    _relative(policy.namespace_relative_path, root=True)
    _require(policy.parent_relative_path == slot.relative_path.rpartition('/')[0], "policy parent is not immediate")
    root = _root(configuration, slot.root_binding_id)
    chain = _parents(policy.parent_relative_path)
    _require(policy.namespace_relative_path in chain and all(p in {d.relative_path for d in root.directories} for p in chain), "missing namespace or ancestry")
    require_sha256(policy.security_descriptor_sha256, name="input descriptor")
    if policy.confidential_profile_file_sha256 is not None:
        require_sha256(policy.confidential_profile_file_sha256, name="input profile")
    if policy.expected_file_identity is not None:
        records._validate_native_identity(policy.expected_file_identity, name="expected file identity")
        _require(policy.expected_file_identity.volume_serial_number_hex == root.volume_serial_number_hex, "file identity volume")
    namespace = _directory(root, policy.namespace_relative_path)
    parent = _directory(root, policy.parent_relative_path)
    if slot.role.startswith("journal_"):
        _require(root.root_binding_id == plan.journal_binding.root_binding_id and policy.namespace_relative_path == "" and namespace.identity == plan.journal_binding.expected_root_identity and parent.identity == plan.journal_binding.expected_attempt_directory_identity and policy.confidential_profile_file_sha256 is None, "journal policy mismatch")
    elif slot.role != "authority_input":
        expected = plan.expected_store
        _require(policy.namespace_relative_path == expected.namespace_relative_path and policy.parent_relative_path == expected.records_parent_relative_path and namespace.identity is not None and parent.identity is not None, "private ancestry mismatch")
        _require(sources._native_identity_text(namespace.identity) == expected.namespace_object_identity and sources._native_identity_text(parent.identity) == expected.records_parent_object_identity and root.volume_identifier_commitment == expected.volume_identifier_commitment and policy.confidential_profile_file_sha256 == expected.confidential_store_profile_sha256 and policy.security_descriptor_sha256 == expected.required_created_object_security_descriptor_sha256, "private policy mismatch")
        _require(parent.security_descriptor_sha256 == expected.records_parent_security_descriptor_sha256, "private parent descriptor mismatch")
    else:
        _require(namespace.identity is not None and parent.identity is not None and policy.confidential_profile_file_sha256 is not None, "authority input needs independent physical policy")


def validate_handoff_windows_configuration(*, expected_plan, configuration):
    records.validate_handoff_plan(expected_plan)
    _require(type(configuration) is HandoffWindowsConfiguration, "wrong configuration type")
    _require(type(configuration.enrollment_file_ref) is records.FileRef, "wrong enrollment reference type")
    records._validate_file_ref(configuration.enrollment_file_ref, name="configuration enrollment")
    _require(configuration.plan_self_sha256 == expected_plan.plan_self_sha256 and configuration.enrollment_file_ref == expected_plan.authority_binding.enrollment_file_ref, "configuration plan/enrollment mismatch")
    _require(type(configuration.transfer_chunk_bytes) is int and 0 < configuration.transfer_chunk_bytes <= native.DWORD_MAX, "unsupported transfer chunk")
    records._text(configuration.witness_creation_sddl, name="explicit witness SDDL")
    roots = configuration.roots
    needed = {s.root_binding_id for s in expected_plan.input_slots} | {expected_plan.witness_store.root_binding_id}
    _require(len(roots) == len(needed) and {r.root_binding_id for r in roots} == needed, "ambiguous/missing roots")
    physical = {}
    for root in roots:
        _validate_root_binding(root, expected_plan)
        for other in roots:
            if other is root:
                continue
            for field in ("canonical_volume_guid_root", "canonical_dos_root"):
                a, b = getattr(root, field).rstrip('\\').casefold(), getattr(other, field).rstrip('\\').casefold()
                _require(a != b and not a.startswith(b + '\\') and not b.startswith(a + '\\'), "ambiguous overlapping roots")
        for directory in root.directories:
            key = directory.canonical_volume_guid_path.casefold()
            _require(key not in physical or physical[key] == directory, "overlapping directory disagreement")
            physical[key] = directory
    _require(len(configuration.input_policies) == len(expected_plan.input_slots), "missing input policies")
    addresses = set()
    for policy, slot in zip(configuration.input_policies, expected_plan.input_slots, strict=True):
        _validate_slot_policy(policy, slot, expected_plan, configuration)
        key = (slot.root_binding_id, slot.relative_path.casefold())
        _require(key not in addresses, "case-insensitive input address collision")
        addresses.add(key)
    store_expectation = expected_plan.witness_store
    root = _root(configuration, store_expectation.root_binding_id)
    for p in _parents(store_expectation.parent_relative_path):
        _directory(root, p)
    namespace = _directory(root, store_expectation.namespace_relative_path)
    parent = _directory(root, store_expectation.parent_relative_path)
    _require(namespace.identity is not None and parent.identity is not None and sources._native_identity_text(namespace.identity) == store_expectation.namespace_object_identity and sources._native_identity_text(parent.identity) == store_expectation.parent_object_identity and parent.security_descriptor_sha256 == store_expectation.parent_security_descriptor_sha256 and root.volume_identifier_commitment == store_expectation.volume_identifier_commitment, "witness physical binding mismatch")
    for name in records.WITNESS_FILENAMES:
        for suffix in ("", ".creating"):
            path = store_expectation.parent_relative_path + '/' + name + suffix
            _relative(path)
            _require((root.root_binding_id, path.casefold()) not in addresses, "input/witness overlap")


@dataclass
class _Registration:
    lease: Any
    scope: Any
    key: tuple[str, str]
    valid: bool
    next_check: int = 0
    blocked: bool = False
    dispatched: bool = False
    finished: bool = False
    checking: bool = False


class BoundHandoffAuthority:
    """Identity-based delegation guard; not an authenticated or durable issuer."""
    def __init__(self, plan, delegate):
        _require(all(callable(getattr(delegate, name, None)) for name in ("acquire", "check_active", "finish")), "explicit authority delegate required")
        self._plan, self._delegate = plan, delegate
        self._lock = threading.RLock()
        self._reserved = set()
        self._leases = {}

    def acquire(self, plan, operation, slot_id):
        if plan != self._plan or operation not in {"read_handoff_input", "retain_handoff_witness"}:
            raise HandoffWindowsRefusal("plan/operation mismatch; delegate_called=False")
        if operation == "read_handoff_input":
            slot = next((s for s in plan.input_slots if s.slot_id == slot_id), None)
            if slot is None:
                raise HandoffWindowsRefusal("unknown read slot; delegate_called=False")
            root, role = slot.root_binding_id, slot.role
        else:
            if slot_id not in records.WITNESS_FILENAMES:
                raise HandoffWindowsRefusal("unknown witness slot; delegate_called=False")
            root, role = plan.witness_store.root_binding_id, "authority_input"
        key = (operation, slot_id)
        with self._lock:
            if key in self._reserved:
                raise HandoffWindowsRefusal("operation already reserved; delegate_called=False")
            self._reserved.add(key)
        result = self._delegate.acquire(plan, operation, slot_id)
        if type(result) is records.HandoffAcquireResult and result.lease is not None:
            lease = result.lease
            try:
                scope = replace(lease.scope) if type(lease.scope) is records.HandoffLeaseScope else None
            except Exception:
                scope = None
            valid = sources._acquire_return_valid(result)
            try:
                sources._validate_lease_scope(lease, plan=plan, operation=operation, slot_id=slot_id, root_binding_id=root, role=role)
            except Exception:
                valid = False
            with self._lock:
                if id(lease) in self._leases:
                    self._leases[id(lease)].blocked = True
                    raise RuntimeError("delegate reused a lease object after actual acquisition; effect unknown")
                self._leases[id(lease)] = _Registration(lease, scope, key, valid)
        return result

    def _registered(self, lease):
        value = self._leases.get(id(lease))
        if value is None or value.lease is not lease or value.finished:
            raise HandoffWindowsRefusal("unregistered/finished lease; delegate_called=False")
        return value

    @staticmethod
    def _names(value):
        return (("initial", "before_open", "before_bytes", "final") if value.key[0] == "read_handoff_input" else
                ("initial", "before_open", "before_bytes", "before_publish", "final"))

    def check_active(self, lease, checkpoint):
        with self._lock:
            value = self._registered(lease)
            names = self._names(value)
            if not value.valid or value.blocked or value.checking or lease.scope != value.scope or value.next_check >= len(names) or checkpoint != names[value.next_check] or (checkpoint == "final" and not value.dispatched):
                value.blocked = True
                raise HandoffWindowsRefusal("inactive/changed/out-of-order lease; delegate_called=False")
            value.checking = True
        try:
            result = self._delegate.check_active(lease, checkpoint)
        except BaseException:
            with self._lock:
                value.blocked, value.checking = True, False
            raise
        try:
            checked = sources._check_result(result, name=checkpoint)
        except Exception:
            with self._lock:
                value.blocked, value.checking = True, False
            return result
        with self._lock:
            value.next_check += 1
            value.blocked = value.blocked or checked.status != "active"
            value.checking = False
        return result

    def _claim_dispatch_once(self, lease, plan, operation, slot):
        with self._lock:
            value = self._registered(lease)
            if plan != self._plan or value.key != (operation, slot) or not value.valid or value.blocked or value.checking or lease.scope != value.scope or value.dispatched or value.next_check != len(self._names(value)) - 1:
                raise HandoffWindowsRefusal("dispatch not authorized by this registration; delegate_called=False")
            value.dispatched = True

    def finish(self, lease, requested_outcome):
        with self._lock:
            value = self._registered(lease)
            if value.scope is None or requested_outcome not in {"succeeded", "refused", "failed"}:
                raise HandoffWindowsRefusal("terminal ownership/outcome unavailable; delegate_called=False")
            value.finished = True
        return self._delegate.finish(lease, requested_outcome)


def build_windows_handoff_providers(*, expected_plan, configuration, authority_delegate, bindings=None):
    validate_handoff_windows_configuration(expected_plan=expected_plan, configuration=configuration)
    authority = BoundHandoffAuthority(expected_plan, authority_delegate)
    boundary = store.Win32EvidenceStoreBindings() if bindings is None else bindings
    api = native.Win32PermissionNativeApi(boundary)
    from .permission_handoff_reader_windows import Win32PermissionHandoffReader
    from .permission_handoff_witness_windows import Win32PermissionHandoffWitnessSink
    return WindowsHandoffProviders(authority,
        Win32PermissionHandoffReader(expected_plan, configuration, authority, boundary, api),
        Win32PermissionHandoffWitnessSink(expected_plan, configuration, authority, boundary, api))


class _NativeFault(RuntimeError):
    def __init__(self, reason="mismatch", *, stage="identity"):
        super().__init__(reason)
        self.reason, self.stage = reason, stage


def _native_failure(error, *, operation, stage):
    code = None
    reason = "raised"
    if isinstance(error, _NativeFault):
        reason, stage = error.reason, error.stage
    elif isinstance(error, observation.PermissionNativeCallError):
        code = error.failure.backend_error_hex
        reason = "unavailable" if code is not None else "malformed_output"
    elif isinstance(error, store.Win32EvidenceStoreError):
        value = error.backend_error_code
        if type(value) is int and 0 < value <= native.DWORD_MAX:
            code = f"0x{value:x}"
        reason = "collision" if error.collision else "unavailable" if code else "malformed_output"
    elif isinstance(error, (ValueError, TypeError)):
        reason = "malformed_output"
    if code == "0x5":
        reason = "refused"
    return records.HandoffFailure(stage, operation, reason, code, None)


class _Operation:
    """Local evidence retains original native carriers, including transfer counts.

    Nothing in this carrier is serialized as authenticated authority evidence.
    """
    def __init__(self, plan, bindings, api, operation):
        self.plan, self.bindings, self.api, self.operation = plan, bindings, api, operation
        self.resources = []
        self.released = set()
        self.cleanup = []
        self.native_evidence = []
        self.stage = "open"
        self.read_ordinal = 0

    def own(self, resource):
        if type(resource) not in {native.Win32OwnedObjectHandle, native.Win32OwnedSecurityBuffer} or resource._bindings is not self.bindings:
            raise _NativeFault("malformed_output", stage="cleanup")
        for item in self.resources:
            field = "raw_handle" if type(resource) is native.Win32OwnedObjectHandle else "pointer"
            if item is not resource and type(item) is type(resource) and id(item) not in self.released and getattr(item, field) == getattr(resource, field):
                raise _NativeFault("malformed_output", stage="cleanup")
        if all(item is not resource for item in self.resources):
            self.resources.append(resource)
        return resource

    def capture_error(self, error):
        self.native_evidence.append(error)
        if isinstance(error, observation.PermissionNativeCallError):
            for item in error.returned_resources:
                try:
                    resource, kind, role, ordinal = item
                    if resource.binding != (kind, role, ordinal):
                        raise ValueError("returned resource provenance mismatch")
                    self.own(resource)
                except Exception as malformed:
                    self.native_evidence.append(malformed)
                    self.cleanup.append(records.HandoffFailure("cleanup", self.operation, "malformed_output", None, None))
        if isinstance(error, store.Win32EvidenceStoreError):
            for failure in error.cleanup_failures:
                value = failure.backend_error_code
                code = f"0x{value:x}" if type(value) is int and 0 < value <= native.DWORD_MAX else None
                self.cleanup.append(records.HandoffFailure("cleanup", self.operation, "cleanup_failed", code, None))

    def close(self):
        for resource in reversed(self.resources):
            _release_owned_once(self, resource)


def _release_owned_once(operation, resource):
    if id(resource) in operation.released:
        return
    operation.released.add(id(resource))
    try:
        value = resource.release()
    except Exception as error:
        operation.native_evidence.append(error)
        operation.cleanup.append(records.HandoffFailure("cleanup", operation.operation, "cleanup_failed", None, None))
    else:
        if value is not None:
            operation.native_evidence.append(value)
            code = value.backend_error_hex if type(value) is observation.ComponentFailureItem else None
            operation.cleanup.append(records.HandoffFailure("cleanup", operation.operation, "cleanup_failed", code, None))


def _native_context(operation, name, *, handle=None, index=None, after=False, ordinal=None):
    binding = None if handle is None else handle.binding
    returned = None
    if name == "open_existing_directory":
        stage, ordinal, returned = "ancestor_before", index, ("directory_handle", "ancestor", index)
    elif name == "open_existing_subject":
        stage, ordinal, returned = "subject_before", None, ("file_handle", "subject", None)
    elif name == "query_object":
        stage = "object_after" if after else "subject_before" if index is None else "ancestor_before"
        ordinal = (0 if index is None else index + 1) if after else index
    elif name in {"acquire_security_descriptor", "parse_security_descriptor"}:
        stage, ordinal = ("security_after" if after else "security_before"), (0 if index is None else index + 1)
        security = ("security_descriptor_buffer", "subject_security" if index is None else "ancestor_security", index)
        if name == "acquire_security_descriptor":
            returned = security
        else:
            binding = security
    elif name == "query_streams":
        stage, ordinal = "stream_inventory", None
    elif name == "read_unnamed_at":
        stage = "content_read"
    else:
        raise ValueError("unsupported native operation")
    return observation.NativeCallContext(operation.plan.expected_request.request_self_sha256, stage, name, binding, returned, ordinal, 0)


def _open_specification(directory):
    return observation.NativeOpenSpecification("directory" if directory else "file", True, False, not directory, True, True,
                                              False, False, False, False, False, False, False, False, True, False, False)


def _copy_security(operation, handle, *, index=None, after=False):
    operation.stage = "security"
    buffer = None
    cleanup_count = len(operation.cleanup)
    primary = None
    try:
        buffer = operation.own(operation.api.acquire_security_descriptor(handle, context=_native_context(operation, "acquire_security_descriptor", handle=handle, index=index, after=after)))
        data = buffer.copy_bytes()
        descriptor = operation.api.parse_security_descriptor(data, context=_native_context(operation, "parse_security_descriptor", handle=handle, index=index, after=after))
        if descriptor.self_relative_descriptor_sha256 != sha256_bytes(data) or not descriptor.dacl_present or not descriptor.dacl_protected or descriptor.dacl_defaulted:
            raise _NativeFault(stage="security")
        return descriptor.self_relative_descriptor_sha256
    except BaseException as error:
        primary = error
        raise
    finally:
        if buffer is not None:
            _release_owned_once(operation, buffer)
        if len(operation.cleanup) != cleanup_count and primary is None:
            raise _NativeFault("cleanup_failed", stage="cleanup")


def _measure_held_object(operation, handle, *, index=None, after=False):
    operation.stage = "identity"
    raw = operation.api.query_object(handle, context=_native_context(operation, "query_object", handle=handle, index=index, after=after))
    security = _copy_security(operation, handle, index=index, after=after)
    operation.native_evidence.append((raw, security))
    return raw, security


def _validate_measurement(measurement, root, relative, *, directory, identity=None, descriptor=None):
    raw, security = measurement
    if type(raw) is not observation.RawNativeObjectSnapshot:
        raise _NativeFault("malformed_output")
    records._validate_native_identity(raw.identity, name="measured Windows identity")
    if (raw.canonical_volume_guid_path != _join(root.canonical_volume_guid_root, relative)
            or raw.canonical_dos_path != _join(root.canonical_dos_root, relative)
            or raw.canonical_name_exact != _join(root.canonical_name_root, relative)
            or not raw.unicode_nfc_exact or raw.directory is not directory
            or raw.identity.volume_serial_number_hex != root.volume_serial_number_hex
            or identity is not None and raw.identity != identity
            or descriptor is not None and security != descriptor
            or raw.delete_pending or int(raw.reparse_tag_hex, 16)
            or bool(int(raw.file_attributes_hex, 16) & 0x10) is not directory
            or int(raw.file_attributes_hex, 16) & (0x400 | 0x1000 | 0x40000 | 0x400000)
            or raw.number_of_links < 1 or not directory and raw.number_of_links != 1):
        raise _NativeFault()


@dataclass
class _HeldDirectory:
    binding: HandoffWindowsDirectoryBinding
    handle: native.Win32OwnedObjectHandle
    index: int
    before: tuple
    current: tuple


def _open_ancestry(operation, root, parent):
    held = []
    for index, path in enumerate(_parents(parent)):
        expected = _directory(root, path)
        operation.stage = "open"
        handle = operation.own(operation.api.open_existing_directory(expected.canonical_volume_guid_path,
            specification=_open_specification(True), context=_native_context(operation, "open_existing_directory", index=index)))
        measured = _measure_held_object(operation, handle, index=index)
        _validate_measurement(measured, root, path, directory=True, identity=expected.identity, descriptor=expected.security_descriptor_sha256)
        held.append(_HeldDirectory(expected, handle, index, measured, measured))
    return held


def _remeasure_ancestry(operation, root, held):
    for directory in held:
        measured = _measure_held_object(operation, directory.handle, index=directory.index, after=True)
        directory.current = measured
        _validate_measurement(measured, root, directory.binding.relative_path, directory=True, identity=directory.before[0].identity, descriptor=directory.binding.security_descriptor_sha256)
        if measured != directory.before:
            raise _NativeFault()


def _streams(operation, handle, raw):
    operation.stage = "identity"
    streams = operation.api.query_streams(handle, context=_native_context(operation, "query_streams", handle=handle))
    if (type(streams) is not tuple or len(streams) != 1 or type(streams[0]) is not observation.NativeStream
            or streams[0].stream_name != "::$DATA" or streams[0].stream_index != 0
            or streams[0].stream_size_bytes != raw.end_of_file_bytes):
        raise _NativeFault()


def _metadata(root, relative, measurement, held, namespace, profile):
    raw, descriptor = measurement
    parent = held[-1].current[0].identity
    namespace_identity = next(x.current[0].identity for x in held if x.binding.relative_path == namespace)
    return records.HandoffObjectMetadata(root.root_binding_id, relative, raw.canonical_volume_guid_path, raw.identity,
        parent, namespace_identity, raw.end_of_file_bytes, descriptor, profile, root.volume_identifier_commitment)


def _read_exact_opaque_bytes(operation, handle, size, chunk_size, copied):
    if type(size) is not int or not 0 <= size < native.SIGNED_LARGE_INTEGER_MAX:
        raise _NativeFault("unavailable", stage="read")
    operation.stage = "read"
    offset = 0
    while offset < size:
        wanted = min(chunk_size, size - offset)
        ordinal = operation.read_ordinal
        operation.read_ordinal += 1
        data = operation.api.read_unnamed_at(handle, offset=offset, requested_bytes=wanted,
            context=_native_context(operation, "read_unnamed_at", handle=handle, ordinal=ordinal))
        copied.extend(data)
        if len(data) != wanted:
            raise _NativeFault("short_io", stage="read")
        offset += wanted
    ordinal = operation.read_ordinal
    operation.read_ordinal += 1
    eof = operation.api.read_unnamed_at(handle, offset=size, requested_bytes=1,
        context=_native_context(operation, "read_unnamed_at", handle=handle, ordinal=ordinal))
    if eof:
        copied.extend(eof)
        raise _NativeFault("mismatch", stage="read")
    return bytes(copied)
