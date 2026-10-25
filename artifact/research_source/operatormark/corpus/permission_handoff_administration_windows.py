"""Explicit held-object control-plane and leased restart reads; no construction I/O."""
from __future__ import annotations

from threading import RLock

from . import permission_handoff_administration_records as records
from . import permission_handoff_administration as administration
from . import permission_handoff_records as old
from . import permission_handoff_windows as windows
from .permission_handoff_capture import _exception_summary


def _failure(error, *, operation, stage):
    code, reason = None, "raised"
    if isinstance(error, windows._NativeFault):
        reason, stage = error.reason, error.stage
    elif isinstance(error, windows.observation.PermissionNativeCallError):
        code = error.failure.backend_error_hex
        reason = "unavailable" if code is not None else "malformed_output"
    elif isinstance(error, windows.store.Win32EvidenceStoreError):
        value = error.backend_error_code
        code = f"0x{value:x}" if type(value) is int and 0 < value <= windows.native.DWORD_MAX else None
        reason = "unavailable" if code is not None else "malformed_output"
    elif isinstance(error, (ValueError, TypeError)):
        reason = "malformed_output"
    if code == "0x5": reason = "refused"
    return records.AdministrativeFailure(stage, operation, reason, code, _exception_summary(error))


class _AdministrativeReadOperation:
    """Exact native resources, separately tagged administrative failure evidence."""
    def __init__(self, plan, bindings, api, operation):
        self.plan, self.bindings, self.api, self.operation = plan, bindings, api, operation
        self.resources, self.cleanup, self.native_evidence = [], [], []
        self.released = set()
        self.stage, self.read_ordinal = "open", 0

    def own(self, resource):
        if type(resource) not in {windows.native.Win32OwnedObjectHandle, windows.native.Win32OwnedSecurityBuffer} or resource._bindings is not self.bindings:
            raise windows._NativeFault("malformed_output", stage="cleanup")
        for item in self.resources:
            field = "raw_handle" if type(resource) is windows.native.Win32OwnedObjectHandle else "pointer"
            if item is not resource and type(item) is type(resource) and id(item) not in self.released and getattr(item, field) == getattr(resource, field):
                raise windows._NativeFault("malformed_output", stage="cleanup")
        if all(item is not resource for item in self.resources): self.resources.append(resource)
        return resource

    def capture_error(self, error):
        self.native_evidence.append(error)
        if isinstance(error, windows.observation.PermissionNativeCallError):
            for item in error.returned_resources:
                try:
                    resource, kind, role, ordinal = item
                    records._require(resource.binding == (kind, role, ordinal), "returned resource provenance mismatch")
                    self.own(resource)
                except BaseException as malformed:
                    self.native_evidence.append(malformed)
                    self.cleanup.append(_failure(malformed, operation=self.operation, stage="cleanup"))
        if isinstance(error, windows.store.Win32EvidenceStoreError):
            for item in error.cleanup_failures:
                code = item.backend_error_code
                self.cleanup.append(records.AdministrativeFailure("cleanup", self.operation, "cleanup_failed",
                    f"0x{code:x}" if type(code) is int and 0 < code <= windows.native.DWORD_MAX else None, None))

    def release(self, resource):
        if id(resource) in self.released: return
        self.released.add(id(resource))
        try:
            result = resource.release()
            if result is not None:
                self.native_evidence.append(result)
                code = result.backend_error_hex if type(result) is windows.observation.ComponentFailureItem else None
                self.cleanup.append(records.AdministrativeFailure("cleanup", self.operation, "cleanup_failed", code, None))
        except BaseException as error:
            self.native_evidence.append(error)
            self.cleanup.append(records.AdministrativeFailure("cleanup", self.operation, "cleanup_failed", None, _exception_summary(error)))

    def close(self):
        for resource in reversed(self.resources): self.release(resource)


def _measure(operation, handle, *, index=None, after=False):
    operation.stage = "identity"
    raw = operation.api.query_object(handle, context=windows._native_context(operation, "query_object", handle=handle, index=index, after=after))
    buffer = None
    primary = None
    cleanup_count = len(operation.cleanup)
    try:
        operation.stage = "security"
        buffer = operation.own(operation.api.acquire_security_descriptor(handle,
            context=windows._native_context(operation, "acquire_security_descriptor", handle=handle, index=index, after=after)))
        data = buffer.copy_bytes()
        descriptor = operation.api.parse_security_descriptor(data,
            context=windows._native_context(operation, "parse_security_descriptor", handle=handle, index=index, after=after))
        if descriptor.self_relative_descriptor_sha256 != records.sha256_bytes(data) or not descriptor.dacl_present or not descriptor.dacl_protected or descriptor.dacl_defaulted:
            raise windows._NativeFault(stage="security")
        result = raw, descriptor.self_relative_descriptor_sha256
        operation.native_evidence.append(result)
        return result
    except BaseException as error:
        primary = error
        raise
    finally:
        if buffer is not None: operation.release(buffer)
        if primary is None and len(operation.cleanup) != cleanup_count:
            raise windows._NativeFault("cleanup_failed", stage="cleanup")


def _ancestry(operation, policy):
    held = []
    for index, path in enumerate(windows._parents(policy.parent_relative_path)):
        binding = windows._directory(policy.root, path)
        records._require(binding.identity is not None, "administrative reader cannot establish unknown ancestry")
        operation.stage = "open"
        handle = operation.own(operation.api.open_existing_directory(binding.canonical_volume_guid_path,
            specification=windows._open_specification(True), context=windows._native_context(operation, "open_existing_directory", index=index)))
        measured = _measure(operation, handle, index=index)
        windows._validate_measurement(measured, policy.root, path, directory=True, identity=binding.identity, descriptor=binding.security_descriptor_sha256)
        held.append(windows._HeldDirectory(binding, handle, index, measured, measured))
    return held


def _remeasure(operation, policy, held):
    for item in held:
        current = _measure(operation, item.handle, index=item.index, after=True)
        item.current = current
        windows._validate_measurement(current, policy.root, item.binding.relative_path, directory=True,
            identity=item.before[0].identity, descriptor=item.binding.security_descriptor_sha256)
        if current != item.before: raise windows._NativeFault()


class Win32HandoffAdministrativeReader:
    def __init__(self, *, expected_plan, authority=None, bindings=None):
        old.validate_handoff_plan(expected_plan)
        self._plan, self._authority = expected_plan, authority
        self._bindings = windows.store.Win32EvidenceStoreBindings() if bindings is None else bindings
        self._api = windows.native.Win32PermissionNativeApi(self._bindings)
        self._lock, self._busy = RLock(), False
        self._evidence = []

    @property
    def failure_evidence(self):
        return tuple(self._evidence)

    def _read(self, input, *, operation, maximum, chunk, authorize):
        op = _AdministrativeReadOperation(self._plan, self._bindings, self._api, operation)
        before = after = None
        copied = bytearray()
        eof = lookup = False
        failure = None
        status = "unavailable"
        with self._lock:
            records._require(not self._busy, "administrative reader reentrant")
            self._busy = True
            try:
                op.stage = "lease_binding"
                authorize()
                p = input.physical_policy
                records._physical(p, plan=self._plan, root_id=input.root_binding_id, relative=input.relative_path,
                    allow_unknown=True, journal=p.confidential_profile_file_sha256 is None)
                records._require(type(maximum) is int and 0 <= maximum < windows.native.SIGNED_LARGE_INTEGER_MAX,
                    "invalid accepted read bound")
                if input.expected_file_ref is not None and input.expected_file_ref.file_bytes > maximum:
                    raise windows._NativeFault("limit_exceeded", stage="read")
                held = _ancestry(op, p)
                op.stage, lookup = "open", True
                handle = op.own(op.api.open_existing_subject(windows._join(p.root.canonical_volume_guid_root, input.relative_path),
                    specification=windows._open_specification(False), context=windows._native_context(op, "open_existing_subject")))
                lookup = False
                measured = _measure(op, handle)
                before = windows._metadata(p.root, input.relative_path, measured, held, p.namespace_relative_path, p.confidential_profile_file_sha256)
                windows._validate_measurement(measured, p.root, input.relative_path, directory=False,
                    identity=p.expected_file_identity, descriptor=p.security_descriptor_sha256)
                if before.size_bytes > maximum:
                    raise windows._NativeFault("limit_exceeded", stage="read")
                windows._streams(op, handle, measured[0])
                _remeasure(op, p, held)
                data = windows._read_exact_opaque_bytes(op, handle, before.size_bytes, chunk, copied)
                eof = True
                final = _measure(op, handle, after=True)
                after = windows._metadata(p.root, input.relative_path, final, held, p.namespace_relative_path, p.confidential_profile_file_sha256)
                windows._validate_measurement(final, p.root, input.relative_path, directory=False, identity=before.identity, descriptor=p.security_descriptor_sha256)
                windows._streams(op, handle, final[0])
                _remeasure(op, p, held)
                if measured != final: raise windows._NativeFault()
                if input.expected_file_ref is not None and (len(data), records.sha256_bytes(data)) != (input.expected_file_ref.file_bytes, input.expected_file_ref.file_sha256):
                    raise windows._NativeFault()
                status = "complete"
            except BaseException as error:
                op.capture_error(error)
                failure = _failure(error, operation=operation, stage=op.stage)
                if lookup and failure.native_error_hex in {"0x2", "0x3"} and not getattr(error, "returned_resources", ()):
                    status = "not_found"
                    failure = records.AdministrativeFailure("open", operation, "not_found", failure.native_error_hex, None)
                elif failure.reason == "mismatch": status = "inconsistent"
                elif eof:
                    status = "inconsistent"
                    if failure.reason == "refused": failure = records.replace(failure, reason="unavailable")
                elif copied or failure.reason == "short_io": status = "partial"
            finally:
                op.close()
                self._busy = False
        if op.cleanup and status == "complete": status, failure = "inconsistent", op.cleanup[0]
        byte_record = records._bytes(bytes(copied)) if copied or eof or status == "partial" else None
        result = records.HandoffRestartRead(input.slot_id, status, byte_record, "complete" if eof else "prefix" if byte_record is not None else "unavailable",
            before, after, records._empty_accounting(), tuple(op.cleanup), failure)
        self._evidence.append((input.slot_id, operation, tuple(op.native_evidence), result))
        return result

    def read_bootstrap_existing(self, pin_or_bound_policy, input):
        value = pin_or_bound_policy
        if type(value) is records.AcceptedAdministrativePin:
            administration._validate_pin(value, self._plan)
            expected = administration._pin_input(value)
            records._require(input == expected and type(input) is records.HandoffRestartInput, "bootstrap pin authorizes only its exact object")
            maximum = value.file_ref.file_bytes
            chunk = min(maximum, windows.native.DWORD_MAX)
        else:
            records._require(type(value) in {administration._BootstrapBinding, administration._BoundAdministration}
                and value.expected_plan == self._plan, "dependency read lacks accepted-byte binding")
            dependency = next((x for x in value.record.dependency_files if administration._dependency_input(x) == input), None)
            records._require(dependency is not None, "dependency not fixed by accepted policy")
            maximum, chunk = value.record.io_limits.maximum_opaque_input_bytes, value.record.io_limits.transfer_chunk_bytes
        return self._read(input, operation="bootstrap_read", maximum=maximum, chunk=chunk, authorize=lambda: None)

    def inspect_revocation_once(self, bound_policy, allowance_id):
        records._require(type(bound_policy) is administration._BoundAdministration and bound_policy.expected_plan == self._plan, "revocation requires bound administration")
        policy = bound_policy.record
        if allowance_id is not None:
            records._require(allowance_id in {x.allowance_id for x in policy.grants}, "unknown revocation allowance")
        p = policy.revocation_policy
        physical = records.AdministrativePhysicalPolicy(p.root, p.namespace_relative_path, p.parent_relative_path, None,
            p.security_descriptor_sha256, p.confidential_profile_file_sha256)
        observations = []
        with self._lock:
            records._require(not self._busy, "reentrant revocation reader")
            self._busy = True
            try:
                for leaf in records._revocation_leaves(policy, bound_policy.pin.file_ref.file_sha256, allowance_id):
                    op = _AdministrativeReadOperation(self._plan, self._bindings, self._api, "revocation_inspection")
                    lookup = False
                    status = "unavailable"
                    try:
                        held = _ancestry(op, physical)
                        op.stage, lookup = "open", True
                        op.own(op.api.open_existing_subject(windows._join(p.root.canonical_volume_guid_root, leaf),
                            specification=windows._open_specification(False), context=windows._native_context(op, "open_existing_subject")))
                        lookup = False
                        # A present marker denies regardless of payload. No payload
                        # read or parsing can convert its presence to absence.
                        _remeasure(op, physical, held)
                        status = "present"
                    except BaseException as error:
                        op.capture_error(error)
                        failure = _failure(error, operation="revocation_inspection", stage=op.stage)
                        if lookup and failure.native_error_hex in {"0x2", "0x3"} and not getattr(error, "returned_resources", ()):
                            try:
                                _remeasure(op, physical, held)
                                status = "absent"
                            except BaseException as changed:
                                op.capture_error(changed)
                    finally:
                        op.close()
                    if op.cleanup: status = "unavailable"
                    self._evidence.append((leaf, "revocation_inspection", tuple(op.native_evidence), tuple(op.cleanup), status))
                    observations.append(records.AdministrativeRevocationObservation(leaf, status))
            finally:
                self._busy = False
        return tuple(observations)

    def read_restart_existing(self, ticket, input, lease):
        records._require(type(ticket) is records.HandoffRestartTicket and type(input) is records.HandoffRestartInput
            and input in ticket.inputs and self._authority is not None, "restart reader has no matching ticket/authority")
        return self._read(input, operation="read_handoff_verification_input", maximum=ticket.io_limits.maximum_opaque_input_bytes,
            chunk=ticket.io_limits.transfer_chunk_bytes, authorize=lambda: self._authority.claim_dispatch_once(lease, ticket, input.slot_id))


def read_bootstrap_existing(reader, pin_or_bound_policy, input):
    return reader.read_bootstrap_existing(pin_or_bound_policy, input)


def inspect_revocation_once(reader, bound_policy, allowance_id):
    return reader.inspect_revocation_once(bound_policy, allowance_id)


def read_restart_existing(reader, ticket, input, lease):
    return reader.read_restart_existing(ticket, input, lease)
