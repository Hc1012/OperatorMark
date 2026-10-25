"""Secured, same-held-object witness publication without retries or repair."""
from __future__ import annotations

from dataclasses import dataclass

from .canonical import sha256_bytes
from . import permission_handoff_records as records
from . import permission_handoff_sources as sources
from . import permission_handoff_windows as windows


def derive_witness_creating_path(plan, slot):
    if slot not in records.WITNESS_FILENAMES:
        raise ValueError("unplanned witness slot")
    return plan.witness_store.parent_relative_path + "/" + slot + ".creating"


@dataclass
class _WriteState:
    status: str = "failed"
    publication: str = "not_created"
    certainty: str = "no_effect"
    ownership: str = "no_object"
    handle: object = None
    last: records.HandoffLastKnownObject | None = None
    failure: records.HandoffFailure | None = None


class Win32PermissionHandoffWitnessSink:
    def __init__(self, plan, configuration, authority, bindings, api):
        self._plan, self._configuration, self._authority = plan, configuration, authority
        self._bindings, self._api = bindings, api
        self._evidence = []

    @property
    def failure_evidence(self):
        return tuple(self._evidence)

    def publish_exclusive(self, plan, slot, data, lease):
        operation = windows._Operation(plan, self._bindings, self._api, "retain_handoff_witness")
        state = _WriteState()
        try:
            windows.validate_handoff_windows_configuration(expected_plan=plan, configuration=self._configuration)
            if plan != self._plan or slot not in records.WITNESS_FILENAMES or type(data) is not bytes or not data:
                raise ValueError("witness plan/slot/bytes mismatch")
            if len(data) >= windows.native.SIGNED_LARGE_INTEGER_MAX:
                raise ValueError("unsupported witness extent including EOF")
            kinds = ("start", "outcome", "acquisition", "verification", "index", "receipt")
            records.strict_handoff_record_loads(data, kind=kinds[records.WITNESS_FILENAMES.index(slot)], expected_plan=plan)
            self._authority._claim_dispatch_once(lease, plan, "retain_handoff_witness", slot)
            self._publish_once(operation, state, plan, slot, data)
        except Exception as error:
            operation.capture_error(error)
            state.failure = windows._native_failure(error, operation=operation.operation, stage=operation.stage)
            if state.certainty == "unknown":
                state.status = "unknown"
        finally:
            operation.close()
        if operation.cleanup and state.status in {"retained", "collision"}:
            state.status = "failed"
            state.failure = state.failure or operation.cleanup[0]
        result = self._project_write_result(slot, state, operation)
        self._evidence.append((slot, tuple(operation.native_evidence)))
        return result

    def _publish_once(self, operation, state, plan, slot, data):
        expected = plan.witness_store
        root = windows._root(self._configuration, expected.root_binding_id)
        creating = derive_witness_creating_path(plan, slot)
        final = creating[:-len(".creating")]
        held = windows._open_ancestry(operation, root, expected.parent_relative_path)
        operation.stage = "security"
        prepared = self._bindings.prepare_create_security(self._configuration.witness_creation_sddl, expected.created_object_security_descriptor_sha256)
        if (type(prepared) is not windows.store.PreparedCreateSecurity or prepared.sddl != self._configuration.witness_creation_sddl
                or type(prepared.descriptor) is not bytes or sha256_bytes(prepared.descriptor) != expected.created_object_security_descriptor_sha256
                or prepared.descriptor_sha256 != expected.created_object_security_descriptor_sha256):
            raise windows._NativeFault("malformed_output", stage="security")
        windows._remeasure_ancestry(operation, root, held)
        operation.stage = "create"
        state.publication, state.certainty, state.ownership = "unknown", "unknown", "unknown"
        try:
            raw_handle = self._bindings.create_secured_exclusive(windows._join(root.canonical_volume_guid_root, creating), prepared)
        except windows.store.Win32EvidenceStoreError as error:
            if error.handle_owned and windows.native._valid_handle(error.returned_handle):
                state.handle = operation.own(windows.native.Win32OwnedObjectHandle(self._bindings, error.returned_handle,
                    ("file_handle", "subject", None), object_kind="file", requested_path=windows._join(root.canonical_volume_guid_root, creating)))
                state.publication, state.ownership = "creating", "owned_by_this_invocation"
            elif error.collision:
                state.status, state.publication, state.certainty, state.ownership = "collision", "not_created", "no_effect", "not_owned"
            elif not error.invoked:
                state.publication, state.certainty, state.ownership = "not_created", "no_effect", "no_object"
            raise
        if not windows.native._valid_handle(raw_handle):
            raise windows._NativeFault("malformed_output", stage="create")
        state.handle = operation.own(windows.native.Win32OwnedObjectHandle(self._bindings, raw_handle,
            ("file_handle", "subject", None), object_kind="file", requested_path=windows._join(root.canonical_volume_guid_root, creating)))
        state.publication, state.certainty, state.ownership = "creating", "confirmed", "owned_by_this_invocation"
        state.last, empty = self._measure_written_object(operation, root, held, creating, state.handle, None)
        if empty:
            raise windows._NativeFault(stage="create")
        identity = state.last.metadata.identity
        windows._remeasure_ancestry(operation, root, held)
        self._write_chunks_once(operation, state, data)
        operation.stage = "flush"
        self._bindings.flush_file(state.handle.raw_handle)
        state.last, actual = self._measure_written_object(operation, root, held, creating, state.handle, identity)
        if actual != data:
            raise windows._NativeFault(stage="read")
        windows._remeasure_ancestry(operation, root, held)
        operation.stage = "publish"
        state.publication, state.certainty = "unknown", "unknown"
        try:
            renamed = self._bindings.rename_no_replace(state.handle.raw_handle,
                windows._join(root.canonical_volume_guid_root, expected.parent_relative_path), slot)
        except windows.store.Win32EvidenceStoreError as error:
            if error.collision or not error.invoked:
                state.publication, state.certainty = "creating", "confirmed"
            raise
        if type(renamed) is not str or renamed != windows._join(root.canonical_volume_guid_root, final):
            raise windows._NativeFault("malformed_output", stage="publish")
        state.publication, state.certainty = "published", "confirmed"
        state.last, actual = self._measure_written_object(operation, root, held, final, state.handle, identity)
        if actual != data:
            raise windows._NativeFault(stage="read")
        operation.stage = "flush"
        self._bindings.flush_file(state.handle.raw_handle)
        state.last, actual = self._measure_written_object(operation, root, held, final, state.handle, identity)
        if actual != data:
            raise windows._NativeFault(stage="read")
        windows._remeasure_ancestry(operation, root, held)
        state.status = "retained"

    def _write_chunks_once(self, operation, state, data):
        operation.stage = "write"
        size = self._configuration.transfer_chunk_bytes
        for offset in range(0, len(data), size):
            chunk = data[offset:offset + size]
            state.certainty = "unknown"
            count = self._bindings.write_once_at(state.handle.raw_handle, offset, chunk)
            operation.native_evidence.append(("write_once_at", offset, len(chunk), count))
            if type(count) is not int or not 0 <= count <= len(chunk):
                raise windows._NativeFault("malformed_output", stage="write")
            state.certainty = "confirmed"
            if count != len(chunk):
                raise windows._NativeFault("short_io", stage="write")

    def _measure_written_object(self, operation, root, held, relative, handle, identity):
        expected = self._plan.witness_store
        measured = windows._measure_held_object(operation, handle, after=identity is not None)
        windows._validate_measurement(measured, root, relative, directory=False, identity=identity, descriptor=expected.created_object_security_descriptor_sha256)
        windows._streams(operation, handle, measured[0])
        if identity is None and measured[0].end_of_file_bytes != 0:
            raise windows._NativeFault(stage="create")
        windows._remeasure_ancestry(operation, root, held)
        copied = bytearray()
        eof = False
        try:
            data = windows._read_exact_opaque_bytes(operation, handle, measured[0].end_of_file_bytes, self._configuration.transfer_chunk_bytes, copied)
            eof = True
        finally:
            operation.native_evidence.append(("readback", bytes(copied), eof))
        final = windows._measure_held_object(operation, handle, after=True)
        windows._validate_measurement(final, root, relative, directory=False, identity=measured[0].identity, descriptor=expected.created_object_security_descriptor_sha256)
        windows._streams(operation, handle, final[0])
        if measured != final:
            raise windows._NativeFault()
        metadata = windows._metadata(root, relative, final, held, expected.namespace_relative_path, expected.confidential_profile_file_ref.file_sha256)
        return records.HandoffLastKnownObject(metadata, len(data), sha256_bytes(data)), data

    @staticmethod
    def _project_write_result(slot, state, operation):
        handle_state = ("unknown" if state.ownership == "unknown" else "not_acquired") if state.handle is None else "released" if state.handle.released else "release_failed"
        return records.HandoffWriteResult(slot, state.status, state.last, state.publication, state.certainty,
            state.ownership, handle_state, sources._empty_accounting(), tuple(operation.cleanup), state.failure)
