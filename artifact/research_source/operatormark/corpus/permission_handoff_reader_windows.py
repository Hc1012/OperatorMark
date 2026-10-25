"""Independent reads through enrolled, held Win32 objects; no native discovery."""
from __future__ import annotations

from dataclasses import asdict
import threading

from .canonical import canonical_json_bytes, sha256_bytes
from . import permission_handoff_records as records
from . import permission_handoff_sources as sources
from . import permission_handoff_windows as windows


class Win32PermissionHandoffReader:
    def __init__(self, plan, configuration, authority, bindings, api):
        self._plan, self._configuration, self._authority = plan, configuration, authority
        self._bindings, self._api = bindings, api
        self._journal_parent = plan.journal_binding.expected_attempt_directory_identity
        self._journal_lock = threading.RLock()
        self._evidence = []

    @property
    def failure_evidence(self):
        return tuple(self._evidence)

    def read_existing(self, plan, slot, lease):
        operation = windows._Operation(plan, self._bindings, self._api, "read_handoff_input")
        try:
            windows.validate_handoff_windows_configuration(expected_plan=plan, configuration=self._configuration)
            if plan != self._plan or slot not in plan.input_slots:
                raise ValueError("provider plan/slot differs from construction")
            self._authority._claim_dispatch_once(lease, plan, "read_handoff_input", slot.slot_id)
            with self._journal_lock:
                result = self._read_existing_once(operation, plan, slot)
        except Exception as error:
            operation.capture_error(error)
            operation.close()
            result = records.HandoffReadResult(slot.slot_id, "unavailable", None, "unavailable", None, None,
                sources._empty_accounting(), tuple(operation.cleanup), windows._native_failure(error, operation=operation.operation, stage=operation.stage))
        self._evidence.append((slot.slot_id, tuple(operation.native_evidence)))
        return result

    def _bind_journal_parent(self, plan, slot, parent, data):
        if slot.role == "journal_activation":
            expected = canonical_json_bytes(asdict(plan.journal_binding.expected_activation))
            if data != expected:
                raise windows._NativeFault()
            if self._journal_parent is not None and parent != self._journal_parent:
                raise windows._NativeFault()
            self._journal_parent = parent

    def _read_existing_once(self, operation, plan, slot):
        policy = self._configuration.input_policies[plan.input_slots.index(slot)]
        root = windows._root(self._configuration, slot.root_binding_id)
        before = after = None
        copied = bytearray()
        eof = False
        target_lookup = False
        failure = None
        status = "unavailable"
        held = []
        try:
            if slot.role.startswith("journal_") and slot.role != "journal_activation" and self._journal_parent is None:
                raise windows._NativeFault("unavailable", stage="lease_binding")
            held = windows._open_ancestry(operation, root, policy.parent_relative_path)
            if slot.role.startswith("journal_") and self._journal_parent is not None and held[-1].current[0].identity != self._journal_parent:
                raise windows._NativeFault()
            operation.stage = "open"
            target_lookup = True
            handle = operation.own(operation.api.open_existing_subject(windows._join(root.canonical_volume_guid_root, slot.relative_path),
                specification=windows._open_specification(False), context=windows._native_context(operation, "open_existing_subject")))
            target_lookup = False
            measured = windows._measure_held_object(operation, handle)
            before = windows._metadata(root, slot.relative_path, measured, held, policy.namespace_relative_path, policy.confidential_profile_file_sha256)
            windows._validate_measurement(measured, root, slot.relative_path, directory=False, identity=policy.expected_file_identity, descriptor=policy.security_descriptor_sha256)
            windows._streams(operation, handle, measured[0])
            windows._remeasure_ancestry(operation, root, held)
            data = self._read_exact_opaque_bytes(operation, handle, before.size_bytes, copied)
            eof = True
            final = windows._measure_held_object(operation, handle, after=True)
            after = windows._metadata(root, slot.relative_path, final, held, policy.namespace_relative_path, policy.confidential_profile_file_sha256)
            windows._validate_measurement(final, root, slot.relative_path, directory=False, identity=before.identity, descriptor=policy.security_descriptor_sha256)
            windows._streams(operation, handle, final[0])
            windows._remeasure_ancestry(operation, root, held)
            if measured != final:
                raise windows._NativeFault()
            if slot.expected_file_ref is not None and (len(data), sha256_bytes(data)) != (slot.expected_file_ref.file_bytes, slot.expected_file_ref.file_sha256):
                raise windows._NativeFault()
            if slot.slot_id == "handoff-plan" and data != canonical_json_bytes(records._wire(plan)) + b"\n":
                raise windows._NativeFault()
            if slot.role == "journal_activation" and data != canonical_json_bytes(asdict(plan.journal_binding.expected_activation)):
                raise windows._NativeFault()
            status = "complete"
        except Exception as error:
            operation.capture_error(error)
            failure = windows._native_failure(error, operation=operation.operation, stage=operation.stage)
            if target_lookup and failure.native_error_hex in {"0x2", "0x3"} and not getattr(error, "returned_resources", ()):
                status = "not_found"
                failure = records.HandoffFailure("open", operation.operation, "not_found", failure.native_error_hex, None)
            elif failure.reason == "mismatch":
                status = "inconsistent"
            elif eof:
                status = "inconsistent"
                if failure.reason == "refused":
                    # Post-copy API denial makes complete verification unavailable;
                    # retain the real Windows code and original native carrier.
                    failure = records.HandoffFailure(failure.stage, operation.operation, "unavailable", failure.native_error_hex, None)
            elif copied or failure.reason == "short_io":
                status = "partial"
        finally:
            operation.close()
        if operation.cleanup and status == "complete":
            status, failure = "inconsistent", operation.cleanup[0]
        result = self._project_read_result(slot, status, before, after, copied, eof, operation, failure)
        if status == "complete":
            try:
                sources._validate_read_facts(result, plan=plan, slot=slot)
                self._bind_journal_parent(plan, slot, after.parent_identity, bytes(copied))
            except Exception as error:
                operation.capture_error(error)
                result = records.HandoffReadResult(slot.slot_id, "inconsistent", result.data, result.data_completeness, before, after,
                    sources._empty_accounting(), tuple(operation.cleanup), windows._native_failure(error, operation=operation.operation, stage="identity"))
        return result

    def _read_exact_opaque_bytes(self, operation, handle, size, copied):
        return windows._read_exact_opaque_bytes(operation, handle, size, self._configuration.transfer_chunk_bytes, copied)

    @staticmethod
    def _project_read_result(slot, status, before, after, copied, eof, operation, failure):
        data = None
        completeness = "unavailable"
        if copied or eof or status == "partial":
            data = records.ByteRecord(bytes(copied).hex(), len(copied), sha256_bytes(bytes(copied)))
            completeness = "complete" if eof else "prefix"
        return records.HandoffReadResult(slot.slot_id, status, data, completeness, before, after,
                                         sources._empty_accounting(), tuple(operation.cleanup), failure)
