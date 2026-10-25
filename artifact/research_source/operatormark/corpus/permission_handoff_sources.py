"""Injected authority/reader/sink ports and pure handoff input projection."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Literal, Protocol, cast

from .canonical import canonical_json_bytes, sha256_bytes, strict_canonical_json_loads
from . import permission_authz_evidence_verifier as authz_verifier
from . import permission_authority
from . import permission_observation_windows as observation
from . import permission_principal_authz_windows as supplied_authz
from . import private_permission_evidence as evidence
from .permission_handoff_records import (
    ByteRecord,
    GRANT_OPERATIONS,
    HandoffAcquireResult,
    HandoffAuthorityAccounting,
    HandoffAuthorityLease,
    HandoffCheckpoint,
    HandoffCheckpointResult,
    HandoffFailure,
    HandoffFinishResult,
    HandoffLastKnownObject,
    HandoffLeaseScope,
    HandoffObjectMetadata,
    HandoffPlan,
    HandoffReadResult,
    HandoffWriteResult,
    InputSlot,
    PreparedAuthzVerifierInputs,
    TokenTrace,
    WITNESS_FILENAMES,
    _decode_dataclass,
    _decode_original_runtime,
    _exact_int,
    _relative_path,
    _text,
    _validate_byte_record,
    _validate_native_identity,
    _validate_token_trace,
    _wire,
    validate_handoff_plan,
)


class PermissionHandoffSourceError(ValueError):
    """An injected provider result or independent projection is invalid."""


class PermissionHandoffAuthority(Protocol):
    def acquire(
        self,
        plan: HandoffPlan,
        operation: Literal["read_handoff_input", "retain_handoff_witness"],
        slot_id: str,
    ) -> HandoffAcquireResult: ...

    def check_active(
        self, lease: HandoffAuthorityLease, checkpoint: str
    ) -> HandoffCheckpointResult: ...

    def finish(
        self,
        lease: HandoffAuthorityLease,
        requested_outcome: Literal["succeeded", "refused", "failed"],
    ) -> HandoffFinishResult: ...


class PermissionHandoffReader(Protocol):
    def read_existing(
        self, plan: HandoffPlan, slot: InputSlot, lease: HandoffAuthorityLease
    ) -> HandoffReadResult: ...


class PermissionHandoffWitnessSink(Protocol):
    def publish_exclusive(
        self,
        plan: HandoffPlan,
        slot: str,
        data: bytes,
        lease: HandoffAuthorityLease,
    ) -> HandoffWriteResult: ...


def _raise(message: str, cause: BaseException | None = None) -> None:
    error = PermissionHandoffSourceError(message)
    if cause is None:
        raise error
    raise error from cause


def _exception_failure(
    *, stage: str, operation: str, reason: str = "malformed_output"
) -> HandoffFailure:
    return HandoffFailure(
        stage=cast(object, stage),
        operation=cast(object, operation),
        reason=cast(object, reason),
        native_error_hex=None,
        exception=None,
    )


def _empty_accounting() -> HandoffAuthorityAccounting:
    return HandoffAuthorityAccounting(
        "not_attempted", False, "not_applicable", (), False, None,
        "not_attempted", None, None,
    )


def _native_identity_text(identity: observation.NativeFileIdentity) -> str:
    _validate_native_identity(identity, name="native identity")
    return (
        "win32-file-id-v1:"
        + identity.volume_serial_number_hex
        + ":"
        + identity.file_id_128_hex
    )


def _expected_scope_identities(
    plan: HandoffPlan, *, role: str
) -> tuple[str, str | None]:
    if role.startswith("journal_"):
        namespace = _native_identity_text(plan.journal_binding.expected_root_identity)
        parent = (
            None
            if plan.journal_binding.expected_attempt_directory_identity is None
            else _native_identity_text(plan.journal_binding.expected_attempt_directory_identity)
        )
        return namespace, parent
    if role in {
        "base_primary", "base_failure_sidecar", "authz_primary", "authz_failure_sidecar"
    }:
        return (
            plan.expected_store.namespace_object_identity,
            plan.expected_store.records_parent_object_identity,
        )
    # Authority inputs remain fixed by their root/slot/path and their explicit
    # FileRef.  Their provider enrollment supplies these two opaque bindings.
    return plan.witness_store.namespace_object_identity, plan.witness_store.parent_object_identity


def _scope_payload(scope: HandoffLeaseScope) -> dict[str, object]:
    payload = asdict(scope)
    payload.pop("scope_self_sha256")
    return payload


def _validate_lease_scope(
    lease: HandoffAuthorityLease,
    *,
    plan: HandoffPlan,
    operation: str,
    slot_id: str,
    root_binding_id: str,
    role: str,
) -> None:
    try:
        scope = lease.scope
    except Exception as exc:
        _raise("handoff lease did not expose its immutable scope", exc)
    if type(scope) is not HandoffLeaseScope:
        _raise("handoff lease scope has the wrong exact type")
    namespace, parent = _expected_scope_identities(plan, role=role)
    expected = {
        "operation": operation,
        "plan_self_sha256": plan.plan_self_sha256,
        "request_self_sha256": plan.expected_request.request_self_sha256,
        "full_state4_authorization_file_sha256": plan.authority_binding.full_state4_authorization_file_sha256,
        "root_binding_id": root_binding_id,
        "slot_id": slot_id,
        "namespace_identity": namespace,
        "parent_identity": parent,
    }
    for name, wanted in expected.items():
        if type(getattr(scope, name)) is not type(wanted) or getattr(scope, name) != wanted:
            _raise(f"handoff lease scope {name} mismatch")
    if scope.scope_self_sha256 != sha256_bytes(canonical_json_bytes(_scope_payload(scope))):
        _raise("handoff lease scope self-hash mismatch")


def _validate_handoff_authority_accounting(
    accounting: HandoffAuthorityAccounting,
    *,
    operation: Literal["read_handoff_input", "retain_handoff_witness"],
) -> None:
    if type(accounting) is not HandoffAuthorityAccounting:
        _raise("authority accounting has the wrong exact type")
    _decode_dataclass(HandoffAuthorityAccounting, _wire(accounting), name="authority accounting")
    if accounting.acquire_status not in {"not_attempted", "granted", "refused", "failed"}:
        _raise("authority acquire status is invalid")
    if accounting.lease_binding_status not in {"not_applicable", "valid", "invalid"}:
        _raise("authority lease binding status is invalid")
    if accounting.finish_status not in {"not_attempted", "succeeded", "failed"}:
        _raise("authority finish status is invalid")
    names = ("initial", "before_open", "before_bytes", "final")
    if operation == "retain_handoff_witness":
        names = ("initial", "before_open", "before_bytes", "before_publish", "final")
    observed_names = tuple(item.name for item in accounting.checkpoints)
    if observed_names != names[: len(observed_names)] or len(set(observed_names)) != len(observed_names):
        _raise("authority checkpoints are not an ordered prefix")
    inactive = False
    for checkpoint in accounting.checkpoints:
        if inactive:
            _raise("authority checkpoint occurred after the first inactive result")
        if type(checkpoint) is not HandoffCheckpoint:
            _raise("authority checkpoint has the wrong exact type")
        if checkpoint.status == "active":
            if checkpoint.reason is not None or inactive:
                _raise("active authority checkpoint is contradictory")
        else:
            if checkpoint.reason not in {"refused", "expired", "scope_mismatch", "unavailable", "malformed_output"}:
                _raise("inactive authority checkpoint lacks a closed reason")
            inactive = True
    if accounting.acquire_status == "granted":
        if not accounting.lease_returned or accounting.lease_binding_status not in {"valid", "invalid"}:
            _raise("granted authority accounting lacks its lease")
    elif accounting.acquire_status == "failed" and accounting.lease_returned:
        if accounting.lease_binding_status != "invalid":
            _raise("malformed acquired lease must remain invalid")
    elif accounting.lease_returned or accounting.lease_binding_status != "not_applicable":
        _raise("non-granted authority accounting invented a lease")
    if accounting.finish_attempted:
        if not accounting.lease_returned or accounting.requested_outcome is None or accounting.finish_status == "not_attempted":
            _raise("authority terminal accounting is incomplete")
        if accounting.effect_certainty is None:
            _raise("authority terminal accounting lacks effect certainty")
        if accounting.effect_certainty == "confirmed":
            if accounting.finish_status != "succeeded" or accounting.recorded_outcome != accounting.requested_outcome:
                _raise("confirmed authority finish did not record the requested outcome")
        elif accounting.recorded_outcome is not None:
            _raise("unconfirmed authority finish invented a recorded outcome")
    elif (
        accounting.requested_outcome is not None
        or accounting.finish_status != "not_attempted"
        or accounting.effect_certainty is not None
        or accounting.recorded_outcome is not None
    ):
        _raise("unattempted authority finish has terminal fields")
    if accounting.requested_outcome == "succeeded" and (
        observed_names != names or inactive or accounting.lease_binding_status != "valid"
    ):
        _raise("successful authority outcome lacks every active checkpoint")


def _validate_failure(failure: HandoffFailure, *, operation: str) -> None:
    if type(failure) is not HandoffFailure or failure.operation != operation:
        _raise("provider failure has the wrong exact operation")
    _decode_dataclass(HandoffFailure, _wire(failure), name="provider failure")
    if failure.native_error_hex is not None:
        import re
        if type(failure.native_error_hex) is not str or re.fullmatch(r"0x(?:0|[1-9a-f][0-9a-f]*)", failure.native_error_hex) is None or failure.native_error_hex == "0x0":
            _raise("provider failure native error is malformed")


def _validate_metadata_shape(
    metadata: HandoffObjectMetadata,
    *,
    slot: InputSlot,
) -> None:
    if type(metadata) is not HandoffObjectMetadata:
        _raise("measured object metadata has the wrong exact type")
    _decode_dataclass(HandoffObjectMetadata, _wire(metadata), name="measured object metadata")
    if metadata.root_binding_id != slot.root_binding_id or metadata.relative_path != slot.relative_path:
        _raise("measured object address differs from its independent slot")
    _relative_path(metadata.relative_path, name="measured metadata relative_path")
    _text(metadata.canonical_volume_guid_path, name="measured canonical volume path")
    for name in ("identity", "parent_identity", "namespace_identity"):
        _validate_native_identity(getattr(metadata, name), name=f"measured metadata {name}")
    _exact_int(metadata.size_bytes, name="measured metadata size")
    for name in ("security_descriptor_sha256", "volume_identifier_commitment"):
        from .canonical import require_sha256
        require_sha256(getattr(metadata, name), name=f"measured metadata {name}")
    if metadata.confidential_profile_file_sha256 is not None:
        require_sha256(metadata.confidential_profile_file_sha256, name="measured metadata profile")


def _validate_measured_identity(
    metadata: HandoffObjectMetadata,
    *,
    plan: HandoffPlan,
    slot: InputSlot,
) -> None:
    _validate_metadata_shape(metadata, slot=slot)
    private_role = slot.role in {
        "base_primary", "base_failure_sidecar", "authz_primary", "authz_failure_sidecar"
    }
    if private_role:
        if metadata.confidential_profile_file_sha256 != plan.expected_store.confidential_store_profile_sha256:
            _raise("measured private object profile mismatch")
        if _native_identity_text(metadata.namespace_identity) != plan.expected_store.namespace_object_identity:
            _raise("measured private namespace identity mismatch")
        if _native_identity_text(metadata.parent_identity) != plan.expected_store.records_parent_object_identity:
            _raise("measured private parent identity mismatch")
        if metadata.volume_identifier_commitment != plan.expected_store.volume_identifier_commitment:
            _raise("measured private volume mismatch")
        if metadata.security_descriptor_sha256 != plan.expected_store.required_created_object_security_descriptor_sha256:
            _raise("measured private security descriptor mismatch")
    elif slot.role.startswith("journal_"):
        if metadata.confidential_profile_file_sha256 is not None:
            _raise("journal metadata pretends to use the private store profile")
        if _native_identity_text(metadata.namespace_identity) != _native_identity_text(plan.journal_binding.expected_root_identity):
            _raise("measured journal root identity mismatch")
        expected_parent = plan.journal_binding.expected_attempt_directory_identity
        if expected_parent is not None and metadata.parent_identity != expected_parent:
            _raise("measured journal parent identity mismatch")


def _validate_read_facts(result: HandoffReadResult, *, plan: HandoffPlan, slot: InputSlot) -> None:
    """Validate available provider facts without assuming a future terminal ack."""
    validate_handoff_plan(plan)
    if slot not in plan.input_slots:
        _raise("read result uses a slot outside the independent plan")
    if type(result) is not HandoffReadResult:
        _raise("read result has the wrong exact type")
    if result.slot_id != slot.slot_id:
        _raise("read result slot mismatch")
    if result.status not in {"complete", "not_found", "unavailable", "partial", "inconsistent"}:
        _raise("read result status is invalid")
    if result.data_completeness not in {"complete", "prefix", "unavailable"}:
        _raise("read data completeness is invalid")
    data: bytes | None = None
    if result.data is not None:
        data = _validate_byte_record(result.data, name="read result data")
    if (data is None) != (result.data_completeness == "unavailable"):
        _raise("read bytes and completeness tag contradict one another")
    if result.status != "complete" and result.failure is None:
        _raise("incomplete read lacks its observed failure")
    for metadata in (result.before, result.after):
        if metadata is not None:
            if result.status == "inconsistent" and type(result.failure) is HandoffFailure and result.failure.reason == "mismatch":
                _validate_metadata_shape(metadata, slot=slot)
            else:
                _validate_measured_identity(metadata, plan=plan, slot=slot)
    for failure in result.cleanup_failures:
        _validate_failure(failure, operation="read_handoff_input")
        if failure.stage != "cleanup":
            _raise("read cleanup failure has the wrong stage")
    if result.failure is not None:
        _validate_failure(result.failure, operation="read_handoff_input")
    if result.status == "complete":
        if (
            data is None
            or result.data_completeness != "complete"
            or result.before is None
            or result.after is None
            or result.before != result.after
            or result.before.size_bytes != len(data)
            or result.failure is not None
            or result.cleanup_failures
        ):
            _raise("complete read result lacks stable bytes or cleanup")
        if slot.expected_file_ref is not None and (
            slot.expected_file_ref.file_bytes != len(data)
            or slot.expected_file_ref.file_sha256 != sha256_bytes(data)
        ):
            _raise("complete read differs from its independently expected file reference")
        if slot.slot_id == "handoff-plan" and data != canonical_json_bytes(_wire(plan)) + b"\n":
            _raise("separately retained plan bytes differ from the independently prepared plan")
    elif result.status == "not_found":
        if (
            data is not None
            or result.before is not None
            or result.after is not None
            or result.failure.stage != "open"
            or result.failure.reason != "not_found"
        ):
            _raise("not-found read contradicts its failed lookup boundary")
        # A held parent can still fail cleanup after the lookup failed. Keep it.
    elif result.status == "unavailable" and data is not None:
        _raise("unavailable read must preserve reached bytes as partial or inconsistent")
    elif result.status == "partial" and (data is None or result.data_completeness != "prefix"):
        _raise("partial read lacks its observed byte prefix")
    elif result.status == "inconsistent" and result.failure.reason == "refused":
        _raise("an observed inconsistency is not an access refusal")


def validate_handoff_read_result(
    result: HandoffReadResult, *, plan: HandoffPlan, slot: InputSlot
) -> None:
    _validate_read_facts(result, plan=plan, slot=slot)
    _validate_handoff_authority_accounting(
        result.authority_accounting, operation="read_handoff_input"
    )
    if result.status == "complete":
        if (result.authority_accounting.effect_certainty != "confirmed"
                or result.authority_accounting.recorded_outcome != "succeeded"):
            _raise("complete read lacks confirmed authority completion")
    elif result.authority_accounting.recorded_outcome == "succeeded":
        _raise("incomplete read contradicts a recorded successful operation")


def _validate_last_known_witness_address(result, *, plan, slot_id):
    metadata = result.last_known_object.metadata
    expected = f"{plan.witness_store.parent_relative_path}/{slot_id}"
    permitted = {expected}
    if (result.status != "retained" and result.ownership in {"owned_by_this_invocation", "unknown"}
            and result.publication_state in {"creating", "published", "unknown"}):
        permitted.add(expected + ".creating")
    if metadata.relative_path not in permitted:
        _raise("last-known write object address mismatch")
    _validate_metadata_shape(metadata, slot=InputSlot(slot_id, plan.witness_store.root_binding_id,
                                                   metadata.relative_path, "authority_input", None))


def _validate_write_facts(
    result: HandoffWriteResult,
    *,
    plan: HandoffPlan,
    slot_id: str,
    data: bytes,
) -> None:
    """Validate storage facts; authority completion is a separate later boundary."""
    validate_handoff_plan(plan)
    if slot_id not in WITNESS_FILENAMES or type(result) is not HandoffWriteResult:
        _raise("write result slot or type is invalid")
    _decode_dataclass(HandoffWriteResult, _wire(result), name="write result")
    if result.slot_id != slot_id or type(data) is not bytes:
        _raise("write result does not bind its requested slot/bytes")
    for failure in result.cleanup_failures:
        _validate_failure(failure, operation="retain_handoff_witness")
        if failure.stage != "cleanup":
            _raise("write cleanup failure has the wrong stage")
    if result.failure is not None:
        _validate_failure(result.failure, operation="retain_handoff_witness")
    if result.status != "retained" and result.failure is None:
        _raise("unretained write lacks its observed failure")
    if result.effect_certainty == "no_effect" and (
        result.publication_state != "not_created"
        or result.ownership not in {"no_object", "not_owned"}
        or result.handle_state != "not_acquired"
    ):
        _raise("no-effect write contradicts publication, ownership, or an acquired object")
    if result.ownership == "no_object" and (
        result.last_known_object is not None
        or result.publication_state != "not_created"
        or result.handle_state != "not_acquired"
    ):
        _raise("no-object write invented an object, publication, or handle")
    if result.ownership == "owned_by_this_invocation" and (
        result.publication_state == "not_created" or result.handle_state == "not_acquired"
    ):
        _raise("owned write contradicts creation or object handle acquisition")
    if result.publication_state in {"creating", "published"} and result.ownership == "not_owned":
        _raise("write cannot create or publish an unowned collision object")
    if result.handle_state == "release_failed" and not result.cleanup_failures:
        _raise("failed object release lacks its cleanup failure")
    if result.last_known_object is not None:
        last = result.last_known_object
        if type(last) is not HandoffLastKnownObject:
            _raise("last-known write object has the wrong exact type")
        _validate_last_known_witness_address(result, plan=plan, slot_id=slot_id)
        _exact_int(last.file_bytes, name="last-known file bytes")
        from .canonical import require_sha256
        require_sha256(last.file_sha256, name="last-known file hash")
    if result.status == "retained":
        last = result.last_known_object
        if (
            last is None
            or last.file_bytes != len(data)
            or last.file_sha256 != sha256_bytes(data)
            or last.metadata.size_bytes != len(data)
            or result.publication_state != "published"
            or result.effect_certainty != "confirmed"
            or result.ownership != "owned_by_this_invocation"
            or result.handle_state != "released"
            or result.cleanup_failures
            or result.failure is not None
        ):
            _raise("retained write lacks exact bytes or release")
        metadata = last.metadata
        for name in ("identity", "namespace_identity", "parent_identity"):
            _validate_native_identity(getattr(metadata, name), name="witness " + name)
        store = plan.witness_store
        if (
            _native_identity_text(metadata.namespace_identity) != store.namespace_object_identity
            or _native_identity_text(metadata.parent_identity) != store.parent_object_identity
            or metadata.volume_identifier_commitment != store.volume_identifier_commitment
            or metadata.confidential_profile_file_sha256 != store.confidential_profile_file_ref.file_sha256
            or metadata.security_descriptor_sha256 != store.created_object_security_descriptor_sha256
        ):
            _raise("retained witness lacks independently bound identity/confidentiality")
    elif result.status == "collision":
        if (
            result.ownership != "not_owned"
            or result.handle_state != "not_acquired"
            or result.effect_certainty != "no_effect"
            or result.publication_state != "not_created"
            or result.cleanup_failures
            or result.failure.reason != "collision"
        ):
            _raise("collision write contradicts its observed refusal, ownership, or effects")
    if result.handle_state == "release_failed" and result.status == "retained":
        _raise("close failure was rewritten as retained")


def validate_handoff_write_result(
    result: HandoffWriteResult,
    *,
    plan: HandoffPlan,
    slot_id: str,
    data: bytes,
) -> None:
    _validate_write_facts(result, plan=plan, slot_id=slot_id, data=data)
    _validate_handoff_authority_accounting(
        result.authority_accounting, operation="retain_handoff_witness"
    )
    if result.status == "retained":
        if (result.authority_accounting.effect_certainty != "confirmed"
                or result.authority_accounting.recorded_outcome != "succeeded"):
            _raise("retained write lacks confirmed authority completion")
    elif result.authority_accounting.recorded_outcome == "succeeded":
        _raise("unretained write contradicts a recorded successful operation")


def _domain_hash(domain: str, value: object) -> str:
    return sha256_bytes(domain.encode("utf-8") + b"\n" + canonical_json_bytes(value))


def _journal_addresses(plan: HandoffPlan) -> dict[str, tuple[str, str]]:
    activation_bytes = canonical_json_bytes(asdict(plan.journal_binding.expected_activation))
    session_id = sha256_bytes(
        b"operatormark-permission-authority-filesystem-session-v1\n" + activation_bytes
    )
    result: dict[str, tuple[str, str]] = {}
    for operation in GRANT_OPERATIONS:
        grant = plan.expected_grants[operation]
        allowance_kind = permission_authority._allowance_kind(cast(object, operation))
        allowance_id = _domain_hash(
            "operatormark-permission-authority-allowance-v1",
            {
                "state4_authorization_file_sha256": grant.state4_authorization_file_sha256,
                "request_self_sha256": grant.expected_request.request_self_sha256,
                "allowance_kind": allowance_kind,
            },
        )
        reservation_payload = {
            "allowance_id": allowance_id,
            "allowance_kind": allowance_kind,
            "session_id": session_id,
            "state4_authorization_file_sha256": grant.state4_authorization_file_sha256,
            "request_self_sha256": grant.expected_request.request_self_sha256,
            "operation": operation,
            "scope_self_sha256": grant.expected_scope.scope_self_sha256,
            "grant_binding_sha256": permission_authority.permission_approval_grant_binding_sha256(grant),
        }
        reservation_id = _domain_hash(
            "operatormark-permission-authority-operation-reservation-v1",
            reservation_payload,
        )
        parent = plan.journal_binding.attempt_relative_path
        result[operation] = (
            f"{parent}/reservation-{allowance_id}.json",
            f"{parent}/terminal-{reservation_id}.json",
        )
    return result


def derive_expected_journal_slots(*, plan: HandoffPlan) -> tuple[InputSlot, ...]:
    validate_handoff_plan(plan)
    expected_paths = [f"{plan.journal_binding.attempt_relative_path}/activation.json"]
    for reservation_path, terminal_path in _journal_addresses(plan).values():
        expected_paths.extend((reservation_path, terminal_path))
    slots = tuple(slot for slot in plan.input_slots if slot.role.startswith("journal_"))
    if tuple(slot.relative_path for slot in slots) != tuple(expected_paths):
        _raise("planned journal slots differ from independently derived addresses/order")
    if slots[0].role != "journal_activation" or any(
        slot.role != ("journal_reservation" if index % 2 else "journal_terminal")
        for index, slot in enumerate(slots[1:], start=1)
    ):
        _raise("planned journal slot roles are invalid")
    if any(slot.root_binding_id != plan.journal_binding.root_binding_id for slot in slots):
        _raise("planned journal slot escaped its pinned root")
    return slots


def _result_for_unreached_read(
    slot: InputSlot,
    *,
    accounting: HandoffAuthorityAccounting,
    failure: HandoffFailure,
) -> HandoffReadResult:
    status = "not_found" if failure.reason == "not_found" else "unavailable"
    return HandoffReadResult(
        slot.slot_id, status, None, "unavailable", None, None,
        accounting, (), failure,
    )


def _check_result(value: object, *, name: str) -> HandoffCheckpointResult:
    if type(value) is not HandoffCheckpointResult:
        _raise(f"{name} returned a malformed checkpoint result")
    _decode_dataclass(HandoffCheckpointResult, _wire(value), name=name)
    if value.status == "active":
        if value.reason is not None:
            _raise(f"{name} active result has a reason")
    elif value.reason not in {"refused", "expired", "scope_mismatch", "unavailable", "malformed_output"}:
        _raise(f"{name} inactive result lacks a closed reason")
    return value


def _finish_result(value: object, *, requested: str) -> HandoffFinishResult:
    if type(value) is not HandoffFinishResult:
        _raise("authority returned a malformed finish result")
    _decode_dataclass(HandoffFinishResult, _wire(value), name="authority finish")
    if value.effect_certainty == "confirmed":
        if value.status != "succeeded" or value.recorded_outcome != requested:
            _raise("confirmed authority finish failed or recorded the wrong outcome")
    elif value.recorded_outcome is not None:
        _raise("unconfirmed authority finish invented a recorded outcome")
    return value


def _acquire_return_valid(value: object) -> bool:
    if type(value) is not HandoffAcquireResult:
        return False
    if value.status == "granted":
        return value.lease is not None and value.reason is None
    return (value.status in {"refused", "failed"} and value.lease is None
            and value.reason in {"refused", "expired", "scope_mismatch", "unavailable", "malformed_output"})


def _malformed_acquire_accounting(value: object, authority: PermissionHandoffAuthority) -> HandoffAuthorityAccounting:
    accounting = replace(_empty_accounting(), acquire_status="failed")
    if type(value) is not HandoffAcquireResult or value.lease is None:
        return accounting
    lease = value.lease
    try:
        established = type(lease.scope) is HandoffLeaseScope
    except Exception:
        established = False
    if not established:
        return replace(accounting, lease_returned=True, lease_binding_status="invalid")
    try:
        finished = _finish_result(authority.finish(lease, "failed"), requested="failed")
    except Exception:
        finished = HandoffFinishResult("failed", "unknown", None)
    return replace(accounting, lease_returned=True, lease_binding_status="invalid",
                   finish_attempted=True, requested_outcome="failed", finish_status=finished.status,
                   effect_certainty=finished.effect_certainty, recorded_outcome=finished.recorded_outcome)


def _acquire_one(
    *,
    plan: HandoffPlan,
    slot: InputSlot,
    authority: PermissionHandoffAuthority,
    reader: PermissionHandoffReader,
) -> HandoffReadResult:
    operation: Literal["read_handoff_input"] = "read_handoff_input"
    try:
        acquired = authority.acquire(plan, operation, slot.slot_id)
    except Exception as exc:
        accounting = replace(_empty_accounting(), acquire_status="failed")
        return _result_for_unreached_read(
            slot, accounting=accounting,
            failure=_exception_failure(stage="acquire", operation=operation, reason="raised"),
        )
    if not _acquire_return_valid(acquired):
        accounting = _malformed_acquire_accounting(acquired, authority)
        return _result_for_unreached_read(
            slot, accounting=accounting,
            failure=_exception_failure(stage="acquire", operation=operation),
        )
    if acquired.status != "granted":
        status = acquired.status if acquired.status in {"refused", "failed"} else "failed"
        accounting = replace(_empty_accounting(), acquire_status=status)
        return _result_for_unreached_read(
            slot, accounting=accounting,
            failure=_exception_failure(
                stage="acquire", operation=operation,
                reason="refused" if status == "refused" else "unavailable",
            ),
        )
    lease = acquired.lease
    accounting = replace(
        _empty_accounting(), acquire_status="granted", lease_returned=lease is not None,
        lease_binding_status="invalid",
    )
    if lease is None:
        return _result_for_unreached_read(
            slot, accounting=accounting,
            failure=_exception_failure(stage="lease_binding", operation=operation),
        )
    binding_valid = True
    try:
        _validate_lease_scope(
            lease, plan=plan, operation=operation, slot_id=slot.slot_id,
            root_binding_id=slot.root_binding_id, role=slot.role,
        )
    except Exception:
        binding_valid = False
    accounting = replace(accounting, lease_binding_status="valid" if binding_valid else "invalid")
    checkpoints: list[HandoffCheckpoint] = []
    raw: HandoffReadResult | None = None
    facts_valid = False
    requested: Literal["succeeded", "refused", "failed"] = "failed"
    failure: HandoffFailure | None = None
    if binding_valid:
        for name in ("initial", "before_open", "before_bytes"):
            try:
                checked = _check_result(authority.check_active(lease, name), name=name)
            except Exception:
                checked = HandoffCheckpointResult("failed", "malformed_output")
            checkpoints.append(HandoffCheckpoint(cast(object, name), checked.status, checked.reason))
            if checked.status != "active":
                requested = "refused" if checked.status == "refused" else "failed"
                failure = _exception_failure(
                    stage="open" if name == "before_open" else "read",
                    operation=operation,
                    reason="refused" if checked.status == "refused" else "unavailable",
                )
                break
        else:
            try:
                candidate = reader.read_existing(plan, slot, lease)
                if type(candidate) is not HandoffReadResult:
                    raise TypeError("reader result has wrong exact type")
                raw = candidate
            except Exception:
                failure = _exception_failure(stage="read", operation=operation, reason="raised")
                requested = "failed"
            else:
                try:
                    _validate_read_facts(raw, plan=plan, slot=slot)
                except Exception:
                    failure = _exception_failure(stage="read", operation=operation)
                else:
                    facts_valid = True
                    requested = "succeeded" if raw.status == "complete" else (
                        "refused" if raw.failure is not None and raw.failure.reason == "refused" else "failed"
                    )
            try:
                checked = _check_result(authority.check_active(lease, "final"), name="final")
            except Exception:
                checked = HandoffCheckpointResult("failed", "malformed_output")
            checkpoints.append(HandoffCheckpoint("final", checked.status, checked.reason))
            if checked.status != "active":
                # Invalid provider facts stay failed even if the final authority
                # checkpoint also refuses. Neither event licenses a retry.
                if raw is None or facts_valid:
                    requested = "refused" if checked.status == "refused" else "failed"
                if failure is None:
                    failure = _exception_failure(stage="finish", operation=operation, reason="terminal_unconfirmed")
    else:
        failure = _exception_failure(stage="lease_binding", operation=operation, reason="mismatch")
    try:
        finished = _finish_result(authority.finish(lease, requested), requested=requested)
    except Exception:
        finished = HandoffFinishResult("failed", "unknown", None)
    accounting = replace(
        accounting,
        checkpoints=tuple(checkpoints),
        finish_attempted=True,
        requested_outcome=requested,
        finish_status=finished.status,
        effect_certainty=finished.effect_certainty,
        recorded_outcome=finished.recorded_outcome,
    )
    if raw is None:
        return _result_for_unreached_read(
            slot, accounting=accounting,
            failure=failure or _exception_failure(stage="read", operation=operation),
        )
    result = replace(raw, authority_accounting=accounting)
    if not facts_valid:
        # Keep reached bytes/metadata/cleanup as unsupported local evidence;
        # do not turn malformed provider claims into a serializable success.
        return replace(result, status="inconsistent", failure=failure)
    if (
        requested != "succeeded"
        or finished.status != "succeeded"
        or finished.effect_certainty != "confirmed"
        or finished.recorded_outcome != requested
    ) and result.status == "complete":
        result = replace(
            result,
            status="inconsistent",
            failure=failure or _exception_failure(stage="finish", operation=operation, reason="terminal_unconfirmed"),
        )
    try:
        validate_handoff_read_result(result, plan=plan, slot=slot)
    except Exception:
        return replace(
            result,
            status="inconsistent",
            failure=_exception_failure(stage="read", operation=operation),
        )
    return result


def acquire_handoff_inputs(
    *,
    plan: HandoffPlan,
    original_outcome: object,
    authority: PermissionHandoffAuthority,
    reader: PermissionHandoffReader,
) -> tuple[HandoffReadResult, ...]:
    validate_handoff_plan(plan)
    # The candidate outcome is deliberately not consulted for addresses.  It is
    # accepted only to keep the caller's chronology explicit.
    if type(original_outcome) is not dict:
        _raise("original_outcome must be a decoded handoff outcome record")
    return tuple(
        _acquire_one(plan=plan, slot=slot, authority=authority, reader=reader)
        for slot in plan.input_slots
    )


def project_store_snapshot(
    *, plan: HandoffPlan, slot: InputSlot, read_result: HandoffReadResult
) -> evidence.StoreObjectSnapshot:
    validate_handoff_read_result(read_result, plan=plan, slot=slot)
    if slot.role not in {
        "base_primary", "base_failure_sidecar", "authz_primary", "authz_failure_sidecar"
    } or read_result.status != "complete" or read_result.data is None or read_result.after is None:
        _raise("only a complete independently measured private-store read can become a snapshot")
    metadata = read_result.after
    data = _validate_byte_record(read_result.data, name="snapshot source bytes")
    published_path = {
        "base_primary": plan.expected_base_destinations[0].primary_accepted_relative_path,
        "base_failure_sidecar": plan.expected_base_destinations[1].retention_failure_accepted_relative_path,
        "authz_primary": plan.expected_authz_destination.primary_accepted_relative_path,
        "authz_failure_sidecar": plan.expected_authz_destination.retention_failure_accepted_relative_path,
    }[slot.role]
    if slot.relative_path != published_path:
        _raise("private-store slot path differs from the independent destination")
    return evidence.StoreObjectSnapshot(
        relative_path=metadata.relative_path,
        namespace_object_identity=_native_identity_text(metadata.namespace_identity),
        parent_relative_path=plan.expected_store.records_parent_relative_path,
        parent_object_identity=_native_identity_text(metadata.parent_identity),
        object_identity=_native_identity_text(metadata.identity),
        volume_identifier_commitment=metadata.volume_identifier_commitment,
        size_bytes=len(data),
        file_sha256=sha256_bytes(data),
        security_descriptor_sha256=metadata.security_descriptor_sha256,
        confidential_store_profile_sha256=cast(str, metadata.confidential_profile_file_sha256),
        publication_state="published",
    )


def _decode_independent_activation(
    *, plan: HandoffPlan, read_result: HandoffReadResult
) -> tuple[permission_authority.PermissionAttemptActivation, str]:
    slots = derive_expected_journal_slots(plan=plan)
    activation_slot = slots[0]
    validate_handoff_read_result(read_result, plan=plan, slot=activation_slot)
    if read_result.status != "complete" or read_result.data is None:
        _raise("independent activation read is unavailable")
    data = _validate_byte_record(read_result.data, name="activation bytes")
    try:
        payload = strict_canonical_json_loads(data)
        activation = _decode_dataclass(
            permission_authority.PermissionAttemptActivation,
            payload,
            name="independent activation",
        )
        permission_authority._validate_attempt_activation(activation)
    except Exception as exc:
        _raise("independent activation is malformed", exc)
    if activation != plan.journal_binding.expected_activation:
        _raise("independent activation differs from the prepared plan")
    return activation, sha256_bytes(
        b"operatormark-permission-authority-filesystem-session-v1\n" + data
    )


def _journal_record_evidence(result: HandoffReadResult | None) -> authz_verifier.IndependentJournalRecordEvidence:
    if result is None or result.status != "complete" or result.data is None:
        return authz_verifier.IndependentJournalRecordEvidence("unavailable", None, None, None)
    data = _validate_byte_record(result.data, name="journal record bytes")
    return authz_verifier.IndependentJournalRecordEvidence(
        "available", data.hex(), len(data), sha256_bytes(data)
    )


def build_independent_authz_inputs(
    *,
    expected_plan: HandoffPlan,
    original_outcome: object,
    ordered_reads: tuple[HandoffReadResult, ...],
) -> PreparedAuthzVerifierInputs:
    validate_handoff_plan(expected_plan)
    if type(original_outcome) is not dict:
        _raise("original outcome must be a decoded handoff record")
    if type(ordered_reads) is not tuple or tuple(item.slot_id for item in ordered_reads) != tuple(slot.slot_id for slot in expected_plan.input_slots):
        _raise("ordered reads do not match the independent plan")
    by_role: dict[str, list[tuple[InputSlot, HandoffReadResult]]] = {}
    by_path: dict[str, HandoffReadResult] = {}
    incomplete: list[str] = []
    for slot, result in zip(expected_plan.input_slots, ordered_reads, strict=True):
        validate_handoff_read_result(result, plan=expected_plan, slot=slot)
        by_role.setdefault(slot.role, []).append((slot, result))
        by_path[slot.relative_path] = result
        if result.status not in {"complete", "not_found"} or (
            result.status == "not_found" and slot.role == "authority_input"
        ):
            incomplete.append(f"read:{slot.slot_id}")
        if result.cleanup_failures:
            incomplete.append(f"cleanup:{slot.slot_id}")

    def complete_private(role: str) -> tuple[bytes | None, evidence.StoreObjectSnapshot | None]:
        entries = by_role.get(role, [])
        if len(entries) != 1:
            return None, None
        slot, result = entries[0]
        if result.status != "complete" or result.data is None:
            return None, None
        return (
            _validate_byte_record(result.data, name=f"{role} bytes"),
            project_store_snapshot(plan=expected_plan, slot=slot, read_result=result),
        )

    primary_bytes, primary_snapshot = complete_private("authz_primary")
    sidecar_bytes, sidecar_snapshot = complete_private("authz_failure_sidecar")
    base_evidence: authz_verifier.IndependentBaseEvidence
    base_bytes, base_snapshot = complete_private("base_primary")
    if base_bytes is not None and base_snapshot is not None:
        # Captured and failed base records share one physical primary slot.
        # Select between the two independent destinations only after strict
        # schema discrimination; never treat the retention sidecar as primary.
        payload = strict_canonical_json_loads(base_bytes.rstrip(b"\n"))
        schema = payload.get("schema")
        if schema == evidence.OBSERVATION_SCHEMA:
            base_destination = expected_plan.expected_base_destinations[0]
        elif schema == evidence.OBSERVATION_FAILURE_SCHEMA:
            base_destination = expected_plan.expected_base_destinations[1]
        else:
            _raise("shared base primary has an unsupported schema")
        base_evidence = authz_verifier.IndependentBaseEvidence(
            "available", base_destination, len(base_bytes),
            sha256_bytes(base_bytes), base_snapshot, None,
        )
    else:
        base_evidence = authz_verifier.IndependentBaseEvidence(
            "unavailable", None, None, None, None, "unavailable"
        )
        incomplete.append("base_evidence_unavailable")

    expected_token_identity: supplied_authz.SuppliedTokenIdentity | None = None
    trace_value = original_outcome.get("token_trace")
    try:
        trace = _validate_token_trace(trace_value)
        statistics = tuple(
            event.statistics
            for event in trace.events
            if event.method == "query_token_statistics"
            and event.outcome == "returned_valid"
            and event.statistics is not None
        )
        if statistics and all(item == statistics[0] for item in statistics):
            first = statistics[0]
            expected_token_identity = supplied_authz.SuppliedTokenIdentity(
                first.token_id_luid_hex,
                first.authentication_id_luid_hex,
                first.modified_id_luid_hex,
                first.token_type,
                first.impersonation_level,
            )
        elif statistics:
            _raise("independent token statistics disagree")
        if trace.capture_status != "complete" or trace.same_borrowed_token != "confirmed":
            incomplete.append("token_witness_unavailable")
    except Exception as exc:
        _raise("supplied token witness is malformed or contradictory", exc)

    unavailable_record = authz_verifier.IndependentJournalRecordEvidence(
        "unavailable", None, None, None
    )
    session_id: str | None = None
    activation_entries = by_role.get("journal_activation", [])
    if len(activation_entries) == 1 and activation_entries[0][1].status == "complete":
        _, session_id = _decode_independent_activation(
            plan=expected_plan, read_result=activation_entries[0][1]
        )
        measured_parent = activation_entries[0][1].after.parent_identity
        for slot, read in zip(expected_plan.input_slots, ordered_reads, strict=True):
            if slot.role.startswith("journal_"):
                for metadata in (read.before, read.after):
                    if metadata is not None and metadata.parent_identity != measured_parent:
                        _raise("journal reads do not bind one measured attempt directory")
    else:
        incomplete.append("journal_evidence_unavailable")
    addresses = _journal_addresses(expected_plan)

    def operation_evidence(operation: str) -> authz_verifier.IndependentJournalOperationEvidence:
        grant = expected_plan.expected_grants[operation]
        if session_id is None:
            return authz_verifier.IndependentJournalOperationEvidence(
                grant, None, unavailable_record, unavailable_record
            )
        reservation_path, terminal_path = addresses[operation]
        return authz_verifier.IndependentJournalOperationEvidence(
            grant,
            session_id,
            _journal_record_evidence(by_path.get(reservation_path)),
            _journal_record_evidence(by_path.get(terminal_path)),
        )

    authority_evidence = authz_verifier.IndependentAuthorityEvidence(
        parent=operation_evidence("observe"),
        child=operation_evidence("observe_authz"),
    )
    return PreparedAuthzVerifierInputs(
        primary_bytes,
        primary_snapshot,
        sidecar_bytes,
        sidecar_snapshot,
        base_evidence,
        authority_evidence,
        expected_token_identity,
        tuple(dict.fromkeys(incomplete)),
    )
