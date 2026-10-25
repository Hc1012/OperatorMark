"""Pure restart verification for retained permission-handoff evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast

from .canonical import canonical_json_bytes, sha256_bytes, strict_canonical_json_loads
from . import permission_authz_evidence_verifier as authz_verifier
from . import permission_authority
from . import private_permission_evidence as evidence
from .permission_handoff_records import (
    AcceptedHandoffAnchor,
    FileRef,
    HandoffPhysicalFile,
    HandoffContentCheck,
    HandoffPlan,
    HandoffReadResult,
    HandoffSourceFile,
    RECORD_SCHEMAS,
    VERIFICATION_REASON_ORDER,
    WITNESS_FILENAMES,
    _decode_dataclass,
    _decode_original_runtime,
    _validate_token_trace,
    _wire,
    canonical_handoff_record_bytes,
    derive_handoff_invocation_id,
    handoff_record_commitment,
    strict_handoff_record_loads,
    validate_handoff_plan,
)
from .permission_handoff_sizing import validate_handoff_physical_inventory
from .permission_handoff_sources import (
    _journal_addresses,
    build_independent_authz_inputs,
    derive_expected_journal_slots,
)


HANDOFF_VERIFICATION_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_handoff_verification_result.v1"
)


class PermissionHandoffVerificationError(ValueError):
    """Supplied historical handoff evidence is malformed or contradictory."""


@dataclass(frozen=True, slots=True)
class HandoffVerification:
    schema: str
    invocation_id: str
    content_verification: authz_verifier.RetainedAuthzEvidenceVerification | None
    original_retention: Literal["confirmed_success", "confirmed_failure", "unknown"]
    anchor_status: Literal["not_checked", "unavailable", "accepted"]
    overall_status: Literal["corroborated", "incomplete", "unanchored"]
    reason_codes: tuple[str, ...]
    permission_acceptance_emitted: Literal[False] = False


def _raise(message: str, cause: BaseException | None = None) -> None:
    error = PermissionHandoffVerificationError(message)
    if cause is None:
        raise error
    raise error from cause


def _ordered_reasons(values: list[str]) -> tuple[str, ...]:
    unknown = set(values) - set(VERIFICATION_REASON_ORDER)
    if unknown:
        _raise(f"internal verification reason vocabulary escaped: {sorted(unknown)}")
    return tuple(code for code in VERIFICATION_REASON_ORDER if code in values)


def _same_file_ref(value: object, *, data: bytes, expected_path: str, root: str) -> None:
    ref = _decode_dataclass(FileRef, value, name="predecessor file reference")
    if (
        ref.root_binding_id != root
        or ref.relative_path != expected_path
        or ref.file_bytes != len(data)
        or ref.file_sha256 != sha256_bytes(data)
    ):
        _raise("predecessor file reference mismatch")


def _verify_original_return(
    outcome: dict[str, object], *, expected_plan: HandoffPlan
) -> tuple[
    Literal["consistent_success_return", "consistent_failure_return", "unavailable"],
    dict[str, object] | None,
]:
    call_outcome = outcome["call_outcome"]
    if call_outcome != "returned_valid":
        return "unavailable", None
    try:
        runtime = _decode_original_runtime(
            outcome["runtime_projection"], expected_plan=expected_plan
        )
    except Exception as exc:
        _raise("original runtime projection is contradictory", exc)
    status = runtime["persistence_status"]
    if status == "retained":
        return "consistent_success_return", runtime
    if status in {"not_attempted", "not_retained"}:
        return "consistent_failure_return", runtime
    return "unavailable", runtime


def _verify_native_token_trace(
    outcome: dict[str, object], *, require_complete: bool
) -> bool:
    try:
        trace = _validate_token_trace(outcome["token_trace"])
    except Exception as exc:
        _raise("native token trace is malformed", exc)
    if trace.capture_status != "complete" or trace.same_borrowed_token != "confirmed":
        if "capture_failed" not in trace.reason_codes and trace.events:
            # Walk reached evaluator stages and required cleanup, including a
            # typed failed call. No synthetic query or invented cycle boundary.
            allowed = {"query_token_statistics"}
            manager = context = False
            before_query = True
            cycles = 0
            for event in trace.events:
                method = event.method
                if method not in allowed:
                    _raise("partial native trace violates the evaluator/cleanup order")
                valid = event.outcome == "returned_valid"
                if method == "query_token_statistics" and before_query:
                    cycles += 1
                    if cycles > 2:
                        _raise("partial native trace exceeds the two-object invocation")
                    allowed = {"initialize_resource_manager"} if valid else set()
                    before_query = False
                elif method == "initialize_resource_manager":
                    manager = valid
                    allowed = {"initialize_context_from_token"} if valid else set()
                elif method == "initialize_context_from_token":
                    context = valid
                    allowed = {"access_check"} if valid else {"free_resource_manager"}
                elif method == "access_check":
                    allowed = {"query_token_statistics", "free_context"} if valid else {"free_context"}
                elif method == "query_token_statistics":
                    allowed = {"free_context"}
                elif method == "free_context":
                    context = False
                    allowed = {"free_resource_manager"}
                else:
                    manager = False
                    before_query = True
                    allowed = {"query_token_statistics"}
            if manager or context:
                _raise("partial native trace omits cleanup for a reached resource")
        return False
    methods = tuple(event.method for event in trace.events)
    cycle = (
        "query_token_statistics",
        "initialize_resource_manager",
        "initialize_context_from_token",
        "access_check",
        "query_token_statistics",
        "free_context",
        "free_resource_manager",
    )
    if len(methods) != 2 * len(cycle) or any(
        methods[index : index + len(cycle)] != cycle
        for index in range(0, len(methods), len(cycle))
    ):
        if require_complete:
            _raise("complete native token trace has the wrong evaluator call order")
        return False
    statistics = tuple(
        event.statistics for event in trace.events
        if event.method == "query_token_statistics"
    )
    if not statistics or any(item is None or item != statistics[0] for item in statistics):
        _raise("native token trace identities disagree")
    return True


def _decode_reads(
    acquisition: dict[str, object], *, expected_plan: HandoffPlan
) -> tuple[HandoffReadResult, ...]:
    raw = acquisition["ordered_reads"]
    if type(raw) is not list:
        _raise("acquisition ordered_reads has the wrong type")
    return tuple(
        _decode_dataclass(HandoffReadResult, item, name=f"ordered_reads[{index}]")
        for index, item in enumerate(cast(list[object], raw))
    )


def _call_current_verifier(
    *,
    plan: HandoffPlan,
    prepared: object,
) -> authz_verifier.RetainedAuthzEvidenceVerification | None:
    inputs = prepared
    primary_result = None
    sidecar_result = None
    try:
        if inputs.primary_bytes is not None:
            if inputs.primary_snapshot is None:
                _raise("primary bytes were not paired with a measured snapshot")
            primary_result = authz_verifier.verify_retained_permission_authz_evidence(
                inputs.primary_bytes,
                expected_request=plan.expected_request,
                expected_access=plan.expected_access,
                expected_binding=plan.expected_authz_binding,
                expected_destination=plan.expected_authz_destination,
                expected_store=plan.expected_store,
                expected_authz_snapshot=inputs.primary_snapshot,
                base_evidence=inputs.base_evidence,
                authority_evidence=inputs.authority_evidence,
                expected_token_identity=inputs.expected_token_identity,
            )
        if inputs.sidecar_bytes is not None:
            if inputs.sidecar_snapshot is None:
                _raise("sidecar bytes were not paired with a measured snapshot")
            sidecar_result = authz_verifier.verify_retained_permission_authz_retention_failure(
                inputs.sidecar_bytes,
                expected_request=plan.expected_request,
                expected_access=plan.expected_access,
                expected_binding=plan.expected_authz_binding,
                expected_destination=plan.expected_authz_destination,
                expected_store=plan.expected_store,
                expected_sidecar_snapshot=inputs.sidecar_snapshot,
                base_evidence=inputs.base_evidence,
                authority_evidence=inputs.authority_evidence,
                expected_token_identity=inputs.expected_token_identity,
            )
    except PermissionHandoffVerificationError:
        raise
    except Exception as exc:
        _raise("current retained Authz verifier rejected the independent inputs", exc)
    return primary_result if primary_result is not None else sidecar_result


def _read_bytes_by_path(
    plan: HandoffPlan, reads: tuple[HandoffReadResult, ...]
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for slot, read in zip(plan.input_slots, reads, strict=True):
        if read.status == "complete" and read.data is not None:
            data = bytes.fromhex(read.data.record_hex)
            if len(data) != read.data.file_bytes or sha256_bytes(data) != read.data.file_sha256:
                _raise("complete independent read has contradictory bytes")
            result[slot.relative_path] = data
    return result


def _verify_one_original_terminal(
    *,
    plan: HandoffPlan,
    operation: str,
    paths: dict[str, bytes],
    expected_outcome: str,
) -> bool:
    reservation_path, terminal_path = _journal_addresses(plan)[operation]
    if reservation_path not in paths or terminal_path not in paths:
        return False
    activation_path = f"{plan.journal_binding.attempt_relative_path}/activation.json"
    if activation_path not in paths:
        return False
    activation_bytes = canonical_json_bytes(asdict(plan.journal_binding.expected_activation))
    if paths[activation_path] != activation_bytes:
        _raise("retention journal activation differs from the independently pinned activation")
    session_id = sha256_bytes(b"operatormark-permission-authority-filesystem-session-v1\n" + activation_bytes)
    try:
        reservation_payload = strict_canonical_json_loads(paths[reservation_path])
        reservation = _decode_dataclass(
            permission_authority.PermissionOperationReservation,
            reservation_payload,
            name=f"{operation} reservation",
        )
        permission_authority._validate_operation_reservation(reservation)
        terminal_payload = strict_canonical_json_loads(paths[terminal_path])
        terminal = _decode_dataclass(
            permission_authority.PermissionTerminalRecord,
            terminal_payload,
            name=f"{operation} terminal",
        )
        permission_authority._validate_terminal_record(terminal)
    except Exception as exc:
        _raise(f"{operation} retention journal record is invalid", exc)
    grant = plan.expected_grants[operation]
    if (
        reservation.operation != operation
        or reservation.session_id != session_id
        or reservation_path.rsplit("/", 1)[-1] != f"reservation-{reservation.allowance_id}.json"
        or terminal_path.rsplit("/", 1)[-1] != f"terminal-{reservation.reservation_id}.json"
        or reservation.request_self_sha256 != plan.expected_request.request_self_sha256
        or reservation.state4_authorization_file_sha256
        != plan.authority_binding.full_state4_authorization_file_sha256
        or reservation.scope_self_sha256 != grant.expected_scope.scope_self_sha256
        or reservation.grant_binding_sha256
        != permission_authority.permission_approval_grant_binding_sha256(grant)
        or terminal.reservation_id != reservation.reservation_id
        or terminal.session_id != reservation.session_id
        or terminal.requested_outcome != expected_outcome
    ):
        _raise(f"{operation} retention journal correspondence mismatch")
    for name in ("allowance_id", "state4_authorization_file_sha256", "request_self_sha256",
                 "operation", "scope_self_sha256"):
        if getattr(terminal, name) != getattr(reservation, name):
            _raise(f"{operation} terminal {name} differs from its reservation")
    return True


def _verify_retention_journal(
    *,
    plan: HandoffPlan,
    runtime: dict[str, object] | None,
    reads: tuple[HandoffReadResult, ...],
) -> bool:
    if runtime is None:
        return False
    try:
        derive_expected_journal_slots(plan=plan)
    except Exception as exc:
        _raise("journal slots are not independently derived", exc)
    paths = _read_bytes_by_path(plan, reads)
    status = runtime["persistence_status"]
    result = runtime["retention_result"]
    if status == "retained":
        if type(result) is not evidence.RetainedPermissionAuthzEvidence:
            _raise("retained original runtime lost its receipt")
        return _verify_one_original_terminal(
            plan=plan,
            operation="retain_authz_evidence",
            paths=paths,
            expected_outcome="succeeded",
        )
    if status == "not_retained":
        if type(result) is not evidence.PermissionAuthzEvidenceRetentionFailure:
            _raise("not-retained original runtime lost its failure result")
        accounting = result.main_authority_accounting
        if (
            not accounting.finish_attempted
            or accounting.effect_certainty != "confirmed"
            or accounting.recorded_outcome != accounting.requested_outcome
            or accounting.requested_outcome is None
        ):
            return False
        if not _verify_one_original_terminal(
            plan=plan,
            operation="retain_authz_evidence",
            paths=paths,
            expected_outcome=accounting.requested_outcome,
        ):
            return False
        if result.failure_record_attempted:
            sidecar = result.failure_record_authority_accounting
            if (
                sidecar is None
                or not sidecar.finish_attempted
                or sidecar.effect_certainty != "confirmed"
                or sidecar.recorded_outcome != sidecar.requested_outcome
                or sidecar.requested_outcome is None
            ):
                return False
            return _verify_one_original_terminal(
                plan=plan,
                operation="retain_authz_retention_failure",
                paths=paths,
                expected_outcome=sidecar.requested_outcome,
            )
        return True
    if status == "not_attempted":
        return True
    return False


def verify_permission_handoff_contents(
    *,
    expected_plan: HandoffPlan,
    expected_source_files: tuple[HandoffSourceFile, ...],
    start_bytes: bytes,
    outcome_bytes: bytes,
    acquisition_bytes: bytes,
) -> HandoffContentCheck:
    try:
        validate_handoff_plan(expected_plan)
    except Exception as exc:
        _raise("independent handoff plan is invalid", exc)
    if type(expected_source_files) is not tuple or expected_source_files != expected_plan.source_files:
        _raise("independently supplied source identities differ from the plan")
    try:
        start = strict_handoff_record_loads(
            start_bytes, kind="start", expected_plan=expected_plan
        )
        outcome = strict_handoff_record_loads(
            outcome_bytes, kind="outcome", expected_plan=expected_plan
        )
        acquisition = strict_handoff_record_loads(
            acquisition_bytes, kind="acquisition", expected_plan=expected_plan
        )
    except Exception as exc:
        _raise("handoff predecessor record is invalid", exc)
    root = expected_plan.witness_store.root_binding_id
    parent = expected_plan.witness_store.parent_relative_path
    _same_file_ref(
        outcome["start_file_ref"], data=start_bytes,
        expected_path=f"{parent}/{WITNESS_FILENAMES[0]}", root=root,
    )
    _same_file_ref(
        acquisition["outcome_file_ref"], data=outcome_bytes,
        expected_path=f"{parent}/{WITNESS_FILENAMES[1]}", root=root,
    )
    consistency, runtime = _verify_original_return(outcome, expected_plan=expected_plan)
    reads = _decode_reads(acquisition, expected_plan=expected_plan)
    try:
        prepared = build_independent_authz_inputs(
            expected_plan=expected_plan,
            original_outcome=outcome,
            ordered_reads=reads,
        )
    except Exception as exc:
        _raise("independent Authz input projection failed", exc)
    token_complete = _verify_native_token_trace(
        outcome, require_complete=consistency != "unavailable"
    )
    content = _call_current_verifier(plan=expected_plan, prepared=prepared)
    if runtime is not None and runtime["intended_authz_record"] is not None and prepared.primary_bytes is not None:
        intended = runtime["intended_authz_record"]
        if prepared.primary_bytes != bytes.fromhex(intended.record_hex):
            _raise("independently read primary differs from the preserved original intended bytes")
    if runtime is not None and runtime["retention_result_kind"] == "success_receipt" and prepared.primary_snapshot is not None:
        receipt = runtime["retention_result"]
        snapshot = prepared.primary_snapshot
        if (receipt.store_object_identity != snapshot.object_identity
                or receipt.file_bytes != snapshot.size_bytes
                or receipt.file_sha256 != snapshot.file_sha256):
            _raise("original receipt contradicts the independently reopened primary object")
    if runtime is not None and runtime["retention_result_kind"] == "retention_failure" and prepared.sidecar_snapshot is not None:
        failure = runtime["retention_result"]
        if failure.retained_failure_record_file_sha256 is not None:
            snapshot = prepared.sidecar_snapshot
            if (failure.retained_failure_record_relative_path != snapshot.relative_path
                    or failure.retained_failure_record_file_bytes != snapshot.size_bytes
                    or failure.retained_failure_record_file_sha256 != snapshot.file_sha256):
                _raise("original failure result contradicts the independently read retained sidecar")
    reasons: list[str] = []
    if consistency == "unavailable":
        reasons.append("original_return_unavailable")
    if not token_complete or prepared.expected_token_identity is None:
        reasons.append("token_witness_unavailable")
    if prepared.incomplete_reasons:
        reasons.append("input_read_incomplete")
    if any(read.cleanup_failures for read in reads):
        reasons.append("input_cleanup_failed")
    if prepared.base_evidence.status != "available":
        reasons.append("base_evidence_unavailable")
    journal_complete = _verify_retention_journal(
        plan=expected_plan, runtime=runtime, reads=reads
    )
    if not journal_complete or (content is not None and content.authority_correspondence_status != "corroborated"):
        reasons.append("journal_evidence_unavailable")
        if runtime is not None and runtime["persistence_status"] in {"retained", "not_retained", "retention_unknown"}:
            reasons.append("original_terminal_unconfirmed")
    if (content is not None and content.payload_kind == "authz_retention_failure") or (
        runtime is not None and runtime["retention_result_kind"] == "retention_failure"
    ):
        reasons.append("retention_failure_evidence")
    if content is None or content.assessment_status != "completed":
        reasons.append("content_verification_incomplete")
    return HandoffContentCheck(content, consistency, _ordered_reasons(reasons))


def _verify_exact_inventory(
    *,
    plan: HandoffPlan,
    index: dict[str, object],
    predecessor_bytes: tuple[bytes, bytes, bytes, bytes],
    acquisition: dict[str, object],
) -> None:
    refs = index["ordered_witness_file_refs"]
    if type(refs) is not list or len(refs) != 4:
        _raise("index witness inventory is not exact")
    for item, filename, data in zip(
        cast(list[object], refs), WITNESS_FILENAMES[:4], predecessor_bytes, strict=True
    ):
        _same_file_ref(
            item,
            data=data,
            expected_path=f"{plan.witness_store.parent_relative_path}/{filename}",
            root=plan.witness_store.root_binding_id,
        )
    reads = _decode_reads(acquisition, expected_plan=plan)
    expected_evidence = tuple(
        FileRef(slot.root_binding_id, slot.relative_path, result.data.file_bytes, result.data.file_sha256)
        for slot, result in zip(plan.input_slots, reads, strict=True)
        if result.status == "complete" and result.data is not None
    )
    raw_evidence = index["ordered_evidence_file_refs"]
    if type(raw_evidence) is not list:
        _raise("index evidence references have the wrong type")
    actual_evidence = tuple(
        _decode_dataclass(FileRef, item, name=f"index evidence ref {i}")
        for i, item in enumerate(cast(list[object], raw_evidence))
    )
    if actual_evidence != expected_evidence:
        _raise("index selected evidence inventory mismatch")
    raw_inventory = index["physical_inventory"]
    if type(raw_inventory) is not list:
        _raise("index physical inventory has the wrong type")
    inventory = tuple(
        _decode_dataclass(HandoffPhysicalFile, item, name=f"physical inventory {i}")
        for i, item in enumerate(cast(list[object], raw_inventory))
    )
    try:
        validate_handoff_physical_inventory(inventory)
    except Exception as exc:
        _raise("index physical inventory is invalid", exc)
    forbidden = {
        f"{plan.witness_store.parent_relative_path}/{WITNESS_FILENAMES[4]}",
        f"{plan.witness_store.parent_relative_path}/{WITNESS_FILENAMES[5]}",
    }
    if any(item.relative_path in forbidden for item in inventory):
        _raise("index sizes its own or receipt's future bytes")
    expected_refs = tuple(
        FileRef(plan.witness_store.root_binding_id,
                f"{plan.witness_store.parent_relative_path}/{filename}", len(data), sha256_bytes(data))
        for filename, data in zip(WITNESS_FILENAMES[:4], predecessor_bytes, strict=True)
    ) + expected_evidence
    if len(inventory) != len(expected_refs):
        _raise("physical inventory omits or invents a selected physical instance")
    for physical, ref in zip(inventory, expected_refs, strict=True):
        if (physical.root_binding_id, physical.relative_path, physical.byte_count, physical.representation,
            physical.classification) != (ref.root_binding_id, ref.relative_path, ref.file_bytes, "loose_file", "exact"):
            _raise("physical inventory differs from the measured predecessor/evidence bytes")


def _verify_anchor(
    *,
    plan: HandoffPlan,
    receipt_bytes: bytes,
    receipt: dict[str, object],
    accepted_anchor: AcceptedHandoffAnchor | None,
) -> Literal["unavailable", "accepted"]:
    if accepted_anchor is None:
        return "unavailable"
    if type(accepted_anchor) is not AcceptedHandoffAnchor:
        _raise("accepted anchor has the wrong exact type")
    source_hash = sha256_bytes(
        canonical_json_bytes([asdict(item) for item in plan.source_files])
    )
    expected = {
        "specification_file_sha256": plan.specification_file_sha256,
        "plan_self_sha256": plan.plan_self_sha256,
        "invocation_id": derive_handoff_invocation_id(plan),
        "receipt_file_bytes": len(receipt_bytes),
        "receipt_file_sha256": sha256_bytes(receipt_bytes),
        "handoff_source_vector_sha256": source_hash,
    }
    for name, value in expected.items():
        if type(getattr(accepted_anchor, name)) is not type(value) or getattr(accepted_anchor, name) != value:
            _raise(f"accepted anchor {name} mismatch")
    if receipt["handoff_source_vector_sha256"] != source_hash:
        _raise("receipt source vector mismatch")
    return "accepted"


def verify_anchored_permission_handoff(
    *,
    expected_plan: HandoffPlan,
    expected_source_files: tuple[HandoffSourceFile, ...],
    start_bytes: bytes,
    outcome_bytes: bytes,
    acquisition_bytes: bytes,
    verification_bytes: bytes,
    index_bytes: bytes,
    receipt_bytes: bytes,
    accepted_anchor: AcceptedHandoffAnchor | None,
) -> HandoffVerification:
    content = verify_permission_handoff_contents(
        expected_plan=expected_plan,
        expected_source_files=expected_source_files,
        start_bytes=start_bytes,
        outcome_bytes=outcome_bytes,
        acquisition_bytes=acquisition_bytes,
    )
    try:
        outcome = strict_handoff_record_loads(
            outcome_bytes, kind="outcome", expected_plan=expected_plan
        )
        acquisition = strict_handoff_record_loads(
            acquisition_bytes, kind="acquisition", expected_plan=expected_plan
        )
        verification = strict_handoff_record_loads(
            verification_bytes, kind="verification", expected_plan=expected_plan
        )
        index = strict_handoff_record_loads(
            index_bytes, kind="index", expected_plan=expected_plan
        )
        receipt = strict_handoff_record_loads(
            receipt_bytes, kind="receipt", expected_plan=expected_plan
        )
    except Exception as exc:
        _raise("anchored handoff record is invalid", exc)
    parent = expected_plan.witness_store.parent_relative_path
    root = expected_plan.witness_store.root_binding_id
    _same_file_ref(
        verification["outcome_file_ref"], data=outcome_bytes,
        expected_path=f"{parent}/{WITNESS_FILENAMES[1]}", root=root,
    )
    _same_file_ref(
        verification["acquisition_file_ref"], data=acquisition_bytes,
        expected_path=f"{parent}/{WITNESS_FILENAMES[2]}", root=root,
    )
    expected_verification_payload = {
        "schema": RECORD_SCHEMAS["verification"],
        "invocation_id": derive_handoff_invocation_id(expected_plan),
        "plan_self_sha256": expected_plan.plan_self_sha256,
        "request_self_sha256": expected_plan.expected_request.request_self_sha256,
        "state4_authorization_file_sha256": expected_plan.authority_binding.full_state4_authorization_file_sha256,
        "outcome_file_ref": verification["outcome_file_ref"],
        "acquisition_file_ref": verification["acquisition_file_ref"],
        "content_result": None if content.content_result is None else _wire(content.content_result),
        "original_return_consistency": content.original_return_consistency,
        "reason_codes": list(content.reason_codes),
        "external_anchor_checked": False,
        "permission_acceptance_emitted": False,
    }
    expected_verification_payload["record_self_sha256"] = sha256_bytes(
        canonical_json_bytes(expected_verification_payload)
    )
    if verification != expected_verification_payload:
        _raise("serialized pre-anchor verification differs from recomputation")
    _verify_exact_inventory(
        plan=expected_plan,
        index=index,
        predecessor_bytes=(start_bytes, outcome_bytes, acquisition_bytes, verification_bytes),
        acquisition=acquisition,
    )
    if (
        receipt["index_file_bytes"] != len(index_bytes)
        or receipt["index_file_sha256"] != sha256_bytes(index_bytes)
        or receipt["index_commitment"]
        != handoff_record_commitment(index, kind="index", expected_plan=expected_plan)
    ):
        _raise("receipt does not bind the exact index bytes")
    anchor_status = _verify_anchor(
        plan=expected_plan,
        receipt_bytes=receipt_bytes,
        receipt=receipt,
        accepted_anchor=accepted_anchor,
    )
    consistency, runtime = _verify_original_return(outcome, expected_plan=expected_plan)
    reads = _decode_reads(acquisition, expected_plan=expected_plan)
    terminal = _verify_retention_journal(
        plan=expected_plan, runtime=runtime, reads=reads
    )
    original_retention: Literal["confirmed_success", "confirmed_failure", "unknown"] = "unknown"
    if anchor_status == "accepted" and terminal:
        if consistency == "consistent_success_return" and content.content_result is not None and content.content_result.payload_kind == "authz_evidence":
            original_retention = "confirmed_success"
        elif consistency == "consistent_failure_return":
            original_retention = "confirmed_failure"
    reasons = list(content.reason_codes)
    if anchor_status == "unavailable":
        reasons.append("external_anchor_unavailable")
    corroborated = (
        anchor_status == "accepted"
        and original_retention in {"confirmed_success", "confirmed_failure"}
        and content.content_result is not None
        and content.content_result.assessment_status == "completed"
        and content.content_result.payload_kind == "authz_evidence"
        and not content.reason_codes
    )
    if corroborated:
        reasons.append("historical_handoff_corroborated")
        overall: Literal["corroborated", "incomplete", "unanchored"] = "corroborated"
    elif anchor_status == "unavailable":
        overall = "unanchored"
    else:
        overall = "incomplete"
    return HandoffVerification(
        HANDOFF_VERIFICATION_SCHEMA,
        derive_handoff_invocation_id(expected_plan),
        content.content_result,
        original_retention,
        anchor_status,
        overall,
        _ordered_reasons(reasons),
        False,
    )
