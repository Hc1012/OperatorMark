"""Finite restart acquisition and effect-free independent comparison.

No subprocess, producer reconstruction, discovery, publication or acceptance.
Historical conclusions and original acknowledgments are never repaired by reads.
"""
from __future__ import annotations

from dataclasses import replace
from threading import RLock

from .canonical import sha256_bytes
from . import permission_handoff_records as old
from . import permission_handoff_administration_records as records
from . import permission_handoff_administration as administration
from . import permission_handoff_administration_journal as journal
from .permission_handoff_capture import _exception_summary
from .permission_handoff_verify import verify_anchored_permission_handoff


def _failure(stage, reason, error=None):
    return records.AdministrativeFailure(stage, "read_handoff_verification_input", reason, None,
        None if error is None else _exception_summary(error))


def _unattempted(input, cause):
    return records.HandoffRestartRead(input.slot_id, "unavailable", None, "unavailable", None, None,
        records._empty_accounting(), (), _failure("acquire", "unavailable", RuntimeError(cause)))


def _check_return(value):
    records._require(type(value) is administration.RestartCheckResult and value.status in {"active", "refused", "failed"}, "malformed restart checkpoint")
    if value.status == "active": records._require(value.reason is None, "active checkpoint has reason")
    elif value.status == "refused": records._require(value.reason in {"refused", "expired", "scope_mismatch"}, "refusal reason mismatch")
    else: records._require(value.reason in {"unavailable", "malformed_output"}, "failed checkpoint reason mismatch")
    return value


def _finish_return(value, requested):
    records._require(type(value) is administration.RestartFinishResult and value.status in {"succeeded", "failed"}
        and value.effect_certainty in {"confirmed", "not_applied", "unknown"}, "malformed restart terminal")
    if value.effect_certainty == "confirmed":
        records._require(value.status == "succeeded" and value.recorded_outcome == requested, "actual terminal did not acknowledge requested outcome")
    else: records._require(value.recorded_outcome is None, "unconfirmed finish invented outcome")
    return value


def _unsupported_read(raw, input, failure):
    """Project only intrinsically valid facts; caller retains the entire raw value."""
    data = before = after = None
    completeness = "unavailable"
    cleanup = []
    if type(raw) is records.HandoffRestartRead:
        if raw.data is not None:
            try:
                old._validate_byte_record(raw.data, name="unsupported provider bytes")
                records._require(raw.data_completeness in {"prefix", "complete"}, "invalid byte completeness")
                data, completeness = raw.data, raw.data_completeness
            except Exception:
                pass
        for name in ("before", "after"):
            value = getattr(raw, name)
            if value is not None:
                try:
                    records._metadata(value, input, match=False)
                except Exception:
                    continue
                if name == "before": before = value
                else: after = value
        for value in raw.cleanup_failures if type(raw.cleanup_failures) is tuple else ():
            try:
                records._failure(value, "read_handoff_verification_input")
                records._require(value.stage == "cleanup", "cleanup stage")
                cleanup.append(value)
            except Exception:
                pass
    return records.HandoffRestartRead(input.slot_id, "inconsistent", data, completeness, before, after,
        records._empty_accounting(), tuple(cleanup), replace(failure, reason="mismatch"))


def _read_once(*, ticket, input, authority, reader, events):
    accounting = records._empty_accounting()
    lease = None
    raw = None
    checkpoints = []
    requested = "failed"
    failure = None
    binding_valid = False
    try:
        acquired = authority.acquire(ticket, input.slot_id)
        events.append((input.slot_id, "acquire_return", acquired))
        records._require(type(acquired) is administration.RestartAcquireResult
            and acquired.status in {"granted", "refused", "failed"}, "malformed restart acquire")
        lease = acquired.lease
        records._require((acquired.status == "granted" and lease is not None and acquired.reason is None)
            or (acquired.status != "granted" and lease is None and acquired.reason in ({"refused", "expired", "scope_mismatch"} if acquired.status == "refused" else {"unavailable", "malformed_output"})), "contradictory restart acquire")
        accounting = replace(accounting, acquire_status=acquired.status, lease_returned=lease is not None,
            lease_binding_status="invalid" if lease is not None else "not_applicable")
        if lease is None:
            reason = "refused" if acquired.status == "refused" else "unavailable"
            return replace(_unattempted(input, "authority acquisition did not grant"), authority_accounting=accounting, failure=_failure("acquire", reason))
        grant = next(x for x in ticket.grants if x.slot_id == input.slot_id)
        expected_scope = administration._restart_scope(ticket, grant)
        records._require(type(lease.scope) is records.HandoffRestartLeaseScope and lease.scope == expected_scope, "restart lease scope mismatch")
        binding_valid = True
        accounting = replace(accounting, lease_binding_status="valid")
        for name in records.READ_CHECKPOINTS[:-1]:
            try:
                returned = authority.check_active(lease, name)
                events.append((input.slot_id, "raw_checkpoint_return", name, returned))
                checked = _check_return(returned)
            except BaseException as error:
                events.append((input.slot_id, "checkpoint_raised_or_malformed", name, error))
                checked = administration.RestartCheckResult("failed", "malformed_output")
            events.append((input.slot_id, name, checked))
            checkpoints.append(records.RestartCheckpoint(name, checked.status, checked.reason))
            if checked.status != "active":
                requested = "refused" if checked.status == "refused" else "failed"
                failure = _failure("acquire", "refused" if checked.status == "refused" else "unavailable")
                break
        else:
            try:
                raw = reader.read_restart_existing(ticket, input, lease)
                events.append((input.slot_id, "raw_provider_return", raw))
                # Operation facts are validated BEFORE selecting terminal success.
                records.validate_handoff_restart_read(raw, input=input, raw=True)
                requested = "succeeded" if raw.status == "complete" else "refused" if raw.failure.reason == "refused" else "failed"
            except BaseException as error:
                events.append((input.slot_id, "provider_or_facts_failed", error, raw))
                requested, failure = "failed", _failure("read", "malformed_output", error)
                raw = _unsupported_read(raw, input, failure)
            try:
                checked = _check_return(authority.check_active(lease, "final"))
            except BaseException as error:
                events.append((input.slot_id, "final_raised", error))
                checked = administration.RestartCheckResult("failed", "malformed_output")
            checkpoints.append(records.RestartCheckpoint("final", checked.status, checked.reason))
            if checked.status != "active":
                if requested == "succeeded": requested = "refused" if checked.status == "refused" else "failed"
                failure = failure or _failure("finish", "terminal_unconfirmed")
    except BaseException as error:
        events.append((input.slot_id, "acquire_or_checkpoint_raised", error))
        failure = _failure("lease_binding" if lease is not None and not binding_valid else "acquire", "unavailable", error)
        accounting = replace(accounting, acquire_status="failed", lease_returned=lease is not None,
            lease_binding_status="invalid" if lease is not None else "not_applicable")
    if lease is not None:
        try:
            returned = authority.finish(lease, requested)
            events.append((input.slot_id, "actual_finish_return", returned))
            finished = _finish_return(returned, requested)
        except BaseException as error:
            events.append((input.slot_id, "finish_unconfirmed", error))
            finished = administration.RestartFinishResult("failed", "unknown", None)
        accounting = replace(accounting, checkpoints=tuple(checkpoints), finish_attempted=True, requested_outcome=requested,
            finish_status=finished.status, effect_certainty=finished.effect_certainty, recorded_outcome=finished.recorded_outcome)
    if raw is None:
        return replace(_unattempted(input, "read not reached"), authority_accounting=accounting,
            failure=failure or _failure("read", "unavailable"))
    result = replace(raw, authority_accounting=accounting)
    if result.status == "complete" and not (requested == "succeeded" and accounting.finish_status == "succeeded"
            and accounting.effect_certainty == "confirmed" and accounting.recorded_outcome == "succeeded"):
        result = replace(result, status="inconsistent", failure=failure or _failure("finish", "terminal_unconfirmed"))
    return result


def _witness_bytes(reads):
    return {item.slot_id: old._validate_byte_record(item.data, name=item.slot_id) for item in reads[:6]
        if item.status == "complete" and item.data is not None}


def _historical(plan, source_files, ticket, content):
    return verify_anchored_permission_handoff(expected_plan=plan, expected_source_files=source_files,
        start_bytes=content["invocation-start.json"], outcome_bytes=content["invocation-outcome.json"], acquisition_bytes=content["acquisition.json"],
        verification_bytes=content["verification.json"], index_bytes=content["index.json"], receipt_bytes=content["receipt.json"], accepted_anchor=ticket.accepted_anchor)


def _chain_prefix(ticket, plan, reads):
    content = _witness_bytes(reads)
    kinds = dict(zip(old.WITNESS_FILENAMES, ("start", "outcome", "acquisition", "verification", "index", "receipt"), strict=True))
    decoded = {name: old.strict_handoff_record_loads(data, kind=kinds[name], expected_plan=plan) for name, data in content.items()}
    if "receipt.json" in content:
        anchor = ticket.accepted_anchor
        records._require((len(content["receipt.json"]), sha256_bytes(content["receipt.json"])) == (anchor.receipt_file_bytes, anchor.receipt_file_sha256), "receipt anchor mismatch")
    if "index.json" in content:
        receipt = decoded["receipt.json"]
        records._require((len(content["index.json"]), sha256_bytes(content["index.json"])) == (receipt["index_file_bytes"], receipt["index_file_sha256"]), "receipt-index mismatch")
        refs = decoded["index.json"]["ordered_witness_file_refs"]
        records._require(len(refs) == 4, "index predecessor count")
        for filename, ref in zip(old.WITNESS_FILENAMES[:4], refs, strict=True):
            input = next(x for x in ticket.inputs[:6] if x.slot_id == filename)
            expected = input.expected_file_ref
            records._require(ref == old._wire(expected), "candidate index changed independent predecessor reference")
    return content


def compare_fresh_handoff_evidence(*, expected_plan, historical_acquisition, ticket, ordered_reads):
    historical_reads = tuple(old._decode_dataclass(old.HandoffReadResult, x, name="historical acquisition read") for x in historical_acquisition["ordered_reads"])
    records._require(tuple(x.slot_id for x in historical_reads) == tuple(x.slot_id for x in expected_plan.input_slots), "historical input order")
    fresh = {input.original_slot_id: read for input, read in zip(ticket.inputs, ordered_reads, strict=True) if input.role == "original_input"}
    content, physical = "matched", "matched"
    for prior in historical_reads:
        current = fresh[prior.slot_id]
        if prior.status == "not_found":
            if current.status == "complete": content = "mismatch"
            elif current.status != "not_found" and content != "mismatch": content = "incomplete"
            if current.status == "complete": physical = "mismatch"
            elif current.status != "not_found" and physical != "mismatch": physical = "incomplete"
            continue
        if prior.status != "complete":
            if content != "mismatch": content = "incomplete"
            if physical != "mismatch": physical = "incomplete"
            continue
        if current.status == "not_found":
            content = physical = "mismatch"
            continue
        if current.status != "complete":
            if current.failure is not None and current.failure.reason == "mismatch":
                if current.data is not None and prior.data is not None and current.data != prior.data: content = "mismatch"
                elif content != "mismatch": content = "incomplete"
                if current.after is not None and prior.after is not None and current.after != prior.after: physical = "mismatch"
                elif physical != "mismatch": physical = "incomplete"
            else:
                if content != "mismatch": content = "incomplete"
                if physical != "mismatch": physical = "incomplete"
            continue
        if current.data != prior.data: content = "mismatch"
        if prior.after is None or current.after is None:
            if physical != "mismatch": physical = "incomplete"
        elif current.after != prior.after: physical = "mismatch"
    return content, physical


def _inventory(ticket, reads):
    return records._restart_inventory(ticket, reads)


def _compute_reacquired_permission_handoff(*, ticket, expected_plan, expected_source_files, ordered_reads):
    """Recompute every conclusion from inputs; never validate a report recursively."""
    records.validate_handoff_restart_ticket(ticket, expected_plan=expected_plan)
    records._require(type(expected_source_files) is tuple and expected_source_files == expected_plan.source_files, "historical source binding mismatch")
    records._require(type(ordered_reads) is tuple and len(ordered_reads) == len(ticket.inputs), "fresh read coverage mismatch")
    for input, read in zip(ticket.inputs, ordered_reads, strict=True): records.validate_handoff_restart_read(read, input=input)
    reasons = {"size_contribution_unresolved", "original_acknowledgment_unavailable"}
    historical = None
    content_status = physical_status = "incomplete"
    acquired = _witness_bytes(ordered_reads)
    if len(acquired) != 6:
        reasons.add("witness_unavailable")
    else:
        try:
            _chain_prefix(ticket, expected_plan, ordered_reads[:6])
            historical = _historical(expected_plan, expected_source_files, ticket, acquired)
        except Exception:
            reasons.add("witness_chain_mismatch")
        else:
            if historical.overall_status != "corroborated": reasons.add("historical_verification_incomplete")
            acquisition = old.strict_handoff_record_loads(acquired["acquisition.json"], kind="acquisition", expected_plan=expected_plan)
            content_status, physical_status = compare_fresh_handoff_evidence(expected_plan=expected_plan, historical_acquisition=acquisition,
                ticket=ticket, ordered_reads=ordered_reads)
    if "incomplete" in {content_status, physical_status}: reasons.add("fresh_input_unavailable")
    if content_status == "mismatch": reasons.add("fresh_content_mismatch")
    if physical_status == "mismatch": reasons.add("fresh_physical_mismatch")
    capture_status = "not_recorded_legacy" if ticket.verification_mode == "legacy_review" else "incomplete"
    if capture_status == "not_recorded_legacy": reasons.add("legacy_authority_not_recorded")
    elif historical is not None:
        policy_read = next((read for input, read in zip(ticket.inputs, ordered_reads, strict=True)
            if ticket.capture_policy_file_ref is not None and (input.root_binding_id, input.relative_path) ==
                (ticket.capture_policy_file_ref.root_binding_id, ticket.capture_policy_file_ref.relative_path)), None)
        if policy_read is not None and policy_read.status == "complete":
            try:
                policy = records.strict_handoff_administrative_loads(old._validate_byte_record(policy_read.data, name="capture policy"), kind="policy", expected_plan=expected_plan)
                records.validate_handoff_restart_ticket(ticket, expected_plan=expected_plan, capture_policy=policy)
                scopes = tuple((g, records._legacy_scope(expected_plan, g.operation, g.slot_id)) for g in policy.grants)
                inputs = tuple((input.relative_path, read) for input, read in zip(ticket.inputs, ordered_reads, strict=True) if input.role == "capture_administration")
                inspection = journal.inspect_handoff_authority_records(policy=policy, expected_scopes=journal._ExpectedScopes(expected_plan, scopes), acquired_records=inputs)
                capture_status = inspection.status
                retained = dict(inspection.records)
                historical_acquisition = old.strict_handoff_record_loads(acquired["acquisition.json"], kind="acquisition", expected_plan=expected_plan)
                activation_id = records._activation_id(policy, expected_plan.expected_request.request_self_sha256)
                for wire in historical_acquisition["ordered_reads"]:
                    original_read = old._decode_dataclass(old.HandoffReadResult, wire, name="historical authority accounting")
                    accounting = original_read.authority_accounting
                    grant = next(x for x in policy.grants if x.operation == "read_handoff_input" and x.slot_id == original_read.slot_id)
                    stem = activation_id + "." + grant.allowance_id
                    if accounting.lease_returned and retained.get(stem + ".reservation.json") is None:
                        capture_status = "incomplete"
                    if accounting.finish_attempted:
                        terminal = retained.get(stem + ".terminal.json")
                        if terminal is None:
                            capture_status = "incomplete"
                        else:
                            records._require(terminal[0].requested_outcome == accounting.requested_outcome, "historical operation and administrative terminal disagree")
                    for ordinal, check in enumerate(accounting.checkpoints):
                        decision = retained.get(stem + f".{ordinal}.checkpoint.json")
                        if decision is None:
                            capture_status = "incomplete"
                        else:
                            records._require((decision[0].checkpoint, decision[0].status, decision[0].reason) == (check.name, check.status, check.reason),
                                "historical checkpoint and administrative decision disagree")
                # Historical witness bytes do not serialize their own final
                # receipt-write finish acknowledgment; matching journal content
                # does not fill this independent evidence gap.
            except ValueError:
                # Closed-record/known-evidence contradictions establish mismatch.
                # An unexpected computation failure is not evidence of mismatch.
                capture_status = "mismatch"
    if capture_status == "incomplete": reasons.add("capture_authority_unavailable")
    if capture_status == "mismatch": reasons.add("capture_authority_mismatch")
    restart_confirmed = records._restart_accounting_confirmed(ordered_reads)
    if not restart_confirmed: reasons.add("restart_authority_unconfirmed")
    if any(x.cleanup_failures for x in ordered_reads): reasons.add("cleanup_failed")
    report = records._seal(records.HandoffRestartReport(records.PREFIX + "restart_report.v1", ticket.verification_id, ticket.record_self_sha256,
        sha256_bytes(records.canonical_json_bytes(old._wire(ticket)) + b"\n"), ticket.accepted_anchor, expected_plan.plan_self_sha256,
        ticket.original_invocation_id, ordered_reads, historical, content_status, physical_status, capture_status, "incomplete",
        "confirmed" if restart_confirmed else "incomplete", tuple(x for x in records.REASONS if x in reasons), _inventory(ticket, ordered_reads), False, False, "0" * 64))
    return report


def verify_reacquired_permission_handoff(*, ticket, expected_plan, expected_source_files, ordered_reads):
    report = _compute_reacquired_permission_handoff(ticket=ticket, expected_plan=expected_plan,
        expected_source_files=expected_source_files, ordered_reads=ordered_reads)
    # Constructed conclusions are already computed; external reports use the full
    # validator, which independently recomputes and compares the entire value.
    records._validate_handoff_restart_report_structure(report, ticket=ticket, expected_plan=expected_plan)
    return report


class HandoffRestartCoordinator:
    def __init__(self):
        self._attempted = False
        self._lock = RLock()
        self._evidence = []

    @property
    def evidence(self): return tuple(self._evidence)

    def run_once(self, *, ticket, expected_plan, expected_source_files, authority, reader):
        with self._lock:
            records._require(not self._attempted, "restart coordinator already attempted/reentrant")
            self._attempted = True
        records.validate_handoff_restart_ticket(ticket, expected_plan=expected_plan)
        reads, cause = [], None
        try:
            activation = _check_return(authority.activate_once())
            self._evidence.append(("activation_return", activation))
            if activation.status != "active": cause = "restart activation did not establish authority"
        except BaseException as error:
            self._evidence.append(("activation_raised", error))
            cause = "restart activation unavailable"
        for index, input in enumerate(ticket.inputs):
            if cause is not None:
                reads.append(_unattempted(input, cause))
                continue
            result = _read_once(ticket=ticket, input=input, authority=authority, reader=reader, events=self._evidence)
            reads.append(result)
            if index < 6:
                try:
                    records._require(result.status == "complete", "required witness unavailable")
                    _chain_prefix(ticket, expected_plan, tuple(reads))
                    if index == 5: _historical(expected_plan, expected_source_files, ticket, _witness_bytes(reads))
                except Exception as error:
                    self._evidence.append((input.slot_id, "dependent_chain_stopped", error))
                    cause = "required fixed witness chain validation failed"
            elif result.authority_accounting.finish_status != "succeeded" or result.authority_accounting.effect_certainty != "confirmed":
                cause = "restart authority accounting unavailable"
        return verify_reacquired_permission_handoff(ticket=ticket, expected_plan=expected_plan,
            expected_source_files=expected_source_files, ordered_reads=tuple(reads))
