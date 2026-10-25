"""Once-only supervision of one already-prepared durable Authz invocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal, cast

from .canonical import canonical_json_bytes, sha256_bytes
from . import permission_controller as controller
from . import permission_principal_authz_windows as supplied_authz
from . import private_permission_evidence as evidence
from .permission_handoff_records import (
    ByteRecord,
    CaptureFailure,
    ExceptionSummary,
    FileRef,
    HandoffAcquireResult,
    HandoffAuthorityAccounting,
    HandoffCheckpoint,
    HandoffCheckpointResult,
    HandoffFailure,
    HandoffFinishResult,
    HandoffLastKnownObject,
    HandoffObjectMetadata,
    HandoffPhysicalFile,
    HandoffPlan,
    HandoffWriteResult,
    RECORD_SCHEMAS,
    TokenEvent,
    TokenTrace,
    WITNESS_FILENAMES,
    _decode_original_runtime,
    _wire,
    _expected_predecessor_refs,
    canonical_handoff_record_bytes,
    derive_handoff_invocation_id,
    handoff_record_commitment,
    validate_handoff_plan,
)
from .permission_handoff_sources import (
    PermissionHandoffAuthority,
    PermissionHandoffReader,
    PermissionHandoffWitnessSink,
    _empty_accounting,
    _exception_failure,
    _finish_result,
    _acquire_return_valid,
    _malformed_acquire_accounting,
    _check_result,
    _validate_lease_scope,
    _validate_write_facts,
    acquire_handoff_inputs,
    validate_handoff_write_result,
)
from .permission_handoff_verify import verify_permission_handoff_contents


@dataclass(frozen=True, slots=True)
class HandoffRunResult:
    invocation_id: str
    status: Literal["recorded", "incomplete"]
    original_call_observed: Literal[
        "not_called", "returned_valid", "returned_malformed",
        "returned_unavailable", "raised", "unknown",
    ]
    original_runtime_return: controller.PermissionDurableAuthzRuntimeResult | None
    ordered_read_results: tuple[object, ...]
    ordered_write_results: tuple[HandoffWriteResult, ...]
    failures: tuple[HandoffFailure, ...]
    permission_acceptance_emitted: Literal[False] = False


class PermissionHandoffCaptureError(RuntimeError):
    """The once-only supervisor could not safely continue."""


def _exception_summary(exc: BaseException) -> ExceptionSummary:
    cls = type(exc)
    try:
        message = str(exc)
    except Exception:
        message = None
        reason: Literal["unavailable", "unencodable"] | None = "unavailable"
    else:
        try:
            message.encode("utf-8")
        except Exception:
            message = None
            reason = "unencodable"
        else:
            reason = None
    return ExceptionSummary(cls.__module__, cls.__qualname__, message, reason)


def _copy_statistics(value: supplied_authz.NativeTokenStatistics) -> supplied_authz.NativeTokenStatistics:
    return supplied_authz.NativeTokenStatistics(**asdict(value))


class WitnessingAuthzBindings:
    """Delegates exactly once and records only call facts and statistics returns."""

    __slots__ = (
        "_delegate", "_events", "_borrowed_token", "_mismatch", "_capture_failed",
        "_managers", "_contexts",
    )

    def __init__(self, delegate: supplied_authz.AuthzNativeBindings) -> None:
        self._delegate = delegate
        self._events: list[TokenEvent] = []
        self._borrowed_token: int | None = None
        self._mismatch = False
        self._capture_failed = False
        self._managers: set[int] = set()
        self._contexts: set[int] = set()

    def _token(self, handle: int) -> None:
        if type(handle) is not int:
            self._capture_failed = True
            return
        if self._borrowed_token is None:
            self._borrowed_token = handle
        elif handle != self._borrowed_token:
            self._mismatch = True

    def _event(
        self,
        method: str,
        outcome: Literal["returned_valid", "returned_malformed", "raised"],
        *,
        statistics: supplied_authz.NativeTokenStatistics | None = None,
        exception: ExceptionSummary | None = None,
    ) -> None:
        try:
            self._events.append(
                TokenEvent(
                    len(self._events), cast(object, method), outcome,
                    statistics, exception,
                )
            )
        except Exception:
            self._capture_failed = True

    def query_token_statistics(self, borrowed_token_handle: int) -> supplied_authz.NativeTokenStatistics:
        self._token(borrowed_token_handle)
        try:
            result = self._delegate.query_token_statistics(borrowed_token_handle)
        except BaseException as exc:
            self._event("query_token_statistics", "raised", exception=_exception_summary(exc))
            raise
        if type(result) is supplied_authz.NativeTokenStatistics:
            try:
                supplied_authz._validate_statistics(result)
            except Exception:
                self._event("query_token_statistics", "returned_malformed")
            else:
                try:
                    copied = _copy_statistics(result)
                except Exception:
                    # The delegate return was valid. Do not fabricate a malformed
                    # return or statistics when only the supervisory copy failed.
                    self._capture_failed = True
                else:
                    self._event("query_token_statistics", "returned_valid", statistics=copied)
        else:
            self._event("query_token_statistics", "returned_malformed")
        return result

    def initialize_resource_manager(self) -> int:
        try:
            result = self._delegate.initialize_resource_manager()
        except BaseException as exc:
            self._event("initialize_resource_manager", "raised", exception=_exception_summary(exc))
            raise
        try:
            supplied_authz._validate_handle(result)
        except Exception:
            valid = False
        else:
            valid = True
        outcome = "returned_valid" if valid else "returned_malformed"
        self._event("initialize_resource_manager", outcome)
        if valid:
            self._managers.add(result)
        return result

    def initialize_context_from_token(
        self, borrowed_token_handle: int, resource_manager_handle: int
    ) -> int:
        self._token(borrowed_token_handle)
        try:
            result = self._delegate.initialize_context_from_token(
                borrowed_token_handle, resource_manager_handle
            )
        except BaseException as exc:
            self._event("initialize_context_from_token", "raised", exception=_exception_summary(exc))
            raise
        try:
            supplied_authz._validate_handle(result)
        except Exception:
            valid = False
        else:
            valid = resource_manager_handle in self._managers
        self._event(
            "initialize_context_from_token",
            "returned_valid" if valid else "returned_malformed",
        )
        if valid:
            self._contexts.add(result)
        return result

    def access_check(
        self,
        context_handle: int,
        *,
        descriptor_bytes: bytes,
        desired_access: int,
    ) -> supplied_authz.NativeAuthzAccessReply:
        try:
            result = self._delegate.access_check(
                context_handle,
                descriptor_bytes=descriptor_bytes,
                desired_access=desired_access,
            )
        except BaseException as exc:
            self._event("access_check", "raised", exception=_exception_summary(exc))
            raise
        valid = (
            type(result) is supplied_authz.NativeAuthzAccessReply
            and context_handle in self._contexts
        )
        if valid:
            try:
                for name in ("result_count", "granted_access_mask", "sacl_evaluation_results", "error_code"):
                    supplied_authz._exact_uint32(getattr(result, name), name=name)
            except Exception:
                valid = False
        self._event("access_check", "returned_valid" if valid else "returned_malformed")
        return result

    def free_context(self, context_handle: int) -> None:
        try:
            result = self._delegate.free_context(context_handle)
        except BaseException as exc:
            self._event("free_context", "raised", exception=_exception_summary(exc))
            raise
        valid = result is None and context_handle in self._contexts
        self._event("free_context", "returned_valid" if valid else "returned_malformed")
        if valid:
            self._contexts.remove(context_handle)
        return result

    def free_resource_manager(self, resource_manager_handle: int) -> None:
        try:
            result = self._delegate.free_resource_manager(resource_manager_handle)
        except BaseException as exc:
            self._event("free_resource_manager", "raised", exception=_exception_summary(exc))
            raise
        valid = result is None and resource_manager_handle in self._managers
        self._event(
            "free_resource_manager", "returned_valid" if valid else "returned_malformed"
        )
        if valid:
            self._managers.remove(resource_manager_handle)
        return result

    @property
    def token_trace(self) -> TokenTrace:
        reasons: list[str] = []
        if self._capture_failed:
            reasons.append("capture_failed")
        if self._mismatch:
            reasons.append("token_handle_mismatch")
        queries = [event for event in self._events if event.method == "query_token_statistics"]
        valid_queries = [event for event in queries if event.outcome == "returned_valid"]
        if len(valid_queries) != 4:
            reasons.append("incomplete_query_pair")
        statistics = [event.statistics for event in valid_queries]
        if statistics and any(item != statistics[0] for item in statistics):
            reasons.append("token_identity_mismatch")
        if any(event.outcome == "returned_malformed" for event in self._events):
            reasons.append("malformed_native_return")
        if any(event.outcome == "raised" for event in self._events):
            reasons.append("native_call_raised")
        incomplete_resources = bool(self._contexts or self._managers)
        if incomplete_resources and "incomplete_query_pair" not in reasons:
            reasons.append("incomplete_query_pair")
        order = (
            "capture_failed", "token_handle_mismatch", "incomplete_query_pair",
            "token_identity_mismatch", "malformed_native_return", "native_call_raised",
        )
        ordered = tuple(code for code in order if code in reasons)
        if not self._events:
            status: Literal["complete", "partial", "unavailable"] = "unavailable"
            same: Literal["confirmed", "mismatch", "unavailable"] = "unavailable"
        else:
            status = "complete" if not ordered else "partial"
            same = "mismatch" if self._mismatch else (
                "confirmed" if self._borrowed_token is not None else "unavailable"
            )
        return TokenTrace(status, tuple(self._events), same, ordered)


def _seal(payload: dict[str, object]) -> dict[str, object]:
    if "record_self_sha256" in payload:
        raise PermissionHandoffCaptureError("record payload already has a self hash")
    result = dict(payload)
    result["record_self_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return result


def _common(plan: HandoffPlan, kind: str) -> dict[str, object]:
    return {
        "schema": RECORD_SCHEMAS[kind],
        "invocation_id": derive_handoff_invocation_id(plan),
        "plan_self_sha256": plan.plan_self_sha256,
        "request_self_sha256": plan.expected_request.request_self_sha256,
        "state4_authorization_file_sha256": plan.authority_binding.full_state4_authorization_file_sha256,
    }


def _start_record(plan: HandoffPlan) -> dict[str, object]:
    return _seal(
        {
            **_common(plan, "start"),
            "dispatch_may_follow": True,
            "predecessor_file_refs": [
                asdict(ref) for ref in _expected_predecessor_refs(plan)
            ],
            "permission_acceptance_emitted": False,
        }
    )


def _file_ref(plan: HandoffPlan, filename: str, data: bytes) -> FileRef:
    return FileRef(
        plan.witness_store.root_binding_id,
        f"{plan.witness_store.parent_relative_path}/{filename}",
        len(data),
        sha256_bytes(data),
    )


def _runtime_projection(
    value: controller.PermissionDurableAuthzRuntimeResult,
) -> dict[str, object]:
    intended = None
    if value.authz_record_bytes is not None:
        if type(value.authz_record_bytes) is not bytes:
            raise TypeError("runtime Authz bytes have the wrong exact type")
        intended = asdict(
            ByteRecord(
                value.authz_record_bytes.hex(),
                len(value.authz_record_bytes),
                sha256_bytes(value.authz_record_bytes),
            )
        )
    retention = value.retention_result
    if retention is None:
        kind = "not_returned"
        projected = None
    elif type(retention) is evidence.RetainedPermissionAuthzEvidence:
        kind = "success_receipt"
        projected = asdict(retention)
    elif type(retention) is evidence.PermissionAuthzEvidenceRetentionFailure:
        kind = "retention_failure"
        projected = asdict(retention)
    else:
        raise TypeError("runtime retention result has the wrong exact type")
    return {
        "persistence_status": value.persistence_status,
        "intended_authz_record": intended,
        "retention_result_kind": kind,
        "retention_result": projected,
        "failure_reason": value.failure_reason,
        "permission_acceptance_emitted": value.permission_acceptance_emitted,
    }


def capture_original_authz_outcome(
    *,
    expected_plan: HandoffPlan,
    original_return: object,
    exception: BaseException | None,
    token_trace: TokenTrace,
) -> dict[str, object]:
    validate_handoff_plan(expected_plan)
    trace = token_trace
    # Validation occurs again in the record codec; constructing from another
    # type is not accepted here.
    if type(trace) is not TokenTrace:
        raise PermissionHandoffCaptureError("token trace has the wrong exact type")
    failures: list[CaptureFailure] = []
    runtime: dict[str, object] | None = None
    summary: ExceptionSummary | None = None
    if exception is not None:
        call_outcome = "raised"
        summary = _exception_summary(exception)
        failures.append(CaptureFailure("delegate_call", "raised", summary))
    elif type(original_return) is controller.PermissionDurableAuthzRuntimeResult:
        try:
            runtime = cast(dict[str, object], _wire(_runtime_projection(original_return)))
        except (TypeError, ValueError) as exc:
            call_outcome = "returned_malformed"
            failures.append(
                CaptureFailure("runtime_capture", "malformed_return", _exception_summary(exc))
            )
            runtime = None
        except Exception as exc:
            call_outcome = "returned_unavailable"
            failures.append(CaptureFailure("runtime_capture", "copy_failed", _exception_summary(exc)))
            runtime = None
        else:
            try:
                _decode_original_runtime(runtime, expected_plan=expected_plan)
            except Exception as exc:
                call_outcome = "returned_malformed"
                failures.append(CaptureFailure("runtime_capture", "malformed_return", _exception_summary(exc)))
                runtime = None
            else:
                call_outcome = "returned_valid"
    elif original_return is None:
        call_outcome = "returned_unavailable"
    else:
        call_outcome = "returned_malformed"
        failures.append(CaptureFailure("runtime_capture", "malformed_return", None))
    if "capture_failed" in trace.reason_codes:
        failures.append(CaptureFailure("token_capture", "copy_failed", None))
    start = _start_record(expected_plan)
    start_bytes = canonical_handoff_record_bytes(
        start, kind="start", expected_plan=expected_plan
    )
    return _seal(
        {
            **_common(expected_plan, "outcome"),
            "start_file_ref": asdict(_file_ref(expected_plan, WITNESS_FILENAMES[0], start_bytes)),
            "call_outcome": call_outcome,
            "runtime_projection": runtime,
            "exception": None if summary is None else asdict(summary),
            "token_trace": asdict(trace),
            "capture_failures": [asdict(item) for item in failures],
            "permission_acceptance_emitted": False,
        }
    )


def _unknown_write(
    slot_id: str,
    *,
    accounting: HandoffAuthorityAccounting,
    failure: HandoffFailure,
) -> HandoffWriteResult:
    return HandoffWriteResult(
        slot_id, "unknown", None, "unknown", "unknown", "unknown", "unknown",
        accounting, (), failure,
    )


def _publish_witness_once(
    *,
    plan: HandoffPlan,
    slot_id: str,
    data: bytes,
    authority: PermissionHandoffAuthority,
    witness_sink: PermissionHandoffWitnessSink,
) -> HandoffWriteResult:
    operation: Literal["retain_handoff_witness"] = "retain_handoff_witness"
    try:
        acquired = authority.acquire(plan, operation, slot_id)
    except Exception:
        accounting = replace(_empty_accounting(), acquire_status="failed")
        return _unknown_write(
            slot_id, accounting=accounting,
            failure=_exception_failure(stage="acquire", operation=operation, reason="raised"),
        )
    if not _acquire_return_valid(acquired):
        return _unknown_write(slot_id, accounting=_malformed_acquire_accounting(acquired, authority),
                              failure=_exception_failure(stage="acquire", operation=operation))
    if type(acquired) is not HandoffAcquireResult or acquired.status != "granted" or acquired.lease is None:
        status = acquired.status if type(acquired) is HandoffAcquireResult and acquired.status in {"refused", "failed"} else "failed"
        accounting = replace(_empty_accounting(), acquire_status=status)
        failure = _exception_failure(
            stage="acquire", operation=operation,
            reason="refused" if status == "refused" else "malformed_output",
        )
        return HandoffWriteResult(
            slot_id,
            "failed" if status == "refused" else "unknown",
            None,
            "not_created" if status == "refused" else "unknown",
            "no_effect" if status == "refused" else "unknown",
            "no_object" if status == "refused" else "unknown",
            "not_acquired" if status == "refused" else "unknown",
            accounting,
            (),
            failure,
        )
    lease = acquired.lease
    accounting = replace(
        _empty_accounting(), acquire_status="granted", lease_returned=True,
        lease_binding_status="invalid",
    )
    binding_valid = True
    try:
        _validate_lease_scope(
            lease,
            plan=plan,
            operation=operation,
            slot_id=slot_id,
            root_binding_id=plan.witness_store.root_binding_id,
            role="authority_input",
        )
    except Exception:
        binding_valid = False
    accounting = replace(accounting, lease_binding_status="valid" if binding_valid else "invalid")
    checkpoints: list[HandoffCheckpoint] = []
    raw: HandoffWriteResult | None = None
    facts_valid = False
    requested: Literal["succeeded", "refused", "failed"] = "failed"
    failure: HandoffFailure | None = None
    if binding_valid:
        for name in ("initial", "before_open", "before_bytes", "before_publish"):
            try:
                checked = _check_result(authority.check_active(lease, name), name=name)
            except Exception:
                checked = HandoffCheckpointResult("failed", "malformed_output")
            checkpoints.append(HandoffCheckpoint(cast(object, name), checked.status, checked.reason))
            if checked.status != "active":
                requested = "refused" if checked.status == "refused" else "failed"
                failure = _exception_failure(
                    stage="publish", operation=operation,
                    reason="refused" if checked.status == "refused" else "unavailable",
                )
                break
        else:
            try:
                candidate = witness_sink.publish_exclusive(plan, slot_id, data, lease)
                if type(candidate) is not HandoffWriteResult:
                    raise TypeError("sink result has wrong exact type")
                raw = candidate
            except Exception:
                failure = _exception_failure(stage="publish", operation=operation, reason="raised")
                requested = "failed"
            else:
                try:
                    _validate_write_facts(raw, plan=plan, slot_id=slot_id, data=data)
                except Exception:
                    failure = _exception_failure(stage="publish", operation=operation)
                else:
                    facts_valid = True
                    requested = "succeeded" if raw.status == "retained" else (
                        "refused" if raw.status == "collision" or raw.failure.reason == "refused" else "failed"
                    )
            try:
                checked = _check_result(authority.check_active(lease, "final"), name="final")
            except Exception:
                checked = HandoffCheckpointResult("failed", "malformed_output")
            checkpoints.append(HandoffCheckpoint("final", checked.status, checked.reason))
            if checked.status != "active":
                if raw is None or facts_valid:
                    requested = "refused" if checked.status == "refused" else "failed"
                if failure is None:
                    failure = _exception_failure(
                        stage="finish", operation=operation, reason="terminal_unconfirmed"
                    )
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
        return _unknown_write(
            slot_id, accounting=accounting,
            failure=failure or _exception_failure(stage="publish", operation=operation),
        )
    result = replace(raw, authority_accounting=accounting)
    if not facts_valid:
        # Preserve last-known object and cleanup evidence, but never affirm
        # effects from a contradictory provider result or invent its finish.
        return replace(result, status="unknown", effect_certainty="unknown", failure=failure)
    if (
        requested != "succeeded"
        or finished.status != "succeeded"
        or finished.effect_certainty != "confirmed"
        or finished.recorded_outcome != "succeeded"
    ) and result.status == "retained":
        result = replace(
            result,
            status="unknown",
            effect_certainty="unknown",
            failure=failure or _exception_failure(
                stage="finish", operation=operation, reason="terminal_unconfirmed"
            ),
        )
    try:
        validate_handoff_write_result(
            result, plan=plan, slot_id=slot_id, data=data
        )
    except Exception:
        return replace(
            result,
            status="unknown",
            effect_certainty="unknown",
            failure=_exception_failure(stage="publish", operation=operation),
        )
    return result


def _require_confirmed_start(result: HandoffWriteResult) -> None:
    if (
        result.status != "retained"
        or result.publication_state != "published"
        or result.effect_certainty != "confirmed"
        or result.ownership != "owned_by_this_invocation"
        or result.handle_state != "released"
        or result.cleanup_failures
        or result.failure is not None
        or result.authority_accounting.effect_certainty != "confirmed"
        or result.authority_accounting.recorded_outcome != "succeeded"
    ):
        raise PermissionHandoffCaptureError("dispatch requires a confirmed start witness")


def _capture_token_event(bindings: WitnessingAuthzBindings) -> TokenTrace:
    return bindings.token_trace


def _physical(
    ref: FileRef, *, owner: str = "accepted_attempt_M", controls: bool = True
) -> HandoffPhysicalFile:
    return HandoffPhysicalFile(
        ref.root_binding_id,
        ref.relative_path,
        "loose_file",
        cast(object, owner),
        controls,
        ref.file_bytes,
        "exact",
        "observed_file_length",
        ("bytes_retained_and_measured",),
    )


class PermissionHandoffCoordinator:
    __slots__ = ("_in_progress", "_attempted")

    def __init__(self) -> None:
        self._in_progress = False
        self._attempted: set[str] = set()

    def _validate_single_invocation(self, invocation_id: str) -> None:
        if self._in_progress:
            raise PermissionHandoffCaptureError("permission handoff is reentrant")
        if invocation_id in self._attempted:
            raise PermissionHandoffCaptureError("permission handoff invocation was already attempted")

    def run_once(
        self,
        *,
        plan: HandoffPlan,
        parent: object,
        native: object,
        authz_bindings: supplied_authz.AuthzNativeBindings,
        observation_store: object,
        observation_failure_store: object,
        authz_store: object,
        authority: PermissionHandoffAuthority,
        reader: PermissionHandoffReader,
        witness_sink: PermissionHandoffWitnessSink,
    ) -> HandoffRunResult:
        validate_handoff_plan(plan)
        invocation_id = derive_handoff_invocation_id(plan)
        self._validate_single_invocation(invocation_id)
        self._attempted.add(invocation_id)
        self._in_progress = True
        writes: list[HandoffWriteResult] = []
        reads: tuple[object, ...] = ()
        failures: list[HandoffFailure] = []
        original: controller.PermissionDurableAuthzRuntimeResult | None = None
        observed: Literal[
            "not_called", "returned_valid", "returned_malformed",
            "returned_unavailable", "raised", "unknown",
        ] = "not_called"
        try:
            required_ports = (
                (parent, ("observe_and_retain_with_durable_authz",)),
                (authority, ("acquire", "check_active", "finish")),
                (reader, ("read_existing",)),
                (witness_sink, ("publish_exclusive",)),
                (authz_bindings, ("query_token_statistics", "initialize_resource_manager",
                                  "initialize_context_from_token", "access_check", "free_context",
                                  "free_resource_manager")),
            )
            if any(not callable(getattr(port, name, None)) for port, names in required_ports for name in names):
                failures.append(_exception_failure(stage="acquire", operation="retain_handoff_witness", reason="unavailable"))
                return HandoffRunResult(invocation_id, "incomplete", "not_called", None, (), (), tuple(failures), False)
            start_record = _start_record(plan)
            start_bytes = canonical_handoff_record_bytes(
                start_record, kind="start", expected_plan=plan
            )
            start_write = _publish_witness_once(
                plan=plan, slot_id=WITNESS_FILENAMES[0], data=start_bytes,
                authority=authority, witness_sink=witness_sink,
            )
            writes.append(start_write)
            try:
                _require_confirmed_start(start_write)
            except PermissionHandoffCaptureError:
                if start_write.failure is not None:
                    failures.append(start_write.failure)
                return HandoffRunResult(
                    invocation_id, "incomplete", observed, None, reads,
                    tuple(writes), tuple(failures), False,
                )

            witnessing = WitnessingAuthzBindings(authz_bindings)
            raised: BaseException | None = None
            returned: object = None
            try:
                method = parent.observe_and_retain_with_durable_authz
                returned = method(
                    plan.expected_request,
                    native=native,
                    authz_bindings=witnessing,
                    observation_store=observation_store,
                    observation_failure_store=observation_failure_store,
                    authz_store=authz_store,
                )
            except BaseException as exc:
                raised = exc
                observed = "raised"
            else:
                if type(returned) is controller.PermissionDurableAuthzRuntimeResult:
                    original = returned
                    observed = "returned_valid"
                elif returned is None:
                    observed = "returned_unavailable"
                else:
                    observed = "returned_malformed"
            outcome_record = capture_original_authz_outcome(
                expected_plan=plan,
                original_return=returned,
                exception=raised,
                token_trace=_capture_token_event(witnessing),
            )
            observed = cast(object, outcome_record["call_outcome"])
            outcome_bytes = canonical_handoff_record_bytes(
                outcome_record, kind="outcome", expected_plan=plan
            )
            outcome_write = _publish_witness_once(
                plan=plan, slot_id=WITNESS_FILENAMES[1], data=outcome_bytes,
                authority=authority, witness_sink=witness_sink,
            )
            writes.append(outcome_write)
            if outcome_write.status != "retained":
                if outcome_write.failure is not None:
                    failures.append(outcome_write.failure)
                return HandoffRunResult(
                    invocation_id, "incomplete", observed, original, reads,
                    tuple(writes), tuple(failures), False,
                )

            reads = acquire_handoff_inputs(
                plan=plan, original_outcome=outcome_record,
                authority=authority, reader=reader,
            )
            acquisition_record = _seal(
                {
                    **_common(plan, "acquisition"),
                    "outcome_file_ref": asdict(_file_ref(plan, WITNESS_FILENAMES[1], outcome_bytes)),
                    "acquisition_mode": "independent_reader",
                    "ordered_reads": [asdict(item) for item in reads],
                    "permission_acceptance_emitted": False,
                }
            )
            acquisition_bytes = canonical_handoff_record_bytes(
                acquisition_record, kind="acquisition", expected_plan=plan
            )
            acquisition_write = _publish_witness_once(
                plan=plan, slot_id=WITNESS_FILENAMES[2], data=acquisition_bytes,
                authority=authority, witness_sink=witness_sink,
            )
            writes.append(acquisition_write)
            if acquisition_write.status != "retained":
                if acquisition_write.failure is not None:
                    failures.append(acquisition_write.failure)
                return HandoffRunResult(
                    invocation_id, "incomplete", observed, original, reads,
                    tuple(writes), tuple(failures), False,
                )

            content = verify_permission_handoff_contents(
                expected_plan=plan,
                expected_source_files=plan.source_files,
                start_bytes=start_bytes,
                outcome_bytes=outcome_bytes,
                acquisition_bytes=acquisition_bytes,
            )
            verification_record = _seal(
                {
                    **_common(plan, "verification"),
                    "outcome_file_ref": asdict(_file_ref(plan, WITNESS_FILENAMES[1], outcome_bytes)),
                    "acquisition_file_ref": asdict(_file_ref(plan, WITNESS_FILENAMES[2], acquisition_bytes)),
                    "content_result": None if content.content_result is None else asdict(content.content_result),
                    "original_return_consistency": content.original_return_consistency,
                    "reason_codes": list(content.reason_codes),
                    "external_anchor_checked": False,
                    "permission_acceptance_emitted": False,
                }
            )
            verification_bytes = canonical_handoff_record_bytes(
                verification_record, kind="verification", expected_plan=plan
            )
            verification_write = _publish_witness_once(
                plan=plan, slot_id=WITNESS_FILENAMES[3], data=verification_bytes,
                authority=authority, witness_sink=witness_sink,
            )
            writes.append(verification_write)
            if verification_write.status != "retained":
                if verification_write.failure is not None:
                    failures.append(verification_write.failure)
                return HandoffRunResult(
                    invocation_id, "incomplete", observed, original, reads,
                    tuple(writes), tuple(failures), False,
                )

            predecessor_bytes = (start_bytes, outcome_bytes, acquisition_bytes, verification_bytes)
            predecessor_refs = tuple(
                _file_ref(plan, filename, data)
                for filename, data in zip(WITNESS_FILENAMES[:4], predecessor_bytes, strict=True)
            )
            evidence_refs = tuple(
                FileRef(slot.root_binding_id, slot.relative_path, item.data.file_bytes, item.data.file_sha256)
                for slot, item in zip(plan.input_slots, reads, strict=True)
                if item.status == "complete" and item.data is not None
            )
            inventory = tuple(_physical(ref) for ref in predecessor_refs) + tuple(
                _physical(ref, owner="external_preexisting", controls=False)
                for ref in evidence_refs
            )
            index_record = _seal(
                {
                    **_common(plan, "index"),
                    "ordered_witness_file_refs": [asdict(item) for item in predecessor_refs],
                    "ordered_evidence_file_refs": [asdict(item) for item in evidence_refs],
                    "physical_inventory": [asdict(item) for item in inventory],
                    "permission_acceptance_emitted": False,
                }
            )
            index_bytes = canonical_handoff_record_bytes(
                index_record, kind="index", expected_plan=plan
            )
            index_write = _publish_witness_once(
                plan=plan, slot_id=WITNESS_FILENAMES[4], data=index_bytes,
                authority=authority, witness_sink=witness_sink,
            )
            writes.append(index_write)
            if index_write.status != "retained":
                if index_write.failure is not None:
                    failures.append(index_write.failure)
                return HandoffRunResult(
                    invocation_id, "incomplete", observed, original, reads,
                    tuple(writes), tuple(failures), False,
                )

            source_vector_hash = sha256_bytes(
                canonical_json_bytes([asdict(item) for item in plan.source_files])
            )
            receipt_record = _seal(
                {
                    **_common(plan, "receipt"),
                    "index_file_bytes": len(index_bytes),
                    "index_file_sha256": sha256_bytes(index_bytes),
                    "index_commitment": handoff_record_commitment(
                        index_record, kind="index", expected_plan=plan
                    ),
                    "specification_file_sha256": plan.specification_file_sha256,
                    "handoff_source_vector_sha256": source_vector_hash,
                    "disposition": "permission_handoff_evidence_only",
                    "permission_acceptance_emitted": False,
                }
            )
            receipt_bytes = canonical_handoff_record_bytes(
                receipt_record, kind="receipt", expected_plan=plan
            )
            receipt_write = _publish_witness_once(
                plan=plan, slot_id=WITNESS_FILENAMES[5], data=receipt_bytes,
                authority=authority, witness_sink=witness_sink,
            )
            writes.append(receipt_write)
            if receipt_write.failure is not None:
                failures.append(receipt_write.failure)
            status: Literal["recorded", "incomplete"] = (
                "recorded" if receipt_write.status == "retained" else "incomplete"
            )
            return HandoffRunResult(
                invocation_id, status, observed, original, reads,
                tuple(writes), tuple(failures), False,
            )
        except Exception as exc:
            failures.append(HandoffFailure("write", "encode_witness", "raised", None, _exception_summary(exc)))
            return HandoffRunResult(invocation_id, "incomplete", observed, original, reads,
                                    tuple(writes), tuple(failures), False)
        finally:
            self._in_progress = False


def _validate_single_invocation(
    coordinator: PermissionHandoffCoordinator, invocation_id: str
) -> None:
    coordinator._validate_single_invocation(invocation_id)
