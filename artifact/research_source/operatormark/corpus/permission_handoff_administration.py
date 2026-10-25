"""Accepted-byte binding and private durable administrative authority.

Trusted custody, clock and control-plane providers are caller obligations. No
credential, root, current-user identity or permissive default is manufactured.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import os
from threading import RLock

from .canonical import canonical_json_bytes, sha256_bytes
from . import permission_approval_resolver as approval
from . import permission_handoff_records as old
from . import permission_handoff_administration_records as records
from .permission_handoff_capture import _exception_summary


@dataclass(frozen=True, slots=True)
class _BoundAdministration:
    pin: records.AcceptedAdministrativePin
    record: records.HandoffAdministrativePolicy | records.HandoffRestartTicket
    expected_plan: old.HandoffPlan
    deployment_source_files: tuple[old.HandoffSourceFile, ...]
    dependency_reads: tuple[records.HandoffRestartRead, ...]
    capture_policy: records.HandoffAdministrativePolicy | None


def _pin_input(pin):
    ref = pin.file_ref
    return records.HandoffRestartInput("accepted-" + pin.accepted_kind, "dependency", ref.root_binding_id, ref.relative_path,
        records.AdministrativePhysicalPolicy(pin.bootstrap_root, "", ref.relative_path.rpartition('/')[0],
            pin.bootstrap_file_identity, pin.bootstrap_file_security_descriptor_sha256, pin.host_profile_file_ref.file_sha256), ref, None)


def _dependency_input(dependency):
    return records.HandoffRestartInput(dependency.role, "dependency", dependency.file_ref.root_binding_id,
        dependency.file_ref.relative_path, dependency.physical_policy, dependency.file_ref, None)


@dataclass(frozen=True, slots=True)
class _BootstrapBinding:
    """Accepted measured bytes; dependency binding is still incomplete.

    Only fixed control-plane dependency reads are permitted. Authorities require
    completed _BoundAdministration, never this intermediate private value.
    """
    pin: records.AcceptedAdministrativePin
    record: records.HandoffAdministrativePolicy | records.HandoffRestartTicket
    expected_plan: old.HandoffPlan


def _bind_bootstrap(*, accepted_pin, acquired_record, expected_plan):
    _validate_pin(accepted_pin, expected_plan)
    records.validate_handoff_restart_read(acquired_record, input=_pin_input(accepted_pin), raw=True, operation="bootstrap_read")
    records._require(acquired_record.status == "complete", "bootstrap acquisition unavailable")
    content = old._validate_byte_record(acquired_record.data, name="bootstrap accepted bytes")
    value = records.strict_handoff_administrative_loads(content, kind=accepted_pin.accepted_kind, expected_plan=expected_plan)
    records._require(value.acceptance_reference == accepted_pin.acceptance_reference and value.host_profile_file_ref == accepted_pin.host_profile_file_ref
        and records.AccountingRootBinding(value.journal_policy.root_path, value.journal_policy.root_identity) == accepted_pin.accounting_root_binding,
        "bootstrap independent acceptance binding mismatch")
    return _BootstrapBinding(accepted_pin, value, expected_plan)


def _validate_pin(pin, plan):
    records._shape(pin, records.AcceptedAdministrativePin)
    old._text(pin.acceptance_reference, name="independent acceptance reference")
    old._validate_file_ref(pin.file_ref, name="accepted file pin")
    old._validate_file_ref(pin.host_profile_file_ref, name="accepted host profile")
    records._require(0 < pin.file_ref.file_bytes < records.windows.native.SIGNED_LARGE_INTEGER_MAX, "bootstrap pin length exceeds native bound")
    input = _pin_input(pin)
    records._physical(input.physical_policy, plan=plan, root_id=input.root_binding_id, relative=input.relative_path)
    records._shape(pin.accounting_root_binding, records.AccountingRootBinding)
    records._stat_identity(pin.accounting_root_binding.root_identity, directory=True)


def _required_dependencies(value, plan):
    binding = plan.authority_binding
    expected = [("controlling_authorization", binding.controlling_authorization_file_ref),
        ("enrollment", binding.enrollment_file_ref), ("accepted_roster", binding.accepted_roster_file_ref),
        ("host_profile", value.host_profile_file_ref),
        ("handoff_specification", next(x.expected_file_ref for x in plan.input_slots if x.slot_id == "handoff-specification")),
        ("handoff_plan", value.original_plan_file_ref)]
    if binding.profile_file_ref != value.host_profile_file_ref:
        expected.insert(3, ("capture_profile", binding.profile_file_ref))
    if type(value) is records.HandoffRestartTicket and value.capture_policy_file_ref is not None:
        expected.append(("capture_policy", value.capture_policy_file_ref))
    return tuple(expected)


def _bind(*, kind, accepted_pin, acquired_record, dependency_reads, expected_plan, expected_deployment_source_files):
    _validate_pin(accepted_pin, expected_plan)
    records._require(accepted_pin.accepted_kind == kind, "accepted pin kind mismatch")
    records.validate_handoff_restart_read(acquired_record, input=_pin_input(accepted_pin), raw=True, operation="bootstrap_read")
    records._require(acquired_record.status == "complete", "accepted administrative bootstrap unavailable")
    data = old._validate_byte_record(acquired_record.data, name="measured accepted administrative bytes")
    value = records.strict_handoff_administrative_loads(data, kind=kind, expected_plan=expected_plan)
    records._require(value.acceptance_reference == accepted_pin.acceptance_reference and value.host_profile_file_ref == accepted_pin.host_profile_file_ref,
        "accepted reference/host profile mismatch")
    records._require(records.AccountingRootBinding(value.journal_policy.root_path, value.journal_policy.root_identity) == accepted_pin.accounting_root_binding,
        "independent accounting root mismatch; replacement roots are not re-enrollment")
    records._require(type(expected_deployment_source_files) is tuple and value.deployment_source_files == expected_deployment_source_files,
        "independently supplied deployment sources mismatch")
    expected = _required_dependencies(value, expected_plan)
    required_roles = tuple(x[0] for x in expected)
    actual_roles = tuple(x.role for x in value.dependency_files)
    catalog_roles = ("observation_catalog", "authz_retention_catalog")
    records._require(actual_roles[:len(required_roles)] == required_roles and actual_roles[len(required_roles):] in ((), catalog_roles), "missing/dangling/reordered dependency roles")
    for dependency, (_, ref) in zip(value.dependency_files, expected):
        records._require(dependency.file_ref == ref, "dependency reference does not correspond to original contract")
    records._require(type(dependency_reads) is tuple and len(dependency_reads) == len(value.dependency_files), "dependency evidence coverage/order mismatch")
    decoded = {}
    enrollment = None
    capture_policy = None
    for dependency, read in zip(value.dependency_files, dependency_reads, strict=True):
        records.validate_handoff_restart_read(read, input=_dependency_input(dependency), raw=True, operation="bootstrap_read")
        records._require(read.status == "complete", "required administrative dependency unavailable")
        content = old._validate_byte_record(read.data, name=dependency.role)
        records._require(len(content) <= value.io_limits.maximum_opaque_input_bytes, "dependency size exceeds accepted limit")
        decoded[dependency.role] = content
        if dependency.role == "enrollment":
            enrollment = approval.strict_permission_approval_enrollment_loads(content)
            records._require(enrollment.expected_context == expected_plan.expected_request.context
                and enrollment.expected_source_binding == expected_plan.expected_request.source_binding,
                "original enrollment request/source mismatch")
        elif dependency.role == "handoff_plan":
            records._require(content == canonical_json_bytes(old._wire(expected_plan)) + b"\n", "original plan bytes changed")
        elif dependency.role == "capture_policy":
            capture_policy = records.strict_handoff_administrative_loads(content, kind="policy", expected_plan=expected_plan)
        elif dependency.role == "observation_catalog":
            records._require(enrollment is not None, "catalog before original enrollment")
            catalog = approval.strict_permission_grant_catalog_loads(content, enrollment=enrollment)
            records._require(all(grant == expected_plan.expected_grants.get(grant.expected_scope.operation) for grant in catalog),
                "observation catalog grants differ from original plan")
            decoded[dependency.role] = catalog
        elif dependency.role == "authz_retention_catalog":
            records._require(enrollment is not None, "catalog before original enrollment")
            catalog = approval.strict_permission_authz_retention_catalog_loads(content, enrollment=enrollment)
            records._require(all(grant == expected_plan.expected_grants.get(grant.expected_scope.operation) for grant in catalog),
                "Authz retention catalog grants differ from original plan")
            decoded[dependency.role] = catalog
    if kind == "restart_ticket":
        records.validate_handoff_restart_ticket(value, expected_plan=expected_plan, capture_policy=capture_policy)
        records._require(value.verification_mode != "administrative_corroboration" or capture_policy is not None,
            "accepted capture policy required for administrative corroboration")
    return _BoundAdministration(accepted_pin, value, expected_plan, expected_deployment_source_files, dependency_reads, capture_policy)


def bind_accepted_handoff_policy(*, accepted_pin, acquired_record, dependency_reads, expected_plan, expected_deployment_source_files):
    return _bind(kind="policy", accepted_pin=accepted_pin, acquired_record=acquired_record, dependency_reads=dependency_reads,
        expected_plan=expected_plan, expected_deployment_source_files=expected_deployment_source_files)


def bind_accepted_restart_ticket(*, accepted_pin, acquired_record, dependency_reads, expected_plan, expected_deployment_source_files):
    return _bind(kind="restart_ticket", accepted_pin=accepted_pin, acquired_record=acquired_record, dependency_reads=dependency_reads,
        expected_plan=expected_plan, expected_deployment_source_files=expected_deployment_source_files)


@dataclass(frozen=True, slots=True)
class _Lease:
    scope: old.HandoffLeaseScope | records.HandoffRestartLeaseScope


@dataclass(frozen=True, slots=True)
class RestartAcquireResult:
    status: str
    lease: _Lease | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class RestartCheckResult:
    status: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class RestartFinishResult:
    status: str
    effect_certainty: str
    recorded_outcome: str | None


@dataclass
class _LeaseState:
    lease: _Lease
    grant: records.HandoffAdministrativeGrant
    snapshot: old.HandoffLeaseScope | records.HandoffRestartLeaseScope
    checkpoints: list
    references: list
    inactive: bool = False
    dispatched: bool = False
    finished: bool = False


def _restart_scope(ticket, grant):
    scope = records.HandoffRestartLeaseScope(ticket.record_self_sha256, ticket.verification_id,
        grant.original_plan_self_sha256, grant.request_self_sha256, grant.full_state4_authorization_file_sha256,
        grant.operation, grant.slot_id, grant.root_binding_id, grant.relative_path, records._hash(grant.physical_policy), grant.allowance_id, "0" * 64)
    payload = old._wire(scope)
    payload.pop("scope_self_sha256")
    return replace(scope, scope_self_sha256=records._hash(payload))


class _DurableAuthority:
    restart = False

    def __init__(self, *, bound, journal, utc_ns_clock, revocation_reader):
        records._require(type(bound) is _BoundAdministration, "authority requires independently bound inputs")
        expected = records.HandoffRestartTicket if self.restart else records.HandoffAdministrativePolicy
        records._require(type(bound.record) is expected, "wrong administrative authority family")
        records._require(callable(utc_ns_clock) and callable(getattr(revocation_reader, "inspect_revocation_once", None)), "explicit clock/revocation capability required")
        for name in ("activate_once", "reserve_once", "record_checkpoint_once", "record_terminal_once"):
            records._require(callable(getattr(journal, name, None)), "explicit journal capability required")
        self.bound, self._journal, self._clock, self._revocation = bound, journal, utc_ns_clock, revocation_reader
        self._lock = RLock()
        self._busy = False
        self._activation_attempted = False
        self._activation = None
        self._session_id = None
        self._pid = os.getpid()
        self._attempted = set()
        self._leases = {}
        self._last_utc = None
        self._inactive = False
        self._events = []

    @property
    def evidence(self):
        return tuple(self._events)

    def _enter(self):
        records._require(not self._busy and self._pid == os.getpid(), "authority reentrancy/inherited process refused")
        self._busy = True

    def _common(self):
        p = self.bound.record
        return dict(purpose="restart" if self.restart else "capture", full_state4_authorization_file_sha256=p.full_state4_authorization_file_sha256,
            original_plan_self_sha256=p.original_plan_self_sha256, request_self_sha256=self.bound.expected_plan.expected_request.request_self_sha256,
            original_invocation_id=p.original_invocation_id, verification_id=p.verification_id if self.restart else None,
            accepted_administration_file_sha256=self.bound.pin.file_ref.file_sha256, record_self_sha256="0" * 64)

    def _record(self, kind, **kwargs):
        return records._seal(records._TYPES[kind](schema=records.PREFIX + kind + ".v1", **self._common(), **kwargs))

    def _clock_once(self):
        value = self._clock()
        self._events.append(("clock_return", value))
        old._exact_int(value, name="actual clock UTC")
        if self._last_utc is not None:
            records._require(value >= self._last_utc, "clock rollback")
        self._last_utc = value
        return value

    def _decision(self, grant=None):
        first = final = None
        observations = ()
        status, reason = "failed", "unavailable"
        interval = self.bound.record if grant is None else grant
        try:
            first = self._clock_once()
            observed = self._revocation.inspect_revocation_once(self.bound, None if grant is None else grant.allowance_id)
            self._events.append(("revocation_return", observed))
            records._require(type(observed) is tuple, "malformed revocation observations")
            leaves = records._revocation_leaves(self.bound.record, self.bound.pin.file_ref.file_sha256, None if grant is None else grant.allowance_id)
            for item in observed:
                records._shape(item, records.AdministrativeRevocationObservation)
            records._require(tuple(x.marker_relative_path for x in observed) == leaves, "revocation coverage/order mismatch")
            observations = observed
            final = self._clock_once()
            if self._inactive:
                status, reason = "failed", "unavailable"
            elif any(x.status == "present" for x in observations):
                status, reason = "refused", "refused"
            elif any(x.status == "unavailable" for x in observations):
                status, reason = "failed", "unavailable"
            elif not interval.not_before_utc_ns <= first <= final < interval.expires_at_utc_ns:
                status, reason = "refused", "expired"
            else:
                status, reason = "active", None
        except BaseException as error:
            self._events.append(("decision_raised", error, _exception_summary(error), first, final, observations))
        if status != "active":
            self._inactive = True
        return status, reason, first, final, observations

    def _ack(self, actual, expected):
        self._events.append(("journal_return", expected, actual))
        records._shape(actual, records.AdministrativeJournalAck)
        expected_bytes = canonical_json_bytes(old._wire(expected)) + b"\n"
        records._require(actual.status == "recorded" and actual.effect_certainty == "confirmed" and actual.record is not None
            and old._validate_byte_record(actual.record, name="actual journal acknowledgment") == expected_bytes
            and actual.measured_file_identity is not None and not actual.cleanup_failures and actual.failure is None,
            "journal did not confirm a new complete matching write")
        records._stat_identity(actual.measured_file_identity, directory=False)
        wanted = expected.requested_outcome if type(expected) is records.HandoffAdministrationTerminal else None
        records._require(actual.recorded_outcome == wanted, "actual journal outcome mismatch")
        return actual

    def activate_once(self):
        with self._lock:
            self._enter()
            try:
                records._require(not self._activation_attempted, "activation already attempted")
                self._activation_attempted = True
                status, reason, first, final, observations = self._decision()
                if status != "active":
                    return self._check_type()(status, reason)
                p = self.bound.record
                record = self._record("activation", activation_id=records._activation_id(p, self.bound.expected_plan.expected_request.request_self_sha256),
                    journal_policy_sha256=records._hash(p.journal_policy), observed_utc_ns=final)
                ack = self._ack(self._journal.activate_once(record), record)
                self._activation = record
                self._session_id = records._session_id(old._validate_byte_record(ack.record, name="activation acknowledgment"), p.journal_policy.root_identity)
                return self._check_type()("active", None)
            except BaseException as error:
                self._events.append(("activation_failed", error))
                self._inactive = True
                return self._check_type()("failed", "unavailable")
            finally:
                self._busy = False

    def _acquire_type(self): return RestartAcquireResult if self.restart else old.HandoffAcquireResult
    def _check_type(self): return RestartCheckResult if self.restart else old.HandoffCheckpointResult
    def _finish_type(self): return RestartFinishResult if self.restart else old.HandoffFinishResult

    def _acquire(self, value, operation, slot_id):
        with self._lock:
            self._enter()
            key = (operation, slot_id)
            try:
                records._require(key not in self._attempted, "allowance already locally attempted")
                self._attempted.add(key)
                p = self.bound.record
                expected_value = p if self.restart else self.bound.expected_plan
                records._require(type(value) is type(expected_value) and value == expected_value, "acquire changed plan/ticket")
                records._require(self._activation is not None and not self._inactive, "authority not actively owned")
                grant = next((g for g in p.grants if (g.operation, g.slot_id) == key), None)
                if grant is None:
                    return self._acquire_type()("refused", None, "scope_mismatch")
                records.validate_handoff_administrative_grant(grant, expected_plan=self.bound.expected_plan, administration=p,
                    input=next(x for x in p.inputs if x.slot_id == slot_id) if self.restart else None)
                status, reason, first, final, observations = self._decision(grant)
                if status != "active":
                    return self._acquire_type()(status, None, reason)
                scope = _restart_scope(p, grant) if self.restart else records._legacy_scope(value, operation, slot_id)
                reservation = self._record("reservation", activation_id=self._activation.activation_id, session_id=self._session_id,
                    allowance_id=grant.allowance_id, grant_self_sha256=grant.record_self_sha256, operation=operation, slot_id=slot_id,
                    scope_self_sha256=scope.scope_self_sha256, observed_utc_ns=final)
                self._ack(self._journal.reserve_once(reservation), reservation)
                lease = _Lease(scope)
                self._leases[id(lease)] = _LeaseState(lease, grant, scope, [], [])
                return self._acquire_type()("granted", lease, None)
            except BaseException as error:
                self._events.append(("acquire_failed", key, error))
                return self._acquire_type()("failed", None, "unavailable")
            finally:
                self._busy = False

    def _lease_state(self, lease, *, finishing=False):
        state = self._leases.get(id(lease))
        records._require(state is not None and state.lease is lease and not state.finished, "lease not owned/finished")
        if type(lease) is not _Lease or lease.scope != state.snapshot:
            state.inactive = True
            records._require(finishing, "owned lease scope changed")
        return state

    def check_active(self, lease, checkpoint):
        with self._lock:
            self._enter()
            state = None
            try:
                state = self._lease_state(lease)
                names = records.WRITE_CHECKPOINTS if state.grant.operation == "retain_handoff_witness" else records.READ_CHECKPOINTS
                records._require(not state.inactive and len(state.checkpoints) < len(names) and checkpoint == names[len(state.checkpoints)], "inactive/out-of-order checkpoint")
                if self.restart and checkpoint == "final":
                    records._require(state.dispatched, "final checkpoint before actual dispatch claim")
                status, reason, first, final, observations = self._decision(state.grant)
                ordinal = len(state.checkpoints)
                record = self._record("checkpoint", activation_id=self._activation.activation_id, session_id=self._session_id,
                    allowance_id=state.grant.allowance_id, operation=state.grant.operation, slot_id=state.grant.slot_id,
                    scope_self_sha256=state.snapshot.scope_self_sha256, ordinal=ordinal, checkpoint=checkpoint,
                    status=status, reason=reason, first_utc_ns=first, final_utc_ns=final, revocation_observations=observations)
                state.checkpoints.append((checkpoint, status, reason))
                if status != "active": state.inactive = True
                ack = self._ack(self._journal.record_checkpoint_once(record), record)
                leaf = record.activation_id + "." + record.allowance_id + f".{ordinal}.checkpoint.json"
                state.references.append(old.FileRef(self.bound.record.journal_policy.root_path, leaf, ack.record.file_bytes, ack.record.file_sha256))
                return self._check_type()(status, reason)
            except BaseException as error:
                if state is not None: state.inactive = True
                self._events.append(("checkpoint_failed", checkpoint, error))
                return self._check_type()("failed", "unavailable")
            finally:
                self._busy = False

    def finish(self, lease, requested_outcome):
        with self._lock:
            self._enter()
            entered_backend = False
            try:
                state = self._lease_state(lease, finishing=True)
                state.finished = True
                records._require(requested_outcome in {"succeeded", "refused", "failed"}, "unknown terminal outcome")
                if requested_outcome == "succeeded":
                    names = records.WRITE_CHECKPOINTS if state.grant.operation == "retain_handoff_witness" else records.READ_CHECKPOINTS
                    records._require(not state.inactive and tuple(x[0] for x in state.checkpoints) == names
                        and all(x[1] == "active" for x in state.checkpoints) and len(state.references) == len(names), "success lacks persisted active sequence")
                record = self._record("terminal", activation_id=self._activation.activation_id, session_id=self._session_id,
                    allowance_id=state.grant.allowance_id, operation=state.grant.operation, slot_id=state.grant.slot_id,
                    scope_self_sha256=state.snapshot.scope_self_sha256, requested_outcome=requested_outcome,
                    checkpoint_record_refs=tuple(state.references))
                entered_backend = True
                self._ack(self._journal.record_terminal_once(record), record)
                return self._finish_type()("succeeded", "confirmed", requested_outcome)
            except BaseException as error:
                self._events.append(("finish_failed", requested_outcome, error, entered_backend))
                return self._finish_type()("failed", "unknown" if entered_backend else "not_applied", None)
            finally:
                self._busy = False


class DurableHandoffAuthority(_DurableAuthority):
    def acquire(self, plan, operation, slot_id):
        return self._acquire(plan, operation, slot_id)


class DurableHandoffRestartAuthority(_DurableAuthority):
    restart = True

    def acquire(self, ticket, slot_id):
        return self._acquire(ticket, "read_handoff_verification_input", slot_id)

    def claim_dispatch_once(self, lease, ticket, slot_id):
        with self._lock:
            self._enter()
            state = None
            try:
                state = self._lease_state(lease)
                records._require(ticket == self.bound.record and type(ticket) is records.HandoffRestartTicket
                    and state.grant.slot_id == slot_id and not state.inactive and not self._inactive and not state.dispatched
                    and tuple(x[0] for x in state.checkpoints) == records.READ_CHECKPOINTS[:-1]
                    and all(x[1] == "active" for x in state.checkpoints) and len(state.references) == 3, "restart dispatch lacks exact live lease/checkpoints")
                state.dispatched = True
            except BaseException:
                if state is not None: state.inactive = True
                raise
            finally:
                self._busy = False
