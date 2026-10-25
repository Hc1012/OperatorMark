"""Conditional E2 ports for the unchanged once-only Batch B supervisor."""
from dataclasses import dataclass
from threading import RLock
from . import state_execution as execution
from . import state_execution_authority as authority
from . import state_guard_capture as guards
from . import final_evidence_records as final_records
from . import state_evidence_windows as evidence
from . import state_provider_bindings as p


@dataclass(frozen=True, slots=True)
class _Command:
    executable: str
    arguments: tuple
    environment: tuple
    bootstrap_file: p.FileRef
    capsule_sha256: str
    capsule_bytes: int
    descriptor: int
    source_commitment: str
    inherited_handles: tuple


@dataclass(frozen=True, slots=True)
class _ExecutionConfiguration:
    prepared: authority.PreparedLaunch
    command: _Command
    reader_requests: tuple
    capture_writes: tuple
    exception_writes: tuple
    guard_expected: guards.GuardExpectations
    clock_observer_id: str
    clock_epoch_id: str
    valid_from: str
    valid_until: str
    cleanup_until: str
    maximum_capture_bytes: int
    transport_seconds: int
    cleanup_seconds: int


def validate_windows_execution_configuration(configuration):
    evidence._exact(configuration, _ExecutionConfiguration)
    prepared = configuration.prepared
    authority.reprepare(prepared, expected=prepared.expected, independent_inputs=prepared.independent_inputs)
    spec = authority.decode(prepared.expected.specification)
    command = configuration.command
    evidence._exact(command, _Command)
    p.text(command.executable)
    evidence._exact(command.arguments, tuple)
    evidence._exact(command.environment, tuple)
    evidence._exact(command.inherited_handles, tuple)
    evidence._need(all(type(x) is str for x in command.arguments), 'exact command arguments')
    for pair in command.environment:
        evidence._exact(pair, tuple)
        evidence._need(len(pair) == 2 and all(type(x) is str for x in pair), 'exact environment pairs')
    evidence._need(tuple(sorted(command.environment)) == command.environment and
                   len({x[0].casefold() for x in command.environment}) == len(command.environment), 'environment aliases/order')
    evidence._need(authority.sha(authority.canonical(dict(command.environment))) == spec['environment_sha256'],
                   'environment commitment mismatch')
    p.validate_file_ref(command.bootstrap_file)
    p.digest(command.capsule_sha256)
    p.integer(command.capsule_bytes)
    evidence._need(type(command.descriptor) is int and command.descriptor == 0, 'accepted inherited capsule descriptor zero')
    evidence._need(command.source_commitment == spec['source_binding_sha256'], 'command/source mismatch')
    evidence._need(command.inherited_handles and len(set(command.inherited_handles)) == len(command.inherited_handles)
                   and all(type(h) is int and h > 0 for h in command.inherited_handles), 'explicit inherited handle allowlist')
    command_bytes = authority.canonical(p.wire(command))
    evidence._need(authority.sha(command_bytes) == spec['command_binding_sha256'], 'command/capsule/bootstrap commitment')
    evidence._need((configuration.clock_observer_id, configuration.clock_epoch_id,
                   configuration.valid_from, configuration.valid_until, configuration.cleanup_until) ==
                   tuple(spec[k] for k in ('observer_id', 'clock_epoch_id', 'valid_from', 'valid_until', 'cleanup_until')),
                   'execution authority/clock substitution')
    for rows in (configuration.reader_requests, configuration.capture_writes, configuration.exception_writes):
        evidence._exact(rows, tuple)
    guards._expectations(configuration.guard_expected)
    for value in (configuration.maximum_capture_bytes, configuration.transport_seconds, configuration.cleanup_seconds):
        evidence._need(type(value) is int and value > 0, 'finite exact transport bounds')
    evidence._need(configuration.transport_seconds <= 600 and configuration.cleanup_seconds <= 30,
                   'execution/cleanup bounds exceed conditional scope')


@dataclass(frozen=True, slots=True)
class _ClockObservation:
    observer_id: str
    clock_epoch_id: str
    sequence: int
    observed_utc: str
    monotonic_ns: int


@dataclass(frozen=True, slots=True)
class _ClockBoundary:
    acquire: object


class WindowsClockPort:
    def __init__(self, *, configuration, boundary):
        validate_windows_execution_configuration(configuration)
        evidence._exact(boundary, _ClockBoundary)
        evidence._function(boundary.acquire)
        self.configuration, self.boundary = configuration, boundary
        self._previous = None
        self._lock = RLock()

    def _acquire(self):
        with self._lock:
            observed = self.boundary.acquire()
            evidence._exact(observed, _ClockObservation)
            p.integer(observed.sequence)
            p.integer(observed.monotonic_ns)
            p.utc(observed.observed_utc)
            c = self.configuration
            evidence._need((observed.observer_id, observed.clock_epoch_id) ==
                           (c.clock_observer_id, c.clock_epoch_id), 'clock/observer substitution')
            evidence._need(c.valid_from <= observed.observed_utc <= c.cleanup_until, 'clock outside admitted interval')
            if self._previous is not None:
                previous = self._previous
                evidence._need(observed.sequence > previous.sequence and observed.monotonic_ns >= previous.monotonic_ns
                               and observed.observed_utc >= previous.observed_utc, 'clock/sequence regression')
            self._previous = observed
            return observed

    def observe(self):
        observed = self._acquire()
        return dict(observer_id=observed.observer_id, clock_epoch_id=observed.clock_epoch_id,
                    sequence=observed.sequence, observed_utc=observed.observed_utc)


@dataclass(frozen=True, slots=True)
class _OwnedProcess:
    process_handle: int
    job_handle: int
    slot_id: str
    owner_file: p.FileRef
    creation_identity: bytes
    ownership_evidence_file: p.FileRef


@dataclass(frozen=True, slots=True)
class _OwnedObservation:
    handle: _OwnedProcess
    command: _Command
    job_assigned_before_resume: bool
    inherited_handles: tuple
    creation: execution.CreationObservation


@dataclass(frozen=True, slots=True)
class _TransportObservation:
    handle: _OwnedProcess
    stdout_chunks: tuple
    stderr_chunks: tuple
    drain_schedule: tuple
    closure: execution.ProcessClosure
    direct_exit_bytes: bytes | None
    primary_failure: evidence._Failure | None
    follow_on_failures: tuple


@dataclass(frozen=True, slots=True)
class _ProcessBoundary:
    create_suspended_owned: object
    inspect_owned: object
    resume: object
    drain_both_and_wait: object
    terminate_and_drain_owned: object


def validate_consumed_dispatch(prepared, *, fence, configuration):
    validate_windows_execution_configuration(configuration)
    evidence._exact(fence, evidence._DispatchFence)
    evidence._need(prepared == configuration.prepared == fence.prepared, 'dispatch/prepared substitution')
    with fence._lock:
        receipt = fence._receipt
        evidence._need(fence._spent and receipt is not None, 'unconsumed or unacknowledged dispatch')
        evidence._need(fence.boundary.authenticate(receipt) == receipt, 'durable dispatch authentication unavailable')
        spec = authority.decode(prepared.expected.specification)
        evidence._need(p.wire(receipt.ownership.owner_file) == spec['owner_binding_file'] and
                       receipt.ownership.store_id == fence.store_id, 'dispatch store/owner mismatch')
        return receipt


def validate_owned_process(handle, *, prepared, configuration):
    evidence._exact(handle, _OwnedProcess)
    evidence._need(prepared == configuration.prepared, 'owned process prepared binding')
    evidence._need(type(handle.process_handle) is int and handle.process_handle > 0 and
                   type(handle.job_handle) is int and handle.job_handle > 0, 'held process/job required')
    evidence._need(handle.slot_id == prepared.slot_id, 'owned slot substitution')
    p.validate_file_ref(handle.owner_file)
    p.validate_file_ref(handle.ownership_evidence_file)
    spec = authority.decode(prepared.expected.specification)
    evidence._need(p.wire(handle.owner_file) == spec['owner_binding_file'], 'process owner substitution')
    authority.process_identity(authority.decode(handle.creation_identity))


def validate_process_transport(result, *, prepared, configuration):
    evidence._exact(result, _TransportObservation)
    validate_owned_process(result.handle, prepared=prepared, configuration=configuration)
    evidence._exact(result.closure, execution.ProcessClosure)
    for chunks in (result.stdout_chunks, result.stderr_chunks):
        evidence._exact(chunks, tuple)
        evidence._need(all(type(chunk) is bytes for chunk in chunks), 'raw binary pipe chunks')
        evidence._need(sum(map(len, chunks)) <= configuration.maximum_capture_bytes, 'capture byte limit')
    evidence._exact(result.drain_schedule, tuple)
    indexes = {'stdout': [], 'stderr': []}
    for item in result.drain_schedule:
        evidence._exact(item, tuple)
        evidence._need(len(item) == 2 and item[0] in indexes and type(item[1]) is int,
                       'explicit concurrent stream schedule')
        indexes[item[0]].append(item[1])
    for key, chunks in (('stdout', result.stdout_chunks), ('stderr', result.stderr_chunks)):
        stream = getattr(result.closure, key)
        evidence._exact(stream, execution.StreamObservation)
        evidence._exact(stream.eof, bool)
        evidence._exact(stream.closed, bool)
        evidence._need(indexes[key] == list(range(len(chunks))), 'missing/duplicate/reordered byte chunks')
        evidence._need(stream.data is None and not chunks or stream.data == b''.join(chunks),
                       'stream bytes differ from raw transport')
        if stream.observation_file is not None:
            authority.ref(authority.decode(stream.observation_file))
    closure = result.closure
    evidence._need(closure.direct_exit_code is None or type(closure.direct_exit_code) is int, 'direct exit code')
    evidence._need(closure.reap_status in ('observed_reaped', 'unresolved', 'failed'), 'reaping observation')
    evidence._exact(closure.evidence_files, tuple)
    if closure.reap_status == 'observed_reaped':
        evidence._need(closure.direct_exit_code is not None and len(closure.evidence_files) == 1 and
                       result.direct_exit_bytes is not None, 'direct exit/reaping evidence unavailable')
    if result.direct_exit_bytes is not None:
        evidence._need(len(closure.evidence_files) == 1, 'exit observation reference')
        identity = authority.validate_process_observation(result.direct_exit_bytes,
            actual_file=authority.decode(closure.evidence_files[0]), prepared=prepared)
        row = authority.decode(result.direct_exit_bytes)
        evidence._need(identity == authority.decode(result.handle.creation_identity) and
                       row['direct_exit_code'] == closure.direct_exit_code and row['reap_status'] == closure.reap_status,
                       'direct exit differs from returned closure')
    evidence._failures(result.follow_on_failures)
    if result.primary_failure is not None:
        evidence._failures((result.primary_failure,))
    return result.closure


class WindowsProcessPort:
    def __init__(self, *, configuration, reader, fence, boundary, clock):
        validate_windows_execution_configuration(configuration)
        evidence._exact(reader, evidence.WindowsEvidenceReader)
        evidence._exact(fence, evidence._DispatchFence)
        evidence._exact(boundary, _ProcessBoundary)
        evidence._exact(clock, WindowsClockPort)
        for function in (boundary.create_suspended_owned, boundary.inspect_owned, boundary.resume,
                         boundary.drain_both_and_wait, boundary.terminate_and_drain_owned):
            evidence._function(function)
        self.configuration, self.reader, self.fence = configuration, reader, fence
        self.boundary, self.clock = boundary, clock
        self._lock = RLock()
        self._owned = {}
        self.failures = []
        self.transports = []

    def create_once(self, prepared):
        c = self.configuration
        validate_consumed_dispatch(prepared, fence=self.fence, configuration=c)
        observed = None
        try:
            now = self.clock._acquire()
            evidence._need(now.observed_utc <= c.valid_until, 'creation rights expired')
            receipt = self.fence.enter_creation(prepared)
            observed = self.boundary.create_suspended_owned(prepared, c.command, receipt)
            evidence._exact(observed, _OwnedObservation)
            validate_owned_process(observed.handle, prepared=prepared, configuration=c)
            # Save the exact held handle before any acknowledgment/retention or
            # resume validation can fail. PID equality never establishes ownership.
            with self._lock:
                self._owned[id(observed.handle)] = observed
            evidence._need(observed.command == c.command and observed.inherited_handles == c.command.inherited_handles
                           and observed.job_assigned_before_resume is True, 'command/inheritance/ownership-before-resume')
            evidence._exact(observed.creation, execution.CreationObservation)
            evidence._need(observed.creation.handle is observed.handle and
                           observed.creation.identity == observed.handle.creation_identity, 'creation identity/handle substitution')
            evidence._need(len(observed.creation.evidence_files) == 1, 'creation observation unavailable')
            reference = authority.decode(observed.creation.evidence_files[0])
            data = self.read(reference)
            evidence._need(data == observed.creation.observation_bytes, 'creation independent readback')
            authority.validate_process_observation(data, actual_file=reference, prepared=prepared,
                dispatch_file=authority.decode(receipt.dispatch_file))
            now = self.clock._acquire()
            evidence._need(now.observed_utc <= c.valid_until, 'creation/resume authority expired')
            evidence._need(self.boundary.resume(observed.handle, receipt) is None, 'resume acknowledgment unavailable')
            return observed.creation
        except BaseException as exc:
            self.failures.append(exc)
            if type(observed) is _OwnedObservation and id(observed.handle) in self._owned:
                creation = observed.creation
                return execution.CreationObservation('outcome_unavailable', 'started', observed.handle.creation_identity,
                    observed.handle, creation.evidence_files if type(creation) is execution.CreationObservation else (),
                    creation.observation_bytes if type(creation) is execution.CreationObservation else None)
            raise

    def owns(self, handle, prepared):
        validate_owned_process(handle, prepared=prepared, configuration=self.configuration)
        with self._lock:
            observed = self._owned.get(id(handle))
            if observed is None or observed.handle is not handle:
                return False
            actual = self.boundary.inspect_owned(handle, prepared)
            return type(actual) is _OwnedObservation and actual == observed and actual.handle is handle

    def read(self, reference):
        authority.ref(reference)
        requests = [r for r in self.configuration.reader_requests if
                    (r.address.root_binding_id, r.address.relative_path) ==
                    (reference['root_binding_id'], reference['relative_path'])]
        evidence._need(len(requests) == 1, 'process observation read scope')
        result = self.reader.acquire(requests[0])
        evidence._need(not result.diagnostics, 'process observation partial/unavailable')
        authority.check_bytes(reference, result.data)
        return result.data

    def _transport(self, handle, cleanup):
        c = self.configuration
        evidence._need(self.owns(handle, c.prepared), 'foreign process/job/slot; no cleanup or capture rights')
        start = self.clock._acquire()
        seconds = c.cleanup_seconds if cleanup else c.transport_seconds
        deadline = start.monotonic_ns + seconds * 1_000_000_000
        operation = self.boundary.terminate_and_drain_owned if cleanup else self.boundary.drain_both_and_wait
        # One deadline is shared by both pipes, the direct wait and every cleanup
        # operation. Boundaries must multiplex both streams under this deadline.
        result = operation(handle, deadline, c.maximum_capture_bytes)
        self.transports.append(result)
        end = self.clock._acquire()
        evidence._need(end.monotonic_ns <= deadline, 'aggregate owned transport/cleanup deadline')
        closure = validate_process_transport(result, prepared=c.prepared, configuration=c)
        if result.direct_exit_bytes is not None:
            evidence._need(self.read(authority.decode(closure.evidence_files[0])) == result.direct_exit_bytes,
                           'direct exit independent readback')
        return result

    def capture_and_reap(self, handle):
        return self._transport(handle, False).closure

    def cleanup_owned(self, handle):
        result = self._transport(handle, True)
        failures = tuple((f.stage, EvidenceBoundaryFailure(f)) for f in result.follow_on_failures)
        if result.primary_failure is not None:
            failures = ((result.primary_failure.stage, EvidenceBoundaryFailure(result.primary_failure)),) + failures
        return result.closure, failures


class EvidenceBoundaryFailure(RuntimeError):
    def __init__(self, fact):
        evidence._exact(fact, evidence._Failure)
        self.fact = fact
        super().__init__('retained native boundary failure; inspect fact')


class WindowsCapturePort:
    def __init__(self, *, configuration, reader, writer):
        validate_windows_execution_configuration(configuration)
        evidence._exact(reader, evidence.WindowsEvidenceReader)
        evidence._exact(writer, evidence.WindowsEvidenceWriter)
        self.configuration, self.reader, self.writer = configuration, reader, writer
        self.failures = []
        self.retention = []

    def _retain(self, request, data):
        result = self.writer.retain_exclusive(request, data)
        self.retention.append(result)
        evidence._need(result.creation_status == 'created' and result.flushed and result.held_readback == data and
                       result.publication is not None and result.publication.call_status == 'returned_success' and
                       result.publication.effect_status == 'expected_effect_observed' and not result.failures and not result.cleanup,
                       'capture retention incomplete; residue preserved')
        return authority.canonical(authority.file_ref(p.wire(request.final.address), data))

    def retain(self, slot, data):
        spec = authority.decode(self.configuration.prepared.expected.specification)
        evidence._need(type(slot) is dict and slot in spec['output_slots'], 'capture output slot substitution')
        rows = [r for r in self.configuration.capture_writes if r.key == slot['key'] and
                p.wire(r.final.address) == slot['address'] and r.accounting_owner == slot['owner']]
        evidence._need(len(rows) == 1, 'capture writer/destination/owner binding')
        return self._retain(rows[0], data)

    def read(self, reference):
        authority.ref(reference)
        requests = [r for r in self.configuration.reader_requests if p.wire(r.address) ==
                    {k: reference[k] for k in ('root_binding_id', 'relative_path')}]
        evidence._need(len(requests) == 1, 'capture independent read scope')
        result = self.reader.acquire(requests[0])
        evidence._need(not result.diagnostics, 'capture read unavailable')
        authority.check_bytes(reference, result.data)
        return result.data

    def retain_failures(self, failure_capture):
        evidence._exact(failure_capture, execution.FailureCapture)
        self.failures.append(failure_capture)
        # FailureCapture preserves the actual primary and arbitrary follow-on
        # stages. It is not relabeled as an exception graph of close failures.
        for stage, exc in failure_capture.follow_on:
            p.text(stage)
            evidence._need(isinstance(exc, BaseException), 'follow-on exception')
        rows = [r for r in self.configuration.exception_writes if r.key == 'failure_capture']
        evidence._need(len(rows) == 1, 'failure retention destination unavailable')
        facts = (() if failure_capture.primary is None else (evidence._failure('primary', failure_capture.primary),))
        facts += tuple(evidence._failure(stage, exc) for stage, exc in failure_capture.follow_on)
        # Raw exception descriptors are a retained product, not a new journal or
        # favorable final-evidence record. The accepted product policy binds it.
        data = authority.canonical(dict(failures=p.wire(facts), unavailable=list(failure_capture.unavailable)))
        return self._retain(rows[0], data)

    def retain_exception_bytes(self, key, data):
        rows = [r for r in self.configuration.exception_writes if r.key == key]
        evidence._need(len(rows) == 1, 'exception destination not independently accepted')
        reference = self._retain(rows[0], data)
        evidence._need(self.read(authority.decode(reference)) == data, 'exception independent readback')
        return reference


@dataclass(frozen=True, slots=True)
class _GuardBoundary:
    acquire: object


class WindowsGuardPort:
    def __init__(self, *, configuration, boundary):
        validate_windows_execution_configuration(configuration)
        evidence._exact(boundary, _GuardBoundary)
        evidence._function(boundary.acquire)
        self.configuration, self.boundary = configuration, boundary
        self.acquisitions = []

    def close_capture(self, prepared, identity, exit_code, reap):
        evidence._need(prepared == self.configuration.prepared, 'guard prepared binding')
        acquired = self.boundary.acquire(prepared, identity, exit_code, reap)
        self.acquisitions.append(acquired)
        evidence._exact(acquired, tuple)
        candidate = guards.derive_guard_capture(expected=self.configuration.guard_expected, independent_files=acquired)
        compared = guards.compare_guard_capture(expected=self.configuration.guard_expected,
                                                candidate=candidate, independent_files=acquired)
        evidence._need(compared.status == 'corroborated' and compared.closure is not None,
                       'guard acquisition unavailable or contradictory')
        final_records.validate_state_guard_closure(authority.decode(compared.closure))
        return compared.closure


@dataclass(frozen=True, slots=True)
class _StateProviders:
    process: WindowsProcessPort
    capture: WindowsCapturePort
    guard: WindowsGuardPort
    clock: WindowsClockPort


def build_windows_state_providers(*, configuration, evidence, claims, process_boundary, guard_boundary, clock_boundary):
    # No environment discovery, implicit enrollment, fake qualification label,
    # default native boundary, or production activation switch is accepted.
    from . import state_evidence_windows as ev
    validate_windows_execution_configuration(configuration)
    ev._exact(evidence, tuple)
    ev._need(len(evidence) == 2, 'explicit reader/writer pair required')
    reader, writer = evidence
    ev._exact(claims, ev.WindowsLaunchClaimStore)
    ev._need(claims.prepared == configuration.prepared, 'claim/provider prepared binding')
    clock = WindowsClockPort(configuration=configuration, boundary=clock_boundary)
    return _StateProviders(WindowsProcessPort(configuration=configuration, reader=reader, fence=claims.fence,
        boundary=process_boundary, clock=clock), WindowsCapturePort(configuration=configuration, reader=reader, writer=writer),
        WindowsGuardPort(configuration=configuration, boundary=guard_boundary), clock)
