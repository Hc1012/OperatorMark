"""Conditional E2 held evidence adapters. Construction performs no acquisition.

Accepted premises are supplied independently. Boundaries have no default native
implementation, and neither an operation acknowledgment nor a self hash admits
evidence. Native enrollment and durability remain separate qualification work.
"""
from dataclasses import dataclass
from types import FunctionType, MethodType, MappingProxyType
from threading import RLock

from . import state_provider_bindings as e1
from . import state_execution_authority as launch
from . import publication_physical_records as publication
from . import corpus_retention as retention


class EvidenceError(ValueError):
    pass


def _need(condition, reason):
    if not condition:
        raise EvidenceError(reason)


def _exact(value, cls):
    _need(type(value) is cls, 'exact ' + cls.__name__ + ' required')


def _function(value):
    _need(type(value) in (FunctionType, MethodType), 'explicit injected function required')


@dataclass(frozen=True, slots=True)
class _NativeIdentity:
    volume_serial_number_hex: str
    file_id_128_hex: str
    legacy_volume_serial_number_hex: str
    legacy_file_index_hex: str


def _identity(value):
    _exact(value, _NativeIdentity)
    publication._identity(e1.wire(value))


@dataclass(frozen=True, slots=True)
class _Address:
    root_binding_id: str
    relative_path: str


def _address(value):
    _exact(value, _Address)
    e1.relative(value.root_binding_id)
    _need('/' not in value.root_binding_id, 'root token')
    e1.relative(value.relative_path)


@dataclass(frozen=True, slots=True)
class _Failure:
    stage: str
    exception_module: str
    exception_type: str
    native_code: int | None = None


def _failure(stage, exc):
    # No exception instance formatting, args access or user attribute hooks.
    cls = type(exc)
    if type(cls) is not type:
        return _Failure(stage, 'unavailable', 'unsafe_exception_metaclass')
    namespace = type.__getattribute__(cls, '__dict__')
    module = namespace.get('__module__', 'builtins')
    name = type.__getattribute__(cls, '__qualname__')
    if type(module) is not str or type(name) is not str:
        return _Failure(stage, 'unavailable', 'unsafe_exception_metadata')
    return _Failure(stage, module, name)


def _failures(rows):
    _exact(rows, tuple)
    for row in rows:
        _exact(row, _Failure)
        for value in (row.stage, row.exception_module, row.exception_type):
            e1.text(value)
        _need(row.native_code is None or type(row.native_code) is int, 'native failure code')


@dataclass(frozen=True, slots=True)
class _ObjectFacts:
    address: _Address | None
    identity: _NativeIdentity | None
    custody_file: e1.FileRef | None
    ancestry: tuple | None
    size: int | None
    streams: tuple | None
    descriptor_sha256: str | None
    presence: str
    reparse: bool | None
    kind: str | None = 'file'


def _facts(value):
    _exact(value, _ObjectFacts)
    _need(value.presence in ('present', 'absent', 'unknown'), 'presence enum')
    if value.address is not None:
        _address(value.address)
    if value.identity is not None:
        _identity(value.identity)
    if value.custody_file is not None:
        e1.validate_file_ref(value.custody_file)
    if value.ancestry is not None:
        _exact(value.ancestry, tuple)
        for item in value.ancestry:
            _identity(item)
        _need(len(set(value.ancestry)) == len(value.ancestry), 'ancestry identity alias')
    if value.size is not None:
        e1.integer(value.size)
    if value.streams is not None:
        _exact(value.streams, tuple)
        _need(all(type(x) is str for x in value.streams), 'stream names')
    if value.descriptor_sha256 is not None:
        e1.digest(value.descriptor_sha256)
    _need(value.reparse is None or type(value.reparse) is bool, 'reparse fact')
    _need(value.kind in ('file','directory',None), 'object kind fact')


@dataclass(frozen=True, slots=True)
class _ReadRequest:
    key: str
    role: str
    scope: str
    cutoff: str
    capture: e1.Capture
    address: _Address
    identity: _NativeIdentity
    custody_file: e1.FileRef
    ancestry: tuple
    descriptor_sha256: str
    source_files: tuple
    validator_files: tuple
    reference: e1.FileRef | None
    maximum_bytes: int


@dataclass(frozen=True, slots=True)
class _RetentionRequest:
    key: str
    final: _ReadRequest
    creating_address: _Address
    component: str
    accounting_owner: str
    writer_file: e1.FileRef
    authority_file: e1.FileRef
    lease_key: str
    creation_residue_rule: str


@dataclass(frozen=True, slots=True)
class _EvidenceConfiguration:
    provider: e1.ProviderExpectations
    reads: tuple
    writes: tuple
    accepted_upstream: tuple
    allowed_cutoffs: tuple


def validate_evidence_configuration(configuration):
    _exact(configuration, _EvidenceConfiguration)
    e1.validate_provider_expectations(configuration.provider)
    _exact(configuration.accepted_upstream, tuple)
    qualified = e1.qualify_provider_acquisition(expected=configuration.provider,
                                               acquisitions=configuration.accepted_upstream)
    _need(not qualified.diagnostics, 'independent provider premises unavailable or contradictory')
    _exact(configuration.reads, tuple)
    _exact(configuration.writes, tuple)
    _exact(configuration.allowed_cutoffs, tuple)
    _need(bool(configuration.allowed_cutoffs), 'explicit cutoff relationship required')
    for cutoff in configuration.allowed_cutoffs:
        e1.text(cutoff)
    seen = set()
    for request in configuration.reads:
        _validate_read_shape(request, configuration)
        key = (request.key, request.role, request.scope, request.cutoff)
        _need(key not in seen, 'duplicate accepted read')
        seen.add(key)
    addresses = set()
    for request in configuration.writes:
        _validate_write_shape(request, configuration)
        for address in (request.final.address, request.creating_address):
            key = (address.root_binding_id.casefold(), address.relative_path.casefold())
            _need(key not in addresses, 'aliased write destination')
            addresses.add(key)
    return qualified


def _validate_read_shape(request, configuration):
    _exact(request, _ReadRequest)
    for value in (request.key, request.role, request.scope, request.cutoff):
        e1.text(value)
    _need(request.cutoff in configuration.allowed_cutoffs, 'cutoff not independently accepted')
    e1.validate_capture(request.capture)
    _need(e1.within(request.capture, configuration.provider.capture), 'read cutoff outside admission')
    _address(request.address)
    _identity(request.identity)
    e1.validate_file_ref(request.custody_file)
    _exact(request.ancestry, tuple)
    _need(bool(request.ancestry), 'held ancestry required')
    for value in request.ancestry:
        _identity(value)
    e1.digest(request.descriptor_sha256)
    _need(request.source_files == configuration.provider.source_files, 'reader source substitution')
    _need(request.validator_files == configuration.provider.validator_files, 'reader validator substitution')
    _need(request.address.root_binding_id in tuple(r.root_binding_id for r in configuration.provider.roots),
          'unaccepted root')
    e1.integer(request.maximum_bytes)
    if request.reference is not None:
        e1.validate_file_ref(request.reference)
        _need((request.reference.root_binding_id, request.reference.relative_path) ==
              (request.address.root_binding_id, request.address.relative_path), 'reference address substitution')
        _need(request.reference.file_bytes <= request.maximum_bytes, 'read maximum')


def validate_acquisition_request(request, *, configuration):
    validate_evidence_configuration(configuration)
    _validate_read_shape(request, configuration)
    _need(request in configuration.reads, 'read role/scope/object not independently accepted')


def _validate_write_shape(request, configuration):
    _exact(request, _RetentionRequest)
    _validate_read_shape(request.final, configuration)
    _need(request.final in configuration.reads, 'independent destination reader required')
    _address(request.creating_address)
    _need(request.creating_address.root_binding_id == request.final.address.root_binding_id and
          request.creating_address != request.final.address, 'creating/final destination binding')
    _need(request.component in retention.COMPONENTS, 'catalog component')
    _need(request.accounting_owner in ('accepted_attempt', 'separate_B', 'external_preexisting', 'unresolved'),
          'explicit accounting owner')
    for value in (request.writer_file, request.authority_file):
        e1.validate_file_ref(value)
    _need(request.writer_file in configuration.provider.source_files, 'writer source binding')
    _need(request.authority_file == configuration.provider.authority_file, 'write authority substitution')
    for value in (request.key, request.lease_key, request.creation_residue_rule):
        e1.text(value)


def validate_retention_request(request, *, configuration):
    validate_evidence_configuration(configuration)
    _validate_write_shape(request, configuration)
    _need(request in configuration.writes, 'write policy/destination not independently accepted')


@dataclass(frozen=True, slots=True)
class _ReadBoundary:
    hold: object
    inspect: object
    read: object
    release: object


@dataclass(frozen=True, slots=True)
class _AcquiredEvidence:
    request: _ReadRequest
    data: bytes | None
    complete: bool
    before: _ObjectFacts | None
    after: _ObjectFacts | None
    failures: tuple
    cleanup: tuple
    diagnostics: tuple


def _object_diagnostics(facts, request, cut):
    if facts is None:
        return ((cut, 'facts_unavailable'),)
    _facts(facts)
    expected = dict(address=request.address, identity=request.identity, custody_file=request.custody_file,
                    ancestry=request.ancestry, descriptor_sha256=request.descriptor_sha256,
                    streams=('::$DATA',), reparse=False, presence='present', kind='file')
    diagnostics = []
    for name, wanted in expected.items():
        actual = getattr(facts, name)
        if actual != wanted:
            diagnostics.append((cut + '/' + name, 'unavailable' if actual is None or actual == 'unknown' else 'mismatch'))
    if facts.size is None:
        diagnostics.append((cut + '/size', 'unavailable'))
    elif facts.size > request.maximum_bytes:
        diagnostics.append((cut + '/size', 'maximum_exceeded'))
    return tuple(diagnostics)


def _acquisition_diagnostics(result):
    request = result.request
    rows = list(_object_diagnostics(result.before, request, 'before'))
    rows.extend(_object_diagnostics(result.after, request, 'after'))
    if result.before is not None and result.after is not None and result.before != result.after:
        rows.append(('object', 'post_read_instability'))
    if result.data is None:
        rows.append(('bytes', 'unavailable'))
    else:
        if result.before is None or len(result.data) != result.before.size:
            rows.append(('bytes', 'length_incomplete'))
        if request.reference is not None and (len(result.data), e1.sha(result.data)) != (
                request.reference.file_bytes, request.reference.file_sha256):
            rows.append(('bytes', 'accepted_reference_mismatch'))
    if not result.complete:
        rows.append(('read', 'incomplete'))
    rows.extend((f.stage, 'native_failure') for f in result.failures)
    rows.extend((f.stage, 'cleanup_failure') for f in result.cleanup)
    return tuple(rows)


def validate_acquired_evidence(result, *, request, configuration):
    validate_acquisition_request(request, configuration=configuration)
    _exact(result, _AcquiredEvidence)
    _need(result.request == request, 'acquisition request substitution')
    _need(result.data is None or type(result.data) is bytes, 'immutable bytes or unavailable')
    _exact(result.complete, bool)
    _failures(result.failures)
    _failures(result.cleanup)
    _exact(result.diagnostics, tuple)
    _need(result.diagnostics == _acquisition_diagnostics(result), 'acquisition diagnostics tampering')
    return result.diagnostics


class WindowsEvidenceReader:
    def __init__(self, *, configuration, boundary):
        validate_evidence_configuration(configuration)
        _need(configuration.provider.provider_role == 'evidence_reader', 'role-specific reader authority required')
        _exact(boundary, _ReadBoundary)
        for function in (boundary.hold, boundary.inspect, boundary.read, boundary.release):
            _function(function)
        self.configuration = configuration
        self.boundary = boundary

    def acquire(self, request):
        validate_acquisition_request(request, configuration=self.configuration)
        held = None
        before = after = data = None
        failures = []
        cleanup = []
        complete = False
        stage = 'hold'
        try:
            held = self.boundary.hold(request)
            _need(held is not None, 'held object unavailable')
            stage = 'inspect_before'
            observed = self.boundary.inspect(held, request)
            _facts(observed)
            before = observed
            # Never read bytes through a detected wrong object or stream scope.
            if not _object_diagnostics(before, request, 'before'):
                stage = 'read_unnamed_stream'
                data = b''
                while len(data) < before.size:
                    chunk = self.boundary.read(held, min(65536, before.size - len(data)))
                    _exact(chunk, bytes)
                    _need(len(chunk) <= before.size - len(data), 'read exceeded authorized size')
                    if not chunk:
                        break
                    data += chunk
                complete = len(data) == before.size
        except BaseException as exc:
            failures.append(_failure(stage, exc))
        finally:
            if held is not None:
                try:
                    observed = self.boundary.inspect(held, request)
                    _facts(observed)
                    after = observed
                except BaseException as exc:
                    failures.append(_failure('inspect_after', exc))
                try:
                    self.boundary.release(held)
                except BaseException as exc:
                    cleanup.append(_failure('release_read', exc))
        result = _AcquiredEvidence(request, data, complete, before, after, tuple(failures), tuple(cleanup), ())
        result = _AcquiredEvidence(request, data, complete, before, after, tuple(failures), tuple(cleanup),
                                   _acquisition_diagnostics(result))
        validate_acquired_evidence(result, request=request, configuration=self.configuration)
        return result


def _reference(result):
    _need(result.data is not None and not result.diagnostics, 'complete qualified acquisition required')
    return e1.FileRef(result.request.address.root_binding_id, result.request.address.relative_path,
                      len(result.data), e1.sha(result.data))


def to_e1_acquisition(result, *, admission, configuration):
    validate_acquired_evidence(result, request=result.request, configuration=configuration)
    e1.validate_admission(admission)
    request = result.request
    _need((admission.role, admission.scope, admission.validator_sources) ==
          (request.role, request.scope, request.validator_files), 'independent E1 admission binding')
    _need(e1.within(request.capture, admission.capture), 'E1 containment')
    _need((request.address.root_binding_id, request.address.relative_path) ==
          (admission.file.root_binding_id, admission.file.relative_path), 'independent E1 address commitment')
    _need(request.reference is None or request.reference == admission.file, 'independent E1 content commitment')
    if result.data is not None and not result.diagnostics:
        _need(_reference(result) == admission.file, 'independently admitted retained bytes')
    adverse = any(reason in ('mismatch', 'accepted_reference_mismatch', 'post_read_instability')
                  for _, reason in result.diagnostics)
    outcome = 'rejected' if adverse else 'unavailable' if result.diagnostics else 'accepted'
    return e1.ProviderAcquisition(request.role, admission.file, result.data, request.validator_files,
                                  request.scope, request.capture, outcome)


def to_launch_acquisition(result, *, acceptance, configuration):
    validate_acquired_evidence(result, request=result.request, configuration=configuration)
    _exact(acceptance, launch.Acquisition)
    request = result.request
    ref = launch.decode(acceptance.file)
    validator = launch.decode(acceptance.validator_file)
    launch.ref(ref)
    launch.ref(validator)
    _need(request.reference is not None and e1.wire(request.reference) == ref, 'independent launch reference')
    _need(validator in tuple(e1.wire(f) for f in request.validator_files), 'launch validator admission')
    _need(acceptance.scope == request.scope and acceptance.status == 'accepted', 'launch acceptance scope')
    _need(acceptance.first_utc <= request.capture.started_utc <= request.capture.finished_utc <= acceptance.last_utc,
          'Batch B admission must cover acquisition')
    _need(acceptance.data == result.data, 'launch independent bytes')
    _need(not result.diagnostics, 'unavailable/contradictory acquisition cannot qualify launch')
    launch.qualify(ref, (acceptance,), validators=(acceptance.validator_file,), scope=request.scope,
                   first=request.capture.started_utc, last=request.capture.finished_utc)
    return launch.Acquisition(acceptance.file, result.data, acceptance.validator_file, acceptance.scope,
                              acceptance.first_utc, acceptance.last_utc, acceptance.status)


def to_publication_envelopes(results, *, accepted_roles, configuration):
    _exact(results, tuple)
    _exact(accepted_roles, tuple)
    _need(len(results) == len(accepted_roles), 'independent publication role population')
    output = []
    for result, accepted in zip(results, accepted_roles):
        _exact(accepted, tuple)
        _need(len(accepted) == 2, 'accepted role and reference required')
        role, reference = accepted
        e1.text(role)
        e1.validate_file_ref(reference)
        validate_acquired_evidence(result, request=result.request, configuration=configuration)
        _need((role, reference) == (result.request.role, result.request.reference), 'publication role substitution')
        publication._ref(e1.wire(reference), positive=False)
        _need(not result.diagnostics, 'publication acquisition incomplete or contradictory')
        _need(_reference(result) == reference, 'publication content substitution')
        output.append(publication._freeze(dict(role=role, file=e1.wire(reference), data=result.data)))
    _need(len({row['role'] for row in output}) == len(output), 'duplicate publication role')
    return tuple(output)


@dataclass(frozen=True, slots=True)
class _SuppliedAuthzBoundary:
    query_token_statistics: object
    initialize_resource_manager: object
    initialize_context_from_token: object
    access_check: object
    free_context: object
    free_resource_manager: object


def _evaluate_independently_bound_authz(*, request, accepted_token_identity, accepted_object_binding,
                                       accepted_descriptor_sha256, bindings):
    """Narrow supplied-token calculation; never substitutes for retained authority.

    This conditional calculation distinguishes a valid denied reply from failed
    acquisition. It emits no enrollment, permission acceptance or launch right.
    Durable Authz/handoff evidence, if supplied by an integrator, still requires
    the existing corresponding retained-evidence verifier and independent inputs.
    """
    from . import permission_principal_authz_windows as authz
    _exact(request, authz.SuppliedTokenAccessRequest)
    _exact(bindings, _SuppliedAuthzBoundary)
    for function in (bindings.query_token_statistics,bindings.initialize_resource_manager,
                     bindings.initialize_context_from_token,bindings.access_check,bindings.free_context,bindings.free_resource_manager):
        _function(function)
    _need(request.expected_token_identity == accepted_token_identity and
          request.object_binding == accepted_object_binding and
          request.descriptor.self_relative_descriptor_sha256 == accepted_descriptor_sha256,
          'independent token/object/descriptor substitution')
    result = authz.evaluate_supplied_token_access(request, bindings=bindings)
    _need(result.permission_acceptance_emitted is False, 'Authz calculation cannot authorize execution')
    return result


@dataclass(frozen=True, slots=True)
class _Ownership:
    store_id: str
    lease_key: str
    reservation: e1.FileRef
    owner_file: e1.FileRef
    evidence_files: tuple


def _ownership(value):
    _exact(value, _Ownership)
    e1.text(value.store_id)
    e1.text(value.lease_key)
    e1.validate_file_ref(value.reservation)
    e1.validate_file_ref(value.owner_file)
    _exact(value.evidence_files, tuple)
    _need(bool(value.evidence_files), 'independent ownership evidence required')
    for ref in value.evidence_files:
        e1.validate_file_ref(ref)


@dataclass(frozen=True, slots=True)
class _CreateObservation:
    status: str
    held: object | None
    ownership: _Ownership | None
    residue: tuple
    failures: tuple


@dataclass(frozen=True, slots=True)
class _WriteBoundary:
    check_lease: object
    create_secured_exclusive: object
    write: object
    flush: object
    inspect: object
    read_held: object
    publish_no_replace: object
    release: object


@dataclass(frozen=True, slots=True)
class _LeaseObservation:
    lease_key: str
    authority_file: e1.FileRef
    request_key: str
    observed_capture: e1.Capture
    status: str
    evidence_file: e1.FileRef


@dataclass(frozen=True, slots=True)
class _PublicationObservation:
    call_status: str
    effect_status: str
    final_facts: _ObjectFacts | None
    residue: tuple
    failures: tuple


@dataclass(frozen=True, slots=True)
class _RetainedEvidence:
    request: _RetentionRequest
    creation_status: str
    write_bytes: int
    flushed: bool
    held_readback: bytes | None
    publication: _PublicationObservation | None
    ownership: _Ownership | None
    residue: tuple
    failures: tuple
    cleanup: tuple


class WindowsEvidenceWriter:
    def __init__(self, *, configuration, boundary):
        validate_evidence_configuration(configuration)
        _need(configuration.provider.provider_role == 'evidence_writer', 'role-specific writer authority required')
        _exact(boundary, _WriteBoundary)
        for function in (boundary.check_lease, boundary.create_secured_exclusive, boundary.write,
                         boundary.flush, boundary.inspect, boundary.read_held,
                         boundary.publish_no_replace, boundary.release):
            _function(function)
        self.configuration = configuration
        self.boundary = boundary

    def retain_exclusive(self, request, data):
        validate_retention_request(request, configuration=self.configuration)
        _exact(data, bytes)
        _need(len(data) <= request.final.maximum_bytes, 'retention maximum')
        if request.final.reference is not None:
            _need((len(data), e1.sha(data)) == (request.final.reference.file_bytes,
                  request.final.reference.file_sha256), 'retention content substitution')
        held = ownership = readback = published = None
        creation = 'unknown'
        count = 0
        flushed = False
        failures = []
        cleanup = []
        residue = []
        stage = 'lease'
        try:
            lease = self.boundary.check_lease(request)
            _exact(lease, _LeaseObservation)
            e1.validate_capture(lease.observed_capture)
            e1.validate_file_ref(lease.evidence_file)
            _need((lease.lease_key, lease.authority_file, lease.request_key, lease.status) ==
                  (request.lease_key, request.authority_file, request.key, 'active'), 'active write lease unavailable')
            _need(e1.within(lease.observed_capture, request.final.capture), 'write lease cutoff')
            stage = 'create_secured_exclusive'
            created = self.boundary.create_secured_exclusive(request, lease, data)
            _exact(created, _CreateObservation)
            _need(created.status in ('created', 'existing', 'refused', 'unknown'), 'creation result')
            _failures(created.failures)
            _exact(created.residue, tuple)
            for address in created.residue:
                _address(address)
            creation = created.status
            held = created.held
            residue.extend(created.residue)
            failures.extend(created.failures)
            if created.ownership is not None:
                _ownership(created.ownership)
                ownership = created.ownership
            _need(creation == 'created' and held is not None and ownership is not None and not failures,
                  'exclusive creation not established')
            _need(ownership.lease_key == request.lease_key, 'creation lease substitution')
            # An accounted creating object survives every subsequent failure.
            if request.creating_address not in residue:
                residue.append(request.creating_address)
            stage = 'inspect_created'
            before = self.boundary.inspect(held, request)
            _facts(before)
            _need(before.address == request.creating_address and before.identity == request.final.identity and
                  before.custody_file == request.final.custody_file and before.ancestry == request.final.ancestry and
                  before.descriptor_sha256 == request.final.descriptor_sha256 and
                  before.streams == ('::$DATA',) and before.reparse is False and before.presence == 'present',
                  'secured created-object binding')
            stage = 'write'
            while count < len(data):
                written = self.boundary.write(held, data[count:])
                _need(type(written) is int and 0 < written <= len(data) - count, 'partial/invalid write count')
                count += written
            stage = 'flush'
            _need(self.boundary.flush(held) is None, 'flush acknowledgment shape')
            flushed = True
            stage = 'held_readback'
            readback = self.boundary.read_held(held, len(data))
            _exact(readback, bytes)
            _need(readback == data, 'held readback mismatch')
            stage = 'inspect_written'
            after = self.boundary.inspect(held, request)
            _facts(after)
            _need(after == _ObjectFacts(before.address, before.identity, before.custody_file, before.ancestry,
                  len(data), before.streams, before.descriptor_sha256, before.presence, before.reparse),
                  'post-write custody/identity/size change')
            stage = 'publish_no_replace'
            published = self.boundary.publish_no_replace(held, request, ownership)
            _exact(published, _PublicationObservation)
            _need(published.call_status in publication._CALLS and published.effect_status in publication._EFFECTS,
                  'publication operation facts')
            _failures(published.failures)
            failures.extend(published.failures)
            _exact(published.residue, tuple)
            for address in published.residue:
                _address(address)
                if address not in residue:
                    residue.append(address)
            if published.final_facts is not None:
                _facts(published.final_facts)
                if published.final_facts.presence == 'present' and request.final.address not in residue:
                    residue.append(request.final.address)
            # Historical creating residue is deliberately retained in the result;
            # only fresh namespace acquisition can decide its current presence.
        except BaseException as exc:
            failures.append(_failure(stage, exc))
        finally:
            if held is not None:
                try:
                    self.boundary.release(held)
                except BaseException as exc:
                    cleanup.append(_failure('release_write', exc))
        return _RetainedEvidence(request, creation, count, flushed, readback, published,
                                 ownership, tuple(residue), tuple(failures), tuple(cleanup))


@dataclass(frozen=True, slots=True)
class _FenceReceipt:
    ownership: _Ownership
    reservation_file: bytes
    dispatch_file: bytes
    ownership_evidence_file: e1.FileRef
    status: str


@dataclass(frozen=True, slots=True)
class _FenceBoundary:
    consume: object
    authenticate: object
    enter_creation: object


class _DispatchFence:
    """Local state supplements independently supplied exclusive native fencing.

    Restart cannot recreate an original ownership object. Missing or lost consume
    acknowledgment blocks creation permanently in this instance; it never retries
    the durable boundary. The native boundary still requires separate qualification.
    """
    def __init__(self, *, store_id, prepared, boundary):
        e1.text(store_id)
        _exact(prepared, launch.PreparedLaunch)
        _exact(boundary, _FenceBoundary)
        for function in (boundary.consume, boundary.authenticate, boundary.enter_creation):
            _function(function)
        self.store_id = store_id
        self.prepared = prepared
        self.boundary = boundary
        self._lock = RLock()
        self._owner = None
        self._spent = False
        self._creation_entered = False
        self._receipt = None

    def register(self, ownership):
        _ownership(ownership)
        with self._lock:
            _need(self._owner is None and not self._spent, 'existing reservation cannot regain ownership')
            _need(ownership.store_id == self.store_id, 'store substitution')
            spec = launch.decode(self.prepared.expected.specification)
            _need(e1.wire(ownership.owner_file) == spec['owner_binding_file'], 'accepted owner substitution')
            _need((ownership.reservation.root_binding_id, ownership.reservation.relative_path) == tuple(
                launch.journal_address(self.prepared, 'reservation')[k] for k in ('root_binding_id', 'relative_path')),
                'reservation slot substitution')
            self._owner = ownership

    def consume(self, ownership, reservation_file, dispatch_file):
        with self._lock:
            _need(ownership is self._owner and not self._spent, 'ownership substitution or consumed reservation')
            self._spent = True
            _need(e1.wire(ownership.reservation) == launch.decode(reservation_file), 'reservation content binding')
            receipt = self.boundary.consume(ownership, reservation_file, dispatch_file)
            _exact(receipt, _FenceReceipt)
            _need(receipt.ownership is ownership and receipt.reservation_file == reservation_file and
                  receipt.dispatch_file == dispatch_file and receipt.status == 'consumed', 'consumption acknowledgment unavailable')
            e1.validate_file_ref(receipt.ownership_evidence_file)
            _need(self.boundary.authenticate(receipt) == receipt, 'durable consumption authentication')
            self._receipt = receipt
            return receipt

    def enter_creation(self, prepared):
        with self._lock:
            _need(type(prepared) is launch.PreparedLaunch and prepared == self.prepared,
                  'prepared launch substitution')
            _need(self._spent and self._receipt is not None and not self._creation_entered,
                  'unfenced, unknown or repeated creation')
            _need(self.boundary.authenticate(self._receipt) == self._receipt, 'durable fence unavailable')
            self._creation_entered = True
            result = self.boundary.enter_creation(self._receipt)
            _need(type(result) is _FenceReceipt and result == self._receipt, 'creation fence acknowledgment unavailable')
            return result


@dataclass(frozen=True, slots=True)
class _ClaimBoundary:
    claim_exclusive: object
    publish_exclusive: object


class WindowsLaunchClaimStore:
    def __init__(self, *, prepared, reader, boundary, fence, read_requests):
        _exact(prepared, launch.PreparedLaunch)
        launch.reprepare(prepared, expected=prepared.expected, independent_inputs=prepared.independent_inputs)
        _exact(reader, WindowsEvidenceReader)
        _exact(boundary, _ClaimBoundary)
        _function(boundary.claim_exclusive)
        _function(boundary.publish_exclusive)
        _exact(fence, _DispatchFence)
        _need(fence.prepared == prepared, 'claim/fence launch mismatch')
        _exact(read_requests, tuple)
        for request in read_requests:
            validate_acquisition_request(request, configuration=reader.configuration)
        self.prepared, self.reader, self.boundary = prepared, reader, boundary
        self.fence, self.read_requests = fence, read_requests

    def claim_exclusive(self, address, data, root):
        spec = launch.decode(self.prepared.expected.specification)
        _need(root == next(r for r in spec['roots'] if r['role'] == 'journal'), 'journal root substitution')
        _need(address == launch.journal_address(self.prepared, 'reservation'), 'claim address substitution')
        actual = launch.file_ref(address, data)
        _need(launch.validate_launch_record(launch.decode(data), prepared=self.prepared,
              actual_file=actual) == 'reservation', 'reservation record required')
        result = self.boundary.claim_exclusive(address, data, root)
        _exact(result, launch.ClaimObservation)
        _need(result.outcome in ('won', 'existing', 'refused', 'unknown'), 'claim outcome')
        _exact(result.evidence_files, tuple)
        for file in result.evidence_files:
            launch.ref(launch.decode(file))
        if result.actual_file is not None:
            launch.check_bytes(launch.decode(result.actual_file), result.retained_bytes)
            _need(self.read(launch.decode(result.actual_file)) == result.retained_bytes, 'independent claim readback')
        if result.outcome == 'won':
            _need(result.actual_file == launch.canonical(actual) and result.retained_bytes == data and
                  bool(result.evidence_files), 'winning claim binding')
            _ownership(result.ownership)
            _need(tuple(launch.canonical(e1.wire(f)) for f in result.ownership.evidence_files) ==
                  result.evidence_files, 'ownership evidence substitution')
            self.fence.register(result.ownership)
        else:
            _need(result.ownership is None, 'existing/refused/unknown claim cannot own')
        return result

    def read(self, reference):
        launch.ref(reference)
        requests = [r for r in self.read_requests if (r.address.root_binding_id, r.address.relative_path) ==
                    (reference['root_binding_id'], reference['relative_path'])]
        _need(len(requests) == 1, 'claim read scope unavailable or ambiguous')
        result = self.reader.acquire(requests[0])
        _need(not result.diagnostics and result.complete, 'claim read partial/unavailable')
        launch.check_bytes(reference, result.data)
        return result.data

    def publish_exclusive(self, address, data):
        actual = launch.file_ref(address, data)
        kind = launch.validate_launch_record(launch.decode(data), prepared=self.prepared, actual_file=actual)
        _need(kind != 'reservation', 'reservation only through exclusive claim')
        receipt = self.boundary.publish_exclusive(address, data)
        _need(type(receipt) is bytes and receipt == launch.canonical(actual), 'journal acknowledgment unavailable')
        return receipt

    def consume_once(self, ownership, reservation_file, dispatch_file):
        data = self.read(launch.decode(dispatch_file))
        record = launch.decode(data)
        _need(launch.validate_launch_record(record, prepared=self.prepared,
              actual_file=launch.decode(dispatch_file)) == 'dispatch', 'dispatch record required')
        _need(record['payload']['reservation_file'] == launch.decode(reservation_file), 'dispatch reservation mismatch')
        receipt = self.fence.consume(ownership, reservation_file, dispatch_file)
        return launch.canonical(dict(reservation_file=launch.decode(reservation_file),
            dispatch_file=launch.decode(dispatch_file), consumed=True,
            ownership_evidence_file=e1.wire(receipt.ownership_evidence_file)))


@dataclass(frozen=True, slots=True)
class _RetentionSnapshotRequest:
    catalog_request: _ReadRequest
    snapshot_requests: tuple
    evidence_requests: tuple
    accepted_catalog: bytes
    admitted_length_observations: tuple
    upstream_admissions: tuple
    metadata_namespace_addresses: tuple


@dataclass(frozen=True, slots=True)
class _RetentionNamespaceObservation:
    observations: tuple
    complete: bool
    failures: tuple


@dataclass(frozen=True, slots=True)
class _RetentionNamespace:
    enumerate_metadata: object


@dataclass(frozen=True, slots=True)
class _RetentionSnapshotAcquisition:
    catalog_bytes: bytes | None
    snapshot_bytes: tuple
    acquisitions: tuple
    namespace: _RetentionNamespaceObservation
    derivation: retention.RetentionDerivation | None
    diagnostics: tuple


def acquire_retention_snapshot(*, request, reader, namespace):
    """Acquire retained snapshots and metadata, never reopen scientific arrays.

    A newly serialized snapshot must first be retained and independently admitted
    by the caller. This function cannot admit its own future snapshot bytes.
    """
    _exact(request, _RetentionSnapshotRequest)
    _exact(reader, WindowsEvidenceReader)
    _exact(namespace, _RetentionNamespace)
    _function(namespace.enumerate_metadata)
    for rows in (request.snapshot_requests, request.evidence_requests, request.admitted_length_observations,
                 request.upstream_admissions, request.metadata_namespace_addresses):
        _exact(rows, tuple)
    _need(bool(request.snapshot_requests), 'explicit retained snapshot population')
    accepted_catalog = retention.strict_retention_loads(request.accepted_catalog)
    retention.validate_corpus_writer_catalog(accepted_catalog)
    for address in request.metadata_namespace_addresses:
        _address(address)
    requests = (request.catalog_request,) + request.snapshot_requests + request.evidence_requests
    _need(all(r.scope in ('retention_catalog', 'retention_snapshot', 'retention_evidence') for r in requests),
          'retention reads require evidence-only acquisition scope')
    _need(not any(r.address.relative_path.lower().endswith(('.npy', '.npz', '.pt', '.pth', '.ckpt')) for r in requests),
          'scientific/model members cannot be opened for inventory')
    acquired = tuple(reader.acquire(r) for r in requests)
    diagnostics = [(r.request.key + '/' + path, reason) for r in acquired for path, reason in r.diagnostics]
    observed = namespace.enumerate_metadata(request.metadata_namespace_addresses)
    _exact(observed, _RetentionNamespaceObservation)
    _exact(observed.observations, tuple)
    _exact(observed.complete, bool)
    _failures(observed.failures)
    if not observed.complete:
        diagnostics.append(('namespace', 'enumeration_incomplete'))
    for failure in observed.failures:
        diagnostics.append((failure.stage, 'namespace_failure'))
    catalog_data = acquired[0].data
    _need(catalog_data == request.accepted_catalog, 'independently fixed catalog substitution')
    snapshots = []
    for result in acquired[1:1 + len(request.snapshot_requests)]:
        if result.data is None or result.diagnostics:
            diagnostics.append((result.request.key, 'snapshot_unavailable'))
            continue
        record = retention.strict_retention_loads(result.data)
        retention.validate_corpus_retention_snapshot(record)
        snapshots.append(record)
    roots = retention._roots(accepted_catalog['root_requirements'])
    locations = []
    for raw in observed.observations:
        _exact(raw, bytes)
        row = launch.decode(raw)
        retention._address(row['address'])
        locations.append(retention._resolve(row['address'], roots).casefold())
    if len(set(locations)) != len(locations):
        diagnostics.append(('namespace', 'duplicate_physical_address'))
    if snapshots:
        expected_rows = {launch.canonical(row) for row in snapshots[-1]['observations']}
        observed_rows = set(observed.observations)
        for raw in observed_rows - expected_rows:
            diagnostics.append(('namespace/' + e1.sha(raw), 'unclassified_or_changed_entry'))
        if expected_rows - observed_rows:
            diagnostics.append(('namespace', 'missing_observation'))
    files = tuple(retention._freeze(dict(role=r.request.role, file=e1.wire(r.request.reference), data=r.data))
                  for r in acquired if r.request.reference is not None and not r.diagnostics)
    derivation = None
    try:
        derivation = retention.derive_retention_components(catalog=accepted_catalog, snapshots=tuple(snapshots),
            independent_files=files, admitted_length_observations=request.admitted_length_observations,
            upstream_admissions=request.upstream_admissions)
    except retention.RetentionContractError:
        diagnostics.append(('derivation', 'independent_accounting_unavailable'))
    return _RetentionSnapshotAcquisition(catalog_data, tuple(r.data for r in acquired[1:1 + len(request.snapshot_requests)]),
                                         acquired, observed, derivation, tuple(diagnostics))
