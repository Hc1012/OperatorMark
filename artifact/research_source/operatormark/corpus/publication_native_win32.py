"""Scoped whole-directory operations behind lazy, injected native boundaries.

No DLL loading, process creation, enumeration, or namespace operation happens on
import or construction. Results describe operations; they never confer authority.
"""
from dataclasses import dataclass
from threading import RLock
from . import state_provider_bindings as p
from . import state_evidence_windows as evidence


@dataclass(frozen=True, slots=True)
class _HeldDirectory:
    handle: int
    identity: evidence._NativeIdentity
    custody_file: p.FileRef
    handle_evidence: p.FileRef


@dataclass(frozen=True, slots=True)
class _HeldAncestry:
    handles: tuple
    identities: tuple
    custody_files: tuple
    evidence_file: p.FileRef


@dataclass(frozen=True, slots=True)
class _NativePublicationRequest:
    operation_id: str
    operation: str
    source: evidence._Address
    destination: evidence._Address | None
    directory_identity: evidence._NativeIdentity
    directory_custody: p.FileRef
    source_ancestry: tuple
    destination_ancestry: tuple
    source_file: p.FileRef
    authority_file: p.FileRef
    accepted_capture: p.Capture
    accepted_addresses: tuple
    expected_children: tuple
    metadata_only: bool


def validate_native_publication_request(request):
    evidence._exact(request, _NativePublicationRequest)
    p.text(request.operation_id)
    evidence._need(request.operation in ('promote', 'quarantine', 'enumerate'), 'native operation scope')
    evidence._address(request.source)
    evidence._identity(request.directory_identity)
    for file in (request.directory_custody, request.source_file, request.authority_file):
        p.validate_file_ref(file)
    p.validate_capture(request.accepted_capture)
    evidence._exact(request.metadata_only, bool)
    evidence._need(request.metadata_only, 'namespace operations cannot open scientific streams')
    for rows in (request.source_ancestry, request.destination_ancestry, request.accepted_addresses,
                 request.expected_children):
        evidence._exact(rows, tuple)
    evidence._need(bool(request.source_ancestry), 'source ancestry missing')
    for identity in request.source_ancestry + request.destination_ancestry:
        evidence._identity(identity)
        evidence._need(identity.volume_serial_number_hex == request.directory_identity.volume_serial_number_hex,
                       'cross-volume operation refused')
    for address in request.accepted_addresses:
        evidence._address(address)
    evidence._need(request.source in request.accepted_addresses, 'source address not independently accepted')
    if request.operation == 'enumerate':
        evidence._need(request.destination is None and not request.destination_ancestry, 'enumeration destination forbidden')
    else:
        evidence._address(request.destination)
        evidence._need(request.destination in request.accepted_addresses and request.destination != request.source and
                       bool(request.destination_ancestry), 'destination ancestry/address scope')
    for child in request.expected_children:
        evidence._exact(child, _NamespaceEntry)
        _entry(child)


@dataclass(frozen=True, slots=True)
class _NamespaceEntry:
    name: str
    kind: str
    identity: evidence._NativeIdentity | None
    reparse: bool | None


def _entry(value):
    evidence._exact(value, _NamespaceEntry)
    p.relative(value.name)
    evidence._need('/' not in value.name, 'unauthorized namespace traversal')
    evidence._need(value.kind in ('file', 'directory', 'unknown'), 'entry kind')
    if value.identity is not None:
        evidence._identity(value.identity)
    evidence._need(value.reparse is None or type(value.reparse) is bool, 'entry reparse fact')


@dataclass(frozen=True, slots=True)
class _NamespaceObservation:
    entries: tuple
    complete: bool
    evidence_file: p.FileRef | None
    failures: tuple


@dataclass(frozen=True, slots=True)
class _MoveObservation:
    call_status: str
    effect_status: str
    evidence_file: p.FileRef | None
    failures: tuple


@dataclass(frozen=True, slots=True)
class _NativeResult:
    request: _NativePublicationRequest
    call_status: str
    effect_status: str
    before: evidence._ObjectFacts | None
    after: evidence._ObjectFacts | None
    enumeration_before: _NamespaceObservation | None
    enumeration_after: _NamespaceObservation | None
    operation: _MoveObservation | None
    failures: tuple
    diagnostics: tuple


@dataclass(frozen=True, slots=True)
class _NativeBindings:
    inspect: object
    inspect_ancestry: object
    destination_presence: object
    move_cross_parent_no_replace: object
    enumerate_metadata: object
    observe_clock: object


def _bindings(bindings):
    evidence._exact(bindings, _NativeBindings)
    for function in (bindings.inspect, bindings.inspect_ancestry, bindings.destination_presence,
                     bindings.move_cross_parent_no_replace, bindings.enumerate_metadata, bindings.observe_clock):
        evidence._function(function)


@dataclass(frozen=True, slots=True)
class _CapabilityReceipt:
    operation_id: str
    source: evidence._Address
    destination: evidence._Address | None
    authority_file: p.FileRef
    ownership_file: p.FileRef
    consumption_file: p.FileRef


class _ConsumedCapability:
    def __init__(self, *, receipt, authenticate, enter_once):
        evidence._exact(receipt, _CapabilityReceipt)
        evidence._function(authenticate)
        evidence._function(enter_once)
        for file in (receipt.authority_file, receipt.ownership_file, receipt.consumption_file):
            p.validate_file_ref(file)
        self.receipt, self.authenticate, self.enter_once = receipt, authenticate, enter_once
        self._entered = False
        self._lock = RLock()

    def enter(self, request):
        with self._lock:
            evidence._need(not self._entered, 'native dispatch cannot retry')
            row = self.receipt
            evidence._need((row.operation_id, row.source, row.destination, row.authority_file) ==
                           (request.operation_id, request.source, request.destination, request.authority_file),
                           'consumed capability binding mismatch')
            authenticated = self.authenticate(row)
            evidence._need(type(authenticated) is _CapabilityReceipt and authenticated == row,
                           'durable original ownership unavailable')
            self._entered = True
            evidence._need(self.enter_once(row) is row, 'durable native entry acknowledgment unavailable')


def _held(request, directory, ancestry, expected):
    evidence._exact(directory, _HeldDirectory)
    evidence._need(type(directory.handle) is int and directory.handle > 0, 'held directory handle')
    evidence._need((directory.identity, directory.custody_file) ==
                   (request.directory_identity, request.directory_custody), 'held directory substitution')
    p.validate_file_ref(directory.handle_evidence)
    evidence._exact(ancestry, _HeldAncestry)
    evidence._exact(ancestry.handles, tuple)
    evidence._exact(ancestry.custody_files, tuple)
    evidence._need(ancestry.identities == expected and len(ancestry.handles) == len(expected) ==
                   len(ancestry.custody_files), 'held ancestry substitution')
    evidence._need(all(type(handle) is int and handle > 0 for handle in ancestry.handles), 'ancestor handles')
    for file in ancestry.custody_files + (ancestry.evidence_file,):
        p.validate_file_ref(file)


def _clock(request, bindings):
    capture = bindings.observe_clock(request)
    p.validate_capture(capture)
    evidence._need(p.within(capture, request.accepted_capture), 'native operation outside authority cutoff')


def _observe(request, directory, ancestry, bindings, address):
    facts = bindings.inspect(directory, request)
    evidence._facts(facts)
    evidence._need(facts.presence == 'present' and facts.kind == 'directory' and facts.reparse is False and
                   facts.identity == request.directory_identity and facts.custody_file == request.directory_custody and
                   facts.address == address and facts.ancestry == ancestry.identities,
                   'current directory identity/custody/ancestry/address mismatch')
    evidence._need(bindings.inspect_ancestry(ancestry, request) == ancestry,
                   'held ancestry changed')
    return facts


def move_directory_no_replace(*, request, held_directory, source_ancestry,
                              destination_ancestry, capability, bindings):
    validate_native_publication_request(request)
    evidence._need(request.operation in ('promote', 'quarantine'), 'move operation required')
    _bindings(bindings)
    _held(request, held_directory, source_ancestry, request.source_ancestry)
    _held(request, held_directory, destination_ancestry, request.destination_ancestry)
    evidence._exact(capability, _ConsumedCapability)
    before = after = operation = None
    failures = []
    diagnostics = []
    call = 'not_called'
    effect = 'none_observed'
    stage = 'pre_dispatch'
    try:
        _clock(request, bindings)
        before = _observe(request, held_directory, source_ancestry, bindings, request.source)
        presence = bindings.destination_presence(destination_ancestry, request)
        evidence._exact(presence, evidence._ObjectFacts)
        evidence._facts(presence)
        evidence._need(presence.address == request.destination and presence.presence == 'absent',
                       'destination absence not independently established')
        capability.enter(request)
        _clock(request, bindings)
        stage = 'move_cross_parent_no_replace'
        # Set uncertain before dispatch; an exception does not prove no effect.
        call, effect = 'outcome_unavailable', 'unknown'
        operation = bindings.move_cross_parent_no_replace(held_directory, source_ancestry,
                                                         destination_ancestry, request, capability.receipt)
        evidence._exact(operation, _MoveObservation)
        evidence._need(operation.call_status in ('not_called', 'returned_success', 'returned_failure', 'outcome_unavailable') and
                       operation.effect_status in ('none_observed', 'expected_effect_observed', 'unexpected_effect_observed', 'unknown'),
                       'native call/effect enums')
        evidence._failures(operation.failures)
        failures.extend(operation.failures)
        call, effect = operation.call_status, operation.effect_status
    except BaseException as exc:
        failures.append(evidence._failure(stage, exc))
    finally:
        # Always take a fresh location observation. Earlier source location is
        # never installed as the current location after an uncertain operation.
        try:
            after = bindings.inspect(held_directory, request)
            evidence._facts(after)
            if after.identity != request.directory_identity or after.custody_file != request.directory_custody or after.kind != 'directory':
                diagnostics.append(('directory', 'identity_or_custody_mismatch'))
            if after.address not in (request.source, request.destination):
                diagnostics.append(('location', 'unknown_or_unexpected'))
            if call == 'returned_success' and after.address != request.destination:
                effect = 'unexpected_effect_observed'
                diagnostics.append(('move', 'postcondition_mismatch'))
        except BaseException as exc:
            after = None
            failures.append(evidence._failure('fresh_location', exc))
            diagnostics.append(('location', 'unavailable'))
    return _NativeResult(request, call, effect, before, after, None, None, operation,
                         tuple(failures), tuple(diagnostics))


def _enumeration(value):
    evidence._exact(value, _NamespaceObservation)
    evidence._exact(value.entries, tuple)
    evidence._exact(value.complete, bool)
    evidence._failures(value.failures)
    for entry in value.entries:
        _entry(entry)
    if value.evidence_file is not None:
        p.validate_file_ref(value.evidence_file)


def enumerate_held_namespace(*, request, held_directory, ancestry, bindings):
    validate_native_publication_request(request)
    evidence._need(request.operation == 'enumerate', 'enumeration scope required')
    _bindings(bindings)
    _held(request, held_directory, ancestry, request.source_ancestry)
    before = after = first = last = None
    failures = []
    diagnostics = []
    try:
        _clock(request, bindings)
        before = _observe(request, held_directory, ancestry, bindings, request.source)
        first = bindings.enumerate_metadata(held_directory, ancestry, request)
        _enumeration(first)
        last = bindings.enumerate_metadata(held_directory, ancestry, request)
        _enumeration(last)
        after = _observe(request, held_directory, ancestry, bindings, request.source)
        _clock(request, bindings)
        for label, enumeration in (('before', first), ('after', last)):
            names = [x.name.casefold() for x in enumeration.entries]
            identities = [x.identity for x in enumeration.entries if x.identity is not None]
            if len(set(names)) != len(names) or len(set(identities)) != len(identities):
                diagnostics.append((label, 'duplicate_entry_or_identity'))
            if not enumeration.complete or enumeration.evidence_file is None:
                diagnostics.append((label, 'enumeration_incomplete'))
            if any(x.reparse is not False or x.identity is None or x.kind == 'unknown' for x in enumeration.entries):
                diagnostics.append((label, 'entry_unavailable_or_reparse'))
            if enumeration.entries != request.expected_children:
                diagnostics.append((label, 'entry_population_mismatch'))
            failures.extend(enumeration.failures)
        if first.entries != last.entries or before != after:
            diagnostics.append(('namespace', 'unstable'))
    except BaseException as exc:
        failures.append(evidence._failure('metadata_enumeration', exc))
        diagnostics.append(('namespace', 'unavailable'))
    return _NativeResult(request, 'returned_failure' if failures else 'returned_success', 'none_observed',
                         before, after, first, last, None, tuple(failures), tuple(diagnostics))
