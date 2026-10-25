"""Private, conditional provider premises. No enrollment, I/O or execution rights."""
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
import hashlib
import json
import re
import unicodedata

_CONTROL_OR_SURROGATE = re.compile(r"[\x00-\x1f\x7f-\x9f\ud800-\udfff]")


class BindingError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise BindingError(message)


def text(value):
    require(type(value) is str and bool(value), 'exact nonempty text')
    require(unicodedata.normalize('NFC', value) == value and
            _CONTROL_OR_SURROGATE.search(value) is None, 'unsafe text')
    return value


def digest(value, length=64):
    require(type(value) is str and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None, 'digest')


def integer(value):
    require(type(value) is int and value >= 0, 'nonnegative exact integer')


def relative(value):
    text(value)
    require(not any(c in value for c in '\\:*?"<>|'), 'path alias')
    for part in value.split('/'):
        require(part not in ('', '.', '..') and part[-1] not in ' .', 'path component')
        require(re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])', part.split('.')[0]) is None, 'device path')


def utc(value):
    text(value)
    require(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z', value) is not None, 'UTC form')
    try:
        datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError as exc:
        raise BindingError('UTC value') from exc


def wire(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: wire(getattr(value, f.name)) for f in fields(value)}
    if type(value) in (tuple, list):
        return [wire(x) for x in value]
    if type(value) is dict:
        return {k: wire(v) for k, v in value.items()}
    require(value is None or type(value) in (str, int, bool), 'data-only value')
    if type(value) is str:
        text(value)
    return value


def encode(value):
    return (json.dumps(wire(value), sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n').encode('ascii')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate key')
        result[key] = value
    return result


def decode(data):
    require(type(data) is bytes, 'immutable bytes required')
    try:
        value = json.loads(data, object_pairs_hook=_pairs)
        require(encode(value) == data, 'noncanonical encoding')
        return value
    except (UnicodeError, ValueError, TypeError) as exc:
        raise BindingError('invalid canonical bytes') from exc


def keys(value, names):
    require(type(value) is dict and set(value) == set(names.split()), 'exact fields: ' + names)


def sha(data):
    require(type(data) is bytes, 'immutable bytes')
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class FileRef:
    root_binding_id: str
    relative_path: str
    file_bytes: int
    file_sha256: str


@dataclass(frozen=True)
class Capture:
    clock_epoch_id: str
    first_sequence: int
    last_sequence: int
    started_utc: str
    finished_utc: str


@dataclass(frozen=True)
class Binding:
    attempt_id: str
    state: str
    role: str
    authorization_sha256: str
    protocol_sha256: str
    implementation_commit: str
    implementation_tree: str
    implementation_parent: str
    source_sha256: str
    plan_file: FileRef


@dataclass(frozen=True)
class RootBinding:
    """Stationary namespace anchor, never the moving attempt directory."""
    root_binding_id: str
    canonical_path: str
    native_identity: str
    evidence_file: FileRef


@dataclass(frozen=True)
class OutputBinding:
    slot: str
    relative_path: str
    native_identity: str
    custody_file: FileRef
    content_file: FileRef
    introduced_phase: str


@dataclass(frozen=True)
class DirectoryLocation:
    root_binding_id: str
    relative_path: str


@dataclass(frozen=True)
class AttemptDirectoryBinding:
    native_identity: str
    custody_file: FileRef
    locations: tuple


@dataclass(frozen=True)
class Admission:
    role: str
    file: FileRef
    validator_sources: tuple
    scope: str
    capture: Capture
    outcome: str


@dataclass(frozen=True)
class ProviderAcquisition:
    role: str
    file: FileRef
    data: bytes | None
    validator_sources: tuple
    scope: str
    capture: Capture
    outcome: str


@dataclass(frozen=True)
class ProviderExpectations:
    binding: Binding
    provider_role: str
    source_files: tuple
    validator_files: tuple
    authority_file: FileRef
    qualification_file: FileRef
    custody_file: FileRef
    roots: tuple
    output_inventory: tuple
    valid_from: str
    valid_until: str
    cleanup_until: str
    capture: Capture
    admissions: tuple
    attempt_directory: AttemptDirectoryBinding


@dataclass(frozen=True)
class Qualification:
    checked: tuple
    diagnostics: tuple


PROVIDER_ROLES = ('observer', 'evidence_reader', 'operator', 'evidence_writer', 'process_supervisor')
PHASES = ('state4_closed', 'state5_entry', 'state5_closed', 'state6_entry', 'pre_finalize',
          'pre_move', 'post_move', 'post_audit', 'post_complete', 'cleanup', 'terminal',
          'failure_location', 'post_quarantine')


def validate_file_ref(value):
    require(type(value) is FileRef, 'FileRef type')
    relative(value.root_binding_id)
    require('/' not in value.root_binding_id, 'root id')
    relative(value.relative_path); integer(value.file_bytes); digest(value.file_sha256)


def file_ref_from(value):
    keys(value, 'root_binding_id relative_path file_bytes file_sha256')
    result = FileRef(**value); validate_file_ref(result)
    return result


def validate_capture(value):
    require(type(value) is Capture, 'Capture type')
    text(value.clock_epoch_id); integer(value.first_sequence); integer(value.last_sequence)
    utc(value.started_utc); utc(value.finished_utc)
    require(value.first_sequence <= value.last_sequence and value.started_utc <= value.finished_utc, 'capture order')


def capture_from(value):
    keys(value, 'clock_epoch_id first_sequence last_sequence started_utc finished_utc')
    result = Capture(**value); validate_capture(result)
    return result


def within(actual, accepted):
    validate_capture(actual); validate_capture(accepted)
    return (actual.clock_epoch_id == accepted.clock_epoch_id and
            accepted.first_sequence <= actual.first_sequence <= actual.last_sequence <= accepted.last_sequence and
            accepted.started_utc <= actual.started_utc <= actual.finished_utc <= accepted.finished_utc)


def validate_binding(value):
    require(type(value) is Binding, 'Binding type')
    relative(value.attempt_id); require('/' not in value.attempt_id, 'attempt id')
    require((value.state, value.role) in (('state4', 'state4_worker'), ('state5', 'state5_worker'),
        ('state6', 'state6_verifier'), ('state6', 'state6_finalizer'), ('state4', 'supervisor'),
        ('state5', 'supervisor'), ('state6', 'supervisor')), 'state/role')
    for name in ('authorization_sha256', 'protocol_sha256', 'source_sha256'):
        digest(getattr(value, name))
    for name in ('implementation_commit', 'implementation_tree', 'implementation_parent'):
        digest(getattr(value, name), 40)
    validate_file_ref(value.plan_file)


def binding_from(value):
    keys(value, 'attempt_id state role authorization_sha256 protocol_sha256 implementation_commit implementation_tree implementation_parent source_sha256 plan_file')
    result = Binding(**{**value, 'plan_file': file_ref_from(value['plan_file'])})
    validate_binding(result)
    return result


def validate_roots(roots):
    require(type(roots) is tuple and len(roots) == 4, 'four independent roots')
    require(tuple(x.root_binding_id for x in roots if type(x) is RootBinding) == ('T', 'F', 'Q', 'L'), 'root inventory/order')
    paths = []; identities = []
    for root in roots:
        text(root.native_identity); validate_file_ref(root.evidence_file)
        path = root.canonical_path
        require(type(path) is str and re.fullmatch('[a-z]:/.+', path) is not None, 'canonical root path')
        relative(path[3:]); key = path.casefold()
        require(all(key != p and not key.startswith(p + '/') and not p.startswith(key + '/') for p in paths), 'root alias/overlap')
        require(root.native_identity not in identities, 'root identity alias')
        paths.append(key); identities.append(root.native_identity)


def validate_outputs(outputs):
    require(type(outputs) is tuple and bool(outputs), 'output inventory')
    slots = set(); paths = set(); identities = set()
    for row in outputs:
        require(type(row) is OutputBinding, 'output type')
        text(row.slot); relative(row.relative_path); text(row.native_identity)
        validate_file_ref(row.custody_file); validate_file_ref(row.content_file)
        require(row.introduced_phase in PHASES[:9], 'introduction phase')
        require(row.slot not in slots and row.relative_path.casefold() not in paths and row.native_identity not in identities, 'output alias')
        slots.add(row.slot); paths.add(row.relative_path.casefold()); identities.add(row.native_identity)


def validate_attempt_directory(directory, roots, outputs):
    require(type(directory) is AttemptDirectoryBinding, 'AttemptDirectoryBinding type')
    text(directory.native_identity); validate_file_ref(directory.custody_file)
    require(directory.native_identity not in {x.native_identity for x in roots + outputs}, 'directory/anchor/member alias')
    require(type(directory.locations) is tuple and
            tuple(x.root_binding_id for x in directory.locations if type(x) is DirectoryLocation) == ('T', 'F', 'Q'), 'directory location inventory')
    for location in directory.locations:
        relative(location.relative_path)


def validate_admission(value):
    require(type(value) is Admission, 'Admission type')
    text(value.role); validate_file_ref(value.file); text(value.scope); validate_capture(value.capture)
    require(type(value.validator_sources) is tuple and bool(value.validator_sources), 'validator inventory')
    for f in value.validator_sources:
        validate_file_ref(f)
    require(len(set(value.validator_sources)) == len(value.validator_sources), 'duplicate validator')
    require(value.outcome in ('accepted', 'rejected', 'unavailable'), 'admission outcome')


def validate_phase_transition(previous, current):
    require(current in PHASES, 'phase')
    if previous is None:
        require(current == 'state4_closed', 'initial phase')
        return
    require(previous in PHASES, 'previous phase')
    edges = set(zip(PHASES[:10], PHASES[1:11]))
    edges.update((p, 'failure_location') for p in PHASES[:9])
    edges.update((('failure_location', 'post_quarantine'), ('failure_location', 'cleanup'), ('post_quarantine', 'cleanup')))
    require((previous, current) in edges, 'phase transition')


def validate_provider_expectations(expected):
    require(type(expected) is ProviderExpectations, 'ProviderExpectations type')
    validate_binding(expected.binding); validate_roots(expected.roots); validate_outputs(expected.output_inventory)
    validate_attempt_directory(expected.attempt_directory, expected.roots, expected.output_inventory)
    require(expected.provider_role in PROVIDER_ROLES, 'provider role')
    validate_capture(expected.capture)
    for name in ('valid_from', 'valid_until', 'cleanup_until'):
        utc(getattr(expected, name))
    require(expected.valid_from <= expected.valid_until <= expected.cleanup_until, 'authority interval')
    needed = {'authority': expected.authority_file, 'qualification': expected.qualification_file, 'custody': expected.custody_file}
    needed['attempt_directory_custody'] = expected.attempt_directory.custody_file
    for prefix, values in (('source', expected.source_files), ('validator', expected.validator_files)):
        require(type(values) is tuple and bool(values) and len(set(values)) == len(values), 'source inventory')
        for i, f in enumerate(values):
            needed[prefix + ':' + str(i)] = f
    needed.update(('root:' + r.root_binding_id, r.evidence_file) for r in expected.roots)
    for f in needed.values():
        validate_file_ref(f)
    require(type(expected.admissions) is tuple and len(expected.admissions) == len(needed), 'admission inventory')
    seen = set()
    for admission in expected.admissions:
        validate_admission(admission)
        require(admission.role not in seen and needed.get(admission.role) == admission.file, 'admission mapping')
        require(admission.validator_sources == expected.validator_files and admission.scope == 'provider:' + expected.provider_role and
                admission.capture == expected.capture and admission.outcome == 'accepted', 'accepted provider admission')
        seen.add(admission.role)


def qualify_admissions(admissions, acquisitions):
    """Check actual immutable evidence independently of its supplied outcome label."""
    require(type(admissions) is tuple and type(acquisitions) is tuple, 'evidence tuples')
    expected = {}
    for admission in admissions:
        validate_admission(admission)
        require(admission.role not in expected and admission.outcome == 'accepted', 'independent admission inventory')
        expected[admission.role] = admission
    checked = []; diagnostics = []; groups = {}
    for index, actual in enumerate(acquisitions):
        if type(actual) is not ProviderAcquisition:
            diagnostics.append(('invalid', str(index), 'acquisition_type')); continue
        try:
            validate_admission(Admission(actual.role, actual.file, actual.validator_sources, actual.scope, actual.capture, actual.outcome))
            require(actual.data is None or type(actual.data) is bytes, 'acquired immutable bytes')
        except (BindingError, TypeError):
            diagnostics.append(('invalid', str(index), 'acquisition_structure')); continue
        groups.setdefault(actual.role, []).append(actual)
        if actual.role not in expected:
            diagnostics.append(('mismatch', actual.role, 'undeclared_acquisition'))
    for role, admission in expected.items():
        rows = groups.get(role, [])
        if len(rows) != 1:
            diagnostics.append(('incomplete' if not rows else 'mismatch', role, 'missing_acquisition' if not rows else 'duplicate_acquisition'))
            continue
        actual = rows[0]; start = len(diagnostics)
        if (actual.file, actual.validator_sources, actual.scope) != (admission.file, admission.validator_sources, admission.scope):
            diagnostics.append(('mismatch', role, 'acquisition_binding'))
        if not within(actual.capture, admission.capture):
            diagnostics.append(('mismatch', role, 'acquisition_interval'))
        if actual.outcome != 'accepted':
            diagnostics.append(('mismatch' if actual.outcome == 'rejected' else 'incomplete', role, 'acquisition_' + actual.outcome))
        if actual.data is None:
            diagnostics.append(('incomplete', role, 'bytes_unavailable'))
        elif (len(actual.data), sha(actual.data)) != (actual.file.file_bytes, actual.file.file_sha256):
            diagnostics.append(('mismatch', role, 'acquisition_bytes'))
        if len(diagnostics) == start:
            checked.append(role)
    return Qualification(tuple(checked), tuple(diagnostics))


def qualify_provider_acquisition(*, expected, acquisitions):
    validate_provider_expectations(expected)
    return qualify_admissions(expected.admissions, acquisitions)
