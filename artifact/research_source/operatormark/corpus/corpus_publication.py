"""Pure publication-plan metadata. No publication or execution authority."""
from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
import unicodedata


class PublicationPlanInputError(ValueError):
    """Malformed publication-plan metadata."""


_BASE = 'operatormark.darcy_branch_c_v1.'
_ROOT = 'artifacts/darcy_branch_c_v1/private_corpus_v1/'
_PHASES = ('state5_outputs_verified', 'publication_plan_reviewed',
           'state6_authorized', 'state6_child_verified_and_exited',
           'state6_metadata_finalized', 'attempt_root_promoted',
           'publication_readback_verified', 'final_audit_verified',
           'complete_published', 'terminal_outcome_retained')
_BINDINGS = ('protocol_file_sha256', 'implementation_commit',
             'implementation_tree', 'expected_parent_commit',
             'state4_authorization_file_sha256', 'state5_authorization_file_sha256')
_REFS = (('state4_pre_solve_manifest', 'state4_pre_solve_manifest_file',
          'state4_controls/pre_solve_manifest.json', _BASE + 'state4_pre_solve_manifest.v2'),
         ('state4_controls_completion', 'state4_controls_completion_file',
          'STATE4_CONTROLS_COMPLETE.json', _BASE + 'state4_controls_complete.v1'))


def _fail(message):
    raise PublicationPlanInputError(message)


def _text(value):
    if type(value) is not str:
        _fail('expected a string')
    try:
        value.encode('utf-8')
    except UnicodeError:
        _fail('invalid UTF-8 string')
    if unicodedata.normalize('NFC', value) != value:
        _fail('string must be NFC')
    return value


def _hex(value, width=64):
    _text(value)
    if re.fullmatch('[0-9a-f]{' + str(width) + '}', value) is None or not int(value, 16):
        _fail('invalid nonzero lowercase hexadecimal binding')


def _size(value):
    if type(value) is not int or value <= 0:
        _fail('file size must be a positive exact integer')


def _basename(value):
    _text(value)
    if (not value or value in ('.', '..') or value[-1] in '. '
            or any(c in '/\\:<>"|?*' or unicodedata.category(c) == 'Cc' for c in value)):
        _fail('invalid basename')
    stem = value.split('.')[0].upper()
    if stem in {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'} or re.fullmatch(r'(COM|LPT)[1-9¹²³]', stem):
        _fail('reserved Windows basename')


def _mapping(value, keys):
    if type(value) not in (dict, MappingProxyType) or set(value) != set(keys):
        _fail('mapping keys/type differ from the contract')


def _freeze(value):
    if type(value) not in (dict, MappingProxyType):
        _fail('descriptor must be a mapping')
    # Descriptor values are scalars. Copy before wrapping: no mutable alias survives.
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class PublicationPlanInputs:
    protocol_file_sha256: str
    implementation_commit: str
    implementation_tree: str
    expected_parent_commit: str
    state4_authorization_file_sha256: str
    state5_authorization_file_sha256: str
    state4_pre_solve_manifest_file: object
    state4_controls_completion_file: object
    state5_execution_evidence: object

    def __post_init__(self):
        for name in ('state4_pre_solve_manifest_file', 'state4_controls_completion_file', 'state5_execution_evidence'):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


def _validate_inputs(inputs):
    if type(inputs) is not PublicationPlanInputs:
        _fail('expected PublicationPlanInputs')
    for name in _BINDINGS:
        _hex(getattr(inputs, name), 40 if name in _BINDINGS[1:4] else 64)
    if inputs.state4_authorization_file_sha256 == inputs.state5_authorization_file_sha256:
        _fail('State 4 and State 5 authorizations must differ')
    for name in ('state4_pre_solve_manifest_file', 'state4_controls_completion_file'):
        value = getattr(inputs, name)
        _mapping(value, ('file_bytes', 'file_sha256'))
        _size(value['file_bytes'])
        _hex(value['file_sha256'])
    value = inputs.state5_execution_evidence
    _mapping(value, ('filename', 'literal_path_sha256', 'sha256', 'size'))
    _basename(value['filename'])
    _hex(value['literal_path_sha256'])
    _hex(value['sha256'])
    _size(value['size'])


def derive_publication_paths(state4_authorization_file_sha256):
    """Derive fixed lexical roots; this is not an existence or custody check."""
    _hex(state4_authorization_file_sha256)
    attempt = 's4-' + state4_authorization_file_sha256[:24]
    return MappingProxyType(dict(attempt_id=attempt,
        attempt_root_relative_path=_ROOT + 'attempts/' + attempt,
        final_root_relative_path=_ROOT + 'corpus',
        quarantine_root_relative_path=_ROOT + 'quarantine/' + attempt,
        publication_evidence_root_relative_path=_ROOT + 'publication_evidence/' + attempt))


def _encode(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, UnicodeError, RecursionError):
        _fail('not canonical JSON data')


def _seal(value):
    return {**value, 'record_self_sha256': hashlib.sha256(_encode(value)).hexdigest()}


def _body(inputs):
    value = dict(schema=_BASE + 'corpus_publication_plan.v1', benchmark='darcy_branch_c_v1',
                 corpus_version='private_corpus_v1', profile='whole_attempt_same_volume_v1')
    value.update({name: getattr(inputs, name) for name in _BINDINGS})
    for key, field, path, schema in _REFS:
        descriptor = getattr(inputs, field)
        value[key] = dict(relative_path=path, size=descriptor['file_bytes'],
                          sha256=descriptor['file_sha256'], schema=schema)
    value['state5_execution_evidence'] = dict(inputs.state5_execution_evidence)
    value.update(derive_publication_paths(inputs.state4_authorization_file_sha256))
    value.update(local_reference_basis='attempt_root_v1', phase_order=list(_PHASES))
    return value


def build_publication_plan(inputs):
    _validate_inputs(inputs)
    return _seal(_body(inputs))


def validate_publication_plan(plan):
    """Validate internal plan facts only; no external binding is accepted here."""
    if type(plan) is not dict:
        _fail('plan must be a plain JSON object')
    keys = set(_BINDINGS) | {r[0] for r in _REFS} | {
        'schema', 'benchmark', 'corpus_version', 'profile', 'state5_execution_evidence',
        'attempt_id', 'attempt_root_relative_path', 'final_root_relative_path',
        'quarantine_root_relative_path', 'publication_evidence_root_relative_path',
        'local_reference_basis', 'phase_order', 'record_self_sha256'}
    _mapping(plan, keys)
    args = {name: plan[name] for name in _BINDINGS}
    for key, field, path, schema in _REFS:
        ref = plan[key]
        if type(ref) is not dict:
            _fail('reference must be a plain object')
        _mapping(ref, ('relative_path', 'size', 'sha256', 'schema'))
        if ref['relative_path'] != path or ref['schema'] != schema:
            _fail('fixed predecessor path/schema differs')
        args[field] = dict(file_bytes=ref['size'], file_sha256=ref['sha256'])
    if type(plan['state5_execution_evidence']) is not dict:
        _fail('external reference must be a plain object')
    args['state5_execution_evidence'] = plan['state5_execution_evidence']
    inputs = PublicationPlanInputs(**args)
    _validate_inputs(inputs)
    _hex(plan['record_self_sha256'])
    # Exact canonical comparison also distinguishes bool/int and tuple/list.
    if type(plan['phase_order']) is not list or _encode(plan) != _encode(_seal(_body(inputs))):
        _fail('plan derivation, fixed value or self-hash differs')


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail('duplicate JSON key')
        value[key] = item
    return value


def strict_publication_plan_loads(data):
    if type(data) is not bytes:
        _fail('canonical plan input must be immutable bytes')
    try:
        value = json.loads(data.decode('utf-8'), object_pairs_hook=_pairs,
                           parse_constant=lambda _: _fail('non-finite JSON'))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise PublicationPlanInputError('invalid JSON encoding') from exc
    validate_publication_plan(value)
    if data != _encode(value):
        _fail('noncanonical original bytes')
    return value


def canonical_publication_plan_bytes(plan):
    validate_publication_plan(plan)
    return _encode(plan)
