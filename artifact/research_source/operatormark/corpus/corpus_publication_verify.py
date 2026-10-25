"""Independent, metadata-only publication-plan verification; no producer imports."""
from dataclasses import dataclass, fields
import hashlib
import json
import re
from types import MappingProxyType
import unicodedata


class PublicationPlanInputError(ValueError):
    """Invalid independent expectations, input envelope, or recomputed report."""


_PREFIX = 'operatormark.darcy_branch_c_v1.'
_PLAN_SCHEMA = _PREFIX + 'corpus_publication_plan.v1'
_REPORT_SCHEMA = _PREFIX + 'corpus_publication_plan_report.v1'
_PROFILE = 'whole_attempt_same_volume_v1'
_ORDER = ['state5_outputs_verified', 'publication_plan_reviewed', 'state6_authorized',
          'state6_child_verified_and_exited', 'state6_metadata_finalized',
          'attempt_root_promoted', 'publication_readback_verified',
          'final_audit_verified', 'complete_published', 'terminal_outcome_retained']
_AUTH = ('protocol_file_sha256', 'state4_authorization_file_sha256', 'state5_authorization_file_sha256')
_GIT = ('implementation_commit', 'implementation_tree', 'expected_parent_commit')
_FILES = ('plan', 'state4_pre_solve_manifest', 'state4_controls_completion', 'state5_execution_evidence')
_ROOT_KEYS = ('attempt_id', 'attempt_root_relative_path', 'final_root_relative_path',
              'quarantine_root_relative_path', 'publication_evidence_root_relative_path')
_KEYS = frozenset((*_AUTH, *_GIT, *_ROOT_KEYS, *_FILES[1:], 'schema', 'benchmark',
                  'corpus_version', 'profile', 'local_reference_basis', 'phase_order', 'record_self_sha256'))


def _require(condition, message):
    if not condition:
        raise PublicationPlanInputError(message)


def _string(value):
    _require(type(value) is str, 'string required')
    try:
        value.encode('utf-8')
    except UnicodeError as exc:
        raise PublicationPlanInputError('invalid UTF-8') from exc
    _require(value == unicodedata.normalize('NFC', value), 'NFC required')


def _digest(value, length=64):
    _string(value)
    _require(len(value) == length and re.fullmatch('[0-9a-f]+', value) is not None
             and value != '0' * length, 'nonzero lowercase digest required')


def _positive(value):
    _require(type(value) is int and value > 0, 'positive exact integer required')


def _name(value):
    _string(value)
    _require(bool(value) and value not in {'.', '..'} and not value.endswith(('.', ' ')), 'invalid basename')
    _require(not any(c in '\\/:<>"|?*' or unicodedata.category(c) == 'Cc' for c in value), 'unsafe basename')
    head = value.partition('.')[0].upper()
    _require(head not in {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'}
             and re.fullmatch('(COM|LPT)[123456789¹²³]', head) is None, 'reserved basename')


def _exact(value, keys, *, frozen=False):
    _require(type(value) in ((dict, MappingProxyType) if frozen else (dict,))
             and set(value) == set(keys), 'exact object keys/type required')


@dataclass(frozen=True)
class PublicationPlanExpectations:
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
        for key in ('state4_pre_solve_manifest_file', 'state4_controls_completion_file', 'state5_execution_evidence'):
            original = getattr(self, key)
            _require(type(original) in (dict, MappingProxyType), 'descriptor mapping required')
            object.__setattr__(self, key, MappingProxyType(dict(original)))


def validate_publication_plan_expectations(expected):
    _require(type(expected) is PublicationPlanExpectations, 'independent expectations required')
    for key in _AUTH:
        _digest(getattr(expected, key))
    for key in _GIT:
        _digest(getattr(expected, key), 40)
    _require(expected.state4_authorization_file_sha256 != expected.state5_authorization_file_sha256,
             'distinct State 4/5 authorization required')
    for key in ('state4_pre_solve_manifest_file', 'state4_controls_completion_file'):
        desc = getattr(expected, key)
        _exact(desc, ('file_bytes', 'file_sha256'), frozen=True)
        _positive(desc['file_bytes'])
        _digest(desc['file_sha256'])
    desc = expected.state5_execution_evidence
    _exact(desc, ('filename', 'literal_path_sha256', 'sha256', 'size'), frozen=True)
    _name(desc['filename'])
    _digest(desc['literal_path_sha256'])
    _digest(desc['sha256'])
    _positive(desc['size'])


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                          allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise PublicationPlanInputError('invalid canonical JSON value') from exc


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _seal(payload):
    return dict(payload, record_self_sha256=_hash(_json(payload)))


def _projection(expected):
    return {field.name: dict(value) if type(value) is MappingProxyType else value
            for field in fields(expected) for value in (getattr(expected, field.name),)}


def _derive(expected):
    """Independently construct the expected metadata, never a producer call."""
    attempt = 's4-' + expected.state4_authorization_file_sha256[:24]
    root = 'artifacts/darcy_branch_c_v1/private_corpus_v1/'
    result = {'schema': _PLAN_SCHEMA, 'benchmark': 'darcy_branch_c_v1',
              'corpus_version': 'private_corpus_v1', 'profile': _PROFILE,
              'local_reference_basis': 'attempt_root_v1', 'phase_order': list(_ORDER),
              'attempt_id': attempt, 'attempt_root_relative_path': root + 'attempts/' + attempt,
              'final_root_relative_path': root + 'corpus',
              'quarantine_root_relative_path': root + 'quarantine/' + attempt,
              'publication_evidence_root_relative_path': root + 'publication_evidence/' + attempt}
    result.update({key: getattr(expected, key) for key in (*_AUTH, *_GIT)})
    for key, path, schema in (
        ('state4_pre_solve_manifest', 'state4_controls/pre_solve_manifest.json', _PREFIX + 'state4_pre_solve_manifest.v2'),
        ('state4_controls_completion', 'STATE4_CONTROLS_COMPLETE.json', _PREFIX + 'state4_controls_complete.v1')):
        desc = getattr(expected, key + '_file')
        result[key] = {'relative_path': path, 'schema': schema, 'size': desc['file_bytes'], 'sha256': desc['file_sha256']}
    result['state5_execution_evidence'] = dict(expected.state5_execution_evidence)
    return _seal(result)


def _unique(pairs):
    out = {}
    for key, value in pairs:
        _require(key not in out, 'duplicate JSON member')
        out[key] = value
    return out


def _no_constant(value):
    raise PublicationPlanInputError('non-finite JSON')


def _load(data):
    try:
        obj = json.loads(data.decode('utf-8'), object_pairs_hook=_unique, parse_constant=_no_constant)
        _require(_json(obj) == data, 'original bytes not canonical')
        return obj
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise PublicationPlanInputError('canonical encoding invalid') from exc


def _candidate(data):
    """Return a valid independent plan or a closed diagnostic code."""
    try:
        plan = _load(data)
    except PublicationPlanInputError:
        return None, 'canonical_encoding_invalid'
    try:
        _exact(plan, _KEYS)
        for key in ('schema', 'benchmark', 'corpus_version', 'profile', 'local_reference_basis', *_ROOT_KEYS):
            _string(plan[key])
        _require(type(plan['phase_order']) is list, 'phase order must be an array')
        for item in plan['phase_order']:
            _string(item)
        args = {key: plan[key] for key in (*_AUTH, *_GIT)}
        for key in _FILES[1:3]:
            ref = plan[key]
            _exact(ref, ('relative_path', 'schema', 'sha256', 'size'))
            _string(ref['relative_path'])
            _string(ref['schema'])
            args[key + '_file'] = dict(file_bytes=ref['size'], file_sha256=ref['sha256'])
        _exact(plan['state5_execution_evidence'], ('filename', 'literal_path_sha256', 'sha256', 'size'))
        args['state5_execution_evidence'] = plan['state5_execution_evidence']
        own = PublicationPlanExpectations(**args)
        validate_publication_plan_expectations(own)
        _digest(plan['record_self_sha256'])
    except PublicationPlanInputError:
        return None, 'schema_invalid'
    if _hash(_json({k: v for k, v in plan.items() if k != 'record_self_sha256'})) != plan['record_self_sha256']:
        return None, 'self_hash_invalid'
    if _json(plan) != _json(_derive(own)):
        return None, 'plan_derivation_invalid'
    return plan, None


def verify_publication_plan(plan_bytes, *, expected, state4_pre_solve_manifest_bytes,
                            state4_controls_completion_bytes, state5_execution_evidence_bytes):
    validate_publication_plan_expectations(expected)
    values = (plan_bytes, state4_pre_solve_manifest_bytes,
              state4_controls_completion_bytes, state5_execution_evidence_bytes)
    for value in values:
        _require(value is None or type(value) is bytes, 'only immutable bytes or None are allowed')
    inputs = dict(zip(_FILES, values))
    mismatches, invalid, unavailable, checked = [], [], [], []

    def add(rows, name, pointer, code):
        rows.append(dict(input=name, pointer=pointer, code=code))

    for name, value in inputs.items():
        if value is None:
            add(unavailable, name, '', 'required_bytes_unavailable')
    plan = None
    if plan_bytes is not None:
        plan, code = _candidate(plan_bytes)
        if code:
            add(invalid, 'plan', '', code)
    wanted = _derive(expected)
    if plan is not None:
        for key in sorted(_KEYS):
            checked.append('/plan/' + key)
            if _json(plan[key]) != _json(wanted[key]):
                add(mismatches, 'plan', '/' + key, 'accepted_binding_mismatch')
    for name in _FILES[1:]:
        raw = inputs[name]
        if raw is None:
            continue
        checked.append('/' + name)
        actual = (len(raw), _hash(raw))
        independent = (wanted[name]['size'], wanted[name]['sha256'])
        if actual != independent:
            add(mismatches, name, '', 'predecessor_file_mismatch')
        if plan is not None and actual != (plan[name]['size'], plan[name]['sha256']):
            add(mismatches, name, '', 'predecessor_file_mismatch')

    def ordered(rows):
        keys = sorted({(row['input'], row['pointer'], row['code']) for row in rows})
        return [dict(input=name, pointer=pointer, code=code) for name, pointer, code in keys]

    report = dict(schema=_REPORT_SCHEMA, profile=_PROFILE,
        expected_inputs_sha256=_hash(_json(_projection(expected))),
        input_files={name: None if raw is None else dict(file_bytes=len(raw), file_sha256=_hash(raw))
                     for name, raw in inputs.items()},
        comparison_status='invalid' if invalid else 'mismatch' if mismatches else 'incomplete' if unavailable else 'matched',
        checked_fields=sorted(set(checked)), mismatches=ordered(mismatches),
        invalid_inputs=ordered(invalid), unavailable_requirements=ordered(unavailable),
        production_preflight_accepted=False, execution_authorized=False)
    return _seal(report)


def _verified_report(report, verification_inputs):
    if type(report) is bytes:
        report = _load(report)
    _require(type(report) is dict, 'report must be a plain object or canonical bytes')
    expected_keys = {'plan_bytes', 'expected', 'state4_pre_solve_manifest_bytes',
                     'state4_controls_completion_bytes', 'state5_execution_evidence_bytes'}
    _require(set(verification_inputs) == expected_keys, 'exact original verification inputs required')
    recomputed = verify_publication_plan(**verification_inputs)
    _require(_json(report) == _json(recomputed), 'report differs from complete independent recomputation')
    return _json(recomputed)


def validate_publication_plan_report(report, **verification_inputs):
    _verified_report(report, verification_inputs)


def canonical_publication_plan_report_bytes(report, **verification_inputs):
    return _verified_report(report, verification_inputs)
