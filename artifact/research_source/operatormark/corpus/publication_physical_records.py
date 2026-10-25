"""Strict inert physical-publication wire records. No I/O or authority issuer."""
from dataclasses import fields, is_dataclass, make_dataclass
from datetime import datetime
import hashlib
import json
import re
from types import MappingProxyType
import unicodedata

# Path punctuation plus Cc only; _text retains the earlier UTF-8/NFC checks.
_UNSAFE_PATH_CHARACTER = re.compile(r'[\\:<>"|?*\x00-\x1f\x7f-\x9f]')


class PublicationRecordError(ValueError):
    pass


_PREFIX = 'operatormark.darcy_branch_c_v1.'
_PROFILE = 'whole_attempt_same_volume_v1'
_PHASES = ('pre_finalize', 'pre_move', 'post_move', 'post_audit', 'post_complete', 'failure_location', 'post_quarantine')
_STEPS = ('finalize_metadata', 'promote', 'publish_audit', 'publish_complete', 'finalize_protection', 'quarantine', 'release_resources')
_CALLS = ('not_called', 'returned_success', 'returned_failure', 'outcome_unavailable')
_EFFECTS = ('none_observed', 'expected_effect_observed', 'unexpected_effect_observed', 'unknown')
_FAILURES = ('access_denied', 'authority_unavailable', 'authority_refused', 'input_invalid', 'custody_mismatch', 'io_failed', 'result_unavailable', 'retention_failed', 'cleanup_failed')
_SCHEMAS = {
    'publication_physical_observation': 'binding phase observer_binding_file authority_evidence_files capture objects enumerations failures',
    'publication_operation_intent': 'binding operation_id step sequence previous_result_file precondition_files source_address destination_address prepared_files',
    'publication_operation_result': 'binding intent_file step capture call_status effect_status before_observation_file after_observation_file retained_files failures',
    'publication_process_outcome': 'binding role source_closure_file authority_scope_file capture creation_status process_identity direct_exit_code stdout stderr guards_file reap_status failures',
    'publication_terminal_outcome': 'binding closed_capture disposition operation_files process_outcome_files physical_comparison_file final_audit_file completion_file located_root failure_evidence_files retention_inventory_file production_preflight_accepted execution_authorized',
    'publication_retention_inventory': 'binding inventory_address capture slots observed_files prior_inventory_file',
    'publication_root_bindings': 'publication_plan_file roots',
}
_BINDING = 'profile attempt_id protocol_file_sha256 implementation_commit implementation_tree expected_parent_commit state4_authorization_file_sha256 state5_authorization_file_sha256 state6_authorization_file_sha256 publication_plan_file'
_WIRE = {
    'FileRef': 'root_binding_id relative_path file_bytes file_sha256',
    'NativeIdentity': 'volume_serial_number_hex file_id_128_hex legacy_volume_serial_number_hex legacy_file_index_hex',
    'Address': 'root_binding_id relative_path',
    'Binding': _BINDING,
    'Capture': 'invocation_id clock_epoch_id first_sequence last_sequence started_utc finished_utc',
    'Failure': 'kind native_error_hex exception_file affected_addresses',
}


def _need(ok, message):
    if not ok:
        raise PublicationRecordError(message)


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if type(value) in (dict, MappingProxyType):
        return {k: _plain(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return [_plain(v) for v in value]
    return value


def _freeze(value):
    if type(value) in (dict, MappingProxyType):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(v) for v in value)
    return value


def _post(self):
    for field in fields(self):
        object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))


# Every wire field is mandatory; no constructor seals or manufactures observations.
for _name, _names in _WIRE.items():
    globals()[_name] = make_dataclass(_name, [(n, object) for n in _names.split()],
                                     frozen=True, namespace={'__post_init__': _post, '__module__': __name__})
for _name, _names in _SCHEMAS.items():
    _class_name = ''.join(part.title() for part in _name.split('_'))
    globals()[_class_name] = make_dataclass(_class_name,
        [(n, object) for n in ('schema ' + _names + ' record_self_sha256').split()],
        frozen=True, namespace={'__post_init__': _post, '__module__': __name__})


def _object(value, keys):
    _need(type(value) is dict and set(value) == set(keys.split() if type(keys) is str else keys), 'exact object keys/type required')


def _text(value, nonempty=True):
    _need(type(value) is str and (bool(value) or not nonempty), 'NFC string required')
    try:
        value.encode('utf-8')
    except UnicodeError as exc:
        raise PublicationRecordError('invalid UTF-8 string') from exc
    _need(unicodedata.normalize('NFC', value) == value, 'non-NFC string')


def _hash(value, width=64):
    _text(value)
    _need(re.fullmatch('[0-9a-f]{' + str(width) + '}', value) is not None and value != '0' * width, 'nonzero lowercase hash/OID required')


def _integer(value, positive=False):
    _need(type(value) is int and value >= int(positive), 'exact nonnegative integer required')


def _enum(value, choices):
    _need(type(value) is str and value in choices, 'closed enum required')


def _path(value, empty=False):
    _text(value, nonempty=not empty)
    if value == '':
        return
    for component in value.split('/'):
        _need(bool(component) and component not in ('.', '..') and component[-1] not in '. ', 'unsafe path component')
        _need(_UNSAFE_PATH_CHARACTER.search(component) is None, 'unsafe path character')
        stem = component.split('.')[0].upper()
        _need(stem not in {'CON','PRN','AUX','NUL','CONIN$','CONOUT$'} and re.fullmatch(r'(COM|LPT)[1-9¹²³]', stem) is None, 'Windows reserved component')


def _address(value):
    _object(value, _WIRE['Address'])
    _text(value['root_binding_id'])
    _path(value['relative_path'], empty=True)


def _ref(value, nullable=False, positive=True):
    if value is None and nullable:
        return
    _object(value, _WIRE['FileRef'])
    _text(value['root_binding_id'])
    _path(value['relative_path'])
    _integer(value['file_bytes'], positive)
    _hash(value['file_sha256'])


def _key(value):
    return (value['root_binding_id'], value['relative_path'])


def _ordered(values, validator, key=None, casefold=False):
    _need(type(values) is list, 'array required')
    for value in values:
        validator(value)
    keys = [key(v) if key else _json(v) for v in values]
    _need(keys == sorted(keys) and len(set(keys)) == len(keys), 'array must be sorted and unique')
    if casefold:
        aliases = [tuple(p.casefold() for p in k) if type(k) is tuple else k.casefold() for k in keys]
        _need(len(set(aliases)) == len(aliases), 'casefold collision')


def _identity(value, nullable=False):
    if value is None and nullable:
        return
    _object(value, _WIRE['NativeIdentity'])
    _uhex(value['volume_serial_number_hex'], 64)
    for key, width in (('file_id_128_hex',32), ('legacy_volume_serial_number_hex',8), ('legacy_file_index_hex',16)):
        _need(type(value[key]) is str and re.fullmatch('[0-9a-f]{' + str(width) + '}', value[key]) is not None, 'native identity width/spelling')


def _uhex(value, bits):
    _need(type(value) is str and re.fullmatch(r'0x(?:0|[1-9a-f][0-9a-f]*)', value) is not None, 'minimal unsigned hex required')
    _need(int(value, 16) < 2 ** bits, 'unsigned hex out of range')


def _utc(value):
    _text(value)
    _need(re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z', value) is not None and len(value) == 27, 'exact UTC timestamp required')
    try:
        datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError as exc:
        raise PublicationRecordError('invalid UTC date/time') from exc


def _invocation(binding):
    version='v2' if binding['profile']=='whole_attempt_same_volume_v2' else 'v1'
    data = ('operatormark-publication-invocation-'+version+'\n' + binding['state6_authorization_file_sha256']
            + '\n' + binding['publication_plan_file']['file_sha256']).encode()
    return hashlib.sha256(data).hexdigest()


def _capture(value, binding, closed=False):
    _object(value, _WIRE['Capture'])
    _hash(value['invocation_id'])
    _need(value['invocation_id'] == _invocation(binding), 'invocation mismatch')
    _text(value['clock_epoch_id'])
    _integer(value['first_sequence'])
    _utc(value['started_utc'])
    _need((value['last_sequence'] is None) == (value['finished_utc'] is None), 'incomplete capture endpoints disagree')
    if value['last_sequence'] is None:
        _need(not closed, 'closed capture required')
    else:
        _integer(value['last_sequence'])
        _utc(value['finished_utc'])
        _need(value['last_sequence'] >= value['first_sequence'] and value['finished_utc'] >= value['started_utc'], 'capture reversed')


def _failure(value):
    _object(value, _WIRE['Failure'])
    _enum(value['kind'], _FAILURES)
    if value['native_error_hex'] is not None:
        _uhex(value['native_error_hex'], 32)
    _ref(value['exception_file'], nullable=True, positive=False)
    _ordered(value['affected_addresses'], _address, _key, True)


def _failures(values):
    _need(type(values) is list, 'ordered failures array required')
    for value in values:
        _failure(value)


def _json(value):
    def strings(item):
        if type(item) is dict:
            for k, v in item.items():
                _text(k, False)
                strings(v)
        elif type(item) is list:
            for v in item:
                strings(v)
        elif type(item) is str:
            _text(item, False)
        else:
            _need(item is None or type(item) in (bool, int), 'not exact JSON wire data')
    try:
        strings(value)
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise PublicationRecordError('invalid canonical wire data') from exc


def _self(value):
    _hash(value['record_self_sha256'])
    digest = hashlib.sha256(_json({k:v for k,v in value.items() if k != 'record_self_sha256'})).hexdigest()
    _need(digest == value['record_self_sha256'], 'self hash mismatch')


def validate_publication_binding(binding, *, allow_v2=False):
    binding = _plain(binding) if is_dataclass(binding) or type(binding) is MappingProxyType else binding
    _object(binding, _BINDING)
    _enum(binding['profile'], (_PROFILE,'whole_attempt_same_volume_v2') if allow_v2 else (_PROFILE,))
    for key in ('protocol_file_sha256','state4_authorization_file_sha256','state5_authorization_file_sha256','state6_authorization_file_sha256'):
        _hash(binding[key])
    for key in ('implementation_commit','implementation_tree','expected_parent_commit'):
        _hash(binding[key], 40)
    auth = [binding['state' + str(n) + '_authorization_file_sha256'] for n in (4,5,6)]
    _need(len(set(auth)) == 3, 'authorizations must be distinct')
    _need(binding['attempt_id'] == 's4-' + auth[0][:24], 'attempt derivation mismatch')
    _ref(binding['publication_plan_file'])


def _observation(record):
    _enum(record['phase'], _PHASES)
    _ref(record['observer_binding_file'])
    _ordered(record['authority_evidence_files'], _ref)
    _capture(record['capture'], record['binding'])
    def obj(row):
        _object(row, 'address presence presence_evidence_file kind native_identity content_file security_file permission_handoff_file stable_read_file custody_file')
        _address(row['address'])
        _enum(row['presence'], ('present','absent','unknown'))
        _need(row['kind'] is None or row['kind'] in ('file','directory'), 'object kind')
        _identity(row['native_identity'], True)
        for key in ('presence_evidence_file','content_file','security_file','permission_handoff_file','stable_read_file','custody_file'):
            _ref(row[key], True, positive=key != 'content_file')
        if row['presence'] == 'present':
            _need(row['kind'] is not None and row['native_identity'] is not None and row['presence_evidence_file'] is not None, 'present facts missing')
        if row['presence'] == 'absent':
            _need(row['presence_evidence_file'] is not None, 'absence witness missing')
            _need(all(row[k] is None for k in row if k not in ('address','presence','presence_evidence_file')), 'contradictory absence')
        if row['kind'] == 'directory':
            _need(row['content_file'] is None, 'directory cannot have content bytes')
    _ordered(record['objects'], obj, lambda row: _key(row['address']), True)
    def enumeration(row):
        _object(row, 'directory children complete evidence_file')
        _address(row['directory'])
        _need(type(row['complete']) is bool, 'enumeration completeness boolean')
        _ref(row['evidence_file'], True)
        _need(not row['complete'] or row['evidence_file'] is not None, 'complete enumeration needs evidence')
        def child(item):
            _object(item, 'name kind')
            _path(item['name'])
            _need('/' not in item['name'], 'child basename only')
            _enum(item['kind'], ('file','directory','unknown'))
        _ordered(row['children'], child, lambda x:x['name'], True)
    _ordered(record['enumerations'], enumeration, lambda row:_key(row['directory']), True)
    _failures(record['failures'])


def _intent(record):
    _enum(record['step'], _STEPS)
    _need(record['operation_id'] == _invocation(record['binding']) + ':' + record['step'], 'operation ID mismatch')
    _integer(record['sequence'])
    _ref(record['previous_result_file'], True)
    _need((record['sequence'] == 0) == (record['previous_result_file'] is None), 'previous-result chain mismatch')
    _ordered(record['precondition_files'], _ref)
    for key in ('source_address','destination_address'):
        if record[key] is not None:
            _address(record[key])
    def prepared(row):
        _object(row, 'destination file')
        _address(row['destination'])
        _ref(row['file'], positive=False)
    _ordered(record['prepared_files'], prepared, lambda x:_key(x['destination']), True)
    if record['step'] in ('promote','quarantine'):
        _need(record['source_address'] is not None and record['destination_address'] is not None and not record['prepared_files'], 'directory-only move intent')
        _need(record['source_address'] != record['destination_address'], 'move source equals destination')
    if record['step'] == 'release_resources':
        _need(record['source_address'] is None and record['destination_address'] is None and not record['prepared_files'], 'release has no content/destinations')
    if record['step'] == 'publish_complete':
        _need(len(record['prepared_files']) == 1, 'one COMPLETE required')
        _need(record['prepared_files'][0]['destination']['relative_path'].split('/')[-1] == 'COMPLETE.json', 'fixed completion basename')


def _result(record):
    _enum(record['step'], _STEPS)
    _ref(record['intent_file'])
    _capture(record['capture'], record['binding'])
    _enum(record['call_status'], _CALLS)
    _enum(record['effect_status'], _EFFECTS)
    _ref(record['before_observation_file'], True)
    _ref(record['after_observation_file'], True)
    _ordered(record['retained_files'], lambda x:_ref(x, positive=False))
    _failures(record['failures'])
    if record['call_status'] == 'not_called':
        _need(record['effect_status'] != 'expected_effect_observed', 'not-called cannot establish delegate effect')
        _need(record['failures'] and record['before_observation_file'] is not None, 'not-called needs observed refusal/guard evidence')
    if record['effect_status'] == 'expected_effect_observed':
        _need(record['after_observation_file'] is not None, 'effect needs observation reference')


def _process(record):
    _enum(record['role'], ('state6_verification_child','publication_finalizer'))
    _ref(record['source_closure_file'])
    _ref(record['authority_scope_file'])
    _capture(record['capture'], record['binding'])
    _enum(record['creation_status'], ('not_started','started','unknown'))
    _enum(record['reap_status'], ('not_applicable','observed_reaped','unresolved','failed'))
    if record['process_identity'] is not None:
        _object(record['process_identity'], 'pid creation_evidence_file')
        _integer(record['process_identity']['pid'], True)
        _ref(record['process_identity']['creation_evidence_file'])
    _need(record['direct_exit_code'] is None or type(record['direct_exit_code']) is int, 'direct exit exact integer or unavailable')
    if record['direct_exit_code'] is not None:
        _need(record['creation_status'] == 'started', 'exit requires known process start')
    if record['reap_status'] == 'observed_reaped':
        _need(record['creation_status'] == 'started' and record['direct_exit_code'] is not None, 'observed reaping requires direct exit')
    for key in ('stdout','stderr'):
        stream = record[key]
        _object(stream, 'state file')
        _enum(stream['state'], ('complete','partial','unavailable','not_started'))
        _ref(stream['file'], True, False)
        _need((stream['file'] is not None) == (stream['state'] in ('complete','partial')), 'stream state/reference mismatch')
    _ref(record['guards_file'], True)
    if record['creation_status'] == 'not_started':
        _need(record['process_identity'] is None and record['direct_exit_code'] is None and record['reap_status'] == 'not_applicable', 'not-started process contradictions')
        _need(all(record[k]['state'] == 'not_started' for k in ('stdout','stderr')), 'not-started stream contradiction')
    _failures(record['failures'])


def _terminal(record):
    _capture(record['closed_capture'], record['binding'], True)
    _enum(record['disposition'], ('publication_observed','refused_before_mutation','failed_preserved','unresolved'))
    _enum(record['located_root'], ('T','F','Q','unknown'))
    _need(record['production_preflight_accepted'] is False and record['execution_authorized'] is False, 'authority flags must be false')
    _need(type(record['operation_files']) is list, 'operation trace array')
    steps = []
    for row in record['operation_files']:
        _object(row, 'step intent_file result_file')
        _enum(row['step'], _STEPS)
        _ref(row['intent_file'])
        _ref(row['result_file'], True)
        _need(row['result_file'] is not None or record['disposition'] == 'unresolved', 'missing result requires unresolved')
        steps.append(row['step'])
    _need(len(set(steps)) == len(steps), 'duplicate operation step')
    for key in ('process_outcome_files','failure_evidence_files'):
        _need(type(record[key]) is list, 'ordered references')
        for item in record[key]:
            _ref(item)
    for key in ('physical_comparison_file','final_audit_file','completion_file'):
        _ref(record[key], True)
    _ref(record['retention_inventory_file'])
    if record['disposition'] == 'publication_observed':
        _need(steps == ['finalize_metadata','promote','publish_audit','publish_complete','finalize_protection','release_resources'], 'success trace incomplete')
        _need(record['located_root'] == 'F' and not record['failure_evidence_files'], 'success with failure/location contradiction')
        _need(all(record[k] is not None for k in ('physical_comparison_file','final_audit_file','completion_file')), 'success evidence missing')
        _need(len(record['process_outcome_files']) >= 2, 'child/finalizer outcomes required')


def _inventory(record):
    _address(record['inventory_address'])
    _capture(record['capture'], record['binding'])
    _ref(record['prior_inventory_file'], True)
    def slot(row):
        _object(row, 'address owner role condition creating_address writer_source_file')
        _address(row['address'])
        _enum(row['owner'], ('accepted_attempt_M','external_reference_only'))
        _enum(row['role'], ('plan_copy','input_projection','intent','result','observation','process','stdout','stderr','details','report','failure','terminal'))
        _enum(row['condition'], ('entry','step_reached','phase_reached','failure_observed','terminal_pending'))
        if row['creating_address'] is not None:
            _address(row['creating_address'])
        _ref(row['writer_source_file'])
    _ordered(record['slots'], slot, lambda x:_key(x['address']), True)
    _need(record['inventory_address'] in [x['address'] for x in record['slots']], 'inventory own slot must be declared')
    def observed(row):
        _object(row, 'address state file')
        _address(row['address'])
        _enum(row['state'], ('retained_complete','retained_partial','unavailable','not_reached'))
        _ref(row['file'], True, False)
        _need(row['state'] != 'retained_complete' or row['file'] is not None, 'complete retained bytes required')
        _need(row['state'] not in ('unavailable','not_reached') or row['file'] is None, 'unavailable is not zero bytes')
        if row['file'] is not None:
            _need(_key(row['file']) == _key(row['address']), 'retained address mismatch')
    _ordered(record['observed_files'], observed, lambda x:_key(x['address']), True)
    _need(record['inventory_address'] not in [x['address'] for x in record['observed_files']], 'inventory must not hash own future bytes')


def _roots(record):
    _ref(record['publication_plan_file'])
    def root(row):
        _object(row, 'root_binding_id repository_relative_root identity_file custody_file')
        _text(row['root_binding_id'])
        if row['repository_relative_root'] is not None:
            _path(row['repository_relative_root'], True)
        _ref(row['identity_file'])
        _ref(row['custody_file'])
    _ordered(record['roots'], root, lambda x:x['root_binding_id'], True)
    paths = [r['repository_relative_root'] for r in record['roots'] if r['repository_relative_root'] is not None]
    _need(paths.count('') == 1, 'exactly one repository anchor required')
    _need(len({p.casefold() for p in paths}) == len(paths), 'root path alias')


_VALIDATORS = dict(zip(_SCHEMAS, (_observation,_intent,_result,_process,_terminal,_inventory,_roots)))


def _validate(record, expected_schema):
    v2 = expected_schema == 'operatormark.darcy_branch_c_v2.publication_process_outcome.v1'
    _need(v2 or expected_schema in {_PREFIX + k + '.v1' for k in _SCHEMAS}, 'closed record schema selector')
    record = _plain(record) if is_dataclass(record) or type(record) is MappingProxyType else record
    short = expected_schema[len(_PREFIX):-3]
    _object(record, 'schema ' + _SCHEMAS[short] + ' record_self_sha256')
    _need(record['schema'] == expected_schema, 'schema mismatch')
    if 'binding' in record:
        validate_publication_binding(record['binding'],allow_v2=v2)
        _need(record['binding']['profile'] == ('whole_attempt_same_volume_v2' if v2 else _PROFILE),
              'publication schema/profile mismatch')
    _VALIDATORS[short](record)
    _self(record)
    return record


def validate_publication_observation(record):
    _validate(record, _PREFIX + 'publication_physical_observation.v1')


def validate_publication_intent(record):
    _validate(record, _PREFIX + 'publication_operation_intent.v1')


def validate_publication_operation_result(record):
    _validate(record, _PREFIX + 'publication_operation_result.v1')


def validate_publication_process_outcome(record, *, version=1):
    _need(type(version) is int and version in (1,2),'explicit publication version')
    prefix=_PREFIX if version==1 else 'operatormark.darcy_branch_c_v2.'
    _validate(record, prefix + 'publication_process_outcome.v1')


def validate_publication_terminal_outcome(record):
    _validate(record, _PREFIX + 'publication_terminal_outcome.v1')


def validate_publication_retention_inventory(record):
    _validate(record, _PREFIX + 'publication_retention_inventory.v1')


def validate_publication_root_bindings(record):
    _validate(record, _PREFIX + 'publication_root_bindings.v1')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _constant(value):
    raise PublicationRecordError('nonfinite JSON')


def strict_publication_record_loads(data, *, expected_schema):
    _need(type(data) is bytes, 'immutable bytes required')
    try:
        record = json.loads(data.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise PublicationRecordError('invalid canonical JSON') from exc
    record = _validate(record, expected_schema)
    _need(_json(record) == data, 'noncanonical original bytes')
    return record


def canonical_publication_record_bytes(record, *, expected_schema):
    return _json(_validate(record, expected_schema))
