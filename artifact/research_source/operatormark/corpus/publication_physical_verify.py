"""Conditional independent physical-evidence comparison. No producer or native imports."""
from dataclasses import dataclass, fields
from datetime import datetime
import hashlib
import json
import re
from types import MappingProxyType
import unicodedata

# Independent path policy: forbidden punctuation and Cc, not generic printability.
_UNSAFE_PATH_CHARACTER = re.compile(r'[\\:<>"|?*\x00-\x1f\x7f-\x9f]')
from operatormark.corpus import corpus_publication_verify as plan_verify


class PublicationComparisonError(ValueError):
    def __init__(self, message, *, bootstrap_status='invalid', diagnostics=()):
        super().__init__(message)
        self.bootstrap_status = bootstrap_status
        self.diagnostics = tuple(_freeze(row) for row in diagnostics)


_PREFIX = 'operatormark.darcy_branch_c_v1.'
_PROFILE = 'whole_attempt_same_volume_v1'
_PHASES = ('pre_finalize','pre_move','post_move','post_audit','post_complete','failure_location','post_quarantine','terminal')
_STEPS = ('finalize_metadata','promote','publish_audit','publish_complete','finalize_protection','quarantine','release_resources')
_FORWARD = ('finalize_metadata','promote','publish_audit','publish_complete','finalize_protection','release_resources')
_REQUIRED = ('protocol','state4_authorization','state5_authorization','state6_authorization','state4_pre_solve_manifest','state4_controls_completion','state5_execution','state6_child_execution','state6_child_verification','state6_child_guards','state4_custody_chain','root_bindings')
_INVALID = {'canonical_encoding_invalid','schema_invalid','self_hash_invalid','chronology_invalid','disclosure_invalid'}
_MISMATCH = {'accepted_binding_mismatch','source_binding_mismatch','identity_mismatch','content_mismatch','security_mismatch','inventory_mismatch','unexpected_effect'}
_UNAVAILABLE = {'authority_evidence_unavailable','qualification_unavailable','required_bytes_unavailable','custody_unavailable','inventory_incomplete','operation_outcome_unavailable','retention_incomplete'}
_RECORDS = {
 'publication_physical_observation': 'binding phase observer_binding_file authority_evidence_files capture objects enumerations failures',
 'publication_operation_intent': 'binding operation_id step sequence previous_result_file precondition_files source_address destination_address prepared_files',
 'publication_operation_result': 'binding intent_file step capture call_status effect_status before_observation_file after_observation_file retained_files failures',
 'publication_process_outcome': 'binding role source_closure_file authority_scope_file capture creation_status process_identity direct_exit_code stdout stderr guards_file reap_status failures',
 'publication_terminal_outcome': 'binding closed_capture disposition operation_files process_outcome_files physical_comparison_file final_audit_file completion_file located_root failure_evidence_files retention_inventory_file production_preflight_accepted execution_authorized',
 'publication_retention_inventory': 'binding inventory_address capture slots observed_files prior_inventory_file',
 'publication_root_bindings': 'publication_plan_file roots',
}
_BINDING_KEYS = 'profile attempt_id protocol_file_sha256 implementation_commit implementation_tree expected_parent_commit state4_authorization_file_sha256 state5_authorization_file_sha256 state6_authorization_file_sha256 publication_plan_file'


def _freeze(value):
    if type(value) in (dict, MappingProxyType):
        return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value):
    if type(value) in (dict, MappingProxyType):
        return {k:_plain(v) for k,v in value.items()}
    if type(value) in (tuple,list):
        return [_plain(v) for v in value]
    return value


def _require(ok, message):
    if not ok:
        raise PublicationComparisonError(message)


def _text(value, empty=False):
    _require(type(value) is str and (empty or bool(value)), 'text type/emptiness')
    try:
        value.encode('utf-8')
    except UnicodeError as exc:
        raise PublicationComparisonError('UTF-8 string required') from exc
    _require(value == unicodedata.normalize('NFC', value), 'NFC required')


def _path(value, empty=False):
    _text(value, empty)
    if not value:
        return
    for part in value.split('/'):
        _require(bool(part) and part not in {'.','..'} and not part.endswith(('.', ' ')), 'path segment')
        _require(_UNSAFE_PATH_CHARACTER.search(part) is None, 'unsafe path')
        stem = part.partition('.')[0].upper()
        _require(stem not in {'CON','PRN','AUX','NUL','CONIN$','CONOUT$'} and re.fullmatch('(COM|LPT)[1-9¹²³]', stem) is None, 'reserved path')


def _obj(value, keys):
    _require(type(value) is dict and set(value) == set(keys.split() if type(keys) is str else keys), 'exact object shape')


def _hash(value, width=64):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{'+str(width)+'}', value) is not None and value != '0'*width, 'hash/OID spelling')


def _int(value, minimum=0):
    _require(type(value) is int and value >= minimum, 'exact integer domain')


def _enum(value, choices):
    _require(type(value) is str and value in choices, 'enum domain')


def _address(value):
    _obj(value, 'root_binding_id relative_path')
    _text(value['root_binding_id'])
    _path(value['relative_path'], True)


def _ref(value, nullable=False, positive=True):
    if value is None and nullable:
        return
    _obj(value, 'root_binding_id relative_path file_bytes file_sha256')
    _text(value['root_binding_id'])
    _path(value['relative_path'])
    _int(value['file_bytes'], int(positive))
    _hash(value['file_sha256'])


def _address_key(value):
    return value['root_binding_id'], value['relative_path']


def _canonical(value):
    value = _plain(value)
    def walk(item):
        if type(item) is dict:
            for k,v in item.items():
                _text(k, True)
                walk(v)
        elif type(item) is list:
            for v in item:
                walk(v)
        elif type(item) is str:
            _text(item, True)
        else:
            _require(item is None or type(item) in (bool,int), 'exact JSON data')
    try:
        walk(value)
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError,TypeError,UnicodeError,RecursionError) as exc:
        raise PublicationComparisonError('canonical data invalid') from exc


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _seal(value):
    return dict(value, record_self_sha256=_digest(_canonical(value)))


def _pairs(pairs):
    value = {}
    for key,item in pairs:
        _require(key not in value, 'duplicate key')
        value[key] = item
    return value


def _constant(value):
    raise PublicationComparisonError('nonfinite JSON')


def _decode(raw):
    _require(type(raw) is bytes, 'immutable JSON bytes')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_constant)
        _require(_canonical(value) == raw, 'noncanonical encoding')
        return value
    except (ValueError,UnicodeError,RecursionError) as exc:
        raise PublicationComparisonError('canonical_encoding_invalid') from exc


def _self_hash(value):
    _hash(value['record_self_sha256'])
    _require(_digest(_canonical({k:v for k,v in value.items() if k != 'record_self_sha256'})) == value['record_self_sha256'], 'self_hash_invalid')


def _unique(values, check, key=None):
    _require(type(values) is list, 'array required')
    for item in values:
        check(item)
    keys = [key(x) if key else _canonical(x) for x in values]
    _require(keys == sorted(keys) and len(set(keys)) == len(keys), 'ordered unique array')
    if key:
        folded = [tuple(y.casefold() for y in x) if type(x) is tuple else x.casefold() for x in keys]
        _require(len(set(folded)) == len(folded), 'casefold aliases')


def _identity(value, nullable=False):
    if value is None and nullable:
        return
    _obj(value, 'volume_serial_number_hex file_id_128_hex legacy_volume_serial_number_hex legacy_file_index_hex')
    v = value['volume_serial_number_hex']
    _require(type(v) is str and re.fullmatch(r'0x(?:0|[1-9a-f][0-9a-f]*)',v) is not None and int(v,16)<2**64, 'native volume spelling')
    for key,width in (('file_id_128_hex',32),('legacy_volume_serial_number_hex',8),('legacy_file_index_hex',16)):
        _require(type(value[key]) is str and re.fullmatch('[0-9a-f]{'+str(width)+'}',value[key]) is not None, 'native identity width')


def _binding(value):
    _obj(value, _BINDING_KEYS)
    _enum(value['profile'], (_PROFILE,))
    for k in ('protocol_file_sha256','state4_authorization_file_sha256','state5_authorization_file_sha256','state6_authorization_file_sha256'):
        _hash(value[k])
    for k in ('implementation_commit','implementation_tree','expected_parent_commit'):
        _hash(value[k],40)
    auth = [value['state'+str(n)+'_authorization_file_sha256'] for n in (4,5,6)]
    _require(len(set(auth))==3 and value['attempt_id']=='s4-'+auth[0][:24], 'authorization/attempt binding')
    _ref(value['publication_plan_file'])


def _invocation(binding):
    return _digest(('operatormark-publication-invocation-v1\n'+binding['state6_authorization_file_sha256']+'\n'+binding['publication_plan_file']['file_sha256']).encode())


def _time(value):
    _require(type(value) is str and len(value)==27 and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z',value) is not None, 'timestamp spelling')
    try:
        datetime.strptime(value,'%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError as exc:
        raise PublicationComparisonError('timestamp domain') from exc


def _capture(value,binding,closed=False):
    _obj(value,'invocation_id clock_epoch_id first_sequence last_sequence started_utc finished_utc')
    _require(value['invocation_id']==_invocation(binding),'invocation')
    _text(value['clock_epoch_id'])
    _int(value['first_sequence'])
    _time(value['started_utc'])
    _require((value['last_sequence'] is None)==(value['finished_utc'] is None),'capture endpoints')
    if value['last_sequence'] is None:
        _require(not closed,'closed capture')
    else:
        _int(value['last_sequence'])
        _time(value['finished_utc'])
        _require(value['last_sequence']>=value['first_sequence'] and value['finished_utc']>=value['started_utc'],'capture ordering')


def _failure(value):
    _obj(value,'kind native_error_hex exception_file affected_addresses')
    _enum(value['kind'],('access_denied','authority_unavailable','authority_refused','input_invalid','custody_mismatch','io_failed','result_unavailable','retention_failed','cleanup_failed'))
    if value['native_error_hex'] is not None:
        v=value['native_error_hex']
        _require(type(v)is str and re.fullmatch(r'0x(?:0|[1-9a-f][0-9a-f]*)',v)is not None and int(v,16)<2**32,'native error')
    _ref(value['exception_file'],True,False)
    _unique(value['affected_addresses'],_address,_address_key)


def _observation_row(row):
    _obj(row,'address presence presence_evidence_file kind native_identity content_file security_file permission_handoff_file stable_read_file custody_file')
    _address(row['address'])
    _enum(row['presence'],('present','absent','unknown'))
    _require(row['kind'] is None or row['kind'] in ('file','directory'),'kind')
    _identity(row['native_identity'],True)
    for k in ('presence_evidence_file','content_file','security_file','permission_handoff_file','stable_read_file','custody_file'):
        _ref(row[k],True,k!='content_file')
    if row['presence']=='present':
        _require(all(row[k]is not None for k in ('kind','native_identity','presence_evidence_file')),'present facts')
    if row['presence']=='absent':
        _require(row['presence_evidence_file']is not None and all(row[k]is None for k in row if k not in ('address','presence','presence_evidence_file')),'absence facts')
    _require(row['kind']!='directory' or row['content_file']is None,'directory content')


def _slot(row):
    _obj(row,'address owner role condition creating_address writer_source_file')
    _address(row['address'])
    _enum(row['owner'],('accepted_attempt_M','external_reference_only'))
    _enum(row['role'],('plan_copy','input_projection','intent','result','observation','process','stdout','stderr','details','report','failure','terminal'))
    _enum(row['condition'],('entry','step_reached','phase_reached','failure_observed','terminal_pending'))
    if row['creating_address']is not None:
        _address(row['creating_address'])
    _ref(row['writer_source_file'])


def _wire(record):
    """Independently declared physical grammar; no producer validation call."""
    _require(type(record)is dict and type(record.get('schema'))is str,'record object')
    schema=record['schema']
    kind=next((k for k in _RECORDS if schema==_PREFIX+k+'.v1'),None)
    _require(kind is not None,'closed schema')
    _obj(record,'schema '+_RECORDS[kind]+' record_self_sha256')
    if 'binding'in record:
        _binding(record['binding'])
    if 'capture'in record:
        _capture(record['capture'],record['binding'])
    if 'failures'in record:
        _require(type(record['failures'])is list,'failures array')
        for row in record['failures']:
            _failure(row)
    if kind=='publication_root_bindings':
        _ref(record['publication_plan_file'])
        def root(row):
            _obj(row,'root_binding_id repository_relative_root identity_file custody_file')
            _text(row['root_binding_id'])
            if row['repository_relative_root']is not None:
                _path(row['repository_relative_root'],True)
            _ref(row['identity_file']); _ref(row['custody_file'])
        _unique(record['roots'],root,lambda x:x['root_binding_id'])
        paths=[x['repository_relative_root']for x in record['roots']if x['repository_relative_root']is not None]
        _require(paths.count('')==1 and len({x.casefold()for x in paths})==len(paths),'root path ambiguity')
    elif kind=='publication_physical_observation':
        _enum(record['phase'],_PHASES[:-1]); _ref(record['observer_binding_file'])
        _unique(record['authority_evidence_files'],_ref)
        _unique(record['objects'],_observation_row,lambda x:_address_key(x['address']))
        def enumeration(row):
            _obj(row,'directory children complete evidence_file'); _address(row['directory'])
            _require(type(row['complete'])is bool,'complete bool'); _ref(row['evidence_file'],True)
            _require(not row['complete']or row['evidence_file']is not None,'enumeration proof')
            def child(x):
                _obj(x,'name kind'); _path(x['name']); _require('/'not in x['name'],'child name')
                _enum(x['kind'],('file','directory','unknown'))
            _unique(row['children'],child,lambda x:x['name'])
        _unique(record['enumerations'],enumeration,lambda x:_address_key(x['directory']))
    elif kind=='publication_operation_intent':
        _enum(record['step'],_STEPS); _int(record['sequence'])
        _require(record['operation_id']==_invocation(record['binding'])+':'+record['step'],'operation ID')
        _ref(record['previous_result_file'],True)
        _require((record['sequence']==0)==(record['previous_result_file']is None),'result chain')
        _unique(record['precondition_files'],_ref)
        for k in ('source_address','destination_address'):
            if record[k]is not None:_address(record[k])
        def prepared(x):
            _obj(x,'destination file'); _address(x['destination']); _ref(x['file'],positive=False)
        _unique(record['prepared_files'],prepared,lambda x:_address_key(x['destination']))
        if record['step']in ('promote','quarantine'):
            _require(record['source_address']is not None and record['destination_address']is not None and record['source_address']!=record['destination_address']and not record['prepared_files'],'move shape')
        if record['step']=='release_resources':
            _require(record['source_address']is None and record['destination_address']is None and not record['prepared_files'],'release shape')
        if record['step']=='publish_complete':
            _require(len(record['prepared_files'])==1 and record['prepared_files'][0]['destination']['relative_path'].split('/')[-1]=='COMPLETE.json','completion shape')
    elif kind=='publication_operation_result':
        _enum(record['step'],_STEPS); _ref(record['intent_file'])
        _enum(record['call_status'],('not_called','returned_success','returned_failure','outcome_unavailable'))
        _enum(record['effect_status'],('none_observed','expected_effect_observed','unexpected_effect_observed','unknown'))
        _ref(record['before_observation_file'],True); _ref(record['after_observation_file'],True)
        _unique(record['retained_files'],lambda x:_ref(x,positive=False))
        if record['call_status']=='not_called':
            _require(record['effect_status']!='expected_effect_observed'and record['failures']and record['before_observation_file']is not None,'not-called proof')
        if record['effect_status']=='expected_effect_observed':
            _require(record['after_observation_file']is not None,'effect observation')
    elif kind=='publication_process_outcome':
        _enum(record['role'],('state6_verification_child','publication_finalizer'))
        _ref(record['source_closure_file']); _ref(record['authority_scope_file']); _ref(record['guards_file'],True)
        _enum(record['creation_status'],('not_started','started','unknown'))
        _enum(record['reap_status'],('not_applicable','observed_reaped','unresolved','failed'))
        if record['process_identity']is not None:
            _obj(record['process_identity'],'pid creation_evidence_file'); _int(record['process_identity']['pid'],1); _ref(record['process_identity']['creation_evidence_file'])
        _require(record['direct_exit_code']is None or type(record['direct_exit_code'])is int,'exit type')
        if record['direct_exit_code']is not None:_require(record['creation_status']=='started','exit before creation')
        if record['reap_status']=='observed_reaped':_require(record['creation_status']=='started'and record['direct_exit_code']is not None,'reap evidence')
        for k in ('stdout','stderr'):
            row=record[k]; _obj(row,'state file'); _enum(row['state'],('complete','partial','unavailable','not_started')); _ref(row['file'],True,False)
            _require((row['file']is not None)==(row['state']in ('complete','partial')),'stream accounting')
        if record['creation_status']=='not_started':
            _require(record['process_identity']is None and record['direct_exit_code']is None and record['reap_status']=='not_applicable'and all(record[k]['state']=='not_started'for k in ('stdout','stderr')),'not-started contradiction')
    elif kind=='publication_retention_inventory':
        _address(record['inventory_address']); _ref(record['prior_inventory_file'],True)
        _unique(record['slots'],_slot,lambda x:_address_key(x['address']))
        _require(record['inventory_address']in [x['address']for x in record['slots']],'own inventory slot')
        def observed(x):
            _obj(x,'address state file'); _address(x['address']); _enum(x['state'],('retained_complete','retained_partial','unavailable','not_reached')); _ref(x['file'],True,False)
            _require(x['state']!='retained_complete'or x['file']is not None,'complete bytes')
            _require(x['state']not in ('unavailable','not_reached')or x['file']is None,'null not zero')
            _require(x['file']is None or _address_key(x['file'])==_address_key(x['address']),'retained location')
        _unique(record['observed_files'],observed,lambda x:_address_key(x['address']))
        _require(record['inventory_address']not in [x['address']for x in record['observed_files']],'self-referential inventory')
    else:
        _capture(record['closed_capture'],record['binding'],True)
        _enum(record['disposition'],('publication_observed','refused_before_mutation','failed_preserved','unresolved'))
        _enum(record['located_root'],('T','F','Q','unknown'))
        _require(record['production_preflight_accepted']is False and record['execution_authorized']is False,'false authority flags')
        _require(type(record['operation_files'])is list,'operation trace')
        steps=[]
        for x in record['operation_files']:
            _obj(x,'step intent_file result_file'); _enum(x['step'],_STEPS); _ref(x['intent_file']); _ref(x['result_file'],True); steps.append(x['step'])
            _require(x['result_file']is not None or record['disposition']=='unresolved','unresolved result')
        _require(len(set(steps))==len(steps),'duplicate steps')
        for k in ('process_outcome_files','failure_evidence_files'):
            _require(type(record[k])is list,'references array')
            for x in record[k]:_ref(x)
        for k in ('physical_comparison_file','final_audit_file','completion_file'):_ref(record[k],True)
        _ref(record['retention_inventory_file'])
        if record['disposition']=='publication_observed':
            _require(tuple(steps)==_FORWARD and record['located_root']=='F'and not record['failure_evidence_files']and len(record['process_outcome_files'])>=2 and all(record[k]is not None for k in ('physical_comparison_file','final_audit_file','completion_file')),'incomplete successful terminal')
    _self_hash(record)
    return kind


@dataclass(frozen=True)
class PublicationExpectations:
    binding: object
    plan_expectations: object
    accepted_input_files: object
    object_requirements: object
    phase_requirements: object
    provider_bindings: object
    source_closure_file: object
    authority_scope_file: object
    qualification_file: object
    retention_inventory_file: object

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self,field.name,_freeze(getattr(self,field.name)))


def _projection(expected):
    return {f.name:_plain(getattr(expected,f.name))for f in fields(expected)}


def validate_publication_expectations(expected):
    _require(type(expected)is PublicationExpectations,'ten-field independent expectations required')
    e=_projection(expected); _binding(e['binding'])
    try:
        p=plan_verify.PublicationPlanExpectations(**e['plan_expectations'])
        plan_verify.validate_publication_plan_expectations(p)
    except (TypeError,ValueError)as exc:
        raise PublicationComparisonError('malformed independent plan expectations')from exc
    for k in e['plan_expectations']:
        if k in e['binding']:_require(e['binding'][k]==e['plan_expectations'][k],'plan/physical expectation binding disagreement')
    def accepted(row):
        _obj(row,'role file'); _text(row['role']); _ref(row['file'])
    _unique(e['accepted_input_files'],accepted,lambda x:x['role'])
    roles={x['role']for x in e['accepted_input_files']}
    _require(set(_REQUIRED)<=roles,'required accepted role absent')
    def requirement(row):
        _obj(row,'address role origin_file native_identity content_file phase_security_files custody_file')
        _address(row['address'])
        _enum(row['role'],('ancestor','attempt_directory','publication_directory','quarantine_directory','original_state4_file','state5_member','state6_metadata','final_audit','completion','evidence_file'))
        _identity(row['native_identity'],True)
        for k in ('origin_file','content_file','custody_file'):_ref(row[k],True,k!='content_file')
        def security(x):
            _obj(x,'phase file'); _enum(x['phase'],_PHASES[:-1]); _ref(x['file'])
        _unique(row['phase_security_files'],security,lambda x:x['phase'])
        if row['role']=='original_state4_file':
            _require(all(row[k]is not None for k in ('origin_file','native_identity','content_file','custody_file')),'historical State 4 evidence cannot be invented')
    _unique(e['object_requirements'],requirement,lambda x:_address_key(x['address']))
    def phase(row):
        _obj(row,'phase required_present required_absent required_unchanged allowed_new required_evidence_roles')
        _enum(row['phase'],_PHASES)
        for k in ('required_present','required_absent','required_unchanged','allowed_new'):_unique(row[k],_address,_address_key)
        sets={k:{_address_key(x)for x in row[k]}for k in ('required_present','required_absent','required_unchanged','allowed_new')}
        _require(not sets['required_absent']&(sets['required_present']|sets['required_unchanged']|sets['allowed_new']),'contradictory phase sets')
        _require(sets['required_unchanged']<=sets['required_present']and not sets['required_unchanged']&sets['allowed_new'],'unchanged/new contradiction')
        _unique(row['required_evidence_roles'],_text,lambda x:x)
    _unique(e['phase_requirements'],phase,lambda x:x['phase'])
    _require({x['phase']for x in e['phase_requirements']}==set(_PHASES),'all phase requirements must be explicit')
    def provider(row):
        _obj(row,'role source_file enrollment_file qualification_file authority_scope_file')
        _enum(row['role'],('operator','observer','evidence_reader','evidence_writer','process_supervisor'))
        for k in ('source_file','enrollment_file','qualification_file','authority_scope_file'):_ref(row[k])
    _unique(e['provider_bindings'],provider,lambda x:x['role'])
    _require({x['role']for x in e['provider_bindings']}=={'operator','observer','evidence_reader','evidence_writer','process_supervisor'},'complete conditional provider bindings')
    for k in ('source_closure_file','authority_scope_file','qualification_file','retention_inventory_file'):_ref(e[k])


class _Comparison:
    def __init__(self, expected, phase):
        self.expected=expected; self.e=_projection(expected); self.phase=phase
        self.rows={k:[]for k in ('mismatches','invalid_inputs','unavailable_requirements')}
        self.checked=set(); self.input_files=[]; self.roots=None; self.plan=None; self.template=None
        self.independent={}; self.candidate={}; self.parsed={}; self.valid={}
        self.located='unknown'; self.effect='unknown'
    def add(self,role,pointer,code):
        bucket='invalid_inputs'if code in _INVALID else 'mismatches'if code in _MISMATCH else 'unavailable_requirements'
        _require(code in _INVALID|_MISMATCH|_UNAVAILABLE,'closed diagnostic code')
        self.rows[bucket].append(dict(input=role,pointer=pointer,code=code))
    def ordered(self):
        return {k:[dict(input=r,pointer=p,code=c)for r,p,c in sorted({(x['input'],x['pointer'],x['code'])for x in v})]for k,v in self.rows.items()}
    def status(self):
        return 'invalid'if self.rows['invalid_inputs']else 'mismatch'if self.rows['mismatches']else 'incomplete'if self.rows['unavailable_requirements']else 'matched'
    def by_ref(self,ref,role='root_bindings',pointer='',code='required_bytes_unavailable'):
        if ref is None:
            self.add(role,pointer,code); return None
        matches=[x for x in self.independent.values()if x['file']==ref and x['data']is not None]
        if not matches:
            self.add(role,pointer,code); return None
        return matches[0]['data']


def _collect(ctx, collection, label):
    known={x['role']:x['file']for x in ctx.e['accepted_input_files']}
    allowed=set(known)|{r for p in ctx.e['phase_requirements']for r in p['required_evidence_roles']}
    result={}; duplicate=set()
    if type(collection)is not tuple:
        ctx.add('root_bindings','/'+label,'schema_invalid'); return result
    for index,frozen in enumerate(collection):
        role='root_bindings'
        try:
            _require(type(frozen)is MappingProxyType,'frozen file envelope')
            row=_plain(frozen); _obj(row,'role file data'); _text(row['role'])
            _require(row['role']in allowed,'unapproved role')
            role=row['role']; _ref(row['file'],positive=False)
            _require(row['data']is None or type(row['data'])is bytes,'immutable bytes/None envelope')
            actual=None if row['data']is None else dict(row['file'],file_bytes=len(row['data']),file_sha256=_digest(row['data']))
            ctx.input_files.append(dict(role=label+':'+role,file=actual))
            if role in result:duplicate.add(role)
            result[role]=row
            if row['data']is None:
                ctx.add(role,'/'+label+'/'+str(index),'required_bytes_unavailable')
            else:
                if actual!=row['file']:ctx.add(role,'/'+label+'/'+str(index)+'/file','accepted_binding_mismatch')
                if role in known and row['file']!=known[role]:ctx.add(role,'/'+label+'/'+str(index)+'/file','accepted_binding_mismatch')
                if label=='independent'and actual!=row['file']:duplicate.add(role)
        except (PublicationComparisonError,TypeError,KeyError):
            ctx.add(role,'/'+label+'/'+str(index),'schema_invalid')
    for role in duplicate:
        ctx.add(role,'/'+label,'schema_invalid'); result.pop(role,None)
    return result


def _read_record(ctx, role, raw):
    try:
        value=_decode(raw)
    except PublicationComparisonError:
        ctx.add(role,'','canonical_encoding_invalid'); return None,False
    try:
        _wire(value)
        return value,True
    except PublicationComparisonError as exc:
        code='self_hash_invalid'if str(exc)=='self_hash_invalid'else 'schema_invalid'
        ctx.add(role,'',code)
        return value if type(value)is dict else None,False


def _resolve_publication_address(address, roots, *, repository_only=False):
    _address(address)
    _require(roots is not None,'unresolved root input')
    table={r['root_binding_id']:r for r in roots['roots']}
    _require(address['root_binding_id']in table,'unknown root ID')
    anchor=table[address['root_binding_id']]['repository_relative_root']
    if anchor is None:
        _require(not repository_only,'external root is read-only input')
        return ('external',address['root_binding_id'],address['relative_path'])
    location='/'.join(x for x in (anchor,address['relative_path'])if x)
    choices=[r for r in roots['roots']if r['repository_relative_root']is not None
             and (not r['repository_relative_root']or location==r['repository_relative_root']or location.startswith(r['repository_relative_root']+'/'))]
    canonical=max(choices,key=lambda r:len(r['repository_relative_root']))
    _require(canonical['root_binding_id']==address['root_binding_id'],'noncanonical anchor selection')
    for r in roots['roots']:
        path=r['repository_relative_root']
        if path and (location.casefold()==path.casefold()or location.casefold().startswith(path.casefold()+'/')):
            _require(location==path or location.startswith(path+'/'),'root path casefold alias')
    return ('repository',location)


def _address_for(location, roots):
    choices=[r for r in roots['roots']if r['repository_relative_root']is not None and
             (not r['repository_relative_root']or location==r['repository_relative_root']or location.startswith(r['repository_relative_root']+'/'))]
    _require(bool(choices),'no repository anchor')
    row=max(choices,key=lambda r:len(r['repository_relative_root']))
    base=row['repository_relative_root']
    return dict(root_binding_id=row['root_binding_id'],relative_path=location[len(base):].lstrip('/'))


def _walk_references(value):
    if type(value)is dict:
        if set(value)=={'root_binding_id','relative_path'}or set(value)=={'root_binding_id','relative_path','file_bytes','file_sha256'}:
            yield value
        else:
            for v in value.values():yield from _walk_references(v)
    elif type(value)is list:
        for v in value:yield from _walk_references(v)


def _decode_root_bindings(ctx):
    role='root_bindings'
    expected_ref=next(x['file']for x in ctx.e['accepted_input_files']if x['role']==role)
    envelope=ctx.independent.get(role)
    if envelope is None or envelope['data']is None:
        ctx.add(role,'','required_bytes_unavailable'); return
    if envelope['file']!=expected_ref:
        ctx.add(role,'/file','accepted_binding_mismatch'); return
    raw=envelope['data']
    if len(raw)!=expected_ref['file_bytes']or _digest(raw)!=expected_ref['file_sha256']:
        ctx.add(role,'/file','accepted_binding_mismatch'); return
    value,valid=_read_record(ctx,role,raw)
    if not valid or value.get('schema')!=_PREFIX+'publication_root_bindings.v1':
        if valid:ctx.add(role,'/schema','schema_invalid')
        return
    ctx.checked.add('/independent/root_bindings/file')
    if value['publication_plan_file']!=ctx.e['binding']['publication_plan_file']:
        ctx.add(role,'/publication_plan_file','accepted_binding_mismatch')
    ctx.roots=value
    accepted=[r['file']for r in ctx.e['accepted_input_files']if r['role']!=role]
    identities=[]
    a4=ctx.e['binding']['state4_authorization_file_sha256']
    base='artifacts/darcy_branch_c_v1/private_corpus_v1/'
    moving=(base+'attempts/s4-'+a4[:24],base+'corpus',base+'quarantine/s4-'+a4[:24])
    for index,row in enumerate(value['roots']):
        ptr='/roots/'+str(index)
        root=row['repository_relative_root']
        if root is not None and any(root==p or root.startswith(p+'/')for p in moving):
            ctx.add(role,ptr+'/repository_relative_root','inventory_mismatch')
        address=dict(root_binding_id=row['root_binding_id'],relative_path='')
        reqs=[r for r in ctx.e['object_requirements']if r['address']==address]
        if len(reqs)!=1:
            ctx.add(role,ptr,'inventory_mismatch'); continue
        req=reqs[0]
        if req['role']!='ancestor'or req['native_identity']is None or req['origin_file']!=row['identity_file']or req['custody_file']!=row['custody_file']:
            ctx.add(role,ptr,'accepted_binding_mismatch')
        else:identities.append(_canonical(req['native_identity']))
        for key,code in (('identity_file','required_bytes_unavailable'),('custody_file','custody_unavailable')):
            if row[key]not in accepted:ctx.add(role,ptr+'/'+key,'accepted_binding_mismatch')
            ctx.by_ref(row[key],role,ptr+'/'+key,code)
        for phase in ctx.e['phase_requirements']:
            if address not in phase['required_present']or address not in phase['required_unchanged']:
                ctx.add(role,ptr,'inventory_mismatch')
    if len(set(identities))!=len(identities):ctx.add(role,'/roots','identity_mismatch')
    locations={}
    for index,req in enumerate(ctx.e['object_requirements']):
        try:
            loc=_resolve_publication_address(req['address'],value)
            folded=tuple(p.casefold()for p in loc)
            if folded in locations:ctx.add(role,'/object_requirements/'+str(index),'inventory_mismatch')
            locations[folded]=index
        except PublicationComparisonError:
            ctx.add(role,'/object_requirements/'+str(index)+'/address','accepted_binding_mismatch')
    reference_locations={}
    for index,ref in enumerate(_walk_references(ctx.e)):
        try:
            loc=_resolve_publication_address({k:ref[k]for k in ('root_binding_id','relative_path')},value)
            if 'file_sha256'in ref:
                descriptor=(ref['file_bytes'],ref['file_sha256'])
                if loc in reference_locations and reference_locations[loc]!=descriptor:
                    ctx.add(role,'/expected_references/'+str(index),'accepted_binding_mismatch')
                reference_locations[loc]=descriptor
        except PublicationComparisonError:ctx.add(role,'/expected_references/'+str(index),'accepted_binding_mismatch')
    for phase in ctx.e['phase_requirements']:
        for index,address in enumerate(phase['allowed_new']):
            try:_resolve_publication_address(address,value,repository_only=True)
            except PublicationComparisonError:ctx.add(role,'/phase_requirements/allowed_new/'+str(index),'inventory_mismatch')


def _plan_and_inputs(ctx):
    e=ctx.e
    for row in e['accepted_input_files']:
        code='custody_unavailable'if 'custody'in row['role']else 'required_bytes_unavailable'
        raw=ctx.by_ref(row['file'],row['role'],'',code)
        if raw is not None:ctx.checked.add('/independent/'+row['role']+'/file')
    rolemap={r['role']:r['file']for r in e['accepted_input_files']}
    for role,key in (('protocol','protocol_file_sha256'),('state4_authorization','state4_authorization_file_sha256'),('state5_authorization','state5_authorization_file_sha256'),('state6_authorization','state6_authorization_file_sha256')):
        if rolemap[role]['file_sha256']!=e['binding'][key]:ctx.add(role,'/file_sha256','accepted_binding_mismatch')
    for key,code in (('source_closure_file','required_bytes_unavailable'),('authority_scope_file','authority_evidence_unavailable'),('qualification_file','qualification_unavailable')):
        ctx.by_ref(e[key],'root_bindings','/'+key,code)
    for i,provider in enumerate(e['provider_bindings']):
        for key in ('source_file','enrollment_file','qualification_file','authority_scope_file'):
            code='qualification_unavailable'if key=='qualification_file'else 'authority_evidence_unavailable'if key=='authority_scope_file'else 'required_bytes_unavailable'
            ctx.by_ref(provider[key],'root_bindings','/provider_bindings/'+str(i)+'/'+key,code)
    raw=ctx.by_ref(e['binding']['publication_plan_file'],'root_bindings','/publication_plan_file')
    pre=ctx.by_ref(rolemap['state4_pre_solve_manifest'],'state4_pre_solve_manifest')
    end=ctx.by_ref(rolemap['state4_controls_completion'],'state4_controls_completion')
    execution=ctx.by_ref(rolemap['state5_execution'],'state5_execution')
    report=plan_verify.verify_publication_plan(raw,expected=plan_verify.PublicationPlanExpectations(**e['plan_expectations']),
        state4_pre_solve_manifest_bytes=pre,state4_controls_completion_bytes=end,state5_execution_evidence_bytes=execution)
    if report['comparison_status']=='matched':
        ctx.plan=_decode(raw);ctx.checked.add('/independent/plan')
    else:
        for category in ('mismatches','invalid_inputs','unavailable_requirements'):
            for row in report[category]:
                code=row['code']
                code={'plan_derivation_invalid':'accepted_binding_mismatch','predecessor_file_mismatch':'content_mismatch'}.get(code,code)
                ctx.add('root_bindings','/plan'+row['pointer'],code)


def _template(ctx):
    raw=ctx.by_ref(ctx.e['retention_inventory_file'],'root_bindings','/retention_inventory_file','retention_incomplete')
    if raw is not None:
        record,valid=_read_record(ctx,'root_bindings',raw)
        if valid and record['schema']==_PREFIX+'publication_retention_inventory.v1':
            ctx.template=record
            if record['binding']!=ctx.e['binding']:ctx.add('root_bindings','/retention_inventory_file/binding','accepted_binding_mismatch')
    if ctx.template is None:
        rows=ctx.ordered()
        raise PublicationComparisonError('valid independent report-slot template unavailable',bootstrap_status=ctx.status(),diagnostics=tuple(x for values in rows.values()for x in values))
    slots=[x for x in ctx.template['slots']if x['role']=='details'and
           (x['address']['relative_path']=='test.sealed/reports/'+ctx.phase+'.json'or x['address']['relative_path'].endswith('/test.sealed/reports/'+ctx.phase+'.json'))]
    if len(slots)!=1:
        ctx.add('root_bindings','/retention_inventory_file/slots','inventory_mismatch')
        rows=ctx.ordered()
        raise PublicationComparisonError('unambiguous details output slot unavailable',bootstrap_status=ctx.status(),diagnostics=tuple(x for values in rows.values()for x in values))
    return slots[0]['address']


def _bootstrap_context(expected,independent_files,phase='pre_finalize',need_template=True):
    validate_publication_expectations(expected)
    _enum(phase,_PHASES)
    ctx=_Comparison(expected,phase)
    ctx.independent=_collect(ctx,independent_files,'independent')
    _decode_root_bindings(ctx)
    _plan_and_inputs(ctx)
    if need_template:
        _template(ctx)
        if ctx.roots is not None and ctx.plan is not None:
            for index,ref in enumerate(_walk_references(ctx.template)):
                _location(ctx,{k:ref[k]for k in ('root_binding_id','relative_path')},'root_bindings','/retention_inventory/references/'+str(index))
            _retention_checks(ctx)
    return ctx


def _bootstrap_entry(expected,entry_files):
    ctx=_bootstrap_context(expected,entry_files)
    if ctx.status()!='matched':
        rows=ctx.ordered()
        raise PublicationComparisonError('entry bootstrap failed',bootstrap_status=ctx.status(),diagnostics=tuple(x for values in rows.values()for x in values))
    return ctx


def _location(ctx,address,role,pointer,write=False):
    try:return _resolve_publication_address(address,ctx.roots,repository_only=write)
    except PublicationComparisonError:
        ctx.add(role,pointer,'accepted_binding_mismatch'); return None


def _record_population(ctx):
    if ctx.template is None:return
    slots={_address_key(x['address']):x for x in ctx.template['slots']}
    physical_roles={'intent','result','observation','process','terminal'}
    for role,envelope in ctx.independent.items():
        if envelope['data']is None:continue
        slot=slots.get(_address_key(envelope['file']))
        if slot is None or (slot['role']not in physical_roles and not (slot['role']=='input_projection' and '/inventories/' in slot['address']['relative_path'])):continue
        record,valid=_read_record(ctx,role,envelope['data'])
        if record is not None:
            ctx.parsed[role]=record;ctx.valid[role]=valid
            if 'binding'in record and record['binding']!=ctx.e['binding']:
                ctx.add(role,'/binding','accepted_binding_mismatch');ctx.valid[role]=False
            if ctx.roots is not None:
                for index,ref in enumerate(_walk_references(record)):
                    _location(ctx,{k:ref[k]for k in ('root_binding_id','relative_path')},role,'/references/'+str(index))
    for role,row in ctx.candidate.items():
        peer=ctx.independent.get(role)
        if row['data']is not None:
            if peer is None or peer['data']is None:
                ctx.add(role,'/independent','required_bytes_unavailable')
            else:
                ctx.checked.add('/candidate/'+role)
                if row['file']!=peer['file']or row['data']!=peer['data']:
                    ctx.add(role,'/candidate','accepted_binding_mismatch')
            slot=slots.get(_address_key(row['file']))
            if slot is not None and slot['role']in physical_roles:
                _read_record(ctx,role,row['data'])


def _records(ctx,kind,valid=True):
    return [(r,x)for r,x in ctx.parsed.items()if x.get('schema')==_PREFIX+kind+'.v1'and (ctx.valid.get(r)or not valid)]


def _one_record(ctx,kind,key,value):
    matches=[(r,x)for r,x in _records(ctx,kind,False)if x.get(key)==value]
    if len(matches)!=1:
        ctx.add('root_bindings','/'+kind+'/'+str(value),'schema_invalid'if matches else 'required_bytes_unavailable')
        return None,None
    return matches[0]


def _usable_rows(ctx,role,rows,check,key,pointer):
    groups={}
    if type(rows)is not list:
        ctx.add(role,pointer,'schema_invalid');return {}
    for index,row in enumerate(rows):
        try:
            check(row)
        except (PublicationComparisonError,KeyError,TypeError):
            ctx.add(role,pointer+'/'+str(index),'schema_invalid')
            continue
        try:
            k=key(row);groups.setdefault(k,[]).append(row)
        except (PublicationComparisonError,KeyError,TypeError):
            ctx.add(role,pointer+'/'+str(index),'accepted_binding_mismatch')
    usable={}
    for k,values in groups.items():
        if len(values)==1:usable[k]=values[0]
        else:ctx.add(role,pointer,'schema_invalid')
    return usable


def _requirements_at(ctx,phase):
    roots={k:ctx.plan[k+'_root_relative_path']for k in ('attempt','final','quarantine','publication_evidence')}
    origin=roots['attempt'];destination=roots['final']if phase in ('post_move','post_audit','post_complete','terminal')else roots['quarantine']if phase=='post_quarantine'else origin
    if phase=='failure_location':
        original=[r for r in ctx.e['object_requirements']if r['role']=='attempt_directory']
        found=[]
        if len(original)==1:
            for _,obs in _records(ctx,'publication_physical_observation'):
                if obs['phase']!='failure_location':continue
                for row in obs['objects']:
                    if row['presence']=='present'and row['native_identity']==original[0]['native_identity']and row['custody_file']==original[0]['custody_file']:
                        loc=_location(ctx,row['address'],'root_bindings','/failure_location')
                        if loc in [('repository',roots[k])for k in ('attempt','final','quarantine')]:found.append(loc[1])
        if len(found)==1:destination=found[0]
    result={}
    for index,req in enumerate(ctx.e['object_requirements']):
        loc=_location(ctx,req['address'],'root_bindings','/object_requirements/'+str(index))
        if loc is None:continue
        if loc[0]=='repository'and destination!=origin and (loc[1]==origin or loc[1].startswith(origin+'/')):
            loc=('repository',destination+loc[1][len(origin):])
        if loc in result:
            ctx.add('root_bindings','/object_requirements/'+str(index),'inventory_mismatch')
        else:result[loc]=(index,req)
    return result


def _observation_checks(ctx,phase):
    # Location belongs to this observation, never to the last usable predecessor.
    # Earlier checks/diagnostics remain in the shared comparison context.
    ctx.located='unknown'
    role,record=_one_record(ctx,'publication_physical_observation','phase',phase)
    if record is None or ctx.roots is None or ctx.plan is None:return
    try:_binding(record['binding']);_capture(record['capture'],record['binding'])
    except (PublicationComparisonError,KeyError):return
    observer=next(p for p in ctx.e['provider_bindings']if p['role']=='observer')
    if record.get('observer_binding_file')!=observer['source_file']:
        ctx.add(role,'/observer_binding_file','source_binding_mismatch')
    ctx.by_ref(record.get('observer_binding_file'),role,'/observer_binding_file')
    if ctx.e['authority_scope_file']not in record.get('authority_evidence_files',[]):
        ctx.add(role,'/authority_evidence_files','authority_evidence_unavailable')
    phase_req=next(x for x in ctx.e['phase_requirements']if x['phase']==phase)
    usable=_usable_rows(ctx,role,record.get('objects'),_observation_row,lambda x:_resolve_publication_address(x['address'],ctx.roots),'/objects')
    native_groups={}
    for loc,row in usable.items():
        if row['presence']=='present':
            native_groups.setdefault(_canonical(row['native_identity']),[]).append(loc)
    ambiguous={loc for values in native_groups.values()if len(values)>1 for loc in values}
    if ambiguous:ctx.add(role,'/objects','identity_mismatch')
    usable={k:v for k,v in usable.items()if k not in ambiguous}
    requirements=_requirements_at(ctx,phase)
    for index,address in enumerate(phase_req['required_present']):
        loc=_location(ctx,address,role,'/required_present/'+str(index))
        row=usable.get(loc);pointer='/objects/'+str(index)
        if row is None or row['presence']=='unknown':
            ctx.add(role,pointer,'required_bytes_unavailable');continue
        ctx.checked.add('/independent/'+role+pointer+'/presence')
        if row['presence']!='present':
            ctx.add(role,pointer+'/presence','inventory_mismatch');continue
        ctx.by_ref(row['presence_evidence_file'],role,pointer+'/presence_evidence_file')
        entry=requirements.get(loc)
        if entry is None:
            ctx.add(role,pointer,'inventory_mismatch');continue
        _,req=entry
        identity=req['native_identity'];content=req['content_file']
        # A new slot is bound only by an independent exclusive-creation result and prepared bytes.
        if identity is None:
            creations=[]
            for ir,intent in _records(ctx,'publication_operation_intent'):
                for prepared in intent['prepared_files']:
                    original_loc=_location(ctx,req['address'],role,pointer+'/origin_address')
                    if _location(ctx,prepared['destination'],ir,'/prepared_files')==original_loc:
                        results=[(rr,res)for rr,res in _records(ctx,'publication_operation_result')if res['intent_file']==ctx.independent[ir]['file']and res['call_status']=='returned_success'and res['effect_status']not in ('unknown','unexpected_effect_observed')]
                        if len(results)==1 and ctx.by_ref(prepared['file'],ir,'/prepared_files')is not None:
                            firsts=[]
                            for orole,obs in _records(ctx,'publication_physical_observation'):
                                for candidate in obs['objects']:
                                    candidate_loc=_location(ctx,candidate['address'],orole,'/objects/address')
                                    if candidate['presence']=='present'and candidate_loc in (original_loc,loc)and candidate['custody_file']is not None:
                                        firsts.append((obs['capture']['first_sequence'],candidate))
                            if firsts:
                                first=min(firsts,key=lambda x:x[0])[1]
                                creations.append((first['native_identity'],prepared['file']))
            if len(creations)==1:identity,content=creations[0]
            else:ctx.add(role,pointer+'/native_identity','custody_unavailable')
        if identity is not None:
            ctx.checked.add('/independent/'+role+pointer+'/native_identity')
            if row['native_identity']!=identity:ctx.add(role,pointer+'/native_identity','identity_mismatch')
        if phase not in ('failure_location','post_quarantine')and content is not None:
            if row['content_file']is None:ctx.add(role,pointer+'/content_file','required_bytes_unavailable')
            else:
                ctx.checked.add('/independent/'+role+pointer+'/content_file')
                if row['content_file']!=content:ctx.add(role,pointer+'/content_file','content_mismatch')
                ctx.by_ref(row['content_file'],role,pointer+'/content_file')
        if req['custody_file']is not None:
            if row['custody_file']is None:ctx.add(role,pointer+'/custody_file','custody_unavailable')
            elif row['custody_file']!=req['custody_file']:ctx.add(role,pointer+'/custody_file','accepted_binding_mismatch')
            else:ctx.by_ref(req['custody_file'],role,pointer+'/custody_file','custody_unavailable')
        elif row['custody_file']is None:ctx.add(role,pointer+'/custody_file','custody_unavailable')
        policies=[x['file']for x in req['phase_security_files']if x['phase']==phase]
        if len(policies)!=1:ctx.add(role,pointer+'/security_file','required_bytes_unavailable')
        elif row['security_file']is None:ctx.add(role,pointer+'/security_file','required_bytes_unavailable')
        else:
            ctx.checked.add('/independent/'+role+pointer+'/security_file')
            if row['security_file']!=policies[0]:ctx.add(role,pointer+'/security_file','security_mismatch')
            ctx.by_ref(policies[0],role,pointer+'/security_file')
        for key in ('permission_handoff_file','stable_read_file'):
            if phase not in ('failure_location','post_quarantine'):
                ctx.by_ref(row[key],role,pointer+'/'+key,'custody_unavailable')
    for index,address in enumerate(phase_req['required_absent']):
        loc=_location(ctx,address,role,'/required_absent/'+str(index));row=usable.get(loc)
        if row is None or row['presence']=='unknown':ctx.add(role,'/required_absent/'+str(index),'required_bytes_unavailable')
        else:
            ctx.checked.add('/independent/'+role+'/required_absent/'+str(index))
            if row['presence']!='absent':ctx.add(role,'/required_absent/'+str(index),'inventory_mismatch')
            ctx.by_ref(row['presence_evidence_file'],role,'/required_absent/'+str(index))
    expected_present={_location(ctx,a,role,'/required_present')for a in phase_req['required_present']}
    expected_absent={_location(ctx,a,role,'/required_absent')for a in phase_req['required_absent']}
    for loc,row in usable.items():
        if row['presence']=='present'and loc not in expected_present:
            ctx.add(role,'/objects','inventory_mismatch')
    enumerations=record.get('enumerations',[])
    if type(enumerations)is not list:
        ctx.add(role,'/enumerations','schema_invalid')
        enumerations=[]
    for index,row in enumerate(enumerations if type(enumerations)is list else []):
        try:
            _obj(row,'directory children complete evidence_file')
            directory=_resolve_publication_address(row['directory'],ctx.roots)
            if row['complete']is not True or row['evidence_file']is None:
                ctx.add(role,'/enumerations/'+str(index),'inventory_incomplete');continue
            ctx.by_ref(row['evidence_file'],role,'/enumerations/'+str(index))
            children=row['children']
            _unique(children,lambda x:(_obj(x,'name kind'),_path(x['name']),_enum(x['kind'],('file','directory','unknown'))),lambda x:x['name'])
            if directory[0]!='repository':continue
            prefix=directory[1]+'/'if directory[1]else ''
            desired=sorted((loc[1][len(prefix):],usable[loc]['kind'])for loc in expected_present if loc in usable and loc[0]=='repository'and loc[1].startswith(prefix)and '/'not in loc[1][len(prefix):]and loc[1]!=directory[1])
            actual=[(x['name'],x['kind'])for x in children]
            if actual!=desired:ctx.add(role,'/enumerations/'+str(index)+'/children','inventory_mismatch')
            elif all(loc in usable and usable[loc]['presence']=='present'for loc in expected_present):
                ctx.checked.add('/independent/'+role+'/enumerations/'+str(index)+'/complete_membership')
        except (PublicationComparisonError,KeyError,TypeError):
            ctx.add(role,'/enumerations/'+str(index),'schema_invalid')
    # Every required present directory needs a complete independently retained enumeration.
    observed_dirs={_location(ctx,x['directory'],role,'/enumerations')for x in enumerations if type(x)is dict and 'directory'in x}
    for loc in expected_present:
        if loc in usable and usable[loc]['kind']=='directory'and loc not in observed_dirs:
            ctx.add(role,'/enumerations','inventory_incomplete')
    attempt_req=[r for r in ctx.e['object_requirements']if r['role']=='attempt_directory'and r['native_identity']is not None]
    if len(attempt_req)==1:
        native=attempt_req[0]['native_identity']
        locations=[]
        for token,key in (('T','attempt_root_relative_path'),('F','final_root_relative_path'),('Q','quarantine_root_relative_path')):
            row=usable.get(('repository',ctx.plan[key]))
            if row and row['presence']=='present'and row['native_identity']==native and row['custody_file']==attempt_req[0]['custody_file']:
                locations.append(token)
        if len(locations)==1 and ctx.valid.get(role) and record['capture']['last_sequence']is not None:
            ctx.located=locations[0]
        elif len(locations)>1:ctx.add(role,'/objects','identity_mismatch')
    if record.get('failures'):
        ctx.add(role,'/failures','unexpected_effect')
    if record['capture']['last_sequence']is None:ctx.add(role,'/capture','custody_unavailable')


def _chronology(ctx,phases,steps):
    previous=None;epoch=None
    observations={}
    for phase in phases:
        role,obs=_one_record(ctx,'publication_physical_observation','phase',phase)
        if obs is None or not ctx.valid.get(role):continue
        cap=obs['capture'];observations[phase]=(role,obs)
        if cap['last_sequence']is None:ctx.add(role,'/capture','custody_unavailable');continue
        if epoch is None:epoch=cap['clock_epoch_id']
        if cap['clock_epoch_id']!=epoch or previous is not None and cap['first_sequence']<=previous:
            ctx.add(role,'/capture','chronology_invalid')
        previous=cap['last_sequence']
    intents=[];results=[];last=None
    for index,step in enumerate(steps):
        ir,intent=_one_record(ctx,'publication_operation_intent','step',step)
        rr,result=_one_record(ctx,'publication_operation_result','step',step)
        if intent is None or result is None:continue
        if not ctx.valid.get(ir)or not ctx.valid.get(rr):continue
        if intent['sequence']!=index or intent['previous_result_file']!=last:
            ctx.add(ir,'/sequence','chronology_invalid')
        if result['intent_file']!=ctx.independent[ir]['file']:
            ctx.add(rr,'/intent_file','accepted_binding_mismatch')
        cap=result['capture']
        if cap['clock_epoch_id']!=epoch or cap['last_sequence']is None:
            ctx.add(rr,'/capture','chronology_invalid'if cap['clock_epoch_id']!=epoch else 'operation_outcome_unavailable')
        if results:
            prior_end=results[-1][1]['capture']['last_sequence']
            if prior_end is None:ctx.add(rr,'/capture','operation_outcome_unavailable')
            elif cap['first_sequence']<=prior_end:ctx.add(rr,'/capture','chronology_invalid')
        if result['call_status']!='returned_success':
            ctx.add(rr,'/call_status','operation_outcome_unavailable'if result['call_status']=='outcome_unavailable'else 'unexpected_effect')
        if result['effect_status']=='unknown':ctx.add(rr,'/effect_status','operation_outcome_unavailable')
        if result['effect_status']=='unexpected_effect_observed'or result['failures']:ctx.add(rr,'/effect_status','unexpected_effect')
        for key in ('before_observation_file','after_observation_file'):
            if result[key]is not None:
                raw=ctx.by_ref(result[key],rr,'/'+key)
                if raw is not None:
                    candidates=[o for _,o in _records(ctx,'publication_physical_observation')if _canonical(o)==raw]
                    if len(candidates)!=1:ctx.add(rr,'/'+key,'accepted_binding_mismatch')
                    else:
                        oc=candidates[0]['capture']
                        if oc['clock_epoch_id']!=cap['clock_epoch_id']or oc['last_sequence']is None:
                            ctx.add(rr,'/'+key,'chronology_invalid')
                        elif key=='before_observation_file'and oc['last_sequence']>=cap['first_sequence']:
                            ctx.add(rr,'/'+key,'chronology_invalid')
                        elif key=='after_observation_file'and not cap['first_sequence']<=oc['first_sequence']<=oc['last_sequence']<=cap['last_sequence']:
                            ctx.add(rr,'/'+key,'chronology_invalid')
        for ref in intent['precondition_files']:ctx.by_ref(ref,ir,'/precondition_files')
        if ctx.roots is not None and ctx.plan is not None:
            if step in ('promote','quarantine'):
                src=_location(ctx,intent['source_address'],ir,'/source_address',True)
                dst=_location(ctx,intent['destination_address'],ir,'/destination_address',True)
                expected_dst=ctx.plan['final_root_relative_path'if step=='promote'else 'quarantine_root_relative_path']
                source_options=[ctx.plan['attempt_root_relative_path']]if step=='promote'else [ctx.plan['attempt_root_relative_path'],ctx.plan['final_root_relative_path']]
                if dst!=('repository',expected_dst)or src not in [('repository',x)for x in source_options]:
                    ctx.add(ir,'/destination_address','inventory_mismatch')
            allowed={_location(ctx,a,ir,'/allowed_new',True)for p in ctx.e['phase_requirements']for a in p['allowed_new']}
            for p in intent['prepared_files']:
                loc=_location(ctx,p['destination'],ir,'/prepared_files',True)
                if loc not in allowed:ctx.add(ir,'/prepared_files','inventory_mismatch')
                ctx.by_ref(p['file'],ir,'/prepared_files')
        last=ctx.independent[rr]['file'];intents.append((ir,intent));results.append((rr,result))
    # Reject an independently supplied later forward dispatch, rather than selecting a successful prefix.
    for role,result in _records(ctx,'publication_operation_result'):
        if result['step']not in steps and result['step']not in ('quarantine','release_resources'):
            ctx.add(role,'/step','chronology_invalid')
    return intents,results


def _process_checks(ctx,terminal=False):
    wanted=['state6_verification_child']+(['publication_finalizer']if terminal else [])
    for wanted_role in wanted:
        role,record=_one_record(ctx,'publication_process_outcome','role',wanted_role)
        if record is None or not ctx.valid.get(role):continue
        if record['source_closure_file']!=ctx.e['source_closure_file']:ctx.add(role,'/source_closure_file','source_binding_mismatch')
        if record['authority_scope_file']!=ctx.e['authority_scope_file']:ctx.add(role,'/authority_scope_file','accepted_binding_mismatch')
        if record['creation_status']!='started'or record['direct_exit_code']is None or record['reap_status']!='observed_reaped':
            ctx.add(role,'/direct_exit_code','operation_outcome_unavailable')
        elif record['direct_exit_code']!=0:ctx.add(role,'/direct_exit_code','unexpected_effect')
        else:ctx.checked.add('/independent/'+role+'/direct_exit_code')
        if record['process_identity']is None:ctx.add(role,'/process_identity','required_bytes_unavailable')
        else:ctx.by_ref(record['process_identity']['creation_evidence_file'],role,'/process_identity')
        for key in ('stdout','stderr'):
            if record[key]['state']!='complete':ctx.add(role,'/'+key,'required_bytes_unavailable')
            ctx.by_ref(record[key]['file'],role,'/'+key)
        ctx.by_ref(record['guards_file'],role,'/guards_file')
        if wanted_role=='state6_verification_child':
            expected_guard=next(x['file']for x in ctx.e['accepted_input_files']if x['role']=='state6_child_guards')
            if record['guards_file']is not None and record['guards_file']!=expected_guard:ctx.add(role,'/guards_file','accepted_binding_mismatch')
        if record['failures']:ctx.add(role,'/failures','unexpected_effect')
        if wanted_role=='state6_verification_child':
            observations=[o for _,o in _records(ctx,'publication_physical_observation')if o['phase']=='pre_finalize']
            if len(observations)==1:
                first=observations[0]['capture'];child=record['capture']
                if child['last_sequence']is None or child['clock_epoch_id']!=first['clock_epoch_id']or child['last_sequence']>=first['first_sequence']:
                    ctx.add(role,'/capture','chronology_invalid')


def _retention_checks(ctx,terminal=False):
    if ctx.roots is None or ctx.plan is None:return
    base=ctx.plan['publication_evidence_root_relative_path']
    slots={_address_key(x['address']):x for x in ctx.template['slots']}
    locations={}
    for index,slot in enumerate(ctx.template['slots']):
        loc=_location(ctx,slot['address'],'root_bindings','/retention_inventory/slots/'+str(index),slot['owner']=='accepted_attempt_M')
        if loc in locations:ctx.add('root_bindings','/retention_inventory/slots/'+str(index),'inventory_mismatch')
        locations[loc]=slot
        if loc is None:continue
        if slot['owner']=='accepted_attempt_M':
            path=loc[1]
            if not(path==base+'/plan.json'or path==base+'/terminal.json'or path.startswith(base+'/reports/')or path.startswith(base+'/test.sealed/')):
                ctx.add('root_bindings','/retention_inventory/slots/'+str(index),'disclosure_invalid')
            if slot['role']in ('intent','result','observation','process','stdout','stderr','details','failure','input_projection')and not path.startswith(base+'/test.sealed/'):
                ctx.add('root_bindings','/retention_inventory/slots/'+str(index),'disclosure_invalid')

    inventories=_records(ctx,'publication_retention_inventory')
    by_file={_canonical(ctx.independent[r]['file']):(r,v)for r,v in inventories}
    by_address={}
    for role,inv in inventories:
        key=_address_key(inv['inventory_address'])
        by_address.setdefault(key,[]).append(role)
        if key!=_address_key(ctx.independent[role]['file']):ctx.add(role,'/inventory_address','accepted_binding_mismatch')
        if inv['slots']!=ctx.template['slots']:ctx.add(role,'/slots','inventory_mismatch')
    for roles in by_address.values():
        if len(roles)!=1:
            for role in roles:ctx.add(role,'/inventory_address','inventory_mismatch')

    terminal_rows=_records(ctx,'publication_terminal_outcome')
    selected=None
    if terminal and len(terminal_rows)==1:
        tr,t=terminal_rows[0]
        selected=by_file.get(_canonical(t['retention_inventory_file']))
        if selected is None:
            ctx.by_ref(t['retention_inventory_file'],tr,'/retention_inventory_file','retention_incomplete')
            ctx.add(tr,'/retention_inventory_file','retention_incomplete')
        elif _location(ctx,selected[1]['inventory_address'],tr,'/retention_inventory_file')!=('repository',base+'/test.sealed/inventories/pre_terminal.json'):
            ctx.add(tr,'/retention_inventory_file','accepted_binding_mismatch')
            selected=None

    def path_of(address,role):
        loc=_location(ctx,address,role,'/inventory_address')
        return loc[1]if loc and loc[0]=='repository'else None

    for role,inv in inventories:
        own=_address_key(inv['inventory_address'])
        current=ctx.independent[role]['file']
        path=path_of(inv['inventory_address'],role)
        cap=inv['capture']
        predecessor=None
        if current==ctx.e['retention_inventory_file']:
            if inv['prior_inventory_file']is not None:ctx.add(role,'/prior_inventory_file','chronology_invalid')
        else:
            predecessor=by_file.get(_canonical(inv['prior_inventory_file']))if inv['prior_inventory_file']is not None else None
            if predecessor is None:
                ctx.add(role,'/prior_inventory_file','retention_incomplete'if inv['prior_inventory_file']is not None else 'chronology_invalid')
            else:
                pr,prior=predecessor;pc=prior['capture']
                if pc['last_sequence']is None or cap['last_sequence']is None:
                    ctx.add(role,'/capture','retention_incomplete')
                elif pc['clock_epoch_id']!=cap['clock_epoch_id']or pc['last_sequence']>=cap['first_sequence']:
                    ctx.add(role,'/prior_inventory_file','chronology_invalid')
                expected_prior=ctx.e['retention_inventory_file']if path==base+'/test.sealed/inventories/pre_terminal.json'else (
                    terminal_rows[0][1]['retention_inventory_file']if terminal_rows and path==base+'/test.sealed/inventories/assessment.json'else None)
                if expected_prior is None or inv['prior_inventory_file']!=expected_prior:
                    ctx.add(role,'/prior_inventory_file','accepted_binding_mismatch')
        observed={_address_key(x['address']):x for x in inv['observed_files']}
        for key,row in observed.items():
            if key not in slots:ctx.add(role,'/observed_files','inventory_mismatch')
            if row['state']=='retained_complete':
                raw=ctx.by_ref(row['file'],role,'/observed_files','retention_incomplete')
                if raw is not None:
                    ctx.checked.add('/independent/'+role+'/observed_files/'+str(sorted(observed).index(key)))
                    for rr,record in ctx.parsed.items():
                        if ctx.independent[rr]['file']!=row['file']:continue
                        rc=record.get('capture',record.get('closed_capture'))
                        if rc is not None:
                            if rc['last_sequence']is None or cap['last_sequence']is None:
                                ctx.add(role,'/observed_files','retention_incomplete')
                            elif rc['clock_epoch_id']!=cap['clock_epoch_id']or rc['last_sequence']>cap['last_sequence']:
                                ctx.add(role,'/observed_files','chronology_invalid')
            elif row['state']in ('retained_partial','unavailable'):
                # A later successful read never rewrites an earlier uncertain cutoff.
                ctx.add(role,'/observed_files','retention_incomplete')
        if predecessor is not None:
            for old in predecessor[1]['observed_files']:
                key=_address_key(old['address']);new=observed.get(key)
                if new is None:ctx.add(role,'/observed_files','retention_incomplete')
                elif old['state']=='retained_complete'and new!=old:
                    ctx.add(role,'/observed_files','inventory_mismatch')
            prior_key=_address_key(inv['prior_inventory_file'])
            if observed.get(prior_key)!=dict(address={k:inv['prior_inventory_file'][k]for k in ('root_binding_id','relative_path')},state='retained_complete',file=inv['prior_inventory_file']):
                ctx.add(role,'/prior_inventory_file','retention_incomplete')
        if selected is None or role!=selected[0]:continue
        # Only this terminal-referenced immutable cutoff has terminal completeness.
        # Entry is an independently accepted template, not a future inventory.
        for key,slot in slots.items():
            if key==own:continue
            row=observed.get(key)
            if row is None:
                ctx.add(role,'/observed_files','retention_incomplete');continue
            if row['state']!='not_reached':continue
            slot_path=path_of(slot['address'],role)
            failure_slot=(slot['condition']=='failure_observed'or
                any('/'+s+'.'in (slot_path or '')for s in ('quarantine',))or
                any((slot_path or '').endswith('/'+p+'.json')for p in ('failure_location','post_quarantine')))
            future_assessment=(slot['role']in ('details','report')and (slot_path or '').endswith('/terminal.json')or
                slot_path==base+'/test.sealed/inventories/assessment.json')
            pending_terminal=slot['role']=='terminal'and slot['condition']=='terminal_pending'
            if not(failure_slot or future_assessment or pending_terminal):
                ctx.add(role,'/observed_files','retention_incomplete')
            # Not reached is a historical claim, not permission to ignore actual
            # contradictory namespace bytes or a failed later dispatch.
            for rr,envelope in ctx.independent.items():
                if envelope['data']is None or _address_key(envelope['file'])!=key:continue
                if pending_terminal or future_assessment:continue
                ctx.add(role,'/observed_files','inventory_mismatch')
        tc=terminal_rows[0][1]['closed_capture']
        if cap['last_sequence']is None:ctx.add(role,'/capture','retention_incomplete')
        elif cap['clock_epoch_id']!=tc['clock_epoch_id']or cap['last_sequence']>=tc['first_sequence']:
            ctx.add(role,'/capture','chronology_invalid')
        # Every supplied owned namespace file must be accounted at this cutoff,
        # apart from its own output and explicitly later terminal/assessment.
        for rr,envelope in ctx.independent.items():
            key=_address_key(envelope['file']);slot=slots.get(key)
            if slot is None or slot['owner']!='accepted_attempt_M'or key==own:continue
            path=path_of(slot['address'],role)
            if slot['role']=='terminal'or (slot['role']in ('report','details')and (path or '').endswith('/terminal.json'))or path==base+'/test.sealed/inventories/assessment.json':continue
            row=observed.get(key)
            if envelope['data']is not None and (row is None or row['state']!='retained_complete'or row['file']!=envelope['file']):
                ctx.add(role,'/observed_files','inventory_mismatch')


def _terminal_assessment(ctx,role,terminal):
    """Recompute the earlier pair, never the current terminal input projection.

    input_files records the original ordered envelopes, not authority for them.
    Required independent roles, actual reader bytes and independently captured
    phase intervals constrain that projection before either product is accepted.
    """
    raw=ctx.by_ref(terminal['physical_comparison_file'],role,'/physical_comparison_file','retention_incomplete')
    if raw is None:return
    try:
        report=_decode(raw)
        _require(type(report)is dict and report.get('schema')==_PREFIX+'publication_physical_report.v1'and report.get('phase')=='post_complete','post-complete assessment')
        _self_hash(report)
        path=ctx.plan['publication_evidence_root_relative_path']+'/reports/post_complete.json'
        _require(_resolve_publication_address({k:terminal['physical_comparison_file'][k]for k in ('root_binding_id','relative_path')},ctx.roots)==('repository',path),'assessment output location')
        _ref(report['sealed_details_file'])
        details_raw=ctx.by_ref(report['sealed_details_file'],role,'/physical_comparison_file/sealed_details_file','retention_incomplete')
        if details_raw is None:return
        details=_decode(details_raw)
        post_role,post=_one_record(ctx,'publication_physical_observation','phase','post_complete')
        if post is None or not ctx.valid.get(post_role):return
        cutoff=post['capture']
        _require(cutoff['last_sequence']is not None,'closed assessment cutoff')
        _require(type(report['input_files'])is list,'original input projection')
        projections={'independent':[],'candidate':[]};seen=set()
        future={_canonical(terminal['physical_comparison_file']),_canonical(report['sealed_details_file'])}
        for row in report['input_files']:
            _obj(row,'role file');_text(row['role']);_ref(row['file'],positive=False)
            label,separator,name=row['role'].partition(':')
            _require(separator and label in projections and name and row['role']not in seen,'unambiguous original input role')
            seen.add(row['role'])
            peer=ctx.independent.get(name)
            if peer is None or peer['data']is None:
                ctx.add(name,'/post_complete_original_inputs','required_bytes_unavailable');return
            _require(peer['file']==row['file']and _canonical(row['file'])not in future,'original actual byte identity')
            record=ctx.parsed.get(name)
            if record is not None:
                _require(ctx.valid.get(name),'valid original evidence')
                rc=record.get('capture',record.get('closed_capture'))
                if rc is not None:
                    _require(rc['last_sequence']is not None and rc['clock_epoch_id']==cutoff['clock_epoch_id']and rc['last_sequence']<=cutoff['last_sequence'],'no future evidence in assessment')
                _require(record.get('schema')!=_PREFIX+'publication_terminal_outcome.v1','no terminal recursion')
            projections[label].append(_freeze(peer))
        required=set(next(x for x in ctx.e['phase_requirements']if x['phase']=='post_complete')['required_evidence_roles'])
        required|={x['role']for x in ctx.e['accepted_input_files']}
        _require(required<={r['role']for r in projections['independent']},'original independent required coverage')
        # A claimed original snapshot cannot omit independently visible records
        # from its own closed interval, even if a candidate rewrites input_files.
        for name,record in ctx.parsed.items():
            rc=record.get('capture')
            if rc is not None and ctx.valid.get(name) and rc['clock_epoch_id']==cutoff['clock_epoch_id']and rc['last_sequence']is not None and rc['last_sequence']<=cutoff['last_sequence']:
                _require('independent:'+name in seen,'omitted evidence within original cutoff')
        actual=verify_publication_evidence(phase='post_complete',expected=ctx.expected,
            candidate_files=tuple(projections['candidate']),independent_files=tuple(projections['independent']))
        _require(actual['report']['comparison_status']=='matched','post-complete evidence does not match')
        _require(_canonical(report)==_canonical(actual['report'])and details_raw==_canonical(actual['sealed_details']),'paired assessment differs from recomputation')
        ctx.checked.add('/independent/'+role+'/physical_comparison_file/recomputed_pair')
    except (PublicationComparisonError,KeyError,TypeError,ValueError):
        ctx.add(role,'/physical_comparison_file','accepted_binding_mismatch')


def _terminal_checks(ctx):
    terminals=_records(ctx,'publication_terminal_outcome',False)
    if len(terminals)!=1:
        ctx.add('root_bindings','/terminal','schema_invalid'if terminals else 'required_bytes_unavailable')
        return
    role,terminal=terminals[0]
    if not ctx.valid.get(role):return
    if terminal['disposition']!='publication_observed':
        ctx.add(role,'/disposition','unexpected_effect');return
    if terminal['located_root']!='F'or ctx.located!='F':ctx.add(role,'/located_root','identity_mismatch')
    if terminal['failure_evidence_files']:ctx.add(role,'/failure_evidence_files','unexpected_effect')
    expected_operations=[]
    for step in _FORWARD:
        ir,intent=_one_record(ctx,'publication_operation_intent','step',step)
        rr,result=_one_record(ctx,'publication_operation_result','step',step)
        if intent is not None and result is not None and ctx.valid.get(ir)and ctx.valid.get(rr):
            expected_operations.append(dict(step=step,intent_file=ctx.independent[ir]['file'],result_file=ctx.independent[rr]['file']))
    if terminal['operation_files']!=expected_operations:ctx.add(role,'/operation_files','accepted_binding_mismatch')
    for kind in ('publication_operation_intent','publication_operation_result'):
        rows=_records(ctx,kind,False)
        if len(rows)!=len(_FORWARD)or {x.get('step')for _,x in rows}!=set(_FORWARD):
            ctx.add(role,'/operation_files','inventory_mismatch')
    processes=[]
    for name in ('state6_verification_child','publication_finalizer'):
        pr,process=_one_record(ctx,'publication_process_outcome','role',name)
        if process is not None and ctx.valid.get(pr):processes.append(ctx.independent[pr]['file'])
    if terminal['process_outcome_files']!=processes:ctx.add(role,'/process_outcome_files','accepted_binding_mismatch')
    if len(_records(ctx,'publication_process_outcome',False))!=2:ctx.add(role,'/process_outcome_files','inventory_mismatch')
    for rr,observation in _records(ctx,'publication_physical_observation',False):
        if observation.get('phase')in ('failure_location','post_quarantine'):ctx.add(rr,'/phase','unexpected_effect')
    _terminal_assessment(ctx,role,terminal)
    # The terminal binds the actual published paths/bytes, not a prepared-file
    # alias, arbitrary available evidence, or a candidate's self-selected hash.
    post=[x for _,x in _records(ctx,'publication_physical_observation')if x['phase']=='post_complete']
    for key,name in (('final_audit_file','final_audit.json'),('completion_file','COMPLETE.json')):
        ref=terminal[key];raw=ctx.by_ref(ref,role,'/'+key,'retention_incomplete')
        if ref is None:continue
        loc=_location(ctx,{k:ref[k]for k in ('root_binding_id','relative_path')},role,'/'+key)
        wanted=('repository',ctx.plan['final_root_relative_path']+'/'+name)
        if loc!=wanted:ctx.add(role,'/'+key,'accepted_binding_mismatch')
        objects=[x for o in post for x in o['objects']if _location(ctx,x['address'],role,'/'+key)==wanted and x['presence']=='present']
        if len(objects)!=1 or objects[0]['content_file']is None:ctx.add(role,'/'+key,'required_bytes_unavailable')
        elif raw is not None and (len(raw)!=objects[0]['content_file']['file_bytes']or _digest(raw)!=objects[0]['content_file']['file_sha256']):
            ctx.add(role,'/'+key,'content_mismatch')
    finalizers=[x for _,x in _records(ctx,'publication_process_outcome')if x['role']=='publication_finalizer']
    releases=[x for _,x in _records(ctx,'publication_operation_result')if x['step']=='release_resources']
    if finalizers:
        fc=finalizers[0]['capture'];tc=terminal['closed_capture']
        if (fc['last_sequence']is None or fc['clock_epoch_id']!=tc['clock_epoch_id']or
                tc['first_sequence']<=fc['last_sequence']or tc['started_utc']<fc['finished_utc']):
            ctx.add(role,'/closed_capture','chronology_invalid')
        if releases:
            rc=releases[0]['capture']
            # The supervised worker's outcome spans creation through exit, so
            # release is inside it. An independently captured later process
            # outcome remains valid for older retained histories.
            later_outcome=(rc['last_sequence']<fc['first_sequence']and
                           rc['finished_utc']<=fc['started_utc']) if rc['last_sequence']is not None else False
            spanning_worker=(fc['last_sequence']is not None and rc['last_sequence']is not None and
                             fc['first_sequence']<rc['first_sequence'] and
                             rc['last_sequence']<fc['last_sequence'] and
                             fc['started_utc']<=rc['started_utc'] and
                             rc['finished_utc']<=fc['finished_utc'])
            if (rc['last_sequence']is None or fc['last_sequence']is None or
                    rc['clock_epoch_id']!=fc['clock_epoch_id']or
                    not(later_outcome or spanning_worker)):
                ctx.add(role,'/process_outcome_files','chronology_invalid')
    wanted=next(x for x in ctx.e['phase_requirements']if x['phase']=='terminal')['required_evidence_roles']
    for name in wanted:
        if name not in ctx.independent or ctx.independent[name]['data']is None:ctx.add(name,'','retention_incomplete')


def verify_publication_evidence(*,phase,expected,candidate_files,independent_files):
    ctx=_bootstrap_context(expected,independent_files,phase)
    ctx.candidate=_collect(ctx,candidate_files,'candidate')
    _record_population(ctx)
    phase_req=next(x for x in ctx.e['phase_requirements']if x['phase']==phase)
    for role in phase_req['required_evidence_roles']:
        if role not in ctx.independent or ctx.independent[role]['data']is None:
            ctx.add(role,'','required_bytes_unavailable')
    if ctx.roots is not None and ctx.plan is not None:
        if phase in ('failure_location','post_quarantine'):
            phases=[phase];steps=[]
        else:
            limit=5 if phase=='terminal'else ('pre_finalize','pre_move','post_move','post_audit','post_complete').index(phase)+1
            phases=list(('pre_finalize','pre_move','post_move','post_audit','post_complete')[:limit])
            counts={'pre_finalize':0,'pre_move':1,'post_move':2,'post_audit':3,'post_complete':5,'terminal':6}
            steps=list(_FORWARD[:counts[phase]])
        for p in phases:_observation_checks(ctx,p)
        if phase not in ('failure_location','post_quarantine'):
            _chronology(ctx,phases,steps);_process_checks(ctx,phase=='terminal')
        else:
            # Failure classification never turns an uncertain native call into success.
            for role,result in _records(ctx,'publication_operation_result'):
                if result['call_status']=='outcome_unavailable'or result['effect_status']=='unknown':
                    ctx.add(role,'/call_status','operation_outcome_unavailable')
            if phase=='post_quarantine':
                ir,intent=_one_record(ctx,'publication_operation_intent','step','quarantine')
                rr,result=_one_record(ctx,'publication_operation_result','step','quarantine')
                if result and ctx.valid.get(rr) and (result['call_status']!='returned_success'or result['failures']):ctx.add(rr,'/call_status','unexpected_effect')
        _retention_checks(ctx,phase=='terminal')
        if phase=='terminal':_terminal_checks(ctx)
    rows=ctx.ordered()
    ctx.effect='expected_effect_observed'if ctx.status()=='matched'and phase in ('post_move','post_audit','post_complete','post_quarantine','terminal')else 'unexpected_effect_observed'if any(x['code']=='unexpected_effect'for x in rows['mismatches'])else 'none_observed'if ctx.status()=='matched'and phase in ('pre_finalize','pre_move')else 'unknown'
    expected_hash=_digest(_canonical(ctx.e))
    details=_seal(dict(schema=_PREFIX+'publication_physical_details.v1',expected_inputs_sha256=expected_hash,phase=phase,
        checked_fields=sorted(ctx.checked),**rows))
    detail_bytes=_canonical(details);address=_template(ctx)
    if ctx.roots is not None and ctx.plan is not None:
        loc=_location(ctx,address,'root_bindings','/sealed_details_file',True)
        wanted=ctx.plan['publication_evidence_root_relative_path']+'/test.sealed/reports/'+phase+'.json'
        if loc!=('repository',wanted):
            ctx.add('root_bindings','/sealed_details_file','disclosure_invalid')
            rows=ctx.ordered()
            details=_seal(dict(schema=_PREFIX+'publication_physical_details.v1',expected_inputs_sha256=expected_hash,phase=phase,checked_fields=sorted(ctx.checked),**rows))
            detail_bytes=_canonical(details)
    outer_rows={}
    for category,values in rows.items():
        outer_rows[category]=[dict(input=r,pointer='/'+r,code=c)for r,c in sorted({(x['input'],x['code'])for x in values})]
    outer_checks=sorted({'/'+x.split('/')[2]for x in ctx.checked if len(x.split('/'))>=3})
    report=_seal(dict(schema=_PREFIX+'publication_physical_report.v1',profile=_PROFILE,phase=phase,
        expected_inputs_sha256=expected_hash,input_files=ctx.input_files,comparison_status=ctx.status(),
        checked_fields=outer_checks,**outer_rows,observed_effect=ctx.effect,located_root=ctx.located,
        sealed_details_file=dict(address,file_bytes=len(detail_bytes),file_sha256=_digest(detail_bytes)),
        production_preflight_accepted=False,execution_authorized=False))
    return _freeze(dict(report=report,sealed_details=details))


def _recomputed(comparison,*,phase,expected,candidate_files,independent_files):
    _require(type(comparison)in (dict,MappingProxyType),'comparison pair required')
    value=_plain(comparison);_obj(value,'report sealed_details')
    for key in value:
        if type(value[key])is bytes:value[key]=_decode(value[key])
        _require(type(value[key])is dict,'record object or canonical immutable bytes')
    actual=verify_publication_evidence(phase=phase,expected=expected,candidate_files=candidate_files,independent_files=independent_files)
    _require(_canonical(value)==_canonical(actual),'comparison differs from complete independent recomputation')
    return actual


def validate_publication_comparison(comparison,*,phase,expected,candidate_files,independent_files):
    _recomputed(comparison,phase=phase,expected=expected,candidate_files=candidate_files,independent_files=independent_files)


def canonical_publication_comparison_bytes(comparison,*,phase,expected,candidate_files,independent_files):
    value=_recomputed(comparison,phase=phase,expected=expected,candidate_files=candidate_files,independent_files=independent_files)
    return MappingProxyType(dict(report_bytes=_canonical(value['report']),sealed_details_bytes=_canonical(value['sealed_details'])))
