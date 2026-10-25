"""Private, injected launch contracts. No credentials, live store or state route.

Independent expectations and qualified acquisition/store boundaries are premises,
not conclusions established by hashes or these immutable values. Recovery never
returns launch ownership. Records here are not an adopted production protocol.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import re
import threading
import unicodedata

_CONTROL_OR_SURROGATE = re.compile(r"[\x00-\x1f\x7f-\x9f\ud800-\udfff]")

from operatormark.corpus import final_evidence_records as records
from operatormark.corpus import publication_physical_records as publication
from operatormark.corpus.source_binding import (
    ExecutionSourceExpectations, SourceBytes, validate_execution_source_manifest, verify_execution_source_bytes,
)


class LaunchAuthorityError(ValueError):
    pass


def need(condition, reason):
    if not condition:
        raise LaunchAuthorityError(reason)


def _expected_publication_process_role(launch_role):
    """Re-derive the publication role from the independently bound launch role."""
    roles = {
        'state6_verifier': 'state6_verification_child',
        'state6_finalizer': 'publication_finalizer',
        'state6_verification_child': 'state6_verification_child',
        'publication_finalizer': 'publication_finalizer',
    }
    need(type(launch_role) is str and launch_role in roles, 'unsupported State 6 launch role')
    return roles[launch_role]


def plain(value):
    t = type(value)
    if t is dict:
        need(all(type(k) is str for k in value), 'exact string keys required')
        for k, v in value.items():
            plain(k); plain(v)
    elif t is list:
        for item in value:
            plain(item)
    elif t is str:
        need(value == unicodedata.normalize('NFC', value) and
             _CONTROL_OR_SURROGATE.search(value) is None, 'noncanonical text')
    else:
        need(t in (int, bool, type(None)), 'non-data JSON value')


def canonical(value):
    plain(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def decode(data):
    need(type(data) is bytes, 'immutable canonical bytes required')
    try:
        result = json.loads(data.decode('utf-8'), object_pairs_hook=_pairs)
        need(canonical(result) == data, 'noncanonical JSON encoding')
        return result
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise LaunchAuthorityError('invalid canonical JSON') from exc


def sha(data):
    need(type(data) is bytes, 'immutable bytes required')
    return hashlib.sha256(data).hexdigest()


def object_keys(value, keys):
    need(type(value) is dict and set(value) == set(keys.split()), 'exact object keys required')


def text(value):
    need(type(value) is str and bool(value), 'nonempty text required')
    plain(value)


def integer(value, minimum=0):
    need(type(value) is int and value >= minimum, 'exact integer required')


def digest(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None
         and value != '0'*64, 'SHA-256 required')


def enum(value, choices):
    need(type(value) is str and value in choices, 'closed token required')


def ref(value, nullable=False):
    try:
        records._ref(value, nullable)
    except records.FinalEvidenceRecordError as exc:
        raise LaunchAuthorityError('invalid FileRef') from exc


def refs(value, nonempty=False):
    need(type(value) is list and (bool(value) or not nonempty), 'reference array required')
    for item in value:
        ref(item)
    need(len({canonical(x) for x in value}) == len(value), 'duplicate references')


def address(value):
    records._address(value)
    need(bool(value['relative_path']), 'nonempty address path required')


def utc(value):
    records._utc(value)
    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')


def file_ref(address_value, data):
    address(address_value)
    return dict(address_value, file_bytes=len(data), file_sha256=sha(data))


def check_bytes(reference, data):
    ref(reference)
    need(type(data) is bytes and len(data) == reference['file_bytes']
         and sha(data) == reference['file_sha256'], 'independent file identity mismatch')


def source_commitment(expected):
    validate_execution_source_manifest(expected.members, expected=expected)
    # asdict sees only the exact already-validated immutable Batch A contracts.
    projection = json.loads(json.dumps(asdict(expected)))
    return sha(b'operatormark-launch-source-v1\n' + canonical(projection))


@dataclass(frozen=True, slots=True)
class Acquisition:
    file: bytes
    data: bytes | None
    validator_file: bytes
    scope: str
    first_utc: str
    last_utc: str
    status: str


@dataclass(frozen=True, slots=True)
class LaunchExpectations:
    specification: bytes
    specification_file: bytes
    source_expectations: ExecutionSourceExpectations
    slot_key: str
    validator_files: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class PreparedLaunch:
    expected: LaunchExpectations
    independent_inputs: tuple[Acquisition, ...]
    authority_id: str
    slot_id: str
    binding_sha256: str


SPEC_KEYS = ('authorization_file acceptance_file grant_key binding operation role child_slots '
             'predecessors source_binding_sha256 command_binding_sha256 interpreter_file '
             'dependency_file environment_sha256 roots output_slots valid_from valid_until '
             'cleanup_until owner_binding_file observer_binding_file observer_id clock_epoch_id recovery_validator_file '
             'source_closure_file supervisor_source_file authority_scope_file publication_binding')


def validate_specification(spec, *, source_expected, slot_key):
    object_keys(spec, SPEC_KEYS)
    for key in ('authorization_file', 'acceptance_file', 'interpreter_file', 'dependency_file',
                'owner_binding_file', 'observer_binding_file', 'recovery_validator_file',
                'source_closure_file','supervisor_source_file','authority_scope_file'):
        ref(spec[key])
    records.execution_prefix(spec['binding'])
    if spec['binding']['state'] == 'state6':
        need(type(spec['publication_binding']) is dict, 'State 6 publication binding required')
        v2 = spec['binding']['benchmark']=='darcy_branch_c_v2'
        publication.validate_publication_binding(spec['publication_binding'],allow_v2=v2)
        need(spec['publication_binding']['profile']==('whole_attempt_same_volume_v2' if v2 else 'whole_attempt_same_volume_v1'),
             'mixed execution/publication versions')
        need(spec['publication_binding']['state6_authorization_file_sha256'] == spec['binding']['authorization_file_sha256']
             and spec['publication_binding']['state4_authorization_file_sha256'] == spec['binding']['state4_authorization_file_sha256']
             and spec['publication_binding']['protocol_file_sha256'] == spec['binding']['protocol_file_sha256']
             and spec['publication_binding']['implementation_commit'] == spec['binding']['implementation']['commit']
             and spec['publication_binding']['implementation_tree'] == spec['binding']['implementation']['tree']
             and spec['publication_binding']['expected_parent_commit'] == spec['binding']['implementation']['sole_parent'], 'publication/stage binding mismatch')
    else:
        need(spec['publication_binding'] is None, 'publication binding only for State 6')
    need(spec['authorization_file']['file_sha256'] == spec['binding']['authorization_file_sha256'],
         'authorization reference/binding mismatch')
    for key in ('grant_key', 'operation', 'role', 'observer_id', 'clock_epoch_id'):
        text(spec[key])
    for key in ('source_binding_sha256', 'command_binding_sha256', 'environment_sha256'):
        digest(spec[key])
    need(spec['source_binding_sha256'] == source_commitment(source_expected), 'source expectation mismatch')
    need((source_expected.operation, source_expected.role) == (spec['operation'], spec['role']),
         'source operation/role mismatch')
    need(source_expected.protocol_sha256 == spec['binding']['protocol_file_sha256']
         and source_expected.implementation_commit == spec['binding']['implementation']['commit']
         and source_expected.implementation_tree == spec['binding']['implementation']['tree'], 'source/stage binding mismatch')
    slots = spec['child_slots']
    need(type(slots) is list and bool(slots), 'explicit finite child graph required')
    seen = set()
    for slot in slots:
        object_keys(slot, 'key parent_key operation role')
        for key in ('key', 'operation', 'role'):
            text(slot[key])
        need(slot['key'] not in seen, 'duplicate child slot')
        need(slot['parent_key'] is None or type(slot['parent_key']) is str and slot['parent_key'] in seen,
             'unknown/forward/cyclic parent slot')
        seen.add(slot['key'])
    selected = [s for s in slots if s['key'] == slot_key]
    need(len(selected) == 1 and (selected[0]['operation'], selected[0]['role']) ==
         (spec['operation'], spec['role']), 'selected operation/role not independently declared')
    refs(spec['predecessors'], True)
    need(type(spec['roots']) is list, 'explicit root inventory required')
    roles = set(); root_ids = set()
    for root in spec['roots']:
        object_keys(root, 'role root_binding_id acceptance_file')
        text(root['role']); text(root['root_binding_id']); ref(root['acceptance_file'])
        need(root['role'] not in roles and root['root_binding_id'] not in root_ids, 'duplicate root binding')
        roles.add(root['role']); root_ids.add(root['root_binding_id'])
    need({'source', 'output', 'journal'} <= roles, 'required root binding omitted')
    need(type(spec['output_slots']) is list and bool(spec['output_slots']), 'explicit output inventory required')
    keys = set(); addresses = set()
    for slot in spec['output_slots']:
        object_keys(slot, 'key address owner writer')
        text(slot['key']); address(slot['address'])
        enum(slot['owner'], ('accepted_attempt', 'separate_B', 'external_preexisting', 'unresolved'))
        enum(slot['writer'], ('canonical_exclusive_preserve_v1', 'raw_supervisor_capture', 'component_existing_contract'))
        location = (slot['address']['root_binding_id'], slot['address']['relative_path'].casefold())
        need(slot['key'] not in keys and location not in addresses, 'duplicate/aliased output product')
        need(slot['address']['root_binding_id'] in root_ids, 'unaccepted output root')
        keys.add(slot['key']); addresses.add(location)
    need(utc(spec['valid_from']) <= utc(spec['valid_until']) <= utc(spec['cleanup_until']), 'authority interval order')
    return selected[0]


def qualify(reference, acquisitions, *, validators, scope, first, last):
    """Caller supplies independent accepted acquisition decisions, not producer labels."""
    ref(reference)
    need(type(acquisitions) is tuple and all(type(x) is Acquisition for x in acquisitions), 'exact acquisitions required')
    matches = []
    for acquisition in acquisitions:
        f = decode(acquisition.file); ref(f)
        vf = decode(acquisition.validator_file); ref(vf)
        enum(acquisition.status, ('accepted', 'rejected', 'unavailable'))
        utc(acquisition.first_utc); utc(acquisition.last_utc)
        need(acquisition.first_utc <= acquisition.last_utc, 'acquisition interval order')
        text(acquisition.scope)
        if f == reference and acquisition.scope == scope:
            matches.append(acquisition)
    need(len(matches) == 1, 'missing/duplicate independent acquisition')
    item = matches[0]
    need(item.status == 'accepted' and item.validator_file in validators and item.scope == scope
         and item.first_utc <= first <= last <= item.last_utc, 'unqualified evidence/source/scope/interval')
    check_bytes(reference, item.data)
    return item.data


def prepare_launch(*, expected, independent_inputs):
    need(type(expected) is LaunchExpectations, 'independent launch expectations required')
    need(type(expected.validator_files) is tuple and bool(expected.validator_files)
         and len(set(expected.validator_files)) == len(expected.validator_files), 'independent validator bindings required')
    for item in expected.validator_files:
        ref(decode(item))
    spec = decode(expected.specification)
    slot = validate_specification(spec, source_expected=expected.source_expectations, slot_key=expected.slot_key)
    specification_file = decode(expected.specification_file)
    check_bytes(specification_file, expected.specification)
    required = [(specification_file, 'launch_authority')]
    required += [(spec[k], 'launch_authority') for k in ('authorization_file', 'acceptance_file',
                 'interpreter_file', 'dependency_file', 'owner_binding_file', 'observer_binding_file', 'recovery_validator_file',
                 'source_closure_file','supervisor_source_file','authority_scope_file')]
    required += [(r, 'launch_authority') for r in spec['predecessors']]
    required += [(r['acceptance_file'], 'root_binding') for r in spec['roots']]
    if spec['publication_binding'] is not None:
        required.append((spec['publication_binding']['publication_plan_file'],'launch_authority'))
    for reference, scope in required:
        qualify(reference, independent_inputs, validators=expected.validator_files, scope=scope,
                first=spec['valid_from'], last=spec['valid_until'])
    authority_id = sha(b'operatormark-launch-authority-v1\n' + canonical(dict(
        authorization_file_sha256=spec['authorization_file']['file_sha256'], grant_key=spec['grant_key'])))
    slot_id = sha(b'operatormark-launch-slot-v1\n' + canonical(dict(authority_id=authority_id, **slot)))
    return PreparedLaunch(expected, independent_inputs, authority_id, slot_id,
                          sha(b'operatormark-launch-binding-v1\n' + expected.specification))


def reprepare(prepared, *, expected, independent_inputs):
    actual = prepare_launch(expected=expected, independent_inputs=independent_inputs)
    need(type(prepared) is PreparedLaunch and prepared == actual, 'prepared value substitution')
    return actual


def journal_address(prepared, kind, ticket=None):
    spec = decode(prepared.expected.specification)
    root = next(r['root_binding_id'] for r in spec['roots'] if r['role'] == 'journal')
    need(kind in ('reservation', 'dispatch', 'creation', 'exit', 'closure', 'recovery'), 'journal kind')
    filename = kind
    if kind == 'recovery':
        ref(ticket); filename += '-' + ticket['file_sha256']
    return dict(root_binding_id=root, relative_path=f'{prepared.authority_id}/{prepared.slot_id}/{filename}.json')


PAYLOADS = {
    'reservation': 'binding_sha256 owner_binding_file claim_address',
    'dispatch': 'reservation_file command_binding_sha256 source_binding_sha256 parent_slot_id',
    'creation': 'dispatch_file call_status creation_status process_identity observation_files failure_files',
    'exit': 'creation_file process_identity direct_exit_code reap_status observation_files failure_files',
    'closure': 'reservation_file dispatch_file creation_file exit_file process_outcome_file guard_closure_file surviving_files failure_files resolution',
    'recovery': 'recovery_ticket_file inspected_files unavailable_addresses ownership_fence_file process_observation_files conclusions conflicts',
}


def process_identity(value, nullable=False):
    if value is None and nullable:
        return
    object_keys(value, 'pid creation_observation_file')
    integer(value['pid'], 1); ref(value['creation_observation_file'])


def validate_launch_record(record, *, prepared, actual_file=None):
    object_keys(record, 'schema authority_id slot_id binding_file prior_record_file observation payload record_self_sha256')
    kinds = {f'state_launch_{k}.v1': k for k in PAYLOADS}
    enum(record['schema'], kinds)
    kind = kinds[record['schema']]
    need(record['authority_id'] == prepared.authority_id and record['slot_id'] == prepared.slot_id
         and record['binding_file'] == decode(prepared.expected.specification_file), 'journal binding mismatch')
    ref(record['prior_record_file'], True)
    o = record['observation']; object_keys(o, 'observer_id clock_epoch_id sequence observed_utc')
    text(o['observer_id']); text(o['clock_epoch_id']); integer(o['sequence']); utc(o['observed_utc'])
    p = record['payload']; object_keys(p, PAYLOADS[kind])
    spec = decode(prepared.expected.specification)
    need(o['observer_id'] == spec['observer_id'] and o['clock_epoch_id'] == spec['clock_epoch_id'], 'journal observer/clock binding')
    if kind == 'reservation':
        need(record['prior_record_file'] is None and p['binding_sha256'] == prepared.binding_sha256
             and p['owner_binding_file'] == spec['owner_binding_file']
             and p['claim_address'] == journal_address(prepared, kind), 'reservation binding')
    elif kind == 'dispatch':
        ref(p['reservation_file'])
        slot = next(s for s in spec['child_slots'] if s['key'] == prepared.expected.slot_key)
        parent = next((s for s in spec['child_slots'] if s['key'] == slot['parent_key']), None)
        parent_id = None if parent is None else sha(b'operatormark-launch-slot-v1\n' + canonical(dict(authority_id=prepared.authority_id, **parent)))
        need(p['parent_slot_id'] == parent_id and p['command_binding_sha256'] == spec['command_binding_sha256']
             and p['source_binding_sha256'] == spec['source_binding_sha256']
             and record['prior_record_file'] == p['reservation_file'], 'dispatch binding')
        need({k:p['reservation_file'][k] for k in ('root_binding_id','relative_path')} == journal_address(prepared,'reservation'), 'dispatch reservation address')
    elif kind == 'creation':
        ref(p['dispatch_file']); enum(p['call_status'], records.CALLS)
        enum(p['creation_status'], ('not_started', 'started', 'unknown'))
        refs(p['observation_files'], True); refs(p['failure_files'])
        process_identity(p['process_identity'], p['creation_status'] != 'started')
        if p['creation_status'] != 'started':
            need(p['process_identity'] is None, 'unestablished PID')
        else:
            need(p['process_identity']['creation_observation_file'] in p['observation_files']
                 and p['call_status'] != 'not_called', 'creation facts contradiction')
        if p['creation_status'] == 'not_started':
            need(p['call_status'] in ('not_called', 'returned_failure'), 'not-started contradiction')
        if p['call_status'] in ('returned_failure', 'outcome_unavailable') or p['creation_status'] == 'unknown':
            need(bool(p['failure_files']), 'unavailable/failure requires evidence')
        need(record['prior_record_file'] == p['dispatch_file'], 'creation predecessor')
        need({k:p['dispatch_file'][k] for k in ('root_binding_id','relative_path')} == journal_address(prepared,'dispatch'), 'creation dispatch address')
    elif kind == 'exit':
        ref(p['creation_file']); process_identity(p['process_identity'])
        need(p['direct_exit_code'] is None or type(p['direct_exit_code']) is int, 'direct exit type')
        enum(p['reap_status'], ('observed_reaped', 'unresolved', 'failed'))
        refs(p['observation_files']); refs(p['failure_files'])
        need(p['reap_status'] != 'observed_reaped' or p['direct_exit_code'] is not None and bool(p['observation_files']), 'reaped requires direct exit evidence')
        need(p['reap_status'] == 'observed_reaped' or bool(p['failure_files']), 'unresolved exit requires failure evidence')
        need(record['prior_record_file'] == p['creation_file'], 'exit predecessor')
        need({k:p['creation_file'][k] for k in ('root_binding_id','relative_path')} == journal_address(prepared,'creation'), 'exit creation address')
    elif kind == 'closure':
        for key in ('reservation_file', 'dispatch_file', 'creation_file', 'exit_file', 'process_outcome_file', 'guard_closure_file'):
            ref(p[key], key != 'reservation_file')
            if key in ('reservation_file','dispatch_file','creation_file','exit_file') and p[key] is not None:
                need({k:p[key][k] for k in ('root_binding_id','relative_path')} == journal_address(prepared,key.removesuffix('_file')), 'closure journal address')
        refs(p['surviving_files']); refs(p['failure_files'])
        enum(p['resolution'], ('refused_before_dispatch', 'created_closed', 'failed_preserved', 'unresolved'))
        if p['resolution'] == 'created_closed':
            need(all(p[k] is not None for k in ('dispatch_file','creation_file','exit_file','process_outcome_file','guard_closure_file')) and not p['failure_files'], 'false complete closure')
        if p['resolution'] == 'refused_before_dispatch':
            need(all(p[k] is None for k in ('dispatch_file','creation_file','exit_file')), 'false pre-dispatch refusal')
        else:
            need(p['resolution'] == 'created_closed' or bool(p['failure_files']), 'failure evidence required')
        need(record['prior_record_file'] == (p['exit_file'] or p['creation_file'] or p['dispatch_file'] or p['reservation_file']), 'closure chronology')
    else:
        ref(p['recovery_ticket_file']); refs(p['inspected_files']); ref(p['ownership_fence_file'], True)
        refs(p['process_observation_files'])
        need(type(p['unavailable_addresses']) is list, 'unavailable address array')
        for a in p['unavailable_addresses']:
            address(a)
        c = p['conclusions']; object_keys(c, 'creation reaping retention authority')
        enum(c['creation'], ('not_started','started','unknown'))
        enum(c['reaping'], ('observed_reaped','unresolved','not_applicable'))
        enum(c['retention'], ('complete','incomplete'))
        need(c['authority'] == 'consumed_or_unavailable', 'recovery cannot renew authority')
        need(type(p['conflicts']) is list and all(type(x) is str for x in p['conflicts'])
             and len(set(p['conflicts'])) == len(p['conflicts']), 'conflict inventory')
        need(c['retention'] != 'complete' or not p['unavailable_addresses'], 'unavailable is incomplete')
        need(p['ownership_fence_file'] is not None or c['creation'] == 'unknown', 'unfenced recovery cannot prove absence')
    digest(record['record_self_sha256'])
    need(record['record_self_sha256'] == sha(canonical({k:v for k,v in record.items() if k != 'record_self_sha256'})), 'record self-hash')
    if actual_file is not None:
        expected_address = journal_address(prepared, kind, p.get('recovery_ticket_file'))
        need({k:actual_file[k] for k in expected_address} == expected_address, 'journal physical address mismatch')
        check_bytes(actual_file, canonical(record))
    return kind


def make_launch_record(kind, *, prepared, prior, observation, payload):
    record = dict(schema=f'state_launch_{kind}.v1', authority_id=prepared.authority_id,
                  slot_id=prepared.slot_id, binding_file=decode(prepared.expected.specification_file),
                  prior_record_file=prior, observation=observation, payload=payload)
    record['record_self_sha256'] = sha(canonical(record))
    validate_launch_record(record, prepared=prepared)
    return canonical(record)


@dataclass(frozen=True, slots=True)
class ClaimObservation:
    outcome: str
    retained_bytes: bytes | None
    actual_file: bytes | None
    ownership: object | None
    evidence_files: tuple[bytes, ...]


class ClaimLease:
    """Local single-use latch; store must independently authenticate/fence handle."""
    __slots__ = ('_lock', '_spent', '_store', '_handle', 'reservation_file', 'prepared')

    def __init__(self, store, handle, reservation_file, prepared):
        self._lock = threading.Lock(); self._spent = False
        self._store = store; self._handle = handle
        self.reservation_file = reservation_file; self.prepared = prepared

    def consume(self, *, store, dispatch_file):
        with self._lock:
            need(not self._spent and store is self._store, 'creation right consumed or store substituted')
            self._spent = True
            actual = decode(dispatch_file)
            data = store.read(actual)
            check_bytes(actual, data)
            record = decode(data)
            need(validate_launch_record(record,prepared=self.prepared,actual_file=actual) == 'dispatch'
                 and record['payload']['reservation_file'] == decode(self.reservation_file), 'unbound dispatch consumption')
            # Store acknowledgment is a separate, trusted-boundary fact. A lost
            # acknowledgment cannot restore the local right or synthesize one.
            receipt = store.consume_once(self._handle, self.reservation_file, dispatch_file)
            need(type(receipt) is bytes, 'ownership acknowledgment unavailable')
            row = decode(receipt)
            object_keys(row, 'reservation_file dispatch_file consumed ownership_evidence_file')
            ref(row['ownership_evidence_file'])
            need(row['consumed'] is True and row['reservation_file'] == decode(self.reservation_file)
                 and row['dispatch_file'] == decode(dispatch_file), 'ownership acknowledgment mismatch')
            return receipt


@dataclass(frozen=True, slots=True)
class ClaimResult:
    outcome: str
    observation: ClaimObservation
    lease: ClaimLease | None


def claim_launch_once(*, prepared, expected, independent_inputs, claim_store, observation):
    prepared = reprepare(prepared, expected=expected, independent_inputs=independent_inputs)
    spec = decode(expected.specification)
    need(spec['valid_from'] <= observation['observed_utc'] <= spec['valid_until'], 'launch validity interval')
    data = make_launch_record('reservation', prepared=prepared, prior=None, observation=observation,
                             payload=dict(binding_sha256=prepared.binding_sha256,
                                          owner_binding_file=spec['owner_binding_file'], claim_address=journal_address(prepared,'reservation')))
    result = claim_store.claim_exclusive(journal_address(prepared,'reservation'), data,
                                        next(r for r in spec['roots'] if r['role'] == 'journal'))
    need(type(result) is ClaimObservation, 'exact claim observation required')
    enum(result.outcome, ('won','existing','refused','unknown'))
    need(type(result.evidence_files) is tuple, 'ownership evidence required')
    for f in result.evidence_files:
        ref(decode(f))
    lease = None
    if result.outcome == 'won':
        need(result.ownership is not None and bool(result.evidence_files), 'missing exclusive ownership evidence')
        actual = decode(result.actual_file)
        need(result.retained_bytes == data, 'claim readback mismatch')
        validate_launch_record(decode(data), prepared=prepared, actual_file=actual)
        lease = ClaimLease(claim_store, result.ownership, result.actual_file, prepared)
    else:
        need(result.ownership is None, 'existing/uncertain claim cannot own')
        if result.actual_file is not None:
            check_bytes(decode(result.actual_file), result.retained_bytes)
    return ClaimResult(result.outcome, result, lease)


@dataclass(frozen=True, slots=True)
class LaunchInspection:
    status: str
    creation: str
    reaping: str
    retention: str
    diagnostics: tuple[str, ...]
    renewed_authority: bool = False


@dataclass(frozen=True, slots=True)
class JournalRetentionExpectations:
    owner: str
    recovery_tickets: tuple[bytes, ...]
    residues: tuple[bytes, ...]


def known_journal_slots(prepared, inventory):
    need(type(inventory) is JournalRetentionExpectations, 'explicit journal retention inventory required')
    enum(inventory.owner, ('accepted_attempt','separate_B','external_preexisting','unresolved'))
    need(type(inventory.recovery_tickets) is tuple and type(inventory.residues) is tuple, 'immutable journal instances required')
    tickets = [decode(f) for f in inventory.recovery_tickets]; refs(tickets)
    slots = [dict(key='journal_'+kind, address=journal_address(prepared,kind), owner=inventory.owner,
                  writer='canonical_exclusive_preserve_v1') for kind in ('reservation','dispatch','creation','exit','closure')]
    slots += [dict(key='recovery_'+t['file_sha256'], address=journal_address(prepared,'recovery',t), owner=inventory.owner,
                   writer='canonical_exclusive_preserve_v1') for t in tickets]
    occupied = {canonical(s['address']) for s in slots}
    occupied.update(canonical(dict(s['address'],relative_path=s['address']['relative_path']+'.__creating__')) for s in slots)
    for raw in inventory.residues:
        row = decode(raw); object_keys(row, 'address owner parent_address')
        address(row['address']); address(row['parent_address'])
        enum(row['owner'], ('accepted_attempt','separate_B','external_preexisting','unresolved'))
        need(canonical(row['parent_address']) in occupied and canonical(row['address']) not in occupied,
             'residue parent/duplicate physical accounting conflict')
        need(row['address']['root_binding_id'] == row['parent_address']['root_binding_id'], 'residue root conflict')
        occupied.add(canonical(row['address']))
        slots.append(dict(key='residue_'+sha(raw),address=row['address'],owner=row['owner'],writer='component_existing_contract'))
    locations = [(s['address']['root_binding_id'],s['address']['relative_path'].casefold()) for s in slots]
    need(len(set(locations)) == len(locations), 'journal instance alias conflict')
    return slots


HISTORY_KEYS = ('slot_id capture journals recoveries ownership_file consumption_file creation_observation_file '
    'exit_observation_file process_file guard_file guard_expectations streams stream_observations failure_files '
    'inventory_file retention parent_process_identity source_files')


def _observation_interval(observation, *, spec, role, creation_right=False,
                          acquisition=None, capture=None):
    """Independent temporal qualification; ordering does not grant authority.

    Recording a creation acknowledgment is retention, not another creation.
    Therefore only reservation/dispatch/claim/consume and actual creation use
    valid_until. Cleanup/retention observations use the separate cleanup_until.
    Neither accepted lifetime is extended, and both endpoints are inclusive.
    """
    object_keys(observation, 'observer_id clock_epoch_id sequence observed_utc')
    integer(observation['sequence'])
    moment=utc(observation['observed_utc'])
    need(observation['observer_id']==spec['observer_id'] and observation['clock_epoch_id']==spec['clock_epoch_id'],
         role+'_observer_clock_conflict')
    need(utc(spec['valid_from'])<=moment<=utc(spec['valid_until'] if creation_right else spec['cleanup_until']),
         role+'_authority_interval_conflict')
    if acquisition is not None:
        need(type(acquisition) is Acquisition, 'independent acquisition required')
        need(utc(acquisition.first_utc)<=moment<=utc(acquisition.last_utc),role+'_acquisition_interval_conflict')
    if capture is not None:
        records._capture(capture,spec['binding'])
        need(capture['clock_epoch_id']==observation['clock_epoch_id'] and
             utc(capture['started_utc'])<=moment<=utc(capture['finished_utc']),role+'_capture_interval_conflict')


def _check_history(prepared, found, by_ref, history, acquisitions, creation, reaping):
    """Reconstruct closure from independent evidence; record presence is insufficient."""
    spec = decode(prepared.expected.specification); h = decode(history)
    object_keys(h, HISTORY_KEYS); need(h['slot_id'] == prepared.slot_id, 'history slot conflict')
    records._capture(h['capture'],spec['binding']); cap=h['capture']; diagnostics=[]; read_failed=False
    need(spec['valid_from'] <= cap['started_utc'] <= cap['finished_utc'] <= spec['cleanup_until'], 'history capture interval conflict')
    def read(f, scope):
        nonlocal read_failed
        try:
            return qualify(f, acquisitions, validators=prepared.expected.validator_files,scope=scope,
                           first=cap['started_utc'],last=cap['finished_utc'])
        except LaunchAuthorityError:
            read_failed=True
            diagnostics.append(scope+'_evidence_unavailable'); raise
    try:
        need(type(h['source_files']) is list,'independent source inventory required')
        sources=[]
        source_root=next(r['root_binding_id'] for r in spec['roots'] if r['role']=='source')
        for item in h['source_files']:
            object_keys(item,'path file');ref(item['file'])
            need(item['file']['root_binding_id']==source_root and item['file']['relative_path']==item['path'], 'source root/path binding conflict')
            sources.append(SourceBytes(item['path'],read(item['file'],'execution_source')))
        verify_execution_source_bytes(expected=prepared.expected.source_expectations,supplied=tuple(sources))
        object_keys(h['journals'], 'reservation dispatch creation exit closure')
        for kind, f in h['journals'].items():
            ref(f)
            read(f,'launch_journal')
            need(canonical(f) in by_ref and by_ref[canonical(f)] == found.get(kind), 'journal inventory correspondence conflict')
            _observation_interval(found[kind]['observation'],spec=spec,role=kind,
                creation_right=kind in ('reservation','dispatch'),capture=cap)
        refs(h['recoveries']); refs(h['failure_files'])
        need(set(found) == set(h['journals']) | {'recovery:'+f['file_sha256'] for f in h['recoveries']}, 'unexpected journal membership conflict')
        res=found['reservation']; dispatch=found['dispatch']; created=found['creation']; exited=found['exit']; closed=found['closure']
        need(dispatch['payload']['reservation_file']==h['journals']['reservation']
             and created['payload']['dispatch_file']==h['journals']['dispatch']
             and exited['payload']['creation_file']==h['journals']['creation'], 'journal predecessor identity conflict')
        creation_raw=read(h['creation_observation_file'],'process_observation')
        identity=validate_process_observation(creation_raw,actual_file=h['creation_observation_file'],prepared=prepared,dispatch_file=h['journals']['dispatch'])
        cr=decode(creation_raw); xr=decode(read(h['exit_observation_file'],'process_observation'))
        validate_process_observation(canonical(xr),actual_file=h['exit_observation_file'],prepared=prepared)
        _observation_interval(cr['observation'],spec=spec,role='actual_creation',creation_right=True,capture=cap)
        _observation_interval(xr['observation'],spec=spec,role='actual_exit',capture=cap)
        need(identity==created['payload']['process_identity']==exited['payload']['process_identity']==xr['process_identity']
             and cr['parent_process_identity']==h['parent_process_identity'], 'process identity/parent conflict')
        need(created['payload']['observation_files']==[h['creation_observation_file']]
             and exited['payload']['observation_files']==[h['exit_observation_file']], 'process observation correspondence conflict')
        ordered=[res['observation'],dispatch['observation'],cr['observation'],created['observation'],xr['observation'],exited['observation'],closed['observation']]
        need(all(x['clock_epoch_id']==spec['clock_epoch_id'] for x in ordered)
             and all(x['sequence']<y['sequence'] and x['observed_utc']<=y['observed_utc'] for x,y in zip(ordered,ordered[1:])), 'process chronology conflict')
        for scope, f, schema, prior, after, before in (
            ('claim_observation',h['ownership_file'],'state_claim_observation.v1',None,res['observation'],dispatch['observation']),
            ('consumption_observation',h['consumption_file'],'state_consumption_observation.v1',h['ownership_file'],dispatch['observation'],cr['observation'])):
            row=decode(read(f,scope))
            object_keys(row,'schema authority_id slot_id reservation_file dispatch_file ownership_file owner_binding_file exclusive consumed_count observation')
            need(row['schema']==schema and row['authority_id']==prepared.authority_id and row['slot_id']==prepared.slot_id
                 and row['reservation_file']==h['journals']['reservation'] and row['ownership_file']==prior
                 and row['dispatch_file']==(h['journals']['dispatch'] if prior else None)
                 and row['owner_binding_file']==spec['owner_binding_file'] and row['exclusive'] is True
                 and type(row['consumed_count']) is int and row['consumed_count']==(1 if prior else 0), 'exclusive claim/consumption conflict')
            o=row['observation']; object_keys(o,'observer_id clock_epoch_id sequence observed_utc')
            _observation_interval(o,spec=spec,role=scope,creation_right=True,capture=cap)
            need(o['observer_id']==spec['observer_id'] and o['clock_epoch_id']==spec['clock_epoch_id']
                 and after['sequence']<o['sequence']<before['sequence']
                 and after['observed_utc']<=o['observed_utc']<=before['observed_utc'], 'claim chronology conflict')
        from operatormark.corpus.state_guard_capture import GuardExpectations, derive_guard_capture
        gx=h['guard_expectations']; object_keys(gx,'specification validator_files')
        gs=gx['specification']
        need(gs['binding']==spec['binding'] and gs['launch_slot_id']==prepared.slot_id and gs['capture']==cap
             and gs['creation_file']==h['creation_observation_file'] and gs['exit_file']==h['exit_observation_file']
             and tuple(canonical(f) for f in gx['validator_files'])==prepared.expected.validator_files, 'guard launch correspondence conflict')
        recomputed=derive_guard_capture(expected=GuardExpectations(canonical(gs),prepared.expected.validator_files),independent_files=acquisitions)
        guard=read(h['guard_file'],'execution_guards')
        need(guard==recomputed.closure, 'guard recomputation conflict')
        if recomputed.diagnostics: diagnostics.extend(recomputed.diagnostics)
        gd=decode(guard)
        if any(v['value'] != 0 or v['measurement']!='measured' for v in gd['forbidden_counts'].values()):diagnostics.append('forbidden_requests_observed_or_unavailable')
        process=decode(read(h['process_file'],'process_product'))
        if spec['binding']['state']=='state6':publication.validate_publication_process_outcome(process,
            version=2 if spec['binding']['benchmark']=='darcy_branch_c_v2' else 1)
        else:records.validate_state_process_outcome(process)
        expected_process_role = (_expected_publication_process_role(spec['role'])
                                 if spec['binding']['state'] == 'state6' else spec['role'])
        need(process['binding']==(spec['publication_binding'] or spec['binding']) and process['role']==expected_process_role
             and process['source_closure_file']==spec['source_closure_file'] and process['authority_scope_file']==spec['authority_scope_file']
             and process['process_identity']==dict(pid=identity['pid'],creation_evidence_file=h['creation_observation_file'])
             and process['creation_status']==creation and process['direct_exit_code']==xr['direct_exit_code'] and process['reap_status']==reaping
             and process['guards_file']==h['guard_file'], 'process product correspondence conflict')
        pc=process['capture']
        need(pc['clock_epoch_id']==spec['clock_epoch_id'] and dispatch['observation']['sequence']<=pc['first_sequence']<=pc['last_sequence']<closed['observation']['sequence']
             and pc['last_sequence']>=exited['observation']['sequence'] and cap['started_utc']<=pc['started_utc']<=pc['finished_utc']<=cap['finished_utc'], 'process product chronology conflict')
        object_keys(h['streams'],'stdout stderr');object_keys(h['stream_observations'],'stdout stderr')
        for role,f in h['streams'].items():
            raw=read(f,'execution_'+role)
            witness=decode(read(h['stream_observations'][role],'stream_observation'))
            object_keys(witness,'slot_id stream file eof closed capture process_identity')
            need(witness['slot_id']==prepared.slot_id and witness['stream']==role and witness['file']==f and witness['capture']==cap
                 and witness['process_identity']==identity, 'stream/process correspondence conflict')
            need(type(witness['eof']) is bool and type(witness['closed']) is bool, 'stream flags conflict')
            state='complete' if witness['eof'] and witness['closed'] else 'partial'
            need(process[role]==dict(state=state,file=f),'process stream conclusion conflict')
            if state!='complete':diagnostics.append(role+'_closure_unavailable')
        for role,f in dict(h['streams'],guards=h['guard_file'],process=h['process_file']).items():
            slots=[s for s in spec['output_slots'] if s['key']==role]
            need(len(slots)==1 and {k:f[k] for k in ('root_binding_id','relative_path')}==slots[0]['address'], 'output root/slot conflict')
        cp=closed['payload']
        need(all(cp[k+'_file']==h['journals'][k] for k in ('reservation','dispatch','creation','exit'))
             and cp['process_outcome_file']==h['process_file'] and cp['guard_closure_file']==h['guard_file'], 'closure reference conflict')
        retained=[h['journals'][k] for k in ('reservation','dispatch','creation','exit')]+list(h['streams'].values())+[h['guard_file'],h['process_file']]
        need(set(map(canonical,cp['surviving_files']))==set(map(canonical,retained)), 'closure survivor inventory conflict')
        for f in h['failure_files']:read(f,'failure_evidence')
        need(cp['failure_files']==h['failure_files'], 'closure failure inventory conflict')
        if h['failure_files'] or process['failures'] or cr['failure_files'] or xr['failure_files']:diagnostics.append('failure_evidence_present')
        if creation!='started' or cr['call_status']!='returned_success' or reaping!='observed_reaped' or xr['direct_exit_code']!=0 or cp['resolution']!='created_closed':diagnostics.append('process_not_successfully_closed')
        rt=h['retention'];object_keys(rt,'owner recovery_tickets residues')
        inventory=JournalRetentionExpectations(rt['owner'],tuple(canonical(f) for f in rt['recovery_tickets']),tuple(canonical(r) for r in rt['residues']))
        slots=known_journal_slots(prepared,inventory); known={canonical(s['address']):s['owner'] for s in slots}
        for s in slots:
            if s['writer']=='canonical_exclusive_preserve_v1':known[canonical(dict(s['address'],relative_path=s['address']['relative_path']+'.__creating__'))]=s['owner']
        inv=decode(read(h['inventory_file'],'journal_inventory'))
        object_keys(inv,'slot_id capture observation enumeration entries unavailable_addresses')
        enum(inv['enumeration'],('complete','unavailable'))
        need(type(inv['unavailable_addresses']) is list,'inventory unavailable addresses required')
        for item in inv['unavailable_addresses']:
            address(item);need(canonical(item) in known,'unavailable address outside journal inventory')
        need(inv['slot_id']==prepared.slot_id and inv['capture']==cap, 'retention inventory binding conflict')
        if inv['enumeration']!='complete' or inv['unavailable_addresses']:diagnostics.append('journal_inventory_incomplete')
        need(type(inv['entries']) is list,'inventory entries required')
        seen={}
        for row in inv['entries']:
            object_keys(row,'address owner presence file'); address(row['address']);key=canonical(row['address'])
            need(key in known and key not in seen and row['owner']==known[key], 'journal inventory ownership/membership conflict')
            seen[key]=row
            enum(row['presence'],('present','absent','unavailable'))
            if row['presence']=='present':
                ref(row['file']);need({k:row['file'][k] for k in ('root_binding_id','relative_path')}==row['address'],'inventory FileRef address conflict')
                read(row['file'],'retained_journal')
            else:
                need(row['file'] is None,'absent/unavailable inventory content conflict')
                if row['presence']=='unavailable':diagnostics.append('journal_entry_unavailable')
        if set(seen)!=set(known):diagnostics.append('journal_inventory_omitted')
        for f in list(h['journals'].values())+h['recoveries']:
            row=seen.get(canonical({k:f[k] for k in ('root_binding_id','relative_path')}))
            need(row is not None and row['presence']=='present' and row['file']==f,'journal inventory retained identity conflict')
        cutoff=closed['observation']
        need(len(h['recoveries'])==len(rt['recovery_tickets']),'recovery inventory conflict')
        for f,ticket in zip(h['recoveries'],rt['recovery_tickets']):
            recovery=decode(read(f,'launch_journal'));validate_launch_record(recovery,prepared=prepared,actual_file=f);p=recovery['payload']
            _observation_interval(recovery['observation'],spec=spec,role='recovery',capture=cap)
            need(p['recovery_ticket_file']==ticket and recovery['prior_record_file']==h['journals']['closure'], 'recovery predecessor/ticket conflict')
            ticket_row=decode(read(ticket,'recovery_authority'))
            object_keys(ticket_row,'slot_id validator_file ownership_fence_file authority')
            need(ticket_row['slot_id']==prepared.slot_id and ticket_row['validator_file']==spec['recovery_validator_file']
                 and ticket_row['authority']=='inspect_only' and ticket_row['ownership_fence_file']==p['ownership_fence_file'],'recovery authority conflict')
            fence=decode(read(p['ownership_fence_file'],'recovery_fence'))
            need(fence==dict(slot_id=prepared.slot_id,consumption_file=h['consumption_file'],creation_right='consumed'), 'recovery fence conflict')
            need(set(map(canonical,p['inspected_files']))==set(map(canonical,h['journals'].values()))
                 and p['process_observation_files']==[h['creation_observation_file'],h['exit_observation_file']], 'recovery evidence population conflict')
            expected_conclusions=dict(creation=creation,reaping=reaping,retention='incomplete' if diagnostics or p['unavailable_addresses'] else 'complete',authority='consumed_or_unavailable')
            need(p['conclusions']==expected_conclusions and not p['conflicts'], 'recovery recomputation conflict')
            need(cutoff['sequence']<recovery['observation']['sequence'] and cutoff['observed_utc']<=recovery['observation']['observed_utc'],'recovery chronology conflict')
            cutoff=recovery['observation']
        o=inv['observation']; object_keys(o,'observer_id clock_epoch_id sequence observed_utc')
        _observation_interval(o,spec=spec,role='inventory',capture=cap)
        need(o['observer_id']==spec['observer_id'] and o['clock_epoch_id']==spec['clock_epoch_id'] and o['sequence']>cutoff['sequence']
             and cutoff['observed_utc']<=o['observed_utc']<=cap['finished_utc'], 'inventory cutoff conflict')
    except (LaunchAuthorityError,records.FinalEvidenceRecordError,KeyError,TypeError,ValueError) as exc:
        if not read_failed:
            args=BaseException.__dict__['args'].__get__(exc,type(exc))
            reason=args[0] if type(args) is tuple and args and type(args[0]) is str else 'invalid structure'
            diagnostics.append('independent_history_conflict:'+reason)
    return diagnostics


def validate_process_observation(data, *, actual_file, prepared, dispatch_file=None):
    """Identity comes from retained bytes at their independently supplied address.

    In particular the creation bytes do NOT contain their own FileRef/hash.
    This function establishes correspondence, not observer custody.
    """
    check_bytes(actual_file, data)
    row = decode(data); spec = decode(prepared.expected.specification)
    kind = row.get('kind')
    if kind == 'creation':
        object_keys(row, 'schema kind authority_id slot_id operation role parent_slot_id parent_process_identity dispatch_file source_binding_sha256 command_binding_sha256 observation call_status creation_status pid failure_files')
        need(row['schema'] == 'state_process_creation_observation.v2', 'creation observation schema')
        selected = next(s for s in spec['child_slots'] if s['key'] == prepared.expected.slot_key)
        parent = next((s for s in spec['child_slots'] if s['key'] == selected['parent_key']), None)
        parent_id = None if parent is None else sha(b'operatormark-launch-slot-v1\n' + canonical(dict(authority_id=prepared.authority_id, **parent)))
        need(row['authority_id'] == prepared.authority_id and row['slot_id'] == prepared.slot_id
             and (row['operation'], row['role']) == (spec['operation'], spec['role'])
             and row['parent_slot_id'] == parent_id and row['dispatch_file'] == dispatch_file
             and row['source_binding_sha256'] == spec['source_binding_sha256']
             and row['command_binding_sha256'] == spec['command_binding_sha256'], 'creation observation launch binding conflict')
        ref(row['dispatch_file'])
        process_identity(row['parent_process_identity'], parent is None)
        need(parent is not None or row['parent_process_identity'] is None, 'unexpected parent process')
        enum(row['call_status'], records.CALLS); enum(row['creation_status'], ('started','not_started','unknown'))
        if row['creation_status'] == 'started':
            integer(row['pid'], 1)
            need(row['call_status'] != 'not_called', 'uncalled creation contradiction')
            identity = dict(pid=row['pid'], creation_observation_file=actual_file)
        else:
            need(row['pid'] is None, 'unestablished process identity')
            need(row['creation_status'] != 'not_started' or row['call_status'] in ('not_called','returned_failure'), 'creation refusal contradiction')
            identity = None
        if row['creation_status'] == 'unknown' or row['call_status'] in ('returned_failure','outcome_unavailable'):
            need(bool(row['failure_files']), 'creation failure evidence missing')
    else:
        object_keys(row, 'schema kind slot_id process_identity direct_exit_code reap_status observation failure_files')
        need(row['schema'] == 'state_process_exit_observation.v2' and kind == 'exit' and row['slot_id'] == prepared.slot_id, 'exit observation binding')
        identity = row['process_identity']; process_identity(identity)
        need(row['direct_exit_code'] is None or type(row['direct_exit_code']) is int, 'direct exit exact integer')
        enum(row['reap_status'], ('observed_reaped','unresolved','failed'))
        need(row['reap_status'] != 'observed_reaped' or row['direct_exit_code'] is not None, 'reaping evidence contradiction')
        need(row['reap_status'] == 'observed_reaped' or bool(row['failure_files']), 'exit failure evidence missing')
    refs(row['failure_files'])
    o = row['observation']; object_keys(o, 'observer_id clock_epoch_id sequence observed_utc')
    integer(o['sequence']); utc(o['observed_utc'])
    need(o['observer_id'] == spec['observer_id'] and o['clock_epoch_id'] == spec['clock_epoch_id']
         and spec['valid_from'] <= o['observed_utc'] <= (spec['valid_until'] if kind == 'creation' else spec['cleanup_until']), 'process observation observer/interval conflict')
    return identity


def inspect_launch_history(*, expected, independent_inputs, independent_records, independent_observations, history=None):
    """Read-only reconstruction. Failed reads remain explicit unknown evidence."""
    prepared = prepare_launch(expected=expected, independent_inputs=independent_inputs)
    spec=decode(expected.specification)
    capture=None
    if history is not None:
        descriptor=decode(history);object_keys(descriptor,HISTORY_KEYS)
        capture=descriptor['capture'];records._capture(capture,spec['binding'])
        need(utc(spec['valid_from'])<=utc(capture['started_utc'])<=utc(capture['finished_utc'])<=utc(spec['cleanup_until']),
             'history capture interval conflict')
    need(type(independent_records) is tuple, 'immutable independently read journal required')
    diagnostics = []; found = {}; by_ref = {}; unavailable = False
    if history is None:
        diagnostics.append('independent_history_inventory_unavailable'); unavailable = True
    for acquisition in independent_records:
        need(type(acquisition) is Acquisition, 'independent journal acquisition required')
        reference = decode(acquisition.file)
        try:
            data = qualify(reference, (acquisition,), validators=expected.validator_files, scope='launch_journal',
                           first=acquisition.first_utc, last=acquisition.last_utc)
        except LaunchAuthorityError:
            unavailable = True; diagnostics.append('journal_read_unavailable'); continue
        record = decode(data)
        kind = validate_launch_record(record, prepared=prepared, actual_file=reference)
        try:
            _observation_interval(record['observation'],spec=spec,role=kind,
                creation_right=kind in ('reservation','dispatch'),acquisition=acquisition,capture=capture)
        except LaunchAuthorityError as exc:
            # Retain usable facts and exact contradictions; never let a later
            # successful closure erase an earlier out-of-interval observation.
            diagnostics.append(BaseException.__dict__['args'].__get__(exc,type(exc))[0])
            unavailable=True
        if kind == 'recovery':kind += ':'+reference['file_sha256']
        if kind in found:
            diagnostics.append('duplicate_journal_kind'); continue
        found[kind] = record; by_ref[canonical(reference)] = record
    for kind, record in found.items():
        previous = record['prior_record_file']
        if previous is not None:
            earlier = by_ref.get(canonical(previous))
            if earlier is None:
                diagnostics.append('predecessor_unavailable'); unavailable = True
            elif (earlier['observation']['clock_epoch_id'] != record['observation']['clock_epoch_id']
                  or earlier['observation']['sequence'] >= record['observation']['sequence']
                  or earlier['observation']['observed_utc'] > record['observation']['observed_utc']):
                diagnostics.append('journal_chronology_conflict')
    creation = 'unknown'; reaping = 'unresolved'
    # Acquisition of a creation record alone is not independent process evidence.
    for kind in ('creation','exit'):
        if kind not in found:
            continue
        p = found[kind]['payload']
        observations = []
        for f in p['observation_files']:
            try:
                raw = qualify(f, independent_observations, validators=expected.validator_files,
                              scope='process_observation', first=found[kind]['observation']['observed_utc'],
                              last=found[kind]['observation']['observed_utc'])
                observed=decode(raw)
                matches=[x for x in independent_observations if decode(x.file)==f and x.scope=='process_observation']
                need(len(matches)==1,'missing/duplicate process interval acquisition')
                _observation_interval(observed['observation'],spec=spec,role='actual_'+kind,
                    creation_right=kind=='creation',acquisition=matches[0],capture=capture)
                observations.append((f, raw))
            except LaunchAuthorityError:
                diagnostics.append('process_observation_unavailable'); unavailable = True
        if not observations:
            continue
        if kind == 'creation':
            try:
                need(len(observations) == 1, 'ambiguous identity observation')
                f, raw = observations[0]
                identity = validate_process_observation(raw, actual_file=f, prepared=prepared, dispatch_file=p['dispatch_file'])
                x = decode(raw)
                need(identity == p['process_identity'] and all(x[k] == p[k] for k in ('call_status','creation_status','failure_files')), 'creation facts conflict')
                need(x['observation']['sequence'] < found[kind]['observation']['sequence'], 'creation journal cutoff conflict')
                creation = p['creation_status']
            except LaunchAuthorityError:
                diagnostics.append('creation_observation_conflict')
        else:
            try:
                need(len(observations) == 1, 'ambiguous exit observation')
                f, raw = observations[0]
                validate_process_observation(raw, actual_file=f, prepared=prepared)
                x = decode(raw)
                need(all(x[k] == p[k] for k in ('process_identity','direct_exit_code','reap_status','failure_files')), 'exit facts conflict')
                need(x['observation']['sequence'] < found[kind]['observation']['sequence'], 'exit journal cutoff conflict')
                reaping = p['reap_status']
            except LaunchAuthorityError:
                diagnostics.append('exit_observation_conflict')
            if 'creation' in found and p['process_identity'] != found['creation']['payload']['process_identity']:
                diagnostics.append('process_identity_conflict')
    if 'creation' not in found:
        diagnostics.append('creation_evidence_unavailable'); unavailable = True
    if 'closure' not in found:
        diagnostics.append('closure_evidence_unavailable'); unavailable = True
    if history is not None:
        extra = _check_history(prepared, found, by_ref, history, independent_observations, creation, reaping)
        diagnostics.extend(extra); unavailable = unavailable or bool(extra)
    conflict = any('conflict' in x or 'duplicate' in x for x in diagnostics)
    return LaunchInspection('mismatch' if conflict else 'incomplete' if unavailable else 'corroborated',
                            creation, reaping, 'incomplete' if unavailable else 'complete', tuple(diagnostics))
