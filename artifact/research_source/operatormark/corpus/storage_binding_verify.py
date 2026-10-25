"""Confidential, pure cross-authorization storage comparison; no acquisition.

Expected inputs are externally accepted trust inputs, not credentials. A match
never authenticates a provider, completes sizing, or authorizes any execution.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import json
import ntpath
import re
from types import MappingProxyType
import unicodedata

from . import state3_audit_verify as audit

PREFIX = 'operatormark.darcy_branch_c_v1.'
OPENING_SCHEMA = PREFIX + 'private_storage_binding_openings.v1'
REPORT_SCHEMA = PREFIX + 'private_storage_binding_report.v1'
CLASSIFICATION = 'confidential_private_storage_binding_evidence'
PROFILE = 'state3_to_state4_new_attempt_v1'
BASE = 'artifacts/darcy_branch_c_v1/private_corpus_v1/'
SELF = 'record_self_sha256'
OPENING_KEYS = frozenset(('schema', 'disclosure_classification', 'protocol_file_sha256',
    'authorization_file_sha256', 'state', 'implementation_commit', 'implementation_tree',
    'expected_parent_commit', 'public_record_file_sha256', 'public_record_bytes',
    'public_record_self_sha256', 'commitment_openings', 'git_ignore_stdout_hex_by_candidate',
    'git_worktree_stdout_hex', SELF))
RESERVED = {'aux', 'clock$', 'con', 'nul', 'prn', *(f'com{i}' for i in range(1,10)), *(f'lpt{i}' for i in range(1,10))}


class StorageBindingInputError(ValueError):
    """Malformed independent input, strict opening or recomputed report."""


def _file_freeze(value):
    return MappingProxyType(dict(value)) if isinstance(value, Mapping) else value


@dataclass(frozen=True)
class StorageBindingRecordExpectation:
    state: str
    authorization_file_sha256: str
    public_file: Mapping
    openings_file: Mapping
    acquisition_binding_file: Mapping
    candidate_relative_paths: tuple[str, ...]

    def __post_init__(self):
        for name in ('public_file', 'openings_file', 'acquisition_binding_file'):
            object.__setattr__(self, name, _file_freeze(getattr(self, name)))


@dataclass(frozen=True)
class StorageBindingExpectedInputs:
    profile: str
    protocol_file_sha256: str
    implementation_commit: str
    implementation_tree: str
    expected_parent_commit: str
    state3_audit_file_sha256: str
    compute_feasibility_report_file_sha256: str
    baseline_storage_member: str
    baseline: StorageBindingRecordExpectation
    current: StorageBindingRecordExpectation
    accepted_local_user_residual_sids: tuple[str, ...]


def _plain(value):
    if type(value) in (StorageBindingExpectedInputs, StorageBindingRecordExpectation):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, UnicodeError) as exc:
        raise StorageBindingInputError('noncanonical_input') from exc


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _descriptor(data):
    return dict(file_sha256=_sha(data), file_bytes=len(data))


def _exact(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise StorageBindingInputError('schema_violation')


def _hash(value, length=64):
    if type(value) is not str or re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is None:
        raise StorageBindingInputError('schema_violation')


def _integer(value):
    if type(value) is not int or value < 0:
        raise StorageBindingInputError('schema_violation')


def _sid(value):
    if type(value) is not str or re.fullmatch(r'S-1-(?:0|[1-9][0-9]*)(?:-(?:0|[1-9][0-9]*))+', value) is None:
        raise StorageBindingInputError('schema_violation')
    parts = value.split('-')[2:]
    if int(parts[0]) >= 2**48 or len(parts[1:]) > 15 or any(int(x) >= 2**32 for x in parts[1:]):
        raise StorageBindingInputError('schema_violation')


def _relative(value, *, base=False):
    if type(value) is not str or not value or '\\' in value or value != unicodedata.normalize('NFC', value):
        raise StorageBindingInputError('schema_violation')
    body = value[:-1] if base and value.endswith('/') else value
    parts = body.split('/')
    if (base and not value.endswith('/')) or any(p in ('', '.', '..') or p[-1:] in (' ', '.') or
            p.split('.')[0].casefold() in RESERVED or any(ord(c)<32 or c in '<>:"|?*' for c in p) for p in parts):
        raise StorageBindingInputError('schema_violation')
    try:
        if any(len(p.encode('utf-16-le'))//2 > 255 for p in parts):
            raise StorageBindingInputError('schema_violation')
    except UnicodeError as exc:
        raise StorageBindingInputError('schema_violation') from exc


def _record_expectation(value):
    if type(value) is not StorageBindingRecordExpectation or type(value.state) is not str or value.state not in ('state3','state4'):
        raise StorageBindingInputError('invalid record expectation')
    _hash(value.authorization_file_sha256)
    if value.authorization_file_sha256 == '0'*64:
        raise StorageBindingInputError('missing accepted authorization')
    for name in ('public_file','openings_file','acquisition_binding_file'):
        row = getattr(value, name)
        if type(row) is not MappingProxyType or set(row) != {'file_sha256','file_bytes'}:
            raise StorageBindingInputError('invalid immutable file expectation')
        _hash(row['file_sha256']); _integer(row['file_bytes'])
        if row['file_sha256'] == '0'*64:
            raise StorageBindingInputError('missing independently accepted file binding')
    candidates = value.candidate_relative_paths
    if type(candidates) is not tuple or not candidates or candidates[0] != BASE:
        raise StorageBindingInputError('invalid expected candidate plan')
    seen = set()
    for index, candidate in enumerate(candidates):
        _relative(candidate, base=index==0)
        if index and not candidate.startswith(BASE):
            raise StorageBindingInputError('candidate outside frozen base')
        alias = unicodedata.normalize('NFC', candidate).casefold().rstrip('/')
        if alias in seen:
            raise StorageBindingInputError('duplicate expected candidate')
        seen.add(alias)


def validate_storage_binding_expected_inputs(expected):
    if type(expected) is not StorageBindingExpectedInputs:
        raise StorageBindingInputError('immutable expected inputs required')
    if expected.profile != PROFILE or expected.baseline_storage_member != 'storage_preflight_after.json':
        raise StorageBindingInputError('unsupported_profile')
    for name in ('protocol_file_sha256','state3_audit_file_sha256','compute_feasibility_report_file_sha256'):
        _hash(getattr(expected,name))
        if getattr(expected,name) == '0'*64:
            raise StorageBindingInputError('missing accepted predecessor/protocol binding')
    for name in ('implementation_commit','implementation_tree','expected_parent_commit'):
        _hash(getattr(expected,name),40)
        if getattr(expected,name) == '0'*40:
            raise StorageBindingInputError('missing accepted source binding')
    _record_expectation(expected.baseline); _record_expectation(expected.current)
    if expected.baseline.state != 'state3' or expected.current.state != 'state4' or expected.baseline.authorization_file_sha256 == expected.current.authorization_file_sha256:
        raise StorageBindingInputError('distinct A3/A4 bindings required')
    if expected.current.candidate_relative_paths != (BASE, BASE+'attempts/s4-'+expected.current.authorization_file_sha256[:24]):
        raise StorageBindingInputError('invalid independently accepted A4 plan')
    sids = expected.accepted_local_user_residual_sids
    if type(sids) is not tuple or any(type(x) is not str for x in sids) or sids != tuple(sorted(set(sids))):
        raise StorageBindingInputError('invalid immutable residual acceptance')
    for sid in sids:
        _sid(sid)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StorageBindingInputError('noncanonical_input')
        result[key] = value
    return result


def _load(data):
    if type(data) is not bytes:
        raise StorageBindingInputError('noncanonical_input')
    try:
        value = json.loads(data.decode('utf-8'), object_pairs_hook=_pairs)
    except (ValueError, UnicodeError) as exc:
        raise StorageBindingInputError('noncanonical_input') from exc
    if _canonical(value) != data:
        raise StorageBindingInputError('noncanonical_input')
    return value


def _self(record):
    if type(record) is not dict or SELF not in record:
        raise StorageBindingInputError('schema_violation')
    _hash(record[SELF])
    if record[SELF] != _sha(_canonical({k:v for k,v in record.items() if k != SELF})):
        raise StorageBindingInputError('self_hash_invalid')
    return record[SELF]


def _sealed(record):
    value = {k:v for k,v in record.items() if k != SELF}
    return {**value, SELF:_sha(_canonical(value))}


def _hex(value):
    if type(value) is not str or re.fullmatch(r'(?:[0-9a-f]{2})*', value) is None:
        raise StorageBindingInputError('schema_violation')
    return bytes.fromhex(value)


def _utf8(value):
    if type(value) is not str:
        raise StorageBindingInputError('schema_violation')
    try:
        return value.encode('utf-8')
    except UnicodeError as exc:
        raise StorageBindingInputError('schema_violation') from exc


def _text(data):
    try:
        return data.decode('utf-8')
    except UnicodeError as exc:
        raise StorageBindingInputError('schema_violation') from exc


def _at(value, pointer):
    for part in pointer.split('/')[1:]:
        value = value[int(part)] if isinstance(value,list) else value[part]
    return value


def _rules(public):
    """Derive every non-null commitment occurrence from the public schema."""
    result = {}
    def add(pointer, domain, encoding='utf8', kind='text'):
        if _at(public,pointer) is not None:
            result[pointer] = (domain,encoding,kind)
    add('/repository_root_commitment','operatormark-repository-root-v1',kind='path')
    add('/destination_absolute_path_commitment','operatormark-private-destination-v1',kind='path')
    for i in range(3):
        p = '/path_component_evidence/'+str(i)
        add(p+'/canonical_path_commitment','operatormark-storage-component-path-v1',kind='path')
        add(p+'/file_identity_commitment','operatormark-storage-component-file-id-v1',kind='file_id')
        add(p+'/volume_identifier_commitment','operatormark-private-volume-v1',kind='volume')
    add('/volume_evidence/volume_identifier_commitment','operatormark-private-volume-v1',kind='volume')
    add('/volume_evidence/volume_root_commitment','operatormark-private-volume-root-v1',kind='path')
    for i in range(len(public['sync_detection_evidence']['configured_sync_roots'])):
        p = '/sync_detection_evidence/configured_sync_roots/'+str(i)
        add(p+'/lexical_root_commitment','operatormark-storage-sync-lexical-root-v1',kind='path')
        add(p+'/canonical_root_commitment','operatormark-storage-sync-canonical-root-v1',kind='path')
    for i in range(len(public['sync_detection_evidence']['active_sync_clients'])):
        add('/sync_detection_evidence/active_sync_clients/'+str(i)+'/process_name_commitment','operatormark-storage-sync-process-v1')
    for i in range(public['git_ignore_evidence']['candidate_count']):
        add('/git_ignore_evidence/candidate_path_commitments/'+str(i),'operatormark-storage-git-candidate-path-v1',kind='candidate')
    add('/permission_evidence/owning_principal_commitment','operatormark-storage-owner-v1',kind='sid')
    add('/permission_evidence/security_descriptor_commitment','operatormark-storage-security-descriptor-v1','hex','descriptor')
    for name in ('broad_principal_effective_access','additional_allow_principal_effective_access'):
        for i in range(len(public['permission_evidence'][name])):
            add('/permission_evidence/'+name+'/'+str(i)+'/principal_commitment','operatormark-storage-principal-v1',kind='sid')
    add('/encryption_evidence/device_encryption_evidence_commitment','operatormark-storage-device-encryption-query-v1',kind='command')
    for service in ('WSearch','wbengine','VSS'):
        add('/backup_and_indexing_evidence/service_query_evidence_commitments/'+service,
            'operatormark-storage-service-query-'+service.casefold()+'-v1',kind='command')
    add('/backup_and_indexing_evidence/registry_query_evidence_commitment','operatormark-storage-exposure-registry-v1','hex','registry')
    for name in ('source_volume_identifier_commitment','renamed_volume_identifier_commitment'):
        add('/probe_evidence/'+name,'operatormark-private-volume-v1',kind='volume')
    return result


def _windows_path(value):
    if (type(value) is not str or value != ntpath.normcase(ntpath.normpath(value)) or
        value != unicodedata.normalize('NFC',value) or re.match(r'^[a-z]:\\',value) is None or 'globalroot' in value):
        raise StorageBindingInputError('semantic_validation_failed')
    for part in value[3:].split('\\'):
        if not part and value[3:]:
            raise StorageBindingInputError('semantic_validation_failed')
        if part and (part[-1] in ' .' or part.split('.')[0] in RESERVED or any(ord(c)<32 or c in '<>:"|?*' for c in part)):
            raise StorageBindingInputError('semantic_validation_failed')
    return value


def _command(text, outcome, interpreted=None):
    first, sep, output = text.partition('\n')
    if not sep or re.fullmatch(r'0|-?[1-9][0-9]*', first) is None or '\r\n' in output or '\n' not in output:
        raise StorageBindingInputError('semantic_validation_failed')
    observed = first == '0'
    if outcome != ('observed' if observed else 'query_failed'):
        raise StorageBindingInputError('semantic_validation_failed')
    if interpreted is not None:
        lowered = output.casefold()
        result = 'unknown_query_failed'
        if observed:
            result = ('reported_protected_not_independently_proven' if 'protection on' in lowered and
                      ('fully encrypted' in lowered or 'percentage encrypted: 100' in lowered) else
                      'reported_not_protected' if 'protection off' in lowered or 'fully decrypted' in lowered else 'observed_but_unclassified')
        if interpreted != result:
            raise StorageBindingInputError('semantic_validation_failed')


def _registry(data, outcome):
    rows = _load(data)
    if type(rows) is not list or len(rows) != 2:
        raise StorageBindingInputError('schema_violation')
    count = 0
    for row,name in zip(rows,('CrashDumpEnabled','PagingFiles'),strict=True):
        _exact(row,('name','present','registry_type'))
        if row['name'] != name:
            raise StorageBindingInputError('schema_violation')
        if row['present'] is None and row['registry_type'] is None:
            continue
        if type(row['present']) is not bool:
            raise StorageBindingInputError('schema_violation')
        _integer(row['registry_type']); count += 1
    if outcome != ('observed' if count==2 else 'partially_observed' if count else 'unavailable'):
        raise StorageBindingInputError('semantic_validation_failed')


def _opening_value(row, rule, public):
    domain, encoding, kind = rule
    if row['domain'] != domain or row['value_encoding'] != encoding:
        raise StorageBindingInputError('schema_violation')
    data = _hex(row['value']) if encoding=='hex' else _utf8(row['value'])
    pointer, value = row['public_pointer'], row['value']
    commitment = _sha(domain.encode()+b'\n'+public['authorization_file_sha256'].encode()+b'\n'+data)
    if commitment != _at(public,pointer):
        raise StorageBindingInputError('commitment_invalid')
    if kind=='path':
        _windows_path(value)
    elif kind in ('volume','file_id'):
        _hash(value,8 if kind=='volume' else 16)
    elif kind=='sid':
        _sid(value)
    elif kind=='descriptor' and not data:
        raise StorageBindingInputError('semantic_validation_failed')
    elif kind=='text' and (not value or '\x00' in value):
        raise StorageBindingInputError('semantic_validation_failed')
    elif kind=='command':
        if pointer.startswith('/encryption_evidence/'):
            e = public['encryption_evidence']
            _command(value,e['device_encryption_query_outcome'],e['device_encryption_interpreted_status'])
        else:
            _command(value,public['backup_and_indexing_evidence']['service_query_outcomes'][pointer.rsplit('/',1)[1]])
    elif kind=='registry':
        _registry(data,public['backup_and_indexing_evidence']['registry_query_outcome'])
    return data


class _Diagnostics:
    def __init__(self):
        self.rows = {k:set() for k in ('permitted_differences','mismatches','unavailable_requirements','invalid_inputs')}
        self.compared = set()
    def add(self, group, side, pointer, code):
        self.rows[group].add((side,pointer,code))
    def invalid(self,side,pointer,code):
        self.add('invalid_inputs',side,pointer,code)
    def mismatch(self,side,pointer,code='expected_binding_mismatch'):
        self.add('mismatches',side,pointer,code)
    def unavailable(self,side,pointer,code):
        self.add('unavailable_requirements',side,pointer,code)


def _load_record(data, expected_file, side, opening, diagnostics):
    label = 'opening_record_unavailable' if opening else 'public_record_unavailable'
    if data is None:
        diagnostics.unavailable(side,'',label); return None,None
    if _descriptor(data) != dict(expected_file):
        diagnostics.invalid(side,'','file_binding_invalid')
    try:
        record = _load(data)
        payload_hash = _self(record)
    except StorageBindingInputError as exc:
        diagnostics.invalid(side,'',str(exc)); return None,None
    return record,payload_hash


def _public(record, expected, side, diagnostics):
    target = getattr(expected,side)
    checks = {'protocol_file_sha256':expected.protocol_file_sha256,
              'implementation_commit':expected.implementation_commit,'implementation_tree':expected.implementation_tree,
              'authorization_file_sha256':target.authorization_file_sha256,'state':target.state}
    for key,value in checks.items():
        if record.get(key) != value:
            diagnostics.mismatch(side,'/'+key)
    try:
        audit._validate_storage_for_protocol(record, expected_protocol_sha256=expected.protocol_file_sha256)
        if not record['canonical_resolution_method'].startswith('GetFinalPathNameByHandleW('):
            raise StorageBindingInputError('unsupported_profile')
        if not isinstance(record['volume_evidence']['filesystem_type'],str) or not record['volume_evidence']['filesystem_type']:
            raise StorageBindingInputError('schema_violation')
    except (audit.State3AuditVerificationError,KeyError,TypeError,ValueError) as exc:
        diagnostics.invalid(side,'',str(exc) if isinstance(exc,StorageBindingInputError) else 'semantic_validation_failed')
        return False
    if record['git_tracking_evidence']['parent_commit'] != expected.expected_parent_commit:
        diagnostics.mismatch(side,'/git_tracking_evidence/parent_commit')
    candidate_commitments = [_sha(b'operatormark-storage-git-candidate-path-v1\n'+
        record['authorization_file_sha256'].encode()+b'\n'+x.encode()) for x in target.candidate_relative_paths]
    if record['git_ignore_evidence']['candidate_path_commitments'] != candidate_commitments:
        diagnostics.mismatch(side,'/git_ignore_evidence/candidate_path_commitments','candidate_plan_mismatch')
    return True


def _inspect_openings(record, public, public_bytes, target, side, diagnostics):
    opened = {}
    rules = _rules(public)
    if record is None:
        for pointer in rules:
            diagnostics.unavailable(side,pointer,'required_opening_unavailable')
        return opened
    try:
        _exact(record,OPENING_KEYS)
        if record['schema'] != OPENING_SCHEMA or record['disclosure_classification'] != CLASSIFICATION:
            raise StorageBindingInputError('schema_violation')
        for key in ('protocol_file_sha256','authorization_file_sha256','public_record_file_sha256','public_record_self_sha256'):
            _hash(record[key])
        for key in ('implementation_commit','implementation_tree','expected_parent_commit'):
            _hash(record[key],40)
        _integer(record['public_record_bytes'])
        if type(record['state']) is not str or not record['state']:
            raise StorageBindingInputError('schema_violation')
        for key in ('protocol_file_sha256','authorization_file_sha256','state','implementation_commit','implementation_tree'):
            if record[key] != public[key]:
                diagnostics.mismatch(side,'/'+key)
        if record['expected_parent_commit'] != public['git_tracking_evidence']['parent_commit']:
            diagnostics.mismatch(side,'/git_tracking_evidence/parent_commit')
        if (record['public_record_file_sha256'] != _sha(public_bytes) or type(record['public_record_bytes']) is not int or
            record['public_record_bytes'] != len(public_bytes) or record['public_record_self_sha256'] != public[SELF]):
            diagnostics.invalid(side,'','file_binding_invalid')
        rows = record['commitment_openings']
        if type(rows) is not list:
            raise StorageBindingInputError('schema_violation')
        pointers = []
        for row in rows:
            _exact(row,('public_pointer','domain','value_encoding','value'))
            p = row['public_pointer']
            if type(p) is not str:
                raise StorageBindingInputError('schema_violation')
            # Never echo an untrusted pointer into diagnostics (it might contain private text).
            if p not in rules:
                diagnostics.invalid(side,'','schema_violation'); continue
            if p in pointers:
                diagnostics.invalid(side,p,'duplicate_identity'); opened.pop(p,None); continue
            pointers.append(p)
            try:
                opened[p] = _opening_value(row,rules[p],public)
            except StorageBindingInputError as exc:
                diagnostics.invalid(side,p,str(exc))
        if pointers != sorted(pointers):
            diagnostics.invalid(side,'','schema_violation')
        for p in rules.keys()-set(pointers):
            diagnostics.unavailable(side,p,'required_opening_unavailable')
        candidates = target.candidate_relative_paths
        actual_candidates = public['git_ignore_evidence']['candidate_path_commitments']
        if len(actual_candidates) != len(candidates):
            diagnostics.mismatch(side,'/git_ignore_evidence/candidate_path_commitments','candidate_plan_mismatch')
        for i,candidate in enumerate(candidates):
            p = '/git_ignore_evidence/candidate_path_commitments/'+str(i)
            if p in opened and opened[p] != candidate.encode():
                diagnostics.mismatch(side,p,'candidate_plan_mismatch')
        outputs = record['git_ignore_stdout_hex_by_candidate']
        if type(outputs) is not list or len(outputs) != len(candidates):
            raise StorageBindingInputError('schema_violation')
        decoded = [_hex(x) for x in outputs]
        outputs_valid = True
        for data,candidate in zip(decoded,candidates,strict=True):
            text = _text(data)
            if re.fullmatch(r'\.gitignore:[1-9][0-9]*:artifacts/\t'+re.escape(candidate)+r'\n',text) is None:
                outputs_valid = False
                diagnostics.invalid(side,'/git_ignore_evidence/check_ignore_output_sha256','semantic_validation_failed')
        if _sha(b''.join(decoded)) != public['git_ignore_evidence']['check_ignore_output_sha256']:
            diagnostics.invalid(side,'/git_ignore_evidence/check_ignore_output_sha256','commitment_invalid')
        elif outputs_valid:
            opened['/git_ignore_evidence/check_ignore_output_sha256'] = b''.join(decoded)
        worktree = _hex(record['git_worktree_stdout_hex'])
        _text(worktree)
        if _sha(worktree) != public['git_tracking_evidence']['worktree_evidence_sha256']:
            diagnostics.invalid(side,'/git_tracking_evidence/worktree_evidence_sha256','commitment_invalid')
        opened['/git_tracking_evidence/worktree_evidence_sha256'] = worktree
    except StorageBindingInputError as exc:
        diagnostics.invalid(side,'',str(exc))
    _private_relations(public,opened,target,side,diagnostics)
    return opened


def _private_relations(public,opened,target,side,d):
    def text(p):
        return _text(opened[p]) if p in opened else None
    root = text('/repository_root_commitment')
    destination = text('/destination_absolute_path_commitment')
    if root and destination and destination != ntpath.join(root,*BASE.rstrip('/').split('/')):
        d.invalid(side,'/destination_absolute_path_commitment','semantic_validation_failed')
    if root:
        for i,role in enumerate(('.', 'artifacts','artifacts/darcy_branch_c_v1')):
            p = '/path_component_evidence/'+str(i)+'/canonical_path_commitment'
            if p in opened and text(p) != (root if role=='.' else ntpath.join(root,*role.split('/'))):
                d.invalid(side,p,'semantic_validation_failed')
        volume_root = text('/volume_evidence/volume_root_commitment')
        if volume_root and volume_root != ntpath.splitdrive(root)[0]+'\\':
            d.invalid(side,'/volume_evidence/volume_root_commitment','semantic_validation_failed')
    for i,(_,sid,_) in enumerate(audit.BROAD_PRINCIPALS):
        p = '/permission_evidence/broad_principal_effective_access/'+str(i)+'/principal_commitment'
        if p in opened and text(p) != sid:
            d.invalid(side,p,'semantic_validation_failed')
    broad = {sid for _,sid,_ in audit.BROAD_PRINCIPALS}
    for i in range(len(public['permission_evidence']['additional_allow_principal_effective_access'])):
        p = '/permission_evidence/additional_allow_principal_effective_access/'+str(i)+'/principal_commitment'
        if p in opened and text(p) in broad:
            d.invalid(side,p,'duplicate_identity')
    roots = public['sync_detection_evidence']['configured_sync_roots']
    active = {x['provider'] for x in public['sync_detection_evidence']['active_sync_clients']}
    seen_clients = set()
    for i,row in enumerate(public['sync_detection_evidence']['active_sync_clients']):
        p = '/sync_detection_evidence/active_sync_clients/'+str(i)
        process = text(p+'/process_name_commitment')
        if process is not None:
            key = (row['provider'],process)
            if key in seen_clients:
                d.invalid(side,p,'duplicate_identity')
            seen_clients.add(key)
    seen_paths = set()
    seen_canonical = {}
    for i,row in enumerate(roots):
        p = '/sync_detection_evidence/configured_sync_roots/'+str(i)
        lexical,canonical = text(p+'/lexical_root_commitment'),text(p+'/canonical_root_commitment')
        if lexical:
            alias = (row['provider'],row['source'],lexical)
            if alias in seen_paths:
                d.invalid(side,p,'duplicate_identity')
            seen_paths.add(alias)
        if lexical and canonical:
            key = (row['provider'],row['source'],canonical)
            if key in seen_canonical and seen_canonical[key] != lexical:
                d.invalid(side,p,'duplicate_identity')
            seen_canonical[key] = lexical
        for value in (lexical,canonical):
            if value and destination and (destination==value or destination.startswith(value.rstrip('\\')+'\\')):
                d.invalid(side,p,'semantic_validation_failed')
        if row['provider'] in active and not row['exists']:
            d.invalid(side,p,'semantic_validation_failed')


def strict_storage_binding_openings_loads(data, *, public_bytes, expected):
    _record_expectation(expected)
    if type(data) is not bytes or type(public_bytes) is not bytes:
        raise StorageBindingInputError('immutable original bytes required')
    public = _load(public_bytes); _self(public)
    try:
        audit._validate_storage_for_protocol(public,expected_protocol_sha256=public['protocol_file_sha256'])
        if not public['canonical_resolution_method'].startswith('GetFinalPathNameByHandleW('):
            raise StorageBindingInputError('unsupported_profile')
    except (audit.State3AuditVerificationError,KeyError,TypeError,ValueError) as exc:
        raise StorageBindingInputError('semantic_validation_failed') from exc
    d = _Diagnostics()
    record,_ = _load_record(data,expected.openings_file,'current',True,d)
    if _descriptor(public_bytes) != dict(expected.public_file):
        d.invalid('current','','file_binding_invalid')
    if public['state'] != expected.state or public['authorization_file_sha256'] != expected.authorization_file_sha256:
        d.mismatch('current','')
    _inspect_openings(record,public,public_bytes,expected,'current',d)
    if any(d.rows.values()):
        raise StorageBindingInputError('opening correspondence failed')
    return record


def canonical_storage_binding_openings_bytes(record, *, public_bytes, expected):
    data = _canonical(record)
    strict_storage_binding_openings_loads(data,public_bytes=public_bytes,expected=expected)
    return data


def _leaves(value,pointer=''):
    if isinstance(value,dict) and value:
        return {p:v for k,x in value.items() for p,v in _leaves(x,pointer+'/'+k).items()}
    if isinstance(value,list) and value:
        return {p:v for i,x in enumerate(value) for p,v in _leaves(x,pointer+'/'+str(i)).items()}
    return {pointer:value}


def _compare(left,right,opened_left,opened_right,d,expected):
    groups = {
        '/sync_detection_evidence/configured_sync_roots':('provider','source','lexical_root_commitment'),
        '/sync_detection_evidence/active_sync_clients':('provider','process_name_commitment'),
        '/permission_evidence/additional_allow_principal_effective_access':('principal_commitment',),
    }
    rules_left,rules_right = _rules(left),_rules(right)
    def leaf(p,q,a,b):
        if p in rules_left or q in rules_right:
            if p not in opened_left or q not in opened_right:
                return
            a,b = opened_left[p],opened_right[q]
            if a==b and _at(left,p) != _at(right,q):
                d.add('permitted_differences','pair',p,'authorization_derived_commitment')
        d.compared.update((p,q))
        if type(a) is not type(b) or a != b:
            d.mismatch('pair',p,'stable_fact_mismatch')
    for group,keys in groups.items():
        maps,orders,completeness = [],[],[]
        for side,record,opened in (('baseline',left,opened_left),('current',right,opened_right)):
            rows = _at(record,group); mapping={}; order=[]; complete=True
            # A repeated public identity commitment is ambiguous even if one
            # occurrence lacks its opening. Never select a surviving duplicate.
            public_keys = [tuple(row[key] for key in keys) for row in rows]
            counts = {}
            for token in public_keys:
                counts[token] = counts.get(token,0)+1
            ambiguous = set()
            for i,row in enumerate(rows):
                if counts[public_keys[i]] != 1:
                    complete=False
                    d.invalid(side,group+'/'+str(i),'duplicate_identity')
                    continue
                identity=[]
                for key in keys:
                    p=group+'/'+str(i)+'/'+key
                    if key.endswith('_commitment'):
                        if p not in opened:
                            complete=False; break
                        identity.append(opened[p])
                    else:
                        identity.append(row[key])
                else:
                    token=tuple(identity)
                    if token in mapping or token in ambiguous:
                        complete=False
                        ambiguous.add(token)
                        mapping.pop(token,None)
                        d.invalid(side,group+'/'+str(i),'duplicate_identity')
                        continue
                    mapping[token]=(i,row); order.append(token)
            # Keep independently known rows when unrelated identities are
            # unavailable. Completeness controls only whole-collection claims.
            maps.append(mapping); orders.append(order); completeness.append(complete)
        if len(_at(left,group)) != len(_at(right,group)):
            d.mismatch('pair',group,'stable_fact_mismatch')
        if all(completeness):
            d.compared.add(group)
            if maps[0].keys()!=maps[1].keys():
                d.mismatch('pair',group,'stable_fact_mismatch')
            if maps[0].keys()==maps[1].keys() and orders[0]!=orders[1]:
                d.add('permitted_differences','pair',group,'commitment_derived_order')
        for key in maps[0].keys() & maps[1].keys():
            i,a=maps[0][key]; j,b=maps[1][key]
            for name in a:
                leaf(group+'/'+str(i)+'/'+name,group+'/'+str(j)+'/'+name,a[name],b[name])
    a,b=_leaves(left),_leaves(right)
    for p in sorted(a.keys()|b.keys()):
        if any(p==g or p.startswith(g+'/') for g in groups):
            continue
        special = {'/state':'state_binding','/authorization_file_sha256':'authorization_binding',
                   '/record_self_sha256':'derived_self_hash','/observed_free_bytes':'volatile_free_space',
                   '/volume_evidence/free_bytes':'volatile_free_space'}
        if p.startswith('/git_ignore_evidence/candidate_path_commitments') or p in {
            '/git_ignore_evidence/candidate_count','/git_ignore_evidence/check_ignore_output_sha256'}:
            # Individual plans/outputs were checked independently; not index-equal lists.
            if (p in rules_left and p not in opened_left) or (p in rules_right and p not in opened_right):
                continue
            if p == '/git_ignore_evidence/check_ignore_output_sha256' and (p not in opened_left or p not in opened_right):
                continue
            d.compared.add(p)
            plan_valid = not any(code=='candidate_plan_mismatch' for _,_,code in d.rows['mismatches'])
            if a.get(p)!=b.get(p) and plan_valid:
                d.add('permitted_differences','pair',p,'authorized_candidate_plan')
            continue
        if p in special:
            d.compared.add(p)
            bound = (p not in ('/state','/authorization_file_sha256') or
                     (a[p] == getattr(expected.baseline,p[1:]) and b[p] == getattr(expected.current,p[1:])))
            if a[p]!=b[p] and bound:
                d.add('permitted_differences','pair',p,special[p])
            continue
        if p not in a or p not in b:
            d.mismatch('pair',p,'stable_fact_mismatch'); continue
        leaf(p,p,a[p],b[p])
    # Exact retained unkeyed worktree preimage, not merely its digest.
    p='/git_tracking_evidence/worktree_evidence_sha256'
    if p in opened_left and p in opened_right and opened_left[p]!=opened_right[p]:
        d.mismatch('pair',p,'stable_fact_mismatch')


def verify_storage_binding(*, baseline_public_bytes, current_public_bytes,
                           baseline_openings_bytes, current_openings_bytes, expected):
    validate_storage_binding_expected_inputs(expected)
    data = dict(baseline_public=baseline_public_bytes,current_public=current_public_bytes,
                baseline_openings=baseline_openings_bytes,current_openings=current_openings_bytes)
    if any(x is not None and type(x) is not bytes for x in data.values()):
        raise StorageBindingInputError('only immutable bytes or explicit unavailable null accepted')
    d=_Diagnostics(); public={}; opened={}; self_hashes={k:None for k in data}
    for side in ('baseline','current'):
        target=getattr(expected,side)
        p,p_hash=_load_record(data[side+'_public'],target.public_file,side,False,d)
        o,o_hash=_load_record(data[side+'_openings'],target.openings_file,side,True,d)
        self_hashes[side+'_public']=p_hash; self_hashes[side+'_openings']=o_hash
        if p is not None and _public(p,expected,side,d):
            public[side]=p
            opened[side]=_inspect_openings(o,p,data[side+'_public'],target,side,d)
            if o is not None and o.get('expected_parent_commit') != expected.expected_parent_commit:
                d.mismatch(side,'/git_tracking_evidence/parent_commit')
            # Residual-set correspondence is independently checkable on current alone.
            if side=='current':
                sids=[]; complete=True
                for name in ('broad_principal_effective_access','additional_allow_principal_effective_access'):
                    for i,row in enumerate(p['permission_evidence'][name]):
                        pointer='/permission_evidence/'+name+'/'+str(i)+'/principal_commitment'
                        if row['residual_risk']:
                            if pointer not in opened[side]:
                                complete=False
                            else:
                                sids.append(_text(opened[side][pointer]))
                if complete and tuple(sorted(sids))!=expected.accepted_local_user_residual_sids:
                    d.mismatch(side,'/permission_evidence','residual_acceptance_mismatch')
    if set(public)=={'baseline','current'}:
        _compare(public['baseline'],public['current'],opened['baseline'],opened['current'],d,expected)
    status = ('invalid' if d.rows['invalid_inputs'] else 'mismatch' if d.rows['mismatches'] else
              'incomplete' if d.rows['unavailable_requirements'] else 'matched')
    return _sealed(dict(schema=REPORT_SCHEMA,disclosure_classification=CLASSIFICATION,profile=expected.profile,
        expected_inputs_sha256=_sha(_canonical(_plain(expected))),
        input_files={k:None if v is None else _descriptor(v) for k,v in data.items()},record_self_hashes=self_hashes,
        comparison_status=status,compared_fields=sorted(d.compared),
        **{key:[dict(side=s,public_pointer=p,code=c) for s,p,c in sorted(rows)] for key,rows in d.rows.items()},
        production_preflight_accepted=False,execution_authorized=False))


def validate_storage_binding_report(report, **inputs):
    candidate = _load(report) if type(report) is bytes else report
    _self(candidate)
    recomputed = verify_storage_binding(**inputs)
    if _canonical(candidate) != _canonical(recomputed):
        raise StorageBindingInputError('report disagrees with complete independent recomputation')


def canonical_storage_binding_report_bytes(report, **inputs):
    validate_storage_binding_report(report,**inputs)
    return _canonical(_load(report) if type(report) is bytes else report)
