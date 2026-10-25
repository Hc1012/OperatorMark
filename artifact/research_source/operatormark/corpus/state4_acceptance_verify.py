"""Pure, conditional State 4 aggregate correspondence; no acceptance authority.

All trusted premises are explicit inputs. This module neither acquires evidence
nor changes any existing State 4 production refusal or serialized representation.
"""
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
import hashlib
import re
from types import MappingProxyType
import unicodedata

from operatormark.corpus.canonical import canonical_json_bytes, strict_canonical_json_loads
from operatormark.corpus import storage
from operatormark.corpus import permission_handoff_records as handoff_records
from operatormark.corpus import permission_handoff_verify as handoff_verify
from operatormark.corpus import permission_authz_records as authz_records
from operatormark.corpus import private_permission_evidence as private_evidence

PREFIX = 'operatormark.darcy_branch_c_v1.'
PROTOCOL_SHA256 = 'a38ef79b6c4bdccf53f599fa08f77e48a111afbf0b2783ad5733bda99996599a'
SPLITS = (('training', 64), ('validation', 4), ('test', 8))
SCOPES = ('source_authority', 'control_capture', 'origin_custody', 'broad_principals',
          'ancestor_policy', 'post_write_protection', 'guard_process_closure')
DIAGNOSTIC_CODES = ('schema_invalid', 'duplicate_or_alias', 'inventory_mismatch', 'binding_mismatch',
    'hash_mismatch', 'identity_mismatch', 'permission_not_satisfied', 'evidence_unavailable',
    'authority_unavailable', 'custody_unavailable', 'policy_coverage_unavailable',
    'protection_unavailable', 'chronology_unavailable', 'cleanup_failed', 'retention_failed',
    'component_rejected', 'upstream_rejected')


class State4AcceptanceInputError(ValueError):
    """Malformed independently supplied contract, without disclosure of evidence."""


def _require(condition, reason):
    if not condition:
        raise State4AcceptanceInputError(reason)


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return [_plain(v) for v in value]
    return value


def _freeze(value):
    if isinstance(value, _Frozen):
        # Our instances have already recursively copied every external field.
        return value
    if isinstance(value, Mapping):
        _require(all(type(k) is str for k in value), 'mapping keys')
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(v) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        result = type(value)(**{f.name: _freeze(getattr(value, f.name)) for f in fields(value) if f.init})
        for field in fields(value):
            if not field.init:
                _require(type(getattr(result, field.name)) is type(getattr(value, field.name)) and
                         getattr(result, field.name) == getattr(value, field.name), 'inconsistent immutable component fields')
        return result
    _require(value is None or type(value) in (str, int, bool, bytes), 'unsupported or mutable value')
    return value


class _Frozen:
    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))


@dataclass(frozen=True, slots=True)
class State4Inventory(_Frozen):
    protected_paths: tuple
    original_paths: tuple


@dataclass(frozen=True, slots=True)
class EvidenceBytes(_Frozen):
    role: str
    file: Mapping
    data: bytes | None


@dataclass(frozen=True, slots=True)
class ExpectedOriginalFile(_Frozen):
    file: Mapping
    native_identity: Mapping
    origin_evidence_file: Mapping
    custody_file: Mapping
    permission_evidence_files: tuple


@dataclass(frozen=True, slots=True)
class ExpectedProtectedFile(_Frozen):
    row_index: int
    expected_metadata_row: Mapping
    expected_plan: handoff_records.HandoffPlan
    expected_source_files: tuple
    accepted_anchor: handoff_records.AcceptedHandoffAnchor | None


@dataclass(frozen=True, slots=True)
class HandoffBundle(_Frozen):
    row_index: int
    start: EvidenceBytes
    outcome: EvidenceBytes
    acquisition: EvidenceBytes
    verification: EvidenceBytes
    index: EvidenceBytes
    receipt: EvidenceBytes


@dataclass(frozen=True, slots=True)
class ExpectedRecord(_Frozen):
    role: str
    file: Mapping
    schema: str
    record: Mapping


@dataclass(frozen=True, slots=True)
class AcceptanceRequirement(_Frozen):
    role: str
    scope: str
    input_files: tuple
    capture: Mapping
    validator_source_files: tuple


@dataclass(frozen=True, slots=True)
class AcceptanceAdmission(_Frozen):
    role: str
    scope: str
    input_files: tuple
    capture: Mapping
    validator_source_files: tuple
    admission_file: Mapping
    status: str


@dataclass(frozen=True, slots=True)
class State4AcceptanceExpectations(_Frozen):
    binding: Mapping
    capture: Mapping
    attempt_root_binding_id: str
    root_requirements: tuple
    metadata_context: Mapping
    original_files: tuple
    protected_files: tuple
    control_shards: tuple
    expected_records: tuple
    record_bindings: tuple
    requirements: tuple
    admissions: tuple


@dataclass(frozen=True, slots=True)
class Diagnostic(_Frozen):
    role: str
    pointer: str
    code: str
    component_reason_codes: tuple


@dataclass(frozen=True, slots=True)
class ProtectedFileVerification(_Frozen):
    row_index: int
    relative_path: str
    handoff_verification: handoff_verify.HandoffVerification | None
    status: str
    diagnostics: tuple


@dataclass(frozen=True, slots=True)
class State4AcceptanceVerification(_Frozen):
    status: str
    protected_results: tuple
    originals_checked: int
    control_shards_checked: int
    diagnostics: tuple
    permission_acceptance_emitted: bool
    production_preflight_accepted: bool
    execution_authorized: bool


def derive_state4_acceptance_inventory() -> State4Inventory:
    protected = []
    for split, count in SPLITS:
        directory = 'test.sealed' if split == 'test' else split
        for index in range(count):
            base = 'state4_controls/' + directory + f'/shard_{index:04d}'
            protected.extend(('corpus_staging/' + directory + f'/shard_{index:04d}/rho.float64.npy',
                              base + '.controls.jsonl', base + '.manifest.json'))
        protected.append('state4_controls/' + directory + '/split_manifest.json')
    protected.append('state4_controls/rng_audit.json')
    originals = protected + ['state4_controls/read_only_evidence.json',
                             'state4_controls/pre_solve_manifest.json', 'STATE4_CONTROLS_COMPLETE.json']
    return State4Inventory(tuple(protected), tuple(sorted(originals)))


def _keys(value, names):
    _require(isinstance(value, Mapping) and set(value) == set(names.split()), 'object keys')


def _integer(value, minimum=0):
    _require(type(value) is int and value >= minimum, 'integer')


def _hash(value, length=64):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None
             and value != '0' * length, 'hash')


def _text(value):
    _require(type(value) is str and bool(value) and unicodedata.normalize('NFC', value) == value and
             not any(unicodedata.category(c) in ('Cc', 'Cs') for c in value), 'text')


def _path(value):
    _text(value)
    _require('\\' not in value and ':' not in value and not value.startswith('/'), 'relative path')
    for part in value.split('/'):
        _require(part not in ('', '.', '..') and not part.endswith((' ', '.')) and
                 not any(c in part for c in '<>"|?*'), 'path component')
        _require(part.split('.')[0].casefold() not in {'con', 'prn', 'aux', 'nul',
                 *('com'+i for i in '123456789¹²³'), *('lpt'+i for i in '123456789¹²³')}, 'Windows alias')


def _file(value):
    _keys(value, 'root_binding_id relative_path file_bytes file_sha256')
    _text(value['root_binding_id'])
    _require(':' not in value['root_binding_id'], 'root ID colon')
    _path(value['relative_path'])
    _integer(value['file_bytes'])
    _hash(value['file_sha256'])


def _roots(requirements):
    _require(type(requirements) is tuple and bool(requirements), 'root requirements')
    result = {}
    for row in requirements:
        _keys(row, 'root_binding_id repository_relative_root identity_file custody_file')
        _text(row['root_binding_id'])
        _require(':' not in row['root_binding_id'], 'root ID colon')
        _path(row['repository_relative_root'])
        _require(row['root_binding_id'].casefold() not in {key.casefold() for key in result} and
                 row['repository_relative_root'].casefold() not in {path.casefold() for path in result.values()}, 'duplicate root')
        _file(row['identity_file'])
        _file(row['custody_file'])
        result[row['root_binding_id']] = row['repository_relative_root']
    for row in requirements:
        _resolved(row['identity_file'], result)
        _resolved(row['custody_file'], result)
    return result


def _resolved(file, roots):
    _require(file['root_binding_id'] in roots, 'unknown root')
    _path(file['relative_path'])
    return roots[file['root_binding_id']] + '/' + file['relative_path']


def _capture(value):
    _keys(value, 'invocation_id clock_epoch_id first_sequence last_sequence started_utc finished_utc')
    _hash(value['invocation_id'])
    _text(value['clock_epoch_id'])
    _integer(value['first_sequence'])
    _integer(value['last_sequence'], value['first_sequence'])
    for name in ('started_utc', 'finished_utc'):
        _require(type(value[name]) is str and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z', value[name]), 'UTC')
        try:
            datetime.strptime(value[name], '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError as exc:
            raise State4AcceptanceInputError('UTC calendar') from exc
    _require(value['started_utc'] <= value['finished_utc'], 'capture order')


def _identity(value):
    _keys(value, 'volume_serial_number_hex file_id_128_hex legacy_volume_serial_number_hex legacy_file_index_hex')
    _require(type(value['volume_serial_number_hex']) is str and
             re.fullmatch(r'0x(0|[1-9a-f][0-9a-f]*)', value['volume_serial_number_hex']) is not None and
             int(value['volume_serial_number_hex'], 16) < 2**64, 'volume identity')
    for name, width in (('file_id_128_hex',32), ('legacy_volume_serial_number_hex',8), ('legacy_file_index_hex',16)):
        _require(type(value[name]) is str and re.fullmatch('[0-9a-f]{'+str(width)+'}', value[name]), 'identity field')


def _equal(left, right):
    return canonical_json_bytes(_plain(left)) == canonical_json_bytes(_plain(right))


def _diagnostic(role, pointer, code, reasons=()):
    _require(code in DIAGNOSTIC_CODES, 'internal diagnostic vocabulary')
    def safe(value):
        if (type(value) is str and re.fullmatch(r'[A-Za-z0-9_./-]+', value) and
                re.search(r'(?i)S-[0-9]+-[0-9]+', value) is None):
            return value
        return 'redacted/' + hashlib.sha256(str(value).encode('utf-8', 'surrogatepass')).hexdigest()
    role, pointer = safe(role), safe(pointer)
    return Diagnostic(role, pointer, code, tuple(sorted(set(reasons))))


def _diagnostics(rows):
    unique = {(r.role,r.pointer,r.code,r.component_reason_codes):r for r in rows}
    return tuple(unique[key] for key in sorted(unique))


def _component_reasons(error):
    """Only declared reason codes, never raw exception messages or paths."""
    values, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        for name in ('reason_code', 'reason_codes'):
            value = getattr(error, name, ())
            value = (value,) if type(value) is str else value
            if type(value) is tuple:
                values.extend(v for v in value if type(v) is str and re.fullmatch('[a-z][a-z0-9_]*', v))
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return tuple(sorted(set(values)))


def _protected_premises(expected, context, original, roots):
    _require(type(expected) is ExpectedProtectedFile and type(original) is ExpectedOriginalFile, 'protected input types')
    _integer(expected.row_index)
    inventory = derive_state4_acceptance_inventory()
    _require(expected.row_index < len(inventory.protected_paths), 'protected row index')
    _file(original.file)
    _identity(original.native_identity)
    _file(original.origin_evidence_file)
    _file(original.custody_file)
    _require(type(original.permission_evidence_files) is tuple, 'permission references')
    for file in original.permission_evidence_files:
        _file(file)
    path = inventory.protected_paths[expected.row_index]
    _require(original.file['relative_path'] == path, 'original path/index binding')
    plan = expected.expected_plan
    _require(type(plan) is handoff_records.HandoffPlan, 'plan type')
    try:
        handoff_records.validate_handoff_plan(plan)
    except Exception as exc:
        raise State4AcceptanceInputError('independent handoff plan rejected') from exc
    subject = plan.expected_request.subject
    _require(subject.row_index == expected.row_index and subject.relative_path == path and
             subject.repository_relative_path == _resolved(original.file, roots) and
             subject.expected_file_bytes == original.file['file_bytes'] and
             subject.expected_file_sha256 == original.file['file_sha256'], 'plan/original subject binding')
    _require(_equal(plan.expected_request.context, context), 'plan/context binding')
    _require(type(expected.expected_source_files) is tuple and expected.expected_source_files == plan.source_files,
             'independent source vector binding')
    sources = {item.relative_path:item for item in expected.expected_source_files}
    source_binding = plan.expected_request.source_binding
    for prefix in ('observer', 'evidence'):
        source = sources[getattr(source_binding, prefix + '_module_relative_path')]
        _require(source.file_sha256 == getattr(source_binding, prefix + '_module_file_sha256'), 'request/source-vector binding')
    _require(expected.accepted_anchor is None or type(expected.accepted_anchor) is handoff_records.AcceptedHandoffAnchor,
             'anchor type')
    if expected.accepted_anchor is not None:
        for field in fields(expected.accepted_anchor):
            value = getattr(expected.accepted_anchor, field.name)
            if field.name == 'receipt_file_bytes':
                _integer(value)
            else:
                _hash(value)
    try:
        storage.validate_state4_file_permission_metadata(_plain(expected.expected_metadata_row),
            expected_context=_plain(context), expected_index=expected.row_index,
            expected_file={k:original.file[k] for k in ('relative_path','file_bytes','file_sha256')})
    except Exception as exc:
        raise State4AcceptanceInputError('independent metadata row rejected') from exc
    return path


def _common_binding_disagreement(data, kind, plan, error):
    """Recognize a proven envelope disagreement, never substitute another plan."""
    expected = {'invocation_id':handoff_records.derive_handoff_invocation_id(plan),
        'plan_self_sha256':plan.plan_self_sha256,
        'request_self_sha256':plan.expected_request.request_self_sha256,
        'state4_authorization_file_sha256':plan.authority_binding.full_state4_authorization_file_sha256}
    if str(error) not in {'handoff '+kind+' '+name+' mismatch' for name in expected}:
        return False
    try:
        _require(type(data) is bytes and data.endswith(b'\n') and not data.endswith(b'\n\n'), 'witness encoding')
        record = strict_canonical_json_loads(data[:-1])
        _require(set(record) == set(handoff_records._RECORD_KEYS[kind]), 'witness envelope keys')
        _require(record['schema'] == handoff_records.RECORD_SCHEMAS[kind] and
                 record['permission_acceptance_emitted'] is False, 'witness envelope schema')
        _self_hash(record)
        for name in expected:
            _hash(record[name])
        return any(record[name] != value for name, value in expected.items())
    except (ValueError, TypeError, KeyError):
        return False


def verify_state4_protected_file(*, expected: ExpectedProtectedFile, metadata_context: Mapping,
                                original: ExpectedOriginalFile, root_requirements: tuple[Mapping, ...],
                                metadata_row: Mapping, handoff: HandoffBundle | None) -> ProtectedFileVerification:
    roots = _roots(root_requirements)
    path = _protected_premises(expected, metadata_context, original, roots)
    role = 'protected/' + str(expected.row_index)
    diagnostics = []
    invalid = mismatch = missing = False
    actual = None
    try:
        storage.validate_state4_file_permission_metadata(_plain(metadata_row),
            expected_context=_plain(metadata_context), expected_index=expected.row_index,
            expected_file={k:original.file[k] for k in ('relative_path','file_bytes','file_sha256')})
        if not _equal(metadata_row, expected.expected_metadata_row):
            mismatch = True
            diagnostics.append(_diagnostic(role, '/metadata_row', 'binding_mismatch'))
    except Exception as exc:
        try:
            candidate = _plain(metadata_row)
            storage.validate_state4_file_permission_metadata(candidate,
                expected_context=candidate['context'], expected_index=candidate['row_index'],
                expected_file={k:candidate[k] for k in ('relative_path','file_bytes','file_sha256')})
        except Exception:
            invalid = True
        else:
            mismatch = True
            diagnostics.append(_diagnostic(role, '/metadata_row', 'binding_mismatch'))
        diagnostics.append(_diagnostic(role, '/metadata_row', 'component_rejected', _component_reasons(exc)))
    plan = expected.expected_plan
    if handoff is None:
        missing = True
        diagnostics.append(_diagnostic(role, '/handoff', 'evidence_unavailable'))
    elif type(handoff) is not HandoffBundle or type(handoff.row_index) is not int:
        invalid = True
        diagnostics.append(_diagnostic(role, '/handoff', 'schema_invalid'))
    else:
        if handoff.row_index != expected.row_index:
            mismatch = True
            diagnostics.append(_diagnostic(role, '/handoff/row_index', 'binding_mismatch'))
        witness_bytes = {}
        for name, filename in zip(('start','outcome','acquisition','verification','index','receipt'), handoff_records.WITNESS_FILENAMES):
            row = getattr(handoff, name)
            try:
                _require(type(row) is EvidenceBytes, 'witness type')
                _file(row.file)
                _resolved(row.file, roots)
                if (row.role != name or row.file['root_binding_id'] != plan.witness_store.root_binding_id or
                    row.file['relative_path'] != plan.witness_store.parent_relative_path + '/' + filename):
                    mismatch = True
                    diagnostics.append(_diagnostic(role, '/handoff/'+name+'/file', 'binding_mismatch'))
                if row.data is None:
                    missing = True
                    diagnostics.append(_diagnostic(role, '/handoff/'+name, 'evidence_unavailable'))
                else:
                    _require(type(row.data) is bytes, 'immutable witness bytes')
                    if len(row.data) != row.file['file_bytes'] or hashlib.sha256(row.data).hexdigest() != row.file['file_sha256']:
                        mismatch = True
                        diagnostics.append(_diagnostic(role, '/handoff/'+name, 'hash_mismatch'))
                    else:
                        witness_bytes[name+'_bytes'] = row.data
            except (ValueError, TypeError, KeyError):
                invalid = True
                diagnostics.append(_diagnostic(role, '/handoff/'+name, 'schema_invalid'))
        if len(witness_bytes) == 6:
            anchor_mismatch = False
            if expected.accepted_anchor is not None:
                anchor_values = {
                    'specification_file_sha256':plan.specification_file_sha256,
                    'plan_self_sha256':plan.plan_self_sha256,
                    'invocation_id':handoff_records.derive_handoff_invocation_id(plan),
                    'receipt_file_bytes':len(witness_bytes['receipt_bytes']),
                    'receipt_file_sha256':hashlib.sha256(witness_bytes['receipt_bytes']).hexdigest(),
                    'handoff_source_vector_sha256':hashlib.sha256(canonical_json_bytes(_plain(plan.source_files))).hexdigest(),
                }
                anchor_mismatch = not _equal(expected.accepted_anchor, anchor_values)
                if anchor_mismatch:
                    mismatch = True
                    diagnostics.append(_diagnostic(role, '/handoff/anchor', 'binding_mismatch'))
            try:
                actual = handoff_verify.verify_anchored_permission_handoff(expected_plan=plan,
                    expected_source_files=expected.expected_source_files,
                    accepted_anchor=expected.accepted_anchor, **witness_bytes)
            except Exception as exc:
                # A proven anchor disagreement is a binding failure, not missing authority.
                # Independently validate the six record grammars before classifying this case.
                malformed = common_mismatch = False
                for name in ('start','outcome','acquisition','verification','index','receipt'):
                    try:
                        handoff_records.strict_handoff_record_loads(witness_bytes[name+'_bytes'], kind=name, expected_plan=plan)
                    except Exception as record_error:
                        if _common_binding_disagreement(witness_bytes[name+'_bytes'], name, plan, record_error):
                            common_mismatch = mismatch = True
                            diagnostics.append(_diagnostic(role, '/handoff/'+name+'/binding', 'binding_mismatch'))
                        else:
                            malformed = True
                invalid = invalid or malformed or not (anchor_mismatch or common_mismatch)
                diagnostics.append(_diagnostic(role, '/handoff', 'component_rejected', _component_reasons(exc)))
            else:
                content = actual.content_verification
                if actual.original_retention == 'confirmed_failure':
                    mismatch = True
                    diagnostics.append(_diagnostic(role, '/handoff', 'retention_failed', actual.reason_codes))
                if actual.anchor_status != 'accepted' or actual.original_retention != 'confirmed_success':
                    missing = True
                    diagnostics.append(_diagnostic(role, '/handoff/anchor', 'authority_unavailable', actual.reason_codes))
                if content is None:
                    missing = True
                    diagnostics.append(_diagnostic(role, '/handoff/content', 'evidence_unavailable', actual.reason_codes))
                else:
                    if (content.payload_kind != 'authz_evidence' or content.recorded_authz_status != 'completed' or
                        content.base_verification_status != 'verified_capture' or
                        content.authority_correspondence_status != 'corroborated' or content.assessment_status != 'completed'):
                        missing = True
                        diagnostics.append(_diagnostic(role, '/handoff/content', 'authority_unavailable', content.reason_codes))
                    assessment = content.assessment
                    if assessment is None:
                        missing = True
                        diagnostics.append(_diagnostic(role, '/assessment', 'evidence_unavailable', content.reason_codes))
                    else:
                        if assessment.supplied_checks_status != 'satisfied':
                            if assessment.supplied_checks_status == 'not_satisfied':
                                mismatch = True
                                code = 'permission_not_satisfied'
                            else:
                                missing = True
                                code = 'evidence_unavailable'
                            diagnostics.append(_diagnostic(role, '/assessment', code, assessment.reason_codes))
                        cleanup = (assessment.subject_authz_cleanup_failure_reasons +
                                   assessment.immediate_parent_authz_cleanup_failure_reasons)
                        if cleanup:
                            mismatch = True
                            diagnostics.append(_diagnostic(role, '/assessment/cleanup', 'cleanup_failed', cleanup))
                        base = assessment.recorded_access_check_assessment
                        expected_row = expected.expected_metadata_row
                        comparisons = [('/private_observation_commitment', base.private_observation_commitment, expected_row['private_observation_commitment'])]
                        permission = expected_row['permission_evidence']
                        for name in ('owning_principal_commitment','current_principal_commitment','security_descriptor_commitment'):
                            comparisons.append(('/permission_evidence/'+name, getattr(base, name), permission[name]))
                        comparisons.extend((('/permission_evidence/current_process_file_access_mask_hex', base.recorded_file_access_mask_hex, permission['current_process_file_access_mask_hex']),
                            ('/permission_evidence/current_process_parent_access_mask_hex', base.recorded_immediate_parent_access_mask_hex, permission['current_process_parent_access_mask_hex']),
                            ('/path_evidence/file_attributes_hex', base.captured_file_has_read_only_attribute, True)))
                        for pointer, observed, required in comparisons:
                            if not _equal(observed, required):
                                mismatch = True
                                diagnostics.append(_diagnostic(role, pointer, 'binding_mismatch'))
                if actual.overall_status != 'corroborated':
                    missing = True
                    diagnostics.append(_diagnostic(role, '/handoff/status', 'evidence_unavailable', actual.reason_codes))
                try:
                    acquisition = handoff_records.strict_handoff_record_loads(witness_bytes['acquisition_bytes'], kind='acquisition', expected_plan=plan)
                    for slot, read in zip(plan.input_slots, acquisition['ordered_reads'], strict=True):
                        for name in ('before','after'):
                            snapshot = read[name]
                            if snapshot is not None and (snapshot['root_binding_id'] != slot.root_binding_id or snapshot['relative_path'] != slot.relative_path):
                                mismatch = True
                                diagnostics.append(_diagnostic(role, '/acquisition/'+slot.slot_id+'/'+name, 'binding_mismatch'))
                        if read['cleanup_failures']:
                            mismatch = True
                            diagnostics.append(_diagnostic(role, '/acquisition/'+slot.slot_id+'/cleanup', 'cleanup_failed',
                                tuple(r['reason'] for r in read['cleanup_failures'])))
                        if (slot.role == 'authz_primary' and read['status'] == 'complete' and read['data'] is not None
                                and (content is None or content.assessment is None)):
                            # An incomplete Authz result cannot have a named assessment,
                            # but its independently read, strictly validated cleanup facts
                            # must not disappear with that absent assessment.
                            record = authz_records.strict_private_permission_authz_loads(
                                bytes.fromhex(read['data']['record_hex']), expected_request=plan.expected_request,
                                expected_access=plan.expected_access, expected_binding=plan.expected_authz_binding,
                                expected_destination_self_sha256=plan.expected_authz_destination.destination_self_sha256,
                                expected_store_expectation_self_sha256=plan.expected_store.expectation_self_sha256)
                            for position, result in enumerate(record['authz']['result_prefix']):
                                cleanup = tuple(row['reason'] for row in result['cleanup_failures'])
                                if cleanup:
                                    mismatch = True
                                    diagnostics.append(_diagnostic(role, '/authz/result_prefix/'+str(position)+'/cleanup',
                                                                   'cleanup_failed', cleanup))
                        if slot.role == 'base_primary' and read['status'] == 'complete' and read['data'] is not None:
                            raw = bytes.fromhex(read['data']['record_hex'])
                            header = strict_canonical_json_loads(raw)
                            if header.get('schema') == private_evidence.OBSERVATION_FAILURE_SCHEMA:
                                private_evidence.strict_private_permission_observation_failure_loads(raw,
                                    expected_request=plan.expected_request, expected_access=plan.expected_access)
                                missing = True
                                diagnostics.append(_diagnostic(role, '/observation', 'evidence_unavailable'))
                                continue
                            observation = private_evidence.strict_private_permission_observation_loads(
                                raw, expected_request=plan.expected_request, expected_access=plan.expected_access)
                            for name in ('subject_before','subject_after'):
                                if not _equal(observation[name]['facts']['identity'], original.native_identity):
                                    mismatch = True
                                    diagnostics.append(_diagnostic(role, '/observation/'+name+'/identity', 'identity_mismatch'))
                except Exception as exc:
                    invalid = True
                    diagnostics.append(_diagnostic(role, '/observation', 'component_rejected', _component_reasons(exc)))
    status = 'invalid' if invalid else 'not_satisfied' if mismatch else 'incomplete' if missing else 'corroborated'
    return ProtectedFileVerification(expected.row_index, path, actual, status, _diagnostics(diagnostics))


_LOCAL_RECORDS = {
    'read_only_evidence': ('state4_controls/read_only_evidence.json', 'state4_read_only_evidence.v1'),
    'pre_solve_manifest': ('state4_controls/pre_solve_manifest.json', 'state4_pre_solve_manifest.v2'),
    'rng_audit': ('state4_controls/rng_audit.json', 'state4_rng_audit.v1'),
    'controls_completion': ('STATE4_CONTROLS_COMPLETE.json', 'state4_controls_complete.v1'),
}
_COMMITMENT_FAMILIES = ('ordered_control_tensor_sha256', 'ordered_rho_storage_tensor_sha256',
                       'ordered_macro_coefficient_tensor_sha256', 'ordered_control_record_self_sha256')


def _sequence(value, nonempty=False):
    _require(type(value) in (list, tuple) and (not nonempty or bool(value)), 'array')


def _refs(value, roots, *, nonempty=False, sorted_required=False):
    _sequence(value, nonempty)
    addresses = []
    for row in value:
        _file(row)
        addresses.append(_resolved(row, roots).casefold())
    _require(len(set(addresses)) == len(addresses), 'duplicate or aliased file references')
    if sorted_required:
        encodings = [canonical_json_bytes(_plain(row)) for row in value]
        _require(encodings == sorted(encodings), 'reference order')


def _binding(value):
    _keys(value, 'benchmark state attempt_id protocol_file_sha256 implementation state4_authorization_file_sha256 authorization_file_sha256')
    _require(value['benchmark'] == 'darcy_branch_c_v1' and value['state'] == 'state4', 'benchmark/state')
    _require(value['protocol_file_sha256'] == PROTOCOL_SHA256, 'unsupported protocol')
    _keys(value['implementation'], 'commit tree sole_parent')
    for oid in value['implementation'].values():
        _hash(oid, 40)
    _hash(value['state4_authorization_file_sha256'])
    _require(value['authorization_file_sha256'] == value['state4_authorization_file_sha256'], 'full A4 binding')
    _require(value['attempt_id'] == 's4-' + value['state4_authorization_file_sha256'][:24], 'attempt binding')


def _local(row, path, schema=None):
    _keys(row, 'relative_path size sha256' + (' schema' if schema else ''))
    _path(row['relative_path'])
    _integer(row['size'], 1)
    _hash(row['sha256'])
    _require(row['relative_path'] == path, 'local path')
    if schema:
        _require(row['schema'] == PREFIX + schema, 'local schema')


def _shards(rows):
    _sequence(rows)
    _require(len(rows) == 76, 'control shard population')
    families = {name: [] for name in _COMMITMENT_FAMILIES}
    position = 0
    for split, count in SPLITS:
        directory = 'test.sealed' if split == 'test' else split
        for index in range(count):
            row = rows[position]
            position += 1
            _keys(row, 'split shard_index controls_manifest controls_jsonl rho rho_tensor_sha256 ' + ' '.join(_COMMITMENT_FAMILIES))
            _integer(row['shard_index'])
            _require(row['split'] == split and row['shard_index'] == index, 'shard order')
            base = 'state4_controls/' + directory + f'/shard_{index:04d}'
            _local(row['controls_manifest'], base + '.manifest.json', 'state4_controls_shard_manifest.v1')
            _local(row['controls_jsonl'], base + '.controls.jsonl')
            _local(row['rho'], 'corpus_staging/' + directory + f'/shard_{index:04d}/rho.float64.npy')
            _hash(row['rho_tensor_sha256'])
            for name in families:
                _sequence(row[name])
                _require(len(row[name]) == 64, 'sample commitment population')
                for child in row[name]:
                    _hash(child)
                families[name].extend(row[name])
    for children in families.values():
        _require(len(children) == 4864 and len(set(children)) == 4864, 'unique sample commitments')


def _self_hash(record):
    _require(isinstance(record, Mapping), 'record object')
    present = [name for name in ('record_self_sha256', 'manifest_self_sha256') if name in record]
    _require(len(present) == 1, 'record self hash field')
    field = present[0]
    _hash(record[field])
    payload = {k: v for k, v in record.items() if k != field}
    _require(hashlib.sha256(canonical_json_bytes(_plain(payload))).hexdigest() == record[field], 'record self hash')


def _record_specs():
    specs = dict(_LOCAL_RECORDS)
    for split, count in SPLITS:
        directory = 'test.sealed' if split == 'test' else split
        for index in range(count):
            specs[f'control/{split}/shard_{index:04d}'] = (
                f'state4_controls/{directory}/shard_{index:04d}.manifest.json', 'state4_controls_shard_manifest.v1')
    return specs


def _admission_shape(row, roots, admission=False):
    _require(type(row) is (AcceptanceAdmission if admission else AcceptanceRequirement), 'admission contract type')
    _text(row.role)
    _require(row.scope in SCOPES, 'admission scope')
    _refs(row.input_files, roots, nonempty=True)
    _refs(row.validator_source_files, roots, nonempty=True)
    _capture(row.capture)
    if admission:
        _file(row.admission_file)
        _resolved(row.admission_file, roots)
        _require(type(row.status) is str and row.status in ('accepted', 'rejected', 'unavailable'), 'admission status')


def _validate_expectations(expected):
    """Validate independent premises, without interpreting candidate approval flags."""
    _require(type(expected) is State4AcceptanceExpectations, 'expectations type')
    _binding(expected.binding)
    _capture(expected.capture)
    a4 = expected.binding['state4_authorization_file_sha256']
    invocation = hashlib.sha256(('operatormark-state-evidence-invocation-v1\nstate4\n' + a4 + '\n' + a4).encode()).hexdigest()
    _require(expected.capture['invocation_id'] == invocation, 'capture invocation')
    roots = _roots(expected.root_requirements)
    _require(expected.attempt_root_binding_id in roots, 'attempt root')
    context = expected.metadata_context
    _keys(context, 'protocol_file_sha256 state4_authorization_file_sha256 implementation_commit implementation_tree storage_preflight_file_sha256 permission_profile_file_sha256 repository_root_commitment volume_identifier_commitment accepted_local_user_residual_principal_commitments')
    for name in ('protocol_file_sha256', 'state4_authorization_file_sha256'):
        _require(context[name] == expected.binding[name], 'metadata/full binding')
    for name in ('commit', 'tree'):
        _require(context['implementation_' + name] == expected.binding['implementation'][name], 'metadata/source binding')
    inventory = derive_state4_acceptance_inventory()
    _require(type(expected.original_files) is tuple and len(expected.original_files) == 235, 'original population')
    identities, physical_ids, paths = [], [], []
    for original in expected.original_files:
        _require(type(original) is ExpectedOriginalFile, 'original type')
        _file(original.file)
        _require(original.file['root_binding_id'] == expected.attempt_root_binding_id, 'original attempt root')
        _identity(original.native_identity)
        paths.append(original.file['relative_path'])
        identities.append(canonical_json_bytes(_plain(original.native_identity)))
        physical_ids.append((original.native_identity['volume_serial_number_hex'], original.native_identity['file_id_128_hex']))
        for ref in (original.origin_evidence_file, original.custody_file):
            _file(ref)
            _resolved(ref, roots)
        _refs(original.permission_evidence_files, roots, sorted_required=True)
    _require(tuple(paths) == inventory.original_paths, 'original inventory order')
    _require(len(set(identities)) == 235 and len(set(physical_ids)) == 235, 'duplicate subject identity')
    by_path = {row.file['relative_path']: row for row in expected.original_files}
    _require(type(expected.protected_files) is tuple and len(expected.protected_files) == 232, 'protected population')
    requests, invocations, commitments = [], [], []
    for index, row in enumerate(expected.protected_files):
        _require(type(row) is ExpectedProtectedFile and type(row.row_index) is int and row.row_index == index, 'protected order')
        _protected_premises(row, context, by_path[inventory.protected_paths[index]], roots)
        plan = row.expected_plan
        requests.append(plan.expected_request.request_self_sha256)
        invocations.append(handoff_records.derive_handoff_invocation_id(plan))
        commitments.append(row.expected_metadata_row['path_evidence']['file_identity_commitment'])
        for slot in plan.input_slots:
            _require(slot.root_binding_id in roots, 'input slot root')
        _require(plan.witness_store.root_binding_id in roots, 'witness root')
    _require(all(len(set(values)) == 232 for values in (requests, invocations, commitments)), 'duplicate protected binding')
    _shards(expected.control_shards)
    for shard in expected.control_shards:
        for name in ('rho', 'controls_jsonl', 'controls_manifest'):
            local = shard[name]
            ref = by_path[local['relative_path']].file
            _require((local['size'], local['sha256']) == (ref['file_bytes'], ref['file_sha256']), 'independent shard/original equation')
    specs = _record_specs()
    _require(type(expected.expected_records) is tuple and len(expected.expected_records) == 80, 'expected record population')
    records = {}
    for row in expected.expected_records:
        _require(type(row) is ExpectedRecord and row.role in specs and row.role not in records, 'expected record role')
        _file(row.file)
        path, schema = specs[row.role]
        _require(row.file['root_binding_id'] == expected.attempt_root_binding_id and row.file['relative_path'] == path, 'expected record address')
        _require(row.schema == PREFIX + schema and row.record.get('schema') == row.schema, 'expected record schema')
        _self_hash(row.record)
        _record_shape(row.record, row.role, expected)
        data = canonical_json_bytes(_plain(row.record))
        _require(row.file['file_bytes'] == len(data) and row.file['file_sha256'] == hashlib.sha256(data).hexdigest(), 'expected record bytes')
        _require(_equal(row.file, by_path[path].file), 'expected record/original binding')
        records[row.role] = row
    _require(set(records) == set(specs), 'expected record roles')
    readonly = records['read_only_evidence'].record
    _require(_equal(readonly.get('ordered_relative_paths'), inventory.protected_paths) and
             _equal(readonly.get('ordered_file_sha256'), [by_path[p].file['file_sha256'] for p in inventory.protected_paths]) and
             _equal(readonly.get('ordered_permission_evidence'), [r.expected_metadata_row for r in expected.protected_files]),
             'independent read-only/table correspondence')
    _require(type(expected.record_bindings) is tuple, 'record bindings')
    bindings, addresses, source_identities = {}, {}, {}
    for row in expected.record_bindings:
        _keys(row, 'role file')
        _text(row['role'])
        _file(row['file'])
        _require(not row['file']['relative_path'].endswith(('.npy', '.controls.jsonl', '/split_manifest.json')), 'opaque original payload binding prohibited')
        address = _resolved(row['file'], roots)
        _require(row['role'] not in bindings and address.casefold() not in addresses, 'duplicate record binding')
        bindings[row['role']] = row['file']
        addresses[address.casefold()] = row['file']
        source_identities[(address, row['file']['file_bytes'], row['file']['file_sha256'])] = row['file']
    _require(set(specs) | {'accepted_controls', 'state3_audit', 'compute_feasibility_report'} <= set(bindings), 'required record bindings')
    for role, row in records.items():
        _require(_equal(row.file, bindings[role]), 'expected record binding')
    accepted_path = ('artifacts/darcy_branch_c_v1/private_corpus_v1/state_evidence/' +
                     expected.binding['attempt_id'] + '/state4/test.sealed/accepted_controls.json')
    _require(_resolved(bindings['accepted_controls'], roots) == accepted_path, 'accepted-controls address')
    for role in ('state3_audit', 'compute_feasibility_report', 'accepted_controls'):
        address = _resolved(bindings[role], roots)
        _require(not address.casefold().startswith(roots[expected.attempt_root_binding_id].casefold() + '/'), 'external original-tree alias')
    for kind, rows in ((False, expected.requirements), (True, expected.admissions)):
        _require(type(rows) is tuple and (kind or bool(rows)), 'requirements/admissions table')
        roles = set()
        for row in rows:
            _admission_shape(row, roots, kind)
            _require(row.role not in roles, 'duplicate admission role')
            roles.add(row.role)
    _require({row.scope for row in expected.requirements} == set(SCOPES), 'required scope coverage')
    _require({row.role for row in expected.admissions} <= {row.role for row in expected.requirements}, 'unexpected admission role')
    # Population membership is independently fixed here, not inferred from a candidate's label.
    populations = {
        'control_capture': [r.file for r in expected.original_files if r.file['relative_path'] in inventory.protected_paths[:-1]],
        'origin_custody': [r.file for r in expected.original_files],
        'broad_principals': [by_path[p].file for p in inventory.protected_paths],
        'ancestor_policy': [by_path[p].file for p in inventory.protected_paths],
        'post_write_protection': [records[r].file for r in ('pre_solve_manifest', 'controls_completion')],
        'guard_process_closure': [records['rng_audit'].file],
        'source_authority': [bindings[r] for r in ('state3_audit', 'compute_feasibility_report')],
    }
    for root in expected.root_requirements:
        populations['source_authority'].extend((root['identity_file'], root['custody_file']))
    for row in expected.protected_files:
        plan = row.expected_plan
        for source in row.expected_source_files:
            match = source_identities.get((source.relative_path, source.file_bytes, source.file_sha256))
            _require(match is not None, 'independently addressed source vector')
            populations['source_authority'].append(match)
        for grant in plan.expected_grants.values():
            _require(grant.accepted_state3_audit_file_sha256 == bindings['state3_audit']['file_sha256'] and
                     grant.accepted_compute_feasibility_report_file_sha256 == bindings['compute_feasibility_report']['file_sha256'],
                     'grant predecessor binding')
        for name in ('controlling_authorization_file_ref', 'enrollment_file_ref', 'profile_file_ref', 'accepted_roster_file_ref'):
            populations['source_authority'].append(_plain(getattr(plan.authority_binding, name)))
        populations['source_authority'].extend(_plain(ref) for ref in handoff_records._expected_predecessor_refs(plan))
        if row.accepted_anchor is not None:
            populations['source_authority'].append({
                'root_binding_id':plan.witness_store.root_binding_id,
                'relative_path':plan.witness_store.parent_relative_path+'/'+handoff_records.WITNESS_FILENAMES[-1],
                'file_bytes':row.accepted_anchor.receipt_file_bytes, 'file_sha256':row.accepted_anchor.receipt_file_sha256})
    for original in expected.original_files:
        populations['origin_custody'].extend((original.origin_evidence_file, original.custody_file))
    for scope, required in populations.items():
        supplied = {canonical_json_bytes(_plain(ref)) for row in expected.requirements if row.scope == scope for ref in row.input_files}
        _require({canonical_json_bytes(_plain(ref)) for ref in required} <= supplied, 'requirement population coverage')
    graph_findings = _Findings()
    _graph({role:row.record for role,row in records.items()}, None, expected, graph_findings)
    _require(not graph_findings.mismatch, 'independent record graph')


def validate_state4_acceptance_expectations(expected: State4AcceptanceExpectations) -> None:
    try:
        _validate_expectations(expected)
    except State4AcceptanceInputError:
        raise
    except Exception as exc:
        raise State4AcceptanceInputError('independent expectations rejected') from exc


_RECORD_KEYS = {
    'read_only_evidence': 'schema attempt_id permission_mechanism ordered_relative_paths ordered_file_sha256 ordered_permission_evidence all_required_files_read_only record_self_sha256',
    'rng_audit': 'schema benchmark state attempt_id protocol_file_sha256 state4_authorization_file_sha256 implementation_commit implementation_tree frozen_base_import_closure_sha256 rng_state_commitments rng_state_equalities_to_after_base_import authorized_corpus_seed_access_count_by_split authorized_pcg64_constructor_count_by_split authorized_generator_constructor_count_by_split authorized_top_level_rng_api_call_count_by_split forbidden_rng_call_count forbidden_or_out_of_scope_corpus_seed_access_count all_rng_checks_passed record_self_sha256',
    'pre_solve_manifest': 'schema benchmark corpus_version state attempt_id protocol_relative_name protocol_file_sha256 state4_authorization_file_sha256 solver_spec_baseline_commit solver_spec_baseline_tree implementation_commit implementation_tree state3_audit_file_sha256 compute_feasibility_report_file_sha256 environment_record_sha256 storage_preflight_before_file_sha256 storage_preflight_after_file_sha256 rng_audit_relative_path rng_audit_file_sha256 read_only_evidence_relative_path read_only_evidence_file_sha256 split_order split_manifest_paths split_manifest_file_sha256 sample_count shard_count global_control_aggregate_sha256 global_macro_coefficient_aggregate_sha256 all_control_hashes_unique all_macro_coefficient_hashes_unique rng_call_count_by_split forbidden_seed_or_data_access_count git_before git_after manifest_self_sha256',
    'controls_completion': 'schema state attempt_id pre_solve_manifest_relative_path pre_solve_manifest_file_sha256 read_only_evidence_file_sha256 state4_authorization_file_sha256 completion_time_utc record_self_sha256',
    'control': 'schema benchmark split sealed shard_index first_local_index last_local_index first_global_index last_global_index sample_count rho_relative_path rho_dtype rho_shape rho_order npy_version rho_file_bytes rho_file_sha256 rho_tensor_sha256 control_records_relative_path control_records_bytes control_records_file_sha256 ordered_record_self_sha256 protocol_file_sha256 state4_authorization_file_sha256 implementation_commit implementation_tree environment_record_sha256 manifest_self_sha256',
}
_PROJECTION_KEYS = 'schema binding capture state3_audit compute_feasibility_report pre_solve_manifest controls_completion rng_audit read_only_evidence ordered_split_manifests ordered_control_shards original_file_inventory acceptance_evidence_files record_self_sha256'


def _record_shape(record, role, expected):
    """Independent, seed-free State 4 record grammar. No producer imports."""
    kind = 'control' if role.startswith('control/') else role
    _keys(record, _RECORD_KEYS[kind])
    _self_hash(record)
    _require(record['schema'] == PREFIX + _record_specs()[role][1], 'record schema')
    for name, value in record.items():
        if name.endswith('_sha256'):
            if type(value) is str:
                _hash(value)
    for name in ('benchmark', 'attempt_id', 'protocol_file_sha256', 'state4_authorization_file_sha256'):
        if name in record:
            _require(record[name] == expected.binding[name], 'record stage binding')
    for name in ('commit', 'tree'):
        if 'implementation_' + name in record:
            _require(record['implementation_' + name] == expected.binding['implementation'][name], 'record source binding')
    if kind == 'read_only_evidence':
        _require(record['permission_mechanism'] == storage.STATE4_FILE_PERMISSION_METADATA_PROFILE, 'permission profile')
        _require(record['all_required_files_read_only'] is True, 'read-only assertion')
        inventory = derive_state4_acceptance_inventory()
        _require(_equal(record['ordered_relative_paths'], inventory.protected_paths), 'read-only inventory')
        for name in ('ordered_file_sha256', 'ordered_permission_evidence'):
            _sequence(record[name])
            _require(len(record[name]) == 232, 'read-only row count')
        for digest in record['ordered_file_sha256']:
            _hash(digest)
    elif kind == 'rng_audit':
        _require(record['state'] == 'state4_controls_rng_audit_complete', 'RNG state')
        checkpoints = ('after_base_import', 'after_guard_install', 'after_project_import', 'after_training',
                       'after_validation', 'after_test', 'after_execution')
        _keys(record['rng_state_commitments'], ' '.join(checkpoints))
        _keys(record['rng_state_equalities_to_after_base_import'], ' '.join(checkpoints[1:]))
        baseline = record['rng_state_commitments']['after_base_import']
        for checkpoint in checkpoints:
            row = record['rng_state_commitments'][checkpoint]
            _keys(row, 'numpy_legacy_sha256 python_random_sha256')
            for digest in row.values():
                _hash(digest)
            if checkpoint != 'after_base_import':
                equalities = record['rng_state_equalities_to_after_base_import'][checkpoint]
                _keys(equalities, 'numpy_legacy_equals_baseline python_random_equals_baseline')
                for family in ('numpy_legacy', 'python_random'):
                    _require(equalities[family + '_equals_baseline'] is True and row[family + '_sha256'] == baseline[family + '_sha256'], 'RNG equality')
        for name in ('authorized_corpus_seed_access_count_by_split', 'authorized_pcg64_constructor_count_by_split',
                     'authorized_generator_constructor_count_by_split'):
            _require(_equal(record[name], {'training': 1, 'validation': 1, 'test': 1}), 'authorized count')
        _require(_equal(record['authorized_top_level_rng_api_call_count_by_split'], {'training':4096,'validation':256,'test':512}), 'RNG API counts')
        for name in ('forbidden_rng_call_count', 'forbidden_or_out_of_scope_corpus_seed_access_count'):
            _integer(record[name])
            _require(record[name] == 0, 'forbidden count')
        _require(record['all_rng_checks_passed'] is True, 'RNG assertion')
    elif kind == 'control':
        _, split, shard = role.split('/')
        index = int(shard.removeprefix('shard_'))
        directory = 'test.sealed' if split == 'test' else split
        offset = {'training':0, 'validation':4096, 'test':4352}[split]
        required = {'split':split, 'sealed':split == 'test', 'shard_index':index,
                    'first_local_index':index * 64, 'last_local_index':index * 64 + 63,
                    'first_global_index':offset + index * 64, 'last_global_index':offset + index * 64 + 63,
                    'sample_count':64, 'rho_dtype':'<f8', 'rho_shape':[64,1,8,8], 'rho_order':'C', 'npy_version':'2.0',
                    'rho_relative_path':f'corpus_staging/{directory}/shard_{index:04d}/rho.float64.npy',
                    'control_records_relative_path':f'state4_controls/{directory}/shard_{index:04d}.controls.jsonl'}
        for name, value in required.items():
            _require(_equal(record[name], value), 'shard field')
        for name in ('rho_file_bytes', 'control_records_bytes'):
            _integer(record[name], 1)
        _sequence(record['ordered_record_self_sha256'])
        _require(len(record['ordered_record_self_sha256']) == 64, 'shard records')
        for digest in record['ordered_record_self_sha256']:
            _hash(digest)
        _require(len(set(record['ordered_record_self_sha256'])) == 64, 'shard duplicate record')
    elif kind == 'pre_solve_manifest':
        required = {'corpus_version':'private_corpus_v1', 'state':'controls_prepared_no_pde_solve',
                    'protocol_relative_name':'OperatorMark_Private_Corpus_PreGeneration_Frozen_Protocol_v1.0.md',
                    'solver_spec_baseline_commit':'e16456395cfe67b73b9d298fd0491cf9524d798a',
                    'solver_spec_baseline_tree':'81f40e24102e51e0da3ce65dafd769a7678565fd',
                    'rng_audit_relative_path':_LOCAL_RECORDS['rng_audit'][0],
                    'read_only_evidence_relative_path':_LOCAL_RECORDS['read_only_evidence'][0],
                    'split_order':['training','validation','test'], 'sample_count':4864, 'shard_count':76,
                    'all_control_hashes_unique':True, 'all_macro_coefficient_hashes_unique':True,
                    'rng_call_count_by_split':{'training':4096,'validation':256,'test':512},
                    'forbidden_seed_or_data_access_count':0,
                    'split_manifest_paths':['state4_controls/'+d+'/split_manifest.json' for d in ('training','validation','test.sealed')]}
        for name, value in required.items():
            _require(_equal(record[name], value), 'pre-solve field')
        _sequence(record['split_manifest_file_sha256'])
        _require(len(record['split_manifest_file_sha256']) == 3, 'split hashes')
        for digest in record['split_manifest_file_sha256']:
            _hash(digest)
        for name in ('git_before', 'git_after'):
            git = record[name]
            _keys(git, 'branch head tree parents status_short_untracked_all tracked_diff_sha256 tracked_diff_size_bytes tracked_changes_empty staged_diff_sha256 staged_diff_size_bytes staged_changes_empty untracked_inclusive_status_empty')
            _require(git['branch'] == 'main' and git['head'] == expected.binding['implementation']['commit'] and
                     git['tree'] == expected.binding['implementation']['tree'] and
                     _equal(git['parents'], [expected.binding['implementation']['sole_parent']]), 'Git source binding')
            _require(git['status_short_untracked_all'] == '' and git['untracked_inclusive_status_empty'] is True, 'Git status')
            for prefix in ('tracked', 'staged'):
                _require(git[prefix+'_diff_sha256'] == hashlib.sha256(b'').hexdigest() and
                         type(git[prefix+'_diff_size_bytes']) is int and git[prefix+'_diff_size_bytes'] == 0 and
                         git[prefix+'_changes_empty'] is True, 'Git difference')
        _require(_equal(record['git_before'], record['git_after']), 'Git closure')
    else:
        _require(record['state'] == 'state4_controls_accepted_no_solutions' and
                 record['pre_solve_manifest_relative_path'] == _LOCAL_RECORDS['pre_solve_manifest'][0], 'completion state')
        value = record['completion_time_utc']
        _require(type(value) is str and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z', value), 'completion UTC')
        try:
            datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError as exc:
            raise State4AcceptanceInputError('completion UTC calendar') from exc


def _projection_shape(record, roots):
    _keys(record, _PROJECTION_KEYS)
    _self_hash(record)
    _require(record['schema'] == PREFIX + 'accepted_controls_projection.v1', 'projection schema')
    _binding(record['binding'])
    _capture(record['capture'])
    for role in ('state3_audit', 'compute_feasibility_report'):
        ref = record[role]
        _keys(ref, 'filename literal_path_sha256 sha256 size')
        _path(ref['filename'])
        _require('/' not in ref['filename'], 'external basename')
        _hash(ref['literal_path_sha256'])
        _hash(ref['sha256'])
        _integer(ref['size'], 1)
    for role, (path, schema) in _LOCAL_RECORDS.items():
        _local(record[role], path, schema)
    _sequence(record['ordered_split_manifests'])
    _require(len(record['ordered_split_manifests']) == 3, 'opaque split population')
    for directory, row in zip(('training','validation','test.sealed'), record['ordered_split_manifests']):
        _local(row, 'state4_controls/' + directory + '/split_manifest.json')
    _shards(record['ordered_control_shards'])
    _sequence(record['original_file_inventory'])
    _require(len(record['original_file_inventory']) == 235, 'projection original population')
    paths, native = [], []
    for row in record['original_file_inventory']:
        _keys(row, 'file native_identity origin_evidence_file custody_file permission_evidence_files')
        for name in ('file', 'origin_evidence_file', 'custody_file'):
            _file(row[name])
            _resolved(row[name], roots)
        _identity(row['native_identity'])
        _refs(row['permission_evidence_files'], roots, sorted_required=True)
        paths.append(row['file']['relative_path'])
        native.append((row['native_identity']['volume_serial_number_hex'], row['native_identity']['file_id_128_hex']))
    _require(tuple(paths) == derive_state4_acceptance_inventory().original_paths and len(set(native)) == 235, 'projection original inventory')
    _refs(record['acceptance_evidence_files'], roots, nonempty=True, sorted_required=True)


class _Findings:
    def __init__(self):
        self.rows = []
        self.invalid = self.mismatch = self.incomplete = False

    def add(self, category, role, pointer, code, reasons=()):
        setattr(self, category, True)
        self.rows.append(_diagnostic(role, pointer, code, reasons))

    def compare(self, left, right, role, pointer, code='binding_mismatch'):
        if not _equal(left, right):
            self.add('mismatch', role, pointer, code)
            return False
        return True


def _index_evidence(rows, bindings, roots, expected, findings, side):
    indexed, records, addresses, duplicates = {}, {}, {}, set()
    if type(rows) is not tuple:
        findings.add('invalid', side, '/', 'schema_invalid')
        return indexed, records
    specs = _record_specs()
    for position, row in enumerate(rows):
        pointer = '/' + str(position)
        role = side
        try:
            _require(type(row) is EvidenceBytes, 'evidence type')
            _text(row.role)
            role = row.role if row.role in bindings else side
            _require(row.role in bindings, 'unexpected evidence role')
            _file(row.file)
            address = _resolved(row.file, roots).casefold()
            if row.role in indexed or row.role in duplicates or address in addresses:
                findings.add('invalid', role, pointer, 'duplicate_or_alias')
                duplicates.add(row.role)
                if address in addresses:
                    duplicates.add(addresses[address])
                continue
            addresses[address] = row.role
            # Reserve the role even when unavailable or malformed; duplicates are never paired.
            indexed[row.role] = None
            bound = findings.compare(row.file, bindings[row.role], role, '/' + side + '/file')
            if row.data is None:
                findings.add('incomplete', role, '/' + side, 'evidence_unavailable')
                continue
            _require(type(row.data) is bytes, 'immutable evidence bytes')
            if len(row.data) != row.file['file_bytes'] or hashlib.sha256(row.data).hexdigest() != row.file['file_sha256']:
                findings.add('mismatch', role, '/' + side, 'hash_mismatch')
                continue
            if role in specs or role == 'accepted_controls':
                record = strict_canonical_json_loads(row.data)
                if role in specs:
                    _record_shape(record, role, expected)
                else:
                    _projection_shape(record, roots)
                if bound:
                    records[role] = record
            if bound:
                indexed[role] = row
        except (ValueError, TypeError, KeyError, UnicodeError, AttributeError):
            findings.add('invalid', role, pointer, 'schema_invalid')
    for role in duplicates:
        indexed.pop(role, None)
        records.pop(role, None)
    return indexed, records


def _covers(observed, required):
    return (observed['invocation_id'] == required['invocation_id'] and
            observed['clock_epoch_id'] == required['clock_epoch_id'] and
            observed['first_sequence'] <= required['first_sequence'] and
            observed['last_sequence'] >= required['last_sequence'] and
            observed['started_utc'] <= required['started_utc'] and
            observed['finished_utc'] >= required['finished_utc'])


def _admissions(expected, independent, roots, findings):
    """Match trusted admissions; bytes never promote their review status."""
    available = {canonical_json_bytes(_plain(row.file)) for row in independent.values() if row is not None}
    originals = {canonical_json_bytes(_plain(row.file)):row for row in expected.original_files}
    opaque = {key for key, row in originals.items() if row.file['relative_path'].endswith(('.npy', '.controls.jsonl', '/split_manifest.json'))}
    admissions = {row.role:row for row in expected.admissions}
    scope_codes = {'source_authority':'authority_unavailable', 'control_capture':'evidence_unavailable',
        'origin_custody':'custody_unavailable', 'broad_principals':'policy_coverage_unavailable',
        'ancestor_policy':'policy_coverage_unavailable', 'post_write_protection':'protection_unavailable',
        'guard_process_closure':'chronology_unavailable'}
    qualified, admitted_refs = set(), []
    source_rows = {_resolved(row.file, roots):row for row in independent.values() if row is not None}
    for index, protected in enumerate(expected.protected_files):
        source_binding = protected.expected_plan.expected_request.source_binding
        for prefix in ('observer', 'evidence'):
            source = source_rows.get(getattr(source_binding, prefix+'_module_relative_path'))
            if source is not None:
                blob = hashlib.sha1(b'blob '+str(len(source.data)).encode('ascii')+b'\0'+source.data).hexdigest()
                findings.compare(blob, getattr(source_binding, prefix+'_module_git_blob_sha1'),
                                 'protected/'+str(index), '/source_binding/'+prefix+'_module_git_blob_sha1')
    for requirement in expected.requirements:
        role, scope = requirement.role, requirement.scope
        admission = admissions.get(role)
        if admission is None:
            findings.add('incomplete', role, '/admission', scope_codes[scope])
            continue
        admitted_refs.append(admission.admission_file)
        okay = True
        for name in ('scope', 'input_files', 'validator_source_files'):
            if not findings.compare(getattr(admission, name), getattr(requirement, name), role, '/admission/' + name):
                okay = False
        if not _covers(admission.capture, requirement.capture):
            findings.add('incomplete', role, '/admission/capture', 'chronology_unavailable')
            okay = False
        if scope == 'guard_process_closure' and not _covers(requirement.capture, expected.capture):
            findings.add('incomplete', role, '/requirement/capture', 'chronology_unavailable')
            okay = False
        if scope in ('control_capture', 'origin_custody', 'post_write_protection', 'guard_process_closure') and (
                requirement.capture['invocation_id'] != expected.capture['invocation_id'] or
                requirement.capture['clock_epoch_id'] != expected.capture['clock_epoch_id']):
            findings.add('incomplete', role, '/requirement/capture', 'chronology_unavailable')
            okay = False
        if admission.status == 'rejected':
            findings.add('mismatch', role, '/admission/status', 'upstream_rejected')
            okay = False
        elif admission.status == 'unavailable':
            findings.add('incomplete', role, '/admission/status', scope_codes[scope])
            okay = False
        # Only opaque ORIGINAL payloads are exempt. Source, admission and support bytes are not.
        for name, refs in (('admission_file', (admission.admission_file,)),
                           ('validator_source_files', requirement.validator_source_files),
                           ('input_files', requirement.input_files)):
            for index, ref in enumerate(refs):
                key = canonical_json_bytes(_plain(ref))
                if key not in available and not (name == 'input_files' and key in opaque):
                    findings.add('incomplete', role, '/'+name+'/'+str(index), scope_codes[scope])
                    okay = False
        if okay:
            qualified.update(canonical_json_bytes(_plain(ref)) for ref in requirement.input_files)
    for index, original in enumerate(expected.original_files):
        for name, refs in (('origin_evidence_file', (original.origin_evidence_file,)),
                           ('custody_file', (original.custody_file,)),
                           ('permission_evidence_files', original.permission_evidence_files)):
            for position, ref in enumerate(refs):
                key = canonical_json_bytes(_plain(ref))
                if key not in available and not (key in opaque and key in qualified):
                    findings.add('incomplete', 'original/' + str(index), '/'+name+'/'+str(position),
                                 'custody_unavailable' if name != 'permission_evidence_files' else 'protection_unavailable')
        key = canonical_json_bytes(_plain(original.file))
        if key in opaque and key not in qualified:
            findings.add('incomplete', 'original/' + str(index), '/file', 'evidence_unavailable')
    unique = {canonical_json_bytes(_plain(ref)): ref for ref in admitted_refs}
    return tuple(unique[key] for key in sorted(unique)), all(row.role in admissions for row in expected.requirements)


def _digest(domain, children):
    return hashlib.sha256(domain.encode('ascii') + b''.join(bytes.fromhex(child) for child in children)).hexdigest()


def _graph(records, projection, expected, findings):
    by_path = {row.file['relative_path']:row.file for row in expected.original_files}
    bindings = {row['role']:row['file'] for row in expected.record_bindings}
    for role, record in records.items():
        if role not in _record_specs():
            continue
        if role.startswith('control/'):
            _, split, shard = role.split('/')
            index = int(shard.removeprefix('shard_'))
            summary = next(row for row in expected.control_shards if row['split'] == split and row['shard_index'] == index)
            for key, local_name in (('rho','rho'), ('controls_jsonl','control_records')):
                local = summary[key]
                for target, suffix in (('relative_path','relative_path'), ('size','file_bytes' if key == 'rho' else 'bytes'),
                                       ('sha256','file_sha256')):
                    findings.compare(record[local_name+'_'+suffix], local[target], role, '/'+local_name+'_'+suffix)
            findings.compare(record['rho_tensor_sha256'], summary['rho_tensor_sha256'], role, '/rho_tensor_sha256')
            findings.compare(record['ordered_record_self_sha256'], summary['ordered_control_record_self_sha256'], role, '/ordered_record_self_sha256')
    pre = records.get('pre_solve_manifest')
    if pre is not None:
        for role in ('state3_audit', 'compute_feasibility_report', 'rng_audit', 'read_only_evidence'):
            findings.compare(pre[role+'_file_sha256'], bindings[role]['file_sha256'], 'pre_solve_manifest', '/'+role+'_file_sha256')
        split_hashes = [by_path[path]['file_sha256'] for path in pre['split_manifest_paths']]
        findings.compare(pre['split_manifest_file_sha256'], split_hashes, 'pre_solve_manifest', '/split_manifest_file_sha256')
        for family, key, suffix in (('ordered_control_tensor_sha256','global_control_aggregate_sha256','control'),
                                     ('ordered_macro_coefficient_tensor_sha256','global_macro_coefficient_aggregate_sha256','macro-coefficient')):
            children = [_digest('operatormark-state4-split-'+suffix+'-aggregate-v1\n',
                        [digest for row in expected.control_shards if row['split'] == split for digest in row[family]]) for split, _ in SPLITS]
            recomputed = _digest('operatormark-state4-global-'+suffix+'-aggregate-v1\n', children)
            findings.compare(pre[key], recomputed, 'pre_solve_manifest', '/'+key)
    complete = records.get('controls_completion')
    if complete is not None:
        for role in ('pre_solve_manifest', 'read_only_evidence'):
            findings.compare(complete[role+'_file_sha256'], bindings[role]['file_sha256'], 'controls_completion', '/'+role+'_file_sha256')
    if projection is not None:
        for role in ('state3_audit', 'compute_feasibility_report'):
            findings.compare((projection[role]['size'], projection[role]['sha256']),
                             (bindings[role]['file_bytes'], bindings[role]['file_sha256']), 'accepted_controls', '/'+role)
        for role in _LOCAL_RECORDS:
            local = projection[role]
            findings.compare((local['relative_path'], local['size'], local['sha256']),
                (bindings[role]['relative_path'], bindings[role]['file_bytes'], bindings[role]['file_sha256']), 'accepted_controls', '/'+role)
        for local in projection['ordered_split_manifests']:
            original = by_path[local['relative_path']]
            findings.compare((local['size'], local['sha256']), (original['file_bytes'], original['file_sha256']),
                             'accepted_controls', '/ordered_split_manifests/'+local['relative_path'])
        for index, summary in enumerate(projection['ordered_control_shards']):
            for name in ('rho', 'controls_jsonl', 'controls_manifest'):
                local = summary[name]
                original = by_path[local['relative_path']]
                findings.compare((local['size'], local['sha256']), (original['file_bytes'], original['file_sha256']),
                                 'accepted_controls', '/ordered_control_shards/'+str(index)+'/'+name)


def verify_state4_acceptance_bundle(*, expected: State4AcceptanceExpectations,
                                    candidate_files: tuple[EvidenceBytes, ...],
                                    independent_files: tuple[EvidenceBytes, ...],
                                    handoff_bundles: tuple[HandoffBundle, ...]) -> State4AcceptanceVerification:
    validate_state4_acceptance_expectations(expected)
    roots = _roots(expected.root_requirements)
    findings = _Findings()
    bindings = {row['role']:row['file'] for row in expected.record_bindings}
    candidate, candidate_records = _index_evidence(candidate_files, bindings, roots, expected, findings, 'candidate')
    independent, independent_records = _index_evidence(independent_files, bindings, roots, expected, findings, 'independent')
    for role in set(candidate) & set(independent):
        if candidate[role] is not None and independent[role] is not None:
            if candidate[role].data != independent[role].data:
                findings.add('mismatch', role, '/candidate_reader', 'hash_mismatch')
    required_roles = set(_record_specs()) | {'accepted_controls'}
    for role in sorted(required_roles):
        for side, index in (('candidate', candidate), ('independent', independent)):
            if index.get(role) is None:
                findings.add('incomplete', role, '/'+side, 'evidence_unavailable')
    for row in expected.expected_records:
        for side, records in (('candidate', candidate_records), ('independent', independent_records)):
            if row.role in records:
                findings.compare(records[row.role], row.record, row.role, '/'+side+'/expected_record')
    expected_admission_files, admission_inventory_complete = _admissions(expected, independent, roots, findings)
    originals_checked = control_shards_checked = 0
    for side, records in (('candidate', candidate_records), ('independent', independent_records)):
        projection = records.get('accepted_controls')
        _graph(records, projection, expected, findings)
        if projection is None:
            continue
        findings.compare(projection['binding'], expected.binding, 'accepted_controls', '/'+side+'/binding')
        findings.compare(projection['capture'], expected.capture, 'accepted_controls', '/'+side+'/capture')
        if admission_inventory_complete:
            findings.compare(projection['acceptance_evidence_files'], expected_admission_files,
                             'accepted_controls', '/'+side+'/acceptance_evidence_files')
        else:
            known = {canonical_json_bytes(_plain(ref)) for ref in expected_admission_files}
            supplied = {canonical_json_bytes(_plain(ref)) for ref in projection['acceptance_evidence_files']}
            if not known <= supplied:
                findings.add('mismatch', 'accepted_controls', '/'+side+'/acceptance_evidence_files', 'binding_mismatch')
        for index, (actual, wanted) in enumerate(zip(projection['original_file_inventory'], expected.original_files, strict=True)):
            equal = findings.compare(actual, wanted, 'accepted_controls', '/'+side+'/original_file_inventory/'+str(index), 'identity_mismatch')
            if side == 'independent' and equal:
                originals_checked += 1
        for index, (actual, wanted) in enumerate(zip(projection['ordered_control_shards'], expected.control_shards, strict=True)):
            equal = findings.compare(actual, wanted, 'accepted_controls', '/'+side+'/ordered_control_shards/'+str(index))
            role = f"control/{wanted['split']}/shard_{wanted['shard_index']:04d}"
            if side == 'independent' and equal and role in records:
                control_shards_checked += 1
    bundles, duplicates = {}, set()
    if type(handoff_bundles) is not tuple:
        findings.add('invalid', 'handoffs', '/', 'schema_invalid')
    else:
        for bundle in handoff_bundles:
            if type(bundle) is not HandoffBundle or type(bundle.row_index) is not int or not 0 <= bundle.row_index < 232:
                findings.add('invalid', 'handoffs', '/', 'schema_invalid')
            elif bundle.row_index in bundles:
                duplicates.add(bundle.row_index)
                findings.add('invalid', 'handoffs', '/'+str(bundle.row_index), 'duplicate_or_alias')
            else:
                bundles[bundle.row_index] = bundle
    for index in duplicates:
        bundles.pop(index, None)
    originals = {row.file['relative_path']:row for row in expected.original_files}
    readonly = candidate_records.get('read_only_evidence')
    protected_results = []
    for index, row in enumerate(expected.protected_files):
        path = derive_state4_acceptance_inventory().protected_paths[index]
        metadata = row.expected_metadata_row if readonly is None else readonly['ordered_permission_evidence'][index]
        result = verify_state4_protected_file(expected=row, metadata_context=expected.metadata_context,
            original=originals[path], root_requirements=expected.root_requirements, metadata_row=metadata,
            handoff=bundles.get(index))
        if readonly is None:
            extra = _diagnostic('protected/'+str(index), '/metadata_row', 'evidence_unavailable')
            status = 'incomplete' if result.status == 'corroborated' else result.status
            result = ProtectedFileVerification(index, path, result.handoff_verification, status, _diagnostics(result.diagnostics+(extra,)))
        if result.status != 'corroborated':
            setattr(findings, {'invalid':'invalid', 'not_satisfied':'mismatch', 'incomplete':'incomplete'}[result.status], True)
        findings.rows.extend(result.diagnostics)
        protected_results.append(result)
    if originals_checked != 235 or control_shards_checked != 76:
        findings.incomplete = True
    status = 'invalid' if findings.invalid else 'mismatch' if findings.mismatch else 'incomplete' if findings.incomplete else 'corroborated'
    return State4AcceptanceVerification(status, tuple(protected_results), originals_checked, control_shards_checked,
                                        _diagnostics(findings.rows), False, False, False)


def validate_state4_acceptance_verification(result: State4AcceptanceVerification, *,
        expected: State4AcceptanceExpectations, candidate_files: tuple[EvidenceBytes, ...],
        independent_files: tuple[EvidenceBytes, ...], handoff_bundles: tuple[HandoffBundle, ...]) -> None:
    _require(type(result) is State4AcceptanceVerification, 'verification result type')
    recomputed = verify_state4_acceptance_bundle(expected=expected, candidate_files=candidate_files,
        independent_files=independent_files, handoff_bundles=handoff_bundles)
    _require(_equal(result, recomputed), 'complete verification recomputation mismatch')
