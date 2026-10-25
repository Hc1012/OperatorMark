"""Private later-phase evidence. Existing storage/public schemas are untouched."""
from dataclasses import dataclass
from . import state_provider_bindings as p

PROFILE = 'operatormark.private.later_storage.v2'
OBSERVATION = PROFILE + '.observation'
REPORT = PROFILE + '.comparison'


@dataclass(frozen=True)
class LaterStorageObservation:
    data: bytes


@dataclass(frozen=True)
class TerminationContext:
    outcome: str
    terminating_state: str
    origin_phase: str
    origin_binding: p.Binding
    origin_capture: p.Capture
    origin_file: p.FileRef
    failure_files: tuple


@dataclass(frozen=True)
class LaterStorageExpectations:
    binding: p.Binding
    prior_binding: p.Binding | None
    phase: str
    previous_phase: str | None
    current_file: p.FileRef
    previous_file: p.FileRef | None
    roots: tuple
    outputs: tuple
    providers: tuple
    prior_providers: tuple
    capture: p.Capture
    prior_capture: p.Capture | None
    admissions: tuple
    attempt_directory: p.AttemptDirectoryBinding
    context: TerminationContext
    prior_context: TerminationContext | None


def phase_binding(binding, phase):
    p.validate_binding(binding)
    if phase == 'state4_closed':
        allowed = (('state4', 'state4_worker'), ('state4', 'supervisor'))
    elif phase in ('state5_entry', 'state5_closed'):
        allowed = (('state5', 'state5_worker'), ('state5', 'supervisor'))
    elif phase == 'state6_entry':
        allowed = (('state6', 'state6_verifier'),)
    elif phase in ('pre_finalize', 'pre_move', 'post_move', 'post_audit', 'post_complete'):
        allowed = (('state6', 'state6_finalizer'),)
    elif phase == 'terminal':
        allowed = tuple((s, 'supervisor') for s in ('state4', 'state5', 'state6'))
    else:
        p.require(phase in ('failure_location', 'post_quarantine', 'cleanup'), 'phase binding')
        allowed = ((binding.state, binding.role),)
    p.require((binding.state, binding.role) in allowed, 'phase state/role')


def validate_termination_context(context):
    p.require(type(context) is TerminationContext, 'TerminationContext type')
    p.require(context.outcome in ('ongoing', 'succeeded', 'failed', 'unavailable'), 'context outcome')
    p.require(context.terminating_state in ('state4', 'state5', 'state6'), 'terminating state')
    p.validate_binding(context.origin_binding); p.validate_capture(context.origin_capture)
    p.validate_file_ref(context.origin_file)
    p.require(context.origin_phase in p.PHASES and context.origin_phase != 'terminal', 'context origin phase')
    phase_binding(context.origin_binding, context.origin_phase)
    p.require(context.origin_binding.state == context.terminating_state, 'context origin state')
    p.require(type(context.failure_files) is tuple and len(set(context.failure_files)) == len(context.failure_files), 'context failure inventory')
    for f in context.failure_files:
        p.validate_file_ref(f)
    if context.outcome == 'succeeded':
        p.require(context.terminating_state == 'state6' and context.origin_phase == 'post_complete' and not context.failure_files, 'success origin')
    if context.outcome == 'failed':
        p.require(bool(context.failure_files), 'failed context evidence')
    if context.outcome == 'ongoing':
        p.require(not context.failure_files, 'ongoing failure contradiction')


def context_from(value):
    p.keys(value, 'outcome terminating_state origin_phase origin_binding origin_capture origin_file failure_files')
    p.require(type(value['failure_files']) is list, 'context failure list')
    context = TerminationContext(value['outcome'], value['terminating_state'], value['origin_phase'],
        p.binding_from(value['origin_binding']), p.capture_from(value['origin_capture']),
        p.file_ref_from(value['origin_file']), tuple(p.file_ref_from(x) for x in value['failure_files']))
    validate_termination_context(context)
    return context


def validate_later_storage_expectations(expected):
    p.require(type(expected) is LaterStorageExpectations, 'LaterStorageExpectations type')
    phase_binding(expected.binding, expected.phase)
    p.validate_phase_transition(expected.previous_phase, expected.phase)
    p.validate_file_ref(expected.current_file); p.validate_roots(expected.roots)
    p.validate_outputs(expected.outputs); p.validate_capture(expected.capture)
    p.validate_attempt_directory(expected.attempt_directory, expected.roots, expected.outputs)
    prior = (expected.prior_binding, expected.previous_file, expected.prior_capture)
    if expected.previous_phase is None:
        p.require(prior == (None, None, None) and expected.prior_context is None, 'initial predecessor')
    else:
        p.require(all(x is not None for x in prior), 'complete accepted predecessor')
        phase_binding(expected.prior_binding, expected.previous_phase)
        p.validate_file_ref(expected.previous_file); p.validate_capture(expected.prior_capture)
        p.require(expected.previous_file != expected.current_file, 'observation slot alias')
        before, after = expected.prior_capture, expected.capture
        p.require(before.clock_epoch_id == after.clock_epoch_id and before.last_sequence < after.first_sequence and
                  before.finished_utc <= after.started_utc, 'accepted cutoff order')
        for name in ('attempt_id', 'protocol_sha256', 'implementation_commit', 'implementation_tree', 'implementation_parent', 'source_sha256', 'plan_file'):
            p.require(getattr(expected.binding, name) == getattr(expected.prior_binding, name), 'cross-phase premise substitution')
    for context, binding, capture, phase in ((expected.context, expected.binding, expected.capture, expected.phase),
                                           (expected.prior_context, expected.prior_binding, expected.prior_capture, expected.previous_phase)):
        if binding is None: continue
        validate_termination_context(context)
        p.require(context.terminating_state == binding.state, 'context/current state')
        for name in ('attempt_id', 'authorization_sha256', 'protocol_sha256', 'implementation_commit', 'implementation_tree', 'implementation_parent', 'source_sha256', 'plan_file'):
            p.require(getattr(context.origin_binding, name) == getattr(binding, name), 'context origin binding')
        p.require(context.origin_capture.clock_epoch_id == capture.clock_epoch_id and
            context.origin_capture.last_sequence <= capture.last_sequence and context.origin_capture.finished_utc <= capture.finished_utc, 'context future cutoff')
        if phase == 'terminal':
            p.require(binding.role == 'supervisor' and context.outcome != 'ongoing', 'terminal context/role')
    p.require(type(expected.providers) is tuple and tuple(x.provider_role for x in expected.providers if type(x) is p.ProviderExpectations) == p.PROVIDER_ROLES, 'complete provider inventory/order')
    for providers, binding, capture in ((expected.providers, expected.binding, expected.capture),
                                       (expected.prior_providers, expected.prior_binding, expected.prior_capture)):
        p.require(type(providers) is tuple, 'provider tuple')
        if binding is None:
            p.require(not providers, 'undeclared prior providers'); continue
        p.require(tuple(x.provider_role for x in providers if type(x) is p.ProviderExpectations) == p.PROVIDER_ROLES, 'provider role inventory')
        intervals = set()
        for provider in providers:
            p.validate_provider_expectations(provider)
            p.require(provider.binding == binding and provider.roots == expected.roots and provider.output_inventory == expected.outputs and provider.capture == capture and provider.attempt_directory == expected.attempt_directory, 'provider premises differ')
            intervals.add((provider.valid_from, provider.valid_until, provider.cleanup_until))
        p.require(len(intervals) == 1, 'conflicting authority intervals')
    p.require(type(expected.admissions) is tuple, 'storage admissions')
    roles = set()
    for admission in expected.admissions:
        p.validate_admission(admission)
        p.require(admission.role not in roles and admission.scope == 'storage_later_phase' and admission.outcome == 'accepted', 'storage admission premise')
        p.require(admission.validator_sources == expected.providers[0].validator_files, 'storage validator source binding')
        roles.add(admission.role)
    files = {a.role: a.file for a in expected.admissions}
    p.require(files.get('current_observation') == expected.current_file, 'current observation admission')
    if expected.previous_file is not None:
        p.require(files.get('previous_observation') == expected.previous_file, 'prior observation admission')


def _optional_ref(value):
    if value is not None:
        p.file_ref_from(value)


def validate_later_storage_observation(record):
    if type(record) is LaterStorageObservation:
        record = p.decode(record.data)
    p.keys(record, 'schema profile binding phase capture roots location location_file attempt_directory context objects operation failure_files cleanup_failure_files free_space_bytes')
    p.require(record['schema'] == OBSERVATION and record['profile'] == PROFILE, 'observation domain')
    binding = p.binding_from(record['binding']); phase_binding(binding, record['phase'])
    p.capture_from(record['capture'])
    context_from(record['context'])
    directory = record['attempt_directory']
    if directory is not None:
        p.keys(directory, 'native_identity custody_file root_binding_id relative_path resolved_path evidence_file')
        if directory['native_identity'] is not None: p.text(directory['native_identity'])
        _optional_ref(directory['custody_file']); _optional_ref(directory['evidence_file'])
        if directory['root_binding_id'] is None:
            p.require(directory['relative_path'] is None and directory['resolved_path'] is None, 'unknown directory path')
        else:
            p.require(directory['root_binding_id'] in ('T', 'F', 'Q'), 'directory root')
            p.relative(directory['relative_path']); p.text(directory['resolved_path'])
    p.require(type(record['roots']) is list, 'observed roots')
    roots = []
    for root in record['roots']:
        p.keys(root, 'root_binding_id canonical_path native_identity evidence_file')
        roots.append(p.RootBinding(**{**root, 'evidence_file': p.file_ref_from(root['evidence_file'])}))
    p.validate_roots(tuple(roots))
    p.require(record['location'] in ('T', 'F', 'Q', 'unknown'), 'location')
    _optional_ref(record['location_file'])
    p.require(type(record['objects']) is list, 'observed objects')
    slots = set(); paths = set(); identities = set()
    for row in record['objects']:
        p.keys(row, 'slot root_binding_id relative_path presence native_identity custody_file content_file presence_file stable_file')
        p.text(row['slot']); p.relative(row['relative_path'])
        p.require(row['root_binding_id'] in ('T', 'F', 'Q'), 'object root')
        p.require(row['presence'] in ('present', 'absent', 'unknown'), 'presence')
        for name in ('custody_file', 'content_file', 'presence_file', 'stable_file'):
            _optional_ref(row[name])
        identity = row['native_identity']
        if identity is not None:
            p.text(identity)
        if row['presence'] != 'present':
            p.require(all(row[k] is None for k in ('native_identity', 'custody_file', 'content_file', 'stable_file')), 'negative presence contradiction')
        if identity is not None:
            p.require(identity not in identities, 'object identity alias'); identities.add(identity)
        path = (row['root_binding_id'], row['relative_path'].casefold())
        p.require(row['slot'] not in slots and path not in paths, 'object slot/path alias')
        slots.add(row['slot']); paths.add(path)
    op = record['operation']
    p.keys(op, 'action call_status effect_status evidence_file authority_file')
    p.require(op['action'] in ('none', 'metadata', 'promote', 'quarantine', 'cleanup'), 'operation action')
    p.require(op['call_status'] in ('not_called', 'returned_success', 'returned_failure', 'unavailable'), 'call status')
    p.require(op['effect_status'] in ('none', 'complete', 'partial', 'unknown'), 'effect status')
    _optional_ref(op['evidence_file']); _optional_ref(op['authority_file'])
    if op['call_status'] == 'not_called':
        p.require(op['effect_status'] == 'none', 'uncalled effect')
    if op['action'] == 'none':
        p.require((op['call_status'], op['effect_status']) == ('not_called', 'none'), 'no operation contradiction')
    for name in ('failure_files', 'cleanup_failure_files'):
        p.require(type(record[name]) is list, 'failure inventory')
        refs = [p.file_ref_from(f) for f in record[name]]
        p.require(len(refs) == len(set(refs)), 'duplicate failure evidence')
    if record['free_space_bytes'] is not None:
        p.integer(record['free_space_bytes'])
    return record


def strict_later_storage_loads(data):
    record = p.decode(data)
    validate_later_storage_observation(record)
    return record


def canonical_later_storage_bytes(record):
    validate_later_storage_observation(record)
    return p.encode(record)
