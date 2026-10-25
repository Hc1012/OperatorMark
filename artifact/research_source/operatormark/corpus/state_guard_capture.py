"""Synthetic guard evidence assembly and independent pure recomputation.

Accepted observer/source/interval bindings come from outside this component.
Readable event bytes and a final child message do not prove lifetime coverage.
No native hooks, RNG access or numerical imports are implemented here.
"""
from dataclasses import dataclass

from operatormark.corpus import final_evidence_records as records
from operatormark.corpus.state_execution_authority import (
    LaunchAuthorityError, canonical, decode, digest, enum, integer, need,
    object_keys, qualify, ref, refs, sha, text,
)


AUTHORIZED = ('seed_access_by_split', 'pcg64_constructors_by_split',
              'generator_constructors_by_split', 'top_level_rng_calls_by_split')
FORBIDDEN = ('rng_calls', 'corpus_seed_access', 'legacy_data_access',
             'model_execution', 'network_access', 'unauthorized_test_access')
CHECKPOINTS = ('after_base_import', 'after_guard_install', 'after_project_import', 'after_execution')


@dataclass(frozen=True, slots=True)
class GuardExpectations:
    """Exact independently declared event population and coverage contract."""
    specification: bytes
    validator_files: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class GuardComparison:
    status: str
    closure: bytes | None
    diagnostics: tuple[str, ...]
    request_accounting: bytes = b''


def validate_guard_event(event):
    object_keys(event, 'schema launch_slot_id observer_id clock_epoch_id sequence kind operation split decision call_status effect_status subject_file prior_event_file record_self_sha256')
    need(event['schema'] == 'state_guard_event.v1', 'guard event schema')
    digest(event['launch_slot_id']); text(event['observer_id']); text(event['clock_epoch_id'])
    integer(event['sequence'])
    enum(event['kind'], ('coverage_start','operation','coverage_end','process_exit'))
    enum(event['operation'], (*AUTHORIZED, *FORBIDDEN, 'lifetime'))
    need(event['split'] is None or type(event['split']) is str and event['split'] in records.SPLITS, 'split binding')
    enum(event['decision'], ('allowed','denied','unavailable'))
    enum(event['call_status'], records.CALLS); enum(event['effect_status'], records.EFFECTS)
    ref(event['subject_file']); ref(event['prior_event_file'], True)
    if event['kind'] == 'operation':
        need(event['operation'] != 'lifetime', 'operation family required')
        need((event['split'] is not None) == (event['operation'] in AUTHORIZED), 'family/split disagreement')
    else:
        need(event['operation'] == 'lifetime' and event['split'] is None, 'lifetime event binding')
    if event['call_status'] == 'not_called':
        need(event['effect_status'] != 'expected_effect_observed', 'uncalled effect contradiction')
    if event['decision'] == 'denied':
        need(event['call_status'] == 'not_called' and event['effect_status'] in ('none_observed','unknown'), 'denial cannot imply performed call')
    need(event['record_self_sha256'] == sha(canonical({k:v for k,v in event.items() if k != 'record_self_sha256'})), 'event self-hash mismatch')


def make_guard_event(**fields):
    event = dict(schema='state_guard_event.v1', **fields)
    event['record_self_sha256'] = sha(canonical(event))
    validate_guard_event(event)
    return canonical(event)


def _expectations(expected):
    need(type(expected) is GuardExpectations, 'independent guard expectations required')
    need(type(expected.validator_files) is tuple and bool(expected.validator_files), 'guard validator bindings required')
    for f in expected.validator_files:
        ref(decode(f))
    spec = decode(expected.specification)
    object_keys(spec, 'binding capture launch_slot_id observer_id event_files lifetime_file exit_file creation_file base_closure_file required_hooks rng_checkpoint_files')
    records.execution_prefix(spec['binding']); records._capture(spec['capture'], spec['binding'])
    digest(spec['launch_slot_id']); text(spec['observer_id'])
    refs(spec['event_files'], True)
    for key in ('lifetime_file','exit_file','creation_file','base_closure_file'):
        ref(spec[key])
    need(type(spec['required_hooks']) is list and spec['required_hooks'] == sorted(set(spec['required_hooks']))
         and bool(spec['required_hooks']), 'explicit unique hook inventory required')
    for hook in spec['required_hooks']:
        text(hook)
    object_keys(spec['rng_checkpoint_files'], ' '.join(CHECKPOINTS))
    for f in spec['rng_checkpoint_files'].values():
        ref(f)
    return spec


def _compute(*, expected, independent_files):
    spec = _expectations(expected)
    capture = spec['capture']; diagnostics = []
    def read(f, scope):
        try:
            return decode(qualify(f, independent_files, validators=expected.validator_files, scope=scope,
                                 first=capture['started_utc'], last=capture['finished_utc']))
        except LaunchAuthorityError:
            diagnostics.append('unavailable_'+scope)
            return None
    events = []
    previous = None
    for index, f in enumerate(spec['event_files']):
        event = read(f, 'guard_event')
        if event is not None:
            validate_guard_event(event)
            need(event['launch_slot_id'] == spec['launch_slot_id']
                 and event['observer_id'] == spec['observer_id']
                 and event['clock_epoch_id'] == capture['clock_epoch_id'], 'guard observer/slot/clock mismatch')
            need(event['sequence'] == capture['first_sequence'] + index
                 and event['prior_event_file'] == previous, 'guard event gap/duplicate/predecessor')
            need(event['subject_file'] == (spec['exit_file'] if event['kind'] == 'process_exit' else spec['creation_file']), 'guard subject substitution')
        events.append(event); previous = f
    need(capture['last_sequence'] == capture['first_sequence'] + len(events)-1, 'declared event interval/population mismatch')
    complete_events = all(e is not None for e in events)
    ordered = complete_events and len(events) >= 3 and events[0]['kind'] == 'coverage_start' and events[-2]['kind'] == 'process_exit' and events[-1]['kind'] == 'coverage_end' and all(e['kind'] == 'operation' for e in events[1:-2])
    if not ordered:
        diagnostics.append('guard_lifetime_event_order_incomplete')
    lifetime = read(spec['lifetime_file'], 'guard_lifetime')
    exit_facts = read(spec['exit_file'], 'process_observation')
    coverage = False
    if lifetime is not None:
        object_keys(lifetime, 'slot_id observer_id clock_epoch_id creation_file exit_file first_sequence last_sequence started_before_project_import through_direct_exit hooks')
        need(lifetime['slot_id'] == spec['launch_slot_id'] and lifetime['observer_id'] == spec['observer_id']
             and lifetime['clock_epoch_id'] == capture['clock_epoch_id'] and lifetime['creation_file'] == spec['creation_file']
             and lifetime['exit_file'] == spec['exit_file'] and lifetime['first_sequence'] == capture['first_sequence']
             and lifetime['last_sequence'] == capture['last_sequence'], 'lifetime binding conflict')
        need(type(lifetime['started_before_project_import']) is bool and type(lifetime['through_direct_exit']) is bool, 'lifetime flags exact')
        need(type(lifetime['hooks']) is list, 'hook evidence inventory')
        coverage = (lifetime['started_before_project_import'] and lifetime['through_direct_exit']
                    and lifetime['hooks'] == spec['required_hooks'])
    if exit_facts is not None:
        object_keys(exit_facts, 'schema kind slot_id process_identity direct_exit_code reap_status observation failure_files'
                    if exit_facts.get('schema') == 'state_process_exit_observation.v2' else 'kind slot_id process_identity direct_exit_code reap_status')
        need(exit_facts['kind'] == 'exit' and exit_facts['slot_id'] == spec['launch_slot_id'], 'guard exit binding')
        need(exit_facts['process_identity']['creation_observation_file'] == spec['creation_file'], 'guard exit/creation correspondence')
        need(exit_facts['direct_exit_code'] is None or type(exit_facts['direct_exit_code']) is int, 'guard direct exit')
        coverage = coverage and exit_facts['direct_exit_code'] is not None and exit_facts['reap_status'] == 'observed_reaped'
    else:
        coverage = False
    coverage = coverage and ordered
    if not coverage:
        diagnostics.append('through_exit_coverage_unavailable')
    uncertain = any(e is not None and (e['decision'] == 'unavailable' or e['call_status'] == 'outcome_unavailable'
                    or e['effect_status'] == 'unknown') for e in events)
    if uncertain:
        diagnostics.append('guard_operation_outcome_unavailable')
    measured = coverage and not uncertain
    evidence = sorted(spec['event_files'] + [spec['lifetime_file'], spec['exit_file']], key=canonical)
    def count(family, split):
        value = sum(e['kind'] == 'operation' and e['operation'] == family and e['split'] == split
                    and (family in FORBIDDEN or e['call_status'] in ('returned_success','returned_failure')) for e in events if e is not None)
        return dict(measurement='measured' if measured else 'unavailable', value=value if measured else None,
                    evidence_files=evidence if measured else [])
    commitments = {}
    for checkpoint, f in spec['rng_checkpoint_files'].items():
        row = read(f, 'rng_commitment')
        if row is None:
            commitments[checkpoint] = dict(numpy_legacy_sha256=None, python_random_sha256=None)
        else:
            object_keys(row, 'slot_id checkpoint numpy_legacy_sha256 python_random_sha256')
            need(row['slot_id'] == spec['launch_slot_id'] and row['checkpoint'] == checkpoint, 'RNG checkpoint identity')
            digest(row['numpy_legacy_sha256']); digest(row['python_random_sha256'])
            commitments[checkpoint] = {k:row[k] for k in ('numpy_legacy_sha256','python_random_sha256')}
    equalities = {}
    for checkpoint in CHECKPOINTS[1:]:
        equalities[checkpoint] = {}
        for family in ('numpy_legacy','python_random'):
            left = commitments['after_base_import'][family+'_sha256']; right = commitments[checkpoint][family+'_sha256']
            equalities[checkpoint][family+'_equals_baseline'] = None if left is None or right is None else left == right
    closure = dict(schema=records.execution_prefix(spec['binding'])+'state_guard_closure.v1', binding=spec['binding'], capture=capture,
                   process_creation_evidence_file=spec['creation_file'], frozen_base_import_closure_file=spec['base_closure_file'],
                   rng_state_commitments=commitments, rng_state_equalities=equalities,
                   authorized_counts={family:{split:count(family,split) for split in records.SPLITS} for family in AUTHORIZED},
                   forbidden_counts={family:count(family,None) for family in FORBIDDEN},
                   guard_lifetime_evidence_files=[spec['lifetime_file']] if lifetime is not None else [],
                   coverage=dict(started_before_project_import=bool(lifetime and lifetime['started_before_project_import']),
                                 continued_through_exit=coverage, all_required_hooks_accounted=coverage,
                                 unavailable_hook_evidence_files=[]))
    closure = records.build_final_evidence_record('state_guard_closure', closure)
    accounting = {}
    for family in (*AUTHORIZED, *FORBIDDEN):
        population = [e for e in events if e is not None and e['kind'] == 'operation' and e['operation'] == family]
        accounting[family] = dict(observed_requests=len(population),
            denied_requests=sum(e['decision'] == 'denied' for e in population),
            population_complete=bool(coverage),
            **{status:sum(e['call_status'] == status for e in population) for status in records.CALLS})
    return closure, tuple(diagnostics), canonical(accounting)


def derive_guard_capture(*, expected, independent_files):
    closure, diagnostics, accounting = _compute(expected=expected, independent_files=independent_files)
    records.validate_state_guard_closure(closure)
    return GuardComparison('incomplete' if diagnostics else 'corroborated', canonical(closure), diagnostics, accounting)


def compare_guard_capture(*, expected, candidate, independent_files):
    """Recompute every conclusion from independent retained evidence, not a trace."""
    actual = derive_guard_capture(expected=expected, independent_files=independent_files)
    need(type(candidate) is GuardComparison, 'exact candidate comparison required')
    need(candidate == actual, 'guard report recomputation mismatch')
    return actual
