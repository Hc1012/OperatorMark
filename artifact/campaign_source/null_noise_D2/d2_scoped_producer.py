"""Fixed 200-model raw960 D2 diagnostic stage after a scoped mechanical gate.

The completed bridge, its independent verifier and durable publisher must all
be closed before this process is admitted.  Historical admissions/pins remain
unchanged.  The shared numerical body retains its development-domain label;
external D2 readback binds its use to the narrow runtime receipt.  No training,
inverse task, task decision, native solve or general runtime admission occurs.
"""
from pathlib import Path
from runtime_values import plain_runtime
import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
import traceback

from compatibility_compare import exact, readback, sha
from compatibility_producer import verify, accepted, prerequisites, HistoricalAccess, context as compatibility_context

POLICY = 'D2_COMBINED_POLICY.json'
SCOPE = 'D2_raw960_diagnostics_only'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def roster_hash(order):
    return hashlib.sha256(json.dumps(order, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_roster(policy, opening):
    """Exact fixed whole-cell roster; metadata only, no outcome selection."""
    need(policy['owners'] == ['O%02d' % i for i in range(1, 11)], 'Ten-owner population differs')
    order = policy['model_order']
    rows = policy['model_jobs']
    need([r['id'] for r in rows] == order and len(order) == len(set(order)) == 200,
         'Fixed remaining raw960 model order differs')
    need(not set(order) & set(policy['completed_model_ids'])
         and not set(order) & set(policy['completed_diagnostic_model_ids'])
         and policy['adaptive_jobs'] == [], 'Completed work must not be repeated')
    need(len(policy['completed_diagnostic_model_ids']) == len(set(policy['completed_diagnostic_model_ids'])) == 200,
         'Completed raw320 population differs')
    cells = {name: cell for name, cell in opening['D2']['cells'].items() if cell['checkpoint'] == 'raw960'}
    required = [member for cell in cells.values() for member in cell['members']]
    need(len(cells) == 10 and len(required) == len(set(required)) == 200 and set(required) == set(order),
         'Exact frozen raw960 D2 whole-cell population differs')
    need(all(sum(r['owner'] == owner for r in rows) == 20 for owner in policy['owners']),
         'Twenty selected models per owner required')
    need(all(row['id'] == row['owner'] + '/' + row['name'] and row['name'].endswith('_raw960')
             for row in rows), 'Raw960 owner/model identity differs')
    need(policy['previous_completed_full_evaluations'] == 851,
         'Previously completed full-evaluation count differs')
    return cells


def scoped_context(historical, policy, policy_sha, admission_sha):
    value = compatibility_context(historical, policy, policy_sha)
    value.bindings.update(domain='scoped_D2_diagnostics', scope=SCOPE,
                          mechanical_runtime_admission_sha256=admission_sha,
                          frozen_raw960_model_order_sha256=roster_hash(policy['model_order']))
    return value


def check_admission_binding(gate, policy, policy_sha):
    """Additional fixed dispatch checks after verify_bridge's full verification."""
    need(gate.get('passed') is True and gate.get('scope') == SCOPE,
         'A passing D2-only mechanical runtime admission is required')
    need(gate.get('policy_sha256') == policy_sha,
         'Mechanical admission does not bind this combined policy')
    need(gate.get('bridge_policy_sha256') == policy['bridge_policy_sha256'],
         'Mechanical admission binds a different bridge policy')
    need(gate.get('model_order_sha256') == roster_hash(policy['model_order']),
         'Mechanical admission does not bind the fixed raw960 roster')
    need(gate.get('candidate_runtime_payload_sha256') == policy['candidate_runtime_payload_sha256'],
         'Mechanical admission candidate runtime differs')


def run(job, policy_sha, deadline):
    job = Path(job).resolve()
    need(sha(job/POLICY) == policy_sha, 'Combined policy differs')
    policy = read(job/POLICY)
    activation = read(job/'ACTIVATION.json')
    need(time.time() < deadline <= activation['deadline'] - policy['return_reserve_seconds'],
         'Fixed raw960 work/final-return window required')
    need(sha(job/policy['bridge_policy']) == policy['bridge_policy_sha256'],
         'Frozen bridge policy changed')
    opening = read(job/policy['opening'])
    cells = check_roster(policy, opening)
    out = job/'output'
    out.mkdir(exist_ok=False)
    started, last, current = time.monotonic(), 0., 'scoped runtime admission'
    completed, durable, diagnostic_index = {}, {}, {}
    counts = {name: 0 for name in policy['operation_limits']}

    def check():
        nonlocal last
        if time.time() >= deadline:
            raise TimeoutError('Fixed raw960 diagnostic deadline reached; no automatic retry')
        if shutil.disk_usage(job).free < 2 * 1024**3:
            raise RuntimeError('Less than 2 GiB local free space remains')
        now = time.monotonic()
        if now - last >= 20:
            last = now
            record = dict(current=current, completed_jobs=len(completed), total_jobs=200,
                          durable_owners=len(durable), seconds=round(now-started, 1))
            with (out/'progress.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, sort_keys=True) + '\n')
                f.flush()
            print(json.dumps(record), flush=True)

    try:
        verify(job, policy, check)
        sys.path[:0] = [str(job.parent/'payload/code/tools'), str(job.parent/'payload/code')]
        from q6_common import THREAD_ENV
        os.environ.update(THREAD_ENV)
        from msfinal import backend as B, materials as A, evaluation as E, orchestration as O
        from msfinal import models, attacks as T
        from msdev import pn_runtime_migration as P
        import q5_model as Q
        import torch
        from verify_bridge import verify_admission

        Q.configure()
        gate = verify_admission(job, policy, policy_sha, check)
        check_admission_binding(gate, policy, policy_sha)
        admission_sha = sha(job/'D2_RUNTIME_ADMISSION.json', check)
        actual = plain_runtime(A.runtime_identity())
        put(out/'RUNTIME_ACTUAL.json', actual)
        exact(actual, policy['expected_runtime'], 'Fresh raw960 runtime')
        observation_path = job/'observations/runtime/CANDIDATE_INTERPRETER.json'
        need(sha(observation_path, check) == gate['observed_interpreter_sha256'],
             'Admitted interpreter observation changed')
        captured_interpreter = read(observation_path)
        observed_interpreter = P.observed_interpreter_identity(check)
        put(out/'INTERPRETER_START.json', observed_interpreter)
        P.target_build(observed_interpreter)
        exact(observed_interpreter, captured_interpreter, 'Raw960 interpreter versus accepted bridge')
        for name, expected_sha in policy['expected_modules'].items():
            module = importlib.import_module(name)
            need(getattr(module, '__file__', None) and sha(module.__file__, check) == expected_sha,
                 'Imported numerical module differs: ' + name)
        start_libraries = P.libraries(check)
        put(out/'LIBRARIES_START.json', start_libraries)
        P.check_libraries(start_libraries, check)

        historical, contexts, specifications, items = {}, {}, {}, {}
        for owner in policy['owners']:
            check()
            admission = B.admit(prerequisites(job, policy, owner))
            A.preflight(admission)
            specifications[owner] = E.preflight(admission, 'final')
            historical[owner] = admission
            contexts[owner] = scoped_context(admission, policy, policy_sha, admission_sha)
        # All 200 descriptors and checkpoint bytes are admitted before the first
        # candidate model is constructed; the bridge cannot expand this roster.
        for row in policy['model_jobs']:
            check()
            descriptor = row['descriptor']
            procedure = descriptor.get('procedure')
            need(row['dependency_paths'] == descriptor.get('dependencies') == {}
                 and descriptor['wrapper_steps'] is None and isinstance(procedure, dict)
                 and set(procedure) == {'architecture', 'transform', 'variant'}
                 and procedure['transform'] is None, 'Frozen D2 reconstruction differs')
            item = O.FinalModel(row['name'], job/row['model_path'], descriptor['model_sha256'],
                                accepted(job, policy, row['acceptance']), descriptor['wrapper_steps'],
                                procedure=procedure, dependencies={})
            O.final_model(item, historical[row['owner']])
            exact(O.model_descriptor(item), descriptor, 'Historical final model descriptor')
            need(O.measurement_identity(item) == row['measurement_sha256'],
                 'Full-descriptor measurement identity differs')
            items[row['id']] = item

        put(out/'INPUT_PREFLIGHT.json', dict(complete=True, selected_models=200, diagnostic_only=True,
            conditional_cells=list(cells), opening_sha256=policy['opening_sha256'],
            contract_bindings_sha256=sha(job/'admission/FINAL_CONTRACT_BINDINGS.json', check),
            mechanical_runtime_admission_sha256=admission_sha, scope=SCOPE,
            source_pins_unchanged=True, historical_admissions_unchanged=True,
            output_numeric_domain='development', all_model_bytes_admitted_before_first_call=True,
            model_calls=0, scientific_acceptance=False))
        put(out/'START.json', dict(policy_sha256=policy_sha, opening_sha256=policy['opening_sha256'],
            scope=SCOPE, mechanical_runtime_admission_sha256=admission_sha,
            final_GPU_diagnostics_started=True, task_selection_started=False,
            inverse_work_started=False, model_updates=0, native_solver_calls=0,
            canonical_worker_advanced=False, full_programme_retained=True))

        from msfinal.drive_transport import GoogleDriveTransfer, RefreshingMetadataToken
        from combined_remote import pack
        transfer = GoogleDriveTransfer(policy['drive_folder'], RefreshingMetadataToken('metadata.google.internal'),
            job/'transport_rolling', byte_limit=policy['rolling_capsules']['total_upload_bytes_upper'],
            metadata_byte_limit=4 * 1024**2)
        (out/'projections').mkdir()
        for owner in policy['owners']:
            store = E.ProjectionStore(out/'projections'/owner)
            context = contexts[owner]
            spec = specifications[owner]
            for row in (r for r in policy['model_jobs'] if r['owner'] == owner):
                current = row['id']
                check()
                item = items[current]
                dest = out/'diagnostics'/owner/row['name']
                dest.mkdir(parents=True, exist_ok=False)
                binding = dict(study_id=policy['study_id'], configuration_sha256=policy['configuration_sha256'],
                    policy_sha256=policy_sha, opening_sha256=policy['opening_sha256'],
                    scope=SCOPE, mechanical_runtime_admission_sha256=admission_sha,
                    phase_name='diagnostics/' + current)
                put(dest/'JOB_INTENT.json', dict(job=current, descriptor=row['descriptor'], binding=binding,
                    completed=False, output_numeric_domain='development'))
                # Same architecture factory and byte-bound tensor loader as the
                # pinned O.load_final_model.  The scoped adapter does not call
                # its historical-runtime public entry or modify that guard.
                model = models.create(item.procedure['architecture']).to('cuda').eval()
                model.load_state_dict(T._parent_weights(item), strict=True)
                evaluation_binding = E.evaluation_binding(context, 'final', row['measurement_sha256'], None)
                E._evaluate_model(model, context, dest/'evaluation', check, phase='final',
                    model_sha256=row['measurement_sha256'], wrapper_steps=None, projection_store=store,
                    access=HistoricalAccess, spec=spec, components='diagnostics')
                result, hashes = readback(dest/'evaluation', spec, evaluation_binding, 'development', check)
                need(result['diagnostics_complete'] is True and result['task_status'] == 'pending'
                     and result['complete'] is False and len(result['scores']) == 4096,
                     'Scoped D2 diagnostic state differs')
                proof = dict(verified=True, scope=SCOPE, output_numeric_domain='development',
                    files=list(hashes), file_hashes=hashes, counts=result['counts'],
                    mechanical_runtime_admission_sha256=admission_sha,
                    evaluation_binding=evaluation_binding, scientific_acceptance=False)
                diagnostic_index[current] = dict(directory=(dest/'evaluation').relative_to(out).as_posix(),
                    result_sha256=sha(dest/'evaluation/RESULT.json', check),
                    evaluation_binding=evaluation_binding, readback=proof,
                    mechanical_runtime_admission_sha256=admission_sha, output_numeric_domain='development')
                completed[current] = dict(kind='D2_raw960_scoped_GPU_diagnostics', model_sha256=item.sha256,
                    measurement_sha256=row['measurement_sha256'], descriptor=row['descriptor'],
                    result_sha256=sha(dest/'evaluation/RESULT.json', check), diagnostic=diagnostic_index[current],
                    scientific_acceptance=False, task_selection_pending=True)
                put(dest/'GPU_READBACK.json', dict(complete=True, **completed[current],
                    opening_sha256=policy['opening_sha256'], scope=SCOPE))
                for name, number in dict(diagnostic_GPU_evaluations=1, validation_controls=512,
                    random_JVPs=1024, keyed_JVPs=512, detector_controls=128).items():
                    counts[name] += number
                print(current + ': F/J/K and all4096 detector scores read back; no task work.', flush=True)
                del model
                torch.cuda.empty_cache()

            # Completed owner outputs are made durable while the producer is
            # alive.  This receipt does not claim final process closure.
            current = owner + '/durable_save'
            check()
            capsule_dir = job/'owner_capsules'
            capsule_dir.mkdir(exist_ok=True)
            entries = {}
            for directory in (out/'diagnostics'/owner, out/'projections'/owner):
                for path in directory.rglob('*'):
                    if path.is_file():
                        entries[path.relative_to(job).as_posix()] = path
            receipt = capsule_dir/(owner + '_COMPLETED.json')
            put(receipt, dict(owner=owner, policy_sha256=policy_sha, opening_sha256=policy['opening_sha256'],
                jobs={key: value for key, value in completed.items() if key.startswith(owner + '/')},
                scope=SCOPE, mechanical_runtime_admission_sha256=admission_sha,
                formal_custody_assurance=False, producer_process_still_running=True,
                control_reference=read(job/'CONTROL_SAVED.json')['reference']))
            entries['OWNER_COMPLETED.json'] = receipt
            capsule = capsule_dir/('FINAL_D2_RAW960_SCOPED_' + activation['attempt'] + '_' + owner + '.zip')
            pack(capsule, entries, policy['owner_capsule_bound_bytes'], check)
            remote = transfer.upload_verified(capsule, deadline=min(deadline, time.time()+240), check=check)
            durable[owner] = dict(reference=remote, sha256=sha(capsule, check), jobs=list(read(receipt)['jobs']),
                                  process_closure_pending=True, scope=SCOPE)
            put(capsule_dir/(owner + '_SAVED.json'), durable[owner])
            print(owner + ': twenty completed diagnostics SAVED AND READ BACK IN DRIVE ('
                  + str(len(durable)) + '/10 owners).', flush=True)
            del store
            torch.cuda.empty_cache()

        need(counts == policy['operation_limits'] and len(completed) == 200 and len(durable) == 10,
             'Incomplete fixed raw960 diagnostic population')
        current = 'final immutable-input and scoped-runtime readback'
        verify(job, policy, check)
        exact(plain_runtime(A.runtime_identity()), actual, 'Raw960 runtime changed')
        final_interpreter = P.observed_interpreter_identity(check)
        put(out/'INTERPRETER_FINAL.json', final_interpreter)
        P.target_build(final_interpreter)
        exact(final_interpreter, captured_interpreter, 'Final raw960 interpreter versus accepted bridge')
        P.check_libraries(start_libraries, check)
        final_libraries = P.libraries(check)
        put(out/'LIBRARIES_FINAL.json', final_libraries)
        P.check_libraries(final_libraries, check)
        for name in set(start_libraries) & set(final_libraries):
            need(start_libraries[name] == final_libraries[name], 'Loaded library changed: ' + name)
        final_gate = verify_admission(job, policy, policy_sha, check)
        check_admission_binding(final_gate, policy, policy_sha)
        exact(final_gate, gate, 'Mechanical runtime admission changed')
        need(sha(job/'D2_RUNTIME_ADMISSION.json', check) == admission_sha, 'Mechanical admission bytes changed')

        put(out/'DIAGNOSTIC_INDEX.json', dict(diagnostics=diagnostic_index, diagnostic_count=200,
            cells=cells, checkpoint='raw960', opening_sha256=policy['opening_sha256'], scope=SCOPE,
            output_numeric_domain='development', mechanical_runtime_admission_sha256=admission_sha,
            candidate_runtime_payload_sha256=policy['candidate_runtime_payload_sha256'],
            task_selection_pending=True, inverse_input_iterations=0, model_updates=0, native_solver_calls=0))
        projection_files = {}
        for path in sorted((out/'projections').rglob('*')):
            if path.is_file():
                projection_files[path.relative_to(out).as_posix()] = dict(bytes=path.stat().st_size,
                                                                         sha256=sha(path, check))
        put(out/'D2_RAW960_SCOPED_DIAGNOSTICS_RECEIPT.json', dict(complete=True, scope=SCOPE,
            policy_sha256=policy_sha, opening_sha256=policy['opening_sha256'],
            mechanical_runtime_admission_sha256=admission_sha, frozen_model_order_sha256=roster_hash(policy['model_order']),
            diagnostic_index_sha256=sha(out/'DIAGNOSTIC_INDEX.json', check), projection_files=projection_files,
            candidate_runtime_payload_sha256=policy['candidate_runtime_payload_sha256'],
            output_numeric_domain='development', domain_rewritten=False, historical_admissions_unchanged=True,
            producer_assertion=True, independent_return_verification_pending=True,
            producer_closure_pending=True, publisher_closure_pending=True,
            scientific_acceptance=False, formal_custody_assurance=False))
        put(out/'RESULT.json', dict(status='D2_RAW960_SCOPED_DIAGNOSTICS_COMPLETE_TASK_DECISION_PENDING', complete=True,
            completion_scope='Fixed200 raw960 diagnostics under the D2-only mechanical runtime receipt; no task decision or full evaluation completion.',
            scope=SCOPE, counts=counts, jobs=completed, durable_owners=durable, seconds=time.monotonic()-started,
            policy_sha256=policy_sha, opening_sha256=policy['opening_sha256'],
            mechanical_runtime_admission_sha256=admission_sha,
            scoped_receipt_sha256=sha(out/'D2_RAW960_SCOPED_DIAGNOSTICS_RECEIPT.json', check),
            output_numeric_domain='development', final_GPU_diagnostics_complete=200,
            previous_completed_raw320_diagnostics=200, combined400_diagnostics_pending_return_verification=True,
            final_full_evaluations_complete=0, adaptive_final_readouts_complete=0,
            previous_primary_full_evaluations=20, previous_completed_full_evaluations=851,
            task_selection_pending=True, conditional_tasks_started=False, canonical_worker_advanced=False,
            scientific_acceptance=False, formal_custody_assurance=False, physical_qualification=False,
            complete_programme=False, remaining=policy['remaining']))
    except BaseException:
        put(out/'FAILURE.json', dict(current=current, traceback=traceback.format_exc(), complete=False,
            completed=completed, durable_owners=durable, counts_completed_jobs_only=counts,
            partial_job_counts='Inspect retained partial evaluation and intents; unknown counts are not zero.',
            seconds=time.monotonic()-started, automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--policy-sha', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    run(args.job, args.policy_sha, args.deadline)
