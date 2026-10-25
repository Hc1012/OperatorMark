"""One fixed O01 raw320 compatibility bridge, never a scientific admission.

Historical final admissions and materials remain unchanged.  This adapter calls
the existing private diagnostic body with explicit compatibility provenance;
the body's development-domain label is retained.  No training, task decisions,
inverse work, native solver call, raw960 continuation or automatic retry.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
import traceback

from compatibility_compare import exact, compare_one, projection_key, compare_projection_pair, sha
from runtime_values import plain_runtime

POLICY = 'D2_PYTHON31316_COMPAT_POLICY.json'
FAMILIES = ('extract_deeponet', 'extract_fno16', 'extract_fno2', 'extract_fno32',
            'extract_fno64', 'extract_jvp', 'extract_unet', 'heal20', 'heal50', 'heal80')


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


def payload_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def verify(job, policy, check=lambda: None):
    expected = {}
    for name in ('files', 'original_files', 'marked_files', 'final_material_files', 'reference_files'):
        for path, record in policy.get(name, {}).items():
            if path in expected:
                exact({k: expected[path][k] for k in ('bytes', 'sha256')},
                      {k: record[k] for k in ('bytes', 'sha256')}, 'Repeated input expectation/' + path)
            expected[path] = record
    for path, record in policy.get('aliases', {}).items():
        expected[path] = record['expected']
    for group in policy.get('additional_sources', {}).values():
        expected.update(group['files'])
    for name, record in expected.items():
        check()
        path = job/name
        need(path.is_file() and not path.is_symlink() and path.stat().st_size == record['bytes']
             and sha(path, check) == record['sha256'], 'Frozen input differs: ' + name)
    for name, expected_sha in policy['source_pins'].items():
        check()
        need(sha(job.parent/'payload/code'/name, check) == expected_sha, 'Scientific source differs: ' + name)


def accepted(job, policy, name, scientific=False):
    from msfinal.backend import AcceptedRecord
    from msfinal.scientific_admission import ScientificRecord
    from msfinal.portable_corpus import PortableCorpus
    path = job/name
    data = path.read_bytes()
    expected = policy['files'][name]['sha256']
    need(hashlib.sha256(data).hexdigest() == expected, 'Historical record differs: ' + name)
    return (ScientificRecord(path, expected, data, PortableCorpus(**policy['corpus']))
            if scientific else AcceptedRecord(path, expected, data))


def prerequisites(job, policy, owner):
    from msfinal.backend import Prerequisites
    row = policy['owner_records'][owner]
    return Prerequisites(study_id=policy['study_id'], owner_id=owner,
                         clean_model_sha256=policy['clean_models'][owner],
                         configuration=accepted(job, policy, 'admission/FINAL_ATTACK_CONFIGURATION.json'),
                         inputs=accepted(job, policy, row['inputs'], True),
                         runtime=accepted(job, policy, row['runtime'], True),
                         calibration=accepted(job, policy, 'admission/FINAL_CALIBRATION.json'),
                         clean_evaluation=accepted(job, policy, row['clean_evaluation']),
                         material_paths={k: job/name for k, name in row['material_paths'].items()},
                         source_root=job.parent/'payload/code',
                         assets=accepted(job, policy, row['assets'], True), assets_root=job,
                         opening=accepted(job, policy, policy['opening'], True))


class HistoricalAccess:
    """Read only historically admitted bytes; never claim a new runtime admission."""
    @staticmethod
    def catalog(context):
        from msfinal import materials as A
        return A.catalog(context.historical_admission)

    @staticmethod
    def array(context, *args):
        from msfinal import materials as A
        return A.array(context.historical_admission, *args)

    @staticmethod
    def weights(context, *args):
        from msfinal import materials as A
        return A.weights(context.historical_admission, *args)


def context(historical, policy, policy_sha):
    """Simple explicit context, deliberately not Admission or AcceptedRecord."""
    bindings = dict(domain='runtime_compatibility', scope='current_D2_diagnostics_only',
                    policy_sha256=policy_sha, historical_final_bindings=dict(historical.bindings),
                    candidate_runtime_payload_sha256=policy['candidate_runtime_payload_sha256'],
                    scientific_acceptance=False, independent_acceptance=False)
    return SimpleNamespace(historical_admission=historical,
                           configuration=historical.configuration, inputs=historical.inputs,
                           calibration=historical.calibration, bindings=bindings,
                           prerequisites=SimpleNamespace(
                               runtime=SimpleNamespace(sha256=policy['candidate_runtime_payload_sha256']),
                               assets=historical.prerequisites.assets,
                               clean_model_sha256=historical.prerequisites.clean_model_sha256))


def run(job, policy_sha, deadline):
    job = Path(job).resolve()
    need(sha(job/POLICY) == policy_sha, 'Compatibility policy differs')
    policy = read(job/POLICY)
    act, gate = read(job/'ACTIVATION.json'), read(job/'RUNTIME_GATE.json')
    need(time.time() < deadline <= act['deadline'] - policy['return_reserve_seconds'],
         'Fixed compatibility work/save window required')
    need(gate.get('passed') is True and gate.get('policy_sha256') == policy_sha,
         'Fresh candidate runtime gate missing')
    need(gate.get('candidate_runtime_payload_sha256') == policy['candidate_runtime_payload_sha256'],
         'Gate candidate runtime binding differs')
    need(gate.get('scope') == 'D2_diagnostic_candidate_observations_only'
         and gate.get('numerical_compatibility_verified') is False,
         'Observation gate must not self-issue numerical compatibility')
    need(gate.get('interpreter_observation') == 'observations/runtime/CANDIDATE_INTERPRETER.json',
         'Candidate interpreter observation path differs')
    need(sha(job/gate['interpreter_observation']) == gate['observed_interpreter_sha256'],
         'Candidate interpreter observation changed')
    exact(gate['runtime'], policy['expected_runtime'], 'Gate/runtime')
    need(hashlib.sha256(payload_bytes(policy['expected_runtime'])).hexdigest()
         == policy['candidate_runtime_payload_sha256'], 'Frozen candidate payload hash differs')
    need(policy['compatibility_owners'] == ['O01'] and policy['owners'] == ['O%02d' % i for i in range(1, 11)],
         'Historical/compatibility owner populations differ')
    expected_ids = ['O01/' + family + '_' + parent + '_raw320'
                    for family in FAMILIES for parent in ('M3', 'O')]
    need(policy['model_order'] == expected_ids and [r['id'] for r in policy['model_jobs']] == expected_ids,
         'Fixed complete O01 raw320 compatibility roster differs')
    need(policy['adaptive_jobs'] == [], 'Compatibility may not add adaptive jobs')

    out = job/'compatibility_output'
    out.mkdir(exist_ok=False)
    started, last, current = time.monotonic(), 0., 'historical preflight'
    completed, projections = {}, {}
    counts = {key: 0 for key in policy['operation_limits']}

    def check():
        nonlocal last
        if time.time() >= deadline:
            raise TimeoutError('Fixed compatibility deadline reached; no automatic retry')
        if shutil.disk_usage(job).free < 2 * 1024**3:
            raise RuntimeError('Less than 2 GiB local free space remains')
        now = time.monotonic()
        if now - last >= 20:
            last = now
            progress = dict(current=current, compared_models=len(completed), total_models=20,
                            seconds=round(now-started, 1))
            with (out/'progress.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(progress, sort_keys=True) + '\n')
                f.flush()
            print(json.dumps(progress), flush=True)

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

        historical = B.admit(prerequisites(job, policy, 'O01'))
        A.preflight(historical)
        spec = E.preflight(historical, 'final')
        candidate = context(historical, policy, policy_sha)
        items = {}
        for row in policy['model_jobs']:
            check()
            descriptor = row['descriptor']
            procedure = descriptor.get('procedure')
            need(row['owner'] == 'O01' and row['dependency_paths'] == descriptor.get('dependencies') == {}
                 and descriptor['wrapper_steps'] is None and isinstance(procedure, dict)
                 and set(procedure) == {'architecture', 'transform', 'variant'}
                 and procedure['transform'] is None, 'Frozen diagnostic model reconstruction differs')
            item = O.FinalModel(row['name'], job/row['model_path'], descriptor['model_sha256'],
                                accepted(job, policy, row['acceptance']), descriptor['wrapper_steps'],
                                procedure=procedure, dependencies={})
            O.final_model(item, historical)
            exact(O.model_descriptor(item), descriptor, 'Historical model descriptor')
            need(O.measurement_identity(item) == row['measurement_sha256'], 'Model measurement identity differs')
            items[row['id']] = item

        Q.configure()
        actual = plain_runtime(A.runtime_identity())
        put(out/'RUNTIME_ACTUAL.json', actual)
        exact(actual, policy['expected_runtime'], 'Fresh producer runtime')
        captured_interpreter = read(job/gate['interpreter_observation'])
        observed_interpreter = P.observed_interpreter_identity(check)
        put(out/'INTERPRETER_START.json', observed_interpreter)
        P.target_build(observed_interpreter)
        exact(observed_interpreter, captured_interpreter, 'Fresh producer interpreter versus observation')
        for name, expected_sha in policy['expected_modules'].items():
            module = importlib.import_module(name)
            need(getattr(module, '__file__', None) and sha(module.__file__, check) == expected_sha,
                 'Imported numerical module differs: ' + name)
        # Preload the same pure transport implementation used by raw960; this
        # performs no authentication or network request. Bind its SSL/HTTP
        # libraries in the compatibility evidence before candidate work.
        from msfinal.drive_transport import GoogleDriveTransfer, RefreshingMetadataToken
        start_libraries = P.libraries(check)
        put(out/'LIBRARIES_START.json', start_libraries)
        P.check_libraries(start_libraries, check)
        put(out/'INPUT_PREFLIGHT.json', dict(complete=True, selected_models=20,
            historical_admission_bindings=historical.bindings, candidate_context=candidate.bindings,
            all_model_bytes_admitted_before_first_call=True, opening_sha256=policy['opening_sha256'],
            repeated_campaign_jobs=0, compatibility_replays=20, scientific_acceptance=False))
        put(out/'START.json', dict(policy_sha256=policy_sha, scope='current_D2_diagnostics_only',
            historical_source_policy_sha256=policy.get('historical_raw320_policy_sha256'),
            candidate_context=candidate.bindings, compatibility_started=True,
            raw960_started=False, training_started=False, task_decision_started=False,
            inverse_work_started=False, native_solver_calls=0, automatic_raw960_launch_by_this_process=False, parent_automatic_continuation_requires_scoped_gate=True))

        (out/'projections').mkdir()
        store = E.ProjectionStore(out/'projections/O01')
        reference_projection_root = job/'reference/output/projections/O01'
        candidate_projection_root = out/'projections/O01'
        for row in policy['model_jobs']:
            current = row['id']
            check()
            item = items[current]
            dest = out/'diagnostics/O01'/row['name']
            dest.mkdir(parents=True, exist_ok=False)
            put(dest/'COMPATIBILITY_INTENT.json', dict(id=current, descriptor=row['descriptor'],
                candidate_context=candidate.bindings, reference_directory='reference/output/diagnostics/O01/'
                + row['name'] + '/evaluation', scientific_acceptance=False))
            # These are the SAME architecture factory and weight loader used by
            # O.load_final_model; its historical runtime gate is not bypassed.
            # This is a separate explicit compatibility use of pinned kernels.
            model = models.create(item.procedure['architecture']).to('cuda').eval()
            model.load_state_dict(T._parent_weights(item), strict=True)
            old_binding = E.evaluation_binding(historical, 'final', row['measurement_sha256'], None)
            new_binding = E.evaluation_binding(candidate, 'final', row['measurement_sha256'], None)
            E._evaluate_model(model, candidate, dest/'evaluation', check, phase='final',
                model_sha256=row['measurement_sha256'], wrapper_steps=None, projection_store=store,
                access=HistoricalAccess, spec=spec, components='diagnostics')
            proof = compare_one(job/'reference/output/diagnostics/O01'/row['name']/'evaluation',
                                dest/'evaluation', spec, old_binding, new_binding, check)
            old_key = projection_key(historical, row['measurement_sha256'], None, A)
            new_key = projection_key(candidate, row['measurement_sha256'], None, HistoricalAccess)
            projection_proof = compare_projection_pair(reference_projection_root,
                candidate_projection_root, old_key, new_key, check)
            projections[current] = projection_proof
            completed[current] = dict(descriptor=row['descriptor'], measurement_sha256=row['measurement_sha256'],
                historical_binding=old_binding, candidate_binding=new_binding, diagnostics=proof,
                projection=projection_proof, scientific_acceptance=False)
            put(dest/'EXACT_COMPARISON.json', completed[current])
            for key, n in dict(diagnostic_GPU_evaluations=1, validation_controls=512,
                               random_JVPs=1024, keyed_JVPs=512, detector_controls=128).items():
                counts[key] += n
            print(current + ': complete diagnostic rows, all4096 scores and projection tensor bits MATCH.', flush=True)
            del model
            torch.cuda.empty_cache()

        current = 'clean reference projection and final readback'
        check()
        clean_sha = historical.prerequisites.clean_model_sha256
        projections['O01/clean_reference'] = compare_projection_pair(reference_projection_root,
            candidate_projection_root, projection_key(historical, clean_sha, None, A),
            projection_key(candidate, clean_sha, None, HistoricalAccess), check)
        need(len(store.records) == len(projections) == 21, 'Projection population differs')
        need(counts == policy['operation_limits'] and len(completed) == 20, 'Compatibility operation population differs')
        verify(job, policy, check)
        exact(plain_runtime(A.runtime_identity()), actual, 'Producer runtime changed')
        final_interpreter = P.observed_interpreter_identity(check)
        put(out/'INTERPRETER_FINAL.json', final_interpreter)
        P.target_build(final_interpreter)
        exact(final_interpreter, captured_interpreter, 'Final producer interpreter versus observation')
        P.check_libraries(start_libraries, check)
        final_libraries = P.libraries(check)
        put(out/'LIBRARIES_FINAL.json', final_libraries)
        P.check_libraries(final_libraries, check)
        for name in set(start_libraries) & set(final_libraries):
            need(start_libraries[name] == final_libraries[name], 'Loaded library changed: ' + name)
        put(out/'EXACT_COMPARISON.json', dict(complete=True, exact=True,
            scope='current_D2_diagnostics_only', jobs=completed, projections=projections,
            reference_archive=policy['reference_source']['reference'], policy_sha256=policy_sha,
            independent_acceptance=False, scientific_acceptance=False, formal_custody_assurance=False))
        put(out/'RESULT.json', dict(status='D2_PYTHON31316_COMPATIBILITY_EVIDENCE_COMPLETE', complete=True,
            compatibility_evidence_complete=True, exact_comparison_passed=True,
            scope='current_D2_diagnostics_only', counts=counts, jobs=completed,
            projection_count=21, seconds=time.monotonic()-started, policy_sha256=policy_sha,
            historical_admission_bindings=historical.bindings,
            candidate_runtime_payload_sha256=policy['candidate_runtime_payload_sha256'],
            completed_campaign_jobs_added=0, raw960_started=False, automatic_raw960_launch_by_this_process=False, parent_automatic_continuation_requires_scoped_gate=True,
            final_scientific_acceptance=False, scientific_acceptance=False, independent_acceptance=False,
            canonical_worker_advanced=False, formal_custody_assurance=False, complete_programme=False,
            explicit_exclusions=['training', 'inverse tasks', 'native solvers', 'API attacks',
                                 'noise transforms', 'general future runtime admission'],
            next_step='Parent orchestration must independently verify, publish and mechanically admit this bridge before automatically dispatching the frozen raw960 roster.'))
    except BaseException:
        put(out/'FAILURE.json', dict(current=current, traceback=traceback.format_exc(), complete=False,
            completed_comparisons=completed, counts_completed_comparisons_only=counts,
            partial_job_counts='Inspect retained partial evaluation; unknown counts are not zero.',
            seconds=time.monotonic()-started, automatic_retry=False, raw960_started=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--policy-sha', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    run(args.job, args.policy_sha, args.deadline)
