"""One frozen noise job per owned child; no retry or native GPU-stage scoring.

Original scientific requests and historical input admissions are reused verbatim.
A distinct prospective gate covers this final-population adapter; no historical
runtime receipt is rewritten or used as current runtime acceptance.
"""
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

SCOPE = 'selected_noise20_only'
OWNERS = ['O%02d' % i for i in range(1, 11)]
NOISE_LABELS = ('noise_small_M3', 'noise_large_M3')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def local(root, name):
    need(isinstance(name, str), 'Input path must be text')
    p = PurePosixPath(name)
    need(p.parts and not p.is_absolute() and '..' not in p.parts
         and '\\' not in name and ':' not in name, 'Unsafe input path: ' + name)
    root = Path(root).resolve()
    path = root.joinpath(*p.parts)
    q = path
    while q != root:
        need(not q.is_symlink(), 'Input symlink: ' + name)
        q = q.parent
    need(path.resolve().is_relative_to(root), 'Input escapes root')
    return path


def check_file(path, expected, check):
    check()
    need(path.is_file() and not path.is_symlink(), 'Frozen input unavailable: ' + str(path))
    if isinstance(expected, str):
        need(sha(path) == expected, 'Frozen source differs: ' + str(path))
    else:
        need(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'],
             'Frozen input differs: ' + str(path))


def verify_inputs(job, policy, member, check):
    source = (job/policy.get('source_root', '../payload/code')).resolve()
    need(source == job.parent/'payload/code', 'Pinned source location differs')
    need(len(policy['source_pins']) == 338, 'Exactly 338 unchanged scientific source pins required')
    for name, expected in policy['source_pins'].items():
        check_file(local(source, name), expected, check)
    expected_files = {}
    for group in ('files', 'input_files', 'original_files', 'marked_files', 'final_material_files'):
        for name, record in policy.get(group, {}).items():
            if name in expected_files:
                need({k:expected_files[name][k] for k in ('bytes','sha256')}
                     == {k:record[k] for k in ('bytes','sha256')}, 'Conflicting input pins')
            expected_files[name] = record
    for name, record in policy.get('aliases', {}).items():
        expected_files[name] = record['expected']
    for name, record in expected_files.items():
        check_file(local(job, name), record, check)
    rows = policy.get('noise_jobs', [])
    members = [r['id'] for r in rows]
    required = {o+'/'+n for o in OWNERS for n in NOISE_LABELS}
    need(len(members) == len(set(members)) == 20 and set(members) == required
         and member in required, 'Frozen twenty-job noise population differs')
    graph = read(job/'admission/FINAL_GRAPH.json')
    index = read(job/'admission/FINAL_CONTRACT_REQUEST_INDEX.json')
    original = {r['id']:r for r in graph['contract_jobs'] if r['arm'] == 'noise'}
    need(set(original) == required, 'Original opened noise population differs')
    selected = None
    for row in rows:
        need(all(row[k] == original[row['id']][k]
                 for k in set(row) & set(original[row['id']])),
             'Frozen noise descriptor changed')
        ix = index[row['id']]
        need(ix['contract_job'] == original[row['id']], 'Noise graph/index differ')
        need(row['request_path'] == ix['request_root']+'/CONTRACT_INPUTS.json'
             and row['request_sha256'] == ix['request_sha256'], 'Noise request changed')
        request_path = local(job, row['request_path'])
        check_file(request_path, row['request_sha256'], check)
        request = read(request_path)
        need(request['domain'] == request['phase'] == 'final'
             and request['owner'] == row['owner'] and request['arm'] == 'noise'
             and request['sd'] == row['sd'] and request['sd'] in (.0001, .001)
             and request['received_model_sha256'] == ix['parent_sha256']
             == request['files']['received']['sha256'], 'Noise request identity differs')
        need(request['source_sha256'] == policy['source_pins']['tools/msfinal/contract_arms.py'],
             'Noise source binding differs')
        for item in request['files'].values():
            check_file(local(request_path.parent, item['path']), item, check)
        if row['id'] == member:
            # The complete original graph row supplies immutable procedure
            # fields; the new policy adds only bounded orchestration metadata.
            selected = (dict(original[row['id']], **row), ix, request, request_path.parent)
    return source, selected


def accepted(job, policy, name, scientific=False):
    """Real original record bytes only; not a new acceptance issuer."""
    from msfinal.backend import AcceptedRecord
    from msfinal.scientific_admission import ScientificRecord
    from msfinal.portable_corpus import PortableCorpus
    path = local(job, name)
    data = path.read_bytes()
    expected = policy['files'][name]['sha256']
    need(hashlib.sha256(data).hexdigest() == expected, 'Historical metadata differs')
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
        material_paths={k:local(job,n) for k,n in row['material_paths'].items()},
        source_root=job.parent/'payload/code',
        assets=accepted(job, policy, row['assets'], True), assets_root=job,
        opening=accepted(job, policy, policy['opening'], True))


class HistoricalAccess:
    """Original admitted input bytes only, never a current runtime approval."""
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


def context(historical, policy, policy_sha, gate_sha, payload_sha, member):
    """Explicit numerical context, deliberately not Admission/AcceptedRecord."""
    bindings = dict(domain='final', scope=SCOPE, member=member, policy_sha256=policy_sha,
        current_runtime_gate_sha256=gate_sha, current_runtime_payload_sha256=payload_sha,
        historical_final_bindings=dict(historical.bindings),
        historical_runtime_acceptance_unchanged=True, scientific_acceptance=False,
        independent_acceptance=False)
    return SimpleNamespace(historical_admission=historical,
        configuration=historical.configuration, inputs=historical.inputs,
        calibration=historical.calibration, bindings=bindings,
        prerequisites=SimpleNamespace(runtime=SimpleNamespace(sha256=gate_sha),
            assets=historical.prerequisites.assets,
            clean_model_sha256=historical.prerequisites.clean_model_sha256))


def cpu_binding(policy, policy_sha, gate_sha, member, row):
    """Prospective D6 campaign identity; its CPU operation gets a later clock."""
    return dict(study_id=policy['study_id'], configuration_sha256=policy['configuration_sha256'],
        policy_sha256=policy_sha, campaign_sha256=policy_sha,
        opening_sha256=policy['opening_sha256'], phase_name=row['path']+'/final',
        scope=SCOPE, member=member, producer_runtime_gate_sha256=gate_sha,
        applicability_sha256=policy['runtime_evidence']['applicability']['sha256'])


def check_pending(result, noise, cpu_state):
    need(result.get('complete') is False and result.get('task_status') == cpu_state
         and result.get('diagnostics_complete') is True, 'Noise GPU/D6 state differs')
    need(result.get('domain') == 'final' and result.get('scope') == SCOPE,
         'New noise result scope differs')
    need(result['counts'] == dict(inverse_input_iterations=480, targets=0,
         native_t128_calls=0, native_p256_calls=0, validation_controls=512,
         detector_controls=128, optimizer_model_updates=0), 'Noise GPU operation counts differ')
    need(len(result['scores']) == 4096 and not result['rows']['T'] and not result['rows']['P256'],
         'Noise detector/scoring population differs')
    need(noise['calls'] == 512 and noise['retained_outputs'] == 16384
         and noise['white_box_only'] is True, 'Noise retained output population differs')


def run(job, policy_sha, deadline, member):
    job = Path(job).resolve()
    need(sha(job/'NOISE_POLICY.json') == policy_sha, 'Noise policy differs')
    policy = read(job/'NOISE_POLICY.json')
    activation = read(job/'ACTIVATION.json')
    need(policy['scope'] == activation.get('scope') == SCOPE
         and activation.get('policy_sha256') == policy_sha
         and activation.get('retries') == 0 and activation.get('previous_clocks_reused') is False,
         'Noise activation identity/scope/retry differs')
    need(activation['start'] <= time.time() < deadline <= activation['deadline'],
         'Noise job is outside fixed activated clock')
    member_path = local(job/'output/jobs', member)
    need(member in {r['id'] for r in policy['noise_jobs']}, 'Unknown noise job')
    member_path.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); last = started; current = 'preflight'

    def check():
        nonlocal last
        if time.time() >= deadline:
            raise TimeoutError('Fixed noise job deadline reached; no automatic retry')
        now = time.monotonic()
        if now-last >= 20:
            last = now
            if shutil.disk_usage(job).free < 2*1024**3:
                raise RuntimeError('Less than 2 GiB local free space remains')
            print(json.dumps(dict(member=member, current=current,
                seconds=round(now-started,1))), flush=True)

    try:
        source, (row, index, request, request_root) = verify_inputs(job, policy, member, check)
        sys.path[:0] = [str(source/'tools'), str(source), str(job)]
        from q6_common import THREAD_ENV
        os.environ.update(THREAD_ENV)
        from noise_runtime import verify_gate
        from msfinal import backend as B, materials as A, evaluation as E
        from msfinal import contract_arms as C, final_contracts as FC, cpu_scoring as CPU
        from noise_evaluate import evaluate_noise, SOURCE_EVALUATION_SHA256
        from noise_cpu_export import export_saved_designs, preflight_export
        need(SOURCE_EVALUATION_SHA256 == policy['source_pins']['tools/msfinal/evaluation.py'],
             'Scoped evaluator derived from different frozen source')
        need(Path(C.__file__).resolve() == source/'tools/msfinal/contract_arms.py',
             'Noise kernel loaded outside pinned source')
        gate = verify_gate(job, policy, policy_sha, deadline, check, phase='start', member=member)
        gate_sha = sha(job/'NOISE_RUNTIME_GATE.json')
        need(gate.get('passed') is True and gate.get('scope') == SCOPE
             and gate.get('policy_sha256') == policy_sha
             and gate.get('activation_sha256') == sha(job/'ACTIVATION.json'),
             'Noise runtime gate binding differs')
        payload_sha = gate['runtime_payload_sha256']
        historical = B.admit(prerequisites(job, policy, row['owner']))
        A.preflight(historical); spec = E.preflight(historical, 'final')
        req = FC.Request(request_root, row['request_sha256'],
            accepted(job, policy, index['input_acceptance']))
        bound = FC.request_metadata(req, index['contract_job'], historical,
            index['parent_sha256'], 'final')
        need(bound == request and request['contract'] == C.CONTRACTS['noise'],
             'Original noise request admission differs')
        scoped = context(historical, policy, policy_sha, gate_sha, payload_sha, member)
        provenance = dict(scope=SCOPE, member=member, policy_sha256=policy_sha,
            original_request_sha256=row['request_sha256'], original_request_domain=request['domain'],
            current_runtime_gate_sha256=gate_sha, current_runtime_payload_sha256=payload_sha,
            historical_runtime_record_sha256=historical.prerequisites.runtime.sha256,
            historical_runtime_payload_sha256=historical.inputs['materials']['runtime']['sha256'],
            opening_sha256=policy['opening_sha256'], original_history_modified=False,
            general_runtime_acceptance=False, scientific_acceptance=False)
        binding = cpu_binding(policy, policy_sha, gate_sha, member, row)
        preflight_export(historical, check, job=job, policy=policy, policy_sha=policy_sha,
            member=member, binding=binding, model_sha256=request['received_model_sha256'],
            wrapper_steps=None)
        put(member_path/'JOB_INTENT.json', dict(member=member, complete=False,
            request_sha256=row['request_sha256'], binding=binding, provenance=provenance,
            original_draw_rule=C.NOISE_DRAW_RULE, expected_retained_forwards=512,
            expected_retained_outputs=16384, native_solver_calls=0, model_updates=0))
        check(); current = 'GPU evaluation and retained noise outputs'
        store = E.ProjectionStore(member_path/'projections')
        # The original loader verifies actual parent/stream bytes and returns a
        # historical public evaluate closure; that closure is intentionally unused.
        data = FC.Loader(historical, member_path, check, store=store)(request, request_root)
        model = C.noise_model(data['received'], request['sd'], data['seed'],
            member_path/'noise', binding=request['noise_binding'], check=check)
        def export(out, result, check):
            return export_saved_designs(out, result, historical, check, job=job,
                policy=policy, policy_sha=policy_sha, member=member, binding=binding,
                model_sha256=request['received_model_sha256'], wrapper_steps=None)
        result = evaluate_noise(model, scoped, member_path/'evaluation', check,
            phase='final', model_sha256=request['received_model_sha256'], wrapper_steps=None,
            projection_store=store, access=HistoricalAccess, spec=spec,
            export=export, provenance=provenance)
        current = 'retained noise and saved-design readback'; check()
        noise = C.noise_readback(member_path/'noise', check,
            expected_calls=512, expected_controls=16)
        check_pending(result, noise, CPU.STATE)
        need(model.calls == 512, 'Noise forward ordinal differs')
        CPU.pending_readback(member_path/'evaluation', check)
        put(member_path/'NOISE_READBACK.json', noise)
        put(member_path/'GPU_READBACK.json', dict(complete=True, member=member, scope=SCOPE,
            result_sha256=sha(member_path/'evaluation/RESULT.json'),
            request_sha256=sha(member_path/'evaluation/D6_REQUEST.json'),
            noise_readback_sha256=sha(member_path/'NOISE_READBACK.json'),
            model_sha256=request['received_model_sha256'],
            saved_designs=64, inverse_batch_iterations=480, native_solver_calls=0,
            model_updates=0, task_scoring_pending=True, scientific_acceptance=False))
        current = 'final input/runtime readback'
        verify_inputs(job, policy, member, check)
        final_gate = verify_gate(job, policy, policy_sha, deadline, check, phase='end', member=member)
        need(final_gate == gate and sha(job/'NOISE_RUNTIME_GATE.json') == gate_sha,
             'Noise runtime gate changed during job')
        check()
        put(member_path/'JOB_RESULT.json', dict(complete=True, scope=SCOPE, member=member,
            policy_sha256=policy_sha, runtime_gate_sha256=gate_sha,
            activation_sha256=sha(job/'ACTIVATION.json'),
            request_sha256=row['request_sha256'], GPU_evaluation_complete=True,
            noise_readback_complete=True, noise=noise, task_scoring_pending=True,
            evaluation_result_sha256=sha(member_path/'evaluation/RESULT.json'),
            D6_request_sha256=sha(member_path/'evaluation/D6_REQUEST.json'),
            GPU_readback_sha256=sha(member_path/'GPU_READBACK.json'),
            noise_readback_sha256=sha(member_path/'NOISE_READBACK.json'),
            saved_designs=64, inverse_batch_iterations=480,
            native_solver_calls=0, model_updates=0, automatic_retries=0,
            elapsed_seconds=time.monotonic()-started, producer_closure_pending=True,
            scientific_acceptance=False, formal_custody_assurance=False,
            physical_qualification=False, canonical_worker_advanced=False))
        print(member+': GPU/noise readback complete; 64 designs saved for CPU scoring.', flush=True)
    except BaseException:
        if not (member_path/'FAILURE.json').exists():
            put(member_path/'FAILURE.json', dict(complete=False, member=member, scope=SCOPE,
                current=current, traceback=traceback.format_exc(), automatic_retries=0,
                elapsed_seconds=time.monotonic()-started, original_history_modified=False,
                scientific_acceptance=False, formal_custody_assurance=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--policy-sha', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--member', required=True)
    args = parser.parse_args()
    run(args.job, args.policy_sha, args.deadline, args.member)
