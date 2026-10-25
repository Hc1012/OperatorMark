"""Scoped final-population API streams, using unchanged service/model kernels.

This adapter does not claim historical final-runtime admission or manufacture
AcceptedRecords. It validates the original requests and a separate, externally
frozen current-runtime gate. No training, derivative or native solve is called.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
import traceback

SCOPE = 'selected_API_streams_only'
POLICIES = ('cache', 'paired_deny', 'rate')
OWNERS = tuple('O%02d' % i for i in range(1, 11))


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
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def local(root, name):
    """A portable, nonsymlink file under the declared input root."""
    need(isinstance(name, str), 'Input path is not text')
    part = PurePosixPath(name)
    need(part.parts and not part.is_absolute() and '..' not in part.parts
         and '\\' not in name and ':' not in name, 'Unsafe input path: ' + name)
    root = Path(root).resolve()
    path = root.joinpath(*part.parts)
    probe = path
    while probe != root:
        need(not probe.is_symlink(), 'Input symlink: ' + name)
        probe = probe.parent
    need(path.resolve().is_relative_to(root), 'Input escaped root')
    return path


def check_file(path, expected):
    if isinstance(expected, str):
        need(sha(path) == expected, 'Frozen file differs: ' + str(path))
    else:
        need(Path(path).stat().st_size == expected['bytes']
             and sha(path) == expected['sha256'], 'Frozen file differs: ' + str(path))


def check_scope(job, policy, policy_sha, deadline):
    activation = read(job/'ACTIVATION.json')
    gate = read(job/'API_RUNTIME_GATE.json')
    need(policy['scope'] == activation.get('scope') == gate.get('scope') == SCOPE,
         'Wrong API execution scope')
    need(activation.get('policy_sha256') == gate.get('policy_sha256') == policy_sha,
         'API policy binding differs')
    need(gate.get('passed') is True and gate.get('activation_sha256') == sha(job/'ACTIVATION.json'),
         'API current-runtime gate unavailable or unbound')
    need(type(activation.get('start')) in (int, float)
         and type(activation.get('deadline')) in (int, float)
         and activation['start'] <= time.time() < deadline <= activation['deadline'],
         'API execution is outside its activated clock')
    need(activation.get('retries') == 0 and activation.get('previous_clocks_reused') is False,
         'API retry/clock reuse prohibited')
    need(type(gate.get('observed_at')) in (int, float)
         and activation['start'] <= gate['observed_at'] <= time.time(),
         'API gate observation predates activation')
    need(gate.get('scientific_acceptance') is False
         and gate.get('general_runtime_acceptance') is False,
         'API gate must remain scoped')
    return activation, gate


def verify_inputs(job, policy, check, contract):
    need(policy.get('owner_order') == list(OWNERS), 'All ten owners in fixed order required')
    source = (job/policy['source_root']).resolve()
    need(source == job.parent/'payload/code', 'Unexpected source root')
    need(len(policy['source_pins']) == 338, 'Frozen science source population differs')
    for name, expected in policy['source_pins'].items():
        check()
        check_file(local(source, name), expected)
    for name, expected in policy['files'].items():
        check()
        check_file(local(job, name), expected)
    rows = policy.get('api_jobs', [])
    expected_ids = [o+'/API_'+p+'_M3' for o in OWNERS for p in POLICIES]
    need([r.get('id') for r in rows] == expected_ids, 'API 30-stream roster/order differs')
    result = {}
    for row in rows:
        check()
        owner, label = row['id'].split('/')
        need(row['owner'] == owner and row['label'] == label and row['parent'] == 'M3'
             and row['arm'] == 'API' and row['attempts'] == 1024
             and row['policy'] in POLICIES and row['preopening'] is False,
             'API job descriptor differs')
        path = local(job, row['request_path'])
        check_file(path, row['request_sha256'])
        r = read(path)
        need(r.get('domain') == r.get('phase') == 'final' and r.get('owner') == owner
             and r.get('arm') == 'API' and r.get('attempts') == 1024
             and r.get('policy') == row['policy'] and r.get('contract') == contract,
             'Frozen API request differs')
        need(r.get('source_sha256') == policy['source_pins']['tools/msfinal/contract_arms.py'],
             'API source binding differs')
        need(r['controls_sha256'] == row['controls_sha256']['final']
             == r['files']['controls']['sha256'], 'API controls binding differs')
        need(r['received_model_sha256'] == r['files']['received']['sha256'],
             'API parent model binding differs')
        # The current adapter keeps the original historical runtime/catalog
        # bindings unchanged. Those bytes are pinned in the new policy.
        need(r['runtime_sha256'] == policy['historical_runtime_sha256'],
             'Historical API runtime identity differs')
        need(set(r['files']) == {'received', 'controls'}, 'Unexpected API request roles')
        for name, expected in r['files'].items():
            need(expected['purpose'] == ('attack' if name == 'received' else 'final'),
                 'API input purpose differs')
            check_file(local(path.parent, expected['path']), expected)
        result[row['id']] = (row, path.parent, r)
    for owner in OWNERS:
        models = {result[owner+'/API_'+p+'_M3'][2]['received_model_sha256'] for p in POLICIES}
        need(len(models) == 1, 'API policies use different parent models')
    return source, result


def run_streams(out, jobs, policy_sha, gate_sha, check, load, contracts, publish,
                on_complete, on_durable, before_stream=None):
    """Numerical calls live only in the supplied unchanged service compute.

    Factored so offline tests can exercise orchestration/failure boundaries with
    compact fake services, without loading model tensors or running a solver.
    """
    for owner in OWNERS:
        owner_root = out/'owners'/owner
        owner_root.mkdir(parents=True, exist_ok=False)
        complete = {}
        for name in POLICIES:
            member = owner+'/API_'+name+'_M3'
            row, request_root, request = jobs[member]
            if before_stream is not None:
                before_stream(member)
            check()
            stage = owner_root/row['label']
            stage.mkdir(exist_ok=False)
            put(stage/'START.json', dict(scope=SCOPE, member=member,
                original_request_sha256=row['request_sha256'], original_domain=request['domain'],
                policy_sha256=policy_sha, runtime_gate_sha256=gate_sha,
                model_sha256=request['received_model_sha256'], phase='final',
                expected_attempts=1024, native_solver_calls=0, model_updates=0,
                scientific_acceptance=False, formal_custody_assurance=False))
            # One new service client only at its predeclared model/owner/phase/
            # policy boundary. No client reset, clock pacing or replay occurs.
            inputs = load(request, request_root)
            service = contracts.APIEvaluation(stage/'service')
            measured = service.forward(request['received_model_sha256'], owner,
                'final', name, inputs['controls'], inputs['compute'], check)
            del inputs, service
            check()
            observed = contracts.api_readback(stage/'service')
            need(observed['attempts'] == observed['responses'] == 1024
                 and len(observed['streams']) == 1, 'API attempt/readback population differs')
            stream = observed['streams'][0]
            need(stream['model'] == request['received_model_sha256'] and stream['owner'] == owner
                 and stream['phase'] == 'final' and stream['policy'] == name,
                 'API readback identity differs')
            need(measured['attempts'] == len(measured['rows']) == 1024
                 and hashlib.sha256(json.dumps(measured['rows'], sort_keys=True).encode()).hexdigest()
                 == observed['response_sha256'], 'API response rows differ from readback')
            need(all(measured.get(k) is None for k in ('J', 'K', 'T', 'B2', 'B3', 'removal')),
                 'API unavailable measurement was fabricated')
            value = dict(complete=True, arm='API', domain='scoped_API', phase='final',
                scope=SCOPE, member=member, original_request_sha256=row['request_sha256'],
                original_request_domain='final', historical_runtime_admission_unchanged=True,
                policy_sha256=policy_sha, runtime_gate_sha256=gate_sha,
                result=dict(measured, readback=observed), native_solver_calls=0,
                model_updates=0, CPU_scoring_required=False, producer_closure_pending=True,
                scientific_acceptance=False, formal_custody_assurance=False)
            put(stage/'RESULT.json', value)
            result_sha = sha(stage/'RESULT.json')
            put(stage/'REVIEW.json', dict(complete=True, scope=SCOPE, member=member,
                result_sha256=result_sha, original_request_sha256=row['request_sha256'],
                runtime_gate_sha256=gate_sha, attempts=1024, readback=observed,
                J=None, K=None, T=None, B2=None, B3=None, removal=None,
                derivative_task_available=False, detector_available=False,
                response_bytes_in_full_owner_capsule=True, producer_closure_pending=True,
                scientific_acceptance=False, formal_custody_assurance=False))
            complete[member] = dict(result_sha256=result_sha, review_sha256=sha(stage/'REVIEW.json'),
                request_sha256=row['request_sha256'], attempts=1024,
                status_counts=observed['status_counts'])
            on_complete(member, complete[member])
        put(owner_root/'OWNER_RESULT.json', dict(complete=True, owner=owner, scope=SCOPE,
            jobs=complete, attempts=3072, streams=3, producer_closure_pending=True,
            scientific_acceptance=False, formal_custody_assurance=False))
        check()
        receipt = publish(owner_root)
        need(receipt.get('complete') is True and receipt.get('durable') is True
             and receipt.get('owner') == owner
             and receipt.get('owner_result_sha256') == sha(owner_root/'OWNER_RESULT.json'),
             'Owner API capsule was not durably verified')
        on_durable(owner, receipt)


def run(job, policy_sha, deadline, *, publish_owner=None):
    job = Path(job).resolve()
    need(sha(job/'API_POLICY.json') == policy_sha, 'Frozen API policy differs')
    policy = read(job/'API_POLICY.json')
    activation, gate = check_scope(job, policy, policy_sha, deadline)
    out = job/'output'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    last = started
    current = 'checking frozen inputs and runtime'
    completed, durable = {}, {}

    def check():
        nonlocal last
        if time.time() >= deadline:
            raise TimeoutError('Fixed API producer deadline reached; no automatic retry')
        now = time.monotonic()
        if now-last >= 30:
            last = now
            print(json.dumps(dict(current=current, completed_streams=len(completed),
                total_streams=30, durable_owners=len(durable), seconds=round(now-started, 1))), flush=True)

    def before_stream(member):
        nonlocal current
        current = member

    def done(member, value):
        completed[member] = value

    def saved(owner, value):
        durable[owner] = value

    try:
        source, jobs = verify_inputs(job, policy, check, policy['api_contract'])
        sys.path[:0] = [str(source/'tools'), str(source), str(job)]
        from q6_common import THREAD_ENV
        os.environ.update(THREAD_ENV)
        import q5_model as Q
        from msfinal import contract_arms as C
        import torch
        import numpy as np
        from api_runtime import verify_gate

        need(C.CONTRACTS['API'] == policy['api_contract'], 'Frozen API contract differs from pinned kernel')
        need(Path(C.__file__).resolve() == source/'tools/msfinal/contract_arms.py'
             and Path(Q.__file__).resolve() == source/'tools/q5_model.py',
             'Numerical module import resolved outside pinned source')
        Q.configure()
        live_gate = verify_gate(job, policy, policy_sha, deadline, check)
        need(live_gate == gate, 'API gate changed before producer')
        gate_sha = sha(job/'API_RUNTIME_GATE.json')
        put(out/'INPUT_PREFLIGHT.json', dict(complete=True, scope=SCOPE,
            jobs=30, owners=10, attempts=30720, all_inputs_checked_before_first_model=True,
            source_pins=policy['source_pins'], policy_sha256=policy_sha,
            runtime_gate_sha256=gate_sha, native_solver_calls=0, model_updates=0,
            scientific_acceptance=False))

        def load(request, root):
            def data(role):
                e = request['files'][role]
                p = local(root, e['path'])
                raw = p.read_bytes()
                need(len(raw) == e['bytes'] and hashlib.sha256(raw).hexdigest() == e['sha256'],
                     'API loaded input bytes changed: ' + role)
                return raw
            x = np.load(io.BytesIO(data('controls')), allow_pickle=False)
            need(x.shape == (1024, 64) and str(x.dtype) == 'float32'
                 and np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all(),
                 'API controls shape/dtype/range differs')
            model = Q.Q5FNO().to('cuda').eval()
            model.load_state_dict(torch.load(io.BytesIO(data('received')),
                map_location='cpu', weights_only=True), strict=True)
            def compute(control):
                with torch.no_grad():
                    return model(torch.tensor([control], device='cuda', dtype=torch.float32))[0].cpu().tolist()
            return dict(controls=x.tolist(), compute=compute)

        if publish_owner is None:
            from api_remote import publish_owner
        run_streams(out, jobs, policy_sha, gate_sha, check, load, C,
            lambda owner_root: publish_owner(job, policy, owner_root, deadline),
            done, saved, before_stream)
        current = 'final unchanged input/runtime readback'
        verify_inputs(job, policy, check, C.CONTRACTS['API'])
        final_gate = verify_gate(job, policy, policy_sha, deadline, check, phase='end')
        need(final_gate == gate and sha(job/'API_RUNTIME_GATE.json') == gate_sha,
             'API gate changed during work')
        need(len(completed) == 30 and len(durable) == 10, 'API programme incomplete')
        put(out/'RESULT.json', dict(complete=True, scope=SCOPE, completed_jobs=30,
            completed_streams=list(completed), jobs=completed, attempts=30720,
            durable_owners=10, owner_receipts=durable, policy_sha256=policy_sha,
            activation_sha256=sha(job/'ACTIVATION.json'), runtime_gate_sha256=gate_sha,
            opening_sha256=policy['opening_sha256'], elapsed_seconds=time.monotonic()-started,
            native_solver_calls=0, model_updates=0, CPU_scoring_required=False,
            noise_evaluations_completed=0, automatic_retries=0,
            producer_closure_pending=True, canonical_worker_advanced=False,
            scientific_acceptance=False, formal_custody_assurance=False))
        print('All 30 API streams and 30,720 attempts completed; ten owner capsules durably saved.', flush=True)
    except BaseException:
        if not (out/'FAILURE.json').exists():
            put(out/'FAILURE.json', dict(complete=False, current=current,
                traceback=traceback.format_exc(), completed=completed, durable=durable,
                elapsed_seconds=time.monotonic()-started, automatic_retries=0,
                scope=SCOPE, scientific_acceptance=False, formal_custody_assurance=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--policy-sha', required=True)
    parser.add_argument('--deadline', required=True, type=float)
    args = parser.parse_args()
    run(args.job, args.policy_sha, args.deadline)
