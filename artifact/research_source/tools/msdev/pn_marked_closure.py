"""One prospective, parent-bound PN marked development closure. Import is inert.

No reservation, setup, tensor loading or numerical operation occurs on import.
Existing numerical kernels, task evaluator, detector and Q5 ownership are reused.
"""
from pathlib import Path
import json
import math
import os
import time
from .records import read, write, sha, verify, progress
from . import rules

OUTPUT = 'OperatorMark_PN_Marked_Development_Closure_20261005'
ATTACHED, WORK, CLEANUP, FINALIZE, SETUP = 3600, 1800, 30, 120, 1380
ENVELOPE = WORK + CLEANUP + FINALIZE
ARMS = tuple((f'Aord_PN_D{i+1}', i, 1920, 'forward32') for i in range(3)) + (
    ('A5B3_cost', 0, 200, 'A5B3'),)
LIMITS = dict(model_updates=5960, inverse_input_updates=360, native_forward_calls=96,
              endpoint_evaluations=3, detector_grids=3, new_B2_distances=9)


def finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def detector(values, theta, delta):
    """Frozen intact theta AND current attacked wrong-key separation; no recalibration."""
    if (not isinstance(values, list) or len(values) != 4096 or
            not all(finite(v) for v in values) or not finite(theta) or
            not finite(delta) or delta < 0):
        return dict(status='unavailable', detected=None, reason='missing/nonfinite detector evidence')
    wrong = max(map(abs, values[1:]))
    return dict(status='complete', detected=values[0] > theta and values[0] > wrong + delta,
                true_score=values[0], fixed_theta=theta, wrong_max=wrong, delta=delta,
                above_fixed_theta=values[0] > theta, above_attacked_null=values[0] > wrong + delta)


def utility(metrics, clean):
    gates = {k: rules.gate(metrics.get(k), clean.get(k)) for k in ('F', 'J', 'K', 'T')}
    available = all(g['status'] == 'complete' for g in gates.values())
    return dict(gates=gates, passed=all(g['passed'] is True for g in gates.values()) if available else None)


def cache_identity(files, candidate, owner):
    # A cache is specific to both tensors, the exact controls/directions and code.
    return dict(candidate=candidate, owner=owner, files={n: files[n] for n in (
        'validation_x.npy', 'validation_v.npy', 'validation_controls_f64.npy',
        'public_basis.npy', 'code/msdev/scoring.py', 'code/msdev/kernels.py',
        'code/msdev/rules.py', 'code/q5_model.py')})


def require_cache(actual, expected):
    if actual != expected:
        raise ValueError('cache source/model/control correspondence differs')


def preflight(bundle, manifest, check=lambda:None):
    bundle = Path(bundle)
    if sha(bundle/'MANIFEST.json') != manifest:
        raise ValueError('upload manifest differs')
    files = read(bundle/'MANIFEST.json')['files']
    for name,h in files.items():
        check()
        if sha(bundle/name,check)!=h:raise ValueError('input/source identity mismatch: '+name)
    p = read(bundle/'CLOSURE.json')
    for key,path in [('PN_output','PN_OUTPUT_MANIFEST.json'),('U_output','U_OUTPUT_MANIFEST.json'),
                     ('U_input','U_INPUT_MANIFEST.json'),('original_stop','ORIGINAL_STOP.json')]:
        if sha(bundle/'evidence'/path)!=p['parents'][key]:raise ValueError('parent evidence binding differs')
    original_stop=read(bundle/'evidence/ORIGINAL_STOP.json')
    if original_stop['state']!='STOPPED' or original_stop['failure_kind']!='scientific' or original_stop['active']!='mark_selection':
        raise ValueError('original scientific stop not preserved')
    if p['arms'] != [list(a) for a in ARMS] or p['limits'] != LIMITS:
        raise ValueError('prescribed arms/counts differ')
    if p['seconds'] != dict(overall=ATTACHED, worker=WORK, cleanup=CLEANUP, finalization=FINALIZE, setup=SETUP):
        raise ValueError('fixed prospective window differs')
    if p['A5B3_start'] != 'PN_B3_3_D1' or p['retries'] != 0 or p['resume'] is not False:
        raise ValueError('cost starting model or no-retry rule differs')
    original = read(bundle/'evidence/ORIGINAL_INPUT_MANIFEST.json')['files']
    for name in p['unchanged_original_inputs']:
        if files[name] != original[name]:
            raise ValueError('original attack/data/runtime input changed: '+name)
    pm = read(bundle/'evidence/PN_OUTPUT_MANIFEST.json')['files']
    um = read(bundle/'evidence/U_OUTPUT_MANIFEST.json')['files']
    for local, source in p['PN_products'].items():
        if files[local] != pm[source]:
            raise ValueError('PN producer correspondence differs: '+local)
    for local, source in p['U_products'].items():
        if files[local] != um[source]:
            raise ValueError('unmarked producer correspondence differs: '+local)
    cal = read(bundle/'calibration.json')
    for i in range(3):
        row = p['owners'][i]
        product = read(bundle/f'intact/PN_B3_3_D{i+1}/PRODUCT.json')
        q = product['qualification']
        if (product['owner'] != i or product['id'] != f'PN_B3_3_D{i+1}' or
                product['factor'] != 3 or product['updates'] != 320 or q['qualified'] is not True or
                q['complete'] is not True or not finite(q['theta']) or
                row['theta'] != q['theta'] or row['delta'] != cal['deltas']['B3'] or
                files[row['model']] != product['model_sha256']):
            raise ValueError('selected PN checkpoint/qualification/threshold differs')
        projection = read(bundle/f'baseline/D{i+1}_projection.json')
        if projection['model'] != files[f'D{i+1}.pt'] or projection['projection'] != files[f'baseline/D{i+1}_projection.pt']:
            raise ValueError('clean owner projection correspondence differs')
    baseline = read(bundle/'BASELINE.json')
    cache = read(bundle/'B2_CACHE_BINDING.json')
    raw = read(bundle/'reuse/unmarked_B2_distances.json')
    keys = [(r['candidate'], r['owner']) for r in raw]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate historical distance')
    for row in cache['pairs']:
        a, b = row['candidate'], row['owner']
        # Model identities are producer-bound even though immutable old models are
        # omitted from this minimal bundle when no new execution needs them.
        expected = cache_identity(files, p['reused_model_hashes'][a], files[b+'.pt'])
        require_cache(row['identity'], expected)
        match = [r for r in raw if (r['candidate'], r['owner']) == (a, b)]
        if len(match) != 1 or rules.complete_mean(match[0]['rows']) != match[0]['mean']:
            raise ValueError('historical distance rows/mean differ')
    if len(cache['pairs']) != 9 or baseline['owners'].keys() != {'0', '1', '2'}:
        raise ValueError('incomplete clean-owner/null population')
    if read(bundle/'reuse/replacement_selection.json')['selected'] != 10240:
        raise ValueError('replacement selection changed')
    if 'runtime_migration' in p:
        from .pn_runtime_migration import require_receipt
        require_receipt(bundle,p)
    return p


def reserve(bundle, root, manifest, credits=None, rate=None, approved=False, clock=time.time):
    """Only invoked after separate approval on CPU. Re-entry never resets time."""
    p = preflight(bundle, manifest)
    root = Path(root)
    if root.name != OUTPUT or approved is not True:
        raise ValueError('explicit full-scope approval and exact separate output required')
    if not finite(credits) or not finite(rate) or rate <= 0 or credits < rate:
        raise ValueError('current credits/rate must cover the full one-hour window')
    now = clock()
    if root.exists():
        s = read(root/'STATUS.json')
        if (s['bindings']['upload'] != manifest or s['deadline'] != s['session']['start']+ATTACHED or
                s['session']['closed'] or s['session']['launch_started'] or s['state']=='STOPPED' or now >= s['deadline']):
            raise ValueError('used/expired/mismatched reservation; no retry, resume or reset')
        return s
    s = dict(state='UNSTARTED', updates=0, bindings=dict(upload=manifest, parents=p['parents']),
             deadline=now+ATTACHED, historical_allowances_reused=False,
             approval=dict(full_scope=True, credits=credits, rate_per_hour=rate),
             session=dict(start=now, end=now+ATTACHED, prebooked=True, closed=False, launch_started=False))
    root.mkdir()
    write(root/'STATUS.json', s)
    return s


def setup(bundle, root, environment):
    """One-use wrapper around the unchanged, independently supervised Q6 setup."""
    from .colab_environment import setup as existing_setup
    root = Path(root); s = read(root/'STATUS.json')
    if s['session'].get('setup_started') or s['session']['closed'] or s['session']['launch_started']:
        raise ValueError('setup already attempted or session consumed; no retry')
    if time.time()+ENVELOPE >= s['deadline']:
        raise TimeoutError('no remaining full worker envelope')
    s['session']['setup_started'] = True; write(root/'STATUS.json',s,replace=True)
    started = time.time()
    try:
        plan=read(Path(bundle)/'CLOSURE.json')
        if 'runtime_migration' in plan:
            from .pn_runtime_migration import setup_closure_environment
            python=setup_closure_environment(bundle,root,environment)
        else:
            python = existing_setup(bundle,root,environment)
        if time.time()-started > SETUP or time.time()+ENVELOPE > s['deadline']:
            raise TimeoutError('setup/full remaining envelope exceeded')
        return python
    except BaseException as exc:
        s = read(root/'STATUS.json');s.update(state='STOPPED',reason=repr(exc))
        write(root/'STATUS.json',s,replace=True)
        write(root/'SETUP_FAILURE.json',dict(error=repr(exc),seconds=time.time()-started,retry_authorized=False))
        raise


def append(path, row):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')
        f.flush()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else []


def counts(root, result, check=lambda: None):
    """Closed-child readback of committed work; missing in-flight work is unknown."""
    root = Path(root); total = 0; done = 0; evaluations = 0; inverse = 0; grids = 0
    for name, owner, steps, recipe in ARMS:
        trace = rows(root/name/'updates.jsonl')
        if len(trace) > steps:
            raise ValueError('too many updates')
        for i, r in enumerate(trace, 1):
            check()
            if r['update'] != i or r['pre_state'] != i-1 or r['arm'] != name:
                raise ValueError('update index/order differs')
        total += len(trace)
        if (root/name/'PRODUCT.json').exists():
            pr = read(root/name/'PRODUCT.json')
            if len(trace) != steps or pr['updates'] != steps or pr['recipe'] != recipe:
                raise ValueError('product count differs')
            verify(root/name, pr['files']); done += 1
        if (root/name/'evaluation/RESULT.json').exists():
            e = read(root/name/'evaluation/RESULT.json')
            if recipe != 'forward32' or e['inverse_steps'] != 120 or len(e['inverse_trace']) != 120:
                raise ValueError('evaluation step count differs')
            for i in range(16):
                if not (root/name/f'evaluation/task_{i:02d}.json').exists():
                    raise ValueError('missing task readback')
            binding = read(root/name/'EVALUATION_BINDING.json')
            if binding['model_sha256'] != sha(root/name/'model.pt'):
                raise ValueError('evaluation model binding differs')
            verify(root/name/'evaluation',binding['files'])
            inverse += len(e['inverse_trace']); evaluations += 1
        if (root/name/'DETECTOR.json').exists():
            d = read(root/name/'DETECTOR.json')
            if recipe != 'forward32' or len(d['scores']) != 4096:
                raise ValueError('detector population differs')
            if detector(d['scores'], d['theta'], d['delta']) != d['decision']:
                raise ValueError('detector decision differs')
            if d['projection_sha256'] != sha(root/name/'projection.pt'):
                raise ValueError('detector projection differs')
            grids += 1
    calls = rows(root/'native_calls.jsonl')
    if len(calls) > 96 or any(r['ordinal'] != i for i, r in enumerate(calls, 1)):
        raise ValueError('native charge order/cap differs')
    distances = rows(root/'B2_new.jsonl')
    if len(distances) > 9 or len({(r['candidate'], r['owner']) for r in distances}) != len(distances):
        raise ValueError('B2 distance population differs')
    observed = dict(model_updates=total, inverse_input_updates=inverse, native_forward_calls=len(calls),
                    endpoint_evaluations=evaluations, detector_grids=grids, new_B2_distances=len(distances))
    return dict(observed=observed, completed_arms=done,
                matches=bool(result and result.get('counts') == observed),
                exact=observed == LIMITS and done == 4,
                in_flight_work_unknown=not bool(result and result.get('complete')),
                note='Native calls charged before entry; inverse steps proven by completed traces. Incomplete work is never refunded/retried.')


def closed_receipt(r):
    c = (r or {}).get('closure', {})
    return bool(c.get('resolved') and c.get('child_reaped') and not c.get('errors') and
                not c.get('survivors') and r.get('stdout_closed'))


def launch(python, bundle, root, manifest):
    from .cli import module
    from .colab_environment import owned
    from q6_common import THREAD_ENV
    root, bundle = Path(root), Path(bundle)
    preflight(bundle, manifest)
    s = read(root/'STATUS.json'); now = time.time()
    if s['bindings']['upload'] != manifest or s['session']['closed'] or s['session']['launch_started'] or s['state']!='UNSTARTED':
        raise ValueError('unused same-candidate window required')
    if s['deadline'] != s['session']['start']+ATTACHED or now+ENVELOPE > s['deadline']:
        raise TimeoutError('full 1950-second remaining envelope required')
    # Setup receipt must bind this interpreter, requirements and window.
    env = read(root/s['session']['environment_receipt'])
    if (str(python) != env['python'] or env['manifest_sha256'] != manifest or
            env['requirements_sha256'] != sha(bundle/'requirements-colab.txt') or
            env['session_start'] != s['session']['start'] or env['seconds'] > SETUP):
        raise ValueError('bound setup/interpreter differs')
    s['session']['launch_started'] = True
    write(root/'STATUS.json', s, replace=True)
    write(root/'CONSUMED.json', dict(start=now, work_deadline=now+WORK, closure_deadline=now+WORK+CLEANUP,
          session_deadline=s['deadline'], manifest_sha256=manifest, limits=LIMITS, retries=0))
    owner = module(bundle/'code/q5_colab_supervisor.py', 'pn_closure_owned')
    receipt = None; outcome = None; error = None
    try:
        receipt = owned([str(python), '-B', '-m', 'msdev.pn_marked_closure', '--bundle', str(bundle), '--root', str(root)],
            WORK, now+WORK+CLEANUP, root/'supervisor', owner, cwd=bundle/'code', acceptable=(0,4),
            env=dict(os.environ, **THREAD_ENV, CUBLAS_WORKSPACE_CONFIG=':4096:8', WANDB_MODE='disabled', WANDB_DISABLED='true'))
    except BaseException as exc:
        error = repr(exc)
    finally:
        if receipt is None and (root/'supervisor/RESULT.json').exists():
            receipt = read(root/'supervisor/RESULT.json')
        closed = closed_receipt(receipt)
        end = min(s['deadline'], time.time()+FINALIZE)
        def check():
            if time.time() >= end:
                raise TimeoutError('finalization limit; preserve partial evidence')
        result = read(root/'RESULT.json') if (root/'RESULT.json').exists() else None
        cr = None
        try:
            if closed:
                cr = counts(root, result, check)
                write(root/'COUNTS_READBACK.json', cr)
                if cr['exact'] and result and result.get('complete'):
                    if readout(bundle,root,preflight(bundle,manifest,check),persist=False) != read(root/'READOUT.json'):
                        raise ValueError('independent descriptive readback differs')
        except BaseException as exc:
            error = repr(exc) if error is None else error+'; '+repr(exc)
        complete = bool(closed and cr and cr['exact'] and cr['matches'] and result and result.get('complete')
                        and receipt.get('returncode') == 0 and not receipt.get('timed_out') and not error)
        outcome = dict(complete=complete, closed=closed, timed_out=(receipt or {}).get('timed_out'), error=error,
                       finalization_deadline=end, overall_deadline=s['deadline'],
                       counts=cr, worker=result, supervisor=receipt, follow_on_authorized=False)
        write(root/'OUTCOME.json', outcome)
        s = read(root/'STATUS.json'); s['session']['closed'] = closed
        s.update(state='COMPLETE' if complete else 'STOPPED', updates=(cr or {}).get('observed', {}).get('model_updates'),
                 reason=error or (result or {}).get('error'), analysis_available=(result or {}).get('analysis_available'))
        write(root/'STATUS.json', s, replace=True)
        if closed:
            files = {p.relative_to(root).as_posix(): sha(p, check) for p in root.rglob('*') if p.is_file()}
            check(); write(root/'OUTPUT_MANIFEST.json', dict(files=files, complete=complete, kind='PN-marked-development-closure'))
    return outcome


def attack_engine(K, data, weights, seeds, owner, device, basis=None, cost=False, **kw):
    """No restore: fresh Adam and exact original initialization/shuffle selection."""
    return K.Engine(data, seeds['owners'][owner]['initialization'],
        seeds['owners'][owner]['shuffle'] if cost else seeds['ordinary_shuffle'], device,
        recipe='A5B3' if cost else 'forward32', weights=weights, basis=basis, **kw)


class NumericalBackend:
    def __init__(self, bundle, root, plan, check):
        import numpy as np
        import torch
        import q5_model as M
        from . import kernels as K, scoring
        from .cli import runtime, reusable_q6_runtime
        self.np, self.torch, self.M, self.K, self.scoring = np, torch, M, K, scoring
        self.bundle, self.root, self.plan, self.check = bundle, root, plan, check
        self.engine = None; self.active = None
        M.configure(); env = runtime()
        if 'runtime_migration' in plan:
            from .pn_runtime_migration import validate_closure_runtime
            receipt=validate_closure_runtime(bundle,plan,env)
            expected=receipt['runtime']['environment']
        else:
            reusable_q6_runtime(bundle, env);expected=plan['runtime']
        if env != expected or not torch.cuda.is_available():
            raise ValueError('exact saved A100 runtime required')
        flags = dict(environment=env, deterministic=torch.are_deterministic_algorithms_enabled(),
                     default_dtype=str(torch.get_default_dtype()), matmul=torch.get_float32_matmul_precision(),
                     tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                     autocast=torch.is_autocast_enabled('cuda'))
        write(root/'RUNTIME.json', flags)
        if not flags['deterministic'] or flags['default_dtype'] != 'torch.float32' or flags['matmul'] != 'highest' or flags['tf32'] or flags['cudnn_tf32'] or flags['autocast']:
            raise ValueError('precision/determinism flags differ')
        self.seeds = read(bundle/'CLOSURE_SEEDS.json')
        self.files = read(bundle/'MANIFEST.json')['files']
        self.basis = torch.from_numpy(self.load('public_basis.npy')).to('cuda')

    def load(self, name):
        return self.np.load(self.bundle/name, allow_pickle=False)

    def model(self, owner=None, attack=None):
        path = self.bundle/f'D{owner+1}.pt' if attack is None else self.root/attack/'model.pt'
        m = self.M.Q5FNO().to('cuda')
        m.load_state_dict(self.torch.load(path, map_location='cpu', weights_only=True), strict=True)
        if m.operator.fno_block_precision != 'full' or {p.dtype for p in m.parameters()} != {self.torch.float32, self.torch.complex64}:
            raise ValueError('FNO architecture/precision differs')
        return m.eval()

    def train(self, name, owner, steps, recipe):
        torch = self.torch; cost = recipe == 'A5B3'; self.active = name
        if cost:
            data = tuple(torch.from_numpy(self.load('training_'+n+'.npy')).to('cuda').repeat((4,1)) for n in ('x','v','y','dy'))
        else:
            x, y = (torch.from_numpy(self.load('attack_'+n+'.npy')).to('cuda') for n in ('x','y'))
            data = (x, torch.zeros_like(x), y, torch.zeros_like(y))
        if len(data[0]) != (16384 if cost else 256):
            raise ValueError('original data population differs')
        weights = torch.load(self.bundle/self.plan['owners'][owner]['model'], map_location='cpu', weights_only=True)
        e = attack_engine(self.K, data, weights, self.seeds, owner, 'cuda', self.basis.float(), cost)
        self.engine = e
        if e.updates != 0 or e.optimizer.state or e.sampler.batch != (16 if cost else 32):
            raise ValueError('fresh attack optimizer/sampling required')
        arm = self.root/name; arm.mkdir(); timings = []; started = time.time()
        # Existing original first-use timing window is retained, no warm-up extras.
        phase_deadline = started + 300
        write(arm/'INITIALIZATION.json', dict(model_sha256=self.files[self.plan['owners'][owner]['model']],
              optimizer='fresh Adam 0.002, default betas/eps, no embedding-state restore',
              seed_source='owner initialization / owner shuffle' if cost else 'owner initialization / ordinary_shuffle',
              recipe=recipe, batch=e.sampler.batch, data_rows=len(data[0]), updates=0))
        for step in range(steps):
            self.check()
            if time.time() >= phase_deadline:
                raise TimeoutError('attack/cost first-use phase ceiling')
            torch.cuda.synchronize(); t = time.monotonic()
            last = e.step(.002); torch.cuda.synchronize(); timings.append(time.monotonic()-t)
            append(arm/'updates.jsonl', dict(last, arm=name, pre_state=step))
            if not cost and step+1 in (320, 960):
                torch.save(e.state(), arm/f'state_{step+1}.pt')
            if (step+1)%32 == 0 or step+1 == steps:
                progress(self.root, 'training', arm=name, updates=step+1, target=steps, loss=last['loss'])
        torch.save(self.M.tensor_state(e.model), arm/'model.pt'); torch.save(e.state(), arm/'state.pt')
        product = dict(id=name, owner=owner, recipe=recipe, updates=steps, seconds=time.time()-started,
                       warmup_updates=20, sustained_updates=steps-20, timing_seconds=timings,
                       sustained_seconds_per_update=sum(timings[20:])/(steps-20), cost_only=cost,
                       data_note='4096 existing development rows repeated four times, not independent new samples' if cost else 'original 256 forward pairs',
                       files={p.name:sha(p) for p in arm.iterdir() if p.is_file()})
        write(arm/'PRODUCT.json', product)
        self.engine = None; del e, data

    def evaluate(self, name, owner):
        from q6_common import true_field
        torch = self.torch; end = time.time()+180
        def check():
            self.check()
            if time.time() >= end: raise TimeoutError('endpoint evaluation ceiling')
        with self.np.load(self.bundle/f'keys_B3_{owner}.npz', allow_pickle=False) as z:
            direction = torch.from_numpy(z['v'][0].copy())
        def oracle(rho, p1):
            check(); count = len(rows(self.root/'native_calls.jsonl'))
            if count >= 96: raise ValueError('native-call cap')
            append(self.root/'native_calls.jsonl', dict(ordinal=count+1, arm=name, P256=p1, charged_before_entry=True))
            return true_field(rho, p1)
        result = self.scoring.evaluate(self.model(attack=name), self.bundle, direction,
            self.load(f'key_truth_{owner}.npy'), self.root/name/'evaluation', check, oracle)
        check()
        write(self.root/name/'EVALUATION_BINDING.json', dict(model_sha256=sha(self.root/name/'model.pt'),
              files={p.name:sha(p) for p in (self.root/name/'evaluation').iterdir() if p.is_file()}))
        return result

    def detect(self, name, owner):
        torch = self.torch; start = time.time()
        def check():
            self.check()
            if time.time()-start >= 180: raise TimeoutError('detector ceiling')
        projected = self.scoring.projections(self.model(attack=name),
            torch.from_numpy(self.load('validation_x.npy')).to('cuda'), self.basis, check=check)
        torch.save(projected, self.root/name/'projection.pt')
        reference = torch.load(self.bundle/f'baseline/D{owner+1}_projection.pt', map_location='cpu', weights_only=True)
        with self.np.load(self.bundle/f'keys_B3_{owner}.npz', allow_pickle=False) as z:
            coeff, v = (torch.from_numpy(z[n].copy()) for n in ('coeff','v'))
        scores = self.scoring.projected_scores(projected, reference,
            torch.from_numpy(self.load('validation_controls_f64.npy').reshape(-1,64)), coeff, v, self.basis,
            math.sqrt(self.plan['owners'][owner]['normalizers'][1]), 'B3')
        theta, delta = (self.plan['owners'][owner][k] for k in ('theta','delta'))
        check()
        write(self.root/name/'DETECTOR.json', dict(scores=scores, theta=theta, delta=delta,
            intact_qualification_sha256=self.files[f'intact/PN_B3_3_D{owner+1}/PRODUCT.json'],
            decision=detector(scores,theta,delta), seconds=time.time()-start,
            projection_sha256=sha(self.root/name/'projection.pt'),
            identity=cache_identity(self.files,sha(self.root/name/'model.pt'),self.files[f'D{owner+1}.pt'])))

    def distances(self):
        torch = self.torch; start = time.time()
        def check():
            self.check()
            if time.time()-start >= 300: raise TimeoutError('B2 distance ceiling')
        x = torch.from_numpy(self.load('validation_x.npy')).to('cuda').repeat_interleave(2,0)
        v = torch.from_numpy(self.load('validation_v.npy')).to('cuda')
        for name, _, _, _ in ARMS[:3]:
            for i in range(3):
                check(); value = self.scoring.distance(self.model(attack=name), self.model(owner=i), x, v, check)
                append(self.root/'B2_new.jsonl', dict(candidate=name,owner=f'D{i+1}',**value,
                    identity=cache_identity(self.files,sha(self.root/name/'model.pt'),self.files[f'D{i+1}.pt'])))
                progress(self.root,'B2_distance',candidate=name,owner=i,mean=value['mean'])

    def partial(self):
        if self.engine is not None:
            self.torch.save(self.engine.state(), self.root/(self.active+'.partial.pt'))


def readout(bundle, root, plan, persist=True):
    baseline = read(bundle/'BASELINE.json'); previous = read(bundle/'reuse/unmarked_analysis.json')
    reused = read(bundle/'reuse/unmarked_B2_distances.json'); fresh = rows(root/'B2_new.jsonl')
    files = read(bundle/'MANIFEST.json')['files']
    for row in fresh:
        require_cache(row['identity'], cache_identity(files,sha(root/row['candidate']/'model.pt'),files[row['owner']+'.pt']))
    distances = {(r['candidate'],r['owner']):r['mean'] for r in reused+fresh}
    rows_out = []; cohort = []
    replacement = {n:read(bundle/f'reuse/{n}_EVALUATION.json')['metrics'] for n in ('Rmain_1920','Rstar_10240')}
    for name, i, _, _ in ARMS[:3]:
        metrics = read(root/name/'evaluation/RESULT.json')['metrics']
        gates = utility(metrics, baseline['owners'][str(i)]['metrics'])
        saved = read(root/name/'DETECTOR.json')
        theta,delta = (plan['owners'][i][k] for k in ('theta','delta'))
        if saved['theta'] != theta or saved['delta'] != delta or saved['intact_qualification_sha256'] != files[f'intact/PN_B3_3_D{i+1}/PRODUCT.json']:
            raise ValueError('attacked detector changed intact threshold binding')
        require_cache(saved['identity'],cache_identity(files,sha(root/name/'model.pt'),files[f'D{i+1}.pt']))
        d = detector(saved['scores'],theta,delta)
        if d != saved['decision']:raise ValueError('attacked detector decision differs')
        own = f'D{i+1}'
        b2 = rules.b2(distances[(name,own)], [distances[(n,own)] for n,_,_,_ in ARMS[:3] if n != name],
                      [distances[(f'D{j+1}',own)] for j in range(3) if j != i]+[distances[('Rstar_10240',own)]])
        ratios = {label:{k:rm[k]/metrics[k] if rules.nonnegative(rm.get(k)) and rules.nonnegative(metrics.get(k)) and metrics[k]>0 else None
                        for k in ('J','T')} for label,rm in replacement.items()}
        removed = (not d['detected'] and gates['passed']) if d['detected'] is not None and gates['passed'] is not None else None
        rows_out.append(dict(owner=i,attack=name,metrics=metrics,utility=gates,B3=d,B2=b2,removed=removed,
                             replacement_over_copy=ratios,clean_attack=previous['comparisons'][i]))
        cohort.append(dict(owner=i,positive=b2['positive'],margin=b2['margin'],utility=gates['passed']))
    result = dict(comparisons=rows_out,B2=rules.b2_cohort(cohort),B1='unavailable: original failed selection preserved',
                  development_descriptive=True,final_paired_experiment=False,physical_qualification=False,
                  original_0_of_18_preserved=True,PN_six_unsuccessful_preserved=True,
                  analysis_available=all(r['utility']['passed'] is not None and r['B3']['detected'] is not None and r['B2']['status']=='complete' for r in rows_out))
    if persist:write(root/'READOUT.json',result)
    return result


def execute(bundle, root, plan, backend, check):
    for name, owner, steps, recipe in ARMS:
        check(); progress(root,'arm_start',arm=name)
        backend.train(name,owner,steps,recipe)
        if recipe == 'forward32':
            progress(root,'evaluation',arm=name); backend.evaluate(name,owner)
            progress(root,'detector',arm=name); backend.detect(name,owner)
        progress(root,'arm_complete',arm=name)
    backend.distances(); check()
    return readout(bundle,root,plan)


def run(bundle, root):
    bundle, root = Path(bundle), Path(root)
    grant = read(root/'CONSUMED.json'); backend = None; started = time.time()
    def check():
        if time.time() >= grant['work_deadline']: raise TimeoutError('1800-second worker limit; no retry')
    try:
        plan = preflight(bundle,grant['manifest_sha256']); check()
        backend = NumericalBackend(bundle,root,plan,check)
        result = execute(bundle,root,plan,backend,check)
        cr = counts(root,None,check)
        if not cr['exact']: raise ValueError('prescribed work count incomplete')
        write(root/'RESULT.json',dict(complete=True,counts=cr['observed'],analysis_available=result['analysis_available'],
            seconds=time.time()-started,bindings=read(root/'STATUS.json')['bindings'],follow_on_authorized=False))
        progress(root,'complete',counts=cr['observed'],analysis_available=result['analysis_available']);return 0
    except BaseException as exc:
        evidence_error = None
        try:
            if backend is not None: backend.partial()
        except BaseException as e: evidence_error = repr(e)
        write(root/'RESULT.json',dict(complete=False,error=repr(exc),evidence_error=evidence_error,
              seconds=time.time()-started,follow_on_authorized=False))
        progress(root,'stopped',error=repr(exc));return 4


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args();raise SystemExit(run(a.bundle,a.root))
