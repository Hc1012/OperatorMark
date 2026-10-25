"""One-cell Colab orchestration around existing bounded setup/transport (R3).

R3 reuses the completed twenty-model O01 compatibility bridge of failed attempt
20261023T183105Z as preserved historical evidence. Its full return is fetched by
pinned Drive revision, re-verified in this session against freshly downloaded
frozen references, and admitted only if this session's runtime observations equal
that attempt's byte-for-byte. Identity/capacity checks, the two prompts and
authentication precede activation: no attempt or stage clock exists until
ACTIVATION.json is written. After activation there is no retry, reset or
relabelling. The Windows helper only packages files.
"""
from pathlib import Path
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

# Fixed R3 stage budgets (seconds). The supervisor kills a stage at launch + its
# budget; stages given --deadline stop themselves CHILD_EXIT_MARGIN_SECONDS earlier.
FIXED = (('host_auth', 40), ('input', 1500), ('setup_0', 60), ('setup_1', 1200), ('setup_2', 60),
         ('worker_auth', 40), ('runtime', 120), ('bridge_verify', 900), ('bridge_save', 420),
         ('bridge_admit', 600), ('producer', 21600))
DEADLINE_STAGES = frozenset({'input', 'runtime', 'bridge_verify', 'bridge_save', 'bridge_admit', 'producer'})
CLOSURE_SECONDS = 30
CHILD_EXIT_MARGIN_SECONDS = 30
EARLY_REFUSAL_RULE = ('Pre-activation refusal: no attempt, clock or Drive object exists. You may connect a fresh '
                      'runtime and run this same notebook again, at most 2 further times (3 runs in total). '
                      'After ACTIVATED is printed, never rerun this notebook.')


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def need(ok, message):
    if not ok: raise ValueError(message)


def read(path): return json.loads(Path(path).read_bytes())


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value); return value


class PreActivationRefusal(ValueError):
    """Refused before ACTIVATION.json: no attempt, clock or Drive object exists."""


def stage_window(fixed, cursor, name, now, deadline, return_reserve):
    """Return (work_seconds, supervisor_end, child_deadline) or refuse if reserves no longer fit."""
    need(0 <= cursor < len(fixed) and fixed[cursor][0] == name, 'Unexpected stage order')
    seconds = fixed[cursor][1]
    after = sum(w + CLOSURE_SECONDS for _, w in fixed[cursor+1:]) + return_reserve
    end = now + seconds + CLOSURE_SECONDS
    need(end + after < deadline, 'Full remaining work and return reserves no longer fit; no reset')
    return seconds, end, end - CLOSURE_SECONDS - CHILD_EXIT_MARGIN_SECONDS


def stage_commands(job, payload, host_python, worker_python, policy_sha):
    """Exact supervised commands (without --deadline, which stage_window supplies)."""
    job, payload, remote = Path(job), Path(payload), str(Path(job)/'combined_remote.py')
    auth = str(payload/'handoff/metadata_auth_observe.py')
    w = str(worker_python)
    return {
        'host_auth': [host_python, '-B', auth],
        'input': [host_python, '-B', remote, 'input', '--job', str(job)],
        'worker_auth': [w, '-B', auth, '--worker'],
        'runtime': [w, '-B', str(job/'combined_observe.py'), '--job', str(job), '--policy-sha', policy_sha],
        'bridge_verify': [w, '-B', str(job/'verify_bridge.py'), 'verify', '--job', str(job), '--policy-sha', policy_sha],
        'bridge_save': [host_python, '-B', remote, 'bridge-save', '--job', str(job)],
        'bridge_admit': [w, '-B', str(job/'verify_bridge.py'), 'admit', '--job', str(job), '--policy-sha', policy_sha],
        'producer': [w, '-B', str(job/'d2_scoped_producer.py'), '--job', str(job), '--policy-sha', policy_sha]}


def check_policy(job, policy_sha):
    """All frozen package and envelope checks; returns (policy, bridge_policy)."""
    job = Path(job)
    need(sha(job/'D2_COMBINED_POLICY.json') == policy_sha, 'Combined policy changed')
    policy = read(job/'D2_COMBINED_POLICY.json')
    need(sha(job/'D2_PYTHON31316_COMPAT_POLICY.json') == policy['bridge_policy_sha256'], 'Frozen bridge policy changed')
    bridge_policy = read(job/'D2_PYTHON31316_COMPAT_POLICY.json')
    need(bridge_policy['expected_runtime'] == policy['expected_runtime'] and
         bridge_policy['candidate_runtime_payload_sha256'] == policy['candidate_runtime_payload_sha256'],
         'Candidate runtime expectations differ between stages')
    need(policy['execution_stage_budgets'] == [dict(name=n, work_seconds=w) for n, w in FIXED], 'Fixed phase budgets differ')
    need(policy['stage_seconds'] == 30600 and policy['return_reserve_seconds'] == 1800 and
         policy['numerical_work_seconds'] == 21600 and policy['execution_stage_closure_seconds'] == CLOSURE_SECONDS and
         policy['maximum_units'] == 51. and policy['reserved_execution_and_return_seconds'] ==
         sum(w + CLOSURE_SECONDS for _, w in FIXED) + policy['return_reserve_seconds'], 'Combined resource envelope differs')
    r = policy['reused_bridge']
    need(r['root'] == 'reused/' + r['attempt'] and r['source_group'] in policy['additional_sources'] and
         all(name in policy['files'] for name in r['superseded_orchestration']) and
         r['bridge_policy_sha256'] == policy['bridge_policy_sha256'], 'Reused bridge contract incomplete')
    return policy, bridge_policy


def nvidia_identity():
    out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=15, check=True).stdout.strip().splitlines()
    need(len(out) == 1 and out[0].count(',') == 1, 'Exactly one visible GPU required')
    name, driver = (v.strip() for v in out[0].split(','))
    return name, driver


def pre_activation_identity(policy, *, python, executable_sha256, gpu, driver, linux_free_bytes, available_ram_bytes):
    """Pure check of the observed host against the reused attempt; refuses before any clock."""
    r, e = policy['reused_bridge'], policy['expected_runtime']
    problems = []
    if python != e['python']:
        problems.append('Python build is ' + repr(python) + ', reused attempt ran ' + repr(e['python']))
    if executable_sha256 != r['host_interpreter_sha256']:
        problems.append('host interpreter SHA-256 is ' + str(executable_sha256) + ', expected ' + r['host_interpreter_sha256'])
    if gpu not in (e['gpu'], e['gpu'].removeprefix('NVIDIA ')):
        problems.append('GPU is ' + repr(gpu) + '; select A100 (40 GB)')
    if driver != r['driver_version']:
        problems.append('NVIDIA driver is ' + repr(driver) + ', reused attempt ran ' + r['driver_version'])
    if linux_free_bytes < policy['minimum_linux_free_bytes']:
        problems.append('Colab free disk ' + str(linux_free_bytes) + ' below ' + str(policy['minimum_linux_free_bytes']))
    if available_ram_bytes < 4 * 1024**3:
        problems.append('less than 4 GiB host RAM available')
    if problems:
        raise PreActivationRefusal('; '.join(problems))
    return dict(python=python, executable_sha256=executable_sha256, gpu=gpu, driver=driver,
                linux_free_bytes=linux_free_bytes, available_ram_bytes=available_ram_bytes)


def rate_reservation(policy, balance, rate):
    if not (math.isfinite(balance) and balance >= 0 and math.isfinite(rate) and 0 < rate <= policy['maximum_units_per_hour']):
        raise PreActivationRefusal('Actual positive rate up to 6 units/hour and a finite balance are required')
    reservation = policy['stage_seconds']/3600*rate
    if not (reservation <= policy['maximum_units'] and balance >= reservation):
        raise PreActivationRefusal('Current balance must cover %.2f units at the displayed rate' % reservation)
    return reservation


def activation_record(policy, policy_sha, started):
    return dict(attempt=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(started)), start=started,
        deadline=started+policy['stage_seconds'], policy_sha256=policy_sha,
        bridge_policy_sha256=policy['bridge_policy_sha256'], stage_seconds=policy['stage_seconds'],
        scope='R3: reused fixed20 O01 comparisons re-verified, then only on mechanical acceptance the remaining200 raw960 D2 diagnostics',
        reused_bridge_attempt=policy['reused_bridge']['attempt'],
        maximum_rate=policy['maximum_units_per_hour'], maximum_units=policy['maximum_units'],
        started_by='user executing the generated notebook cell', previous_clocks_reused=False, retries=0,
        pre_activation_checks_passed=True, prompts_and_authentication_before_activation=True,
        early_refusal_rule=EARLY_REFUSAL_RULE, scientific_acceptance=False)


def runtime_gate(job, policy, policy_sha):
    folder = job / 'observations/runtime'
    actual = read(folder / 'GPU_RUNTIME_PAYLOAD.json')
    need(actual == policy['expected_runtime'], 'Runtime differs from the frozen expected A100 environment')
    observed = read(folder / 'MODULE_SOURCES.json')
    need(set(observed) == set(policy['expected_modules']), 'Runtime module coverage differs')
    for name, expected in policy['expected_modules'].items():
        row = observed[name]
        need(row['sha256'] == expected and
             row['path'] == '/content/OperatorMark_Stage1/payload/code/' + row['source'] and
             policy['source_pins'][row['source']] == expected, 'Imported source differs: ' + name)
    serial = read(folder / 'SERIALIZER_OBSERVATION.json')
    for key, expected in policy['expected_serializer'].items():
        need(serial.get(key) == expected, 'Serializer premise differs: ' + key)
    applicability=read(folder / 'APPLICABILITY_FOR_REVIEW.json')
    need(applicability['candidate_matches_frozen_expectation'] is True and
         set(applicability['historical_differences']) == {'python'}, 'Unexpected runtime difference')
    need(sha(folder/'GPU_RUNTIME_PAYLOAD.json') == policy['candidate_runtime_payload_sha256'],
         'Candidate runtime payload serialization differs')
    put(job / 'RUNTIME_GATE.json', dict(passed=True, policy_sha256=policy_sha,
        runtime=actual, candidate_runtime_payload_sha256=policy['candidate_runtime_payload_sha256'],
        observed_interpreter_sha256=sha(folder/'CANDIDATE_INTERPRETER.json'),
        interpreter_observation='observations/runtime/CANDIDATE_INTERPRETER.json',
        expected_runtime_basis='Frozen candidate identity; new compatibility evidence still required',
        actual_runtime_sha256=sha(folder / 'GPU_RUNTIME_PAYLOAD.json'),
        actual_modules_sha256=sha(folder / 'MODULE_SOURCES.json'),
        serializer_observation_sha256=sha(folder / 'SERIALIZER_OBSERVATION.json'),
        scope='D2_diagnostic_candidate_observations_only',
        expectations_frozen_before_observation=True, final_calibration_accepted=False,
        numerical_compatibility_verified=False, compatibility_accepted=False, formal_assurance=False))


def reused_identity(job, policy):
    """This session's observations must equal the reused attempt's byte-for-byte; never normalised."""
    job = Path(job); r = policy['reused_bridge']; root = job/r['root']
    differing = []
    for name in r['identical_observations']:
        mine, theirs = job/name, root/name
        need(mine.is_file() and not mine.is_symlink() and theirs.is_file() and not theirs.is_symlink(),
             'Identity observation missing: ' + name)
        if mine.read_bytes() != theirs.read_bytes(): differing.append(name)
    need(not differing, 'This session differs from reused attempt ' + r['attempt'] + ' in: ' + ', '.join(differing) +
         '. Reuse refused; no admission. A fresh 20-model bridge would need a separate decision.')
    need(sha(job/'RUNTIME_GATE.json') == sha(root/'RUNTIME_GATE.json') == r['runtime_gate_sha256'],
         'Runtime gate identity differs from the reused attempt')
    put(job/'REUSED_RUNTIME_IDENTITY.json', dict(reused_attempt=r['attempt'], identical=list(r['identical_observations']),
        runtime_gate_sha256=r['runtime_gate_sha256'], numerical_compatibility_issued=False,
        basis='Byte-identical observations only; compatibility evidence is the reused attempt bridge, re-verified next'))


def run(job, policy_sha, archive_expectation):
    job = Path(job).resolve(); root = job.parent; payload = root/'payload'
    policy, bridge_policy = check_policy(job, policy_sha)
    from google.colab import auth, runtime
    need(callable(getattr(runtime, 'unassign', None)), 'Automatic disconnect unavailable; no dispatch')
    if (job/'ACTIVATION.json').exists():
        print('This combined attempt already exists. No retry or clock reset.', flush=True); return

    # ---- Pre-activation: no attempt, clock or Drive object exists yet. ----
    try:
        host_observation = dict(python=sys.version, executable=sys.executable,
            executable_sha256=sha(Path(sys.executable).resolve()), observed_at=time.time(),
            expected_python=policy['expected_runtime']['python'], compatibility_verified=False)
        gpu, driver = nvidia_identity()
    except PreActivationRefusal:
        raise
    except BaseException as exc:
        raise PreActivationRefusal('Host/GPU observation unavailable: ' + repr(exc)) from None
    observed = pre_activation_identity(policy, python=sys.version, executable_sha256=host_observation['executable_sha256'],
        gpu=gpu, driver=driver, linux_free_bytes=shutil.disk_usage('/content').free, available_ram_bytes=sysconf_available())
    print('Pre-activation identity matched the reused attempt (Python, interpreter, A100, driver ' + driver + ').', flush=True)
    print('One launch: re-verify the twenty saved reference comparisons, then the remaining200 diagnostics if the fixed gate passes.', flush=True)
    print('Enter the TWO actual Colab numbers. Do not guess a rate shown as zero.', flush=True)
    try:
        balance = float(input('Current COMPUTE UNIT BALANCE: ').strip())
        rate = float(input('Current COMPUTE UNITS PER HOUR for this connected A100: ').strip())
    except ValueError:
        raise PreActivationRefusal('Both prompts need plain numbers') from None
    reservation = rate_reservation(policy, balance, rate)
    print('Authenticate the same Google account that owns the existing results folder.', flush=True)
    try:
        auth.authenticate_user()
    except BaseException as exc:
        raise PreActivationRefusal('Authentication did not complete: ' + type(exc).__name__) from None

    # ---- Activation: the single fixed clock for this attempt starts here. ----
    started = time.time()
    activation = activation_record(policy, policy_sha, started)
    deadline = activation['deadline']; attempt = activation['attempt']
    put(job/'ACTIVATION.json', activation); put(job/'INPUT_ARCHIVE_EXPECTATION.json', archive_expectation)
    put(job/'HOST_INTERPRETER_OBSERVED.json', host_observation)
    put(job/'LIVE_PREFLIGHT.json', dict(observed_at=time.time(), gpu_display=gpu, driver_version=driver,
        host_python=sys.version, linux_free_bytes=observed['linux_free_bytes'], compute_unit_balance=balance,
        compute_units_per_hour=rate, maximum_stage_units_at_observed_rate=reservation,
        before_cell_connect_time_and_billing_unknown=True, prompts_before_activation=True, whole_campaign_allocation=False))
    print('ACTIVATED attempt ' + attempt + '. From here on, never rerun this notebook; any stop ends this attempt.', flush=True)
    print('Overall maximum 8.5 hours including setup and saving; numerical limit 6 h for the remaining200.', flush=True)
    done = threading.Event()

    def cutoff():
        if not done.wait(max(0., deadline-time.time())):
            print('Fixed combined deadline reached. Requesting disconnect.', flush=True)
            try: runtime.unassign()
            except BaseException as exc: print('Use Runtime > Disconnect and delete runtime:', type(exc).__name__, flush=True)
    threading.Thread(target=cutoff, daemon=True).start()
    phase = 'post-activation preparation'; setup_ready = False; saved = False; download_window = False
    fixed = list(FIXED)
    try:
        remaining_bound = sum(w+CLOSURE_SECONDS for _, w in fixed)+policy['return_reserve_seconds']
        need(time.time()+remaining_bound < deadline, 'Fixed startup reserve unavailable; no clock reset')
        sys.path[:0] = [str(payload/'code/tools'), str(payload/'code')]
        from q6_common import THREAD_ENV
        from msdev.colab_environment import setup_commands, owned
        env = dict(os.environ, PYTHONPATH=str(payload/'code/tools')+os.pathsep+str(payload/'code'), PYTHONUNBUFFERED='1')
        env.update(THREAD_ENV)
        need(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0, 'Owned setup subreaper unavailable')
        supervisor = module(payload/'code/q5_colab_supervisor.py', 'calibration_owned_supervisor')
        setup_ready = True
        environment = root/'environment'; python = environment/'bin/python'
        commands = stage_commands(job, payload, sys.executable, python, policy_sha)

        def execute(name, command, seconds, end, acceptable=(0,)):
            folder = job / 'supervision' / name; folder.parent.mkdir(exist_ok=True)
            print(name.upper() + ': starting.', flush=True)
            stop = threading.Event(); began = time.monotonic(); positions = {}

            def heartbeat():
                while not stop.wait(20):
                    print(f'{name}: {int(time.monotonic()-began)} seconds elapsed.', flush=True)
                    for filename in ('stdout.bin', 'stderr.bin'):
                        path = folder / filename
                        if path.is_file():
                            length = path.stat().st_size
                            if length > positions.get(filename, 0):
                                with path.open('rb') as f:
                                    f.seek(max(positions.get(filename, 0), length-1400))
                                    print(f.read().decode('utf-8', 'replace'), end='', flush=True)
                                positions[filename] = length

            thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
            try:
                result = owned(command, seconds, end, folder, supervisor, env=env, acceptable=acceptable)
                print(name.upper() + ': process closed.', flush=True)
                return result
            finally:
                stop.set(); thread.join(timeout=1)
                for filename in ('stdout.bin', 'stderr.bin'):
                    path = folder / filename
                    if path.is_file() and path.stat().st_size > positions.get(filename, 0):
                        with path.open('rb') as f:
                            f.seek(max(positions.get(filename, 0), path.stat().st_size-6000))
                            print(f.read().decode('utf-8', 'replace'), end='', flush=True)

        cursor = 0

        def stage(name, command=None, acceptable=(0,)):
            nonlocal cursor
            seconds, end, child_deadline = stage_window(fixed, cursor, name, time.time(), deadline,
                                                        policy['return_reserve_seconds'])
            command = list(commands[name] if command is None else command)
            if name in DEADLINE_STAGES: command += ['--deadline', str(child_deadline)]
            result = execute(name, command, seconds, end, acceptable=acceptable)
            cursor += 1
            return result

        phase = 'host authentication observation'
        stage('host_auth')
        phase = 'saved checkpoint, reference and reused-bridge transfer'
        stage('input')
        phase = 'existing pinned environment setup'
        need(not environment.exists(), 'Environment already exists; no repeated setup')
        for i, (command, seconds) in enumerate(setup_commands(sys.executable, environment, payload)):
            need(seconds == fixed[cursor][1], 'Setup command budget differs')
            stage('setup_'+str(i), command)
        ops = module(job/'continuation_ops.py', 'calibration_interpreter_preparation')
        need(sha(Path(sys.executable).resolve()) == host_observation['executable_sha256'], 'Host interpreter changed during setup')
        witness = ops.regular_worker_interpreter(python, host_observation['executable_sha256'],
            lambda: need(time.time()+sum(w+CLOSURE_SECONDS for _, w in fixed[cursor:])+policy['return_reserve_seconds'] < deadline,
                         'Interpreter setup deadline'))
        witness.update(basis='Byte-identical path preparation from observed host; not numerical acceptance',
            host_observation_sha256=sha(job/'HOST_INTERPRETER_OBSERVED.json'), numerical_compatibility_verified=False)
        put(job/'INTERPRETER_PATH_PREPARATION.json', witness)
        phase = 'worker authentication and candidate runtime observation'
        stage('worker_auth')
        stage('runtime')
        runtime_gate(job, bridge_policy, policy['bridge_policy_sha256'])
        phase = 'byte-identity with the reused attempt runtime'
        reused_identity(job, policy)
        phase = 'independent re-verification of the reused twenty comparisons'
        stage('bridge_verify')
        phase = 'durable compatibility evidence save'
        stage('bridge_save')
        phase = 'scoped mechanical admission after closed verifier and publisher'
        stage('bridge_admit')
        print('COMPATIBILITY GATE PASSED. Starting the remaining200 raw960 diagnostics automatically.', flush=True)
        phase = 'remaining200 raw960 diagnostics'
        stage('producer')
        result = read(job/'output/RESULT.json')
        need(result['complete'] is True and result['counts'] == policy['operation_limits'], 'Remaining diagnostic population incomplete')
        need(cursor == len(fixed), 'Combined stage sequence incomplete')
        print('All200 raw960 diagnostics completed. Saving the combined return; verification of the return is still required.', flush=True)
    except BaseException as exc:
        put(job/'FIRST_FAILURE.json', dict(phase=phase, error=repr(exc), traceback=traceback.format_exc(),
            failed_at=time.time(), original_deadline=deadline, retry=False, scientific_acceptance=False,
            reused_bridge_attempt=policy['reused_bridge']['attempt'],
            reused_bridge_verified=(job/'BRIDGE_VERIFICATION.json').is_file(),
            raw960_diagnostics_started=(job/'output/START.json').is_file(), training_started=False,
            native_solver_work_started=False, expected_python=policy['expected_runtime']['python'], actual_host_python=sys.version))
        print('STOPPED AT: '+phase+'\n'+str(exc), flush=True)
    finally:
        # Saving uses host Python and the unchanged stdlib REST transport, even
        # if package setup failed. Child closure evidence precedes the archive.
        if setup_ready:
            try:
                phase = 'durable return saving'
                save_end = min(deadline-360, time.time()+1500)
                need(save_end-time.time() > 60, 'No complete save reserve remains')
                execute('save', [sys.executable, '-B', str(job/'combined_remote.py'), 'save',
                    '--job', str(job), '--deadline', str(save_end-30)], save_end-time.time()-30, save_end)
                receipt = read(job / 'SAVE_RESULT.json'); saved = True
                print('\nFULL RETURN SAVED IN DRIVE: ' + receipt['archive'], flush=True)
                print('DOWNLOAD THE SMALL REVIEW: '+receipt['review_url'], flush=True)
                try:
                    close_end = min(deadline-10, time.time()+90)
                    execute('closing_record', [sys.executable, '-B', str(job/'combined_remote.py'), 'closure',
                        '--job', str(job), '--deadline', str(close_end-30)], close_end-time.time()-30, close_end)
                    print('Closing record: '+read(job/'CLOSING_RECORD_SAVED.json')['closure_url'], flush=True)
                except BaseException as exc:
                    print('Main return remains saved. Additional publisher closing record failed: '+repr(exc), flush=True)
            except BaseException as exc:
                print('Drive return save failed: '+repr(exc), flush=True)
        if not saved:
            # Small, bounded local fallback remains available even without Drive
            # authentication. This reports failure, never claims science saved.
            try:
                import zipfile
                report = job / ('D2_COMBINED_FULL_'+attempt+'.zip')
                if not report.is_file() or not zipfile.is_zipfile(report):
                    report = Path('/content') / ('D2_COMBINED_DIAGNOSTIC_'+attempt+'.zip')
                    names = [p for p in job.glob('*.json') if p.stat().st_size < 2*1024**2]
                    with zipfile.ZipFile(report, 'x', zipfile.ZIP_DEFLATED) as z:
                        for p in names: z.write(p, p.name)
                print('Direct-download fallback: '+str(report)+'. Also save this notebook with outputs.', flush=True)
                from google.colab import files
                files.download(str(report))
                # Do not immediately destroy the runtime that is serving the
                # fallback download. This finite window stays within the cap.
                seconds = max(0., min(300., deadline-time.time()))
                print('Download now. Automatic disconnect in at most '+str(int(seconds))+' seconds.', flush=True)
                timer = threading.Timer(seconds, runtime.unassign); timer.daemon = True; timer.start()
                download_window = True
            except BaseException as exc:
                print('Diagnostic download unavailable: '+type(exc).__name__, flush=True)
        if not download_window:
            print('Requesting disconnect now. Existing input files and all old results are preserved.', flush=True)
            try: runtime.unassign()
            except BaseException as exc:
                print('Use Runtime > Disconnect and delete runtime now: '+type(exc).__name__, flush=True)
            done.set()


def sysconf_available():
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
