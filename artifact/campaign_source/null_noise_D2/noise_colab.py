"""Single-cell noise20 launch around the retained owned setup/transport functions.

This is new orchestration only. It does not alter frozen scientific kernels,
replay diagnostics, admit native solvers, or transfer a historical runtime gate.
The fresh noise-scoped gate is required before the producer starts. Host identity,
resource prompts and Google authentication precede this attempt's single clock.
Connected Colab time before activation can still be billed by Colab.
"""
from pathlib import Path
import ctypes
import hashlib
import importlib
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

STARTUP = (('host_auth', 40), ('input', 1500), ('setup_0', 60), ('setup_1', 1200),
           ('setup_2', 60), ('worker_auth', 40), ('runtime', 600))
PREACTIVATION_CAPACITY_SECONDS = 120
DEADLINE_STAGES = frozenset({'input', 'runtime'})
CLOSURE_SECONDS = 30
CHILD_EXIT_MARGIN_SECONDS = 30
SCOPE = 'selected_noise20_only'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


class PreActivationRefusal(ValueError):
    """No Noise attempt or scientific work has started."""


def stage_window(fixed, cursor, name, now, deadline, return_reserve):
    need(0 <= cursor < len(fixed) and fixed[cursor][0] == name, 'Unexpected stage order')
    seconds = fixed[cursor][1]
    after = sum(w + CLOSURE_SECONDS for _, w in fixed[cursor + 1:]) + return_reserve
    end = now + seconds + CLOSURE_SECONDS
    need(end + after < deadline, 'Remaining work and save reserves no longer fit; no clock reset')
    return seconds, end, end - CLOSURE_SECONDS - CHILD_EXIT_MARGIN_SECONDS


def publication_window(fixed, cursor, name, now, deadline, return_reserve,
                       pair_deadline, maximum_work):
    """Borrow only unused producer time inside the already fixed job window."""
    need(0 <= cursor < len(fixed) and fixed[cursor][0] == name
         and name.startswith('publish_'), 'Unexpected publication stage order')
    after = sum(w + CLOSURE_SECONDS for _, w in fixed[cursor + 1:]) + return_reserve
    need(pair_deadline + after < deadline,
         'Fixed job window and future save reserves no longer fit; no clock reset')
    seconds = min(maximum_work, pair_deadline - now - CLOSURE_SECONDS)
    need(seconds > CHILD_EXIT_MARGIN_SECONDS,
         'Fixed job publication window expired; no retry or clock reset')
    end = now + seconds + CLOSURE_SECONDS
    need(end <= pair_deadline, 'Publication exceeds the original job window')
    return seconds, end, end - CLOSURE_SECONDS - CHILD_EXIT_MARGIN_SECONDS


def member_token(member):
    need(isinstance(member, str) and len(member.split('/')) == 2
         and member.split('/')[0] in ['O%02d' % i for i in range(1, 11)]
         and member.split('/')[1] in ('noise_small_M3', 'noise_large_M3'),
         'Unexpected frozen noise member')
    return member.replace('/', '_')


def fixed_stages(policy):
    stages = list(STARTUP)
    for row in policy['noise_jobs']:
        token = member_token(row['id'])
        stages.extend((('producer_' + token, 900), ('publish_' + token, 900)))
    return stages


def stage_commands(job, payload, host_python, worker_python, policy_sha, policy):
    job, payload = Path(job), Path(payload)
    auth = str(payload / 'handoff/metadata_auth_observe.py')
    w = str(worker_python)
    commands = {
        'capacity': [host_python, '-B', str(job / 'noise_remote.py'), 'capacity', '--job', str(job)],
        'host_auth': [host_python, '-B', auth],
        'input': [host_python, '-B', str(job / 'noise_remote.py'), 'input', '--job', str(job)],
        'worker_auth': [w, '-B', auth, '--worker'],
        'runtime': [w, '-B', str(job / 'noise_runtime.py'), '--job', str(job), '--policy-sha', policy_sha],
    }
    for row in policy['noise_jobs']:
        token = member_token(row['id'])
        commands['producer_' + token] = [w, '-B', str(job / 'noise_producer.py'),
            '--job', str(job), '--policy-sha', policy_sha, '--member', row['id']]
        commands['publish_' + token] = [host_python, '-B', str(job / 'noise_remote.py'),
            'publish', '--job', str(job), '--job-id', row['id']]
    return commands


def check_policy(job, policy_sha):
    job = Path(job)
    need(sha(job / 'NOISE_POLICY.json') == policy_sha, 'Noise policy changed')
    p = read(job / 'NOISE_POLICY.json')
    need(p['scope'] == SCOPE, 'noise-only scope required')
    expected_members = ['O%02d/%s' % (i, label)
                        for label in ('noise_large_M3', 'noise_small_M3') for i in range(1, 11)]
    need([row['id'] for row in p['noise_jobs']] == expected_members,
         'Complete ordered frozen noise20 population required')
    for row in p['noise_jobs']:
        need(row['output'] == 'output/jobs/' + row['id']
             and row['closure'] == 'supervision/producer_' + member_token(row['id']) + '/RESULT.json',
             'Noise output/closure path differs')
    fixed = fixed_stages(p)
    need(p['execution_stage_budgets'] == [dict(name=n, work_seconds=w) for n, w in fixed],
         'Fixed stage budgets differ')
    need(p['stage_seconds'] == 43200 and p['return_reserve_seconds'] == 1200
         and p['numerical_work_seconds'] == 18000
         and p['execution_stage_closure_seconds'] == CLOSURE_SECONDS
         and p['maximum_units_per_hour'] == 6.0 and p['maximum_units'] == 72.0
         and p['minimum_drive_free_bytes'] == 104 * 1024**3
         and p['minimum_linux_free_bytes'] == 48 * 1024**3
         and p['minimum_job_start_free_bytes'] == 10 * 1024**3
         and p['minimum_job_start_free_inodes'] == 60000
         and p['minimum_publish_free_bytes'] == 5 * 1024**3
         and p['shared_job_window_seconds'] == 1860
         and p['producer_max_work_seconds'] == 900
         and p['publisher_max_work_seconds'] == 1800
         and p['reserved_execution_and_return_seconds'] == sum(w + CLOSURE_SECONDS for _, w in fixed) + 1200,
         'Noise resource envelope differs')
    for name in ('noise_colab.py', 'noise_producer.py', 'noise_evaluate.py', 'noise_cpu_export.py',
                 'noise_runtime.py', 'noise_remote.py',
                 'continuation_ops.py', 'runtime_values.py'):
        item = p['files'][name]
        target = job / name
        need(target.is_file() and not target.is_symlink()
             and target.stat().st_size == item['bytes'] and sha(target) == item['sha256'],
             'Launch script differs: ' + name)
    payload = job.parent / 'payload'
    required_payload = {'requirements-colab.txt', 'handoff/metadata_auth_observe.py',
                        'code/q5_colab_supervisor.py'}
    need(required_payload <= set(p['payload_files']), 'Pinned setup/authentication payload incomplete')
    for name, item in p['payload_files'].items():
        target = payload / name
        need(target.is_file() and not target.is_symlink()
             and target.resolve().is_relative_to(payload.resolve())
             and target.stat().st_size == item['bytes'] and sha(target) == item['sha256'],
             'Setup/authentication payload differs: ' + name)
    source = payload / 'code'
    need(len(p['source_pins']) == 338, 'Complete frozen scientific source population required')
    for name, expected in p['source_pins'].items():
        target = source / name
        need(target.is_file() and not target.is_symlink()
             and target.resolve().is_relative_to(source.resolve()) and sha(target) == expected,
             'Pinned setup/supervisor/scientific source differs: ' + name)
    return p


def prepare_launch_dependencies(job, policy):
    """Import actual non-numerical launch dependencies before prompts/activation.

    Authentication has top-level network work, so its script is compiled only.
    No model/runtime observation, credentials, setup command or subprocess runs
    here. Returned objects are the same ones subsequently used by the launcher.
    """
    job = Path(job).resolve()
    payload = job.parent / 'payload'
    source = payload / 'code'
    sys.path[:0] = [str(source / 'tools'), str(source)]
    observed = {}

    def record(name, path, value, required):
        path = Path(path).resolve()
        need(Path(value.__file__).resolve() == path, 'Launch import resolved outside pinned package: ' + name)
        need(all(callable(getattr(value, n, None)) for n in required),
             'Launch dependency entry point unavailable: ' + name)
        observed[name] = dict(path=str(path), sha256=sha(path), entry_points=list(required),
                              actual_import_completed=True)
        return value

    supervisor = record('supervisor', source / 'q5_colab_supervisor.py',
        module(source / 'q5_colab_supervisor.py', 'noise_owned_supervisor'),
        ('process_identity', 'cleanup'))
    setup = record('setup', source / 'tools/msdev/colab_environment.py',
        importlib.import_module('msdev.colab_environment'), ('setup_commands', 'owned'))
    capture = record('capture', source / 'tools/msdev/bounded_capture.py',
        importlib.import_module('msdev.bounded_capture'), ('capture',))
    record('stream_retention', source / 'operatormark/corpus/bounded_stream.py',
        importlib.import_module('operatormark.corpus.bounded_stream'), ('StreamRetention', 'verify_retained'))
    common = record('thread_environment', source / 'tools/q6_common.py',
        importlib.import_module('q6_common'), ())
    need(isinstance(common.THREAD_ENV, dict) and all(isinstance(k, str) and isinstance(v, str)
         for k, v in common.THREAD_ENV.items()), 'Thread environment definition unavailable')
    record('drive_transport', source / 'tools/msfinal/drive_transport.py',
        importlib.import_module('msfinal.drive_transport'), ('RefreshingMetadataToken', 'GoogleDriveTransfer'))
    entries = {}
    for filename, required in (
        ('continuation_ops.py', ('regular_worker_interpreter',)),
        ('runtime_values.py', ('plain_runtime',)),
        ('noise_remote.py', ('transfer', 'validate_job_commit')),
        ('noise_runtime.py', ('observe', 'verify_gate')),
        ('noise_producer.py', ('run',)),
        ('noise_evaluate.py', ('evaluate_noise',)),
        ('noise_cpu_export.py', ('export_saved_designs', 'preflight_export')),
    ):
        entries[filename] = record(filename, job / filename,
            module(job / filename, 'noise_preflight_' + filename.removesuffix('.py')), required)
    # Metadata auth is intentionally not imported: its top-level code performs
    # authenticated requests. Compilation still catches missing/invalid bytes.
    auth = payload / 'handoff/metadata_auth_observe.py'
    compile(auth.read_bytes(), str(auth), 'exec')
    observed['metadata_auth'] = dict(path=str(auth.resolve()), sha256=sha(auth),
        syntax_checked=True, actual_import_completed=False, network_work_executed=False)
    environment = job.parent / 'environment'
    setup_plan = setup.setup_commands(sys.executable, environment, payload)
    need(len(setup_plan) == 3 and [seconds for _, seconds in setup_plan] == [60, 1200, 60],
         'Pinned setup command plan differs')
    commands = stage_commands(job, payload, sys.executable, environment / 'bin/python', '0' * 64, policy)
    for name, command in commands.items():
        for operand in command:
            if operand.endswith('.py'):
                target = Path(operand)
                need(target.is_file() and not target.is_symlink(),
                     'Stage entry script missing before activation: ' + name)
                compile(target.read_bytes(), str(target), 'exec')
    return dict(supervisor=supervisor, setup_commands=setup.setup_commands, owned=setup.owned,
                thread_env=common.THREAD_ENV, interpreter_ops=entries['continuation_ops.py'],
                transport_ops=entries['noise_remote.py'],
                receipt=dict(complete=True, observed=observed, before_authentication=True,
                    before_activation=True, subprocesses_run=0, numerical_work_run=0,
                    network_work_run=0, source_pins_verified=len(policy['source_pins'])))


def nvidia_identity():
    out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=15, check=True).stdout.strip().splitlines()
    need(len(out) == 1 and out[0].count(',') == 1, 'Exactly one visible GPU required')
    return tuple(v.strip() for v in out[0].split(','))


def pre_activation_identity(p, *, python, executable_sha256, gpu, driver,
                            linux_free_bytes, available_ram_bytes):
    e = p['expected_runtime']
    problems = []
    if python != e['python']:
        problems.append('Python build differs from the prepared expectation')
    if executable_sha256 != p['host_interpreter_sha256']:
        problems.append('Host interpreter bytes differ from the prepared expectation')
    if gpu not in (e['gpu'], e['gpu'].removeprefix('NVIDIA ')):
        problems.append('GPU differs; this package requires A100 40 GB')
    if driver != p['driver_version']:
        problems.append('NVIDIA driver differs from the prepared expectation')
    if linux_free_bytes < p['minimum_linux_free_bytes']:
        problems.append('Colab free disk below prepared minimum')
    if available_ram_bytes < 4 * 1024**3:
        problems.append('Less than 4 GiB host RAM available')
    if problems:
        raise PreActivationRefusal('; '.join(problems))
    return dict(python=python, executable_sha256=executable_sha256, gpu=gpu, driver=driver,
                linux_free_bytes=linux_free_bytes, available_ram_bytes=available_ram_bytes)


def rate_reservation(p, balance, rate):
    if not (math.isfinite(balance) and balance >= 0 and math.isfinite(rate)
            and 0 < rate <= p['maximum_units_per_hour']):
        raise PreActivationRefusal('Enter a finite balance and actual positive rate up to 6 units/hour')
    reservation = p['stage_seconds'] / 3600 * rate
    if not (reservation <= p['maximum_units'] and balance >= reservation):
        raise PreActivationRefusal('Current balance must cover %.2f units at the displayed rate' % reservation)
    return reservation


def activation_record(p, policy_sha, started):
    return dict(attempt=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(started)), start=started,
        deadline=started + p['stage_seconds'], policy_sha256=policy_sha, scope=SCOPE,
        stage_seconds=p['stage_seconds'], maximum_rate=p['maximum_units_per_hour'],
        maximum_units=p['maximum_units'], started_by='user executing the generated notebook cell',
        previous_clocks_reused=False, retries=0, pre_activation_checks_passed=True,
        prompts_and_authentication_before_activation=True, scientific_acceptance=False,
        formal_custody_assurance=False, native_solver_calls_authorized=0,
        connected_time_before_activation_can_be_billed=True)


def completed_output(members, p):
    need(members == [row['id'] for row in p['noise_jobs']],
         'Complete ordered frozen noise20 population unavailable')
    return dict(complete=True, completed_jobs=20, completed_members=members,
        durable_jobs=20, GPU_evaluation_complete=True, task_scoring_pending=True,
        native_solver_calls=0, model_updates=0, scope=SCOPE,
        scientific_acceptance=False, formal_custody_assurance=False)


def noise_dispatch_started(job):
    """A dispatch marker is not proof that a numerical output completed."""
    job = Path(job)
    return (any(job.glob('output/jobs/*/*/START.json'))
            or any(job.glob('output/jobs/*/*/JOB_RESULT.json'))
            or any(job.glob('job_receipts/*.json'))
            or any(job.glob('supervision/producer_*/IDENTITY.json')))


def execute_owned(job, launch, env, name, command, seconds, end, acceptable=(0,)):
    """One real owned child with the retained closure/stream implementation."""
    folder = Path(job) / 'supervision' / name
    folder.parent.mkdir(exist_ok=True)
    print(name.upper() + ': starting.', flush=True)
    stop, began, positions = threading.Event(), time.monotonic(), {}

    def show(new_only=True):
        for filename in ('stdout.bin', 'stderr.bin'):
            path = folder / filename
            if path.is_file():
                length = path.stat().st_size
                if length > positions.get(filename, 0):
                    with path.open('rb') as f:
                        f.seek(max(positions.get(filename, 0), length - (1400 if new_only else 6000)))
                        print(f.read().decode('utf-8', 'replace'), end='', flush=True)
                    positions[filename] = length

    def heartbeat():
        while not stop.wait(20):
            print(f'{name}: {int(time.monotonic() - began)} seconds elapsed.', flush=True)
            show()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        result = launch['owned'](command, seconds, end, folder, launch['supervisor'],
                                 env=env, acceptable=acceptable)
        print(name.upper() + ': process closed.', flush=True)
        return result
    finally:
        stop.set()
        thread.join(timeout=1)
        show(False)


def validate_capacity(receipt, policy_sha, p, observed_after):
    need(receipt.get('passed') is True
         and receipt.get('policy_sha256') == policy_sha
         and receipt.get('scope') == SCOPE
         and receipt.get('before_activation') is True
         and receipt.get('private_owner_only') is True
         and receipt.get('drive_free_bytes', -1) >= p['minimum_drive_free_bytes']
         and receipt.get('linux_free_bytes', -1) >= p['minimum_linux_free_bytes']
         and observed_after <= receipt.get('observed_at', 0) <= time.time(),
         'Fresh capacity/privacy preflight did not pass; no activation')


def rolling_storage_check(job, policy, member, phase):
    """Observe fresh scratch capacity before each producer or publication child."""
    need(phase in ('producer', 'publish'), 'Unknown rolling storage phase')
    free = shutil.disk_usage(job).free
    inodes = os.statvfs(job).f_favail
    minimum = policy['minimum_job_start_free_bytes' if phase == 'producer'
                     else 'minimum_publish_free_bytes']
    enough = free >= minimum and (phase != 'producer' or inodes >= policy['minimum_job_start_free_inodes'])
    observation = dict(member=member, phase=phase, observed_at=time.time(), free_bytes=free,
        free_inodes=inodes, minimum_free_bytes=minimum,
        minimum_free_inodes=policy['minimum_job_start_free_inodes'] if phase == 'producer' else None,
        passed=enough, policy_sha256=sha(Path(job) / 'NOISE_POLICY.json'))
    put(Path(job) / 'observations/rolling_storage' / member_token(member) / (phase + '.json'), observation)
    need(enough, 'Rolling scratch capacity below frozen ' + phase + ' bound; no next child or retry')


def run(job, policy_sha, archive_expectation):
    job = Path(job).resolve()
    root, payload = job.parent, job.parent / 'payload'
    p = check_policy(job, policy_sha)
    launch = prepare_launch_dependencies(job, p)
    from google.colab import auth, runtime
    need(callable(getattr(runtime, 'unassign', None)), 'Automatic disconnect unavailable; no dispatch')
    need(not (job / 'ACTIVATION.json').exists(), 'An attempt already exists. No retry or clock reset.')
    env = dict(os.environ, PYTHONPATH=str(payload / 'code/tools') + os.pathsep + str(payload / 'code'),
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    env.update(launch['thread_env'])
    need(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0,
         'Owned setup subreaper unavailable; no activation')
    environment = root / 'environment'
    python = environment / 'bin/python'
    commands = stage_commands(job, payload, sys.executable, python, policy_sha, p)

    host_observation = dict(python=sys.version, executable=sys.executable,
        executable_sha256=sha(Path(sys.executable).resolve()), observed_at=time.time(),
        expected_python=p['expected_runtime']['python'], compatibility_verified=False)
    try:
        gpu, driver = nvidia_identity()
    except BaseException as exc:
        raise PreActivationRefusal('Host/GPU observation unavailable: ' + type(exc).__name__) from None
    observed = pre_activation_identity(p, python=sys.version,
        executable_sha256=host_observation['executable_sha256'], gpu=gpu, driver=driver,
        linux_free_bytes=shutil.disk_usage('/content').free,
        available_ram_bytes=os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES'))
    print('Host identity matched the prepared A100 environment.', flush=True)
    print('This notebook runs all 20 noise evaluations, saving and verifying each before the next.', flush=True)
    print('The 12-hour hard ceiling includes setup and saving; no whole-wave duration has been measured.', flush=True)
    print('Colab can bill connected time before activation. Enter the TWO actual numbers; never guess a zero rate.', flush=True)
    try:
        balance = float(input('Current COMPUTE UNIT BALANCE: ').strip())
        rate = float(input('Current COMPUTE UNITS PER HOUR for this connected A100: ').strip())
    except ValueError:
        raise PreActivationRefusal('Both prompts need plain numbers') from None
    reservation = rate_reservation(p, balance, rate)
    print('Authenticate the Google account that owns the existing private results folder.', flush=True)
    try:
        auth.authenticate_user()
    except BaseException as exc:
        raise PreActivationRefusal('Authentication did not complete: ' + type(exc).__name__) from None

    print('Checking Drive capacity and privacy before activation or environment setup.', flush=True)
    capacity_started = time.time()
    capacity_end = capacity_started + PREACTIVATION_CAPACITY_SECONDS + CLOSURE_SECONDS
    try:
        execute_owned(job, launch, env, 'preactivation_capacity',
            commands['capacity'] + ['--deadline', str(capacity_end - CLOSURE_SECONDS - CHILD_EXIT_MARGIN_SECONDS)],
            PREACTIVATION_CAPACITY_SECONDS, capacity_end)
        validate_capacity(read(job / 'STORAGE_PREFLIGHT.json'), policy_sha, p, capacity_started)
    except BaseException as exc:
        raise PreActivationRefusal('Capacity/privacy preflight refused; no activation: ' + str(exc)) from None

    started = time.time()
    activation = activation_record(p, policy_sha, started)
    deadline, attempt = activation['deadline'], activation['attempt']
    put(job / 'ACTIVATION.json', activation)
    put(job / 'INPUT_ARCHIVE_EXPECTATION.json', archive_expectation)
    put(job / 'HOST_INTERPRETER_OBSERVED.json', host_observation)
    put(job / 'LAUNCH_DEPENDENCY_PREFLIGHT.json', launch['receipt'])
    put(job / 'LIVE_PREFLIGHT.json', dict(observed_at=time.time(), gpu_display=gpu, driver_version=driver,
        host_python=sys.version, linux_free_bytes=observed['linux_free_bytes'],
        compute_unit_balance=balance, compute_units_per_hour=rate,
        maximum_stage_units_at_observed_rate=reservation,
        before_cell_connect_time_and_billing_unknown=True, whole_campaign_allocation=False))
    print('ACTIVATED attempt ' + attempt + '. Never rerun this notebook after this line.', flush=True)
    print('Overall hard cutoff: 12 hours. All 20 jobs run automatically if the scoped gate passes.', flush=True)
    done = threading.Event()

    def cutoff():
        if not done.wait(max(0., deadline - time.time())):
            print('Fixed deadline reached. Requesting disconnect.', flush=True)
            try:
                runtime.unassign()
            except BaseException as exc:
                print('Use Runtime > Disconnect and delete runtime: ' + type(exc).__name__, flush=True)
    threading.Thread(target=cutoff, daemon=True).start()
    phase, setup_ready, saved, download_window = 'post-activation preparation', False, False, False
    fixed = fixed_stages(p)
    completed_members = []
    try:
        need(time.time() + p['reserved_execution_and_return_seconds'] < deadline,
             'Fixed startup reserve unavailable; no reset')
        setup_commands = launch['setup_commands']
        setup_ready = True

        def execute(name, command, seconds, end, acceptable=(0,)):
            return execute_owned(job, launch, env, name, command, seconds, end, acceptable)

        cursor = 0

        def stage(name, command=None, acceptable=(0,), pair_deadline=None):
            nonlocal cursor
            now = time.time()
            if name.startswith('publish_'):
                need(pair_deadline is not None, 'Fixed publication pair deadline required')
                seconds, end, child_deadline = publication_window(fixed, cursor, name, now,
                    deadline, p['return_reserve_seconds'], pair_deadline, p['publisher_max_work_seconds'])
            else:
                seconds, end, child_deadline = stage_window(fixed, cursor, name, now, deadline,
                                                            p['return_reserve_seconds'])
                if name.startswith('producer_'):
                    need(seconds <= p['producer_max_work_seconds'] and pair_deadline is not None
                         and end <= pair_deadline, 'Producer exceeds its fixed work or job window')
            command = list(commands[name] if command is None else command)
            deadline_argument = name in DEADLINE_STAGES or name.startswith(('producer_', 'publish_'))
            if deadline_argument:
                command += ['--deadline', str(child_deadline)]
            put(job / 'stage_budgets' / (name + '.json'), dict(name=name, observed_at=now,
                work_seconds=seconds, closure_seconds=CLOSURE_SECONDS, owned_deadline=end,
                child_deadline=child_deadline if deadline_argument else None,
                child_deadline_argument=deadline_argument, pair_deadline=pair_deadline,
                original_deadline=deadline, original_nominal_work_reservation_seconds=fixed[cursor][1],
                borrowed_unused_producer_seconds=max(0, seconds - fixed[cursor][1]), clock_reset=False))
            result = execute(name, command, seconds, end, acceptable=acceptable)
            cursor += 1
            return result

        phase = 'host authentication observation'
        stage('host_auth')
        phase = 'frozen model/control transfer and Drive capacity/privacy'
        stage('input')
        phase = 'existing pinned environment setup'
        need(not environment.exists(), 'Environment already exists; no repeated setup')
        for i, (command, seconds) in enumerate(setup_commands(sys.executable, environment, payload)):
            need(seconds == fixed[cursor][1], 'Setup command budget differs')
            stage('setup_' + str(i), command)
        ops = launch['interpreter_ops']
        need(sha(Path(sys.executable).resolve()) == host_observation['executable_sha256'],
             'Host interpreter changed during setup')
        witness = ops.regular_worker_interpreter(python, host_observation['executable_sha256'],
            lambda: need(time.time() + sum(w + CLOSURE_SECONDS for _, w in fixed[cursor:])
                         + p['return_reserve_seconds'] < deadline, 'Interpreter setup deadline'))
        witness.update(basis='Byte-identical path preparation from observed host; not numerical acceptance',
            host_observation_sha256=sha(job / 'HOST_INTERPRETER_OBSERVED.json'),
            numerical_compatibility_verified=False)
        put(job / 'INTERPRETER_PATH_PREPARATION.json', witness)
        phase = 'worker authentication'
        stage('worker_auth')
        phase = 'fresh noise-scoped runtime observation and admission'
        stage('runtime')
        gate = read(job / 'NOISE_RUNTIME_GATE.json')
        need(gate.get('passed') is True and gate.get('scope') == SCOPE
             and gate.get('policy_sha256') == policy_sha
             and gate.get('activation_sha256') == sha(job / 'ACTIVATION.json'),
             'Fresh noise-scoped gate unavailable')
        print('NOISE SCOPED GATE PASSED. Starting all 20 ordered jobs.', flush=True)
        for row in p['noise_jobs']:
            member, token = row['id'], member_token(row['id'])
            phase = 'noise producer ' + member
            rolling_storage_check(job, p, member, 'producer')
            pair_started = time.time()
            pair_deadline = pair_started + p['shared_job_window_seconds']
            after_pair = sum(w + CLOSURE_SECONDS for _, w in fixed[cursor + 2:]) + p['return_reserve_seconds']
            need(pair_deadline + after_pair < deadline, 'Fixed job and future save reserves unavailable')
            put(job / 'job_windows' / (token + '.json'), dict(member=member, start=pair_started,
                deadline=pair_deadline, seconds=p['shared_job_window_seconds'], original_deadline=deadline,
                producer_max_work_seconds=p['producer_max_work_seconds'],
                publisher_max_work_seconds=p['publisher_max_work_seconds'], clock_reset=False))
            stage('producer_' + token, pair_deadline=pair_deadline)
            phase = 'durable publication and local cleanup ' + member
            rolling_storage_check(job, p, member, 'publish')
            stage('publish_' + token, pair_deadline=pair_deadline)
            launch['transport_ops'].validate_job_commit(job, p, member, require_cleanup=True)
            completed_members.append(member)
            print(f'{len(completed_members)}/20 jobs saved and read back in Drive.', flush=True)
        put(job / 'output/RESULT.json', completed_output(completed_members, p))
        need(cursor == len(fixed), 'Stage sequence incomplete')
        print('All 20 noise jobs completed and saved. Saving the final small review and closure records.', flush=True)
    except BaseException as exc:
        put(job / 'FIRST_FAILURE.json', dict(phase=phase, error=repr(exc), traceback=traceback.format_exc(),
            failed_at=time.time(), original_deadline=deadline, retry=False, scientific_acceptance=False,
            noise_dispatch_started=noise_dispatch_started(job), completed_members=completed_members,
            training_started=False, native_solver_work_started=False))
        print('STOPPED AT: ' + phase + '\n' + str(exc), flush=True)
    finally:
        if setup_ready:
            try:
                save_end = min(deadline - 180, time.time() + 900)
                need(save_end - time.time() > 60, 'No complete save reserve remains')
                execute('save', [sys.executable, '-B', str(job / 'noise_remote.py'), 'save',
                    '--job', str(job), '--deadline', str(save_end - 30)],
                    save_end - time.time() - 30, save_end)
                receipt = read(job / 'SAVE_RESULT.json')
                saved = True
                print('\nFULL RESULTS SAVED IN DRIVE: ' + receipt['archive'], flush=True)
                print('DOWNLOAD THE SMALL REVIEW: ' + receipt['review_url'], flush=True)
                try:
                    close_end = min(deadline - 10, time.time() + 120)
                    need(close_end - time.time() > 60, 'Closing metadata window unavailable')
                    execute('closing_record', [sys.executable, '-B', str(job / 'noise_remote.py'), 'closure',
                        '--job', str(job), '--deadline', str(close_end - 30)],
                        close_end - time.time() - 30, close_end)
                    print('DOWNLOAD THE SAVE RECEIPT: '
                          + read(job / 'CLOSING_RECORD_SAVED.json')['closure_url'], flush=True)
                except BaseException as exc:
                    print('Main results remain saved. Additional closing record failed: ' + repr(exc), flush=True)
            except BaseException as exc:
                print('Drive return save failed: ' + repr(exc), flush=True)
        if not saved:
            try:
                import zipfile
                report = Path('/content') / ('NOISE_DIAGNOSTIC_' + attempt + '.zip')
                names = sorted(p for p in job.glob('*.json') if p.stat().st_size < 2 * 1024**2)
                need(sum(p.stat().st_size for p in names) <= 16 * 1024**2, 'Local diagnostic bound exceeded')
                with zipfile.ZipFile(report, 'x', zipfile.ZIP_DEFLATED) as z:
                    for path in names:
                        z.write(path, path.name)
                print('Download fallback: ' + str(report) + '. Also save this notebook with outputs.', flush=True)
                from google.colab import files
                files.download(str(report))
                seconds = max(0., min(300., deadline - time.time()))
                print('Automatic disconnect in at most ' + str(int(seconds)) + ' seconds.', flush=True)
                timer = threading.Timer(seconds, runtime.unassign)
                timer.daemon = True
                timer.start()
                download_window = True
            except BaseException as exc:
                print('Diagnostic download unavailable: ' + type(exc).__name__, flush=True)
        if not download_window:
            print('Requesting disconnect now.', flush=True)
            try:
                runtime.unassign()
            except BaseException as exc:
                print('Use Runtime > Disconnect and delete runtime now: ' + type(exc).__name__, flush=True)
            done.set()
