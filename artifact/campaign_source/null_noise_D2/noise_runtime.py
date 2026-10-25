"""Fresh noise20-only runtime observation against pre-frozen R3 bytes.

No model is constructed or evaluated. The separately issued applicability
decision, rather than these observations or an old D2 receipt, supplies the
limited noise scope. Scientific source files and historical evidence are read-only.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback

SCOPE = 'selected_noise20_only'
POLICY = 'NOISE_POLICY.json'
GATE = 'NOISE_RUNTIME_GATE.json'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            check()
            block = f.read(8 * 1024 * 1024)
            if not block:
                return h.hexdigest()
            h.update(block)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())


def read(path):
    return json.loads(Path(path).read_bytes())


def local(job, name):
    p = PurePosixPath(name)
    need(p.parts and not p.is_absolute() and '..' not in p.parts
         and '\\' not in name and ':' not in name, 'Unsafe runtime evidence path')
    path = Path(job).joinpath(*p.parts)
    need(path.resolve().is_relative_to(Path(job).resolve()) and not path.is_symlink(),
         'Runtime evidence leaves package')
    return path


def check_policy(job, policy, policy_sha, check=lambda: None):
    need(sha(Path(job)/POLICY, check) == policy_sha, 'NOISE policy bytes differ')
    need(read(Path(job)/POLICY) == policy and policy.get('scope') == SCOPE,
         'NOISE policy scope differs')
    need(hashlib.sha256(canonical(policy['expected_runtime'])).hexdigest()
         == policy['candidate_runtime_payload_sha256'], 'Frozen runtime identity differs')


def expectations(job, policy, check=lambda: None):
    expected = {}
    for role in ('interpreter', 'libraries', 'inventory', 'serializer', 'applicability'):
        record = policy['runtime_evidence'][role]
        path = local(job, record['path'])
        need(sha(path, check) == record['sha256'], 'Frozen runtime evidence differs: '+role)
        expected[role] = read(path)
    decision = expected['applicability']
    need(decision.get('accepted') is True and decision.get('scope') == SCOPE
         and decision.get('candidate_runtime_payload_sha256') == policy['candidate_runtime_payload_sha256']
         and decision.get('source_pins') == policy['source_pins']
         and decision.get('general_runtime_acceptance') is False
         and decision.get('scientific_acceptance') is False,
         'Separately frozen NOISE applicability decision is unavailable')
    rows = policy['noise_jobs']
    required = {f'O{i:02d}/noise_{level}_M3' for i in range(1, 11)
                for level in ('small', 'large')}
    need(isinstance(rows, list) and len(rows) == 20
         and {r['id'] for r in rows} == required
         and sorted(required) == decision['frozen_job_ids']
         and {r['id']:r['request_sha256'] for r in rows}
             == decision['frozen_request_sha256'],
         'Noise applicability does not bind the exact frozen20 requests')
    return expected


def member_path(policy, member):
    """A fixed roster member, never a user-provided arbitrary output path."""
    need(isinstance(member, str) and member in {r['id'] for r in policy['noise_jobs']},
         'Unknown noise runtime member')
    p = PurePosixPath(member)
    need(len(p.parts) == 2 and p.parts[0] in {f'O{i:02d}' for i in range(1, 11)}
         and p.parts[1] in ('noise_small_M3', 'noise_large_M3'),
         'Unsafe noise runtime member')
    return p


def pre_activation_identity(policy, *, python, executable_sha256, gpu, driver):
    """Pure comparison usable before any clock/download; never fills expectations."""
    expected = policy['expected_runtime']
    need(python == expected['python'], 'Python build differs from the frozen NOISE runtime')
    need(executable_sha256 == policy['host_interpreter_sha256'],
         'Host interpreter bytes differ from the frozen NOISE runtime')
    need(gpu in (expected['gpu'], expected['gpu'].removeprefix('NVIDIA ')),
         'Exactly the frozen A100 40 GB GPU is required')
    need(driver == policy['driver_version'], 'NVIDIA driver differs from the frozen NOISE runtime')
    return dict(python=python, executable_sha256=executable_sha256, gpu=gpu, driver=driver)


def driver_observation(policy):
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
        capture_output=True, text=True, check=True, timeout=15)
    rows = result.stdout.strip().splitlines()
    need(len(rows) == 1 and rows[0].count(',') == 1, 'Exactly one visible GPU is required')
    gpu, driver = (x.strip() for x in rows[0].split(','))
    return pre_activation_identity(policy, python=sys.version,
        executable_sha256=sha(Path(sys.executable).resolve()), gpu=gpu, driver=driver)


def compare_observation(observed, policy, expected):
    """Pure refusal checks, also exercised offline without importing Torch."""
    need(observed['runtime'] == policy['expected_runtime'], 'Fresh NOISE runtime differs')
    need(observed['interpreter'] == expected['interpreter'], 'Fresh interpreter/stdlib bytes differ')
    need(observed['inventory'] == expected['inventory'], 'Installed package inventory differs')
    need(observed['serializer'] == expected['serializer'], 'Tensor serializer settings or source differ')
    need(observed['modules'] == policy['expected_modules'], 'Imported numerical source differs')
    libraries = observed['libraries']
    need(isinstance(libraries, dict) and libraries and set(libraries) <= set(expected['libraries']),
         'An actually loaded library lacks frozen runtime evidence')
    need(all(value == expected['libraries'][name] for name, value in libraries.items()),
         'An actually loaded library differs from its frozen bytes')


def observe_runtime(job, policy, check=lambda: None, *, configure=True):
    """One fresh interpreter/library pass; expected records never become observations."""
    job = Path(job).resolve()
    source = job.parent/'payload/code'
    expected = expectations(job, policy, check)
    for name, digest in policy['source_pins'].items():
        check()
        need(sha(source/name, check) == digest, 'Pinned scientific source differs: '+name)
    sys.path[:0] = [str(source/'tools'), str(source)]
    from q6_common import THREAD_ENV
    if configure:
        os.environ.update(THREAD_ENV)
    else:
        need(all(os.environ.get(k) == v for k,v in THREAD_ENV.items()),
             'Thread environment changed during NOISE production')
    import q5_model as Q
    from msfinal import materials as A
    from msdev import pn_runtime_migration as P
    # Same dependency preload used by R3, without authentication or network calls.
    from msfinal.drive_transport import GoogleDriveTransfer, RefreshingMetadataToken
    if configure:
        Q.configure()
    modules = {}
    for name, digest in policy['expected_modules'].items():
        check()
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        need(path.is_relative_to(source.resolve()), 'Imported source outside pinned tree: '+name)
        actual_sha = sha(path, check)
        need(actual_sha == digest == policy['source_pins'][path.relative_to(source.resolve()).as_posix()],
             'Imported source identity differs: '+name)
        modules[name] = actual_sha
    identity = P.observed_interpreter_identity(check)
    P.target_build(identity)
    runtime = A.runtime_identity()
    need(isinstance(runtime.get('torch'), str), 'Torch runtime version is not text')
    runtime['torch'] = str.__str__(runtime['torch'])
    import torch
    # Preload the additional pure Python optimizer/autograd entry modules used
    # by noise inverse work. This constructs no tensor, model or optimizer and
    # performs no draw, forward, gradient or native solve.
    importlib.import_module('torch.optim.adam')
    importlib.import_module('torch.autograd')
    importlib.import_module('torch._dynamo')
    from torch.utils.serialization import config as serialization_config
    torch_root = Path(torch.__file__).resolve().parent
    serializer = dict(
        runtime_sha256=hashlib.sha256(canonical(runtime)).hexdigest(),
        source_hashes={name:sha(torch_root/name, check) for name in expected['serializer']['source_hashes']},
        default_protocol=torch.serialization.DEFAULT_PROTOCOL,
        storage_alignment=serialization_config.save.storage_alignment,
        compute_crc32=serialization_config.save.compute_crc32,
        skip_data=torch.serialization._serialization_tls.skip_data,
        cuda_device_count=torch.cuda.device_count(),
        storage_registry=[dict(priority=k, module=getattr(t,'__module__',None),
            name=getattr(t,'__qualname__',None), deserializer_module=getattr(d,'__module__',None),
            deserializer_name=getattr(d,'__qualname__',None)) for k,t,d in torch.serialization._package_registry],
        independently_accepted=False)
    observed = dict(runtime=runtime, interpreter=identity, modules=modules,
        inventory={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        serializer=serializer, driver=driver_observation(policy),
        libraries=P.libraries(check))
    compare_observation(observed, policy, expected)
    check()
    return observed


def activation(job, policy_sha, deadline):
    path = Path(job)/'ACTIVATION.json'
    value = read(path)
    need(value.get('scope') == SCOPE and value.get('policy_sha256') == policy_sha
         and value.get('retries') == 0 and value.get('previous_clocks_reused') is False,
         'NOISE activation binding differs')
    need(value['start'] <= time.time() < deadline <= value['deadline'], 'NOISE fixed clock is inactive or expired')
    return value, sha(path)


def observe(job, policy, policy_sha, deadline, check=lambda: None):
    job = Path(job).resolve()
    check_policy(job, policy, policy_sha, check)
    act, activation_sha = activation(job, policy_sha, deadline)
    out = job/'observations/noise_runtime'
    out.mkdir(parents=True, exist_ok=False)
    try:
        observed = observe_runtime(job, policy, check)
        put(out/'OBSERVED.json', observed)
        record = dict(schema='OperatorMark.NOISE_scoped_runtime_gate.v1', passed=True,
            scope=SCOPE, policy_sha256=policy_sha, activation_sha256=activation_sha,
            observed_at=time.time(), observation='observations/noise_runtime/OBSERVED.json',
            observation_sha256=sha(out/'OBSERVED.json'),
            applicability_sha256=policy['runtime_evidence']['applicability']['sha256'],
            candidate_runtime_payload_sha256=policy['candidate_runtime_payload_sha256'],
            deadline=act['deadline'], numerical_compatibility_replayed=False,
            general_runtime_acceptance=False, scientific_acceptance=False,
            formal_custody_assurance=False, models_called=0)
        check()
        put(job/GATE, record)
        return record
    except BaseException:
        put(out/'FAILURE.json', dict(passed=False, scope=SCOPE, traceback=traceback.format_exc()))
        raise


def verify_gate(job, policy, policy_sha, deadline, check=lambda: None, *, phase='start', member):
    """Fresh producer start/end check, no model call and no clock reset."""
    need(phase in ('start', 'end'), 'Unknown NOISE runtime recheck phase')
    job = Path(job).resolve()
    member_relative = member_path(policy, member)
    check_policy(job, policy, policy_sha, check)
    act, activation_sha = activation(job, policy_sha, deadline)
    gate = read(job/GATE)
    need(gate.get('passed') is True and gate.get('scope') == SCOPE
         and gate.get('policy_sha256') == policy_sha and gate.get('activation_sha256') == activation_sha
         and gate.get('applicability_sha256') == policy['runtime_evidence']['applicability']['sha256']
         and gate.get('candidate_runtime_payload_sha256') == policy['candidate_runtime_payload_sha256']
         and gate.get('deadline') == act['deadline'] and act['start'] <= gate['observed_at'] <= time.time()
         and gate.get('general_runtime_acceptance') is False and gate.get('scientific_acceptance') is False,
         'NOISE runtime gate does not bind this activation')
    need(sha(local(job,gate['observation']),check) == gate['observation_sha256'],
         'NOISE startup observation changed')
    record_root = job/'observations/noise_jobs'/member_relative
    destination = record_root/('RUNTIME_'+phase.upper()+'.json')
    need(not destination.exists(), 'Noise runtime recheck already exists; no retry')
    if phase == 'end':
        started = read(record_root/'RUNTIME_START.json')
        need(started.get('passed') is True and started.get('member') == member
             and started.get('policy_sha256') == policy_sha
             and started.get('activation_sha256') == activation_sha
             and started.get('startup_gate_sha256') == sha(job/GATE),
             'Noise runtime end check lacks its original member start')
    observed = observe_runtime(job, policy, check, configure=phase != 'end')
    record = dict(scope=SCOPE, member=member, phase=phase, passed=True, observed_at=time.time(),
        policy_sha256=policy_sha, activation_sha256=activation_sha,
        startup_gate_sha256=sha(job/GATE), observation=observed,
        scientific_acceptance=False, general_runtime_acceptance=False)
    put(destination,record)
    check()
    # Aliases are returned to the caller only: a persisted record cannot carry
    # its own content hash. These identify the actual scoped runtime gate and
    # exact observed runtime payload separately for projection/D6 bindings.
    return dict(gate, runtime_record_sha256=sha(job/GATE),
                runtime_payload_sha256=policy['candidate_runtime_payload_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--policy-sha', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    def check():
        if time.time() >= args.deadline:
            raise TimeoutError('NOISE runtime observation deadline reached; no retry')
    value = observe(args.job, read(args.job/POLICY), args.policy_sha, args.deadline, check)
    print(json.dumps(dict(passed=value['passed'], scope=value['scope'], models_called=0)), flush=True)
