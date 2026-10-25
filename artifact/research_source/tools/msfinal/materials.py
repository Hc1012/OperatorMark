"""Accepted final material custody. No admission issuer, seed draw or launch.

Only externally trusted formal or explicit scientific expectations admit a catalog. The
catalog is metadata, not permission to open sealed test material. Tensor loading
is lazy, CPU-first, weights-only, from the same bytes whose digest was checked.
"""
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath

from .scientific_admission import require as require_accepted, runtime_ready, verify_material_lineage, same_corpus
from msdev.records import write
from . import marking as M

PURPOSES = {'marking', 'qualification', 'attack', 'final'}
RUNTIME_FIELDS = {'python', 'executable', 'torch', 'cuda', 'cudnn', 'gpu', 'packages',
                  'threads', 'interop_threads', 'deterministic', 'matmul_precision',
                  'matmul_tf32', 'cudnn_tf32', 'cudnn_benchmark', 'thread_environment'}


def scope(admission):
    p = admission.prerequisites
    return dict(study_id=p.study_id, owner_id=p.owner_id,
                configuration_sha256=p.configuration.sha256, inputs_sha256=p.inputs.sha256,
                clean_model_sha256=p.clean_model_sha256)


def catalog(admission):
    p = admission.prerequisites
    if p.assets is None or p.assets_root is None:
        raise ValueError('materials.py requires independently accepted final asset catalog/root')
    value = require_accepted(p.assets, kind='main_study_final_assets', bindings=scope(admission))
    same_corpus(p.assets,p.inputs)
    if (value.get('domain') != 'final' or value.get('disjointness_sha256') != admission.inputs['disjointness_sha256']
            or value.get('admission_sha256') != admission.inputs['admission_sha256']
            or not isinstance(value.get('entries'), dict) or not value['entries']):
        raise ValueError('incomplete final custody/disjointness binding')
    paths = []
    for name, entry in value['entries'].items():
        rel = PurePosixPath(entry['path'])
        if (not name or rel.is_absolute() or '..' in rel.parts or '\\' in entry['path'] or ':' in entry['path']
                or not rel.parts or entry.get('purpose') not in PURPOSES or entry.get('domain') != 'final'):
            raise ValueError('unsafe/unscoped asset entry: ' + name)
        M.identity(entry['sha256'])
        if type(entry.get('bytes')) is not int or entry['bytes'] < 0:
            raise ValueError('asset byte length missing')
        verify_material_lineage(p.assets,entry)
        paths.append(entry['path'])
    if len(paths) != len(set(paths)):
        raise ValueError('aliased asset entries')
    return value


def require_opening(admission):
    p = admission.prerequisites
    if p.opening is None:
        raise ValueError('coordinated final opening remains unavailable')
    record = require_accepted(p.opening, kind='main_study_final_opening',
        bindings=dict(study_id=p.study_id, configuration_sha256=p.configuration.sha256))
    if (record.get('domain') != 'final' or record.get('complete_roster') != admission.configuration['owners']
            or record.get('models_and_calibration_frozen') is not True
            or not isinstance(record.get('asset_catalogs'), dict)
            or record['asset_catalogs'].get(p.owner_id) != p.assets.sha256):
        raise ValueError('opening does not bind complete frozen roster/catalogs')
    M.identity(record.get('closure_manifest_sha256'))
    return record


def entry(admission, name, purpose):
    value = catalog(admission)['entries'][name]
    if purpose not in PURPOSES or value['purpose'] != purpose:
        raise ValueError('asset split/purpose substitution: ' + name)
    # Do this before even resolving/statting a final payload path.
    if purpose == 'final':
        require_opening(admission)
    base = Path(admission.prerequisites.assets_root).resolve()
    path = base.joinpath(*PurePosixPath(value['path']).parts).resolve()
    if not path.is_relative_to(base):
        raise ValueError('asset escapes admitted custody root')
    return path, value


def read_bytes(admission, name, purpose):
    path, value = entry(admission, name, purpose)
    data = path.read_bytes()
    if len(data) != value['bytes'] or hashlib.sha256(data).hexdigest() != value['sha256']:
        raise ValueError('asset content drift: ' + name)
    return data


def json_asset(admission, name, purpose):
    return json.loads(read_bytes(admission, name, purpose))


def array(admission, name, purpose, shape, dtype):
    import numpy as np
    value = np.load(io.BytesIO(read_bytes(admission, name, purpose)), allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.shape != tuple(shape) or str(value.dtype) != dtype or not np.isfinite(value).all():
        raise ValueError('array shape/dtype/finiteness: ' + name)
    return value


def weights(admission, name, purpose):
    import torch
    value = torch.load(io.BytesIO(read_bytes(admission, name, purpose)), map_location='cpu', weights_only=True)
    if not isinstance(value, dict) or not value or any(not isinstance(k, str) or not torch.is_tensor(v)
            or not torch.isfinite(v).all() for k, v in value.items()):
        raise ValueError('finite tensor-only state dictionary required')
    return value


def runtime_identity():
    """Observe actual runtime; never transcribe/normalize a Python build string."""
    import importlib.metadata as md
    import sys
    import torch
    from q6_common import THREAD_ENV
    return dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        packages={k:md.version(k) for k in ('numpy','scipy','neuraloperator','tensorly','tensorly-torch')},
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(), matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        thread_environment={k:os.environ.get(k) for k in THREAD_ENV})


def admit_runtime(admission, observer=runtime_identity):
    """Exact accepted runtime + imported implementation identities, no fallback."""
    p = admission.prerequisites
    record = p.runtime.read()
    expected = M.bound_record(p.material_paths['runtime'], admission.inputs['materials']['runtime']['sha256'])
    if set(expected) != RUNTIME_FIELDS or observer() != expected:
        raise ValueError('actual runtime differs from independently accepted exact identity')
    modules = record.get('module_sources')
    required = {'q5_model', 'q6_common', 'q6_train', 'msdev.kernels', 'msdev.scoring',
                'operatormark_numerics_v2.tpfa_family', 'operatormark_numerics_v2.p1_family',
                'operatormark_numerics_v2.directions', 'operatormark_numerics_v2.task',
                'msfinal.models', 'msfinal.expanded', 'msfinal.evaluation',
                'msfinal.contract_arms', 'msfinal.prospective_contracts',
                'msfinal.p256_adjoint', 'msfinal.p256_tasks', 'msfinal.final_contracts', 'msfinal.resources'}
    if (record.get('complete_runtime_inventory') is not True or not isinstance(modules, dict)
            or not required <= set(modules)):
        raise ValueError('complete independently reviewed runtime/native source inventory required')
    for name, expected_sha in modules.items():
        module = importlib.import_module(name)
        if not getattr(module, '__file__', None) or M.digest(module.__file__) != M.identity(expected_sha):
            raise ValueError('imported runtime source drift: ' + name)
    return expected


def preflight(admission):
    """Metadata-only: do not open sealed final assets or import tensor libraries."""
    entries = catalog(admission)['entries']
    for role in M.MATERIALS:
        if role not in entries or entries[role]['purpose'] != 'marking' or entries[role]['sha256'] != admission.inputs['materials'][role]['sha256']:
            raise ValueError('material/catalog correspondence missing: ' + role)
        path, _ = entry(admission, role, 'marking')
        if path != Path(admission.prerequisites.material_paths[role]).resolve():
            raise ValueError('material path substitution: ' + role)
    for name in ('embedding_x', 'embedding_v', 'embedding_y', 'embedding_dy', 'embedding_rho',
                 'key_v', 'key_coeff', 'key_pattern', 'public_basis'):
        if name not in entries or entries[name]['purpose'] != 'marking':
            raise ValueError('marking asset missing: ' + name)
    # Explicit native/runtime acceptance, not Q6/development receipt promotion.
    runtime = admission.prerequisites.runtime.read()
    runtime_ready(admission.prerequisites.runtime)



def configure_and_check(admission):
    import q5_model
    q5_model.configure()
    observed = admit_runtime(admission)
    if observed['gpu'] is None:
        raise ValueError('accepted GPU numerical worker required; no CPU fallback')
    return 'cuda'


def load_accepted_marking_inputs(admission, check):
    preflight(admission); check()
    device = configure_and_check(admission)
    import torch
    import q5_model as Q
    from msdev import kernels as K
    data = tuple(torch.from_numpy(array(admission, 'embedding_'+n, 'marking', (128,w), 'float32')).to(device)
                 for n,w in (('x',64),('v',64),('y',4096),('dy',4096)))
    rho = array(admission,'embedding_rho','marking',(128,8,8),'float64')
    if not torch.equal(data[0].cpu(), torch.from_numpy(rho.reshape(128,64).astype('float32'))):
        raise ValueError('model controls differ from admitted float64 controls')
    nkeys = 1 + admission.calibration['wrong_key_count']
    key_v = array(admission,'key_v','marking',(nkeys,64),'float32')
    directions = K.physical_key_directions(rho, key_v[0], device)
    pattern = torch.from_numpy(array(admission,'key_pattern','marking',(4096,),'float32')).to(device)
    import numpy as np
    basis = array(admission,'public_basis','marking',(16,4096),'float64')
    coeff = array(admission,'key_coeff','marking',(nkeys,16),'float64')
    expected_pattern = coeff[0] @ basis
    expected_pattern = (expected_pattern / np.sqrt(np.mean(expected_pattern*expected_pattern))).astype('float32')
    if not np.array_equal(expected_pattern,pattern.cpu().numpy()):
        raise ValueError('true key pattern correspondence differs')
    norms = json_asset(admission,'normalizers','marking')
    if norms.get('values') != admission.inputs['normalizers'] or tuple(float(v) for v in Q.scales(data[2],data[3])) != tuple(norms['values']):
        raise ValueError('actual embedding normalization differs')
    seeds = json_asset(admission,'shuffle','marking')
    if any(type(seeds.get(k)) is not int or seeds[k] < 0 for k in ('initialization','shuffle')):
        raise ValueError('admitted initialization/shuffle state missing; never draw replacements')
    state = weights(admission,'clean_model','marking')
    reference = Q.Q5FNO().to(device).eval(); reference.load_state_dict(state, strict=True)
    reference_rows = []
    with torch.no_grad():
        for lo in range(0,128,16):
            check(); reference_rows.append(torch.func.jvp(reference,(data[0][lo:lo+16],),(directions[lo:lo+16],))[1].detach())
    return dict(data=data, weights=state, initialization=seeds['initialization'], shuffle=seeds['shuffle'],
                device=device, directions=directions, pattern=pattern, reference_jvp=torch.cat(reference_rows))


TENSOR_PUBLICATION = 'D5.exclusive-hardlink.v1'


def save_tensor_record(path, value):
    """One completed tensor. Only this writer's successful partial is unlinked.

    Hard-link creation is atomic and refuses an existing destination on Windows
    and POSIX. Unsupported filesystems fail closed; no overwrite fallback.
    Any failed publication retains its partial and any published bytes.
    """
    import torch
    path = Path(path); pending = path.with_name(path.name+'.partial')
    receipt = path.with_name(path.name+'.publication.json')
    if path.exists() or receipt.exists():
        raise FileExistsError(path)
    with pending.open('xb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    expected = M.digest(pending)
    os.link(pending, path)
    if not os.path.samefile(path, pending) or M.digest(path) != expected:
        raise ValueError('checkpoint publication mismatch')
    size = path.stat().st_size
    write(receipt,dict(schema=TENSOR_PUBLICATION,name=path.name,sha256=expected,
        bytes=size,exclusive=True,completed_tensor_files=1))
    pending.unlink()
    if pending.exists() or M.digest(path) != expected:
        raise ValueError('checkpoint publication final verification failed')
    return path


def save_marked(engine, out, check):
    import q5_model as Q
    check(); model = save_tensor_record(Path(out)/'MODEL.pt', Q.tensor_state(engine.model))
    check(); state = save_tensor_record(Path(out)/'STATE.pt', engine.state())
    write(Path(out)/'CHECKPOINT.json',dict(model_sha256=M.digest(model),state_sha256=M.digest(state),
        updates=engine.updates, recipe=engine.recipe, independent_acceptance=False))
    return model,state


def save_failed(engine,out,check):
    """Best-effort catchable-failure custody, never a completed trajectory."""
    import q5_model as Q
    check();model=save_tensor_record(Path(out)/'FAILED_MODEL.pt',Q.tensor_state(engine.model))
    check();state=save_tensor_record(Path(out)/'FAILED_STATE.pt',engine.state())
    record=dict(model_sha256=M.digest(model),state_sha256=M.digest(state),observed_updates=engine.updates,
                completed=False,independent_acceptance=False)
    write(Path(out)/'FAILED_CHECKPOINT.json',record);return record
