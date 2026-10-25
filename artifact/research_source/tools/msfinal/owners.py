"""Pre-mark owner/B10 D9 adapter, requiring separate accepted training inputs.

Final calibration/mark qualification is deliberately not a prerequisite for
creating clean owners. This consumer does not issue training/data acceptance.
"""
from dataclasses import dataclass
import hashlib
import io
import re
import types
import time
from pathlib import Path, PurePosixPath

from operatormark.corpus.state_v2_prerequisites import AcceptedRecord
from .scientific_admission import require as require_accepted, runtime_ready, verify_material_lineage, same_corpus
from . import marking as M, materials as A, attacks as T


@dataclass(frozen=True)
class TrainingInputs:
    study_id: str
    owner_id: str
    role: str                       # O or one shared B10
    configuration: AcceptedRecord
    data: AcceptedRecord
    runtime: AcceptedRecord
    root: Path
    source_root: Path


def schedule():
    return [dict(id='main40960',parent=None,start=0,end=36864,cooldown=False,evaluate=False,recipe='D9'),
        dict(id='cooled40960',parent='main40960',start=36864,end=40960,cooldown=True,evaluate=True,recipe='D9'),
        dict(id='main81920',parent='main40960',start=36864,end=73728,cooldown=False,evaluate=False,recipe='D9'),
        dict(id='cooled81920',parent='main81920',start=73728,end=81920,cooldown=True,evaluate=True,recipe='D9')]


def monitor_names():
    return {'validation_'+n+'.npy' for n in ('x','v','y','dy','truth_f64','tangent_truth_f64')} | {'starts.npy'} | {
        f'tasks/target_{i:02d}_{method}.npy' for i in range(16) for method in ('f64','p256')}


def admit(p):
    from .backend import REQUIRED_SOURCES
    if type(p) is not TrainingInputs or p.role not in ('O','B10') or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',p.owner_id):
        raise ValueError('explicit clean-training role required')
    config=require_accepted(p.configuration,kind='main_study_final_owner_training_configuration',bindings=dict(study_id=p.study_id))
    if (config.get('domain')!='final' or config.get('approved') is not True or config.get('owner_length')!=81920
            or config.get('baseline_samples')!=410 or not isinstance(config.get('owners'),list)
            or len(set(config['owners']))!=len(config['owners']) or not config['owners']
            or (p.role=='O' and p.owner_id not in config['owners'])
            or (p.role=='B10' and p.owner_id!=config.get('shared_baseline_id'))):
        raise ValueError('clean training configuration incomplete; no implicit final population')
    monitor_indices=config.get('monitor_target_indices')
    if (not isinstance(monitor_indices,list) or len(monitor_indices)!=16 or len(set(monitor_indices))!=16
            or any(type(i) is not int or i<0 or i>=64 for i in monitor_indices)):
        raise ValueError('explicit fixed 16-of-64 calibration monitor selection required')
    screen=config.get('clean_screen_medians',{})
    if set(screen)!={'F','J'} or any(not M.finite(v) or v<=0 for v in screen.values()):
        raise ValueError('bound development clean-screen medians required')
    M.identity(config.get('clean_screen_reference_sha256'))
    bound=dict(study_id=p.study_id,owner_id=p.owner_id,role=p.role,configuration_sha256=p.configuration.sha256)
    data=require_accepted(p.data,kind='main_study_final_owner_training_inputs',bindings=bound)
    if (data.get('domain')!='final' or data.get('samples')!=4096 or data.get('directions_per_sample')!=1
            or data.get('target_scale')!=.01 or data.get('model_dtype')!='float32'):
        raise ValueError('D9 production training population/units differ')
    for key in ('admission_sha256','disjointness_sha256','production_readback_sha256'):M.identity(data.get(key))
    for key in ('initialization','shuffle'):
        if type(data.get(key)) is not int or data[key]<0:raise ValueError('persisted training streams missing')
    pins=config.get('source_pins',{})
    if not set(REQUIRED_SOURCES)<=set(pins):raise ValueError('complete component source binding required')
    repo=Path(__file__).resolve().parents[2]
    for name,h in pins.items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:raise ValueError('unsafe source inventory entry')
        path=Path(p.source_root).resolve().joinpath(*rel.parts).resolve()
        if not path.is_relative_to(Path(p.source_root).resolve()) or M.digest(path)!=M.identity(h):
            raise ValueError('training source drift: '+name)
        if name in REQUIRED_SOURCES and M.digest(repo/name)!=h:raise ValueError('running training component drift')
    for name,expected in M.SOURCE_PINS.items():
        if pins['tools/msdev/'+name]!=expected:raise ValueError('frozen D9/marking kernels changed')
    runtime=require_accepted(p.runtime,kind='main_study_final_owner_training_runtime',
        bindings=dict(bound,data_sha256=p.data.sha256,runtime_record_sha256=data['entries']['runtime']['sha256']))
    same_corpus(p.data,p.runtime)
    if runtime.get('domain')!='final':raise ValueError('final runtime domain required')
    runtime_ready(p.runtime)
    names={'x','v','y','dy','runtime'}
    if set(data.get('entries',{}))!=names:raise ValueError('exact owner training/runtime inventory required')
    paths=[]
    for name in names:
        e=data['entries'][name];rel=PurePosixPath(e['path'])
        if (rel.is_absolute() or '..' in rel.parts or '\\' in e['path'] or ':' in e['path']
                or e.get('domain')!='final' or e.get('purpose')!='owner_training'):
            raise ValueError('training custody path/split differs')
        verify_material_lineage(p.data,e)
        M.identity(e['sha256']);paths.append(e['path'])
        if type(e.get('bytes')) is not int or e['bytes']<0:raise ValueError('training byte length missing')
    if len(paths)!=len(set(paths)):raise ValueError('training asset aliases')
    if set(data.get('monitor_files',{}))!=monitor_names() or data.get('monitor_target_indices')!=monitor_indices:
        raise ValueError('complete bound D9 monitor material required; no final-test use')
    for name,e in data['monitor_files'].items():
        rel=PurePosixPath(e['path'])
        if (rel.is_absolute() or '..' in rel.parts or '\\' in e['path'] or ':' in e['path']
                or e.get('purpose')!='calibration_monitor' or e.get('domain')!='final'):
            raise ValueError('unsafe/incorrect monitor split')
        verify_material_lineage(p.data,e)
        M.identity(e['sha256'])
        if type(e.get('bytes')) is not int or e['bytes']<0:raise ValueError('monitor byte bound missing')
    return config,data


def _entry_bytes(p,e):
    root=Path(p.root).resolve();path=root.joinpath(*PurePosixPath(e['path']).parts).resolve()
    if not path.is_relative_to(root):raise ValueError('training path escape')
    b=path.read_bytes()
    if len(b)!=e['bytes'] or hashlib.sha256(b).hexdigest()!=e['sha256']:raise ValueError('training/monitor asset drift')
    return b


def _bytes(p,data,name):return _entry_bytes(p,data['entries'][name])


def execute(p,out,check,*,deadline):
    from . import cpu_scoring as CPU
    CPU.monitor_policy()
    config,data=admit(p);check()
    if not M.finite(deadline) or deadline<=time.time():raise ValueError('existing owned worker deadline required')
    # Reuse exact runtime admission without inventing a nonexistent clean-model
    # identity or requiring its future marked qualification.
    runtime_path=Path(p.root)/data['entries']['runtime']['path'];_bytes(p,data,'runtime')
    proxy=types.SimpleNamespace(prerequisites=types.SimpleNamespace(runtime=p.runtime,material_paths={'runtime':runtime_path}),
        inputs={'materials':{'runtime':{'sha256':data['entries']['runtime']['sha256']}}})
    device=A.configure_and_check(proxy)
    loaded={}
    def build(node):
        import numpy as np
        import torch
        from msdev.kernels import Engine
        if not loaded:
            values=[]
            for name,width in (('x',64),('v',64),('y',4096),('dy',4096)):
                array=np.load(io.BytesIO(_bytes(p,data,name)),allow_pickle=False)
                if array.shape!=(4096,width) or str(array.dtype)!='float32' or not np.isfinite(array).all():
                    raise ValueError('training shape/dtype/finiteness differs')
                # B10 is precisely the first 410 from the same admitted split.
                values.append(torch.from_numpy(array[:410] if p.role=='B10' else array).to(device))
            loaded['data']=tuple(values)
        return Engine(loaded['data'],data['initialization'],data['shuffle'],device,recipe='D9')
    return T._execute_nodes(schedule(),out,check,build,lambda:admit(p),
        dict(study_id=p.study_id,owner_id=p.owner_id,role=p.role,configuration_sha256=p.configuration.sha256,
             data_sha256=p.data.sha256,runtime_sha256=p.runtime.sha256),p.role,
        on_checkpoint=monitor_callback(p,config,data,check,deadline,clean_screen=True))


def monitor_callback(p,config,data,check,deadline,*,clean_screen=False):
    """Same admitted sixteen-target monitor for O/B10 and R* cooled states."""
    from . import cpu_scoring as CPU
    def monitor(engine,node,output):
        if not node['cooldown']:return None
        from q6_train import evaluate
        bundle=output/'monitor_inputs'
        if not bundle.exists():
            bundle.mkdir()
            for name,e in data['monitor_files'].items():
                path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as f:f.write(_entry_bytes(p,e))
        for name,e in data['monitor_files'].items():
            if M.digest(bundle/name)!=e['sha256']:raise ValueError('monitor staging drift')
        from . import cpu_scoring as CPU
        def defer(out,bundle,forward,jvp,trace,updates,seconds):
            return CPU.freeze_monitor(out,bundle,forward,jvp,trace,updates,seconds,p,config,data,node,check)
        check();result=evaluate(engine,bundle,output/(node['id']+'_monitor'),deadline,defer_native=defer);check()
        for name,e in data['monitor_files'].items():
            if M.digest(bundle/name)!=e['sha256']:raise ValueError('monitor mutated during evaluation')
        if clean_screen and p.role=='O' and node['id']=='cooled81920':
            means=result['means'];screen=config['clean_screen_medians']
            passed=means['forward']<=2*screen['F'] and means['jvp']<=2*screen['J']
            from msdev.records import write
            write(output/'CLEAN_SCREEN.json',dict(passed=passed,means=means,medians=screen,
                reference_sha256=config['clean_screen_reference_sha256'],engineering_adequacy=False))
            if not passed:raise ValueError('scientific clean-cohort stop: final owner failed fixed F/J screen')
        return result
    monitor.preflight=CPU.monitor_policy
    return monitor
