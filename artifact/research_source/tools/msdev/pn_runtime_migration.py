"""Isolated, prospective PN runtime compatibility verification. Import is inert.

No receipt means no migrated closure. Historical parity is never relabelled.
Only the generated notebook's separately approved CPU reservation permits work.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import struct
import sys
import sysconfig
import time
from .records import read, write, sha, verify, progress

OUTPUT = 'OperatorMark_PN_Runtime_Migration_20261005'
OVERALL, SETUP, WORK, CLEANUP, FINALIZE = 2700, 1380, 900, 30, 120
ENVELOPE = WORK + CLEANUP + FINALIZE
LIMITS = dict(projections=6, detector_grids=3, FJK_sets=3, inverse_input_updates=120,
              native_forward_calls=38, historical_model_updates=0, synthetic_updates=6)
SCOPE = 'PN-marked-closure: fixed six projections, three detectors/FJK, D3 inverse, six native fixtures, tiny kernels'


def exact(a, b, where='record'):
    """Exact finite saved numeric values, including signed zero; no tolerances."""
    if type(a) is not type(b): raise ValueError(where+': type differs')
    if isinstance(a, float):
        if not math.isfinite(a) or not math.isfinite(b) or struct.pack('!d',a)!=struct.pack('!d',b):
            raise ValueError(where+': float differs/nonfinite')
    elif isinstance(a, dict):
        if a.keys()!=b.keys(): raise ValueError(where+': keys differ')
        for k in a: exact(a[k],b[k],where+'/'+k)
    elif isinstance(a, (list,tuple)):
        if len(a)!=len(b): raise ValueError(where+': length differs')
        for i,(x,y) in enumerate(zip(a,b)): exact(x,y,where+'/'+str(i))
    elif a!=b: raise ValueError(where+': value differs')


def target_build(record):
    # Whitespace is retained verbatim in the record; semantic date parsing only
    # identifies the prospectively specified build BEFORE exact binding.
    raw=record['python']
    import re
    pattern=r'3\.13\.16 \(main, Oct\s+1 2026, 15:40:24\) \[GCC 13\.3\.0\]'
    if (record['version_info'] != [3,13,16,'final',0] or record['implementation']!='CPython'
            or re.fullmatch(pattern,raw) is None or record['compiler']!='GCC 13.3.0'
            or record['python_utf8_sha256']!=hashlib.sha256(raw.encode('utf-8')).hexdigest()):
        raise ValueError('not the prospectively specified CPython build; raw string preserved')


def observed_interpreter_identity(check=lambda:None):
    """Raw identity only; observation must survive a later target rejection."""
    std=Path(sysconfig.get_path('stdlib')).resolve(); members={}
    for current,dirs,files in os.walk(std):
        dirs[:]=sorted(d for d in dirs if d not in ('site-packages','dist-packages','__pycache__'))
        for name in sorted(files):
            if name.endswith(('.py','.so','.zip')):
                check();p=Path(current)/name;members[p.relative_to(std).as_posix()]=sha(p,check)
    raw=sys.version
    r=dict(python=raw,python_utf8_sha256=hashlib.sha256(raw.encode('utf-8')).hexdigest(),
           version_info=list(sys.version_info),implementation=platform.python_implementation(),
           compiler=platform.python_compiler(),build=list(platform.python_build()),
           executable_sha256=sha(Path(sys.executable).resolve(),check),stdlib_files=members,
           SOABI=sysconfig.get_config_var('SOABI'),MULTIARCH=sysconfig.get_config_var('MULTIARCH'))
    return r


def interpreter_identity(check=lambda:None):
    """Retain the existing strict gate for every execution/runtime consumer."""
    r=observed_interpreter_identity(check);target_build(r);return r


def capture(base_python, destination):
    """CPU stdlib-only probe in the explicit base interpreter; no reservation."""
    import subprocess
    destination=Path(destination)
    if OUTPUT in destination.parts:
        raise ValueError('pre-reservation diagnostic must be outside reservation directory')
    if destination.exists(): raise FileExistsError('candidate already captured; no overwrite')
    logs=destination.with_name(destination.name+'.capture');logs.mkdir(parents=True,exist_ok=False)
    cmd=[str(base_python),'-B','-m','msdev.pn_runtime_migration','probe','--out',str(destination)]
    cwd=Path(__file__).resolve().parents[1]
    write(logs/'REQUEST.json',dict(command=cmd,cwd=str(cwd),requested_base_interpreter=str(base_python),
          started_epoch=time.time(),source_sha256=sha(Path(__file__)),observation_only=True))
    child=None;error=None;timed_out=False;start=time.monotonic()
    # Stdlib-only child, no descendants. Keep the original 60s total bound,
    # reserving its last five seconds for killing/reaping this exact child.
    with (logs/'stdout.bin').open('xb') as stdout, (logs/'stderr.bin').open('xb') as stderr:
        try:
            child=subprocess.Popen(cmd,cwd=cwd,stdout=stdout,stderr=stderr)
            try:child.wait(timeout=max(0,55-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                timed_out=True;child.kill();child.wait(timeout=max(0,60-(time.monotonic()-start)))
        except BaseException as exc:
            error=repr(exc)
            if child is not None and child.poll() is None:
                try:child.kill();child.wait(timeout=max(0,60-(time.monotonic()-start)))
                except BaseException as cleanup_error:error+='; cleanup: '+repr(cleanup_error)
        finally:
            for stream in (stdout,stderr):stream.flush();os.fsync(stream.fileno())
    rc=None if child is None else child.poll()
    write(logs/'RESULT.json',dict(command=cmd,return_code=rc,timed_out=timed_out,error=error,
          child_reaped=child is not None and rc is not None,seconds=time.monotonic()-start,
          stdout='stdout.bin',stderr='stderr.bin',
          process_observation=str(destination.with_name(destination.name+'.process.json')),
          observation_present=destination.exists(),compatibility_passed=False))
    if error or timed_out or rc!=0:
        raise RuntimeError('interpreter observation failed; preserve '+str(logs/'RESULT.json'))
    observed=read(destination)
    try:target_build(observed)
    except BaseException as exc:
        write(logs/'TARGET_VALIDATION.json',dict(passed=False,error=repr(exc),observation_sha256=sha(destination)))
        raise
    write(logs/'TARGET_VALIDATION.json',dict(passed=True,observation_sha256=sha(destination),compatibility_passed=False))
    return observed


def inspect_inputs(bundle,out):
    """CPU-only safe deserialization, without importing/constructing any model.

    The host's explicit interpreter is only a file parser here, NOT scientific
    runtime compatibility evidence. Pinned worker versions are checked separately.
    """
    import torch
    import numpy as np
    bundle,out=Path(bundle),Path(out);p=preflight(bundle,sha(bundle/'MANIFEST.json'))
    x=np.load(bundle/'validation_x.npy',allow_pickle=False);basis=np.load(bundle/'public_basis.npy',allow_pickle=False)
    schemas={};base=None
    for row in p['models']:
        tensors=torch.load(bundle/row['model'],map_location='cpu',weights_only=True)
        if not isinstance(tensors,dict) or not tensors:raise ValueError('safe state dictionary required')
        schema={}
        for name,t in tensors.items():
            if not isinstance(t,torch.Tensor) or t.device.type!='cpu' or not torch.isfinite(t).all():raise ValueError('invalid starting tensor')
            schema[name]=dict(shape=list(t.shape),dtype=str(t.dtype))
        if base is None:base=schema
        elif schema!=base:raise ValueError('owner/PN starting tensor schema differs')
        schemas[row['id']]=dict(sha256=sha(bundle/row['model']),tensors=schema)
        projection=torch.load(bundle/row['projection'],map_location='cpu',weights_only=True)
        if projection.keys()!={'forward','jacobian'}:raise ValueError('projection dictionary differs')
        for key,t in projection.items():
            if not isinstance(t,torch.Tensor) or t.dtype!=torch.float64 or t.device.type!='cpu' or not torch.isfinite(t).all():raise ValueError('projection dtype/finite evidence')
            shape=(len(x),len(basis)) if key=='forward' else (len(x),len(basis),x.shape[-1])
            if tuple(t.shape)!=shape:raise ValueError('projection population/shape differs')
            schemas[row['id']][key]=dict(shape=list(t.shape),dtype=str(t.dtype))
    design=np.load(bundle/p['designs'],allow_pickle=False)
    if design.dtype!=np.float64 or design.shape!=(16,8,8) or not np.isfinite(design).all():raise ValueError('saved D3 design shape/dtype/finite')
    write(out,dict(passed=True,manifest=sha(bundle/'MANIFEST.json'),references_sha256=sha(bundle/'REFERENCES.json'),models=schemas,
          design=dict(sha256=sha(bundle/p['designs']),shape=list(design.shape),dtype=str(design.dtype)),
          parser=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,executable=sys.executable),
          scientific_runtime_compatibility=False,model_construction=False,model_execution=False,weights_only=True))


def cpu_inspection(base_python,bundle,destination):
    import subprocess
    destination=Path(destination)
    if destination.exists():
        r=read(destination)
        if not r.get('passed') or r['manifest']!=sha(Path(bundle)/'MANIFEST.json') or r['references_sha256']!=sha(Path(bundle)/'REFERENCES.json'):
            raise ValueError('CPU input inspection binding differs')
        return r
    subprocess.run([str(base_python),'-B','-m','msdev.pn_runtime_migration','inspect-inputs','--bundle',str(bundle),'--out',str(destination)],
        cwd=Path(bundle)/'code',check=True,timeout=60,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    return read(destination)


def preflight(bundle, manifest, check=lambda:None):
    bundle=Path(bundle)
    if sha(bundle/'MANIFEST.json',check)!=manifest: raise ValueError('migration manifest differs')
    files=read(bundle/'MANIFEST.json')['files'];verify(bundle,files)
    p=read(bundle/'MIGRATION.json')
    if p['limits']!=LIMITS or p['seconds']!=dict(overall=OVERALL,setup=SETUP,worker=WORK,cleanup=CLEANUP,finalization=FINALIZE):
        raise ValueError('frozen workload/window differs')
    refs=read(bundle/'REFERENCES.json')
    for name,row in refs.items():
        check();producer=read(bundle/row['producer_manifest'])
        if sha(bundle/row['producer_manifest'],check)!=row['producer_identity']:
            raise ValueError('producer manifest drift: '+name)
        if producer['files'].get(row['original_filename'])!=row['sha256'] or files.get(name)!=row['sha256']:
            raise ValueError('reference not producer-bound: '+name)
    if len(p['models'])!=6 or len(p['detectors'])!=3: raise ValueError('comparison population differs')
    ids=[f'D{i}' for i in range(1,4)]+[f'PN_B3_3_D{i}' for i in range(1,4)]
    if sorted(r['id'] for r in p['models'])!=sorted(ids) or [r['id'] for r in p['detectors']]!=ids[3:]:raise ValueError('prescribed owner/model roster differs')
    for row in p['models']:
        for key in ('model','projection'):
            if row[key] not in refs: raise ValueError('missing comparison reference')
    closure=read(bundle/'HISTORICAL_CLOSURE.json')
    for i,row in enumerate(p['detectors']):
        for key in ('detector','evaluation','product'):
            if row[key] not in refs: raise ValueError('missing comparison reference')
        e=read(bundle/row['evaluation'])
        if e['inverse_steps']!=120 or len(e['inverse_trace'])!=120: raise ValueError('saved inverse trace absent')
        product=read(bundle/row['product']);model=next(r for r in p['models'] if r['id']==row['id'])
        if product['id']!=row['id'] or product['owner']!=i or product['factor']!=3 or product['updates']!=320 or product['model_sha256']!=files[model['model']] or not product['qualification']['qualified'] or not product['qualification']['complete']:
            raise ValueError('selected PN model/qualification correspondence differs')
        exact(product['qualification']['theta'],closure['owners'][i]['theta'],'fixed PN theta')
        exact(product['qualification'],read(bundle/row['detector'])['qualification'],'producer detector qualification')
        old_projection=read(bundle/f'baseline/D{i+1}_projection.json')
        if old_projection['model']!=files[f'D{i+1}.pt'] or old_projection['projection']!=files[f'baseline/D{i+1}_projection.pt']:
            raise ValueError('owner/projection producer correspondence differs')
    if p['designs'] not in refs or len(p['tasks'])!=16 or len(p['native_references'])!=6:
        raise ValueError('inverse/native references incomplete')
    for name in [p['designs']]+p['tasks']+p['native_references']:
        if name not in refs: raise ValueError('unbound reference: '+name)
    if sha(bundle/'HISTORICAL_CLOSURE.json')!=p['historical_closure_sha256']:
        raise ValueError('historical closure changed')
    import ast
    binding=read(bundle/'FJK_PREFIX_BINDING.json')
    if sha(bundle/binding['source'])!=binding['sha256'] or sha(bundle/binding['derived_file'])!=binding['derived_sha256']:
        raise ValueError('frozen FJK source binding drift')
    old=next(n for n in ast.parse((bundle/binding['source']).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    new=next(n for n in ast.parse((bundle/binding['derived_file']).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_fjk')
    dump=lambda nodes:ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)
    if dump(old.body[:binding['prefix_statements']])!=dump(new.body[:-1]) or not isinstance(new.body[-1],ast.Return) or ast.unparse(new.body[-1].value)!='rows':
        raise ValueError('FJK prefix is not the frozen formula')
    if hashlib.sha256(dump(new.body[:-1]).encode()).hexdigest()!=binding['ast_sha256']:raise ValueError('FJK AST binding differs')
    return p


def flags(torch):
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                default_dtype=str(torch.get_default_dtype()),matmul=torch.get_float32_matmul_precision(),
                tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                autocast=torch.is_autocast_enabled('cuda'))


FLAGS=dict(deterministic=True,default_dtype='torch.float32',matmul='highest',tf32=False,cudnn_tf32=False,autocast=False)


def runtime_record(bundle,candidate,check=lambda:None):
    import importlib.metadata as md
    import torch
    import q5_model as M
    from .cli import runtime
    M.configure();current=runtime();f=flags(torch)
    target_build(candidate);exact(interpreter_identity(check),candidate,'candidate interpreter')
    old=read(Path(bundle)/'HISTORICAL_CLOSURE.json')['runtime']
    expected=dict(old,python=candidate['python'])
    exact(current,expected,'only Python migration is approved')
    if f!=FLAGS or not torch.cuda.is_available(): raise ValueError('precision/determinism/CUDA mismatch')
    # Every historical distribution pin, not just the seven headline packages.
    packages={k:md.version(k) for k in read(Path(bundle)/'Q6_ENVIRONMENT.json')['packages']}
    exact(packages,read(Path(bundle)/'Q6_ENVIRONMENT.json')['packages'],'dependency versions')
    from q6_common import THREAD_ENV
    if any(os.environ.get(k)!=v for k,v in THREAD_ENV.items()):raise ValueError('thread environment drift')
    return dict(environment=current,flags=f,packages=packages,interpreter=candidate)


def libraries(check=lambda:None):
    """Actual loaded ELF mappings, captured again after work, not historical hashes."""
    site=Path(sysconfig.get_path('purelib')).resolve();result={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split(None,5)
        if len(parts)!=6 or not parts[5].startswith('/') or '.so' not in parts[5]:continue
        p=Path(parts[5])
        if not p.is_file():raise ValueError('loaded library is missing/deleted: '+str(p))
        try:key='site:'+p.resolve().relative_to(site).as_posix()
        except ValueError:key='system:'+str(p.resolve())
        if key not in result:check();result[key]=sha(p,check)
    if not result:raise ValueError('loaded-library evidence absent')
    return result


def check_libraries(expected,check=lambda:None):
    site=Path(sysconfig.get_path('purelib'))
    for key,digest in expected.items():
        p=site/key[5:] if key.startswith('site:') else Path(key[7:])
        check()
        if sha(p,check)!=digest:raise ValueError('runtime binary drift: '+key)


def reserve(bundle,root,manifest,candidate_file,credits,rate,approved=False,clock=time.time,input_inspection=None):
    p=preflight(bundle,manifest);candidate=read(candidate_file);target_build(candidate)
    if input_inspection is None:raise ValueError('CPU safe input inspection required before reservation')
    inp=read(input_inspection)
    if not inp.get('passed') or inp['manifest']!=manifest or inp.get('weights_only') is not True or inp.get('model_execution') is not False:
        raise ValueError('CPU input inspection does not bind this package')
    if approved is not True or not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in (credits,rate)) or rate<=0 or credits<rate*.75:
        raise ValueError('explicit full-scope approval and current credits/rate covering 0.75h required')
    root=Path(root)
    if root.name!=OUTPUT:raise ValueError('separate migration output required')
    now=clock()
    if root.exists():
        s=read(root/'STATUS.json')
        if (s['manifest']!=manifest or s['candidate_sha256']!=sha(candidate_file) or s['deadline']!=s['start']+OVERALL
                or s['input_inspection_sha256']!=sha(input_inspection) or s['state']!='UNSTARTED' or s['launch_started'] or s['closed'] or now>=s['deadline']):
            raise ValueError('consumed/drifted/expired window; no retry or reset')
        return s
    root.mkdir();write(root/'CANDIDATE_INTERPRETER.json',candidate)
    write(root/'CPU_INPUT_CHECKS.json',inp)
    s=dict(state='UNSTARTED',start=now,deadline=now+OVERALL,manifest=manifest,candidate_sha256=sha(root/'CANDIDATE_INTERPRETER.json'),
           input_inspection_sha256=sha(root/'CPU_INPUT_CHECKS.json'),approval=dict(credits=credits,rate_per_hour=rate,full_scope=True),setup_started=False,launch_started=False,closed=False)
    # Candidate records use the same canonical writer, so their identity is stable.
    if s['candidate_sha256']!=sha(candidate_file):raise ValueError('candidate serialization mismatch')
    write(root/'STATUS.json',s);return s


def owner(bundle):
    from .cli import module
    import ctypes
    if os.name!='posix' or ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
        raise OSError('existing POSIX subreaper ownership required')
    return module(Path(bundle)/'code/q5_colab_supervisor.py','migration_owned')


def bound_candidate(root):
    root=Path(root);s=read(root/'STATUS.json')
    if s['deadline']!=s['start']+OVERALL or sha(root/'CANDIDATE_INTERPRETER.json')!=s['candidate_sha256']:
        raise ValueError('immutable window/candidate binding differs')
    if sha(root/'CPU_INPUT_CHECKS.json')!=s['input_inspection_sha256']:raise ValueError('CPU input inspection drift')
    inp=read(root/'CPU_INPUT_CHECKS.json')
    if inp['manifest']!=s['manifest'] or not inp['passed'] or not inp['weights_only'] or inp['model_execution'] is not False:
        raise ValueError('CPU inspection binding/scope differs')
    record=read(root/'CANDIDATE_INTERPRETER.json');target_build(record);return record


def execution_env():
    from q6_common import THREAD_ENV
    return dict(os.environ,**THREAD_ENV,CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled',WANDB_DISABLED='true')


def setup_environment(bundle,root,environment,base_python):
    from .colab_environment import owned,setup_commands
    root,bundle,environment=Path(root),Path(bundle),Path(environment)
    s=read(root/'STATUS.json');preflight(bundle,s['manifest']);bound_candidate(root)
    if s['setup_started'] or s['closed'] or s['launch_started'] or environment.exists():raise ValueError('one-use setup; no retries')
    if time.time()+ENVELOPE>s['deadline']:raise TimeoutError('full remaining envelope unavailable')
    s['setup_started']=True;write(root/'STATUS.json',s,replace=True)
    start=time.time();end=min(start+SETUP,s['deadline']-ENVELOPE);own=owner(bundle)
    try:
        # Explicit base process, never the notebook kernel's imported packages.
        commands=[([str(base_python),'-B','-m','msdev.pn_runtime_migration','check-base','--root',str(root)],60)]
        commands+=setup_commands(str(base_python),environment,bundle)
        py=environment/'bin/python'
        commands+=[([str(py),'-B','-m','msdev.pn_runtime_migration','runtime','--bundle',str(bundle),'--root',str(root)],60)]
        for i,(cmd,seconds) in enumerate(commands):
            owned(cmd,seconds,end,root/f'setup/{i}',own,cwd=bundle/'code',env=execution_env())
        if time.time()>end:raise TimeoutError('setup ceiling')
        write(root/'ENVIRONMENT_READY.json',dict(python=str(py),base_python=str(base_python),seconds=time.time()-start,
              session_start=s['start'],manifest=s['manifest'],candidate_sha256=s['candidate_sha256'],requirements=sha(bundle/'requirements-colab.txt')))
        return py
    except BaseException as exc:
        s=read(root/'STATUS.json');s.update(state='STOPPED',reason=repr(exc));write(root/'STATUS.json',s,replace=True)
        write(root/'SETUP_FAILURE.json',dict(error=repr(exc),seconds=time.time()-start,retry=False));raise


def tensor_exact(actual,reference,name):
    import torch
    if isinstance(actual,dict):
        if not isinstance(reference,dict) or actual.keys()!=reference.keys():raise ValueError(name+': tensor keys')
        for k in actual:tensor_exact(actual[k],reference[k],name+'/'+k)
    elif actual is None:
        if reference is not None:raise ValueError(name+': missing tensor')
    elif (not isinstance(actual,torch.Tensor) or not isinstance(reference,torch.Tensor) or
          actual.shape!=reference.shape or actual.dtype!=reference.dtype or
          not torch.isfinite(actual).all() or not torch.isfinite(reference).all() or
          not torch.equal(actual.detach().cpu().contiguous().view(torch.uint8),reference.detach().cpu().contiguous().view(torch.uint8))):
        raise ValueError(name+': tensor bits/shape/dtype/nonfinite differ')


def tiny_kernels(device='cpu'):
    """Exactly six tiny updates; independent direct Adam arithmetic, not FNOs."""
    import torch
    from . import kernels as K,rules
    from .pn_marked_closure import attack_engine
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.w=torch.nn.Parameter(torch.arange(12,dtype=torch.float64).reshape(3,4)/20)
        def forward(self,x):return (x@self.w).tanh()
    x=torch.arange(192,dtype=torch.float64,device=device).reshape(64,3)/200
    data=(x,torch.ones_like(x)/7,torch.ones(64,4,dtype=torch.float64,device=device)*.2,torch.ones(64,4,dtype=torch.float64,device=device)*.04)
    basis=torch.arange(1,17,dtype=torch.float64,device=device).reshape(4,4)/20
    seeds=dict(ordinary_shuffle=19,owners=[dict(initialization=17,shuffle=23)])
    result=[]
    for cost in (False,True):
        e=attack_engine(K,data,Tiny().state_dict(),seeds,0,device,basis,cost,model_factory=Tiny)
        if e.optimizer.state or e.updates or e.sampler.batch!=(16 if cost else 32):raise ValueError('attack initialization')
        sampler=K.Sampler(64,23 if cost else 19,16 if cost else 32)
        for _ in range(3):
            ids=sampler.next();before=e.model.w.detach().clone();state=e.optimizer.state.get(e.model.w,{})
            m=state.get('exp_avg',torch.zeros_like(before)).clone();v=state.get('exp_avg_sq',torch.zeros_like(before)).clone()
            loss=K.loss_with_scales(e.model,tuple(a[ids] for a in data),e.recipe,e.scales,basis=basis)
            g=torch.autograd.grad(loss,e.model.w)[0]
            n=e.updates+1;m=.9*m+.1*g;v=.999*v+.001*g.square()
            expected=before-.002*(m/(1-.9**n))/((v/(1-.999**n)).sqrt()+1e-8)
            row=e.step(.002)
            if not torch.allclose(e.model.w,expected,rtol=1e-12,atol=1e-14):raise ValueError('tiny Adam arithmetic')
            if not torch.equal(e.sampler.order,sampler.order) or e.sampler.offset!=sampler.offset:raise ValueError('utility sampling')
            result.append(dict(recipe=e.recipe,update=n,loss=row['loss']))
    b2=rules.b2(.25,[.5,.75],[1.,1.25,1.5])
    if not b2['positive'] or b2['margin']!=.25 or b2['rank']!=1:raise ValueError('B2 rank kernel')
    from .rules import relative_rows
    import numpy as np
    rr=relative_rows(np.array([[3.,4.]]),np.array([[6.,8.]]))
    if rr[0]['relative']!=.5:raise ValueError('B2 relative kernel')
    return dict(updates=len(result),rows=result,B2=b2,relative=rr)


def worker(bundle,root):
    bundle,root=Path(bundle),Path(root);g=read(root/'CONSUMED.json');start=time.time();counts=dict.fromkeys(LIMITS,0)
    def check():
        if time.time()>=g['work_deadline']:raise TimeoutError('900-second compatibility ceiling; no retry')
    try:
        p=preflight(bundle,g['manifest'],check);r=runtime_record(bundle,bound_candidate(root),check)
        write(root/'RUNTIME.json',r);write(root/'LIBRARIES_START.json',libraries(check))
        import torch
        import numpy as np
        import q5_model as M
        from . import scoring,rules
        from .migration_fjk import evaluate_fjk
        from .pn_marked_closure import append
        load=lambda name:np.load(bundle/name,allow_pickle=False)
        # Safe CPU inspection of all starting weights BEFORE model execution.
        template=M.Q5FNO().cpu().state_dict();states={}
        for row in p['models']:
            st=torch.load(bundle/row['model'],map_location='cpu',weights_only=True)
            if st.keys()!=template.keys():raise ValueError('starting model keys')
            for k in st:
                if st[k].shape!=template[k].shape or st[k].dtype!=template[k].dtype or not torch.isfinite(st[k]).all():raise ValueError('starting tensor shape/dtype/finite')
            states[row['id']]=st
        write(root/'STARTING_TENSORS.json',dict(passed=True,models=[dict(id=r['id'],sha256=sha(bundle/r['model'])) for r in p['models']]))
        check();write(root/'TINY_KERNELS.json',tiny_kernels('cuda'));counts['synthetic_updates']=6
        basis=torch.from_numpy(load('public_basis.npy')).to('cuda');x=torch.from_numpy(load('validation_x.npy')).to('cuda')
        actual={}
        def model(id):
            m=M.Q5FNO().to('cuda');m.load_state_dict(states[id],strict=True);return m.eval()
        (root/'projections').mkdir()
        for row in p['models']:
            check();progress(root,'projection',model=row['id']);t=time.monotonic()
            obj=scoring.projections(model(row['id']),x,basis,check=check)
            torch.save(obj,root/'projections'/(row['id']+'.pt'));counts['projections']+=1
            tensor_exact(obj,torch.load(bundle/row['projection'],map_location='cpu',weights_only=True),row['id'])
            actual[row['id']]=obj;progress(root,'projection_complete',model=row['id'],seconds=time.monotonic()-t)
        closure=read(bundle/'HISTORICAL_CLOSURE.json')
        for i,row in enumerate(p['detectors']):
            check();id=row['id'];out=root/id;out.mkdir();progress(root,'detector_FJK',model=id)
            with np.load(bundle/f'keys_B3_{i}.npz',allow_pickle=False) as z:
                coeff,v=(torch.from_numpy(z[n].copy()) for n in ('coeff','v'))
            scores=scoring.projected_scores(actual[id],actual[f'D{i+1}'],torch.from_numpy(load('validation_controls_f64.npy').reshape(-1,64)),coeff,v,basis,math.sqrt(closure['owners'][i]['normalizers'][1]),'B3')
            saved=read(bundle/row['detector']);product=read(bundle/row['product'])
            write(out/'DETECTOR.json',dict(scores=scores,theta=product['qualification']['theta'],delta=closure['owners'][i]['delta']))
            exact(scores,saved['scores'],id+'/scores');counts['detector_grids']+=1
            def oracle(rho,p1):
                check()
                if counts['native_forward_calls']>=38:raise ValueError('native cap')
                counts['native_forward_calls']+=1
                append(root/'native_calls.jsonl',dict(ordinal=counts['native_forward_calls'],stage='D3_task',p1=p1))
                from q6_common import true_field
                return true_field(rho,p1)
            m=model(id)
            if i<2:
                rows=evaluate_fjk(m,bundle,v[0],load(f'key_truth_{i}.npy'),out/'evaluation',check,None)
                write(out/'FJK.json',{k:rows[k] for k in ('F','J','K')})
            else:
                ev=scoring.evaluate(m,bundle,v[0],load(f'key_truth_{i}.npy'),out/'evaluation',check,oracle)
                rows=ev['rows'];counts['inverse_input_updates']=len(ev['inverse_trace'])
            saved_e=read(bundle/row['evaluation'])
            for k in ('F','J','K'):exact(rows[k],saved_e['rows'][k],id+'/'+k)
            counts['FJK_sets']+=1
            # Fixed intact theta and development qualification are reused only
            # after exact score/row comparisons, never recalibrated by this run.
            if scores[0]<=product['qualification']['theta']:raise ValueError('intact detection decision differs')
            from .pn_development import qualification
            metrics=dict(saved_e['metrics'])
            metrics.update({k:rules.complete_mean(rows[k]) for k in ('F','J','K')})
            q=qualification(scores,metrics,read(bundle/'BASELINE.json')['owners'][str(i)]['metrics'],read(bundle/'calibration.json'),i)
            exact(q,product['qualification'],'intact qualification')
            exact(q,saved['qualification'],'saved detector qualification')
            write(out/'QUALIFICATION.json',q)
            if i==2:compare_inverse(bundle,root,p)
        from operatormark_numerics_v2 import pins
        from operatormark_numerics_v2.lock import FIXTURES
        from q6_common import true_field
        pins.verify_frozen_sources();native=[]
        for name,rho in FIXTURES.items():
            for method in ('f64','p256'):
                check()
                if counts['native_forward_calls']>=38:raise ValueError('native cap')
                counts['native_forward_calls']+=1
                append(root/'native_calls.jsonl',dict(ordinal=counts['native_forward_calls'],stage='fixture',fixture=name,method=method))
                field=true_field(rho,p1=method=='p256');ref=load(f'compatibility/{name}_{method}.npy')
                err=float(np.linalg.norm(field-ref)/np.linalg.norm(ref));row=dict(fixture=name,method=method,relative_l2=err,limit=1e-10,hard_gates_pass=True,passed=bool(np.isfinite(err) and err<=1e-10))
                native.append(row);write(root/f'NATIVE_{len(native)}.json',row)
                if not row['passed']:raise ValueError('existing fixture compatibility gate failed')
        if counts!=LIMITS:raise ValueError('incomplete compatibility workload')
        check();write(root/'LIBRARIES_FINAL.json',libraries(check));exact(runtime_record(bundle,r['interpreter'],check),r,'final runtime')
        write(root/'RESULT.json',dict(complete=True,counts=counts,seconds=time.time()-start,manifest=g['manifest'],scope=SCOPE))
        return 0
    except BaseException as exc:
        write(root/'RESULT.json',dict(complete=False,counts=counts,seconds=time.time()-start,error=repr(exc),scope=SCOPE));return 4


def compare_inverse(bundle,root,p):
    import numpy as np
    out=Path(root)/'PN_B3_3_D3/evaluation';old=read(Path(bundle)/p['detectors'][2]['evaluation']);new=read(out/'RESULT.json')
    a=np.load(out/'designs.npy',allow_pickle=False);b=np.load(Path(bundle)/p['designs'],allow_pickle=False)
    if a.dtype!=b.dtype or a.shape!=b.shape or not np.isfinite(a).all() or a.tobytes()!=b.tobytes():raise ValueError('D3 design tensor bits differ')
    for key in ('inverse_trace','inverse_steps','batch_loss_divisor','rows','metrics','complete','score','source_rule','physical_qualification'):
        exact(new[key],old[key],'D3 inverse/'+key)
    for i,name in enumerate(p['tasks']):exact(read(out/f'task_{i:02d}.json'),read(Path(bundle)/name),'D3 task '+str(i))


def independent_readback(bundle,root,deadline):
    """A fresh owned interpreter reads completed bytes; no scientific execution."""
    from .pn_marked_closure import closed_receipt,rows
    import torch
    bundle,root=Path(bundle),Path(root)
    def check():
        if time.time()>=deadline:raise TimeoutError('independent final readback deadline')
    g=read(root/'CONSUMED.json');p=preflight(bundle,g['manifest'],check);r=read(root/'RESULT.json');sup=read(root/'supervisor/RESULT.json')
    if not closed_receipt(sup) or sup['returncode']!=0 or sup['timed_out'] or not r['complete'] or r['counts']!=LIMITS:raise ValueError('worker incomplete/unclosed')
    for row in p['models']:
        check();tensor_exact(torch.load(root/'projections'/(row['id']+'.pt'),map_location='cpu',weights_only=True),torch.load(bundle/row['projection'],map_location='cpu',weights_only=True),row['id'])
    for i,row in enumerate(p['detectors']):
        out=root/row['id'];d=read(out/'DETECTOR.json');product=read(bundle/row['product'])
        exact(d['scores'],read(bundle/row['detector'])['scores'],'readback scores')
        exact(d['theta'],product['qualification']['theta'],'fixed theta')
        exact(d['delta'],read(bundle/'HISTORICAL_CLOSURE.json')['owners'][i]['delta'],'fixed delta')
        rr=read(out/'FJK.json') if i<2 else read(out/'evaluation/RESULT.json')['rows']
        for k in ('F','J','K'):exact(rr[k],read(bundle/row['evaluation'])['rows'][k],'readback '+k)
        exact(read(out/'QUALIFICATION.json'),product['qualification'],'readback qualification')
    compare_inverse(bundle,root,p)
    calls=rows(root/'native_calls.jsonl')
    if len(calls)!=38 or [r['ordinal'] for r in calls]!=list(range(1,39)):raise ValueError('native count evidence')
    if len([r for r in calls if r['stage']=='D3_task'])!=32:raise ValueError('task call count')
    for i in range(1,7):
        n=read(root/f'NATIVE_{i}.json')
        if n['limit']!=1e-10 or not n['passed'] or not n['hard_gates_pass'] or not math.isfinite(n['relative_l2']) or n['relative_l2']>1e-10:raise ValueError('fixture evidence incomplete')
    if read(root/'TINY_KERNELS.json')['updates']!=6:raise ValueError('tiny kernel count')
    rr=read(root/'RUNTIME.json');exact(runtime_record(bundle,rr['interpreter'],check),rr,'readback runtime')
    check_libraries(read(root/'LIBRARIES_FINAL.json'),check)
    files={f.relative_to(root).as_posix():sha(f,check) for f in root.rglob('*') if f.is_file() and not f.relative_to(root).as_posix().startswith('readback_supervisor/')}
    write(root/'INDEPENDENT_READBACK.json',dict(passed=True,counts=LIMITS,files=files,scope=SCOPE,manifest=g['manifest']))


def launch(python,bundle,root,manifest):
    from .colab_environment import owned
    from .pn_marked_closure import closed_receipt
    root,bundle=Path(root),Path(bundle);preflight(bundle,manifest);s=read(root/'STATUS.json');env=read(root/'ENVIRONMENT_READY.json')
    bound_candidate(root)
    if (s['state']!='UNSTARTED' or s['launch_started'] or s['closed'] or s['manifest']!=manifest or
        s['deadline']!=s['start']+OVERALL or env['python']!=str(python) or env['manifest']!=manifest or env['session_start']!=s['start'] or env['seconds']>SETUP or
        env['candidate_sha256']!=s['candidate_sha256'] or env['requirements']!=sha(bundle/'requirements-colab.txt')):
        raise ValueError('unused same-candidate setup/window required')
    now=time.time()
    if now+ENVELOPE>s['deadline']:raise TimeoutError('1050 seconds must remain')
    s['launch_started']=True;write(root/'STATUS.json',s,replace=True)
    write(root/'CONSUMED.json',dict(work_deadline=now+WORK,cleanup_deadline=now+WORK+CLEANUP,manifest=manifest,limits=LIMITS))
    own=owner(bundle);error=None;closed=False;passed=False
    try:
        owned([str(python),'-B','-m','msdev.pn_runtime_migration','worker','--bundle',str(bundle),'--root',str(root)],WORK,now+WORK+CLEANUP,root/'supervisor',own,cwd=bundle/'code',env=execution_env())
    except BaseException as exc:error=repr(exc)
    final_end=min(s['deadline'],time.time()+FINALIZE)
    try:
        sup=read(root/'supervisor/RESULT.json');closed=closed_receipt(sup)
        if not error and closed:
            owned([str(python),'-B','-m','msdev.pn_runtime_migration','readback','--bundle',str(bundle),'--root',str(root),'--deadline',str(final_end-30)],90,final_end,root/'readback_supervisor',own,cwd=bundle/'code',env=execution_env())
            rd=read(root/'readback_supervisor/RESULT.json')
            if not closed_receipt(rd) or not read(root/'INDEPENDENT_READBACK.json')['passed']:raise ValueError('readback closure incomplete')
            if time.time()>=final_end:raise TimeoutError('finalization expired')
            passed=True
    except BaseException as exc:error=(error+'; ' if error else '')+repr(exc)
    outcome=dict(complete=passed,closed=closed,timed_out=(read(root/'supervisor/RESULT.json').get('timed_out') if (root/'supervisor/RESULT.json').exists() else None),error=error,finalization_deadline=final_end,overall_deadline=s['deadline'],closure_launch_authorized=False)
    write(root/'OUTCOME.json',outcome)
    if passed:
        write(root/'SCOPED_COMPATIBILITY.json',dict(passed=True,scope=SCOPE,counts=LIMITS,manifest=manifest,
              references_sha256=sha(bundle/'REFERENCES.json'),historical_closure_sha256=sha(bundle/'HISTORICAL_CLOSURE.json'),
              runtime=read(root/'RUNTIME.json'),libraries=read(root/'LIBRARIES_FINAL.json'),
              evidence={n:sha(root/n) for n in ('INDEPENDENT_READBACK.json','supervisor/RESULT.json','readback_supervisor/RESULT.json','OUTCOME.json')},
              limitations='Only listed fixed comparisons; no historical interpreter equality, universal trajectory equivalence, physical qualification or closure execution authority.'))
    s.update(state='COMPLETE' if passed else 'STOPPED',closed=closed,reason=error);write(root/'STATUS.json',s,replace=True)
    def check():
        if time.time()>=final_end:raise TimeoutError('final manifest incomplete')
    files={f.relative_to(root).as_posix():sha(f,check) for f in root.rglob('*') if f.is_file()};check()
    write(root/'OUTPUT_MANIFEST.json',dict(files=files,complete=passed,scope=SCOPE))
    return outcome


def require_receipt(bundle,plan):
    """Future closure preflight: explicit bound receipt, never a version bypass."""
    bundle=Path(bundle);binding=plan.get('runtime_migration')
    if not binding:raise ValueError('scoped compatibility receipt required')
    folder=bundle/'migration_receipt';receipt=read(folder/'SCOPED_COMPATIBILITY.json')
    if sha(folder/'SCOPED_COMPATIBILITY.json')!=binding['receipt_sha256'] or sha(folder/'OUTPUT_MANIFEST.json')!=binding['output_manifest_sha256']:
        raise ValueError('migration receipt binding differs')
    om=read(folder/'OUTPUT_MANIFEST.json')
    for name in ('SCOPED_COMPATIBILITY.json','INDEPENDENT_READBACK.json','supervisor/RESULT.json','readback_supervisor/RESULT.json','OUTCOME.json'):
        if sha(folder/name)!=om['files'].get(name):raise ValueError('migration evidence drift: '+name)
    from .pn_marked_closure import closed_receipt
    if not all(closed_receipt(read(folder/n)) for n in ('supervisor/RESULT.json','readback_supervisor/RESULT.json')):raise ValueError('migration process closure absent')
    for name in ('supervisor/RESULT.json','readback_supervisor/RESULT.json'):
        s=read(folder/name)
        if s['returncode']!=0 or s['timed_out']:raise ValueError('migration process failed/timed out')
    if (not om['complete'] or receipt.get('passed') is not True or receipt.get('scope')!=SCOPE or receipt.get('counts')!=LIMITS or
        receipt['manifest']!=binding['migration_manifest_sha256'] or receipt['historical_closure_sha256']!=binding['historical_closure_sha256'] or
        read(folder/'OUTCOME.json')['complete'] is not True or read(folder/'INDEPENDENT_READBACK.json')['passed'] is not True):raise ValueError('incomplete/wrong scoped receipt')
    for name,digest in receipt['evidence'].items():
        if sha(folder/name)!=digest:raise ValueError('receipt evidence binding differs')
    mf=read(folder/'INPUT_MANIFEST.json')
    if sha(folder/'INPUT_MANIFEST.json')!=receipt['manifest']:raise ValueError('migration input manifest differs')
    # Bind every reused scientific/source/input file shared with this closure.
    files=read(bundle/'MANIFEST.json')['files']
    for name,digest in binding['shared_inputs'].items():
        if files.get(name)!=digest or mf['files'].get(name)!=digest or sha(bundle/name)!=digest:raise ValueError('tested source/input correspondence differs: '+name)
    required={n for n in mf['files'] if n.startswith(('code/','intact/','baseline/','tasks/')) or n.endswith(('.npy','.npz'))}
    required&=set(files)
    if set(binding['shared_inputs'])!=required:raise ValueError('incomplete compatibility input coverage')
    return receipt


def validate_closure_runtime(bundle,plan,environment):
    receipt=require_receipt(bundle,plan)
    exact(environment,receipt['runtime']['environment'],'migrated closure runtime')
    exact(interpreter_identity(),receipt['runtime']['interpreter'],'migrated interpreter')
    check_libraries(receipt['libraries'])
    import torch
    import importlib.metadata as md
    exact({k:md.version(k) for k in receipt['runtime']['packages']},receipt['runtime']['packages'],'migrated dependency versions')
    exact(flags(torch),receipt['runtime']['flags'],'migrated precision')
    from q6_common import THREAD_ENV
    if any(os.environ.get(k)!=v for k,v in THREAD_ENV.items()):raise ValueError('migrated thread environment drift')
    return receipt


def setup_closure_environment(bundle,root,environment):
    """Future closure route; passing migration receipt remains mandatory."""
    from .colab_environment import owned,setup_commands
    bundle,root,environment=Path(bundle),Path(root),Path(environment)
    plan=read(bundle/'CLOSURE.json');receipt=require_receipt(bundle,plan)
    exact(interpreter_identity(),receipt['runtime']['interpreter'],'closure base interpreter')
    if environment.exists():raise FileExistsError('one-use closure setup')
    s=read(root/'STATUS.json');started=time.time();deadline=min(started+1380,s['deadline']-1950)
    own=owner(bundle);python=environment/'bin/python'
    commands=setup_commands(sys.executable,environment,bundle)
    commands.append(([str(python),'-B','-m','msdev.pn_runtime_migration','closure-runtime','--bundle',str(bundle),'--root',str(root)],60))
    for i,(cmd,seconds) in enumerate(commands):owned(cmd,seconds,deadline,root/f'migrated_setup/{i}',own,cwd=bundle/'code',env=execution_env())
    write(root/'MIGRATED_ENVIRONMENT_READY.json',dict(python=str(python),requirements_sha256=sha(bundle/'requirements-colab.txt'),
          manifest_sha256=sha(bundle/'MANIFEST.json'),seconds=time.time()-started,session_start=s['session']['start'],
          scoped_compatibility_sha256=plan['runtime_migration']['receipt_sha256']))
    s=read(root/'STATUS.json');s['session']['environment_receipt']='MIGRATED_ENVIRONMENT_READY.json';write(root/'STATUS.json',s,replace=True)
    return python


def main():
    import argparse
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['probe','inspect-inputs','check-base','runtime','worker','readback','closure-runtime']);a.add_argument('--bundle',type=Path);a.add_argument('--root',type=Path);a.add_argument('--out',type=Path);a.add_argument('--deadline',type=float)
    p=a.parse_args()
    if p.mode=='probe':
        # Persist lightweight process metadata FIRST, even if the subsequent
        # executable/stdlib hashing fails. No target-version assertion here.
        raw=dict(python=sys.version,executable=sys.executable,resolved_executable=str(Path(sys.executable).resolve()),
                 version_info=list(sys.version_info),implementation=platform.python_implementation(),
                 compiler=platform.python_compiler(),build=list(platform.python_build()),pid=os.getpid(),
                 observed_epoch=time.time(),compatibility_passed=False)
        write(p.out.with_name(p.out.name+'.process.json'),raw)
        print(json.dumps(raw,ensure_ascii=False),flush=True)
        write(p.out,observed_interpreter_identity())
    elif p.mode=='inspect-inputs':inspect_inputs(p.bundle,p.out)
    elif p.mode=='check-base':exact(interpreter_identity(),bound_candidate(p.root),'base interpreter')
    elif p.mode=='runtime':write(p.root/'SETUP_RUNTIME.json',runtime_record(p.bundle,bound_candidate(p.root)))
    elif p.mode=='worker':return worker(p.bundle,p.root)
    elif p.mode=='readback':independent_readback(p.bundle,p.root,p.deadline)
    elif p.mode=='closure-runtime':
        import q5_model as M
        from .cli import runtime
        M.configure();r=runtime();validate_closure_runtime(p.bundle,read(p.bundle/'CLOSURE.json'),r)
        write(p.root/'MIGRATED_RUNTIME.json',r)
    return 0


if __name__=='__main__':raise SystemExit(main())
