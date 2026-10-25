"""Development-only numerical screen, inert until separately approved execution.

Never constructs final AcceptedRecords or consumes final material. The available
screen does NOT substitute sixteen existing targets for the blocked64-target path.
"""
from pathlib import Path
import json,time,sys,math
from msdev.records import read,write,sha,verify,progress

COUNTS=dict(model_updates=36,batched_inverse_updates=120,native_T128=16,native_P256=16,
    student_forward_queries=512,student_JVP_queries=256,transform_observations=18,architecture_observations=7)
LIMITS=dict(overall=2700,setup=1380,worker=900,cleanup=30,finalization=120,
    required_before_worker=1050,evidence_bytes=512*1024**2,memory_bytes=8000000000)

def preflight(bundle,manifest_sha256,scope='complete_block1'):
    root=Path(bundle)
    if sha(root/'MANIFEST.json')!=manifest_sha256:raise ValueError('upload manifest identity differs')
    verify(root,read(root/'MANIFEST.json')['files'])
    spec=read(root/'VERIFICATION.json')
    if spec.get('development_only') is not True or spec.get('counts')!=COUNTS or spec.get('limits')!=LIMITS:
        raise ValueError('frozen development workload/limits differ')
    if scope=='complete_block1':
        raise ValueError('complete Block1 numerical verification blocked: distinct64-target development inputs, final-population evaluator bridge and P256 VJP unavailable')
    if scope!='available_kernels':raise ValueError('unknown verification scope')
    if not spec.get('blocked_obligations'):raise ValueError('missing wider verification limitations')
    return spec

def require_window(value,manifest_sha256,now):
    if (value.get('approved_scope')!='available_kernels' or value.get('manifest_sha256')!=manifest_sha256
            or value.get('limits')!=LIMITS or value.get('counts')!=COUNTS
            or value.get('retries')!=0 or value.get('resume') is not False):
        raise ValueError('separate explicit scope/count/limit approval required')
    start=value.get('start');deadline=value.get('deadline')
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (start,deadline)) or deadline-start!=LIMITS['overall']:
        raise ValueError('one immutable original2700-second window required')
    if not start<=now<deadline:raise TimeoutError('original verification window inactive/expired')
    if not all(type(value.get(k)) in (int,float) and math.isfinite(value[k]) for k in ('credits','rate')) or value['rate']<=0 or value['credits']<.75*value['rate']:
        raise ValueError('current approved credits/rate insufficient or missing')
    return value

def runtime(bundle,out):
    """Inside the actual worker interpreter; raw observation precedes validation."""
    import torch
    import q5_model as Q
    from msdev.cli import runtime as observe,reusable_q6_runtime
    Q.configure();value=observe()
    flags=dict(deterministic=torch.are_deterministic_algorithms_enabled(),matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,autocast=torch.is_autocast_enabled('cuda'))
    write(out,dict(observed=value,flags=flags,executable=sys.executable))
    reusable_q6_runtime(bundle,value)
    prior=read(Path(bundle)/'Q6_ENVIRONMENT.json')
    import importlib.metadata as md
    mismatches={k:dict(expected=v,observed=md.version(k)) for k,v in prior['packages'].items() if md.version(k)!=v}
    if mismatches:raise ValueError('historical complete package inventory differs: '+repr(mismatches))
    if (value['platform']!=prior['platform'] or value['threads']!=1 or value['interop_threads']!=1
            or not torch.cuda.is_available() or not flags['deterministic'] or flags['matmul_precision']!='highest'
            or flags['matmul_tf32'] or flags['cudnn_tf32'] or flags['cudnn_benchmark'] or flags['autocast']):
        raise ValueError('historical platform/device/thread/precision mismatch')
    return value

def run(bundle,out,manifest_sha256,window,*,stage_authority=None):
    bundle=Path(bundle);root=Path(out);root.mkdir(exist_ok=False)
    started=time.time();deadline=min(window['deadline']-150,started+LIMITS['worker'])
    counts=dict(model_updates=0,batched_inverse_updates=0,native_T128=0,native_P256=0,
        student_forward_queries=0,student_JVP_queries=0,transform_observations=0,architecture_observations=0)
    engine=None;journal=None;result=None
    from .staged_verification import check_factory
    check=check_factory(root,deadline,byte_limit=LIMITS['evidence_bytes'])
    try:
        preflight(bundle,manifest_sha256,'available_kernels')
        if stage_authority is None:require_window(window,manifest_sha256,time.time())
        else:
            from .staged_verification import require_window
            require_window(stage_authority,manifest_sha256,time.time())
            if window!=stage_authority:raise ValueError('kernel stage must retain original complete-workflow window')
        if window['deadline']-time.time()<LIMITS['required_before_worker']:raise TimeoutError('full worker+cleanup+finalization does not fit')
        env=runtime(bundle,root/'RUNTIME.json')
        import numpy as np
        import torch
        import q5_model as Q
        from msdev import kernels as K,pn_development as PN
        from . import models,variants as V,materials as A,readback as R,expanded as X
        from .update_journal import UpdateJournal,read_prefix
        torch.cuda.reset_peak_memory_stats();seeds=read(bundle/'training_seeds.json');seed=seeds['owners'][0]
        weights=torch.load(bundle/'D1.pt',map_location='cpu',weights_only=True)
        def array(name):return np.load(bundle/name,allow_pickle=False)
        data=tuple(torch.from_numpy(array('training_'+n+'.npy')).to('cuda') for n in ('x','v','y','dy'))
        ordinary=(torch.from_numpy(array('attack_x.npy')).to('cuda'),None,torch.from_numpy(array('attack_y.npy')).to('cuda'),None)
        from .attacks import ForwardOnlyAbsent
        ordinary=(ordinary[0],ForwardOnlyAbsent(),ordinary[2],ForwardOnlyAbsent())
        with np.load(bundle/'embedding_0.npz',allow_pickle=False) as z:
            embed=tuple(torch.from_numpy(z[n].copy()).to('cuda') for n in ('x','v','y','dy'));rho=z['rho'].copy()
        with np.load(bundle/'keys_B3_0.npz',allow_pickle=False) as z:
            key=z['v'][0].copy();pattern=torch.from_numpy(z['w_true'].copy()).to('cuda')
        basis=torch.from_numpy(array('public_basis.npy')).to('cuda').float()
        reference=Q.Q5FNO().to('cuda').eval();reference.load_state_dict(weights,strict=True)
        direction=K.physical_key_directions(rho,key,'cuda');refs=[]
        with torch.no_grad():
            for lo in range(0,128,16):check();refs.append(torch.func.jvp(reference,(embed[0][lo:lo+16],),(direction[lo:lo+16],))[1].detach())
        reference_jvp=torch.cat(refs);target=3*read(bundle/'calibration.json')['floors']['B3']
        normalizers=read(bundle/'PN_REFERENCE.json')['normalizers']['0']
        journal=UpdateJournal(root/'updates.jsonl');timings=[];core_engine=None
        def updates(name,which,n,recipe,architecture='fno32',mask=None,pn=False):
            nonlocal engine,core_engine
            check()
            if mask or name.startswith('student_'):
                variant='heal'+str(mask) if mask else 'extract_jvp' if name=='student_jvp' else 'extract_'+architecture
                engine=X.build_engine(which,seed['initialization'],seeds['ordinary_shuffle'],variant,weights=weights if mask else None)
            else:
                engine=K.Engine(which,seed['initialization'],seed['embedding_shuffle'] if pn else seeds['ordinary_shuffle'],
                    'cuda',recipe=recipe,model_factory=lambda:models.create(architecture),weights=weights,basis=basis)
            if pn and tuple(engine.scales)!=tuple(normalizers):raise ValueError('bound P-N normalizer differs')
            if mask:masks=engine.fixed_masks
            for i in range(n):
                check();journal.begin(name,engine.updates,.0003 if pn else .002);counts['model_updates']+=1
                torch.cuda.synchronize();t=time.perf_counter()
                row=PN.pn_step(engine,direction,pattern,reference_jvp,target,check) if pn else engine.step(.002)
                if mask and any(torch.count_nonzero(p.detach()[~masks[key]]) for key,p in engine.model.named_parameters() if key in masks):
                    raise ValueError('fixed pruning mask changed during healing')
                torch.cuda.synchronize();elapsed=time.perf_counter()-t;journal.finish(row)
                timings.append(dict(stage=name,update=i+1,seconds=elapsed,loss=row['loss']))
                progress(root,'update',stage=name,completed=counts['model_updates'],seconds=elapsed)
            state=A.save_tensor_record(root/(name+'.state.pt'),engine.state());anchor=journal.snapshot()
            restored=K.Engine(which,seed['initialization'],seeds['ordinary_shuffle'],'cuda',recipe=recipe,
                model_factory=lambda:models.create(architecture),basis=basis).restore(torch.load(state,map_location='cpu',weights_only=True))
            def equal(a,b):
                if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
                if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
                if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
                return a==b
            if not equal(engine.state(),restored.state()):raise ValueError('checkpoint model/optimizer/shuffle/RNG restore differs')
            write(root/(name+'.anchor.json'),dict(state_sha256=sha(state),journal=anchor,restore_equal=True))
            if name.startswith('student_'):
                check();counts['architecture_observations']+=1;xx=query_x[:16];vv=query_v[:16]
                yy,jj=torch.func.jvp(engine.model,(xx,),(vv,));ind=xx.detach().clone().requires_grad_(True)
                cotangent=torch.ones_like(yy)/yy.numel();grad=torch.autograd.grad((engine.model(ind)*cotangent).sum(),ind)[0]
                lhs=(jj*cotangent).sum();rhs=(vv*grad).sum()
                if not all(torch.isfinite(z).all() for z in (yy,jj,grad)) or not torch.allclose(lhs,rhs,rtol=2e-3,atol=2e-5):
                    raise ValueError('student forward/JVP/VJP numerical consistency: '+name)
                write(root/(name+'.derivative.json'),dict(finite=True,lhs=float(lhs),rhs=float(rhs),
                    rtol=.002,atol=.00002,development_implementation_tolerance_not_scientific_gate=True))
            if name=='core_D9':core_engine=engine
            return engine.model
        updates('core_D9',data,4,'D9');endpoint=updates('core_ordinary',ordinary,4,'forward32')
        updates('core_PN',embed,4,'B3',pn=True);updates('core_A4',data,2,'A4');updates('core_A5B3',data,2,'A5B3')
        # Existing256 validation controls/directions are a kernel fixture for
        # JVP students, not production ordinary-attack input admission.
        query_x=torch.from_numpy(array('validation_x.npy')).to('cuda')
        query_v=torch.from_numpy(array('validation_v.npy').reshape(256,2,64)[:,0].copy()).to('cuda')
        def queries(label,x,v):
            def intent(kind,lo,n):
                field='student_forward_queries' if kind=='forward' else 'student_JVP_queries'
                counts[field]+=n;write(root/f'{label}_{kind}_intent_{lo:03d}.json',dict(counts=counts))
            def observation(kind,lo,value):
                with (root/f'{label}_{kind}_{lo:03d}.npy').open('xb') as f:np.save(f,value.cpu().numpy(),allow_pickle=False)
            return X.query_batches(reference,x,v,check,intent,observation)
        ordinary_y,_=queries('ordinary',ordinary[0],None);teacher_y,teacher_j=queries('jvp',query_x,query_v)
        forward=(ordinary[0],ForwardOnlyAbsent(),ordinary_y,ForwardOnlyAbsent())
        for arch in V.ARCHITECTURES:updates('student_'+arch,forward,2,'forward32',arch)
        updates('student_jvp',(query_x,query_v,teacher_y,teacher_j),2,'D9')
        for percent in (20,50,80):updates('healing'+str(percent),ordinary,2,'forward32',mask=percent)
        # Every introduced deterministic transform gets its true F/JVP/VJP.
        observations=[];x=query_x[:16];v=query_v[:16]
        from msdev.kernels import Jacobi
        for name in (*V.TRANSFORMS,'jacobi16','jacobi256'):
            check();counts['transform_observations']+=1;torch.cuda.synchronize();t=time.perf_counter()
            model=Jacobi(reference,int(name[6:])) if name.startswith('jacobi') else models.transformed(reference,name,endpoint=endpoint)
            y,j=torch.func.jvp(model,(x,),(v,));xx=x.detach().clone().requires_grad_(True)
            cotangent=torch.ones_like(y)/y.numel();rev=torch.autograd.grad((model(xx)*cotangent).sum(),xx)[0]
            if not all(torch.isfinite(z).all() for z in (y,j,rev)):raise FloatingPointError('transform F/JVP/VJP nonfinite')
            lhs=(j*cotangent).sum();rhs=(v*rev).sum()
            if not torch.allclose(lhs,rhs,rtol=2e-3,atol=2e-5):raise ValueError('JVP/VJP inner product mismatch: '+name)
            if name.startswith('round') and (torch.count_nonzero(j) or torch.count_nonzero(rev)):raise ValueError('rounding used a surrogate derivative')
            torch.cuda.synchronize();observations.append(dict(variant=name,seconds=time.perf_counter()-t,
                finite=True,jvp_vjp_lhs=float(lhs),jvp_vjp_rhs=float(rhs),actual_rounding_derivative=name.startswith('round')))
            progress(root,'transform',variant=name)
        # Existing exact16-target monitor, reserved conservatively before entry.
        counts.update(batched_inverse_updates=120,native_T128=16,native_P256=16)
        write(root/'MONITOR_CHARGE.json',dict(counts=counts,partial_work_charged_conservatively=True))
        from q6_train import evaluate
        check();monitor=evaluate(core_engine,bundle,root/'monitor',deadline);check()
        if monitor.get('complete') is not True:raise ValueError('incomplete monitor')
        observed=read_prefix(root/'updates.jsonl')
        if observed['completed_updates']!=COUNTS['model_updates'] or not observed['complete'] or counts!=COUNTS:
            raise ValueError('saved count/journal reconciliation differs')
        result=dict(complete_available_screen=True,complete_block1=False,counts=counts,limits=LIMITS,
            timings=timings,transforms=observations,journal=observed,seconds=time.time()-started,
            peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved(),
            blocked=['64-target new evaluator','P256 challenger','full-scale journal/storage','broader unresolved scientific interfaces'],
            runtime=env,follow_on_authorized=False)
    except BaseException as exc:
        error=None
        if engine is not None:
            try:
                from .materials import save_tensor_record
                save_tensor_record(root/'FAILED_STATE.pt',engine.state())
            except BaseException as save:error=repr(save)
        result=dict(complete_available_screen=False,complete_block1=False,error=repr(exc),counts=counts,
            seconds=time.time()-started,preservation_error=error,journal=None if journal is None else journal.snapshot())
    check.storage()
    write(root/'RESULT.json',result);return 0 if result.get('complete_available_screen') else 4
