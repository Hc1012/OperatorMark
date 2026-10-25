"""Final-study populations with frozen scoring/task kernels, loaded lazily.

Qualification uses validation/calibration material. Final evaluation additionally
requires the single coordinated opening. No evaluator issues acceptance. Zero
truth norms remain unavailable; rows are never dropped to complete a population.
"""
import hashlib
import json
import time
from pathlib import Path

from msdev.records import write
from . import materials as A


def specification(admission, phase):
    specs = admission.configuration.get('evaluation')
    if not isinstance(specs, dict) or phase not in specs or phase not in ('qualification','final'):
        raise ValueError('explicit final-study evaluation configuration required')
    spec = specs[phase]
    validate_specification(spec,phase)
    if phase=='final':A.require_opening(admission)
    return spec


def validate_specification(spec,phase):
    """Shared fixed numerical population; this alone does not admit material."""
    if phase not in ('qualification','final'):raise ValueError('unknown numerical population')
    expected = dict(samples=256 if phase=='qualification' else 512, random_directions=2,
                    detector_controls=256 if phase=='qualification' else 128,
                    targets=64, task_batch=16, task_updates=120, task_lr=.03,
                    target_scale=.01, model_dtype='float32')
    if any(spec.get(k) != v for k,v in expected.items()) or type(spec.get('p256')) is not bool:
        raise ValueError('evaluation population/task semantics differ from proposed source contract')
    if phase=='final' and not spec['p256']:
        raise ValueError('final P256 robustness comparison required')
    return spec


def required_arrays(spec):
    n=spec['samples'];d=spec['detector_controls']
    values = dict(x=((n,64),'float32'), v=((n,2,64),'float32'), rho=((n,8,8),'float64'),
        truth=((n,4096),'float64'), tangent=((n,2,4096),'float64'), key_truth=((n,4096),'float64'),
        starts=((64,8,8),'float64'), targets=((64,64,64),'float64'),
        detector_x=((d,64),'float32'), detector_rho=((d,8,8),'float64'))
    if spec['p256']:values['targets_p256']=((64,64,64),'float64')
    return values


def preflight(admission, phase='qualification'):
    spec=specification(admission,phase);entries=A.catalog(admission)['entries']
    for suffix in required_arrays(spec):
        name=phase+'_'+suffix
        if name not in entries or entries[name]['purpose']!=phase:
            raise ValueError('evaluation.py requires admitted '+name)
    for suffix in ('key_v','key_coeff'):
        name=phase+'_'+suffix
        if name not in entries or entries[name]['purpose']!=phase:
            raise ValueError('evaluation.py requires separate admitted keys: '+name)
    clean=admission.clean_evaluation
    if phase=='qualification' and clean.get('evaluation_contract')!=spec:
        raise ValueError('clean-owner evaluation population/settings differ')
    return spec


def cache_identity(*, model_sha256, runtime_sha256, catalog_sha256, controls_sha256,
                   basis_sha256, scoring_sha256, phase):
    """Exact cache key; changing any dependency invalidates a proposed reuse."""
    from .marking import identity
    bindings=dict(model=model_sha256,runtime=runtime_sha256,catalog=catalog_sha256,
                  controls=controls_sha256,basis=basis_sha256,scoring=scoring_sha256)
    for value in bindings.values():identity(value)
    if phase not in ('qualification','final'):raise ValueError('cache phase missing')
    return hashlib.sha256(json.dumps(dict(bindings,phase=phase),sort_keys=True).encode()).hexdigest()


def evaluate_owner(model, admission, out, check):
    from .marking import digest
    return evaluate_model(model,admission,Path(out)/'evaluation',check,phase='qualification',
                          model_sha256=digest(Path(out)/'MODEL.pt'))


class ProjectionStore:
    """Optional in-worker reuse, with exact identity and byte readback.

    Only products written by this store are reused. It never discovers historical
    unbound caches by filename. A caller may share it across final model/key work.
    """
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(exist_ok=False);self.records={}

    def obtain(self,key,compute):
        from .marking import digest
        import io,torch
        if key in self.records:
            path,expected=self.records[key];data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('projection cache byte drift')
            value=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
            if value['identity']!=key:raise ValueError('projection cache binding differs')
            return value['projection']
        result=compute()
        path=A.save_tensor_record(self.root/(key+'.pt'),dict(identity=key,projection=result))
        h=digest(path);self.records[key]=(path,h)
        write(self.root/(key+'.json'),dict(identity=key,sha256=h))
        return result


def projection(model,admission,phase,model_sha256,wrapper_steps,x,basis,check,store):
    return _projection(model,admission,phase,model_sha256,wrapper_steps,x,basis,check,store,A)


def _projection(model,admission,phase,model_sha256,wrapper_steps,x,basis,check,store,access):
    from msdev.scoring import projections
    entries=access.catalog(admission)['entries']
    descriptor=hashlib.sha256(json.dumps(dict(model=model_sha256,wrapper_steps=wrapper_steps),sort_keys=True).encode()).hexdigest()
    key=cache_identity(model_sha256=descriptor,runtime_sha256=admission.prerequisites.runtime.sha256,
        catalog_sha256=admission.prerequisites.assets.sha256,controls_sha256=entries[phase+'_detector_x']['sha256'],
        basis_sha256=entries['public_basis']['sha256'],scoring_sha256=admission.configuration['source_pins']['tools/msdev/scoring.py'],phase=phase)
    # Additive white-box noise has the base Jacobian exactly. B3 consumes the
    # Jacobian only; no noisy value-based/B1 claim is made by this projection.
    projected=getattr(model,'pathwise_projection_model',model)
    compute=lambda:projections(projected,x,basis,True,check=check)
    return compute() if store is None else store.obtain(key,compute)


def detector_scores(model,admission,phase,check,*,model_sha256,wrapper_steps=None,store=None):
    """Score only; reusable for null grids without rerunning inverse tasks."""
    spec=specification(admission,phase);check()
    return _detector_scores(model,admission,phase,check,model_sha256=model_sha256,
                            wrapper_steps=wrapper_steps,store=store,access=A,spec=spec)


def _detector_scores(model,admission,phase,check,*,model_sha256,wrapper_steps,store,access,spec):
    from .marking import identity
    identity(model_sha256);check()
    if access is A:A.configure_and_check(admission)
    import numpy as np
    import torch
    import q5_model as Q
    from msdev import scoring as S
    n=spec['detector_controls'];nkeys=admission.calibration['wrong_key_count']+1
    x=access.array(admission,phase+'_detector_x',phase,(n,64),'float32')
    rho=access.array(admission,phase+'_detector_rho',phase,(n,8,8),'float64')
    if not np.array_equal(x,rho.reshape(n,64).astype('float32')):raise ValueError('detector control correspondence differs')
    v=access.array(admission,phase+'_key_v',phase,(nkeys,64),'float32')
    coeff=access.array(admission,phase+'_key_coeff',phase,(nkeys,16),'float64')
    if (not np.array_equal(v[0],access.array(admission,'key_v','marking',(nkeys,64),'float32')[0])
            or not np.array_equal(coeff[0],access.array(admission,'key_coeff','marking',(nkeys,16),'float64')[0])):
        raise ValueError('true key differs from embedded key')
    device=next(model.parameters()).device
    owner=Q.Q5FNO().to(device).eval();owner.load_state_dict(access.weights(admission,'clean_model','marking'),strict=True)
    basis=torch.from_numpy(access.array(admission,'public_basis','marking',(16,4096),'float64')).to(device)
    dx=torch.from_numpy(x).to(device)
    candidate=_projection(model,admission,phase,model_sha256,wrapper_steps,dx,basis,check,store,access)
    reference=_projection(owner,admission,phase,admission.prerequisites.clean_model_sha256,None,dx,basis,check,store,access)
    return S.projected_scores(candidate,reference,torch.from_numpy(rho.reshape(n,64)),torch.from_numpy(coeff),
        torch.from_numpy(v),basis,admission.inputs['normalizers'][1]**.5,'B3')


def evaluate_model(model, admission, out, check, *, phase, model_sha256, wrapper_steps=None, projection_store=None):
    # Production always retains strict opening/custody/runtime admission. The
    # development bridge calls the SAME private numerical body, never this gate
    # with forged AcceptedRecords or a configurable bypass flag.
    from . import cpu_scoring as CPU
    CPU.policy(admission)
    spec=preflight(admission,phase);check();A.configure_and_check(admission)
    return _evaluate_model(model,admission,out,check,phase=phase,model_sha256=model_sha256,
                           wrapper_steps=wrapper_steps,projection_store=projection_store,access=A,spec=spec)


def evaluation_binding(admission,phase,model_sha256,wrapper_steps):
    return dict(model_sha256=model_sha256,wrapper_steps=wrapper_steps,phase=phase,
        bindings=admission.bindings,assets=admission.prerequisites.assets.sha256,
        wrong_key_count=admission.calibration['wrong_key_count'])


def evaluate_diagnostics(model,admission,out,check,*,model_sha256,wrapper_steps=None,projection_store=None):
    spec=preflight(admission,'final');check();A.configure_and_check(admission)
    return _evaluate_model(model,admission,out,check,phase='final',model_sha256=model_sha256,
        wrapper_steps=wrapper_steps,projection_store=projection_store,access=A,spec=spec,components='diagnostics')


def evaluate_tasks(model,admission,out,check,*,model_sha256,diagnostic_root,wrapper_steps=None):
    from . import cpu_scoring as CPU
    CPU.policy(admission)
    spec=preflight(admission,'final');check();A.configure_and_check(admission)
    return _evaluate_model(model,admission,out,check,phase='final',model_sha256=model_sha256,
        wrapper_steps=wrapper_steps,projection_store=None,access=A,spec=spec,components='tasks',diagnostic_root=diagnostic_root)


def reuse_diagnostics(root,out,spec,check,*,expected_binding):
    from . import readback as R
    from msdev.records import read,sha
    root=Path(root);out=Path(out)
    proof=R.diagnostics(root,spec,check,expected_binding=expected_binding)
    files={}
    for name in proof['files']:
        check();target={'RESULT.json':'DIAGNOSTIC_RESULT.json','START.json':'DIAGNOSTIC_START.json'}.get(name,name)
        raw=(root/name).read_bytes();h=hashlib.sha256(raw).hexdigest()
        with (out/target).open('xb') as f:f.write(raw);f.flush();__import__('os').fsync(f.fileno())
        if sha(out/target)!=h:raise ValueError('diagnostic copy changed')
        files[target]=h
    write(out/'DIAGNOSTIC_REUSE.json',dict(files=files,binding=expected_binding))
    R.diagnostics(out,spec,check,expected_binding=expected_binding,copied=True)
    return read(out/'DIAGNOSTIC_RESULT.json')


def skip_tasks(admission,out,check,*,model_sha256,diagnostic_root,decision,member,wrapper_steps=None):
    from msdev.records import read
    spec=preflight(admission,'final');check();out=Path(out);out.mkdir(exist_ok=False)
    if decision.get('task_required') is not False or member not in decision['cell']['members']:
        raise ValueError('D2 task skip lacks whole-cell decision')
    binding=evaluation_binding(admission,'final',model_sha256,wrapper_steps)
    cached=reuse_diagnostics(diagnostic_root,out,spec,check,expected_binding=binding)
    write(out/'START.json',read(out/'DIAGNOSTIC_START.json'))
    result=dict(cached,task_status='unmeasured',task_reason=decision['reason'],conditional_complete=True,
        d2_decision=decision,d2_member=member,complete=False,complete_task_curve=False)
    write(out/'RESULT.json',result);return result


def _evaluate_model(model, admission, out, check, *, phase, model_sha256, wrapper_steps, projection_store,access,spec, components="full", diagnostic_root=None):
    from .marking import identity
    identity(model_sha256)
    if components not in ("full","diagnostics","tasks"):raise ValueError("unknown evaluation component")
    if components=="tasks" and diagnostic_root is None:raise ValueError("bound diagnostics required before tasks")
    validate_specification(spec,phase);check()
    import numpy as np
    import torch
    import q5_model as Q
    from . import cpu_scoring as CPU
    deferred=access is A and components!='diagnostics'
    if deferred:CPU.policy(admission)
    else:
        from q6_common import true_field
    from msdev import scoring as S, kernels as K
    out=Path(out);out.mkdir(exist_ok=False)
    start=time.monotonic();device=next(model.parameters()).device
    prior_mode=model.training;model.eval()
    counts=dict(inverse_input_iterations=0,targets=0,native_t128_calls=0,native_p256_calls=0,
                validation_controls=0,detector_controls=0,optimizer_model_updates=0)
    rows={k:[] for k in ('F','J','K','T','P256')}
    domain='final' if access is A else 'development'
    write(out/'START.json',dict(domain=domain,phase=phase,specification=spec,bindings=admission.bindings,
                                assets=admission.prerequisites.assets.sha256,
                                evaluation_binding=evaluation_binding(admission,phase,model_sha256,wrapper_steps)))
    try:
        if components=='tasks':
            cached=reuse_diagnostics(diagnostic_root,out,spec,check,
                expected_binding=evaluation_binding(admission,phase,model_sha256,wrapper_steps))
            rows.update(cached['rows']);scores=cached['scores'];counts.update(cached['counts'])
        else:
            arrays={k:access.array(admission,phase+'_'+k,phase,*shape_dtype) for k,shape_dtype in required_arrays(spec).items() if components=="full" or k not in ("starts","targets","targets_p256")}
            if (not np.array_equal(arrays['x'],arrays['rho'].reshape(-1,64).astype('float32'))
                    or not np.array_equal(arrays['detector_x'],arrays['detector_rho'].reshape(-1,64).astype('float32'))):
                raise ValueError('physical/model control correspondence differs')
            nkeys=admission.calibration['wrong_key_count']+1
            key_v=access.array(admission,phase+'_key_v',phase,(nkeys,64),'float32')
            coeff=access.array(admission,phase+'_key_coeff',phase,(nkeys,16),'float64')
            # Qualification true key must be the embedded key; final wrong keys are a
            # separate post-closure population, with that same true key at index zero.
            embedded_v=access.array(admission,'key_v','marking',(nkeys,64),'float32')
            embedded_coeff=access.array(admission,'key_coeff','marking',(nkeys,16),'float64')
            if not np.array_equal(key_v[0],embedded_v[0]) or not np.array_equal(coeff[0],embedded_coeff[0]):
                raise ValueError('detector true key differs from embedded key')
            x=torch.from_numpy(arrays['x']).to(device);v=torch.from_numpy(arrays['v']).to(device)
            keyed=K.physical_key_directions(arrays['rho'],key_v[0],device)
            for lo in range(0,len(x),16):
                check();c=x[lo:lo+16]
                with torch.no_grad():
                    batch=dict(F=S.relative_rows(model(c).double().cpu().numpy()*.01,arrays['truth'][lo:lo+16]),J=[])
                    for k in range(2):
                        j=_jvp(model,c,v[lo:lo+16,k])
                        batch['J']+=S.relative_rows(j.double().cpu().numpy()*.01,arrays['tangent'][lo:lo+16,k])
                    j=_jvp(model,c,keyed[lo:lo+16])
                    batch['K']=S.relative_rows(j.double().cpu().numpy()*.01,arrays['key_truth'][lo:lo+16])
                write(out/f'validation_{lo:04d}.json',batch)
                for k,rr in batch.items():rows[k]+=rr
                counts['validation_controls']+=len(c)
            if components=='diagnostics':
                scores=_detector_scores(model,admission,phase,check,model_sha256=model_sha256,wrapper_steps=wrapper_steps,
                                       store=projection_store,access=access,spec=spec)
                counts['detector_controls']=spec['detector_controls']
                write(out/'DETECTOR_SCORES.json',dict(scores=scores,wrong_key_count=nkeys-1))
        if components!='diagnostics':
            if components=="tasks":
                arrays={k:access.array(admission,phase+"_"+k,phase,*shape_dtype) for k,shape_dtype in required_arrays(spec).items() if k in ("starts","targets","targets_p256")}
            # Four independent Adam states, 16 targets each. Loss is averaged over
            # exactly 16*4096 elements. Never substitute one batch of 64.
            for lo in range(0,64,16):
                check();target=arrays['targets'][lo:lo+16]
                designs=torch.tensor(arrays['starts'][lo:lo+16].reshape(16,64),dtype=torch.float32,device=device,requires_grad=True)
                tt=torch.tensor(target.reshape(16,4096)/.01,dtype=torch.float32,device=device)
                opt=torch.optim.Adam([designs],lr=.03);trace=[]
                for step in range(120):
                    check();opt.zero_grad(set_to_none=True);loss=(model(designs)-tt).square().mean()
                    gradient=torch.autograd.grad(loss,designs)[0]
                    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():raise FloatingPointError('inverse loss/gradient')
                    designs.grad=gradient;opt.step();counts['inverse_input_iterations']+=1
                    with torch.no_grad():designs.clamp_(0,1)
                    trace.append(float(loss.detach()))
                rho=designs.detach().cpu().numpy().astype(np.float64).reshape(16,8,8)
                with (out/f'designs_{lo:02d}.npy').open('xb') as f:np.save(f,rho,allow_pickle=False)
                write(out/f'inverse_{lo:02d}.json',dict(trace=trace,updates=120,batch_size=16,batch_loss_divisor=16))
                for i,c in enumerate(rho) if not deferred else ():
                    check();counts['native_t128_calls']+=1;f=true_field(c,False)
                    rr=S.relative_rows(np.asarray(f).reshape(1,-1),target[i].reshape(1,-1))[0];rows['T'].append(rr)
                    record=dict(target=lo+i,F64=rr)
                    if spec['p256']:
                        check();counts['native_p256_calls']+=1;p=true_field(c,True)
                        pr=S.relative_rows(np.asarray(p).reshape(1,-1),arrays['targets_p256'][lo+i].reshape(1,-1))[0]
                        rows['P256'].append(pr);record['P256']=pr
                    counts['targets']+=1;write(out/f'task_{lo+i:02d}.json',record)
        if components=='full':
            scores=_detector_scores(model,admission,phase,check,model_sha256=model_sha256,wrapper_steps=wrapper_steps,
                                   store=projection_store,access=access,spec=spec)
            counts['detector_controls']=spec['detector_controls']
            write(out/'DETECTOR_SCORES.json',dict(scores=scores,wrong_key_count=nkeys-1))
        metrics={k:S.complete_mean(rr) for k,rr in rows.items() if k!='P256' or spec['p256']}
        result=dict(complete=all(v is not None for v in metrics.values()),metrics=metrics,scores=scores,
            counts=counts,rows=rows,seconds=time.monotonic()-start,domain=domain,phase=phase,specification=spec,
            physical_qualification=False,independent_acceptance=False)
        if components=='diagnostics':
            result.update(complete=False,diagnostics_complete=all(metrics[k] is not None for k in 'FJK') and all(__import__('math').isfinite(x) for x in scores),
                task_status='pending',components='diagnostics')
        elif components=='tasks':result.update(diagnostics_complete=True,task_status='measured',components='combined_D2')
        if deferred:
            result.update(complete=False,task_status=CPU.STATE,diagnostics_complete=True)
            result=CPU.freeze(out,result,admission,phase,model_sha256,wrapper_steps,check)
        write(out/'RESULT.json',result);return result
    except BaseException as exc:
        write(out/'FAILURE.json',dict(domain=domain,error=type(exc).__name__+': '+str(exc),counts=counts,
                                     seconds=time.monotonic()-start,complete=False))
        raise
    finally:
        model.train(prior_mode)


def _jvp(model,x,v):
    import torch
    # D5: derivative-only validation must not consume/retain discarded primals.
    # Pair-returning pathwise_jvp remains unchanged for callers needing y and Jv.
    if hasattr(model,'pathwise_derivative'):return model.pathwise_derivative(x,v)
    if hasattr(model,'pathwise_jvp'):return model.pathwise_jvp(x,v)[1]
    return torch.func.jvp(model,(x,),(v,))[1]


def b2_distances(model, owner_models, admission, out, check):
    """Final opened controls, all supplied clean owners; no guessed null roster."""
    A.require_opening(admission);A.configure_and_check(admission)
    expected=admission.configuration['owners']
    if set(owner_models)!=set(expected):raise ValueError('complete original clean-owner roster required')
    import torch
    from msdev.scoring import distance
    device=next(model.parameters()).device
    x=torch.from_numpy(A.array(admission,'final_detector_x','final',(128,64),'float32')).to(device)
    v=torch.from_numpy(A.array(admission,'final_detector_v','final',(128,2,64),'float32')).to(device)
    result={}
    for owner in expected:
        check();rr=[]
        for k in range(2):rr+=distance(model,owner_models[owner],x,v[:,k],check)['rows']
        from msdev.rules import complete_mean
        result[owner]=dict(rows=rr,mean=complete_mean(rr))
    write(Path(out)/'B2_DISTANCES.json',result);return result
