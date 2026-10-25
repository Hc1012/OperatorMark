"""New scoped final-noise evaluator with explicit saved-design CPU dispatch.

Derived from the frozen msfinal.evaluation._evaluate_model numerical body.
Validation, projected-Adam optimization, detector calculations and draw order are
unchanged. Admission, new-record provenance and the deferred export boundary are
explicit here; this module never changes or monkeypatches the frozen sources.
"""
import hashlib
import json
import time
from pathlib import Path

SOURCE_EVALUATION_SHA256 = '02c710b57d74735d34ebb712fc7a220d136f07441da4432ad23d409d34dad912'


def _helpers():
    from msdev.records import write
    from msfinal.evaluation import (validate_specification, required_arrays,
        evaluation_binding, _detector_scores, _jvp, reuse_diagnostics)
    return locals()

def evaluate_noise(model, admission, out, check, *, phase, model_sha256, wrapper_steps, projection_store,access,spec,export,provenance):
    if (phase != 'final' or wrapper_steps is not None
            or provenance.get('scope') != 'selected_noise20_only'
            or admission.bindings.get('scope') != 'selected_noise20_only'
            or admission.bindings.get('domain') != 'final' or not callable(export)):
        raise ValueError('Scoped final-noise deferred evaluation required')
    components="full"; diagnostic_root=None
    helpers=_helpers()
    write=helpers['write']; validate_specification=helpers['validate_specification']
    required_arrays=helpers['required_arrays']; evaluation_binding=helpers['evaluation_binding']
    _detector_scores=helpers['_detector_scores']; _jvp=helpers['_jvp']; reuse_diagnostics=helpers['reuse_diagnostics']
    from msfinal.marking import identity
    identity(model_sha256)
    if components not in ("full","diagnostics","tasks"):raise ValueError("unknown evaluation component")
    if components=="tasks" and diagnostic_root is None:raise ValueError("bound diagnostics required before tasks")
    validate_specification(spec,phase);check()
    import numpy as np
    import torch
    import q5_model as Q
    from msfinal import cpu_scoring as CPU
    # New scoped dispatch explicitly defers all native scoring.
    deferred=True
    from msdev import scoring as S, kernels as K
    out=Path(out);out.mkdir(exist_ok=False)
    start=time.monotonic();device=next(model.parameters()).device
    prior_mode=model.training;model.eval()
    counts=dict(inverse_input_iterations=0,targets=0,native_t128_calls=0,native_p256_calls=0,
                validation_controls=0,detector_controls=0,optimizer_model_updates=0)
    rows={k:[] for k in ('F','J','K','T','P256')}
    # Prospective final-population record under its own narrow runtime gate.
    domain='final'
    write(out/'START.json',dict(domain=domain,scope='selected_noise20_only',provenance=provenance,phase=phase,specification=spec,bindings=admission.bindings,
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
                # Native scoring is exported below; no native calls in GPU stage.
        if components=='full':
            scores=_detector_scores(model,admission,phase,check,model_sha256=model_sha256,wrapper_steps=wrapper_steps,
                                   store=projection_store,access=access,spec=spec)
            counts['detector_controls']=spec['detector_controls']
            write(out/'DETECTOR_SCORES.json',dict(scores=scores,wrong_key_count=nkeys-1))
        metrics={k:S.complete_mean(rr) for k,rr in rows.items() if k!='P256' or spec['p256']}
        result=dict(complete=all(v is not None for v in metrics.values()),metrics=metrics,scores=scores,
            counts=counts,rows=rows,seconds=time.monotonic()-start,domain=domain,phase=phase,specification=spec,
            physical_qualification=False,independent_acceptance=False,
            scope='selected_noise20_only',provenance=provenance)
        if components=='diagnostics':
            result.update(complete=False,diagnostics_complete=all(metrics[k] is not None for k in 'FJK') and all(__import__('math').isfinite(x) for x in scores),
                task_status='pending',components='diagnostics')
        elif components=='tasks':result.update(diagnostics_complete=True,task_status='measured',components='combined_D2')
        if deferred:
            result.update(complete=False,task_status=CPU.STATE,diagnostics_complete=True)
            result=export(out,result,check)
        write(out/'RESULT.json',result);return result
    except BaseException as exc:
        write(out/'FAILURE.json',dict(domain=domain,error=type(exc).__name__+': '+str(exc),counts=counts,
                                     seconds=time.monotonic()-start,complete=False))
        raise
    finally:
        model.train(prior_mode)
