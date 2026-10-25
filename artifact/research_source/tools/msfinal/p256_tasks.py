"""P256 challenger using frozen task.optimize unchanged.

T128 target family matches the learned/F64 objective. Development and externally
accepted final consumer entries are separate. Neither issues admission. The
two-step development probe is not a64-task result or physical qualification.
"""
from pathlib import Path
from msdev.records import write,progress

def readback(out):
    from msdev.records import read
    out=Path(out);r=read(out/'RESULT.json')
    if r.get('complete') is not True or r.get('counts')!=workload(64,120) or len(r.get('rows',[]))!=64:
        raise ValueError('incomplete final P256 readback')
    for i,row in enumerate(r['rows']):
        if (row.get('target')!=i or row.get('steps')!=120 or row.get('batch_loss_divisor')!=16
                or row.get('objective_target_family')!='T128' or read(out/f'task_{i:02d}.json')!=row
                or not (out/f'design_{i:02d}.npy').is_file()):raise ValueError('P256 task membership differs')
    return r

def workload(count,steps):
    if (count,steps) not in ((2,2),(64,120)):raise ValueError('only prospective probe/full contracts supported')
    return dict(P256_optimization_forward=count*(steps+1),P256_adjoint=count*steps,
                T128_scoring=count,P256_scoring=count,per_target_inverse_updates=count*steps)

def run(starts,t128_targets,p256_targets,out,check,*,target_family,scope):
    if scope not in ('development_probe','development_full'):raise ValueError('no production admission in development entry')
    return _run(starts,t128_targets,p256_targets,out,check,target_family=target_family,scope=scope)


def final_run(admission,acceptance,out,check):
    """Consume independently issued final request; never create acceptance."""
    from operatormark.corpus.state_v2_prerequisites import require_accepted
    from . import materials as A,evaluation as E,marking as M
    E.specification(admission,'final');opening=A.require_opening(admission)
    entries=A.catalog(admission)['entries']
    names=('final_starts','final_targets','final_targets_p256')
    bindings=dict(A.scope(admission),opening_sha256=admission.prerequisites.opening.sha256,
                  inputs={n:entries[n]['sha256'] for n in names},source_sha256=M.digest(__file__))
    r=require_accepted(acceptance,kind='main_study_final_P256_task',bindings=bindings)
    if (r.get('domain')!='final' or r.get('target_family')!='T128'
            or r.get('counts')!=workload(64,120) or r.get('physical_qualification') is not False):
        raise ValueError('final P256 task contract differs')
    if opening.get('P256_task')!={'source_sha256':bindings['source_sha256'],'inputs':bindings['inputs'],'counts':workload(64,120)}:
        raise ValueError('P256 absent from coordinated opening')
    A.preflight(admission);check();A.configure_and_check(admission)
    starts=A.array(admission,names[0],'final',(64,8,8),'float64')
    targets=A.array(admission,names[1],'final',(64,64,64),'float64')
    ptargets=A.array(admission,names[2],'final',(64,64,64),'float64')
    return _run(starts,targets,ptargets,out,check,target_family='T128',scope='accepted_final')


def _run(starts,t128_targets,p256_targets,out,check,*,target_family,scope):
    import numpy as np
    from operatormark_numerics_v2 import task
    from q6_common import true_field
    from .p256_adjoint import Session
    if target_family!='T128':raise ValueError('adopted primary objective requires T128 targets; no extra P256-target arm')
    if scope not in ('development_probe','development_full','accepted_final'):raise ValueError('unknown task scope')
    count,steps=(2,2) if scope=='development_probe' else (64,120)
    if starts.shape!=(count,8,8) or t128_targets.shape!=(count,64,64) or p256_targets.shape!=(count,64,64):raise ValueError('bound task population differs')
    out=Path(out);out.mkdir(exist_ok=False);counts={k:0 for k in workload(count,steps)};rows=[]
    target=t128_targets
    def charge(k):check();counts[k]+=1;progress(out,'native_intent',operation=k,counts=counts)
    try:
        for i in range(count):
            calls=0;active=[]
            def oracle(rho):
                nonlocal calls
                charge('P256_optimization_forward');calls+=1
                s=Session(rho);active.append(s);field=s.field64
                if calls==steps+1:
                    s.close()
                    def unused(w):raise ValueError('unexpected final-field derivative')
                    return field,unused
                if calls>steps+1:raise ValueError('task call population drift')
                def vjp(w):
                    charge('P256_adjoint')
                    try:return s.vjp64(w)
                    finally:s.close()
                return field,vjp
            try:r=task.optimize(oracle,starts[i:i+1],target[i:i+1],steps=steps,batch_size=16)
            finally:
                for s in active:
                    if not s.closed:s.close()
            counts['per_target_inverse_updates']+=steps
            with (out/f'design_{i:02d}.npy').open('xb') as f:np.save(f,r.designs[0],allow_pickle=False)
            charge('T128_scoring');t=true_field(r.designs[0])
            charge('P256_scoring');p=true_field(r.designs[0],True)
            row=dict(target=i,T=task.score(t,t128_targets[i]),P256=task.score(p,p256_targets[i]),trace=r.loss_trace,
                     objective_target_family=target_family,batch_loss_divisor=16,steps=steps)
            rows.append(row);write(out/f'task_{i:02d}.json',row)
        if counts!=workload(count,steps):raise ValueError('P256 task counts')
        result=dict(complete=True,scope=scope,counts=counts,rows=rows,physical_qualification=False,production_acceptance=False)
        write(out/'RESULT.json',result);return result
    except BaseException as exc:
        write(out/'FAILURE.json',dict(error=repr(exc),counts=counts,rows=rows,complete=False));raise
