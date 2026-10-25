"""Isolated D3 reproduction/trace entry point. Importing this file runs nothing."""
from pathlib import Path
from contextlib import contextmanager
import copy
import math
import statistics
import time
from .records import read,write,sha,verify,progress

ARMS=('B3_3_D3','B3_1.5_D3','S3')
OUTPUT='OperatorMark_D3_Replay_20261005'

def tensor_bits_equal(a,b):
    import torch
    if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor):return False
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))

def reproduction(actual,expected,last,expected_last,step=320):
    mismatches=sorted(set(actual)^set(expected))
    mismatches += [n for n in actual.keys() & expected.keys() if not tensor_bits_equal(actual[n],expected[n])]
    loss_ok=last['update']==step and expected_last['update']==step and last['loss']==expected_last['loss']
    return dict(passed=not mismatches and loss_ok,model_tensor_mismatches=sorted(set(mismatches)),last_objective_equal=loss_ok,
        compared_final_state=step,last_objective_pre_state=step-1,actual_last_objective=last['loss'],historical_last_objective=expected_last['loss'],
        meaning='REPRODUCTION consistency only; not proof of freedom from bugs. File serialization bytes are not compared.')

@contextmanager
def capture_loss(K,callback):
    """Same FP32 operation order as frozen loss; observe existing scalars only."""
    import torch
    original=K.loss_with_scales
    def observed(model,data,recipe,scales,reference=None,key=None,target=None,tau=None,basis=None):
        if recipe not in ('B3','sham'):raise ValueError('isolated replay loss scope')
        x,v,y,dy=data;p=model(x);fs,js=scales;j=torch.func.jvp(model,(x,),(v,))[1]
        f=(p-y).square().mean()/fs
        jj=.5*(j-dy).square().mean()/js
        objective=f+jj;s=None;mark=None
        if recipe=='B3':
            direction=key['batch_directions'] if 'batch_directions' in key else K.feasible(x,key['v'])
            pred=torch.func.jvp(model,(x,),(direction,))[1]
            with torch.no_grad():ref=torch.func.jvp(reference,(x,),(direction,))[1]
            residual=pred-ref;scale=math.sqrt(js)
            s=(residual*key['w']).mean()/K.positive_scale(scale,'detector scale')
            K.positive_scale(target,'embedding target')
            mark=.05*((s-target)/target).square();total=objective+mark
        else:total=objective
        vals=dict(forward_component=float(f.detach()),weighted_jvp_component=float(jj.detach()),
            mark_component=0. if mark is None else float(mark.detach()),batch_score=None if s is None else float(s.detach()),total=float(total.detach()))
        if not all(math.isfinite(v) for v in vals.values() if v is not None):raise FloatingPointError('nonfinite observed loss')
        # Float32 recombination, same left-to-right order as the actual objective.
        recombined=f.detach()+jj.detach()
        if mark is not None:recombined=recombined+mark.detach()
        if not tensor_bits_equal(recombined,total.detach()):raise ValueError('observed components do not reconstruct actual objective')
        callback(vals);return total
    K.loss_with_scales=observed
    try:yield
    finally:K.loss_with_scales=original

@contextmanager
def observation_model(model):
    """Clone buffers/modes/parameters and preserve RNG; never touch live gradients."""
    import torch
    devices=list(range(torch.cuda.device_count())) if next(model.parameters()).is_cuda else []
    with torch.random.fork_rng(devices=devices):
        clone=copy.deepcopy(model);clone.eval()
        try:yield clone
        finally:del clone

def observe(model,x,directions,w,reference_jvp,scale,validation=None,passive_ids=None,check=lambda:None):
    import torch
    import numpy as np
    values=[];passive=None;forward=None
    with observation_model(model) as clone,torch.no_grad():
        for i in range(0,len(x),16):
            check();j=torch.func.jvp(clone,(x[i:i+16],),(directions[i:i+16],))[1]
            values.extend((((j-reference_jvp[i:i+16])*w).mean(-1)/scale).detach().cpu().double().tolist())
        if passive_ids is not None:
            j=torch.func.jvp(clone,(x[passive_ids],),(directions[passive_ids],))[1]
            passive=float(((j-reference_jvp[passive_ids])*w).mean()/scale)
        if validation is not None:
            vx,truth=validation;errors=[]
            for i in range(0,len(vx),16):
                check();pred=clone(vx[i:i+16]).double().cpu().numpy()*.01;t=truth[i:i+16].cpu().numpy()
                den=np.linalg.norm(t,axis=-1)
                if (den<=0).any() or not np.isfinite(den).all():raise ValueError('unavailable validation forward denominator')
                errors.extend((np.linalg.norm(pred-t,axis=-1)/den).tolist())
            forward=statistics.mean(errors)
    result=dict(full_score=statistics.mean(values) if values else None,validation_F=forward,passive_batch_score=passive)
    if not all(math.isfinite(v) for v in result.values() if v is not None):raise FloatingPointError('nonfinite observation')
    return result

def classify(trace,sham,target):
    """Descriptive map: states 0..320; batch[t] precedes update t+1."""
    unknown=lambda why:dict(classification='unclassified',reason=why)
    if not math.isfinite(target) or target<=0:return unknown('target unavailable/nonpositive')
    if len(trace)!=321 or len(sham)!=321 or [r['state'] for r in trace]!=list(range(321)) or [r['state'] for r in sham]!=list(range(321)):return unknown('incomplete states')
    if any(not isinstance(r.get('full_score'),(int,float)) or not math.isfinite(r['full_score']) for r in trace):return unknown('full-set score unavailable')
    c=next((t for t,r in enumerate(trace) if r['full_score']>=target/2),None)
    if c is None:return unknown('no T/2 crossing')
    if c==320:return unknown('empty late phase')
    try:
        f=[trace[c]['validation_F'],sham[c]['validation_F'],trace[320]['validation_F'],sham[320]['validation_F']]
        if any(x is None or not math.isfinite(x) or x<=0 for x in f):return unknown('forward error unavailable/nonpositive')
        end=math.log(f[2]/f[3])
        if end<=0:return unknown('no positive final excess log-forward error')
        approach=math.log(f[0]/f[1])/end
        late=trace[c:320]
        if any(r.get('batch_score') is None or not math.isfinite(r['batch_score']) for r in late):return unknown('late batch score unavailable')
        noise=sum((r['batch_score']-r['full_score'])**2 for r in late)
        bias=sum((r['full_score']-target)**2 for r in late)
        if noise+bias<=0:return unknown('zero noise-share denominator')
        ns=noise/(noise+bias)
        signs=[1 if r['full_score']>target else -1 for r in trace[c:] if r['full_score']!=target]
        crossings=sum(a!=b for a,b in zip(signs,signs[1:]))
        step=statistics.median(abs(trace[t+1]['full_score']-trace[t]['full_score'])/target for t in range(c,320))
        label='approach' if approach>=.5 else 'noise' if ns>=.5 else 'overshoot' if crossings>=3 or step>=.25 else 'unclassified'
        return dict(classification=label,first_crossing_state=c,a_A=approach,NS=ns,crossings=crossings,median_step_over_T=step,
            interpretation='descriptive trace classification only, not causal proof')
    except (KeyError,TypeError,ValueError,ZeroDivisionError):return unknown('required quantity unavailable')

def joint(rows):
    labels=[r['classification'] for r in rows]
    action='P-N may be considered under separate authority' if labels==['noise','noise'] else 'P-S may be considered under separate authority' if labels==['overshoot','overshoot'] else 'early-drift design review; does not rule out all optimization changes' if labels==['approach','approach'] else 'mixed/unclassified: stop and report'
    return dict(classifications=labels,descriptive_action=action,follow_on_authorized=False,qualification=False)

def preflight(bundle,manifest):
    bundle=Path(bundle)
    if sha(bundle/'MANIFEST.json')!=manifest:raise ValueError('upload binding mismatch')
    verify(bundle,read(bundle/'MANIFEST.json')['files']);p=read(bundle/'REPLAY.json')
    if p['arms']!=list(ARMS) or p['updates_per_arm']!=320 or p['total_updates']!=960:raise ValueError('scope differs')
    return p

def reserve(bundle,root,manifest,credits=None,rate=None,approved=False):
    p=preflight(bundle,manifest);root=Path(root)
    if root.name!=OUTPUT or not approved:raise ValueError('fixed output and separate explicit launch approval required')
    if root.exists():
        s=read(root/'STATUS.json')
        if s['bindings']['upload']!=manifest or s['session']['closed'] or s['session']['launch_started'] or time.time()>=s['deadline']:raise ValueError('no reset/retry of existing replay')
        return s
    if not credits or not rate or rate<=0 or credits<rate:raise ValueError('current credits must cover the full one-hour window')
    start=time.time();s=dict(state='UNSTARTED',bindings=dict(upload=manifest,parent=p['parent_status_sha256']),deadline=start+3600,
        session=dict(prebooked=True,closed=False,launch_started=False,start=start,end=start+3600,reserved=3600),
        historical_allowances_reused=False,updates=0)
    root.mkdir();write(root/'STATUS.json',s);return s

def launch(python,bundle,root,manifest):
    from .cli import module
    from .colab_environment import owned
    from q6_common import THREAD_ENV
    import os
    bundle=Path(bundle);root=Path(root);preflight(bundle,manifest);s=read(root/'STATUS.json')
    if s['bindings']['upload']!=manifest or s['session']['launch_started'] or s['session']['closed']:raise ValueError('one unused replay session required')
    now=time.time()
    if now+1260>s['deadline']:raise TimeoutError('full 1200 work +30 cleanup +30 finalization envelope cannot fit')
    s['session']['launch_started']=True;write(root/'STATUS.json',s,replace=True)
    write(root/'CONSUMED.json',dict(start=now,work_deadline=now+1200,closure_deadline=now+1230,session_deadline=s['deadline'],manifest_sha256=manifest,maximum_updates=960))
    owner=module(bundle/'code/q5_colab_supervisor.py','d3_replay_owned')
    command=[str(python),'-B','-m','msdev.d3_replay','--bundle',str(bundle),'--root',str(root)]
    receipt=None
    try:
        receipt=owned(command,1200,now+1230,root/'supervisor',owner,cwd=bundle/'code',acceptable=(0,4),
            env=dict(os.environ,**THREAD_ENV,CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled',WANDB_DISABLED='true'))
    finally:
        if receipt is None and (root/'supervisor/RESULT.json').exists():receipt=read(root/'supervisor/RESULT.json')
        if receipt is not None:
            result=read(root/'RESULT.json') if (root/'RESULT.json').exists() else None
            closed=bool(receipt.get('closure',{}).get('resolved') and receipt.get('stdout_closed'))
            outcome=dict(complete=bool(closed and not receipt.get('timed_out') and not receipt.get('error') and receipt.get('returncode')==0 and result and result.get('complete')),
                closed=closed,timed_out=receipt.get('timed_out'),worker=result,supervisor=receipt,follow_on_authorized=False)
            write(root/'OUTCOME.json',outcome)
            s=read(root/'STATUS.json');s['session']['closed']=closed;s['state']='COMPLETE' if outcome['complete'] else 'STOPPED';write(root/'STATUS.json',s,replace=True)
            if closed:
                def check():
                    if time.time()>=s['deadline']:raise TimeoutError('final record deadline')
                hashes={p.relative_to(root).as_posix():sha(p,check) for p in root.rglob('*') if p.is_file()}
                check();write(root/'OUTPUT_MANIFEST.json',dict(kind='D3-reproduction-return',files=hashes,complete=outcome['complete']))
    return outcome

def run(bundle,root):
    import numpy as np
    import torch
    import q5_model as M
    from . import kernels as K
    from .cli import runtime,reusable_q6_runtime
    bundle=Path(bundle);root=Path(root);grant=read(root/'CONSUMED.json');traces={};gates={};engine=None;name=None;count=0
    def check():
        if time.time()>=grant['work_deadline']:raise TimeoutError('replay numerical deadline')
    try:
        p=preflight(bundle,grant['manifest_sha256']);check();M.configure();env=runtime();reusable_q6_runtime(bundle,env)
        if env!=p['runtime']:raise ValueError('historical runtime mismatch; no approximation')
        if not torch.cuda.is_available():raise ValueError('A100 CUDA required')
        load=lambda path:torch.load(bundle/path,map_location='cpu',weights_only=True)
        weights=load('D3.pt');reference=M.Q5FNO().to('cuda');reference.load_state_dict(weights,strict=True);reference.eval()
        flags=dict(default_dtype=str(torch.get_default_dtype()),matmul=torch.get_float32_matmul_precision(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,autocast=torch.is_autocast_enabled('cuda'),deterministic=torch.are_deterministic_algorithms_enabled(),
            parameter_dtypes=sorted({str(x.dtype) for x in reference.parameters()}),fno_precision=reference.operator.fno_block_precision)
        write(root/'RUNTIME.json',dict(environment=env,actual=flags))
        if flags!=dict(default_dtype='torch.float32',matmul='highest',matmul_tf32=False,cudnn_tf32=False,autocast=False,deterministic=True,parameter_dtypes=['torch.complex64','torch.float32'],fno_precision='full'):raise ValueError('precision/configuration mismatch')
        with np.load(bundle/'embedding_2.npz',allow_pickle=False) as z:
            data=tuple(torch.from_numpy(z[n].copy()).to('cuda') for n in ('x','v','y','dy'));rho=z['rho'].copy()
        with np.load(bundle/'true_key.npz',allow_pickle=False) as z:v=z['v_true'].copy();w=torch.from_numpy(z['w_true'].copy()).to('cuda')
        directions=K.physical_key_directions(rho,v,'cuda');x=data[0];validation=(torch.from_numpy(np.load(bundle/'validation_x.npy',allow_pickle=False)).to('cuda'),torch.from_numpy(np.load(bundle/'validation_truth_f64.npy',allow_pickle=False).reshape(-1,4096)).to('cuda'))
        ref=[]
        with torch.no_grad():
            for i in range(0,len(x),16):check();ref.append(torch.func.jvp(reference,(x[i:i+16],),(directions[i:i+16],))[1].detach())
        reference_jvp=torch.cat(ref);seed=read(bundle/'D3_SEEDS.json');crossings=set();start=time.time()
        for name in ARMS:
            check();target=p['targets'].get(name);recipe='sham' if name=='S3' else 'B3'
            key=dict(w=w,v=torch.from_numpy(v).to('cuda'),directions=directions) if recipe=='B3' else None
            engine=K.Engine(data,seed['initialization'],seed['embedding_shuffle'],'cuda',recipe=recipe,weights=weights,
                reference=reference if recipe=='B3' else None,key=key,target=target)
            expected_state=load('historical/states/'+name+'.pt');expected_model=load('historical/models/'+name+'.pt')
            if expected_state['updates']!=320 or expected_state['recipe']!=recipe or tuple(expected_state['scales'])!=engine.scales:raise ValueError('historical recipe/normalizer binding mismatch')
            if any(not tensor_bits_equal(expected_state['model'][n],expected_model[n]) for n in expected_model):raise ValueError('historical endpoint model/state differ')
            rows=[];first=None;armdir=root/name;armdir.mkdir();progress(root,'arm_start',arm=name)
            for state in range(321):
                check();want_F=state%8==0 or (name=='S3' and state in crossings)
                # Independent sampler copy previews exactly the impending real batch.
                ids=K.Sampler.restore(engine.sampler.state()).next().to('cuda') if state<320 else None
                obs=observe(engine.model,x,directions,w,reference_jvp,math.sqrt(engine.scales[1]),validation if want_F else None,
                    ids if name=='S3' and state<320 else None,check)
                if target is not None and first is None and obs['full_score']>=target/2:
                    first=state;crossings.add(state)
                    if not want_F:obs['validation_F']=observe(engine.model,x[:0],directions[:0],w,reference_jvp[:0],math.sqrt(engine.scales[1]),validation,None,check)['validation_F']
                row=dict(state=state,full_score=obs['full_score'],validation_F=obs['validation_F'],batch_score=None,
                    target=target,first_crossing_state=first,score_mode='eval clone of exact saved training state')
                if state<320:
                    captured=[]
                    with capture_loss(K,captured.append):engine.step(.0003)
                    count+=1;loss=captured[0]
                    if loss['total']!=engine.last['loss']:raise ValueError('actual training objective capture differs')
                    row.update(batch_score=loss['batch_score'] if recipe=='B3' else obs['passive_batch_score'],loss_components=loss,
                        batch_indices=ids.detach().cpu().tolist(),applied_update=state+1,
                        batch_score_kind='actual loss forward' if recipe=='B3' else 'passive pre-state clone; not a sham loss component')
                rows.append(row);write(armdir/f'state_{state:03d}.json',row)
                if state%8==0:progress(root,'trace',arm=name,state=state,updates_total=count,first_crossing=first,full_score=obs['full_score'],validation_F=row['validation_F'])
            gate=reproduction(M.tensor_state(engine.model),expected_model,engine.last,expected_state['last']);write(armdir/'REPRODUCTION.json',gate);gates[name]=gate
            torch.save(engine.state(),armdir/'endpoint.pt');traces[name]=rows
            if not gate['passed']:raise ValueError('unresolved REPRODUCTION mismatch; remaining arms stopped')
            write(armdir/'REUSED_ENDPOINT_EVIDENCE.json',p['endpoints'][name]);progress(root,'arm_reproduced',arm=name,updates_total=count)
        classes=[classify(traces[n],traces['S3'],p['targets'][n]) for n in ARMS[:2]]
        write(root/'RESULT.json',dict(complete=True,updates=count,arms=list(ARMS),reproduction=gates,classifications=classes,decision=joint(classes),seconds=time.time()-start,
            original_qualification_unchanged=True,physical_qualification=False,follow_on_authorized=False));return 0
    except BaseException as exc:
        persist_error=None
        if engine is not None:
            try:torch.save(engine.state(),root/f'{name}.partial.pt')
            except BaseException as ee:persist_error=repr(ee)
        write(root/'RESULT.json',dict(complete=False,error=repr(exc),active_arm=name,updates=count,reproduction=gates,evidence_error=persist_error,
            meaning='unresolved reproduction/operational finding; not a proved implementation defect',follow_on_authorized=False));return 4

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    raise SystemExit(run(a.bundle,a.root))
