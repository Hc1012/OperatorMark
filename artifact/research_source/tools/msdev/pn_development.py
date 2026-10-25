"""One prospective P-N variant. Importing this module launches nothing."""
from pathlib import Path
import math
import time
from .records import read,write,sha,verify,progress

OUTPUT='OperatorMark_PN_Development_20261005'
ARMS=tuple((f'PN_B3_{f:g}_D{i+1}',i,f) for f in (1.5,2.,3.) for i in range(3))
ATTACHED=10800
WORK=9000

def global_penalty_backward(model,x,directions,w,reference,scale,target,chunk=16,check=lambda:None):
    """Two-pass chain rule for ONE squared global mean, bounded activation memory.

    First pass obtains s without reverse graphs; second pass accumulates
    [0.1*(s-T)/T^2] * ds/dtheta in chunks. No per-chunk squared penalties.
    The pinned deterministic, buffer-free FNO is unchanged between passes.
    """
    import torch
    if len(x)==0 or chunk<=0 or not math.isfinite(scale) or scale<=0 or not math.isfinite(target) or target<=0:raise ValueError('positive full-set inputs/scale/target required')
    def contribution(lo):
        hi=min(lo+chunk,len(x));j=torch.func.jvp(model,(x[lo:hi],),(directions[lo:hi],))[1]
        return ((j-reference[lo:hi])*w).mean()/scale*((hi-lo)/len(x))
    with torch.no_grad():
        parts=[]
        for lo in range(0,len(x),chunk):check();parts.append(contribution(lo).detach())
        s=sum(parts);penalty=.05*((s-target)/target).square();coefficient=.1*(s-target)/(target*target)
    if not torch.isfinite(s) or not torch.isfinite(penalty):raise FloatingPointError('nonfinite global marking score/loss')
    for j,lo in enumerate(range(0,len(x),chunk)):
        check();part=contribution(lo)
        if not torch.equal(part.detach(),parts[j]):raise ValueError('two-pass marking score changed; no stochastic/buffer substitution')
        (part*coefficient).backward()
    return dict(global_score=float(s),mark_loss=float(penalty),controls=len(x),chunk=chunk)

def pn_step(engine,directions,w,reference_jvp,target,check=lambda:None):
    """Original utility sampler/normalizers/Adam; only mark gradient changes."""
    import torch
    engine.install_rng();ids=engine.sampler.next().to(engine.device)
    for group in engine.optimizer.param_groups:group['lr']=.0003
    engine.optimizer.zero_grad(set_to_none=True)
    x,v,y,dy=(a[ids] for a in engine.data);fs,js=engine.scales
    p=engine.model(x);j=torch.func.jvp(engine.model,(x,),(v,))[1]
    f=(p-y).square().mean()/fs;jj=.5*(j-dy).square().mean()/js;utility=f+jj
    if not torch.isfinite(utility):raise FloatingPointError('utility loss nonfinite')
    utility.backward()
    mark=global_penalty_backward(engine.model,engine.data[0],directions,w,reference_jvp,math.sqrt(js),target,16,check)
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in engine.model.parameters()):raise FloatingPointError('gradient nonfinite/missing')
    check();engine.optimizer.step()
    if any(not torch.isfinite(p).all() for p in engine.model.parameters()):raise FloatingPointError('model nonfinite')
    engine.updates+=1;engine.capture_rng()
    engine.last=dict(update=engine.updates,pre_state=engine.updates-1,lr=.0003,utility_loss=float(utility.detach()),forward_component=float(f.detach()),
        weighted_jvp_component=float(jj.detach()),loss=float(utility.detach()+torch.tensor(mark['mark_loss'],device=utility.device,dtype=utility.dtype)),
        batch_indices=ids.cpu().tolist(),**mark)
    return engine.last

def qualification(values,metrics,owner_metrics,calibration,owner):
    from . import rules
    if len(values)!=4096 or not all(math.isfinite(x) for x in values):raise ValueError('complete unchanged 4096-key detector required')
    nullmax=max(abs(r['score']) for r in calibration['nulls']['B3'] if r['owner']==owner)
    theta=max(calibration['floors']['B3'],nullmax,max(map(abs,values[1:])))+calibration['deltas']['B3']
    gates={k:rules.gate(metrics[k],owner_metrics[k]) for k in ('F','J','K','T')}
    return dict(complete=all(g['status']=='complete' for g in gates.values()),qualified=values[0]>theta and all(g['passed'] is True for g in gates.values()),
        margin=values[0]-theta,theta=theta,true_score=values[0],wrong_max=max(map(abs,values[1:])),gates=gates)

def preflight(bundle,manifest):
    bundle=Path(bundle)
    if sha(bundle/'MANIFEST.json')!=manifest:raise ValueError('upload drift')
    verify(bundle,read(bundle/'MANIFEST.json')['files']);p=read(bundle/'PN.json')
    if p['arms']!=[list(a) for a in ARMS] or p['model_updates']!=2880 or p['updates_per_arm']!=320:raise ValueError('scope drift')
    if p['attached_seconds']!=ATTACHED or p['worker_seconds']!=WORK:raise ValueError('prospective ceiling drift')
    return p

def reserve(bundle,root,manifest,credits=None,rate=None,approved=False):
    p=preflight(bundle,manifest);root=Path(root)
    if root.name!=OUTPUT or not approved:raise ValueError('separate explicit full-evaluation launch approval required')
    if root.exists():
        s=read(root/'STATUS.json')
        if s['bindings']['upload']!=manifest or s['session']['closed'] or s['session']['launch_started'] or time.time()>=s['deadline']:raise ValueError('no retry/reset')
        return s
    if not credits or not rate or rate<=0 or credits<3*rate:raise ValueError('credits must cover full three-hour proposed window')
    now=time.time();s=dict(state='UNSTARTED',updates=0,completed_candidates=0,bindings=dict(upload=manifest,parent=p['parent_status_sha256']),
        deadline=now+ATTACHED,historical_allowances_reused=False,session=dict(start=now,end=now+ATTACHED,reserved=ATTACHED,prebooked=True,closed=False,launch_started=False))
    root.mkdir();write(root/'STATUS.json',s);return s

def recorded_counts(root,result,check=lambda:None):
    """Reconcile committed update/product records after owned child closure."""
    root=Path(root);updates=0;completed=0;per_arm={}
    for name,owner,factor in ARMS:
        files=sorted((root/name).glob('update_*.json'));per_arm[name]=len(files)
        if len(files)>320:raise ValueError('too many update records')
        for i,path in enumerate(files,1):
            check();row=read(path)
            if path.name!=f'update_{i:03d}.json' or row['update']!=i or row['pre_state']!=i-1:raise ValueError('update record gap/index mismatch')
        updates+=len(files);product=root/name/'PRODUCT.json'
        if product.exists():
            check();p=read(product)
            if p['id']!=name or p['updates']!=320 or len(files)!=320:raise ValueError('incomplete candidate product')
            completed+=1
    matches=bool(result is not None and result['updates']==updates and result['completed_candidates']==completed)
    return dict(updates=updates,completed_candidates=completed,per_arm=per_arm,worker_counts_match=matches,
        unreceipted_update_possible=not bool(result and result.get('complete')),
        meaning='committed records; interruption can leave an unreceipted in-flight operation, never refundable or retryable')

def launch(python,bundle,root,manifest):
    from .cli import module
    from .colab_environment import owned
    from q6_common import THREAD_ENV
    import os
    root=Path(root);bundle=Path(bundle);preflight(bundle,manifest);s=read(root/'STATUS.json');now=time.time()
    if s['bindings']['upload']!=manifest or s['session']['closed'] or s['session']['launch_started']:raise ValueError('unused bound session required')
    if now+WORK+150>s['deadline']:raise TimeoutError('complete work/cleanup/finalization envelope does not fit')
    s['session']['launch_started']=True;write(root/'STATUS.json',s,replace=True)
    write(root/'CONSUMED.json',dict(start=now,work_deadline=now+WORK,closure_deadline=now+WORK+30,session_deadline=s['deadline'],manifest_sha256=manifest,
        maximum_model_updates=2880,maximum_inverse_input_updates=1080,maximum_native_forward_calls=288,retries=0))
    owner=module(bundle/'code/q5_colab_supervisor.py','pn_owned');receipt=None;outcome=None
    try:
        receipt=owned([str(python),'-B','-m','msdev.pn_development','--bundle',str(bundle),'--root',str(root)],WORK,now+WORK+30,root/'supervisor',owner,
            cwd=bundle/'code',acceptable=(0,4),env=dict(os.environ,**THREAD_ENV,CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled',WANDB_DISABLED='true'))
    finally:
        if receipt is None and (root/'supervisor/RESULT.json').exists():receipt=read(root/'supervisor/RESULT.json')
        if receipt is not None:
            result=read(root/'RESULT.json') if (root/'RESULT.json').exists() else None
            closed=bool(receipt.get('closure',{}).get('resolved') and receipt.get('stdout_closed'))
            final_deadline=min(s['deadline'],time.time()+120)
            def check():
                if time.time()>=final_deadline:raise TimeoutError('finalization deadline')
            counts=recorded_counts(root,result,check) if closed else None
            if counts is not None:write(root/'COUNTS_READBACK.json',counts)
            outcome=dict(complete=bool(closed and counts['worker_counts_match'] and counts['updates']==2880 and counts['completed_candidates']==9 and not receipt.get('timed_out') and not receipt.get('error') and receipt.get('returncode')==0 and result and result.get('complete')),
                closed=closed,timed_out=receipt.get('timed_out'),worker=result,supervisor=receipt,counts=counts,follow_on_authorized=False)
            write(root/'OUTCOME.json',outcome)
            s=read(root/'STATUS.json');s['session']['closed']=closed
            if result is not None:
                s['updates']=result['updates'];s['completed_candidates']=result['completed_candidates'];s['scientific_pass']=result.get('scientific_pass');s['scientific_stop']=result.get('scientific_stop')
            if counts is not None:s.update(updates=counts['updates'],completed_candidates=counts['completed_candidates'],unreceipted_update_possible=counts['unreceipted_update_possible'])
            s['state']=('COMPLETE' if result.get('scientific_pass') else 'STOPPED_SCIENTIFIC') if outcome['complete'] else 'STOPPED'
            write(root/'STATUS.json',s,replace=True)
            if closed:
                hashes={p.relative_to(root).as_posix():sha(p,check) for p in root.rglob('*') if p.is_file()}
                check();write(root/'OUTPUT_MANIFEST.json',dict(files=hashes,complete=outcome['complete'],kind='one-PN-variant'))
    return outcome

def run(bundle,root):
    import numpy as np
    import torch
    import q5_model as M
    from q6_common import true_field
    from . import kernels as K,scoring,rules
    from .cli import runtime,reusable_q6_runtime
    bundle=Path(bundle);root=Path(root);grant=read(root/'CONSUMED.json');count=0;products=[];engine=None;name=None;native_calls=0;phase_end=grant['work_deadline'];started=time.time()
    def check():
        if time.time()>=min(grant['work_deadline'],phase_end):raise TimeoutError('P-N phase/worker limit; no retry')
    def status():
        s=read(root/'STATUS.json');s.update(updates=count,completed_candidates=len(products),active=name,native_forward_calls=native_calls);write(root/'STATUS.json',s,replace=True)
    try:
        p=preflight(bundle,grant['manifest_sha256']);M.configure();env=runtime();reusable_q6_runtime(bundle,env)
        if env!=p['runtime'] or not torch.cuda.is_available():raise ValueError('exact historical A100 runtime required')
        write(root/'RUNTIME.json',dict(environment=env,deterministic=torch.are_deterministic_algorithms_enabled(),matmul=torch.get_float32_matmul_precision(),
            tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,autocast=torch.is_autocast_enabled('cuda')))
        if not torch.are_deterministic_algorithms_enabled() or torch.get_default_dtype()!=torch.float32 or torch.get_float32_matmul_precision()!='highest' or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.is_autocast_enabled('cuda'):raise ValueError('runtime precision/flags differ')
        seeds=read(bundle/'PN_SEEDS.json');cal=read(bundle/'calibration.json');baseline=read(bundle/'BASELINE.json')
        load=lambda n:np.load(bundle/n,allow_pickle=False)
        xval=torch.from_numpy(load('validation_x.npy')).to('cuda');basis=torch.from_numpy(load('public_basis.npy')).to('cuda')
        controls=torch.from_numpy(load('validation_controls_f64.npy').reshape(-1,64));grids={f:[] for f in (1.5,2.,3.)}
        for name,owner,factor in ARMS:
            phase_end=min(grant['work_deadline'],time.time()+600);check();progress(root,'candidate_start',candidate=name,updates=count)
            weights=torch.load(bundle/f'D{owner+1}.pt',map_location='cpu',weights_only=True);reference=M.Q5FNO().to('cuda');reference.load_state_dict(weights,strict=True);reference.eval()
            if reference.operator.fno_block_precision!='full' or {p.dtype for p in reference.parameters()}!={torch.float32,torch.complex64}:raise ValueError('FNO precision drift')
            with np.load(bundle/f'embedding_{owner}.npz',allow_pickle=False) as z:data=tuple(torch.from_numpy(z[n].copy()).to('cuda') for n in ('x','v','y','dy'));rho=z['rho'].copy()
            with np.load(bundle/f'keys_B3_{owner}.npz',allow_pickle=False) as z:coeff=torch.from_numpy(z['coeff'].copy());v=torch.from_numpy(z['v'].copy());w=torch.from_numpy(z['w_true'].copy()).to('cuda')
            directions=K.physical_key_directions(rho,v[0].numpy(),'cuda');cached=[]
            with torch.no_grad():
                for lo in range(0,128,16):check();cached.append(torch.func.jvp(reference,(data[0][lo:lo+16],),(directions[lo:lo+16],))[1].detach())
            reference_jvp=torch.cat(cached);target=factor*cal['floors']['B3'];seed=seeds['owners'][owner]
            engine=K.Engine(data,seed['initialization'],seed['embedding_shuffle'],'cuda',recipe='B3',weights=weights,reference=reference)
            if len(data[0])!=128 or tuple(engine.scales)!=tuple(p['normalizers'][str(owner)]):raise ValueError('embedding population/normalizer drift')
            if any(isinstance(m,(torch.nn.modules.dropout._DropoutNd,torch.nn.modules.batchnorm._BatchNorm)) for m in engine.model.modules()) or list(engine.model.buffers()):raise ValueError('two-pass deterministic buffer-free premise unavailable')
            arm=root/name;arm.mkdir();trace=[];train_start=time.time()
            for step in range(320):
                check();last=pn_step(engine,directions,w,reference_jvp,target,check);count+=1;trace.append(last)
                write(arm/f'update_{step+1:03d}.json',last)
                if (step+1)%8==0:status();progress(root,'training',candidate=name,updates=count,arm_update=step+1,score=last['global_score'],target=target,loss=last['loss'])
            torch.save(M.tensor_state(engine.model),arm/'model.pt');torch.save(engine.state(),arm/'state.pt');train_seconds=time.time()-train_start;status();check()
            phase_end=min(grant['work_deadline'],time.time()+180);evaluation_start=time.time();progress(root,'evaluation',candidate=name)
            def oracle(rho,p1):
                nonlocal native_calls
                check()
                if native_calls>=288:raise ValueError('native evaluation call cap')
                native_calls+=1;status();return true_field(rho,p1)
            evaluation=scoring.evaluate(engine.model,bundle,v[0],load(f'key_truth_{owner}.npy'),arm/'evaluation',check,oracle)
            check()
            if not evaluation['complete']:raise ValueError('incomplete unchanged evaluation')
            evaluation_seconds=time.time()-evaluation_start
            phase_end=min(grant['work_deadline'],time.time()+180);score_start=time.time();progress(root,'detector',candidate=name)
            projected=scoring.projections(engine.model,xval,basis,check=check);torch.save(projected,arm/'projection.pt')
            old_projection=torch.load(bundle/f'baseline/D{owner+1}_projection.pt',map_location='cpu',weights_only=True)
            values=scoring.projected_scores(projected,old_projection,controls,coeff,v,basis,math.sqrt(engine.scales[1]),'B3')
            check()
            row=qualification(values,evaluation['metrics'],baseline['owners'][str(owner)]['metrics'],cal,owner);grids[factor].append(row)
            write(arm/'DETECTOR.json',dict(scores=values,true_index=0,wrong_count=4095,qualification=row))
            product=dict(id=name,owner=owner,factor=factor,updates=320,model_sha256=sha(arm/'model.pt'),state_sha256=sha(arm/'state.pt'),
                metrics=evaluation['metrics'],qualification=row,train_seconds=train_seconds,evaluation_seconds=evaluation_seconds,detector_seconds=time.time()-score_start,
                peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved())
            write(arm/'PRODUCT.json',product);products.append(product);status();progress(root,'candidate_complete',candidate=name,updates=count,qualified=row['qualified'])
            del engine,reference,reference_jvp;engine=None
        phase_end=grant['work_deadline'];check();grid=[dict(factor=f,owners=grids[f]) for f in (1.5,2.,3.)];write(root/'QUALIFICATION_GRID.json',grid)
        selected=None;failure=None
        try:selected=rules.select_mark(grid,'B3')
        except ValueError as exc:
            if str(exc)!='B3 has fewer than two qualifying development owners':raise
            failure=str(exc)
        write(root/'RESULT.json',dict(complete=True,scientific_pass=failure is None,selection=selected,scientific_stop=failure,updates=count,completed_candidates=len(products),
            native_forward_calls=native_calls,inverse_input_updates=1080,products=products,seconds=time.time()-started,original_0_of_18_unchanged=True,follow_on_authorized=False))
        status();progress(root,'completed',updates=count,scientific_pass=failure is None,reason=failure);return 0
    except BaseException as exc:
        persistence=None
        if engine is not None:
            try:torch.save(engine.state(),root/f'{name}.partial.pt')
            except BaseException as ee:persistence=repr(ee)
        write(root/'RESULT.json',dict(complete=False,error=repr(exc),active=name,updates=count,completed_candidates=len(products),products=products,native_forward_calls=native_calls,evidence_error=persistence,follow_on_authorized=False))
        status();return 4

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();raise SystemExit(run(a.bundle,a.root))
