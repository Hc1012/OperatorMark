"""One-shot fixed-model embedding-control scores; no training or selection path."""
from pathlib import Path
import math
import statistics
import time
from .records import read,write,sha,verify,progress

def summarize(values,target):
    if len(values)!=128 or not all(math.isfinite(v) for v in values):raise ValueError('128 finite control scores required')
    if target is not None and (not math.isfinite(target) or target<=0):raise ValueError('positive target required')
    def stats(v):return dict(mean=statistics.mean(v),population_sd=statistics.pstdev(v),minimum=min(v),maximum=max(v))
    s=stats(values)
    return dict(full_set=s,score_target_ratio=None if target is None else s['mean']/target,
        per_control_scores=values,batch_definition='Eight contiguous groups of 16 in saved embedding-array order; NOT historical optimizer batches',
        batches=[dict(first_control=i,last_control=i+15,count=16,**stats(values[i:i+16])) for i in range(0,128,16)])

def score(model,owner,x,directions,w,normalizer,method,target,check=lambda:None,on_batch=lambda i:None):
    import torch
    if method not in ('B1','B3') or len(x)!=128:raise ValueError('fixed B1/B3, 128 controls required')
    if not math.isfinite(normalizer) or normalizer<=0:raise ValueError('saved positive normalizer required')
    model.eval();owner.eval();values=[]
    # no_grad preserves forward-mode AD. inference_mode would suppress JVPs.
    with torch.no_grad():
        for i in range(0,128,16):
            check();xx=x[i:i+16]
            if method=='B1':a=model(xx);b=owner(xx)
            else:
                a=torch.func.jvp(model,(xx,),(directions[i:i+16],))[1]
                b=torch.func.jvp(owner,(xx,),(directions[i:i+16],))[1]
            v=((a-b)*w).mean(-1)/math.sqrt(normalizer)
            if not torch.isfinite(v).all():raise ValueError('nonfinite diagnostic score')
            values.extend(v.detach().cpu().double().tolist());on_batch(i//16+1);check()
    return summarize(values,target)

def launch(python,bundle,root,manifest):
    """Independent result; same charged window. Never retry a consumed diagnostic."""
    from .cli import module
    from .colab_environment import owned
    from q6_common import THREAD_ENV
    import os
    root=Path(root);bundle=Path(bundle);s=read(root/'STATUS.json');session=s['session']
    if not session.get('prebooked') or session.get('closed') or session.get('launch_started'):raise ValueError('unused attached continuation window required')
    if sha(bundle/'MANIFEST.json')!=manifest or s['bindings']!=dict(read(bundle/'CONFIG.json')['bindings'],upload=manifest):raise ValueError('diagnostic binding drift')
    end=time.time()+600
    if end>min(session['end']-60,s['deadline']-60):raise TimeoutError('full diagnostic envelope unavailable in original window')
    folder=root/'marking_diagnostic';folder.mkdir(exist_ok=False)
    write(folder/'CONSUMED.json',dict(start=time.time(),deadline=end,work_deadline=end-30,manifest_sha256=manifest,
        original_deadline=s['deadline'],session_end=session['end'],updates=0,scope='one inference-only diagnostic; independent of unmarked completion'))
    supervisor=module(bundle/'code/q5_colab_supervisor.py','msdev_diagnostic_owned')
    command=[str(python),'-B','-m','msdev.marking_diagnostic','--bundle',str(bundle),'--root',str(root)]
    try:
        receipt=owned(command,570,end,folder/'supervisor',supervisor,cwd=bundle/'code',acceptable=(0,4),
            env=dict(os.environ,**THREAD_ENV,CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled',WANDB_DISABLED='true'))
    except RuntimeError:
        p=folder/'supervisor/RESULT.json'
        if not p.exists():raise
        receipt=read(p)
        if not receipt.get('closure',{}).get('resolved') or not receipt.get('stdout_closed'):raise
    result=read(folder/'RESULT.json') if (folder/'RESULT.json').exists() else None
    report=dict(complete=bool(receipt.get('returncode')==0 and not receipt.get('timed_out') and result and result.get('complete')),
        timed_out=receipt.get('timed_out'),worker_result=result,supervisor_receipt='supervisor/RESULT.json',
        closure=receipt.get('closure'),independent_of_unmarked_completion=True)
    write(folder/'OUTCOME.json',report)
    progress(root,'marking_diagnostic_closed',complete=report['complete'],timed_out=report['timed_out'])
    return report

def run(bundle,root):
    import os
    from q6_common import THREAD_ENV
    os.environ.update(THREAD_ENV,CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled',WANDB_DISABLED='true')
    import numpy as np
    import torch
    import q5_model as M
    from . import kernels as K
    from .cli import runtime,reusable_q6_runtime
    from .unmarked import validate_parent
    bundle=Path(bundle);root=Path(root);folder=root/'marking_diagnostic';grant=read(folder/'CONSUMED.json');rows=[];start=time.time()
    def check():
        if time.time()>=grant['work_deadline']:raise TimeoutError('diagnostic work deadline; cleanup reserved')
    try:
        check()
        if sha(bundle/'MANIFEST.json')!=grant['manifest_sha256']:raise ValueError('diagnostic manifest drift')
        verify(bundle,read(bundle/'MANIFEST.json')['files']);validate_parent(bundle)
        plan=read(bundle/'DIAGNOSTIC_INPUTS.json')
        expected={(f'{method}_{factor:g}_D{i+1}',method,i) for method in ('B1','B3') for factor in (1.5,2,3) for i in range(3)}
        expected.update((f'S{i+1}',method,i) for method in ('B1','B3') for i in range(3))
        if len(plan['rows'])!=24 or {(r['id'],r['method'],r['owner_index']) for r in plan['rows']}!=expected:raise ValueError('exact 21-model / 24-row diagnostic roster required')
        M.configure();environment=runtime();reusable_q6_runtime(bundle,environment)
        if environment!=read(bundle/'PARENT.json')['runtime']:raise ValueError('parent runtime mismatch')
        if not torch.cuda.is_available():raise ValueError('CUDA required; no fallback')
        def load(name):
            check();model=M.Q5FNO().to('cuda');model.load_state_dict(torch.load(bundle/name,map_location='cpu',weights_only=True),strict=True);model.eval()
            if any(p.dtype not in (torch.float32,torch.complex64) for p in model.parameters()):raise ValueError('actual parameter precision differs')
            return model
        owners={i:load(f'parent/models/D{i+1}.pt') for i in range(3)}
        flags=dict(default_dtype=str(torch.get_default_dtype()),matmul_precision=torch.get_float32_matmul_precision(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
            autocast_cuda=torch.is_autocast_enabled('cuda'),parameter_dtypes=sorted({str(p.dtype) for m in owners.values() for p in m.parameters()}),
            fno_block_precision=sorted({m.operator.fno_block_precision for m in owners.values()}))
        write(folder/'RUNTIME.json',dict(environment=environment,actual=flags))
        if flags['matmul_tf32'] or flags['cudnn_tf32'] or flags['autocast_cuda'] or flags['default_dtype']!='torch.float32' or flags['matmul_precision']!='highest' or flags['fno_block_precision']!=['full']:raise ValueError('actual precision differs; no automatic correction')
        for row in plan['rows']:
            check();i=row['owner_index'];method=row['method']
            with np.load(bundle/f'diagnostic/embedding_{i}.npz',allow_pickle=False) as z:
                x=torch.from_numpy(z['x'].copy()).to('cuda');rho=z['rho'].copy()
            with np.load(bundle/f'diagnostic/true_key_{method}_{i}.npz',allow_pickle=False) as z:
                w=torch.from_numpy(z['w_true'].copy()).to('cuda');v=z['v_true'].copy()
            directions=K.physical_key_directions(rho,v,'cuda') if method=='B3' else None
            candidate=load('diagnostic/models/'+row['id']+'.pt')
            scored=score(candidate,owners[i],x,directions,w,row['normalizer'],method,row['target'],check,
                lambda batch:progress(root,'marking_diagnostic_batch',model=row['id'],method=method,batch=batch,total_batches=8))
            result=dict(id=row['id'],method=method,owner_index=i,target=row['target'],normalizer=row['normalizer'],
                validation=row['validation'],model_sha256=row['model_sha256'],**scored)
            write(folder/(row['id']+'_'+method+'.json'),result);rows.append(result);del candidate
        check();torch.cuda.synchronize()
        write(folder/'RESULT.json',dict(complete=True,models=21,scoring_rows=len(rows),controls_per_row=128,batch_size=16,
            rows=rows,seconds=time.time()-start,scope='fixed saved models; NOT historical training-time batch scores',
            optimizer_updates=0,operation_set='model forward and forward-mode JVP, tensor reductions, saved-file reads/writes only; no optimizer, solver or wrong-key scoring',
            evidence_limit='Saved validation grids lack independent producer attestations; these new embedding scores do not seal those grids.'))
        return 0
    except BaseException as exc:
        write(folder/'RESULT.json',dict(complete=False,error=repr(exc),completed_rows=len(rows),seconds=time.time()-start,
            partial_rows='individually committed row JSON files',independent_of_unmarked_completion=True))
        return 4

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();raise SystemExit(run(a.bundle,a.root))
