"""Prepared full-population reconstruction/readback matrix; no launcher/grant.

Caller supplies the existing supervisor's closed kernel receipt and bound output
manifest, independent per-stage reservations/checks and strict runtime observer.
Every context/result here remains development-only, not final acceptance.
"""
from pathlib import Path
import io
from msdev.records import read,write,sha,verify,progress
from . import development_evaluator as D,models as M,verification_plan as P,readback as R
from .variants import TRANSFORMS

def run(catalog_root,catalog_sha256,screen_root,screen_manifest_sha256,closure_path,closure_sha256,
        out,check,reserve,*,runtime_check):
    from .session import closure_ok
    root=Path(screen_root);out=Path(out);check()
    if sha(root/'OUTPUT_MANIFEST.json')!=screen_manifest_sha256 or sha(closure_path)!=closure_sha256:raise ValueError('kernel output/closure binding')
    closure=read(closure_path)
    if not closure_ok(closure) or closure.get('returncode')!=0 or closure.get('timed_out'):raise ValueError('closed successful kernel worker required')
    manifest=read(root/'OUTPUT_MANIFEST.json');verify(root,manifest['files'])
    if manifest.get('complete') is not True or manifest['files'].get('supervisor/RESULT.json')!=closure_sha256:raise ValueError('closure is not bound to this completed kernel export')
    c,spec=D.context(catalog_root,catalog_sha256,'final')
    if runtime_check()!=c.record['runtime_sha256']:raise ValueError('historical exact worker runtime mismatch')
    import torch
    from msdev import kernels as K,scoring as S
    work=root/'work';result=read(work/'RESULT.json')
    from .development_check import COUNTS
    if not result.get('complete_available_screen') or result.get('counts')!=COUNTS:raise ValueError('complete prescribed kernel counts required')
    from .update_journal import read_prefix
    observed=read_prefix(work/'updates.jsonl')
    if not observed['complete'] or observed['completed_updates']!=36:raise ValueError('kernel journal readback differs')
    out.mkdir(exist_ok=False);completed=[];weights=D.Access.weights(c,'clean_model','marking')
    def load(name,arch='fno32'):
        if name=='D1':state=weights
        else:
            p=work/(name+'.state.pt');anchor=read(work/(name+'.anchor.json'))
            data=p.read_bytes()
            import hashlib
            if hashlib.sha256(data).hexdigest()!=anchor['state_sha256']:raise ValueError('saved model state drift')
            state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)['model']
        model=M.create(arch).to('cuda').eval();model.load_state_dict(state,strict=True)
        return model
    reference=load('D1');endpoint=load('core_ordinary')
    from .evaluation import ProjectionStore
    store=ProjectionStore(out/'projections')
    try:
        reserve('qualification');check()
        D.run(reference,catalog_root,catalog_sha256,out/'qualification',check,phase='qualification',
            model_sha256=c.prerequisites.clean_model_sha256,runtime_check=runtime_check,projection_store=store)
        R.evaluation(out/'qualification',c.record['specifications']['qualification'],check,expected_domain='development')
        for name in P.plan()['full_evaluation_models']:
            reserve('evaluation/'+name);check()
            if name in TRANSFORMS:model=M.transformed(reference,name,endpoint=endpoint)
            elif name.startswith('jacobi'):model=K.Jacobi(reference,int(name[6:]))
            else:model=load(name,'fno32' if not name.startswith('student_') or name=='student_jvp' else name[8:])
            from .graph import identity
            binding=identity(dict(screen_manifest=screen_manifest_sha256,catalog=catalog_sha256,procedure=name,
                models_sha256=sha(M.__file__),source_sha256=sha(__file__)))
            if name=='core_ordinary':
                from q5_model import tensor_state
                left,right=tensor_state(model),tensor_state(reference)
                if left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left):raise ValueError('full-sized evaluation requires a nonidentity saved endpoint')
            D.run(model,catalog_root,catalog_sha256,out/name,check,phase='final',model_sha256=binding,runtime_check=runtime_check,projection_store=store)
            R.evaluation(out/name,spec,check,expected_domain='development')
            reserve('B2/'+name);check()
            x=torch.from_numpy(D.Access.array(c,'final_detector_x','final',(128,64),'float32')[:16]).to('cuda')
            v=torch.from_numpy(D.Access.array(c,'final_detector_v','final',(128,2,64),'float32')[:16]).to('cuda')
            rows=[]
            for k in range(2):rows+=S.distance(model,reference,x,v[:,k],check)['rows']
            write(out/(name+'_B2.json'),dict(rows=rows,development_single_pair=True,complete_null_cohort=False))
            completed.append(name);progress(out,'full_evaluator_complete',model=name,count=len(completed))
            del model
        reserve('reconstruction');check()
        x=torch.from_numpy(D.Access.array(c,'final_x','final',(512,64),'float32')[:32]).to('cuda')
        v=torch.from_numpy(D.Access.array(c,'final_v','final',(512,2,64),'float32')[:32,0]).to('cuda')
        from .variants import ARCHITECTURES
        names=['student_'+a for a in ARCHITECTURES]+['student_jvp']+list(TRANSFORMS)+['jacobi16','jacobi256']
        for name in names:
            check()
            if name in TRANSFORMS:model=M.transformed(reference,name,endpoint=endpoint)
            elif name.startswith('jacobi'):model=K.Jacobi(reference,int(name[6:]))
            else:model=load(name,'fno32' if name=='student_jvp' else name[8:])
            batch=32 if name.startswith('student_') and name!='student_jvp' else 16
            xx=x[:batch];vv=v[:batch];y,j=torch.func.jvp(model,(xx,),(vv,))
            z=xx.detach().clone().requires_grad_(True);w=torch.ones_like(y)/y.numel()
            reverse=torch.autograd.grad((model(z)*w).sum(),z)[0]
            row=dict(name=name,batch=batch,lhs=float((j*w).sum()),rhs=float((vv*reverse).sum()),finite=all(bool(torch.isfinite(a).all()) for a in (y,j,reverse)))
            write(out/(name+'_AD.json'),row)
            if not row['finite'] or not torch.allclose((j*w).sum(),(vv*reverse).sum(),rtol=.002,atol=.00002):raise ValueError('reconstructed F/JVP/reverse path: '+name)
            if name.startswith('round') and (torch.count_nonzero(j) or torch.count_nonzero(reverse)):raise ValueError('rounding surrogate')
        reserve('jacobi_inverse')
        starts=D.Access.array(c,'final_starts','final',(64,8,8),'float64')[:16]
        targets=D.Access.array(c,'final_targets','final',(64,64,64),'float64')[:16]
        for depth,steps in ((16,2),(256,120)):
            model=K.Jacobi(reference,depth);design=torch.tensor(starts.reshape(16,64),device='cuda',dtype=torch.float32,requires_grad=True)
            target=torch.tensor(targets.reshape(16,4096)/.01,device='cuda',dtype=torch.float32);opt=torch.optim.Adam([design],lr=.03)
            traces=[]
            for i in range(steps):
                check();opt.zero_grad(set_to_none=True);loss=(model(design)-target).square().mean();grad=torch.autograd.grad(loss,design)[0]
                row=dict(step=i,loss=float(loss.detach()),finite=bool(torch.isfinite(grad).all() and torch.isfinite(loss)))
                write(out/f'jacobi{depth}_inverse_{i:03d}.json',row)
                if not row['finite']:raise FloatingPointError('Jacobi sustained batch16 inverse')
                design.grad=grad;opt.step()
                with torch.no_grad():design.clamp_(0,1)
                traces.append(row)
            from .materials import save_tensor_record
            save_tensor_record(out/f'jacobi{depth}_designs.pt',dict(designs=design.detach().cpu()))
            write(out/f'jacobi{depth}_INVERSE.json',dict(updates=steps,batch=16,divisor=16,native_calls=0,rows=traces))
        write(out/'RESULT.json',dict(complete=True,completed=completed,qualification=True,development_only=True,
            plan=P.plan(),screen_manifest_sha256=screen_manifest_sha256,closure_sha256=closure_sha256,
            scientific_adoption=False,production_acceptance=False))
    except BaseException as exc:
        write(out/'FAILURE.json',dict(complete=False,error=repr(exc),completed=completed));raise
