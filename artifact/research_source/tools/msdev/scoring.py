"""Symmetric streamed keyed scoring; immutable owner work can be reused."""
from __future__ import annotations
import time
import numpy as np
import torch
from .kernels import feasible,patterns,positive_scale,physical_key_directions
from .rules import relative_rows,complete_mean,score

def projections(model,x,basis,need_jacobian=True,batch=4,check=lambda:None):
    forwards=[];jac=[];random=[]
    for start in range(0,len(x),batch):
        check();c=x[start:start+batch]
        with torch.no_grad():
            p=model(c);forwards.append((p.double()@basis.double().T/p.shape[-1]).cpu())
            if need_jacobian:
                cols=[]
                for k in range(c.shape[-1]):
                    check();v=torch.zeros_like(c);v[:,k]=1
                    j=torch.func.jvp(model,(c,),(v,))[1]
                    cols.append((j.double()@basis.double().T/j.shape[-1]).cpu())
                jac.append(torch.stack(cols,-1))
    return dict(forward=torch.cat(forwards),jacobian=torch.cat(jac) if jac else None)

def projected_scores(candidate,reference,x,coeff,v,basis,scale,method,key_batch=256):
    positive_scale(scale,'detector scale')
    coeff=coeff.double().cpu();basis=basis.double().cpu();x=x.double().cpu();v=v.double().cpu()
    gram=basis@basis.T/basis.shape[-1]
    norm=torch.einsum('kb,bd,kd->k',coeff,gram,coeff).sqrt()
    if not torch.isfinite(norm).all() or (norm<=0).any():raise ValueError('key pattern normalization')
    rows=[]
    for lo in range(0,len(coeff),key_batch):
        a=coeff[lo:lo+key_batch]/norm[lo:lo+key_batch,None]
        if method=='B1':
            delta=(candidate['forward']-reference['forward']).double().cpu()
            s=(delta@a.T).mean(0)/scale
        elif method=='B3':
            delta=(candidate['jacobian']-reference['jacobian']).double().cpu()
            directions=feasible(x[:,None,:],v[None,lo:lo+key_batch,:])
            s=torch.einsum('cbd,kb,ckd->k',delta,a,directions)/(len(x)*scale)
        else:raise ValueError(method)
        if not torch.isfinite(s).all():raise ValueError('nonfinite detector score')
        rows.extend(s.tolist())
    return rows

def direct_scores(model,owner,x,coeff,v,basis,scale,method):
    rows=[]
    with torch.no_grad():
        for w,d in zip(patterns(coeff,basis),v):
            if method=='B1':delta=model(x)-owner(x)
            else:
                phi=feasible(x,d)
                delta=torch.func.jvp(model,(x,),(phi,))[1]-torch.func.jvp(owner,(x,),(phi,))[1]
            rows.append(float((delta*w).mean()/scale))
    return rows

def distance(model,owner,x,v,check=lambda:None):
    rows=[]
    with torch.no_grad():
        for i in range(0,len(x),16):
            check();a=torch.func.jvp(model,(x[i:i+16],),(v[i:i+16],))[1]
            b=torch.func.jvp(owner,(x[i:i+16],),(v[i:i+16],))[1]
            rows.extend(relative_rows(a.double().cpu().numpy(),b.double().cpu().numpy()))
    return dict(rows=rows,mean=complete_mean(rows))

def evaluate(model,bundle,key_direction,key_truth,out,check,oracle):
    """Q6 tasks and normalization, plus K and explicit undefined relative records.

    oracle(rho,p1) is injected in synthetic verification. Real use calls unchanged
    q6_common.true_field. 16 starts, 120 updates, mean over exactly 16*4096.
    """
    from pathlib import Path
    from .records import write
    out=Path(out);out.mkdir(exist_ok=False);bundle=Path(bundle);device=next(model.parameters()).device
    started=time.monotonic();model.eval();rows={'F':[],'J':[],'K':[],'T':[],'P256':[]}
    def load(name):return np.load(bundle/name,allow_pickle=False)
    x=torch.from_numpy(load('validation_x.npy')).to(device)
    v=torch.from_numpy(load('validation_v.npy')).to(device).reshape(len(x),2,-1)
    truth=load('validation_truth_f64.npy').reshape(len(x),-1)
    tangent=load('validation_tangent_truth_f64.npy').reshape(len(x),2,-1)
    kt=np.asarray(key_truth).reshape(len(x),-1)
    keyed=physical_key_directions(load('validation_controls_f64.npy'),key_direction.detach().cpu().numpy(),device)
    for i in range(0,len(x),16):
        check();c=x[i:i+16]
        with torch.no_grad():
            rows['F']+=relative_rows(model(c).double().cpu().numpy()*.01,truth[i:i+16])
            for k in range(2):
                j=torch.func.jvp(model,(c,),(v[i:i+16,k],))[1]
                rows['J']+=relative_rows(j.double().cpu().numpy()*.01,tangent[i:i+16,k])
            kv=keyed[i:i+16]
            kj=torch.func.jvp(model,(c,),(kv,))[1]
            rows['K']+=relative_rows(kj.double().cpu().numpy()*.01,kt[i:i+16])
    target=np.stack([load(f'tasks/target_{i:02d}_f64.npy') for i in range(16)])
    designs=torch.tensor(load('starts.npy').reshape(16,64),dtype=torch.float32,device=device,requires_grad=True)
    tt=torch.tensor(target.reshape(16,-1)/.01,dtype=torch.float32,device=device)
    opt=torch.optim.Adam([designs],lr=.03);trace=[]
    for step in range(120):
        check();opt.zero_grad(set_to_none=True);loss=(model(designs)-tt).square().mean()
        gradient=torch.autograd.grad(loss,designs)[0]
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():raise FloatingPointError('inverse loss/gradient')
        designs.grad=gradient;opt.step()
        with torch.no_grad():designs.clamp_(0,1)
        trace.append(float(loss.detach()))
    rho=designs.detach().cpu().numpy().astype(np.float64).reshape(16,8,8);np.save(out/'designs.npy',rho,allow_pickle=False)
    for i,c in enumerate(rho):
        check();f=oracle(c,False);p=oracle(c,True)
        rows['T']+=relative_rows(np.asarray(f).reshape(1,-1),target[i].reshape(1,-1))
        rows['P256']+=relative_rows(np.asarray(p).reshape(1,-1),load(f'tasks/target_{i:02d}_p256.npy').reshape(1,-1))
        write(out/f'task_{i:02d}.json',dict(F64=rows['T'][-1],P256=rows['P256'][-1],native_hard_gates=True))
    metrics={k:complete_mean(v) for k,v in rows.items()}
    result=dict(complete=all(v is not None for v in metrics.values()),metrics=metrics,rows=rows,score=score(metrics),
        inverse_steps=120,batch_loss_divisor=16,inverse_trace=trace,seconds=time.monotonic()-started,
        source_rule='actual L2 truth norm; no epsilon; zero unavailable, no population exclusion',physical_qualification=False)
    write(out/'RESULT.json',result);return result
