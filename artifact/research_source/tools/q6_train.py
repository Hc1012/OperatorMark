"""D9 branch-isolated training, restoration and true-F64 checkpoint evaluation."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import time
import numpy as np
import torch
import q5_model as M
from q6_common import check_time, plateau, save, array_save, true_field

class Sampler:
    """All examples retained; a batch spanning an epoch is filled from the next shuffle."""
    def __init__(self,count,seed):
        self.count=count;self.rng=torch.Generator(device='cpu').manual_seed(seed)
        self.order=torch.empty(0,dtype=torch.int64);self.offset=0
    def next(self):
        values=[];left=16
        while left:
            if self.offset==len(self.order):self.order=torch.randperm(self.count,generator=self.rng);self.offset=0
            n=min(left,len(self.order)-self.offset);values.append(self.order[self.offset:self.offset+n])
            self.offset+=n;left-=n
        return torch.cat(values)
    def state(self):return dict(count=self.count,rng=self.rng.get_state().clone(),order=self.order.clone(),offset=self.offset)
    @classmethod
    def restore(cls,state):
        obj=cls(state['count'],0);obj.rng.set_state(state['rng'].cpu());obj.order=state['order'].cpu().clone();obj.offset=state['offset'];return obj

class Engine:
    def __init__(self,data,initialization,shuffle,device,model_factory=M.Q5FNO):
        torch.manual_seed(initialization);self.model=model_factory().to(device).train()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=.002)
        self.data=data;self.device=device;self.factory=model_factory;self.sampler=Sampler(len(data[0]),shuffle);self.updates=0
        self.normalizations=tuple(float(x) for x in M.scales(data[2],data[3]));self.last=None;self.capture_rng()
    def capture_rng(self):
        self.cpu_rng=torch.get_rng_state().clone();self.cuda_rng=torch.cuda.get_rng_state_all() if self.device=='cuda' else []
    def install_rng(self):
        torch.set_rng_state(self.cpu_rng.cpu())
        if self.cuda_rng:torch.cuda.set_rng_state_all([x.cpu() for x in self.cuda_rng])
    def state(self):
        return copy.deepcopy(dict(model=M.tensor_state(self.model),optimizer=self.optimizer.state_dict(),
            sampler=self.sampler.state(),updates=self.updates,normalizations=self.normalizations,
            cpu_rng=self.cpu_rng,cuda_rng=self.cuda_rng,last=self.last))
    @classmethod
    def restore(cls,state,data,device,model_factory=M.Q5FNO):
        # Optimizer.load_state_dict may retain same-device tensor storage.
        # Restoration must not let a branch mutate its saved parent snapshot.
        state=copy.deepcopy(state)
        obj=cls(data,0,0,device,model_factory);obj.model.load_state_dict(state['model'],strict=True)
        obj.optimizer.load_state_dict(state['optimizer']);obj.sampler=Sampler.restore(state['sampler'])
        if tuple(obj.normalizations)!=tuple(state['normalizations']):raise ValueError('restoration normalization mismatch')
        obj.updates=state['updates'];obj.cpu_rng=state['cpu_rng'].cpu();obj.cuda_rng=state['cuda_rng'];obj.last=state['last']
        obj.install_rng();return obj
    def step(self,lr=.002):
        self.install_rng();ids=self.sampler.next().to(self.device)
        for group in self.optimizer.param_groups:group['lr']=lr
        self.optimizer.zero_grad(set_to_none=True)
        x,v,y,dy=(a[ids] for a in self.data)
        loss,lf,lj=M.objective(self.model,x,v,y,dy,*self.normalizations)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):raise FloatingPointError('nonfinite/missing gradient')
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('nonfinite trained parameter')
        self.updates+=1;self.capture_rng()
        self.last=dict(update=self.updates,loss=float(loss.detach()),forward_loss=float(lf.detach()),jvp_loss=float(lj.detach()),lr=lr)
        return self.last

def advance(engine,target,deadline,checkpoint=lambda:None,progress=lambda row:None,cool_start=None,interval=2048):
    if target<engine.updates:raise ValueError('cannot move training backwards')
    while engine.updates<target:
        check_time(deadline)
        lr=.002 if cool_start is None else .002*(target-engine.updates-1)/(target-cool_start)
        engine.step(lr)
        if engine.updates%512==0:progress(engine.last)
        if engine.updates%interval==0:checkpoint()
    checkpoint()

def load_training(bundle,split,device,count=None):
    data=[]
    for name in ('x','v','y','dy'):
        a=np.load(Path(bundle)/f'{split}_{name}.npy',allow_pickle=False)
        if a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('invalid benchmark tensors')
        data.append(torch.from_numpy(a[:count] if count else a).to(device))
    return tuple(data)

def _relative(pred,target):
    den=torch.linalg.vector_norm(target,dim=-1)
    if not torch.isfinite(den).all() or (den<=0).any():raise ValueError('invalid relative-error denominator')
    error=torch.linalg.vector_norm(pred-target,dim=-1)/den
    if not torch.isfinite(error).all():raise ValueError('nonfinite validation error')
    return error.detach().cpu().tolist()

def evaluate(engine,bundle,out,deadline,*,defer_native=None):
    """Evaluation has its own cloned model; training model/optimizer/RNG are untouched."""
    out=Path(out);out.mkdir(exist_ok=False);bundle=Path(bundle);device=engine.device
    before=engine.state();model=engine.factory().to(device);model.load_state_dict(before['model'],strict=True);model.eval()
    started=time.monotonic();forward=[];jvp=[]
    x,v,y,dy=load_training(bundle,'validation',device)
    v=v.reshape(256,2,64);dy=dy.reshape(256,2,4096)
    truth=torch.from_numpy(np.load(bundle/'validation_truth_f64.npy',allow_pickle=False)).to(device)
    tangent_truth=torch.from_numpy(np.load(bundle/'validation_tangent_truth_f64.npy',allow_pickle=False)).to(device).reshape(256,2,4096)
    for i in range(0,256,16):
        check_time(deadline)
        with torch.no_grad():forward.extend(_relative(model(x[i:i+16]).double()*.01,truth[i:i+16]))
        for k in range(2):
            with torch.no_grad():pred=torch.func.jvp(model,(x[i:i+16],),(v[i:i+16,k],))[1]
            jvp.extend(_relative(pred.double()*.01,tangent_truth[i:i+16,k]))
    target=np.stack([np.load(bundle/f'tasks/target_{i:02d}_f64.npy',allow_pickle=False) for i in range(16)])
    target_t=torch.from_numpy((target/.01).reshape(16,4096).astype(np.float32)).to(device)
    starts=np.load(bundle/'starts.npy',allow_pickle=False)
    designs=torch.tensor(starts.reshape(16,64),dtype=torch.float32,device=device,requires_grad=True)
    optimizer=torch.optim.Adam([designs],lr=.03);trace=[]
    for _ in range(120):
        check_time(deadline);optimizer.zero_grad(set_to_none=True)
        loss=(model(designs)-target_t).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite inverse loss')
        # Only the design gradient is needed; do not accumulate model gradients.
        gradient=torch.autograd.grad(loss,designs)[0]
        if not torch.isfinite(gradient).all():raise FloatingPointError('nonfinite inverse gradient')
        designs.grad=gradient;optimizer.step()
        with torch.no_grad():designs.clamp_(0,1)
        trace.append(float(loss.detach()))
    actual=designs.detach().cpu().numpy().astype(np.float64).reshape(16,8,8);array_save(out/'designs.npy',actual)
    if defer_native is not None:
        # Caller persists the same design/trace/targets and binds CPU scoring.
        # No change to preceding F/J/inverse operations or the historical default.
        try:return defer_native(out,bundle,forward,jvp,trace,engine.updates,time.monotonic()-started)
        finally:engine.install_rng()
    from operatormark_numerics_v2.task import score
    tasks=[];robust=[]
    for i,rho in enumerate(actual):
        check_time(deadline)
        f=true_field(rho);p=true_field(rho,p1=True)
        ef=score(f,target[i]);ep=score(p,np.load(bundle/f'tasks/target_{i:02d}_p256.npy',allow_pickle=False))
        tasks.append(ef);robust.append(ep)
        save(out/f'task_{i:02d}.json',dict(target=i,f64_error=ef,p256_error=ep,hard_gates_pass=True))
    values=[sum(a)/len(a) for a in (forward,jvp,tasks)]
    if any(not math.isfinite(a) or a<0 for a in values):raise ValueError('nonnegative finite score components required')
    result=dict(complete=True,model_updates=engine.updates,forward=forward,jvp=jvp,task_f64=tasks,task_p256=robust,
        means=dict(zip(('forward','jvp','task_f64'),values)),score=0. if 0. in values else math.exp(sum(math.log(a) for a in values)/3),
        inverse_steps=120,batch_loss_divisor=16,inverse_precision='float32 pilot model; true solver scoring float64',
        inverse_loss_trace=trace,seconds=time.monotonic()-started,physical_claims='none')
    save(out/'RESULT.json',result)
    engine.install_rng();return result
