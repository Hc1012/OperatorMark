"""D9, matched forward32, embedding and cost-only adaptive kernels."""
from __future__ import annotations
import copy
import math
import torch
import q5_model as M
from q6_train import Sampler as Q6Sampler

class Sampler(Q6Sampler):
    def __init__(self,count,seed,batch=16):super().__init__(count,seed);self.batch=batch
    def next(self):
        chunks=[];left=self.batch
        while left:
            if self.offset==len(self.order):self.order=torch.randperm(self.count,generator=self.rng);self.offset=0
            n=min(left,len(self.order)-self.offset);chunks.append(self.order[self.offset:self.offset+n]);self.offset+=n;left-=n
        return torch.cat(chunks)
    def state(self):return dict(super().state(),batch=self.batch)
    @classmethod
    def restore(cls,s):
        obj=cls(s['count'],0,s['batch']);obj.rng.set_state(s['rng'].cpu());obj.order=s['order'].cpu().clone();obj.offset=s['offset'];return obj

def feasible(x,v):
    x,v=torch.broadcast_tensors(x,v)
    v=torch.where(x<=.02,v.abs(),torch.where(x>=.98,-v.abs(),v))
    norm=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    if not torch.isfinite(norm).all() or (norm<=0).any():raise ValueError('invalid key direction')
    return v/norm

def physical_key_directions(controls,v,device='cpu'):
    """D6 on original float64 controls, then cast directions for the model.

    Never choose near-bound signs from a rounded float32 model input.
    """
    import numpy as np
    from operatormark_numerics_v2.directions import feasible_direction
    controls=np.asarray(controls,dtype=np.float64).reshape(-1,8,8)
    vv=np.asarray(v,dtype=np.float64).reshape(8,8)
    out=np.stack([feasible_direction(rho,vv).flatten() for rho in controls])
    return torch.from_numpy(out.astype(np.float32)).to(device)

def smooth_basis(grid=64,device='cpu',dtype=torch.float32):
    # Pilot's cell centres and yy,xx orientation, extended from 32 to 64.
    # Compute once on Windows and export the public basis; no CPU/CUDA trig drift.
    import numpy as np
    axis=(np.arange(grid,dtype=np.float64)+.5)/grid
    yy,xx=np.meshgrid(axis,axis,indexing='ij');rows=[]
    for k,l in ((1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2)):
        for f in (np.sin,np.cos):
            z=f(2*math.pi*(k*xx+l*yy));rows.append((z-z.mean()).flatten())
    return torch.tensor(np.stack(rows),device=device,dtype=dtype)

def patterns(coeff,basis):
    w=coeff@basis;norm=w.square().mean(-1,keepdim=True).sqrt()
    if not torch.isfinite(norm).all() or (norm<=0).any():raise ValueError('zero/nonfinite smooth key')
    return w/norm

def positive_scale(value,name):
    if not math.isfinite(float(value)) or float(value)<=0:raise ValueError('finite positive '+name+' required')
    return value

def b2_penalty(distance,tau):
    positive_scale(tau,'tau5')
    return .1*(torch.relu(torch.as_tensor(tau,device=distance.device,dtype=distance.dtype)-distance)/tau).square()

def loss_with_scales(model,data,recipe,scales,reference=None,key=None,target=None,tau=None,basis=None):
    x,v,y,dy=data;p=model(x)
    if recipe=='forward32':return (p-y).square().mean()
    fs,js=scales;j=torch.func.jvp(model,(x,),(v,))[1]
    objective=(p-y).square().mean()/fs+.5*(j-dy).square().mean()/js
    if recipe in ('D9','sham','A4'):return objective
    if recipe in ('B1','B3'):
        positive_scale(target,'embedding target')
        if recipe=='B1':
            with torch.no_grad():ref=reference(x)
            residual=p-ref;scale=math.sqrt(fs)
        else:
            direction=key['batch_directions'] if 'batch_directions' in key else feasible(x,key['v'])
            pred=torch.func.jvp(model,(x,),(direction,))[1]
            with torch.no_grad():ref=torch.func.jvp(reference,(x,),(direction,))[1]
            residual=pred-ref;scale=math.sqrt(js)
        s=(residual*key['w']).mean()/positive_scale(scale,'detector scale')
        return objective+.05*((s-target)/target).square()
    if recipe in ('A5B1','A5B3'):
        residual=p-y if recipe=='A5B1' else j-dy
        # Public RMS-one sine/cosine basis; dot is pixel MEAN in model units.
        wb=basis/basis.square().mean(-1,keepdim=True).sqrt()
        projections=residual@wb.T/residual.shape[-1]
        return objective+.1*projections.square().sum(-1).mean()/(fs if recipe=='A5B1' else js)
    if recipe=='A5B2':
        with torch.no_grad():ref=torch.func.jvp(reference,(x,),(v,))[1]
        norm=torch.linalg.vector_norm(ref,dim=-1)
        if (norm<=0).any() or not torch.isfinite(norm).all():raise ValueError('undefined B2 starting-reference distance')
        distance=(torch.linalg.vector_norm(j-ref,dim=-1)/norm).mean()
        return objective+b2_penalty(distance,tau)
    raise ValueError('unknown recipe '+recipe)

class Engine:
    def __init__(self,data,initialization,shuffle,device,recipe='D9',model_factory=M.Q5FNO,
                 weights=None,reference=None,key=None,target=None,tau=None,basis=None):
        torch.manual_seed(initialization);self.model=model_factory().to(device).train();self.factory=model_factory
        if weights is not None:self.model.load_state_dict(weights,strict=True)
        self.recipe=recipe;self.data=data;self.device=device;self.updates=0
        self.sampler=Sampler(len(data[0]),shuffle,32 if recipe=='forward32' else 16)
        self.scales=(1.,1.) if recipe=='forward32' else tuple(float(x) for x in M.scales(data[2],data[3]))
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=.0003 if recipe in ('sham','B1','B3') else .002)
        self.reference=reference;self.key=key;self.target=target;self.tau=tau;self.basis=basis;self.last=None
        self.capture_rng()
    def capture_rng(self):
        self.cpu_rng=torch.get_rng_state().clone();self.cuda_rng=torch.cuda.get_rng_state_all() if self.device=='cuda' else []
    def install_rng(self):
        torch.set_rng_state(self.cpu_rng.cpu())
        if self.cuda_rng:torch.cuda.set_rng_state_all([x.cpu() for x in self.cuda_rng])
    def state(self):return copy.deepcopy(dict(model=M.tensor_state(self.model),optimizer=self.optimizer.state_dict(),
        sampler=self.sampler.state(),updates=self.updates,recipe=self.recipe,scales=self.scales,
        cpu_rng=self.cpu_rng,cuda_rng=self.cuda_rng,last=self.last))
    def restore(self,state):
        s=copy.deepcopy(state)
        if s['recipe']!=self.recipe or tuple(s['scales'])!=self.scales:raise ValueError('trajectory recipe/normalization changed')
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(s['optimizer'])
        self.sampler=Sampler.restore(s['sampler']);self.updates=s['updates'];self.cpu_rng=s['cpu_rng'];self.cuda_rng=s['cuda_rng'];self.last=s['last'];self.install_rng()
        return self
    def step(self,lr=None):
        self.install_rng();ids=self.sampler.next().to(self.device)
        if lr is not None:
            for g in self.optimizer.param_groups:g['lr']=lr
        self.optimizer.zero_grad(set_to_none=True)
        key=self.key
        if key is not None and 'directions' in key:key=dict(key,batch_directions=key['directions'][ids])
        objective=loss_with_scales(self.model,tuple(a[ids] for a in self.data),self.recipe,self.scales,
            self.reference,key,self.target,self.tau,self.basis)
        if not torch.isfinite(objective):raise FloatingPointError('nonfinite loss')
        objective.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):raise FloatingPointError('nonfinite/missing gradient')
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('nonfinite model')
        self.updates+=1;self.capture_rng();self.last=dict(update=self.updates,loss=float(objective.detach()),lr=self.optimizer.param_groups[0]['lr'])
        return self.last

class Jacobi(torch.nn.Module):
    """Differentiable T128 assembly adapter; no calls to frozen solver code.

    For fixed rho this is affine in initial physical u. rho affects all coefficients.
    """
    def __init__(self,model,steps,native=128,observed=64,macro=8):
        super().__init__();self.model=model;self.steps=steps;self.native=native;self.observed=observed;self.macro=macro
    def forward(self,x):
        n=self.native;b=x.shape[0];k=(3+9*x.reshape(b,self.macro,self.macro)).repeat_interleave(n//self.macro,-1).repeat_interleave(n//self.macro,-2)
        h=2*k[:,:,:-1]*k[:,:,1:]/(k[:,:,:-1]+k[:,:,1:]);v=2*k[:,:-1,:]*k[:,1:,:]/(k[:,:-1,:]+k[:,1:,:])
        pad=torch.nn.functional.pad
        left=pad(h,(1,0));right=pad(h,(0,1));up=pad(v,(0,0,1,0));down=pad(v,(0,0,0,1))
        boundary=torch.zeros_like(k);boundary[:,:,0]+=2*k[:,:,0];boundary[:,:,-1]+=2*k[:,:,-1]
        boundary[:,0,:]+=2*k[:,0,:];boundary[:,-1,:]+=2*k[:,-1,:]
        diag=left+right+up+down+boundary
        u=(self.model(x).reshape(b,self.observed,self.observed)*.01).repeat_interleave(n//self.observed,-1).repeat_interleave(n//self.observed,-2)
        for _ in range(self.steps):
            au=diag*u-left*pad(u[:,:,:-1],(1,0))-right*pad(u[:,:,1:],(0,1))-up*pad(u[:,:-1,:],(0,0,1,0))-down*pad(u[:,1:,:],(0,0,0,1))
            u=u+(2/3)*((1/n**2)-au)/diag
        q=n//self.observed
        return u.reshape(b,self.observed,q,self.observed,q).mean((2,4)).flatten(1)/.01
