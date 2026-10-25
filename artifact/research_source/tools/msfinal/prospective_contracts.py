"""Selected development contracts and deterministic service primitives.

Final material/runtime acceptance and numerical verification remain required.
"""
import math,threading
from collections import OrderedDict

OVERWRITE=dict(reference='received checkpoint frozen before attacker marking',access='weights+public method; no defender key/reference',
    controls=128,updates=320,target_factor=3,penalty=.05,utility_batch=16,lr=.0003,
    calibration='12 independently bound development null scores under received-reference procedure; positive floor and delta=.15*floor',
    selection='one preregistered key; no best-key or target search',labels='received-model self-distillation F/JVP',adopted=True)
NOISE=dict(sd=[.0001,.001],units='u/0.01',repetitions=32,aggregation='mean outputs before errors/loss',
    randomness='explicit independent development stream per source/phase/control/step/repetition',
    derivative='white-box pathwise; additive noise has zero input derivative',
    inverse='new32 draws each120 updates; hold draws fixed within each loss/gradient pair',adopted=True)
API=dict(policies=['paired_deny','cache','rate'],distance='control L-infinity',radius=.001,cache_capacity=4096,
    rate=16.,burst=32.,attempt_limit=32768,ordinary_acquisition_limit=256,
    derivative='unavailable; no adopted service derivative/task procedure',reset='declared phase boundaries only, one stream per model/owner',adopted=True)
ACCUSER=dict(score_queries=4096,selection='largest SIGNED development score; first-index tie; no sign flip',
    final_evaluations=1,reference='independently trained bound Rstar, frozen before search',
    calibration='independent attacker-reference procedure before untouched final readout; no defender floor',adopted=True)

def accuser_selection(scores):
    if len(scores)!=4096 or any(type(v) not in (int,float) or not math.isfinite(v) for v in scores):raise ValueError('complete fixed4096 score population required')
    return max(range(4096),key=lambda i:(scores[i],-i))

class Service:
    """Candidate single-client service state; proposed reset/denial rules explicit.

    Persist caller-supplied intent before mutation/compute; callbacks own durable
    failure evidence. No hidden derivative, sleeps, seeds or network service.
    """
    def __init__(self,policy):
        if policy not in API['policies']:raise ValueError('unknown fixed service policy')
        self.policy=policy;self.attempts=0;self.last=None;self.time=None;self.tokens=32.;self.cache=OrderedDict();self.lock=threading.Lock()
    def call(self,x,now,compute,record):
        x=tuple(x)
        if len(x)!=64 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in x):raise ValueError('invalid service controls')
        if type(now) not in (int,float) or not math.isfinite(now):raise ValueError('explicit monotonic timestamp required')
        with self.lock:
            if self.time is not None and now<self.time:raise ValueError('service time rollback')
            if self.attempts>=32768:raise ValueError('fixed total attempt ceiling')
            record(dict(kind='intent',attempt=self.attempts+1,time=now,policy=self.policy))
            self.attempts+=1;old=self.last;self.last=x
            self.tokens=min(32.,self.tokens+(0 if self.time is None else (now-self.time)*16));self.time=now
            status='computed';value=None
            if self.policy=='paired_deny' and old is not None and max(abs(a-b) for a,b in zip(x,old))<=.001:status='denied_near_previous'
            elif self.policy=='rate' and self.tokens<1:status='denied_rate'
            else:
                if self.policy=='rate':self.tokens-=1
                if self.policy=='cache':
                    candidates=[(max(abs(a-b) for a,b in zip(x,k)),i,k) for i,k in enumerate(self.cache)]
                    near=[r for r in candidates if r[0]<=.001]
                    if near:
                        _,_,key=min(near);value=self.cache[key];self.cache.move_to_end(key);status='cached'
                if value is None:
                    value=tuple(compute(x))
                    if len(value)!=4096 or any(not math.isfinite(v) for v in value):raise ValueError('invalid service output')
                    if self.policy=='cache':
                        self.cache[x]=value
                        if len(self.cache)>4096:self.cache.popitem(last=False)
            record(dict(kind='result',attempt=self.attempts,status=status,time=now))
            return dict(status=status,value=value,derivative_available=False)

def noise_mean(model,x,draw,record):
    """Same explicit draws for a forward/gradient pair; caller fixes draw identity.

    draw(i,shape,device,dtype) supplies already amplitude-scaled independent noise.
    All32 raw outputs retained via record before aggregation; no RNG is created.
    """
    import torch
    rows=[]
    # Deterministic underlying model: reuse its exact graph for all32 outputs.
    # Reverse/JVP differentiates the prescribed mean with the same fixed draws.
    y=model(x)
    for i in range(32):
        eps=draw(i,y.shape,y.device,y.dtype)
        if eps.requires_grad or eps.shape!=y.shape or not torch.isfinite(eps).all():raise ValueError('explicit fixed independent noise required')
        z=y+eps
        if not torch.isfinite(z).all():raise ValueError('nonfinite noisy output')
        record(i,z.detach());rows.append(z)
    return torch.stack(rows).mean(0)
