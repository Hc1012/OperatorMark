"""Explicitly disposable toy inputs; only accessible inside revision2 verification."""
import copy
import os
import time
import numpy as np
import torch
from core import write
from dev_campaign import Development
from dev_state import counts
from numerics import generate,controls,directions,forward,derivatives,Observations,task_data,fresh
from training import Trajectory
from dev_process import publish_handshake


class TinyDevelopment(Development):
    fixture=True;grid=(10,20);owner_steps=2;mark_steps=2;attack_steps=2;chunk=1
    rate_limits={'R-a':100.,'R-b':100.}

    def __init__(self,*args):
        if os.environ.get('TPFA_CATEGORY')!='development_r2_verification':raise PermissionError('toy fixture only')
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        super().__init__(*args)

    def solve(self,row):return np.array([1+row.sum(),2+np.dot(row,row)],dtype=np.float64)
    def model(self,seed):return fresh(seed,lambda:torch.nn.Linear(64,2))

    def legacy_payload(self):
        cases=[]
        for name in ('pilot0','pilot1'):
            model=self.model(self.seeds.get(name+'/initialization'))
            cases.append(dict(id=name,models={'O':model,'M':copy.deepcopy(model)},costs={}))
        return cases,self.splits.seen

    def data(self,label,n,k,seeds=None):
        def build():
            value=generate(self.oracle,self.seeds.get(label+'/controls'),self.seeds.get(label+'/directions'),4,k,label,self.splits)
            return value,self.splits.seen
        value,seen=self.journal.block('data/'+label,counts(),build);self.splits.seen=seen;return value

    def tasks(self):
        def build():return task_data(self.oracle,self.seeds,1,'development_tasks',self.splits),self.splits.seen
        value,seen=self.journal.block('data/tasks',counts(),build);self.splits.seen=seen;return value

    def attack_data(self,case):
        name=case['id']
        def build():
            shared=forward(self.oracle,controls(self.seeds.get(name+'/attack_controls'),4),name,self.splits)
            subset=Observations(shared.x64[:3],shared.y64[:3],information_calls=3)
            rb=derivatives(self.oracle,subset,directions(self.seeds.get(name+'/Rb_directions'),3,1),name+'/Rb',self.splits)
            return (shared,rb),self.splits.seen
        value,seen=self.journal.block('data/'+name+'/attack',counts(),build);self.splits.seen=seen;return value

    def trajectory(self,model,data,shuffle,recipe='forward',reference=None,key=None):
        # Toy marking is explicitly forward-only; no scientific marking claim.
        return Trajectory(model,data,shuffle,self.budget,recipe='forward' if recipe=='mark' else recipe,
                          batch=3 if recipe=='forward' else 2,real=True)

    def key(self):return {'fixture':self.seeds.get('D1/private_key',(0,10000))}
    def gate(self,model,data):return dict(forward=.1,random_jvp=.2,fixture=True)

    def copied(self,case,arm,data):
        tr=self.trajectory(copy.deepcopy(case['models']['M' if arm=='A3' else 'O']),data,
                           self.seeds.get(case['id']+'/shared_attack_shuffle'))
        tr.extend(self.attack_steps);return tr.model,tr.record()

    def measure_model(self,case,arm,model,data,tasks):
        disposable=copy.deepcopy(model).eval()
        x,v,y,dy=data.tensors()
        with torch.no_grad():f=float((disposable(x)-y).square().mean())
        j=torch.func.jvp(disposable,(x,),(v[:,0],))[1]
        je=float((j-dy[:,0]).square().mean().detach())
        for p in disposable.parameters():p.requires_grad_(False)
        start=torch.nn.Parameter(torch.tensor(tasks['starts'],dtype=torch.float32))
        optimizer=torch.optim.Adam([start],lr=.03)
        loss=(disposable(start)-torch.tensor(tasks['targets'],dtype=torch.float32)).square().mean()
        loss.backward();self.budget.charge('inverse_updates');optimizer.step()
        # The parent verification test accounts these bounded children conservatively
        # as complete fixtures before launching; no second live counter writer here.
        return dict(forward=f,random_jvp=je,task=float((disposable(start)-torch.tensor(tasks['targets'],dtype=torch.float32)).square().mean().detach()))

    def measure(self,case,arm,model,data,tasks,label):
        return self.journal.block('evaluate/'+label+'/'+case['id']+'/'+arm,counts(inverse_updates=1),
                                  lambda:self.measure_model(case,arm,model,data,tasks))

    def fixture_pause(self,key):
        if self.attempt==0 and self.config.get('pause_key')==key:
            publish_handshake(self.path/'FIXTURE_PAUSED.json',dict(key=key,pid=os.getpid(),known_complete=self.journal.last_complete))
            while True:time.sleep(.05)
