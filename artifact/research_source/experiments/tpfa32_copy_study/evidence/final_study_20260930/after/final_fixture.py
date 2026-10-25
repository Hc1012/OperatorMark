"""Explicit ten-owner synthetic orchestration; no TPFA, SpatialFNO or inverse updates."""
import copy
import os
import time
import numpy as np
import torch
from final_campaign import FinalCampaign
from dev_state import counts
from dev_process import publish_handshake
from numerics import fresh
from training import Trajectory


class TinyFinal(FinalCampaign):
    fixture=True;owner_steps=2;mark_steps=2;attack_steps=2;chunk=1
    base_steps=4;cool_steps=2;equal_steps=2
    owner_n=4;mark_n=4;gate_n=4;eval_n=4;tasks_n=2;attack_n=4
    inverse_bound=0;optimized_bound=0;rate_limit=100.

    def __init__(self,*args):
        if os.environ.get('TPFA_CATEGORY')!='final_study_verification':raise PermissionError('fixture only')
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        super().__init__(*args)
        if not self.config.get('fixture_only'):raise PermissionError('fixture cannot run scientific configuration')

    def solve(self,row):return np.array([1+row.sum(),2+np.dot(row,row)],dtype=np.float64)
    def model(self,seed):return fresh(seed,lambda:torch.nn.Linear(64,2))

    def trajectory(self,model,data,shuffle,recipe='forward',reference=None,key=None):
        if recipe=='mark':model.fixture_marked=True
        tr=Trajectory(model,data,shuffle,self.budget,recipe='forward' if recipe=='mark' else recipe,
                      batch=3,real=True)
        tr.score_scale=1.
        return tr

    def key_for(self,name):return dict(owner=int(name[5:]),wrong=False)
    def wrong_keys(self,name,seeds,final):
        from final_campaign import WRONG_OFFSETS
        seeds.get(name+('/wrong' if final else '/calibration_wrong'),WRONG_OFFSETS)
        return [dict(owner=int(name[5:]),wrong=True,index=i) for i in range(16)]
    def gate(self,model,data):return dict(forward=.1,random_jvp=.2)

    def truth(self,case,data,stage):
        return self.unit('truth/'+stage+'/'+case['id'],counts(),lambda:torch.ones((len(data.x64),2)))

    def copied(self,case,arm,data):
        tr=self.trajectory(copy.deepcopy(case['models']['M' if arm=='A3' else 'O']),data,
                           self.seeds.get(case['id']+'/shared_attack_shuffle'))
        tr.extend(self.attack_steps);tr.model.fixture_attacked=True
        return tr.model,tr.record()

    def detector(self,model,reference,data,key,scale):
        if model is reference:return 0.
        return (.05 if getattr(model,'fixture_marked',False) and not getattr(model,'fixture_attacked',False)
                and not key['wrong'] and key['owner']!=9 else .001)

    def measure_value(self,case,arm,data,tasks,stage,truth):
        self.fixture_pause('evaluate/'+stage+'/'+case['id']+'/'+arm)
        model=copy.deepcopy(case['models'][arm]).eval();x,v,_,_=data.tensors()
        j=torch.func.jvp(model,(x,),(v[:,0],))[1].detach().numpy().astype(np.float64)
        i=int(case['id'][5:]);base=.1+.001*i
        value=base*(1.4+.01*i) if arm=='R-a' else base*(1.1 if arm=='M' else 1)
        return dict(forward=value,random_jvp=value,key_jvp=value,task=value*(1+.002*i)),dict(jprediction=j[:,None,:])

    def fixture_pause(self,key):
        if self.attempt==0 and self.config.get('pause_key')==key:
            publish_handshake(self.path/'FIXTURE_PAUSED.json',dict(key=key,pid=os.getpid(),last_complete=self.journal.last_complete))
            while True:time.sleep(.05)
