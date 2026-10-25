"""Fabricated orchestration fixtures. Expensive methods replaced; no scientific runs."""
import copy
import numpy as np
import pytest
import torch
import campaign
from campaign import Campaign
from core import HERE, Budget, Output, Seeds, read, write, identity
from numerics import Oracle, Splits, controls, forward
from test_contracts import toy, synthetic


class FixtureCampaign(Campaign):
    def __init__(self,path):
        self.config=read(HERE/'draft.json');self.config['selection']={
            'primary':'R-b','recipes':{k:dict(updates=1920) for k in ('R-a','R-b')}}
        self.mode='verification';self.out=Output(path,'verification')
        self.budget=Budget();self.runtime={'fixture':True}
        self.seeds=Seeds(path/'PRIVATE_SEEDS.json','fixture',fixture=15001)
        self.splits=Splits();self.oracle=Oracle(synthetic,self.budget)
        self.cases=[];self.facts=[];self.events=[]
        write(path/'CONFIG.json',self.config)

    def train_owner(self,name):
        self.events.append('train/'+name)
        return dict(id=name,models={'O':toy()},costs={})

    def mark(self,case):
        self.events.append('mark/'+case['id']);case['models']['M']=toy(123)
        case['key']={'v':np.ones(64,dtype=np.float32)/8,'w':np.ones(2,dtype=np.float32)}
        case['scale']=1.

    def data(self,label,n,k,seeds=None):
        if label=='final':
            assert (self.out.path/'TRAINING_CALIBRATION_CLOSED.json').exists()
            self.events.append('final_data_after_closure')
        return forward(self.oracle,controls(4321,2),label)

    def calibrate(self):
        assert sum(e.startswith('mark/') for e in self.events)==10
        for case in self.cases:case['qualified']=True;case['threshold']=.023
        write(self.out.path/'QUALIFICATION.json',dict(locked=True,fixture=True))

    def attack_data(self,case,include_rb=True):return self.data('attack',2,0),self.data('rb',2,0)

    def copied_attacks(self,case,data):
        for arm in ('A0','A3'):case['models'][arm]=toy(456)

    def replacement(self,case,recipe,data):
        class FabricatedTrajectory:
            def __init__(self):self.model=toy(789);self.losses=[];self.updates=0
            def extend(self,n):self.updates=n
            def record(self):return dict(fixture=True,updates=self.updates)
            def save(self,path):write(path,dict(fixture=True,updates=self.updates))
        return FabricatedTrajectory()

    def final_evaluation(self,data,tasks,seeds,names):
        assert self.events[:10]==['train/'+name for name in names]
        assert len(names)==10 and self.events[-1]=='final_data_after_closure'
        assert all('R-a@1920' in c['models'] and 'R-b@1920' in c['models'] for c in self.cases)
        assert all(c['models']['Rstar'] is c['models']['R-b'] for c in self.cases)
        return dict(fixture=True,gate_order=self.events)


def fixture_environment(monkeypatch,passing=True):
    class Old:
        def evaluate(self,*args):return dict(forward=.1 if passing else .13,random_jvp=.4),{}
    monkeypatch.setattr(campaign,'legacy',lambda:(None,Old()))
    monkeypatch.setattr(campaign,'task_data',lambda *args:dict(initial_errors=[1]))
    original=Seeds
    monkeypatch.setattr(campaign,'Seeds',lambda path,stage,forbidden=():
                        original(path,stage,fixture=70001,forbidden=forbidden))


def test_final_gate_order_and_retained_selected_states(tmp_path,monkeypatch):
    fixture_environment(monkeypatch)
    c=FixtureCampaign(tmp_path/'workflow');result=c.execute()
    assert result['fixture'] and (c.out.path/'COMPLETED.json').exists()
    closure=read(c.out.path/'TRAINING_CALIBRATION_CLOSED.json')
    assert len(closure['models'])==40
    assert (c.out.path/'PRIVATE_FINAL_SEEDS.json').exists()


def test_failed_clean_cohort_never_marks_or_draws_final(tmp_path,monkeypatch):
    fixture_environment(monkeypatch,False);c=FixtureCampaign(tmp_path/'failed')
    with pytest.raises(ValueError,match='clean-cohort'):c.execute()
    assert len(c.cases)==10 and not any(e.startswith('mark/') for e in c.events)
    assert not (c.out.path/'PRIVATE_FINAL_SEEDS.json').exists()
    assert (c.out.path/'FAILED.json').exists()


def test_real_final_evaluation_record_and_analysis_wiring(tmp_path,monkeypatch):
    c=FixtureCampaign(tmp_path/'evaluation')
    names=[f'owner{i}' for i in range(10)]
    for i,name in enumerate(names):
        case=c.train_owner(name);c.mark(case);case['qualified']=i!=9;case['threshold']=.023
        for arm in ('A0','A3','R-a','Rstar'):case['models'][arm]=toy(222+i)
        c.cases.append(case)
    class Old:
        def b3_key(self,seed):return {'fixture_key':seed}
        def b3_score(self,model,owner,x,key,scale):return .001 if 'fixture_key' in key else .03
    monkeypatch.setattr(campaign,'legacy',lambda:(None,Old()))
    monkeypatch.setattr(campaign,'keyed_truth',lambda *args:torch.ones((2,2)))
    def measure(case,arm,data,tasks,label,key=False,truth=None):
        i=int(case['id'][-1]);ratio=(2+i*.02 if arm=='Rstar' else 1.)
        return dict(forward=.1,random_jvp=.1*ratio,key_jvp=.1,task=.1*ratio),{
            'jprediction':np.full((2,2,2),1+i*.1+(0 if arm=='O' else .01)).tolist()}
    c.measure=measure
    data=c.data('evaluation',2,0)
    result=Campaign.final_evaluation(c,data,{'initial_errors':[.1]},c.seeds,names)
    record=read(c.out.path/'RESULTS.json');analysis=read(c.out.path/'analysis/analysis.json')
    assert len(record['null_pairs'])==190 and analysis['nulls']['status']=='complete'
    assert analysis['N_q']==9 and len(analysis['joint_table'])==10
    assert result['counts']==c.budget.counts
