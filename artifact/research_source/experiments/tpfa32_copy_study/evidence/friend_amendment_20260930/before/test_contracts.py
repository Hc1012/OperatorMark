"""Explicit, cheap fixtures; no scientific campaign or real solver calls."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import pytest
import torch
from core import (HERE, Budget, Output, Seeds, cohort_gate, utility, detection, threshold,
                  selection_score, select_recipe, read, write, digest, authorize_campaign)
from numerics import (Oracle, Splits, Observations, allocation, controls, directions,
                      forward, derivatives, fresh, inverse, legacy, runtime)
from training import Trajectory, legacy_attack
from analyze import interval, quality_label, analyze, b2_rank, report


@pytest.fixture(scope='session', autouse=True)
def bounded_runtime():
    Budget.verification().check()
    runtime()


def synthetic(row):
    return np.array([1+row.sum(), 2+np.dot(row,row)], dtype=np.float64)


def observations(n=85):
    return forward(Oracle(synthetic,Budget()),controls(8101,n),'toy')


def toy(seed=8111):
    return fresh(seed,lambda:torch.nn.Linear(64,2))


def equal(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def test_allocation_exact_original_rows_and_information():
    oracle=Oracle(synthetic,Budget());splits=Splits()
    shared,rb=allocation(oracle,8101,8102,splits)
    assert shared.information_calls==256 and rb.information_calls==255
    assert oracle.record()==dict(requests=426,attempted=426,successful=426,failed=0,cache_served=0)
    assert shared.x64[:85].tobytes()==rb.x64.tobytes()
    assert shared.y64[:85].tobytes()==rb.y64.tobytes()
    assert rb.x64.tobytes()!=rb.x64.astype(np.float32).astype(np.float64).tobytes()
    assert len(rb.y64)==85 and not np.shares_memory(rb.y64,shared.y64)
    assert set(vars(rb))=={'x64','y64','v64','dy64','information_calls'}
    plus=rb.x64[:,None,:]+.001*rb.v64
    assert all(row.tobytes() in oracle.cache for row in plus[:,0])
    assert np.allclose(rb.dy64[:,0,0],rb.v64[:,0].sum(axis=1),atol=1e-10)
    with pytest.raises(ValueError): rb.y64[0,0]=0
    with pytest.raises(ValueError): splits.add('reuse',shared.x64[:1])


def test_fd_unclipped_cache_and_failed_attempts():
    rows=np.full((1,64),.2,dtype=np.float64);v=np.zeros((1,1,64));v[0,0,0]=1
    seen=[]
    oracle=Oracle(lambda r:(seen.append(r.copy()) or synthetic(r)),Budget())
    base=forward(oracle,rows,'base');derivatives(oracle,base,v,'fd')
    assert min(row[0] for row in seen)==.199
    oracle(rows[0],'cache');assert oracle.cache_served==1 and oracle.attempted==3
    def fail(row): raise ValueError('fixture failure')
    failed=Oracle(fail,Budget())
    with pytest.raises(ValueError): failed(rows[0],'failed')
    assert failed.record()['failed']==1 and failed.successful==0


def test_fresh_optimizer_partial_batches_and_continuous_resume(tmp_path):
    data=observations();a=Trajectory(toy(),data,8112,Budget(),batch=16)
    b=Trajectory(toy(),data,8112,Budget(),batch=16)
    assert not a.optimizer.state and not b.optimizer.state
    assert not equal(toy(8111).state_dict(),toy(8113).state_dict())
    a.extend(14);b.extend(7)
    saved=tmp_path/'continuous.pt';pin=b.save(saved)
    resumed=Trajectory(toy(8999),data,9999,Budget(),batch=16)
    resumed.resume(saved,pin);resumed.extend(14)
    assert equal(a.model.state_dict(),resumed.model.state_dict())
    assert equal(a.optimizer.state_dict(),resumed.optimizer.state_dict())
    assert equal(a.shuffle.get_state(),resumed.shuffle.get_state())
    assert equal(a.order,resumed.order) and a.offset==resumed.offset
    assert a.batch_sizes[:6]==[16,16,16,16,16,5]
    assert a.examples==202 and a.updates==14
    weights=tmp_path/'weights.pt';torch.save(a.model.state_dict(),weights)
    with pytest.raises(ValueError,match='weights-only'): resumed.resume(weights,digest(weights))
    with pytest.raises(FileExistsError): a.save(saved)
    with pytest.raises(ValueError,match='key or teacher'):
        Trajectory(toy(),data,1,Budget(),key={'secret':1})


def test_legacy_adapter_exact_weights_and_shared_shuffle():
    data=observations(256);old=legacy()[1]
    a,b,c=toy(),toy(),toy()
    class Clock:
        def guard(self): pass
        def elapsed(self): return 0
    x,_,y,_=data.tensors()
    old.attack(a,x,y,8131,Clock(),lambda row:None,0,'fixture')
    stats=legacy_attack(b,data,8131,Budget())
    continuous=Trajectory(c,data,8131,Budget());continuous.extend(1920)
    assert stats['updates']==1920 and stats['examples']==61440
    assert equal(a.state_dict(),b.state_dict()) and equal(b.state_dict(),c.state_dict())
    assert torch.optim.Adam.__name__=='Adam'


def test_disposable_inverse_keeps_training_state():
    data=observations(2);trainer=Trajectory(toy(),data,1,Budget());trainer.extend(1)
    before=copy.deepcopy(trainer.optimizer.state_dict());weights=copy.deepcopy(trainer.model.state_dict())
    budget=Budget(limits=dict(inverse_updates=120))
    result=inverse(trainer.model,data.x64,data.y64,Oracle(synthetic,Budget()),budget,'toy')
    assert budget.counts['inverse_updates']==120 and result['input_updates']==120
    assert all(p.requires_grad for p in trainer.model.parameters())
    assert equal(weights,trainer.model.state_dict()) and equal(before,trainer.optimizer.state_dict())
    trainer.extend(2);assert not equal(weights,trainer.model.state_dict())


@pytest.mark.parametrize('metric',('forward','random_jvp','key_jvp','task'))
def test_each_utility_gate_required(metric):
    owner={k:.1 for k in ('forward','random_jvp','key_jvp','task')};candidate=owner.copy()
    assert utility(candidate,owner)['passed']
    candidate[metric]=1
    assert not utility(candidate,owner)['passed']
    del candidate[metric];assert not utility(candidate,owner)['passed']


def test_detection_cohort_and_complete_null_population():
    wrong=[.02]*16
    assert not detection(.023,wrong,.023)['detected']
    assert detection(.024,wrong,.023)['detected']
    assert not detection(.033,[.03]*16,.023)['detected']
    assert threshold([-.04]*5,wrong)==pytest.approx(.043,abs=1e-15)
    with pytest.raises(ValueError): threshold([.01]*4,wrong)
    passing=[dict(forward=.12,random_jvp=.50)]*6
    assert cohort_gate(passing,6)
    assert not cohort_gate(passing[:-1],6)
    assert not cohort_gate(passing[:-1]+[dict(forward=.121,random_jvp=.1)],6)


def test_selection_completeness_and_ties():
    rows=[dict(clean=dict(forward=1,random_jvp=2,task=4),
               replacement=dict(forward=2,random_jvp=4,task=8))]*3
    assert selection_score(rows)==pytest.approx(2)
    with pytest.raises(ValueError): selection_score(rows[:2])
    bad=copy.deepcopy(rows);bad[1]['replacement']['task']=0
    with pytest.raises(ValueError): selection_score(bad)
    history={k:[dict(score=1.,updates=3840),dict(score=1.,updates=1920)] for k in ('R-a','R-b')}
    selected=select_recipe(history)
    assert selected['primary']=='R-a' and selected['recipes']['R-b']['updates']==1920


def fixture_record():
    names=[f'fixture{i}' for i in range(6)]
    result=dict(schema='tpfa32_results.v1',fixture=True,planned_owner_ids=names,owners=[],b2={},null_pairs=[])
    for i,name in enumerate(names):
        result['owners'].append(dict(id=name,qualified=True,intact=dict(detected=True,utility=True),
          attacked=dict(detected=False,utility=True),
          metrics={'A3':dict(random_jvp=1,task=1),'Rstar':dict(random_jvp=2+i*.02,task=1+i*.001)}))
        result['b2'][name]=dict(descendant=1,clean=[2]*5,replacements=[3]*6)
        for other in names:
            if name!=other:result['null_pairs'].append(dict(reference=name,model='O/'+other,detected=False))
            result['null_pairs'].append(dict(reference=name,model='Rstar/'+other,detected=False))
    return result


def test_quantile_and_predetermined_intervals():
    r=interval([.99,1,1.01,.995,1.005,1.002])
    assert r['critical']==pytest.approx(3.163381449748624,abs=1e-8)
    assert r['df']==5 and r['pointwise_coverage']==.975
    assert .98<r['lower']<1<r['upper']<1.02
    both={k:r for k in ('random_jvp','task')}
    assert quality_label(both)=='equivalence'
    assert interval([1]*6)['status']=='degenerate'
    for v in (0,-1,float('nan'),float('inf')):
        assert interval([1,1,1,1,1,v])['status']=='incomplete'
    assert interval([1]*5)['status']=='incomplete'


@pytest.mark.parametrize('lower,upper,label',[(.8,1.2,'trade-off/inconclusive'),
    (.81,1.25,'trade-off/inconclusive'),(1.26,2,'copy advantage'),
    (.6,.79,'replacement advantage'),(.9,1.2,'equivalence')])
def test_primary_strict_boundaries(lower,upper,label):
    assert quality_label({k:dict(status='complete',lower=lower,upper=upper)
                          for k in ('random_jvp','task')})==label


def test_locked_qualification_missing_owners_and_joint_same_owners():
    data=fixture_record();result=analyze(data)
    assert result['primary']['label']=='copy advantage'
    assert result['joint_labels']['removal']['passed'] and result['N_q']==6
    data['owners'][0]['intact']['detected']=False
    r=analyze(data);assert r['N_q']==6 and r['removal_count']==5 and r['joint_labels']['removal']['passed']
    data['owners'][1]['attacked']['utility']=False
    assert not analyze(data)['joint_labels']['removal']['passed']
    for row in data['owners']:row['qualified']=False
    r=analyze(data);assert r['removal_fraction_qualified'] is None and r['removal_count']==0
    data['owners'].pop();r=analyze(data)
    assert len(r['joint_table'])==6 and r['primary']['label']=='incomplete/degenerate'


def test_b2_ties_and_dependent_null_matrix():
    rank=b2_rank(1,[.5,1,2,2,2],[3]*6)
    assert rank['rank']==2 and rank['ties']==1 and rank['margin']==-.5
    result=analyze(fixture_record())
    assert result['nulls']['status']=='complete' and result['nulls']['observed_pairs']==66
    assert result['nulls']['dependent'] and result['nulls']['population_FPR_bound'] is None


def test_default_and_draft_final_before_numerical_import(monkeypatch,capsys):
    import study
    import builtins
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name in ('campaign','numerics','training','spatial','run'): raise AssertionError('numerical import')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    assert study.main([])==0
    assert json.loads(capsys.readouterr().out)['numerical_work'] is False
    with pytest.raises(PermissionError,match='draft'): study.main(['final'])
    config=read(HERE/'draft.json');config['status']='locked';config['selection']={'placeholder':True}
    with pytest.raises(PermissionError,match='separate'): authorize_campaign(config,None,'final')
    assert analyze(fixture_record())['fixture']


def test_seed_offsets_outputs_and_durable_quota(tmp_path,monkeypatch):
    import secrets
    monkeypatch.setattr(secrets,'randbits',lambda _:pytest.fail('no OS seed draws in verification'))
    seeds=Seeds(tmp_path/'fixture_seeds.json','fixture',fixture=1)
    assert seeds.get('key',(0,10000))==1
    with pytest.raises(ValueError): seeds.get('key')
    seeds.used.add(100001)
    with pytest.raises(ValueError,match='no redraw'): seeds.get('collision')
    assert 'fixture/collision' not in seeds.values
    out=Output(tmp_path/'out','verification');out.finish(dict(passed=True))
    with pytest.raises(FileExistsError):out.finish(dict(passed=False))
    with pytest.raises(FileExistsError):out.finish(dict(passed=False),failed=True)
    with pytest.raises(FileExistsError):Output(tmp_path/'out','verification')
    budget=Budget(tmp_path/'count.json',dict(solver_attempts=1));budget.charge('solver_attempts')
    with pytest.raises(RuntimeError): budget.charge('solver_attempts')
    assert read(tmp_path/'count.json')['counts']['solver_attempts']==1


def test_fixture_analysis_artifact():
    attempt=Path(os.environ['TPFA_ATTEMPT'])
    data=fixture_record();write(attempt/'ANALYSIS_FIXTURE.json',data)
    result=report(data,attempt/'fixture_analysis')
    assert result['fixture'] and len(result['joint_table'])==6
    assert result['secondary']['fixture0']['ratios']['Rstar/A3']['random_jvp']==2
    assert result['secondary']['fixture0']['ratios']['R-a@1920/A0']['task'] is None


def test_all_new_sources_parse():
    import ast
    for source in HERE.glob('*.py'):ast.parse(source.read_text(encoding='utf-8'),filename=str(source))
