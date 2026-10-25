"""Affected friend-amendment checks. No real solvers or full inverse sets execute."""
import copy
import numpy as np
import pytest
import torch
import numerics
import campaign
from core import HERE, Budget, read, validate_config
from analyze import analyze, quality_label, interval, b2_rank
from test_contracts import fixture_record, lock_fixture, toy, synthetic, bounded_runtime


def bounds(lower,upper):
    return {m:dict(status='complete',lower=lower,upper=upper) for m in ('random_jvp','task')}


def test_distinct_equivalence_intervals_and_advantage_boundaries():
    assert quality_label(bounds(.79,1.26),bounds(.81,1.24))=='equivalence'
    assert quality_label(bounds(.79,1.26),bounds(.79,1.26))=='trade-off/inconclusive'
    assert quality_label(bounds(.79,1.26),bounds(.8,1.25))=='equivalence'
    assert quality_label(bounds(1.25,1.4),bounds(1.26,1.39))=='trade-off/inconclusive'
    assert quality_label(bounds(.6,.8),bounds(.61,.79))=='trade-off/inconclusive'
    for coverage in (.975,.95):
        assert interval([1]*10,coverage)['status']=='degenerate'
        for invalid in (0,-1,float('inf'),float('nan')):
            assert interval([1]*9+[invalid],coverage)['status']=='incomplete'


def test_all_planned_owners_primary_and_nine_qualified_minimum():
    data=fixture_record()
    data['owners'][-1]['qualified']=False
    lock_fixture(data)
    nine=analyze(data)
    assert nine['N_q']==9 and nine['minimum_qualified']==9
    assert nine['joint_labels']['removal']['passed']
    primary=copy.deepcopy(nine['primary'])
    data['owners'][-2]['qualified']=False
    lock_fixture(data)
    eight=analyze(data)
    assert eight['primary']==primary and eight['N_q']==8
    assert not eight['joint_labels']['removal']['passed']
    assert eight['removal_endpoint']=='insufficiently qualified'
    assert eight['removal_fraction_planned']==.8
    del data['owners'][-1]['metrics']['Rstar']['task']
    missing=analyze(data)
    assert missing['primary']['label']=='incomplete/degenerate'
    assert missing['primary']['planned_n']==10 and len(missing['joint_table'])==10
    assert missing['primary']['errors'][0]['owner']=='fixture9'


def test_null_population_and_no_six_owner_reinterpretation():
    data=fixture_record();result=analyze(data)
    assert result['nulls']['per_key']==19 and result['nulls']['expected_pairs']==190
    data['null_pairs'].pop();assert analyze(data)['nulls']['status']=='incomplete'
    assert b2_rank(1,[2]*8,[3]*10)['status']=='unavailable'
    data['planned_owner_ids']=data['planned_owner_ids'][:6]
    with pytest.raises(ValueError,match='ten planned'):analyze(data)


def test_fixed_draft_no_expansion_or_batch64():
    config=read(HERE/'draft.json');validate_config(config)
    assert config['status']=='draft' and config['owners']==10 and config['oracle_budget']==256
    for key,value in [('owners',6),('owners',11),('oracle_budget',512),
                      ('inverse_batch_size',64),('development_tasks',16),('final_tasks',65)]:
        changed=copy.deepcopy(config);changed[key]=value
        with pytest.raises(ValueError):validate_config(changed)


def test_four_fixed_batches_dispatch_only_no_inverse_set_executed(monkeypatch):
    starts=np.arange(64*64,dtype=np.float64).reshape(64,64)
    targets=np.arange(64*2,dtype=np.float64).reshape(64,2)
    calls=[];model=object();oracle=object();budget=Budget()
    def stub(m,x,y,o,b,label):
        assert m is model and o is oracle and b is budget and len(x)==16
        calls.append((x.copy(),y.copy(),label))
        begin=int(y[0,0]/2)
        return dict(per_target=list(range(begin,begin+16)),optimized_controls=x.tolist(),input_updates=120)
    monkeypatch.setattr(numerics,'inverse',stub)
    result=numerics.inverse_batched(model,starts,targets,oracle,budget,'fixture')
    assert [r['begin'] for r in result['batches']]==[0,16,32,48]
    assert np.array_equal(np.concatenate([c[0] for c in calls]),starts)
    assert np.array_equal(np.concatenate([c[1] for c in calls]),targets)
    assert result['per_target']==list(range(64)) and result['mean']==31.5
    assert result['input_updates']==480 and result['updates_per_target']==120 and result['target_updates']==7680
    assert all('batch of 16' in r['loss_reduction'] for r in result['batches'])
    with pytest.raises(ValueError):numerics.inverse_batched(model,starts[:63],targets[:63],oracle,budget,'bad')


def test_batch16_loss_averaging_with_inert_optimizer(monkeypatch):
    # One cheap batch; fake Adam inspects gradient but performs zero optimizer updates.
    captured=[]
    class Model(torch.nn.Module):
        def forward(self,x):return x[:,:2]
    class AdamSpy:
        def __init__(self,parameters,lr):
            self.x=list(parameters)[0];assert lr==.03
        def zero_grad(self):self.x.grad=None
        def step(self):captured.append(self.x.grad.clone())
    monkeypatch.setattr(torch.optim,'Adam',AdamSpy)
    starts=np.full((16,64),.4,dtype=np.float64);targets=np.full((16,2),.7,dtype=np.float64)
    budget=Budget(limits={'inverse_updates':120}) # toy accounting, not real milestone counter
    result=numerics.inverse(Model(),starts,targets,lambda row,label:synthetic(row),budget,'toy')
    assert len(captured)==120 and result['input_updates']==120
    assert torch.allclose(captured[0][:,:2],torch.full((16,2),-.3/16),atol=1e-8)
    assert torch.count_nonzero(captured[0][:,2:])==0
    with pytest.raises(ValueError,match='at most 16'):
        numerics.inverse(Model(),np.zeros((64,64)),np.ones((64,2)),None,budget,'forbidden')


def test_campaign_measure_enforces_fixed_dispatch(monkeypatch):
    calls=[]
    def fake(*args,**kwargs):calls.append(kwargs);return {'task':.1},{'fixture':True}
    monkeypatch.setattr(campaign,'evaluate',fake)
    class Log:
        def log(self,row):pass
    c=campaign.Campaign.__new__(campaign.Campaign)
    c.out=Log();c.oracle=None;c.budget=Budget()
    c.measure(dict(id='fixture',models={'O':object()}),'O',None,None,'development')
    assert len(calls)==1 and calls[0]['fixed_inverse_batches'] is True and callable(calls[0]['progress'])


def test_evaluate_routes_to_fixed_batches(monkeypatch):
    class Old:
        def evaluate(self,*args):return {'forward':.1},{}
    class Data:
        def tensors(self):return ()
    calls=[]
    monkeypatch.setattr(numerics,'legacy',lambda:(None,Old()))
    monkeypatch.setattr(numerics,'inverse',lambda *args:pytest.fail('single-batch route forbidden'))
    def fixed(*args,**kwargs):calls.append(args);return {'mean':.2,'fixture':True}
    monkeypatch.setattr(numerics,'inverse_batched',fixed)
    metrics,raw=numerics.evaluate(toy(),Data(),{'starts':'fixture-starts','targets':'fixture-targets'},
                                  None,Budget(),'fixture',fixed_inverse_batches=True)
    assert len(calls)==1 and calls[0][1:3]==('fixture-starts','fixture-targets')
    assert metrics['task']==.2 and raw['task']['fixture']
