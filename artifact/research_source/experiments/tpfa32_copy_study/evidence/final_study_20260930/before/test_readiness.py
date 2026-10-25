"""Reporting regressions and static cost-fixture checks; no real numerical work."""
import copy
import json
import os
from pathlib import Path
import pytest
import torch
import campaign
import numerics
from core import HERE, read, write, Budget
from analyze import analyze, report
from outcomes import (qualification_packet, start_final_records, owner_record,
                      retained_results, incomplete_report, qualification_status)
from test_contracts import fixture_record, lock_fixture, bounded_runtime, toy
from test_workflows import FixtureCampaign


def test_ten_qualified_missing_final_retains_ten_and_new_fixture():
    data=fixture_record();data['owners'].pop()
    result=analyze(data)
    assert result['N_q']==10 and result['removal_count']==9
    assert result['removal_fraction_qualified']==.9 and result['removal_fraction_planned']==.9
    assert result['primary']['label']=='incomplete/degenerate'
    assert result['joint_table'][-1]['qualification'] is True
    assert result['joint_table'][-1]['outcome_status']=='absent'
    attempt=Path(os.environ['TPFA_ATTEMPT'])
    write(attempt/'MISSING_OWNER_FIXTURE.json',data)
    report(data,attempt/'missing_owner_analysis')


def test_nine_actually_qualified_and_observed_intact_failure():
    data=fixture_record();data['owners'][-1]['qualified']=False;lock_fixture(data)
    result=analyze(data);assert result['N_q']==9 and result['removal_fraction_qualified']==1
    data['owners'][0]['intact']['detected']=False
    result=analyze(data)
    assert result['N_q']==9 and result['removal_count']==8
    assert result['joint_table'][0]['observed'] and result['joint_table'][0]['outcome_status']=='complete'
    assert not result['joint_table'][0]['removal']


@pytest.mark.parametrize('defect',['missing','hash','missing_id','conflict','duplicate','unlocked','wrong_count'])
def test_bad_qualification_never_shrinks_denominator(defect):
    data=fixture_record()
    if defect=='missing':del data['locked_qualification']
    elif defect=='hash':data['locked_qualification']['identity']['sha256']='0'*64
    elif defect=='conflict':data['owners'][0]['qualified']=False
    else:
        doc=json.loads(data['locked_qualification']['content'])
        if defect=='missing_id':doc['owners'].pop()
        if defect=='duplicate':doc['owners'][-1]=doc['owners'][0]
        if defect=='unlocked':doc['locked_before_final']=False
        if defect=='wrong_count':doc['N_q']=9
        import hashlib
        text=json.dumps(doc);data['locked_qualification']['content']=text
        data['locked_qualification']['identity'].update(bytes=len(text.encode()),sha256=hashlib.sha256(text.encode()).hexdigest())
    result=analyze(data)
    assert result['N_q'] is None and result['removal_count'] is None
    assert result['removal_fraction_qualified'] is None and result['removal_fraction_planned'] is None
    assert result['removal_endpoint']=='unavailable/incomplete qualification evidence'
    assert result['primary']['label']=='copy advantage' # quality population is independent
    assert not result['joint_labels']['removal']['passed']


def test_retained_summaries_after_later_injected_failure(tmp_path,monkeypatch):
    c=FixtureCampaign(tmp_path/'partial')
    names=[f'owner{i}' for i in range(10)]
    for name in names:
        case=c.train_owner(name);c.mark(case);case['qualified']=True;case['threshold']=.023
        for arm in ('A0','A3','Rstar'):case['models'][arm]=toy(42)
        c.cases.append(case)
    write(c.out.path/'QUALIFICATION.json',dict(planned_owner_ids=names,locked_before_final=True,N_q=10,
          owners=[dict(owner=n,qualified=True) for n in names]))
    class Old:
        def b3_key(self,seed):return {'fixture':True}
        def b3_score(self,*args):return .001
    monkeypatch.setattr(campaign,'legacy',lambda:(None,Old()))
    monkeypatch.setattr(campaign,'keyed_truth',lambda *args:torch.ones((2,2)))
    def measure(case,arm,*args,**kwargs):
        if case['id']=='owner1' and arm=='M':raise RuntimeError('injected later-owner failure')
        return dict(forward=.1,random_jvp=.1,key_jvp=.1,task=.1),{'jprediction':[[[1.,1.]]]}
    c.measure=measure;data=c.data('test',2,0)
    c.run_final=lambda:campaign.Campaign.final_evaluation(c,data,{'initial_errors':[1]},c.seeds,names)
    with pytest.raises(RuntimeError,match='later-owner'):c.execute()
    retained=read(c.out.path/'RESULTS_INCOMPLETE.json');result=read(c.out.path/'analysis_incomplete/analysis.json')
    assert len(retained['owners'])==1 and retained['owners'][0]['id']=='owner0'
    assert retained['owner_outcomes']['owner1']=='failed' and retained['owner_outcomes']['owner2']=='absent'
    assert retained['partial_owner_measurements']['owner1'][0]['arm']=='O'
    assert result['N_q']==10 and result['primary']['label']=='incomplete/degenerate'
    assert read(c.out.path/'FAILED.json')['incomplete_reporting_error'] is None
    assert retained_results(c.out.path)['locked_qualification']==retained['locked_qualification']


def test_forward_measurements_survive_inverse_failure(monkeypatch):
    saved=[]
    class Old:
        def evaluate(self,*args):return {'forward':.1},{'forward':[.1,.1],'prediction':torch.ones(2,2)}
    class Data:
        def tensors(self):return ()
    monkeypatch.setattr(numerics,'legacy',lambda:(None,Old()))
    def fail(*args,**kwargs):raise RuntimeError('inverse failed')
    monkeypatch.setattr(numerics,'inverse_batched',fail)
    with pytest.raises(RuntimeError):numerics.evaluate(toy(),Data(),{'starts':None,'targets':None},
                              None,Budget(),'fixture',fixed_inverse_batches=True,progress=saved.append)
    assert saved[0]['type']=='forward_jvp_complete' and saved[0]['samples']['forward']==[.1,.1]


def test_cost_sources_parse_and_budget_constants():
    import ast
    for path in HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    import cost_supervise
    assert cost_supervise.LIMITS==dict(solver_attempts=1600,model_updates=96,inverse_updates=480)
    assert cost_supervise.ROOT.name=='readiness_cost_20260930'
