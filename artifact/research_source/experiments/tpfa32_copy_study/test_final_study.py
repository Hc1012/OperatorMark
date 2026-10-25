"""Bounded final-only synthetic orchestration and unchanged analysis safeguards."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest
import torch
from core import HERE,Budget,read,write,identity,digest,source_hashes
from numerics import runtime
from dev_state import counts,load_owned
from dev_process import process_identity
from final_supervise import resume_allowed,validate_pins,validate_run
from final_contract import validate_final_config
from final_campaign import StageSeeds,seed_plan
from test_contracts import equal,fixture_record
from analyze import analyze,report


@pytest.fixture(scope='session',autouse=True)
def environment():
    assert os.environ['TPFA_CATEGORY']=='final_study_verification'
    Budget.verification().check()
    write(Path(os.environ['TPFA_ATTEMPT'])/'RUNTIME.json',runtime())


def fixture_root(path,pause=None):
    path.mkdir();now=time.time()
    write(path/'AUTHORITY.json',dict(started_unix=now,deadline_unix=min(now+180,float(os.environ['TPFA_DEADLINE'])-12),
        closure_deadline_unix=min(now+190,float(os.environ['TPFA_DEADLINE'])-5),
        limits=counts(model_updates=5000,inverse_updates=0),evidence_bytes=256*1024**2))
    config=read(HERE/'final.json');config.update(fixture_only=True,pause_key=pause)
    write(path/'CONFIG.json',config);return path


def start(root,mode):
    stream=(root/(mode+'_supervisor.txt')).open('xb')
    child=subprocess.Popen([sys.executable,'-B',str(HERE/'final_supervise.py'),mode,'--output',str(root)],stdout=stream,stderr=stream)
    return child,stream


def wait(child,stream):
    try:return child.wait(timeout=150)
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)
        stream.close()


def test_ten_owner_final_control_interruption_resume(tmp_path):
    control=fixture_root(tmp_path/'control');child,stream=start(control,'fixture-start')
    assert wait(child,stream)==0,(control/'fixture-start_supervisor.txt').read_text()
    interrupted=fixture_root(tmp_path/'interrupted','evaluate/final/owner0/M')
    child,stream=start(interrupted,'fixture-start')
    try:
        until=time.monotonic()+120
        while not (interrupted/'FIXTURE_PAUSED.json').exists():
            if child.poll() is not None:pytest.fail((interrupted/'fixture-start_supervisor.txt').read_text())
            if time.monotonic()>until:raise TimeoutError('fixture final pause unavailable')
            time.sleep(.05)
        assert (interrupted/'PRIVATE_FINAL_SEEDS.json').exists()
        assert read(interrupted/'FIXTURE_PAUSED.json')['last_complete']['key']=='evaluate/final/owner0/O'
        with pytest.raises((RuntimeError,FileNotFoundError)):resume_allowed(interrupted)
        write(interrupted/'INTERRUPT_REQUEST_0.json',dict(reason='external operator interruption'))
        assert wait(child,stream)==1
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)
        stream.close()
    prior=read(interrupted/'attempt_0/result.json')
    assert prior['owned_worker_ended'] and prior['infrastructure']
    assert read(interrupted/'attempt_0/analysis_incomplete/analysis.json')['primary']['label']=='incomplete/degenerate'
    deadline=read(interrupted/'AUTHORITY.json')['deadline_unix']
    child,stream=start(interrupted,'fixture-resume')
    assert wait(child,stream)==0,(interrupted/'fixture-resume_supervisor.txt').read_text()
    assert read(interrupted/'AUTHORITY.json')['deadline_unix']==deadline
    with pytest.raises(RuntimeError,match='continuation'):resume_allowed(interrupted)
    a,b=read(control/'RESULTS.json'),read(interrupted/'RESULTS.json')
    assert len(a['owners'])==len(b['owners'])==10
    qa=json.loads(a['locked_qualification']['content']);qb=json.loads(b['locked_qualification']['content'])
    assert qa==qb and qa['N_q']==9 and not qa['owners'][9]['qualified']
    for ca,cb in zip(a['owners'],b['owners']):
        assert ca['metrics']==cb['metrics'] and ca['detectors']==cb['detectors']
        assert ca['checkpoints']['equal_step']!=ca['checkpoints']['selected']
        for label in ca['checkpoints']:
            sa,sb=load_owned(ca['checkpoints'][label]),load_owned(cb['checkpoints'][label])
            for field in ('model','optimizer','shuffle_rng','torch_rng','order','offset','updates','epochs','data','cooldown','training_flags'):
                assert equal(sa[field],sb[field]),(ca['id'],label,field)
    for name in ('PRIVATE_SEEDS.json','PRIVATE_FINAL_SEEDS.json'):
        assert digest(control/name)==digest(interrupted/name)
    assert a['null_pairs']==b['null_pairs'] and len(a['null_pairs'])==190
    aa,ab=read(control/'analysis/analysis.json'),read(interrupted/'analysis/analysis.json')
    assert aa['primary']==ab['primary'] and aa['joint_labels']==ab['joint_labels']
    assert aa['primary']['planned_n']==10 and aa['nulls']['status']=='complete'
    assert aa['rule_status']=='approved final milestone 20260930'
    for root in (control,interrupted):
        events=[json.loads(line) for line in (root/'progress.jsonl').read_text().splitlines()]
        complete=[e for e in events if e['type']=='complete']
        assert len([e for e in complete if e['key'].startswith('evaluate/')])==80
        assert len({e['key'] for e in complete})==len(complete)
        first_final=next(e for e in events if e['type']=='begin' and e['key']=='data/final')
        assert read(root/'FIRST_FINAL_GENERATION.json')['started_unix']<first_final['unix']
        assert read(root/'CLEAN_COHORT.json')['passed']
    write(Path(os.environ['TPFA_ATTEMPT'])/'FINAL_FIXTURE.json',dict(passed=True,control=str(control),interrupted=str(interrupted),
        ten_owners=True,failed_mark_retained=True,late_seeds_and_exact_resume=True,all_190_null_identities=True,
        actual_solver_calls=0,actual_spatial_updates=0,actual_inverse_updates=0,
        fixture_note='Synthetic errors and detector; real toy Adam trajectories, no scientific evaluation claim'))


def test_contract_identity_seed_and_resume_refusals(tmp_path,monkeypatch):
    config=read(HERE/'final.json');validate_final_config(config)
    bad=copy.deepcopy(config);bad['selection']['recipes']['R-a']['updates']=1920
    with pytest.raises(ValueError):validate_final_config(bad)
    bad=copy.deepcopy(config);bad['fixture_only']=True
    with pytest.raises(ValueError):validate_final_config(bad)
    auth=dict(source_sha256=source_hashes(),config_identity=identity(HERE/'final.json'),rule_identity=identity(HERE/'RULE_FINAL.md'),development_inputs={})
    validate_pins(auth);auth['source_sha256']['changed']='0'
    with pytest.raises(ValueError,match='source'):validate_pins(auth)
    auth['source_sha256']=source_hashes();auth['rule_identity']['sha256']='0'*64
    with pytest.raises(ValueError,match='rule'):validate_pins(auth)
    seeds=StageSeeds(tmp_path/'seeds.json','fixture',seed_plan(),fixture=51)
    assert StageSeeds(seeds.path,'fixture',seed_plan(),fixture=51).values==seeds.values
    seeds.path.write_text('{}')
    with pytest.raises(ValueError,match='identity'):StageSeeds(seeds.path,'fixture',seed_plan(),fixture=51)
    damaged=tmp_path/'damaged.pt';damaged.write_bytes(b'preserved corrupt fixture')
    fact=identity(damaged);fact['sha256']='0'*64
    with pytest.raises(ValueError,match='identity'):load_owned(fact)
    live=fixture_root(tmp_path/'live');(live/'attempt_0').mkdir()
    who=process_identity(os.getpid())
    write(live/'attempt_0/result.json',dict(owned_worker_ended=True,infrastructure=True,worker=who,launcher=who,supervisor=who))
    with pytest.raises(RuntimeError,match='alive'):resume_allowed(live)
    refused=fixture_root(tmp_path/'scientific_failure');(refused/'attempt_0').mkdir()
    write(refused/'attempt_0/result.json',dict(owned_worker_ended=True,infrastructure=False))
    with pytest.raises(RuntimeError,match='infrastructure'):resume_allowed(refused)


def test_approved_summary_and_missing_owner_population(tmp_path):
    record=fixture_record();record.update(rule_status='approved final milestone 20260930',analysis_rule='t10_adv975_eq95')
    record['owners'].pop();result=report(record,tmp_path/'analysis')
    assert result['N_q']==10 and result['removal_count']==9
    assert result['primary']['label']=='incomplete/degenerate'
    text=(tmp_path/'analysis/SUMMARY.md').read_text()
    assert 'provisional' not in text and 'equivalence_intervals' in text


def test_clean_failure_prevents_marking_and_final_seeds(tmp_path,monkeypatch):
    from final_fixture import TinyFinal
    root=fixture_root(tmp_path/'clean_failure')
    monkeypatch.setattr(TinyFinal,'gate',lambda *a:dict(forward=.13,random_jvp=.2))
    campaign=TinyFinal(root,read(root/'AUTHORITY.json'),0)
    with pytest.raises(ValueError,match='clean cohort failed'):campaign.execute()
    assert len(campaign.cases)==10
    assert not any('/mark/' in key for key in campaign.journal.done)
    assert not (root/'PRIVATE_FINAL_SEEDS.json').exists()
    assert not read(root/'CLEAN_COHORT.json')['passed']
    failure=read(root/'INCOMPLETE_attempt0.json')
    assert failure['type']=='ValueError' and not failure['infrastructure']
    last=json.loads((root/'progress.jsonl').read_text().splitlines()[-1])
    assert last['type']=='stopped' and last['failure']==failure
    from final_summary import terminal_summary
    (root/'attempt_0').mkdir()
    terminal_summary(root,0,dict(scientifically_complete=False,owned_worker_ended=True,infrastructure=False))
    summary=read(root/'attempt_0/analysis_incomplete/analysis.json')
    assert summary['primary']['label']=='incomplete/degenerate' and summary['N_q'] is None
    assert len(summary['unobserved_final_owners'])==10


def test_reused_stage0_and_development_numerical_pins():
    ready=read(HERE/'evidence/development_r2_20260930/READY.json');now=source_hashes()
    for name in ('training.py','numerics.py','dev_state.py','dev_process.py'):
        key='experiments/tpfa32_copy_study/'+name
        assert ready['source_sha256'][key]==now[key]
    for key in ready['source_sha256']:
        if 'darcy_fno32_pilot/' in key or key.endswith('darcy_tpfa.py'):assert ready['source_sha256'][key]==now[key]
    stage=read(HERE/'evidence/stage0_01/COMPLETED.json')
    assert not stage['blockers']
    for fact in stage['evidence']:assert digest(fact['path'])==fact['sha256']
    write(Path(os.environ['TPFA_ATTEMPT'])/'STAGE0_REUSE.json',dict(compatible=True,stage0=identity(HERE/'evidence/stage0_01/COMPLETED.json'),
        development_ready=identity(HERE/'evidence/development_r2_20260930/READY.json'),no_replay=True,
        paths_unchanged=['pinned pilot solver/model/attack','training','numerics','coarse journal','owned process helpers']))
