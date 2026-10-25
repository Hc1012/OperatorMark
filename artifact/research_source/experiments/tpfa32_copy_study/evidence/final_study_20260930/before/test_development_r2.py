"""One bounded verification set: toy orchestration plus changed SpatialFNO path only."""
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
from core import HERE,Budget,read,write,identity,digest,source_hashes,select_recipe,validate_config
from numerics import runtime,legacy,Observations,controls,fresh
from training import Trajectory,legacy_attack
from dev_state import Journal,MemoryBudget,counts,FrozenSeeds,load_owned
from dev_campaign import seed_plan,visit_record
from dev_process import process_identity,require_dead
from dev_supervise import resume_allowed
from test_contracts import equal,toy,observations


@pytest.fixture(scope='session',autouse=True)
def environment():
    assert os.environ['TPFA_CATEGORY']=='development_r2_verification'
    Budget.verification().check()
    write(Path(os.environ['TPFA_ATTEMPT'])/'RUNTIME.json',runtime())


def memory_budget(**limits):return MemoryBudget(counts(model_updates=10000,inverse_updates=1000,**limits),time.time()+120)


def state(tr):
    return copy.deepcopy(dict(model=tr.model.state_dict(),optimizer=tr.optimizer.state_dict(),
        shuffle=tr.shuffle.get_state(),order=tr.order,offset=tr.offset,updates=tr.updates,
        flags={n:m.training for n,m in tr.model.named_modules()},rng=torch.get_rng_state()))


def test_cooldown_exact_schedule_base_rng_isolation_and_resume(tmp_path):
    b=memory_budget();tr=Trajectory(toy(),observations(),311,b,real=True)
    tr.extend(5);base=state(tr);branch=tr.fork_cooldown(3)
    assert branch.budget is b and branch.data is tr.data
    rates=[];original=branch.optimizer.step
    def step(*a,**k):rates.append(branch.optimizer.param_groups[0]['lr']);return original(*a,**k)
    branch.optimizer.step=step
    branch.extend(8)
    assert rates==[.002*(1-j/3) for j in (1,2,3)] and b.counts['model_updates']==8
    assert equal(base,state(tr))
    other=tr.fork_cooldown(3);other.extend(6);path=tmp_path/'branch.pt';pin=other.save(path)
    resumed=Trajectory(toy(555),observations(),999,b,real=True);resumed.resume(path,pin);resumed.extend(8)
    assert equal(branch.model.state_dict(),resumed.model.state_dict())
    assert equal(branch.optimizer.state_dict(),resumed.optimizer.state_dict())
    assert equal(branch.shuffle.get_state(),resumed.shuffle.get_state()) and branch.offset==resumed.offset
    assert equal(branch.branch_rng,resumed.branch_rng) and resumed.optimizer.param_groups[0]['lr']==0
    assert resumed.cooldown==dict(base=5,steps=3)
    # An untouched equal-step snapshot stays distinct from every cooled candidate.
    assert equal(base['model'],tr.model.state_dict()) and not equal(base['model'],branch.model.state_dict())


def test_spatial_branch_checkpoint_changed_path(tmp_path):
    b=Budget.verification()
    data=Observations(controls(5501,4),np.ones((4,1024),dtype=np.float64),information_calls=4)
    base=Trajectory(fresh(5502),data,5503,b,batch=4,real=True);base.extend(2)
    before=state(base);branch=base.fork_cooldown(2);branch.extend(3)
    path=tmp_path/'spatial_branch.pt';pin=branch.save(path)
    branch.extend(4)
    resumed=Trajectory(fresh(5504),data,999,b,batch=4,real=True);resumed.resume(path,pin);resumed.extend(4)
    assert equal(branch.model.state_dict(),resumed.model.state_dict())
    assert equal(branch.optimizer.state_dict(),resumed.optimizer.state_dict())
    assert equal(branch.shuffle.get_state(),resumed.shuffle.get_state())
    assert equal(before['model'],base.model.state_dict()) and base.updates==2
    write(Path(os.environ['TPFA_ATTEMPT'])/'SPATIAL_CHANGED_PATH.json',dict(passed=True,actual_optimizer_steps=5,
          solver_calls=0,branch_checkpoint=identity(path),model_tensors=len(base.model.state_dict())))


def test_stage0_pins_and_memory_adapter_parity(monkeypatch):
    evidence=read(HERE/'evidence/stage0_01/COMPLETED.json')
    assert not evidence['blockers'] and len(evidence['comparisons'])==2
    for fact in evidence['evidence']:assert digest(fact['path'])==fact['sha256']
    data=observations(256);a,b=toy(),toy();old=legacy()[1];adam=torch.optim.Adam
    class Clock:
        def guard(self):pass
        def elapsed(self):return 0
    x,_,y,_=data.tensors();old.attack(a,x,y,113,Clock(),lambda row:None,0,'fixture')
    budget=memory_budget();stats=legacy_attack(b,data,113,budget,real=True)
    assert stats['updates']==1920 and budget.counts['model_updates']==1920
    assert equal(a.state_dict(),b.state_dict()) and torch.optim.Adam is adam
    blocked=MemoryBudget(counts(model_updates=1),time.time()+30)
    with pytest.raises(RuntimeError):legacy_attack(toy(),data,113,blocked,real=True)
    assert torch.optim.Adam is adam
    write(Path(os.environ['TPFA_ATTEMPT'])/'STAGE0_DECISION.json',dict(compatible=True,
          reused=identity(HERE/'evidence/stage0_01/COMPLETED.json'),evidence_identities_checked=len(evidence['evidence']),
          toy_memory_charging_parity=True,adam_restored_success_and_failure=True,full_training_replay=False,
          limitation='Historical endpoint equality is local retained evidence, not a cross-platform guarantee.'))


def test_selection_full_cases_stops_ties_and_final_projection():
    rows=[dict(owner=name,clean=dict(forward=1.,random_jvp=1.,task=1.),
               replacement=dict(forward=2.,random_jvp=2.,task=2.)) for name in ('pilot0','pilot1','D1')]
    v=visit_record('R-a',1920,rows,None,15360)
    assert len(v['ratios'])==9 and v['endpoint_updates']==2112 and v['cooldown_updates']==192
    assert visit_record('R-a',3840,rows,1.,15360)['stop_reason']=='improvement_below_5_percent'
    with pytest.raises(ValueError):visit_record('R-a',1920,rows[:2],None,15360)
    sel=select_recipe({k:[dict(v,updates=3840),v] for k in ('R-a','R-b')})
    assert sel['primary']=='R-a' and sel['recipes']['R-a']['updates']==1920
    from dev_cost import final_projection
    projected=final_projection(sel)
    assert projected['inverse_model_sets']==80 and projected['equal_step_distinct']
    assert projected['model_updates']==60800+21120
    config=read(HERE/'draft.json');validate_config(config)
    config['selection_updates'].append(30720)
    with pytest.raises(ValueError):validate_config(config)


def test_unsettled_bound_deadline_corrupt_state_and_seed_reload(tmp_path):
    limits=counts(model_updates=10)
    b=MemoryBudget(limits,time.time()+60);j=Journal(tmp_path/'journal',b,0)
    def fail():b.charge('model_updates',2);raise OSError('injected interruption')
    with pytest.raises(OSError):j.block('work',counts(model_updates=3),fail)
    restored=MemoryBudget(limits,b.deadline);r=Journal(j.path,restored,1)
    assert restored.counts['model_updates']==3 and r.uncertainty['model_updates']==3
    def finish():restored.charge('model_updates',2);return 'done'
    assert r.block('work',counts(model_updates=3),finish)=='done'
    assert restored.counts['model_updates']==5 and restored.deadline==b.deadline
    artifact=r.done['work']['artifact'];Path(artifact['path']).write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError,match='identity'):r.block('work',counts(),lambda:None)
    with pytest.raises(TimeoutError):MemoryBudget(limits,time.time()-1)
    seeds=FrozenSeeds(tmp_path/'PRIVATE_SEEDS.json',seed_plan(),fixture=11)
    again=FrozenSeeds(seeds.path,seed_plan(),fixture=11)
    assert seeds.values==again.values and seeds.used==again.used and seeds.events==again.events
    with pytest.raises(ValueError):again.get('unplanned')
    with pytest.raises(RuntimeError,match='alive'):require_dead(process_identity(os.getpid()))


def fixture_root(path,pause=None,seconds=100):
    path.mkdir();now=time.time()
    write(path/'AUTHORITY.json',dict(started_unix=now,deadline_unix=min(now+seconds,float(os.environ['TPFA_DEADLINE'])-10),
          closure_deadline_unix=min(now+seconds+5,float(os.environ['TPFA_DEADLINE'])-5),
          limits=counts(model_updates=1000,inverse_updates=100),evidence_bytes=128*1024**2))
    config=read(HERE/'draft.json');config.update(fixture_only=True,pause_key=pause)
    write(path/'CONFIG.json',config)
    return path


def start_fixture(root,mode):
    stream=(root/(mode+'_supervisor.txt')).open('wb')
    child=subprocess.Popen([sys.executable,'-B',str(HERE/'dev_supervise.py'),mode,'--output',str(root)],stdout=stream,stderr=stream)
    return child,stream


def wait_end(child,stream,seconds=100):
    try:return child.wait(timeout=seconds)
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)
        stream.close()


def test_one_tiny_end_to_end_control_crash_branch_resume(tmp_path):
    # Reserve conservative full child inverse bounds with the sole verification writer.
    budget=Budget.verification()
    previous=HERE/'evidence/development_r2_20260930/verification/20260930T140025_0e12fbd7'
    saved=previous/'test_tmp/test_one_tiny_end_to_end_contr0/control'
    if (saved/'COMPLETED.json').exists():
        # Reuse the successful control; only process closure and this test changed.
        pins=read(previous/'started.json')['sources'];current=source_hashes()
        exempt=('dev_process.py','dev_supervise.py','test_development_r2.py')
        assert all(v==current[k] for k,v in pins.items() if Path(k).name not in exempt)
        control=saved
    else:
        budget.charge('inverse_updates',21)
        control=fixture_root(tmp_path/'control');child,stream=start_fixture(control,'fixture-start')
        assert wait_end(child,stream)==0,(control/'fixture-start_supervisor.txt').read_text()
    budget.charge('inverse_updates',21)
    interrupted=fixture_root(tmp_path/'interrupted','R-a/20/pilot0/cool/22')
    child,stream=start_fixture(interrupted,'fixture-start')
    try:
        until=time.monotonic()+80
        while not (interrupted/'FIXTURE_PAUSED.json').exists():
            if child.poll() is not None:pytest.fail((interrupted/'fixture-start_supervisor.txt').read_text())
            if time.monotonic()>until:raise TimeoutError('fixture did not reach branch checkpoint')
            time.sleep(.05)
        paused=read(interrupted/'FIXTURE_PAUSED.json')
        assert paused['known_complete']['key']=='R-a/20/pilot0/cool/21'
        with pytest.raises((FileNotFoundError,RuntimeError)):resume_allowed(interrupted)
        write(interrupted/'INTERRUPT_REQUEST_0.json',dict(reason='external operator interruption'))
        assert wait_end(child,stream)==1
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)
        stream.close()
    assert not (interrupted/'COMPLETED.json').exists()
    prior=read(interrupted/'attempt_0/result.json')
    assert prior['owned_worker_ended'] and prior['infrastructure'] and (interrupted/'attempt_0/INCOMPLETE.json').exists()
    deadline=read(interrupted/'AUTHORITY.json')['deadline_unix']
    budget.charge('inverse_updates',21)
    child,stream=start_fixture(interrupted,'fixture-resume')
    assert wait_end(child,stream)==0,(interrupted/'fixture-resume_supervisor.txt').read_text()
    assert read(interrupted/'AUTHORITY.json')['deadline_unix']==deadline
    with pytest.raises(RuntimeError,match='continuation'):resume_allowed(interrupted)
    a,b=read(control/'SELECTION.json'),read(interrupted/'SELECTION.json')
    assert a['primary']==b['primary'] and a['baselines']==b['baselines'] and a['copied_metrics']==b['copied_metrics']
    assert b['recovery_uncertainty']['model_updates']==1
    for recipe in ('R-a','R-b'):
        assert len(a['histories'][recipe])==len(b['histories'][recipe])==2
        for left,right in zip(a['histories'][recipe],b['histories'][recipe]):
            assert left['ratios']==right['ratios'] and left['score']==right['score']
            for ca,cb in zip(left['cases'],right['cases']):
                assert ca['replacement']==cb['replacement']
                for label in ('checkpoint','base_checkpoint','equal_step_checkpoint'):
                    sa,sb=load_owned(ca[label]),load_owned(cb[label])
                    for field in ('model','optimizer','shuffle_rng','torch_rng','order','offset','updates','epochs','data','cooldown','training_flags'):
                        assert equal(sa[field],sb[field]),(recipe,label,field)
    assert digest(control/'PRIVATE_SEEDS.json')==digest(interrupted/'PRIVATE_SEEDS.json')
    # A recorded scientific failure never becomes an infrastructure continuation.
    refused=fixture_root(tmp_path/'refused');(refused/'attempt_0').mkdir()
    write(refused/'attempt_0/result.json',dict(owned_worker_ended=True,infrastructure=False))
    with pytest.raises(RuntimeError,match='infrastructure'):resume_allowed(refused)
    write(Path(os.environ['TPFA_ATTEMPT'])/'TINY_END_TO_END.json',dict(passed=True,control=str(control),interrupted=str(interrupted),
        equal_tensors_optimizer_rng_cursor_metrics_selection=True,unchanged_seed_sha256=digest(control/'PRIVATE_SEEDS.json'),
        verified_complete_branch_checkpoint=paused['known_complete']['key'],inverse_updates_conservatively_reserved=63,
        actual_fixture_inverse_updates=42,solver_calls=0,spatial_updates=0))


def test_independent_deadline_closes_owned_worker(tmp_path):
    root=fixture_root(tmp_path/'deadline',seconds=2.)
    child,stream=start_fixture(root,'fixture-start')
    assert wait_end(child,stream,10)==1
    result=read(root/'attempt_0/result.json')
    assert result['stop']=='work deadline' and result['owned_worker_ended'] and not result['infrastructure']
    require_dead(result['worker'])
