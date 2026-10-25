"""Revision 2 development orchestration; every reusable unit has a complete identity."""
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from core import read, write, identity, cohort_gate, selection_score, select_recipe, positive
from campaign import Campaign
from numerics import (Oracle,Splits,generate,task_data,allocation,fresh,legacy,evaluate)
from training import Trajectory,legacy_attack
from dev_state import FrozenSeeds,Journal,MemoryBudget,counts,load_owned


def seed_plan():
    plan={}
    for label in ('D1/owner','D1/embedding','D1_clean_gate','development_evaluation'):
        for suffix in ('controls','directions'):plan[label+'/'+suffix]=(0,)
    for name in ('pilot0','pilot1','D1'):
        for label in ('initialization','owner_shuffle','embed_shuffle','attack_controls','Rb_directions',
                      'shared_attack_shuffle','Rb_shuffle','R-a/initialization','R-b/initialization'):
            plan[name+'/'+label]=(0,)
    plan['D1/private_key']=(0,10000)
    for label in ('targets','starts'):plan['development_tasks/'+label]=(0,)
    return plan


def visit_record(recipe,length,rows,previous,cap):
    if [r['owner'] for r in rows]!=['pilot0','pilot1','D1']:raise ValueError('all three ordered development cases required')
    score=selection_score(rows)
    ratios=[]
    for row in rows:
        for metric in ('forward','random_jvp','task'):
            ratios.append(dict(owner=row['owner'],metric=metric,ratio=row['replacement'][metric]/row['clean'][metric]))
    improvement=None if previous is None else (previous-score)/previous
    stop=('improvement_below_5_percent' if improvement is not None and improvement<.05 else
          'base_cap' if length==cap else None)
    return dict(updates=length,base_updates=length,cooldown_updates=length//10,
                endpoint_updates=length+length//10,score=score,ratios=ratios,cases=rows,
                fractional_improvement=improvement,stop_reason=stop,
                tie_rule='lowest score, then fewer base updates',recipe=recipe)


class Development(Campaign):
    fixture=False
    grid=(1920,3840,7680,15360)
    owner_steps=1920;mark_steps=320;attack_steps=1920;chunk=128
    rate_limits={'R-a':.2476903,'R-b':.3835388}

    def __init__(self,path,authority,attempt):
        self.path=Path(path);self.authority=authority;self.config=read(self.path/'CONFIG.json')
        self.mode='development';self.cases=[];self.facts=[];self.splits=Splits()
        self.budget=MemoryBudget(authority['limits'],authority['deadline_unix'])
        self.journal=Journal(self.path,self.budget,attempt,authority['evidence_bytes'])
        self.attempt=attempt;self.seeds=FrozenSeeds(self.path/'PRIVATE_SEEDS.json',seed_plan(),
                                                 fixture=4400001 if self.fixture else None)
        self.oracle=Oracle(self.solve,self.budget,real=not self.fixture)
        self.baselines={};self.attack_metrics={};self.equal_step={};self.rate_results={}
        # Existing scientific gate semantics/config are not changed for a tiny fixture.
        if not self.fixture and tuple(self.config['selection_updates'])!=self.grid:raise ValueError('development grid drift')

    def solve(self,row):return legacy()[0].solve(row)
    def model(self,seed):return fresh(seed)

    def raw(self,row):
        data=(json.dumps(row,allow_nan=False)+'\n').encode()
        if self.journal.bytes+len(data)>self.journal.limit:raise RuntimeError('campaign evidence storage limit')
        with (self.path/'PRIVATE_measurements.jsonl').open('ab') as out:out.write(data);out.flush()
        self.journal.bytes+=len(data)

    def legacy_payload(self):
        self.legacy_cases()
        return self.cases,self.splits.seen

    def data(self,label,n,k,seeds=None):
        def build():
            value=generate(self.oracle,self.seeds.get(label+'/controls'),self.seeds.get(label+'/directions'),n,k,label,self.splits)
            return value,self.splits.seen
        value,seen=self.journal.block('data/'+label,counts(solver_attempts=n*(1+2*k)),build)
        self.splits.seen=seen
        return value

    def tasks(self):
        def build():return task_data(self.oracle,self.seeds,64,'development_tasks',self.splits),self.splits.seen
        value,seen=self.journal.block('data/tasks',counts(solver_attempts=128),build)
        self.splits.seen=seen
        return value

    def attack_data(self,case):
        name=case['id']
        def build():
            value=allocation(self.oracle,self.seeds.get(name+'/attack_controls'),self.seeds.get(name+'/Rb_directions'),self.splits)
            return value,self.splits.seen
        value,seen=self.journal.block('data/'+name+'/attack',counts(solver_attempts=426),build)
        self.splits.seen=seen
        shared,rb=value
        if not self.fixture and (shared.information_calls!=256 or rb.information_calls!=255 or
                shared.x64[:85].tobytes()!=rb.x64.tobytes() or shared.y64[:85].tobytes()!=rb.y64.tobytes()):
            raise ValueError('attacker information/subset contract changed')
        return shared,rb

    def trajectory(self,model,data,shuffle,recipe='forward',reference=None,key=None):
        return Trajectory(model,data,shuffle,self.budget,recipe=recipe,batch=32 if recipe=='forward' else 16,
                          lr=.0003 if recipe=='mark' else .002,real=not self.fixture,reference=reference,key=key)

    def train(self,label,trainer,target):
        fact=None
        for endpoint in range(trainer.updates+self.chunk,target+self.chunk,self.chunk):
            endpoint=min(endpoint,target)
            remaining=endpoint-trainer.updates
            key=label+'/'+str(endpoint)
            def work(endpoint=endpoint,key=key):
                self.fixture_pause(key)
                trainer.extend(endpoint)
                filename=self.path/(key.replace('/','_')+'_attempt'+str(self.attempt)+'.pt')
                trainer.save(filename);self.journal.bytes+=filename.stat().st_size
                return identity(filename)
            fact=self.journal.block(key,counts(model_updates=remaining),work)
            trainer.resume(fact['path'],fact['sha256'])
        return fact

    def fixture_pause(self,key):pass

    def gate(self,model,data):
        metrics,raw=legacy()[1].evaluate(copy.deepcopy(model).eval(),data.tensors())
        self.raw(dict(type='D1_clean_gate',metrics=metrics,
                      samples={k:v.tolist() if torch.is_tensor(v) else v for k,v in raw.items()}))
        return metrics

    def copied(self,case,arm,data):
        model=copy.deepcopy(case['models']['M' if arm=='A3' else 'O'])
        stats=legacy_attack(model,data,self.seeds.get(case['id']+'/shared_attack_shuffle'),self.budget,
                            lambda row:self.journal.event(dict(type='attack_progress',owner=case['id'],arm=arm,record=row)),real=True)
        return model,stats

    def measure_model(self,case,arm,model,data,tasks):
        metrics,raw=evaluate(model,data,tasks,self.oracle,self.budget,case['id']+'/'+arm,
                             fixed_inverse_batches=True,progress=self.raw)
        self.raw(dict(type='evaluation',owner=case['id'],arm=arm,metrics=metrics,raw=raw))
        return metrics

    def measure(self,case,arm,model,data,tasks,label):
        return self.journal.block('evaluate/'+label+'/'+case['id']+'/'+arm,
            counts(solver_attempts=64,inverse_updates=480),lambda:self.measure_model(case,arm,model,data,tasks))

    def run_development(self):
        self.cases,seen=self.journal.block('legacy_inputs',counts(),self.legacy_payload);self.splits.seen=seen
        owner=self.data('D1/owner',384,1)
        d1=dict(id='D1',models={},costs={})
        trainer=self.trajectory(self.model(self.seeds.get('D1/initialization')),owner,
                               self.seeds.get('D1/owner_shuffle'),'derivative')
        self.train('D1/owner',trainer,self.owner_steps)
        d1['models']['O']=trainer.model;d1['costs']['O']=trainer.record();self.cases.append(d1)
        gate=self.data('D1_clean_gate',64,2)
        gate_metrics=self.journal.block('D1_clean_gate',counts(),lambda:self.gate(d1['models']['O'],gate))
        self.journal.json('D1_CLEAN_GATE.json',gate_metrics)
        if not cohort_gate([gate_metrics],1):raise ValueError('D1 clean gate failed; no marking or selection')
        embedding=self.data('D1/embedding',128,1)
        key=self.key();d1['key']=key
        marked=self.trajectory(copy.deepcopy(d1['models']['O']),embedding,self.seeds.get('D1/embed_shuffle'),
                               'mark',d1['models']['O'],key)
        self.train('D1/mark',marked,self.mark_steps)
        d1['models']['M']=marked.model;d1['scale']=marked.score_scale;d1['costs']['M']=marked.record()
        data=self.data('development_evaluation',128,2);tasks=self.tasks()
        trajectories={'R-a':[],'R-b':[]}
        for case in self.cases:
            shared,rb=self.attack_data(case)
            for arm in ('A3','A0'):
                model,stats=self.journal.block('attack/'+case['id']+'/'+arm,counts(model_updates=self.attack_steps),
                                             lambda c=case,a=arm:self.copied(c,a,shared))
                case['models'][arm]=model;case['costs'][arm]=stats
            self.baselines[case['id']]=self.measure(case,'O',case['models']['O'],data,tasks,'baseline')
            self.attack_metrics[case['id']]={arm:self.measure(case,arm,case['models'][arm],data,tasks,'baseline') for arm in ('A0','A3')}
            for recipe,observed in (('R-a',shared),('R-b',rb)):
                shuffle=self.seeds.get(case['id']+'/shared_attack_shuffle' if recipe=='R-a' else case['id']+'/Rb_shuffle')
                trajectories[recipe].append(self.trajectory(self.model(self.seeds.get(case['id']+'/'+recipe+'/initialization')),
                                            observed,shuffle,'forward' if recipe=='R-a' else 'derivative'))
        histories={'R-a':[],'R-b':[]}
        for recipe in histories:
            previous=None
            for length in self.grid:
                # First base length across all cases is rate-gated before its cooldown work.
                bases=[]
                for case,tr in zip(self.cases,trajectories[recipe]):
                    fact=self.train(recipe+'/'+str(length)+'/'+case['id']+'/base',tr,length)
                    bases.append(fact)
                    if length==self.grid[0]:self.equal_step[recipe+'/'+case['id']]=fact
                if length==self.grid[0]:
                    rate=sum(t.wall_seconds for t in trajectories[recipe])/(3*length)
                    self.rate_results[recipe]=dict(seconds_per_update=rate,limit=self.rate_limits[recipe],
                                                   included='active uncooled extend wall only',passed=rate<=self.rate_limits[recipe])
                    self.journal.json(recipe+'_RATE.json',self.rate_results[recipe])
                    if not self.rate_results[recipe]['passed']:raise RuntimeError('slow-rate stop: '+recipe)
                rows=[]
                for case,tr,base in zip(self.cases,trajectories[recipe],bases):
                    with torch.random.fork_rng(devices=[]):
                        branch=tr.fork_cooldown(length//10)
                        fact=self.train(recipe+'/'+str(length)+'/'+case['id']+'/cool',branch,length+length//10)
                        metrics=self.measure(case,recipe,branch.model,data,tasks,'cooled/'+str(length))
                    attack=self.attack_metrics[case['id']]['A3']
                    if any(not positive(attack.get(k)) for k in ('random_jvp','task')):raise ValueError('invalid descriptive attack denominator')
                    rows.append(dict(owner=case['id'],clean=self.baselines[case['id']],replacement=metrics,
                        replacement_over_A3={k:metrics[k]/attack[k] for k in ('random_jvp','task')},
                        base_checkpoint=base,checkpoint=fact,equal_step_checkpoint=self.equal_step[recipe+'/'+case['id']],
                        costs=branch.record(),base_active_seconds=tr.wall_seconds,
                        cooldown_active_seconds=branch.wall_seconds-tr.wall_seconds))
                visit=visit_record(recipe,length,rows,previous,self.grid[-1]);histories[recipe].append(visit)
                self.journal.json(recipe+'_'+str(length)+'_selection.json',visit)
                if visit['stop_reason']:break
                previous=visit['score']
        selected=select_recipe(histories)
        result=dict(schema='tpfa32_development_selection.v2',**selected,histories=histories,
            equal_step_checkpoints=self.equal_step,baselines=self.baselines,copied_metrics=self.attack_metrics,
            rates=self.rate_results,counts=self.budget.counts,recovery_uncertainty=self.journal.uncertainty,
            known_completed_operation_counts={k:self.budget.counts[k]-self.journal.uncertainty[k] for k in self.budget.counts},
            total_search_model_updates=sum(3*(h[-1]['updates']+sum(v['cooldown_updates'] for v in h)) for h in histories.values()),
            provisional_final_rules=True,not_convergence_proof=True,fixture=self.fixture)
        self.journal.json('SELECTION.json',result)
        self.journal.json('DEVELOPMENT_SUMMARY.json',result)
        if not self.fixture:
            from dev_cost import final_projection
            self.journal.json('FINAL_COST_PROJECTION.json',final_projection(selected))
        return result

    def key(self):return legacy()[1].b3_key(self.seeds.get('D1/private_key',offsets=(0,10000)))

    def execute(self):
        try:
            value=self.run_development()
            self.journal.json('COMPLETED.json',dict(passed=True,selection=identity(self.path/'SELECTION.json'),
                              counts=self.budget.counts,fixture=self.fixture))
            return value
        except BaseException as exc:
            infrastructure=isinstance(exc,OSError) and getattr(exc,'winerror',None) in (5,32,33)
            failure=dict(type=type(exc).__name__,message=str(exc),infrastructure=infrastructure,
                counts=self.budget.counts,completed_blocks=list(self.journal.done),
                completed_case_metrics=self.baselines,copied_metrics=self.attack_metrics,
                last_complete=self.journal.last_complete,selection_complete=False)
            write(self.path/('INCOMPLETE_attempt'+str(self.attempt)+'.json'),failure)
            self.journal.event(dict(type='stopped',**failure))
            raise
