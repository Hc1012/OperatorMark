"""Fixed final cohort, using the proven coarse journal and continuous trajectories."""
import copy
import json
import os
from pathlib import Path
import secrets
import time
import numpy as np
import torch
from core import read, write, identity, digest, cohort_gate, threshold, detection, utility
from dev_campaign import Development
from dev_state import Journal, MemoryBudget, counts, load_owned
from numerics import (Oracle, Splits, controls, forward, generate, task_data, keyed_truth,
                      evaluate, legacy, b2)
from outcomes import qualification_packet

NAMES = [f'owner{i}' for i in range(10)]
WRONG_OFFSETS = tuple(range(16)) + tuple(10000+i for i in range(16))


def seed_plan(final=False):
    plan = {}
    for label in (('final',) if final else ('clean_gate', 'calibration')):
        for suffix in ('controls', 'directions'):plan[label+'/'+suffix]=(0,)
    for suffix in ('targets', 'starts'):
        plan[('final_tasks' if final else 'calibration_tasks')+'/'+suffix]=(0,)
    for name in NAMES:
        if final:
            plan[name+'/wrong']=WRONG_OFFSETS
        else:
            for label in ('owner','embedding'):
                for suffix in ('controls','directions'):plan[name+'/'+label+'/'+suffix]=(0,)
            for label in ('initialization','owner_shuffle','embed_shuffle','attack_controls',
                          'shared_attack_shuffle','R-a/initialization'):plan[name+'/'+label]=(0,)
            plan[name+'/private_key']=(0,10000)
            plan[name+'/calibration_wrong']=WRONG_OFFSETS
    return plan


class StageSeeds:
    """Immutable complete labelled manifest, persisted before use; no redraw on resume."""
    def __init__(self,path,stage,plan,forbidden=(),fixture=None):
        self.path=Path(path);pin=self.path.with_name(self.path.stem+'_IDENTITY.json')
        self.used=set(forbidden);self.plan=plan;self.stage=stage
        if not self.path.exists():
            if pin.exists():raise ValueError('seed manifest missing behind existing pin')
            values={};events=[]
            for i,(label,offsets) in enumerate(plan.items()):
                value=secrets.randbits(63) if fixture is None else fixture+100000*i
                expanded={value+o for o in offsets};collision=bool(expanded & self.used)
                events.append(dict(label=stage+'/'+label,offsets=list(offsets),collision=collision))
                values[stage+'/'+label]=value;self.used.update(expanded)
            write(self.path,dict(schema='private_seeds.v1',stage=stage,fixture=fixture is not None,
                  values=values,events=events,provenance='OS random draws; no proof of unseen seeds'))
            write(pin,identity(self.path))
        if not pin.exists() or identity(self.path)!=read(pin):raise ValueError('seed identity missing/changed')
        record=read(self.path);self.values=record['values'];self.events=record['events']
        if record['stage']!=stage or record['fixture']!=(fixture is not None):raise ValueError('seed stage/fixture mismatch')
        if set(self.values)!={stage+'/'+k for k in plan} or len(self.events)!=len(plan):raise ValueError('seed label set mismatch')
        self.used=set(forbidden)
        for event,(label,offsets) in zip(self.events,plan.items()):
            if event!=dict(label=stage+'/'+label,offsets=list(offsets),collision=False):raise ValueError('seed event contract/collision')
            expanded={self.values[stage+'/'+label]+o for o in offsets}
            if expanded & self.used:raise ValueError('seed collision on reload')
            self.used.update(expanded)

    def get(self,label,offsets=(0,)):
        if label not in self.plan or tuple(offsets)!=tuple(self.plan[label]):raise ValueError('unplanned seed/offset')
        return self.values[self.stage+'/'+label]


class FinalCampaign(Development):
    base_steps=15360;cool_steps=1536;equal_steps=1920
    owner_n=384;mark_n=128;gate_n=64;eval_n=128;tasks_n=64;attack_n=256
    inverse_bound=480;optimized_bound=64;rate_limit=.1427789

    def __init__(self,path,authority,attempt):
        self.path=Path(path);self.authority=authority;self.config=read(self.path/'CONFIG.json')
        self.mode='final';self.cases=[];self.facts=[];self.splits=Splits();self.attempt=attempt
        self.budget=MemoryBudget(authority['limits'],authority['deadline_unix'])
        # Reserve metadata/stdout/seeds/terminal-report space outside journal writes.
        self.journal=Journal(self.path,self.budget,attempt,authority['evidence_bytes']-16*1024**2)
        self.oracle=Oracle(self.solve,self.budget,real=not self.fixture)
        self.oracle_totals={k:0 for k in self.oracle.record()}
        self.runtime=authority.get('runtime',{'fixture':True});self.rows=[];self.final_started=False
        forbidden=set()
        if not self.fixture:
            for fact in authority['development_inputs'].values():
                if identity(fact['path'])!=fact:raise ValueError('preserved development input changed')
            prior=read(authority['development_inputs']['seeds']['path'])
            for event in prior['events']:
                forbidden.update(prior['values'][event['label']]+o for o in event['offsets'])
            # Register spent development/pilot query rows, without loading their models.
            _,seen=load_owned(authority['development_inputs']['splits'])
            self.splits.seen=seen
        self.seeds=StageSeeds(self.path/'PRIVATE_SEEDS.json','final/training_calibration',seed_plan(),
                              forbidden,fixture=10001 if self.fixture else None)
        self.journal.json('PLANNED_OWNERS.json',dict(ids=NAMES,no_replacement=True,fixture=self.fixture))

    def unit(self,key,bound,fn):
        def work():
            before=self.oracle.record();value=fn();after=self.oracle.record()
            return dict(value=value,oracle={k:after[k]-before[k] for k in before})
        item=self.journal.block(key,bound,work)
        for k,v in item['oracle'].items():self.oracle_totals[k]+=v
        return item['value']

    def data(self,label,n,k,seeds=None):
        seeds=seeds or self.seeds
        def work():
            value=generate(self.oracle,seeds.get(label+'/controls'),seeds.get(label+'/directions'),n,k,label,self.splits)
            return value,self.splits.seen
        value,seen=self.unit('data/'+label,counts(solver_attempts=0 if self.fixture else n*(1+2*k)),work)
        self.splits.seen=seen;return value

    def tasks(self,label,seeds):
        def work():return task_data(self.oracle,seeds,self.tasks_n,label,self.splits),self.splits.seen
        value,seen=self.unit('data/'+label,counts(solver_attempts=0 if self.fixture else 2*self.tasks_n),work)
        self.splits.seen=seen;return value

    def key_for(self,name):return legacy()[1].b3_key(self.seeds.get(name+'/private_key',(0,10000)))

    def gate(self,model,data):
        metrics,raw=legacy()[1].evaluate(copy.deepcopy(model).eval(),data.tensors())
        self.raw(dict(type='clean_cohort_gate',metrics=metrics,
                 samples={k:v.tolist() if torch.is_tensor(v) else v for k,v in raw.items()}))
        return metrics

    def wrong_keys(self,name,seeds,final):
        base=seeds.get(name+('/wrong' if final else '/calibration_wrong'),WRONG_OFFSETS)
        return [legacy()[1].b3_key(base+i) for i in range(16)]

    def truth(self,case,data,stage):
        def work():return keyed_truth(self.oracle,data.x64,case['key'],stage+'_key/'+case['id'],self.splits),self.splits.seen
        value,seen=self.unit('truth/'+stage+'/'+case['id'],counts(solver_attempts=0 if self.fixture else 2*self.eval_n),work)
        self.splits.seen=seen;return value

    def detector(self,model,reference,data,key,scale):
        return legacy()[1].b3_score(model,reference,data.tensors()[0],key,scale)

    def measure_value(self,case,arm,data,tasks,stage,truth):
        metrics,raw=evaluate(case['models'][arm],data,tasks,self.oracle,self.budget,
             stage+'/'+case['id']+'/'+arm,case['key'],truth,fixed_inverse_batches=True,progress=self.raw)
        # Preserve every Python-float value losslessly as float64, avoiding retained
        # nested Python-float arrays. Existing detector/B2 arithmetic is unchanged.
        for key in ('prediction','jprediction','key_prediction'):
            if key in raw:raw[key]=np.asarray(raw[key],dtype=np.float64)
        return metrics,raw

    def measurement(self,case,arm,data,tasks,stage,truth):
        return self.unit('evaluate/'+stage+'/'+case['id']+'/'+arm,
            counts(solver_attempts=0 if self.fixture else self.optimized_bound,inverse_updates=self.inverse_bound),
            lambda:self.measure_value(case,arm,data,tasks,stage,truth))

    def prepare_replacement(self,case):
        name=case['id']
        def work():
            data=forward(self.oracle,controls(self.seeds.get(name+'/attack_controls'),self.attack_n),name+'/attack',self.splits)
            return data,self.splits.seen
        shared,seen=self.unit('data/'+name+'/attack',counts(solver_attempts=0 if self.fixture else self.attack_n),work)
        self.splits.seen=seen
        for arm in ('A3','A0'):
            model,stats=self.unit('attack/'+name+'/'+arm,counts(model_updates=self.attack_steps),lambda a=arm:self.copied(case,a,shared))
            case['models'][arm]=model;case['costs'][arm]=stats
        tr=self.trajectory(self.model(self.seeds.get(name+'/R-a/initialization')),shared,
                            self.seeds.get(name+'/shared_attack_shuffle'))
        fact=self.train(name+'/R-a/base',tr,self.equal_steps)
        case['models']['R-a@1920']=copy.deepcopy(tr.model);case['costs']['R-a@1920']=tr.record()
        case['checkpoints']['equal_step']=fact
        return tr

    def calibration(self,case,data,tasks):
        name=case['id'];truth=self.truth(case,data,'calibration')
        clean,_=self.measurement(case,'O',data,tasks,'calibration',truth)
        marked,_=self.measurement(case,'M',data,tasks,'calibration',truth)
        def work():
            score=lambda model,key:self.detector(model,case['models']['O'],data,key,case['scale'])
            wrong=[score(case['models']['M'],k) for k in self.wrong_keys(name,self.seeds,False)]
            others=[score(c['models']['O'],case['key']) for c in self.cases if c['id']!=name]
            tau=threshold(others,wrong,10)
            if score(case['models']['O'],case['key'])!=0:raise ValueError('self-clean score must be zero')
            detected=detection(score(case['models']['M'],case['key']),wrong,tau);gates=utility(marked,clean)
            return dict(owner=name,qualified=detected['detected'] and gates['passed'],clean_gate_passed=True,
                clean=clean,marked=marked,detection=detected,utility=gates,other_clean_true_scores=others,embedding_scale=case['scale'])
        row=self.unit('calibration/'+name,counts(),work)
        case['threshold']=row['detection']['threshold'];case['qualified']=row['qualified'];return row

    def run_final(self):
        for name in NAMES:
            data=self.data(name+'/owner',self.owner_n,1)
            tr=self.trajectory(self.model(self.seeds.get(name+'/initialization')),data,self.seeds.get(name+'/owner_shuffle'),'derivative')
            fact=self.train(name+'/owner',tr,self.owner_steps)
            self.cases.append(dict(id=name,models={'O':tr.model},costs={'O':tr.record()},checkpoints={'O':fact}))
        gate=self.data('clean_gate',self.gate_n,2)
        gates=[self.unit('clean_gate/'+c['id'],counts(),lambda c=c:self.gate(c['models']['O'],gate)) for c in self.cases]
        passed=cohort_gate(gates,10)
        self.journal.json('CLEAN_COHORT.json',dict(ids=NAMES,metrics=gates,passed=passed))
        if not passed:raise ValueError('clean cohort failed; no marking or substitution')
        for case in self.cases:
            name=case['id'];data=self.data(name+'/embedding',self.mark_n,1);case['key']=self.key_for(name)
            tr=self.trajectory(copy.deepcopy(case['models']['O']),data,self.seeds.get(name+'/embed_shuffle'),
                               'mark',case['models']['O'],case['key'])
            fact=self.train(name+'/mark',tr,self.mark_steps)
            case['models']['M']=tr.model;case['costs']['M']=tr.record();case['scale']=tr.score_scale
            case['checkpoints']['M']=fact
        data=self.data('calibration',self.eval_n,2);tasks=self.tasks('calibration_tasks',self.seeds)
        qualified=[self.calibration(c,data,tasks) for c in self.cases]
        self.journal.json('QUALIFICATION.json',dict(schema='tpfa32_qualification.v1',planned_owner_ids=NAMES,
                    owners=qualified,N_q=sum(r['qualified'] for r in qualified),locked_before_final=True))
        first=[self.prepare_replacement(c) for c in self.cases[:3]]
        rate=sum(tr.wall_seconds for tr in first)/(3*self.equal_steps)
        self.journal.json('REPLACEMENT_RATE.json',dict(seconds_per_update=rate,limit=self.rate_limit,passed=rate<=self.rate_limit,
             included='first three active raw base trajectories; excludes setup/checkpoint/cooldown/evaluation'))
        if rate>self.rate_limit:raise ValueError('slow-machine stop; no retry or tuning')
        for i,case in enumerate(self.cases):
            tr=first[i] if i<3 else self.prepare_replacement(case)
            fact=self.train(case['id']+'/R-a/base',tr,self.base_steps);case['checkpoints']['base']=fact
            branch=tr.fork_cooldown(self.cool_steps)
            fact=self.train(case['id']+'/R-a/cool',branch,self.base_steps+self.cool_steps)
            case['models']['R-a']=branch.model;case['models']['Rstar']=branch.model
            case['costs']['R-a']=branch.record();case['checkpoints']['selected']=fact
        del first,tr,branch,data,tasks
        if not self.fixture:
            from final_supervise import validate_run
            validate_run(self.path)
        # Every training/calibration artifact, private input and trajectory is bound.
        closure=dict(all_training_completed=True,models={c['id']:c['checkpoints'] for c in self.cases},
            blocks=[r['artifact'] for r in self.journal.done.values() if not r['key'].startswith(('final/','evaluate/final','truth/final','data/final'))],
            qualification=identity(self.path/'QUALIFICATION.json'),seeds=identity(self.seeds.path),
            config=identity(self.path/'CONFIG.json'),authority=identity(self.path/'AUTHORITY.json'))
        self.journal.json('TRAINING_CALIBRATION_CLOSED.json',closure)
        for fact in closure['blocks']+[closure[k] for k in ('qualification','seeds','config','authority')]:
            if identity(fact['path'])!=fact:raise ValueError('training/calibration closure changed')
        self.qualification_identity=closure['qualification']
        self.journal.json('FINAL_GENERATION_BOUNDARY.json',dict(closed=identity(self.path/'TRAINING_CALIBRATION_CLOSED.json'),
                           rule=self.authority.get('rule_identity'),no_claim_of_unseen_seeds=True))
        if not (self.path/'FIRST_FINAL_GENERATION.json').exists():
            self.journal.json('FIRST_FINAL_GENERATION.json',dict(started_unix=time.time(),attempt=self.attempt,
                                 boundary=identity(self.path/'FINAL_GENERATION_BOUNDARY.json')))
        final_seeds=StageSeeds(self.path/'PRIVATE_FINAL_SEEDS.json','final/evaluation',seed_plan(True),
                              self.seeds.used,fixture=100000001 if self.fixture else None)
        self.journal.json('FINAL_INPUT_IDENTITY.json',dict(seeds=identity(final_seeds.path),
                    generation=identity(self.path/'FIRST_FINAL_GENERATION.json'),boundary=identity(self.path/'FINAL_GENERATION_BOUNDARY.json')))
        self.fixture_pause('after_final_seeds')
        self.final_started=True
        data=self.data('final',self.eval_n,2,final_seeds);tasks=self.tasks('final_tasks',final_seeds)
        return self.finish_evaluation(data,tasks,final_seeds)

    def finish_evaluation(self,data,tasks,seeds):
        result=dict(schema='tpfa32_results.v2',fixture=self.fixture,planned_owner_ids=NAMES,owners=[],b2={},null_pairs=[],secondary={},
            locked_qualification=qualification_packet(self.path/'QUALIFICATION.json',self.qualification_identity),
            rule_status='approved final milestone 20260930',analysis_rule='t10_adv975_eq95',costs={})
        self.journal.json('FINAL_REPORT_INPUT.json',result)
        fingerprints={};wrongs={}
        for case in self.cases:
            name=case['id'];truth=self.truth(case,data,'final');metrics={};raws={};detectors={}
            wrongs[name]=self.wrong_keys(name,seeds,True)
            for arm in ('O','M','A0','A3','R-a@1920','R-a'):
                metrics[arm],raw=self.measurement(case,arm,data,tasks,'final',truth)
                # Only the random-JVP arrays are needed for subsequent B2 readbacks.
                raws[arm]=raw['jprediction']
                if arm in ('M','A3'):
                    def work(a=arm):
                        score=lambda key:self.detector(case['models'][a],case['models']['O'],data,key,case['scale'])
                        return detection(score(case['key']),[score(k) for k in wrongs[name]],case['threshold'])
                    detectors[arm]=self.unit('final/detection/'+name+'/'+arm,counts(),work)
                self.journal.json('FINAL_'+name+'_'+arm.replace('@','_')+'.json',dict(owner=name,arm=arm,metrics=metrics[arm],detector=detectors.get(arm)))
            metrics['Rstar']=metrics['R-a'];raws['Rstar']=raws['R-a'];fingerprints[name]=raws
            row=dict(id=name,qualified=case['qualified'],metrics=metrics,costs=case['costs'],checkpoints=case['checkpoints'],detectors=detectors,
                intact=dict(detected=detectors['M']['detected'],utility=utility(metrics['M'],metrics['O'])['passed']),
                attacked=dict(detected=detectors['A3']['detected'],utility=utility(metrics['A3'],metrics['O'])['passed']))
            self.journal.json('OWNER_'+name+'.json',row);result['owners'].append(row);self.rows=list(result['owners'])
        for case in self.cases:
            name=case['id'];reference=fingerprints[name]['O']
            distances=dict(descendant=b2(fingerprints[name]['A0'],reference)['mean'],clean=[],replacements=[])
            for other in self.cases:
                for arm in ('O','Rstar'):
                    if arm=='O' and other['id']==name:continue
                    def work(other=other,arm=arm):
                        model=other['models'][arm]
                        score=lambda key:self.detector(model,case['models']['O'],data,key,case['scale'])
                        return dict(reference=name,model=arm+'/'+other['id'],
                            b2=b2(fingerprints[other['id']][arm],reference),
                            **detection(score(case['key']),[score(k) for k in wrongs[name]],case['threshold']))
                    pair=self.unit('final/null/'+name+'/'+arm+'/'+other['id'],counts(),work)
                    result['null_pairs'].append(pair);distances['clean' if arm=='O' else 'replacements'].append(pair['b2']['mean'])
            result['b2'][name]=distances
            result['secondary'][name]={arm:b2(fingerprints[name][arm],reference) for arm in ('M','A3')}
        result.update(evaluation_complete=True,owner_outcomes={name:'complete' for name in NAMES})
        result['costs']=self.costs();result['costs']['task_initial_errors']=tasks['initial_errors']
        self.journal.json('RESULTS.json',result)
        self.save_analysis(result,'analysis')
        return result

    def costs(self):
        return dict(counts=dict(self.budget.counts),completed_oracle=dict(self.oracle_totals),
            recovery_uncertainty=dict(self.journal.uncertainty),
            known_completed_operation_counts={k:self.budget.counts[k]-self.journal.uncertainty[k] for k in self.budget.counts},
            completed_phase_times={k:v['wall_seconds'] for k,v in self.journal.done.items()},runtime=self.runtime,
            cache_note='Completed-block oracle counts; lost interrupted work bounded separately. Resume cache is reconstructed only by subsequent reads; no refunds.')

    def save_analysis(self,result,name):
        from analyze import analyze,summary_lines
        directory=self.path/name;directory.mkdir(exist_ok=True)
        analysis=analyze(result);self.journal.json(name+'/analysis.json',analysis)
        target=directory/'SUMMARY.md';text='\n'.join(summary_lines(analysis))+'\n'
        if target.exists():
            if target.read_text(encoding='utf-8')!=text:raise ValueError('saved summary changed')
        else:
            with target.open('x',encoding='utf-8') as out:out.write(text);out.flush();os.fsync(out.fileno())
            self.journal.bytes+=target.stat().st_size

    def execute(self):
        try:
            value=self.run_final();self.budget.check()
            if not self.fixture:
                from final_supervise import validate_run
                validate_run(self.path)
                completed=self.costs()['known_completed_operation_counts']
                if completed['model_updates']!=229760 or completed['inverse_updates']!=38400:
                    raise ValueError('prescribed final workload incomplete')
            self.journal.json('COSTS.json',self.costs())
            self.journal.json('COMPLETED.json',dict(passed=True,fixture=self.fixture,counts=self.budget.counts,
                results=identity(self.path/'RESULTS.json'),analysis=identity(self.path/'analysis/analysis.json'),
                seeds=identity(self.path/'PRIVATE_FINAL_SEEDS.json'),rule=self.authority.get('rule_identity')))
            return value
        except BaseException as exc:
            failure=dict(type=type(exc).__name__,message=str(exc),infrastructure=isinstance(exc,OSError) and getattr(exc,'winerror',None) in (5,32,33),
                counts=dict(self.budget.counts),uncertainty=dict(self.journal.uncertainty),last_complete=self.journal.last_complete,
                completed_owner_ids=[c['id'] for c in self.rows],scientifically_complete=False)
            write(self.path/('INCOMPLETE_attempt'+str(self.attempt)+'.json'),failure)
            self.journal.event(dict(type='stopped',failure=failure))
            raise
