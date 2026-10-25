"""Development and future final workflows. Import only after explicit CLI authorization."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from core import (HERE, PILOT, Budget, Output, Seeds, read, write, digest, identity,
                  cohort_gate, detection, threshold, utility, selection_score, select_recipe,
                  authorize_campaign, source_hashes)
from numerics import (Oracle, Splits, Observations, runtime, legacy, fresh, controls, directions,
                      forward, derivatives, generate, evaluate, task_data, keyed_truth, b2)
from training import Trajectory, legacy_attack


def save_model(path, model):
    with Path(path).open('xb') as stream:
        torch.save(model.state_dict(), stream)
    return identity(path)


def owned_model(path, expected):
    if digest(path)!=expected: raise ValueError('owned checkpoint provenance mismatch: '+str(path))
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=legacy()[0].SpatialFNO()
    expected_state=model.state_dict()
    actual={k:v for k,v in state.items() if k!='_metadata'}
    tensor_keys={k for k,v in expected_state.items() if torch.is_tensor(v)}
    if tensor_keys!={k for k,v in actual.items() if torch.is_tensor(v)}:
        raise ValueError('owned checkpoint named tensor schema')
    for k in tensor_keys:
        if actual[k].shape!=expected_state[k].shape or actual[k].dtype!=expected_state[k].dtype or not torch.isfinite(actual[k]).all():
            raise ValueError('owned checkpoint tensor shape/dtype/value')
    model.load_state_dict(actual,strict=True)
    return model.eval()


class Campaign:
    def __init__(self, config, authority, path, mode):
        authorize_campaign(config,authority,mode)
        self.config=config;self.authority=authority;self.mode=mode
        consumption=HERE/'private_runs';consumption.mkdir(exist_ok=True)
        execution=str(authority['execution_id'])
        if not execution.isascii() or not all(c.isalnum() or c in '-_' for c in execution):
            raise ValueError('execution identity must be an ordinary filename token')
        write(consumption/(execution+'.json'),dict(authority=authority,output=str(Path(path).resolve()),
                                                  consumed_unix=time.time()))
        self.out=Output(path,mode)
        self.budget=Budget(self.out.path/'COUNTERS.json',authority['limits'],authority['deadline_unix'])
        self.runtime=runtime()
        required=authority.get('runtime')
        if not required or any(self.runtime.get(k)!=v for k,v in required.items()):
            self.out.finish(dict(reason='approved runtime binding missing/different'),failed=True)
            raise ValueError('approved runtime binding missing/different')
        self.seeds=Seeds(self.out.path/'PRIVATE_SEEDS.json',mode+'/training')
        self.splits=Splits()
        self.oracle=Oracle(legacy()[0].solve,self.budget,real=True,log=self.out.log)
        self.cases=[];self.facts=[]
        write(self.out.path/'CONFIG.json',config)
        write(self.out.path/'RUNTIME.json',self.runtime)
        write(self.out.path/'SOURCE_IDENTITIES.json',source_hashes())

    def data(self, label, n, k, seeds=None):
        seeds=seeds or self.seeds
        value=generate(self.oracle,seeds.get(label+'/controls'),seeds.get(label+'/directions'),
                       n,k,label,self.splits)
        self.save_data(label,value)
        return value

    def save_data(self,label,data):
        filename=label.replace('/','_')+'.npz'
        path=self.out.path/filename
        with path.open('xb') as stream:
            np.savez(stream,**{name:getattr(data,name) for name in ('x64','y64','v64','dy64') if getattr(data,name) is not None})
        self.out.log(dict(type='permitted_observations',label=label,identities=data.identities(),
                          counterfactual_calls=data.information_calls,file=identity(path)))

    def train_owner(self, name):
        model=fresh(self.seeds.get(name+'/initialization'))
        data=self.data(name+'/owner',384,1)
        trainer=Trajectory(model,data,self.seeds.get(name+'/owner_shuffle'),self.budget,
                           recipe='derivative',batch=16,real=True)
        stats=trainer.extend(1920)
        self.out.log(dict(type='owner_training',owner=name,costs=stats,losses=trainer.losses))
        fact=save_model(self.out.path/(name+'_O.pt'),model);self.facts.append(fact)
        return dict(id=name,models={'O':model},costs={'O':stats})

    def mark(self, case):
        name=case['id'];old=legacy()[1]
        data=self.data(name+'/embedding',128,1)
        seed=self.seeds.get(name+'/private_key',offsets=(0,10000))
        key=old.b3_key(seed)
        write(self.out.path/(name+'_PRIVATE_KEY.json'),dict(seed=seed,**{k:v.tolist() for k,v in key.items()}))
        model=copy.deepcopy(case['models']['O'])
        trainer=Trajectory(model,data,self.seeds.get(name+'/embed_shuffle'),self.budget,
                           recipe='mark',batch=16,lr=.0003,real=True,
                           reference=case['models']['O'],key=key)
        stats=trainer.extend(320)
        case['models']['M']=model;case['key']=key;case['scale']=trainer.score_scale
        case['costs']['M']=stats
        self.facts.append(save_model(self.out.path/(name+'_M.pt'),model))
        self.out.log(dict(type='marking',owner=name,costs=stats,losses=trainer.losses,embedding_scale=case['scale']))

    def legacy_cases(self):
        old=PILOT/'run_20260924'
        baseline=read(HERE/'evidence/baseline/identities.json')
        pins={r['path']:r['sha256'] for r in baseline['files']}
        for path in (old/'identity.json',PILOT/'continuation_20260924/precheck.json'):
            if digest(path)!=pins.get(str(path.resolve())):
                raise ValueError('preserved legacy identity/precheck changed')
        original=read(old/'identity.json');pre=read(PILOT/'continuation_20260924/precheck.json')
        for key,path in dict(rule_sha256=PILOT/'RULE.md',runner_sha256=PILOT/'run.py',
                             spatial_sha256=PILOT/'spatial.py',solver_sha256=HERE.parents[1]/'operatormark/solvers/darcy_tpfa.py').items():
            if digest(path)!=original[key]: raise ValueError('pilot source drift')
        # Register spent pilot controls, including all perturbations, as unavailable new splits.
        used=set()
        seeds=original['seeds']
        for owner in (0,1):
            for label,n in (('train',384),('embed',128)):
                x=controls(seeds[label+'_controls'][owner],n)
                v=directions(seeds[label+'_directions'][owner],n,1)[:,0]
                for rows in (x,x+.001*v,x-.001*v): used.update(row.tobytes() for row in rows)
            used.update(row.tobytes() for row in controls(seeds['attack_controls'][owner],256))
        for label,n in (('clean_gate',64),('development',48),('final',64)):
            x=controls(seeds[label+'_controls'],n);v=directions(seeds[label+'_directions'],n,2)
            used.update(row.tobytes() for row in x)
            for k in range(2):
                used.update(row.tobytes() for row in x+.001*v[:,k]);used.update(row.tobytes() for row in x-.001*v[:,k])
        for label in ('task_targets','task_starts'):
            used.update(row.tobytes() for row in controls(seeds[label],4))
        self.splits.seen.update({row:'spent_pilot' for row in used})
        for owner in (0,1):
            models={}
            for arm,filename in (('O',f'clean_owner_{owner}.pt'),('M',f'owner_{owner}_b3.pt')):
                models[arm]=owned_model(old/filename,pre['artifact_sha256'][filename])
            keypath=old/f'private_owner_{owner}_b3_key.json'
            if digest(keypath)!=pre['artifact_sha256'][keypath.name]: raise ValueError('legacy key provenance')
            keyrow=read(keypath)
            key={k:np.asarray(keyrow[k],dtype=np.float32) for k in ('v','w')}
            self.cases.append(dict(id=f'pilot{owner}',models=models,key=key,
              scale=pre['scales'][str(owner)]['b3'],costs={'O':{'historical_reuse':True},'M':{'historical_reuse':True}}))

    def measure(self,case,arm,data,tasks,label,key=False,truth=None):
        metrics,raw=evaluate(case['models'][arm],data,tasks,self.oracle,self.budget,
                             label+'/'+case['id']+'/'+arm,case.get('key') if key else None,truth)
        # Per-sample values written before aggregation enters a result/selection record.
        self.out.log(dict(type='evaluation_raw',stage=label,owner=case['id'],arm=arm,raw=raw))
        return metrics,raw

    def clean_gate_measure(self,model,data,label):
        self.budget.check()
        metrics,raw=legacy()[1].evaluate(copy.deepcopy(model).eval(),data.tensors())
        self.out.log(dict(type='clean_gate_raw',label=label,
                          raw={k:v.tolist() if torch.is_tensor(v) else v for k,v in raw.items()}))
        return metrics

    def attack_data(self,case,include_rb=True):
        name=case['id']
        x=controls(self.seeds.get(name+'/attack_controls'),256)
        shared=forward(self.oracle,x,name+'/attack',self.splits)
        self.save_data(name+'/attack',shared)
        rb=None
        if include_rb:
            subset=Observations(shared.x64[:85],shared.y64[:85],information_calls=85)
            rb=derivatives(self.oracle,subset,directions(self.seeds.get(name+'/Rb_directions'),85,1),
                           name+'/Rb',self.splits)
            self.save_data(name+'/Rb',rb)
        return shared,rb

    def copied_attacks(self,case,data):
        seed=self.seeds.get(case['id']+'/shared_attack_shuffle')
        for arm,parent in (('A3','M'),('A0','O')):
            model=copy.deepcopy(case['models'][parent])
            stats=legacy_attack(model,data,seed,self.budget,self.out.log,real=True)
            case['models'][arm]=model;case['costs'][arm]=stats
            self.facts.append(save_model(self.out.path/(case['id']+'_'+arm+'.pt'),model))
            self.out.log(dict(type='copied_attack',owner=case['id'],arm=arm,costs=stats))

    def replacement(self,case,recipe,data):
        name=case['id']
        model=fresh(self.seeds.get(name+'/'+recipe+'/initialization'))
        shuffle=self.seeds.get(name+'/shared_attack_shuffle' if recipe=='R-a' else name+'/Rb_shuffle')
        return Trajectory(model,data,shuffle,self.budget,recipe='forward' if recipe=='R-a' else 'derivative',
                          batch=32 if recipe=='R-a' else 16,real=True)

    def run_development(self):
        self.legacy_cases()
        d1=self.train_owner('D1');self.cases.append(d1)
        clean_gate=self.data('D1_clean_gate',64,2)
        gate_metrics=self.clean_gate_measure(d1['models']['O'],clean_gate,'D1')
        write(self.out.path/'D1_CLEAN_GATE.json',gate_metrics)
        if not cohort_gate([gate_metrics],1):
            raise ValueError('D1 clean gate failed; preserve attempted owner and stop selection')
        self.mark(d1)
        data=self.data('development_evaluation',self.config['development_controls'],2)
        tasks=task_data(self.oracle,self.seeds,self.config['development_tasks'],'development_tasks',self.splits)
        trajectories={'R-a':[],'R-b':[]};baselines=[]
        for case in self.cases:
            shared,rb=self.attack_data(case)
            self.copied_attacks(case,shared)
            baseline,_=self.measure(case,'O',data,tasks,'development')
            baselines.append(baseline)
            for arm in ('A0','A3'):
                metrics,_=self.measure(case,arm,data,tasks,'development')
                self.out.log(dict(type='development_copied_metrics',owner=case['id'],arm=arm,metrics=metrics))
            for recipe,observations in (('R-a',shared),('R-b',rb)):
                trajectories[recipe].append(self.replacement(case,recipe,observations))
        histories={'R-a':[],'R-b':[]}
        for recipe in ('R-a','R-b'):
            previous=None
            for updates in self.config['selection_updates']:
                rows=[]
                for case,trainer,baseline in zip(self.cases,trajectories[recipe],baselines):
                    stats=trainer.extend(updates);case['models'][recipe]=trainer.model
                    self.out.log(dict(type='continuous_training',owner=case['id'],recipe=recipe,
                                      costs=stats,losses=trainer.losses))
                    checkpoint=self.out.path/(case['id']+'_'+recipe+'_'+str(updates)+'_continuous.pt')
                    trainer.save(checkpoint)
                    metrics,_=self.measure(case,recipe,data,tasks,'development/'+str(updates))
                    rows.append(dict(owner=case['id'],clean=baseline,replacement=metrics,costs=stats,
                                     checkpoint=identity(checkpoint)))
                score=selection_score(rows)
                visit=dict(updates=updates,score=score,cases=rows)
                histories[recipe].append(visit)
                write(self.out.path/(recipe+'_'+str(updates)+'_selection.json'),visit)
                if previous is not None and (previous-score)/previous<.05: break
                previous=score
        selected=select_recipe(histories)
        write(self.out.path/'SELECTION.json',dict(**selected,histories=histories,
              provisional=True,not_convergence_proof=True,actual_oracle=self.oracle.record()))
        return dict(selection=selected,actual_oracle=self.oracle.record(),counts=self.budget.counts,
                    next='Review/freeze rule and measured whole-campaign authority before any final owners')

    def calibrate(self):
        old=legacy()[1]
        data=self.data('calibration',self.config['calibration_controls'],2)
        tasks=task_data(self.oracle,self.seeds,self.config['calibration_tasks'],'calibration_tasks',self.splits)
        qualification=[]
        for case in self.cases:
            name=case['id'];key=case['key'];scale=case['scale']
            truth=keyed_truth(self.oracle,data.x64,key,'calibration_key/'+name,self.splits)
            clean,_=self.measure(case,'O',data,tasks,'calibration',True,truth)
            marked,_=self.measure(case,'M',data,tasks,'calibration',True,truth)
            base=self.seeds.get(name+'/calibration_wrong',offsets=tuple(range(16))+tuple(10000+i for i in range(16)))
            wrong=[old.b3_key(base+i) for i in range(16)]
            x=data.tensors()[0]
            wrong_scores=[old.b3_score(case['models']['M'],case['models']['O'],x,k,scale) for k in wrong]
            others=[old.b3_score(other['models']['O'],case['models']['O'],x,key,scale) for other in self.cases if other is not case]
            tau=threshold(others,wrong_scores,len(self.cases))
            self_score=old.b3_score(case['models']['O'],case['models']['O'],x,key,scale)
            if self_score != 0: raise ValueError('self-clean reference sanity check failed')
            detect=detection(old.b3_score(case['models']['M'],case['models']['O'],x,key,scale),wrong_scores,tau)
            gates=utility(marked,clean)
            case['threshold']=tau;case['qualified']=detect['detected'] and gates['passed']
            row=dict(owner=name,qualified=case['qualified'],clean_gate_passed=True,
                     clean=clean,marked=marked,detection=detect,utility=gates,
                     other_clean_true_scores=others,embedding_scale=scale)
            qualification.append(row);self.out.log(dict(type='qualification',**row))
        write(self.out.path/'QUALIFICATION.json',dict(owners=qualification,N_q=sum(c['qualified'] for c in self.cases),
                                                    locked_before_final=True))

    def run_final(self):
        selection=self.config['selection']
        if selection['primary'] not in ('R-a','R-b'): raise ValueError('global selection missing')
        recipes=['R-a']+(['R-b'] if selection['primary']=='R-b' else [])
        for recipe in recipes:
            if selection['recipes'][recipe]['updates'] not in self.config['selection_updates']:
                raise ValueError('selected recipe length outside prospective grid')
        names=[f'owner{i}' for i in range(self.config['owners'])]
        write(self.out.path/'PLANNED_OWNERS.json',dict(ids=names,no_replacement=True))
        # Draw all owner initialization identities once before the cohort starts.
        for name in names: self.seeds.get(name+'/initialization')
        for name in names: self.cases.append(self.train_owner(name))
        gate=self.data('clean_gate',64,2)
        rows=[self.clean_gate_measure(c['models']['O'],gate,c['id']) for c in self.cases]
        write(self.out.path/'CLEAN_COHORT.json',dict(ids=names,metrics=rows,passed=cohort_gate(rows,len(names))))
        if not cohort_gate(rows,len(names)): raise ValueError('clean-cohort gate failed; no marking or owner substitution')
        for case in self.cases: self.mark(case)
        self.calibrate()
        for case in self.cases:
            shared,rb=self.attack_data(case,include_rb='R-b' in recipes)
            self.copied_attacks(case,shared)
            for recipe in recipes:
                trainer=self.replacement(case,recipe,shared if recipe=='R-a' else rb)
                trainer.extend(1920)
                case['models'][recipe+'@1920']=copy.deepcopy(trainer.model)
                case['costs'][recipe+'@1920']=trainer.record()
                continuous=self.out.path/(case['id']+'_'+recipe+'_1920_continuous.pt')
                trainer.save(continuous);self.facts.append(identity(continuous))
                trainer.extend(selection['recipes'][recipe]['updates'])
                self.out.log(dict(type='continuous_training',owner=case['id'],recipe=recipe,
                                  costs=trainer.record(),losses=trainer.losses))
                case['models'][recipe]=trainer.model;case['costs'][recipe]=trainer.record()
                if selection['recipes'][recipe]['updates']==1920:
                    case['models'][recipe+'@1920']=trainer.model
                self.facts.append(save_model(self.out.path/(case['id']+'_'+recipe+'.pt'),trainer.model))
            case['models']['Rstar']=case['models'][selection['primary']]
        closure=dict(models=self.facts,private_inputs=[identity(p) for p in self.out.path.glob('*')
                      if p.suffix=='.npz' or p.name.endswith('_PRIVATE_KEY.json')],
                     qualification=identity(self.out.path/'QUALIFICATION.json'),
                     seeds=identity(self.out.path/'PRIVATE_SEEDS.json'),config=identity(self.out.path/'CONFIG.json'),
                     selection=selection,all_training_completed=True)
        write(self.out.path/'TRAINING_CALIBRATION_CLOSED.json',closure)
        for fact in self.facts+closure['private_inputs']+[closure['qualification'],closure['seeds'],closure['config']]:
            if digest(fact['path'])!=fact['sha256']: raise ValueError('closure identity changed')
        write(self.out.path/'FIRST_FINAL_GENERATION.json',dict(closed=identity(self.out.path/'TRAINING_CALIBRATION_CLOSED.json'),
                                                             started_unix=time.time()))
        final_seeds=Seeds(self.out.path/'PRIVATE_FINAL_SEEDS.json','final/evaluation',forbidden=self.seeds.used)
        data=self.data('final',self.config['final_controls'],2,final_seeds)
        tasks=task_data(self.oracle,final_seeds,self.config['final_tasks'],'final_tasks',self.splits)
        return self.final_evaluation(data,tasks,final_seeds,names)

    def final_evaluation(self,data,tasks,seeds,names):
        old=legacy()[1];x=data.tensors()[0]
        result=dict(schema='tpfa32_results.v1',fixture=self.mode=='verification',planned_owner_ids=names,
                    owners=[],b2={},null_pairs=[],secondary={},costs={})
        raw_by_case={};wrong_by_case={}
        for case in self.cases:
            name=case['id'];truth=keyed_truth(self.oracle,data.x64,case['key'],'final_key/'+name,self.splits)
            base=seeds.get(name+'/wrong',offsets=tuple(range(16))+tuple(10000+i for i in range(16)))
            wrong_by_case[name]=[old.b3_key(base+i) for i in range(16)]
            metrics={};raws={};detectors={};cached={}
            for arm,model in case['models'].items():
                if id(model) not in cached:
                    cached[id(model)]=self.measure(case,arm,data,tasks,'final',True,truth)
                metrics[arm],raws[arm]=cached[id(model)]
                if arm in ('M','A3'):
                    wrong=[old.b3_score(model,case['models']['O'],x,k,case['scale']) for k in wrong_by_case[name]]
                    detectors[arm]=detection(old.b3_score(model,case['models']['O'],x,case['key'],case['scale']),wrong,case['threshold'])
            raw_by_case[name]=raws
            row=dict(id=name,qualified=case['qualified'],qualification_status='locked calibration',
                     metrics=metrics,costs=case['costs'],detectors=detectors,
                     intact=dict(detected=detectors['M']['detected'],utility=utility(metrics['M'],metrics['O'])['passed']),
                     attacked=dict(detected=detectors['A3']['detected'],utility=utility(metrics['A3'],metrics['O'])['passed']))
            result['owners'].append(row)
        for case in self.cases:
            name=case['id'];reference=raw_by_case[name]['O']['jprediction']
            distances={'descendant':b2(raw_by_case[name]['A0']['jprediction'],reference)['mean'],
                       'clean':[],'replacements':[]}
            for other in self.cases:
                for arm,prefix in (('O','O'),('Rstar','Rstar')):
                    if arm=='O' and other is case: continue
                    model=other['models'][arm]
                    distance=b2(raw_by_case[other['id']][arm]['jprediction'],reference)
                    distances['clean' if arm=='O' else 'replacements'].append(distance['mean'])
                    wrong=[old.b3_score(model,case['models']['O'],x,k,case['scale']) for k in wrong_by_case[name]]
                    detected=detection(old.b3_score(model,case['models']['O'],x,case['key'],case['scale']),wrong,case['threshold'])
                    pair=dict(reference=name,model=prefix+'/'+other['id'],b2=distance,**detected)
                    self.out.log(dict(type='dependent_null_pair',**pair));result['null_pairs'].append(pair)
            result['b2'][name]=distances
            result['secondary'][name]={arm:b2(raw_by_case[name][arm]['jprediction'],reference)
                                        for arm in ('M','A3')}
        result['costs']=dict(actual_oracle=self.oracle.record(),counts=self.budget.counts,
                             runtime=self.runtime,task_initial_errors=tasks['initial_errors'])
        write(self.out.path/'RESULTS.json',result)
        from analyze import report
        report(result,self.out.path/'analysis')
        return dict(results=identity(self.out.path/'RESULTS.json'),actual_oracle=self.oracle.record(),counts=self.budget.counts)

    def execute(self):
        try:
            result=self.run_development() if self.mode=='development' else self.run_final()
            self.out.finish(result)
            return result
        except BaseException as exc:
            self.out.finish(dict(type=type(exc).__name__,message=str(exc),
                                 attempted_owner_ids=[c['id'] for c in self.cases],
                                 actual_oracle=self.oracle.record(),counts=self.budget.counts),failed=True)
            raise
