"""Explicit unmarked successor. Never clears or recovers the scientific stop.

Only reserve() creates a later, explicitly authorized session. Import/preparation
does no training, tensor deserialization, seed generation or environment setup.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
import shutil
import statistics
import time
from .records import read,write,sha,verify,progress
from .campaign import Campaign,begin
from .graph import operation_graph

DEADLINE=1791222784.4786708
PARENT_CHARGED=21600.
REMAINING=9900.
PARENT_UPDATES=180872
KIND='unmarked_remaining_v1'

def continuation_graph():
    original=operation_graph()
    # Exact original dictionaries: parent trajectory, cooldown branches and seeds
    # are inherited; only independent node ordering and scope change.
    selected=[copy.deepcopy(n) for n in original if n['id']=='A5B2_cost']
    selected += [copy.deepcopy(n) for n in original if n['id'].startswith('Aord_O_')]
    selected += [copy.deepcopy(n) for n in original if
                 (n['id'].startswith(('Rmain_','Rstar_')) and n['id']!='Rmain_1920')
                 or n['id']=='replacement_selection']
    nodes=[dict(id='parent_import',kind='parent_import')]+selected+[dict(id='unmarked_analysis',kind='unmarked_analysis')]
    if sum(n.get('steps',0) for n in nodes)!=93128:raise ValueError('continuation schedule drift')
    return nodes

def validate_parent(bundle):
    """Verify selected bytes against BOTH parent receipts and successor manifest."""
    bundle=Path(bundle);config=read(bundle/'CONFIG.json');p=read(bundle/'PARENT.json')
    if config.get('continuation_kind')!=KIND or sha(bundle/'PARENT.json')!=config['parent_sha256']:
        raise ValueError('parent binding mismatch')
    if sha(bundle/'ORIGINAL_UPLOAD_MANIFEST.json')!=p['bindings']['upload']:
        raise ValueError('original input manifest differs')
    original=read(bundle/'ORIGINAL_UPLOAD_MANIFEST.json')['files']
    for name,h in p['imported_input_identities'].items():
        if original.get(name)!=h or sha(bundle/name)!=h:raise ValueError('inherited data/settings changed: '+name)
    for name,d in p['derived_key_direction_bindings'].items():
        if original.get(d['source'])!=d['source_sha256'] or sha(bundle/name)!=d['sha256']:
            raise ValueError('existing evaluation direction binding changed')
    if sha(bundle/'SOURCE_BINDINGS.json')!=config['bindings']['sources']:raise ValueError('successor source binding changed')
    verify(bundle,p['evidence_files']);verify(bundle,p['import_files'])
    s=read(bundle/'parent/STATUS.json')
    if (s['state'],s['failure_kind'],s['active'],s['executed_updates'],s['gpu_charged'],s['deadline'])!=(
        'STOPPED','scientific','mark_selection',PARENT_UPDATES,PARENT_CHARGED,DEADLINE):
        raise ValueError('parent scientific stop/accounting changed')
    if not s['session'].get('closed') or s['bindings']!=p['bindings'] or s['runtime']!=p['runtime']:
        raise ValueError('parent identity/runtime/closure mismatch')
    if len(s['completed'])!=37 or len(set(s['completed']))!=37 or [q['id'] for q in s['products']]!=s['completed']:
        raise ValueError('parent completed-product roster mismatch')
    for relative in p['closure_records']:
        c=read(bundle/relative)
        if c.get('timed_out') is not False or c.get('stdout_closed') is not True or not c.get('closure',{}).get('resolved') or not c['closure'].get('child_reaped') or c['closure'].get('errors') or c['closure'].get('survivors'):
            raise ValueError('parent closure unresolved')
    products={q['id']:q for q in s['products']}
    expected={}
    for q in s['products']:
        if q['bindings']!=s['bindings'] or q['runtime']!=s['runtime']:raise ValueError('parent product binding mismatch')
        for n,h in q['files'].items():
            if n in expected and expected[n]!=h:raise ValueError('conflicting parent identities')
            expected[n]=h
    for n,h in p['import_files'].items():
        rel=n.removeprefix('parent/')
        if not n.startswith('parent/') or expected.get(rel)!=h:raise ValueError('artifact lacks parent product binding: '+n)
    required=['models/'+n+'.pt' for n in ('D1','D2','D3','Rmain_1920')]+['states/Rmain_1920.pt']
    required+=['evaluations/'+n+'/RESULT.json' for n in ('D1','D2','D3','Rmain_1920')]
    if any('parent/'+n not in p['import_files'] for n in required):raise ValueError('required parent artifact absent')
    for name in ('D1','D2','D3','Rmain_1920','calibration'):
        if read(bundle/'parent/products'/f'{name}.json')!=products[name]:raise ValueError('parent receipt differs: '+name)
    r=products['Rmain_1920']
    if r.get('recipe')!='forward32' or r.get('model_updates')!=1920 or r.get('updates')!=1920:
        raise ValueError('exact Rmain1920 trajectory required')
    if not math.isfinite(products['calibration']['tau5']) or products['calibration']['tau5']<=0:raise ValueError('tau5 unavailable')
    return p,s

def reserve(bundle,root,manifest,credits,rate,approved=False,clock=time.time):
    """One separate prebooked window; never writes to the parent campaign."""
    bundle=Path(bundle);root=Path(root)
    if not approved:raise ValueError('subsequent explicit real-launch instruction required')
    if sha(bundle/'MANIFEST.json')!=manifest:raise ValueError('successor upload identity mismatch')
    verify(bundle,read(bundle/'MANIFEST.json')['files']);p,parent=validate_parent(bundle)
    config=read(bundle/'CONFIG.json');bindings=dict(config['bindings'],upload=manifest)
    if root.name!=config['output_name']:raise ValueError('use the single bound successor output directory')
    if root.exists():
        if not (root/'STATUS.json').exists():raise ValueError('existing output without committed status; preserve it')
        old=read(root/'STATUS.json')
        if old['bindings']!=bindings:raise ValueError('existing successor identity mismatch')
        if old['session'].get('prebooked') and not old['session'].get('launch_started') and not old['session'].get('closed') and clock()<old['session']['end']-60:
            return old # fresh notebook kernel reuses the SAME charged reservation
        raise ValueError('successor already started/closed/expired; no retry or clock reset')
    now=clock()
    if now+REMAINING>parent['deadline']:raise TimeoutError('full 9900-second successor envelope no longer fits original deadline')
    if credits is None or rate is None or credits<=0 or rate<=0 or credits<REMAINING/3600*rate:raise ValueError('current credits/rate do not cover the bounded session')
    root.mkdir(parents=True,exist_ok=False)
    state=dict(state='UNSTARTED',bindings=bindings,start=parent['start'],deadline=parent['deadline'],
        windows_charged=parent['windows_charged'],gpu_charged=parent['gpu_charged'],
        recoveries=copy.deepcopy(parent['recoveries']),completed=[],products=[],rates=copy.deepcopy(parent['rates']),
        continuation=dict(parent_sha256=config['parent_sha256'],bindings=parent['bindings'],
            status_sha256=p['evidence_files']['parent/STATUS.json'],scientific_stop='mark_selection; scientific; preserved',
            executed_updates=parent['executed_updates'],charged_gpu_seconds=parent['gpu_charged'],
            original_deadline=parent['deadline'],not_infrastructure_recovery=True),
        session=dict(closed=True),executed_updates=0)
    write(root/'STATUS.json',state)
    state=begin(root,bindings,'cuda',REMAINING,credits,rate,clock)
    state['session'].update(prebooked=True,launch_started=False,previous_state='UNSTARTED',resume=False,incident=None,prior_closure=None)
    write(root/'STATUS.json',state,replace=True)
    progress(root,'successor_reserved',end=state['session']['end'],charged=state['gpu_charged'],parent_stop_preserved=True)
    return state

def projection(state,clock=time.time):
    pending=[n for n in continuation_graph() if n['id'] not in state.get('completed',[])]
    rates=state['rates'];terms=[]
    def add(name,count,rate,kind='measured'):
        terms.append(dict(name=name,count=count,rate=rate,seconds=None if count and rate is None else count*(rate or 0),kind=kind))
    for recipe in ('forward32','A5B2'):
        add(recipe,sum(n.get('steps',0) for n in pending if n.get('recipe')==recipe),rates.get(recipe))
    add('evaluations',sum(bool(n.get('evaluate')) for n in pending),rates.get('evaluation'))
    add('checkpoint_and_model_writes',sum(math.ceil(n.get('steps',0)/2048)+4 for n in pending if n['kind']=='train'),rates.get('checkpoint_seconds'))
    add('input_checks',2*len(pending),rates.get('input_check'))
    add('unmarked_B2_readout',int(any(n['kind']=='unmarked_analysis' for n in pending)),rates.get('phase_unmarked_analysis'))
    add('parent_import',int(any(n['kind']=='parent_import' for n in pending)),rates.get('phase_parent_import'))
    add('final_hashing_transfer_and_idle',int(bool(pending)),None,'unmeasured; must fit absolute attached window')
    known=sum(t['seconds'] for t in terms if t['seconds'] is not None)
    unknown=[t['name'] for t in terms if t['seconds'] is None]
    available=min(state['deadline'],state['session']['end'])-clock()
    return dict(terms=terms,known_remaining_seconds=known,unknown=unknown,
        with_20_percent_margin_and_cleanup=1.2*known+60,available_seconds=available,
        full_campaign_fits=None if unknown else 1.2*known+60<=available,
        known_lower_bound_exceeds=1.2*known+60>available,
        inherited_setup_ceiling=1380,analysis_work_ceiling=600,
        recovery='no automatic recovery; historical limits/consumption retained; any authorized recovery must fit the same remaining window',
        not_a_runtime_guarantee=True)

class UnmarkedCampaign(Campaign):
    def project_remaining(self):
        p=projection(self.state,self.budget.clock)
        write(self.root/f'REMAINING_PROJECTION_{time.time_ns()}.json',p)
        self.state['remaining_projection']=p;self.save()
        if p['known_lower_bound_exceeds']:raise TimeoutError('remaining unmarked known costs cannot fit; no scope/budget reduction')

def verify_resume_state(state,model,updates=1920,count=256):
    import torch
    if state['updates']!=updates or state['recipe']!='forward32' or state['sampler']['count']!=count or state['sampler']['batch']!=32:
        raise ValueError('Rstar optimizer/shuffle trajectory mismatch')
    if state['model'].keys()!=model.keys() or any(not torch.equal(state['model'][k],model[k]) for k in model):
        raise ValueError('Rstar saved model and optimizer-state model differ')
    if not state['optimizer'].get('state') or not state['optimizer'].get('param_groups'):
        raise ValueError('Rstar optimizer moments absent')

def make_backend(bundle,root,device='cuda'):
    from .backend import Backend
    from . import scoring,rules
    import numpy as np
    import torch
    class UnmarkedBackend(Backend):
        def key(self,owner,method):
            if method!='B3':raise ValueError('marked detector unavailable in unmarked scope')
            return dict(v=torch.from_numpy(self.load(f'key_direction_{owner}.npy')).to(self.device))
        def selection(self):
            p=self.root/'products/replacement_selection.json'
            return {'replacement_selection':read(p)} if p.exists() else {}
        def selected_name(self,owner,source):
            if source!='O':raise ValueError('marked weights forbidden in unmarked successor')
            return f'D{owner+1}'
        def execute(self,node,c,restored=None):
            if node not in continuation_graph():raise ValueError('node outside exact unmarked graph')
            if node['kind']=='parent_import':
                p,_=validate_parent(self.bundle);files={}
                for source,h in p['import_files'].items():
                    c.check();dest=self.root/source.removeprefix('parent/');dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists():
                        if sha(dest)!=h:raise ValueError('imported parent bytes drifted')
                    else:shutil.copyfile(self.bundle/source,dest)
                    files[dest.relative_to(self.root).as_posix()]=h
                for source,h in dict(p['evidence_files'],**{'PARENT.json':sha(self.bundle/'PARENT.json')}).items():
                    c.check();dest=self.root/'parent_evidence'/source;dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists():
                        if sha(dest)!=h:raise ValueError('preserved parent evidence drift')
                    else:shutil.copyfile(self.bundle/source,dest)
                    files[dest.relative_to(self.root).as_posix()]=h
                verify_resume_state(torch.load(self.root/'states/Rmain_1920.pt',map_location='cpu',weights_only=True),
                    torch.load(self.root/'models/Rmain_1920.pt',map_location='cpu',weights_only=True))
                # Calibration is a read-only predecessor input, not new acceptance.
                calibration=self.root/'parent_calibration.json'
                write(calibration,read(self.bundle/'parent/products/calibration.json'))
                return dict(id=node['id'],files=files,parent_sha256=sha(self.bundle/'PARENT.json'),new_scientific_work=False)
            if node['id']=='A5B2_cost':
                # Backend requires this input path; preserve the original receipt
                # separately, never claim it as a successor-produced calibration.
                path=self.root/'products/calibration.json'
                if not path.exists():shutil.copyfile(self.bundle/'parent/products/calibration.json',path)
                elif sha(path)!=sha(self.bundle/'parent/products/calibration.json'):raise ValueError('calibration drift')
            if node['kind']!='unmarked_analysis':return super().execute(node,c,restored)
            c.budget.reserve(600,0,30);started=time.monotonic()
            def check():
                c.check()
                if time.monotonic()-started>=600:raise TimeoutError('unmarked descriptive readout cap')
            x=torch.from_numpy(self.load('validation_x.npy')).to(self.device).repeat_interleave(2,0)
            v=torch.from_numpy(self.load('validation_v.npy')).to(self.device)
            selected=read(self.root/'products/replacement_selection.json')['selected'];replacement=f'Rstar_{selected}'
            cache={};rows=[];cohort=[]
            def dist(candidate,owner):
                pair=(candidate,owner)
                if pair not in cache:cache[pair]=scoring.distance(self.model(candidate),self.model(owner),x,v,check)
                return cache[pair]['mean']
            for owner in range(3):
                check();name=f'Aord_O_D{owner+1}';metrics=self.metrics(name)
                gates={k:rules.gate(metrics[k],self.metrics(f'D{owner+1}')[k]) for k in ('F','J','K','T')}
                matched=[f'Aord_O_D{j+1}' for j in range(3) if j!=owner]
                independent=[f'D{j+1}' for j in range(3) if j!=owner]+[replacement]
                lineage=rules.b2(dist(name,f'D{owner+1}'),[dist(n,f'D{owner+1}') for n in matched],
                    [dist(n,f'D{owner+1}') for n in independent])
                utility=all(g['passed'] is True for g in gates.values()) if all(g['status']=='complete' for g in gates.values()) else None
                cohort.append(dict(owner=owner,positive=lineage['positive'],utility=utility,margin=lineage['margin']))
                ratios={}
                for label,n in [('equal_step','Rmain_1920'),('selected_long',replacement)]:
                    rm=self.metrics(n);ratios[label]={k:rm[k]/metrics[k] if rules.nonnegative(rm[k]) and rules.nonnegative(metrics[k]) and metrics[k]>0 else None for k in ('J','T')}
                prior=read(self.bundle/'D1_RESULT.json')['task_f64'] if owner==0 else [a['relative'] for a in read(self.root/f'evaluations/D{owner+1}/RESULT.json')['rows']['T']]
                actual=[a['relative'] for a in read(self.root/f'evaluations/{name}/RESULT.json')['rows']['T']]
                delta=[a-b if a is not None and b is not None else None for a,b in zip(actual,prior)]
                rows.append(dict(owner=owner,source='O',metrics=metrics,utility_gates=gates,B2=lineage,
                    replacement_over_copy=ratios,paired_task_error_differences=delta,
                    paired_task_sd=statistics.stdev(delta) if len(delta)==16 and all(z is not None for z in delta) else None))
            raw=self.root/'scores/unmarked_B2_distances.json'
            write(raw,[dict(candidate=a,owner=b,**value) for (a,b),value in cache.items()])
            return dict(id=node['id'],files={raw.relative_to(self.root).as_posix():None},comparisons=rows,B2=rules.b2_cohort(cohort),
                selection=self.selection(),rates=c.state['rates'],development_descriptive=True,final_decision_applied=False,
                marked_conclusions=False,physical_claims=False,parent_stop_preserved=True,
                note='One development replacement reused descriptively across three clean owners; not a paired final-owner experiment.')
    return UnmarkedBackend(bundle,root,device)

def make_campaign(bundle,root,runtime):
    _,parent=validate_parent(bundle)
    if runtime!=parent['runtime']:raise ValueError('parent runtime differs; no automatic parity replay')
    s=read(Path(root)/'STATUS.json')
    if s.get('continuation',{}).get('status_sha256')!=sha(Path(bundle)/'parent/STATUS.json'):raise ValueError('successor parent status binding differs')
    return UnmarkedCampaign(root,make_backend(bundle,root),continuation_graph(),s['bindings'],runtime)

def verify_result_scope(root,result):
    """Independent final readback cannot turn this into completed marked work."""
    root=Path(root);parent=result.get('parent',{})
    p=root/'parent_evidence/parent/STATUS.json'
    if not p.is_file() or sha(p)!=parent.get('status_sha256'):raise ValueError('returned parent status missing/drifted')
    original=read(p)
    if original.get('state')!='STOPPED' or original.get('failure_kind')!='scientific' or original.get('active')!='mark_selection':
        raise ValueError('historical scientific stop not preserved')
    if result.get('parent_scientific_stop_preserved') is not True or result.get('marked_work_complete') is not False:
        raise ValueError('unmarked scope/stop misreported')
    if result.get('observed_optimizer_updates')!=93128 or result.get('cumulative_observed_updates')!=274000:
        raise ValueError('unmarked cumulative update accounting differs')
    if result.get('campaign_deadline')!=DEADLINE or result.get('charged_attached_seconds')!=31500:
        raise ValueError('original aggregate ceiling/deadline differs')
    for node,row in zip(continuation_graph(),result['products']):
        if node['kind']=='train' and (row.get('updates')!=node['steps'] or row.get('skipped')):
            raise ValueError('prescribed unmarked update count differs')
