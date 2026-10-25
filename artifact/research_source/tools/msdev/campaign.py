"""Resumable sequential development graph with one status and boundary log.

No execution authority is created here. Callers must explicitly launch a later
development run. A session reservation is conservative until a clean close.
"""
from __future__ import annotations
import time
from pathlib import Path
from .records import read,write,sha,verify,progress,manifest

class SessionBoundary(Exception):
    """Planned pause before a phase; no failed scientific attempt or new allowance."""

class InfrastructureInterruption(Exception):
    """Explicit synthetic fault injection; real abrupt process loss leaves no receipt."""

class Budget:
    def __init__(self,state,clock=time.time):self.state=state;self.clock=clock
    def check(self):
        s=self.state;now=self.clock()
        if now>=s['deadline'] or now>=min(s['session']['end']-30,s['session'].get('worker_deadline',s['session']['end'])):raise TimeoutError('development deadline/cleanup reserve')
    def reserve(self,work,replay=0,transfer=0):
        self.check()
        if any(x is None or x<0 for x in (work,replay,transfer)):raise ValueError('unmeasured phase/recovery/transfer cost')
        remaining=min(self.state['deadline'],self.state['session']['end'],self.state['session'].get('worker_deadline',self.state['session']['end'])+30)-self.clock()
        required=1.2*(work+replay+transfer)+30
        if required>remaining:
            s=self.state
            later=min(s['deadline']-self.clock(),31500-s.get('gpu_charged',31500))
            if s['session']['device']=='cuda' and required<=later:
                raise SessionBoundary(f'phase needs {required:.3f}s; close this attached session before a later bounded session')
            raise TimeoutError(f'phase needs {required:.3f}s including margin/recovery/cleanup; remaining {remaining:.3f}s')
        return required

def begin(root,bindings,device,seconds,credits=None,rate=None,clock=time.time):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);p=root/'STATUS.json';now=clock()
    if p.exists():s=read(p)
    else:
        if any(root.iterdir()):raise ValueError('initial committed status missing; preserve partials, cannot restart the clock')
        s=dict(state='UNSTARTED',bindings=bindings,start=now,deadline=now+86400,
               windows_charged=0.,gpu_charged=0.,recoveries={},completed=[],products=[],rates={})
    if s['bindings']!=bindings:raise ValueError('source/config/input mismatch')
    if s.get('session',{}).get('closed') is not True and 'session' in s:raise ValueError('previous attached session/owned closure unresolved')
    if now>=s['deadline']:raise TimeoutError('original 24-hour deadline expired')
    cap=7200 if device=='windows' else 31500;field='windows_charged' if device=='windows' else 'gpu_charged'
    if device=='cuda':
        if credits is None or rate is None or credits<=0 or rate<=0:raise ValueError('current supplied credits and billing rate required')
        if credits<seconds/3600*rate:raise ValueError('insufficient supplied credits; no purchase')
    if seconds<=30 or s[field]+seconds>cap or now+seconds>s['deadline']:raise TimeoutError('aggregate session allowance exhausted')
    s[field]+=seconds # pessimistically charged until explicit clean closure, including lost sessions
    s['session']=dict(device=device,start=now,end=now+seconds,reserved=seconds,closed=False)
    s['state']='RUNNING';write(p,s,replace=p.exists());return s

def close_session(root,closed,clock=time.time):
    p=Path(root)/'STATUS.json';s=read(p);session=s['session']
    if session['closed']:raise ValueError('session already reconciled')
    elapsed=max(0,clock()-session['start'])
    if closed and elapsed<=session['reserved']:
        field='windows_charged' if session['device']=='windows' else 'gpu_charged'
        s[field]-=session['reserved']-elapsed
    session.update(closed=bool(closed),actual_seconds=elapsed)
    if not closed:s['state']='CLOSURE_UNRESOLVED'
    write(p,s,replace=True);return s

class Campaign:
    def __init__(self,root,backend,nodes,bindings,runtime,clock=time.time):
        self.root=Path(root);self.backend=backend;self.nodes=nodes;self.bindings=bindings;self.runtime=runtime
        self.path=self.root/'STATUS.json';self.state=read(self.path);self.budget=Budget(self.state,clock)
        if self.state['bindings']!=bindings:raise ValueError('source/config/input mismatch')
        if 'runtime' in self.state and self.state['runtime']!=runtime:raise ValueError('resume runtime mismatch')
        self.state['runtime']=runtime;self.save()
    def save(self):write(self.path,self.state,replace=True)
    def check(self):self.budget.check()
    def checkpoint(self,node,state):
        import torch
        began=time.monotonic()
        folder=self.root/'checkpoints';folder.mkdir(exist_ok=True)
        serial=max([self.state.get('serial',0)]+[int(p.stem) for p in folder.glob('*.pt') if p.stem.isdigit()])+1
        path=folder/f'{serial:05d}.pt';torch.save(state,path)
        receipt=dict(node=node['id'],arm=node.get('arm',node['id']),sha256=sha(path),bindings=self.bindings,
                     runtime=self.runtime,file=path.relative_to(self.root).as_posix())
        write(path.with_suffix('.json'),receipt)
        self.state.update(serial=serial,checkpoint=receipt);self.save()
        elapsed=time.monotonic()-began
        self.state['rates']['checkpoint_seconds']=max(elapsed,self.state['rates'].get('checkpoint_seconds',0))
        self.state['checkpoint_max_bytes']=max(path.stat().st_size,self.state.get('checkpoint_max_bytes',0));self.save()
        progress(self.root,'checkpoint',node=node['id'],serial=serial,updates=state.get('updates'),seconds=elapsed,bytes=path.stat().st_size)
    def recovery(self,incident,closure):
        if incident!='infrastructure' or not closure.get('resolved'):raise ValueError('infrastructure incident and established owned closure required')
        if self.state.get('failure_kind') in ('scientific','cap','unknown'):raise ValueError('scientific/cap/unknown failures cannot recover')
        cp=self.state.get('checkpoint');arm=self.state.get('active_arm')
        if not cp or cp['arm']!=arm:raise ValueError('no matched durable checkpoint for interrupted arm')
        if self.state['recoveries'].get(arm,0):raise ValueError('single recovery already consumed for '+arm)
        self.state['recoveries'][arm]=1;self.state['failure_kind']=None;self.save()
        upper=max((min(2048,n.get('steps',0)) for n in self.nodes if n.get('arm',n['id'])==arm),default=0)
        self.state.setdefault('uncertain_updates_by_arm',{})[arm]=upper;self.save()
        if any(n.get('evaluate') for n in self.nodes if n.get('arm',n['id'])==arm):
            self.state.setdefault('uncertain_evaluation_solves_by_arm',{})[arm]=dict(T128=16,P256=16);self.save()
        progress(self.root,'infrastructure_recovery',arm=arm,checkpoint=cp['file'],unobserved_updates_conservatively_charged=upper)
    def run(self,resume=False):
        try:
            for node in self.nodes:
                if node['id'] in self.state['completed']:continue
                self.check();self.backend.verify_inputs()
                produced=self.root/'products'/(node['id']+'.json')
                if produced.exists():
                    product=read(produced)
                    if product.get('bindings')!=self.bindings or product.get('runtime')!=self.runtime:raise ValueError('completed product binding drift')
                    verify(self.root,product['files'])
                    self.state['products'].append(product);self.state['completed'].append(node['id']);self.state['checkpoint']=None;self.save();continue
                self.state.update(active=node['id'],active_arm=node.get('arm',node['id']));self.save()
                if self.state['session']['device']=='cuda':self.project_remaining()
                progress(self.root,'phase_start',node=node['id'])
                restored=None;cp=self.state.get('checkpoint')
                if resume and cp and cp['node']==node['id']:
                    import torch
                    if cp['bindings']!=self.bindings or cp['runtime']!=self.runtime or sha(self.root/cp['file'])!=cp['sha256']:raise ValueError('checkpoint binding mismatch')
                    restored=torch.load(self.root/cp['file'],map_location='cpu',weights_only=True)
                if node['kind']!='train' and restored is None:self.checkpoint(node,dict(kind='phase_boundary',node=node['id']))
                phase_started=time.monotonic();product=self.backend.execute(node,self,restored)
                self.state['rates']['phase_'+node['kind']]=max(time.monotonic()-phase_started,self.state['rates'].get('phase_'+node['kind'],0))
                began=time.monotonic();self.backend.verify_inputs()
                self.state['rates']['input_check']=max(time.monotonic()-began,self.state['rates'].get('input_check',0))
                # Bind model, design and scientific result bytes before progression.
                product['files']={n:sha(self.root/n) for n in product.get('files',{})}
                product.update(bindings=self.bindings,runtime=self.runtime)
                write(self.root/'products'/(node['id']+'.json'),product)
                self.state['products'].append(product);self.state['completed'].append(node['id']);self.state['checkpoint']=None;self.save()
                progress(self.root,'phase_complete',node=node['id'])
            result=dict(complete=True,development_only=True,final_study_authorized=False,products=self.state['products'],
                        bindings=self.bindings,runtime=self.runtime,selection=self.backend.selection(),physical_qualification=False,
                        observed_optimizer_updates=self.state.get('executed_updates',0),
                        uncertain_updates_upper_bound=sum(self.state.get('uncertain_updates_by_arm',{}).values()),
                        uncertain_evaluation_solves_by_arm=self.state.get('uncertain_evaluation_solves_by_arm',{}),
                        campaign_start=self.state['start'],campaign_deadline=self.state['deadline'],
                        charged_windows_seconds=self.state['windows_charged'],charged_attached_seconds=self.state['gpu_charged'],
                        recoveries=self.state['recoveries'])
            if 'continuation' in self.state:
                result.update(scope='unmarked remaining development only',parent=self.state['continuation'],
                    parent_scientific_stop_preserved=True,marked_work_complete=False,
                    cumulative_observed_updates=self.state['continuation']['executed_updates']+result['observed_optimizer_updates'])
            write(self.root/'RESULT.json',result);self.state['state']='COMPLETE';self.save();return result
        except BaseException as exc:
            if isinstance(exc,SessionBoundary):
                self.state.update(state='PAUSED_SESSION',failure_kind=None,error=str(exc));self.save();raise
            kind='cap' if isinstance(exc,(TimeoutError,MemoryError)) else 'scientific' if isinstance(exc,(FloatingPointError,ValueError)) else 'infrastructure' if isinstance(exc,InfrastructureInterruption) else 'unknown'
            self.state.update(state='STOPPED',failure_kind=kind,error=repr(exc));self.save()
            p=self.root/f'failure_{self.state.get("serial",0):05d}_{time.time_ns()}.json'
            write(p,dict(error=repr(exc),kind=kind,active=self.state.get('active'),checkpoint=self.state.get('checkpoint')))
            raise

    def project_remaining(self):
        from .costs import remaining_campaign
        projection=remaining_campaign(self.nodes,self.state,self.budget.clock)
        write(self.root/f'REMAINING_PROJECTION_{time.time_ns()}.json',projection)
        self.state['remaining_projection']=projection;self.save()
        if projection['known_lower_bound_exceeds']:raise TimeoutError('remaining campaign measured/declared lower bound cannot fit; no arm removal or budget increase')

def prepare_output_inventory(root,check):
    root=Path(root)
    names=[p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
           and p.name not in ('OUTPUT_MANIFEST.json','OUTPUT_INVENTORY.json','STATUS.json','progress.jsonl','stdout.txt')]
    m=dict(kind='mainstudy-development-return',files={n:sha(root/n,check) for n in sorted(names)},closure_pending=True)
    write(root/'OUTPUT_INVENTORY.json',m)

def finish_manifest(root,check=lambda:None):
    root=Path(root);m=read(root/'OUTPUT_INVENTORY.json');m['closure_pending']=False
    # Large artifacts were hashed under the supervised worker deadline. Only
    # closed streams and small independent closure records are added afterward.
    for p in root.rglob('*'):
        if p.is_file() and (p.name in ('stdout.txt','SUPERVISOR.json','OUTPUT_INVENTORY.json','STATUS.json','progress.jsonl')
                           or any(part.startswith('notebook_supervisor_') for part in p.relative_to(root).parts)):
            m['files'][p.relative_to(root).as_posix()]=sha(p,check)
    m['complete']=read(root/'SUPERVISOR.json').get('complete') is True
    write(root/'OUTPUT_MANIFEST.json',m);return m
