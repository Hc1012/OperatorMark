"""Development-only in-memory limits and coarse, append-only recovery records."""
import json
import os
from pathlib import Path
import time
import uuid
import torch
from core import Seeds, read, write, identity, digest

COUNTERS=('solver_attempts','model_updates','inverse_updates')


def counts(**values):
    if any(k not in COUNTERS or type(v) is not int or v<0 for k,v in values.items()):
        raise ValueError('invalid operation count/bound')
    return {name:values.get(name,0) for name in COUNTERS}


class MemoryBudget:
    def __init__(self,limits,deadline,initial=None):
        self.limits=dict(limits);self.deadline=deadline
        self.counts=counts(**(initial or {}));self.block_end=None
        self.check()

    def check(self):
        if time.time()>=self.deadline:raise TimeoutError('immutable campaign work deadline')
        if any(self.counts[k]>self.limits[k] for k in COUNTERS):raise RuntimeError('operational quota exhausted')

    def charge(self,name,count=1):
        self.check()
        if type(count) is not int or count<0:raise ValueError('nonnegative integral charge required')
        end=self.counts[name]+count
        if end>self.limits[name] or (self.block_end is not None and end>self.block_end[name]):
            raise RuntimeError('operational/block quota exhausted: '+name)
        self.counts[name]=end


class FrozenSeeds:
    def __init__(self,path,plan,fixture=None):
        self.path=Path(path)
        if not self.path.exists():
            seeds=Seeds(self.path,'development_r2' if fixture is None else 'fixture_only',fixture=fixture)
            for label,offsets in plan.items():seeds.get(label,offsets)
            write(self.path.with_name('SEEDS_IDENTITY.json'),identity(self.path))
        pin=read(self.path.with_name('SEEDS_IDENTITY.json'))
        if digest(self.path)!=pin['sha256']:raise ValueError('seed identity mismatch')
        self.record=read(self.path);self.values=self.record['values'];self.events=self.record['events']
        self.stage=self.record['stage'];self.used=set();self.plan=plan
        if self.record['fixture']!=(fixture is not None):raise ValueError('seed fixture boundary')
        if set(self.values)!={self.stage+'/'+label for label in plan}:raise ValueError('seed label set changed')
        if len(self.events)!=len(plan):raise ValueError('seed event set incomplete')
        for event in self.events:
            label=event['label'];short=label[len(self.stage)+1:]
            if event['collision'] or event['offsets']!=list(plan[short]):raise ValueError('seed offset/collision contract')
            expanded={self.values[label]+offset for offset in event['offsets']}
            if expanded & self.used:raise ValueError('seed collision on reload')
            self.used.update(expanded)

    def get(self,label,offsets=(0,)):
        if label not in self.plan or tuple(offsets)!=tuple(self.plan[label]):
            raise ValueError('unplanned development seed label/offset')
        return self.values[self.stage+'/'+label]


def load_owned(fact):
    path=Path(fact['path'])
    if identity(path)!=fact:raise ValueError('complete artifact identity mismatch: '+str(path))
    return torch.load(path,map_location='cpu',weights_only=False)


class Journal:
    """One progress writer. Unsettled block bounds conservatively charge one interruption."""
    def __init__(self,path,budget,attempt,storage_limit=4*1024**3):
        self.path=Path(path);self.path.mkdir(exist_ok=True)
        self.budget=budget;self.attempt=attempt;self.limit=storage_limit
        self.progress=self.path/'progress.jsonl';self.done={};self.pending=None;self.bytes=0
        self.last_complete=None;self.uncertainty=counts();self.events=[]
        if self.progress.exists():
            raw=self.progress.read_bytes();lines=raw.splitlines(keepends=True)
            for i,line in enumerate(lines):
                try:event=json.loads(line)
                except (ValueError,UnicodeDecodeError):
                    raise ValueError('corrupt progress record; no verified recovery')
                self.events.append(event)
                if event['type']=='begin':
                    counts(**event['bound']);counts(**event['counts']);self.pending=event
                if event['type']=='complete':
                    if not self.pending or event['key']!=self.pending['key']:raise ValueError('unpaired complete record')
                    self.done[event['key']]=event;self.pending=None;self.last_complete=event
                if event['type']=='recovery_charge':
                    self.uncertainty=event['uncertainty'];self.pending=None
            # Scan once on worker construction, never every supervisor tick.
            self.bytes=sum(p.stat().st_size for p in self.path.rglob('*') if p.is_file())
        latest=next((e for e in reversed(self.events) if e['type'] in ('complete','recovery_charge')),None)
        if latest:self.budget.counts=dict(latest['counts'])
        if self.pending:
            if attempt!=1:raise ValueError('unsettled block requires sole classified continuation')
            bound=self.pending['bound'];initial=self.pending['counts']
            self.budget.counts={k:initial[k]+bound[k] for k in COUNTERS}
            self.uncertainty={k:self.uncertainty[k]+bound[k] for k in COUNTERS}
            self.replay_rng=self.pending['rng']
            self.event(dict(type='recovery_charge',key=self.pending['key'],counts=self.budget.counts,
                            uncertainty=self.uncertainty,meaning='upper bound, not observed lost operations'))
            self.pending=None
        self.budget.check()

    def event(self,row):
        row=dict(unix=time.time(),attempt=self.attempt,**row)
        raw=(json.dumps(row,allow_nan=False)+'\n').encode()
        if self.bytes+len(raw)>self.limit:raise RuntimeError('campaign evidence storage limit')
        with self.progress.open('ab') as out:out.write(raw);out.flush();os.fsync(out.fileno())
        self.bytes+=len(raw)

    def save(self,value,label):
        self.budget.check()
        if self.bytes+64*1024**2>self.limit:raise RuntimeError('checkpoint storage reserve unavailable')
        path=self.path/(label.replace('/','_')+'_'+uuid.uuid4().hex+'.pt')
        with path.open('xb') as out:torch.save(value,out);out.flush();os.fsync(out.fileno())
        self.bytes+=path.stat().st_size
        if self.bytes>self.limit:raise RuntimeError('campaign evidence storage limit')
        return identity(path)

    def block(self,key,bound,fn):
        if key in self.done:
            row=self.done[key];value=load_owned(row['artifact']);torch.set_rng_state(load_owned(row['rng']))
            return value
        self.budget.check();bound=counts(**bound)
        if any(self.budget.counts[k]+bound[k]>self.budget.limits[k] for k in COUNTERS):
            raise RuntimeError('full recoverable block cannot fit remaining quota')
        if hasattr(self,'replay_rng'):
            torch.set_rng_state(load_owned(self.replay_rng));del self.replay_rng
        start=dict(self.budget.counts);rng=self.save(torch.get_rng_state(),'rng_before')
        self.event(dict(type='begin',key=key,counts=start,bound=bound,rng=rng))
        self.budget.block_end={k:start[k]+bound[k] for k in COUNTERS}
        began=time.monotonic()
        print('BEGIN '+key,flush=True)
        value=fn()
        artifact=self.save(value,key);rng=self.save(torch.get_rng_state(),'rng_after')
        row=dict(type='complete',key=key,artifact=artifact,rng=rng,counts=dict(self.budget.counts),
                 wall_seconds=time.monotonic()-began,uncertainty=dict(self.uncertainty))
        self.event(row);self.done[key]=row;self.last_complete=row;self.budget.block_end=None
        print('COMPLETE '+key+' '+json.dumps(dict(counts=self.budget.counts,wall_seconds=row['wall_seconds'])),flush=True)
        return value

    def json(self,name,value):
        path=self.path/name
        if path.exists():
            if read(path)!=value:raise ValueError('completed JSON differs: '+name)
        else:
            write(path,value);self.bytes+=path.stat().st_size
        return identity(path)
