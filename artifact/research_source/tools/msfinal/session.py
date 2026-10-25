"""Consume existing accepted sessions; never create/reset/refund reservations."""
from pathlib import Path
import time
import os
from operatormark.corpus.state_v2_prerequisites import require_accepted
from msdev.campaign import Budget,SessionBoundary
from msdev.records import write,sha
from .marking import finite,identity


class Session:
    def __init__(self,record,configuration_sha256,worker_sha256,*,previous=None,clock=time.time,stage_amendment=None,finalization_only=False):
        self.finalization_only=finalization_only
        self.record=record;self.clock=clock;self.bindings=dict(configuration_sha256=configuration_sha256,worker_sha256=worker_sha256)
        self.value=require_accepted(record,kind='main_study_final_session',bindings=self.bindings)
        s=self.value
        if s.get('domain')!='final' or s.get('execution_authorized') is not True:
            raise ValueError('fresh independently accepted execution/session authority required')
        w=s.get('session',{})
        for value in (s.get('start'),s.get('deadline'),s.get('finalization_deadline'),w.get('start'),w.get('end'),w.get('worker_deadline')):
            if not finite(value):raise ValueError('immutable explicit session deadlines required')
        if not s['start']<=w['start']<w['worker_deadline']<=w['end']-30<=s['deadline']-30:
            raise ValueError('worker/owned cleanup/overall envelope differs')
        if not w['end']<=s['finalization_deadline']<=s['deadline']:raise ValueError('finalization outside original window')
        if w.get('closed') is not False or s.get('retries')!=0:raise ValueError('closed/retry session refused')
        if (not isinstance(s.get('consumed'),dict) or not s['consumed']
                or any(not finite(v) or v<0 for v in s['consumed'].values())):
            raise ValueError('existing consumed accounting required')
        if (set(s.get('limits',{}))!=set(s['consumed'])
                or any(not finite(s['limits'][k]) or s['limits'][k]<s['consumed'][k] for k in s['consumed'])
                or not isinstance(s.get('worker_root'),str) or not isinstance(s.get('launch_root'),str)):
            raise ValueError('unchanged cumulative limits and bound output/launch paths required')
        self.stage_amendment=stage_amendment;self.stage_policy=None
        if stage_amendment is not None:
            from .bounded_stage import admit
            self.stage_policy=admit(stage_amendment,s,self.bindings,clock,finalization_only=finalization_only)
        if previous is not None:
            old=require_accepted(previous,kind='main_study_final_session',bindings=self.bindings)
            limits_preserved=s['limits']==old['limits']
            if self.stage_policy is not None:
                limits_preserved=(self.stage_policy.get('previous_limits')==old['limits']
                    and set(s['limits'])==set(old['limits'])
                    and all(s['limits'][k]>=old['limits'][k] for k in old['limits']))
            window_preserved=s['start']==old['start'] and s['deadline']==old['deadline']
            if self.stage_policy and self.stage_policy.get('deadline_policy')=='individual_stages_v1':
                closed_at=self.stage_policy.get('previous_stage_closed_at')
                identity(self.stage_policy.get('previous_stage_closure_sha256'))
                window_preserved=(s['campaign_sha256']==old.get('campaign_sha256')
                    and finite(closed_at) and old['start']<=closed_at<=old['deadline']
                    and s['start']>=closed_at)
            if (s.get('previous_session_sha256')!=previous.sha256 or not window_preserved
                    or set(s['consumed'])!=set(old['consumed'])
                    or not limits_preserved or s['worker_root']!=old['worker_root']
                    or any(s['consumed'][k]<old['consumed'][k] for k in old['consumed'])):
                raise ValueError('original deadline/accounting reset refused')
        elif s.get('previous_session_sha256') is not None:raise ValueError('previous session evidence missing')
        projection=s.get('remaining_projection',{})
        identity(projection.get('sha256'))
        if self.stage_policy is None and (projection.get('complete') is not True or not finite(projection.get('upper_seconds'))
                or projection['upper_seconds']<0 or projection.get('unknowns')!=[]
                or projection['upper_seconds']>s['deadline']-clock()):
            raise ValueError('complete remaining-campaign cost envelope unavailable')
        self.budget=Budget(s,clock);self.check()

    def check(self):
        if self.record.read()!=self.value:raise ValueError('accepted session drift')
        if self.stage_amendment is not None and self.stage_amendment.read()!=self.stage_policy:
            raise ValueError('adopted stage amendment drift')
        if self.finalization_only:
            if self.clock()>=self.value['finalization_deadline']:
                raise TimeoutError('immutable finalization deadline')
        else:self.budget.check()

    @property
    def deadline(self):return self.value['session']['worker_deadline']

    def fit(self,node):
        if self.finalization_only:raise ValueError('finalization-only session cannot dispatch numerical work')
        self.check()
        if self.stage_policy is not None and node not in self.stage_policy['phases']:
            raise SessionBoundary('next phase requires separate bounded-stage approval: '+node)
        cost=self.value.get('node_envelopes',{}).get(node)
        if not finite(cost) or cost<=0:raise ValueError('bounded node cost unavailable: '+node)
        if self.clock()+cost>self.deadline:
            raise SessionBoundary('next complete node does not fit the existing worker session: '+node)


def closure_ok(r):
    c=(r or {}).get('closure',{})
    return bool(c.get('resolved') is True and c.get('child_reaped') is True and c.get('errors')==[]
        and c.get('survivors')==[] and r.get('stdout_closed') is True and r.get('errors')==[]
        and r.get('stream_overflow') is False and set(r.get('streams',{}))=={'stdout','stderr'}
        and all(x.get('capture_complete') is True and x.get('readback',{}).get('readback_verified') is True
                for x in r['streams'].values()))

def require_owned_host():
    """Establish the Linux subreaper premise used by the existing Q5 cleanup.

    Merely importing its helper functions does not run its CLI's prctl setup.
    Do this in each actual supervising process before it can launch children.
    """
    import ctypes,sys
    if os.name!='posix' or not sys.platform.startswith('linux'):
        raise ValueError('existing Q5 process identity/cleanup requires Linux')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'cannot establish owned descendant subreaper')


def supervise(session,folder,supervisor):
    """Existing Q5 owned route, using externally bound command/interpreter.

    No CLI/reservation creator is added. A future approved handoff supplies the
    same reviewed supervisor module and independently accepted session record.
    """
    from msdev.colab_environment import owned
    if session.finalization_only:raise ValueError('finalization-only session cannot launch worker')
    if os.name!='posix':raise ValueError('existing Q5 subreaper supervision is the POSIX GPU route; no Windows process-tree substitution')
    session.check();s=session.value;command=s.get('command');cwd=s.get('cwd')
    if Path(folder).resolve()!=Path(s['launch_root']).resolve():raise ValueError('accepted launch directory substitution')
    if not isinstance(command,list) or len(command)<2 or not all(isinstance(x,str) for x in command):
        raise ValueError('exact accepted worker command required')
    if sha(command[0])!=identity(s.get('interpreter_sha256')):
        raise ValueError('bound interpreter drift')
    if sha(supervisor.__file__)!=identity(s.get('supervisor_sha256')):
        raise ValueError('bound existing supervisor drift')
    if not Path(cwd).is_dir():raise ValueError('accepted worker directory missing')
    require_owned_host()
    parent=Path(folder);parent.mkdir(exist_ok=False)
    # Exclusive local consumed marker. It references the existing reservation;
    # it neither creates one nor removes/reconciles old consumed markers.
    write(parent/'LAUNCH_USED.json',dict(session_sha256=session.record.sha256,command=command,
        original_start=s['start'],original_deadline=s['deadline'],consumed=s['consumed']))
    return owned(command,session.deadline-session.clock(),s['session']['end'],parent/'supervisor',
        supervisor,cwd=cwd,acceptable=(0,4,5))


def close_result(session,worker_root,supervisor_record,output,check=lambda:None,*,phase_store=None):
    """Independent post-process readback; preserve original supervisor receipt."""
    from msdev.records import read
    from . import readback as R
    root=Path(worker_root);r=read(supervisor_record)
    external_check=check
    def check():
        external_check()
        if session.clock()>=session.value['finalization_deadline']:raise TimeoutError('original finalization deadline expired')
    if root.resolve()!=Path(session.value['worker_root']).resolve():raise ValueError('closed worker root substitution')
    if r.get('command')!=session.value.get('command') or not closure_ok(r):
        raise ValueError('owned supervisor/command closure unavailable')
    check();value=read(root/'RESULT.json')
    if value.get('selected_complete') is True or value.get('core_complete') is True:
        from .final_contracts import completion
        completion(read(root/'GRAPH.json'),value['products'])
    if (value.get('session_sha256')!=session.record.sha256
            or value.get('binding',{}).get('configuration_sha256')!=session.bindings['configuration_sha256']):
        raise ValueError('worker/accepted session correspondence differs')
    expected_store=value.get('phase_storage_sha256')
    if expected_store!=(None if phase_store is None else phase_store.record.sha256):raise ValueError('independent closer storage binding differs')
    for name in value['products']:
        from .graph import identity
        path=root/'products'/(identity(name)+'.json')
        if value.get('product_receipts',{}).get(name)!=sha(path):raise ValueError('product receipt identity differs')
        product=read(path)
        if product.get('binding')!=value['binding'] or product.get('name')!=name:
            raise ValueError('closed product binding differs')
        source=root/'work'/name
        if phase_store is not None and phase_store.archived(name):
            phase_store.verify_archived(name,product,sha(path))
            if source.exists() and R.inventory(source,check,**session.value['evidence_limits'])!=product['files']:
                raise ValueError('local archived phase changed before closure')
            continue
        if not source.exists():
            if phase_store is None:raise ValueError('closed phase files missing; restore evidence required')
            source=phase_store.resolve(name,product,sha(path))
        current=R.inventory(source,check,**session.value['evidence_limits'])
        if current!=product['files']:raise ValueError('closed product bytes/membership differ')
    receipt=dict(r,worker_result_sha256=sha(root/'RESULT.json'),supervisor_record_sha256=sha(supervisor_record),
        session_sha256=session.record.sha256,product_readback=True,
        original_deadline=session.value['deadline'],finalization_deadline=session.value['finalization_deadline'],
        core_complete=value.get('core_complete') is True and r.get('returncode')==0 and r.get('timed_out') is False,
        complete_programme=False,phase_storage_sha256=expected_store)
    # Canonical bytes make this receipt consumable by the existing externally
    # selected AcceptedRecord interface. This does not issue an acceptance.
    from operatormark.corpus.canonical import canonical_json_bytes
    check()
    with Path(output).open('xb') as f:f.write(canonical_json_bytes(receipt));f.flush();os.fsync(f.fileno())
    return receipt
