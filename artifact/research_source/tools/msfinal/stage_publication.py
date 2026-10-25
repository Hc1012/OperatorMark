"""Independent host verification under an explicitly delegated publication policy.

This is operational verification only, never input/runtime/scientific acceptance.
The policy must be independently adopted before launch; a producer flag or index
execution switch cannot substitute for it. Historical closure records stay raw.
"""
from pathlib import Path
import os
from msdev.records import read, sha
from operatormark.corpus.canonical import canonical_json_bytes
from operatormark.corpus.state_v2_prerequisites import require_accepted, AcceptedRecord
from . import session as S, readback as R, graph as G, owners


class HostPublication:
    def __init__(self, policy, session, transport, destination, *, phase_store=None):
        self.policy=policy;self.session=session;self.transport=transport
        self.phase_store=phase_store
        self.root=Path(session.value['worker_root']).resolve()
        self.destination=Path(destination).resolve()
        self.value=require_accepted(policy,kind='main_study_stage_publication_policy',bindings=dict(
            configuration_sha256=session.bindings['configuration_sha256'],
            worker_sha256=session.bindings['worker_sha256'],
            stage_amendment_sha256=session.stage_amendment.sha256 if session.stage_amendment else None,
            worker_root=str(self.root),destination=str(self.destination)))
        v=self.value
        if (session.stage_policy is None or v.get('domain')!='final'
                or v.get('delegate_closed_stage_verification') is not True
                or v.get('phases')!=session.stage_policy['phases']
                or v.get('scientific_acceptance') is not False
                or v.get('provider')!=transport.binding
                or v.get('file_limit')!=session.value['evidence_limits']['file_limit']
                or v.get('byte_limit')!=session.value['evidence_limits']['byte_limit']):
            raise ValueError('independently approved scoped publication delegation required')
        prior=v.get('prior_product_receipts',{})
        if (not isinstance(prior,dict) or set(prior)&set(v['phases'])
                or (session.value.get('previous_session_sha256') is None and prior)):
            raise ValueError('prior independently closed product membership differs')

    def check(self):
        if (self.policy.read()!=self.value or self.session.record.read()!=self.session.value
                or self.session.stage_amendment.read()!=self.session.stage_policy):
            raise ValueError('publication authority drift')
        if self.session.clock()>=self.session.value['finalization_deadline']:
            raise TimeoutError('original stage finalization deadline')

    def record(self, name, value):
        self.check()
        path=Path(self.session.value['launch_root'])/name
        raw=canonical_json_bytes(dict(value,verification_policy_sha256=self.policy.sha256))
        with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        return AcceptedRecord(path,sha(path),raw)

    def closure(self,path):
        self.check();s=self.session;result=read(self.root/'RESULT.json');v=read(path)
        expected=self.value['phases']
        prior=self.value.get('prior_product_receipts',{})
        products=result.get('products',[])
        if (Path(path).resolve()!=Path(s.value['launch_root']).resolve()/'CLOSURE.json'
                or len(products)!=len(set(products)) or set(products)!=set(prior)|set(expected)
                or any(result.get('product_receipts',{}).get(n)!=h for n,h in prior.items())
                or result.get('status')!='PAUSED_SESSION'
                or not S.closure_ok(v) or v.get('timed_out') is not False
                or v.get('returncode')!=5 or v.get('product_readback') is not True
                or v.get('session_sha256')!=s.record.sha256
                or v.get('worker_result_sha256')!=sha(self.root/'RESULT.json')):
            raise ValueError('exact completed first-owner phase/independent closure required')
        # Independent semantic readback as well as the closer's full byte inventory.
        from . import cpu_scoring as CPU
        for name in expected:
            with CPU.phase_context(self.root/'work'/name,result['binding'],producer_readback=True):
                R.training(self.root/'work'/name,owners.schedule(),self.check,monitors=True)
        raw=Path(path).read_bytes()
        return AcceptedRecord(Path(path),sha(path),raw)

    def permissions(self,root,destination,closure):
        self.check()
        if Path(root).resolve()!=self.root or Path(destination).resolve()!=self.destination:
            raise ValueError('publication path substitution')
        # Persist original closure/streams inside the return as well as their
        # external original files. No omitted raw capture or overwritten receipt.
        launch=Path(self.session.value['launch_root'])
        kept=self.root/'publication_evidence';kept.mkdir(exist_ok=False)
        sources=[launch/'CLOSURE.json',self.policy.path,self.session.record.path,self.session.stage_amendment.path]
        sources+=sorted(p for p in (launch/'supervisor').rglob('*') if p.is_file())
        copied=[]
        for i,p in enumerate(sources):
            self.check()
            if p.is_symlink():raise ValueError('closure source symlink')
            target=kept/(str(i)+'_'+p.name)
            before=sha(p)
            with p.open('rb') as src,target.open('xb') as dst:
                while block:=src.read(1<<20):self.check();dst.write(block)
                dst.flush();os.fsync(dst.fileno())
            if before!=sha(p) or before!=sha(target):raise ValueError('closure source changed')
            copied.append(dict(original=str(p),member=target.name,sha256=before))
        from msdev.records import write
        write(kept/'ORIGINS.json',dict(files=copied),durable=True)
        files=R.inventory(root,self.check,file_limit=self.value['file_limit'],byte_limit=self.value['byte_limit'])
        # Match export._publish's archive-only layout. Full archive/source checks
        # remain in the real closer/exporter; no silent omission or bulk restore.
        if self.phase_store is not None:
            result=read(self.root/'RESULT.json')
            if result.get('phase_storage_sha256')!=self.phase_store.record.sha256:
                raise ValueError('publication permanent-custody binding differs')
            for name in result['products']:
                if self.phase_store.archived(name):
                    prefix='work/'+name+'/'
                    files={n:v for n,v in files.items() if not n.startswith(prefix)}
        deadline=self.session.value['finalization_deadline']
        permit=self.record('HOST_EXPORT_PERMISSION.json',dict(accepted=True,synthetic=False,
            kind='main_study_final_private_export',domain='final',private_export=True,
            bindings=dict(worker_root=str(root),closure_sha256=closure.sha256,destination=str(destination)),
            deadline=deadline,file_limit=self.value['file_limit'],byte_limit=self.value['byte_limit'],
            membership_sha256=G.identity(files)))
        transfer=self.record('HOST_TRANSFER_PERMISSION.json',dict(accepted=True,synthetic=False,
            kind='main_study_final_stage_transfer',domain='final',closed_stage_only=True,active_streams=[],
            local_storage=True,local_storage_root=self.value['local_storage_root'],
            copy_graph=['retained_local_source','local_archive','remote_object'],deadline=deadline,
            provider=self.value['provider'],remote_capacity=self.value['remote_capacity'],
            bindings=dict(worker_root=str(root),local_export_permission_sha256=permit.sha256,
                closure_sha256=closure.sha256,destination=str(destination))))
        return permit,transfer
