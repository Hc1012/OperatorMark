"""Core worker using actual adapters and externally supplied acceptance bridges.

No command-line launch, reservation creation, acceptance issuer or scientific
configuration default. Missing independent receipts stop at a saved boundary.
"""
from pathlib import Path
from dataclasses import dataclass,replace
from contextlib import contextmanager
import time
import os
from operatormark.corpus.state_v2_prerequisites import require_accepted
from msdev.records import read,write,sha,verify,progress
from msdev.campaign import SessionBoundary
from . import backend as B, owners, attacks as T, orchestration as O, materials as A
from . import evaluation as E, nulls, analysis, graph as G, readback as R
from . import variants as V,expanded as X
from . import d2 as D2
from . import final_contracts as F,p256_tasks as P,resources as U
from . import cpu_scoring as CPU

SUPPORTED_POPULATION=10  # Existing paired-analysis implementation; not adoption.


class MissingEvidence(RuntimeError):
    """External acceptance unavailable; do not manufacture or poll for it."""


@dataclass(frozen=True)
class MarkedInputs:
    qualifications: dict
    marked_models: dict
    roster_acceptance: object
    attacks: tuple


class Evidence:
    """Required external trust-provider seam; every returned receipt is checked.

    Implementations load independently selected AcceptedRecord expectations,
    never turn this worker's JSON success flags into acceptance. No default
    production issuer is present. These methods may raise MissingEvidence.
    """
    def marking(self,owner_results):raise MissingEvidence('Evidence.marking: accepted final Prerequisites')
    def marked(self,admissions,results):raise MissingEvidence('Evidence.marked: model/qualification/roster/attack inputs')
    def opening(self,graph,descriptors,admissions):raise MissingEvidence('Evidence.opening: coordinated final opening')
    def model(self,admission,name,descriptor):raise MissingEvidence('Evidence.model: final-evaluation model acceptance')
    def substitution(self,admission):raise MissingEvidence('Evidence.substitution: accepted F64 request')
    def final_clean(self,admission,result):raise MissingEvidence('Evidence.final_clean: independent final clean readback')
    def final_intact(self,admission,result):raise MissingEvidence('Evidence.final_intact: independent final intact readback')
    def paired(self,graph,measurements):raise MissingEvidence('Evidence.paired: complete planned-owner paired readback')
    def additional(self,admissions,roster):raise MissingEvidence('Evidence.additional: independently admitted additional attack requests')
    def selected(self,phase,admissions,graph,products):raise MissingEvidence('Evidence.selected: independent '+phase+' contract requests')
    def p256(self,admission):raise MissingEvidence('Evidence.p256: independently accepted final P256 task request')
    def durable_boundary(self):raise MissingEvidence('Evidence.durable_boundary: independent remote custody after closed stage')
    def contract_bindings(self,roster_acceptance):raise MissingEvidence('Evidence.contract_bindings: accepted later selected-contract controls/memberships')
    def cpu_scores(self,name,requests):raise MissingEvidence('Evidence.cpu_scores: independently closed CPU scoring return')


class RecordEvidence(Evidence):
    """Concrete bridge over independently supplied objects, not self-attestation.

    Missing entries pause. After independent verification and owned closure a
    later session may supply additional records. This class never constructs an
    AcceptedRecord from a producer's output or copies a success flag into one.
    """
    def __init__(self,records):self.records=dict(records)
    def get(self,key):
        if key not in self.records:raise MissingEvidence('RecordEvidence missing independent entry: '+repr(key))
        return self.records[key]
    def marking(self,owner_results):return self.get('marking_inputs')
    def marked(self,admissions,results):return self.get('marked_inputs')
    def additional(self,admissions,roster):return self.get('additional_requests')
    def selected(self,phase,admissions,graph,products):return self.get(('selected_contracts',phase))
    def p256(self,admission):return self.get('P256_task')
    def durable_boundary(self):return self.get('durable_boundary')
    def contract_bindings(self,roster_acceptance):return self.get('contract_bindings')
    def cpu_scores(self,name,requests):return self.get(('CPU_scores',name))
    def add_cpu_batch(self,input_root,batch_sha,output_root,output_manifest_sha,acceptances,check=lambda:None,**transport):
        """Attach only independently admitted, fully read-back physical returns.

        Multiple physical batches may fill a logical phase. No partial import
        mutates the bridge, and conflicting earlier accepted results are refused.
        """
        from .cpu_batches import consume_return
        returned=consume_return(input_root,batch_sha,output_root,output_manifest_sha,acceptances,check,**transport)
        staged={}
        for phase,rows in returned.items():
            key=('CPU_scores',phase);old=self.records.get(key,{})
            if not isinstance(old,dict):raise ValueError('CPU return bridge type')
            if any(n in old and old[n]!=v for n,v in rows.items()):raise ValueError('conflicting accepted CPU return')
            staged[key]=dict(old,**rows)
        self.records.update(staged)
        return {phase:sorted(rows) for phase,rows in returned.items()}
    def opening(self,graph,descriptors,admissions):
        record=self.get('opening')
        return [replace(a,prerequisites=replace(a.prerequisites,opening=record)) for a in admissions]
    def model(self,admission,name,descriptor):return self.get(('model',admission.prerequisites.owner_id,name))
    def substitution(self,admission):return self.get('substitution_request')
    def final_clean(self,admission,result):
        record=self.get(('final_clean',admission.prerequisites.owner_id));value=record.read()
        if any(value.get('metrics',{}).get(k)!=result.get('metrics',{}).get(k) for k in 'FJKT'):
            raise ValueError('independent clean readback differs from produced observations')
        return record
    def final_intact(self,admission,result):
        record=self.get(('final_intact',admission.prerequisites.owner_id));value=record.read()
        if (any(value.get('metrics',{}).get(k)!=result.get('metrics',{}).get(k) for k in 'FJKT')
                or value.get('scores')!=result.get('scores')):
            raise ValueError('independent intact readback differs from produced observations')
        return record
    def paired(self,graph,measurements):
        record=self.get('paired_quality');rows=record.read().get('owners',[])
        if (len(rows)!=len(graph['planned_owners']) or len({r['owner_id'] for r in rows})!=len(rows)
                or {r['owner_id'] for r in rows}!=set(graph['planned_owners'])):
            raise ValueError('independent paired record omits planned owner')
        for row in rows:
            o=row['owner_id']
            for label,name in [('attacked_B3','Aord_M3_raw1920'),('replacement','Rstar')]:
                expected=measurements[o].get(name,{}).get('metrics',{})
                if any(row.get(label,{}).get(k)!=expected.get(k) for k in ('J','T')):
                    raise ValueError('independent paired observations differ')
        return record


class Worker:
    def __init__(self,root,configuration,session,evidence,*,resume=None,phase_store=None):
        self.root=Path(root);self.work=self.root/'work';self.configuration=configuration
        self.session=session;self.evidence=evidence;self.products=[];self.graph=None;self.phase_store=phase_store
        self.config=require_accepted(configuration,kind='main_study_final_configuration',
            bindings=dict(study_id=session.value['study_id']))
        if self.root.resolve()!=Path(session.value['worker_root']).resolve():raise ValueError('accepted worker output directory substitution')
        if (self.config.get('domain')!='final' or self.config.get('approved') is not True
                or session.bindings['configuration_sha256']!=configuration.sha256
                or session.bindings['worker_sha256']!=sha(__file__)):
            raise ValueError('current approved candidate/session correspondence required')
        # Validate all defined scope/settings before owner dispatch; qualification
        # remains a later output and is never presumed successful here.
        G.core(self.config,{o:dict(marked_checkpoint_complete=False,B3_qualified=None)
                            for o in self.config.get('owners',[])})
        V.selected(self.config)
        F.require_adopted(self.config)
        # CPU identities are bound later by accepted permit/compatibility records.
        CPU.validate_policy(self.config.get('cpu_scoring'),deferred=True)
        if self.config['population']!=SUPPORTED_POPULATION:
            raise ValueError('primary analysis supports ten owners only; alternate population implementation missing')
        from .bounded_stage import worker_binding
        self.binding=worker_binding(session)
        if self.root.exists():
            prior=read(self.root/'RESULT.json')
            previous_version=self.root/'sessions'/prior['session_sha256']/'RESULT.json'
            if (not previous_version.is_file() or read(previous_version)!=prior
                    or any(not (p/'RESULT.json').is_file() for p in (self.root/'sessions').iterdir() if p.is_dir())):
                raise ValueError('unclosed/unversioned worker session evidence; no automatic retry')
            if prior.get('status') not in ('WAITING_ACCEPTANCE','PAUSED_SESSION'):
                raise ValueError('failed/completed/unknown worker cannot retry')
            old=require_accepted(resume,kind='main_study_final_boundary_resume',
                bindings=dict(self.binding,previous_result_sha256=sha(self.root/'RESULT.json')))
            if old.get('closed') is not True or old.get('domain')!='final':raise ValueError('prior owned closure missing')
            # Only closed phases may be reused across an export/session boundary.
            # A phase_complete log or uploader success cannot establish custody.
            from .export import require_durable_boundary
            custody=self.evidence.durable_boundary()
            require_durable_boundary(custody,sha(self.root/'RESULT.json'),old.get('closure_sha256'),
                self.binding.get('original_deadline',self.binding))
            if read(self.root/'BINDINGS.json')!=self.binding:raise ValueError('worker binding/deadline reset')
            self.hold(custody)
        else:
            self.root.mkdir(parents=True);write(self.root/'BINDINGS.json',self.binding)
        self.session_dir=self.root/'sessions'/session.record.sha256
        self.session_dir.mkdir(parents=True,exist_ok=False)
        self.work.mkdir(exist_ok=True);(self.root/'products').mkdir(exist_ok=True)
        self.hold(configuration);self.hold(session.record)
        if phase_store is not None:
            from .phase_storage import PhaseStore
            if type(phase_store) is not PhaseStore:raise ValueError('bound phase storage required')
            expected=dict(self.binding,transport_source_sha256=phase_store.transport.binding['source_sha256'])
            require_accepted(phase_store.record,kind='main_study_final_phase_storage',bindings=expected)
            self.hold(phase_store.record)
        if resume is not None:self.hold(resume)

    def check(self):self.session.check()

    def hold(self,record):
        """Retain exact externally supplied bytes, never create an acceptance."""
        # Retention must support the already-adopted scientific admission route
        # as well as the unchanged formal route. ScientificRecord.read enforces
        # its restricted kinds, corpus lineage and D7/D8 limitations; retaining
        # its bytes does not convert it into formal acceptance.
        if type(record) not in (B.AcceptedRecord,B.ScientificRecord):
            raise ValueError('independently supplied accepted/scientific record required')
        record.read();folder=self.root/'accepted_records';folder.mkdir(exist_ok=True)
        path=folder/(record.sha256+'.json')
        if path.exists():
            if path.read_bytes()!=record.expected:raise ValueError('held acceptance record drift')
        else:
            with path.open('xb') as f:f.write(record.expected);f.flush();os.fsync(f.fileno())
        return record

    def phase_path(self,name):
        from msdev.records import safe_name
        safe_name(name);path=self.work/name
        if path.exists():return path
        receipt=self.root/'products'/(G.identity(name)+'.json')
        if receipt.exists():
            if self.phase_store is None:raise MissingEvidence('completed phase needs accepted restore: '+name)
            return self.phase_store.resolve(name,read(receipt),sha(receipt))
        return path

    @contextmanager
    def cpu_wave(self,name,expected):
        """Collect independent producers, never expose pending values past a barrier.

        Session.fit may interrupt a logical wave. The closed session exports the
        immutable subset already ready; resume reuses it and completes membership.
        """
        if getattr(self,'_cpu_wave',None) is not None:raise ValueError('nested CPU wave')
        expected=list(expected)
        if len(expected)!=len(set(expected)):raise ValueError('duplicate wave phase membership')
        wave=dict(name=name,expected=sorted(expected),seen=set(),pending={})
        self._cpu_wave=wave
        try:
            yield
        except SessionBoundary:
            self._cpu_wave=None
            self._inspect_cpu_wave_returns(wave)
            self._publish_cpu_wave(wave,complete=False)
            raise
        else:
            if wave['seen']!=set(expected):raise ValueError('CPU wave phase membership incomplete/extra')
            self._cpu_wave=None
            missing=self._inspect_cpu_wave_returns(wave)
            self._publish_cpu_wave(wave,complete=True)
            if missing:raise SessionBoundary('D6 wave ready; owned close/export and CPU returns required: '+name+' '+','.join(missing))
            for phase,(value,rb,finalize) in wave['pending'].items():
                done=self.phase(phase,lambda _:(_ for _ in ()).throw(ValueError('CPU wave producer replay refused')),rb,cpu_finalize=finalize)
                if CPU.members(done):raise ValueError('CPU wave remains incomplete')
                value.clear();value.update(done)
        finally:self._cpu_wave=None

    def _inspect_cpu_wave_returns(self,wave):
        # Partial physical batches may already be verified without releasing a
        # logical dependency barrier. Never put their requests back in the queue.
        verified={};missing=[]
        for phase,(value,rb,finalize) in wave['pending'].items():
            completed=phase+'__cpu';members=CPU.members(value)
            receipt=self.root/'products'/(G.identity(completed)+'.json')
            if receipt.exists():
                done=self.phase(completed,lambda _:(_ for _ in ()).throw(ValueError('completed CPU view replay')),rb)
                if CPU.members(done):raise ValueError('incomplete completed CPU view')
                for m in members.values():verified[m['sha256']]=dict(completed_phase_receipt_sha256=sha(receipt))
                continue
            try:returned=self.evidence.cpu_scores(phase,members)
            except MissingEvidence:missing.append(phase);continue
            if not isinstance(returned,dict) or set(returned)-set(members):raise ValueError('CPU wave extra return membership')
            if set(returned)!=set(members):missing.append(phase)
            dest=self.phase_path(phase) if returned else None
            for directory,r in returned.items():
                m=members[directory]
                CPU.return_readback(dest/directory,m['sha256'],r['root'],r['acceptance'],r['closure'],self.check,observations=r.get('observations'))
                verified[m['sha256']]=dict(acceptance_sha256=r['acceptance'].sha256,closure_sha256=r['closure'].sha256,
                    manifest_sha256=sha(Path(r['root'])/'RETURN_MANIFEST.json'))
        wave['verified_returns']=verified
        return missing

    def _publish_cpu_wave(self,wave,*,complete):
        phases={}
        for name,(value,_,_) in wave['pending'].items():
            receipt=self.root/'products'/(G.identity(name)+'.json')
            phases[name]=dict(receipt_sha256=sha(receipt),requests=CPU.members(value))
        record=dict(schema='D6.wave.v1',wave=wave['name'],binding=self.binding,
            required_phases=wave['expected'],visited_phases=sorted(wave['seen']),
            all_producers_visited=complete,phases=phases,verified_returns=wave.get('verified_returns',{}))
        path=self.root/'cpu_waves'/wave['name']/(G.identity(record)+'.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if read(path)!=record:raise ValueError('immutable CPU wave drift')
        else:write(path,record)
        if not hasattr(self,'cpu_wave_records'):self.cpu_wave_records={}
        self.cpu_wave_records[path.relative_to(self.root).as_posix()]=sha(path)

    def _wave_value(self,name,value,readback,finalize):
        wave=getattr(self,'_cpu_wave',None)
        if wave is None:return False
        if name not in wave['expected'] or name in wave['seen']:raise ValueError('unexpected/duplicate CPU wave phase')
        wave['seen'].add(name)
        if CPU.members(value):wave['pending'][name]=(value,readback,finalize)
        return True

    def phase(self,name,run,readback=lambda path:None,*,cpu_finalize=None):
        self.check();receipt=self.root/'products'/(G.identity(name)+'.json')
        if receipt.exists() and self.phase_store is not None and self.phase_store.archived(name):
            product=read(receipt)
            if product.get('binding')!=self.binding or product.get('name')!=name:raise ValueError('archived phase binding differs')
            value=self.phase_store.verify_archived(name,product,sha(receipt))
            local=self.work/name
            if local.exists() and R.inventory(local,self.check,**self.session.value['evidence_limits'])!=product['files']:
                raise ValueError('local archived phase drift')
            self.products.append(product)
            if self._wave_value(name,value,readback,cpu_finalize):return value
            if not CPU.members(value):return value
            completed=name+'__cpu'
            if (self.root/'products'/(G.identity(completed)+'.json')).exists():
                return self.phase(completed,lambda _:(_ for _ in ()).throw(ValueError('archived producer replay')),readback)
            # Pending data still needs the original dependencies for its first
            # merge; capacity-bound restore refuses if that active set cannot fit.
            self.products.pop()
        dest=self.phase_path(name)
        if receipt.exists():
            value=read(receipt)
            if value.get('binding')!=self.binding or value.get('name')!=name:raise ValueError('completed phase binding differs')
            now=R.inventory(dest,self.check,**self.session.value['evidence_limits'])
            if now!=value['files']:raise ValueError('completed phase membership/content drift')
            for directory,m in CPU.members(value['value']).items():CPU.verify_request(dest/directory,m['sha256'],self.check)
            with CPU.phase_context(dest,{},producer_readback=bool(CPU.members(value['value']))):readback(dest)
            self.products.append(value)
            if self._wave_value(name,value['value'],readback,cpu_finalize):return value['value']
            if CPU.members(value['value']):
                completed=name+'__cpu'
                if (self.root/'products'/(G.identity(completed)+'.json')).exists():
                    return self.phase(completed,lambda out:(_ for _ in ()).throw(ValueError('completed CPU phase must not rerun')),readback)
                returned=self.evidence.cpu_scores(name,CPU.members(value['value']))
                return self.phase(completed,lambda out:CPU.finish_phase(dest,out,value['value'],returned,receipt,self.check,cpu_finalize),readback)
            return value['value']
        if dest.exists():raise ValueError('unreceipted partial phase retained; no automatic replay: '+name)
        self.session.fit(name);dest.parent.mkdir(parents=True,exist_ok=True)
        progress(self.root,'phase_start',node=name)
        cell=next((k for k,c in (self.graph or {}).get('d2_cells',{}).items() if name.startswith('evaluations/') and name[len('evaluations/'):] in c['members']),None)
        binding=dict(self.binding,phase_name=name,comparison_cell=cell,cpu_policy=self.config.get('cpu_scoring'))
        with CPU.phase_context(dest,binding):value=run(dest)
        dest.mkdir(exist_ok=True)
        write(dest/'PHASE_VALUE.json',value)
        for directory,m in CPU.members(value).items():CPU.verify_request(dest/directory,m['sha256'],self.check)
        with CPU.phase_context(dest,binding,producer_readback=bool(CPU.members(value))):readback(dest)
        files=R.inventory(dest,self.check,**self.session.value['evidence_limits'])
        record=dict(name=name,binding=self.binding,files=files,value=value)
        write(receipt,record);self.products.append(record)
        progress(self.root,'phase_complete',node=name)
        if self._wave_value(name,value,readback,cpu_finalize):return value
        if CPU.members(value):raise SessionBoundary('D6 designs ready: close/export this phase before CPU scoring; no paid wait: '+name)
        return value

    def run(self,training_inputs):
        status='STOPPED';error=None;measurements={};details={}
        try:
            self.check();ids=self.config['owners'];base=self.config['shared_baseline_id']
            if [(p.owner_id,p.role) for p in training_inputs]!=[(o,'O') for o in ids]+[(base,'B10')]:
                raise ValueError('complete owner/B10 source roster required')
            admitted=[owners.admit(p) for p in training_inputs]
            for p in training_inputs:
                for record in (p.configuration,p.data,p.runtime):self.hold(record)
            if (len({p.configuration.sha256 for p in training_inputs})!=1
                    or training_inputs[0].configuration.sha256!=self.config.get('owner_training_configuration_sha256')):
                raise ValueError('final/owner-training configuration binding missing')
            inventories=[tuple(d['entries'][k]['sha256'] for k in ('x','v','y','dy')) for _,d in admitted]
            if len(set(inventories))!=1:raise ValueError('O/B10 same-corpus correspondence differs')
            owner_results={}
            with self.cpu_wave('owners',['owners/'+p.owner_id for p in training_inputs]):
                for p in training_inputs:
                    owner_results[p.owner_id]=self.phase('owners/'+p.owner_id,
                        lambda out,p=p:owners.execute(p,out,self.check,deadline=self.session.deadline),
                        lambda out:R.training(out,owners.schedule(),self.check,monitors=True))
            prerequisites=self.evidence.marking(owner_results)
            if [p.owner_id for p in prerequisites]!=ids:raise ValueError('all planned marking owners required')
            admissions=[]
            for p in prerequisites:
                if (p.configuration.sha256!=self.configuration.sha256
                        or p.clean_model_sha256!=owner_results[p.owner_id]['products']['cooled81920']['sha256']):
                    raise ValueError('clean training to marking model/configuration substitution')
                admissions.append(B.admit(p))
                for record in (p.configuration,p.inputs,p.runtime,p.calibration,p.clean_evaluation,p.assets):self.hold(record)
            marks={}
            with self.cpu_wave('qualification',['marking/'+p.owner_id for p in prerequisites]):
                for p in prerequisites:
                    marks[p.owner_id]=self.phase('marking/'+p.owner_id,
                        lambda out,p=p:B.run_owner(p,out,check=self.check),lambda out:self.mark_readback(out),
                        cpu_finalize=lambda source,out,replacements,value,p=p:B.complete_pending_owner(p,source,out,next(iter(replacements.values())),self.check))
                    if marks[p.owner_id]['state'] in ('FAILED','BLOCKED','UNATTEMPTED'):
                        raise ValueError('marking operational failure; no later owner/attack')
            bound=self.evidence.marked(admissions,marks)
            if type(bound) is not MarkedInputs:raise ValueError('explicit marked-model/roster bindings required')
            self.hold(bound.roster_acceptance)
            for o in ids:
                model=bound.marked_models.get(o)
                if marks[o].get('marked_checkpoint_complete') is True:
                    if model is None or model.sha256!=marks[o]['model_sha256']:
                        raise ValueError('marked roster does not bind produced mark')
                    self.hold(model.acceptance)
                    if model.qualification is not None:self.hold(model.qualification)
                elif model is not None:raise ValueError('incomplete mark promoted into roster')
            admitted_by_owner={a.prerequisites.owner_id:a for a in admissions}
            for request in bound.attacks:
                self.hold(request.training);self.hold(request.parent.acceptance)
                o=request.admission.prerequisites.owner_id
                if o not in admitted_by_owner or request.admission.bindings!=admitted_by_owner[o].bindings:
                    raise ValueError('attack request changed candidate/input/runtime binding')
            roster,_,_=O.admit_training_wave(bound.attacks,admissions,bound.qualifications,
                bound.roster_acceptance,marked_models=bound.marked_models)
            core_graph=G.core(self.config,roster)
            self.graph=V.augment(core_graph,self.contract_configuration(bound),roster)
            # All currently available additional inputs/parents are checked
            # before any post-qualification core training or teacher queries.
            additional=self.admit_additional(admissions,roster,bound)
            definition=self.root/'GRAPH.json'
            if definition.exists():
                if read(definition)!=self.graph:raise ValueError('post-qualification graph drift')
            else:write(definition,self.graph)
            write_path=self.root/'WORKLOAD.json';workload=U.ledger(self.graph,self.config)
            if write_path.exists():
                if read(write_path)!=workload:raise ValueError('workload changed')
            else:write(write_path,workload)
            requests={p.owner_id:p for p in training_inputs}
            with self.cpu_wave('replacement',['attacks/'+r.admission.prerequisites.owner_id+'_'+r.parent.kind+'_'+r.variant for r in bound.attacks]):
                for request in bound.attacks:
                    owner=request.admission.prerequisites.owner_id
                    key='attacks/'+owner+'_'+request.parent.kind+'_'+request.variant
                    monitor=None
                    if request.variant=='replacement':
                        p=requests[owner];cfg,data=owners.admit(p)
                        monitor=owners.monitor_callback(p,cfg,data,self.check,self.session.deadline)
                    self.phase(key,lambda out,r=request,m=monitor:T.execute_training(r,out,self.check,on_checkpoint=m),
                        lambda out,nodes=self.graph['trajectories'][key],m=monitor:R.training(out,nodes,self.check,monitors=m is not None))
            if self.graph['additional_jobs']:
                keys=[(r.admission.prerequisites.owner_id,r.parent.kind,r.variant) for r in additional]
                caches={}
                for o,parent in sorted({(k[0],k[1]) for k in keys if k[2] in V.STUDENTS}):
                    candidates=[r for r in additional if r.admission.prerequisites.owner_id==o and r.parent.kind==parent and r.variant in V.STUDENTS]
                    jvp=any(r.variant=='extract_jvp' for r in candidates)
                    r=next((r for r in candidates if r.variant=='extract_jvp'),candidates[0]);name='queries/'+o+'_'+parent
                    self.phase(name,lambda out,r=r,jvp=jvp:X.acquire(r,out,self.check,jvp=jvp),
                        lambda out,r=r,jvp=jvp:X.query_readback(out,X.query_binding(r,jvp),self.check))
                    caches[(o,parent)]=self.phase_path(name)
                for r in additional:
                    o=r.admission.prerequisites.owner_id;key='additional/'+o+'_'+r.parent.kind+'_'+r.variant
                    cache=caches.get((o,r.parent.kind)) if r.variant in V.STUDENTS else None
                    self.phase(key,lambda out,r=r,c=cache:X.execute(r,out,self.check,cache=c),
                        lambda out,r=r:R.training(out,V.schedule(r.variant),self.check))
            selected_pre=self.selected_before_opening(admissions)
            descriptors=self.descriptors()
            selected_final,selected_binding=self.selected_final_bindings(admissions,selected_pre)
            p256_binding=None
            if self.graph['P256_task_enabled']:
                reference=next(a for a in admissions if a.prerequisites.owner_id==self.graph['baseline_reference_owner'])
                entries=A.catalog(reference)['entries']
                p256_binding=dict(source_sha256=sha(P.__file__),inputs={n:entries[n]['sha256'] for n in ('final_starts','final_targets','final_targets_p256')},counts=P.workload(64,120))
            lock=dict(selected_contracts=selected_binding,P256_task=p256_binding)
            if self.graph.get('d2_cells'):lock['D2']=dict(policy=self.graph['d2_policy'],cells=self.graph['d2_cells'])
            lockpath=self.root/'SELECTED_OPENING_INPUTS.json'
            if lockpath.exists():
                if read(lockpath)!=lock:raise ValueError('selected pre-opening lock drift')
            else:write(lockpath,lock)
            original_bindings={a.prerequisites.owner_id:a.bindings for a in admissions}
            admissions=self.evidence.opening(self.graph,descriptors,admissions)
            if [a.prerequisites.owner_id for a in admissions]!=ids:raise ValueError('incomplete opening roster')
            b2_expected={o:{o+'/'+k:dict(matched=v['matched'],independent=v['independent'])
                for k,v in groups.items()} for o,groups in self.graph['b2_groups'].items()}
            for a in admissions:
                if a.bindings!=original_bindings[a.prerequisites.owner_id]:raise ValueError('opening changed admitted candidate')
                opening=A.require_opening(a)
                self.hold(a.prerequisites.opening)
                if opening.get('models')!=descriptors or opening.get('b2_groups')!=b2_expected:
                    raise ValueError('coordinated opening omits derived model/null membership')
                if opening.get('selected_contracts')!=selected_binding or opening.get('P256_task')!=p256_binding:
                    raise ValueError('coordinated opening omits contract requests/selection/P256')
                if self.graph.get('d2_cells') and opening.get('D2')!=lock['D2']:
                    raise ValueError('coordinated opening omits frozen D2 rule/cell membership')
            if len({a.prerequisites.opening.sha256 for a in admissions})!=1:raise ValueError('multiple final openings')
            by_owner={a.prerequisites.owner_id:a for a in admissions};items={}
            for o in ids:
                items[o]={}
                for name,d in self.graph['models'][o].items():
                    desc=descriptors[o][name];path=self.model_path(d)
                    deps={k:(self.model_path(v),sha(self.model_path(v))) for k,v in d.get('dependencies',{}).items()}
                    item=O.FinalModel(name,path,desc['model_sha256'],self.hold(self.evidence.model(by_owner[o],name,desc)),
                        d['wrapper_steps'],d.get('procedure'),deps)
                    O.final_model(item,by_owner[o]);items[o][name]=item
            # Require common solver/task inputs before billing the shared F64 reference once.
            reference=by_owner[self.graph['baseline_reference_owner']]
            for a in admissions:
                for name in ('final_x','final_rho','final_v','final_truth','final_tangent','final_starts','final_targets','final_targets_p256'):
                    if A.catalog(a)['entries'][name]['sha256']!=A.catalog(reference)['entries'][name]['sha256']:
                        raise ValueError('shared final task/utility population differs')
            projection_stores={}
            conditional={n for c in self.graph.get('d2_cells',{}).values() for n in c['members']}
            final_phases=['evaluations/'+o+'/'+n for o in ids for n in items[o] if o+'/'+n not in conditional]
            final_phases += [j['path']+'/final' for j in self.graph['contract_jobs'] if j['arm']!='overwrite']
            with self.cpu_wave('final',final_phases):
                for o in ids:
                    a=by_owner[o];measurements[o]={}
                    # One store per owner/phase, shared across candidates. Complete
                    # model/runtime/catalog/control/basis/scoring keys and byte
                    # readback remain enforced by ProjectionStore.
                    projection_store=E.ProjectionStore(self.session_dir/('projections_'+o))
                    projection_stores[o]=projection_store
                    for name,item in items[o].items():
                        if o+'/'+name in conditional:continue
                        def evaluate(out,a=a,item=item):
                            out.mkdir()
                            model=O.load_final_model(item,a)
                            return E.evaluate_model(model,a,out/'evaluation',self.check,phase='final',model_sha256=O.measurement_identity(item),
                                wrapper_steps=item.wrapper_steps,projection_store=projection_store)
                        measurements[o][name]=self.phase('evaluations/'+o+'/'+name,evaluate,
                            lambda out,a=a:R.evaluation(out/'evaluation',E.specification(a,'final'),self.check))
                selected_results=self.selected_after_opening(by_owner,selected_final,selected_pre,projection_stores)
            request=self.evidence.substitution(reference)
            self.phase('shared/F64',lambda out:T.substitution_tasks(request,out,self.check))
            if self.graph['P256_task_enabled']:
                request=self.hold(self.evidence.p256(reference))
                self.phase('shared/P256',lambda out:P.final_run(reference,request,out,self.check),P.readback)
            with self.cpu_wave('D2',sorted(D2.required_products(self.graph))):
                self.conditional_evaluations(by_owner,roster,items,measurements,projection_stores)
            unavailable=[]
            for o in ids:
                a=by_owner[o];parent=bound.marked_models.get(o)
                if bound.qualifications.get(o) is None or parent is None or parent.qualification is None:
                    unavailable.append(dict(owner=o,kind='keyed_nulls',reason='missing accepted qualification'))
                else:
                    candidates={'O/'+j:(by_owner[j],items[j]['O']) for j in ids if j!=o}
                    candidates.update({'Rstar/'+j:(by_owner[j],items[j]['Rstar']) for j in ids})
                    candidates['S/'+o]=(a,items[o]['S'])
                    self.phase('nulls/'+o+'/keyed',lambda out,a=a,p=parent,c=candidates:
                        nulls.keyed_grid(a,p.qualification_path,p.qualification,c,out,self.check,store=projection_stores[o]))
                    self.phase('nulls/'+o+'/F64',lambda out,a=a,p=parent:
                        nulls.substitution_null(a,p.qualification_path,p.qualification,out,self.check,store=projection_stores[o]))
                for label,group in self.graph['b2_groups'][o].items():
                    if group['missing']:
                        unavailable.append(dict(owner=o,kind='B2',model=label,missing=group['missing']));continue
                    full=o+'/'+label;names=[full]+group['matched']+group['independent']
                    # Additive noise has the unchanged pathwise Jacobian. B2
                    # uses that exact derivative, not a restarted noisy service.
                    def candidate(name):
                        owner,label=name.split('/');actual=label.rsplit('_',1)[1] if label.startswith(('noise_small_','noise_large_')) else label
                        return by_owner[owner],items[owner][actual]
                    candidates={name:candidate(name) for name in names}
                    self.phase('b2/'+o+'/'+label,lambda out,a=a,c=candidates,g=group,f=full:
                        nulls.b2_group(a,c,g['matched'],g['independent'],f,out,self.check))
            details=self.readouts(admissions,roster,items,measurements)
            details['selected_contracts']=selected_results
            details['selected_ineligible']=self.graph['contract_unavailable']
            details['selected_omitted']=self.graph['contract_omitted']
            details['noise_claim_scope']=F.C.NOISE_CLAIM
            for job in self.graph['contract_jobs']:
                if job['arm']!='noise' or job['parent']!='M3':continue
                o=job['owner'];a=by_owner[o]
                route=next(r for r in roster[o]['descendants'] if r['variant']=='ordinary')
                details[o]['removal'][job['label']]=O.final_descendant_readout(dict(route,variant=job['variant']),
                    selected_results[job['id']],self.hold(self.evidence.final_clean(a,measurements[o]['O'])),
                    self.hold(self.evidence.final_intact(a,measurements[o]['M3'])),a)
            details['unavailable']=unavailable
            details['primary']=analysis.primary_quality(reference,self.hold(self.evidence.paired(self.graph,measurements)))
            owner_details={o:details[o] for o in ids}
            details['assembled_readouts']=analysis.assembled_readouts(self.graph,owner_details,details['primary'],measurements)
            F.completion(self.graph,[r['name'] for r in self.products])
            status='CORE_COMPLETE' if not unavailable and not self.graph['missing_primary'] else 'CORE_INCOMPLETE'
        except MissingEvidence as exc:status='WAITING_ACCEPTANCE';error=str(exc)
        except SessionBoundary as exc:status='PAUSED_SESSION';error=str(exc)
        except BaseException as exc:error=repr(exc)
        result=dict(status=status,error=error,core_complete=status=='CORE_COMPLETE',complete_programme=False,
            selected_complete=status in ('CORE_COMPLETE','CORE_INCOMPLETE'),
            binding=self.binding,session_sha256=self.session.record.sha256,products=[r['name'] for r in self.products],
            product_receipts={r['name']:sha(self.root/'products'/(G.identity(r['name'])+'.json')) for r in self.products},
            graph_sha256=None if self.graph is None else self.graph['sha256'],details=details,
            planned_owners=self.config.get('owners'),independent_acceptance=False,follow_on_authorized=False,
            phase_storage_sha256=None if self.phase_store is None else self.phase_store.record.sha256)
        result['cpu_wave_records']=getattr(self,'cpu_wave_records',{})
        write(self.session_dir/'RESULT.json',result)
        # Prior immutable session results survive every latest-pointer update.
        write(self.root/'RESULT.json',result,replace=(self.root/'RESULT.json').exists())
        return result

    def conditional_evaluations(self,by_owner,roster,items,measurements,stores):
        """Actual D2 campaign path. All diagnostics precede each whole-cell task decision."""
        for key,cell in self.graph.get('d2_cells',{}).items():
            diagnostics={};contexts={}
            for member in cell['members']:
                o,name=member.split('/');a=by_owner[o];item=items[o][name]
                binding=E.evaluation_binding(a,'final',O.measurement_identity(item),item.wrapper_steps)
                def run(out,a=a,item=item,o=o):
                    out.mkdir();model=O.load_final_model(item,a)
                    return E.evaluate_diagnostics(model,a,out/'evaluation',self.check,model_sha256=O.measurement_identity(item),
                        wrapper_steps=item.wrapper_steps,projection_store=stores[o])
                diagnostics[member]=self.phase('diagnostics/'+member,run,
                    lambda out,a=a,b=binding:R.diagnostics(out/'evaluation',E.specification(a,'final'),self.check,expected_binding=b))
                if member in cell['marked_members']:
                    route=next(r for r in roster[o]['descendants'] if r['variant']=='ordinary')
                    contexts[member]=O.final_comparison_context(dict(route,variant=cell['family']),
                        self.hold(self.evidence.final_clean(a,measurements[o]['O'])),
                        self.hold(self.evidence.final_intact(a,measurements[o]['M3'])),a)
            controls={n:measurements[n.split('/')[0]][n.split('/')[1]] for n in cell['controls']}
            shared={n:sha(self.phase_path('shared/'+n)/'PHASE_VALUE.json') for n in cell['shared_controls']}
            decision=D2.decide(cell,diagnostics,contexts,controls,shared)
            def record(out):out.mkdir();write(out/'DECISION.json',decision);return decision
            self.phase('d2/'+key,record,lambda out:D2.read_decision(out,decision))
            for member in cell['members']:
                o,name=member.split('/');a=by_owner[o];item=items[o][name]
                diagnostic_root=self.phase_path('diagnostics/'+member)/'evaluation'
                binding=E.evaluation_binding(a,'final',O.measurement_identity(item),item.wrapper_steps)
                def run(out,a=a,item=item,dr=diagnostic_root,n=member):
                    out.mkdir();kwargs=dict(model_sha256=O.measurement_identity(item),diagnostic_root=dr,wrapper_steps=item.wrapper_steps)
                    if decision['task_required']:
                        return E.evaluate_tasks(O.load_final_model(item,a),a,out/'evaluation',self.check,**kwargs)
                    return E.skip_tasks(a,out/'evaluation',self.check,decision=decision,member=n,**kwargs)
                measurements[o][name]=self.phase('evaluations/'+member,run,
                    lambda out,a=a,b=binding,n=member:R.conditional_evaluation(out/'evaluation',E.specification(a,'final'),self.check,
                        expected_binding=b,decision=decision,member=n))

    def contract_configuration(self,bound):
        """Deferred contract controls/memberships bind only via one accepted later record."""
        if not F.pending(self.config):return self.config
        record=self.evidence.contract_bindings(bound.roster_acceptance)
        value=require_accepted(record,kind=F.BINDING_KIND,bindings=dict(study_id=self.session.value['study_id'],
            configuration_sha256=self.configuration.sha256,roster_acceptance_sha256=bound.roster_acceptance.sha256))
        self.hold(record)
        return F.bind(self.config,value.get('selected_contracts'))

    def selected_requests(self,phase,admissions,jobs,products):
        if not jobs:return {}
        requests=self.evidence.selected(phase,admissions,self.graph,products)
        if not isinstance(requests,dict) or set(requests)!={j['id'] for j in jobs}:raise ValueError('selected request membership incomplete/extra')
        by={a.prerequisites.owner_id:a for a in admissions}
        for j in jobs:
            r=requests[j['id']]
            if type(r) is not F.Request:raise ValueError('typed accepted selected request required')
            parent=self.graph['models'][j['owner']][j['parent']]
            F.request_metadata(r,j,by[j['owner']],sha(self.model_path(parent)),'qualification' if phase=='pre' else 'final')
            self.hold(r.acceptance)
            if r.calibration is not None:self.hold(r.calibration)
        return requests

    def selected_before_opening(self,admissions):
        jobs=[j for j in self.graph['contract_jobs'] if j['preopening']]
        requests=self.selected_requests('pre',admissions,jobs,{})
        by={a.prerequisites.owner_id:a for a in admissions};results={}
        store=E.ProjectionStore(self.session_dir/'selected_pre_projections') if jobs else None
        for j in jobs:
            a=by[j['owner']];r=requests[j['id']]
            value=self.phase(j['path']+'/pre',lambda out,r=r,j=j,a=a:F.execute(r,j,a,out,self.check,phase='qualification',store=store),
                lambda out,j=j,a=a:F.readback(out,j,'qualification',a,self.check))
            product=self.phase_path(j['path']+'/pre')/('mark/MODEL.pt' if j['arm']=='overwrite' else 'SELECTION.json')
            results[j['id']]=dict(value=value,path=str(product),sha256=sha(product),request_sha256=r.sha256,
                acceptance_sha256=r.acceptance.sha256,calibration_sha256=r.calibration.sha256)
            if j['arm']=='adaptive_accuser':
                value=read(Path(r.root)/'CONTRACT_INPUTS.json')
                results[j['id']]['fixed_files']={n:value['files'][n]['sha256'] for n in ('Rstar','keys','directions','basis','normalization','calibration')}
        return results

    def selected_final_bindings(self,admissions,pre):
        jobs=[j for j in self.graph['contract_jobs'] if j['arm']!='overwrite']
        requests=self.selected_requests('final',admissions,jobs,pre);binding={};streams=set()
        for j in self.graph['contract_jobs']:
            b=dict(job=j,pre=pre.get(j['id']))
            if j['id'] in requests:
                r=requests[j['id']]
                value=read(Path(r.root)/'CONTRACT_INPUTS.json')
                if j['arm'] in ('noise','API'):
                    key=(j['arm'],j['owner'],value['received_model_sha256'],j.get('policy'),j.get('sd'))
                    if key in streams:raise ValueError('duplicate phase-persistent client/model stream')
                    streams.add(key)
                if j['arm']=='adaptive_accuser' and value.get('selection_sha256')!=pre[j['id']]['sha256']:raise ValueError('selection substitution before opening')
                if j['arm']=='adaptive_accuser' and {n:value['files'][n]['sha256'] for n in pre[j['id']]['fixed_files']}!=pre[j['id']]['fixed_files']:
                    raise ValueError('adaptive reference/key/normalization/calibration changed after selection')
                b.update(request_sha256=r.sha256,acceptance_sha256=r.acceptance.sha256,
                    calibration_sha256=None if r.calibration is None else r.calibration.sha256)
            binding[j['id']]=b
        return requests,binding

    def selected_after_opening(self,by,requests,pre,stores):
        results={}
        for j in self.graph['contract_jobs']:
            if j['arm']=='overwrite':continue
            a=by[j['owner']];r=requests[j['id']]
            selection=Path(pre[j['id']]['path']) if j['arm']=='adaptive_accuser' else None
            results[j['id']]=self.phase(j['path']+'/final',
                lambda out,r=r,j=j,a=a,s=selection:F.execute(r,j,a,out,self.check,phase='final',selection=s,store=stores[j['owner']]),
                lambda out,j=j,a=a:F.readback(out,j,'final',a,self.check))
        if set(results)!={j['id'] for j in self.graph['contract_jobs'] if j['arm']!='overwrite'}:raise ValueError('selected final result membership differs')
        return results

    def admit_additional(self,admissions,roster,bound):
        if not self.graph['additional_jobs']:return ()
        additional=tuple(self.evidence.additional(admissions,roster))
        expected={(j['owner'],j['parent'],j['variant']) for j in self.graph['additional_jobs']}
        keys=[(r.admission.prerequisites.owner_id,r.parent.kind,r.variant) for r in additional]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('additional training roster incomplete/duplicated')
        admitted={a.prerequisites.owner_id:a for a in admissions}
        for r in additional:
            X.admit(r);self.hold(r.training);self.hold(r.parent.acceptance)
            o=r.admission.prerequisites.owner_id
            if r.admission.bindings!=admitted[o].bindings:raise ValueError('additional candidate/input drift')
            if r.parent.kind=='M3' and r.parent.sha256!=bound.marked_models[o].sha256:raise ValueError('additional marked parent substitution')
            X.available_inputs(r)
        # Mixed-output endpoint dependencies are produced by the core loop and
        # remain bound/checked at reconstruction; they are not required early.
        return additional

    def mark_readback(self,out):
        r=read(out/'RESULT.json')
        source=CPU.source_root(out)
        if r.get('marked_checkpoint_complete') is True:
            R.tensor_file(source/'MODEL.pt',r['model_sha256'],self.check)
            R.tensor_file(source/'STATE.pt',r['state_sha256'],self.check)
            for i in range(1,321):
                self.check();row=read(source/f'update_{i:03d}.json')
                if row.get('update')!=i or row.get('pre_state')!=i-1 or row.get('controls')!=128:
                    raise ValueError('marking raw count/state differs')
            if r.get('committed_mark_updates')!=320:raise ValueError('marking update reconciliation differs')
            if r.get('qualification_available') is True:
                R.evaluation(out/'evaluation',self.config['evaluation']['qualification'],self.check)
        return r

    def model_path(self,descriptor):
        name=descriptor['trajectory'];member=descriptor['checkpoint']+'.pt' if descriptor['checkpoint']=='MODEL' else descriptor['checkpoint']+'.model.pt'
        if self.phase_store is not None and self.phase_store.archived(name):
            receipt=self.root/'products'/(G.identity(name)+'.json')
            return self.phase_store.member(name,read(receipt),sha(receipt),member)
        return self.phase_path(name)/member

    def descriptors(self):
        values={}
        for o,models in self.graph['models'].items():
            values[o]={}
            for name,d in models.items():
                desc=dict(model_sha256=sha(self.model_path(d)),wrapper_steps=d['wrapper_steps'])
                if d.get('procedure') is not None:
                    desc.update(procedure=d['procedure'],dependencies={k:sha(self.model_path(v)) for k,v in d['dependencies'].items()})
                values[o][name]=desc
        return values

    def readouts(self,admissions,roster,items,measurements):
        result={}
        for a in admissions:
            o=a.prerequisites.owner_id;result[o]=dict(qualification=roster[o]['B3_qualified'],removal={},P2={})
            for name in ('Rstar','Rraw1920','Aord_M3_raw1920','Aord_O_raw1920'):
                result[o]['P2'][name]=measurements[o].get(name,{'available':False,'reason':'missing planned marked checkpoint'})
            if not roster[o]['B3_qualified']:continue
            clean=self.hold(self.evidence.final_clean(a,measurements[o]['O']))
            intact=self.hold(self.evidence.final_intact(a,measurements[o]['M3']))
            routes={r['variant']:r for r in roster[o]['descendants']}
            for name,d in self.graph['models'][o].items():
                if d.get('procedure') is not None:
                    if d.get('source_parent')!='M3':continue
                    v=d['procedure']['variant']
                    # Use the already independently locked owner's route, not
                    # a newly trusted read of mutable qualification JSON.
                    route=dict(routes['ordinary'],variant=v)
                    result[o]['removal'][name]=O.final_descendant_readout(route,measurements[o][name],clean,intact,a)
                    continue
                if d.get('selected_contract')=='overwrite':
                    if d['source_parent']=='M3':
                        result[o]['removal'][name]=O.final_descendant_readout(dict(routes['ordinary'],variant='overwrite'),measurements[o][name],clean,intact,a)
                    continue
                if d['wrapper_steps'] is not None and ('M3' in name):v='jacobi'+str(d['wrapper_steps'])
                elif name.startswith('Aord_M3_'):v='ordinary'
                elif name.startswith('A4_M3_'):v='A4'
                elif name.startswith('A5B3_M3_'):v='A5B3'
                else:continue
                result[o]['removal'][name]=O.final_descendant_readout(routes[v],measurements[o][name],clean,intact,a)
        return result
