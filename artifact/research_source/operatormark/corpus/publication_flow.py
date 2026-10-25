"""Single-use injected simulation coordinator. No native provider, I/O, or process launch."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol
from operatormark.corpus import publication_physical_records as records
from operatormark.corpus import publication_physical_verify as verifier


class PublicationFlowError(ValueError):
    def __init__(self,message,*,bootstrap_status='invalid',diagnostics=()):
        super().__init__(message)
        self.bootstrap_status=bootstrap_status
        self.diagnostics=tuple(verifier._freeze(x)for x in diagnostics)


class PublicationPorts(Protocol):
    def check_scope(self,checkpoint): ...
    def prepare(self,step,evidence_files): ...
    def retain_intent(self,step,data): ...
    def read_retained(self,file_ref): ...
    def perform(self,step,intent_bytes): ...
    def observe(self,phase): ...
    def retain_result(self,step,data): ...
    def release_resources(self,intent_bytes): ...


class _Stopped(Exception):
    def __init__(self,status):
        self.status=status


class PublicationCoordinator:
    """One in-memory use only; not a durable replay latch or an authorization issuer."""
    def __init__(self,*,expected,entry_files,ports):
        self.expected=expected
        self.entry_files=entry_files
        self.ports=ports
        self._used=False
        self._files={}
        self._operations=[]
        self._failures=[]
        self._comparison=None
        self._located='unknown'
        self._attempted=set()
        self._dispatched=set()
        self._uncertain=False
        self._root_bytes=None

    def _envelopes(self):
        return tuple(verifier._freeze(self._files[k])for k in sorted(self._files))

    def _merge(self,collection):
        if type(collection)is not tuple:
            raise _Stopped('unresolved')
        allowed={x['role']for x in verifier._plain(self.expected.accepted_input_files)}
        allowed|={r for p in verifier._plain(self.expected.phase_requirements)for r in p['required_evidence_roles']}
        for item in collection:
            if type(item)is not MappingProxyType:raise _Stopped('unresolved')
            row=verifier._plain(item)
            try:
                verifier._obj(row,'role file data')
                verifier._ref(row['file'],positive=False)
                verifier._require(row['role']in allowed and (row['data']is None or type(row['data'])is bytes),'file envelope')
                if row['data']is not None:
                    verifier._require(len(row['data'])==row['file']['file_bytes']and verifier._digest(row['data'])==row['file']['file_sha256'],'actual reader bytes')
            except (verifier.PublicationComparisonError,KeyError,TypeError):
                raise _Stopped('unresolved')
            old=self._files.get(row['role'])
            if old is not None and old!=row:
                self._uncertain=True
                raise _Stopped('unresolved')
            if row['role']=='root_bindings'and self._root_bytes is not None and row['data']!=self._root_bytes:
                self._uncertain=True
                raise _Stopped('unresolved')
            self._files[row['role']]=row

    def _raw_readback(self,ref,data=None):
        # Authority for this read is the independently accepted entry reader premise.
        # It is not bootstrapped from the new scope/ack being examined.
        try:
            projection=self.ports.read_retained(verifier._freeze(ref))
            self._merge(projection)
        except _Stopped:raise
        except Exception:
            self._uncertain=True;raise _Stopped('unresolved')
        matches=[r for r in self._files.values()if r['file']==ref and r['data']is not None]
        if not matches or data is not None and not any(r['data']==data for r in matches):
            self._uncertain=True;raise _Stopped('unresolved')

    def _scope(self,checkpoint):
        try:
            result=verifier._plain(self.ports.check_scope(checkpoint))
            verifier._obj(result,'status evidence_file')
            verifier._enum(result['status'],('active','refused','unavailable'))
            if result['evidence_file']is not None:verifier._ref(result['evidence_file'])
            if result['status']=='unavailable'or result['evidence_file']is None:
                self._uncertain=True
            if result['evidence_file']is not None:self._raw_readback(result['evidence_file'])
            if result['status']!='active'and result['evidence_file']is not None:
                self._failures.append(result['evidence_file'])
            if result['status']=='unavailable'or result['evidence_file']is None:
                raise _Stopped('unresolved')
            if result['status']=='refused':
                raise _Stopped('refused')
        except _Stopped:raise
        except Exception:
            self._uncertain=True;raise _Stopped('unresolved')

    def _retention(self,step,data,result=False):
        self._scope('before_intent')
        try:
            ack=verifier._plain((self.ports.retain_result if result else self.ports.retain_intent)(step,data))
            verifier._obj(ack,'call_status file failure')
            verifier._enum(ack['call_status'],('not_called','returned_success','returned_failure','outcome_unavailable'))
            if ack['file']is not None:verifier._ref(ack['file'],positive=False)
            if ack['failure']is not None:
                verifier._failure(ack['failure'])
                if ack['failure']['exception_file']is not None:self._failures.append(ack['failure']['exception_file'])
            # Validate operation facts before deciding how to treat the actual acknowledgment.
            if ack['call_status']!='returned_success'or ack['file']is None or ack['failure']is not None:
                if ack['file']is not None:
                    self._scope('before_evidence_read')
                    self._raw_readback(ack['file'])
                    self._failures.append(ack['file'])
                self._uncertain=True;raise _Stopped('unresolved')
            if ack['file']['file_bytes']!=len(data)or ack['file']['file_sha256']!=verifier._digest(data):
                self._uncertain=True;raise _Stopped('unresolved')
            slot=self._slot(step,'result'if result else 'intent')
            if verifier._address_key(ack['file'])!=verifier._address_key(slot):
                self._uncertain=True;raise _Stopped('unresolved')
            self._scope('before_evidence_read')
            self._raw_readback(ack['file'],data)
            return ack['file']
        except _Stopped:raise
        except Exception:
            self._uncertain=True;raise _Stopped('unresolved')

    def _slot(self,step,kind):
        # Locations are resolved from the fixed L root, not from root-ID spelling.
        location=self._bootstrap.plan['publication_evidence_root_relative_path']+'/test.sealed/operations/'+step+'.'+kind+'.json'
        slots=[x['address']for x in self._bootstrap.template['slots']if x['role']==kind and verifier._resolve_publication_address(x['address'],self._bootstrap.roots)==('repository',location)]
        if len(slots)!=1:raise _Stopped('unresolved')
        return slots[0]

    def _gate(self,phase):
        self._located='unknown'
        self._scope('before_observation')
        try:self._merge(self.ports.observe(phase))
        except _Stopped:raise
        except Exception:
            self._uncertain=True;raise _Stopped('unresolved')
        try:
            comparison=verifier.verify_publication_evidence(phase=phase,expected=self.expected,
                candidate_files=(),independent_files=self._envelopes())
        except verifier.PublicationComparisonError:
            self._uncertain=True;raise _Stopped('unresolved')
        self._comparison=comparison
        self._located=comparison['report']['located_root']
        if comparison['report']['comparison_status']!='matched':
            raise _Stopped('unresolved'if comparison['report']['comparison_status']=='incomplete'else 'failed')
        return comparison

    def _step(self,step):
        if step in self._attempted:raise _Stopped('unresolved')
        self._attempted.add(step)
        for kind in ('intent','result'):
            slot=self._slot(step,kind)
            if any(verifier._address_key(r['file'])==verifier._address_key(slot)for r in self._files.values()):
                raise _Stopped('unresolved')
        self._scope('before_prepare')
        try:
            supplied=self.ports.prepare(step,self._envelopes())
            verifier._require(type(supplied)in (dict,MappingProxyType) and type(supplied.get('independent_files'))is tuple,'frozen independent prepared projections')
            prepared=verifier._plain(supplied)
            verifier._obj(prepared,'prepared_files independent_files')
            # Preserve the explicitly frozen independently read projections, not thawed copies.
            original=self.ports_prepare_independent(prepared['independent_files'])
            self._merge(original)
            verifier._require(type(prepared['prepared_files'])is list,'prepared rows')
            rows=[];destinations=[]
            allowed={verifier._resolve_publication_address(a,self._bootstrap.roots,repository_only=True)
                     for p in verifier._plain(self.expected.phase_requirements)for a in p['allowed_new']}
            for row in prepared['prepared_files']:
                verifier._obj(row,'destination file data');verifier._address(row['destination']);verifier._ref(row['file'],positive=False)
                verifier._require(type(row['data'])is bytes and len(row['data'])==row['file']['file_bytes']and verifier._digest(row['data'])==row['file']['file_sha256'],'prepared byte identity')
                loc=verifier._resolve_publication_address(row['destination'],self._bootstrap.roots,repository_only=True)
                verifier._require(loc in allowed,'exact independent writer destinations')
                # Independently acquired prepared bytes must corroborate the preparation port.
                matches=[x for x in self._files.values()if x['file']==row['file']and x['data']==row['data']]
                verifier._require(bool(matches),'independent prepared bytes unavailable')
                destinations.append(loc);rows.append(dict(destination=row['destination'],file=row['file']))
            verifier._require(len(set(destinations))==len(destinations),'duplicate prepared destinations')
            rows.sort(key=lambda r:verifier._address_key(r['destination']))
        except _Stopped:raise
        except Exception:
            self._uncertain=True;raise _Stopped('unresolved')
        plan=self._bootstrap.plan;roots=self._bootstrap.roots
        source=destination=None
        if step=='promote':
            source=verifier._address_for(plan['attempt_root_relative_path'],roots)
            destination=verifier._address_for(plan['final_root_relative_path'],roots)
        elif step=='quarantine':
            if self._located not in ('T','F'):raise _Stopped('unresolved')
            source=verifier._address_for(plan['attempt_root_relative_path'if self._located=='T'else 'final_root_relative_path'],roots)
            destination=verifier._address_for(plan['quarantine_root_relative_path'],roots)
        elif len(rows)==1:destination=rows[0]['destination']
        preconditions={verifier._canonical(r['file']):r['file']for r in self._files.values()if r['data']is not None and r['file']['file_bytes']>0}
        body=dict(schema=verifier._PREFIX+'publication_operation_intent.v1',binding=verifier._plain(self.expected.binding),
            operation_id=verifier._invocation(verifier._plain(self.expected.binding))+':'+step,step=step,sequence=len(self._operations),
            previous_result_file=self._operations[-1]['result_file']if self._operations else None,
            precondition_files=[preconditions[k]for k in sorted(preconditions)],source_address=source,destination_address=destination,prepared_files=rows)
        intent=verifier._seal(body)
        try:data=records.canonical_publication_record_bytes(intent,expected_schema=intent['schema'])
        except records.PublicationRecordError:raise _Stopped('unresolved')
        ref=self._retention(step,data)
        operation=dict(step=step,intent_file=ref,result_file=None)
        self._operations.append(operation)
        self._scope('before_dispatch')
        self._dispatched.add(step)
        if step in ('promote','quarantine'):
            self._located='unknown'
        try:
            returned=self.ports.release_resources(data)if step=='release_resources'else self.ports.perform(step,data)
            verifier._require(type(returned)in (dict,MappingProxyType),'operation envelope')
            verifier._require(set(returned)=={'result','independent_files'},'operation result and independent evidence')
            self._merge(returned['independent_files'])
            result=verifier._plain(returned['result'])
            records.validate_publication_operation_result(result)
            verifier._require(result['binding']==verifier._plain(self.expected.binding)and result['step']==step and result['intent_file']==ref,'result binding')
            result_bytes=records.canonical_publication_record_bytes(result,expected_schema=result['schema'])
        except _Stopped:raise
        except Exception:
            self._uncertain=True;raise _Stopped('unresolved')
        # Preserve the actual returned projection; never manufacture a successful finish/exit.
        self._scope('after_dispatch')
        result_ref=self._retention(step,result_bytes,True)
        operation['result_file']=result_ref
        if result['call_status']!='returned_success'or result['effect_status']in ('unknown','unexpected_effect_observed')or result['failures']:
            self._failures.append(result_ref)
            if result['call_status']=='outcome_unavailable'or result['effect_status']=='unknown':self._uncertain=True
            raise _Stopped('unresolved'if self._uncertain else 'failed')

    @staticmethod
    def ports_prepare_independent(value):
        # A plain conversion is used only for validation; restore an immutable copied envelope.
        if type(value)is not list:raise _Stopped('unresolved')
        return tuple(verifier._freeze(x)for x in value)

    def _failure_path(self):
        try:
            comparison=self._gate('failure_location')
            safe=comparison['report']['located_root']in ('T','F')
        except _Stopped as stop:
            if stop.status=='unresolved':self._uncertain=True
            safe=False
            if self._comparison is not None and self._comparison['report']['phase']=='failure_location':
                details=self._comparison['sealed_details']
                codes={x['code']for k in ('mismatches','invalid_inputs','unavailable_requirements')for x in details[k]}
                safe=self._located in ('T','F')and codes<={'operation_outcome_unavailable'}
        if safe and 'quarantine'not in self._attempted:
            try:self._step('quarantine')
            except _Stopped as stop:
                if stop.status=='unresolved':self._uncertain=True
            if 'quarantine'in self._dispatched:
                try:self._gate('post_quarantine')
                except _Stopped as stop:
                    if stop.status=='unresolved':self._uncertain=True
        if 'release_resources'not in self._attempted:
            try:self._step('release_resources')
            except _Stopped as stop:
                if stop.status=='unresolved':self._uncertain=True

    def run(self):
        if self._used:raise PublicationFlowError('coordinator already used; no retry/resume')
        self._used=True
        try:self._bootstrap=verifier._bootstrap_entry(self.expected,self.entry_files)
        except verifier.PublicationComparisonError as exc:
            raise PublicationFlowError('independent entry bootstrap failed',bootstrap_status=exc.bootstrap_status,diagnostics=exc.diagnostics)from exc
        self._root_bytes=self._bootstrap.independent['root_bindings']['data']
        self._files=dict(self._bootstrap.independent)
        status='completed'
        try:
            self._scope('entry')
            # Reject any preoccupied invocation output namespace before forward dispatch.
            output_addresses={verifier._address_key(s['address'])for s in self._bootstrap.template['slots']if s['role']in ('intent','result','terminal')}
            if any(verifier._address_key(row['file'])in output_addresses for row in self._files.values()):raise _Stopped('refused')
            self._gate('pre_finalize')
            self._step('finalize_metadata');self._gate('pre_move')
            self._step('promote');self._gate('post_move')
            self._step('publish_audit');self._gate('post_audit')
            self._step('publish_complete');self._step('finalize_protection');self._gate('post_complete')
            self._step('release_resources')
            self._scope('before_terminal_return')
        except _Stopped as stop:
            status=stop.status
            if stop.status=='unresolved':self._uncertain=True
            if self._operations or self._attempted:
                self._failure_path()
            if self._uncertain or any(x['result_file']is None for x in self._operations):status='unresolved'
            elif self._operations:status='failed'
        return verifier._freeze(dict(binding=verifier._plain(self.expected.binding),operation_files=self._operations,
            last_comparison=self._comparison,failure_evidence_files=self._failures,located_root=self._located,
            coordinator_return_status=status))


def publish_corpus_once(*,expected,entry_files,ports):
    return PublicationCoordinator(expected=expected,entry_files=entry_files,ports=ports).run()
