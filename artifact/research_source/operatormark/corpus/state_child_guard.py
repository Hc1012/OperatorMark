"""Source-bound cooperative State child observations, not an OS sandbox.

Install before project entry. Keep hooks installed until os._exit; the parent
must independently observe exit/reaping and qualify all retained records. No
child message establishes through-exit coverage or production acceptance alone.
Foreign native code and arbitrary in-process tampering require the existing
reviewed-source/runtime premises; neither is inferred from a hook inventory.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import os
import sys
import threading
import _thread
from . import state_execution_authority as a
from . import state_guard_capture as g


class GuardViolation(RuntimeError):pass


# Authorized fail-fast: an already-invalid child ends on its FIRST guard violation.
VIOLATION_EXIT_CODE=86
VIOLATION_RECORD='guard_violation.json'


@dataclass(frozen=True)
class GuardPolicy:
    # Supplied by the independently bound bootstrap, never from child input.
    source_commitment: str
    required_hooks: tuple
    denied_roots: tuple  # (family, canonical absolute path)
    denied_modules: tuple  # (family, prefix); model/legacy/test roots
    bootstrap_files: tuple  # exact reviewed pre-entry project file paths/hashes
    authorized_counts: tuple = ()  # ((family,split), maximum) from bound State role
    authorization_file_sha256: str | None = None


HOOKS=tuple(sorted(('audit_files','audit_imports','audit_network','audit_native',
       'monitor_python','monitor_native','rng_modules','thread_creation',
       'source_import_correspondence','controlled_exit')))


class ChildGuard:
    def __init__(self,store,clock,slot,creation_file,policy,rng,source_integrity):
        a.need(type(policy) is GuardPolicy and policy.required_hooks==HOOKS,'complete fixed hook inventory')
        a.digest(policy.source_commitment);a.digest(slot);a.ref(creation_file)
        a.need(callable(source_integrity),'held-source integrity callback required')
        from .rng_guard import RngCallGuard
        a.need(type(rng) is RngCallGuard,'version-pinned RNG guard required')
        for family,path in policy.denied_roots:
            a.need(family in ('corpus_seed_access','legacy_data_access','unauthorized_test_access') and
                Path(path).is_absolute(),'family-specific forbidden root')
        a.need({f for f,_ in policy.denied_roots}==
            {'corpus_seed_access','legacy_data_access','unauthorized_test_access'},'all data families required')
        a.need({f for f,_ in policy.denied_modules}==
            {'model_execution','legacy_data_access','unauthorized_test_access'},'all module families required')
        self.store,self.clock,self.slot,self.creation,self.policy=store,clock,slot,creation_file,policy
        self.rng,self.source_integrity=rng,source_integrity
        self.previous=None;self.sequence=0;self.lock=threading.RLock();self.local=threading.local()
        self.failed=False;self.installed=False;self.events=[];self.checkpoints={};self.authorized=set()
        self._violating=False
        self._exit=os._exit;self._audit_callable=self.audit;self._call_callable=self.native_call;self._start_callable=self.python_start
        self._dispatch=(type(self).audit,type(self).native_call,type(self).python_start,type(self).deny,
            type(self).integrity,type(self).finish_and_exit,type(self).authorized_call,
            type(self).event,type(self).checkpoint,type(self)._probe,type(self).seed_once)
        self._policy=policy;self._source_integrity=source_integrity
        self._dispatch_codes=tuple(f.__code__ for f in self._dispatch)
        self._effect_reader=observe_rng_result
        self._audit_seen=False;self._native_seen=False;self._python_seen=False
        self._limits=dict(policy.authorized_counts)
        if self._limits:a.digest(policy.authorization_file_sha256)
        a.need(len(self._limits)==len(policy.authorized_counts),'duplicate authorized operation')
        for (family,split),count in self._limits.items():
            a.need(family in g.AUTHORIZED and split in ('training','validation','test') and
                type(count) is int and count>=0,'exact authorized-family population')
        self._counts={key:0 for key in self._limits};self._generators={};self._bit_generators={}

    def event(self,kind,family='lifetime',split=None,decision='allowed',call_status='not_called',effect='none_observed'):
        with self.lock:
            stamp=self.clock.acquire()
            data=g.make_guard_event(launch_slot_id=self.slot,observer_id=stamp.observer_id,
                clock_epoch_id=stamp.clock_epoch_id,sequence=self.sequence,kind=kind,operation=family,
                split=split,decision=decision,call_status=call_status,effect_status=effect,
                subject_file=self.creation,prior_event_file=self.previous)
            self.previous=self.store.put(f'guard_{self.sequence:08d}.json',data)
            self.events.append(self.previous);self.sequence+=1

    def deny(self,family):
        """Fail-fast: at most one denied event and one bounded violation record.

        Hooks stay installed; the held os._exit ends the child before any caller
        can catch and repeat a request. No exit-ready or lifetime evidence is
        written, so parent refusal and actual parent-observed closure still decide.
        """
        self.failed=True
        first=not self._violating;self._violating=True
        if first:
            written=None;failure=None
            try:
                # Denial precedes the call. No operation or physical effect is inferred.
                self.event('operation',family,decision='denied');written=self.previous
            except BaseException as exc:failure=type(exc).__name__[:128]
            try:self.store.json(VIOLATION_RECORD,dict(slot_id=self.slot,family=family,
                denied_event_file=written,denied_event_unavailable=failure,exit_code=VIOLATION_EXIT_CODE,
                exit_primitive='os._exit',fail_fast=True,lifetime_evidence_written=False,production_accepted=False))
            except BaseException:pass
            for stream in (sys.stdout,sys.stderr):
                try:stream.flush()
                except BaseException:pass
        self._exit(VIOLATION_EXIT_CODE)
        raise GuardViolation('denied '+family)  # only if the held primitive returned

    def audit(self,event,args):
        if not self.installed:return
        if event=='operatormark.state.guard_probe':
            if args==(self.slot,):self._audit_seen=True
            return
        if event.startswith(('socket.','http.client.')):self.deny('network_access')
        if event in ('ctypes.dlopen','ctypes.dlsym','subprocess.Popen','os.system','os.exec','os.posix_spawn'):
            self.deny('unauthorized_test_access')
        if event=='import' and type(args[0]) is str:
            for family,prefix in self.policy.denied_modules:
                if args[0]==prefix or args[0].startswith(prefix+'.'):self.deny(family)
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).absolute()
            for family,root in self.policy.denied_roots:
                if path==Path(root) or path.is_relative_to(root):self.deny(family)

    def native_call(self,code,offset,call,arg0):
        if not self.installed or getattr(self.local,'internal',False):return
        if call is len:self._native_seen=True
        target=getattr(call,'__func__',call)
        if any(target is x for x in self._thread_originals):self.deny('unauthorized_test_access')
        receiver=getattr(call,'__self__',None)
        module=getattr(type(receiver),'__module__','')
        rng_call=(any(target is getattr(x,'__func__',x) for x in self._rng_originals) or
                  module=='random' or module=='_random' or module.startswith('numpy.random'))
        if rng_call and not getattr(self.local,'authorized',False):self.deny('rng_calls')

    def python_start(self,code,offset):
        if not self.installed:return
        if code is type(self)._probe.__code__:self._python_seen=True
        filename=os.path.normcase(code.co_filename)
        for family,root in self.policy.denied_roots:
            if filename.startswith(os.path.normcase(root)+os.sep):self.deny(family)
        # Preloaded model code is covered as well as import attempts.
        for family,prefix in self.policy.denied_modules:
            marker=os.sep+prefix.replace('.',os.sep)+os.sep
            if marker in filename:self.deny(family)

    def checkpoint(self,name):
        a.need(name in g.CHECKPOINTS and name not in self.checkpoints,'unique RNG checkpoint')
        self.local.internal=True
        try:values=self.rng.state_commitments() if self.rng._installed else self._baseline()
        finally:self.local.internal=False
        self.checkpoints[name]=self.store.json('rng_'+name+'.json',dict(slot_id=self.slot,checkpoint=name,**values))

    def _baseline(self):
        from . import rng_guard as r
        return dict(numpy_legacy_sha256=r.numpy_legacy_state_commitment(r.np.random.get_state()),
            python_random_sha256=r.python_random_state_commitment(r.random.getstate()))

    def install(self):
        a.need(not self.installed and len(sys._current_frames())==1,'fresh single-Python-thread child required')
        actual={str(Path(m.__file__).resolve()) for name,m in tuple(sys.modules.items())
            if name.startswith(('operatormark.','operatormark_corpus_v2.')) and getattr(m,'__file__',None)}
        a.need(actual=={str(Path(path).resolve()) for path,_ in self.policy.bootstrap_files},
            'pre-entry project imports differ from reviewed bootstrap inventory')
        for path,digest in self.policy.bootstrap_files:
            a.need(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'bootstrap source drift')
        a.need(bool(self.policy.bootstrap_files),'reviewed bootstrap premises absent')
        for family,prefix in self.policy.denied_modules:
            a.need(not any(name==prefix or name.startswith(prefix+'.') for name in sys.modules),
                'forbidden module already loaded before guard: '+family)
        self.checkpoint('after_base_import')
        self._rng_observer=lambda name:self.deny('rng_calls')
        self.rng._on_violation=self._rng_observer
        self.rng.install()
        self._rng_originals=tuple(self.rng._originals.values())
        self._installed_rng={key:getattr(*key) for key in self.rng._originals}
        names=tuple(n for n in ('start_new_thread','start_new','start_joinable_thread') if hasattr(_thread,n))
        self._thread_originals=tuple(getattr(_thread,n) for n in names)+(threading.Thread.start,)
        def blocked(*args,**kw):return self.deny('unauthorized_test_access')
        self._thread_wrapper=blocked
        self._thread_bindings=tuple((_thread,n) for n in names)+((threading,'Thread'),)
        for n in names:setattr(_thread,n,blocked)
        threading.Thread.start=blocked
        self.tool=next((x for x in range(5,-1,-1) if sys.monitoring.get_tool(x) is None),None)
        a.need(self.tool is not None,'monitoring slot unavailable')
        sys.monitoring.use_tool_id(self.tool,'state-child-guard')
        sys.monitoring.register_callback(self.tool,sys.monitoring.events.CALL,self._call_callable)
        sys.monitoring.register_callback(self.tool,sys.monitoring.events.PY_START,self._start_callable)
        self.mask=sys.monitoring.events.CALL|sys.monitoring.events.PY_START
        sys.monitoring.set_events(self.tool,self.mask)
        sys.addaudithook(self._audit_callable)
        self.installed=True
        self._probe()
        a.need(self._audit_seen and self._native_seen and self._python_seen,'actual hook installation probe unavailable')
        self.policy_file=self.store.json('guard_policy.json',json.loads(json.dumps(asdict(self.policy))))
        self.event('coverage_start');self.checkpoint('after_guard_install')
        self.store.json('guard_installed.json',dict(source_commitment=self.policy.source_commitment,
            hooks=list(HOOKS),monitoring_mask=self.mask,thread_scope='all interpreter threads; new threads denied',
            terminal_boundary='held os._exit; no Python finalizers or exit handlers',production_accepted=False))
        return self

    def integrity(self):
        current=(type(self).audit,type(self).native_call,type(self).python_start,type(self).deny,
            type(self).integrity,type(self).finish_and_exit,type(self).authorized_call,
            type(self).event,type(self).checkpoint,type(self)._probe,type(self).seed_once)
        callbacks=(sys.monitoring.register_callback(self.tool,sys.monitoring.events.CALL,self._call_callable),
            sys.monitoring.register_callback(self.tool,sys.monitoring.events.PY_START,self._start_callable))
        self._audit_seen=self._native_seen=self._python_seen=False;self._probe()
        okay=(self.installed and self._audit_seen and self._native_seen and self._python_seen and
            self.rng._on_violation is self._rng_observer and current==self._dispatch and
            tuple(f.__code__ for f in current)==self._dispatch_codes and observe_rng_result is self._effect_reader and
            self.policy is self._policy and self.source_integrity is self._source_integrity and
            callbacks==(self._call_callable,self._start_callable) and _thread.start_new_thread is self._thread_wrapper and
            all(getattr(module,name) is self._thread_wrapper for module,name in self._thread_bindings if name!='Thread') and
            threading.Thread.start is self._thread_wrapper and sys.monitoring.get_events(self.tool)==self.mask and
            all(getattr(module,name) is wrapper for (module,name),wrapper in self._installed_rng.items()))
        if not okay:self.deny('unauthorized_test_access')
        try:self.source_integrity()
        except BaseException:
            self.failed=True
            self.store.json('guard_source_invalidation.json',dict(reason='held_source_integrity_failed',slot_id=self.slot))
            raise
        if self.failed:raise GuardViolation('sticky guard invalidation')

    def _probe(self):
        sys.audit('operatormark.state.guard_probe',self.slot)
        len(())

    def project_imported(self):
        self.integrity();self.checkpoint('after_project_import')

    def authorized_call(self,family,split,call,*args,**kwargs):
        """Only independently routed State 4 requests; no general allow bypass.

        Caller must pass a original pinned RNG entry. The seed provider is never
        enabled by this method: it needs the existing State4SeedCapability.
        """
        self.integrity()
        a.need(family in ('pcg64_constructors_by_split','generator_constructors_by_split','top_level_rng_calls_by_split')
            and split in ('training','validation','test'),'bound State4 RNG family')
        from . import rng_guard as r
        key=(family,split)
        if key not in self._limits or self._counts[key]>=self._limits[key]:self.deny('rng_calls')
        if family=='pcg64_constructors_by_split':
            a.need(call is self.rng._originals[(r.np.random,'PCG64')],'unbound PCG64 entry')
        elif family=='generator_constructors_by_split':
            a.need(call is self.rng._originals[(r.np.random,'Generator')] and len(args)==1 and
                args[0] is self._bit_generators.get(split),'unbound Generator/bit-generator entry')
        else:
            a.need(getattr(call,'__self__',None) is self._generators.get(split) and split in self._generators and
                getattr(call,'__name__',None) in r.GENERATOR_METHODS,'unbound split-generator method')
        self._counts[key]+=1;self.local.authorized=True
        try:
            result=call(*args,**kwargs)
        except BaseException:
            self.failed=True;self.event('operation',family,split,call_status='returned_failure',effect='unknown');raise
        finally:self.local.authorized=False
        # Family-specific observer must actually inspect the result; unknown is a refusal.
        effect=self._effect_reader(family,call,result,args,kwargs)
        self.event('operation',family,split,call_status='returned_success',effect=effect)
        if effect!='expected_effect_observed':self.failed=True;raise GuardViolation('authorized effect unavailable')
        if family=='pcg64_constructors_by_split':self._bit_generators[split]=result
        if family=='generator_constructors_by_split':self._generators[split]=result
        return result

    def seed_once(self,capability,split,provider):
        from .rng_guard import State4SeedCapability
        self.integrity()
        key=('seed_access_by_split',split)
        if (type(capability) is not State4SeedCapability or capability.authorization_file_sha256!=self.policy.authorization_file_sha256 or key not in self._limits or
            self._counts[key]>=self._limits[key]):self.deny('corpus_seed_access')
        self._counts[key]+=1
        try:value=capability.retrieve_once(split,provider)
        except BaseException:
            self.failed=True;self.event('operation','seed_access_by_split',split,
                call_status='returned_failure',effect='unknown');raise
        effect='expected_effect_observed' if type(value) is int and capability.access_count_by_split[split]==1 else 'unknown'
        self.event('operation','seed_access_by_split',split,call_status='returned_success',effect=effect)
        if effect=='unknown':self.failed=True;raise GuardViolation('seed acquisition unavailable')
        return value

    def finish_and_exit(self,code=0):
        """No stop/uninstall API. Evidence precedes immediate held primitive exit.

        Parent still requires actual exit/reaping and no missing child records.
        Timeout/kill does not manufacture this receipt.
        """
        self.integrity();a.need(self._exit is os._exit,'controlled exit primitive substitution')
        a.need(self._counts==self._limits,'prescribed authorized operation population incomplete')
        a.need(len(sys._current_frames())==1,'unexpected raw/Python thread at terminal boundary')
        self.checkpoint('after_execution')
        # Flush ordinary output while hooks and the parent's deadline are active.
        sys.stdout.flush();sys.stderr.flush()
        a.need(set(self.checkpoints)==set(g.CHECKPOINTS),'missing RNG checkpoint')
        self.store.json('guard_exit_ready.json',dict(slot_id=self.slot,source_commitment=self.policy.source_commitment,
            policy_file=self.policy_file,
            hooks=list(HOOKS),events=self.events,rng_checkpoint_files=self.checkpoints,failed=self.failed,
            exit_code=code,exit_primitive='os._exit',no_python_finalization=True))
        self._exit(code)


def observe_rng_result(family,call,result,args,kwargs):
    """Observe constructor identity/state or returned top-level array metadata.

    These are call/effect observations, not numerical or distribution qualification.
    Bootstrap binds this observer and the exact allowed entry/argument contract.
    """
    from . import rng_guard as r
    if family=='pcg64_constructors_by_split':
        try:return 'expected_effect_observed' if type(result) is call and result.state['bit_generator']=='PCG64' else 'unknown'
        except BaseException:return 'unknown'
    if family=='generator_constructors_by_split':
        try:return 'expected_effect_observed' if len(args)==1 and result.bit_generator is args[0] else 'unknown'
        except BaseException:return 'unknown'
    if family=='top_level_rng_calls_by_split':
        # Actual array result is observed, not an inferred seed/draw-count claim.
        return 'expected_effect_observed' if type(result) is r.np.ndarray and result.dtype.kind in 'fiu' else 'unknown'
    return 'unknown'


def retain_exit_observations(*,process,child_store,parent_store,ready_file,source_commitment,policy_sha256,observer_id,clock_epoch_id):
    """Parent-only composition AFTER actual held-process transport closure.

    Returns raw FileRefs for independent qualification, never accepted Acquisition
    records. The final guard consumer still re-reads and qualifies every input.
    """
    from .state_host_windows import ProcessObserver,stable_bytes
    a.need(type(process) is ProcessObserver and process.closed and process.transport is not None,
        'actual closed host transport required')
    a.need(not process.capture_overflow and process.transport.closure.reap_status=='observed_reaped',
        'incomplete process transport')
    data=stable_bytes(child_store.path(ready_file['relative_path']));a.check_bytes(ready_file,data);ready=a.decode(data)
    a.digest(policy_sha256)
    policy=stable_bytes(child_store.path(ready['policy_file']['relative_path']));a.check_bytes(ready['policy_file'],policy)
    a.need(a.sha(policy)==policy_sha256 and process.command.source_commitment==source_commitment,
        'independently bound guard policy/source differs')
    a.need(ready['slot_id']==process.prepared.slot_id and ready['source_commitment']==source_commitment and
        ready['failed'] is False and ready['hooks']==list(HOOKS) and ready['no_python_finalization'] is True and
        ready['exit_primitive']=='os._exit' and ready['exit_code']==process.transport.closure.direct_exit_code,
        'guard exit/held process contradiction')
    refs=list(ready['events']);previous=None
    for i,ref in enumerate(refs):
        raw=stable_bytes(child_store.path(ref['relative_path']));a.check_bytes(ref,raw);row=a.decode(raw);g.validate_guard_event(row)
        a.need(row['sequence']==i and row['prior_event_file']==previous,'child event sequence')
        a.need(row['effect_status']!='unknown' and row['decision']!='denied','invalidated or unknown guard event')
        previous=ref
    a.need(refs and set(ready['rng_checkpoint_files'])==set(g.CHECKPOINTS),'missing guard population')
    exit_ref=a.decode(process.transport.closure.evidence_files[0])
    for kind,subject in (('process_exit',exit_ref),('coverage_end',a.decode(process.owned.handle.creation_identity)['creation_observation_file'])):
        raw=g.make_guard_event(launch_slot_id=process.prepared.slot_id,observer_id=observer_id,clock_epoch_id=clock_epoch_id,
            sequence=len(refs),kind=kind,operation='lifetime',split=None,decision='allowed',call_status='not_called',
            effect_status='none_observed',subject_file=subject,prior_event_file=previous)
        previous=parent_store.put('guard_'+kind+'.json',raw);refs.append(previous)
    life=parent_store.json('guard_lifetime.json',dict(slot_id=process.prepared.slot_id,observer_id=observer_id,
        clock_epoch_id=clock_epoch_id,creation_file=a.decode(process.owned.handle.creation_identity)['creation_observation_file'],
        exit_file=exit_ref,first_sequence=0,last_sequence=len(refs)-1,started_before_project_import=True,
        through_direct_exit=True,hooks=list(HOOKS)))
    return dict(event_files=refs,lifetime_file=life,exit_file=exit_ref,rng_checkpoint_files=ready['rng_checkpoint_files'])
