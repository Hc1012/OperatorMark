"""Windows State host observations; never a production admission issuer.

Caller supplies genuine accepted bindings. Raw observations are durably retained
before validation. A/B/C retention and handoff rules do not issue premise-P admission.
"""
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import ctypes
import hashlib
import json
import os
import stat
import threading
import time

from . import state_execution_authority as a
from . import state_execution_windows as w
from . import state_execution as ex
from . import state_evidence_windows as ev
from . import state_provider_bindings as p
from . import state_guard_capture as guard
from . import state_storage_windows as storage
from . import state_storage_records as sr
from .state_host_native_win32 import NativeBindings
from .bounded_stream import StreamRetention, MAX_CHUNKS, QUOTA_BYTES, verify_retained


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def stable_bytes(path):
    """Reparse refusal and held-file/path correspondence, not continuous P."""
    path = Path(path).absolute()
    for component in (path, *path.parents):
        st = component.lstat()
        need(not (getattr(st, 'st_file_attributes', 0) & 0x400), 'reparse path')
    before = path.stat()
    need(stat.S_ISREG(before.st_mode), 'regular file required')
    with path.open('rb') as stream:
        held = os.fstat(stream.fileno())
        data = stream.read()
        after = os.fstat(stream.fileno())
    from .corpus_v2_custody import _identity, _cross_api_identity
    # Reuse the existing Windows 3.13.5 distinction: path ctime is birth time,
    # descriptor ctime is change time. Still check mutation within EACH API.
    need(_cross_api_identity(before)==_cross_api_identity(held) and
         _identity(held)==_identity(after) and _identity(before)==_identity(path.stat()),
         'file changed during acquisition')
    need(len(data) == held.st_size, 'incomplete acquisition')
    return data


class RecordStore:
    """Exclusive durable observation records, no ACL or acceptance mutations."""
    def __init__(self, root, root_binding_id):
        self.root = Path(root).absolute()
        need(self.root.is_dir(), 'existing evidence directory required')
        p.relative(root_binding_id)
        self.root_binding_id = root_binding_id

    def path(self, relative):
        p.relative(relative)
        path = self.root.joinpath(*relative.split('/'))
        need(path.is_relative_to(self.root), 'observation path escape')
        for part in (self.root, *path.parent.relative_to(self.root).parents):
            if part.is_absolute():
                need(not part.is_symlink(), 'reparse evidence directory')
        # The host never invents nested output directories.
        need(path.parent == self.root, 'flat observation destination required')
        need(not self.root.lstat().st_file_attributes & 0x400 if os.name == 'nt' else not self.root.is_symlink(),
             'reparse evidence root')
        return path

    def put(self, name, data):
        need(type(data) is bytes, 'raw bytes required')
        path = self.path(name)
        with path.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        need(stable_bytes(path) == data, 'observation readback mismatch')
        return a.file_ref(dict(root_binding_id=self.root_binding_id, relative_path=name), data)

    def json(self, name, value):
        return self.put(name, a.canonical(value))


class ClockObserver:
    def __init__(self, observer_id, clock_epoch_id, first_sequence=0):
        p.text(observer_id); p.text(clock_epoch_id); p.integer(first_sequence)
        self.observer_id, self.clock_epoch_id = observer_id, clock_epoch_id
        self.sequence = first_sequence
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')
            row = w._ClockObservation(self.observer_id, self.clock_epoch_id, self.sequence, now, time.monotonic_ns())
            self.sequence += 1
            return row

    def observe(self):
        row = self.acquire()
        return dict(observer_id=row.observer_id, clock_epoch_id=row.clock_epoch_id,
                    sequence=row.sequence, observed_utc=row.observed_utc)

    def boundary(self):
        return w._ClockBoundary(self.acquire)


class OwnedPipes:
    """Prepare explicit inheritable handles before command binding, never spawn."""
    def __init__(self, capsule_path):
        need(os.name == 'nt', 'Windows required')
        import msvcrt
        self.stdin = Path(capsule_path).open('rb')
        self.readers, self.writers = {}, {}
        try:
            for name in ('stdout','stderr'):
                rd, wr = os.pipe()
                self.readers[name] = os.fdopen(rd, 'rb', buffering=0)
                self.writers[name] = os.fdopen(wr, 'wb', buffering=0)
            self.handles = tuple(msvcrt.get_osfhandle(f.fileno()) for f in
                                 (self.stdin, self.writers['stdout'], self.writers['stderr']))
        except BaseException:
            self.close(); raise

    def close_writers(self):
        for stream in (self.stdin, *self.writers.values()):
            if not stream.closed: stream.close()

    def close(self):
        self.close_writers()
        for stream in self.readers.values():
            if not stream.closed: stream.close()


class ProcessObserver:
    """Held-handle once-only process boundary with multiplexed durable pipes.

    Adopted B bounds each retained raw stream at 64 MiB and the transport
    schedule at 4096 entries. Overflow kills owned work and refuses acceptance;
    cleanup continues observing/hash-counting reads while disclosing discarded bytes.
    """
    def __init__(self, *, pipes, cwd, store, clock, creation_name, exit_name,
                 raw_prefix, parent_identity=None, cleanup_seconds=30):
        need(os.name == 'nt', 'Windows required')
        need(0 < cleanup_seconds <= 30, 'cleanup bound')
        self.pipes, self.cwd, self.store, self.clock = pipes, Path(cwd), store, clock
        self.creation_name, self.exit_name, self.prefix = creation_name, exit_name, raw_prefix
        self.parent_identity = parent_identity
        self.cleanup_seconds = cleanup_seconds
        self.native = NativeBindings()
        self.owned = None; self.attempted = False; self.resumed = False; self.closed = False
        self.ph = self.th = self.job = None
        self.chunks = dict(stdout=[], stderr=[]); self.schedule = []; self.eof = dict(stdout=False,stderr=False)
        self.sinks = {}; self.retention = {}; self.transport = None; self.serial = 0
        self.termination_errors=[]; self.kill_requested=False; self.capture_overflow=False
        self.emergency_attempted=False
        c = self.native.c; wt = self.native.w
        self.peek = self.native.api('PeekNamedPipe', [wt.HANDLE,c.c_void_p,wt.DWORD,c.c_void_p,c.POINTER(wt.DWORD),c.c_void_p],wt.BOOL)

    def boundary(self):
        return w._ProcessBoundary(self.create_suspended_owned,self.inspect_owned,self.resume,
                                  self.drain_both_and_wait,self.terminate_and_drain_owned)

    def _raw(self, label, value):
        self.serial += 1
        return self.store.json(f'{self.prefix}_{self.serial:04d}_{label}.json',value)

    def _deadline(self,requested,cleanup):
        spec=a.decode(self.prepared.expected.specification)
        end=datetime.fromisoformat(spec['cleanup_until' if cleanup else 'valid_until'].replace('Z','+00:00'))
        remaining=max(0.,(end-datetime.now(timezone.utc)).total_seconds())
        return min(requested,time.monotonic_ns()+int(remaining*1e9))

    def create_suspended_owned(self, prepared, command, receipt):
        need(not self.attempted, 'one creation only'); self.attempted = True
        need(command.inherited_handles == self.pipes.handles, 'inherited handle mismatch')
        capsule = stable_bytes(self.pipes.stdin.name)
        need((len(capsule),a.sha(capsule)) == (command.capsule_bytes,command.capsule_sha256), 'capsule drift')
        spec = a.decode(prepared.expected.specification)
        need(command.source_commitment == spec['source_binding_sha256'] and
             a.sha(a.canonical(p.wire(command))) == spec['command_binding_sha256'], 'command binding drift')
        self.prepared, self.command, self.receipt = prepared, command, receipt
        try:
            # Open sinks before spawn: no child can outlive failed sink setup.
            for name in self.chunks:
                self.sinks[name] = self.store.path(self.prefix+'_'+name+'.bin').open('xb',buffering=0)
                self.retention[name]=StreamRetention(self.sinks[name])
            self.job = self.native.create_private_job()
            self.ph,self.th,pid = self.native.create_suspended((command.executable,*command.arguments),self.cwd,
                dict(command.environment),self.pipes.stdin,self.pipes.writers['stdout'],self.pipes.writers['stderr'])
            self.native.assign_job(self.job,self.ph)
            need(self.native.in_job(self.ph,self.job), 'job assignment not observed')
            raw = self.native.describe(self.ph,spec['role'])
            need(raw['pid'] == pid, 'creation PID disagreement')
            self.creation_native = raw
            ownership = self._raw('ownership',dict(process=raw,job=self.native.observation(self.job),
                inherited_handles=list(command.inherited_handles),assigned_before_resume=True))
            selected = next(x for x in spec['child_slots'] if x['key'] == prepared.expected.slot_key)
            parent = next((x for x in spec['child_slots'] if x['key'] == selected['parent_key']),None)
            need((parent is None) == (self.parent_identity is None), 'parent identity required exactly for child slots')
            row = dict(schema='state_process_creation_observation.v2',kind='creation',authority_id=prepared.authority_id,
                slot_id=prepared.slot_id,operation=spec['operation'],role=spec['role'],
                parent_slot_id=None if parent is None else a.sha(b'operatormark-launch-slot-v1\n'+a.canonical(dict(authority_id=prepared.authority_id,**parent))),
                parent_process_identity=self.parent_identity,dispatch_file=a.decode(receipt.dispatch_file),
                source_binding_sha256=spec['source_binding_sha256'],command_binding_sha256=spec['command_binding_sha256'],
                observation=self.clock.observe(),call_status='returned_success',creation_status='started',pid=pid,failure_files=[])
            data = a.canonical(row); ref = self.store.put(self.creation_name,data)
            identity = a.validate_process_observation(data,actual_file=ref,prepared=prepared,dispatch_file=a.decode(receipt.dispatch_file))
            handle = w._OwnedProcess(self.ph,self.job,prepared.slot_id,p.file_ref_from(spec['owner_binding_file']),
                a.canonical(identity),p.file_ref_from(ownership))
            creation = ex.CreationObservation('returned_success','started',handle.creation_identity,handle,(a.canonical(ref),),data)
            self.owned = w._OwnedObservation(handle,command,True,command.inherited_handles,creation)
            return self.owned
        except BaseException as primary:
            # Even an acknowledgment/identity-persistence failure after CreateProcess
            # retains and closes the held child. Never infer ownership from PID.
            try: self._emergency_close(time.monotonic_ns()+int(self.cleanup_seconds*1e9))
            except BaseException as cleanup:
                try: primary.add_note('owned cleanup failed: '+repr(cleanup))
                except BaseException: pass
            raise
        finally:
            if self.ph is not None: self.pipes.close_writers()

    def inspect_owned(self, handle, prepared):
        need(self.owned is not None and handle is self.owned.handle and prepared is self.prepared, 'foreign process handle')
        if not self.closed:
            actual = self.native.describe(self.ph,a.decode(prepared.expected.specification)['role'])
            need((actual['pid'],actual['creation_filetime']) ==
                 (self.creation_native['pid'],self.creation_native['creation_filetime']), 'held process identity drift')
            need(self.native.in_job(self.ph,self.job),'process left owned job')
        return self.owned

    def resume(self, handle, receipt):
        self.inspect_owned(handle,self.prepared)
        need(receipt is self.receipt and not self.resumed,'resume receipt/repetition')
        self.native.resume(self.th); self.resumed=True
        self.native.close_handle(self.th); self.th=None

    def _drain(self):
        for name,stream in self.pipes.readers.items():
            if self.eof[name]: continue
            count = self.native.w.DWORD()
            handle = self.native.msvcrt.get_osfhandle(stream.fileno())
            ok = self.peek(handle,None,0,None,ctypes.byref(count),None)
            if not ok:
                code = ctypes.get_last_error()
                if code == 109:
                    self.eof[name]=True;self.retention[name].eof=True;stream.close();continue
                raise ctypes.WinError(code)
            if count.value:
                data = os.read(stream.fileno(),min(count.value,65536))
                need(bool(data),'pipe read lost available bytes')
                try:
                    keep=self.retention[name].feed(data,metadata_available=
                        len(self.schedule)<MAX_CHUNKS and not self.retention[name].write_failed)
                except OSError as exc:
                    # One error per stream: subsequent observed bytes are drained
                    # and explicitly discarded, never repeatedly written/retried.
                    self._termination_error(name+'_spool_write',exc)
                    keep=b''
                    self.capture_overflow=True
                    self._terminate_operations()
                if keep:
                    self.chunks[name].append(keep);self.schedule.append((name,len(self.chunks[name])-1))
                if self.retention[name].overflow:
                    self.capture_overflow=True
                    self._terminate_operations()

    def _termination_error(self,operation,exc):
        # Fixed fields, no arbitrary exception rendering or per-loop events.
        self.termination_errors.append(dict(operation=operation,type=type(exc).__name__[:128],
            winerror=getattr(exc,'winerror',None),errno=getattr(exc,'errno',None)))

    def _terminate_operations(self):
        if self.kill_requested:return
        self.kill_requested=True
        if self.job is not None:
            try:self.native.terminate_job(self.job,124)
            except BaseException as exc:self._termination_error('TerminateJobObject',exc)
        if self.ph is not None:
            try:
                if not self.native.signaled(self.ph):self.native.terminate_process(self.ph,124)
            except BaseException as exc:self._termination_error('TerminateProcess',exc)

    def _wait(self,deadline,kill=False):
        if kill:self._terminate_operations()
        while time.monotonic_ns() < deadline:
            self._drain()
            state=self.native.observation(self.job)
            need(state['members_query_ok'] and state['accounting_query_ok'],'owned job observation unavailable')
            if self.native.signaled(self.ph) and state['active_processes']==0 and all(self.eof.values()):
                return state
            time.sleep(min(.002,max(0,(deadline-time.monotonic_ns())/1e9)))
        raise TimeoutError('owned process/streams did not close by shared deadline')

    def _release(self):
        errors=[]
        for stream in self.sinks.values():
            if not stream.closed:
                for operation in (stream.flush,lambda stream=stream:os.fsync(stream.fileno()),stream.close):
                    try:operation()
                    except BaseException as exc:errors.append(exc)
        # A failed flush/close of one stream must not skip any other handle.
        for stream in (self.pipes.stdin,*self.pipes.writers.values(),*self.pipes.readers.values()):
            try:
                if not stream.closed:stream.close()
            except BaseException as exc:errors.append(exc)
        for name in ('th','ph','job'):
            handle=getattr(self,name)
            if handle is not None:
                try:self.native.close_handle(handle);setattr(self,name,None)
                except BaseException as exc:errors.append(exc)
        self.closed=not errors
        if errors:raise RuntimeError('owned handle/stream release failed',errors)

    def _emergency_close(self,deadline):
        need(not self.emergency_attempted,'one emergency cleanup only; prior evidence retained')
        self.emergency_attempted=True
        deadline=self._deadline(deadline,True)
        state=None;failure=None;released=False
        try:
            self._terminate_operations()
            try:self.pipes.close_writers()
            except BaseException as exc:self._termination_error('close_parent_writers',exc)
            if self.ph is not None:
                state=self._wait(deadline)
        except BaseException as exc:failure=exc
        finally:
            try:self._release();released=True
            except BaseException as exc:
                self._termination_error('release',exc)
                if failure is None:failure=exc
            self._raw('emergency_closure',dict(job=state,resolved=state is not None and released,
                operation_errors=self.termination_errors,streams={k:v.record() for k,v in self.retention.items()},
                handles_released=released,error=None if failure is None else type(failure).__name__))
        if failure is not None:raise failure

    def _transport(self,handle,deadline,maximum,kill):
        self.inspect_owned(handle,self.prepared)
        if self.transport is not None:
            need(not self.capture_overflow,'capture overflow remains unacceptable')
            return self.transport
        need(not self.closed,'owned process already closed without transport')
        # The calling supervisor provides a separate cleanup deadline on work failure.
        deadline=self._deadline(deadline,kill)
        state=self._wait(deadline-2_000_000,kill=kill)
        code=self.native.exit_code(self.ph)
        raw=self._raw('closure',dict(job=state,direct_exit_code=code,streams_eof=self.eof,
            bytes={k:sum(map(len,v)) for k,v in self.chunks.items()},terminated=self.kill_requested,
            streams={k:v.record() for k,v in self.retention.items()},operation_errors=self.termination_errors,
            capture_overflow=self.capture_overflow))
        row=dict(schema='state_process_exit_observation.v2',kind='exit',slot_id=self.prepared.slot_id,
            process_identity=a.decode(handle.creation_identity),direct_exit_code=code,reap_status='observed_reaped',
            observation=self.clock.observe(),failure_files=[])
        data=a.canonical(row);ref=self.store.put(self.exit_name,data)
        a.validate_process_observation(data,actual_file=ref,prepared=self.prepared)
        self._release()
        readbacks={k:verify_retained(v.record(),self.store.path(self.prefix+'_'+k+'.bin')) for k,v in self.retention.items()}
        self._raw('stream_readback',readbacks)
        streams={k:ex.StreamObservation(b''.join(v),self.eof[k],self.pipes.readers[k].closed,a.canonical(raw))
                 for k,v in self.chunks.items()}
        closure=ex.ProcessClosure(code,'observed_reaped',streams['stdout'],streams['stderr'],(a.canonical(ref),))
        self.transport=w._TransportObservation(handle,tuple(self.chunks['stdout']),tuple(self.chunks['stderr']),
            tuple(self.schedule),closure,data,None,())
        need(not self.capture_overflow and all(sum(map(len,v))<=maximum for v in self.chunks.values()),
             'capture overflow; observed-only hash/count and loss disclosed; transport unacceptable')
        return self.transport

    def drain_both_and_wait(self,handle,deadline,maximum):
        return self._transport(handle,deadline,maximum,False)

    def terminate_and_drain_owned(self,handle,deadline,maximum):
        return self._transport(handle,deadline,maximum,True)


class BoundAcquisitionReader:
    """Reread exact externally admitted files; never mint acquisition premises."""
    def __init__(self, paths):
        self.paths=dict(paths)

    def read(self,ref):
        if isinstance(ref,p.FileRef):ref=p.wire(ref)
        a.ref(ref);key=(ref['root_binding_id'],ref['relative_path'])
        need(key in self.paths,'missing bound physical source')
        data=stable_bytes(self.paths[key]);a.check_bytes(ref,data);return data

    def refresh(self,value):
        if is_dataclass(value):
            updates={f.name:self.refresh(getattr(value,f.name)) for f in fields(value)}
            # Qualified Acquisition metadata is retained; only current bytes are
            # independently reread. No new admission/observer labels are created.
            if 'file' in updates and 'data' in updates:
                ref=updates['file'];ref=a.decode(ref) if type(ref) is bytes else ref
                updates['data']=self.read(ref)
            return replace(value,**updates)
        if type(value) is tuple:return tuple(self.refresh(x) for x in value)
        if type(value) is list:return [self.refresh(x) for x in value]
        if type(value) is dict:return {k:self.refresh(v) for k,v in value.items()}
        return value


class GuardAcquirer:
    """Acquire actual retained observations; do not invent hooks or zero counts."""
    def __init__(self,expected,independent_files,reader):
        guard._expectations(expected)
        self.expected,self.files,self.reader=expected,independent_files,reader

    def acquire(self,prepared,identity,exit_code,reap):
        spec=guard._expectations(self.expected)
        need(spec['launch_slot_id']==prepared.slot_id,'guard/launch slot mismatch')
        files=self.reader.refresh(self.files)
        result=guard.derive_guard_capture(expected=self.expected,independent_files=files)
        need(result.status=='corroborated','guard lifetime/hook/RNG observations incomplete')
        actual=a.decode(self.reader.read(spec['exit_file']))
        need(actual['process_identity']==a.decode(identity) and actual['direct_exit_code']==exit_code and
             actual['reap_status']==reap,'guard/direct process disagreement')
        return files

    def boundary(self):return w._GuardBoundary(self.acquire)


class GuardEventRecorder:
    """Real observed call outcomes. Does not certify global hook/lifetime coverage.

    The deployment installer must connect every required hook and supply actual
    RNG/lifetime observations. Missing installer evidence remains unavailable.
    """
    def __init__(self,store,clock,slot_id,subject_file):
        a.digest(slot_id);a.ref(subject_file)
        self.store,self.clock,self.slot,self.subject=store,clock,slot_id,subject_file
        self.previous=None;self.count=0;self.lock=threading.RLock()

    def observe_call(self,family,split,call,*args,**kwargs):
        need(family in guard.AUTHORIZED or family in guard.FORBIDDEN,'operation family')
        need((split in ('training','validation','test')) if family in guard.AUTHORIZED else split is None,'operation split')
        with self.lock:
            status='outcome_unavailable';effect='unknown';primary=None
            try:
                # Return is observed; physical side effects are not inferred from
                # it. Until a qualified family observer supplies effect evidence,
                # this record deliberately cannot establish a complete guard.
                result=call(*args,**kwargs);status='returned_success'
                return result
            except BaseException as exc:
                primary=exc;status='returned_failure';raise
            finally:
                stamp=self.clock.acquire()
                try:
                    data=guard.make_guard_event(launch_slot_id=self.slot,observer_id=stamp.observer_id,
                        clock_epoch_id=stamp.clock_epoch_id,sequence=stamp.sequence,kind='operation',operation=family,
                        split=split,decision='allowed',call_status=status,effect_status=effect,
                        subject_file=self.subject,prior_event_file=self.previous)
                    self.previous=self.store.put(f'guard_operation_{self.count:08d}.json',data);self.count+=1
                except BaseException as failure:
                    if primary is None:raise
                    try:primary.add_note('guard failure evidence unavailable: '+repr(failure))
                    except BaseException:pass


class NativeInputHost:
    """Concrete NativePorts routing over externally issued, reread prerequisites."""
    def __init__(self,*,launches,storage_inputs,executions,retained,reader,source_sha256,authorized_counts,
                 retention_context=None, support_stores=None, expected_support_roots=None):
        self.launches=dict(launches);self.storage=dict(storage_inputs);self.executions=dict(executions)
        self.retained,self.reader=retained,reader
        self.source_sha256,self.counts=source_sha256,authorized_counts
        self.retention_context=retention_context
        self.support_stores=support_stores;self.expected_support_roots=expected_support_roots

    def launch_inputs(self,stage):
        need(stage in self.launches,'missing independently admitted launch')
        expected,inputs=self.reader.refresh(self.launches[stage])
        a.prepare_launch(expected=expected,independent_inputs=inputs)
        return expected,inputs

    def storage_inputs(self,phase,files):
        from .state_v2_prerequisites import verify_storage
        need(phase in self.storage,'missing independent storage boundary')
        args=self.reader.refresh(self.storage[phase])
        verify_storage(args,phase=phase,source_sha256=self.source_sha256,files=files)
        return args

    def execution_inputs(self,stage):
        from .state_v2_prerequisites import verify_execution
        need(stage in self.executions and stage in self.counts,'missing independent execution/count observations')
        args=self.reader.refresh(self.executions[stage])
        verify_execution(args,stage=stage,expected_source_sha256=self.source_sha256,authorized_counts=self.counts[stage])
        return args

    def retention_members(self):
        """Actual complete raw support, under separately admitted store bindings.

        Includes spools and partial/failed records, even when no verifier consumed
        them. No new evidence is written by this readback; terminal runs only
        after all owned producers have closed. This is not an acceptance issuer.
        """
        from . import corpus_v2_custody as custody
        need(type(self.support_stores) is tuple and bool(self.support_stores) and
             all(type(x) is RecordStore for x in self.support_stores),'complete observed support stores required')
        roots=tuple(x.root_binding_id for x in self.support_stores)
        need(type(self.expected_support_roots) is tuple and roots==self.expected_support_roots and
             len(set(roots))==len(roots) and len({x.root for x in self.support_stores})==len(roots),
             'independently selected support-root bindings differ/duplicate')
        result=[]
        for store in self.support_stores:
            names=custody.names(store.root)
            facts=custody.snapshot(store.root,names)
            need(custody.names(store.root)==names,'host support membership changed during acquisition')
            result.extend((store.root_binding_id,x.name,x.size,x.sha256) for x in facts)
        return tuple(sorted(result))

    def retention_inputs(self):
        from .state_v2_storage import verify_retained_inputs
        need(self.retained is not None,'missing independent retained membership')
        value=self.reader.refresh(self.retained)
        need(type(self.retention_context) is dict and set(self.retention_context)=={'attempt_root','required','local'},
             'missing retained root and complete required membership')
        context=dict(self.retention_context)
        context['required']=set(context['required'])|set(self.retention_members())
        verify_retained_inputs(value,**context)
        return value

    def ports(self):
        from .state_v2_workflow import NativePorts
        return NativePorts(self.launch_inputs,self.storage_inputs,self.execution_inputs,self.retention_inputs,self.retention_members)


def windows_file_observation(path):
    """Held Windows object identity and actual bytes; no custody inference."""
    need(os.name=='nt','Windows required')
    from ctypes import wintypes as wt
    path=Path(path).absolute()
    for part in (path,*path.parents):
        if part.exists():need(not part.lstat().st_file_attributes & 0x400,'reparse object/ancestor')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wt.LPCWSTR,wt.DWORD,wt.DWORD,ctypes.c_void_p,wt.DWORD,wt.DWORD,wt.HANDLE];create.restype=wt.HANDLE
    close=kernel.CloseHandle;close.argtypes=[wt.HANDLE];close.restype=wt.BOOL
    query=kernel.GetFileInformationByHandleEx
    query.argtypes=[wt.HANDLE,ctypes.c_int,ctypes.c_void_p,wt.DWORD];query.restype=wt.BOOL
    legacy=kernel.GetFileInformationByHandle
    legacy.argtypes=[wt.HANDLE,ctypes.c_void_p];legacy.restype=wt.BOOL
    class IdInfo(ctypes.Structure):
        _fields_=[('volume',ctypes.c_ulonglong),('identifier',ctypes.c_ubyte*16)]
    class Info(ctypes.Structure):
        _fields_=[('attributes',wt.DWORD),('created',wt.FILETIME),('accessed',wt.FILETIME),('written',wt.FILETIME),
                  ('volume',wt.DWORD),('size_high',wt.DWORD),('size_low',wt.DWORD),('links',wt.DWORD),
                  ('index_high',wt.DWORD),('index_low',wt.DWORD)]
    h=create(str(path),0x80,7,None,3,0x02000000|0x00200000,None)
    if h==ctypes.c_void_p(-1).value:
        code=ctypes.get_last_error()
        if code in (2,3):return dict(presence='absent',identity=None,size=None,sha256=None)
        raise ctypes.WinError(code)
    def info():
        ident=IdInfo();old=Info()
        if not query(h,18,ctypes.byref(ident),ctypes.sizeof(ident)) or not legacy(h,ctypes.byref(old)):
            raise ctypes.WinError(ctypes.get_last_error())
        need(not old.attributes & 0x400,'held reparse object')
        identity=ev._NativeIdentity(hex(ident.volume),bytes(ident.identifier).hex(),f'{old.volume:08x}',
                                    f'{((old.index_high<<32)|old.index_low):016x}')
        ev._identity(identity)
        return identity,old
    try:
        identity,before=info()
        data=None if before.attributes & 0x10 else stable_bytes(path)
        current,after=info()
        def stable_info(info):
            # This read may update last-access time. It cannot change identity,
            # creation/write times, size, attributes or link count unnoticed.
            return (info.attributes,bytes(info.created),bytes(info.written),info.volume,
                    info.size_high,info.size_low,info.links,info.index_high,info.index_low)
        need(current==identity and stable_info(before)==stable_info(after),'held native object mutated')
        # Reopen identity to distinguish path substitution from held-handle stability.
        second=create(str(path),0x80,7,None,3,0x02000000|0x00200000,None)
        if second==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
        try:
            other=IdInfo()
            if not query(second,18,ctypes.byref(other),ctypes.sizeof(other)):raise ctypes.WinError(ctypes.get_last_error())
            need(other.volume==int(identity.volume_serial_number_hex,16) and bytes(other.identifier).hex()==identity.file_id_128_hex,
                 'native path substitution')
        finally:
            if not close(second):raise ctypes.WinError(ctypes.get_last_error())
        return dict(presence='present',identity=identity,directory=bool(before.attributes&0x10),
                    size=None if data is None else len(data),sha256=None if data is None else hashlib.sha256(data).hexdigest())
    finally:
        if not close(h):raise ctypes.WinError(ctypes.get_last_error())


MEMBERSHIP_FAILURE_BYTES = 64 * 1024


def membership_failure_bytes(*, path, kind, observed_entries):
    """Adopted D: bounded partial-observation diagnostic, never a tree digest."""
    def encode(prefix):
        return json.dumps(dict(schema='phase_membership_failure.v1',
            reason='first_unexpected_entry',entry_kind=kind,pathname_prefix=prefix,
            pathname_characters=len(path),omitted_pathname_characters=len(path)-len(prefix),
            pathname_complete=len(prefix)==len(path),traversal_complete=False,
            observed_entries=observed_entries,count_scope='observed prefix only',
            complete_tree_count=None,complete_tree_sha256=None,
            unexpected_entries_preserved=True,phase_refused=True),ensure_ascii=True,
            sort_keys=True,separators=(',',':')).encode('utf-8')
    # A prefix is selected by encoded size, including the omission fields.
    low=0;high=min(len(path),MEMBERSHIP_FAILURE_BYTES)
    need(len(encode(''))<=MEMBERSHIP_FAILURE_BYTES,'membership diagnostic envelope')
    while low<high:
        mid=(low+high+1)//2
        if len(encode(path[:mid]))<=MEMBERSHIP_FAILURE_BYTES:low=mid
        else:high=mid-1
    return encode(path[:low])


def observed_phase_membership(root, expected_files, store, name):
    """Stream entries; stop at the first foreign entry without enumerating more.

    Success retains exactly the previous complete sorted tree JSON. Missing
    expected members still retain the complete observation and fail afterward.
    This is a membership check, not continuous-custody evidence.
    """
    import stat
    from contextlib import ExitStack
    from . import corpus_v2_custody as custody
    root=custody.checked_directory(root)
    expected=(set(custody.ancestor_directories(expected_files)),set(expected_files))
    found=([],[]);seen=set();count=0
    with ExitStack() as stack:
        scans=[(root,stack.enter_context(os.scandir(root)))]
        while scans:
            directory,scan=scans[-1]
            entry=next(scan,None)
            if entry is None:scan.close();scans.pop();continue
            path=directory/entry.name;relative=path.relative_to(root).as_posix()
            info=entry.stat(follow_symlinks=False);is_dir=stat.S_ISDIR(info.st_mode)
            reparse=stat.S_ISLNK(info.st_mode) or bool(getattr(info,'st_file_attributes',0)&0x400)
            index=0 if is_dir else 1;count+=1
            if reparse or relative not in expected[index] or relative in seen:
                # No traversal of this entry, no next() after detection, no
                # whole-tree count/hash. Failure to persist remains a failure.
                store.put(name,membership_failure_bytes(path=relative,
                    kind='reparse' if reparse else 'duplicate' if relative in seen else 'directory' if is_dir else 'file',
                    observed_entries=count))
                raise RuntimeError('unexplained actual phase membership; first unexpected entry; traversal incomplete')
            seen.add(relative);found[index].append(relative)
            if is_dir:scans.append((path,stack.enter_context(os.scandir(path))))
    actual=tuple(sorted(found[0])),tuple(sorted(found[1]))
    store.put(name,json.dumps(dict(directories=actual[0],files=actual[1]),ensure_ascii=True).encode())
    need(actual==(custody.ancestor_directories(expected_files),tuple(sorted(expected_files))),
         'unexplained actual phase membership')
    return actual


class PhaseStorageObserver:
    """Observe actual roots/objects; emitted facts must match pre-admitted slots.

    Does not claim continuous exclusivity. Accepted request templates supply
    destinations/identity correspondences, never a replacement for observation.
    """
    def __init__(self,store):self.store=store;self.serial=0;self.raw_serial=0

    def boundary(self):return storage._StorageObserver(self.capture_phase)

    def _observe(self,path):
        result=windows_file_observation(path)
        self.store.put(f'phase_native_{self.raw_serial:06d}.json',
                       json.dumps(dict(path=str(path),observation=p.wire(result)),sort_keys=True,ensure_ascii=True).encode())
        self.raw_serial+=1
        return result

    def capture_phase(self,request,acquired):
        import shutil
        storage.validate_storage_request(request)
        now=datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
        need(request.capture.started_utc<=now<=request.capture.finished_utc,'phase acquisition outside capture interval')
        need(len(acquired)==len(request.upstream_reads),'incomplete upstream acquisitions')
        roots={r.root_binding_id:r for r in request.roots}
        maps={m.opaque:m.native for m in request.identities}
        observed={};directory=[]
        locations={r.root_binding_id:r.relative_path for r in request.directory.locations}
        need(len(locations)==len(request.directory.locations),'duplicate directory locations')
        for root in request.roots:
            actual=self._observe(root.canonical_path)
            need(actual['presence']=='present' and actual['directory'] and actual['identity']==maps[root.native_identity],
                 'root identity changed/unavailable')
            if root.root_binding_id in ('T','F','Q'):
                need(root.root_binding_id in locations,'missing possible directory location')
                path=Path(root.canonical_path)/locations[root.root_binding_id]
                found=self._observe(path)
                observed[root.root_binding_id]=found
                if found['presence']=='present' and found['identity']==maps[request.directory.native_identity]:
                    directory.append((root,path,found))
        need(len(directory)==1,'attempt directory location absent/ambiguous')
        root,path,found=directory[0];location=root.root_binding_id
        facts={};rows=[]
        def fact(kind,subject,value):
            slots=[x for x in request.allowed_fact_products if (x.kind,x.subject)==(kind,subject)]
            need(len(slots)==1,'missing/duplicate declared observation fact')
            slot=slots[0]
            need(p.decode(slot.value_bytes)==value,'observed fact differs from independently declared value')
            facts[(kind,subject)]=slot
            # capture_later_storage_boundary publishes this actual matched value
            # through the existing writer and independently acquires it.
            address=slot.acquisition_request.address
            data=p.encode(dict(schema=sr.PROFILE+'.fact',binding=p.wire(request.binding),capture=p.wire(request.capture),
                kind=kind,subject=subject,value=value))
            return a.file_ref(p.wire(address),data)
        for wanted in request.outputs:
            actual=self._observe(path/wanted.relative_path)
            present=actual['presence']=='present'
            if present:
                need(not actual['directory'] and actual['identity']==maps[wanted.native_identity],'output native identity mismatch')
                content=dict(root_binding_id=location,relative_path=locations[location]+'/'+wanted.relative_path,
                             file_bytes=actual['size'],file_sha256=actual['sha256'])
                need((actual['size'],actual['sha256'])==(wanted.content_file.file_bytes,wanted.content_file.file_sha256),
                     'observed output content mismatch')
            else:content=None
            row=dict(slot=wanted.slot,root_binding_id=location,relative_path=wanted.relative_path,presence=actual['presence'],
                native_identity=wanted.native_identity if present else None,custody_file=p.wire(wanted.custody_file) if present else None,
                content_file=content,presence_file=None,stable_file=None)
            row['presence_file']=fact('presence',wanted.slot,{k:row[k] for k in ('presence','native_identity','root_binding_id','relative_path')})
            if present:row['stable_file']=fact('stable',wanted.slot,{k:row[k] for k in ('native_identity','root_binding_id','relative_path','content_file')})
            rows.append(row)
        from . import corpus_v2_custody as custody
        present_names=tuple(sorted(r['relative_path'] for r in rows if r['presence']=='present'))
        observed_phase_membership(path,present_names,self.store,f'phase_membership_{self.serial:04d}.json')
        observed_directory=dict(native_identity=request.directory.native_identity,custody_file=p.wire(request.directory.custody_file),
            root_binding_id=location,relative_path=locations[location],resolved_path=root.canonical_path+'/'+locations[location],evidence_file=None)
        observed_directory['evidence_file']=fact('attempt_directory',request.binding.attempt_id,
                                                {k:v for k,v in observed_directory.items() if k!='evidence_file'})
        original=request.original_operation
        if original is None:operation=dict(action='none',call_status='not_called',effect_status='none',evidence_file=None,authority_file=None)
        else:
            storage.validate_operation_projection(original,original_result=original.original_bytes,accepted_inputs=request.accepted_operation_files)
            operation=dict(action=original.action,call_status=original.call_status,effect_status=original.effect_status,
                evidence_file=fact('operation',request.binding.attempt_id,dict(action=original.action,call_status=original.call_status,effect_status=original.effect_status)),
                authority_file=p.wire(original.authority_file))
        row=dict(schema=sr.OBSERVATION,profile=sr.PROFILE,binding=p.wire(request.binding),phase=request.phase,
            capture=p.wire(request.capture),roots=p.wire(request.roots),location=location,
            location_file=fact('location',request.binding.attempt_id,location),attempt_directory=observed_directory,
            context=p.wire(request.context),objects=rows,operation=operation,failure_files=p.wire(request.context.failure_files),
            cleanup_failure_files=[],free_space_bytes=shutil.disk_usage(root.canonical_path).free)
        data=sr.canonical_later_storage_bytes(row)
        self.store.put(f'phase_observation_{self.serial:04d}.json',data);self.serial+=1
        now=datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
        need(now<=request.capture.finished_utc,'phase acquisition overran capture interval')
        return storage._PhaseInventory(data,tuple(facts.values()),True,tuple(r['slot'] for r in rows),())


def build_host_providers(*,configuration,evidence,claims,process,guard_acquirer,clock):
    """Existing validators still gate every native provider; no admission minted."""
    return w.build_windows_state_providers(configuration=configuration,evidence=evidence,claims=claims,
        process_boundary=process.boundary(),guard_boundary=guard_acquirer.boundary(),clock_boundary=clock.boundary())
