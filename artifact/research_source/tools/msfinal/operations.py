"""Closed-session operational route over the existing acceptance interfaces.

An independently reviewed host supplies typed evidence and an accepted Session;
this module never issues them, reserves a window, retries, installs, or deletes.
The host must run this route inside its bound outer owned-process envelope.
There is deliberately no default credential, approval, or scientific input.
"""
from pathlib import Path
import time
import traceback
import os

from msdev.records import read, write, sha
from . import session as S
from . import export as E
from .phase_storage import PhaseStore, restore_worker
from .worker import Worker, MissingEvidence


def supervise_closed_stage(session, supervisor):
    """Existing owned route around the *whole* restore/worker/export host.

    The independently bound host command calls closed_stage and supplies its
    accepted objects/OAuth callback. Host Python may differ from numerical Python;
    both executable identities are checked in their own existing session fields.
    No setup, auth wait, network call or finalizer escapes this outer deadline.
    """
    from msdev.colab_environment import owned
    from .marking import identity
    if os.name != 'posix':
        raise ValueError('whole-stage ownership uses the existing POSIX route')
    session.check()
    s=session.value
    command=s.get('operational_command')
    if (not isinstance(command,list) or len(command)<3 or command[1]!='-B'
            or not all(isinstance(x,str) for x in command)):
        raise ValueError('exact independently bound operational host command required')
    tail=s.get('closure_persistence_command')
    if (len(command)<4 or command[3]!='stage-host' or not isinstance(tail,list)
            or tail!=[('persist-closure' if i==3 else v) for i,v in enumerate(command)]):
        raise ValueError('exact bound closure persistence child required before launch')
    if (not Path(command[0]).is_absolute() or not Path(command[2]).is_absolute()
            or not Path(s['cwd']).is_absolute() or not Path(s['cwd']).is_dir()):
        raise ValueError('absolute operational interpreter/script and existing bound cwd required')
    if (sha(command[0])!=identity(s.get('operational_interpreter_sha256'))
            or sha(command[2])!=identity(s.get('operational_entry_sha256'))
            or sha(supervisor.__file__)!=identity(s.get('supervisor_sha256'))):
        raise ValueError('operational host/interpreter/supervisor drift')
    outer_deadline=s.get('stage_deadline',s['deadline'])
    if outer_deadline>s['deadline'] or s['finalization_deadline']>outer_deadline-30:
        raise ValueError('final outer owned cleanup reserve missing')
    S.require_owned_host()
    folder=Path(s['operational_root']).resolve()
    worker=Path(s['worker_root']).resolve();launch=Path(s['launch_root']).resolve()
    if any(folder==p or folder.is_relative_to(p) or p.is_relative_to(folder) for p in (worker,launch)):
        raise ValueError('separate bound operational supervision root required')
    folder.mkdir(exist_ok=False)
    write(folder/'OPERATION_USED.json',dict(session_sha256=session.record.sha256,
          command=command,original_start=s['start'],original_deadline=s['deadline']),durable=True)
    # Reserve the last three minutes of finalization for durable outer records.
    # The host's archive cannot contain records written after that host exits.
    result=owned(command,s['finalization_deadline']-180-session.clock(),outer_deadline-180,
                 folder/'supervisor',supervisor,cwd=s['cwd'],acceptable=(0,4,5))
    durable_path=launch/'DURABLE_STAGE.json'
    value=read(durable_path) if durable_path.is_file() else None
    if (result['returncode']!=0 or not S.closure_ok(result) or value is None
            or value.get('session_sha256')!=session.record.sha256
            or value.get('durable_success') is not True):
        raise ValueError('whole-stage durable result/outer closure incomplete; no follow-on')
    receipt=dict(session_sha256=session.record.sha256,outer_closure=result,
                 durable_stage_sha256=sha(durable_path),remote_commit=value['durable_export']['remote_commit'],
                 durable_success=True,scientific_acceptance=False,follow_on_authorized=False,
                 stage_deadline=outer_deadline,outer_receipt_requires_preservation=True)
    write(folder/'OPERATION_CLOSED.json',receipt,durable=True)
    persisted=owned(tail,max(0,s['finalization_deadline']-session.clock()-30),outer_deadline,
        folder/'persistence_supervisor',supervisor,cwd=s['cwd'],acceptable=(0,))
    tail_receipt=folder/'closure_persistence'/'PERSISTED.json'
    if (not S.closure_ok(persisted) or not tail_receipt.is_file()
            or read(tail_receipt).get('operation_closed_sha256')!=sha(folder/'OPERATION_CLOSED.json')
            or read(tail_receipt).get('remote_records_verified') is not True):
        raise ValueError('outer evidence not durably persisted; release refused')
    return dict(receipt,outer_receipt_requires_preservation=False,
        closure_persistence=read(tail_receipt),persistence_child_closed=True,disconnect_permitted=True)


def persist_operation_closure(session, transport):
    """Owned metadata-only child, after the scientific/publication host closes.

    Main outer receipts and streams are immutable inputs. This does not claim
    that a publisher can archive its own future termination receipt.
    """
    s=session.value;folder=Path(s['operational_root']);stage=Path(s['launch_root'])/'DURABLE_STAGE.json'
    operation=folder/'OPERATION_CLOSED.json';value=read(operation)
    def check():
        if session.record.read()!=s or session.clock()>=s['finalization_deadline']:
            raise TimeoutError('closure persistence immutable finalization deadline')
    check()
    if (value.get('session_sha256')!=session.record.sha256
            or value.get('durable_stage_sha256')!=sha(stage)
            or value.get('outer_closure')!=read(folder/'supervisor/RESULT.json')
            or not S.closure_ok(value['outer_closure']) or value['outer_closure'].get('returncode')!=0):
        raise ValueError('closed main host evidence differs')
    dest=folder/'closure_persistence';dest.mkdir(exist_ok=False)
    paths=[operation,stage]+sorted(p for p in (folder/'supervisor').rglob('*') if p.is_file())
    if any(p.is_symlink() for p in paths):raise ValueError('closure evidence symlink')
    origins={}
    for i,p in enumerate(paths):
        check();name=f'{i:04d}_{p.name}';before=sha(p);data=p.read_bytes()
        import hashlib
        if hashlib.sha256(data).hexdigest()!=before:raise ValueError('closure evidence mutation')
        with (dest/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        if sha(p)!=before:raise ValueError('closure evidence changed during copy')
        origins[name]=dict(original=str(p),sha256=before,bytes=len(data))
    write(dest/'ORIGINS.json',origins,durable=True)
    committed=transport.commit_records(dest,list(origins)+['ORIGINS.json'],
        deadline=s['finalization_deadline'],check=check)
    check()
    if committed.get('remote_records_verified') is not True:raise ValueError('remote closure readback missing')
    result=dict(operation_closed_sha256=sha(operation),session_sha256=session.record.sha256,
        remote_records_verified=True,remote_commit=committed,scientific_acceptance=False)
    write(dest/'PERSISTED.json',result,durable=True)
    return result


class PublicationEvidence:
    """Independent verifier port, not a producer-success-to-acceptance adapter.

    The host supplies existing AcceptedRecord objects after independent selection
    and verification of their exact expectations. A missing result stops once;
    there is no polling, synthetic issuer or acceptance manufactured here.
    """
    def closure(self, path):
        raise MissingEvidence('independently selected canonical closure required: '+str(path))

    def permissions(self, root, destination, closure):
        raise MissingEvidence('independent private-export and stage-transfer permissions required')


def worker_entry(configuration, session, evidence, training_inputs, *, resume=None,
                 phase_store=None):
    """Call in the exact accepted supervised worker command, not in the host.

    Existing Worker admission, counters, state, opening, readback and exceptions
    remain authoritative. Return codes are transport status, never acceptance.
    """
    worker = Worker(session.value['worker_root'], configuration, session, evidence,
                    resume=resume, phase_store=phase_store)
    result = worker.run(training_inputs)
    if result['status'] in ('CORE_COMPLETE', 'CORE_INCOMPLETE'):
        return 0
    if result['status'] in ('WAITING_ACCEPTANCE', 'PAUSED_SESSION'):
        return 5
    return 4


def closed_stage(session, supervisor, transport, publication, destination, *,
                 phase_store=None, restore_permission=None, restore_staging=None,
                 clock=time.time):
    """Restore if admitted -> owned worker -> independent close -> durable export.

    One invocation of the existing accepted command. No disconnect is requested
    here: the outer host may disconnect only after this returns a verified remote
    commit AND that host's own process/stream closure is complete. On any failure
    evidence stays local/partial, durable_success is false, and no retry follows.
    An existing session can be finalized separately by finalize_closed_stage;
    it must never be relaunched to obtain a missing acceptance.
    """
    root = Path(session.value['worker_root'])
    launch = Path(session.value['launch_root'])
    session.check()
    if launch.exists():
        raise FileExistsError('existing launch/consumed evidence preserved')
    if restore_permission is not None:
        if restore_staging is None:
            raise ValueError('fresh explicitly bound restore staging required')
        grant = restore_permission.read()
        from .bounded_stage import worker_binding
        expected = worker_binding(session)
        if grant.get('worker_binding') != expected:
            raise ValueError('restore changes candidate or original window')
        restore_worker(transport, restore_permission, root, restore_staging,
                       check=session.check, clock=clock)
    if phase_store is not None:
        if type(phase_store) is not PhaseStore or phase_store.transport is not transport:
            raise ValueError('same bound phase-store transport required')
    try:
        S.supervise(session, launch, supervisor)
    except BaseException as exc:
        if launch.is_dir():
            _failure(launch, exc)
        raise
    return finalize_closed_stage(session, transport, publication, destination,
                                 phase_store=phase_store, clock=clock)


def _failure(folder, exc):
    try:
        write(Path(folder)/('OPERATION_FAILURE_'+str(time.time_ns())+'.json'),
              dict(error=repr(exc),traceback=traceback.format_exc(),durable_success=False,
                   disconnect_permitted=False,retry_authorized=False),durable=True)
    except BaseException as reporting:
        exc.add_note('operational failure record unavailable: '+repr(reporting))


def finalize_closed_stage(session, transport, publication, destination, *,
                          phase_store=None, clock=time.time):
    """Finalize existing closed evidence under its original deadline; no launch.

    A previous missing independent acceptance can be supplied without replaying
    Worker. Existing exports/partial attempts still refuse reuse; this function
    does not authorize another finalization attempt or extend its allowance.
    """
    root = Path(session.value['worker_root'])
    launch = Path(session.value['launch_root'])
    supervisor_record = launch/'supervisor'/'RESULT.json'
    closure_path = launch/'CLOSURE.json'

    def check():
        if session.record.read() != session.value:
            raise ValueError('accepted operational session drift')
        if clock() >= session.value['finalization_deadline']:
            raise TimeoutError('original finalization deadline expired')

    try:
        check()
        # Unresolved ownership/timeout is a stop, even when partial worker output
        # happens to exist. Its original supervisor receipt is never rewritten.
        raw = read(supervisor_record)
        if (not S.closure_ok(raw) or raw.get('timed_out') is not False
                or raw.get('error') or raw.get('returncode') not in (0,4,5)):
            raise ValueError('closed non-timeout owned worker required for publication')
        if not closure_path.exists():
            S.close_result(session, root, supervisor_record, closure_path, check,
                           phase_store=phase_store)
        closure = publication.closure(closure_path)
        # Trust comes from the external verifier; correspondence comes from bytes.
        from .backend import AcceptedRecord
        if type(closure) is not AcceptedRecord:
            raise ValueError('independently selected AcceptedRecord closure required')
        if closure.sha256 != sha(closure_path) or closure.read() != read(closure_path):
            raise ValueError('independent closure does not bind actual closer output')
        if closure.read().get('supervisor_record_sha256') != sha(supervisor_record):
            raise ValueError('closed supervisor bytes drifted')
        permit, transfer_permit = publication.permissions(root, Path(destination), closure)
        durable = E.publish_durable(root, destination, closure, permit, transfer_permit,
                                    transport, check, clock=clock, phase_store=phase_store)
        check()
        if not (durable.get('durable_remote') is True and durable.get('disconnect_permitted') is True):
            raise ValueError('full remote archive/record commit missing')
        result = read(root/'RESULT.json')
        receipt = dict(worker_status=result['status'],worker_result_sha256=sha(root/'RESULT.json'),
                       session_sha256=session.record.sha256,closure_sha256=closure.sha256,
                       durable_export=durable,original_deadline=session.value['deadline'],
                       durable_success=True,outer_closure_still_required=True,
                       scientific_acceptance=False,follow_on_authorized=False)
        write(launch/'DURABLE_STAGE.json', receipt, durable=True)
        return receipt
    except BaseException as exc:
        if launch.is_dir():
            _failure(launch, exc)
        raise
