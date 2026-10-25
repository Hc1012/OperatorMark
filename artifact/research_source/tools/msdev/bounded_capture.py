"""POSIX B capture using existing owned-group cleanup; no pipe-capacity premise."""
from pathlib import Path
import os
import signal
import selectors
import subprocess
import time
from operatormark.corpus.bounded_stream import StreamRetention,MAX_READ,verify_retained


def capture(command,*,cwd,env,work_deadline,cleanup_deadline,folder,supervisor,on_identity):
    if os.name!='posix':raise ValueError('POSIX owned group required')
    folder=Path(folder);streams={};child=None;identity=None;closure=None;reason=None;errors=[]
    selector=selectors.DefaultSelector();killed=False;leader_early=None;registered=set()
    def terminate(why):
        nonlocal killed,reason
        if reason is None:reason=why
        # Direct group signal only while the unreaped leader holds its pid/group
        # number; afterwards only the identity/membership-checked supervisor signals.
        if not killed and child is not None and child.returncode is None:
            killed=True
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            except OSError as exc:errors.append(dict(operation='killpg',errno=exc.errno))
    def pump(wait,enforce):
        for key,_ in selector.select(wait):
            data=os.read(key.fileobj.fileno(),MAX_READ);row=streams[key.data]
            if not data:row.eof=True;registered.discard(key.data);selector.unregister(key.fileobj);key.fileobj.close()
            else:
                try:row.feed(data,metadata_available=not row.write_failed)
                except OSError as exc:
                    errors.append(dict(operation='stream_write',errno=exc.errno))
                    if enforce:terminate('capture_error')
                if row.overflow and enforce:terminate('stream_quota')
    try:
        # Sinks exist before spawn; never use unbounded inherited log handles.
        for name in ('stdout','stderr'):
            streams[name]=StreamRetention((folder/(name+'.bin')).open('xb',buffering=0))
        child=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        identity=supervisor.process_identity(child.pid);on_identity(identity)
        for name in streams:
            pipe=getattr(child,name);os.set_blocking(pipe.fileno(),False);selector.register(pipe,selectors.EVENT_READ,name);registered.add(name)
        while time.monotonic()<cleanup_deadline:
            if time.monotonic()>=work_deadline:terminate('work_deadline')
            pump(min(.05,max(0,cleanup_deadline-time.monotonic())),True)
            # Leader exit ends owned work (existing Q5 contract). Pipe-holding
            # descendants are closed by supervisor cleanup now, not after the
            # work window; remaining bytes are drained after that cleanup.
            if child.poll() is not None:
                leader_early=not all(x.eof for x in streams.values());break
        if child.poll() is None:terminate('incomplete_capture')
    except BaseException as exc:
        errors.append(dict(operation='capture',type=type(exc).__name__[:128]));terminate('capture_error')
    finally:
        if child is not None and identity is not None:
            try:closure=supervisor.cleanup(child,identity,cleanup_deadline)
            except BaseException as exc:
                errors.append(dict(operation='cleanup',type=type(exc).__name__[:128]))
                closure=dict(resolved=False,reason='cleanup observation unavailable')
        elif child is not None:
            terminate('identity_unavailable')
            try:child.wait(timeout=max(0,cleanup_deadline-time.monotonic()))
            except subprocess.TimeoutExpired:pass
            # Direct reap is not independently established tree closure.
            closure=dict(resolved=False,reason='identity acquisition missing')
        try:   # bounded post-cleanup drain to EOF; quota still applies; no signalling
            while registered and time.monotonic()<cleanup_deadline:
                pump(min(.05,max(0,cleanup_deadline-time.monotonic())),False)
            if reason is None and not all(x.eof for x in streams.values()):reason='incomplete_capture'
        except BaseException as exc:
            errors.append(dict(operation='post_cleanup_drain',type=type(exc).__name__[:128]))
        def release(operation, callback):
            try:callback()
            except BaseException as exc:
                errors.append(dict(operation=operation,type=type(exc).__name__[:128]))
        for pipe in (() if child is None else (child.stdout,child.stderr)):
            if pipe is not None:release('pipe_close',pipe.close)
        release('selector_close',selector.close)
        for row in streams.values():
            release('spool_flush',row.sink.flush)
            release('spool_fsync',lambda:os.fsync(row.sink.fileno()))
            release('spool_close',row.sink.close)
    rows={}
    for key,value in streams.items():
        record=value.record();readback=None
        try:readback=verify_retained(record,folder/(key+'.bin'))
        except BaseException as exc:
            errors.append(dict(operation='spool_readback',stream=key,type=type(exc).__name__[:128]))
            record['capture_complete']=False
        rows[key]=dict(record,readback=readback)
    return dict(streams=rows,closure=closure,termination=reason,
        returncode=None if child is None else child.returncode,errors=errors,
        timed_out=reason=='work_deadline',stream_overflow=any(x.overflow for x in streams.values()),
        leader_exit_before_streams_eof=leader_early,
        stdout_closed=bool(streams) and all(x.eof and x.sink.closed for x in streams.values()))
