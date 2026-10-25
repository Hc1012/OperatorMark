"""One authorized readiness recovery; old ledger and consumed marker stay intact."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from core import HERE, read, write, source_hashes, identity

EVIDENCE=HERE/'evidence/readiness_cost_20260930'
ROOT=EVIDENCE/'recovery_20260930'
LIMITS=dict(solver_attempts=1800,model_updates=96,inverse_updates=480)
CARRIED=dict(solver_attempts=278,model_updates=0,inverse_updates=0)
PRIOR_SECONDS=32.163734900066629


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=['focused','measurement']);p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:p.print_help();return 0
    if not (ROOT/'AUTHORIZATION.md').exists():raise PermissionError('Part B record missing')
    ledger=ROOT/'ledger';ledger.mkdir(exist_ok=True)
    lock=ledger/'active.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        prior=[read(f) for f in ledger.glob('*/result.json')]
        if len(list(ledger.glob('*/started.json')))!=len(prior):raise RuntimeError('unresolved prior child')
        if any(r['timed_out'] or not r['owned_process_ended'] for r in prior):raise RuntimeError('prior timeout/ownership stop')
        used=PRIOR_SECONDS+sum(r['wall_seconds'] for r in prior);focused=[r for r in prior if r['kind']=='focused']
        if args.kind=='measurement':
            if (ledger/'MEASUREMENT_CONSUMED.json').exists():raise RuntimeError('one measurement attempt already consumed')
            if not focused or focused[-1]['exit_code']!=0:raise RuntimeError('recovery checks must pass')
            review=read(ROOT/'REVIEWED_PINS.json')
            if review['sources']!=source_hashes() or review['config']!=identity(HERE/'draft.json'):
                raise RuntimeError('candidate changed after review')
            tested=read(Path(focused[-1]['attempt'])/'started.json')
            if tested['sources']!=source_hashes():raise RuntimeError('candidate changed after checks')
            work=450.;shutdown=30.
        else:
            if len(focused)>=2:raise RuntimeError('two focused invocations consumed')
            work=min(55.,60.-sum(r['wall_seconds'] for r in focused)-5.)
            shutdown=5.
        if work<=0 or used+work+shutdown>600:raise RuntimeError('full child envelope cannot fit')
        if sum(p.stat().st_size for p in EVIDENCE.rglob('*') if p.is_file())>=128*1024**2:
            raise RuntimeError('evidence ceiling reached')
        attempt=ledger/(time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]);attempt.mkdir()
        if args.kind=='measurement':write(ledger/'MEASUREMENT_CONSUMED.json',dict(attempt=str(attempt),recorded_unix=time.time()))
        counter=ledger/'COUNTERS.json'
        if not counter.exists():write(counter,dict(limits=LIMITS,counts=CARRIED))
        began=time.monotonic();deadline=time.time()+work
        command=[word.replace('{attempt}',str(attempt)) for word in command]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                 TPFA_ATTEMPT=str(attempt),TPFA_BUDGET=str(counter),TPFA_LIMITS=json.dumps(LIMITS),
                 TPFA_DEADLINE=str(deadline),TPFA_CATEGORY='part_b_'+args.kind)
        write(attempt/'started.json',dict(kind=args.kind,command=command,work_deadline_unix=deadline,
              work_seconds=work,shutdown_reserved=shutdown,prior_wall_seconds=used,counter_before=read(counter),
              sources=source_hashes(),config=identity(HERE/'draft.json'),started_unix=time.time()))
        child=None;timed_out=False;stop=None;cleanup=[];code=None
        with (attempt/'stdout.txt').open('wb') as out,(attempt/'stderr.txt').open('wb') as err:
            try:
                child=subprocess.Popen(command,stdout=out,stderr=err,env=env)
                while child.poll() is None:
                    remaining=deadline-time.time()
                    if remaining<=0:timed_out=True;stop='immutable work deadline';break
                    if sum(p.stat().st_size for p in EVIDENCE.rglob('*') if p.is_file())>=128*1024**2:
                        stop='128 MiB evidence ceiling';break
                    try:child.wait(timeout=min(.5,remaining))
                    except subprocess.TimeoutExpired:pass
                if stop:
                    cleanup_deadline=time.monotonic()+shutdown
                    if os.name=='nt':
                        killed=subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,
                                              timeout=max(.1,min(10,shutdown/2)))
                        cleanup.append(dict(exit_code=killed.returncode,stdout=killed.stdout.decode(errors='replace'),
                                            stderr=killed.stderr.decode(errors='replace')))
                    else:child.kill()
                    child.wait(timeout=max(.1,cleanup_deadline-time.monotonic()))
                code=child.returncode
            except BaseException as exc:
                stop=stop or type(exc).__name__+': '+str(exc)
                if child is not None and child.poll() is None:
                    try:
                        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=5)
                        child.wait(timeout=5)
                    except Exception as close:cleanup.append(dict(error=str(close)))
        record=dict(kind=args.kind,attempt=str(attempt),command=command,pid=None if child is None else child.pid,
                    wall_seconds=time.monotonic()-began,exit_code=code,
                    timed_out=timed_out or (attempt/'TIMEOUT.json').exists(),stop=stop,cleanup=cleanup,
                    owned_process_ended=child is None or child.poll() is not None,counts=read(counter))
        write(attempt/'result.json',record);print(json.dumps(record))
        return 124 if record['timed_out'] else (code if code is not None else 125)
    finally:os.close(fd);lock.unlink()


if __name__=='__main__':raise SystemExit(main())
