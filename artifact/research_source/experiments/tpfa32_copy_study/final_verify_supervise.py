"""Final-study focused allowance only: three invocations,600 seconds including closure."""
import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
import time
import uuid
from core import HERE,read,write,source_hashes,identity
from dev_process import OwnedJob,process_identity

ROOT=HERE/'evidence/final_study_20260930'


def main():
    p=argparse.ArgumentParser();p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    grant=read(ROOT/'GRANT.json');engineering=datetime.fromisoformat(grant['engineering_deadline_utc']).timestamp()
    ledger=ROOT/'verification';ledger.mkdir(exist_ok=True)
    prior=[read(p) for p in ledger.glob('*/result.json')]
    if len(list(ledger.glob('*/started.json')))!=len(prior):raise RuntimeError('unresolved verification child')
    if len(prior)>=3:raise RuntimeError('three invocations consumed')
    if any(not r['closed'] for r in prior):raise RuntimeError('unresolved owned verification process')
    used=sum(r['wall_seconds'] for r in prior);envelope=min(300.,600.-used,engineering-time.time())
    if envelope<=5:raise RuntimeError('verification envelope cannot fit')
    attempt=ledger/(time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]);attempt.mkdir()
    counter=ledger/'COUNTERS.json';limits=dict(solver_attempts=0,model_updates=64,inverse_updates=128)
    if not counter.exists():write(counter,dict(limits=limits,counts={k:0 for k in limits}))
    begin=time.monotonic();deadline=time.time()+envelope-5
    command=[s.replace('{attempt}',str(attempt)) for s in command]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             TPFA_CATEGORY='final_study_verification',TPFA_ATTEMPT=str(attempt),TPFA_BUDGET=str(counter),
             TPFA_LIMITS=json.dumps(limits),TPFA_DEADLINE=str(deadline))
    write(attempt/'started.json',dict(command=command,sources=source_hashes(),config=identity(HERE/'final.json'),
          deadline_unix=deadline,closure_seconds=5,prior_seconds=used,counts_before=read(counter)))
    job=OwnedJob();child=None;stop=None
    try:
        with (attempt/'stdout.txt').open('xb') as out,(attempt/'stderr.txt').open('xb') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err,env=env);job.assign(child)
            try:child.wait(timeout=max(.01,deadline-time.time()))
            except subprocess.TimeoutExpired:stop='focused deadline';job.close();child.wait(timeout=5)
    except BaseException as exc:
        stop=type(exc).__name__+': '+str(exc);job.close()
        if child is not None:child.wait(timeout=5)
    finally:job.close()
    record=dict(command=command,wall_seconds=time.monotonic()-begin,exit_code=None if child is None else child.returncode,
                stop=stop,closed=child is not None and child.poll() is not None,counts=read(counter))
    write(attempt/'result.json',record);print(json.dumps(dict(attempt=str(attempt),**record)))
    return 0 if record['exit_code']==0 and stop is None else 1


if __name__=='__main__':raise SystemExit(main())
