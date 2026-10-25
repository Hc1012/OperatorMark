"""One fixed final campaign, immutable sixteen-hour window, one classified continuation."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from core import HERE,read,write,identity,source_hashes,validate_config
from final_contract import validate_final_config
from dev_process import OwnedJob,process_identity,require_dead,available_ram,publish_handshake,wait_dead

RECORDS=HERE/'evidence/final_study_20260930'
LIMITS=dict(solver_attempts=33000,model_updates=235000,inverse_updates=40000)


def validate_pins(authority):
    if authority['source_sha256']!=source_hashes():raise ValueError('reviewed source identities changed')
    for name,path in (('config_identity',HERE/'final.json'),('rule_identity',HERE/'RULE_FINAL.md')):
        if identity(path)!=authority[name]:raise ValueError('reviewed final rule/configuration changed')
    for fact in authority['development_inputs'].values():
        if identity(fact['path'])!=fact:raise ValueError('development identity changed')
    for fact in authority.get('verification_inputs',[]):
        if identity(fact['path'])!=fact:raise ValueError('verification identity changed')


def validate_run(root):
    authority=read(root/'AUTHORITY.json');validate_pins(authority)
    if identity(root/'CONFIG.json')['sha256']!=authority['config_identity']['sha256']:
        raise ValueError('campaign configuration changed')
    started=root/'attempt_0/started.json'
    if started.exists() and read(started)['authority']!=identity(root/'AUTHORITY.json'):
        raise ValueError('original authority/deadline changed')
    return authority

def resume_allowed(root):
    if (root/'attempt_1').exists():raise RuntimeError('one continuation already consumed')
    if (root/'COMPLETED.json').exists():raise RuntimeError('completed campaign cannot resume')
    prior=read(root/'attempt_0/result.json')
    if not prior['owned_worker_ended'] or not prior['infrastructure']:
        raise RuntimeError('prior stop is not a verified infrastructure interruption')
    require_dead(prior['worker']);require_dead(prior['launcher']);require_dead(prior['supervisor'])
    if not (root/'progress.jsonl').exists():raise ValueError('no recovery progress')
    if time.time()>=read(root/'AUTHORITY.json')['deadline_unix']:raise TimeoutError('original work deadline expired')
    return prior


def supervise(root,attempt,*,fixture=False):
    authority=read(root/'AUTHORITY.json')
    if not fixture:validate_run(root)
    deadline=authority['deadline_unix'];closure=authority['closure_deadline_unix']
    if time.time()>=deadline:raise TimeoutError('immutable work deadline expired')
    attempt_path=root/('attempt_'+str(attempt));attempt_path.mkdir(exist_ok=False)
    command=[sys.executable,'-B',str(HERE/'final_worker.py'),'--output',str(root),'--attempt',str(attempt)]
    if fixture:command+=['--fixture']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    supervisor=process_identity(os.getpid());worker=None;launcher=None;child=None;job=None;stop=None;infrastructure=False;cleanup=[]
    start=time.time()
    write(attempt_path/'started.json',dict(command=command,supervisor=supervisor,authority=identity(root/'AUTHORITY.json'),
                                         started_unix=start,work_deadline_unix=deadline,closure_deadline_unix=closure))
    try:
        job=OwnedJob()
        with (attempt_path/'stdout.txt').open('xb') as out,(attempt_path/'stderr.txt').open('xb') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err,env=env)
            job.assign(child);launcher=process_identity(child.pid)
            handoff_end=min(deadline,time.time()+5)
            while not (attempt_path/'HELLO.json').exists():
                if child.poll() is not None or time.time()>=handoff_end:raise RuntimeError('worker identity handshake unavailable')
                time.sleep(.02)
            hello=read(attempt_path/'HELLO.json');worker=hello['worker']
            expected_parent=os.getpid() if worker['pid']==child.pid else child.pid
            if hello['parent_pid']!=expected_parent or worker!=process_identity(worker['pid']):
                raise RuntimeError('worker/virtualenv-launcher ancestry mismatch')
            job.assign_pid(worker['pid'])
            if worker is None:raise RuntimeError('worker ended before binding')
            publish_handshake(attempt_path/'BOUND.json',dict(worker=worker,launcher=launcher,supervisor=supervisor,job_kill_on_close=True))
            while child.poll() is None:
                if time.time()>=deadline:stop='work deadline';break
                request=root/('INTERRUPT_REQUEST_'+str(attempt)+'.json')
                if request.exists():
                    record=read(request)
                    if record.get('reason')!='external operator interruption':raise ValueError('invalid interruption request')
                    stop='documented external interruption';infrastructure=True;break
                try:child.wait(timeout=min(.5,max(.001,deadline-time.time())))
                except subprocess.TimeoutExpired:pass
            if stop:
                job.close();child.wait(timeout=max(.1,min(30,closure-time.time())))
            code=child.wait(timeout=max(.1,min(30,closure-time.time())))
            failed=root/('INCOMPLETE_attempt'+str(attempt)+'.json')
            if failed.exists():
                failed_record=read(failed);infrastructure=failed_record.get('infrastructure') is True
                if failed_record.get('type')=='TimeoutError':stop='work deadline';infrastructure=False
            if code in (0xC000013A,-1073741510) and not failed.exists():
                infrastructure=True;stop='Windows console interruption exit status'
            if stop=='work deadline':infrastructure=False
    except BaseException as exc:
        stop=stop or type(exc).__name__+': '+str(exc);infrastructure=False
        if job:job.close()
        if child is not None:
            try:child.wait(timeout=max(.1,min(30,closure-time.time())))
            except Exception as error:cleanup.append(str(error))
    finally:
        if job:job.close()
    ended=child is not None and child.poll() is not None
    if worker is not None:
        try:wait_dead(worker,max(0,min(30,closure-time.time())))
        except Exception as exc:ended=False;cleanup.append(str(exc))
    result=dict(supervisor=supervisor,worker=worker,launcher=launcher,exit_code=None if child is None else child.returncode,
        stop=stop,infrastructure=infrastructure and ended,owned_worker_ended=ended,cleanup=cleanup,
        elapsed_attempt_seconds=time.time()-start,elapsed_campaign_seconds=time.time()-authority['started_unix'],
        work_deadline_unix=deadline,closure_deadline_unix=closure,
        scientifically_complete=stop is None and ended and child.returncode==0 and (root/'COMPLETED.json').exists())
    from final_summary import terminal_summary
    try:terminal_summary(root,attempt,result)
    except Exception as exc:
        result['scientifically_complete']=False
        result['infrastructure']=False  # Corrupt/ambiguous terminal evidence cannot authorize continuation.
        result['terminal_readback_error']=type(exc).__name__+': '+str(exc)
        write(attempt_path/'TERMINAL_READBACK_ERROR.json',result)
    result['elapsed_attempt_seconds']=time.time()-start
    result['elapsed_campaign_seconds']=time.time()-authority['started_unix']
    if time.time()>closure:
        result['scientifically_complete']=False;result['stop']='closure deadline exceeded'
        result['infrastructure']=False
    write(attempt_path/'result.json',result)
    if not result['scientifically_complete']:
        write(attempt_path/'INCOMPLETE.json',dict(**result,last_progress='progress.jsonl',
                continuation_available=result['infrastructure'] and attempt==0 and time.time()<deadline))
    print(json.dumps(result),flush=True)
    return 0 if result['scientifically_complete'] else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['start','resume','interrupt','status','fixture-start','fixture-resume'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);root=args.output.resolve();fixture=args.command.startswith('fixture-')
    if fixture and os.environ.get('TPFA_CATEGORY')!='final_study_verification':
        raise PermissionError('fixture unusable outside bounded verification')
    if args.command=='status':
        for path in [root/'COMPLETED.json',root/'attempt_0/result.json',root/'attempt_1/result.json']:
            if path.exists():print(path.name+': '+path.read_text())
        if (root/'progress.jsonl').exists():print('\n'.join((root/'progress.jsonl').read_text().splitlines()[-4:]))
        return 0
    if args.command=='interrupt':
        attempt=1 if (root/'attempt_1').exists() else 0
        write(root/('INTERRUPT_REQUEST_'+str(attempt)+'.json'),dict(reason='external operator interruption',unix=time.time()))
        return 0
    if args.command.endswith('resume'):
        resume_allowed(root)
        return supervise(root,1,fixture=fixture)
    if fixture:return supervise(root,0,fixture=True)
    grant=read(RECORDS/'GRANT.json')
    ready=read(RECORDS/'READY.json');validate_pins(ready)
    if ready.get('engineering_completed_unix',float('inf'))>datetime.fromisoformat(grant['engineering_deadline_utc']).timestamp():
        raise TimeoutError('engineering handoff was not completed within its allocation')
    if not ready['all_required_checks_passed'] or not ready['stage0_compatible']:raise PermissionError('prerequisites incomplete')
    config=read(HERE/'final.json')
    validate_final_config(config)
    if not root.is_relative_to(HERE/'private_runs'):raise ValueError('private final output required')
    root.parent.mkdir(exist_ok=True)
    free=shutil.disk_usage(root.parent).free;ram=available_ram()
    if free<16*1024**3:raise RuntimeError('less than 16 GiB output-volume space')
    if ram['available_bytes']<ready['required_available_ram_bytes']:raise RuntimeError('insufficient documented available RAM')
    if root.exists():raise FileExistsError('fresh campaign output required')
    write(RECORDS/'SCIENTIFIC_CONSUMED.json',dict(output=str(root),recorded_unix=time.time()))
    root.mkdir();now=time.time()
    authority=dict(scope='final',execution_id='tpfa32_final_study_20260930',approved_by_user=True,
        started_unix=now,deadline_unix=now+57300,closure_deadline_unix=now+57600,limits=LIMITS,
        evidence_bytes=8*1024**3,source_sha256=source_hashes(),config_identity=identity(HERE/'final.json'),
        rule_identity=identity(HERE/'RULE_FINAL.md'),development_inputs=ready['development_inputs'],
        verification_inputs=ready['verification_inputs'],runtime=ready['runtime'],free_disk_bytes=free,ram=ram,
        required_available_ram_bytes=ready['required_available_ram_bytes'],ram_estimate=ready['ram_estimate'],
        verification=identity(RECORDS/'READY.json'))
    write(root/'AUTHORITY.json',authority)
    # Keep exact reviewed configuration bytes, not a differently formatted copy.
    with (root/'CONFIG.json').open('xb') as out:out.write((HERE/'final.json').read_bytes())
    return supervise(root,0)


if __name__=='__main__':raise SystemExit(main())
