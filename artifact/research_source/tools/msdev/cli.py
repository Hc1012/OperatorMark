"""Explicit later-run entry points. No real job is started by preparation of source."""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from .records import read,write,sha,verify,verify_return,extract,safe_name
from .campaign import begin,close_session,Campaign,finish_manifest,prepare_output_inventory,SessionBoundary
from .graph import operation_graph
from .prepare import bindings

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def runtime():
    import importlib.metadata as md
    import torch
    want={'numpy':'2.5.2','scipy':'1.18.1','neuraloperator':'2.0.0','tensorly':'0.9.0','tensorly-torch':'0.5.0'}
    versions={k:md.version(k) for k in want}
    if versions!=want or torch.__version__.split('+')[0]!='2.12.0':raise ValueError('dependency drift; no automatic installation')
    return dict(python=sys.version,platform=platform.platform(),versions=versions,torch=torch.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        precision='FP32/complex64, no AMP/TF32',threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())

def reusable_q6_runtime(bundle,current):
    prior=read(Path(bundle)/'Q6_ENVIRONMENT.json');evidence=read(Path(bundle)/'Q6_COMPATIBILITY.json')
    if not evidence.get('passed') or len(evidence.get('rows',[]))!=6 or any(r.get('passed') is not True for r in evidence['rows']):
        raise ValueError('Q6 six-fixture parity evidence incomplete')
    for key in ('python','torch','cuda','cudnn','gpu'):
        if current[key]!=prior[key]:raise ValueError('Q6 parity runtime applicability differs: '+key+'; no automatic replay')
    for key,value in current['versions'].items():
        if prior['packages'].get(key)!=value:raise ValueError('Q6 parity dependency applicability differs: '+key)
    return dict(reused=True,receipt_sha256=sha(Path(bundle)/'Q6_COMPATIBILITY.json'),new_numerical_check=False)

def worker(args):
    root=args.out
    try:
        if args.mode=='windows-worker':
            from .prepare import run
            c,b=bindings(args.config,args.pins);run(c,root,args.pins)
        else:
            from .backend import Backend
            manifest=read(args.bundle/'MANIFEST.json')
            if sha(args.bundle/'MANIFEST.json')!=args.manifest_sha256:raise ValueError('upload manifest binding drift')
            verify(args.bundle,manifest['files']);config=read(args.bundle/'CONFIG.json')
            if read(root/'STATUS.json')['bindings']!=dict(config['bindings'],upload=args.manifest_sha256):raise ValueError('campaign input binding mismatch')
            import q5_model as M
            M.configure();r=runtime();reusable_q6_runtime(args.bundle,r)
            if config.get('continuation_kind')=='unmarked_remaining_v1':
                from .unmarked import make_campaign
                campaign=make_campaign(args.bundle,root,r)
            else:
                backend=Backend(args.bundle,root)
                campaign=Campaign(root,backend,operation_graph(),read(root/'STATUS.json')['bindings'],r)
            if args.resume and args.incident:
                old=read(root/safe_name(args.prior_closure))
                if old.get('bindings')!=campaign.bindings or old.get('timed_out'):raise ValueError('recovery closure/binding mismatch')
                campaign.recovery(args.incident,old['closure'])
            campaign.run(resume=args.resume);prepare_output_inventory(root,campaign.check)
    except SessionBoundary:
        return 5
    except BaseException as exc:
        write(root/f'WORKER_FAILURE_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    return 0

def windows(args):
    c,b=bindings(args.config,args.pins)
    if not args.execute_development:raise ValueError('explicit later development launch required')
    if Path(sys.executable).resolve()!=Path(c['interpreter']).resolve():raise ValueError('pinned Windows interpreter required')
    if args.credits is None or args.billing_rate is None or min(args.credits,args.billing_rate)<=0:raise ValueError('supply current credits/billing rate before real development begins')
    if args.credits<8.75*args.billing_rate:raise ValueError('supplied credits do not cover proposed attached ceiling; no purchase')
    root=args.out
    if (root/'STATUS.json').exists():
        if not args.resume:raise ValueError('existing campaign: explicit infrastructure recovery required')
        s=read(root/'STATUS.json')
        old=read(root/safe_name(args.prior_closure))
        if args.incident!='infrastructure' or not old.get('cleanup',{}).get('resolved') or s.get('recoveries',{}).get('windows_preparation'):
            raise ValueError('Windows recovery ineligible or closure unresolved')
        if s.get('state') in ('WINDOWS_COMPLETE','COMPLETE') or old.get('timed_out'):raise ValueError('completion/cap is not recovery')
        if list(root.glob('WORKER_FAILURE_*.json')):raise ValueError('recorded scientific/unknown failure requires diagnosis; no automatic replay')
        if not (root/'PRIVATE_SEEDS.json').exists():raise ValueError('missing original persisted development seeds; never redraw on recovery')
        if not old.get('job_handle_closed') or 'windows-worker' not in old.get('command',[]) or str(root) not in old['command']:
            raise ValueError('closure does not bind this Windows worker')
        if not s['session']['closed']:
            # After-interruption reconciliation, not a fabricated historical receipt.
            # Conservatively charge through this established current readout.
            close_session(root,True);s=read(root/'STATUS.json')
        s['recoveries']['windows_preparation']=1;write(root/'STATUS.json',s,replace=True)
    remaining=7200-read(root/'STATUS.json').get('windows_charged',0) if (root/'STATUS.json').exists() else 7200
    seconds=min(remaining,args.seconds)
    s=begin(root,b,'windows',seconds)
    import qualification_controller as Q
    owned=Q.wrapper(read(c['q6_config'])) # unchanged Q0 wrapper plus already-qualified 8GB Job limit
    attempt=root/f'windows_attempt_{time.time_ns()}';attempt.mkdir()
    cmd=[sys.executable,'-B',str(Path(__file__).parents[1]/'mainstudy_development.py'),'windows-worker','--config',str(args.config),
        '--pins',str(args.pins),'--out',str(root)]
    from q6_common import THREAD_ENV
    work=min(s['session']['end'],s['deadline'])-time.time()-31
    if work<=0:raise TimeoutError('Windows launch envelope no longer fits')
    record=owned.run_with_timeout(cmd,cwd=Path(c['repo']),env=dict(os.environ,**THREAD_ENV),timeout_s=work,
        cleanup_timeout_s=30,out_dir=attempt,stdout_name='stdout.txt')
    closed=record.get('cleanup',{}).get('resolved') is True
    close_session(root,closed)
    print(str(attempt/'invocation.json'),flush=True)
    return 0 if closed and record.get('exit_code')==0 and not record.get('timed_out') and (root/'READY_FOR_COLAB.json').exists() else 4

def reserve_colab(args):
    if not args.execute_development:raise ValueError('explicit later development launch required')
    if os.name!='posix':raise ValueError('Colab Linux route required')
    root=args.out;config=read(args.bundle/'CONFIG.json');m=read(args.bundle/'MANIFEST.json')
    if sha(args.bundle/'MANIFEST.json')!=args.manifest_sha256:raise ValueError('manifest identity')
    verify(args.bundle,m['files'])
    imported=read(args.windows_status)
    if imported['state']!='WINDOWS_COMPLETE' or not imported['session']['closed']:raise ValueError('Windows generation/closure incomplete')
    root.mkdir(parents=True,exist_ok=True)
    p=root/'STATUS.json'
    if not p.exists():
        s=dict(imported);s['bindings']=dict(config['bindings'],upload=args.manifest_sha256);s['completed']=[];s['products']=[]
        write(p,s)
    elif read(p)['bindings']!=dict(config['bindings'],upload=args.manifest_sha256):raise ValueError('resume bindings differ')
    previous=read(p)
    if previous.get('state')=='COMPLETE':raise ValueError('completed campaign cannot run again')
    if previous.get('state') not in ('WINDOWS_COMPLETE','PAUSED_SESSION') and not (args.resume and args.incident=='infrastructure'):
        raise ValueError('explicit eligible infrastructure recovery required')
    if previous.get('failure_kind') in ('scientific','cap','unknown'):raise ValueError('scientific/cap/unknown failure blocks continuation')
    if previous.get('state')=='PAUSED_SESSION':args.resume=True
    if args.incident:
        old=read(root/safe_name(args.prior_closure))
        if old.get('bindings')!=previous['bindings'] or old.get('timed_out') or not old.get('closure',{}).get('resolved'):
            raise ValueError('matched closed predecessor required for infrastructure recovery')
        arm=previous.get('active_arm')
        if previous.get('recoveries',{}).get(arm) or not previous.get('checkpoint'):raise ValueError('recovery consumed or checkpoint missing')
    # All reserved attached time remains charged: no unobserved post-receipt idle refund.
    s=begin(root,read(p)['bindings'],'cuda',args.seconds,args.credits,args.billing_rate)
    s['session'].update(prebooked=True,launch_started=False,previous_state=previous['state'],resume=bool(args.resume),
                        incident=args.incident,prior_closure=args.prior_closure);write(p,s,replace=True)
    print(dict(reserved_before_GPU_connection=True,deadline=s['session']['end'],total_charged=s['gpu_charged']))
    return 0

def colab(args):
    if not args.execute_development or os.name!='posix':raise ValueError('explicit later Linux development launch required')
    root=args.out;p=root/'STATUS.json';s=read(p);config=read(args.bundle/'CONFIG.json')
    if sha(args.bundle/'MANIFEST.json')!=args.manifest_sha256 or s['bindings']!=dict(config['bindings'],upload=args.manifest_sha256):raise ValueError('prebooked input/source binding differs')
    verify(args.bundle,read(args.bundle/'MANIFEST.json')['files'])
    session=s['session']
    if not session.get('prebooked') or session.get('launch_started') or session['closed']:raise ValueError('one prebooked unused GPU session required')
    if time.time()>=min(session['end']-30,s['deadline']-30):raise TimeoutError('prebooked window expired')
    args.resume=session.get('resume',False);args.incident=session.get('incident');args.prior_closure=session.get('prior_closure')
    args.seconds=session['reserved'];args.attached_at=session['start']
    s['session']['launch_started']=True
    # Nested ownership: worker cleanup30s, then supervisor stream/receipt closure30s
    # within the SAME attached deadline; no reservation or clock increase.
    s['session']['worker_deadline']=s['session']['end']-60;write(p,s,replace=True)
    deadline=s['session']['end'];owned=module(args.bundle/'code/q5_colab_supervisor.py','msdev_qualified_linux')
    import ctypes,signal
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError('subreaper unavailable')
    attempt=root/f'colab_attempt_{time.time_ns()}';attempt.mkdir();child=None;identity=None;r=dict(timed_out=False,bindings=s['bindings'],session=dict(session))
    command=[sys.executable,'-B',str(args.bundle/'code/mainstudy_development.py'),'gpu-worker','--bundle',str(args.bundle),
             '--out',str(root),'--manifest-sha256',args.manifest_sha256]
    if args.resume:
        command+=['--resume']
        if args.incident:command+=['--incident',args.incident,'--prior-closure',args.prior_closure]
    r['command']=command
    def interrupted(signum,frame):raise InterruptedError('supervisor signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        from q6_common import THREAD_ENV
        from .bounded_capture import capture
        now=time.monotonic();remaining=max(0,deadline-time.time())
        r.update(capture(command,cwd=None,
            env=dict(os.environ,**THREAD_ENV,CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled',WANDB_DISABLED='true'),
            work_deadline=now+max(0,remaining-60),cleanup_deadline=now+max(0,remaining-30),
            folder=attempt,supervisor=owned,on_identity=lambda identity:write(attempt/'WORKER_IDENTITY.json',identity)))
    except BaseException as exc:
        # capture owns cleanup; preserve unavailable closure rather than infer it.
        r['error']=type(exc).__name__
    stream_ok=(r.get('stream_overflow') is False and r.get('stdout_closed') is True and
               not r.get('errors') and bool(r.get('streams')) and
               all(row.get('capture_complete') is True for row in r['streams'].values()))
    r['complete']=bool(stream_ok and r.get('returncode')==0 and r.get('closure',{}).get('resolved') and not r['timed_out'] and not r.get('error') and (root/'RESULT.json').exists())
    r['paused']=bool(stream_ok and r.get('returncode')==5 and r.get('closure',{}).get('resolved') and not r['timed_out'] and read(p).get('state')=='PAUSED_SESSION')
    r['attached_reserved_seconds']=args.seconds;r['attached_seconds_at_receipt']=time.time()-args.attached_at
    write(attempt/'SUPERVISOR.json',r)
    if r.get('closure',{}).get('resolved'):
        s=read(p);s['session']['closed']=True;s['session']['closure']=str((attempt/'SUPERVISOR.json').relative_to(root));write(p,s,replace=True)
    if r['complete']:
        # The notebook's pinned-interpreter parent finalizes after THIS supervisor
        # exits and its captured stream closes; never hash a still-live replay log.
        write(root/'SUPERVISOR.json',r)
    print(r,flush=True);return 0 if r['complete'] else 5 if r['paused'] else 4

def main(argv=None):
    p=argparse.ArgumentParser(description='Prepared DEVELOPMENT ONLY; real execution requires a later explicit instruction')
    p.add_argument('mode',choices=['prepare','windows-worker','reserve-colab','colab','gpu-worker','status','verify-return','verify-unmarked-return','return','check-runtime'])
    for n in ('config','pins','out','bundle','windows-status','ready','archive'):p.add_argument('--'+n,type=Path)
    p.add_argument('--manifest-sha256');p.add_argument('--seconds',type=float,default=7200)
    p.add_argument('--credits',type=float);p.add_argument('--billing-rate',type=float);p.add_argument('--attached-at',type=float)
    p.add_argument('--execute-development',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--incident',choices=['infrastructure']);p.add_argument('--prior-closure')
    a=p.parse_args(argv)
    if a.mode=='check-runtime':
        import q5_model as M
        M.configure();r=runtime();receipt=reusable_q6_runtime(a.bundle,r)
        write(a.out/'RUNTIME.json',dict(environment=r,parity=receipt,interpreter=sys.executable));return 0
    if a.mode=='status':print((a.out/'STATUS.json').read_text(encoding='utf-8'));return 0
    if a.mode=='verify-unmarked-return':
        result=read(a.out/'RESULT.json')
        if not a.manifest_sha256 or result.get('scope')!='unmarked remaining development only' or result.get('bindings',{}).get('upload')!=a.manifest_sha256:
            raise ValueError('expected unmarked upload/scope binding required')
        print(verify_return(a.out));return 0
    if a.mode in ('verify-return','return'):
        c,b=bindings(a.config,a.pins);ready=read(a.ready)
        expected=dict(b,upload=ready['manifest_sha256'])
        if a.mode=='return':
            extract(a.archive,a.out)
            manifests=list(a.out.rglob('OUTPUT_MANIFEST.json'))
            if len(manifests)!=1:raise ValueError('one complete producer output manifest required')
            a.out=manifests[0].parent
        if read(a.out/'RESULT.json').get('bindings')!=expected:raise ValueError('returned source/config/input bindings differ')
        receipt=verify_return(a.out)
        destination=a.out/f'WINDOWS_RETURN_VERIFICATION_{time.time_ns()}.json';write(destination,receipt)
        print(dict(receipt,record=str(destination)));return 0
    if a.mode.endswith('worker'):return worker(a)
    if a.mode=='prepare':return windows(a)
    if a.mode=='reserve-colab':return reserve_colab(a)
    return colab(a)
