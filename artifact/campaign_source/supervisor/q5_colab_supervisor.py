"""Colab-only Q5 supervisor. No solver/data-generation/CPU fallback route."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(expected):
    if digest(ROOT/'UPLOAD_MANIFEST.json') != expected:
        raise ValueError('upload manifest identity mismatch')
    manifest=read(ROOT/'UPLOAD_MANIFEST.json')
    actual={p.name for p in ROOT.iterdir() if p.is_file()}
    if actual != set(manifest['files']) | {'UPLOAD_MANIFEST.json'}:
        raise ValueError('unexpected or missing bundle files')
    for name, facts in manifest['files'].items():
        if Path(name).name != name or digest(ROOT/name) != facts['sha256'] or (ROOT/name).stat().st_size != facts['bytes']:
            raise ValueError('bundle identity mismatch: '+name)
    return manifest


def command_output(command, timeout=15):
    try:
        p=subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except Exception as exc:
        return dict(command=command,error=repr(exc))


def worker(out, deadline, expected):
    verify(expected)
    baseline=read(ROOT/'WINDOWS_BASELINE.json')
    env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),machine=platform.machine(),
             packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
             thread_environment={k:os.environ.get(k) for k in baseline['thread_environment']},
             cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
             nvidia_smi=command_output(['nvidia-smi']),
             os='Linux',cpu=platform.processor(),cpu_count=os.cpu_count())
    import torch
    env.update(torch=torch.__version__,cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
               cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        prop=torch.cuda.get_device_properties(0)
        env.update(gpu_name=prop.name,gpu_total_memory=prop.total_memory,
                   gpu_capability=list(torch.cuda.get_device_capability(0)),
                   compiled_cuda_architectures=torch.cuda.get_arch_list())
    env['differences_from_windows']={k:dict(windows=v,colab=env.get(k)) for k,v in baseline.items()
         if k not in ('notes','packages') and v!=env.get(k)}
    env['package_differences']={k:dict(windows=v,colab=env['packages'].get(k))
         for k,v in baseline['packages'].items() if v!=env['packages'].get(k)}
    save(out/'ENVIRONMENT.json',env)
    if not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError('CUDA GPU required; no CPU fallback. Check Colab GPU/driver and pinned CUDA wheel compatibility.')
    import q5_model as M
    # Compare a fresh local CPU permutation generator against every saved CPU batch.
    # train() creates the same generator again; this check performs no model work.
    cpu=read(ROOT/'CPU_REFERENCE.json')
    manifest=read(ROOT/'MANIFEST.json')
    batches=[x.tolist() for x in M.batches(manifest['shuffle_seed'])]
    if batches != [r['batch_indices'] for r in cpu['updates']]:
        raise ValueError('batch order differs from the actual Windows benchmark')
    save(out/'INPUT_VERIFICATION.json',dict(upload_manifest_sha256=expected,
        handoff_manifest_sha256=digest(ROOT/'MANIFEST.json'),batch_order_exact=True,
        samples=64,updates=200,warmup_updates=20,device='cuda'))
    M.train(ROOT,out/'training','cuda',deadline)
    verify(expected)


def process_identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text()
        fields=text[text.rfind(')')+2:].split()
        return dict(pid=pid,state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),
                    sid=int(fields[3]),start_ticks=int(fields[19]))
    except FileNotFoundError:
        return None


def members(pgid, sid):
    found=[]
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            row=process_identity(int(p.name))
            if row and row['pgid']==pgid and row['sid']==sid:
                found.append(row)
    return found


def cleanup(child, identity, deadline):
    evidence=dict(initial_members=[],signals=[],errors=[])
    try:
        pgid=sid=child.pid
        live=process_identity(child.pid)
        if live and live['start_ticks']!=identity['start_ticks']:
            raise RuntimeError('worker identity changed; refusing to signal')
        evidence['initial_members']=members(pgid,sid)
        for sig, grace in ((signal.SIGTERM,5),(signal.SIGKILL,20)):
            remaining=members(pgid,sid)
            if not remaining:break
            if time.monotonic()>=deadline:break
            try:
                os.killpg(pgid,sig);evidence['signals'].append(sig.name)
            except ProcessLookupError:
                pass
            until=min(deadline,time.monotonic()+grace)
            while time.monotonic()<until:
                child.poll()
                # This fresh supervisor is a subreaper and has no other live children.
                if child.returncode is not None:
                    try:
                        while os.waitpid(-1,os.WNOHANG)[0]:pass
                    except ChildProcessError:pass
                if not members(pgid,sid):break
                time.sleep(min(.1,max(0,until-time.monotonic())))
        child.poll()
        evidence.update(returncode=child.returncode,survivors=members(pgid,sid),
                        child_reaped=child.returncode is not None)
        evidence['resolved']=evidence['child_reaped'] and not evidence['survivors']
    except BaseException as exc:
        evidence['errors'].append(repr(exc));evidence['resolved']=False
    return evidence


def summary(out, receipt):
    cpu=read(ROOT/'CPU_REFERENCE.json')
    path=out/'training/updates.jsonl'
    rows=[];parse_error=None
    if path.exists():
        for line in path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except Exception as exc:parse_error=repr(exc);break
    okay=(receipt.get('returncode')==0 and not receipt['timed_out'] and not receipt.get('error')
          and receipt.get('closure',{}).get('resolved') and not parse_error and len(rows)==200
          and (out/'training/RESULT.json').exists())
    checks=[]
    for i,r in enumerate(rows):
        checks.append(r.get('update')==i+1 and r.get('batch_indices')==cpu['updates'][i]['batch_indices']
            and r.get('phase')==('warmup' if i<20 else 'sustained') and r.get('finite_gradients') is True
            and all(math.isfinite(r[k]) for k in ('loss','forward_loss','jvp_loss','seconds')) and r['seconds']>0)
    okay=bool(okay and all(checks))
    result=dict(complete=okay,updates_recorded=len(rows),parse_error=parse_error,
        comparison_is_not_bitwise_training_equivalence=True,cpu_reference_seconds_per_update=4.575,
        no_qualification_or_convergence_claim=True,supervision=receipt)
    if rows:
        result['numerical_differences']={k:dict(cpu_first=cpu['updates'][0][k],gpu_first=rows[0][k],
            cpu_at_last_recorded_update=cpu['updates'][len(rows)-1][k],gpu_last=rows[-1][k],
            max_absolute_difference=max(abs(r[k]-cpu['updates'][i][k]) for i,r in enumerate(rows)))
            for k in ('loss','forward_loss','jvp_loss')}
    if okay:
        measured=read(out/'training/RESULT.json')
        if not (measured['complete'] and measured['updates']==200 and measured['warmup_updates']==20
                and measured['sustained_updates']==180 and measured['device']=='cuda'):
            raise ValueError('worker result population mismatch')
        rate=sum(r['seconds'] for r in rows[20:])/180
        result.update(seconds_per_sustained_update=rate,speedup_vs_windows=4.575/rate,
            updates_per_second=1/rate,examples_per_second=16/rate,worker_measurements=measured,
            training_projections=[dict(L=L,cumulative_owner_updates=n,owner_hours=n*rate/3600,
                separate_10pct_baseline_updates=1.05*L,separate_10pct_baseline_hours=1.05*L*rate/3600)
                for L,n in [(5120,5120),(10240,10752),(20480,22016),(40960,44544),(81920,89600)]],
            projection_exclusions='evaluation, larger-dataset effects, checkpoint/cooldown copy costs, generation and readback; not measured or zero')
    save(out/'SUMMARY.json',result)
    return result


def supervise(out, expected):
    start=time.monotonic();whole_end=start+1800;work_end=start+1760
    out.mkdir(exist_ok=False)  # Durable consumed marker; never automatically retry.
    receipt=dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        envelope_seconds=1800,work_seconds=1760,cleanup_reserve_seconds=30,
        reporting_reserve_seconds=10,timed_out=False,returncode=None)
    child=None;identity=None
    def interrupted(signum, frame):
        raise InterruptedError('supervisor signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        verify(expected)
        libc=ctypes.CDLL(None,use_errno=True)
        if libc.prctl(36,1,0,0,0)!=0:raise OSError('cannot establish owned descendant subreaper')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                 VECLIB_MAXIMUM_THREADS='1',BLIS_NUM_THREADS='1',GOTO_NUM_THREADS='1',OMP_THREAD_LIMIT='1',
                 OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE',WANDB_MODE='disabled',WANDB_DISABLED='true')
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--out',str(out),
             '--manifest-sha256',expected,'--deadline',str(time.time()+max(0,work_end-time.monotonic())-2)]
        save(out/'LAUNCH.json',dict(command=cmd,upload_manifest_sha256=expected))
        with (out/'stdout.txt').open('x') as stream:
            child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=ROOT)
            identity=process_identity(child.pid)
            if not identity:raise RuntimeError('worker disappeared before identity capture')
            save(out/'WORKER_IDENTITY.json',identity)
            try:child.wait(timeout=max(0,work_end-time.monotonic()))
            except subprocess.TimeoutExpired:receipt['timed_out']=True
            finally:
                receipt['closure']=cleanup(child,identity,min(whole_end-10,time.monotonic()+30))
                receipt['returncode']=child.returncode
        receipt['stdout_stream_closed']=True
    except BaseException as exc:
        receipt['error']=repr(exc)
        if child is not None and identity is not None and 'closure' not in receipt:
            receipt['closure']=cleanup(child,identity,min(whole_end-10,time.monotonic()+30))
    finally:
        receipt['elapsed_seconds']=time.monotonic()-start
        save(out/'SUPERVISOR.json',receipt)
        try:result=summary(out,receipt)
        except BaseException as exc:
            result=dict(complete=False,error='summary: '+repr(exc),supervision=receipt)
            if not (out/'SUMMARY.json').exists():save(out/'SUMMARY.json',result)
        # Compact records only: never package tensors, models or environment folders.
        record_files=[p for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.txt')]
        save(out/'RESULT_MANIFEST.json',{p.relative_to(out).as_posix():digest(p) for p in record_files})
        with zipfile.ZipFile(out.parent/'Q5_GPU_RESULTS.zip','x',zipfile.ZIP_DEFLATED) as z:
            for p in record_files+[out/'RESULT_MANIFEST.json']:
                z.write(p,p.relative_to(out).as_posix())
        print(json.dumps(result,indent=2),flush=True)
    return 0 if result['complete'] else 4


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','supervise'])
    p.add_argument('--out',required=True,type=Path);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--deadline',type=float);a=p.parse_args()
    if a.mode=='worker':
        try:worker(a.out,a.deadline,a.manifest_sha256)
        except BaseException as exc:
            save(a.out/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            raise
    else:raise SystemExit(supervise(a.out,a.manifest_sha256))
