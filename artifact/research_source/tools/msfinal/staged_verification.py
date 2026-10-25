"""One coordinated, externally authorized development workflow. No reserve().

Uses Q0 Windows Job Object wrapper and Q5 POSIX owned/subreaper cleanup. Historical
standalone screen remains untouched;36 updates are one internal stage only.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata,json,math,os,sys,time
from msdev.records import read,write,sha,verify,progress
from . import verification_plan as P,development_check as K,readback as R
from .session import closure_ok

LIMITS=dict(overall=21600,cpu=3600,setup=1380,kernel=900,matrix=3600,contracts=3600,
    cleanup=30,finalization=120,transfer_idle=900,memory_bytes=8000000000,evidence_bytes=4*1024**3,evidence_files=20000)

def require_window(w,manifest,now):
    if (w.get('approved') is not True or w.get('scope')!='Block1_selected_development_verification'
        or w.get('manifest_sha256')!=manifest or w.get('limits')!=LIMITS or w.get('plan')!=P.plan()
        or w.get('retries')!=0 or w.get('resume') is not False):raise ValueError('separate complete-scope numerical authority required')
    if any(type(w.get(k)) not in (float,int) or not math.isfinite(w[k]) for k in ('start','deadline','credits','rate')):raise ValueError('immutable time and approved current credits/rate required')
    if w['deadline']-w['start']!=LIMITS['overall'] or not w['start']<=now<w['deadline']:raise TimeoutError('original whole-workflow deadline')
    if w['rate']<=0 or w['credits']<6*w['rate']:raise ValueError('approved current rate/credits unavailable')
    return w

def preflight(bundle,manifest):
    bundle=Path(bundle)
    if sha(bundle/'MANIFEST.json')!=manifest:raise ValueError('staged bundle drift')
    verify(bundle,read(bundle/'MANIFEST.json')['files']);r=read(bundle/'STAGED.json')
    if r.get('plan')!=P.plan() or r.get('limits')!=LIMITS or r.get('development_only') is not True:raise ValueError('complete prospective stage plan binding')
    if r.get('unresolved')!=[]:raise ValueError('unresolved prerequisites: '+repr(r.get('unresolved')))
    # Full contract metadata/roster checked before any native or paid work.
    from .contract_verification import input_plan
    input_plan(bundle/r['contract_root'],r['contract_plan_sha256'])
    for name in ('seed_record','contract_seeds','exclusions','normalizers','cpu_environment'):
        value=r[name]
        if sha(bundle/value['path'])!=value['sha256']:raise ValueError('stage input drift: '+name)
    return r

def check_factory(root,deadline,*,byte_limit=None,storage_interval=60):
    root=Path(root)
    last=[float('-inf')]
    last_memory=[float('-inf')]
    def storage():
        if time.time()>=deadline:raise TimeoutError('original stage deadline')
        files=[p for p in root.rglob('*') if p.is_file()]
        if len(files)>LIMITS['evidence_files'] or sum(p.stat().st_size for p in files)>(LIMITS['evidence_bytes'] if byte_limit is None else byte_limit):raise ValueError('evidence ceiling; retain partials')
        last[0]=time.monotonic()
    def check():
        if time.time()>=deadline:raise TimeoutError('original stage deadline')
        if time.monotonic()-last[0]>=storage_interval:storage()
        if sys.platform=='linux':
            import resource
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>LIMITS['memory_bytes']:raise MemoryError('host memory ceiling')
        elif sys.platform=='win32' and time.monotonic()-last_memory[0]>=5:
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('peak','working','pool_peak','pool','nonpaged_peak','nonpaged','pagefile','pagefile_peak')]
            value=Counters();value.cb=ctypes.sizeof(value)
            get=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
            get.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];get.restype=wintypes.BOOL
            if not get(wintypes.HANDLE(-1),ctypes.byref(value),value.cb):raise ctypes.WinError(ctypes.get_last_error())
            if value.peak>LIMITS['memory_bytes']:raise MemoryError('Windows peak working-set ceiling')
            last_memory[0]=time.monotonic()
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available() and sys.modules['torch'].cuda.max_memory_allocated()>LIMITS['memory_bytes']:raise MemoryError('GPU memory ceiling')
    check.storage=storage
    return check

def stage_envelope(w,name,now):
    work=LIMITS[name]
    if w['deadline']-now<work+LIMITS['cleanup']+LIMITS['finalization']:raise TimeoutError('full '+name+' envelope cannot fit')
    return now+work,now+work+LIMITS['cleanup']

def worker(bundle,root,manifest,window_path,stage,stage_binding):
    bundle=Path(bundle);root=Path(root);w=read(window_path);require_window(w,manifest,time.time());cfg=preflight(bundle,manifest)
    sb=read(stage_binding)
    if sb['window_sha256']!=sha(window_path) or sb['stage']!=stage or sb['work_deadline']>w['deadline']-150:raise ValueError('bound worker stage/window')
    check=check_factory(root,sb['work_deadline']);check()
    out=root/stage/'work';out.parent.mkdir(exist_ok=True)
    try:
        if stage=='cpu':
            actual=dict(python=sys.version,executable=sys.executable,executable_sha256=sha(sys.executable),
                packages={k:importlib.metadata.version(k) for k in ('numpy','scipy','torch','neuraloperator')})
            write(root/'cpu/RUNTIME.json',actual)
            expected=read(bundle/cfg['cpu_environment']['path'])
            if actual!=expected:raise ValueError('bound Windows Q0 runtime differs')
            from .development_inputs import assemble
            from .contract_inputs import generate,direction_split
            direction_split()  # Refuse a frozen split incompatibility before any input generation.
            from .p256_verification import run as derivatives
            seed=cfg['seed_record'];exclude=cfg['exclusions'];normal=cfg['normalizers'];base=bundle/cfg['base']
            assemble(base,cfg['base_manifest_sha256'],bundle/seed['path'],seed['sha256'],bundle/exclude['path'],exclude['sha256'],
                root/'catalog',check,runtime_sha256=sha(base/'Q6_ENVIRONMENT.json'),source_pins=cfg['source_pins'],
                normalizers=read(bundle/normal['path'])['normalizers']['0'],normalizer_source=bundle/normal['path'],normalizer_sha256=normal['sha256'])
            import numpy as np
            from operatormark.corpus.canonical import tensor_sha256
            raw=root/'catalog/raw'
            added_controls=[tensor_sha256(v) for name in ('new_controls','new_targets') for v in np.load(raw/(name+'.npy'),allow_pickle=False)]
            added_directions=[tensor_sha256(v) for v in np.load(raw/'new_directions.npy',allow_pickle=False).reshape(-1,8,8)]
            s=cfg['contract_seeds'];generate(bundle/s['path'],s['sha256'],bundle/exclude['path'],exclude['sha256'],root/'contract_inputs',check,
                additional_controls=added_controls,additional_directions=added_directions)
            from .contract_verification import bind_generated
            bind_generated(bundle/cfg['contract_root'],cfg['contract_plan_sha256'],root)
            derivatives(root/'p256_derivative',check)
            from .p256_tasks import run as task
            import numpy as np
            task(np.load(root/'catalog/final_starts.npy',allow_pickle=False)[:2],np.load(root/'catalog/final_targets.npy',allow_pickle=False)[:2],
                np.load(root/'catalog/final_targets_p256.npy',allow_pickle=False)[:2],root/'p256_task',check,target_family='T128',scope='development_probe')
            out.mkdir(exist_ok=False);write(out/'RESULT.json',dict(complete=True,scope='development CPU inputs/derivative/task',plan=P.plan()))
        else:
            def runtime():
                path=root/stage/'RUNTIME.json'
                if not path.exists():K.runtime(bundle,path)
                return sha(bundle/'Q6_ENVIRONMENT.json')
            runtime()
            if stage=='kernel':
                code=K.run(bundle,out,manifest,w,stage_authority=w)
                if code:raise ValueError('internal kernel screen failed')
            elif stage=='matrix':
                from .development_matrix import run
                run(root/'catalog',sha(root/'catalog/DEVELOPMENT_CATALOG.json'),root/'kernel',sha(root/'kernel/OUTPUT_MANIFEST.json'),
                    root/'kernel/supervisor/RESULT.json',sha(root/'kernel/supervisor/RESULT.json'),out,check,lambda name:check(),runtime_check=runtime)
            elif stage=='contracts':
                from .contract_verification import run
                run(root,sha(root/'CONTRACT_VERIFICATION.json'),out,check,runtime_check=runtime)
            else:raise ValueError('unknown stage')
        check.storage();check();return 0
    except BaseException as exc:
        write(root/stage/'WORKER_FAILURE.json',dict(stage=stage,error=repr(exc),window_sha256=sha(window_path)));return 4

def close_stage(root,name,*,deadline=None):
    root=Path(root);r=read(root/name/'supervisor/RESULT.json')
    if not closure_ok(r) or r.get('returncode')!=0 or r.get('timed_out'):raise ValueError('owned stage did not close successfully: '+name)
    value=read(root/name/'work/RESULT.json')
    if not value.get('complete',value.get('complete_available_screen',False)):raise ValueError('stage result incomplete')
    if name=='kernel':
        from .update_journal import read_prefix
        journal=read_prefix(root/name/'work/updates.jsonl')
        if not journal['complete'] or journal['completed_updates']!=36 or value['counts']!=K.COUNTS:raise ValueError('kernel independent count readback')
    if deadline is None:raise ValueError('explicit independent readback deadline required')
    files=R.inventory(root/name,check_factory(root,deadline),file_limit=LIMITS['evidence_files'],byte_limit=LIMITS['evidence_bytes'])
    write(root/name/'OUTPUT_MANIFEST.json',dict(complete=True,files={n:v['sha256'] for n,v in files.items()}))

def launch(bundle,root,manifest,window_path,mode):
    """No clock creation. A later manually approved WINDOW must already exist."""
    bundle=Path(bundle);root=Path(root);cfg=preflight(bundle,manifest);w=read(window_path);require_window(w,manifest,time.time())
    if root/'WINDOW.json'!=Path(window_path) or sha(root/'WINDOW.json')!=sha(window_path):raise ValueError('one persistent window path required')
    from block1_development_verify import module,base_observation,worker_env
    from msdev.colab_environment import owned,setup_commands
    if mode=='cpu':
        if os.name!='nt':raise ValueError('pinned Windows CPU generation route required')
        from run_corpus_v2_suite import _drop1_runner
        runner=_drop1_runner(Path(cfg['drop1_src']));python=sys.executable;stages=['cpu']
    else:
        if os.name!='posix':raise ValueError('historical POSIX A100 route required')
        close=read(root/'cpu/CLOSURE.json')
        if not close.get('closed') or close['window_sha256']!=sha(window_path):raise ValueError('verified same-window Windows predecessor required')
        verify(root,close['files'])
        import ctypes
        if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError('subreaper unavailable')
        supervisor=module(bundle/'code/q5_colab_supervisor.py');setup=root/'setup';setup.mkdir(exist_ok=False)
        end,cleanup_end=stage_envelope(w,'setup',time.time());environment=Path('/content/operatormark_block1_selected_environment')
        if environment.exists():raise FileExistsError('prior environment preserved; no retry')
        base_observation(bundle,setup/'base',supervisor,end)
        for i,(command,seconds) in enumerate(setup_commands('/usr/bin/python3',environment,bundle)):
            owned(command,seconds,cleanup_end,setup/str(i),supervisor)
        python=str(environment/'bin/python')
        owned([python,'-B',str(bundle/'code/tools/block1_remesh_precheck.py'),str(bundle),str(setup/'remesh_work')],
              60,cleanup_end,setup/'remesh_supervisor',supervisor,env=worker_env(bundle))
        stages=['kernel','matrix','contracts']
    for name in stages:
        start=time.time();work_end,end=stage_envelope(w,name,start);stage=root/name;stage.mkdir(exist_ok=False)
        progress(stage,'stage_start',stage=name,work_deadline=work_end,cleanup_deadline=end)
        binding=stage/'STAGE.json';write(binding,dict(stage=name,window_sha256=sha(window_path),start=start,work_deadline=work_end,cleanup_deadline=end))
        command=[python,'-B',str(bundle/'code/tools/block1_staged_verification.py'),'worker','--bundle',str(bundle),'--out',str(root),
            '--manifest',manifest,'--window',str(window_path),'--stage',name,'--stage-binding',str(binding)]
        if mode=='cpu':
            (stage/'supervisor').mkdir(exist_ok=False)
            env=worker_env(bundle);receipt=runner.run_with_timeout(command,cwd=bundle,env=env,timeout_s=LIMITS[name],cleanup_timeout_s=30,out_dir=stage/'supervisor',extra=dict(stage=name))
            # Q0 schema checked explicitly; unknown Windows tree closure fails.
            cleanup=receipt.get('cleanup',{})
            if (receipt.get('exit_code')!=0 or receipt.get('timed_out') or cleanup.get('resolved') is not True
                or cleanup.get('child_reaped') is not True or cleanup.get('group_survivors') is not False
                or cleanup.get('job_accounting',{}).get('active_processes')!=0 or receipt.get('job_handle_closed') is not True):raise ValueError('Q0 Windows owned closure unavailable')
            value=read(stage/'work/RESULT.json')
            if not value.get('complete'):raise ValueError('CPU numerical stage incomplete')
            cpu_finalize_deadline=min(w['deadline'],time.time()+LIMITS['finalization'])
            files=R.inventory(root,check_factory(root,cpu_finalize_deadline),file_limit=LIMITS['evidence_files'],byte_limit=LIMITS['evidence_bytes'])
            write(stage/'CLOSURE.json',dict(closed=True,closed_epoch=time.time(),window_sha256=sha(window_path),receipt=receipt,files={n:v['sha256'] for n,v in files.items()}))
        else:
            owned(command,LIMITS[name],end,stage/'supervisor',supervisor,env=worker_env(bundle),acceptable=(0,4))
            close_stage(root,name,deadline=min(w['deadline'],time.time()+120))
    if mode=='gpu':finalize(root,w)
    else:
        from .staged_transfer import export_cpu
        export_cpu(root,cpu_finalize_deadline)

def finalize(root,w):
    """Post-closure full-byte inventory/export, bounded by original deadline."""
    import zipfile
    root=Path(root);check=check_factory(root,min(w['deadline'],time.time()+LIMITS['finalization']))
    for name in ('kernel','matrix','contracts'):
        r=read(root/name/'supervisor/RESULT.json')
        if not closure_ok(r) or r['returncode']!=0 or r['timed_out']:raise ValueError('incomplete stage closure')
        verify(root/name,read(root/name/'OUTPUT_MANIFEST.json')['files'])
    write(root/'OUTCOME.json',dict(complete_prepared_verification=True,complete_block1=False,closed=True,export_pending=True,plan=P.plan(),original_deadline=w['deadline']))
    files=R.inventory(root,check,file_limit=LIMITS['evidence_files'],byte_limit=LIMITS['evidence_bytes'])
    write(root/'OUTPUT_MANIFEST.json',dict(files={n:v['sha256'] for n,v in files.items()},complete=True))
    names=list(files)+['OUTPUT_MANIFEST.json'];archive=root.with_name(root.name+'_RETURN.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for name in names:check();z.write(root/name,name)
    with zipfile.ZipFile(archive) as z:
        for name in names:
            check()
            with z.open(name) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=sha(root/name,check):raise ValueError('export readback drift')
    write(root/'FINALIZATION.json',dict(complete=True,archive_sha256=sha(archive,check),archive_bytes=archive.stat().st_size,
        archive_readback=True,manifest_sha256=sha(root/'OUTPUT_MANIFEST.json'),original_deadline=w['deadline']))
    return archive
