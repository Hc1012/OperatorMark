"""Single-cell bounded null/B2 launch using the retained owned supervisor.

Prompts/authentication/capacity checks precede this attempt's one clock. Connected
Colab time can still be billed before activation. No retry or clock reset.
"""
from pathlib import Path
import ctypes, hashlib, importlib, importlib.util, json, math, os, shutil, sys, threading, time, traceback
from null_io import need,read,put,sha,_closed
from noise_colab import execute_owned,pre_activation_identity,rate_reservation,nvidia_identity,PreActivationRefusal,stage_window

POLICY='NULL_B2_POLICY.json'
SCOPE='fixed_null210_B2_1240_only'
STAGES=(('host_auth',40),('claim',180),('setup_0',60),('setup_1',1200),('setup_2',60),
        ('worker_auth',40),('runtime',900),('input',3600),('producer',21600))
CLOSURE=30
DEADLINE_STAGES={'claim','input','runtime','producer'}


def stage_commands(job,host,worker,policy_sha):
    job=Path(job);payload=job.parent/'payload';auth=str(payload/'handoff/metadata_auth_observe.py')
    return dict(capacity=[host,'-B',str(job/'null_remote.py'),'capacity','--job',str(job)],
        host_auth=[host,'-B',auth],claim=[host,'-B',str(job/'null_remote.py'),'claim','--job',str(job)],
        input=[host,'-B',str(job/'null_remote.py'),'input','--job',str(job)],
        worker_auth=[str(worker),'-B',auth,'--worker'],
        runtime=[str(worker),'-B',str(job/'null_runtime.py'),'--job',str(job),'--policy-sha',policy_sha],
        producer=[str(worker),'-B',str(job/'null_producer.py'),'--job',str(job),'--policy-sha',policy_sha])


def check_policy(job,policy_sha):
    need(sha(job/POLICY)==policy_sha,'Frozen null policy differs')
    p=read(job/POLICY)
    need(p['scope']==SCOPE and p['owners']==[f'O{i:02d}' for i in range(1,11)],'Null/B2 scope differs')
    need(p['stage_seconds']==30600 and p['numerical_work_seconds']==21600 and p['return_reserve_seconds']==1800
         and p['maximum_units_per_hour']==6.0 and p['maximum_units']==51.0,'Fixed launch envelope differs')
    need(p['execution_stage_budgets']==[dict(name=n,work_seconds=s) for n,s in STAGES]
         and p['reserved_execution_and_return_seconds']==sum(s+CLOSURE for _,s in STAGES)+1800
         and p['reserved_execution_and_return_seconds']<p['stage_seconds'],'Stage budgets differ')
    need(sha(job/'NULL_B2_ROSTER.json')==p['roster_sha256'],'Frozen roster differs')
    for name,expected in p['files'].items():
        target=job/name
        need(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(job.resolve())
             and target.stat().st_size==expected['bytes'] and sha(target)==expected['sha256'],
             'Prepared file differs: '+name)
    payload=job.parent/'payload'
    for name,expected in p['payload_files'].items():
        target=payload/name
        need(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(payload.resolve())
             and target.stat().st_size==expected['bytes'] and sha(target)==expected['sha256'],'Prepared setup file differs: '+name)
    need(len(p['source_pins'])==338,'All original scientific source pins required')
    for name,h in p['source_pins'].items():need(sha(payload/'code'/name)==h,'Scientific source drift: '+name)
    return p


def prepare_dependencies(job,p):
    """Real import/entrypoint checks, without starting numerical/network work."""
    source=job.parent/'payload/code';sys.path[:0]=[str(job),str(source/'tools'),str(source)]
    observed={}
    def module(name,path,entries):
        spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
        sys.modules[name]=value;spec.loader.exec_module(value)
        need(Path(value.__file__).resolve()==path.resolve() and all(callable(getattr(value,n,None)) for n in entries),
             'Prepared entrypoint unavailable: '+name)
        observed[name]=dict(path=str(path),sha256=sha(path),entrypoints=list(entries),import_completed=True)
        return value
    supervisor=module('null_owned_supervisor',source/'q5_colab_supervisor.py',('process_identity','cleanup'))
    from msdev.colab_environment import setup_commands,owned
    from msdev.bounded_capture import capture
    from q6_common import THREAD_ENV
    for name,entries in [('null_remote',('transfer',)),('null_runtime',('observe','verify_gate')),
        ('null_producer',('run',)),('null_compute',('collect_jvp','distance_rows','keyed_readout')),
        ('null_reference',('prepare','collect_native')),
        ('continuation_ops',('regular_worker_interpreter',))]:
        module(name,job/(name+'.py'),entries)
    auth=job.parent/'payload/handoff/metadata_auth_observe.py';compile(auth.read_bytes(),str(auth),'exec')
    plan=setup_commands(sys.executable,job.parent/'environment',job.parent/'payload')
    need([s for _,s in plan]==[60,1200,60],'Retained environment setup plan differs')
    commands=stage_commands(job,sys.executable,job.parent/'environment/bin/python','0'*64)
    for command in commands.values():
        for operand in command:
            if operand.endswith('.py'):compile(Path(operand).read_bytes(),operand,'exec')
    return dict(supervisor=supervisor,owned=owned,setup_commands=setup_commands,thread_env=THREAD_ENV,
        interpreter_ops=sys.modules['continuation_ops'],receipt=dict(complete=True,observed=observed,
            before_activation=True,subprocesses_run=0,model_calls=0,network_work=0))


def run(job,policy_sha,archive_expectation):
    job=Path(job).resolve();p=check_policy(job,policy_sha);launch=prepare_dependencies(job,p)
    from google.colab import auth,runtime
    need(callable(getattr(runtime,'unassign',None)),'Automatic disconnect unavailable')
    need(not (job/'ACTIVATION.json').exists(),'Attempt already exists; no rerun')
    payload=job.parent/'payload';environment=job.parent/'environment';worker=environment/'bin/python'
    env=dict(os.environ,PYTHONPATH=str(job)+os.pathsep+str(payload/'code/tools')+os.pathsep+str(payload/'code'),
             PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1');env.update(launch['thread_env'])
    need(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0,'Owned subprocess subreaper unavailable')
    commands=stage_commands(job,sys.executable,worker,policy_sha)
    gpu,driver=nvidia_identity();host_sha=sha(Path(sys.executable).resolve())
    observed=pre_activation_identity(p,python=sys.version,executable_sha256=host_sha,gpu=gpu,driver=driver,
        linux_free_bytes=shutil.disk_usage('/content').free,
        available_ram_bytes=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_AVPHYS_PAGES'))
    print('Prepared A100 identity matched. This runs the complete fixed null/B2 comparison block.',flush=True)
    print('Saved models come from Drive. Only the missing F64 reference derivatives are added; no training or inverse optimization.',flush=True)
    print('Planning estimate 2–4 hours; this exact batch is not yet timed. Hard cutoff: 8.5 hours.',flush=True)
    print('Colab can bill connected time before activation. Enter the TWO real numbers; never guess a zero rate.',flush=True)
    try:
        balance=float(input('Current COMPUTE UNIT BALANCE: ').strip())
        rate=float(input('Current COMPUTE UNITS PER HOUR for this connected A100: ').strip())
    except ValueError:raise PreActivationRefusal('Both prompts require plain numbers') from None
    reservation=rate_reservation(p,balance,rate)
    print('Authenticate the Google account that owns the existing private results folder.',flush=True)
    auth.authenticate_user()
    capacity_start=time.time();capacity_end=capacity_start+150
    execute_owned(job,launch,env,'preactivation_capacity',commands['capacity']+['--deadline',str(capacity_end-60)],120,capacity_end)
    capacity=read(job/'STORAGE_PREFLIGHT.json')
    need(capacity['passed'] is True and capacity['before_activation'] is True and capacity['scope']==SCOPE
         and capacity['policy_sha256']==policy_sha and capacity_start<=capacity['observed_at']<=time.time()
         and capacity['private_owner_only'] is True,'Fresh capacity/privacy observation required')
    started=time.time();deadline=started+p['stage_seconds'];attempt=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(started))
    activation=dict(attempt=attempt,start=started,deadline=deadline,scope=SCOPE,policy_sha256=policy_sha,
        stage_seconds=p['stage_seconds'],maximum_rate=6.,maximum_units=51.,previous_clocks_reused=False,retries=0,
        started_by='user executing the generated notebook cell',prompts_and_authentication_before_activation=True,
        scientific_acceptance=False,formal_custody_assurance=False,connected_time_before_activation_can_be_billed=True)
    put(job/'ACTIVATION.json',activation);put(job/'INPUT_ARCHIVE_EXPECTATION.json',archive_expectation)
    put(job/'LAUNCH_DEPENDENCY_PREFLIGHT.json',launch['receipt'])
    put(job/'HOST_INTERPRETER_OBSERVED.json',dict(observed_at=time.time(),python=sys.version,executable_sha256=host_sha))
    put(job/'LIVE_PREFLIGHT.json',dict(observed,observed_at=time.time(),compute_unit_balance=balance,
        compute_units_per_hour=rate,maximum_stage_units_at_observed_rate=reservation,whole_campaign_allocation=False))
    print('ACTIVATED '+attempt+'. Never rerun after this line. All ten owners proceed automatically.',flush=True)
    done=threading.Event()
    def cutoff():
        if not done.wait(max(0.,deadline-time.time())):
            print('Fixed cutoff reached; requesting disconnect.',flush=True)
            try:runtime.unassign()
            except BaseException:print('Select Runtime > Disconnect and delete runtime.',flush=True)
    threading.Thread(target=cutoff,daemon=True).start()
    cursor=0;phase='post-activation';saved=False;download_window=False
    def stage(name,command=None):
        nonlocal cursor
        seconds,end,child_deadline=stage_window(STAGES,cursor,name,time.time(),deadline,p['return_reserve_seconds'])
        command=list(commands[name] if command is None else command)
        if name in DEADLINE_STAGES:command+=['--deadline',str(child_deadline)]
        put(job/'stage_budgets'/(name+'.json'),dict(name=name,work_seconds=seconds,owned_deadline=end,
            child_deadline=child_deadline if name in DEADLINE_STAGES else None,original_deadline=deadline,clock_reset=False))
        value=execute_owned(job,launch,env,name,command,seconds,end);cursor+=1;return value
    try:
        need(time.time()+p['reserved_execution_and_return_seconds']<deadline,'Fixed startup reserve unavailable')
        phase='host authentication';stage('host_auth')
        phase='once-only attempt claim';stage('claim')
        phase='retained pinned environment setup';need(not environment.exists(),'Environment already exists')
        for i,(command,seconds) in enumerate(launch['setup_commands'](sys.executable,environment,payload)):
            need(seconds==STAGES[cursor][1],'Setup budget differs');stage('setup_'+str(i),command)
        need(sha(Path(sys.executable).resolve())==host_sha,'Host interpreter changed during setup')
        witness=launch['interpreter_ops'].regular_worker_interpreter(worker,host_sha,
            lambda:need(time.time()+sum(s+CLOSURE for _,s in STAGES[cursor:])+p['return_reserve_seconds']<deadline,
                        'Interpreter preparation clock expired'))
        put(job/'INTERPRETER_PATH_PREPARATION.json',dict(witness,numerical_compatibility_verified=False))
        phase='worker authentication';stage('worker_auth')
        phase='fresh null/B2 runtime gate';stage('runtime')
        gate=read(job/'NULL_B2_RUNTIME_GATE.json')
        need(gate.get('passed') is True and gate['scope']==SCOPE and gate['policy_sha256']==policy_sha
             and gate['activation_sha256']==sha(job/'ACTIVATION.json'),'Fresh scoped gate required')
        phase='existing inputs from Drive';stage('input')
        phase='fixed null/B2 producer';stage('producer')
        need(cursor==len(STAGES),'Incomplete stage order')
        print('All fixed null/B2 calculations completed. Saving review and closure evidence.',flush=True)
    except BaseException as exc:
        put(job/'FIRST_FAILURE.json',dict(phase=phase,error=repr(exc),traceback=traceback.format_exc(),
            original_deadline=deadline,retry=False,scientific_acceptance=False))
        print('STOPPED AT '+phase+': '+str(exc),flush=True)
    finally:
        try:
            save_end=min(deadline-180,time.time()+1440)
            need(save_end-time.time()>60,'Final save reserve unavailable')
            execute_owned(job,launch,env,'save',[sys.executable,'-B',str(job/'null_remote.py'),'save','--job',str(job),
                '--deadline',str(save_end-30)],save_end-time.time()-30,save_end)
            receipt=read(job/'SAVE_RESULT.json');saved=True
            print('RESULT: '+('all 210 null entries and 1,240 B2 groups complete.' if receipt['complete']
                  else 'stopped before full completion; available evidence was saved.'),flush=True)
            print('DOWNLOAD THE REVIEW: '+receipt['review_url'],flush=True)
            print('Large raw returns remain in Drive. CPU scoring is included in this notebook.',flush=True)
            close_end=min(deadline-10,time.time()+120)
            need(close_end-time.time()>60,'Final metadata window unavailable')
            execute_owned(job,launch,env,'closing_record',[sys.executable,'-B',str(job/'null_remote.py'),'closure',
                '--job',str(job),'--deadline',str(close_end-30)],close_end-time.time()-30,close_end)
            print('DOWNLOAD THE SAVE RECEIPT: '+read(job/'CLOSING_RECORD_SAVED.json')['closure_url'],flush=True)
        except BaseException as exc:
            print('Final publication issue: '+repr(exc)+'. Already read-back owner capsules remain saved.',flush=True)
        if not saved:
            try:
                import zipfile
                report=Path('/content')/('NULL_B2_DIAGNOSTIC_'+attempt+'.zip')
                entries={p.relative_to(job).as_posix():p for p in job.glob('*.json') if p.stat().st_size<8*1024**2}
                for folder in ('supervision','owner_receipts'):
                    for p in (job/folder).rglob('*'):
                        if p.is_file() and p.stat().st_size<8*1024**2:entries[p.relative_to(job).as_posix()]=p
                need(sum(p.stat().st_size for p in entries.values())<64*1024**2,'Diagnostic fallback bound exceeded')
                with zipfile.ZipFile(report,'x',zipfile.ZIP_DEFLATED) as z:
                    for name,p in entries.items():z.write(p,name)
                from google.colab import files
                files.download(str(report));print('Save this diagnostic and the notebook output.',flush=True)
                timer=threading.Timer(max(0.,min(300.,deadline-time.time())),runtime.unassign);timer.daemon=True;timer.start()
                download_window=True
            except BaseException as exc:print('Fallback download unavailable: '+type(exc).__name__,flush=True)
        if not download_window:
            try:runtime.unassign();done.set()
            except BaseException:print('Select Runtime > Disconnect and delete runtime now.',flush=True)
