"""Static handoff generation: no seed draws, array loads, or numerical execution."""
from __future__ import annotations
import json
from pathlib import Path
from .records import read,write,sha
from .graph import operation_graph,cost_manifest

def build(repo,out):
    repo=Path(repo).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    q6=repo/'artifacts/q6_development_20261003T163650Z_private';ret=repo/'artifacts/q6_return_20261004T131221Z_private'
    attempt=ret/'drive/OperatorMark_Q6_20261003_private/attempt01'
    c=dict(repo=str(repo),interpreter=str(repo/'.venv/Scripts/python.exe'),frozen=r'LOCAL_ARCHIVE/D1Q0_20261002T224115Z/frozen',
        implementation_brief_sha256=sha(repo/'artifacts/mainstudy_development_20261004T150444Z/REQUEST.txt'),
        specification_sha256='2055c4eb4af864e14275d2c72e7d476b8614ede7d67ba61ec88762a7c2374e61',
        q6_config=str(q6/'CONFIG_08.json'),q6_bundle=str(q6/'windows/attempt01/colab_bundle'),
        q6_original_inputs=str(q6/'windows/inputs/attempt01'),
        q6_manifest_sha256='0be21b41da5c8d4d637477975ee8048a44df2842eb74573d9e774153c6a0b6ff',
        D1=str(attempt/'owner_81920_evaluation/model.pt'),B10=str(attempt/'baseline_81920_evaluation/model.pt'),
        D1_result=str(attempt/'owner_81920_evaluation/RESULT.json'),
        q6_gpu_environment=str(attempt/'ENVIRONMENT.json'),q6_compatibility=str(attempt/'COMPATIBILITY.json'),
        windows_wrapper=r'LOCAL_ARCHIVE/D1Q0_20261002T224115Z/OperatorMark_Drop1_numerics_v2_correction1/src/tools/run_suite.py',
        scientific=dict(status='development only; final study provisional',n_ref=128,reference='R128to64 o T128',physical_qualification=False,
          D1_updates=81920,D1_plateaued=False,D1_last_improvement_percent=11.4787,architecture='64x64 FNO modes12x12 width32 layers4',
          units='model output u/0.01',D9_loss='mean((f-y)^2)/mean(y^2) + 0.5*mean((Jv-dy)^2)/mean(dy^2); fixed complete training-set normalizers',
          relative_normalization='actual L2 truth norm; no epsilon or median exclusion; zero relative unavailable with absolute error retained',
          utility='finite nonnegative candidate <= 1.25*owner for F/J/K/T; no uplift',clean_screen='F,J <= 2*respective median of D1-D3; not engineering adequacy',
          development_owners=3,wrong_keys=4095,wrong_bound_per_method=210/4096,
          wrong_bound_assumptions=['exchangeable independent keys','key-independent null generation and shams','symmetric scoring'],
          wrong_bound_is_population_FPR=False,wrong_bound_is_joint_5_percent=False,
          embedding_updates=320,embedding_batch=16,embedding_lr=.0003,embedding_penalty=.05,mark_factors=[1.5,2.,3.],
          ordinary_pairs=256,ordinary_updates=1920,ordinary_batch=32,ordinary_lr=.002,ordinary_loss='raw forward MSE in model units',
          replacement_lengths=[10240,20480,40960,81920],replacement_main=73728,replacement_cooldowns=15360,
          replacement_selection='all four complete finite F/J/T geometric scores; shortest within 1% of minimum',
          A5B2='.1*(relu(tau5-D_start)/tau5)^2; tau5 finite positive',A5B1_B3='new cost-only constructions, public-pattern projection penalties, lambda=.1',
          inverse_steps=120,inverse_lr=.03,inverse_targets=16,inverse_batch_loss_divisor=16,inverse_box=[0,1],
          task_robustness='unchanged P256 check; not physical confirmation',
          hard_caps=dict(attached_A100_seconds=31500,Windows_numerical_seconds=7200,wall_seconds=86400),checkpoint_interval=2048,
          later_launch_required=True,final_owners_authorized=False,
          first_use_work_ceilings=dict(Windows_solver_cost=180,Windows_basis_chunk=180,model_evaluation=180,
                                      full_projection=600,publication_if_copy_timing_missing=300,environment_setup_and_cleanup=1380),
          feasibility='existing D1/B10 scoring, Rmain1920 evaluation, A4 cost and wrappers precede D2/D3; remaining projection is incomplete while costs unknown'))
    config=out/'CONFIG.json';write(config,c)
    old=read(q6/'BINDINGS_08.json')['files'];external={}
    # Reuse exactly recorded numerical/runtime identities, excluding closed qualification payloads and seed files.
    for p,h in old.items():
        if Path(p).suffix.lower() in ('.py','.dll','.exe','.pth'):
            if sha(p)!=h:raise ValueError('reused Q6 source/runtime drift: '+p)
            external[p]=h
    for p in [q6/'CONFIG_08.json',Path(c['q6_bundle'])/'MANIFEST.json',q6/'windows/inputs/attempt01/INPUTS.json',
              ret/'RETURN_VERIFICATION.txt',ret/'MAIN_STUDY_IMPLEMENTATION_BRIEF.txt',ret/'SOURCE_EXCERPTS.txt',
              Path(c['D1']),Path(c['B10']),Path(c['D1_result']),Path(c['q6_bundle'])/'requirements-colab.txt',Path(read(q6/'CONFIG_08.json')['q5_runtime']),
              Path(c['q6_gpu_environment']),Path(c['q6_compatibility'])]:
        external[str(p)]=sha(p)
    originals=read(Path(c['q6_original_inputs'])/'INPUTS.json')['files']
    for n,h in originals.items():
        p=Path(c['q6_original_inputs'])/n
        if sha(p)!=h:raise ValueError('original Q6 development control drift')
        external[str(p)]=h
    if external[c['D1']]!='41643fd9baa930494effa05ebba2c2e023fc33caf60e2bd9eb07676ac7712084' or external[c['B10']]!='ecba8b40aafd2c626f61d3b598478c1542383fd73e261db2e15e7db388e329f0':raise ValueError('returned model differs from verification')
    files=list((repo/'tools/msdev').glob('*.py'))+[repo/'tools/mainstudy_development.py',repo/'tools/test_msdev.py',repo/'tools/test_msdev_readiness.py']
    files+=[repo/'experiments/tpfa32_copy_study'/n for n in ('analyze.py','core.py','outcomes.py')]
    files+=[repo/'experiments/darcy_fno32_pilot/run.py',repo/'experiments/darcy_fno32_pilot/spatial.py']
    write(out/'OPERATION_GRAPH.json',operation_graph());write(out/'COST_MANIFEST.json',cost_manifest())
    prepare_script(out,c,repo,config);notebook(out,c)
    for p in [out/'MainStudy_Development.ipynb',*out.glob('*.ps1'),out/'OPERATION_GRAPH.json',out/'COST_MANIFEST.json']:external[str(p)]=sha(p)
    pins=dict(configuration_sha256=sha(config),source_files={p.relative_to(repo).as_posix():sha(p) for p in files},external_files=external,
        reused_return_verification=True,repeated_full_return=False,real_job='UNSTARTED')
    write(out/'PINS.json',pins)
    return dict(configuration_sha256=sha(config),pins_sha256=sha(out/'PINS.json'),sources=len(pins['source_files']),real_job='UNSTARTED')

def prepare_script(out,c,repo,config):
    private=repo/'artifacts/mainstudy_development_run_private'
    base=f"& '{c['interpreter']}' -B '{repo/'tools/mainstudy_development.py'}'"
    common=f"--config '{config}' --pins '{out/'PINS.json'}' --out '{private}'"
    scripts={
      'Prepare-Development.ps1':f'''param([Parameter(Mandatory=$true)][double]$CurrentCredits,[Parameter(Mandatory=$true)][double]$UnitsPerHour)
$ErrorActionPreference='Stop'
# Only after separate real DEVELOPMENT execution approval. Keep this terminal open.
{base} prepare {common} --seconds 7200 --credits $CurrentCredits --billing-rate $UnitsPerHour --execute-development
exit $LASTEXITCODE
''',
      'Status-Development.ps1':f'''$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath '{private/'STATUS.json'}') {{ Get-Content -Raw -LiteralPath '{private/'STATUS.json'}' }} else {{ Write-Output 'UNSTARTED: no real job or clock exists.' }}
''',
      'Resume-Windows.ps1':f'''param([Parameter(Mandatory=$true)][double]$CurrentCredits,[Parameter(Mandatory=$true)][double]$UnitsPerHour,[Parameter(Mandatory=$true)][string]$PriorClosure)
$ErrorActionPreference='Stop'
# PriorClosure is the saved relative windows_attempt_*/invocation.json, NOT a PID.
# Infrastructure interruption only, after positively established closure; never scientific failure/cap.
{base} prepare {common} --seconds 7200 --credits $CurrentCredits --billing-rate $UnitsPerHour --execute-development --resume --incident infrastructure --prior-closure $PriorClosure
exit $LASTEXITCODE
''',
      'Verify-Return.ps1':f'''param([Parameter(Mandatory=$true)][string]$ReturnedDirectory)
$ErrorActionPreference='Stop'
{base} verify-return --out $ReturnedDirectory --config '{config}' --pins '{out/'PINS.json'}' --ready '{private/'READY_FOR_COLAB.json'}'
exit $LASTEXITCODE
''',
      'Return-Development.ps1':f'''param([Parameter(Mandatory=$true)][string]$Archive)
$ErrorActionPreference='Stop'
$returned=Join-Path '{repo/'artifacts'}' ('mainstudy_development_return_'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')+'_private')
{base} return --archive $Archive --out $returned --config '{config}' --pins '{out/'PINS.json'}' --ready '{private/'READY_FOR_COLAB.json'}'
exit $LASTEXITCODE
'''}
    for n,text in scripts.items():
        with (out/n).open('x',encoding='utf-8') as f:f.write(text)

def notebook(out,c):
    cells=[]
    def md(n,text):cells.append(dict(id=f's{n}_text',cell_type='markdown',metadata={},source=f'## {n}. {text}'.splitlines(keepends=True)))
    def code(n,text):cells.append(dict(id=f's{n}_code',cell_type='code',metadata={},execution_count=None,outputs=[],source=(f'# Cell {n}\n'+text).splitlines(keepends=True)))
    md(1,'Development only — UNSTARTED\nDo not run until separately authorized. FIRST use CPU for cells 1–4 to reserve the window on Drive before GPU connection. Switch to A100, repeat 1–4 (reuse that reservation), then run 5–7. No CPU training fallback. D1 is capped, not plateaued; physical negatives stand. Keep Colab connected during cell 6. Windows may close after preparation reports clean closure.')
    code(1,"""import time, os, json, pathlib, hashlib, zipfile, sys, threading, subprocess
print('Cells 1–4 prepare on CPU. The GPU window is persistently charged BEFORE connecting A100.')
sys.dont_write_bytecode=True
try:
    HAS_GPU=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,timeout=10).returncode==0
except FileNotFoundError:
    HAS_GPU=False
if HAS_GPU:
    from google.colab import drive, runtime as colab_runtime
    drive.mount('/content/drive')
    early_status=pathlib.Path('/content/drive/MyDrive/OperatorMark_MainStudy_Development/STATUS.json')
    if not early_status.exists():
        colab_runtime.unassign();raise RuntimeError('No prebooked window; do CPU preparation first')
    early=json.loads(early_status.read_text())
    assert early['session'].get('prebooked') and not early['session']['closed']
    deadline=min(early['session']['end'],early['deadline'])
    assert time.time()<deadline-30
    if 'disconnect_timer' not in globals():
        disconnect_timer=threading.Timer(max(0,deadline-time.time()),colab_runtime.unassign)
        disconnect_timer.daemon=True;disconnect_timer.start()
""")
    md(2,'Upload exactly three files\nFrom `LOCAL_PROJECT/artifacts/mainstudy_development_run_private`: `MainStudy_Development_Upload.zip`, `READY_FOR_COLAB.json`, and `STATUS.json` (the closed Windows status). Upload these again when reconnecting; Drive holds the continuing aggregate status. Never upload PRIVATE_SEEDS.json, the repository, qualification or final material.')
    code(2,"""from google.colab import files
uploaded=files.upload()
assert set(uploaded)=={'MainStudy_Development_Upload.zip','READY_FOR_COLAB.json','STATUS.json'}
ready=json.loads(uploaded['READY_FOR_COLAB.json'])
assert hashlib.sha256(uploaded['MainStudy_Development_Upload.zip']).hexdigest()==ready['archive_sha256']
bundle=pathlib.Path('/content/msdev_upload');already_extracted=bundle.exists();bundle.mkdir(exist_ok=True)
with zipfile.ZipFile('MainStudy_Development_Upload.zip') as z:
    infos=z.infolist();names=[r.filename for r in infos]
    assert len(names)==len(set(names))
    for r in infos:
        p=pathlib.PurePosixPath(r.filename)
        assert not p.is_absolute() and '..' not in p.parts and '\\\\' not in r.filename and ':' not in r.filename
        assert (r.external_attr>>16)&0o170000 != 0o120000
    if not already_extracted:z.extractall(bundle)
assert hashlib.sha256((bundle/'MANIFEST.json').read_bytes()).hexdigest()==ready['manifest_sha256']
m=json.loads((bundle/'MANIFEST.json').read_text())
for name,digest in m['files'].items():
    assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==digest, name
print('Development identities verified; no authenticity/anti-tampering attestation claimed.')
""")
    md(3,'Import the verified launcher only\nThis imports standard-library preparation code, not models or solvers. Cell 5 explicitly recreates the isolated Q6 package environment from the bound requirements, then checks exact runtime/parity applicability. No dependency substitutions or installations into the notebook kernel.')
    code(3,"""sys.path.insert(0,str(bundle/'code'))
from msdev.cli import main
""")
    md(4,'Reserve on CPU before attaching A100; reuse after switching\nNever rename this directory to obtain a new budget. First session at most 21,600s; total at most 31,500s. Its entire window is charged before GPU attachment, including setup/idle/transfer. The Windows-started 24h clock never resets. After CPU reservation switch to A100 and repeat 1–4. Failed setup remains charged.')
    code(4,"""from google.colab import drive
drive.mount('/content/drive')
root=pathlib.Path('/content/drive/MyDrive/OperatorMark_MainStudy_Development')
root.mkdir(exist_ok=True)
saved=json.loads((root/'STATUS.json').read_text()) if (root/'STATUS.json').exists() else None
pending=saved and saved.get('session',{}).get('prebooked') and not saved['session']['closed']
if pending:
    assert not saved['session'].get('launch_started'), 'Started worker must close; no duplicate launch'
    print('Reusing charged window',saved['session'])
else:
    assert not HAS_GPU, 'Reserve BEFORE attaching GPU; use CPU for preparation'
    credits=float(input('Current available compute units (no purchase): '))
    rate=float(input('Current A100 units/hour shown by Colab: '))
    seconds=float(input('Session seconds, <=21600 and within remaining aggregate allowance: '))
    assert 30<seconds<=21600
    reserve_args=['reserve-colab','--bundle',str(bundle),'--out',str(root),'--windows-status','/content/STATUS.json',
          '--manifest-sha256',ready['manifest_sha256'],'--seconds',str(seconds),'--credits',str(credits),
          '--billing-rate',str(rate),'--execute-development']
    if saved and saved['state'] not in ('WINDOWS_COMPLETE','PAUSED_SESSION'):
        assert input('Only a diagnosed infrastructure interruption with saved owned closure? Type infrastructure: ')=='infrastructure'
        prior=input('Saved relative colab_attempt_*/SUPERVISOR.json path: ')
        reserve_args+=['--resume','--incident','infrastructure','--prior-closure',prior]
    main(reserve_args)
    print('Switch to A100, repeat 1–4, then 5–7. This window is already running.')
""")
    md(5,'Recreate the pinned Q6 environment and check applicability\nUse the original bound requirements, an isolated venv without pip, and system pip --python exactly as Q6. No Windows dependency changes or version substitutions. Setup has a 1380s ceiling including owned cleanup, charged inside the existing attached window. A base-Python, dependency, CUDA or parity mismatch stops; no automatic Q6 replay. Do not repeat failed setup in this session.')
    code(5,"""RESUME=False
INCIDENT=None
PRIOR_CLOSURE=None
s=json.loads((root/'STATUS.json').read_text());session=s['session']
assert session.get('prebooked') and not session['closed'] and not session.get('launch_started')
assert time.time()<session['end']-30
from google.colab import runtime as colab_runtime
if 'disconnect_timer' not in globals():
    disconnect_timer=threading.Timer(max(0,session['end']-time.time()),colab_runtime.unassign)
    disconnect_timer.daemon=True;disconnect_timer.start()
from msdev.colab_environment import setup
ENV=pathlib.Path('/content/msdev_q6_venv')
try:
    PYTHON=setup(bundle,root,ENV)
except BaseException as exc:
    from msdev.records import write
    write(root/('SETUP_FAILURE_'+str(time.time_ns())+'.json'),{'error':repr(exc),'kernel_python':sys.version})
    colab_runtime.unassign()
    raise
print('Verified isolated interpreter:',PYTHON)
RESUME=session['previous_state']=='PAUSED_SESSION'
""")
    md(6,'Run the complete finite development graph\nThis is the real job. It is NOT run by the implementation milestone. The qualified Q5 cleanup helpers reserve 30 seconds for worker cleanup, followed by 30 seconds for supervisor/stream closure and manifest publication, all inside the unchanged attached deadline. Every session’s full reservation remains charged, covering setup/idle/transfer. First-use kernels are timed within prescribed updates, not extra training. Missing rates are not priced as D9.')
    code(6,"""from msdev.colab_environment import launch
# Both the supervisor and its worker inherit this exact recreated interpreter.
try:
    receipt=launch(PYTHON,bundle,root,ready['manifest_sha256'])
except BaseException:
    # Missing owned closure remains missing; never call VM disconnect a receipt.
    colab_runtime.unassign()
    raise
print('Supervisor result',receipt)
print((root/'STATUS.json').read_text())
""")
    md(7,'Download records and disconnect\nDownload this small status file. All models, designs, failures and producer manifests remain in the private Drive directory. Download that complete directory from Drive to Windows after disconnecting; use Verify-Return.ps1 on its extracted directory. Do not delete Drive copies before verified return. A paused run is incomplete, not a successful campaign.')
    code(7,"""files.download(str(root/'STATUS.json'))
if (root/'RESULT.json').exists():files.download(str(root/'RESULT.json'))
from google.colab import runtime as colab_runtime
colab_runtime.unassign()  # disconnect GPU; no further training
""")
    write(out/'MainStudy_Development.ipynb',dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();print(build(a.repo,a.out))
