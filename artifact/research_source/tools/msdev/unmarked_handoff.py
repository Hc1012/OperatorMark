"""Static, allowlisted packaging for the explicitly unstarted unmarked successor."""
from pathlib import Path
import json
import shutil
import zipfile
from .records import read,write,sha,verify,manifest
from .unmarked import KIND,DEADLINE,validate_parent,continuation_graph,projection

OUTPUT='OperatorMark_MainStudy_Unmarked_Continuation_20261005'
ARCHIVE='MainStudy_Unmarked_Upload.zip'
NOTEBOOK='MainStudy_Unmarked_Continuation.ipynb'

def build(repo,returned,out,diagnostic_review=None):
    repo=Path(repo);returned=Path(returned);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    old=repo/'artifacts/mainstudy_development_run_private/upload';original=read(old/'MANIFEST.json')
    status=read(returned/'STATUS.json')
    if sha(old/'MANIFEST.json')!=status['bindings']['upload']:raise ValueError('original upload differs from returned binding')
    verify(old,original['files'])
    bundle=out/'upload';bundle.mkdir()
    selected={}
    def cp(base,name,target=None):
        dest=bundle/(target or name);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(base)/name,dest)
        return sha(dest)
    needed={'D1_RESULT.json','Q6_COMPATIBILITY.json','Q6_ENVIRONMENT.json','requirements-colab.txt','training_seeds.json',
        'attack_x.npy','attack_y.npy','starts.npy','public_basis.npy','INPUT_PROVENANCE.json',
        'WINDOWS_IO_COST.json','WINDOWS_SOLVER_COST.json',
        'validation_controls_f64.npy','validation_tangent_truth_f64.npy','validation_truth_f64.npy','validation_v.npy','validation_x.npy'}
    needed.update('training_'+n+'.npy' for n in ('x','v','y','dy'))
    needed.update('key_truth_'+str(i)+'.npy' for i in range(3))
    needed.update(n for n in original['files'] if n.startswith('tasks/'))
    needed.update(n for n in original['files'] if n.startswith('code/') and not n.startswith('code/tpfa32_analysis/'))
    for name in sorted(needed):selected[name]=cp(old,name)
    cp(old,'MANIFEST.json','ORIGINAL_UPLOAD_MANIFEST.json')
    # No new key draws: export only the exact existing v[0] needed for the K gate.
    # Drop wrong-key coefficients, pattern tensors and all B1 key files.
    import numpy as np
    key_bindings={}
    for i in range(3):
        name=f'keys_B3_{i}.npz'
        with np.load(old/name,allow_pickle=False) as z:v=z['v'][0].copy()
        target=f'key_direction_{i}.npy';np.save(bundle/target,v,allow_pickle=False)
        key_bindings[target]=dict(source=name,source_sha256=original['files'][name],selection='v[0], original dtype and bytes',sha256=sha(bundle/target))
    # Refresh successor source ONLY. Parent source snapshot/manifest stays immutable.
    source_files={}
    for p in (repo/'tools/msdev').glob('*.py'):
        if p.name in ('handoff.py','unmarked_handoff.py','marking_handoff.py'):continue
        dest='code/msdev/'+p.name;cp(p.parent,p.name,dest);source_files['tools/msdev/'+p.name]=sha(p)
    cp(repo/'tools','mainstudy_development.py','code/mainstudy_development.py')
    source_files['tools/mainstudy_development.py']=sha(repo/'tools/mainstudy_development.py')
    write(bundle/'SOURCE_BINDINGS.json',dict(files=source_files,parent_source_binding=status['bindings']['sources'],kind='reviewed successor source; not inherited acceptance'))
    evidence={};imports={}
    evidence['parent/STATUS.json']=cp(returned,'STATUS.json','parent/STATUS.json')
    closures=[status['session']['closure']]+[str(p.relative_to(returned).as_posix()) for p in returned.glob('notebook_supervisor_*/RESULT.json')]
    if len(closures)!=2:raise ValueError('exact two parent closure records required')
    for name in closures:evidence['parent/'+name]=cp(returned,name,'parent/'+name)
    for p in returned.glob('failure_*.json'):evidence['parent/'+p.name]=cp(returned,p.name,'parent/'+p.name)
    for p in returned.glob('WORKER_FAILURE_*.json'):evidence['parent/'+p.name]=cp(returned,p.name,'parent/'+p.name)
    products={p['id']:p for p in status['products']}
    for name in ('D1','D2','D3','Rmain_1920','calibration','A4_cost','wrapper_cost'):
        if read(returned/'products'/f'{name}.json')!=products[name]:raise ValueError('parent receipt mismatch')
        rel=f'products/{name}.json';evidence['parent/'+rel]=cp(returned,rel,'parent/'+rel)
    for name in ('D1','D2','D3','Rmain_1920'):
        for f,h in products[name]['files'].items():
            if f.startswith('states/') and name!='Rmain_1920':continue
            if sha(returned/f)!=h:raise ValueError('returned parent product drift: '+f)
            imports['parent/'+f]=cp(returned,f,'parent/'+f)
    parent=dict(bindings=status['bindings'],runtime=status['runtime'],evidence_files=evidence,import_files=imports,
        closure_records=['parent/'+n for n in closures],historical_scientific_stop='preserved; not recovered',
        original_upload_sha256=sha(old/'MANIFEST.json'),imported_input_identities={n:original['files'][n] for n in needed if not n.startswith('code/')},
        derived_key_direction_bindings=key_bindings,excluded='159 recovery checkpoints, marked models/states, wrong keys, B10, embedding sets, full projection caches and unrelated work')
    write(bundle/'PARENT.json',parent)
    config=dict(read(old/'CONFIG.json'),continuation_kind=KIND,parent_sha256=sha(bundle/'PARENT.json'),output_name=OUTPUT,
        real_job='UNSTARTED; separate explicit launch instruction required',
        continuation_scope='93128 updates: Rstar87168 + ordinary clean5760 + A5B2cost200; no marked work',
        bindings=dict(parent=status['bindings']['sources'],sources=sha(bundle/'SOURCE_BINDINGS.json'),Q6_manifest=status['bindings']['Q6_manifest'],
                      parent_receipt=sha(bundle/'PARENT.json')))
    # Config identity is bound by the manifest; no self-referential hash.
    write(bundle/'CONFIG.json',config);validate_parent(bundle)
    if diagnostic_review is not None:
        from .marking_handoff import add_inputs
        add_inputs(bundle,old,returned,diagnostic_review)
    names=[p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()]
    m=manifest(bundle,names,'unmarked-remaining-development-upload');write(bundle/'MANIFEST.json',m);verify(bundle,m['files'])
    archive=out/ARCHIVE
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_STORED) as z:
        for name in sorted(names+['MANIFEST.json']):z.write(bundle/name,name)
    ready=dict(archive=ARCHIVE,archive_sha256=sha(archive),manifest_sha256=sha(bundle/'MANIFEST.json'),bytes=archive.stat().st_size,
        parent_sha256=sha(bundle/'PARENT.json'),sources_sha256=sha(bundle/'SOURCE_BINDINGS.json'),real_job='UNSTARTED',
        original_deadline=DEADLINE,parent_charged_seconds=21600,remaining_seconds=9900,graph=continuation_graph())
    if diagnostic_review is not None:
        ready['diagnostic']=dict(input_sha256=sha(bundle/'DIAGNOSTIC_INPUTS.json'),models=21,scoring_rows=24,total_seconds=600,
            separate_outcome=True,real_job='UNSTARTED')
    write(out/'READY.json',ready);notebook(out,ready)
    state=dict(completed=[],rates=status['rates'],deadline=DEADLINE,session=dict(end=9900))
    cost=projection(state,clock=lambda:0)
    if diagnostic_review is not None:
        cost.update(diagnostic_runtime='unmeasured; 600-second full envelope, inside the same 9900 seconds',
            diagnostic_envelope_seconds=600,setup_plus_diagnostic_ceilings=1980,
            full_campaign_fits=None,extra_upload_transfer_cost='unmeasured; inside charged window',
            diagnostic_not_an_optimizer_update=True)
    write(out/'COST_PROJECTION.json',cost)
    return ready

def notebook(out,ready):
    cells=[]
    def md(n,s):cells.append(dict(id=f'md{n}',cell_type='markdown',metadata={},source=f'## {n}. {s}'.splitlines(keepends=True)))
    def code(n,s):cells.append(dict(id=f'code{n}',cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True)))
    md(1,'UNSTARTED — wait for explicit launch approval\nUse a CPU runtime for cells 1–3. Upload only the named successor ZIP. It contains private DEVELOPMENT inputs, no final material. The original scientific stop remains immutable. This is a separate unmarked continuation, not infrastructure recovery. After reservation switch to A100 and repeat cells 1–3; then run 4–5. The same charged window must cover the switch, setup, idle, computation, transfers and disconnect. Latest possible reservation: 16:08:04 UK, 5 October 2026. No new window after interruption.')
    code(1,f'''import sys, pathlib, json, time, threading, subprocess, hashlib, zipfile
sys.dont_write_bytecode=True
from google.colab import drive, files, runtime as colab_runtime
drive.mount('/content/drive')
root=pathlib.Path('/content/drive/MyDrive/{OUTPUT}')
try:
    HAS_GPU=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,timeout=10).returncode==0
except FileNotFoundError: HAS_GPU=False
if HAS_GPU:
    if not (root/'STATUS.json').exists():
        colab_runtime.unassign();raise RuntimeError('No prebooked window. Reserve on CPU first.')
    early=json.loads((root/'STATUS.json').read_text())
    if not early['session'].get('prebooked') or early['session']['closed'] or time.time()>=early['session']['end']-60:
        colab_runtime.unassign();raise RuntimeError('Window closed/expired. No clock reset.')
    if 'disconnect_timer' not in globals():
        disconnect_timer=threading.Timer(max(0,min(early['deadline'],early['session']['end'])-time.time()),colab_runtime.unassign)
        disconnect_timer.daemon=True;disconnect_timer.start()
print('GPU connected:',HAS_GPU,'Original deadline: 18:53:04 UK 5 October 2026')
''')
    md(2,'Verify the minimal upload\nOn CPU, upload MainStudy_Unmarked_Upload.zip. A hash-verified copy is kept on Drive for the GPU switch. Never upload the original whole return or replace the parent directory.')
    code(2,f'''ARCHIVE='{ARCHIVE}'
ARCHIVE_SHA='{ready['archive_sha256']}'
MANIFEST_SHA='{ready['manifest_sha256']}'
stored=pathlib.Path('/content/drive/MyDrive/OperatorMark_MainStudy_Unmarked_Input_20261005.zip')
def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
if not stored.exists():
    assert not HAS_GPU,'Upload on CPU before reservation.'
    uploaded=files.upload()
    assert set(uploaded)=={{ARCHIVE}},'Upload exactly the successor ZIP'
    assert hashlib.sha256(uploaded[ARCHIVE]).hexdigest()==ARCHIVE_SHA
    with stored.open('xb') as f:f.write(uploaded[ARCHIVE])
assert digest(stored)==ARCHIVE_SHA,'Upload drift'
bundle=pathlib.Path('/content/msdev_unmarked_upload')
if not bundle.exists():
    with zipfile.ZipFile(stored) as z:
        names=z.namelist()
        assert len(names)==len(set(n.casefold() for n in names))
        for item in z.infolist():
            p=pathlib.PurePosixPath(item.filename)
            assert not p.is_absolute() and '..' not in p.parts and '\\\\' not in item.filename and ':' not in item.filename
            assert (item.external_attr>>16)&0o170000!=0o120000 and not item.flag_bits&1
        bundle.mkdir();z.extractall(bundle)
assert digest(bundle/'MANIFEST.json')==MANIFEST_SHA
for name,h in json.loads((bundle/'MANIFEST.json').read_text())['files'].items():
    assert digest(bundle/name)==h,name
sys.path.insert(0,str(bundle/'code'))
from msdev.unmarked import reserve, validate_parent
validate_parent(bundle)
print('Upload and stopped-parent bindings verified. No model executed.')
''')
    md(3,'Reserve once, only after separate approval\nAll 9,900 remaining seconds are charged prospectively. No refund, extra recovery or extension. This cell refuses if the full envelope no longer fits the original deadline. Re-running after the GPU switch only reads the same unused reservation. Ordinary scientific failures remain stops.')
    code(3,'''if (root/'STATUS.json').exists():
    s=reserve(bundle,root,MANIFEST_SHA,None,None,approved=True)
else:
    assert not HAS_GPU,'Reserve before attaching A100'
    assert input('After explicit user launch approval, type APPROVED UNMARKED: ')=='APPROVED UNMARKED'
    credits=float(input('Current available compute units: '))
    rate=float(input('Current A100 units/hour: '))
    s=reserve(bundle,root,MANIFEST_SHA,credits,rate,approved=True)
print('Charged aggregate:',s['gpu_charged'],'seconds; session ends UTC',time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(s['session']['end'])))
print('If on CPU: switch to A100 now, then repeat cells 1–3. Do not recreate this directory.')
''')
    md(4,'Recreate the exact Q6 environment\nA100 required. The bound requirements, isolated interpreter and existing exact runtime/parity checks are retained. Setup ceiling 1,380 seconds, inside the charged window. No automatic setup retry, dependency substitution or numerical replay.')
    code(4,'''assert HAS_GPU,'A100 GPU runtime required'
from msdev.colab_environment import setup,launch
from msdev.records import write
try:
    PYTHON=setup(bundle,root,pathlib.Path('/content/msdev_unmarked_q6_venv'))
except BaseException as exc:
    write(root/('SETUP_FAILURE_'+str(time.time_ns())+'.json'),{'error':repr(exc),'kernel_python':sys.version})
    print('STOPPED: setup/runtime mismatch; see saved receipt. No retry.')
    colab_runtime.unassign();raise
print('Supervisor and worker interpreter:',PYTHON)
''')
    md(5,'Run once with live stage/checkpoint progress\nOnly parent import, A5-B2 cost, clean-owner attacks, remaining R* branches/selection and unmarked readout are reachable. The exact prescribed 93,128 updates include the timing window; no extra warm-up training. Known remaining costs retain 20% margin; unknowns never establish fit. Hard stops preserve partial work. Nested owned cleanup and finalization remain inside the charged deadline. This cell reports terminal state and then disconnects automatically.')
    code(5,'''stop_progress=threading.Event()
def show_progress():
    offset=0
    while not stop_progress.wait(2):
        path=root/'progress.jsonl'
        if path.exists():
            with path.open('r') as f:
                f.seek(offset)
                for line in f:print(line.rstrip(),flush=True)
                offset=f.tell()
watcher=threading.Thread(target=show_progress,daemon=True);watcher.start()
receipt=None;launch_error=None
try:
    receipt=launch(PYTHON,bundle,root,MANIFEST_SHA)
    print('Owned outer closure:',receipt,flush=True)
except BaseException as exc:
    launch_error=repr(exc)
    print('STOPPED or unresolved: ',repr(exc),'Inspect saved supervisor receipts; no automatic restart.',flush=True)
    raise
finally:
    stop_progress.set();watcher.join(timeout=5)
    saved=json.loads((root/'STATUS.json').read_text())
    finalized=(root/'OUTPUT_MANIFEST.json').exists() and json.loads((root/'OUTPUT_MANIFEST.json').read_text()).get('complete') is True
    closed=receipt is not None and receipt.get('closure',{}).get('resolved') is True
    label='UNMARKED COMPLETE' if finalized and closed and saved['state']=='COMPLETE' else 'PAUSED, INCOMPLETE' if closed and saved['state']=='PAUSED_SESSION' else 'FAILED/INCOMPLETE — inspect closure receipts'
    print(label,'Launch reason:',launch_error,flush=True)
    print('STATE:',saved['state'],'ACTIVE:',saved.get('active'),'REASON:',saved.get('error'),flush=True)
    print('Completed nodes:',saved['completed'],'new updates:',saved.get('executed_updates'),flush=True)
    print('Original scientific stop remains binding; marked development is not complete.',flush=True)
    try:
        files.download(str(root/'STATUS.json'))
        if (root/'RESULT.json').exists():files.download(str(root/'RESULT.json'))
    finally:colab_runtime.unassign()
''')
    md(6,'Retrieve evidence after disconnect\nThe complete successor output is in MyDrive/'+OUTPUT+'. Download that folder from Drive to Windows. Preserve the input ZIP and original stopped campaign. COMPLETE means only the unmarked successor finished; PAUSED_SESSION means incomplete; STOPPED includes its reason; missing closure is unresolved. Never reset STATUS or repeat training to obtain a different result.')
    if ready.get('diagnostic'):
        for cell in cells:
            s=''.join(cell['source'])
            if cell['id']=='md2':s+='\nThis revised upload also contains the 21 saved DEVELOPMENT marking/sham models, their 128-control embedding sets and true keys. No wrong-key arrays or marking optimizer states are included.'
            if cell['id']=='code2':s=s.replace('OperatorMark_MainStudy_Unmarked_Input_20261005.zip','OperatorMark_MainStudy_Unmarked_Diagnostic_Input_20261005.zip')
            if cell['id']=='md5':s+='\nFirst, one separate inference-only marking diagnostic runs, at most 570 seconds work plus 30 seconds owned cleanup. It reports eight fixed batches of 16 and saved validation comparisons; these are not historical training-batch scores. Its failure/timeout is reported separately and cannot declare unmarked failure or success. Unresolved ownership stops everything. The original 93,128-update graph is unchanged. All setup, diagnostic, training, idle and closure still share 9,900 seconds and the original deadline.'
            if cell['id']=='code5':
                s=s.replace('    receipt=launch(PYTHON,bundle,root,MANIFEST_SHA)',
                    "    from msdev.marking_diagnostic import launch as launch_diagnostic\n    diagnostic=launch_diagnostic(PYTHON,bundle,root,MANIFEST_SHA)\n    print('SEPARATE MARKING DIAGNOSTIC:', 'COMPLETE' if diagnostic['complete'] else 'FAILED/INCOMPLETE', 'timeout:', diagnostic['timed_out'], flush=True)\n    receipt=launch(PYTHON,bundle,root,MANIFEST_SHA)")
                s=s.replace("        if (root/'RESULT.json').exists():files.download(str(root/'RESULT.json'))", "        if (root/'RESULT.json').exists():files.download(str(root/'RESULT.json'))\n        if (root/'marking_diagnostic/OUTCOME.json').exists():files.download(str(root/'marking_diagnostic/OUTCOME.json'))")
            if cell['id']=='md6':s+='\nThe diagnostic has its own marking_diagnostic/OUTCOME.json and individually committed score rows. Its completion is independent of UNMARKED COMPLETE and does not reopen B3 selection or establish marked-owner conclusions.'
            cell['source']=s.splitlines(keepends=True)
    write(Path(out)/NOTEBOOK,dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5))
