"""Private, non-numerical continuation preparation; accepted production unchanged."""
from pathlib import Path
import hashlib,json,os,shutil

GiB=1<<30
MiB=1<<20
POLICY=dict(owners=11,metadata_per_stage_bytes=16*MiB,pin_map_bytes=42048,
    destination_bytes=1024,cpu_return_per_owner_bytes=512*MiB,
    file_limit=12000,worker_byte_limit=32*GiB,transport_byte_limit=36*GiB,
    failure_reserve_per_stage=2*GiB,headroom=5*GiB)

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def inventory(root):
    root=Path(root);rows={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('symlink in retained publication')
        if p.is_file():rows[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=digest(p))
    return rows

def relocate_publication(root,previous_session,expected,permission,check=lambda:None):
    """Only a fresh restored COPY moves; immutable original archive is untouched.

    Caller must independently admit boundary/custody and hash-bind this helper.
    This additional exact relocation permission is consumed before any mutation.
    """
    root=Path(root).resolve();src=root/'publication_evidence'
    if len(previous_session)!=64 or any(c not in '0123456789abcdef' for c in previous_session):raise ValueError('session identity')
    target=root/'publication_history'/previous_session/'publication_evidence'
    if not target.resolve().is_relative_to(root) or target.exists():raise ValueError('fresh owned relocation destination required')
    wanted=dict(previous_session_sha256=previous_session,files=expected,
                source='publication_evidence',destination=target.relative_to(root).as_posix())
    if permission!=wanted:raise ValueError('independently reviewed exact relocation differs')
    check()
    if not src.is_dir() or src.is_symlink() or inventory(src)!=expected:raise ValueError('retained publication identity differs')
    target.parent.mkdir(parents=True,exist_ok=False)
    os.rename(src,target)
    if inventory(target)!=expected:raise ValueError('relocated receipt changed; preserve partials')
    check()
    raw=(json.dumps(dict(wanted,original_archive_unchanged=True),sort_keys=True,indent=2)+'\n').encode()
    with (target.parent/'RELOCATION.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return wanted

def envelope(n,*,merged=False,input_bytes=0,expanded_input_bytes=0,runtime_bytes=0):
    if type(n) is not int or not 1<=n<=11 or merged and n!=11:raise ValueError('owner-wave population')
    if any(type(x) is not int or x<0 for x in (input_bytes,expanded_input_bytes,runtime_bytes)):raise ValueError('nonnegative measured operand required')
    p=POLICY;H=8*p['metadata_per_stage_bytes']+2*p['pin_map_bytes']+4*MiB
    W=633301784+H
    # Conservatively include a full second training phase even where finalize
    # retains fewer dependencies; each of two monitor merges can copy the same
    # bounded CPU output/closure observations. No compression or omission credit.
    merged_extra=n*(W+2*p['cpu_return_per_owner_bytes']+8*MiB) if merged else 0
    worker=n*W+merged_extra
    framing=p['file_limit']*(200+2*512)+1024
    archive=worker+framing
    R=3146240+p['destination_bytes']
    local_meta=256*R+4*H+16*MiB
    remote_meta=32*R+H+8*MiB
    transfer=archive+128*MiB+remote_meta+1024
    # Prior source outputs counted in worker, not again as payload. Two input
    # copies are explicit compressed upload and expanded staging; CPU returns
    # retain both raw archives and extracted evidence until permanent custody.
    cpu_copies=2*n*p['cpu_return_per_owner_bytes']
    linux=input_bytes+expanded_input_bytes+runtime_bytes+worker+archive+20*64*MiB+local_meta+9318536+cpu_copies+p['failure_reserve_per_stage']+p['headroom']
    windows=input_bytes+2*archive+22*64*MiB+local_meta+cpu_copies+p['failure_reserve_per_stage']+p['headroom']
    drive=transfer+p['failure_reserve_per_stage']
    return dict(owners=n,merged=merged,H=H,per_owner_W=W,worker=worker,archive=archive,
        transfer=transfer,cpu_archive_and_expansion=cpu_copies,Linux=linux,Windows=windows,Drive=drive,
        runtime_bytes=runtime_bytes,input_bytes=input_bytes,expanded_input_bytes=expanded_input_bytes)

def admit_operands(paths,pin_map_bytes,destination,session,index,capacity):
    """Run before dispatch; policy proposal must first be independently accepted."""
    unique={Path(p).resolve() for p in paths}
    total=sum(p.stat().st_size for p in unique)
    if total>POLICY['metadata_per_stage_bytes']:raise ValueError('normal metadata operand exceeds approved proposal; preserve and stop')
    if pin_map_bytes>POLICY['pin_map_bytes'] or len(destination.encode())>POLICY['destination_bytes']:raise ValueError('source/destination operand exceeds proposed envelope')
    if session['evidence_limits']!={'file_limit':POLICY['file_limit'],'byte_limit':POLICY['worker_byte_limit']}:raise ValueError('worker envelope not independently bound')
    if index['transport']['byte_limit']!=POLICY['transport_byte_limit']:raise ValueError('transport envelope not independently bound')
    if capacity.get('operational_policy')!=POLICY:raise ValueError('capacity policy not independently bound')
    if capacity.get('accepted') is not True:raise ValueError('independent capacity acceptance required')
    return total

def regular_worker_interpreter(path,expected_sha256,check=lambda:None):
    """Carry forward O01's reviewed byte-identical venv-path correction."""
    p=Path(path);check()
    if digest(p)!=expected_sha256 or p.stat().st_size>32*MiB:raise ValueError('worker interpreter bytes differ')
    if not p.is_symlink():return dict(path=str(p),sha256=digest(p),already_regular=True)
    target=p.resolve();backup=p.with_name('python.original_symlink_O02');temp=p.with_name('python.identical_copy_O02')
    if backup.exists() or backup.is_symlink() or temp.exists():raise FileExistsError('preserve earlier interpreter preparation')
    shutil.copy2(target,temp)
    with temp.open('rb') as f:os.fsync(f.fileno())
    check()
    if digest(temp)!=expected_sha256:raise ValueError('interpreter copy differs')
    old_link=os.readlink(p);p.rename(backup);temp.rename(p)
    if p.resolve()!=p.absolute() or digest(p)!=expected_sha256:raise ValueError('worker path correction incomplete')
    return dict(path=str(p),sha256=digest(p),original_target=str(target),original_link=old_link,preserved_symlink=str(backup),already_regular=False)

def admit_cpu_return_manifest(path):
    path=Path(path);v=json.loads(path.read_bytes());total=path.stat().st_size
    for name,row in v['files'].items():
        p=(path.parent/name).resolve()
        if not p.is_relative_to(path.parent.resolve()) or p.is_symlink():raise ValueError('CPU return path escape')
        if p.stat().st_size!=row['bytes']:raise ValueError('CPU return member size differs')
        total+=row['bytes']
    if total>POLICY['cpu_return_per_owner_bytes']:raise ValueError('CPU return exceeds proposed envelope; preserve and stop')
    # Exact hashes/membership/science are still checked by existing D6 consumers.
    return total
