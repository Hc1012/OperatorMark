"""Small, ordinary records and immutable input/output identities."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

def sha(path,check=lambda:None):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):check();h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value,replace=False,*,durable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if not replace:
        if path.exists():raise FileExistsError(path)
        pending=path.with_name(path.name+f'.writing_{time.time_ns()}')
        with pending.open('x',encoding='utf-8') as f:
            f.write(text)
            if durable:f.flush();os.fsync(f.fileno())
        # One writer; interrupted partials remain named as such, never complete JSON.
        if path.exists():raise FileExistsError(path)
        os.rename(pending,path)
    else:
        # A killed sole writer may leave an earlier pending file. It is evidence,
        # never a commit and never reused; controller ownership checks precede resume.
        pending=path.with_name(path.name+f'.pending_{os.getpid()}_{time.time_ns()}')
        with pending.open('x',encoding='utf-8') as f:
            f.write(text)
            if durable:f.flush();os.fsync(f.fileno())
        # One writer. No live monitoring of replace-in-place files by the worker.
        for i in range(5):
            try:os.replace(pending,path);break
            except PermissionError:
                if i==4:raise
                time.sleep(.02)

def ensure_record(path,value):
    """Idempotent immutable publication; a different committed value is drift."""
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('committed record differs: '+str(path))
    else:write(path,value)

def publish_zip(archive,root,names,check=lambda:None):
    """Publish complete bytes last. Unreceipted old/partial archives are retained."""
    archive=Path(archive);root=Path(root)
    if archive.exists():
        archive.rename(archive.with_name(archive.name+f'.unreceipted_{time.time_ns()}'))
    pending=archive.with_name(archive.name+f'.writing_{os.getpid()}_{time.time_ns()}')
    # Bound deadline checks within a large member, not only between members.
    # Flush the completed local ZIP before publication; this is NOT evidence of
    # persistence at a remote object service or through a mounted-Drive cache.
    with pending.open('xb') as output:
        with zipfile.ZipFile(output,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
            for n in names:
                check();path=root/safe_name(n)
                info=zipfile.ZipInfo.from_file(path,n);info.compress_type=zipfile.ZIP_STORED
                with path.open('rb') as source,z.open(info,'w',force_zip64=True) as member:
                    while data:=source.read(1<<20):check();member.write(data)
        output.flush();os.fsync(output.fileno())
    check();os.rename(pending,archive)

def progress(root,phase,**fields):
    row=dict(time=time.time(),phase=phase,**fields)
    with (Path(root)/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row,allow_nan=False),flush=True)

def safe_name(name):
    from pathlib import PurePosixPath
    p=Path(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or PurePosixPath(name).as_posix()!=name or any(x in ('..','.','') for x in name.split('/')):
        raise ValueError('unsafe relative member: '+repr(name))
    reserved={'CON','PRN','AUX','NUL'}|{f'{p}{i}' for p in ('COM','LPT') for i in range(1,10)}
    if any(part.rstrip(' .')!=part or part.split('.')[0].upper() in reserved or any(c in part for c in '<>|?*') for part in name.split('/')):
        raise ValueError('unsafe Windows member: '+repr(name))
    return p

def verify(root,files):
    root=Path(root).resolve()
    for name,want in files.items():
        p=root/safe_name(name)
        if p.is_symlink() or root not in p.resolve().parents or not p.is_file() or sha(p)!=want:
            raise ValueError('input/source identity mismatch: '+name)

def manifest(root,names,kind):
    root=Path(root)
    names=sorted(names)
    if len(names)!=len(set(names)):raise ValueError('duplicate manifest member')
    return dict(kind=kind,files={n:sha(root/safe_name(n)) for n in names})

def verify_return(root):
    root=Path(root);m=read(root/'OUTPUT_MANIFEST.json');verify(root,m['files'])
    if m.get('complete') is not True:raise ValueError('incomplete return')
    if any(n not in m['files'] for n in ('RESULT.json','SUPERVISOR.json','OUTPUT_INVENTORY.json','STATUS.json')):raise ValueError('missing bound completion/closure records')
    result=read(root/'RESULT.json')
    if result.get('complete') is not True:raise ValueError('missing scientific completion')
    from .graph import operation_graph
    if result.get('scope')=='unmarked remaining development only':
        from .unmarked import continuation_graph,verify_result_scope
        verify_result_scope(root,result)
        graph=continuation_graph()
    else:graph=operation_graph()
    products=result['products']
    if [r['id'] for r in products]!=[n['id'] for n in graph]:raise ValueError('incomplete/duplicate prescribed operation graph')
    for node,row in zip(graph,products):
        if row.get('bindings')!=result['bindings'] or row.get('runtime')!=result['runtime']:raise ValueError('product correspondence differs')
        for name,digest in row['files'].items():
            if m['files'].get(name)!=digest:raise ValueError('unbound model/design/result: '+name)
        if node['kind']=='train':
            if row.get('skipped'):
                if not node.get('conditional'):raise ValueError('required arm omitted')
                selection=next(r for r in products if r['id']=='mark_selection')
                factor=selection['selection']['B1']['factor']
                grid=next(r for r in selection['grid']['B1'] if r['factor']==factor)
                if grid['owners'][node['owner']]['qualified']:raise ValueError('qualified conditional arm omitted')
            else:
                required=['models/'+node['id']+'.pt','states/'+node['id']+'.pt']
                if node.get('evaluate'):required+=['evaluations/'+node['id']+'/'+n for n in ['RESULT.json','designs.npy']+[f'task_{i:02d}.json' for i in range(16)]]
                if any(n not in row['files'] for n in required):raise ValueError('required model/state/design/task missing: '+node['id'])
    closure=read(root/'SUPERVISOR.json')
    if (closure.get('complete') is not True or not closure.get('closure',{}).get('resolved') or closure.get('returncode')!=0
        or closure.get('timed_out') is not False or closure.get('stdout_closed') is not True or closure.get('bindings')!=result['bindings']):
        raise ValueError('return lacks clean process closure')
    return dict(verified=True,manifest_sha256=sha(root/'OUTPUT_MANIFEST.json'),
                producer_hashes_are_attestation=False,files=len(m['files']))

def extract(archive,target):
    target=Path(target);target.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        rows=z.infolist();names=[r.filename for r in rows]
        if len(names)!=len(set(n.casefold().rstrip('/') for n in names)):raise ValueError('duplicate/case-colliding archive entry')
        if len(rows)>20000 or sum(r.file_size for r in rows)>8*1024**3:raise ValueError('return archive safety ceiling; originals retained')
        for r in rows:
            safe_name(r.filename.rstrip('/'))
            if (r.external_attr>>16)&0o170000==0o120000:raise ValueError('archive symlink')
            if r.flag_bits&1:raise ValueError('encrypted archive')
        z.extractall(target)
