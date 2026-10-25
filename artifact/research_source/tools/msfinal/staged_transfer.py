"""Private stage transfer: exact membership/readback, no clocks or numerics."""
from pathlib import Path,PurePosixPath
import hashlib,zipfile,time
from msdev.records import read,write,sha,verify

def export_cpu(root,deadline):
    root=Path(root);receipt=read(root/'cpu/CLOSURE.json')
    if receipt.get('closed') is not True:raise ValueError('CPU closure required')
    verify(root,receipt['files'])
    names=dict(receipt['files']);names['cpu/CLOSURE.json']=sha(root/'cpu/CLOSURE.json')
    archive=root.with_name(root.name+'_CPU_RETURN.zip')
    def check():
        if time.time()>=deadline:raise TimeoutError('CPU export deadline')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for n,h in names.items():
            check()
            if sha(root/n)!=h:raise ValueError('CPU source changed')
            z.write(root/n,n)
    with zipfile.ZipFile(archive) as z:
        if sorted(z.namelist())!=sorted(names):raise ValueError('CPU archive membership')
        for n,h in names.items():
            check()
            with z.open(n) as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=h:raise ValueError('CPU archive bytes')
    out=dict(archive_sha256=sha(archive),files=names,window_sha256=sha(root/'WINDOW.json'),closed=True)
    write(archive.with_suffix('.json'),out);return archive

def extract(archive,expected,destination):
    archive=Path(archive);destination=Path(destination)
    if sha(archive)!=expected:raise ValueError('upload archive identity')
    if destination.exists():raise FileExistsError('preserve existing destination')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('duplicate archive members')
        for i in z.infolist():
            p=PurePosixPath(i.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in i.filename or ':' in i.filename or (i.external_attr>>16)&0o170000==0o120000:raise ValueError('unsafe archive member')
        destination.mkdir(parents=True)
        z.extractall(destination)
    return destination

def import_cpu(archive,receipt,destination):
    r=read(receipt)
    if r.get('closed') is not True:raise ValueError('CPU closure unavailable')
    with zipfile.ZipFile(archive) as z:
        if sorted(z.namelist())!=sorted(r['files']):raise ValueError('CPU transfer membership')
    root=extract(archive,r['archive_sha256'],destination);verify(root,r['files'])
    c=read(root/'cpu/CLOSURE.json')
    if sha(root/'WINDOW.json')!=r['window_sha256'] or c['window_sha256']!=r['window_sha256']:raise ValueError('CPU window drift')
    verify(root,c['files']);return root
