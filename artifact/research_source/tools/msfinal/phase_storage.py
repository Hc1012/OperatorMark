"""Exact phase capsules and capacity-bounded restore; no eviction or replay.

All trust is supplied externally using existing AcceptedRecord interfaces. This
module never promotes a producer receipt into independent acceptance. Failed
downloads/extractions remain in fresh directories. Nothing is deleted.
"""
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from msdev.records import read,write,sha,safe_name,publish_zip
from operatormark.corpus.state_v2_prerequisites import require_accepted
from . import readback as R
from .graph import identity
from .marking import identity as require_hash
from .session import closure_ok

def _record_bytes(value):
    """Exact bytes emitted by records.write on this host, including newlines."""
    text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    return len(text.replace('\n',os.linesep).encode('utf-8'))

def _write_record(path,value):
    # Later remote identities/exception text cannot be guessed at admission.
    # Check the actual record before writing; never truncate it to a guessed cap.
    if shutil.disk_usage(Path(path).parent).free<_record_bytes(value):
        raise OSError('complete phase metadata record capacity unavailable')
    write(path,value,durable=True)

def verify_archive(archive,files,check):
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or len(names)!=len(set(n.casefold() for n in names)) or set(names)!=set(files):
            raise ValueError('archive exact membership differs')
        for info in z.infolist():
            check();safe_name(info.filename)
            if info.is_dir() or (info.external_attr>>16)&0o170000==0o120000 or info.flag_bits&1:
                raise ValueError('non-regular archive member')
            if info.file_size!=files[info.filename]['bytes']:raise ValueError('archive size declaration differs')
            h=hashlib.sha256();size=0
            with z.open(info) as f:
                while data:=f.read(1<<20):
                    check();size+=len(data)
                    if size>info.file_size:raise ValueError('archive member overflow')
                    h.update(data)
            if dict(bytes=size,sha256=h.hexdigest())!=files[info.filename]:raise ValueError('archive member changed')

def publish_phase(root,name,destination,closure,permit,transport,*,check=lambda:None,clock=None):
    """Export exactly one independently closed phase; no cumulative mega-ZIP."""
    import time
    clock=clock or time.time;root=Path(root).resolve();safe_name(name);dest=Path(destination).resolve()
    receipt_path=root/'products'/(identity(name)+'.json');product=read(receipt_path)
    closed=closure.read();result=read(root/'RESULT.json')
    if (not closure_ok(closed) or closed.get('product_readback') is not True
        or closed.get('worker_result_sha256')!=sha(root/'RESULT.json')
        or result.get('product_receipts',{}).get(name)!=sha(receipt_path)
        or product.get('name')!=name or product.get('binding')!=result.get('binding')):
        raise ValueError('independently closed exact phase required')
    grant=require_accepted(permit,kind='main_study_final_phase_export',bindings=dict(
        worker_result_sha256=sha(root/'RESULT.json'),closure_sha256=closure.sha256,
        phase_receipt_sha256=sha(receipt_path),transport_source_sha256=transport.binding['source_sha256'],destination=str(dest)))
    if (grant.get('domain')!='final' or grant.get('active_streams')!=[] or grant.get('deadline',float('inf'))>closed['finalization_deadline']):
        raise ValueError('closed-phase deadline/stream scope differs')
    def bounded():
        check()
        if clock()>=grant['deadline']:raise TimeoutError('phase export original deadline')
    bounded();source=root/'work'/name
    files=R.inventory(source,bounded,file_limit=grant['file_limit'],byte_limit=grant['byte_limit'])
    if files!=product['files']:raise ValueError('phase source differs from accepted closure')
    if dest==source or dest.is_relative_to(root) or root.is_relative_to(dest):raise ValueError('separate local export required')
    archive_bound=sum(v['bytes'] for v in files.values())+sum(200+2*len(n.encode()) for n in files)+1024
    inventory=dict(files=files,phase=name,phase_receipt_sha256=sha(receipt_path))
    manifest_bound=dict(kind='closed_phase_capsule_v1',files=files,phase=name,phase_receipt_sha256=sha(receipt_path),
        worker_result_sha256=sha(root/'RESULT.json'),closure_sha256=closure.sha256,
        archive_sha256='0'*64,archive_bytes=archive_bound)
    # Source already occupies its volume. Include the actual large metadata,
    # not a fixed allowance that can be smaller than one inventory. Transport
    # logs/failures remain additional session-plan terms, not presumed zero.
    reserve=archive_bound+_record_bytes(inventory)+_record_bytes(manifest_bound)
    reserve+=receipt_path.stat().st_size+len(closure.expected)+len(permit.expected)
    if shutil.disk_usage(dest.parent).free<reserve:raise OSError('phase archive capacity missing')
    dest.mkdir(exist_ok=False)
    try:
        _write_record(dest/'OUTPUT_INVENTORY.json',inventory)
        with (dest/'PHASE_RECEIPT.json').open('xb') as f:
            f.write(receipt_path.read_bytes());f.flush();os.fsync(f.fileno())
        publish_zip(dest/'OUTPUT.zip',source,sorted(files),bounded);verify_archive(dest/'OUTPUT.zip',files,bounded)
        if R.inventory(source,bounded,file_limit=grant['file_limit'],byte_limit=grant['byte_limit'])!=files:raise ValueError('phase source mutation')
        manifest=dict(kind='closed_phase_capsule_v1',files=files,phase=name,phase_receipt_sha256=sha(receipt_path),
            worker_result_sha256=sha(root/'RESULT.json'),closure_sha256=closure.sha256,
            archive_sha256=sha(dest/'OUTPUT.zip',bounded),archive_bytes=(dest/'OUTPUT.zip').stat().st_size)
        _write_record(dest/'OUTPUT_MANIFEST.json',manifest)
        archive=transport.upload_verified(dest/'OUTPUT.zip',deadline=grant['deadline'],check=bounded)
        _write_record(dest/'DURABLE_RECEIPT.json',dict(archive=archive,archive_remote_verified=True,disconnect_permitted=False,
            scientific_acceptance=False,manifest_sha256=sha(dest/'OUTPUT_MANIFEST.json')))
        for filename,record in [('CLOSURE.json',closure),('PHASE_EXPORT_PERMISSION.json',permit)]:
            record.read()
            with (dest/filename).open('xb') as f:f.write(record.expected);f.flush();os.fsync(f.fileno())
        return transport.commit_records(dest,['OUTPUT_INVENTORY.json','OUTPUT_MANIFEST.json','DURABLE_RECEIPT.json',
            'CLOSURE.json','PHASE_EXPORT_PERMISSION.json','PHASE_RECEIPT.json'],deadline=grant['deadline'],check=bounded,archive=archive)
    except BaseException as exc:
        import traceback
        try:_write_record(dest/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),disconnect_permitted=False))
        except BaseException as reporting:exc.add_note('phase failure record unavailable: '+repr(reporting))
        raise

def restore_capsule(transport,commit_reference,expected_files,destination,*,deadline,check,file_limit,byte_limit):
    """Restore into an exclusive fresh tree; full byte readback, no acceptance.

    Archive+expanded contents remain local. Caller must account for both and for
    every earlier restore/failure. Missing capacity stops before network transfer.
    """
    import time
    caller_check=check
    def check():
        caller_check()
        if time.time()>=deadline:raise TimeoutError('original restore deadline expired')
    check();destination=Path(destination)
    if destination.exists():raise FileExistsError('restore destination/partial preserved; no automatic retry')
    if type(file_limit) is not int or type(byte_limit) is not int or len(expected_files)>file_limit:raise ValueError('restore membership bound')
    total=0
    for n,row in expected_files.items():
        safe_name(n);require_hash(row['sha256'])
        if type(row['bytes']) is not int or row['bytes']<0:raise ValueError('member size required')
        total+=row['bytes']
    if total>byte_limit:raise ValueError('restore payload bound')
    commit=transport.read_json(commit_reference,deadline=deadline,check=check)
    if commit.get('kind')!='OperatorMark_remote_export_commit_v1':raise ValueError('remote capsule commit schema')
    refs=commit['objects']
    for n in ('OUTPUT.zip','OUTPUT_INVENTORY.json','OUTPUT_MANIFEST.json','DURABLE_RECEIPT.json','CLOSURE.json'):
        if n not in refs:raise ValueError('remote recovery record missing: '+n)
    manifest=transport.read_json(refs['OUTPUT_MANIFEST.json'],deadline=deadline,check=check)
    inventory=transport.read_json(refs['OUTPUT_INVENTORY.json'],deadline=deadline,check=check)
    if manifest.get('files')!=expected_files or inventory.get('files')!=expected_files:raise ValueError('externally accepted exact membership differs')
    archive=refs['OUTPUT.zip']
    if archive['sha256']!=manifest['archive_sha256'] or archive['bytes']!=manifest['archive_bytes']:raise ValueError('archive binding differs')
    # All metadata, including closure and permissions, must survive runtime loss.
    metadata={}
    for n,ref in refs.items():
        if n!='OUTPUT.zip':metadata[n]=transport.read_json(ref,deadline=deadline,check=check)
    if manifest.get('kind')=='closed_phase_capsule_v1':
        if ('PHASE_RECEIPT.json' not in refs or refs['PHASE_RECEIPT.json']['sha256']!=manifest['phase_receipt_sha256']
                or metadata['PHASE_RECEIPT.json'].get('files')!=expected_files):raise ValueError('original phase receipt missing or inconsistent')
    restored=dict(commit=commit_reference,files=expected_files,metadata=metadata,
        complete=True,scientific_acceptance=False,deadline=deadline)
    # The fetched metadata is retained inside RESTORED.json, at its actual new
    # indentation depth, together with a further exact-membership inventory.
    extra=archive['bytes']+total+_record_bytes(restored)
    if shutil.disk_usage(destination.parent).free<extra:raise OSError('archive + restored phase + metadata capacity unavailable')
    destination.mkdir(exist_ok=False);partial=destination/'OUTPUT.zip.partial'
    try:
        with partial.open('xb') as stream:
            transport.verify(archive,deadline=deadline,check=check,sink=stream);stream.flush();os.fsync(stream.fileno())
        path=destination/'OUTPUT.zip';os.rename(partial,path)
        verify_archive(path,expected_files,check)
        contents=destination/'files';contents.mkdir()
        with zipfile.ZipFile(path) as z:
            for name,row in expected_files.items():
                check();target=contents/safe_name(name);target.parent.mkdir(parents=True,exist_ok=True)
                h=hashlib.sha256();size=0
                with z.open(name) as source,target.open('xb') as sink:
                    while data:=source.read(1<<20):
                        check();size+=len(data)
                        if size>row['bytes']:raise ValueError('extracted member overflow')
                        sink.write(data);h.update(data)
                    sink.flush();os.fsync(sink.fileno())
                if dict(bytes=size,sha256=h.hexdigest())!=row:raise ValueError('extraction mutation')
        if R.inventory(contents,check,file_limit=file_limit,byte_limit=byte_limit)!=expected_files:raise ValueError('restored independent readback differs')
        _write_record(destination/'RESTORED.json',restored)
        return contents
    except BaseException as exc:
        try:_write_record(destination/'RESTORE_FAILURE.json',dict(error=repr(exc),complete=False,partials_preserved=True))
        except BaseException as reporting:exc.add_note('restore failure record unavailable: '+repr(reporting))
        raise

class PhaseStore:
    """Externally accepted phase-to-volume/capsule map; no automatic eviction.

    Multiple bounded scratch volumes are supported. If their cumulative retained
    restores do not fit, stop; a new runtime/volume requires fresh admission under
    the SAME original window. Never pretend a phase file exists or rerun its body.
    """
    def __init__(self,record,transport,worker_binding,*,check,clock):
        self.record=record;self.transport=transport;self.check=check;self.clock=clock
        import inspect
        if sha(inspect.getfile(type(transport)))!=transport.binding['source_sha256']:raise ValueError('phase transport source drift')
        self.value=require_accepted(record,kind='main_study_final_phase_storage',bindings=dict(worker_binding,
            transport_source_sha256=transport.binding['source_sha256']))
        if self.value.get('domain')!='final' or self.value.get('active_streams')!=[]:raise ValueError('closed-phase storage only')
        from .bounded_stage import admitted_deadline
        self.deadline=admitted_deadline(worker_binding,self.value);self.paths={}
        self.member_paths={}
        entries=self.value.get('phases',{})
        resolved=[]
        for name,row in entries.items():
            safe_name(name)
            locations=[row['restore_directory']]
            if row.get('dependencies'):locations.append(row['dependencies']['restore_directory'])
            for location in locations:
                path=Path(location).resolve()
                if any(path==p or path.is_relative_to(p) or p.is_relative_to(path) for p in resolved):raise ValueError('overlapping phase restore locations')
                resolved.append(path)
    def archived(self,name):
        return 'permanent' in self.value.get('phases',{}).get(name,{})

    def verify_archived(self,name,product,receipt_sha256):
        """Independently admitted permanent receipt, original closure; no restore.

        Only the prospectively adopted custody route reuses the recorded semantic
        phase result. Exact original product bytes and archive membership were
        independently checked at permanent receipt, and are re-bound here.
        """
        from .archive_custody import POLICY,RECORDS
        from pathlib import PureWindowsPath
        if self.record.read()!=self.value:raise ValueError('phase storage snapshot drift')
        self.check()
        if self.clock()>=self.deadline:raise TimeoutError('original archive deadline')
        entry=self.value['phases'][name];r=entry.get('permanent',{})
        if (r.get('policy')!=POLICY or r.get('full_member_readback') is not True
                or r.get('phase')!=name or r.get('phase_receipt_sha256')!=receipt_sha256
                or entry['phase_receipt_sha256']!=receipt_sha256 or r.get('phase_files')!=product['files']
                or r.get('worker_binding')!=product['binding']):raise ValueError('permanent phase custody/receipt mismatch')
        locator=entry.get('permanent_receipt',{})
        require_hash(locator.get('sha256'))
        if (not PureWindowsPath(locator.get('path','')).is_absolute()
            or locator.get('snapshot_identity')!=identity(r)
            or not RECORDS|{'OUTPUT.zip','REMOTE_COMMIT_BYTES.json','ORIGINAL_REMOTE_COMMIT.json'}<=r.get('recovery_files',{}).keys()
            or r['recovery_files']['OUTPUT.zip']!=dict(bytes=r['archive_bytes'],sha256=r['archive_sha256'])
            or r['recovery_files']['PHASE_RECEIPT.json']['sha256']!=receipt_sha256):
            raise ValueError('permanent recovery locator/membership incomplete')
        # Existing AcceptedRecord admission supplies independent custody approval;
        # the receive() observation alone is never admission. Linux does not
        # pretend to read C: or require all retired Drive objects simultaneously.
        # Windows receive/release/restage stream the archive, and every restored
        # dependency gets fresh byte verification. A new storage admission binds
        # any relocation without rewriting the original scientific receipt.
        self.record.read();return product['value']

    def member(self,name,product,receipt_sha256,member):
        """Restore only an admitted dependency capsule, not trajectory journals."""
        safe_name(member)
        if not self.archived(name):return self.resolve(name,product,receipt_sha256)/member
        self.verify_archived(name,product,receipt_sha256)
        entry=self.value['phases'][name];dep=entry.get('dependencies')
        if not isinstance(dep,dict) or member not in dep.get('files',{}):raise ValueError('required dependency transfer unavailable: '+name+'/'+member)
        if any(product['files'].get(n)!=v for n,v in dep['files'].items()):raise ValueError('dependency member not in original phase')
        commit=self.transport.read_json(dep['remote_commit'],deadline=self.deadline,check=self.check)
        m=self.transport.read_json(commit['objects']['OUTPUT_MANIFEST.json'],deadline=self.deadline,check=self.check)
        if (m.get('kind')!='phase_dependency_capsule_v1' or m.get('files')!=dep['files']
            or m.get('original_archive_sha256')!=entry['permanent']['archive_sha256']
            or m.get('phase_receipt_sha256')!=receipt_sha256):raise ValueError('dependency transfer lineage differs')
        key=(name,identity(dep))
        if key not in self.member_paths:
            folder=Path(dep['restore_directory'])
            if folder.exists():
                restored=read(folder/'RESTORED.json')
                if restored.get('complete') is not True or restored.get('commit')!=dep['remote_commit'] or restored.get('files')!=dep['files']:
                    raise ValueError('partial/stale dependency restore preserved')
                if sha(folder/'OUTPUT.zip',self.check)!=m['archive_sha256']:raise ValueError('dependency archive drift')
                verify_archive(folder/'OUTPUT.zip',dep['files'],self.check)
                self.member_paths[key]=folder/'files'
            else:
                self.member_paths[key]=restore_capsule(self.transport,dep['remote_commit'],dep['files'],dep['restore_directory'],
                    deadline=self.deadline,check=self.check,file_limit=dep['file_limit'],byte_limit=dep['byte_limit'])
        if R.inventory(self.member_paths[key],self.check,file_limit=dep['file_limit'],byte_limit=dep['byte_limit'])!=dep['files']:
            raise ValueError('dependency extraction membership drift')
        p=self.member_paths[key]/member;expected=product['files'][member]
        if p.stat().st_size!=expected['bytes'] or sha(p,self.check)!=expected['sha256']:raise ValueError('restored dependency drift')
        return p

    def resolve(self,name,product,receipt_sha256):
        self.record.read();self.check()
        if self.clock()>=self.deadline:raise TimeoutError('phase restore original deadline')
        entry=self.value['phases'].get(name)
        if entry is None or entry['phase_receipt_sha256']!=receipt_sha256:raise ValueError('exact phase storage binding missing/drifted')
        if name not in self.paths:
            folder=Path(entry['restore_directory'])
            if folder.exists():
                # A previously COMMITTED restore can be reused after the caller's
                # accepted process closure. An abandoned partial is never a commit.
                restored=read(folder/'RESTORED.json')
                if restored.get('complete') is not True or restored.get('commit')!=entry['remote_commit'] or restored.get('files')!=product['files']:
                    raise ValueError('existing restore is partial or belongs to different evidence')
                commit=self.transport.read_json(entry['remote_commit'],deadline=self.deadline,check=self.check)
                archive=commit['objects']['OUTPUT.zip']
                if sha(folder/'OUTPUT.zip',self.check)!=archive['sha256'] or (folder/'OUTPUT.zip').stat().st_size!=archive['bytes']:
                    raise ValueError('committed restore archive drifted')
                verify_archive(folder/'OUTPUT.zip',product['files'],self.check)
                self.paths[name]=folder/'files'
            else:
                self.paths[name]=restore_capsule(self.transport,entry['remote_commit'],product['files'],entry['restore_directory'],
                    deadline=self.deadline,check=self.check,file_limit=entry['file_limit'],byte_limit=entry['byte_limit'])
        path=self.paths[name]
        if R.inventory(path,self.check,file_limit=entry['file_limit'],byte_limit=entry['byte_limit'])!=product['files']:
            raise ValueError('restored phase subsequently drifted')
        return path

def restore_worker(transport,permit,worker_root,staging,*,check,clock):
    """Restore the exact closed worker metadata/root before constructing Worker.

    This consumes independent recovery permission, never derives it from a remote
    success flag. The worker's original path, configuration, start/deadline and
    session records remain exact; independent PhaseStore restores missing phases.
    Only freshly downloaded files are moved. Originals/archives remain retained.
    """
    root=Path(worker_root).resolve();staging=Path(staging).resolve()
    grant=require_accepted(permit,kind='main_study_final_worker_restore',bindings=dict(
        worker_root=str(root),staging=str(staging),transport_source_sha256=transport.binding['source_sha256']))
    if grant.get('domain')!='final' or grant.get('previous_owned_closure') is not True or grant.get('active_streams')!=[]:
        raise ValueError('independent closed worker recovery required')
    from .bounded_stage import admitted_deadline
    binding=grant['worker_binding'];deadline=admitted_deadline(binding,grant)
    if clock()>=deadline:raise TimeoutError('original worker deadline expired; no restore clock reset')
    if root.exists():raise FileExistsError('existing worker evidence preserved')
    if root==staging or root.is_relative_to(staging) or staging.is_relative_to(root):raise ValueError('separate restore staging required')
    restored=restore_capsule(transport,grant['remote_commit'],grant['files'],staging,deadline=deadline,check=check,
        file_limit=grant['file_limit'],byte_limit=grant['byte_limit'])
    if sha(restored/'RESULT.json',check)!=require_hash(grant['worker_result_sha256']) or read(restored/'BINDINGS.json')!=binding:
        raise ValueError('restored worker result/configuration/clock differs')
    previous=read(restored/'RESULT.json')
    if previous.get('status') not in ('WAITING_ACCEPTANCE','PAUSED_SESSION'):
        raise ValueError('stopped/failed/completed worker is not a continuation')
    version=restored/'sessions'/require_hash(previous['session_sha256'])/'RESULT.json'
    if not version.is_file() or version.read_bytes()!=(restored/'RESULT.json').read_bytes():raise ValueError('closed session version differs')
    check()
    if clock()>=deadline:raise TimeoutError('original worker recovery deadline')
    if root.exists():raise FileExistsError('worker target appeared; no overwrite')
    os.rename(restored,root)  # same-volume sole-writer publication; cross-volume failure preserves staging
    write(staging/'PROMOTED.json',dict(worker_root=str(root),worker_result_sha256=grant['worker_result_sha256'],
        recovery_permission_sha256=permit.sha256,original_deadline=deadline,scientific_acceptance=False),durable=True)
    return root
