"""Scientific archive/staging custody. No issuer, automatic deletion or retry.

Original phase ZIPs and recovery bytes are immutable. A locally produced receipt
is only an observation until independently bound by the existing record ports.
"""
from pathlib import Path
import os,time,shutil,zipfile
from msdev.records import read,write,sha,safe_name,publish_zip
from operatormark.corpus.state_v2_prerequisites import require_accepted
from .graph import identity
from .session import closure_ok

POLICY='scientific_archive_staging.v1'
RECORDS={'OUTPUT_MANIFEST.json','OUTPUT_INVENTORY.json','PHASE_RECEIPT.json',
         'CLOSURE.json','PHASE_EXPORT_PERMISSION.json','DURABLE_RECEIPT.json'}

def _need(ok,why):
    if not ok:raise ValueError(why)

def phase_metadata(manifest,product,closure,receipt_sha):
    _need(manifest.get('kind')=='closed_phase_capsule_v1' and manifest.get('phase')==product.get('name')
          and manifest.get('files')==product.get('files') and manifest.get('phase_receipt_sha256')==receipt_sha,
          'original phase membership/receipt differs')
    _need(closure_ok(closure) and closure.get('product_readback') is True
          and closure.get('worker_result_sha256')==manifest.get('worker_result_sha256')
          and closure.get('timed_out') is False,'independent closed phase readback required')

def receive(transport,commit_reference,destination,permit,*,check=lambda:None,clock=time.time):
    """Stream one published phase to Windows archive; never expand its payload."""
    from .phase_storage import verify_archive
    dest=Path(destination).resolve()
    g=require_accepted(permit,kind='main_study_final_archive_receive',bindings=dict(
        commit_identity=identity(commit_reference),destination=str(dest),transport_source_sha256=transport.binding['source_sha256']))
    _need(g.get('policy')==POLICY and g.get('new_campaign') is True and g.get('active_streams')==[], 'new closed campaign archive scope required')
    archive_root=Path(g['archive_root']).resolve()
    _need(dest.is_relative_to(archive_root) and dest!=archive_root,'archive outside bound permanent root')
    def bounded():
        check();permit.read()
        if clock()>=g['deadline']:raise TimeoutError('original archive receipt deadline')
    bounded();commit=transport.read_json(commit_reference,deadline=g['deadline'],check=bounded)
    _need(commit.get('kind')=='OperatorMark_remote_export_commit_v1','phase commit required')
    refs=commit['objects'];_need(RECORDS|{'OUTPUT.zip'}<=refs.keys(),'complete recovery records missing')
    _need(all(Path(n).name==n and safe_name(n) for n in refs),'unsafe recovery member')
    total=sum(r['bytes'] for r in refs.values())
    _need(type(g.get('additional_bytes')) is int and g['additional_bytes']>total,'complete archive/metadata capacity envelope required')
    if shutil.disk_usage(dest.parent).free<g['additional_bytes']:raise OSError('permanent archive capacity unavailable')
    dest.mkdir(exist_ok=False)
    # Failed transfers remain in this exclusive directory. No retry/rename over it.
    for n,ref in refs.items():
        bounded()
        with (dest/n).open('xb') as sink:
            transport.verify(ref,deadline=g['deadline'],check=bounded,sink=sink);sink.flush();os.fsync(sink.fileno())
    manifest=read(dest/'OUTPUT_MANIFEST.json');product=read(dest/'PHASE_RECEIPT.json')
    phase_metadata(manifest,product,read(dest/'CLOSURE.json'),sha(dest/'PHASE_RECEIPT.json'))
    _need(read(dest/'OUTPUT_INVENTORY.json')['files']==product['files'],'phase inventory differs')
    _need(sha(dest/'CLOSURE.json')==manifest['closure_sha256'],'original closure identity differs')
    _need(sha(dest/'OUTPUT.zip')==manifest['archive_sha256'] and (dest/'OUTPUT.zip').stat().st_size==manifest['archive_bytes'],'archive identity differs')
    verify_archive(dest/'OUTPUT.zip',product['files'],bounded)
    with (dest/'REMOTE_COMMIT_BYTES.json').open('xb') as sink:
        transport.verify(commit_reference,deadline=g['deadline'],check=bounded,sink=sink);sink.flush();os.fsync(sink.fileno())
    write(dest/'ORIGINAL_REMOTE_COMMIT.json',dict(reference=commit_reference,value=commit),durable=True)
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p,bounded)) for p in dest.iterdir() if p.is_file()}
    receipt=dict(policy=POLICY,campaign_id=g['campaign_id'],worker_binding=product['binding'],phase=product['name'],
        phase_receipt_sha256=sha(dest/'PHASE_RECEIPT.json'),archive_sha256=manifest['archive_sha256'],
        archive_bytes=manifest['archive_bytes'],phase_files=product['files'],recovery_files=files,
        full_member_readback=True,original_records_preserved=True,expanded_payload=False,independent_acceptance=False)
    write(dest/'ARCHIVE_RECEIPT.json',receipt,durable=True);return receipt

def verify_local(folder,expected,*,check=lambda:None):
    from .phase_storage import verify_archive
    folder=Path(folder);_need(not folder.is_symlink(),'archive alias refused')
    _need(sha(folder/'ARCHIVE_RECEIPT.json',check)==expected,'permanent receipt missing/drifted')
    r=read(folder/'ARCHIVE_RECEIPT.json');_need(r.get('policy')==POLICY and r.get('full_member_readback') is True,'archive readback missing')
    _need({p.name for p in folder.iterdir()}==set(r['recovery_files'])|{'ARCHIVE_RECEIPT.json'},'permanent recovery membership differs')
    for n,f in r['recovery_files'].items():
        check();safe_name(n);p=folder/n
        _need(p.is_file() and not p.is_symlink() and p.stat().st_size==f['bytes'] and sha(p,check)==f['sha256'],'permanent archive/recovery corruption: '+n)
    _need(r['recovery_files']['OUTPUT.zip']==dict(bytes=r['archive_bytes'],sha256=r['archive_sha256']),'archive receipt correspondence')
    phase_metadata(read(folder/'OUTPUT_MANIFEST.json'),read(folder/'PHASE_RECEIPT.json'),read(folder/'CLOSURE.json'),r['phase_receipt_sha256'])
    _need(read(folder/'PHASE_RECEIPT.json')['files']==r['phase_files'],'archive phase population differs')
    verify_archive(folder/'OUTPUT.zip',r['phase_files'],check)
    _need(sha(folder/'ARCHIVE_RECEIPT.json',check)==expected,'receipt mutation');return r

def restage(folder,expected,destination,transport,permit,*,members=None,check=lambda:None,clock=time.time):
    """New transfer version, same originals. Optional hash-bound dependency subset.

    A subset is a disposable transfer, NOT a replacement permanent phase archive.
    No evidence is removed from the permanent archive or old transfers.
    """
    from .phase_storage import verify_archive
    folder=Path(folder);dest=Path(destination).resolve();r=verify_local(folder,expected,check=check)
    selected=sorted(r['phase_files'] if members is None else members)
    _need(selected and len(selected)==len(set(selected)) and set(selected)<=r['phase_files'].keys(),'missing/duplicate dependency member')
    g=require_accepted(permit,kind='main_study_final_archive_restage',bindings=dict(archive_receipt_sha256=expected,
        archive_sha256=r['archive_sha256'],destination=str(dest),members=selected,transport_source_sha256=transport.binding['source_sha256']))
    _need(g.get('policy')==POLICY and g.get('campaign_id')==r['campaign_id'],'restage campaign mismatch')
    def bounded():
        check();permit.read()
        if clock()>=g['deadline']:raise TimeoutError('original restaging deadline')
    bound=2*sum(r['phase_files'][n]['bytes'] for n in selected)+sum(200+2*len(n.encode()) for n in selected)+1024
    _need(type(g.get('additional_bytes')) is int and g['additional_bytes']>bound,'bounded dependency staging capacity required')
    if shutil.disk_usage(dest.parent).free<g['additional_bytes']:raise OSError('restage capacity')
    bounded();dest.mkdir(exist_ok=False)
    if members is None:
        archive=folder/'OUTPUT.zip'
        for n in RECORDS:shutil.copyfile(folder/n,dest/n)
    else:
        payload=dest/'payload';payload.mkdir()
        with zipfile.ZipFile(folder/'OUTPUT.zip') as z:
            for n in selected:
                bounded();p=payload/safe_name(n);p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(n) as source,p.open('xb') as sink:shutil.copyfileobj(source,sink,1<<20);sink.flush();os.fsync(sink.fileno())
                _need(sha(p,bounded)==r['phase_files'][n]['sha256'] and p.stat().st_size==r['phase_files'][n]['bytes'],'dependency extraction differs')
        archive=dest/'OUTPUT.zip';publish_zip(archive,payload,selected,bounded)
        files={n:r['phase_files'][n] for n in selected};verify_archive(archive,files,bounded)
        write(dest/'OUTPUT_MANIFEST.json',dict(kind='phase_dependency_capsule_v1',files=files,
            archive_sha256=sha(archive,bounded),archive_bytes=archive.stat().st_size,
            original_archive_sha256=r['archive_sha256'],phase_receipt_sha256=r['phase_receipt_sha256']),durable=True)
        write(dest/'OUTPUT_INVENTORY.json',dict(files=files),durable=True)
        for n in RECORDS-{'OUTPUT_MANIFEST.json','OUTPUT_INVENTORY.json'}:shutil.copyfile(folder/n,dest/n)
    # Original metadata still names original objects; this additive record is
    # the live content-bound locator. It never rewrites historical receipts.
    write(dest/'ARCHIVE_RECEIPT.json',r,durable=True)
    write(dest/'RELOCATION.json',dict(original_archive_sha256=r['archive_sha256'],archive_receipt_sha256=expected,
        phase_receipt_sha256=r['phase_receipt_sha256'],members=selected,scientific_acceptance=False),durable=True)
    ref=transport.upload_verified(archive,deadline=g['deadline'],check=bounded)
    return transport.commit_records(dest,sorted(RECORDS|{'ARCHIVE_RECEIPT.json','RELOCATION.json'}),deadline=g['deadline'],check=bounded,archive=ref)

def release_plan(folder,expected,permission,closure,*,check=lambda:None,clock=time.time):
    """Prepare exact redundant-copy release. No deletion/network call occurs.

    A separately authorized executor must re-run this gate immediately before
    each mutation; a saved plan alone is not permission. Unique failures cannot
    enter the source inventory unless represented in permanent recovery bytes.
    """
    from .readback import inventory
    r=verify_local(folder,expected,check=check)
    g=require_accepted(permission,kind='main_study_final_copy_release',bindings=dict(
        archive_receipt_sha256=expected,closure_sha256=closure.sha256,campaign_id=r['campaign_id']))
    _need(g.get('policy')==POLICY and g.get('new_campaign_only') is True and g.get('active_streams')==[],'new redundant closed copies only')
    _need(closure_ok(closure.read()) and not closure.read().get('timed_out'),'owned closed sources required')
    _need(clock()<g['deadline'],'release deadline expired')
    # The latest durable recovery catalog must no longer depend on the retired
    # objects. It is written separately on the permanent Windows volume.
    catalog=Path(g['recovery_catalog']);_need(sha(catalog,check)==g['recovery_catalog_sha256'],'latest recovery catalog missing/stale')
    index=read(catalog)
    _need(index.get('policy')==POLICY and index.get('campaign_id')==r['campaign_id']
          and index.get('archive_receipts',{}).get(expected)==str(Path(folder).resolve())
          and index.get('retired_objects')==g['remote_objects'],'recovery catalog does not replace retiring locators')
    _need(not set(identity(v) for v in index.get('active_objects',[]))&set(identity(v) for v in g['remote_objects']),'live recovery reference would be deleted')
    outputs=[];permanent=Path(folder).resolve();catalog=catalog.resolve()
    for copy in g['local_copies']:
        raw=Path(copy['root']);p=raw.resolve();scope=Path(g['new_campaign_root']).resolve()
        _unalias(raw)
        _need(p!=scope and p.is_relative_to(scope)
              and not (p==permanent or p.is_relative_to(permanent) or permanent.is_relative_to(p) or catalog.is_relative_to(p)), 'protected/outside new-campaign copy')
        actual=inventory(p,check,file_limit=copy['file_limit'],byte_limit=copy['byte_limit'])
        _need(actual==copy['files'],'redundant source changed/extra failed artifact')
        originals={v['sha256']:v['bytes'] for v in r['phase_files'].values()}
        originals.update({v['sha256']:v['bytes'] for v in r['recovery_files'].values()})
        _need(all(originals.get(v['sha256'])==v['bytes'] for v in actual.values()),'unique source/failure not in permanent archive')
        outputs.append(dict(root=str(p),files=actual))
    known={(v['sha256'],v['bytes']) for v in r['recovery_files'].values()}
    known.add((r['archive_sha256'],r['archive_bytes']))
    _need(all((v['sha256'],v['bytes']) in known for v in g['remote_objects']),'unique/unpreserved remote object')
    permission.read();_need(sha(catalog,check)==g['recovery_catalog_sha256'],'recovery catalog changed during release review')
    return dict(eligible=True,executed=False,permission_sha256=permission.sha256,archive_receipt_sha256=expected,
        recovery_catalog_sha256=g['recovery_catalog_sha256'],local_copies=outputs,remote_objects=g['remote_objects'],deadline=g['deadline'])

def _unalias(path):
    import stat
    p=Path(path)
    for x in (p,*p.parents):
        s=x.lstat()
        _need(not (stat.S_ISLNK(s.st_mode) or getattr(s,'st_file_attributes',0)&0x400),
              'release symlink/junction/reparse point refused')

def _remove_copy(copy,check):
    """Closed, exclusively held staging only. No recursive shell deletion."""
    from .readback import inventory
    root=Path(copy['root']);remaining=dict(copy['files'])
    for n in sorted(remaining):
        check();_unalias(root)
        _need(inventory(root,check,file_limit=len(copy['files']),
                        byte_limit=sum(v['bytes'] for v in copy['files'].values()))==remaining,
              'release membership changed')
        p=root/safe_name(n);_unalias(p)
        _need(p.is_file() and sha(p,check)==remaining[n]['sha256'],'release member drift')
        p.unlink();del remaining[n]
    # Empty directories only. An unexpected file/alias makes rmdir fail closed.
    for p in sorted((x for x in root.rglob('*') if x.is_dir()),key=lambda p:len(p.parts),reverse=True):
        check();_unalias(p);p.rmdir()
    check();_unalias(root);root.rmdir()

def execute_release(folder,expected,permission,closure,output,*,transport=None,
                    execute=False,check=lambda:None,clock=time.time):
    """Explicit future authority, fresh gate, durable one-shot action receipts.

    Preparation/default is dry-run. The accepted permission must bind exclusive
    custody and the exact output path; it is not minted here. Run under the
    existing owned supervisor. No retry, historical cleanup or quota credit.
    Concurrent hostile mutation is not defeated by membership checks.
    """
    plan=release_plan(folder,expected,permission,closure,check=check,clock=clock)
    if not execute:return plan
    g=permission.read();out=Path(output).resolve();scope=Path(g['new_campaign_root']).resolve()
    _need(g.get('execution_authorized') is True and g.get('exclusive_custody') is True
          and g.get('release_output')==str(out),'explicit bound release execution/custody required')
    _need(out!=scope and out.is_relative_to(scope) and not out.exists(),'fresh owned release output required')
    protected=[Path(folder).resolve(),Path(g['recovery_catalog']).resolve()]+[Path(c['root']) for c in plan['local_copies']]
    _need(all(not (out==p or out.is_relative_to(p) or p.is_relative_to(out)) for p in protected),'release records overlap protected evidence')
    if plan['remote_objects']:
        _need(transport is not None and g.get('transport_source_sha256')==transport.binding['source_sha256']
              and g.get('new_campaign_drive_folder_id')==transport.folder_id,'bound new campaign Drive provider required')
    _unalias(out.parent)
    def bounded():
        check();_need(permission.read()==g,'release authority drift')
        _need(clock()<plan['deadline'],'release deadline expired')
        _need(sha(g['recovery_catalog'],check)==plan['recovery_catalog_sha256'],'recovery catalog stale')
        _need(sha(Path(folder)/'ARCHIVE_RECEIPT.json',check)==expected,'archive receipt drift')
    bounded();out.mkdir(exist_ok=False);write(out/'CONSUMED.json',plan,durable=True)
    done=[]
    actions=[('local',c) for c in plan['local_copies']]+[('drive',r) for r in plan['remote_objects']]
    try:
        for i,(kind,item) in enumerate(actions):
            bounded();verify_local(folder,expected,check=bounded)
            write(out/f'{i:04d}_INTENT.json',dict(kind=kind,item=item),durable=True)
            if kind=='local':_remove_copy(item,bounded)
            else:transport.remove_redundant(item,deadline=plan['deadline'],check=bounded)
            write(out/f'{i:04d}_DONE.json',dict(kind=kind,item=item),durable=True);done.append(i)
        result=dict(executed=True,complete=True,actions=done,permission_sha256=permission.sha256,
                    archive_receipt_sha256=expected,capacity_credit_bytes=None,
                    note='observe actual free capacity after closure; logical deletion is not quota release')
        write(out/'RESULT.json',result,durable=True);return result
    except BaseException as exc:
        write(out/'FAILURE.json',dict(error=repr(exc),completed_actions=done,complete=False,
              no_retry=True,capacity_credit_bytes=0),durable=True);raise
