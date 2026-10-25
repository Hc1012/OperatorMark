"""Private complete-return publication after owned closure; no acceptance issuer."""
from pathlib import Path
import hashlib
import zipfile
import time
import shutil
import json
import os
import traceback
from operatormark.corpus.state_v2_prerequisites import require_accepted
from msdev.records import read,write,publish_zip,sha
from . import readback as R
from .session import closure_ok
from .graph import identity
from .marking import finite, identity as require_hash


def publish(root,destination,closure,permit,check=lambda:None,*,clock=time.time,phase_store=None):
    dest=Path(destination);existed=dest.exists()
    try:return _publish(root,destination,closure,permit,check,clock,phase_store)
    except BaseException as exc:
        if not existed and dest.is_dir():
            try:write(dest/f'EXPORT_FAILURE_{time.time_ns()}.json',dict(error=repr(exc),complete=False,originals_preserved=True))
            except BaseException as reporting:exc.add_note('export failure receipt unavailable: '+repr(reporting))
        raise


def _publish(root,destination,closure,permit,check,clock,phase_store=None):
    root=Path(root).resolve();dest=Path(destination).resolve()
    if dest==root or dest.is_relative_to(root) or root.is_relative_to(dest):raise ValueError('export must be separate from worker evidence')
    r=closure.read()
    if not closure_ok(r) or r.get('command') is None:raise ValueError('owned process/stream closure unresolved')
    if r.get('product_readback') is not True or not isinstance(r.get('supervisor_record_sha256'),str):
        raise ValueError('independent post-closure product readback missing')
    grant=require_accepted(permit,kind='main_study_final_private_export',
        bindings=dict(worker_root=str(root),closure_sha256=closure.sha256,destination=str(dest)))
    if grant.get('domain')!='final' or grant.get('private_export') is not True:
        raise ValueError('explicit private export acceptance required')
    if (not finite(grant.get('deadline')) or not finite(r.get('finalization_deadline'))
            or grant['deadline']>r['finalization_deadline']):raise ValueError('export exceeds accepted finalization window')
    external_check=check
    def check():
        external_check()
        if clock()>=grant['deadline']:raise TimeoutError('bound export deadline expired')
    check()
    result=read(root/'RESULT.json')
    if result.get('selected_complete') is True or result.get('core_complete') is True:
        from .final_contracts import completion
        completion(read(root/'GRAPH.json'),result['products'])
    if r.get('worker_result_sha256')!=sha(root/'RESULT.json'):
        raise ValueError('closure does not bind exact worker result')
    phase_capsules={}
    expected_store=result.get('phase_storage_sha256')
    if expected_store!=r.get('phase_storage_sha256') or expected_store!=(None if phase_store is None else phase_store.record.sha256):
        raise ValueError('export/closure phase storage binding differs')
    for name in result.get('products',[]):
        if phase_store is not None and phase_store.archived(name):
            receipt=root/'products'/(identity(name)+'.json');product=read(receipt)
            if sha(receipt)!=result['product_receipts'][name]:raise ValueError('archived phase receipt differs')
            phase_store.verify_archived(name,product,sha(receipt))
            source=root/'work'/name
            if source.exists() and R.inventory(source,check,file_limit=grant['file_limit'],byte_limit=grant['byte_limit'])!=product['files']:
                raise ValueError('local archived phase changed before export')
            phase_capsules[name]=phase_store.value['phases'][name]
            continue
        if (root/'work'/name).exists():continue
        if phase_store is None:raise ValueError('export missing phase without accepted custody')
        receipt=root/'products'/(identity(name)+'.json');product=read(receipt)
        if sha(receipt)!=result['product_receipts'][name]:raise ValueError('external phase receipt differs')
        restored=phase_store.resolve(name,product,sha(receipt))
        if R.inventory(restored,check,file_limit=grant['file_limit'],byte_limit=grant['byte_limit'])!=product['files']:
            raise ValueError('external phase export independent readback differs')
        phase_capsules[name]=phase_store.value['phases'][name]
    def retained_inventory():
        inventory=R.inventory(root,check,file_limit=grant.get('file_limit'),byte_limit=grant.get('byte_limit'))
        # Explicit permanent custody, never silently omit an unarchived phase,
        # unique failure or runtime/session metadata from the recovery export.
        prefixes=['work/'+n+'/' for n in phase_capsules if phase_store.archived(n)]
        for name in phase_capsules:
            if not phase_store.archived(name):continue
            prefix='work/'+name+'/'
            local={n[len(prefix):]:v for n,v in inventory.items() if n.startswith(prefix)}
            if (root/'work'/name).exists() and local!=read(root/'products'/(identity(name)+'.json'))['files']:
                raise ValueError('local archived phase mutated during export')
        return {n:v for n,v in inventory.items() if not any(n.startswith(prefix) for prefix in prefixes)}
    files=retained_inventory()
    if grant.get('membership_sha256')!=identity(files):raise ValueError('complete export membership not approved')
    # Free bytes are incremental: the retained source is already occupying its
    # volume. Include archive headers and both metadata records; no compression
    # credit, deletion, or implicit second copy is assumed.
    payload=sum(v['bytes'] for v in files.values())
    archive_bound=payload+sum(200+2*len(n.encode('utf-8')) for n in files)+1024
    manifest=dict(files=files,worker_result_sha256=sha(root/'RESULT.json'),closure_sha256=closure.sha256,
        core_complete=result.get('core_complete') is True and r.get('returncode')==0 and r.get('timed_out') is False,
        complete_programme=False,
        worker_status=result.get('status'),independent_acceptance=False,phase_capsules=phase_capsules,
        phase_storage_sha256=expected_store)
    # Exact record domains include potentially large external phase-capsule
    # mappings. Multiplying only the files dictionary misses those copies.
    final_bound=dict(manifest,archive_sha256='0'*64,archive_bytes=archive_bound,
        exact_membership=True,full_member_readback=True,unchanged_source=True,owned_closure=True,
        durable_remote=False,disconnect_permitted=False)
    metadata_bound=sum(len((json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').replace('\n',os.linesep).encode('utf-8'))
        for v in (manifest,final_bound))
    if shutil.disk_usage(dest.parent).free<archive_bound+metadata_bound:
        raise OSError('local export capacity unavailable; retained source unchanged')
    dest.mkdir(exist_ok=False)
    write(dest/'OUTPUT_INVENTORY.json',manifest,durable=True)
    archive=dest/'OUTPUT.zip'
    publish_zip(archive,root,sorted(files),check)
    # Full independent archive readback, bounded chunks, then a fresh source
    # inventory. Partial/failed files are included, not silently cleaned away.
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(files):raise ValueError('archive membership differs')
        for name in names:
            check();h=hashlib.sha256();size=0
            with z.open(name) as f:
                while data:=f.read(1<<20):check();h.update(data);size+=len(data)
            if files[name]!=dict(bytes=size,sha256=h.hexdigest()):raise ValueError('archive member differs: '+name)
    after=retained_inventory()
    if after!=files:
        write(dest/'SOURCE_DIFFERENCES.json',dict(added=sorted(set(after)-set(files)),missing=sorted(set(files)-set(after)),
            changed=sorted(n for n in set(files)&set(after) if files[n]!=after[n])))
        raise ValueError('source membership/bytes mutated during export')
    value=dict(manifest,archive_sha256=sha(archive,check),archive_bytes=archive.stat().st_size,
        exact_membership=True,full_member_readback=True,unchanged_source=True,owned_closure=True,
        durable_remote=False,disconnect_permitted=False)
    write(dest/'OUTPUT_MANIFEST.json',value,durable=True);return value


class RemoteTransfer:
    """External authenticated object-service adapter, NEVER a mounted-path copy.

    The independently supplied transfer permit binds this implementation's hash,
    destination, writer and independent reader identities. Implementations must
    enforce the supplied deadline on network calls, retain partial remote objects
    and request a fresh version-bound stream bypassing the uploader's cache.
    No default provider, credentials, quota assertion or acceptance is issued here.
    """
    def put(self,archive,*,deadline,check):raise NotImplementedError('remote upload adapter required')
    def open_remote(self,object_id,version,*,deadline,check):
        raise NotImplementedError('independent remote byte-stream adapter required')
    def commit_records(self,folder,names,*,deadline,check,archive=None):
        raise NotImplementedError('remote manifest/inventory/receipt commit required')


def publish_durable(root,destination,closure,permit,transfer_permit,transport,check=lambda:None,*,clock=time.time,phase_store=None):
    """Closed-worker-stage -> local ZIP/readback -> remote upload/full GET.

    A stage is a fully closed worker session, not a live Worker.phase event or
    an arbitrary cut through API/noise/RNG state. No disconnection or continuation
    is triggered here. Missing remote evidence leaves local/partial bytes intact.
    """
    dest=Path(destination).resolve();root=Path(root).resolve();existed=dest.exists()
    try:
        grant=require_accepted(transfer_permit,kind='main_study_final_stage_transfer',bindings=dict(
            worker_root=str(root),local_export_permission_sha256=permit.sha256,closure_sha256=closure.sha256,
            destination=str(dest)))
        local=permit.read();closed=closure.read()
        if (grant.get('domain')!='final' or grant.get('closed_stage_only') is not True
            or grant.get('active_streams')!=[] or grant.get('copy_graph')!=['retained_local_source','local_archive','remote_object']
            or grant.get('local_storage') is not True):raise ValueError('closed-stage/local-storage/copy-graph contract required')
        local_root=Path(grant['local_storage_root']).resolve()
        if not root.is_relative_to(local_root) or not dest.is_relative_to(local_root):
            raise ValueError('source/export outside independently bound local volume')
        if (not finite(grant.get('deadline')) or not finite(local.get('deadline'))
            or grant['deadline']>local['deadline'] or grant['deadline']>closed.get('finalization_deadline',0)):
            raise ValueError('transfer extends existing finalization deadline')
        def bounded():
            check()
            if clock()>=grant['deadline']:raise TimeoutError('original durable-export deadline')
        bounded()
        provider=grant.get('provider',{})
        if (provider.get('readback_kind')!='authenticated_remote_object_stream' or provider.get('cache_bypass') is not True
            or not provider.get('destination_id') or not provider.get('writer_id') or not provider.get('reader_id')
            or provider['reader_id']==provider['writer_id']):raise ValueError('independent remote reader required; mount/cache reread refused')
        require_hash(provider.get('source_sha256'))
        import inspect
        if (type(transport) is RemoteTransfer or transport.binding!=provider
            or sha(inspect.getfile(type(transport)))!=provider['source_sha256']):raise ValueError('bound remote implementation differs')
        # Capacity is an independently supplied recent destination observation,
        # not shutil.disk_usage on a Drive mount or a local writer success flag.
        quota=grant.get('remote_capacity',{})
        if (quota.get('destination_id')!=provider['destination_id'] or type(quota.get('available_bytes')) is not int
            or quota['available_bytes']<0 or not finite(quota.get('valid_until')) or quota['valid_until']<grant['deadline']):
            raise ValueError('bound remote capacity unavailable')
        require_hash(quota.get('observation_sha256'))
        if dest.exists():raise FileExistsError('existing export preserved; no automatic retry')
        value=publish(root,dest,closure,permit,bounded,clock=clock,phase_store=phase_store)
        archive=dest/'OUTPUT.zip';metadata=dest/'OUTPUT_MANIFEST.json'
        metadata_hash=sha(metadata,bounded)
        if value['archive_bytes']>quota['available_bytes']:raise OSError('remote capacity cannot retain complete archive')
        write(dest/'TRANSFER_INTENT.json',dict(provider=provider,archive_sha256=value['archive_sha256'],
            archive_bytes=value['archive_bytes'],transfer_permit_sha256=transfer_permit.sha256,
            original_deadline=grant['deadline']),durable=True)
        bounded();uploaded=transport.put(archive,deadline=grant['deadline'],check=bounded);bounded()
        write(dest/'UPLOAD_RESPONSE.json',uploaded,durable=True)
        if (not isinstance(uploaded,dict) or not isinstance(uploaded.get('object_id'),str) or not uploaded['object_id']
            or not isinstance(uploaded.get('version'),str) or not uploaded['version']
            or uploaded.get('bytes')!=value['archive_bytes']):raise ValueError('remote immutable object/version/size missing')
        digest=hashlib.sha256();count=0
        with transport.open_remote(uploaded['object_id'],uploaded['version'],deadline=grant['deadline'],check=bounded) as stream:
            while True:
                bounded();data=stream.read(1<<20)
                if not data:break
                count+=len(data)
                if count>value['archive_bytes']:raise ValueError('remote object grew or wrong object returned')
                digest.update(data)
        bounded()
        if count!=value['archive_bytes'] or digest.hexdigest()!=value['archive_sha256']:
            raise ValueError('independent remote readback incomplete/corrupt')
        if sha(metadata,bounded)!=metadata_hash or sha(archive,bounded)!=value['archive_sha256']:
            raise ValueError('local export drift during transfer')
        result=dict(durable_remote=False,archive_remote_verified=True,disconnect_permitted=False,scientific_acceptance=False,
            worker_result_sha256=value['worker_result_sha256'],closure_sha256=closure.sha256,
            local_manifest_sha256=metadata_hash,archive_sha256=value['archive_sha256'],archive_bytes=count,
            remote_object=uploaded,provider=provider,transfer_permit_sha256=transfer_permit.sha256,
            original_deadline=grant['deadline'],full_remote_readback=True,active_streams=[],originals_preserved=True)
        bounded();write(dest/'DURABLE_RECEIPT.json',result,durable=True)
        # The originating runtime may disappear: remote archive bytes alone are
        # insufficient without their inventory, manifest, receipt and closure.
        for name,record in [('CLOSURE.json',closure),('EXPORT_PERMISSION.json',permit),('TRANSFER_PERMISSION.json',transfer_permit)]:
            record.read()
            with (dest/name).open('xb') as stream:
                import os
                stream.write(record.expected);stream.flush();os.fsync(stream.fileno())
        uploaded=dict(uploaded,sha256=value['archive_sha256'])
        committed=transport.commit_records(dest,['OUTPUT_INVENTORY.json','OUTPUT_MANIFEST.json','DURABLE_RECEIPT.json',
            'CLOSURE.json','EXPORT_PERMISSION.json','TRANSFER_PERMISSION.json'],deadline=grant['deadline'],check=bounded,archive=uploaded)
        bounded()
        if committed.get('remote_records_verified') is not True or committed.get('disconnect_permitted') is not True:
            raise ValueError('remote recovery records unverified; disconnection refused')
        return dict(result,durable_remote=True,disconnect_permitted=True,remote_commit=committed)
    except BaseException as exc:
        if not existed and dest.is_dir():
            try:write(dest/f'DURABILITY_FAILURE_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),
                durable_remote=False,disconnect_permitted=False,originals_preserved=True),durable=True)
            except BaseException as reporting:exc.add_note('durability failure record unavailable: '+repr(reporting))
        raise


def require_durable_boundary(record,previous_result_sha256,closure_sha256,original_deadline):
    """Consume an independent custody acceptance; a producer receipt is not it."""
    require_hash(previous_result_sha256);require_hash(closure_sha256)
    if isinstance(original_deadline,dict):
        require_hash(original_deadline.get('campaign_sha256'))
        time_binding=dict(worker_binding=original_deadline)
    else:
        if not finite(original_deadline):raise ValueError('original deadline required')
        time_binding=dict(original_deadline=original_deadline)
    value=require_accepted(record,kind='main_study_final_remote_custody',bindings=dict(
        previous_result_sha256=previous_result_sha256,closure_sha256=closure_sha256,**time_binding))
    if (value.get('domain')!='final' or value.get('durable_remote') is not True or value.get('full_remote_readback') is not True
        or value.get('remote_records_verified') is not True
        or value.get('active_streams')!=[] or value.get('closed_stage_only') is not True):
        raise ValueError('durable closed-stage custody required; mid-stream continuation refused')
    for key in ('archive_sha256','durable_receipt_sha256','membership_sha256'):require_hash(value.get(key))
    commit=value.get('remote_commit',{})
    require_hash(commit.get('sha256'))
    if (not isinstance(commit.get('object_id'),str) or not commit['object_id'] or not isinstance(commit.get('version'),str)
        or not commit['version'] or type(commit.get('bytes')) is not int or commit['bytes']<=0):
        raise ValueError('accepted remote recovery commit identity required')
    return value
