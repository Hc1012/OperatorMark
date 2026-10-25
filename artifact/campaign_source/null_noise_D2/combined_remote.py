"""Pinned input reuse and bounded Drive saves using the existing transport."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,os,stat,sys,time,zipfile
from urllib.parse import urlencode
def need(ok,msg):
    if not ok:raise ValueError(msg)
def safe(n):
    p=PurePosixPath(n)
    need(bool(p.parts) and p.as_posix()==n and not p.is_absolute() and '..' not in p.parts and ':' not in n and '\\' not in n,'Unsafe member')
    return p
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def pack(path,entries,limit,check):
    need(len(entries)<=24000 and sum(p.stat().st_size for p in entries.values())<=limit,'Return fixed size/member bound exceeded')
    manifest={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in entries.items()}
    with Path(path).open('xb') as sink:
        with zipfile.ZipFile(sink,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for n,p in sorted(entries.items()):
                check();safe(n);need(not p.is_symlink(),'Return symlink refused')
                h=hashlib.sha256();before=p.stat()
                with p.open('rb') as src,z.open(n,'w',force_zip64=True) as dst:
                    while b:=src.read(1024**2):check();dst.write(b);h.update(b)
                after=p.stat();need((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and h.hexdigest()==manifest[n]['sha256'],'Return changed while packaging')
            z.writestr('RETURN_MANIFEST.json',json.dumps(dict(files=manifest,formal_custody_assurance=False),sort_keys=True,indent=2)+'\n')
        sink.flush();os.fsync(sink.fileno())
    need(Path(path).stat().st_size<=limit,'Compressed return exceeds bound')
    return manifest

def extract(zpath,job,mapping,check):
    with zipfile.ZipFile(zpath) as z:
        names=z.namelist();need(len(names)==len(set(names)),'Duplicate remote archive entries')
        for n,e in mapping.items():
            check();safe(n);m=e.get('member',n);safe(m);i=z.getinfo(m)
            need(not i.is_dir() and not stat.S_ISLNK(i.external_attr>>16) and not i.flag_bits&1 and i.file_size==e['bytes'],'Input member differs: '+m)
            p=job/n;p.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
            with z.open(i) as src,p.open('xb') as dst:
                while b:=src.read(1024**2):check();dst.write(b);h.update(b)
                dst.flush();os.fsync(dst.fileno())
            need(h.hexdigest()==e['sha256'],'Input checksum differs: '+n)

def entries_for(job,p,bridge_saving=False):
    entries={n:job/n for n in p['files'] if (job/n).is_file() and Path(n).suffix in ('.json','.jsonl','.py','.txt','.md')}
    for folder in ('output','compatibility_output','observations','supervision','transport_input','transport_bridge-save','transport_rolling','owner_capsules','reference','reused'):
        root=job/folder
        if root.exists():
            for path in sorted(root.rglob('*')):
                n=path.relative_to(job).as_posix()
                if n.startswith(('supervision/save/','supervision/closing_record/')) or (bridge_saving and n.startswith('supervision/bridge_save/')) or path.suffix=='.zip':continue
                if path.is_file():entries[n]=path
    for path in job.iterdir():
        if path.is_file() and path.suffix in ('.json','.jsonl','.py'):entries[path.name]=path
    return entries

def uploaded_control(job,p):
    """Check the notebook's frozen upload identity before any Drive operation."""
    want=read(job/'INPUT_ARCHIVE_EXPECTATION.json')
    need(type(want) is dict and set(want)=={'name','bytes','sha256'},'Control expectation fields differ')
    name=want['name']
    need(type(name) is str and len(safe(name).parts)==1 and name.endswith('_CONTROL.zip'),
         'Control archive name must be a single safe filename')
    need(name==p['control_archive_name'],'Control archive name differs from the frozen policy')
    need(type(want['bytes']) is int and 0<want['bytes']<=128*1024**2,
         'Control archive byte count is invalid')
    digest=want['sha256']
    need(type(digest) is str and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),
         'Control archive checksum is invalid')
    control=job.parent.parent/name
    need(not control.is_symlink() and control.is_file(),'Verified control archive missing at '+str(control))
    need(control.stat().st_size==want['bytes'] and sha(control)==digest,
         'Control archive bytes differ at '+str(control))
    return control,want

def transfer(job,action,deadline):
    p=read(job/'D2_COMBINED_POLICY.json');act=read(job/'ACTIVATION.json')
    need(time.time()<deadline<=act['deadline'],'Original transport deadline required')
    def check():
        if time.time()>=deadline:raise TimeoutError('Original transfer deadline reached; no retry')
    if action=='input':
        control,want=uploaded_control(job,p)
    from msfinal.drive_transport import GoogleDriveTransfer,RefreshingMetadataToken,API
    complete_entries=None;full=None
    if action=='save':
        complete_entries=entries_for(job,p);full=job/('D2_COMBINED_FULL_'+act['attempt']+'.zip')
        pack(full,complete_entries,p['return_bound_bytes'],check)
    http=GoogleDriveTransfer(p['drive_folder'],RefreshingMetadataToken('metadata.google.internal'),job/('transport_'+action),
        byte_limit=(p['input_transport_bound_bytes'] if action=='input' else 2*p['return_bound_bytes']+2*p['review_bound_bytes']+4*1024**2),metadata_byte_limit=4*1024**2)
    quota=http.quota(deadline=deadline,check=check)
    permissions=http.json('GET',API+'files/'+p['drive_folder']+'/permissions?'+urlencode(dict(fields='nextPageToken,permissions(id,type,role,deleted)',pageSize=100)),deadline=deadline,check=check)
    need(not permissions.get('nextPageToken') and permissions.get('permissions') and all(v.get('type')=='user' and v.get('role')=='owner' and not v.get('deleted') for v in permissions['permissions']),'Private owner-only destination required')
    put(job/(action.upper()+'_DRIVE_OBSERVATION.json'),dict(quota=quota,permissions=permissions,observed_at=time.time(),formal_custody_assurance=False))
    if action=='input':
        need(quota['available_bytes']>=p['minimum_drive_free_bytes'],'At least 8 GiB free in the existing Drive is required for this bounded batch')
        # Reuse the exact previously verified Drive object and revision.
        final_ref=p['final_material_reference']
        need(final_ref['keepForever'] is True and final_ref['folder_id']==p['drive_folder'],'Retained final material reference differs')
        marker_name='D2_COMBINED_ONCE_'+sha(job/'D2_COMBINED_POLICY.json')+'.json'
        listing=http.json('GET',API+'files?'+urlencode(dict(q="'%s' in parents and trashed=false and name='%s'"%(p['drive_folder'],marker_name),fields='nextPageToken,files(id,name)',pageSize=100)),deadline=deadline,check=check)
        need(not listing.get('nextPageToken') and listing.get('files')==[],'This exact evaluation batch already has an attempt; no replay or clock reset')
        marker=job/marker_name;put(marker,dict(activation=act,policy_sha256=sha(job/'D2_COMBINED_POLICY.json'),scope='one combined attempt: fixed20 reference checks followed only on scoped mechanical acceptance by the fixed200 raw960 diagnostics'))
        put(job/'ONCE_MARKER_SAVED.json',dict(reference=http.upload_verified(marker,deadline=deadline,check=check),formal_exclusive_custody=False))
        sources=[('final materials',final_ref,p['final_material_files']),('clean owners',p['original_reference'],p['original_files'])]
        sources += [('checkpoint_'+role,group['reference'],group['files']) for role,group in p['additional_sources'].items()]
        sources += [('compatibility reference',p['reference_source']['reference'],p['reference_files'])]
        need(sum(ref['bytes'] for _,ref,_ in sources)==p['exact_input_download_bytes']<=p['input_download_bound_bytes'],'Fixed input download allowance differs')
        for role,ref,mapping in sources:
            path=job/('INPUT_'+role.replace(' ','_')+'.zip')
            print('Fetching '+role+' from Drive ('+str(round(ref['bytes']/1024**2))+' MiB); no laptop transfer.',flush=True)
            with path.open('xb') as sink:
                http.verify(ref,deadline=deadline,check=check,sink=sink);sink.flush();os.fsync(sink.fileno())
            extract(path,job,mapping,check)
        for n,e in p['aliases'].items():
            check();source=job/e['source'];target=job/n;target.parent.mkdir(parents=True,exist_ok=True)
            need(source.stat().st_size==e['expected']['bytes'] and sha(source)==e['expected']['sha256'],'Alias source differs: '+n)
            with source.open('rb') as src,target.open('xb') as dst:
                while b:=src.read(1024**2):check();dst.write(b)
                dst.flush();os.fsync(dst.fileno())
            need(sha(target)==e['expected']['sha256'],'Alias copy differs: '+n)
        final_control,final_want=uploaded_control(job,p)
        need(final_control==control and final_want==want,'Control expectation changed during input transfer')
        put(job/'CONTROL_SAVED.json',dict(reference=http.upload_verified(control,deadline=deadline,check=check),policy_sha256=sha(job/'D2_COMBINED_POLICY.json')))
        put(job/'INPUT_READBACK.json',dict(complete=True,final_reference=final_ref,original_reference=p['original_reference'],marking_reference=p['marking_reference'],
            final_files=len(p['final_material_files']),original_files=len(p['original_files']),marked_files=len(p['marked_files']),aliases=len(p['aliases']),
            additional_sources={role:group['reference'] for role,group in p['additional_sources'].items()},
            exact_input_download_bytes=p['exact_input_download_bytes'],observed_location_is_not_expected_content=True,independently_frozen_SHA256_checked=True,new_scientific_calls=0))
        print('Existing inputs, all200 raw960 checkpoints, saved comparison references and the reused attempt '+p['reused_bridge']['attempt']+' bridge evidence verified. The control package is saved in Drive.',flush=True);return
    if action=='bridge-save':
        need((job/'BRIDGE_VERIFICATION.json').is_file(), 'Independent comparison readback required before bridge save')
        entries=entries_for(job,p,bridge_saving=True)
        full=job/('D2_COMBINED_BRIDGE_'+act['attempt']+'.zip')
        pack(full,entries,p['return_bound_bytes'],check)
        need(quota['available_bytes']>=full.stat().st_size+p['return_bound_bytes'], 'Bridge and final return space insufficient')
        reference=http.upload_verified(full,deadline=deadline,check=check)
        receipt=http.log/f'{http.sequence:06d}_readback.json'
        observed=read(receipt)
        need(observed['kind']=='readback' and observed['object_id']==reference['object_id'] and
             observed['revision']==reference['version'] and observed['bytes']==reference['bytes'] and
             observed['sha256']==reference['sha256'],'Bridge remote readback event differs')
        with zipfile.ZipFile(full) as z:manifest_sha=hashlib.sha256(z.read('RETURN_MANIFEST.json')).hexdigest()
        put(job/'BRIDGE_DURABLE.json',dict(reference=reference,local_archive=full.name,
            bytes=full.stat().st_size,sha256=sha(full),readback_verified=True,manifest_sha256=manifest_sha,
            combined_policy_sha256=sha(job/'D2_COMBINED_POLICY.json'),bridge_policy_sha256=p['bridge_policy_sha256'],
            bridge_verification_sha256=sha(job/'BRIDGE_VERIFICATION.json'),
            transport_readback_receipt=receipt.relative_to(job).as_posix(),transport_readback_sha256=sha(receipt),
            formal_custody_assurance=False,scientific_acceptance=False))
        print('Compatibility evidence saved and remotely read back. Publisher closure is checked before admission.',flush=True)
        return
    if action=='save':
        need(quota['available_bytes']>=full.stat().st_size+p['review_bound_bytes'],'Drive return space insufficient')
        ref=http.upload_verified(full,deadline=deadline,check=check)
        descriptor=job/'DURABLE_FULL_RETURN.json';put(descriptor,dict(reference=ref,sha256=sha(full),bytes=full.stat().st_size,
            result_url='https://drive.google.com/file/d/'+ref['object_id']+'/view',keep_full_return=True,formal_custody_assurance=False))
        selected={n:path for n,path in complete_entries.items() if path.suffix in ('.json','.jsonl','.py','.txt','.md') or (path.suffix=='.npy' and path.name.startswith('designs_'))}
        selected['DURABLE_FULL_RETURN.json']=descriptor
        review=job/('D2_COMBINED_REVIEW_'+act['attempt']+'.zip');pack(review,selected,p['review_bound_bytes'],check)
        review_ref=http.upload_verified(review,deadline=deadline,check=check)
        done=(job/'output/RESULT.json').is_file()
        result=dict(complete=True,archive=full.name,reference=ref,review_archive=review.name,review_reference=review_ref,
            local_sha256=sha(full),review_sha256=sha(review),gpu_work_complete=done,compatibility_comparison_complete=(job/'BRIDGE_VERIFICATION.json').is_file(),compatibility_source='reused attempt '+p['reused_bridge']['attempt']+' bridge, re-verified in this attempt',new_compatibility_computations=0,
            mechanical_D2_admission_issued=(job/'D2_RUNTIME_ADMISSION.json').is_file(),
            compatibility_scope='D2 raw960 diagnostics only',campaign_diagnostics_produced=200 if done else None,
            campaign_full_evaluations_added=0,final_full_evaluation_complete=False,
            canonical_worker_advanced=False,formal_custody_assurance=False,
            result_url='https://drive.google.com/file/d/'+ref['object_id']+'/view',review_url='https://drive.google.com/file/d/'+review_ref['object_id']+'/view')
        put(job/'SAVE_RESULT.json',result);print('DOWNLOAD THIS SMALL REVIEW: '+result['review_url'],flush=True);return
    need(action=='closure','Unknown action')
    saved=read(job/'SAVE_RESULT.json');closure=read(job/'supervision/save/RESULT.json')
    need(closure['returncode']==0 and closure['closure']['resolved'] and closure['stdout_closed'] and not closure['errors'],'Return publisher unresolved')
    path=job/('D2_COMBINED_SAVED_'+act['attempt']+'.json');put(path,dict(saved=saved,publisher_closure=closure,activation=act,
        note='Producer and main publisher closed; this last metadata publisher is separate.',formal_custody_assurance=False))
    ref=http.upload_verified(path,deadline=deadline,check=check)
    put(job/'CLOSING_RECORD_SAVED.json',dict(reference=ref,result_url=saved['result_url'],review_url=saved['review_url'],closure_url='https://drive.google.com/file/d/'+ref['object_id']+'/view'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['input','bridge-save','save','closure']);parser.add_argument('--job',required=True,type=Path);parser.add_argument('--deadline',required=True,type=float)
    args=parser.parse_args();transfer(args.job.resolve(),args.action,args.deadline)
