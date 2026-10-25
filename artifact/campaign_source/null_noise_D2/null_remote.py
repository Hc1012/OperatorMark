"""Pinned Drive inputs and lossless bounded returns for null/B2 only."""
from pathlib import Path
from urllib.parse import urlencode
import argparse, json, os, shutil, time
from null_io import (need,read,put,sha,source,pack,extract,_http,_privacy_and_quota,
                     uploaded_control,_upload,_validate_reference,_closed)

POLICY='NULL_B2_POLICY.json'
SCOPE='fixed_null210_B2_1240_only'


def fixed_deadline(job,deadline):
    policy=read(job/POLICY);activation=read(job/'ACTIVATION.json')
    need(activation.get('scope')==policy['scope']==SCOPE and activation.get('policy_sha256')==sha(job/POLICY)
         and activation.get('retries')==0 and activation.get('previous_clocks_reused') is False,
         'Fixed null/B2 activation required')
    need(activation['start']<=time.time()<deadline<=activation['deadline'],'Fixed transport clock expired')
    def check():
        if time.time()>=deadline:raise TimeoutError('Fixed null transport deadline expired; no retry')
    return policy,activation,check


def capacity(job,deadline):
    policy=read(job/POLICY)
    need(policy['scope']==SCOPE and not (job/'ACTIVATION.json').exists(),'Capacity observation belongs before activation')
    need(time.time()<deadline<=time.time()+900,'Bounded capacity deadline required')
    def check():
        if time.time()>=deadline:raise TimeoutError('Pre-activation capacity timeout')
    http=_http(job,policy,'capacity',4*1024**2)
    observation=_privacy_and_quota(http,policy,deadline,check)
    free=shutil.disk_usage(job).free
    record=dict(passed=observation['quota']['available_bytes']>=policy['minimum_drive_free_bytes'] and free>=policy['minimum_linux_free_bytes'],
        scope=SCOPE,policy_sha256=sha(job/POLICY),before_activation=True,observed_at=time.time(),
        private_owner_only=True,drive_free_bytes=observation['quota']['available_bytes'],linux_free_bytes=free,
        observation=observation,not_a_reservation=True,purchases=0,scientific_calls=0)
    put(job/'STORAGE_PREFLIGHT.json',record)
    need(record['passed'],'Prepared storage bound unavailable; no activation')
    return record


def claim(job,policy,activation,http,deadline,check):
    from msfinal.drive_transport import API
    control,want=uploaded_control(job,policy)
    observation=_privacy_and_quota(http,policy,deadline,check)
    need(observation['quota']['available_bytes']>=policy['minimum_drive_free_bytes']
         and shutil.disk_usage(job).free>=policy['minimum_linux_free_bytes'],'Whole-wave storage changed before input')
    marker_name='NULL_B2_ONCE_'+sha(job/POLICY)+'.json'
    listing=http.json('GET',API+'files?'+urlencode(dict(
        q="'%s' in parents and trashed=false and name='%s'"%(policy['drive_folder'],marker_name),
        fields='nextPageToken,files(id,name)',pageSize=100)),deadline=deadline,check=check)
    need(not listing.get('nextPageToken') and listing.get('files')==[],
         'This exact null/B2 campaign already has an attempt; no replay')
    marker=job/marker_name
    put(marker,dict(activation=activation,policy_sha256=sha(job/POLICY),attempt_only=True,formal_exclusive_custody=False))
    put(job/'ONCE_MARKER_SAVED.json',dict(_upload(job,policy,http,marker,deadline,check),formal_exclusive_custody=False))
    put(job/'CONTROL_SAVED.json',dict(_upload(job,policy,http,control,deadline,check),policy_sha256=sha(job/POLICY)))
    put(job/'CLAIM_COMPLETE.json',dict(complete=True,policy_sha256=sha(job/POLICY),
        activation_sha256=sha(job/'ACTIVATION.json'),before_environment_setup=True))


def input_files(job,policy,activation,http,deadline,check):
    control,want=uploaded_control(job,policy)
    claimed=read(job/'CLAIM_COMPLETE.json')
    need(claimed['complete'] is True and claimed['policy_sha256']==sha(job/POLICY)
         and claimed['activation_sha256']==sha(job/'ACTIVATION.json'),'Original once-only claim required')
    _validate_reference(job,policy,read(job/'ONCE_MARKER_SAVED.json'))
    _validate_reference(job,policy,read(job/'CONTROL_SAVED.json'))
    gate=read(job/'NULL_B2_RUNTIME_GATE.json')
    need(gate['passed'] is True and gate['scope']==SCOPE and gate['policy_sha256']==sha(job/POLICY)
         and gate['activation_sha256']==sha(job/'ACTIVATION.json'),'Fresh scoped runtime gate required before large input downloads')
    groups=policy['sources']
    need(sum(s['reference']['bytes'] for s in groups)==policy['exact_input_download_bytes']<=policy['input_download_bound_bytes'],
         'Frozen input-byte count differs')
    destinations=[n for s in groups for n in s['files']]
    need(len(destinations)==len(set(destinations)),'Duplicate pinned input destination')
    for ordinal,group in enumerate(groups):
        ref=group['reference'];check()
        need(ref['folder_id']==policy['drive_folder'] and ref['keepForever'] is True,'Input revision scope differs')
        archive=job/f'INPUT_{ordinal:02d}.zip'
        print(f'Fetching existing Drive archive {ordinal+1}/{len(groups)}; only required inputs will be extracted.',flush=True)
        with archive.open('xb') as sink:
            http.verify(ref,deadline=deadline,check=check,sink=sink);sink.flush();os.fsync(sink.fileno())
        extract(archive,job,group['files'],check)
        # Only this newly downloaded scratch copy is discarded. The pinned
        # original Drive revision and every required extracted byte remain.
        need(sha(archive,check)==ref['sha256'],'Downloaded archive changed before scratch cleanup')
        archive.unlink()
        put(job/'input_receipts'/f'{ordinal:02d}.json',dict(reference=ref,files=group['files'],
            scratch_archive_discarded_after_verification=True,remote_original_unchanged=True))
    for name,entry in policy['aliases'].items():
        check();src=source(job,entry['source']);expected=entry['expected']
        need(src.stat().st_size==expected['bytes'] and sha(src,check)==expected['sha256'],'Alias source differs: '+name)
        dst=job/name;dst.parent.mkdir(parents=True,exist_ok=True)
        need(dst.parent.resolve().is_relative_to(job),'Alias path escaped job')
        with src.open('rb') as a,dst.open('xb') as b:
            while chunk:=a.read(1024**2):check();b.write(chunk)
            b.flush();os.fsync(b.fileno())
        need(sha(dst,check)==expected['sha256'],'Alias readback differs')
    need(uploaded_control(job,policy)==(control,want),'Control archive changed during input')
    put(job/'INPUT_READBACK.json',dict(complete=True,files=len(destinations),aliases=len(policy['aliases']),
        exact_input_download_bytes=policy['exact_input_download_bytes'],sources=[s['reference'] for s in groups],
        expectations_frozen_before_launch=True,new_scientific_calls=0))


def metadata_entries(job,policy):
    entries={name:source(job,name) for name in policy['files']}
    # Full raw JVPs/rows are in per-owner capsules. Only summaries and their
    # exact hashes enter the review; no laptop copy of the large capsules.
    for root in ['output','observations','supervision','stage_budgets','input_receipts','owner_receipts','partial_receipts']:
        for path in (job/root).rglob('*'):
            if not path.is_file():continue
            name=path.relative_to(job).as_posix()
            if '/raw/' in name or path.suffix=='.npy' or name.startswith(('supervision/save/','supervision/closing_record/')):continue
            entries[name]=source(job,name)
    for folder in job.glob('transport_*'):
        if not folder.is_dir() or folder.name in ('transport_save','transport_closure'):continue
        for path in folder.rglob('*'):
            if path.is_file():entries[path.relative_to(job).as_posix()]=source(job,path.relative_to(job).as_posix())
    for p in job.iterdir():
        if p.is_file() and p.suffix in ('.json','.jsonl','.py','.md','.txt'):entries[p.name]=source(job,p.name)
    return entries


def producer_terminal(job):
    path=job/'supervision/producer/RESULT.json'
    if not path.exists():return not (job/'supervision/producer/IDENTITY.json').exists()
    r=read(path);c=r.get('closure',{})
    return (c.get('resolved') is True and c.get('child_reaped') is True and not c.get('survivors')
            and not c.get('errors') and r.get('stdout_closed') is True)


def save(job,policy,activation,http,deadline,check):
    need(producer_terminal(job),'Producer custody unresolved; no concurrent output packaging')
    receipts={};partials={}
    for owner in policy['owners']:
        path=job/'owner_receipts'/(owner+'.json')
        if path.is_file():
            row=read(path);_validate_reference(job,policy,row)
            roster=read(job/'NULL_B2_ROSTER.json')
            need(row['owner']==owner and row['models']==[r['id'] for r in roster['model_jobs'] if r['owner']==owner],
                 'Owner capsule model population differs');receipts[owner]=row
        elif (job/'output/owners'/owner).is_dir():
            root=job/'output/owners'/owner
            entries={p.relative_to(job).as_posix():p for p in root.rglob('*') if p.is_file()}
            for p in (job/'output/reference_jvp').glob('*'):
                if p.is_file():entries[p.relative_to(job).as_posix()]=p
            archive=job/f'NULL_B2_PARTIAL_{activation["attempt"]}_{owner}.zip'
            pack(archive,entries,policy['owner_capsule_bound_bytes'],check)
            record=dict(_upload(job,policy,http,archive,deadline,check),complete=False,owner=owner)
            put(job/'partial_receipts'/(owner+'.json'),record);partials[owner]=record
    entries=metadata_entries(job,policy)
    # Preserve all reference derivative outputs even on an early producer stop.
    full_entries=dict(entries)
    for p in (job/'output/reference_jvp').glob('*.npy'):full_entries[p.relative_to(job).as_posix()]=p
    for p in (job/'output/f64_reference').rglob('*.npy'):full_entries[p.relative_to(job).as_posix()]=p
    full=job/f'NULL_B2_FULL_{activation["attempt"]}.zip'
    pack(full,full_entries,policy['return_bound_bytes'],check)
    full_record=_upload(job,policy,http,full,deadline,check)
    descriptor=job/'DURABLE_FULL_RETURN.json'
    put(descriptor,dict(full_record,owner_capsules=receipts,partial_capsules=partials,formal_custody_assurance=False))
    entries['DURABLE_FULL_RETURN.json']=descriptor
    review=job/f'NULL_B2_REVIEW_{activation["attempt"]}.zip'
    pack(review,entries,policy['review_bound_bytes'],check)
    review_record=_upload(job,policy,http,review,deadline,check)
    result=read(job/'output/RESULT.json') if (job/'output/RESULT.json').exists() else (
        read(job/'output/FAILURE.json') if (job/'output/FAILURE.json').exists() else {})
    complete=(result.get('complete') is True and result.get('B2_groups')==1240
        and result.get('counts')==dict(model_variants=1240,jvp_batches=19840,jvp_samples=317440,B2_pairs=12390,keyed_entries=210,
                                      T128_forward=128,T128_tangent=8192)
        and set(receipts)==set(policy['owners']) and not partials and _closed(job/'supervision/producer/RESULT.json'))
    saved=dict(complete=complete,saved=True,scope=SCOPE,archive=full.name,reference=full_record['reference'],
        local_sha256=sha(full),review_archive=review.name,review_reference=review_record['reference'],review_sha256=sha(review),
        owner_capsules=receipts,partial_capsules=partials,counts=result.get('counts',{}),B2_groups=result.get('B2_groups',0),
        CPU_scoring_required=False,training_updates=0,inverse_updates=0,
        native_solver_calls=sum(result.get('counts',{}).get(k,0) for k in ('T128_forward','T128_tangent')),
        native_optimization_repeated=False,
        scientific_acceptance=False,formal_custody_assurance=False,canonical_worker_advanced=False,
        result_url='https://drive.google.com/file/d/'+full_record['reference']['object_id']+'/view',
        review_url='https://drive.google.com/file/d/'+review_record['reference']['object_id']+'/view')
    put(job/'SAVE_RESULT.json',saved)
    return saved


def transfer(job,action,deadline):
    job=Path(job).resolve()
    if action=='capacity':return capacity(job,deadline)
    policy,activation,check=fixed_deadline(job,deadline)
    need(action in ('claim','input','save','closure'),'Unknown null transport action')
    limit=policy['input_transport_bound_bytes'] if action in ('claim','input') else 2*policy['aggregate_return_bound_bytes']
    http=_http(job,policy,action,limit)
    if action=='claim':return claim(job,policy,activation,http,deadline,check)
    if action=='input':return input_files(job,policy,activation,http,deadline,check)
    if action=='save':return save(job,policy,activation,http,deadline,check)
    need(_closed(job/'supervision/save/RESULT.json'),'Main publisher closure unresolved')
    saved=read(job/'SAVE_RESULT.json')
    path=job/f'NULL_B2_SAVED_{activation["attempt"]}.json'
    put(path,dict(saved=saved,activation=activation,publisher_closure=read(job/'supervision/save/RESULT.json'),
        note='Producer and main publisher closure are recorded. This final metadata publisher is separate.',
        formal_custody_assurance=False))
    record=_upload(job,policy,http,path,deadline,check)
    put(job/'CLOSING_RECORD_SAVED.json',dict(record,
        closure_url='https://drive.google.com/file/d/'+record['reference']['object_id']+'/view'))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['capacity','claim','input','save','closure'])
    p.add_argument('--job',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();transfer(a.job,a.action,a.deadline)
