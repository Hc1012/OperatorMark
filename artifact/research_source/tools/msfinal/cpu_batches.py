"""D6 wave transport and bounded CPU batches; no authority or acceptance issuer."""
from pathlib import Path
from .bounded_stage import admitted_deadline
import shutil,time
from msdev.records import read,write,sha,publish_zip,safe_name
from . import cpu_scoring as C
from .graph import identity
from .scientific_admission import require as require_accepted
from .session import closure_ok
from .phase_storage import verify_archive

def limits(value):
    names={'requests','input_bytes','return_bytes','work_seconds'}
    if not isinstance(value,dict) or set(value)!=names or any(type(value[n]) is not int or value[n]<=0 for n in names):
        raise ValueError('explicit positive CPU batch capacity/work limits required')
    return value

def requests(wave,expected,roots,check=lambda:None):
    if sha(wave,check)!=expected:raise ValueError('wave binding drift')
    w=read(wave)
    if w.get('schema')!='D6.wave.v1' or len(w['required_phases'])!=len(set(w['required_phases'])):
        raise ValueError('wave schema/duplicate membership')
    if (len(w['visited_phases'])!=len(set(w['visited_phases'])) or not set(w['phases'])<=set(w['visited_phases'])
        or not set(w['visited_phases'])<=set(w['required_phases']) or (w['all_producers_visited'] and set(w['visited_phases'])!=set(w['required_phases']))):
        raise ValueError('wave visitation incomplete')
    if set(roots)!=set(w['phases']):raise ValueError('exact pending producer roots required')
    all_requests={m['sha256'] for p in w['phases'].values() for m in p['requests'].values()}
    if not set(w.get('verified_returns',{}))<=all_requests:raise ValueError('verified return outside wave membership')
    result=[]
    for phase,entry in w['phases'].items():
        root,receipt=roots[phase];root=Path(root);receipt=Path(receipt)
        if sha(receipt,check)!=entry['receipt_sha256']:raise ValueError('wave producer receipt drift')
        saved=read(receipt)
        if saved['name']!=phase or saved['binding']!=w['binding'] or C.members(saved['value'])!=entry['requests']:
            raise ValueError('wave/request membership substitution')
        for directory,m in entry['requests'].items():
            q=C.verify_request(root/directory,m['sha256'],check)
            if m['sha256'] in w.get('verified_returns',{}):continue
            result.append(dict(id=identity([phase,directory]),phase=phase,directory=directory,root=str(root/directory),
                request_sha256=m['sha256'],phase_receipt_sha256=entry['receipt_sha256'],operations=q['operations'],
                input_bytes=sum(v['bytes'] for v in q['files'].values())+(root/directory/'D6_REQUEST.json').stat().st_size))
    if len({r['request_sha256'] for r in result})!=len(result):raise ValueError('duplicate scoring request')
    return w,sorted(result,key=lambda r:(r['phase'],r['directory']))

def partition(rows,bounds,capacity):
    """Deterministic physical partitions; unknown cost is never zero."""
    cap=limits(capacity)
    if set(bounds)!={r['id'] for r in rows}:raise ValueError('complete per-request work/return bounds required')
    batches=[];batch=[];used=dict.fromkeys(cap,0)
    for r in rows:
        b=bounds[r['id']]
        if set(b)!={'work_seconds','return_bytes'} or any(type(v) is not int or v<=0 for v in b.values()):raise ValueError('unknown/nonpositive request envelope')
        cost=dict(requests=1,input_bytes=r['input_bytes'],**b)
        if any(cost[k]>cap[k] for k in cap):raise ValueError('one CPU request cannot fit the bounded batch')
        if batch and any(used[k]+cost[k]>cap[k] for k in cap):batches.append(batch);batch=[];used=dict.fromkeys(cap,0)
        batch.append(dict(r,envelope=b));used={k:used[k]+cost[k] for k in cap}
    if batch:batches.append(batch)
    return batches

def export_wave(wave,expected,roots,worker_result,closure,export_permits,bounds,capacity,destination,check=lambda:None,clock=time.time):
    """Export closed ready subsets into one non-nested archive per physical batch.

    Uses the existing per-request input-export admission, including destination.
    The closure binds the wave record via the actual Worker result.
    """
    w,rows=requests(wave,expected,roots,check);closed=closure.read();saved=read(worker_result)
    if not closure_ok(closed) or closed.get('product_readback') is not True or closed.get('worker_result_sha256')!=sha(worker_result):raise ValueError('closed producer required')
    if expected not in saved.get('cpu_wave_records',{}).values():raise ValueError('closed result does not bind wave')
    if set(export_permits)!={r['id'] for r in rows}:raise ValueError('export permission membership')
    batches=partition(rows,bounds,capacity);dest=Path(destination)
    for group in batches:
        for r in group:
            grant=require_accepted(export_permits[r['id']],kind='main_study_D6_input_export',bindings=dict(
                request_sha256=r['request_sha256'],phase_receipt_sha256=r['phase_receipt_sha256'],closure_sha256=closure.sha256,destination=str(dest.resolve())))
            q=C.verify_request(r['root'],r['request_sha256'],check)
            from .bounded_stage import admitted_deadline
            if grant.get('domain')!='final' or grant.get('deadline',float('inf'))>admitted_deadline(q['binding'],grant) or clock()>=grant['deadline']:raise ValueError('export original deadline/domain')
            if saved.get('product_receipts',{}).get(r['phase'])!=r['phase_receipt_sha256']:raise ValueError('producer closure membership')
    original_check=check;deadline=min((p.read()['deadline'] for p in export_permits.values()),default=clock())
    def check():
        original_check();closure.read()
        if clock()>=deadline:raise TimeoutError('CPU export immutable deadline')
    dest.mkdir(parents=True,exist_ok=False);outputs=[]
    for index,group in enumerate(batches):
        folder=dest/f'batch_{index:04d}';payload=folder/'payload';payload.mkdir(parents=True)
        files={};jobs=[]
        for r in group:
            check();g=export_permits[r['id']].read()
            if clock()>=g['deadline']:raise TimeoutError('wave export original deadline')
            root=Path(r['root']);q=C.verify_request(root,r['request_sha256'],check)
            members=dict(q['files']);members['D6_REQUEST.json']=dict(bytes=(root/'D6_REQUEST.json').stat().st_size,sha256=r['request_sha256'])
            for n,e in members.items():
                check();rel=r['id']+'/'+n;target=payload/rel;target.parent.mkdir(parents=True,exist_ok=True)
                if shutil.disk_usage(dest).free<e['bytes']:raise OSError('CPU batch staging capacity exhausted')
                shutil.copyfile(root/n,target)
                if target.stat().st_size!=e['bytes'] or sha(target,check)!=e['sha256']:raise ValueError('CPU batch staging drift')
                files[rel]=e
            jobs.append({k:v for k,v in r.items() if k!='root'})
        plan=dict(schema='D6.batch.v1',wave_sha256=expected,wave=w['wave'],jobs=jobs,capacity=capacity,files=files,
            producer_closure_sha256=closure.sha256,producer_worker_result_sha256=sha(worker_result))
        write(payload/'BATCH.json',plan);files=dict(files,**{'BATCH.json':dict(bytes=(payload/'BATCH.json').stat().st_size,sha256=sha(payload/'BATCH.json'))})
        archive=folder/'CPU_BATCH_INPUT.zip'
        need=sum(v['bytes'] for v in files.values())+sum(200+2*len(n.encode()) for n in files)+1024
        if shutil.disk_usage(folder).free<need:raise OSError('CPU batch archive capacity exhausted')
        publish_zip(archive,payload,sorted(files),check);verify_archive(archive,files,check)
        manifest=dict(files=files,archive_sha256=sha(archive,check),batch_sha256=sha(payload/'BATCH.json'),wave_sha256=expected)
        write(folder/'OUTPUT_MANIFEST.json',manifest);outputs.append(dict(directory=str(folder),manifest_sha256=sha(folder/'OUTPUT_MANIFEST.json'),batch_sha256=manifest['batch_sha256']))
    requests(wave,expected,roots,check)
    write(dest/'BATCHES.json',dict(wave_sha256=expected,batches=outputs,request_count=len(rows),independent_acceptance=False));return outputs

def verify_batch(root,expected,check=lambda:None):
    root=Path(root)
    if sha(root/'BATCH.json',check)!=expected:raise ValueError('batch identity drift')
    b=read(root/'BATCH.json');limits(b['capacity'])
    if b.get('schema')!='D6.batch.v1' or not b['jobs']:raise ValueError('empty/invalid CPU batch')
    if len({j['id'] for j in b['jobs']})!=len(b['jobs']) or len({j['request_sha256'] for j in b['jobs']})!=len(b['jobs']):raise ValueError('duplicate CPU batch request')
    files={};rows=[]
    for j in b['jobs']:
        safe_name(j['id'])
        if j['id']!=identity([j['phase'],j['directory']]):raise ValueError('batch phase/request route differs')
        q=C.verify_request(root/j['id'],j['request_sha256'],check)
        if j['operations']!=q['operations']:raise ValueError('batch operation population')
        members=dict(q['files']);members['D6_REQUEST.json']=dict(bytes=(root/j['id']/'D6_REQUEST.json').stat().st_size,sha256=j['request_sha256'])
        if j['input_bytes']!=sum(v['bytes'] for v in members.values()):raise ValueError('batch input byte claim')
        files.update({j['id']+'/'+n:v for n,v in members.items()});rows.append(j)
    if files!=b['files'] or {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}!=set(files)|{'BATCH.json'}:raise ValueError('exact CPU batch file membership')
    if len(partition(rows,{j['id']:j['envelope'] for j in rows},b['capacity']))!=1:raise ValueError('batch capacity exceeded')
    return b

def score_batch(root,expected,out,permit,runtime,compatibility,request_permits,check,*,source_root,runtime_observer=C.runtime_check,kernels=None,clock=time.time):
    b=verify_batch(root,expected,check);out=Path(out)
    g=require_accepted(permit,kind='main_study_D6_CPU_batch',bindings=dict(batch_sha256=expected,cpu_runtime_sha256=runtime.sha256,compatibility_sha256=compatibility.sha256))
    if g.get('domain')!='final' or g.get('device')!='CPU' or g.get('retries')!=0 or g.get('capacity')!=b['capacity'] or Path(g['scoring_output']).resolve()!=out.resolve():raise ValueError('CPU batch scope/output/capacity')
    if set(request_permits)!={j['id'] for j in b['jobs']}:raise ValueError('CPU batch permit membership')
    if g.get('request_permit_sha256')!={k:v.sha256 for k,v in request_permits.items()}:raise ValueError('CPU batch permission substitution')
    for j in b['jobs']:
        q=C.verify_request(Path(root)/j['id'],j['request_sha256'],check)
        if C.scoring_identities(q,runtime,compatibility)!=(runtime.sha256,compatibility.sha256):raise ValueError('batch runtime/compatibility correspondence')
        require_accepted(compatibility,kind='main_study_D6_numerical_compatibility',bindings=dict(
            producer_runtime_payload_sha256=q['binding']['producer_runtime_payload_sha256'],cpu_runtime_sha256=runtime.sha256,source_pins=q['cpu_policy']['source_pins']))
        a=require_accepted(request_permits[j['id']],kind='main_study_D6_CPU_scoring',bindings=dict(request_sha256=j['request_sha256'],cpu_runtime_sha256=runtime.sha256,compatibility_sha256=compatibility.sha256))
        if (a.get('domain')!='final' or a.get('device')!='CPU' or a.get('retries')!=0 or a.get('operations')!=j['operations']
            or a['deadline']>g['deadline'] or g['deadline']>admitted_deadline(q['binding'],g)
            or Path(a['scoring_output']).resolve()!=(out/j['id']).resolve() or Path(a['source_root']).resolve()!=Path(source_root).resolve()):raise ValueError('CPU batch child admission')
    if clock()+sum(j['envelope']['work_seconds'] for j in b['jobs'])>g['deadline']:raise TimeoutError('whole CPU batch work does not fit')
    out.mkdir(parents=True,exist_ok=False);done=[]
    try:
        for j in b['jobs']:
            end=min(g['deadline'],clock()+j['envelope']['work_seconds'])
            next_storage=[0.]
            def bounded(force_storage=False):
                check();permit.read()
                if clock()>=end:raise TimeoutError('CPU request envelope exhausted; no retry')
                # Deadline remains immediate. Avoid rescanning a growing request
                # on every hash/read callback; practical capacity is checked each
                # second and unconditionally at the completed request boundary.
                if force_storage or time.monotonic()>=next_storage[0]:
                    used=sum(p.stat().st_size for p in (out/j['id']).glob('*') if p.is_file()) if (out/j['id']).exists() else 0
                    if used>j['envelope']['return_bytes']:raise OSError('CPU request output capacity exhausted')
                    next_storage[0]=time.monotonic()+1.
            C.score(Path(root)/j['id'],j['request_sha256'],out/j['id'],request_permits[j['id']],runtime,compatibility,bounded,
                source_root=source_root,runtime_observer=runtime_observer,kernels=kernels,clock=clock)
            bounded(True);done.append(j['id'])
        result=dict(complete=True,batch_sha256=expected,requests=done,manifests={i:sha(out/i/'RETURN_MANIFEST.json') for i in done},independent_acceptance=False)
        write(out/'BATCH_RESULT.json',result);return result
    except BaseException as e:
        write(out/'BATCH_FAILURE.json',dict(error=repr(e),completed=done,complete=False));raise

def consume_return(input_root,batch_sha,output_root,output_manifest_sha,acceptances,check=lambda:None,*,output_manifest=None,archive=None):
    """One batch -> existing per-phase Evidence.cpu_scores mappings."""
    b=verify_batch(input_root,batch_sha,check);root=Path(output_root)
    output_manifest=Path(output_manifest) if output_manifest is not None else root/'OUTPUT_MANIFEST.json'
    archive=Path(archive) if archive is not None else root/'CPU_RETURN.zip'
    if sha(output_manifest,check)!=output_manifest_sha:raise ValueError('batch return manifest substitution')
    m=read(output_manifest)
    if m.get('batch_sha256')!=batch_sha:raise ValueError('batch return input identity')
    for n,e in m['files'].items():
        safe_name(n);p=root/n
        if p.is_symlink() or p.stat().st_size!=e['bytes'] or sha(p,check)!=e['sha256']:raise ValueError('batch return content drift')
    extras={p.relative_to(root.resolve()).as_posix() for p in (output_manifest.resolve(),archive.resolve()) if p.is_relative_to(root.resolve())}
    if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}!=set(m['files'])|extras:raise ValueError('batch return exact membership')
    verify_archive(archive,m['files'],check)
    if sha(archive,check)!=m['archive_sha256']:raise ValueError('batch return archive drift')
    result=read(root/'worker/BATCH_RESULT.json');ids=[j['id'] for j in b['jobs']]
    if (result.get('complete') is not True or result.get('requests')!=ids or result.get('batch_sha256')!=batch_sha or set(acceptances)!=set(ids)
        or set(result.get('manifests',{}))!=set(ids) or {p.name for p in (root/'worker').iterdir()}!=set(ids)|{'BATCH_RESULT.json'}):raise ValueError('partial/extra batch cannot release returns')
    from .cpu_score_entry import record
    returned={}
    for j in b['jobs']:
        folder=root/'worker'/j['id'];closure_path=root/'closures'/(j['id']+'.json')
        closure=record(closure_path,m['files']['closures/'+j['id']+'.json']['sha256'])
        r=dict(root=folder,acceptance=acceptances[j['id']],closure=closure,observations=root)
        if result['manifests'].get(j['id'])!=sha(folder/'RETURN_MANIFEST.json'):raise ValueError('batch member manifest drift')
        C.return_readback(Path(input_root)/j['id'],j['request_sha256'],folder,r['acceptance'],closure,check,observations=root)
        returned.setdefault(j['phase'],{})[j['directory']]=r
    return returned
