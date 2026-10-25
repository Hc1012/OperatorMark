"""Owned, disabled-by-default D6 multi-request CPU entry. No grant issuer/retry."""
from pathlib import Path
import argparse,importlib.util,os,sys,time,traceback,shutil
from msdev.records import read,write,sha,publish_zip
from .cpu_score_entry import record,persist_closure,Q0_SHA
from . import cpu_batches as B
from .scientific_admission import require as require_accepted
from .phase_storage import verify_archive

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('batch','batch-sha','permit','permit-sha','runtime','runtime-sha','compatibility','compatibility-sha','source-root','output','supervisor'):p.add_argument('--'+n,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--owned-child',action='store_true');a=p.parse_args()
    if not a.execute:raise ValueError('UNSTARTED: explicit batch execution authority required')
    permit=record(a.permit,a.permit_sha);runtime=record(a.runtime,a.runtime_sha);compat=record(a.compatibility,a.compatibility_sha)
    plan=B.verify_batch(a.batch,a.batch_sha)
    grant=require_accepted(permit,kind='main_study_D6_CPU_batch',bindings=dict(batch_sha256=a.batch_sha,cpu_runtime_sha256=a.runtime_sha,compatibility_sha256=a.compatibility_sha))
    out=Path(a.output)
    if (grant.get('domain')!='final' or grant.get('device')!='CPU' or grant.get('retries')!=0
        or grant.get('capacity')!=plan['capacity'] or Path(grant['output_root']).resolve()!=out.resolve()
        or Path(grant['scoring_output']).resolve()!=(out/'worker').resolve()
        or Path(grant['source_root']).resolve()!=Path(a.source_root).resolve()
        or Path(grant['interpreter']).resolve()!=Path(sys.executable).resolve()):raise ValueError('bound batch output/source/interpreter/capacity required')
    permits={k:record(v['path'],v['sha256']) for k,v in grant['request_permits'].items()}
    if set(permits)!={j['id'] for j in plan['jobs']} or {k:v.sha256 for k,v in permits.items()}!=grant['request_permit_sha256']:raise ValueError('batch request authority membership')
    def check():
        permit.read()
        if time.time()>=grant['deadline']:raise TimeoutError('original CPU batch deadline')
    check();consumed=dict(permit_sha256=a.permit_sha,batch_sha256=a.batch_sha,deadline=grant['deadline'],retries=0)
    if a.owned_child:
        if read(out/'CONSUMED.json')!=consumed:raise ValueError('owned batch consumption missing')
        B.score_batch(a.batch,a.batch_sha,out/'worker',permit,runtime,compat,permits,check,source_root=a.source_root);return
    if os.name!='nt' or sha(a.supervisor)!=Q0_SHA:raise ValueError('verified Windows Q0 ownership required')
    work=grant['work_seconds'];cleanup=grant['cleanup_seconds'];finalization=grant['finalization_seconds']
    if (any(type(v) not in (int,float) or v<=0 for v in (work,cleanup,finalization))
        or work<sum(j['envelope']['work_seconds'] for j in plan['jobs'])
        or work>plan['capacity']['work_seconds'] or time.time()+work+cleanup+finalization>grant['deadline']):raise ValueError('whole CPU batch envelope does not fit')
    if type(grant.get('storage_bytes')) is not int or grant['storage_bytes']<=0 or shutil.disk_usage(out.parent).free<grant['storage_bytes']:raise OSError('bound CPU batch practical capacity unavailable')
    env=dict(os.environ,**grant['environment']);env['PYTHONPATH']=os.pathsep.join([str(Path(a.source_root)/'tools'),a.source_root])
    from q6_common import THREAD_ENV
    if any(env.get(k)!=v for k,v in THREAD_ENV.items()):raise ValueError('pinned CPU environment required')
    out.mkdir(parents=True,exist_ok=False);(out/'supervisor').mkdir();write(out/'CONSUMED.json',consumed)
    spec=importlib.util.spec_from_file_location('q0',a.supervisor);q0=importlib.util.module_from_spec(spec);spec.loader.exec_module(q0)
    entry=Path(__file__).with_name('cpu_score_entry.py')
    try:
        r=q0.run_with_timeout([sys.executable,'-B',str(entry),*sys.argv[1:],'--owned-child'],cwd=a.source_root,env=env,timeout_s=work,cleanup_timeout_s=cleanup,out_dir=out/'supervisor')
        write(out/'SUPERVISOR_RESULT.json',r);end=min(grant['deadline'],time.time()+finalization)
        def final_check():
            check()
            if time.time()>=end:raise TimeoutError('CPU batch finalization exhausted')
        final_check();c=r['cleanup']
        if not (r['exit_code']==0 and not r['timed_out'] and c['resolved'] and c['child_reaped'] and c['group_survivors'] is False and not c.get('errors') and c.get('job_accounting',{}).get('active_processes')==0 and r['job_handle_closed']):raise RuntimeError('batch failed or unresolved owned closure; no retry')
        finalize_batch(plan,a.batch_sha,out,grant,final_check)
    except BaseException as e:
        write(out/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),complete=False));raise

def finalize_batch(plan,batch_sha,out,grant,final_check):
    """After real owned closure: exact per-request readback and one return archive."""
    br=read(out/'worker/BATCH_RESULT.json')
    if br.get('complete') is not True or br.get('batch_sha256')!=batch_sha or br.get('requests')!=[j['id'] for j in plan['jobs']]:raise ValueError('incomplete batch result')
    (out/'closures').mkdir()
    for j in plan['jobs']:
        folder=out/'worker'/j['id'];m=read(folder/'RETURN_MANIFEST.json')
        if br['manifests'].get(j['id'])!=sha(folder/'RETURN_MANIFEST.json'):raise ValueError('batch member binding')
        if {p.name for p in folder.iterdir()}!=set(m['files'])|{'RETURN_MANIFEST.json'}:raise ValueError('batch member membership')
        for n,e in m['files'].items():
            final_check();safe=folder/n
            if safe.stat().st_size!=e['bytes'] or sha(safe,final_check)!=e['sha256']:raise ValueError('batch member readback drift')
        persist_closure(out/'closures'/(j['id']+'.json'),dict(schema='D6.Q0.closed.v1',child_reaped=True,active_processes=0,survivors=False,resolved=True,
            job_handle_closed=True,output_readback=True,timed_out=False,returncode=0,result_sha256=sha(folder/'RESULT.json'),
            supervisor_sha256=sha(out/'SUPERVISOR_RESULT.json'),combined_stream_sha256=sha(out/'supervisor/pytest_output.txt'),
            batch_sha256=batch_sha,batch_result_sha256=sha(out/'worker/BATCH_RESULT.json'),deadline=grant['deadline'],independent_acceptance=False))
    files={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p,final_check)) for p in out.rglob('*') if p.is_file()}
    archive_bound=sum(e['bytes'] for e in files.values())+sum(200+2*len(n.encode()) for n in files)+1024
    if sum(e['bytes'] for e in files.values())+archive_bound>grant['storage_bytes'] or shutil.disk_usage(out).free<archive_bound:raise OSError('batch return capacity exhausted')
    archive=out/'CPU_RETURN.zip';publish_zip(archive,out,sorted(files),final_check);verify_archive(archive,files,final_check)
    write(out/'OUTPUT_MANIFEST.json',dict(files=files,archive_sha256=sha(archive,final_check),batch_sha256=batch_sha,independent_acceptance=False));final_check()

