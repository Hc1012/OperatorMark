"""Disabled unless explicitly invoked with --execute and externally approved pins.

Windows CPU only; delegates ownership/deadlines to the already verified Q0 Job
supervisor. No reservation, permission issuer, package setup or retry is created.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from msdev.records import read,write,sha
from msfinal import cpu_scoring as C
from operatormark.corpus.state_v2_prerequisites import AcceptedRecord,require_accepted
Q0_SHA='6493f6823ff2e357ac80b60ee1ef8117009262ec54bda189468224d71a3394c9'

def record(path,expected):
    path=Path(path);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('externally selected record identity differs')
    return AcceptedRecord(path,expected,raw)

def persist_closure(path,value):
    # AcceptedRecord reads canonical bytes; ordinary pretty records.write output
    # is intentionally not substituted for this independent-readback interface.
    from operatormark.corpus.canonical import canonical_json_bytes
    with Path(path).open('xb') as f:
        f.write(canonical_json_bytes(value));f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('request','request-sha','permit','permit-sha','runtime','runtime-sha','compatibility','compatibility-sha','source-root','output','supervisor'):p.add_argument('--'+n,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--owned-child',action='store_true')
    a=p.parse_args()
    if not a.execute:raise ValueError('UNSTARTED: explicit execution authority and --execute required')
    permit=record(a.permit,a.permit_sha);runtime=record(a.runtime,a.runtime_sha);compat=record(a.compatibility,a.compatibility_sha)
    q=C.verify_request(a.request,a.request_sha)
    grant=require_accepted(permit,kind='main_study_D6_CPU_scoring',bindings=dict(request_sha256=a.request_sha,
        cpu_runtime_sha256=a.runtime_sha,compatibility_sha256=a.compatibility_sha))
    from msfinal.bounded_stage import admitted_deadline
    if grant.get('retries')!=0 or grant.get('device')!='CPU' or grant['deadline']>admitted_deadline(q['binding'],grant):raise ValueError('scope/deadline drift')
    if str(Path(sys.executable).resolve())!=str(Path(grant['interpreter']).resolve()):raise ValueError('bound CPU interpreter required')
    def check():
        if time.time()>=grant['deadline']:raise TimeoutError('immutable CPU grant/original deadline')
        permit.read()
    check();out=Path(a.output)
    if (Path(grant['output_root']).resolve()!=out.resolve() or Path(grant['scoring_output']).resolve()!=(out/'worker').resolve()
        or Path(grant['source_root']).resolve()!=Path(a.source_root).resolve()):raise ValueError('single-use CPU output/source route differs')
    if a.owned_child:
        consumed=read(out/'CONSUMED.json')
        if consumed!=dict(permit_sha256=a.permit_sha,request_sha256=a.request_sha,deadline=grant['deadline'],retries=0):raise ValueError('owned parent consumption binding missing')
        C.score(a.request,a.request_sha,out/'worker',permit,runtime,compat,check,source_root=a.source_root);return
    if os.name!='nt':raise ValueError('this owned entry uses the verified Windows Q0 supervisor only')
    if sha(a.supervisor)!=Q0_SHA:raise ValueError('verified Q0 supervisor bytes required')
    work=grant['work_seconds'];cleanup=grant['cleanup_seconds'];finalization=grant['finalization_seconds']
    if any(type(v) not in (int,float) or v<=0 for v in (work,cleanup,finalization)) or time.time()+work+cleanup+finalization>grant['deadline']:raise ValueError('full CPU envelope does not fit')
    env=dict(os.environ,**grant['environment']);env['PYTHONPATH']=os.pathsep.join([str(Path(a.source_root)/'tools'),a.source_root])
    from q6_common import THREAD_ENV
    if any(env.get(k)!=v for k,v in THREAD_ENV.items()):raise ValueError('pinned CPU thread/determinism environment required')
    out.mkdir(parents=True,exist_ok=False);(out/'supervisor').mkdir()
    write(out/'CONSUMED.json',dict(permit_sha256=a.permit_sha,request_sha256=a.request_sha,deadline=grant['deadline'],retries=0))
    spec=importlib.util.spec_from_file_location('q0',a.supervisor);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--owned-child']
    try:
        result=module.run_with_timeout(command,cwd=a.source_root,env=env,timeout_s=work,cleanup_timeout_s=cleanup,out_dir=out/'supervisor')
        final_deadline=min(grant['deadline'],time.time()+finalization)
        def final_check():
            check()
            if time.time()>=final_deadline:raise TimeoutError('CPU independent finalization bound')
        write(out/'SUPERVISOR_RESULT.json',result);final_check();c=result['cleanup']
        good=(result['exit_code']==0 and not result['timed_out'] and c['resolved'] and c['child_reaped']
            and c['group_survivors'] is False and c.get('job_accounting',{}).get('active_processes')==0 and result['job_handle_closed'])
        if not good:raise RuntimeError('CPU attempt failed/timed out or unresolved ownership; no retry')
        manifest=read(out/'worker/RETURN_MANIFEST.json')
        for name,row in manifest['files'].items():
            final_check();path=out/'worker'/name
            if path.stat().st_size!=row['bytes'] or sha(path,final_check)!=row['sha256']:raise ValueError('post-closure CPU output drift')
        if {x.name for x in (out/'worker').iterdir()}!=set(manifest['files'])|{'RETURN_MANIFEST.json'}:raise ValueError('post-closure return membership')
        persist_closure(out/'CLOSURE.json',dict(schema='D6.Q0.closed.v1',child_reaped=True,active_processes=0,survivors=False,
            resolved=True,job_handle_closed=True,output_readback=True,timed_out=False,returncode=0,
            result_sha256=sha(out/'worker/RESULT.json'),supervisor_sha256=sha(out/'SUPERVISOR_RESULT.json'),
            combined_stream_sha256=sha(out/'supervisor/pytest_output.txt'),deadline=grant['deadline'],independent_acceptance=False))
        from msdev.records import publish_zip
        from msfinal.phase_storage import verify_archive
        final_check()
        files={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p,final_check)) for p in out.rglob('*') if p.is_file()}
        # One return archive, including owned closure/logs; originals and failures
        # remain. This is transport verification, never independent acceptance.
        bound=sum(v['bytes'] for v in files.values())+sum(200+2*len(n.encode()) for n in files)+1024
        if __import__('shutil').disk_usage(out).free<bound:raise OSError('CPU return archive capacity unavailable')
        archive=out/'CPU_RETURN.zip';publish_zip(archive,out,sorted(files),final_check);verify_archive(archive,files,final_check)
        write(out/'OUTPUT_MANIFEST.json',dict(files=files,archive_sha256=sha(archive,final_check),request_sha256=a.request_sha,
            closure_sha256=sha(out/'CLOSURE.json'),independent_acceptance=False))
        final_check()
    except BaseException as exc:
        write(out/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),complete=False));raise

if __name__=='__main__':
    if '--batch' in sys.argv:
        from msfinal.cpu_batch_entry import main as batch_main
        batch_main()
    else:main()
