"""D6 saved-design CPU scoring. No optimizer/model/noise execution or issuer.

The external host supplies accepted runtime/compatibility/closure/permit bytes.
Producer phases remain immutable; completed views refer back to their held phase.
"""
from pathlib import Path
from .bounded_stage import admitted_deadline
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib,json,os,shutil,time,math
from msdev.records import read,write,sha,safe_name
from .graph import identity
from .marking import finite
from .scientific_admission import require as require_accepted
from .session import closure_ok

STATE='designs_ready_pending_scoring'
_context=ContextVar('D6_phase',default=None)
@contextmanager
def phase_context(root,binding,*,producer_readback=False):
    token=_context.set(dict(root=str(Path(root)),binding=binding,producer_readback=producer_readback))
    try:yield
    finally:_context.reset(token)
def producer_readback():return bool((_context.get() or {}).get('producer_readback'))

def validate_policy(p,*,deferred=False):
    """Structural admission only; runtime observation occurs in the CPU child.

    deferred=True admits the immutable producer-side policy before the producer
    and CPU runtimes exist: BOTH identities null plus complete source pins. Such a
    request can only be scored/merged through explicitly accepted permit and
    compatibility records (scoring_identities/return_readback); never by default.
    """
    if (not isinstance(p,dict) or set(p)!={'runtime_sha256','compatibility_sha256','source_pins'}
            or not isinstance(p['source_pins'],dict) or not p['source_pins']):raise ValueError('separately reviewed CPU runtime/source/compatibility policy required')
    from .marking import identity as digest
    if not (deferred and p['runtime_sha256'] is None and p['compatibility_sha256'] is None):
        digest(p['runtime_sha256']);digest(p['compatibility_sha256'])
    for n,h in p['source_pins'].items():safe_name(n);digest(h)
    repo=Path(__file__).resolve().parents[2]
    from .backend import REQUIRED_SOURCES
    required={x.relative_to(repo).as_posix() for x in (repo/'operatormark_numerics_v2').glob('*.py')}
    required.update(REQUIRED_SOURCES)
    required.update(('tools/q6_common.py','tools/msdev/scoring.py','tools/msdev/rules.py',
        'tools/msfinal/cpu_scoring.py','tools/msfinal/cpu_score_entry.py'))
    if not required<=p['source_pins'].keys():raise ValueError('complete CPU numerical dependency pins required')
    return p

def scoring_identities(q,runtime,compatibility):
    """CPU runtime/compatibility identities required before scoring a request.

    A concrete request policy must match exactly. A deferred request policy takes
    them only from the explicitly passed accepted records; callers still require
    the permit and compatibility acceptance bound to these exact identities.
    """
    p=validate_policy(q['cpu_policy'],deferred=True)
    if p['runtime_sha256'] is None:return runtime.sha256,compatibility.sha256
    if runtime.sha256!=p['runtime_sha256'] or compatibility.sha256!=p['compatibility_sha256']:raise ValueError('CPU runtime/compatibility substitution')
    return p['runtime_sha256'],p['compatibility_sha256']

def returned_identities(q,out,expected,b):
    """Identities a returned scoring output must carry; deferred ones re-admitted from returned bytes."""
    from operatormark.corpus.state_v2_prerequisites import AcceptedRecord
    p=validate_policy(q['cpu_policy'],deferred=True)
    if p['runtime_sha256'] is not None:return p['runtime_sha256'],p['compatibility_sha256']
    rt,ct=b.get('cpu_runtime_sha256'),b.get('compatibility_sha256');identity(rt);identity(ct)
    def record(name):
        raw=(Path(out)/name).read_bytes();return AcceptedRecord(Path(out)/name,hashlib.sha256(raw).hexdigest(),raw)
    require_accepted(record('PERMIT_RECORD.json'),kind='main_study_D6_CPU_scoring',bindings=dict(request_sha256=expected,
        cpu_runtime_sha256=rt,compatibility_sha256=ct))
    require_accepted(record('COMPATIBILITY_RECORD.json'),kind='main_study_D6_numerical_compatibility',bindings=dict(
        producer_runtime_payload_sha256=q['binding']['producer_runtime_payload_sha256'],cpu_runtime_sha256=rt,source_pins=p['source_pins']))
    return rt,ct

def policy(admission):
    c=_context.get()
    if not c:raise ValueError('D6 durable phase binding required before inverse work')
    return validate_policy(admission.configuration.get('cpu_scoring'),deferred=True)

def monitor_policy():
    c=_context.get()
    if not c:raise ValueError('D6 bound monitor phase required before training')
    return validate_policy(c['binding'].get('cpu_policy'),deferred=True)

def population(q):return 16 if q.get('kind')=='monitor' else 64

def limitations(row):
    return {k:row[k] for k in ('historical_coverage','d8_policy','physical_verdicts','admission_timing') if k in row}

def valid_metric(value,monitor):
    if monitor:return finite(value) and value>=0
    return isinstance(value,dict) and set(value)=={'relative','absolute','truth_norm'} and all(finite(v) and v>=0 for v in value.values())

def freeze_monitor(out,bundle,forward,jvp,trace,updates,seconds,p,config,data,node,check):
    """D9's existing sixteen-target monitor, before its unchanged native metric."""
    import numpy as np
    from q6_common import array_save
    out=Path(out);c=_context.get()
    if not c:raise ValueError('D6 monitor phase required')
    policy_value=validate_policy(c['binding'].get('cpu_policy'),deferred=True)
    targets={}
    for label,suffix in [('targets','f64'),('targets_p256','p256')]:
        names=[f'tasks/target_{i:02d}_{suffix}.npy' for i in range(16)]
        sources={n:data['monitor_files'][n]['sha256'] for n in names}
        if any(sha(Path(bundle)/n)!=h for n,h in sources.items()):raise ValueError('monitor target binding drift')
        path=out/('cpu_'+label+'.npy')
        array_save(path,np.stack([np.load(Path(bundle)/n,allow_pickle=False) for n in names]))
        targets[label]=dict(file=path.name,source_files=sources,source_sha256=sha(path))
    write(out/'inverse_00.json',dict(updates=120,batch_size=16,batch_loss_divisor=16,trace=trace))
    value=dict(complete=False,task_status=STATE,model_updates=updates,forward=forward,jvp=jvp,
        task_f64=[],task_p256=[],means=dict(forward=sum(forward)/len(forward),jvp=sum(jvp)/len(jvp),task_f64=None),
        score=None,inverse_steps=120,batch_loss_divisor=16,inverse_precision='float32 pilot model; true solver scoring float64',
        inverse_loss_trace=trace,seconds=seconds,physical_claims='none')
    files={x.name:dict(bytes=x.stat().st_size,sha256=sha(x,check)) for x in out.iterdir() if x.is_file()}
    binding=dict(c['binding'],owner=p.owner_id,model_sha256=sha(out.parent/(node['id']+'.model.pt')),
        checkpoint=node['id'],phase='cooled_monitor',producer_runtime_sha256=p.runtime.sha256,
        producer_runtime_payload_sha256=data['entries']['runtime']['sha256'],
        assets_sha256=p.data.sha256,source_pins=config['source_pins'],calibration_target_indices=config['monitor_target_indices'],limitations=limitations(data))
    request=dict(schema='D6.saved-designs.v1',kind='monitor',binding=binding,cpu_policy=policy_value,
        files=files,targets=targets,target_order=list(range(16)),
        design_batches=[dict(file='designs.npy',trace='inverse_00.json',indices=list(range(16)))],
        pending_result=value,metric='operatormark_numerics_v2.task.score; unchanged Q6 monitor',
        operations=dict(T128_scoring=16,P256_scoring=16,inverse_updates=0,model_updates=0,noise_draws=0))
    write(out/'D6_REQUEST.json',request)
    value['cpu_request']=dict(directory=out.relative_to(c['root']).as_posix(),sha256=sha(out/'D6_REQUEST.json'))
    write(out/'RESULT.json',value);return value

def members(value):
    found={}
    def walk(v):
        if isinstance(v,dict):
            if 'cpu_request' in v:
                r=v['cpu_request'];key=r['directory']
                if key!='.':safe_name(key)
                if key in found and found[key]!=r:raise ValueError('conflicting CPU request')
                found[key]=r
            for k,x in v.items():
                if k!='cpu_request':walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(value);return found

def freeze(out,result,admission,phase,model_sha,wrapper,check):
    from . import materials as A
    out=Path(out);c=_context.get();p=policy(admission)
    result=dict(result,**limitations(admission.inputs))
    directory=out.relative_to(Path(c['root'])).as_posix()
    if directory!='.':safe_name(directory)
    targets={}
    for key in ('targets','targets_p256'):
        if key=='targets_p256' and not result['specification']['p256']:continue
        name=phase+'_'+key;raw=A.read_bytes(admission,name,phase)
        path=out/('cpu_'+key+'.npy')
        with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        targets[key]=dict(source_sha256=A.catalog(admission)['entries'][name]['sha256'],file=path.name)
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p,check)) for p in out.iterdir() if p.is_file()}
    b=dict(c['binding'],owner=admission.prerequisites.owner_id,model_sha256=model_sha,
        checkpoint=c['binding']['phase_name'],phase=phase,wrapper_steps=wrapper,
        producer_runtime_sha256=admission.prerequisites.runtime.sha256,
        producer_runtime_payload_sha256=admission.inputs['materials']['runtime']['sha256'],
        assets_sha256=admission.prerequisites.assets.sha256,source_pins=admission.configuration['source_pins'],limitations=limitations(admission.inputs))
    request=dict(schema='D6.saved-designs.v1',binding=b,cpu_policy=p,files=files,targets=targets,
        target_order=list(range(64)),design_batches=[dict(file=f'designs_{i:02d}.npy',trace=f'inverse_{i:02d}.json',indices=list(range(i,i+16))) for i in range(0,64,16)],
        pending_result=result,metric='msdev.scoring.relative_rows; physical field units; unchanged hard gates',
        operations=dict(T128_scoring=64,P256_scoring=64 if result['specification']['p256'] else 0,
            inverse_updates=0,model_updates=0,noise_draws=0))
    write(out/'D6_REQUEST.json',request)
    marker=dict(directory=directory,sha256=sha(out/'D6_REQUEST.json'))
    return dict(result,cpu_request=marker)

def verify_request(root,expected,check=lambda:None):
    root=Path(root);check()
    if sha(root/'D6_REQUEST.json',check)!=expected:raise ValueError('CPU request identity drift')
    q=read(root/'D6_REQUEST.json')
    n=population(q);monitor=q.get('kind')=='monitor'
    if q.get('kind') not in (None,'monitor') or q.get('schema')!='D6.saved-designs.v1' or q.get('target_order')!=list(range(n)):
        raise ValueError('CPU request/order incomplete')
    batches=([dict(file='designs.npy',trace='inverse_00.json',indices=list(range(16)))] if monitor else
        [dict(file=f'designs_{i:02d}.npy',trace=f'inverse_{i:02d}.json',indices=list(range(i,i+16))) for i in range(0,64,16)])
    if q['design_batches']!=batches:raise ValueError('design/target batch correspondence')
    pending=q['pending_result'];s=pending.get('specification',dict(p256=True))
    if pending.get('complete') is not False or pending.get('task_status')!=STATE:raise ValueError('pending state invalid')
    if not monitor and (
            pending['counts']['inverse_input_iterations']!=480
            or any(pending['counts'][k]!=0 for k in ('native_t128_calls','native_p256_calls','targets'))
            or pending['rows']['T'] or pending['rows']['P256']):raise ValueError('invalid pending scoring population')
    required={'cpu_targets.npy'} if monitor else {'START.json','DETECTOR_SCORES.json','cpu_targets.npy'}
    if s['p256']:required.add('cpu_targets_p256.npy')
    if not monitor:required.update(f'validation_{i:04d}.json' for i in range(0,s['samples'],16))
    if q['operations']!=dict(T128_scoring=n,P256_scoring=n if s['p256'] else 0,inverse_updates=0,model_updates=0,noise_draws=0):raise ValueError('scoring-only operation binding')
    required.update(n for b in q['design_batches'] for n in (b['file'],b['trace']))
    if not required<=q['files'].keys():raise ValueError('complete saved designs/diagnostics/inputs required')
    for name,record in q['files'].items():
        check();safe_name(name);p=root/name
        if p.is_symlink() or p.stat().st_size!=record['bytes'] or sha(p,check)!=record['sha256']:raise ValueError('persisted scoring input drift: '+name)
    for b in q['design_batches']:
        t=read(root/b['trace'])
        if t.get('updates')!=120 or t.get('batch_size')!=16 or t.get('batch_loss_divisor')!=16 or len(t.get('trace',[]))!=120 or not all(finite(x) for x in t['trace']):raise ValueError('inverse trace incomplete')
    for target,e in q['targets'].items():
        if q['files'][e['file']]['sha256']!=e['source_sha256']:raise ValueError('target identity correspondence')
    return q

def pending_readback(root,check=lambda:None):
    root=Path(root);r=read(root/'RESULT.json');m=r.get('cpu_request',{})
    q=verify_request(root,m.get('sha256'),check)
    if r!=dict(q['pending_result'],cpu_request=m):raise ValueError('pending evaluation changed')
    if q.get('kind')=='monitor':
        if (len(r['forward'])!=256 or len(r['jvp'])!=512 or r['task_f64'] or r['task_p256']
            or r['inverse_loss_trace']!=read(root/'inverse_00.json')['trace']
            or any(not finite(v) or v<0 for v in r['forward']+r['jvp'])
            or r['means']!=dict(forward=sum(r['forward'])/256,jvp=sum(r['jvp'])/512,task_f64=None)):
            raise ValueError('pending monitor incomplete')
        return dict(verified=True,designs_ready=True,scoring_complete=False)
    from msdev.rules import complete_mean
    for k in 'FJK':
        rows=[]
        for lo in range(0,r['specification']['samples'],16):
            batch=read(root/f'validation_{lo:04d}.json')[k]
            if len(batch)!=(32 if k=='J' else 16):raise ValueError('pending diagnostic population')
            rows+=batch
        if rows!=r['rows'][k] or complete_mean(rows)!=r['metrics'][k] or not finite(r['metrics'][k]):raise ValueError('pending diagnostic drift')
    scores=read(root/'DETECTOR_SCORES.json')
    if scores['scores']!=r['scores'] or len(r['scores'])!=scores['wrong_key_count']+1 or not all(finite(x) for x in r['scores']):raise ValueError('pending detector incomplete')
    return dict(verified=True,designs_ready=True,scoring_complete=False)

def runtime_check(record,root,pins):
    from operatormark_numerics_v2.scope import determinism_scope
    for n,h in pins.items():
        safe_name(n)
        if sha(Path(root)/n)!=h:raise ValueError('CPU numerical/source drift: '+n)
        if sha(Path(__file__).resolve().parents[2]/n)!=h:raise ValueError('loaded CPU source differs: '+n)
    observed=determinism_scope()
    if observed!=record['scope']:raise ValueError('actual pinned CPU numerical runtime differs')
    return observed

class Kernels:
    """Only saved arrays, native forwards and the unchanged metric."""
    def load(self,path,shape):
        import numpy as np
        a=np.load(path,allow_pickle=False)
        if a.shape!=shape or str(a.dtype)!='float64' or not np.isfinite(a).all():raise ValueError('CPU scoring array shape/dtype/nonfinite')
        return a
    def score(self,rho,target,p256,*,monitor=False):
        import numpy as np
        from q6_common import true_field
        # scoring.relative_rows is this exact imported function; avoid importing
        # the unrelated torch detector implementation in a scoring-only worker.
        from msdev.rules import relative_rows
        field=np.asarray(true_field(rho,p256))
        if monitor:
            from operatormark_numerics_v2.task import score
            return score(field,target)
        return relative_rows(field.reshape(1,-1),target.reshape(1,-1))[0]

def score(root,expected,out,permit,runtime,compatibility,check,*,source_root,runtime_observer=runtime_check,kernels=None,clock=time.time):
    started=time.monotonic()
    root=Path(root);out=Path(out);q=verify_request(root,expected,check);p=q['cpu_policy']
    rt,ct=scoring_identities(q,runtime,compatibility)
    grant=require_accepted(permit,kind='main_study_D6_CPU_scoring',bindings=dict(request_sha256=expected,
        cpu_runtime_sha256=rt,compatibility_sha256=ct))
    if (grant.get('domain')!='final' or grant.get('device')!='CPU' or grant.get('retries')!=0
            or not finite(grant.get('deadline')) or grant['deadline']>admitted_deadline(q['binding'],grant)
            or grant.get('operations')!=q['operations']):raise ValueError('CPU scope/deadline/count admission')
    if Path(grant['scoring_output']).resolve()!=out.resolve() or Path(grant['source_root']).resolve()!=Path(source_root).resolve():raise ValueError('CPU output/source binding substitution')
    rv=runtime.read();cv=require_accepted(compatibility,kind='main_study_D6_numerical_compatibility',bindings=dict(
        producer_runtime_payload_sha256=q['binding']['producer_runtime_payload_sha256'],cpu_runtime_sha256=runtime.sha256,source_pins=p['source_pins']))
    if cv.get('scope')!='saved_float64_design_T128_P256_relative_rows_and_Q6_task_score' or not cv.get('evidence_files'):raise ValueError('tested numerical compatibility scope missing')
    for name,h in cv['evidence_files'].items():
        if sha(name)!=h:raise ValueError('compatibility evidence drift')
    def bounded():
        check()
        if clock()>=grant['deadline']:raise TimeoutError('original bounded CPU scoring deadline')
    bounded();observed=runtime_observer(rv,source_root,p['source_pins']);out.mkdir(exist_ok=False)
    counts=dict.fromkeys(q['operations'],0);rows=[];k=Kernels() if kernels is None else kernels
    write(out/'BINDINGS.json',dict(request_sha256=expected,cpu_runtime_sha256=runtime.sha256,
        compatibility_sha256=compatibility.sha256,permit_sha256=permit.sha256,producer=q['binding']))
    write(out/'RUNTIME.json',observed)
    with (out/'CPU_RUNTIME_RECORD.json').open('xb') as f:f.write(runtime.expected)
    with (out/'COMPATIBILITY_RECORD.json').open('xb') as f:f.write(compatibility.expected)
    with (out/'PERMIT_RECORD.json').open('xb') as f:f.write(permit.expected)
    try:
        n=population(q);monitor=q.get('kind')=='monitor'
        target=k.load(root/q['targets']['targets']['file'],(n,64,64))
        pt=k.load(root/q['targets']['targets_p256']['file'],(n,64,64)) if q['operations']['P256_scoring'] else None
        for batch in q['design_batches']:
            designs=k.load(root/batch['file'],(16,8,8))
            for index,rho in zip(batch['indices'],designs):
                bounded();counts['T128_scoring']+=1;write(out/f'intent_{index:02d}_T128.json',dict(target=index,counts=counts))
                row=dict(target=index,F64=k.score(rho,target[index],False,monitor=monitor))
                if pt is not None:
                    bounded();counts['P256_scoring']+=1;write(out/f'intent_{index:02d}_P256.json',dict(target=index,counts=counts))
                    row['P256']=k.score(rho,pt[index],True,monitor=monitor)
                write(out/f'task_{index:02d}.json',row);rows.append(row)
                if not all(valid_metric(row[key],monitor) for key in ('F64','P256') if key in row):raise ValueError('native score unavailable/nonfinite; retained without completion')
        verify_request(root,expected,bounded)
        if runtime_observer(rv,source_root,p['source_pins'])!=observed:raise ValueError('CPU runtime changed during scoring')
        if counts!=q['operations']:raise ValueError('CPU operation reconciliation')
        result=dict(complete=True,request_sha256=expected,counts=counts,rows=rows,cpu_runtime_sha256=runtime.sha256,
            compatibility_sha256=compatibility.sha256,producer_binding=q['binding'],physical_qualification=False,
            scoring_elapsed_seconds=time.monotonic()-started,elapsed_scope='child verification/runtime/native scoring; supervisor closure and export separate')
        write(out/'RESULT.json',result)
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p,bounded)) for p in out.iterdir() if p.is_file()}
        write(out/'RETURN_MANIFEST.json',dict(files=files,request_sha256=expected));return result
    except BaseException as exc:
        try:write(out/'FAILURE.json',dict(error=repr(exc),counts=counts,rows=rows,complete=False))
        except BaseException as reporting:exc.add_note('CPU failure evidence unavailable: '+repr(reporting))
        raise

def return_readback(root,expected,out,acceptance,closure,check=lambda:None,*,observations=None):
    root=Path(root);out=Path(out);q=verify_request(root,expected,check)
    accepted=require_accepted(acceptance,kind='main_study_D6_CPU_return',bindings=dict(request_sha256=expected,
        manifest_sha256=sha(out/'RETURN_MANIFEST.json'),closure_sha256=closure.sha256))
    closed=closure.read()
    q0_closed=(closed.get('schema')=='D6.Q0.closed.v1' and closed.get('child_reaped') is True
        and closed.get('active_processes')==0 and closed.get('survivors') is False
        and closed.get('resolved') is True and closed.get('job_handle_closed') is True
        and closed.get('output_readback') is True)
    if q0_closed:
        observations=Path(observations) if observations is not None else out.parent
        raw_path=observations/'SUPERVISOR_RESULT.json';log=observations/'supervisor/pytest_output.txt'
        if sha(raw_path,check)!=closed.get('supervisor_sha256') or sha(log,check)!=closed.get('combined_stream_sha256'):raise ValueError('CPU raw closure/stream identity differs')
        raw=read(raw_path);c=raw.get('cleanup',{})
        if (raw.get('exit_code')!=0 or raw.get('timed_out') is not False or raw.get('job_handle_closed') is not True
            or c.get('child_reaped') is not True or c.get('resolved') is not True or c.get('group_survivors') is not False
            or c.get('job_accounting',{}).get('active_processes')!=0 or c.get('errors')):raise ValueError('CPU raw owned-process closure incomplete')
    if accepted.get('domain')!='final' or not (closure_ok(closed) or q0_closed) or closed.get('timed_out') is not False or closed.get('returncode')!=0 or closed.get('result_sha256')!=sha(out/'RESULT.json'):raise ValueError('independent CPU closure absent/incomplete')
    manifest=read(out/'RETURN_MANIFEST.json');files=manifest['files']
    if {p.name for p in out.iterdir()}!=set(files)|{'RETURN_MANIFEST.json'}:raise ValueError('CPU return membership differs')
    for name,r in files.items():
        safe_name(name);p=out/name;check()
        if p.is_symlink() or p.stat().st_size!=r['bytes'] or sha(p,check)!=r['sha256']:raise ValueError('CPU return bytes drift')
    result=read(out/'RESULT.json');b=read(out/'BINDINGS.json');rt,ct=returned_identities(q,out,expected,b)
    if sha(out/'CPU_RUNTIME_RECORD.json')!=rt or read(out/'RUNTIME.json')!=read(out/'CPU_RUNTIME_RECORD.json')['scope']:raise ValueError('actual CPU runtime return mismatch')
    if (sha(out/'COMPATIBILITY_RECORD.json')!=ct
        or sha(out/'PERMIT_RECORD.json')!=b.get('permit_sha256')
        or b.get('cpu_runtime_sha256')!=rt
        or b.get('compatibility_sha256')!=ct):raise ValueError('CPU returned authority/compatibility record drift')
    if (result.get('complete') is not True or result.get('request_sha256')!=expected
        or result.get('counts')!=q['operations'] or result.get('producer_binding')!=q['binding']
        or result.get('cpu_runtime_sha256')!=rt
        or result.get('compatibility_sha256')!=ct
        or b.get('request_sha256')!=expected or b.get('producer')!=q['binding'] or len(result.get('rows',[]))!=population(q)):raise ValueError('CPU result binding/population differs')
    for i,row in enumerate(result['rows']):
        if row.get('target')!=i or read(out/f'task_{i:02d}.json')!=row:raise ValueError('CPU task order/raw result differs')
        for key in ('F64','P256') if q['operations']['P256_scoring'] else ('F64',):
            if not valid_metric(row[key],q.get('kind')=='monitor'):raise ValueError('CPU metric incomplete/nonfinite')
        for family in ('T128','P256') if q['operations']['P256_scoring'] else ('T128',):
            intent=read(out/f'intent_{i:02d}_{family}.json')
            expected_counts=dict(T128_scoring=i+1,P256_scoring=(i+1 if family=='P256' else i) if q['operations']['P256_scoring'] else 0,inverse_updates=0,model_updates=0,noise_draws=0)
            if intent!=dict(target=i,counts=expected_counts):raise ValueError('CPU attempt count evidence differs')
    return result

def merge(root,expected,out,returned,check=lambda:None):
    """Verified metadata/file merge, never recomputation or overwrite."""
    from msdev.rules import complete_mean
    root=Path(root);out=Path(out);q=verify_request(root,expected,check)
    result=return_readback(root,expected,returned['root'],returned['acceptance'],returned['closure'],check,observations=returned.get('observations'))
    out.mkdir(parents=True,exist_ok=False)
    for n in q['files']:
        check();shutil.copyfile(root/n,out/n)
        if sha(out/n)!=q['files'][n]['sha256']:raise ValueError('design/diagnostic transfer drift')
    shutil.copyfile(root/'D6_REQUEST.json',out/'D6_REQUEST.json')
    return_copy=out/'CPU_RETURN';return_copy.mkdir()
    for p in Path(returned['root']).iterdir():
        check();shutil.copyfile(p,return_copy/p.name)
        if sha(p)!=sha(return_copy/p.name):raise ValueError('CPU return transfer drift')
    for key in ('acceptance','closure'):
        with (out/('CPU_'+key.upper()+'.json')).open('xb') as f:f.write(returned[key].expected)
    if returned['closure'].read().get('schema')=='D6.Q0.closed.v1':
        observation=out/'CPU_OBSERVATION';(observation/'supervisor').mkdir(parents=True)
        for n in ('SUPERVISOR_RESULT.json','supervisor/pytest_output.txt'):
            source=Path(returned.get('observations',Path(returned['root']).parent))/n;shutil.copyfile(source,observation/n)
            if sha(source,check)!=sha(observation/n,check):raise ValueError('CPU closure evidence transfer drift')
    value=json.loads(json.dumps(q['pending_result']))
    monitor=q.get('kind')=='monitor'
    if monitor:
        value['task_f64']=[r['F64'] for r in result['rows']];value['task_p256']=[r['P256'] for r in result['rows']]
        value['means']['task_f64']=sum(value['task_f64'])/16
        means=list(value['means'].values())
        value['score']=0. if 0. in means else math.exp(sum(math.log(a) for a in means)/3)
    else:
        value['rows']['T']=[r['F64'] for r in result['rows']]
        value['rows']['P256']=[r['P256'] for r in result['rows']] if q['operations']['P256_scoring'] else []
        value['metrics']['T']=complete_mean(value['rows']['T'])
        if q['operations']['P256_scoring']:value['metrics']['P256']=complete_mean(value['rows']['P256'])
        value['counts'].update(targets=64,native_t128_calls=64,native_p256_calls=q['operations']['P256_scoring'])
    value.update(complete=True,task_status='measured',cpu_return=dict(request_sha256=expected,
        manifest_sha256=sha(Path(returned['root'])/'RETURN_MANIFEST.json'),acceptance_sha256=returned['acceptance'].sha256,
        closure_sha256=returned['closure'].sha256,cpu_runtime_sha256=result['cpu_runtime_sha256']))
    value['D6_timing']=dict(producer_seconds=value.get('seconds'),CPU_scoring_seconds=result['scoring_elapsed_seconds'],
        scope='separate clocks; transfer/setup/closure and paused elapsed time are additional; not a GPU speedup measurement')
    for row in result['rows']:
        saved=dict(target=row['target'],f64_error=row['F64'],p256_error=row['P256'],hard_gates_pass=True) if monitor else row
        write(out/f"task_{row['target']:02d}.json",saved)
    write(out/'RESULT.json',value)
    from .readback import evaluation
    if monitor:verify_merged(out,value,check)
    else:evaluation(out,value['specification'],check)
    return value

def verify_merged(out,value,check=lambda:None):
    """Independent later readbacks repeat binding/byte/row checks, not solves."""
    from operatormark.corpus.state_v2_prerequisites import AcceptedRecord
    out=Path(out);binding=value['cpu_return'];records={}
    for key in ('acceptance','closure'):
        path=out/('CPU_'+key.upper()+'.json');raw=path.read_bytes()
        records[key]=AcceptedRecord(path,binding[key+'_sha256'],raw)
    result=return_readback(out,binding['request_sha256'],out/'CPU_RETURN',records['acceptance'],records['closure'],check,observations=out/'CPU_OBSERVATION')
    q=read(out/'D6_REQUEST.json')
    if sha(out/'CPU_RETURN/RETURN_MANIFEST.json')!=binding['manifest_sha256']:raise ValueError('merged CPU manifest drift')
    if q.get('kind')=='monitor':
        if value['task_f64']!=[r['F64'] for r in result['rows']] or value['task_p256']!=[r['P256'] for r in result['rows']]:raise ValueError('merged monitor scores differ')
        for row in result['rows']:
            if read(out/f"task_{row['target']:02d}.json")!=dict(target=row['target'],f64_error=row['F64'],p256_error=row['P256'],hard_gates_pass=True):raise ValueError('merged monitor raw row differs')
        means=dict(forward=sum(value['forward'])/256,jvp=sum(value['jvp'])/512,task_f64=sum(value['task_f64'])/16)
        score=0. if 0. in means.values() else math.exp(sum(math.log(a) for a in means.values())/3)
        if value['means']!=means or value['score']!=score:raise ValueError('merged monitor selection aggregate differs')
    else:
        expected_p=[r['P256'] for r in result['rows']] if q['operations']['P256_scoring'] else []
        if value['rows']['T']!=[r['F64'] for r in result['rows']] or value['rows']['P256']!=expected_p:raise ValueError('merged CPU scoring rows differ')
    return result

def export_request(root,expected,destination,phase_receipt,worker_result,closure,permit,check=lambda:None,clock=time.time):
    """Small scoring-input archive after real producer closure; no model/noise copy."""
    from msdev.records import publish_zip
    from .phase_storage import verify_archive
    root=Path(root);destination=Path(destination);q=verify_request(root,expected,check)
    receipt=read(phase_receipt);closed=closure.read()
    markers=members(receipt['value'])
    if expected not in {m['sha256'] for m in markers.values()} or not closure_ok(closed) or closed.get('product_readback') is not True:
        raise ValueError('independently closed producer phase required')
    saved=read(worker_result)
    if closed.get('worker_result_sha256')!=sha(worker_result) or saved.get('product_receipts',{}).get(receipt['name'])!=sha(phase_receipt):raise ValueError('closure does not bind producer receipt')
    grant=require_accepted(permit,kind='main_study_D6_input_export',bindings=dict(request_sha256=expected,
        phase_receipt_sha256=sha(phase_receipt),closure_sha256=closure.sha256,destination=str(destination.resolve())))
    if grant.get('domain')!='final' or grant.get('deadline',float('inf'))>admitted_deadline(q['binding'],grant):raise ValueError('export deadline/domain')
    def bounded():
        check()
        if clock()>=grant['deadline']:raise TimeoutError('D6 export immutable deadline')
    bounded();destination.mkdir(parents=True,exist_ok=False)
    files=dict(q['files']);files['D6_REQUEST.json']=dict(bytes=(root/'D6_REQUEST.json').stat().st_size,sha256=expected)
    archive_bound=sum(r['bytes'] for r in files.values())+sum(200+2*len(n.encode()) for n in files)+1024
    if shutil.disk_usage(destination).free<archive_bound:raise OSError('CPU input archive capacity unavailable')
    archive=destination/'CPU_INPUT.zip';publish_zip(archive,root,sorted(files),bounded);verify_archive(archive,files,bounded)
    verify_request(root,expected,bounded)
    result=dict(files=files,request_sha256=expected,archive_sha256=sha(archive,bounded),archive_bytes=archive.stat().st_size,
        phase_receipt_sha256=sha(phase_receipt),closure_sha256=closure.sha256,permit_sha256=permit.sha256)
    write(destination/'CPU_INPUT_MANIFEST.json',result);return result

def import_bundle(archive,manifest,expected,destination,*,file_limit,byte_limit,check=lambda:None):
    """Same exact-membership readback for a CPU input or return transport ZIP.

    The caller supplies the independently selected manifest identity and its
    existing storage/deadline policy. This is not an acceptance or a new grant.
    """
    import zipfile
    from .phase_storage import verify_archive
    archive=Path(archive);manifest=Path(manifest);dest=Path(destination)
    if sha(manifest,check)!=expected:raise ValueError('transport manifest substitution')
    m=read(manifest);files=m['files']
    if (type(file_limit) is not int or file_limit<=0 or type(byte_limit) is not int or byte_limit<=0
        or len(files)>file_limit or sum(r['bytes'] for r in files.values())>byte_limit):raise ValueError('explicit transfer capacity exceeded')
    if sha(archive,check)!=m['archive_sha256']:raise ValueError('transport archive drift')
    verify_archive(archive,files,check)
    dest.mkdir(parents=True,exist_ok=False);payload=dest/'payload';payload.mkdir()
    if shutil.disk_usage(dest).free<sum(r['bytes'] for r in files.values()):raise OSError('extracted payload capacity unavailable')
    with zipfile.ZipFile(archive) as z:
        for n,r in files.items():
            check();safe_name(n);path=payload/n;path.parent.mkdir(parents=True,exist_ok=True)
            with z.open(n) as source,path.open('xb') as target:
                while chunk:=source.read(1<<20):check();target.write(chunk)
                target.flush();os.fsync(target.fileno())
            if path.stat().st_size!=r['bytes'] or sha(path,check)!=r['sha256']:raise ValueError('extracted CPU bytes differ')
    if sha(manifest,check)!=expected or sha(archive,check)!=m['archive_sha256']:raise ValueError('transport changed during readback')
    write(dest/'IMPORT.json',dict(manifest_sha256=expected,archive_sha256=m['archive_sha256'],files=files,independent_acceptance=False))
    return payload

# Phase adapters reuse the existing Worker receipt/export/session machinery.
def source_root(out):
    p=Path(out)/'D6_ORIGIN.json'
    if not p.exists():return Path(out)
    r=read(p)
    if sha(r['receipt'])!=r['receipt_sha256']:raise ValueError('held producer receipt drift')
    return Path(r['root'])

def finish_phase(source,out,value,returned,receipt,check,finalize=None):
    source=Path(source);out=Path(out);requested=members(value)
    if set(returned)!=set(requested):raise ValueError('missing/extra CPU return membership')
    if out.exists():raise ValueError('fresh CPU-completed phase required')
    replacements={}
    for directory,m in requested.items():
        replacements[m['sha256']]=merge(source/directory,m['sha256'],out/directory,returned[directory],check)
    def substitute(v):
        if isinstance(v,dict):
            if v.get('task_status')==STATE:return dict(replacements[v['cpu_request']['sha256']],**{k:x for k,x in v.items() if k in ('noise','white_box_only')})
            return {k:substitute(x) for k,x in v.items()}
        if isinstance(v,list):return [substitute(x) for x in v]
        return v
    result=substitute(value)
    write(out/'D6_ORIGIN.json',dict(root=str(source),receipt=str(receipt),receipt_sha256=sha(receipt)))
    if finalize is not None:result=finalize(source,out,replacements,result)
    if (source/'RESULT.json').exists() and not (out/'RESULT.json').exists():
        original=read(source/'RESULT.json');write(out/'RESULT.json',substitute(original))
    return result
