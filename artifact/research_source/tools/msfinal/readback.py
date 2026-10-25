"""Bounded-memory file readback, not tensor loading or scientific acceptance."""
from pathlib import Path
import hashlib
import os
import math
from msdev.records import read,sha,safe_name
from .update_journal import read_prefix


def tensor_file(path,expected,check,*,publication='D5.exclusive-hardlink.v1'):
    check();path=Path(path)
    if not path.is_file() or path.is_symlink() or sha(path,check)!=expected:
        raise ValueError('checkpoint/model bytes missing or changed: '+str(path))
    partial=path.with_name(path.name+'.partial')
    if publication=='legacy-retained-partial':
        if not partial.is_file() or sha(partial,check)!=expected:
            raise ValueError('retained checkpoint publication copy differs: '+str(path))
    elif publication=='D5.exclusive-hardlink.v1':
        receipt=read(path.with_name(path.name+'.publication.json'))
        expected_receipt=dict(schema=publication,name=path.name,sha256=expected,
            bytes=path.stat().st_size,exclusive=True,completed_tensor_files=1)
        if receipt!=expected_receipt or partial.exists():
            raise ValueError('single tensor publication receipt/membership differs: '+str(path))
    else:raise ValueError('unknown tensor publication contract')


def training(root,nodes,check=lambda:None,*,monitors=False):
    """Verify raw sequence, branch metadata, all checkpoints and durable anchors."""
    from . import cpu_scoring as CPU
    monitor_root=Path(root);root=CPU.source_root(monitor_root)
    bindings=read(root/'BINDINGS.json');result=read(monitor_root/'RESULT.json')
    if bindings.get('plan')!=nodes or result.get('complete') is not True:raise ValueError('incomplete/different trajectory')
    anchors=[];expected_products={};total=sum(n['end']-n['start'] for n in nodes)
    for node in nodes:
        check();record=read(root/(node['id']+'.json'))
        if record.get('node')!=node:raise ValueError('node/branch metadata differs')
        tensor_file(root/(node['id']+'.state.pt'),record['state_sha256'],check)
        anchors.append(record['journal'])
        for width in range(2048,node['end']-node['start']+1,2048):
            base=root/f"{node['id']}_{node['start']+width}.state.pt"
            anchor=read(base.with_suffix('.anchor.json'))
            tensor_file(base,anchor['state_sha256'],check);anchors.append(anchor['journal'])
        if node['evaluate']:
            product=result['products'][node['id']];path=root/(node['id']+'.model.pt')
            # Returned producer paths may be absolute in a different machine;
            # validate the named owned member, never follow that absolute path.
            if Path(product['path']).name!=path.name or product['updates']!=node['end'] or product['state_sha256']!=record['state_sha256']:
                raise ValueError('model/checkpoint correspondence differs')
            tensor_file(path,product['sha256'],check);expected_products[node['id']]=product
            if monitors and node['cooldown']:
                folder=monitor_root/(node['id']+'_monitor');m=read(folder/'RESULT.json')
                if m.get('task_status')==CPU.STATE:
                    if not CPU.producer_readback():raise ValueError('cooled monitor CPU scoring pending')
                    CPU.pending_readback(folder,check)
                    if product.get('monitor')!=m:raise ValueError('pending monitor product differs')
                    continue
                if (m.get('complete') is not True or m.get('model_updates')!=node['end']
                        or m.get('inverse_steps')!=120 or m.get('batch_loss_divisor')!=16
                        or any(len(m.get(k,[]))!=n for k,n in [('forward',256),('jvp',512),('task_f64',16),('task_p256',16),('inverse_loss_trace',120)])
                        or product.get('monitor')!=m):
                    raise ValueError('required cooled monitor missing/inconsistent')
                if not (folder/'designs.npy').is_file() or any(not (folder/f'task_{i:02d}.json').is_file() for i in range(16)):
                    raise ValueError('monitor design/task member missing')
                if 'cpu_return' in m:CPU.verify_merged(folder,m,check)
    if result.get('products')!=expected_products:raise ValueError('extra/missing trajectory product')
    if result.get('executed_updates')!=total or result.get('attempted_updates')!=total:
        raise ValueError('trajectory update count differs')
    cursor=iter((n,i) for n in nodes for i in range(n['start'],n['end']))
    current=None
    def row(value):
        nonlocal current
        check()
        if value['event']=='attempt':
            current=next(cursor,None)
            if current is None or value['node']!=current[0]['id'] or value['pre_update']!=current[1]:
                raise ValueError('raw journal differs from branch schedule')
        elif value['row']['update']!=current[1]+1:raise ValueError('raw completion differs')
    observed=read_prefix(root/'updates.jsonl',row)
    if not observed['complete'] or observed['attempted_updates']!=total or next(cursor,None) is not None:
        raise ValueError('incomplete update journal')
    anchors.append(result['journal'])
    h=hashlib.sha256();position=0
    with (root/'updates.jsonl').open('rb') as stream:
        for anchor in sorted(anchors,key=lambda a:a['acknowledged_bytes']):
            end=anchor['acknowledged_bytes']
            if (anchor['path']!='updates.jsonl' or anchor.get('write_failed') or anchor.get('pending_attempt') is not None
                    or anchor.get('uncommitted_record') is not None or end>observed['verified_prefix_bytes']):
                raise ValueError('unacknowledged/incomplete checkpoint journal anchor')
            while position<end:
                check();chunk=stream.read(min(1<<20,end-position))
                if not chunk:raise ValueError('short journal prefix')
                h.update(chunk);position+=len(chunk)
            if h.hexdigest()!=anchor['sha256']:raise ValueError('checkpoint journal prefix hash differs')
    final=result['journal']
    if (final['acknowledged_bytes']!=observed['verified_prefix_bytes'] or final['sha256']!=observed['sha256']
            or final['attempted_updates']!=total or final['completed_updates']!=total):
        raise ValueError('final journal receipt differs')
    return dict(verified=True,updates=total,journal=observed,products=expected_products,independent_acceptance=False)


def evaluation(root,spec,check=lambda:None,*,expected_domain='final'):
    root=Path(root);r=read(root/'RESULT.json');check()
    from . import cpu_scoring as CPU
    if r.get('task_status')==CPU.STATE:
        if not CPU.producer_readback():raise ValueError('required CPU scoring remains pending')
        return CPU.pending_readback(root,check)
    if expected_domain not in ('final','development') or r.get('domain')!=expected_domain:
        raise ValueError('evaluation domain missing or incompatible; development is not final acceptance')
    expected=dict(inverse_input_iterations=480,targets=64,native_t128_calls=64,
        native_p256_calls=64 if spec['p256'] else 0,validation_controls=spec['samples'],
        detector_controls=spec['detector_controls'],optimizer_model_updates=0)
    if r.get('complete') is not True or r.get('specification')!=spec or r.get('counts')!=expected:
        raise ValueError('evaluation population/count/specification incomplete')
    if 'cpu_return' in r:CPU.verify_merged(root,r,check)
    required=['START.json','RESULT.json','DETECTOR_SCORES.json'];rows={k:[] for k in ('F','J','K','T','P256')}
    for i in range(0,spec['samples'],16):
        batch=read(root/f'validation_{i:04d}.json')
        for key,n in (('F',16),('J',32),('K',16)):
            if len(batch.get(key,[]))!=n:raise ValueError('validation row population differs')
            rows[key].extend(batch[key])
    for i in range(64):
        task=read(root/f'task_{i:02d}.json')
        if task.get('target')!=i:raise ValueError('task record index differs')
        rows['T'].append(task['F64'])
        if spec['p256']:rows['P256'].append(task['P256'])
    from msdev.rules import complete_mean
    metrics={k:complete_mean(v) for k,v in rows.items() if k!='P256' or spec['p256']}
    if any(v is None or not math.isfinite(v) for v in metrics.values()):
        raise ValueError('complete evaluation contains unavailable/nonfinite metrics')
    if r.get('rows')!=rows or r.get('metrics')!=metrics:
        raise ValueError('saved raw rows/aggregate metrics differ')
    scores=read(root/'DETECTOR_SCORES.json')
    if scores.get('scores')!=r.get('scores') or len(scores['scores'])!=scores.get('wrong_key_count',-1)+1:
        raise ValueError('detector vector/count differs')
    if not scores['scores'] or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in scores['scores']):
        raise ValueError('complete detector contains unavailable/nonfinite scores')
    required += [f'validation_{i:04d}.json' for i in range(0,spec['samples'],16)]
    required += [f'task_{i:02d}.json' for i in range(64)]
    for i in range(0,64,16):
        record=read(root/f'inverse_{i:02d}.json')
        if record.get('updates')!=120 or record.get('batch_size')!=16 or record.get('batch_loss_divisor')!=16 or len(record.get('trace',[]))!=120:
            raise ValueError('inverse batch/trace incomplete')
        required += [f'inverse_{i:02d}.json',f'designs_{i:02d}.npy']
    for name in required:
        check()
        if not (root/name).is_file():raise ValueError('missing evaluation member: '+name)
    return dict(verified=True,counts=expected,files=required,scientific_values_recomputed=False)


def diagnostics(root,spec,check=lambda:None,*,expected_binding,copied=False):
    """Full saved F/J/K and detector populations; no missing-as-failed shortcut."""
    root=Path(root);rn='DIAGNOSTIC_RESULT.json' if copied else 'RESULT.json'
    sn='DIAGNOSTIC_START.json' if copied else 'START.json'
    r=read(root/rn);start=read(root/sn);check()
    counts=dict(inverse_input_iterations=0,targets=0,native_t128_calls=0,native_p256_calls=0,
        validation_controls=spec['samples'],detector_controls=spec['detector_controls'],optimizer_model_updates=0)
    if (r.get('domain')!='final' or r.get('phase')!='final' or r.get('specification')!=spec
        or r.get('diagnostics_complete') is not True or r.get('complete') is not False
        or r.get('counts')!=counts or r.get('task_status')!='pending'
        or start.get('evaluation_binding')!=expected_binding):raise ValueError('D2 diagnostic binding/count/scope incomplete')
    rows={k:[] for k in ('F','J','K','T','P256')};files=[sn,rn,'DETECTOR_SCORES.json']
    for i in range(0,spec['samples'],16):
        check();name=f'validation_{i:04d}.json';batch=read(root/name);files.append(name)
        for k,n in (('F',16),('J',32),('K',16)):
            if len(batch.get(k,[]))!=n:raise ValueError('D2 diagnostic row population')
            rows[k].extend(batch[k])
    from msdev.rules import complete_mean
    metrics={k:complete_mean(rows[k]) for k in 'FJK'}
    if any(v is None or not math.isfinite(v) or v<0 for v in metrics.values()):raise ValueError('D2 invalid diagnostics; not failed utility')
    if r.get('rows')!=rows or r.get('metrics')!=dict(metrics,T=None,P256=None):raise ValueError('D2 saved diagnostic aggregates differ')
    scores=read(root/'DETECTOR_SCORES.json')
    if (scores.get('scores')!=r.get('scores') or scores.get('wrong_key_count')!=expected_binding['wrong_key_count']
        or len(r['scores'])!=expected_binding['wrong_key_count']+1 or not r['scores']
        or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in r['scores'])):
        raise ValueError('D2 complete detector evidence unavailable')
    return dict(verified=True,files=files,counts=counts)


def conditional_evaluation(root,spec,check=lambda:None,*,expected_binding,decision,member):
    root=Path(root);r=read(root/'RESULT.json');reuse=read(root/'DIAGNOSTIC_REUSE.json')
    if read(root/'START.json').get('evaluation_binding')!=expected_binding:
        raise ValueError('D2 task/diagnostic model or input binding differs')
    proof=diagnostics(root,spec,check,expected_binding=expected_binding,copied=True)
    if reuse.get('binding')!=expected_binding or set(reuse.get('files',{}))!=set(proof['files']):raise ValueError('D2 reused diagnostic membership differs')
    for name,h in reuse['files'].items():
        if sha(root/name,check)!=h:raise ValueError('D2 reused diagnostic drift')
    from .graph import identity
    cached=read(root/'DIAGNOSTIC_RESULT.json')
    if decision['diagnostic_identities'].get(member)!=identity(cached):raise ValueError('D2 decision uses different diagnostics')
    if decision['task_required']:
        from . import cpu_scoring as CPU
        if r.get('task_status')!='measured' and not (CPU.producer_readback() and r.get('task_status')==CPU.STATE):raise ValueError('D2 mandatory tasks not measured')
        for k in 'FJK':
            if r['rows'][k]!=cached['rows'][k] or r['metrics'][k]!=cached['metrics'][k]:raise ValueError('D2 diagnostics repeated/substituted')
        if r['scores']!=cached['scores']:raise ValueError('D2 detector repeated/substituted')
        return evaluation(root,spec,check)
    if (r.get('task_status')!='unmeasured' or r.get('task_reason')!=decision['reason']
        or r.get('d2_decision')!=decision or r.get('d2_member')!=member
        or r.get('complete') is not False or r.get('conditional_complete') is not True
        or r.get('counts')!=cached['counts'] or r.get('metrics')!=cached['metrics']
        or r.get('rows')!=cached['rows'] or r.get('scores')!=cached['scores']):raise ValueError('D2 unmeasured task disposition differs')
    if list(root.glob('task_*.json')) or list(root.glob('inverse_*.json')) or list(root.glob('designs_*.npy')):
        raise ValueError('D2 skipped stage contains task evidence')
    return dict(verified=True,conditional_complete=True,task_status='unmeasured',reason=decision['reason'])


def inventory(root,check,*,file_limit,byte_limit):
    """All outputs, including partials/failures. No exclusion by success flag."""
    root=Path(root).resolve();result={};total=0
    if type(file_limit) is not int or file_limit<1 or type(byte_limit) is not int or byte_limit<1:
        raise ValueError('explicit accepted evidence limits required; unknown is not zero')
    for directory,dirs,files in os.walk(root,followlinks=False):
        check()
        for name in dirs+files:
            p=Path(directory)/name
            if p.is_symlink() or p.is_junction():raise ValueError('evidence link refused')
        for name in sorted(files):
            check();p=Path(directory)/name;rel=p.relative_to(root).as_posix();safe_name(rel)
            before=p.stat();h=sha(p,check);after=p.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
                raise ValueError('evidence mutated while hashing: '+rel)
            result[rel]=dict(sha256=h,bytes=before.st_size);total+=before.st_size
            if len(result)>file_limit or total>byte_limit:raise ValueError('evidence cap; retain originals and partials')
    if len({n.casefold() for n in result})!=len(result):raise ValueError('case-colliding evidence members')
    return result
