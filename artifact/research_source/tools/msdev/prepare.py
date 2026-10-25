"""Later-run Windows data and allowlisted upload preparation; not run on import."""
from __future__ import annotations
import dataclasses
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile
from .records import read,write,sha,verify,progress,manifest,ensure_record,publish_zip
from .campaign import Budget

def bindings(config_path,pins_path):
    c=read(config_path);p=read(pins_path)
    if sha(config_path)!=p['configuration_sha256']:raise ValueError('configuration drift')
    verify(c['repo'],p['source_files'])
    for name,want in p['external_files'].items():
        if sha(name)!=want:raise ValueError('bound external input changed: '+name)
    return c,dict(config=sha(config_path),sources=sha(pins_path),Q6_manifest=c['q6_manifest_sha256'])

def import_paths(c):
    # Exactly the Q6 frozen snapshot takes precedence for numerical imports.
    sys.path[:0]=[c['frozen'],c['repo'],str(Path(c['repo'])/'tools')]

def finalize_bundle(root,bundle,portable,check=lambda:None):
    """Resume publication using committed identical records; preserve all partials.

    Controller must establish owned closure before re-entering preparation.
    No incomplete JSON/ZIP is admitted as a committed product.
    """
    root=Path(root);bundle=Path(bundle);ready=root/'READY_FOR_COLAB.json'
    ensure_record(bundle/'CONFIG.json',portable)
    files=[p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()
           and p.name!='MANIFEST.json' and not any(s in p.name for s in ('.writing_', '.pending', '.partial_', '.unreceipted_'))]
    m=manifest(bundle,files,'development-upload')
    m['export_classes']={n:('development-key-values' if n.startswith('keys_') else 'training-seeds' if n=='training_seeds.json' else 'development-data-model-record' if not n.startswith('code/') else 'necessary-code') for n in files}
    m['excluded']=['PRIVATE_SEEDS.json','qualification/confirmation/final material','ownership keys','repository and unrelated evidence']
    ensure_record(bundle/'MANIFEST.json',m);verify(bundle,m['files']);check()
    archive=root/'MainStudy_Development_Upload.zip'
    if ready.exists():
        old=read(ready)
        if sha(archive)!=old['archive_sha256'] or sha(bundle/'MANIFEST.json')!=old['manifest_sha256']:
            raise ValueError('committed upload identity drift')
        return old
    publish_zip(archive,bundle,files+['MANIFEST.json'],check)
    result=dict(complete=True,archive=str(archive.resolve()),archive_sha256=sha(archive,check),manifest_sha256=sha(bundle/'MANIFEST.json'),
        no_production_acceptance=True,physical_qualification=False,Windows_runtime=sha(root/'WINDOWS_RUNTIME.json'))
    ensure_record(ready,result);return result

def seed_inputs(c,root,bundle):
    import secrets
    import numpy as np
    from q6_common import development_controls
    from operatormark.corpus import canonical
    from operatormark_numerics_v2.lock import public_deny_list
    seeds_path=root/'PRIVATE_SEEDS.json'
    bound=root/'INPUT_PROVENANCE.json'
    if seeds_path.exists():
        seeds=read(seeds_path)
        if bound.exists() and read(bound)['private_seed_file_sha256']!=sha(seeds_path):raise ValueError('private development seed binding changed')
    else:
        roles=['embedding'+str(i) for i in range(3)]+['directions','ordinary_data']+['keys_'+m+'_'+str(i) for m in ('B1','B3') for i in range(3)]
        roles+=['init'+str(i) for i in range(3)]+['shuffle'+str(i) for i in range(3)]+['embed_shuffle'+str(i) for i in range(3)]+['replacement_init','ordinary_shuffle']
        seeds={role:secrets.randbits(63) for role in roles}
        if len(set(seeds.values()))!=len(seeds):raise ValueError('seed role collision; no redraw')
        write(seeds_path,seeds) # private generation/key seeds NEVER exported
    from .kernels import smooth_basis
    basis=smooth_basis(dtype=__import__('torch').float64).numpy()
    if not (bundle/'public_basis.npy').exists():np.save(bundle/'public_basis.npy',basis,allow_pickle=False)
    elif not np.array_equal(np.load(bundle/'public_basis.npy',allow_pickle=False),basis):raise ValueError('public pattern basis drift')
    xs=[development_controls(seeds['embedding'+str(i)],128) for i in range(3)]
    attack=development_controls(seeds['ordinary_data'],256)
    q6c=read(c['q6_config']);deny=set(q6c['deny_controls'])|public_deny_list()['control_sha256']
    q6bundle=Path(c['q6_bundle'])
    original_inputs=Path(c['q6_original_inputs'])
    for name in ('training.npy','validation.npy','tasks.npy'):
        a=np.load(original_inputs/name,allow_pickle=False).reshape(-1,8,8)
        deny.update(canonical.tensor_sha256(np.ascontiguousarray(z,dtype=np.float64)) for z in a)
    hs=[canonical.tensor_sha256(z) for group in xs+[attack] for z in group]
    if len(hs)!=len(set(hs)) or set(hs)&deny:raise ValueError('development control collision; no redraw')
    for method in ('B1','B3'):
        for i in range(3):
            path=bundle/f'keys_{method}_{i}.npz'
            rng=np.random.Generator(np.random.PCG64(seeds[f'keys_{method}_{i}']))
            # All 4096 keys IID under ONE generator, independent of training/shams.
            coeff=rng.normal(size=(4096,16));v=rng.normal(size=(4096,64));v=(v/np.linalg.norm(v,axis=1,keepdims=True)).astype(np.float32)
            w=coeff[0]@basis;w=(w/np.sqrt(np.mean(w*w))).astype(np.float32)
            if not path.exists():np.savez(path,coeff=coeff,v=v,w_true=w)
            else:
                with np.load(path,allow_pickle=False) as z:
                    if not np.array_equal(z['coeff'],coeff) or not np.array_equal(z['v'],v) or not np.array_equal(z['w_true'],w):raise ValueError('key input drift')
    train=dict(owners=[dict(initialization=seeds['init'+str(i)],shuffle=seeds['shuffle'+str(i)],embedding_shuffle=seeds['embed_shuffle'+str(i)]) for i in range(3)],
               replacement_initialization=seeds['replacement_init'],ordinary_shuffle=seeds['ordinary_shuffle'])
    if not (bundle/'training_seeds.json').exists():write(bundle/'training_seeds.json',train)
    elif read(bundle/'training_seeds.json')!=train:raise ValueError('training seed drift')
    provenance=dict(private_seed_file_sha256=sha(seeds_path),controls=hs,
        export_values='development keys and training seeds only; generation/key seed values remain private',
        independent_roles=sorted(seeds),keys='4096 IID keys per owner/method; true index0, no null/key seed coupling')
    if bound.exists():
        if read(bound)!=provenance:raise ValueError('development input identity mismatch')
    else:write(bound,provenance)
    if not (bundle/'INPUT_PROVENANCE.json').exists():shutil.copyfile(bound,bundle/'INPUT_PROVENANCE.json')
    elif sha(bound)!=sha(bundle/'INPUT_PROVENANCE.json'):raise ValueError('exported input provenance drift')
    return xs,attack,seeds

def run(c,root,pins_path):
    import_paths(c)
    import numpy as np
    import torch
    from q6_windows import runtime
    from q6_common import true_field
    from operatormark_numerics_v2.tpfa_family import TPFAFamily
    from operatormark_numerics_v2.directions import feasible_direction
    from operatormark_corpus_v2.d10 import TANGENT_GATE
    from .kernels import smooth_basis
    root=Path(root);state=read(root/'STATUS.json');budget=Budget(state);bundle=root/'upload';bundle.mkdir(exist_ok=True)
    q6runtime=runtime(read(c['q6_config']))
    if not (root/'WINDOWS_RUNTIME.json').exists():write(root/'WINDOWS_RUNTIME.json',q6runtime)
    elif read(root/'WINDOWS_RUNTIME.json')!=q6runtime:raise ValueError('Windows runtime changed')
    q6=Path(c['q6_bundle']);old_manifest=read(q6/'MANIFEST.json')
    if sha(q6/'MANIFEST.json')!=c['q6_manifest_sha256']:raise ValueError('Q6 manifest changed')
    # Explicit reusable development tensor/record/code allowlist. No old config seeds.
    names=[n for n in old_manifest['files'] if n.startswith(('training_','validation_','tasks/','exact_baseline/','compatibility/','code/'))
           and n not in ('code/q6_colab.py',)]
    names += ['starts.npy','target_controls.npy','requirements-colab.txt']
    names=list(dict.fromkeys(names))
    io_start=time.monotonic();read_bytes=sum((q6/n).stat().st_size for n in names)
    verify(q6,{n:old_manifest['files'][n] for n in names})
    hash_seconds=time.monotonic()-io_start;copy_start=time.monotonic();copied_bytes=0
    for name in names:
        target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(target)!=old_manifest['files'][name]:raise ValueError('earlier upload input differs')
        else:shutil.copyfile(q6/name,target);copied_bytes+=target.stat().st_size
    io=dict(hash_bytes=read_bytes,hash_seconds=hash_seconds,copy_bytes=copied_bytes,copy_seconds=time.monotonic()-copy_start)
    if not (root/'IO_COST.json').exists():write(root/'IO_COST.json',io)
    io=read(root/'IO_COST.json')
    original_validation=Path(c['q6_original_inputs'])/'validation.npy'
    if not (bundle/'validation_controls_f64.npy').exists():shutil.copyfile(original_validation,bundle/'validation_controls_f64.npy')
    elif sha(original_validation)!=sha(bundle/'validation_controls_f64.npy'):raise ValueError('original validation control drift')
    for source,name in ((c['D1'],'D1.pt'),(c['B10'],'B10.pt'),(c['D1_result'],'D1_RESULT.json'),
                        (c['q6_gpu_environment'],'Q6_ENVIRONMENT.json'),(c['q6_compatibility'],'Q6_COMPATIBILITY.json')):
        target=bundle/name
        if not target.exists():shutil.copyfile(source,target)
        if sha(target)!=sha(source):raise ValueError('returned model/result copy mismatch')
    xs,attack,seeds=seed_inputs(c,root,bundle)
    if state.get('development_input_sha256') not in (None,sha(root/'INPUT_PROVENANCE.json')):raise ValueError('saved development inputs changed')
    state['development_input_sha256']=sha(root/'INPUT_PROVENANCE.json');write(root/'STATUS.json',state,replace=True)
    rng=np.random.Generator(np.random.PCG64(seeds['directions']))
    directions=[[feasible_direction(rho,rng.normal(size=(8,8))) for rho in group] for group in xs]
    generated=root/'generated';generated.mkdir(exist_ok=True)
    first_use_end=None
    def check():
        budget.check()
        if first_use_end is not None and time.time()>=first_use_end:raise TimeoutError('explicit native first-use timing allowance expired')
        from operatormark_numerics_v2.resources import peak_rss_bytes
        if peak_rss_bytes()>8_000_000_000:raise MemoryError('Windows 8GB ceiling')
    if not (root/'SOLVER_COST.json').exists():
        # The prescribed128-operation cost window is consumed once, before bulk.
        # 180s is a stop bound for unknown first use, not a cost estimate.
        budget.reserve(180,0,0);first_use_end=time.time()+180
        from operatormark_numerics_v2.lock import FIXTURES
        rows=[];fixtures=list(FIXTURES.values())
        try:
            for i in range(32):
                check();rho=fixtures[i%3];v=feasible_direction(rho,np.ones((8,8)));row={};rows.append(row)
                t=time.monotonic()
                with TPFAFamily(levels=(128,)).open(rho,128) as s:
                    row['forward']=time.monotonic()-t
                    if not s.forward.gates_pass:raise ValueError('benchmark numerical gate')
                    t=time.monotonic();j=s.tangent(v);row['tangent']=time.monotonic()-t
                    if not np.isfinite(j.tangent_backward_error) or j.tangent_backward_error>TANGENT_GATE:raise ValueError('benchmark tangent gate')
                    t=time.monotonic();g=s.vjp64(np.ones((64,64)));row['adjoint']=time.monotonic()-t
                    if not np.isfinite(g).all():raise ValueError('benchmark adjoint nonfinite')
                t=time.monotonic();true_field(rho,p1=True);row['P256']=time.monotonic()-t
        except BaseException as exc:
            write(root/f'SOLVER_COST_FAILURE_{time.time_ns()}.json',dict(rows=rows,error=repr(exc),complete=False))
            raise
        write(root/'SOLVER_COST.json',dict(rows=rows,operations=128,fixtures='three exposed; no ladder'))
        first_use_end=None
    cost=read(root/'SOLVER_COST.json')
    # Truth Jacobians streamed into public-pattern projections, not 256*4096*64 retained arrays.
    basis=np.load(bundle/'public_basis.npy',allow_pickle=False);validation=np.load(Path(c['q6_original_inputs'])/'validation.npy',allow_pickle=False).reshape(-1,8,8)
    truthdir=generated/'truth';truthdir.mkdir(exist_ok=True)
    keyvs=[]
    for i in range(3):
        with np.load(bundle/f'keys_B3_{i}.npz',allow_pickle=False) as z:keyvs.append(z['v'][0].astype(np.float64))
    for lo in range(0,256,16):
        p=truthdir/f'{lo:03d}.npz';r=p.with_suffix('.json')
        if r.exists():
            if sha(p)!=read(r)['sha256']:raise ValueError('truth projection drift')
            continue
        if p.exists():p.rename(p.with_name(p.name+f'.partial_{time.time_ns()}'))
        prior=[read(r) for r in truthdir.glob('[0-9][0-9][0-9].json')]
        estimate=max((r['seconds'] for r in prior),default=None)
        if estimate is None:
            budget.reserve(180,0,0);first_use_end=time.time()+180
        else:budget.reserve(estimate,estimate,0)
        start=time.monotonic();fps=[];jps=[];kts=[];timings=[]
        try:
            for rho in validation[lo:lo+16]:
                check();fp=[];jp=[];kt=np.zeros((3,4096));phis=[feasible_direction(rho,v).flatten() for v in keyvs]
                factor_start=time.monotonic()
                with TPFAFamily(levels=(128,)).open(rho,128) as s:
                    forward_seconds=time.monotonic()-factor_start;basis_start=time.monotonic()
                    factorization=s._native.lu # pinned session API; verify actual reuse
                    if not s.forward.gates_pass:raise ValueError('truth forward gate')
                    fp=s.forward.field64.flatten()/.01@basis.T/4096
                    for k in range(64):
                        check();v=np.zeros((8,8));sign=-1. if rho.flat[k]>=.98 else 1.;v.flat[k]=sign
                        t=s.tangent(v)
                        if s._native.lu is not factorization:raise RuntimeError('basis tangent factorization was replaced')
                        if not np.isfinite(t.tangent_backward_error) or t.tangent_backward_error>TANGENT_GATE:raise ValueError('basis tangent gate')
                        ambient=t.field64.flatten()*sign
                        jp.append((ambient/.01)@basis.T/4096)
                        for owner in range(3):kt[owner]+=ambient*phis[owner][k]
                    del factorization
                fps.append(fp);jps.append(np.stack(jp,-1));kts.append(kt)
                timings.append(dict(forward_factorization_seconds=forward_seconds,basis_64_seconds=time.monotonic()-basis_start,
                    forward_sessions=1,basis_tangents=64,factorization_reused=True))
            np.savez(p,forward=np.stack(fps),jacobian=np.stack(jps),key_truth=np.stack(kts));write(r,dict(sha256=sha(p),count=len(fps),basis_tangents=64*len(fps),seconds=time.monotonic()-start,timings=timings))
        except BaseException as exc:
            if fps:np.savez(truthdir/f'{lo:03d}_partial_{time.time_ns()}.npz',forward=np.stack(fps),jacobian=np.stack(jps),key_truth=np.stack(kts))
            write(truthdir/f'{lo:03d}_failure_{time.time_ns()}.json',dict(error=repr(exc),completed_controls=len(fps),timings=timings));raise
        progress(root,'truth_projection',completed=lo+len(fps),seconds=time.monotonic()-start)
        first_use_end=None
        from .costs import windows_projection
        measured=max([time.monotonic()-start]+[r['seconds'] for r in prior])
        projection=windows_projection(cost,measured,256-(lo+len(fps)),384,256,
            min(state['deadline'],state['session']['end'])-time.time())
        write(root/f'WINDOWS_PROJECTION_{lo:03d}_{time.time_ns()}.json',projection)
        if projection['known_lower_bound_exceeds']:raise TimeoutError('remaining Windows measured lower bound exceeds allowance before bulk generation')
    allrows={k:[] for k in ('forward','jacobian','key_truth')}
    for p in sorted(truthdir.glob('[0-9][0-9][0-9].npz')):
        with np.load(p,allow_pickle=False) as z:
            for k in allrows:allrows[k].append(z[k].copy())
    allrows={k:np.concatenate(v) for k,v in allrows.items()}
    if not (bundle/'truth_projection.npz').exists():np.savez(bundle/'truth_projection.npz',**{k:allrows[k] for k in ('forward','jacobian')})
    for i in range(3):
        if not (bundle/f'key_truth_{i}.npy').exists():np.save(bundle/f'key_truth_{i}.npy',allrows['key_truth'][:,i],allow_pickle=False)
    for group,(controls,dirs) in enumerate(list(zip(xs,directions))+[(attack,None)]):
        label=f'embedding_{group}' if group<3 else 'attack'
        for lo in range(0,len(controls),32):
            path=generated/f'{label}_{lo:04d}.npz';receipt=path.with_suffix('.json')
            if receipt.exists():
                if sha(path)!=read(receipt)['sha256']:raise ValueError('completed data drift')
                continue
            if path.exists():path.rename(path.with_name(path.name+f'.partial_{time.time_ns()}'))
            count=min(32,len(controls)-lo)
            f=max(r['forward'] for r in cost['rows']);t=max(r['tangent'] for r in cost['rows'])
            estimate=count*(f+(t if dirs is not None else 0))
            budget.reserve(estimate,estimate,0);started=time.monotonic();records=[];outputs=[];failure=None
            counts=dict(T128_open_attempts=0,tangent_attempts=0)
            try:
                for i in range(lo,min(lo+32,len(controls))):
                    check();rho=controls[i]
                    counts['T128_open_attempts']+=1
                    with TPFAFamily(levels=(128,)).open(rho,128) as session:
                        f=session.forward.field64.copy();j=None
                        outputs.append((i,f,None));records.append(dict(index=i,forward_gates=bool(session.forward.gates_pass),tangent_gate=None))
                        if not session.forward.gates_pass:raise ValueError('T128 numerical gate')
                        if dirs is not None:
                            counts['tangent_attempts']+=1
                            tangent=session.tangent(dirs[i])
                            j=tangent.field64.copy()
                            outputs[-1]=(i,f,j)
                            records[-1]['tangent_gate']=float(tangent.tangent_backward_error) if np.isfinite(tangent.tangent_backward_error) else repr(tangent.tangent_backward_error)
                            if not np.isfinite(tangent.tangent_backward_error) or tangent.tangent_backward_error>TANGENT_GATE:raise ValueError('tangent gate')
                data=dict(rho=controls[lo:lo+len(outputs)].copy(),x=controls[lo:lo+len(outputs)].reshape(-1,64).astype(np.float32),y=np.stack([r[1].flatten()/.01 for r in outputs]).astype(np.float32))
                if dirs is not None:data.update(v=np.stack(dirs[lo:lo+len(outputs)]).reshape(-1,64).astype(np.float32),dy=np.stack([r[2].flatten()/.01 for r in outputs]).astype(np.float32))
                np.savez(path,**data);write(receipt,dict(complete=True,sha256=sha(path),rows=records,counts=counts,seconds=time.monotonic()-started))
            except BaseException as exc:
                failure=repr(exc)
                if outputs:
                    pending=generated/f'{label}_{lo:04d}_partial_{time.time_ns()}.npz'
                    completed_tangents=[r for r in outputs if r[2] is not None]
                    np.savez(pending,indices=np.array([r[0] for r in outputs]),y=np.stack([r[1] for r in outputs]),
                             tangent_indices=np.array([r[0] for r in completed_tangents]),
                             dy=np.stack([r[2] for r in completed_tangents]) if completed_tangents else np.empty((0,64,64)))
                write(generated/f'{label}_{lo:04d}_FAILURE_{time.time_ns()}.json',dict(error=failure,observed=records,counts=counts));raise
            progress(root,'data_chunk',group=label,completed=lo+len(outputs),seconds=time.monotonic()-started)
        arrays={}
        for p in sorted(generated.glob(label+'_[0-9][0-9][0-9][0-9].npz')):
            with np.load(p,allow_pickle=False) as z:
                for k in z.files:arrays.setdefault(k,[]).append(z[k].copy())
        arrays={k:np.concatenate(v) for k,v in arrays.items()}
        if group<3:
            target=bundle/(label+'.npz')
            if not target.exists():np.savez(target,**arrays)
        else:
            for k in ('x','y'):
                if not (bundle/f'attack_{k}.npy').exists():np.save(bundle/f'attack_{k}.npy',arrays[k],allow_pickle=False)
    # Replace Q6 driver copies only in the new upload, never in historical bundle.
    for p in (Path(c['repo'])/'tools/msdev').glob('*.py'):
        if p.name=='handoff.py':continue # static Windows notebook/package authoring is not a cloud dependency
        target=bundle/'code/msdev'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
    shutil.copyfile(Path(c['repo'])/'tools/mainstudy_development.py',bundle/'code/mainstudy_development.py')
    for name in ('core','outcomes','analyze'):
        target=bundle/'code/tpfa32_analysis'/(name+'.py');target.parent.mkdir(exist_ok=True);shutil.copyfile(Path(c['repo'])/'experiments/tpfa32_copy_study'/(name+'.py'),target)
    portable=dict(c['scientific'],bindings=state['bindings'],development_only=True,physical_qualification=False,
                  development_input_sha256=state['development_input_sha256'])
    if not (bundle/'WINDOWS_SOLVER_COST.json').exists():shutil.copyfile(root/'SOLVER_COST.json',bundle/'WINDOWS_SOLVER_COST.json')
    elif sha(bundle/'WINDOWS_SOLVER_COST.json')!=sha(root/'SOLVER_COST.json'):raise ValueError('Windows cost evidence changed')
    ensure_record(bundle/'WINDOWS_IO_COST.json',io)
    size=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    if io['copy_bytes'] and io['copy_seconds']>0 and io['hash_bytes'] and io['hash_seconds']>0:
        work=size*(io['copy_seconds']/io['copy_bytes']+2*io['hash_seconds']/io['hash_bytes'])
        publication=dict(projected_seconds=work,basis='measured same-filesystem copy and hash throughput; ZIP_STORED estimate, not network rate',
                         archive_bytes_estimate=size,network_upload_seconds=None)
        budget.reserve(work,work,0)
    else:
        work=300;publication=dict(projected_seconds=None,first_use_work_ceiling=300,basis='copy timing unavailable after interruption',network_upload_seconds=None)
        budget.reserve(300,300,0);first_use_end=time.time()+300
    write(root/f'PUBLICATION_PROJECTION_{time.time_ns()}.json',publication)
    began=time.monotonic()
    finalize_bundle(root,bundle,portable,check)
    write(root/f'PUBLICATION_COST_{time.time_ns()}.json',dict(seconds=time.monotonic()-began,source_bytes=size))
    state['state']='WINDOWS_COMPLETE';write(root/'STATUS.json',state,replace=True)
