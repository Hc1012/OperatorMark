"""Prospective full-size DEVELOPMENT fixture assembly; NEVER run on import.

No seeds are drawn here. A future separately approved producer supplies recorded
development streams/exclusions/runtime/source bindings. Original assets readonly.
"""
from pathlib import Path
import json,math,shutil
from msdev.records import read,write,sha,progress
from .development_evaluator import inventory

def specifications():
    return {p:dict(samples=n,random_directions=2,detector_controls=d,targets=64,task_batch=16,
        task_updates=120,task_lr=.03,target_scale=.01,model_dtype='float32',p256=True)
        for p,n,d in [('qualification',256,256),('final',512,128)]}

def plan():
    return dict(development_only=True,adopted=False,
        existing_validation=256,new_validation=256,new_random_directions=512,new_keyed_directions=256,
        existing_targets=16,new_targets=48,existing_starts=16,new_starts=48,new_wrong_keys=4095,
        native_T128_forward=304,native_T128_tangent=768,native_P256_forward=48,
        tpfa_factorizations=304,tpfa_primary_rhs_solves=1072,tpfa_refinement_rhs_solves=1072,
        P256_eta_continuations=0,model_updates=0,inverse_input_updates=0,
        scopes={p:inventory(s,p) for p,s in specifications().items()},
        policy='verification fixture reuses old256 as qualification and first256 final-sized controls; NOT independent scientific splits')

def assemble(base,producer_manifest_sha256,seed_file,seed_sha256,exclusions_file,exclusions_sha256,
             out,check,*,runtime_sha256,source_pins,normalizers,normalizer_source,normalizer_sha256):
    """One future generation, fixed populations, no redraw/replacement/resume.

    Fully saved old labels reused; each new native observation saved immediately.
    Generation and seed approval must precede this call under owned supervision.
    """
    import numpy as np
    from q6_common import development_controls,development_starts,true_field
    from operatormark_numerics_v2.tpfa_family import TPFAFamily
    from operatormark_numerics_v2.directions import feasible_direction
    from operatormark_corpus_v2.config import D6Directions
    from operatormark_corpus_v2.d10 import TANGENT_GATE
    from operatormark.corpus.canonical import tensor_sha256
    from operatormark_numerics_v2.lock import public_deny_list
    base=Path(base);out=Path(out);check()
    for p,h in ((base/'MANIFEST.json',producer_manifest_sha256),(seed_file,seed_sha256),(exclusions_file,exclusions_sha256),
                (normalizer_source,normalizer_sha256)):
        if sha(p)!=h:raise ValueError('prospective input binding drift')
    if tuple(read(normalizer_source)['normalizers']['0'])!=tuple(normalizers):raise ValueError('historical D1 PN normalizer substitution')
    if sha(base/'Q6_ENVIRONMENT.json')!=runtime_sha256:raise ValueError('historical numerical-worker runtime binding differs')
    seed=read(seed_file);deny=read(exclusions_file)
    roles={'validation','directions','targets','starts','wrong_keys'}
    if seed.get('domain')!='development' or set(seed.get('values',{}))!=roles:raise ValueError('recorded distinct development streams required')
    seeds=seed['values']
    if any(type(v) is not int or not 0<=v<2**64 for v in seeds.values()) or len(set(seeds.values()))!=len(roles):raise ValueError('invalid/colliding development streams')
    if deny.get('domain')!='development_exclusions' or not all(k in deny for k in ('controls','directions','starts','seed_values')):raise ValueError('full externally supplied exclusion inventory required')
    if set(seeds.values())&set(deny['seed_values']):raise ValueError('existing seed reused; no redraw')
    m=read(base/'MANIFEST.json')['files'];used={}
    def data(name):
        check();raw=(base/name).read_bytes()
        import hashlib,io
        if hashlib.sha256(raw).hexdigest()!=m[name]:raise ValueError('original input drift: '+name)
        used[name]=m[name];return np.load(io.BytesIO(raw),allow_pickle=False)
    oldrho=data('validation_controls_f64.npy');oldv=data('validation_v.npy').reshape(256,2,64)
    oldtruth=data('validation_truth_f64.npy').reshape(256,4096);oldtan=data('validation_tangent_truth_f64.npy').reshape(256,2,4096)
    oldkey=data('key_truth_0.npy').reshape(256,4096);oldstarts=data('starts.npy')
    oldtargets=data('target_controls.npy');basis=data('public_basis.npy')
    with data('keys_B3_0.npz') as z:kv=z['v'].copy();coeff=z['coeff'].copy()
    if oldrho.shape!=(256,8,8) or oldstarts.shape!=(16,8,8) or kv.shape!=(4096,64) or coeff.shape!=(4096,16):raise ValueError('original population mismatch')
    fresh=development_controls(seeds['validation'],256);targets=development_controls(seeds['targets'],48)
    starts=development_starts(seeds['starts'],48);directions=D6Directions(seeds['directions'],synthetic=False)
    newv=np.stack([[directions.direction('validation',256+i,k,r) for k in range(2)] for i,r in enumerate(fresh)])
    def distinct(values,excluded):
        hs=[tensor_sha256(np.ascontiguousarray(v,dtype=np.float64)) for v in values]
        if len(set(hs))!=len(hs) or set(hs)&set(excluded):raise ValueError('development collision; no redraw')
    oldcontrols=[tensor_sha256(v) for v in list(oldrho)+list(oldtargets)]
    distinct(list(fresh)+list(targets),set(deny['controls'])|set(oldcontrols)|public_deny_list()['control_sha256'])
    distinct(starts,set(deny['starts'])|{tensor_sha256(s) for s in oldstarts})
    distinct(newv.reshape(-1,8,8),deny['directions'])
    out.mkdir(exist_ok=False);rawdir=out/'raw';rawdir.mkdir();counts={k:0 for k in ('T128_forward','T128_tangent','P256_forward')}
    write(out/'INPUT_BINDINGS.json',dict(domain='development',producer_manifest_sha256=producer_manifest_sha256,
        seed_record_sha256=seed_sha256,exclusions_sha256=exclusions_sha256,plan=plan(),source_pins=source_pins,
        runtime_sha256=runtime_sha256,normalizer_source_sha256=normalizer_sha256))
    def put(path,x):
        with path.open('xb') as f:np.save(f,np.ascontiguousarray(x),allow_pickle=False)
    def charge(k):check();counts[k]+=1;progress(out,'native_intent',operation=k,counts=counts)
    entries={}
    def asset(name,x,purpose):
        if not np.isfinite(x).all():raise ValueError('nonfinite generated development asset: '+name)
        p=out/(name+'.npy');put(p,x);entries[name]=dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size,
            purpose=purpose,shape=list(x.shape),dtype=str(x.dtype),domain='development')
    try:
        for name,x in [('new_controls',fresh),('new_directions',newv),('new_targets',targets),('new_starts',starts)]:put(rawdir/(name+'.npy'),x)
        truth=[];tangent=[];keytruth=[]
        for i,rho in enumerate(fresh):
            charge('T128_forward')
            with TPFAFamily(levels=(128,)).open(rho,128) as s:
                if not s.forward.gates_pass:raise ValueError('development T128 forward gate')
                f=s.forward.field64.copy();put(rawdir/f'validation_{i:03d}_field.npy',f);truth.append(f.reshape(-1))
                row=[];keydir=feasible_direction(rho,kv[0].astype(np.float64).reshape(8,8))
                for k,v in enumerate([*newv[i],keydir]):
                    charge('T128_tangent');t=s.tangent(v)
                    if not math.isfinite(t.tangent_backward_error) or t.tangent_backward_error>TANGENT_GATE:raise ValueError('development tangent gate')
                    put(rawdir/f'validation_{i:03d}_tangent{k}.npy',t.field64)
                    (row if k<2 else keytruth).append(t.field64.reshape(-1).copy())
                tangent.append(row)
        tf=[data(f'tasks/target_{i:02d}_f64.npy') for i in range(16)]
        pf=[data(f'tasks/target_{i:02d}_p256.npy') for i in range(16)]
        for i,rho in enumerate(targets):
            charge('T128_forward');v=true_field(rho);put(rawdir/f'target_{i:02d}_T.npy',v);tf.append(v)
            charge('P256_forward');v=true_field(rho,True);put(rawdir/f'target_{i:02d}_P.npy',v);pf.append(v)
        fullrho=np.concatenate((oldrho,fresh));fullv=np.concatenate((oldv,newv.reshape(256,2,64).astype(np.float32)))
        values=dict(rho=fullrho,x=fullrho.reshape(512,64).astype(np.float32),v=fullv,
            truth=np.concatenate((oldtruth,truth)),tangent=np.concatenate((oldtan,np.asarray(tangent))),key_truth=np.concatenate((oldkey,keytruth)))
        rng=np.random.Generator(np.random.PCG64(seeds['wrong_keys']))
        wc=rng.normal(size=(4095,16));wv=rng.normal(size=(4095,64));wv=(wv/np.linalg.norm(wv,axis=1,keepdims=True)).astype(np.float32)
        finalkv=np.concatenate((kv[:1],wv));finalcoeff=np.concatenate((coeff[:1],wc))
        for phase,spec in specifications().items():
            n=spec['samples'];d=spec['detector_controls']
            for name,x in values.items():asset(phase+'_'+name,x[:n],phase)
            for name,x in dict(starts=np.concatenate((oldstarts,starts)),targets=np.asarray(tf),targets_p256=np.asarray(pf),
                detector_x=fullrho[:d].reshape(d,64).astype(np.float32),detector_rho=fullrho[:d],detector_v=fullv[:d],
                key_v=kv if phase=='qualification' else finalkv,key_coeff=coeff if phase=='qualification' else finalcoeff).items():asset(phase+'_'+name,x,phase)
        asset('key_v',kv,'marking');asset('key_coeff',coeff,'marking');asset('public_basis',basis,'marking')
        if sha(base/'D1.pt')!=m['D1.pt']:raise ValueError('clean owner bytes drift')
        shutil.copyfile(base/'D1.pt',out/'D1.pt');entries['clean_model']=dict(path='D1.pt',sha256=m['D1.pt'],bytes=(out/'D1.pt').stat().st_size,purpose='marking',domain='development')
        write(out/'DEVELOPMENT_CATALOG.json',dict(domain='development',verification_only=True,entries=entries,
            specifications=specifications(),runtime_sha256=runtime_sha256,source_pins=source_pins,normalizers=normalizers,
            original_producer_hashes=used,independent_acceptance=False,disjoint_final_scientific_splits=False))
        if counts!=dict(T128_forward=304,T128_tangent=768,P256_forward=48):raise ValueError('prospective native count mismatch')
        write(out/'RESULT.json',dict(complete=True,counts=counts,catalog_sha256=sha(out/'DEVELOPMENT_CATALOG.json'),production_acceptance=False))
    except BaseException as exc:
        write(out/'FAILURE.json',dict(error=repr(exc),counts=counts,complete=False));raise
