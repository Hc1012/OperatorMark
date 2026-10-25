"""Unexecuted development overwrite input producer; frozen numerics reused."""
from pathlib import Path
from msdev.records import read,write,sha,progress

COUNTS=dict(controls=384,keys=3,T128_forward=384,T128_tangent=384,model_updates=0)

def direction_split():
    from operatormark_corpus_v2 import layout as LY
    split='validation'
    if split not in LY.SPLITS:raise ValueError('frozen overwrite direction split unavailable')
    return split

def generate(seed_path,seed_sha256,exclusions_path,exclusions_sha256,out,check,*,additional_controls=(),additional_directions=()):
    if sha(seed_path)!=seed_sha256 or sha(exclusions_path)!=exclusions_sha256:raise ValueError('contract generation binding')
    s=read(seed_path);deny=read(exclusions_path)
    if s.get('domain')!='development' or deny.get('domain')!='development_exclusions':raise ValueError('development inputs only')
    split=direction_split()  # Validate before any generation; independent committed seeds remain unchanged.
    import numpy as np
    from q6_common import development_controls
    from operatormark_numerics_v2.tpfa_family import TPFAFamily
    from operatormark_numerics_v2.directions import feasible_direction
    from operatormark_corpus_v2.config import D6Directions
    from operatormark.corpus.canonical import tensor_sha256
    from operatormark_numerics_v2.lock import public_deny_list
    out=Path(out);out.mkdir(exist_ok=False);files={};counts={k:0 for k in COUNTS};seen=set(deny['controls'])|public_deny_list()['control_sha256']|set(additional_controls)
    seen_directions=set(deny['directions'])|set(additional_directions)
    def save(name,array):
        import os
        p=out/(name+'.npy')
        with p.open('xb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
        files[p.name]=sha(p)
    try:
        for i in range(3):
            prefix=f'overwrite_{i}_';v=s['values'];rho=development_controls(v[prefix+'controls'],128)
            hashes=[tensor_sha256(r) for r in rho]
            if len(set(hashes))!=128 or set(hashes)&seen:raise ValueError('new attacker-control collision; no redraw')
            seen.update(hashes);counts['controls']+=128
            rng=np.random.Generator(np.random.PCG64(v[prefix+'key']));key=rng.normal(size=64);key=(key/np.linalg.norm(key)).astype(np.float32)
            coeff=rng.normal(size=16);counts['keys']+=1
            d=D6Directions(v[prefix+'directions'],synthetic=False)
            utility=np.stack([d.direction(split,j,0,r) for j,r in enumerate(rho)])
            dh=[tensor_sha256(np.ascontiguousarray(v,dtype=np.float64)) for v in utility]
            if len(set(dh))!=128 or set(dh)&seen_directions:raise ValueError('attacker direction collision; no redraw')
            seen_directions.update(dh)
            save(prefix+'rho',rho);save(prefix+'x',rho.reshape(128,64).astype(np.float32));save(prefix+'v',utility.reshape(128,64).astype(np.float32))
            save(prefix+'key_v',key);save(prefix+'key_coeff',coeff);truth=[]
            for j,r in enumerate(rho):
                check();counts['T128_forward']+=1;progress(out,'native_intent',counts=counts)
                with TPFAFamily(levels=(128,)).open(r,128) as session:
                    save(prefix+f'{j:03d}_field',session.forward.field64)
                    if not session.forward.gates_pass:raise ValueError('overwrite calibration F64 gate')
                    counts['T128_tangent']+=1;progress(out,'native_intent',counts=counts)
                    t=session.tangent(feasible_direction(r,key.astype(np.float64).reshape(8,8)))
                    save(prefix+f'{j:03d}_tangent',t.field64)
                    if not np.isfinite(t.tangent_backward_error) or t.tangent_backward_error>1e-10:raise ValueError('overwrite calibration tangent gate')
                    truth.append(t.field64.reshape(-1)/.01)
            save(prefix+'truth_key_jvp',np.asarray(truth,dtype=np.float32))
        if counts!=COUNTS:raise ValueError('contract generation population')
        write(out/'MANIFEST.json',dict(files=files,domain='development',counts=counts,seed_record_sha256=seed_sha256,exclusions_sha256=exclusions_sha256))
        write(out/'RESULT.json',dict(complete=True,counts=counts,manifest_sha256=sha(out/'MANIFEST.json')))
    except BaseException as exc:
        write(out/'FAILURE.json',dict(complete=False,error=repr(exc),counts=counts,completed_files=files));raise
