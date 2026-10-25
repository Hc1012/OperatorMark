"""Complete only the missing fixed F64-null projection material.

The old material producer saved final truth and random tangents, but not the
64-coordinate Jacobian at the 128 detector controls. This supplement uses the
unchanged TPFAFamily(128), one factorization per control and all 64 fixed unit
directions. It neither replaces old truth nor repeats native optimization.
"""
from pathlib import Path
from dataclasses import fields
import hashlib,json,math,os,time
from null_io import need,put,sha


def save_array(path,array):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
    observed=np.load(path,allow_pickle=False)
    need(observed.shape==array.shape and observed.dtype==array.dtype
         and np.array_equal(observed,array,equal_nan=True),'Native reference raw readback differs')
    return dict(bytes=path.stat().st_size,sha256=sha(path),shape=list(array.shape),dtype=str(array.dtype))


def scalar_result(value):
    def safe(x):
        if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
        if isinstance(x,float) and not math.isfinite(x):return dict(nonfinite=repr(x))
        return x
    return {f.name:safe(getattr(value,f.name)) for f in fields(value) if f.name not in ('field64','correction64')}


def collect_native(rhos,out,check,session_factory,*,tangent_gate):
    """Factorization reuse; injected factories are for offline wiring only."""
    import numpy as np
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    counts=dict(T128_forward=0,T128_tangent=0);completed=dict(counts)
    start=time.monotonic();directions=np.eye(64,dtype=np.float64).reshape(64,8,8)
    def event(value):
        with (out/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(value,seconds=time.monotonic()-start),sort_keys=True,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
    def charge(kind,control,direction=None):
        check();counts[kind]+=1
        limit=len(rhos)*(64 if kind=='T128_tangent' else 1)
        need(counts[kind]<=limit,'Fixed F64 reference operation cap exceeded')
        event(dict(event='intent',kind=kind,ordinal=counts[kind],control=control,direction=direction))
        return counts[kind]
    def finish(kind,ordinal,value,path,control,direction=None):
        saved=save_array(path,value.field64)
        event(dict(event='completion',kind=kind,ordinal=ordinal,control=control,direction=direction,
            native=scalar_result(value),field=saved,path=path.relative_to(out).as_posix()))
        completed[kind]+=1
    for i,rho in enumerate(rhos):
        ordinal=charge('T128_forward',i)
        with session_factory(rho) as session:
            finish('T128_forward',ordinal,session.forward,out/f'control_{i:03d}'/'forward.npy',i)
            need(session.forward.gates_pass and np.isfinite(session.forward.field64).all(),
                 'Native F64 reference forward hard gate refused; no retry')
            for j,direction in enumerate(directions):
                ordinal=charge('T128_tangent',i,j);result=session.tangent(direction)
                finish('T128_tangent',ordinal,result,out/f'control_{i:03d}'/f'tangent_{j:02d}.npy',i,j)
                need(np.isfinite(result.field64).all() and math.isfinite(result.tangent_backward_error)
                     and 0<=result.tangent_backward_error<=tangent_gate,
                     'Native F64 reference tangent hard gate refused; no retry')
        if (i+1)%16==0:print(f'F64 null reference: {i+1}/{len(rhos)} fixed controls complete.',flush=True)
    need(counts==completed==dict(T128_forward=len(rhos),T128_tangent=64*len(rhos)),
         'Incomplete F64 reference attempts/completions')
    return dict(counts=counts,completed=completed,seconds=time.monotonic()-start,
        attempts_and_completions=2*sum(counts.values()),native_hard_gates=True)


def prepare(admission,out,check,binding):
    import numpy as np
    import torch
    from msfinal import materials as A
    from operatormark_numerics_v2.tpfa_family import TPFAFamily,TANGENT_GATE
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    catalog=A.catalog(admission);entries=catalog['entries']
    rho=A.array(admission,'final_detector_rho','final',(128,8,8),'float64')
    full_rho=A.array(admission,'final_rho','final',(512,8,8),'float64')
    dv=A.array(admission,'final_detector_v','final',(128,2,64),'float32')
    full_v=A.array(admission,'final_v','final',(512,2,64),'float32')
    truth=A.array(admission,'final_truth','final',(512,4096),'float64')
    tangent=A.array(admission,'final_tangent','final',(512,2,4096),'float64')
    need(np.array_equal(rho,full_rho[:128]) and np.array_equal(dv,full_v[:128]),
         'Original first-128 detector mapping differs; no substitute controls')
    basis=A.array(admission,'public_basis','marking',(16,4096),'float64')
    input_hashes={n:entries[n]['sha256'] for n in ('final_detector_rho','final_detector_v','final_rho',
        'final_v','final_truth','final_tangent','public_basis')}
    put(out/'INTENT.json',dict(binding=binding,scope='missing_F64_null_reference_material_only',
        input_hashes=input_hashes,controls=128,directions='all 64 C-order coordinate unit directions',
        forward_projection_source='existing final_truth[:128]/0.01',
        random_tangent_source='existing final_tangent[:128]/0.01',
        Jacobian_source='unchanged TPFAFamily(128).tangent(unit_direction)/0.01',
        model_calls=0,optimization_updates=0,training_updates=0,
        expected_native_calls=dict(T128_forward=128,T128_tangent=8192),
        old_catalogs_and_truth_unchanged=True,scientific_acceptance=False))
    native=collect_native(rho,out/'native',check,lambda r:TPFAFamily(levels=(128,)).open(r,128),tangent_gate=TANGENT_GATE)
    # Same double projection expression and batch size as the pinned projection
    # kernel; reference fields are physical-unit float64 divided by 0.01.
    tb=torch.from_numpy(basis).to('cuda');forward=[];jacobian=[];auxiliary_max=0.
    with torch.no_grad():
        for lo in range(0,128,4):
            check();f=torch.from_numpy(truth[lo:lo+4]/.01).to('cuda')
            forward.append((f@tb.T/4096).cpu())
            columns=[]
            for direction in range(64):
                check()
                raw=np.stack([np.load(out/'native'/f'control_{i:03d}'/f'tangent_{direction:02d}.npy',allow_pickle=False)
                              .reshape(4096) for i in range(lo,lo+4)])
                need(raw.shape==(4,4096) and raw.dtype==np.float64 and np.isfinite(raw).all(),'Native tangent assembly differs')
                j=torch.from_numpy(raw/.01).to('cuda')
                columns.append((j@tb.T/4096).cpu())
            jacobian.append(torch.stack(columns,dim=-1))
            for i in range(lo,lo+4):
                fresh=np.load(out/'native'/f'control_{i:03d}'/'forward.npy',allow_pickle=False).reshape(4096)
                auxiliary_max=max(auxiliary_max,float(np.max(np.abs(fresh-truth[i]))))
    projection=dict(forward=torch.cat(forward),jacobian=torch.cat(jacobian))
    records={}
    for name,array in projection.items():records[name]=save_array(out/(name+'.npy'),array.numpy())
    result=dict(complete=True,binding=binding,native=native,input_hashes=input_hashes,projections=records,
        projection_contract=dict(reference='F64=area64(T128)',units='model_units_divided_by_0.01',
            controls_sha256=entries['final_detector_rho']['sha256'],basis_sha256=entries['public_basis']['sha256']),
        historical_catalogs_rewritten=False,historical_truth_replaced=False,
        fresh_forward_max_abs_difference_from_retained_truth=auxiliary_max,
        random_tangents_recomputed=False,native_optimization_repeated=False,
        scientific_acceptance=False,physical_claim=False,formal_custody_assurance=False)
    put(out/'REFERENCE_RESULT.json',result)
    return projection,result
