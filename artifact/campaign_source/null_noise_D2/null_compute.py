"""Shared numerical work for the frozen null/B2 roster.

Only scheduling/storage changes: the 16-row JVP calls, two direction order,
relative_rows, complete_mean, projected_scores and b2_readout are the original
pinned kernels. No projected 16-basis approximation is used for B2 distance.
"""
import hashlib
import json
import os
from pathlib import Path


def collect_jvp(model, x, v, check=lambda: None):
    import numpy as np
    import torch
    if tuple(x.shape)!=(128,64) or tuple(v.shape)!=(128,2,64):
        raise ValueError('Fixed final B2 control/direction shape required')
    if x.dtype!=torch.float32 or v.dtype!=torch.float32:
        raise ValueError('Original float32 model inputs required')
    values=[]
    with torch.no_grad():
        for k in range(2):
            chunks=[]
            for lo in range(0,128,16):
                check()
                result=torch.func.jvp(model,(x[lo:lo+16],),(v[lo:lo+16,k],))[1]
                if tuple(result.shape)!=(16,4096) or result.dtype!=torch.float32:
                    raise ValueError('Unexpected JVP output layout')
                chunks.append(result.detach().cpu().numpy().copy())
            values.append(np.concatenate(chunks,axis=0))
    array=np.stack(values)
    if not np.isfinite(array).all():raise ValueError('Nonfinite B2 JVP output')
    return array


def distance_rows(candidate, reference, check=lambda: None):
    import numpy as np
    from msdev.scoring import relative_rows, complete_mean
    if candidate.shape!=reference.shape or candidate.shape!=(2,128,4096):
        raise ValueError('B2 raw derivative shape differs')
    if candidate.dtype!=np.float32 or reference.dtype!=np.float32:
        raise ValueError('Original float32 JVP bytes required')
    rows=[]
    for k in range(2):
        for lo in range(0,128,16):
            check()
            # Exactly the conversion/order in msdev.scoring.distance: each
            # float32 JVP batch becomes double before the pinned row kernel.
            rows+=relative_rows(candidate[k,lo:lo+16].astype('float64'),
                                reference[k,lo:lo+16].astype('float64'))
    return dict(rows=rows,mean=complete_mean(rows))


def save_jvp(path, value):
    import numpy as np
    from null_io import sha
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
    observed=np.load(path,allow_pickle=False)
    if observed.dtype!=value.dtype or not np.array_equal(observed,value):
        raise ValueError('Saved JVP readback differs')
    return dict(bytes=path.stat().st_size,sha256=sha(path),shape=list(value.shape),
                dtype=str(value.dtype),raw_float32_output=True)


def load_projection(path, expected):
    import torch
    from null_io import sha
    if sha(path)!=expected['sha256']:raise ValueError('Reused projection bytes changed')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if set(value)!={'identity','projection'} or value['identity']!=expected['identity']:
        raise ValueError('Historical projection identity differs')
    projection=value['projection']
    if set(projection)!={'forward','jacobian'}:
        raise ValueError('Historical projection members differ')
    for name,shape in [('forward',(128,16)),('jacobian',(128,16,64))]:
        x=projection[name]
        if not torch.is_tensor(x) or tuple(x.shape)!=shape or x.dtype!=torch.float64 or not torch.isfinite(x).all():
            raise ValueError('Historical projection layout/finiteness differs')
    return projection


def frozen_b2_readouts(roster, pair_means):
    from msfinal.analysis import b2_readout
    expected={(owner,row['id']) for row in roster['model_jobs'] for owner in row['reference_owners']}
    if set(pair_means)!=expected:raise ValueError('Incomplete frozen B2 pair population')
    result={}
    for name,g in roster['groups'].items():
        values={label:pair_means[(g['reference_owner'],actual)]
                for label,actual in g['canonical_candidates'].items()}
        result[name]=b2_readout(g['descendant'],values,g['matched'],g['independent'])
    return result


def keyed_readout(admission, qualification, projections, roster, truth, check=lambda: None):
    """Original B3 score on already saved final projections; no model call."""
    import numpy as np
    import torch
    from msdev import scoring as S
    from msfinal import materials as A
    owner=admission.prerequisites.owner_id
    rho=A.array(admission,'final_detector_rho','final',(128,8,8),'float64').reshape(128,64)
    dx=A.array(admission,'final_detector_x','final',(128,64),'float32')
    if not np.array_equal(dx,rho.astype('float32')):raise ValueError('Detector control correspondence differs')
    basis=torch.from_numpy(A.array(admission,'public_basis','marking',(16,4096),'float64'))
    count=admission.calibration['wrong_key_count']+1
    if count!=4096:raise ValueError('Fixed 4096-key population required')
    coeff=A.array(admission,'final_key_coeff','final',(count,16),'float64')
    v=A.array(admission,'final_key_v','final',(count,64),'float32')
    if not np.array_equal(coeff[0],A.array(admission,'key_coeff','marking',(count,16),'float64')[0]) or not np.array_equal(v[0],A.array(admission,'key_v','marking',(count,64),'float32')[0]):
        raise ValueError('True key differs from original embedding')
    reference=projections[owner+'/O'];rows=[]
    if set(truth)!={'forward','jacobian'} or tuple(truth['forward'].shape)!=(128,16) or tuple(truth['jacobian'].shape)!=(128,16,64):
        raise ValueError('F64 reference supplement layout differs')
    for label in roster['keyed'][owner]:
        check()
        if label=='F64':candidate=truth
        else:
            role,other=label.split('/');candidate=projections[other+'/'+role]
        values=S.projected_scores(candidate,reference,torch.from_numpy(rho),torch.from_numpy(coeff),
            torch.from_numpy(v),basis,admission.inputs['normalizers'][1]**.5,'B3')
        if len(values)!=4096:raise ValueError('Incomplete keyed scores')
        wrong=max(map(abs,values[1:]));theta=qualification['theta'];delta=qualification['delta']
        model_sha=(None if label=='F64' else next(r['descriptor']['model_sha256'] for r in roster['model_jobs'] if r['id']==other+'/'+role))
        rows.append(dict(model=label,model_sha256=model_sha,scores=values,theta=theta,delta=delta,
            detected=values[0]>theta and values[0]>wrong+delta,
            projection_reused=label!='F64',uses_admitted_truth_projections=False,
            uses_pinned_reference_supplement=label=='F64',historical_catalog_truth_projection_present=False,
            scientific_acceptance=False,population_FPR=None))
    return rows


def f64_distance(admission, clean_jvp, check=lambda: None):
    from msfinal import materials as A
    from msdev import scoring as S
    tangent=A.array(admission,'final_tangent','final',(512,2,4096),'float64')[:128]
    rows=[]
    # Preserve substitution_null's lo-then-direction ordering (B2 itself uses
    # direction-then-lo). Both use the same saved clean JVP batches.
    for lo in range(0,128,16):
        for k in range(2):
            check();rows+=S.relative_rows(tangent[lo:lo+16,k]/.01,
                                          clean_jvp[k,lo:lo+16].astype('float64'))
    return dict(B2_rows=rows,B2_mean=S.complete_mean(rows),physical_claim=False,ownership_removal_claim=False)
