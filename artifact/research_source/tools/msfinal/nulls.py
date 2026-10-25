"""Bound null/B2 readouts on one already opened final control set."""
from pathlib import Path
import hashlib
from operatormark.corpus.state_v2_prerequisites import require_accepted
from msdev.records import write
from . import materials as A, marking as M, routing as R, orchestration as O, evaluation as E, analysis


def common_controls(reference,other):
    rp=reference.prerequisites;op=other.prerequisites
    A.require_opening(reference);A.require_opening(other)
    if rp.configuration.sha256!=op.configuration.sha256 or rp.opening.sha256!=op.opening.sha256:
        raise ValueError('null model from a different configuration/opening')
    for name in ('final_detector_x','final_detector_rho','final_detector_v'):
        if A.catalog(reference)['entries'][name]['sha256']!=A.catalog(other)['entries'][name]['sha256']:
            raise ValueError('null comparison control/direction substitution')


def qualification(admission,path,acceptance):
    p=admission.prerequisites;h=M.digest(path);q=M.bound_record(path,h)
    if require_accepted(acceptance,kind='main_study_final_qualification',bindings=dict(A.scope(admission),
                      model_sha256=q['model_sha256'],qualification_sha256=h)).get('domain')!='final':
        raise ValueError('development qualification refused')
    R.owner_route(p.study_id,p.owner_id,q['model_sha256'],p.clean_model_sha256,path,h,q['bindings'],[])
    if any(q['bindings'].get(k)!=v for k,v in dict(inputs=p.inputs.sha256,calibration=p.calibration.sha256,
                                                  clean_evaluation=p.clean_evaluation.sha256).items()):
        raise ValueError('null threshold qualification substitution')
    return q


def keyed_grid(admission,qualification_path,qualification_acceptance,candidates,out,check,*,store=None):
    """N-1 other-clean + N replacements + own sham; F64 separately required.

    candidates maps each prescribed label to (its own admission, FinalModel).
    No missing-model exclusion or post-outcome threshold calibration.
    """
    p=admission.prerequisites;owners=admission.configuration['owners']
    expected={'O/'+o for o in owners if o!=p.owner_id}|{'Rstar/'+o for o in owners}|{'S/'+p.owner_id}
    if set(candidates)!=expected:raise ValueError('incomplete prescribed keyed null population')
    q=qualification(admission,qualification_path,qualification_acceptance)
    for label,(other,item) in candidates.items():
        if label.split('/')[1]!=other.prerequisites.owner_id:raise ValueError('null owner label differs')
        if item.name!=label.split('/')[0] or item.wrapper_steps is not None:
            raise ValueError('keyed null model role differs from its closed canonical alias')
        common_controls(admission,other);O.final_model(item,other)
    out=Path(out);out.mkdir(exist_ok=False);store=E.ProjectionStore(out/'projections') if store is None else store;rows=[]
    for label,(other,item) in candidates.items():
        check();model=O.load_final_model(item,other)
        values=E.detector_scores(model,admission,'final',check,model_sha256=O.measurement_identity(item),wrapper_steps=item.wrapper_steps,store=store)
        wrong=max(map(abs,values[1:]));detected=values[0]>q['theta'] and values[0]>wrong+q['delta']
        row=dict(model=label,model_sha256=item.sha256,scores=values,theta=q['theta'],delta=q['delta'],detected=detected)
        rows.append(row);write(out/(label.replace('/','_')+'.json'),row)
    result=dict(rows=rows,required_F64_separate=True,population_FPR=None,independent_acceptance=False)
    write(out/'KEYED_NULLS.json',result);return result


def b2_group(reference,candidates,matched_ids,independent_ids,descendant_id,out,check):
    """Matched procedure and independent null identities must be fixed externally."""
    expected={descendant_id,*matched_ids,*independent_ids}
    if len(expected)!=1+len(matched_ids)+len(independent_ids) or set(candidates)!=expected:
        raise ValueError('incomplete/duplicate B2 comparison population')
    A.require_opening(reference);A.configure_and_check(reference)
    # The closed protocol explicitly binds which models share arm, source,
    # checkpoint and wrapper. A caller cannot supply an easier arbitrary list.
    owner=reference.prerequisites.owner_id
    group=reference.prerequisites.opening.read().get('b2_groups',{}).get(owner,{}).get(descendant_id)
    if group!=dict(matched=list(matched_ids),independent=list(independent_ids)):
        raise ValueError('B2 matched/independent identities absent from closed protocol')
    for other,item in candidates.values():common_controls(reference,other);O.final_model(item,other)
    import torch
    import q5_model as Q
    from msdev.scoring import distance
    from msdev.rules import complete_mean
    original=Q.Q5FNO().to('cuda').eval();original.load_state_dict(A.weights(reference,'clean_model','marking'),strict=True)
    x=torch.from_numpy(A.array(reference,'final_detector_x','final',(128,64),'float32')).to('cuda')
    v=torch.from_numpy(A.array(reference,'final_detector_v','final',(128,2,64),'float32')).to('cuda')
    out=Path(out);out.mkdir(exist_ok=False);values={}
    for label,(other,item) in candidates.items():
        check();model=O.load_final_model(item,other);rows=[]
        for k in range(2):rows+=distance(model,original,x,v[:,k],check)['rows']
        values[label]=complete_mean(rows)
        filename=hashlib.sha256(label.encode()).hexdigest()+'.json'
        write(out/filename,dict(label=label,model_sha256=item.sha256,rows=rows,mean=values[label]))
    result=analysis.b2_readout(descendant_id,values,matched_ids,independent_ids)
    write(out/'B2.json',result);return result


def substitution_null(admission,qualification_path,qualification_acceptance,out,check,*,store=None):
    """F64 null from admitted exact-reference projections, never fabricated zeros."""
    q=qualification(admission,qualification_path,qualification_acceptance)
    A.require_opening(admission);check();A.configure_and_check(admission)
    manifest=A.catalog(admission)
    contract=manifest.get('truth_projection_contract')
    expected=dict(reference='F64=area64(T128)',units='model_units_divided_by_0.01',
        controls_sha256=manifest['entries']['final_detector_rho']['sha256'],
        basis_sha256=manifest['entries']['public_basis']['sha256'])
    if contract!=expected:raise ValueError('admitted F64 projection units/source/controls unavailable')
    import torch
    import numpy as np
    import q5_model as Q
    from msdev import scoring as S
    nkeys=admission.calibration['wrong_key_count']+1
    dx=torch.from_numpy(A.array(admission,'final_detector_x','final',(128,64),'float32')).to('cuda')
    rho=torch.from_numpy(A.array(admission,'final_detector_rho','final',(128,8,8),'float64').reshape(128,64))
    if not np.array_equal(dx.cpu().numpy(),rho.numpy().astype('float32')):raise ValueError('F64 null control correspondence differs')
    basis=torch.from_numpy(A.array(admission,'public_basis','marking',(16,4096),'float64')).to('cuda')
    coeff=torch.from_numpy(A.array(admission,'final_key_coeff','final',(nkeys,16),'float64'))
    v=torch.from_numpy(A.array(admission,'final_key_v','final',(nkeys,64),'float32'))
    embedded_v=torch.from_numpy(A.array(admission,'key_v','marking',(nkeys,64),'float32'))
    embedded_coeff=torch.from_numpy(A.array(admission,'key_coeff','marking',(nkeys,16),'float64'))
    if not torch.equal(v[0],embedded_v[0]) or not torch.equal(coeff[0],embedded_coeff[0]):raise ValueError('true-key substitution')
    candidate=dict(forward=torch.from_numpy(A.array(admission,'final_truth_projection_forward','final',(128,16),'float64')),
        jacobian=torch.from_numpy(A.array(admission,'final_truth_projection_jacobian','final',(128,16,64),'float64')))
    owner=Q.Q5FNO().to('cuda').eval();owner.load_state_dict(A.weights(admission,'clean_model','marking'),strict=True)
    reference=E.projection(owner,admission,'final',admission.prerequisites.clean_model_sha256,None,dx,basis,check,store)
    values=S.projected_scores(candidate,reference,rho,coeff,v,basis,admission.inputs['normalizers'][1]**.5,'B3')
    random_v=torch.from_numpy(A.array(admission,'final_detector_v','final',(128,2,64),'float32')).to('cuda')
    tangent=A.array(admission,'final_detector_truth_tangent','final',(128,2,4096),'float64');rows=[]
    with torch.no_grad():
        for lo in range(0,128,16):
            for k in range(2):
                check();j=torch.func.jvp(owner,(dx[lo:lo+16],),(random_v[lo:lo+16,k],))[1]
                rows+=S.relative_rows(tangent[lo:lo+16,k]/.01,j.double().cpu().numpy())
    wrong=max(map(abs,values[1:]));result=dict(scores=values,theta=q['theta'],delta=q['delta'],
        detected=values[0]>q['theta'] and values[0]>wrong+q['delta'],B2_rows=rows,B2_mean=S.complete_mean(rows),
        uses_admitted_truth_projections=True,physical_claim=False,ownership_removal_claim=False)
    out=Path(out);out.mkdir(exist_ok=False);write(out/'F64_NULL.json',result);return result
