"""Finite source-level waves. No execution grant, reservation or launcher.

The external acceptance interfaces remain mandatory between waves. Producer
outputs below are never turned into independent acceptances by this module.
"""
from pathlib import Path
from dataclasses import dataclass
import re

from operatormark.corpus.state_v2_prerequisites import AcceptedRecord
from .scientific_admission import require as require_accepted, runtime_ready, verify_material_lineage
from msdev.records import write
from . import backend as B, attacks as T, materials as A, marking as M, evaluation as E, routing as R


def clean_training_wave(requests,out,check,*,deadline):
    """New prospective owners plus one shared B10; no historical-owner replay."""
    from . import owners
    if not requests:raise ValueError('empty clean-training roster')
    admitted=[owners.admit(p) for p in requests];config=admitted[0][0]
    expected=[(o,'O') for o in config['owners']]+[(config['shared_baseline_id'],'B10')]
    if ([(p.owner_id,p.role) for p in requests]!=expected
            or len({p.configuration.sha256 for p in requests})!=1):
        raise ValueError('complete same-config owner/B10 order required')
    inventories=[tuple(d['entries'][n]['sha256'] for n in ('x','v','y','dy')) for _,d in admitted]
    if len(set(inventories))!=1:raise ValueError('same-corpus O/B10 data correspondence differs')
    out=Path(out);out.mkdir(exist_ok=False);results=[]
    try:
        for p in requests:
            check();result=owners.execute(p,out/p.owner_id,check,deadline=deadline)
            results.append(dict(owner_id=p.owner_id,role=p.role,result=result))
            if result.get('complete') is not True:raise ValueError('incomplete clean training; no next owner')
        value=dict(complete=True,results=results,planned=expected,independent_acceptance=False)
        write(out/'CLEAN_TRAINING_WAVE.json',value);return value
    except BaseException as exc:
        write(out/'FAILURE.json',dict(complete=False,error=repr(exc),completed=results,not_completed=expected[len(results):]));raise


def marking_wave(prerequisites, out, check, *, numerical_factory=None):
    """Keep every owner, including unavailable/failed qualification outcomes."""
    if not prerequisites:raise ValueError('empty owner roster')
    admissions=[B.admit(p) for p in prerequisites]
    config=admissions[0].configuration;binding=prerequisites[0].configuration.sha256
    if ([p.owner_id for p in prerequisites]!=config['owners']
            or any(p.configuration.sha256!=binding for p in prerequisites)):
        raise ValueError('complete same-configuration roster required')
    out=Path(out);out.mkdir(exist_ok=False);results=[]
    for p in prerequisites:
        check()
        r=B.run_owner(p,out/p.owner_id,check=check,numerical=None if numerical_factory is None else numerical_factory(p))
        results.append(r)
        if r['state'] in ('FAILED','BLOCKED'):
            # Do not execute the remaining owners after an operational failure;
            # retain them explicitly as unattempted, not a smaller population.
            for missing in prerequisites[len(results):]:
                results.append(dict(owner_id=missing.owner_id,state='UNATTEMPTED',qualification_available=False,
                                    B3_qualified=None,descendants=[],include_primary_quality=True))
            break
    value=dict(results=results,accounting=B.owner_accounting(results,config['owners']),
               configuration_sha256=binding,independent_acceptance=False)
    write(out/'MARKING_WAVE.json',value);return value


def lock_roster(admissions, qualifications, acceptance, *, marked_models):
    """Read actual final qualifications and an external complete-roster receipt."""
    if not admissions:raise ValueError('empty roster')
    first=admissions[0];owners=first.configuration['owners'];config=first.prerequisites.configuration.sha256
    if ([a.prerequisites.owner_id for a in admissions]!=owners or set(qualifications)!=set(owners)
            or set(marked_models)!=set(owners)):
        raise ValueError('complete planned roster required, including failures')
    hashes={o:M.digest(qualifications[o]) if qualifications[o] is not None else None for o in owners}
    models={o:marked_models[o].sha256 if marked_models[o] is not None else None for o in owners}
    accepted_models={o:marked_models[o].acceptance.sha256 if marked_models[o] is not None else None for o in owners}
    receipt=require_accepted(acceptance,kind='main_study_final_mark_roster',
        bindings=dict(study_id=first.prerequisites.study_id,configuration_sha256=config,qualification_hashes=hashes,
                      marked_model_hashes=models,marked_acceptance_hashes=accepted_models))
    if receipt.get('domain')!='final':raise ValueError('development roster is not final')
    result={}
    for a in admissions:
        p=a.prerequisites;path=qualifications[p.owner_id];model=marked_models[p.owner_id]
        if p.configuration.sha256!=config:raise ValueError('mixed configuration roster')
        quality=[]
        if model is not None:
            if model.kind!='M3' or M.digest(model.path)!=M.identity(model.sha256):
                raise ValueError('completed final marked roster identity differs')
            record=require_accepted(model.acceptance,kind='main_study_final_model',
                bindings=dict(A.scope(a),model_sha256=model.sha256,model_kind='M3'))
            T.completed_mark(record)
            quality=[R.quality_route(p.study_id,p.owner_id,model.sha256,p.clean_model_sha256)]
        if path is None:
            result[p.owner_id]=dict(owner_id=p.owner_id,B3_qualified=None,descendants=[],
                include_primary_quality=True,qualification_available=False)
        else:
            q=M.bound_record(path,hashes[p.owner_id])
            if model is None or q.get('model_sha256')!=model.sha256:
                raise ValueError('qualification/model roster correspondence differs')
            if any(q['bindings'].get(k)!=v for k,v in dict(inputs=p.inputs.sha256,clean_evaluation=p.clean_evaluation.sha256,
                                                          calibration=p.calibration.sha256).items()):
                raise ValueError('roster qualification input substitution')
            result[p.owner_id]=R.owner_route(p.study_id,p.owner_id,q['model_sha256'],p.clean_model_sha256,
                    path,hashes[p.owner_id],q['bindings'],[v for v in a.configuration['attack_variants'] if v in MARKED_VARIANTS])
            result[p.owner_id]['qualification_available']=True
        result[p.owner_id].update(marked_checkpoint_complete=model is not None,quality_descendants=quality,
            primary_measurement_status='required_not_measured' if quality else 'missing_marked_checkpoint')
    return result


MARKED_VARIANTS=('ordinary','A4','A5B3','jacobi16','jacobi256')


def expected_training_jobs(config,roster):
    if list(roster)!=config['owners']:raise ValueError('all owners required')
    required={'ordinary','replacement','A4','A5B2','A5B3','sham','jacobi16','jacobi256','F64_substitution'}
    if not required<=set(config['attack_variants']):raise ValueError('specified core arms missing; no automatic omission')
    jobs=[]
    for owner in config['owners']:
        jobs += [(owner,'O',v) for v in ('ordinary','replacement','A4','A5B2','sham')]
        if roster[owner].get('marked_checkpoint_complete') is True:
            jobs += [(owner,'M3','ordinary')]
        if roster[owner]['B3_qualified']:
            if roster[owner].get('marked_checkpoint_complete') is not True:
                raise ValueError('qualified owner without completed checkpoint')
            jobs += [(owner,'M3',v) for v in ('A4','A5B3')]
    return jobs


def admit_training_wave(requests,admissions,qualifications,roster_acceptance,*,marked_models):
    roster=lock_roster(admissions,qualifications,roster_acceptance,marked_models=marked_models);config=admissions[0].configuration
    expected=expected_training_jobs(config,roster)
    keys=[(r.admission.prerequisites.owner_id,r.parent.kind,r.variant) for r in requests]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):raise ValueError('core training graph incomplete/duplicated')
    for request in requests:
        if request.parent.kind=='M3' and request.parent.sha256!=marked_models[request.admission.prerequisites.owner_id].sha256:
            raise ValueError('training parent differs from independently locked marked roster')
    # Validate every dependency before starting this wave; no partial dispatch on
    # a missing final owner's receipt.
    admitted=[T.admit(request) for request in requests]
    shared=[(t['data_sha256'],t['shuffle']) for _,t,_,_ in admitted if t.get('data_group')=='shared16384']
    if not shared or len(set(shared))!=1:
        raise ValueError('A4/A5 must share the prescribed data and initial shuffle stream')
    return roster,expected,keys


def training_wave(requests,admissions,qualifications,roster_acceptance,out,check,*,marked_models,execute=T.execute_training):
    roster,expected,keys=admit_training_wave(requests,admissions,qualifications,roster_acceptance,marked_models=marked_models)
    config=admissions[0].configuration
    out=Path(out);out.mkdir(exist_ok=False);results=[]
    try:
        for request,key in zip(requests,keys):
            check();dest=out/('_'.join(key));r=execute(request,dest,check);results.append(dict(job=key,result=r))
            if r.get('complete') is not True:raise ValueError('incomplete attack; no automatic continuation')
        value=dict(complete=True,results=results,planned_owners=config['owners'],roster=roster,independent_acceptance=False,
                   primary_complete=False,missing_marked_checkpoints=[o for o,r in roster.items() if not r['marked_checkpoint_complete']])
        write(out/'TRAINING_WAVE.json',value);return value
    except BaseException as exc:
        write(out/'FAILURE.json',dict(error=repr(exc),completed=results,remaining=expected[len(results):],complete=False));raise


@dataclass(frozen=True)
class FinalModel:
    name: str
    path: Path
    sha256: str
    acceptance: AcceptedRecord
    wrapper_steps: int | None = None
    procedure: dict | None = None
    dependencies: dict | None = None


def model_descriptor(model):
    value=dict(model_sha256=model.sha256,wrapper_steps=model.wrapper_steps)
    if model.procedure is not None:
        value.update(procedure=model.procedure,dependencies={k:v[1] for k,v in (model.dependencies or {}).items()})
    return value


def measurement_identity(model):
    """Different architectures/transforms/dependencies cannot reuse raw caches."""
    if model.procedure is None:return model.sha256
    from .graph import identity
    return identity(model_descriptor(model))


def final_model(model,admission):
    if not isinstance(model.name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',model.name):
        raise ValueError('safe closed model identifier required')
    A.require_opening(admission)
    descriptor=model_descriptor(model)
    if require_accepted(model.acceptance,kind='main_study_final_evaluation_model',bindings=dict(A.scope(admission),
        name=model.name,**descriptor)).get('domain')!='final':
        raise ValueError('development model acceptance refused')
    opening=admission.prerequisites.opening.read()
    if opening.get('models',{}).get(admission.prerequisites.owner_id,{}).get(model.name)!=descriptor:
        raise ValueError('model is absent from coordinated hash closure')
    if model.wrapper_steps not in (None,16,256) or M.digest(model.path)!=M.identity(model.sha256):
        raise ValueError('final model/wrapper binding differs')
    if model.name=='O' and model.sha256!=admission.prerequisites.clean_model_sha256:
        raise ValueError('original clean-owner alias substituted')
    if model.procedure is not None:
        from . import variants as V
        p=model.procedure;v=p.get('variant')
        if v not in V.selected(admission.configuration) or model.wrapper_steps is not None:
            raise ValueError('additional procedure is not explicitly adopted/bound')
        spec=V.definition(v);expected=dict(architecture=spec['architecture'],variant=v,
            transform=v if spec['kind']=='transform' else None)
        if p!=expected:raise ValueError('model reconstruction procedure differs')
        deps=model.dependencies or {};names={'endpoint'} if v.startswith('mix') else set()
        if set(deps)!=names:raise ValueError('transform dependencies missing/extra')
        for path,h in deps.values():
            if M.digest(path)!=M.identity(h):raise ValueError('transform dependency drift')


def load_final_model(item,admission):
    final_model(item,admission);A.configure_and_check(admission)
    import q5_model as Q
    from msdev.kernels import Jacobi
    from . import models
    model=(Q.Q5FNO() if item.procedure is None else models.create(item.procedure['architecture'])).to('cuda').eval()
    # Shared safe byte-bound loader; no optimizer or initialization substitution.
    model.load_state_dict(T._parent_weights(item),strict=True)
    if item.procedure is not None:
        transform=item.procedure['transform']
        if transform is not None:
            endpoint=None
            if item.dependencies:
                from types import SimpleNamespace
                path,h=item.dependencies['endpoint'];endpoint=Q.Q5FNO().to('cuda').eval()
                endpoint.load_state_dict(T._parent_weights(SimpleNamespace(path=path,sha256=h)),strict=True)
            model=models.transformed(model,transform,endpoint=endpoint)
        return model
    return model if item.wrapper_steps is None else Jacobi(model,item.wrapper_steps)


def final_wave(admission,items,out,check):
    """Exact externally frozen per-owner model list, opened/evaluated once.

    Closure must list all prescribed checkpoints/wrappers. Completeness of that
    scientific list is an independent closure obligation, not inferred here from
    whichever files a caller supplies. This function does not issue that receipt.
    """
    admission=B.admit(admission.prerequisites);opening=A.require_opening(admission)
    if any(admission.prerequisites.owner_id+'/'+item.name in cell['members']
           for cell in opening.get('D2',{}).get('cells',{}).values() for item in items):
        raise ValueError('D2 intermediate evaluation requires coordinated whole-cell Worker route')
    expected=opening['models'][admission.prerequisites.owner_id]
    if len(items)!=len({i.name for i in items}) or {i.name for i in items}!=set(expected):
        raise ValueError('incomplete frozen final model population')
    for item in items:final_model(item,admission)
    out=Path(out);out.mkdir(exist_ok=False);result={};store=E.ProjectionStore(out/'projections')
    try:
        for item in items:
            check();model=load_final_model(item,admission)
            value=E.evaluate_model(model,admission,out/item.name,check,phase='final',model_sha256=measurement_identity(item),
                                   wrapper_steps=item.wrapper_steps,projection_store=store)
            result[item.name]=dict(model_sha256=item.sha256,evaluation=value)
            if value.get('complete') is not True:raise ValueError('incomplete final evaluation, no exclusions')
        write(out/'FINAL_WAVE.json',dict(complete=True,items=result,independent_acceptance=False));return result
    except BaseException as exc:
        write(out/'FAILURE.json',dict(error=repr(exc),completed=result,complete=False));raise


def final_comparison_context(route,clean_final,intact_final,admission):
    """Independently admitted same-test owner/intact evidence for D2 and readout."""
    if route.get('eligibility')!='qualified_final_owner':
        raise ValueError('P2-only route cannot establish removal eligibility')
    A.require_opening(admission);p=admission.prerequisites
    clean=require_accepted(clean_final,kind='main_study_final_clean_test_evaluation',
        bindings=dict(A.scope(admission),opening_sha256=p.opening.sha256,model_sha256=p.clean_model_sha256))
    if clean.get('domain')!='final' or clean.get('complete') is not True or clean.get('evaluation_contract')!=E.specification(admission,'final'):
        raise ValueError('same-test original clean-owner evaluation unavailable')
    if any(not M.finite(clean.get('metrics',{}).get(k)) or clean['metrics'][k]<0 for k in 'FJKT'):
        raise ValueError('invalid original clean-owner final metrics')
    if route.get('owner_id')!=p.owner_id or route.get('clean_owner_sha256')!=p.clean_model_sha256:
        raise ValueError('final comparison owner substitution')
    final_route=dict(route,clean_metrics=clean['metrics'])
    intact=require_accepted(intact_final,kind='main_study_final_intact_test_evaluation',
        bindings=dict(A.scope(admission),opening_sha256=p.opening.sha256,
                      model_sha256=route['parent_model_sha256'],qualification_sha256=route['qualification_sha256']))
    if intact.get('domain')!='final' or intact.get('complete') is not True or intact.get('evaluation_contract')!=E.specification(admission,'final'):
        raise ValueError('same-test final intact outcome unavailable')
    initial=R.assess_descendant(final_route,intact.get('metrics',{}),intact.get('scores',[]))
    return final_route,initial


def final_descendant_readout(route,measurements,clean_final,intact_final,admission):
    """Use final clean-owner errors, never calibration errors for the final gate."""
    if measurements.get('task_status')=='designs_ready_pending_scoring':
        raise ValueError('D6 required CPU scoring pending; no removal or survival decision')
    final_route,initial=final_comparison_context(route,clean_final,intact_final,admission)
    if measurements.get('task_status')=='unmeasured':
        from .d2 import skipped_readout
        return skipped_readout(final_route,measurements,initial)
    result=R.assess_descendant(final_route,measurements.get('metrics',{}),measurements.get('scores',[]))
    result['final_intact']=initial
    result['qualified_parent_attack_evasion']=result.get('qualified_removal')
    if not initial['available'] or not result['available']:
        result.update(qualified_removal=None,retained_detection=None)
    else:
        prerequisite=initial['detected'] and all(initial['utility_gates'].values())
        result.update(qualified_removal=prerequisite and result['qualified_removal'],
                      retained_detection=prerequisite and result['detected'] and all(result['utility_gates'].values()))
    return result
