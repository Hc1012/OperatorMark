"""Measured remaining-work projections, distinct from per-node reservations."""
import math
import time
from .graph import cost_manifest

def remaining_campaign(nodes,state,clock=time.time):
    done=set(state.get('completed',[]));rates=state.get('rates',{});terms=[]
    def term(name,count,rate,kind='measured',unit='seconds'):
        terms.append(dict(name=name,count=count,rate=rate,seconds=None if rate is None and count else count*(rate or 0),kind=kind,unit=unit))
    pending=[n for n in nodes if n['id'] not in done]
    for recipe in sorted({n['recipe'] for n in pending if n['kind']=='train'}):
        count=sum(n['steps'] for n in pending if n.get('recipe')==recipe)
        rate=rates.get(recipe)
        if rate is None and recipe=='D9':rate=.0741567831
        term('training_'+recipe,count,rate,'transferred_Q6' if recipe=='D9' and recipe not in rates else 'measured_or_unknown')
    evaluations=sum(bool(n.get('evaluate')) for n in pending)
    term('all_remaining_evaluations',evaluations,rates.get('evaluation'))
    # Upper bounds remain explicit; measured cached work is not charged twice.
    projections=max(0,26-state.get('model_projections_completed',0)) if pending else 0
    term('remaining_model_Jacobians',projections,rates.get('projection'))
    # Current cache has distinct forward/full-Jacobian files:3 owners+3 shams+
    # B10+9 B1 grid+9 ordinary attacks+the selected M3 scoring-cost check =26.
    term('remaining_forward_only_models',max(0,26-state.get('forward_projections_completed',0)) if pending else 0,rates.get('forward_projection'))
    checks=sum(math.ceil(n.get('steps',0)/2048)+2 for n in pending if n['kind']=='train')
    term('checkpoints_and_model_publication',checks,rates.get('checkpoint_seconds'))
    for kind in ('calibrate','select_marks','select_replacement','wrappers','scoring_cost','analysis','feasibility'):
        term(kind,sum(n['kind']==kind for n in pending),rates.get('phase_'+kind))
    # Setup is an operational upper bound, NOT an invented measured kernel rate.
    term('possible_second_session_setup',int(bool(pending) and state.get('gpu_charged',0)<31500),1380,'declared_bounded_setup')
    term('remaining_return_transfer_and_hashing',int(bool(pending)),rates.get('final_transfer'))
    term('input_checks_and_reload',len(pending),rates.get('input_check'))
    arms=cost_manifest(nodes)['recovery_arms'];remaining_arms={n.get('arm',n['id']) for n in pending}
    for arm,r in arms.items():
        if arm not in remaining_arms or state.get('recoveries',{}).get(arm):continue
        these=[n for n in pending if n.get('arm',n['id'])==arm]
        recipe=next((n.get('recipe') for n in these if n.get('recipe')),None)
        term('recovery_updates_'+arm,r['max_replay_updates'],rates.get(recipe) if recipe else None)
        if any(n.get('evaluate') for n in these):term('recovery_evaluation_'+arm,1,rates.get('evaluation'))
        if not recipe:term('recovery_phase_'+arm,1,rates.get('phase_'+these[0]['kind']))
        term('recovery_reload_'+arm,1,rates.get('reload'))
    term('additional_recovery_environment_setups',int(bool(remaining_arms)),None,'incident_dependent; each setup <=1380s, within same aggregate cap')
    known=sum(t['seconds'] for t in terms if t['seconds'] is not None)
    unknown=[t['name'] for t in terms if t['seconds'] is None]
    # Charged windows include elapsed setup/idle; unused current window is usable,
    # while previous full reservations remain consumed (no double subtraction).
    now=clock();s=state['session']
    current_end=min(s['end'],s.get('worker_deadline',s['end'])+30)
    available=min(state['deadline']-now,max(0,current_end-now)+max(0,31500-state['gpu_charged']))
    required=1.2*known+30
    return dict(terms=terms,known_remaining_seconds=known,with_20_percent_margin_and_cleanup=required,
                unknown=unknown,available_seconds=available,full_campaign_fits=None if unknown else required<=available,
                known_lower_bound_exceeds=required>available,not_a_runtime_guarantee=True,
                consumed_attached_reservations=state['gpu_charged'],current_session_elapsed=max(0,now-s.get('start',now)),
                measured_current_setup_seconds=s.get('setup_seconds'),recovery_is_conditional_on_remaining_caps=True)

def windows_projection(cost,truth_chunk_seconds,remaining_truth_controls,embedding_controls,attack_controls,remaining_seconds):
    rows=cost['rows'];f=max(r['forward'] for r in rows);t=max(r['tangent'] for r in rows)
    basis=None if truth_chunk_seconds is None else truth_chunk_seconds/16
    terms=dict(remaining_basis_work=None if basis is None else remaining_truth_controls*basis,
               remaining_embedding=embedding_controls*(f+t),remaining_ordinary=attack_controls*f,
               upload_hash_copy_archive=None,recovery_basis_chunk=None if basis is None else 16*basis,
               recovery_bulk_chunk=32*(f+t))
    known=sum(v for v in terms.values() if v is not None);required=1.2*known+30
    return dict(terms=terms,known_seconds=known,required_known_with_margin=required,remaining_seconds=remaining_seconds,
                unknown=[k for k,v in terms.items() if v is None],full_campaign_fits=None,
                known_lower_bound_exceeds=required>remaining_seconds,
                basis_factorization_reuse='one forward/session followed by64 tangent solves; chunk wall time includes writes',
                windows_fit_not_claimed_with_unknown_publication=True)
