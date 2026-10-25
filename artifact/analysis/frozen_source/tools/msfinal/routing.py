"""Final-owner qualification/descendant routing; never a reservation or runner."""
from .marking import RECIPE, bound_record, identity, finite

UNAPPROVED = dict(approved=False, population=None, kappa=None, attack_variants=None,
                  final_calibration=None, B1='unavailable; final disposition pending')


def require_approved_configuration(config):
    if config.get('approved') is not True or any(config.get(k) is None for k in
            ('population', 'kappa', 'attack_variants', 'final_calibration')):
        raise ValueError('final configuration unapproved; no implicit N/kappa/variants/calibration')
    # A complete campaign/runtime/authority validator does not exist in this batch.
    raise NotImplementedError('whole-runner admission and execution are not implemented')


def quality_route(study_id, owner_id, model_sha256, clean_model_sha256):
    """Ordinary P2 metadata only; caller must independently admit the final model.

    This route intentionally contains no detector threshold/removal entitlement.
    A failed or unavailable qualification must not suppress primary measurement.
    """
    identity(model_sha256); identity(clean_model_sha256)
    return dict(variant='ordinary', study_id=study_id, owner_id=owner_id,
                parent_model_sha256=model_sha256, clean_owner_sha256=clean_model_sha256,
                eligibility='completed_final_mark_P2_only', removal_eligible=False,
                execution_authorized=False)


def owner_route(study_id, owner_id, model_sha256, clean_model_sha256, qualification_path,
                qualification_sha256, expected_bindings, variants):
    """Route metadata only, from each owner's externally pinned actual receipt.

    variants is an explicit proposed list, not adoption of an attack matrix.
    No development cohort success or selected-recipe success is an eligibility flag.
    """
    q = bound_record(qualification_path, qualification_sha256)
    identity(model_sha256); identity(clean_model_sha256)
    for key in ('inputs', 'evaluation', 'clean_evaluation', 'detector', 'calibration'):
        identity(expected_bindings[key])
    if (q.get('domain') != 'final' or q.get('study_id') != study_id or q.get('owner_id') != owner_id
            or q.get('recipe') != RECIPE or q.get('model_sha256') != model_sha256
            or q.get('clean_model_sha256') != clean_model_sha256 or q.get('bindings') != expected_bindings
            or q.get('complete') is not True or type(q.get('qualified')) is not bool):
        raise ValueError('missing/incompatible actual final-owner qualification')
    gates = q.get('gates', {})
    if set(gates) != {'F', 'J', 'K', 'T'} or any(
            not all(finite(g.get(k)) and g[k] >= 0 for k in ('candidate', 'clean_owner', 'threshold'))
            or g['threshold'] != 1.25*g['clean_owner'] or g.get('passed') is not (g['candidate'] <= g['threshold'])
            for g in gates.values()):
        raise ValueError('incomplete/inconsistent final utility qualification')
    if (not all(finite(q.get(k)) for k in ('true_score', 'theta', 'delta', 'wrong_max'))
            or min(q['theta'], q['delta'], q['wrong_max']) < 0
            or q['theta'] < q['wrong_max'] + q['delta']
            or q['qualified'] is not (q['true_score'] > q['theta'] and all(g['passed'] for g in gates.values()))):
        raise ValueError('inconsistent final detection qualification')
    if type(q.get('wrong_key_count')) is not int or q['wrong_key_count'] < 1:
        raise ValueError('missing final detector population')
    if not isinstance(variants, list) or any(not isinstance(v, str) or not v for v in variants) or len(variants) != len(set(variants)):
        raise ValueError('explicit unique proposed descendant variants required')
    descendants = [dict(variant=v, study_id=study_id, owner_id=owner_id,
                        parent_model_sha256=model_sha256, clean_owner_sha256=clean_model_sha256,
                        qualification_sha256=qualification_sha256, fixed_theta=q['theta'],
                        delta=q['delta'], wrong_key_count=q['wrong_key_count'],
                        clean_metrics={k:g['clean_owner'] for k,g in gates.items()},
                        eligibility='qualified_final_owner', execution_authorized=False)
                   for v in variants] if q['qualified'] else []
    return dict(owner_id=owner_id, include_primary_quality=True, B3_qualified=q['qualified'],
                B1=UNAPPROVED['B1'], descendants=descendants, execution_authorized=False)


def routing_counts(routes, expected_owner_ids):
    if (len({r['owner_id'] for r in routes}) != len(routes)
            or len(set(expected_owner_ids)) != len(expected_owner_ids)
            or {r['owner_id'] for r in routes} != set(expected_owner_ids)):
        raise ValueError('duplicate or missing final owner; incomplete primary roster')
    return dict(planned_owners=len(expected_owner_ids), primary_quality_owners=len(routes),
                B3_qualified=sum(r['B3_qualified'] is True for r in routes),
                unavailable_qualification=sum(r['B3_qualified'] is None for r in routes),
                ordinary_P2_required=sum(r.get('marked_checkpoint_complete') is True for r in routes),
                missing_marked_checkpoints=[r['owner_id'] for r in routes if r.get('marked_checkpoint_complete') is not True],
                B3_descendants=sum(len(r['descendants']) for r in routes),
                B1_available=False, execution_authorized=False)


def assess_descendant(route, metrics, scores):
    """Frozen intact threshold plus attacked-null separation; clean-owner utility."""
    if route.get('eligibility') != 'qualified_final_owner':
        raise ValueError('qualified final-parent route required')
    if (len(scores) != route['wrong_key_count'] + 1 or not all(finite(v) for v in scores)
            or any(not finite(metrics.get(k)) or metrics[k] < 0 for k in ('F','J','K','T'))):
        return dict(available=False, detected=None, qualified_removal=None)
    utility = {k:metrics[k] <= 1.25*route['clean_metrics'][k] for k in ('F','J','K','T')}
    wrong = max(abs(v) for v in scores[1:])
    detected = scores[0] > route['fixed_theta'] and scores[0] > wrong + route['delta']
    return dict(available=True, utility_gates=utility, detected=detected,
                qualified_removal=not detected and all(utility.values()),
                fixed_theta=route['fixed_theta'], wrong_max=wrong, delta=route['delta'])
