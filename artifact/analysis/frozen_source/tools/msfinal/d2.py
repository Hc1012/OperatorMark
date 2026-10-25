"""Adopted D2: frozen whole-cell task decisions, never inferred task values."""
from pathlib import Path
from msdev.records import read,write,sha
from . import marking as M,graph as G

POLICY=dict(schema='scientific_D2.v1',intermediate_updates=[320,960],endpoint_updates=1920,
    trigger='task can change qualified removal OR retained detection/survival',
    grouping='one family/checkpoint across all eligible parents and planned owners',
    missing='unavailable; never a failed gate',task_scoring=['T128','P256'])

def cells(graph):
    from .variants import STUDENTS,HEALING
    result={}
    controls=[o+'/'+n for o in graph['planned_owners'] for n in ('O','M3','Rstar','S','B10') if n in graph['models'][o]]
    for o,models in graph['models'].items():
        for n,d in models.items():
            v=d.get('procedure',{}).get('variant');cp=d.get('checkpoint')
            if v not in STUDENTS+HEALING or cp not in ('raw320','raw960'):continue
            key=v+'_'+cp
            cell=result.setdefault(key,dict(family=v,checkpoint=cp,members=[],controls=sorted(controls),
                shared_controls=['F64']+(['P256'] if graph['P256_task_enabled'] else []),marked_members=[]))
            cell['members'].append(o+'/'+n)
            if d['source_parent']=='M3':cell['marked_members'].append(o+'/'+n)
    for c in result.values():
        c['members'].sort();c['marked_members'].sort()
    return result

def partial(route,measurement,initial):
    """Both final Boolean outcomes are independent of T iff their prerequisite
    is false or a measured F/J/K gate fails. Detection alone cannot settle both.
    """
    metrics=measurement.get('metrics',{});scores=measurement.get('scores',[])
    valid=(measurement.get('diagnostics_complete') is True and initial.get('available') is True
        and type(initial.get('detected')) is bool and set(initial.get('utility_gates',{}))==set('FJKT')
        and all(type(v) is bool for v in initial['utility_gates'].values())
        and len(scores)==route['wrong_key_count']+1 and all(M.finite(v) for v in scores)
        and all(M.finite(metrics.get(k)) and metrics[k]>=0 and M.finite(route['clean_metrics'].get(k))
                and route['clean_metrics'][k]>=0 for k in 'FJK'))
    if not valid:return dict(state='unavailable',reason='missing/invalid diagnostics or same-test intact evidence')
    gates={k:metrics[k]<=1.25*route['clean_metrics'][k] for k in 'FJK'}
    detected=scores[0]>route['fixed_theta'] and scores[0]>max(abs(v) for v in scores[1:])+route['delta']
    prerequisite=initial['detected'] and all(initial['utility_gates'].values())
    settled=not prerequisite or not all(gates.values())
    return dict(state='settled' if settled else 'task_required',detected=detected,utility_gates=gates,
        qualified_removal=False if settled else None,retained_detection=False if settled else None,
        reason='same-test intact prerequisite false' if not prerequisite else
            ('measured F/J/K failure settles both outcomes for every T' if settled else 'T can change removal or survival'))

def decide(cell,diagnostics,contexts,controls,shared):
    if set(diagnostics)!=set(cell['members']) or set(contexts)!=set(cell['marked_members']):
        raise ValueError('D2 complete predefined cell/marked membership required')
    if set(controls)!=set(cell['controls']) or set(shared)!=set(cell['shared_controls']):
        raise ValueError('D2 required owner/control/shared membership missing')
    if any(r.get('complete') is not True for r in controls.values()):raise ValueError('D2 full control evaluation unavailable')
    if any(not r for r in shared.values()):raise ValueError('D2 shared task evidence unavailable')
    if any(r.get('diagnostics_complete') is not True or any(not M.finite(r.get('metrics',{}).get(k))
        or r['metrics'][k]<0 for k in 'FJK') or not r.get('scores') or not all(M.finite(v) for v in r['scores'])
        for r in diagnostics.values()):raise ValueError('D2 invalid diagnostics; no skip')
    outcomes={n:partial(contexts[n][0],diagnostics[n],contexts[n][1]) for n in cell['marked_members']}
    if any(r['state']=='unavailable' for r in outcomes.values()):raise ValueError('D2 unavailable evidence; no settled verdict')
    required=any(r['state']=='task_required' for r in outcomes.values())
    return dict(policy=POLICY,cell=cell,task_required=required,outcomes=outcomes,
        reason='whole cell required: removal or survival may change' if required else
            'both applicable outcomes settled for every task value; no task-utility or curve claim',
        diagnostic_identities={n:G.identity(r) for n,r in diagnostics.items()},
        control_identities={n:G.identity(r) for n,r in controls.items()},shared_products=shared)

def skipped_readout(route,measurement,initial):
    decision=measurement.get('d2_decision',{});cell=decision.get('cell',{});member=measurement.get('d2_member')
    if (measurement.get('task_status')!='unmeasured' or measurement.get('complete') is not False
        or decision.get('policy')!=POLICY or decision.get('task_required') is not False
        or member not in cell.get('marked_members',[]) or measurement.get('task_reason')!=decision.get('reason')):
        raise ValueError('D2 skipped task proof absent/incompatible')
    outcome=partial(route,measurement,initial)
    if outcome['state']!='settled' or outcome!=decision['outcomes'].get(member):raise ValueError('D2 skip does not settle both outcomes')
    return dict(available=True,detected=outcome['detected'],utility_gates=dict(outcome['utility_gates'],T=None),
        qualified_removal=False,retained_detection=False,qualified_parent_attack_evasion=None,
        task_status='unmeasured',task_reason=decision['reason'],task_utility=None,
        final_intact=initial,conditional_D2=True,complete_task_curve=False)

def required_products(graph):
    result=set()
    for key,c in graph.get('d2_cells',{}).items():
        result.add('d2/'+key)
        for n in c['members']:result.update(('diagnostics/'+n,'evaluations/'+n))
    return result

def read_decision(root,expected):
    if read(Path(root)/'DECISION.json')!=expected:raise ValueError('D2 decision drift')
    return dict(verified=True,task_required=expected['task_required'])
