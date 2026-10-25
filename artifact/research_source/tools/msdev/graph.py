"""Explicit finite development graph. No final-material or qualification node."""
from __future__ import annotations

def operation_graph():
    nodes=[]
    def add(name,kind,**kw):nodes.append(dict(id=name,kind=kind,**kw))
    add('D1','reuse',owner=0,source='D1.pt')
    add('B10','reuse',owner=0,source='B10.pt')
    add('early_feasibility','feasibility')
    # Existing work, moved before expensive owners. These independent trajectories
    # retain their original inputs/initialization and exact total update counts.
    add('Rmain_1920','train',arm='Rstar',owner=0,recipe='forward32',steps=1920,parent=None,evaluate=True)
    add('A4_cost','train',arm='A4_cost',owner=0,recipe='A4',steps=200,cost_only=True,selected_weights='O')
    add('wrapper_cost','wrappers')
    for owner in (1,2):
        name=f'D{owner+1}'
        add(name+'_main40960','train',arm=name,owner=owner,recipe='D9',steps=36864)
        add(name+'_40960','train',arm=name,owner=owner,recipe='D9',steps=4096,parent=name+'_main40960',cooldown=True,evaluate=True)
        add(name+'_main81920','train',arm=name,owner=owner,recipe='D9',steps=36864,parent=name+'_main40960')
        add(name,'train',arm=name,owner=owner,recipe='D9',steps=8192,parent=name+'_main81920',cooldown=True,evaluate=True)
    add('clean_screen','screen')
    for owner in range(3):add(f'S{owner+1}','train',arm=f'S{owner+1}',owner=owner,recipe='sham',steps=320,weights=f'D{owner+1}',evaluate=True)
    add('calibration','calibrate')
    for method in ('B1','B3'):
        for factor in (1.5,2.,3.):
            for owner in range(3):
                name=f'{method}_{factor:g}_D{owner+1}'
                add(name,'train',arm=name,owner=owner,recipe=method,steps=320,weights=f'D{owner+1}',factor=factor,evaluate=True)
    add('mark_selection','select_marks')
    for owner in range(3):
        for source in ('O','M3','M1'):
            name=f'Aord_{source}_D{owner+1}'
            add(name,'train',arm=name,owner=owner,recipe='forward32',steps=1920,selected_weights=source,conditional=source=='M1',evaluate=True)
    parent='Rmain_1920';position=1920
    for target in (9216,18432,36864,73728):
        name=f'Rmain_{target}'
        add(name,'train',arm='Rstar',owner=0,recipe='forward32',steps=target-position,parent=parent,evaluate=target==1920)
        parent=name;position=target
        if target!=1920:
            length=target*10//9
            add(f'Rstar_{length}','train',arm='Rstar',owner=0,recipe='forward32',steps=length-target,parent=parent,cooldown=True,evaluate=True)
    add('replacement_selection','select_replacement')
    for recipe in ('A5B1','A5B2','A5B3'):
        add(recipe+'_cost','train',arm=recipe+'_cost',owner=0,recipe=recipe,steps=200,cost_only=True,
            selected_weights='M1' if recipe=='A5B1' else 'M3' if recipe=='A5B3' else 'O')
    add('scoring_cost','scoring_cost');add('analysis','analysis')
    assert sum(n.get('steps',0) for n in nodes)==285920
    return nodes

def cost_manifest(nodes=None):
    nodes=nodes or operation_graph();arms={}
    for n in nodes:
        arm=n.get('arm',n['id']);r=arms.setdefault(arm,dict(nodes=[],max_replay_updates=0,nontraining_replay_seconds=None))
        r['nodes'].append(n['id']);r['max_replay_updates']=max(r['max_replay_updates'],min(2048,n.get('steps',0)))
    arms['windows_preparation']=dict(nodes=['embedding_data','ordinary_data','truth_projection','solver_cost','upload'],max_replay_updates=0,
        nontraining_replay_seconds=None,componentwise_replay_attempt_bounds=dict(T128_forward=32,tangent=1024,adjoint=32,P256=32),
        note='One recovery; actual active chunk determines the bound. Missing original seeds refuses recovery.')
    return dict(maximum_scheduled_updates=sum(n.get('steps',0) for n in nodes),
        caps=dict(attached_A100_seconds=31500,windows_numerical_seconds=7200,wall_seconds=86400),
        measured_rates=dict(Q6_matching_D9_seconds_per_update=.0741567831,Q6_evaluation_seconds_range=[24.59,26.10]),
        transferred_rates_are_estimates=True,unknown_rates=['forward32','embedding','A4','A5B1','A5B2','A5B3',
            'full_Jacobian_scores','wrapper16','wrapper256','new_evaluation_K','Windows_basis_tangents','uploads','reload'],
        recovery_arms=arms,total_worst_case_replayed_updates=sum(x['max_replay_updates'] for x in arms.values()),
        recovery_policy='one infrastructure recovery per declared arm, same bindings/runtime; all replay and idle time charged',
        Windows_operations=dict(embedding_forward=384,embedding_tangents=384,attack_forward=256,
            validation_forward=256,validation_basis_tangents=16384,benchmark_forward=32,benchmark_tangent=32,benchmark_adjoint=32,benchmark_P256=32),
        evaluation_operations=dict(max_new_model_evaluations=39,targets_per_evaluation=16,inverse_input_updates=4680,
            Colab_T128_scoring=624,Colab_P256_scoring=624,D1_inverse_evaluation_reused=True,
            full_Jacobian_controls=256,columns_per_control=64,wrong_keys_per_owner_method=4095,
            cached_model_Jacobians_at_most=26,cached_forward_projections_at_most=26,random_JVP_B2_distances='complete declared matched/unmatched development rosters; streamed',
            B1_only_grid_models_require_no_Jacobian=True),
        early_work=['prescribed128-operation Windows window','first planned16-control/1024-tangent chunk with factorization reuse',
                    'D1/B10 cached projections','existing Rmain1920 and its evaluation','existing A4_cost','existing wrapper_cost'],
        projection_contract='remaining_campaign includes all pending training/evaluation/scoring, IO, setup and eligible recovery; unknowns propagate and fit cannot be asserted',
        note='No placeholder prices for unlike kernels. Evaluation/transfer/retention/reload time is additive and measured at first use. 20% projection margin; all hard caps remain absolute.')
