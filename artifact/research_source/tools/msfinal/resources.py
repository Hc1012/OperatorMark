"""Pure source-derived accounting; estimates never authorize an execution."""
import copy
from . import graph as G,variants as V,p256_tasks as P

def ledger(graph,config):
    from .evaluation import validate_specification
    for phase in ('qualification','final'):validate_specification(config['evaluation'][phase],phase)
    # Separate source families without pricing additional architectures as FNO.
    core=copy.deepcopy(graph)
    core['trajectories']={k:v for k,v in core['trajectories'].items() if not k.startswith('additional/')}
    core['models']={o:{n:d for n,d in m.items() if d.get('procedure') is None and not d.get('selected_contract')} for o,m in core['models'].items()}
    core['b2_groups']={o:{n:g for n,g in m.items() if n in core['models'][o]} for o,m in core['b2_groups'].items()}
    c=G.workload(core,config['evaluation']['qualification']['p256']);extra=V.workload(graph)
    jobs=graph['contract_jobs'];ow=sum(j['arm']=='overwrite' for j in jobs)
    noise=sum(j['arm']=='noise' for j in jobs);acc=sum(j['arm']=='adaptive_accuser' for j in jobs)
    api=sum(j.get('attempts',0) for j in jobs if j['arm']=='API')
    all_evals=c['final_evaluations']+extra['full_model_evaluations']+ow+noise
    added=extra['full_model_evaluations']+ow+noise
    groups=[v for m in graph['b2_groups'].values() for v in m.values()]
    b2=sum(1+len(v['matched'])+len(v['independent']) for v in groups if not v['missing'])
    wrappers={str(k):sum(d['wrapper_steps']==k for m in graph['models'].values() for d in m.values()) for k in (16,256)}
    unknown=['architecture/healing sustained throughput','wrapper/noise full evaluator throughput',
        'native production and independent re-solves','all final calibration/null material generation',
        'variable JSON/tensors/caches and failure retention bytes','fsync/transfer/export throughput',
        'sessions/setup/idle/permitted recovery','current credits and billing rate']
    policy=config.get('resource_policy')
    if not isinstance(policy,dict):policy=None
    p=P.workload(64,120) if graph['P256_task_enabled'] else None
    # Ceilings remain prospective policy inputs. None is not a zero charge.
    costs={k:None if policy is None else policy.get(k) for k in
        ('setup_seconds','transfer_seconds','sessions','cleanup_seconds_per_session','finalization_seconds_per_session',
         'idle_seconds','recovery_seconds','model_state_bytes_upper','optimizer_state_bytes_upper',
         'variable_records_bytes_upper','cache_bytes_upper','failure_retention_bytes_upper','archive_bytes_upper')}
    for k,v in costs.items():
        if v is not None and (type(v) not in (int,float) or v<0):raise ValueError('invalid explicit resource policy: '+k)
    from . import storage_plan
    layout=config.get('storage_layout',{})
    domains=dict.fromkeys(storage_plan.DOMAINS)
    supplied=layout.get('domains',{})
    if set(supplied)-set(domains):raise ValueError('unknown storage accounting domain')
    domains.update(supplied)
    # D5 keeps only validation primals + inverse primals, one published tensor.
    # Derivative-only J/K validation has no draw or discarded primal. Failed
    # partials live in historical_and_new_failures, never a zero/dedup credit.
    noise_calls=512//16+4*120
    raw_noise=noise_calls*32*16*4096*4*noise
    if domains['noise_draws_including_partials'] not in (None,raw_noise):raise ValueError('source-counted retained noise cannot be changed')
    domains['noise_draws_including_partials']=raw_noise
    storage=storage_plan.account(domains,**{k:layout.get(k) for k in (
        'phase_archives','remote_records','retained_restore_archives','active_restore_payload',
        'local_inputs','local_failures','remote_failures','duplicate_archives','remote_available','local_available')})
    # D6 changes placement/copies, not scientific populations or inverse updates.
    learned_full=all_evals+c['qualification_evaluations'];monitors=c['monitor_evaluations']
    d6=dict(full_scoring_requests_upper=learned_full,monitor_scoring_requests=monitors,
        logical_waves=['owners','qualification','replacement','final','D2_if_required'],
        logical_barriers_without_session_capacity_splits=4,
        additional_D2_barrier_upper=1,
        physical_batches=None,physical_GPU_sessions=None,
        batching='explicit per-request work/return/input bounds; split on count, bytes and work; session.fit may interrupt a wave; no replay',
        CPU_T128_forwards_upper=64*learned_full+16*monitors,
        CPU_P256_forwards_upper=64*all_evals+(64*c['qualification_evaluations'] if config['evaluation']['qualification']['p256'] else 0)+16*monitors,
        repeated_inverse_updates=0,repeated_model_updates=0,repeated_noise_draws=0,
        copied_target_payload_bytes_upper=64*4096*8*(all_evals*2+c['qualification_evaluations']*(2 if config['evaluation']['qualification']['p256'] else 1))+16*4096*8*2*monitors,
        persisted_design_payload_bytes=64*64*8*learned_full+16*64*8*monitors,
        successful_copies='producer request files; batch export staging payload; one batch transfer ZIP; CPU extracted request; merged readback view; independently retained phase/export/restore copies additional',
        return_copies='CPU raw rows/intents/runtime/manifest; one batch return archive; imported return; merged CPU_RETURN/task rows and per-request copies of shared supervisor/stream evidence; custody/export copies additional',
        trace_and_JSON_bytes=None,archive_headers_bytes=None,retained_failure_and_partial_bytes=None,
        CPU_seconds=None,transfer_seconds=None,setup_seconds=None,session_count=None,guaranteed_GPU_savings=None,
        native_inverse='F64/P256 optimization remains separate; no CPU saved-design scoring request includes optimizer calls',
        admission='original immutable deadline; each CPU work+cleanup+finalization envelope and completed-view node must fit; no automatic retry',
        fit_established=False)
    return dict(population=len(graph['planned_owners']),kappa=config['kappa'],core=c,additional=extra,storage_layout=storage,D6=d6,
        selected=dict(overwrite_endpoints=ow,overwrite_updates=320*ow,overwrite_label_F=128*ow,
            overwrite_label_utility_JVP=128*ow,overwrite_label_key_JVP=128*ow,
            overwrite_raw_intent_and_update_files=640*ow,overwrite_model_state_files_including_partials=2*ow,
            overwrite_model_state_publication_receipts=2*ow,
            noise_evaluations=noise,noise_repetitions=32,noise_native_calls_outside_evaluator=0,
            noise_forward_calls_per_evaluation=noise_calls,
            noise_derivative_only_batches_per_evaluation=3*(512//16),
            noise_retained_draw_tensors=noise_calls*32*noise,
            noise_raw_files=noise_calls*(1+32*3)*noise+noise,
            noise_tensor_payload_bytes_including_partials=raw_noise,
            noise_completed_tensor_copies=1,noise_failed_partial_bytes=None,
            noise_storage_scope='successful payload only; publication/address receipts and serialization overhead separate; historical/new failed partials retained separately',
            noise_draw_rule=graph['noise_draw_rule'],omitted_measurements=graph['contract_omitted'],
            noise_serialization_header_bytes=None,
            API_attempts=api,API_raw_records=3*api,API_derivative_task_available=False,
            API_computed_forward_calls_upper=api,API_computed_forward_calls=None,
            API_denials=None,API_response_value_components_upper=4096*api,
            API_JSON_bytes_and_result_copies=None,API_counts_basis='observed outcomes; 1024 attempts per retained M3 policy stream',
            adaptive_selections=acc,adaptive_final_readouts=acc,adaptive_score_values=8192*acc,
            adaptive_projection_JVP_upper=4*128*64*acc,
            adaptive_projection_primal_output_upper=4*128*acc,
            externally_accepted_calibration_requests=ow+acc,calibration_null_score_obligations=12*(ow+acc),
            calibration_generation_seconds=None,P256_task=p),
        totals=dict(model_updates=c['total_model_updates']+extra['model_updates']+320*ow,
            D2_conditional_task_completions_upper=extra['conditional_task_completions_upper'],
            D2_unconditional_final_evaluations=all_evals-extra['conditional_task_completions_upper'],
            D2_inverse_and_native_counts_below='upper bounds, not measured fit; diagnostics never repeated',
            final_evaluations=all_evals,qualification_evaluations=c['qualification_evaluations'],
            monitor_evaluations=c['monitor_evaluations'],
            batched_inverse_iterations=c['inverse_input_iterations']+480*added,
            P256_per_target_inverse_updates=0 if p is None else p['per_target_inverse_updates'],
            native_T128_scoring=c['native_t128_scoring_calls']+64*added+c['substitution']['t128_scoring']+(0 if p is None else p['T128_scoring']),
            native_P256_scoring=c['native_p256_scoring_calls']+64*added+c['substitution']['p256_scoring']+(0 if p is None else p['P256_scoring']),
            F64_optimization=c['substitution'],P256_optimization=p,
            final_validation_controls=512*all_evals,final_random_JVPs=1024*all_evals,
            final_keyed_JVPs=512*all_evals,final_detector_control_population=128*all_evals,
            matched_B2_groups=len(groups),incomplete_B2_groups=sum(bool(v['missing']) for v in groups),
            B2_model_and_reference_JVPs=512*b2,
            journals=c['journals']+extra['journals'],normal_journal_records=c['journal_records']+extra['journal_records'],
            journal_bytes_upper_bound=c['journal_bytes_upper_bound']+extra['journal_bytes_upper_bound'],
            periodic_anchors=c['periodic_anchors']+extra['periodic_anchors'],
            source_counted_trajectory_files=c['core_writer_files']+extra['core_trajectory_writer_files'],
            wrapper_evaluations=wrappers,wrapper_sweeps_per_model_call={'16':16,'256':256}),
        supported_estimates=dict(core_training_seconds=c['training_projection_seconds'],
            plain_final_evaluation_seconds_measured=172.0454962799995,
            plain_qualification_seconds_measured=175.33990549600003,
            applicability='historical coupled evaluator, single bound plain FNO; native scoring already included; not a post-D6 GPU rate or transferable CPU score-only rate',
            extra_training_seconds=None,total_campaign_seconds=None),
        enforced_source_limits=dict(journal_record_bytes=65536,API_attempts_per_stream=1024,
            fixed_updates='source schedules; no retry',memory_source='accepted session',
            evidence_source='accepted session',deadline_source='accepted session; every full node envelope must fit'),
        prospective_policy_ceilings=costs,unknowns=unknown,complete_total_budget=False,complete_storage_bound=False,
        corpus_disk_formula='E=R+H+J+M; C subtotal; required_free=max(5GiB,4E+B+1GiB)',
        storage_note='trajectory files exclude evaluations/monitors/marking/calibration/custody; variable terms stay unknown',
        external_production=dict(native_solves=None,independent_readback_resolves=None,
            control_admission_and_custody_bytes=None,qualification_scope='separate genuine accepted inputs; no fabricated zero'),
        final_configuration_adopted=False)

def alternatives(config,roster):
    results={}
    for k in (1.,.5,.25):
        candidate=copy.deepcopy(config);candidate['kappa']=k
        graph=V.augment(G.core(candidate,roster),candidate,roster)
        results[str(k)]=ledger(graph,candidate)
    return results
