"""Source-derived core membership; proposal dependencies never become approval."""
import hashlib
import json
import re
from . import attacks as T, owners

OBLIGATIONS = {
 '01_same_FNO_distillation':'victim_query_student', '02_DeepONet':'DeepONet_adapter',
 '03_UNet':'UNet_adapter', '04_capacity_sweep':'capacity_adapters',
 '05_forward_extraction':'victim_query_student', '06_victim_JVP':'victim_JVP_adapter',
 '07_solver_canonicalisation':'P256_task_challenger', '08_residual_substitution':'core',
 '09_fourier_filter':'filter_adapter', '10_clean_finetuning':'core',
 '11_pruning_quantisation':'pruning_quantisation_adapters', '12_averaging':'averaging_adapter',
 '13_buyer_collusion':'conditional_customer_keys', '14_overwrite':'attacker_reference_calibration',
 '15_random_key_flattening':'core', '16_rounding_precision':'nonsmooth_service_contract',
 '17_noise':'stochastic_calibration_contract', '18_API_policy':'API_task_detector_contract',
 '19_remeshing':'remeshing_adapter', '20_null_false_accuser':'adaptive_accuser_contract',
 'B1':'unresolved_failed_comparator_disposition', 'topology_dcap':'physical_task_qualification',
 'same_data_ordinary':'conditional_identical_exposure', 'ordinary_1024':'conditional_query_tier',
 'scratch_R4':'conditional_solver_attribution', 'train_through_wrapper':'conditional_adaptive_wrapper',
 'disjoint_B0':'conditional_data_scope', 'blackbox_derivative':'conditional_query_protocol',
 'N16':'conditional_population_analysis', 'calibration':'independent_final_calibration',
 'native_runtime':'independent_N0_N1_runtime', 'production_custody':'independent_production_readback',
 'E3':'candidate_specific_full_revalidation', 'export':'private_export_acceptance',
}


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def programme_readiness(decisions):
    """Exact inventory of decisions, not acceptance of self-declared dispositions."""
    if set(decisions)!=set(OBLIGATIONS):raise ValueError('complete original obligation/conditional inventory required')
    rows=[]
    for name,dependency in OBLIGATIONS.items():
        value=decisions[name]
        if not isinstance(value,dict) or value.get('status') not in ('unresolved','proposed','accepted'):
            raise ValueError('explicit obligation status required: '+name)
        rows.append(dict(id=name,dependency=dependency,decision=value,
            implementation_available=dependency=='core',
            independently_verified=False))
    # No full-programme verification issuer exists here. In particular an
    # "accepted" string does not retire B1, topology or an absent adapter.
    return dict(complete_programme_ready=False,obligations=rows,
                reason='independent scope/adoption and broader adapter/readiness evidence required')


def core(config,roster):
    ids=config.get('owners');k=config.get('kappa')
    if (not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)) or list(roster)!=ids
            or any(not isinstance(o,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',o) for o in ids)
            or config.get('population')!=len(ids) or k not in (1.,.5,.25) or isinstance(k,bool)
            or set(config.get('attack_variants',[]))!=set(T.IMPLEMENTED)
            or config.get('replacement_length')!=10240):
        raise ValueError('explicit complete supported core configuration/roster required')
    base=config.get('shared_baseline_id')
    if not isinstance(base,str) or base in ids or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',base):
        raise ValueError('distinct safe shared baseline identifier required')
    if config.get('B1_disposition')!='unavailable':raise ValueError('B1 implementation unresolved; no invented arm')
    reference=config.get('shared_baseline_reference_owner')
    if reference not in ids:raise ValueError('explicit shared-baseline evaluation/key reference owner required')
    readiness=programme_readiness(config.get('programme_obligations',{}))
    trajectories={};models={o:{} for o in ids};missing=[]
    def add(owner,label,trajectory,checkpoint,wrapper=None):
        if label in models[owner]:raise ValueError('duplicate final model alias')
        models[owner][label]=dict(trajectory=trajectory,checkpoint=checkpoint,wrapper_steps=wrapper)
    for o in ids:
        trajectories['owners/'+o]=owners.schedule()
        add(o,'O','owners/'+o,'cooled81920')
        r=roster[o];marked=r.get('marked_checkpoint_complete') is True;q=r.get('B3_qualified') is True
        if q and not marked:raise ValueError('qualified mark has no completed checkpoint')
        if marked:add(o,'M3','marking/'+o,'MODEL')
        else:missing.append(dict(owner=o,measurement='ordinary_P2',reason='missing completed mark'))
        jobs=[('O',v) for v in ('ordinary','replacement','A4','A5B2','sham')]
        if marked:jobs.append(('M3','ordinary'))
        if q:jobs.extend([('M3','A4'),('M3','A5B3')])
        for parent,v in jobs:
            key='attacks/'+o+'_'+parent+'_'+v
            schedule=T.plan(v,k,replacement_length=10240);trajectories[key]=schedule
            for node in schedule:
                if not node['evaluate']:continue
                if v=='replacement':
                    # Half length is the retained S(L) monitor, not another
                    # full final evaluation or outcome-based selection.
                    labels={'raw1920':'Rraw1920','cooled10240':'Rstar'}
                    if node['id'] not in labels:continue
                    label=labels[node['id']]
                elif v=='sham':label='S'
                else:label=('Aord' if v=='ordinary' else v)+'_'+parent+'_'+node['id']
                add(o,label,key,node['id'])
        # Qualified-only wrappers: the ordinary quality exception does not leak.
        for parent in ['O']+(['M3'] if q else []):
            for label in (parent,'Aord_'+parent+'_raw1920'):
                base=models[o][label]
                for m in (16,256):add(o,'W'+str(m)+'_'+label,base['trajectory'],base['checkpoint'],m)
    trajectories['owners/'+config['shared_baseline_id']]=owners.schedule()
    add(reference,'B10','owners/'+config['shared_baseline_id'],'cooled81920')
    keyed={o:['O/'+j for j in ids if j!=o]+['Rstar/'+j for j in ids]+['S/'+o,'F64'] for o in ids}
    b2={}
    for o in ids:
        b2[o]={}
        for label in models[o]:
            if label in ('O','B10','Rstar','Rraw1920'):continue
            matched=[j+'/'+label for j in ids if j!=o]
            independent=[j+'/O' for j in ids if j!=o]+[j+'/Rstar' for j in ids]
            absent=[name for name in matched if name.split('/',1)[1] not in models[name.split('/',1)[0]]]
            b2[o][label]=dict(matched=matched,independent=independent,missing=absent,
                matching='same source type/arm/checkpoint/wrapper; Aord data matched in distribution only')
    value=dict(version=1,models=models,trajectories=trajectories,keyed_nulls=keyed,b2_groups=b2,
        planned_owners=ids,missing_primary=missing,shared_substitution='F64',baseline_reference_owner=reference,
        marking_updates_per_owner=320,readiness=readiness,core_only=True)
    value['sha256']=identity(value);return value


def workload(graph,qualification_p256):
    from .update_journal import normal_bound
    if type(qualification_p256) is not bool:raise ValueError('calibration P256 policy unresolved')
    per={};updates=files=anchors=0
    for key,nodes in graph['trajectories'].items():
        count=T.updates(nodes);periodic=sum((n['end']-n['start'])//2048 for n in nodes)
        checkpoints=len(nodes)+periodic;models=sum(n['evaluate'] for n in nodes)
        # D5: one tensor plus its publication receipt. Failed partials are separate.
        count_files=3+2*checkpoints+2*models+len(nodes)+periodic
        per[key]=dict(updates=count,journal_events=2*count,periodic_anchors=periodic,
            normal_journal=normal_bound(nodes),
            tensor_files_including_partials=checkpoints+models,publication_receipts=checkpoints+models,
            failed_partial_bytes=None,core_writer_files=count_files)
        updates+=count;files+=count_files;anchors+=periodic
    n=len(graph['planned_owners']);states=sum(len(v) for v in graph['models'].values())
    wrappers=sum(d['wrapper_steps'] is not None for m in graph['models'].values() for d in m.values())
    monitors=2*(n+1)+2*n # owners/B10 and R*; exact fixed two checkpoint branches
    available_b2=[g for groups in graph['b2_groups'].values() for g in groups.values() if not g['missing']]
    b2_comparisons=sum(1+len(g['matched'])+len(g['independent']) for g in available_b2)
    rates=dict(D9=.0741568,ordinary=.0335,replacement=.0335,sham=.0765,A4=.0741644,A5B2=.0912556,A5B3=.07437171595)
    projected=sum(v['updates']*rates['D9' if key.startswith('owners/') else key.rsplit('_',1)[1]] for key,v in per.items())+320*n*.5066
    return dict(trajectory_updates=updates,marking_updates=320*n,total_model_updates=updates+320*n,
        trajectories=per,journals=len(per),journal_records=2*updates,
        journal_record_max_bytes=65536,journal_bytes_upper_bound=sum(v['normal_journal']['bytes_upper'] for v in per.values()),
        periodic_anchors=anchors,core_writer_files=files,marking_raw_record_files=320*n,
        final_evaluations=states,wrapper_evaluations=wrappers,qualification_evaluations=n,
        monitor_evaluations=monitors,
        inverse_input_iterations=480*(states+n)+120*monitors,
        native_t128_scoring_calls=64*(states+n)+16*monitors,
        native_p256_scoring_calls=64*states+(64*n if qualification_p256 else 0)+16*monitors,
        substitution=dict(optimization_forwards=7744,adjoints=7680,t128_scoring=64,p256_scoring=64),
        final_validation_controls=512*states,final_random_JVPs=1024*states,final_keyed_JVPs=512*states,
        final_detector_projection_JVPs_upper=(2*states-n)*128*64,
        qualification_detector_projection_JVPs_upper=2*n*256*64,
        keyed_null_projection_JVPs_upper=n*(2*n+1)*128*64,
        F64_null_owner_projection_JVPs=n*128*64,
        F64_null_random_owner_JVPs=n*128*2,
        b2_available_comparisons=b2_comparisons,b2_logical_JVPs=512*b2_comparisons,
        detector_projection_note='logical control-directions, not GPU kernel calls; identity caching may reduce upper bounds',
        keyed_null_comparisons=n*(2*n+1),b2_groups=sum(len(v) for v in graph['b2_groups'].values()),
        reused_measured_rates_seconds_per_update=dict(rates,PN_mark=.5066),
        training_projection_seconds=projected,training_projection_is_total_budget=False,
        bytes_unknown=['serialized models/optimizer states','variable raw/evaluation/monitor JSON',
            'projection caches','failure/partial records','audit/production/custody','export container'],
        cost_unknown=['complete evaluation and wrapper throughput','native production/P1/readback',
            'journal fsync latency','setup/session idle','transfers/export','permitted recovery'],
        complete_total_budget=False,complete_storage_bound=False)
