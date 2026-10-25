"""Explicit candidate definitions. Availability is not scientific adoption."""
from .graph import identity

ARCHITECTURES=('fno32','deeponet','unet','fno16','fno64','fno2')
STUDENTS=tuple('extract_'+a for a in ARCHITECTURES)+('extract_jvp',)
HEALING=tuple('heal'+str(p) for p in (20,50,80))
TRANSFORMS=('fft4','fft8','fft12','prune20','prune50','prune80','quant8',
            'mix25','mix50','mix75','round1e4','round1e3','storagefp16','storagebf16','remesh16','remesh32')
AVAILABLE=STUDENTS+HEALING+TRANSFORMS
# These kernels/contracts passed development verification. They are not members
# of this deterministic student/transform dispatcher; final graph integration
# and independently accepted inputs/calibration remain separate obligations.
SELECTED_CONTRACT_ROUTE=('overwrite','noise_small','noise_large','API_paired_deny','API_cache','API_rate','adaptive_accuser','P256')
BLOCKED={'topology_dcap':'physical qualification and task/filter/feature/load definition',
    'buyer_collusion':'per-customer trigger, two-key inputs and calibration',
    'B1':'failed comparator disposition or separately approved scientific repair'}

def definition(name):
    if name not in AVAILABLE:raise ValueError('unimplemented or scientifically unresolved variant: '+str(name))
    common=dict(version=1,variant=name,input_shape=[64],output_shape=[4096],target_scale=.01,
                dtype='float32',scientific_adoption=False)
    if name in STUDENTS:
        jvp=name=='extract_jvp';arch='fno32' if jvp else name[8:]
        common.update(kind='student',architecture=arch,updates=1920,checkpoints=[320,960,1920],
            batch=16 if jvp else 32,lr=.002,optimizer='Adam',forward_queries=256,
            jvp_queries=256 if jvp else 0,loss='normalized_forward_plus_half_JVP' if jvp else 'forward_MSE_model_units',
            label_source='received_model',shuffle='bound ordinary initial shuffle',initialization='explicit admitted stream')
    elif name in HEALING:
        common.update(kind='healing',architecture='fno32',pruning_percent=int(name[4:]),updates=1920,
            checkpoints=[320,960,1920],batch=32,lr=.002,optimizer='fresh Adam',loss='forward_MSE_model_units',
            label_source='ordinary solver labels',mask='fixed global magnitude, stable flattened-index tie break')
    else:
        common.update(kind='transform',architecture='fno32',updates=0)
        if name.startswith('mix'):common['endpoint_weight']=int(name[3:])/100
        if name.startswith('fft'):common.update(modes=int(name[3:]),fft_norm='ortho',mask='abs(kx)<m and 0<=ky<m')
        if name.startswith('prune'):common.update(pruning_percent=int(name[5:]),mask='fixed global magnitude, stable flattened-index tie break')
        if name=='quant8':common.update(bits=8,quantization='symmetric per-output-channel, nearest-even, complex real/imag separate, zero scale unchanged')
        if name.startswith('storage'):common['storage_dtype']=name[7:]
        if name.startswith('round'):common.update(quantum=.0001 if name=='round1e4' else .001,derivative='actual rounding; no straight-through')
        if name.startswith('remesh'):common.update(coarse=int(name[6:]),down='area average',up='bilinear align_corners=False')
    return common

def selected(config):
    from .final_contracts import specs
    specs(config,deferred=True)
    value=config.get('additional_variants',{})
    if not isinstance(value,dict):raise ValueError('explicit additional variant definitions required')
    for name,spec in value.items():
        if spec!=definition(name):raise ValueError('candidate variant definition differs: '+name)
    return list(value)

def schedule(name):
    spec=definition(name)
    if not spec['updates']:return []
    recipe='D9' if name=='extract_jvp' else 'forward32';prior=None;start=0;nodes=[]
    for end in spec['checkpoints']:
        node=dict(id='raw'+str(end),parent=prior,start=start,end=end,cooldown=False,evaluate=True,recipe=recipe)
        nodes.append(node);prior=node['id'];start=end
    return nodes

def augment(graph,config,roster):
    """Called before opening. Every extra model has an explicit procedure."""
    import copy
    g=copy.deepcopy(graph);g.pop('sha256',None);g['additional_jobs']=[]
    names=selected(config)
    covers={'01_same_FNO_distillation':['extract_fno32'],'05_forward_extraction':['extract_fno32'],
        '02_DeepONet':['extract_deeponet'],'03_UNet':['extract_unet'],
        '04_capacity_sweep':['extract_fno16','extract_fno64','extract_fno2'],
        '06_victim_JVP':['extract_jvp'],'09_fourier_filter':['fft4','fft8','fft12'],
        '11_pruning_quantisation':['prune20','prune50','prune80','heal20','heal50','heal80','quant8'],
        '12_averaging':['mix25','mix50','mix75'],'16_rounding_precision':['round1e4','round1e3','storagefp16','storagebf16'],
        '19_remeshing':['remesh16','remesh32']}
    for row in g['readiness']['obligations']:
        if row['id'] in covers:
            row.update(candidate_implementations=covers[row['id']],
                implementation_available=True,selected_candidate_variants=[v for v in covers[row['id']] if v in names],
                numerical_verification=False,scientific_adoption=False)
    for owner in g['planned_owners']:
        for parent in ['O']+(['M3'] if roster[owner]['B3_qualified'] is True else []):
            base=g['models'][owner][parent]
            for name in names:
                spec=definition(name);key='additional/'+owner+'_'+parent+'_'+name
                if spec['updates']:
                    g['trajectories'][key]=schedule(name)
                    g['additional_jobs'].append(dict(owner=owner,parent=parent,variant=name,path=key))
                    for node in schedule(name):
                        label=name+'_'+parent+'_'+node['id']
                        g['models'][owner][label]=dict(trajectory=key,checkpoint=node['id'],wrapper_steps=None,
                            procedure=dict(architecture=spec['architecture'],variant=name,transform=None),dependencies={},source_parent=parent)
                else:
                    label=name+'_'+parent;deps={}
                    if name.startswith('mix'):deps['endpoint']=g['models'][owner]['Aord_'+parent+'_raw1920']
                    g['models'][owner][label]=dict(trajectory=base['trajectory'],checkpoint=base['checkpoint'],wrapper_steps=None,
                        procedure=dict(architecture='fno32',variant=name,transform=name),dependencies=deps,source_parent=parent)
    # Recompute every matched group including newly introduced procedures.
    for owner in g['planned_owners']:
        for label in g['models'][owner]:
            if label in ('O','B10','Rstar','Rraw1920'):continue
            matched=[o+'/'+label for o in g['planned_owners'] if o!=owner]
            independent=[o+'/O' for o in g['planned_owners'] if o!=owner]+[o+'/Rstar' for o in g['planned_owners']]
            missing=[v for v in matched if v.split('/')[1] not in g['models'][v.split('/')[0]]]
            g['b2_groups'][owner][label]=dict(matched=matched,independent=independent,missing=missing,
                matching='same source type/arm/checkpoint/wrapper; distribution-matched controls')
    from .final_contracts import augment as contracts
    contracts(g,config,roster)
    from . import d2
    g['d2_policy']=d2.POLICY;g['d2_cells']=d2.cells(g)
    g['sha256']=identity(g);return g

def workload(graph):
    from .update_journal import normal_bound
    jobs=graph.get('additional_jobs',[]);owners={}
    for j in jobs:
        if j['variant'] in STUDENTS:owners[(j['owner'],j['parent'])]=True
    models=[d for group in graph['models'].values() for d in group.values() if d.get('procedure')]
    updates=sum(definition(j['variant'])['updates'] for j in jobs)
    jvp_sources={(j['owner'],j['parent']) for j in jobs if j['variant']=='extract_jvp'}
    conditional=sum(len(c['members']) for c in graph.get('d2_cells',{}).values())
    return dict(model_updates=updates,trajectories=len(jobs),journals=len(jobs),journal_records=2*updates,
        journal_bytes_upper_bound=sum(normal_bound(schedule(j['variant']))['bytes_upper'] for j in jobs),
        retained_model_checkpoints=3*len(jobs),periodic_anchors=0,
        received_forward_queries=256*len(owners),received_JVP_queries=256*len(jvp_sources),
        full_model_evaluations=len(models),inverse_input_iterations=480*len(models),
        evaluation_counts_are_upper_bounds=True,unconditional_full_evaluations=len(models)-conditional,
        conditional_task_completions_upper=conditional,diagnostic_evaluations=conditional,
        diagnostic_copy_policy='retain originals plus one verified copy per conditional result; variable JSON bytes unknown',
        native_scoring_T128=64*len(models),native_scoring_P256=64*len(models),
        core_trajectory_writer_files=18*len(jobs),
        cache_payload_bytes_per_source_forward=256*4096*4,
        cache_payload_bytes_per_source_JVP=256*4096*4,
        transfer_setup_idle_export_recovery_seconds=None,full_evaluation_seconds=None,
        training_seconds=None,complete_budget=False,complete_storage_bound=False)
