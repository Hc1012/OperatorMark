"""Prepared selected-contract numerical stage. Inert until a separate grant."""
from pathlib import Path
import math
from msdev.records import read,write,sha,verify
from . import contract_arms as C,development_evaluator as D,evaluation as E

NORMALIZATION_RECIPE=dict(model='Rstar',controls=128,outputs=4096,direction='first_recorded_validation_direction',
    units='existing_model_units',formula='sqrt(mean(JVP(Rstar,x,v)^2))',JVP_observations=128,returned_primal_outputs=128)

def input_plan(root,expected):
    """No generated arrays needed to validate the complete future roster."""
    from .marking import bound_record,identity
    root=Path(root);p=bound_record(root/'CONTRACT_PLAN.json',expected)
    if p.get('domain')!='development' or p.get('adaptive_roster_approved') is not True:raise ValueError('prospective adaptive calibration null roster not approved/bound')
    if len(p.get('adaptive_nulls',[]))!=12 or len(set(p['adaptive_nulls']))!=12:raise ValueError('exact twelve adaptive null models required')
    required={'D1','D2','D3','S1','S2','S3','Rstar'}|set(p['adaptive_nulls'])|{p['accused_model']}
    if not required<=set(p['models']):raise ValueError('complete existing-model roster required')
    if p.get('Rstar_role')!='independently_trained_Rstar10240':raise ValueError('selected Rstar correspondence')
    verify(root,p['files'])
    for name in required:
        if p['models'][name] not in p['files']:raise ValueError('model omitted from bound inventory')
        producer=p['producer_records'][name]
        record=bound_record(root/producer['record'],p['producer_manifests'][name])
        if record['files'].get(producer['producer_member'])!=p['files'][p['models'][name]]:
            raise ValueError('producer/model correspondence differs: '+name)
    if p.get('normalization_recipe')!=NORMALIZATION_RECIPE:raise ValueError('Rstar normalization recipe not bound')
    hashes=[p['files'][p['models'][n]] for n in p['adaptive_nulls']]
    if len(set(hashes))!=12:raise ValueError('duplicate adaptive null identities')
    for name in ('seeds',):
        if p[name] not in p['files']:raise ValueError('contract normalizer/stream binding')
    for h in p['producer_manifests'].values():identity(h)
    return p

def bind_generated(plan_root,plan_sha256,root):
    """After approved CPU generation only: bind actual bytes without new science.

    Model copies retain producer identities; disjoint calibration/search/readout
    control slices are fixed here, not selected from outcomes.
    """
    import shutil,numpy as np
    from .graph import identity
    from .verification_plan import plan as workload
    plan_root=Path(plan_root);root=Path(root);p=input_plan(plan_root,plan_sha256)
    catalog=root/'catalog';c=read(catalog/'DEVELOPMENT_CATALOG.json');runtime=c['runtime_sha256'];source=sha(C.__file__)
    generated=root/'contract_inputs';verify(generated,read(generated/'MANIFEST.json')['files'])
    models=root/'contract_models';models.mkdir(exist_ok=False)
    names={}
    for name,relative in p['models'].items():
        to=models/(name+'.pt');shutil.copyfile(plan_root/relative,to);names[name]=to.relative_to(root).as_posix()
    shutil.copyfile(plan_root/p['seeds'],root/'CONTRACT_SEEDS.json')
    seed=read(root/'CONTRACT_SEEDS.json')['values'];assets=root/'contract_assets';assets.mkdir(exist_ok=False)
    def save(name,a):
        path=assets/(name+'.npy')
        with path.open('xb') as f:np.save(f,a,allow_pickle=False)
        return path.relative_to(root).as_posix()
    rho=np.load(catalog/'final_rho.npy',allow_pickle=False)
    controls={phase:dict(rho=save(phase+'_rho',rho[lo:lo+128]),x=save(phase+'_x',rho[lo:lo+128].reshape(128,64).astype(np.float32)))
              for phase,lo in (('calibration',0),('selection',128),('readout',256))}
    rng=np.random.Generator(np.random.PCG64(seed['accuser_keys']));v=rng.normal(size=(4096,64));v=(v/np.linalg.norm(v,axis=1,keepdims=True)).astype(np.float32)
    keys=dict(v=save('accuser_v',v),coeff=save('accuser_coeff',rng.normal(size=(4096,16))))
    normalization_v=save('normalization_v',np.load(catalog/'final_v.npy',allow_pickle=False)[:128,0])
    scale=None  # Measured only in the separately approved GPU stage.
    files={}
    for folder in (catalog,generated,models,assets):
        for file in folder.rglob('*'):
            if file.is_file():files[file.relative_to(root).as_posix()]=sha(file)
    files['CONTRACT_SEEDS.json']=sha(root/'CONTRACT_SEEDS.json')
    def row(label,candidate,reference,control,key_v,key_coeff,**extra):
        b=dict(candidate=identity(dict(kind='F64',manifest=sha(generated/'MANIFEST.json'))) if candidate=='F64' else files[names[candidate]],
            reference=files[names[reference]],runtime=runtime,controls=files[control['x']],basis=c['entries']['public_basis']['sha256'],
            scoring=c['source_pins']['tools/msdev/scoring.py'],procedure=identity(dict(source=source,keys=[files[key_v],files[key_coeff]])))
        record=dict(reference_sha256=b['reference'],candidate_sha256=b['candidate'],controls_sha256=b['controls'],
            key_population_sha256=identity([files[key_v],files[key_coeff]]),producer_sha256=plan_sha256)
        return dict(label=label,candidate=candidate,reference=reference,**control,key_v=key_v,key_coeff=key_coeff,scale=scale,bindings=b,record=record,**extra)
    overwrites=[]
    for i in range(3):
        prefix=f'contract_inputs/overwrite_{i}_';ref='D'+str(i+1);control=dict(x=prefix+'x.npy',rho=prefix+'rho.npy')
        for candidate in [n for n in ('D1','D2','D3') if n!=ref]+['S'+str(i+1),'F64']:
            overwrites.append(row(ref+'/'+candidate,candidate,ref,control,prefix+'key_v.npy',prefix+'key_coeff.npy',
                utility_v=prefix+'v.npy',truth_jvp=prefix+'truth_key_jvp.npy'))
    adaptive=[row('null/'+n,n,'Rstar',controls['calibration'],keys['v'],keys['coeff']) for n in p['adaptive_nulls']]
    calibration_binding=dict(roster=plan_sha256,runtime=runtime,source=source,controls=identity({n:files[v['x']] for n,v in controls.items()}),keys=identity({n:files[v] for n,v in keys.items()}))
    selection_binding=dict(reference=files[names['Rstar']],keys=identity({k:files[v] for k,v in keys.items()}),controls=files[controls['selection']['x']],
        source=source,runtime=runtime,calibration=plan_sha256)
    r=dict(domain='development',source_sha256=sha(__file__),files=files,models=names,catalog='catalog',catalog_sha256=sha(catalog/'DEVELOPMENT_CATALOG.json'),
        normalization=dict(recipe=NORMALIZATION_RECIPE,x=controls['calibration']['x'],v=normalization_v),
        seeds='CONTRACT_SEEDS.json',counts=workload(),Rstar_role=p['Rstar_role'],adaptive_roster_approved=True,overwrite_nulls=overwrites,adaptive_nulls=adaptive,
        overwrite_calibration_bindings=calibration_binding,adaptive_calibration_bindings=calibration_binding,selection_bindings=selection_binding,
        overwrite=dict(x='contract_inputs/overwrite_0_x.npy',v='contract_inputs/overwrite_0_v.npy',rho='contract_inputs/overwrite_0_rho.npy',
            key_v='contract_inputs/overwrite_0_key_v.npy',key_coeff='contract_inputs/overwrite_0_key_coeff.npy'),
        adaptive_selection=row('selection',p['accused_model'],'Rstar',controls['selection'],keys['v'],keys['coeff']),
        adaptive_readout=row('readout',p['accused_model'],'Rstar',controls['readout'],keys['v'],keys['coeff']),
        development_readout_commitment=identity(dict(controls=files[controls['readout']['x']],source=plan_sha256)))
    write(root/'CONTRACT_VERIFICATION.json',r);return r

def metadata(root,expected):
    root=Path(root)
    from .marking import bound_record
    r=bound_record(root/'CONTRACT_VERIFICATION.json',expected)
    if r.get('domain')!='development' or r.get('source_sha256')!=sha(__file__):raise ValueError('development contract verification binding')
    verify(root,r['files'])
    if len(r.get('overwrite_nulls',[]))!=12 or len(r.get('adaptive_nulls',[]))!=12:raise ValueError('independently precommitted calibration null rosters missing')
    if not r.get('adaptive_roster_approved') or r.get('Rstar_role')!='independently_trained_Rstar10240':raise ValueError('attacker null roster/selected reference not bound')
    if r.get('counts')!=__import__('msfinal.verification_plan',fromlist=['plan']).plan():raise ValueError('complete stage workload binding differs')
    return r

def run(root,expected,out,check,*,runtime_check):
    root=Path(root);r=metadata(root,expected);check()
    c,spec=D.context(root/r['catalog'],r['catalog_sha256'],'final')
    if runtime_check()!=c.record['runtime_sha256']:raise ValueError('exact runtime mismatch')
    import numpy as np,torch
    from msdev import scoring as S,kernels as K
    import q5_model as Q
    out=Path(out);out.mkdir(exist_ok=False);store=E.ProjectionStore(out/'projections')
    models={}
    def model(name):
        if name not in models:
            file=root/r['models'][name];m=Q.Q5FNO().to('cuda').eval();m.load_state_dict(torch.load(file,map_location='cpu',weights_only=True),strict=True);models[name]=m
        return models[name]
    def array(name):return np.load(root/name,allow_pickle=False)
    def tensor(name):return torch.from_numpy(array(name)).to('cuda')
    basis=torch.from_numpy(D.Access.array(c,'public_basis','marking',(16,4096),'float64')).to('cuda')
    def cases(rows):
        values=[]
        for row in rows:
            keys=tensor(row['key_v']);coeff=tensor(row['key_coeff']);x=tensor(row['x']);rho=tensor(row['rho'])
            candidate=None if row['candidate']=='F64' else model(row['candidate'])
            value=dict(label=row['label'],candidate=candidate,reference=model(row['reference']),x=x,rho=rho,
                coeff=coeff.reshape(-1,16),v=keys.reshape(-1,64),basis=basis,scale=row['scale'],bindings=row['bindings'],record=row['record'])
            if candidate is None:value['truth_jvp']=tensor(row['truth_jvp'])
            if 'utility_v' in row:
                value['utility_v']=tensor(row['utility_v']);value['utility_direction_sha256']=r['files'][row['utility_v']]
            values.append(value)
        return values
    completed=[]
    try:
        cal=C.calibrate_numerical(cases(r['overwrite_nulls']),r['overwrite_calibration_bindings'],out/'overwrite_calibration',check,arm='overwrite',store=store)
        C.calibration_readback(out/'overwrite_calibration/CALIBRATION.json',sha(out/'overwrite_calibration/CALIBRATION.json'),arm='overwrite',bindings=r['overwrite_calibration_bindings'])
        data=r['overwrite'];x=tensor(data['x']);v=tensor(data['v']);rho=array(data['rho']);key=array(data['key_v']);coeff=tensor(data['key_coeff'])
        pattern=K.patterns(coeff.reshape(1,16),basis)[0].float()
        seeds=read(root/r['seeds'])['values']
        marked,result=C.overwrite(model('D1'),x,v,rho,key,pattern,seeds['overwrite_0_initialization'],seeds['overwrite_0_shuffle'],cal,out/'overwrite',check)
        C.overwrite_readback(out/'overwrite',check);completed.append('overwrite')
        # Endpoint F/JVP/reverse after actual320 updates; full evaluator itself
        # is exercised separately, not repeated for every recipe.
        z=x[:16].clone().requires_grad_(True);y,j=torch.func.jvp(marked,(z,),(v[:16],));g=torch.autograd.grad(y.square().mean(),z)[0]
        write(out/'OVERWRITE_ENDPOINT.json',dict(finite=all(bool(torch.isfinite(a).all()) for a in (y,j,g)),updates=320,full_utility_qualification=False))
        if not all(torch.isfinite(a).all() for a in (y,j,g)):raise FloatingPointError('overwrite endpoint')
        reference=model('D1');x=torch.from_numpy(D.Access.array(c,'final_x','final',(512,64),'float32')[:16]).to('cuda')
        v=torch.from_numpy(D.Access.array(c,'final_v','final',(512,2,64),'float32')[:16,0]).to('cuda')
        for sd,role in ((.0001,'noise_small'),(.001,'noise_large')):
            noisy=C.noise_model(reference,sd,seeds[role],out/role,binding=dict(model=r['files'][r['models']['D1']],phase='development',stream=r['files'][r['seeds']]+':'+role),check=check)
            y=noisy(x);_,j=noisy.pathwise_jvp(x,v);z=x.detach().clone().requires_grad_(True);target=y.detach();opt=torch.optim.Adam([z],lr=.03)
            for i in range(2):
                check();opt.zero_grad(set_to_none=True);loss=(noisy(z)-target).square().mean();g=torch.autograd.grad(loss,z)[0]
                if not torch.isfinite(g).all():raise FloatingPointError('noise paired gradient')
                z.grad=g;opt.step()
                with torch.no_grad():z.clamp_(0,1)
            values=E._detector_scores(noisy,c,'final',check,model_sha256=r['files'][r['models']['D1']],wrapper_steps=None,store=store,access=D.Access,spec=spec)
            write(out/(role+'_READBACK.json'),dict(C.noise_readback(out/role,check),scores=values,pathwise_B3=True,independent_null_cohort=False));completed.append(role)
        api=C.APIEvaluation(out/'API');rows=[row.tolist() for row in x.detach().cpu()]
        controls=[row for value in rows for row in (value,value)]+[rows[0],rows[0]]
        def compute(control):
            with torch.no_grad():return reference(torch.tensor([control],dtype=torch.float32,device='cuda'))[0].cpu().tolist()
        for policy in ('paired_deny','cache','rate'):
            api.forward(r['files'][r['models']['D1']],'D1','development',policy,controls,compute,check)
        write(out/'API_READBACK.json',C.api_readback(out/'API'));completed.append('API')
        # One recorded JVP observation per control, including its returned primal.
        norm=r['normalization'];nx=tensor(norm['x']);nv=tensor(norm['v']);primal=[];tangents=[]
        if norm['recipe']!=NORMALIZATION_RECIPE or tuple(nx.shape)!=(128,64) or tuple(nv.shape)!=(128,64):raise ValueError('normalizer recipe/input shape')
        with torch.no_grad():
            for lo in range(0,128,16):
                check();y,j=torch.func.jvp(model('Rstar'),(nx[lo:lo+16],),(nv[lo:lo+16],))
                if tuple(y.shape)!=(16,4096) or not torch.isfinite(y).all() or not torch.isfinite(j).all():raise ValueError('normalizer observations')
                primal.append(y.cpu());tangents.append(j.cpu())
        observed=torch.cat(tangents);scale=float(observed.double().square().mean().sqrt())
        if not math.isfinite(scale) or scale<=0:raise ValueError('Rstar normalization unavailable')
        from .materials import save_tensor_record
        save_tensor_record(out/'RSTAR_NORMALIZATION.pt',dict(primal=torch.cat(primal),jvp=observed))
        write(out/'RSTAR_NORMALIZATION.json',dict(recipe=norm['recipe'],value=scale,model_sha256=r['files'][r['models']['Rstar']],
            controls_sha256=r['files'][norm['x']],directions_sha256=r['files'][norm['v']],runtime_sha256=c.record['runtime_sha256'],
            observations_sha256=sha(out/'RSTAR_NORMALIZATION.pt'),JVP_observations=128,returned_primal_outputs=128))
        for row in [*r['adaptive_nulls'],r['adaptive_selection'],r['adaptive_readout']]:row['scale']=scale
        r['adaptive_calibration_bindings']=dict(r['adaptive_calibration_bindings'],normalization=sha(out/'RSTAR_NORMALIZATION.json'))
        ac=C.calibrate_numerical(cases(r['adaptive_nulls']),r['adaptive_calibration_bindings'],out/'adaptive_calibration',check,arm='adaptive_accuser',store=store)
        selected=None
        for phase in ('selection','readout'):
            case=cases([r['adaptive_'+phase]])[0]
            scores=C.score_grid(case['candidate'],case['reference'],case['x'],case['rho'],case['coeff'],case['v'],basis,case['scale'],bindings=case['bindings'],store=store,check=check)
            if phase=='selection':selected=C.adaptive_select(scores,ac,dict(r['selection_bindings'],calibration=sha(out/'adaptive_calibration/CALIBRATION.json')),out/'SELECTION.json')
            else:C.adaptive_final(out/'SELECTION.json',sha(out/'SELECTION.json'),scores,out/'ACCUSATION.json',opening_sha256=r['development_readout_commitment'])
        completed.append('adaptive_accuser')
        write(out/'RESULT.json',dict(complete=True,completed=completed,updates=320,native_calls=0,domain='development',full_final_acceptance=False))
    except BaseException as exc:
        write(out/'FAILURE.json',dict(error=repr(exc),completed=completed,complete=False));raise
