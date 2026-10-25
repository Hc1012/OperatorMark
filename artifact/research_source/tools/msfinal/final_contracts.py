"""Parameterized final selected-contract wave. No defaults, issuer or launch.

Pre-opening overwrite endpoints and signed selections are frozen into the same
opening as all control/null memberships. Each stochastic/service evaluator runs
once per model/phase; its live state is not recreated between batches.
"""
from dataclasses import dataclass,replace
from pathlib import Path
import io,json,hashlib
from operatormark.corpus.state_v2_prerequisites import require_accepted
from msdev.records import read,write,sha
from . import materials as A,marking as M,contract_arms as C,evaluation as E

NAMES=('overwrite','noise_small','noise_large','API_paired_deny','API_cache','API_rate','adaptive_accuser','P256')
NORMALIZATION_RECIPE=dict(model='Rstar',controls=128,outputs=4096,direction='first_recorded_validation_direction',
    units='existing_model_units',formula='sqrt(mean(JVP(Rstar,x,v)^2))',JVP_observations=128,returned_primal_outputs=128)

def specs(config,*,deferred=False):
    """Selected-contract procedure commitments.

    deferred=True is the pre-qualification structural form: per-owner control
    identities may be explicitly null (all phases together) and matched null
    memberships may be null. augment() never accepts that form; it needs bind().
    """
    values=config.get('selected_contracts')
    if not isinstance(values,dict) or set(values)!=set(NAMES):raise ValueError('explicit complete selected-contract configuration required')
    for name,s in values.items():
        if not isinstance(s,dict) or type(s.get('enabled')) is not bool:raise ValueError('explicit selected-contract disposition required')
        if not s['enabled']:
            M.identity(s.get('disposition_sha256'));continue
        if name=='P256':
            if s.get('target_family')!='T128':raise ValueError('P256 primary target must remain T128')
            continue
        parents=s.get('parents')
        if (not isinstance(parents,list) or not parents or len(set(parents))!=len(parents)
                or not set(parents)<=({'O'} if name=='adaptive_accuser' else {'O','M3'})):
            raise ValueError('explicit supported contract parent population required')
        if (name.startswith('noise') or name.startswith('API')) and parents!=['M3']:
            raise ValueError('D3/D4 retain eligible M3 only; O measurements are omitted')
        if set(s.get('controls',{}))!=set(config['owners']):raise ValueError('all-owner control commitments required')
        phases={'qualification','final'} if name in ('overwrite','adaptive_accuser') else {'final'}
        for commitments in s['controls'].values():
            if not isinstance(commitments,dict) or set(commitments)!=phases:raise ValueError('separate pre-opening/final control commitments required')
            if deferred and all(h is None for h in commitments.values()):continue
            for h in commitments.values():M.identity(h)
            if name=='adaptive_accuser' and len(set(commitments.values()))!=2:raise ValueError('adaptive selection/readout controls must be distinct')
        if name in ('overwrite','adaptive_accuser'):M.identity(s.get('calibration_procedure_sha256'))
        if name=='overwrite' or name.startswith('noise'):
            if set(s.get('nulls',{}))!=set(config['owners']):raise ValueError('explicit all-owner contract null membership required')
            for o in config['owners']:
                if set(s['nulls'][o])!=set(parents):raise ValueError('contract null source types missing')
                for group in s['nulls'][o].values():
                    if (set(group)!= {'matched','independent'} or not isinstance(group['independent'],list)
                            or not (isinstance(group['matched'],list) or (deferred and group['matched'] is None))):
                        raise ValueError('explicit matched/independent lists required')
                    ids=(group['matched'] or [])+group['independent']
                    if len(ids)!=len(set(ids)) or any(not isinstance(v,str) or len(v.split('/'))!=2 for v in ids):raise ValueError('duplicate/invalid contract null identities')
        if name.startswith('API'):
            n=s.get('attempts')
            if type(n) is not int or n!=1024:raise ValueError('D4 requires exactly 1024 ordered API attempts')
    for family in (('noise_small','noise_large'),('API_paired_deny','API_cache','API_rate')):
        if len({values[n]['enabled'] for n in family})!=1:raise ValueError('complete adopted noise/API family required')
    return values

def require_adopted(config):
    """Planning dispositions cannot silently remove D3/D4 at final dispatch."""
    values=specs(config,deferred=True)
    if any(not values[n]['enabled'] for n in NAMES if n.startswith(('noise','API'))):
        raise ValueError('adopted D3/D4 require both noise amplitudes and all three API policies')

BINDING_KIND='main_study_final_contract_bindings'

def pending(config):
    """Explicitly deferred later slots: (contract, owner, field, key)."""
    specs(config,deferred=True);slots=[]
    for name,s in config['selected_contracts'].items():
        for o,c in (s.get('controls') or {}).items():
            slots+=[(name,o,'controls',k) for k,h in c.items() if h is None]
        for o,parents in (s.get('nulls') or {}).items():
            slots+=[(name,o,'nulls',k) for k,g in parents.items() if g['matched'] is None]
    return slots

def bind(config,supplement):
    """Fill exactly the deferred slots from one accepted later record.

    Committed procedure fields, independent cohorts and every non-null identity
    are immutable; the result must satisfy the complete specs() check.
    """
    import copy
    open_slots=set(pending(config));merged=copy.deepcopy(config);sc=merged['selected_contracts'];filled=set()
    if not isinstance(supplement,dict) or not set(supplement)<=set(sc):raise ValueError('contract binding supplement names')
    for name,fill in supplement.items():
        if not isinstance(fill,dict) or not set(fill)<={'controls','nulls'}:raise ValueError('only control identities/matched memberships bind later')
        for o,values in fill.get('controls',{}).items():
            for k,h in values.items():
                if (name,o,'controls',k) not in open_slots:raise ValueError('committed/unknown control slot cannot be bound')
                sc[name]['controls'][o][k]=h;filled.add((name,o,'controls',k))
        for o,values in fill.get('nulls',{}).items():
            for k,matched in values.items():
                if (name,o,'nulls',k) not in open_slots:raise ValueError('committed/unknown null membership cannot be bound')
                sc[name]['nulls'][o][k]['matched']=matched;filled.add((name,o,'nulls',k))
    if filled!=open_slots:raise ValueError('later contract bindings incomplete')
    specs(merged)
    return merged

def augment(g,config,roster):
    values=specs(config);g['contract_jobs']=[];g['contract_unavailable']=[]
    g['contract_omitted']=[dict(owner=o,parent='O',variant=name,measured=False,
        reason='adopted D3/D4 O-parent measurement omitted; clean reference retained')
        for o in g['planned_owners'] for name in NAMES if name.startswith(('noise','API'))]
    g['noise_draw_rule']=C.NOISE_DRAW_RULE
    g['P256_task_enabled']=values['P256']['enabled']
    for name,s in values.items():
        if not s['enabled'] or name=='P256':continue
        for o in g['planned_owners']:
            for parent in s['parents']:
                if parent=='M3' and roster[o]['B3_qualified'] is not True:
                    g['contract_unavailable'].append(dict(owner=o,variant=name,parent=parent,reason='not independently qualified'));continue
                label=name+'_'+parent;node=o+'/'+label
                arm='noise' if name.startswith('noise') else 'API' if name.startswith('API') else name
                j=dict(id=node,owner=o,parent=parent,label=label,variant=name,arm=arm,
                    controls_sha256=s['controls'][o],preopening=name in ('overwrite','adaptive_accuser'),
                    path='selected/'+o+'_'+label)
                if arm=='noise':j.update(sd=.0001 if name=='noise_small' else .001,draw_rule=C.NOISE_DRAW_RULE,claim_scope=C.NOISE_CLAIM)
                if arm=='API':j.update(policy=name[4:],attempts=s['attempts'])
                if name in ('overwrite','adaptive_accuser'):j['calibration_procedure_sha256']=s['calibration_procedure_sha256']
                if arm in ('overwrite','noise'):
                    group=s['nulls'][o][parent]
                    if node in group['matched']+group['independent']:raise ValueError('self included in contract null cohort')
                    j['nulls']=group
                if name=='overwrite':
                    g['models'][o][label]=dict(trajectory=j['path']+'/pre/mark',checkpoint='MODEL',wrapper_steps=None,
                        selected_contract='overwrite',source_parent=parent)
                    g['b2_groups'][o][label]=dict(j['nulls'],missing=[],matching='explicit adopted overwrite cohort')
                elif arm=='noise':
                    g['b2_groups'][o][label]=dict(j['nulls'],missing=[],matching='explicit pathwise noise cohort')
                g['contract_jobs'].append(j)
    known={o+'/'+n for o,m in g['models'].items() for n in m}|{j['id'] for j in g['contract_jobs']}
    for j in g['contract_jobs']:
        if 'nulls' in j:
            j['missing_nulls']=[n for n in j['nulls']['matched']+j['nulls']['independent'] if n not in known]
            g['b2_groups'][j['owner']][j['label']]['missing']=j['missing_nulls']
            for n in j['nulls']['matched']:
                if n.split('/')[1]!=j['label']:raise ValueError('matched cohort must use same arm/source type')
            for n in j['nulls']['independent']:
                if n.split('/')[1] not in ('O','Rstar'):raise ValueError('independent cohort must name explicit clean/replacement identities')
    return g

@dataclass(frozen=True)
class Request:
    root: Path
    sha256: str
    acceptance: object
    calibration: object=None

def request_metadata(request,job,admission,parent_sha,phase):
    """Metadata only before opening: never read referenced final tensor bytes."""
    r=M.bound_record(Path(request.root)/'CONTRACT_INPUTS.json',request.sha256)
    b=dict(A.scope(admission),request_sha256=request.sha256,arm=job['arm'])
    accepted=require_accepted(request.acceptance,kind='main_study_contract_inputs',bindings=b)
    if accepted.get('domain')!='final' or r.get('domain')!='final':raise ValueError('contract requires genuine final-domain acceptance')
    if (r.get('arm')!=job['arm'] or r.get('owner')!=job['owner'] or r.get('received_model_sha256')!=parent_sha
            or r.get('source_sha256')!=sha(C.__file__) or r.get('controls_sha256')!=job['controls_sha256'][phase]
            or r.get('phase')!=phase or r.get('contract')!=C.CONTRACTS[job['arm']]):raise ValueError('contract graph/request binding drift')
    if r.get('runtime_sha256')!=admission.inputs['materials']['runtime']['sha256']:raise ValueError('contract runtime binding drift')
    if r.get('input_catalog_sha256')!=admission.prerequisites.assets.sha256:raise ValueError('contract catalog/custody substitution')
    if job['arm']!='noise' and r.get('files',{}).get('controls',{}).get('sha256')!=r['controls_sha256']:
        raise ValueError('actual contract controls do not match commitment')
    if job['arm']=='noise' and r['controls_sha256']!=admission.prerequisites.assets.sha256:
        raise ValueError('noise evaluation must bind complete accepted control catalog')
    if job['arm']=='adaptive_accuser':
        from .graph import identity
        files=r['files']
        procedure=identity(dict(source=sha(C.__file__),keys=files['keys']['sha256'],directions=files['directions']['sha256'],normalization=files['normalization']['sha256']))
        expected=dict(candidate=parent_sha,reference=files['Rstar']['sha256'],runtime=r['runtime_sha256'],
            controls=files['controls']['sha256'],basis=files['basis']['sha256'],
            scoring=admission.configuration['source_pins']['tools/msdev/scoring.py'],procedure=procedure)
        if r.get('score_bindings')!=expected:raise ValueError('adaptive actual-file cache identity differs')
        if phase=='qualification':
            selection=dict(reference=expected['reference'],keys=identity([files['keys']['sha256'],files['directions']['sha256']]),
                controls=expected['controls'],source=sha(C.__file__),runtime=r['runtime_sha256'],calibration=files['calibration']['sha256'])
            if r.get('selection_bindings')!=selection:raise ValueError('adaptive pre-opening selection bindings differ')
    if job['arm']=='noise':
        if r.get('sd')!=job['sd']:raise ValueError('noise SD drift')
        noise_binding(r)
    if job['arm']=='API' and (r.get('policy')!=job['policy'] or r.get('attempts')!=job['attempts']):raise ValueError('API population/policy drift')
    if job['arm'] in ('overwrite','adaptive_accuser'):
        h=r.get('files',{}).get('calibration',{}).get('sha256')
        cal=require_accepted(request.calibration,kind='main_study_attacker_calibration',
            bindings=dict(A.scope(admission),arm=job['arm'],calibration_sha256=h))
        if (cal.get('domain')!='final' or cal.get('defender_calibration_used') is not False
                or cal.get('procedure_sha256')!=job['calibration_procedure_sha256']):raise ValueError('attacker calibration scope differs')
    allowed={'attack','calibration','qualification'} if phase=='qualification' else {'attack','calibration','final'}
    if any(e.get('purpose') not in allowed for e in r.get('files',{}).values()):
        raise ValueError('unknown or premature contract material purpose')
    return r

def noise_binding(request):
    """Finite commitment address, checked before numerical dispatch."""
    binding=request.get('noise_binding',{})
    M.identity(binding.get('stream'))
    M.identity(request.get('received_model_sha256'))
    if binding!=dict(model=request['received_model_sha256'],phase='final',stream=binding['stream']):
        raise ValueError('noise persistent stream binding')
    return binding

def required_products(graph):
    names=set()
    from .d2 import required_products as d2_required
    names.update(d2_required(graph))
    for j in graph['contract_jobs']:
        if j['preopening']:names.add(j['path']+'/pre')
        if j['arm']=='overwrite':names.add('evaluations/'+j['owner']+'/'+j['label'])
        else:names.add(j['path']+'/final')
        if 'nulls' in j and not j['missing_nulls']:names.add('b2/'+j['owner']+'/'+j['label'])
    if graph['P256_task_enabled']:names.add('shared/P256')
    return names

def completion(graph,products):
    if len(products)!=len(set(products)) or not required_products(graph)<=set(products):
        raise ValueError('selected contract/P256 products missing or duplicated')

class Loader:
    """Concrete safe loader for the existing kernels, from accepted file bytes."""
    def __init__(self,admission,out,check,*,selection=None,store=None):
        self.a=admission;self.out=Path(out);self.check=check;self.selection=selection;self.store=store
    def __call__(self,r,root):
        import numpy as np,torch
        import q5_model as Q
        def data(role):
            e=r['files'][role];p=(root/e['path']).resolve()
            if not p.is_relative_to(root.resolve()):raise ValueError('contract role path escape')
            raw=p.read_bytes()
            if len(raw)!=e['bytes'] or hashlib.sha256(raw).hexdigest()!=e['sha256']:raise ValueError('contract loaded-byte drift: '+role)
            return raw
        def array(role):
            value=np.load(io.BytesIO(data(role)),allow_pickle=False)
            shapes={'controls':((r['attempts'],64) if r['arm']=='API' else (128,64),'float32'),
                'rho':((128,8,8),'float64'),'pattern':((4096,),'float32'),'key':((64,),'float32'),
                'keys':((1,16) if r['arm']=='overwrite' else (4096,16),'float64'),'basis':((16,4096),'float64'),
                'directions':((4096,64) if r['arm']=='adaptive_accuser' else (128,64),'float32')}
            shape,dtype=shapes[role]
            if value.shape!=shape or str(value.dtype)!=dtype or not np.isfinite(value).all():raise ValueError('contract array population/dtype/nonfinite: '+role)
            return value
        def tensor(role):return torch.from_numpy(array(role)).to('cuda')
        def model(role):
            m=Q.Q5FNO().to('cuda').eval();m.load_state_dict(torch.load(io.BytesIO(data(role)),map_location='cpu',weights_only=True),strict=True);return m
        def evaluate(m,out,check):
            return E.evaluate_model(m,self.a,out,check,phase='final',model_sha256=r['received_model_sha256'],projection_store=self.store)
        arm=r['arm']
        if arm=='noise':
            stream=json.loads(data('streams'))
            if noise_binding(r)['stream']!=stream.get('commitment'):
                raise ValueError('noise stream file/commitment mismatch')
        result=dict(received=model('received'),evaluate=evaluate)
        if arm=='overwrite':
            from msdev.kernels import patterns
            x=tensor('controls');rho=array('rho');pattern=patterns(tensor('keys'),tensor('basis'))[0].float()
            if not torch.equal(x,torch.from_numpy(rho.reshape(128,64).astype('float32')).to('cuda')):
                raise ValueError('overwrite physical/model controls differ')
            if not torch.equal(pattern,tensor('pattern')):raise ValueError('overwrite key/pattern correspondence differs')
            seeds=json.loads(data('shuffle'));result.update(x=x,v=tensor('directions'),rho=rho,
                key_v=array('key'),pattern=pattern,initialization=seeds['initialization'],shuffle=seeds['shuffle'],
                calibration_path=root/r['files']['calibration']['path'])
        elif arm=='noise':
            result['seed']=stream['seed']
        elif arm=='API':
            x=array('controls')
            if x.shape!=(r['attempts'],64):raise ValueError('API ordered input population')
            result['controls']=x.tolist()
            def compute(x):
                with torch.no_grad():return result['received'](torch.tensor([x],device='cuda',dtype=torch.float32))[0].cpu().tolist()
            result['compute']=compute
        else:
            norm=json.loads(data('normalization'))
            if (not M.finite(norm.get('value')) or norm['value']<=0 or norm.get('model_sha256')!=r['files']['Rstar']['sha256']
                    or norm.get('recipe')!=NORMALIZATION_RECIPE):raise ValueError('accepted Rstar normalization unavailable')
            ref=model('Rstar')
            result.update(calibration_path=root/r['files']['calibration']['path'],
                scores=lambda check:C.score_grid(result['received'],ref,tensor('controls'),tensor('rho'),tensor('keys'),tensor('directions'),tensor('basis'),norm['value'],bindings=r['score_bindings'],store=self.store,check=check))
            if self.selection is not None:result['selection_path']=self.selection
        return result

def execute(request,job,admission,out,check,*,phase,selection=None,store=None):
    if selection is not None:
        r=M.bound_record(Path(request.root)/'CONTRACT_INPUTS.json',request.sha256)
        if r.get('selection_sha256')!=sha(selection):raise ValueError('frozen pre-opening selection differs')
    return C.worker(request.root,request.sha256,out,check,
        runtime_check=lambda:(A.configure_and_check(admission),sha(admission.prerequisites.material_paths['runtime']))[1],
        load=Loader(admission,out,check,selection=selection,store=store),acceptance=request.acceptance,
        calibration_acceptance=request.calibration,admission=admission,phase=phase,
        defer_evaluation=job['arm']=='overwrite' and phase=='qualification')

def readback(out,job,phase,admission,check):
    out=Path(out);r=read(out/'RESULT.json')
    from .cpu_scoring import source_root
    producer=source_root(out)
    if r.get('complete') is not True or r.get('arm')!=job['arm'] or r.get('domain')!='final':raise ValueError('selected contract incomplete')
    if job['arm']=='overwrite':C.overwrite_readback(out/'mark',check)
    elif job['arm']=='noise':
        observed=C.noise_readback(producer/'noise',check,expected_calls=512//16+4*120,expected_controls=16)
        if r['result'].get('noise')!=observed:raise ValueError('noise result/readback mismatch')
        from . import readback as R
        R.evaluation(out/'evaluation',E.specification(admission,'final'),check)
        measured=read(out/'evaluation'/'RESULT.json')
        if r['result']!=dict(measured,noise=observed,white_box_only=True):
            raise ValueError('noise analysis payload differs from independently read evaluation')
    elif job['arm']=='API':
        observed=C.api_readback(out/'service')
        if (observed['attempts']!=job['attempts'] or len(observed['streams'])!=1
                or observed['streams'][0]['owner']!=job['owner'] or observed['streams'][0]['phase']!=phase
                or observed['streams'][0]['policy']!=job['policy'] or r['result'].get('readback')!=observed
                or r['result'].get('attempts')!=job['attempts'] or len(r['result'].get('rows',[]))!=job['attempts']
                or hashlib.sha256(json.dumps(r['result']['rows'],sort_keys=True).encode()).hexdigest()!=observed['response_sha256']
                or any(r['result'].get(k) is not None for k in ('J','K','T','B2','B3','removal'))):
            raise ValueError('API population/binding/unavailability mismatch')
    else:
        s=read(out/('SELECTION.json' if phase=='qualification' else 'ACCUSATION.json'))
        if phase=='qualification' and C.C.accuser_selection(s['scores'])!=s['selected_index']:raise ValueError('selection drift')
        if phase=='final' and s.get('final_readouts')!=1:raise ValueError('adaptive final population')
    return r
