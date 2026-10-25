"""Specified core adapters; no scientific choices, launch, retries or acceptance.

All numerical imports are lazy. The caller owns supervision. Arbitrary attacks
are not dispatched through a permissive plugin/name fallback.
"""
from dataclasses import dataclass
from pathlib import Path
import time
import hashlib
import json

from operatormark.corpus.state_v2_prerequisites import AcceptedRecord, require_accepted
from msdev.records import write
from . import marking as M, materials as A, routing as R
from .update_journal import UpdateJournal, normal_bound

IMPLEMENTED = ('ordinary','replacement','A4','A5B2','A5B3','sham','jacobi16','jacobi256','F64_substitution')


def plan(variant, kappa, *, replacement_length):
    """Exact main/cooldown DAG; branch states never feed the main trajectory."""
    if kappa not in (1.,.5,.25) or isinstance(kappa,bool):raise ValueError('explicit approved kappa required')
    if replacement_length != 10240:raise ValueError('bound development R* selection required; no reselection')
    if variant not in IMPLEMENTED:raise ValueError('attack definition not implemented: '+str(variant))
    if variant in ('jacobi16','jacobi256','F64_substitution'):return []
    if variant=='sham':return [dict(id='sham',parent=None,start=0,end=320,cooldown=False,evaluate=True,recipe='sham')]
    recipe='forward32' if variant in ('ordinary','replacement') else variant
    nodes=[]
    def add(name,parent,start,end,cooldown=False,evaluate=False):
        nodes.append(dict(id=name,parent=parent,start=start,end=end,cooldown=cooldown,evaluate=evaluate,recipe=recipe))
    if variant=='ordinary':
        last=None;pos=0
        for n in (320,960,1920):add('raw'+str(n),last,pos,n,evaluate=True);last='raw'+str(n);pos=n
        return nodes
    length=10240 if variant=='replacement' else int(81920*kappa)
    branches=(length//2,) if variant=='replacement' else (length//16,length//4)
    last=None;pos=0
    if variant=='replacement':
        add('raw1920',None,0,1920,evaluate=True);last='raw1920';pos=1920
    for n in branches+(length,):
        warm=n*9//10;main='main'+str(warm)
        add(main,last,pos,warm);add('cooled'+str(n),main,warm,n,True,True)
        last=main;pos=warm
    return nodes


def updates(nodes):return sum(n['end']-n['start'] for n in nodes)


@dataclass(frozen=True)
class Parent:
    path: Path
    sha256: str
    kind: str                      # O, M3, Aord_O or Aord_M3; never development
    acceptance: AcceptedRecord
    qualification: AcceptedRecord | None = None
    qualification_path: Path | None = None
    intact_model_sha256: str | None = None


@dataclass(frozen=True)
class Request:
    admission: object
    variant: str
    parent: Parent
    training: AcceptedRecord


def admit(request):
    from .backend import admit as readmit
    a=readmit(request.admission.prerequisites);A.preflight(a)
    p=a.prerequisites;parent=request.parent;v=request.variant
    if v not in IMPLEMENTED or v not in a.configuration['attack_variants']:
        raise ValueError('variant unspecified/unapproved')
    if parent.kind not in ('O','M3','Aord_O','Aord_M3') or M.digest(parent.path)!=M.identity(parent.sha256):
        raise ValueError('attack parent drift/type')
    pb=dict(A.scope(a),model_sha256=parent.sha256,model_kind=parent.kind)
    if parent.kind.startswith('Aord_'):
        if v not in ('jacobi16','jacobi256'):raise ValueError('ordinary descendants only feed fixed wrappers')
        M.identity(parent.intact_model_sha256);pb['intact_model_sha256']=parent.intact_model_sha256
    model_record=require_accepted(parent.acceptance,kind='main_study_final_model',bindings=pb)
    if model_record.get('domain')!='final':
        raise ValueError('development parent acceptance refused')
    route=None
    if parent.kind in ('O','Aord_O'):
        if (parent.sha256 if parent.kind=='O' else parent.intact_model_sha256)!=p.clean_model_sha256:
            raise ValueError('original clean-owner identity differs')
        if v=='A5B3':raise ValueError('A5-B3 needs an actually qualified marked parent')
    else:
        if v in ('replacement','sham','A5B2','F64_substitution'):raise ValueError('invalid marked-parent variant')
        if parent.kind=='M3' and v=='ordinary':
            # P2 is a quality obligation, independent of qualification. The
            # accepted model must still be a completed final selected mark.
            completed_mark(model_record)
            route=R.quality_route(p.study_id,p.owner_id,parent.sha256,p.clean_model_sha256)
        else:
            route=_qualified_route(parent,a,v)
    training=require_accepted(request.training,kind='main_study_final_attack_inputs',
        bindings=dict(A.scope(a),variant=v,parent_model_sha256=parent.sha256,
                      assets_sha256=p.assets.sha256))
    if training.get('domain')!='final':raise ValueError('development training inputs refused')
    if a.configuration.get('replacement_length')!=10240:
        raise ValueError('final R* length not explicitly approved/bound')
    M.identity(a.configuration.get('replacement_selection_sha256'))
    nodes=plan(v,a.configuration['kappa'],replacement_length=10240)
    if training.get('plan')!=nodes:raise ValueError('accepted attack schedule differs')
    group='ordinary' if v in ('ordinary','replacement') else 'embedding' if v=='sham' else 'shared16384'
    if v not in ('jacobi16','jacobi256','F64_substitution'):
        if training.get('data_group')!=group:raise ValueError('attack data population differs')
        for k in ('initialization','shuffle'):
            if type(training.get(k)) is not int or training[k]<0:raise ValueError('accepted saved stream missing')
        if v in ('ordinary','replacement'):
            pair=A.json_asset(a,'ordinary_pairing','attack')
            if (training['shuffle']!=pair.get('shuffle') or pair.get('owner_id')!=p.owner_id
                    or pair.get('data_sha256')!=training.get('data_sha256')):
                raise ValueError('ordinary/R* data or initial shuffle differs')
            if v=='replacement' and training['initialization']!=pair.get('replacement_initialization'):
                raise ValueError('replacement initialization differs')
    if v=='A5B2' and (not M.finite(training.get('tau5')) or training['tau5']<=0
                       or training['tau5']!=a.calibration.get('tau5')):
        raise ValueError('independently calibrated positive tau5 required')
    return a,training,nodes,route


def completed_mark(record):
    if (record.get('domain')!='final' or record.get('complete') is not True
            or record.get('recipe')!=M.RECIPE or type(record.get('updates')) is not int
            or record['updates']!=320):
        raise ValueError('independently accepted completed final marking checkpoint required')


def _qualified_route(parent,a,v):
    p=a.prerequisites
    if parent.qualification is None or parent.qualification_path is None:
        raise ValueError('independently accepted owner-specific qualification required')
    qsha=M.digest(parent.qualification_path)
    intact=parent.sha256 if parent.kind=='M3' else parent.intact_model_sha256
    if require_accepted(parent.qualification,kind='main_study_final_qualification',
        bindings=dict(A.scope(a),model_sha256=intact,qualification_sha256=qsha)).get('domain')!='final':
        raise ValueError('development qualification acceptance refused')
    q=M.bound_record(parent.qualification_path,qsha)
    route=R.owner_route(p.study_id,p.owner_id,intact,p.clean_model_sha256,
                       parent.qualification_path,qsha,q['bindings'],[v])
    expected=dict(inputs=p.inputs.sha256,clean_evaluation=p.clean_evaluation.sha256,calibration=p.calibration.sha256)
    if any(q['bindings'].get(k)!=h for k,h in expected.items()) or not route['B3_qualified']:
        raise ValueError('unqualified or mismatched final marked parent')
    route=route['descendants'][0]
    return route


def _parent_weights(parent):
    import io,hashlib,torch
    b=Path(parent.path).read_bytes()
    if hashlib.sha256(b).hexdigest()!=parent.sha256:raise ValueError('parent changed before tensor load')
    value=torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
    if not isinstance(value,dict) or not value or any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in value.values()):
        raise ValueError('finite tensor-only parent required')
    return value


class ForwardOnlyAbsent:
    """Engine slices all four tuple members; forward32 deliberately ignores two.

    This represents absence, not zero-valued derivative labels or extra work.
    Any kernel that tries to operate on the resulting None fails closed.
    """
    def __getitem__(self,indices):return None


def _training_data(a,t):
    import torch
    group=t['data_group'];n={'ordinary':256,'embedding':128,'shared16384':16384}[group]
    purpose='marking' if group=='embedding' else 'attack'
    names=('x','y') if group=='ordinary' else ('x','v','y','dy')
    entries=A.catalog(a)['entries']
    actual=hashlib.sha256(json.dumps({k:entries[group+'_'+k]['sha256'] for k in names},sort_keys=True).encode()).hexdigest()
    if actual!=t['data_sha256']:raise ValueError('attack data content group differs')
    loaded={k:torch.from_numpy(A.array(a,group+'_'+k,purpose,(n,64 if k in ('x','v') else 4096),'float32')).to('cuda') for k in names}
    # forward32 never accesses v/dy. None is explicit absence, not forged labels.
    return loaded['x'],loaded.get('v',ForwardOnlyAbsent()),loaded['y'],loaded.get('dy',ForwardOnlyAbsent())


def execute_training(request,out,check,*,on_checkpoint=None):
    """Checkpointed actual frozen kernels; caller supplies bounded supervision."""
    a,t,nodes,route=admit(request)
    if not nodes:raise ValueError('non-training adapter has a separate entry point')
    check();A.configure_and_check(a)
    import torch
    import q5_model as Q
    from msdev import kernels as K
    loaded={}
    def build(node):
        if not loaded:
            data=_training_data(a,t);weights=None if request.variant=='replacement' else _parent_weights(request.parent)
            reference=None;basis=None
            if request.variant=='A5B2':
                reference=Q.Q5FNO().to('cuda').eval();reference.load_state_dict(weights,strict=True)
            if request.variant=='A5B3':
                basis=torch.from_numpy(A.array(a,'public_basis','marking',(16,4096),'float64')).to('cuda').float()
            loaded.update(data=data,weights=weights,reference=reference,basis=basis)
        engine=K.Engine(loaded['data'],t['initialization'],t['shuffle'],'cuda',recipe=node['recipe'],
                        weights=loaded['weights'],reference=loaded['reference'],tau=t.get('tau5'),basis=loaded['basis'])
        if request.variant=='sham' and tuple(engine.scales)!=tuple(a.inputs['normalizers']):
            raise ValueError('sham normalization differs from the embedding population')
        return engine
    return _execute_nodes(nodes,out,check,build,lambda:admit(request),
        dict(scope=A.scope(a),parent=request.parent.sha256,training=request.training.sha256),request.variant,route,
        on_checkpoint=on_checkpoint)


def _execute_nodes(nodes,out,check,build,revalidate,bindings,variant,route=None,on_checkpoint=None):
    """Shared branch/checkpoint mechanism, not a public authority boundary."""
    normal_bound(nodes)  # source-bound storage admission before any numerical build/dispatch
    if on_checkpoint is not None and hasattr(on_checkpoint,'preflight'):on_checkpoint.preflight()
    import torch,io
    import q5_model as Q
    out=Path(out);out.mkdir(exist_ok=False);started=time.monotonic();executed=0;attempted=0;states={};products={};engine=None
    write(out/'BINDINGS.json',dict(bindings,plan=nodes,independent_acceptance=False))
    journal=None
    try:
        journal=UpdateJournal(out/'updates.jsonl',normal_nodes=nodes)
        for node in nodes:
            check();revalidate();engine=build(node)
            if node['parent'] is not None:
                state_path,expected=states[node['parent']]
                payload=state_path.read_bytes()
                if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('branch parent checkpoint drift')
                engine.restore(torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True))
            if engine.updates!=node['start']:raise ValueError('branch start state mismatch')
            width=node['end']-node['start'];base_lr=.0003 if node['recipe']=='sham' else .002
            for i in range(width):
                check();lr=base_lr*(max(0,(width-i-1)/width) if node['cooldown'] else 1.)
                journal.begin(node['id'],engine.updates,lr)
                attempted+=1
                row=engine.step(lr);executed+=1
                journal.finish(row)
                if (i+1)%2048==0:
                    periodic=A.save_tensor_record(out/f"{node['id']}_{engine.updates}.state.pt",engine.state())
                    write(periodic.with_suffix('.anchor.json'),dict(state_sha256=M.digest(periodic),journal=journal.snapshot()))
            state=A.save_tensor_record(out/(node['id']+'.state.pt'),engine.state());states[node['id']]=(state,M.digest(state))
            if node['evaluate']:
                model=A.save_tensor_record(out/(node['id']+'.model.pt'),Q.tensor_state(engine.model))
                products[node['id']]=dict(path=str(model),sha256=M.digest(model),state_sha256=M.digest(state),updates=node['end'])
                if on_checkpoint is not None:
                    check();products[node['id']]['monitor']=on_checkpoint(engine,node,out)
            write(out/(node['id']+'.json'),dict(node=node,state_sha256=M.digest(state),executed_updates=executed,
                                             journal=journal.snapshot()))
        if executed!=updates(nodes):raise ValueError('attack update reconciliation differs')
        if journal.attempts!=attempted or journal.completed!=executed or journal.pending is not None:
            raise ValueError('durable update journal reconciliation differs')
        result=dict(complete=True,variant=variant,executed_updates=executed,attempted_updates=attempted,products=products,
                    seconds=time.monotonic()-started,route=route,independent_acceptance=False,journal=journal.snapshot())
        write(out/'RESULT.json',result);return result
    except BaseException as exc:
        preservation_error=None
        if engine is not None:
            try:A.save_tensor_record(out/'FAILED_STATE.pt',engine.state())
            except BaseException as save_error:preservation_error=repr(save_error)
        failure=dict(error=repr(exc),executed_updates=executed,attempted_updates=attempted,complete=False,
                    journal=None if journal is None else journal.snapshot(),checkpoint_error=preservation_error,
                    seconds=time.monotonic()-started)
        try:write(out/'FAILURE.json',failure)
        except BaseException as report_error:
            exc.add_note('failure receipt unavailable: '+repr(report_error))
        raise


def wrapper(request,check):
    a,_,_,route=admit(request)
    if request.variant not in ('jacobi16','jacobi256'):raise ValueError('fixed Jacobi variant required')
    check();A.configure_and_check(a)
    import q5_model as Q
    from msdev.kernels import Jacobi
    model=Q.Q5FNO().to('cuda').eval();model.load_state_dict(_parent_weights(request.parent),strict=True)
    return Jacobi(model,16 if request.variant=='jacobi16' else 256),route


def substitution_tasks(request,out,check):
    """Discrete F64 exact-gradient reference, never watermark removal evidence."""
    a,_,_,_=admit(request)
    if request.variant!='F64_substitution':raise ValueError('substitution request required')
    from .evaluation import specification
    specification(a,'final');A.require_opening(a);check();A.configure_and_check(a)
    import numpy as np
    from operatormark_numerics_v2 import task
    from q6_common import true_field,true_field_vjp
    out=Path(out);out.mkdir(exist_ok=False)
    starts=A.array(a,'final_starts','final',(64,8,8),'float64')
    targets=A.array(a,'final_targets','final',(64,64,64),'float64')
    pt=A.array(a,'final_targets_p256','final',(64,64,64),'float64')
    counts=dict(forward=0,adjoint=0,scoring_t128=0,scoring_p256=0,completed_targets=0)
    rows=[]
    try:
        for i in range(64):
            calls=0
            def oracle(rho):
                nonlocal calls
                check();counts['forward']+=1;calls+=1
                # Frozen optimize requests one final field without consuming a
                # VJP. Use a closed forward session for that last call.
                if calls==121:
                    def unused(_):raise RuntimeError('unexpected final-field derivative')
                    return true_field(rho),unused
                if calls>121:raise ValueError('frozen task call population changed')
                f,vjp=true_field_vjp(rho)
                def derivative(w):counts['adjoint']+=1;return vjp(w)
                return f,derivative
            result=task.optimize(oracle,starts[i:i+1],targets[i:i+1],batch_size=16)
            design=result.designs[0];check();counts['scoring_t128']+=1;f=true_field(design)
            check();counts['scoring_p256']+=1;p=true_field(design,True)
            with (out/f'design_{i:02d}.npy').open('xb') as stream:np.save(stream,design,allow_pickle=False)
            row=dict(target=i,T=task.score(f,targets[i]),P256=task.score(p,pt[i]),trace=result.loss_trace,
                     final_loss=result.final_loss,batch_loss_divisor=16)
            rows.append(row);counts['completed_targets']+=1;write(out/f'task_{i:02d}.json',row)
        result=dict(complete=True,counts=counts,rows=rows,physical_qualification=False,ownership_removal_claim=False)
        write(out/'RESULT.json',result);return result
    except BaseException as exc:
        write(out/'FAILURE.json',dict(error=repr(exc),counts=counts,complete=False));raise
