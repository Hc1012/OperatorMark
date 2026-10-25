"""Specified deterministic attack waves, using existing admission and journals."""
from dataclasses import dataclass
from pathlib import Path
import io
from operatormark.corpus.state_v2_prerequisites import require_accepted
from msdev.records import read,write,sha
from . import attacks as T,materials as A,backend as B,models,variants as V,marking as M

@dataclass(frozen=True)
class Request:
    admission: object
    variant: str
    parent: object
    training: object

def admit(request):
    a=B.admit(request.admission.prerequisites);A.preflight(a);v=request.variant
    if v not in V.selected(a.configuration) or not V.definition(v)['updates']:raise ValueError('additional training variant absent')
    parent=request.parent;p=a.prerequisites
    if parent.kind not in ('O','M3') or sha(parent.path)!=M.identity(parent.sha256):raise ValueError('additional parent mismatch')
    r=require_accepted(parent.acceptance,kind='main_study_final_model',bindings=dict(A.scope(a),model_sha256=parent.sha256,model_kind=parent.kind))
    if r.get('domain')!='final':raise ValueError('development parent is not final evidence')
    if parent.kind=='O':
        if parent.sha256!=p.clean_model_sha256:raise ValueError('clean owner substitution')
        route=None
    else:route=T._qualified_route(parent,a,v)
    t=require_accepted(request.training,kind='main_study_final_attack_inputs',bindings=dict(A.scope(a),
        variant=v,parent_model_sha256=parent.sha256,assets_sha256=p.assets.sha256))
    if (t.get('domain')!='final' or t.get('definition')!=V.definition(v) or t.get('plan')!=V.schedule(v)):
        raise ValueError('explicit additional schedule/definition binding required')
    pair=A.json_asset(a,'ordinary_pairing','attack')
    if t.get('shuffle')!=pair.get('shuffle') or type(t.get('initialization')) is not int or t['initialization']<0:
        raise ValueError('existing ordinary shuffle/explicit initialization required')
    if v in V.HEALING:
        if t.get('data_group')!='ordinary' or t.get('data_sha256')!=pair.get('data_sha256'):
            raise ValueError('healing must use the same ordinary solver-label data')
    else:
        entries=A.catalog(a)['entries']
        if (t.get('query_controls_sha256')!=entries['ordinary_x']['sha256']
                or t.get('label_source')!='received_model' or t.get('query_limit')!=256):
            raise ValueError('received-model query acquisition binding differs')
        if v=='extract_jvp' and (t.get('direction_sha256')!=entries.get('ordinary_v',{}).get('sha256')
                or 'ordinary_v' not in entries or t.get('jvp_query_limit')!=256):
            raise ValueError('independently admitted ordinary feasible directions missing')
    return a,t,V.schedule(v),route

def query_binding(request,jvp):
    a,t,_,_=admit(request);entries=A.catalog(a)['entries']
    return dict(scope=A.scope(a),model_sha256=request.parent.sha256,runtime_sha256=a.prerequisites.runtime.sha256,
        assets_sha256=a.prerequisites.assets.sha256,controls_sha256=entries['ordinary_x']['sha256'],
        directions_sha256=entries['ordinary_v']['sha256'] if jvp else None,
        forward_queries=256,jvp_queries=256 if jvp else 0,source_sha256=sha(__file__))

def available_inputs(request):
    """Byte/hash custody check before core training; no numerical imports.

    These inputs already exist after qualification. Future core-produced mix
    endpoints/cache products are deliberately not demanded at this boundary.
    """
    names=['ordinary_x']
    if request.variant in V.HEALING:names.append('ordinary_y')
    if request.variant=='extract_jvp':names.append('ordinary_v')
    for name in names:A.read_bytes(request.admission,name,'attack')

def query_batches(teacher,x,v,check,intent,observation):
    """Shared numerical query kernel; callbacks persist intent and raw output."""
    import torch
    if x.shape!=(256,64) or (v is not None and v.shape!=(256,64)):raise ValueError('query population differs')
    ys=[];dys=[]
    with torch.no_grad():
        for lo in range(0,256,16):
            check();intent('forward',lo,16);y=teacher(x[lo:lo+16])
            if not torch.isfinite(y).all():raise FloatingPointError('nonfinite teacher labels')
            observation('forward',lo,y);ys.append(y.detach())
            if v is not None:
                check();intent('jvp',lo,16);d=torch.func.jvp(teacher,(x[lo:lo+16],),(v[lo:lo+16],))[1]
                if not torch.isfinite(d).all():raise FloatingPointError('nonfinite teacher JVP labels')
                observation('jvp',lo,d);dys.append(d.detach())
    return torch.cat(ys),torch.cat(dys) if v is not None else None

def build_engine(data,initialization,shuffle,variant,*,weights=None,device='cuda'):
    """Shared production/development engine construction; no admission bypass."""
    from msdev.kernels import Engine
    spec=V.definition(variant)
    if variant not in V.STUDENTS+V.HEALING:raise ValueError('numerical training variant required')
    if (variant in V.HEALING)!=(weights is not None):raise ValueError('healing weights or fresh-student initialization differs')
    engine=Engine(data,initialization,shuffle,device,recipe='D9' if variant=='extract_jvp' else 'forward32',
        model_factory=lambda:models.create(spec['architecture']),weights=weights)
    if variant in V.HEALING:
        masks=models.pruning_masks(engine.model,spec['pruning_percent']);models.mask_weights(engine.model,masks)
        for name,p in engine.model.named_parameters():
            if name in masks:p.register_hook(lambda g,m=masks[name]:g*m)
        engine.fixed_masks=masks
    return engine

def acquire(request,out,check,*,jvp):
    """One shared, identity-bound label cache for all student's architectures."""
    a,_,_,_=admit(request);binding=query_binding(request,jvp);check();A.configure_and_check(a)
    import numpy as np
    import torch
    from q5_model import Q5FNO
    root=Path(out);root.mkdir(exist_ok=False);counts=dict(forward=0,jvp=0)
    write(root/'BINDINGS.json',binding)
    try:
        x=A.array(a,'ordinary_x','attack',(256,64),'float32')
        v=A.array(a,'ordinary_v','attack',(256,64),'float32') if jvp else None
        teacher=Q5FNO().to('cuda').eval();teacher.load_state_dict(T._parent_weights(request.parent),strict=True)
        def intent(kind,lo,n):
            counts[kind]+=n;write(root/f'{kind}_intent_{lo:03d}.json',dict(counts=counts))
        def observation(kind,lo,value):
            with (root/f'{kind}_{lo:03d}.npy').open('xb') as f:np.save(f,value.cpu().numpy(),allow_pickle=False)
        y,dy=query_batches(teacher,torch.from_numpy(x).to('cuda'),None if v is None else torch.from_numpy(v).to('cuda'),
                          check,intent,observation)
        values={'x':x,'y':y.cpu().numpy()}
        if jvp:values.update(v=v,dy=dy.cpu().numpy())
        files={}
        for name,value in values.items():
            with (root/(name+'.npy')).open('xb') as f:np.save(f,value,allow_pickle=False)
            files[name+'.npy']=sha(root/(name+'.npy'))
        r=dict(complete=True,binding=binding,counts=counts,files=files,independent_acceptance=False)
        write(root/'RESULT.json',r);return r
    except BaseException as exc:
        write(root/'FAILURE.json',dict(error=repr(exc),counts=counts,complete=False));raise

def query_readback(root,expected,check):
    r=read(Path(root)/'RESULT.json')
    names={'x.npy','y.npy'}|({'v.npy','dy.npy'} if expected['jvp_queries'] else set())
    if (r.get('complete') is not True or r.get('binding')!=expected or set(r.get('files',{}))!=names
            or r.get('counts')!=dict(forward=256,jvp=expected['jvp_queries'])):
        raise ValueError('query cache producer/scope/population differs')
    for name,h in r['files'].items():
        check()
        if sha(Path(root)/name,check)!=h:raise ValueError('query cache bytes drift')
    return r

def execute(request,out,check,*,cache=None):
    a,t,nodes,route=admit(request);v=request.variant;spec=V.definition(v);check();A.configure_and_check(a)
    import numpy as np
    import torch
    from msdev.kernels import Engine
    loaded={}
    def build(node):
        if not loaded:
            weights=None
            if v in V.STUDENTS:
                if cache is None:raise ValueError('shared query cache missing')
                record=read(Path(cache)/'RESULT.json');jvp=record['binding']['jvp_queries']==256
                query_readback(cache,query_binding(request,jvp),check)
                def tensor(name,width):
                    arr=np.load(io.BytesIO((Path(cache)/(name+'.npy')).read_bytes()),allow_pickle=False)
                    if arr.shape!=(256,width) or arr.dtype!=np.float32 or not np.isfinite(arr).all():raise ValueError('teacher cache tensor shape/dtype')
                    return torch.from_numpy(arr).to('cuda')
                absent=T.ForwardOnlyAbsent()
                data=(tensor('x',64),tensor('v',64) if v=='extract_jvp' else absent,
                    tensor('y',4096),tensor('dy',4096) if v=='extract_jvp' else absent)
            else:data=T._training_data(a,t);weights=T._parent_weights(request.parent)
            loaded.update(data=data,weights=weights)
        return build_engine(loaded['data'],t['initialization'],t['shuffle'],v,weights=loaded['weights'])
    return T._execute_nodes(nodes,out,check,build,lambda:admit(request),
        dict(scope=A.scope(a),parent=request.parent.sha256,training=request.training.sha256,
             definition=spec,query_receipt_sha256=None if cache is None else sha(Path(cache)/'RESULT.json')),
        v,route)
