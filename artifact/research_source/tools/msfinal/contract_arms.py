"""Selected attack contracts: bound admission, kernels, calibration and readback.

No issuer, final defaults, opening, launcher or numerical imports on import.
Development requests are byte-bound fixtures, NEVER final AcceptedRecords.
"""
from pathlib import Path
import hashlib,json,math,time
from msdev.records import read,write,sha,progress
from . import prospective_contracts as C,marking as M

NOISE_DRAW_RULE=dict(schema='scientific_D5.retained-forward.v1',
    address='(bound model, phase, stream, SD, retained forward ordinal, repetition 0..31, tensor row/component)',
    order='validation batches in order, then inverse batches and updates in order',
    consumption='32 consecutive full-shape draws per retained forward; none for derivative-only validation or detector/B2 projections',
    generator='unchanged phase-persistent torch.Generator; no backward redraw or reset',
    publication='D5.exclusive-hardlink.v1',historical_stream_equivalent=False)
NOISE_CLAIM='additive white-box pathwise only; same controls, thresholds and intact-parent prerequisites; no black-box claim'
CONTRACTS=dict(overwrite=C.OVERWRITE,noise=dict(C.NOISE,draw_rule=NOISE_DRAW_RULE),API=C.API,adaptive_accuser=C.ACCUSER)

def admit(root,expected,*,acceptance=None,calibration_acceptance=None,admission=None,phase='development'):
    root=Path(root);p=root/'CONTRACT_INPUTS.json'
    value=M.bound_record(p,expected)
    if value.get('arm') not in CONTRACTS or value.get('contract')!=CONTRACTS[value['arm']]:raise ValueError('selected contract differs')
    if phase=='development':
        if value.get('domain')!='development' or acceptance is not None:raise ValueError('development contract domain')
    elif phase in ('qualification','final'):
        from operatormark.corpus.state_v2_prerequisites import require_accepted
        from . import materials as A
        if admission is None or acceptance is None:raise ValueError('independent final contract acceptance required')
        bindings=dict(A.scope(admission),request_sha256=expected,arm=value['arm'])
        if require_accepted(acceptance,kind='main_study_contract_inputs',bindings=bindings).get('domain')!='final':raise ValueError('final contract evidence domain')
        if value.get('domain')!='final':raise ValueError('development-to-final substitution')
        if value['arm'] in ('overwrite','adaptive_accuser'):
            h=value['files']['calibration']['sha256']
            accepted=require_accepted(calibration_acceptance,kind='main_study_attacker_calibration',
                bindings=dict(A.scope(admission),arm=value['arm'],calibration_sha256=h))
            if accepted.get('domain')!='final' or accepted.get('defender_calibration_used') is not False:raise ValueError('independent attacker calibration acceptance required')
        if phase=='final':A.require_opening(admission)
        A.configure_and_check(admission)
    else:raise ValueError('contract phase')
    for k in ('runtime_sha256','source_sha256','received_model_sha256'):M.identity(value[k])
    if value['source_sha256']!=sha(__file__):raise ValueError('contract implementation drift')
    if not isinstance(value.get('files'),dict) or not value['files']:raise ValueError('complete contract inputs required')
    for name,entry in value['files'].items():
        relative=Path(entry['path']);p=(root/relative).resolve()
        if relative.is_absolute() or not p.is_relative_to(root.resolve()):raise ValueError('contract path escape')
        data=p.read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('contract input drift: '+name)
    if value['files'].get('received',{}).get('sha256')!=value['received_model_sha256']:raise ValueError('received-model binding')
    roles={'overwrite':{'received','controls','directions','key','shuffle','calibration'},
           'noise':{'received','streams'},'API':{'received'},
           'adaptive_accuser':{'received','Rstar','keys','calibration'}}[value['arm']]
    if not roles<=value['files'].keys():raise ValueError('missing contract inputs')
    if value['arm']=='adaptive_accuser' and value.get('reference_role')!='independently_trained_Rstar':raise ValueError('attacker Rstar reference required')
    return value

def calibration(rows,bindings,*,arm):
    """Twelve bound null observations; no defender floor or final outcome input."""
    if arm not in ('overwrite','adaptive_accuser') or len(rows)!=12:raise ValueError('twelve independently bound calibration rows required')
    for name in ('roster','runtime','source','controls','keys'):M.identity(bindings[name])
    labels=[r['label'] for r in rows]
    if len(set(labels))!=12:raise ValueError('duplicate calibration observations')
    values=[]
    for r in rows:
        for name in ('reference_sha256','candidate_sha256','controls_sha256','key_population_sha256','producer_sha256'):M.identity(r[name])
        if r.get('phase')!='development_calibration':raise ValueError('calibration must precede final readout')
        if arm=='adaptive_accuser':
            index=C.accuser_selection(r['scores']);value=r['scores'][index]
            if r.get('selected_index')!=index:raise ValueError('calibration must apply signed4096 selection procedure')
        else:value=r['score']
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('missing/nonfinite calibration')
        values.append(value)
    floor=max(map(abs,values))
    if floor<=0:raise ValueError('zero calibration scale is unavailable')
    return dict(arm=arm,rows=rows,bindings=bindings,floor=floor,delta=.15*floor,target=3*floor,
        theta=1.15*floor,independent_acceptance=False,phase='development_calibration')

def calibration_readback(path,expected,*,arm,bindings):
    r=M.bound_record(path,expected);actual=calibration(r['rows'],r['bindings'],arm=arm)
    if r!=actual or r['bindings']!=bindings:raise ValueError('calibration readback/identity mismatch')
    return r

def overwrite(received,x,v,rho,key_v,pattern,initialization,shuffle,cal,out,check):
    """320 updates, received-model F/JVP utility and full128 two-pass penalty."""
    import torch
    import q5_model as Q
    from msdev import kernels as K
    pn=M.verified_pn();out=Path(out);out.mkdir(exist_ok=False)
    if len(x)!=128 or len(v)!=128 or len(rho)!=128 or cal.get('arm')!='overwrite' or cal['target']!=3*cal['floor']:raise ValueError('overwrite input/calibration contract')
    reference=Q.Q5FNO().to(x.device).eval();reference.load_state_dict(Q.tensor_state(received),strict=True)
    if list(reference.buffers()):raise ValueError('deterministic buffer-free received reference required')
    directions=K.physical_key_directions(rho,key_v,x.device);ys=[];dys=[];refs=[]
    counts=dict(updates=0,received_forward=0,received_utility_JVP=0,received_key_JVP=0)
    try:
        with torch.no_grad():
            for lo in range(0,128,16):
                check();counts['received_forward']+=16;counts['received_utility_JVP']+=16;counts['received_key_JVP']+=16
                write(out/f'query_intent_{lo:03d}.json',counts)
                y,j=torch.func.jvp(reference,(x[lo:lo+16],),(v[lo:lo+16],));r=torch.func.jvp(reference,(x[lo:lo+16],),(directions[lo:lo+16],))[1]
                from .materials import save_tensor_record
                save_tensor_record(out/f'labels_{lo:03d}.pt',dict(y=y.cpu(),dy=j.cpu(),key_jvp=r.cpu()))
                ys.append(y.detach());dys.append(j.detach());refs.append(r.detach())
        data=(x,v,torch.cat(ys),torch.cat(dys));engine=K.Engine(data,initialization,shuffle,str(x.device),recipe='B3',weights=Q.tensor_state(received))
        for i in range(320):
            check();counts['updates']+=1;write(out/f'intent_{i+1:03d}.json',counts)
            row=pn.pn_step(engine,directions,pattern,torch.cat(refs),cal['target'],check)
            write(out/f'update_{i+1:03d}.json',row)
            if engine.updates!=i+1:raise ValueError('overwrite update drift')
        from .materials import save_tensor_record
        model=save_tensor_record(out/'MODEL.pt',Q.tensor_state(engine.model));state=save_tensor_record(out/'STATE.pt',engine.state())
        result=dict(complete=True,counts=counts,model_sha256=sha(model),state_sha256=sha(state),
            label_source='received-model self-distillation',normalizers=list(engine.scales),target=cal['target'],defender_calibration_used=False)
        write(out/'RESULT.json',result);return engine.model,result
    except BaseException as exc:
        write(out/'FAILURE.json',dict(complete=False,error=repr(exc),counts=counts));raise

def overwrite_readback(root,check):
    root=Path(root);r=read(root/'RESULT.json')
    if not r['complete'] or r['counts']!=dict(updates=320,received_forward=128,received_utility_JVP=128,received_key_JVP=128):raise ValueError('overwrite count mismatch')
    from .readback import tensor_file
    for name in ('MODEL','STATE'):tensor_file(root/(name+'.pt'),r[name.lower()+'_sha256'],check)
    for i in range(1,321):
        check();row=read(root/f'update_{i:03d}.json')
        if row['pre_state']!=i-1 or row['controls']!=128:raise ValueError('overwrite raw update mismatch')
    return r

def score_grid(candidate,reference,x,rho,coeff,v,basis,scale,*,bindings,store,check):
    """Both real model projections cached under full, independently bound inputs."""
    from msdev import scoring as S
    import torch
    if tuple(rho.shape)==(len(x),8,8):rho=rho.reshape(len(x),64)
    if tuple(rho.shape)!=(len(x),64) or tuple(x.shape)!=(len(x),64) or not torch.isfinite(rho).all():
        raise ValueError('physical scoring controls must be finite [N,64] or [N,8,8]')
    if not torch.equal(rho.to(device=x.device,dtype=x.dtype),x):raise ValueError('physical/model scoring controls differ')
    for name in ('candidate','reference','runtime','controls','basis','scoring','procedure'):M.identity(bindings[name])
    def projection(model,role):
        key=hashlib.sha256(json.dumps(dict(model=bindings[role],**{k:bindings[k] for k in ('runtime','controls','basis','scoring','procedure')}),sort_keys=True).encode()).hexdigest()
        return store.obtain(key,lambda:S.projections(model,x,basis,True,check=check))
    return S.projected_scores(projection(candidate,'candidate'),projection(reference,'reference'),rho,coeff,v,basis,scale,'B3')

def adaptive_select(scores,cal,bindings,out):
    for name in ('reference','keys','controls','source','runtime','calibration'):M.identity(bindings[name])
    if cal.get('arm')!='adaptive_accuser':raise ValueError('attacker procedure calibration required')
    index=C.accuser_selection(scores)
    result=dict(scores=scores,selected_index=index,selected_score=scores[index],theta=cal['theta'],delta=cal['delta'],
        bindings=bindings,reference_role='independently_trained_Rstar',sign_flipped=False,final_readouts=0)
    write(out,result);return result

def adaptive_final(selection_path,selection_sha256,scores,out,*,opening_sha256):
    """One actual untouched-final grid; no reselection, calibration or sign flip."""
    selection=M.bound_record(selection_path,selection_sha256);M.identity(opening_sha256)
    if C.accuser_selection(selection['scores'])!=selection['selected_index'] or selection['sign_flipped']:raise ValueError('frozen signed selection mismatch')
    if len(scores)!=4096 or any(not M.finite(x) for x in scores):raise ValueError('complete final score population required')
    index=selection['selected_index'];wrong=max(abs(v) for i,v in enumerate(scores) if i!=index)
    result=dict(score=scores[index],wrong_max=wrong,theta=selection['theta'],delta=selection['delta'],
        detected=scores[index]>selection['theta'] and scores[index]>wrong+selection['delta'],
        selected_index=index,selection_sha256=selection_sha256,opening_sha256=opening_sha256,final_readouts=1)
    # Exclusive write: the same final invocation cannot be silently repeated.
    write(out,result);return result

class APIEvaluation:
    """Registry persists one client per model/owner/phase; no per-task reset."""
    def __init__(self,out):self.root=Path(out);self.root.mkdir(exist_ok=False);self.clients={}
    def forward(self,model,owner,phase,policy,controls,compute,check,*,now=time.monotonic):
        M.identity(model)
        if phase not in ('development','qualification','final'):raise ValueError('declared API phase required')
        key=(model,owner,phase,policy)
        if key not in self.clients:self.clients[key]=C.Service(policy)
        client=self.clients[key];rows=[]
        for control in controls:
            check();attempt=client.attempts+1;prefix=hashlib.sha256(repr(key).encode()).hexdigest()+'_'+str(attempt)
            def record(r):write(self.root/(prefix+'_'+r['kind']+'.json'),dict(r,model=model,owner=owner,phase=phase,control=list(control)))
            response=client.call(control,now(),compute,record);write(self.root/(prefix+'_response.json'),response);rows.append(response)
        return dict(rows=rows,attempts=client.attempts,J=None,K=None,T=None,B2=None,B3=None,
            removal=None,unavailable_reason='no adopted service-visible derivative/task/detector procedure')

def api_readback(root):
    root=Path(root);intents=list(root.glob('*_intent.json'));rows=[];groups={};members=set();statuses={}
    intents.sort(key=lambda p:(p.name.rsplit('_',2)[0],read(p)['attempt']))
    for p in intents:
        prefix=p.name[:-len('_intent.json')];intent=read(p);r=read(root/(prefix+'_result.json'));v=read(root/(prefix+'_response.json'))
        members.update(prefix+'_'+suffix+'.json' for suffix in ('intent','result','response'))
        key=tuple(intent.get(k) for k in ('model','owner','phase','policy'))
        if (key[3] not in C.API['policies'] or not all(isinstance(v,str) for v in key)
                or prefix!=hashlib.sha256(repr(key[:3]+(key[3],)).encode()).hexdigest()+'_'+str(intent.get('attempt'))):
            raise ValueError('API stream identity/filename differs')
        if (intent.get('kind')!='intent' or r.get('kind')!='result' or type(intent.get('attempt')) is not int
                or intent['attempt']<1 or not M.finite(intent.get('time'))
                or any(r.get(k)!=intent.get(k) for k in ('attempt','time','model','owner','phase','control'))):
            raise ValueError('API ordered intent/result binding differs')
        groups.setdefault(key,[]).append((intent['attempt'],intent['time']))
        if r['status']!=v['status'] or v['derivative_available'] is not False:raise ValueError('API response readback mismatch')
        if r['status'].startswith('denied') and v['value'] is not None:raise ValueError('refusal replaced with hidden output')
        if r['status'] not in ('computed','cached','denied_near_previous','denied_rate'):raise ValueError('unknown API outcome')
        statuses[r['status']]=statuses.get(r['status'],0)+1
        rows.append(v)
    streams=[]
    for key,events in sorted(groups.items()):
        events.sort()
        if [a for a,t in events]!=list(range(1,len(events)+1)) or any(b[1]<a[1] for a,b in zip(events,events[1:])):
            raise ValueError('API missing/duplicate attempt or timestamp rollback')
        streams.append(dict(model=key[0],owner=key[1],phase=key[2],policy=key[3],attempts=len(events),
            first_timestamp=events[0][1],last_timestamp=events[-1][1]))
    if {p.name for p in root.iterdir()}!=members:raise ValueError('API evidence membership differs')
    return dict(attempts=len(intents),responses=len(rows),streams=streams,status_counts=statuses,
        response_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        rate_outcomes='measured from retained events, not inferred from schedule',
        derivative_task_available=False,detector_available=False,removal=None)

def noise_model(base,sd,stream_seed,out,*,binding,check):
    """One independent stream per bound phase/model; actual32 output records.

    The same graph and draws yield each loss/gradient pair. JVP is pathwise.
    No hidden redraw in backward, no global RNG or model mode mutation.
    """
    import torch
    for name in ('model','phase','stream'):
        if not binding.get(name):raise ValueError('noise identity required')
    if sd not in C.NOISE['sd'] or type(stream_seed) is not int:raise ValueError('fixed noise contract')
    out=Path(out);out.mkdir(exist_ok=False)
    stream_digest=hashlib.sha256(json.dumps(dict(seed=stream_seed,binding=binding,sd=sd),sort_keys=True).encode()).hexdigest()
    derived_seed=int(stream_digest[:16],16)%(2**63)
    write(out/'STREAM.json',dict(commitment=stream_digest,binding=binding,sd=sd,
        derivation='sha256(canonical seed/model/phase/stream/SD) first64bits modulo2^63',independent_by_model_phase=True,
        draw_rule=NOISE_DRAW_RULE))
    class Noisy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.base=base;self.calls=0;self.generator=torch.Generator(device=next(base.parameters()).device).manual_seed(derived_seed)
            self.train(base.training)
        @property
        def pathwise_projection_model(self):return self.base
        def forward(self,x):
            check();call=self.calls;self.calls+=1
            write(out/f'call_{call:06d}.json',dict(binding=binding,sd=sd,outputs=32,controls=len(x),
                call=call,draw_rule=NOISE_DRAW_RULE,white_box_pathwise=True))
            # Data-dependent Python/serialization is kept outside torch.func
            # transforms: JVP wrapper below supplies the unchanged base JVP.
            from .materials import save_tensor_record
            def draw(i,shape,device,dtype):return torch.randn(shape,generator=self.generator,device=device,dtype=dtype)*sd
            def record(i,z):
                path=save_tensor_record(out/f'{call:06d}_{i:02d}.pt',dict(output=z.cpu()))
                write(out/f'{call:06d}_{i:02d}.json',dict(sha256=sha(path),binding=binding,call=call,repetition=i))
            return C.noise_mean(self.base,x,draw,record)
        def pathwise_jvp(self,x,v):
            y=self(x);j=torch.func.jvp(self.base,(x,),(v,))[1]
            return y,j
        def pathwise_derivative(self,x,v):
            # No noisy primal is requested or discarded; no draw state advances.
            return torch.func.jvp(self.base,(x,),(v,))[1]
    return Noisy()

def noise_readback(root,check,*,expected_calls=None,expected_controls=None):
    root=Path(root);calls=sorted(root.glob('call_*.json'));outputs=0
    from .readback import tensor_file
    stream=read(root/'STREAM.json');members={'STREAM.json'}
    if stream.get('draw_rule')!=NOISE_DRAW_RULE:raise ValueError('noise draw rule absent/drifted; historical source required')
    if expected_calls is not None and len(calls)!=expected_calls:raise ValueError('noise retained call population differs')
    for ordinal,p in enumerate(calls):
        check();r=read(p);index=int(p.stem.split('_')[1])
        members.add(p.name)
        if (index!=ordinal or r.get('call')!=ordinal or r.get('binding')!=stream['binding'] or r.get('sd')!=stream['sd']
                or r.get('draw_rule')!=NOISE_DRAW_RULE or r['outputs']!=32 or not r['white_box_pathwise']
                or (expected_controls is not None and r.get('controls')!=expected_controls)):
            raise ValueError('noise address/binding/repetition/access drift')
        for i in range(32):
            file=root/f'{index:06d}_{i:02d}.pt';receipt=read(root/f'{index:06d}_{i:02d}.json')
            if receipt.get('binding')!=stream['binding'] or receipt.get('call')!=index or receipt.get('repetition')!=i:
                raise ValueError('noise output address differs')
            members.update((file.name,file.name+'.publication.json',file.with_suffix('.json').name))
            tensor_file(file,receipt['sha256'],check);outputs+=1
    if {p.name for p in root.iterdir()}!=members:raise ValueError('noise retained output membership differs')
    return dict(calls=len(calls),retained_outputs=outputs,white_box_only=True,draw_rule=NOISE_DRAW_RULE,claim_scope=NOISE_CLAIM)

def calibrate_numerical(cases,bindings,out,check,*,arm,store):
    """Execute exactly the independently precommitted twelve null cases.

    Each case supplies real models and a full bound input tuple. F64 overwrite
    cases may provide independently byte-checked exact tangent observations;
    those cannot be invented or replaced with zero scores.
    """
    if len(cases)!=12 or len({c['label'] for c in cases})!=12:raise ValueError('calibration roster must be fixed before scoring')
    out=Path(out);out.mkdir(exist_ok=False);rows=[]
    try:
        for i,c in enumerate(cases):
            check();write(out/f'intent_{i:02d}.json',dict(label=c['label'],bindings=c['bindings']))
            if arm=='overwrite':
                import torch
                from msdev.kernels import physical_key_directions,patterns
                x=c['x'];direction=physical_key_directions(c['rho'].detach().cpu().numpy(),c['v'][0].detach().cpu().numpy(),x.device)
                w=patterns(c['coeff'][:1],c['basis'])[0];parts=[]
                b=c['bindings']
                normalizer_key=hashlib.sha256(json.dumps(dict(reference=b['reference'],controls=b['controls'],
                    utility_direction=c['utility_direction_sha256'],runtime=b['runtime'],source=b['scoring'],kind='received_self_distillation_scales'),sort_keys=True).encode()).hexdigest()
                def normalizer():
                    ys=[];js=[]
                    with torch.no_grad():
                        for lo in range(0,len(x),16):
                            check();y,j=torch.func.jvp(c['reference'],(x[lo:lo+16],),(c['utility_v'][lo:lo+16],));ys.append(y.cpu());js.append(j.cpu())
                    from q5_model import scales
                    fs,ss=scales(torch.cat(ys),torch.cat(js))
                    return dict(forward=float(fs),jvp=float(ss))
                scales=store.obtain(normalizer_key,normalizer);scale=math.sqrt(scales['jvp'])
                write(out/f'normalizers_{i:02d}.json',dict(scales=scales,binding=normalizer_key,source='received self-distillation'))
                def cached(role):
                    b=c['bindings'];key=hashlib.sha256(json.dumps(dict(model=b[role],runtime=b['runtime'],controls=b['controls'],
                        keys=c['record']['key_population_sha256'],source=b['scoring'],procedure=b['procedure'],kind='direct_true_JVP'),sort_keys=True).encode()).hexdigest()
                    def compute():
                        chunks=[]
                        with torch.no_grad():
                            for lo in range(0,len(x),16):
                                check();chunks.append(torch.func.jvp(c[role],(x[lo:lo+16],),(direction[lo:lo+16],))[1].detach().cpu())
                        return torch.cat(chunks)
                    return store.obtain(key,compute).to(x.device)
                with torch.no_grad():
                    j=cached('reference');pred=c['truth_jvp'] if c.get('candidate') is None else cached('candidate')
                    parts.append(float(((pred-j)*w).mean())/scale)
                value=dict(score=sum(parts))
            else:
                scores=score_grid(c['candidate'],c['reference'],c['x'],c['rho'],c['coeff'],c['v'],c['basis'],c['scale'],bindings=c['bindings'],store=store,check=check)
                value=dict(scores=scores,selected_index=C.accuser_selection(scores))
            row=dict(c['record'],label=c['label'],phase='development_calibration',**value)
            write(out/f'row_{i:02d}.json',row);rows.append(row)
        result=calibration(rows,bindings,arm=arm);write(out/'CALIBRATION.json',result);return result
    except BaseException as exc:
        write(out/'FAILURE.json',dict(error=repr(exc),completed_rows=len(rows)));raise

def worker(root,request_sha256,out,check,*,runtime_check,load,acceptance=None,calibration_acceptance=None,admission=None,phase='development',defer_evaluation=False):
    """Complete bound dispatch seam; load resolves only the already hashed files.

    Runtime/admission runs before the numerical loader. Calibration outputs must
    be independently supplied/accepted before final execution; this never signs
    its own calibration. The result reports unavailable service evidence honestly.
    """
    request=admit(root,request_sha256,acceptance=acceptance,calibration_acceptance=calibration_acceptance,admission=admission,phase=phase)
    if defer_evaluation and (phase!='qualification' or request['arm']!='overwrite'):
        raise ValueError('only pre-opening overwrite may defer final evaluation')
    if runtime_check()!=request['runtime_sha256']:raise ValueError('contract runtime mismatch before numerical load')
    check();out=Path(out);out.mkdir(exist_ok=False)
    write(out/'BINDINGS.json',dict(request_sha256=request_sha256,phase=phase,domain=request['domain'],contract=request['contract']))
    arm=request['arm']
    try:
        data=load(request,Path(root));check()
        if arm=='overwrite':
            cal=calibration_readback(data['calibration_path'],request['files']['calibration']['sha256'],arm=arm,bindings=request['calibration_bindings'])
            model,result=overwrite(data['received'],data['x'],data['v'],data['rho'],data['key_v'],data['pattern'],data['initialization'],data['shuffle'],cal,out/'mark',check)
            overwrite_readback(out/'mark',check)
            # Owner qualification is never inferred from the attacker's mark.
            evaluated=(dict(available=False,reason='final evaluation deferred until coordinated opening')
                       if defer_evaluation else data['evaluate'](model,out/'evaluation',check))
            result=dict(result,evaluation=evaluated,owner_qualification_inherited=False)
        elif arm=='noise':
            model=noise_model(data['received'],request['sd'],data['seed'],out/'noise',binding=request['noise_binding'],check=check)
            result=data['evaluate'](model,out/'evaluation',check);result=dict(result,noise=noise_readback(out/'noise',check),white_box_only=True)
        elif arm=='API':
            service=APIEvaluation(out/'service')
            result=service.forward(request['received_model_sha256'],request['owner'],phase,request['policy'],data['controls'],data['compute'],check)
            result['readback']=api_readback(out/'service')
        else:
            cal=calibration_readback(data['calibration_path'],request['files']['calibration']['sha256'],arm=arm,bindings=request['calibration_bindings'])
            scores=data['scores'](check)
            if phase=='final':result=adaptive_final(data['selection_path'],request['selection_sha256'],scores,out/'ACCUSATION.json',opening_sha256=admission.prerequisites.opening.sha256)
            else:result=adaptive_select(scores,cal,request['selection_bindings'],out/'SELECTION.json')
        write(out/'RESULT.json',dict(complete=True,arm=arm,result=result,domain=request['domain'],independent_acceptance=False));return result
    except BaseException as exc:
        write(out/'FAILURE.json',dict(complete=False,arm=arm,error=repr(exc)));raise
