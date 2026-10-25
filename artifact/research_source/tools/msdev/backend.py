"""Real development backend for the finite graph. Never selected by tests implicitly."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import statistics
import time
import numpy as np
import torch
import q5_model as M
from . import kernels as K,rules,scoring
from .records import read,write,sha,verify

class Backend:
    def __init__(self,bundle,root,device='cuda'):
        if device!='cuda' or not torch.cuda.is_available():raise RuntimeError('A100 CUDA required; no CPU fallback')
        if 'A100' not in torch.cuda.get_device_name():raise RuntimeError('development budget is for A100 only')
        self.bundle=Path(bundle);self.root=Path(root);self.device=device;self.cache={}
        self.basis=torch.from_numpy(np.load(self.bundle/'public_basis.npy',allow_pickle=False)).to(device);self.factory=M.Q5FNO
        self.input_manifest=read(self.bundle/'MANIFEST.json');self.config=read(self.bundle/'CONFIG.json')
        self.seeds=read(self.bundle/'training_seeds.json');self.selected={}
        for d in ('models','states','evaluations','scores','products'): (self.root/d).mkdir(exist_ok=True)
        M.configure();self.verify_inputs()
    def verify_inputs(self):verify(self.bundle,self.input_manifest['files'])
    def synchronize(self):torch.cuda.synchronize()
    def load(self,name):return np.load(self.bundle/name,allow_pickle=False)
    def data(self,kind,owner=0):
        key=(kind,owner)
        if key not in self.cache:
            if kind=='training':a=[self.load(f'training_{n}.npy') for n in ('x','v','y','dy')]
            elif kind=='embedding':
                with np.load(self.bundle/f'embedding_{owner}.npz',allow_pickle=False) as z:a=[z[n].copy() for n in ('x','v','y','dy')]
            elif kind=='attack':
                x=self.load('attack_x.npy');y=self.load('attack_y.npy');a=[x,np.zeros_like(x),y,np.zeros_like(y)]
            elif kind=='timing16384':
                # Memory fixture, NOT 16,384 independent labels.
                base=self.data('training');self.cache[key]=tuple(t.repeat((4,1)) for t in base);return self.cache[key]
            self.cache[key]=tuple(torch.from_numpy(z).to(self.device) for z in a)
        return self.cache[key]
    def weights(self,name):
        path=self.root/'models'/(name+'.pt')
        if not path.exists():path=self.bundle/(name+'.pt')
        return torch.load(path,map_location='cpu',weights_only=True)
    def model(self,name):
        model=M.Q5FNO().to(self.device);model.load_state_dict(self.weights(name),strict=True);model.eval();return model
    def keys(self,owner,method):
        with np.load(self.bundle/f'keys_{method}_{owner}.npz',allow_pickle=False) as z:
            return tuple(torch.from_numpy(z[n].copy()).to(self.device) for n in ('coeff','v'))
    def key(self,owner,method):
        coeff,v=self.keys(owner,method)
        with np.load(self.bundle/f'keys_{method}_{owner}.npz',allow_pickle=False) as z:w=torch.from_numpy(z['w_true'].copy()).to(self.device)
        return dict(w=w,v=v[0])
    def selection(self):
        return {name:read(self.root/'products'/(name+'.json')) for name in ('mark_selection','replacement_selection') if (self.root/'products'/(name+'.json')).exists()}
    def selected_name(self,owner,source):
        if source=='O':return f'D{owner+1}'
        method='B1' if source=='M1' else 'B3'
        selected=read(self.root/'products/mark_selection.json')['selection'][method]
        return f'{method}_{selected["factor"]:g}_D{owner+1}'
    def metrics(self,name):return read(self.root/'evaluations'/name/'RESULT.json')['metrics']
    def evaluate(self,name,owner,c):
        path=self.root/'evaluations'/name
        if path.exists():
            bound=path/'EVALUATION_BINDING.json'
            if bound.exists():
                receipt=read(bound)
                if receipt['model_sha256']!=sha(self.root/'models'/(name+'.pt')):raise ValueError('evaluation model correspondence changed')
                verify(path,receipt['files']);return read(path/'RESULT.json')
            # Never overwrite a partial evaluation; explicit infrastructure recovery
            # uses a fresh numbered directory and retains the failed one.
            i=1
            while path.with_name(name+f'_partial{i}').exists():i+=1
            path.rename(path.with_name(name+f'_partial{i}'))
        measured=c.state['rates'].get('evaluation')
        if measured is None:
            # Explicit bounded first use; NOT a predicted cost or proof of fit.
            c.budget.reserve(180,0,0);allowance=180
        else:
            c.budget.reserve(measured,measured,0);allowance=1.2*measured
        start=time.time();phase_end=min(start+allowance,c.state['session']['end']-30)
        def check():
            c.check()
            if time.time()>=phase_end:raise TimeoutError('development evaluation measured/first-use envelope')
        from q6_common import true_field
        def oracle(rho,p1):
            try:return true_field(rho,p1=p1)
            except RuntimeError as exc:raise ValueError('native evaluation failure: '+str(exc)) from exc
        result=scoring.evaluate(self.model(name),self.bundle,self.key(owner,'B3')['v'],self.load(f'key_truth_{owner}.npy'),path,check,
                                oracle)
        write(path/'EVALUATION_BINDING.json',dict(model_sha256=sha(self.root/'models'/(name+'.pt')),
              files={p.name:sha(p) for p in path.iterdir() if p.is_file()}))
        c.state['rates']['evaluation']=max(c.state['rates'].get('evaluation',0),result['seconds']);c.save()
        if not result['complete']:raise ValueError('required evaluation has unavailable metrics: '+name)
        return result
    def projection(self,name,c,need_jacobian=True):
        path=self.root/'scores'/(name+('_projection.pt' if need_jacobian else '_forward.pt'));model_path=self.root/'models'/(name+'.pt')
        if not model_path.exists():model_path=self.bundle/(name+'.pt')
        identity=sha(model_path)
        if path.exists() and not path.with_suffix('.json').exists():
            path.rename(path.with_name(path.name+f'.partial_{time.time_ns()}'))
        if path.exists():
            r=read(path.with_suffix('.json'))
            if r['model']!=identity or r['projection']!=sha(path):raise ValueError('projection/model binding drift')
            return torch.load(path,map_location='cpu',weights_only=True)
        rate_name='projection' if need_jacobian else 'forward_projection'
        measured=c.state['rates'].get(rate_name)
        if measured is None:c.budget.reserve(600,0,0);allowance=600
        else:c.budget.reserve(measured,measured,0);allowance=1.2*measured
        start=time.time();end=min(start+allowance,c.state['session']['end']-30)
        def check():
            c.check()
            if time.time()>=end:raise TimeoutError('full Jacobian projection phase cap')
        x=torch.from_numpy(self.load('validation_x.npy')).to(self.device)
        result=scoring.projections(self.model(name),x,self.basis,need_jacobian=need_jacobian,check=check)
        elapsed=time.time()-start
        torch.save(result,path);write(path.with_suffix('.json'),dict(model=identity,projection=sha(path),seconds=elapsed))
        c.state['rates'][rate_name]=max(elapsed,c.state['rates'].get(rate_name,0))
        field='model_projections_completed' if need_jacobian else 'forward_projections_completed'
        c.state[field]=c.state.get(field,0)+1;c.save()
        return result
    def scores(self,name,owner,method,c):
        path=self.root/'scores'/f'{name}_{method}_D{owner+1}.json'
        if path.exists():return read(path)['scores']
        if name=='F64':
            with np.load(self.bundle/'truth_projection.npz',allow_pickle=False) as z:
                candidate={n:torch.from_numpy(z[n].copy()) for n in ('forward','jacobian')}
        else:candidate=self.projection(name,c,method=='B3')
        reference=self.projection(f'D{owner+1}',c,method=='B3')
        coeff,v=self.keys(owner,method);data=self.data('embedding',owner)
        scale=math.sqrt(float(M.scales(data[2],data[3])[0 if method=='B1' else 1]))
        x=torch.from_numpy(self.load('validation_controls_f64.npy').reshape(-1,64))
        values=scoring.projected_scores(candidate,reference,x,coeff,v,self.basis,scale,method)
        write(path,dict(scores=values,true_index=0,wrong_count=4095,scale=scale,symmetric=True));return values
    def train(self,node,c,restored):
        owner=node['owner'];recipe=node['recipe'];name=node['id']
        if node.get('conditional'):
            selected=read(self.root/'products/mark_selection.json');setting=selected['selection']['B1']
            row=next(r for r in selected['grid']['B1'] if r['factor']==setting['factor'])
            if not row['owners'][owner]['qualified']:return dict(id=name,skipped='selected B1 did not qualify for this development owner',files={})
        kind='attack' if recipe=='forward32' else 'embedding' if recipe in ('sham','B1','B3') else 'timing16384' if node.get('cost_only') else 'training'
        data=self.data(kind,owner);reference=None;key=None;target=None;tau=None
        weights_name=node.get('weights') or (self.selected_name(owner,node['selected_weights']) if node.get('selected_weights') else None)
        weights=self.weights(weights_name) if weights_name else None
        if recipe in ('B1','B3'):
            reference=self.model(f'D{owner+1}');key=self.key(owner,recipe)
            with np.load(self.bundle/f'embedding_{owner}.npz',allow_pickle=False) as z:
                key['directions']=K.physical_key_directions(z['rho'],key['v'].detach().cpu().numpy(),self.device)
            target=node['factor']*read(self.root/'products/calibration.json')['floors'][recipe]
        if recipe=='A5B2':reference=self.model(weights_name);tau=read(self.root/'products/calibration.json')['tau5']
        seed=self.seeds['owners'][owner]
        shuffle=self.seeds['ordinary_shuffle'] if recipe=='forward32' else seed['embedding_shuffle'] if kind=='embedding' else seed['shuffle']
        init=self.seeds['replacement_initialization'] if name.startswith('R') else seed['initialization']
        engine=K.Engine(data,init,shuffle,self.device,recipe,model_factory=self.factory,weights=weights,reference=reference,key=key,target=target,tau=tau,basis=self.basis.float())
        start_updates=0
        if node.get('parent'):
            parent=torch.load(self.root/'states'/(node['parent']+'.pt'),map_location='cpu',weights_only=True)
            engine.restore(parent);start_updates=engine.updates
        target_updates=start_updates+node['steps']
        if restored is not None:engine.restore(restored)
        else:c.checkpoint(node,engine.state())
        timings=[];first=engine.updates;started=time.monotonic();bootstrap_end=time.time()+120
        rate=c.state['rates'].get(recipe)
        transfer=lambda:(math.ceil((target_updates-engine.updates)/2048)+4)*c.state['rates']['checkpoint_seconds']
        if rate is not None:c.budget.reserve((target_updates-engine.updates)*rate,min(2048,node['steps'])*rate,transfer())
        else:c.budget.reserve(120,120,30)
        try:
            while engine.updates<target_updates:
                c.check()
                if rate is None and time.time()>=bootstrap_end:raise TimeoutError('unmeasured kernel first-use timing cap')
                self.synchronize();t=time.monotonic()
                lr=(.0003 if recipe in ('sham','B1','B3') else .002)
                if node.get('cooldown'):lr*=max(0,(target_updates-engine.updates-1)/node['steps'])
                engine.step(lr);c.state['executed_updates']=c.state.get('executed_updates',0)+1;self.synchronize();timings.append(time.monotonic()-t)
                if len(timings)==20 and rate is None:
                    c.checkpoint(node,engine.state()) # full optimizer checkpoint, not an empty-state proxy
                    rate=statistics.mean(timings[-10:]);c.state['rates'][recipe]=rate
                    c.budget.reserve((target_updates-engine.updates)*rate,min(2048,node['steps'])*rate,transfer());c.save()
                    if c.state['session']['device']=='cuda':c.project_remaining()
                if (engine.updates-start_updates)%2048==0:c.checkpoint(node,engine.state())
        except BaseException as exc:
            partial=self.root/'states'/f'{name}.failed_{time.time_ns()}.pt';evidence_error=None
            try:torch.save(engine.state(),partial)
            except BaseException as persist:evidence_error=repr(persist)
            write(partial.with_suffix('.json'),dict(original_error=repr(exc),snapshot=partial.name if evidence_error is None else None,
                evidence_error=evidence_error,updates=engine.updates,timing_seconds=timings,accepted=False))
            raise
        c.checkpoint(node,engine.state())
        statepath=self.root/'states'/(name+'.pt');modelpath=self.root/'models'/(name+'.pt')
        # A product exists only after its receipt. A recovery preserves partials.
        for p in (statepath,modelpath):
            if p.exists():p.rename(p.with_name(p.name+f'.partial_{time.time_ns()}'))
        torch.save(engine.state(),statepath);torch.save(M.tensor_state(engine.model),modelpath)
        files=[p.relative_to(self.root).as_posix() for p in (statepath,modelpath)]
        result=dict(id=name,files=dict.fromkeys(files),updates=node['steps'],restored_update=first,model_updates=engine.updates,
            recipe=recipe,seconds=time.monotonic()-started,cost_only=bool(node.get('cost_only')),
            warmup_updates=min(20,len(timings)),timing_seconds=timings,
            peak_gpu_allocated=torch.cuda.max_memory_allocated() if self.device=='cuda' else None,
            peak_gpu_reserved=torch.cuda.max_memory_reserved() if self.device=='cuda' else None,
            timing_data='4096 development rows repeated four times; not independent samples' if node.get('cost_only') else None)
        result['warmup_seconds']=sum(timings[:20]);result['sustained_seconds']=sum(timings[20:])
        result['sustained_updates']=max(0,len(timings)-20)
        result['sustained_seconds_per_update']=statistics.mean(timings[20:]) if len(timings)>20 else None
        if len(timings)>20:c.state['rates'][recipe]=statistics.mean(timings[20:]);c.save()
        if node.get('evaluate'):
            evaluation=self.evaluate(name,owner,c);result['evaluation']=evaluation['metrics'];result['score']=evaluation['score']
            result['files'].update({p.relative_to(self.root).as_posix():None for p in (self.root/'evaluations'/name).rglob('*') if p.is_file()})
        return result
    def execute(self,node,c,restored=None):
        kind=node['kind'];name=node['id'];result=dict(id=name,files={})
        if kind=='train':return self.train(node,c,restored)
        if kind=='feasibility':
            # Required later calibration work on available D1/B10, cached and reused.
            for method in ('B1','B3'):self.scores('B10',0,method,c)
            result.update(measured_before_D2_D3=True,reused_by_calibration=True,
                rates=dict(c.state['rates']),full_campaign_fit_not_claimed=True)
            return result
        if kind=='reuse':
            import shutil
            target=self.root/'models'/(name+'.pt')
            if not target.exists():shutil.copyfile(self.bundle/node['source'],target)
            elif sha(target)!=sha(self.bundle/node['source']):raise ValueError('reused model drift')
            result['files'][target.relative_to(self.root).as_posix()]=None
            if name=='D1':
                # The complete Q6 measurements and designs remain immutable inputs.
                # K was not measured by Q6, so compute it without replaying its task.
                prior=read(self.bundle/'D1_RESULT.json');x=torch.from_numpy(self.load('validation_x.npy')).to(self.device)
                truth=self.load('key_truth_0.npy');rows=[];model=self.model(name);key=self.key(0,'B3')['v']
                phi=K.physical_key_directions(self.load('validation_controls_f64.npy'),key.detach().cpu().numpy(),self.device)
                with torch.no_grad():
                    for i in range(0,len(x),16):
                        c.check();xx=x[i:i+16];j=torch.func.jvp(model,(xx,),(phi[i:i+16],))[1]
                        rows+=rules.relative_rows(j.double().cpu().numpy()*.01,truth[i:i+16])
                means=prior['means'];metrics=dict(F=means['forward'],J=means['jvp'],T=means['task_f64'],K=rules.complete_mean(rows))
                p=self.root/'evaluations/D1/RESULT.json';write(p,dict(complete=metrics['K'] is not None,metrics=metrics,key_rows=rows,
                    reused_Q6_result=sha(self.bundle/'D1_RESULT.json'),task_replayed=False,plateaued=False,cap=81920,improvement_percent=11.4787))
                result['files'][p.relative_to(self.root).as_posix()]=None
            return result
        if kind=='screen':
            result.update(rules.clean_screen([self.metrics(f'D{i}') for i in (1,2,3)]))
            screen=self.root/'scores/CLEAN_SCREEN.json'
            if screen.exists():
                if read(screen)!=result:raise ValueError('restored clean screen differs')
            else:write(screen,result)
            result['files'][screen.relative_to(self.root).as_posix()]=None
            if not result['passed']:raise ValueError('clean training screen failed; cohort stops')
            result['descriptive_cross_gates']=[[{k:rules.gate(a[k],b[k]) for k in ('F','J','K','T')} for b in [self.metrics(f'D{i}') for i in (1,2,3)]] for a in [self.metrics(f'D{i}') for i in (1,2,3)]]
        elif kind=='calibrate':
            floors={};nulls={}
            for method in ('B1','B3'):
                rows=[]
                for owner in range(3):
                    for n in [f'D{j+1}' for j in range(3) if j!=owner]+[f'S{owner+1}','F64']:
                        rows.append(dict(owner=owner,null=n,score=self.scores(n,owner,method,c)[0]))
                    self.scores('B10',owner,method,c) # descriptive weak-null reference
                path=self.root/'scores'/(method+'_null_calibration.json')
                if path.exists():
                    if read(path)!=rows:raise ValueError('restored null calibration differs')
                else:write(path,rows)
                result['files'][path.relative_to(self.root).as_posix()]=None
                floors[method]=K.positive_scale(max(abs(r['score']) for r in rows),'null floor');nulls[method]=rows
            x=torch.from_numpy(self.load('validation_x.npy')).to(self.device).repeat_interleave(2,0)
            v=torch.from_numpy(self.load('validation_v.npy')).to(self.device);distances=[]
            for i in (1,2,3):
                for j in (1,2,3):
                    if i!=j:distances.append(scoring.distance(self.model(f'D{j}'),self.model(f'D{i}'),x,v,c.check))
            if len([d for d in distances if d['mean'] is not None])!=6:raise ValueError('six ordered B2 distances required')
            tau=statistics.median(d['mean'] for d in distances)
            result.update(floors=floors,deltas={k:.15*v for k,v in floors.items()},nulls=nulls,distances=distances,tau5=K.positive_scale(tau,'tau5'))
        elif kind=='select_marks':
            calibration=read(self.root/'products/calibration.json');grids={};selection={}
            for method in ('B1','B3'):
                rows=[]
                for factor in (1.5,2.,3.):
                    owners=[]
                    for owner in range(3):
                        n=f'{method}_{factor:g}_D{owner+1}';values=self.scores(n,owner,method,c)
                        nullmax=max(abs(r['score']) for r in calibration['nulls'][method] if r['owner']==owner)
                        theta=max(calibration['floors'][method],nullmax,max(map(abs,values[1:])))+calibration['deltas'][method]
                        gates={k:rules.gate(self.metrics(n)[k],self.metrics(f'D{owner+1}')[k]) for k in ('F','J','K','T')}
                        owners.append(dict(complete=all(g['status']=='complete' for g in gates.values()),
                            qualified=values[0]>theta and all(g['passed'] is True for g in gates.values()),margin=values[0]-theta,
                            theta=theta,true_score=values[0],wrong_max=max(map(abs,values[1:])),gates=gates))
                    rows.append(dict(factor=factor,owners=owners))
                gridpath=self.root/'scores'/(method+'_qualification_grid.json')
                if gridpath.exists():
                    if read(gridpath)!=rows:raise ValueError('restored mark qualification differs')
                else:write(gridpath,rows)
                result['files'][gridpath.relative_to(self.root).as_posix()]=None
                grids[method]=rows;selection[method]=rules.select_mark(rows,method)
            result.update(grid=grids,selection=selection)
        elif kind=='select_replacement':
            rows=[dict(length=L,score=read(self.root/'products'/f'Rstar_{L}.json')['score']) for L in (10240,20480,40960,81920)]
            result.update(rows=rows,selected=rules.select_replacement(rows),frozen_before_final_owners=True)
        elif kind=='wrappers':
            x=torch.from_numpy(self.load('validation_x.npy')[:1]).to(self.device);v=torch.from_numpy(self.load('validation_v.npy')[:1]).to(self.device)
            rows=[]
            for m in (16,256):
                c.budget.reserve(180,180,30);start=time.monotonic();torch.cuda.reset_peak_memory_stats();w=K.Jacobi(self.model('D1'),m)
                torch.cuda.synchronize();a=time.monotonic();p=w(x);torch.cuda.synchronize();f=time.monotonic()-a
                a=time.monotonic();j=torch.func.jvp(w,(x,),(v,))[1];torch.cuda.synchronize();jt=time.monotonic()-a
                xx=x.clone().requires_grad_();a=time.monotonic();g=torch.autograd.grad(w(xx).square().mean(),xx)[0];torch.cuda.synchronize();rev=time.monotonic()-a
                if any(not torch.isfinite(z).all() for z in (p,j,g)):raise FloatingPointError('wrapper nonfinite')
                if time.monotonic()-start>180:raise TimeoutError('wrapper timing phase cap')
                rows.append(dict(m=m,forward_seconds=f,jvp_seconds=jt,reverse_seconds=rev,peak_bytes=torch.cuda.max_memory_allocated(),
                    change_from_model=float(torch.linalg.vector_norm(p-self.model('D1')(x))),
                    relative_F64_error=rules.relative_rows(p.double().detach().cpu().numpy()*.01,self.load('validation_truth_f64.npy')[:1].reshape(1,-1))[0],
                    physical_convergence_claim=False))
            result['rows']=rows
        elif kind=='scoring_cost':
            start=time.monotonic()
            for method in ('B1','B3'):self.scores(self.selected_name(0,'M3'),0,method,c)
            result.update(cache_read_seconds=time.monotonic()-start,full_projection_receipts=[p.name for p in (self.root/'scores').glob('*projection.json')],
                note='Actual full-Jacobian times recorded on first use; no repeated profiling pass',final_population_projection_only=True)
        elif kind=='analysis':
            c.budget.reserve(600,600,30);began=time.monotonic();cal=read(self.root/'products/calibration.json')
            comparisons=[];b2_rows={source:[] for source in ('O','M3','M1')}
            x=torch.from_numpy(self.load('validation_x.npy')).to(self.device).repeat_interleave(2,0)
            v=torch.from_numpy(self.load('validation_v.npy')).to(self.device)
            length=read(self.root/'products/replacement_selection.json')['selected'];replacement=f'Rstar_{length}'
            distance_cache={}
            def dist(candidate,owner):
                key=(candidate,owner)
                if key not in distance_cache:distance_cache[key]=scoring.distance(self.model(candidate),self.model(owner),x,v,c.check)
                return distance_cache[key]['mean']
            for owner in range(3):
                for source in ('O','M3','M1'):
                    c.check()
                    if time.monotonic()-began>=600:raise TimeoutError('development descriptive readout initial cap')
                    name=f'Aord_{source}_D{owner+1}';product=read(self.root/'products'/(name+'.json'))
                    if product.get('skipped'):
                        comparisons.append(dict(owner=owner,source=source,not_run=product['skipped']));continue
                    metrics=self.metrics(name);gates={k:rules.gate(metrics[k],self.metrics(f'D{owner+1}')[k]) for k in ('F','J','K','T')}
                    detectors={}
                    for method in ('B1','B3'):
                        values=self.scores(name,owner,method,c)
                        selected=read(self.root/'products/mark_selection.json')
                        setting=selected['selection'][method]['factor']
                        threshold=next(r for r in selected['grid'][method] if r['factor']==setting)['owners'][owner]['theta']
                        detectors[method]=dict(true=values[0],wrong_max=max(map(abs,values[1:])),threshold=threshold,
                            detected=values[0]>threshold and values[0]>max(map(abs,values[1:]))+cal['deltas'][method])
                    matched=[f'Aord_{source}_D{j+1}' for j in range(3) if j!=owner and (self.root/'models'/f'Aord_{source}_D{j+1}.pt').exists()]
                    independent=[f'D{j+1}' for j in range(3) if j!=owner]+[replacement]
                    lineage=rules.b2(dist(name,f'D{owner+1}'),[dist(n,f'D{owner+1}') for n in matched],[dist(n,f'D{owner+1}') for n in independent])
                    utility=all(g['passed'] is True for g in gates.values()) if all(g['status']=='complete' for g in gates.values()) else None
                    b2_rows[source].append(dict(owner=owner,positive=lineage['positive'],utility=utility,margin=lineage['margin']))
                    rmetrics=self.metrics(replacement)
                    ratios={k:(rmetrics[k]/metrics[k] if rules.nonnegative(rmetrics[k]) and rules.nonnegative(metrics[k]) and metrics[k]>0 else None) for k in ('J','T')}
                    prior=read(self.bundle/'D1_RESULT.json')['task_f64'] if owner==0 else [r['relative'] for r in read(self.root/'evaluations'/f'D{owner+1}'/'RESULT.json')['rows']['T']]
                    actual=[r['relative'] for r in read(self.root/'evaluations'/name/'RESULT.json')['rows']['T']]
                    delta=[a-b if a is not None and b is not None else None for a,b in zip(actual,prior)]
                    row=dict(owner=owner,source=source,metrics=metrics,utility_gates=gates,detectors=detectors,B2=lineage,
                        replacement_over_copy=ratios,paired_task_error_differences=delta,
                        paired_task_sd=statistics.stdev(delta) if len(delta)==16 and all(z is not None for z in delta) else None)
                    comparisons.append(row)
            result.update(selection=self.selection(),rates=c.state['rates'],development_descriptive=True,final_decision_applied=False,
                wrong_key_conditional_bound_per_method=210/4096,empirical_FPR=False,physical_claims=False,
                comparisons=comparisons,B2={k:rules.b2_cohort(v) for k,v in b2_rows.items()},
                note='One development replacement trajectory is reused descriptively; this is not the final paired ten-owner experiment.')
        else:raise ValueError('unknown graph operation '+kind)
        return result
