"""One owned process, frozen null210/B2 population, durable per-owner output.

This prospective scope does not promote a D2 or noise runtime receipt. The
new null-scoped gate verifies frozen expectations and records fresh observations.
Original model admissions, kernels, checkpoints and historical outputs stay intact.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, os, shutil, sys, time, traceback
from null_io import need, read, put, sha, source, pack, _http, _upload
from noise_producer import accepted, prerequisites

SCOPE='fixed_null210_B2_1240_only'
POLICY='NULL_B2_POLICY.json'


def verify_inputs(job,policy,check):
    code=job.parent/'payload/code'
    need(len(policy['source_pins'])==338,'Complete unchanged scientific source pins required')
    for name,h in policy['source_pins'].items():
        need(sha(source(code,name),check)==h,'Scientific source drift: '+name)
    maps=[policy['files']]+[s['files'] for s in policy['sources']]
    maps.append({k:v['expected'] for k,v in policy['aliases'].items()})
    seen={}
    for group in maps:
        for name,expected in group.items():
            check();identity={k:expected[k] for k in ('bytes','sha256')}
            if name in seen:
                need(seen[name]==identity,'Conflicting frozen pins');continue
            p=source(job,name)
            need(p.stat().st_size==expected['bytes'] and sha(p,check)==expected['sha256'],
                 'Frozen input differs: '+name)
            seen[name]=identity
    need(sha(job/'NULL_B2_ROSTER.json',check)==policy['roster_sha256'],'Roster bytes differ')
    return read(job/'NULL_B2_ROSTER.json')


def check_roster(roster,opening,graph):
    need(roster['owners']==[f'O{i:02d}' for i in range(1,11)],'Ten owners required')
    need(len(roster['model_jobs'])==1240 and len(roster['groups'])==1240,'Frozen B2 population differs')
    jobs={r['id']:r for r in roster['model_jobs']}
    need(len(jobs)==1240,'Duplicate fixed model identity')
    expected_pairs=set();expected_groups=set()
    for owner in roster['owners']:
        need(roster['keyed'][owner]==graph['keyed_nulls'][owner],'Keyed null population differs')
        need(len(roster['keyed'][owner])==21 and all('B10' not in v for v in roster['keyed'][owner]),'B10 keyed exclusion required')
        for label,g in graph['b2_groups'][owner].items():
            name=owner+'/'+label;expected_groups.add(name)
            need(not g['missing'],'Frozen group has unavailable original inputs')
            bound=opening['b2_groups'][owner][name]
            row=roster['groups'][name]
            need(bound=={k:g[k] for k in ('matched','independent')}
                 =={k:row[k] for k in ('matched','independent')},'B2 comparison identities changed')
            labels=[name]+g['matched']+g['independent']
            need(len(labels)==len(set(labels))==29 and row['reference_owner']==owner and row['descendant']==name,
                 'B2 matched/independent population differs')
            actual={}
            for value in labels:
                other,variant=value.split('/')
                if variant in ('noise_small_M3','noise_large_M3'):variant='M3'
                actual[value]=other+'/'+variant
                expected_pairs.add((owner,actual[value]))
            need(row['canonical_candidates']==actual,'Noise pathwise or model alias differs')
    need(set(roster['groups'])==expected_groups,'Unexpected comparison group')
    need({(owner,r['id']) for r in jobs.values() for owner in r['reference_owners']}==expected_pairs
         and len(expected_pairs)==12390,'Unique B2 pair population differs')
    need({v for _,v in expected_pairs}==set(jobs),'Unused/missing model variant')
    for name,row in jobs.items():
        need(name==row['owner']+'/'+row['name'] and row['descriptor']==opening['models'][row['owner']][row['name']],
             'Frozen model descriptor differs')
    need(set(roster['reused_projections'])=={o+'/'+r for o in roster['owners'] for r in ('O','Rstar','S')},
         'Exactly thirty frozen final projections required')


def load_model(item):
    """Same constructor/transform body as the pinned O.load_final_model.

    Historical metadata admission is checked independently for every item.
    Its old runtime guard is preserved; the prospective scoped gate is separate.
    """
    import q5_model as Q
    from msfinal import models, attacks as T
    from msdev.kernels import Jacobi
    model=(Q.Q5FNO() if item.procedure is None else models.create(item.procedure['architecture'])).to('cuda').eval()
    model.load_state_dict(T._parent_weights(item),strict=True)
    if item.procedure is not None:
        transform=item.procedure['transform']
        if transform is not None:
            endpoint=None
            if item.dependencies:
                path,h=item.dependencies['endpoint'];endpoint=Q.Q5FNO().to('cuda').eval()
                endpoint.load_state_dict(T._parent_weights(SimpleNamespace(path=path,sha256=h)),strict=True)
            model=models.transformed(model,transform,endpoint=endpoint)
        return model
    return model if item.wrapper_steps is None else Jacobi(model,item.wrapper_steps)


def run(job,policy_sha,deadline):
    job=Path(job).resolve();need(sha(job/POLICY)==policy_sha,'Null policy bytes differ')
    policy=read(job/POLICY);activation=read(job/'ACTIVATION.json')
    need(policy['scope']==activation.get('scope')==SCOPE and activation['policy_sha256']==policy_sha
         and activation['retries']==0 and activation['previous_clocks_reused'] is False,
         'Fixed null activation required')
    need(activation['start']<=time.time()<deadline<=activation['deadline']-policy['return_reserve_seconds'],
         'Fixed producer window expired')
    out=job/'output';out.mkdir(exist_ok=False)
    start=time.monotonic();last=start;current='input admission';completed={};durable={};pair_means={}
    counts=dict(model_variants=0,jvp_batches=0,jvp_samples=0,B2_pairs=0,keyed_entries=0,T128_forward=0,T128_tangent=0)
    def check():
        nonlocal last
        if time.time()>=deadline:raise TimeoutError('Fixed null/B2 work deadline reached; no retry')
        now=time.monotonic()
        if now-last>=20:
            last=now
            need(shutil.disk_usage(job).free>=2*1024**3,'Less than 2 GiB scratch remains')
            print(json.dumps(dict(current=current,model_variants=counts['model_variants'],total=1240,
                durable_owners=len(durable),keyed_entries=counts['keyed_entries'],seconds=round(now-start,1))),flush=True)
    try:
        roster=verify_inputs(job,policy,check)
        opening=read(job/policy['opening']);graph=read(job/'admission/FINAL_GRAPH.json')
        check_roster(roster,opening,graph)
        code=job.parent/'payload/code';sys.path[:0]=[str(code/'tools'),str(code)]
        from q6_common import THREAD_ENV
        os.environ.update(THREAD_ENV)
        from null_runtime import verify_gate
        from msfinal import backend as B, materials as A, evaluation as E, orchestration as O, nulls
        import torch
        import numpy as np
        from null_compute import collect_jvp,distance_rows,save_jvp,load_projection,keyed_readout,f64_distance,frozen_b2_readouts
        gate=verify_gate(job,policy,policy_sha,deadline,check,phase='start',member='all_owners')
        gate_sha=sha(job/'NULL_B2_RUNTIME_GATE.json')
        admissions={};items={};qualifications={};catalogs={}
        for owner in roster['owners']:
            check();a=B.admit(prerequisites(job,policy,owner));A.preflight(a);E.preflight(a,'final')
            admissions[owner]=a;catalogs[owner]=A.catalog(a)
            qualifications[owner]=nulls.qualification(a,job/f'qualification_current/{owner}/QUALIFICATION.json',
                accepted(job,policy,f'qualification_current/{owner}/QUALIFICATION_ACCEPTED.json'))
        for owner in roster['owners']:
            nulls.common_controls(admissions['O01'],admissions[owner])
            need(catalogs[owner]['entries']['public_basis']['sha256']==catalogs['O01']['entries']['public_basis']['sha256'],
                 'Common projection basis differs')
        for row in roster['model_jobs']:
            check();d=row['descriptor']
            dependencies={k:(source(job,row['dependency_paths'][k]),h) for k,h in d.get('dependencies',{}).items()}
            item=O.FinalModel(row['name'],source(job,row['model_path']),d['model_sha256'],
                accepted(job,policy,row['acceptance']),d['wrapper_steps'],procedure=d.get('procedure'),dependencies=dependencies)
            O.final_model(item,admissions[row['owner']])
            need(O.model_descriptor(item)==d,'Admitted model descriptor differs')
            items[row['id']]=item
        projections={}
        for label,expected in roster['reused_projections'].items():
            owner,role=label.split('/');a=admissions[owner];cat=catalogs[owner]
            descriptor=hashlib.sha256(json.dumps(dict(model=items[label].sha256,wrapper_steps=None),sort_keys=True).encode()).hexdigest()
            bindings=dict(model=descriptor,runtime=a.prerequisites.runtime.sha256,
                catalog=a.prerequisites.assets.sha256,controls=cat['entries']['final_detector_x']['sha256'],
                basis=cat['entries']['public_basis']['sha256'],scoring=policy['source_pins']['tools/msdev/scoring.py'])
            need(bindings==expected['bindings'] and expected['phase']=='final','Historical projection binding substitution')
            key=E.cache_identity(model_sha256=descriptor,runtime_sha256=bindings['runtime'],catalog_sha256=bindings['catalog'],
                controls_sha256=bindings['controls'],basis_sha256=bindings['basis'],scoring_sha256=bindings['scoring'],phase='final')
            need(key==expected['identity'],'Historical projection identity recomputation differs')
            projections[label]=load_projection(source(job,expected['path']),expected)
        put(out/'INPUT_PREFLIGHT.json',dict(complete=True,models=1240,projections_reused=30,
            all_model_bytes_admitted_before_first_call=True,scope=SCOPE,policy_sha256=policy_sha,
            opening_sha256=policy['opening_sha256'],runtime_gate_sha256=gate_sha,
            historical_runtime_records_unchanged=True,new_projection_model_calls=0,
            model_calls=0,scientific_acceptance=False))
        common=admissions['O01'];x=torch.from_numpy(A.array(common,'final_detector_x','final',(128,64),'float32')).to('cuda')
        v=torch.from_numpy(A.array(common,'final_detector_v','final',(128,2,64),'float32')).to('cuda')
        binding=dict(scope=SCOPE,policy_sha256=policy_sha,roster_sha256=policy['roster_sha256'],
            opening_sha256=policy['opening_sha256'],runtime_gate_sha256=gate_sha,
            runtime_payload_sha256=gate['runtime_payload_sha256'],
            controls_sha256=catalogs['O01']['entries']['final_detector_x']['sha256'],
            directions_sha256=catalogs['O01']['entries']['final_detector_v']['sha256'],
            scoring_source_sha256=policy['source_pins']['tools/msdev/scoring.py'],
            null_source_sha256=policy['source_pins']['tools/msfinal/nulls.py'])
        put(out/'START.json',dict(binding=binding,training_updates=0,inverse_updates=0,
                                 native_reference_calls_authorized=8320,native_optimization_repeated=False,
                                 scientific_acceptance=False,canonical_worker_advanced=False))
        from null_reference import prepare
        current='missing F64 null reference projections';check()
        truth_projection,truth_record=prepare(common,out/'f64_reference',check,binding)
        need(truth_record['native']['counts']==truth_record['native']['completed']==dict(T128_forward=128,T128_tangent=8192),
             'Incomplete F64 reference supplement')
        counts.update(truth_record['native']['counts'])
        clean={};clean_meta={}
        for owner in roster['owners']:
            current=owner+'/original reference JVP';check()
            model=load_model(items[owner+'/O']);array=collect_jvp(model,x,v,check)
            clean[owner]=array;clean_meta[owner]=save_jvp(out/'reference_jvp'/(owner+'.npy'),array)
            put(out/'reference_jvp'/(owner+'.json'),dict(binding=binding,model=owner+'/O',
                descriptor=O.model_descriptor(items[owner+'/O']),array=clean_meta[owner]))
            counts['jvp_batches']+=16;counts['jvp_samples']+=256
            del model;torch.cuda.empty_cache()
        for owner in roster['owners']:
            current=owner+'/keyed nulls';check()
            rows=keyed_readout(admissions[owner],qualifications[owner],projections,roster,truth_projection,check)
            f64=next(r for r in rows if r['model']=='F64');f64.update(f64_distance(admissions[owner],clean[owner],check))
            put(out/'nulls'/owner/'KEYED_NULLS.json',dict(rows=[r for r in rows if r['model']!='F64'],binding=binding,
                qualification_sha256=sha(job/f'qualification_current/{owner}/QUALIFICATION.json'),
                projection_inputs={label:value for label,value in roster['reused_projections'].items()
                    if label.endswith(('/O','/Rstar')) or label==owner+'/S'},
                population_FPR=None,independent_acceptance=False,required_F64_separate=True,complete=True))
            put(out/'nulls'/owner/'F64_NULL.json',dict(f64,binding=binding,complete=True,
                reference_supplement_sha256=sha(out/'f64_reference/REFERENCE_RESULT.json')))
            counts['keyed_entries']+=len(rows)
            print(owner+': all 21 keyed-null entries complete (saved model projections plus the shared F64 supplement).',flush=True)
        # One transfer log for the entire owned producer; every upload gets a
        # durable readback receipt. No Colab-local evidence is deleted here.
        transfer=_http(job,policy,'rolling',policy['aggregate_return_bound_bytes']*2)
        for owner in roster['owners']:
            root=out/'owners'/owner
            for row in (r for r in roster['model_jobs'] if r['owner']==owner):
                current=row['id'];check();item=items[current]
                if row['name']=='O':array=clean[owner];array_record=dict(clean_meta[owner],path='output/reference_jvp/'+owner+'.npy')
                else:
                    model=load_model(item);array=collect_jvp(model,x,v,check)
                    array_record=save_jvp(root/'raw'/row['name']/ 'JVP.npy',array)
                    array_record['path']=(root/'raw'/row['name']/'JVP.npy').relative_to(job).as_posix()
                    counts['jvp_batches']+=16;counts['jvp_samples']+=256
                    del model;torch.cuda.empty_cache()
                rows={};means={}
                for ref in row['reference_owners']:
                    value=distance_rows(array,clean[ref],check);rows[ref]=value['rows'];means[ref]=value['mean']
                    pair_means[(ref,current)]=value['mean'];counts['B2_pairs']+=1
                raw_path=root/'raw'/row['name']/'DISTANCE_ROWS.json'
                put(raw_path,dict(model=current,binding=binding,rows=rows,means=means,array=array_record))
                summary=dict(model=current,binding=binding,descriptor=row['descriptor'],array=array_record,
                    reference_means=means,rows_per_reference=256,rows_sha256=sha(raw_path),
                    rows_path=raw_path.relative_to(job).as_posix(),complete=True,
                    full_field_JVP=True,projection_approximation=False)
                put(root/'summaries'/(row['name']+'.json'),summary)
                completed[current]=summary;counts['model_variants']+=1
            current=owner+'/durable save';check()
            owner_result=dict(owner=owner,models=[r['id'] for r in roster['model_jobs'] if r['owner']==owner],
                binding=binding,complete=True,producer_process_still_running=True,formal_custody_assurance=False)
            put(root/'OWNER_RESULT.json',owner_result)
            entries={p.relative_to(job).as_posix():p for p in root.rglob('*') if p.is_file()}
            for p in (out/'reference_jvp').iterdir():entries[p.relative_to(job).as_posix()]=p
            for p in (out/'nulls').rglob('*.json'):entries[p.relative_to(job).as_posix()]=p
            for name in ('NULL_B2_POLICY.json','NULL_B2_ROSTER.json','NULL_B2_RUNTIME_GATE.json','ACTIVATION.json'):
                entries[name]=job/name
            archive=job/'capsules'/f'NULL_B2_{activation["attempt"]}_{owner}.zip';archive.parent.mkdir(exist_ok=True)
            manifest=pack(archive,entries,policy['owner_capsule_bound_bytes'],check)
            receipt=_upload(job,policy,transfer,archive,min(deadline,time.time()+policy['owner_upload_seconds']),check)
            durable[owner]=dict(receipt,owner=owner,models=owner_result['models'],
                manifest_sha256=manifest['sha256'],producer_closure_pending=True)
            put(job/'owner_receipts'/(owner+'.json'),durable[owner])
            print(owner+': 124 model variants SAVED AND READ BACK IN DRIVE. '+str(len(durable))+'/10 owners durable.',flush=True)
        current='B2 fixed comparison readouts';check()
        readouts=frozen_b2_readouts(roster,pair_means)
        for label,result in readouts.items():
            put(out/'b2'/(label+'.json'),dict(result,binding=binding,descendant=label,
                comparison=roster['groups'][label],scientific_acceptance=False))
        expected=dict(model_variants=1240,jvp_batches=19840,jvp_samples=317440,B2_pairs=12390,keyed_entries=210,
                      T128_forward=128,T128_tangent=8192)
        need(counts==expected and len(durable)==10 and len(readouts)==1240,'Incomplete null/B2 population')
        current='final immutable-input and runtime readback'
        verify_inputs(job,policy,check)
        verify_gate(job,policy,policy_sha,deadline,check,phase='end',member='all_owners')
        result=dict(complete=True,status='NULL210_B2_1240_COMPUTED_CLOSURE_PENDING',counts=counts,
            B2_groups=len(readouts),durable_owners=durable,binding=binding,
            final_projections_reused=30,new_projection_model_calls=0,
            training_updates=0,inverse_updates=0,native_solver_calls=8320,native_optimization_repeated=False,
            scientific_acceptance=False,formal_custody_assurance=False,canonical_worker_advanced=False,
            complete_programme=False,remaining=['owner-level synthesis and uncertainty','figures and cost analysis',
                'reproducibility and release','paper preparation'],seconds=time.monotonic()-start)
        put(out/'RESULT.json',result);return result
    except BaseException:
        events=out/'f64_reference/native/events.jsonl'
        if events.is_file():
            for kind in ('T128_forward','T128_tangent'):
                counts[kind]=sum(json.loads(line).get('event')=='intent' and json.loads(line).get('kind')==kind
                                 for line in events.read_text().splitlines())
        put(out/'FAILURE.json',dict(current=current,traceback=traceback.format_exc(),counts=counts,
            completed_models=list(completed),durable_owners=durable,seconds=time.monotonic()-start,
            complete=False,retry=False,scientific_acceptance=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    p.add_argument('--policy-sha',required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();run(a.job,a.policy_sha,a.deadline)
