"""Deterministic analysis of saved measurements. No model/solver execution."""
import ast, collections, hashlib, json, math, pathlib, statistics, types
import argparse
HERE=pathlib.Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=pathlib.Path,required=True)
OUT=parser.parse_args().output.resolve();OUT.mkdir(parents=True,exist_ok=True)
INPUT=HERE/'inputs'
CODE=HERE/'frozen_source'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
pins=read(HERE/'ANALYSIS_SOURCE_PINS.json');rule_provenance={}
def functions(rel,names,namespace):
    p=CODE/rel;b=p.read_bytes();digest=hashlib.sha256(b).hexdigest()
    assert pins[rel]==digest,('analysis source pin differs',rel)
    tree=ast.parse(b);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    # Only the named pure statistical functions execute; module imports/workers do not.
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),namespace)
    rule_provenance[rel]={'sha256':digest,'functions':names}
    return namespace
ns={'math':math,'statistics':statistics,'OWNER_COUNT':10,'PRIMARY':('random_jvp','task')}
functions('experiments/tpfa32_copy_study/core.py',['positive'],ns)
functions('experiments/tpfa32_copy_study/analyze.py',['interval','quality_label'],ns)
an={'M':types.SimpleNamespace(finite=finite)}
functions('tools/msfinal/analysis.py',['cohort','joint','assembled_readouts'],an)
bn={};functions('tools/msdev/rules.py',['b2_cohort'],bn)
d=read(INPUT/'MODEL_ANALYSIS_INPUTS.json');s=read(INPUT/'SUPPLEMENT_ANALYSIS_INPUTS.json');g=read(INPUT/'FROZEN_GRAPH.json');owners=g['planned_owners']
qualified=[o for o in owners if s['qualification'][o]['qualified'] is True]
assert len(qualified)==10
def utility(o,m):
    c=d[o+'/O']['metrics']
    return {k:(m[k]<=1.25*c[k] if finite(m.get(k)) and m[k]>=0 else None) for k in 'FJKT'}
def all_gates(gates):
    if any(v is False for v in gates.values()):return False
    return True if all(v is True for v in gates.values()) else None
def detection(o,row):
    q=s['qualification'][o]
    if row['score_count']!=4096:return None
    return row['score_true']>q['theta'] and row['score_true']>row['wrong_max_abs']+q['delta']
intact={o:{'detected':detection(o,d[o+'/M3']),'utility_gates':utility(o,d[o+'/M3']['metrics'])} for o in owners}
per={o:{'qualification':o in qualified,'removal':{}} for o in owners};readouts={}
for key,row in d.items():
    o,v=key.split('/');u=utility(o,row['metrics']);det=detection(o,row)
    r={'utility_gates':u,'all_utility':all_gates(u),'detected':det,'task_status':row['task_status']}
    if 'M3' in v:
        initial=intact[o];prereq=initial['detected'] is True and all_gates(initial['utility_gates']) is True
        if 'frozen_D2_readout' in row:
            frozen=row['frozen_D2_readout'];assert frozen['utility_gates']==u;assert frozen['final_intact']['detected']==initial['detected']
            r.update(frozen)
        else:
            assert det is not None and r['all_utility'] is not None,key
            r.update(qualified_parent_attack_evasion=(not det and r['all_utility']),
                qualified_removal=prereq and not det and r['all_utility'],
                retained_detection=prereq and det and r['all_utility'],final_intact=initial)
        r['same_test_intact_prerequisite']=prereq
        if v!='M3':per[o]['removal'][v]=r
    readouts[key]=r

ratios={k:[d[o+'/Rstar']['metrics'][m]/d[o+'/Aord_M3_raw1920']['metrics'][m] for o in owners] for k,m in [('random_jvp','J'),('task','T')]}
adv={k:ns['interval'](v,.975) for k,v in ratios.items()};eq={k:ns['interval'](v,.95) for k,v in ratios.items()}
primary={'planned_owners':owners,'N':10,'qualification_filters_primary':False,'advantage':adv,'equivalence':eq,'label':ns['quality_label'](adv,eq),
    'ratio':'Rstar error / Aord_M3_raw1920 error','assumptions':'iid approximately normal owner log ratios, conditional on shared fixed benchmark and development procedure',
    'equivalence_coverage':'Separate 95% pointwise intervals; all components must pass; not simultaneous 95% coverage.'}
measure={o:{v:r for key,r in d.items() if (o2:=key.split('/')[0])==o for v in [key.split('/')[1]]} for o in owners}
cohorts=an['assembled_readouts'](g,per,primary,measure)
assert not any(cohorts['missing'].values()),cohorts['missing']
by_variant={}
for b in s['null_B2']['B2_rows']:
    key=b['owner']+'/'+b['variant'];r=readouts[key]
    row=dict(b,utility=r['all_utility'],utility_gates=r['utility_gates'])
    by_variant.setdefault(b['variant'],[]).append(row)
b2={v:bn['b2_cohort'](rows) for v,rows in by_variant.items()}
assert len(b2)==124 and all(v['n']==10 for v in b2.values())
# All variant summaries descriptive; only the frozen paired primary is inferential.
variants={}
for v in sorted({k.split('/')[1] for k in d}):
    keys=[o+'/'+v for o in owners if o+'/'+v in d]
    variants[v]={'n':len(keys),'detected':sum(readouts[k]['detected'] is True for k in keys),
      'all_utility':sum(readouts[k]['all_utility'] is True for k in keys),
      'qualified_removal':sum(readouts[k].get('qualified_removal') is True for k in keys) if 'M3' in v else None,
      'retained_detection_and_utility':sum(readouts[k].get('retained_detection') is True for k in keys) if 'M3' in v else None,
      'utility_failures':{m:sum(readouts[k]['utility_gates'][m] is False for k in keys) for m in 'FJKT'},
      'metric_medians':{m:statistics.median(d[k]['metrics'][m] for k in keys) if all(finite(d[k]['metrics'].get(m)) for k in keys) else None for m in ['F','J','K','T','P256']},
      'metric_ranges':{m:[min(d[k]['metrics'][m] for k in keys),max(d[k]['metrics'][m] for k in keys)] if all(finite(d[k]['metrics'].get(m)) for k in keys) else None for m in ['F','J','K','T','P256']},
      'task_unmeasured':sum(d[k]['task_status']=='unmeasured' for k in keys)}
nulls=s['null_B2']['null_rows']
api={}
for key,row in s['API']['jobs'].items():
    o,v=key.split('/');a=api.setdefault(v,{'streams':0,'attempts':0,'status_counts':{},'per_owner':{}});a['streams']+=1;a['attempts']+=row['attempts'];a['per_owner'][o]=row['status_counts']
    for status,n in row['status_counts'].items():a['status_counts'][status]=a['status_counts'].get(status,0)+n
write(OUT/'FINAL_ANALYSIS.json',{'schema':'OperatorMark.saved_result_analysis.20261025.v1','primary':primary,'intact':intact,'qualification_roster':qualified,
    'cohorts':cohorts,'variants':variants,'B2':b2,'native':s['native'],'API':api,'adaptive':s['adaptive'],
    'nulls':{'entries':len(nulls),'detected':sum(r['detected'] for r in nulls),'population_FPR_bound':None,'dependent_fixed_cases':True},
    'rule_source_pins':rule_provenance,'model_readouts':readouts,'D2':s['D2_summary'],
    'formal_custody_assurance':False,'canonical_worker_advanced':False,'physical_qualification':False,'new_numerical_experiments':0})
print('PRIMARY',primary['label'],{k:{f:v[f] for f in ['geometric_mean','lower','upper']} for k,v in adv.items()})
print('EQUIVALENCE',{k:{f:v[f] for f in ['lower','upper']} for k,v in eq.items()})
print('INTACT',{o:r['detected'] for o,r in intact.items()})
print('COHORTS',[(v,x['primary'],x['readout']['category'],x['readout']['counts'][x['primary']]) for v,x in cohorts['cohorts'].items()])
print('PRIMARY VARIANT',variants['Aord_M3_raw1920'])
print('B2_PRIMARY',{k:{f:b2[k][f] for f in ['positive','utility','positive_and_utility','evaded_and_utility','category']} for k in ['M3','Aord_M3_raw1920','Aord_O_raw1920']})
print('API', {v:a['status_counts'] for v,a in api.items()})
