"""Brief corrections; no new floors, exclusions, or physical claims."""
from __future__ import annotations
import math
import statistics

def nonnegative(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0

def gate(candidate,owner):
    if not nonnegative(candidate) or not nonnegative(owner):return dict(status='unavailable',passed=None,candidate=candidate,owner=owner)
    return dict(status='complete',passed=candidate<=1.25*owner,candidate=candidate,owner=owner,threshold=1.25*owner)

def relative_rows(pred,truth):
    import numpy as np
    p,t=np.asarray(pred,dtype=np.float64),np.asarray(truth,dtype=np.float64)
    if p.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(p).all():raise ValueError('invalid/nonfinite truth or prediction')
    # Source Q6 uses the actual Euclidean truth norm, with NO fixed epsilon.
    absolute=np.linalg.norm(p-t,axis=-1);den=np.linalg.norm(t,axis=-1)
    if not np.isfinite(absolute).all() or not np.isfinite(den).all():raise ValueError('nonfinite error/truth norm')
    if any(d>0 and not math.isfinite(float(a/d)) for a,d in zip(absolute,den)):raise ValueError('nonfinite relative error')
    return [dict(absolute=float(a),truth_norm=float(d),relative=None if d==0 else float(a/d)) for a,d in zip(absolute,den)]

def complete_mean(rows,key='relative'):
    values=[r[key] for r in rows]
    return None if not values or any(not nonnegative(v) for v in values) else statistics.mean(values)

def score(metrics):
    v=[metrics[k] for k in ('F','J','T')]
    if any(not nonnegative(x) for x in v):return None
    return 0. if 0. in v else math.exp(sum(map(math.log,v))/3)

def select_replacement(rows,lengths=(10240,20480,40960,81920)):
    if [r['length'] for r in rows]!=list(lengths) or any(not nonnegative(r.get('score')) for r in rows):
        raise ValueError('all required replacement evaluations must be complete and finite')
    minimum=min(r['score'] for r in rows)
    return min(r['length'] for r in rows if r['score']<=1.01*minimum)

def select_mark(rows,method):
    if len(rows)!=3 or sorted(r['factor'] for r in rows)!=[1.5,2.,3.]:raise ValueError('complete three-setting grid required')
    for r in rows:
        if len(r['owners'])!=3 or any(o.get('complete') is not True for o in r['owners']):raise ValueError('incomplete mark grid')
    ranked=sorted(rows,key=lambda r:(-sum(o['qualified'] for o in r['owners']),
        -min(o['margin'] for o in r['owners']),r['factor']))
    chosen=ranked[0];n=sum(o['qualified'] for o in chosen['owners'])
    if n<2 and method=='B3':raise ValueError('B3 has fewer than two qualifying development owners')
    if n<2:chosen=next(r for r in rows if r['factor']==2.)
    return dict(factor=chosen['factor'],insufficient=n<2,qualified_count=sum(o['qualified'] for o in chosen['owners']))

def clean_screen(metrics):
    if len(metrics)!=3 or any(not nonnegative(m.get(k)) for m in metrics for k in ('F','J')):raise ValueError('clean screen incomplete')
    med={k:statistics.median(m[k] for m in metrics) for k in ('F','J')}
    return dict(medians=med,passed=all(m[k]<=2*med[k] for m in metrics for k in med),engineering_adequacy=False)

def b2(distance,matched,unmatched):
    vals=[distance]+matched+unmatched
    if not matched or not unmatched or any(not nonnegative(v) for v in vals):return dict(status='unavailable',positive=None,margin=None)
    nulls=matched+unmatched; margin=min(nulls)-distance
    return dict(status='complete',positive=margin>0,margin=margin,ties=sum(x==distance for x in nulls),rank=1+sum(x<distance for x in nulls))

def b2_cohort(rows):
    n=len(rows);positive=sum(r.get('positive') is True for r in rows);utility=sum(r.get('utility') is True for r in rows)
    joint=sum(r.get('positive') is True and r.get('utility') is True for r in rows)
    evaded=sum(r.get('positive') is False and r.get('utility') is True for r in rows)
    if n<3:category='insufficient roster for cohort category'
    elif any(r.get('positive') is None or r.get('utility') is None for r in rows):category='unavailable'
    elif positive>=n-1:category='retained'
    elif positive<=1 and utility>=n-1:category='evaded'
    else:category='mixed'
    return dict(n=n,positive=positive,utility=utility,positive_and_utility=joint,evaded_and_utility=evaded,category=category,rows=rows)

def primary_module(path):
    # Reuse the existing tested TPFA32 roster, intervals, qualification and joint rules.
    import importlib.util,sys
    from pathlib import Path
    path=Path(path); saved={n:sys.modules.get(n) for n in ('core','outcomes')}
    try:
        for name in ('core','outcomes','analyze'):
            spec=importlib.util.spec_from_file_location(name,path/(name+'.py'));m=importlib.util.module_from_spec(spec)
            if name!='analyze':sys.modules[name]=m
            spec.loader.exec_module(m)
        return m
    finally:
        for n,old in saved.items():
            if old is None:sys.modules.pop(n,None)
            else:sys.modules[n]=old
