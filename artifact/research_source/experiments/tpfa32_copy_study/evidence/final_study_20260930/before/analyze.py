"""Analyze stored records only. No training, solver or runner imports."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import statistics
from core import read, write, positive, OWNER_COUNT, ANALYSIS_RULE
from outcomes import qualification_status

PRIMARY = ('random_jvp', 'task')


def interval(values, coverage=.975):
    if coverage not in (.975,.95):
        raise ValueError('only the reviewed advantage/equivalence coverages are supported')
    if len(values) != OWNER_COUNT or not all(positive(v) for v in values):
        return dict(status='incomplete', reason='ten finite positive ratios required')
    logs = [math.log(v) for v in values]
    if max(logs) == min(logs):
        return dict(status='degenerate', reason='zero log-ratio variance', ratios=values)
    from scipy.stats import t
    quantile=(1+coverage)/2
    critical = float(t.ppf(quantile, df=OWNER_COUNT-1))
    mean = statistics.mean(logs)
    half = critical*statistics.stdev(logs)/math.sqrt(OWNER_COUNT)
    try:
        result = dict(status='complete', ratios=values, log_ratios=logs,
                      geometric_mean=math.exp(mean), lower=math.exp(mean-half),
                      upper=math.exp(mean+half), critical=critical, quantile=quantile, df=OWNER_COUNT-1,
                      pointwise_coverage=coverage,
                      nominal_joint_coverage_at_least=.95 if coverage==.975 else None,
                      purpose='advantage' if coverage==.975 else 'all-components-pass equivalence',
                      signs=dict(positive=sum(v>1 for v in values),
                                 negative=sum(v<1 for v in values), tied=sum(v==1 for v in values)))
        if not all(positive(result[k]) for k in ('geometric_mean','lower','upper')):
            raise OverflowError('unrepresentable interval')
        return result
    except OverflowError:
        return dict(status='degenerate', reason='unrepresentable exponentiated interval')


def quality_label(intervals, equivalence_intervals):
    if any(group.get(m,{}).get('status') != 'complete'
           for group in (intervals,equivalence_intervals) for m in PRIMARY):
        return 'incomplete/degenerate'
    bounds = [(intervals[m]['lower'], intervals[m]['upper']) for m in PRIMARY]
    if any(l>1.25 for l,u in bounds) and all(l>.8 for l,u in bounds):
        return 'copy advantage'
    if all(equivalence_intervals[m]['lower']>=.8 and
           equivalence_intervals[m]['upper']<=1.25 for m in PRIMARY):
        return 'equivalence'
    if any(u<.8 for l,u in bounds) and all(u<1.25 for l,u in bounds):
        return 'replacement advantage'
    return 'trade-off/inconclusive'


def b2_rank(descendant, clean, replacements):
    values = [descendant]+clean+replacements
    if len(clean)!=OWNER_COUNT-1 or len(replacements)!=OWNER_COUNT or any(
        x is None or not math.isfinite(x) or x<0 for x in values):
        return dict(status='unavailable', reason='complete finite nonnegative 9+10 nulls required')
    nulls=clean+replacements
    margin=min(nulls)-descendant
    return dict(status='complete', descendant=descendant, other_clean=clean,
                selected_replacements=replacements, rank=1+sum(x<descendant for x in nulls),
                ties=sum(x==descendant for x in nulls), margin=margin,
                strictly_nearest=margin>0, calibrated_detector=False)


def analyze(record):
    if record.get('schema') not in ('tpfa32_results.v1','tpfa32_results.v2'): raise ValueError('result schema')
    planned=record['planned_owner_ids']
    if len(planned)!=OWNER_COUNT or len(set(planned))!=OWNER_COUNT:
        raise ValueError('provisional primary rule requires exactly ten planned owner identities')
    n=len(planned)
    found={}
    for row in record.get('owners',[]):
        if row['id'] not in planned or row['id'] in found: raise ValueError('duplicate/unplanned owner')
        found[row['id']]=row
    qualification=qualification_status(record)
    qualification_complete=qualification['status']=='complete'
    table=[]; ratios={m:[] for m in PRIMARY}; primary_errors=[]
    for owner in planned:
        row=found.get(owner,{})
        locked_q=qualification['roster'].get(owner)
        q=locked_q is True
        intact=row.get('intact',{}); attacked=row.get('attacked',{})
        intact_pass=intact.get('detected') is True and intact.get('utility') is True
        removed=q and intact_pass and attacked.get('detected') is False and attacked.get('utility') is True
        retained=q and intact_pass and attacked.get('detected') is True and attacked.get('utility') is True
        pair={}
        for metric in PRIMARY:
            a=row.get('metrics',{}).get('A3',{}).get(metric)
            r=row.get('metrics',{}).get('Rstar',{}).get(metric)
            if not positive(a) or not positive(r):
                pair[metric]=None;primary_errors.append(dict(owner=owner,metric=metric,reason='missing/nonpositive/nonfinite primary error'))
            else:
                ratio=r/a
                if not positive(ratio):
                    pair[metric]=None;primary_errors.append(dict(owner=owner,metric=metric,reason='nonfinite ratio'))
                else: pair[metric]=ratio;ratios[metric].append(ratio)
        table.append(dict(owner=owner, qualification=locked_q,
                          qualification_status=qualification['status'],
                          final_intact_qualified=intact_pass, removal=removed,
                          attacked_detection_utility=retained, ratios=pair,costs=row.get('costs'),
                          observed=owner in found,
                          outcome_status=record.get('owner_outcomes',{}).get(owner,
                              'complete' if owner in found else 'absent')))
    intervals={m:interval(ratios[m]) for m in PRIMARY}
    equivalence_intervals={m:interval(ratios[m],.95) for m in PRIMARY}
    label=quality_label(intervals,equivalence_intervals)
    nq=qualification['N_q']
    removed=sum(r['removal'] for r in table) if qualification_complete else None
    joint={}
    for outcome in ('removal','attacked_detection_utility'):
        qualifying={}
        if label=='copy advantage' and qualification_complete and nq>=n-1:
            for metric in PRIMARY:
                other=PRIMARY[1] if metric==PRIMARY[0] else PRIMARY[0]
                if intervals[metric]['lower']>1.25:
                    qualifying[metric]=[r['owner'] for r in table if r[outcome] and
                        r['ratios'][metric] is not None and r['ratios'][other] is not None and
                        r['ratios'][metric]>1.25 and r['ratios'][other]>=.8]
        joint[outcome]=dict(qualifying_owner_ids=qualifying,
                           passed=any(len(ids)>=nq-1 for ids in qualifying.values())
                                  if qualification_complete and nq>=n-1 else False)
    b2={owner:b2_rank(**record['b2'][owner]) if owner in record.get('b2',{}) else
        dict(status='unavailable') for owner in planned}
    pairs=record.get('null_pairs',[])
    pair_ids={(p.get('reference'),p.get('model')) for p in pairs}
    expected={(i,'O/'+j) for i in planned for j in planned if i!=j}|{(i,'Rstar/'+j) for i in planned for j in planned}
    valid_nulls=(len(pairs)==n*(2*n-1) and pair_ids==expected and all(type(p.get('detected')) is bool for p in pairs))
    secondary={}
    for owner in planned:
        metrics=found.get(owner,{}).get('metrics',{})
        comparisons={}
        for replacement in ('Rstar','R-a@1920','R-b@1920'):
            for descendant in ('A3','A0'):
                comparisons[replacement+'/'+descendant]={}
                for metric in ('forward','random_jvp','task'):
                    numerator=metrics.get(replacement,{}).get(metric)
                    denominator=metrics.get(descendant,{}).get(metric)
                    ratio=numerator/denominator if positive(numerator) and positive(denominator) else None
                    comparisons[replacement+'/'+descendant][metric]=ratio if positive(ratio) else None
        secondary[owner]=dict(metrics=metrics,ratios=comparisons,
                              inferential_status='descriptive; not additional primary tests')
    return dict(schema='tpfa32_analysis.v3',fixture=record.get('fixture',False),
                rule_status='provisional friend amendment; pending final agreement',analysis_rule=ANALYSIS_RULE,
                primary=dict(label=label,intervals=intervals,equivalence_intervals=equivalence_intervals,
                  errors=primary_errors,planned_n=n,complete_case_substitution=False,
                  qualification_filters_primary=False,
                  equivalence_coverage='separate 95% pointwise intervals; all components pass; not 95% simultaneous coverage',
                  assumptions='iid approximately normal owner log ratios conditional on common benchmark/fixed development procedure'),
                joint_table=table,qualification=qualification,N_q=nq,removal_count=removed,
                removal_fraction_qualified=None if not nq else removed/nq,
                removal_fraction_planned=None if removed is None else removed/n,
                removal_endpoint=('unavailable/incomplete qualification evidence' if not qualification_complete
                    else 'insufficiently qualified' if nq<n-1 else 'qualified denominator locked before final'),
                unobserved_final_owners=[r['owner'] for r in table if not r['observed']],
                partial_owner_measurements=record.get('partial_owner_measurements',{}),
                minimum_qualified=n-1,joint_labels=joint,b2=b2,
                b2_strictly_nearest_count=sum(r.get('strictly_nearest',False) for r in b2.values()),
                nulls=dict(status='complete' if valid_nulls else 'incomplete',pairs=pairs,
                           observed_positives=sum(p.get('detected') is True for p in pairs),
                           observed_pairs=len(pairs),expected_pairs=n*(2*n-1),per_key=2*n-1,
                           dependent=True,population_FPR_bound=None),
                costs=record.get('costs'), secondary=secondary,secondary_b2=record.get('secondary',{}),
                limits=['No universal claim about budgets >=64','Failed superiority is not equivalence',
                        'Equal updates are not equal compute','Ten signs are not a power guarantee',
                        'Friend simulation percentages are conditional unverified planning estimates',
                        'Utility calibration uses development/calibration only and is secondary to primary quality'])


def report(record, output):
    result=analyze(record)
    path=Path(output);path.mkdir(parents=True,exist_ok=False)
    write(path/'analysis.json',result)
    lines=['# TPFA32 analysis — '+('SYNTHETIC FIXTURE ONLY' if result['fixture'] else 'stored study observations'),
           '',result['rule_status'],'','Primary quality: '+result['primary']['label'],
           ('Removal: unavailable (qualification evidence incomplete/conflicting).' if result['N_q'] is None
            else 'Removal: '+str(result['removal_count'])+'/'+str(result['N_q'])+' qualified; '+str(result['removal_count'])+'/'+str(result['primary']['planned_n'])+' planned.'),
           'Endpoint: '+result['removal_endpoint'],'',
           '| Owner | Q | Final intact | Removal | R*/A3 JVP | R*/A3 task | Outcome |',
           '|---|---|---|---|---|---|---|']
    for row in result['joint_table']:
        lines.append('| '+ ' | '.join(str(row[k]) for k in ('owner','qualification','final_intact_qualified','removal'))+
                     ' | '+str(row['ratios']['random_jvp'])+' | '+str(row['ratios']['task'])+' | '+row['outcome_status']+' |')
    lines+=['','Intervals, per-owner costs, signs, descriptive B2/null matrices and missing values are in analysis.json.',
            'Two 97.5% pointwise t intervals give nominal at least 95% joint coverage under the stated model.',
            'Equivalence uses separate 95% pointwise intervals and requires both within [0.8,1.25]; this is not 95% simultaneous coverage.',
            'All ten planned owners enter primary quality regardless of qualification. Utility calibration is secondary and uses development/calibration data only.',
            'Friend simulation percentages are conditional unverified planning estimates, not established power.',
            'No independent-binomial FPR bound or experimental verification is implied.']
    (path/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.input is None or args.output is None: parser.print_help()
    else: report(read(args.input),args.output)
