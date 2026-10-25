"""Saved-result analysis bindings, preserving fixed rules and missing owners."""
from pathlib import Path
from operatormark.corpus.state_v2_prerequisites import require_accepted
from . import materials as A, marking as M


def primary_quality(admission,paired_record):
    """Reuse the frozen ten-owner interval implementation; no N=16 assumption."""
    A.require_opening(admission);p=admission.prerequisites
    record=require_accepted(paired_record,kind='main_study_final_paired_quality',
        bindings=dict(study_id=p.study_id,configuration_sha256=p.configuration.sha256,opening_sha256=p.opening.sha256))
    planned=admission.configuration['owners']
    if len(planned)!=10:raise ValueError('only the existing ten-owner inferential implementation is supported')
    if record.get('domain')!='final':raise ValueError('development comparisons cannot enter final inference')
    rows=record.get('owners',[]);found={r['owner_id']:r for r in rows}
    if len(found)!=len(rows) or not set(found)<=set(planned):raise ValueError('duplicate/unplanned primary owner')
    ratios={'random_jvp':[],'task':[]};errors=[]
    for owner in planned:
        row=found.get(owner,{})
        for key,metric in (('random_jvp','J'),('task','T')):
            copy_error=row.get('attacked_B3',{}).get(metric);replacement=row.get('replacement',{}).get(metric)
            if (not M.finite(copy_error) or copy_error<=0 or not M.finite(replacement) or replacement<=0
                    or not M.finite(replacement/copy_error)):
                errors.append(dict(owner_id=owner,metric=metric,reason='missing/nonpositive/nonfinite measurement'))
            else:ratios[key].append(replacement/copy_error)
    from msdev.rules import primary_module
    root=Path(p.source_root)/'experiments/tpfa32_copy_study'
    for name in ('analyze.py','core.py','outcomes.py'):
        key='experiments/tpfa32_copy_study/'+name
        if admission.configuration['source_pins'].get(key)!=M.digest(root/name):raise ValueError('primary analysis source binding missing/drifted')
    frozen=primary_module(root)
    advantage={k:frozen.interval(v,.975) for k,v in ratios.items()}
    equivalence={k:frozen.interval(v,.95) for k,v in ratios.items()}
    return dict(planned_owners=planned,errors=errors,qualification_filters_primary=False,
        advantage=advantage,equivalence=equivalence,label=frozen.quality_label(advantage,equivalence),
        equivalence_coverage='separate 95% pointwise; all-components pass, not simultaneous 95%',
        interpretation='conditional owner log-ratio model; no physical or population-FPR claim')


def b2_readout(owner,distances,matched_ids,independent_ids):
    """Missing a null is unavailable, never a smaller easier comparison set."""
    from msdev.rules import b2
    all_ids=[owner]+list(matched_ids)+list(independent_ids)
    if len(all_ids)!=len(set(all_ids)):raise ValueError('overlapping/duplicate B2 comparison identities')
    value=b2(distances.get(owner),[distances.get(k) for k in matched_ids],[distances.get(k) for k in independent_ids])
    value.update(matched_ids=list(matched_ids),independent_ids=list(independent_ids),calibrated_population_FPR=False)
    return value


def cohort(planned,qualified,checkpoints,primary):
    """All checkpoints can defeat survival; missing evidence gets no success."""
    if not planned or len(planned)!=len(set(planned)) or not set(qualified)<=set(planned) or len(qualified)!=len(set(qualified)):
        raise ValueError('invalid planned/qualified roster')
    if primary not in checkpoints or any(set(rows)!=set(qualified) for rows in checkpoints.values()):
        return dict(category='unavailable',reason='missing/conflicting locked-roster observations',N=len(planned),N_q=len(qualified))
    n=len(planned);nq=len(qualified);counts={}
    for name,rows in checkpoints.items():
        counts[name]=dict(removal=sum(r.get('qualified_removal') is True for r in rows.values()),
                         retained=sum(r.get('retained_detection') is True for r in rows.values()),
                         tasks_unmeasured=sum(r.get('task_status')=='unmeasured' for r in rows.values()),
                         missing=sum(r.get('qualified_removal') is None or r.get('retained_detection') is None for r in rows.values()))
    c=counts[primary]
    if nq<n-1:category='insufficient qualified marks'
    elif c['removal']>=nq-1:category='removed'
    elif (c['retained']>=nq-1 and all(r['removal']==0 and r['missing']==0 for r in counts.values())):category='survives within tested bounds'
    else:category='mixed'
    return dict(category=category,N=n,N_q=nq,counts=counts,
                removed_fraction_qualified=None if not nq else c['removal']/nq,removed_fraction_planned=c['removal']/n)


def joint(primary,planned,qualified,outcomes,owner_ratios):
    """Separate removal/retention headline conditions; missing is not success."""
    if len(planned)!=10 or len(set(planned))!=10 or len(qualified)!=len(set(qualified)) or not set(qualified)<=set(planned):
        raise ValueError('reviewed ten-owner roster required')
    result={}
    for outcome in ('qualified_removal','retained_detection'):
        by_metric={}
        if primary['label']=='copy advantage' and len(qualified)>=9:
            for metric,other in (('J','T'),('T','J')):
                key='random_jvp' if metric=='J' else 'task'
                if primary['advantage'][key]['lower']>1.25:
                    by_metric[metric]=[o for o in qualified if outcomes.get(o,{}).get(outcome) is True
                        and M.finite(owner_ratios.get(o,{}).get(metric)) and owner_ratios[o][metric]>1.25
                        and M.finite(owner_ratios[o].get(other)) and owner_ratios[o][other]>=.8]
        result[outcome]=dict(owners=by_metric,passed=bool(by_metric) and any(len(v)>=len(qualified)-1 for v in by_metric.values()))
    return result


def assembled_readouts(graph,per_owner,primary,measurements):
    """Fixed source-schedule endpoints, never a best observed checkpoint.

    This reconciles implemented source families only. Programme coverage/adoption
    remains a separate false readiness gate in the graph.
    """
    planned=graph['planned_owners']
    if set(per_owner)!=set(planned):raise ValueError('incomplete planned-owner readout')
    qualified=[o for o in planned if per_owner[o]['qualification'] is True]
    groups={}
    for models in graph['models'].values():
        for name,d in models.items():
            if d.get('selected_contract') and d.get('source_parent')=='M3':group=d['selected_contract']+'_M3'
            elif d.get('source_parent')=='M3':group=d['procedure']['variant']+'_M3'
            elif name.startswith(('Aord_M3_','A4_M3_','A5B3_M3_')):group=name.rsplit('_',1)[0]
            elif name.startswith('W') and 'M3' in name:group=name
            else:continue
            end=next((n['end'] for n in graph['trajectories'].get(d['trajectory'],[]) if n['id']==d['checkpoint']),320)
            groups.setdefault(group,{})[name]=end
    for job in graph.get('contract_jobs',[]):
        if job['arm']=='noise' and job['parent']=='M3':
            groups.setdefault(job['variant']+'_M3',{})[job['label']]=0
    summaries={};missing={}
    for group,names in groups.items():
        # The prescribed largest endpoint is fixed from the source graph before
        # inspecting values; intermediate checkpoints remain in survival tests.
        primary_name=max(names,key=lambda name:(names[name],name))
        rows={name:{o:per_owner[o]['removal'][name] for o in qualified if name in per_owner[o]['removal']} for name in names}
        summaries[group]=dict(primary=primary_name,readout=cohort(planned,qualified,rows,primary_name))
        missing[group]={name:[o for o in qualified if o not in row] for name,row in rows.items() if len(row)!=len(qualified)}
    ratios={};outcomes={}
    for o in planned:
        copy=measurements.get(o,{}).get('Aord_M3_raw1920',{}).get('metrics',{})
        replacement=measurements.get(o,{}).get('Rstar',{}).get('metrics',{})
        ratios[o]={k:replacement[k]/copy[k] if M.finite(copy.get(k)) and copy[k]>0 and M.finite(replacement.get(k)) else None for k in ('J','T')}
        outcomes[o]=per_owner[o]['removal'].get('Aord_M3_raw1920',{})
    headline=(joint(primary,planned,qualified,outcomes,ratios) if len(planned)==10 and 'label' in primary
              else dict(available=False,reason='complete ten-owner primary readout required'))
    return dict(planned_owners=planned,qualified_owners=qualified,cohorts=summaries,missing=missing,
        omitted_measurements=graph.get('contract_omitted',[]),
        noise_claim_scope='additive white-box pathwise only, with same-test intact prerequisites',
        API_evidence_scope='observed service responses only; derivative/task/detector unavailable',
        ordinary_P2_joint=headline,complete_programme=False,source_scope_only=True)
