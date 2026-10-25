"""Terminal readback only, including abruptly interrupted workers. No model imports."""
from core import read,write


def terminal_summary(root,attempt,status):
    directory=root/('attempt_'+str(attempt))
    authority=read(root/'AUTHORITY.json')
    # One terminal scan after the owned worker ends; never a live recursive monitor.
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if size+1024**2>authority['evidence_bytes']:
        raise RuntimeError('terminal evidence ceiling/reserve exceeded')
    status['evidence_bytes_before_terminal_report']=size
    if status['scientifically_complete']:
        completed=read(root/'COMPLETED.json')
        from core import identity
        for name in ('results','analysis','seeds'):
            if identity(completed[name]['path'])!=completed[name]:raise ValueError('terminal result identity mismatch')
        result=read(root/'RESULTS.json');analysis=read(root/'analysis/analysis.json')
        names=[f'owner{i}' for i in range(10)]
        if (not completed['passed'] or result['planned_owner_ids']!=names or
            [r['id'] for r in result['owners']]!=names or not result['evaluation_complete'] or
            analysis['qualification']['status']!='complete' or analysis['nulls']['status']!='complete'):
            raise ValueError('terminal complete-cohort/qualification/null evidence missing')
        if any(completed['counts'][k]>v for k,v in authority['limits'].items()):raise ValueError('terminal quota mismatch')
        return
    from analyze import report
    from outcomes import qualification_packet
    names=[f'owner{i}' for i in range(10)]
    result=dict(schema='tpfa32_results.v2',fixture=read(root/'CONFIG.json').get('fixture_only',False),
        planned_owner_ids=names,owners=[],b2={},null_pairs=[],secondary={},costs={},evaluation_complete=False,
        rule_status='approved final milestone 20260930',analysis_rule='t10_adv975_eq95',owner_outcomes={},partial_owner_measurements={})
    if (root/'QUALIFICATION.json').exists():result['locked_qualification']=qualification_packet(root/'QUALIFICATION.json')
    for name in names:
        path=root/('OWNER_'+name+'.json')
        if path.exists():result['owners'].append(read(path));result['owner_outcomes'][name]='complete'
        else:result['owner_outcomes'][name]='absent/interrupted'
        result['partial_owner_measurements'][name]=[read(p) for p in sorted(root.glob('FINAL_'+name+'_*.json'))]
    result['costs']=dict(supervisor=status,accounting='See immutable progress journal; unsettled block bound is not refunded.')
    write(directory/'RESULTS_INCOMPLETE.json',result)
    report(result,directory/'analysis_incomplete')
