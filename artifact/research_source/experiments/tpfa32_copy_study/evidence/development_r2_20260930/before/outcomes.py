"""Locked qualification and additive final-owner summaries; no numerical imports."""
import hashlib
import json
from pathlib import Path
from core import read, write, identity, digest


def qualification_packet(path, expected=None):
    path=Path(path);fact=identity(path)
    if expected is not None and fact['sha256']!=expected['sha256']:
        raise ValueError('qualification changed after training/calibration closure')
    return dict(identity=fact,content=path.read_bytes().decode('utf-8'))


def qualification_status(record):
    """Validate the exact independent file bytes, roster and observed-row agreement."""
    try:
        packet=record['locked_qualification'];fact=packet['identity']
        raw=packet['content'].encode('utf-8')
        if hashlib.sha256(raw).hexdigest()!=fact['sha256'] or len(raw)!=fact['bytes']:
            raise ValueError('qualification identity/content mismatch')
        document=json.loads(packet['content']);planned=record['planned_owner_ids']
        if document.get('locked_before_final') is not True:
            raise ValueError('qualification was not locked before final')
        if document.get('planned_owner_ids')!=planned:
            raise ValueError('qualification planned identities differ')
        roster={}
        for row in document['owners']:
            name=row['owner']
            if name in roster or name not in planned or type(row.get('qualified')) is not bool:
                raise ValueError('duplicate/unplanned/incomplete qualification row')
            roster[name]=row['qualified']
        if set(roster)!=set(planned) or len(roster)!=len(planned):
            raise ValueError('incomplete locked qualification roster')
        nq=sum(roster.values())
        if type(document.get('N_q')) is not int or document['N_q']!=nq:
            raise ValueError('locked qualification count disagrees with roster')
        for row in record.get('owners',[]):
            if 'qualified' in row and (type(row['qualified']) is not bool or
                                      row['qualified']!=roster.get(row['id'])):
                raise ValueError('per-owner qualification conflicts with locked roster: '+row['id'])
        return dict(status='complete',N_q=nq,roster=roster,identity=fact,errors=[])
    except (KeyError,TypeError,ValueError,AttributeError) as exc:
        return dict(status='unavailable/incomplete',N_q=None,roster={},errors=[str(exc)])


def start_final_records(root,names,packet,fixture=False):
    header=dict(schema='tpfa32_results.v2',fixture=fixture,planned_owner_ids=names,
                locked_qualification=packet,owners=[],b2={},null_pairs=[],secondary={},costs={})
    status=qualification_status(header)
    if status['status']!='complete':raise ValueError(status['errors'])
    write(Path(root)/'FINAL_REPORT_INPUT.json',header)
    (Path(root)/'final_owners').mkdir(exist_ok=False)
    return header


def owner_record(root,name,kind,row):
    root=Path(root);names=read(root/'FINAL_REPORT_INPUT.json')['planned_owner_ids']
    ordinal=names.index(name)
    directory=root/'final_owners'
    if kind=='partial':
        sequence=len(list(directory.glob(f'{ordinal:02d}_partial_*.json')))
        filename=f'{ordinal:02d}_partial_{sequence:03d}.json'
    else:filename=f'{ordinal:02d}_{kind}.json'
    write(directory/filename,dict(owner=name,**row))


def retained_results(root):
    """Read-only reporting recovery, never execution or owner replacement."""
    root=Path(root);result=read(root/'FINAL_REPORT_INPUT.json')
    directory=root/'final_owners';outcomes={};partials={}
    for ordinal,name in enumerate(result['planned_owner_ids']):
        completed=directory/f'{ordinal:02d}_completed.json'
        failed=directory/f'{ordinal:02d}_failed.json'
        started=directory/f'{ordinal:02d}_started.json'
        if completed.exists():
            row=read(completed)
            if row['id']!=name or row['owner']!=name:raise ValueError('retained owner identity mismatch')
            result['owners'].append(row);outcomes[name]='complete'
        elif failed.exists():outcomes[name]='failed'
        elif started.exists():outcomes[name]='interrupted/unobserved'
        else:outcomes[name]='absent'
        partials[name]=[read(p) for p in sorted(directory.glob(f'{ordinal:02d}_partial_*.json'))]
    result['owner_outcomes']=outcomes;result['partial_owner_measurements']=partials
    result['evaluation_complete']=False
    return result


def incomplete_report(root):
    root=Path(root);result=retained_results(root)
    write(root/'RESULTS_INCOMPLETE.json',result)
    from analyze import report
    return report(result,root/'analysis_incomplete')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from analyze import report
    report(retained_results(args.root),args.output)
