"""Consume independently issued premise-P evidence; never issue it locally."""
from dataclasses import dataclass
from pathlib import PureWindowsPath
from . import state_execution_authority as a


@dataclass(frozen=True)
class CustodyAdmission:
    acceptance_file: dict
    validators: tuple
    acquisitions: tuple
    binding: dict
    first_utc: str
    last_utc: str


def require_custody(value, *, source_sha256, roots=None, at_utc=None):
    a.need(type(value) is CustodyAdmission,'independently accepted premise P required')
    a.utc(value.first_utc);a.utc(value.last_utc)
    a.need(value.first_utc<value.last_utc,'custody interval')
    if at_utc is not None:
        a.utc(at_utc);a.need(value.first_utc<=at_utc<=value.last_utc,'custody interval expired/not started')
    data=a.qualify(value.acceptance_file,value.acquisitions,validators=value.validators,
        scope='state6_continuous_exclusive_custody',first=value.first_utc,last=value.last_utc)
    row=a.decode(data)
    a.object_keys(row,'schema accepted synthetic binding interval roots operating_basis evidence_files residuals')
    a.need(row['schema']=='state6_exclusive_custody.v1' and row['accepted'] is True and
        row['synthetic'] is False and row['binding']==value.binding and
        row['binding'].get('source_inventory_sha256')==source_sha256,'custody acceptance/source scope')
    a.need(row['interval']==[value.first_utc,value.last_utc],'custody interval substitution')
    a.need(type(row['roots']) is list and len(row['roots'])==2,'both exact roots required')
    a.need({x['role'] for x in row['roots']}=={'payload','evidence'},'custody root roles')
    for root in row['roots']:
        a.object_keys(root,'role canonical_paths anchor_path anchor_native_identity acceptance_file')
        a.need(type(root['canonical_paths']) is list and len(root['canonical_paths'])==2,
            'exact staging and published paths required')
        anchor=PureWindowsPath(root['anchor_path'])
        a.need(anchor.is_absolute(),'absolute admitted NTFS anchor required')
        paths=[PureWindowsPath(x) for x in root['canonical_paths']]
        a.need(len(set(paths))==2 and all(x!=anchor and x.is_relative_to(anchor) for x in paths),
            'custody locations must be distinct descendants of the admitted anchor')
        from .publication_physical_records import _identity
        _identity(root['anchor_native_identity']);a.ref(root['acceptance_file'])
        admitted=a.decode(a.qualify(root['acceptance_file'],value.acquisitions,validators=value.validators,
            scope='state6_custody_root_acceptance',first=value.first_utc,last=value.last_utc))
        a.need(admitted.get('accepted') is True and admitted.get('synthetic') is False and
            admitted.get('binding')==value.binding and admitted.get('interval')==row['interval'] and
            admitted.get('root')=={k:root[k] for k in ('role','canonical_paths','anchor_path','anchor_native_identity')},
            'independent per-root P admission mismatch')
    allpaths=[PureWindowsPath(p) for r in row['roots'] for p in r['canonical_paths']]
    a.need(len(set(allpaths))==4 and all(not x.is_relative_to(y) for x in allpaths for y in allpaths if x!=y),
        'distinct nonoverlapping custody roots')
    if roots is not None:
        a.need(set(roots)=={'payload','evidence'} and all(roots[x['role']] in x['canonical_paths'] for x in row['roots']),
            'actual custody roots differ')
    a.object_keys(row['operating_basis'],'dedicated_account local_ntfs operator_only_dacl no_other_account_processes indexing_av_residual_disposition sole_writer_handoffs')
    for name in ('dedicated_account','local_ntfs','operator_only_dacl','no_other_account_processes','sole_writer_handoffs'):
        a.need(row['operating_basis'][name] is True,'unestablished custody operating basis: '+name)
    a.need(row['operating_basis']['indexing_av_residual_disposition'] in ('excluded','independently_accepted_residual'),
        'indexing/AV condition unavailable')
    a.refs(row['evidence_files'],True);a.need(bool(row['evidence_files']),'actual custody evidence absent')
    for f in row['evidence_files']:
        a.qualify(f,value.acquisitions,validators=value.validators,scope='state6_custody_operating_evidence',
            first=value.first_utc,last=value.last_utc)
    a.need(type(row['residuals']) is list and set(row['residuals'])==
        {'administrator','SYSTEM','backup_privilege','malware'},'explicit residual acceptance required')
    return row


def recheck_live_custody(value,*,source_sha256,roots):
    """Point-check the already admitted anchors, not continuous exclusivity.

    Anchors exist before fresh State 5 destinations. Exact future staging and
    publication subpaths are admitted; child identities are observed separately
    by the existing physical phase verifier after creation and after promotion.
    """
    from datetime import datetime,timezone
    from .state_host_windows import windows_file_observation
    from .state_provider_bindings import wire
    now=datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
    row=require_custody(value,source_sha256=source_sha256,roots=roots,at_utc=now)
    for root in row['roots']:
        actual=windows_file_observation(root['anchor_path'])
        a.need(actual['presence']=='present' and actual['directory'] and
            wire(actual['identity'])==root['anchor_native_identity'],'admitted custody anchor drift/unavailable')
    return row
