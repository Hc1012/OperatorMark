"""Prospective, independently adopted exception to whole-forecast admission.

No authority is issued here. Only one existing complete clean-owner producer
phase is supported; scoring/selection are still pending and retain their gates.
"""
from operatormark.corpus.state_v2_prerequisites import require_accepted
from .marking import finite, identity
from pathlib import Path
import shutil
import os


def admit(record, session, bindings, clock, *, finalization_only=False):
    v = require_accepted(record, kind='main_study_bounded_stage_amendment', bindings=bindings)
    if (v.get('adopted') is not True or v.get('domain') != 'final'
            or v.get('policy') != 'one_clean_owner_phase_v1'
            or v.get('scientific_scope_unchanged') is not True
            or v.get('partial_forecast_explicit') is not True):
        raise ValueError('separately adopted bounded-stage amendment required')
    p = session.get('bounded_stage', {})
    names = p.get('phases', [])
    if (p.get('amendment_sha256') != record.sha256 or names != v.get('phases')
            or len(names) != 1 or not isinstance(names[0], str)
            or not names[0].startswith('owners/')
            or p.get('resource_plan_sha256') != v.get('resource_plan_sha256')
            or session['limits'] != v.get('cumulative_limits')
            or session.get('previous_session_sha256') != v.get('previous_session_sha256')):
        raise ValueError('bounded stage scope/resource/accounting binding differs')
    identity(p.get('resource_plan_sha256'))
    if v.get('deadline_policy') == 'individual_stages_v1':
        identity(v.get('campaign_sha256'))
        if (session.get('campaign_sha256')!=v['campaign_sha256']
                or session['start']!=session['session']['start']
                or session['deadline']!=session['stage_deadline']):
            raise ValueError('individual immutable stage window/campaign binding differs')
    cap = v.get('maximum_attached_seconds')
    work = v.get('maximum_worker_seconds')
    cost = session.get('node_envelopes', {}).get(names[0])
    if (not finite(cap) or cap <= 60 or not finite(work) or work <= 0
            or not finite(cost) or cost <= 0 or cost > work
            or session['stage_deadline'] - session['session']['start'] > cap
            or session['session']['worker_deadline'] - clock() > work
            or (not finalization_only and clock() + cost > session['session']['worker_deadline'])
            or session['finalization_deadline'] > session['stage_deadline'] - 30
            or session['stage_deadline'] > session['deadline']):
        raise ValueError('complete stage/work/cleanup envelope unavailable')
    # A cap permits a bounded attempt, not an assertion of demonstrated fit.
    if p.get('fit_demonstrated') is not False or not isinstance(p.get('unknowns'), list):
        raise ValueError('stage timing/storage uncertainties must remain explicit')
    return v


def worker_binding(session):
    if session.stage_policy and session.stage_policy.get('deadline_policy')=='individual_stages_v1':
        return dict(configuration_sha256=session.bindings['configuration_sha256'],
                    campaign_sha256=session.stage_policy['campaign_sha256'])
    return dict(configuration_sha256=session.bindings['configuration_sha256'],
                original_start=session.value['start'],original_deadline=session.value['deadline'])


def admitted_deadline(binding, grant):
    """Separate future accepted operation window; old records retain old expiry.

    Call only after the existing kind/binding acceptance interface. A request or
    stored product is never an execution grant. No authority is created here.
    """
    if 'original_deadline' in binding:return binding['original_deadline']
    identity(binding.get('campaign_sha256'))
    w=grant.get('stage_window',{})
    if (grant.get('campaign_sha256')!=binding['campaign_sha256']
            or any(not finite(w.get(k)) for k in ('start','deadline','maximum_seconds'))
            or w['maximum_seconds']<=0 or not 0<w['deadline']-w['start']<=w['maximum_seconds']
            or not finite(grant.get('deadline')) or grant['deadline']>w['deadline']):
        raise ValueError('fresh independently accepted bounded operation window required')
    return w['deadline']


def capacity(record, session, destination, provider):
    """Consume actual startup sizing/admission; no fixed quota becomes proof."""
    v=require_accepted(record,kind='main_study_stage_capacity',bindings=dict(
        configuration_sha256=session.bindings['configuration_sha256'],
        worker_sha256=session.bindings['worker_sha256'],
        amendment_sha256=session.stage_amendment.sha256,
        resource_plan_sha256=session.stage_policy['resource_plan_sha256']))
    if (v.get('domain')!='final' or v.get('normal_stage_bound_complete') is not True
            or v.get('arbitrary_failure_lossless_proved') is not False
            or v.get('unknown_normal_terms')!=[] or not finite(v.get('valid_until'))
            or v['valid_until']<session.value['stage_deadline']):
        raise ValueError('actual first-stage sizing and practical failure admission missing')
    for key in ('input_inventory_sha256','runtime_sizing_sha256','windows_capacity_sha256','drive_capacity_sha256'):
        identity(v.get(key))
    roots=[Path(session.value['worker_root']).parent,Path(destination).parent,
           Path(session.value['launch_root']).parent,Path(session.value['operational_root']).parent]
    volume=Path(v['linux_volume']).resolve()
    if any(not p.resolve().is_relative_to(volume) or os.stat(p).st_dev!=os.stat(volume).st_dev for p in roots):
        raise ValueError('stage copy layout is not the admitted Linux volume')
    for key in ('linux_incremental_required','windows_incremental_required','drive_incremental_required'):
        if type(v.get(key)) is not int or v[key]<=0:raise ValueError('complete first-stage capacity operand missing')
    if (shutil.disk_usage(volume).free<v['linux_incremental_required']
            or v.get('windows_available_bytes',-1)<v['windows_incremental_required']
            or v.get('drive_available_bytes',-1)<v['drive_incremental_required']
            or v.get('drive_destination_id')!=provider['destination_id']):
        raise ValueError('stage destination capacity insufficient or mismatched')
    return v
