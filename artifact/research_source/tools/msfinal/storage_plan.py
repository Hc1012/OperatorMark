"""Copy-graph accounting: explicit unknowns, never a storage-plan approval."""
DOMAINS=(
    'sources_runtime_inputs', 'production_data_and_independent_readback',
    'owner_baseline_replacement_models_optimizer_checkpoints_and_partials',
    'marking_attack_overwrite_models_and_partials', 'training_journals_and_monitors',
    'forward_jvp_inverse_evaluations_designs_and_native_records',
    'D6_CPU_scoring_transfer_return_and_completed_views',
    'noise_draws_including_partials', 'noise_headers_and_record_metadata',
    'query_projection_and_wrapper_caches', 'calibration_keys_nulls_and_B2',
    'API_stream_records', 'qualification_and_corpus_State_custody',
    'supervision_runtime_sessions_closure_and_acceptances',
    'historical_and_new_failures', 'transport_logs_and_failed_upload_payloads')

ARCHIVE_LAYOUT={
    'windows_retained':('phase_payloads','zip_headers','original_receipts_and_recovery_catalogs',
                        'unique_failures_not_in_phase_payloads','session_and_batch_archives'),
    'windows_transient':('input_assets_not_already_present','dependency_subset_payload_and_zip',
                         'CPU_batch_input_return_work_and_archives','partial_transfers_not_in_retained',
                         'headroom'),
    'drive_staging':('inflight_phase_archives','session_metadata_exports','dependency_transfers',
                     'CPU_batch_inputs_returns','unarchived_failures','recovery_metadata','headroom'),
    'linux_scratch':('runtime_and_inputs','active_phase_payload_and_archive','dependency_subset_payload_and_zip',
                     'pending_CPU_inputs_returns','session_metadata_exports','unarchived_failures','headroom')}

def archive_layout(terms,capacities):
    """One permanent phase copy; exact disjoint terms, unknowns forbid admission.

    Already retained history is reflected in observed free capacity, not silently
    deleted. The caller must supply a lossless writer/member bound for every term.
    No compression or unperformed reclamation credit is implicit in this helper.
    """
    if set(terms)!=set(ARCHIVE_LAYOUT):raise ValueError('all archive layout destinations required')
    totals={};known={};unknown=[]
    for volume,names in ARCHIVE_LAYOUT.items():
        if set(terms[volume])!=set(names):raise ValueError('complete copy layout required: '+volume)
        for name,value in terms[volume].items():
            if value is None:unknown.append(volume+'.'+name)
            elif type(value) is not int or value<0:raise ValueError('nonnegative writer bound required')
        known[volume]=sum(v for v in terms[volume].values() if v is not None)
        totals[volume]=None if any(v is None for v in terms[volume].values()) else known[volume]
    peaks=dict(windows=None if None in (totals['windows_retained'],totals['windows_transient']) else
               totals['windows_retained']+totals['windows_transient'],drive=totals['drive_staging'],linux=totals['linux_scratch'])
    subtotals=dict(windows=known['windows_retained']+known['windows_transient'],drive=known['drive_staging'],linux=known['linux_scratch'])
    if set(capacities)!=set(peaks):raise ValueError('current capacity observations required')
    for value in capacities.values():
        if value is not None and (type(value) is not int or value<0):raise ValueError('invalid capacity')
    fits={k:None if peaks[k] is None or capacities[k] is None else peaks[k]<=capacities[k] for k in peaks}
    return dict(terms=terms,unknowns=unknown,retained_bytes=totals['windows_retained'],peaks=peaks,
        calculated_subtotals=subtotals,capacity=capacities,fits=fits,
        subtotal_shortfalls={k:None if capacities[k] is None else max(0,subtotals[k]-capacities[k]) for k in peaks},
        admitted=all(v is True for v in fits.values()),compression_credit=0,actual_reclamation=False)

def account(domains,*,phase_archives,remote_records,retained_restore_archives,
            active_restore_payload,local_inputs,local_failures,remote_failures,
            duplicate_archives,remote_available=None,local_available=None):
    """Byte upper bounds from actual writers/approved policies or None.

    Canonical phase payload is retained once in its ZIP remotely. A source copy
    remains wherever generated: NO implicit deletion credit. Local working-set
    inputs include all simultaneously needed prior-phase files, not just new work.
    Arguments describing local/remote copies are additional, disjoint terms.
    """
    if set(domains)!=set(DOMAINS):raise ValueError('complete programme storage domains required')
    terms=dict(domains)
    terms.update(phase_archives=phase_archives,remote_records=remote_records,
        retained_restore_archives=retained_restore_archives,active_restore_payload=active_restore_payload,
        local_inputs=local_inputs,local_failures=local_failures,remote_failures=remote_failures,
        duplicate_archives=duplicate_archives)
    for name,value in terms.items():
        if value is not None and (type(value) is not int or value<0):raise ValueError('nonnegative explicit byte bound: '+name)
    unknown=sorted(k for k,v in terms.items() if v is None)
    source=sum(domains.values()) if all(v is not None for v in domains.values()) else None
    def total(names):return None if any(terms[k] is None for k in names) else sum(terms[k] for k in names)
    remote=total(('phase_archives','remote_records','remote_failures','duplicate_archives'))
    local=total(('retained_restore_archives','active_restore_payload','local_inputs','local_failures'))
    def fits(required,available):
        if available is not None and (type(available) is not int or available<0):raise ValueError('capacity observation required')
        return None if required is None or available is None else required<=available
    return dict(domains=domains,terms=terms,unknowns=unknown,complete_bound=not unknown,
        canonical_source_upper=source,additional_remote_upper=remote,local_restore_peak_upper=local,
        remote_fits=fits(remote,remote_available),local_fits=fits(local,local_available),
        automatic_eviction=False,compression_credit=0,programme_ready=False,
        warning='Retained originating source copies and pre-existing usage are additional; quota is not a reservation. '
                'Session projection must include the union of prior phases visited by Worker.run/readback, '
                'archive copies, all current dependencies and retained failed restores. '
                'A 2 TB subscription alone proves neither capacity nor fit.')
