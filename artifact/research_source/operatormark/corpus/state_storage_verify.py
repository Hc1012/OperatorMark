"""Independent later-phase comparison; no candidate conclusions confer authority."""
from . import state_provider_bindings as p
from . import state_storage_records as r


def verify_later_storage_boundary(*, expected, current, previous, acquisitions, provider_acquisitions):
    r.validate_later_storage_expectations(expected)
    p.require(current is None or type(current) is bytes, 'current immutable bytes')
    p.require(previous is None or type(previous) is bytes, 'previous immutable bytes')
    p.require(type(provider_acquisitions) is tuple, 'provider acquisition inventory')
    diagnostics = []; checked = []; established_locations = {}; established_contexts = {}

    def add(kind, path, reason):
        diagnostics.append((kind, path, reason))

    groups = {}
    for pair in provider_acquisitions:
        if type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str or type(pair[1]) is not tuple:
            add('invalid', 'providers', 'provider_acquisition_structure'); continue
        groups.setdefault(pair[0], []).append(pair[1])
    provider_sets = [('current', expected.providers), ('previous', expected.prior_providers)]
    allowed_provider_keys = {cut + '/' + x.provider_role for cut, values in provider_sets for x in values}
    for name in groups:
        if name not in allowed_provider_keys:
            add('mismatch', name, 'undeclared_provider')
    for cut, provider in ((cut, x) for cut, values in provider_sets for x in values):
        name = cut + '/' + provider.provider_role; rows = groups.get(name, [])
        if len(rows) != 1:
            add('mismatch' if rows else 'incomplete', name, 'provider_inventory'); continue
        result = p.qualify_provider_acquisition(expected=provider, acquisitions=rows[0])
        checked.extend('provider/' + name + '/' + x for x in result.checked)
        diagnostics.extend((kind, 'provider/' + name + '/' + path, reason) for kind, path, reason in result.diagnostics)
    qualified = p.qualify_admissions(expected.admissions, acquisitions)
    diagnostics.extend(qualified.diagnostics)
    checked.extend('admission/' + x for x in qualified.checked)
    acquired = {x.role: x for x in acquisitions if type(x) is p.ProviderAcquisition and x.role in qualified.checked}

    def evidence(reference, path, capture, fact=None, binding=None):
        if reference is None:
            add('incomplete', path, 'evidence_unavailable'); return False
        f = p.file_ref_from(reference)
        matches = [a for a in expected.admissions if a.file == f and a.role in qualified.checked and p.within(acquired[a.role].capture, capture)]
        if len(matches) != 1:
            add('incomplete' if not matches else 'mismatch', path, 'evidence_unqualified'); return False
        if fact is not None:
            try:
                actual = p.decode(acquired[matches[0].role].data)
                p.keys(actual, 'schema binding capture kind subject value')
                p.require(actual['schema'] == r.PROFILE + '.fact' and actual['binding'] == p.wire(binding) and
                    p.within(p.capture_from(actual['capture']), capture) and
                    (actual['kind'], actual['subject'], actual['value']) == fact, 'fact binding')
            except (p.BindingError, TypeError, KeyError):
                add('mismatch', path, 'fact_binding'); return False
        checked.append(path)
        return True

    def observation(data, f, role, phase, binding, capture, providers, context):
        if data is None:
            add('incomplete', role, 'observation_unavailable'); return None
        if f is None:
            add('mismatch', role, 'undeclared_observation'); return None
        if (len(data), p.sha(data)) != (f.file_bytes, f.file_sha256):
            add('mismatch', role, 'observation_bytes')
        if role not in qualified.checked or acquired[role].data != data:
            add('incomplete', role, 'observation_unqualified')
        try:
            record = r.strict_later_storage_loads(data)
        except (p.BindingError, TypeError, KeyError) :
            add('invalid', role, 'observation_invalid'); return None
        if record['binding'] != p.wire(binding):
            add('mismatch', role, 'observation_binding')
        if record['phase'] != phase:
            add('mismatch', role, 'observation_phase')
        actual_capture = p.capture_from(record['capture'])
        if actual_capture != capture:
            add('mismatch', role, 'observation_cutoff')
        if record['roots'] != p.wire(expected.roots):
            add('mismatch', role, 'resolved_root_binding')
        for root in record['roots']:
            evidence(root['evidence_file'], role + '/root/' + root['root_binding_id'], capture)
        location_ok = evidence(record['location_file'], role + '/location', capture,
            ('location', binding.attempt_id, record['location']), binding)
        context_ok = True
        if record['context'] != p.wire(context):
            add('mismatch', role, 'termination_context_binding'); context_ok = False
        context_value = {k: p.wire(getattr(context, k)) for k in ('outcome', 'terminating_state', 'origin_phase', 'failure_files')}
        context_ok = evidence(p.wire(context.origin_file), role + '/context', context.origin_capture,
            ('termination_context', binding.attempt_id, context_value), context.origin_binding) and context_ok
        for index, failure in enumerate(context.failure_files):
            context_ok = evidence(p.wire(failure), role + '/context/failure/' + str(index), context.origin_capture,
                ('failure', binding.attempt_id, True), context.origin_binding) and context_ok
        if context.outcome == 'failed':
            add('incomplete', role, 'retained_failure_context')
            if not context_ok: add('incomplete', role, 'quarantine_failure_unavailable')
        if context.outcome == 'unavailable':
            add('incomplete', role, 'termination_context_unavailable')
        if context_ok: established_contexts[role] = context
        directory = record['attempt_directory']; directory_ok = True
        directory_path = None
        if directory is None:
            add('incomplete', role, 'directory_unavailable'); directory_ok = False
        else:
            if directory['native_identity'] is None:
                add('incomplete', role, 'directory_identity_unavailable'); directory_ok = False
            elif directory['native_identity'] != expected.attempt_directory.native_identity:
                add('mismatch', role, 'directory_identity'); directory_ok = False
            if directory['custody_file'] is None:
                add('incomplete', role, 'directory_custody_unavailable'); directory_ok = False
            elif directory['custody_file'] != p.wire(expected.attempt_directory.custody_file):
                add('mismatch', role, 'directory_custody'); directory_ok = False
            directory_ok = evidence(directory['custody_file'], role + '/directory/custody', capture) and directory_ok
            places = [x for x in expected.attempt_directory.locations if x.root_binding_id == directory['root_binding_id']]
            if not places:
                add('incomplete', role, 'directory_location_unavailable'); directory_ok = False
            else:
                place = places[0]; directory_path = place.relative_path
                anchor = next(x for x in expected.roots if x.root_binding_id == place.root_binding_id)
                if (directory['root_binding_id'], directory['relative_path'], directory['resolved_path']) != (
                    record['location'], place.relative_path, anchor.canonical_path + '/' + place.relative_path):
                    add('mismatch', role, 'directory_resolved_location'); directory_ok = False
            directory_ok = evidence(directory['evidence_file'], role + '/directory/fact', capture,
                ('attempt_directory', binding.attempt_id, {k: directory[k] for k in (
                    'native_identity', 'custody_file', 'root_binding_id', 'relative_path', 'resolved_path')}), binding) and directory_ok
        scope_ok = (role in qualified.checked and acquired[role].data == data and
            record['binding'] == p.wire(binding) and record['phase'] == phase and actual_capture == capture and
            record['roots'] == p.wire(expected.roots))
        if directory_ok and location_ok and scope_ok and record['location'] != 'unknown':
            established_locations[role] = record['location']
            checked.append(role + '/directory/continuity_binding')
        if record['location'] == 'unknown':
            add('incomplete', role + '/location', 'location_unknown')
        if phase in p.PHASES[:6] and record['location'] not in ('T', 'unknown'):
            add('mismatch', role, 'phase_location')
        if (phase in ('post_move', 'post_audit', 'post_complete') or
            phase in ('cleanup', 'terminal') and context.outcome == 'succeeded') and record['location'] not in ('F', 'unknown'):
            add('mismatch', role, 'phase_location')
        if phase == 'post_quarantine' and record['location'] not in ('Q', 'unknown'):
            add('mismatch', role, 'phase_location')
        by_slot = {x['slot']: x for x in record['objects']}
        if set(by_slot) != {x.slot for x in expected.outputs}:
            add('mismatch', role, 'output_inventory')
        for wanted in expected.outputs:
            path = role + '/object/' + wanted.slot
            row = by_slot.get(wanted.slot)
            if row is None:
                add('incomplete', path, 'object_unavailable'); continue
            if row['relative_path'] != wanted.relative_path:
                add('mismatch', path, 'object_address')
            if record['location'] != 'unknown' and row['root_binding_id'] != record['location']:
                add('mismatch', path, 'object_location')
            if row['presence'] == 'unknown':
                add('incomplete', path, 'presence_unknown')
            presence_ok = evidence(row['presence_file'], path + '/presence', capture,
                ('presence', wanted.slot, {k: row[k] for k in ('presence', 'native_identity', 'root_binding_id', 'relative_path')}), binding)
            is_forward = phase in p.PHASES[:9]
            required = p.PHASES.index(wanted.introduced_phase) <= p.PHASES.index(phase) if is_forward else None
            if not is_forward and context.origin_phase in p.PHASES[:9] and context.outcome in ('failed', 'succeeded'):
                required = p.PHASES.index(wanted.introduced_phase) <= p.PHASES.index(context.origin_phase)
            if required is not None and row['presence'] != 'unknown' and row['presence'] != ('present' if required else 'absent'):
                add('mismatch', path, 'phase_presence')
            if row['presence'] == 'absent' and not presence_ok:
                add('incomplete', path, 'absence_unqualified')
            if row['presence'] == 'present':
                if row['native_identity'] is None:
                    add('incomplete', path, 'identity_unavailable')
                elif row['native_identity'] != wanted.native_identity:
                    add('mismatch', path, 'object_identity')
                if row['custody_file'] != p.wire(wanted.custody_file):
                    add('incomplete' if row['custody_file'] is None else 'mismatch', path, 'object_custody')
                evidence(row['custody_file'], path + '/custody', capture)
                evidence(row['stable_file'], path + '/stable', capture,
                    ('stable', wanted.slot, {k: row[k] for k in ('native_identity', 'root_binding_id', 'relative_path', 'content_file')}), binding)
                content = row['content_file']
                if content is None:
                    add('incomplete', path, 'content_unavailable')
                else:
                    actual = p.file_ref_from(content)
                    if directory_path is None:
                        add('incomplete', path, 'member_parent_unavailable')
                    elif (actual.root_binding_id, actual.relative_path) != (row['root_binding_id'], directory_path + '/' + row['relative_path']):
                        add('mismatch', path, 'content_address')
                    if (actual.file_bytes, actual.file_sha256) != (wanted.content_file.file_bytes, wanted.content_file.file_sha256):
                        add('mismatch', path, 'content_identity')
                    evidence(content, path + '/content', capture)
                if presence_ok and location_ok and row['native_identity'] is not None:
                    checked.append(path + '/identified')
        for name in ('failure_files', 'cleanup_failure_files'):
            for i, f in enumerate(record[name]):
                if f in p.wire(context.failure_files):
                    continue  # Already qualified against its retained original cutoff.
                evidence(f, role + '/' + name + '/' + str(i), capture,
                    ('failure' if name == 'failure_files' else 'cleanup_failure', binding.attempt_id, True), binding)
            if record[name]:
                add('incomplete', role, 'retained_' + name)
        op = record['operation']
        phase_action = {'post_move': 'promote', 'post_quarantine': 'quarantine', 'cleanup': 'cleanup'}.get(phase)
        if phase_action is not None and op['action'] != phase_action:
            add('mismatch', role, 'required_operation')
        if op['action'] in ('promote', 'quarantine', 'cleanup') and op['action'] != phase_action:
            add('mismatch', role, 'unexpected_operation')
        if op['action'] == 'metadata' and phase not in ('pre_move', 'post_audit', 'post_complete'):
            add('mismatch', role, 'unexpected_metadata_operation')
        if op['action'] != 'none':
            evidence(op['evidence_file'], role + '/operation', capture,
                ('operation', binding.attempt_id, {k: op[k] for k in ('action', 'call_status', 'effect_status')}), binding)
            evidence(op['authority_file'], role + '/operation_authority', capture)
            if op['authority_file'] != p.wire(providers[2].authority_file):
                add('incomplete' if op['authority_file'] is None else 'mismatch', role, 'operation_authority_binding')
            if op['effect_status'] != 'complete' or op['call_status'] != 'returned_success':
                add('incomplete', role, 'operation_unresolved')
        first, last, cleanup = (getattr(providers[0], n) for n in ('valid_from', 'valid_until', 'cleanup_until'))
        limit = cleanup if phase in ('failure_location', 'post_quarantine', 'cleanup', 'terminal') else last
        if not first <= actual_capture.started_utc <= actual_capture.finished_utc <= limit:
            add('mismatch', role, 'authority_interval')
        return record

    before = observation(previous, expected.previous_file, 'previous_observation', expected.previous_phase,
                         expected.prior_binding, expected.prior_capture, expected.prior_providers, expected.prior_context) if expected.previous_file is not None else None
    if expected.previous_file is None and previous is not None:
        add('mismatch', 'previous_observation', 'undeclared_observation')
    after = observation(current, expected.current_file, 'current_observation', expected.phase, expected.binding, expected.capture, expected.providers, expected.context)
    if expected.previous_file is not None and before is None:
        add('incomplete', 'transition', 'predecessor_unavailable')
    if after is not None:
        action = after['operation']['action']
        wanted_action = {'post_move': 'promote', 'post_quarantine': 'quarantine', 'cleanup': 'cleanup'}.get(expected.phase)
        if wanted_action is not None and action != wanted_action:
            add('mismatch', 'transition', 'required_operation')
        if action in ('promote', 'quarantine', 'cleanup') and action != wanted_action:
            add('mismatch', 'transition', 'unexpected_operation')
        if action == 'metadata' and expected.phase not in ('pre_move', 'post_audit', 'post_complete'):
            add('mismatch', 'transition', 'unexpected_metadata_operation')
        if expected.phase == 'post_quarantine':
            if established_locations.get('previous_observation') not in ('T', 'F'):
                add('incomplete', 'transition', 'quarantine_source_unestablished')
            if established_contexts.get('previous_observation') is None or expected.prior_context.outcome != 'failed':
                add('incomplete', 'transition', 'quarantine_failure_unavailable')
        if expected.phase == 'terminal' and expected.context.outcome == 'succeeded':
            if expected.prior_context is None or expected.prior_context.outcome != 'succeeded':
                add('mismatch', 'transition', 'success_context_unestablished')
        if before is not None:
            before_context = r.context_from(before['context']); after_context = r.context_from(after['context'])
            if before_context.outcome in ('failed', 'succeeded', 'unavailable') and before_context != after_context:
                add('mismatch', 'transition', 'termination_context_changed')
            before_directory, after_directory = before['attempt_directory'], after['attempt_directory']
            if before_directory is not None and after_directory is not None:
                for key in ('native_identity', 'custody_file'):
                    if before_directory[key] is not None and after_directory[key] is not None and before_directory[key] != after_directory[key]:
                        add('mismatch', 'transition', 'directory_continuity_' + key)
            old_failures = {p.encode(f) for f in before['failure_files'] + before['cleanup_failure_files'] + p.wire(before_context.failure_files)}
            new_failures = {p.encode(f) for f in after['failure_files'] + after['cleanup_failure_files'] + p.wire(after_context.failure_files)}
            if not old_failures <= new_failures:
                add('mismatch', 'transition', 'failure_evidence_omitted')
            old = {x['slot']: x for x in before['objects']}
            for row in after['objects']:
                peer = old.get(row['slot'])
                if peer is not None and peer['presence'] == row['presence'] == 'present':
                    for key in ('native_identity', 'custody_file'):
                        if peer[key] is not None and row[key] is not None and peer[key] != row[key]:
                            add('mismatch', row['slot'], 'continuity_' + key)
                elif peer is not None and peer['presence'] == 'present' and row['presence'] == 'absent':
                    add('mismatch', row['slot'], 'object_disappeared')
                elif peer is not None and peer['presence'] == 'absent' and row['presence'] == 'present':
                    wanted = next((x for x in expected.outputs if x.slot == row['slot']), None)
                    if wanted is None or wanted.introduced_phase != expected.phase or action != 'metadata':
                        add('mismatch', row['slot'], 'undeclared_addition')
            if before['location'] != 'unknown' and after['location'] != 'unknown':
                if action == 'promote' and (before['location'], after['location']) != ('T', 'F'):
                    add('mismatch', 'transition', 'promotion_location')
                if action == 'quarantine' and (before['location'] not in ('T', 'F') or after['location'] != 'Q'):
                    add('mismatch', 'transition', 'quarantine_location')
                if action not in ('promote', 'quarantine') and before['location'] != after['location']:
                    add('mismatch', 'transition', 'unexplained_movement')
    status = 'mismatch' if any(x[0] in ('mismatch', 'invalid') for x in diagnostics) else 'incomplete' if diagnostics else 'corroborated'
    report = dict(schema=r.REPORT, profile=r.PROFILE, binding=p.wire(expected.binding), phase=expected.phase,
        current_file=p.wire(expected.current_file), previous_file=p.wire(expected.previous_file),
        expectations_sha256=p.sha(p.encode(expected)), current_bytes_sha256=p.sha(current) if current is not None else None,
        previous_bytes_sha256=p.sha(previous) if previous is not None else None,
        status=status, checked=checked, diagnostics=[dict(kind=k, path=x, reason=y) for k, x, y in diagnostics],
        location=established_locations.get('current_observation', 'unknown'),
        termination_outcome=established_contexts['current_observation'].outcome if 'current_observation' in established_contexts else 'unavailable',
        terminating_state=established_contexts['current_observation'].terminating_state if 'current_observation' in established_contexts else None,
        free_space_bytes=after['free_space_bytes'] if after else None, requirements_known=False, capacity_sufficient=None,
        production_accepted=False, execution_authorized=False)
    report['report_self_sha256'] = p.sha(p.encode(report))
    return report


def validate_later_storage_report(report, *, expected, current, previous, acquisitions, provider_acquisitions):
    """Full recomputation, including coordinated conclusion/self-hash changes."""
    actual = verify_later_storage_boundary(expected=expected, current=current, previous=previous,
        acquisitions=acquisitions, provider_acquisitions=provider_acquisitions)
    p.require(p.encode(report) == p.encode(actual), 'report differs from independent recomputation')
    return actual
