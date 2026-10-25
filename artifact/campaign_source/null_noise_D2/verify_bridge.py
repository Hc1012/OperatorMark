"""Separate readback and scoped admission for one combined D2 attempt (R3).

R3 reuses the completed twenty-model bridge of failed attempt 20261023T183105Z.
Its preserved records live under reused/<attempt>/ exactly as fetched from that
attempt's pinned full return; nothing there is rewritten. Historical pins are
checked against historical bytes, this attempt's code against this attempt's
pins, and every scientific input is pinned identically in both. No models are
constructed or evaluated here. No clock, retry, broad runtime acceptance or
canonical-Worker authority is created by either phase.
"""
from pathlib import Path, PurePosixPath
from runtime_values import plain_runtime
import argparse
import hashlib
import importlib
import json
import math
import os
import stat
import sys
import time
import zipfile

from compatibility_compare import (exact, compare_one, compare_projection_pair,
                                   projection_key, sha)

POLICY = 'D2_COMBINED_POLICY.json'
BRIDGE_POLICY = 'D2_PYTHON31316_COMPAT_POLICY.json'
SCOPE = 'D2_raw960_diagnostics_only'
SCHEMA = 'OperatorMark.D2-scoped-runtime-admission.v2-reused-bridge'
VERIFICATION_SCHEMA = 'OperatorMark.D2-bridge-verification.v2-reused-bridge'
OUT = 'compatibility_output'
PROGRESS = dict(phase='start')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def local(job, name):
    need(isinstance(name, str), 'Evidence member must be a relative string')
    rel = PurePosixPath(name)
    need(bool(rel.parts) and rel.as_posix() == name and not rel.is_absolute()
         and '..' not in rel.parts and ':' not in name and '\\' not in name,
         'Unsafe evidence member: ' + name)
    path = job.joinpath(*rel.parts)
    need(path.resolve().is_relative_to(job.resolve()) and not path.is_symlink(),
         'Evidence escapes the attempt: ' + name)
    return path


def fact(path, check=lambda: None):
    need(path.is_file() and not path.is_symlink(), 'Missing regular evidence: ' + str(path))
    before = path.stat()
    value = dict(bytes=before.st_size, sha256=sha(path, check))
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino)
         == (after.st_size, after.st_mtime_ns, after.st_ino), 'Evidence changed while hashing')
    return value


def facts(record):
    return {k: record[k] for k in ('bytes', 'sha256')} if isinstance(record, dict) else None


def files_match(job, files, check=lambda: None):
    need(isinstance(files, dict) and files, 'Evidence inventory is empty')
    for name, expected in files.items():
        check()
        exact(fact(local(job, name), check), expected, 'Evidence/' + name)


def activation(job, policy_sha):
    a = read(job/'ACTIVATION.json')
    need(a.get('policy_sha256') == policy_sha and a.get('stage_seconds') == 30600
         and type(a.get('start')) in (int, float) and type(a.get('deadline')) in (int, float)
         and math.isfinite(a['start']) and math.isfinite(a['deadline'])
         and a['deadline'] - a['start'] == 30600
         and a.get('previous_clocks_reused') is False and a.get('retries') == 0,
         'The original fixed combined-stage activation is required')
    return a


def reused(job, p):
    """The reused attempt's preserved root and its frozen identities from this policy."""
    r = p.get('reused_bridge')
    need(isinstance(r, dict) and isinstance(r.get('attempt'), str) and r.get('root') == 'reused/' + r['attempt']
         and r.get('bridge_policy_sha256') == p['bridge_policy_sha256'], 'Reused bridge contract missing')
    root = local(job, r['root'])
    need(root.is_dir(), 'Reused attempt evidence was not fetched: ' + r['root'])
    return r, root


def reused_activation(job, p, check=lambda: None):
    r, root = reused(job, p)
    need(sha(root/'ACTIVATION.json', check) == r['activation_sha256']
         and sha(root/'FIRST_FAILURE.json', check) == r['first_failure_sha256']
         and sha(root/POLICY, check) == r['combined_policy_sha256']
         and sha(root/BRIDGE_POLICY, check) == r['bridge_policy_sha256'] == p['bridge_policy_sha256'],
         'Reused attempt identity differs')
    failure = read(root/'FIRST_FAILURE.json')
    need(failure.get('phase') == 'independent compatibility readback' and failure.get('retry') is False
         and failure.get('raw960_diagnostics_started') is False and failure.get('compatibility_comparisons_started') is True,
         'Reused attempt failure scope differs')
    return r, root, activation(root, r['combined_policy_sha256'])


def policies(job, policy_sha, check=lambda: None):
    need(sha(job/POLICY, check) == policy_sha, 'Combined policy changed')
    p = read(job/POLICY)
    need(p.get('scoped_admission_required') is True
         and p.get('compatibility_authority_scope') == SCOPE,
         'Prospective scoped mechanical-admission contract missing')
    need(sha(job/BRIDGE_POLICY, check) == p['bridge_policy_sha256'], 'Bridge policy changed')
    bp = read(job/BRIDGE_POLICY)
    exact(bp['source_pins'], p['source_pins'], 'Bridge/campaign scientific source pins')
    need(len(p['source_pins']) == 338, 'Frozen scientific source population differs')
    exact(bp['expected_runtime'], p['expected_runtime'], 'Bridge/campaign candidate runtime')
    need(bp['candidate_runtime_payload_sha256'] == p['candidate_runtime_payload_sha256'],
         'Candidate runtime payload binding differs')
    order = p['model_order']
    need(len(order) == len(set(order)) == 200
         and [row['id'] for row in p['model_jobs']] == order,
         'Exact 200-model continuation roster required')
    for owner in ['O%02d' % i for i in range(1, 11)]:
        need(sum(row['owner'] == owner for row in p['model_jobs']) == 20,
             'Continuation owner population differs')
    for row in p['model_jobs']:
        need(row['id'] == row['owner'] + '/' + row['name'] and row['name'].endswith('_raw960')
             and row['descriptor']['wrapper_steps'] is None
             and row['descriptor']['procedure']['transform'] is None,
             'Continuation exceeds frozen raw960 diagnostic scope')
    from compatibility_producer import FAMILIES
    expected = ['O01/' + family + '_' + parent + '_raw320'
                for family in FAMILIES for parent in ('M3', 'O')]
    exact(bp['model_order'], expected, 'Bridge roster')
    need([row['id'] for row in bp['model_jobs']] == expected, 'Bridge job roster differs')
    activation(job, policy_sha)
    reused(job, p)
    return p, bp


def pin_partition(job, p, bp, check=lambda: None):
    """Historical pins on historical bytes; new orchestration on new pins; science unchanged."""
    PROGRESS['phase'] = 'historical/new-code pin partition'
    r, root = reused(job, p)
    superseded = r.get('superseded_orchestration')
    need(isinstance(superseded, dict) and superseded, 'Superseded orchestration declaration missing')
    differing = {k for k, v in bp['files'].items() if facts(p['files'].get(k)) != facts(v)}
    need(set(superseded) == differing, 'Declared superseded orchestration differs from the actual pin difference: '
         + ', '.join(sorted(set(superseded) ^ differing)))
    for name, row in superseded.items():
        need('/' not in name and name.endswith('.py') and name not in bp['source_pins']
             and name not in p['source_pins'], 'Only top-level orchestration scripts may be superseded: ' + name)
        need(facts(row.get('historical')) == facts(bp['files'][name]) and facts(row.get('current')) == facts(p['files'][name])
             and row['historical']['sha256'] != row['current']['sha256'], 'Superseded pin record differs: ' + name)
    for group in ('original_files', 'marked_files', 'final_material_files', 'reference_files'):
        need({k: facts(v) for k, v in bp.get(group, {}).items()} == {k: facts(v) for k, v in p.get(group, {}).items()},
             'Historical scientific input group changed: ' + group)
    need(bp.get('aliases', {}) == p.get('aliases', {}), 'Historical alias inputs changed')
    for role, group in bp['additional_sources'].items():
        need(p['additional_sources'].get(role) == group, 'Historical input source changed: ' + role)
    need(set(p['additional_sources']) == set(bp['additional_sources']) | {r['source_group']},
         'Only the reused-bridge source may be added')
    import compatibility_producer as C
    # 1. every historical pin except the superseded scripts, on this attempt's bytes
    C.verify(job, dict(bp, files={k: v for k, v in bp['files'].items() if k not in superseded}), check)
    # 2. the superseded scripts' historical pins, on the reused attempt's preserved copies
    C.verify(root, dict(files={k: bp['files'][k] for k in superseded}, source_pins={}), check)
    # 3. this attempt's replacements, on this attempt's pins
    for name in sorted(superseded):
        exact(fact(local(job, name), check), facts(p['files'][name]), 'New orchestration/' + name)
    # 4. the reused attempt's fetched members, on their pinned bytes
    C.verify(job, dict(files=p['additional_sources'][r['source_group']]['files'], source_pins={}), check)
    return dict(superseded=sorted(superseded), historical_pins_checked_on=r['root'],
                unchanged_historical_files=len(bp['files']) - len(superseded),
                scientific_input_groups_identical=True, source_pins=len(p['source_pins']))


def reused_observation_identity(job, p, check=lambda: None):
    PROGRESS['phase'] = 'byte identity with reused runtime observations'
    r, root = reused(job, p)
    differing = []
    for name in r['identical_observations']:
        check()
        if fact(local(job, name), check) != fact(local(root, name), check):
            differing.append(name)
    need(not differing, 'This session differs from reused attempt in: ' + ', '.join(differing))
    need(sha(job/'RUNTIME_GATE.json', check) == r['runtime_gate_sha256'], 'Runtime gate identity differs')
    return list(r['identical_observations'])


def command_value(command, flag):
    need(command.count(flag) == 1, 'Missing or duplicate supervised argument: ' + flag)
    i = command.index(flag)
    need(i + 1 < len(command), 'Supervised argument has no value: ' + flag)
    return command[i + 1]


def closed(job, name, policy_sha, bridge_sha, check=lambda: None, *, record_root=None, a=None):
    """Read captured streams as well as the supervisor's process-closure record."""
    a = activation(job, policy_sha) if a is None else a
    root = job/'supervision'/name if record_root is None else record_root
    r = read(root/'RESULT.json')
    c = r.get('closure', {})
    need(r.get('returncode') == 0 and r.get('errors') == [] and r.get('timed_out') is False
         and r.get('stream_overflow') is False and r.get('stdout_closed') is True
         and r.get('leader_exit_before_streams_eof') is False and r.get('termination') is None
         and c.get('resolved') is True and c.get('child_reaped') is True
         and c.get('returncode') == 0 and c.get('errors') == [] and c.get('survivors') == [],
         'Unresolved/unsuccessful supervised phase: ' + name)
    need(set(r.get('streams', {})) == {'stdout', 'stderr'}, 'Complete captured streams required')
    for stream_name, stream in r['streams'].items():
        check()
        f = fact(root/(stream_name + '.bin'), check)
        rb = stream.get('readback', {})
        need(stream.get('capture_complete') is True and stream.get('eof_observed') is True
             and stream.get('overflow') is False and stream.get('write_failed') is False
             and stream.get('discarded_observed_bytes') == 0
             and stream.get('observed_count_exact') is True
             and stream.get('observed_bytes') == stream.get('retained_bytes') == f['bytes']
             and stream.get('observed_sha256') == stream.get('retained_sha256') == f['sha256']
             and rb.get('readback_verified') is True and rb.get('retained_bytes') == f['bytes']
             and rb.get('retained_sha256') == f['sha256'],
             'Supervised stream readback differs: ' + name + '/' + stream_name)
    command = r.get('command')
    need(isinstance(command, list) and all(isinstance(v, str) for v in command)
         and command_value(command, '--job') == str(job), 'Supervised job binding differs')
    expected_script = 'compatibility_producer.py' if name == 'bridge' else (
        'combined_remote.py' if name == 'bridge_save' else 'verify_bridge.py')
    need(command.count(str(job/expected_script)) == 1, 'Supervised implementation differs')
    if name == 'bridge_save':
        need(command.count('bridge-save') == 1, 'Bridge publisher action differs')
    else:
        need(command_value(command, '--policy-sha') == (bridge_sha if name == 'bridge' else policy_sha),
             'Supervised policy binding differs')
        if name != 'bridge':
            need(command.count('verify' if name == 'bridge_verify' else 'admit') == 1,
                 'Supervised verifier action differs')
    end = float(command_value(command, '--deadline'))
    need(type(r.get('started')) in (int, float) and type(r.get('seconds')) in (int, float)
         and math.isfinite(r['started']) and math.isfinite(r['seconds']) and r['seconds'] >= 0
         and a['start'] <= r['started'] <= end <= a['deadline']
         and r['started'] + r['seconds'] <= a['deadline'], 'Phase escaped original stage clock')
    return r


def bridge_closed(job, p, check=lambda: None):
    """The reused attempt's completed bridge process, checked against that attempt's own clock."""
    PROGRESS['phase'] = 'reused bridge process closure'
    r, root, a = reused_activation(job, p, check)
    return closed(job, 'bridge', r['combined_policy_sha256'], p['bridge_policy_sha256'], check,
                  record_root=root/'supervision/bridge', a=a)


def bridge_paths(job, p):
    r, root = reused(job, p)
    return dict(out=root/OUT, closure=root/'supervision/bridge/RESULT.json',
                libraries_file=r['root'] + '/' + OUT + '/LIBRARIES_FINAL.json',
                folders=[r['root'] + '/' + OUT, r['root'] + '/supervision/bridge', r['root'] + '/observations/runtime'],
                root_files=[r['root'] + '/' + n for n in ('ACTIVATION.json', 'FIRST_FAILURE.json', 'RUNTIME_GATE.json',
                                                          POLICY, BRIDGE_POLICY, *sorted(r['superseded_orchestration']))])


def reused_summary(p):
    r = p['reused_bridge']
    return dict(attempt=r['attempt'], root=r['root'], activation_sha256=r['activation_sha256'],
                first_failure_sha256=r['first_failure_sha256'], combined_policy_sha256=r['combined_policy_sha256'],
                runtime_gate_sha256=r['runtime_gate_sha256'], full_return_sha256=r['full_return']['sha256'],
                identical_observations=list(r['identical_observations']))


def runtime_check(job, p, bp, libraries_file, check=lambda: None):
    """Fresh observation plus hash-bound bridge premises; no work or acceptance."""
    PROGRESS['phase'] = 'fresh runtime, interpreter and library observation'
    from q6_common import THREAD_ENV
    os.environ.update(THREAD_ENV)
    import q5_model as Q
    from msfinal import materials as A
    from msdev import pn_runtime_migration as P
    # Match the bridge's frozen transport dependency preload.  Import only:
    # no transport object, authentication, request or background work is made.
    from msfinal.drive_transport import GoogleDriveTransfer, RefreshingMetadataToken
    Q.configure()
    exact(plain_runtime(A.runtime_identity()), p['expected_runtime'], 'Fresh verifier runtime')
    captured = read(job/'observations/runtime/CANDIDATE_INTERPRETER.json')
    fresh = P.observed_interpreter_identity(check)
    P.target_build(fresh)
    exact(fresh, captured, 'Fresh interpreter binary/stdlib identity')
    need(fresh['python'] == p['expected_runtime']['python'], 'Frozen candidate Python differs')
    for name, expected in bp['expected_modules'].items():
        check()
        module = importlib.import_module(name)
        need(getattr(module, '__file__', None) and sha(module.__file__, check) == expected,
             'Imported numerical source differs: ' + name)
    libraries = read(local(job, libraries_file))
    need(isinstance(libraries, dict) and libraries, 'Bridge library evidence missing')
    P.check_libraries(libraries, check)
    observed = P.libraries(check)
    need(set(observed) <= set(libraries),
         'A loaded library was not observed in the completed bridge: '
         + ', '.join(sorted(set(observed) - set(libraries))))
    for name in observed:
        need(observed[name] == libraries[name], 'Fresh loaded library differs: ' + name)
    return dict(runtime=p['expected_runtime'], interpreter=fresh, libraries=libraries,
                observed_libraries=observed)


def inventory(job, names, folders, check=lambda: None):
    result = {}
    for name in names:
        result[name] = fact(local(job, name), check)
    for name in folders:
        root = local(job, name)
        need(root.is_dir(), 'Evidence directory missing: ' + name)
        for path in sorted(root.rglob('*')):
            check()
            need(not path.is_symlink(), 'Evidence symlink refused')
            if path.is_file():
                result[path.relative_to(job).as_posix()] = fact(path, check)
    return result


def projection_members(root, keys, check=lambda: None):
    """Validate all three artifacts per exclusive completed projection write."""
    root = Path(root)
    need(len(keys) == len(set(keys)) == 21, 'Bridge projection key population differs')
    expected = {key + suffix for key in keys for suffix in ('.pt', '.json', '.pt.publication.json')}
    need({path.name for path in root.iterdir()} == expected,
         'Bridge projection members missing, additional or partial')
    for key in keys:
        check()
        tensor = fact(root/(key + '.pt'), check)
        exact(read(root/(key + '.json')), dict(identity=key, sha256=tensor['sha256']),
              'Projection cache metadata')
        exact(read(root/(key + '.pt.publication.json')),
              dict(schema='D5.exclusive-hardlink.v1', name=key + '.pt',
                   sha256=tensor['sha256'], bytes=tensor['bytes'], exclusive=True,
                   completed_tensor_files=1), 'Projection exclusive publication')
    return dict(projections=21, files=len(expected))


def verify(job, policy_sha, check=lambda: None):
    p, bp = policies(job, policy_sha, check)
    bsha = p['bridge_policy_sha256']
    paths = bridge_paths(job, p)
    bridge_closed(job, p, check)
    partition = pin_partition(job, p, bp, check)
    import compatibility_producer as C
    PROGRESS['phase'] = 'historical O01 admission and specification'
    from msfinal import backend as B, materials as A, evaluation as E
    historical = B.admit(C.prerequisites(job, bp, 'O01'))
    A.preflight(historical)
    spec = E.preflight(historical, 'final')
    candidate = C.context(historical, bp, bsha)
    out = paths['out']
    need(not (out/'FAILURE.json').exists(), 'Bridge failure evidence forbids admission')
    identical = reused_observation_identity(job, p, check)
    observed = runtime_check(job, p, bp, paths['libraries_file'], check)
    exact(read(out/'RUNTIME_ACTUAL.json'), p['expected_runtime'], 'Bridge producer runtime')
    exact(read(out/'INTERPRETER_START.json'), observed['interpreter'], 'Bridge initial interpreter')
    exact(read(out/'INTERPRETER_FINAL.json'), observed['interpreter'], 'Bridge final interpreter')
    from msdev import pn_runtime_migration as P
    P.check_libraries(read(out/'LIBRARIES_START.json'), check)
    completed, projections = {}, {}
    for row in bp['model_jobs']:
        check()
        name = row['name']
        PROGRESS['phase'] = 'comparison ' + row['id'] + ' (' + str(len(completed) + 1) + '/20)'
        old_binding = E.evaluation_binding(historical, 'final', row['measurement_sha256'], None)
        new_binding = E.evaluation_binding(candidate, 'final', row['measurement_sha256'], None)
        proof = compare_one(job/'reference/output/diagnostics/O01'/name/'evaluation',
                            out/'diagnostics/O01'/name/'evaluation', spec,
                            old_binding, new_binding, check)
        projection = compare_projection_pair(job/'reference/output/projections/O01',
            out/'projections/O01', projection_key(historical, row['measurement_sha256'], None, A),
            projection_key(candidate, row['measurement_sha256'], None, C.HistoricalAccess), check)
        record = dict(descriptor=row['descriptor'], measurement_sha256=row['measurement_sha256'],
                      historical_binding=old_binding, candidate_binding=new_binding,
                      diagnostics=proof, projection=projection, scientific_acceptance=False)
        exact(read(out/'diagnostics/O01'/name/'EXACT_COMPARISON.json'), record, 'Per-model comparison')
        completed[row['id']] = record
        projections[row['id']] = projection
    PROGRESS['phase'] = 'clean reference projection and record readback'
    clean = historical.prerequisites.clean_model_sha256
    projections['O01/clean_reference'] = compare_projection_pair(job/'reference/output/projections/O01',
        out/'projections/O01', projection_key(historical, clean, None, A),
        projection_key(candidate, clean, None, C.HistoricalAccess), check)
    projection_members(out/'projections/O01', [row['candidate']['identity']
                       for row in projections.values()], check)
    result, comparison = read(out/'RESULT.json'), read(out/'EXACT_COMPARISON.json')
    for key in ('complete', 'compatibility_evidence_complete', 'exact_comparison_passed'):
        need(result.get(key) is True, 'Bridge completion field missing: ' + key)
    need(result.get('policy_sha256') == bsha and result.get('projection_count') == 21
         and result.get('completed_campaign_jobs_added') == 0
         and result.get('candidate_runtime_payload_sha256') == p['candidate_runtime_payload_sha256']
         and result.get('canonical_worker_advanced') is False
         and result.get('final_scientific_acceptance') is False
         and result.get('scientific_acceptance') is False,
         'Bridge output scope differs')
    exact(result['counts'], bp['operation_limits'], 'Bridge operation counts')
    exact(result['jobs'], completed, 'Bridge result complete roster')
    exact(result['historical_admission_bindings'], historical.bindings, 'Historical admission bindings')
    need(comparison.get('complete') is True and comparison.get('exact') is True
         and comparison.get('policy_sha256') == bsha, 'Bridge comparison summary incomplete')
    exact(comparison['jobs'], completed, 'Bridge comparison complete roster')
    exact(comparison['projections'], projections, 'Bridge complete projection population')
    exact(comparison['reference_archive'], bp['reference_source']['reference'], 'Reference archive binding')
    PROGRESS['phase'] = 'evidence inventory'
    names = [POLICY, BRIDGE_POLICY, 'ACTIVATION.json', 'RUNTIME_GATE.json'] + paths['root_files']
    evidence = inventory(job, names, paths['folders'] + ['observations/runtime'], check)
    for name in bp['reference_files']:
        evidence[name] = fact(local(job, name), check)
    record = dict(schema=VERIFICATION_SCHEMA, passed=True,
        policy_sha256=policy_sha, bridge_policy_sha256=bsha,
        verifier_source_sha256=sha(Path(__file__), check),
        bridge_evidence_sha256=sha(out/'EXACT_COMPARISON.json', check),
        bridge_result_sha256=sha(out/'RESULT.json', check),
        closure_record_sha256=sha(paths['closure'], check),
        candidate_runtime_payload_sha256=p['candidate_runtime_payload_sha256'],
        observed_interpreter_sha256=sha(job/'observations/runtime/CANDIDATE_INTERPRETER.json', check),
        libraries_file=paths['libraries_file'], libraries_sha256=sha(out/'LIBRARIES_FINAL.json', check),
        model_order=bp['model_order'], model_order_sha256=canonical_hash(bp['model_order']),
        compared_models=20, compared_projections=21, exact=True, zero_tolerance=True,
        historical_admission_bindings=historical.bindings, evidence_files=evidence,
        reused_bridge=reused_summary(p), identical_observations=identical, pin_partition=partition,
        new_compatibility_computations=0, scope=SCOPE, scientific_acceptance=False, general_runtime_acceptance=False,
        formal_custody_assurance=False, canonical_worker_advanced=False,
        observed_at=time.time())
    put(job/'BRIDGE_VERIFICATION.json', record)
    return record


def verification_record(job, p, policy_sha, check=lambda: None):
    paths = bridge_paths(job, p)
    v = read(job/'BRIDGE_VERIFICATION.json')
    need(v.get('schema') == VERIFICATION_SCHEMA and v.get('passed') is True
         and v.get('policy_sha256') == policy_sha and v.get('bridge_policy_sha256') == p['bridge_policy_sha256']
         and v.get('scope') == SCOPE and v.get('compared_models') == 20 and v.get('compared_projections') == 21
         and v.get('exact') is True and v.get('zero_tolerance') is True
         and v.get('scientific_acceptance') is False and v.get('general_runtime_acceptance') is False
         and v.get('reused_bridge') == reused_summary(p)
         and v.get('verifier_source_sha256') == sha(Path(__file__), check), 'Scoped bridge verification differs')
    files_match(job, v['evidence_files'], check)
    need(v['bridge_evidence_sha256'] == sha(paths['out']/'EXACT_COMPARISON.json', check)
         and v['closure_record_sha256'] == sha(paths['closure'], check)
         and v['observed_interpreter_sha256'] == sha(job/'observations/runtime/CANDIDATE_INTERPRETER.json', check)
         and v['libraries_file'] == paths['libraries_file']
         and v['libraries_sha256'] == sha(job/v['libraries_file'], check), 'Verification proof binding differs')
    return v


def durable_bridge(job, p, policy_sha, verification, check=lambda: None):
    d = read(job/'BRIDGE_DURABLE.json')
    need(d.get('combined_policy_sha256') == policy_sha
         and d.get('bridge_policy_sha256') == p['bridge_policy_sha256']
         and d.get('bridge_verification_sha256') == sha(job/'BRIDGE_VERIFICATION.json', check)
         and d.get('readback_verified') is True, 'Bridge durable receipt binding differs')
    ref = d['reference']
    need(ref.get('keepForever') is True and ref.get('folder_id') == p['drive_folder']
         and isinstance(ref.get('object_id'), str) and bool(ref['object_id'])
         and isinstance(ref.get('version'), str) and bool(ref['version'])
         and ref.get('bytes') == d['bytes'] and ref.get('sha256') == d['sha256'],
         'Pinned bridge remote revision differs')
    archive = local(job, d['local_archive'])
    exact(fact(archive, check), dict(bytes=d['bytes'], sha256=d['sha256']), 'Durable bridge archive')
    receipt = local(job, d['transport_readback_receipt'])
    need(sha(receipt, check) == d['transport_readback_sha256'], 'Fresh remote readback receipt changed')
    observed = read(receipt)
    need(observed.get('kind') == 'readback' and observed.get('object_id') == ref['object_id']
         and observed.get('revision') == ref['version'] and observed.get('bytes') == ref['bytes']
         and observed.get('sha256') == ref['sha256'], 'Actual authenticated revision readback missing')
    expected = dict(verification['evidence_files'])
    expected['BRIDGE_VERIFICATION.json'] = fact(job/'BRIDGE_VERIFICATION.json', check)
    expected.update(inventory(job, [], ['supervision/bridge_verify'], check))
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        need(len(names) == len(set(names)), 'Duplicate bridge return members')
        for info in z.infolist():
            local(job, info.filename)
            need(not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16) and not info.flag_bits & 1,
                 'Unsafe bridge return member')
        manifest_raw = z.read('RETURN_MANIFEST.json')
        need(hashlib.sha256(manifest_raw).hexdigest() == d['manifest_sha256'], 'Bridge manifest identity differs')
        manifest = json.loads(manifest_raw)['files']
        need(set(names) == set(manifest) | {'RETURN_MANIFEST.json'}, 'Bridge return manifest membership differs')
        need(set(expected) <= set(manifest), 'Bridge return omits accepted evidence')
        for name, expected_fact in expected.items():
            check()
            exact(manifest[name], expected_fact, 'Published evidence/' + name)
        for name, info in manifest.items():
            check()
            member = z.getinfo(name)
            need(member.file_size == info['bytes'], 'Published member size differs')
            h = hashlib.sha256()
            with z.open(member) as stream:
                while block := stream.read(4 * 1024**2):
                    check()
                    h.update(block)
            need(h.hexdigest() == info['sha256'], 'Published member hash differs: ' + name)
    return d


def roster_binding(p):
    return {row['id']: dict(descriptor=row['descriptor'], measurement_sha256=row['measurement_sha256'],
                            model_path=row['model_path'], acceptance=row['acceptance'])
            for row in p['model_jobs']}


def admit(job, policy_sha, check=lambda: None):
    p, bp = policies(job, policy_sha, check)
    bsha = p['bridge_policy_sha256']
    paths = bridge_paths(job, p)
    bridge_closed(job, p, check)
    PROGRESS['phase'] = 'this attempt verifier and publisher closures'
    closed(job, 'bridge_verify', policy_sha, bsha, check)
    closed(job, 'bridge_save', policy_sha, bsha, check)
    v = verification_record(job, p, policy_sha, check)
    PROGRESS['phase'] = 'durable bridge readback'
    durable = durable_bridge(job, p, policy_sha, v, check)
    reused_observation_identity(job, p, check)
    runtime_check(job, p, bp, v['libraries_file'], check)
    PROGRESS['phase'] = 'complete frozen input and package readback'
    import compatibility_producer as C
    C.verify(job, p, check)
    proof_files = dict(v['evidence_files'])
    proof_files.update(inventory(job, ['BRIDGE_VERIFICATION.json', 'BRIDGE_DURABLE.json',
        durable['transport_readback_receipt']], ['supervision/bridge_verify', 'supervision/bridge_save'], check))
    record = dict(schema=SCHEMA, passed=True, scope=SCOPE, policy_sha256=policy_sha,
        bridge_policy_sha256=bsha, model_order_sha256=canonical_hash(p['model_order']),
        model_order=p['model_order'], model_bindings=roster_binding(p),
        candidate_runtime_payload_sha256=p['candidate_runtime_payload_sha256'],
        observed_interpreter_sha256=v['observed_interpreter_sha256'],
        bridge_evidence_sha256=v['bridge_evidence_sha256'],
        bridge_verification_sha256=sha(job/'BRIDGE_VERIFICATION.json', check),
        closure_record_sha256=sha(paths['closure'], check),
        verifier_record_sha256=sha(job/'supervision/bridge_verify/RESULT.json', check),
        publisher_record_sha256=sha(job/'supervision/bridge_save/RESULT.json', check),
        durable_receipt_sha256=sha(job/'BRIDGE_DURABLE.json', check),
        durable_reference=durable['reference'], libraries_file=v['libraries_file'],
        libraries_sha256=v['libraries_sha256'], proof_files=proof_files,
        reused_bridge=reused_summary(p),
        source_pins_sha256=canonical_hash(p['source_pins']),
        gate_source_sha256=sha(Path(__file__), check), activation_sha256=sha(job/'ACTIVATION.json', check),
        allowed_calls=['fixed raw960 F/J/K validation', 'fixed B3 detector projections and scores'],
        exclusions=['training', 'inverse optimization', 'native solvers', 'noise/API work',
                    'D2 task decisions', 'general runtime acceptance', 'canonical Worker advancement'],
        scientific_acceptance=False, general_runtime_acceptance=False, formal_custody_assurance=False,
        canonical_worker_advanced=False, observations_alone_accepted=False,
        previous_clocks_reset=False, issued_at=time.time())
    put(job/'D2_RUNTIME_ADMISSION.json', record)
    return record


def verify_admission(job, policy, policy_sha, check=lambda: None):
    """Reusable before/after the 200-model child; return the verified narrow gate."""
    job = Path(job).resolve()
    p, bp = policies(job, policy_sha, check)
    exact(policy, p, 'Consumer policy')
    a = activation(job, policy_sha)
    need(time.time() < a['deadline'], 'Fixed combined-stage clock expired')
    paths = bridge_paths(job, p)
    gate = read(job/'D2_RUNTIME_ADMISSION.json')
    need(gate.get('schema') == SCHEMA and gate.get('passed') is True and gate.get('scope') == SCOPE
         and gate.get('policy_sha256') == policy_sha and gate.get('bridge_policy_sha256') == p['bridge_policy_sha256']
         and gate.get('model_order_sha256') == canonical_hash(p['model_order'])
         and gate.get('candidate_runtime_payload_sha256') == p['candidate_runtime_payload_sha256']
         and gate.get('source_pins_sha256') == canonical_hash(p['source_pins'])
         and gate.get('gate_source_sha256') == sha(Path(__file__), check)
         and gate.get('activation_sha256') == sha(job/'ACTIVATION.json', check)
         and gate.get('reused_bridge') == reused_summary(p)
         and gate.get('scientific_acceptance') is False and gate.get('general_runtime_acceptance') is False
         and gate.get('canonical_worker_advanced') is False and gate.get('observations_alone_accepted') is False,
         'Scoped D2 runtime admission unavailable or incompatible')
    exact(gate['model_order'], p['model_order'], 'Admitted continuation roster')
    exact(gate['model_bindings'], roster_binding(p), 'Admitted model descriptors/weights')
    files_match(job, gate['proof_files'], check)
    v = verification_record(job, p, policy_sha, check)
    bridge_closed(job, p, check)
    need(gate['closure_record_sha256'] == sha(paths['closure'], check), 'Gate closure binding differs')
    for name, key in [('bridge_verify', 'verifier_record_sha256'), ('bridge_save', 'publisher_record_sha256')]:
        closed(job, name, policy_sha, p['bridge_policy_sha256'], check)
        need(gate[key] == sha(job/'supervision'/name/'RESULT.json', check), 'Gate closure binding differs')
    gate_closure = closed(job, 'bridge_admit', policy_sha, p['bridge_policy_sha256'], check)
    need(gate_closure['started'] <= gate['issued_at'] <= gate_closure['started'] + gate_closure['seconds'],
         'Admission issuance is outside its supervised phase')
    need(gate['bridge_verification_sha256'] == sha(job/'BRIDGE_VERIFICATION.json', check)
         and gate['durable_receipt_sha256'] == sha(job/'BRIDGE_DURABLE.json', check)
         and gate['bridge_evidence_sha256'] == v['bridge_evidence_sha256']
         and gate['observed_interpreter_sha256'] == v['observed_interpreter_sha256']
         and gate['libraries_file'] == v['libraries_file'] and gate['libraries_sha256'] == v['libraries_sha256'],
         'Gate evidence identity differs')
    reused_observation_identity(job, p, check)
    runtime_check(job, p, bp, gate['libraries_file'], check)
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify', 'admit'])
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--policy-sha', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    job = args.job.resolve()
    sys.path[:0] = [str(job.parent/'payload/code/tools'), str(job.parent/'payload/code')]
    a = activation(job, args.policy_sha)
    need(time.time() < args.deadline <= a['deadline'], 'Verifier needs the existing fixed-stage deadline')
    began, last = time.monotonic(), [0.]

    def check():
        if time.time() >= args.deadline:
            raise TimeoutError('Fixed verifier deadline reached; no retry or reset')
        now = time.monotonic()
        if now - last[0] >= 20:
            last[0] = now
            print(json.dumps(dict(action=args.action, phase=PROGRESS['phase'], seconds=round(now - began, 1))), flush=True)
    result = (verify if args.action == 'verify' else admit)(job, args.policy_sha, check)
    print(json.dumps(dict(action=args.action, passed=result['passed'], scope=result['scope'],
                          seconds=round(time.monotonic() - began, 1)), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
