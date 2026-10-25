"""Bounded Drive transport for the fixed API-only campaign.

The numerical producer owns the actual service streams. This module packs only
finished owner directories, retains every source file, and writes immutable
remote references. It neither launches science nor admits a runtime.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import stat
import sys
import time
import zipfile
from urllib.parse import urlencode


POLICY_NAME = 'API_POLICY.json'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def safe(name):
    p = PurePosixPath(name)
    need(isinstance(name, str) and bool(p.parts) and p.as_posix() == name
         and not p.is_absolute() and '..' not in p.parts and ':' not in name
         and '\\' not in name, 'Unsafe transport member')
    return p


def sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            check()
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def source(root, name):
    safe(name)
    root = Path(root).resolve()
    p = root / name
    need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root),
         'Transport source missing or unsafe: ' + name)
    for parent in p.parents:
        if parent == root:
            break
        need(not parent.is_symlink(), 'Transport source symlink ancestor')
    return p


def pack(path, entries, limit, check, *, member_limit=24000):
    """Bound uncompressed membership and compressed output; preserve source bytes."""
    need(type(limit) is int and limit > 0, 'Positive archive limit required')
    need(member_limit in (24000, 80000), 'Fixed API archive member limit required')
    need(len(entries) <= member_limit and sum(p.stat().st_size for p in entries.values()) <= limit,
         'API archive uncompressed size/member bound exceeded')
    manifest = {}
    for name, p in sorted(entries.items()):
        check()
        safe(name)
        need(p.is_file() and not p.is_symlink(), 'API archive source differs')
        manifest[name] = dict(bytes=p.stat().st_size, sha256=sha(p, check))
    manifest_bytes = (json.dumps(dict(files=manifest, formal_custody_assurance=False),
                                 sort_keys=True, indent=2) + '\n').encode()
    with Path(path).open('xb') as sink:
        with zipfile.ZipFile(sink, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for name, p in sorted(entries.items()):
                check()
                before = p.stat()
                h = hashlib.sha256()
                with p.open('rb') as src, z.open(name, 'w', force_zip64=True) as dst:
                    while chunk := src.read(1024 * 1024):
                        check()
                        dst.write(chunk)
                        h.update(chunk)
                after = p.stat()
                need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                     and h.hexdigest() == manifest[name]['sha256'],
                     'API archive source changed during packaging')
            z.writestr('RETURN_MANIFEST.json', manifest_bytes)
        sink.flush()
        os.fsync(sink.fileno())
    need(Path(path).stat().st_size <= limit, 'API compressed archive bound exceeded')
    # Independent local ZIP member readback before remote transfer.
    with zipfile.ZipFile(path) as z:
        need(len(z.namelist()) == len(set(z.namelist()))
             and set(z.namelist()) == set(manifest) | {'RETURN_MANIFEST.json'},
             'API archive membership readback differs')
        need(z.read('RETURN_MANIFEST.json') == manifest_bytes, 'API manifest readback differs')
        for name, expected in manifest.items():
            check()
            h = hashlib.sha256()
            size = 0
            with z.open(name) as f:
                while chunk := f.read(1024 * 1024):
                    check()
                    size += len(chunk)
                    h.update(chunk)
            need(size == expected['bytes'] and h.hexdigest() == expected['sha256'],
                 'API archive member readback differs')
    return dict(files=manifest, sha256=hashlib.sha256(manifest_bytes).hexdigest())


def extract(archive, job, mapping, check):
    job = Path(job).resolve()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        need(len(names) == len(set(names)), 'Duplicate downloaded archive members')
        for name, expected in sorted(mapping.items()):
            check()
            safe(name)
            member = expected.get('member', name)
            safe(member)
            info = z.getinfo(member)
            need(not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16)
                 and not info.flag_bits & 1 and info.file_size == expected['bytes'],
                 'Pinned input member differs: ' + member)
            p = job / name
            p.parent.mkdir(parents=True, exist_ok=True)
            need(p.parent.resolve().is_relative_to(job), 'Input parent escaped job')
            h = hashlib.sha256()
            with z.open(info) as src, p.open('xb') as dst:
                while chunk := src.read(1024 * 1024):
                    check()
                    dst.write(chunk)
                    h.update(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            need(h.hexdigest() == expected['sha256'], 'Pinned input checksum differs: ' + name)


def _http(job, policy, role, byte_limit):
    code = Path(job).parent / 'payload' / 'code'
    sys.path[:0] = [str(code / 'tools'), str(code)]
    from msfinal.drive_transport import GoogleDriveTransfer, RefreshingMetadataToken
    return GoogleDriveTransfer(policy['drive_folder'], RefreshingMetadataToken('metadata.google.internal'),
                              Path(job) / ('transport_' + role), byte_limit=byte_limit,
                              metadata_byte_limit=4 * 1024 ** 2)


def _deadline(job, deadline):
    policy_path = Path(job) / POLICY_NAME
    policy = read(policy_path)
    activation = read(Path(job) / 'ACTIVATION.json')
    need(activation.get('policy_sha256') == sha(policy_path)
         and activation.get('scope') == policy.get('scope') == 'selected_API_streams_only'
         and activation.get('retries') == 0 and activation.get('previous_clocks_reused') is False,
         'API activation policy/scope/retry binding differs')
    need(time.time() < deadline <= activation['deadline'], 'Original API deadline required')

    def check():
        if time.time() >= deadline:
            raise TimeoutError('Original API transport deadline expired; no retry')
    return activation, check


def _privacy_and_quota(http, policy, deadline, check):
    from msfinal.drive_transport import API
    quota = http.quota(deadline=deadline, check=check)
    permissions = http.json('GET', API + 'files/' + policy['drive_folder'] + '/permissions?'
                            + urlencode(dict(fields='nextPageToken,permissions(id,type,role,deleted)', pageSize=100)),
                            deadline=deadline, check=check)
    need(not permissions.get('nextPageToken') and permissions.get('permissions')
         and all(v.get('type') == 'user' and v.get('role') == 'owner'
                 and not v.get('deleted') for v in permissions['permissions']),
         'Existing private owner-only result folder required')
    return dict(quota=quota, permissions=permissions, observed_at=time.time(),
                formal_custody_assurance=False)


def uploaded_control(job, policy):
    want = read(job / 'INPUT_ARCHIVE_EXPECTATION.json')
    need(set(want) == {'name', 'bytes', 'sha256'}, 'Frozen control identity fields differ')
    need(len(safe(want['name']).parts) == 1 and want['name'] == policy['control_archive_name'],
         'Frozen control archive name differs')
    control = job.parent.parent / want['name']
    need(control.is_file() and not control.is_symlink()
         and control.stat().st_size == want['bytes'] and sha(control) == want['sha256'],
         'Uploaded control bytes differ')
    return control, want


def publish_owner(job, policy, owner_root, deadline):
    """Publish three completed streams once, after their readbacks, never midstream."""
    job = Path(job).resolve()
    owner_root = Path(owner_root).resolve()
    owner = owner_root.name
    need(read(job / POLICY_NAME) == policy and owner in policy['owner_order']
         and owner_root.parent == job / 'output' / 'owners', 'API owner capsule binding differs')
    activation, check = _deadline(job, deadline)
    result_path = owner_root / 'OWNER_RESULT.json'
    owner_result = read(result_path)
    need(owner_result.get('complete') is True, 'Incomplete owner cannot be published as completed')
    receipt_path = job / 'owner_receipts' / (owner + '.json')
    need(not receipt_path.exists(), 'Owner already published; no duplicate upload')
    files = [p for p in owner_root.rglob('*') if p.is_file()]
    entries = {p.relative_to(job).as_posix(): source(job, p.relative_to(job).as_posix()) for p in files}
    archive = job / ('API_OWNER_' + activation['attempt'] + '_' + owner + '.zip')
    manifest = pack(archive, entries, policy['owner_capsule_bound_bytes'], check)
    previous = [read(job / 'owner_receipts' / (o + '.json')) for o in policy['owner_order']
                if (job / 'owner_receipts' / (o + '.json')).is_file()]
    need(sum(r['reference']['bytes'] for r in previous) + archive.stat().st_size
         <= policy['overall_return_bound_bytes'], 'Aggregate owner return bound exceeded')
    http = _http(job, policy, 'owner-' + owner, policy['owner_capsule_bound_bytes'])
    observation = _privacy_and_quota(http, policy, deadline, check)
    need(observation['quota']['available_bytes'] >= archive.stat().st_size + policy['review_bound_bytes'],
         'Drive cannot preserve this owner and final review')
    reference = http.upload_verified(archive, deadline=deadline, check=check)
    readback_path = http.log / f'{http.sequence:06d}_readback.json'
    observed = read(readback_path)
    need(observed.get('kind') == 'readback' and observed.get('object_id') == reference['object_id']
         and observed.get('revision') == reference['version'] and observed.get('bytes') == reference['bytes']
         and observed.get('sha256') == reference['sha256'], 'API owner remote readback event differs')
    receipt = dict(complete=True, durable=True, owner=owner, reference=reference,
                   local_archive=archive.name, manifest_sha256=manifest['sha256'],
                   files=len(manifest['files']), uncompressed_bytes=sum(x['bytes'] for x in manifest['files'].values()),
                   owner_result_sha256=sha(result_path), policy_sha256=sha(job / POLICY_NAME),
                   transport_readback=readback_path.relative_to(job).as_posix(),
                   transport_readback_sha256=sha(readback_path), producer_closure_pending=True,
                   scientific_acceptance=False, formal_custody_assurance=False)
    put(job / 'owner_receipts' / (owner + '_MANIFEST.json'),
        dict(owner=owner, archive=archive.name, manifest_sha256=manifest['sha256'], files=manifest['files']))
    put(receipt_path, receipt)
    print('Saved and remotely verified all three API streams for ' + owner + '.', flush=True)
    return receipt


def _closed(path):
    if not path.is_file():
        return False
    r = read(path)
    c = r.get('closure', {})
    streams = r.get('streams', {})
    streams_ok = set(streams) == {'stdout', 'stderr'} and all(
        s.get('capture_complete') is True and s.get('eof_observed') is True
        and s.get('discarded_observed_bytes') == 0 and not s.get('overflow')
        and s.get('readback', {}).get('readback_verified') is True
        for s in streams.values())
    return (r.get('returncode') == 0 and c.get('resolved') is True
            and c.get('child_reaped') is True and not c.get('survivors')
            and not c.get('errors') and r.get('stdout_closed') is True
            and not r.get('errors') and not r.get('timed_out')
            and not r.get('stream_overflow') and streams_ok)


def _validate_owner_receipt(job, policy, owner, receipt):
    need(receipt.get('owner') == owner and receipt.get('complete') is True
         and receipt.get('durable') is True and receipt.get('policy_sha256') == sha(job / POLICY_NAME),
         'API owner receipt scope differs')
    ref = receipt['reference']
    need(ref.get('folder_id') == policy['drive_folder'] and ref.get('keepForever') is True
         and 0 < ref['bytes'] <= policy['owner_capsule_bound_bytes'], 'API owner reference differs')
    result = source(job, 'output/owners/' + owner + '/OWNER_RESULT.json')
    need(sha(result) == receipt['owner_result_sha256'], 'API owner result changed after publication')
    event_path = source(job, receipt['transport_readback'])
    need(sha(event_path) == receipt['transport_readback_sha256'], 'API owner readback record changed')
    event = read(event_path)
    need(event.get('kind') == 'readback' and event.get('object_id') == ref['object_id']
         and event.get('revision') == ref['version'] and event.get('bytes') == ref['bytes']
         and event.get('sha256') == ref['sha256'], 'API owner readback identity differs')
    manifest = read(job / 'owner_receipts' / (owner + '_MANIFEST.json'))
    manifest_bytes = (json.dumps(dict(files=manifest['files'], formal_custody_assurance=False),
                                 sort_keys=True, indent=2) + '\n').encode()
    need(manifest.get('owner') == owner and manifest.get('manifest_sha256') == receipt['manifest_sha256']
         and hashlib.sha256(manifest_bytes).hexdigest() == receipt['manifest_sha256'],
         'API owner manifest binding differs')
    for name, expected in manifest['files'].items():
        p = Path(name)
        if p.name in ('OWNER_RESULT.json', 'REVIEW.json') or p.name.endswith(('_intent.json', '_result.json')):
            actual = source(job, name)
            need(actual.stat().st_size == expected['bytes'] and sha(actual) == expected['sha256'],
                 'API review bytes changed after owner publication')


def _metadata_entries(job, policy):
    entries = {}
    for name in policy['files']:
        p = job / name
        if p.is_file() and p.suffix in ('.json', '.jsonl', '.py', '.txt', '.md'):
            entries[name] = source(job, name)
    for folder in ('supervision', 'observations', 'owner_receipts'):
        for p in (job / folder).rglob('*'):
            name = p.relative_to(job).as_posix()
            if p.is_file() and not name.startswith(('supervision/save/', 'supervision/closing_record/')):
                entries[name] = source(job, name)
    for folder in job.glob('transport_*'):
        if folder.is_dir() and folder.name not in ('transport_save', 'transport_closure'):
            for p in folder.rglob('*'):
                if p.is_file():
                    name = p.relative_to(job).as_posix()
                    entries[name] = source(job, name)
    # Completed owner contents are already durable capsules. Retain only owner
    # summaries here; do not duplicate their large response arrays or raw streams.
    for p in (job / 'output').rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(job / 'output')
        if rel.parts[0] == 'owners':
            if (p.name not in ('OWNER_RESULT.json', 'FAILURE.json', 'REVIEW.json')
                    and not p.name.endswith(('_intent.json', '_result.json'))):
                continue
        name = p.relative_to(job).as_posix()
        entries[name] = source(job, name)
    for p in job.iterdir():
        if p.is_file() and p.suffix in ('.json', '.jsonl', '.py', '.txt', '.md'):
            entries[p.name] = source(job, p.name)
    return entries


def transfer(job, action, deadline):
    job = Path(job).resolve()
    policy = read(job / POLICY_NAME)
    activation, check = _deadline(job, deadline)
    if action == 'input':
        control, want = uploaded_control(job, policy)
    byte_limit = (policy['input_transport_bound_bytes'] if action == 'input'
                  else 2 * policy['return_bound_bytes'] + 2 * policy['review_bound_bytes'] + 4 * 1024 ** 2)
    http = _http(job, policy, action, byte_limit)
    observation = _privacy_and_quota(http, policy, deadline, check)
    put(job / (action.upper() + '_DRIVE_OBSERVATION.json'), observation)
    if action == 'input':
        from msfinal.drive_transport import API
        need(observation['quota']['available_bytes'] >= policy['minimum_drive_free_bytes'],
             'Observed Drive space is below this fixed API batch requirement')
        marker_name = 'API_ONCE_' + sha(job / POLICY_NAME) + '.json'
        listing = http.json('GET', API + 'files?' + urlencode(dict(
            q="'%s' in parents and trashed=false and name='%s'" % (policy['drive_folder'], marker_name),
            fields='nextPageToken,files(id,name)', pageSize=100)), deadline=deadline, check=check)
        need(not listing.get('nextPageToken') and listing.get('files') == [],
             'This fixed API campaign already has an attempt; no replay or clock reset')
        marker = job / marker_name
        put(marker, dict(activation=activation, policy_sha256=sha(job / POLICY_NAME),
                         scope=policy['scope'], attempt_only=True, formal_exclusive_custody=False))
        put(job / 'ONCE_MARKER_SAVED.json', dict(reference=http.upload_verified(marker, deadline=deadline, check=check),
                                              formal_exclusive_custody=False))
        sources = policy['sources']
        need(sum(s['reference']['bytes'] for s in sources) == policy['exact_input_download_bytes']
             <= policy['input_download_bound_bytes'], 'Fixed API input-byte allowance differs')
        destinations = [n for group in sources for n in group['files']]
        need(len(destinations) == len(set(destinations)), 'Duplicate API input destinations')
        for i, group in enumerate(sources):
            ref = group['reference']
            need(ref.get('keepForever') is True and ref.get('folder_id') == policy['drive_folder'],
                 'Pinned API input revision scope differs')
            archive = job / f'INPUT_{i:02d}.zip'
            print('Fetching pinned API input ' + str(i + 1) + '/' + str(len(sources)) + ' from Drive.', flush=True)
            with archive.open('xb') as sink:
                http.verify(ref, deadline=deadline, check=check, sink=sink)
                sink.flush()
                os.fsync(sink.fileno())
            extract(archive, job, group['files'], check)
        for name, entry in policy.get('aliases', {}).items():
            check()
            src = source(job, entry['source'])
            expected = entry['expected']
            need(src.stat().st_size == expected['bytes'] and sha(src, check) == expected['sha256'],
                 'API alias source differs: ' + name)
            safe(name)
            dst = job / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            need(dst.parent.resolve().is_relative_to(job), 'API alias parent escaped job')
            with src.open('rb') as a, dst.open('xb') as b:
                while chunk := a.read(1024 * 1024):
                    check()
                    b.write(chunk)
                b.flush()
                os.fsync(b.fileno())
            need(dst.stat().st_size == expected['bytes'] and sha(dst, check) == expected['sha256'],
                 'API alias byte readback differs')
        need(uploaded_control(job, policy) == (control, want), 'API control changed during inputs')
        put(job / 'CONTROL_SAVED.json', dict(reference=http.upload_verified(control, deadline=deadline, check=check),
                                          policy_sha256=sha(job / POLICY_NAME)))
        put(job / 'INPUT_READBACK.json', dict(complete=True, sources=[s['reference'] for s in sources],
                  files=len(destinations), aliases=len(policy.get('aliases', {})),
                  exact_input_download_bytes=policy['exact_input_download_bytes'],
                  frozen_expectations_checked_against_fresh_bytes=True, new_scientific_calls=0))
        return
    if action == 'save':
        # Completed capsules stay separate. On a failure, preserve incomplete
        # owner bytes as a separately marked evidence capsule, without replay.
        owner_receipts = {}
        for owner in policy['owner_order']:
            p = job / 'owner_receipts' / (owner + '.json')
            if p.is_file():
                r = read(p)
                _validate_owner_receipt(job, policy, owner, r)
                owner_receipts[owner] = r
        incomplete = []
        for owner_root in sorted((job / 'output' / 'owners').glob('*')):
            if not owner_root.is_dir() or owner_root.name in owner_receipts:
                continue
            entries = {p.relative_to(job).as_posix(): source(job, p.relative_to(job).as_posix())
                       for p in owner_root.rglob('*') if p.is_file()}
            if entries:
                archive = job / ('API_PARTIAL_' + activation['attempt'] + '_' + owner_root.name + '.zip')
                manifest = pack(archive, entries, policy['owner_capsule_bound_bytes'], check)
                ref = http.upload_verified(archive, deadline=deadline, check=check)
                incomplete.append(dict(owner=owner_root.name, reference=ref,
                                       manifest_sha256=manifest['sha256'], numerical_complete=False))
        index = dict(owners=owner_receipts, partial=incomplete, policy_sha256=sha(job / POLICY_NAME),
                     all_sources_retained_locally=True, formal_custody_assurance=False)
        put(job / 'OWNER_CAPSULE_INDEX.json', index)
        entries = _metadata_entries(job, policy)
        full = job / ('API_FULL_' + activation['attempt'] + '.zip')
        pack(full, entries, policy['return_bound_bytes'], check,
             member_limit=policy['metadata_review_member_bound'])
        need(observation['quota']['available_bytes'] >= full.stat().st_size + policy['review_bound_bytes'],
             'Drive cannot preserve API metadata and review')
        ref = http.upload_verified(full, deadline=deadline, check=check)
        descriptor = job / 'DURABLE_FULL_RETURN.json'
        put(descriptor, dict(reference=ref, owner_capsules=owner_receipts, partial_capsules=incomplete,
                             result_url='https://drive.google.com/file/d/' + ref['object_id'] + '/view',
                             formal_custody_assurance=False))
        selected = {n: p for n, p in entries.items() if p.suffix in ('.json', '.jsonl', '.py', '.txt', '.md')}
        selected['DURABLE_FULL_RETURN.json'] = descriptor
        review = job / ('API_REVIEW_' + activation['attempt'] + '.zip')
        pack(review, selected, policy['review_bound_bytes'], check,
             member_limit=policy['metadata_review_member_bound'])
        review_ref = http.upload_verified(review, deadline=deadline, check=check)
        producer = read(job / 'output' / 'RESULT.json') if (job / 'output' / 'RESULT.json').is_file() else {}
        closed = _closed(job / 'supervision' / 'producer' / 'RESULT.json')
        expected_jobs = {row['id'] for row in policy['api_jobs']}
        population_complete = (producer.get('complete') is True
                               and producer.get('completed_jobs') == len(expected_jobs)
                               and set(producer.get('completed_streams', [])) == expected_jobs
                               and producer.get('attempts') == sum(row['attempts'] for row in policy['api_jobs'])
                               and producer.get('durable_owners') == len(policy['owner_order']))
        complete = (population_complete and closed
                    and set(owner_receipts) == set(policy['owner_order']) and not incomplete)
        saved = dict(complete=complete, saved=True, archive=full.name, reference=ref,
                     review_archive=review.name, review_reference=review_ref,
                     local_sha256=sha(full), review_sha256=sha(review),
                     owner_capsules=owner_receipts, partial_capsules=incomplete,
                     producer_complete=producer.get('complete') is True, producer_closed=closed,
                     scientific_acceptance=False, formal_custody_assurance=False, canonical_worker_advanced=False,
                     result_url='https://drive.google.com/file/d/' + ref['object_id'] + '/view',
                     review_url='https://drive.google.com/file/d/' + review_ref['object_id'] + '/view')
        put(job / 'SAVE_RESULT.json', saved)
        print('DOWNLOAD THIS SMALL REVIEW: ' + saved['review_url'], flush=True)
        return
    need(action == 'closure', 'Unknown API transport action')
    need(_closed(job / 'supervision' / 'save' / 'RESULT.json'), 'API metadata publisher closure unresolved')
    saved = read(job / 'SAVE_RESULT.json')
    closure = read(job / 'supervision' / 'save' / 'RESULT.json')
    path = job / ('API_SAVED_' + activation['attempt'] + '.json')
    put(path, dict(saved=saved, publisher_closure=closure, activation=activation,
                   note='Producer and main publisher records preserved; this final metadata publisher is separate.',
                   formal_custody_assurance=False))
    ref = http.upload_verified(path, deadline=deadline, check=check)
    put(job / 'CLOSING_RECORD_SAVED.json', dict(reference=ref, result_url=saved['result_url'],
                 review_url=saved['review_url'], closure_url='https://drive.google.com/file/d/' + ref['object_id'] + '/view'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['input', 'save', 'closure'])
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--deadline', required=True, type=float)
    args = parser.parse_args()
    transfer(args.job, args.action, args.deadline)
