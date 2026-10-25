"""Lossless bounded transport for the fixed twenty-job noise campaign.

No model, solver, noise draw or runtime admission occurs here. Exact original
output bytes are split at call boundaries, read back locally and remotely,
then indexed durably before redundant scratch tensors can be pruned.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import stat
import shutil
import re
import sys
import time
import zipfile
from urllib.parse import urlencode


POLICY_NAME = 'NOISE_POLICY.json'


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
    need(member_limit in (24000, 80000), 'Fixed noise archive member limit required')
    need(len(entries) + 1 <= member_limit and sum(p.stat().st_size for p in entries.values()) <= limit,
         'noise archive uncompressed size/member bound exceeded')
    manifest = {}
    for name, p in sorted(entries.items()):
        check()
        safe(name)
        need(p.is_file() and not p.is_symlink(), 'noise archive source differs')
        manifest[name] = dict(bytes=p.stat().st_size, sha256=sha(p, check))
    manifest_bytes = (json.dumps(dict(files=manifest, formal_custody_assurance=False),
                                 sort_keys=True, indent=2) + '\n').encode()
    need(sum(v['bytes'] for v in manifest.values()) + len(manifest_bytes) <= limit,
         'Noise archive including manifest exceeds its uncompressed bound')
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
                     'noise archive source changed during packaging')
            z.writestr('RETURN_MANIFEST.json', manifest_bytes)
        sink.flush()
        os.fsync(sink.fileno())
    need(Path(path).stat().st_size <= limit, 'noise compressed archive bound exceeded')
    # Independent local ZIP member readback before remote transfer.
    with zipfile.ZipFile(path) as z:
        need(len(z.namelist()) == len(set(z.namelist()))
             and set(z.namelist()) == set(manifest) | {'RETURN_MANIFEST.json'},
             'noise archive membership readback differs')
        need(z.read('RETURN_MANIFEST.json') == manifest_bytes, 'noise manifest readback differs')
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
                 'noise archive member readback differs')
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
         and activation.get('scope') == policy.get('scope') == 'selected_noise20_only'
         and activation.get('retries') == 0 and activation.get('previous_clocks_reused') is False,
         'noise activation policy/scope/retry binding differs')
    need(time.time() < deadline <= activation['deadline'], 'Original noise deadline required')

    def check():
        if time.time() >= deadline:
            raise TimeoutError('Original noise transport deadline expired; no retry')
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


def _job_row(policy, member):
    matches = [r for r in policy['noise_jobs'] if r['id'] == member]
    need(len(matches) == 1, 'Exactly one frozen noise job required')
    row = matches[0]
    need(re.fullmatch(r'O(?:0[1-9]|10)/noise_(?:small|large)_M3', member) is not None,
         'Unexpected noise member')
    token = member.replace('/', '_')
    need(row['output'] == 'output/jobs/' + member
         and row['closure'] == 'supervision/producer_' + token + '/RESULT.json',
         'Frozen noise output/closure path differs')
    return row, token


def _bound_identity(job, policy):
    need(read(job/POLICY_NAME) == policy, 'Noise policy bytes/object differ')
    activation = read(job/'ACTIVATION.json')
    policy_sha = sha(job/POLICY_NAME)
    activation_sha = sha(job/'ACTIVATION.json')
    need(activation['policy_sha256'] == policy_sha
         and activation['scope'] == policy['scope'] == 'selected_noise20_only'
         and activation['retries'] == 0 and activation['previous_clocks_reused'] is False,
         'Noise fixed activation binding differs')
    return activation, policy_sha, activation_sha


def _receipt(job, http, reference):
    path = http.log/f'{http.sequence:06d}_readback.json'
    value = read(path)
    need(value.get('kind') == 'readback' and value.get('object_id') == reference['object_id']
         and value.get('revision') == reference['version'] and value.get('bytes') == reference['bytes']
         and value.get('sha256') == reference['sha256'], 'Remote readback event differs')
    return dict(reference=reference, transport_readback=path.relative_to(job).as_posix(),
                transport_readback_sha256=sha(path))


def _validate_reference(job, policy, record):
    ref = record['reference']
    need(ref['folder_id'] == policy['drive_folder'] and ref['keepForever'] is True
         and ref['bytes'] > 0, 'Durable noise reference scope differs')
    path = source(job, record['transport_readback'])
    need(sha(path) == record['transport_readback_sha256'], 'Transport readback bytes changed')
    event = read(path)
    need(event['kind'] == 'readback' and event['object_id'] == ref['object_id']
         and event['revision'] == ref['version'] and event['bytes'] == ref['bytes']
         and event['sha256'] == ref['sha256'], 'Durable noise reference/readback differs')


def _global_uploaded_bytes(job):
    """Frozen helper charges each new upload once, including failed attempts.

    Downloads of already-held input revisions incur no new-storage charge.
    """
    total = 0
    for p in job.glob('transport_*/*_upload_charge.json'):
        value = read(p)
        need(value.get('kind') == 'upload_charge' and type(value.get('bytes')) is int
             and value['bytes'] > 0, 'Invalid new-upload charge event')
        total += value['bytes']
    return total


def _upload(job, policy, http, archive, deadline, check):
    check()
    need(_global_uploaded_bytes(job) + archive.stat().st_size <= policy['aggregate_return_bound_bytes'],
         'Fixed whole-wave transport bound exceeded')
    quota = http.quota(deadline=deadline, check=check)
    need(quota['available_bytes'] >= archive.stat().st_size + policy['review_bound_bytes'],
         'Drive cannot preserve this shard and final review')
    return _receipt(job, http, http.upload_verified(archive, deadline=deadline, check=check))


def capacity(job, deadline):
    """Authenticated, bounded capacity observation before activation or download."""
    job = Path(job).resolve()
    policy = read(job/POLICY_NAME)
    need(policy['scope'] == 'selected_noise20_only' and not (job/'ACTIVATION.json').exists(),
         'Capacity check belongs before activation')
    need(time.time() < deadline <= time.time() + 900, 'Bounded pre-activation deadline required')
    def check():
        if time.time() >= deadline:
            raise TimeoutError('Pre-activation capacity observation expired')
    http = _http(job, policy, 'capacity', 4*1024**2)
    observation = _privacy_and_quota(http, policy, deadline, check)
    local_free = shutil.disk_usage(job).free
    value = dict(scope=policy['scope'], policy_sha256=sha(job/POLICY_NAME),
                 before_activation=True, observed_at=time.time(), drive=observation,
                 private_owner_only=True,drive_free_bytes=observation['quota']['available_bytes'],
                 linux_free_bytes=local_free, minimum_drive_free_bytes=policy['minimum_drive_free_bytes'],
                 minimum_linux_free_bytes=policy['minimum_linux_free_bytes'],
                 passed=observation['quota']['available_bytes'] >= policy['minimum_drive_free_bytes']
                    and local_free >= policy['minimum_linux_free_bytes'],
                 not_a_reservation=True, purchases=0, deletions=0, scientific_calls=0)
    put(job/'STORAGE_PREFLIGHT.json', value)
    need(value['passed'], 'Insufficient fixed-wave storage; no activation, download or science started')
    return value


def _noise_members(job, root):
    """Exact frozen 512 x 32 layout; only integer call-boundary grouping."""
    noise = root/'noise'
    expected = {'STREAM.json'}
    groups = [[] for _ in range(4)]
    groups[0].append('STREAM.json')
    for call in range(512):
        names = [f'call_{call:06d}.json']
        for repetition in range(32):
            stem = f'{call:06d}_{repetition:02d}'
            names.extend((stem+'.pt', stem+'.json', stem+'.pt.publication.json'))
        groups[call//128].extend(names)
        expected.update(names)
    need(noise.is_dir() and not noise.is_symlink()
         and {p.name for p in noise.iterdir()} == expected, 'Noise original membership differs from frozen512x32')
    return [{(noise/name).relative_to(job).as_posix(): source(job, (noise/name).relative_to(job).as_posix())
             for name in group} for group in groups]


def _closed_job(job, policy, row):
    root = job/row['output']
    closure_path = source(job, row['closure'])
    need(_closed(closure_path), 'Noise numerical child closure is not successful')
    command = read(closure_path)['command']
    need(command.count('--member') == 1 and command[command.index('--member')+1] == row['id']
         and command.count('--policy-sha') == 1
         and command[command.index('--policy-sha')+1] == sha(job/POLICY_NAME),
         'Noise child closure command binds a different member/policy')
    result = read(root/'JOB_RESULT.json')
    noise = read(root/'NOISE_READBACK.json')
    gpu = read(root/'GPU_READBACK.json')
    need(result.get('complete') is True and result.get('member',result.get('id')) == row['id']
         and result.get('policy_sha256') == sha(job/POLICY_NAME)
         and result.get('activation_sha256') == sha(job/'ACTIVATION.json')
         and result.get('native_solver_calls') == 0 and result.get('task_scoring_pending') is True,
         'Noise job result scope/completion differs')
    need(noise.get('calls') == 512 and noise.get('retained_outputs') == 16384
         and noise.get('white_box_only') is True, 'Frozen noise readback population differs')
    need(gpu.get('complete') is True and gpu.get('task_scoring_pending') is True,
         'Noise GPU readback incomplete')
    need(gpu.get('member') == row['id'] and gpu.get('scope') == result.get('scope') == policy['scope']
         and gpu.get('model_sha256') == row['parent_sha256']
         and result.get('request_sha256') == row['request_sha256']
         and result.get('GPU_evaluation_complete') is True and result.get('noise_readback_complete') is True,
         'Noise result request/model/scope binding differs')
    need(gpu.get('result_sha256') == result.get('evaluation_result_sha256') == sha(root/'evaluation/RESULT.json')
         and gpu.get('request_sha256') == result.get('D6_request_sha256') == sha(root/'evaluation/D6_REQUEST.json')
         and gpu.get('noise_readback_sha256') == result.get('noise_readback_sha256') == sha(root/'NOISE_READBACK.json')
         and result.get('GPU_readback_sha256') == sha(root/'GPU_READBACK.json'),
         'Noise output/readback hashes differ')
    need(gpu.get('saved_designs') == result.get('saved_designs') == 64
         and gpu.get('inverse_batch_iterations') == result.get('inverse_batch_iterations') == 480
         and gpu.get('native_solver_calls') == gpu.get('model_updates') == result.get('model_updates') == 0,
         'Noise saved-design operation counts differ')
    return dict(closure_sha256=sha(closure_path), noise_readback_sha256=sha(root/'NOISE_READBACK.json'),
                gpu_readback_sha256=sha(root/'GPU_READBACK.json'), job_result_sha256=sha(root/'JOB_RESULT.json'))


def _shard_manifest(job, token, ordinal, archive, manifest):
    path = job/'job_manifests'/token/f'SHARD_{ordinal:02d}.json'
    value = dict(ordinal=ordinal, call_first=128*ordinal, call_last=128*ordinal+127,
                 archive=archive.name, files=manifest['files'], manifest_sha256=manifest['sha256'])
    put(path, value)
    return path, value


def _metadata_job_entries(job, root, token):
    entries = {}
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if p.parent == root/'noise' and p.name != 'STREAM.json' and not re.fullmatch(r'call_\d{6}\.json', p.name):
            continue
        name = p.relative_to(job).as_posix()
        entries[name] = source(job, name)
    for folder in ('job_manifests','shard_receipts'):
        for p in (job/folder/token).rglob('*'):
            if p.is_file():
                name = p.relative_to(job).as_posix()
                entries[name] = source(job,name)
    return entries


def publish_job(job, policy, job_root, deadline):
    """One completed job: four raw shards, metadata, durable index, scratch cleanup."""
    job = Path(job).resolve()
    job_root = Path(job_root).resolve()
    matches = [r for r in policy['noise_jobs'] if job/ r['output'] == job_root]
    need(len(matches) == 1, 'Publication root is not one frozen noise job')
    row, token = _job_row(policy, matches[0]['id'])
    activation, check = _deadline(job,deadline)
    _, policy_sha, activation_sha = _bound_identity(job,policy)
    proof = _closed_job(job,policy,row)
    receipt_path = job/'job_receipts'/(token+'.json')
    need(not receipt_path.exists(), 'Job already committed; no publication retry')
    groups = _noise_members(job, job_root)
    raw_bytes = sum(p.stat().st_size for group in groups for p in group.values())
    need(raw_bytes <= policy['job_uncompressed_bound_bytes'], 'Frozen per-job raw retention bound exceeded')
    artifact_root = job/'transport_artifacts'/token
    artifact_root.mkdir(parents=True,exist_ok=False)
    http = _http(job,policy,'job-'+token,2*policy['job_uncompressed_bound_bytes']+policy['job_metadata_bound_bytes']*2)
    _privacy_and_quota(http,policy,deadline,check)
    shards = []
    for ordinal, entries in enumerate(groups):
        print(f'{row["id"]}: preserving exact noise calls {ordinal*128}–{ordinal*128+127} (shard {ordinal+1}/4).',flush=True)
        archive = artifact_root/f'NOISE_RAW_{activation["attempt"]}_{token}_{ordinal:02d}.zip'
        manifest = pack(archive,entries,policy['shard_bound_bytes'],check)
        manifest_path, value = _shard_manifest(job,token,ordinal,archive,manifest)
        uploaded = _upload(job,policy,http,archive,deadline,check)
        record = dict(uploaded, member=row['id'], ordinal=ordinal,
                      manifest=manifest_path.relative_to(job).as_posix(), manifest_record_sha256=sha(manifest_path),
                      manifest_sha256=manifest['sha256'], files=len(entries),
                      raw_bytes=sum(p.stat().st_size for p in entries.values()), local_archive=archive.relative_to(job).as_posix())
        put(job/'shard_receipts'/token/f'{ordinal:02d}.json',record)
        shards.append(record)
        print(f'{row["id"]}: shard {ordinal+1}/4 uploaded and independently read back.',flush=True)
    metadata = artifact_root/f'NOISE_JOB_{activation["attempt"]}_{token}.zip'
    entries = _metadata_job_entries(job,job_root,token)
    entries[row['closure']] = source(job,row['closure'])
    manifest = pack(metadata,entries,policy['job_metadata_bound_bytes'],check)
    metadata_record = _upload(job,policy,http,metadata,deadline,check)
    metadata_manifest_path = job/'job_manifests'/token/'METADATA.json'
    put(metadata_manifest_path,dict(files=manifest['files'],manifest_sha256=manifest['sha256']))
    index = dict(member=row['id'],id=row['id'],scope=policy['scope'],policy_sha256=policy_sha,
                 activation_sha256=activation_sha, complete=True, durable=True, shards=shards,
                 metadata=metadata_record,metadata_manifest_sha256=manifest['sha256'],
                 metadata_manifest=metadata_manifest_path.relative_to(job).as_posix(),
                 metadata_manifest_record_sha256=sha(metadata_manifest_path),
                 original_calls=512,original_outputs=16384,raw_bytes=raw_bytes,
                 native_solver_calls=0,task_scoring_pending=True,retained_designs=True,
                 cleanup_permitted='Only committed per-output .pt, .json, .pt.publication.json and exact temporary archives; retain STREAM/call records, evaluations and projections.',
                 scientific_acceptance=False,formal_custody_assurance=False,**proof)
    commit = job/'commits'/(token+'.json')
    put(commit,index)
    commit_record = _upload(job,policy,http,commit,deadline,check)
    receipt = dict(index,commit=commit_record,commit_reference=commit_record['reference'],
                   commit_sha256=sha(commit),metadata_reference=metadata_record['reference'])
    put(receipt_path,receipt)
    validate_job_commit(job,policy,row['id'],require_cleanup=False)
    _cleanup(job,policy,row,receipt,check)
    validate_job_commit(job,policy,row['id'],require_cleanup=True)
    print(row['id']+': durable raw evidence and saved designs verified; redundant scratch tensors released.',flush=True)
    return receipt


def _manifest_files(job,policy,receipt):
    files = {}
    need(len(receipt['shards']) == 4 and [r['ordinal'] for r in receipt['shards']] == list(range(4)),
         'Exactly four ordered noise shards required')
    for record in receipt['shards']:
        _validate_reference(job,policy,record)
        path = source(job,record['manifest'])
        need(sha(path) == record['manifest_record_sha256'], 'Noise shard manifest record changed')
        manifest = read(path)
        expected_hash = hashlib.sha256((json.dumps(dict(files=manifest['files'],formal_custody_assurance=False),sort_keys=True,indent=2)+'\n').encode()).hexdigest()
        need(manifest['manifest_sha256'] == record['manifest_sha256'] == expected_hash,
             'Noise shard exact-member manifest binding differs')
        need(not (set(files)&set(manifest['files'])), 'Noise shard membership overlaps')
        files.update(manifest['files'])
    return files


def _cleanup_candidates(row, files):
    prefix = row['output']+'/noise/'
    result = {}
    for name, expected in files.items():
        need(name.startswith(prefix), 'Noise shard includes a path outside this job')
        tail = name[len(prefix):]
        if re.fullmatch(r'\d{6}_\d{2}\.(?:pt|json|pt\.publication\.json)',tail):
            result[name] = expected
        else:
            need(tail == 'STREAM.json' or re.fullmatch(r'call_\d{6}\.json',tail), 'Unexpected noise deletion class')
    need(len(result) == 512*32*3, 'Noise deletion population differs')
    return result


def _cleanup(job,policy,row,receipt,check):
    token = row['id'].replace('/','_')
    files = _manifest_files(job,policy,receipt)
    candidates = _cleanup_candidates(row,files)
    base = dict(member=row['id'],id=row['id'],policy_sha256=receipt['policy_sha256'],
                activation_sha256=receipt['activation_sha256'],commit_sha256=receipt['commit_sha256'])
    intent = dict(base, candidate_files=len(candidates),candidate_bytes=sum(r['bytes'] for r in candidates.values()),
                  candidate_map_sha256=hashlib.sha256((json.dumps(candidates,sort_keys=True,indent=2)+'\n').encode()).hexdigest(),
                  historical_files_deleted=False, durable_remote_commit_verified=True)
    out = job/'cleanup'/token
    put(out/'INTENT.json',intent)
    deleted = deleted_bytes = 0
    archives_deleted = []
    try:
        for name, expected in sorted(candidates.items()):
            check()
            p = source(job,name)
            need(p.stat().st_size == expected['bytes'] and sha(p,check) == expected['sha256'],
                 'Refusing cleanup: local noise bytes differ from committed remote capsule')
            p.unlink()
            deleted += 1
            deleted_bytes += expected['bytes']
        for record in receipt['shards']:
            p = source(job,record['local_archive'])
            ref = record['reference']
            need(p.stat().st_size == ref['bytes'] and sha(p,check) == ref['sha256'], 'Temporary shard changed after publication')
            p.unlink()
            archives_deleted.append(record['local_archive'])
        metadata_name = 'transport_artifacts/'+token+'/NOISE_JOB_'+read(job/'ACTIVATION.json')['attempt']+'_'+token+'.zip'
        p = source(job,metadata_name)
        ref = receipt['metadata_reference']
        need(p.stat().st_size == ref['bytes'] and sha(p,check) == ref['sha256'], 'Temporary metadata archive changed')
        p.unlink()
        archives_deleted.append(metadata_name)
        need(not any((job/n).exists() for n in candidates), 'Committed raw scratch cleanup incomplete')
        put(out/'RESULT.json',dict(base,complete=True,deleted_raw_files=deleted,deleted_raw_bytes=deleted_bytes,
                                  deleted_local_archives=archives_deleted,retained_designs=True,
                                  historical_files_deleted=False,scientific_acceptance=False,formal_custody_assurance=False))
    except BaseException as exc:
        put(out/'FAILURE.json',dict(base,complete=False,deleted_raw_files=deleted,deleted_raw_bytes=deleted_bytes,
                                   deleted_local_archives=archives_deleted,error=repr(exc),
                                   raw_bytes_remain_durably_recoverable=True))
        raise


def validate_job_commit(job,policy,member,*,require_cleanup=True):
    """Pure local verification for the launcher; no network, science or pruning."""
    job = Path(job).resolve()
    row,token = _job_row(policy,member)
    activation,policy_sha,activation_sha = _bound_identity(job,policy)
    receipt = read(job/'job_receipts'/(token+'.json'))
    need(receipt['member'] == receipt['id'] == member and receipt['policy_sha256'] == policy_sha
         and receipt['activation_sha256'] == activation_sha and receipt['complete'] is True
         and receipt['durable'] is True, 'Noise commit binding differs')
    proof = _closed_job(job,policy,row)
    need(all(receipt[k] == v for k,v in proof.items()), 'Producer evidence changed after noise commit')
    commit = source(job,'commits/'+token+'.json')
    need(sha(commit) == receipt['commit_sha256'], 'Durable commit bytes differ')
    committed = read(commit)
    need(all(receipt.get(k) == v for k,v in committed.items()), 'Receipt differs from committed index')
    _validate_reference(job,policy,receipt['commit'])
    need(receipt['commit_reference'] == receipt['commit']['reference']
         and receipt['commit_reference']['sha256'] == receipt['commit_sha256'], 'Remote commit hash differs')
    _validate_reference(job,policy,receipt['metadata'])
    need(receipt['metadata_reference'] == receipt['metadata']['reference'], 'Metadata durable reference differs')
    metadata_path = source(job,receipt['metadata_manifest'])
    need(sha(metadata_path) == receipt['metadata_manifest_record_sha256'], 'Metadata manifest record changed')
    metadata_manifest = read(metadata_path)
    metadata_hash = hashlib.sha256((json.dumps(dict(files=metadata_manifest['files'],formal_custody_assurance=False),
                                              sort_keys=True,indent=2)+'\n').encode()).hexdigest()
    need(metadata_manifest['manifest_sha256'] == receipt['metadata_manifest_sha256'] == metadata_hash,
         'Job metadata member manifest binding differs')
    for name,expected in metadata_manifest['files'].items():
        path = source(job,name)
        need(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'],
             'Retained design/evaluation/provenance changed after publication: '+name)
    files = _manifest_files(job,policy,receipt)
    need(len(files) == 49665 and sum(x['bytes'] for x in files.values()) == receipt['raw_bytes'], 'Committed noise member totals differ')
    for name, expected in files.items():
        if Path(name).name == 'STREAM.json' or Path(name).name.startswith('call_'):
            path = source(job,name)
            need(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], 'Retained noise metadata changed')
    if require_cleanup:
        cleanup = read(job/'cleanup'/token/'RESULT.json')
        need(cleanup.get('complete') is True and cleanup.get('retained_designs') is True
             and all(cleanup[k] == receipt[k] for k in ('member','id','policy_sha256','activation_sha256','commit_sha256')),
             'Scratch cleanup/result binding incomplete')
        candidates = _cleanup_candidates(row,files)
        need(cleanup['deleted_raw_files'] == len(candidates)
             and cleanup['deleted_raw_bytes'] == sum(v['bytes'] for v in candidates.values())
             and not any((job/n).exists() for n in candidates), 'Scratch cleanup population differs')
    return receipt


def _input(job,policy,activation,http,observation,deadline,check):
    control,want = uploaded_control(job,policy)
    capacity_record = read(job/'STORAGE_PREFLIGHT.json')
    need(capacity_record['passed'] is True and capacity_record['before_activation'] is True
         and capacity_record['policy_sha256'] == sha(job/POLICY_NAME)
         and observation['quota']['available_bytes'] >= policy['minimum_drive_free_bytes']
         and shutil.disk_usage(job).free >= policy['minimum_linux_free_bytes'],
         'Fixed-wave storage changed before inputs')
    from msfinal.drive_transport import API
    marker_name = 'NOISE_ONCE_'+sha(job/POLICY_NAME)+'.json'
    listing = http.json('GET',API+'files?'+urlencode(dict(
        q="'%s' in parents and trashed=false and name='%s'"%(policy['drive_folder'],marker_name),
        fields='nextPageToken,files(id,name)',pageSize=100)),deadline=deadline,check=check)
    need(not listing.get('nextPageToken') and listing.get('files') == [],
         'This fixed noise campaign already has an attempt; no replay or clock reset')
    marker = job/marker_name
    put(marker,dict(activation=activation,policy_sha256=sha(job/POLICY_NAME),scope=policy['scope'],
                    attempt_only=True,formal_exclusive_custody=False))
    put(job/'ONCE_MARKER_SAVED.json',dict(_upload(job,policy,http,marker,deadline,check),formal_exclusive_custody=False))
    sources = policy['sources']
    need(sum(s['reference']['bytes'] for s in sources) == policy['exact_input_download_bytes']
         <= policy['input_download_bound_bytes'],'Fixed noise input-byte allowance differs')
    destinations = [n for group in sources for n in group['files']]
    need(len(destinations) == len(set(destinations)),'Duplicate noise input destinations')
    for ordinal,group in enumerate(sources):
        ref = group['reference']
        need(ref.get('keepForever') is True and ref['folder_id'] == policy['drive_folder'],
             'Pinned noise input revision scope differs')
        archive = job/f'INPUT_{ordinal:02d}.zip'
        print(f'Fetching pinned noise input {ordinal+1}/{len(sources)} from Drive.',flush=True)
        with archive.open('xb') as sink:
            http.verify(ref,deadline=deadline,check=check,sink=sink)
            sink.flush();os.fsync(sink.fileno())
        extract(archive,job,group['files'],check)
    for name,entry in policy.get('aliases',{}).items():
        check()
        src = source(job,entry['source'])
        expected = entry['expected']
        need(src.stat().st_size == expected['bytes'] and sha(src,check) == expected['sha256'],
             'Noise alias source differs: '+name)
        safe(name)
        dst = job/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        need(dst.parent.resolve().is_relative_to(job),'Noise alias parent escaped job')
        with src.open('rb') as a,dst.open('xb') as b:
            while chunk := a.read(1024*1024):
                check();b.write(chunk)
            b.flush();os.fsync(b.fileno())
        need(dst.stat().st_size == expected['bytes'] and sha(dst,check) == expected['sha256'],'Noise alias byte readback differs')
    need(uploaded_control(job,policy) == (control,want),'Noise control changed during inputs')
    put(job/'CONTROL_SAVED.json',dict(_upload(job,policy,http,control,deadline,check),policy_sha256=sha(job/POLICY_NAME)))
    put(job/'INPUT_READBACK.json',dict(complete=True,sources=[s['reference'] for s in sources],files=len(destinations),
               aliases=len(policy.get('aliases',{})),exact_input_download_bytes=policy['exact_input_download_bytes'],
               frozen_expectations_checked_against_fresh_bytes=True,new_scientific_calls=0))


def _partial(job,policy,row,http,activation,deadline,check):
    """Preserve unfinished current-job bytes without a new model call or cleanup."""
    root = job/row['output'];token = row['id'].replace('/','_')
    prior = []
    covered = {}
    for path in sorted((job/'shard_receipts'/token).glob('*.json')):
        record = read(path)
        _validate_reference(job,policy,record)
        manifest_path = source(job,record['manifest'])
        need(sha(manifest_path) == record['manifest_record_sha256'],'Partial preservation manifest changed')
        manifest = read(manifest_path)
        covered.update(manifest['files'])
        prior.append(record)
    groups = [dict() for _ in range(4)]
    noise = root/'noise'
    for path in sorted(noise.glob('*')):
        if not path.is_file():
            need(False,'Unexpected partial noise subdirectory')
        name = path.relative_to(job).as_posix()
        path = source(job,name)
        if name in covered:
            expected = covered[name]
            need(path.stat().st_size == expected['bytes'] and sha(path,check) == expected['sha256'],
                 'Previously saved partial noise bytes changed')
            continue
        match = re.match(r'(?:call_)?(\d{6})(?:_|\.)',path.name)
        ordinal = min(int(match.group(1))//128,3) if match else 0
        groups[ordinal][name] = path
    added = []
    for ordinal,entries in enumerate(groups):
        if not entries:
            continue
        archive = job/f'NOISE_PARTIAL_{activation["attempt"]}_{token}_{ordinal:02d}.zip'
        manifest = pack(archive,entries,policy['shard_bound_bytes'],check)
        published = _upload(job,policy,http,archive,deadline,check)
        manifest_path = job/'partial_manifests'/token/f'{ordinal:02d}.json'
        put(manifest_path,dict(files=manifest['files'],manifest_sha256=manifest['sha256']))
        added.append(dict(published,manifest=manifest_path.relative_to(job).as_posix(),
                          manifest_sha256=manifest['sha256'],files=len(entries)))
    return dict(member=row['id'],complete=False,existing_shards=prior,additional_shards=added,
                local_raw_files_retained=True,scientific_acceptance=False,formal_custody_assurance=False)


def _metadata_entries(job,policy):
    entries = {}
    for name in policy['files']:
        path = job/name
        if path.is_file() and path.suffix in ('.json','.jsonl','.py','.txt','.md'):
            entries[name] = source(job,name)
    folders = ['supervision','observations','job_manifests','partial_manifests','shard_receipts','job_receipts','commits','cleanup',
               'job_windows','stage_budgets']
    folders += [p.name for p in job.glob('transport_*') if p.is_dir() and p.name not in ('transport_artifacts','transport_save','transport_closure')]
    for folder in folders:
        for path in (job/folder).rglob('*'):
            if path.is_file():
                name = path.relative_to(job).as_posix()
                if not name.startswith(('supervision/save/','supervision/closing_record/')):
                    entries[name] = source(job,name)
    for path in (job/'output').rglob('*'):
        if not path.is_file():
            continue
        if path.parent.name == 'noise' and path.name != 'STREAM.json' and not re.fullmatch(r'call_\d{6}\.json',path.name):
            continue
        name = path.relative_to(job).as_posix()
        entries[name] = source(job,name)
    for path in job.iterdir():
        if path.is_file() and path.suffix in ('.json','.jsonl','.py','.txt','.md'):
            entries[path.name] = source(job,path.name)
    return entries


def transfer(job,action,deadline,member=None):
    job = Path(job).resolve()
    if action == 'capacity':
        return capacity(job,deadline)
    policy = read(job/POLICY_NAME)
    activation,check = _deadline(job,deadline)
    if action == 'publish':
        row,_ = _job_row(policy,member)
        return publish_job(job,policy,job/row['output'],deadline)
    need(action in ('input','save','closure'),'Unknown noise transport action')
    if action == 'input':
        uploaded_control(job,policy)
    byte_limit = policy['input_transport_bound_bytes'] if action == 'input' else 2*(
        policy['job_uncompressed_bound_bytes']+policy['return_bound_bytes']+
        policy['design_return_bound_bytes']+policy['review_bound_bytes'])
    http = _http(job,policy,action,byte_limit)
    observation = _privacy_and_quota(http,policy,deadline,check)
    put(job/(action.upper()+'_DRIVE_OBSERVATION.json'),observation)
    if action == 'input':
        return _input(job,policy,activation,http,observation,deadline,check)
    if action == 'save':
        receipts = {};partials = []
        for row in policy['noise_jobs']:
            _,token = _job_row(policy,row['id'])
            path = job/'job_receipts'/(token+'.json')
            if path.is_file():
                # A failure during scratch cleanup does not revoke durable bytes.
                receipt = validate_job_commit(job,policy,row['id'],require_cleanup=False)
                receipts[row['id']] = receipt
            elif (job/row['output']).is_dir():
                partials.append(_partial(job,policy,row,http,activation,deadline,check))
        put(job/'NOISE_CAPSULE_INDEX.json',dict(jobs=receipts,partial=partials,policy_sha256=sha(job/POLICY_NAME),
                                             formal_custody_assurance=False))
        entries = _metadata_entries(job,policy)
        full = job/f'NOISE_FULL_{activation["attempt"]}.zip'
        pack(full,entries,policy['return_bound_bytes'],check,member_limit=policy['metadata_review_member_bound'])
        full_record = _upload(job,policy,http,full,deadline,check)
        descriptor = job/'DURABLE_FULL_RETURN.json'
        put(descriptor,dict(full_record,jobs=receipts,partial=partials,formal_custody_assurance=False,
                            result_url='https://drive.google.com/file/d/'+full_record['reference']['object_id']+'/view'))
        # D6 inputs contain designs, targets, traces and provenance; no raw-noise tensor is copied.
        design_entries = {n:p for n,p in entries.items() if '/evaluation/' in n or n in ('NOISE_POLICY.json','ACTIVATION.json','NOISE_CAPSULE_INDEX.json')
                          or n.startswith(('job_receipts/','commits/')) or n.endswith(('/JOB_RESULT.json','/NOISE_READBACK.json','/GPU_READBACK.json'))}
        designs = job/f'NOISE_DESIGNS_{activation["attempt"]}.zip'
        pack(designs,design_entries,policy['design_return_bound_bytes'],check)
        designs_record = _upload(job,policy,http,designs,deadline,check)
        design_descriptor = job/'DURABLE_DESIGN_RETURN.json'
        put(design_descriptor,dict(designs_record,archive=designs.name,noise_draws_repeated=0,native_scoring_pending=True))
        review_entries = {n:p for n,p in entries.items() if p.suffix in ('.json','.jsonl','.py','.txt','.md')}
        review_entries['DURABLE_FULL_RETURN.json'] = descriptor
        review_entries['DURABLE_DESIGN_RETURN.json'] = design_descriptor
        review = job/f'NOISE_REVIEW_{activation["attempt"]}.zip'
        pack(review,review_entries,policy['review_bound_bytes'],check,member_limit=policy['metadata_review_member_bound'])
        review_record = _upload(job,policy,http,review,deadline,check)
        producer = read(job/'output/RESULT.json') if (job/'output/RESULT.json').is_file() else {}
        expected = {r['id'] for r in policy['noise_jobs']}
        complete = (producer.get('complete') is True and producer.get('completed_jobs') == len(expected)
                    and set(producer.get('completed_members',[])) == expected and set(receipts) == expected
                    and producer.get('durable_jobs') == len(expected) and producer.get('GPU_evaluation_complete') is True
                    and producer.get('task_scoring_pending') is True and producer.get('native_solver_calls') == 0
                    and not partials)
        if complete:
            for row in policy['noise_jobs']:
                validate_job_commit(job,policy,row['id'],require_cleanup=True)
                need(_closed(source(job,row['publication_closure'])), 'Per-job publisher closure unresolved')
        saved = dict(complete=complete,saved=True,archive=full.name,reference=full_record['reference'],
                     local_sha256=sha(full),review_archive=review.name,review_reference=review_record['reference'],
                     review_sha256=sha(review),design_archive=designs.name,design_reference=designs_record['reference'],
                     design_sha256=sha(designs),jobs=receipts,partial=partials,GPU_evaluation_complete=complete,
                     task_scoring_pending=True,native_solver_calls=0,scientific_acceptance=False,
                     formal_custody_assurance=False,canonical_worker_advanced=False,
                     result_url='https://drive.google.com/file/d/'+full_record['reference']['object_id']+'/view',
                     review_url='https://drive.google.com/file/d/'+review_record['reference']['object_id']+'/view',
                     design_url='https://drive.google.com/file/d/'+designs_record['reference']['object_id']+'/view')
        put(job/'SAVE_RESULT.json',saved)
        print('DOWNLOAD THIS SMALL REVIEW: '+saved['review_url'],flush=True)
        print('SAVED DESIGNS FOR EXISTING CPU SCORER: '+saved['design_url'],flush=True)
        return saved
    need(_closed(job/'supervision/save/RESULT.json'),'Noise final return publisher closure unresolved')
    saved = read(job/'SAVE_RESULT.json')
    path = job/f'NOISE_SAVED_{activation["attempt"]}.json'
    put(path,dict(saved=saved,publisher_closure=read(job/'supervision/save/RESULT.json'),activation=activation,
                  note='Every job producer and publisher is recorded separately; this last metadata publisher is separate.',
                  formal_custody_assurance=False))
    record = _upload(job,policy,http,path,deadline,check)
    put(job/'CLOSING_RECORD_SAVED.json',dict(record,result_url=saved['result_url'],review_url=saved['review_url'],
                   design_url=saved['design_url'],closure_url='https://drive.google.com/file/d/'+record['reference']['object_id']+'/view'))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['capacity','input','publish','save','closure'])
    parser.add_argument('--job',required=True,type=Path)
    parser.add_argument('--job-id')
    parser.add_argument('--deadline',required=True,type=float)
    args = parser.parse_args()
    transfer(args.job,args.action,args.deadline,args.job_id)

