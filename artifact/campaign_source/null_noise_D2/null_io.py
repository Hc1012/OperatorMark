"""Retained byte transport helpers, extracted without numerical changes.
Original implementation: noise_remote.py, independently pinned in this package.
These helpers do not issue activation, a runtime gate, or scientific acceptance.
"""
from pathlib import Path, PurePosixPath
import hashlib, json, os, stat, shutil, sys, time, zipfile
from urllib.parse import urlencode

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
