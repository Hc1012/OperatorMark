"""File-backed scientific campaign entry. Disabled without external authority.

The index hash is an independently reviewed expectation, NOT an acceptance
issuer. The index contains descriptors for existing receipts, never credentials
or embedded scientific tensors. Missing later-stage descriptors pause through
RecordEvidence; they are not prerequisites for producing clean owners.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ''):
    sys.path[:0] = [str(SOURCE_ROOT / 'tools'), str(SOURCE_ROOT)]


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError('duplicate index key: ' + k)
        result[k] = v
    return result


def load_index(path, expected):
    path = Path(path).resolve()
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('independently reviewed index SHA-256 required')
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('index is metadata only; oversized index refused')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('operating index drift')
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    if value.get('schema') != 'msfinal.operating_index.v1':
        raise ValueError('operating index schema')
    if Path(value['source_root']).resolve() != SOURCE_ROOT:
        raise ValueError('entry source root substitution')
    # Literal source inventory only: do not import numerical dependencies or
    # inspect a material payload during the default metadata preflight.
    pins = value.get('source_pins', {})
    if not isinstance(pins, dict) or 'tools/msfinal/operating_entry.py' not in pins:
        raise ValueError('complete candidate source inventory required')
    import ast
    tree = ast.parse((SOURCE_ROOT / 'tools/msfinal/backend.py').read_text(encoding='utf-8'))
    required = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                       t.id == 'REQUIRED_SOURCES' for t in n.targets))
    if not set(required) <= set(pins):
        raise ValueError('candidate source inventory coverage missing')
    for name, h in pins.items():
        target = (SOURCE_ROOT / name).resolve()
        if not target.is_relative_to(SOURCE_ROOT) or digest(target) != h:
            raise ValueError('candidate source drift: ' + name)
    return value, path.parent


class Decoder:
    """Closed set of existing consumer types; no pickle/eval/plugin factory."""
    def __init__(self, base):
        self.base = Path(base)

    def path(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError('explicit path required')
        p = Path(value)
        return (p if p.is_absolute() else self.base / p).resolve()

    def decode(self, value):
        if isinstance(value, list):
            return [self.decode(v) for v in value]
        if not isinstance(value, dict):
            return value
        if '$path' in value:
            if set(value) != {'$path'}: raise ValueError('path descriptor fields')
            return self.path(value['$path'])
        if '$tuple' in value:
            if set(value) != {'$tuple'}: raise ValueError('tuple descriptor fields')
            return tuple(self.decode(v) for v in value['$tuple'])
        if '$record' in value:
            if set(value) != {'$record'}: raise ValueError('record descriptor fields')
            from msfinal.backend import AcceptedRecord
            from msfinal.scientific_admission import ScientificRecord
            r = value['$record']; p = self.path(r['path'])
            raw = p.read_bytes()
            if hashlib.sha256(raw).hexdigest() != r['sha256']:
                raise ValueError('independently selected record drift: ' + str(p))
            if r['kind'] == 'accepted' and set(r) == {'kind','path','sha256'}:
                result = AcceptedRecord(p, r['sha256'], raw)
            elif r['kind'] == 'scientific' and set(r) == {'kind','path','sha256','corpus'}:
                result = ScientificRecord(p, r['sha256'], raw, self.decode(r['corpus']))
            else:
                raise ValueError('record route must be explicit; no formal/scientific conversion')
            result.read()
            return result
        if '$type' in value:
            if set(value) != {'$type','fields'}: raise ValueError('typed descriptor fields')
            from msfinal import backend, owners, attacks, expanded, final_contracts, worker
            from operatormark_corpus_v2.scientific import CorpusAdmission
            from msfinal.portable_corpus import PortableCorpus
            types = {'TrainingInputs':owners.TrainingInputs, 'Prerequisites':backend.Prerequisites,
                     'Parent':attacks.Parent, 'AttackRequest':attacks.Request,
                     'ExpandedRequest':expanded.Request, 'ContractRequest':final_contracts.Request,
                     'MarkedInputs':worker.MarkedInputs, 'CorpusAdmission':CorpusAdmission,
                     'PortableCorpus':PortableCorpus}
            fields = self.decode(value['fields'])
            if value['$type'] == 'Admission':
                if set(fields) != {'prerequisites'}: raise ValueError('admission fields')
                return backend.admit(fields['prerequisites'])
            if value['$type'] not in types: raise ValueError('unsupported consumer type')
            return types[value['$type']](**fields)
        if any(k.startswith('$') for k in value): raise ValueError('unknown descriptor tag')
        return {k:self.decode(v) for k,v in value.items()}


def evidence(rows, decoder):
    from msfinal.worker import RecordEvidence, MissingEvidence
    class FileEvidence(RecordEvidence):
        def __init__(self):
            super().__init__({})
            self.pending = {}
            for row in rows:
                if set(row) != {'key','value'}: raise ValueError('evidence entry fields')
                key = decoder.decode(row['key'])
                if key in self.pending: raise ValueError('duplicate evidence bridge entry')
                self.pending[key] = row['value']
        def get(self, key):
            if key in self.records: return super().get(key)
            if key not in self.pending: raise MissingEvidence('missing independent entry: ' + repr(key))
            # Decode at the actual boundary, not before owner production.
            try: return decoder.decode(self.pending[key])
            except FileNotFoundError as e: raise MissingEvidence(str(e)) from e
    return FileEvidence()


def supervisor(index, decoder):
    p = decoder.path(index['supervisor']['path'])
    if digest(p) != index['supervisor']['sha256']:
        raise ValueError('owned supervisor drift')
    spec = importlib.util.spec_from_file_location('bound_msfinal_supervisor', p)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def publication(index, decoder):
    """Existing publication port backed by independent, stage-specific records.

    Called by the bound operations.closed_stage/finalize_closed_stage host.
    Absent future closure expectations are not created from producer flags.
    """
    from msfinal.operations import PublicationEvidence
    from msfinal.worker import MissingEvidence
    class FilePublication(PublicationEvidence):
        def selected(self, name):
            descriptor=index.get('publication',{}).get(name)
            if descriptor is None: raise MissingEvidence('independent publication entry missing: '+name)
            return decoder.decode(descriptor)
        def closure(self, path):
            record=self.selected('closure')
            if record.path.resolve()!=Path(path).resolve(): raise ValueError('publication closure path differs')
            return record
        def permissions(self, root, destination, closure):
            return self.selected('export_permission'),self.selected('transfer_permission')
    return FilePublication()


def transport(index, decoder, session, role):
    from msfinal.drive_transport import GoogleDriveTransfer, RefreshingAuthorizedUserToken, RefreshingMetadataToken
    spec=index['transport']
    if digest(SOURCE_ROOT/'tools/msfinal/drive_transport.py') != spec['source_sha256']:
        raise ValueError('transport source drift')
    auth=spec.get('authentication')
    if auth is not None:
        if auth.get('mode')=='colab_metadata_refresh_v1':
            if set(auth)!={'mode','metadata_host'}:
                raise ValueError('explicit metadata authentication policy required')
            token=RefreshingMetadataToken(auth['metadata_host'])
        elif auth.get('mode')=='authorized_user_refresh_v1':
            if set(auth)!={'mode','credential_path_environment'}:
                raise ValueError('explicit refresh authentication policy required')
            name=auth['credential_path_environment']
            if not isinstance(name,str) or not name.startswith('OPERATORMARK_'):
                raise ValueError('explicit credential path environment required')
            path=os.environ.get(name)
            if not path:raise ValueError('existing refreshable credential path unavailable')
            token=RefreshingAuthorizedUserToken(path)
        else:raise ValueError('explicit refresh authentication policy required')
        # Establish refreshability before training, not after a long worker run.
        token(session.value['finalization_deadline'])
    else:
        name=spec['token_environment']
        if not isinstance(name,str) or not name.startswith('OPERATORMARK_'):
            raise ValueError('explicit existing OAuth environment provider required')
        def token(deadline):
            if time.time()>=deadline: raise TimeoutError('original transport deadline')
            value=os.environ.get(name)
            if not value: raise ValueError('existing authenticated provider unavailable; no login/installation fallback')
            return value
    remote=GoogleDriveTransfer(spec['folder_id'],token,decoder.path(spec['log_roots'][role]),
        byte_limit=spec['byte_limit'],metadata_byte_limit=spec['metadata_byte_limit'])
    if auth is not None:
        remote.event('credential_ready',mechanism=auth['mode'],refreshes=token.refreshes,
                     expires_at=token.expires,credentials_retained=False)
    return remote


def phase_store(index, decoder, session, role, *, remote=None):
    if index.get('phase_storage') is None:
        return None
    from msfinal.phase_storage import PhaseStore
    remote=remote if remote is not None else transport(index,decoder,session,role)
    def check():
        if session.record.read()!=session.value or time.time()>=session.value['finalization_deadline']:
            raise TimeoutError('source session drift/finalization deadline')
    from msfinal.bounded_stage import worker_binding
    binding=worker_binding(session)
    return PhaseStore(decoder.decode(index['phase_storage']),remote,binding,check=check,clock=time.time)


def child_command(index, decoder, index_path, index_sha, session_path, action):
    return [str(decoder.path(index['worker_interpreter'])),'-B',str(Path(__file__).resolve()),
            action,'--index',str(Path(index_path).resolve()),'--index-sha',index_sha,
            '--session',str(decoder.path(session_path)),'--execute']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('preflight','supervise','supervise-stage','stage-host','persist-closure','worker'))
    p.add_argument('--index', required=True); p.add_argument('--index-sha', required=True)
    p.add_argument('--session'); p.add_argument('--session-sha')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv); index, base = load_index(a.index, a.index_sha)
    if a.action == 'preflight':
        print(json.dumps(dict(source_inventory_verified=True, material_payloads_opened=False,
            execution_authorized=False, missing_pre_campaign=index.get('missing_pre_campaign',[]),
            statement='metadata only; no runtime/capacity/scientific acceptance')))
        return 0
    if not a.execute or index.get('execution_authorized') is not True:
        raise ValueError('UNSTARTED: explicit independently bound execution authority required')
    if os.name != 'posix' or not sys.platform.startswith('linux'):
        raise ValueError('campaign worker/merged-view deployment is Linux; CPU scoring has its separate Windows entry')
    from msfinal import session as S, operations as O
    from msdev.records import read
    d = Decoder(base); config = d.decode(index['configuration'])
    # Authority is separate from the data/source index: putting its own hash in
    # its command would create a circular binding. The parent takes the externally
    # selected hash; the child inherits it from the parent's durable launch marker.
    if not a.session: raise ValueError('separate externally accepted session required')
    used=None
    expected_session=a.session_sha
    if a.action in ('worker','stage-host','persist-closure'):
        if a.session_sha is not None: raise ValueError('worker uses owning launcher session binding')
        marker=(d.path(index['launch_root'])/'LAUNCH_USED.json' if a.action=='worker'
                else d.path(index['operational_root'])/'OPERATION_USED.json')
        used=read(marker)
        expected_session=used['session_sha256']
    if expected_session is None: raise ValueError('independently selected session identity required')
    sr=d.decode({'$record':dict(kind='accepted',path=a.session,sha256=expected_session)})
    session = S.Session(sr, config.sha256,
                        digest(SOURCE_ROOT/'tools/msfinal/worker.py'),
                        previous=d.decode(index.get('previous_session')),
                        stage_amendment=d.decode(index.get('stage_amendment')),
                        finalization_only=a.action=='persist-closure')
    if Path(session.value['launch_root']).resolve()!=d.path(index['launch_root']):
        raise ValueError('bound launch directory drift')
    expected=child_command(index,d,a.index,a.index_sha,a.session,'worker')
    if session.value['command'] != expected:
        raise ValueError('exact accepted operating worker command differs')
    if session.stage_policy is not None:
        roster=['owners/'+n for n in config.read()['owners']]
        if session.stage_policy['phases'][0] not in roster:
            raise ValueError('bounded stage must be an existing configured clean-owner phase')
        if index.get('resume') is None and session.stage_policy['phases']!=roster[:1]:
            raise ValueError('initial bounded stage must be the first configured owner phase')
    if a.action=='persist-closure':
        expected=child_command(index,d,a.index,a.index_sha,a.session,'persist-closure')
        host=child_command(index,d,a.index,a.index_sha,a.session,'stage-host')
        if session.value.get('closure_persistence_command')!=expected or used.get('command')!=host:
            raise ValueError('bound owned metadata-persistence command differs')
        O.persist_operation_closure(session,transport(index,d,session,'closure'))
        return 0
    if a.action in ('supervise-stage','stage-host'):
        host=child_command(index,d,a.index,a.index_sha,a.session,'stage-host')
        if (session.value.get('operational_command')!=host
                or Path(session.value['operational_root']).resolve()!=d.path(index['operational_root'])):
            raise ValueError('exact bound publication host command/root differs')
        if a.action=='supervise-stage':
            result=O.supervise_closed_stage(session,supervisor(index,d))
            print(json.dumps(result,sort_keys=True));return 0
        if used.get('command')!=host or used.get('session_sha256')!=session.record.sha256:
            raise ValueError('owning operational supervisor marker required')
        from msfinal.stage_publication import HostPublication
        remote=transport(index,d,session,'publisher')
        destination=d.path(index['export_root'])
        store=phase_store(index,d,session,'publisher',remote=remote)
        publisher=HostPublication(d.decode(index['publication_policy']),session,remote,destination,phase_store=store)
        from msfinal.bounded_stage import capacity
        capacity(d.decode(index['stage_capacity']),session,destination,remote.binding)
        O.closed_stage(session,supervisor(index,d),remote,publisher,destination,
            phase_store=store,
            restore_permission=d.decode(index.get('restore_permission')),
            restore_staging=d.decode(index.get('restore_staging')))
        return 0
    if a.action == 'supervise':
        result= S.supervise(session,session.value['launch_root'],supervisor(index,d))
        if not S.closure_ok(result) or result.get('timed_out') or result.get('returncode') not in (0,5):
            raise RuntimeError('owned worker failed/unresolved; preserve evidence, no retry')
        store=phase_store(index,d,session,'closer')
        S.close_result(session,session.value['worker_root'],
            Path(session.value['launch_root'])/'supervisor/RESULT.json',
            Path(session.value['launch_root'])/'CLOSURE.json',phase_store=store)
        print('CLOSED: independent publication/custody is still required before releasing this runtime')
        return 5
    if used.get('session_sha256') != session.record.sha256 or used.get('command') != session.value['command']:
        raise ValueError('existing owned launcher marker/command required')
    return O.worker_entry(config,session,evidence(index.get('evidence',[]),d),
                          d.decode(index['training_inputs']),resume=d.decode(index.get('resume')),
                          phase_store=phase_store(index,d,session,'worker'))


if __name__ == '__main__':
    raise SystemExit(main())
