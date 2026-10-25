"""Injected synthetic supervisor, data-only held-byte bootstrap and adapters.

This is not production authorization or OS sandboxing. Process, capture, claim,
clock and independent acquisition providers are separately accepted boundaries.
No default process provider, filesystem writer or later-state handler exists.
"""
from dataclasses import dataclass
from types import ModuleType

from operatormark.corpus import final_evidence_records as records
from operatormark.corpus import publication_physical_records as publication
from operatormark.corpus import corpus_retention as retention
from operatormark.corpus import state_guard_capture as guard_capture
from operatormark.corpus import state_execution_authority as authority
from operatormark.corpus.source_binding import SourceBytes, verify_execution_source_bytes
from operatormark.corpus.source_import_guard import install_bound_project_imports, validate_bound_import_trace
from operatormark.corpus.state_execution_authority import (
    ClaimResult, LaunchAuthorityError, canonical, check_bytes, decode, digest, enum,
    file_ref, integer, journal_address, make_launch_record, need, object_keys,
    process_identity, qualify, ref, refs, reprepare, sha, text, validate_launch_record,
)


FRAME_MAGIC = b'OPMB1\x00'


@dataclass(frozen=True, slots=True)
class BootstrapInputs:
    payload_length: int
    payload_sha256: str
    maximum_payload_length: int
    input_descriptor: int
    operation: str
    role: str
    source_expectations: object


@dataclass(frozen=True, slots=True)
class BoundRun:
    stdout: bytes
    stderr: bytes
    direct_result: int
    trace: object
    cleanup_failures: tuple


def encode_capsule(*, expected, supplied):
    verified = verify_execution_source_bytes(expected=expected, supplied=supplied)
    value = dict(schema='state_launch_capsule.v1', operation=expected.operation, role=expected.role,
                 entry_point=expected.entry_point,
                 sources=[dict(path=row.path, data_hex=row.data.hex()) for row in verified.sources])
    payload = canonical(value)
    return FRAME_MAGIC + len(payload).to_bytes(8, 'big') + payload


def decode_capsule(*, stream, bootstrap_inputs, descriptor):
    """Expectations are separate trusted bootstrap inputs, never capsule fields."""
    need(type(bootstrap_inputs) is BootstrapInputs, 'independently fixed bootstrap inputs required')
    pin = bootstrap_inputs
    integer(pin.payload_length, 1); integer(pin.maximum_payload_length, 1); digest(pin.payload_sha256)
    need(pin.payload_length <= pin.maximum_payload_length and type(pin.input_descriptor) is int
         and type(descriptor) is int and descriptor == pin.input_descriptor == 0, 'unapproved inherited capsule handle/bound')
    expected = pin.source_expectations
    need((pin.operation, pin.role) == (expected.operation, expected.role), 'bootstrap operation/role mismatch')
    def exact(length):
        chunks = []; remaining = length
        while remaining:
            part = stream.read(remaining)
            need(type(part) is bytes and 0 < len(part) <= remaining, 'capsule truncation/invalid read')
            chunks.append(part); remaining -= len(part)
        return b''.join(chunks)
    header = exact(len(FRAME_MAGIC)+8)
    need(header[:len(FRAME_MAGIC)] == FRAME_MAGIC, 'capsule magic')
    length = int.from_bytes(header[len(FRAME_MAGIC):], 'big')
    need(length == pin.payload_length and length <= pin.maximum_payload_length, 'capsule length binding')
    payload = exact(length)
    need(stream.read(1) == b'', 'capsule trailing bytes')
    need(sha(payload) == pin.payload_sha256, 'independent capsule identity mismatch')
    value = decode(payload)
    object_keys(value, 'schema operation role entry_point sources')
    need(value['schema'] == 'state_launch_capsule.v1' and value['operation'] == pin.operation
         and value['role'] == pin.role and value['entry_point'] == expected.entry_point, 'capsule scope mismatch')
    need(type(value['sources']) is list and len(value['sources']) == len(expected.members), 'capsule source membership')
    supplied = []
    for item, member in zip(value['sources'], expected.members, strict=True):
        object_keys(item, 'path data_hex')
        h = item['data_hex']
        need(item['path'] == member.path and type(h) is str and len(h) == 2*member.file_bytes
             and all(c in '0123456789abcdef' for c in h), 'capsule source framing')
        supplied.append(SourceBytes(item['path'], bytes.fromhex(h)))
    return verify_execution_source_bytes(expected=expected, supplied=tuple(supplied))


def execute_bound_capsule(*, stream, bootstrap_inputs, descriptor, preloaded, mode):
    need(mode == 'synthetic', 'production bound entry disabled')
    verified = decode_capsule(stream=stream, bootstrap_inputs=bootstrap_inputs, descriptor=descriptor)
    session = install_bound_project_imports(verified, expected=bootstrap_inputs.source_expectations, preloaded=preloaded)
    primary = None; result = None; cleanup = []
    try:
        module = session.import_entry()
        need(type(module) is ModuleType, 'exact held entry module required')
        namespace = ModuleType.__getattribute__(module, '__dict__')
        output, error, code = (namespace.get(k) for k in ('SYNTHETIC_STDOUT', 'SYNTHETIC_STDERR', 'SYNTHETIC_EXIT'))
        need(type(output) is bytes and type(error) is bytes and type(code) is int, 'inert binary result contract')
        trace = session.trace(); validate_bound_import_trace(trace, session=session)
        result = BoundRun(output, error, code, trace, ())
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            session.close()
        except BaseException as exc:
            if primary is None:
                raise
            # Side-channel on this exact exception object's dictionary bypasses
            # user attribute hooks without changing cause/context/traceback.
            dictionary = BaseException.__dict__['__dict__'].__get__(primary, type(primary))
            dictionary['_state_execution_cleanup_failures'] = (('source_session_close', exc),)
    return result


@dataclass(frozen=True, slots=True)
class CreationObservation:
    call_status: str
    creation_status: str
    identity: bytes | None
    handle: object | None
    evidence_files: tuple[bytes, ...]
    observation_bytes: bytes | None = None


@dataclass(frozen=True, slots=True)
class StreamObservation:
    data: bytes | None
    eof: bool
    closed: bool
    observation_file: bytes | None


@dataclass(frozen=True, slots=True)
class ProcessClosure:
    direct_exit_code: int | None
    reap_status: str
    stdout: StreamObservation
    stderr: StreamObservation
    evidence_files: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class FailureCapture:
    primary: BaseException | None
    follow_on: tuple[tuple[str, BaseException], ...]
    unavailable: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExecutionCapture:
    creation_status: str
    creation_acknowledgment: str
    creation_identity: bytes | None
    direct_exit_code: int | None
    reap_status: str
    stdout: bytes
    stderr: bytes
    retained_files: tuple[bytes, ...]
    guard_file: bytes | None
    failures: FailureCapture
    resolution: str


def _retain(capture_port, key, data, *, spec):
    slots = [s for s in spec['output_slots'] if s['key'] == key]
    need(len(slots) == 1, 'undeclared capture slot')
    slot = slots[0]
    receipt = capture_port.retain(slot, data)
    need(type(receipt) is bytes, 'capture acknowledgment unavailable')
    result = decode(receipt)
    need(result == file_ref(slot['address'], data), 'capture location/content acknowledgment mismatch')
    # Explicit second observation through the accepted reader port.
    need(capture_port.read(result) == data, 'capture independent readback unavailable/mismatch')
    return receipt


def _journal(store, prepared, kind, data):
    actual = store.publish_exclusive(journal_address(prepared, kind), data)
    need(type(actual) is bytes, 'journal publication acknowledgment unavailable')
    reference = decode(actual)
    validate_launch_record(decode(data), prepared=prepared, actual_file=reference)
    need(store.read(reference) == data, 'journal readback unavailable/mismatch')
    return actual


def _stream(stream, key, capture_port, spec, retained, failures, unavailable):
    need(type(stream) is StreamObservation, 'exact stream observation required')
    need(type(stream.eof) is bool and type(stream.closed) is bool, 'stream closure booleans')
    if stream.observation_file is not None:
        ref(decode(stream.observation_file))
    if stream.data is None:
        unavailable.append(key+'_bytes')
        return canonical(dict(state='unavailable', file=None))
    need(type(stream.data) is bytes, 'binary stream bytes only')
    try:
        f = _retain(capture_port, key, stream.data, spec=spec); retained.append(f)
        complete = stream.eof and stream.closed and stream.observation_file is not None
        if not complete:
            unavailable.append(key+'_closure')
        return canonical(dict(state='complete' if complete else 'partial', file=decode(f)))
    except BaseException as exc:
        failures.append((key+'_retention', exc)); unavailable.append(key+'_retention')
        return canonical(dict(state='unavailable', file=None))


def supervise_once(*, prepared, expected, independent_inputs, claim, claim_store,
                   process_port, capture_port, guard_port, clock_port, on_child_started=None):
    """One create call maximum. Cleanup never erases uncertainty or primary error."""
    prepared = reprepare(prepared, expected=expected, independent_inputs=independent_inputs)
    need(type(claim) is ClaimResult and claim.outcome == 'won' and claim.lease is not None
         and claim.lease.prepared == prepared, 'exclusive claim required')
    spec = decode(expected.specification)
    retained = [claim.lease.reservation_file]; follow = []; unavailable = []
    primary = None; creation = 'unknown'; acknowledgment = 'unavailable'; identity = None
    handle = None; closure = None; guard_file = None; exit_code = None; reap = 'unresolved'
    streams = {k:canonical(dict(state='unavailable',file=None)) for k in ('stdout','stderr')}
    dispatch_file = None; creation_file = None; exit_file = None
    first_observation = None
    try:
        observation = clock_port.observe()
        first_observation = observation
        need(spec['valid_from'] <= observation['observed_utc'] <= spec['valid_until'], 'creation authority expired/not yet valid')
        slot = next(s for s in spec['child_slots'] if s['key'] == expected.slot_key)
        parent = next((s for s in spec['child_slots'] if s['key'] == slot['parent_key']), None)
        parent_id = None if parent is None else sha(b'operatormark-launch-slot-v1\n'+canonical(dict(authority_id=prepared.authority_id,**parent)))
        dispatch = make_launch_record('dispatch', prepared=prepared, prior=decode(claim.lease.reservation_file),
                                     observation=observation, payload=dict(reservation_file=decode(claim.lease.reservation_file),
                                     command_binding_sha256=spec['command_binding_sha256'], source_binding_sha256=spec['source_binding_sha256'], parent_slot_id=parent_id))
        dispatch_file = _journal(claim_store, prepared, 'dispatch', dispatch); retained.append(dispatch_file)
        claim.lease.consume(store=claim_store, dispatch_file=dispatch_file)
        # Any exception once create is invoked remains unknown unless the
        # accepted process boundary returns affirmative, qualified facts.
        result = process_port.create_once(prepared)
        need(type(result) is CreationObservation, 'exact creation facts required')
        if result.handle is not None:
            # Preserve an independently owned handle before validating the rest
            # of an acknowledgment that may be incomplete or contradictory.
            need(process_port.owns(result.handle, prepared) is True, 'unbound process handle')
            handle = result.handle
            if result.identity is not None:
                process_identity(decode(result.identity))
                identity = result.identity; creation = 'started'
        enum(result.call_status, records.CALLS); enum(result.creation_status, ('not_started','started','unknown'))
        need(type(result.evidence_files) is tuple and bool(result.evidence_files), 'creation observation evidence required')
        for f in result.evidence_files:
            ref(decode(f))
        if result.creation_status == 'started':
            process_identity(decode(result.identity))
            need(result.handle is not None and result.call_status != 'not_called', 'created process ownership missing')
            # Save known ownership before subsequent publication/callback can fail.
            creation = 'started'; identity = result.identity
        else:
            need(result.identity is None and result.handle is None, 'unknown/not-started process identity contradiction')
            need(result.creation_status != 'not_started' or result.call_status in ('not_called','returned_failure'), 'false no-child conclusion')
            creation = result.creation_status
        acknowledgment = result.call_status
        if creation != 'started' or result.call_status != 'returned_success':
            raise LaunchAuthorityError('creation not affirmatively acknowledged; preserved unresolved/refusal')
        need(len(result.evidence_files)==1, 'one identity-defining creation observation required')
        observation_ref=decode(result.evidence_files[0])
        need(process_port.read(observation_ref)==result.observation_bytes, 'creation retained readback conflict')
        derived=authority.validate_process_observation(result.observation_bytes,actual_file=observation_ref,
            prepared=prepared,dispatch_file=decode(dispatch_file))
        observed=decode(result.observation_bytes)
        need(derived==decode(identity) and observed['creation_status']==creation and observed['call_status']==acknowledgment,
             'creation acknowledgment/retained observation conflict')
        data = make_launch_record('creation', prepared=prepared, prior=decode(dispatch_file), observation=clock_port.observe(),
                                  payload=dict(dispatch_file=decode(dispatch_file), call_status=result.call_status,
                                  creation_status=creation, process_identity=decode(identity),
                                  observation_files=[decode(f) for f in result.evidence_files], failure_files=[]))
        creation_file = _journal(claim_store, prepared, 'creation', data); retained.append(creation_file)
        if on_child_started is not None:
            on_child_started()
        closure = process_port.capture_and_reap(handle)
    except BaseException as exc:
        primary = exc
        unavailable.append('primary_operation_failed')
    finally:
        if handle is not None and closure is None:
            # Cleanup authority is separate, finite and restricted to this handle.
            try:
                observation = clock_port.observe()
                need(observation['observed_utc'] <= spec['cleanup_until'], 'cleanup authority expired')
                closure, cleanup_failures = process_port.cleanup_owned(handle)
                need(type(cleanup_failures) is tuple, 'ordered cleanup failures required')
                for stage, exc in cleanup_failures:
                    text(stage); need(isinstance(exc, BaseException), 'cleanup exception required')
                    follow.append((stage, exc))
            except BaseException as exc:
                follow.append(('cleanup_owned', exc)); unavailable.append('cleanup_outcome')
    if closure is not None:
        try:
            need(type(closure) is ProcessClosure, 'exact direct process closure required')
            need(closure.direct_exit_code is None or type(closure.direct_exit_code) is int, 'direct exit exact integer')
            enum(closure.reap_status, ('observed_reaped','unresolved','failed'))
            need(type(closure.evidence_files) is tuple, 'exit evidence tuple')
            for f in closure.evidence_files:
                ref(decode(f))
            need(closure.reap_status != 'observed_reaped' or closure.direct_exit_code is not None and bool(closure.evidence_files), 'reaping needs direct observation')
            exit_code = closure.direct_exit_code; reap = closure.reap_status
            for key in ('stdout','stderr'):
                stream = closure.stdout if key == 'stdout' else closure.stderr
                streams[key] = _stream(stream, key, capture_port, spec, retained, follow, unavailable)
            if creation_file is not None:
                need(len(closure.evidence_files)==1, 'one direct exit observation required')
                observed_ref=decode(closure.evidence_files[0]); observed_data=process_port.read(observed_ref)
                derived=authority.validate_process_observation(observed_data,actual_file=observed_ref,prepared=prepared)
                observed=decode(observed_data)
                need(derived==decode(identity) and observed['direct_exit_code']==exit_code and observed['reap_status']==reap,
                     'exit acknowledgment/retained observation conflict')
                data = make_launch_record('exit', prepared=prepared, prior=decode(creation_file), observation=clock_port.observe(),
                                          payload=dict(creation_file=decode(creation_file), process_identity=decode(identity),
                                          direct_exit_code=exit_code, reap_status=reap,
                                          observation_files=[decode(f) for f in closure.evidence_files], failure_files=[]))
                exit_file = _journal(claim_store, prepared, 'exit', data); retained.append(exit_file)
        except BaseException as exc:
            follow.append(('process_closure', exc)); unavailable.append('process_closure')
    else:
        unavailable.append('direct_process_closure')
    try:
        # Guard port must use independent coverage inputs; it cannot infer them
        # from successful cleanup or a favorable child message.
        guard_data = guard_port.close_capture(prepared, identity, exit_code, reap)
        guard_record = records.strict_final_evidence_loads(guard_data)
        need(guard_record['schema'] == records.execution_prefix(spec['binding'])+'state_guard_closure.v1'
             and guard_record['binding'] == spec['binding'], 'guard closure type/stage binding')
        need(identity is not None and guard_record['process_creation_evidence_file'] == decode(identity)['creation_observation_file'],
             'guard/process creation reference mismatch')
        if not all(guard_record['coverage'][k] for k in ('started_before_project_import','continued_through_exit','all_required_hooks_accounted')):
            unavailable.append('guard_lifetime_coverage')
        if any(v['measurement'] != 'measured' or v['value'] != 0 for v in guard_record['forbidden_counts'].values()):
            unavailable.append('forbidden_action_accounting')
        guard_file = _retain(capture_port, 'guards', guard_data, spec=spec); retained.append(guard_file)
    except BaseException as exc:
        follow.append(('guard_capture', exc)); unavailable.append('guard_capture')
    if primary is not None or follow or unavailable:
        # Failures may themselves be unrecordable. Keep actual objects in memory;
        # never create a successful durable closure with invented references.
        try:
            capture_port.retain_failures(FailureCapture(primary, tuple(follow), tuple(unavailable)))
        except BaseException as exc:
            follow.append(('failure_capture', exc)); unavailable.append('failure_capture')
    complete = (primary is None and not follow and not unavailable and creation == 'started'
                and acknowledgment == 'returned_success' and reap == 'observed_reaped' and exit_code == 0
                and all(decode(s)['state'] == 'complete' for s in streams.values()) and guard_file is not None)
    if complete:
        try:
            last = clock_port.observe()
            capture = stage_capture(spec['binding'],clock_epoch_id=last['clock_epoch_id'],
                first_sequence=first_observation['sequence'],last_sequence=last['sequence'],
                started_utc=first_observation['observed_utc'],finished_utc=last['observed_utc'],publication_binding=spec['publication_binding'])
            provisional = ExecutionCapture(creation,acknowledgment,identity,exit_code,reap,streams['stdout'],streams['stderr'],tuple(retained),guard_file,
                                            FailureCapture(None,(),()),'created_closed')
            outcome = adapt_process_outcome(binding=spec['binding'],capture=capture,
                role=_publication_process_role(spec['role']) if spec['binding']['state'] == 'state6' else spec['role'],
                source_closure_file=spec['source_closure_file'],authority_scope_file=spec['authority_scope_file'],
                creation_file=decode(identity)['creation_observation_file'],execution=provisional,failures=[],publication_binding=spec['publication_binding'])
            outcome_file = _retain(capture_port,'process',outcome,spec=spec);retained.append(outcome_file)
            data = make_launch_record('closure',prepared=prepared,prior=decode(exit_file),observation=clock_port.observe(),
                payload=dict(reservation_file=decode(claim.lease.reservation_file),dispatch_file=decode(dispatch_file),creation_file=decode(creation_file),
                exit_file=decode(exit_file),process_outcome_file=decode(outcome_file),guard_closure_file=decode(guard_file),
                surviving_files=[decode(f) for f in retained],failure_files=[],resolution='created_closed'))
            retained.append(_journal(claim_store,prepared,'closure',data))
        except BaseException as exc:
            follow.append(('terminal_capture',exc));unavailable.append('terminal_capture');complete=False
            try:
                capture_port.retain_failures(FailureCapture(primary,tuple(follow),tuple(unavailable)))
            except BaseException as failure:
                follow.append(('terminal_failure_capture',failure));unavailable.append('terminal_failure_capture')
    return ExecutionCapture(creation, acknowledgment, identity, exit_code, reap, streams['stdout'], streams['stderr'],
                            tuple(retained), guard_file, FailureCapture(primary,tuple(follow),tuple(unavailable)),
                            'created_closed' if complete else 'unresolved' if unavailable or creation == 'unknown' else 'failed_preserved')


def stage_capture(binding, *, clock_epoch_id, first_sequence, last_sequence, started_utc, finished_utc,
                  publication_binding=None):
    records.execution_prefix(binding)
    if publication_binding is None:
        domain = (records.execution_domain(binding)+'\n'+binding['state']+'\n'+
                  binding['authorization_file_sha256']+'\n'+binding['state4_authorization_file_sha256'])
    else:
        v2=binding['benchmark']=='darcy_branch_c_v2'
        publication.validate_publication_binding(publication_binding,allow_v2=v2)
        need(publication_binding['profile']==('whole_attempt_same_volume_v2' if v2 else 'whole_attempt_same_volume_v1'),
             'mixed execution/publication versions')
        need(binding['state'] == 'state6' and publication_binding['state6_authorization_file_sha256'] == binding['authorization_file_sha256']
             and publication_binding['state4_authorization_file_sha256'] == binding['state4_authorization_file_sha256'], 'State 6 stage/publication mismatch')
        domain = ('operatormark-publication-invocation-'+('v2' if v2 else 'v1')+'\n'+binding['authorization_file_sha256']+'\n'+publication_binding['publication_plan_file']['file_sha256'])
    capture = dict(invocation_id=sha(domain.encode()),clock_epoch_id=clock_epoch_id,first_sequence=first_sequence,
                   last_sequence=last_sequence,started_utc=started_utc,finished_utc=finished_utc)
    if publication_binding is None:
        records._capture(capture,binding,True)
    else:
        publication._capture(capture,publication_binding)
    return capture


def adapt_lifecycle_event(*, binding, capture, event, pid, child_source_closure_file,
                          supervisor_source_file, authority_scope_file,
                          creation_record_file, direct_exit_code):
    """No synthesized references: all three source/authority files are required."""
    payload = dict(schema=records.execution_prefix(binding)+'process_lifecycle_event.v1',binding=binding,event=event,
                   capture=capture,pid=pid,child_source_closure_file=child_source_closure_file,
                   supervisor_source_file=supervisor_source_file,authority_scope_file=authority_scope_file,
                   creation_record_file=creation_record_file,direct_exit_code=direct_exit_code)
    return records.canonical_final_evidence_bytes(records.build_final_evidence_record('process_lifecycle_event',payload))


def adapt_execution_evidence(*, binding, capture, supplied_payload, execution):
    """Preserve existing strict execution schema; missing scientific/prerequisite
    inputs cannot be manufactured from a successful process-capture result.
    """
    need(type(execution) is ExecutionCapture and type(supplied_payload) is bytes, 'explicit immutable adapter inputs')
    payload = decode(supplied_payload)
    need(payload['binding'] == binding and payload['capture'] == capture, 'execution adapter binding mismatch')
    need(payload['outcome'] != 'completed' or execution.resolution == 'created_closed', 'unresolved capture cannot complete execution')
    return records.canonical_final_evidence_bytes(records.build_final_evidence_record('state_execution_evidence',payload))


def _publication_process_role(launch_role):
    """Translate only State 6 launch roles to the publication record vocabulary."""
    roles = {
        'state6_verifier': 'state6_verification_child',
        'state6_finalizer': 'publication_finalizer',
        # Preserve earlier E2 launch specifications already using record roles.
        'state6_verification_child': 'state6_verification_child',
        'publication_finalizer': 'publication_finalizer',
    }
    need(type(launch_role) is str and launch_role in roles, 'unsupported State 6 launch role')
    return roles[launch_role]


def adapt_process_outcome(*, binding, capture, role, source_closure_file, authority_scope_file,
                          creation_file, execution, failures, publication_binding=None):
    need(type(execution) is ExecutionCapture, 'actual capture required')
    process = None
    if execution.creation_status == 'started':
        ref(creation_file)
        process = dict(pid=decode(execution.creation_identity)['pid'], creation_evidence_file=creation_file)
    payload = dict(binding=binding if publication_binding is None else publication_binding, role=role,
                   capture=capture, source_closure_file=source_closure_file, authority_scope_file=authority_scope_file,
                   creation_status=execution.creation_status, process_identity=process,
                   direct_exit_code=execution.direct_exit_code, stdout=decode(execution.stdout), stderr=decode(execution.stderr),
                   guards_file=None if execution.guard_file is None else decode(execution.guard_file),
                   reap_status=execution.reap_status, failures=failures)
    if publication_binding is None:
        need(binding['state'] in ('state4','state5'), 'State 6 requires publication process adapter')
        payload['schema'] = records.execution_prefix(binding)+'state_process_outcome.v1'
        return records.canonical_final_evidence_bytes(records.build_final_evidence_record('state_process_outcome',payload))
    need(binding['state'] == 'state6', 'publication process is State 6 only')
    # Recompute/check the distinct publication invocation; never relabel State 4/5 bytes.
    expected_capture = stage_capture(binding, publication_binding=publication_binding, **{k:capture[k] for k in capture if k != 'invocation_id'})
    need(capture == expected_capture, 'publication invocation mismatch')
    payload['schema'] = records.execution_prefix(binding)+'publication_process_outcome.v1'
    payload['record_self_sha256'] = sha(canonical(payload))
    return publication.canonical_publication_record_bytes(payload,expected_schema=payload['schema'])


def additional_retention_slots(*, prepared, source_file, source_closure_file, existing_addresses, journal_inventory):
    """Catalog additional slots only, explicit owner independent of component."""
    spec = decode(prepared.expected.specification); result = []
    ref(source_file); ref(source_closure_file)
    need(type(existing_addresses) is tuple, 'independent existing product inventory required')
    items=spec['output_slots']+authority.known_journal_slots(prepared,journal_inventory)
    locations=[]
    for item in items:
        location=(item['address']['root_binding_id'],item['address']['relative_path'].casefold())
        need(location not in locations, 'duplicate physical retention product')
        locations.append(location)
    for item in items:
        if canonical(item['address']) in existing_addresses:
            continue
        a = item['address']; policy = item['writer']
        slot = dict(slot_id='stage_execution:'+a['root_binding_id']+':'+a['relative_path'], component='stage_execution',
                    address=a, writer=dict(source_file=source_file,function='supervise_once',source_closure_file=source_closure_file,write_policy=policy),
                    representation='loose_file',owner=item['owner'],controls_subset=False,
                    condition=dict(kind='operation_reached',source_file=source_file,selector=item['key']),
                    creating_address=dict(a,relative_path=a['relative_path']+'.__creating__') if policy == 'canonical_exclusive_preserve_v1' else None,
                    content_profile='raw_bytes',domain=dict(classification='unresolved',rule='unresolved',input_files=[],prerequisite_files=[],
                    parameters=dict(reason='population_missing',required_evidence_roles=['writer_derived_bound'])))
        retention._slot(slot)
        result.append(slot)
    return tuple(canonical(x) for x in result)


def _exception_descriptor(exc, name):
    return BaseException.__dict__[name].__get__(exc, type(exc))


def capture_exception_graph(*, primary, cleanup_failures, message_captures, capture_port, binding, capture, operation_id):
    """Never invokes exception __getattribute__, __str__, __repr__ or linecache."""
    need(isinstance(primary, BaseException) and type(cleanup_failures) is tuple and type(message_captures) is tuple, 'exception capture inputs')
    need(all(type(row) is tuple and len(row) == 2 for row in cleanup_failures + message_captures), 'exact exception input pairs')
    identities = {}; nodes = []; failures = []; pending_roots = [primary]
    close = []
    for stage, exc in cleanup_failures:
        need(stage == 'close' and isinstance(exc, BaseException) and exc is not primary, 'only distinct close failures fit existing graph schema')
        pending_roots.append(exc)
    def unavailable(kind='result_unavailable'):
        failures.append(dict(kind=kind,native_error_hex=None,exception_file=None,affected_addresses=[]))
        return dict(state='unavailable',file=None)
    def capture_bytes(key, data):
        if data is None:
            return unavailable()
        try:
            need(type(data) is bytes, 'trusted exact message bytes required')
            f = capture_port.retain_exception_bytes(key,data)
            need(type(f) is bytes, 'exception capture reference unavailable')
            reference = decode(f); check_bytes(reference,data)
            return dict(state='complete',file=reference)
        except BaseException:
            return unavailable('retention_failed')
    def visit(exc):
        if exc is None:
            return None
        identifier = identities.get(id(exc))
        if identifier is not None:
            return identifier
        identifier = 'e'+str(len(nodes)); identities[id(exc)] = identifier
        cls = type(exc)
        # Reading a metaclass's attributes can invoke descriptors even when its
        # __getattribute__ is bypassed. Reject nonstandard metaclasses outright.
        need(type(cls) is type, 'unsafe exception metaclass')
        namespace = type.__getattribute__(cls,'__dict__')
        module = namespace.get('__module__', 'builtins'); qualname = type.__getattribute__(cls,'__qualname__')
        need(type(module) is str and type(qualname) is str, 'unavailable exception class metadata')
        text(module); text(qualname)
        tb = _exception_descriptor(exc,'__traceback__'); frames = []
        while tb is not None:
            code = tb.tb_frame.f_code
            # Code-object metadata only; no frame globals/locals or source lines.
            frames.append(dict(function=code.co_name,filename=code.co_filename,line=tb.tb_lineno))
            tb = tb.tb_next
        messages = [data for obj,data in message_captures if obj is exc]
        message = messages[0] if len(messages)==1 else None
        row = dict(exception_id=identifier,class_module=module,class_qualname=qualname,
                   message_bytes=capture_bytes(identifier+'.message',message),
                   traceback_bytes=capture_bytes(identifier+'.traceback',canonical(frames)),
                   cause_id=None,context_id=None,suppress_context=_exception_descriptor(exc,'__suppress_context__'))
        nodes.append(row)
        row['cause_id'] = visit(_exception_descriptor(exc,'__cause__'))
        row['context_id'] = visit(_exception_descriptor(exc,'__context__'))
        return identifier
    visit(primary)
    for ordinal, (_,exc) in enumerate(cleanup_failures):
        close.append(dict(ordinal=ordinal,stage='close',exception_id=visit(exc)))
    payload = dict(schema=records.execution_prefix(binding)+'writer_exception_graph.v1',binding=binding,capture=capture,operation_id=operation_id,
                   primary_exception_id='e0',exceptions=nodes,ordered_cleanup_failures=close,capture_failures=failures)
    return records.canonical_final_evidence_bytes(records.build_final_evidence_record('writer_exception_graph',payload))


@dataclass(frozen=True, slots=True)
class ExecutionComparison:
    status: str
    facts: bytes
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExecutionVerificationExpectations:
    products: bytes
    launches: tuple
    launch_inputs: tuple
    graph: bytes


def _verify_graph(expected, independent_files):
    need(type(expected) is ExecutionVerificationExpectations, 'independent launch graph expectations required')
    need(type(expected.launches) is tuple and bool(expected.launches) and type(expected.launch_inputs) is tuple
         and len(expected.launch_inputs)==len(expected.launches), 'complete independent launch inputs required')
    graph=decode(expected.graph);object_keys(graph,'selected_slot_id capture process_inventory_file slots')
    products=decode(expected.products)
    need(graph['selected_slot_id']==products['slot_id'] and graph['capture']==products['capture'], 'selected graph binding conflict')
    need(type(graph['slots']) is list,'finite graph descriptors required')
    prepared=[authority.prepare_launch(expected=x,independent_inputs=y) for x,y in zip(expected.launches,expected.launch_inputs)]
    by_id={p.slot_id:p for p in prepared}
    need(len(by_id)==len(prepared),'duplicate launch expectation')
    declared={}; authorities={}
    for p in prepared:
        spec=decode(p.expected.specification)
        if p.authority_id in authorities:need(authorities[p.authority_id]==spec['child_slots'],'inconsistent accepted authority graph')
        authorities[p.authority_id]=spec['child_slots']
        for slot in spec['child_slots']:
            identity=sha(b'operatormark-launch-slot-v1\n'+canonical(dict(authority_id=p.authority_id,**slot)))
            if identity in declared:need(declared[identity]==slot,'conflicting finite graph declaration')
            declared[identity]=slot
    need(set(declared)==set(by_id),'omitted/unauthorized finite launch member')
    histories={h['slot_id']:h for h in graph['slots']}
    need(len(histories)==len(graph['slots']) and set(histories)==set(by_id), 'history graph membership conflict')
    physical={}
    for h in histories.values():
        bound=list(h['journals'].values())+h['recoveries']+list(h['streams'].values())+[h[k] for k in
            ('creation_observation_file','exit_observation_file','process_file','guard_file','inventory_file','ownership_file','consumption_file')]
        for f in bound:
            ref(f);location=(f['root_binding_id'],f['relative_path'].casefold())
            need(location not in physical or physical[location]==f,'conflicting physical graph evidence')
            physical[location]=f
    diagnostics=[]; identities={}; rows=[]
    for p in prepared:
        h=histories[p.slot_id];spec=decode(p.expected.specification)
        need(h['capture']==graph['capture'], 'graph capture conflict')
        journal_refs=list(h['journals'].values())+h['recoveries']
        journal_root=journal_address(p,'reservation')['root_binding_id'];prefix=p.authority_id+'/'+p.slot_id+'/'
        journal=tuple(x for x in independent_files if x.scope=='launch_journal' and
            (decode(x.file) in journal_refs or decode(x.file)['root_binding_id']==journal_root and decode(x.file)['relative_path'].startswith(prefix)))
        result=authority.inspect_launch_history(expected=p.expected,independent_inputs=p.independent_inputs,
            independent_records=journal,independent_observations=independent_files,history=canonical(h))
        diagnostics.extend(p.slot_id+':'+d for d in result.diagnostics)
        try:
            raw=qualify(h['creation_observation_file'],independent_files,validators=p.expected.validator_files,
                scope='process_observation',first=h['capture']['started_utc'],last=h['capture']['finished_utc'])
            identity=authority.validate_process_observation(raw,actual_file=h['creation_observation_file'],prepared=p,dispatch_file=h['journals']['dispatch'])
            row=decode(raw);identities[p.slot_id]=identity
            rows.append(dict(slot_id=p.slot_id,operation=spec['operation'],role=spec['role'],parent_slot_id=row['parent_slot_id'],
                parent_process_identity=row['parent_process_identity'],process_identity=identity,creation_file=h['creation_observation_file'],exit_file=h['exit_observation_file']))
        except LaunchAuthorityError:diagnostics.append(p.slot_id+':graph_identity_unavailable')
    for row in rows:
        parent=row['parent_slot_id']
        need((parent is None and row['parent_process_identity'] is None) or parent in identities and row['parent_process_identity']==identities[parent], 'finite graph parent identity conflict')
        if parent is not None:
            parent_launch=by_id[parent];ph=histories[parent];ch=histories[row['slot_id']]
            try:
                def observation(f,p):
                    return decode(qualify(f,independent_files,validators=p.expected.validator_files,scope='process_observation',
                        first=graph['capture']['started_utc'],last=graph['capture']['finished_utc']))['observation']
                start=observation(ph['creation_observation_file'],parent_launch)
                end=observation(ph['exit_observation_file'],parent_launch)
                child=observation(ch['creation_observation_file'],by_id[row['slot_id']])
                need(start['observed_utc']<=child['observed_utc']<=end['observed_utc'], 'parent/child lifetime chronology conflict')
                if start['clock_epoch_id']==child['clock_epoch_id']==end['clock_epoch_id']:
                    need(start['sequence']<child['sequence']<end['sequence'],'parent/child sequence chronology conflict')
            except LaunchAuthorityError:diagnostics.append('parent_lifetime_unavailable_or_conflict')
    need(len({canonical(r['process_identity']) for r in rows})==len(rows),'aliased process graph identity')
    for index,left in enumerate(rows):
        for right in rows[index+1:]:
            if left['process_identity'] is not None and right['process_identity'] is not None and left['process_identity']['pid']==right['process_identity']['pid']:
                # Without an independently established non-overlap, PID reuse
                # is ambiguous, not proof of two simultaneously owned children.
                diagnostics.append('process_pid_reuse_requires_nonoverlap_evidence')
    selected=histories[graph['selected_slot_id']]
    need(products['products']==dict(creation=selected['creation_observation_file'],exit=selected['exit_observation_file'],
         stdout=selected['streams']['stdout'],stderr=selected['streams']['stderr'],guards=selected['guard_file'])
         and products['stream_observations']==selected['stream_observations'] and products['guard_expectations']==selected['guard_expectations'], 'product/graph mapping substitution')
    p=by_id[graph['selected_slot_id']]
    need(tuple(canonical(f) for f in products['validator_files'])==p.expected.validator_files,'selected validator substitution')
    try:
        raw=qualify(graph['process_inventory_file'],independent_files,validators=p.expected.validator_files,
            scope='process_inventory',first=graph['capture']['started_utc'],last=graph['capture']['finished_utc'])
        inv=decode(raw);object_keys(inv,'capture enumeration processes failure_files')
        need(inv['capture']==graph['capture'] and type(inv['processes']) is list,'process enumeration binding conflict')
        refs(inv['failure_files'])
        if inv['enumeration']!='complete' or inv['failure_files']:diagnostics.append('process_population_unavailable_or_failed')
        need(sorted(inv['processes'],key=canonical)==sorted(rows,key=canonical),'omitted/unauthorized observed process graph')
    except LaunchAuthorityError as exc:
        diagnostics.append('process_inventory_unavailable_or_conflict')
    return diagnostics, by_id[graph['selected_slot_id']]


def compare_execution_capture(*, expected, candidate_files, independent_files, upstream_admissions):
    """Independent facts from exact qualified records, not producer objects.

    expected is immutable canonical {slot_id, validator_files, capture, products,
    stream_observations, guard_expectations}.
products maps creation/exit/stdout/stderr/guards to independent FileRefs. This
private report cannot confer production acceptance. No producer import needed.
"""
    graph_diagnostics, selected_launch = _verify_graph(expected, independent_files)
    spec = decode(expected.products)
    object_keys(spec, 'slot_id validator_files capture products stream_observations guard_expectations')
    digest(spec['slot_id']); records._capture(spec['capture'])
    object_keys(spec['products'], 'creation exit stdout stderr guards')
    object_keys(spec['stream_observations'], 'stdout stderr')
    object_keys(spec['guard_expectations'], 'specification validator_files')
    need(type(candidate_files) is bytes, 'candidate facts canonical bytes required')
    proposed = decode(candidate_files)
    diagnostics = list(graph_diagnostics); facts = {}
    need(type(upstream_admissions) is tuple, 'independent admissions required')
    validators = tuple(canonical(f) for f in spec['validator_files'])
    for role, reference in spec['products'].items():
        ref(reference)
        matches = [a for a in upstream_admissions if type(a) is bytes and decode(a).get('role') == role]
        try:
            need(len(matches) == 1, 'missing/duplicate product admission')
            admission = decode(matches[0]); object_keys(admission,'role file validator_file scope first_utc last_utc status')
            need(admission['file'] == reference and canonical(admission['validator_file']) in validators
                 and admission['scope'] == 'execution_'+role and admission['status'] == 'accepted'
                 and admission['first_utc'] <= spec['capture']['started_utc']
                 and admission['last_utc'] >= spec['capture']['finished_utc'], 'execution admission binding')
            data = qualify(reference,independent_files,validators=validators,scope='execution_'+role,
                           first=spec['capture']['started_utc'],last=spec['capture']['finished_utc'])
            if role in ('stdout','stderr'):
                observation = decode(qualify(spec['stream_observations'][role],independent_files,validators=validators,
                                     scope='stream_observation',first=spec['capture']['started_utc'],last=spec['capture']['finished_utc']))
                object_keys(observation,'slot_id stream file eof closed capture process_identity')
                need(observation['slot_id'] == spec['slot_id'] and observation['stream'] == role
                     and observation['file'] == reference and observation['capture'] == spec['capture'], 'stream identity/interval binding')
                need(type(observation['eof']) is bool and type(observation['closed']) is bool, 'stream closure facts')
                complete = observation['eof'] and observation['closed']
                if not complete:
                    diagnostics.append(role+'_closure_unavailable')
                facts[role] = dict(state='complete' if complete else 'partial',file=reference)
            elif role == 'guards':
                record = records.strict_final_evidence_loads(data)
                need(record['schema'] == records.execution_prefix(decode(selected_launch.expected.specification)['binding'])+'state_guard_closure.v1', 'guard product schema')
                gx = spec['guard_expectations']
                recomputed = guard_capture.derive_guard_capture(expected=guard_capture.GuardExpectations(
                    canonical(gx['specification']),tuple(canonical(f) for f in gx['validator_files'])),independent_files=independent_files)
                need(data == recomputed.closure, 'guard product differs from independent event recomputation')
                if recomputed.diagnostics:
                    diagnostics.extend(recomputed.diagnostics)
                facts[role] = record
            else:
                record = decode(data)
                need(record['slot_id'] == spec['slot_id'] and record['kind'] == role, 'independent process observation slot/kind')
                if role == 'creation':
                    identity=authority.validate_process_observation(data,actual_file=reference,prepared=selected_launch,
                        dispatch_file=decode(expected.graph)['slots'][next(i for i,h in enumerate(decode(expected.graph)['slots']) if h['slot_id']==spec['slot_id'])]['journals']['dispatch'])
                    record=dict(record,process_identity=identity)
                    enum(record['call_status'],records.CALLS); enum(record['creation_status'],('not_started','started','unknown'))
                    process_identity(record['process_identity'],record['creation_status']!='started')
                    need(record['creation_status'] == 'started' or record['process_identity'] is None, 'unknown process cannot establish PID')
                    need(record['creation_status'] != 'not_started' or record['call_status'] in ('not_called','returned_failure'), 'creation refusal contradiction')
                    need(record['creation_status'] != 'started' or record['call_status'] != 'not_called', 'creation effect contradiction')
                else:
                    authority.validate_process_observation(data,actual_file=reference,prepared=selected_launch)
                    process_identity(record['process_identity'])
                    need(record['direct_exit_code'] is None or type(record['direct_exit_code']) is int, 'independent direct exit')
                    enum(record['reap_status'],('observed_reaped','unresolved','failed'))
                    need(record['reap_status'] != 'observed_reaped' or record['direct_exit_code'] is not None, 'reaping without direct exit')
                facts[role] = record
        except (LaunchAuthorityError,records.FinalEvidenceRecordError):
            diagnostics.append(role+'_evidence_unavailable_or_invalid'); facts[role] = None
    if facts.get('creation') and facts.get('exit'):
        need(facts['creation']['process_identity'] == facts['exit']['process_identity'], 'independent process identity disagreement')
    need(proposed == facts, 'execution capture recomputation mismatch')
    return ExecutionComparison('incomplete' if diagnostics else 'corroborated',canonical(facts),tuple(diagnostics))
