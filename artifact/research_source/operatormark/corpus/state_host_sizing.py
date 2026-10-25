"""Source-derived Windows host record sizes. Arithmetic, never admission.

The caller supplies actual selected strings/objects, not guessed byte caps.
Adopted D bounds first-foreign-entry evidence; premise P is NOT extended here.
"""
from pathlib import Path
import builtins
import json
from . import state_execution_authority as a, state_provider_bindings as p

U32=2**32-1
U64=2**64-1
HEX='f'*64


def encoded_worst_character(encoder):
    # Canonical UTF-8 JSON: ordinary scalars use <=4 bytes; backslash/quote and
    # named controls use 2; remaining C0 controls use six-byte \u00XX escapes.
    candidates=('\0','\x01','\b','\n','"','\\','a','\u07ff','\uffff','\U0010ffff')
    return max(candidates,key=lambda c:len(encoder(c)))


def json_encoding(value):
    """The actual canonical serializer, without pretending sizing is admission.

    a.canonical first rejects controls/non-NFC text. Our upper envelope includes
    the serializer's six-byte escapes conservatively; the real validator stays
    unchanged and can still make an actual error record unavailable.
    """
    return json.dumps(value,ensure_ascii=False,sort_keys=True,
                      separators=(',',':'),allow_nan=False).encode('utf-8')


def _size(value):return len(json_encoding(value))


def _job():
    # NativeBindings.Members: two DWORDs and eight 64-bit ULONG_PTRs on the
    # pinned 64-bit Windows interpreter; accounting buffer is 48 bytes.
    return dict(members_query_ok=False,members_error=U32,accounting_query_ok=False,
        accounting_error=U32,members_raw='f'*(72*2),accounting_raw='f'*96,
        assigned=U32,listed=U32,members=[U64]*8,total_page_fault_count=U32,
        total_processes=U32,active_processes=U32,limit_terminated_processes=U32)


def _stream():
    from .bounded_stream import QUOTA_BYTES,MAX_COUNTER
    return dict(quota_bytes=QUOTA_BYTES,observed_bytes=MAX_COUNTER,
        observed_bytes_lower_bound=MAX_COUNTER,observed_count_exact=False,
        observed_sha256=HEX,retained_bytes=QUOTA_BYTES,retained_sha256=HEX,
        discarded_observed_bytes=MAX_COUNTER,eof_observed=False,capture_complete=False,
        unobserved_bytes=None,overflow=False,write_failed=False)


def process_record_bound(*,prepared,command,receipt,creation_address,exit_address,
                         clock_sequence_max,parent_identity=None):
    """Seven JSON slots plus two spools of one actual ProcessObserver.

    Includes both normal and emergency prefixes when they coexist. Exclusive
    xb records have one slot each; any failed prefix is within that slot.
    Bound runtime is the pinned 64-bit Win32/stdlib producer, not injected
    callbacks. Unknown runtime exception classes cannot inherit this proof.
    """
    a.need(type(prepared) is a.PreparedLaunch,'actual prepared launch required')
    a.integer(clock_sequence_max);a.address(creation_address);a.address(exit_address)
    spec=a.decode(prepared.expected.specification)
    a.need(command.source_commitment==spec['source_binding_sha256'] and
        a.sha(a.canonical(p.wire(command)))==spec['command_binding_sha256'],'command/source substitution')
    dispatch=a.decode(receipt.dispatch_file);a.ref(dispatch)
    selected=next(x for x in spec['child_slots'] if x['key']==prepared.expected.slot_key)
    parent=next((x for x in spec['child_slots'] if x['key']==selected['parent_key']),None)
    a.need((parent is None)==(parent_identity is None),'actual parent identity required')
    a.process_identity(parent_identity,parent is None)
    observation=dict(observer_id=spec['observer_id'],clock_epoch_id=spec['clock_epoch_id'],
        sequence=clock_sequence_max,observed_utc='9999-12-31T23:59:59.999999Z')
    creation=dict(schema='state_process_creation_observation.v2',kind='creation',
        authority_id=prepared.authority_id,slot_id=prepared.slot_id,operation=spec['operation'],role=spec['role'],
        parent_slot_id=None if parent is None else HEX,parent_process_identity=parent_identity,
        dispatch_file=dispatch,source_binding_sha256=spec['source_binding_sha256'],
        command_binding_sha256=spec['command_binding_sha256'],observation=observation,
        call_status='returned_success',creation_status='started',pid=U32,failure_files=[])
    creation_size=_size(creation)
    creation_ref=dict(creation_address,file_bytes=creation_size,file_sha256=HEX)
    exit_record=dict(schema='state_process_exit_observation.v2',kind='exit',slot_id=prepared.slot_id,
        process_identity=dict(pid=U32,creation_observation_file=creation_ref),direct_exit_code=-2**31,
        reap_status='observed_reaped',observation=observation,failure_files=[])
    # Longer of signed interpreter representation and Windows DWORD code.
    exit_size=max(_size(exit_record),_size(dict(exit_record,direct_exit_code=U32)))
    job=_job();streams={k:_stream() for k in ('stdout','stderr')}
    char=encoded_worst_character(json_encoding)
    # At most one error for each termination, stream-write and release operation.
    operations=('TerminateJobObject','TerminateProcess','stdout_spool_write','stderr_spool_write',
                'close_parent_writers','release')
    errors=[dict(operation=op,type=char*128,winerror=-2**63,errno=-2**63) for op in operations]
    ownership=dict(process=dict(pid=U32,role=spec['role'],creation_filetime=U64,exit_code=U32),
        job=job,inherited_handles=list(command.inherited_handles),assigned_before_resume=True)
    closure=dict(job=job,direct_exit_code=-2**31,streams_eof={k:False for k in streams},
        bytes={k:streams[k]['quota_bytes'] for k in streams},terminated=False,
        streams=streams,operation_errors=errors,capture_overflow=False)
    # _emergency_close's untruncated type name comes from these pinned stdlib
    # operations. Bind the complete builtin exception-name domain, not 128 emoji.
    names=tuple(sorted(name for name,value in vars(builtins).items()
        if isinstance(value,type) and issubclass(value,BaseException)))
    error_name=max(names,key=lambda x:len(a.canonical(x)))
    emergency=dict(job=job,resolved=False,operation_errors=errors,streams=streams,
        handles_released=False,error=error_name)
    readback={k:dict(retained_bytes=v['quota_bytes'],retained_sha256=HEX,readback_verified=True,
        discarded_bytes_independently_verified=False) for k,v in streams.items()}
    # state_execution.supervise_once invokes capture once and cleanup at most once. A
    # persistence error after the first _raw('closure') can leave BOTH raw
    # records. The fixed exit name still occupies one exclusive slot. Direct
    # unbounded calls to the low-level boundary are not this reviewed graph.
    records=dict(ownership=_size(ownership),creation=creation_size,closure=_size(closure),
        cleanup_closure=_size(closure),
        exit=exit_size,stream_readback=_size(readback),emergency_closure=_size(emergency))
    from .bounded_stream import QUOTA_BYTES
    return dict(record_bytes=sum(records.values()),records=records,json_slots=7,spool_slots=2,
        spool_bytes=2*QUOTA_BYTES,one_copy_bytes=sum(records.values())+2*QUOTA_BYTES,
        creation_file_bound=creation_ref,exit_file_bound=dict(exit_address,file_bytes=exit_size,file_sha256=HEX),
        clock_sequence_max=clock_sequence_max,runtime_exception_names=names,
        conditional_on=('pinned 64-bit Win32/stdlib producer; no substituted callbacks',
            'independently bound clock acquisition count and complete launch inputs',
            'reviewed execution graph: one capture, at most one cleanup and one emergency close'),
        excludes=('capsule/input bytes','WindowsCapturePort copies and journals','support/archive copies',
            'retained historical launches','arbitrary third-party exception classes'))


def phase_observation_bound(request):
    """Sizes every actual normal-path raw record and published fact product.

    Adopted D stops on the first foreign entry and retains at most 64 KiB,
    explicitly disclosing partial traversal/path omission. Successful complete
    membership is unchanged. This does not establish or expand custody P.
    """
    from . import state_storage_windows as sw,state_storage_records as sr,corpus_v2_custody as custody
    sw.validate_storage_request(request)
    size_json=lambda v,**kw:len(json.dumps(v,**kw).encode())
    identity=dict(volume_serial_number_hex='0xffffffffffffffff',file_id_128_hex='f'*32,
        legacy_volume_serial_number_hex='f'*8,legacy_file_index_hex='f'*16)
    shapes=(dict(presence='absent',identity=None,size=None,sha256=None),
        dict(presence='present',identity=identity,directory=True,size=None,sha256=None),
        dict(presence='present',identity=identity,directory=False,size=2**63-1,sha256=HEX))
    roots={r.root_binding_id:r for r in request.roots}
    locations={r.root_binding_id:r.relative_path for r in request.directory.locations}
    slots={(x.kind,x.subject):x for x in request.allowed_fact_products}
    a.need(len(slots)==len(request.allowed_fact_products),'duplicate fact product')
    location=p.decode(slots['location',request.binding.attempt_id].value_bytes)
    a.need(location in roots and location in locations,'bound observed location missing')
    root=roots[location];path=Path(root.canonical_path)/locations[location]
    paths=[]
    for r in request.roots:
        paths.append(str(Path(r.canonical_path)))
        if r.root_binding_id in ('T','F','Q'):paths.append(str(Path(r.canonical_path)/locations[r.root_binding_id]))
    paths.extend(str(path/x.relative_path) for x in request.outputs)
    raw_sizes=[max(size_json(dict(path=name,observation=shape),sort_keys=True,ensure_ascii=True)
                   for shape in shapes) for name in paths]
    facts={}
    def fact(kind,subject):
        slot=slots[kind,subject]
        value=p.decode(slot.value_bytes)
        data=p.encode(dict(schema=sr.PROFILE+'.fact',binding=p.wire(request.binding),capture=p.wire(request.capture),
            kind=kind,subject=subject,value=value))
        address=p.wire(slot.acquisition_request.address)
        facts[kind,subject]=dict(address=address,bytes=len(data))
        return dict(address,file_bytes=len(data),file_sha256=HEX)
    rows=[]
    for wanted in request.outputs:
        presence=p.decode(slots['presence',wanted.slot].value_bytes)
        present=presence['presence']=='present'
        content=None if not present else dict(root_binding_id=location,
            relative_path=locations[location]+'/'+wanted.relative_path,
            file_bytes=wanted.content_file.file_bytes,file_sha256=wanted.content_file.file_sha256)
        row=dict(slot=wanted.slot,root_binding_id=location,relative_path=wanted.relative_path,
            presence=presence['presence'],native_identity=wanted.native_identity if present else None,
            custody_file=p.wire(wanted.custody_file) if present else None,content_file=content,
            presence_file=fact('presence',wanted.slot),stable_file=fact('stable',wanted.slot) if present else None)
        rows.append(row)
    present_names=tuple(sorted(r['relative_path'] for r in rows if r['presence']=='present'))
    membership=size_json(dict(directories=custody.ancestor_directories(present_names),files=present_names),ensure_ascii=True)
    directory=dict(native_identity=request.directory.native_identity,custody_file=p.wire(request.directory.custody_file),
        root_binding_id=location,relative_path=locations[location],resolved_path=root.canonical_path+'/'+locations[location],
        evidence_file=fact('attempt_directory',request.binding.attempt_id))
    original=request.original_operation
    operation=dict(action='none',call_status='not_called',effect_status='none',evidence_file=None,authority_file=None)
    if original is not None:
        operation=dict(action=original.action,call_status=original.call_status,effect_status=original.effect_status,
            evidence_file=fact('operation',request.binding.attempt_id),authority_file=p.wire(original.authority_file))
    row=dict(schema=sr.OBSERVATION,profile=sr.PROFILE,binding=p.wire(request.binding),phase=request.phase,
        capture=p.wire(request.capture),roots=p.wire(request.roots),location=location,
        location_file=fact('location',request.binding.attempt_id),attempt_directory=directory,
        context=p.wire(request.context),objects=rows,operation=operation,
        failure_files=p.wire(request.context.failure_files),cleanup_failure_files=[],free_space_bytes=U64)
    # Use the real encoder/validator, including all bound path/fact metadata.
    observation=len(sr.canonical_later_storage_bytes(row))
    fact_bytes=sum(x['bytes'] for x in facts.values())
    from .state_host_windows import MEMBERSHIP_FAILURE_BYTES
    success=sum(raw_sizes)+membership+observation+2*(fact_bytes+observation)
    # The same exclusive membership slot contains success or D failure. All
    # earlier raw observations may coexist with a refused membership scan.
    membership_failure=sum(raw_sizes)+MEMBERSHIP_FAILURE_BYTES
    return dict(raw_record_count=len(paths)+2,raw_native_sizes=raw_sizes,
        membership_success_bytes=membership,observation_bytes=observation,
        raw_success_bytes=sum(raw_sizes)+membership+observation,
        fact_products=list(facts.values()),fact_product_bytes=fact_bytes,
        published_fact_and_observation_slots=2*(len(facts)+1),
        published_fact_and_observation_bytes=2*(fact_bytes+observation),
        successful_path_bytes=success,
        complete_attempt_bytes=max(success,membership_failure),foreign_membership_bytes=MEMBERSHIP_FAILURE_BYTES,
        membership_failure_path_bytes=membership_failure,
        bound_scope='one phase observation and its publications; other domains excluded below',
        missing=(),
        adopted='D: first unexpected entry, at most64KiB encoded diagnostic, incomplete traversal explicit',
        not_adopted=('extension of P to every observed phase directory',),
        excludes=('support/archive copies','upstream acquisition copies','failure exception/stream domain'))
