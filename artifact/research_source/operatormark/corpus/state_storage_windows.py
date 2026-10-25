"""E2 acquisition of v2 storage facts, separate from independent acceptance."""
from dataclasses import dataclass
from . import state_provider_bindings as p
from . import state_storage_records as records
from . import state_storage_verify as verifier
from . import publication_physical_records as physical
from . import publication_physical_verify as physical_verify
from . import state_evidence_windows as evidence


@dataclass(frozen=True, slots=True)
class _IdentityCorrespondence:
    opaque: str
    native: evidence._NativeIdentity
    custody_file: p.FileRef
    mapping_file: p.FileRef


def validate_identity_correspondence(mapping, *, accepted_inputs):
    evidence._exact(mapping, _IdentityCorrespondence)
    evidence._exact(accepted_inputs, tuple)
    p.text(mapping.opaque)
    evidence._identity(mapping.native)
    p.validate_file_ref(mapping.custody_file)
    p.validate_file_ref(mapping.mapping_file)
    evidence._need(mapping in accepted_inputs, 'identity correspondence not independently admitted')
    evidence._need(sum(x.opaque == mapping.opaque for x in accepted_inputs) == 1 and
                   sum(x.native == mapping.native for x in accepted_inputs) == 1,
                   'opaque/full identity correspondence alias')
    return mapping.native


@dataclass(frozen=True, slots=True)
class _RootCorrespondence:
    e1_root: p.RootBinding
    publication_address: bytes
    repository_relative_anchor: str


@dataclass(frozen=True, slots=True)
class _OperationProjection:
    original_bytes: bytes
    original_file: p.FileRef
    action: str
    call_status: str
    effect_status: str
    authority_file: p.FileRef | None
    diagnostics: tuple


def validate_operation_projection(projection, *, original_result, accepted_inputs):
    evidence._exact(projection, _OperationProjection)
    evidence._exact(original_result, bytes)
    evidence._exact(accepted_inputs, tuple)
    evidence._need(projection.original_bytes == original_result and
                   projection.original_file in accepted_inputs, 'original operation not independently admitted')
    p.validate_file_ref(projection.original_file)
    evidence._need((len(original_result), p.sha(original_result)) ==
                   (projection.original_file.file_bytes, projection.original_file.file_sha256), 'original operation bytes')
    row = physical.strict_publication_record_loads(original_result,
        expected_schema='operatormark.darcy_branch_c_v1.publication_operation_result.v1')
    action = {'finalize_metadata': 'metadata', 'publish_audit': 'metadata', 'publish_complete': 'metadata',
              'finalize_protection': 'metadata', 'promote': 'promote', 'quarantine': 'quarantine',
              'release_resources': 'cleanup'}[row['step']]
    call = 'unavailable' if row['call_status'] == 'outcome_unavailable' else row['call_status']
    effect = {'none_observed': 'none', 'unknown': 'unknown', 'unexpected_effect_observed': 'partial',
              'expected_effect_observed': 'complete'}[row['effect_status']]
    diagnostics = []
    if row['effect_status'] == 'unexpected_effect_observed':
        diagnostics.append(('operation', 'unexpected_effect_observed'))
    if row['call_status'] != 'returned_success' or row['effect_status'] == 'unknown':
        diagnostics.append(('operation', 'outcome_unavailable'))
    if row['failures']:
        diagnostics.append(('operation', 'retained_native_failures'))
    if effect == 'complete' and (row['after_observation_file'] is None or
                                row['after_observation_file'] not in [p.wire(x) for x in accepted_inputs]):
        effect = 'unknown'
        diagnostics.append(('operation', 'independent_postconditions_unavailable'))
    evidence._need((projection.action, projection.call_status, projection.effect_status, projection.diagnostics) ==
                   (action, call, effect, tuple(diagnostics)), 'operation projection erased or changed facts')
    if projection.authority_file is not None:
        p.validate_file_ref(projection.authority_file)
        evidence._need(projection.authority_file in accepted_inputs, 'operation authority not independently admitted')
    return projection.diagnostics


@dataclass(frozen=True, slots=True)
class _FactProduct:
    kind: str
    subject: str
    value_bytes: bytes
    retention_request: evidence._RetentionRequest
    acquisition_request: evidence._ReadRequest


@dataclass(frozen=True, slots=True)
class _PhaseInventory:
    observation_bytes: bytes
    fact_products: tuple
    complete: bool
    observed_slots: tuple
    failures: tuple


@dataclass(frozen=True, slots=True)
class _StorageRequest:
    binding: p.Binding
    phase: str
    capture: p.Capture
    roots: tuple
    directory: p.AttemptDirectoryBinding
    outputs: tuple
    context: records.TerminationContext
    identities: tuple
    root_correspondence: tuple
    root_bindings_bytes: bytes
    accepted_root_bindings_file: p.FileRef
    original_operation: _OperationProjection | None
    accepted_operation_files: tuple
    upstream_reads: tuple
    observation_write: evidence._RetentionRequest
    observation_read: evidence._ReadRequest
    allowed_fact_products: tuple
    retention_cutoff: str
    accepted_cutoff_relationships: tuple


def validate_storage_request(request):
    evidence._exact(request, _StorageRequest)
    records.phase_binding(request.binding, request.phase)
    p.validate_capture(request.capture)
    p.validate_roots(request.roots)
    p.validate_outputs(request.outputs)
    p.validate_attempt_directory(request.directory, request.roots, request.outputs)
    records.validate_termination_context(request.context)
    evidence._need(request.context.terminating_state == request.binding.state, 'terminal supervisor state')
    evidence._need((request.phase, request.retention_cutoff) in request.accepted_cutoff_relationships,
                   'phase/retention cutoff relationship not independently admitted')
    evidence._need(request.context.origin_capture.clock_epoch_id == request.capture.clock_epoch_id and
                   request.context.origin_capture.last_sequence <= request.capture.last_sequence and
                   request.context.origin_capture.finished_utc <= request.capture.finished_utc,
                   'future termination origin')
    evidence._exact(request.identities, tuple)
    for item in request.identities:
        validate_identity_correspondence(item, accepted_inputs=request.identities)
    required = {r.native_identity for r in request.roots} | {request.directory.native_identity}
    required.update(o.native_identity for o in request.outputs)
    evidence._need(required <= {r.opaque for r in request.identities}, 'missing independently admitted object identity')
    p.validate_file_ref(request.accepted_root_bindings_file)
    evidence._need((len(request.root_bindings_bytes), p.sha(request.root_bindings_bytes)) ==
                   (request.accepted_root_bindings_file.file_bytes, request.accepted_root_bindings_file.file_sha256),
                   'independent publication roots content')
    roots = physical.strict_publication_record_loads(request.root_bindings_bytes,
        expected_schema='operatormark.darcy_branch_c_v1.publication_root_bindings.v1')
    evidence._exact(request.root_correspondence, tuple)
    evidence._need(tuple(r.e1_root for r in request.root_correspondence) == request.roots, 'root correspondence order')
    locations = []
    for item in request.root_correspondence:
        evidence._exact(item, _RootCorrespondence)
        address = physical_verify._decode(item.publication_address)
        location = physical_verify._resolve_publication_address(address, roots, repository_only=True)
        evidence._need(location == ('repository', item.repository_relative_anchor), 'root map or alternate anchor substitution')
        locations.append(location)
    evidence._need(len(set(locations)) == len(locations), 'duplicate physical root correspondence')
    evidence._exact(request.upstream_reads, tuple)
    evidence._exact(request.allowed_fact_products, tuple)
    for product in request.allowed_fact_products:
        evidence._exact(product, _FactProduct)
        p.text(product.kind)
        p.text(product.subject)
        p.decode(product.value_bytes)
        evidence._need(product.retention_request.final == product.acquisition_request,
                       'fact writer/independent reader binding')
    evidence._need(request.observation_write.final == request.observation_read and
                   request.observation_read.capture == request.capture, 'observation destination/read/capture binding')
    if request.original_operation is not None:
        validate_operation_projection(request.original_operation,
            original_result=request.original_operation.original_bytes, accepted_inputs=request.accepted_operation_files)


@dataclass(frozen=True, slots=True)
class _StorageObserver:
    capture_phase: object


def capture_phase_inventory(*, request, observer, reader):
    validate_storage_request(request)
    evidence._exact(observer, _StorageObserver)
    evidence._function(observer.capture_phase)
    evidence._exact(reader, evidence.WindowsEvidenceReader)
    acquired = tuple(reader.acquire(r) for r in request.upstream_reads)
    observed = observer.capture_phase(request, acquired)
    evidence._exact(observed, _PhaseInventory)
    evidence._exact(observed.complete, bool)
    evidence._exact(observed.observed_slots, tuple)
    evidence._exact(observed.fact_products, tuple)
    evidence._failures(observed.failures)
    record = records.strict_later_storage_loads(observed.observation_bytes)
    evidence._need(record['binding'] == p.wire(request.binding) and record['phase'] == request.phase and
                   record['capture'] == p.wire(request.capture) and record['roots'] == p.wire(request.roots),
                   'observer scope/binding substitution')
    evidence._need(record['context'] == p.wire(request.context), 'termination context substitution or resurrection')
    evidence._need(all(x in request.allowed_fact_products for x in observed.fact_products), 'undeclared fact write')
    # Missing/extra/unknown populations remain explicit evidence; never turn an
    # unobserved object into an absent object to fill the expected inventory.
    evidence._need(observed.observed_slots == tuple(row['slot'] for row in record['objects']),
                   'enumeration/observation disagreement')
    return observed, acquired


@dataclass(frozen=True, slots=True)
class _StorageAcquisition:
    observation: evidence._AcquiredEvidence | None
    facts: tuple
    upstream: tuple
    retention: tuple
    original_inventory: _PhaseInventory
    diagnostics: tuple


def _storage_diagnostics(result, request):
    inventory = result.original_inventory
    record = records.strict_later_storage_loads(inventory.observation_bytes)
    rows = []
    if not inventory.complete:
        rows.append(('inventory', 'enumeration_incomplete'))
    if set(inventory.observed_slots) != {o.slot for o in request.outputs}:
        rows.append(('inventory', 'output_population_mismatch'))
    for item in record['objects']:
        if item['presence'] == 'unknown':
            rows.append((item['slot'], 'presence_unknown'))
    directory = record['attempt_directory']
    if directory is None:
        rows.append(('directory', 'unavailable'))
    else:
        for name, accepted in (('native_identity', request.directory.native_identity),
                               ('custody_file', p.wire(request.directory.custody_file))):
            if directory[name] != accepted:
                rows.append(('directory/' + name, 'unavailable' if directory[name] is None else 'mismatch'))
        places = [x for x in request.directory.locations if x.root_binding_id == directory['root_binding_id']]
        if len(places) != 1:
            rows.append(('directory/location', 'unavailable'))
        else:
            place = places[0]
            anchor = next(x for x in request.roots if x.root_binding_id == place.root_binding_id)
            if (directory['root_binding_id'], directory['relative_path'], directory['resolved_path']) != (
                    record['location'], place.relative_path, anchor.canonical_path + '/' + place.relative_path):
                rows.append(('directory/location', 'mismatch'))
    if request.context.outcome == 'failed':
        rows.append(('context', 'retained_failure_context'))
    if request.original_operation is not None:
        rows.extend(request.original_operation.diagnostics)
    for acquired in result.upstream + result.facts + (() if result.observation is None else (result.observation,)):
        rows.extend((acquired.request.key + '/' + path, reason) for path, reason in acquired.diagnostics)
    if result.observation is None:
        rows.append(('observation', 'unavailable'))
    for retained in result.retention:
        rows.extend((retained.request.key + '/' + f.stage, 'retention_failure') for f in retained.failures + retained.cleanup)
    return tuple(rows)


def validate_storage_acquisition(result, *, request):
    validate_storage_request(request)
    evidence._exact(result, _StorageAcquisition)
    for rows in (result.facts, result.upstream, result.retention, result.diagnostics):
        evidence._exact(rows, tuple)
    evidence._need(result.diagnostics == _storage_diagnostics(result, request), 'storage acquisition diagnostics substitution')
    if result.observation is not None and result.observation.data is not None:
        evidence._need(result.observation.data == result.original_inventory.observation_bytes,
                       'retained observation differs from actual observed facts')
    return result.diagnostics


def capture_later_storage_boundary(*, request, observer, reader, writer):
    inventory, upstream = capture_phase_inventory(request=request, observer=observer, reader=reader)
    evidence._exact(writer, evidence.WindowsEvidenceWriter)
    retained = []
    facts = []
    for product in inventory.fact_products:
        data = p.encode(dict(schema=records.PROFILE + '.fact', binding=p.wire(request.binding),
                             capture=p.wire(request.capture), kind=product.kind, subject=product.subject,
                             value=p.decode(product.value_bytes)))
        retained.append(writer.retain_exclusive(product.retention_request, data))
        facts.append(reader.acquire(product.acquisition_request))
    retained.append(writer.retain_exclusive(request.observation_write, inventory.observation_bytes))
    observation = reader.acquire(request.observation_read)
    result = _StorageAcquisition(observation, tuple(facts), upstream, tuple(retained), inventory, ())
    result = _StorageAcquisition(observation, tuple(facts), upstream, tuple(retained), inventory,
                                 _storage_diagnostics(result, request))
    validate_storage_acquisition(result, request=request)
    return result


def _compare_independently(*, expected, current, previous, acquisitions, provider_acquisitions):
    """Receiving comparison requires the caller's separately admitted expectations."""
    records.validate_later_storage_expectations(expected)
    result = verifier.verify_later_storage_boundary(expected=expected, current=current, previous=previous,
        acquisitions=acquisitions, provider_acquisitions=provider_acquisitions)
    verifier.validate_later_storage_report(result, expected=expected, current=current, previous=previous,
        acquisitions=acquisitions, provider_acquisitions=provider_acquisitions)
    return result
