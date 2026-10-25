"""Private E3 phase bindings.  Construction does not authorize a State run.

The adapters retain E2 boundaries supplied by the caller.  They neither find a
host capability nor manufacture a successful comparison from a port's reply.
"""

from dataclasses import dataclass
from types import MappingProxyType

from . import state_execution_authority as authority
from . import state_provider_bindings as e1
from . import state_evidence_windows as evidence_windows
from . import state_execution_windows as execution_windows
from . import publication_windows
from . import final_verify
from . import final_evidence_records
from . import state_scientific_adapters as scientific
from . import state4_workflow as workflow
from . import state5_workflow
from . import state6_workflow


@dataclass(frozen=True, slots=True)
class QualifiedBoundary:
    """An E1 admission and its separately acquired E2 bytes, not a label."""

    name: str
    expected: e1.ProviderExpectations
    acquisitions: tuple


@dataclass(frozen=True, slots=True)
class PhaseConfiguration:
    expected: workflow.WorkflowExpectations
    independent_inputs: workflow.IndependentInputs
    boundaries: tuple
    source_file: e1.FileRef
    enrollment_file: e1.FileRef
    capture: e1.Capture


@dataclass(frozen=True, slots=True)
class State4Dependencies:
    prepare: object
    generate_shard: object
    protect: object
    complete: object
    acquire_acceptance: object


@dataclass(frozen=True, slots=True)
class State5Dependencies:
    acquire_original: object
    solve_original: object
    owns_ephemeral: object
    member_inputs: object
    retain_member: object
    release_ephemeral: object
    finish: object


@dataclass(frozen=True, slots=True)
class MemberInputs:
    expected: scientific.MemberVerificationExpected
    accepted_control: object
    candidate_record_bytes: bytes


class MemberScienceVerifier:
    """Actual held-P1 member verification, tied to the last acquired original."""

    def __init__(self, *, acquire_original, member_inputs):
        workflow.need(callable(acquire_original) and callable(member_inputs),
                      'explicit independent member inputs required')
        self._acquire_original = acquire_original
        self._member_inputs = member_inputs
        self._acquisition = None

    def acquire(self, original):
        workflow.need(self._acquisition is None, 'previous acquired original not consumed')
        reply = self._acquire_original(original)
        if (type(reply) is workflow.OperationReply and reply.call_status == 'returned_success' and
                reply.effect_status == 'completed' and
                type(reply.payload) is scientific.OriginalAcquisition and
                reply.payload.original == original):
            self._acquisition = reply.payload
        return reply

    def __call__(self, owned_solve):
        acquisition = self._acquisition
        self._acquisition = None
        workflow.need(type(acquisition) is scientific.OriginalAcquisition and
                      type(owned_solve) is scientific.OwnedNumericalSolve and
                      acquisition.original == owned_solve.original,
                      'current acquired original/owned solve required')
        supplied = self._member_inputs(acquisition.original)
        workflow.need(type(supplied) is MemberInputs and
                      type(supplied.expected) is scientific.MemberVerificationExpected and
                      supplied.expected.original == acquisition.original and
                      type(supplied.candidate_record_bytes) is bytes,
                      'independently bound member inputs required')
        checked = scientific.verify_original_before_release(
            expected=supplied.expected, accepted_control=supplied.accepted_control,
            owned_solve=owned_solve, candidate_record_bytes=supplied.candidate_record_bytes,
            acquisition=acquisition)
        return workflow.OperationReply('returned_success', 'completed', (), (), (), (), checked)


@dataclass(frozen=True, slots=True)
class ChildClosure:
    expected: workflow.WorkflowExpectations
    independent_inputs: workflow.IndependentInputs
    worker_result: workflow.WorkflowResult
    closure_inputs: tuple
    result: workflow.WorkflowResult


@dataclass(frozen=True, slots=True)
class FinalMetadataInputs:
    """Finalizer-owned prepared bytes and already retained predecessors."""

    products: tuple
    input_files: tuple


@dataclass(frozen=True, slots=True)
class FinalizedReadback:
    """Actual acquired bytes at pre-move, distinct from prepared bytes."""

    candidate_files: tuple
    independent_files: tuple


@dataclass(frozen=True, slots=True)
class State6Dependencies:
    retained_inputs: object = None
    closed_child: ChildClosure | None = None
    metadata_inputs: FinalMetadataInputs | None = None
    publication_ports: publication_windows.WindowsPublicationPorts | None = None
    acquire_finalized: object = None
    terminal_inputs: object = None
    corpus_expected: final_verify.CorpusExpectations | None = None
    corpus_reader: object = None
    scientific_evidence: tuple | None = None


@dataclass(frozen=True, slots=True)
class RetainedScienceInputs:
    member_shards: tuple
    final_splits: tuple
    scientific_evidence: tuple
    reader: object
    prepare_child: object
    retain_child: object
    prepare_science_closure: object
    retain_science_closure: object


class RetainedScienceVerifier:
    """State 6 child: inspect old members, then retained child manifests."""

    def __init__(self, *, expected, inputs):
        workflow.need(type(expected) is workflow.WorkflowExpectations and
                      (expected.stage, expected.role) == ('state6', 'state6_verifier') and
                      type(inputs) is RetainedScienceInputs and
                      type(inputs.member_shards) is tuple and len(inputs.member_shards) == 76 and
                      type(inputs.final_splits) is tuple and len(inputs.final_splits) == 3 and
                      type(inputs.scientific_evidence) is tuple and
                      inputs.reader is not None and callable(inputs.prepare_child) and
                      callable(inputs.retain_child) and
                      callable(inputs.prepare_science_closure) and
                      callable(inputs.retain_science_closure),
                      'exact retained science inputs required')
        self.expected = expected
        self.inputs = inputs
        self.used = False

    def __call__(self, originals):
        workflow.need(not self.used and originals == self.expected.originals,
                      'retained verification called once for exact originals')
        self.used = True
        inputs = self.inputs
        members = scientific.verify_retained_members(expected_shards=inputs.member_shards,
            reader=inputs.reader, scientific_evidence=inputs.scientific_evidence)
        workflow.need(members.status == 'matched' and members.sample_population == 4864,
                      'retained State 5 members rejected')
        prepared = inputs.prepare_child(members)
        _prepare_child_manifest_products(expected=self.expected, inputs=prepared)
        retained = inputs.retain_child(prepared)
        workflow.need(type(retained) is workflow.OperationReply and
                      retained.call_status == 'returned_success' and
                      retained.effect_status == 'completed' and
                      not retained.failures and not retained.unavailable and
                      retained.products == prepared and retained.independent_products == prepared,
                      'child manifest independent retention/readback rejected')
        manifests = scientific.verify_final_manifests(phase='state6_manifests',
            expected=inputs.final_splits, reader=inputs.reader,
            scientific_evidence=inputs.scientific_evidence, final_evidence_inputs=None)
        combined = scientific.combine_retained_science(members=members, manifests=manifests)
        closure_product = inputs.prepare_science_closure(combined)
        workflow.need(type(closure_product) is workflow.Product and
                      closure_product.key in {slot.key for slot in self.expected.outputs} and
                      closure_product.key not in {product.key for product in prepared},
                      'science closure product outside declared child inventory')
        ref = workflow.file_ref(closure_product.file)
        slot = next(slot for slot in self.expected.outputs if slot.key == closure_product.key)
        workflow.need(authority.decode(slot.address) ==
                      {key: ref[key] for key in ('root_binding_id', 'relative_path')},
                      'science closure destination substitution')
        authority.check_bytes(ref, closure_product.data)
        record = final_evidence_records.strict_final_evidence_loads(closure_product.data)
        source = authority.decode(self.expected.launch.specification)['binding']
        workflow.need(record['binding'] == source,
                      'science closure source or phase substitution')
        _science_closure_matches(record, combined, inputs.final_splits,
                                 self.expected.originals)
        closure_reply = inputs.retain_science_closure(closure_product)
        workflow.need(type(closure_reply) is workflow.OperationReply and
                      closure_reply.call_status == 'returned_success' and
                      closure_reply.effect_status == 'completed' and
                      not closure_reply.failures and not closure_reply.unavailable and
                      closure_reply.products == (closure_product,) and
                      closure_reply.independent_products == (closure_product,),
                      'science closure independent retention/readback rejected')
        all_products = prepared + (closure_product,)
        return workflow.OperationReply('returned_success', 'completed', all_products,
                                       all_products, (), (), combined)


@dataclass(frozen=True, slots=True)
class ExternalClosureFacts:
    """The supervisor's retained inputs, in actual comparison order."""

    execution: workflow.ComponentInput
    guard: workflow.ComponentInput
    final_evidence: workflow.ComponentInput
    retention: workflow.ComponentInput


@dataclass(frozen=True, slots=True)
class TerminalObservations:
    publication: workflow.ComponentInput
    located_root: str
    post_complete: workflow.ComponentInput


def _same_binding(left, spec):
    """Compare the shared exact phase/source fields of E1 and Batch B."""
    right = spec['binding']
    implementation = right['implementation']
    return (left.attempt_id == right['attempt_id'] and
            left.state == right['state'] and
            left.authorization_sha256 == right['authorization_file_sha256'] and
            left.protocol_sha256 == right['protocol_file_sha256'] and
            left.source_sha256 == spec['source_binding_sha256'] and
            (e1.wire(left.plan_file) == spec['publication_binding']['publication_plan_file']
             if right['state'] == 'state6' else
             e1.wire(left.plan_file) in spec['predecessors']) and
            (left.implementation_commit, left.implementation_tree, left.implementation_parent) ==
            (implementation['commit'], implementation['tree'], implementation['sole_parent']))


def _science_closure_matches(record, combined, final_splits, originals):
    """Bind the retained closure record to actual child comparisons."""
    workflow.need(record['phase'] == 'state6_manifests' and not record['failure_files'],
                  'successful child science closure phase/failure contradiction')
    shards = tuple(shard for split in final_splits for shard in split.shards)
    workflow.need(len(shards) == len(record['ordered_shards']) == 76 and
                  len(record['ordered_splits']) == len(final_splits) == 3,
                  'child science closure population contradiction')
    workflow.need(type(originals) is tuple and len(originals) == 4864,
                  'complete accepted original custody required')
    for position, ((split, index), row, assessment, expected_shard) in enumerate(zip(
            combined.ordered_shards, record['ordered_shards'],
            combined.shard_assessments, shards, strict=True)):
        files = {item.role: item.file for item in expected_shard.files}
        group = originals[position * 64:(position + 1) * 64]
        workflow.need(len(group) == 64 and
                      all((item.split, item.shard, item.custody) ==
                          (split, index, group[0].custody) for item in group),
                      'accepted original shard custody substitution')
        origin = workflow.file_ref(group[0].custody)
        workflow.need((row['split'], row['shard_index']) == (split, index) and
                      row['sample_counts'] == dict(assessment.counts) and
                      row['maxima'] == dict(assessment.maxima) and
                      row['shard_tensor_sha256'] == dict(assessment.commitments) and
                      row['verification_record_commitment'] ==
                      files['verification'].file_sha256 and
                      row['member_files'] ==
                      [scientific.science.plain(files[role]) for role in
                       (*scientific.science.FAMILIES, 'verification')] and
                      row['controls_manifest_file'] ==
                      scientific.science.plain(files['controls_manifest']) and
                      row['controls_jsonl_file'] ==
                      scientific.science.plain(files['controls']) and
                      row['original_rho_origin_file'] == origin and
                      row['evidence_file'] ==
                      scientific.science.plain(expected_shard.scientific.evidence_file) and
                      row['shard_manifest_file'] ==
                      scientific.science.plain(files['manifest']),
                      'child science closure/shard comparison contradiction')
    for split, row, assessment in zip(final_splits, record['ordered_splits'],
                                       combined.split_assessments, strict=True):
        workflow.need(row['split'] == split.split and
                      row['split_manifest_file'] ==
                      scientific.science.plain(split.manifest.file) and
                      row['sample_counts'] == dict(assessment.counts) and
                      row['maxima'] == dict(assessment.maxima) and
                      row['split_tensor_sha256'] == dict(assessment.commitments),
                      'child science closure/split comparison contradiction')
    aggregate = record['aggregate']
    counts = {key: sum(item.counts[key] for item in combined.shard_assessments)
              for key in scientific.science.COUNTS}
    maxima = {key: max(float.fromhex(item.maxima[key])
                       for item in combined.shard_assessments).hex()
              for key in scientific.science.MAXIMA}
    workflow.need(aggregate['sample_counts'] == counts and
                  aggregate['maxima'] == maxima and
                  aggregate['shard_count'] == 76 and aggregate['split_count'] == 3,
                  'child science closure aggregate contradiction')


def _corpus_identity(expected, spec):
    workflow.need(type(expected) is final_verify.CorpusExpectations and
                  type(expected.splits) is tuple and len(expected.splits) == 3 and
                  expected.metadata_expected is not None,
                  'complete accepted corpus expectations required')
    shards = tuple(shard for split in expected.splits for shard in split.shards)
    workflow.need(len(shards) == 76, 'complete accepted corpus shard schedule')
    binding = spec['binding']
    publication = spec['publication_binding']
    for shard in shards:
        expected_binding = shard.bindings
        workflow.need(expected_binding['protocol_file_sha256'] == binding['protocol_file_sha256'] and
                      expected_binding['implementation_commit'] == binding['implementation']['commit'] and
                      expected_binding['implementation_tree'] == binding['implementation']['tree'] and
                      expected_binding['state4_authorization_file_sha256'] ==
                      binding['state4_authorization_file_sha256'] and
                      expected_binding['state5_authorization_file_sha256'] ==
                      publication['state5_authorization_file_sha256'] and
                      expected_binding['state6_authorization_file_sha256'] ==
                      publication['state6_authorization_file_sha256'],
                      'corpus science/source or authority substitution')
    metadata = expected.metadata_expected
    workflow.need(metadata.implementation == binding['implementation'] and
                  metadata.authorization_bindings['state6']['file']['sha256'] ==
                  publication['state6_authorization_file_sha256'],
                  'corpus metadata/source substitution')


def _validate_configuration(configuration, *, stage, role):
    workflow.need(type(configuration) is PhaseConfiguration, 'exact phase configuration required')
    workflow.immutable(configuration)
    workflow.need((configuration.expected.stage, configuration.expected.role) == (stage, role),
                  'phase configuration role substitution')
    workflow.validate_workflow_inputs(expected=configuration.expected,
        independent_inputs=configuration.independent_inputs, mode='component_validation')
    e1.validate_file_ref(configuration.source_file)
    e1.validate_file_ref(configuration.enrollment_file)
    e1.validate_capture(configuration.capture)
    workflow.need(type(configuration.boundaries) is tuple and
                  tuple(x.name for x in configuration.boundaries if type(x) is QualifiedBoundary) ==
                  ('evidence_reader', 'evidence_writer', 'process_supervisor', 'storage_observer', 'clock_observer'),
                  'exact E2 boundary population/order required')
    spec = authority.decode(configuration.expected.launch.specification)
    sources = set()
    for row in configuration.boundaries:
        workflow.need(type(row.acquisitions) is tuple, 'independent E2 acquisitions required')
        qualification = e1.qualify_provider_acquisition(expected=row.expected, acquisitions=row.acquisitions)
        workflow.need(not qualification.diagnostics and qualification.checked ==
                      tuple(a.role for a in row.expected.admissions), 'E2 boundary not independently qualified')
        workflow.need(row.expected.provider_role ==
                      ('observer' if row.name in ('storage_observer', 'clock_observer') else row.name),
                      'E2 provider role substitution')
        workflow.need(_same_binding(row.expected.binding, spec) and
                      row.expected.binding.role in (role, 'supervisor'),
                      'E1/Batch B attempt, source or role substitution')
        workflow.need(e1.within(configuration.capture, row.expected.capture),
                      'phase capture outside accepted E2 interval')
        sources.update(row.expected.source_files)
    workflow.need(configuration.source_file in sources, 'source file not independently admitted')
    workflow.need(configuration.enrollment_file in
                  {a.file for row in configuration.boundaries for a in row.expected.admissions},
                  'enrollment file not independently admitted')
    return spec


def _validate_ports(configuration, *, evidence, process, storage, clock):
    workflow.need(type(evidence) is tuple and len(evidence) == 2 and
                  type(evidence[0]) is evidence_windows.WindowsEvidenceReader and
                  type(evidence[1]) is evidence_windows.WindowsEvidenceWriter,
                  'explicit E2 evidence reader/writer required')
    workflow.need(type(process) is execution_windows._StateProviders and
                  type(clock) is execution_windows.WindowsClockPort and
                  process.clock is clock, 'explicit owned process and clock ports required')
    workflow.need(type(storage) is workflow.ComponentInput and storage.kind == 'retention' and
                  storage in configuration.expected.component_inputs,
                  'independently declared storage/retention input required')
    workflow._arguments(storage)
    prepared = process.process.configuration.prepared
    authority.reprepare(prepared, expected=configuration.expected.launch,
                        independent_inputs=configuration.independent_inputs.launch_inputs)
    for port, name in zip(evidence, ('evidence_reader', 'evidence_writer')):
        evidence_windows.validate_evidence_configuration(port.configuration)
        accepted = next(x for x in configuration.boundaries if x.name == name)
        workflow.need(port.configuration.provider == accepted.expected and
                      port.configuration.accepted_upstream == accepted.acquisitions,
                      'E2 evidence port/qualification substitution')
    workflow.need(clock.configuration is process.process.configuration,
                  'process/clock source substitution')


def _functions(instance, names):
    for name in names:
        workflow.need(callable(getattr(instance, name)), 'explicit phase dependency required: ' + name)


def build_state4_ports(*, configuration, evidence, process, storage, clock, dependencies):
    _validate_configuration(configuration, stage='state4', role='state4_worker')
    _validate_ports(configuration, evidence=evidence, process=process, storage=storage, clock=clock)
    workflow.need(type(dependencies) is State4Dependencies, 'exact State 4 dependencies required')
    _functions(dependencies, ('prepare', 'generate_shard', 'protect', 'complete', 'acquire_acceptance'))
    # Only the workflow's fixed population and closed comparators can assess
    # the returned products.  This builder neither samples a seed nor writes.
    return workflow.State4Ports(dependencies.prepare, dependencies.generate_shard,
        dependencies.protect, dependencies.complete, dependencies.acquire_acceptance)


def build_state5_ports(*, configuration, evidence, process, storage, clock, dependencies):
    _validate_configuration(configuration, stage='state5', role='state5_worker')
    _validate_ports(configuration, evidence=evidence, process=process, storage=storage, clock=clock)
    workflow.need(type(dependencies) is State5Dependencies, 'exact State 5 dependencies required')
    _functions(dependencies, ('acquire_original', 'solve_original', 'owns_ephemeral',
        'member_inputs', 'retain_member', 'release_ephemeral', 'finish'))
    verifier = MemberScienceVerifier(acquire_original=dependencies.acquire_original,
                                     member_inputs=dependencies.member_inputs)
    return state5_workflow.State5Ports(verifier.acquire, dependencies.solve_original,
        dependencies.owns_ephemeral, verifier, dependencies.retain_member,
        dependencies.release_ephemeral, dependencies.finish)


def _closed_child(expected, closed_child, *, mode='component_validation'):
    workflow.need(mode in ('synthetic', 'component_validation'),
                  'explicit verification child route required')
    workflow.need(type(closed_child) is ChildClosure and
                  (closed_child.expected.stage, closed_child.expected.role) == ('state6', 'state6_verifier'),
                  'actual closed verification child required')
    child = state6_workflow.close_state6_verification(expected=closed_child.expected,
        independent_inputs=closed_child.independent_inputs, worker_result=closed_child.worker_result,
        closure_inputs=closed_child.closure_inputs, mode=mode)
    completed = ('completed_simulation' if mode == 'synthetic'
                 else 'completed_component_validation')
    workflow.need(child == closed_child.result and child.outcome == completed and
                  not child.production_accepted and not child.execution_authorized,
                  'independently checked child closure unavailable')
    final_spec = authority.decode(expected.launch.specification)
    child_spec = authority.decode(closed_child.expected.launch.specification)
    for name in ('attempt_id', 'protocol_file_sha256', 'implementation',
                 'state4_authorization_file_sha256'):
        workflow.need(child_spec['binding'][name] == final_spec['binding'][name],
                      'child/finalizer identity substitution')
    workflow.need(child_spec['publication_binding'] == final_spec['publication_binding'],
                  'child/finalizer publication-plan substitution')
    return child


def prepare_final_metadata(*, expected, closed_child, inputs,
                           mode='component_validation'):
    """Check finalizer-only bytes after child closure; perform no retention."""
    workflow.need(type(expected) is workflow.WorkflowExpectations and
                  (expected.stage, expected.role) == ('state6', 'state6_finalizer'),
                  'State 6 finalizer expectations required')
    _closed_child(expected, closed_child, mode=mode)
    workflow.need(type(inputs) is FinalMetadataInputs, 'exact final metadata inputs required')
    workflow.immutable(inputs)
    workflow.need(type(inputs.products) is tuple and type(inputs.input_files) is tuple and
                  bool(inputs.products) and bool(inputs.input_files), 'complete prepared metadata inputs required')
    slots = {slot.key: slot for slot in expected.outputs}
    keys, addresses = set(), set()
    for product in inputs.products:
        workflow.need(type(product) is workflow.Product and product.key in slots and product.key not in keys,
                      'undeclared or duplicate finalizer product')
        ref = workflow.file_ref(product.file)
        address = authority.decode(slots[product.key].address)
        workflow.need((ref['root_binding_id'], ref['relative_path']) ==
                      (address['root_binding_id'], address['relative_path']),
                      'final metadata destination substitution')
        authority.check_bytes(ref, product.data)
        destination = (address['root_binding_id'], address['relative_path'].casefold())
        workflow.need(destination not in addresses, 'write-once metadata destination')
        keys.add(product.key)
        addresses.add(destination)
    workflow.need(keys == set(slots), 'complete finalizer-owned metadata inventory required')
    for ref in inputs.input_files:
        workflow.file_ref(ref)
    workflow.need(len(set(inputs.input_files)) == len(inputs.input_files) and
                  not set(inputs.input_files) & {p.file for p in inputs.products},
                  'final metadata predecessor collision')
    # The workflow binds these predecessors to independent pre-finalize bytes;
    # preparation is not independent readback and is never an authorization.
    return state6_workflow.PreparedMetadata(inputs.products, inputs.input_files)


def _finalized_readback(prepared, reply):
    workflow.need(type(reply) is workflow.OperationReply and
                  reply.call_status == 'returned_success' and reply.effect_status == 'completed' and
                  not reply.failures and not reply.unavailable and
                  type(reply.payload) is FinalizedReadback,
                  'independent finalized readback unavailable')
    acquired = reply.payload
    workflow.need(type(acquired.candidate_files) is tuple and type(acquired.independent_files) is tuple,
                  'finalized readback inventories required')
    for product in prepared.products:
        ref = authority.decode(product.file)
        matches = [row for row in acquired.independent_files
                   if type(row) in (dict, MappingProxyType) and
                   row.get('file') == ref and row.get('data') == product.data]
        workflow.need(len(matches) == 1, 'prepared bytes lack independent finalized readback')
    workflow.need(reply.products == prepared.products and
                  reply.independent_products == prepared.products,
                  'finalized products/readback population disagreement')
    return acquired


class PreMoveCorpusVerifier:
    """The sole component-route pre-promotion science gate.

    Its acquisition is independent of the prepared byte producer.  The fixed
    corpus verifier performs both the array/member traversal and metadata
    comparison.  The class is exact so a favorable callback cannot replace it.
    """

    def __init__(self, *, acquire_finalized, expected, reader, scientific_evidence):
        workflow.need(callable(acquire_finalized) and type(expected) is final_verify.CorpusExpectations and
                      reader is not None and type(scientific_evidence) is tuple,
                      'explicit pre-move verifier dependencies required')
        self.acquire_finalized = acquire_finalized
        self.expected = expected
        self.reader = reader
        self.scientific_evidence = scientific_evidence
        self.used = False
        self.readback_reply = None
        self._prepared = None
        self._handed_off = False

    def __call__(self, prepared):
        workflow.need(not self.used, 'pre-move corpus verifier used twice')
        self.used = True
        reply = self.acquire_finalized(prepared)
        acquired = _finalized_readback(prepared, reply)
        checked = scientific.verify_final_manifests(phase='corpus', expected=self.expected,
            reader=self.reader, scientific_evidence=self.scientific_evidence,
            final_evidence_inputs={'candidate_files': acquired.candidate_files,
                                   'independent_files': acquired.independent_files})
        workflow.need(type(checked) is scientific.ComponentRetainedScience and checked.status == 'matched' and
                      checked.sample_population == 4864 and checked.corpus_assessment.status == 'matched',
                      'full pre-move corpus/science correspondence rejected')
        self.readback_reply = reply
        self._prepared = prepared
        return checked.corpus_assessment

    def readback_for_workflow(self, prepared):
        workflow.need(self.used and not self._handed_off and self._prepared == prepared and
                      self.readback_reply is not None,
                      'verified pre-move readback unavailable or already consumed')
        self._handed_off = True
        return self.readback_reply


def _validate_child_dependencies(dependencies):
    """Refuse later-route capabilities before constructing the child verifier."""
    workflow.need(type(dependencies) is State6Dependencies,
                  'exact State 6 dependencies required')
    workflow.need(type(dependencies.retained_inputs) is RetainedScienceInputs,
                  'retained science verifier inputs required')
    workflow.need(all(getattr(dependencies, field) is None for field in
        ('closed_child', 'metadata_inputs', 'publication_ports', 'acquire_finalized',
         'terminal_inputs', 'corpus_expected', 'corpus_reader', 'scientific_evidence')),
        'verification child cannot carry finalizer or seed capability')


def build_state6_ports(*, configuration, evidence, process, storage, clock, dependencies):
    workflow.need(type(configuration) is PhaseConfiguration, 'exact phase configuration required')
    role = configuration.expected.role
    workflow.need(role in ('state6_verifier', 'state6_finalizer'), 'State 6 role required')
    spec = _validate_configuration(configuration, stage='state6', role=role)
    _validate_ports(configuration, evidence=evidence, process=process, storage=storage, clock=clock)
    workflow.need(type(dependencies) is State6Dependencies,
                  'exact State 6 dependencies required')
    if role == 'state6_verifier':
        _validate_child_dependencies(dependencies)
        verifier = RetainedScienceVerifier(expected=configuration.expected,
                                            inputs=dependencies.retained_inputs)
        return state6_workflow.State6Ports(verifier, None, None, None, None)

    workflow.need(dependencies.retained_inputs is None and
                  type(dependencies.closed_child) is ChildClosure and
                  type(dependencies.metadata_inputs) is FinalMetadataInputs and
                  type(dependencies.publication_ports) is publication_windows.WindowsPublicationPorts and
                  callable(dependencies.acquire_finalized) and callable(dependencies.terminal_inputs) and
                  type(dependencies.corpus_expected) is final_verify.CorpusExpectations and
                  dependencies.corpus_reader is not None and type(dependencies.scientific_evidence) is tuple,
                  'complete explicit finalizer dependencies required')
    publication_windows.validate_publication_provider_configuration(
        dependencies.publication_ports.configuration,
        expected=dependencies.publication_ports.expected)
    accepted = authority.decode(configuration.expected.launch.specification)['publication_binding']
    workflow.need(dependencies.publication_ports.expected.binding == accepted,
                  'publication authorization/plan substitution')
    workflow.need(dependencies.publication_ports.configuration.source_file == configuration.source_file and
                  dependencies.publication_ports.configuration.enrollment_file == configuration.enrollment_file,
                  'publication source/enrollment substitution')
    _corpus_identity(dependencies.corpus_expected, spec)

    def prepare():
        return prepare_final_metadata(expected=configuration.expected,
            closed_child=dependencies.closed_child, inputs=dependencies.metadata_inputs)

    pre_move_verify = PreMoveCorpusVerifier(acquire_finalized=dependencies.acquire_finalized,
        expected=dependencies.corpus_expected, reader=dependencies.corpus_reader,
        scientific_evidence=dependencies.scientific_evidence)
    return state6_workflow.State6Ports(None, prepare,
        dependencies.publication_ports, pre_move_verify.readback_for_workflow,
        dependencies.terminal_inputs, pre_move_verify)


def _prepare_child_manifest_products(*, expected, inputs):
    """Only State 6 child context, shard and split products; never final metadata."""
    workflow.need(type(expected) is workflow.WorkflowExpectations and expected.role == 'state6_verifier',
                  'verification child required')
    workflow.need(type(inputs) is tuple and inputs, 'child product inventory required')
    slots = {x.key: x for x in expected.outputs}
    seen = set()
    for product in inputs:
        workflow.need(type(product) is workflow.Product and product.key in slots and product.key not in seen,
                      'child product outside declared inventory')
        ref = workflow.file_ref(product.file)
        workflow.need(ref['relative_path'].endswith(('final_context.json', 'shard_manifest.json', 'split_manifest.json')),
                      'child cannot prepare finalizer metadata')
        workflow.need(authority.decode(slots[product.key].address) ==
                      {key: ref[key] for key in ('root_binding_id', 'relative_path')},
                      'child product destination substitution')
        authority.check_bytes(ref, product.data)
        seen.add(product.key)
    workflow.need(len(inputs) == 80, 'one context, 76 shard and three split manifests required')
    return inputs


def assemble_external_closure(*, expected, acquired):
    """Package retained supervisor facts for the existing closed comparators."""
    workflow.need(type(expected) is workflow.WorkflowExpectations and type(acquired) is ExternalClosureFacts,
                  'exact external closure inputs required')
    parts = (acquired.execution, acquired.guard, acquired.final_evidence)
    workflow.need(tuple(x.kind for x in parts) == ('execution', 'guard', 'final_evidence') and
                  type(acquired.retention) is workflow.ComponentInput and acquired.retention.kind == 'retention',
                  'execution/guard/final-evidence/retention order')
    workflow.require_execution_component(parts[0], expected, expected.role)
    phase = ('terminal' if expected.role == 'state6_finalizer' else
             'state6_child_closed' if expected.stage == 'state6' else
             expected.stage + '_closed')
    workflow.need(workflow._arguments(parts[2])['expected'].phase == phase,
                  'external closure cutoff substitution')
    if expected.role == 'state6_finalizer':
        # The worker's pre-publication observation remains in its result.  A
        # later exit cannot be backpatched into that earlier inventory.
        earlier = tuple(c for c in expected.component_inputs if c.kind == 'retention')
        workflow.need(len(earlier) == 1 and acquired.retention != earlier[0],
                      'separate post-finalizer retention required')
        from . import corpus_retention
        late = workflow._arguments(acquired.retention)
        snapshots = corpus_retention._plain(late['snapshots'])
        workflow.need(type(snapshots) is list and bool(snapshots) and
                      snapshots[-1]['cutoff'] == 'pre_terminal',
                      'post-finalizer retention cutoff')
    else:
        workflow.need(acquired.retention in expected.component_inputs,
                      'retention catalog/observation substitution')
    for component in (*parts, acquired.retention):
        workflow._arguments(component)
    return parts + (acquired.retention,) if expected.role == 'state6_finalizer' else parts


def assemble_terminal_inputs(*, expected, finalized, external_closure, observations):
    """Bind finalizer exit and post-complete facts before terminal comparison."""
    workflow.need(type(expected) is workflow.WorkflowExpectations and
                  (expected.stage, expected.role) == ('state6', 'state6_finalizer'),
                  'terminal finalizer role required')
    workflow.need(type(finalized) is FinalizedReadback and type(external_closure) is workflow.WorkflowResult and
                  type(observations) is TerminalObservations,
                  'independent finalizer/terminal inputs required')
    workflow.need((external_closure.stage, external_closure.role) ==
                  ('state6', 'state6_finalizer') and
                  external_closure.outcome in ('completed_component_validation', 'completed_simulation') and
                  not external_closure.production_accepted and not external_closure.execution_authorized and
                  not external_closure.unavailable and external_closure.primary is None,
                  'actual finalizer closure unavailable')
    closure_kinds = tuple(kind for kind, _ in external_closure.comparisons)
    workflow.need(len(closure_kinds) >= 4 and
                  closure_kinds[-4:] == ('execution', 'guard', 'final_evidence', 'retention') and
                  all(workflow.satisfied(kind, result,
                            state6_finalizer_cutoff=('pre_terminal' if kind == 'retention' else None))
                      for kind, result in external_closure.comparisons[-4:]),
                  'actual finalizer closure comparisons unavailable')
    workflow.need(bool(finalized.independent_files) and observations.located_root == 'F',
                  'independent final location/readback unavailable')
    workflow.need(type(observations.post_complete) is workflow.ComponentInput and
                  observations.post_complete.kind == 'publication' and
                  workflow._arguments(observations.post_complete)['phase'] == 'post_complete',
                  'original post-complete cutoff required')
    terminal = observations.publication
    workflow.need(type(terminal) is workflow.ComponentInput and terminal.kind == 'publication' and
                  workflow._arguments(terminal)['phase'] == 'terminal', 'external terminal evidence required')
    accepted = authority.decode(expected.launch.specification)['publication_binding']
    for component in (observations.post_complete, terminal):
        workflow.need(workflow._arguments(component)['expected'].binding == accepted,
                      'terminal publication binding substitution')
    # Both assessments remain separate.  The terminal verifier recomputes from
    # its retained inputs; no favorable result is inserted into this component.
    return terminal
