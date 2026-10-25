"""Injected verification/finalizer boundaries; no direct solve or native launch."""
from dataclasses import dataclass
from types import MappingProxyType

from . import publication_flow
from . import publication_physical_records
from .state4_workflow import (ComponentInput, WorkflowResult, _entry, need,
    validate_workflow_inputs, close_execution, _arguments, require_execution_component,
    Product, file_ref, immutable)
from . import state_execution_authority as authority


@dataclass(frozen=True, slots=True)
class State6Ports:
    verify_retained: object
    prepare_metadata: object
    publication_ports: object
    acquire_finalized: object
    terminal_inputs: object
    pre_move_verify: object = None


@dataclass(frozen=True, slots=True)
class SyntheticRetainedScience:
    original_bindings: tuple
    ordered_shards: tuple
    ordered_splits: tuple
    input_commitments: tuple
    diagnostics: tuple
    simulation_only: bool = True
    numerical_qualification: bool = False


@dataclass(frozen=True, slots=True)
class PreparedMetadata:
    products: tuple
    input_files: tuple


class PublicationBoundaryFailure(RuntimeError):
    """Diagnostic returned fact, never a substitute for independently retained evidence."""
    def __init__(self, stage, step, observation):
        self.stage, self.step = stage, step
        self.observation = publication_flow.verifier._freeze(observation)
        super().__init__('publication ' + stage + ' failed: ' + step)


class _PublicationPorts:
    """Bind the immutable preparation to the coordinator's sole write step."""
    def __init__(self, ports, prepared, expected, pre_move_verify=None):
        self.ports, self.prepared, self.expected = ports, prepared, expected
        self.pre_move_verify = pre_move_verify
        self.pre_move_assessment = None
        self.pre_move_error = None
        self.preparation_used = False
        self.failures = []

    def _observed_call(self, stage, step, function, *args):
        try:
            reply = function(*args)
        except BaseException as exc:
            self.failures.append((stage, step, exc))
            raise
        body = (reply.get('result') if type(reply) in (dict, MappingProxyType) else None
                ) if stage in ('operation', 'cleanup') else reply
        if type(body) in (dict, MappingProxyType) and (
                body.get('call_status') != 'returned_success' or
                body.get('failures') or body.get('failure') or
                body.get('effect_status') in ('unknown', 'unexpected_effect_observed')):
            self.failures.append((stage, step, PublicationBoundaryFailure(stage, step, body)))
        return reply

    def preserve_failures(self, run):
        for stage, step, failure in self.failures:
            run.fail(failure, cleanup=stage == 'cleanup')
            if stage == 'retention':
                run.unavailable.append('publication_retention:' + step)

    def prepare(self, step, evidence_files):
        reply = self.ports.prepare(step, evidence_files)
        if step == 'finalize_metadata':
            need(not self.preparation_used, 'metadata preparation used twice')
            self.preparation_used = True
            need(type(reply) in (dict, MappingProxyType), 'prepared publication envelope')
            rows = reply['prepared_files']
            slots = {s.key:s for s in self.expected.outputs}
            wanted = []
            for product in self.prepared.products:
                need(type(product) is Product and product.key in slots,
                     'projected metadata output required')
                projected = file_ref(product.file)
                need({key: projected[key] for key in ('root_binding_id', 'relative_path')} ==
                     authority.decode(slots[product.key].address),
                     'projected metadata destination substitution')
                authority.check_bytes(projected, product.data)
                wanted.append((authority.decode(slots[product.key].address), product.data))
            need(type(rows) in (tuple, list), 'prepared publication rows')
            actual = tuple((r['destination'], r['data']) for r in rows)
            need(actual == tuple(wanted), 'publication metadata preparation substitution')
            # The E2 preparation file is a retained source in L.  The Product
            # describes bytes projected to the separately accepted T destination;
            # its FileRef is not a claim that those bytes already exist there.
            sources = tuple(authority.canonical(publication_flow.verifier._plain(r['file']))
                            for r in rows)
            need(len(set(sources)) == len(sources) and
                 set(sources) <= set(self.prepared.input_files),
                 'undeclared or duplicate prepared metadata source')
            for source, row in zip(sources, rows, strict=True):
                authority.check_bytes(file_ref(source), row['data'])
        return reply

    def check_scope(self, checkpoint): return self.ports.check_scope(checkpoint)
    def retain_intent(self, step, data):
        return self._observed_call('retention', step + ':intent', self.ports.retain_intent, step, data)
    def read_retained(self, ref): return self.ports.read_retained(ref)
    def perform(self, step, data):
        return self._observed_call('operation', step, self.ports.perform, step, data)
    def observe(self, phase):
        if phase == 'pre_move' and self.pre_move_verify is not None:
            try:
                from .final_scientific_verify import Assessment
                need(self.preparation_used and self.pre_move_assessment is None,
                     'single pre-move corpus check after preparation required')
                assessment = self.pre_move_verify(self.prepared)
                need(type(assessment) is Assessment and assessment.level == 'corpus'
                     and assessment.status == 'matched' and assessment.diagnostics == (),
                     'independent pre-move corpus/science comparison required')
                self.pre_move_assessment = assessment
            except BaseException as exc:
                self.pre_move_error = exc
                raise
        return self.ports.observe(phase)
    def retain_result(self, step, data):
        return self._observed_call('retention', step + ':result', self.ports.retain_result, step, data)
    def release_resources(self, data):
        return self._observed_call('cleanup', 'release_resources', self.ports.release_resources, data)


def _selected_process_products(component):
    """Use the selected independent execution graph, not a process label."""
    expected = _arguments(component)['expected']
    selected = authority.decode(expected.products)['slot_id']
    histories = authority.decode(expected.graph)['slots']
    matches = [row for row in histories if row['slot_id'] == selected]
    need(len(matches) == 1, 'one selected independent process history required')
    return dict(process=matches[0]['process_file'], guards=matches[0]['guard_file'])


def _bind_publication_child(publication_expected, execution_component):
    """The publication entry must contain this compared child's exact products."""
    products = _selected_process_products(execution_component)
    accepted = publication_flow.verifier._plain(publication_expected.accepted_input_files)
    for key, role in (('process', 'state6_child_execution'),
                      ('guards', 'state6_child_guards')):
        matches = [row['file'] for row in accepted if row['role'] == role]
        need(len(matches) == 1 and products[key] == matches[0],
             'publication child ' + key + ' substitution')


def _independent_publication_bytes(rows, ref):
    matches = [row['data'] for row in rows if row['file'] == ref and
               type(row['data']) is bytes]
    need(len(matches) == 1, 'independent terminal publication bytes required')
    authority.check_bytes(ref, matches[0])
    return matches[0]


def _bind_terminal_to_worker(*, expected, worker_result, closure_inputs, terminal_input):
    """Preserve one coordinator history across the later finalizer exit."""
    completed = [value for kind, value in worker_result.comparisons
                 if kind == 'publication_coordinator']
    need(len(completed) == 1, 'one returned publication coordinator required')
    publication = publication_flow.verifier._plain(completed[0])
    need(publication['coordinator_return_status'] == 'completed' and
         publication['located_root'] == 'F' and
         publication['last_comparison']['report']['phase'] == 'post_complete' and
         publication['last_comparison']['report']['comparison_status'] == 'matched',
         'completed original post-complete assessment required')
    rows = publication_flow.verifier._plain(_arguments(terminal_input)['independent_files'])
    schema = publication_physical_records._PREFIX + 'publication_terminal_outcome.v1'
    terminals = []
    for row in rows:
        raw = row['data']
        if type(raw) is not bytes:
            continue
        try:
            candidate = publication_flow.verifier._decode(raw)
        except publication_flow.verifier.PublicationComparisonError:
            continue
        if type(candidate) is dict and candidate.get('schema') == schema:
            terminals.append(publication_physical_records.strict_publication_record_loads(
                raw, expected_schema=schema))
    need(len(terminals) == 1, 'one independently retained terminal record required')
    terminal = terminals[0]
    need(terminal['binding'] == authority.decode(expected.launch.specification)['publication_binding'] and
         terminal['operation_files'] == publication['operation_files'],
         'terminal operation history substitution')
    original = publication['last_comparison']
    report_bytes = publication_flow.verifier._canonical(original['report'])
    report = terminal['physical_comparison_file']
    need(_independent_publication_bytes(rows, report) == report_bytes,
         'original post-complete report substitution')
    details = original['report']['sealed_details_file']
    need(_independent_publication_bytes(rows, details) ==
         publication_flow.verifier._canonical(original['sealed_details']),
         'original post-complete details substitution')
    child = tuple(c for c in expected.component_inputs if c.kind == 'execution')
    need(len(child) == 1, 'one selected verification child required')
    child_process = _selected_process_products(child[0])['process']
    finalizer_process = _selected_process_products(closure_inputs[0])['process']
    need(terminal['process_outcome_files'] == [child_process, finalizer_process],
         'terminal process/independent exit substitution')


def validate_state6_inputs(*, expected, independent_inputs, mode):
    validate_workflow_inputs(expected=expected, independent_inputs=independent_inputs, mode=mode)
    _validate_state6_contract(expected)


def _validate_state6_contract(expected):
    # Called only after full workflow validation, either above or by _entry.
    # No qualification is cached or carried across an injected boundary.
    need(expected.stage == 'state6' and expected.role in ('state6_verifier', 'state6_finalizer'), 'State 6 role')
    need({'state5_closed', 'publication_plan', 'state6_authorization'} <= {r.role for r in expected.prerequisites}, 'actual A6 and predecessors required')
    need(all(s.kind != 'rho' for s in expected.outputs), 'original rho cannot be rewritten')


def run_state6_verification(*, expected, independent_inputs, ports, mode):
    run = _entry(expected, independent_inputs, mode)
    if run.failed:
        return run.result()
    try:
        _validate_state6_contract(expected)
        need(expected.role == 'state6_verifier' and type(ports) is State6Ports, 'verification child ports')
        if mode == 'component_validation':
            from .state_phase_adapters import RetainedScienceVerifier
            need(type(ports.verify_retained) is RetainedScienceVerifier,
                 'actual retained-science component port required')
        checked = run.call('verify_retained', ports.verify_retained, expected.originals)
        wanted = tuple((s, i) for s, n in (('training', 64), ('validation', 4), ('test', 8)) for i in range(n))
        if mode == 'synthetic':
            need(type(checked) is SyntheticRetainedScience and checked.simulation_only is True
                 and checked.numerical_qualification is False, 'explicit retained-science simulation')
            need(checked.original_bindings == expected.originals, 'retained original custody substitution')
            need(checked.ordered_shards == wanted and checked.ordered_splits == ('training', 'validation', 'test'), 'complete science schedule')
            need(type(checked.input_commitments) is tuple and len(checked.input_commitments) == 76
                 and all(type(x) is bytes and x for x in checked.input_commitments), 'retained scientific commitments unavailable')
            need(checked.diagnostics == (), 'scientific simulation not complete')
            run.steps.append('synthetic_science_only')
        else:
            from .state_scientific_adapters import ComponentRetainedScience
            from .final_scientific_verify import Assessment
            source = authority.decode(expected.launch.specification)['binding']['implementation']
            need(type(checked) is ComponentRetainedScience and checked.phase == 'state6_manifests'
                 and checked.status == 'matched' and checked.simulation_only is False
                 and checked.numerical_qualification is False, 'exact retained-science component required')
            need(checked.source_identity == (source['commit'], source['tree']),
                 'retained science/source substitution')
            need(checked.ordered_shards == wanted and checked.ordered_splits == ('training', 'validation', 'test')
                 and checked.sample_population == 4864, 'complete retained-science schedule')
            need(type(checked.shard_assessments) is tuple and len(checked.shard_assessments) == 76
                 and all(type(a) is Assessment and a.level == 'shard' and a.status == 'matched'
                         and a.diagnostics == () for a in checked.shard_assessments),
                 'actual retained shard comparisons required')
            need(type(checked.split_assessments) is tuple and len(checked.split_assessments) == 3
                 and all(type(a) is Assessment and a.level == 'split' and a.status == 'matched'
                         and a.diagnostics == () for a in checked.split_assessments)
                 and checked.corpus_assessment is None, 'actual child split comparisons required')
            run.steps.append('retained_science_and_child_manifests_checked')
        # No corpus manifest, audit, own exit or terminal is constructed here.
    except BaseException as exc:
        run.fail(exc)
    return run.result()


def close_state6_verification(*, expected, independent_inputs, worker_result, closure_inputs, mode):
    need(expected.stage == 'state6' and expected.role == 'state6_verifier', 'verification closure role')
    return close_execution(expected=expected, independent_inputs=independent_inputs,
                           worker_result=worker_result, closure_inputs=closure_inputs, mode=mode)


def run_state6_finalizer(*, expected, independent_inputs, ports, mode):
    run = _entry(expected, independent_inputs, mode)
    if run.failed:
        return run.result()
    try:
        _validate_state6_contract(expected)
        need(expected.role == 'state6_finalizer' and type(ports) is State6Ports, 'finalizer ports')
        if mode == 'component_validation':
            from .state_phase_adapters import PreMoveCorpusVerifier
            need(type(ports.pre_move_verify) is PreMoveCorpusVerifier,
                 'independent pre-move corpus/science check required')
            need(getattr(ports.acquire_finalized, '__self__', None) is ports.pre_move_verify and
                 getattr(ports.acquire_finalized, '__func__', None) is
                 PreMoveCorpusVerifier.readback_for_workflow,
                 'one-use pre-move readback handoff required')
        need('state6_child_closed' in {r.role for r in expected.prerequisites}, 'child closure prerequisite')
        component = tuple(c for c in expected.component_inputs if c.kind == 'publication')
        need(len(component) == 1, 'independent publication entry')
        args = _arguments(component[0])
        need(args['phase'] == 'pre_finalize', 'publication phase cutoff')
        need(type(args['expected']) is publication_flow.verifier.PublicationExpectations,
             'independent publication expectations required')
        # This coordinator path does not pass through _Run.component. Bind its
        # complete expectations here, before any metadata preparation or ports.
        need(args['expected'].binding == authority.decode(expected.launch.specification)['publication_binding'],
             'publication authorization/plan substitution')
        closure = tuple(c for c in expected.component_inputs if c.kind in ('execution', 'guard', 'final_evidence'))
        need(tuple(c.kind for c in closure) == ('execution', 'guard', 'final_evidence'), 'actual independent child closure checks')
        require_execution_component(closure[0], expected, 'state6_verifier')
        need(_arguments(closure[2])['expected'].phase == 'state6_child_closed', 'child metadata cutoff')
        for component in closure:
            run.component(component)
        need(authority.decode(run.comparisons[-3][1].facts)['guards'] == authority.decode(run.comparisons[-2][1].closure),
             'child guard/process correspondence')
        _bind_publication_child(args['expected'], closure[0])
        inventory = tuple(c for c in expected.component_inputs if c.kind == 'retention')
        need(len(inventory) == 1, 'independent pre-publication retention observation required')
        run.component(inventory[0])
        prepared = ports.prepare_metadata()
        need(type(prepared) is PreparedMetadata and type(prepared.products) is tuple
             and type(prepared.input_files) is tuple, 'immutable preparation only')
        immutable(prepared)
        need(bool(prepared.products) and len({p.key for p in prepared.products}) == len(prepared.products), 'prepared metadata inventory')
        slots = {s.key:s for s in expected.outputs}
        phase_rows = publication_flow.verifier._plain(args['expected'].phase_requirements)
        pre_move = next(row for row in phase_rows if row['phase'] == 'pre_move')
        wanted_metadata = {(row['root_binding_id'], row['relative_path'].casefold())
                           for row in pre_move['allowed_new']}
        prepared_metadata = {(authority.decode(slots[p.key].address)['root_binding_id'],
                              authority.decode(slots[p.key].address)['relative_path'].casefold())
                             for p in prepared.products if type(p) is Product and p.key in slots}
        need(bool(wanted_metadata) and prepared_metadata == wanted_metadata,
             'complete pre-move metadata inventory required before dispatch')
        if mode == 'component_validation':
            need({p.key for p in prepared.products} == set(slots),
                 'complete finalizer-owned metadata output inventory required')
        for product in prepared.products:
            need(type(product) is Product and product.key in slots, 'prepared metadata outside declared slots')
            ref = file_ref(product.file)
            need({key: ref[key] for key in ('root_binding_id', 'relative_path')} ==
                 authority.decode(slots[product.key].address),
                 'prepared metadata destination substitution')
            authority.check_bytes(ref, product.data)
        need(bool(prepared.input_files) and len(set(prepared.input_files)) == len(prepared.input_files), 'prepared metadata predecessor inventory')
        for ref in prepared.input_files: file_ref(ref)
        independent_refs = {authority.canonical(publication_flow.verifier._plain(row['file']))
                            for row in args['independent_files'] if row['data'] is not None}
        need(set(prepared.input_files) <= independent_refs, 'prepared metadata predecessors unavailable')
        # Preparing bytes is not a write. The unchanged coordinator is the sole
        # dispatcher of finalize_metadata, after its actual pre_finalize gate.
        publication_ports = _PublicationPorts(ports.publication_ports, prepared, expected,
                                              ports.pre_move_verify if mode == 'component_validation' else None)
        result = publication_flow.publish_corpus_once(expected=args['expected'], entry_files=args['independent_files'],
                                                      ports=publication_ports)
        run.comparisons.append(('publication_coordinator', result))
        publication_ports.preserve_failures(run)
        if publication_ports.pre_move_assessment is not None:
            run.comparisons.append(('pre_move_corpus_science', publication_ports.pre_move_assessment))
        if publication_ports.pre_move_error is not None:
            run.fail(publication_ports.pre_move_error)
        need(result['coordinator_return_status'] == 'completed', 'publication did not complete')
        if mode == 'component_validation':
            need(publication_ports.pre_move_assessment is not None,
                 'pre-move corpus/science check did not occur')
        readback_step = ('verified_pre_move_product_readback' if mode == 'component_validation'
                         else 'independent_finalized_product_readback')
        run.call(readback_step, ports.acquire_finalized, prepared)
        run.steps.append('coordinator_returned_not_finalizer_exit')
    except BaseException as exc:
        run.fail(exc)
        run.unavailable.append('publication_closure_unestablished')
    return run.result()


def close_state6_publication(*, expected, independent_inputs, worker_result, closure_inputs, terminal_input, mode):
    need(expected.stage == 'state6' and expected.role == 'state6_finalizer', 'external finalizer closure')
    result = close_execution(expected=expected, independent_inputs=independent_inputs,
                             worker_result=worker_result, closure_inputs=closure_inputs, mode=mode)
    run = _entry(expected, independent_inputs, mode)
    run.steps.extend(result.steps); run.products.extend(result.products); run.comparisons.extend(result.comparisons)
    run.primary = result.primary; run.follow_on.extend(result.follow_on); run.unavailable.extend(result.unavailable)
    completed = ('completed_simulation' if mode == 'synthetic'
                 else 'completed_component_validation')
    run.failed = result.outcome != completed or result.primary is not None
    try:
        need(type(terminal_input) is ComponentInput and terminal_input.kind == 'publication', 'independent terminal input')
        need(_arguments(terminal_input)['phase'] == 'terminal', 'actual terminal phase required')
        # Terminal evidence is acquired after the finalizer's independent
        # direct exit.  The worker can only predeclare the pre_finalize
        # publication premise, so compare this later input as external while
        # retaining the same binding and the real publication comparator.
        need(not run.failed and not run.unavailable, 'prior finalizer/closure failure remains')
        _bind_terminal_to_worker(expected=expected, worker_result=worker_result,
                                 closure_inputs=closure_inputs, terminal_input=terminal_input)
        run.component(terminal_input, external=True)
        run.steps.append('independent_terminal_checked')
    except BaseException as exc:
        run.fail(exc)
        run.unavailable.append('terminal_unestablished')
    return run.result()
