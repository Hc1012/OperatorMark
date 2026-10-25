"""Private injected orchestration. Never grants authority or operates on a host."""
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from types import MappingProxyType

from . import state_execution_authority as authority
from . import state_execution as execution
from . import state_guard_capture as guards
from . import state4_acceptance_verify as acceptance
from . import final_evidence_verify as evidence
from . import corpus_retention as retention
from . import publication_physical_verify as publication


class WorkflowError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise WorkflowError(message)


def immutable(value):
    # Shared frozen expectations form a graph, not an expanded tree. Check each
    # object once within this invocation; do not cache across calls or skip any
    # later semantic/member/identity comparison. Cycles are invalid inputs.
    memo, active = {}, set()
    def visit(item):
        if type(item) in (str, bytes, int, bool, type(None)):
            return item
        key = id(item)
        if key in active:
            raise WorkflowError('cycle in immutable private input')
        if key in memo:
            return memo[key][1]
        active.add(key)
        try:
            if type(item) is tuple:
                result = tuple(visit(x) for x in item)
            elif type(item) is MappingProxyType:
                result = MappingProxyType({visit(k): visit(v) for k, v in item.items()})
            elif is_dataclass(item) and item.__dataclass_params__.frozen:
                for f in fields(item):
                    visit(getattr(item, f.name))
                result = item
            else:
                raise WorkflowError('mutable or unrecognized private input')
            memo[key] = (item, result)
            return result
        finally:
            active.remove(key)
    return visit(value)


def utc(value):
    need(type(value) is str and value.endswith('Z'), 'qualified UTC required')
    try:
        return datetime.fromisoformat(value[:-1] + '+00:00')
    except ValueError as exc:
        raise WorkflowError('invalid UTC') from exc


def file_ref(raw):
    need(type(raw) is bytes, 'immutable FileRef')
    ref = authority.decode(raw)
    authority.ref(ref)
    need(authority.canonical(ref) == raw, 'noncanonical FileRef')
    return ref


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    role: str
    file: bytes
    validator_file: bytes
    scope: str
    first_utc: str
    last_utc: str


@dataclass(frozen=True, slots=True)
class AcquiredInput:
    role: str
    file: bytes | None
    data: bytes | None
    validator_file: bytes | None
    scope: str
    first_utc: str | None
    last_utc: str | None
    status: str


@dataclass(frozen=True, slots=True)
class IndependentInputs:
    launch_inputs: tuple
    acquired: tuple


@dataclass(frozen=True, slots=True)
class OutputSlot:
    key: str
    address: bytes
    owner: str
    kind: str
    stage: str


@dataclass(frozen=True, slots=True)
class OriginalBinding:
    split: str
    shard: int
    offset: int
    token: bytes
    file: bytes
    custody: bytes


@dataclass(frozen=True, slots=True)
class ComponentInput:
    kind: str
    arguments: tuple


@dataclass(frozen=True, slots=True)
class WorkflowExpectations:
    stage: str
    role: str
    launch: authority.LaunchExpectations
    prerequisites: tuple
    outputs: tuple
    originals: tuple
    component_inputs: tuple


@dataclass(frozen=True, slots=True)
class Product:
    key: str
    file: bytes
    data: bytes


@dataclass(frozen=True, slots=True)
class OperationReply:
    call_status: str
    effect_status: str
    products: tuple
    independent_products: tuple
    failures: tuple
    unavailable: tuple
    payload: object = None


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    stage: str
    role: str
    outcome: str
    steps: tuple
    products: tuple
    comparisons: tuple
    primary: BaseException | None
    follow_on: tuple
    unavailable: tuple
    production_accepted: bool = False
    execution_authorized: bool = False
    route: str = 'synthetic'


SPLITS = (('training', 64), ('validation', 4), ('test', 8))
ROUTES = (('state4', 'state4_worker', 'state4-controls'),
          ('state5', 'state5_worker', 'state5-members'),
          ('state6', 'state6_verifier', 'state6-verification'),
          ('state6', 'state6_finalizer', 'state6-finalization'))


def schedule():
    return tuple((split, shard, offset) for split, count in SPLITS
                 for shard in range(count) for offset in range(64))


def _arguments(component):
    need(type(component) is ComponentInput and type(component.arguments) is tuple, 'component input')
    need(all(type(x) is tuple and len(x) == 2 and type(x[0]) is str for x in component.arguments), 'component arguments')
    result = dict(component.arguments)
    need(len(result) == len(component.arguments), 'duplicate component argument')
    immutable(component.arguments)
    return result


def compare_component(component):
    """Closed dispatch to actual independent components, not injected comparisons."""
    functions = {
        'state4_protected': acceptance.verify_state4_protected_file,
        'state4_acceptance': acceptance.verify_state4_acceptance_bundle,
        'execution': execution.compare_execution_capture,
        'guard': guards.compare_guard_capture,
        'final_evidence': evidence.verify_final_evidence,
        'publication': publication.verify_publication_evidence,
        'retention': retention.derive_retention_components,
    }
    args = _arguments(component)
    need(component.kind in functions, 'unknown component boundary')
    return functions[component.kind](**args)


def require_execution_component(component, expected, role):
    args = _arguments(component)
    need(component.kind == 'execution' and type(args.get('expected')) is execution.ExecutionVerificationExpectations,
         'independent finite execution graph required')
    graph = args['expected']
    selected = authority.decode(graph.products)['slot_id']
    matches = [launch for launch, acquired in zip(graph.launches, graph.launch_inputs, strict=True)
               if authority.prepare_launch(expected=launch, independent_inputs=acquired).slot_id == selected]
    need(len(matches) == 1, 'selected independent process unavailable')
    actual = authority.decode(matches[0].specification)
    current = authority.decode(expected.launch.specification)
    need(actual['binding'] == current['binding'] and actual['role'] == role, 'closure process/stage substitution')
    if role == expected.role:
        need(matches[0] == expected.launch, 'closure launch/source substitution')


def satisfied(kind, result, *, state6_finalizer_cutoff=None):
    # These checks classify actual recomputed objects, never authenticate them.
    if kind in ('state4_protected', 'state4_acceptance'):
        return result.status == 'corroborated'
    if kind == 'final_evidence':
        return result.report['status'] == 'matched'
    if kind == 'publication':
        return result['report']['comparison_status'] == 'matched'
    if kind == 'guard':
        return result.status == 'corroborated' and _guard_satisfied(authority.decode(result.closure))
    if kind == 'execution':
        if result.status != 'corroborated':
            return False
        facts = authority.decode(result.facts)
        creation, exited = facts['creation'], facts['exit']
        return (creation['creation_status'] == 'started' and creation['call_status'] == 'returned_success'
                and not creation['failure_files'] and exited['direct_exit_code'] == 0
                and exited['reap_status'] == 'observed_reaped' and not exited['failure_files']
                and all(facts[s]['state'] == 'complete' for s in ('stdout', 'stderr'))
                and _guard_satisfied(facts['guards']))
    if kind == 'retention':
        if state6_finalizer_cutoff is not None:
            # This narrow State 6 route still requires an independently
            # qualified, complete cutoff inventory.  Unknown future writer
            # sizes stay unresolved in the returned accounting record, and
            # never become measured zero or a production disk estimate.
            return (state6_finalizer_cutoff in ('pre_manifest', 'pre_terminal') and
                    result.latest_cutoff == state6_finalizer_cutoff and
                    not result.cutoff_unresolved_requirements and
                    not result.required_cutoff_products_missing and
                    result.observed_retained_bytes is not None and
                    result.observed_namespace_bytes is not None and
                    bool(result.snapshot_files))
        return not result.unresolved_requirements
    return False


def _guard_satisfied(closure):
    coverage = closure['coverage']
    if not all(coverage[k] is True for k in ('started_before_project_import', 'continued_through_exit', 'all_required_hooks_accounted')):
        return False
    if coverage['unavailable_hook_evidence_files']:
        return False
    if not all(r['measurement'] == 'measured' and type(r['value']) is int and r['value'] == 0
               for r in closure['forbidden_counts'].values()):
        return False
    if not all(value is True for row in closure['rng_state_equalities'].values() for value in row.values()):
        return False
    for family, rows in closure['authorized_counts'].items():
        for split, count in rows.items():
            wanted = 0
            if closure['binding']['state'] == 'state4':
                wanted = {'training':4096, 'validation':256, 'test':512}[split] if family == 'top_level_rng_calls_by_split' else 1
            if count['measurement'] != 'measured' or type(count['value']) is not int or count['value'] != wanted:
                return False
    return True


def validate_workflow_inputs(*, expected, independent_inputs, mode):
    need(type(mode) is str and mode in ('synthetic', 'component_validation'),
         'production workflow disabled')
    need(type(expected) is WorkflowExpectations and type(independent_inputs) is IndependentInputs, 'exact workflow contracts')
    immutable(expected); immutable(independent_inputs)
    need((expected.stage, expected.role) in {(s, r) for s, r, _ in ROUTES}, 'stage/role mismatch')
    prepared = authority.prepare_launch(expected=expected.launch, independent_inputs=independent_inputs.launch_inputs)
    spec = authority.decode(expected.launch.specification)
    operation = next(o for s, r, o in ROUTES if (s, r) == (expected.stage, expected.role))
    need((spec['binding']['state'], spec['role'], spec['operation']) == (expected.stage, expected.role, operation), 'launch/workflow binding')
    need(type(expected.prerequisites) is tuple and bool(expected.prerequisites), 'independent prerequisites required')
    required = {r.role: r for r in expected.prerequisites if type(r) is EvidenceRequirement}
    need(len(required) == len(expected.prerequisites), 'duplicate/invalid prerequisite')
    acquired = {}
    for row in independent_inputs.acquired:
        need(type(row) is AcquiredInput and row.role not in acquired, 'duplicate/invalid acquired role')
        acquired[row.role] = row
    need(set(acquired) == set(required), 'missing/extra acquired prerequisite')
    need({'authority', 'source', 'custody', 'storage', 'environment'} <= set(required), 'required prerequisite roles omitted')
    for role, r in required.items():
        file_ref(r.file); file_ref(r.validator_file)
        need(type(r.scope) is str and bool(r.scope) and utc(r.first_utc) <= utc(r.last_utc), 'prerequisite scope/interval')
        a = acquired[role]
        need(a.status in ('accepted', 'rejected', 'unavailable'), 'acquisition status')
        need(a.status == 'accepted' and a.data is not None, 'unavailable/rejected prerequisite: ' + role)
        need((a.file, a.validator_file, a.scope) == (r.file, r.validator_file, r.scope), 'acquisition binding: ' + role)
        need(utc(a.first_utc) <= utc(r.first_utc) <= utc(r.last_utc) <= utc(a.last_utc), 'acquisition interval: ' + role)
        authority.check_bytes(file_ref(a.file), a.data)
    roots = {r['root_binding_id'] for r in spec['roots']}
    addresses, keys = set(), set()
    for slot in expected.outputs:
        need(type(slot) is OutputSlot and slot.stage == expected.stage, 'output stage')
        address = authority.decode(slot.address); authority.address(address)
        location = (address['root_binding_id'], address['relative_path'].casefold())
        need(address['root_binding_id'] in roots and location not in addresses and slot.key not in keys, 'aliased/unbound output')
        need(slot.owner in ('accepted_attempt', 'separate_B', 'external_preexisting', 'unresolved'), 'output owner')
        need(type(slot.key) is str and bool(slot.key) and type(slot.kind) is str and bool(slot.kind), 'output key/kind')
        addresses.add(location); keys.add(slot.key)
    need(bool(keys), 'empty output inventory')
    # Launch outputs may describe component-bound collections, but every explicitly
    # declared workflow slot must have a corresponding independently accepted slot.
    accepted = {(r['address']['root_binding_id'], r['address']['relative_path'].casefold()): r for r in spec['output_slots']}
    for slot in expected.outputs:
        a = authority.decode(slot.address); match = accepted.get((a['root_binding_id'], a['relative_path'].casefold()))
        need(match is not None and match['owner'] == slot.owner, 'output missing from accepted launch inventory')
    for c in expected.component_inputs:
        _arguments(c)
    if expected.stage != 'state4':
        identities = []
        for original in expected.originals:
            need(type(original) is OriginalBinding and type(original.shard) is int and type(original.offset) is int, 'original binding')
            ref = file_ref(original.file)
            directory = 'test.sealed' if original.split == 'test' else original.split
            need(ref['relative_path'] == f'corpus_staging/{directory}/shard_{original.shard:04d}/rho.float64.npy', 'original rho address')
            need(ref['root_binding_id'] in roots, 'original root not accepted')
            need(type(original.token) is bytes and bool(original.token) and type(original.custody) is bytes and bool(original.custody), 'original custody unavailable')
            identities.append((original.split, original.shard, original.offset))
        need(tuple(identities) == schedule(), 'complete original schedule required')
        need(len({r.token for r in expected.originals}) == 4864, 'aliased original token')
        for start in range(0, 4864, 64):
            group = expected.originals[start:start + 64]
            need(len({(r.file, r.custody) for r in group}) == 1, 'shard original custody disagreement')
    return prepared


def _execution_retention_slots(*, expected, independent_inputs, source_file, source_closure_file,
                              existing_addresses, journal_inventory, mode):
    """Extend the existing catalog mechanism; unresolved domains remain unresolved.

    The independently accepted catalog adapter registers these slots alongside
    existing products. Component labels never assign an accounting owner.
    """
    prepared = validate_workflow_inputs(expected=expected, independent_inputs=independent_inputs, mode=mode)
    need(type(journal_inventory) is authority.JournalRetentionExpectations, 'explicit known journal inventory required')
    return execution.additional_retention_slots(prepared=prepared, source_file=file_ref(source_file),
        source_closure_file=file_ref(source_closure_file), existing_addresses=existing_addresses,
        journal_inventory=journal_inventory)


class _Run:
    def __init__(self, expected, mode='synthetic'):
        self.expected, self.mode = expected, mode
        self.steps, self.products, self.comparisons, self.follow_on, self.unavailable = [], [], [], [], []
        self.primary = None
        self.failed = False
        self.called = set()
        self._premises = None

    def fail(self, exc, *, cleanup=False):
        need(isinstance(exc, BaseException), 'exception identity required')
        if self.primary is None and not cleanup:
            self.primary = exc
        else:
            self.follow_on.append(exc)
        self.failed = True

    def call(self, key, function, *args):
        need(key not in self.called, 'one call only: ' + key)
        self.called.add(key)
        try:
            reply = function(*args)
        except BaseException:
            self.unavailable.append(key + ':call_outcome_unavailable')
            raise
        need(type(reply) is OperationReply, 'exact operation reply')
        need(type(reply.products) is tuple and type(reply.independent_products) is tuple and type(reply.failures) is tuple
             and type(reply.unavailable) is tuple, 'immutable operation facts')
        self.products.extend(reply.products)
        for failure in reply.failures:
            self.fail(failure)
        need(all(type(x) is str for x in reply.unavailable), 'unavailable diagnostics')
        self.unavailable.extend(reply.unavailable)
        need(reply.call_status in ('not_called', 'returned_success', 'returned_failure', 'outcome_unavailable'), 'call status')
        need(reply.effect_status in ('none', 'completed', 'partial', 'unknown'), 'effect status')
        if reply.call_status == 'outcome_unavailable' or reply.effect_status == 'unknown':
            self.unavailable.append(key + ':effect_unknown')
        need(not (reply.products and reply.effect_status == 'none'), 'published product with no effect')
        need(not (reply.call_status == 'not_called' and reply.effect_status != 'none'), 'uncalled operation with effect')
        self._products(reply)
        need(reply.call_status == 'returned_success' and reply.effect_status == 'completed'
             and not reply.failures and not reply.unavailable, 'operation did not complete: ' + key)
        self.steps.append(key)
        return reply.payload

    def _products(self, reply):
        slots = {s.key: s for s in self.expected.outputs}
        indexes = []
        for products in (reply.products, reply.independent_products):
            index = {}
            for product in products:
                need(type(product) is Product and product.key not in index and product.key in slots, 'duplicate/unexpected product')
                ref = file_ref(product.file); authority.check_bytes(ref, product.data)
                need({k: ref[k] for k in ('root_binding_id', 'relative_path')} == authority.decode(slots[product.key].address), 'product address substitution')
                index[product.key] = product
            indexes.append(index)
        need(indexes[0] == indexes[1], 'independent product readback disagreement')

    def component(self, value, *, external=False):
        # Worker premises cannot be supplied by an operation reply.  The
        # finalizer's external components are bound by close_execution after
        # the worker returns and by the independent comparators below.
        args = _arguments(value)
        premise_key = 'catalog' if value.kind == 'retention' else 'expected'
        wanted = args.get(premise_key)
        if external:
            # Only the finalizer's post-return closure is acquired after its
            # worker result.  Its future inputs cannot be worker premises.
            need(self.expected.role == 'state6_finalizer', 'external component role')
        else:
            if self._premises is None:
                self._premises = tuple((c.kind, _arguments(c).get('catalog' if c.kind == 'retention' else 'expected'))
                                       for c in self.expected.component_inputs)
            need(any(kind == value.kind and supplied == wanted for kind, supplied in self._premises), 'component expectation substitution')
        spec = authority.decode(self.expected.launch.specification)
        if value.kind == 'state4_acceptance':
            need(wanted.binding == spec['binding'], 'aggregate stage/authorization substitution')
        elif value.kind == 'state4_protected':
            context = wanted.expected_plan.expected_request.context
            binding = spec['binding']
            need((context.protocol_file_sha256, context.state4_authorization_file_sha256,
                  context.implementation_commit, context.implementation_tree) ==
                 (binding['protocol_file_sha256'], binding['state4_authorization_file_sha256'],
                  binding['implementation']['commit'], binding['implementation']['tree']), 'protection stage/source substitution')
        elif value.kind == 'guard':
            need(authority.decode(wanted.specification)['binding'] == spec['binding'], 'guard stage/authorization substitution')
        elif value.kind == 'final_evidence':
            need(wanted.implementation == spec['binding']['implementation'], 'metadata implementation substitution')
            for state, key in ((self.expected.stage, 'authorization_file_sha256'), ('state4', 'state4_authorization_file_sha256')):
                need(wanted.authorization_bindings[state]['file']['sha256'] == spec['binding'][key], 'metadata authority substitution')
        elif value.kind == 'publication':
            need(wanted.binding == spec['publication_binding'], 'publication authorization/plan substitution')
        result = compare_component(value)
        self.comparisons.append((value.kind, result))
        # The State 6 verifier closes at the independently observed early cut;
        # only the finalizer can supply a later external cut after its exit.
        limited_cutoff = ('pre_terminal' if external else 'pre_manifest') if (
            value.kind == 'retention' and self.expected.stage == 'state6'
            and (self.expected.role == 'state6_finalizer' or
                 (self.expected.role == 'state6_verifier' and not external))) else None
        need(satisfied(value.kind, result, state6_finalizer_cutoff=limited_cutoff),
             'component not satisfied: ' + value.kind)
        return result

    def result(self):
        if not self.failed and not self.unavailable:
            actual = [p.key for p in self.products if type(p) is Product]
            if len(set(actual)) != len(actual):
                self.unavailable.append('duplicate_output_publication')
            if set(actual) != {s.key for s in self.expected.outputs}:
                self.unavailable.append('declared_output_inventory_incomplete')
        completed = ('completed_simulation' if self.mode == 'synthetic'
                     else 'completed_component_validation')
        outcome = 'unresolved' if self.unavailable else 'failed' if self.failed else completed
        return WorkflowResult(self.expected.stage, self.expected.role, outcome, tuple(self.steps), tuple(self.products),
                              tuple(self.comparisons), self.primary, tuple(self.follow_on), tuple(self.unavailable),
                              route=self.mode)


def _entry(expected, independent_inputs, mode):
    run = _Run(expected, mode)
    try:
        validate_workflow_inputs(expected=expected, independent_inputs=independent_inputs, mode=mode)
    except BaseException as exc:
        run.fail(exc)
        run.unavailable.append('entry_prerequisites_unestablished')
    return run


@dataclass(frozen=True, slots=True)
class State4Ports:
    prepare: object
    generate_shard: object
    protect: object
    complete: object
    acquire_acceptance: object


@dataclass(frozen=True, slots=True)
class GeneratedControlShard:
    split: str
    shard: int
    rho_file: bytes
    controls_file: bytes
    manifest_file: bytes
    control_hashes: tuple


def validate_state4_inputs(*, expected, independent_inputs, mode):
    validate_workflow_inputs(expected=expected, independent_inputs=independent_inputs, mode=mode)
    need((expected.stage, expected.role) == ('state4', 'state4_worker'), 'State 4 worker required')
    need(all(s.kind in ('rho', 'controls', 'control_manifest', 'split_seed_manifest', 'rng_audit',
                        'read_only', 'pre_solve', 'controls_complete', 'evidence') for s in expected.outputs), 'State 4 controls only')


def run_state4_controls(*, expected, independent_inputs, ports, mode):
    run = _entry(expected, independent_inputs, mode)
    if run.failed:
        return run.result()
    try:
        validate_state4_inputs(expected=expected, independent_inputs=independent_inputs, mode=mode)
        need(type(ports) is State4Ports, 'explicit State4Ports')
        run.call('prepare', ports.prepare)
        generated = []
        for split, count in SPLITS:
            for shard in range(count):
                row = run.call(f'generate/{split}/{shard}', ports.generate_shard, split, shard, 64)
                need(type(row) is GeneratedControlShard and (row.split, row.shard) == (split, shard), 'generated shard binding')
                immutable(row)
                need(type(row.control_hashes) is tuple and len(row.control_hashes) == 64, 'generated control population')
                for h in row.control_hashes: authority.digest(h)
                for ref in (row.rho_file, row.controls_file, row.manifest_file): file_ref(ref)
                generated.append(row)
        need(len({h for row in generated for h in row.control_hashes}) == 4864, 'generated control collision')
        protected = run.call('protect', ports.protect)
        need(type(protected) is tuple and len(protected) == 232, '232 independent protection inputs required')
        for index, component in enumerate(protected):
            need(type(component) is ComponentInput and component.kind == 'state4_protected', 'protected component required')
            need(_arguments(component)['expected'].row_index == index, 'protected order')
            run.component(component)
        observed = {_arguments(c)['original'].file['relative_path']: _arguments(c)['original'].file for c in protected}
        for row in generated:
            for raw in (row.rho_file, row.controls_file, row.manifest_file):
                ref = file_ref(raw)
                need(observed.get(ref['relative_path']) == ref, 'generated/protected original substitution')
        run.call('complete', ports.complete)
        acquired = run.call('acquire_acceptance', ports.acquire_acceptance)
        need(type(acquired) is ComponentInput and acquired.kind == 'state4_acceptance', 'aggregate acquisition required')
        result = run.component(acquired)
        need(result.originals_checked == 235 and result.control_shards_checked == 76, 'complete original acceptance inventory')
        for made, accepted in zip(generated, _arguments(acquired)['expected'].control_shards, strict=True):
            need(made.control_hashes == tuple(accepted['ordered_control_tensor_sha256']), 'generated/accepted control substitution')
    except BaseException as exc:
        run.fail(exc)
    return run.result()


def _bind_finalizer_retention(*, expected, closure_inputs, execution_result):
    """Bind a late catalog observation to the compared finalizer exit."""
    spec = authority.decode(expected.launch.specification)
    execution_args = _arguments(closure_inputs[0])
    selected = authority.decode(execution_args['expected'].products)
    exit_file = selected['products']['exit']
    exited = authority.decode(execution_result.facts)['exit']
    need(exited is not None and exited['direct_exit_code'] == 0 and
         exited['reap_status'] == 'observed_reaped', 'independent finalizer direct exit required')
    exit_observation = exited['observation']

    late = closure_inputs[3]
    args = _arguments(late)
    catalog = retention._plain(args['catalog'])
    snapshots = retention._plain(args['snapshots'])
    files = retention._plain(args['independent_files'])
    admissions = retention._plain(args['upstream_admissions'])
    earlier = tuple(c for c in expected.component_inputs if c.kind == 'retention')
    need(len(earlier) == 1 and catalog == retention._plain(_arguments(earlier[0])['catalog']),
         'post-finalizer retention catalog substitution')
    earlier_history = retention._plain(_arguments(earlier[0])['snapshots'])
    need(type(earlier_history) is list and bool(earlier_history) and
         len(snapshots) > len(earlier_history) and
         snapshots[:len(earlier_history)] == earlier_history,
         'post-finalizer retention history substitution')
    need(type(snapshots) is list and bool(snapshots) and
         snapshots[-1]['cutoff'] == 'pre_terminal', 'post-finalizer retention cutoff')
    latest = snapshots[-1]
    capture = latest['capture']
    need(capture['invocation_id'] == publication._invocation(spec['publication_binding']),
         'post-finalizer retention publication invocation')
    if capture['clock_epoch_id'] == exit_observation['clock_epoch_id']:
        need(capture['first_sequence'] > exit_observation['sequence'] and
             capture['started_utc'] >= exit_observation['observed_utc'],
             'post-finalizer retention chronology')
    # Different trusted clock epochs have no comparable sequence numbers.  In
    # that case the qualified late admission below must include the exact
    # independently compared exit file at this snapshot's capture interval.
    binding = spec['binding']
    need(catalog['protocol_file_sha256'] == binding['protocol_file_sha256'] and
         catalog['implementation'] == binding['implementation'] and
         catalog['attempt_binding'] == dict(
             attempt_id=spec['publication_binding']['attempt_id'],
             state4_authorization_file_sha256=binding['state4_authorization_file_sha256']),
         'post-finalizer retention source/attempt substitution')
    registrations = {r['component']: r for r in catalog['component_registrations']}
    publication_registration = registrations['publication']
    need(spec['source_closure_file'] in publication_registration['source_files'] and
         spec['supervisor_source_file'] in publication_registration['source_files'] and
         spec['publication_binding']['publication_plan_file'] in publication_registration['plan_files'],
         'post-finalizer retention source/plan substitution')
    roots = {r['root_binding_id']: r for r in catalog['root_requirements']}
    snapshot_slots = [s for s in catalog['slots'] if s['address'] == latest['snapshot_address']]
    need(len(snapshot_slots) == 1 and snapshot_slots[0]['component'] == 'publication',
         'post-finalizer retention snapshot substitution')
    slot = snapshot_slots[0]
    root = roots[slot['address']['root_binding_id']]
    exit_address = {key: exit_file[key] for key in ('root_binding_id', 'relative_path')}
    exit_slots = [item for item in catalog['slots']
                  if item['component'] == 'publication' and item['address'] == exit_address]
    need(len(exit_slots) == 1, 'finalizer direct exit retention slot required')
    exit_slot = exit_slots[0]
    need(exit_slot['owner'] == 'accepted_attempt' and
         exit_slot['representation'] == 'loose_file',
         'finalizer direct exit ownership substitution')
    need(exit_slot['writer']['source_closure_file'] == spec['source_closure_file'] and
         exit_slot['writer']['source_file'] in
         (spec['source_closure_file'], spec['supervisor_source_file']),
         'finalizer direct exit writer source substitution')
    exit_root = roots[exit_address['root_binding_id']]
    exit_observations = [row for row in latest['observations']
                         if row['slot_id'] == exit_slot['slot_id'] and
                         row['address'] == exit_address]
    need(len(exit_observations) == 1 and
         exit_observations[0]['presence'] == 'present' and
         exit_observations[0]['content_state'] == 'complete' and
         exit_observations[0]['file'] == exit_file,
         'finalizer direct exit missing at pre_terminal cutoff')
    required = (root['identity_file'], root['custody_file'],
                exit_root['identity_file'], exit_root['custody_file'],
                slot['writer']['source_closure_file'], exit_file)
    available_files = {authority.canonical(row['file']) for row in files
                       if row['file'] is not None and row['data'] is not None}
    need(any(row['file'] == exit_file and row['data'] is not None for row in files),
         'finalizer direct exit absent from independent retention files')
    need(any(row['status'] == 'accepted' and row['scope'] == 'authority_and_custody' and
             all(ref in row['input_files'] for ref in required) and
             row['admission_file'] in publication_registration['admission_files'] and
             all(ref in publication_registration['source_files'] for ref in row['validator_source_files']) and
             all(authority.canonical(ref) in available_files for ref in
                 row['input_files'] + row['validator_source_files'] + [row['admission_file']]) and
             row['capture']['clock_epoch_id'] == capture['clock_epoch_id'] and
             row['capture']['invocation_id'] == capture['invocation_id'] and
             row['capture']['first_sequence'] <= capture['first_sequence'] and
             row['capture']['last_sequence'] >= capture['last_sequence'] and
             row['capture']['started_utc'] <= capture['started_utc'] and
             row['capture']['finished_utc'] >= capture['finished_utc']
             for row in admissions),
         'qualified finalizer exit/retention admission required')


def close_execution(*, expected, independent_inputs, worker_result, closure_inputs, mode):
    run = _entry(expected, independent_inputs, mode)
    if run.failed:
        return run.result()
    try:
        need(type(worker_result) is WorkflowResult and (worker_result.stage, worker_result.role) == (expected.stage, expected.role), 'bound worker result')
        need(worker_result.production_accepted is False and worker_result.execution_authorized is False, 'non-authorizing worker result')
        need(worker_result.route == mode, 'worker route substitution')
        run.steps.extend(worker_result.steps); run.products.extend(worker_result.products)
        run.comparisons.extend(worker_result.comparisons); run.unavailable.extend(worker_result.unavailable)
        run.primary = worker_result.primary; run.follow_on.extend(worker_result.follow_on)
        completed = ('completed_simulation' if mode == 'synthetic'
                     else 'completed_component_validation')
        run.failed = worker_result.outcome != completed or worker_result.primary is not None
        finalizer = expected.role == 'state6_finalizer'
        kinds = ('execution', 'guard', 'final_evidence', 'retention') if finalizer else ('execution', 'guard', 'final_evidence')
        need(type(closure_inputs) is tuple and tuple(c.kind for c in closure_inputs) == kinds,
             'independent closure boundaries required')
        require_execution_component(closure_inputs[0], expected, expected.role)
        phase = {'state4':'state4_closed', 'state5':'state5_closed', 'state6':'state6_child_closed'}[expected.stage]
        if expected.role == 'state6_finalizer':
            phase = 'terminal'
        need(_arguments(closure_inputs[2])['expected'].phase == phase, 'closure phase cutoff')
        for component in closure_inputs[:3]:
            run.component(component, external=finalizer)
        execution_result = run.comparisons[-3][1]
        guard_result = run.comparisons[-2][1]
        need(authority.decode(execution_result.facts)['guards'] == authority.decode(guard_result.closure), 'separate guard/process correspondence')
        inventory = tuple(c for c in expected.component_inputs if c.kind == 'retention')
        need(len(inventory) == 1, 'independent retention observation required')
        if finalizer:
            # The worker has already compared the pre-publication inventory.
            # Keep that assessment and compare separately acquired evidence
            # after the direct finalizer exit; do not re-date old observations.
            need(any(kind == 'retention' and satisfied(kind, result, state6_finalizer_cutoff='pre_manifest')
                     for kind, result in worker_result.comparisons),
                 'pre-publication retention assessment unavailable')
            need(closure_inputs[3] != inventory[0],
                 'post-finalizer retention must be separately acquired')
            _bind_finalizer_retention(expected=expected, closure_inputs=closure_inputs,
                                      execution_result=execution_result)
            run.component(closure_inputs[3], external=True)
        else:
            run.component(inventory[0])
        run.steps.append('independent_finalizer_closed' if finalizer else 'independent_child_closed')
    except BaseException as exc:
        run.fail(exc)
        run.unavailable.append('child_closure_unestablished')
    return run.result()


def close_state4_execution(*, expected, independent_inputs, worker_result, closure_inputs, mode):
    need(expected.stage == 'state4', 'State 4 closure')
    return close_execution(expected=expected, independent_inputs=independent_inputs,
                           worker_result=worker_result, closure_inputs=closure_inputs, mode=mode)
