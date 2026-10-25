"""V2 conditional prerequisites, recomputed from externally trusted evidence.

No acceptance issuer or execution grant. The external trust anchor must be an
independently accepted record, not a writer's self-description. Synthetic records
exercise correspondence only and are never eligible for a native State path.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib

from . import canonical as C
from .corpus_v2_custody import need, Population
from .state_v2_support import V2AdapterInputs
from .source_binding import reverify_source_set


def identity(value):
    return hashlib.sha256(C.canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class AcceptedRecord:
    """Externally selected bytes AND expectation; construction confers no trust."""
    path: Path
    sha256: str
    expected: bytes

    def read(self):
        data = Path(self.path).read_bytes()
        need(data == self.expected and hashlib.sha256(data).hexdigest() == self.sha256,
             'accepted prerequisite missing, changed or substituted')
        return C.strict_canonical_json_loads(data)


@dataclass(frozen=True)
class NativePrerequisites:
    adapter: V2AdapterInputs
    source_root: Path
    # These records are acquired independently of the corpus being judged.
    qualification_acceptance: AcceptedRecord
    state4_arguments: dict
    native_readiness: AcceptedRecord
    scientific_decisions: AcceptedRecord
    storage: object


@dataclass(frozen=True)
class AcceptedState4Inputs:
    """An independently accepted aggregate predecessor, not a writer claim.

    The validator/interval/scope premises are supplied by the upstream acceptance
    boundary. This consumes its immutable projection; it does not rerun permission
    handoffs or turn a serialized success boolean into State 4 acceptance.
    """
    projection_file: dict
    validator_files: tuple
    first_utc: str
    last_utc: str
    independent_acquisitions: tuple
    expected_binding: dict
    synthetic: bool


def read_accepted_state4(inputs):
    from .state_execution_authority import qualify, decode
    from .final_evidence_records import validate_accepted_controls_projection
    need(type(inputs) is AcceptedState4Inputs, 'independently accepted State 4 predecessor required')
    data=qualify(inputs.projection_file,inputs.independent_acquisitions,
        validators=inputs.validator_files,scope='state4_complete_aggregate_acceptance',
        first=inputs.first_utc,last=inputs.last_utc)
    projection=decode(data)
    validate_accepted_controls_projection(projection)
    need(projection['binding']==inputs.expected_binding,'State 4 predecessor binding substitution')
    need(projection['capture']['started_utc']>=inputs.first_utc and
         projection['capture']['finished_utc']<=inputs.last_utc,'State 4 predecessor cutoff')
    return data,projection


def require_accepted(record, *, kind, bindings):
    need(type(record) is AcceptedRecord, 'independently accepted '+kind+' record required')
    value = record.read()
    need(value.get('kind') == kind and value.get('accepted') is True and
         value.get('synthetic') is False and value.get('bindings') == bindings,
         'accepted '+kind+' scope/binding mismatch')
    return value


def verify_state4_controls(arguments, expected_controls, controls, *, synthetic=False):
    """Use the actual State 4 verifier, not a label or shared lock identity."""
    from .state4_acceptance_verify import verify_state4_acceptance_bundle
    from operatormark_corpus_v2.input_bindings import State4ControlExpectations
    need(type(synthetic) is bool and type(expected_controls) is State4ControlExpectations and
         expected_controls.synthetic is synthetic,
         'native State 4 control expectations required')
    if type(arguments) is AcceptedState4Inputs:
        need(arguments.synthetic is synthetic,'State 4 synthetic/native predecessor mismatch')
        data,projection=read_accepted_state4(arguments)
        shards=projection['ordered_control_shards']
    else:
        result = verify_state4_acceptance_bundle(**arguments)
        need(result.status == 'corroborated', 'State 4 independent acceptance correspondence incomplete')
        shards=arguments['expected'].control_shards
        rows = [row for row in arguments['independent_files'] if row.role == 'accepted_controls']
        need(len(rows)==1 and rows[0].data is not None,'State 4 independent projection missing')
        data=rows[0].data
    need(hashlib.sha256(data).hexdigest() == expected_controls.admission_record_sha256 and
         Path(expected_controls.admission_record).read_bytes() == data,
         'State 4 admission identity is not the independently verified accepted-controls record')
    admitted = {}
    for shard in shards:
        for offset, sha in enumerate(shard['ordered_control_tensor_sha256']):
            slot = (shard['split'], shard['shard_index'] * 64 + offset)
            need(slot not in admitted, 'duplicate State 4 control slot')
            admitted[slot] = sha
    expected_controls.verify(admitted, production=not synthetic)
    expected_controls.verify(controls.control_hashes(), production=not synthetic)
    return dict(admission_record_sha256=expected_controls.admission_record_sha256,
                aggregate_control_content_sha256=expected_controls.aggregate_control_content_sha256)


def operation_plan(population):
    """Closed algorithmic call set, NOT measured counts or a runtime estimate.

    One full independent pass normally; at most a second after required ladder
    publication. Each ladder runs T64/128/256 and P128/256/512 with eta correction.
    These are sessions, not assertions about underlying factorization iterations.
    """
    need(type(population) is Population, 'population required')
    n, d = population.samples, population.directions
    return dict(generation=dict(native_forward_sessions=n, native_tangents=d, p1_256_sessions=n),
        independent_readback_per_pass=dict(
            native_forward_sessions=n if population.layout.retention == 'E' else 0,
            native_tangents=d if population.layout.retention == 'E' else 0,
            native_assembly_residual_samples=n if population.layout.retention == 'S' else 0,
            native_tangent_matvecs=d if population.layout.retention == 'S' else 0,
            p1_256_sessions=n, direction_rederivations=d),
        full_readback_passes_maximum=2, anomaly_ladders_maximum=64, anomaly_policy='delivered_64_total',
        per_ladder_per_side=dict(tpfa_resolutions=[64,128,256], p1_resolutions=[128,256,512],
                                 eta_correction=True),
        ladder_sides=['writer','independent_reader'], measured=False,
        runtime_bound_seconds=None, forbidden_operation_counts=None)


def verify_native_prerequisites(prerequisites, *, population, config, readback, controls, directions, solver):
    from .state_v2_storage import require_complete_projection
    from operatormark_corpus_v2.config import load_qualification
    need(type(prerequisites) is NativePrerequisites and not population.synthetic,
         'native State prerequisites unavailable')
    a = prerequisites.adapter
    need(type(a) is V2AdapterInputs and a.expected.layout == population.layout,
         'source/runtime adapter mismatch')
    verified = reverify_source_set(a.source, expected=a.expected.source)
    paths = {row.path for row in verified.sources}
    required = {'operatormark/corpus/'+name+'.py' for name in (
        'state_v2_prerequisites','state_v2_workflow','state_v2_storage','state_v2_writer_bounds','corpus_v2_custody',
        'state_v2_support','corpus_v2_schema','writer_sizing','runtime','source_binding',
        'state4_acceptance_verify','state_storage_verify','state_execution','state_execution_authority',
        'state_guard_capture','state_custody_admission','bounded_exception','bounded_stream',
        'state_host_windows','state_host_native_win32','state_host_sizing','state_child_guard','rng_guard','source_import_guard',
        'final_evidence_records','publication_physical_records','canonical')}
    # Bind every delivered production module, including transitive numerical code.
    for package in ('operatormark_corpus_v2','operatormark_numerics_v2'):
        required.update(p.relative_to(prerequisites.source_root).as_posix()
                        for p in (prerequisites.source_root/package).rglob('*.py'))
    need(required <= paths, 'v2 source inventory incomplete')
    for row in verified.sources:
        need((prerequisites.source_root/row.path).read_bytes() == row.data, 'current v2 source drift')
    q = config.qualification
    need(q is not None and q.expectations is not None, 'accepted Q1-Q4 prerequisites missing')
    need(load_qualification(q.source_dirs, expected=q.expectations) == q, 'Q1-Q4 payload drift')
    q_bindings = dict(payloads={x.stage:x.payload_sha256 for x in q.expectations.entries},
                     review_record_sha256=q.expectations.review_record_sha256)
    require_accepted(prerequisites.qualification_acceptance, kind='corpus_v2_qualification', bindings=q_bindings)
    need(readback.qualification_expectations == q.expectations and readback.qualification_dirs == q.source_dirs,
         'writer/readback qualification substitution')
    state4 = verify_state4_controls(prerequisites.state4_arguments, readback.state4_expectations, controls)
    need(readback.controls_sha256 == state4['aggregate_control_content_sha256'], 'control aggregate substitution')
    source_hash = identity([dict(path=r.path,sha256=hashlib.sha256(r.data).hexdigest()) for r in verified.sources])
    bindings = dict(source_inventory_sha256=source_hash, runtime_record_sha256=a.expected.runtime_record_sha256,
                    inventory_sha256=a.expected.inventory_sha256, protocol_sha256=a.expected.protocol_sha256,
                    operation_plan_sha256=identity(operation_plan(population)))
    readiness=require_accepted(prerequisites.native_readiness, kind='corpus_v2_native_readiness', bindings=bindings)
    validate_qualified_operation_counts(readiness.get('authorized_operation_counts'))
    require_accepted(prerequisites.scientific_decisions, kind='corpus_v2_decisions', bindings=dict(
        n_ref=population.layout.n_ref, retention=population.layout.retention,
        retention_decision_sha256=config.retention.decision_reference_sha256,
        choices_reference_sha256=readback.choices_reference_sha256))
    need(prerequisites.storage.source_inventory_sha256==source_hash,'storage/P source binding differs')
    require_complete_projection(prerequisites.storage, population=population)
    # The configuration storage term is the length of the writer's own record
    # over these admitted inputs, not a number chosen by the projection's author.
    from .state_v2_writer_bounds import configuration_template
    template=next(x for x in prerequisites.storage.domains if x.domain=='generation_configuration_summary')
    need(template.basis=='configuration_template_summary_v1' and
         template.template==configuration_template(config,controls,directions,solver,'stop'),
         'configuration storage template differs from the admitted generation inputs')
    # A genuine State 4 projection cannot be paired with a smaller, unrelated
    # directory merely labelled as the prospective controls domain.
    data,projection=read_accepted_state4(prerequisites.state4_arguments)
    domain=next(x for x in prerequisites.storage.domains if x.domain=='state4_admission_control_records')
    verify_state4_retention(projection,data,domain.retained_files)
    config.require_production_inputs(controls, directions, solver, 'stop')
    return dict(**bindings, **state4, qualification_acceptance_sha256=prerequisites.qualification_acceptance.sha256)


def verify_state4_retention(projection,data,files):
    """Join independently accepted originals to the actual retained population."""
    required={(x['file']['relative_path'],x['file']['file_bytes'],x['file']['file_sha256'])
              for x in projection['original_file_inventory']}
    actual={(x.name,x.size,x.sha256) for x in files}
    need(required<=actual and any(x.size==len(data) and x.sha256==hashlib.sha256(data).hexdigest() for x in files),
         'State 4 control/admission retained population incomplete')


def verify_predecessor_audit_retention(projection,files):
    """The accepted State 4 projection names mandatory State 3/feasibility bytes.

    This does not infer the rest of the State 3 run/archive footprint from those
    two reports; that closed population remains an external retention premise.
    """
    actual={(x.name,x.size,x.sha256) for x in files}
    required={(projection[k]['filename'],projection[k]['size'],projection[k]['sha256'])
              for k in ('state3_audit','compute_feasibility_report')}
    need(required<=actual,'accepted predecessor audit/feasibility bytes not retained')


def validate_qualified_operation_counts(counts):
    """Expectations from accepted native qualification, never invented zeros.

    The existing observer's authorized families describe RNG/seed operations,
    not solver calls. The separate source-bound operation_plan states the closed
    solver/readback algorithm. Native observer calibration remains a prerequisite.
    """
    from .state_guard_capture import AUTHORIZED
    need(type(counts) is dict and set(counts)=={'state5','state6_readback','state6_finalizer'},
         'complete qualified stage operation expectations required')
    need(type(counts['state6_readback']) is dict and set(counts['state6_readback'])=={'1','2'},
         'qualified one/two-pass readback operation expectations required')
    rows=(counts['state5'],counts['state6_finalizer'],*counts['state6_readback'].values())
    for row in rows:
        need(type(row) is dict and set(row)==set(AUTHORIZED),'qualified operation family inventory')
        for family,splits in row.items():
            need(type(splits) is dict and set(splits)=={'training','validation','test'} and
                 all(type(n) is int and n>=0 for n in splits.values()),'qualified operation count/split')
    return counts


def qualified_operation_counts(counts,stage,*,readback_passes=0):
    validate_qualified_operation_counts(counts)
    need(stage in counts,'qualified operation stage')
    if stage=='state6_readback':
        need(type(readback_passes) is int and readback_passes in (1,2),'independent readback pass population')
        return counts[stage][str(readback_passes)]
    need(readback_passes==0,'unexpected readback operation population')
    return counts[stage]


def verify_launch_role(prepared,stage,source_expectations):
    """Refuse wrong version, role or source BEFORE a numerical entry point."""
    from .state_execution_authority import decode
    roles={'state5':'state5_solve_child','state6_readback':'state6_verification_child','state6_finalizer':'publication_finalizer'}
    spec=decode(prepared.expected.specification)
    need(stage in roles and spec['role']==roles[stage] and
         spec['binding']['benchmark']=='darcy_branch_c_v2' and
         spec['binding']['state']==('state5' if stage=='state5' else 'state6'),
         'v2 launch version/stage/role substitution')
    # Role/entry-point are stage-specific and already independently checked by
    # prepare_launch. The shared byte inventory and trust premises must match.
    actual=prepared.expected.source_expectations
    fields=('members','protocol_sha256','implementation_commit','implementation_tree','project_roots',
            'permitted_imports','external_imports','trust_premises')
    need(all(getattr(actual,k)==getattr(source_expectations,k) for k in fields),
         'launch/source expectations differ')
    return spec


def verify_execution(arguments, *, stage, expected_source_sha256, authorized_counts=None):
    """Actual independent process/stream/guard recomputation through exit."""
    from .state_execution import compare_execution_capture
    from .state_execution_authority import decode, prepare_launch
    from .state_guard_capture import AUTHORIZED, FORBIDDEN
    result = compare_execution_capture(**arguments)
    need(result.status == 'corroborated', stage+' process/stream evidence incomplete')
    facts = decode(result.facts)
    graph=decode(arguments['expected'].graph)
    launches=[prepare_launch(expected=x,independent_inputs=y) for x,y in
              zip(arguments['expected'].launches,arguments['expected'].launch_inputs,strict=True)]
    selected=[p for p in launches if p.slot_id==graph['selected_slot_id']]
    need(len(selected)==1,'unique independently verified launch required')
    launch=selected[0];spec=decode(launch.expected.specification)
    source=identity([dict(path=m.path,sha256=m.file_sha256) for m in launch.expected.source_expectations.members])
    roles={'state5':'state5_solve_child','state6_readback':'state6_verification_child','state6_finalizer':'publication_finalizer'}
    need(stage in roles and spec['role']==roles[stage] and source==expected_source_sha256,
         'execution role/source substitution')
    need(spec['binding']['benchmark']=='darcy_branch_c_v2','native v2 execution requires v2 records')
    need(facts['exit']['direct_exit_code'] == 0 and facts['exit']['reap_status'] == 'observed_reaped',
         stage+' process did not close successfully')
    guard = facts['guards']
    need(guard['binding']==spec['binding'] and guard['binding']['state'] == ('state5' if stage == 'state5' else 'state6'),
         'execution guard/stage substitution')
    need(set(guard['forbidden_counts'])==set(FORBIDDEN),'complete forbidden operation inventory')
    need(all(v.get('measurement') == 'measured' and v.get('value') == 0
             for v in guard['forbidden_counts'].values()), 'forbidden counts unmeasured or nonzero')
    need(all(guard['coverage'].get(k) is True for k in
             ('started_before_project_import','continued_through_exit','all_required_hooks_accounted')),
         'guard lifetime incomplete')
    need(type(authorized_counts) is dict and set(authorized_counts)==set(AUTHORIZED),
         'qualified v2 authorized-operation plan missing')
    for family in AUTHORIZED:
        need(set(authorized_counts[family])=={'training','validation','test'},'operation split inventory')
        for split,count in authorized_counts[family].items():
            need(type(count) is int and count>=0,'operation count expectation')
            actual=guard['authorized_counts'][family][split]
            need(actual['measurement']=='measured' and actual['value']==count,'authorized operation-plan mismatch')
    need(all(v is True for row in guard['rng_state_equalities'].values() for v in row.values()),
         'legacy RNG state changed or unmeasured')
    # Count observations are retained, never fabricated from the plan.
    return facts


def verify_storage(arguments, *, phase, source_sha256, files):
    from .state_storage_verify import verify_later_storage_boundary
    expected = arguments['expected']
    need(expected.phase == phase and expected.binding.source_sha256 == source_sha256,
         'physical phase/source substitution')
    content = {(x.relative_path,x.content_file.file_bytes,x.content_file.file_sha256)
               for x in expected.outputs}
    need({(x.name,x.size,x.sha256) for x in files} <= content, 'physical output population missing')
    result = verify_later_storage_boundary(**arguments)
    need(result['status'] == 'corroborated', 'independent '+phase+' physical custody incomplete')
    return result
