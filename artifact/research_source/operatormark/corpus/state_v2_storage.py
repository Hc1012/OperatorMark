"""Prospective v2 writer accounting and exact after-write retention accounting.

Finite tensor/row populations do not bound arbitrary exception text or observer
streams. Unknowns propagate; a proposed operational limit is never a proven byte
bound. C is included in E, never added a second time.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math

from . import canonical as C
from .corpus_v2_custody import Population, FileFact, file_population, classic_archive_size, need

GiB = 2**30
# Every State 6 bound (readback records, sole-writer custody counts) assumes
# continuous exclusive custody of both roots. No in-tree record establishes it,
# so it stays an explicit unresolved admission term rather than vanishing once
# a conditional arithmetic bound is supplied.
PREMISE_P = 'P:continuous exclusive custody of the State 6 payload and evidence roots (no accepted premise record)'
VARIABLE_DOMAINS = ('state4_admission_control_records', 'state3_6_provider_guard_process_records',
    'source_runtime_records', 'generation_configuration_summary', 'generation_failure_records',
    'generation_run_and_exception_records', 'readback_diagnostics', 'custody_records',
    'prior_failure_retention', 'stdout_stderr')


@dataclass(frozen=True)
class DomainEvidence:
    domain: str
    # Actual immutable files, or a source-qualified bound supplied externally.
    bytes: int | None
    basis: str
    source_sha256: str | None = None
    retained_root: Path | None = None
    retained_files: tuple[FileFact,...] = ()
    population: Population | None = None
    # Canonical configuration template bytes (generation_configuration_summary).
    template: bytes | None = None


@dataclass(frozen=True)
class RetainedInputs:
    """Independently selected original record populations, outside the attempt.

    Support includes the consumed launch/observer/provider records and streams;
    State 4 includes its original control/admission population. These expectations
    are not discovered from a writer's own manifest or an acceptance boolean.
    """
    state4_root: Path
    state4_files: tuple[FileFact,...]
    support_root: Path
    support_files: tuple[FileFact,...]
    # ((original root_binding_id, original relative_path), (retained group, name))
    # Exact injective placement: equal bytes at distinct source addresses do not
    # silently erase one retained original.
    placements: tuple = ()


def acquired_retention_members(inputs):
    """Exact raw acquisition files actually consumed by existing verifiers."""
    from dataclasses import fields, is_dataclass
    from .state_execution_authority import Acquisition, decode
    from .state_provider_bindings import ProviderAcquisition
    result=set()
    def walk(value):
        if type(value) in (Acquisition,ProviderAcquisition):
            if value.data is not None:
                f=decode(value.file) if type(value) is Acquisition else {
                    'root_binding_id':value.file.root_binding_id,
                    'relative_path':value.file.relative_path,'file_bytes':value.file.file_bytes,
                    'file_sha256':value.file.file_sha256}
                need((len(value.data),hashlib.sha256(value.data).hexdigest())==
                     (f['file_bytes'],f['file_sha256']),'retained acquisition bytes mismatch')
                result.add((f['root_binding_id'],f['relative_path'],f['file_bytes'],f['file_sha256']))
        elif is_dataclass(value):
            for field in fields(value):walk(getattr(value,field.name))
        elif type(value) is dict:
            for item in value.values():walk(item)
        elif type(value) in (tuple,list):
            for item in value:walk(item)
    walk(inputs)
    return result


def verify_retained_inputs(inputs, *, attempt_root, required, local=()):
    from . import corpus_v2_custody as custody
    need(type(inputs) is RetainedInputs,'independent complete retained-input populations required')
    roots=tuple(Path(x).absolute() for x in (inputs.state4_root,inputs.support_root,attempt_root))
    need(all(not a.is_relative_to(b) for a in roots for b in roots if a!=b) and len(set(roots))==3,
         'retained domains overlap or duplicate attempt bytes')
    state4=custody.snapshot(roots[0],[x.name for x in inputs.state4_files])
    support=custody.snapshot(roots[1],[x.name for x in inputs.support_files])
    custody.same_content(inputs.state4_files,state4);custody.same_content(inputs.support_files,support)
    # The attempt's own independently read files need not be copied into support.
    # Keep exact path/size/hash correspondence; a same-hash alias is not enough.
    actual={(group,x.name):(x.size,x.sha256) for group,rows in
            (('state4',state4),('support',support),('attempt',tuple(local))) for x in rows}
    need(type(inputs.placements) is tuple,'independent retained placement inventory')
    places={}
    for address,target in inputs.placements:
        need(type(address) is tuple and len(address)==2 and type(target) is tuple and len(target)==2 and
             address not in places and target not in places.values(),'duplicate retained source/target address')
        places[address]=target
    need(set(places)=={(root,name) for root,name,_,_ in required},'consumed retention address coverage')
    for root,name,size,sha in required:
        need(actual.get(places[root,name])==(size,sha),'consumed independent evidence absent from retained domains')
    return state4,support


def recheck_domain(item):
    """Only closed, already retained populations have an implemented proof.

    Future variable writers need qualified lossless operational policies. A label,
    copied source hash or a number cannot stand in for that missing proof.
    """
    from . import corpus_v2_custody as custody
    if item.domain=='generation_failure_records' and item.basis=='pinned_native_failure_rows_v1':
        from . import state_v2_writer_bounds as writers
        need(item.retained_root is None and not item.retained_files,
             'prospective failure bound cannot substitute retained-file claims')
        need(item.source_sha256==hashlib.sha256(Path(writers.__file__).read_bytes()).hexdigest(),
             'failure-domain verifier source drift')
        value=writers.native_failure_bound(item.population)
        need(item.bytes==value['bytes'],'failure-domain claimed bytes differ from source-derived bound')
        return value['bytes']
    if item.domain=='generation_configuration_summary' and item.basis=='configuration_template_summary_v1':
        from . import state_v2_writer_bounds as writers
        need(item.retained_root is None and not item.retained_files,
             'prospective configuration bound cannot substitute retained-file claims')
        need(item.source_sha256==hashlib.sha256(Path(writers.__file__).read_bytes()).hexdigest(),
             'configuration-domain verifier source drift')
        value=writers.configuration_summary_bound(item.population,item.template)
        need(item.bytes==value['bytes'],'configuration-domain claimed bytes differ from source-derived bound')
        return value['bytes']
    if item.domain=='readback_diagnostics' and item.basis=='readback_result_records_v1':
        from . import state_v2_writer_bounds as writers
        need(item.retained_root is None and not item.retained_files and item.template is None,
             'prospective readback bound cannot substitute retained-file claims')
        need(item.source_sha256==hashlib.sha256(Path(writers.__file__).read_bytes()).hexdigest(),
             'readback-domain verifier source drift')
        value=writers.readback_records_bound(item.population)
        need(item.bytes==value['bytes'],'readback-domain claimed bytes differ from source-derived bound')
        return value['bytes']
    if item.domain=='generation_run_and_exception_records' and item.basis=='adopted_A_run_exception_v1':
        from . import state_v2_writer_bounds as writers
        need(item.retained_root is None and not item.retained_files and item.template is None,
             'prospective exception bound cannot replace retained evidence')
        need(item.source_sha256==hashlib.sha256(Path(writers.__file__).read_bytes()).hexdigest(),'exception proof source drift')
        value=writers.generation_run_exception_bound(item.population)
        need(item.bytes==value['bytes'],'run/exception source-derived bound differs')
        return value['bytes']
    need(item.domain in ('state4_admission_control_records','source_runtime_records','prior_failure_retention'),
         'future variable domain has no qualified finite producer bound')
    need(item.basis=='independently_retained_preexisting' and item.retained_root is not None and item.retained_files,
         'unproved prospective domain bound')
    producer=hashlib.sha256(Path(custody.__file__).read_bytes()).hexdigest()
    need(item.source_sha256==producer,'retained-domain reader/source drift')
    actual=custody.snapshot(item.retained_root,[x.name for x in item.retained_files])
    custody.same_content(item.retained_files,actual)
    size=sum(x.size for x in actual)
    need(item.bytes==size,'domain byte claim differs from independent readback')
    return size


@dataclass(frozen=True)
class Projection:
    population: Population
    raw: int
    headers: int
    primary_jsonl: int
    fixed_metadata: int
    domains: tuple[DomainEvidence, ...]
    controls_subtotal: int | None
    archive_bytes: int | None
    estimated_accepted_bytes: int | None
    required_free_bytes: int | None
    unresolved: tuple[str, ...]
    # Even a complete estimate isn't native capacity qualification.
    native_capacity_qualified: bool = False
    custody_admission: object = None
    archive_members: tuple | None = None
    source_inventory_sha256: str | None = None


def proposed_operational_policy():
    """Concrete proposal for review; not activated or used as a sizing fact."""
    return dict(status='A_B_C_adopted_P_requires_external_acceptance', attempts=1,
        per_stream_bytes=64*2**20, retained_exception_bytes=16*2**20,
        observer_custody_bytes=None,
        action='refuse before launch without complete producer accounting and failure reserve',
        overflow='stop owned work at 64 MiB per stream; retain bounded prefix; disclose discarded observed bytes and unknown unobserved bytes',
        limitation='retained rendering is bounded, formatting time/memory is not; EOF and full retention are distinct')


def prospective_projection(population, *, domains=(), controls_subtotal=None, archive_members=None,
                           custody_admission=None, source_inventory_sha256=None):
    """Arithmetic only. Domain numbers require a reviewed producer-bound proof.

    Passing zero with no proof cannot fill a gap. Existing actual records can be
    budgeted exactly, but their size is not a bound for future records.
    """
    from operatormark_corpus_v2 import layout as L, sizing
    need(type(population) is Population, 'population required')
    layout = L.Layout(population.counts, population.shard_size, population.synthetic)
    plan = sizing.file_plan(population.layout.n_ref, population.layout.retention, layout)
    own = {name: shape for name, kind, shape in file_population(population) if kind == 'tensor'}
    need({x['relative_path']:tuple(x['shape']) for x in plan} == own, 'writer/inventory disagreement')
    raw = sum(8*math.prod(shape) for shape in own.values())
    need(raw == sum(x['data_bytes'] for x in plan), 'raw writer arithmetic')
    headers = sum(x['header_bytes'] for x in plan)
    bounds = sizing.record_bounds(layout,population.layout.n_ref,population.layout.retention)
    primary = sum(count*(bounds[s]['sample_record']+bounds[s]['direction_record']*L.DIRECTIONS_PER_SAMPLE[s])
                  for s,count in zip(L.SPLITS,population.counts) if count)
    # Shard manifests have closed key sets, hash widths and fixed tensor slots.
    fixed = sum(layout.shards(s)*sizing._shard_manifest_bound(layout,s,population.layout.n_ref,
                  population.layout.retention,bounds[s]) for s in bounds)
    indexed = {}
    for item in domains:
        need(type(item) is DomainEvidence and item.domain in VARIABLE_DOMAINS and item.domain not in indexed,
             'exact unique storage domain')
        need(item.bytes is None or type(item.bytes) is int and item.bytes >= 0, 'storage domain byte bound')
        if item.bytes is not None:
            if item.basis in ('pinned_native_failure_rows_v1','configuration_template_summary_v1',
                              'readback_result_records_v1','adopted_A_run_exception_v1'):
                need(item.population==population,'writer-bound population substitution')
            recheck_domain(item)
        indexed[item.domain] = item
    rows = tuple(indexed.get(name,DomainEvidence(name,None,'unresolved writer/operational domain'))
                 for name in VARIABLE_DOMAINS)
    unresolved = [x.domain for x in rows if x.bytes is None]
    state4=indexed.get('state4_admission_control_records')
    derived_c=None
    if state4 is not None and state4.bytes is not None:
        # Actual retained State 4 domain plus this writer's rho copies. Both are
        # real retained bytes when staging consumes, rather than moves, controls.
        derived_c=state4.bytes+sum(x['data_bytes']+x['header_bytes'] for x in plan
                                 if x['relative_path'].endswith('/rho.float64.npy'))
    need(controls_subtotal is None or type(controls_subtotal) is int and controls_subtotal==derived_c,
         'C requires independently recomputed State 4 and writer populations')
    controls_subtotal=derived_c
    if derived_c is None:unresolved.append('C')
    # A whole-attempt move creates no second corpus copy. A partial file replaces
    # its intended final slot; it is retained on failure, not counted twice.
    # Unwritten measurements can coexist with an interrupted shard: reserve N rows.
    unwritten = sum(count*sizing._measurement_bound(layout,s,population.layout.n_ref,population.layout.retention)
                    for s,count in zip(L.SPLITS,population.counts) if count)
    # Final ladder publication plus failure-only completed-row retention: each
    # exclusive writer may leave both temporary and final names (four total).
    fixed += unwritten + 4*sizing.MAX_ANOMALY_LADDERS*sizing._ladder_bound()
    archive = None if archive_members is None else classic_archive_size(archive_members)
    if archive is None: unresolved.append('B:exact audit membership and byte bounds')
    if custody_admission is None:unresolved.append(PREMISE_P)
    else:
        from .state_custody_admission import require_custody
        require_custody(custody_admission,source_sha256=source_inventory_sha256)
    e = None if any(x.bytes is None for x in rows) else raw+headers+primary+fixed+sum(x.bytes for x in rows)
    if e is not None and controls_subtotal is not None: need(controls_subtotal <= e, 'C must be a subtotal of E')
    free = None if unresolved else max(5*GiB,4*e+archive+GiB)
    return Projection(population,raw,headers,primary,fixed,rows,controls_subtotal,archive,e,free,tuple(unresolved),False,custody_admission,
                      None if archive_members is None else tuple(archive_members),source_inventory_sha256)


def require_complete_projection(value, *, population):
    need(type(value) is Projection and value.population == population, 'v2 storage projection missing/substituted')
    # Recompute, do not trust supplied E/free fields.
    need(not value.unresolved and value.estimated_accepted_bytes is not None and value.archive_bytes is not None,
         'prospective storage unknown: '+', '.join(value.unresolved))
    # The caller's `unresolved` field is not trusted to carry P: an accepted P
    # record cannot be replaced by a caller assertion; requalify its actual evidence.
    from .state_custody_admission import require_custody
    need(value.custody_admission is not None,'prospective storage unknown: '+PREMISE_P)
    require_custody(value.custody_admission,source_sha256=value.source_inventory_sha256)
    need(value.archive_members is not None and classic_archive_size(value.archive_members)==value.archive_bytes,
         'archive bound requires exact prospective membership')
    need(tuple(x.domain for x in value.domains)==VARIABLE_DOMAINS,'complete unique storage domains required')
    for domain in value.domains: recheck_domain(domain)
    independent=prospective_projection(population)
    need((value.raw,value.headers,value.primary_jsonl,value.fixed_metadata)==
         (independent.raw,independent.headers,independent.primary_jsonl,independent.fixed_metadata),
         'fixed writer-domain substitution')
    derived=prospective_projection(population,domains=value.domains)
    need(value.controls_subtotal==derived.controls_subtotal,'C retained-domain substitution')
    expected = value.raw+value.headers+value.primary_jsonl+value.fixed_metadata+sum(x.bytes for x in value.domains)
    need(value.estimated_accepted_bytes == expected and value.controls_subtotal <= expected and
         value.required_free_bytes == max(5*GiB,4*expected+value.archive_bytes+GiB), 'storage formula mismatch')
    # Conditional correspondence is not a native free-space observation.
    return value.required_free_bytes


def exact_retention_accounting(*, payload, evidence, archive, controls=None, sidecars=(), state4=(), support=(), terminal_bytes=None):
    """Closed retained byte population, including failed/partial files supplied.

    The caller must independently enumerate each retained root. This never calls
    a past failed attempt complete, nor predicts future runs from present sizes.
    """
    groups = (('payload',payload),('evidence',evidence),('sidecars',sidecars),('state4',state4),('support',support))
    allrows = [(root,row) for root,rows in groups for row in rows]
    need(all(type(row) is FileFact for _,row in allrows), 'independent FileFact population')
    need(len({(root,row.name) for root,row in allrows}) == len(allrows), 'duplicate retention slot')
    members = {(root,row.name):row for root,row in allrows}
    need(controls is None or len(set(controls)) == len(controls) and set(controls) <= set(members), 'C membership is not a subtotal')
    # Actual NPY headers are accounted at readback, not guessed from file suffix.
    # This aggregate is bytes on disk, separately from prospective R/H/J/M.
    need(terminal_bytes is None or type(terminal_bytes) is int and terminal_bytes>=0,'terminal byte count')
    known = sum(row.size for _,row in allrows)
    total = None if terminal_bytes is None else known+terminal_bytes
    c = None if controls is None else sum(members[key].size for key in controls)
    need(type(archive) is FileFact, 'independently verified audit archive required')
    return dict(retained_primary_bytes=total, known_primary_bytes_excluding_terminal=known,
        controls_subtotal_bytes=c, archive_bytes=archive.size,
        terminal_bytes=terminal_bytes,retained_join_complete=controls is not None and terminal_bytes is not None,
        required_free_if_this_were_the_complete_bound=None if controls is None or terminal_bytes is None else max(5*GiB,4*total+archive.size+GiB),
        prospective_bound=False, native_capacity_qualified=False,
        formula='E=R+H+J+M; C subtotal; required_free=max(5 GiB,4E+B+1 GiB)')
