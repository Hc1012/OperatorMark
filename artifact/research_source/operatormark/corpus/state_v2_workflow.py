"""Versioned Drop 2 State 5/6 staging, independent readback and custody route.

Native use is conditional on external qualification, State 4, source/runtime,
capacity, launch and provider evidence. This module grants no execution authority.
The synthetic route exercises bytes and transitions, never physical acceptance.
No existing v1 workflow, population, scientific selector or guard is changed.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import sys
import traceback
import hashlib

from . import canonical as C
from . import corpus_v2_custody as custody
from .corpus_v2_custody import need, Population
from .state_v2_prerequisites import (verify_native_prerequisites, verify_execution,
    verify_storage, verify_launch_role, qualified_operation_counts, verify_state4_retention,
    verify_predecessor_audit_retention, operation_plan, identity)
from .state_v2_storage import exact_retention_accounting, acquired_retention_members, verify_retained_inputs


@dataclass(frozen=True)
class Paths:
    staging: Path
    published: Path
    evidence: Path
    archive: Path
    complete: Path
    terminal: Path

    def validate(self, *, whole_attempt=False):
        values = tuple(Path(getattr(self,k)).absolute() for k in self.__dataclass_fields__)
        need(len(set(values)) == 6,'distinct custody destinations required')
        if whole_attempt:
            need(all(p.parent==values[0] for p in values[2:]),'whole-attempt sidecars must be direct children')
            need(not values[0].is_relative_to(values[1]) and not values[1].is_relative_to(values[0]),'publication root overlap')
            checked=values[:2]
        else:
            need(all(not a.is_relative_to(b) for a in values for b in values if a!=b),
             'distinct nonoverlapping custody destinations required')
            checked=values
        need(all(not p.exists() for p in values), 'fresh destinations required; no retry/overwrite')
        for p in checked:custody.checked_directory(p.parent)
        need(values[0].parent.stat().st_dev == values[1].parent.stat().st_dev,'same-volume publication required')


@dataclass(frozen=True)
class NativePorts:
    """Existing independent acquisition/launch mechanisms, not outcome callbacks.

    Each callback returns inputs for an actual existing verifier. The trusted
    caller supplies their independently accepted premises. No bool/status result
    can substitute for these comparisons.
    """
    launch_inputs: object       # stage -> (LaunchExpectations, IndependentLaunchInputs)
    storage_inputs: object      # (phase, current FileFact tuple) -> verifier kwargs
    execution_inputs: object    # stage -> actual compare_execution_capture kwargs
    retention_inputs: object   # () -> independently selected RetainedInputs
    retention_members: object = None  # () -> complete independently bound host support readback


def validate_assessment(assessment, population, *, native):
    checked = assessment['checked']
    need((checked['samples'],checked['directions'],checked['shards_expected'],checked['shards_written']) ==
         (population.samples,population.directions,len(population.shards()),len(population.shards())),
         'independent readback population incomplete')
    need(not assessment['recomputed']['failures_retained'] and
         not assessment['recomputed']['unwritten_measurements'], 'retained failed measurements prevent publication')
    if native:
        need(assessment['status'] == 'matched' and assessment['d10']['verdict'] == 'accepted' and
             assessment['production_usable'] is True, 'physical qualification/readback incomplete or failed')
        need(checked['numerics'] == 'full' and checked['p1'] == 'resolve' and
             checked['native_numerics_checked'] == population.samples and
             checked['p1_resolved'] == population.samples and checked['direction_rule_rederived'] is True,
             'native/P1/direction independent obligations incomplete')
    else:
        need(checked['numerics'] == 'structural' and checked['p1'] == 'skip' and
             checked['native_numerics_checked'] == 0 and checked['p1_resolved'] == 0,
             'synthetic route cannot perform native validation')
        need(assessment['status'] in ('matched','incomplete') and not assessment['production_usable'],
             'synthetic consistency failure or false production claim')
        need(all(row[0] == 'incomplete' and row[2] in
                 ('native_numerics_not_verified','p1_not_resolved','anomaly_ladder_not_independently_verified')
                 for row in assessment['diagnostics']), 'unexpected incomplete readback obligation')


def _class_name(obj):
    """Class qualname without metaclass descriptors or user hooks."""
    cls=type(obj)
    try:
        name=type.__getattribute__(cls,'__qualname__') if type(cls) is type else None
    except BaseException:
        name=None
    return name if type(name) is str else '?unavailable'


def _record_text(text):
    """Canonical UTF-8 JSON cannot carry lone surrogates (e.g. an undecodable
    path in a message). Keep such text exactly as surrogatepass hex instead of
    losing the whole failure record; ordinary text is unchanged."""
    try:
        text.encode('utf-8')
        return text,None
    except UnicodeEncodeError:
        return None,text.encode('utf-8','surrogatepass').hex()


class V2Workflow:
    """One attempt, no retry, explicit phase entry points for owned workers.

    Retained stage records survive catchable errors. Native callers must close
    each owned worker independently before entering its dependent phase. The
    terminal reader is called by the supervisor after finalizer exit, never by
    the finalizer itself. Component fixtures are explicitly in-process/synthetic.
    """
    def __init__(self, *, population, config, expected_readback, controls, directions,
                 solver, paths, prerequisites=None, ports=None, handoff=None):
        from operatormark_corpus_v2.config import GenerationConfig
        from operatormark_corpus_v2.readback import Expected
        need(type(population) is Population and type(config) is GenerationConfig and
             type(expected_readback) is Expected and type(paths) is Paths, 'exact v2 inputs required')
        need((config.n_ref,config.retention.option,config.layout.counts,config.layout.shard_size,config.synthetic) ==
             (population.layout.n_ref,population.layout.retention,population.counts,population.shard_size,population.synthetic),
             'generation population substitution')
        need((expected_readback.n_ref,expected_readback.retention,expected_readback.counts,
              expected_readback.shard_size,expected_readback.mode) ==
             (config.n_ref,config.retention.option,population.counts,population.shard_size,config.mode),
             'readback population substitution')
        need(tuple(config.shards()) == population.shards() and tuple(expected_readback.shards()) == population.shards(),
             'partial/duplicate selection cannot publish a corpus')
        self.population,self.config,self.readback=population,config,expected_readback
        self.controls,self.directions,self.solver,self.paths=controls,directions,solver,paths
        self.prerequisites,self.ports=prerequisites,ports
        self.native=not population.synthetic
        self.whole_attempt=paths.evidence.parent in (paths.staging,paths.published)
        need(not self.native or self.whole_attempt,'native publication must move the whole attempt, including custody')
        self.phase='new';self.files=();self.bindings={};self.assessment=None
        self.closed=set();self.events=[];self.failed=False
        self.required_support=set()
        if self.native:
            need(type(ports) is NativePorts,'native acquisition/closure ports required')
            self._prerequisites()
        else:
            need(prerequisites is None and ports is None,'synthetic fixtures cannot supply native acceptances')
            need(controls.binding().get('synthetic') is True and directions.binding().get('synthetic') is True and
                 solver.kind == 'synthetic', 'synthetic writer inputs required')
        if handoff is not None:
            self._restore(handoff)
            return
        paths.validate(whole_attempt=self.whole_attempt)
        if self.whole_attempt:paths.staging.mkdir()
        paths.evidence.mkdir()
        self._record('entry',dict(population=population.record(),bindings=self.bindings,
                                 operations=operation_plan(population),native_readiness=self.native))

    def _prerequisites(self):
        if self.native:
            self.bindings=verify_native_prerequisites(self.prerequisites,population=self.population,
                config=self.config,readback=self.readback,controls=self.controls,directions=self.directions,solver=self.solver)
            from .state_custody_admission import recheck_live_custody
            payload=self._data_root()
            recheck_live_custody(self.prerequisites.storage.custody_admission,
                source_sha256=self.bindings['source_inventory_sha256'],
                roots={'payload':str(payload.absolute()),'evidence':str(self.paths.evidence.absolute())})
            self.required_support.update(acquired_retention_members(self.prerequisites.storage.custody_admission))
            self.required_support.update(acquired_retention_members(self.prerequisites.state4_arguments))
            for role,record in (('accepted_qualification',self.prerequisites.qualification_acceptance),
                                ('accepted_readiness',self.prerequisites.native_readiness),
                                ('accepted_decisions',self.prerequisites.scientific_decisions)):
                self.required_support.add((role,Path(record.path).name,len(record.expected),record.sha256))

    def _context(self):
        # Only commitments, never private source values or seeds, are retained.
        source_names=('state_v2_workflow','state_v2_prerequisites','state_v2_storage','corpus_v2_custody',
                      'state_execution','state_execution_authority','state_guard_capture','final_evidence_records',
                      'publication_physical_records')
        return identity(dict(population=self.population.record(),whole_attempt=self.whole_attempt,
            location_commitment=identity(dict(staging=str(self.paths.staging.absolute()),published=str(self.paths.published.absolute()),
                sidecars={k:getattr(self.paths,k).name for k in ('evidence','archive','complete','terminal')})),
            config=hashlib.sha256(repr(self.config).encode()).hexdigest(),
            readback=hashlib.sha256(repr(self.readback).encode()).hexdigest(),
            controls=self.controls.binding(),directions=self.directions.binding(),solver=self.solver.kind,
            source={n:hashlib.sha256(Path(__file__).with_name(n+'.py').read_bytes()).hexdigest() for n in source_names}))

    def _data_root(self):
        root=self.paths.published if self.paths.published.exists() and not self.paths.staging.exists() else self.paths.staging
        return root/'payload' if self.whole_attempt else root

    def handoff(self):
        """Immutable, source-bound phase handoff; no live objects are deserialized.

        Adopted C: written only by the sole evidence writer, and never
        twice in a row: each handoff must follow at least one non-handoff record
        written since the previous one. The intended native lifecycle uses two per
        phase (the worker's, then the supervisor's after close_stage), six in all;
        recovery restores from the existing handoff and writes nothing.
        """
        need(not self.failed and self.phase in ('staged','verified','published'),'handoff boundary')
        custody.same_content(tuple(self.events),
            custody.exact_record_root(self.paths.evidence,[x.name for x in self.events]))
        need(not self.events[-1].name.endswith('_handoff.json'),
             'consecutive handoff: restore from the existing handoff instead')
        need(sum(x.name.endswith('_handoff.json') for x in self.events)<21,'handoff policy maximum 21')
        from dataclasses import asdict
        body=dict(context_sha256=self._context(),phase=self.phase,closed=sorted(self.closed),
            files=[asdict(x) for x in self.files],events=[asdict(x) for x in self.events],
            assessment=self.assessment,bindings=self.bindings,
            required_support=sorted(self.required_support),
            archive_members=[asdict(x) for x in getattr(self,'archive_members',())],
            archive_fact=asdict(self.archive_fact) if hasattr(self,'archive_fact') else None,
            complete_fact=asdict(self.complete_fact) if hasattr(self,'complete_fact') else None)
        return self._record('handoff',body)

    def _restore(self,handoff):
        """Reacquire every predecessor, then recheck native process closure."""
        need(type(handoff) is custody.FileFact,'independently selected handoff FileFact required')
        custody.same_content((handoff,),(custody.acquire_file(self.paths.evidence,handoff.name),))
        row=C.strict_canonical_json_loads((self.paths.evidence/handoff.name).read_bytes())
        C.validate_self_hash(row,'record_self_sha256')
        need(row['kind']=='handoff' and row['synthetic'] is self.population.synthetic,'handoff kind/mode')
        value=row['value'];need(value['context_sha256']==self._context() and value['bindings']==self.bindings,
                                'handoff source/configuration substitution')
        self.required_support.update(tuple(x) for x in value['required_support'])
        self.events=[custody.FileFact(**x) for x in value['events']]
        custody.same_content(tuple(self.events)+(handoff,),custody.exact_record_root(self.paths.evidence,
                             [x.name for x in self.events]+[handoff.name]))
        previous=None;kinds=[]
        for fact in self.events+[handoff]:
            event=C.strict_canonical_json_loads((self.paths.evidence/fact.name).read_bytes())
            C.validate_self_hash(event,'record_self_sha256')
            need(event['previous_sha256']==previous and event['synthetic'] is self.population.synthetic,
                 'handoff predecessor chain mismatch')
            previous=fact.sha256;kinds.append(event['kind'])
        phase=value['phase'];need(phase in ('staged','verified','published'),'handoff phase')
        required={'staged':'staged','verified':'verified','published':'promoted'}[phase]
        need(required in kinds and 'failure' not in kinds,'handoff phase evidence missing')
        self.files=tuple(custody.FileFact(**x) for x in value['files'])
        phase_records=[C.strict_canonical_json_loads((self.paths.evidence/x.name).read_bytes()) for x in self.events
                       if x.name.endswith('_'+required+'.json')]
        need(len(phase_records)==1 and phase_records[0]['value']['files']==[x.content() for x in self.files],
             'handoff differs from retained phase output')
        location=self._data_root()
        custody.same_content(self.files,custody.exact_generated_population(location,self.population,
            ladders=any(x.name=='anomaly_ladders.jsonl' for x in self.files)))
        self.phase=phase;self.assessment=value['assessment']
        if phase in ('verified','published'):
            readbacks=[C.strict_canonical_json_loads((self.paths.evidence/x.name).read_bytes())['value']
                       for x in self.events if '_readback_' in x.name]
            need(readbacks and self.assessment==readbacks[-1],'handoff independent readback substitution')
            validate_assessment(self.assessment,self.population,native=self.native)
        if phase=='published':
            self.archive_members=tuple(custody.FileFact(**x) for x in value['archive_members'])
            self.archive_fact=custody.FileFact(**value['archive_fact']);self.complete_fact=custody.FileFact(**value['complete_fact'])
            custody.verify_archive(self.paths.archive,self.archive_members)
        for stage in value['closed']:
            need(stage in ('state5','state6_readback','state6_finalizer') and 'closure_'+stage in kinds,
                 'handoff closure evidence missing')
            if self.native:self._verify_closed(stage)
            self.closed.add(stage)
        self.events.append(handoff)

    def _verify_closed(self,stage):
        readiness=self.prerequisites.native_readiness.read()
        passes=sum('_readback_' in x.name for x in self.events) if stage=='state6_readback' else 0
        counts=qualified_operation_counts(readiness.get('authorized_operation_counts'),stage,readback_passes=passes)
        inputs=self.ports.execution_inputs(stage)
        result=verify_execution(inputs,stage=stage,
            expected_source_sha256=self.bindings['source_inventory_sha256'],authorized_counts=counts)
        self.required_support.update(acquired_retention_members(inputs))
        return result

    def _record(self,kind,value):
        name=f'{len(self.events):02d}_{kind}.json'
        fact=custody.write_record(self.paths.evidence,name,dict(schema='corpus_v2_custody.v1',
            kind=kind,synthetic=self.population.synthetic,previous_sha256=self.events[-1].sha256 if self.events else None,
            value=value))
        self.events.append(fact)
        return fact

    def _failure(self,primary):
        """Retain the failure without ever replacing or losing the primary exception."""
        self.failed=True
        from .bounded_exception import bounded_exception_record     # Adopted A
        try:
            self._record('failure',dict(phase=self.phase,exception=bounded_exception_record(primary),
                staging_exists=self.paths.staging.exists(),published_exists=self.paths.published.exists(),
                disposition='retain_every_partial_no_retry'))
        except BaseException as secondary:
            try:
                try:
                    described=repr(secondary)
                except BaseException:
                    described='<'+_class_name(secondary)+'; repr unavailable>'
                message='v2 failure evidence unavailable: '+described
                primary.add_note(message)
                try:
                    sys.stderr.write(message+'\n');sys.stderr.flush()
                except BaseException as reporting:      # never replaces the primary exception
                    primary.add_note('v2 failure reporting unavailable: '+_class_name(reporting))
            except BaseException:
                pass        # a note cannot be attached; the primary still propagates unchanged

    def _enter(self,phase,previous):
        need(not self.failed and self.phase == previous,'invalid/repeated v2 transition')
        self._prerequisites()
        self.phase=phase

    def _launch(self,stage):
        if self.native:
            from .state_execution_authority import prepare_launch
            expected,inputs=self.ports.launch_inputs(stage)
            prepared=prepare_launch(expected=expected,independent_inputs=inputs)
            verify_launch_role(prepared,stage,self.prerequisites.adapter.expected.source)
            from .state_execution_authority import decode
            spec=decode(prepared.expected.specification);premise=self.prerequisites.storage.custody_admission
            need(premise.first_utc<=spec['valid_from']<=spec['valid_until']<=spec['cleanup_until']<=premise.last_utc,
                 'whole owned work/cleanup exceeds admitted custody interval')
            self.required_support.update(acquired_retention_members(inputs))

    def _retained_attempt(self,terminal=None):
        """Exact full-root membership, including late sidecars, after promotion."""
        if not self.whole_attempt:return ()
        from dataclasses import replace
        files=tuple(replace(x,name='payload/'+x.name) for x in self.files)
        files+=tuple(replace(x,name=self.paths.evidence.name+'/'+x.name) for x in self.events)
        files+=(self.archive_fact,self.complete_fact)
        if terminal is not None:files+=(terminal,)
        files=tuple(sorted(files,key=lambda x:x.name))
        custody.same_content(files,custody.exact_tree(self.paths.published,[x.name for x in files]))
        return files

    def _boundary(self,phase):
        if self.native:
            from .state_custody_admission import recheck_live_custody
            recheck_live_custody(self.prerequisites.storage.custody_admission,
                source_sha256=self.bindings['source_inventory_sha256'],
                roots={'payload':str(self._data_root().absolute()),'evidence':str(self.paths.evidence.absolute())})
            from dataclasses import replace
            files=tuple(replace(x,name='payload/'+x.name) for x in self.files)
            files+=tuple(replace(x,name=self.paths.evidence.name+'/'+x.name) for x in self.events)
            for attribute in ('archive_fact','complete_fact'):
                if hasattr(self,attribute):files+=(getattr(self,attribute),)
            inputs=self.ports.storage_inputs(phase,files)
            result=verify_storage(inputs,phase=phase,
                source_sha256=self.bindings['source_inventory_sha256'],files=files)
            self.required_support.update(acquired_retention_members(inputs))
            if phase=='state5_entry':
                need(type(result['free_space_bytes']) is int and result['free_space_bytes']>=self.prerequisites.storage.required_free_bytes,
                     'independent free-space observation insufficient')
            self._record('physical_'+phase,result)

    def close_stage(self,stage):
        need(stage in ('state5','state6_readback','state6_finalizer') and stage not in self.closed,
             'exact unclosed stage required')
        required={'state5':'staged','state6_readback':'verified','state6_finalizer':'published'}
        need(self.phase == required[stage] and not self.failed,'stage not complete')
        if self.native:
            result=self._verify_closed(stage)
            self._record('closure_'+stage,result)
        else:
            self._record('closure_'+stage,dict(component_returned=True,owned_process_evidence=None,
                physical_acceptance=False))
        self.closed.add(stage)

    def stage5(self, *, time_cap_s, mem_cap_bytes=None):
        """The actual writer API. No retry or reduced production population."""
        from operatormark_corpus_v2.generator import generate, check_run_parameters
        check_run_parameters(time_cap_s, mem_cap_bytes)   # refuse before the one-shot attempt is entered
        self._enter('state5','new')
        try:
            self._launch('state5');self._boundary('state5_entry')
            bound=None
            if self.native:
                bound=len(next(x for x in self.prerequisites.storage.domains
                               if x.domain=='generation_configuration_summary').template)
            result=generate(self.config,self.controls,self.directions,self._data_root(),
                solver=self.solver,on_failure='stop',time_cap_s=time_cap_s,mem_cap_bytes=mem_cap_bytes,
                max_configuration_bytes=bound)
            need(result['summary']['completion']=='complete','State 5 generation incomplete; retain partial output')
            self.files=custody.exact_generated_population(self._data_root(),self.population)
            need(self.readback.configuration_sha256 in (None,result['configuration_sha256']),
                 'independent configuration binding mismatch')
            self._record('staged',dict(configuration_sha256=result['configuration_sha256'],
                files=[x.content() for x in self.files]))
            self._boundary('state5_closed');self.phase='staged'
            return result
        except BaseException as exc:
            self._failure(exc);raise

    def stage6(self):
        """Independently reopen all tensors and records; native ladders re-solved."""
        from operatormark_corpus_v2.readback import validate_corpus
        from operatormark_corpus_v2.anomaly import run_anomaly_ladders
        need('state5' in self.closed,'State 5 independent closure required')
        self._enter('state6','staged')
        try:
            # The evidence root must hold exactly this executor's events, files and
            # no directory (as _restore, finalize and terminal require), at entry and immediately
            # before each readback record: an executor forked from an earlier or
            # concurrent handoff cannot add readback records (<= 2 per attempt).
            # Files AND directories: an empty foreign directory is refused too.
            sole_writer=lambda:custody.same_content(tuple(self.events),
                custody.exact_record_root(self.paths.evidence,[x.name for x in self.events]))
            sole_writer()
            self._launch('state6_readback');self._boundary('state6_entry')
            # Exact file and directory membership immediately before each pass:
            # the reader's input-derived names are then the staged population.
            custody.same_content(self.files,custody.exact_generated_population(self._data_root(),self.population))
            kwargs=dict(numerics='full',p1='resolve') if self.native else dict(numerics='structural',p1='skip')
            assessment=validate_corpus(self._data_root(),self.readback,**kwargs)
            sole_writer();self._record('readback_1',assessment)
            ladders=False
            if self.native:
                anomalies=sum(len(v) for v in assessment['recomputed']['anomalies'].values())
                if 0 < anomalies <= 64 and assessment['status'] in ('matched','incomplete'):
                    run_anomaly_ladders(self._data_root(),assessment)
                    ladders=True
                    current={x.name:x for x in custody.exact_generated_population(self._data_root(),self.population,
                                                                                 ladders=True)}
                    custody.same_content(self.files,tuple(current[x.name] for x in self.files))
                    assessment=validate_corpus(self._data_root(),self.readback,**kwargs)
                    sole_writer();self._record('readback_2',assessment)
            validate_assessment(assessment,self.population,native=self.native)
            after=custody.exact_generated_population(self._data_root(),self.population,ladders=ladders)
            old={x.name:x for x in self.files};new={x.name:x for x in after}
            custody.same_content(self.files,tuple(new[x.name] for x in self.files))
            self.files=after;self.assessment=assessment
            self._record('verified',dict(files=[x.content() for x in after],
                physical_qualification=bool(self.native and assessment['production_usable']),
                synthetic_component_complete=not self.native))
            self.phase='verified'
            return assessment
        except BaseException as exc:
            self._failure(exc);raise

    def finalize(self):
        """Whole-attempt rename, audit archive/readback, then completion record.

        A completion record is only a publication product; terminal acceptance
        still requires independent finalizer exit, custody and readback below.
        """
        need('state6_readback' in self.closed,'independent State 6 reader closure required')
        self._enter('finalizing','verified')
        try:
            self._launch('state6_finalizer');self._boundary('pre_finalize');self._boundary('pre_move')
            self._record('promotion_intent',dict(files=[x.content() for x in self.files]))
            if self.whole_attempt:
                from dataclasses import replace
                whole=tuple(replace(x,name='payload/'+x.name) for x in self.files)+tuple(
                    replace(x,name=self.paths.evidence.name+'/'+x.name) for x in self.events)
                whole=tuple(sorted(whole,key=lambda x:x.name))
                custody.promote_once(self.paths.staging,self.paths.published,whole)
                self.paths=replace(self.paths,**{k:self.paths.published/getattr(self.paths,k).name
                    for k in ('evidence','archive','complete','terminal')})
                custody.same_content(self.files,custody.snapshot(self._data_root(),[x.name for x in self.files]))
            else:
                self.files=custody.promote_once(self.paths.staging,self.paths.published,self.files)
            self._boundary('post_move')
            self._record('promoted',dict(files=[x.content() for x in self.files]))
            self.archive_members=custody.exact_record_root(self.paths.evidence,[x.name for x in self.events])
            self.archive_fact=custody.archive_once(self.paths.evidence,self.paths.archive,self.archive_members)
            self._boundary('post_audit')
            self.complete_fact=custody.write_record(self.paths.complete.parent,self.paths.complete.name,
                dict(schema='corpus_v2_publication.v1',synthetic=self.population.synthetic,
                    population=self.population.record(),bindings=self.bindings,
                    files=[x.content() for x in self.files],archive=self.archive_fact.content(),
                    archive_members=[x.content() for x in self.archive_members],
                    terminal_accepted=False))
            self._boundary('post_complete');self.phase='published'
            return self.complete_fact
        except BaseException as exc:
            self._failure(exc);raise

    def terminal(self):
        """Supervisor boundary AFTER finalizer exit; independent bytes and seal."""
        need('state6_finalizer' in self.closed,'finalizer exit/streams/guard closure required')
        self._enter('terminal','published')
        try:
            self._boundary('cleanup');self._boundary('terminal')
            custody.same_content(self.files,custody.snapshot(self._data_root(),[x.name for x in self.files]))
            custody.verify_archive(self.paths.archive,self.archive_members)
            custody.same_content((self.archive_fact,),
                (custody.acquire_file(self.paths.archive.parent,self.paths.archive.name),))
            custody.same_content((self.complete_fact,),
                (custody.acquire_file(self.paths.complete.parent,self.paths.complete.name),))
            evidence=custody.exact_record_root(self.paths.evidence,[x.name for x in self.events])
            local=self._retained_attempt()
            state4=();support=();controls=[('payload',x.name) for x in self.files if x.name.endswith('/rho.float64.npy')]
            if self.native:
                from .state_v2_prerequisites import read_accepted_state4, AcceptedState4Inputs
                need(type(self.prerequisites.state4_arguments) is AcceptedState4Inputs,
                     'complete independently accepted State 4 retention projection required')
                data,projection=read_accepted_state4(self.prerequisites.state4_arguments)
                need(callable(self.ports.retention_members),
                     'complete host support producer membership unavailable')
                host_support=tuple(self.ports.retention_members())
                self.required_support.update(host_support)
                retained=self.ports.retention_inputs()
                state4,support=verify_retained_inputs(retained,attempt_root=self.paths.published,
                    required=self.required_support,local=local)
                verify_state4_retention(projection,data,state4)
                verify_predecessor_audit_retention(projection,state4+support)
                controls.extend(('state4',x.name) for x in state4)
            accounting_args=dict(payload=self.files,evidence=evidence,archive=self.archive_fact,
                controls=tuple(controls),sidecars=(self.complete_fact,),state4=state4,support=support)
            result=dict(schema='corpus_v2_terminal.v1',synthetic=self.population.synthetic,
                component_complete=True,production_accepted=False,execution_authorized=False,
                native_correspondence_complete=self.native,bindings=self.bindings,
                completion=self.complete_fact.content(),archive=self.archive_fact.content(),
                retained_evidence=[x.content() for x in evidence],
                retained_state4=[x.content() for x in state4],
                retained_support=[x.content() for x in support],
                retained_input_placements=list(retained.placements) if self.native else [],
                note='independent correspondence only; no acceptance authority is issued')
            # Exact self-size fixed point, without a circular self-content hash.
            size=0;seen=set()
            while True:
                need(size not in seen,'terminal sizing did not converge');seen.add(size)
                result['accounting']=exact_retention_accounting(**accounting_args,terminal_bytes=size)
                encoded=C.canonical_json_bytes(C.with_self_hash(result,'record_self_sha256'))
                if len(encoded)==size:break
                size=len(encoded)
            fact=custody.write_record(self.paths.terminal.parent,self.paths.terminal.name,result)
            need(fact.size==size,'terminal writer-size disagreement')
            reread=C.strict_canonical_json_loads((self.paths.terminal).read_bytes());C.validate_self_hash(reread,'record_self_sha256')
            custody.same_content(self.files,custody.snapshot(self._data_root(),[x.name for x in self.files]))
            custody.same_content(evidence,custody.exact_record_root(self.paths.evidence,[x.name for x in evidence]))
            local=self._retained_attempt(fact)
            if self.native:
                need(tuple(self.ports.retention_members())==host_support,
                     'host support changed during terminal readback')
                verify_retained_inputs(retained,attempt_root=self.paths.published,
                    required=self.required_support,local=local)
            need(reread['accounting']==exact_retention_accounting(**accounting_args,terminal_bytes=fact.size),
                 'independent terminal accounting mismatch')
            self.phase='closed'
            return result
        except BaseException as exc:
            self._failure(exc);raise
