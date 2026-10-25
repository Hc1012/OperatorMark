"""One final-owner connection, not a launcher or an acceptance issuer.

Externally accepted expectations are premises supplied by an independent caller.
The existing AcceptedRecord/require_accepted interface checks exact bytes and
scope; it does not authenticate a writer's self-declared success. Concrete lazy
providers still require externally accepted custody and runtime evidence.
Test doubles exercise wiring only, never establish final acceptance.
"""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re

from operatormark.corpus.state_v2_prerequisites import AcceptedRecord
from .scientific_admission import require as require_accepted, runtime_ready, verify_material_lineage, same_corpus, ScientificRecord
from msdev.records import write
from . import marking as M, routing as R

REQUIRED_SOURCES = (
    'tools/msfinal/portable_corpus.py',
    'tools/msfinal/scientific_admission.py', 'operatormark_corpus_v2/scientific.py',
    'tools/scientific_corpus.py',
    'tools/msfinal/backend.py', 'tools/msfinal/marking.py', 'tools/msfinal/routing.py',
    'tools/msdev/pn_development.py', 'tools/msdev/kernels.py', 'tools/msdev/rules.py',
    'tools/msfinal/materials.py', 'tools/msfinal/evaluation.py', 'tools/msfinal/d2.py',
    'tools/msfinal/cpu_scoring.py', 'tools/msfinal/cpu_score_entry.py',
    'tools/msfinal/cpu_batches.py', 'tools/msfinal/cpu_batch_entry.py',
    'tools/msfinal/attacks.py', 'tools/msfinal/orchestration.py',
    'tools/msfinal/update_journal.py',
    'tools/msfinal/graph.py', 'tools/msfinal/readback.py',
    'tools/msfinal/worker.py', 'tools/msfinal/session.py', 'tools/msfinal/export.py',
    'tools/msfinal/drive_transport.py', 'tools/msfinal/phase_storage.py', 'tools/msfinal/storage_plan.py',
    'tools/msfinal/archive_custody.py',
    'tools/msfinal/operations.py',
    'tools/msfinal/operating_entry.py',
    'tools/msfinal/bounded_stage.py', 'tools/msfinal/stage_publication.py',
    'tools/msfinal/variants.py', 'tools/msfinal/models.py', 'tools/msfinal/expanded.py',
    'tools/msfinal/contract_arms.py', 'tools/msfinal/prospective_contracts.py',
    'tools/msfinal/p256_adjoint.py', 'tools/msfinal/p256_tasks.py',
    'tools/msfinal/final_contracts.py', 'tools/msfinal/resources.py',
    'tools/msdev/campaign.py', 'tools/msdev/colab_environment.py',
    'tools/msfinal/analysis.py',
    'tools/msfinal/nulls.py',
    'tools/msfinal/owners.py',
    'tools/msdev/scoring.py', 'tools/msdev/records.py', 'tools/q5_model.py',
    'tools/q6_common.py', 'tools/q6_train.py',
    'operatormark/corpus/state_v2_prerequisites.py',
    'operatormark/corpus/state_v2_storage.py', 'operatormark/corpus/state_v2_writer_bounds.py',
    'operatormark/corpus/canonical.py',
    'operatormark_numerics_v2/task.py',
    'experiments/tpfa32_copy_study/analyze.py', 'experiments/tpfa32_copy_study/core.py',
    'experiments/tpfa32_copy_study/outcomes.py',
)


class MissingInterface(RuntimeError):
    pass


@dataclass(frozen=True)
class Prerequisites:
    study_id: str
    owner_id: str
    clean_model_sha256: str
    configuration: AcceptedRecord
    inputs: AcceptedRecord
    runtime: AcceptedRecord
    calibration: AcceptedRecord
    clean_evaluation: AcceptedRecord
    material_paths: dict
    source_root: Path
    assets: AcceptedRecord | None = None
    assets_root: Path | None = None
    opening: AcceptedRecord | None = None


@dataclass(frozen=True)
class Admission:
    prerequisites: Prerequisites
    configuration: dict
    inputs: dict
    calibration: dict
    clean_evaluation: dict
    bindings: dict


def calibration_correspondence(config, configuration_sha256, cal, calibration_sha256,
                               owner, clean_model_sha256):
    """Bind a fixed receipt, or a prospectively fixed post-owner procedure.

    The latter avoids requiring future owner-dependent calibration bytes before
    owner training. It never infers a floor/delta, accepts development calibration,
    or chooses a procedure from outcomes. The caller still requires independent
    final-calibration acceptance and validates every scientific field.
    """
    if config.get('final_calibration') == calibration_sha256:
        return
    deferred=config.get('final_calibration_binding',{})
    if (config.get('final_calibration') is not None
            or set(deferred)!={'mode','procedure_sha256'}
            or deferred.get('mode')!='post_owner_readback'):
        raise ValueError('final calibration receipt/procedure correspondence missing')
    M.identity(deferred.get('procedure_sha256'))
    models=cal.get('clean_owner_models',{})
    if (cal.get('procedure_sha256')!=deferred['procedure_sha256']
            or cal.get('configuration_sha256')!=configuration_sha256
            or set(models)!=set(config['owners'])
            or models.get(owner)!=clean_model_sha256):
        raise ValueError('post-owner calibration procedure/configuration/clean-owner drift')
    for h in models.values():M.identity(h)


def admit(p):
    """Pure file/metadata gate; no model import, seed draw, solver or tensor load."""
    if type(p) is not Prerequisites or not p.study_id or not p.owner_id:
        raise ValueError('explicit study/owner prerequisites required')
    M.identity(p.clean_model_sha256)
    config = require_accepted(p.configuration, kind='main_study_final_configuration',
                              bindings=dict(study_id=p.study_id))
    owners = config.get('owners')
    if (config.get('domain') != 'final' or config.get('approved') is not True
            or not isinstance(owners, list) or not owners or any(not isinstance(o, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',o) for o in owners)
            or len(owners) != len(set(owners)) or p.owner_id not in owners
            or type(config.get('population')) is not int or config['population'] != len(owners)
            or not M.finite(config.get('kappa')) or config['kappa'] not in (1., .5, .25)
            or config.get('B1_disposition') != 'unavailable'
            or not isinstance(config.get('attack_variants'), list) or not config['attack_variants']
            or any(not isinstance(v, str) or not v for v in config['attack_variants'])
            or len(set(config['attack_variants'])) != len(config['attack_variants'])):
        raise ValueError('unapproved/incomplete final configuration; no inferred scientific choices')
    cal = require_accepted(p.calibration, kind='main_study_final_calibration',
                           bindings=dict(study_id=p.study_id))
    calibration_correspondence(config,p.configuration.sha256,cal,p.calibration.sha256,
                               p.owner_id,p.clean_model_sha256)
    common = dict(study_id=p.study_id, owner_id=p.owner_id,
                  configuration_sha256=p.configuration.sha256, clean_model_sha256=p.clean_model_sha256)
    inputs = require_accepted(p.inputs, kind='main_study_final_inputs', bindings=common)
    M.validate_calibration(inputs, cal)
    if (inputs['study_id'] != p.study_id or inputs['owner_id'] != p.owner_id
            or inputs['materials']['clean_model']['sha256'] != p.clean_model_sha256):
        raise ValueError('final input / original clean owner substitution')
    runtime_bindings = dict(common, inputs_sha256=p.inputs.sha256,
                            runtime_record_sha256=inputs['materials']['runtime']['sha256'])
    runtime = require_accepted(p.runtime, kind='main_study_final_runtime', bindings=runtime_bindings)
    same_corpus(p.inputs,p.runtime)
    if type(p.runtime) is ScientificRecord:runtime_ready(p.runtime)
    if runtime.get('domain') != 'final':
        raise ValueError('development-only runtime acceptance is inapplicable')
    clean = require_accepted(p.clean_evaluation, kind='main_study_final_clean_evaluation',
                             bindings=dict(common, inputs_sha256=p.inputs.sha256))
    if (clean.get('domain') != 'final' or clean.get('complete') is not True
            or clean.get('study_id') != p.study_id or clean.get('owner_id') != p.owner_id
            or clean.get('model_sha256') != p.clean_model_sha256 or clean.get('inputs_sha256') != p.inputs.sha256
            or any(not M.finite(clean.get('metrics', {}).get(k)) or clean['metrics'][k] < 0 for k in ('F','J','K','T'))):
        raise ValueError('original clean-owner evaluation unavailable/incompatible')
    medians=cal.get('clean_screen_medians',{})
    M.identity(cal.get('clean_screen_reference_sha256'))
    if (set(medians)!={'F','J'} or any(not M.finite(v) or v<=0 for v in medians.values())
            or any(clean['metrics'][k]>2*medians[k] for k in ('F','J'))):
        raise ValueError('clean-cohort screen unavailable/failed; marking must not start')
    pins = config.get('source_pins', {})
    if not set(REQUIRED_SOURCES) <= set(pins):
        raise ValueError('final component source coverage missing')
    base = Path(p.source_root).resolve()
    for name, expected in pins.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
            raise ValueError('invalid source binding path')
        path = base.joinpath(*rel.parts).resolve()
        if not path.is_relative_to(base) or M.digest(path) != M.identity(expected):
            raise ValueError('final candidate source drift: ' + name)
        if name in REQUIRED_SOURCES:
            actual = Path(__file__).resolve().parents[2].joinpath(*rel.parts)
            if M.digest(actual) != expected:
                raise ValueError('running component differs from candidate: ' + name)
    # These three hashes identify the unchanged selected scientific functions.
    for name, expected in M.SOURCE_PINS.items():
        if pins['tools/msdev/' + name] != expected:
            raise ValueError('selected P-N implementation differs')
    if set(p.material_paths) != set(M.MATERIALS):
        raise ValueError('missing/extra final material mapping')
    for role, path in p.material_paths.items():
        if M.digest(path) != inputs['materials'][role]['sha256']:
            raise ValueError('accepted material drift: ' + role)
    return Admission(p, config, inputs, cal, clean,
                     dict(configuration=p.configuration.sha256, inputs=p.inputs.sha256,
                          runtime=p.runtime.sha256, calibration=p.calibration.sha256,
                          clean_evaluation=p.clean_evaluation.sha256, clean_model=p.clean_model_sha256))


class NumericalConnection:
    """Lazy final providers; injected doubles are for synthetic wiring only."""
    def __init__(self, materials=None, evaluation=None):
        self.materials, self.evaluation = materials, evaluation

    def preflight(self, admission):
        from . import cpu_scoring as CPU
        CPU.policy(admission)
        if self.materials is None:
            from . import materials
            self.materials = materials
        if self.evaluation is None:
            from . import evaluation
            self.evaluation = evaluation
        self.materials.preflight(admission)
        self.evaluation.preflight(admission)

    def prepare(self, admission, check):
        loaded = self.materials.load_accepted_marking_inputs(admission, check)
        engine = M.create_engine(loaded['data'], loaded['weights'], loaded['initialization'],
                                 loaded['shuffle'], loaded['device'], admission.inputs, admission.calibration)
        return engine, loaded

    def mark(self, engine, loaded, admission, check, record):
        return M.mark(engine, loaded['directions'], loaded['pattern'], loaded['reference_jvp'],
                      admission.inputs, admission.calibration, check, record)

    def save(self, engine, out, check):
        return self.materials.save_marked(engine, out, check)

    def evaluate(self, engine, admission, out, check):
        return self.evaluation.evaluate_owner(engine.model, admission, out, check)

    def save_failure(self,engine,out,check):
        save=getattr(self.materials,'save_failed',None)
        if save is None:return dict(available=False,reason='provider has no failure-state saver')
        return save(engine,out,check)


def _retained_file(out, path):
    path = Path(path).resolve()
    if not path.is_relative_to(out.resolve()) or not path.is_file():
        raise ValueError('marked model/state must be retained in this fresh output')
    return path


def run_owner(prerequisites, out, *, check, numerical=None):
    """Admission -> P-N marking -> evaluation -> qualification -> routing.

    Called only by a future authorized, owned worker. No clock/reservation,
    subprocess, retry, automatic attack, material opening or launch is created.
    check is the existing caller's deadline/ownership callback. A qualification
    is an OUTPUT, never a prerequisite for creating the mark.
    """
    p = prerequisites
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    result = dict(owner_id=p.owner_id, include_primary_quality=True, state='BLOCKED',
                  qualification_available=False, B3_qualified=None, descendants=[],
                  marked_checkpoint_complete=False, quality_descendants=[],
                  B1='unavailable; final disposition pending', committed_mark_updates=0,
                  execution_authorized=False, independent_qualification_acceptance=False)
    numerical = NumericalConnection() if numerical is None else numerical
    dispatched = False; engine = None
    try:
        check()
        admission = admit(p)
        numerical.preflight(admission)
        # Re-read after provider preflight and immediately before any tensor load.
        admission = admit(p)
        result['bindings'] = admission.bindings
        write(out/'ADMISSION.json', dict(bindings=admission.bindings, correspondence=True,
                                         independent_acceptance_issued=False))
        check(); dispatched = True
        engine, loaded = numerical.prepare(admission, check)
        def record(row):
            n = result['committed_mark_updates'] + 1
            if row.get('update') != n or row.get('pre_state') != n-1 or row.get('controls') != 128 or n > 320:
                raise ValueError('marking update accounting mismatch')
            write(out/f'update_{n:03d}.json', row)
            result['committed_mark_updates'] = n
        mark = numerical.mark(engine, loaded, admission, check, record)
        if mark.get('recipe') != M.RECIPE or mark.get('updates') != 320 or result['committed_mark_updates'] != 320:
            raise ValueError('incomplete selected marking trajectory')
        write(out/'MARKING.json', mark)
        model_path, state_path = numerical.save(engine, out, check)
        model_path, state_path = (_retained_file(out, x) for x in (model_path, state_path))
        model_hash, state_hash = M.digest(model_path), M.digest(state_path)
        result.update(model_sha256=model_hash, state_sha256=state_hash,marked_checkpoint_complete=True,
            quality_descendants=[R.quality_route(p.study_id,p.owner_id,model_hash,p.clean_model_sha256)])
        check(); admission = admit(p)
        measured = numerical.evaluate(engine, admission, out, check)
        # Preserve failed/incomplete raw measurements before classifying them.
        write(out/'MEASUREMENTS.json', measured)
        if M.digest(model_path) != model_hash or M.digest(state_path) != state_hash:
            raise ValueError('marked model/state changed during evaluation')
        check(); admission = admit(p)
        if measured.get('task_status')=='designs_ready_pending_scoring':
            result.update(state='PENDING_CPU_SCORING',cpu_request=measured['cpu_request'],reason='independent CPU scoring required before qualification')
        elif measured.get('complete') is not True:
            result.update(state='UNAVAILABLE', reason='final evaluation/detector evidence incomplete')
        else:
            common = dict(domain='final', study_id=p.study_id, owner_id=p.owner_id,
                          model_sha256=model_hash, complete=True, inputs_sha256=p.inputs.sha256)
            write(out/'EVALUATION.json', dict(common, metrics=measured.get('metrics')))
            write(out/'DETECTOR.json', dict(common, scores=measured.get('scores')))
            try:
                q = M.qualify_records(p.inputs.path, p.inputs.sha256, model_hash,
                    out/'EVALUATION.json', M.digest(out/'EVALUATION.json'),
                    p.clean_evaluation.path, p.clean_evaluation.sha256,
                    out/'DETECTOR.json', M.digest(out/'DETECTOR.json'), p.calibration.path, p.calibration.sha256)
            except (ValueError, TypeError, KeyError) as exc:
                result.update(state='UNAVAILABLE', reason=str(exc))
            else:
                write(out/'QUALIFICATION.json', q)
                from .orchestration import MARKED_VARIANTS
                routed = R.owner_route(p.study_id, p.owner_id, model_hash, p.clean_model_sha256,
                    out/'QUALIFICATION.json', M.digest(out/'QUALIFICATION.json'), q['bindings'],
                    [v for v in admission.configuration['attack_variants'] if v in MARKED_VARIANTS])
                result.update(state='COMPLETE', qualification_available=True, B3_qualified=q['qualified'],
                              qualification_sha256=M.digest(out/'QUALIFICATION.json'), descendants=routed['descendants'])
                write(out/'ROUTING.json', routed)
    except Exception as exc:
        result.update(state='FAILED' if dispatched else 'BLOCKED', reason=type(exc).__name__+': '+str(exc))
        if engine is not None and hasattr(numerical,'save_failure'):
            try:result['failure_preservation']=numerical.save_failure(engine,out,check)
            except Exception as save_error:result['failure_preservation']=dict(available=False,error=repr(save_error))
        # All committed/partial products remain; neither reset nor automatic retry.
    write(out/'RESULT.json', result)
    return result


def owner_accounting(results, expected_owner_ids):
    ids = [r['owner_id'] for r in results]
    if len(ids) != len(set(ids)) or len(expected_owner_ids) != len(set(expected_owner_ids)) or set(ids) != set(expected_owner_ids):
        raise ValueError('missing/duplicate owner; primary analysis incomplete')
    return dict(planned_owners=len(ids), primary_quality_owners=len(ids),
                marking_measurements_complete=all(r['qualification_available'] for r in results),
                primary_complete=False,primary_status='final paired measurements not collected by marking stage',
                B3_qualified=sum(r['B3_qualified'] is True for r in results),
                B3_failed_qualification=sum(r['B3_qualified'] is False for r in results),
                unavailable_owners=sum(r['B3_qualified'] is None for r in results),
                completed_marked_checkpoints=sum(r.get('marked_checkpoint_complete') is True for r in results),
                missing_marked_checkpoints=[r['owner_id'] for r in results if r.get('marked_checkpoint_complete') is not True],
                ordinary_P2_measurements_required=sum(r.get('marked_checkpoint_complete') is True for r in results),
                descendants=sum(len(r['descendants']) for r in results), execution_authorized=False)


def complete_pending_owner(p,source,out,measured,check):
    """Qualify an already saved mark after CPU return, with no model dispatch."""
    source=Path(source);out=Path(out);admission=admit(p);check()
    from .cpu_scoring import verify_merged
    if __import__('msdev.records',fromlist=['read']).read(out/'evaluation/RESULT.json')!=measured:
        raise ValueError('qualification CPU evaluation substitution')
    verify_merged(out/'evaluation',measured,check)
    result=__import__('msdev.records',fromlist=['read']).read(source/'RESULT.json')
    if result.get('state')!='PENDING_CPU_SCORING' or measured.get('complete') is not True:
        raise ValueError('pending mark and complete CPU-scored evaluation required')
    model_hash=result['model_sha256']
    request=__import__('msdev.records',fromlist=['read']).read(out/'evaluation/D6_REQUEST.json')
    expected=dict(owner=p.owner_id,model_sha256=model_hash,configuration_sha256=p.configuration.sha256,
        producer_runtime_sha256=p.runtime.sha256)
    if result.get('bindings')!=admission.bindings or any(request['binding'].get(k)!=v for k,v in expected.items()):
        raise ValueError('qualification producer/owner/candidate/runtime mismatch')
    if M.digest(source/'MODEL.pt')!=model_hash or M.digest(source/'STATE.pt')!=result['state_sha256']:
        raise ValueError('held marked checkpoint/state changed')
    result=dict(result);result.pop('cpu_request',None);result.pop('reason',None)
    common=dict(domain='final',study_id=p.study_id,owner_id=p.owner_id,model_sha256=model_hash,
        complete=True,inputs_sha256=p.inputs.sha256)
    write(out/'MEASUREMENTS.json',measured)
    write(out/'EVALUATION.json',dict(common,metrics=measured['metrics']))
    write(out/'DETECTOR.json',dict(common,scores=measured['scores']))
    q=M.qualify_records(p.inputs.path,p.inputs.sha256,model_hash,out/'EVALUATION.json',M.digest(out/'EVALUATION.json'),
        p.clean_evaluation.path,p.clean_evaluation.sha256,out/'DETECTOR.json',M.digest(out/'DETECTOR.json'),p.calibration.path,p.calibration.sha256)
    write(out/'QUALIFICATION.json',q)
    from .orchestration import MARKED_VARIANTS
    routed=R.owner_route(p.study_id,p.owner_id,model_hash,p.clean_model_sha256,out/'QUALIFICATION.json',
        M.digest(out/'QUALIFICATION.json'),q['bindings'],[v for v in admission.configuration['attack_variants'] if v in MARKED_VARIANTS])
    result.update(state='COMPLETE',qualification_available=True,B3_qualified=q['qualified'],
        qualification_sha256=M.digest(out/'QUALIFICATION.json'),qualification_path=str(out/'QUALIFICATION.json'),descendants=routed['descendants'])
    write(out/'ROUTING.json',routed);write(out/'RESULT.json',result);return result
