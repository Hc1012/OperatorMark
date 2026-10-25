"""Export scoped noise designs through the unchanged D6 saved-design schema.

This file constructs no model, tensor, optimizer or native solver. The original
accepted material admission is used only to read its exact target bytes and
metadata. Current producer authority is supplied separately by NOISE_POLICY and
its fresh scoped runtime gate; historical AcceptedRecords are never rewritten.
The CPU run and its independent operation window remain future work.
"""
from pathlib import Path
import hashlib
import json
import os

SCOPE = 'selected_noise20_only'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024):
            check()
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(value):
    need(isinstance(value, str) and len(value) == 64
         and all(c in '0123456789abcdef' for c in value), 'Invalid identity digest')
    return value


def _bytes(path, raw):
    with Path(path).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def preflight_export(historical_admission, check, *, job, policy, policy_sha,
                     member, binding, model_sha256, wrapper_steps=None):
    """Read-only D6 contract checks before any model construction or forward.

    Reused verbatim by export after numerical work. Original accepted target
    bytes are read and hashed; no array/model/native kernel is loaded or run.
    """
    import noise_runtime as R
    from msfinal import materials as A, cpu_scoring as CPU

    check()
    job = Path(job).resolve()
    R.check_policy(job, policy, policy_sha, check)
    R.expectations(job, policy, check)
    relative = R.member_path(policy, member)
    row = next(r for r in policy['noise_jobs'] if r['id'] == member)
    need(digest(model_sha256) == row['parent_sha256'] and row['parent'] == 'M3'
         and row['owner'] == historical_admission.prerequisites.owner_id,
         'Noise model/owner differs from frozen request')
    need(wrapper_steps is None, 'Noise D6 export cannot add wrapper steps')
    gate_path = job/R.GATE
    gate = read(gate_path)
    activation = read(job/'ACTIVATION.json')
    gate_sha = sha(gate_path, check)
    applicability_sha = policy['runtime_evidence']['applicability']['sha256']
    need(gate.get('passed') is True and gate.get('scope') == SCOPE
         and gate.get('policy_sha256') == policy_sha
         and gate.get('activation_sha256') == sha(job/'ACTIVATION.json', check)
         and activation.get('scope') == SCOPE and activation.get('policy_sha256') == policy_sha
         and gate.get('candidate_runtime_payload_sha256') == policy['candidate_runtime_payload_sha256']
         and gate.get('applicability_sha256') == applicability_sha
         and gate.get('scientific_acceptance') is False
         and gate.get('general_runtime_acceptance') is False,
         'D6 export lacks the actual current noise runtime gate')
    start_path = job/'observations/noise_jobs'/relative/'RUNTIME_START.json'
    current_start = read(start_path)
    need(current_start.get('passed') is True and current_start.get('member') == member
         and current_start.get('scope') == SCOPE and current_start.get('phase') == 'start'
         and current_start.get('policy_sha256') == policy_sha
         and current_start.get('activation_sha256') == gate['activation_sha256']
         and current_start.get('startup_gate_sha256') == gate_sha,
         'Current member runtime observation is missing or substituted')

    need(historical_admission.configuration['source_pins'] == policy['source_pins']
         and historical_admission.prerequisites.assets.sha256 == row['catalog_sha256'],
         'Historical source or accepted asset catalog differs')
    cpu_policy = CPU.validate_policy(historical_admission.configuration['cpu_scoring'], deferred=True)
    need(cpu_policy['source_pins'] == policy['source_pins']
         and cpu_policy['runtime_sha256'] is None and cpu_policy['compatibility_sha256'] is None,
         'Existing deferred CPU policy required; no old runtime promotion')
    need(binding.get('scope') == SCOPE and binding.get('member') == member
         and binding.get('policy_sha256') == policy_sha
         and binding.get('campaign_sha256') == policy_sha
         and binding.get('configuration_sha256') == policy['configuration_sha256']
         and binding.get('producer_runtime_gate_sha256') == gate_sha
         and binding.get('applicability_sha256') == applicability_sha
         and isinstance(binding.get('phase_name'), str) and binding['phase_name']
         and 'original_deadline' not in binding and 'original_start' not in binding,
         'D6 binding must use current scope and a separate future CPU campaign window')
    for name in ('owner','model_sha256','checkpoint','phase','wrapper_steps',
                 'producer_runtime_sha256','producer_runtime_payload_sha256',
                 'assets_sha256','source_pins','limitations','historical_producer'):
        need(name not in binding, 'Reserved D6 field supplied twice: '+name)
    # Real material access retains the already accepted opening/catalog route.
    # A fresh scoped context is deliberately not passed as an old Admission.
    catalog = A.catalog(historical_admission)
    target_bytes = {}
    targets = {}
    for key in ('targets','targets_p256'):
        name = 'final_'+key
        raw = A.read_bytes(historical_admission, name, 'final')
        record = catalog['entries'][name]
        need(len(raw) == record['bytes']
             and hashlib.sha256(raw).hexdigest() == record['sha256'],
             'Original final reference target bytes differ')
        target_bytes[key] = raw
        targets[key] = dict(source_sha256=record['sha256'], file='cpu_'+key+'.npy')

    check()
    return dict(row=row, relative=relative, gate_sha=gate_sha,
                applicability_sha=applicability_sha, start_path=start_path,
                cpu_policy=cpu_policy, target_bytes=target_bytes, targets=targets)


def export_saved_designs(out, result, historical_admission, check, *, job,
                         policy, policy_sha, member, binding, model_sha256,
                         wrapper_steps=None):
    """Return pending evaluation plus cpu_request; caller writes RESULT.json.

    The numerical evaluator supplies four already saved16-target designs and
    diagnostics. D6 target/order/metric/operation semantics match CPU.freeze.
    The binding uses a campaign rather than the expiring GPU clock so a later
    independently admitted CPU operation can have its own bounded window.
    """
    import noise_runtime as R
    from msfinal import materials as A, cpu_scoring as CPU
    from msfinal.evaluation import validate_specification
    from msdev.records import write

    check()
    job = Path(job).resolve()
    out = Path(out).resolve()
    prepared = preflight_export(historical_admission, check, job=job, policy=policy,
        policy_sha=policy_sha, member=member, binding=binding,
        model_sha256=model_sha256, wrapper_steps=wrapper_steps)
    row, relative = prepared['row'], prepared['relative']
    gate_sha, applicability_sha = prepared['gate_sha'], prepared['applicability_sha']
    start_path, cpu_policy = prepared['start_path'], prepared['cpu_policy']
    target_bytes, targets = prepared['target_bytes'], prepared['targets']
    need(out == job/'output/jobs'/relative/'evaluation' and out.is_dir()
         and not out.is_symlink(), 'D6 export is outside its frozen noise member')

    # This is prospective final-population work. Never repair a development
    # result by changing its label during export or later CPU merge.
    need(result.get('domain') == 'final' and result.get('phase') == 'final'
         and result.get('scope') == SCOPE and result.get('complete') is False
         and result.get('task_status') == CPU.STATE
         and result.get('diagnostics_complete') is True,
         'Scoped final pending result required; no domain relabeling')
    spec = result['specification']
    validate_specification(spec, 'final')
    need(result['counts'] == dict(inverse_input_iterations=480, targets=0,
         native_t128_calls=0, native_p256_calls=0, validation_controls=512,
         detector_controls=128, optimizer_model_updates=0)
         and result['rows']['T'] == [] and result['rows']['P256'] == [],
         'D6 export must contain complete inverse work and no native scoring')
    start = read(out/'START.json')
    need(start.get('domain') == 'final' and start.get('phase') == 'final'
         and start.get('specification') == spec
         and start.get('assets') == row['catalog_sha256']
         and start.get('evaluation_binding', {}).get('model_sha256') == model_sha256,
         'Original scoped evaluator start record differs')
    original_names = {'START.json','DETECTOR_SCORES.json'}
    original_names.update(f'validation_{i:04d}.json' for i in range(0,512,16))
    original_names.update(f'{name}_{i:02d}.{suffix}' for i in range(0,64,16)
                          for name,suffix in (('designs','npy'),('inverse','json')))
    need({p.name for p in out.iterdir()} == original_names
         and all(p.is_file() and not p.is_symlink() for p in out.iterdir()),
         'Evaluation member population differs before D6 export')

    historical = dict(
        runtime_record_sha256=historical_admission.prerequisites.runtime.sha256,
        runtime_payload_sha256=historical_admission.inputs['materials']['runtime']['sha256'],
        assets_sha256=historical_admission.prerequisites.assets.sha256,
        bindings=historical_admission.bindings,
        request_sha256=row['request_sha256'])
    provenance = dict(schema='OperatorMark.NOISE_D6_export.v1', scope=SCOPE,
        member=member, policy_sha256=policy_sha, campaign_sha256=policy_sha,
        original_request_sha256=row['request_sha256'], historical_producer=historical,
        current_runtime_record_kind='OperatorMark.NOISE_scoped_runtime_gate.v1',
        producer_runtime_sha256=gate_sha,
        producer_runtime_payload_sha256=policy['candidate_runtime_payload_sha256'],
        member_runtime_start_sha256=sha(start_path, check),
        applicability_sha256=applicability_sha, source_pins_unchanged=True,
        native_calls=0, models_called_by_exporter=0, scientific_acceptance=False,
        formal_custody_assurance=False, independent_CPU_window_pending=True,
        original_admissions_rewritten=False)
    result = dict(result, **CPU.limitations(historical_admission.inputs))
    result['noise_D6_provenance'] = provenance
    for key,raw in target_bytes.items():
        check()
        _bytes(out/targets[key]['file'], raw)
    write(out/'NOISE_D6_PROVENANCE.json', provenance)
    files = {p.name:dict(bytes=p.stat().st_size, sha256=sha(p,check))
             for p in out.iterdir() if p.is_file()}
    b = dict(binding, owner=row['owner'], model_sha256=model_sha256,
        checkpoint=binding['phase_name'], phase='final', wrapper_steps=None,
        producer_runtime_sha256=gate_sha,
        producer_runtime_payload_sha256=policy['candidate_runtime_payload_sha256'],
        assets_sha256=row['catalog_sha256'], source_pins=policy['source_pins'],
        limitations=CPU.limitations(historical_admission.inputs),
        historical_producer=historical)
    request = dict(schema='D6.saved-designs.v1', binding=b, cpu_policy=cpu_policy,
        files=files, targets=targets, target_order=list(range(64)),
        design_batches=[dict(file=f'designs_{i:02d}.npy', trace=f'inverse_{i:02d}.json',
                             indices=list(range(i,i+16))) for i in range(0,64,16)],
        pending_result=result,
        metric='msdev.scoring.relative_rows; physical field units; unchanged hard gates',
        operations=dict(T128_scoring=64,P256_scoring=64,inverse_updates=0,
                        model_updates=0,noise_draws=0))
    write(out/'D6_REQUEST.json', request)
    expected = sha(out/'D6_REQUEST.json', check)
    CPU.verify_request(out, expected, check)
    check()
    return dict(result, cpu_request=dict(directory=out.relative_to(job/'output').as_posix(),
                                        sha256=expected))
