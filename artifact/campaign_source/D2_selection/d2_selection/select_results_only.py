"""Read existing result bytes and apply the unchanged frozen D2 decision.

No model, native solver, evaluator, worker, launch or tests are called.
Outputs are new scoped result records; no original result is edited.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib, json, math
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
JOB = ROOT / 'r3/ordinary'
CODE = ROOT / 'r3/payload/code'
CONTRACT_SHA = '0d970217502bb2778ac1139613853b880242b5de7f88e15122f1ec138cb399f9'
evidence = {}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p, expected=None):
    p = Path(p)
    value = sha(p)
    if expected is not None and value != expected:
        raise ValueError('Frozen result differs: ' + str(p))
    evidence[str(p.relative_to(ROOT))] = {'sha256': value, 'bytes': p.stat().st_size}
    return json.loads(p.read_text())

def need(ok, msg):
    if not ok:
        raise ValueError(msg)

def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        f.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')

contract = read(OUT/'D2_RESULTS_ONLY_ADAPTER_CONTRACT.json', CONTRACT_SHA)
for name, expected in contract['science_files'].items():
    need(sha(CODE/name) == expected, 'Frozen science source differs: ' + name)
sys.path[:0] = [str(CODE/'tools'), str(CODE)]
from msfinal import d2 as D2, routing as R
from msfinal.graph import identity

policy = read(JOB/'D2_COMBINED_POLICY.json', contract['inputs']['raw960']['policy_sha256'])
graph = read(JOB/'admission/FINAL_GRAPH.json', contract['graph_sha256'])
opening = read(JOB/'admission/FINAL_OPENING.json', contract['opening_sha256'])
config = read(JOB/'admission/FINAL_ATTACK_CONFIGURATION.json', contract['configuration_sha256'])
model_index = read(JOB/'admission/FINAL_MODEL_INDEX.json', policy['files']['admission/FINAL_MODEL_INDEX.json']['sha256'])
roster = read(JOB/'admission/FINAL_ROSTER_ACCEPTED.json', policy['files']['admission/FINAL_ROSTER_ACCEPTED.json']['sha256'])
need(graph['d2_policy'] == D2.POLICY and opening['D2']['cells'] == graph['d2_cells'], 'Frozen D2 policy/cells differ')
need(len(graph['d2_cells']) == 20, 'Expected 20 cells')
spec = config['evaluation']['final']
opening_sha = contract['opening_sha256']
config_sha = contract['configuration_sha256']

primary_verified = read(JOB/'evidence/completed_primary/FINAL_CPU_RETURN_VERIFIED.json', policy['files']['evidence/completed_primary/FINAL_CPU_RETURN_VERIFIED.json']['sha256'])
ordinary_verified = read(JOB/'evidence/completed_ordinary/ORDINARY_CPU_RETURN_VERIFIED.json', policy['files']['evidence/completed_ordinary/ORDINARY_CPU_RETURN_VERIFIED.json']['sha256'])
need(primary_verified['all_twenty_results_only_merges_verified'] is True, 'Primary verification absent')
need(ordinary_verified['all_results_only_merges_reconstructed_and_matched'] is True, 'Ordinary verification absent')
controls = {}
control_bindings = {}
all_controls = sorted({n for c in graph['d2_cells'].values() for n in c['controls']})
need(len(all_controls) == 41, 'Expected 41 controls')
for member in all_controls:
    owner, name = member.split('/')
    primary = name in ('O', 'M3')
    base = ROOT/'prior_controls'/('primary' if primary else 'ordinary')/'merged'/member
    previous = (primary_verified if primary else ordinary_verified)['results'][member]
    expected_sha = previous['result_sha256' if primary else 'merged_result_sha256']
    record = read(base/'RESULT.json', expected_sha)
    need(record['complete'] is True and record['domain'] == record['phase'] == 'final', 'Incomplete/non-final control: '+member)
    need(record['specification'] == spec and record['task_status'] == 'measured', 'Control specification/task status differs: '+member)
    need(all(math.isfinite(record['metrics'][k]) and record['metrics'][k] >= 0 for k in 'FJKT'), 'Invalid full control metrics')
    need(len(record['scores']) == 4096 and all(math.isfinite(x) for x in record['scores']), 'Invalid control scores')
    req = read(base/'D6_REQUEST.json', record['cpu_return']['request_sha256'])
    binding = req['binding']
    need(binding['owner'] == owner and binding['opening_sha256'] == opening_sha and binding['configuration_sha256'] == config_sha and binding['phase'] == 'final', 'Control binding differs: '+member)
    need(binding['model_sha256'] == model_index[member]['descriptor']['model_sha256'], 'Control model identity differs: '+member)
    need(binding['phase_name'] == 'evaluations/'+member, 'Control phase name differs')
    need(binding['source_pins'] == policy['source_pins'], 'Control science version differs')
    if primary:
        need(record['metrics'] == previous['metrics'], 'Previously reviewed primary metrics differ')
    controls[member] = record
    control_bindings[member] = {'result_sha256': expected_sha, 'request_sha256': record['cpu_return']['request_sha256'], 'model_sha256': binding['model_sha256'], 'opening_sha256': opening_sha}

owner_contexts = {}
for owner in graph['planned_owners']:
    qpath = JOB/'qualification_current'/owner/'QUALIFICATION.json'
    qsha = roster['bindings']['qualification_hashes'][owner]
    q = read(qpath, qsha)
    route = R.owner_route(policy['study_id'], owner, roster['bindings']['marked_model_hashes'][owner], policy['clean_models'][owner], qpath, qsha, q['bindings'], ['ordinary'])
    need(route['B3_qualified'] is True and len(route['descendants']) == 1, 'Marked membership lacks qualified parent')
    current = dict(route['descendants'][0], clean_metrics=controls[owner+'/O']['metrics'])
    initial = R.assess_descendant(current, controls[owner+'/M3']['metrics'], controls[owner+'/M3']['scores'])
    need(initial['available'] is True, 'Same-test intact evidence unavailable')
    owner_contexts[owner] = (current, initial)

raw320_index = read(JOB/'evidence/completed_D2_raw320/D2_RAW320_DIAGNOSTICS_INDEX.json', policy['files']['evidence/completed_D2_raw320/D2_RAW320_DIAGNOSTICS_INDEX.json']['sha256'])
raw960_receipt = read(ROOT/'d2_return/output/D2_RAW960_SCOPED_DIAGNOSTICS_RECEIPT.json')
raw960_index = read(ROOT/'d2_return/output/DIAGNOSTIC_INDEX.json', raw960_receipt['diagnostic_index_sha256'])
need(raw960_receipt['complete'] is True and raw960_receipt['domain_rewritten'] is False, 'Raw960 scope receipt unavailable')
need(raw960_receipt['policy_sha256'] == contract['inputs']['raw960']['policy_sha256'] and raw960_receipt['mechanical_runtime_admission_sha256'] == contract['inputs']['raw960']['runtime_admission_sha256'], 'Raw960 scope differs')
need(raw960_receipt['opening_sha256'] == opening_sha and raw960_index['opening_sha256'] == opening_sha, 'Raw960 opening differs')
need(raw960_index['cells'] == {k:v for k,v in graph['d2_cells'].items() if v['checkpoint']=='raw960'}, 'Raw960 cells differ')
models960 = {x['id']:x for x in policy['model_jobs']}
diagnostics = {}
diagnostic_bindings = {}
for key, cell in graph['d2_cells'].items():
    for member in cell['members']:
        owner, name = member.split('/')
        raw960 = cell['checkpoint'] == 'raw960'
        if raw960:
            ix = raw960_index['diagnostics'][member]
            base = ROOT/'d2_return/output'/ix['directory']
        else:
            ix = raw320_index['models'][member]
            base = ROOT/'raw320'/Path(ix['result_member']).parent
        value = read(base/'RESULT.json', ix['result_sha256'])
        start = read(base/'START.json')
        expected_domain = 'development' if raw960 else 'final'
        need(value['domain'] == start['domain'] == expected_domain, 'Original diagnostic domain differs')
        need(value['specification'] == start['specification'] == spec and value['phase'] == 'final', 'Diagnostic population differs')
        need(value['diagnostics_complete'] is True and value['complete'] is False and value['task_status'] == 'pending', 'Diagnostic task state differs')
        eb = start['evaluation_binding']
        expected_measurement = models960[member]['measurement_sha256'] if raw960 else ix['measurement_sha256']
        need(eb['model_sha256'] == expected_measurement and eb['wrong_key_count'] == 4095 and eb['wrapper_steps'] is None, 'Diagnostic measurement binding differs')
        original_bindings = eb['bindings']
        if raw960:
            need(eb == ix['evaluation_binding'], 'Raw960 indexed binding differs')
            need(original_bindings['scope']=='D2_raw960_diagnostics_only' and original_bindings['mechanical_runtime_admission_sha256']==contract['inputs']['raw960']['runtime_admission_sha256'] and original_bindings['policy_sha256']==contract['inputs']['raw960']['policy_sha256'], 'Raw960 runtime scope differs')
            original_bindings = original_bindings['historical_final_bindings']
        orow = policy['owner_records'][owner]
        expected_bindings = {'configuration': config_sha, 'calibration': policy['files']['admission/FINAL_CALIBRATION.json']['sha256'], 'clean_model': policy['clean_models'][owner], 'inputs': policy['files'][orow['inputs']]['sha256'], 'runtime': policy['files'][orow['runtime']]['sha256'], 'clean_evaluation': policy['files'][orow['clean_evaluation']]['sha256']}
        need(original_bindings == expected_bindings and eb['assets'] == policy['files'][orow['assets']]['sha256'], 'Historical final diagnostic inputs differ')
        need(len(value['scores']) == 4096 and value['counts']['validation_controls'] == 512 and value['counts']['detector_controls'] == 128, 'Diagnostic population incomplete')
        diagnostics[member] = value
        diagnostic_bindings[member] = {'result_sha256': ix['result_sha256'], 'start_sha256': sha(base/'START.json'), 'original_domain': expected_domain, 'measurement_sha256': expected_measurement, 'scope': 'D2_raw960_diagnostics_only' if raw960 else 'historically_admitted_final_diagnostics'}
need(len(diagnostics) == 400, 'Expected 400 distinct diagnostics')

native_verified = read(ROOT/'NATIVE_RETURN_VERIFIED_20261024T072602Z.json')
native_result = read(ROOT/'return/output/RESULT.json')
native_closure = read(ROOT/'return/supervision/RESULT.json')
need(native_result['complete'] and native_closure['complete'], 'Shared native products/closure incomplete')
need(native_result['opening_sha256'] == opening_sha, 'Shared opening differs')
shared = {}
for name in ('F64', 'P256'):
    p = ROOT/'return/output'/name/'RESULT.json'
    value = read(p, native_result['results'][name])
    need(value['complete'] is True, 'Shared task incomplete')
    shared[name] = sha(p)

decisions = {}
needed = []
skipped = []
for key, cell in sorted(graph['d2_cells'].items()):
    contexts = {n:(dict(owner_contexts[n.split('/')[0]][0],variant=cell['family']),owner_contexts[n.split('/')[0]][1]) for n in cell['marked_members']}
    value = D2.decide(cell, {n:diagnostics[n] for n in cell['members']}, contexts, {n:controls[n] for n in cell['controls']}, shared)
    decisions[key] = value
    write(OUT/'decisions'/key/'DECISION.json', value)
    (needed if value['task_required'] else skipped).extend(cell['members'])
    if not value['task_required']:
        for n in cell['members']:
            record = dict(schema='OperatorMark_D2_unmeasured_scoped_record.v1',source_result_sha256=diagnostic_bindings[n]['result_sha256'],original_domain=diagnostics[n]['domain'],source_record_modified=False,complete=False,task_status='unmeasured',task_reason=value['reason'],d2_member=n,d2_decision_sha256=sha(OUT/'decisions'/key/'DECISION.json'),task_utility=None,complete_task_curve=False,scientific_acceptance=False,formal_custody_assurance=False)
            if n in contexts:
                route, initial = contexts[n]
                view = dict(diagnostics[n], task_status='unmeasured',d2_decision=value,d2_member=n,task_reason=value['reason'])
                record['marked_readout'] = D2.skipped_readout(route,view,initial)
            write(OUT/'unmeasured'/n/'TASK_STATUS.json',record)

write(OUT/'SOURCE_MAPPING.json',{'adapter_contract_sha256':CONTRACT_SHA,'files':evidence,'control_bindings':control_bindings,'diagnostic_bindings':diagnostic_bindings,'shared_result_hashes':shared,'same_test_owner_contexts':{k:{'route':v[0],'initial':v[1]} for k,v in owner_contexts.items()},'preserved_input_labels':True,'scientific_source_files_modified':False})
summary = dict(schema='OperatorMark_D2_results_only_selection.v1',status='D2_TASK_SELECTION_COMPLETE',adapter_contract_sha256=CONTRACT_SHA,script_sha256=sha(__file__),cells=20,diagnostics_reused=400,controls_reused=41,shared_products_reused=2,task_required_cells=[k for k,v in decisions.items() if v['task_required']],task_unmeasured_cells=[k for k,v in decisions.items() if not v['task_required']],task_required_members=sorted(needed),task_unmeasured_members=sorted(skipped),task_required_count=len(needed),task_unmeasured_count=len(skipped),marked_outcomes=dict(Counter(x['state'] for v in decisions.values() for x in v['outcomes'].values())),decision_files={k:sha(OUT/'decisions'/k/'DECISION.json') for k in decisions},models_evaluated=0,solver_calls=0,training_updates=0,new_runtime_activated=False,full_evaluations_unchanged=851,canonical_worker_advanced=False,formal_custody_assurance=False,scientific_acceptance=False,physical_qualification=False,source_domain_rewritten=False,remaining='Selected conditional tasks if any; noise20, API30, keyed-null/matched-independent readouts, owner analysis and release remain.')
write(OUT/'D2_SELECTION_SUMMARY.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ('task_required_members','task_unmeasured_members','decision_files')},indent=2))
