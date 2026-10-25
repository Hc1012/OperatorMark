"""Explicit post-generation scientific integrity; physical D10 stays unchanged."""
from pathlib import Path
from datetime import datetime,timedelta
from . import scientific as S
from .d7 import policy as d7_policy

SCHEMA='operatormark.scientific_corpus.D8.v1'
PHYSICAL_REASON='anomaly field qualification not confirmed'

def policy(binding):
    S.need(type(binding) is dict and set(binding)=={'path','sha256'},'D8 external binding required')
    S.need(S.sha(binding['path'])==binding['sha256'],'D8 amendment drift')
    row=S.read(binding['path'])
    S.need(row.get('schema')=='scientific_D8_adoption.v1' and row.get('adopted') is True
        and row.get('post_generation') is True and row.get('observed_relaxed_anomalies')==76
        and row.get('original_rejection_preserved') is True
        and row.get('only_nonblocking_condition')==PHYSICAL_REASON,'D8 scope/timing differs')
    return row

def integrity(row,binding):
    policy(binding)
    c=row.get('checked',{});r=row.get('recomputed',{});physical=row.get('d10',{})
    complete=(c.get('samples')==4864 and c.get('directions')==5632 and c.get('shards_expected')==76
        and c.get('shards_written')==76 and c.get('native_numerics_checked')==4864 and c.get('p1_resolved')==4864
        and c.get('numerics')=='full' and c.get('p1')=='resolve' and c.get('direction_rule_rederived') is True)
    retained=all(k in r and not r[k] for k in ('gate_failures','failures_retained','unwritten_measurements'))
    physical_ok=(physical==dict(verdict='accepted',reasons=[]) or
        (physical==dict(verdict='not_accepted',reasons=[PHYSICAL_REASON])
         and row.get('anomaly_field_status',{}).get('state')=='evaluated'
         and 'not_confirmed' in (row['anomaly_field_status'].get('relaxed'),row['anomaly_field_status'].get('binary'))
         and all(row['anomaly_field_status'].get(k) in ('not_confirmed','unchanged') for k in ('relaxed','binary'))))
    coverage=row.get('historical_coverage')
    coverage_ok=isinstance(coverage,dict) and coverage==d7_policy(coverage.get('policy'))
    ok=(row.get('status')=='matched' and row.get('diagnostics')==[] and complete and retained and physical_ok
        and row.get('writer_claims')=='consistent' and row.get('production_usable') is False and coverage_ok)
    return dict(schema='scientific_integrity.D8.v1',amendment=binding,
        verdict='accepted' if ok else 'not_accepted',physical_d10=physical,
        post_generation=True,physical_qualification='preserved_negative',historical_coverage=coverage)

def require(row,binding):
    decision=integrity(row,binding)
    S.need(row.get('scientific_integrity')==decision and decision['verdict']=='accepted'
        and row.get('scientific_usable') is True,'D8 scientific integrity incomplete/failed')
    return decision

def propagate(row,receipt):
    if receipt.get('schema')==SCHEMA:
        binding=receipt.get('d8_policy');policy(binding)
        S.need(row.get('d8_policy')==binding and row.get('physical_verdicts')==receipt.get('physical_verdicts')
            and row.get('admission_timing')=='post_generation_after_76_relaxed_anomalies',
            'D8 downstream amendment/physical verdict missing')

def verify_reader(plan):
    required={'operatormark_corpus_v2/d8.py','operatormark_corpus_v2/readback.py',
        'operatormark_corpus_v2/scientific.py','tools/msfinal/scientific_admission.py','tools/scientific_d8_continue.py'}
    S.need(required<=set(plan['reader_pins']),'D8 reader inventory coverage')
    for n,h in plan['reader_pins'].items():S.need(S.sha(S.member(plan['reader_root'],n))==h,'D8 reader drift: '+n)

def verify_plan(plan):
    amendment=policy(plan['d8_policy']);verify_reader(plan)
    for path,h in plan['original_bindings'].items():S.need(S.sha(path)==h,'original evidence drift: '+path)
    sp=S.read(plan['producer_spec'])
    approval=S.read(plan['original_approval'])
    anchor=datetime.fromisoformat(approval['recorded_utc'].replace('Z','+00:00'))
    deadline=datetime.fromisoformat(plan['deadline_utc'].replace('Z','+00:00'))
    S.need(plan['deadline_anchor_utc']==approval['recorded_utc'] and deadline==anchor+timedelta(seconds=sp['overall_seconds'])
        and sp['overall_seconds']==64800 and plan['work_seconds']==dict(readback1=8400,readback2=14400,seal=1800)
        and plan['cleanup_seconds']==30 and plan['finalization_seconds']==30,'D8 original deadline/bounds differ')
    old=S.read(Path(plan['original_root'])/'readback1/invocation.json')['wall_seconds']
    S.need(plan['prior_readback1_seconds']==old and plan['prior_diagnostic_seconds']>=123.0266314
        and plan['preparation_check_reserve_seconds']==120
        and old+plan['prior_diagnostic_seconds']+120+8400+60<=sp['work_seconds']['readback1'],
        'cumulative readback1 allowance exhausted')
    S.need(S.sha(plan['producer_spec'])==amendment['producer_spec_sha256']
        and str(Path(plan['original_root']).resolve())==str(Path(amendment['original_root']).resolve()),'D8 original corpus mismatch')
    S.verify_sources(sp['source_root'],sp['source_pins'])
    S.need(S.closed(S.read(Path(plan['original_root'])/'generate/invocation.json'))
        and S.read(Path(plan['original_root'])/'generate/RESULT.json').get('complete') is True,'original generation incomplete')
    S.need(S.read(Path(plan['original_root'])/'generate/RESULT.json')['spec_sha256']==S.digest(sp),'original generation/spec binding')
    dep=S.read(sp['frozen_dependency_binding']['path'])
    S.need(S.sha(sp['frozen_dependency_binding']['path'])==sp['frozen_dependency_binding']['sha256'],'frozen dependency binding drift')
    S.verify_files(dep['source_root'],dep['files'])
    return sp

def pair(plan,root,*,current_inventory=None):
    """Independent receipt checks, no numerical re-execution."""
    root=Path(root);sp=verify_plan(plan);rows=[];identities=[]
    for name in ('readback1','readback2'):
        receipt=S.read(root/name/'invocation.json');S.need(S.closed(receipt),'D8 readback closure')
        S.need(not (root/name/'FAILURE.json').exists() and not (root/name/'RESOURCE_FAILURE.json').exists(),'D8 failed readback retained')
        identities.append((receipt['child_pid'],receipt['started_utc']))
        row=S.read(root/name/'RESULT.json')
        S.need(row.get('stage')==name and row.get('plan_sha256')==S.digest(plan)
            and row.get('producer_spec_sha256')==S.sha(plan['producer_spec'])
            and row.get('reader_pins')==plan['reader_pins'] and row.get('expectation_sha256')==sp['expected_sha256'],
            'D8 readback identity/binding')
        evidence=S.read(root/name/'ASSESSMENT.json')
        S.need(S.sha(root/name/'ASSESSMENT.json')==row.get('assessment_sha256')
            and evidence.get('assessment')==row['assessment'] and evidence.get('accepted_readback') is False
            and evidence.get('plan_sha256')==S.digest(plan) and evidence.get('reader_pins')==plan['reader_pins']
            and evidence.get('producer_spec_sha256')==S.sha(plan['producer_spec']),'D8 raw assessment binding')
        require(row['assessment'],plan['d8_policy'])
        S.need(row['assessment']['historical_coverage']==d7_policy(sp['d7_policy']),'D7 lost in readback')
        rows.append(row)
    S.need(len(set(identities))==2,'distinct independent readers required')
    S.need(rows[0]['corpus_inventory']==rows[1]['corpus_inventory'],'corpus mutation across readers')
    if current_inventory is not None:S.need(rows[0]['corpus_inventory']==current_inventory,'corpus changed at sealing')
    return sp,rows

def seal(plan,root):
    root=Path(root);sp,rows=pair(plan,root,current_inventory=S.inventory(Path(plan['original_root'])/'corpus'))
    files=S.inventory(root,exclude_top=('seal',))
    S.write(root/'OUTPUT_MANIFEST.json',dict(schema=SCHEMA,files=files,corpus_files=rows[0]['corpus_inventory']))
    receipt=dict(schema=SCHEMA,scope='scientific_only',synthetic=False,complete=True,production_accepted=False,
        source_pins=sp['source_pins'],reader_pins=plan['reader_pins'],reader_root=plan['reader_root'],
        original_root=plan['original_root'],decision_sha256=sp['decision_sha256'],split_commitment=sp['split_commitment'],
        test_sealed=True,manifest_sha256=S.sha(root/'OUTPUT_MANIFEST.json'),plan_sha256=S.digest(plan),
        d8_policy=plan['d8_policy'],historical_coverage=d7_policy(sp['d7_policy']),
        admission_timing='post_generation_after_76_relaxed_anomalies',physical_qualification='preserved_negative',
        physical_verdicts=[r['assessment']['d10'] for r in rows],
        scientific_integrity=[r['assessment']['scientific_integrity'] for r in rows],
        authority_sha256=S.read(root/'AUTHORITY_BINDING.json')['sha256'])
    S.write(root/'SCIENTIFIC_RECEIPT.json',receipt);return receipt

def read_admission(admission):
    root=Path(admission.root);r=S.read(root/'SCIENTIFIC_RECEIPT.json')
    S.need(not (root/'FAILURE.json').exists() and not (root/'seal/FAILURE.json').exists()
        and not (root/'seal/RESOURCE_FAILURE.json').exists(),'D8 retained continuation failure')
    S.need(S.sha(root/'SCIENTIFIC_RECEIPT.json')==admission.receipt_sha256 and r.get('schema')==SCHEMA,'D8 receipt identity')
    S.need(S.sha(root/'CLOSURE.json')==admission.closure_sha256,'D8 closure drift')
    close=S.read(root/'CLOSURE.json')
    S.need(close.get('complete') is True and close.get('timed_out') is False
        and close.get('stages_closed')==['readback1','readback2','seal']
        and close.get('scientific_receipt_sha256')==admission.receipt_sha256
        and close.get('sealing_invocation_sha256')==S.sha(root/'seal/invocation.json')
        and S.closed(S.read(root/'seal/invocation.json')),'D8 final closure incomplete')
    plan=S.read(root/'PLAN.json');S.need(S.digest(plan)==r['plan_sha256'],'D8 plan drift')
    S.need(close.get('original_deadline_utc')==plan['deadline_utc']
        and datetime.fromisoformat(close['finished_utc'])<=datetime.fromisoformat(plan['deadline_utc'].replace('Z','+00:00')),
        'D8 closure beyond original deadline')
    sp,rows=pair(plan,root)
    S.need(Path(admission.source_root).resolve()==Path(sp['source_root']).resolve()
        and admission.decision_sha256==sp['decision_sha256']==r['decision_sha256'],'D8 producer admission binding')
    S.need(r.get('d8_policy')==plan['d8_policy'] and r.get('source_pins')==sp['source_pins']
        and r.get('reader_pins')==plan['reader_pins'] and r.get('original_root')==plan['original_root']
        and r.get('historical_coverage')==d7_policy(sp['d7_policy'])
        and r.get('physical_verdicts')==[x['assessment']['d10'] for x in rows]
        and r.get('scientific_integrity')==[x['assessment']['scientific_integrity'] for x in rows]
        and r.get('scope')=='scientific_only' and r.get('synthetic') is False and r.get('complete') is True
        and r.get('production_accepted') is False and r.get('test_sealed') is True,'D8 receipt scope/results differ')
    S.need(r.get('split_commitment')==sp['split_commitment'] and r.get('reader_root')==plan['reader_root']
        and r.get('admission_timing')=='post_generation_after_76_relaxed_anomalies'
        and r.get('physical_qualification')=='preserved_negative','D8 reporting/reader binding differs')
    binding=S.read(root/'AUTHORITY_BINDING.json')
    S.need(binding['sha256']==r['authority_sha256']==S.sha(binding['path']),'D8 authority drift')
    auth=S.read(binding['path']);S.need(auth.get('approved') is True and auth.get('plan_sha256')==S.digest(plan)
        and auth.get('scope')=='one_D8_existing_corpus_two_readbacks_and_seal'
        and auth.get('deadline_utc')==plan['deadline_utc'] and auth.get('reader_pins')==plan['reader_pins']
        and auth.get('output_root')==str(root),'D8 authority/plan mismatch')
    S.need(S.sha(root/'OUTPUT_MANIFEST.json')==r['manifest_sha256'],'D8 manifest drift')
    m=S.read(root/'OUTPUT_MANIFEST.json');S.verify_files(root,m['files'])
    S.need(m['corpus_files']==rows[0]['corpus_inventory'],'D8 corpus manifest correspondence')
    return r,{**m['files'],**{'corpus/'+n:f for n,f in m['corpus_files'].items()}}
