"""Scientific-only S128 route. No State, E3, P attestation or execution grant.

The caller independently supplies reviewed expectations. Receipts bind observed
work and bytes, not a producer's assertion of formal production acceptance.
Numerical imports are lazy. Metadata admission never opens sealed test arrays.
"""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import time

SCHEMA='operatormark.scientific_corpus.v1'
STAGES=('generate','readback1','readback2')
COUNTS=(4096,256,512)

def need(ok,message):
    if not ok:raise ValueError(message)

def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
def digest(value):return hashlib.sha256(encode(value)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):
    path=Path(path)
    with path.open('xb') as f:f.write(encode(value));f.flush();os.fsync(f.fileno())
    need(path.read_bytes()==encode(value),'record readback differs')
    return sha(path)

def relative(name):
    p=PurePosixPath(name)
    need(type(name) is str and bool(p.parts) and not p.is_absolute() and '..' not in p.parts
         and '\\' not in name and ':' not in name and p.as_posix()==name,'unsafe member path')
    return p

def member(root,name):
    root=Path(root).absolute();p=root.joinpath(*relative(name).parts)
    for x in (p,*p.parents):
        if x.exists():need(not x.is_symlink() and not getattr(x.lstat(),'st_file_attributes',0)&0x400,'reparse member')
        if x==root:break
    need(p.resolve().is_relative_to(root.resolve()),'member escapes root')
    return p

def inventory(root,*,exclude_top=()):
    root=Path(root);out={}
    for directory,dirs,files in os.walk(root,followlinks=False):
        if Path(directory)==root:
            dirs[:]=[n for n in dirs if n not in exclude_top]
            files=[n for n in files if n not in exclude_top]
        for n in dirs+files:
            p=Path(directory)/n;member(root,p.relative_to(root).as_posix())
        for n in files:
            p=Path(directory)/n;before=p.stat();h=sha(p);after=p.stat()
            need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'member mutated')
            out[p.relative_to(root).as_posix()]=dict(bytes=after.st_size,sha256=h)
    return dict(sorted(out.items()))

def verify_files(root,files):
    for name,f in files.items():
        p=member(root,name)
        need(p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],'bound member missing/drifted: '+name)

def verify_sources(root,pins):
    need(type(pins) is dict and bool(pins),'source inventory missing')
    required={'operatormark_corpus_v2/'+n+'.py' for n in ('scientific','config','generator','readback','input_bindings','records','layout','native','pins','anomaly','d10')}
    required|={'tools/scientific_corpus.py','tools/q6_common.py','tools/msfinal/scientific_admission.py','tools/msfinal/owners.py','tools/msfinal/materials.py','tools/msfinal/backend.py','tools/msfinal/orchestration.py'}
    for package in ('operatormark_corpus_v2','operatormark_numerics_v2'):
        required.update(p.relative_to(root).as_posix() for p in (Path(root)/package).rglob('*.py'))
    need(required<=set(pins),'scientific source coverage missing')
    for n,h in pins.items():need(sha(member(root,n))==h,'source drift: '+n)

def closed(r):
    c=r.get('cleanup',{})
    return (r.get('exit_code')==0 and r.get('timed_out') is False and c.get('resolved') is True
        and c.get('child_reaped') is True and c.get('group_survivors') is False
        and c.get('job_accounting',{}).get('active_processes')==0 and r.get('job_handle_closed') is True)

def numerical_pass(row,*,synthetic=False):
    c=row.get('checked',{});r=row.get('recomputed',{})
    if synthetic:
        need(row.get('status') in ('matched','incomplete') and row.get('production_usable') is False
            and c.get('numerics')=='structural' and c.get('p1')=='skip'
            and c.get('native_numerics_checked')==0 and c.get('p1_resolved')==0,'engineering readback scope')
        need(all(d[0]=='incomplete' and d[2] in ('native_numerics_not_verified','p1_not_resolved','anomaly_ladder_not_independently_verified')
                 for d in row.get('diagnostics',[])),'engineering readback inconsistency')
        return
    need(row.get('status')=='matched' and row.get('d10',{}).get('verdict')=='accepted','readback incomplete/failed')
    need(not r.get('gate_failures') and not r.get('failures_retained') and not r.get('unwritten_measurements'),'retained scientific failures')
    if not synthetic:
        need(row.get('scientific_usable') is True and row.get('production_usable') is False,'distinct scientific result required')
        need(c.get('samples')==4864 and c.get('directions')==5632 and c.get('shards_expected')==76
            and c.get('shards_written')==76 and c.get('native_numerics_checked')==4864
            and c.get('p1_resolved')==4864 and c.get('numerics')=='full' and c.get('p1')=='resolve'
            and c.get('direction_rule_rederived') is True,'full S128 numerical population required')

def seal(root,spec,*,synthetic=False):
    """Parent-only after both separately supervised readers; never opens models."""
    root=Path(root);verify_sources(spec['source_root'],spec['source_pins'])
    need(spec['retention']=='S' and spec['n_ref']==128 and spec['counts']==list(COUNTS),'adopted population differs')
    receipts=[read(root/s/'invocation.json') for s in STAGES]
    need(all(closed(r) for r in receipts),'owned process/stream closure incomplete')
    need(len({(r['child_pid'],r['started_utc']) for r in receipts})==3,'distinct stage process identities required')
    for stage in STAGES:
        value=read(root/stage/'RESULT.json')
        need(value['spec_sha256']==digest(spec) and value['stage']==stage,'stage expectation substitution')
    a=read(root/'readback1/RESULT.json');b=read(root/'readback2/RESULT.json')
    if spec.get('d7_policy') is not None:
        from .d7 import policy
        coverage=policy(spec['d7_policy'],decision_sha256=spec['decision_sha256'])
        need(a['assessment'].get('historical_coverage')==b['assessment'].get('historical_coverage')==coverage,
             'D7 independent readback limitation differs')
    numerical_pass(a['assessment'],synthetic=synthetic);numerical_pass(b['assessment'],synthetic=synthetic)
    need(a['corpus_inventory']==b['corpus_inventory']==inventory(root/'corpus'),'corpus changed across independent passes')
    need(a['expectation_sha256']==b['expectation_sha256']==spec['expected_sha256'],'independent expectations differ')
    need(read(root/'generate/RESULT.json')['complete'] is True,'generation incomplete')
    # The finalization worker's own stdout/receipt are still being produced.
    # Its independent parent binds those after closure, without re-reading data.
    files=inventory(root,exclude_top=('seal',))
    write(root/'OUTPUT_MANIFEST.json',dict(schema=SCHEMA,files=files))
    result=dict(schema=SCHEMA,scope='scientific_only',synthetic=synthetic,complete=True,
        production_accepted=False,formal_assurance='deferred',physical_qualification='preserved_negative',
        decision_sha256=spec['decision_sha256'],spec_sha256=digest(spec),source_pins=spec['source_pins'],
        manifest_sha256=sha(root/'OUTPUT_MANIFEST.json'),expectation_sha256=spec['expected_sha256'],
        split_commitment=spec['split_commitment'],test_sealed=True,readbacks=['readback1','readback2'])
    if spec.get('d7_policy') is not None:
        from .d7 import policy
        result['historical_coverage']=policy(spec['d7_policy'],decision_sha256=spec['decision_sha256'])
    write(root/'SCIENTIFIC_RECEIPT.json',result)
    return result

@dataclass(frozen=True)
class CorpusAdmission:
    root: Path
    receipt_sha256: str
    decision_sha256: str
    source_root: Path
    closure_sha256: str

    def read(self):
        root=Path(self.root);p=root/'SCIENTIFIC_RECEIPT.json'
        need(sha(p)==self.receipt_sha256,'scientific receipt drift')
        if read(p).get('schema')=='operatormark.scientific_corpus.D8.v1':
            from .d8 import read_admission
            return read_admission(self)
        need(sha(root/'CLOSURE.json')==self.closure_sha256,'scientific closure drift')
        closure=read(root/'CLOSURE.json')
        need(closure.get('complete') is True and closure.get('timed_out') is False
            and closure.get('stages_closed')==list(STAGES)+['seal']
            and closure.get('scientific_receipt_sha256')==self.receipt_sha256,'scientific final closure incomplete')
        need(sha(root/'seal/invocation.json')==closure.get('sealing_invocation_sha256')
            and closed(read(root/'seal/invocation.json')),'sealing process closure incomplete')
        r=read(p)
        if 'historical_coverage' in r:
            from .d7 import policy
            need(r['historical_coverage']==policy(r['historical_coverage']['policy'],
                decision_sha256=self.decision_sha256),'D7 corpus limitation differs')
        need(r.get('schema')==SCHEMA and r.get('scope')=='scientific_only' and r.get('synthetic') is False
            and r.get('complete') is True and r.get('production_accepted') is False
            and r.get('test_sealed') is True and r.get('decision_sha256')==self.decision_sha256,'scientific receipt inapplicable')
        verify_sources(self.source_root,r['source_pins'])
        need(sha(root/'OUTPUT_MANIFEST.json')==r['manifest_sha256'],'manifest drift')
        manifest=read(root/'OUTPUT_MANIFEST.json')['files']
        # Metadata-only admission. Do not read the sealed corpus here.
        names=[s+'/'+n for s in STAGES for n in ('RESULT.json','invocation.json')]
        verify_files(root,{n:manifest[n] for n in names})
        need(all(closed(read(root/s/'invocation.json')) for s in STAGES),'closure drift')
        for s in ('readback1','readback2'):
            q=read(root/s/'RESULT.json');numerical_pass(q['assessment'])
            if 'historical_coverage' in r:
                need(q['assessment'].get('historical_coverage')==r['historical_coverage'],'D7 readback receipt limitation differs')
            need(q['spec_sha256']==r['spec_sha256'] and q['expectation_sha256']==r['expectation_sha256'],'readback binding')
        a=read(root/'readback1/RESULT.json');b=read(root/'readback2/RESULT.json')
        need(a['corpus_inventory']==b['corpus_inventory']=={
            n[len('corpus/'):]:f for n,f in manifest.items() if n.startswith('corpus/')},'independent readback/manifest membership differs')
        need(read(root/'generate/RESULT.json').get('complete') is True,'generation receipt incomplete')
        return r,manifest

def corpus_bytes(admission,name,*,split,opening=None):
    r,m=admission.read();need(split in ('training','validation','test'),'split')
    directory={'training':'training','validation':'validation','test':'test.sealed'}[split]
    need(name.startswith('corpus/corpus_staging/'+directory+'/'),'split substitution')
    if split=='test':
        # The numerical corpus loader cannot accept a caller-supplied boolean.
        # Final test reads belong to msfinal.materials.entry after its existing
        # independently accepted whole-roster opening gate.
        raise ValueError('sealed corpus: use coordinated msfinal asset opening')
    need(name in m,'unmanifested corpus member')
    source_root=r['original_root'] if r.get('schema')=='operatormark.scientific_corpus.D8.v1' else admission.root
    verify_files(source_root,{name:m[name]})
    return member(source_root,name).read_bytes()

def scientific_controls(arrays,expected):
    from .config import MappingControls
    from .input_bindings import ScientificControlExpectations
    need(type(expected) is ScientificControlExpectations,'scientific controls expectation')
    source=MappingControls({(s,i):a for s,rows in arrays.items() for i,a in enumerate(rows)},kind='scientific_controls',synthetic=False)
    expected.verify(source.control_hashes(),production=False)
    source.admission_expectations=expected
    source._binding['admission_sha256']=expected.review_record_sha256
    return source

def prepare_control_material(seed_file,seed_sha256,out,*,exclusion_file,exclusion_sha256,decision_sha256,source_root,source_pins,d7_policy=None):
    """Future authorized material production; no seed draws and no acceptance.

    Reuses the section-6 sampler; seeds must be supplied in separate committed
    streams. Qualification/confirmation/development exclusions are mandatory.
    Every collision refuses; there is no redraw or replacement.
    """
    import numpy as np
    from q6_common import development_controls,array_save
    from .config import D6Directions
    from .input_bindings import control_content_sha256
    from operatormark.corpus import canonical as C
    from operatormark_numerics_v2.lock import public_deny_list
    verify_sources(source_root,source_pins)
    need(sha(seed_file)==seed_sha256 and sha(exclusion_file)==exclusion_sha256,'material prerequisite drift')
    seeds=read(seed_file);exclusion=read(exclusion_file)
    need(set(seeds)=={'training','validation','test','directions'} and len(set(seeds.values()))==4
        and all(type(v) is int and 2**64<=v<2**128 for v in seeds.values()),'separate supplied formal-range streams required')
    coverage=None
    if d7_policy is not None:
        from .d7 import policy
        coverage=policy(d7_policy,decision_sha256=decision_sha256,exclusion=exclusion,exclusion_sha256=exclusion_sha256)
    need((coverage is not None or exclusion.get('complete_roles')==['qualification','confirmation','development'])
        and isinstance(exclusion.get('control_hashes'),list) and bool(exclusion['control_hashes'])
        and isinstance(exclusion.get('direction_hashes'),list) and bool(exclusion['direction_hashes']),
        'complete independently reviewed exclusion population required')
    arrays={s:development_controls(seeds[s],n) for s,n in zip(('training','validation','test'),COUNTS)}
    hashes={(s,i):C.tensor_sha256(a) for s,rows in arrays.items() for i,a in enumerate(rows)}
    denied=set(exclusion['control_hashes'])|public_deny_list()['control_sha256']
    need(len(set(hashes.values()))==4864 and not set(hashes.values())&denied,'control collision; no redraw')
    directions=D6Directions(seeds['directions'],synthetic=False);dh=[];saved_directions={}
    for s,rows in arrays.items():
        saved_directions[s]=[]
        for i,a in enumerate(rows):
            values=[directions.direction(s,i,k,a) for k in range(1 if s=='training' else 2)]
            dh.extend(C.tensor_sha256(v) for v in values)
            saved_directions[s].append(values)
    need(len(set(dh))==5632 and not set(dh)&set(exclusion['direction_hashes']),'direction collision; no redraw')
    out=Path(out);out.mkdir(exist_ok=False)
    for s,a in arrays.items():array_save(out/(s+'.npy'),a)
    if coverage is not None:
        for s,rows in saved_directions.items():array_save(out/(s+'_directions.npy'),np.asarray(rows))
    write(out/'direction_seed.json',dict(seed=seeds['directions']))
    # Candidate for independent review; never carries accepted=true or a State role.
    candidate=dict(schema='scientific_controls_review.v1',scope='scientific_only',decision_sha256=decision_sha256,
        controls=[[s,i,h] for (s,i),h in sorted(hashes.items())],aggregate_control_content_sha256=control_content_sha256(hashes),
        excluded_control_hashes=sorted(denied),disjointness_complete=True)
    if coverage is not None:
        candidate.update(schema='scientific_controls_candidate.D7.v1',disjointness_complete=False,
                         supplied_set_disjointness=True,historical_coverage=coverage)
    with (out/'CONTROL_REVIEW_CANDIDATE.json').open('xb') as f:
        f.write(C.canonical_json_bytes(candidate));f.flush();os.fsync(f.fileno())
    write(out/'INPUT_MANIFEST.json',dict(files=inventory(out),seed_file_sha256=seed_sha256,
        exclusion_file_sha256=exclusion_sha256,direction_hashes=dh,source_pins=source_pins,
        decision_sha256=decision_sha256,scientific_accepted=False,production_accepted=False,
        **({'historical_coverage':coverage} if coverage is not None else {})))
    return candidate


def material_commitments(spec,controls,directions):
    """Recheck actual disjoint controls/directions, not a writer's status flag."""
    from operatormark.corpus import canonical as C
    material=read(member(spec['input_root'],'INPUT_MANIFEST.json'))
    exclusion=spec['exclusions']
    need(sha(exclusion['path'])==exclusion['sha256'],'disjointness exclusions drift')
    denied=read(exclusion['path'])
    coverage=None
    if spec.get('d7_policy') is not None:
        from .d7 import policy
        coverage=policy(spec['d7_policy'],decision_sha256=spec['decision_sha256'],exclusion=denied,exclusion_sha256=exclusion['sha256'])
        need(material.get('historical_coverage')==coverage and spec['controls'].get('d7_policy')==spec['d7_policy'],'D7 material binding differs')
        vr=spec['controls']['verification_receipt']
        need(sha(vr['path'])==vr['sha256'],'D7 independent verification drift')
        verified=read(vr['path'])
        need(verified.get('input_manifest_sha256')==sha(member(spec['input_root'],'INPUT_MANIFEST.json'))
            and verified.get('exclusion_sha256')==exclusion['sha256']
            and verified.get('source_pins')==spec['source_pins']
            and verified.get('controls')==4864 and verified.get('directions')==5632,'D7 independent material correspondence differs')
    need((coverage is not None or denied.get('complete_roles')==['qualification','confirmation','development'])
        and bool(denied.get('control_hashes')) and bool(denied.get('direction_hashes')),'complete external exclusions required')
    need(material['source_pins']==spec['source_pins'] and material['decision_sha256']==spec['decision_sha256']
        and material['exclusion_file_sha256']==exclusion['sha256'],'input producer binding differs')
    verify_files(spec['input_root'],material['files'])
    hashes=controls.control_hashes()
    need(not set(hashes.values())&set(denied['control_hashes']),'excluded control reused')
    dh=[]
    for split,count in zip(('training','validation','test'),COUNTS):
        for i in range(count):
            rho=controls.control(split,i)
            for k in range(1 if split=='training' else 2):dh.append(C.tensor_sha256(directions.direction(split,i,k,rho)))
    need(len(dh)==5632 and len(set(dh))==5632 and dh==material['direction_hashes']
        and not set(dh)&set(denied['direction_hashes']),'direction population/disjointness differs')

def build(spec,*,bind_only=False):
    """Actual live native writer/reader inputs, bound before generation."""
    import numpy as np
    from . import config as C,readback as B,layout as L,pins
    from .input_bindings import QualificationExpectations,QualificationExpectation,ScientificControlExpectations
    from .native import NativeSolver
    from operatormark_numerics_v2 import scope
    from operatormark.corpus import canonical
    verify_sources(spec['source_root'],spec['source_pins'])
    verify_files(spec['input_root'],spec['input_files'])
    reference=spec['runtime_reference']
    need(sha(reference['path'])==reference['sha256'] and read(reference['path'])['scope']==spec['runtime_scope'],'recorded runtime reference drift')
    need(scope.determinism_scope()==spec['runtime_scope'],'actual numerical runtime differs')
    numerical=pins.verify_dependencies()
    qspec=spec['qualification']
    qexp=QualificationExpectations(tuple(QualificationExpectation(**x) for x in qspec['entries']),
        Path(qspec['review_record']),qspec['review_record_sha256'],False)
    q=C.load_qualification(qspec['directories'],expected=qexp)
    ce=ScientificControlExpectations(**dict(spec['controls'],review_record=Path(spec['controls']['review_record']),
        controls=tuple(tuple(x) for x in spec['controls']['controls']),excluded_control_hashes=tuple(spec['controls']['excluded_control_hashes'])))
    arrays={s:np.load(member(spec['input_root'],s+'.npy'),allow_pickle=False) for s in L.SPLITS}
    need(all(a.shape==(n,8,8) and a.dtype==np.dtype('<f8') for a,n in zip(arrays.values(),COUNTS)),'control arrays shape/type')
    controls=scientific_controls(arrays,ce)
    directions=C.D6Directions(read(member(spec['input_root'],'direction_seed.json'))['seed'],synthetic=False)
    material_commitments(spec,controls,directions)
    choices=C.ScientificChoices('scientific_controls',C.D6Directions.RULE,C.REQUIRED_CHOICES,spec['decision_sha256'])
    cfg=C.GenerationConfig('scientific',128,C.RetentionDecision('S',spec['decision_sha256']),L.Layout(COUNTS,64,False),
        q.anomaly_policy(),qualification=q,choices=choices)
    solver=NativeSolver(128);cfg.require_production_inputs(controls,directions,solver,'stop')
    from .generator import configuration_record
    ident=dict(drop2=pins.drop2_identity(),drop1=numerical['drop1'],frozen=numerical['frozen'])
    configuration=configuration_record(cfg,controls,directions,solver,'stop',identity=ident,scope_sha256=scope.scope_sha256(scope.determinism_scope()))
    x=B.Expected(128,'S',COUNTS,64,'scientific',configuration_sha256=canonical.sha256_bytes(canonical.canonical_json_bytes(configuration)),
        qualification_dirs=q.source_dirs,qualification_expectations=qexp,scientific_expectations=ce,
        controls_sha256=ce.aggregate_control_content_sha256,direction_seed=directions._seed,
        lock_commitment_sha256=q.lock_commitment_sha256,retention_decision_sha256=spec['decision_sha256'],
        choices_reference_sha256=spec['decision_sha256'],producer_code_identity_sha256=canonical.sha256_bytes(canonical.canonical_json_bytes(ident)),
        producer_runtime_scope_sha256=scope.scope_sha256(scope.determinism_scope()))
    need(digest(spec['controls'])==spec['split_commitment'],'split commitment mismatch')
    # Separate pre-issued expectations cover every reader input; no writer manifest supplies it.
    expectation=digest(dict(configuration_sha256=x.configuration_sha256,controls=spec['controls'],qualification=qspec,
        source_pins=spec['source_pins'],runtime_scope=spec['runtime_scope'],direction_seed_file=spec['input_files']['direction_seed.json']))
    if bind_only:return expectation
    need(expectation==spec['expected_sha256'],'independent scientific expectation differs')
    return cfg,controls,directions,solver,x

def preflight(spec):
    """Metadata and bound-file checks, no numerical imports or material creation."""
    verify_sources(spec['source_root'],spec['source_pins'])
    if spec.get('d7_policy') is not None:
        from .d7 import policy
        policy(spec['d7_policy'],decision_sha256=spec['decision_sha256'],
               exclusion=read(spec['exclusions']['path']),exclusion_sha256=spec['exclusions']['sha256'])
        need(spec['controls'].get('d7_policy')==spec['d7_policy'],'D7 review/spec mismatch')
    need(spec['n_ref']==128 and spec['retention']=='S' and spec['counts']==list(COUNTS),'adopted S128 layout required')
    for k in ('overall_seconds','seal_seconds','cleanup_seconds','storage_bytes','minimum_free_bytes','start_free_bytes','memory_bytes'):
        need(type(spec[k]) is int and spec[k]>0,'positive finite resource limit: '+k)
    need(spec['start_free_bytes']>=spec['minimum_free_bytes']+2*spec['storage_bytes'],'capacity must include retained attempt plus one full return copy and headroom')
    need(spec['cleanup_seconds']<=30 and set(spec['work_seconds'])==set(STAGES),'phase resource scope')
    need(all(type(v) is int and v>0 for v in spec['work_seconds'].values()),'finite stage work windows')
    need(sum(spec['work_seconds'].values())+4*spec['cleanup_seconds']+spec['seal_seconds']<=spec['overall_seconds'],'whole phase envelopes do not fit')
    need(sha(spec['decision_file'])==spec['decision_sha256'],'adopted decision drift')
    verify_files(spec['input_root'],spec['input_files'])
    need({'training.npy','validation.npy','test.npy','direction_seed.json','INPUT_MANIFEST.json','CONTROL_REVIEW_CANDIDATE.json'}<=set(spec['input_files']),'complete controlled input population required')
    need(digest(spec['controls'])==spec['split_commitment'],'split commitment')
    need(isinstance(spec['runtime_scope'],dict) and bool(spec['runtime_scope']),'numerical runtime scope missing')
    return True

def stage(spec,root,name):
    from .generator import generate
    from .readback import validate_corpus
    cfg,controls,directions,solver,expected=build(spec);root=Path(root)
    if name=='generate':
        result=generate(cfg,controls,directions,root/'corpus',solver=solver,on_failure='stop',
            time_cap_s=spec['work_seconds'][name],mem_cap_bytes=spec['memory_bytes'])
        need(result['summary']['completion']=='complete','generation incomplete')
        value=dict(complete=True)
    else:
        need(name in ('readback1','readback2'),'unknown stage')
        before=inventory(root/'corpus')
        assessment=validate_corpus(root/'corpus',expected,numerics='full',p1='resolve')
        # Preserve the returned diagnostics even when the acceptance or mutation
        # assertion below rejects. This record is not an accepted readback and
        # must not substitute for RESULT.json or independent process closure.
        write(root/name/'ASSESSMENT.json',dict(
            schema='scientific_readback_assessment.v1',stage=name,
            spec_sha256=digest(spec),expectation_sha256=spec['expected_sha256'],
            producer_source_pins=spec['source_pins'],
            reader_sources=dict(scientific=sha(__file__),
                readback=sha(validate_corpus.__code__.co_filename)),
            accepted_readback=False,assessment=assessment))
        numerical_pass(assessment)
        need(before==inventory(root/'corpus'),'mutation during numerical readback')
        value=dict(assessment=assessment,corpus_inventory=before,expectation_sha256=spec['expected_sha256'])
    write(root/name/'RESULT.json',dict(stage=name,spec_sha256=digest(spec),**value))
