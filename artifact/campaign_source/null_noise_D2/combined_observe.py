"""Fresh candidate observations only; no numerical acceptance or model calls."""
from pathlib import Path
from runtime_values import plain_runtime
import argparse,hashlib,importlib,importlib.metadata as md,json,os,sys,time,traceback

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def need(ok,msg):
    if not ok:raise ValueError(msg)

def run(job,policy_sha,deadline):
    job=Path(job).resolve();payload=job.parent/'payload';out=job/'observations/runtime'
    need(payload==Path('/content/OperatorMark_Stage1/payload'),'Absolute source root differs')
    out.mkdir(parents=True,exist_ok=False)
    put(out/'INTERPRETER.json',dict(python=sys.version,executable=sys.executable,argv=sys.argv,independent_acceptance=False))
    def check():
        if time.time()>=deadline:raise TimeoutError('Candidate observation deadline reached')
    try:
        need(sha(job/'D2_COMBINED_POLICY.json')==policy_sha,'Frozen compatibility policy changed')
        p=read(job/'D2_COMBINED_POLICY.json');source=payload/'code'
        sys.path[:0]=[str(source/'tools'),str(source)]
        for n,h in p['source_pins'].items():
            check();need(sha(source/n)==h,'Pinned source differs: '+n)
        from q6_common import THREAD_ENV
        os.environ.update(THREAD_ENV)
        from msdev import pn_runtime_migration as P
        identity=P.observed_interpreter_identity(check)
        put(out/'CANDIDATE_INTERPRETER.json',identity)
        P.target_build(identity)
        need(identity['python']==p['expected_runtime']['python'],'Candidate Python target differs')
        host=read(job/'HOST_INTERPRETER_OBSERVED.json')
        need(identity['python']==host['python'] and identity['executable_sha256']==host['executable_sha256'],
             'Worker differs from observed host interpreter')
        import q5_model
        q5_model.configure()
        from msfinal import materials as A
        actual=plain_runtime(A.runtime_identity());put(out/'GPU_RUNTIME_PAYLOAD.json',actual)
        put(out/'PACKAGE_INVENTORY.json',{d.metadata['Name']:d.version for d in md.distributions()})
        modules={}
        for name,expected in p['expected_modules'].items():
            check();m=importlib.import_module(name);path=Path(m.__file__).resolve();rel=path.relative_to(source).as_posix()
            digest=sha(path);need(digest==expected==p['source_pins'].get(rel),'Imported source differs: '+name)
            modules[name]=dict(path=str(path),source=rel,sha256=digest)
        put(out/'MODULE_SOURCES.json',modules)
        import torch
        from torch.utils.serialization import config as serialization_config
        torch_root=Path(torch.__file__).resolve().parent
        hashes={n:sha(torch_root/n) for n in p['expected_serializer']['source_hashes']}
        serial=dict(runtime_sha256=sha(out/'GPU_RUNTIME_PAYLOAD.json'),source_hashes=hashes,
            default_protocol=torch.serialization.DEFAULT_PROTOCOL,storage_alignment=serialization_config.save.storage_alignment,
            compute_crc32=serialization_config.save.compute_crc32,skip_data=torch.serialization._serialization_tls.skip_data,
            cuda_device_count=torch.cuda.device_count(),
            storage_registry=[dict(priority=k,module=getattr(t,'__module__',None),name=getattr(t,'__qualname__',None),
                deserializer_module=getattr(d,'__module__',None),deserializer_name=getattr(d,'__qualname__',None))
                for k,t,d in torch.serialization._package_registry],
            independently_accepted=False)
        put(out/'SERIALIZER_OBSERVATION.json',serial)
        historical=read(job/'runtime.json')
        differences={k:dict(historical=historical.get(k),candidate=actual.get(k))
                     for k in set(historical)|set(actual) if historical.get(k)!=actual.get(k)}
        put(out/'APPLICABILITY_FOR_REVIEW.json',dict(historical_differences=differences,
            candidate_matches_frozen_expectation=actual==p['expected_runtime'],
            candidate_runtime_payload_sha256=sha(out/'GPU_RUNTIME_PAYLOAD.json'),
            numerical_compatibility_verified=False,independently_accepted=False,
            scope='D2_diagnostic_candidate_observations_only'))
        need(actual==p['expected_runtime'],'Observed candidate differs from independently frozen runtime fields')
        need(set(differences)=={'python'},'Candidate changed more than the prescribed Python field')
        need(sha(out/'GPU_RUNTIME_PAYLOAD.json')==p['candidate_runtime_payload_sha256'],'Candidate payload identity differs')
        for k,v in p['expected_serializer'].items():need(serial.get(k)==v,'Serializer premise differs: '+k)
        check()
        print('Frozen candidate metadata matched. Numerical compatibility remains unverified.',flush=True)
    except BaseException:
        put(out/'FAILURE.json',dict(traceback=traceback.format_exc(),accepted=False))
        raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--job',required=True,type=Path);a.add_argument('--policy-sha',required=True);a.add_argument('--deadline',required=True,type=float)
    v=a.parse_args();run(v.job,v.policy_sha,v.deadline)
