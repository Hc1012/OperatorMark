"""Future authorized worker observation only. Never issues compatibility/admission."""
import argparse,hashlib,importlib,importlib.metadata as md,json,os,sys,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--payload',required=True);p.add_argument('--out',required=True)
a=p.parse_args();payload=Path(a.payload).resolve();out=Path(a.out).resolve()
if payload!=Path('/content/OperatorMark_Stage1/payload'):raise ValueError('absolute provenance root differs')
out.mkdir(parents=True,exist_ok=False)
def put(n,v):
    with (out/n).open('x',encoding='utf-8') as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
put('INTERPRETER.json',dict(python=sys.version,executable=sys.executable,argv=sys.argv,independent_acceptance=False))
try:
    source=payload/'code';sys.path[:0]=[str(source/'tools'),str(source)]
    cfg=json.loads((payload/'acceptance_r4/MAIN_STUDY_FINAL_CONFIGURATION_r4.json').read_bytes())
    for n,h in cfg['source_pins'].items():
        if sha(source/n)!=h:raise ValueError('packaged source drift: '+n)
    from q6_common import THREAD_ENV
    os.environ.update(THREAD_ENV)
    import q5_model
    q5_model.configure()
    from msfinal import materials as A
    observed=A.runtime_identity();put('GPU_RUNTIME_PAYLOAD.json',observed)
    inventory={d.metadata['Name']:d.version for d in md.distributions()}
    put('PACKAGE_INVENTORY.json',inventory)
    names=('q5_model','q6_common','q6_train','msdev.kernels','msdev.scoring',
        'operatormark_numerics_v2.tpfa_family','operatormark_numerics_v2.p1_family',
        'operatormark_numerics_v2.directions','operatormark_numerics_v2.task',
        'msfinal.models','msfinal.expanded','msfinal.evaluation','msfinal.contract_arms',
        'msfinal.prospective_contracts','msfinal.p256_adjoint','msfinal.p256_tasks',
        'msfinal.final_contracts','msfinal.resources')
    modules={}
    for name in names:
        m=importlib.import_module(name);path=Path(m.__file__).resolve();rel=path.relative_to(source).as_posix()
        if sha(path)!=cfg['source_pins'].get(rel):raise ValueError('actual module source differs: '+name)
        modules[name]=dict(path=str(path),source=rel,sha256=sha(path))
    put('MODULE_SOURCES.json',modules)
    # Metadata-only serializer premises, in the same owned observation job.
    import torch
    from torch.utils.serialization import config as serialization_config
    reference=json.loads((payload/'handoff/storage_basis/RUNTIME.json').read_bytes())
    torch_root=Path(torch.__file__).resolve().parent
    serializer_hashes={n:sha(torch_root/n) for n in reference['serializer_sources']}
    put('SERIALIZER_OBSERVATION.json',dict(runtime_sha256=sha(out/'GPU_RUNTIME_PAYLOAD.json'),
        source_hashes=serializer_hashes,reference_source_hashes=reference['serializer_sources'],
        source_hashes_match=serializer_hashes==reference['serializer_sources'],
        default_protocol=torch.serialization.DEFAULT_PROTOCOL,
        storage_alignment=serialization_config.save.storage_alignment,
        compute_crc32=serialization_config.save.compute_crc32,
        skip_data=torch.serialization._serialization_tls.skip_data,
        cuda_device_count=torch.cuda.device_count(),
        storage_registry=[dict(priority=p,module=getattr(t,'__module__',None),name=getattr(t,'__qualname__',None),
            deserializer_module=getattr(d,'__module__',None),deserializer_name=getattr(d,'__qualname__',None))
            for p,t,d in torch.serialization._package_registry],
        independently_accepted=False,requires_independent_premise_review=True))
    prior=json.loads((payload/'Q6_ENVIRONMENT.json').read_bytes())
    differences={k:dict(expected=prior.get(k),observed=observed.get(k)) for k in ('python','torch','cuda','cudnn','gpu','deterministic') if observed.get(k)!=prior.get(k)}
    for k,v in prior['packages'].items():
        if inventory.get(k)!=v:differences['package:'+k]=dict(expected=v,observed=inventory.get(k))
    put('APPLICABILITY_FOR_REVIEW.json',dict(runtime_sha256=sha(out/'GPU_RUNTIME_PAYLOAD.json'),
        differences=differences,independently_accepted=False,numerical_compatibility_verified=False,
        note='Observed metadata only. Reviewer must judge existing GPU evidence scope, source deltas and every runtime difference. D6 CPU four-solve evidence is separate. No replay performed.'))
    if observed['gpu'] is None or differences:raise ValueError('runtime differences: preserve observations for independent review; no fallback')
    print('Runtime observations saved. Independent numerical-compatibility/input/runtime acceptance is still required.')
except BaseException:
    put('FAILURE.json',dict(traceback=traceback.format_exc(),accepted=False))
    raise
