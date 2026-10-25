"""Development-only byte-bound bridge to the actual shared evaluator body.

No AcceptedRecord, final admission, key draw or numerical work on import.
"""
from dataclasses import dataclass
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
import hashlib,io,json,math
from msdev.records import read,write,sha
from . import evaluation as E

def inventory(spec,phase):
    E.validate_specification(spec,phase)
    arrays={phase+'_'+k:dict(shape=list(v[0]),dtype=v[1],purpose=phase) for k,v in E.required_arrays(spec).items()}
    for prefix in (phase+'_',''):
        arrays[prefix+'key_v']=dict(shape=[4096,64],dtype='float32',purpose=phase if prefix else 'marking')
        arrays[prefix+'key_coeff']=dict(shape=[4096,16],dtype='float64',purpose=phase if prefix else 'marking')
    arrays['public_basis']=dict(shape=[16,4096],dtype='float64',purpose='marking')
    # B2 is not part of evaluate_model but belongs to the complete verification.
    arrays[phase+'_detector_v']=dict(shape=[spec['detector_controls'],2,64],dtype='float32',purpose=phase)
    return arrays

class Access:
    @staticmethod
    def catalog(c):return c.record
    @staticmethod
    def bytes(c,name,purpose):
        r=c.record['entries'][name]
        if r['purpose']!=purpose:raise ValueError('development asset purpose mismatch')
        path=(c.root/r['path']).resolve()
        if not path.is_relative_to(c.root):raise ValueError('development path escape')
        data=path.read_bytes()
        if len(data)!=r['bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('development payload drift: '+name)
        return data
    @staticmethod
    def array(c,name,purpose,shape,dtype):
        import numpy as np
        x=np.load(io.BytesIO(Access.bytes(c,name,purpose)),allow_pickle=False)
        if not isinstance(x,np.ndarray) or x.shape!=tuple(shape) or str(x.dtype)!=dtype or not np.isfinite(x).all():raise ValueError('development array contract: '+name)
        return x
    @staticmethod
    def weights(c,name,purpose):
        import torch
        x=torch.load(io.BytesIO(Access.bytes(c,name,purpose)),map_location='cpu',weights_only=True)
        if not isinstance(x,dict) or not x or any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in x.values()):raise ValueError('development tensor state')
        return x

def context(root,catalog_sha256,phase):
    root=Path(root).resolve();p=root/'DEVELOPMENT_CATALOG.json'
    if sha(p)!=catalog_sha256:raise ValueError('development catalog identity')
    r=read(p);spec=r['specifications'][phase];requirements=inventory(spec,phase)
    if r.get('domain')!='development' or r.get('verification_only') is not True:raise ValueError('development-only scope required')
    if not set(requirements)|{'clean_model'}<=set(r['entries']):raise ValueError('complete development evaluator inventory missing')
    for name,entry in r['entries'].items():
        if entry.get('domain')!='development':raise ValueError('non-development entry refused')
        rel=PurePosixPath(entry['path'])
        if rel.is_absolute() or '..' in rel.parts or '\\' in entry['path'] or ':' in entry['path']:raise ValueError('unsafe development asset path')
        if name in requirements and any(entry.get(k)!=v for k,v in requirements[name].items()):raise ValueError('development metadata contract: '+name)
        Access.bytes(SimpleNamespace(root=root,record=r),name,entry['purpose'])
    if len(r.get('normalizers',[]))!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in r['normalizers']):raise ValueError('positive bound normalizers required')
    from . import marking
    for value in (r['runtime_sha256'],r['source_pins']['tools/msdev/scoring.py']):marking.identity(value)
    for path,name in ((E.__file__,'tools/msfinal/evaluation.py'),(__file__,'tools/msfinal/development_evaluator.py')):
        if sha(path)!=r['source_pins'][name]:raise ValueError('shared evaluator/bridge source drift')
    from importlib.util import find_spec
    if sha(find_spec('msdev.scoring').origin)!=r['source_pins']['tools/msdev/scoring.py']:raise ValueError('scoring source drift')
    c=SimpleNamespace(root=root,record=r,configuration=dict(source_pins=r['source_pins']),
        calibration=dict(wrong_key_count=4095),inputs=dict(normalizers=r['normalizers']),
        bindings=dict(domain='development',catalog=catalog_sha256,independent_acceptance=False),
        prerequisites=SimpleNamespace(runtime=SimpleNamespace(sha256=r['runtime_sha256']),
            assets=SimpleNamespace(sha256=catalog_sha256),clean_model_sha256=r['entries']['clean_model']['sha256']))
    return c,spec

def run(model,root,catalog_sha256,out,check,*,phase,model_sha256,runtime_check,projection_store=None):
    c,spec=context(root,catalog_sha256,phase);check()
    # Caller uses the already verified historical-runtime inspector/owned route.
    # Matching string is insufficient: the callback must complete all checks and
    # bind the exact immutable runtime record before this kernel dispatch.
    if runtime_check()!=c.record['runtime_sha256']:raise ValueError('development runtime receipt mismatch')
    return E._evaluate_model(model,c,out,check,phase=phase,model_sha256=model_sha256,
        wrapper_steps=None,projection_store=projection_store,access=Access,spec=spec)
