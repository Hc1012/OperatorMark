"""Purpose-limited conversion of an already accepted scientific corpus.

No solver, model, final-test access or acceptance issuer. The task targets are a
separate admitted input: P256 hashes in retention S cannot recreate their arrays.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import secrets
import sys


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def array(path, value):
    import numpy as np
    with Path(path).open('xb') as f:
        np.save(f, value, allow_pickle=False); f.flush(); os.fsync(f.fileno())


def prepare(admission, out, check):
    import numpy as np
    out = Path(out); out.mkdir(exist_ok=False)
    receipt, manifest = admission.read()  # reuse accepted readbacks, no solves
    save(out/'CORPUS_BINDING.json', dict(receipt_sha256=admission.receipt_sha256,
        closure_sha256=admission.closure_sha256, decision_sha256=admission.decision_sha256,
        manifest_sha256=receipt['manifest_sha256'], source_pins=receipt['source_pins'],
        limitations={k:receipt[k] for k in ('historical_coverage','d8_policy','physical_verdicts','admission_timing')}))
    roles=[f'{owner}/{role}' for owner in [f'O{i:02d}' for i in range(1,11)]+['B10']
           for role in ('initialization','shuffle')]
    seeds={name:secrets.randbits(63) for name in roles}
    if len(set(seeds.values())) != len(seeds):
        save(out/'STREAM_COLLISION.json', dict(roles=roles, retry=False))
        raise ValueError('stream collision; no redraw')
    save(out/'PRIVATE_TRAINING_STREAMS.json', dict(domain='final', values=seeds,
         recipe='independent secrets.randbits(63); D9 Engine initialization/shuffle', used=False))
    save(out/'STREAM_COMMITMENTS.json', dict(
        private_file_sha256=digest(out/'PRIVATE_TRAINING_STREAMS.json'),
        commitments={k:hashlib.sha256((k+'\n'+str(v)).encode()).hexdigest() for k,v in seeds.items()},
        committed_before_material_conversion=True, model_initializations=0, optimizer_updates=0))
    root=Path(receipt['original_root']); files={}; sources={}
    for split, shards in (('training',64),('validation',4)):
        parts={k:[] for k in ('x','v','y','dy')}; truths={k:[] for k in ('y','dy')}
        for shard in range(shards):
            check()
            prefix=f'corpus/corpus_staging/{split}/shard_{shard:04d}/'
            for key,family,width in [('x','rho',64),('v','dir',64),('y','u_f64',4096),('dy','du_f64',4096)]:
                name=prefix+family+'.float64.npy'; p=root/name; expected=manifest[name]
                raw=p.read_bytes()
                if dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) != expected:
                    raise ValueError('accepted source member drift: '+name)
                a=np.load(io.BytesIO(raw),allow_pickle=False)
                count=128 if split=='validation' and key in ('v','dy') else 64
                if a.dtype != np.dtype('<f8') or a.shape!=(count,1,int(width**.5),int(width**.5)) or not np.isfinite(a).all():
                    raise ValueError('source shape/dtype/nonfinite: '+name)
                a=a.reshape(count,width)
                if split=='validation' and key in truths:truths[key].append(a.copy())
                parts[key].append((a if key in ('x','v') else a/.01).astype(np.float32))
                sources[name]=expected
        for key, rows in parts.items():
            name=split+'_'+key+'.npy'; a=np.concatenate(rows); array(out/name,a)
            files[name]=dict(bytes=(out/name).stat().st_size,sha256=digest(out/name),shape=list(a.shape),dtype=str(a.dtype))
        if split=='validation':
            for key,name in [('y','validation_truth_f64.npy'),('dy','validation_tangent_truth_f64.npy')]:
                a=np.concatenate(truths[key]);array(out/name,a)
                files[name]=dict(bytes=(out/name).stat().st_size,sha256=digest(out/name),shape=list(a.shape),dtype=str(a.dtype))
        print('converted '+split,flush=True)
    # Before/after source checks cover only read members, never test.sealed.
    for name,expected in sources.items():
        check()
        if digest(root/name)!=expected['sha256'] or (root/name).stat().st_size!=expected['bytes']:
            raise ValueError('source changed during conversion: '+name)
    missing=['starts.npy']+[f'tasks/target_{i:02d}_{method}.npy' for i in range(16) for method in ('f64','p256')]
    value=dict(schema='first_stage_material_conversion.v1',files=files,corpus_members=sources,
        corpus_receipt_sha256=admission.receipt_sha256,streams_sha256=digest(out/'STREAM_COMMITMENTS.json'),
        python=sys.version,executable=sys.executable,numpy=np.__version__,
        source_sha256=digest(__file__),training_samples=4096,validation_samples=256,
        training_directions=4096,validation_directions=512,target_scale=.01,
        conversion='float64 division by .01 then float32; x/v unscaled; shard/local order',
        missing_monitor_files=missing,complete_monitor=False,scientific_accepted=False,
        final_test_opened=False,model_calls=0,solver_calls=0,optimizer_updates=0)
    save(out/'MATERIAL_CANDIDATE.json',value)
    return value


def verify(admission, out, check):
    """Independent reread of every converted value against its admitted source."""
    import numpy as np
    out=Path(out); candidate=json.loads((out/'MATERIAL_CANDIDATE.json').read_bytes())
    receipt,manifest=admission.read();root=Path(receipt['original_root']);rows=0
    if candidate['corpus_receipt_sha256']!=admission.receipt_sha256:raise ValueError('corpus substituted')
    for name,entry in candidate['files'].items():
        check()
        if digest(out/name)!=entry['sha256'] or (out/name).stat().st_size!=entry['bytes']:raise ValueError('derived drift')
    for split,count in [('training',64),('validation',4)]:
        for key,fam,width in [('x','rho',64),('v','dir',64),('y','u_f64',4096),('dy','du_f64',4096)]:
            dest=np.load(out/(split+'_'+key+'.npy'),allow_pickle=False);offset=0
            for i in range(count):
                check();name=f'corpus/corpus_staging/{split}/shard_{i:04d}/{fam}.float64.npy'
                raw=(root/name).read_bytes();expected=manifest[name]
                if candidate['corpus_members'].get(name)!=expected or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
                    raise ValueError('independent source binding differs')
                a=np.load(io.BytesIO(raw),allow_pickle=False).reshape(-1,width)
                want=(a if key in ('x','v') else np.divide(a,np.float64(.01))).astype(np.float32)
                if dest[offset:offset+len(a)].tobytes()!=want.tobytes():raise ValueError('conversion correspondence differs')
                if split=='validation' and key in ('y','dy'):
                    truth='validation_'+('truth_f64' if key=='y' else 'tangent_truth_f64')+'.npy'
                    if np.load(out/truth,allow_pickle=False)[offset:offset+len(a)].tobytes()!=a.tobytes():raise ValueError('truth conversion differs')
                offset+=len(a);rows+=len(a)
            if offset!=len(dest):raise ValueError('derived population differs')
    private=json.loads((out/'PRIVATE_TRAINING_STREAMS.json').read_bytes());c=json.loads((out/'STREAM_COMMITMENTS.json').read_bytes())
    if (digest(out/'PRIVATE_TRAINING_STREAMS.json')!=c['private_file_sha256'] or
        c['commitments']!={k:hashlib.sha256((k+'\n'+str(v)).encode()).hexdigest() for k,v in private['values'].items()}):
        raise ValueError('committed stream differs')
    result=dict(candidate_sha256=digest(out/'MATERIAL_CANDIDATE.json'),files_verified=len(candidate['files']),
        source_members_verified=len(candidate['corpus_members']),value_rows_verified=rows,committed_streams=22,
        bytewise_conversion_correspondence=True,complete_monitor=False,scientific_accepted=False,
        solver_calls=0,model_calls=0,optimizer_updates=0,final_test_opened=False)
    save(out/'INDEPENDENT_MATERIAL_READBACK.json',result)
    return result
