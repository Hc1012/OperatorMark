"""Static extraction of already saved DEVELOPMENT diagnostic inputs."""
import json
import math
import pickletools
import shutil
import zipfile
from .records import read,write,sha

def add_inputs(bundle,old,returned,review_zip):
    import numpy as np
    original=read(old/'MANIFEST.json');status=read(returned/'STATUS.json');products={p['id']:p for p in status['products']}
    with zipfile.ZipFile(review_zip) as z:
        summary=json.loads(z.read('derived/CANDIDATE_AND_SHAM_SUMMARY.json'))['rows']
    target=bundle/'diagnostic';target.mkdir();(target/'models').mkdir();bindings={};rows=[]
    for i in range(3):
        name=f'embedding_{i}.npz';shutil.copyfile(old/name,target/name)
        if sha(target/name)!=original['files'][name]:raise ValueError('embedding input drift')
        bindings[name]=dict(source=name,source_sha256=original['files'][name],selection='unchanged original archive member')
        for method in ('B1','B3'):
            name=f'keys_{method}_{i}.npz'
            if sha(old/name)!=original['files'][name]:raise ValueError('key input drift')
            with np.load(old/name,allow_pickle=False) as z:
                np.savez(target/f'true_key_{method}_{i}.npz',v_true=z['v'][0].copy(),w_true=z['w_true'].copy())
            bindings[f'true_key_{method}_{i}.npz']=dict(source=name,source_sha256=original['files'][name],selection='v[0] and w_true unchanged; wrong keys excluded')
    for row in summary:
        name=row['id'];p=products[name];state=returned/'states'/f'{name}.pt';scalars=row['normalizers_and_last_loss']
        if sha(state)!=p['files'][f'states/{name}.pt'] or sha(state)!=scalars['state_sha256']:raise ValueError('saved normalizer state identity differs')
        # Recheck the scalar extraction without unpickling or loading tensor storage.
        with zipfile.ZipFile(state) as z:
            b=z.read(scalars['pickle_member']);ops=list(pickletools.genops(b))
        indices=[j for j,(op,arg,pos) in enumerate(ops) if arg=='scales']
        if len(indices)!=1:raise ValueError('ambiguous saved scales')
        vals=[]
        for op,arg,pos in ops[indices[0]+1:]:
            if op.name in ('MEMOIZE','BINPUT','LONG_BINPUT'):continue
            if op.name=='BINFLOAT':vals.append(arg);continue
            if op.name=='TUPLE2':break
            raise ValueError('unexpected saved normalizer encoding')
        expected=[scalars['forward_normalizer_saved'],scalars['jvp_normalizer_saved']]
        if vals!=expected:raise ValueError('saved scalar extraction mismatch')
        model=returned/'models'/f'{name}.pt';h=sha(model)
        if h!=p['files'][f'models/{name}.pt']:raise ValueError('model product binding differs')
        shutil.copyfile(model,target/'models'/model.name)
        for method in (('B1','B3') if name.startswith('S') else (row['kind'],)):
            i=int(row['owner'][1:])-1;normalizer=vals[0 if method=='B1' else 1]
            validation=row['detectors'][method] if name.startswith('S') else row['detector']
            raw=read(returned/'scores'/f'{name}_{method}_D{i+1}.json')
            if raw['scores'][0]!=validation['true_score'] or not math.isclose(raw['scale'],math.sqrt(normalizer),rel_tol=1e-14):raise ValueError('saved validation/normalizer correspondence differs')
            rows.append(dict(id=name,method=method,owner_index=i,model_sha256=h,normalizer=normalizer,
                target=row['target'],validation=validation,state_sha256=scalars['state_sha256'],
                saved_scalar_provenance='exact scales tuple statically rechecked in parent-bound state archive',
                validation_source_sha256=sha(returned/'scores'/f'{name}_{method}_D{i+1}.json')))
    if len(summary)!=21 or len(rows)!=24:raise ValueError('exact 18 marked + three shams / 24 scoring rows required')
    write(bundle/'DIAGNOSTIC_INPUTS.json',dict(rows=rows,input_bindings=bindings,review_summary_sha256=sha(review_zip),
        parent_status_sha256=sha(returned/'STATUS.json'),scope='true-key scores on fixed saved models; separate from unmarked outcome',
        limits=dict(total_seconds=600,work_seconds=570,cleanup_seconds=30,optimizer_updates=0),
        caveat='Validation summaries are arithmetic from saved unsealed score grids, not independent producer attestations.'))
