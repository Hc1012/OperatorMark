"""Q6 development utilities; no formal qualification or State entry point."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import time

LENGTHS=(5120,10240,20480,40960,81920)
THREAD_ENV=dict(PYTHONHASHSEED='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',BLIS_NUM_THREADS='1',GOTO_NUM_THREADS='1',
    OMP_THREAD_LIMIT='1',OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE',PYTHONDONTWRITEBYTECODE='1')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def event(root,**row):
    row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**row)
    with (Path(root)/'progress.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row,sort_keys=True),flush=True)

def check_time(deadline):
    if time.time()>=deadline:raise TimeoutError('Q6 immutable phase deadline')

def verify_files(base,files):
    for name,expected in files.items():
        if sha(Path(base)/name)!=expected:raise ValueError('bound file changed: '+name)

def plateau(scores,lengths=LENGTHS):
    if not scores or len(scores)>len(lengths) or any(not math.isfinite(x) or x<0 for x in scores):
        raise ValueError('nonnegative finite validation scores required')
    for i in range(1,len(scores)):
        if scores[i]>=.95*scores[i-1]:return dict(selected=lengths[i],plateaued=True,complete=True)
    return dict(selected=lengths[len(scores)-1] if len(scores)==len(lengths) else None,
                plateaued=False,complete=len(scores)==len(lengths))

def development_controls(seed,count):
    """The unchanged section-6 PCG64 sampling rule, explicitly real development.

    Drop1's generic sampler accepts only formal qualification/test seed labels;
    neither label describes Q6. This thin sampler is checked against that rule.
    """
    import numpy as np
    r=np.random.Generator(np.random.PCG64(seed));a=np.empty((count,8,8),dtype=np.float64)
    for i in range(count):
        if i%2==0:a[i]=r.uniform(0,1,(8,8))
        else:
            v=np.zeros(64);v[r.permutation(64)[:32]]=1;a[i]=v.reshape(8,8)
    return a

def development_starts(seed,count):
    import numpy as np
    return np.random.Generator(np.random.PCG64(seed)).uniform(.2,.8,(count,64)).reshape(count,8,8)

def array_save(path,a):
    import numpy as np
    with Path(path).open('xb') as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())

def true_field(rho,p1=False):
    """Pinned native routines, hard gates unchanged; no qualification ladder."""
    if p1:
        from operatormark_numerics_v2.p1_family import P1Family
        with P1Family(levels=(256,)).open(rho,256,compute_eta=False) as s:
            if not s.forward.gates_pass:raise RuntimeError('P256 hard numerical gate failure')
            return s.forward.field64.copy()
    from operatormark_numerics_v2.tpfa_family import TPFAFamily
    with TPFAFamily(levels=(128,)).open(rho,128) as s:
        if not s.forward.gates_pass:raise RuntimeError('T128 hard numerical gate failure')
        return s.forward.field64.copy()

def true_field_vjp(rho):
    from operatormark_numerics_v2.tpfa_family import TPFAFamily
    session=TPFAFamily(levels=(128,)).open(rho,128)
    if not session.forward.gates_pass:
        session.close();raise RuntimeError('exact-gradient T128 gate failure')
    def vjp(w):
        try:return session.vjp64(w)
        finally:session.close()
    return session.forward.field64.copy(),vjp
