"""Explicit scientific admission alternative; the old State verifier is unchanged.

Externally selected expected bytes remain required. This is not a generic
AcceptedRecord replacement and issues no formal provenance or numerical result.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from operatormark.corpus.state_v2_prerequisites import require_accepted as formal
from operatormark_corpus_v2.scientific import CorpusAdmission,need,sha,corpus_bytes

KINDS={'main_study_final_inputs','main_study_final_assets','main_study_final_runtime',
       'main_study_final_owner_training_inputs','main_study_final_owner_training_runtime','main_study_final_opening'}

@dataclass(frozen=True)
class ScientificRecord:
    path: Path
    sha256: str
    expected: bytes
    corpus: CorpusAdmission

    def read(self):
        data=Path(self.path).read_bytes()
        need(data==self.expected and hashlib.sha256(data).hexdigest()==self.sha256,'scientific downstream record drift')
        row=json.loads(data);receipt,_=self.corpus.read()
        from .portable_corpus import PortableCorpus
        if type(self.corpus) is PortableCorpus:
            self.corpus.validate_reporting(row,receipt)
        else:
            from operatormark_corpus_v2.d7 import propagate
            propagate(row,receipt)
            from operatormark_corpus_v2.d8 import propagate as propagate_d8
            propagate_d8(row,receipt)
        need(row.get('schema')=='main_study_scientific_admission.v1' and row.get('scope')=='scientific_only'
            and row.get('synthetic') is False and row.get('scientific_accepted') is True
            and row.get('corpus_receipt_sha256')==self.corpus.receipt_sha256
            and row.get('decision_sha256')==receipt['decision_sha256']
            and row.get('split_commitment')==receipt['split_commitment'],'scientific data/runtime scope mismatch')
        need(row.get('kind') in KINDS and row.get('domain')=='final','scientific kind/domain unavailable')
        return row

def require(record,*,kind,bindings):
    if type(record) is not ScientificRecord:return formal(record,kind=kind,bindings=bindings)
    r=record.read();need(r['kind']==kind and r.get('bindings')==bindings,'scientific downstream bindings differ')
    return r


def same_corpus(*records):
    scientific=[r for r in records if type(r) is ScientificRecord]
    if not scientific:return
    need(len(scientific)==len(records),'scientific/formal input route mixing')
    need(len({(r.corpus.receipt_sha256,r.corpus.closure_sha256,r.corpus.decision_sha256)
              for r in scientific})==1,'scientific corpus substitution across consumers')

def runtime_ready(record):
    row=record.read()
    if type(record) is ScientificRecord:
        proof=row.get('numerical_compatibility')
        need(isinstance(proof,dict) and proof.get('verified') is True and proof.get('runtime_sha256')==row.get('bindings',{}).get('runtime_record_sha256')
            and proof.get('scope')=='scientific_runtime_numerics' and isinstance(proof.get('files'),dict) and bool(proof['files']),
            'actual scientific numerical compatibility required')
        for p,h in proof['files'].items():need(sha(p)==h,'compatibility evidence drift')
        need(row.get('complete_runtime_inventory') is True,'complete scientific runtime inventory required')
    else:
        need(row.get('native_readiness_accepted') is True and row.get('complete_runtime_inventory') is True,'final native/runtime acceptance missing')
        from .marking import identity
        identity(row.get('native_readiness_receipt_sha256'))
    return row

def verify_material_lineage(record,entry):
    """An externally reviewed derived material binds exact verified source members.

    This is metadata-only. Final arrays are not opened during admission.
    """
    if type(record) is not ScientificRecord:return
    _,manifest=record.corpus.read()
    if entry.get('origin')=='external':
        evidence=entry.get('provenance_files')
        need(isinstance(evidence,dict) and bool(evidence),'external scientific asset provenance missing')
        for p,h in evidence.items():need(sha(p)==h,'external scientific asset provenance drift')
        return
    lineage=entry.get('corpus_members')
    need(isinstance(lineage,dict) and bool(lineage),'scientific material lineage missing')
    purpose=entry.get('purpose');split=entry.get('corpus_split')
    allowed={'owner_training':('training',),'calibration_monitor':('validation',),
        'marking':('training','validation'),'qualification':('validation',),'attack':('training','validation'),'final':('test',)}
    need(split in allowed.get(purpose,()),'scientific split/purpose substitution')
    prefix='corpus/corpus_staging/'+('test.sealed' if split=='test' else split)+'/'
    need(all(n.startswith(prefix) and manifest.get(n)==f for n,f in lineage.items()),'scientific lineage differs from sealed manifest')

def training_arrays(corpus):
    """Actual corpus-to-owner material connection, with unchanged model units.

    Does not touch test.sealed. The externally reviewed derived-material catalog
    must bind the resulting arrays and their complete corpus-member lineage.
    """
    import io
    import numpy as np
    result={k:[] for k in ('x','v','y','dy')};lineage={}
    _,manifest=corpus.read()
    for shard in range(64):
        prefix=f'corpus/corpus_staging/training/shard_{shard:04d}/'
        for key,fam,width in (('x','rho',64),('v','dir',64),('y','u_f64',4096),('dy','du_f64',4096)):
            name=prefix+fam+'.float64.npy';b=corpus_bytes(corpus,name,split='training')
            a=np.load(io.BytesIO(b),allow_pickle=False)
            need(a.shape==(64,1,int(width**.5),int(width**.5)) and a.dtype==np.dtype('<f8') and np.isfinite(a).all(),'scientific training tensor shape')
            # Exact q6_windows material convention: scale in float64, then cast.
            values=a.reshape(64,width)
            result[key].append((values if key in ('x','v') else values/.01).astype('float32'));lineage[name]=manifest[name]
    return {k:np.concatenate(v) for k,v in result.items()},lineage
