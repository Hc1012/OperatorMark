"""Relocated metadata for an externally pinned, already accepted D8 corpus.

No original record is rewritten and no scientific acceptance is issued. This
view exposes only receipt/manifest metadata, never final arrays. The caller
selects the original receipt and closure hashes independently of the packet.
"""
from dataclasses import dataclass
from pathlib import Path
from operatormark_corpus_v2.scientific import read,sha,need,verify_files

@dataclass(frozen=True)
class PortableCorpus:
    root: Path
    receipt_sha256: str
    closure_sha256: str
    decision_sha256: str

    def read(self):
        root=Path(self.root)
        need(sha(root/'SCIENTIFIC_RECEIPT.json')==self.receipt_sha256,'original D8 receipt substitution')
        need(sha(root/'CLOSURE.json')==self.closure_sha256,'original D8 closure substitution')
        r=read(root/'SCIENTIFIC_RECEIPT.json');c=read(root/'CLOSURE.json')
        need(r.get('schema')=='operatormark.scientific_corpus.D8.v1' and r.get('complete') is True
            and r.get('scope')=='scientific_only' and r.get('production_accepted') is False
            and r.get('synthetic') is False and r.get('test_sealed') is True
            and r.get('decision_sha256')==self.decision_sha256,'relocated D8 scope differs')
        need(c.get('complete') is True and c.get('timed_out') is False
            and c.get('scientific_receipt_sha256')==self.receipt_sha256
            and c.get('stages_closed')==['readback1','readback2','seal'],'relocated closure incomplete')
        need(sha(root/'OUTPUT_MANIFEST.json')==r['manifest_sha256'],'relocated manifest drift')
        m=read(root/'OUTPUT_MANIFEST.json');verify_files(root,m['files'])
        from operatormark_corpus_v2.scientific import closed
        need(sha(root/'seal/invocation.json')==c['sealing_invocation_sha256']
             and closed(read(root/'seal/invocation.json')),'relocated seal closure')
        for role,field in [('producer','source_pins'),('reader','reader_pins')]:
            for name,h in r[field].items():
                p=(root/'sources'/role/name).resolve()
                need(p.is_relative_to((root/'sources'/role).resolve()) and sha(p)==h,'preserved '+role+' source drift')
        # Validate the unchanged original amendment bytes at new locations.
        from operatormark_corpus_v2.d7 import policy as d7
        from operatormark_corpus_v2.d8 import policy as d8
        b7=r['historical_coverage']['policy'];b8=r['d8_policy']
        p7=dict(path=str(root/'D7_ADOPTION.json'),sha256=b7['sha256'])
        p8=dict(path=str(root/'D8_ADOPTION.json'),sha256=b8['sha256'])
        checked=d7(p7,decision_sha256=self.decision_sha256);d8(p8)
        need({k:v for k,v in checked.items() if k!='policy'}==
             {k:v for k,v in r['historical_coverage'].items() if k!='policy'},'relocated D7 limitation')
        need(r.get('admission_timing')=='post_generation_after_76_relaxed_anomalies'
            and r.get('physical_qualification')=='preserved_negative'
            and len(r.get('physical_verdicts',[]))==2
            and all(v.get('verdict')=='not_accepted' for v in r['physical_verdicts']), 'physical negatives lost')
        for phase in ('readback1','readback2'):
            value=read(root/phase/'RESULT.json')
            need(closed(read(root/phase/'invocation.json')) and value['corpus_inventory']==m['corpus_files'],
                 'relocated readback population/closure')
        return r,{**m['files'],**{'corpus/'+n:f for n,f in m['corpus_files'].items()}}

    def validate_reporting(self,row,receipt):
        for key in ('historical_coverage','d8_policy','physical_verdicts','admission_timing'):
            need(row.get(key)==receipt[key],'relocated limitation missing/substituted: '+key)
