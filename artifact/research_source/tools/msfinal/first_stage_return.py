"""Stream-read a first-stage whole-worker OUTPUT.zip; never extract or accept it.

Expected identities must be selected independently from the preserved remote
commits/approved candidate. They must not be inferred from the received ZIP.
"""
from pathlib import Path
import argparse,json,hashlib,time,zipfile,sys

if __package__ in (None,''):
    sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2])]
from msdev.records import read,write,sha
from msfinal.phase_storage import verify_archive
from msfinal.session import closure_ok
from msfinal.graph import identity

REQUIRED={'OUTPUT.zip','OUTPUT_MANIFEST.json','OUTPUT_INVENTORY.json','CLOSURE.json',
          'DURABLE_STAGE.json','OPERATION_CLOSED.json','OUTER_RESULT.json'}

def verify(root,expectations,check):
    root=Path(root)
    if expectations.get('format')!='msfinal.whole_worker.stage1.v1' or set(expectations.get('files',{}))!=REQUIRED:
        raise ValueError('independently selected whole-worker return identities required')
    for name,want in expectations['files'].items():
        check();p=root/name
        if p.is_symlink() or p.stat().st_size!=want['bytes'] or sha(p,check)!=want['sha256']:
            raise ValueError('return identity differs: '+name)
    m=read(root/'OUTPUT_MANIFEST.json');i=read(root/'OUTPUT_INVENTORY.json')
    if any(m.get(k)!=v for k,v in i.items()) or m.get('phase_capsules')!={}:
        raise ValueError('initial whole-worker inventory/capsule differs')
    if (m.get('archive_sha256')!=expectations['files']['OUTPUT.zip']['sha256']
            or m.get('archive_bytes')!=expectations['files']['OUTPUT.zip']['bytes']
            or any(m.get(k) is not True for k in ('exact_membership','full_member_readback','unchanged_source','owned_closure'))):
        raise ValueError('archive declaration incomplete')
    verify_archive(root/'OUTPUT.zip',m['files'],check)
    closed=read(root/'CLOSURE.json');operation=read(root/'OPERATION_CLOSED.json')
    outer=read(root/'OUTER_RESULT.json');durable=read(root/'DURABLE_STAGE.json')
    if (not closure_ok(closed) or closed.get('returncode')!=5 or closed.get('timed_out') is not False
            or closed.get('product_readback') is not True or not closure_ok(outer)
            or outer.get('returncode')!=0 or outer.get('timed_out') is not False
            or operation.get('outer_closure')!=outer or operation.get('durable_success') is not True
            or operation.get('durable_stage_sha256')!=sha(root/'DURABLE_STAGE.json')
            or durable.get('closure_sha256')!=sha(root/'CLOSURE.json')
            or m.get('closure_sha256')!=sha(root/'CLOSURE.json')):
        raise ValueError('independent closure chain differs')
    session=expectations['session_sha256'];binding=expectations['worker_binding'];phase=expectations['phase']
    if any(r.get('session_sha256')!=session for r in (closed,operation,durable)):
        raise ValueError('session substituted')
    with zipfile.ZipFile(root/'OUTPUT.zip') as z:
        result=json.loads(z.read('RESULT.json'));product_name='products/'+identity(phase)+'.json'
        if (result.get('status')!='PAUSED_SESSION' or result.get('products')!=[phase]
                or result.get('session_sha256')!=session or result.get('binding')!=binding
                or result.get('product_receipts',{}).get(phase)!=m['files'][product_name]['sha256']
                or m['worker_result_sha256']!=m['files']['RESULT.json']['sha256']
                or closed['worker_result_sha256']!=m['worker_result_sha256']
                or durable.get('worker_result_sha256')!=m['worker_result_sha256']):
            raise ValueError('stage/candidate/product correspondence differs')
        product=json.loads(z.read(product_name));prefix='work/'+phase+'/'
        if (product.get('name')!=phase or product.get('binding')!=binding
                or product.get('files')!={n[len(prefix):]:v for n,v in m['files'].items() if n.startswith(prefix)}):
            raise ValueError('exact phase membership differs')
    check()
    for name,want in expectations['files'].items():
        if sha(root/name,check)!=want['sha256']:raise ValueError('return mutation')
    return dict(verified=True,format=expectations['format'],archive_sha256=m['archive_sha256'],
        manifest_sha256=sha(root/'OUTPUT_MANIFEST.json'),members=len(m['files']),phase=phase,
        full_archive_member_readback=True,closure_verified=True,session_sha256=session,worker_binding=binding,
        scientific_acceptance=False,owner_qualification=False,native_scoring='pending',expanded_copy_created=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--return-root',required=True);p.add_argument('--expectations',required=True)
    p.add_argument('--expectations-sha256',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);deadline=time.monotonic()+150
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('return readback envelope')
    try:
        if sha(a.expectations)!=a.expectations_sha256:raise ValueError('independent expectations drift')
        r=verify(a.return_root,read(a.expectations),check)
        r['expectations_sha256']=a.expectations_sha256;write(out/'RETURN_VERIFICATION.json',r,durable=True)
    except BaseException as exc:
        import traceback
        write(out/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),durable=True)
        raise
    print(json.dumps(r))

if __name__=='__main__':main()
