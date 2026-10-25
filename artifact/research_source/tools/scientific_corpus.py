"""Disabled-by-default Windows scientific corpus entry. No formal State route."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from operatormark_corpus_v2 import scientific as S

Q0_SHA='6493f6823ff2e357ac80b60ee1ef8117009262ec54bda189468224d71a3394c9'


def resource_observation(root,spec):
    """Practical observation, not a lossless provider byte-bound proof."""
    from operatormark_numerics_v2.resources import peak_rss_bytes
    used=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:S.member(root,(Path(directory)/name).relative_to(root).as_posix())
        for name in files:
            try:used+=(Path(directory)/name).stat().st_size
            except FileNotFoundError:pass  # atomic publication; check again next boundary
    peak=peak_rss_bytes();free=shutil.disk_usage(root).free
    S.need(peak is not None,'memory observation unavailable')
    return dict(retained_bytes=used,free_bytes=free,peak_worker_memory_bytes=peak,
        within_limits=used<=spec['storage_bytes'] and free>=spec['minimum_free_bytes'] and peak<=spec['memory_bytes'])


class ResourceWatch:
    """Immediate stage boundaries, ten-second practical checks; Q0 owns time/tree.

    Polling can overshoot a storage threshold between observations. It is not P
    custody or a hard provider allocation bound. Failures are never deleted.
    """
    def __init__(self,root,spec,stage):
        self.root=Path(root);self.spec=spec;self.stage=stage;self.stop=threading.Event();self.last=None
    def check(self):
        self.last=resource_observation(self.root,self.spec)
        S.need(self.last['within_limits'],'practical storage/free-space/memory limit')
    def loop(self):
        while not self.stop.wait(10):
            try:self.check()
            except BaseException as exc:
                try:S.write(self.root/self.stage/'RESOURCE_FAILURE.json',dict(error=repr(exc),observation=self.last,traceback=traceback.format_exc()))
                except BaseException:traceback.print_exc()
                os._exit(5)  # parent Job Object closes any owned descendants
    def __enter__(self):
        self.check();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join(timeout=2)
        S.need(not self.thread.is_alive(),'resource observer did not close')
        self.check();S.write(self.root/self.stage/'RESOURCES.json',self.last)

def supervisor(path):
    S.need(S.sha(path)==Q0_SHA,'Q0 supervisor identity mismatch')
    spec=importlib.util.spec_from_file_location('scientific_q0',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.run_with_timeout

def supervise_stages(root,spec,commands,run,*,synthetic=False):
    """One job per stage; no retry or resume. Injectable only for tiny fixtures."""
    root=Path(root);root.mkdir(exist_ok=False)
    start=time.monotonic();deadline=start+spec['overall_seconds']
    S.write(root/'SPEC.json',spec)
    try:
        S.need(shutil.disk_usage(root).free>=spec['start_free_bytes'],'insufficient measured start capacity')
        for name in S.STAGES:
            S.verify_sources(spec['source_root'],spec['source_pins'])
            work=spec['work_seconds'][name];cleanup=spec['cleanup_seconds']
            S.need(work>0 and 0<cleanup<=30 and time.monotonic()+work+cleanup+spec['seal_seconds']<=deadline,'full remaining envelope does not fit')
            S.need(resource_observation(root,spec)['within_limits'],'practical capacity exhausted')
            out=root/name;out.mkdir()
            print('START '+name,flush=True)
            r=run(commands[name],cwd=Path(spec['source_root']),env=dict(os.environ,**spec['environment']),
                timeout_s=work,cleanup_timeout_s=cleanup,out_dir=out)
            S.need(S.closed(r),'failed stage/timeout/owned closure: '+name)
            S.need(time.monotonic()<deadline,'overall deadline')
        out=root/'seal';out.mkdir()
        S.need(time.monotonic()+spec['seal_seconds']+spec['cleanup_seconds']<=deadline,'sealing envelope does not fit')
        r=run(commands['seal'],cwd=Path(spec['source_root']),env=dict(os.environ,**spec['environment']),
            timeout_s=spec['seal_seconds'],cleanup_timeout_s=spec['cleanup_seconds'],out_dir=out)
        S.need(S.closed(r),'failed sealing/timeout/owned closure')
        result=S.read(root/'SCIENTIFIC_RECEIPT.json')
        S.need(result['synthetic'] is synthetic,'sealing route substitution')
        S.need(time.monotonic()<=deadline,'seal overran overall deadline')
        S.write(root/'CLOSURE.json',dict(complete=True,seconds=time.monotonic()-start,
            stages_closed=list(S.STAGES)+['seal'],timed_out=False,scientific_receipt_sha256=S.sha(root/'SCIENTIFIC_RECEIPT.json'),
            sealing_invocation_sha256=S.sha(root/'seal/invocation.json')))
        return result
    except BaseException as exc:
        try:S.write(root/'FAILURE.json',dict(complete=False,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start))
        except BaseException:traceback.print_exc()
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--out',required=True);p.add_argument('--stage',choices=S.STAGES+('seal',))
    p.add_argument('--approved-execution-sha256');args=p.parse_args()
    S.need(S.sha(args.spec)==args.sha256,'spec changed');spec=S.read(args.spec)
    # No available package contains an enabled authority. Future exact approval
    # must bind this spec and candidate; toggling a notebook flag is insufficient.
    authority=spec.get('execution_approval')
    S.need(authority and args.approved_execution_sha256 and
        S.sha(authority)==args.approved_execution_sha256,'UNSTARTED: separate explicit execution approval required')
    grant=S.read(authority)
    S.need(grant.get('spec_sha256')==args.sha256 and grant.get('source_pins')==spec['source_pins']
        and grant.get('scope')=='one_scientific_S128_corpus' and grant.get('approved') is True
        and Path(grant.get('output_root','')).absolute()==Path(args.out).absolute(),'execution scope mismatch')
    S.need(os.name=='nt' and Path(sys.executable).resolve()==Path(spec['interpreter']).resolve(),'pinned Windows interpreter required')
    S.preflight(spec)
    sys.path[:0]=[spec['frozen'],spec['source_root'],str(Path(__file__).resolve().parent)]
    if args.stage:
        try:
            with ResourceWatch(args.out,spec,args.stage):
                if args.stage=='seal':S.seal(args.out,spec)
                else:S.stage(spec,args.out,args.stage)
        except BaseException as exc:
            try:S.write(Path(args.out)/args.stage/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),complete=False))
            finally:raise
        return
    run=supervisor(spec['supervisor'])
    commands={s:[sys.executable,'-B',str(Path(__file__).resolve()),'--spec',args.spec,'--sha256',args.sha256,
        '--out',args.out,'--stage',s,'--approved-execution-sha256',args.approved_execution_sha256] for s in S.STAGES+('seal',)}
    supervise_stages(args.out,spec,commands,run)

if __name__=='__main__':main()
