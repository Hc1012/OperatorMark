"""Worker only; waits for independent owned-job binding before importing numerics."""
import argparse
import os
import time
from core import read,write,identity
from dev_process import process_identity,publish_handshake


def main():
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--attempt',type=int,choices=(0,1),required=True);p.add_argument('--fixture',action='store_true')
    args=p.parse_args();root=args.output;attempt=root/('attempt_'+str(args.attempt))
    publish_handshake(attempt/'HELLO.json',dict(worker=process_identity(os.getpid()),parent_pid=os.getppid()))
    for _ in range(100):
        if (attempt/'BOUND.json').exists():break
        time.sleep(.05)
    bound=read(attempt/'BOUND.json')
    if bound['worker']!=process_identity(os.getpid()) or bound['supervisor']!=process_identity(bound['supervisor']['pid']):
        raise RuntimeError('independent supervisor/process binding mismatch')
    authority=read(root/'AUTHORITY.json')
    if args.fixture:
        if os.environ.get('TPFA_CATEGORY')!='final_study_verification':raise PermissionError('fixture not authorized')
        from final_fixture import TinyFinal
        campaign=TinyFinal(root,authority,args.attempt)
    else:
        from final_supervise import validate_run
        validate_run(root)
        from numerics import runtime
        actual=runtime()
        if actual!=authority['runtime']:raise ValueError('pinned CPU runtime mismatch')
        write(attempt/'RUNTIME.json',actual)
        from final_campaign import FinalCampaign
        campaign=FinalCampaign(root,authority,args.attempt)
    publish_handshake(attempt/'STARTUP.json',dict(worker=bound['worker'],runtime_verified=True,fixture=args.fixture,
                                    counts=campaign.budget.counts,seed_identity=identity(root/'PRIVATE_SEEDS.json')))
    campaign.execute()


if __name__=='__main__':main()
