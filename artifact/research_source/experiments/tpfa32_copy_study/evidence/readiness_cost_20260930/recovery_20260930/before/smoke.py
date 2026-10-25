"""One tiny real integration fixture. Not owner qualification or experimental evidence."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from core import Budget, Output, Seeds, source_hashes, write
from numerics import (Oracle, Splits, runtime, legacy, generate, fresh, forward,
                      controls, keyed_truth, evaluate, task_data)
from training import Trajectory


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();budget=Budget.verification();out=Output(args.output,'verification')
    initial=dict(budget.counts);oracle=None
    try:
        metadata=runtime();spatial,old=legacy()
        seeds=Seeds(out.path/'FIXTURE_SEEDS.json','integration_fixture',fixture=94001)
        oracle=Oracle(spatial.solve,budget,real=True,log=out.log);splits=Splits()
        data=generate(oracle,seeds.get('controls'),seeds.get('directions'),2,1,'tiny',splits)
        model=fresh(seeds.get('initialization'))
        clean=Trajectory(model,data,seeds.get('shuffle'),budget,recipe='derivative',batch=16,real=True)
        clean.extend(1)
        key=old.b3_key(seeds.get('fixture_key',(0,10000)))
        marked=copy.deepcopy(model)
        mark=Trajectory(marked,data,seeds.get('mark_shuffle'),budget,recipe='mark',batch=16,
                        lr=.0003,reference=model,key=key,real=True)
        mark.extend(1)
        shared=forward(oracle,controls(seeds.get('attack_controls'),2),'tiny_attack',splits)
        copied=Trajectory(copy.deepcopy(marked),shared,seeds.get('shared_attack_shuffle'),budget,real=True)
        replacement=Trajectory(fresh(seeds.get('replacement')),shared,
                               seeds.get('shared_attack_shuffle'),budget,real=True)
        copied.extend(1);replacement.extend(1)
        rb=Trajectory(fresh(seeds.get('Rb_initialization')),data,seeds.get('Rb_shuffle'),budget,
                      recipe='derivative',batch=16,real=True);rb.extend(1)
        truth=keyed_truth(oracle,data.x64,key,'tiny_key',splits)
        tasks=task_data(oracle,seeds,1,'tiny_tasks',splits)
        before={k:v.detach().clone() for k,v in marked.state_dict().items() if torch.is_tensor(v)}
        metrics,raw=evaluate(marked,data,tasks,oracle,budget,'tiny_evaluation',key,truth)
        out.log(dict(type='evaluation_raw',raw=raw))
        assert all(torch.equal(v,marked.state_dict()[k]) for k,v in before.items())
        assert all(p.requires_grad for p in marked.parameters())
        mark.extend(2)
        score=old.b3_score(marked,model,data.tensors()[0],key,mark.score_scale)
        assert np.isfinite(score) and all(np.isfinite(v) for v in metrics.values())
        for name,trainer in [('clean',clean),('mark',mark),('copy',copied),('Ra',replacement),('Rb',rb)]:
            out.log(dict(type='fixture_training',arm=name,costs=trainer.record(),losses=trainer.losses))
        out.finish(dict(passed=True,fixture=True,scientific_qualification=False,runtime=metadata,
             sources=source_hashes(),counts={k:budget.counts[k]-initial[k] for k in initial},
             oracle=oracle.record(),metrics=metrics,key_score=score,
             paths=['normalized forward/JVP','keyed embedding loss','copied/fresh forward update',
                    'fresh derivative update','keyed truth/evaluation','complete disposable 120-step inverse',
                    'post-evaluation marking update'],legacy_full_attack_executed=False))
    except BaseException as exc:
        out.finish(dict(type=type(exc).__name__,message=str(exc),
             counts={k:budget.counts[k]-initial[k] for k in initial},
             oracle=None if oracle is None else oracle.record()),failed=True)
        raise


if __name__=='__main__':main()
