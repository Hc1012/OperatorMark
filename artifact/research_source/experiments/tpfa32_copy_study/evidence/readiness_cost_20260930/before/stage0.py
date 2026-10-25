"""Existing-evidence Stage 0 inspection: zero solver calls and no training."""
from __future__ import annotations
import json
import os
from pathlib import Path
from core import HERE, ROOT, PILOT, Budget, Output, read, digest, identity


def inspect(output):
    Budget.verification().check()
    out = Output(output, 'verification')
    evidence, comparisons, blockers = [], [], []
    def bind(path, expected=None):
        fact = identity(path)
        if expected is not None and fact['sha256'] != expected:
            raise ValueError('historical binding mismatch: '+str(path))
        evidence.append(fact)
        return read(path) if str(path).endswith('.json') else fact
    try:
        old, cont, cost = (PILOT/name for name in ('run_20260924','continuation_20260924','cost_20260924'))
        baseline = read(HERE/'evidence/baseline/identities.json')
        known = {r['path']: r['sha256'] for r in baseline['files']}
        # Existing local ownership/provenance anchored before any pickle load.
        for directory in (old,cont,cost):
            for path in directory.iterdir():
                if path.is_file() and path.suffix in ('.json','.jsonl','.pt'):
                    expected = known.get(str(path.resolve()))
                    if expected is None: raise ValueError('not in preserved owned baseline: '+str(path))
                    bind(path, expected)
        original = read(old/'identity.json'); ci = read(cont/'identity.json')
        pre = bind(cont/'precheck.json',ci['precheck_sha256'])
        bind(old/'identity.json',ci['prior_identity_sha256'])
        bind(old/'early_stop.json',ci['prior_early_stop_sha256'])
        source_paths = dict(rule_sha256=PILOT/'RULE.md', runner_sha256=PILOT/'run.py',
                            spatial_sha256=PILOT/'spatial.py', solver_sha256=ROOT/'operatormark/solvers/darcy_tpfa.py')
        for name,path in source_paths.items():
            bind(path,original[name])
            if ci['source_sha256'][name] != original[name] or pre['source_sha256'][name] != original[name]:
                raise ValueError('inconsistent original source provenance')
        bind(PILOT/'continuation.py',ci['continuation_script_sha256'])
        bind(PILOT/'CONTINUATION_RULE.md',ci['continuation_rule_sha256'])
        for name,sha in pre['artifact_sha256'].items(): bind(old/name,sha)
        co = read(cost/'identity.json'); ev = read(cost/'eval_identity.json')
        bind(PILOT/'COST_RULE.md',co['rule_sha256'])
        bind(PILOT/'cost_study.py',co['script_sha256'])
        bind(PILOT/'cost_evaluate_only.py',ev['evaluation_script_sha256'])
        bind(cont/'precheck.json',co['prior_precheck_sha256'])
        cr = bind(cont/'result.json',co['prior_result_sha256'])
        for name,key in (('identity.json','training_identity_sha256'),('raw.jsonl','training_raw_sha256'),
                         ('failure.json','training_failure_sha256')): bind(cost/name,ev[key])
        for directory,code in ((cont,0),(cost,1)):
            r = read(directory/'supervision.json')
            if r['return_code'] != code or r['timed_out']: raise ValueError('historical supervision differs')
        es = read(cost/'eval_supervision.json')
        if es['return_code'] != 0 or es['timed_out']: raise ValueError('historical evaluation incomplete')
        crow = [json.loads(line) for line in (cont/'raw.jsonl').read_text().splitlines()]
        rows = [json.loads(line) for line in (cost/'raw.jsonl').read_text().splitlines()]
        from numerics import runtime, legacy, controls, array_identity
        import torch
        import numpy as np
        current = runtime()
        mismatches = {key: dict(original=original[key], current=current[key])
                      for key in ('python','torch','numpy','scipy','threads') if original[key] != current[key]}
        if mismatches: blockers.append(dict(runtime_mismatch=mismatches))
        spatial, legacy_run = legacy()
        for owner in (0,1):
            original_endpoint = next(r for r in crow if r.get('type') == 'attack_complete'
                                     and r['owner']==owner and r['method']=='b3')
            reproduced = next(r for r in rows if r.get('type')=='attack_complete' and r['owner']==owner)
            if reproduced['attack_data_hashes'] != cr['data_hashes'][f'owner_{owner}_attack']:
                raise ValueError('attack observation identity mismatch')
            x = controls(original['seeds']['attack_controls'][owner],256)
            if array_identity(x)['sha256'] != reproduced['attack_data_hashes']['controls']:
                raise ValueError('original float64 control generation differs')
            left_path = cont/f'owner_{owner}_b3_attacked.pt'
            right_path = cost/f'owner_{owner}_b3_epoch_240.pt'
            bind(left_path,original_endpoint['checkpoint_sha256'])
            bind(right_path,ev['checkpoint_sha256'][str(owner)]['240'])
            # Local owned artifacts, now verified against their producer records.
            left = torch.load(left_path,map_location='cpu',weights_only=False)
            right = torch.load(right_path,map_location='cpu',weights_only=False)
            tensors = []
            if left.keys() != right.keys(): raise ValueError('state key set mismatch')
            for name in left:
                a,b = left[name],right[name]
                if torch.is_tensor(a):
                    equal = torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
                    if not equal or not torch.isfinite(a).all(): raise ValueError('tensor mismatch: '+name)
                    tensors.append(dict(name=name,shape=list(a.shape),dtype=str(a.dtype),
                                        sha256=array_identity(a.cpu().numpy())['sha256']))
                elif name != '_metadata':
                    if type(a)!=type(b) or a!=b: raise ValueError('non-tensor state mismatch: '+name)
            schema = spatial.SpatialFNO().state_dict()
            for name,value in schema.items():
                if torch.is_tensor(value) and (name not in left or value.shape != left[name].shape or value.dtype != left[name].dtype):
                    raise ValueError('current model schema differs: '+name)
            initial = old/f'owner_{owner}_b3.pt'
            initial_fact = bind(initial,pre['artifact_sha256'][initial.name])
            zero = cost/f'owner_{owner}_b3_epoch_0.pt'
            bind(zero,ev['checkpoint_sha256'][str(owner)]['0'])
            a = torch.load(initial,map_location='cpu',weights_only=False)
            b = torch.load(zero,map_location='cpu',weights_only=False)
            if a.keys()!=b.keys() or any(torch.is_tensor(v) and
                (v.dtype!=b[k].dtype or v.shape!=b[k].shape or not torch.equal(v,b[k])) for k,v in a.items()):
                raise ValueError('historical trajectory starting tensors differ')
            comparisons.append(dict(owner=owner,named_tensors=tensors,
                complete_state_key_set=list(left), same_tensor_values_shapes_dtypes=True,
                file_container_equal=digest(left_path)==digest(right_path),
                metadata_type_left=type(left.get('_metadata')).__name__,
                metadata_type_right=type(right.get('_metadata')).__name__,
                metadata_compared=False, initial_mark=initial_fact,
                inputs=reproduced['attack_data_hashes'], original_input_rows=array_identity(x),
                input_tensor_dtype='torch.float32', optimizer='fresh Adam lr=.002 defaults',
                shuffle_seed_binding='original owned seed table', epochs=240,updates=1920))
        decision = 'blocked/unresolved' if blockers else 'covered by existing evidence'
        result = dict(decision=decision, blockers=blockers, runtime=current, comparisons=comparisons,
          evidence=evidence, replay_started=False, solver_calls=0, model_updates=0,
          historical_backend_cpu_details='Not recorded completely; no cross-platform repeatability claim',
          coverage='Both owned legacy attack trajectories: identical original data hashes, initial tensors and 240-epoch endpoint named tensors/schema/dtypes. Adapter delegates to this unchanged attack; focused wrapper tests are separate.',
          limitations=['_metadata callable/container identity is not asserted; tensor/container identities distinguished',
                       'Existing evidence does not validate R-b, new calibration, continuous replacements or analysis',
                       'No new final controls, solver outputs or key arrays were generated'])
    except Exception as exc:
        result = dict(decision='blocked/unresolved',blockers=[dict(type=type(exc).__name__,message=str(exc))],
                      evidence=evidence,comparisons=comparisons,replay_started=False)
    out.finish(result)
    print(json.dumps(dict(output=str(out.path),decision=result['decision'],blockers=result['blockers'])))
    return result


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    inspect(parser.parse_args().output)
