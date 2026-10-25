"""Owned Windows child: read-only preflight, then one fixed native run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from native_launch import read,put,sha,require,verify_launcher,SCOPE

def historical(root, policy):
    from native_worker import beneath
    expected={"p256_derivative":(1902,"7e4303c8d8599504e00eba8b77194b058e6ef71394fb59905b8a2a163a9c9f4e"),
              "p256_task":(894,"2c04633b1ed82481d71b380786a05e70f1d71207410cb2437c3c002b613c0cf0")}
    require(set(policy["historical_evidence"])==set(expected),"Historical numerical evidence membership differs")
    records={}
    for name,(size,digest) in expected.items():
        fact=policy["historical_evidence"][name];path=beneath(root,fact["path"])
        require(fact["bytes"]==size and fact["sha256"]==digest,"Historical evidence expectation differs")
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==size and sha(path)==digest,
                "Pinned historical native evidence missing or different: "+str(path))
        value=read(path)
        require(value.get("complete") is True,"Historical native check was incomplete: "+name)
        data=path.read_bytes();dest=root/"evidence_observed"/(name+".json");dest.parent.mkdir(exist_ok=True)
        with dest.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
        records[name]=dict(bytes=size,sha256=digest,original_path=str(path),complete=True)
    return records

def runtime(source, policy):
    sys.path[:0]=[str(source/"tools"),str(source)]
    from operatormark_numerics_v2.scope import THREAD_VARIABLES,determinism_scope
    require(all(os.environ.get(k)==v for k,v in THREAD_VARIABLES.items()),"Frozen thread environment differs")
    observed=determinism_scope()
    if observed != policy["expected_runtime_scope"]:
        changed=[k for k in sorted(set(observed)|set(policy["expected_runtime_scope"]))
                 if observed.get(k)!=policy["expected_runtime_scope"].get(k)]
        raise ValueError("Native runtime differs from frozen Windows expectations: "+", ".join(changed))
    return observed

def input_shapes(root, policy):
    """Read the three already hash-bound arrays before starting the clock."""
    import numpy as np
    from native_worker import beneath,INPUT_SHAPES
    observed={}
    for name,shape in INPUT_SHAPES.items():
        path=beneath(root,policy["inputs"][name]["path"])
        array=np.load(path,allow_pickle=False)
        require(array.shape==shape and array.dtype==np.float64 and np.isfinite(array).all(),
                "Native input array shape/dtype/finiteness differs: "+name)
        if name=="final_starts":
            require(np.all((array>=0)&(array<=1)),"Native starts outside the frozen box")
        else:
            require(all(np.sum(row*row,dtype=np.float64)>0 for row in array),"Native target with zero norm")
        observed[name]=dict(shape=list(array.shape),dtype=str(array.dtype),sha256=sha(path),bytes=path.stat().st_size)
    return observed

def child(root, policy_sha, preflight_deadline):
    root=Path(root).resolve();sup=root/"supervision"
    try:
        policy,source=verify_launcher(root,policy_sha)
        sys.path[:0]=[str(source/"experiments/tpfa32_copy_study"),str(source)]
        from dev_process import process_identity,publish_handshake
        worker=process_identity(os.getpid())
        publish_handshake(sup/"HELLO.json",dict(worker=worker,parent_pid=os.getppid(),policy_sha256=policy_sha))
        while not (sup/"BOUND.json").exists():
            require(time.time()<preflight_deadline,"Ownership handshake deadline reached")
            time.sleep(.05)
        bound=read(sup/"BOUND.json")
        require(bound["worker"]==worker and bound["policy_sha256"]==policy_sha,"Ownership binding differs")
        observations=runtime(source,policy)
        history=historical(root,policy)
        arrays=input_shapes(root,policy)
        put(sup/"PREFLIGHT_RUNTIME.json",observations)
        publish_handshake(sup/"READY.json",dict(passed=True,policy_sha256=policy_sha,worker=worker,
                          runtime_sha256=sha(sup/"PREFLIGHT_RUNTIME.json"),historical_evidence=history,arrays=arrays))
        while not (root/"ACTIVATION.json").exists():
            require(time.time()<preflight_deadline,"Activation handoff deadline reached")
            time.sleep(.05)
        act=read(root/"ACTIVATION.json")
        require(act["policy_sha256"]==policy_sha and act["scope"]==SCOPE,"Activation scope differs")
        # Observe again after activation so the receipt timestamp is literal,
        # rather than relabelling the earlier preflight observation as fresh.
        current=runtime(source,policy)
        put(sup/"ACTIVATED_RUNTIME.json",current)
        put(root/"NATIVE_RUNTIME_GATE.json",dict(passed=True,scope=SCOPE,policy_sha256=policy_sha,
            activation_sha256=sha(root/"ACTIVATION.json"),observed_at=time.time(),deadline=act["deadline"],
            runtime_sha256=sha(sup/"ACTIVATED_RUNTIME.json"),historical_evidence=history,
            scientific_acceptance=False,physical_qualification=False,formal_custody_assurance=False,
            general_runtime_acceptance=False))
        from native_worker import run
        run(root,root/"NATIVE_POLICY.json",act["work_deadline"],expected_policy_sha=policy_sha)
        final_runtime=runtime(source,policy)
        put(sup/"NATIVE_RUNTIME_FINAL.json",final_runtime)
        require(final_runtime==current,"Native runtime changed during the work")
        return 0
    except BaseException:
        failure=dict(traceback=traceback.format_exc(),activated=(root/"ACTIVATION.json").exists(),
                     native_execution_started=(root/"output/START.json").exists(),automatic_retry=False)
        if not (sup/"ENTRY_FAILURE.json").exists():put(sup/"ENTRY_FAILURE.json",failure)
        traceback.print_exc();return 1

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--policy-sha",required=True)
    p.add_argument("--preflight-deadline",type=float,required=True);a=p.parse_args()
    return child(a.root,a.policy_sha,a.preflight_deadline)

if __name__=="__main__":raise SystemExit(main())
