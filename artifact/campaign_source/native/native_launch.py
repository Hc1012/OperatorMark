"""One-shot Windows CPU supervision for the frozen native challengers.

No dependency installation, network access, retry, clock reset or numerical
kernel is implemented here. Windows ownership helpers are reused unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile

SCOPE = "native_F64_P256_optimisation_only"
QUOTA = 64 * 1024**2

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def read(path):
    return json.loads(Path(path).read_bytes())

def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())

def require(ok, message):
    if not ok:
        raise ValueError(message)

def verify_launcher(root, policy_sha):
    require(sha(root / "NATIVE_POLICY.json") == policy_sha, "Native policy bytes differ")
    p = read(root / "NATIVE_POLICY.json")
    require(p.get("scope") == SCOPE, "Native scope differs")
    require(p.get("stage_seconds") == 57600 and p.get("closure_seconds") == 60
            and p.get("finalization_seconds") == 120, "Fixed native envelope differs")
    require(set(p["launcher_files"]) == {"native_launch.py", "native_process_entry.py"},
            "Launcher membership differs")
    for name, fact in p["launcher_files"].items():
        path = root/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == fact["bytes"]
                and sha(path) == fact["sha256"], "Launcher bytes differ: " + name)
    from native_worker import verify_files
    source = verify_files(root, p)
    for name in ("experiments/tpfa32_copy_study/dev_process.py", "experiments/tpfa32_copy_study/core.py",
                 "operatormark_numerics_v2/scope.py"):
        require(name in p["source_pins"], "Required supervision/runtime source pin missing: " + name)
    return p, source

class Capture:
    """Retain a declared prefix; overflow stops work and remains explicit."""
    def __init__(self, pipe, path, quota=QUOTA):
        self.pipe, self.path, self.quota = pipe, Path(path), quota
        self.observed = self.retained = 0
        self.observed_hash, self.retained_hash = hashlib.sha256(), hashlib.sha256()
        self.eof = False; self.overflow = False; self.error = None
        self.thread = threading.Thread(target=self._drain, daemon=True)

    def _drain(self):
        try:
            with self.path.open("xb") as f:
                while True:
                    data = self.pipe.read(4096)
                    if not data:
                        self.eof = True; break
                    self.observed += len(data); self.observed_hash.update(data)
                    take = data[:max(0, self.quota-self.retained)]
                    if take:
                        f.write(take); f.flush()
                        self.retained += len(take); self.retained_hash.update(take)
                    if len(take) != len(data): self.overflow = True
                f.flush(); os.fsync(f.fileno())
        except BaseException as exc:
            self.error = repr(exc)

    def record(self):
        verified = self.path.is_file() and self.path.stat().st_size == self.retained and sha(self.path) == self.retained_hash.hexdigest()
        return dict(observed_bytes=self.observed, retained_bytes=self.retained,
                    observed_sha256=self.observed_hash.hexdigest(), retained_sha256=self.retained_hash.hexdigest(),
                    eof_observed=self.eof, overflow=self.overflow, quota_bytes=self.quota,
                    discarded_observed_bytes=self.observed-self.retained,
                    unobserved_bytes=None, capture_complete=self.eof and self.error is None and not self.overflow,
                    readback_verified=verified, error=self.error)

def archive_return(root, attempt, deadline):
    entries = {}
    for name in ("NATIVE_POLICY.json", "ACTIVATION.json", "NATIVE_RUNTIME_GATE.json",
                 "native_launch.py", "native_process_entry.py", "native_worker.py"):
        p = root/name
        if p.is_file(): entries[name] = p
    for folder in ("output", "supervision", "evidence_observed"):
        base = root/folder
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    require(not p.is_symlink(), "Return symlink refused")
                    entries[p.relative_to(root).as_posix()] = p
    require(sum(p.stat().st_size for p in entries.values()) <= 512*1024**2, "Return payload bound exceeded")
    manifest = {n:dict(bytes=p.stat().st_size, sha256=sha(p)) for n,p in entries.items()}
    target = root / ("NATIVE_F64_P256_RETURN_" + attempt + ".zip")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, p in entries.items():
            require(time.time() < deadline, "Fixed return deadline reached; partial archive retained")
            z.write(p, name)
        z.writestr("RETURN_MANIFEST.json", json.dumps(dict(files=manifest, formal_custody_assurance=False), sort_keys=True, indent=2)+"\n")
    with zipfile.ZipFile(target) as z:
        require(set(z.namelist()) == set(manifest)|{"RETURN_MANIFEST.json"}, "Return membership differs")
        for n, fact in manifest.items():
            require(time.time() < deadline, "Fixed return readback deadline reached")
            b=z.read(n)
            require(len(b)==fact["bytes"] and hashlib.sha256(b).hexdigest()==fact["sha256"], "Return readback differs")
    producer_complete=(root/"output/RESULT.json").is_file() and read(root/"output/RESULT.json").get("complete") is True
    supervision_complete=(root/"supervision/RESULT.json").is_file() and read(root/"supervision/RESULT.json").get("complete") is True
    saved = dict(archive=target.name, bytes=target.stat().st_size, sha256=sha(target),
                 saved=True, scientific_acceptance=False, formal_custody_assurance=False,
                 producer_numerical_complete=producer_complete,
                 native_complete=producer_complete and supervision_complete,
                 complete=producer_complete and supervision_complete)
    put(root/("NATIVE_F64_P256_SAVED_"+attempt+".json"), saved)
    print("RETURN SAVED: " + str(target), flush=True)
    print("RECEIPT: " + str(root/("NATIVE_F64_P256_SAVED_"+attempt+".json")), flush=True)
    return saved

def launch(root, policy_sha):
    root = Path(root).resolve()
    require(os.name == "nt", "This package uses the existing Windows native environment")
    require(not (root/"supervision").exists() and not (root/"ACTIVATION.json").exists()
            and not (root/"output").exists(), "This package was already attempted; no rerun or overwrite")
    p, source = verify_launcher(root, policy_sha)
    sys.path[:0] = [str(source/"experiments/tpfa32_copy_study"), str(source)]
    from dev_process import OwnedJob, process_identity, wait_dead, publish_handshake, available_ram
    require(shutil.disk_usage(root).free >= p["minimum_free_bytes"], "Insufficient free local space")
    require(available_ram()["available_bytes"] >= p["minimum_ram_bytes"], "Insufficient currently available RAM")
    sup = root/"supervision"; sup.mkdir()
    started=time.time(); child=None; owner=None; worker=None; launcher=None; captures=[]
    reason=None; errors=[]; activated=False; act=None; ready=None; rc=None
    preflight_end=started+300
    command=[sys.executable,"-B",str(root/"native_process_entry.py"),"--root",str(root),"--policy-sha",policy_sha,
             "--preflight-deadline",str(preflight_end)]
    put(sup/"LAUNCH.json",dict(command=command, policy_sha256=policy_sha, supervisor=process_identity(os.getpid()),
         preflight_deadline=preflight_end, numerical_clock_active=False, automatic_retries=0))
    try:
        owner=OwnedJob()
        child=subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), cwd=str(root))
        owner.assign(child); launcher=process_identity(child.pid)
        captures=[Capture(child.stdout,sup/"stdout.bin"), Capture(child.stderr,sup/"stderr.bin")]
        for c in captures:c.thread.start()
        hello_end=min(preflight_end,time.time()+30)
        while not (sup/"HELLO.json").exists():
            require(child.poll() is None and time.time()<hello_end, "Worker identity handshake unavailable")
            time.sleep(.05)
        hello=read(sup/"HELLO.json");worker=hello["worker"]
        expected_parent=os.getpid() if worker["pid"]==child.pid else child.pid
        require(hello["parent_pid"]==expected_parent and worker==process_identity(worker["pid"]), "Worker ancestry/identity differs")
        owner.assign_pid(worker["pid"])
        publish_handshake(sup/"BOUND.json",dict(worker=worker,launcher=launcher,policy_sha256=policy_sha,job_kill_on_close=True))
        print("Checking the frozen Windows runtime and native evidence. No numerical work has started.",flush=True)
        while not (sup/"READY.json").exists():
            require(time.time()<preflight_end, "Pre-activation checks exceeded their bound")
            require(child.poll() is None, "Pre-activation checks refused; read the saved error record")
            require(not any(c.overflow or c.error for c in captures), "Preflight stream capture refused")
            time.sleep(.1)
        ready=read(sup/"READY.json")
        require(ready.get("passed") is True and ready["policy_sha256"]==policy_sha
                and ready["worker"]==worker, "Native readiness binding differs")
        require(child.poll() is None, "Owned child ended before activation")
        now=time.time();attempt=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime(now))
        act=dict(attempt=attempt,start=now,deadline=now+p["stage_seconds"],
                 work_deadline=now+p["stage_seconds"]-p["closure_seconds"]-p["finalization_seconds"],
                 policy_sha256=policy_sha,scope=SCOPE,retries=0,previous_clocks_reused=False,
                 scientific_acceptance=False,started_by="user running the prepared Windows launcher")
        publish_handshake(root/"ACTIVATION.json",act);activated=True
        print("ACTIVATED: 128 fixed native targets (64 F64 + 64 P256). No Colab credits are used. Do not rerun.",flush=True)
        print("Work deadline: "+time.strftime("%Y-%m-%d %H:%M:%S %Z",time.localtime(act["work_deadline"])),flush=True)
        last_display=0
        while child.poll() is None:
            if time.time()>=act["work_deadline"]:reason="fixed work deadline";break
            if any(c.overflow or c.error for c in captures):reason="stream capture refused";break
            if shutil.disk_usage(root).free < 256*1024**2:reason="local free space reserve";break
            if time.time()-last_display>=30:
                last_display=time.time()
                progress=root/"output/progress.jsonl"
                if progress.is_file():
                    with progress.open("rb") as f:
                        f.seek(max(0,progress.stat().st_size-8192));data=f.read().splitlines()
                    if data:print(data[-1].decode("utf-8",errors="replace"),flush=True)
            try:child.wait(timeout=.5)
            except subprocess.TimeoutExpired:pass
    except BaseException as exc:
        reason=reason or type(exc).__name__+": "+str(exc)
        errors.append(traceback.format_exc())
    finally:
        closure_end=min(act["deadline"]-p["finalization_seconds"],time.time()+p["closure_seconds"]) if act else time.time()+60
        if owner:
            try:owner.close()
            except BaseException as exc:errors.append("Job close: "+repr(exc))
        if child:
            try:rc=child.wait(timeout=max(.1,closure_end-time.time()))
            except BaseException as exc:errors.append("Child wait: "+repr(exc))
        if worker:
            try:wait_dead(worker,max(0,closure_end-time.time()))
            except BaseException as exc:errors.append("Worker closure: "+repr(exc))
        for c in captures:c.thread.join(timeout=max(0,closure_end-time.time()))
    streams={name:c.record() for name,c in zip(("stdout","stderr"),captures)}
    ended=child is not None and child.poll() is not None
    worker_dead=False
    if worker is not None:
        try:worker_dead=process_identity(worker["pid"])!=worker
        except BaseException as exc:errors.append("Final worker identity: "+repr(exc))
    success=activated and ended and worker_dead and rc==0 and not errors and reason is None
    success=success and len(streams)==2 and all(s["capture_complete"] and s["readback_verified"] for s in streams.values())
    success=success and (root/"output/RESULT.json").is_file() and read(root/"output/RESULT.json").get("complete") is True
    result=dict(complete=bool(success),returncode=rc,activated=activated,reason=reason,errors=errors,
                child_reaped=ended,worker_ended=worker_dead,worker=worker,launcher=launcher,
                streams=streams,seconds=time.time()-started,automatic_retries=0,
                scientific_acceptance=False,formal_custody_assurance=False,
                supervisor_own_closure_not_self_certified=True)
    put(sup/"RESULT.json",result)
    attempt=act["attempt"] if act else "PREFLIGHT_"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime(started))
    try:archive_return(root,attempt,act["deadline"] if act else time.time()+120)
    except BaseException:
        put(sup/"SAVE_FAILURE.json",dict(traceback=traceback.format_exc(),evidence_preserved_in=str(root)))
        print("Return packaging stopped; existing evidence is preserved in "+str(root),flush=True)
        return 1
    print("NATIVE WORK COMPLETE" if success else "STOPPED. Evidence saved; do not rerun.",flush=True)
    return 0 if success else 1

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--policy-sha",required=True)
    a=parser.parse_args()
    try:return launch(a.root,a.policy_sha)
    except BaseException:
        traceback.print_exc();return 1

if __name__=="__main__":raise SystemExit(main())
