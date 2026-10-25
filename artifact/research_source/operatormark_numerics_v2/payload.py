"""Deterministic payload versus variable run record.

Payload (deterministic; compared for L1 equality):
  payload/<stage>.header.json      schema, code identity, scope hash, inputs, settings
  payload/<stage>.instances.jsonl  one canonical record per instance, canonical order
  payload/arrays/*.npy             64-grid images and [8,8] arrays (canonical NumPy v2.0)
  payload/<stage>.footer.json      completion, counts, file hashes, derived results, payload_sha256

Run record (variable; never read by payload validation):
  run/<stage>.progress.jsonl       per-instance phase timings and memory
  run/<stage>.run.json             timestamps, wall time, peak memory, caps, stop reason, process facts

Payload records may not contain Python floats: scientific scalars are float.hex
strings, so encoding is exact and platform-independent.  Nothing derived from
a clock, a path, a pid or a hostname may enter the payload (test H8).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from operatormark.corpus import canonical

PAYLOAD_SCHEMA = "operatormark_numerics_v2.payload.v1"
RUN_SCHEMA = "operatormark_numerics_v2.run_record.v1"
PAYLOAD_DOMAIN = "operatormark_numerics_v2/payload/v1"
_FORBIDDEN_KEYS = {"time", "timestamp", "utc", "seconds", "duration", "pid", "host", "hostname", "path", "cwd",
                   "tmp", "temp", "rss", "memory"}


class PayloadError(ValueError):
    pass


def hexf(value: float | None) -> str | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return "inf" if value > 0 else ("-inf" if value < 0 else "nan")
    return value.hex()


def check_payload_value(value: Any, where: str = "record") -> None:
    """Reject floats, non-string keys and clock/path/process-looking keys anywhere in a payload record."""

    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return
    if isinstance(value, float):
        raise PayloadError(f"{where}: Python float not allowed in payload (use float.hex)")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PayloadError(f"{where}: non-string key")
            lowered = key.lower()
            if any(lowered == bad or lowered.endswith("_" + bad) or lowered.startswith(bad + "_") for bad in _FORBIDDEN_KEYS):
                raise PayloadError(f"{where}.{key}: key looks like run metadata")
            check_payload_value(item, f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            check_payload_value(item, f"{where}[{index}]")
        return
    raise PayloadError(f"{where}: unsupported type {type(value).__name__}")


def _status_sha(complete: bool, ids) -> str:
    return canonical.sha256_bytes(canonical.canonical_json_bytes(
        {"completion": "complete" if complete else "incomplete", "finished_instance_ids": list(ids)}))


def _fsync_append(path: Path, data: bytes) -> None:
    with path.open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class PayloadWriter:
    def __init__(self, out_dir: Path, stage: str, header: Mapping[str, Any]):
        self.root = Path(out_dir)
        if self.root.exists():
            raise PayloadError("output directory must not exist yet")
        self.stage = stage
        self.payload_dir = self.root / "payload"
        self.arrays_dir = self.payload_dir / "arrays"
        self.arrays_dir.mkdir(parents=True)
        body = dict(header)
        body.update(schema=PAYLOAD_SCHEMA, stage=stage)
        check_payload_value(body, "header")
        record = canonical.with_self_hash(body, "header_self_sha256")
        self.header_bytes = canonical.write_canonical_json_exclusive(self.payload_dir / f"{stage}.header.json", record)
        self.instances_path = self.payload_dir / f"{stage}.instances.jsonl"
        self.instances_path.touch(exist_ok=False)
        self.ids: list[str] = []
        self.array_files: dict[str, str] = {}
        self.finalized = False

    def add_instance(self, instance_id: str, record: Mapping[str, Any], arrays: Mapping[str, Any] | None = None) -> dict:
        if self.finalized:
            raise PayloadError("payload already finalized")
        if self.ids and instance_id <= self.ids[-1]:
            raise PayloadError("instances must be added in strictly increasing canonical order")
        body = dict(record)
        body["instance_id"] = instance_id
        files = {}
        for name, array in sorted((arrays or {}).items()):
            filename = f"{instance_id.replace('/', '__')}__{name}.npy"
            data = canonical.write_npy_v2_exclusive(self.arrays_dir / filename, array)
            digest = canonical.sha256_bytes(data)
            self.array_files[filename] = digest
            files[name] = {"file": f"arrays/{filename}", "file_sha256": digest, "tensor_sha256": canonical.tensor_sha256(array)}
        if files:
            body["arrays"] = files
        check_payload_value(body, instance_id)
        _fsync_append(self.instances_path, canonical.canonical_json_bytes(body) + b"\n")
        self.ids.append(instance_id)
        return body

    def finalize(self, *, complete: bool, derived: Mapping[str, Any] | None) -> dict:
        if self.finalized:
            raise PayloadError("payload already finalized")
        instances_bytes = self.instances_path.read_bytes()
        if instances_bytes:
            canonical.strict_canonical_jsonl_loads(instances_bytes)
        derived_body = dict(derived) if (complete and derived is not None) else None
        if derived_body is not None:
            check_payload_value(derived_body, "derived")
        arrays_digest = canonical.domain_separated_digest(
            PAYLOAD_DOMAIN + "/arrays", [h for _, h in sorted(self.array_files.items())])
        derived_sha = canonical.sha256_bytes(canonical.canonical_json_bytes(derived_body))
        status_sha = _status_sha(complete, self.ids)
        payload_sha = canonical.domain_separated_digest(PAYLOAD_DOMAIN, [
            canonical.sha256_bytes(self.header_bytes), canonical.sha256_bytes(instances_bytes), arrays_digest, derived_sha,
            status_sha])
        footer = {
            "schema": PAYLOAD_SCHEMA,
            "stage": self.stage,
            "completion": "complete" if complete else "incomplete",
            # For an incomplete payload this list depends on where the stop fell; incomplete
            # payloads are never compared as wholes (instance records still are).
            "finished_instance_ids": list(self.ids),
            "instance_count": len(self.ids),
            "header_file_sha256": canonical.sha256_bytes(self.header_bytes),
            "instances_file_sha256": canonical.sha256_bytes(instances_bytes),
            "array_files": [[name, digest] for name, digest in sorted(self.array_files.items())],
            "derived": derived_body,
            "payload_sha256": payload_sha,
        }
        record = canonical.with_self_hash(footer, "footer_self_sha256")
        canonical.write_canonical_json_exclusive(self.payload_dir / f"{self.stage}.footer.json", record)
        self.finalized = True
        return record


def read_payload(out_dir: Path, stage: str) -> dict:
    """Validate a payload directory (never consults the run record)."""

    payload_dir = Path(out_dir) / "payload"
    header_bytes = (payload_dir / f"{stage}.header.json").read_bytes()
    header = canonical.strict_canonical_json_loads(header_bytes)
    canonical.validate_self_hash(header, "header_self_sha256")
    footer = canonical.strict_canonical_json_loads((payload_dir / f"{stage}.footer.json").read_bytes())
    canonical.validate_self_hash(footer, "footer_self_sha256")
    instances_bytes = (payload_dir / f"{stage}.instances.jsonl").read_bytes()
    instances = canonical.strict_canonical_jsonl_loads(instances_bytes) if instances_bytes else []
    if canonical.sha256_bytes(instances_bytes) != footer["instances_file_sha256"]:
        raise PayloadError("instances file hash mismatch")
    if canonical.sha256_bytes(header_bytes) != footer["header_file_sha256"]:
        raise PayloadError("header file hash mismatch")
    on_disk = sorted(p.name for p in (payload_dir / "arrays").iterdir())
    if on_disk != [name for name, _ in footer["array_files"]]:
        raise PayloadError("array file inventory mismatch")
    for name, digest in footer["array_files"]:
        if canonical.file_sha256(payload_dir / "arrays" / name) != digest:
            raise PayloadError(f"array file hash mismatch: {name}")
    arrays_digest = canonical.domain_separated_digest(PAYLOAD_DOMAIN + "/arrays", [d for _, d in footer["array_files"]])
    derived_sha = canonical.sha256_bytes(canonical.canonical_json_bytes(footer["derived"]))
    if footer["completion"] not in ("complete", "incomplete"):
        raise PayloadError("invalid completion value")
    status_sha = _status_sha(footer["completion"] == "complete", footer["finished_instance_ids"])
    expected = canonical.domain_separated_digest(PAYLOAD_DOMAIN, [
        footer["header_file_sha256"], footer["instances_file_sha256"], arrays_digest, derived_sha, status_sha])
    if expected != footer["payload_sha256"]:
        raise PayloadError("payload_sha256 mismatch")
    if [r["instance_id"] for r in instances] != footer["finished_instance_ids"]:
        raise PayloadError("instance id list mismatch")
    return {"header": header, "instances": instances, "footer": footer}


def compare_payload_dirs(a: Path, b: Path) -> dict:
    """L1 comparison: byte equality of every payload file (run records ignored)."""

    def listing(root: Path) -> dict[str, str]:
        base = Path(root) / "payload"
        return {str(p.relative_to(base)): canonical.file_sha256(p) for p in sorted(base.rglob("*")) if p.is_file()}

    la, lb = listing(a), listing(b)
    differing = sorted(k for k in set(la) | set(lb) if la.get(k) != lb.get(k))
    return {"level": "L1", "equal": not differing, "differing_files": differing}


def instance_records(out_dir: Path, stage: str) -> dict[str, bytes]:
    data = (Path(out_dir) / "payload" / f"{stage}.instances.jsonl").read_bytes()
    out = {}
    for line in data.splitlines():
        record = json.loads(line)
        out[record["instance_id"]] = line
    return out


class RunRecord:
    """Variable metadata.  Separate files; nothing here feeds the payload."""

    def __init__(self, out_dir: Path, stage: str, caps: Mapping[str, Any]):
        self.dir = Path(out_dir) / "run"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.stage = stage
        self.caps = dict(caps)
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.t0 = time.monotonic()
        self.progress = self.dir / f"{stage}.progress.jsonl"
        self.regions: dict[str, float] = {}

    def log(self, **entry: Any) -> None:
        entry.setdefault("elapsed_seconds", time.monotonic() - self.t0)
        _fsync_append(self.progress, (json.dumps(entry, sort_keys=True) + "\n").encode("utf-8"))

    def add_region(self, name: str, seconds: float) -> None:
        self.regions[name] = self.regions.get(name, 0.0) + float(seconds)

    def finish(self, *, stop_reason: str, payload_sha256: str | None, peak_rss_bytes: int | None,
               extra: Mapping[str, Any] | None = None) -> dict:
        record = {
            "schema": RUN_SCHEMA,
            "stage": self.stage,
            "started_utc": self.started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.monotonic() - self.t0,
            "region_seconds": dict(sorted(self.regions.items())),
            "peak_rss_bytes": peak_rss_bytes,
            "caps": self.caps,
            "stop_reason": stop_reason,
            "pid": os.getpid(),
            "host_commitment": hashlib.sha256(b"operatormark_numerics_v2/host\x00" + socket.gethostname().encode()).hexdigest(),
            "cwd_commitment": hashlib.sha256(b"operatormark_numerics_v2/cwd\x00" + os.getcwd().encode()).hexdigest(),
            "executable": Path(sys.executable).name,
            "platform": platform.platform(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "payload_sha256": payload_sha256,
        }
        if extra:
            record["extra"] = dict(extra)
        (self.dir / f"{self.stage}.run.json").write_text(json.dumps(record, sort_keys=True, indent=1) + "\n")
        return record
