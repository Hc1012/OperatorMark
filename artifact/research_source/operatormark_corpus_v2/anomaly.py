"""D10 anomaly ladders: run after generation, on the anomalies the *validator* re-derived.

For each anomalous sample (at most 64 in total) the stored control is read
back from its shard bytes and a full forward ladder with eta (T64/128/256,
P128/256/512, Drop 1 code unchanged) is computed.  Results go to
``anomaly_ladders.jsonl`` (exclusive write).  The samples themselves are never
touched.  With more than 64 anomalies no ladder is run: the downgrade is
already determined.
"""

from __future__ import annotations

from pathlib import Path
import sys

from operatormark.corpus import canonical

from . import d10, records as R
from .config import MAX_ANOMALY_LADDERS
from .layout import SPLIT_DIRECTORIES

# Failure evidence only: never a completed ladder result or readback input.
PARTIAL_LADDERS_FILE = "run/anomaly_ladders.partial.jsonl"


def _describe(exc) -> str:
    """repr() for an evidence note; a raising __repr__ yields the bare class name."""
    try:
        return repr(exc)
    except BaseException:
        cls = type(exc)
        name = type.__getattribute__(cls, "__qualname__") if type(cls) is type else None
        return "<" + (name if type(name) is str else "?unavailable") + "; repr unavailable>"


def _stored_control(root: Path, sample_id: str, shard_size: int):
    _, split, local = sample_id.split("/")
    shard, offset = divmod(int(local), shard_size)
    data = (root / "corpus_staging" / SPLIT_DIRECTORIES[split] / f"shard_{shard:04d}" / "rho.float64.npy").read_bytes()
    array = canonical.validate_npy_v2_bytes(data, expected_shape=(shard_size, 1, 8, 8))
    return array[offset, 0].copy()


def run_anomaly_ladders(root, assessment: dict) -> dict:
    """N_ref, eps_F and the shard size come from the corpus configuration; readback recomputes
    every ``within_budget`` from (valid, e_hat) with its own expected eps_F, so a wrong value is caught."""

    root = Path(root)
    if assessment["status"] not in ("matched", "incomplete") or assessment["checked"]["p1"] != "resolve":
        raise ValueError("anomaly ladders need a consistent assessment with P1-256 re-solved")
    data = (root / R.CONFIGURATION_FILE).read_bytes()
    cfg = canonical.strict_canonical_json_loads(data)
    canonical.validate_self_hash(cfg, "record_self_sha256")
    cfg_sha = canonical.sha256_bytes(data)
    n_ref, shard_size = cfg["n_ref"], cfg["layout"]["shard_size"]
    eps_field = float.fromhex(cfg["anomaly_procedure"]["eps_field"])
    anomalies = assessment["recomputed"]["anomalies"]
    total = sum(len(v) for v in anomalies.values())
    if total == 0:
        return {"ran": 0, "anomalies": 0}
    if total > MAX_ANOMALY_LADDERS:
        return {"ran": 0, "anomalies": total, "reason": f"more than {MAX_ANOMALY_LADDERS} anomalies: field not confirmed"}
    rows, results = [], {}
    try:
        for stratum in ("relaxed", "binary"):
            for sample_id in anomalies[stratum]:
                result = d10.anomaly_ladder(_stored_control(root, sample_id, shard_size), n_ref, eps_field)
                results[sample_id] = result
                rows.append(R.ladder_record(configuration_sha256=cfg_sha, sample_id=sample_id, stratum=stratum, result=result))
        canonical.write_canonical_jsonl_exclusive(root / "anomaly_ladders.jsonl", rows)
    except BaseException as primary:
        # Same exclusive, fsynced, independently reread canonical publisher used
        # by existing evidence. No ladder retry or successful partial publication.
        # A final-file publication error may coexist with its temporary/final
        # name and this recovery's temporary/final name: sizing reserves all four.
        if rows:
            try:
                data = canonical.write_canonical_jsonl_exclusive(root / PARTIAL_LADDERS_FILE, rows)
                primary.add_note(f"completed anomaly ladder rows retained: {len(rows)}; "
                                 f"{PARTIAL_LADDERS_FILE}; sha256={canonical.sha256_bytes(data)}")
            except BaseException as secondary:
                try:      # describing or reporting a secondary failure never replaces the primary
                    message = "completed anomaly ladder evidence unavailable: " + _describe(secondary)
                    primary.add_note(message)
                    try:
                        sys.stderr.write(message + "\n"); sys.stderr.flush()
                    except BaseException as reporting:
                        primary.add_note("anomaly ladder evidence reporting unavailable: " + _describe(reporting))
                except BaseException:
                    pass
        raise
    return {"ran": len(rows), "anomalies": total, "results": results}
