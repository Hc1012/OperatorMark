"""Canonical v2 records (schema identifiers are Drop 2 proposals for the schema review).

Every record is canonical JSON (frozen ``canonical.canonical_json_bytes``:
sorted keys, compact, UTF-8, no NaN) with a self-hash.  Scientific scalars are
``float.hex`` strings; no Python float, clock, path-from-the-host, pid or
hostname enters a record.  Relative paths are corpus-root relative.

Files under the corpus root:
  corpus_configuration.json                       configuration (self-hashed); bound by every other record
  corpus_staging/<split dir>/shard_NNNN/
      <family>.float64.npy                        canonical .npy v2.0, shard arrays
      verification.jsonl                          one sample record per sample, in order
      directions.jsonl                            one direction record per direction, in order
      shard_manifest.json                         members (size, sha256, tensor sha256), ordered ids
  failures.jsonl                                  retained failure records (append-only; may be empty)
  unwritten_shard_measurements.jsonl              D10 evidence of solved samples in shards that could not be written
  generation_summary.json                         writer's summary: a claim, re-derived by the validator
  run/generation_run.json                         variable run record (timings, memory); never validated
"""

from __future__ import annotations

from typing import Iterable

from operatormark.corpus import canonical

from . import layout as LY

PREFIX = "operatormark.darcy_branch_c_v2"
CONFIGURATION_SCHEMA = f"{PREFIX}.corpus_configuration.v1"
SAMPLE_SCHEMA = f"{PREFIX}.sample_verification.v1"
DIRECTION_SCHEMA = f"{PREFIX}.direction_verification.v1"
SHARD_SCHEMA = f"{PREFIX}.shard_manifest.v1"
FAILURE_SCHEMA = f"{PREFIX}.generation_failure.v1"
SUMMARY_SCHEMA = f"{PREFIX}.generation_summary.v1"
LADDER_SCHEMA = f"{PREFIX}.anomaly_ladder.v1"
CONFIGURATION_FILE = "corpus_configuration.json"
FAILURES_FILE = "failures.jsonl"
MEASUREMENTS_FILE = "unwritten_shard_measurements.jsonl"
MEASUREMENT_SCHEMA = f"{PREFIX}.unwritten_shard_measurement.v1"
SUMMARY_FILE = "generation_summary.json"
VERIFICATION_FILE = "verification.jsonl"
DIRECTIONS_FILE = "directions.jsonl"
SHARD_MANIFEST_FILE = "shard_manifest.json"
SUMMARY_DOMAIN = "operatormark_corpus_v2/summary/shard-manifests/v1"


def hx(value) -> str | None:
    return None if value is None else float(value).hex()


def self_hashed(record: dict, field: str = "record_self_sha256") -> dict:
    return canonical.with_self_hash(record, field)


def record_bytes(record: dict) -> bytes:
    return canonical.canonical_json_bytes(record)


# ---------------------------------------------------------------------------- configuration
def configuration(cfg, *, controls_binding: dict, directions_binding: dict, solver_kind: str, identity: dict,
                  scope_sha256: str, d10_settings: dict, failure_policy: str) -> dict:
    retention = {"option": cfg.retention.option,
                 "classes": LY.retention_classes(cfg.n_ref, cfg.retention.option),
                 "decision_reference_sha256": cfg.retention.decision_reference_sha256}
    choices = None if cfg.choices is None else cfg.choices.record()
    return self_hashed({
        **({'development': {'request_sha256': cfg.development_reference_sha256,
                            'physical_field_confirmed': False, 'ladder_trigger': 'confirmed_only',
                            'ladder_cap_per_stratum': 64, 'state4_admission': 'not_claimed'}}
           if cfg.mode == 'real_development' else {}),
        "schema": CONFIGURATION_SCHEMA, "benchmark": LY.BENCHMARK, "mode": cfg.mode,
        "synthetic": cfg.synthetic, "production_authorized": cfg.mode == "production",
        "n_ref": cfg.n_ref, "retention": retention, "layout": cfg.layout.record(),
        "selection": [[s, i] for s, i in cfg.shards()], "selection_is_complete": cfg.shards() == cfg.layout.all_shards(),
        "sources": {"controls": controls_binding, "directions": directions_binding, "solver": solver_kind},
        "failure_policy": {"on_failure": failure_policy, "failed_shard": "not_written; failure records retained",
                           "resampling": "never"},
        "qualification": None if cfg.qualification is None else cfg.qualification.record(),
        "scientific_choices": choices, "anomaly_procedure": cfg.anomaly.record(), "d10": d10_settings,
        "code_identity": identity, "determinism_scope_sha256": scope_sha256,
    })


# ---------------------------------------------------------------------------- per-item records
def direction_record(*, configuration_sha256: str, identity: dict, control_sha256: str, direction_sha256: str,
                     tensor_sha256: dict, du_native_sha256: str, backward_error, r_integral_error, facts: dict,
                     gates: dict) -> dict:
    return self_hashed({
        "schema": DIRECTION_SCHEMA, "configuration_sha256": configuration_sha256, **identity,
        "control_tensor_sha256": control_sha256, "direction_tensor_sha256": direction_sha256,
        "tensor_sha256": dict(sorted(tensor_sha256.items())),
        "native_evidence": {"solution_tensor_sha256": du_native_sha256, "backward_error": hx(backward_error),
                            "r_integral_error": hx(r_integral_error)},
        "feasibility": {"finite": facts["finite"], "sign_rule": facts["sign_rule"], "unit_norm": facts["unit_norm"],
                        "norm_squared_fsum": hx(facts["norm_squared"])},
        "gates": dict(gates),
    })


def sample_record(*, configuration_sha256: str, identity: dict, control_sha256: str, rho_storage_sha256: str,
                  macro_sha256: str, tensor_sha256: dict, direction_records: list[dict], native: dict, p1: dict,
                  metrics: dict, anomaly: dict, gates: dict) -> dict:
    ident = {k: identity[k] for k in ("benchmark", "split", "shard_index", "local_index", "global_index", "sample_id",
                                      "stratum")}
    return self_hashed({
        "schema": SAMPLE_SCHEMA, "configuration_sha256": configuration_sha256, **ident,
        "control_tensor_sha256": control_sha256, "rho_storage_tensor_sha256": rho_storage_sha256,
        "macro_coefficient_tensor_sha256": macro_sha256, "tensor_sha256": dict(sorted(tensor_sha256.items())),
        "directions": {"count": len(direction_records), "ids": [r["direction_id"] for r in direction_records],
                       "direction_tensor_sha256": [r["direction_tensor_sha256"] for r in direction_records],
                       "record_self_sha256": [r["record_self_sha256"] for r in direction_records]},
        "native_evidence": native, "p1_256": p1, "metrics": metrics, "anomaly": anomaly, "gates": dict(gates),
    })


def failure_record(*, configuration_sha256: str, identity: dict, stage: str, code: str, partial: dict) -> dict:
    ident = {k: identity[k] for k in ("benchmark", "split", "shard_index", "local_index", "global_index", "sample_id",
                                      "stratum")}
    return self_hashed({"schema": FAILURE_SCHEMA, "configuration_sha256": configuration_sha256, **ident,
                        "stage": stage, "code": code, "partial": partial, "disposition": "retained_not_replaced"})


# ---------------------------------------------------------------------------- shard manifest
def npy_member(relative_path: str, family: str, data: bytes, shape: tuple, tensor_sha: str) -> dict:
    return {"relative_path": relative_path, "size": len(data), "sha256": canonical.sha256_bytes(data),
            "tensor_name": family, "dtype": "<f8", "shape": list(shape), "order": "C", "npy_version": [2, 0],
            "tensor_sha256": tensor_sha}


def jsonl_member(relative_path: str, schema: str, data: bytes, rows: list[dict]) -> dict:
    return {"relative_path": relative_path, "size": len(data), "sha256": canonical.sha256_bytes(data),
            "schema": schema, "record_count": len(rows), "ordered_record_self_sha256": [r["record_self_sha256"] for r in rows]}


def shard_manifest(*, configuration_sha256: str, layout: LY.Layout, split: str, shard: int, members: list[dict]) -> dict:
    locals_ = list(layout.shard_samples(split, shard))
    dirs = list(layout.iter_shard_directions(split, shard))
    first_dir = layout.direction_identity(split, *dirs[0])
    last_dir = layout.direction_identity(split, *dirs[-1])
    return canonical.with_self_hash({
        "schema": SHARD_SCHEMA, "benchmark": LY.BENCHMARK, "configuration_sha256": configuration_sha256,
        "split": split, "shard_index": shard, "sample_count": len(locals_), "direction_count": len(dirs),
        "first_local_index": locals_[0], "last_local_index": locals_[-1],
        "first_global_index": layout.sample_offset(split) + locals_[0],
        "last_global_index": layout.sample_offset(split) + locals_[-1],
        "first_direction_global_index": first_dir["direction_global_index"],
        "last_direction_global_index": last_dir["direction_global_index"],
        "ordered_sample_ids": [layout.sample_identity(split, l)["sample_id"] for l in locals_],
        "ordered_direction_ids": [layout.direction_identity(split, l, k)["direction_id"] for l, k in dirs],
        "members": members,
    }, "manifest_self_sha256")


# ---------------------------------------------------------------------------- summary
def measurement_record(*, configuration_sha256: str, rho, directions, sample_record: dict, direction_records: list,
                       reason: str) -> dict:
    """Full D10 evidence of a sample whose shard could not be written (another sample in it failed).

    The control and directions are kept as exact float.hex values so the
    measurement can be re-verified (or an anomaly ladder run) later.
    """

    return self_hashed({"schema": MEASUREMENT_SCHEMA, "configuration_sha256": configuration_sha256,
                        "sample_id": sample_record["sample_id"], "disposition": "measured_shard_not_written",
                        "reason": reason,
                        "control_hex": [float(x).hex() for x in rho.reshape(-1)],
                        "directions_hex": [[float(x).hex() for x in v.reshape(-1)] for v in directions],
                        "sample_record": sample_record, "direction_records": list(direction_records)})


def summary(*, configuration_sha256: str, completion: str, stop_reason: str | None, shards: list[dict],
            counts: dict, anomalies: dict, gate_failures: dict, delta_max: dict, failures_sha256: str,
            measurements_sha256: str, unwritten_anomalies: dict) -> dict:
    written = [s["manifest_sha256"] for s in shards if s["status"] == "written"]
    return canonical.with_self_hash({
        "schema": SUMMARY_SCHEMA, "benchmark": LY.BENCHMARK, "configuration_sha256": configuration_sha256,
        "completion": completion, "stop_reason": stop_reason, "shards": shards, "counts": counts,
        "anomalies": anomalies, "gate_failures": gate_failures, "delta_max": delta_max,
        "failures_file_sha256": failures_sha256, "measurements_file_sha256": measurements_sha256,
        "anomalies_in_unwritten_shards": unwritten_anomalies,
        "written_manifest_aggregate_sha256": canonical.domain_separated_digest(SUMMARY_DOMAIN, written),
        "note": "writer claim; the readback validator re-derives every fact from bytes",
    }, "summary_self_sha256")


def ladder_record(*, configuration_sha256: str, sample_id: str, stratum: str, result: dict) -> dict:
    out = dict(result)
    out["e_hat"] = hx(out.get("e_hat"))
    return self_hashed({"schema": LADDER_SCHEMA, "configuration_sha256": configuration_sha256,
                        "sample_id": sample_id, "stratum": stratum, "result": out})


def jsonl_bytes(rows: Iterable[dict]) -> bytes:
    rows = list(rows)
    return canonical.canonical_jsonl_bytes(rows) if rows else b""
