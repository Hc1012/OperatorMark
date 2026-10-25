"""Parameterized byte sizing for integration (storage/retention gates, §9.1 free space).

``size_corpus(n_ref, retention, layout=PRODUCTION_LAYOUT, ...)`` returns:

* exact raw tensor bytes and exact .npy header bytes per family (the header
  length comes from NumPy's own v2.0 header writer for each shard shape, so
  the 128-direction validation/test shards are sized correctly);
* metadata *upper bounds* built by the same record builders the writer uses,
  filled with worst-case values (every float.hex at its 24-character maximum,
  every boolean ``false``, the longest enumerated strings, the largest
  indices).  Records written by the generator never exceed these (tested);
* explicit unknowns: audit bytes, staging copies, failure-record retention and
  run records are ``{"status": "unknown"}`` unless the caller supplies a number;
* the §9.1 required-free-space formula max(5 GiB, 4 E + audit + 1 GiB) as a
  lower bound (audit taken as 0 while unknown), once on tensors plus headers
  and once with the metadata bound added.

Nothing is allocated; no array of corpus size is ever created.
"""

from __future__ import annotations

from operatormark.corpus import canonical

from . import layout as LY
from . import records as R
from .config import ANOMALY_STATES, MAX_ANOMALY_LADDERS
from .native import CODE_LIMIT, P1_SETTINGS, tpfa_settings

GiB = 2**30
WORST_HEX = "-0x1.fffffffffffffp+1023"          # longest float.hex of a finite binary64
SHA = "f" * 64
TKEYS = ("algebraic_backward_error", "boundary_transmissibility_error", "global_balance_ratio",
         "maximum_interior_face_antisymmetry", "maximum_local_conservation_ratio", "rhs_relative_l2_residual")
PKEYS = ("algebraic_backward_error", "boundary_reconstruction_error", "integrated_source", "outward_boundary_flux",
         "reaction_balance_ratio", "reaction_denominator", "rhs_relative_l2_residual")
CONFIGURATION_BOUND = 65536                       # bytes; the written configuration is tested against it
GATES = ("structural", "tpfa_native", "p1_256", "r_integral", "tangent_residual", "directions", "overall")


def unknown(reason: str) -> dict:
    return {"status": "unknown", "reason": reason}


def _worst_direction_record(layout: LY.Layout, split: str, n_ref: int, option: str) -> dict:
    local = layout.count(split) - 1
    k = LY.DIRECTIONS_PER_SAMPLE[split] - 1
    families = [name for name, _ in LY.DIRECTION_FAMILIES] + ([LY.native_tangent_family(n_ref)] if option == "S" else [])
    return R.direction_record(
        configuration_sha256=SHA, identity=layout.direction_identity(split, local, k), control_sha256=SHA,
        direction_sha256=SHA, tensor_sha256={f: SHA for f in families}, du_native_sha256=SHA,
        backward_error=None, r_integral_error=None,
        facts={"finite": False, "sign_rule": False, "unit_norm": False, "norm_squared": None},
        gates={"feasible": False, "tangent_residual": False, "r_integral": False, "overall": False}) | {
        "native_evidence": {"solution_tensor_sha256": SHA, "backward_error": WORST_HEX, "r_integral_error": WORST_HEX},
        "feasibility": {"finite": False, "sign_rule": False, "unit_norm": False, "norm_squared_fsum": WORST_HEX}}


def _worst_sample_record(layout: LY.Layout, split: str, n_ref: int, option: str) -> dict:
    local = layout.count(split) - 1
    d = LY.DIRECTIONS_PER_SAMPLE[split]
    drows = [{"direction_id": layout.direction_identity(split, local, k)["direction_id"],
              "direction_tensor_sha256": SHA, "record_self_sha256": SHA} for k in range(d)]
    families = [name for name, _ in LY.SAMPLE_FAMILIES] + ([LY.native_forward_family(n_ref)] if option == "S" else [])
    return R.sample_record(
        configuration_sha256=SHA, identity=layout.sample_identity(split, local), control_sha256=SHA,
        rho_storage_sha256=SHA, macro_sha256=SHA, tensor_sha256={f: SHA for f in families}, direction_records=drows,
        native={"n_ref": n_ref, "retention": option, "settings": tpfa_settings(n_ref), "coefficient_tensor_sha256": SHA,
                "solution_tensor_sha256": SHA, "diagnostics": {k: WORST_HEX for k in TKEYS}, "passes": False,
                "r_integral_error": WORST_HEX},
        p1={"settings": dict(P1_SETTINGS), "outcome": "failed:" + "X" * CODE_LIMIT, "nodal_tensor_sha256": SHA,
            "projected_tensor_sha256": SHA, "diagnostics": {k: WORST_HEX for k in PKEYS}, "iterations": 10000,
            "passes": False},
        metrics={k: WORST_HEX for k in ("delta_field", "norm_f64", "norm_projected_p256", "compliance_f64",
                                        "compliance_projected_p256")},
        anomaly={"procedure": max(ANOMALY_STATES, key=len), "bound": WORST_HEX, "flagged": False,
                 "decision": max(("not_applicable", "decided", "undecidable_p1_failed"), key=len)},
        gates={g: False for g in GATES})


def record_bounds(layout: LY.Layout, n_ref: int, option: str) -> dict:
    """Per-record upper bounds (bytes including the JSONL newline), by split."""

    out = {}
    for split in LY.SPLITS:
        if layout.count(split) == 0:
            continue
        out[split] = {"sample_record": len(R.record_bytes(_worst_sample_record(layout, split, n_ref, option))) + 1,
                      "direction_record": len(R.record_bytes(_worst_direction_record(layout, split, n_ref, option))) + 1}
    return out


def _measurement_bound(layout: LY.Layout, split: str, n_ref: int, option: str) -> int:
    d = LY.DIRECTIONS_PER_SAMPLE[split]
    record = {"schema": R.MEASUREMENT_SCHEMA, "configuration_sha256": SHA, "sample_id": "x" * 40,
              "disposition": "measured_shard_not_written", "reason": "generation_stopped",
              "control_hex": [WORST_HEX] * 64, "directions_hex": [[WORST_HEX] * 64] * d,
              "sample_record": _worst_sample_record(layout, split, n_ref, option),
              "direction_records": [_worst_direction_record(layout, split, n_ref, option)] * d}
    return len(R.record_bytes(R.self_hashed(record))) + 1


def _shard_manifest_bound(layout: LY.Layout, split: str, n_ref: int, option: str, bounds: dict) -> int:
    shard = layout.shards(split) - 1
    rel = layout.shard_dir(split, shard)
    members = []
    for name, _, shape in layout.shard_families(split, n_ref, option):
        size = LY.npy_header_bytes(shape) + 8 * int(__import__("math").prod(shape))
        members.append({"relative_path": f"{rel}/{LY.file_name(name)}", "size": size, "sha256": SHA,
                        "tensor_name": name, "dtype": "<f8", "shape": list(shape), "order": "C", "npy_version": [2, 0],
                        "tensor_sha256": SHA})
    b, d = layout.shard_size, LY.DIRECTIONS_PER_SAMPLE[split]
    for filename, schema, count, per in ((R.VERIFICATION_FILE, R.SAMPLE_SCHEMA, b, bounds["sample_record"]),
                                         (R.DIRECTIONS_FILE, R.DIRECTION_SCHEMA, b * d, bounds["direction_record"])):
        members.append({"relative_path": f"{rel}/{filename}", "size": count * per, "sha256": SHA, "schema": schema,
                        "record_count": count, "ordered_record_self_sha256": [SHA] * count})
    manifest = R.shard_manifest(configuration_sha256=SHA, layout=layout, split=split, shard=shard, members=members)
    return len(R.record_bytes(manifest))


def _summary_bound(layout: LY.Layout) -> int:
    ids = [layout.sample_identity(s, l)["sample_id"] for s in LY.SPLITS for l in range(layout.count(s))]
    shards = [{"split": s, "shard": i, "status": "not_reached", "manifest_sha256": SHA} for s, i in layout.all_shards()]
    big = 10**12
    record = R.summary(configuration_sha256=SHA, completion="incomplete", stop_reason="exception:" + "X" * 64,
                       shards=shards, counts={"samples_written": big, "directions_written": big, "failure_records": big,
                                              "unwritten_measurements": big},
                       anomalies={"relaxed": ids, "binary": ids}, gate_failures={g: ids for g in GATES[:-1]},
                       delta_max={"relaxed": WORST_HEX, "binary": WORST_HEX}, failures_sha256=SHA,
                       measurements_sha256=SHA, unwritten_anomalies={"relaxed": ids, "binary": ids})
    return len(R.record_bytes(record))


def _ladder_bound() -> int:
    result = {"valid": False, "reason": "ladder_invalid_or_inconsistent", "tpfa_failed": ["V1", "V2", "V3", "V4"],
              "p1_failed": ["V1", "V2", "V3", "V4"], "consistent": False, "e_hat": -1.7976931348623157e308,
              "within_budget": False}
    sid = f"{LY.BENCHMARK}/validation/999999"
    return len(R.record_bytes(R.ladder_record(configuration_sha256=SHA, sample_id=sid, stratum="relaxed", result=result))) + 1


def size_corpus(n_ref: int, retention: str, layout: LY.Layout = LY.PRODUCTION_LAYOUT, *, audit_bytes: int | None = None,
                staging_bytes: int | None = None, failure_retention_bytes: int | None = None,
                run_record_bytes: int | None = None) -> dict:
    LY.check_n_ref(n_ref)
    LY.check_retention(retention)
    rows = LY.tensor_inventory(layout, n_ref, retention)
    raw = sum(r.raw_bytes for r in rows)
    headers = sum(r.header_bytes for r in rows)
    files = sum(r.files for r in rows)
    bounds = record_bounds(layout, n_ref, retention)
    per_split = {}
    metadata = 0
    for split in LY.SPLITS:
        if layout.count(split) == 0:
            continue
        b = bounds[split]
        jsonl = layout.count(split) * (b["sample_record"] + b["direction_record"] * LY.DIRECTIONS_PER_SAMPLE[split])
        manifests = layout.shards(split) * _shard_manifest_bound(layout, split, n_ref, retention, b)
        per_split[split] = {"jsonl_upper_bound": jsonl, "shard_manifests_upper_bound": manifests}
        metadata += jsonl + manifests
    summary_bound = _summary_bound(layout)
    # Success and failure-only recovery may each leave temporary + final names.
    # Both populations are retained; no hard-link/deduplication credit is assumed.
    ladders = 4 * MAX_ANOMALY_LADDERS * _ladder_bound()
    metadata += CONFIGURATION_BOUND + summary_bound + ladders
    supplied = {"audit_bytes": audit_bytes, "staging_bytes": staging_bytes,
                "failure_retention_bytes": failure_retention_bytes, "run_record_bytes": run_record_bytes}
    for name, value in supplied.items():
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(name + ' must be a nonnegative integer or unknown')
    # failure records and unwritten-shard measurement rows: at most one row per sample, data-dependent count
    reasons = {"audit_bytes": "State 3-6 audit/evidence bytes are produced by Codex's State adapters, not by this package",
               "staging_bytes": "staging-to-final promotion copies depend on the State 5/6 custody design",
               "failure_retention_bytes": ("data-dependent: one failure record per failed sample, never retried, and one "
                                           "unwritten_shard_measurements row per solved sample of a shard that was not "
                                           "written; per-row upper bounds are in metadata_upper_bound"),
               "run_record_bytes": "variable run records (timings, memory) are outside the deterministic corpus"}
    extra = {k: (v if v is not None else unknown(reasons[k])) for k, v in supplied.items()}
    known_extra = sum(v for v in supplied.values() if v is not None)
    all_known = all(v is not None for v in supplied.values())

    def free(e: int, audit: int) -> int:
        return max(5 * GiB, 4 * e + audit + GiB)

    return {
        "n_ref": n_ref, "retention": retention, "layout": layout.record(),
        "samples": layout.total_samples, "directions": layout.total_directions,
        "families": [r.__dict__ | {"item_shape": list(r.item_shape)} for r in rows],
        "tensor_raw_bytes": raw, "npy_header_bytes": headers, "npy_files": files,
        "tensor_bytes_with_headers": raw + headers,
        "metadata_upper_bound": {"per_record_bytes": bounds, "per_split": per_split,
                                 "per_unwritten_measurement_row": {s: _measurement_bound(layout, s, n_ref, retention)
                                                                   for s in bounds},
                                 "configuration_bound": CONFIGURATION_BOUND, "summary_bound": summary_bound,
                                 "anomaly_ladders_bound": ladders, "total": metadata},
        "explicit_inputs": extra,
        "total": ({"status": "known", "bytes": raw + headers + metadata + known_extra} if all_known else
                  {"status": "lower_bound", "bytes": raw + headers + known_extra,
                   "with_metadata_upper_bound": raw + headers + metadata + known_extra,
                   "note": "unknown terms are excluded, so this is not a total"}),
        "section_9_1_required_free_bytes": {
            "formula": "max(5 GiB, 4 E + projected_audit_bytes + 1 GiB)",
            "lower_bound_tensors_and_headers": free(raw + headers, audit_bytes or 0),
            "lower_bound_with_metadata_bound": free(raw + headers + metadata, audit_bytes or 0),
            "audit_term": "supplied" if audit_bytes is not None else "unknown (taken as 0: lower bound only)"},
    }


def canonical_sizing_bytes(report: dict) -> bytes:
    return canonical.canonical_json_bytes(report)


def file_plan(n_ref: int, retention: str, layout: LY.Layout = LY.PRODUCTION_LAYOUT) -> list[dict]:
    """Every .npy file of the corpus with its exact byte size (header + data), in manifest order.

    For integration: replaces the three hard-coded v1 sites (Addendum A §6.3) -
    ``RAW_TENSOR_BYTES`` (sum of ``data_bytes``), the seven-family payload
    (the ``v1_family`` rows) and the header rule that assumed ``(64, 1, side, side)``
    (``header_bytes`` comes from each file's real shard shape).
    """

    import math

    v1 = {name for name, _ in LY.SAMPLE_FAMILIES}
    rows = []
    for split, shard in layout.all_shards():
        rel = layout.shard_dir(split, shard)
        for name, kind, shape in layout.shard_families(split, n_ref, retention):
            data = 8 * math.prod(shape)
            header = LY.npy_header_bytes(shape)
            rows.append({"relative_path": f"{rel}/{LY.file_name(name)}", "family": name, "kind": kind,
                         "shape": list(shape), "header_bytes": header, "data_bytes": data, "file_bytes": header + data,
                         "v1_family": name in v1})
    return rows
