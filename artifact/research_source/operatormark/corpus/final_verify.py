"""Independent verification of synthetic final-corpus shard bytes.

This module deliberately defines its own frozen expectations.  It reuses only
the generic canonical byte codecs and never calls a producer-side record
validator.  It does not solve a PDE, retrieve a seed, promote a directory, or
write a completion marker.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .canonical import (
    CanonicalRecordError,
    canonical_json_bytes,
    sha256_bytes,
    strict_canonical_json_loads,
    strict_canonical_jsonl_loads,
    tensor_sha256,
    validate_npy_v2_bytes,
)


class FinalVerificationError(CanonicalRecordError):
    """Raised when independently inspected final-shard evidence is invalid."""


_BENCHMARK: Final = "darcy_branch_c_v1"
_PROTOCOL_SHA256: Final = (
    "a38ef79b6c4bdccf53f599fa08f77e48a111afbf0b2783ad5733bda99996599a"
)
_FINAL_RECORD_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.sample_verification.v1"
)
_FINAL_SHARD_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.final_shard_manifest.v1"
)
_FINAL_CONTEXT_SCHEMA: Final = "operatormark.darcy_branch_c_v1.final_context.v1"
_PRE_SOLVE_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.state4_pre_solve_manifest.v2"
)
_CONTROL_RECORD_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.state4_control_record.v1"
)
_CONTROL_SHARD_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.state4_controls_shard_manifest.v1"
)
_COMPLETE_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.state4_controls_complete.v1"
)
# Independently repeated adopted Section 9 version token, not imported from
# the producer-side format module.
_CORPUS_VERSION: Final = "private_corpus_v1"
_PROTOCOL_RELATIVE_NAME: Final = (
    "OperatorMark_Private_Corpus_PreGeneration_Frozen_Protocol_v1.0.md"
)
_SOLVER_BASELINE_COMMIT: Final = "e16456395cfe67b73b9d298fd0491cf9524d798a"
_SOLVER_BASELINE_TREE: Final = "81f40e24102e51e0da3ce65dafd769a7678565fd"
_EMPTY_SHA256: Final = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_SPLIT_ORDER: Final = ("training", "validation", "test")

_SPLIT_COUNTS: Final = {"training": 4096, "validation": 256, "test": 512}
_SPLIT_SHARDS: Final = {"training": 64, "validation": 4, "test": 8}
_SPLIT_OFFSETS: Final = {"training": 0, "validation": 4096, "test": 4352}
_SPLIT_DIRECTORIES: Final = {
    "training": "training",
    "validation": "validation",
    "test": "test.sealed",
}

# Order is a protocol fact used for positional manifest members.  JSON object
# key insertion order is intentionally not used as validation evidence.
_FAMILIES: Final = (
    ("rho", "rho.float64.npy", (1, 8, 8), (64, 1, 8, 8)),
    ("kappa64", "kappa64.float64.npy", (1, 64, 64), (64, 1, 64, 64)),
    ("u_tpfa64", "u_tpfa64.float64.npy", (1, 64, 64), (64, 1, 64, 64)),
    (
        "kappa32_areaavg",
        "kappa32_areaavg.float64.npy",
        (1, 32, 32),
        (64, 1, 32, 32),
    ),
    ("u32_areaavg", "u32_areaavg.float64.npy", (1, 32, 32), (64, 1, 32, 32)),
    (
        "kappa16_areaavg",
        "kappa16_areaavg.float64.npy",
        (1, 16, 16),
        (64, 1, 16, 16),
    ),
    ("u16_areaavg", "u16_areaavg.float64.npy", (1, 16, 16), (64, 1, 16, 16)),
)
_FAMILY_NAMES: Final = tuple(item[0] for item in _FAMILIES)

_FINAL_RECORD_KEYS: Final = {
    "schema",
    "benchmark",
    "split",
    "shard_index",
    "local_index",
    "global_index",
    "sample_id",
    "stratum",
    "control_tensor_sha256",
    "rho_storage_tensor_sha256",
    "macro_coefficient_tensor_sha256",
    "tensor_sha256",
    "verifier_tensor_sha256",
    "solver_diagnostics",
    "metrics",
    "gates",
    "record_self_sha256",
}
_CONTROL_RECORD_KEYS: Final = {
    "schema",
    "benchmark",
    "split",
    "shard_index",
    "local_index",
    "global_index",
    "sample_id",
    "stratum",
    "control_tensor_sha256",
    "rho_storage_tensor_sha256",
    "macro_coefficient_tensor_sha256",
    "record_self_sha256",
}
_T64_KEYS: Final = {
    "algebraic_backward_error",
    "boundary_transmissibility_error",
    "maximum_local_conservation_ratio",
    "maximum_interior_face_antisymmetry",
    "global_balance_ratio",
    "rhs_relative_l2_residual",
    "passes",
}
_P256_KEYS: Final = {
    "algebraic_backward_error",
    "boundary_reconstruction_error",
    "reaction_denominator",
    "reaction_balance_ratio",
    "outward_boundary_flux",
    "integrated_source",
    "rhs_relative_l2_residual",
    "iterations",
    "passes",
}
_GATE_KEYS: Final = {"structural", "TPFA64", "P1_256", "cross_solver", "overall"}
_FINAL_SHARD_KEYS: Final = {
    "schema",
    "benchmark",
    "split",
    "shard_index",
    "sample_count",
    "first_local_index",
    "last_local_index",
    "first_global_index",
    "last_global_index",
    "ordered_sample_ids",
    "parent_context",
    "pre_solve_manifest",
    "state4_controls_shard_manifest",
    "members",
    "manifest_self_sha256",
}
_JSON_REFERENCE_KEYS: Final = {"relative_path", "size", "sha256", "schema"}
_NPY_MEMBER_KEYS: Final = {
    "relative_path",
    "size",
    "sha256",
    "tensor_name",
    "dtype",
    "shape",
    "order",
    "npy_version",
    "tensor_sha256",
}
_JSONL_MEMBER_KEYS: Final = {
    "relative_path",
    "size",
    "sha256",
    "schema",
    "record_count",
    "ordered_record_self_sha256",
}
_CONTROL_SHARD_KEYS: Final = {
    "schema",
    "benchmark",
    "split",
    "sealed",
    "shard_index",
    "first_local_index",
    "last_local_index",
    "first_global_index",
    "last_global_index",
    "sample_count",
    "rho_relative_path",
    "rho_dtype",
    "rho_shape",
    "rho_order",
    "npy_version",
    "rho_file_bytes",
    "rho_file_sha256",
    "rho_tensor_sha256",
    "control_records_relative_path",
    "control_records_bytes",
    "control_records_file_sha256",
    "ordered_record_self_sha256",
    "protocol_file_sha256",
    "state4_authorization_file_sha256",
    "implementation_commit",
    "implementation_tree",
    "environment_record_sha256",
    "manifest_self_sha256",
}
_FINAL_CONTEXT_KEYS: Final = {
    "schema",
    "benchmark",
    "corpus_version",
    "attempt_id",
    "protocol_file_sha256",
    "solver_spec_baseline_commit",
    "solver_spec_baseline_tree",
    "implementation_commit",
    "implementation_tree",
    "state4_authorization_file_sha256",
    "state5_authorization_file_sha256",
    "state6_authorization_file_sha256",
    "pre_solve_manifest",
    "controls_completion",
    "split_order",
    "record_self_sha256",
}
_PRE_SOLVE_KEYS: Final = {
    "schema",
    "benchmark",
    "corpus_version",
    "state",
    "attempt_id",
    "protocol_relative_name",
    "protocol_file_sha256",
    "state4_authorization_file_sha256",
    "solver_spec_baseline_commit",
    "solver_spec_baseline_tree",
    "implementation_commit",
    "implementation_tree",
    "state3_audit_file_sha256",
    "compute_feasibility_report_file_sha256",
    "environment_record_sha256",
    "storage_preflight_before_file_sha256",
    "storage_preflight_after_file_sha256",
    "rng_audit_relative_path",
    "rng_audit_file_sha256",
    "read_only_evidence_relative_path",
    "read_only_evidence_file_sha256",
    "split_order",
    "split_manifest_paths",
    "split_manifest_file_sha256",
    "sample_count",
    "shard_count",
    "global_control_aggregate_sha256",
    "global_macro_coefficient_aggregate_sha256",
    "all_control_hashes_unique",
    "all_macro_coefficient_hashes_unique",
    "rng_call_count_by_split",
    "forbidden_seed_or_data_access_count",
    "git_before",
    "git_after",
    "manifest_self_sha256",
}
_COMPLETE_KEYS: Final = {
    "schema",
    "state",
    "attempt_id",
    "pre_solve_manifest_relative_path",
    "pre_solve_manifest_file_sha256",
    "read_only_evidence_file_sha256",
    "state4_authorization_file_sha256",
    "completion_time_utc",
    "record_self_sha256",
}
_GIT_RECORD_KEYS: Final = {
    "branch",
    "head",
    "tree",
    "parents",
    "status_short_untracked_all",
    "tracked_diff_sha256",
    "tracked_diff_size_bytes",
    "staged_diff_sha256",
    "staged_diff_size_bytes",
    "tracked_changes_empty",
    "staged_changes_empty",
    "untracked_inclusive_status_empty",
}

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_BOUNDARY_TOLERANCE: Final = 100.0 * float.fromhex("0x1.0000000000000p-52")


def _mapping(value: object, keys: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalVerificationError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != keys:
        raise FinalVerificationError(
            f"{name} keys mismatch; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _integer(
    value: object, *, name: str, minimum: int = 0, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinalVerificationError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise FinalVerificationError(f"{name} must be <= {maximum}")
    return value


def _true(value: object, *, name: str) -> None:
    if value is not True:
        raise FinalVerificationError(f"{name} must be JSON true")


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise FinalVerificationError(f"{name} must be lowercase SHA-256 hexadecimal")
    return value


def _git_oid(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not _GIT_OID_RE.fullmatch(value):
        raise FinalVerificationError(f"{name} must be a lowercase 40-hex Git object ID")
    return value


def _self_hash(record: Mapping[str, Any], field: str, *, name: str) -> None:
    claimed = _sha256(record.get(field), name=f"{name}.{field}")
    payload = dict(record)
    payload.pop(field, None)
    if claimed != sha256_bytes(canonical_json_bytes(payload)):
        raise FinalVerificationError(f"{name} self-hash mismatch")


def _hex_scalar(value: object, *, name: str, nonnegative: bool) -> float:
    if not isinstance(value, str):
        raise FinalVerificationError(f"{name} must be canonical binary64 hexadecimal")
    try:
        scalar = float.fromhex(value)
    except (ValueError, OverflowError) as exc:
        raise FinalVerificationError(
            f"{name} must be canonical binary64 hexadecimal"
        ) from exc
    if not math.isfinite(scalar) or value != scalar.hex().lower():
        raise FinalVerificationError(f"{name} must be finite canonical float.hex()")
    if nonnegative and scalar < 0.0:
        raise FinalVerificationError(f"{name} must be nonnegative")
    return scalar


def _split_and_shard(split: object, shard_index: object) -> tuple[str, int]:
    if split not in _SPLIT_COUNTS:
        raise FinalVerificationError("split must be training, validation, or test")
    split_name = str(split)
    shard = _integer(shard_index, name="shard_index")
    if shard >= _SPLIT_SHARDS[split_name]:
        raise FinalVerificationError("shard_index is outside the frozen split")
    return split_name, shard


def _expected_identity(split: str, shard: int, offset: int) -> tuple[int, int, str, str]:
    local = shard * 64 + offset
    global_index = _SPLIT_OFFSETS[split] + local
    sample_id = f"{_BENCHMARK}/{split}/{local:06d}"
    stratum = "relaxed" if local % 2 == 0 else "binary_32_of_64"
    return local, global_index, sample_id, stratum


def _validate_identity_fields(
    record: Mapping[str, Any], *, split: str, shard: int, offset: int, name: str
) -> None:
    local, global_index, sample_id, stratum = _expected_identity(split, shard, offset)
    if record["benchmark"] != _BENCHMARK:
        raise FinalVerificationError(f"{name} benchmark mismatch")
    if record["split"] != split or record["shard_index"] != shard:
        raise FinalVerificationError(f"{name} split/shard mismatch")
    for field in ("shard_index", "local_index", "global_index"):
        _integer(record[field], name=f"{name}.{field}")
    if record["local_index"] != local or record["global_index"] != global_index:
        raise FinalVerificationError(f"{name} index/order mismatch")
    if record["sample_id"] != sample_id:
        raise FinalVerificationError(f"{name} sample identifier mismatch")
    if record["stratum"] != stratum:
        raise FinalVerificationError(f"{name} frozen stratum parity mismatch")


def _validate_control_record(
    record: Mapping[str, Any], *, split: str, shard: int, offset: int
) -> None:
    row = _mapping(record, _CONTROL_RECORD_KEYS, name="control record")
    if row["schema"] != _CONTROL_RECORD_SCHEMA:
        raise FinalVerificationError("control-record schema mismatch")
    _validate_identity_fields(row, split=split, shard=shard, offset=offset, name="control")
    for field in (
        "control_tensor_sha256",
        "rho_storage_tensor_sha256",
        "macro_coefficient_tensor_sha256",
    ):
        _sha256(row[field], name=f"control.{field}")
    _self_hash(row, "record_self_sha256", name="control record")


def _frozen_safe_ratio(numerator: float, denominator: float) -> float:
    if (
        not math.isfinite(numerator)
        or numerator < 0.0
        or not math.isfinite(denominator)
        or denominator < 0.0
    ):
        raise FinalVerificationError(
            "P256 reaction-balance ratio operands must be finite and nonnegative"
        )
    if denominator > 0.0:
        result = numerator / denominator
        if not math.isfinite(result):
            raise FinalVerificationError("P256 reaction-balance ratio must be finite")
        return result
    if numerator == 0.0:
        return 0.0
    return math.inf


def _validate_solver_evidence(record: Mapping[str, Any]) -> None:
    diagnostics = _mapping(
        record["solver_diagnostics"], {"T64", "P256"}, name="solver_diagnostics"
    )
    t64 = _mapping(diagnostics["T64"], _T64_KEYS, name="solver_diagnostics.T64")
    t64_values = {
        field: _hex_scalar(t64[field], name=f"T64.{field}", nonnegative=True)
        for field in _T64_KEYS - {"passes"}
    }
    for field in (
        "algebraic_backward_error",
        "maximum_local_conservation_ratio",
        "maximum_interior_face_antisymmetry",
        "global_balance_ratio",
    ):
        if t64_values[field] > 1.0e-10:
            raise FinalVerificationError(f"T64.{field} exceeds its frozen gate")
    if t64_values["boundary_transmissibility_error"] > _BOUNDARY_TOLERANCE:
        raise FinalVerificationError("T64 boundary error exceeds its frozen gate")
    _true(t64["passes"], name="T64.passes")

    p256 = _mapping(diagnostics["P256"], _P256_KEYS, name="solver_diagnostics.P256")
    p256_values: dict[str, float] = {}
    for field in _P256_KEYS - {"iterations", "passes"}:
        p256_values[field] = _hex_scalar(
            p256[field],
            name=f"P256.{field}",
            nonnegative=field not in {"outward_boundary_flux", "integrated_source"},
        )
    balance_numerator = abs(
        p256_values["outward_boundary_flux"] - p256_values["integrated_source"]
    )
    expected_balance_ratio = _frozen_safe_ratio(
        balance_numerator, p256_values["reaction_denominator"]
    )
    if (
        not math.isfinite(expected_balance_ratio)
        or p256["reaction_balance_ratio"] != expected_balance_ratio.hex().lower()
    ):
        raise FinalVerificationError(
            "P256 reaction_balance_ratio does not match the frozen balance formula"
        )
    if p256_values["algebraic_backward_error"] > 1.0e-10:
        raise FinalVerificationError("P256 backward error exceeds its frozen gate")
    if p256_values["boundary_reconstruction_error"] > _BOUNDARY_TOLERANCE:
        raise FinalVerificationError("P256 boundary error exceeds its frozen gate")
    if p256_values["reaction_balance_ratio"] > 1.0e-10:
        raise FinalVerificationError("P256 balance error exceeds its frozen gate")
    _integer(
        p256["iterations"], name="P256.iterations", minimum=1, maximum=10_000
    )
    _true(p256["passes"], name="P256.passes")

    metrics = _mapping(
        record["metrics"], {"scales", "compliances", "delta_L2", "delta_J"}, name="metrics"
    )
    scales = _mapping(metrics["scales"], {"S_L2", "S_J"}, name="metrics.scales")
    decoded_scales = {
        field: _hex_scalar(
            scales[field], name=f"metrics.scales.{field}", nonnegative=True
        )
        for field in ("S_L2", "S_J")
    }
    compliances = _mapping(
        metrics["compliances"], {"T64", "bar_P256"}, name="metrics.compliances"
    )
    decoded_compliances = {
        field: _hex_scalar(
            compliances[field],
            name=f"metrics.compliances.{field}",
            nonnegative=False,
        )
        for field in ("T64", "bar_P256")
    }
    delta_l2 = _hex_scalar(metrics["delta_L2"], name="metrics.delta_L2", nonnegative=True)
    delta_j = _hex_scalar(metrics["delta_J"], name="metrics.delta_J", nonnegative=True)
    if decoded_scales["S_J"] != abs(decoded_compliances["bar_P256"]):
        raise FinalVerificationError(
            "metrics.scales.S_J does not equal abs(bar_P256 compliance)"
        )
    numerator = abs(
        decoded_compliances["T64"] - decoded_compliances["bar_P256"]
    )
    denominator = decoded_scales["S_J"]
    expected_delta_j = (
        numerator / denominator
        if denominator > 0.0
        else 0.0
        if numerator == 0.0
        else math.inf
    )
    if not math.isfinite(expected_delta_j) or delta_j != expected_delta_j:
        raise FinalVerificationError(
            "metrics.delta_J does not match the frozen compliance formula"
        )
    if delta_l2 > 0.02543117962436324:
        raise FinalVerificationError("delta_L2 exceeds its frozen gate")
    if delta_j > 0.022582030716092032:
        raise FinalVerificationError("delta_J exceeds its frozen gate")

    gates = _mapping(record["gates"], _GATE_KEYS, name="gates")
    for field in _GATE_KEYS:
        _true(gates[field], name=f"gates.{field}")


def _validate_final_record(
    record: Mapping[str, Any], *, split: str, shard: int, offset: int
) -> None:
    row = _mapping(record, _FINAL_RECORD_KEYS, name="final verification record")
    if row["schema"] != _FINAL_RECORD_SCHEMA:
        raise FinalVerificationError("final-verification schema mismatch")
    _validate_identity_fields(row, split=split, shard=shard, offset=offset, name="final")
    for field in (
        "control_tensor_sha256",
        "rho_storage_tensor_sha256",
        "macro_coefficient_tensor_sha256",
    ):
        _sha256(row[field], name=f"final.{field}")
    tensors = _mapping(row["tensor_sha256"], set(_FAMILY_NAMES), name="tensor_sha256")
    for family in _FAMILY_NAMES:
        _sha256(tensors[family], name=f"tensor_sha256.{family}")
    if tensors["rho"] != row["rho_storage_tensor_sha256"]:
        raise FinalVerificationError("rho hash cross-check mismatch")
    verifier = _mapping(
        row["verifier_tensor_sha256"], {"P256", "bar_P256"}, name="verifier_tensor_sha256"
    )
    for field in ("P256", "bar_P256"):
        _sha256(verifier[field], name=f"verifier_tensor_sha256.{field}")
    _validate_solver_evidence(row)
    _self_hash(row, "record_self_sha256", name="final verification record")


def validate_final_verification_jsonl(
    data: bytes, *, split: str, shard_index: int
) -> list[dict[str, Any]]:
    """Independently validate one canonical, accepted-domain 64-row JSONL value."""

    split_name, shard = _split_and_shard(split, shard_index)
    if not isinstance(data, bytes):
        raise FinalVerificationError("final verification JSONL input must be bytes")
    try:
        records = strict_canonical_jsonl_loads(data)
    except CanonicalRecordError as exc:
        raise FinalVerificationError(str(exc)) from exc
    if len(records) != 64:
        raise FinalVerificationError("final verification JSONL must contain exactly 64 rows")
    for offset, record in enumerate(records):
        _validate_final_record(record, split=split_name, shard=shard, offset=offset)
    return records


def _canonical_relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != unicodedata.normalize("NFC", value):
        raise FinalVerificationError(f"{name} must be nonempty UTF-8 NFC path text")
    if (
        "\\" in value
        or ":" in value
        or value.endswith("/")
        or any(ord(character) < 32 for character in value)
    ):
        raise FinalVerificationError(f"{name} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise FinalVerificationError(f"{name} must be a canonical repository-relative path")
    return value


def _stable_file_bytes(path: Path, *, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise FinalVerificationError(f"{name} must be an existing non-symlink file")
    try:
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise FinalVerificationError(f"unable to read {name}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(data) != after.st_size:
        raise FinalVerificationError(f"{name} changed while being read")
    return data


def _canonical_record_bytes(path: Path, *, name: str) -> tuple[bytes, dict[str, Any]]:
    data = _stable_file_bytes(path, name=name)
    try:
        return data, strict_canonical_json_loads(data)
    except CanonicalRecordError as exc:
        raise FinalVerificationError(f"invalid {name}: {exc}") from exc


def _validate_json_reference(
    value: object,
    *,
    name: str,
    target_data: bytes,
    target_record: Mapping[str, Any],
    expected_schema: str,
    expected_relative_path: str | None = None,
) -> None:
    reference = _mapping(value, _JSON_REFERENCE_KEYS, name=name)
    relative_path = _canonical_relative_path(reference["relative_path"], name=f"{name}.relative_path")
    if expected_relative_path is not None and relative_path != expected_relative_path:
        raise FinalVerificationError(f"{name} relative path mismatch")
    if _integer(reference["size"], name=f"{name}.size") != len(target_data):
        raise FinalVerificationError(f"{name} size mismatch")
    if _sha256(reference["sha256"], name=f"{name}.sha256") != sha256_bytes(target_data):
        raise FinalVerificationError(f"{name} file hash mismatch")
    if reference["schema"] != expected_schema or target_record.get("schema") != expected_schema:
        raise FinalVerificationError(f"{name} target schema mismatch")


def _attempt_id(value: object, *, name: str = "attempt_id") -> str:
    if not isinstance(value, str) or re.fullmatch(r"s4-[0-9a-f]{24}", value) is None:
        raise FinalVerificationError(f"{name} must use the frozen State 4 attempt format")
    return value


def _hash_list(value: object, *, name: str, count: int) -> list[str]:
    if not isinstance(value, list) or len(value) != count:
        raise FinalVerificationError(f"{name} must contain exactly {count} hashes")
    return [_sha256(item, name=f"{name}[{index}]") for index, item in enumerate(value)]


def _exact_split_map(
    value: object, *, name: str, expected: Mapping[str, int]
) -> Mapping[str, Any]:
    mapping = _mapping(value, set(_SPLIT_ORDER), name=name)
    for split in _SPLIT_ORDER:
        if _integer(mapping[split], name=f"{name}.{split}") != expected[split]:
            raise FinalVerificationError(f"{name}.{split} mismatch")
    return mapping


def _validate_git_record(
    value: object,
    *,
    name: str,
    implementation_commit: str,
    implementation_tree: str,
) -> Mapping[str, Any]:
    record = _mapping(value, _GIT_RECORD_KEYS, name=name)
    if record["branch"] != "main":
        raise FinalVerificationError(f"{name}.branch must be main")
    if (
        _git_oid(record["head"], name=f"{name}.head") != implementation_commit
        or _git_oid(record["tree"], name=f"{name}.tree") != implementation_tree
    ):
        raise FinalVerificationError(f"{name} implementation identity mismatch")
    parents = record["parents"]
    if not isinstance(parents, list) or not parents:
        raise FinalVerificationError(f"{name}.parents must be a non-empty array")
    checked_parents = [
        _git_oid(parent, name=f"{name}.parents[{index}]")
        for index, parent in enumerate(parents)
    ]
    if len(set(checked_parents)) != len(checked_parents):
        raise FinalVerificationError(f"{name}.parents must be unique")
    if record["status_short_untracked_all"] != "":
        raise FinalVerificationError(f"{name} must have empty untracked-inclusive status")
    for prefix in ("tracked", "staged"):
        if record[f"{prefix}_diff_sha256"] != _EMPTY_SHA256:
            raise FinalVerificationError(
                f"{name}.{prefix}_diff_sha256 must bind an empty diff"
            )
        if _integer(
            record[f"{prefix}_diff_size_bytes"],
            name=f"{name}.{prefix}_diff_size_bytes",
        ) != 0:
            raise FinalVerificationError(f"{name}.{prefix}_diff_size_bytes must be zero")
        _true(record[f"{prefix}_changes_empty"], name=f"{name}.{prefix}_changes_empty")
    _true(
        record["untracked_inclusive_status_empty"],
        name=f"{name}.untracked_inclusive_status_empty",
    )
    return record


def _validate_pre_solve_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest = _mapping(record, _PRE_SOLVE_KEYS, name="State 4 pre-solve manifest")
    if (
        manifest["schema"] != _PRE_SOLVE_SCHEMA
        or manifest["benchmark"] != _BENCHMARK
        or manifest["corpus_version"] != _CORPUS_VERSION
        or manifest["state"] != "controls_prepared_no_pde_solve"
    ):
        raise FinalVerificationError("State 4 pre-solve identity mismatch")
    authorization_sha256 = _sha256(
        manifest["state4_authorization_file_sha256"],
        name="pre_solve.state4_authorization_file_sha256",
    )
    attempt = _attempt_id(manifest["attempt_id"], name="pre_solve.attempt_id")
    if attempt != f"s4-{authorization_sha256[:24]}":
        raise FinalVerificationError("State 4 pre-solve attempt/authorization mismatch")
    if (
        _canonical_relative_path(
            manifest["protocol_relative_name"], name="pre_solve.protocol_relative_name"
        )
        != _PROTOCOL_RELATIVE_NAME
        or manifest["protocol_file_sha256"] != _PROTOCOL_SHA256
    ):
        raise FinalVerificationError("State 4 pre-solve protocol binding mismatch")
    if (
        manifest["solver_spec_baseline_commit"] != _SOLVER_BASELINE_COMMIT
        or manifest["solver_spec_baseline_tree"] != _SOLVER_BASELINE_TREE
    ):
        raise FinalVerificationError("State 4 pre-solve solver baseline mismatch")

    implementation_commit = _git_oid(
        manifest["implementation_commit"], name="pre_solve.implementation_commit"
    )
    implementation_tree = _git_oid(
        manifest["implementation_tree"], name="pre_solve.implementation_tree"
    )
    for field in (
        "protocol_file_sha256",
        "state3_audit_file_sha256",
        "compute_feasibility_report_file_sha256",
        "environment_record_sha256",
        "storage_preflight_before_file_sha256",
        "storage_preflight_after_file_sha256",
        "rng_audit_file_sha256",
        "read_only_evidence_file_sha256",
        "global_control_aggregate_sha256",
        "global_macro_coefficient_aggregate_sha256",
        "manifest_self_sha256",
    ):
        _sha256(manifest[field], name=f"pre_solve.{field}")
    if (
        _canonical_relative_path(
            manifest["rng_audit_relative_path"], name="pre_solve.rng_audit_relative_path"
        )
        != "state4_controls/rng_audit.json"
        or _canonical_relative_path(
            manifest["read_only_evidence_relative_path"],
            name="pre_solve.read_only_evidence_relative_path",
        )
        != "state4_controls/read_only_evidence.json"
    ):
        raise FinalVerificationError("State 4 pre-solve evidence path mismatch")
    if manifest["split_order"] != list(_SPLIT_ORDER):
        raise FinalVerificationError("State 4 pre-solve split order mismatch")
    expected_split_paths = [
        f"state4_controls/{_SPLIT_DIRECTORIES[split]}/split_manifest.json"
        for split in _SPLIT_ORDER
    ]
    split_paths = manifest["split_manifest_paths"]
    if not isinstance(split_paths, list) or len(split_paths) != len(_SPLIT_ORDER):
        raise FinalVerificationError("State 4 pre-solve split paths length mismatch")
    checked_paths = [
        _canonical_relative_path(path, name=f"pre_solve.split_manifest_paths[{index}]")
        for index, path in enumerate(split_paths)
    ]
    if checked_paths != expected_split_paths:
        raise FinalVerificationError("State 4 pre-solve split paths mismatch")
    _hash_list(
        manifest["split_manifest_file_sha256"],
        name="pre_solve.split_manifest_file_sha256",
        count=len(_SPLIT_ORDER),
    )
    if (
        _integer(manifest["sample_count"], name="pre_solve.sample_count") != 4864
        or _integer(manifest["shard_count"], name="pre_solve.shard_count") != 76
    ):
        raise FinalVerificationError("State 4 pre-solve corpus counts mismatch")
    _exact_split_map(
        manifest["rng_call_count_by_split"],
        name="pre_solve.rng_call_count_by_split",
        expected=_SPLIT_COUNTS,
    )
    if _integer(
        manifest["forbidden_seed_or_data_access_count"],
        name="pre_solve.forbidden_seed_or_data_access_count",
    ) != 0:
        raise FinalVerificationError("State 4 pre-solve forbidden-access count mismatch")
    _true(manifest["all_control_hashes_unique"], name="pre_solve.all_control_hashes_unique")
    _true(
        manifest["all_macro_coefficient_hashes_unique"],
        name="pre_solve.all_macro_coefficient_hashes_unique",
    )
    git_before = _validate_git_record(
        manifest["git_before"],
        name="pre_solve.git_before",
        implementation_commit=implementation_commit,
        implementation_tree=implementation_tree,
    )
    git_after = _validate_git_record(
        manifest["git_after"],
        name="pre_solve.git_after",
        implementation_commit=implementation_commit,
        implementation_tree=implementation_tree,
    )
    if git_before != git_after:
        raise FinalVerificationError("State 4 pre-solve Git records differ")
    _self_hash(manifest, "manifest_self_sha256", name="State 4 pre-solve manifest")
    return manifest


def _validate_completion_record(
    record: Mapping[str, Any],
    *,
    pre_solve_data: bytes,
    pre_solve_record: Mapping[str, Any],
) -> Mapping[str, Any]:
    completion = _mapping(record, _COMPLETE_KEYS, name="State 4 completion marker")
    if (
        completion["schema"] != _COMPLETE_SCHEMA
        or completion["state"] != "state4_controls_accepted_no_solutions"
    ):
        raise FinalVerificationError("State 4 completion identity mismatch")
    authorization_sha256 = _sha256(
        completion["state4_authorization_file_sha256"],
        name="completion.state4_authorization_file_sha256",
    )
    attempt = _attempt_id(completion["attempt_id"], name="completion.attempt_id")
    if attempt != f"s4-{authorization_sha256[:24]}":
        raise FinalVerificationError("State 4 completion attempt/authorization mismatch")
    if (
        _canonical_relative_path(
            completion["pre_solve_manifest_relative_path"],
            name="completion.pre_solve_manifest_relative_path",
        )
        != "state4_controls/pre_solve_manifest.json"
    ):
        raise FinalVerificationError("State 4 completion pre-solve path mismatch")
    if _sha256(
        completion["pre_solve_manifest_file_sha256"],
        name="completion.pre_solve_manifest_file_sha256",
    ) != sha256_bytes(pre_solve_data):
        raise FinalVerificationError("State 4 completion pre-solve hash mismatch")
    _sha256(
        completion["read_only_evidence_file_sha256"],
        name="completion.read_only_evidence_file_sha256",
    )
    timestamp = completion["completion_time_utc"]
    if not isinstance(timestamp, str) or len(timestamp) != 27:
        raise FinalVerificationError("State 4 completion timestamp format mismatch")
    try:
        parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise FinalVerificationError("State 4 completion timestamp format mismatch") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != timestamp:
        raise FinalVerificationError("State 4 completion timestamp format mismatch")
    if (
        completion["attempt_id"] != pre_solve_record["attempt_id"]
        or completion["state4_authorization_file_sha256"]
        != pre_solve_record["state4_authorization_file_sha256"]
        or completion["read_only_evidence_file_sha256"]
        != pre_solve_record["read_only_evidence_file_sha256"]
    ):
        raise FinalVerificationError("State 4 completion/pre-solve binding mismatch")
    _self_hash(completion, "record_self_sha256", name="State 4 completion marker")
    return completion


def _validate_final_context_record(
    record: Mapping[str, Any],
    *,
    pre_solve_data: bytes,
    pre_solve_record: Mapping[str, Any],
    completion_data: bytes,
    completion_record: Mapping[str, Any],
) -> Mapping[str, Any]:
    context = _mapping(record, _FINAL_CONTEXT_KEYS, name="final context")
    if (
        context["schema"] != _FINAL_CONTEXT_SCHEMA
        or context["benchmark"] != _BENCHMARK
        or context["corpus_version"] != _CORPUS_VERSION
    ):
        raise FinalVerificationError("final context identity mismatch")
    _attempt_id(context["attempt_id"], name="final_context.attempt_id")
    if context["protocol_file_sha256"] != _PROTOCOL_SHA256:
        raise FinalVerificationError("final context protocol binding mismatch")
    if (
        context["solver_spec_baseline_commit"] != _SOLVER_BASELINE_COMMIT
        or context["solver_spec_baseline_tree"] != _SOLVER_BASELINE_TREE
    ):
        raise FinalVerificationError("final context solver baseline mismatch")
    for field in (
        "solver_spec_baseline_commit",
        "solver_spec_baseline_tree",
        "implementation_commit",
        "implementation_tree",
    ):
        _git_oid(context[field], name=f"final_context.{field}")
    for field in (
        "protocol_file_sha256",
        "state4_authorization_file_sha256",
        "state5_authorization_file_sha256",
        "state6_authorization_file_sha256",
        "record_self_sha256",
    ):
        _sha256(context[field], name=f"final_context.{field}")
    if context["split_order"] != list(_SPLIT_ORDER):
        raise FinalVerificationError("final context split order mismatch")
    _validate_json_reference(
        context["pre_solve_manifest"],
        name="final_context.pre_solve_manifest",
        target_data=pre_solve_data,
        target_record=pre_solve_record,
        expected_schema=_PRE_SOLVE_SCHEMA,
        expected_relative_path="state4_controls/pre_solve_manifest.json",
    )
    _validate_json_reference(
        context["controls_completion"],
        name="final_context.controls_completion",
        target_data=completion_data,
        target_record=completion_record,
        expected_schema=_COMPLETE_SCHEMA,
        expected_relative_path="STATE4_CONTROLS_COMPLETE.json",
    )
    for field in (
        "attempt_id",
        "protocol_file_sha256",
        "solver_spec_baseline_commit",
        "solver_spec_baseline_tree",
        "implementation_commit",
        "implementation_tree",
        "state4_authorization_file_sha256",
    ):
        if context[field] != pre_solve_record[field]:
            raise FinalVerificationError(f"final context/pre-solve {field} mismatch")
    if (
        context["attempt_id"] != completion_record["attempt_id"]
        or context["state4_authorization_file_sha256"]
        != completion_record["state4_authorization_file_sha256"]
    ):
        raise FinalVerificationError("final context/completion binding mismatch")
    _self_hash(context, "record_self_sha256", name="final context")
    return context


def _validate_control_shard_manifest(
    record: Mapping[str, Any],
    *,
    split: str,
    shard: int,
    rho_data: bytes,
    rho_array: Any,
    controls_data: bytes,
    controls_records: Sequence[Mapping[str, Any]],
) -> None:
    manifest = _mapping(record, _CONTROL_SHARD_KEYS, name="State 4 controls-shard manifest")
    if manifest["schema"] != _CONTROL_SHARD_SCHEMA or manifest["benchmark"] != _BENCHMARK:
        raise FinalVerificationError("State 4 controls-shard identity mismatch")
    if manifest["split"] != split or manifest["shard_index"] != shard:
        raise FinalVerificationError("State 4 controls-shard split/index mismatch")
    _integer(manifest["shard_index"], name="State4.shard_index")
    if manifest["sealed"] is not (split == "test"):
        raise FinalVerificationError("State 4 controls-shard sealed flag mismatch")
    first_local = shard * 64
    first_global = _SPLIT_OFFSETS[split] + first_local
    fixed_integers = {
        "first_local_index": first_local,
        "last_local_index": first_local + 63,
        "first_global_index": first_global,
        "last_global_index": first_global + 63,
        "sample_count": 64,
    }
    for field, expected in fixed_integers.items():
        if _integer(manifest[field], name=f"State4.{field}") != expected:
            raise FinalVerificationError(f"State 4 controls-shard {field} mismatch")
    directory = _SPLIT_DIRECTORIES[split]
    prefix = f"shard_{shard:04d}"
    if manifest["rho_relative_path"] != (
        f"corpus_staging/{directory}/{prefix}/rho.float64.npy"
    ):
        raise FinalVerificationError("State 4 rho path mismatch")
    if manifest["control_records_relative_path"] != (
        f"state4_controls/{directory}/{prefix}.controls.jsonl"
    ):
        raise FinalVerificationError("State 4 controls JSONL path mismatch")
    rho_shape = manifest["rho_shape"]
    if (
        manifest["rho_dtype"] != "<f8"
        or not isinstance(rho_shape, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in rho_shape)
        or rho_shape != [64, 1, 8, 8]
        or manifest["rho_order"] != "C"
        or manifest["npy_version"] != "2.0"
    ):
        raise FinalVerificationError("State 4 rho metadata mismatch")
    for field in ("rho_file_bytes", "control_records_bytes"):
        _integer(manifest[field], name=f"State4.{field}")
    if manifest["rho_file_bytes"] != len(rho_data):
        raise FinalVerificationError("State 4 rho byte count mismatch")
    if manifest["control_records_bytes"] != len(controls_data):
        raise FinalVerificationError("State 4 controls byte count mismatch")
    if _sha256(manifest["rho_file_sha256"], name="State4.rho_file_sha256") != sha256_bytes(rho_data):
        raise FinalVerificationError("State 4 rho file hash mismatch")
    if _sha256(manifest["rho_tensor_sha256"], name="State4.rho_tensor_sha256") != tensor_sha256(rho_array):
        raise FinalVerificationError("State 4 rho tensor hash mismatch")
    if _sha256(
        manifest["control_records_file_sha256"], name="State4.control_records_file_sha256"
    ) != sha256_bytes(controls_data):
        raise FinalVerificationError("State 4 controls file hash mismatch")
    row_hashes = manifest["ordered_record_self_sha256"]
    if not isinstance(row_hashes, list) or len(row_hashes) != 64:
        raise FinalVerificationError("State 4 ordered row hashes must contain 64 entries")
    expected_row_hashes = [record["record_self_sha256"] for record in controls_records]
    for index, item in enumerate(row_hashes):
        _sha256(item, name=f"State4.ordered_record_self_sha256[{index}]")
    if row_hashes != expected_row_hashes:
        raise FinalVerificationError("State 4 controls row hashes/order mismatch")
    if manifest["protocol_file_sha256"] != _PROTOCOL_SHA256:
        raise FinalVerificationError("State 4 protocol binding mismatch")
    for field in (
        "state4_authorization_file_sha256",
        "environment_record_sha256",
        "manifest_self_sha256",
    ):
        _sha256(manifest[field], name=f"State4.{field}")
    for field in ("implementation_commit", "implementation_tree"):
        _git_oid(manifest[field], name=f"State4.{field}")
    _self_hash(manifest, "manifest_self_sha256", name="State 4 controls-shard manifest")


def _derive_area_view(np: Any, source: Any, *, target_size: int) -> Any:
    if target_size == 32:
        width = 2
        factor = np.float64(0.25)
    elif target_size == 16:
        width = 4
        factor = np.float64(0.0625)
    else:  # pragma: no cover - private fixed call sites
        raise AssertionError(target_size)
    result = np.empty((1, target_size, target_size), dtype=np.dtype("<f8"), order="C")
    for j in range(target_size):
        for i in range(target_size):
            if width == 2:
                total = np.float64(source[0, 2 * j, 2 * i])
                total = np.float64(total + source[0, 2 * j, 2 * i + 1])
                total = np.float64(total + source[0, 2 * j + 1, 2 * i])
                total = np.float64(total + source[0, 2 * j + 1, 2 * i + 1])
            else:
                total = np.float64(0.0)
                for a in range(4):
                    for b in range(4):
                        total = np.float64(total + source[0, 4 * j + a, 4 * i + b])
            result[0, j, i] = np.float64(total * factor)
    return result


def _exact_tensor_equal(left: Any, right: Any) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.flags.c_contiguous
        and right.flags.c_contiguous
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _frozen_compliance_64(np: Any, field: Any) -> float:
    if (
        not isinstance(field, np.ndarray)
        or np.ma.isMaskedArray(field)
        or field.shape != (64, 64)
        or field.dtype != np.dtype("<f8")
        or not field.flags.c_contiguous
        or not bool(np.isfinite(field).all())
    ):
        raise FinalVerificationError(
            "stored T64 field must be finite canonical <f8 C-order (64,64)"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.sum(field, dtype=np.float64) / np.float64(64**2))
    if not math.isfinite(value):
        raise FinalVerificationError("stored T64 compliance must be finite")
    return value


def _control_and_family_correspondence(
    np: Any,
    *,
    arrays: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
) -> None:
    observed_control_hashes: set[str] = set()
    observed_macro_hashes: set[str] = set()
    for sample in range(64):
        control_row = controls[sample]
        final_row = final_rows[sample]
        for field in (
            "split",
            "shard_index",
            "local_index",
            "global_index",
            "sample_id",
            "stratum",
            "control_tensor_sha256",
            "rho_storage_tensor_sha256",
            "macro_coefficient_tensor_sha256",
        ):
            if final_row[field] != control_row[field]:
                raise FinalVerificationError(f"final/control row mismatch at {sample}: {field}")

        storage = arrays["rho"][sample]
        control = storage[0]
        if control_row["stratum"] == "relaxed":
            if (
                bool(np.signbit(control).any())
                or not bool(np.all(control >= np.float64(0.0)))
                or not bool(np.all(control < np.float64(1.0)))
            ):
                raise FinalVerificationError(f"relaxed rho domain mismatch at row {sample}")
        else:
            bits = control.view(np.dtype("<u8"))
            exact_zero = np.uint64(0x0000000000000000)
            exact_one = np.uint64(0x3FF0000000000000)
            zeros = int(np.count_nonzero(bits == exact_zero))
            ones = int(np.count_nonzero(bits == exact_one))
            if (
                zeros != 32
                or ones != 32
                or not bool(np.all((bits == exact_zero) | (bits == exact_one)))
            ):
                raise FinalVerificationError(f"binary rho domain mismatch at row {sample}")

        macro = np.empty((8, 8), dtype=np.dtype("<f8"), order="C")
        for j in range(8):
            for i in range(8):
                product = np.float64(np.float64(9.0) * control[j, i])
                macro[j, i] = np.float64(np.float64(3.0) + product)
        actual_control_hash = tensor_sha256(control)
        actual_storage_hash = tensor_sha256(storage)
        actual_macro_hash = tensor_sha256(macro)
        if actual_control_hash in observed_control_hashes:
            raise FinalVerificationError(f"duplicate scientific control at row {sample}")
        if actual_macro_hash in observed_macro_hashes:
            raise FinalVerificationError(f"duplicate macro coefficient at row {sample}")
        observed_control_hashes.add(actual_control_hash)
        observed_macro_hashes.add(actual_macro_hash)
        if control_row["control_tensor_sha256"] != actual_control_hash:
            raise FinalVerificationError(f"scientific control hash mismatch at row {sample}")
        if control_row["rho_storage_tensor_sha256"] != actual_storage_hash:
            raise FinalVerificationError(f"rho storage hash mismatch at row {sample}")
        if control_row["macro_coefficient_tensor_sha256"] != actual_macro_hash:
            raise FinalVerificationError(f"macro coefficient hash mismatch at row {sample}")

        expanded = np.repeat(np.repeat(macro, 8, axis=0), 8, axis=1)[None, :, :]
        expanded = np.ascontiguousarray(expanded, dtype=np.dtype("<f8"))
        if not _exact_tensor_equal(arrays["kappa64"][sample], expanded):
            raise FinalVerificationError(f"kappa64 expansion mismatch at row {sample}")
        expected_views = {
            "kappa32_areaavg": _derive_area_view(
                np, arrays["kappa64"][sample], target_size=32
            ),
            "kappa16_areaavg": _derive_area_view(
                np, arrays["kappa64"][sample], target_size=16
            ),
            "u32_areaavg": _derive_area_view(
                np, arrays["u_tpfa64"][sample], target_size=32
            ),
            "u16_areaavg": _derive_area_view(
                np, arrays["u_tpfa64"][sample], target_size=16
            ),
        }
        for family, expected in expected_views.items():
            if not _exact_tensor_equal(arrays[family][sample], expected):
                raise FinalVerificationError(f"{family} derived-view mismatch at row {sample}")

        stored_t64_compliance = _frozen_compliance_64(
            np, arrays["u_tpfa64"][sample, 0]
        )
        recorded_t64_compliance = final_row["metrics"]["compliances"]["T64"]
        if recorded_t64_compliance != stored_t64_compliance.hex().lower():
            raise FinalVerificationError(
                f"stored T64 compliance mismatch at row {sample}"
            )

        tensor_hashes = final_row["tensor_sha256"]
        for family in _FAMILY_NAMES:
            actual = tensor_sha256(arrays[family][sample])
            if tensor_hashes[family] != actual:
                raise FinalVerificationError(
                    f"final row tensor hash mismatch at {sample}: {family}"
                )


def _validate_final_shard_manifest(
    record: Mapping[str, Any],
    *,
    split: str,
    shard: int,
    member_data: Mapping[str, bytes],
    arrays: Mapping[str, Any],
    final_rows: Sequence[Mapping[str, Any]],
    context_data: bytes,
    context_record: Mapping[str, Any],
    pre_solve_data: bytes,
    pre_solve_record: Mapping[str, Any],
    state4_manifest_data: bytes,
    state4_manifest_record: Mapping[str, Any],
) -> None:
    manifest = _mapping(record, _FINAL_SHARD_KEYS, name="final shard manifest")
    if manifest["schema"] != _FINAL_SHARD_SCHEMA or manifest["benchmark"] != _BENCHMARK:
        raise FinalVerificationError("final shard manifest identity mismatch")
    if manifest["split"] != split or manifest["shard_index"] != shard:
        raise FinalVerificationError("final shard manifest split/index mismatch")
    _integer(manifest["shard_index"], name="final_manifest.shard_index")
    first_local = shard * 64
    first_global = _SPLIT_OFFSETS[split] + first_local
    fixed = {
        "sample_count": 64,
        "first_local_index": first_local,
        "last_local_index": first_local + 63,
        "first_global_index": first_global,
        "last_global_index": first_global + 63,
    }
    for field, expected in fixed.items():
        if _integer(manifest[field], name=f"final_manifest.{field}") != expected:
            raise FinalVerificationError(f"final shard manifest {field} mismatch")
    expected_ids = [
        f"{_BENCHMARK}/{split}/{local:06d}"
        for local in range(first_local, first_local + 64)
    ]
    if manifest["ordered_sample_ids"] != expected_ids:
        raise FinalVerificationError("final shard manifest sample IDs/order mismatch")

    directory = _SPLIT_DIRECTORIES[split]
    shard_name = f"shard_{shard:04d}"
    _validate_json_reference(
        manifest["parent_context"],
        name="parent_context",
        target_data=context_data,
        target_record=context_record,
        expected_schema=_FINAL_CONTEXT_SCHEMA,
    )
    _validate_json_reference(
        manifest["pre_solve_manifest"],
        name="pre_solve_manifest",
        target_data=pre_solve_data,
        target_record=pre_solve_record,
        expected_schema=_PRE_SOLVE_SCHEMA,
        expected_relative_path="state4_controls/pre_solve_manifest.json",
    )
    _validate_json_reference(
        manifest["state4_controls_shard_manifest"],
        name="state4_controls_shard_manifest",
        target_data=state4_manifest_data,
        target_record=state4_manifest_record,
        expected_schema=_CONTROL_SHARD_SCHEMA,
        expected_relative_path=(
            f"state4_controls/{directory}/{shard_name}.manifest.json"
        ),
    )

    members = manifest["members"]
    if not isinstance(members, list) or len(members) != 8:
        raise FinalVerificationError("final shard manifest must contain eight ordered members")
    for position, (family, filename, _sample_shape, shard_shape) in enumerate(_FAMILIES):
        member = _mapping(members[position], _NPY_MEMBER_KEYS, name=f"members[{position}]")
        expected_path = f"corpus_staging/{directory}/{shard_name}/{filename}"
        if member["relative_path"] != expected_path:
            raise FinalVerificationError(f"{family} member path/order mismatch")
        if member["tensor_name"] != family:
            raise FinalVerificationError(f"{family} tensor name mismatch")
        shape = member["shape"]
        version = member["npy_version"]
        if (
            member["dtype"] != "<f8"
            or not isinstance(shape, list)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
            or shape != list(shard_shape)
            or member["order"] != "C"
            or not isinstance(version, list)
            or len(version) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in version)
            or version != [2, 0]
        ):
            raise FinalVerificationError(f"{family} member metadata mismatch")
        data = member_data[filename]
        if _integer(member["size"], name=f"{family}.size") != len(data):
            raise FinalVerificationError(f"{family} member size mismatch")
        if _sha256(member["sha256"], name=f"{family}.sha256") != sha256_bytes(data):
            raise FinalVerificationError(f"{family} member file hash mismatch")
        if _sha256(member["tensor_sha256"], name=f"{family}.tensor_sha256") != tensor_sha256(
            arrays[family]
        ):
            raise FinalVerificationError(f"{family} whole-shard tensor hash mismatch")

    jsonl_member = _mapping(members[7], _JSONL_MEMBER_KEYS, name="members[7]")
    expected_jsonl_path = (
        f"corpus_staging/{directory}/{shard_name}/verification.jsonl"
    )
    if jsonl_member["relative_path"] != expected_jsonl_path:
        raise FinalVerificationError("verification JSONL member path/order mismatch")
    if jsonl_member["schema"] != _FINAL_RECORD_SCHEMA:
        raise FinalVerificationError("verification JSONL member schema mismatch")
    jsonl_data = member_data["verification.jsonl"]
    if _integer(jsonl_member["size"], name="verification.size") != len(jsonl_data):
        raise FinalVerificationError("verification JSONL member size mismatch")
    if _sha256(jsonl_member["sha256"], name="verification.sha256") != sha256_bytes(jsonl_data):
        raise FinalVerificationError("verification JSONL member file hash mismatch")
    if _integer(jsonl_member["record_count"], name="verification.record_count") != 64:
        raise FinalVerificationError("verification JSONL record count mismatch")
    row_hashes = jsonl_member["ordered_record_self_sha256"]
    expected_hashes = [row["record_self_sha256"] for row in final_rows]
    if not isinstance(row_hashes, list) or len(row_hashes) != 64:
        raise FinalVerificationError("verification ordered hashes must contain 64 entries")
    for index, value in enumerate(row_hashes):
        _sha256(value, name=f"verification.ordered_record_self_sha256[{index}]")
    if row_hashes != expected_hashes:
        raise FinalVerificationError("verification row hashes/order mismatch")
    _self_hash(manifest, "manifest_self_sha256", name="final shard manifest")


def validate_final_shard_bundle(
    shard_directory: str | os.PathLike[str],
    *,
    split: str,
    shard_index: int,
    controls_jsonl_path: str | os.PathLike[str],
    accepted_rho_path: str | os.PathLike[str],
    state4_controls_shard_manifest_path: str | os.PathLike[str],
    parent_context_path: str | os.PathLike[str],
    pre_solve_manifest_path: str | os.PathLike[str],
    controls_completion_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate one final shard against actual members and predecessor bytes.

    All paths must identify already-created files.  In particular,
    ``accepted_rho_path`` must be the same filesystem file as the shard's rho
    member, not a replacement containing merely equal values.  This validates
    the supplied predecessor records and their byte commitments.  The caller's
    designation of the accepted controls-shard manifest/rho remains a trusted
    input because this later-state verifier does not open the seed-bearing
    State 4 split manifest that transitively commits it.  That accepted-path
    provenance, the unresolved nested permission proof, and whole-corpus
    publication remain external production prerequisites.
    """

    # NumPy is imported only when this explicitly invoked tensor operation runs.
    import numpy as np

    split_name, shard = _split_and_shard(split, shard_index)
    root = Path(shard_directory)
    if root.is_symlink() or not root.is_dir():
        raise FinalVerificationError("shard_directory must be an existing non-symlink directory")
    expected_manifest_path = root / "shard_manifest.json"
    supplied_manifest_path = (
        expected_manifest_path if manifest_path is None else Path(manifest_path)
    )
    try:
        if not supplied_manifest_path.samefile(expected_manifest_path):
            raise FinalVerificationError("manifest_path must name this shard's shard_manifest.json")
    except OSError as exc:
        raise FinalVerificationError("final shard manifest is unavailable") from exc

    expected_names = {item[1] for item in _FAMILIES} | {
        "verification.jsonl",
        "shard_manifest.json",
    }
    try:
        children = list(root.iterdir())
    except OSError as exc:
        raise FinalVerificationError("unable to enumerate final shard") from exc
    actual_names = {child.name for child in children}
    if actual_names != expected_names or any(child.is_symlink() or not child.is_file() for child in children):
        raise FinalVerificationError(
            f"final shard members mismatch; missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    rho_path = root / "rho.float64.npy"
    accepted_rho = Path(accepted_rho_path)
    try:
        if accepted_rho.is_symlink() or not accepted_rho.samefile(rho_path):
            raise FinalVerificationError("rho member is not the accepted State 4 file")
    except OSError as exc:
        raise FinalVerificationError("accepted rho file is unavailable") from exc

    member_data: dict[str, bytes] = {}
    arrays: dict[str, Any] = {}
    for family, filename, _sample_shape, shard_shape in _FAMILIES:
        data = _stable_file_bytes(root / filename, name=filename)
        member_data[filename] = data
        try:
            arrays[family] = validate_npy_v2_bytes(data, expected_shape=shard_shape)
        except CanonicalRecordError as exc:
            raise FinalVerificationError(f"invalid {filename}: {exc}") from exc
    if member_data["rho.float64.npy"] != _stable_file_bytes(
        accepted_rho, name="accepted State 4 rho"
    ):
        raise FinalVerificationError("rho bytes differ from accepted State 4 rho")

    controls_path = Path(controls_jsonl_path)
    controls_data = _stable_file_bytes(controls_path, name="State 4 controls JSONL")
    try:
        controls_records = strict_canonical_jsonl_loads(controls_data)
    except CanonicalRecordError as exc:
        raise FinalVerificationError(f"invalid State 4 controls JSONL: {exc}") from exc
    if len(controls_records) != 64:
        raise FinalVerificationError("State 4 controls JSONL must contain 64 rows")
    for offset, record in enumerate(controls_records):
        _validate_control_record(record, split=split_name, shard=shard, offset=offset)

    state4_manifest_data, state4_manifest = _canonical_record_bytes(
        Path(state4_controls_shard_manifest_path), name="State 4 controls-shard manifest"
    )
    _validate_control_shard_manifest(
        state4_manifest,
        split=split_name,
        shard=shard,
        rho_data=member_data["rho.float64.npy"],
        rho_array=arrays["rho"],
        controls_data=controls_data,
        controls_records=controls_records,
    )

    final_data = _stable_file_bytes(root / "verification.jsonl", name="verification.jsonl")
    member_data["verification.jsonl"] = final_data
    final_rows = validate_final_verification_jsonl(
        final_data, split=split_name, shard_index=shard
    )
    _control_and_family_correspondence(
        np, arrays=arrays, controls=controls_records, final_rows=final_rows
    )

    context_data, context_record = _canonical_record_bytes(
        Path(parent_context_path), name="parent final context"
    )
    pre_solve_data, pre_solve_record = _canonical_record_bytes(
        Path(pre_solve_manifest_path), name="State 4 pre-solve manifest"
    )
    completion_data, completion_record = _canonical_record_bytes(
        Path(controls_completion_path), name="State 4 completion marker"
    )
    _validate_pre_solve_record(pre_solve_record)
    _validate_completion_record(
        completion_record,
        pre_solve_data=pre_solve_data,
        pre_solve_record=pre_solve_record,
    )
    _validate_final_context_record(
        context_record,
        pre_solve_data=pre_solve_data,
        pre_solve_record=pre_solve_record,
        completion_data=completion_data,
        completion_record=completion_record,
    )
    for field in (
        "protocol_file_sha256",
        "state4_authorization_file_sha256",
        "implementation_commit",
        "implementation_tree",
        "environment_record_sha256",
    ):
        if state4_manifest[field] != pre_solve_record[field]:
            raise FinalVerificationError(
                f"State 4 controls-shard/pre-solve {field} mismatch"
            )
    manifest_data, manifest = _canonical_record_bytes(
        supplied_manifest_path, name="final shard manifest"
    )
    _validate_final_shard_manifest(
        manifest,
        split=split_name,
        shard=shard,
        member_data=member_data,
        arrays=arrays,
        final_rows=final_rows,
        context_data=context_data,
        context_record=context_record,
        pre_solve_data=pre_solve_data,
        pre_solve_record=pre_solve_record,
        state4_manifest_data=state4_manifest_data,
        state4_manifest_record=state4_manifest,
    )

    # A final stable-byte pass detects mutation during the multi-file check.
    for filename, original in member_data.items():
        if _stable_file_bytes(root / filename, name=filename) != original:
            raise FinalVerificationError(f"{filename} changed during bundle validation")
    stable_evidence = (
        (controls_path, "State 4 controls JSONL", controls_data),
        (
            Path(state4_controls_shard_manifest_path),
            "State 4 controls-shard manifest",
            state4_manifest_data,
        ),
        (Path(parent_context_path), "parent final context", context_data),
        (Path(pre_solve_manifest_path), "State 4 pre-solve manifest", pre_solve_data),
        (Path(controls_completion_path), "State 4 completion marker", completion_data),
        (supplied_manifest_path, "final shard manifest", manifest_data),
    )
    for path, name, original in stable_evidence:
        if _stable_file_bytes(path, name=name) != original:
            raise FinalVerificationError(f"{name} changed during bundle validation")
    return manifest


__all__ = [
    "FinalVerificationError",
    "validate_final_shard_bundle",
    "validate_final_verification_jsonl",
]


# Batch C adds held-byte consumers without changing the legacy path API above.
from dataclasses import dataclass
import hashlib
from . import final_scientific_verify as _science


@dataclass(frozen=True)
class FileExpectation:
    role: str
    file: _science.FileRef
    custody_identity: str
    admission: _science.Admission


@dataclass(frozen=True)
class AcquiredFile:
    file: _science.FileRef
    data: bytes | None
    custody_identity: str | None
    capture: _science.Capture | None
    admission: _science.Admission | None
    state: str


@dataclass(frozen=True)
class InventoryExpectation:
    root_binding_id: str
    relative_path: str
    files: tuple
    admission: _science.Admission


@dataclass(frozen=True)
class AcquiredInventory:
    files: tuple
    capture: _science.Capture | None
    admission: _science.Admission | None
    state: str


@dataclass(frozen=True)
class ShardExpectations(_science._Frozen):
    phase: str
    split: str
    shard_index: int
    files: tuple
    inventory: InventoryExpectation
    scientific: _science.RetainedShardExpectations
    bindings: object


@dataclass(frozen=True)
class SplitExpectations(_science._Frozen):
    split: str
    shards: tuple
    manifest: FileExpectation
    opaque_seed_manifest: _science.FileRef
    inventory: InventoryExpectation


@dataclass(frozen=True)
class CorpusExpectations(_science._Frozen):
    splits: tuple
    manifest: FileExpectation
    inventory: InventoryExpectation
    metadata_expected: object


@dataclass(frozen=True)
class RetainedScientificInput:
    split: str
    shard_index: int
    independent_verification_bytes: bytes | None
    acquisition_evidence: _science.EvidenceAcquisition | None
    admissions: tuple


def _expected_file(value):
    _science._need(type(value) is FileExpectation, "file_expectation_type")
    _science._text(value.role)
    _science.validate_file_ref(value.file)
    _science._text(value.custody_identity)
    _science.validate_admission(value.admission)
    _science._need(value.admission.outcome == "accepted" and value.admission.scope == "custody", "custody_expectation")
    binding = canonical_json_bytes(dict(file=_science.plain(value.file), custody_identity=value.custody_identity))
    _science._need(value.admission.subject_sha256 == sha256_bytes(binding), "custody_subject_binding")


def _acquire(reader, expected, diagnostics):
    _expected_file(expected)
    role = expected.role
    try:
        actual = reader.read(expected.file)
    except Exception:
        diagnostics.append(("incomplete", role, "read_unavailable"))
        return None
    if type(actual) is not AcquiredFile:
        diagnostics.append(("invalid", role, "acquisition_type"))
        return None
    if actual.state not in ("present", "unavailable"):
        diagnostics.append(("invalid", role, "acquisition_state"))
        return None
    if actual.file != expected.file or actual.custody_identity != expected.custody_identity:
        diagnostics.append(("mismatch", role, "original_object_binding"))
    _science.qualify(actual.admission, expected.admission, diagnostics, role)
    try:
        if actual.capture is None or not _science.within_capture(actual.capture, expected.admission.capture):
            diagnostics.append(("incomplete" if actual.capture is None else "mismatch", role, "acquisition_interval"))
    except _science.FinalScientificInputError:
        diagnostics.append(("invalid", role, "acquisition_interval"))
    if actual.state == "unavailable":
        diagnostics.append(("incomplete", role, "bytes_unavailable"))
        if actual.data is not None:
            diagnostics.append(("invalid", role, "unavailable_with_bytes"))
        return None
    if type(actual.data) is not bytes or len(actual.data) != expected.file.file_bytes or sha256_bytes(actual.data) != expected.file.file_sha256:
        diagnostics.append(("mismatch", role, "file_bytes_binding"))
        return None
    # Still inspect known held bytes when independent custody is unavailable.
    return actual.data


def _inventory(reader, expected, files, diagnostics):
    _science._need(type(expected) is InventoryExpectation and type(expected.files) is tuple, "inventory_expectation")
    _science._need(expected.files == tuple(files), "inventory_expected_membership")
    _science._need(len(set(files)) == len(files), "inventory_expected_duplicate")
    for file in files:
        _science.validate_file_ref(file)
    _science._need(len({(f.root_binding_id.casefold(), f.relative_path.casefold()) for f in files}) == len(files), "inventory_expected_alias")
    _science.validate_admission(expected.admission)
    _science._need(expected.admission.scope == "custody" and expected.admission.outcome == "accepted", "inventory_admission")
    binding = dict(root_binding_id=expected.root_binding_id, relative_path=expected.relative_path, files=[_science.plain(f) for f in files])
    _science._need(expected.admission.subject_sha256 == sha256_bytes(canonical_json_bytes(binding)), "inventory_subject_binding")
    try:
        actual = reader.inventory(expected.root_binding_id, expected.relative_path)
    except Exception:
        diagnostics.append(("incomplete", "inventory", "enumeration_unavailable"))
        return
    if type(actual) is not AcquiredInventory or type(actual.files) is not tuple or actual.state not in ("present", "unavailable"):
        diagnostics.append(("invalid", "inventory", "inventory_type"))
        return
    _science.qualify(actual.admission, expected.admission, diagnostics, "inventory")
    try:
        if actual.capture is None or not _science.within_capture(actual.capture, expected.admission.capture):
            diagnostics.append(("incomplete" if actual.capture is None else "mismatch", "inventory", "inventory_interval"))
    except _science.FinalScientificInputError:
        diagnostics.append(("invalid", "inventory", "inventory_interval"))
    if actual.state == "unavailable":
        diagnostics.append(("incomplete", "inventory", "inventory_unavailable"))
        if actual.files:
            diagnostics.append(("invalid", "inventory", "unavailable_inventory_nonempty"))
    elif actual.files != expected.files:
        diagnostics.append(("mismatch", "inventory", "inventory_membership_order"))


def _shard_expected(expected):
    _science._need(type(expected) is ShardExpectations, "shard_expectations")
    _split_and_shard(expected.split, expected.shard_index)
    _science._need(expected.phase in ("state5_members", "state6_manifests"), "shard_phase")
    final = expected.phase == "state6_manifests"
    roles = (*_science.FAMILIES, "verification", "controls", "controls_manifest", "pre_solve", "completion") + (("context", "manifest") if final else ())
    _science._need(tuple(f.role for f in expected.files) == roles, "shard_expected_roles")
    _science._need(len({f.custody_identity for f in expected.files}) == len(expected.files), "expected_custody_alias")
    for f in expected.files:
        _expected_file(f)
    table = {f.role:f for f in expected.files}
    root = table["rho"].file.root_binding_id
    base = f"corpus_staging/{_SPLIT_DIRECTORIES[expected.split]}/shard_{expected.shard_index:04d}"
    paths = {family:base + "/" + filename for family, filename, _, _ in _FAMILIES}
    paths.update(verification=base + "/verification.jsonl", controls=f"state4_controls/{_SPLIT_DIRECTORIES[expected.split]}/shard_{expected.shard_index:04d}.controls.jsonl",
                 controls_manifest=f"state4_controls/{_SPLIT_DIRECTORIES[expected.split]}/shard_{expected.shard_index:04d}.manifest.json",
                 pre_solve="state4_controls/pre_solve_manifest.json", completion="STATE4_CONTROLS_COMPLETE.json")
    if final:
        paths.update(context="final_context.json", manifest=base + "/shard_manifest.json")
    _science._need(all(f.file.root_binding_id == root and f.file.relative_path == paths[f.role] for f in expected.files), "shard_expected_paths")
    _science._need(expected.inventory.root_binding_id == root and expected.inventory.relative_path == base, "shard_inventory_root")
    _science._need(expected.scientific.split == expected.split and expected.scientific.shard_index == expected.shard_index
                  and expected.scientific.member_files == tuple(table[r].file for r in roles[:8]), "scientific_member_inventory")
    required_bindings = {"protocol_file_sha256", "state4_authorization_file_sha256", "implementation_commit", "implementation_tree", "environment_record_sha256"}
    if final:
        required_bindings |= {"state5_authorization_file_sha256", "state6_authorization_file_sha256"}
    _science._need(set(expected.bindings) == required_bindings, "expected_bindings")
    _science._need(expected.bindings["protocol_file_sha256"] == _PROTOCOL_SHA256, "frozen_protocol_binding")
    for key, value in expected.bindings.items():
        (_git_oid if key in ("implementation_commit", "implementation_tree") else _sha256)(value, name=key)
    return table


def _scientific_input(inputs, split, shard, diagnostics):
    _science._need(type(inputs) is tuple, "scientific_inputs_type")
    matches = [row for row in inputs if type(row) is RetainedScientificInput and row.split == split and row.shard_index == shard]
    if len(matches) != 1:
        diagnostics.append(("invalid" if matches else "incomplete", "scientific", "scientific_input_membership"))
        return None
    return matches[0]


def _held_shard(expected, reader, inputs):
    import numpy as np
    table = _shard_expected(expected)
    diagnostics = []
    members = [table[r].file for r in (*_science.FAMILIES, "verification")]
    if expected.phase == "state6_manifests":
        members.append(table["manifest"].file)
    _inventory(reader, expected.inventory, members, diagnostics)
    data = {role:_acquire(reader, item, diagnostics) for role, item in table.items()}
    scientific = _scientific_input(inputs, expected.split, expected.shard_index, diagnostics)
    assessment = _science.verify_retained_scientific_shard(expected=expected.scientific,
        candidate_verification_bytes=data["verification"], independent_verification_bytes=scientific.independent_verification_bytes if scientific else None,
        acquisition_evidence=scientific.acquisition_evidence if scientific else None, admissions=scientific.admissions if scientific else ())
    diagnostics.extend(assessment.diagnostics)
    arrays = {}
    control_hashes = ()
    recomputed = []
    def checked(label, function):
        try:
            value = function()
            recomputed.append(label)
            return value
        except (ValueError, TypeError, KeyError, OverflowError):
            diagnostics.append(("invalid", label, "shard_semantic_validation"))
            return None
    if any(value is None for value in data.values()):
        diagnostics.append(("incomplete", "shard", "semantic_inputs_unavailable"))
    for family, _, _, shape in _FAMILIES:
        if data[family] is not None:
            array = checked(family + "/npy", lambda: validate_npy_v2_bytes(data[family], expected_shape=shape))
            if array is not None:
                arrays[family] = array
    def control_records():
        controls = strict_canonical_jsonl_loads(data["controls"])
        if len(controls) != 64:
            raise FinalVerificationError("control row count")
        for offset, row in enumerate(controls):
            _validate_control_record(row, split=expected.split, shard=expected.shard_index, offset=offset)
        return controls
    controls = checked("controls", control_records) if data["controls"] is not None else None
    rows = checked("verification", lambda: validate_final_verification_jsonl(data["verification"], split=expected.split,
        shard_index=expected.shard_index)) if data["verification"] is not None else None
    def object_record(raw):
        record = strict_canonical_json_loads(raw)
        _science._need(type(record) is dict, "object_record")
        return record
    parsed = {role:checked(role + "/json", lambda: object_record(data[role]))
              for role in ("controls_manifest", "pre_solve", "completion", "context", "manifest") if data.get(role) is not None}
    cm, pre, complete, context = (parsed.get(role) for role in ("controls_manifest", "pre_solve", "completion", "context"))
    if controls is not None:
        control_hashes = tuple(tuple(row[k] for k in ("control_tensor_sha256", "rho_storage_tensor_sha256", "macro_coefficient_tensor_sha256")) for row in controls)
        if control_hashes != expected.scientific.control_hashes:
            diagnostics.append(("mismatch", "controls", "accepted_control_bindings"))
    if cm is not None and controls is not None and "rho" in arrays:
        checked("controls_manifest", lambda: _validate_control_shard_manifest(cm, split=expected.split, shard=expected.shard_index,
            rho_data=data["rho"], rho_array=arrays["rho"], controls_data=data["controls"], controls_records=controls))
    if controls is not None and rows is not None and len(arrays) == len(_FAMILIES):
        checked("original_control_correspondence", lambda: _control_and_family_correspondence(np, arrays=arrays, controls=controls, final_rows=rows))
    else:
        # Per-family facts do not depend on unrelated missing arrays or records.
        if controls is not None and rows is not None:
            for field in ("sample_id","stratum","control_tensor_sha256","rho_storage_tensor_sha256","macro_coefficient_tensor_sha256"):
                if any(left[field] != right[field] for left,right in zip(controls,rows)):
                    diagnostics.append(("mismatch", field, "available_control_record_binding"))
        if "rho" in arrays:
            def rho_checks():
                for i,storage in enumerate(arrays["rho"]):
                    control=storage[0]
                    bits=control.view("<u8")
                    valid=(not bool(np.signbit(control).any()) and bool(((control>=0)&(control<1)).all())) if i%2==0 else (
                        bool(((bits==0)|(bits==0x3ff0000000000000)).all()) and int((bits==0).sum())==32)
                    if not valid:
                        diagnostics.append(("mismatch", "rho", "available_control_stratum"))
                    macro=np.float64(3.0)+np.float64(9.0)*control
                    actual=(tensor_sha256(control),tensor_sha256(storage),tensor_sha256(macro))
                    if actual != expected.scientific.control_hashes[i]:
                        diagnostics.append(("mismatch", "rho", "available_control_hash_binding"))
                    if "kappa64" in arrays and not _exact_tensor_equal(arrays["kappa64"][i],np.repeat(np.repeat(macro,8,axis=0),8,axis=1)[None]):
                        diagnostics.append(("mismatch", "kappa64", "available_coefficient_expansion"))
            checked("available_rho", rho_checks)
        if rows is not None:
            for family,array in arrays.items():
                if any(row["tensor_sha256"][family] != tensor_sha256(array[i]) for i,row in enumerate(rows)):
                    diagnostics.append(("mismatch", family, "available_tensor_commitment"))
            if "u_tpfa64" in arrays and any(row["metrics"]["compliances"]["T64"] != _frozen_compliance_64(np,arrays["u_tpfa64"][i,0]).hex() for i,row in enumerate(rows)):
                diagnostics.append(("mismatch", "u_tpfa64", "available_stored_compliance"))
        for family,source,n in (("kappa32_areaavg","kappa64",32),("kappa16_areaavg","kappa64",16),("u32_areaavg","u_tpfa64",32),("u16_areaavg","u_tpfa64",16)):
            if family in arrays and source in arrays:
                if any(not _exact_tensor_equal(arrays[family][i],_derive_area_view(np,arrays[source][i],target_size=n)) for i in range(64)):
                    diagnostics.append(("mismatch", family, "available_derived_view"))
    if pre is not None:
        pre = checked("pre_solve", lambda: _validate_pre_solve_record(pre))
    if pre is not None and complete is not None:
        checked("completion", lambda: _validate_completion_record(complete, pre_solve_data=data["pre_solve"], pre_solve_record=pre))
    for role, record in (("controls_manifest",cm),("pre_solve",pre)):
        if record is not None:
            for key in ("protocol_file_sha256", "state4_authorization_file_sha256", "implementation_commit", "implementation_tree", "environment_record_sha256"):
                if record.get(key) != expected.bindings[key]:
                    diagnostics.append(("mismatch", role + "/" + key, "accepted_predecessor_binding"))
    if context is not None:
        for key in ("state5_authorization_file_sha256", "state6_authorization_file_sha256"):
            if context.get(key) != expected.bindings[key]:
                diagnostics.append(("mismatch", key, "accepted_authorization_binding"))
        if pre is not None and complete is not None:
            checked("context", lambda: _validate_final_context_record(context, pre_solve_data=data["pre_solve"], pre_solve_record=pre,
                completion_data=data["completion"], completion_record=complete))
    if (expected.phase == "state6_manifests" and all(value is not None for value in (parsed.get("manifest"),pre,context,cm,rows))
            and len(arrays) == len(_FAMILIES)):
        checked("manifest", lambda: _validate_final_shard_manifest(parsed["manifest"], split=expected.split, shard=expected.shard_index,
            member_data={filename:data[family] for family, filename, _, _ in _FAMILIES} | {"verification.jsonl":data["verification"]},
            arrays=arrays, final_rows=rows, context_data=data["context"], context_record=context,
            pre_solve_data=data["pre_solve"], pre_solve_record=pre, state4_manifest_data=data["controls_manifest"], state4_manifest_record=cm))
    counts = dict(assessment.counts) if not diagnostics else {key:0 for key in _science.COUNTS}
    maxima = dict(assessment.maxima) if not diagnostics else {key:None for key in _science.MAXIMA}
    return (_science.make_assessment("shard", diagnostics, recomputed=recomputed, admitted=assessment.admitted,
             counts=counts, maxima=maxima, commitments={k:tensor_sha256(v) for k,v in arrays.items()},
             details=dict(control_hashes=control_hashes, opaque_split_commitment=(dict(
                 relative_path=pre["split_manifest_paths"][_SPLIT_ORDER.index(expected.split)],
                 file_sha256=pre["split_manifest_file_sha256"][_SPLIT_ORDER.index(expected.split)]) if pre is not None else None),
                 numerical_recomputation_performed=False, terminal_acceptance=False)), arrays)


def validate_final_shard_members(*, expected, reader, scientific_evidence):
    _science._need(type(expected) is ShardExpectations, "shard_expectations")
    _science._need(type(scientific_evidence) is tuple and all(type(r) is RetainedScientificInput and r.split == expected.split
                  and r.shard_index == expected.shard_index for r in scientific_evidence), "extra_scientific_input")
    return _held_shard(expected, reader, scientific_evidence)[0]


def _tensor_streams(split):
    result = {}
    for family, _, sample_shape, _ in _FAMILIES:
        shape = (_SPLIT_COUNTS[split], *sample_shape)
        header = f"dtype=<f8\nshape={','.join(str(n) for n in shape)}\norder=C\n".encode("utf-8")
        result[family] = hashlib.sha256(header)
    return result


def _aggregate(assessments, population):
    counts = {key:sum(item.counts[key] for item in assessments) for key in _science.COUNTS}
    maxima = {key:(max(float.fromhex(item.maxima[key]) for item in assessments).hex()
                   if assessments and all(item.maxima[key] is not None for item in assessments) and counts["overall"] == population else None)
              for key in _science.MAXIMA}
    return counts, maxima


def _update_tensor_stream(stream, array):
    """One ordered held payload, exposed privately for scoped timing capture."""
    stream.update(memoryview(array).cast("B"))


def _local_ref(record, expected, schema=None):
    keys = {"relative_path", "size", "sha256"} | ({"schema"} if schema else set())
    _mapping(record, keys, name="local reference")
    _science._need(record["relative_path"] == expected.relative_path and type(record["size"]) is int
                  and record["size"] == expected.file_bytes and record["sha256"] == expected.file_sha256
                  and (schema is None or record["schema"] == schema), "local_reference_binding")


def _held_split(expected, reader, inputs):
    _science._need(type(expected) is SplitExpectations and expected.split in _SPLIT_ORDER, "split_expectations")
    count = _SPLIT_SHARDS[expected.split]
    _science._need(type(expected.shards) is tuple and len(expected.shards) == count, "expected_shard_count")
    _science._need(all(type(s) is ShardExpectations and s.split == expected.split and s.shard_index == i and s.phase == "state6_manifests"
                      for i,s in enumerate(expected.shards)), "expected_shard_order")
    _expected_file(expected.manifest)
    _science.validate_file_ref(expected.opaque_seed_manifest)
    base = "corpus_staging/" + _SPLIT_DIRECTORIES[expected.split]
    root = expected.manifest.file.root_binding_id
    _science._need(all(shard.inventory.root_binding_id == root for shard in expected.shards), "split_root_binding")
    _science._need(expected.manifest.file.relative_path == base + "/split_manifest.json" and expected.inventory.relative_path == base
                  and expected.inventory.root_binding_id == root, "split_expected_path")
    _science._need(expected.opaque_seed_manifest.root_binding_id == root and expected.opaque_seed_manifest.relative_path ==
                  "state4_controls/" + _SPLIT_DIRECTORIES[expected.split] + "/split_manifest.json", "opaque_reference_path")
    diagnostics = []
    expected_files = tuple(file for shard in expected.shards for file in shard.inventory.files) + (expected.manifest.file,)
    _inventory(reader, expected.inventory, expected_files, diagnostics)
    manifest_data = _acquire(reader, expected.manifest, diagnostics)
    streams = _tensor_streams(expected.split)
    coverage = {k:0 for k in streams}
    seen_controls, seen_macro = set(), set()
    rows = []
    shard_results = []
    for shard in expected.shards:
        assessment, arrays = _held_shard(shard, reader, inputs)
        shard_results.append(assessment)
        diagnostics.extend((kind, f"shard_{shard.shard_index:04d}/{role}", code) for kind,role,code in assessment.diagnostics)
        commitment = assessment.details["opaque_split_commitment"]
        if commitment is None:
            diagnostics.append(("incomplete", "opaque_seed_manifest", "pre_solve_commitment_unavailable"))
        elif commitment != dict(relative_path=expected.opaque_seed_manifest.relative_path, file_sha256=expected.opaque_seed_manifest.file_sha256):
            diagnostics.append(("mismatch", "opaque_seed_manifest", "pre_solve_split_manifest_binding"))
        for family, array in arrays.items():
            _update_tensor_stream(streams[family], array)
            coverage[family] += 64
        for control, storage, macro in assessment.details["control_hashes"]:
            if control in seen_controls:
                diagnostics.append(("mismatch", "split", "duplicate_control"))
            if macro in seen_macro:
                diagnostics.append(("mismatch", "split", "duplicate_macro_coefficient"))
            seen_controls.add(control)
            seen_macro.add(macro)
            rows.append((control, storage, macro))
        del arrays
    hashes = {k:v.hexdigest() if coverage[k] == _SPLIT_COUNTS[expected.split] else None for k,v in streams.items()}
    try:
        if manifest_data is not None:
            record = strict_canonical_json_loads(manifest_data)
            _mapping(record, {"schema", "benchmark", "split", "sealed", "sample_count", "shard_count", "parent_context", "state4_split_manifest",
                             "seed_binding", "ordered_shard_manifests", "split_tensor_sha256", "access_state", "manifest_self_sha256"}, name="split manifest")
            _science._need(record["schema"] == "operatormark.darcy_branch_c_v1.final_split_manifest.v1" and record["benchmark"] == _BENCHMARK
                          and record["split"] == expected.split and record["sealed"] is (expected.split == "test"), "split_identity")
            _science._need(type(record["sample_count"]) is int and record["sample_count"] == _SPLIT_COUNTS[expected.split]
                          and type(record["shard_count"]) is int and record["shard_count"] == count, "split_counts")
            reference_files = [next(f.file for f in shard.files if f.role == "manifest") for shard in expected.shards]
            _science._need(type(record["ordered_shard_manifests"]) is list and len(record["ordered_shard_manifests"]) == count, "split_manifest_count")
            for ref, file in zip(record["ordered_shard_manifests"], reference_files):
                _local_ref(ref, file, _FINAL_SHARD_SCHEMA)
            contexts = [next(f.file for f in s.files if f.role == "context") for s in expected.shards]
            _science._need(all(c == contexts[0] for c in contexts), "shared_context")
            _local_ref(record["parent_context"], contexts[0], _FINAL_CONTEXT_SCHEMA)
            _local_ref(record["state4_split_manifest"], expected.opaque_seed_manifest)
            _mapping(record["seed_binding"], {"state4_split_manifest_file_sha256", "json_pointer"}, name="seed reference")
            _science._need(record["seed_binding"] == dict(state4_split_manifest_file_sha256=expected.opaque_seed_manifest.file_sha256,
                          json_pointer="/pcg64_seed"), "opaque_seed_binding")
            _science._need(record["access_state"] == {"training":"accepted_training_authorization_required", "validation":"accepted_validation_authorization_required",
                          "test":"sealed_test_opening_authorization_required"}[expected.split], "access_state")
            _mapping(record["split_tensor_sha256"], set(_science.FAMILIES), name="split hashes")
            for family, value in record["split_tensor_sha256"].items():
                _sha256(value, name=family)
                if hashes[family] is None:
                    diagnostics.append(("incomplete", family, "split_tensor_unavailable"))
                elif hashes[family] != value:
                    diagnostics.append(("mismatch", family, "whole_split_tensor_hash"))
            _self_hash(record, "manifest_self_sha256", name="split")
    except (ValueError, TypeError, KeyError, OverflowError):
        diagnostics.append(("invalid", "split_manifest", "split_manifest_validation"))
    counts, maxima = _aggregate(shard_results, _SPLIT_COUNTS[expected.split])
    if len(rows) != _SPLIT_COUNTS[expected.split]:
        diagnostics.append(("incomplete", "split", "control_coverage"))
    return _science.make_assessment("split", diagnostics, recomputed=("ordered_tensor_streams", "split_membership", "cross_shard_uniqueness"),
         counts=counts, maxima=maxima, commitments=hashes, details=dict(control_hashes=tuple(rows), coverage=coverage,
         shard_summaries=tuple(dict(shard_index=i, counts=dict(a.counts), maxima=dict(a.maxima), commitments=dict(a.commitments), status=a.status)
                               for i,a in enumerate(shard_results)),
         terminal_acceptance=False, sealed=expected.split == "test", numerical_evidence="historically_admitted_not_resolved_again"))


def validate_final_split_bundle(*, expected, reader, scientific_evidence):
    _science._need(type(expected) is SplitExpectations, "split_expectations")
    _science._need(type(scientific_evidence) is tuple and all(type(r) is RetainedScientificInput and r.split == expected.split
                  and type(r.shard_index) is int and 0 <= r.shard_index < _SPLIT_SHARDS[expected.split] for r in scientific_evidence), "extra_scientific_input")
    return _held_split(expected, reader, scientific_evidence)


def validate_final_corpus_bundle(*, expected, reader, scientific_evidence, final_evidence_inputs):
    _science._need(type(expected) is CorpusExpectations and type(expected.splits) is tuple and len(expected.splits) == 3
                  and tuple(s.split for s in expected.splits) == _SPLIT_ORDER, "corpus_split_order")
    _expected_file(expected.manifest)
    _science._need(expected.manifest.file.relative_path == "corpus_manifest.json", "corpus_manifest_path")
    _science._need(type(scientific_evidence) is tuple and all(type(r) is RetainedScientificInput and r.split in _SPLIT_ORDER
                  and type(r.shard_index) is int and 0 <= r.shard_index < _SPLIT_SHARDS[r.split] for r in scientific_evidence), "extra_scientific_input")
    diagnostics = []
    files = tuple(file for split in expected.splits for file in split.inventory.files) + (expected.manifest.file,)
    _inventory(reader, expected.inventory, files, diagnostics)
    manifest = _acquire(reader, expected.manifest, diagnostics)
    results = [_held_split(split, reader, scientific_evidence) for split in expected.splits]
    controls, macros = set(), set()
    for split, result in zip(expected.splits, results):
        diagnostics.extend((kind, split.split + "/" + role, code) for kind,role,code in result.diagnostics)
        for control, _, macro in result.details["control_hashes"]:
            if control in controls:
                diagnostics.append(("mismatch", "corpus", "cross_split_control_collision"))
            if macro in macros:
                diagnostics.append(("mismatch", "corpus", "cross_split_coefficient_collision"))
            controls.add(control)
            macros.add(macro)
    if sum(len(result.details["control_hashes"]) for result in results) != 4864:
        diagnostics.append(("incomplete", "corpus", "global_unique_population_unestablished"))
    shards = [shard for split in expected.splits for shard in split.shards]
    first_files = {file.role:file.file for file in shards[0].files}
    for shard in shards:
        if shard.bindings != shards[0].bindings or any(next(f.file for f in shard.files if f.role == role) != first_files[role]
              for role in ("pre_solve", "completion", "context")):
            diagnostics.append(("mismatch", "corpus", "cross_split_predecessor_binding"))
    try:
        if manifest is not None:
            record = strict_canonical_json_loads(manifest)
            _science._need(type(record.get("ordered_split_manifests")) is list and len(record["ordered_split_manifests"]) == 3, "corpus_split_refs")
            for value, split in zip(record["ordered_split_manifests"], expected.splits):
                _local_ref(value, split.manifest.file, "operatormark.darcy_branch_c_v1.final_split_manifest.v1")
            _local_ref(record["parent_context"], first_files["context"], _FINAL_CONTEXT_SCHEMA)
            _local_ref(record["controls_pre_solve_manifest"], first_files["pre_solve"], _PRE_SOLVE_SCHEMA)
            _local_ref(record["controls_completion"], first_files["completion"], _COMPLETE_SCHEMA)
            for key in ("protocol_file_sha256", "implementation_commit", "implementation_tree"):
                _science._need(record[key] == shards[0].bindings[key], "corpus_binding")
            for state in ("state4", "state5", "state6"):
                _science._need(record["authorization_file_sha256_by_state"][state] == shards[0].bindings[state + "_authorization_file_sha256"], "corpus_authorization")
            _self_hash(record, "manifest_self_sha256", name="corpus")
    except (ValueError, TypeError, KeyError, OverflowError):
        diagnostics.append(("invalid", "corpus_manifest", "corpus_manifest_correspondence"))
    counts, maxima = _aggregate(results, 4864)
    metadata = None
    if final_evidence_inputs is None or expected.metadata_expected is None:
        diagnostics.append(("incomplete", "metadata", "final_evidence_unavailable"))
    else:
        from .final_evidence_verify import verify_final_evidence
        try:
            _science._need(type(final_evidence_inputs) is dict and set(final_evidence_inputs) == {"candidate_files", "independent_files"}, "metadata_inputs")
            metadata = verify_final_evidence(expected=expected.metadata_expected, **final_evidence_inputs)
            # Reuse its public verifier; never accept a caller-provided result.
            if metadata.report["status"] != "matched":
                diagnostics.append((metadata.report["status"], "metadata", "final_evidence_not_complete"))
            _metadata_science_correspondence(expected, manifest, results, final_evidence_inputs, diagnostics)
        except (ValueError, TypeError, KeyError):
            diagnostics.append(("invalid", "metadata", "final_evidence_validation"))
    return _science.make_assessment("corpus", diagnostics, recomputed=("all_splits", "global_uniqueness", "corpus_manifest_correspondence"),
        counts=counts, maxima=maxima, commitments={split.split:dict(result.commitments) for split,result in zip(expected.splits,results)},
        details=dict(sample_population=4864, shard_population=76, terminal_acceptance=False, production_ready=False,
                     metadata_available=metadata is not None,
                     metadata_report=metadata.report if metadata is not None else None,
                     split_summary=tuple(dict(split=s.split, sample_count=_SPLIT_COUNTS[s.split], shard_count=_SPLIT_SHARDS[s.split],
                        sample_counts=dict(r.counts), maxima=dict(r.maxima), split_file_commitment=s.manifest.file.file_sha256,
                        sealed=s.split == "test") for s,r in zip(expected.splits,results))))


def _metadata_science_correspondence(expected, manifest, results, inputs, diagnostics):
    """Join independently checked metadata to THIS byte/scientific traversal.

    Called in addition to (never instead of) the existing full metadata verifier.
    Unknown numerical inputs do not manufacture zero-count contradictions.
    """
    shards = [shard for split in expected.splits for shard in split.shards]
    summaries = [row for result in results for row in result.details["shard_summaries"]]
    for side in ("candidate_files", "independent_files"):
        supplied = inputs[side]
        _science._need(type(supplied) is tuple, "metadata_file_inventory")
        for role in ("corpus_manifest", "state5_scientific", "state6_scientific"):
            matches = [row for row in supplied if type(row) is dict and row.get("role") == role]
            if len(matches) != 1:
                diagnostics.append(("invalid" if matches else "incomplete", role, "metadata_science_membership"))
                continue
            row = matches[0]
            if row.get("data") is None:
                diagnostics.append(("incomplete", role, "metadata_science_bytes_unavailable"))
                continue
            if role == "corpus_manifest":
                if row.get("file") != _science.plain(expected.manifest.file) or row["data"] != manifest:
                    diagnostics.append(("mismatch", role, "metadata_corpus_substitution"))
                continue
            record = strict_canonical_json_loads(row["data"])
            final = role == "state6_scientific"
            if record.get("phase") != ("state6_manifests" if final else "state5_members"):
                diagnostics.append(("mismatch", role, "scientific_phase_binding"))
            rows = record.get("ordered_shards")
            _science._need(type(rows) is list and len(rows) == 76, "scientific_closure_population")
            for candidate, shard, summary in zip(rows, shards, summaries):
                files = {f.role:f.file for f in shard.files}
                fixed = dict(split=shard.split, shard_index=shard.shard_index,
                    member_files=[_science.plain(files[name]) for name in (*_science.FAMILIES,"verification")],
                    controls_manifest_file=_science.plain(files["controls_manifest"]), controls_jsonl_file=_science.plain(files["controls"]),
                    verification_record_commitment=files["verification"].file_sha256,
                    shard_manifest_file=_science.plain(files["manifest"]) if final else None,
                    evidence_file=_science.plain(shard.scientific.evidence_file))
                # Original rho origin may be represented by an independently
                # bound origin-evidence FileRef; actual rho custody was checked
                # above. The metadata verifier checks that origin relationship.
                if any(candidate.get(key) != value for key,value in fixed.items()):
                    diagnostics.append(("mismatch", role, "scientific_shard_file_binding"))
                for family, digest in summary["commitments"].items():
                    if candidate.get("shard_tensor_sha256",{}).get(family) != digest:
                        diagnostics.append(("mismatch", role, "scientific_shard_tensor_binding"))
                if summary["status"] == "matched" and (candidate.get("sample_counts") != dict(summary["counts"]) or candidate.get("maxima") != dict(summary["maxima"])):
                    diagnostics.append(("mismatch", role, "scientific_shard_summary_binding"))
            if final:
                split_rows=record.get("ordered_splits")
                _science._need(type(split_rows) is list and len(split_rows)==3,"scientific_split_population")
                for candidate, split, assessment in zip(split_rows,expected.splits,results):
                    if candidate.get("split")!=split.split or candidate.get("split_manifest_file")!=_science.plain(split.manifest.file):
                        diagnostics.append(("mismatch",role,"scientific_split_file_binding"))
                    for family,digest in assessment.commitments.items():
                        if digest is not None and candidate.get("split_tensor_sha256",{}).get(family)!=digest:
                            diagnostics.append(("mismatch",role,"scientific_split_tensor_binding"))
                    if assessment.status=="matched" and (candidate.get("sample_counts")!=dict(assessment.counts) or candidate.get("maxima")!=dict(assessment.maxima)):
                        diagnostics.append(("mismatch",role,"scientific_split_summary_binding"))


def validate_final_bundle_assessment(assessment, *, level, **inputs):
    functions = {"shard":validate_final_shard_members, "split":validate_final_split_bundle, "corpus":validate_final_corpus_bundle}
    _science._need(level in functions and type(assessment) is _science.Assessment and assessment == functions[level](**inputs), "bundle_assessment_recomputation")


__all__ += ["validate_final_shard_members", "validate_final_split_bundle", "validate_final_corpus_bundle", "validate_final_bundle_assessment"]
