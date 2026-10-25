"""Producer-side validators and exclusive writers for final corpus records.

Importing this module imports no numerical package.  NumPy is imported only by
``write_final_tensor_exclusive``, the explicitly invoked tensor operation.
Record validation proves canonical structure and record-local semantics; it is
not the independent whole-bundle scientific verifier.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .canonical import (
    CanonicalRecordError,
    canonical_jsonl_bytes,
    file_sha256,
    require_exact_keys,
    require_git_oid,
    require_sha256,
    strict_canonical_json_loads,
    strict_canonical_jsonl_loads,
    validate_npy_v2_bytes,
    validate_self_hash,
    write_canonical_json_exclusive,
    write_canonical_jsonl_exclusive,
    write_npy_v2_exclusive,
)
from .corpus_format import (
    ALGEBRAIC_ERROR_GATE,
    BENCHMARK,
    BOUNDARY_ERROR_GATE,
    CANONICAL_NPY_VERSION,
    CANONICAL_TENSOR_DTYPE,
    CANONICAL_TENSOR_ORDER,
    CONSERVATION_ERROR_GATE,
    CORPUS_VERSION,
    EXTERNAL_REFERENCE_KEYS,
    FINAL_CONTEXT_KEYS,
    FINAL_CONTEXT_SCHEMA,
    FINAL_GATE_KEYS,
    FINAL_JSONL_MEMBER_KEYS,
    FINAL_NPY_MEMBER_KEYS,
    FINAL_SHARD_MANIFEST_KEYS,
    FINAL_SHARD_MANIFEST_SCHEMA,
    FINAL_SPLIT_MANIFEST_KEYS,
    FINAL_SPLIT_MANIFEST_SCHEMA,
    FINAL_TENSOR_SHA256_KEYS,
    FINAL_VERIFICATION_KEYS,
    FINAL_VERIFICATION_SCHEMA,
    GATE9_RECORD_FILE_SHA256,
    GENERATED_DATA_RIGHTS_IDENTIFIER,
    GENERATED_DATA_RIGHTS_JSON_POINTER,
    GENERATED_DATA_RIGHTS_KEYS,
    GENERATED_DATA_RIGHTS_SCHEMA,
    GENERATED_DATA_RIGHTS_TEXT,
    GENERATED_DATA_RIGHTS_TEXT_SHA256,
    LOCAL_JSON_REFERENCE_KEYS,
    LOCAL_OPAQUE_REFERENCE_KEYS,
    METRIC_COMPLIANCE_KEYS,
    METRIC_SCALE_KEYS,
    METRICS_KEYS,
    P256_DIAGNOSTIC_KEYS,
    P256_MAX_ITERATIONS,
    PROTOCOL_SHA256,
    SEED_BINDING_KEYS,
    SHARD_MANIFEST_FILENAME,
    SHARD_SIZE,
    SOLVER_DIAGNOSTIC_KEYS,
    SOLVER_SPEC_BASELINE_COMMIT,
    SOLVER_SPEC_BASELINE_TREE,
    SPLIT_ACCESS_STATES,
    SPLIT_GLOBAL_RANGES,
    SPLIT_SAMPLE_COUNTS,
    SPLIT_SHARD_COUNTS,
    SPLIT_STORAGE_DIRECTORIES,
    SPLITS,
    STATE4_CONTROLS_COMPLETE_SCHEMA,
    STATE4_CONTROLS_SHARD_MANIFEST_SCHEMA,
    STATE4_PRE_SOLVE_MANIFEST_SCHEMA,
    T64_DIAGNOSTIC_KEYS,
    TAU_J,
    TAU_L2,
    TENSOR_FAMILY_ORDER,
    TENSOR_FAMILY_SPECS,
    THIRD_PARTY_INVENTORY_FILE_SHA256,
    VERIFICATION_FILENAME,
    VERIFIER_TENSOR_SHA256_KEYS,
)


class FinalRecordError(CanonicalRecordError):
    """Raised when a proposed final record violates the adopted contract."""


RecordValidator = Callable[[Mapping[str, Any]], None]


def _mapping(value: object, keys: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalRecordError(f"{name} must be a JSON object")
    try:
        require_exact_keys(value, keys, name=name)  # type: ignore[arg-type]
    except CanonicalRecordError as exc:
        raise FinalRecordError(str(exc)) from exc
    return value


def _sha256(value: object, *, name: str) -> str:
    try:
        return require_sha256(value, name=name)
    except CanonicalRecordError as exc:
        raise FinalRecordError(str(exc)) from exc


def _git_oid(value: object, *, name: str) -> str:
    try:
        return require_git_oid(value, name=name)
    except CanonicalRecordError as exc:
        raise FinalRecordError(str(exc)) from exc


def _validate_self_hash(record: Mapping[str, Any], field: str) -> None:
    try:
        validate_self_hash(record, field)
    except CanonicalRecordError as exc:
        raise FinalRecordError(str(exc)) from exc


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinalRecordError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise FinalRecordError(f"{name} must be a JSON boolean")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or any(ord(character) < 32 for character in value):
        raise FinalRecordError(f"{name} must be non-empty text without control characters")
    if unicodedata.normalize("NFC", value) != value:
        raise FinalRecordError(f"{name} must be Unicode NFC")
    return value


def _relative_path(value: object, *, name: str) -> str:
    path = _text(value, name=name)
    if (
        "\\" in path
        or ":" in path
        or path.startswith("/")
        or path.endswith("/")
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or PurePosixPath(path).is_absolute()
    ):
        raise FinalRecordError(f"{name} must be a canonical repository-relative POSIX path")
    return path


def _filename(value: object, *, name: str) -> str:
    filename = _text(value, name=name)
    if "/" in filename or "\\" in filename or ":" in filename or filename in {".", ".."}:
        raise FinalRecordError(f"{name} must be a basename, not a path")
    return filename


def _float_hex(value: object, *, name: str, nonnegative: bool) -> float:
    if not isinstance(value, str):
        raise FinalRecordError(f"{name} must be canonical float.hex text")
    try:
        parsed = float.fromhex(value)
    except (ValueError, OverflowError) as exc:
        raise FinalRecordError(f"{name} is not valid float.hex text") from exc
    if not math.isfinite(parsed) or parsed.hex().lower() != value:
        raise FinalRecordError(f"{name} must be canonical finite lowercase float.hex text")
    if nonnegative and parsed < 0.0:
        raise FinalRecordError(f"{name} must be non-negative")
    return parsed


def _split(value: object) -> str:
    if value not in SPLITS:
        raise FinalRecordError(f"split must be one of {SPLITS}")
    return str(value)


def _shard_index(value: object, *, split: str) -> int:
    shard_index = _integer(value, name="shard_index")
    if shard_index >= SPLIT_SHARD_COUNTS[split]:
        raise FinalRecordError("shard_index exceeds the frozen split shard count")
    return shard_index


def _hash_mapping(value: object, keys: object, *, name: str) -> Mapping[str, Any]:
    record = _mapping(value, keys, name=name)
    for key, item in record.items():
        _sha256(item, name=f"{name}.{key}")
    return record


def _hash_list(value: object, *, name: str, count: int) -> list[str]:
    if not isinstance(value, list) or len(value) != count:
        raise FinalRecordError(f"{name} must be a JSON array of length {count}")
    return [_sha256(item, name=f"{name}[{index}]") for index, item in enumerate(value)]


def _storage_split(split: str) -> str:
    return SPLIT_STORAGE_DIRECTORIES[split]


def _final_member_path(split: str, shard_index: int, filename: str) -> str:
    return (
        f"corpus_staging/{_storage_split(split)}/"
        f"shard_{shard_index:04d}/{filename}"
    )


def _final_shard_manifest_path(split: str, shard_index: int) -> str:
    return _final_member_path(split, shard_index, SHARD_MANIFEST_FILENAME)


def _state4_controls_shard_manifest_path(split: str, shard_index: int) -> str:
    return (
        f"state4_controls/{_storage_split(split)}/"
        f"shard_{shard_index:04d}.manifest.json"
    )


def _state4_split_manifest_path(split: str) -> str:
    return f"state4_controls/{_storage_split(split)}/split_manifest.json"


def validate_local_json_reference(
    value: object,
    *,
    name: str = "local JSON reference",
    expected_schema: str | None = None,
    expected_path: str | None = None,
) -> None:
    """Validate reference structure only; this does not authorize target access."""

    reference = _mapping(value, LOCAL_JSON_REFERENCE_KEYS, name=name)
    path = _relative_path(reference["relative_path"], name=f"{name}.relative_path")
    _integer(reference["size"], name=f"{name}.size")
    _sha256(reference["sha256"], name=f"{name}.sha256")
    schema = _text(reference["schema"], name=f"{name}.schema")
    if expected_path is not None and path != expected_path:
        raise FinalRecordError(f"{name} path mismatch")
    if expected_schema is not None and schema != expected_schema:
        raise FinalRecordError(f"{name} schema mismatch")


def validate_local_opaque_reference(
    value: object,
    *,
    name: str = "local opaque-file reference",
    expected_path: str | None = None,
) -> None:
    """Validate an opaque reference without opening or parsing its target."""

    reference = _mapping(value, LOCAL_OPAQUE_REFERENCE_KEYS, name=name)
    path = _relative_path(reference["relative_path"], name=f"{name}.relative_path")
    _integer(reference["size"], name=f"{name}.size")
    _sha256(reference["sha256"], name=f"{name}.sha256")
    if expected_path is not None and path != expected_path:
        raise FinalRecordError(f"{name} path mismatch")


def validate_external_reference(
    value: object, *, name: str = "external reference"
) -> None:
    """Validate the receipt reference shape without resolving an external path."""

    reference = _mapping(value, EXTERNAL_REFERENCE_KEYS, name=name)
    _filename(reference["filename"], name=f"{name}.filename")
    _sha256(reference["literal_path_sha256"], name=f"{name}.literal_path_sha256")
    _sha256(reference["sha256"], name=f"{name}.sha256")
    _integer(reference["size"], name=f"{name}.size")


def write_final_tensor_exclusive(
    path: str | Path, array: object, *, tensor_name: str
) -> bytes:
    """Write one non-rho final tensor using the frozen canonical NPY codec."""

    import numpy as np

    if tensor_name not in TENSOR_FAMILY_SPECS:
        raise FinalRecordError("unrecognized final tensor family")
    if tensor_name == "rho":
        raise FinalRecordError("rho is State 4-owned and must not be rewritten")
    specification = TENSOR_FAMILY_SPECS[tensor_name]
    destination = Path(path)
    if destination.name != specification.filename:
        raise FinalRecordError("final tensor basename does not match its logical family")
    if not isinstance(array, np.ndarray):
        raise FinalRecordError("final tensor must be a NumPy ndarray")
    if array.shape != specification.shard_shape:
        raise FinalRecordError(
            f"{tensor_name} shape mismatch: {array.shape} != {specification.shard_shape}"
        )
    if array.dtype != np.dtype(CANONICAL_TENSOR_DTYPE) or array.dtype.hasobject:
        raise FinalRecordError("final tensor dtype must be exactly <f8 and non-object")
    if not array.flags.c_contiguous:
        raise FinalRecordError("final tensor must be C-contiguous")
    if not np.isfinite(array).all():
        raise FinalRecordError("final tensor must contain only finite values")
    try:
        data = write_npy_v2_exclusive(destination, array)
        validate_npy_v2_bytes(data, expected_shape=specification.shard_shape)
    except CanonicalRecordError as exc:
        raise FinalRecordError(str(exc)) from exc
    return data


def _frozen_safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or numerator < 0.0:
        raise FinalRecordError("frozen safe-ratio numerator must be finite and non-negative")
    if not math.isfinite(denominator) or denominator < 0.0:
        raise FinalRecordError("frozen safe-ratio denominator must be finite and non-negative")
    ratio = (
        numerator / denominator
        if denominator > 0.0
        else 0.0
        if numerator == 0.0
        else math.inf
    )
    if not math.isfinite(ratio) or ratio < 0.0:
        raise FinalRecordError("frozen safe-ratio result must be finite and non-negative")
    return ratio


def _validate_solver_diagnostics(value: object) -> tuple[bool, bool]:
    diagnostics = _mapping(value, SOLVER_DIAGNOSTIC_KEYS, name="solver_diagnostics")
    t64 = _mapping(diagnostics["T64"], T64_DIAGNOSTIC_KEYS, name="solver_diagnostics.T64")
    p256 = _mapping(
        diagnostics["P256"], P256_DIAGNOSTIC_KEYS, name="solver_diagnostics.P256"
    )

    decoded_t64 = {
        key: _float_hex(t64[key], name=f"solver_diagnostics.T64.{key}", nonnegative=True)
        for key in T64_DIAGNOSTIC_KEYS - {"passes"}
    }
    decoded_p256 = {
        key: _float_hex(
            p256[key],
            name=f"solver_diagnostics.P256.{key}",
            nonnegative=key not in {"outward_boundary_flux", "integrated_source"},
        )
        for key in P256_DIAGNOSTIC_KEYS - {"iterations", "passes"}
    }
    t64_passes = _boolean(t64["passes"], name="solver_diagnostics.T64.passes")
    p256_passes = _boolean(p256["passes"], name="solver_diagnostics.P256.passes")
    iterations = _integer(
        p256["iterations"], name="solver_diagnostics.P256.iterations", minimum=1
    )
    if iterations > P256_MAX_ITERATIONS:
        raise FinalRecordError("P256 iterations exceed the frozen maxiter=10000")

    p256_balance_ratio = _frozen_safe_ratio(
        abs(decoded_p256["outward_boundary_flux"] - decoded_p256["integrated_source"]),
        decoded_p256["reaction_denominator"],
    )
    if p256["reaction_balance_ratio"] != p256_balance_ratio.hex().lower():
        raise FinalRecordError(
            "solver_diagnostics.P256.reaction_balance_ratio does not match "
            "the frozen P1 balance formula"
        )

    t64_thresholds = {
        "algebraic_backward_error": ALGEBRAIC_ERROR_GATE,
        "boundary_transmissibility_error": BOUNDARY_ERROR_GATE,
        "maximum_local_conservation_ratio": CONSERVATION_ERROR_GATE,
        "maximum_interior_face_antisymmetry": CONSERVATION_ERROR_GATE,
        "global_balance_ratio": CONSERVATION_ERROR_GATE,
    }
    p256_thresholds = {
        "algebraic_backward_error": ALGEBRAIC_ERROR_GATE,
        "boundary_reconstruction_error": BOUNDARY_ERROR_GATE,
        "reaction_balance_ratio": CONSERVATION_ERROR_GATE,
    }
    t64_within_gates = all(
        decoded_t64[key] <= limit for key, limit in t64_thresholds.items()
    )
    p256_within_gates = all(
        decoded_p256[key] <= limit for key, limit in p256_thresholds.items()
    )
    if not t64_passes or not t64_within_gates:
        raise FinalRecordError("T64 diagnostics do not satisfy the frozen accepted gates")
    if not p256_passes or not p256_within_gates:
        raise FinalRecordError("P256 diagnostics do not satisfy the frozen accepted gates")
    return True, True


def _validate_metrics(value: object) -> tuple[float, float]:
    metrics = _mapping(value, METRICS_KEYS, name="metrics")
    scales = _mapping(metrics["scales"], METRIC_SCALE_KEYS, name="metrics.scales")
    compliances = _mapping(
        metrics["compliances"], METRIC_COMPLIANCE_KEYS, name="metrics.compliances"
    )
    decoded_scales = {
        key: _float_hex(item, name=f"metrics.scales.{key}", nonnegative=True)
        for key, item in scales.items()
    }
    decoded_compliances = {
        key: _float_hex(item, name=f"metrics.compliances.{key}", nonnegative=False)
        for key, item in compliances.items()
    }
    delta_l2 = _float_hex(metrics["delta_L2"], name="metrics.delta_L2", nonnegative=True)
    delta_j = _float_hex(metrics["delta_J"], name="metrics.delta_J", nonnegative=True)

    if decoded_scales["S_J"] != abs(decoded_compliances["bar_P256"]):
        raise FinalRecordError("metrics.scales.S_J does not equal abs(bar_P256 compliance)")
    numerator = abs(decoded_compliances["T64"] - decoded_compliances["bar_P256"])
    denominator = decoded_scales["S_J"]
    expected_delta_j = (
        numerator / denominator
        if denominator > 0.0
        else 0.0
        if numerator == 0.0
        else math.inf
    )
    if not math.isfinite(expected_delta_j) or delta_j != expected_delta_j:
        raise FinalRecordError("metrics.delta_J does not match the frozen compliance formula")
    return delta_l2, delta_j


def validate_final_verification_record(record: Mapping[str, Any]) -> None:
    value = _mapping(record, FINAL_VERIFICATION_KEYS, name="final verification record")
    if value["schema"] != FINAL_VERIFICATION_SCHEMA or value["benchmark"] != BENCHMARK:
        raise FinalRecordError("final verification identity mismatch")

    split = _split(value["split"])
    shard_index = _shard_index(value["shard_index"], split=split)
    local_index = _integer(value["local_index"], name="local_index")
    global_index = _integer(value["global_index"], name="global_index")
    if local_index >= SPLIT_SAMPLE_COUNTS[split] or shard_index != local_index // SHARD_SIZE:
        raise FinalRecordError("final verification shard/local index mismatch")
    if global_index != SPLIT_GLOBAL_RANGES[split][0] + local_index:
        raise FinalRecordError("final verification global index mismatch")
    if value["sample_id"] != f"{BENCHMARK}/{split}/{local_index:06d}":
        raise FinalRecordError("final verification sample identifier mismatch")
    expected_stratum = "relaxed" if local_index % 2 == 0 else "binary_32_of_64"
    if value["stratum"] != expected_stratum:
        raise FinalRecordError("final verification stratum parity mismatch")

    for field in (
        "control_tensor_sha256",
        "rho_storage_tensor_sha256",
        "macro_coefficient_tensor_sha256",
    ):
        _sha256(value[field], name=field)
    tensor_hashes = _hash_mapping(
        value["tensor_sha256"], FINAL_TENSOR_SHA256_KEYS, name="tensor_sha256"
    )
    _hash_mapping(
        value["verifier_tensor_sha256"],
        VERIFIER_TENSOR_SHA256_KEYS,
        name="verifier_tensor_sha256",
    )
    if tensor_hashes["rho"] != value["rho_storage_tensor_sha256"]:
        raise FinalRecordError("tensor_sha256.rho must equal rho_storage_tensor_sha256")

    t64_passes, p256_passes = _validate_solver_diagnostics(value["solver_diagnostics"])
    delta_l2, delta_j = _validate_metrics(value["metrics"])
    cross_solver_passes = delta_l2 <= TAU_L2 and delta_j <= TAU_J

    gates = _mapping(value["gates"], FINAL_GATE_KEYS, name="gates")
    for key, gate in gates.items():
        _boolean(gate, name=f"gates.{key}")
    expected_gates = {
        "structural": True,
        "TPFA64": t64_passes,
        "P1_256": p256_passes,
        "cross_solver": cross_solver_passes,
        "overall": True and t64_passes and p256_passes and cross_solver_passes,
    }
    if any(gates[key] is not expected for key, expected in expected_gates.items()):
        raise FinalRecordError("final verification gate flags do not match accepted semantics")
    if gates["overall"] is not True:
        raise FinalRecordError("final verification record is not accepted")

    _sha256(value["record_self_sha256"], name="record_self_sha256")
    _validate_self_hash(value, "record_self_sha256")


def _validate_verification_rows(
    records: Sequence[Mapping[str, Any]], *, split: str, shard_index: int
) -> None:
    checked_split = _split(split)
    checked_shard = _shard_index(shard_index, split=checked_split)
    if len(records) != SHARD_SIZE:
        raise FinalRecordError("verification JSONL must contain exactly 64 records")
    for offset, record in enumerate(records):
        validate_final_verification_record(record)
        if record["split"] != checked_split or record["shard_index"] != checked_shard:
            raise FinalRecordError("verification JSONL split/shard mismatch")
        if record["local_index"] != checked_shard * SHARD_SIZE + offset:
            raise FinalRecordError("verification JSONL row order mismatch")


def write_final_verification_jsonl_exclusive(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    shard_index: int,
) -> bytes:
    """Validate, exclusively publish, and read back one final verification JSONL."""

    destination = Path(path)
    if destination.name != VERIFICATION_FILENAME:
        raise FinalRecordError("final verification JSONL basename must be verification.jsonl")
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise FinalRecordError("verification records must be a sequence")
    _validate_verification_rows(records, split=split, shard_index=shard_index)
    try:
        data = canonical_jsonl_bytes(records)
        written = write_canonical_jsonl_exclusive(destination, records)
        loaded = strict_canonical_jsonl_loads(written)
    except CanonicalRecordError as exc:
        raise FinalRecordError(str(exc)) from exc
    if written != data:
        raise FinalRecordError("verification JSONL writer changed canonical bytes")
    _validate_verification_rows(loaded, split=split, shard_index=shard_index)
    return written


def _validate_npy_member(
    value: object, *, tensor_name: str, expected_path: str
) -> None:
    member = _mapping(value, FINAL_NPY_MEMBER_KEYS, name=f"members.{tensor_name}")
    specification = TENSOR_FAMILY_SPECS[tensor_name]
    path = _relative_path(
        member["relative_path"], name=f"members.{tensor_name}.relative_path"
    )
    if path != expected_path:
        raise FinalRecordError(f"{tensor_name} member path mismatch")
    _integer(member["size"], name=f"members.{tensor_name}.size", minimum=1)
    _sha256(member["sha256"], name=f"members.{tensor_name}.sha256")
    if member["tensor_name"] != tensor_name:
        raise FinalRecordError(f"{tensor_name} member logical-name mismatch")
    shape = member["shape"]
    if (
        member["dtype"] != CANONICAL_TENSOR_DTYPE
        or not isinstance(shape, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
        or shape != list(specification.shard_shape)
        or member["order"] != CANONICAL_TENSOR_ORDER
    ):
        raise FinalRecordError(f"{tensor_name} member storage metadata mismatch")
    version = member["npy_version"]
    if (
        not isinstance(version, list)
        or len(version) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in version)
        or tuple(version) != CANONICAL_NPY_VERSION
    ):
        raise FinalRecordError(f"{tensor_name} member NPY version mismatch")
    _sha256(member["tensor_sha256"], name=f"members.{tensor_name}.tensor_sha256")


def _validate_jsonl_member(value: object, *, expected_path: str) -> None:
    member = _mapping(value, FINAL_JSONL_MEMBER_KEYS, name="members.verification")
    path = _relative_path(
        member["relative_path"], name="members.verification.relative_path"
    )
    if path != expected_path:
        raise FinalRecordError("verification member path mismatch")
    _integer(member["size"], name="members.verification.size", minimum=1)
    _sha256(member["sha256"], name="members.verification.sha256")
    if member["schema"] != FINAL_VERIFICATION_SCHEMA:
        raise FinalRecordError("verification member schema mismatch")
    if _integer(member["record_count"], name="members.verification.record_count") != SHARD_SIZE:
        raise FinalRecordError("verification member record_count must be 64")
    _hash_list(
        member["ordered_record_self_sha256"],
        name="members.verification.ordered_record_self_sha256",
        count=SHARD_SIZE,
    )


def validate_final_shard_manifest(record: Mapping[str, Any]) -> None:
    value = _mapping(record, FINAL_SHARD_MANIFEST_KEYS, name="final shard manifest")
    if value["schema"] != FINAL_SHARD_MANIFEST_SCHEMA or value["benchmark"] != BENCHMARK:
        raise FinalRecordError("final shard-manifest identity mismatch")
    split = _split(value["split"])
    shard_index = _shard_index(value["shard_index"], split=split)
    if _integer(value["sample_count"], name="sample_count") != SHARD_SIZE:
        raise FinalRecordError("final shard sample_count must be 64")

    first_local = shard_index * SHARD_SIZE
    last_local = first_local + SHARD_SIZE - 1
    global_offset = SPLIT_GLOBAL_RANGES[split][0]
    expected_ranges = {
        "first_local_index": first_local,
        "last_local_index": last_local,
        "first_global_index": global_offset + first_local,
        "last_global_index": global_offset + last_local,
    }
    for field, expected in expected_ranges.items():
        if _integer(value[field], name=field) != expected:
            raise FinalRecordError("final shard index range mismatch")

    sample_ids = value["ordered_sample_ids"]
    expected_ids = [
        f"{BENCHMARK}/{split}/{local_index:06d}"
        for local_index in range(first_local, last_local + 1)
    ]
    if not isinstance(sample_ids, list) or sample_ids != expected_ids:
        raise FinalRecordError("final shard ordered_sample_ids mismatch")

    validate_local_json_reference(
        value["parent_context"],
        name="parent_context",
        expected_schema=FINAL_CONTEXT_SCHEMA,
    )
    validate_local_json_reference(
        value["pre_solve_manifest"],
        name="pre_solve_manifest",
        expected_schema=STATE4_PRE_SOLVE_MANIFEST_SCHEMA,
        expected_path="state4_controls/pre_solve_manifest.json",
    )
    validate_local_json_reference(
        value["state4_controls_shard_manifest"],
        name="state4_controls_shard_manifest",
        expected_schema=STATE4_CONTROLS_SHARD_MANIFEST_SCHEMA,
        expected_path=_state4_controls_shard_manifest_path(split, shard_index),
    )

    members = value["members"]
    if not isinstance(members, list) or len(members) != len(TENSOR_FAMILY_ORDER) + 1:
        raise FinalRecordError("final shard members must be the ordered eight-entry array")
    for index, tensor_name in enumerate(TENSOR_FAMILY_ORDER):
        specification = TENSOR_FAMILY_SPECS[tensor_name]
        _validate_npy_member(
            members[index],
            tensor_name=tensor_name,
            expected_path=_final_member_path(split, shard_index, specification.filename),
        )
    _validate_jsonl_member(
        members[-1],
        expected_path=_final_member_path(split, shard_index, VERIFICATION_FILENAME),
    )
    _sha256(value["manifest_self_sha256"], name="manifest_self_sha256")
    _validate_self_hash(value, "manifest_self_sha256")


def validate_final_split_manifest(record: Mapping[str, Any]) -> None:
    """Validate the record-local split schema and transitive seed reference.

    Recomputing streamed logical split commitments from every accepted shard is
    a separate whole-split verification operation, not evidence supplied by
    this record-local validator.
    """

    value = _mapping(record, FINAL_SPLIT_MANIFEST_KEYS, name="final split manifest")
    if value["schema"] != FINAL_SPLIT_MANIFEST_SCHEMA or value["benchmark"] != BENCHMARK:
        raise FinalRecordError("final split-manifest identity mismatch")
    split = _split(value["split"])
    if _boolean(value["sealed"], name="sealed") is not (split == "test"):
        raise FinalRecordError("final split sealed flag mismatch")
    if (
        _integer(value["sample_count"], name="sample_count") != SPLIT_SAMPLE_COUNTS[split]
        or _integer(value["shard_count"], name="shard_count") != SPLIT_SHARD_COUNTS[split]
    ):
        raise FinalRecordError("final split counts mismatch")
    validate_local_json_reference(
        value["parent_context"],
        name="parent_context",
        expected_schema=FINAL_CONTEXT_SCHEMA,
    )
    validate_local_opaque_reference(
        value["state4_split_manifest"],
        name="state4_split_manifest",
        expected_path=_state4_split_manifest_path(split),
    )

    seed_binding = _mapping(value["seed_binding"], SEED_BINDING_KEYS, name="seed_binding")
    bound_hash = _sha256(
        seed_binding["state4_split_manifest_file_sha256"],
        name="seed_binding.state4_split_manifest_file_sha256",
    )
    if bound_hash != value["state4_split_manifest"]["sha256"]:
        raise FinalRecordError("seed binding does not bind the opaque State 4 split manifest")
    if seed_binding["json_pointer"] != "/pcg64_seed":
        raise FinalRecordError("seed binding JSON pointer must be /pcg64_seed")

    references = value["ordered_shard_manifests"]
    shard_count = SPLIT_SHARD_COUNTS[split]
    if not isinstance(references, list) or len(references) != shard_count:
        raise FinalRecordError("ordered_shard_manifests length mismatch")
    for shard_index, reference in enumerate(references):
        validate_local_json_reference(
            reference,
            name=f"ordered_shard_manifests[{shard_index}]",
            expected_schema=FINAL_SHARD_MANIFEST_SCHEMA,
            expected_path=_final_shard_manifest_path(split, shard_index),
        )

    _hash_mapping(
        value["split_tensor_sha256"],
        FINAL_TENSOR_SHA256_KEYS,
        name="split_tensor_sha256",
    )
    if value["access_state"] != SPLIT_ACCESS_STATES[split]:
        raise FinalRecordError("final split access_state mismatch")
    _sha256(value["manifest_self_sha256"], name="manifest_self_sha256")
    _validate_self_hash(value, "manifest_self_sha256")


def validate_final_context(record: Mapping[str, Any]) -> None:
    """Validate the record-local final-context format and fixed literals.

    Equality to the actual later authorization and implementation files is an
    external binding check; syntactically valid hashes here do not prove it.
    """

    value = _mapping(record, FINAL_CONTEXT_KEYS, name="final context")
    if (
        value["schema"] != FINAL_CONTEXT_SCHEMA
        or value["benchmark"] != BENCHMARK
        or value["corpus_version"] != CORPUS_VERSION
    ):
        raise FinalRecordError("final context identity mismatch")
    attempt_id = _text(value["attempt_id"], name="attempt_id")
    if value["protocol_file_sha256"] != PROTOCOL_SHA256:
        raise FinalRecordError("final context protocol binding mismatch")
    if (
        value["solver_spec_baseline_commit"] != SOLVER_SPEC_BASELINE_COMMIT
        or value["solver_spec_baseline_tree"] != SOLVER_SPEC_BASELINE_TREE
    ):
        raise FinalRecordError("final context solver baseline binding mismatch")
    for field in (
        "solver_spec_baseline_commit",
        "solver_spec_baseline_tree",
        "implementation_commit",
        "implementation_tree",
    ):
        _git_oid(value[field], name=field)
    for field in (
        "protocol_file_sha256",
        "state4_authorization_file_sha256",
        "state5_authorization_file_sha256",
        "state6_authorization_file_sha256",
    ):
        _sha256(value[field], name=field)
    if attempt_id != f"s4-{value['state4_authorization_file_sha256'][:24]}":
        raise FinalRecordError("final context attempt/State 4 authorization mismatch")
    validate_local_json_reference(
        value["pre_solve_manifest"],
        name="pre_solve_manifest",
        expected_schema=STATE4_PRE_SOLVE_MANIFEST_SCHEMA,
        expected_path="state4_controls/pre_solve_manifest.json",
    )
    validate_local_json_reference(
        value["controls_completion"],
        name="controls_completion",
        expected_schema=STATE4_CONTROLS_COMPLETE_SCHEMA,
        expected_path="STATE4_CONTROLS_COMPLETE.json",
    )
    if not isinstance(value["split_order"], list) or value["split_order"] != list(SPLITS):
        raise FinalRecordError("final context split_order mismatch")
    _sha256(value["record_self_sha256"], name="record_self_sha256")
    _validate_self_hash(value, "record_self_sha256")


def validate_generated_data_rights(record: Mapping[str, Any]) -> None:
    value = _mapping(record, GENERATED_DATA_RIGHTS_KEYS, name="generated-data rights record")
    expected = {
        "schema": GENERATED_DATA_RIGHTS_SCHEMA,
        "benchmark": BENCHMARK,
        "identifier": GENERATED_DATA_RIGHTS_IDENTIFIER,
        "gate9_record_file_sha256": GATE9_RECORD_FILE_SHA256,
        "rights_text_json_pointer": GENERATED_DATA_RIGHTS_JSON_POINTER,
        "rights_text": GENERATED_DATA_RIGHTS_TEXT,
        "rights_text_sha256": GENERATED_DATA_RIGHTS_TEXT_SHA256,
        "third_party_inventory_file_sha256": THIRD_PARTY_INVENTORY_FILE_SHA256,
    }
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        raise FinalRecordError("generated-data rights record differs from frozen Gate 9 evidence")
    for field in (
        "gate9_record_file_sha256",
        "rights_text_sha256",
        "third_party_inventory_file_sha256",
        "record_self_sha256",
    ):
        _sha256(value[field], name=field)
    _validate_self_hash(value, "record_self_sha256")


def validate_final_record_file(
    path: str | Path, validator: RecordValidator
) -> dict[str, Any]:
    """Read one canonical record, validate it, and detect concurrent mutation."""

    target = Path(path)
    digest_before = file_sha256(target)
    try:
        record = strict_canonical_json_loads(target.read_bytes())
    except CanonicalRecordError as exc:
        raise FinalRecordError(str(exc)) from exc
    validator(record)
    if digest_before != file_sha256(target):
        raise FinalRecordError("final record changed during validation")
    return record


def write_final_record_exclusive(
    path: str | Path, record: Mapping[str, Any], validator: RecordValidator
) -> bytes:
    """Exclusively publish and read back a validated canonical final record."""

    validator(record)
    try:
        data = write_canonical_json_exclusive(path, record)
        loaded = strict_canonical_json_loads(data)
    except CanonicalRecordError as exc:
        raise FinalRecordError(str(exc)) from exc
    validator(loaded)
    return data
