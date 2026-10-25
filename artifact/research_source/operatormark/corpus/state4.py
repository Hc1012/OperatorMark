"""Exact State 4 record schemas and writers; no control generation occurs here."""

from __future__ import annotations

import os
import re
import stat
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from .canonical import (
    CanonicalRecordError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    domain_separated_digest,
    file_sha256,
    require_git_oid as canonical_require_git_oid,
    require_exact_keys as canonical_require_exact_keys,
    require_sha256 as canonical_require_sha256,
    strict_canonical_json_loads,
    strict_canonical_jsonl_loads,
    sha256_bytes,
    tensor_sha256,
    validate_npy_v2_bytes,
    validate_self_hash as canonical_validate_self_hash,
    write_canonical_json_exclusive,
    write_canonical_jsonl_exclusive,
    write_npy_v2_exclusive,
)
from .rng_guard import RngGuardViolation, State4SeedCapability
from .storage import (
    STATE4_FILE_PERMISSION_METADATA_PROFILE,
    StoragePreflightError,
    validate_state4_file_permission_metadata,
)


BENCHMARK: Final = "darcy_branch_c_v1"
# State 2 representation choice matching the frozen destination component;
# the protocol declares this field but not a separate value literal for it.
CORPUS_VERSION: Final = "private_corpus_v1"
PROTOCOL_RELATIVE_NAME: Final = "OperatorMark_Private_Corpus_PreGeneration_Frozen_Protocol_v1.0.md"
PROTOCOL_SHA256: Final = "a38ef79b6c4bdccf53f599fa08f77e48a111afbf0b2783ad5733bda99996599a"
SPLITS: Final = ("training", "validation", "test")
SPLIT_COUNTS: Final = {"training": 4096, "validation": 256, "test": 512}
SPLIT_SHARDS: Final = {"training": 64, "validation": 4, "test": 8}
SPLIT_GLOBAL_RANGES: Final = {
    "training": (0, 4095),
    "validation": (4096, 4351),
    "test": (4352, 4863),
}
SOLVER_SPEC_BASELINE_COMMIT: Final = "e16456395cfe67b73b9d298fd0491cf9524d798a"
SOLVER_SPEC_BASELINE_TREE: Final = "81f40e24102e51e0da3ce65dafd769a7678565fd"
EMPTY_SHA256: Final = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
STATE4_GIT_RECORD_KEYS: Final = {
    "branch", "head", "tree", "parents", "status_short_untracked_all",
    "tracked_diff_sha256", "tracked_diff_size_bytes", "staged_diff_sha256",
    "staged_diff_size_bytes", "tracked_changes_empty", "staged_changes_empty",
    "untracked_inclusive_status_empty",
}
_ATTEMPT_ID_RE: Final = re.compile(r"^s4-[0-9a-f]{24}$")

CONTROL_RECORD_KEYS: Final = {
    "schema", "benchmark", "split", "shard_index", "local_index", "global_index",
    "sample_id", "stratum", "control_tensor_sha256", "rho_storage_tensor_sha256",
    "macro_coefficient_tensor_sha256", "record_self_sha256",
}
SHARD_MANIFEST_KEYS: Final = {
    "schema", "benchmark", "split", "sealed", "shard_index", "first_local_index",
    "last_local_index", "first_global_index", "last_global_index", "sample_count",
    "rho_relative_path", "rho_dtype", "rho_shape", "rho_order", "npy_version",
    "rho_file_bytes", "rho_file_sha256", "rho_tensor_sha256",
    "control_records_relative_path", "control_records_bytes",
    "control_records_file_sha256", "ordered_record_self_sha256", "protocol_file_sha256",
    "state4_authorization_file_sha256", "implementation_commit", "implementation_tree",
    "environment_record_sha256", "manifest_self_sha256",
}
SPLIT_MANIFEST_KEYS: Final = {
    "schema", "benchmark", "split", "sealed", "sample_count", "shard_count",
    "pcg64_seed", "rng_api_call_count", "initial_pcg64_state_sha256",
    "final_pcg64_state_sha256", "first_global_index", "last_global_index",
    "ordered_shard_manifest_paths", "ordered_shard_manifest_file_sha256",
    "ordered_rho_file_sha256", "ordered_rho_tensor_sha256",
    "ordered_control_tensor_sha256", "ordered_macro_coefficient_tensor_sha256",
    "split_control_aggregate_sha256", "split_macro_coefficient_aggregate_sha256",
    "protocol_file_sha256", "state4_authorization_file_sha256", "implementation_commit",
    "implementation_tree", "manifest_self_sha256",
}
RNG_AUDIT_KEYS: Final = {
    "schema", "benchmark", "state", "attempt_id", "protocol_file_sha256",
    "state4_authorization_file_sha256", "implementation_commit", "implementation_tree",
    "frozen_base_import_closure_sha256", "rng_state_commitments",
    "rng_state_equalities_to_after_base_import", "authorized_corpus_seed_access_count_by_split",
    "authorized_pcg64_constructor_count_by_split", "authorized_generator_constructor_count_by_split",
    "authorized_top_level_rng_api_call_count_by_split", "forbidden_rng_call_count",
    "forbidden_or_out_of_scope_corpus_seed_access_count", "all_rng_checks_passed",
    "record_self_sha256",
}
READ_ONLY_KEYS: Final = {
    "schema", "attempt_id", "permission_mechanism", "ordered_relative_paths",
    "ordered_file_sha256", "ordered_permission_evidence", "all_required_files_read_only",
    "record_self_sha256",
}
PRE_SOLVE_KEYS: Final = {
    "schema", "benchmark", "corpus_version", "state", "attempt_id",
    "protocol_relative_name", "protocol_file_sha256", "state4_authorization_file_sha256",
    "solver_spec_baseline_commit", "solver_spec_baseline_tree", "implementation_commit",
    "implementation_tree", "state3_audit_file_sha256", "compute_feasibility_report_file_sha256",
    "environment_record_sha256", "storage_preflight_before_file_sha256",
    "storage_preflight_after_file_sha256", "rng_audit_relative_path", "rng_audit_file_sha256",
    "read_only_evidence_relative_path", "read_only_evidence_file_sha256", "split_order",
    "split_manifest_paths", "split_manifest_file_sha256", "sample_count", "shard_count",
    "global_control_aggregate_sha256", "global_macro_coefficient_aggregate_sha256",
    "all_control_hashes_unique", "all_macro_coefficient_hashes_unique",
    "rng_call_count_by_split", "forbidden_seed_or_data_access_count", "git_before",
    "git_after", "manifest_self_sha256",
}
COMPLETE_KEYS: Final = {
    "schema", "state", "attempt_id", "pre_solve_manifest_relative_path",
    "pre_solve_manifest_file_sha256", "read_only_evidence_file_sha256",
    "state4_authorization_file_sha256", "completion_time_utc", "record_self_sha256",
}


class State4RecordError(CanonicalRecordError):
    """Raised when a State 4 record violates an exact frozen schema."""


def require_exact_keys(record: Mapping[str, Any], keys: Sequence[str] | set[str], *, name: str) -> None:
    try:
        canonical_require_exact_keys(record, keys, name=name)
    except CanonicalRecordError as exc:
        raise State4RecordError(str(exc)) from exc


def require_sha256(value: object, *, name: str) -> str:
    try:
        return canonical_require_sha256(value, name=name)
    except CanonicalRecordError as exc:
        raise State4RecordError(str(exc)) from exc


def require_git_oid(value: object, *, name: str) -> str:
    try:
        return canonical_require_git_oid(value, name=name)
    except CanonicalRecordError as exc:
        raise State4RecordError(str(exc)) from exc


def validate_self_hash(record: Mapping[str, Any], field: str) -> None:
    try:
        canonical_validate_self_hash(record, field)
    except CanonicalRecordError as exc:
        raise State4RecordError(str(exc)) from exc


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise State4RecordError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise State4RecordError(f"{name} must be a JSON boolean")
    return value


def _split(value: object) -> str:
    if value not in SPLITS:
        raise State4RecordError(f"split must be one of {SPLITS}")
    return str(value)


def _hash_list(value: object, *, name: str, count: int | None = None) -> list[str]:
    if not isinstance(value, list) or (count is not None and len(value) != count):
        raise State4RecordError(f"{name} must be a JSON array with the required length")
    return [require_sha256(item, name=f"{name}[{index}]") for index, item in enumerate(value)]


def _exact_split_map(value: object, *, name: str, expected: Mapping[str, int]) -> None:
    if not isinstance(value, dict) or set(value) != set(SPLITS):
        raise State4RecordError(f"{name} must contain exactly training, validation, and test")
    for split, expected_value in expected.items():
        actual = _integer(value[split], name=f"{name}.{split}")
        if actual != expected_value:
            raise State4RecordError(f"{name}.{split} must equal {expected_value}")


def _relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise State4RecordError(f"{name} must be a non-empty relative path")
    if unicodedata.normalize("NFC", value) != value:
        raise State4RecordError(f"{name} must use NFC text")
    if (
        "\x00" in value or "\\" in value or ":" in value
        or value.startswith("/") or value.endswith("/")
    ):
        raise State4RecordError(f"{name} must be a canonical repository-relative POSIX path")
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise State4RecordError(f"{name} must be a canonical repository-relative POSIX path")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise State4RecordError(f"{name} must be a non-empty string")
    return value


def _attempt_id(value: object, *, name: str = "attempt_id") -> str:
    if not isinstance(value, str) or _ATTEMPT_ID_RE.fullmatch(value) is None:
        raise State4RecordError(f"{name} must be s4- followed by 24 lowercase hexadecimal characters")
    return value


def _expected_attempt_id(authorization_sha256: str) -> str:
    return f"s4-{authorization_sha256[:24]}"


def _split_path_component(split: str) -> str:
    return "test.sealed" if split == "test" else split


def _rho_relative_path(split: str, shard_index: int) -> str:
    component = _split_path_component(split)
    return f"corpus_staging/{component}/shard_{shard_index:04d}/rho.float64.npy"


def _control_records_relative_path(split: str, shard_index: int) -> str:
    component = _split_path_component(split)
    return f"state4_controls/{component}/shard_{shard_index:04d}.controls.jsonl"


def _shard_manifest_relative_path(split: str, shard_index: int) -> str:
    component = _split_path_component(split)
    return f"state4_controls/{component}/shard_{shard_index:04d}.manifest.json"


def _split_manifest_relative_path(split: str) -> str:
    return f"state4_controls/{_split_path_component(split)}/split_manifest.json"


def _required_read_only_paths() -> set[str]:
    paths: set[str] = set()
    for split in SPLITS:
        for shard_index in range(SPLIT_SHARDS[split]):
            paths.add(_rho_relative_path(split, shard_index))
            paths.add(_control_records_relative_path(split, shard_index))
            paths.add(_shard_manifest_relative_path(split, shard_index))
        paths.add(_split_manifest_relative_path(split))
    paths.add("state4_controls/rng_audit.json")
    return paths


def _ordered_required_read_only_paths() -> tuple[str, ...]:
    paths: list[str] = []
    for split in SPLITS:
        for shard_index in range(SPLIT_SHARDS[split]):
            paths.extend(
                (
                    _rho_relative_path(split, shard_index),
                    _control_records_relative_path(split, shard_index),
                    _shard_manifest_relative_path(split, shard_index),
                )
            )
        paths.append(_split_manifest_relative_path(split))
    paths.append("state4_controls/rng_audit.json")
    return tuple(paths)


def _validate_git_record(
    value: object, *, name: str, implementation_commit: str, implementation_tree: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise State4RecordError(f"{name} must be a Git record object")
    require_exact_keys(value, STATE4_GIT_RECORD_KEYS, name=name)
    if value["branch"] != "main":
        raise State4RecordError(f"{name}.branch must be main")
    head = require_git_oid(value["head"], name=f"{name}.head")
    tree = require_git_oid(value["tree"], name=f"{name}.tree")
    if head != implementation_commit or tree != implementation_tree:
        raise State4RecordError(f"{name} implementation identity mismatch")
    parents = value["parents"]
    if not isinstance(parents, list) or not parents:
        raise State4RecordError(f"{name}.parents must be a non-empty array")
    checked_parents = [
        require_git_oid(parent, name=f"{name}.parents[{index}]")
        for index, parent in enumerate(parents)
    ]
    if len(set(checked_parents)) != len(checked_parents):
        raise State4RecordError(f"{name}.parents must be unique")
    if value["status_short_untracked_all"] != "":
        raise State4RecordError(f"{name} must have empty untracked-inclusive status")
    for prefix in ("tracked", "staged"):
        if value[f"{prefix}_diff_sha256"] != EMPTY_SHA256:
            raise State4RecordError(f"{name}.{prefix}_diff_sha256 must bind an empty diff")
        if _integer(value[f"{prefix}_diff_size_bytes"], name=f"{name}.{prefix}_diff_size_bytes") != 0:
            raise State4RecordError(f"{name}.{prefix}_diff_size_bytes must be zero")
        if _boolean(value[f"{prefix}_changes_empty"], name=f"{name}.{prefix}_changes_empty") is not True:
            raise State4RecordError(f"{name}.{prefix}_changes_empty must be true")
    if _boolean(
        value["untracked_inclusive_status_empty"],
        name=f"{name}.untracked_inclusive_status_empty",
    ) is not True:
        raise State4RecordError(f"{name}.untracked_inclusive_status_empty must be true")
    return value


def _validate_utc_timestamp(value: object) -> str:
    if not isinstance(value, str) or len(value) != 27:
        raise State4RecordError("completion timestamp must use frozen UTC format")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise State4RecordError("completion timestamp must use frozen UTC format") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:
        raise State4RecordError("completion timestamp must use frozen UTC format")
    return value


def validate_control_record(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, CONTROL_RECORD_KEYS, name="State 4 control record")
    if record["schema"] != "operatormark.darcy_branch_c_v1.state4_control_record.v1":
        raise State4RecordError("control-record schema mismatch")
    if record["benchmark"] != BENCHMARK:
        raise State4RecordError("benchmark mismatch")
    split = _split(record["split"])
    shard = _integer(record["shard_index"], name="shard_index")
    local = _integer(record["local_index"], name="local_index")
    global_index = _integer(record["global_index"], name="global_index")
    if shard != local // 64 or local >= SPLIT_COUNTS[split]:
        raise State4RecordError("control record shard/local index mismatch")
    offset = SPLIT_GLOBAL_RANGES[split][0]
    if global_index != offset + local:
        raise State4RecordError("control record global index mismatch")
    if record["sample_id"] != f"{BENCHMARK}/{split}/{local:06d}":
        raise State4RecordError("sample identifier mismatch")
    if record["stratum"] not in {"relaxed", "binary_32_of_64"}:
        raise State4RecordError("invalid control stratum")
    expected_stratum = "relaxed" if local % 2 == 0 else "binary_32_of_64"
    if record["stratum"] != expected_stratum:
        raise State4RecordError("control stratum parity mismatch")
    for name in (
        "control_tensor_sha256", "rho_storage_tensor_sha256",
        "macro_coefficient_tensor_sha256", "record_self_sha256",
    ):
        require_sha256(record[name], name=name)
    validate_self_hash(record, "record_self_sha256")


def validate_control_records_jsonl(data: bytes, *, split: str, shard_index: int) -> list[dict[str, Any]]:
    records = strict_canonical_jsonl_loads(data)
    if len(records) != 64:
        raise State4RecordError("controls JSONL must contain exactly 64 records")
    for offset, record in enumerate(records):
        validate_control_record(record)
        if record["split"] != split or record["shard_index"] != shard_index:
            raise State4RecordError("controls JSONL split/shard mismatch")
        if record["local_index"] != shard_index * 64 + offset:
            raise State4RecordError("controls JSONL order mismatch")
    return records


def validate_shard_manifest(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, SHARD_MANIFEST_KEYS, name="State 4 shard manifest")
    if record["schema"] != "operatormark.darcy_branch_c_v1.state4_controls_shard_manifest.v1":
        raise State4RecordError("shard-manifest schema mismatch")
    if record["benchmark"] != BENCHMARK:
        raise State4RecordError("benchmark mismatch")
    split = _split(record["split"])
    shard = _integer(record["shard_index"], name="shard_index")
    if shard >= SPLIT_SHARDS[split]:
        raise State4RecordError("shard index is outside the frozen split range")
    if _boolean(record["sealed"], name="sealed") is not (split == "test"):
        raise State4RecordError("sealed flag mismatch")
    sample_count = _integer(record["sample_count"], name="sample_count")
    first_local = _integer(record["first_local_index"], name="first_local_index")
    last_local = _integer(record["last_local_index"], name="last_local_index")
    if sample_count != 64 or first_local != shard * 64 or last_local != shard * 64 + 63:
        raise State4RecordError("shard local range mismatch")
    offset = SPLIT_GLOBAL_RANGES[split][0]
    first_global = _integer(record["first_global_index"], name="first_global_index")
    last_global = _integer(record["last_global_index"], name="last_global_index")
    if first_global != offset + shard * 64 or last_global != offset + shard * 64 + 63:
        raise State4RecordError("shard global range mismatch")
    rho_shape = record["rho_shape"]
    if (
        record["rho_dtype"] != "<f8"
        or not isinstance(rho_shape, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in rho_shape)
        or rho_shape != [64, 1, 8, 8]
        or record["rho_order"] != "C"
        or record["npy_version"] != "2.0"
    ):
        raise State4RecordError("frozen rho storage metadata mismatch")
    rho_path = _relative_path(record["rho_relative_path"], name="rho_relative_path")
    controls_path = _relative_path(
        record["control_records_relative_path"], name="control_records_relative_path",
    )
    if rho_path != _rho_relative_path(split, shard):
        raise State4RecordError("rho_relative_path mismatch")
    if controls_path != _control_records_relative_path(split, shard):
        raise State4RecordError("control_records_relative_path mismatch")
    _integer(record["rho_file_bytes"], name="rho_file_bytes", minimum=1)
    _integer(record["control_records_bytes"], name="control_records_bytes", minimum=1)
    for name in (
        "rho_file_sha256", "rho_tensor_sha256", "control_records_file_sha256",
        "protocol_file_sha256", "state4_authorization_file_sha256",
        "environment_record_sha256", "manifest_self_sha256",
    ):
        require_sha256(record[name], name=name)
    if record["protocol_file_sha256"] != PROTOCOL_SHA256:
        raise State4RecordError("protocol_file_sha256 mismatch")
    for name in ("implementation_commit", "implementation_tree"):
        require_git_oid(record[name], name=name)
    _hash_list(record["ordered_record_self_sha256"], name="ordered_record_self_sha256", count=64)
    validate_self_hash(record, "manifest_self_sha256")


def _state4_frozen_split_seed(split: str) -> int:
    if split == "training":
        seed_text = "2026083103"
    elif split == "validation":
        seed_text = "2026083104"
    elif split == "test":
        seed_text = "2026083105"
    else:
        raise State4RecordError(f"split must be one of {SPLITS}")
    return int(seed_text, 10)


def _validate_split_manifest_fields(
    record: Mapping[str, Any],
    *,
    expected_split: str,
    expected_authorization_file_sha256: str,
    expected_seed: int,
) -> None:
    require_exact_keys(record, SPLIT_MANIFEST_KEYS, name="State 4 split manifest")
    if record["schema"] != "operatormark.darcy_branch_c_v1.state4_controls_split_manifest.v1":
        raise State4RecordError("split-manifest schema mismatch")
    if record["benchmark"] != BENCHMARK:
        raise State4RecordError("benchmark mismatch")
    split = _split(record["split"])
    if split != expected_split:
        raise State4RecordError("split does not match the bound State 4 seed capability")
    authorization_file_sha256 = require_sha256(
        record["state4_authorization_file_sha256"],
        name="state4_authorization_file_sha256",
    )
    if authorization_file_sha256 != expected_authorization_file_sha256:
        raise State4RecordError("State 4 authorization commitment mismatch")
    if _integer(record["pcg64_seed"], name="pcg64_seed") != expected_seed:
        raise State4RecordError("pcg64_seed mismatch")
    if _boolean(record["sealed"], name="sealed") is not (split == "test"):
        raise State4RecordError("sealed flag mismatch")
    sample_count = _integer(record["sample_count"], name="sample_count")
    shard_count = _integer(record["shard_count"], name="shard_count")
    rng_call_count = _integer(record["rng_api_call_count"], name="rng_api_call_count")
    if (
        sample_count != SPLIT_COUNTS[split]
        or shard_count != SPLIT_SHARDS[split]
        or rng_call_count != SPLIT_COUNTS[split]
    ):
        raise State4RecordError("split counts mismatch")
    first_global = _integer(record["first_global_index"], name="first_global_index")
    last_global = _integer(record["last_global_index"], name="last_global_index")
    if (first_global, last_global) != SPLIT_GLOBAL_RANGES[split]:
        raise State4RecordError("split global range mismatch")
    for name in (
        "initial_pcg64_state_sha256", "final_pcg64_state_sha256",
        "split_control_aggregate_sha256", "split_macro_coefficient_aggregate_sha256",
        "protocol_file_sha256", "manifest_self_sha256",
    ):
        require_sha256(record[name], name=name)
    if record["protocol_file_sha256"] != PROTOCOL_SHA256:
        raise State4RecordError("protocol_file_sha256 mismatch")
    for name in ("implementation_commit", "implementation_tree"):
        require_git_oid(record[name], name=name)
    for name in (
        "ordered_shard_manifest_file_sha256", "ordered_rho_file_sha256",
        "ordered_rho_tensor_sha256",
    ):
        _hash_list(record[name], name=name, count=shard_count)
    control_hashes = _hash_list(
        record["ordered_control_tensor_sha256"],
        name="ordered_control_tensor_sha256",
        count=sample_count,
    )
    macro_hashes = _hash_list(
        record["ordered_macro_coefficient_tensor_sha256"],
        name="ordered_macro_coefficient_tensor_sha256",
        count=sample_count,
    )
    if len(set(control_hashes)) != sample_count:
        raise State4RecordError("split control tensor hashes must be unique")
    if len(set(macro_hashes)) != sample_count:
        raise State4RecordError("split macro-coefficient tensor hashes must be unique")
    paths = record["ordered_shard_manifest_paths"]
    if not isinstance(paths, list) or len(paths) != shard_count:
        raise State4RecordError("ordered shard paths length mismatch")
    checked_paths = [
        _relative_path(path, name=f"ordered_shard_manifest_paths[{index}]")
        for index, path in enumerate(paths)
    ]
    expected_paths = [
        _shard_manifest_relative_path(split, index) for index in range(shard_count)
    ]
    if checked_paths != expected_paths:
        raise State4RecordError("ordered shard paths mismatch")
    if record["split_control_aggregate_sha256"] != split_control_aggregate(control_hashes):
        raise State4RecordError("split control aggregate mismatch")
    if (
        record["split_macro_coefficient_aggregate_sha256"]
        != split_macro_coefficient_aggregate(macro_hashes)
    ):
        raise State4RecordError("split macro-coefficient aggregate mismatch")
    validate_self_hash(record, "manifest_self_sha256")


class _BoundSplitManifestValidator:
    __slots__ = (
        "_authorization_file_sha256",
        "_expected_seed",
        "_seed_capability",
        "_split",
    )

    def __init__(
        self,
        *,
        split: str,
        authorization_file_sha256: str,
        expected_seed: int,
        seed_capability: State4SeedCapability,
    ) -> None:
        self._split = split
        self._authorization_file_sha256 = authorization_file_sha256
        self._expected_seed = expected_seed
        self._seed_capability = seed_capability

    def require_active(self) -> None:
        capability = self._seed_capability
        if capability._expired is not False:
            raise RngGuardViolation("invalid or expired State 4 seed capability")
        if capability.authorization_file_sha256 != self._authorization_file_sha256:
            raise RngGuardViolation("State 4 authorization commitment changed")
        counts = capability.access_count_by_split
        if not isinstance(counts, dict) or set(counts) != set(SPLITS):
            raise RngGuardViolation("invalid State 4 seed access counters")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
            for value in counts.values()
        ):
            raise RngGuardViolation("invalid State 4 seed access counter value")
        if counts[self._split] != 1:
            raise RngGuardViolation("bound State 4 seed was not retrieved exactly once")

    def __call__(self, record: Mapping[str, Any]) -> None:
        self.require_active()
        _validate_split_manifest_fields(
            record,
            expected_split=self._split,
            expected_authorization_file_sha256=self._authorization_file_sha256,
            expected_seed=self._expected_seed,
        )


def build_state4_split_manifest_validator(
    *,
    split: str,
    seed_capability: State4SeedCapability,
) -> _BoundSplitManifestValidator:
    checked_split = _split(split)
    if type(seed_capability) is not State4SeedCapability:
        raise RngGuardViolation("invalid State 4 seed capability type")
    if seed_capability._expired is not False:
        raise RngGuardViolation("invalid or expired State 4 seed capability")
    authorization_file_sha256 = seed_capability.authorization_file_sha256
    if (
        not isinstance(authorization_file_sha256, str)
        or len(authorization_file_sha256) != 64
        or any(character not in "0123456789abcdef" for character in authorization_file_sha256)
    ):
        raise RngGuardViolation("invalid State 4 authorization commitment")
    counts = seed_capability.access_count_by_split
    if not isinstance(counts, dict) or set(counts) != set(SPLITS):
        raise RngGuardViolation("invalid State 4 seed access counters")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
        for value in counts.values()
    ):
        raise RngGuardViolation("invalid State 4 seed access counter value")
    if counts[checked_split] != 0:
        raise RngGuardViolation(f"State 4 seed already retrieved for {checked_split}")
    expected_seed = seed_capability.retrieve_once(
        checked_split,
        lambda: _state4_frozen_split_seed(checked_split),
    )
    validator = _BoundSplitManifestValidator(
        split=checked_split,
        authorization_file_sha256=authorization_file_sha256,
        expected_seed=expected_seed,
        seed_capability=seed_capability,
    )
    validator.require_active()
    return validator


def validate_split_manifest(record: Mapping[str, Any]) -> None:
    raise State4RecordError(
        "State 4 split manifest validation requires a bound State 4 seed capability"
    )


def validate_rng_audit(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, RNG_AUDIT_KEYS, name="State 4 RNG audit")
    if record["schema"] != "operatormark.darcy_branch_c_v1.state4_rng_audit.v1" or record["benchmark"] != BENCHMARK or record["state"] != "state4_controls_rng_audit_complete":
        raise State4RecordError("RNG-audit identity mismatch")
    commitments = record["rng_state_commitments"]
    checkpoints = (
        "after_base_import", "after_guard_install", "after_project_import", "after_training",
        "after_validation", "after_test", "after_execution",
    )
    if not isinstance(commitments, dict):
        raise State4RecordError("RNG commitment checkpoints mismatch")
    require_exact_keys(commitments, set(checkpoints), name="RNG commitment checkpoints")
    commitment_keys = {"numpy_legacy_sha256", "python_random_sha256"}
    for checkpoint in checkpoints:
        evidence = commitments[checkpoint]
        if not isinstance(evidence, dict):
            raise State4RecordError(f"{checkpoint} commitment evidence must be an object")
        require_exact_keys(evidence, commitment_keys, name=checkpoint)
        require_sha256(evidence["numpy_legacy_sha256"], name=f"{checkpoint}.numpy")
        require_sha256(evidence["python_random_sha256"], name=f"{checkpoint}.python")
    equalities = record["rng_state_equalities_to_after_base_import"]
    post = checkpoints[1:]
    if not isinstance(equalities, dict):
        raise State4RecordError("RNG equality checkpoints mismatch")
    require_exact_keys(equalities, set(post), name="RNG equality checkpoints")
    equality_keys = {
        "numpy_legacy_equals_baseline", "python_random_equals_baseline",
    }
    baseline = commitments["after_base_import"]
    for checkpoint in post:
        evidence = equalities[checkpoint]
        if not isinstance(evidence, dict):
            raise State4RecordError(f"{checkpoint} equality evidence must be an object")
        require_exact_keys(evidence, equality_keys, name=checkpoint)
        for name in equality_keys:
            if _boolean(evidence[name], name=f"{checkpoint}.{name}") is not True:
                raise State4RecordError("RNG equality must be true")
        if (
            commitments[checkpoint]["numpy_legacy_sha256"]
            != baseline["numpy_legacy_sha256"]
            or commitments[checkpoint]["python_random_sha256"]
            != baseline["python_random_sha256"]
        ):
            raise State4RecordError("RNG commitment differs from the baseline")
    ones = {split: 1 for split in SPLITS}
    authorized_counts = (
        ("authorized_corpus_seed_access_count_by_split", "seed access", ones),
        ("authorized_pcg64_constructor_count_by_split", "PCG64 constructors", ones),
        ("authorized_generator_constructor_count_by_split", "Generator constructors", ones),
        ("authorized_top_level_rng_api_call_count_by_split", "top-level calls", SPLIT_COUNTS),
    )
    for field, name, expected in authorized_counts:
        counts = record[field]
        if not isinstance(counts, dict):
            raise State4RecordError(f"{name} must be an object")
        _exact_split_map(counts, name=name, expected=expected)
        for split in SPLITS:
            _integer(counts[split], name=f"{name}.{split}")
    forbidden_rng_calls = _integer(
        record["forbidden_rng_call_count"], name="forbidden_rng_call_count",
    )
    forbidden_seed_access = _integer(
        record["forbidden_or_out_of_scope_corpus_seed_access_count"],
        name="forbidden_or_out_of_scope_corpus_seed_access_count",
    )
    all_checks_passed = _boolean(
        record["all_rng_checks_passed"], name="all_rng_checks_passed",
    )
    if forbidden_rng_calls != 0 or forbidden_seed_access != 0 or all_checks_passed is not True:
        raise State4RecordError("RNG audit forbidden counts or final gate mismatch")
    for name in (
        "protocol_file_sha256", "state4_authorization_file_sha256",
        "frozen_base_import_closure_sha256", "record_self_sha256",
    ):
        require_sha256(record[name], name=name)
    for name in ("implementation_commit", "implementation_tree"):
        require_git_oid(record[name], name=name)
    validate_self_hash(record, "record_self_sha256")


def validate_read_only_evidence(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, READ_ONLY_KEYS, name="State 4 read-only evidence")
    if record["schema"] != "operatormark.darcy_branch_c_v1.state4_read_only_evidence.v1":
        raise State4RecordError("read-only evidence schema mismatch")
    _attempt_id(record["attempt_id"])
    _nonempty_string(record["permission_mechanism"], name="permission_mechanism")
    paths = record["ordered_relative_paths"]
    required_paths = _required_read_only_paths()
    if not isinstance(paths, list) or len(paths) != len(required_paths):
        raise State4RecordError("ordered read-only paths must cover every required State 4 file")
    checked_paths = [
        _relative_path(path, name=f"ordered_relative_paths[{index}]")
        for index, path in enumerate(paths)
    ]
    if len(set(checked_paths)) != len(checked_paths) or set(checked_paths) != required_paths:
        raise State4RecordError("ordered read-only paths do not exactly cover required State 4 files")
    _hash_list(record["ordered_file_sha256"], name="ordered_file_sha256", count=len(paths))
    if not isinstance(record["ordered_permission_evidence"], list) or len(record["ordered_permission_evidence"]) != len(paths):
        raise State4RecordError("permission evidence length mismatch")
    if _boolean(
        record["all_required_files_read_only"], name="all_required_files_read_only",
    ) is not True:
        raise State4RecordError("all required State 4 files must be read-only")
    validate_self_hash(record, "record_self_sha256")
    raise State4RecordError(
        "nested State 4 permission evidence schema is unresolved; "
        "production read-only proof cannot be validated"
    )


def validate_read_only_evidence_metadata(
    record: Mapping[str, Any],
    *,
    expected_context: Mapping[str, Any],
    expected_files: Sequence[Mapping[str, Any]],
) -> None:
    context_keys = {
        "protocol_file_sha256",
        "state4_authorization_file_sha256",
        "implementation_commit",
        "implementation_tree",
        "storage_preflight_file_sha256",
        "permission_profile_file_sha256",
        "repository_root_commitment",
        "volume_identifier_commitment",
        "accepted_local_user_residual_principal_commitments",
    }
    if not isinstance(expected_context, Mapping):
        raise State4RecordError("expected permission metadata context must be an object")
    require_exact_keys(
        expected_context,
        context_keys,
        name="expected permission metadata context",
    )
    protocol_sha256 = require_sha256(
        expected_context["protocol_file_sha256"],
        name="expected_context.protocol_file_sha256",
    )
    if protocol_sha256 != PROTOCOL_SHA256:
        raise State4RecordError("permission metadata protocol commitment mismatch")
    authorization_sha256 = require_sha256(
        expected_context["state4_authorization_file_sha256"],
        name="expected_context.state4_authorization_file_sha256",
    )
    for name in (
        "storage_preflight_file_sha256",
        "permission_profile_file_sha256",
        "repository_root_commitment",
        "volume_identifier_commitment",
    ):
        require_sha256(expected_context[name], name=f"expected_context.{name}")
    for name in ("implementation_commit", "implementation_tree"):
        require_git_oid(expected_context[name], name=f"expected_context.{name}")
    residual_commitments = expected_context[
        "accepted_local_user_residual_principal_commitments"
    ]
    if not isinstance(residual_commitments, list):
        raise State4RecordError(
            "expected residual principal commitments must be a JSON array"
        )
    checked_residual_commitments = [
        require_sha256(
            commitment,
            name=(
                "expected_context.accepted_local_user_residual_principal_commitments"
                f"[{index}]"
            ),
        )
        for index, commitment in enumerate(residual_commitments)
    ]
    if checked_residual_commitments != sorted(set(checked_residual_commitments)):
        raise State4RecordError(
            "expected residual principal commitments must be sorted and unique"
        )

    required_paths = _ordered_required_read_only_paths()
    if not isinstance(expected_files, list) or len(expected_files) != len(required_paths):
        raise State4RecordError("expected permission metadata inventory must contain 232 files")
    checked_expected_files: list[Mapping[str, Any]] = []
    expected_hashes: list[str] = []
    for index, expected_file in enumerate(expected_files):
        if not isinstance(expected_file, Mapping):
            raise State4RecordError(f"expected_files[{index}] must be an object")
        require_exact_keys(
            expected_file,
            {"relative_path", "file_bytes", "file_sha256"},
            name=f"expected_files[{index}]",
        )
        relative_path = _relative_path(
            expected_file["relative_path"],
            name=f"expected_files[{index}].relative_path",
        )
        if relative_path != required_paths[index]:
            raise State4RecordError("expected permission metadata path order mismatch")
        _integer(
            expected_file["file_bytes"],
            name=f"expected_files[{index}].file_bytes",
            minimum=1,
        )
        expected_hashes.append(
            require_sha256(
                expected_file["file_sha256"],
                name=f"expected_files[{index}].file_sha256",
            )
        )
        checked_expected_files.append(expected_file)

    require_exact_keys(record, READ_ONLY_KEYS, name="State 4 read-only evidence metadata")
    if record["schema"] != "operatormark.darcy_branch_c_v1.state4_read_only_evidence.v1":
        raise State4RecordError("read-only evidence metadata schema mismatch")
    attempt_id = _attempt_id(record["attempt_id"])
    if attempt_id != _expected_attempt_id(authorization_sha256):
        raise State4RecordError(
            "read-only evidence metadata attempt_id does not match State 4 authorization"
        )
    if record["permission_mechanism"] != STATE4_FILE_PERMISSION_METADATA_PROFILE:
        raise State4RecordError("read-only evidence metadata permission mechanism mismatch")

    paths = record["ordered_relative_paths"]
    if not isinstance(paths, list) or len(paths) != len(required_paths):
        raise State4RecordError("ordered read-only metadata paths must contain 232 files")
    checked_paths = [
        _relative_path(path, name=f"ordered_relative_paths[{index}]")
        for index, path in enumerate(paths)
    ]
    if tuple(checked_paths) != required_paths:
        raise State4RecordError("ordered read-only metadata path order mismatch")

    ordered_hashes = _hash_list(
        record["ordered_file_sha256"],
        name="ordered_file_sha256",
        count=len(required_paths),
    )
    if ordered_hashes != expected_hashes:
        raise State4RecordError("ordered read-only metadata hashes do not match inventory")
    permission_rows = record["ordered_permission_evidence"]
    if not isinstance(permission_rows, list) or len(permission_rows) != len(required_paths):
        raise State4RecordError("ordered permission metadata must contain 232 rows")
    if _boolean(
        record["all_required_files_read_only"],
        name="all_required_files_read_only",
    ) is not True:
        raise State4RecordError("all required State 4 files must report read-only metadata")

    file_identity_commitments: list[str] = []
    for index, permission_row in enumerate(permission_rows):
        try:
            validate_state4_file_permission_metadata(
                permission_row,
                expected_context=expected_context,
                expected_index=index,
                expected_file=checked_expected_files[index],
            )
        except StoragePreflightError as exc:
            raise State4RecordError(str(exc)) from exc
        file_identity_commitments.append(
            permission_row["path_evidence"]["file_identity_commitment"]
        )
    if len(set(file_identity_commitments)) != len(file_identity_commitments):
        raise State4RecordError("permission metadata file identities must be unique")
    validate_self_hash(record, "record_self_sha256")


def validate_pre_solve_manifest(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, PRE_SOLVE_KEYS, name="State 4 pre-solve manifest")
    if record["schema"] != "operatormark.darcy_branch_c_v1.state4_pre_solve_manifest.v2" or record["state"] != "controls_prepared_no_pde_solve" or record["benchmark"] != BENCHMARK:
        raise State4RecordError("pre-solve manifest identity mismatch")
    if record["corpus_version"] != CORPUS_VERSION:
        raise State4RecordError("corpus_version mismatch")
    protocol_relative_name = _relative_path(
        record["protocol_relative_name"], name="protocol_relative_name",
    )
    if protocol_relative_name != PROTOCOL_RELATIVE_NAME:
        raise State4RecordError("protocol_relative_name mismatch")
    authorization_sha256 = require_sha256(
        record["state4_authorization_file_sha256"],
        name="state4_authorization_file_sha256",
    )
    attempt_id = _attempt_id(record["attempt_id"])
    if attempt_id != _expected_attempt_id(authorization_sha256):
        raise State4RecordError("attempt_id does not match State 4 authorization hash")
    if record["protocol_file_sha256"] != PROTOCOL_SHA256:
        raise State4RecordError("protocol_file_sha256 mismatch")
    rng_audit_path = _relative_path(
        record["rng_audit_relative_path"], name="rng_audit_relative_path",
    )
    read_only_path = _relative_path(
        record["read_only_evidence_relative_path"],
        name="read_only_evidence_relative_path",
    )
    sample_count = _integer(record["sample_count"], name="sample_count")
    shard_count = _integer(record["shard_count"], name="shard_count")
    if (
        rng_audit_path != "state4_controls/rng_audit.json"
        or read_only_path != "state4_controls/read_only_evidence.json"
        or sample_count != 4864
        or shard_count != 76
    ):
        raise State4RecordError("pre-solve fixed values mismatch")
    if record["split_order"] != list(SPLITS):
        raise State4RecordError("pre-solve split order or RNG counts mismatch")
    split_paths = record["split_manifest_paths"]
    expected_split_paths = [_split_manifest_relative_path(split) for split in SPLITS]
    if not isinstance(split_paths, list) or len(split_paths) != len(SPLITS):
        raise State4RecordError("split_manifest_paths length mismatch")
    checked_split_paths = [
        _relative_path(path, name=f"split_manifest_paths[{index}]")
        for index, path in enumerate(split_paths)
    ]
    if checked_split_paths != expected_split_paths:
        raise State4RecordError("split_manifest_paths mismatch")
    _hash_list(
        record["split_manifest_file_sha256"],
        name="split_manifest_file_sha256",
        count=len(SPLITS),
    )
    _exact_split_map(
        record["rng_call_count_by_split"],
        name="rng_call_count_by_split",
        expected=SPLIT_COUNTS,
    )
    forbidden_count = _integer(
        record["forbidden_seed_or_data_access_count"],
        name="forbidden_seed_or_data_access_count",
    )
    controls_unique = _boolean(
        record["all_control_hashes_unique"], name="all_control_hashes_unique",
    )
    macro_unique = _boolean(
        record["all_macro_coefficient_hashes_unique"],
        name="all_macro_coefficient_hashes_unique",
    )
    if forbidden_count != 0 or controls_unique is not True or macro_unique is not True:
        raise State4RecordError("pre-solve gates mismatch")
    for name in (
        "protocol_file_sha256", "state4_authorization_file_sha256",
        "state3_audit_file_sha256", "compute_feasibility_report_file_sha256",
        "environment_record_sha256", "storage_preflight_before_file_sha256",
        "storage_preflight_after_file_sha256", "rng_audit_file_sha256",
        "read_only_evidence_file_sha256", "global_control_aggregate_sha256",
        "global_macro_coefficient_aggregate_sha256", "manifest_self_sha256",
    ):
        require_sha256(record[name], name=name)
    if record["solver_spec_baseline_commit"] != SOLVER_SPEC_BASELINE_COMMIT:
        raise State4RecordError("solver_spec_baseline_commit mismatch")
    if record["solver_spec_baseline_tree"] != SOLVER_SPEC_BASELINE_TREE:
        raise State4RecordError("solver_spec_baseline_tree mismatch")
    implementation_commit = require_git_oid(
        record["implementation_commit"], name="implementation_commit",
    )
    implementation_tree = require_git_oid(
        record["implementation_tree"], name="implementation_tree",
    )
    git_before = _validate_git_record(
        record["git_before"],
        name="git_before",
        implementation_commit=implementation_commit,
        implementation_tree=implementation_tree,
    )
    git_after = _validate_git_record(
        record["git_after"],
        name="git_after",
        implementation_commit=implementation_commit,
        implementation_tree=implementation_tree,
    )
    if git_before != git_after:
        raise State4RecordError("git_before and git_after must bind the same clean implementation")
    validate_self_hash(record, "manifest_self_sha256")


def validate_completion_marker(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, COMPLETE_KEYS, name="State 4 completion marker")
    if record["schema"] != "operatormark.darcy_branch_c_v1.state4_controls_complete.v1" or record["state"] != "state4_controls_accepted_no_solutions":
        raise State4RecordError("completion marker identity mismatch")
    authorization_sha256 = require_sha256(
        record["state4_authorization_file_sha256"],
        name="state4_authorization_file_sha256",
    )
    if _attempt_id(record["attempt_id"]) != _expected_attempt_id(authorization_sha256):
        raise State4RecordError("attempt_id does not match State 4 authorization hash")
    pre_solve_path = _relative_path(
        record["pre_solve_manifest_relative_path"],
        name="pre_solve_manifest_relative_path",
    )
    if pre_solve_path != "state4_controls/pre_solve_manifest.json":
        raise State4RecordError("pre_solve_manifest_relative_path mismatch")
    _validate_utc_timestamp(record["completion_time_utc"])
    for name in (
        "pre_solve_manifest_file_sha256", "read_only_evidence_file_sha256",
        "state4_authorization_file_sha256", "record_self_sha256",
    ):
        require_sha256(record[name], name=name)
    validate_self_hash(record, "record_self_sha256")


def validate_control_shard_bundle(
    rho_bytes: bytes,
    controls_jsonl_bytes: bytes,
    shard_manifest: Mapping[str, Any],
    *,
    rho_relative_path: str,
    control_records_relative_path: str,
) -> list[dict[str, Any]]:
    """Validate actual State 4 rho/JSONL bytes against one controls manifest."""

    if not isinstance(rho_bytes, bytes) or not isinstance(controls_jsonl_bytes, bytes):
        raise State4RecordError("control shard bundle members must be bytes")
    validate_shard_manifest(shard_manifest)
    actual_rho_path = _relative_path(rho_relative_path, name="rho_relative_path")
    actual_controls_path = _relative_path(
        control_records_relative_path, name="control_records_relative_path",
    )
    if actual_rho_path != shard_manifest["rho_relative_path"]:
        raise State4RecordError("actual rho path does not match shard manifest")
    if actual_controls_path != shard_manifest["control_records_relative_path"]:
        raise State4RecordError("actual controls JSONL path does not match shard manifest")

    try:
        rho = validate_npy_v2_bytes(rho_bytes, expected_shape=(64, 1, 8, 8))
    except CanonicalRecordError as exc:
        raise State4RecordError(str(exc)) from exc
    if len(rho_bytes) != shard_manifest["rho_file_bytes"]:
        raise State4RecordError("rho file byte count mismatch")
    if sha256_bytes(rho_bytes) != shard_manifest["rho_file_sha256"]:
        raise State4RecordError("rho file SHA-256 mismatch")
    if tensor_sha256(rho) != shard_manifest["rho_tensor_sha256"]:
        raise State4RecordError("rho shard tensor SHA-256 mismatch")

    split = str(shard_manifest["split"])
    shard_index = int(shard_manifest["shard_index"])
    try:
        records = validate_control_records_jsonl(
            controls_jsonl_bytes, split=split, shard_index=shard_index,
        )
    except CanonicalRecordError as exc:
        raise State4RecordError(str(exc)) from exc
    if len(controls_jsonl_bytes) != shard_manifest["control_records_bytes"]:
        raise State4RecordError("controls JSONL byte count mismatch")
    if sha256_bytes(controls_jsonl_bytes) != shard_manifest["control_records_file_sha256"]:
        raise State4RecordError("controls JSONL file SHA-256 mismatch")
    ordered_record_hashes = [str(record["record_self_sha256"]) for record in records]
    if ordered_record_hashes != shard_manifest["ordered_record_self_sha256"]:
        raise State4RecordError("controls JSONL ordered record hashes mismatch")

    for offset, record in enumerate(records):
        storage = rho[offset]
        control = storage[0]
        if record["stratum"] == "relaxed":
            if np.signbit(control).any() or not np.logical_and(control >= 0.0, control < 1.0).all():
                raise State4RecordError("relaxed control is outside frozen [0,1) domain")
        else:
            bit_patterns = control.view(np.dtype("<u8"))
            zero_count = int(np.count_nonzero(bit_patterns == np.uint64(0x0000000000000000)))
            one_count = int(np.count_nonzero(bit_patterns == np.uint64(0x3FF0000000000000)))
            if zero_count != 32 or one_count != 32 or zero_count + one_count != control.size:
                raise State4RecordError("binary control must contain exactly 32 zeros and 32 ones")
        macro = np.ascontiguousarray(
            np.float64(3.0) + np.float64(9.0) * control,
            dtype=np.dtype("<f8"),
        )
        if tensor_sha256(control) != record["control_tensor_sha256"]:
            raise State4RecordError("control tensor SHA-256 mismatch")
        if tensor_sha256(storage) != record["rho_storage_tensor_sha256"]:
            raise State4RecordError("rho storage tensor SHA-256 mismatch")
        if tensor_sha256(macro) != record["macro_coefficient_tensor_sha256"]:
            raise State4RecordError("macro-coefficient tensor SHA-256 mismatch")
    return records


def _preflight_record_validator(validator: Any) -> None:
    if validator is validate_split_manifest:
        raise State4RecordError(
            "State 4 split manifest validation requires a bound State 4 seed capability"
        )
    if isinstance(validator, _BoundSplitManifestValidator):
        validator.require_active()


def validate_record_file(path: str | Path, validator: Any) -> dict[str, Any]:
    _preflight_record_validator(validator)
    digest_before = file_sha256(path)
    data = Path(path).read_bytes()
    _preflight_record_validator(validator)
    record = strict_canonical_json_loads(data)
    validator(record)
    if digest_before != file_sha256(path):
        raise State4RecordError("record changed during validation")
    return record


def write_record_exclusive(path: str | Path, record: Mapping[str, Any], validator: Any) -> bytes:
    _preflight_record_validator(validator)
    validator(record)
    _preflight_record_validator(validator)
    data = write_canonical_json_exclusive(path, record)
    _preflight_record_validator(validator)
    readback = strict_canonical_json_loads(data)
    validator(readback)
    return data


def write_controls_jsonl_exclusive(
    path: str | Path, records: Sequence[Mapping[str, Any]], *, split: str, shard_index: int
) -> bytes:
    data = canonical_jsonl_bytes(records)
    validate_control_records_jsonl(data, split=split, shard_index=shard_index)
    written = write_canonical_jsonl_exclusive(path, records)
    validate_control_records_jsonl(written, split=split, shard_index=shard_index)
    return written


def write_rho_shard_exclusive(path: str | Path, array: np.ndarray) -> bytes:
    if array.shape != (64, 1, 8, 8):
        raise State4RecordError("rho shard shape must be exactly [64,1,8,8]")
    data = write_npy_v2_exclusive(path, array)
    validate_npy_v2_bytes(data, expected_shape=(64, 1, 8, 8))
    return data


def set_read_only(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    before = stat.S_IMODE(target.stat().st_mode)
    target.chmod(before & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    after = stat.S_IMODE(target.stat().st_mode)
    writable = bool(after & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    if writable:
        raise State4RecordError("file remained writable after read-only transition")
    return {"mechanism": "chmod_remove_write_bits", "mode_before_octal": oct(before), "mode_after_octal": oct(after)}


def split_control_aggregate(child_hashes: Sequence[str]) -> str:
    return domain_separated_digest("operatormark-state4-split-control-aggregate-v1\n", child_hashes)


def split_macro_coefficient_aggregate(child_hashes: Sequence[str]) -> str:
    return domain_separated_digest("operatormark-state4-split-macro-coefficient-aggregate-v1\n", child_hashes)


def global_control_aggregate(split_hashes: Sequence[str]) -> str:
    if len(split_hashes) != 3:
        raise State4RecordError("global control aggregate requires three split hashes")
    return domain_separated_digest("operatormark-state4-global-control-aggregate-v1\n", split_hashes)


def global_macro_coefficient_aggregate(split_hashes: Sequence[str]) -> str:
    if len(split_hashes) != 3:
        raise State4RecordError("global macro aggregate requires three split hashes")
    return domain_separated_digest("operatormark-state4-global-macro-coefficient-aggregate-v1\n", split_hashes)


def run_injected_controls(*, expected, independent_inputs, ports, mode):
    """Explicit synthetic bridge only; existing production entry remains refused."""
    from .state4_workflow import run_state4_controls
    return run_state4_controls(expected=expected, independent_inputs=independent_inputs,
                               ports=ports, mode=mode)


def main() -> int:
    if os.environ.get("OPERATORMARK_STATE") != "state4":
        raise State4RecordError("State 4 preparation requires separate exact authorization")
    raise State4RecordError("State 4 control generation is not authorized by State 2")


if __name__ == "__main__":
    raise SystemExit(main())
