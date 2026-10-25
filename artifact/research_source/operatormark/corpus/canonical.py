"""Canonical bytes and exclusive writers for the frozen corpus protocol."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")


class CanonicalRecordError(ValueError):
    """Raised when bytes or logical records violate the frozen encoding."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Return compact, sorted-key UTF-8 JSON with no trailing newline."""

    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalRecordError("record is not canonical-JSON encodable") from exc
    return text.encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalRecordError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_canonical_json_loads(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CanonicalRecordError(f"non-finite JSON token: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalRecordError("invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CanonicalRecordError("canonical record root must be a JSON object")
    if data != canonical_json_bytes(value):
        raise CanonicalRecordError("JSON bytes are not canonical compact sorted-key bytes")
    return value


def with_self_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        raise CanonicalRecordError(f"payload already contains self-hash field {field}")
    result = dict(payload)
    result[field] = sha256_bytes(canonical_json_bytes(dict(payload)))
    return result


def validate_self_hash(record: Mapping[str, Any], field: str) -> None:
    if field not in record or not isinstance(record[field], str):
        raise CanonicalRecordError(f"missing string self-hash field {field}")
    payload = dict(record)
    claimed = payload.pop(field)
    if not SHA256_RE.fullmatch(claimed):
        raise CanonicalRecordError(f"invalid SHA-256 field {field}")
    actual = sha256_bytes(canonical_json_bytes(payload))
    if claimed != actual:
        raise CanonicalRecordError(f"self-hash mismatch for {field}")


def require_exact_keys(record: Mapping[str, Any], keys: Iterable[str], *, name: str) -> None:
    expected = set(keys)
    actual = set(record)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CanonicalRecordError(f"{name} keys mismatch; missing={missing}, extra={extra}")


def require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise CanonicalRecordError(f"{name} must be lowercase SHA-256 hexadecimal")
    return value


def require_git_oid(value: object, *, name: str) -> str:
    """Require this repository's full lowercase SHA-1 object-name encoding."""

    if not isinstance(value, str) or not GIT_OID_RE.fullmatch(value):
        raise CanonicalRecordError(f"{name} must be a lowercase 40-hex Git object ID")
    return value


def float64_hex(value: object, *, name: str) -> str:
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        raise CanonicalRecordError(f"{name} must be a finite binary64 scalar")
    try:
        scalar = np.float64(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CanonicalRecordError(f"{name} must be binary64-convertible") from exc
    if not np.isfinite(scalar):
        raise CanonicalRecordError(f"{name} must be finite")
    return float(scalar).hex().lower()


def tensor_sha256(tensor: Any) -> str:
    import numpy as np

    if not isinstance(tensor, np.ndarray) or tensor.dtype.hasobject:
        raise CanonicalRecordError("tensor fingerprint requires a non-object NumPy array")
    contiguous = np.array(tensor, dtype=tensor.dtype, order="C", copy=True, subok=False)
    dimensions = ",".join(str(dimension) for dimension in tensor.shape)
    header = f"dtype={tensor.dtype.str}\nshape={dimensions}\norder=C\n".encode("utf-8")
    return sha256_bytes(header + contiguous.tobytes(order="C"))


def domain_separated_digest(domain: str, child_hashes: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    for index, child in enumerate(child_hashes):
        require_sha256(child, name=f"child_hashes[{index}]")
        digest.update(bytes.fromhex(child))
    return digest.hexdigest()


def _close_preserving_failure(stream: Any, primary: BaseException | None) -> None:
    """Close once, retaining a secondary close error on the original failure."""

    if primary is None:
        stream.close()
        return
    traceback = primary.__traceback__
    cause = primary.__cause__
    context = primary.__context__
    suppress_context = primary.__suppress_context__
    try:
        stream.close()
    except BaseException as cleanup:
        if cleanup is not primary:
            previous = getattr(primary, "_publication_cleanup_failures", ())
            primary._publication_cleanup_failures = previous + (("close", cleanup),)
    finally:
        primary.__traceback__ = traceback
        primary.__cause__ = cause
        primary.__context__ = context
        primary.__suppress_context__ = suppress_context


def _readback_exclusive_bytes(path: Path) -> bytes:
    """Read through an owned stream without masking a read failure on close."""

    stream = path.open("rb")
    try:
        data = stream.read()
    except BaseException as primary:
        _close_preserving_failure(stream, primary)
        raise
    else:
        _close_preserving_failure(stream, None)
    return data


def _publish_exclusive(path: Path, data: bytes) -> None:
    """Publish verified bytes without overwriting an existing destination."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".__creating__")
    if temporary.exists():
        raise FileExistsError(temporary)
    stream = temporary.open("xb")
    try:
        written = stream.write(data)
        if type(written) is not int or written != len(data):
            raise CanonicalRecordError("write did not return the exact byte count")
        stream.flush()
        os.fsync(stream.fileno())
    except BaseException as primary:
        _close_preserving_failure(stream, primary)
        raise
    else:
        _close_preserving_failure(stream, None)
    if _readback_exclusive_bytes(temporary) != data:
        raise CanonicalRecordError("temporary write failed byte-for-byte read-back")
    if os.name == "nt":
        os.rename(temporary, path)
    else:
        os.link(temporary, path)
    if _readback_exclusive_bytes(path) != data:
        raise CanonicalRecordError("published file failed byte-for-byte read-back")
    if os.name != "nt":
        temporary.unlink()


def write_canonical_json_exclusive(path: str | Path, record: Mapping[str, Any]) -> bytes:
    data = canonical_json_bytes(dict(record))
    _publish_exclusive(Path(path), data)
    strict_canonical_json_loads(data)
    return data


def canonical_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    if not records:
        raise CanonicalRecordError("canonical JSONL requires at least one record")
    return b"".join(canonical_json_bytes(dict(record)) + b"\n" for record in records)


def strict_canonical_jsonl_loads(data: bytes) -> list[dict[str, Any]]:
    if not data or not data.endswith(b"\n") or b"\n\n" in data:
        raise CanonicalRecordError("JSONL must be non-empty with one final LF and no blank lines")
    records = [strict_canonical_json_loads(line) for line in data[:-1].split(b"\n")]
    if data != canonical_jsonl_bytes(records):
        raise CanonicalRecordError("JSONL bytes are not canonical")
    return records


def write_canonical_jsonl_exclusive(
    path: str | Path, records: Sequence[Mapping[str, Any]]
) -> bytes:
    data = canonical_jsonl_bytes(records)
    _publish_exclusive(Path(path), data)
    strict_canonical_jsonl_loads(data)
    return data


def npy_v2_bytes(array: Any) -> bytes:
    import numpy as np

    if not isinstance(array, np.ndarray):
        raise CanonicalRecordError("NumPy writer requires ndarray input")
    if array.dtype != np.dtype("<f8") or not array.flags.c_contiguous:
        raise CanonicalRecordError("array must be little-endian float64 and C-contiguous")
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise CanonicalRecordError("array must contain finite non-object values")
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, version=(2, 0), allow_pickle=False)
    return buffer.getvalue()


def validate_npy_v2_bytes(data: bytes, *, expected_shape: tuple[int, ...]) -> Any:
    import numpy as np

    stream = io.BytesIO(data)
    if np.lib.format.read_magic(stream) != (2, 0):
        raise CanonicalRecordError("NumPy file format must be exactly v2.0")
    stream.seek(0)
    try:
        array = np.lib.format.read_array(stream, allow_pickle=False)
    except (ValueError, EOFError) as exc:
        raise CanonicalRecordError("invalid NumPy v2.0 bytes") from exc
    if stream.read(1) != b"":
        raise CanonicalRecordError("NumPy file has trailing bytes")
    if array.shape != expected_shape:
        raise CanonicalRecordError(f"NumPy shape mismatch: {array.shape} != {expected_shape}")
    if array.dtype != np.dtype("<f8") or not array.flags.c_contiguous:
        raise CanonicalRecordError("NumPy array must be <f8 and C-contiguous")
    if not np.isfinite(array).all():
        raise CanonicalRecordError("NumPy array must contain only finite values")
    if data != npy_v2_bytes(array):
        raise CanonicalRecordError("NumPy bytes are not canonical v2.0 writer output")
    return array


def write_npy_v2_exclusive(path: str | Path, array: Any) -> bytes:
    data = npy_v2_bytes(array)
    validate_npy_v2_bytes(data, expected_shape=array.shape)
    _publish_exclusive(Path(path), data)
    validate_npy_v2_bytes(Path(path).read_bytes(), expected_shape=array.shape)
    return data
