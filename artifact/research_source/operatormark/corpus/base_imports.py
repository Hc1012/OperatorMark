"""Version-pinned passive numerical base-import closure commitments."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Final


BASE_IMPORT_ROOTS: Final = (
    "numpy",
    "numpy.typing",
    "scipy",
    "scipy.sparse",
    "scipy.sparse.linalg",
)
NUMERICAL_PREFIXES: Final = ("numpy", "scipy")
CLOSURE_SCHEMA: Final = "operatormark.darcy_branch_c_v1.base_import_closure.v1"
CLOSURE_KEYS: Final = frozenset(
    {
        "closure_sha256",
        "numpy_version",
        "ordered_modules",
        "root_modules",
        "schema",
        "scipy_version",
    }
)
MODULE_RECORD_KEYS: Final = frozenset({"file_sha256", "kind", "module", "suffix"})
FROZEN_NUMPY_VERSION: Final = "2.5.2"
FROZEN_SCIPY_VERSION: Final = "1.18.1"
SOURCE_SUFFIXES: Final = frozenset({".py", ".pyi"})
EXTENSION_SUFFIXES: Final = frozenset({".pyd", ".so", ".dll", ".dylib"})


class BaseImportClosureError(RuntimeError):
    """Raised when the passive numerical import closure cannot be committed."""


def load_base_import_roots() -> tuple[ModuleType, ...]:
    """Import only the roots needed by the frozen Darcy solvers and harness."""

    return tuple(importlib.import_module(name) for name in BASE_IMPORT_ROOTS)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _closure_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "numpy_version": record["numpy_version"],
        "ordered_modules": record["ordered_modules"],
        "root_modules": record["root_modules"],
        "schema": record["schema"],
        "scipy_version": record["scipy_version"],
    }


def validate_base_import_closure(record: Mapping[str, Any]) -> None:
    """Reject any non-canonical or internally inconsistent closure record."""

    if not isinstance(record, dict) or set(record) != CLOSURE_KEYS:
        raise BaseImportClosureError("base-import closure has non-canonical top-level keys")
    if record["schema"] != CLOSURE_SCHEMA:
        raise BaseImportClosureError("base-import closure schema mismatch")
    if record["root_modules"] != list(BASE_IMPORT_ROOTS):
        raise BaseImportClosureError("base-import closure root-module mismatch")
    if record["numpy_version"] != FROZEN_NUMPY_VERSION:
        raise BaseImportClosureError("base-import closure NumPy version mismatch")
    if record["scipy_version"] != FROZEN_SCIPY_VERSION:
        raise BaseImportClosureError("base-import closure SciPy version mismatch")

    modules = record["ordered_modules"]
    if not isinstance(modules, list) or not modules:
        raise BaseImportClosureError("base-import closure module records must be a non-empty list")
    names: list[str] = []
    for item in modules:
        if not isinstance(item, dict) or set(item) != MODULE_RECORD_KEYS:
            raise BaseImportClosureError("base-import closure module record keys mismatch")
        name = item["module"]
        if not isinstance(name, str) or not any(
            name == prefix or name.startswith(prefix + ".") for prefix in NUMERICAL_PREFIXES
        ):
            raise BaseImportClosureError("base-import closure contains a non-numerical module")
        names.append(name)
        suffix = item["suffix"]
        if not isinstance(suffix, str) or suffix != suffix.lower():
            raise BaseImportClosureError("base-import closure suffix is not canonical")
        if item["kind"] == "source_module":
            if not any(suffix.endswith(expected) for expected in SOURCE_SUFFIXES):
                raise BaseImportClosureError("source module has an unsupported suffix")
            if not _is_sha256(item["file_sha256"]):
                raise BaseImportClosureError("source module has no valid SHA-256")
        elif item["kind"] == "extension_binary":
            if not any(suffix.endswith(expected) for expected in EXTENSION_SUFFIXES):
                raise BaseImportClosureError("extension module has an unsupported suffix")
            if not _is_sha256(item["file_sha256"]):
                raise BaseImportClosureError("extension module has no valid SHA-256")
        elif item["kind"] == "namespace_package_no_module_bytes":
            if item["file_sha256"] is not None or suffix != "":
                raise BaseImportClosureError("namespace-package record contains byte identity")
        else:
            raise BaseImportClosureError("base-import closure module kind is not canonical")
    if names != sorted(set(names)):
        raise BaseImportClosureError("base-import closure modules are not unique and sorted")
    if not set(BASE_IMPORT_ROOTS).issubset(names):
        raise BaseImportClosureError("base-import closure omits a frozen root module")

    expected_hash = hashlib.sha256(
        json.dumps(
            _closure_payload(record),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if record["closure_sha256"] != expected_hash:
        raise BaseImportClosureError("base-import closure self-commitment mismatch")


def capture_loaded_base_import_closure() -> dict[str, Any]:
    """Commit the exact loaded NumPy/SciPy transitive module closure."""

    records: list[dict[str, Any]] = []
    for name in sorted(
        candidate
        for candidate in sys.modules
        if any(candidate == prefix or candidate.startswith(prefix + ".") for prefix in NUMERICAL_PREFIXES)
    ):
        module = sys.modules[name]
        source = getattr(module, "__file__", None)
        if isinstance(source, str):
            path = Path(source).resolve(strict=True)
            if path.suffix.casefold() in {".pyc", ".pyo"}:
                try:
                    source_path = Path(importlib.util.source_from_cache(str(path))).resolve(
                        strict=True
                    )
                except (ValueError, OSError) as exc:
                    raise BaseImportClosureError(
                        f"numerical module has bytecode but no source bytes: {name}"
                    ) from exc
                path = source_path
            final_suffix = path.suffix.casefold()
            if final_suffix in SOURCE_SUFFIXES:
                kind = "source_module"
            elif final_suffix in EXTENSION_SUFFIXES:
                kind = "extension_binary"
            else:
                raise BaseImportClosureError(
                    f"numerical module has unsupported byte identity: {name}"
                )
            records.append(
                {
                    "file_sha256": _file_sha256(path),
                    "kind": kind,
                    "module": name,
                    "suffix": "".join(path.suffixes).lower(),
                }
            )
        elif getattr(getattr(module, "__spec__", None), "submodule_search_locations", None) is not None:
            records.append(
                {
                    "file_sha256": None,
                    "kind": "namespace_package_no_module_bytes",
                    "module": name,
                    "suffix": "",
                }
            )
        else:
            raise BaseImportClosureError(f"numerical module has no commit-able identity: {name}")
    if not records:
        raise BaseImportClosureError("numerical base-import closure is empty")
    payload = {
        "numpy_version": sys.modules["numpy"].__version__,
        "ordered_modules": records,
        "root_modules": list(BASE_IMPORT_ROOTS),
        "schema": CLOSURE_SCHEMA,
        "scipy_version": sys.modules["scipy"].__version__,
    }
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    record = {**payload, "closure_sha256": hashlib.sha256(raw).hexdigest()}
    validate_base_import_closure(record)
    return record


def load_and_capture_base_import_closure() -> dict[str, Any]:
    load_base_import_roots()
    return capture_loaded_base_import_closure()


def verify_base_import_closure_equal(expected: Mapping[str, Any]) -> dict[str, Any]:
    """Reinspect the loaded closure and require exact canonical equality."""

    validate_base_import_closure(expected)
    observed = capture_loaded_base_import_closure()
    validate_base_import_closure(observed)
    if observed != expected:
        raise BaseImportClosureError("loaded numerical import closure differs from its baseline")
    return observed
