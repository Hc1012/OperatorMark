"""Runtime-inspection primitives used only after the fresh-child env gate."""

from __future__ import annotations

import ctypes
import hashlib
import json
import locale
import os
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .launcher import (
    ADDITIONAL_THREAD_CONTROL_VARIABLES,
    ALL_RECOGNIZED_THREAD_VARIABLES,
    LaunchGuardError,
    MANDATORY_PARENT_ENVIRONMENT,
    environment_commitment,
    verify_child_preimport_environment,
)


RUNTIME_SCHEMA: Final = "operatormark.darcy_branch_c_v1.numerical_runtime.v2"
POLICY_CANDIDATE_SCHEMA: Final = (
    "operatormark.darcy_branch_c_v1.additional_thread_policy_candidate.v1"
)
POLICY_CANDIDATE_STATUS: Final = "review_only_candidate_not_state3_configuration"
BACKEND_TOKENS: Final = {
    "openblas": ("openblas",),
    "mkl": ("mkl",),
    "intel_openmp": ("iomp",),
    "openmp": ("libomp", "vcomp", "gomp"),
    "blis": ("blis",),
    "tbb": ("tbb",),
    "lapack": ("lapack",),
    "blas": ("blas",),
}
WRAPPER_INTERFACE_TOKENS: Final = {
    "blas": ("_fblas", "cython_blas", "_cblas"),
    "lapack": ("_flapack", "cython_lapack", "_clapack", "lapack_lite"),
}
NON_OPENBLAS_THREAD_SYMBOLS: Final = {
    "mkl": ("MKL_Get_Max_Threads",),
    "intel_openmp": ("omp_get_max_threads",),
    "openmp": ("omp_get_max_threads",),
    "blis": ("bli_thread_get_num_threads",),
    "tbb": (),
    "lapack": (),
    "blas": (),
}
OPENBLAS_PARALLEL_LAYERS: Final = {0: "sequential", 1: "pthreads", 2: "openmp"}
REVIEW_ONLY_CANDIDATE_VALUES: Final = {"OPENBLAS_DEFAULT_NUM_THREADS": "1"}
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
OPENBLAS_VERSION_PATTERN: Final = re.compile(r"\bOpenBLAS\s+([^\s]+)", re.IGNORECASE)

ACTIVE_BACKEND_RECORD_KEYS: Final = frozenset(
    {
        "architecture",
        "backend",
        "binary_sha256",
        "effective_thread_count",
        "inspection_complete",
        "inspection_symbols",
        "path_commitment",
        "role",
        "runtime_configuration",
        "threading_layer",
        "version",
    }
)
WRAPPER_RECORD_KEYS: Final = frozenset(
    {"architecture", "binary_sha256", "interface", "path_commitment", "role"}
)
INSPECTION_SYMBOL_KEYS: Final = frozenset(
    {"effective_thread_count", "runtime_configuration", "threading_layer"}
)
RUNTIME_RECORD_KEYS: Final = frozenset(
    {
        "active_backend_kinds",
        "active_backends",
        "all_active_backends_inspectable",
        "all_effective_thread_counts_one",
        "all_parent_thread_policy_bound",
        "all_runtime_requirements_satisfied",
        "applicable_additional_thread_variables",
        "bound_applicable_thread_policy",
        "missing_applicable_thread_policy_variables",
        "non_single_thread_backend_binary_sha256",
        "numerical_wrappers",
        "runtime_record_sha256",
        "schema",
        "unexpected_additional_thread_policy_variables",
        "uninspectable_backend_binary_sha256",
    }
)
POLICY_CANDIDATE_KEYS: Final = frozenset(
    {
        "active_backend_kinds",
        "active_threading_layers",
        "applicable_additional_thread_variables",
        "base_import_closure_sha256",
        "candidate_self_sha256",
        "parent_launch_spec_additional_environment",
        "proposed_environment",
        "runtime_record_sha256",
        "schema",
        "status",
        "unresolved_applicable_thread_variables",
    }
)


class RuntimeInspectionError(RuntimeError):
    """Raised when active numerical runtime state cannot be inspected safely."""


def recognized_additional_thread_variables(
    backends: Sequence[str], threading_layers: Sequence[str] = ()
) -> list[str]:
    variables: set[str] = set()
    for backend in backends:
        variables.update(ADDITIONAL_THREAD_CONTROL_VARIABLES.get(backend, ()))
        if backend == "intel_openmp":
            variables.update(ADDITIONAL_THREAD_CONTROL_VARIABLES.get("openmp", ()))
    if "openmp" in threading_layers:
        variables.update(ADDITIONAL_THREAD_CONTROL_VARIABLES.get("openmp", ()))
    return sorted(variables)


def _loaded_module_paths_windows() -> list[Path]:
    if os.name != "nt":
        return []
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.EnumProcessModulesEx.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
    ]
    psapi.EnumProcessModulesEx.restype = ctypes.c_int
    psapi.GetModuleFileNameExW.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong
    ]
    psapi.GetModuleFileNameExW.restype = ctypes.c_ulong
    process = kernel32.GetCurrentProcess()
    LIST_MODULES_ALL = 0x03
    capacity = 256
    modules: Any = None
    count = 0
    for _ in range(8):
        array_type = ctypes.c_void_p * capacity
        candidate_modules = array_type()
        needed = ctypes.c_ulong()
        if not psapi.EnumProcessModulesEx(
            process,
            candidate_modules,
            ctypes.sizeof(candidate_modules),
            ctypes.byref(needed),
            LIST_MODULES_ALL,
        ):
            raise RuntimeInspectionError("EnumProcessModulesEx failed")
        required_count = (
            needed.value + ctypes.sizeof(ctypes.c_void_p) - 1
        ) // ctypes.sizeof(ctypes.c_void_p)
        if required_count <= capacity:
            modules = candidate_modules
            count = required_count
            break
        capacity = max(capacity * 2, required_count)
    if modules is None:
        raise RuntimeInspectionError("loaded-module enumeration did not stabilize")
    paths: list[Path] = []
    for index in range(count):
        observed_path: str | None = None
        buffer_size = 1024
        while buffer_size <= 32768:
            buffer = ctypes.create_unicode_buffer(buffer_size)
            length = int(
                psapi.GetModuleFileNameExW(
                    process, modules[index], buffer, buffer_size
                )
            )
            if length == 0:
                raise RuntimeInspectionError(
                    "GetModuleFileNameExW failed for an enumerated module"
                )
            if length < buffer_size - 1:
                observed_path = buffer.value
                break
            buffer_size *= 2
        if observed_path is None:
            raise RuntimeInspectionError("loaded-module path exceeds inspection capacity")
        paths.append(Path(observed_path))
    return paths


def _loaded_extension_paths() -> list[Path]:
    paths = {
        Path(module.__file__).resolve()
        for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None)
        and str(module.__file__).lower().endswith((".pyd", ".dll", ".so", ".dylib"))
    }
    paths.update(path.resolve() for path in _loaded_module_paths_windows())
    return sorted(paths, key=lambda path: str(path).casefold())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _path_commitment(path: Path, authorization_file_sha256: str) -> str:
    resolved = path.resolve(strict=True)
    normalized_text = os.path.normcase(str(resolved))
    if os.name == "nt":
        normalized_text = normalized_text.casefold()
    return hashlib.sha256(
        b"operatormark-runtime-library-path-v1\n"
        + authorization_file_sha256.encode("ascii")
        + b"\n"
        + normalized_text.encode("utf-8")
    ).hexdigest()


def _wrapper_interface(path: Path) -> str | None:
    lowered = path.name.casefold()
    for interface, tokens in WRAPPER_INTERFACE_TOKENS.items():
        if any(token in lowered for token in tokens):
            return interface
    stem = path.stem.casefold()
    if stem.startswith("_") and "lapack" in stem:
        return "lapack"
    if stem.startswith("_") and "blas" in stem:
        return "blas"
    return None


def _backend_kind(path: Path) -> str | None:
    if _wrapper_interface(path) is not None:
        return None
    lowered = path.name.casefold()
    for backend, tokens in BACKEND_TOKENS.items():
        if any(token in lowered for token in tokens):
            return backend
    return None


def _open_library(path: Path) -> Any:
    return ctypes.CDLL(str(path))


def _derived_openblas_prefixes(path: Path) -> tuple[str, ...]:
    lowered = path.name.casefold()
    marker = lowered.find("openblas")
    prefixes: list[str] = []
    if marker >= 0:
        prefix = lowered[:marker]
        if prefix.startswith("lib"):
            prefix = prefix[3:]
        if prefix and all(character.isalnum() or character == "_" for character in prefix):
            prefixes.append(prefix if prefix.endswith("_") else prefix + "_")
    prefixes.extend(("scipy_", ""))
    return tuple(dict.fromkeys(prefixes))


def _openblas_symbols(path: Path, operation: str) -> tuple[str, ...]:
    base_names = [f"{prefix}openblas_{operation}" for prefix in _derived_openblas_prefixes(path)]
    prefer_ilp64 = "openblas64" in path.name.casefold()
    suffixes = ("64_", "_64_", "", "_") if prefer_ilp64 else ("", "_", "64_", "_64_")
    return tuple(dict.fromkeys(base + suffix for base in base_names for suffix in suffixes))


def _call_integer_symbol(library: Any, symbols: Sequence[str]) -> tuple[int | None, str | None]:
    for symbol in symbols:
        function = getattr(library, symbol, None)
        if function is None:
            continue
        try:
            function.restype = ctypes.c_int
            function.argtypes = []
            value = function()
            if isinstance(value, bool):
                continue
            integer = int(value)
        except (AttributeError, OSError, TypeError, ValueError, ctypes.ArgumentError):
            continue
        return integer, symbol
    return None, None


def _call_text_symbol(library: Any, symbols: Sequence[str]) -> tuple[str | None, str | None]:
    for symbol in symbols:
        function = getattr(library, symbol, None)
        if function is None:
            continue
        try:
            function.restype = ctypes.c_char_p
            function.argtypes = []
            raw = function()
            if not isinstance(raw, bytes):
                continue
            value = raw.decode("utf-8", errors="strict").strip()
        except (AttributeError, OSError, TypeError, ValueError, UnicodeDecodeError, ctypes.ArgumentError):
            continue
        if value and "\x00" not in value:
            return value, symbol
    return None, None


def _inspect_active_backend(path: Path, backend: str) -> dict[str, Any]:
    try:
        library = _open_library(path)
    except OSError:
        library = None

    threads: int | None = None
    thread_symbol: str | None = None
    configuration: str | None = None
    configuration_symbol: str | None = None
    threading_layer: str | None = None
    threading_layer_symbol: str | None = None
    version: str | None = None
    if library is not None and backend == "openblas":
        threads, thread_symbol = _call_integer_symbol(
            library, _openblas_symbols(path, "get_num_threads")
        )
        configuration, configuration_symbol = _call_text_symbol(
            library, _openblas_symbols(path, "get_config")
        )
        parallel, threading_layer_symbol = _call_integer_symbol(
            library, _openblas_symbols(path, "get_parallel")
        )
        threading_layer = OPENBLAS_PARALLEL_LAYERS.get(parallel)
        if configuration is not None:
            match = OPENBLAS_VERSION_PATTERN.search(configuration)
            version = match.group(1) if match is not None else None
    elif library is not None:
        threads, thread_symbol = _call_integer_symbol(
            library, NON_OPENBLAS_THREAD_SYMBOLS.get(backend, ())
        )
        if backend in {"openmp", "intel_openmp"} and threads is not None:
            threading_layer = "openmp"
            threading_layer_symbol = thread_symbol

    if threads is not None and threads <= 0:
        threads = None
        thread_symbol = None
    inspection_complete = all(
        value is not None for value in (threads, thread_symbol, threading_layer, version)
    )
    return {
        "effective_thread_count": threads,
        "inspection_complete": inspection_complete,
        "inspection_symbols": {
            "effective_thread_count": thread_symbol,
            "runtime_configuration": configuration_symbol,
            "threading_layer": threading_layer_symbol,
        },
        "runtime_configuration": configuration,
        "threading_layer": threading_layer,
        "version": version,
    }


def _runtime_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in sorted(RUNTIME_RECORD_KEYS - {"runtime_record_sha256"})}


def _sorted_active_identity(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return record["backend"], record["binary_sha256"], record["path_commitment"]


def _sorted_wrapper_identity(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return record["interface"], record["binary_sha256"], record["path_commitment"]


def validate_numerical_runtime_record(record: Mapping[str, Any]) -> None:
    """Reject extra fields, malformed values, and inconsistent runtime summaries."""

    if not isinstance(record, dict) or set(record) != RUNTIME_RECORD_KEYS:
        raise RuntimeInspectionError("numerical-runtime record has non-canonical top-level keys")
    if record["schema"] != RUNTIME_SCHEMA:
        raise RuntimeInspectionError("numerical-runtime schema mismatch")

    active = record["active_backends"]
    wrappers = record["numerical_wrappers"]
    if not isinstance(active, list) or not isinstance(wrappers, list):
        raise RuntimeInspectionError("numerical-runtime library records must be lists")
    for item in active:
        if not isinstance(item, dict) or set(item) != ACTIVE_BACKEND_RECORD_KEYS:
            raise RuntimeInspectionError("active-backend record keys mismatch")
        if item["role"] != "active_thread_pool" or item["backend"] not in BACKEND_TOKENS:
            raise RuntimeInspectionError("active-backend identity is not canonical")
        if not _is_sha256(item["binary_sha256"]) or not _is_sha256(item["path_commitment"]):
            raise RuntimeInspectionError("active-backend commitment is malformed")
        if not isinstance(item["architecture"], str) or not item["architecture"]:
            raise RuntimeInspectionError("active-backend architecture is missing")
        threads = item["effective_thread_count"]
        if threads is not None and (not isinstance(threads, int) or isinstance(threads, bool) or threads <= 0):
            raise RuntimeInspectionError("effective thread count is not a positive integer or null")
        symbols = item["inspection_symbols"]
        if not isinstance(symbols, dict) or set(symbols) != INSPECTION_SYMBOL_KEYS:
            raise RuntimeInspectionError("active-backend inspection-symbol record mismatch")
        if not all(value is None or isinstance(value, str) and value for value in symbols.values()):
            raise RuntimeInspectionError("active-backend inspection symbol is malformed")
        for name in ("runtime_configuration", "threading_layer", "version"):
            if item[name] is not None and (not isinstance(item[name], str) or not item[name]):
                raise RuntimeInspectionError(f"active-backend {name} is malformed")
        complete = all(
            value is not None
            for value in (
                threads,
                symbols["effective_thread_count"],
                item["threading_layer"],
                item["version"],
            )
        )
        if not isinstance(item["inspection_complete"], bool) or item["inspection_complete"] != complete:
            raise RuntimeInspectionError("active-backend inspection-completeness mismatch")
    if active != sorted(active, key=_sorted_active_identity):
        raise RuntimeInspectionError("active-backend records are not canonically sorted")

    for item in wrappers:
        if not isinstance(item, dict) or set(item) != WRAPPER_RECORD_KEYS:
            raise RuntimeInspectionError("numerical-wrapper record keys mismatch")
        if item["role"] != "numerical_interface_wrapper" or item["interface"] not in {"blas", "lapack"}:
            raise RuntimeInspectionError("numerical-wrapper identity is not canonical")
        if not _is_sha256(item["binary_sha256"]) or not _is_sha256(item["path_commitment"]):
            raise RuntimeInspectionError("numerical-wrapper commitment is malformed")
        if not isinstance(item["architecture"], str) or not item["architecture"]:
            raise RuntimeInspectionError("numerical-wrapper architecture is missing")
    if wrappers != sorted(wrappers, key=_sorted_wrapper_identity):
        raise RuntimeInspectionError("numerical-wrapper records are not canonically sorted")

    backend_kinds = sorted({item["backend"] for item in active})
    if record["active_backend_kinds"] != backend_kinds:
        raise RuntimeInspectionError("active-backend-kind summary mismatch")
    active_threading_layers = sorted(
        {
            item["threading_layer"]
            for item in active
            if isinstance(item["threading_layer"], str)
        }
    )
    applicable = recognized_additional_thread_variables(
        backend_kinds, active_threading_layers
    )
    if record["applicable_additional_thread_variables"] != applicable:
        raise RuntimeInspectionError("applicable additional-thread-variable summary mismatch")
    bound = record["bound_applicable_thread_policy"]
    missing = record["missing_applicable_thread_policy_variables"]
    unexpected = record["unexpected_additional_thread_policy_variables"]
    if not isinstance(bound, dict) or not all(
        name in applicable and isinstance(value, str) and value for name, value in bound.items()
    ):
        raise RuntimeInspectionError("bound additional-thread policy is malformed")
    if missing != sorted(set(applicable) - set(bound)):
        raise RuntimeInspectionError("missing additional-thread-policy summary mismatch")
    recognized_additional = {
        name for names in ADDITIONAL_THREAD_CONTROL_VARIABLES.values() for name in names
    }
    if not isinstance(unexpected, list) or unexpected != sorted(set(unexpected)) or any(
        not isinstance(name, str) or name in applicable or name not in recognized_additional
        for name in unexpected
    ):
        raise RuntimeInspectionError("unexpected additional-thread-policy summary is malformed")

    uninspectable = sorted(item["binary_sha256"] for item in active if not item["inspection_complete"])
    non_single = sorted(
        item["binary_sha256"]
        for item in active
        if item["effective_thread_count"] is not None and item["effective_thread_count"] != 1
    )
    if record["uninspectable_backend_binary_sha256"] != uninspectable:
        raise RuntimeInspectionError("uninspectable-backend summary mismatch")
    if record["non_single_thread_backend_binary_sha256"] != non_single:
        raise RuntimeInspectionError("non-single-thread-backend summary mismatch")
    inspectable = bool(active) and not uninspectable
    single = bool(active) and not non_single and all(
        item["effective_thread_count"] == 1 for item in active
    )
    policy_bound = not missing and not unexpected
    expected_booleans = {
        "all_active_backends_inspectable": inspectable,
        "all_effective_thread_counts_one": single,
        "all_parent_thread_policy_bound": policy_bound,
        "all_runtime_requirements_satisfied": inspectable and single and policy_bound,
    }
    if any(record[name] is not value for name, value in expected_booleans.items()):
        raise RuntimeInspectionError("numerical-runtime boolean summary mismatch")

    expected_hash = hashlib.sha256(
        b"operatormark-numerical-runtime-record-v2\n" + _canonical_bytes(_runtime_payload(record))
    ).hexdigest()
    if record["runtime_record_sha256"] != expected_hash:
        raise RuntimeInspectionError("numerical-runtime self-commitment mismatch")


def inspect_loaded_numerical_runtime(
    *, authorization_file_sha256: str, required_environment: Mapping[str, str]
) -> dict[str, Any]:
    if not _is_sha256(authorization_file_sha256):
        raise RuntimeInspectionError("authorization-file commitment is not lowercase SHA-256")
    if not isinstance(required_environment, Mapping) or not all(
        isinstance(name, str) and isinstance(value, str) and value
        for name, value in required_environment.items()
    ):
        raise RuntimeInspectionError("required environment must be a non-empty string map")
    if any(
        required_environment.get(name) != value
        for name, value in MANDATORY_PARENT_ENVIRONMENT.items()
    ):
        raise RuntimeInspectionError("required environment differs from the mandatory parent policy")
    recognized_additional = {
        name for names in ADDITIONAL_THREAD_CONTROL_VARIABLES.values() for name in names
    }
    unknown_required = sorted(
        set(required_environment) - set(MANDATORY_PARENT_ENVIRONMENT) - recognized_additional
    )
    if unknown_required:
        raise RuntimeInspectionError("required environment contains an unrecognized policy variable")
    for name, value in required_environment.items():
        if os.environ.get(name) != value:
            raise RuntimeInspectionError(
                f"current process environment differs from the bound policy for {name}"
            )
    unexpected_current = sorted(
        name
        for name in ALL_RECOGNIZED_THREAD_VARIABLES
        if name in os.environ and name not in required_environment
    )
    if unexpected_current:
        raise RuntimeInspectionError(
            "current process contains recognized but unbound thread-policy variables"
        )

    backends: list[dict[str, Any]] = []
    wrappers: list[dict[str, Any]] = []
    for path in _loaded_extension_paths():
        wrapper = _wrapper_interface(path)
        if wrapper is not None:
            wrappers.append(
                {
                    "architecture": platform.machine(),
                    "binary_sha256": _file_sha256(path),
                    "interface": wrapper,
                    "path_commitment": _path_commitment(path, authorization_file_sha256),
                    "role": "numerical_interface_wrapper",
                }
            )
            continue
        backend = _backend_kind(path)
        if backend is None:
            continue
        inspection = _inspect_active_backend(path, backend)
        backends.append(
            {
                "architecture": platform.machine(),
                "backend": backend,
                "binary_sha256": _file_sha256(path),
                **inspection,
                "path_commitment": _path_commitment(path, authorization_file_sha256),
                "role": "active_thread_pool",
            }
        )
    backends.sort(key=_sorted_active_identity)
    wrappers.sort(key=_sorted_wrapper_identity)
    backend_kinds = sorted({record["backend"] for record in backends})
    active_threading_layers = sorted(
        {
            record["threading_layer"]
            for record in backends
            if isinstance(record["threading_layer"], str)
        }
    )
    applicable = recognized_additional_thread_variables(
        backend_kinds, active_threading_layers
    )
    bound_policy = {
        variable: required_environment[variable]
        for variable in applicable
        if variable in required_environment
    }
    missing_policy = sorted(set(applicable) - set(bound_policy))
    unexpected_policy = sorted(
        variable
        for variable in required_environment
        if variable in recognized_additional and variable not in applicable
    )
    uninspectable = sorted(
        record["binary_sha256"] for record in backends if not record["inspection_complete"]
    )
    non_single = sorted(
        record["binary_sha256"]
        for record in backends
        if record["effective_thread_count"] is not None and record["effective_thread_count"] != 1
    )
    inspectable = bool(backends) and not uninspectable
    single = bool(backends) and not non_single and all(
        record["effective_thread_count"] == 1 for record in backends
    )
    policy_bound = not missing_policy and not unexpected_policy
    payload = {
        "active_backend_kinds": backend_kinds,
        "active_backends": backends,
        "all_active_backends_inspectable": inspectable,
        "all_effective_thread_counts_one": single,
        "all_parent_thread_policy_bound": policy_bound,
        "all_runtime_requirements_satisfied": inspectable and single and policy_bound,
        "applicable_additional_thread_variables": applicable,
        "bound_applicable_thread_policy": bound_policy,
        "missing_applicable_thread_policy_variables": missing_policy,
        "non_single_thread_backend_binary_sha256": non_single,
        "numerical_wrappers": wrappers,
        "schema": RUNTIME_SCHEMA,
        "unexpected_additional_thread_policy_variables": unexpected_policy,
        "uninspectable_backend_binary_sha256": uninspectable,
    }
    record = {
        **payload,
        "runtime_record_sha256": hashlib.sha256(
            b"operatormark-numerical-runtime-record-v2\n" + _canonical_bytes(payload)
        ).hexdigest(),
    }
    validate_numerical_runtime_record(record)
    return record


def verify_runtime_record_equal(
    expected: Mapping[str, Any],
    *,
    authorization_file_sha256: str,
    required_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Reinspect active runtimes and require exact equality to a strict baseline."""

    validate_numerical_runtime_record(expected)
    observed = inspect_loaded_numerical_runtime(
        authorization_file_sha256=authorization_file_sha256,
        required_environment=required_environment,
    )
    if observed != expected:
        raise RuntimeInspectionError("loaded numerical runtime differs from its baseline")
    return observed


def _policy_candidate_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in sorted(POLICY_CANDIDATE_KEYS - {"candidate_self_sha256"})}


def validate_review_only_thread_policy_candidate(record: Mapping[str, Any]) -> None:
    if not isinstance(record, dict) or set(record) != POLICY_CANDIDATE_KEYS:
        raise RuntimeInspectionError("review-only policy candidate has non-canonical keys")
    if record["schema"] != POLICY_CANDIDATE_SCHEMA or record["status"] != POLICY_CANDIDATE_STATUS:
        raise RuntimeInspectionError("review-only policy candidate identity mismatch")
    if not _is_sha256(record["runtime_record_sha256"]) or not _is_sha256(
        record["base_import_closure_sha256"]
    ):
        raise RuntimeInspectionError("review-only policy candidate binding is malformed")
    applicable = record["applicable_additional_thread_variables"]
    unresolved = record["unresolved_applicable_thread_variables"]
    proposed = record["proposed_environment"]
    items = record["parent_launch_spec_additional_environment"]
    backend_kinds = record["active_backend_kinds"]
    threading_layers = record["active_threading_layers"]
    if (
        not isinstance(backend_kinds, list)
        or backend_kinds != sorted(set(backend_kinds))
        or any(name not in BACKEND_TOKENS for name in backend_kinds)
    ):
        raise RuntimeInspectionError("review-only active-backend kinds are malformed")
    if (
        not isinstance(threading_layers, list)
        or threading_layers != sorted(set(threading_layers))
        or any(
            not isinstance(name, str)
            or name not in set(OPENBLAS_PARALLEL_LAYERS.values()) | {"openmp"}
            for name in threading_layers
        )
    ):
        raise RuntimeInspectionError("review-only active threading layers are malformed")
    if not isinstance(applicable, list) or applicable != sorted(set(applicable)):
        raise RuntimeInspectionError("review-only candidate applicable variables are malformed")
    if applicable != recognized_additional_thread_variables(backend_kinds, threading_layers):
        raise RuntimeInspectionError("review-only candidate backend-policy mapping mismatch")
    if not isinstance(proposed, dict) or not all(
        name in applicable and isinstance(value, str) and value for name, value in proposed.items()
    ):
        raise RuntimeInspectionError("review-only proposed environment is malformed")
    expected_proposed = {
        name: REVIEW_ONLY_CANDIDATE_VALUES[name]
        for name in applicable
        if name in REVIEW_ONLY_CANDIDATE_VALUES
    }
    if proposed != expected_proposed:
        raise RuntimeInspectionError("review-only proposed policy values are not frozen")
    if unresolved != sorted(set(applicable) - set(expected_proposed)):
        raise RuntimeInspectionError("review-only unresolved-variable summary mismatch")
    if items != [[name, proposed[name]] for name in sorted(proposed)]:
        raise RuntimeInspectionError("review-only parent-launch policy tuple encoding mismatch")
    expected_hash = hashlib.sha256(
        b"operatormark-additional-thread-policy-candidate-v1\n"
        + _canonical_bytes(_policy_candidate_payload(record))
    ).hexdigest()
    if record["candidate_self_sha256"] != expected_hash:
        raise RuntimeInspectionError("review-only policy candidate self-commitment mismatch")


def build_review_only_thread_policy_candidate(
    *, runtime_record: Mapping[str, Any], base_import_closure: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind, but never approve, a candidate parent policy for human review."""

    from .base_imports import validate_base_import_closure

    validate_numerical_runtime_record(runtime_record)
    validate_base_import_closure(base_import_closure)
    if not runtime_record["all_active_backends_inspectable"]:
        raise RuntimeInspectionError("cannot propose a policy for an uninspectable active backend")
    if not runtime_record["all_effective_thread_counts_one"]:
        raise RuntimeInspectionError("cannot propose a policy for a non-single-thread runtime")
    applicable = list(runtime_record["applicable_additional_thread_variables"])
    threading_layers = sorted(
        {
            item["threading_layer"]
            for item in runtime_record["active_backends"]
            if isinstance(item["threading_layer"], str)
        }
    )
    proposed = {
        name: REVIEW_ONLY_CANDIDATE_VALUES[name]
        for name in applicable
        if name in REVIEW_ONLY_CANDIDATE_VALUES
    }
    payload = {
        "active_backend_kinds": list(runtime_record["active_backend_kinds"]),
        "active_threading_layers": threading_layers,
        "applicable_additional_thread_variables": applicable,
        "base_import_closure_sha256": base_import_closure["closure_sha256"],
        "parent_launch_spec_additional_environment": [
            [name, proposed[name]] for name in sorted(proposed)
        ],
        "proposed_environment": proposed,
        "runtime_record_sha256": runtime_record["runtime_record_sha256"],
        "schema": POLICY_CANDIDATE_SCHEMA,
        "status": POLICY_CANDIDATE_STATUS,
        "unresolved_applicable_thread_variables": sorted(set(applicable) - set(proposed)),
    }
    record = {
        **payload,
        "candidate_self_sha256": hashlib.sha256(
            b"operatormark-additional-thread-policy-candidate-v1\n" + _canonical_bytes(payload)
        ).hexdigest(),
    }
    validate_review_only_thread_policy_candidate(record)
    return record


def capture_runtime_record() -> dict[str, Any]:
    required = verify_child_preimport_environment()
    import numpy as np
    import scipy

    authorization_hash = os.environ.get("OPERATORMARK_AUTHORIZATION_FILE_SHA256", "")
    if len(authorization_hash) != 64:
        raise LaunchGuardError("authorization commitment missing before runtime inspection")
    numerical = inspect_loaded_numerical_runtime(
        authorization_file_sha256=authorization_hash,
        required_environment=required,
    )
    return {
        "architecture": platform.machine(),
        "environment_commitment": environment_commitment(required),
        "locale": locale.setlocale(locale.LC_ALL, None),
        "numpy_version": np.__version__,
        "operating_system": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "scipy_version": scipy.__version__,
        "thread_runtime": numerical,
    }


def main(argv: Sequence[str] | None = None) -> int:
    if argv not in (None, [], ()):
        raise LaunchGuardError("runtime inspector accepts no child arguments")
    print(json.dumps(capture_runtime_record(), ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
