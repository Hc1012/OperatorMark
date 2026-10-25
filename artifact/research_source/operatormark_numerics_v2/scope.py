"""Determinism-scope record: what an L1 (bitwise reproducibility) claim is bound to.

The scope SHA-256 sits in every payload header.  Two payloads can only be
claimed equal at level L1 when their scopes are equal.  The repository's
§7.3 runtime inspector (operatormark/corpus/runtime.py) is not in the review
bundle, so it is reported as unavailable rather than reconstructed.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import platform
import sys
from pathlib import Path

THREAD_VARIABLES = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}


def _sha256_file(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _openblas_cores() -> list[dict]:
    """Best-effort OpenBLAS core names from the bundled NumPy/SciPy libraries."""

    import numpy
    import scipy

    found = []
    roots = []
    for module in (numpy, scipy):
        base = Path(module.__file__).resolve().parent
        roots += [base.parent / f"{base.name}.libs", base / ".libs", base.parent / f"{base.name}_libs"]
    candidates = []
    for root in roots:
        candidates += glob.glob(str(root / "*openblas*"))
    names = ("openblas_get_corename", "scipy_openblas_get_corename64_", "scipy_openblas_get_corename",
             "openblas_get_corename64_")
    configs = ("openblas_get_config", "scipy_openblas_get_config64_", "scipy_openblas_get_config", "openblas_get_config64_")
    for path in sorted(set(candidates)):
        entry = {"library": Path(path).name, "sha256": _sha256_file(path), "corename": "unavailable", "config": "unavailable"}
        try:
            lib = ctypes.CDLL(path)
            for name in names:
                fn = getattr(lib, name, None)
                if fn is not None:
                    fn.restype = ctypes.c_char_p
                    entry["corename"] = fn().decode("ascii", "replace")
                    break
            for name in configs:
                fn = getattr(lib, name, None)
                if fn is not None:
                    fn.restype = ctypes.c_char_p
                    entry["config"] = fn().decode("ascii", "replace")
                    break
        except OSError:
            pass
        found.append(entry)
    return found


def determinism_scope() -> dict:
    import numpy as np
    import scipy

    try:
        from numpy._core import _multiarray_umath as mu
        cpu = {"baseline": sorted(mu.__cpu_baseline__), "dispatch": sorted(mu.__cpu_dispatch__),
               "enabled": sorted(k for k, v in mu.__cpu_features__.items() if v)}
    except Exception:  # pragma: no cover
        cpu = None
    try:
        build = np.show_config(mode="dicts")["Build Dependencies"]
        numpy_blas = {k: build.get(k) for k in ("blas", "lapack")}
    except Exception:  # pragma: no cover
        numpy_blas = None
    try:
        sbuild = scipy.show_config(mode="dicts")["Build Dependencies"]
        scipy_blas = {k: sbuild.get(k) for k in ("blas", "lapack")}
    except Exception:  # pragma: no cover
        scipy_blas = None
    base_executable = getattr(sys, "_base_executable", sys.executable)
    runtime_libraries = {}
    for pattern in ("python3*.dll", "libpython3*.so*"):
        for candidate in sorted(glob.glob(str(Path(base_executable).resolve().parent / pattern))):
            runtime_libraries[Path(candidate).name] = _sha256_file(candidate)
    return {
        "schema": "operatormark_numerics_v2.determinism_scope.v1",
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "executable_sha256": _sha256_file(sys.executable),
        "base_executable_sha256": _sha256_file(base_executable),
        "python_runtime_libraries": runtime_libraries,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "numpy_cpu": cpu,
        "numpy_blas_lapack": numpy_blas,
        "scipy_blas_lapack": scipy_blas,
        "openblas_libraries": _openblas_cores(),
        # PYTHONHASHSEED is deliberately excluded: results must not depend on it (test H5);
        # it is checked at preflight and recorded in the run record.
        "thread_environment": {k: os.environ.get(k) for k in sorted(THREAD_VARIABLES) if k != "PYTHONHASHSEED"},
        "os": platform.system(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "repository_runtime_inspector": "unavailable: operatormark/corpus/runtime.py is not in the review bundle",
    }


def scope_sha256(scope: dict) -> str:
    from operatormark.corpus import canonical

    return canonical.sha256_bytes(canonical.canonical_json_bytes(scope))
