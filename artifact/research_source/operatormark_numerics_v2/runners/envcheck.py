"""Fail-closed environment preflight.  Standard library only; must run before
NumPy, SciPy or any project module is imported (v1 §7.3 practice)."""

from __future__ import annotations

import os
import sys

REQUIRED = {
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
NUMERICAL_MODULES = ("numpy", "scipy", "operatormark", "operatormark_numerics_v2.api")


class EnvironmentRefused(SystemExit):
    pass


def check_environment(*, test_mode: bool) -> dict[str, str]:
    """Return the checked values or raise EnvironmentRefused (exit code 3).

    In test mode PYTHONHASHSEED must be set but may differ from 0 (test H5
    shows results do not depend on it); every other value must match.
    """

    loaded = [name for name in NUMERICAL_MODULES if name in sys.modules]
    problems = []
    if loaded:
        problems.append(f"numerical modules already imported: {loaded}")
    for name, expected in REQUIRED.items():
        actual = os.environ.get(name)
        if actual is None:
            problems.append(f"{name} missing")
        elif name == "PYTHONHASHSEED" and test_mode:
            if not actual.isdigit():
                problems.append("PYTHONHASHSEED must be a decimal integer")
        elif actual != expected:
            problems.append(f"{name}={actual!r}, required {expected!r}")
    if problems:
        sys.stderr.write("environment refused: " + "; ".join(problems) + "\n")
        raise EnvironmentRefused(3)
    return {name: os.environ[name] for name in REQUIRED}
