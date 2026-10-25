"""Guarded State 3 compute-feasibility coordinator and one-fixture worker.

State 2 tests this module only with injected hand-written fakes. Real solver
execution remains unreachable unless both the fresh-child bootstrap and a
verified State 3 capability are present. The coordinator launches exactly
fifteen workers; every worker is a new interpreter and measures one fixture.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from .canonical import (
    CanonicalRecordError,
    canonical_json_bytes,
    float64_hex,
    require_exact_keys,
    require_git_oid,
    require_sha256,
    strict_canonical_json_loads,
    validate_self_hash,
    with_self_hash,
)


FIXTURE_ORDER: Final = ("constant_half", "vertical_jump", "checkerboard")
PROTOCOL_SHA256: Final = "a38ef79b6c4bdccf53f599fa08f77e48a111afbf0b2783ad5733bda99996599a"
SOLVER_SPEC_BASELINE_COMMIT: Final = "e16456395cfe67b73b9d298fd0491cf9524d798a"
SOLVER_SPEC_BASELINE_TREE: Final = "81f40e24102e51e0da3ce65dafd769a7678565fd"
MEASURED_COUNT_PER_FIXTURE: Final = 5
CORPUS_SAMPLE_COUNT: Final = 4864
FROZEN_SAFETY_FACTOR: Final = 2.0
RAW_SEVEN_TENSOR_PAYLOAD_BYTES: Final = 420_872_192
ACCEPTED_TAU_L2: Final = 0.02543117962436324
ACCEPTED_TAU_J: Final = 0.022582030716092032
ALGEBRAIC_GATE: Final = 1.0e-10
CONSERVATION_GATE: Final = 1.0e-10
BOUNDARY_RECONSTRUCTION_GATE: Final = 100.0 * np.finfo(np.float64).eps

TIMING_FIELDS: Final = (
    "tpfa_assembly_ns", "tpfa_solve_ns", "tpfa_diagnostics_ns", "p1_assembly_ns",
    "p1_solve_ns", "p1_diagnostics_ns", "p1_projection_ns", "paired_solver_ns",
    "child_wall_ns",
)
MEMORY_FIELDS: Final = (
    "baseline_rss_bytes", "peak_rss_bytes", "incremental_peak_rss_bytes",
)
SUMMARY_FIELDS: Final = TIMING_FIELDS + MEMORY_FIELDS

REPORT_KEYS: Final = {
    "schema", "state", "protocol_file_sha256", "state3_authorization_file_sha256",
    "solver_spec_baseline_commit", "solver_spec_baseline_tree", "implementation_commit",
    "implementation_tree", "launcher_file_sha256", "runtime_inspector_file_sha256",
    "environment_record_sha256", "storage_preflight_before_file_sha256",
    "storage_preflight_after_file_sha256", "fixture_definitions", "warmup_count_per_fixture",
    "measured_count_per_fixture", "measurement_order", "timing_clock", "memory_counter",
    "rng_state_commitment_specification", "raw_measurements", "summary",
    "corpus_sample_count", "frozen_safety_factor", "extrapolation", "disk_estimate",
    "all_fixture_gates_passed", "all_rng_state_equalities_passed",
    "forbidden_rng_call_count", "forbidden_corpus_seed_access_count", "git_before",
    "git_after", "report_self_sha256",
}
RAW_MEASUREMENT_KEYS: Final = {
    "fixture", "repeat_index", "child_exit_code", "tpfa_assembly_ns", "tpfa_solve_ns",
    "tpfa_diagnostics_ns", "p1_assembly_ns", "p1_solve_ns", "p1_diagnostics_ns",
    "p1_projection_ns", "paired_solver_ns", "child_wall_ns", "rss_counter_name",
    "baseline_rss_bytes", "peak_rss_bytes", "incremental_peak_rss_bytes",
    "rng_state_commitments", "rng_state_equalities", "forbidden_rng_call_count",
    "forbidden_corpus_seed_access_count", "tpfa_diagnostics", "p1_diagnostics",
    "cross_solver_metrics", "gates_passed",
}
RNG_COMMITMENT_KEYS: Final = {
    "numpy_legacy_after_base_import_sha256", "python_random_after_base_import_sha256",
    "numpy_legacy_after_guard_install_sha256", "python_random_after_guard_install_sha256",
    "numpy_legacy_after_project_import_sha256", "python_random_after_project_import_sha256",
    "numpy_legacy_after_execution_sha256", "python_random_after_execution_sha256",
}
RNG_EQUALITY_KEYS: Final = {
    "numpy_legacy_after_guard_install_equals_baseline",
    "python_random_after_guard_install_equals_baseline",
    "numpy_legacy_after_project_import_equals_baseline",
    "python_random_after_project_import_equals_baseline",
    "numpy_legacy_after_execution_equals_baseline",
    "python_random_after_execution_equals_baseline",
}
TPFA_DIAGNOSTIC_KEYS: Final = {
    "algebraic_backward_error", "rhs_relative_l2_residual",
    "boundary_transmissibility_error", "maximum_local_conservation_ratio",
    "maximum_interior_face_antisymmetry", "global_balance_ratio", "passes",
}
P1_DIAGNOSTIC_KEYS: Final = {
    "algebraic_backward_error", "rhs_relative_l2_residual",
    "boundary_reconstruction_error", "outward_boundary_flux", "integrated_source",
    "reaction_denominator", "reaction_balance_ratio", "iterations", "passes",
}
CROSS_SOLVER_KEYS: Final = {
    "scales", "compliances", "delta_L2", "delta_J",
}
CROSS_SOLVER_SCALE_KEYS: Final = {"S_L2", "S_J"}
CROSS_SOLVER_COMPLIANCE_KEYS: Final = {"T64", "bar_P256"}
SUMMARY_KEYS: Final = {"fields", "global", "per_fixture"}
SUMMARY_STATISTIC_KEYS: Final = {"median", "maximum"}
EXTRAPOLATION_KEYS: Final = {
    "median_pair_seconds", "max_pair_seconds", "median_extrapolated_seconds",
    "max_extrapolated_seconds", "frozen_safety_factor",
    "safety_factored_compute_budget_seconds", "safety_factored_peak_memory_budget_bytes",
    "median_extrapolated_hours", "max_extrapolated_hours",
    "safety_factored_compute_budget_hours",
}
DISK_ESTIMATE_KEYS: Final = {
    "raw_seven_tensor_payload_bytes", "projected_npy_header_bytes",
    "projected_jsonl_bytes", "projected_manifest_and_audit_bytes",
    "controls_only_bytes", "estimated_accepted_bytes", "active_staging_bytes",
    "readback_atomic_publication_bytes", "preserved_failed_attempt_allowance_bytes",
    "projected_audit_bytes", "fixed_headroom_bytes", "minimum_floor_bytes",
    "four_copy_formula_bytes", "total_required_free_space_bytes",
}
GIT_RECORD_KEYS: Final = {
    "branch", "head", "tree", "parents", "status_short_untracked_all",
    "tracked_diff_sha256", "tracked_diff_size_bytes", "staged_diff_sha256",
    "staged_diff_size_bytes", "tracked_changes_empty", "staged_changes_empty",
    "untracked_inclusive_status_empty",
}
EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()
BINDINGS_KEYS: Final = {
    "schema", "protocol_file_sha256", "state3_authorization_file_sha256",
    "solver_spec_baseline_commit", "solver_spec_baseline_tree", "implementation_commit",
    "implementation_tree", "launcher_file_sha256", "runtime_inspector_file_sha256",
    "environment_record_sha256", "storage_preflight_before_file_sha256",
    "storage_preflight_after_file_sha256", "disk_estimate", "git_before", "git_after",
    "bindings_self_sha256",
}
PREFLIGHT_BINDINGS_KEYS: Final = BINDINGS_KEYS - {
    "storage_preflight_after_file_sha256", "git_after", "bindings_self_sha256",
} | {"preflight_bindings_self_sha256"}
MEASUREMENT_BUNDLE_KEYS: Final = {
    "schema", "state", "protocol_file_sha256", "state3_authorization_file_sha256",
    "implementation_commit", "implementation_tree", "warmup_count_per_fixture",
    "measured_count_per_fixture", "measurement_order", "raw_measurements",
    "all_fixture_gates_passed", "all_rng_state_equalities_passed",
    "forbidden_rng_call_count", "forbidden_corpus_seed_access_count",
    "measurement_bundle_self_sha256",
}

FIXTURE_DEFINITIONS: Final = {
    "constant_half": {
        "construction": "deterministic_indexed_assignment_no_rng", "dtype": "<f8",
        "formula": "rho[j,i]=0.5", "shape": [8, 8],
    },
    "vertical_jump": {
        "construction": "deterministic_indexed_assignment_no_rng", "dtype": "<f8",
        "formula": "rho[j,i]=0.0 if i<4 else 1.0", "shape": [8, 8],
    },
    "checkerboard": {
        "construction": "deterministic_indexed_assignment_no_rng", "dtype": "<f8",
        "formula": "rho[j,i]=float((i+j) mod 2)", "shape": [8, 8],
    },
}
RNG_STATE_COMMITMENT_SPECIFICATION: Final = {
    "numpy_legacy_domain_separator": "operatormark-numpy-legacy-rng-state-v1\n",
    "numpy_legacy_canonical_fields": [
        "bit_generator", "key_dtype", "key_shape", "key_c_order_hex", "position",
        "has_gauss", "cached_gaussian_hex",
    ],
    "python_random_domain_separator": "operatormark-python-random-state-v1\n",
    "python_random_canonical_fields": ["version", "internal_state", "gauss_next"],
    "canonical_json_encoding": "utf8_compact_sorted_keys_no_trailing_newline",
    "ordered_checkpoints": [
        "after_base_import", "after_guard_install", "after_project_import", "after_execution",
    ],
    "passive_module_import_increments_forbidden_call_count": False,
}


class MicrobenchmarkAuthorizationError(RuntimeError):
    """Raised before fixture or solver work when State 3 is not authorized."""


@dataclass(frozen=True, slots=True)
class VerifiedState3Capability:
    authorization_file_sha256: str
    implementation_commit: str
    implementation_tree: str
    state: str = "state3"


@dataclass(frozen=True, slots=True)
class ComputeFeasibilityBindings:
    protocol_file_sha256: str
    state3_authorization_file_sha256: str
    solver_spec_baseline_commit: str
    solver_spec_baseline_tree: str
    implementation_commit: str
    implementation_tree: str
    launcher_file_sha256: str
    runtime_inspector_file_sha256: str
    environment_record_sha256: str
    storage_preflight_before_file_sha256: str
    storage_preflight_after_file_sha256: str
    disk_estimate: Mapping[str, Any]
    git_before: Mapping[str, Any]
    git_after: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ComputeFeasibilityPreflightBindings:
    protocol_file_sha256: str
    state3_authorization_file_sha256: str
    solver_spec_baseline_commit: str
    solver_spec_baseline_tree: str
    implementation_commit: str
    implementation_tree: str
    launcher_file_sha256: str
    runtime_inspector_file_sha256: str
    environment_record_sha256: str
    storage_preflight_before_file_sha256: str
    disk_estimate: Mapping[str, Any]
    git_before: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FixtureOperations:
    coefficient_on_tpfa_grid: Callable[[np.ndarray, int], Any]
    assemble_tpfa: Callable[[Any], Any]
    solve_tpfa: Callable[[Any], Any]
    tpfa_diagnostics: Callable[[Any, Any], Any]
    assemble_p1: Callable[[np.ndarray, int], Any]
    solve_p1: Callable[[Any], Any]
    p1_diagnostics: Callable[[Any, Any], Any]
    project_p1: Callable[[Any], np.ndarray]


@dataclass(frozen=True, slots=True)
class WorkerAuditContext:
    child_wall_start_ns: int
    baseline_rng_commitments: Mapping[str, str]
    after_guard_rng_commitments: Mapping[str, str]
    after_project_rng_commitments: Mapping[str, str]
    rng_checkpoint: Callable[[], Mapping[str, str]]
    expected_base_import_closure: Mapping[str, Any]
    reinspect_base_import_closure: Callable[[], Mapping[str, Any]]
    expected_runtime: Mapping[str, Any]
    reinspect_runtime: Callable[[], Mapping[str, Any]]
    forbidden_rng_call_count: Callable[[], int]
    forbidden_corpus_seed_access_count: Callable[[], int]
    memory_probe: Callable[[], tuple[str, int, int]]
    clock_ns: Callable[[], int] = time.perf_counter_ns


def require_state3_capability(
    capability: object, *, expected_action: str = "state3-microbenchmark"
) -> VerifiedState3Capability:
    if not isinstance(capability, VerifiedState3Capability) or capability.state != "state3":
        raise MicrobenchmarkAuthorizationError(
            "compute-feasibility execution requires separate State 3 authorization"
        )
    require_sha256(
        capability.authorization_file_sha256, name="authorization_file_sha256"
    )
    require_git_oid(capability.implementation_commit, name="implementation_commit")
    require_git_oid(capability.implementation_tree, name="implementation_tree")
    context = sys.modules.get("_operatormark_fresh_child_bootstrap")
    if getattr(context, "sentinel", None) != "operatormark-fresh-child-bootstrap-v2":
        raise MicrobenchmarkAuthorizationError(
            "compute-feasibility execution requires the guarded fresh-child bootstrap"
        )
    context_bindings = {
        "action": expected_action,
        "state": "state3",
        "authorization_file_sha256": capability.authorization_file_sha256,
        "implementation_commit": capability.implementation_commit,
        "implementation_tree": capability.implementation_tree,
    }
    if any(getattr(context, name, None) != value for name, value in context_bindings.items()):
        raise MicrobenchmarkAuthorizationError(
            "guarded bootstrap identity differs from the requested State 3 role"
        )
    if os.environ.get("OPERATORMARK_STATE") != "state3" or os.environ.get(
        "OPERATORMARK_ACTION"
    ) != expected_action:
        raise MicrobenchmarkAuthorizationError("fresh child action is not the required State 3 role")
    if os.environ.get("OPERATORMARK_AUTHORIZATION_FILE_SHA256") != capability.authorization_file_sha256:
        raise MicrobenchmarkAuthorizationError("State 3 authorization commitment mismatch")
    if os.environ.get("OPERATORMARK_IMPLEMENTATION_COMMIT") != capability.implementation_commit:
        raise MicrobenchmarkAuthorizationError("State 3 implementation commit mismatch")
    if os.environ.get("OPERATORMARK_IMPLEMENTATION_TREE") != capability.implementation_tree:
        raise MicrobenchmarkAuthorizationError("State 3 implementation tree mismatch")
    if expected_action == "state3-microbenchmark-worker" and os.environ.get(
        "OPERATORMARK_INTERNAL_WORKER"
    ) != "1":
        raise MicrobenchmarkAuthorizationError("State 3 worker is not coordinator-bound")
    return capability


def fixed_fixture(name: str) -> np.ndarray:
    """Construct one protocol fixture by indexed assignment without RNG."""
    if name not in FIXTURE_ORDER:
        raise ValueError("fixture name is not frozen")
    control = np.empty((8, 8), dtype=np.dtype("<f8"), order="C")
    for j in range(8):
        for i in range(8):
            if name == "constant_half":
                value = 0.5
            elif name == "vertical_jump":
                value = 0.0 if i < 4 else 1.0
            else:
                value = float((i + j) % 2)
            control[j, i] = value
    validate_fixture(name, control)
    return control


def validate_fixture(name: str, control: np.ndarray) -> None:
    if name not in FIXTURE_ORDER:
        raise ValueError("unknown fixture")
    if (
        not isinstance(control, np.ndarray) or control.shape != (8, 8)
        or control.dtype != np.dtype("<f8") or not control.flags.c_contiguous
    ):
        raise ValueError("fixture must be C-contiguous <f8 [8,8]")
    for j in range(8):
        for i in range(8):
            expected = (
                0.5 if name == "constant_half"
                else (0.0 if i < 4 else 1.0) if name == "vertical_jump"
                else float((i + j) % 2)
            )
            if control[j, i] != expected:
                raise ValueError("fixture value mismatch")


def windows_process_memory() -> tuple[str, int, int]:
    """Return exact counter name, current RSS, and absolute peak RSS bytes."""
    if os.name != "nt":
        import resource
        import sys
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak = value if sys.platform == "darwin" else value * 1024
        return "ru_maxrss_bytes", peak, peak
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), ctypes.sizeof(counters)
    ):
        raise RuntimeError("GetProcessMemoryInfo failed")
    return (
        "windows_process_peak_working_set_bytes", int(counters.WorkingSetSize),
        int(counters.PeakWorkingSetSize),
    )


def _timed(function: Callable[[], Any], clock_ns: Callable[[], int]) -> tuple[Any, int]:
    start = clock_ns()
    value = function()
    elapsed = clock_ns() - start
    if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
        raise RuntimeError("perf_counter_ns produced an invalid duration")
    return value, elapsed


def _real_fixture_operations() -> FixtureOperations:
    from operatormark.benchmarks.darcy_spec import coefficient_on_tpfa_grid
    from operatormark.solvers.darcy_p1 import (
        assemble_p1_zero_dirichlet_benchmark, compute_p1_diagnostics,
        project_p1_to_64, solve_p1_with_info,
    )
    from operatormark.solvers.darcy_tpfa import (
        assemble_zero_dirichlet_benchmark, compute_tpfa_diagnostics, solve_tpfa,
    )
    return FixtureOperations(
        coefficient_on_tpfa_grid, assemble_zero_dirichlet_benchmark, solve_tpfa,
        compute_tpfa_diagnostics, assemble_p1_zero_dirichlet_benchmark,
        solve_p1_with_info, compute_p1_diagnostics, project_p1_to_64,
    )


def _hex(value: object, name: str) -> str:
    return float64_hex(value, name=name)


def _encode_tpfa_diagnostics(value: Any) -> dict[str, Any]:
    if value.passes is not True or value.rhs_relative_l2_residual is None:
        raise CanonicalRecordError("TPFA diagnostic gate or RHS residual failed")
    return {
        "algebraic_backward_error": _hex(value.algebraic_backward_error, "TPFA algebraic error"),
        "rhs_relative_l2_residual": _hex(value.rhs_relative_l2_residual, "TPFA RHS residual"),
        "boundary_transmissibility_error": _hex(value.boundary_transmissibility_error, "TPFA boundary error"),
        "maximum_local_conservation_ratio": _hex(value.maximum_local_conservation_ratio, "TPFA local conservation"),
        "maximum_interior_face_antisymmetry": _hex(value.maximum_interior_face_antisymmetry, "TPFA antisymmetry"),
        "global_balance_ratio": _hex(value.global_balance_ratio, "TPFA global balance"),
        "passes": True,
    }


def _encode_p1_diagnostics(value: Any, iterations: int) -> dict[str, Any]:
    if value.passes is not True or value.rhs_relative_l2_residual is None:
        raise CanonicalRecordError("P1 diagnostic gate or RHS residual failed")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise CanonicalRecordError("P1 iteration count must be a positive integer")
    return {
        "algebraic_backward_error": _hex(value.algebraic_backward_error, "P1 algebraic error"),
        "rhs_relative_l2_residual": _hex(value.rhs_relative_l2_residual, "P1 RHS residual"),
        "boundary_reconstruction_error": _hex(value.boundary_reconstruction_error, "P1 boundary reconstruction"),
        "outward_boundary_flux": _hex(value.outward_boundary_flux, "P1 boundary flux"),
        "integrated_source": _hex(value.integrated_source, "P1 integrated source"),
        "reaction_denominator": _hex(value.reaction_denominator, "P1 reaction denominator"),
        "reaction_balance_ratio": _hex(value.reaction_balance_ratio, "P1 reaction balance"),
        "iterations": iterations,
        "passes": True,
    }


def _weighted_l2_64(field: np.ndarray) -> float:
    values = np.asarray(field)
    if values.shape != (64, 64) or values.dtype != np.dtype("<f8") or not values.flags.c_contiguous:
        raise CanonicalRecordError("cross-solver field must be C-contiguous <f8 [64,64]")
    with np.errstate(over="ignore", invalid="ignore"):
        squared = float(np.sum(values * values, dtype=np.float64) / np.float64(64**2))
    if not math.isfinite(squared) or squared < 0.0:
        raise CanonicalRecordError("cross-solver L2 integral is invalid")
    return math.sqrt(squared)


def _finite_ratio(numerator: float, denominator: float, name: str) -> float:
    result = numerator / denominator if denominator > 0.0 else 0.0 if numerator == 0.0 else math.inf
    if not math.isfinite(result):
        raise CanonicalRecordError(f"{name} is non-finite")
    return result


def _cross_solver_metrics(tpfa: np.ndarray, projected_p1: np.ndarray) -> dict[str, Any]:
    t64 = np.ascontiguousarray(tpfa, dtype=np.dtype("<f8"))
    p256 = np.ascontiguousarray(projected_p1, dtype=np.dtype("<f8"))
    if t64.shape != (64, 64) or p256.shape != (64, 64):
        raise CanonicalRecordError("cross-solver output shape mismatch")
    scale_l2 = _weighted_l2_64(p256)
    with np.errstate(over="ignore", invalid="ignore"):
        compliance_t = float(np.sum(t64, dtype=np.float64) / np.float64(64**2))
        compliance_p = float(np.sum(p256, dtype=np.float64) / np.float64(64**2))
    if not math.isfinite(compliance_t) or not math.isfinite(compliance_p):
        raise CanonicalRecordError("cross-solver compliance is non-finite")
    scale_j = abs(compliance_p)
    with np.errstate(over="ignore", invalid="ignore"):
        discrepancy = t64 - p256
    delta_l2 = _finite_ratio(_weighted_l2_64(discrepancy), scale_l2, "delta_L2")
    delta_j = _finite_ratio(abs(compliance_t - compliance_p), scale_j, "delta_J")
    return {
        "scales": {"S_L2": _hex(scale_l2, "S_L2"), "S_J": _hex(scale_j, "S_J")},
        "compliances": {
            "T64": _hex(compliance_t, "T64 compliance"),
            "bar_P256": _hex(compliance_p, "bar_P256 compliance"),
        },
        "delta_L2": _hex(delta_l2, "delta_L2"),
        "delta_J": _hex(delta_j, "delta_J"),
    }


def _checkpoint_fields(
    baseline: Mapping[str, str], after_guard: Mapping[str, str],
    after_project: Mapping[str, str], after_execution: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, bool]]:
    checkpoints = (
        ("baseline", baseline), ("after_guard", after_guard),
        ("after_project", after_project), ("after_execution", after_execution),
    )
    for checkpoint_name, checkpoint in checkpoints:
        require_exact_keys(checkpoint, {"numpy_legacy_sha256", "python_random_sha256"}, name=checkpoint_name)
        for name, value in checkpoint.items():
            require_sha256(value, name=f"{checkpoint_name}.{name}")
    commitments = {
        "numpy_legacy_after_base_import_sha256": baseline["numpy_legacy_sha256"],
        "python_random_after_base_import_sha256": baseline["python_random_sha256"],
        "numpy_legacy_after_guard_install_sha256": after_guard["numpy_legacy_sha256"],
        "python_random_after_guard_install_sha256": after_guard["python_random_sha256"],
        "numpy_legacy_after_project_import_sha256": after_project["numpy_legacy_sha256"],
        "python_random_after_project_import_sha256": after_project["python_random_sha256"],
        "numpy_legacy_after_execution_sha256": after_execution["numpy_legacy_sha256"],
        "python_random_after_execution_sha256": after_execution["python_random_sha256"],
    }
    equalities = {
        "numpy_legacy_after_guard_install_equals_baseline": after_guard["numpy_legacy_sha256"] == baseline["numpy_legacy_sha256"],
        "python_random_after_guard_install_equals_baseline": after_guard["python_random_sha256"] == baseline["python_random_sha256"],
        "numpy_legacy_after_project_import_equals_baseline": after_project["numpy_legacy_sha256"] == baseline["numpy_legacy_sha256"],
        "python_random_after_project_import_equals_baseline": after_project["python_random_sha256"] == baseline["python_random_sha256"],
        "numpy_legacy_after_execution_equals_baseline": after_execution["numpy_legacy_sha256"] == baseline["numpy_legacy_sha256"],
        "python_random_after_execution_equals_baseline": after_execution["python_random_sha256"] == baseline["python_random_sha256"],
    }
    if not all(equalities.values()):
        raise CanonicalRecordError("RNG commitment equality failed")
    return commitments, equalities


def measure_one_fixed_fixture(
    name: str, repeat_index: int, *, capability: object | None = None,
    audit_context: WorkerAuditContext | None = None,
    operations: FixtureOperations | None = None,
) -> dict[str, Any]:
    """Measure exactly one fixture; one invocation is one fresh State 3 worker."""
    require_state3_capability(capability, expected_action="state3-microbenchmark-worker")
    if audit_context is None:
        raise MicrobenchmarkAuthorizationError("State 3 worker audit context is required")
    if name not in FIXTURE_ORDER or isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index not in range(MEASURED_COUNT_PER_FIXTURE):
        raise MicrobenchmarkAuthorizationError("worker fixture or repeat is outside frozen coverage")
    if audit_context.after_guard_rng_commitments != audit_context.baseline_rng_commitments:
        raise CanonicalRecordError("after-guard RNG commitment differs from baseline")
    if audit_context.after_project_rng_commitments != audit_context.baseline_rng_commitments:
        raise CanonicalRecordError("after-project RNG commitment differs from baseline")

    control = fixed_fixture(name)
    ops = _real_fixture_operations() if operations is None else operations
    counter_name, baseline_rss, _ = audit_context.memory_probe()
    kappa = ops.coefficient_on_tpfa_grid(control, 64)
    tpfa_system, tpfa_assembly_ns = _timed(lambda: ops.assemble_tpfa(kappa), audit_context.clock_ns)
    tpfa_solution, tpfa_solve_ns = _timed(lambda: ops.solve_tpfa(tpfa_system), audit_context.clock_ns)
    tpfa_diagnostics, tpfa_diagnostics_ns = _timed(lambda: ops.tpfa_diagnostics(tpfa_system, tpfa_solution), audit_context.clock_ns)
    p1_system, p1_assembly_ns = _timed(lambda: ops.assemble_p1(control, 256), audit_context.clock_ns)
    p1_result, p1_solve_ns = _timed(lambda: ops.solve_p1(p1_system), audit_context.clock_ns)
    p1_diagnostics, p1_diagnostics_ns = _timed(lambda: ops.p1_diagnostics(p1_system, p1_result.solution), audit_context.clock_ns)
    projection, p1_projection_ns = _timed(lambda: ops.project_p1(p1_result.solution), audit_context.clock_ns)

    # Complete every scientific calculation and canonical scientific encoding
    # before the post-execution closure/runtime/RNG checkpoints.
    paired_solver_ns = sum((
        tpfa_assembly_ns, tpfa_solve_ns, tpfa_diagnostics_ns, p1_assembly_ns,
        p1_solve_ns, p1_diagnostics_ns, p1_projection_ns,
    ))
    tpfa_record = _encode_tpfa_diagnostics(tpfa_diagnostics)
    p1_record = _encode_p1_diagnostics(p1_diagnostics, p1_result.iterations)
    cross_record = _cross_solver_metrics(tpfa_solution, projection)

    closure_after = audit_context.reinspect_base_import_closure()
    if canonical_json_bytes(closure_after) != canonical_json_bytes(audit_context.expected_base_import_closure):
        raise CanonicalRecordError("numerical base-import closure changed during worker execution")
    runtime_after = audit_context.reinspect_runtime()
    if canonical_json_bytes(runtime_after) != canonical_json_bytes(audit_context.expected_runtime):
        raise CanonicalRecordError("numerical runtime changed during worker execution")
    after_execution = audit_context.rng_checkpoint()
    commitments, equalities = _checkpoint_fields(
        audit_context.baseline_rng_commitments, audit_context.after_guard_rng_commitments,
        audit_context.after_project_rng_commitments, after_execution,
    )
    forbidden_rng = audit_context.forbidden_rng_call_count()
    forbidden_seed = audit_context.forbidden_corpus_seed_access_count()
    if forbidden_rng != 0 or forbidden_seed != 0:
        raise CanonicalRecordError("worker recorded a prohibited RNG call or corpus-seed access")
    # Sample the peak only after diagnostics, projection, closure/runtime
    # reinspection, RNG checkpointing, and canonical scientific encoding.
    second_counter_name, _, peak_rss = audit_context.memory_probe()
    if second_counter_name != counter_name:
        raise CanonicalRecordError("process-memory counter changed during worker")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (baseline_rss, peak_rss)):
        raise CanonicalRecordError("process-memory readings must be non-negative integers")
    provisional_wall = audit_context.clock_ns() - audit_context.child_wall_start_ns
    record = {
        "fixture": name, "repeat_index": repeat_index, "child_exit_code": 0,
        "tpfa_assembly_ns": tpfa_assembly_ns, "tpfa_solve_ns": tpfa_solve_ns,
        "tpfa_diagnostics_ns": tpfa_diagnostics_ns, "p1_assembly_ns": p1_assembly_ns,
        "p1_solve_ns": p1_solve_ns, "p1_diagnostics_ns": p1_diagnostics_ns,
        "p1_projection_ns": p1_projection_ns, "paired_solver_ns": paired_solver_ns,
        "child_wall_ns": provisional_wall, "rss_counter_name": counter_name,
        "baseline_rss_bytes": baseline_rss, "peak_rss_bytes": peak_rss,
        "incremental_peak_rss_bytes": max(0, peak_rss - baseline_rss),
        "rng_state_commitments": commitments, "rng_state_equalities": equalities,
        "forbidden_rng_call_count": forbidden_rng,
        "forbidden_corpus_seed_access_count": forbidden_seed,
        "tpfa_diagnostics": tpfa_record, "p1_diagnostics": p1_record,
        "cross_solver_metrics": cross_record,
        "gates_passed": bool(
            tpfa_record["passes"] and p1_record["passes"]
            and float.fromhex(cross_record["delta_L2"]) <= ACCEPTED_TAU_L2
            and float.fromhex(cross_record["delta_J"]) <= ACCEPTED_TAU_J
        ),
    }
    validate_raw_measurement(record)
    final_wall = audit_context.clock_ns() - audit_context.child_wall_start_ns
    if final_wall < provisional_wall:
        raise CanonicalRecordError("child wall clock moved backwards")
    record["child_wall_ns"] = final_wall
    validate_raw_measurement(record)
    return record


def summarize_nonnegative_integers(values: Sequence[int]) -> dict[str, int | float]:
    if not values or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("summary requires non-empty non-negative integer values")
    return {"maximum": max(values), "median": statistics.median(values)}


def _canonical_hex(value: object, *, name: str, nonnegative: bool) -> float:
    if not isinstance(value, str):
        raise CanonicalRecordError(f"{name} must be canonical float.hex text")
    try:
        scalar = float.fromhex(value)
    except (ValueError, OverflowError) as exc:
        raise CanonicalRecordError(f"{name} is not float.hex text") from exc
    if not math.isfinite(scalar) or value != scalar.hex().lower():
        raise CanonicalRecordError(f"{name} is not canonical finite float.hex text")
    if nonnegative and scalar < 0.0:
        raise CanonicalRecordError(f"{name} must be non-negative")
    return scalar


def _validate_diagnostics(record: object, *, solver: str) -> None:
    if not isinstance(record, Mapping):
        raise CanonicalRecordError(f"{solver} diagnostics must be an object")
    expected = TPFA_DIAGNOSTIC_KEYS if solver == "tpfa" else P1_DIAGNOSTIC_KEYS
    require_exact_keys(record, expected, name=f"{solver} diagnostics")
    if record["passes"] is not True:
        raise CanonicalRecordError(f"{solver} diagnostics did not pass")
    signed = {"outward_boundary_flux", "integrated_source"}
    decoded: dict[str, float] = {}
    for key in expected - {"passes", "iterations"}:
        decoded[key] = _canonical_hex(
            record[key], name=f"{solver}.{key}", nonnegative=key not in signed
        )
    if solver == "p1" and (
        isinstance(record["iterations"], bool) or not isinstance(record["iterations"], int)
        or record["iterations"] <= 0
    ):
        raise CanonicalRecordError("P1 iterations must be a positive integer")
    gates = (
        {
            "algebraic_backward_error": ALGEBRAIC_GATE,
            "boundary_transmissibility_error": BOUNDARY_RECONSTRUCTION_GATE,
            "maximum_local_conservation_ratio": CONSERVATION_GATE,
            "maximum_interior_face_antisymmetry": CONSERVATION_GATE,
            "global_balance_ratio": CONSERVATION_GATE,
        }
        if solver == "tpfa"
        else {
            "algebraic_backward_error": ALGEBRAIC_GATE,
            "boundary_reconstruction_error": BOUNDARY_RECONSTRUCTION_GATE,
            "reaction_balance_ratio": CONSERVATION_GATE,
        }
    )
    if any(decoded[name] > gate for name, gate in gates.items()):
        raise CanonicalRecordError(f"{solver} diagnostic exceeds its frozen gate")


def validate_raw_measurement(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, RAW_MEASUREMENT_KEYS, name="raw measurement")
    if record["fixture"] not in FIXTURE_ORDER:
        raise CanonicalRecordError("raw measurement fixture is not frozen")
    if (
        isinstance(record["repeat_index"], bool) or not isinstance(record["repeat_index"], int)
        or record["repeat_index"] not in range(MEASURED_COUNT_PER_FIXTURE)
        or isinstance(record["child_exit_code"], bool)
        or not isinstance(record["child_exit_code"], int)
        or record["child_exit_code"] != 0
    ):
        raise CanonicalRecordError("raw measurement repeat or exit code is invalid")
    for name in SUMMARY_FIELDS:
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CanonicalRecordError(f"{name} must be a non-negative integer")
    if record["paired_solver_ns"] != sum(record[name] for name in TIMING_FIELDS[:7]):
        raise CanonicalRecordError("paired solver time is not the exact seven-region sum")
    if record["child_wall_ns"] < record["paired_solver_ns"]:
        raise CanonicalRecordError("child wall time is shorter than paired solver time")
    if record["incremental_peak_rss_bytes"] != max(0, record["peak_rss_bytes"] - record["baseline_rss_bytes"]):
        raise CanonicalRecordError("incremental peak RSS formula mismatch")
    if not isinstance(record["rss_counter_name"], str) or not record["rss_counter_name"]:
        raise CanonicalRecordError("RSS counter name is missing")
    commitments = record["rng_state_commitments"]
    equalities = record["rng_state_equalities"]
    if not isinstance(commitments, Mapping) or not isinstance(equalities, Mapping):
        raise CanonicalRecordError("RNG checkpoint records must be objects")
    require_exact_keys(commitments, RNG_COMMITMENT_KEYS, name="RNG commitments")
    require_exact_keys(equalities, RNG_EQUALITY_KEYS, name="RNG equalities")
    for key, value in commitments.items():
        require_sha256(value, name=f"rng_state_commitments.{key}")
    if any(value is not True for value in equalities.values()):
        raise CanonicalRecordError("an RNG checkpoint differs from its child baseline")
    if any(
        isinstance(record[name], bool) or not isinstance(record[name], int) or record[name] != 0
        for name in ("forbidden_rng_call_count", "forbidden_corpus_seed_access_count")
    ):
        raise CanonicalRecordError("a forbidden count is nonzero")
    _validate_diagnostics(record["tpfa_diagnostics"], solver="tpfa")
    _validate_diagnostics(record["p1_diagnostics"], solver="p1")
    cross = record["cross_solver_metrics"]
    if not isinstance(cross, Mapping):
        raise CanonicalRecordError("cross-solver metrics must be an object")
    require_exact_keys(cross, CROSS_SOLVER_KEYS, name="cross-solver metrics")
    scales = cross["scales"]
    compliances = cross["compliances"]
    if not isinstance(scales, Mapping) or not isinstance(compliances, Mapping):
        raise CanonicalRecordError("cross-solver scales and compliances must be objects")
    require_exact_keys(scales, CROSS_SOLVER_SCALE_KEYS, name="cross-solver scales")
    require_exact_keys(
        compliances, CROSS_SOLVER_COMPLIANCE_KEYS, name="cross-solver compliances"
    )
    decoded_scales = {
        key: _canonical_hex(
            value, name=f"cross_solver_metrics.scales.{key}", nonnegative=True
        )
        for key, value in scales.items()
    }
    decoded_compliances = {
        key: _canonical_hex(
            value, name=f"cross_solver_metrics.compliances.{key}", nonnegative=False
        )
        for key, value in compliances.items()
    }
    delta_l2 = _canonical_hex(cross["delta_L2"], name="delta_L2", nonnegative=True)
    delta_j = _canonical_hex(cross["delta_J"], name="delta_J", nonnegative=True)
    if decoded_scales["S_J"] != abs(decoded_compliances["bar_P256"]):
        raise CanonicalRecordError("S_J is not the absolute bar_P256 compliance")
    expected_delta_j = _finite_ratio(
        abs(decoded_compliances["T64"] - decoded_compliances["bar_P256"]),
        decoded_scales["S_J"],
        "delta_J",
    )
    if delta_j != expected_delta_j:
        raise CanonicalRecordError("delta_J does not match recorded compliances and scale")
    expected_gate = (
        record["tpfa_diagnostics"]["passes"]
        and record["p1_diagnostics"]["passes"]
        and delta_l2 <= ACCEPTED_TAU_L2
        and delta_j <= ACCEPTED_TAU_J
    )
    if record["gates_passed"] is not expected_gate or record["gates_passed"] is not True:
        raise CanonicalRecordError("raw measurement gate aggregation mismatch")


def _build_summary(raw: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            field: summarize_nonnegative_integers([int(record[field]) for record in records])
            for field in SUMMARY_FIELDS
        }
    return {
        "fields": list(SUMMARY_FIELDS), "global": group(raw),
        "per_fixture": {
            fixture: group([record for record in raw if record["fixture"] == fixture])
            for fixture in FIXTURE_ORDER
        },
    }


def _build_extrapolation(raw: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    pair_values = [int(record["paired_solver_ns"]) for record in raw]
    peak_values = [int(record["peak_rss_bytes"]) for record in raw]
    median_pair_seconds = float(statistics.median(pair_values)) / 1e9
    max_pair_seconds = float(max(pair_values)) / 1e9
    median_extrapolated_seconds = CORPUS_SAMPLE_COUNT * median_pair_seconds
    max_extrapolated_seconds = CORPUS_SAMPLE_COUNT * max_pair_seconds
    budget_seconds = FROZEN_SAFETY_FACTOR * max_extrapolated_seconds
    return {
        "median_pair_seconds": median_pair_seconds, "max_pair_seconds": max_pair_seconds,
        "median_extrapolated_seconds": median_extrapolated_seconds,
        "max_extrapolated_seconds": max_extrapolated_seconds,
        "frozen_safety_factor": FROZEN_SAFETY_FACTOR,
        "safety_factored_compute_budget_seconds": budget_seconds,
        "safety_factored_peak_memory_budget_bytes": 2 * max(peak_values),
        "median_extrapolated_hours": median_extrapolated_seconds / 3600.0,
        "max_extrapolated_hours": max_extrapolated_seconds / 3600.0,
        "safety_factored_compute_budget_hours": budget_seconds / 3600.0,
    }


def build_writer_disk_estimate(
    *, projected_npy_header_bytes: int, projected_jsonl_bytes: int,
    projected_manifest_and_audit_bytes: int, controls_only_bytes: int,
    projected_audit_bytes: int,
) -> dict[str, int]:
    """Build the frozen writer-derived storage formula without scientific data."""
    inputs = (
        projected_npy_header_bytes, projected_jsonl_bytes,
        projected_manifest_and_audit_bytes, controls_only_bytes, projected_audit_bytes,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in inputs):
        raise CanonicalRecordError("writer disk-estimate inputs must be non-negative integers")
    accepted = (
        RAW_SEVEN_TENSOR_PAYLOAD_BYTES + projected_npy_header_bytes
        + projected_jsonl_bytes + projected_manifest_and_audit_bytes
    )
    if controls_only_bytes > accepted:
        raise CanonicalRecordError("controls-only writer estimate exceeds accepted corpus estimate")
    formula = 4 * accepted + projected_audit_bytes + 2**30
    record = {
        "raw_seven_tensor_payload_bytes": RAW_SEVEN_TENSOR_PAYLOAD_BYTES,
        "projected_npy_header_bytes": projected_npy_header_bytes,
        "projected_jsonl_bytes": projected_jsonl_bytes,
        "projected_manifest_and_audit_bytes": projected_manifest_and_audit_bytes,
        "controls_only_bytes": controls_only_bytes,
        "estimated_accepted_bytes": accepted,
        "active_staging_bytes": accepted,
        "readback_atomic_publication_bytes": accepted,
        "preserved_failed_attempt_allowance_bytes": accepted,
        "projected_audit_bytes": projected_audit_bytes,
        "fixed_headroom_bytes": 2**30,
        "minimum_floor_bytes": 5 * 2**30,
        "four_copy_formula_bytes": formula,
        "total_required_free_space_bytes": max(5 * 2**30, formula),
    }
    validate_disk_estimate(record)
    return record


def validate_disk_estimate(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, DISK_ESTIMATE_KEYS, name="disk estimate")
    for name, value in record.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CanonicalRecordError(f"disk_estimate.{name} must be a non-negative integer")
    if record["raw_seven_tensor_payload_bytes"] != RAW_SEVEN_TENSOR_PAYLOAD_BYTES:
        raise CanonicalRecordError("raw seven-tensor payload estimate changed")
    expected_accepted = (
        record["raw_seven_tensor_payload_bytes"] + record["projected_npy_header_bytes"]
        + record["projected_jsonl_bytes"] + record["projected_manifest_and_audit_bytes"]
    )
    if record["estimated_accepted_bytes"] != expected_accepted:
        raise CanonicalRecordError("accepted corpus byte formula mismatch")
    if record["controls_only_bytes"] > record["estimated_accepted_bytes"]:
        raise CanonicalRecordError("controls-only estimate exceeds accepted corpus estimate")
    for key in (
        "active_staging_bytes", "readback_atomic_publication_bytes",
        "preserved_failed_attempt_allowance_bytes",
    ):
        if record[key] != record["estimated_accepted_bytes"]:
            raise CanonicalRecordError(f"disk estimate {key} is not one complete accepted copy")
    if record["fixed_headroom_bytes"] != 2**30 or record["minimum_floor_bytes"] != 5 * 2**30:
        raise CanonicalRecordError("frozen storage floor or headroom changed")
    four_copy = 4 * record["estimated_accepted_bytes"] + record["projected_audit_bytes"] + record["fixed_headroom_bytes"]
    if record["four_copy_formula_bytes"] != four_copy or record["total_required_free_space_bytes"] != max(record["minimum_floor_bytes"], four_copy):
        raise CanonicalRecordError("required free-space formula mismatch")


def _validate_git_record(
    record: Mapping[str, Any], *, implementation_commit: str, implementation_tree: str,
    name: str,
) -> None:
    require_exact_keys(record, GIT_RECORD_KEYS, name=name)
    require_git_oid(record["head"], name=f"{name}.head")
    require_git_oid(record["tree"], name=f"{name}.tree")
    parents = record["parents"]
    if not isinstance(parents, list) or not parents:
        raise CanonicalRecordError(f"{name}.parents must be a non-empty ordered array")
    for index, parent in enumerate(parents):
        require_git_oid(parent, name=f"{name}.parents[{index}]")
    if len(set(parents)) != len(parents):
        raise CanonicalRecordError(f"{name}.parents must be unique")
    require_sha256(record["tracked_diff_sha256"], name=f"{name}.tracked_diff_sha256")
    require_sha256(record["staged_diff_sha256"], name=f"{name}.staged_diff_sha256")
    if (
        record["branch"] != "main" or record["head"] != implementation_commit
        or record["tree"] != implementation_tree
        or record["status_short_untracked_all"] != ""
        or type(record["tracked_diff_size_bytes"]) is not int
        or record["tracked_diff_size_bytes"] != 0
        or type(record["staged_diff_size_bytes"]) is not int
        or record["staged_diff_size_bytes"] != 0
        or record["tracked_diff_sha256"] != EMPTY_SHA256
        or record["staged_diff_sha256"] != EMPTY_SHA256
        or record["tracked_changes_empty"] is not True
        or record["staged_changes_empty"] is not True
        or record["untracked_inclusive_status_empty"] is not True
    ):
        raise CanonicalRecordError(f"{name} is not the clean frozen implementation state")


def _validate_preflight_bindings(
    bindings: ComputeFeasibilityPreflightBindings,
    capability: VerifiedState3Capability,
) -> None:
    for name in (
        "protocol_file_sha256", "state3_authorization_file_sha256",
        "launcher_file_sha256", "runtime_inspector_file_sha256",
        "environment_record_sha256", "storage_preflight_before_file_sha256",
    ):
        require_sha256(getattr(bindings, name), name=f"bindings.{name}")
    for name in (
        "solver_spec_baseline_commit", "solver_spec_baseline_tree",
        "implementation_commit", "implementation_tree",
    ):
        require_git_oid(getattr(bindings, name), name=f"bindings.{name}")
    if (
        bindings.state3_authorization_file_sha256 != capability.authorization_file_sha256
        or bindings.implementation_commit != capability.implementation_commit
        or bindings.implementation_tree != capability.implementation_tree
    ):
        raise MicrobenchmarkAuthorizationError("report bindings differ from State 3 capability")
    if (
        bindings.protocol_file_sha256 != PROTOCOL_SHA256
        or bindings.solver_spec_baseline_commit != SOLVER_SPEC_BASELINE_COMMIT
        or bindings.solver_spec_baseline_tree != SOLVER_SPEC_BASELINE_TREE
    ):
        raise MicrobenchmarkAuthorizationError("protocol or solver/specification baseline changed")
    if not isinstance(bindings.disk_estimate, Mapping):
        raise CanonicalRecordError("disk estimate binding must be an object")
    validate_disk_estimate(bindings.disk_estimate)
    if not isinstance(bindings.git_before, Mapping):
        raise CanonicalRecordError("pre-measurement Git binding must be an object")
    _validate_git_record(
        bindings.git_before, implementation_commit=bindings.implementation_commit,
        implementation_tree=bindings.implementation_tree, name="git_before",
    )


def _preflight_from_final(
    bindings: ComputeFeasibilityBindings,
) -> ComputeFeasibilityPreflightBindings:
    return ComputeFeasibilityPreflightBindings(
        protocol_file_sha256=bindings.protocol_file_sha256,
        state3_authorization_file_sha256=bindings.state3_authorization_file_sha256,
        solver_spec_baseline_commit=bindings.solver_spec_baseline_commit,
        solver_spec_baseline_tree=bindings.solver_spec_baseline_tree,
        implementation_commit=bindings.implementation_commit,
        implementation_tree=bindings.implementation_tree,
        launcher_file_sha256=bindings.launcher_file_sha256,
        runtime_inspector_file_sha256=bindings.runtime_inspector_file_sha256,
        environment_record_sha256=bindings.environment_record_sha256,
        storage_preflight_before_file_sha256=bindings.storage_preflight_before_file_sha256,
        disk_estimate=bindings.disk_estimate,
        git_before=bindings.git_before,
    )


def _validate_bindings(bindings: ComputeFeasibilityBindings, capability: VerifiedState3Capability) -> None:
    _validate_preflight_bindings(_preflight_from_final(bindings), capability)
    require_sha256(
        bindings.storage_preflight_after_file_sha256,
        name="bindings.storage_preflight_after_file_sha256",
    )
    if not isinstance(bindings.git_after, Mapping):
        raise CanonicalRecordError("post-measurement Git binding must be an object")
    _validate_git_record(
        bindings.git_after, implementation_commit=bindings.implementation_commit,
        implementation_tree=bindings.implementation_tree, name="git_after",
    )
    if canonical_json_bytes(bindings.git_before) != canonical_json_bytes(bindings.git_after):
        raise CanonicalRecordError("Git state changed across State 3 feasibility measurement")


def _require_final_extends_preflight(
    preflight: ComputeFeasibilityPreflightBindings,
    final: ComputeFeasibilityBindings,
) -> None:
    final_preflight = _preflight_from_final(final)
    if canonical_json_bytes(final_preflight.__dict__ if hasattr(final_preflight, "__dict__") else {
        name: getattr(final_preflight, name) for name in final_preflight.__dataclass_fields__
    }) != canonical_json_bytes({
        name: getattr(preflight, name) for name in preflight.__dataclass_fields__
    }):
        raise CanonicalRecordError("post-measurement bindings changed a preflight binding")


def load_compute_feasibility_bindings(path: str | Path) -> ComputeFeasibilityBindings:
    """Load one canonical metadata-only binding record outside the corpus root."""
    candidate = Path(path).resolve(strict=True)
    private_root = (
        Path(__file__).resolve().parents[2] / "artifacts" / "darcy_branch_c_v1"
        / "private_corpus_v1"
    ).resolve()
    try:
        candidate.relative_to(private_root)
    except ValueError:
        pass
    else:
        raise MicrobenchmarkAuthorizationError("State 3 bindings must remain outside private_corpus_v1")
    record = strict_canonical_json_loads(candidate.read_bytes())
    require_exact_keys(record, BINDINGS_KEYS, name="compute-feasibility bindings")
    if record["schema"] != "operatormark.darcy_branch_c_v1.compute_feasibility_bindings.v1":
        raise CanonicalRecordError("compute-feasibility bindings schema mismatch")
    validate_self_hash(record, "bindings_self_sha256")
    return ComputeFeasibilityBindings(**{
        key: record[key] for key in BINDINGS_KEYS - {"schema", "bindings_self_sha256"}
    })


def load_compute_feasibility_preflight_bindings(
    path: str | Path,
) -> ComputeFeasibilityPreflightBindings:
    """Load metadata known before measurement; no post-storage hash is allowed."""

    candidate = Path(path).resolve(strict=True)
    private_root = (
        Path(__file__).resolve().parents[2] / "artifacts" / "darcy_branch_c_v1"
        / "private_corpus_v1"
    ).resolve()
    try:
        candidate.relative_to(private_root)
    except ValueError:
        pass
    else:
        raise MicrobenchmarkAuthorizationError(
            "State 3 preflight bindings must remain outside private_corpus_v1"
        )
    record = strict_canonical_json_loads(candidate.read_bytes())
    require_exact_keys(record, PREFLIGHT_BINDINGS_KEYS, name="compute-feasibility preflight bindings")
    if record["schema"] != "operatormark.darcy_branch_c_v1.compute_feasibility_preflight_bindings.v1":
        raise CanonicalRecordError("compute-feasibility preflight bindings schema mismatch")
    validate_self_hash(record, "preflight_bindings_self_sha256")
    return ComputeFeasibilityPreflightBindings(**{
        key: record[key]
        for key in PREFLIGHT_BINDINGS_KEYS - {"schema", "preflight_bindings_self_sha256"}
    })


def _coerce_worker_result(result: object) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        return result
    if not isinstance(result, subprocess.CompletedProcess):
        raise CanonicalRecordError("worker launcher returned an invalid result type")
    if result.returncode != 0:
        raise CanonicalRecordError(f"fresh State 3 worker failed with exit code {result.returncode}")
    if result.stderr not in ("", b""):
        raise CanonicalRecordError("fresh State 3 worker emitted stderr")
    stdout = result.stdout.encode("utf-8") if isinstance(result.stdout, str) else result.stdout
    if not isinstance(stdout, bytes):
        raise CanonicalRecordError("fresh State 3 worker stdout is not bytes or text")
    return strict_canonical_json_loads(stdout)


def _default_launch_worker(
    capability: VerifiedState3Capability, fixture: str, repeat_index: int,
) -> subprocess.CompletedProcess[str]:
    from .launcher import MANDATORY_PARENT_ENVIRONMENT, ParentLaunchSpec, launch_fresh_child
    try:
        required = json.loads(os.environ["OPERATORMARK_REQUIRED_ENVIRONMENT_JSON"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise MicrobenchmarkAuthorizationError("coordinator environment binding is missing") from exc
    if not isinstance(required, dict):
        raise MicrobenchmarkAuthorizationError("coordinator environment binding is not an object")
    additional = tuple(sorted(
        (name, value) for name, value in required.items()
        if name not in MANDATORY_PARENT_ENVIRONMENT
    ))
    if not all(isinstance(name, str) and isinstance(value, str) for name, value in additional):
        raise MicrobenchmarkAuthorizationError("coordinator additional thread policy is invalid")
    spec = ParentLaunchSpec(
        state="state3", action="state3-microbenchmark-worker",
        authorization_file_sha256=capability.authorization_file_sha256,
        implementation_commit=capability.implementation_commit,
        implementation_tree=capability.implementation_tree,
        additional_environment=additional, internal_worker=True,
    )
    return launch_fresh_child(spec, (fixture, str(repeat_index)))


def run_compute_feasibility_measurements(
    capability: object,
    preflight_bindings: ComputeFeasibilityPreflightBindings | None,
    *,
    launch_worker: Callable[[str, int], object] | None = None,
) -> list[dict[str, Any]]:
    """Run only the 15 fresh measurements after before-storage is bound."""

    verified = require_state3_capability(capability)
    if preflight_bindings is None:
        raise MicrobenchmarkAuthorizationError(
            "canonical pre-measurement State 3 bindings are required before workers"
        )
    _validate_preflight_bindings(preflight_bindings, verified)
    worker_launcher = (
        (lambda fixture, repeat: _default_launch_worker(verified, fixture, repeat))
        if launch_worker is None else launch_worker
    )
    raw: list[dict[str, Any]] = []
    for fixture in FIXTURE_ORDER:
        for repeat_index in range(MEASURED_COUNT_PER_FIXTURE):
            measurement = _coerce_worker_result(worker_launcher(fixture, repeat_index))
            validate_raw_measurement(measurement)
            if measurement["fixture"] != fixture or measurement["repeat_index"] != repeat_index:
                raise CanonicalRecordError("worker result does not match requested fixture/repeat")
            raw.append(dict(measurement))
    return raw


def build_measurement_bundle(
    preflight_bindings: ComputeFeasibilityPreflightBindings,
    raw_measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind ordered raw results for post-storage report finalization."""

    raw = [dict(item) for item in raw_measurements]
    expected = [
        (fixture, repeat) for fixture in FIXTURE_ORDER
        for repeat in range(MEASURED_COUNT_PER_FIXTURE)
    ]
    if [(item.get("fixture"), item.get("repeat_index")) for item in raw] != expected:
        raise CanonicalRecordError("measurement-bundle coverage is incomplete or reordered")
    for item in raw:
        validate_raw_measurement(item)
    payload = {
        "schema": "operatormark.darcy_branch_c_v1.compute_feasibility_measurements.v1",
        "state": "state3_feasibility_measurements_complete_pending_post_storage",
        "protocol_file_sha256": preflight_bindings.protocol_file_sha256,
        "state3_authorization_file_sha256": preflight_bindings.state3_authorization_file_sha256,
        "implementation_commit": preflight_bindings.implementation_commit,
        "implementation_tree": preflight_bindings.implementation_tree,
        "warmup_count_per_fixture": 0,
        "measured_count_per_fixture": MEASURED_COUNT_PER_FIXTURE,
        "measurement_order": list(FIXTURE_ORDER),
        "raw_measurements": raw,
        "all_fixture_gates_passed": all(item["gates_passed"] for item in raw),
        "all_rng_state_equalities_passed": all(
            all(item["rng_state_equalities"].values()) for item in raw
        ),
        "forbidden_rng_call_count": sum(item["forbidden_rng_call_count"] for item in raw),
        "forbidden_corpus_seed_access_count": sum(
            item["forbidden_corpus_seed_access_count"] for item in raw
        ),
    }
    bundle = with_self_hash(payload, "measurement_bundle_self_sha256")
    validate_measurement_bundle(bundle)
    return bundle


def validate_measurement_bundle(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, MEASUREMENT_BUNDLE_KEYS, name="measurement bundle")
    if (
        record["schema"]
        != "operatormark.darcy_branch_c_v1.compute_feasibility_measurements.v1"
        or record["state"]
        != "state3_feasibility_measurements_complete_pending_post_storage"
        or record["protocol_file_sha256"] != PROTOCOL_SHA256
        or record["warmup_count_per_fixture"] != 0
        or record["measured_count_per_fixture"] != MEASURED_COUNT_PER_FIXTURE
        or record["measurement_order"] != list(FIXTURE_ORDER)
    ):
        raise CanonicalRecordError("measurement-bundle identity, counts, or order mismatch")
    for name in (
        "protocol_file_sha256", "state3_authorization_file_sha256",
        "measurement_bundle_self_sha256",
    ):
        require_sha256(record[name], name=f"measurement_bundle.{name}")
    require_git_oid(
        record["implementation_commit"], name="measurement_bundle.implementation_commit"
    )
    require_git_oid(
        record["implementation_tree"], name="measurement_bundle.implementation_tree"
    )
    raw = record["raw_measurements"]
    if not isinstance(raw, list) or len(raw) != 15:
        raise CanonicalRecordError("measurement bundle must contain exactly fifteen records")
    expected = [
        (fixture, repeat) for fixture in FIXTURE_ORDER
        for repeat in range(MEASURED_COUNT_PER_FIXTURE)
    ]
    if [(item.get("fixture"), item.get("repeat_index")) for item in raw] != expected:
        raise CanonicalRecordError("measurement-bundle fixture-major coverage mismatch")
    for item in raw:
        if not isinstance(item, Mapping):
            raise CanonicalRecordError("measurement-bundle item must be an object")
        validate_raw_measurement(item)
    expected_fixture = all(item["gates_passed"] for item in raw)
    expected_rng = all(all(item["rng_state_equalities"].values()) for item in raw)
    expected_rng_count = sum(item["forbidden_rng_call_count"] for item in raw)
    expected_seed_count = sum(item["forbidden_corpus_seed_access_count"] for item in raw)
    if (
        record["all_fixture_gates_passed"] is not expected_fixture
        or record["all_rng_state_equalities_passed"] is not expected_rng
        or record["forbidden_rng_call_count"] != expected_rng_count
        or record["forbidden_corpus_seed_access_count"] != expected_seed_count
        or expected_fixture is not True or expected_rng is not True
        or expected_rng_count != 0 or expected_seed_count != 0
    ):
        raise CanonicalRecordError("measurement-bundle aggregate mismatch")
    validate_self_hash(record, "measurement_bundle_self_sha256")


def build_compute_feasibility_report(
    bindings: ComputeFeasibilityBindings,
    raw_measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        bindings.protocol_file_sha256 != PROTOCOL_SHA256
        or bindings.solver_spec_baseline_commit != SOLVER_SPEC_BASELINE_COMMIT
        or bindings.solver_spec_baseline_tree != SOLVER_SPEC_BASELINE_TREE
    ):
        raise CanonicalRecordError("final report bindings changed the frozen baseline")
    validate_disk_estimate(bindings.disk_estimate)
    _validate_git_record(
        bindings.git_before,
        implementation_commit=bindings.implementation_commit,
        implementation_tree=bindings.implementation_tree,
        name="git_before",
    )
    _validate_git_record(
        bindings.git_after,
        implementation_commit=bindings.implementation_commit,
        implementation_tree=bindings.implementation_tree,
        name="git_after",
    )
    raw = [dict(item) for item in raw_measurements]
    expected_coverage = [
        (fixture, repeat) for fixture in FIXTURE_ORDER
        for repeat in range(MEASURED_COUNT_PER_FIXTURE)
    ]
    if [(item.get("fixture"), item.get("repeat_index")) for item in raw] != expected_coverage:
        raise CanonicalRecordError("raw measurements are incomplete, duplicated, or reordered")
    for item in raw:
        validate_raw_measurement(item)
    counter_names = {item["rss_counter_name"] for item in raw}
    if len(counter_names) != 1:
        raise CanonicalRecordError("workers used different process-memory counters")
    counter_name = next(iter(counter_names))
    payload = {
        "schema": "operatormark.darcy_branch_c_v1.compute_feasibility.v2",
        "state": "state3_feasibility_only_no_corpus_controls",
        "protocol_file_sha256": bindings.protocol_file_sha256,
        "state3_authorization_file_sha256": bindings.state3_authorization_file_sha256,
        "solver_spec_baseline_commit": bindings.solver_spec_baseline_commit,
        "solver_spec_baseline_tree": bindings.solver_spec_baseline_tree,
        "implementation_commit": bindings.implementation_commit,
        "implementation_tree": bindings.implementation_tree,
        "launcher_file_sha256": bindings.launcher_file_sha256,
        "runtime_inspector_file_sha256": bindings.runtime_inspector_file_sha256,
        "environment_record_sha256": bindings.environment_record_sha256,
        "storage_preflight_before_file_sha256": bindings.storage_preflight_before_file_sha256,
        "storage_preflight_after_file_sha256": bindings.storage_preflight_after_file_sha256,
        "fixture_definitions": FIXTURE_DEFINITIONS,
        "warmup_count_per_fixture": 0,
        "measured_count_per_fixture": MEASURED_COUNT_PER_FIXTURE,
        "measurement_order": list(FIXTURE_ORDER),
        "timing_clock": {
            "name": "time.perf_counter_ns", "unit": "nanoseconds", "monotonic": True,
            "interpretation": "same_platform_feasibility_estimates_not_performance_guarantees",
        },
        "memory_counter": {
            "name": counter_name, "unit": "bytes",
            "definition": "absolute_peak_working_set" if counter_name == "windows_process_peak_working_set_bytes" else "platform_ru_maxrss_normalized_to_bytes",
            "cross_platform_equivalence_claimed": False,
        },
        "rng_state_commitment_specification": RNG_STATE_COMMITMENT_SPECIFICATION,
        "raw_measurements": raw, "summary": _build_summary(raw),
        "corpus_sample_count": CORPUS_SAMPLE_COUNT,
        "frozen_safety_factor": FROZEN_SAFETY_FACTOR,
        "extrapolation": _build_extrapolation(raw),
        "disk_estimate": dict(bindings.disk_estimate),
        "all_fixture_gates_passed": all(item["gates_passed"] for item in raw),
        "all_rng_state_equalities_passed": all(all(item["rng_state_equalities"].values()) for item in raw),
        "forbidden_rng_call_count": sum(item["forbidden_rng_call_count"] for item in raw),
        "forbidden_corpus_seed_access_count": sum(item["forbidden_corpus_seed_access_count"] for item in raw),
        "git_before": dict(bindings.git_before), "git_after": dict(bindings.git_after),
    }
    report = with_self_hash(payload, "report_self_sha256")
    validate_compute_feasibility_report(report)
    return report


def validate_compute_feasibility_report(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, REPORT_KEYS, name="compute-feasibility report")
    if (
        record["schema"] != "operatormark.darcy_branch_c_v1.compute_feasibility.v2"
        or record["state"] != "state3_feasibility_only_no_corpus_controls"
    ):
        raise CanonicalRecordError("compute-feasibility report identity mismatch")
    if (
        record["protocol_file_sha256"] != PROTOCOL_SHA256
        or record["solver_spec_baseline_commit"] != SOLVER_SPEC_BASELINE_COMMIT
        or record["solver_spec_baseline_tree"] != SOLVER_SPEC_BASELINE_TREE
    ):
        raise CanonicalRecordError("compute-feasibility frozen baseline mismatch")
    if (
        type(record["warmup_count_per_fixture"]) is not int
        or record["warmup_count_per_fixture"] != 0
        or type(record["measured_count_per_fixture"]) is not int
        or record["measured_count_per_fixture"] != MEASURED_COUNT_PER_FIXTURE
        or type(record["corpus_sample_count"]) is not int
        or record["corpus_sample_count"] != CORPUS_SAMPLE_COUNT
        or type(record["frozen_safety_factor"]) is not float
        or record["frozen_safety_factor"] != FROZEN_SAFETY_FACTOR
        or record["measurement_order"] != list(FIXTURE_ORDER)
    ):
        raise CanonicalRecordError("compute-feasibility frozen counts or order mismatch")
    if record["fixture_definitions"] != FIXTURE_DEFINITIONS:
        raise CanonicalRecordError("fixture definitions changed")
    if record["rng_state_commitment_specification"] != RNG_STATE_COMMITMENT_SPECIFICATION:
        raise CanonicalRecordError("RNG state commitment specification changed")
    expected_clock = {
        "name": "time.perf_counter_ns", "unit": "nanoseconds", "monotonic": True,
        "interpretation": "same_platform_feasibility_estimates_not_performance_guarantees",
    }
    if record["timing_clock"] != expected_clock:
        raise CanonicalRecordError("timing clock record mismatch")
    memory = record["memory_counter"]
    if not isinstance(memory, Mapping):
        raise CanonicalRecordError("memory counter record must be an object")
    require_exact_keys(memory, {"name", "unit", "definition", "cross_platform_equivalence_claimed"}, name="memory counter")
    if (
        not isinstance(memory["name"], str) or not memory["name"] or memory["unit"] != "bytes"
        or memory["cross_platform_equivalence_claimed"] is not False
    ):
        raise CanonicalRecordError("memory counter identity is invalid")
    raw = record["raw_measurements"]
    if not isinstance(raw, list) or len(raw) != len(FIXTURE_ORDER) * MEASURED_COUNT_PER_FIXTURE:
        raise CanonicalRecordError("exactly fifteen raw measurements are required")
    expected_coverage = [
        (fixture, repeat) for fixture in FIXTURE_ORDER
        for repeat in range(MEASURED_COUNT_PER_FIXTURE)
    ]
    if [(item.get("fixture"), item.get("repeat_index")) for item in raw] != expected_coverage:
        raise CanonicalRecordError("fixture-major repeat coverage mismatch")
    for item in raw:
        if not isinstance(item, Mapping):
            raise CanonicalRecordError("raw measurement must be an object")
        validate_raw_measurement(item)
        if item["rss_counter_name"] != memory["name"]:
            raise CanonicalRecordError("raw memory counter differs from report counter")
    expected_summary = _build_summary(raw)
    summary = record["summary"]
    if not isinstance(summary, Mapping):
        raise CanonicalRecordError("summary must be an object")
    require_exact_keys(summary, SUMMARY_KEYS, name="summary")
    if (
        not isinstance(summary["global"], Mapping)
        or not isinstance(summary["per_fixture"], Mapping)
    ):
        raise CanonicalRecordError("summary groups must be objects")
    require_exact_keys(summary["per_fixture"], FIXTURE_ORDER, name="per-fixture summary")
    if summary["fields"] != list(SUMMARY_FIELDS) or summary != expected_summary:
        raise CanonicalRecordError("global or per-fixture summary formula mismatch")
    for group in [summary["global"], *summary["per_fixture"].values()]:
        require_exact_keys(group, SUMMARY_FIELDS, name="summary group")
        for statistic in group.values():
            if not isinstance(statistic, Mapping):
                raise CanonicalRecordError("summary statistic must be an object")
            require_exact_keys(statistic, SUMMARY_STATISTIC_KEYS, name="summary statistic")
            if type(statistic["maximum"]) is not int or (
                isinstance(statistic["median"], bool)
                or not isinstance(statistic["median"], (int, float))
                or not math.isfinite(float(statistic["median"]))
            ):
                raise CanonicalRecordError("summary statistic types are invalid")
    extrapolation = record["extrapolation"]
    if not isinstance(extrapolation, Mapping):
        raise CanonicalRecordError("extrapolation must be an object")
    require_exact_keys(extrapolation, EXTRAPOLATION_KEYS, name="extrapolation")
    for name, value in extrapolation.items():
        if name == "safety_factored_peak_memory_budget_bytes":
            if type(value) is not int or value < 0:
                raise CanonicalRecordError("peak-memory budget must be a non-negative integer")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise CanonicalRecordError(f"extrapolation.{name} must be finite and non-negative")
    if extrapolation != _build_extrapolation(raw):
        raise CanonicalRecordError("extrapolation formula mismatch")
    if not isinstance(record["disk_estimate"], Mapping):
        raise CanonicalRecordError("disk estimate must be an object")
    validate_disk_estimate(record["disk_estimate"])
    expected_fixture_gate = all(item["gates_passed"] for item in raw)
    expected_rng_gate = all(all(item["rng_state_equalities"].values()) for item in raw)
    expected_rng_count = sum(item["forbidden_rng_call_count"] for item in raw)
    expected_seed_count = sum(item["forbidden_corpus_seed_access_count"] for item in raw)
    if (
        record["all_fixture_gates_passed"] is not expected_fixture_gate
        or record["all_rng_state_equalities_passed"] is not expected_rng_gate
        or record["forbidden_rng_call_count"] != expected_rng_count
        or record["forbidden_corpus_seed_access_count"] != expected_seed_count
        or expected_fixture_gate is not True or expected_rng_gate is not True
        or expected_rng_count != 0 or expected_seed_count != 0
        or type(record["forbidden_rng_call_count"]) is not int
        or type(record["forbidden_corpus_seed_access_count"]) is not int
    ):
        raise CanonicalRecordError("top-level feasibility aggregation mismatch")
    for name in (
        "protocol_file_sha256", "state3_authorization_file_sha256",
        "launcher_file_sha256", "runtime_inspector_file_sha256",
        "environment_record_sha256", "storage_preflight_before_file_sha256",
        "storage_preflight_after_file_sha256", "report_self_sha256",
    ):
        require_sha256(record[name], name=name)
    for name in (
        "solver_spec_baseline_commit", "solver_spec_baseline_tree",
        "implementation_commit", "implementation_tree",
    ):
        require_git_oid(record[name], name=name)
    if not isinstance(record["git_before"], Mapping) or not isinstance(record["git_after"], Mapping):
        raise CanonicalRecordError("Git records must be objects")
    _validate_git_record(
        record["git_before"], implementation_commit=record["implementation_commit"],
        implementation_tree=record["implementation_tree"], name="git_before",
    )
    _validate_git_record(
        record["git_after"], implementation_commit=record["implementation_commit"],
        implementation_tree=record["implementation_tree"], name="git_after",
    )
    if canonical_json_bytes(record["git_before"]) != canonical_json_bytes(record["git_after"]):
        raise CanonicalRecordError("Git state changed across feasibility report")
    validate_self_hash(record, "report_self_sha256")


def run_compute_feasibility(
    capability: object,
    preflight_bindings: ComputeFeasibilityPreflightBindings | None = None,
    *,
    finalize_bindings: Callable[
        [Sequence[Mapping[str, Any]]], ComputeFeasibilityBindings
    ] | None = None,
    launch_worker: Callable[[str, int], object] | None = None,
) -> dict[str, Any]:
    """Measure, then obtain post-storage bindings, then finalize the report."""

    verified = require_state3_capability(capability)
    if preflight_bindings is None or finalize_bindings is None:
        raise MicrobenchmarkAuthorizationError(
            "preflight bindings and a post-measurement finalizer are required before workers"
        )
    raw = run_compute_feasibility_measurements(
        verified, preflight_bindings, launch_worker=launch_worker
    )
    final_bindings = finalize_bindings(tuple(raw))
    if not isinstance(final_bindings, ComputeFeasibilityBindings):
        raise CanonicalRecordError("post-measurement finalizer returned invalid bindings")
    _validate_bindings(final_bindings, verified)
    _require_final_extends_preflight(preflight_bindings, final_bindings)
    return build_compute_feasibility_report(final_bindings, raw)


def main(
    argv: Sequence[str] | None = None,
    *,
    capability: VerifiedState3Capability | None = None,
    launch_worker: Callable[[str, int], object] | None = None,
) -> int:
    """Bound coordinator entry point; direct/unbootstrapped use fails closed."""

    import sys

    context = sys.modules.get("_operatormark_fresh_child_bootstrap")
    if getattr(context, "sentinel", None) != "operatormark-fresh-child-bootstrap-v2":
        raise MicrobenchmarkAuthorizationError(
            "direct microbenchmark execution lacks the guarded fresh-child bootstrap"
        )
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise MicrobenchmarkAuthorizationError(
            "State 3 coordinator entry point requires one canonical preflight-bindings path"
        )
    bound_capability = capability or VerifiedState3Capability(
        authorization_file_sha256=os.environ.get("OPERATORMARK_AUTHORIZATION_FILE_SHA256", ""),
        implementation_commit=os.environ.get("OPERATORMARK_IMPLEMENTATION_COMMIT", ""),
        implementation_tree=os.environ.get("OPERATORMARK_IMPLEMENTATION_TREE", ""),
    )
    preflight = load_compute_feasibility_preflight_bindings(arguments[0])
    raw = run_compute_feasibility_measurements(
        bound_capability, preflight, launch_worker=launch_worker
    )
    sys.stdout.write(canonical_json_bytes(build_measurement_bundle(preflight, raw)).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
