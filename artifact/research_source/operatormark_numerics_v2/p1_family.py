"""P1 family: frozen Jacobi-PCG forward, signed element-action tangent, PCG
continuation for eta, exact projection to 64 (N = 128, 256, 512), and a
direct-solve finite-difference oracle.

Independence (v1 §7.2): this module references only the frozen P1 solver
(``operatormark.solvers.darcy_p1``), the problem definition in
``operatormark.benchmarks.darcy_spec`` and ``api``.  It never references the
TPFA solver, the TPFA family or ``views`` (checked by test C7a/C7b).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, splu

from operatormark.benchmarks import darcy_spec as _spec
from operatormark.solvers import darcy_p1 as _p1

from .api import P1_LEVELS, ForwardResult, NumericsError, TangentResult

DIAGNOSTIC_KEYS = (
    "algebraic_backward_error",
    "boundary_reconstruction_error",
    "reaction_denominator",
    "reaction_balance_ratio",
    "outward_boundary_flux",
    "integrated_source",
    "rhs_relative_l2_residual",
)
ETA_RTOL = 1.0e-14
TANGENT_GATE = 1.0e-10
PROJECTION_RESOLUTIONS = (128, 256, 512)
TEST_ONLY_PROJECTION_RESOLUTIONS = (64,)
_EPS = float(np.finfo(np.float64).eps)


def project_to_64(nodal: np.ndarray, n: int, *, allow_test_levels: bool = False) -> tuple[np.ndarray, float]:
    """Exact cell-integral projection of a nodal P1 field to the 64-grid.

    At N = 128 and 256 this is the frozen project_p1_to_64 expression (bitwise,
    test A3); N = 512 extends it with the same formula.  N = 64 is accepted only
    under test level sets.
    """

    allowed = PROJECTION_RESOLUTIONS + (TEST_ONLY_PROJECTION_RESOLUTIONS if allow_test_levels else ())
    if n not in allowed:
        raise ValueError(f"P1 projection resolution {n} not allowed")
    if not isinstance(nodal, np.ndarray) or nodal.shape != (n + 1, n + 1) or nodal.dtype != np.float64:
        raise TypeError("nodal field must be float64 [N+1,N+1]")
    if not np.isfinite(nodal).all():
        raise ValueError("nodal field must be finite")
    values = np.array(nodal, dtype=np.float64, order="C", copy=True)
    block = n // 64
    square_integrals = _p1._p1_square_integrals(values, n)
    reshaped = square_integrals.reshape(64, block, 64, block)
    cell_integrals = np.sum(reshaped, axis=(1, 3), dtype=np.float64)
    projected = np.ascontiguousarray(cell_integrals * np.float64(64**2), dtype=np.float64)
    original = float(np.sum(square_integrals, dtype=np.float64))
    projected_integral = float(np.sum(projected, dtype=np.float64) / np.float64(64**2))
    scale = float(np.sum(np.abs(square_integrals), dtype=np.float64))
    tolerance = 64.0 * _EPS * max(1.0, scale)
    error = abs(projected_integral - original)
    if not math.isfinite(error) or error > tolerance:
        raise NumericsError("p1_projection_integral_failure")
    return projected, error / max(1.0, scale)


def _jacobi(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    return sparse.diags(1.0 / diagonal, offsets=0, shape=matrix.shape, format="csr", dtype=np.float64)


def _continue_pcg(system, interior_start: np.ndarray, rhs: np.ndarray, iterations: int) -> tuple[np.ndarray, int]:
    """Continue PCG from the returned solution for at most `iterations` steps or to rtol 1e-14.

    info > 0 (not converged within the budget) is the expected outcome, not an error.
    Returns (continued - returned, iterations used).
    """

    if iterations <= 0 or interior_start.size == 0:
        return np.zeros_like(interior_start), 0
    count = 0

    def _count(_):
        nonlocal count
        count += 1

    continued, info = cg(
        system.reduced_matrix,
        rhs,
        x0=interior_start.copy(),
        rtol=ETA_RTOL,
        atol=0.0,
        maxiter=int(iterations),
        M=_jacobi(system.reduced_matrix),
        callback=_count,
    )
    if info < 0 or not np.isfinite(continued).all():
        raise NumericsError("p1_eta_continuation_breakdown")
    return np.ascontiguousarray(continued - interior_start), count


def _nodal(system, interior: np.ndarray) -> np.ndarray:
    flat = np.zeros(system.mesh.node_count, dtype=np.float64)
    flat[system.mesh.boundary_nodes] = system.boundary_values
    flat[system.mesh.interior_nodes] = interior
    width = system.mesh.resolution + 1
    return np.ascontiguousarray(flat.reshape((width, width)))


def _nodal_correction(system, interior_delta: np.ndarray) -> np.ndarray:
    flat = np.zeros(system.mesh.node_count, dtype=np.float64)
    flat[system.mesh.interior_nodes] = interior_delta
    width = system.mesh.resolution + 1
    return np.ascontiguousarray(flat.reshape((width, width)))


def _check_direction(v: np.ndarray) -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.shape != (8, 8) or v.dtype != np.float64:
        raise TypeError("direction must be a float64 [8,8] array")
    if not np.isfinite(v).all():
        raise ValueError("direction must be finite")
    return np.ascontiguousarray(v)


def element_action(system, dsquare: np.ndarray, nodal: np.ndarray) -> np.ndarray:
    """Full nodal vector dK u for a signed per-square coefficient perturbation.

    dK = sum_t dkappa_t * area_t * G_t G_t^T, applied without assembling dK
    (the frozen _finalize_assembly rejects non-positive coefficient integrals).
    """

    _, determinants, gradients = _p1._mesh_triangle_geometry(system.mesh)
    areas = 0.5 * determinants
    triangle_d = np.repeat(np.ascontiguousarray(dsquare).reshape(-1, order="C"), 2)
    weights = triangle_d * areas
    u_local = nodal.reshape(-1)[system.mesh.triangles]                 # (E, 3)
    grad_u = np.einsum("tai,ta->ti", gradients, u_local)                # (E, 2)
    local = weights[:, None] * np.einsum("tai,ti->ta", gradients, grad_u)
    return np.bincount(
        system.mesh.triangles.reshape(-1), weights=local.reshape(-1), minlength=system.mesh.node_count
    ).astype(np.float64, copy=False)


def _frozen_failure(exc: Exception) -> NumericsError:
    """Map a frozen-module ValueError (P1Error, DarcySpecError) to a deterministic outcome code."""

    return NumericsError("p1_frozen_error:" + type(exc).__name__)


def dsquare_from_direction(v: np.ndarray, n: int) -> np.ndarray:
    """This family's own expansion of dkappa = 9 v to the N x N squares."""

    repeats = n // 8
    return np.ascontiguousarray(np.repeat(np.repeat(np.float64(9.0) * _check_direction(v), repeats, axis=0), repeats, axis=1))


class P1Session:
    method = "P1"

    def __init__(self, rho: np.ndarray, level: int, *, levels: tuple[int, ...] = P1_LEVELS, allow_test_levels: bool = False,
                 compute_eta: bool = True):
        if level not in levels:
            raise ValueError(f"P1 level {level} not allowed")
        try:
            self._open(rho, level, allow_test_levels, compute_eta)
        except NumericsError:
            raise
        except (ValueError, ArithmeticError) as exc:
            raise _frozen_failure(exc) from exc

    def _open(self, rho: np.ndarray, level: int, allow_test_levels: bool, compute_eta: bool) -> None:
        """compute_eta=False skips the forward continuation (used when only tangents are needed)."""

        self.rho = _spec.validate_control(rho)
        self.level = int(level)
        self._test = bool(allow_test_levels)
        self.system = _p1.assemble_p1_zero_dirichlet_benchmark(self.rho, self.level)
        try:
            result = _p1.solve_p1_with_info(self.system)
        except _p1.P1Error as exc:
            raise NumericsError("p1_pcg_failure") from exc
        self.nodal = result.solution
        self.iterations = int(result.iterations)
        diag = _p1.compute_p1_diagnostics(self.system, self.nodal)
        diagnostics = {key: (None if getattr(diag, key) is None else float(getattr(diag, key))) for key in DIAGNOSTIC_KEYS}
        field64, transfer = project_to_64(self.nodal, self.level, allow_test_levels=self._test)
        interior = self.nodal.reshape(-1)[self.system.mesh.interior_nodes]
        if compute_eta:
            delta, eta_iterations = _continue_pcg(self.system, interior, self.system.reduced_rhs, self.iterations)
            correction64, _ = project_to_64(_nodal_correction(self.system, delta), self.level, allow_test_levels=self._test)
        else:
            correction64, eta_iterations = np.full((64, 64), np.nan), None
        self.control_sha256 = _spec.tensor_fingerprint(self.rho)
        coefficients = _p1.benchmark_p1_element_coefficients(self.rho, self.level)
        self.forward = ForwardResult(
            method="P1",
            level=self.level,
            control_sha256=self.control_sha256,
            native_sha256=_spec.tensor_fingerprint(self.nodal),
            coefficient_sha256=_spec.tensor_fingerprint(coefficients),
            field64=field64,
            correction64=correction64,
            diagnostics=diagnostics,
            gates_pass=bool(diag.passes),
            transfer_integral_error=transfer,
            iterations=self.iterations,
            eta_iterations=eta_iterations,
        )

    def native_solution(self) -> np.ndarray:
        self._require_open()
        return self.nodal.copy()

    def tangent(self, v: np.ndarray) -> TangentResult:
        self._require_open()
        try:
            return self._tangent(v)
        except NumericsError:
            raise
        except (ValueError, ArithmeticError) as exc:
            raise _frozen_failure(exc) from exc

    def _tangent(self, v: np.ndarray) -> TangentResult:
        dsquare = dsquare_from_direction(v, self.level)
        action = element_action(self.system, dsquare, self.nodal)
        rhs = np.ascontiguousarray(-action[self.system.mesh.interior_nodes])
        tangent_system = dataclasses.replace(self.system, reduced_rhs=rhs)
        try:
            result = _p1.solve_p1_with_info(tangent_system)
        except _p1.P1Error as exc:
            raise NumericsError("p1_tangent_pcg_failure") from exc
        du = result.solution
        interior = du.reshape(-1)[self.system.mesh.interior_nodes]
        backward = float(_p1.p1_algebraic_backward_error(self.system.reduced_matrix, interior, rhs))
        delta, eta_iterations = _continue_pcg(self.system, interior, rhs, int(result.iterations))
        field64, _ = project_to_64(du, self.level, allow_test_levels=self._test)
        correction64, _ = project_to_64(_nodal_correction(self.system, delta), self.level, allow_test_levels=self._test)
        return TangentResult(
            method="P1",
            level=self.level,
            control_sha256=self.control_sha256,
            direction_sha256=_spec.tensor_fingerprint(_check_direction(v)),
            native_sha256=_spec.tensor_fingerprint(du),
            field64=field64,
            correction64=correction64,
            tangent_backward_error=backward,
            iterations=int(result.iterations),
            eta_iterations=eta_iterations,
        )

    def close(self) -> None:
        self.system = None
        self.nodal = None

    def _require_open(self) -> None:
        if self.system is None:
            raise RuntimeError("session is closed")

    def __enter__(self) -> "P1Session":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class P1Family:
    name = "P1"

    def __init__(self, levels: tuple[int, ...] = P1_LEVELS, *, allow_test_levels: bool = False):
        self.levels = tuple(levels)
        self._test = bool(allow_test_levels)

    def open(self, rho: np.ndarray, level: int, *, compute_eta: bool = True) -> P1Session:
        return P1Session(rho, level, levels=self.levels, allow_test_levels=self._test, compute_eta=compute_eta)


class P1DirectOracle:
    """Finite-difference oracle: the frozen assembly solved by this family's own sparse LU.

    Used only to evaluate the map for the D6 slope check.  PCG's stopping noise
    (cond x ~1e-11 relative) divided by h would swamp the O(h^2) truncation error.
    """

    def __init__(self, *, allow_test_levels: bool = False):
        self._test = bool(allow_test_levels)

    def field64(self, rho: np.ndarray, level: int) -> np.ndarray:
        try:
            system = _p1.assemble_p1_zero_dirichlet_benchmark(_spec.validate_control(rho), level)
            interior = splu(system.reduced_matrix.tocsc()).solve(system.reduced_rhs)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            if isinstance(exc, NumericsError):
                raise
            raise _frozen_failure(exc) from exc
        if not np.isfinite(interior).all():
            raise NumericsError("p1_direct_oracle_failure")
        field, _ = project_to_64(_nodal(system, interior), level, allow_test_levels=self._test)
        return field
