"""TPFA family: forward, factorization reuse, tangent, adjoint VJP, eta.

Independence (v1 §7.2): this module references only the frozen TPFA solver
(``operatormark.solvers.darcy_tpfa``), the problem definition in
``operatormark.benchmarks.darcy_spec``, ``views`` and ``api``.  It never
references the P1 solver or the P1 family (checked by test C7a/C7b).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import splu

from operatormark.benchmarks import darcy_spec as _spec
from operatormark.solvers import darcy_tpfa as _tpfa

from . import views
from .api import TPFA_LEVELS, ForwardResult, NumericsError, TangentResult

DIAGNOSTIC_KEYS = (
    "algebraic_backward_error",
    "boundary_transmissibility_error",
    "maximum_local_conservation_ratio",
    "maximum_interior_face_antisymmetry",
    "global_balance_ratio",
    "rhs_relative_l2_residual",
)
TANGENT_GATE = 1.0e-10
_ALLOWED_LEVELS = frozenset(TPFA_LEVELS)


def _expand(values8: np.ndarray, n: int) -> np.ndarray:
    """This family's own macro-cell expansion (used for dkappa = 9 v)."""

    repeats = n // 8
    return np.ascontiguousarray(np.repeat(np.repeat(values8, repeats, axis=0), repeats, axis=1))


def _check_direction(v: np.ndarray) -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.shape != (8, 8) or v.dtype != np.float64:
        raise TypeError("direction must be a float64 [8,8] array")
    if not np.isfinite(v).all():
        raise ValueError("direction must be finite")
    return np.ascontiguousarray(v)


class NativeTPFA:
    """Frozen assembly plus one retained SuperLU factorization on any N-grid."""

    def __init__(self, kappa: np.ndarray):
        self.system = _tpfa.assemble_zero_dirichlet_benchmark(kappa)
        self.n = int(self.system.resolution)
        self.kappa = self.system.kappa
        try:
            # Same call and default options as the frozen solve_tpfa.
            self.lu = splu(self.system.matrix.tocsc())
            flat = self.lu.solve(self.system.rhs)
        except RuntimeError as exc:  # pragma: no cover - frozen path never observed failing
            raise NumericsError("tpfa_superlu_failure") from exc
        if flat.shape != self.system.rhs.shape or flat.dtype != np.float64 or not np.isfinite(flat).all():
            raise NumericsError("tpfa_invalid_solution")
        self.u_flat = flat
        self.u = np.ascontiguousarray(flat.reshape((self.n, self.n)))

    # --- algebra -----------------------------------------------------------
    def solve(self, rhs_flat: np.ndarray) -> np.ndarray:
        out = self.lu.solve(np.ascontiguousarray(rhs_flat, dtype=np.float64))
        if not np.isfinite(out).all():
            raise NumericsError("tpfa_invalid_solution")
        return out

    def refinement_correction(self, x_flat: np.ndarray, rhs_flat: np.ndarray) -> np.ndarray:
        residual = rhs_flat - self.system.matrix @ x_flat
        return self.solve(residual)

    def dA_times(self, dkappa: np.ndarray, w: np.ndarray) -> np.ndarray:
        """(dA w) for the coefficient perturbation dkappa, both on the native grid."""

        k = self.kappa
        out = np.zeros_like(w)
        # interior x-faces between (j,i) and (j,i+1)
        ka, kb = k[:, :-1], k[:, 1:]
        dka, dkb = dkappa[:, :-1], dkappa[:, 1:]
        dt = 2.0 * (kb * kb * dka + ka * ka * dkb) / ((ka + kb) * (ka + kb))
        flux = dt * (w[:, :-1] - w[:, 1:])
        out[:, :-1] += flux
        out[:, 1:] -= flux
        # interior y-faces between (j,i) and (j+1,i)
        ka, kb = k[:-1, :], k[1:, :]
        dka, dkb = dkappa[:-1, :], dkappa[1:, :]
        dt = 2.0 * (kb * kb * dka + ka * ka * dkb) / ((ka + kb) * (ka + kb))
        flux = dt * (w[:-1, :] - w[1:, :])
        out[:-1, :] += flux
        out[1:, :] -= flux
        # boundary faces: T = kappa*h/(h/2) = 2 kappa exactly, so dT = 2 dkappa
        out[:, 0] += 2.0 * dkappa[:, 0] * w[:, 0]
        out[:, -1] += 2.0 * dkappa[:, -1] * w[:, -1]
        out[0, :] += 2.0 * dkappa[0, :] * w[0, :]
        out[-1, :] += 2.0 * dkappa[-1, :] * w[-1, :]
        return out

    def kappa_gradient(self, lam: np.ndarray, u: np.ndarray) -> np.ndarray:
        """d<g,u>/dkappa given the adjoint lam (A lam = g):  -lam^T (dA/dkappa_c) u."""

        k = self.kappa
        grad = np.zeros_like(k)
        ka, kb = k[:, :-1], k[:, 1:]
        prod = (lam[:, :-1] - lam[:, 1:]) * (u[:, :-1] - u[:, 1:])
        denom = (ka + kb) * (ka + kb)
        grad[:, :-1] -= 2.0 * kb * kb / denom * prod
        grad[:, 1:] -= 2.0 * ka * ka / denom * prod
        ka, kb = k[:-1, :], k[1:, :]
        prod = (lam[:-1, :] - lam[1:, :]) * (u[:-1, :] - u[1:, :])
        denom = (ka + kb) * (ka + kb)
        grad[:-1, :] -= 2.0 * kb * kb / denom * prod
        grad[1:, :] -= 2.0 * ka * ka / denom * prod
        grad[:, 0] -= 2.0 * lam[:, 0] * u[:, 0]
        grad[:, -1] -= 2.0 * lam[:, -1] * u[:, -1]
        grad[0, :] -= 2.0 * lam[0, :] * u[0, :]
        grad[-1, :] -= 2.0 * lam[-1, :] * u[-1, :]
        return grad

    def tangent_native(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (du [N,N], rhs_flat, backward error) for direction v [8,8]."""

        dkappa = _expand(np.float64(9.0) * _check_direction(v), self.n)
        rhs = -self.dA_times(dkappa, self.u).reshape(-1)
        du_flat = self.solve(rhs)
        backward = float(_tpfa.algebraic_backward_error(self.system.matrix, du_flat, rhs))
        return np.ascontiguousarray(du_flat.reshape((self.n, self.n))), rhs, backward

    def rho_gradient(self, g_native: np.ndarray) -> np.ndarray:
        """d<g,u>/drho [8,8] via one adjoint solve with the retained factorization."""

        lam = self.solve(np.ascontiguousarray(g_native, dtype=np.float64).reshape(-1)).reshape((self.n, self.n))
        gk = self.kappa_gradient(lam, self.u)
        block = self.n // 8
        return np.ascontiguousarray(np.float64(9.0) * gk.reshape(8, block, 8, block).sum(axis=(1, 3)))


def _frozen_failure(exc: Exception) -> NumericsError:
    """Map a frozen-module ValueError (TPFAError, DarcySpecError) to a deterministic outcome code."""

    return NumericsError("tpfa_frozen_error:" + type(exc).__name__)


class TPFASession:
    """One control at one level.  Holds the factorization until close()."""

    method = "TPFA"

    def __init__(self, rho: np.ndarray, level: int, *, levels: frozenset[int] = _ALLOWED_LEVELS):
        if level not in levels:
            raise ValueError(f"TPFA level {level} not allowed")
        try:
            self._open(rho, level)
        except NumericsError:
            raise
        except (ValueError, ArithmeticError) as exc:
            raise _frozen_failure(exc) from exc

    def _open(self, rho: np.ndarray, level: int) -> None:
        self.rho = _spec.validate_control(rho)
        self.level = int(level)
        kappa = _spec.coefficient_on_tpfa_grid(self.rho, self.level)
        self._native = NativeTPFA(kappa)
        diag = _tpfa.compute_tpfa_diagnostics(self._native.system, self._native.u)
        diagnostics = {key: (None if getattr(diag, key) is None else float(getattr(diag, key))) for key in DIAGNOSTIC_KEYS}
        try:
            field64, transfer = views.checked_restrict_to_64(self._native.u, self.level)
        except ValueError as exc:
            raise NumericsError("tpfa_transfer_integral_failure") from exc
        delta = self._native.refinement_correction(self._native.u_flat, self._native.system.rhs)
        correction64 = views.restrict_to_64(np.ascontiguousarray(delta.reshape((self.level, self.level))), self.level)
        self.control_sha256 = _spec.tensor_fingerprint(self.rho)
        self.forward = ForwardResult(
            method="TPFA",
            level=self.level,
            control_sha256=self.control_sha256,
            native_sha256=_spec.tensor_fingerprint(self._native.u),
            coefficient_sha256=_spec.tensor_fingerprint(self._native.kappa),
            field64=field64,
            correction64=correction64,
            diagnostics=diagnostics,
            gates_pass=bool(diag.passes),
            transfer_integral_error=transfer,
            iterations=None,
            eta_iterations=None,
        )

    def native_solution(self) -> np.ndarray:
        self._require_open()
        return self._native.u.copy()

    def tangent(self, v: np.ndarray) -> TangentResult:
        self._require_open()
        try:
            return self._tangent(v)
        except NumericsError:
            raise
        except (ValueError, ArithmeticError) as exc:
            raise _frozen_failure(exc) from exc

    def _tangent(self, v: np.ndarray) -> TangentResult:
        du, rhs, backward = self._native.tangent_native(v)
        delta = self._native.refinement_correction(du.reshape(-1), rhs)
        try:
            field64, _ = views.checked_restrict_to_64(du, self.level)
        except ValueError as exc:
            raise NumericsError("tpfa_tangent_transfer_integral_failure") from exc
        correction64 = views.restrict_to_64(np.ascontiguousarray(delta.reshape((self.level, self.level))), self.level)
        return TangentResult(
            method="TPFA",
            level=self.level,
            control_sha256=self.control_sha256,
            direction_sha256=_spec.tensor_fingerprint(_check_direction(v)),
            native_sha256=_spec.tensor_fingerprint(du),
            field64=field64,
            correction64=correction64,
            tangent_backward_error=backward,
            iterations=None,
            eta_iterations=None,
        )

    def vjp64(self, w64: np.ndarray) -> np.ndarray:
        """(D F64)^T w for a 64-grid weight w -> [8,8]."""

        self._require_open()
        g_native = views.restrict_transpose(np.ascontiguousarray(w64, dtype=np.float64), self.level)
        return self._native.rho_gradient(g_native)

    def close(self) -> None:
        self._native = None

    def _require_open(self) -> None:
        if self._native is None:
            raise RuntimeError("session is closed")

    def __enter__(self) -> "TPFASession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TPFAFamily:
    name = "TPFA"

    def __init__(self, levels: tuple[int, ...] = TPFA_LEVELS):
        self.levels = tuple(levels)
        self._allowed = frozenset(levels)

    def open(self, rho: np.ndarray, level: int, *, compute_eta: bool = True) -> TPFASession:
        # The TPFA eta (one refinement correction) is negligible, so it is always computed.
        return TPFASession(rho, level, levels=self._allowed)


class TPFAFiniteDifferenceOracle:
    """Field evaluation for finite-difference checks: the frozen direct solve, mapped by R."""

    def field64(self, rho: np.ndarray, level: int) -> np.ndarray:
        try:
            kappa = _spec.coefficient_on_tpfa_grid(_spec.validate_control(rho), level)
            _, u = _tpfa.solve_zero_dirichlet_benchmark(kappa)
        except (ValueError, ArithmeticError) as exc:
            raise _frozen_failure(exc) from exc
        return views.restrict_to_64(u, level)


def task_field_and_vjp(level: int):
    """Lightweight (field64, vjp) closure for the D8 optimizer: no diagnostics, no eta."""

    if level not in _ALLOWED_LEVELS:
        raise ValueError(f"TPFA level {level} not allowed")

    def field_and_vjp(rho: np.ndarray):
        try:
            control = _spec.validate_control(rho)
            native = NativeTPFA(_spec.coefficient_on_tpfa_grid(control, level))
        except NumericsError:
            raise
        except (ValueError, ArithmeticError) as exc:
            raise _frozen_failure(exc) from exc
        field = views.restrict_to_64(native.u, level)

        def vjp(w64: np.ndarray) -> np.ndarray:
            return native.rho_gradient(views.restrict_transpose(np.ascontiguousarray(w64, dtype=np.float64), level))

        return field, vjp

    return field_and_vjp
