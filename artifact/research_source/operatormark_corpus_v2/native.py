"""Drop 2 adapter: native T_{N_ref} arrays and P1-256 evidence from the unchanged Drop 1 numerics.

Drop 1's ``TPFASession`` exposes the native forward solution but returns only
the hash of a native tangent, and it raises (rather than records) when the
R integral check fails.  Generation needs the native tangent bytes under
retention S and must *retain* a failed R check as a measurement.  So this
adapter composes the same Drop 1 routines directly - ``NativeTPFA`` (frozen
assembly plus one SuperLU factorization, ``tangent_native``), the frozen
``compute_tpfa_diagnostics``, ``views.restrict_to_64`` and
``views.transfer_integral_error`` - without modifying or re-implementing any
of them.  Test ``test_e_numeric`` checks bitwise equality with
``TPFASession`` (field, tangent field, native hashes, diagnostics).

P1-256 is opened through Drop 1's ``P1Family`` with ``compute_eta=False``
(the generation check needs no eta).

``SyntheticSolver`` makes structurally consistent arrays without any PDE
solve, for byte-level tests only.  Production refuses it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from operatormark.benchmarks import darcy_spec as _spec
from operatormark.corpus import canonical
from operatormark.solvers import darcy_tpfa as _tpfa
from operatormark_numerics_v2 import views
from operatormark_numerics_v2.api import NumericsError
from operatormark_numerics_v2.p1_family import DIAGNOSTIC_KEYS as PKEYS
from operatormark_numerics_v2.p1_family import P1Family
from operatormark_numerics_v2.tpfa_family import DIAGNOSTIC_KEYS as TKEYS
from operatormark_numerics_v2.tpfa_family import NativeTPFA

from . import layout as LY

TPFA_SETTINGS_BASE = {"method": "SuperLU", "dtype": "<f8", "forcing": "unit", "boundary": "zero",
                      "factorization_reuse": "tangents"}
P1_SETTINGS = {"resolution": 256, "method": "CG", "preconditioner": "Jacobi", "initial_guess": "zero",
               "rtol": float(1e-12).hex(), "atol": float(0).hex(), "maxiter": 10000, "dtype": "<f8", "forcing": "unit",
               "boundary": "zero", "diagonal": "SW_NE", "eta": "not_computed"}


def tpfa_settings(n_ref: int) -> dict:
    return dict(TPFA_SETTINGS_BASE, resolution=n_ref)


def p1_settings_record() -> dict:
    return dict(P1_SETTINGS)


class SampleFailure(Exception):
    """A solve produced no usable arrays.  Retained as a failure record; never retried or replaced."""

    def __init__(self, stage: str, code: str, partial: dict | None = None):
        super().__init__(f"{stage}: {code}")
        self.stage, self.code, self.partial = stage, code, dict(partial or {})


@dataclass
class TangentSolution:
    du_native: np.ndarray            # [N,N]
    du_native_sha256: str
    du64: np.ndarray                 # [64,64] = R du
    backward_error: float
    r_integral_error: float


@dataclass
class P1Evidence:
    nodal_sha256: str | None         # [257,257]; None if the verifier solve failed
    projected: np.ndarray | None     # [64,64]
    projected_sha256: str | None
    diagnostics: dict | None
    passes: bool
    iterations: int | None
    failure: str | None = None       # e.g. "p1_pcg_failure": a failed P1-256 gate, recorded; the sample is kept


@dataclass
class SampleSolution:
    n_ref: int
    u_native: np.ndarray             # [N,N]
    u_native_sha256: str
    kappa_native_sha256: str
    f64: np.ndarray                  # [64,64] = R u
    r_integral_error: float
    tpfa_diagnostics: dict
    tpfa_passes: bool
    tangents: list[TangentSolution] = field(default_factory=list)
    p1: P1Evidence | None = None


CODE_LIMIT = 64


def _code(exc: Exception) -> str:
    code = exc.code if isinstance(exc, NumericsError) else "frozen_error:" + type(exc).__name__
    return str(code)[:CODE_LIMIT]


class NativeSolver:
    """T_{N_ref} forward + tangents with one factorization, and P1-256 verification."""

    kind = "native"

    def __init__(self, n_ref: int):
        self.n_ref = LY.check_n_ref(n_ref)
        self._p1 = P1Family(levels=(256,))

    def solve(self, rho: np.ndarray, directions: list[np.ndarray]) -> SampleSolution:
        n = self.n_ref
        try:
            control = _spec.validate_control(rho)
            kappa = _spec.coefficient_on_tpfa_grid(control, n)
            native = NativeTPFA(kappa)
            diag = _tpfa.compute_tpfa_diagnostics(native.system, native.u)
        except (NumericsError, ValueError, ArithmeticError) as exc:
            raise SampleFailure("tpfa_forward", _code(exc)) from None
        diagnostics = {k: (None if getattr(diag, k) is None else float(getattr(diag, k))) for k in TKEYS}
        f64 = views.restrict_to_64(native.u, n)
        solution = SampleSolution(
            n_ref=n, u_native=native.u, u_native_sha256=canonical.tensor_sha256(native.u),
            kappa_native_sha256=canonical.tensor_sha256(native.kappa), f64=f64,
            r_integral_error=views.transfer_integral_error(native.u, n, f64),
            tpfa_diagnostics=diagnostics, tpfa_passes=bool(diag.passes))
        partial = {"tpfa_passes": solution.tpfa_passes, "u_native_sha256": solution.u_native_sha256}
        for k, v in enumerate(directions):
            try:
                du, _rhs, backward = native.tangent_native(v)
            except (NumericsError, ValueError, ArithmeticError, TypeError) as exc:
                raise SampleFailure(f"tpfa_tangent_d{k}", _code(exc), partial) from None
            du64 = views.restrict_to_64(du, n)
            solution.tangents.append(TangentSolution(du, canonical.tensor_sha256(du), du64, float(backward),
                                                     views.transfer_integral_error(du, n, du64)))
        try:
            with self._p1.open(control, 256, compute_eta=False) as session:
                f = session.forward
        except (NumericsError, ValueError, ArithmeticError) as exc:
            # P1-256 is the verifier: its failure is a failed hard gate on a sample whose
            # observations exist, so the sample is stored with p1_256 = false, not dropped.
            solution.p1 = P1Evidence(None, None, None, None, False, None, failure=_code(exc))
            return solution
        solution.p1 = P1Evidence(f.native_sha256, np.ascontiguousarray(f.field64), canonical.tensor_sha256(f.field64),
                                 {k: f.diagnostics[k] for k in PKEYS}, bool(f.gates_pass), int(f.iterations))
        return solution


class SyntheticSolver:
    """No PDE: deterministic arrays with the right shapes and the exact R/view relations.

    Diagnostics are fixed small numbers that pass every gate unless a test
    overrides them.  Used only for byte-level tests (corruption, missing or
    duplicate directions, bindings, failure retention, incomplete output).
    """

    kind = "synthetic"

    def __init__(self, n_ref: int, *, fail: dict | None = None, overrides: dict | None = None):
        self.n_ref = LY.check_n_ref(n_ref)
        self.fail = dict(fail or {})            # control sha256 -> (stage, code)
        self.overrides = dict(overrides or {})  # control sha256 -> {"tpfa_passes": False, ...}

    def _rng(self, *parts) -> np.random.Generator:
        digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
        return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))

    def solve(self, rho: np.ndarray, directions: list[np.ndarray]) -> SampleSolution:
        n = self.n_ref
        key = canonical.tensor_sha256(np.ascontiguousarray(rho, dtype=np.float64))
        if key in self.fail:
            stage, code = self.fail[key]
            raise SampleFailure(stage, code, {"synthetic": True})
        kappa = _spec.coefficient_on_tpfa_grid(rho, n)
        u = np.ascontiguousarray(0.05 + 0.01 * self._rng("u", key).random((n, n)))
        f64 = views.restrict_to_64(u, n)
        diag = {k: 1e-15 for k in TKEYS}
        sol = SampleSolution(n, u, canonical.tensor_sha256(u), canonical.tensor_sha256(kappa), f64,
                             views.transfer_integral_error(u, n, f64), diag, True)
        for k, v in enumerate(directions):
            du = np.ascontiguousarray(self._rng("du", key, k, canonical.tensor_sha256(np.asarray(v, dtype=np.float64)))
                                      .standard_normal((n, n)) * 1e-3)
            du64 = views.restrict_to_64(du, n)
            sol.tangents.append(TangentSolution(du, canonical.tensor_sha256(du), du64, 1e-16,
                                                views.transfer_integral_error(du, n, du64)))
        projected = np.ascontiguousarray(f64 * np.float64(1.0 + 1e-3))
        nodal = self._rng("p", key).random((257, 257))
        sol.p1 = P1Evidence(canonical.tensor_sha256(nodal), projected, canonical.tensor_sha256(projected),
                            {k: 1e-15 for k in PKEYS}, True, 100)
        for name, value in self.overrides.get(key, {}).items():
            if name == "p1_failure":
                sol.p1 = P1Evidence(None, None, None, None, False, None, failure=value)
            else:
                setattr(sol, name, value)
        return sol
