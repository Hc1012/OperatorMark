"""Float64 cell-centred TPFA discretisation for the Branch C Darcy benchmark.

Assembly accepts already integrated cell sources.  It deliberately does not
choose a quadrature rule for general forcing functions.  Algebraic diagnostics
below establish consistency of an assembled solve; they are not evidence, by
themselves, that the discretisation is a correct PDE implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import splu


FloatArray: TypeAlias = NDArray[np.float64]
BoundaryInput: TypeAlias = (
    "BoundaryFaceValues | Real | Mapping[str, ArrayLike | Real]"
)

SIDES = ("left", "right", "bottom", "top")
ALGEBRAIC_GATE = 1.0e-10
CONSERVATION_GATE = 1.0e-10
BOUNDARY_TRANSMISSIBILITY_GATE = 100.0 * np.finfo(np.float64).eps


class TPFAError(ValueError):
    """Raised when TPFA inputs or diagnostics violate the frozen conventions."""


@dataclass(frozen=True, slots=True)
class BoundaryFaceValues:
    """Dirichlet values at physical boundary-face centres.

    Left and right arrays are indexed by ``j`` (in increasing ``y``); bottom
    and top arrays are indexed by ``i`` (in increasing ``x``).
    """

    left: FloatArray
    right: FloatArray
    bottom: FloatArray
    top: FloatArray

    @property
    def resolution(self) -> int:
        return int(self.left.shape[0])


@dataclass(frozen=True, slots=True)
class TPFATransmissibilities:
    """Integrated face transmissibilities used by an assembly."""

    interior_x: FloatArray
    interior_y: FloatArray
    left: FloatArray
    right: FloatArray
    bottom: FloatArray
    top: FloatArray


@dataclass(frozen=True, slots=True)
class TPFAAssembly:
    """A fully assembled ``A u = b`` TPFA system in C-order ``[y, x]``."""

    resolution: int
    h: float
    kappa: FloatArray
    integrated_source: FloatArray
    boundary_values: BoundaryFaceValues
    transmissibilities: TPFATransmissibilities
    matrix: sparse.csr_matrix
    rhs: FloatArray


@dataclass(frozen=True, slots=True)
class TPFAFluxes:
    """Integrated outward/sign-oriented fluxes reconstructed from a solution.

    ``interior_x[j, i]`` is oriented from cell ``(j, i)`` to ``(j, i+1)``;
    ``interior_y[j, i]`` is oriented from ``(j, i)`` to ``(j+1, i)``.
    Boundary arrays use the physical outward orientation.
    """

    interior_x: FloatArray
    interior_y: FloatArray
    left: FloatArray
    right: FloatArray
    bottom: FloatArray
    top: FloatArray


@dataclass(frozen=True, slots=True)
class TPFAOrientedInteriorFluxes:
    """Interior outward fluxes rebuilt by four separate cell traversals.

    East/west arrays share interface index ``[j, i]`` between cells ``i`` and
    ``i+1``.  North/south arrays share ``[j, i]`` between cells ``j`` and
    ``j+1``.  Opposite arrays are computed independently, not formed by
    negating one cached unique-face flux.
    """

    east: FloatArray
    west: FloatArray
    north: FloatArray
    south: FloatArray


@dataclass(frozen=True, slots=True)
class TPFADiagnostics:
    """Frozen algebraic, construction, and balance diagnostics."""

    algebraic_backward_error: float
    rhs_relative_l2_residual: float | None
    boundary_transmissibility_error: float
    local_conservation_ratios: FloatArray
    maximum_local_conservation_ratio: float
    maximum_interior_face_antisymmetry: float
    global_balance_ratio: float

    @property
    def passes(self) -> bool:
        """Whether every gated diagnostic is finite and below its frozen gate."""

        gated = (
            (self.algebraic_backward_error, ALGEBRAIC_GATE),
            (
                self.boundary_transmissibility_error,
                BOUNDARY_TRANSMISSIBILITY_GATE,
            ),
            (self.maximum_local_conservation_ratio, CONSERVATION_GATE),
            (self.maximum_interior_face_antisymmetry, CONSERVATION_GATE),
            (self.global_balance_ratio, CONSERVATION_GATE),
        )
        return all(math.isfinite(value) and value <= gate for value, gate in gated)


def _validate_resolution(resolution: object) -> int:
    if isinstance(resolution, (bool, np.bool_)) or not isinstance(
        resolution, (int, np.integer)
    ):
        raise TPFAError("resolution must be a positive integer number of cells per axis")
    value = int(resolution)
    if value <= 0:
        raise TPFAError("resolution must be a positive integer number of cells per axis")
    return value


def _float64_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int,
    shape: tuple[int, ...] | None = None,
) -> FloatArray:
    """Validate a real numeric array and make the API's float64 conversion explicit."""

    if np.ma.isMaskedArray(values):
        raise TPFAError(f"{name} must not be a masked array")
    try:
        original = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TPFAError(f"{name} must be convertible to a rectangular numeric array") from exc
    if original.dtype.kind not in "iuf":
        raise TPFAError(f"{name} must have real numeric dtype semantics")
    if original.ndim != ndim:
        raise TPFAError(f"{name} must have exactly {ndim} dimensions")
    if shape is not None and original.shape != shape:
        raise TPFAError(f"{name} shape must be exactly {shape}, got {original.shape}")
    try:
        converted = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TPFAError(f"{name} must be safely convertible to float64") from exc
    if converted.shape != original.shape:
        raise TPFAError(f"{name} conversion must not reshape the input")
    if not np.isfinite(converted).all():
        raise TPFAError(f"{name} must contain only finite values")
    return np.array(converted, dtype=np.float64, order="C", copy=True)


def _validate_kappa(kappa: ArrayLike) -> FloatArray:
    values = _float64_array(kappa, name="kappa", ndim=2)
    if values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise TPFAError("kappa must have non-empty square shape (N, N)")
    if np.any(values <= 0.0):
        raise TPFAError("kappa must be strictly positive in every cell")
    return values


def _boundary_side(values: ArrayLike | Real, resolution: int, side: str) -> FloatArray:
    if isinstance(values, (Real, np.integer, np.floating)) and not isinstance(
        values, (bool, np.bool_)
    ):
        try:
            scalar = float(values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TPFAError(f"{side} boundary value must be float64-convertible") from exc
        if not math.isfinite(scalar):
            raise TPFAError(f"{side} boundary value must be finite")
        return np.full(resolution, scalar, dtype=np.float64)
    return _float64_array(
        values,
        name=f"{side} boundary values",
        ndim=1,
        shape=(resolution,),
    )


def make_boundary_face_values(
    resolution: int,
    values: BoundaryInput = 0.0,
) -> BoundaryFaceValues:
    """Construct strict boundary-face arrays without reshaping or transposing data."""

    n = _validate_resolution(resolution)
    if isinstance(values, BoundaryFaceValues):
        sides: Mapping[str, ArrayLike | Real] = {
            "left": values.left,
            "right": values.right,
            "bottom": values.bottom,
            "top": values.top,
        }
    elif isinstance(values, Mapping):
        if set(values) != set(SIDES):
            raise TPFAError(f"boundary mapping must contain exactly {SIDES}")
        sides = values
    elif isinstance(values, (Real, np.integer, np.floating)) and not isinstance(
        values, (bool, np.bool_)
    ):
        sides = {side: values for side in SIDES}
    else:
        raise TPFAError(
            "boundary values must be a finite real scalar, BoundaryFaceValues, "
            "or a mapping of all four sides"
        )
    return BoundaryFaceValues(
        left=_boundary_side(sides["left"], n, "left"),
        right=_boundary_side(sides["right"], n, "right"),
        bottom=_boundary_side(sides["bottom"], n, "bottom"),
        top=_boundary_side(sides["top"], n, "top"),
    )


def _construct_transmissibilities(kappa: FloatArray, h: float) -> TPFATransmissibilities:
    half_h = 0.5 * h
    # On this uniform grid, |face|=h and both centre-to-face distances are
    # h/2, giving the exact harmonic-coefficient form below.
    interior_x = (
        2.0 * kappa[:, :-1] * kappa[:, 1:]
        / (kappa[:, :-1] + kappa[:, 1:])
    )
    interior_y = (
        2.0 * kappa[:-1, :] * kappa[1:, :]
        / (kappa[:-1, :] + kappa[1:, :])
    )
    left = kappa[:, 0] * h / half_h
    right = kappa[:, -1] * h / half_h
    bottom = kappa[0, :] * h / half_h
    top = kappa[-1, :] * h / half_h
    arrays = (interior_x, interior_y, left, right, bottom, top)
    if any(not np.isfinite(array).all() or np.any(array <= 0.0) for array in arrays):
        raise TPFAError("face transmissibilities must be finite and strictly positive")
    return TPFATransmissibilities(
        interior_x=np.ascontiguousarray(interior_x, dtype=np.float64),
        interior_y=np.ascontiguousarray(interior_y, dtype=np.float64),
        left=np.ascontiguousarray(left, dtype=np.float64),
        right=np.ascontiguousarray(right, dtype=np.float64),
        bottom=np.ascontiguousarray(bottom, dtype=np.float64),
        top=np.ascontiguousarray(top, dtype=np.float64),
    )


def assemble_tpfa(
    kappa: ArrayLike,
    integrated_source: ArrayLike,
    boundary_values: BoundaryInput = 0.0,
) -> TPFAAssembly:
    """Assemble the SPD TPFA system using integrated cell sources.

    Unknown ``p = j*N + i`` represents ``u[j, i]``.  Interior off-diagonals
    are negative transmissibilities; each boundary face contributes its
    transmissibility to the diagonal and ``T*g`` to the right-hand side.
    """

    coefficient = _validate_kappa(kappa)
    n = _validate_resolution(coefficient.shape[0])
    source = _float64_array(
        integrated_source,
        name="integrated_source",
        ndim=2,
        shape=(n, n),
    )
    boundary = make_boundary_face_values(n, boundary_values)
    h = 1.0 / n
    transmissibilities = _construct_transmissibilities(coefficient, h)

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    rhs = source.reshape(-1, order="C").copy()

    for j in range(n):
        for i in range(n):
            cell = j * n + i
            diagonal = 0.0
            if i > 0:
                value = float(transmissibilities.interior_x[j, i - 1])
                diagonal += value
                rows.append(cell)
                columns.append(cell - 1)
                data.append(-value)
            else:
                value = float(transmissibilities.left[j])
                diagonal += value
                rhs[cell] += value * boundary.left[j]
            if i + 1 < n:
                value = float(transmissibilities.interior_x[j, i])
                diagonal += value
                rows.append(cell)
                columns.append(cell + 1)
                data.append(-value)
            else:
                value = float(transmissibilities.right[j])
                diagonal += value
                rhs[cell] += value * boundary.right[j]
            if j > 0:
                value = float(transmissibilities.interior_y[j - 1, i])
                diagonal += value
                rows.append(cell)
                columns.append(cell - n)
                data.append(-value)
            else:
                value = float(transmissibilities.bottom[i])
                diagonal += value
                rhs[cell] += value * boundary.bottom[i]
            if j + 1 < n:
                value = float(transmissibilities.interior_y[j, i])
                diagonal += value
                rows.append(cell)
                columns.append(cell + n)
                data.append(-value)
            else:
                value = float(transmissibilities.top[i])
                diagonal += value
                rhs[cell] += value * boundary.top[i]
            rows.append(cell)
            columns.append(cell)
            data.append(diagonal)

    matrix = sparse.coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, columns)),
        shape=(n * n, n * n),
        dtype=np.float64,
    ).tocsr()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all() or not np.isfinite(rhs).all():
        raise TPFAError("assembled matrix and right-hand side must be finite")
    return TPFAAssembly(
        resolution=n,
        h=h,
        kappa=coefficient,
        integrated_source=source,
        boundary_values=boundary,
        transmissibilities=transmissibilities,
        matrix=matrix,
        rhs=np.ascontiguousarray(rhs, dtype=np.float64),
    )


def _validate_stored_array(
    values: object,
    name: str,
    shape: tuple[int, ...],
    *,
    positive: bool = False,
) -> None:
    """Validate an already-realized internal array without coercing it."""

    if (
        not isinstance(values, np.ndarray)
        or np.ma.isMaskedArray(values)
        or values.shape != shape
        or values.dtype != np.dtype(np.float64)
    ):
        raise TPFAError(f"{name} must be an unmasked float64 array of shape {shape}")
    if not values.flags.c_contiguous:
        raise TPFAError(f"{name} must retain C-order storage")
    if not np.isfinite(values).all():
        raise TPFAError(f"{name} must contain only finite values")
    if positive and np.any(values <= 0.0):
        raise TPFAError(f"{name} must contain only strictly positive values")


def _validate_assembly(system: TPFAAssembly) -> None:
    if not isinstance(system, TPFAAssembly):
        raise TypeError("system must be a TPFAAssembly")
    n = _validate_resolution(system.resolution)
    try:
        h = float(system.h)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TPFAError("system h must be finite and consistent with its resolution") from exc
    if not math.isfinite(h) or h != 1.0 / n:
        raise TPFAError("system h is inconsistent with its resolution")
    _validate_stored_array(system.kappa, "system kappa", (n, n), positive=True)
    _validate_stored_array(
        system.integrated_source,
        "system integrated_source",
        (n, n),
    )
    if not isinstance(system.boundary_values, BoundaryFaceValues):
        raise TPFAError("system boundary_values must be BoundaryFaceValues")
    for side in SIDES:
        _validate_stored_array(
            getattr(system.boundary_values, side),
            f"system {side} boundary values",
            (n,),
        )
    if not isinstance(system.transmissibilities, TPFATransmissibilities):
        raise TPFAError("system transmissibilities must be TPFATransmissibilities")
    transmission_shapes = {
        "interior_x": (n, n - 1),
        "interior_y": (n - 1, n),
        "left": (n,),
        "right": (n,),
        "bottom": (n,),
        "top": (n,),
    }
    for name, shape in transmission_shapes.items():
        _validate_stored_array(
            getattr(system.transmissibilities, name),
            f"system {name} transmissibilities",
            shape,
            positive=True,
        )
    if not sparse.issparse(system.matrix) or system.matrix.ndim != 2:
        raise TPFAError("system matrix must be a SciPy sparse matrix")
    if system.matrix.shape != (n * n, n * n):
        raise TPFAError("system matrix dimensions are inconsistent with its resolution")
    if system.matrix.dtype != np.dtype(np.float64):
        raise TPFAError("system matrix and rhs must retain float64 dtype")
    _validate_stored_array(system.rhs, "system rhs", (n * n,))
    if system.rhs.dtype != np.dtype(np.float64):
        raise TPFAError("system matrix and rhs must retain float64 dtype")
    if not np.isfinite(system.matrix.data).all() or not np.isfinite(system.rhs).all():
        raise TPFAError("system matrix and rhs must contain only finite values")


def solve_tpfa(system: TPFAAssembly) -> FloatArray:
    """Solve an assembled TPFA system through SciPy's SuperLU interface."""

    _validate_assembly(system)
    try:
        factorization = splu(system.matrix.tocsc())
        flat_solution = factorization.solve(system.rhs)
    except RuntimeError as exc:
        raise TPFAError(f"SuperLU failed to solve the TPFA system: {exc}") from exc
    if flat_solution.shape != system.rhs.shape or flat_solution.dtype != np.dtype(np.float64):
        raise TPFAError("SuperLU returned an invalid solution shape or dtype")
    if not np.isfinite(flat_solution).all():
        raise TPFAError("SuperLU returned a non-finite solution")
    return np.ascontiguousarray(flat_solution.reshape((system.resolution, system.resolution)))


def assemble_zero_dirichlet_benchmark(kappa: ArrayLike) -> TPFAAssembly:
    """Assemble the Branch C operation ``f=1``, ``g=0`` for a kappa grid."""

    coefficient = _validate_kappa(kappa)
    n = coefficient.shape[0]
    source = np.full((n, n), (1.0 / n) ** 2, dtype=np.float64)
    return assemble_tpfa(coefficient, source, 0.0)


def solve_zero_dirichlet_benchmark(kappa: ArrayLike) -> tuple[TPFAAssembly, FloatArray]:
    """Assemble and solve the zero-Dirichlet, unit-forcing benchmark operation."""

    system = assemble_zero_dirichlet_benchmark(kappa)
    return system, solve_tpfa(system)


def _solution_array(system: TPFAAssembly, solution: ArrayLike) -> FloatArray:
    _validate_assembly(system)
    return _float64_array(
        solution,
        name="solution",
        ndim=2,
        shape=(system.resolution, system.resolution),
    )


def reconstruct_tpfa_fluxes(system: TPFAAssembly, solution: ArrayLike) -> TPFAFluxes:
    """Independently reconstruct integrated face fluxes from physical fields."""

    values = _solution_array(system, solution)
    kappa = system.kappa
    h = system.h
    half_h = 0.5 * h

    # These transmissibilities are intentionally recomputed from kappa instead
    # of copied from the assembly record.
    tx = 2.0 * kappa[:, :-1] * kappa[:, 1:] / (kappa[:, :-1] + kappa[:, 1:])
    ty = 2.0 * kappa[:-1, :] * kappa[1:, :] / (kappa[:-1, :] + kappa[1:, :])
    interior_x = tx * (values[:, :-1] - values[:, 1:])
    interior_y = ty * (values[:-1, :] - values[1:, :])
    boundary = system.boundary_values
    left = (kappa[:, 0] * h / half_h) * (values[:, 0] - boundary.left)
    right = (kappa[:, -1] * h / half_h) * (values[:, -1] - boundary.right)
    bottom = (kappa[0, :] * h / half_h) * (values[0, :] - boundary.bottom)
    top = (kappa[-1, :] * h / half_h) * (values[-1, :] - boundary.top)
    arrays = (interior_x, interior_y, left, right, bottom, top)
    if any(not np.isfinite(array).all() for array in arrays):
        raise TPFAError("reconstructed face fluxes must be finite")
    return TPFAFluxes(
        interior_x=np.ascontiguousarray(interior_x),
        interior_y=np.ascontiguousarray(interior_y),
        left=np.ascontiguousarray(left),
        right=np.ascontiguousarray(right),
        bottom=np.ascontiguousarray(bottom),
        top=np.ascontiguousarray(top),
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    """Apply the common nonnegative ratio convention from the specification."""

    a = float(numerator)
    d = float(denominator)
    if not math.isfinite(a) or not math.isfinite(d):
        raise TPFAError("ratio operands must be finite")
    if a < 0.0 or d < 0.0:
        raise TPFAError("ratio operands must be nonnegative")
    if d > 0.0:
        value = a / d
        if not math.isfinite(value):
            raise TPFAError("a non-finite ratio blocks progression")
        return value
    if a == 0.0:
        return 0.0
    return math.inf


def algebraic_backward_error(
    matrix: sparse.spmatrix,
    solution: ArrayLike,
    rhs: ArrayLike,
) -> float:
    """Compute the normwise algebraic backward error for ``A u = b``."""

    if not sparse.issparse(matrix) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise TPFAError("matrix must be a square SciPy sparse matrix")
    if matrix.dtype != np.dtype(np.float64) or not np.isfinite(matrix.data).all():
        raise TPFAError("matrix must contain finite float64 values")
    size = matrix.shape[0]
    u = _float64_array(solution, name="flat solution", ndim=1, shape=(size,))
    b = _float64_array(rhs, name="rhs", ndim=1, shape=(size,))
    residual = matrix @ u - b
    if not np.isfinite(residual).all():
        raise TPFAError("algebraic residual must be finite")
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))
    row_sums = np.asarray(abs(matrix).sum(axis=1), dtype=np.float64).reshape(-1)
    matrix_norm = float(np.max(row_sums))
    solution_norm = float(np.linalg.norm(u, ord=np.inf))
    rhs_norm = float(np.linalg.norm(b, ord=np.inf))
    return safe_ratio(residual_norm, matrix_norm * solution_norm + rhs_norm)


def rhs_relative_l2_residual(
    matrix: sparse.spmatrix,
    solution: ArrayLike,
    rhs: ArrayLike,
) -> float | None:
    """Return the diagnostic-only RHS-relative L2 residual when ``||b||_2>0``."""

    if not sparse.issparse(matrix) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise TPFAError("matrix must be a square SciPy sparse matrix")
    if matrix.dtype != np.dtype(np.float64) or not np.isfinite(matrix.data).all():
        raise TPFAError("matrix must contain finite float64 values")
    size = matrix.shape[0]
    u = _float64_array(solution, name="flat solution", ndim=1, shape=(size,))
    b = _float64_array(rhs, name="rhs", ndim=1, shape=(size,))
    residual = matrix @ u - b
    if not np.isfinite(residual).all():
        raise TPFAError("algebraic residual must be finite")
    rhs_norm = float(np.linalg.norm(b, ord=2))
    if rhs_norm == 0.0:
        return None
    return safe_ratio(float(np.linalg.norm(residual, ord=2)), rhs_norm)


def boundary_transmissibility_error(system: TPFAAssembly) -> float:
    """Independently check every implemented half-cell boundary transmissibility."""

    _validate_assembly(system)
    h = system.h
    half_h = 0.5 * h
    implemented = system.transmissibilities
    comparisons = (
        (implemented.left, system.kappa[:, 0] * h / half_h),
        (implemented.right, system.kappa[:, -1] * h / half_h),
        (implemented.bottom, system.kappa[0, :] * h / half_h),
        (implemented.top, system.kappa[-1, :] * h / half_h),
    )
    maximum = 0.0
    for actual, expected in comparisons:
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise TPFAError("implemented boundary transmissibilities are malformed")
        relative = np.abs(actual - expected) / expected
        if not np.isfinite(relative).all():
            raise TPFAError("boundary transmissibility diagnostic is non-finite")
        maximum = max(maximum, float(np.max(relative)))
    return maximum


def _validate_fluxes(system: TPFAAssembly, fluxes: TPFAFluxes) -> None:
    if not isinstance(fluxes, TPFAFluxes):
        raise TypeError("fluxes must be TPFAFluxes")
    n = system.resolution
    expected_shapes = (
        (fluxes.interior_x, (n, n - 1)),
        (fluxes.interior_y, (n - 1, n)),
        (fluxes.left, (n,)),
        (fluxes.right, (n,)),
        (fluxes.bottom, (n,)),
        (fluxes.top, (n,)),
    )
    for values, shape in expected_shapes:
        _validate_stored_array(values, "reconstructed fluxes", shape)


def local_conservation_ratios(system: TPFAAssembly, fluxes: TPFAFluxes) -> FloatArray:
    """Return every cell's integrated source/outward-flux balance ratio."""

    _validate_assembly(system)
    _validate_fluxes(system, fluxes)
    balance = np.zeros_like(system.integrated_source)
    denominator = np.abs(system.integrated_source).copy()

    balance[:, :-1] += fluxes.interior_x
    balance[:, 1:] -= fluxes.interior_x
    denominator[:, :-1] += np.abs(fluxes.interior_x)
    denominator[:, 1:] += np.abs(fluxes.interior_x)
    balance[:-1, :] += fluxes.interior_y
    balance[1:, :] -= fluxes.interior_y
    denominator[:-1, :] += np.abs(fluxes.interior_y)
    denominator[1:, :] += np.abs(fluxes.interior_y)

    balance[:, 0] += fluxes.left
    balance[:, -1] += fluxes.right
    balance[0, :] += fluxes.bottom
    balance[-1, :] += fluxes.top
    denominator[:, 0] += np.abs(fluxes.left)
    denominator[:, -1] += np.abs(fluxes.right)
    denominator[0, :] += np.abs(fluxes.bottom)
    denominator[-1, :] += np.abs(fluxes.top)

    numerator = np.abs(balance - system.integrated_source)
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        raise TPFAError("local conservation operands must be finite")
    ratios = np.empty_like(numerator)
    positive = denominator > 0.0
    ratios[positive] = numerator[positive] / denominator[positive]
    ratios[~positive & (numerator == 0.0)] = 0.0
    ratios[~positive & (numerator > 0.0)] = math.inf
    return np.ascontiguousarray(ratios)


def reconstruct_oriented_interior_fluxes(
    system: TPFAAssembly,
    solution: ArrayLike,
) -> TPFAOrientedInteriorFluxes:
    """Rebuild east, west, north, and south fluxes in separate cell traversals."""

    values = _solution_array(system, solution)
    kappa = system.kappa
    h = system.h
    half_h = 0.5 * h
    n = system.resolution
    east = np.empty((n, n - 1), dtype=np.float64)
    west = np.empty((n, n - 1), dtype=np.float64)
    north = np.empty((n - 1, n), dtype=np.float64)
    south = np.empty((n - 1, n), dtype=np.float64)

    # Each direction is intentionally a separate cell-centric pass.  West and
    # south are not defined as the negatives of east and north.
    for j in range(n):
        for i in range(n - 1):
            transmissibility = h / (
                half_h / kappa[j, i] + half_h / kappa[j, i + 1]
            )
            east[j, i] = transmissibility * (values[j, i] - values[j, i + 1])
    for j in range(n):
        for i in range(1, n):
            transmissibility = h / (
                half_h / kappa[j, i] + half_h / kappa[j, i - 1]
            )
            west[j, i - 1] = transmissibility * (values[j, i] - values[j, i - 1])
    for j in range(n - 1):
        for i in range(n):
            transmissibility = h / (
                half_h / kappa[j, i] + half_h / kappa[j + 1, i]
            )
            north[j, i] = transmissibility * (values[j, i] - values[j + 1, i])
    for j in range(1, n):
        for i in range(n):
            transmissibility = h / (
                half_h / kappa[j, i] + half_h / kappa[j - 1, i]
            )
            south[j - 1, i] = transmissibility * (values[j, i] - values[j - 1, i])
    return TPFAOrientedInteriorFluxes(
        east=east,
        west=west,
        north=north,
        south=south,
    )


def interior_flux_antisymmetry(fluxes: TPFAOrientedInteriorFluxes) -> float:
    """Evaluate opposite-orientation flux pairs using the common ratio rule."""

    if not isinstance(fluxes, TPFAOrientedInteriorFluxes):
        raise TypeError("fluxes must be TPFAOrientedInteriorFluxes")
    if fluxes.east.ndim != 2 or fluxes.east.shape[0] <= 0:
        raise TPFAError("oriented interior flux arrays have invalid dimensions")
    n = fluxes.east.shape[0]
    shapes = {
        "east": (n, n - 1),
        "west": (n, n - 1),
        "north": (n - 1, n),
        "south": (n - 1, n),
    }
    for name, shape in shapes.items():
        _validate_stored_array(
            getattr(fluxes, name),
            f"oriented {name} fluxes",
            shape,
        )

    maximum = 0.0
    for forward, reverse in (
        (fluxes.east, fluxes.west),
        (fluxes.north, fluxes.south),
    ):
        numerator = np.abs(forward + reverse)
        denominator = np.abs(forward) + np.abs(reverse)
        positive = denominator > 0.0
        ratios = np.zeros_like(numerator)
        ratios[positive] = numerator[positive] / denominator[positive]
        ratios[~positive & (numerator > 0.0)] = math.inf
        if ratios.size:
            maximum = max(maximum, float(np.max(ratios)))
    return maximum


def interior_face_antisymmetry(system: TPFAAssembly, solution: ArrayLike) -> float:
    """Reconstruct then compare both cell orientations of every interior face."""

    return interior_flux_antisymmetry(
        reconstruct_oriented_interior_fluxes(system, solution)
    )


def global_balance_ratio(system: TPFAAssembly, fluxes: TPFAFluxes) -> float:
    """Compare total outward boundary flux with total integrated source."""

    _validate_assembly(system)
    _validate_fluxes(system, fluxes)
    boundary_flux = float(
        np.sum(fluxes.left)
        + np.sum(fluxes.right)
        + np.sum(fluxes.bottom)
        + np.sum(fluxes.top)
    )
    source = float(np.sum(system.integrated_source))
    denominator = float(
        np.sum(np.abs(fluxes.left))
        + np.sum(np.abs(fluxes.right))
        + np.sum(np.abs(fluxes.bottom))
        + np.sum(np.abs(fluxes.top))
        + np.sum(np.abs(system.integrated_source))
    )
    return safe_ratio(abs(boundary_flux - source), denominator)


def compute_tpfa_diagnostics(
    system: TPFAAssembly,
    solution: ArrayLike,
) -> TPFADiagnostics:
    """Compute all required TPFA diagnostics from independently rebuilt fluxes."""

    values = _solution_array(system, solution)
    flat = values.reshape(-1, order="C")
    fluxes = reconstruct_tpfa_fluxes(system, values)
    local = local_conservation_ratios(system, fluxes)
    maximum_local = float(np.max(local))
    return TPFADiagnostics(
        algebraic_backward_error=algebraic_backward_error(system.matrix, flat, system.rhs),
        rhs_relative_l2_residual=rhs_relative_l2_residual(system.matrix, flat, system.rhs),
        boundary_transmissibility_error=boundary_transmissibility_error(system),
        local_conservation_ratios=local,
        maximum_local_conservation_ratio=maximum_local,
        maximum_interior_face_antisymmetry=interior_face_antisymmetry(system, values),
        global_balance_ratio=global_balance_ratio(system, fluxes),
    )
