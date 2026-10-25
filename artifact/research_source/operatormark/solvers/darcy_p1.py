"""Independent conforming-P1 verifier for the Branch C v1 Darcy benchmark.

The implementation in this module owns its mesh, finite-element assembly,
Dirichlet elimination, Jacobi-preconditioned CG path, cell-average projection,
and diagnostics.  It intentionally does not import or reuse the TPFA solver.

The frozen numerical choices are made before convergence results are examined:

* CG starts from zero, uses Jacobi inverse-diagonal preconditioning,
  ``rtol=1e-12``, ``atol=0``, and at most 10,000 iterations.
* General element assembly uses an order-8 tensor Gauss--Legendre rule mapped
  to the reference triangle by the Duffy transform.
* Physical L2/H1 measurements use the same construction at order 10; the
  preregistered quadrature-stability audit repeats them at order 12.
* Interface line-L2 measurements use order-8 Gauss--Legendre quadrature.
* Unit benchmark forcing and aligned piecewise-constant benchmark
  coefficients are integrated by closed-form P1 element formulas, not by
  numerical quadrature.

Algebraic and reaction checks establish properties of the assembled equations
and their reconstruction.  They do not independently prove PDE correctness.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import cg

from operatormark.benchmarks.darcy_spec import (
    CANONICAL_DTYPE,
    COEFFICIENT_BOUNDS,
    CONTROL_SHAPE,
    control_to_coefficient,
)


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
FieldFunction: TypeAlias = Callable[[FloatArray, FloatArray], ArrayLike]
ScalarFieldInput: TypeAlias = Real | FieldFunction
BoundaryInput: TypeAlias = Real | FieldFunction | ArrayLike

P1_ALGEBRAIC_GATE = 1.0e-10
P1_REACTION_GATE = 1.0e-10
P1_BOUNDARY_RECONSTRUCTION_GATE = 100.0 * np.finfo(np.float64).eps

P1_CG_RELATIVE_TOLERANCE = 1.0e-12
P1_CG_ABSOLUTE_TOLERANCE = 0.0
P1_CG_MAXIMUM_ITERATIONS = 10_000
P1_CG_INITIAL_GUESS = "zero"

P1_ASSEMBLY_QUADRATURE_ORDER = 8
P1_ERROR_QUADRATURE_ORDER = 10
P1_AUDIT_QUADRATURE_ORDER = 12
P1_EDGE_QUADRATURE_ORDER = 8
P1_QUADRATURE_BATCH_SIZE = 4096

P1_PROJECTION_TARGET = 64
P1_PROJECTION_RESOLUTIONS = (128, 256)


class P1Error(ValueError):
    """Raised when P1 inputs, solves, or diagnostics violate frozen rules."""


@dataclass(frozen=True, slots=True)
class P1Mesh:
    """Uniform P1 mesh with C-order node number ``p=j*(N+1)+i``.

    ``coordinates[p]`` is ``(x_i, y_j)``.  Triangles are counter-clockwise and
    occur in pairs ``(SW, SE, NE)``, ``(SW, NE, NW)`` for each square traversed
    in C-order ``[j, i]``.
    """

    resolution: int
    h: float
    coordinates: FloatArray
    triangles: IntArray
    boundary_nodes: IntArray
    interior_nodes: IntArray

    @property
    def node_count(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def element_count(self) -> int:
        return int(self.triangles.shape[0])


@dataclass(frozen=True, slots=True)
class P1Assembly:
    """Full and strongly eliminated P1 equations.

    ``full_stiffness`` and ``full_load`` are the original assembled objects;
    they are never row-replaced or otherwise modified during elimination.
    ``boundary_values`` follows ``mesh.boundary_nodes`` ordering.
    """

    mesh: P1Mesh
    full_stiffness: sparse.csr_matrix
    full_load: FloatArray
    integrated_source: float
    integrated_absolute_source: float
    boundary_values: FloatArray
    reduced_matrix: sparse.csr_matrix
    reduced_rhs: FloatArray
    integration_description: str


@dataclass(frozen=True, slots=True)
class P1SolveResult:
    """A reconstructed nodal ``solution[j,i]`` and deterministic CG metadata."""

    solution: FloatArray
    iterations: int


@dataclass(frozen=True, slots=True)
class P1Diagnostics:
    """Frozen reduced-system, reconstruction, and boundary-reaction checks."""

    algebraic_backward_error: float
    rhs_relative_l2_residual: float | None
    boundary_reconstruction_error: float
    boundary_reactions: FloatArray
    outward_boundary_flux: float
    integrated_source: float
    reaction_denominator: float
    reaction_balance_ratio: float

    @property
    def passes(self) -> bool:
        gated = (
            (self.algebraic_backward_error, P1_ALGEBRAIC_GATE),
            (
                self.boundary_reconstruction_error,
                P1_BOUNDARY_RECONSTRUCTION_GATE,
            ),
            (self.reaction_balance_ratio, P1_REACTION_GATE),
        )
        return all(math.isfinite(value) and value <= gate for value, gate in gated)


@dataclass(frozen=True, slots=True)
class P1PhysicalErrors:
    """Physical L2 norm and H1 seminorm of a nodal P1 error field."""

    l2: float
    h1_seminorm: float


@dataclass(frozen=True, slots=True)
class P1InterfaceFluxErrors:
    """Relative line-L2 errors of left and right P1 interface traces."""

    left: float
    right: float

    @property
    def maximum(self) -> float:
        return max(self.left, self.right)


def _validate_resolution(resolution: object, *, aligned: bool = False) -> int:
    if isinstance(resolution, (bool, np.bool_)) or not isinstance(
        resolution, (int, np.integer)
    ):
        raise P1Error("resolution must be a positive integer number of squares per axis")
    value = int(resolution)
    if value <= 0:
        raise P1Error("resolution must be a positive integer number of squares per axis")
    if aligned and value % CONTROL_SHAPE[0] != 0:
        raise P1Error("resolution must be divisible by 8 for macro-cell alignment")
    return value


def _validate_quadrature_order(order: object, *, name: str = "quadrature order") -> int:
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise P1Error(f"{name} must be a positive integer")
    value = int(order)
    if value <= 0:
        raise P1Error(f"{name} must be a positive integer")
    return value


def _stored_float64_array(values: object, *, name: str, shape: tuple[int, ...]) -> None:
    if (
        not isinstance(values, np.ndarray)
        or np.ma.isMaskedArray(values)
        or values.shape != shape
        or values.dtype != CANONICAL_DTYPE
        or not values.flags.c_contiguous
    ):
        raise P1Error(f"{name} must be a C-contiguous float64 array of shape {shape}")
    if not np.isfinite(values).all():
        raise P1Error(f"{name} must contain only finite values")


def _float64_array(
    values: ArrayLike,
    *,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    if np.ma.isMaskedArray(values):
        raise P1Error(f"{name} must not be a masked array")
    try:
        original = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise P1Error(f"{name} must be convertible to a rectangular numeric array") from exc
    if original.shape != shape:
        raise P1Error(f"{name} shape must be exactly {shape}, got {original.shape}")
    if original.dtype.kind not in "iuf":
        raise P1Error(f"{name} must have real numeric dtype semantics")
    try:
        converted = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1Error(f"{name} must be safely convertible to float64") from exc
    if not np.isfinite(converted).all():
        raise P1Error(f"{name} must contain only finite values")
    return np.array(converted, dtype=np.float64, order="C", copy=True)


def _finite_scalar(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise P1Error(f"{name} must be a finite real scalar")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1Error(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(converted):
        raise P1Error(f"{name} must be a finite real scalar")
    if positive and converted <= 0.0:
        raise P1Error(f"{name} must be strictly positive")
    return converted


def build_p1_mesh(resolution: int) -> P1Mesh:
    """Construct the frozen uniform mesh without any TPFA grid utility."""

    n = _validate_resolution(resolution)
    width = n + 1
    axis = np.arange(width, dtype=np.float64) / np.float64(n)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    coordinates = np.column_stack((x.reshape(-1), y.reshape(-1))).astype(
        np.float64, copy=False
    )

    j, i = np.meshgrid(
        np.arange(n, dtype=np.int64),
        np.arange(n, dtype=np.int64),
        indexing="ij",
    )
    southwest = (j * width + i).reshape(-1, order="C")
    southeast = southwest + 1
    northwest = southwest + width
    northeast = northwest + 1
    triangles = np.empty((2 * n * n, 3), dtype=np.int64)
    triangles[0::2] = np.column_stack((southwest, southeast, northeast))
    triangles[1::2] = np.column_stack((southwest, northeast, northwest))

    boundary_mask = np.zeros((width, width), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_nodes = np.flatnonzero(boundary_mask.reshape(-1, order="C")).astype(
        np.int64, copy=False
    )
    interior_nodes = np.flatnonzero((~boundary_mask).reshape(-1, order="C")).astype(
        np.int64, copy=False
    )

    mesh = P1Mesh(
        resolution=n,
        h=1.0 / n,
        coordinates=np.ascontiguousarray(coordinates, dtype=np.float64),
        triangles=np.ascontiguousarray(triangles, dtype=np.int64),
        boundary_nodes=np.ascontiguousarray(boundary_nodes, dtype=np.int64),
        interior_nodes=np.ascontiguousarray(interior_nodes, dtype=np.int64),
    )
    _validate_mesh(mesh)
    return mesh


def _validate_mesh(mesh: P1Mesh) -> None:
    if not isinstance(mesh, P1Mesh):
        raise TypeError("mesh must be a P1Mesh")
    n = _validate_resolution(mesh.resolution)
    if not math.isfinite(mesh.h) or mesh.h != 1.0 / n:
        raise P1Error("mesh h is inconsistent with its resolution")
    node_count = (n + 1) ** 2
    _stored_float64_array(mesh.coordinates, name="mesh coordinates", shape=(node_count, 2))
    if (
        not isinstance(mesh.triangles, np.ndarray)
        or mesh.triangles.shape != (2 * n * n, 3)
        or mesh.triangles.dtype != np.dtype(np.int64)
        or not mesh.triangles.flags.c_contiguous
    ):
        raise P1Error("mesh triangles must be a C-contiguous int64 connectivity array")
    if np.any(mesh.triangles < 0) or np.any(mesh.triangles >= node_count):
        raise P1Error("mesh triangles contain invalid node indices")
    width = n + 1
    axis = np.arange(width, dtype=np.float64) / np.float64(n)
    expected_x, expected_y = np.meshgrid(axis, axis, indexing="xy")
    expected_coordinates = np.column_stack(
        (expected_x.reshape(-1), expected_y.reshape(-1))
    )
    if not np.array_equal(mesh.coordinates, expected_coordinates):
        raise P1Error("mesh coordinates do not follow the frozen [j,i] nodal grid")

    square_j, square_i = np.meshgrid(
        np.arange(n, dtype=np.int64),
        np.arange(n, dtype=np.int64),
        indexing="ij",
    )
    southwest = (square_j * width + square_i).reshape(-1, order="C")
    expected_triangles = np.empty((2 * n * n, 3), dtype=np.int64)
    expected_triangles[0::2] = np.column_stack(
        (southwest, southwest + 1, southwest + width + 1)
    )
    expected_triangles[1::2] = np.column_stack(
        (southwest, southwest + width + 1, southwest + width)
    )
    if not np.array_equal(mesh.triangles, expected_triangles):
        raise P1Error("mesh connectivity does not follow the frozen SW-NE triangles")

    expected_boundary_mask = np.zeros((width, width), dtype=bool)
    expected_boundary_mask[0, :] = True
    expected_boundary_mask[-1, :] = True
    expected_boundary_mask[:, 0] = True
    expected_boundary_mask[:, -1] = True
    expected_boundary_nodes = np.flatnonzero(
        expected_boundary_mask.reshape(-1, order="C")
    ).astype(np.int64, copy=False)
    expected_interior_nodes = np.flatnonzero(
        (~expected_boundary_mask).reshape(-1, order="C")
    ).astype(np.int64, copy=False)
    expected_boundary = expected_boundary_nodes.size
    expected_interior = expected_interior_nodes.size
    for values, name, size in (
        (mesh.boundary_nodes, "boundary", expected_boundary),
        (mesh.interior_nodes, "interior", expected_interior),
    ):
        if (
            not isinstance(values, np.ndarray)
            or values.shape != (size,)
            or values.dtype != np.dtype(np.int64)
            or not values.flags.c_contiguous
        ):
            raise P1Error(f"mesh {name} nodes have invalid representation")
    partition = np.sort(np.concatenate((mesh.boundary_nodes, mesh.interior_nodes)))
    if not np.array_equal(partition, np.arange(node_count, dtype=np.int64)):
        raise P1Error("mesh boundary/interior nodes must exactly partition all nodes")
    if not np.array_equal(mesh.boundary_nodes, expected_boundary_nodes) or not np.array_equal(
        mesh.interior_nodes, expected_interior_nodes
    ):
        raise P1Error("mesh boundary/interior partition must match geometric boundary nodes")


def _triangle_geometry(vertices: ArrayLike) -> tuple[FloatArray, float, FloatArray]:
    points = _float64_array(vertices, name="triangle vertices", shape=(3, 2))
    determinant = float(
        (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1])
        - (points[2, 0] - points[0, 0]) * (points[1, 1] - points[0, 1])
    )
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise P1Error("triangle vertices must be finite, nondegenerate, and counter-clockwise")
    gradients = np.array(
        [
            [points[1, 1] - points[2, 1], points[2, 0] - points[1, 0]],
            [points[2, 1] - points[0, 1], points[0, 0] - points[2, 0]],
            [points[0, 1] - points[1, 1], points[1, 0] - points[0, 0]],
        ],
        dtype=np.float64,
    ) / determinant
    return points, 0.5 * determinant, gradients


def constant_triangle_stiffness(vertices: ArrayLike, coefficient: Real) -> FloatArray:
    """Closed-form local ``int_T k grad(phi_a).grad(phi_b)`` for constant k."""

    _, area, gradients = _triangle_geometry(vertices)
    kappa = _finite_scalar(coefficient, name="coefficient", positive=True)
    matrix = kappa * area * (gradients @ gradients.T)
    if not np.isfinite(matrix).all():
        raise P1Error("local stiffness must contain only finite values")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def constant_triangle_load(vertices: ArrayLike, source: Real) -> FloatArray:
    """Closed-form local ``int_T f phi_a`` for a constant source."""

    _, area, _ = _triangle_geometry(vertices)
    value = _finite_scalar(source, name="source")
    load = np.full(3, value * area / 3.0, dtype=np.float64)
    if not np.isfinite(load).all():
        raise P1Error("local load must contain only finite values")
    return load


@lru_cache(maxsize=None)
def _reference_triangle_quadrature(order: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    checked = _validate_quadrature_order(order)
    nodes, weights = np.polynomial.legendre.leggauss(checked)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    a, b = np.meshgrid(unit_nodes, unit_nodes, indexing="ij")
    wa, wb = np.meshgrid(unit_weights, unit_weights, indexing="ij")
    xi = a.reshape(-1)
    eta = ((1.0 - a) * b).reshape(-1)
    quadrature_weights = (wa * wb * (1.0 - a)).reshape(-1)
    basis = np.column_stack((1.0 - xi - eta, xi, eta))
    if not math.isclose(float(np.sum(quadrature_weights)), 0.5, rel_tol=0.0, abs_tol=2e-15):
        raise P1Error("reference-triangle quadrature failed its area invariant")
    return (
        np.ascontiguousarray(xi, dtype=np.float64),
        np.ascontiguousarray(eta, dtype=np.float64),
        np.ascontiguousarray(quadrature_weights, dtype=np.float64),
    )


def duffy_triangle_quadrature(order: int) -> tuple[FloatArray, FloatArray]:
    """Return independent copies of frozen reference points and weights."""

    xi, eta, weights = _reference_triangle_quadrature(order)
    return np.column_stack((xi, eta)), weights.copy(order="C")


def _evaluate_field(
    field: ScalarFieldInput,
    x: FloatArray,
    y: FloatArray,
    *,
    name: str,
    positive: bool,
) -> FloatArray:
    if isinstance(field, (Real, np.integer, np.floating)) and not isinstance(
        field, (bool, np.bool_)
    ):
        scalar = _finite_scalar(field, name=name, positive=positive)
        return np.full(x.shape, scalar, dtype=np.float64)
    if not callable(field):
        raise P1Error(f"{name} must be a finite real scalar or a callable f(x, y)")
    try:
        raw = field(x, y)
    except Exception as exc:
        raise P1Error(f"{name} callable failed during quadrature evaluation") from exc
    if np.ma.isMaskedArray(raw):
        raise P1Error(f"{name} callable must not return a masked array")
    array = np.asarray(raw)
    if array.ndim == 0:
        array = np.full(x.shape, array.item())
    if array.shape != x.shape:
        raise P1Error(f"{name} callable must return a scalar or shape {x.shape}")
    if array.dtype.kind not in "iuf":
        raise P1Error(f"{name} callable must return real numeric values")
    try:
        values = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise P1Error(f"{name} values must be safely convertible to float64") from exc
    if not np.isfinite(values).all():
        raise P1Error(f"{name} values must all be finite")
    if positive and np.any(values <= 0.0):
        raise P1Error(f"{name} values must all be strictly positive")
    return np.ascontiguousarray(values, dtype=np.float64)


def _mesh_triangle_geometry(mesh: P1Mesh) -> tuple[FloatArray, FloatArray, FloatArray]:
    vertices = mesh.coordinates[mesh.triangles]
    x0 = vertices[:, 0, 0]
    y0 = vertices[:, 0, 1]
    x1 = vertices[:, 1, 0]
    y1 = vertices[:, 1, 1]
    x2 = vertices[:, 2, 0]
    y2 = vertices[:, 2, 1]
    determinants = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if not np.isfinite(determinants).all() or np.any(determinants <= 0.0):
        raise P1Error("every P1 triangle must be finite and counter-clockwise")
    gradients = np.empty((mesh.element_count, 3, 2), dtype=np.float64)
    gradients[:, 0, 0] = y1 - y2
    gradients[:, 0, 1] = x2 - x1
    gradients[:, 1, 0] = y2 - y0
    gradients[:, 1, 1] = x0 - x2
    gradients[:, 2, 0] = y0 - y1
    gradients[:, 2, 1] = x1 - x0
    gradients /= determinants[:, None, None]
    return (
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(determinants, dtype=np.float64),
        np.ascontiguousarray(gradients, dtype=np.float64),
    )


def _evaluate_boundary_values(mesh: P1Mesh, boundary: BoundaryInput) -> FloatArray:
    nodes = mesh.boundary_nodes
    x = mesh.coordinates[nodes, 0]
    y = mesh.coordinates[nodes, 1]
    if isinstance(boundary, (Real, np.integer, np.floating)) and not isinstance(
        boundary, (bool, np.bool_)
    ):
        scalar = _finite_scalar(boundary, name="boundary data")
        return np.full(nodes.shape, scalar, dtype=np.float64)
    if callable(boundary):
        try:
            raw = boundary(x, y)
        except Exception as exc:
            raise P1Error("boundary callable failed at P1 boundary nodes") from exc
        if np.ma.isMaskedArray(raw):
            raise P1Error("boundary callable must not return a masked array")
        array = np.asarray(raw)
        if array.ndim == 0:
            array = np.full(nodes.shape, array.item())
        if array.shape != nodes.shape:
            raise P1Error(
                f"boundary callable must return a scalar or shape {nodes.shape}"
            )
        return _float64_array(array, name="boundary values", shape=nodes.shape)

    if np.ma.isMaskedArray(boundary):
        raise P1Error("boundary data must not be a masked array")
    try:
        array = np.asarray(boundary)
    except (TypeError, ValueError) as exc:
        raise P1Error("boundary data must be a scalar, callable, or nodal array") from exc
    width = mesh.resolution + 1
    if array.shape == (width, width):
        nodal = _float64_array(
            array,
            name="nodal boundary data",
            shape=(width, width),
        )
        return np.ascontiguousarray(nodal.reshape(-1, order="C")[nodes])
    if array.shape == nodes.shape:
        return _float64_array(array, name="boundary values", shape=nodes.shape)
    raise P1Error(
        "boundary array must have shape (N+1, N+1) or one value per boundary node"
    )


def _finalize_assembly(
    mesh: P1Mesh,
    gradients: FloatArray,
    coefficient_integrals: FloatArray,
    local_loads: FloatArray,
    integrated_source: float,
    integrated_absolute_source: float,
    boundary_values: BoundaryInput,
    *,
    integration_description: str,
) -> P1Assembly:
    element_count = mesh.element_count
    _stored_float64_array(
        gradients,
        name="element gradients",
        shape=(element_count, 3, 2),
    )
    _stored_float64_array(
        coefficient_integrals,
        name="element coefficient integrals",
        shape=(element_count,),
    )
    _stored_float64_array(
        local_loads,
        name="element load vectors",
        shape=(element_count, 3),
    )
    if np.any(coefficient_integrals <= 0.0):
        raise P1Error("element coefficient integrals must be strictly positive")
    source_integral = _finite_scalar(integrated_source, name="integrated source")
    absolute_source_integral = _finite_scalar(
        integrated_absolute_source,
        name="integrated absolute source",
    )
    if absolute_source_integral < 0.0:
        raise P1Error("integrated absolute source must be nonnegative")

    gradient_products = np.einsum(
        "tai,tbi->tab", gradients, gradients, optimize=True
    )
    local_stiffness = coefficient_integrals[:, None, None] * gradient_products
    if not np.isfinite(local_stiffness).all():
        raise P1Error("local stiffness matrices must contain only finite values")

    connectivity = mesh.triangles
    rows = np.repeat(connectivity, 3, axis=1).reshape(-1)
    columns = np.tile(connectivity, (1, 3)).reshape(-1)
    full_stiffness = sparse.coo_matrix(
        (local_stiffness.reshape(-1), (rows, columns)),
        shape=(mesh.node_count, mesh.node_count),
        dtype=np.float64,
    ).tocsr()
    full_stiffness.sum_duplicates()
    full_stiffness.eliminate_zeros()
    full_stiffness.sort_indices()
    full_load = np.bincount(
        connectivity.reshape(-1),
        weights=local_loads.reshape(-1),
        minlength=mesh.node_count,
    ).astype(np.float64, copy=False)
    full_load = np.ascontiguousarray(full_load, dtype=np.float64)
    if not np.isfinite(full_stiffness.data).all() or not np.isfinite(full_load).all():
        raise P1Error("assembled full stiffness and load must be finite")

    boundary = _evaluate_boundary_values(mesh, boundary_values)
    interior = mesh.interior_nodes
    boundary_nodes = mesh.boundary_nodes
    reduced_matrix = full_stiffness[interior][:, interior].tocsr(copy=True)
    reduced_matrix.sort_indices()
    if interior.size:
        coupling = full_stiffness[interior][:, boundary_nodes]
        reduced_rhs = full_load[interior] - coupling @ boundary
    else:
        reduced_rhs = np.empty((0,), dtype=np.float64)
    reduced_rhs = np.ascontiguousarray(reduced_rhs, dtype=np.float64)
    if not np.isfinite(reduced_matrix.data).all() or not np.isfinite(reduced_rhs).all():
        raise P1Error("reduced P1 matrix and right-hand side must be finite")

    system = P1Assembly(
        mesh=mesh,
        full_stiffness=full_stiffness,
        full_load=full_load,
        integrated_source=source_integral,
        integrated_absolute_source=absolute_source_integral,
        boundary_values=boundary,
        reduced_matrix=reduced_matrix,
        reduced_rhs=reduced_rhs,
        integration_description=integration_description,
    )
    _validate_assembly(system)
    return system


def assemble_p1(
    resolution: int,
    coefficient: ScalarFieldInput,
    source: ScalarFieldInput,
    boundary_values: BoundaryInput = 0.0,
    *,
    quadrature_order: int = P1_ASSEMBLY_QUADRATURE_ORDER,
) -> P1Assembly:
    """Assemble ``int k grad(u).grad(v) = int f v`` independently.

    Finite scalar coefficients and sources use exact local formulas.  Otherwise
    both functions are evaluated by the documented Duffy-mapped tensor rule.
    Callable coefficients must be strictly positive at every quadrature point.
    """

    mesh = build_p1_mesh(resolution)
    vertices, determinants, gradients = _mesh_triangle_geometry(mesh)
    areas = 0.5 * determinants
    element_count = mesh.element_count

    coefficient_scalar: float | None = None
    source_scalar: float | None = None
    if isinstance(coefficient, (Real, np.integer, np.floating)) and not isinstance(
        coefficient, (bool, np.bool_)
    ):
        coefficient_scalar = _finite_scalar(
            coefficient, name="coefficient", positive=True
        )
    elif not callable(coefficient):
        raise P1Error("coefficient must be a finite positive scalar or callable k(x, y)")
    if isinstance(source, (Real, np.integer, np.floating)) and not isinstance(
        source, (bool, np.bool_)
    ):
        source_scalar = _finite_scalar(source, name="source")
    elif not callable(source):
        raise P1Error("source must be a finite real scalar or callable f(x, y)")

    coefficient_integrals = np.empty(element_count, dtype=np.float64)
    local_loads = np.empty((element_count, 3), dtype=np.float64)

    if coefficient_scalar is not None:
        coefficient_integrals[:] = coefficient_scalar * areas
    if source_scalar is not None:
        local_loads[:] = (source_scalar * areas / 3.0)[:, None]
        integrated_source = source_scalar * float(np.sum(areas))
        integrated_absolute_source = abs(source_scalar) * float(np.sum(areas))

    if coefficient_scalar is not None and source_scalar is not None:
        description = "closed-form constant coefficient and source element integration"
    else:
        order = _validate_quadrature_order(quadrature_order)
        xi, eta, reference_weights = _reference_triangle_quadrature(order)
        basis = np.column_stack((1.0 - xi - eta, xi, eta))
        source_sum = 0.0
        absolute_source_sum = 0.0
        for start in range(0, element_count, P1_QUADRATURE_BATCH_SIZE):
            stop = min(start + P1_QUADRATURE_BATCH_SIZE, element_count)
            batch_vertices = vertices[start:stop]
            v0 = batch_vertices[:, 0, :]
            edge1 = batch_vertices[:, 1, :] - v0
            edge2 = batch_vertices[:, 2, :] - v0
            x = (
                v0[:, 0, None]
                + edge1[:, 0, None] * xi[None, :]
                + edge2[:, 0, None] * eta[None, :]
            )
            y = (
                v0[:, 1, None]
                + edge1[:, 1, None] * xi[None, :]
                + edge2[:, 1, None] * eta[None, :]
            )
            weights = determinants[start:stop, None] * reference_weights[None, :]
            if coefficient_scalar is None:
                kappa = _evaluate_field(
                    coefficient,
                    x,
                    y,
                    name="coefficient",
                    positive=True,
                )
                coefficient_integrals[start:stop] = np.sum(
                    weights * kappa, axis=1, dtype=np.float64
                )
            if source_scalar is None:
                forcing = _evaluate_field(
                    source,
                    x,
                    y,
                    name="source",
                    positive=False,
                )
                weighted_source = weights * forcing
                local_loads[start:stop] = weighted_source @ basis
                source_sum += float(np.sum(weighted_source, dtype=np.float64))
                absolute_source_sum += float(
                    np.sum(weights * np.abs(forcing), dtype=np.float64)
                )
        if source_scalar is None:
            integrated_source = source_sum
            integrated_absolute_source = absolute_source_sum
        description = (
            f"Duffy-mapped tensor Gauss-Legendre triangle quadrature, order {order}"
        )

    if not np.isfinite(coefficient_integrals).all() or np.any(
        coefficient_integrals <= 0.0
    ):
        raise P1Error("integrated coefficients must be finite and strictly positive")
    if not np.isfinite(local_loads).all():
        raise P1Error("element load vectors must contain only finite values")
    return _finalize_assembly(
        mesh,
        gradients,
        np.ascontiguousarray(coefficient_integrals),
        np.ascontiguousarray(local_loads),
        integrated_source,
        integrated_absolute_source,
        boundary_values,
        integration_description=description,
    )


def benchmark_p1_element_coefficients(control: ArrayLike, resolution: int) -> FloatArray:
    """Map ``rho[j,i]`` to aligned P1 square coefficients independently."""

    n = _validate_resolution(resolution, aligned=True)
    try:
        macro_coefficients = control_to_coefficient(control)
    except ValueError as exc:
        raise P1Error(f"invalid Branch C control: {exc}") from exc
    repeats = n // CONTROL_SHAPE[0]
    coefficients = np.repeat(
        np.repeat(macro_coefficients, repeats, axis=0), repeats, axis=1
    )
    coefficients = np.ascontiguousarray(coefficients, dtype=np.float64)
    if coefficients.shape != (n, n):
        raise P1Error("aligned benchmark coefficient expansion produced an invalid shape")
    lower, upper = COEFFICIENT_BOUNDS
    if not np.isfinite(coefficients).all() or np.any(coefficients < lower) or np.any(
        coefficients > upper
    ):
        raise P1Error("benchmark coefficients must remain within [3, 12]")
    return coefficients


def assemble_p1_zero_dirichlet_benchmark(
    control: ArrayLike,
    resolution: int,
) -> P1Assembly:
    """Exactly assemble the aligned Branch C ``f=1``, ``g=0`` P1 system."""

    mesh = build_p1_mesh(_validate_resolution(resolution, aligned=True))
    square_coefficients = benchmark_p1_element_coefficients(control, mesh.resolution)
    _, determinants, gradients = _mesh_triangle_geometry(mesh)
    areas = 0.5 * determinants
    triangle_coefficients = np.repeat(
        square_coefficients.reshape(-1, order="C"), 2
    )
    coefficient_integrals = np.ascontiguousarray(
        triangle_coefficients * areas,
        dtype=np.float64,
    )
    local_loads = np.ascontiguousarray(
        np.repeat((areas / 3.0)[:, None], 3, axis=1),
        dtype=np.float64,
    )
    return _finalize_assembly(
        mesh,
        gradients,
        coefficient_integrals,
        local_loads,
        1.0,
        1.0,
        0.0,
        integration_description=(
            "closed-form exact P1 integration for aligned piecewise-constant "
            "benchmark coefficient and unit source"
        ),
    )


def _validate_assembly(system: P1Assembly) -> None:
    if not isinstance(system, P1Assembly):
        raise TypeError("system must be a P1Assembly")
    _validate_mesh(system.mesh)
    node_count = system.mesh.node_count
    interior_count = system.mesh.interior_nodes.size
    boundary_count = system.mesh.boundary_nodes.size

    if (
        not sparse.isspmatrix_csr(system.full_stiffness)
        or system.full_stiffness.shape != (node_count, node_count)
        or system.full_stiffness.dtype != CANONICAL_DTYPE
    ):
        raise P1Error("full stiffness must be a float64 CSR square matrix")
    if not np.isfinite(system.full_stiffness.data).all():
        raise P1Error("full stiffness must contain only finite values")
    _stored_float64_array(
        system.full_load,
        name="full load",
        shape=(node_count,),
    )
    _stored_float64_array(
        system.boundary_values,
        name="boundary values",
        shape=(boundary_count,),
    )
    if (
        not sparse.isspmatrix_csr(system.reduced_matrix)
        or system.reduced_matrix.shape != (interior_count, interior_count)
        or system.reduced_matrix.dtype != CANONICAL_DTYPE
    ):
        raise P1Error("reduced stiffness must be a float64 CSR square matrix")
    if not np.isfinite(system.reduced_matrix.data).all():
        raise P1Error("reduced stiffness must contain only finite values")
    _stored_float64_array(
        system.reduced_rhs,
        name="reduced right-hand side",
        shape=(interior_count,),
    )
    source = _finite_scalar(system.integrated_source, name="integrated source")
    absolute_source = _finite_scalar(
        system.integrated_absolute_source,
        name="integrated absolute source",
    )
    source_consistency_tolerance = (
        100.0 * np.finfo(np.float64).eps * max(1.0, absolute_source)
    )
    if absolute_source < 0.0 or abs(source) > (
        absolute_source + source_consistency_tolerance
    ):
        raise P1Error("stored source integrals violate |integral f| <= integral |f|")
    if not isinstance(system.integration_description, str) or not system.integration_description:
        raise P1Error("integration description must be a non-empty string")


def p1_safe_ratio(numerator: float, denominator: float) -> float:
    """Apply the committed common convention for nonnegative ratios."""

    a = _finite_scalar(numerator, name="ratio numerator")
    d = _finite_scalar(denominator, name="ratio denominator")
    if a < 0.0 or d < 0.0:
        raise P1Error("ratio operands must be nonnegative")
    if d > 0.0:
        value = a / d
        if not math.isfinite(value):
            raise P1Error("a non-finite ratio blocks progression")
        return value
    if a == 0.0:
        return 0.0
    return math.inf


def p1_algebraic_backward_error(
    matrix: sparse.spmatrix,
    solution: ArrayLike,
    rhs: ArrayLike,
) -> float:
    """Reduced-system normwise backward error with induced infinity norm."""

    if (
        not sparse.issparse(matrix)
        or matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.dtype != CANONICAL_DTYPE
        or not np.isfinite(matrix.data).all()
    ):
        raise P1Error("matrix must be a finite float64 square SciPy sparse matrix")
    size = matrix.shape[0]
    u = _float64_array(solution, name="reduced solution", shape=(size,))
    b = _float64_array(rhs, name="reduced right-hand side", shape=(size,))
    if size == 0:
        return p1_safe_ratio(0.0, 0.0)
    residual = matrix @ u - b
    if not np.isfinite(residual).all():
        raise P1Error("algebraic residual must contain only finite values")
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))
    row_sums = np.asarray(abs(matrix).sum(axis=1), dtype=np.float64).reshape(-1)
    matrix_norm = float(np.max(row_sums))
    solution_norm = float(np.linalg.norm(u, ord=np.inf))
    rhs_norm = float(np.linalg.norm(b, ord=np.inf))
    denominator = matrix_norm * solution_norm + rhs_norm
    if not math.isfinite(denominator):
        raise P1Error("algebraic backward-error denominator must be finite")
    return p1_safe_ratio(residual_norm, denominator)


def p1_rhs_relative_l2_residual(
    matrix: sparse.spmatrix,
    solution: ArrayLike,
    rhs: ArrayLike,
) -> float | None:
    """Diagnostic-only reduced residual ``||r||_2/||b||_2`` when defined."""

    if (
        not sparse.issparse(matrix)
        or matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.dtype != CANONICAL_DTYPE
        or not np.isfinite(matrix.data).all()
    ):
        raise P1Error("matrix must be a finite float64 square SciPy sparse matrix")
    size = matrix.shape[0]
    u = _float64_array(solution, name="reduced solution", shape=(size,))
    b = _float64_array(rhs, name="reduced right-hand side", shape=(size,))
    if size == 0:
        return None
    residual = matrix @ u - b
    if not np.isfinite(residual).all():
        raise P1Error("algebraic residual must contain only finite values")
    rhs_norm = float(np.linalg.norm(b, ord=2))
    if rhs_norm == 0.0:
        return None
    return p1_safe_ratio(float(np.linalg.norm(residual, ord=2)), rhs_norm)


def _solution_flat(system: P1Assembly, solution: ArrayLike) -> FloatArray:
    _validate_assembly(system)
    width = system.mesh.resolution + 1
    nodal = _float64_array(
        solution,
        name="nodal solution",
        shape=(width, width),
    )
    return np.ascontiguousarray(nodal.reshape(-1, order="C"), dtype=np.float64)


def solve_p1_with_info(system: P1Assembly) -> P1SolveResult:
    """Solve the reduced SPD equations with the frozen Jacobi-PCG path."""

    _validate_assembly(system)
    matrix = system.reduced_matrix
    rhs = system.reduced_rhs
    interior_count = system.mesh.interior_nodes.size
    iterations = 0
    if interior_count:
        diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
        if diagonal.shape != (interior_count,) or not np.isfinite(diagonal).all():
            raise P1Error("Jacobi diagonal must be finite and have the reduced size")
        if np.any(diagonal <= 0.0):
            raise P1Error("Jacobi diagonal must be strictly positive")
        inverse_diagonal = 1.0 / diagonal
        if not np.isfinite(inverse_diagonal).all():
            raise P1Error("Jacobi inverse diagonal must be finite")
        preconditioner = sparse.diags(
            inverse_diagonal,
            offsets=0,
            shape=matrix.shape,
            format="csr",
            dtype=np.float64,
        )

        def count_iteration(_: FloatArray) -> None:
            nonlocal iterations
            iterations += 1

        try:
            interior_solution, info = cg(
                matrix,
                rhs,
                x0=np.zeros(interior_count, dtype=np.float64),
                rtol=P1_CG_RELATIVE_TOLERANCE,
                atol=P1_CG_ABSOLUTE_TOLERANCE,
                maxiter=P1_CG_MAXIMUM_ITERATIONS,
                M=preconditioner,
                callback=count_iteration,
            )
        except Exception as exc:
            raise P1Error(f"Jacobi-preconditioned CG failed: {exc}") from exc
        if info != 0:
            raise P1Error(
                "Jacobi-preconditioned CG did not converge within "
                f"{P1_CG_MAXIMUM_ITERATIONS} iterations (info={info})"
            )
        if (
            not isinstance(interior_solution, np.ndarray)
            or interior_solution.shape != (interior_count,)
            or interior_solution.dtype != CANONICAL_DTYPE
            or not np.isfinite(interior_solution).all()
        ):
            raise P1Error("CG returned an invalid, non-finite, or non-float64 solution")
        backward_error = p1_algebraic_backward_error(matrix, interior_solution, rhs)
        if not math.isfinite(backward_error) or backward_error > P1_ALGEBRAIC_GATE:
            raise P1Error("CG result failed the reduced algebraic backward-error gate")
    else:
        interior_solution = np.empty((0,), dtype=np.float64)

    flat_solution = np.empty(system.mesh.node_count, dtype=np.float64)
    flat_solution[system.mesh.boundary_nodes] = system.boundary_values
    flat_solution[system.mesh.interior_nodes] = interior_solution
    if not np.isfinite(flat_solution).all():
        raise P1Error("reconstructed nodal solution must contain only finite values")
    width = system.mesh.resolution + 1
    solution = np.ascontiguousarray(
        flat_solution.reshape((width, width), order="C"),
        dtype=np.float64,
    )
    return P1SolveResult(solution=solution, iterations=iterations)


def solve_p1(system: P1Assembly) -> FloatArray:
    """Return only the reconstructed nodal array from the frozen solve path."""

    return solve_p1_with_info(system).solution


def solve_p1_zero_dirichlet_benchmark(
    control: ArrayLike,
    resolution: int,
) -> tuple[P1Assembly, FloatArray]:
    """Assemble and solve one handcrafted/in-memory Branch C P1 control."""

    system = assemble_p1_zero_dirichlet_benchmark(control, resolution)
    return system, solve_p1(system)


def p1_boundary_reconstruction_error(
    system: P1Assembly,
    solution: ArrayLike,
) -> float:
    """Evaluate the committed normalized strong-boundary reconstruction gate."""

    flat = _solution_flat(system, solution)
    actual = flat[system.mesh.boundary_nodes]
    discrepancy = float(np.max(np.abs(actual - system.boundary_values)))
    boundary_norm = float(np.max(np.abs(system.boundary_values)))
    denominator = max(1.0, boundary_norm)
    return p1_safe_ratio(discrepancy, denominator)


def p1_boundary_reactions(
    system: P1Assembly,
    solution: ArrayLike,
) -> FloatArray:
    """Compute ``lambda_B = K_B*U-l_B`` from untouched full equations."""

    flat = _solution_flat(system, solution)
    boundary = system.mesh.boundary_nodes
    reactions = system.full_stiffness[boundary, :] @ flat - system.full_load[boundary]
    reactions = np.ascontiguousarray(reactions, dtype=np.float64)
    if reactions.shape != boundary.shape or not np.isfinite(reactions).all():
        raise P1Error("boundary reactions must be finite and match the boundary partition")
    return reactions


def p1_reaction_balance_ratio(
    boundary_reactions: ArrayLike,
    integrated_source: float,
    integrated_absolute_source: float,
) -> float:
    """Compare ``-sum(lambda_B)`` with the independently stored source."""

    if np.ma.isMaskedArray(boundary_reactions):
        raise P1Error("boundary reactions must not be a masked array")
    array = np.asarray(boundary_reactions)
    if array.ndim != 1:
        raise P1Error("boundary reactions must be a one-dimensional array")
    reactions = _float64_array(
        array,
        name="boundary reactions",
        shape=array.shape,
    )
    source = _finite_scalar(integrated_source, name="integrated source")
    absolute_source = _finite_scalar(
        integrated_absolute_source,
        name="integrated absolute source",
    )
    if absolute_source < 0.0:
        raise P1Error("integrated absolute source must be nonnegative")
    source_consistency_tolerance = (
        100.0 * np.finfo(np.float64).eps * max(1.0, absolute_source)
    )
    if abs(source) > absolute_source + source_consistency_tolerance:
        raise P1Error("source integrals must satisfy |integral f| <= integral |f|")
    outward_flux = -float(np.sum(reactions, dtype=np.float64))
    reaction_norm = float(np.sum(np.abs(reactions), dtype=np.float64))
    denominator = reaction_norm + absolute_source
    if not math.isfinite(outward_flux) or not math.isfinite(denominator):
        raise P1Error("reaction-balance operands must be finite")
    return p1_safe_ratio(abs(outward_flux - source), denominator)


def compute_p1_diagnostics(
    system: P1Assembly,
    solution: ArrayLike,
) -> P1Diagnostics:
    """Compute all committed P1 diagnostics from original and reduced systems."""

    flat = _solution_flat(system, solution)
    interior_solution = flat[system.mesh.interior_nodes]
    reactions = p1_boundary_reactions(system, solution)
    outward_flux = -float(np.sum(reactions, dtype=np.float64))
    reaction_denominator = float(
        np.sum(np.abs(reactions), dtype=np.float64)
        + system.integrated_absolute_source
    )
    reaction_ratio = p1_reaction_balance_ratio(
        reactions,
        system.integrated_source,
        system.integrated_absolute_source,
    )
    diagnostics = P1Diagnostics(
        algebraic_backward_error=p1_algebraic_backward_error(
            system.reduced_matrix,
            interior_solution,
            system.reduced_rhs,
        ),
        rhs_relative_l2_residual=p1_rhs_relative_l2_residual(
            system.reduced_matrix,
            interior_solution,
            system.reduced_rhs,
        ),
        boundary_reconstruction_error=p1_boundary_reconstruction_error(
            system,
            solution,
        ),
        boundary_reactions=reactions,
        outward_boundary_flux=outward_flux,
        integrated_source=system.integrated_source,
        reaction_denominator=reaction_denominator,
        reaction_balance_ratio=reaction_ratio,
    )
    operands = (
        diagnostics.algebraic_backward_error,
        diagnostics.boundary_reconstruction_error,
        diagnostics.outward_boundary_flux,
        diagnostics.integrated_source,
        diagnostics.reaction_denominator,
        diagnostics.reaction_balance_ratio,
    )
    if any(not math.isfinite(value) for value in operands):
        raise P1Error("non-finite P1 diagnostic blocks progression")
    return diagnostics


def _strict_nodal_field(
    solution: object,
    *,
    allowed_resolutions: tuple[int, ...] | None = None,
) -> tuple[FloatArray, int]:
    if not isinstance(solution, np.ndarray):
        raise TypeError("P1 nodal field must be a numpy.ndarray")
    if np.ma.isMaskedArray(solution):
        raise P1Error("P1 nodal field must not be a masked array")
    if solution.ndim != 2 or solution.shape[0] != solution.shape[1]:
        raise P1Error("P1 nodal field must have square shape (N+1, N+1)")
    resolution = solution.shape[0] - 1
    _validate_resolution(resolution)
    if allowed_resolutions is not None and resolution not in allowed_resolutions:
        raise P1Error(f"P1 nodal field resolution must be one of {allowed_resolutions}")
    if solution.dtype != CANONICAL_DTYPE:
        raise P1Error("P1 nodal field must have canonical float64 dtype")
    if not np.isfinite(solution).all():
        raise P1Error("P1 nodal field values must all be finite")
    return np.array(solution, dtype=np.float64, order="C", copy=True), resolution


def _p1_square_integrals(values: FloatArray, resolution: int) -> FloatArray:
    h = 1.0 / resolution
    southwest = values[:-1, :-1]
    southeast = values[:-1, 1:]
    northwest = values[1:, :-1]
    northeast = values[1:, 1:]
    # Each triangle integral is area/3 times the sum of its three nodal values;
    # area=h^2/2.  The two frozen SW-NE triangles are kept explicit here.
    lower_right = (h * h / 6.0) * (southwest + southeast + northeast)
    upper_left = (h * h / 6.0) * (southwest + northeast + northwest)
    integrals = lower_right + upper_left
    if not np.isfinite(integrals).all():
        raise P1Error("exact P1 square integrals must contain only finite values")
    return np.ascontiguousarray(integrals, dtype=np.float64)


def integrate_p1_field(solution: np.ndarray) -> float:
    """Exactly integrate a continuous nodal P1 field over the unit square."""

    values, resolution = _strict_nodal_field(solution)
    square_integrals = _p1_square_integrals(values, resolution)
    integral = float(np.sum(square_integrals, dtype=np.float64))
    if not math.isfinite(integral):
        raise P1Error("P1 field integral must be finite")
    return integral


def project_p1_to_64(solution: np.ndarray) -> FloatArray:
    """Exact ``Q_N_to_64`` for aligned P1-128 or P1-256 nodal fields.

    The output is a C-order ``[y,x]`` array.  Every entry is the exact integral
    of the frozen continuous P1 reconstruction over that 64-cell, divided by
    its area.  No TPFA restriction code is used.
    """

    values, resolution = _strict_nodal_field(
        solution,
        allowed_resolutions=P1_PROJECTION_RESOLUTIONS,
    )
    block = resolution // P1_PROJECTION_TARGET
    square_integrals = _p1_square_integrals(values, resolution)
    reshaped = square_integrals.reshape(
        P1_PROJECTION_TARGET,
        block,
        P1_PROJECTION_TARGET,
        block,
    )
    cell_integrals = np.sum(reshaped, axis=(1, 3), dtype=np.float64)
    projected = cell_integrals * np.float64(P1_PROJECTION_TARGET**2)
    projected = np.ascontiguousarray(projected, dtype=np.float64)
    if projected.shape != (P1_PROJECTION_TARGET, P1_PROJECTION_TARGET):
        raise P1Error("P1-to-64 projection produced an invalid output shape")
    if not np.isfinite(projected).all():
        raise P1Error("P1-to-64 projection must contain only finite values")

    original_integral = float(np.sum(square_integrals, dtype=np.float64))
    projected_integral = float(
        np.sum(projected, dtype=np.float64) / np.float64(P1_PROJECTION_TARGET**2)
    )
    absolute_scale = float(np.sum(np.abs(square_integrals), dtype=np.float64))
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, absolute_scale)
    if (
        not math.isfinite(original_integral)
        or not math.isfinite(projected_integral)
        or abs(projected_integral - original_integral) > tolerance
    ):
        raise P1Error("P1-to-64 projection failed its integral-preservation invariant")
    return projected


def _exact_gradient_values(
    gradient: Callable[[FloatArray, FloatArray], tuple[ArrayLike, ArrayLike]],
    x: FloatArray,
    y: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    if not callable(gradient):
        raise P1Error("exact gradient must be a callable returning (du/dx, du/dy)")
    try:
        raw = gradient(x, y)
    except Exception as exc:
        raise P1Error("exact gradient callable failed during quadrature evaluation") from exc
    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        raise P1Error("exact gradient callable must return exactly two components")
    components: list[FloatArray] = []
    for axis, component in zip(("x", "y"), raw):
        if np.ma.isMaskedArray(component):
            raise P1Error(f"exact {axis}-gradient must not be a masked array")
        array = np.asarray(component)
        if array.ndim == 0:
            array = np.full(x.shape, array.item())
        if array.shape != x.shape or array.dtype.kind not in "iuf":
            raise P1Error(
                f"exact {axis}-gradient must be a real scalar or shape {x.shape}"
            )
        try:
            converted = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise P1Error(f"exact {axis}-gradient must be float64-convertible") from exc
        if not np.isfinite(converted).all():
            raise P1Error(f"exact {axis}-gradient must contain only finite values")
        components.append(np.ascontiguousarray(converted, dtype=np.float64))
    return components[0], components[1]


def p1_physical_errors(
    solution: np.ndarray,
    exact_solution: ScalarFieldInput,
    exact_gradient: Callable[[FloatArray, FloatArray], tuple[ArrayLike, ArrayLike]],
    *,
    quadrature_order: int = P1_ERROR_QUADRATURE_ORDER,
) -> P1PhysicalErrors:
    """Evaluate physical L2 and H1-seminorm errors by frozen triangle quadrature."""

    nodal, resolution = _strict_nodal_field(solution)
    order = _validate_quadrature_order(quadrature_order, name="error quadrature order")
    mesh = build_p1_mesh(resolution)
    vertices, determinants, gradients = _mesh_triangle_geometry(mesh)
    flat = nodal.reshape(-1, order="C")
    element_values = flat[mesh.triangles]
    xi, eta, reference_weights = _reference_triangle_quadrature(order)
    basis = np.column_stack((1.0 - xi - eta, xi, eta))
    l2_squared = 0.0
    h1_squared = 0.0
    for start in range(0, mesh.element_count, P1_QUADRATURE_BATCH_SIZE):
        stop = min(start + P1_QUADRATURE_BATCH_SIZE, mesh.element_count)
        batch_vertices = vertices[start:stop]
        v0 = batch_vertices[:, 0, :]
        edge1 = batch_vertices[:, 1, :] - v0
        edge2 = batch_vertices[:, 2, :] - v0
        x = (
            v0[:, 0, None]
            + edge1[:, 0, None] * xi[None, :]
            + edge2[:, 0, None] * eta[None, :]
        )
        y = (
            v0[:, 1, None]
            + edge1[:, 1, None] * xi[None, :]
            + edge2[:, 1, None] * eta[None, :]
        )
        weights = determinants[start:stop, None] * reference_weights[None, :]
        exact = _evaluate_field(
            exact_solution,
            x,
            y,
            name="exact solution",
            positive=False,
        )
        numerical = element_values[start:stop] @ basis.T
        difference = numerical - exact
        l2_squared += float(np.sum(weights * difference**2, dtype=np.float64))

        grad_x, grad_y = _exact_gradient_values(exact_gradient, x, y)
        numerical_gradient = np.einsum(
            "ta,tad->td",
            element_values[start:stop],
            gradients[start:stop],
            optimize=True,
        )
        gradient_error_squared = (
            numerical_gradient[:, 0, None] - grad_x
        ) ** 2 + (
            numerical_gradient[:, 1, None] - grad_y
        ) ** 2
        h1_squared += float(
            np.sum(weights * gradient_error_squared, dtype=np.float64)
        )
    if (
        not math.isfinite(l2_squared)
        or not math.isfinite(h1_squared)
        or l2_squared < 0.0
        or h1_squared < 0.0
    ):
        raise P1Error("physical error integrals must be finite and nonnegative")
    return P1PhysicalErrors(l2=math.sqrt(l2_squared), h1_seminorm=math.sqrt(h1_squared))


def _evaluate_line_function(
    function: Callable[[FloatArray], ArrayLike],
    points: FloatArray,
    *,
    name: str,
) -> FloatArray:
    if not callable(function):
        raise P1Error(f"{name} must be callable")
    try:
        raw = function(points)
    except Exception as exc:
        raise P1Error(f"{name} callable failed on interface quadrature points") from exc
    if np.ma.isMaskedArray(raw):
        raise P1Error(f"{name} must not return a masked array")
    array = np.asarray(raw)
    if array.ndim == 0:
        array = np.full(points.shape, array.item())
    if array.shape != points.shape or array.dtype.kind not in "iuf":
        raise P1Error(f"{name} must return a real scalar or shape {points.shape}")
    values = np.asarray(array, dtype=np.float64)
    if not np.isfinite(values).all():
        raise P1Error(f"{name} must return only finite values")
    return np.ascontiguousarray(values, dtype=np.float64)


def p1_vertical_interface_flux_errors(
    solution: np.ndarray,
    coefficient_left: Real,
    coefficient_right: Real,
    exact_flux: Callable[[FloatArray], ArrayLike],
    *,
    interface_x: float = 0.5,
    quadrature_order: int = P1_EDGE_QUADRATURE_ORDER,
) -> P1InterfaceFluxErrors:
    """Relative line-L2 errors of the two aligned vertical P1 flux traces.

    For each interface segment, the left trace is taken from that left square's
    ``(SW,SE,NE)`` triangle and the right trace from the right square's
    ``(SW,NE,NW)`` triangle.  Both use the same orientation ``q_x=-k du/dx``.
    """

    nodal, resolution = _strict_nodal_field(solution)
    location = _finite_scalar(interface_x, name="interface_x")
    scaled = location * resolution
    interface_column = int(round(scaled))
    if (
        not 0 < interface_column < resolution
        or not math.isclose(scaled, interface_column, rel_tol=0.0, abs_tol=1.0e-13)
    ):
        raise P1Error("interface_x must align with an interior P1 node column")
    left_kappa = _finite_scalar(
        coefficient_left,
        name="left interface coefficient",
        positive=True,
    )
    right_kappa = _finite_scalar(
        coefficient_right,
        name="right interface coefficient",
        positive=True,
    )
    order = _validate_quadrature_order(
        quadrature_order,
        name="edge quadrature order",
    )

    left_trace = np.empty(resolution, dtype=np.float64)
    right_trace = np.empty(resolution, dtype=np.float64)
    h = 1.0 / resolution
    left_square = interface_column - 1
    right_square = interface_column
    for j in range(resolution):
        left_vertices = np.array(
            [
                [left_square * h, j * h],
                [interface_column * h, j * h],
                [interface_column * h, (j + 1) * h],
            ],
            dtype=np.float64,
        )
        _, _, left_gradients = _triangle_geometry(left_vertices)
        left_values = np.array(
            [
                nodal[j, left_square],
                nodal[j, interface_column],
                nodal[j + 1, interface_column],
            ],
            dtype=np.float64,
        )
        left_trace[j] = -left_kappa * float(left_values @ left_gradients[:, 0])

        right_vertices = np.array(
            [
                [interface_column * h, j * h],
                [(right_square + 1) * h, (j + 1) * h],
                [interface_column * h, (j + 1) * h],
            ],
            dtype=np.float64,
        )
        _, _, right_gradients = _triangle_geometry(right_vertices)
        right_values = np.array(
            [
                nodal[j, interface_column],
                nodal[j + 1, right_square + 1],
                nodal[j + 1, interface_column],
            ],
            dtype=np.float64,
        )
        right_trace[j] = -right_kappa * float(right_values @ right_gradients[:, 0])

    nodes, weights = np.polynomial.legendre.leggauss(order)
    lower = np.arange(resolution, dtype=np.float64)[:, None] * h
    y = lower + 0.5 * h * (nodes[None, :] + 1.0)
    exact = _evaluate_line_function(exact_flux, y, name="exact interface flux")
    physical_weights = 0.5 * h * weights[None, :]
    denominator = float(np.sum(physical_weights * exact**2, dtype=np.float64))
    left_numerator = float(
        np.sum(physical_weights * (left_trace[:, None] - exact) ** 2, dtype=np.float64)
    )
    right_numerator = float(
        np.sum(physical_weights * (right_trace[:, None] - exact) ** 2, dtype=np.float64)
    )
    left_ratio = p1_safe_ratio(left_numerator, denominator)
    right_ratio = p1_safe_ratio(right_numerator, denominator)
    if not math.isfinite(left_ratio) or not math.isfinite(right_ratio):
        raise P1Error("non-finite relative interface-flux error blocks progression")
    return P1InterfaceFluxErrors(
        left=math.sqrt(left_ratio),
        right=math.sqrt(right_ratio),
    )
