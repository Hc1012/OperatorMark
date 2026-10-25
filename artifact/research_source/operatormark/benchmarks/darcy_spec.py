"""Frozen Branch C v1 Darcy specification utilities.

This module deliberately exposes only the development split.  It contains no
generic split/seed sampler, so callers cannot accidentally generate a sealed or
corpus split through this API.
"""

from __future__ import annotations

import hashlib
from numbers import Integral
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray


Float64Array = NDArray[np.float64]

SCHEMA_VERSION: Final = "operatormark/darcy-transparent/v1"
BENCHMARK_ID: Final = "darcy_branch_c_v1"
NUMPY_VERSION: Final = "2.5.2"
CANONICAL_DTYPE: Final = np.dtype(np.float64)
TENSOR_AXIS_ORDER: Final = ("sample", "channel", "y", "x")

CONTROL_SHAPE: Final = (8, 8)
CONTROL_SIZE: Final = 64
CONTROL_BOUNDS: Final = (0.0, 1.0)
COEFFICIENT_OFFSET: Final = 3.0
COEFFICIENT_SCALE: Final = 9.0
COEFFICIENT_BOUNDS: Final = (3.0, 12.0)

NATIVE_TPFA_RESOLUTION: Final = 64
DERIVED_TPFA_RESOLUTIONS: Final = (32, 16)

DEVELOPMENT_SPLIT: Final = "development"
DEVELOPMENT_SAMPLE_COUNT: Final = 32
DEVELOPMENT_SEED: Final = 2026083101
DEVELOPMENT_BIT_GENERATOR: Final = "PCG64"

TENSOR_FINGERPRINT_SCHEMA: Final = "sha256-tensor-v1"


class DarcySpecError(ValueError):
    """Raised when data violate the frozen Branch C v1 specification."""


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise DarcySpecError(f"{name} must be an integer")
    return int(value)


def _resolution(value: object, *, require_macro_alignment: bool) -> int:
    resolution = _integer(value, name="resolution")
    if resolution <= 0:
        raise DarcySpecError("resolution must be positive")
    if require_macro_alignment and resolution % CONTROL_SHAPE[0] != 0:
        raise DarcySpecError("resolution must be divisible by 8 for macro-cell alignment")
    return resolution


def validate_control(control: ArrayLike) -> Float64Array:
    """Return a validated C-contiguous float64 ``rho[y, x]`` copy.

    Shape, orientation, and values are never repaired: in particular this
    function does not reshape, transpose, or clip its input.
    """

    if np.ma.isMaskedArray(control):
        raise DarcySpecError("control must not be a masked array")
    try:
        source = np.asarray(control)
    except (TypeError, ValueError) as exc:
        raise DarcySpecError("control must be convertible to a numeric array") from exc
    if source.shape != CONTROL_SHAPE:
        raise DarcySpecError(f"control shape must be {CONTROL_SHAPE}, got {source.shape}")
    if source.dtype.kind not in "buif":
        raise DarcySpecError("control dtype must be real numeric and float64-convertible")
    # Validate in the source dtype before narrowing.  In particular, a
    # longdouble immediately outside [0, 1] must not round into the interval
    # during the canonical float64 conversion.
    if not np.isfinite(source).all():
        raise DarcySpecError("control values must all be finite")
    lower, upper = CONTROL_BOUNDS
    if np.any(source < lower) or np.any(source > upper):
        raise DarcySpecError("control values must lie in [0, 1]")
    try:
        validated = np.array(source, dtype=CANONICAL_DTYPE, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DarcySpecError("control dtype must be float64-convertible") from exc
    if not np.isfinite(validated).all():
        raise DarcySpecError("control values must all be finite")
    if np.any(validated < lower) or np.any(validated > upper):
        raise DarcySpecError("control values must lie in [0, 1]")
    return validated


def control_to_coefficient(control: ArrayLike) -> Float64Array:
    """Map an 8-by-8 control to ``kappa = 3 + 9*rho`` in ``[3, 12]``."""

    rho = validate_control(control)
    coefficient = COEFFICIENT_OFFSET + COEFFICIENT_SCALE * rho
    return np.ascontiguousarray(coefficient, dtype=CANONICAL_DTYPE)


def _expand_macro_cells(macro_values: Float64Array, resolution: int) -> Float64Array:
    repeats = resolution // CONTROL_SHAPE[0]
    expanded = np.repeat(np.repeat(macro_values, repeats, axis=0), repeats, axis=1)
    return np.ascontiguousarray(expanded, dtype=CANONICAL_DTYPE)


def expand_control_to_tpfa(control: ArrayLike, resolution: int) -> Float64Array:
    """Expand ``rho[j, i]`` exactly over an aligned TPFA cell grid."""

    aligned_resolution = _resolution(resolution, require_macro_alignment=True)
    return _expand_macro_cells(validate_control(control), aligned_resolution)


def coefficient_on_tpfa_grid(control: ArrayLike, resolution: int) -> Float64Array:
    """Expand the Branch C coefficient exactly over an aligned TPFA grid."""

    aligned_resolution = _resolution(resolution, require_macro_alignment=True)
    return _expand_macro_cells(control_to_coefficient(control), aligned_resolution)


def tpfa_cell_centers(resolution: int) -> tuple[Float64Array, Float64Array]:
    """Return the one-dimensional ``(x_i, y_j)`` TPFA cell centres."""

    checked_resolution = _resolution(resolution, require_macro_alignment=False)
    coordinates = (
        np.arange(checked_resolution, dtype=CANONICAL_DTYPE) + np.float64(0.5)
    ) / np.float64(checked_resolution)
    x = np.ascontiguousarray(coordinates, dtype=CANONICAL_DTYPE)
    y = x.copy(order="C")
    return x, y


def tpfa_cell_center_mesh(resolution: int) -> tuple[Float64Array, Float64Array]:
    """Return ``X[j,i]=x_i`` and ``Y[j,i]=y_j`` on the TPFA grid."""

    x, y = tpfa_cell_centers(resolution)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="xy")
    return (
        np.ascontiguousarray(x_mesh, dtype=CANONICAL_DTYPE),
        np.ascontiguousarray(y_mesh, dtype=CANONICAL_DTYPE),
    )


def _check_numpy_sampling_version() -> None:
    if np.__version__ != NUMPY_VERSION:
        raise DarcySpecError(
            f"development sampling requires NumPy {NUMPY_VERSION}, got {np.__version__}"
        )


def _development_prefix(count: int) -> Float64Array:
    _check_numpy_sampling_version()
    generator = np.random.Generator(np.random.PCG64(DEVELOPMENT_SEED))
    controls = np.empty((count, *CONTROL_SHAPE), dtype=CANONICAL_DTYPE, order="C")
    for local_index in range(count):
        if local_index % 2 == 0:
            controls[local_index] = generator.uniform(0.0, 1.0, size=CONTROL_SHAPE)
        else:
            sample = np.zeros(CONTROL_SIZE, dtype=CANONICAL_DTYPE)
            permutation = generator.permutation(CONTROL_SIZE)
            sample[permutation[: CONTROL_SIZE // 2]] = 1.0
            controls[local_index] = sample.reshape(CONTROL_SHAPE, order="C")
    return controls


def development_controls() -> Float64Array:
    """Generate all 32 development controls in ascending local-index order."""

    return _development_prefix(DEVELOPMENT_SAMPLE_COUNT)


def development_control(local_index: int) -> Float64Array:
    """Generate one development control while consuming all preceding draws."""

    checked_index = _integer(local_index, name="local_index")
    if not 0 <= checked_index < DEVELOPMENT_SAMPLE_COUNT:
        raise DarcySpecError(
            f"local_index must be in [0, {DEVELOPMENT_SAMPLE_COUNT - 1}]"
        )
    return _development_prefix(checked_index + 1)[checked_index].copy(order="C")


def tensor_fingerprint(tensor: np.ndarray) -> str:
    """Compute the exact ``sha256-tensor-v1`` digest of a NumPy tensor."""

    if not isinstance(tensor, np.ndarray):
        raise TypeError("tensor must be a numpy.ndarray")
    if np.ma.isMaskedArray(tensor):
        raise DarcySpecError("masked tensors cannot be fingerprinted reproducibly")
    if tensor.dtype.hasobject:
        raise DarcySpecError("object-containing tensors cannot be fingerprinted reproducibly")
    # np.ascontiguousarray promotes a zero-dimensional array to shape (1,),
    # which would violate the fingerprint's shape binding.  np.array preserves
    # rank while producing the required independent C-order byte layout.
    contiguous = np.array(tensor, dtype=tensor.dtype, order="C", copy=True, subok=False)
    dimensions = ",".join(str(dimension) for dimension in tensor.shape)
    header = (
        f"dtype={tensor.dtype.str}\n"
        f"shape={dimensions}\n"
        "order=C\n"
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def restrict_tpfa64(tensor: np.ndarray, target_resolution: int) -> Float64Array:
    """Area-average a float64 TPFA-64 tensor to exactly 32 or 16 cells."""

    if not isinstance(tensor, np.ndarray):
        raise TypeError("tensor must be a numpy.ndarray")
    if np.ma.isMaskedArray(tensor):
        raise DarcySpecError("tensor must not be a masked array")
    target = _integer(target_resolution, name="target_resolution")
    if target not in DERIVED_TPFA_RESOLUTIONS:
        raise DarcySpecError("target_resolution must be exactly 32 or 16")
    if tensor.ndim < 2 or tensor.shape[-2:] != (
        NATIVE_TPFA_RESOLUTION,
        NATIVE_TPFA_RESOLUTION,
    ):
        raise DarcySpecError("tensor must have trailing shape (64, 64)")
    if tensor.dtype != CANONICAL_DTYPE:
        raise DarcySpecError("tensor must have canonical float64 dtype")
    if not np.isfinite(tensor).all():
        raise DarcySpecError("tensor values must all be finite")

    block = NATIVE_TPFA_RESOLUTION // target
    reshaped = tensor.reshape(*tensor.shape[:-2], target, block, target, block)
    with np.errstate(over="ignore", invalid="ignore"):
        restricted = reshaped.sum(
            axis=(-3, -1), dtype=CANONICAL_DTYPE
        ) / np.float64(block * block)
    if not np.isfinite(restricted).all():
        raise DarcySpecError("area averaging must produce only finite values")
    return np.ascontiguousarray(restricted, dtype=CANONICAL_DTYPE)


def restrict_tpfa64_to_32(tensor: np.ndarray) -> Float64Array:
    """Exact 2-by-2 area averaging of a TPFA-64 tensor."""

    return restrict_tpfa64(tensor, 32)


def restrict_tpfa64_to_16(tensor: np.ndarray) -> Float64Array:
    """Exact 4-by-4 area averaging of a TPFA-64 tensor."""

    return restrict_tpfa64(tensor, 16)
