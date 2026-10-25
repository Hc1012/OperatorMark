"""Shared types and Protocols.  No numerics, no solver imports.

The shared consumers (ladder, directions, task, runners) see only these types,
64-grid arrays and scalars.  Matrices, factorizations and native arrays stay
inside a family's Session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol

from numpy.typing import NDArray

Method = Literal["TPFA", "P1"]

TPFA_LEVELS: tuple[int, int, int] = (64, 128, 256)   # ladder levels l = 1, 2, 3
P1_LEVELS: tuple[int, int, int] = (128, 256, 512)    # ladder levels l = 1, 2, 3
# Test level sets (synthetic material only): three levels each so every ladder path runs.
TEST_TPFA_LEVELS: tuple[int, int, int] = (64, 128, 256)
TEST_P1_LEVELS: tuple[int, int, int] = (64, 128, 256)

FIELD_SHAPE: tuple[int, int] = (64, 64)
CONTROL_SHAPE: tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class ForwardResult:
    method: Method
    level: int                          # native N
    control_sha256: str                 # sha256-tensor-v1 of rho [8,8]
    native_sha256: str                  # TPFA u[N,N]; P1 nodal [N+1,N+1]
    coefficient_sha256: str             # native coefficient (hash only)
    field64: NDArray                    # [64,64]
    correction64: NDArray               # [64,64] image of one algebraic correction
    diagnostics: Mapping[str, float | None]
    gates_pass: bool
    transfer_integral_error: float
    iterations: int | None              # P1 PCG iterations; None for TPFA
    eta_iterations: int | None          # P1 continuation iterations; None for TPFA


@dataclass(frozen=True)
class TangentResult:
    method: Method
    level: int
    control_sha256: str
    direction_sha256: str
    native_sha256: str
    field64: NDArray                    # D F64[v] image at this level
    correction64: NDArray
    tangent_backward_error: float
    iterations: int | None
    eta_iterations: int | None


class Session(Protocol):
    forward: ForwardResult

    def tangent(self, v: NDArray) -> TangentResult: ...

    def native_solution(self) -> NDArray: ...

    def close(self) -> None: ...

    def __enter__(self) -> "Session": ...

    def __exit__(self, *exc: object) -> None: ...


class MethodFamily(Protocol):
    name: Method
    levels: tuple[int, ...]

    def open(self, rho: NDArray, level: int) -> Session: ...


class AdjointSession(Session, Protocol):
    def vjp64(self, w64: NDArray) -> NDArray: ...


class FiniteDifferenceOracle(Protocol):
    def field64(self, rho: NDArray, level: int) -> NDArray: ...


class NumericsError(RuntimeError):
    """A deterministic numerical failure (recorded as an outcome, never retried)."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code
