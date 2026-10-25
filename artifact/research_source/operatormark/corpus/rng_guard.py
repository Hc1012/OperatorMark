"""Frozen RNG-state commitments and fail-closed call/seed guards."""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import wraps
from types import ModuleType
from typing import Any, Final

import numpy as np


NUMPY_LEGACY_DOMAIN: Final = "operatormark-numpy-legacy-rng-state-v1\n"
PYTHON_RANDOM_DOMAIN: Final = "operatormark-python-random-state-v1\n"
PCG64_STATE_DOMAIN: Final = "operatormark-pcg64-state-v1\n"

NUMPY_RANDOM_MODULE_CALLS: Final = (
    "Generator", "MT19937", "PCG64", "PCG64DXSM", "Philox", "RandomState",
    "SFC64", "SeedSequence", "beta", "binomial", "bytes", "chisquare",
    "choice", "default_rng", "dirichlet", "exponential", "f", "gamma",
    "geometric", "get_state", "gumbel", "hypergeometric", "laplace",
    "logistic", "lognormal", "logseries", "multinomial", "multivariate_normal",
    "negative_binomial", "noncentral_chisquare", "noncentral_f", "normal",
    "pareto", "permutation", "poisson", "power", "rand", "randint", "randn",
    "random", "random_integers", "random_sample", "ranf", "rayleigh", "sample",
    "seed", "set_state", "shuffle", "standard_cauchy", "standard_exponential",
    "standard_gamma", "standard_normal", "standard_t", "triangular", "uniform",
    "vonmises", "wald", "weibull", "zipf",
)
PYTHON_RANDOM_MODULE_CALLS: Final = (
    "Random", "SystemRandom", "betavariate", "binomialvariate", "choice", "choices",
    "expovariate", "gammavariate", "gauss", "getrandbits", "getstate", "lognormvariate",
    "normalvariate", "paretovariate", "randbytes", "randint", "random", "randrange",
    "sample", "seed", "setstate", "shuffle", "triangular", "uniform",
    "vonmisesvariate", "weibullvariate",
)
GENERATOR_METHODS: Final = (
    "beta", "binomial", "bytes", "chisquare", "choice", "dirichlet", "exponential",
    "f", "gamma", "geometric", "gumbel", "hypergeometric", "integers", "laplace",
    "logistic", "lognormal", "logseries", "multinomial", "multivariate_hypergeometric",
    "multivariate_normal", "negative_binomial", "noncentral_chisquare", "noncentral_f",
    "normal", "pareto", "permutation", "poisson", "power", "random", "rayleigh",
    "shuffle", "spawn", "standard_cauchy", "standard_exponential", "standard_gamma",
    "standard_normal", "standard_t", "triangular", "uniform", "vonmises", "wald",
    "weibull", "zipf",
)


class RngGuardViolation(RuntimeError):
    """Raised after a forbidden explicit RNG call or seed access is counted."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def canonical_numpy_legacy_state(state: tuple[Any, ...]) -> dict[str, Any]:
    if not isinstance(state, tuple) or len(state) != 5:
        raise RngGuardViolation("unexpected NumPy legacy RNG state structure")
    bit_generator, key, position, has_gauss, cached_gaussian = state
    key_array = np.asarray(key)
    if key_array.ndim != 1 or key_array.dtype.kind != "u":
        raise RngGuardViolation("NumPy legacy key must be a one-dimensional unsigned array")
    normalized = np.ascontiguousarray(key_array.astype("<u4", copy=False))
    cached = np.float64(cached_gaussian)
    if not np.isfinite(cached):
        raise RngGuardViolation("NumPy cached Gaussian must be finite")
    return {
        "bit_generator": str(bit_generator),
        "cached_gaussian_hex": float(cached).hex().lower(),
        "has_gauss": int(has_gauss),
        "key_c_order_hex": normalized.tobytes(order="C").hex(),
        "key_dtype": normalized.dtype.str,
        "key_shape": list(normalized.shape),
        "position": int(position),
    }


def canonical_python_random_state(state: tuple[Any, ...]) -> dict[str, Any]:
    if not isinstance(state, tuple) or len(state) != 3:
        raise RngGuardViolation("unexpected Python random state structure")
    version, internal_state, gauss_next = state
    if not isinstance(internal_state, tuple) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in internal_state
    ):
        raise RngGuardViolation("Python random internal state must be an integer tuple")
    if gauss_next is None:
        encoded_gauss: str | None = None
    else:
        scalar = float(gauss_next)
        if not np.isfinite(scalar):
            raise RngGuardViolation("Python cached Gaussian must be finite")
        encoded_gauss = scalar.hex().lower()
    return {
        "gauss_next": encoded_gauss,
        "internal_state": list(internal_state),
        "version": int(version),
    }


def numpy_legacy_state_commitment(state: tuple[Any, ...]) -> str:
    return hashlib.sha256(
        NUMPY_LEGACY_DOMAIN.encode("utf-8") + _canonical_bytes(canonical_numpy_legacy_state(state))
    ).hexdigest()


def python_random_state_commitment(state: tuple[Any, ...]) -> str:
    return hashlib.sha256(
        PYTHON_RANDOM_DOMAIN.encode("utf-8") + _canonical_bytes(canonical_python_random_state(state))
    ).hexdigest()


def pcg64_state_commitment(state: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        PCG64_STATE_DOMAIN.encode("utf-8") + _canonical_bytes(dict(state))
    ).hexdigest()


def guarded_call_manifest() -> dict[str, Any]:
    if np.__version__ != "2.5.2":
        raise RngGuardViolation(f"guard manifest requires NumPy 2.5.2, got {np.__version__}")
    if sys.version_info[:3] != (3, 13, 5):
        raise RngGuardViolation(
            f"guard manifest requires Python 3.13.5, got {platform_python_version()}"
        )
    missing_numpy = [name for name in NUMPY_RANDOM_MODULE_CALLS if not callable(getattr(np.random, name, None))]
    missing_python = [name for name in PYTHON_RANDOM_MODULE_CALLS if not callable(getattr(random, name, None))]
    missing_generator = [name for name in GENERATOR_METHODS if not callable(getattr(np.random.Generator, name, None))]
    if missing_numpy or missing_python or missing_generator:
        raise RngGuardViolation(
            f"guard manifest mismatch: numpy={missing_numpy}, python={missing_python}, generator={missing_generator}"
        )
    payload = {
        "generator_methods": list(GENERATOR_METHODS),
        "numpy_random_module_calls": list(NUMPY_RANDOM_MODULE_CALLS),
        "numpy_version": np.__version__,
        "python_random_module_calls": list(PYTHON_RANDOM_MODULE_CALLS),
        "python_version": platform_python_version(),
        "schema": "operatormark.darcy_branch_c_v1.rng_guard_manifest.v1",
    }
    return {**payload, "manifest_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest()}


def platform_python_version() -> str:
    return ".".join(str(value) for value in sys.version_info[:3])


@dataclass(slots=True)
class RngCallGuard:
    """Patch version-pinned RNG entry points and count every violation."""

    forbidden_rng_call_count: int = 0
    _installed: bool = False
    _controller_local: threading.local = field(default_factory=threading.local)
    _originals: dict[tuple[ModuleType, str], Any] = field(default_factory=dict)
    _global_replacements: list[tuple[dict[str, Any], str, Any]] = field(default_factory=list)
    _on_violation: Any = None

    def _violate(self, qualified_name: str) -> None:
        self.forbidden_rng_call_count += 1
        if self._on_violation is not None:
            self._on_violation(qualified_name)
        raise RngGuardViolation(f"forbidden explicit RNG call: {qualified_name}")

    def _wrapper(self, qualified_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            if qualified_name in {"numpy.random.get_state", "random.getstate"} and getattr(
                self._controller_local, "checkpoint", False
            ):
                return original(*args, **kwargs)
            self._violate(qualified_name)

        return guarded

    def _constructor_wrapper(self, qualified_name: str, original: type[Any]) -> type[Any]:
        guard = self

        class GuardedConstructorMeta(type):
            def __instancecheck__(cls, instance: object) -> bool:
                return isinstance(instance, original)

            def __subclasscheck__(cls, subclass: type[Any]) -> bool:
                return issubclass(subclass, original)

        class GuardedConstructor(metaclass=GuardedConstructorMeta):
            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                guard._violate(qualified_name)

        GuardedConstructor.__module__ = original.__module__
        GuardedConstructor.__name__ = original.__name__
        GuardedConstructor.__qualname__ = original.__qualname__
        GuardedConstructor.__doc__ = original.__doc__
        return GuardedConstructor

    def _replace_cached_module_globals(self, replacements: Mapping[int, Any]) -> None:
        for module in tuple(sys.modules.values()):
            namespace = getattr(module, "__dict__", None)
            if not isinstance(namespace, dict):
                continue
            for name, value in tuple(namespace.items()):
                replacement = replacements.get(id(value))
                if replacement is not None:
                    self._global_replacements.append((namespace, name, value))
                    namespace[name] = replacement

    def install(self) -> None:
        if self._installed:
            raise RngGuardViolation("RNG guard already installed")
        guarded_call_manifest()
        replacements: dict[int, Any] = {}
        for module, names, prefix in (
            (np.random, NUMPY_RANDOM_MODULE_CALLS, "numpy.random"),
            (random, PYTHON_RANDOM_MODULE_CALLS, "random"),
        ):
            for name in names:
                original = getattr(module, name)
                qualified_name = f"{prefix}.{name}"
                wrapper = (
                    self._constructor_wrapper(qualified_name, original)
                    if isinstance(original, type)
                    else self._wrapper(qualified_name, original)
                )
                self._originals[(module, name)] = original
                replacements[id(original)] = wrapper
                setattr(module, name, wrapper)
        self._replace_cached_module_globals(replacements)
        self._installed = True

    @contextlib.contextmanager
    def audit_checkpoint(self) -> Iterator[None]:
        if not self._installed or getattr(self._controller_local, "checkpoint", False):
            raise RngGuardViolation("invalid nested or inactive audit checkpoint")
        self._controller_local.checkpoint = True
        try:
            yield
        finally:
            self._controller_local.checkpoint = False

    def state_commitments(self) -> dict[str, str]:
        with self.audit_checkpoint():
            numpy_state = np.random.get_state()
            python_state = random.getstate()
        return {
            "numpy_legacy_sha256": numpy_legacy_state_commitment(numpy_state),
            "python_random_sha256": python_random_state_commitment(python_state),
        }

    def uninstall_for_state2_test_only(self) -> None:
        if not self._installed:
            return
        for namespace, name, original in reversed(self._global_replacements):
            namespace[name] = original
        for (module, name), original in self._originals.items():
            setattr(module, name, original)
        self._global_replacements.clear()
        self._originals.clear()
        self._installed = False


@dataclass(slots=True)
class CorpusSeedAccessGuard:
    """Count and deny every corpus-seed lookup outside a State 4 capability."""

    forbidden_corpus_seed_access_count: int = 0

    def deny_lookup(self, split: str) -> None:
        self.forbidden_corpus_seed_access_count += 1
        raise RngGuardViolation(f"corpus split seed access denied outside State 4: {split}")

    def authorize_state4(self, *, state: str, authorization_file_sha256: str) -> "State4SeedCapability":
        if state != "state4":
            self.forbidden_corpus_seed_access_count += 1
            raise RngGuardViolation("corpus seed capability is available only in State 4")
        if len(authorization_file_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in authorization_file_sha256
        ):
            raise RngGuardViolation("invalid State 4 authorization commitment")
        return State4SeedCapability(authorization_file_sha256=authorization_file_sha256)


@dataclass(slots=True)
class State4SeedCapability:
    """Single-use-per-split seed access accounting; it stores no seed literal."""

    authorization_file_sha256: str
    access_count_by_split: dict[str, int] = field(
        default_factory=lambda: {"training": 0, "validation": 0, "test": 0}
    )
    _expired: bool = False

    def retrieve_once(self, split: str, provider: Callable[[], int]) -> int:
        if self._expired or split not in self.access_count_by_split:
            raise RngGuardViolation("invalid or expired State 4 seed capability")
        if self.access_count_by_split[split] != 0:
            raise RngGuardViolation(f"State 4 seed already retrieved for {split}")
        value = provider()
        if isinstance(value, bool) or not isinstance(value, int):
            raise RngGuardViolation("State 4 seed provider returned a non-integer")
        self.access_count_by_split[split] = 1
        return value

    def expire(self) -> None:
        self._expired = True
