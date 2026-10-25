"""Self-bootstrapping fresh-child controller for guarded scientific actions."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import random
import sys
import time
import types
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any, Final


# Duplicated as a literal audit binding. The bootstrap and base_imports module
# must both match this exact tuple before any action is dispatched.
BASE_IMPORT_ROOTS: Final = (
    "numpy",
    "numpy.typing",
    "scipy",
    "scipy.sparse",
    "scipy.sparse.linalg",
)
BOOTSTRAP_MODULE: Final = "_operatormark_fresh_child_bootstrap"
BOOTSTRAP_SENTINEL: Final = "operatormark-fresh-child-bootstrap-v2"
PROTOCOL_SHA256: Final = "a38ef79b6c4bdccf53f599fa08f77e48a111afbf0b2783ad5733bda99996599a"


class _BootstrapRngViolation(RuntimeError):
    pass


class _BootstrapRngGuard:
    """Pre-project version-pinned deny guard retained for the child's lifetime."""

    NUMPY_CALLS: Final = (
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
    PYTHON_CALLS: Final = (
        "Random", "SystemRandom", "betavariate", "binomialvariate", "choice", "choices",
        "expovariate", "gammavariate", "gauss", "getrandbits", "getstate",
        "lognormvariate", "normalvariate", "paretovariate", "randbytes", "randint",
        "random", "randrange", "sample", "seed", "setstate", "shuffle",
        "triangular", "uniform", "vonmisesvariate", "weibullvariate",
    )

    def __init__(self, numpy: Any) -> None:
        self.numpy = numpy
        self.forbidden_rng_call_count = 0
        self._checkpoint = False
        self._replacements: dict[int, Any] = {}

    def _violate(self, qualified_name: str) -> None:
        self.forbidden_rng_call_count += 1
        raise _BootstrapRngViolation(f"forbidden explicit RNG call: {qualified_name}")

    def _wrapper(self, qualified_name: str, original: Any) -> Any:
        def guarded(*args: Any, **kwargs: Any) -> Any:
            if qualified_name in {"numpy.random.get_state", "random.getstate"} and self._checkpoint:
                return original(*args, **kwargs)
            self._violate(qualified_name)

        guarded.__name__ = getattr(original, "__name__", qualified_name.rsplit(".", 1)[-1])
        guarded.__qualname__ = getattr(original, "__qualname__", guarded.__name__)
        return guarded

    def _constructor(self, qualified_name: str, original: type[Any]) -> type[Any]:
        guard = self

        class GuardedConstructorMeta(type):
            def __instancecheck__(cls, instance: object) -> bool:
                return isinstance(instance, original)

            def __subclasscheck__(cls, subclass: type[Any]) -> bool:
                return issubclass(subclass, original)

        class GuardedConstructor(metaclass=GuardedConstructorMeta):
            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                guard._violate(qualified_name)

        GuardedConstructor.__name__ = original.__name__
        GuardedConstructor.__qualname__ = original.__qualname__
        GuardedConstructor.__module__ = original.__module__
        return GuardedConstructor

    def install(self) -> None:
        for module, names, prefix in (
            (self.numpy.random, self.NUMPY_CALLS, "numpy.random"),
            (random, self.PYTHON_CALLS, "random"),
        ):
            missing = [name for name in names if not callable(getattr(module, name, None))]
            if missing:
                raise ChildControllerError(
                    "frozen RNG guard manifest mismatch: " + ",".join(missing)
                )
            for name in names:
                original = getattr(module, name)
                qualified_name = f"{prefix}.{name}"
                replacement = (
                    self._constructor(qualified_name, original)
                    if isinstance(original, type)
                    else self._wrapper(qualified_name, original)
                )
                self._replacements[id(original)] = replacement
                setattr(module, name, replacement)
        for loaded_module in tuple(sys.modules.values()):
            namespace = getattr(loaded_module, "__dict__", None)
            if not isinstance(namespace, dict):
                continue
            for name, value in tuple(namespace.items()):
                replacement = self._replacements.get(id(value))
                if replacement is not None:
                    namespace[name] = replacement

    def guarded_alias(self, original: Any) -> Any:
        replacement = self._replacements.get(id(original))
        if replacement is None:
            raise ChildControllerError("cached RNG alias was not covered by the guard")
        return replacement

    def state_commitments(self) -> dict[str, str]:
        if self._checkpoint:
            raise _BootstrapRngViolation("nested RNG audit checkpoint")
        self._checkpoint = True
        try:
            numpy_state = self.numpy.random.get_state()
            python_state = random.getstate()
        finally:
            self._checkpoint = False
        return {
            "numpy_legacy_sha256": _numpy_commitment(self.numpy, numpy_state),
            "python_random_sha256": _python_commitment(python_state),
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _numpy_commitment(numpy: Any, state: tuple[Any, ...]) -> str:
    bit_generator, key, position, has_gauss, cached_gaussian = state
    key_array = numpy.asarray(key)
    if key_array.ndim != 1 or key_array.dtype.kind != "u":
        raise ChildControllerError("unexpected NumPy legacy RNG key")
    normalized = numpy.ascontiguousarray(key_array.astype("<u4", copy=False))
    cached = numpy.float64(cached_gaussian)
    if not numpy.isfinite(cached):
        raise ChildControllerError("non-finite NumPy legacy cached Gaussian")
    payload = {
        "bit_generator": str(bit_generator),
        "cached_gaussian_hex": float(cached).hex().lower(),
        "has_gauss": int(has_gauss),
        "key_c_order_hex": normalized.tobytes(order="C").hex(),
        "key_dtype": normalized.dtype.str,
        "key_shape": list(normalized.shape),
        "position": int(position),
    }
    return hashlib.sha256(
        b"operatormark-numpy-legacy-rng-state-v1\n" + _canonical_json(payload)
    ).hexdigest()


def _python_commitment(state: tuple[Any, ...]) -> str:
    version, internal_state, gauss_next = state
    if not isinstance(internal_state, tuple) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in internal_state
    ):
        raise ChildControllerError("unexpected Python random state")
    payload = {
        "gauss_next": None if gauss_next is None else float(gauss_next).hex().lower(),
        "internal_state": list(internal_state),
        "version": int(version),
    }
    return hashlib.sha256(
        b"operatormark-python-random-state-v1\n" + _canonical_json(payload)
    ).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_base_import_closure(numpy: Any) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    names = sorted(
        name for name in sys.modules
        if name == "numpy" or name.startswith("numpy.")
        or name == "scipy" or name.startswith("scipy.")
    )
    for name in names:
        module = sys.modules[name]
        source = getattr(module, "__file__", None)
        if isinstance(source, str):
            path = pathlib.Path(source).resolve(strict=True)
            if path.suffix.casefold() in {".pyc", ".pyo"}:
                try:
                    path = pathlib.Path(importlib.util.source_from_cache(str(path))).resolve(
                        strict=True
                    )
                except (ValueError, OSError) as exc:
                    raise ChildControllerError(
                        f"numerical module has bytecode but no source bytes: {name}"
                    ) from exc
            suffix = path.suffix.casefold()
            if suffix in {".py", ".pyi"}:
                kind = "source_module"
            elif suffix in {".pyd", ".so", ".dll", ".dylib"}:
                kind = "extension_binary"
            else:
                raise ChildControllerError(
                    f"numerical module has unsupported byte identity: {name}"
                )
            records.append({
                "file_sha256": _file_sha256(path), "kind": kind, "module": name,
                "suffix": "".join(path.suffixes).lower(),
            })
        elif getattr(getattr(module, "__spec__", None), "submodule_search_locations", None) is not None:
            records.append({
                "file_sha256": None, "kind": "namespace_package_no_module_bytes",
                "module": name, "suffix": "",
            })
        else:
            raise ChildControllerError(f"numerical module has no commit-able identity: {name}")
    payload = {
        "numpy_version": numpy.__version__, "ordered_modules": records,
        "root_modules": list(BASE_IMPORT_ROOTS),
        "schema": "operatormark.darcy_branch_c_v1.base_import_closure.v1",
        "scipy_version": sys.modules["scipy"].__version__,
    }
    return {**payload, "closure_sha256": hashlib.sha256(_canonical_json(payload)).hexdigest()}


class ChildControllerError(RuntimeError):
    """Raised when a fresh child cannot remain within its authorized action."""


def _bootstrap_context() -> ModuleType:
    context = sys.modules.get(BOOTSTRAP_MODULE)
    if not isinstance(context, ModuleType) or getattr(context, "sentinel", None) != BOOTSTRAP_SENTINEL:
        raise ChildControllerError("child controller requires the guarded fresh-child bootstrap")
    required_attributes = {
        "required_environment", "base_import_roots", "base_numerical_modules",
        "base_import_closure", "baseline_rng_commitments",
        "after_guard_rng_commitments", "rng_guard", "rng_violation_type",
        "child_wall_start_ns", "cached_numpy_random", "action", "state",
        "authorization_file_sha256", "implementation_commit", "implementation_tree",
    }
    if any(not hasattr(context, name) for name in required_attributes):
        raise ChildControllerError("fresh-child bootstrap context is incomplete")
    if tuple(context.base_import_roots) != BASE_IMPORT_ROOTS:
        raise ChildControllerError("bootstrap base-import roots differ from controller roots")
    bound_environment = {
        "action": os.environ.get("OPERATORMARK_ACTION", ""),
        "state": os.environ.get("OPERATORMARK_STATE", ""),
        "authorization_file_sha256": os.environ.get(
            "OPERATORMARK_AUTHORIZATION_FILE_SHA256", ""
        ),
        "implementation_commit": os.environ.get("OPERATORMARK_IMPLEMENTATION_COMMIT", ""),
        "implementation_tree": os.environ.get("OPERATORMARK_IMPLEMENTATION_TREE", ""),
    }
    if any(getattr(context, name) != value for name, value in bound_environment.items()):
        raise ChildControllerError("bootstrap identity differs from the child environment")
    return context


def _loaded_numerical_module_names() -> tuple[str, ...]:
    return tuple(sorted(
        name for name in sys.modules
        if name == "numpy" or name.startswith("numpy.")
        or name == "scipy" or name.startswith("scipy.")
    ))


def _require_rng_equal(
    observed: Mapping[str, str], baseline: Mapping[str, str], *, checkpoint: str,
) -> None:
    if observed != baseline:
        raise ChildControllerError(f"RNG state changed at {checkpoint}")


def _state2_rng_probe(context: ModuleType, arguments: Sequence[str]) -> dict[str, Any]:
    if len(arguments) != 1:
        raise ChildControllerError("State 2 RNG probe requires exactly one frozen probe name")
    numpy = sys.modules["numpy"]
    random = sys.modules["random"]
    probes = {
        "cached_numpy_random": lambda: context.cached_numpy_random(),
        "dynamic_numpy_uniform": lambda: getattr(numpy.random, "uniform")(),
        "helper_numpy_default_rng": lambda: _helper_default_rng(numpy),
        "numpy_randomstate_constructor": lambda: numpy.random.RandomState(),
        "python_random_constructor": lambda: random.Random(),
    }
    name = arguments[0]
    if name not in probes:
        raise ChildControllerError("State 2 RNG probe name is not frozen")
    try:
        probes[name]()
    except context.rng_violation_type as exc:
        return {
            "blocked_probe": name,
            "denial": str(exc),
            "forbidden_rng_call_count": context.rng_guard.forbidden_rng_call_count,
            "outcome": "expected_fail_closed",
            "schema": "operatormark.darcy_branch_c_v1.state2_rng_subprocess_probe.v1",
        }
    raise ChildControllerError("forbidden RNG subprocess probe did not fail closed")


def _helper_default_rng(numpy: Any) -> object:
    return numpy.random.default_rng()


def _state3_capability() -> Any:
    from .microbenchmark import VerifiedState3Capability

    return VerifiedState3Capability(
        authorization_file_sha256=os.environ.get("OPERATORMARK_AUTHORIZATION_FILE_SHA256", ""),
        implementation_commit=os.environ.get("OPERATORMARK_IMPLEMENTATION_COMMIT", ""),
        implementation_tree=os.environ.get("OPERATORMARK_IMPLEMENTATION_TREE", ""),
    )


def _write_canonical_stdout(record: Mapping[str, Any]) -> None:
    from .canonical import canonical_json_bytes

    sys.stdout.write(canonical_json_bytes(record).decode("utf-8"))


def _bound_git_output(source_root: pathlib.Path, *arguments: str) -> str:
    import subprocess

    result = subprocess.run(
        ["git", *arguments], cwd=source_root, check=False,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="strict",
    )
    if result.returncode != 0 or result.stderr:
        raise ChildControllerError(
            "State 3 environment probe Git metadata query failed: "
            + " ".join(arguments)
        )
    return result.stdout.rstrip("\r\n")


def _verify_state3_probe_git_identity(source_root: pathlib.Path, context: ModuleType) -> str:
    if _bound_git_output(source_root, "symbolic-ref", "--quiet", "--short", "HEAD") != "main":
        raise ChildControllerError("State 3 environment probe branch is not main")
    if _bound_git_output(source_root, "rev-parse", "HEAD") != context.implementation_commit:
        raise ChildControllerError("State 3 environment probe implementation commit mismatch")
    if _bound_git_output(source_root, "rev-parse", "HEAD^{tree}") != context.implementation_tree:
        raise ChildControllerError("State 3 environment probe implementation tree mismatch")
    for arguments in (("diff", "--quiet", "--"), ("diff", "--cached", "--quiet", "--")):
        import subprocess

        result = subprocess.run(
            ["git", *arguments], cwd=source_root, check=False,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode != 0 or result.stdout or result.stderr:
            raise ChildControllerError("State 3 environment probe found a tracked Git change")
    if _bound_git_output(
        source_root, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise ChildControllerError("State 3 environment probe found a non-clean Git status")
    lock_blob = _bound_git_output(source_root, "rev-parse", "HEAD:requirements-lock-cpu.txt")
    observed_blob = _bound_git_output(
        source_root, "hash-object", "--no-filters", "--", "requirements-lock-cpu.txt"
    )
    if lock_blob != observed_blob:
        raise ChildControllerError("State 3 environment probe dependency-lock blob mismatch")
    return lock_blob


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    action = os.environ.get("OPERATORMARK_ACTION", "")
    state = os.environ.get("OPERATORMARK_STATE", "")
    context = _bootstrap_context()
    if _loaded_numerical_module_names() != tuple(context.base_numerical_modules):
        raise ChildControllerError("numerical import closure changed before controller verification")

    from .base_imports import (
        BASE_IMPORT_ROOTS as COMMITTED_BASE_IMPORT_ROOTS,
        verify_base_import_closure_equal,
    )

    if BASE_IMPORT_ROOTS != COMMITTED_BASE_IMPORT_ROOTS:
        raise ChildControllerError("controller base-import roots differ from committed roots")
    base_closure = verify_base_import_closure_equal(context.base_import_closure)
    baseline = dict(context.baseline_rng_commitments)
    after_guard = dict(context.after_guard_rng_commitments)
    _require_rng_equal(after_guard, baseline, checkpoint="after_guard_install")

    if action == "state2-rng-adversarial-probe" and state == "state2":
        result = _state2_rng_probe(context, arguments)
        after_project = context.rng_guard.state_commitments()
        _require_rng_equal(after_project, baseline, checkpoint="after_project_import")
        verify_base_import_closure_equal(base_closure)
        result.update({
            "forbidden_corpus_seed_access_count": 0,
            "rng_state_equal_after_guard": True,
            "rng_state_equal_after_action": True,
        })
        _write_canonical_stdout(result)
        return 0

    if action == "state2-environment-probe" and state == "state2":
        if arguments:
            raise ChildControllerError("State 2 environment probe accepts no arguments")
        from .runtime import (
            build_review_only_thread_policy_candidate,
            inspect_loaded_numerical_runtime,
            verify_runtime_record_equal,
        )

        runtime = inspect_loaded_numerical_runtime(
            authorization_file_sha256=os.environ["OPERATORMARK_AUTHORIZATION_FILE_SHA256"],
            required_environment=context.required_environment,
        )
        candidate = build_review_only_thread_policy_candidate(
            runtime_record=runtime, base_import_closure=base_closure
        )
        after_project = context.rng_guard.state_commitments()
        _require_rng_equal(after_project, baseline, checkpoint="after_project_import")
        verify_base_import_closure_equal(base_closure)
        verify_runtime_record_equal(
            runtime,
            authorization_file_sha256=os.environ["OPERATORMARK_AUTHORIZATION_FILE_SHA256"],
            required_environment=context.required_environment,
        )
        after_action = context.rng_guard.state_commitments()
        _require_rng_equal(after_action, baseline, checkpoint="after_execution")
        result = {
            "base_import_closure": base_closure,
            "forbidden_corpus_seed_access_count": 0,
            "forbidden_rng_call_count": context.rng_guard.forbidden_rng_call_count,
            "review_only_thread_policy_candidate": candidate,
            "rng_state_commitments": after_guard,
            "rng_state_equal_after_action": True,
            "rng_state_equal_after_guard": True,
            "runtime": runtime,
            "schema": "operatormark.darcy_branch_c_v1.state2_environment_probe.v2",
        }
        if result["forbidden_rng_call_count"] != 0:
            raise ChildControllerError("environment probe recorded a forbidden RNG call")
        _write_canonical_stdout(result)
        return 0

    if action == "state3-environment-probe" and state == "state3":
        if arguments:
            raise ChildControllerError("State 3 environment probe accepts no arguments")
        import locale
        import platform

        from .runtime import inspect_loaded_numerical_runtime, verify_runtime_record_equal
        from .state3_audit import build_environment_runtime_record

        source_root = pathlib.Path(os.environ["OPERATORMARK_SOURCE_ROOT"]).resolve(strict=True)
        lock_blob = _verify_state3_probe_git_identity(source_root, context)
        runtime = inspect_loaded_numerical_runtime(
            authorization_file_sha256=context.authorization_file_sha256,
            required_environment=context.required_environment,
        )
        if runtime["all_runtime_requirements_satisfied"] is not True:
            raise ChildControllerError("State 3 environment probe runtime is not fully bound")
        verify_base_import_closure_equal(base_closure)
        after_project = context.rng_guard.state_commitments()
        _require_rng_equal(after_project, baseline, checkpoint="after_project_import")
        verify_runtime_record_equal(
            runtime,
            authorization_file_sha256=context.authorization_file_sha256,
            required_environment=context.required_environment,
        )
        launcher_path = source_root / "operatormark" / "corpus" / "launcher.py"
        runtime_path = source_root / "operatormark" / "corpus" / "runtime.py"
        lock_path = source_root / "requirements-lock-cpu.txt"
        result = build_environment_runtime_record(
            protocol_file_sha256=PROTOCOL_SHA256,
            state3_authorization_file_sha256=context.authorization_file_sha256,
            implementation_commit=context.implementation_commit,
            implementation_tree=context.implementation_tree,
            parent_environment=context.required_environment,
            launcher_file_sha256=_file_sha256(launcher_path.resolve(strict=True)),
            runtime_inspector_file_sha256=_file_sha256(runtime_path.resolve(strict=True)),
            requirements_lock_git_blob=lock_blob,
            requirements_lock_file_sha256=_file_sha256(lock_path.resolve(strict=True)),
            base_import_closure=base_closure,
            numerical_runtime=runtime,
            python_version=platform.python_version(),
            numpy_version=sys.modules["numpy"].__version__,
            scipy_version=sys.modules["scipy"].__version__,
            operating_system=platform.platform(),
            architecture=platform.machine(),
            locale=locale.setlocale(locale.LC_ALL, None),
        )
        verify_base_import_closure_equal(base_closure)
        verify_runtime_record_equal(
            runtime,
            authorization_file_sha256=context.authorization_file_sha256,
            required_environment=context.required_environment,
        )
        after_action = context.rng_guard.state_commitments()
        _require_rng_equal(after_action, baseline, checkpoint="after_execution")
        if context.rng_guard.forbidden_rng_call_count != 0:
            raise ChildControllerError("State 3 environment probe recorded a forbidden RNG call")
        _write_canonical_stdout(result)
        return 0

    if action == "state3-microbenchmark-worker" and state == "state3":
        if len(arguments) != 2:
            raise ChildControllerError("State 3 worker requires fixture and repeat only")
        fixture = arguments[0]
        try:
            repeat_index = int(arguments[1], 10)
        except ValueError as exc:
            raise ChildControllerError("State 3 worker repeat is not a decimal integer") from exc
        if str(repeat_index) != arguments[1]:
            raise ChildControllerError("State 3 worker repeat is not canonically encoded")

        from .microbenchmark import (
            WorkerAuditContext,
            _real_fixture_operations,
            measure_one_fixed_fixture,
            windows_process_memory,
        )
        from .rng_guard import CorpusSeedAccessGuard
        from .runtime import inspect_loaded_numerical_runtime, verify_runtime_record_equal

        # Import every project/scientific helper before the post-project checkpoint.
        operations = _real_fixture_operations()
        verify_base_import_closure_equal(base_closure)
        after_project = context.rng_guard.state_commitments()
        _require_rng_equal(after_project, baseline, checkpoint="after_project_import")
        runtime = inspect_loaded_numerical_runtime(
            authorization_file_sha256=os.environ["OPERATORMARK_AUTHORIZATION_FILE_SHA256"],
            required_environment=context.required_environment,
        )
        if not runtime["all_runtime_requirements_satisfied"]:
            raise ChildControllerError("State 3 worker numerical runtime is not fully bound")
        verify_runtime_record_equal(
            runtime,
            authorization_file_sha256=os.environ["OPERATORMARK_AUTHORIZATION_FILE_SHA256"],
            required_environment=context.required_environment,
        )
        seed_guard = CorpusSeedAccessGuard()
        audit_context = WorkerAuditContext(
            child_wall_start_ns=context.child_wall_start_ns,
            baseline_rng_commitments=baseline,
            after_guard_rng_commitments=after_guard,
            after_project_rng_commitments=after_project,
            rng_checkpoint=context.rng_guard.state_commitments,
            expected_base_import_closure=base_closure,
            reinspect_base_import_closure=lambda: verify_base_import_closure_equal(base_closure),
            expected_runtime=runtime,
            reinspect_runtime=lambda: verify_runtime_record_equal(
                runtime,
                authorization_file_sha256=os.environ["OPERATORMARK_AUTHORIZATION_FILE_SHA256"],
                required_environment=context.required_environment,
            ),
            forbidden_rng_call_count=lambda: context.rng_guard.forbidden_rng_call_count,
            forbidden_corpus_seed_access_count=lambda: seed_guard.forbidden_corpus_seed_access_count,
            memory_probe=windows_process_memory,
        )
        result = measure_one_fixed_fixture(
            fixture, repeat_index, capability=_state3_capability(),
            audit_context=audit_context, operations=operations,
        )
        _write_canonical_stdout(result)
        return 0

    if action == "state3-microbenchmark" and state == "state3":
        if len(arguments) != 1:
            raise ChildControllerError("State 3 coordinator requires one metadata-only bindings path")
        from .microbenchmark import (
            build_measurement_bundle,
            load_compute_feasibility_preflight_bindings,
            run_compute_feasibility_measurements,
        )
        from .runtime import inspect_loaded_numerical_runtime, verify_runtime_record_equal

        preflight_bindings = load_compute_feasibility_preflight_bindings(arguments[0])
        verify_base_import_closure_equal(base_closure)
        after_project = context.rng_guard.state_commitments()
        _require_rng_equal(after_project, baseline, checkpoint="after_project_import")
        runtime = inspect_loaded_numerical_runtime(
            authorization_file_sha256=os.environ["OPERATORMARK_AUTHORIZATION_FILE_SHA256"],
            required_environment=context.required_environment,
        )
        if not runtime["all_runtime_requirements_satisfied"]:
            raise ChildControllerError("State 3 coordinator numerical runtime is not fully bound")
        verify_runtime_record_equal(
            runtime,
            authorization_file_sha256=os.environ["OPERATORMARK_AUTHORIZATION_FILE_SHA256"],
            required_environment=context.required_environment,
        )
        raw_measurements = run_compute_feasibility_measurements(
            _state3_capability(), preflight_bindings
        )
        verify_base_import_closure_equal(base_closure)
        verify_runtime_record_equal(
            runtime,
            authorization_file_sha256=os.environ["OPERATORMARK_AUTHORIZATION_FILE_SHA256"],
            required_environment=context.required_environment,
        )
        after_action = context.rng_guard.state_commitments()
        _require_rng_equal(after_action, baseline, checkpoint="after_execution")
        if context.rng_guard.forbidden_rng_call_count != 0:
            raise ChildControllerError("State 3 coordinator recorded a forbidden RNG call")
        _write_canonical_stdout(
            build_measurement_bundle(preflight_bindings, raw_measurements)
        )
        return 0

    raise ChildControllerError("child action is unavailable or lacks separate authorization")


def _run_fresh_child_bootstrap() -> int:
    """Script entry: establish every pre-project invariant, then import controller."""

    if os.environ.get("OPERATORMARK_STATE") in {"state4", "state5", "state6"}:
        raise ChildControllerError("production later-state bootstrap is disabled")

    if any(
        name == "numpy" or name.startswith("numpy.")
        or name == "scipy" or name.startswith("scipy.")
        for name in sys.modules
    ):
        raise ChildControllerError("numerical module loaded before child environment gate")
    try:
        required = json.loads(os.environ["OPERATORMARK_REQUIRED_ENVIRONMENT_JSON"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ChildControllerError("missing or invalid frozen child environment") from exc
    if not isinstance(required, dict) or not all(
        isinstance(name, str) and isinstance(value, str) and value
        for name, value in required.items()
    ):
        raise ChildControllerError("invalid frozen child environment map")
    if any(os.environ.get(name) != value for name, value in required.items()):
        raise ChildControllerError("child environment differs before numerical import")
    if os.environ.get("OPERATORMARK_PROTOCOL_SHA256") != PROTOCOL_SHA256:
        raise ChildControllerError("protocol commitment mismatch before numerical import")
    action = os.environ.get("OPERATORMARK_ACTION", "")
    state = os.environ.get("OPERATORMARK_STATE", "")
    allowed = {
        "state2-environment-probe": "state2",
        "state2-rng-adversarial-probe": "state2",
        "state3-environment-probe": "state3",
        "state3-microbenchmark": "state3",
        "state3-microbenchmark-worker": "state3",
        "state4-prepare": "state4",
        "state5-solve": "state5",
        "state6-seal": "state6",
    }
    if allowed.get(action) != state:
        raise ChildControllerError("child action/state mismatch before numerical import")
    if action == "state3-microbenchmark-worker":
        if os.environ.get("OPERATORMARK_INTERNAL_WORKER") != "1":
            raise ChildControllerError("State 3 worker lacks coordinator-only dispatch binding")
    elif "OPERATORMARK_INTERNAL_WORKER" in os.environ:
        raise ChildControllerError("unexpected internal-worker dispatch binding")
    source_root = pathlib.Path(os.environ.get("OPERATORMARK_SOURCE_ROOT", "")).resolve(
        strict=True
    )
    controller_path = source_root / "operatormark" / "corpus" / "child_controller.py"
    if not controller_path.is_file() or controller_path.resolve(strict=True) != pathlib.Path(
        __file__
    ).resolve(strict=True):
        raise ChildControllerError("bound OperatorMark controller path is unavailable or changed")

    child_wall_start_ns = time.perf_counter_ns()
    for module_name in BASE_IMPORT_ROOTS:
        importlib.import_module(module_name)
    numpy = sys.modules["numpy"]
    cached_numpy_random = numpy.random.random
    raw_numpy_state = numpy.random.get_state()
    raw_python_state = random.getstate()
    baseline = {
        "numpy_legacy_sha256": _numpy_commitment(numpy, raw_numpy_state),
        "python_random_sha256": _python_commitment(raw_python_state),
    }
    closure = _capture_base_import_closure(numpy)
    numerical_names = _loaded_numerical_module_names()
    guard = _BootstrapRngGuard(numpy)
    guard.install()
    after_guard = guard.state_commitments()
    _require_rng_equal(after_guard, baseline, checkpoint="after_guard_install")

    context = types.ModuleType(BOOTSTRAP_MODULE)
    context.sentinel = BOOTSTRAP_SENTINEL
    context.action = action
    context.state = state
    context.authorization_file_sha256 = os.environ.get(
        "OPERATORMARK_AUTHORIZATION_FILE_SHA256", ""
    )
    context.implementation_commit = os.environ.get("OPERATORMARK_IMPLEMENTATION_COMMIT", "")
    context.implementation_tree = os.environ.get("OPERATORMARK_IMPLEMENTATION_TREE", "")
    context.required_environment = required
    context.base_import_roots = BASE_IMPORT_ROOTS
    context.base_numerical_modules = numerical_names
    context.base_import_closure = closure
    context.baseline_rng_commitments = baseline
    context.after_guard_rng_commitments = after_guard
    context.rng_guard = guard
    context.rng_violation_type = _BootstrapRngViolation
    context.child_wall_start_ns = child_wall_start_ns
    context.cached_numpy_random = guard.guarded_alias(cached_numpy_random)
    sys.modules[BOOTSTRAP_MODULE] = context

    sys.path.insert(0, str(source_root))
    controller = importlib.import_module("operatormark.corpus.child_controller")
    return int(controller.main(sys.argv[1:]))


def run_bound_synthetic_child(*, stream, bootstrap_inputs, descriptor, preloaded, mode=None):
    """Called only by a source-bound external bootstrap, never the legacy route.

    Its independently fixed input object is not derived from capsule contents.
    State 2/3 bootstrap, callbacks and latch semantics are unchanged.
    """
    if mode != "synthetic":
        raise ChildControllerError("production bound controller is disabled")
    from operatormark.corpus.state_execution import execute_bound_capsule
    return execute_bound_capsule(stream=stream, bootstrap_inputs=bootstrap_inputs,
                                 descriptor=descriptor, preloaded=preloaded, mode=mode)


def dispatch_bound_workflow(*, stream, bootstrap_inputs, descriptor, preloaded,
                            expected, independent_inputs, ports, mode):
    """Explicit synthetic held-byte function dispatch; never an environment route."""
    if mode != 'synthetic' or type(mode) is not str:
        raise ChildControllerError('production workflow dispatch is disabled')
    from types import FunctionType
    from .state_execution import decode_capsule
    from .source_import_guard import install_bound_project_imports, validate_bound_import_trace
    from .state4_workflow import WorkflowResult, validate_workflow_inputs
    validate_workflow_inputs(expected=expected, independent_inputs=independent_inputs, mode=mode)
    routes = {
        ('state4-controls', 'state4_worker'): 'run_state4_controls',
        ('state5-members', 'state5_worker'): 'run_state5_members',
        ('state6-verification', 'state6_verifier'): 'run_state6_verification',
        ('state6-finalization', 'state6_finalizer'): 'run_state6_finalizer',
    }
    key = (bootstrap_inputs.operation, bootstrap_inputs.role)
    if key not in routes or bootstrap_inputs.source_expectations != expected.launch.source_expectations:
        raise ChildControllerError('dispatch source/operation/role is not independently bound')
    verified = decode_capsule(stream=stream, bootstrap_inputs=bootstrap_inputs, descriptor=descriptor)
    session = install_bound_project_imports(verified, expected=bootstrap_inputs.source_expectations, preloaded=preloaded)
    primary = None
    try:
        module = session.import_entry()
        function = vars(module).get(routes[key])
        if type(function) is not FunctionType or function.__globals__ is not vars(module):
            raise ChildControllerError('bound entry must define the fixed workflow callable')
        result = function(expected=expected, independent_inputs=independent_inputs, ports=ports, mode=mode)
        if type(result) is not WorkflowResult or result.production_accepted is not False or result.execution_authorized is not False:
            raise ChildControllerError('non-authorizing exact workflow result required')
        validate_bound_import_trace(session.trace(), session=session)
        return result
    except BaseException as exc:
        primary = exc
        try:
            validate_bound_import_trace(session.trace(), session=session)
        except BaseException as trace_error:
            failures = BaseException.__dict__['__dict__'].__get__(exc, type(exc))
            failures['_workflow_trace_failures'] = (trace_error,)
        raise
    finally:
        try:
            session.close()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            failures = BaseException.__dict__['__dict__'].__get__(primary, type(primary))
            failures['_workflow_cleanup_failures'] = (cleanup_error,)


if __name__ == "__main__":
    raise SystemExit(_run_fresh_child_bootstrap())
