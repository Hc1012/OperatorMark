"""Hash bindings for the frozen sources this package depends on.

Standard library only.  The pins are the SHA-256 values recorded in the
friend-source bundle MANIFEST (source bytes equal bundle bytes, no
redaction).  ``verify_frozen_sources`` must pass before any test or runner
uses the frozen modules.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

# Paths are relative to the directory that contains the ``operatormark`` package.
FROZEN_MODULE_PINS: dict[str, str] = {
    "operatormark/__init__.py": "91447944015cec709e8aa7655f7e9d64e1e4508e7023a57fe3746911c0fc6fed",
    "operatormark/solvers/__init__.py": "db12bc2a9d9865a25a34ddabec3594026bea637ddd87e6eeeb723cd113718c8f",
    "operatormark/solvers/darcy_tpfa.py": "f40897db7974c8ef6cc043ed01c701cdfa467af102eae9db702ffd22ba302aa1",
    "operatormark/solvers/darcy_p1.py": "6d97cbd7349b6f665496d452d436d1c3b0684f8dd07897baaf27fda6f9818a6d",
    "operatormark/benchmarks/__init__.py": "26f0dbfd42d2ec0cea538d942a672e103d8bd93a9326bad54b183dd27cbeb431",
    "operatormark/benchmarks/darcy_spec.py": "03253a6a09d259ed0bfc8ba87c3ded12965d77c934548a928b5edc8e8f5d1c51",
    "operatormark/corpus/__init__.py": "b5132f0d992dc42c4ccfb7687dd2d387dbdcec2511c0ec809faa17a85c10bd0c",
    "operatormark/corpus/canonical.py": "7c90315043028e99924cf55074f98622b55871b1c5d4a17d39e5ee2747126d8b",
}

# Test-only frozen inputs (not imported by the package at run time).
FROZEN_TEST_PINS: dict[str, str] = {
    "operatormark/corpus/final_verify.py": "07e48b132999d340a7e04ee875cc09057efca095be49c60479ed3776846803ea",
    "operatormark/corpus/final_scientific_verify.py": "7fda0dbc8580fe75a93ef6d2f5dd7ec2e36ebb12e32fa68f43e0a4ae7c567ad0",
    "historical_v1/fixed-numerical-summary.jsonl": "a9eedda32941fed43a25b3ca8b560a6f2c646ed36bbfbce56e60fac610a8ba9a",
}

# Named in the bundle's LOCAL_IMPORTS / MANIFEST as not bundled.  Drop 1 does
# not reconstruct them; their absence is reported, never worked around.
KNOWN_MISSING_DEPENDENCIES: tuple[str, ...] = (
    "operatormark/corpus/runtime.py",
    "operatormark/corpus/base_imports.py",
    "operatormark/corpus/final_evidence_verify.py",
    "operatormark/corpus/state3_audit_verify.py",
)


class FrozenSourceError(RuntimeError):
    """A frozen dependency is missing or its bytes differ from the pin."""


def frozen_root() -> Path:
    """Directory containing the ``operatormark`` package, located without importing it."""

    spec = importlib.util.find_spec("operatormark")
    if spec is None or not spec.submodule_search_locations:
        raise FrozenSourceError("frozen package 'operatormark' is not importable")
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_sources(*, include_test_pins: bool = False, root: Path | None = None,
                          locations: dict[str, Path] | None = None) -> dict[str, str]:
    """Return {relative path: sha256} after checking every pin; raise on any difference.

    ``locations`` may relocate individual pinned files (e.g. the historical v1
    summary, which lives outside the package tree in the repository); the
    content must still match the pin.
    """

    base = frozen_root() if root is None else Path(root)
    pins = dict(FROZEN_MODULE_PINS)
    if include_test_pins:
        pins.update(FROZEN_TEST_PINS)
    observed: dict[str, str] = {}
    problems: list[str] = []
    for relative, expected in sorted(pins.items()):
        path = Path(locations[relative]) if locations and relative in locations else base / relative
        if not path.is_file():
            problems.append(f"missing frozen file: {relative}")
            continue
        actual = file_sha256(path)
        observed[relative] = actual
        if actual != expected:
            problems.append(f"hash mismatch: {relative} expected {expected} got {actual}")
    if problems:
        raise FrozenSourceError("; ".join(problems))
    return observed


def missing_dependency_report(root: Path | None = None) -> dict[str, bool]:
    base = frozen_root() if root is None else Path(root)
    return {relative: (base / relative).is_file() for relative in KNOWN_MISSING_DEPENDENCIES}


def verify_loaded_modules(root: Path | None = None) -> None:
    """After import: every loaded frozen module must come from its pinned file."""

    import sys

    base = frozen_root() if root is None else Path(root)
    for relative in FROZEN_MODULE_PINS:
        module_name = relative[:-3].replace("/", ".")
        if module_name.endswith(".__init__"):
            module_name = module_name[: -len(".__init__")]
        module = sys.modules.get(module_name)
        if module is None:
            continue
        loaded = Path(getattr(module, "__file__", "") or "").resolve()
        if loaded != (base / relative).resolve():
            raise FrozenSourceError(f"{module_name} loaded from {loaded}, not the pinned file")
