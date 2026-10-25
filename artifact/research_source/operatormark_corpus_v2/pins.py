"""Dependency binding: the frozen v1 sources and the accepted Drop 1 package, by SHA-256.

Drop 2 reuses Drop 1 unchanged.  These are the correction-1 hashes from
``DROP1_MANIFEST.json`` (the package Windows Q0 accepted).  Generation and
validation refuse to run if any imported Drop 1 file differs.  Standard
library only.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

DROP1_PINS: dict[str, str] = {
    "operatormark_numerics_v2/__init__.py": "85f6d689bb173bb535922c782cca0b2dc218df2984c7d54d6e0c7e8e3a63a8ca",
    "operatormark_numerics_v2/api.py": "df219fb633bb6d16d3ded3b0f8b79265de5fc6de757d6fb26dbdcad75e5311ef",
    "operatormark_numerics_v2/directions.py": "5428068cff57767bb5757b5c484638a282a95045afb5ff7fdc6ddcc65465dcb3",
    "operatormark_numerics_v2/ladder.py": "5cfeb63bd67ffe7a8e6c2aec2c1e086dd91b35324e151546e7fa62dc34907052",
    "operatormark_numerics_v2/lock.py": "9603ad42cf61fbf1ccdeaa0fbf962bc7517779855fea3beefa0eeed7b2fcd77a",
    "operatormark_numerics_v2/p1_family.py": "ad8eff0754015bec1f89e4d875211d99ba19be3484b51d09753c338ba217c0e8",
    "operatormark_numerics_v2/payload.py": "2b99bd7a1b99753ca165bddbe9784e844932fd7229dd267cccb34af120665212",
    "operatormark_numerics_v2/pins.py": "e3debed9c38eb9fb4539504858e9a9d8b7d1233de7a2b536f428e953944d2e00",
    "operatormark_numerics_v2/resources.py": "16e186d570cca55a59a22601bbe10effb8bc67d8bf3e3fc824568e488acdb995",
    "operatormark_numerics_v2/scope.py": "72abe66eb9eb9e2423f03e31150f96dc443b8de5800e374147d4059cf8d3ca2e",
    "operatormark_numerics_v2/tpfa_family.py": "e800e91e39863eb00569646431ffadecbc3a5ac1b18195c09993a5fad3a01bc9",
    "operatormark_numerics_v2/views.py": "2f95a0e662c8f165f6c5f7f42078d146d6238b2107145b2cadd791ac268a3650",
}


class DependencyError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_parent(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise DependencyError(f"package {name!r} is not importable")
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


def verify_dependencies() -> dict:
    """Check the frozen pins (via Drop 1's own verifier) and the Drop 1 pins; return both identity maps."""

    from operatormark_numerics_v2 import pins as drop1_pins

    try:
        frozen = drop1_pins.verify_frozen_sources()
    except drop1_pins.FrozenSourceError as exc:
        raise DependencyError(str(exc)) from None
    root = _package_parent("operatormark_numerics_v2")
    observed, problems = {}, []
    for relative, expected in sorted(DROP1_PINS.items()):
        path = root / relative
        if not path.is_file():
            problems.append(f"missing Drop 1 file {relative}")
            continue
        observed[relative] = _sha(path)
        if observed[relative] != expected:
            problems.append(f"Drop 1 file differs from the accepted correction-1 bytes: {relative}")
    if problems:
        raise DependencyError("; ".join(problems))
    return {"frozen": frozen, "drop1": observed}


def drop2_identity() -> dict:
    root = Path(__file__).resolve().parent
    return {f"operatormark_corpus_v2/{p.relative_to(root).as_posix()}": _sha(p)
            for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts}
