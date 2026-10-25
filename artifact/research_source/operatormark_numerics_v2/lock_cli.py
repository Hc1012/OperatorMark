"""Lock-builder entry point.  Verifies the frozen-source pins (standard library
only) before NumPy or any frozen module is imported, then builds the lock.

    python -m operatormark_numerics_v2.lock_cli build --seed-file SEEDS.json --out DIR

Building the real lock (formal seeds) is a separate approval and happens on the
owner's machine only.
"""

from __future__ import annotations

from . import pins


def run(argv: list[str] | None = None) -> int:
    pins.verify_frozen_sources()
    from . import lock
    pins.verify_loaded_modules()
    return lock.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
