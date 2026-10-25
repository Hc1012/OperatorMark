"""Runner entry point.

    python -m operatormark_numerics_v2.runners.cli q1 --lock DIR --out DIR [--prior q1=DIR ...]
           [--budgets FILE] [--time-cap-s S --mem-cap-bytes B --cap-override] [--test-mode]
           [--test-delay-s X] [--only-instance ID]

Order of checks (fail closed, exit code 3):
  1. the v1 §7.3 environment, before any numerical import;
  2. the frozen-source pins (standard library only), before the frozen modules are imported;
  3. after import: the loaded frozen modules come from the pinned files; in formal mode the exact
     Python/NumPy/SciPy versions, the confirmed budgets file, and the revision 3 stage caps
     (other caps only with --cap-override, recorded in the run record).
Exit code 4 means the stage stopped incomplete (cap, selection).  Running a formal stage is
not authorized by Drop 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import pins
from .envcheck import check_environment

REQUIRED_VERSIONS = {"python": "3.13.5", "numpy": "2.5.2", "scipy": "1.18.1"}


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="operatormark_numerics_v2.runners.cli")
    parser.add_argument("stage", choices=("q1", "q2", "q3", "q4"))
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prior", action="append", default=[], help="stage=DIR for each required prior payload")
    parser.add_argument("--budgets", type=Path, default=None,
                        help="JSON {field, compliance, derivative, task}; required in formal mode (confirmed D3 budgets)")
    parser.add_argument("--time-cap-s", type=float, default=None)
    parser.add_argument("--mem-cap-bytes", type=int, default=None)
    parser.add_argument("--cap-override", action="store_true", help="formal mode: allow caps other than revision 3's")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--test-delay-s", type=float, default=0.0)
    parser.add_argument("--only-instance", default=None)
    return parser.parse_args(argv)


def refuse(message: str) -> int:
    sys.stderr.write(f"environment refused: {message}\n")
    return 3


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    check_environment(test_mode=args.test_mode)          # 1. before any numerical import
    try:
        pins.verify_frozen_sources()                      # 2. before the frozen modules are imported
    except pins.FrozenSourceError as exc:
        return refuse(str(exc))
    import platform

    import numpy
    import scipy

    from .common import STAGE_CAPS, StageConfig
    from .stages import run_stage

    try:
        pins.verify_loaded_modules()                      # 3. loaded from the pinned files
    except pins.FrozenSourceError as exc:
        return refuse(str(exc))
    if not args.test_mode:
        found = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}
        if found != REQUIRED_VERSIONS:
            return refuse(f"versions {found}, required {REQUIRED_VERSIONS}")
        if args.budgets is None:
            return refuse("formal stages require --budgets with the confirmed D3 budgets")
        caps = STAGE_CAPS[args.stage]
        requested = {"time_cap_s": args.time_cap_s, "mem_cap_bytes": args.mem_cap_bytes}
        if any(v is not None and v != caps[k] for k, v in requested.items()) and not args.cap_override:
            return refuse(f"caps differ from revision 3's {caps}; use --cap-override to record an approved change")
    budgets = None
    if args.budgets is not None:
        budgets = json.loads(args.budgets.read_text(encoding="utf-8"))
    priors = {}
    for item in args.prior:
        name, sep, path = item.partition("=")
        if not sep or name not in ("q1", "q2", "q3"):
            return refuse(f"bad --prior {item!r}")
        priors[name] = Path(path)
    config = StageConfig(stage=args.stage, mode="test" if args.test_mode else "formal", lock_dir=args.lock,
                         out_dir=args.out, priors=priors, time_cap_s=args.time_cap_s, mem_cap_bytes=args.mem_cap_bytes,
                         budgets=budgets, test_delay_s=args.test_delay_s, only_instance=args.only_instance)
    footer = run_stage(config)
    print(json.dumps({"stage": args.stage, "completion": footer["completion"], "payload_sha256": footer["payload_sha256"]}))
    return 0 if footer["completion"] == "complete" else 4


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
