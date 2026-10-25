"""Command line: sizing and readback validation (generation is driven from Python or Codex's State adapters).

    python -m operatormark_corpus_v2.cli size --n-ref 128|256 --retention E|S [--audit-bytes N] [--files]
    python -m operatormark_corpus_v2.cli validate ROOT --n-ref N --retention E|S --mode production|synthetic_test
                      [--counts 4096,256,512 --shard-size 64] [--select split:shard ...]
                      [--anomaly-state active|inactive_field_incomplete --anomaly-bounds relaxed=X,binary=Y --eps-field E]
                      [--qualification-dir q1=DIR q2=DIR q3=DIR q4=DIR] [--lock-commitment-sha256 H]
                      [--configuration-sha256 H] [--controls-sha256 H] [--direction-seed-file FILE]
                      [--retention-decision-sha256 H] [--choices-reference-sha256 H]
                      [--numerics full|structural] [--p1 resolve|skip] [--out FILE]

The direction seed is read from a file (ASCII decimal) so it never appears on a command line.

Exit codes: 0 = matched and D10 accepted; 1 = anything else (the JSON says why); 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import pins


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="operatormark_corpus_v2")
    sub = parser.add_subparsers(dest="command", required=True)
    size = sub.add_parser("size")
    size.add_argument("--n-ref", type=int, required=True, choices=(128, 256))
    size.add_argument("--retention", required=True, choices=("E", "S"))
    size.add_argument("--audit-bytes", type=int, default=None)
    size.add_argument("--files", action="store_true", help="also list every .npy file with its exact size")
    val = sub.add_parser("validate")
    val.add_argument("root", type=Path)
    val.add_argument("--n-ref", type=int, required=True, choices=(128, 256))
    val.add_argument("--retention", required=True, choices=("E", "S"))
    val.add_argument("--mode", required=True, choices=("production", "synthetic_test"))
    val.add_argument("--counts", default="4096,256,512")
    val.add_argument("--shard-size", type=int, default=64)
    val.add_argument("--select", nargs="*", default=None, help="split:shard, e.g. training:0")
    val.add_argument("--configuration-sha256", default=None)
    val.add_argument("--anomaly-state", default=None, choices=("active", "inactive_field_incomplete"))
    val.add_argument("--anomaly-bounds", default=None, help="relaxed=X,binary=Y (float or float.hex)")
    val.add_argument("--eps-field", default=None)
    val.add_argument("--qualification-dir", nargs="*", default=None, help="q1=DIR q2=DIR q3=DIR q4=DIR")
    val.add_argument("--lock-commitment-sha256", default=None)
    val.add_argument("--controls-sha256", default=None)
    val.add_argument("--directions-sha256", default=None)
    val.add_argument("--direction-seed-file", type=Path, default=None)
    val.add_argument("--retention-decision-sha256", default=None)
    val.add_argument("--choices-reference-sha256", default=None)
    val.add_argument("--numerics", default="full", choices=("full", "structural"))
    val.add_argument("--p1", default="resolve", choices=("resolve", "skip"))
    val.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    pins.verify_dependencies()
    if args.command == "size":
        from . import sizing
        report = sizing.size_corpus(args.n_ref, args.retention, audit_bytes=args.audit_bytes)
        if args.files:
            report["files"] = sizing.file_plan(args.n_ref, args.retention)
        print(json.dumps(report, indent=1, sort_keys=True, default=list))
        return 0
    from . import readback
    counts = tuple(int(x) for x in args.counts.split(","))
    selection = None if args.select is None else tuple((s.split(":")[0], int(s.split(":")[1])) for s in args.select)
    def number(text):
        return float.fromhex(text) if "x" in text.lower() else float(text)

    bounds = None if args.anomaly_bounds is None else {
        k: number(v) for k, v in (item.split("=") for item in args.anomaly_bounds.split(","))}
    qdirs = None if args.qualification_dir is None else {
        k: Path(v) for k, v in (item.split("=", 1) for item in args.qualification_dir)}
    seed = None
    if args.direction_seed_file is not None:
        text = args.direction_seed_file.read_text().strip()
        if not text.isascii() or not text.isdigit():
            parser.error("the direction seed file must hold one ASCII decimal integer")
        seed = int(text)
    expected = readback.Expected(
        args.n_ref, args.retention, counts, args.shard_size, args.mode, selection=selection,
        configuration_sha256=args.configuration_sha256, anomaly_state=args.anomaly_state, anomaly_bounds=bounds,
        eps_field=None if args.eps_field is None else number(args.eps_field), qualification_dirs=qdirs,
        controls_sha256=args.controls_sha256, direction_seed=seed, directions_sha256=args.directions_sha256,
        lock_commitment_sha256=args.lock_commitment_sha256, retention_decision_sha256=args.retention_decision_sha256,
        choices_reference_sha256=args.choices_reference_sha256)
    result = readback.validate_corpus(args.root, expected, numerics=args.numerics, p1=args.p1)
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0 if result["status"] == "matched" and result["d10"]["verdict"] == "accepted" else 1


if __name__ == "__main__":
    sys.exit(main())
