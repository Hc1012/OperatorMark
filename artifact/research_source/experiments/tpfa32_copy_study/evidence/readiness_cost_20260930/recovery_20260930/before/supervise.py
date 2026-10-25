"""Small serial child supervisor and additive verification budget ledger."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from core import HERE, read, write, replace_record

CAPS = {'focused': (1200, 300, dict(solver_attempts=64, model_updates=96, inverse_updates=240)),
        'replay': (600, 600, dict(solver_attempts=256, model_updates=1920, inverse_updates=0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', choices=CAPS, default='focused')
    parser.add_argument('--ledger', type=Path, default=HERE/'evidence'/'verification')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.print_help(); return 0
    root = args.ledger.resolve(); root.mkdir(parents=True, exist_ok=True)
    if root != (HERE/'evidence'/'verification').resolve():
        raise ValueError('this milestone uses one cumulative ledger; no alternate allowance')
    lock = root/'active.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    attempt = None
    try:
        prior = [read(p) for p in root.glob('*/result.json')]
        if any(r['timed_out'] or not r['owned_process_ended'] for r in prior):
            raise RuntimeError('verification stopped after timeout/unresolved child')
        starts = list(root.glob('*/started.json'))
        if len(starts) != len(prior):
            raise RuntimeError('prior child accounting unresolved')
        ceiling, individual, limits = CAPS[args.category]
        used = sum(r['wall_seconds'] for r in prior if r['category'] == args.category)
        duration = min(individual, ceiling-used, 1800-sum(r['wall_seconds'] for r in prior))
        if duration <= 0 or sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) >= 1024**3:
            raise RuntimeError('wall/output budget exhausted')
        attempt = root/(time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
        attempt.mkdir()
        counter = root/(args.category+'_counts.json')
        if not counter.exists():
            write(counter, dict(limits=limits, counts={k: 0 for k in limits}))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', TPFA_BUDGET=str(counter),
                   TPFA_LIMITS=json.dumps(limits), TPFA_DEADLINE=str(time.time()+duration),
                   TPFA_ATTEMPT=str(attempt), TPFA_CATEGORY=args.category)
        began = time.monotonic()
        write(attempt/'started.json', dict(command=command, category=args.category,
              ceiling_seconds=duration, counter_before=read(counter), started_unix=time.time()))
        timed_out = False; cleanup = []
        with (attempt/'stdout.txt').open('wb') as out, (attempt/'stderr.txt').open('wb') as err:
            child = subprocess.Popen(command, stdout=out, stderr=err, env=env)
            try:
                code = child.wait(timeout=duration)
            except subprocess.TimeoutExpired:
                timed_out = True
                if os.name == 'nt':
                    killed = subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                                            capture_output=True, timeout=15)
                    cleanup.append(dict(exit=killed.returncode, stdout=killed.stdout.decode(errors='replace'),
                                        stderr=killed.stderr.decode(errors='replace')))
                else:
                    child.kill()
                try: code = child.wait(timeout=15)
                except subprocess.TimeoutExpired: code = None
        record = dict(category=args.category, command=command, pid=child.pid,
                      wall_seconds=time.monotonic()-began, exit_code=code,
                      timed_out=timed_out or (attempt/'TIMEOUT.json').exists(),
                      owned_process_ended=child.poll() is not None, cleanup=cleanup,
                      counts=read(counter), source_editing_included=False)
        write(attempt/'result.json', record)
        print(json.dumps(dict(attempt=str(attempt), **record)))
        return 124 if record['timed_out'] else (code if code is not None else 125)
    finally:
        os.close(fd)
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
