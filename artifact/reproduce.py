"""Reproduce OperatorMark's published analysis from saved observations, offline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files() -> dict:
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f'Missing or unexpected release file: {name}')
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise RuntimeError(f'Release file differs: {name}')
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Verify shipped hashes using only the Python standard library.')
    parser.add_argument('--output', type=Path, default=ROOT / 'build' / 'reproduced', help='Output folder, which must be inside build/.')
    args = parser.parse_args()
    start = time.perf_counter()
    manifest = verify_files()
    print(f"Verified {len(manifest['files'])} release files.", flush=True)
    if args.verify_only:
        return 0
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / 'build').resolve()):
        raise RuntimeError('--output must be within this release folder\'s build/ directory.')
    expected_versions = {'numpy': '2.3.5', 'scipy': '1.17.0', 'matplotlib': '3.10.8'}
    versions = {}
    for package, version in expected_versions.items():
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError('Install requirements-analysis.txt in a virtual environment; see README.md.') from error
        if versions[package] != version:
            raise RuntimeError(f'{package} {versions[package]} is installed; the tested release uses {version}. Use requirements-analysis.txt in a separate virtual environment.')
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(output / '.matplotlib'))
    for script in ('analyze_saved_results.py', 'make_figures.py'):
        print(f"Running {script} on saved measurements...", flush=True)
        with (output / (script + '.log')).open('wb') as log:
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'analysis' / script), '--output', str(output)],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise RuntimeError(f'{script} failed; see {output / (script + ".log")}. No experiment was started.')
    actual = output / 'FINAL_ANALYSIS.json'
    expected = ROOT / 'analysis' / 'expected' / 'FINAL_ANALYSIS.json'
    if actual.read_bytes() != expected.read_bytes():
        raise RuntimeError('Reproduced analysis differs from the shipped expected result. Preserve the output for inspection.')
    analysis = json.loads(actual.read_text())
    rows = ['# All model-variant summaries', '',
            'Counts use the planned ten-owner roster except B10 (one O01 context). D2 task omissions remain unmeasured. Empty removal cells mean not applicable.', '',
            '| Variant | N | Detected | Full utility | Qualified removal | Retained + utility | Task unmeasured |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, values in sorted(analysis['variants'].items()):
        fields = [name] + [values[key] for key in ('n', 'detected', 'all_utility', 'qualified_removal', 'retained_detection_and_utility', 'task_unmeasured')]
        rows.append('| ' + ' | '.join('—' if value is None else str(value) for value in fields) + ' |')
    (output / 'ALL_VARIANT_TABLES.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    pngs = sorted((output / 'figures').glob('*.png'))
    if len(pngs) != 4:
        raise RuntimeError('Expected four generated figures.')
    # PNG bytes are checked on the validated platform; PDF/SVG may contain timestamps/IDs.
    png_matches = {p.name: digest(p) == digest(ROOT / 'docs' / 'figures' / p.name) for p in pngs}
    verification = {'status': 'passed', 'analysis_byte_identical': True,
                    'analysis_sha256': digest(actual), 'python': sys.version.split()[0], 'packages': versions,
                    'model_observations': len(analysis['model_readouts']), 'variant_summaries': len(analysis['variants']),
                    'figures_generated': len(pngs), 'figure_png_byte_matches': png_matches,
                    'figure_note': 'Font/rendering differences can affect image bytes across operating systems; analysis equality is required.',
                    'seconds': time.perf_counter() - start, 'new_training_or_solver_runs': 0,
                    'network_access_required': False, 'full_campaign_reproduced': False}
    (output / 'REPRODUCTION.json').write_text(json.dumps(verification, indent=2, sort_keys=True) + '\n')
    print(f'PASS: exact analysis match, {len(pngs)} figures and 128 variant summaries.', flush=True)
    print(f'Output: {output}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        raise SystemExit(1)
