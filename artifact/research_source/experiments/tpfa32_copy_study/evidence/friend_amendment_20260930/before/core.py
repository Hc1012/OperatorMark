"""Non-numerical contracts, records and bounded accounting for the study."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT = HERE.parent / 'darcy_fno32_pilot'
METRICS = ('forward', 'random_jvp', 'key_jvp', 'task')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def write(path, value):
    """Never overwrite an outcome, manifest or attempt."""
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())


def replace_record(path, value):
    """Only for live counters, never for completed outputs."""
    path = Path(path)
    temp = path.with_suffix('.next')
    with temp.open('w', encoding='utf-8') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


def positive(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def finite(value):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError('finite numeric value required')
    return float(value)


def utility(candidate, owner):
    offsets = dict(forward=.01, random_jvp=.05, key_jvp=.05, task=.02)
    gates = {}
    for name in METRICS:
        a, b = candidate.get(name), owner.get(name)
        gates[name] = (a is not None and b is not None and
                       math.isfinite(float(a)) and math.isfinite(float(b)) and
                       a >= 0 and b >= 0 and a <= 1.25 * b + offsets[name])
    return dict(gates=gates, passed=all(gates.values()))


def detection(score, wrong, threshold):
    if len(wrong) != 16 or not positive(threshold):
        raise ValueError('locked threshold and exactly 16 wrong keys required')
    score = finite(score)
    maximum = max(abs(finite(x)) for x in wrong)
    return dict(true_score=score, wrong_scores=wrong, wrong_max_abs=maximum,
                threshold=threshold, margin_to_threshold=score-threshold,
                margin_to_wrong=score-maximum-.003,
                detected=score > threshold and score > maximum + .003)


def threshold(other_clean, marked_wrong, owner_count=6):
    if len(other_clean) != owner_count-1 or len(marked_wrong) != 16:
        raise ValueError('complete clean/null population required')
    return max(.02, *(abs(finite(x)) for x in other_clean + marked_wrong)) + .003


def cohort_gate(rows, expected):
    return (len(rows) == expected and all(
        r is not None and all(k in r and math.isfinite(r[k]) and r[k] >= 0
                             for k in ('forward', 'random_jvp')) and
        r['forward'] <= .12 and r['random_jvp'] <= .50 for r in rows))


def selection_score(rows):
    if len(rows) != 3:
        raise ValueError('all three development cases required')
    logs = []
    for row in rows:
        for name in ('forward', 'random_jvp', 'task'):
            a, b = row['replacement'].get(name), row['clean'].get(name)
            if not positive(a) or not positive(b):
                raise ValueError('selection needs complete finite positive errors')
            logs.append(math.log(a) - math.log(b))
    return math.exp(sum(logs)/len(logs))


def select_recipe(histories):
    selected = {}
    for recipe in ('R-a', 'R-b'):
        history = histories[recipe]
        if not history or any(not positive(r['score']) for r in history):
            raise ValueError('both recipes must be evaluated')
        selected[recipe] = min(history, key=lambda r: (r['score'], r['updates']))
    winner = min(selected, key=lambda k: (selected[k]['score'], k != 'R-a'))
    return dict(primary=winner, recipes=selected, retain_Ra=True)


def validate_config(config):
    if config['schema'] != 'tpfa32_copy_study.v1':
        raise ValueError('configuration schema')
    if config['owners'] < 2 or config['oracle_budget'] != 256:
        raise ValueError('owner count or implemented oracle allocation unsupported')
    if config['selection_updates'] != [1920, 3840, 7680, 15360, 30720]:
        raise ValueError('selection schedule requires a prospective rule revision')
    if config['analysis_rule'] != 'provisional_simultaneous_t':
        raise ValueError('unimplemented analysis rule')
    if config.get('random_directions') != 2:
        raise ValueError('two evaluation directions required by implemented design')
    for name in ('development_controls','development_tasks','calibration_controls',
                 'calibration_tasks','final_controls','final_tasks'):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError('positive integer configuration size required: '+name)
    return config


def source_hashes():
    paths = list(HERE.glob('*.py')) + [PILOT/'run.py', PILOT/'spatial.py',
        PILOT/'RULE.md', ROOT/'operatormark/solvers/darcy_tpfa.py',
        ROOT/'requirements-lock-cpu.txt', ROOT/'pyproject.toml']
    return {str(p.relative_to(ROOT)).replace('\\','/'): digest(p) for p in paths}


def authorize_campaign(config, authority, mode):
    """Called before numerical imports, output creation or any seed draw."""
    validate_config(config)
    if mode == 'final' and (config['status'] != 'locked' or not config.get('selection')):
        raise PermissionError('draft/unselected configuration cannot start final work')
    if mode == 'final' and config['owners'] != 6:
        raise PermissionError('owner-count revision requires matching reviewed analysis implementation')
    if not authority or authority.get('scope') != mode:
        raise PermissionError('separate prospective execution authority required')
    canonical = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if authority.get('config_sha256') != canonical or not authority.get('approved_by_user'):
        raise PermissionError('authority does not bind this configuration')
    if authority.get('deadline_unix', 0) <= time.time():
        raise PermissionError('execution authority expired')
    if authority.get('source_sha256') != source_hashes() or not authority.get('execution_id'):
        raise PermissionError('authority must bind current sources and one execution identity')
    for name in ('solver_attempts', 'model_updates', 'inverse_updates'):
        if not positive(authority.get('limits', {}).get(name)):
            raise PermissionError('explicit measured campaign limits required')


class Seeds:
    """Private labelled, once-drawn streams; no root seed or silent redraw."""
    def __init__(self, path, stage, fixture=None, forbidden=()):
        self.path, self.stage = Path(path), stage
        if self.path.exists():
            raise FileExistsError(self.path)
        self.fixture = fixture
        self.values = {}
        self.used = set(forbidden)
        self.events = []
        write(self.path, self.record())

    def record(self):
        return dict(schema='private_seeds.v1', stage=self.stage,
                    fixture=self.fixture is not None, values=self.values,
                    events=self.events, provenance='OS random draws, not proof of unseen seeds')

    def get(self, label, offsets=(0,)):
        label = self.stage + '/' + label
        if label in self.values:
            prior = next(e for e in self.events if e['label'] == label)
            if prior['offsets'] != list(offsets):
                raise ValueError('seed offset contract changed')
            return self.values[label]
        value = (secrets.randbits(63) if self.fixture is None
                 else self.fixture + 100000 * len(self.values))
        expanded = {value + offset for offset in offsets}
        collision = bool(expanded & self.used)
        self.events.append(dict(label=label, offsets=list(offsets), collision=collision))
        if collision:
            replace_record(self.path, self.record())
            raise ValueError('seed collision: preserve manifest; no redraw')
        self.values[label] = value
        self.used.update(expanded)
        replace_record(self.path, self.record())
        return value


class Budget:
    """Charge attempted operations durably before the operation; never refund."""
    def __init__(self, path=None, limits=None, deadline=None):
        self.path = Path(path) if path else None
        self.limits = limits or dict(solver_attempts=0, model_updates=0, inverse_updates=0)
        self.deadline = deadline
        self.counts = {name: 0 for name in self.limits}
        if self.path and self.path.exists():
            self.counts = read(self.path)['counts']
        self.flush()

    @classmethod
    def verification(cls):
        if 'TPFA_BUDGET' not in os.environ:
            raise PermissionError('run numerical verification through supervise.py')
        return cls(os.environ['TPFA_BUDGET'], json.loads(os.environ['TPFA_LIMITS']),
                   float(os.environ['TPFA_DEADLINE']))

    def flush(self):
        if self.path:
            replace_record(self.path, dict(limits=self.limits, counts=self.counts))

    def check(self):
        if self.deadline is not None and time.time() >= self.deadline:
            if str(self.deadline) == os.environ.get('TPFA_DEADLINE') and 'TPFA_ATTEMPT' in os.environ:
                marker=Path(os.environ['TPFA_ATTEMPT'])/'TIMEOUT.json'
                if not marker.exists():write(marker,dict(deadline=self.deadline,observed_unix=time.time()))
            raise TimeoutError('child numerical deadline')

    def charge(self, name, count=1):
        self.check()
        if count < 0 or self.counts[name] + count > self.limits[name]:
            raise RuntimeError('quota exhausted: ' + name)
        self.counts[name] += count
        self.flush()


class Output:
    def __init__(self, path, kind):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        write(self.path/'STARTED.json', dict(kind=kind, started_unix=time.time(),
                                            fixture=kind == 'verification'))

    def log(self, row):
        with (self.path/'raw.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(row, allow_nan=False) + '\n')
            out.flush()

    def finish(self, value, failed=False):
        if any((self.path/name).exists() for name in ('FAILED.json','COMPLETED.json')):
            raise FileExistsError('terminal outcome already recorded')
        write(self.path/('FAILED.json' if failed else 'COMPLETED.json'),
              dict(elapsed=time.monotonic()-self.started, **value))
