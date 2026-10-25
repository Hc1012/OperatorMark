"""Explicit data access, float64 query identities and disposable evaluation."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
import importlib
import platform
import sys
import time
import numpy as np
import torch
import scipy
from core import PILOT, Budget, finite, positive


def legacy():
    # Import only on an explicitly numerical path. Never call the old entrypoints.
    directory = str(PILOT)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spatial = importlib.import_module('spatial')
    old = importlib.import_module('run')
    if str(PILOT) not in str(old.__file__) or str(PILOT) not in str(spatial.__file__):
        raise RuntimeError('legacy module-name collision')
    return spatial, old


def runtime():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    import importlib.metadata
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                scipy=scipy.__version__, neuraloperator=importlib.metadata.version('neuraloperator'),
                platform=platform.platform(), processor=platform.processor(),
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(), device='cpu',
                mkldnn=torch.backends.mkldnn.enabled, backend_config=torch.__config__.show())


def array_identity(a):
    a = np.ascontiguousarray(a)
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(a.tobytes()).hexdigest())


def row_bytes(row):
    if row.dtype != np.float64 or row.shape != (64,) or not np.isfinite(row).all():
        raise ValueError('original finite float64 64-control row required')
    return np.ascontiguousarray(row).tobytes()


class Splits:
    def __init__(self):
        self.seen = {}

    def add(self, label, rows):
        keys = [row_bytes(row) for row in rows]
        if len(keys) != len(set(keys)) or any(key in self.seen for key in keys):
            raise ValueError('exact control/query collision before use: ' + label)
        self.seen.update({key: label for key in keys})


class Oracle:
    def __init__(self, solve, budget, *, real=False, log=lambda row: None):
        self.solve, self.budget, self.real, self.log = solve, budget, real, log
        self.cache = {}
        self.requests = self.attempted = self.successful = self.cache_served = 0

    def __call__(self, row, label):
        self.budget.check()
        key = row_bytes(row)
        self.requests += 1
        if key in self.cache:
            self.cache_served += 1
            self.log(dict(type='oracle_cache', label=label, row=array_identity(row)))
            return self.cache[key].copy()
        if self.real:
            self.budget.charge('solver_attempts')
        self.attempted += 1
        self.log(dict(type='oracle_attempt', label=label, row=array_identity(row)))
        answer = np.asarray(self.solve(row.copy()), dtype=np.float64)
        if answer.ndim != 1 or not np.isfinite(answer).all():
            raise ValueError('invalid solver response')
        self.successful += 1
        self.cache[key] = answer.copy()
        self.log(dict(type='oracle_success', label=label, response=array_identity(answer)))
        return answer

    def record(self):
        return dict(requests=self.requests, attempted=self.attempted,
                    successful=self.successful, failed=self.attempted-self.successful,
                    cache_served=self.cache_served)


@dataclass(frozen=True)
class Observations:
    x64: np.ndarray
    y64: np.ndarray
    v64: np.ndarray | None = None
    dy64: np.ndarray | None = None
    information_calls: int = 0

    def __post_init__(self):
        for name in ('x64', 'y64', 'v64', 'dy64'):
            a = getattr(self, name)
            if a is not None:
                if a.dtype != np.float64 or not np.isfinite(a).all():
                    raise ValueError('observation dtype/finiteness')
                a = a.copy(); a.flags.writeable = False
                object.__setattr__(self, name, a)
        if (self.v64 is None) != (self.dy64 is None):
            raise ValueError('incomplete derivative observations')
        n = len(self.x64)
        if n == 0 or self.x64.shape != (n,64) or self.y64.ndim != 2 or len(self.y64) != n:
            raise ValueError('observation geometry')
        if self.v64 is not None and (self.v64.ndim != 3 or self.v64.shape[0] != n or
                self.v64.shape[2] != 64 or self.dy64.shape != (*self.v64.shape[:2],self.y64.shape[1])):
            raise ValueError('derivative observation geometry')

    def tensors(self):
        arrays = (self.x64, self.v64, self.y64, self.dy64)
        return tuple(None if a is None else torch.tensor(a, dtype=torch.float32) for a in arrays)

    def identities(self):
        return {name: array_identity(getattr(self, name)) for name in
                ('x64', 'y64', 'v64', 'dy64') if getattr(self, name) is not None}


def controls(seed, n):
    return np.random.Generator(np.random.PCG64(seed)).uniform(.2, .8, (n, 64))


def directions(seed, n, k):
    v = np.random.Generator(np.random.PCG64(seed)).normal(size=(n, k, 64))
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError('invalid direction norm')
    return v / norm


def forward(oracle, x64, label, splits=None):
    if splits is not None:
        splits.add(label, x64)
    y = np.stack([oracle(row, label) for row in x64])
    return Observations(x64, y, information_calls=len(x64))


def derivatives(oracle, base, v64, label, splits=None):
    if v64.shape[:1] != base.x64.shape[:1] or v64.shape[-1] != 64:
        raise ValueError('direction geometry')
    plus = base.x64[:, None, :] + .001*v64
    minus = base.x64[:, None, :] - .001*v64
    if splits is not None:
        splits.add(label+'/perturbations', np.concatenate([plus.reshape(-1,64), minus.reshape(-1,64)]))
    dy = np.stack([np.stack([(oracle(p, label)-oracle(m, label))/.002
                              for p, m in zip(ps, ms)]) for ps, ms in zip(plus, minus)])
    return Observations(base.x64, base.y64, v64, dy,
                        len(base.x64)*(1+2*v64.shape[1]))


def generate(oracle, control_seed, direction_seed, n, k, label, splits=None):
    base = forward(oracle, controls(control_seed, n), label, splits)
    return derivatives(oracle, base, directions(direction_seed, n, k), label, splits)


def allocation(oracle, control_seed, direction_seed, splits=None):
    common = forward(oracle, controls(control_seed, 256), 'attack', splits)
    # Copy only the permitted first 85 labels; never pass common to the R-b trainer.
    subset = Observations(common.x64[:85], common.y64[:85], information_calls=85)
    rb = derivatives(oracle, subset, directions(direction_seed, 85, 1), 'R-b', splits)
    return common, rb


def fresh(seed, factory=None):
    if factory is None:
        factory = legacy()[0].SpatialFNO
    torch.manual_seed(seed)  # immediately before construction, including after legacy loading
    return factory()


def positive_scale(tensor):
    value = tensor.square().mean().detach()
    if not torch.isfinite(value) or value <= 0:
        raise ValueError('nonpositive/nonfinite normalization; no epsilon repair')
    return value


def inverse(model, starts64, targets64, oracle, budget, label):
    model = copy.deepcopy(model).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    x = torch.nn.Parameter(torch.tensor(starts64, dtype=torch.float32))
    target = torch.tensor(targets64, dtype=torch.float32)
    optimizer = torch.optim.Adam([x], lr=.03)
    for _ in range(120):
        budget.check()
        optimizer.zero_grad()
        loss = (model(x)-target).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite inverse loss')
        loss.backward()
        budget.charge('inverse_updates')
        optimizer.step()
        with torch.no_grad():
            x.clamp_(.2, .8)
    optimized = x.detach().numpy().astype(np.float64)
    norm = np.linalg.norm(targets64, axis=-1)
    if np.any(norm <= 0) or not np.isfinite(norm).all():
        raise ValueError('invalid inverse target norm')
    errors = [finite(np.linalg.norm(oracle(row, label)-truth)/n)
              for row, truth, n in zip(optimized, targets64, norm)]
    return dict(mean=float(np.mean(errors)), per_target=errors,
                optimized_controls=optimized.tolist(), input_updates=120)


def evaluate(model, data, tasks, oracle, budget, label, key=None, key_truth=None):
    spatial, old = legacy()
    disposable = copy.deepcopy(model).eval()
    tensors = data.tensors()
    metrics, detail = old.evaluate(disposable, tensors, key, key_truth)
    task = inverse(disposable, tasks['starts'], tasks['targets'], oracle, budget, label)
    metrics['task'] = task['mean']
    raw = {k: (v.tolist() if torch.is_tensor(v) else v) for k,v in detail.items()}
    raw['task'] = task
    return metrics, raw


def task_data(oracle, seeds, n, label, splits):
    targets = controls(seeds.get(label+'/targets'), n)
    starts = controls(seeds.get(label+'/starts'), n)
    target_truth = forward(oracle, targets, label+'/targets', splits).y64
    start_truth = forward(oracle, starts, label+'/starts', splits).y64
    denominator = np.linalg.norm(target_truth, axis=-1)
    if np.any(denominator <= 0):
        raise ValueError('zero target denominator')
    return dict(starts=starts, targets=target_truth,
                initial_errors=(np.linalg.norm(start_truth-target_truth, axis=-1)/denominator).tolist())


def keyed_truth(oracle, x64, key, label, splits=None):
    v = np.asarray(key['v'], dtype=np.float64)
    plus, minus = x64+.001*v, x64-.001*v
    if splits is not None:
        splits.add(label, np.concatenate([plus, minus]))
    values = np.stack([(oracle(p,label)-oracle(m,label))/.002 for p,m in zip(plus,minus)])
    return torch.tensor(values, dtype=torch.float32)


def b2(jvp, reference):
    a, b = np.asarray(jvp), np.asarray(reference)
    denominator = np.linalg.norm(b, axis=-1)
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError('invalid B2 denominator')
    samples = np.linalg.norm(a-b, axis=-1)/denominator
    if not np.isfinite(samples).all():
        raise ValueError('nonfinite B2')
    return dict(mean=float(np.mean(samples)), per_direction=samples.tolist())
