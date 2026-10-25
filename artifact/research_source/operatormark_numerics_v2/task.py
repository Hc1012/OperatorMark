"""D8 task: pilot-form loss with the pilots' output scale, float64 projected
Adam (torch.optim.Adam single-tensor formula, default flags), and the score.

The pilot oracle returned fields divided by SCALE = 0.01 and the pilot loss was
the mean squared difference of those scaled fields over the batch and all
coordinates.  Dropping the scale makes gradients 1e4 times smaller and leaves
most components below Adam's eps, so it is kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

SCALE = 0.01
STEPS = 120
LEARNING_RATE = 0.03
BETAS = (0.9, 0.999)
EPS = 1.0e-8
BOX = (0.0, 1.0)          # [C] revision 3 box; the pilots clamped to [0.2, 0.8]
BATCH = 16

FieldAndVjp = Callable[[np.ndarray], tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]]


def loss_weight(batch: int) -> float:
    """Gradient factor 2 / (B * 4096 * SCALE^2) applied to (F64 - target)."""

    return 2.0 / (batch * 4096 * SCALE * SCALE)


def batch_loss(fields: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> float:
    total = 0.0
    for f, t in zip(fields, targets):
        d = (f - t) / SCALE
        total += float(np.sum(d * d, dtype=np.float64))
    return total / (len(fields) * 4096)


class Adam:
    """torch.optim.Adam (single-tensor path, amsgrad=False, weight_decay=0, maximize=False) in float64."""

    def __init__(self, shape, lr: float = LEARNING_RATE, betas: tuple[float, float] = BETAS, eps: float = EPS):
        self.lr, (self.beta1, self.beta2), self.eps = float(lr), (float(betas[0]), float(betas[1])), float(eps)
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.step_count = 0

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.step_count += 1
        t = float(self.step_count)
        # exp_avg.lerp_(grad, 1 - beta1): weight < 0.5 -> start + weight * (end - start)
        self.m = self.m + (1.0 - self.beta1) * (grad - self.m)
        # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        self.v = self.v * self.beta2 + (1.0 - self.beta2) * grad * grad
        bias_correction1 = 1.0 - self.beta1 ** t
        bias_correction2 = 1.0 - self.beta2 ** t
        step_size = self.lr / bias_correction1
        bias_correction2_sqrt = bias_correction2 ** 0.5
        denom = (np.sqrt(self.v) / bias_correction2_sqrt) + self.eps
        return x + (-step_size) * (self.m / denom)


@dataclass(frozen=True)
class OptimizeResult:
    designs: np.ndarray            # [B,8,8]
    loss_trace: tuple[float, ...]  # loss before each step (at the iterate the gradient was taken)
    final_loss: float


def optimize(field_and_vjp: FieldAndVjp, starts: np.ndarray, targets: np.ndarray, *, steps: int = STEPS,
             lr: float = LEARNING_RATE, box: tuple[float, float] = BOX, batch_size: int | None = None) -> OptimizeResult:
    """Projected Adam on the pilot loss.

    Adam acts per coordinate and the batch enters only through the loss factor
    1/B, so optimizing one target with batch_size=B reproduces that target's
    trajectory inside a batch of B exactly (used for per-target instances).
    The loss trace is the batch loss over the targets actually passed, scaled by
    len(targets)/batch_size so that it is that subset's contribution.
    """

    starts = np.ascontiguousarray(starts, dtype=np.float64)
    targets = np.ascontiguousarray(targets, dtype=np.float64)
    if starts.ndim != 3 or starts.shape[1:] != (8, 8) or targets.shape != (starts.shape[0], 64, 64):
        raise ValueError("starts [B,8,8] and targets [B,64,64] required")
    batch = starts.shape[0] if batch_size is None else int(batch_size)
    if batch < starts.shape[0]:
        raise ValueError("batch_size cannot be smaller than the number of targets")
    weight = loss_weight(batch)
    count = starts.shape[0]
    x = np.clip(starts, box[0], box[1])
    if not np.array_equal(x, starts):
        raise ValueError("starts must lie in the box")
    adam = Adam(x.shape, lr=lr)
    trace = []
    for _ in range(steps):
        grads = np.empty_like(x)
        fields = []
        for k in range(count):
            f, vjp = field_and_vjp(x[k])
            fields.append(f)
            grads[k] = vjp(np.float64(weight) * (f - targets[k]))
        trace.append(batch_loss(fields, list(targets)) * len(fields) / batch)
        x = np.clip(adam.step(x, grads), box[0], box[1])
    final_fields = [field_and_vjp(x[k])[0] for k in range(count)]
    return OptimizeResult(np.ascontiguousarray(x), tuple(trace), batch_loss(final_fields, list(targets)) * len(final_fields) / batch)


def score(field_design: np.ndarray, field_target: np.ndarray) -> float:
    d = field_design - field_target
    num = math.sqrt(float(np.sum(d * d, dtype=np.float64)))
    den = math.sqrt(float(np.sum(field_target * field_target, dtype=np.float64)))
    if not den > 0.0:
        raise ValueError("target field has zero norm")
    return num / den
