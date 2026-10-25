"""Q5-only 64-grid pilot adapter and identical-tensor CPU/GPU timing leg.

No solvers, qualification material, or production-corpus entry points.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from neuralop.models import FNO


class Q5FNO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.operator = FNO(n_modes=(12, 12), in_channels=1, out_channels=1,
                            hidden_channels=32, n_layers=4,
                            lifting_channel_ratio=1, projection_channel_ratio=1)

    def forward(self, control):
        if control.ndim != 2 or control.shape[1] != 64:
            raise ValueError('expected [batch,64] control, not coefficient input')
        x = control.reshape(-1, 1, 8, 8).repeat_interleave(8, -2).repeat_interleave(8, -1)
        return self.operator(x).reshape(-1, 4096)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def configure():
    if getattr(configure, '_done', False):
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    configure._done = True


def scales(y, dy):
    values = (y.square().mean().detach(), dy.square().mean().detach())
    if any(not torch.isfinite(v) or v <= 0 for v in values):
        raise ValueError('nonpositive/nonfinite dataset normalization; no epsilon repair')
    return values


def tensor_state(model):
    # NeuralOperator BaseModel.state_dict injects a non-tensor _metadata key
    # containing callables. Architecture is already explicit/source-bound here.
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if torch.is_tensor(v)}


def objective(model, x, v, y, dy, fscale, jscale):
    prediction = model(x)
    derivative = torch.func.jvp(model, (x,), (v,))[1]
    forward = (prediction-y).square().mean()/fscale
    tangent = (derivative-dy).square().mean()/jscale
    return forward + .5*tangent, forward, tangent


def batches(seed, count=64, updates=200):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    for _ in range((updates+count//16-1)//(count//16)):
        order = torch.randperm(count, generator=generator)
        for offset in range(0, count, 16):
            if updates <= 0:
                return
            yield order[offset:offset+16]
            updates -= 1


def train(bundle, out, device, deadline):
    """Exactly 200 updates, first 20 warm-up. No automatic retry or convergence test."""
    bundle, out = Path(bundle), Path(out)
    out.mkdir(exist_ok=False)
    manifest = json.loads((bundle/'MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(bundle/name) != expected:
            raise ValueError('GPU/CPU benchmark input drift: '+name)
    if sha(__file__) != manifest['adapter_sha256']:
        raise ValueError('model adapter drift')
    configure()
    import importlib.metadata
    if (torch.__version__.split('+')[0] != '2.12.0' or np.__version__ != '2.5.2'
            or importlib.metadata.version('neuraloperator') != '2.0.0'):
        raise RuntimeError('benchmark dependency version drift; no automatic installation')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback and no dependency installation')
    torch.manual_seed(manifest['initialization_seed'])
    model = Q5FNO()
    model.load_state_dict(torch.load(bundle/'initialization.pt', map_location='cpu', weights_only=True), strict=True)
    x, v, y, dy = (torch.from_numpy(np.load(bundle/(name+'.npy'), allow_pickle=False)).to(device)
                    for name in ('x','v','y','dy'))
    if any(tuple(t.shape) != (64, n) for t,n in zip((x,v,y,dy),(64,64,4096,4096))):
        raise ValueError('benchmark tensor population/shape mismatch')
    model.to(device).train()
    fscale, jscale = (torch.tensor(z, dtype=torch.float32, device=device) for z in manifest['normalizations'])
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    sync = torch.cuda.synchronize if device == 'cuda' else lambda: None
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    sync(); started = time.monotonic(); rows = []
    save(out/'START.json', dict(device=device, torch=torch.__version__, numpy=np.__version__,
        cuda=torch.version.cuda, device_name=torch.cuda.get_device_name() if device=='cuda' else 'CPU',
        cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        adapter_sha256=sha(__file__), manifest_sha256=sha(bundle/'MANIFEST.json'),
        parameter_count=sum(p.numel() for p in model.parameters()), normalizations=manifest['normalizations']))
    for index, ids in enumerate(batches(manifest['shuffle_seed']), 1):
        if time.time() >= deadline:
            raise TimeoutError('Q5 work deadline')
        ids = ids.to(device)
        sync(); t0=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss, lf, lj = objective(model,x[ids],v[ids],y[ids],dy[ids],fscale,jscale)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite loss before update '+str(index))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if not gradients or any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('missing/nonfinite gradient before update '+str(index))
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite parameters at update '+str(index))
        sync(); elapsed = time.monotonic()-t0
        row=dict(update=index, phase='warmup' if index<=20 else 'sustained', seconds=elapsed,
                 loss=float(loss.detach()), forward_loss=float(lf.detach()), jvp_loss=float(lj.detach()),
                 finite_gradients=True, batch_indices=ids.cpu().tolist())
        append(out/'updates.jsonl',row); rows.append(row)
        if index % 20 == 0:
            print(json.dumps(dict(stage='training',update=index,elapsed=time.monotonic()-started)),flush=True)
    result = dict(complete=True, device=device, updates=len(rows), warmup_updates=20, sustained_updates=180,
                  examples=3200, jvp_examples=3200, epoch_passes=50,
                  warmup_seconds=sum(r['seconds'] for r in rows[:20]),
                  sustained_seconds=sum(r['seconds'] for r in rows[20:]),
                  wall_seconds=time.monotonic()-started, finite_losses_gradients=True,
                  peak_gpu_allocated=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                  peak_gpu_reserved=torch.cuda.max_memory_reserved() if device=='cuda' else None)
    save(out/'RESULT.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Prepared Q5 comparison; run only under later GPU authority.')
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', required=True, choices=['cuda'])
    parser.add_argument('--work-seconds', type=float, default=3570)
    args=parser.parse_args()
    train(args.bundle,args.out,args.device,time.time()+args.work_seconds)
