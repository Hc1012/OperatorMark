"""Continuous step-based trajectories and an unchanged legacy attack adapter."""
from __future__ import annotations
import copy
import json
import os
import time
from pathlib import Path
import torch
from core import digest
from numerics import legacy, positive_scale, array_identity


class Trajectory:
    def __init__(self, model, data, shuffle_seed, budget, *, recipe='forward',
                 batch=32, lr=.002, real=False, reference=None, key=None):
        if recipe not in ('forward','derivative','mark'):
            raise ValueError('recipe')
        if not all(p.requires_grad for p in model.parameters()):
            raise ValueError('training parameters frozen by prior evaluation')
        self.model, self.data, self.budget = model, data, budget
        self.recipe, self.batch, self.lr, self.real = recipe, batch, lr, real
        self.x, self.v, self.y, self.dy = data.tensors()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.shuffle = torch.Generator().manual_seed(shuffle_seed)
        self.order = None; self.offset = 0; self.updates = self.epochs = self.examples = 0
        self.jvp_examples = 0; self.wall_seconds = self.cpu_seconds = 0.
        self.batch_sizes = []
        self.losses = []
        self.fscale = self.jscale = None
        self.key = key
        if recipe != 'forward':
            if self.v is None or self.v.shape[1] != 1:
                raise ValueError('one owned derivative target per training row required')
            self.fscale, self.jscale = positive_scale(self.y), positive_scale(self.dy)
        self.score_scale = None
        self.cooldown = None
        self.branch_rng = None
        if recipe == 'mark':
            if reference is None or key is None:
                raise ValueError('marking needs reference/key; replacement paths do not')
            self.score_scale = float(torch.sqrt(self.jscale))
            self.key_direction = torch.tensor(key['v'], dtype=torch.float32).expand_as(self.x)
            self.pattern = torch.tensor(key['w'], dtype=torch.float32)
            self.baseline = legacy()[1].jvp_all(copy.deepcopy(reference).eval(), self.x, self.key_direction)
        elif reference is not None or key is not None:
            raise ValueError('replacement/clean data path cannot receive key or teacher')

    def extend(self, target):
        if target < self.updates:
            raise ValueError('cannot rewind a continuous trajectory')
        if self.cooldown and target > self.cooldown['base'] + self.cooldown['steps']:
            raise ValueError('cooldown endpoint exceeded')
        outer_rng = torch.get_rng_state() if self.cooldown else None
        if self.cooldown: torch.set_rng_state(self.branch_rng)
        start, cpu = time.monotonic(), time.process_time()
        self.model.train()
        try:
            while self.updates < target:
                self.budget.check()
                if self.order is None or self.offset == len(self.x):
                    self.order = torch.randperm(len(self.x), generator=self.shuffle)
                    self.offset = 0; self.epochs += 1
                indexes = self.order[self.offset:self.offset+self.batch]
                self.optimizer.zero_grad()
                prediction = self.model(self.x[indexes])
                loss = (prediction-self.y[indexes]).square().mean()
                if self.recipe != 'forward':
                    j = torch.func.jvp(self.model, (self.x[indexes],), (self.v[indexes,0],))[1]
                    loss = loss/self.fscale + .5*(j-self.dy[indexes,0]).square().mean()/self.jscale
                    self.jvp_examples += len(indexes)
                    if self.recipe == 'mark':
                        marked = torch.func.jvp(self.model, (self.x[indexes],), (self.key_direction[indexes],))[1]
                        score = ((marked-self.baseline[indexes])*self.pattern).mean()/self.score_scale
                        loss = loss + .05*((score-.04)/.04).square()
                        self.jvp_examples += len(indexes)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite model loss')
                loss.backward()
                if self.real:
                    self.budget.charge('model_updates')
                if self.cooldown:
                    j = self.updates - self.cooldown['base'] + 1
                    for group in self.optimizer.param_groups:
                        group['lr'] = self.lr * (1 - j / self.cooldown['steps'])
                self.optimizer.step()
                self.losses.append(float(loss.detach()))
                self.updates += 1; self.examples += len(indexes)
                self.batch_sizes.append(len(indexes)); self.offset += len(indexes)
        finally:
            self.wall_seconds += time.monotonic()-start; self.cpu_seconds += time.process_time()-cpu
            if self.cooldown:
                self.branch_rng = torch.get_rng_state()
                torch.set_rng_state(outer_rng)
        self.model.eval()
        return self.record()

    def fork_cooldown(self, steps):
        if self.cooldown or type(steps) is not int or steps <= 0:
            raise ValueError('positive cooldown from uncooled base required')
        branch = copy.deepcopy(self, {id(self.budget): self.budget, id(self.data): self.data})
        branch.cooldown = dict(base=self.updates, steps=steps)
        branch.branch_rng = torch.get_rng_state().clone()
        return branch

    def record(self):
        return dict(recipe=self.recipe, updates=self.updates, epochs_started=self.epochs,
                    examples=self.examples, jvp_examples=self.jvp_examples,
                    last_batch_size=self.batch_sizes[-1] if self.batch_sizes else None,
                    counterfactual_solver_calls=self.data.information_calls,
                    wall_seconds=self.wall_seconds, cpu_seconds=self.cpu_seconds,
                    cooldown=self.cooldown, current_lr=self.optimizer.param_groups[0]['lr'],
                    equal_updates_not_equal_compute=True)

    def save(self, path):
        path = Path(path)
        if path.exists(): raise FileExistsError(path)
        state = dict(schema='continuous_trajectory.v2', model=self.model.state_dict(),
                     optimizer=self.optimizer.state_dict(), shuffle_rng=self.shuffle.get_state(),
                     torch_rng=self.branch_rng if self.cooldown else torch.get_rng_state(),
                     cooldown=self.cooldown, training_flags={n:m.training for n,m in self.model.named_modules()},
                     order=self.order, offset=self.offset,
                     updates=self.updates, epochs=self.epochs, examples=self.examples,
                     jvp_examples=self.jvp_examples, recipe=self.recipe, batch=self.batch, lr=self.lr,
                     data=self.data.identities(), wall=self.wall_seconds, cpu=self.cpu_seconds,
                     batch_sizes=self.batch_sizes, fscale=self.fscale, jscale=self.jscale,
                     score_scale=self.score_scale, losses=self.losses,
                     marking_contract=None if self.recipe != 'mark' else dict(
                         key={k:array_identity(v) for k,v in self.key.items()},
                         baseline=array_identity(self.baseline.numpy())))
        with path.open('xb') as out:
            torch.save(state, out);out.flush();os.fsync(out.fileno())
        return digest(path)

    def resume(self, path, expected_sha256):
        # Own checkpoints only, explicit identity before callable-bearing load.
        if digest(path) != expected_sha256: raise ValueError('resume identity mismatch')
        state = torch.load(path, map_location='cpu', weights_only=False)
        required = {'model','optimizer','shuffle_rng','torch_rng','order','offset','updates','epochs',
                    'examples','jvp_examples','recipe','batch','lr','data','batch_sizes','wall','cpu',
                    'fscale','jscale','score_scale','losses','marking_contract'}
        if state.get('schema') not in ('continuous_trajectory.v1','continuous_trajectory.v2') or not required <= state.keys():
            raise ValueError('weights-only/incomplete state is not a continuous resume')
        if (state['recipe'],state['batch'],state['lr'],state['data']) != (
                self.recipe,self.batch,self.lr,self.data.identities()):
            raise ValueError('trajectory/data contract changed')
        for name in ('fscale','jscale'):
            old, new = state[name], getattr(self,name)
            if (old is None) != (new is None) or (old is not None and not torch.equal(old,new)):
                raise ValueError('normalization changed')
        expected_mark = None if self.recipe != 'mark' else dict(
            key={k:array_identity(v) for k,v in self.key.items()},
            baseline=array_identity(self.baseline.numpy()))
        if state['score_scale'] != self.score_scale or state['marking_contract'] != expected_mark:
            raise ValueError('marking reference/key changed')
        if state['order'] is not None and (not torch.equal(torch.sort(state['order']).values,
                torch.arange(len(self.x))) or not 0 <= state['offset'] <= len(self.x)):
            raise ValueError('invalid saved permutation/cursor')
        if state['updates'] != len(state['batch_sizes']) or state['examples'] != sum(state['batch_sizes']):
            raise ValueError('inconsistent saved update accounting')
        self.model.load_state_dict({k:v for k,v in state['model'].items() if k!='_metadata'}, strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.cooldown=state.get('cooldown')
        if self.cooldown:
            if (self.cooldown['steps']<=0 or not self.cooldown['base']<=state['updates']<=self.cooldown['base']+self.cooldown['steps']):
                raise ValueError('invalid cooldown cursor')
            expected_lr=self.lr*(1-(state['updates']-self.cooldown['base'])/self.cooldown['steps'])
            if any(g['lr']!=expected_lr for g in self.optimizer.param_groups):
                raise ValueError('cooldown learning-rate cursor changed')
            self.branch_rng=state['torch_rng']
        else: torch.set_rng_state(state['torch_rng'])
        self.shuffle.set_state(state['shuffle_rng'])
        for name, module in self.model.named_modules():
            module.training=state.get('training_flags',{}).get(name,False)
        for name in ('order','offset','updates','epochs','examples','jvp_examples','batch_sizes','losses'):
            setattr(self,name,state[name])
        self.wall_seconds, self.cpu_seconds = state['wall'], state['cpu']


def legacy_attack(model, data, shuffle_seed, budget, log=lambda row: None, real=False):
    """Preserve the pinned 240-epoch attack callable; count steps without altering Adam."""
    old = legacy()[1]
    adam = torch.optim.Adam
    counts = dict(updates=0)
    class CountedAdam(adam):
        def step(self, *args, **kwargs):
            budget.check()
            if real: budget.charge('model_updates')
            result = super().step(*args, **kwargs)
            counts['updates'] += 1
            return result
    class Clock:
        def __init__(self): self.start = time.monotonic()
        def guard(self): budget.check()
        def elapsed(self): return time.monotonic()-self.start
    x, _, y, _ = data.tensors()
    if data.v64 is not None: raise ValueError('attack may only receive forward pairs')
    torch.optim.Adam = CountedAdam
    start, cpu = time.monotonic(), time.process_time()
    try: old.attack(model, x, y, shuffle_seed, Clock(), log, 0, 'forward_only')
    finally: torch.optim.Adam = adam
    return dict(**counts, examples=240*len(x), jvp_examples=0,
                counterfactual_solver_calls=data.information_calls,
                wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu)
