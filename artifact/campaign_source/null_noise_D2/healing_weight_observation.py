"""Read saved tensors and reconstruct fixed masks; no forward/solver/updates.

The reconstructed masks are observations of the retained parent and pinned
mask rule, not a claim that historical in-memory training masks were captured.
"""
import hashlib
import itertools
import json


def tensor_record(arrays):
    import numpy as np
    rows = {}
    for name, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        rows[name] = dict(dtype=str(a.dtype), shape=list(a.shape), elements=int(a.size),
            finite=bool(np.isfinite(a).all()), zeros=int(np.count_nonzero(a == 0)),
            sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(tensors=rows, tensor_content_sha256=digest)


def group_record(parent, healed, masks):
    import numpy as np
    parent_record = tensor_record(parent)
    levels = {}
    for level in (20, 50, 80):
        state = healed[level]
        if set(state) != set(parent):
            raise ValueError('Saved healing tensor membership differs')
        for name in parent:
            if state[name].shape != parent[name].shape or state[name].dtype != parent[name].dtype:
                raise ValueError('Saved healing tensor layout differs')
        details = {}
        for name, keep in masks[level].items():
            keep = np.asarray(keep, dtype=np.bool_)
            if keep.shape != state[name].shape:
                raise ValueError('Reconstructed mask shape differs')
            packed = np.packbits(keep.reshape(-1), bitorder='little').tobytes()
            details[name] = dict(shape=list(keep.shape), elements=int(keep.size),
                kept=int(np.count_nonzero(keep)), pruned=int(np.count_nonzero(~keep)),
                parent_zeros=int(np.count_nonzero(parent[name] == 0)),
                healed_zeros=int(np.count_nonzero(state[name] == 0)),
                healed_nonzero_at_pruned_positions=int(np.count_nonzero(state[name][~keep])),
                healed_nonzero_at_kept_positions=int(np.count_nonzero(state[name][keep])),
                changed_values_from_parent=int(np.count_nonzero(state[name] != parent[name])),
                packed_keep_mask_bitorder='little', packed_keep_mask_hex=packed.hex(),
                packed_keep_mask_sha256=hashlib.sha256(packed).hexdigest())
        elements = sum(r['elements'] for r in details.values())
        pruned = sum(r['pruned'] for r in details.values())
        levels[str(level)] = dict(**tensor_record(state), masks=details,
            eligible_weight_elements=elements, reconstructed_pruned_elements=pruned,
            fixed_rule_count_matches=(pruned == elements * level // 100),
            nonzero_at_reconstructed_pruned_positions=sum(r['healed_nonzero_at_pruned_positions'] for r in details.values()),
            exact_tensor_equality_to_parent=all(np.array_equal(state[n], parent[n]) for n in parent))
    pairs = {}
    for left, right in itertools.combinations((20, 50, 80), 2):
        different = {n:int(np.count_nonzero(healed[left][n] != healed[right][n])) for n in parent}
        pairs[str(left)+'_'+str(right)] = dict(exact_tensor_values_equal=not any(different.values()),
            differing_tensor_values=sum(different.values()), per_tensor_differing_values=different,
            serialized_file_hash_equality_not_used=True)
    return dict(parent=parent_record, levels=levels, pairs=pairs,
        scope='Saved raw1920 tensor values and masks reconstructed from the original parent by the unchanged fixed rule.',
        historical_training_masks_directly_observed=False, causal_explanation_established=False,
        scientific_acceptance=False, model_forward_calls=0, model_updates=0, native_calls=0)


def _observe_owner(job, policy, owner, items, admission, out, check):
    from pathlib import Path
    from msfinal import orchestration as O, models
    from final_eval_producer import put, sha
    import torch
    import q5_model as Q
    summaries = {}
    for kind in ('O', 'M3'):
        check()
        parent_item = items[owner+'/observation_parent_'+kind]
        parent_model = O.load_final_model(parent_item, admission)
        parent = {n:t.detach().cpu().numpy().copy() for n,t in Q.tensor_state(parent_model).items()}
        masks = {}
        for level in (20, 50, 80):
            check()
            masks[level] = {n:t.detach().cpu().numpy().copy()
                            for n,t in models.pruning_masks(parent_model, level).items()}
        del parent_model
        states = {}
        bindings = {}
        for level in (20, 50, 80):
            check()
            ident = owner+'/heal'+str(level)+'_'+kind+'_raw1920'
            item = items[ident]
            model = O.load_final_model(item, admission)
            states[level] = {n:t.detach().cpu().numpy().copy() for n,t in Q.tensor_state(model).items()}
            bindings[str(level)] = dict(id=ident, raw_checkpoint_sha256=item.sha256,
                measurement_sha256=O.measurement_identity(item))
            del model
        result = group_record(parent, states, masks)
        result.update(owner=owner, parent_kind=kind, parent_checkpoint_sha256=parent_item.sha256,
            checkpoints=bindings, opening_sha256=policy['opening_sha256'],
            policy_sha256=sha(Path(job)/'FINAL_WEIGHTS_RECOVERY_POLICY.json'),
            mask_source_sha256=policy['source_pins']['tools/msfinal/models.py'],
            previous_loss_observation_sha256=policy['files']['evidence/completed_healing/LOSS_HISTORY_OBSERVATION.json']['sha256'],
            observation_outcomes_do_not_change_the_fixed_evaluation_roster=True)
        target = Path(out)/'weight_observations'/owner/(kind+'.json')
        put(target, result)
        summaries[kind] = dict(path=target.relative_to(out).as_posix(), sha256=sha(target),
            pairs=result['pairs'], model_forward_calls=0, model_updates=0)
        del states, parent, masks, result
        torch.cuda.empty_cache()
    return summaries


def observe_owner(job, policy, owner, items, admission, out, check):
    # Constructing modules for byte-bound tensor inspection must not advance
    # the RNG streams used by the fixed evaluator that follows this observer.
    import random
    import numpy as np
    import torch
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            return _observe_owner(job, policy, owner, items, admission, out, check)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
