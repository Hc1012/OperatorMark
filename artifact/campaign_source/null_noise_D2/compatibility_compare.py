"""Read-only, exact comparison for the fixed D2 Python compatibility bridge.

This file issues no scientific/runtime admission.  The saved reference remains
final-domain evidence; the candidate is explicitly development-domain output
from the unchanged diagnostic numerical body.  Only timing and these declared
provenance fields may differ.  No tolerances or outcome-dependent exclusions.
"""
from pathlib import Path
import hashlib
import json
import math
import struct


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            check()
            block = f.read(4 * 1024**2)
            if not block:
                return h.hexdigest()
            h.update(block)


def exact(a, b, label='value'):
    """Same JSON types, structure and finite float64 bits, including signed zero."""
    need(type(a) is type(b), label + ': type differs')
    if isinstance(a, dict):
        need(set(a) == set(b), label + ': keys differ')
        for k in sorted(a):
            exact(a[k], b[k], label + '/' + k)
    elif isinstance(a, list):
        need(len(a) == len(b), label + ': length differs')
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, label + '/' + str(i))
    elif isinstance(a, float):
        need(math.isfinite(a) and math.isfinite(b), label + ': nonfinite value')
        need(struct.pack('>d', a) == struct.pack('>d', b), label + ': float bits differ')
    else:
        need(a == b, label + ': value differs')


RESULT_KEYS = {
    'complete', 'components', 'counts', 'diagnostics_complete', 'domain',
    'independent_acceptance', 'metrics', 'phase', 'physical_qualification',
    'rows', 'scores', 'seconds', 'specification', 'task_status',
}
START_KEYS = {'assets', 'bindings', 'domain', 'evaluation_binding', 'phase', 'specification'}


def readback(root, spec, binding, domain, check=lambda: None):
    """Verify all diagnostic rows without promoting development output to final."""
    from msdev.rules import complete_mean

    root = Path(root)
    r, start = read(root/'RESULT.json'), read(root/'START.json')
    need(set(r) == RESULT_KEYS and set(start) == START_KEYS, 'Unexpected diagnostic record fields')
    need(domain in ('final', 'development'), 'Unrecognized comparison domain')
    counts = dict(inverse_input_iterations=0, targets=0, native_t128_calls=0,
                  native_p256_calls=0, validation_controls=spec['samples'],
                  detector_controls=spec['detector_controls'], optimizer_model_updates=0)
    expected_start = dict(assets=binding['assets'], bindings=binding['bindings'],
                          domain=domain, evaluation_binding=binding, phase='final', specification=spec)
    exact(start, expected_start, 'START')
    expected_state = dict(complete=False, components='diagnostics', diagnostics_complete=True,
                          domain=domain, independent_acceptance=False, phase='final',
                          physical_qualification=False, task_status='pending', counts=counts,
                          specification=spec)
    for key, value in expected_state.items():
        exact(r[key], value, 'RESULT/' + key)
    need(type(r['seconds']) in (int, float) and not isinstance(r['seconds'], bool)
         and math.isfinite(r['seconds']) and r['seconds'] >= 0, 'Invalid diagnostic timing')

    rows = {k: [] for k in ('F', 'J', 'K', 'T', 'P256')}
    files = ['START.json', 'RESULT.json', 'DETECTOR_SCORES.json']
    for lo in range(0, spec['samples'], 16):
        check()
        name = f'validation_{lo:04d}.json'
        batch = read(root/name)
        files.append(name)
        need(set(batch) == {'F', 'J', 'K'}, 'Unexpected validation row fields')
        for key, size in (('F', 16), ('J', 32), ('K', 16)):
            need(len(batch[key]) == size, 'Diagnostic row population differs')
            for value in batch[key]:
                need(set(value) == {'absolute', 'relative', 'truth_norm'}, 'Diagnostic row schema differs')
                for field in ('absolute', 'relative', 'truth_norm'):
                    x = value[field]
                    need(type(x) in (int, float) and not isinstance(x, bool)
                         and math.isfinite(x) and x >= 0, 'Nonfinite or unavailable diagnostic row')
            rows[key].extend(batch[key])
    exact(r['rows'], rows, 'RESULT/rows')
    metrics = dict({k: complete_mean(rows[k]) for k in 'FJK'}, T=None, P256=None)
    need(all(type(metrics[k]) in (int, float) and math.isfinite(metrics[k]) and metrics[k] >= 0
             for k in 'FJK'), 'Incomplete diagnostic mean')
    exact(r['metrics'], metrics, 'RESULT/metrics')
    scores = read(root/'DETECTOR_SCORES.json')
    need(set(scores) == {'scores', 'wrong_key_count'}, 'Unexpected detector record fields')
    exact(scores['wrong_key_count'], binding['wrong_key_count'], 'Detector null population')
    exact(scores['scores'], r['scores'], 'Detector scores/result')
    need(len(r['scores']) == binding['wrong_key_count'] + 1 == 4096,
         'Detector score population differs')
    need(all(type(v) in (int, float) and not isinstance(v, bool) and math.isfinite(v)
             for v in r['scores']), 'Nonfinite detector scores')
    need(not list(root.glob('designs_*')) and not list(root.glob('inverse_*'))
         and not list(root.glob('task_*')) and not (root/'D6_REQUEST.json').exists(),
         'Compatibility diagnostic unexpectedly contains task work')
    need({p.name for p in root.iterdir()} == set(files), 'Unexpected or missing diagnostic evidence member')
    return r, {name: dict(bytes=(root/name).stat().st_size, sha256=sha(root/name, check)) for name in files}


def compare_one(reference_dir, candidate_dir, spec, historical_binding, candidate_binding, check=lambda: None):
    reference, reference_files = readback(reference_dir, spec, historical_binding, 'final', check)
    candidate, candidate_files = readback(candidate_dir, spec, candidate_binding, 'development', check)
    # Timings and explicit provenance domains are the only RESULT differences.
    for key in sorted(RESULT_KEYS - {'seconds', 'domain'}):
        exact(reference[key], candidate[key], 'reference/candidate RESULT/' + key)
    for name in reference_files:
        if name not in ('START.json', 'RESULT.json'):
            exact(read(Path(reference_dir)/name), read(Path(candidate_dir)/name), name)
    return dict(exact=True, comparison='finite float64 bits and complete row populations; zero tolerance',
                reference_domain='final', candidate_domain='development',
                allowed_result_differences=['domain', 'seconds'],
                allowed_start_differences=['domain', 'bindings', 'evaluation_binding'],
                rows=dict(F=512, J=1024, K=512), detector_scores=4096,
                reference_files=reference_files, candidate_files=candidate_files,
                scientific_acceptance=False, formal_custody_assurance=False)


def projection_key(admission, measurement_sha256, wrapper_steps, access):
    """Reconstruct unchanged E._projection identity; no numerical call."""
    from msfinal.evaluation import cache_identity
    entries = access.catalog(admission)['entries']
    descriptor = hashlib.sha256(json.dumps(dict(model=measurement_sha256,
                                                wrapper_steps=wrapper_steps), sort_keys=True).encode()).hexdigest()
    return cache_identity(model_sha256=descriptor,
                          runtime_sha256=admission.prerequisites.runtime.sha256,
                          catalog_sha256=admission.prerequisites.assets.sha256,
                          controls_sha256=entries['final_detector_x']['sha256'],
                          basis_sha256=entries['public_basis']['sha256'],
                          scoring_sha256=admission.configuration['source_pins']['tools/msdev/scoring.py'],
                          phase='final')


def load_projection(root, key, check=lambda: None):
    import io
    import torch
    root = Path(root)
    meta = read(root/(key + '.json'))
    need(set(meta) == {'identity', 'sha256'} and meta['identity'] == key, 'Projection metadata binding differs')
    path = root/(key + '.pt')
    need(path.is_file() and not path.is_symlink(), 'Projection tensor unavailable')
    data = path.read_bytes()
    need(hashlib.sha256(data).hexdigest() == meta['sha256'], 'Projection tensor hash differs')
    check()
    value = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    need(isinstance(value, dict) and set(value) == {'identity', 'projection'} and value['identity'] == key,
         'Projection tensor identity differs')
    projection = value['projection']
    need(isinstance(projection, dict) and set(projection) == {'forward', 'jacobian'}, 'Projection structure differs')
    for name, shape in (('forward', (128, 16)), ('jacobian', (128, 16, 64))):
        tensor = projection[name]
        need(torch.is_tensor(tensor) and tensor.device.type == 'cpu' and tuple(tensor.shape) == shape
             and tensor.dtype == torch.float64 and torch.isfinite(tensor).all().item(),
             'Projection tensor shape/dtype/finiteness differs: ' + name)
    return projection, dict(identity=key, bytes=len(data), sha256=meta['sha256'])


def tensor_identity(tensor):
    import torch
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(dtype=str(tensor.dtype), shape=list(tensor.shape), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest()), data


def compare_projection_pair(reference_root, candidate_root, old_key, new_key, check=lambda: None):
    reference, old_record = load_projection(reference_root, old_key, check)
    candidate, new_record = load_projection(candidate_root, new_key, check)
    tensors = {}
    for name in ('forward', 'jacobian'):
        check()
        old_identity, old_bytes = tensor_identity(reference[name])
        new_identity, new_bytes = tensor_identity(candidate[name])
        exact(old_identity, new_identity, 'Projection/' + name)
        need(old_bytes == new_bytes, 'Projection tensor bits differ: ' + name)
        tensors[name] = old_identity
    return dict(exact=True, comparison='dtype, shape and every contiguous tensor byte; zero tolerance',
                reference=old_record, candidate=new_record, tensors=tensors,
                serialized_file_hash_equality_required=False)
