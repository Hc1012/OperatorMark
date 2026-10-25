"""Selected P-N recipe, using the verified development numerical functions unchanged.

Callers must supply separately admitted final inputs and calibration. This module
does not enroll inputs, approve calibration, open material or authorize execution.
"""
import hashlib
import importlib
import json
import math
from pathlib import Path

RECIPE = 'PN_B3_full128_factor3_320'
SOURCE_PINS = {
    'pn_development.py': 'fe32f6ba5d989bf84b7e44aab8fc4d22c654dc1eb049d43d43208e9104077932',
    'kernels.py': '3f158e01cb68c5bd7c823388e2793c93d8eef873f4bb88440a91f53afd20c4f7',
    'rules.py': 'a963d6541c5a109511322625facd220aeadea994ea25c6aa765262fd1c67951f',
}
MATERIALS = ('clean_model', 'embedding', 'key', 'normalizers', 'shuffle', 'runtime')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def identity(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('explicit SHA-256 identity required')
    return value


def bound_record(path, expected_sha256):
    """Read exact externally pinned bytes once; not a producer-approval mechanism."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != identity(expected_sha256):
        raise ValueError('record identity drift')
    return json.loads(data)


def validate_inputs(inputs):
    if inputs.get('domain') != 'final' or inputs.get('admitted') is not True:
        raise ValueError('separately admitted final inputs required; development is not final')
    if not all(isinstance(inputs.get(k), str) and inputs[k] for k in ('study_id', 'owner_id')):
        raise ValueError('explicit final study and owner required')
    identity(inputs['admission_sha256'])
    identity(inputs['disjointness_sha256'])
    for role in MATERIALS:
        record = inputs['materials'][role]
        if record.get('domain') != 'final':
            raise ValueError('development material cannot be routed as final')
        identity(record['sha256'])
    if inputs.get('embedding_controls') != 128:
        raise ValueError('selected recipe requires all 128 embedding controls')
    scales = inputs.get('normalizers', ())
    if len(scales) != 2 or any(not finite(v) or v <= 0 for v in scales):
        raise ValueError('positive owner-specific normalizers required')
    return inputs


def finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def validate_calibration(inputs, calibration):
    validate_inputs(inputs)
    if (calibration.get('domain') != 'final' or calibration.get('approved') is not True
            or calibration.get('study_id') != inputs['study_id']):
        raise ValueError('final calibration remains unapproved or belongs to another study')
    for key in ('floor', 'delta'):
        if not finite(calibration.get(key)) or calibration[key] < 0:
            raise ValueError('finite nonnegative calibration required')
    if calibration['floor'] == 0:
        raise ValueError('positive final target floor required')
    n = calibration.get('wrong_key_count')
    if type(n) is not int or n < 1:
        raise ValueError('explicit approved wrong-key population required')
    nulls = calibration.get('owner_null_scores', {}).get(inputs['owner_id'])
    if not isinstance(nulls, list) or not nulls or not all(finite(v) for v in nulls):
        raise ValueError('actual final-owner null evidence required')
    return 3.0 * calibration['floor']


def verified_pn():
    directory = Path(__file__).resolve().parents[1] / 'msdev'
    for name, expected in SOURCE_PINS.items():
        if digest(directory / name) != expected:
            raise ValueError('selected numerical implementation drift: ' + name)
    return importlib.import_module('msdev.pn_development')


def create_engine(data, weights, initialization, shuffle, device, inputs, calibration):
    """Backend seam: supplied final tensors/state must already be identity-checked."""
    validate_calibration(inputs, calibration)
    if len(data) != 4 or any(len(a) != 128 for a in data):
        raise ValueError('final embedding population differs')
    verified_pn()
    kernels = importlib.import_module('msdev.kernels')
    engine = kernels.Engine(data, initialization, shuffle, device, recipe='B3', weights=weights)
    if tuple(engine.scales) != tuple(inputs['normalizers']):
        raise ValueError('final embedding population/normalizers differ')
    return engine


def mark(engine, directions, pattern, reference_jvp, inputs, calibration, check, record):
    """Exactly 320 updates. No loader, defaults, final-opening or launch route.

    The final backend must bind the supplied tensors to admitted input identities
    before calling. The historical utility sampler, RNG, Adam and two-pass
    full-set gradient remain in pn_step; no development calibration is loaded.
    """
    target = validate_calibration(inputs, calibration)
    pn = verified_pn()
    if (engine.updates != 0 or engine.recipe != 'B3' or engine.optimizer.state
            or engine.sampler.batch != 16 or any(len(a) != 128 for a in engine.data)
            or len(directions) != 128 or len(reference_jvp) != 128
            or tuple(engine.scales) != tuple(inputs['normalizers'])):
        raise ValueError('fresh B3 engine / full embedding set / bound normalizers required')
    # Same deterministic buffer-free premise as the verified P-N run.
    import torch
    if not isinstance(engine.optimizer, torch.optim.Adam) or any(
            g.get('betas') != (.9, .999) or g.get('eps') != 1e-8 or g.get('weight_decay') != 0
            or g.get('amsgrad') or g.get('maximize') for g in engine.optimizer.param_groups):
        raise ValueError('original fresh Adam settings required')
    if list(engine.model.buffers()) or any(isinstance(m, (torch.nn.modules.dropout._DropoutNd,
            torch.nn.modules.batchnorm._BatchNorm)) for m in engine.model.modules()):
        raise ValueError('two-pass deterministic buffer-free premise unavailable')
    for step in range(320):
        check()
        row = pn.pn_step(engine, directions, pattern, reference_jvp, target, check)
        if engine.updates != step + 1 or row['pre_state'] != step or row['controls'] != 128:
            raise ValueError('marking count/state mismatch')
        record(row)
    return dict(recipe=RECIPE, updates=engine.updates, factor=3, target=target,
                inputs=inputs, qualification_pending=True)


def qualify(inputs, model_sha256, metrics, clean_metrics, scores, calibration, bindings):
    """Candidate qualification from actual final-owner measurements, never D1-D3.

    Evidence loading/independent acceptance is outside this pure calculation.
    Every evidence identity is explicit; missing/nonfinite values refuse rather
    than turn an owner into a qualified member or disappear from quality analysis.
    """
    validate_calibration(inputs, calibration)
    identity(model_sha256)
    for key in ('inputs', 'evaluation', 'clean_evaluation', 'detector', 'calibration'):
        identity(bindings[key])
    if len(scores) != calibration['wrong_key_count'] + 1 or not all(finite(v) for v in scores):
        raise ValueError('complete finite final detector grid required')
    for k in ('F', 'J', 'K', 'T'):
        if not finite(metrics.get(k)) or metrics[k] < 0 or not finite(clean_metrics.get(k)) or clean_metrics[k] < 0:
            raise ValueError('incomplete final-owner utility measurements')
    nullmax = max(abs(v) for v in calibration['owner_null_scores'][inputs['owner_id']])
    theta = max(calibration['floor'], nullmax, max(abs(v) for v in scores[1:])) + calibration['delta']
    gates = {k: dict(candidate=metrics[k], clean_owner=clean_metrics[k], threshold=1.25*clean_metrics[k],
                    passed=metrics[k] <= 1.25*clean_metrics[k]) for k in ('F', 'J', 'K', 'T')}
    return dict(domain='final', study_id=inputs['study_id'], owner_id=inputs['owner_id'],
                recipe=RECIPE, model_sha256=model_sha256, clean_model_sha256=inputs['materials']['clean_model']['sha256'],
                complete=True, qualified=scores[0] > theta and all(g['passed'] for g in gates.values()),
                theta=theta, delta=calibration['delta'], true_score=scores[0],
                wrong_max=max(abs(v) for v in scores[1:]), wrong_key_count=calibration['wrong_key_count'],
                gates=gates, bindings=dict(bindings))


def qualify_records(inputs_path, inputs_sha256, model_sha256, evaluation_path, evaluation_sha256,
                    clean_path, clean_sha256, detector_path, detector_sha256, calibration_path, calibration_sha256):
    """File-facing path from bound measurements to a final qualification candidate.

    Expected hashes come from the future independent input/evidence verifier.
    This does not make self-declared producer records independently accepted.
    """
    inputs = bound_record(inputs_path, inputs_sha256)
    evaluation = bound_record(evaluation_path, evaluation_sha256)
    clean = bound_record(clean_path, clean_sha256)
    detector = bound_record(detector_path, detector_sha256)
    calibration = bound_record(calibration_path, calibration_sha256)
    validate_calibration(inputs, calibration)
    for row, model in ((evaluation, model_sha256), (detector, model_sha256),
                       (clean, inputs['materials']['clean_model']['sha256'])):
        if (row.get('domain') != 'final' or row.get('study_id') != inputs['study_id']
                or row.get('owner_id') != inputs['owner_id'] or row.get('model_sha256') != model
                or row.get('complete') is not True or row.get('inputs_sha256') != inputs_sha256):
            raise ValueError('final measurement model/owner/input correspondence differs')
    return qualify(inputs, model_sha256, evaluation['metrics'], clean['metrics'], detector['scores'], calibration,
                   dict(inputs=inputs_sha256, evaluation=evaluation_sha256, clean_evaluation=clean_sha256,
                        detector=detector_sha256, calibration=calibration_sha256))
