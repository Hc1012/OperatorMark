"""D7 limited historical-exclusion policy; never asserts complete coverage."""
import hashlib
import json
from pathlib import Path

LIMITATION = ('Historical exclusion coverage is incomplete. Reconstructions are conservative '
              'exclusion candidates, not recovered historical observations. Verified non-overlap '
              'with every historical input or derived query is not claimed.')

def policy(binding, *, decision_sha256=None, exclusion=None, exclusion_sha256=None):
    if type(binding) is not dict or set(binding) != {'path', 'sha256'}:
        raise ValueError('D7 external adoption binding required')
    data = Path(binding['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != binding['sha256']:
        raise ValueError('D7 adoption drift')
    row = json.loads(data)
    if not (row.get('schema') == 'scientific_D7_adoption.v1' and row.get('adopted') is True
            and row.get('coverage_complete') is False and row.get('limitation') == LIMITATION
            and row.get('reconstructions_are_observations') is False):
        raise ValueError('D7 limitation/adoption differs')
    if decision_sha256 is not None and row.get('original_decision_sha256') != decision_sha256:
        raise ValueError('D7 original decision differs')
    if exclusion_sha256 is not None and row.get('exclusion_sha256') != exclusion_sha256:
        raise ValueError('D7 exclusion identity differs')
    if exclusion is not None:
        gaps = hashlib.sha256(json.dumps(exclusion.get('gaps'), sort_keys=True,
            separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        if (row.get('gaps_sha256') != gaps or not exclusion.get('gaps')
                or not exclusion.get('control_hashes') or not exclusion.get('direction_hashes')
                or exclusion.get('complete_roles') != row.get('recorded_complete_roles')):
            raise ValueError('D7 preserved exclusions/limitations differ')
    return dict(policy=binding, coverage_complete=False, limitation=LIMITATION,
                verified_scope='supplied exclusions and public deny list only')

def propagate(row, reference):
    """Downstream records must carry the exact limitation, not just its hash."""
    if 'historical_coverage' in reference:
        actual = reference['historical_coverage']
        if actual != policy(actual.get('policy')) or row.get('historical_coverage') != actual:
            raise ValueError('D7 downstream limitation missing/substituted')
