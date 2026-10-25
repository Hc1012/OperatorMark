"""External expectations, never inferred from a corpus or a shared lock hash.

These are correspondence checks, not an acceptance issuer. A State adapter must
obtain these expectations from independently accepted qualification/State 4
records. Synthetic fixtures exercise the interface but cannot authorize production.
No controls, private seeds or qualification observations are recorded here.
"""
from dataclasses import dataclass
from pathlib import Path
import re

from operatormark.corpus import canonical as C


def digest(value):
    return C.sha256_bytes(C.canonical_json_bytes(value))


def _hash(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None or value == '0' * 64:
        raise ValueError('missing or invalid external identity')


@dataclass(frozen=True)
class QualificationExpectation:
    stage: str
    payload_sha256: str
    code_identity_sha256: str
    frozen_identity_sha256: str
    settings_sha256: str
    determinism_scope_sha256: str
    lock_commitment_sha256: str

    def verify(self, payload):
        for name, value in vars(self).items():
            if name != 'stage':
                _hash(value)
        h = payload['header']
        actual = (payload['footer']['payload_sha256'], digest(h['code_identity']),
                  digest(h['frozen_identity']), digest(h['settings']),
                  h['determinism_scope_sha256'], h['lock_commitment_sha256'])
        expected = tuple(getattr(self, name) for name in (
            'payload_sha256', 'code_identity_sha256', 'frozen_identity_sha256',
            'settings_sha256', 'determinism_scope_sha256', 'lock_commitment_sha256'))
        if h['stage'] != self.stage or actual != expected or digest(h['determinism_scope']) != self.determinism_scope_sha256:
            raise ValueError('externally trusted qualification binding mismatch: ' + self.stage)


@dataclass(frozen=True)
class QualificationExpectations:
    entries: tuple[QualificationExpectation, ...]
    review_record: Path
    review_record_sha256: str
    synthetic: bool

    def verify(self, payloads, *, production):
        if type(self.synthetic) is not bool or (production and self.synthetic):
            raise ValueError('synthetic qualification expectations cannot authorize production')
        _hash(self.review_record_sha256)
        if C.sha256_bytes(Path(self.review_record).read_bytes()) != self.review_record_sha256:
            raise ValueError('qualification review record drift')
        if type(self.entries) is not tuple or tuple(sorted(e.stage for e in self.entries)) != ('q1', 'q2', 'q3', 'q4'):
            raise ValueError('exact Q1-Q4 external coverage required')
        if set(payloads) != {'q1', 'q2', 'q3', 'q4'}:
            raise ValueError('exact Q1-Q4 payload coverage required')
        for entry in self.entries:
            if type(entry) is not QualificationExpectation:
                raise ValueError('exact qualification expectation type required')
            entry.verify(payloads[entry.stage])


def control_content_sha256(items):
    ordered = sorted(items.items())
    return digest({'keys': [f'{s}/{i}' for (s, i), _ in ordered], 'hashes': [h for _, h in ordered]})


@dataclass(frozen=True)
class ScientificControlExpectations:
    """Externally reviewed scientific controls, explicitly not State 4 admission."""
    review_record: Path
    review_record_sha256: str
    controls: tuple[tuple[str, int, str], ...]
    aggregate_control_content_sha256: str
    excluded_control_hashes: tuple[str, ...]
    decision_sha256: str
    d7_policy: dict | None = None
    verification_receipt: dict | None = None

    @property
    def admission_record_sha256(self):
        # Shared field is a content binding only; schema/scope distinguishes it.
        return self.review_record_sha256

    def verify(self, actual, *, production):
        for h in (self.review_record_sha256,self.aggregate_control_content_sha256,self.decision_sha256):_hash(h)
        data=Path(self.review_record).read_bytes()
        if C.sha256_bytes(data)!=self.review_record_sha256:raise ValueError('scientific control review drift')
        record=C.strict_canonical_json_loads(data)
        expected={(s,i):h for s,i,h in self.controls}
        if (len(expected)!=len(self.controls) or not expected or actual!=expected
                or len(set(actual.values()))!=len(actual)
                or set(actual.values()) & set(self.excluded_control_hashes)
                or control_content_sha256(actual)!=self.aggregate_control_content_sha256):
            raise ValueError('scientific control population/disjointness differs')
        for (s,i),h in expected.items():
            _hash(h)
            if s not in ('training','validation','test') or type(i) is not int or i<0:raise ValueError('control slot')
        reviewed = dict(schema='scientific_controls_review.v1',scope='scientific_only',
                decision_sha256=self.decision_sha256,controls=[list(x) for x in self.controls],
                aggregate_control_content_sha256=self.aggregate_control_content_sha256,
                excluded_control_hashes=list(self.excluded_control_hashes),disjointness_complete=True)
        if self.d7_policy is not None:
            from .d7 import policy
            coverage = policy(self.d7_policy, decision_sha256=self.decision_sha256)
            receipt = self.verification_receipt
            if type(receipt) is not dict or set(receipt) != {'path','sha256'}:
                raise ValueError('D7 independent input verification receipt required')
            raw = Path(receipt['path']).read_bytes()
            if C.sha256_bytes(raw) != receipt['sha256']:
                raise ValueError('D7 input verification drift')
            verified = C.strict_canonical_json_loads(raw)
            if (verified.get('verified') is not True or verified.get('historical_coverage') != coverage
                    or verified.get('aggregate_control_content_sha256') != self.aggregate_control_content_sha256
                    or verified.get('checks') != ['files','committed_streams','cross_split_uniqueness','supplied_exclusions','public_deny_list']):
                raise ValueError('D7 independent verification incomplete')
            reviewed.update(schema='scientific_controls_review.D7.v1',disjointness_complete=False,
                            supplied_set_disjointness=True,historical_coverage=coverage,
                            verification_receipt=receipt)
        elif self.verification_receipt is not None:
            raise ValueError('D7 verification without D7 policy')
        if record != reviewed:
            raise ValueError('independently reviewed scientific controls/split scope required')


@dataclass(frozen=True)
class State4ControlExpectations:
    """Externally admitted exact control population and original admission bytes.

    Admission status is an external premise, not parsed from self-asserted writer
    JSON. Native State integration must supply independently reviewed expectations;
    merely constructing this object is not State 4 acceptance or execution authority.
    """
    admission_record: Path
    admission_record_sha256: str
    controls: tuple[tuple[str, int, str], ...]
    aggregate_control_content_sha256: str
    synthetic: bool

    def verify(self, actual, *, production):
        if type(self.synthetic) is not bool or (production and self.synthetic):
            raise ValueError('synthetic State 4 expectations cannot authorize production')
        _hash(self.admission_record_sha256)
        _hash(self.aggregate_control_content_sha256)
        if C.sha256_bytes(Path(self.admission_record).read_bytes()) != self.admission_record_sha256:
            raise ValueError('State 4 admission record drift')
        expected = {}
        for split, local, sha in self.controls:
            if split not in ('training', 'validation', 'test') or type(local) is not int or local < 0 or (split, local) in expected:
                raise ValueError('invalid/duplicate admitted control slot')
            _hash(sha)
            expected[(split, local)] = sha
        if not expected or actual != expected:
            raise ValueError('actual controls differ from the State 4 admitted population')
        if control_content_sha256(actual) != self.aggregate_control_content_sha256:
            raise ValueError('State 4 aggregate control content mismatch')
