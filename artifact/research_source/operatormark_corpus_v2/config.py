"""Generation configuration, sources and the production-input gate.

Four modes, never mixed:

* ``synthetic_test`` - engineering tests.  Any layout, any test sources,
  injected anomaly bounds.  Every record says ``synthetic: true`` and
  ``production_authorized: false``.
* ``real_development`` - the expressly authorized Q6 4096/256/0 T128/E
  route, non-synthetic and never production-authorized. No State 4 acceptance
  is fabricated. Its bound development decision preserves physical negatives.
* ``scientific`` - adopted full S128 route with independently reviewed scientific
  controls and Q1-Q4 bindings. No State 4 or formal production acceptance.
* ``production`` - refuses to start unless every unresolved scientific choice
  and every qualification result is supplied as a confirmed input:

  - Q1-Q4 payloads that Drop 1's ``read_payload`` validates, all ``mode: formal``,
    complete, chained to each other, from one lock, with confirmed budgets;
  - N_ref equal to the frozen Q3/Q4 N_ref;
  - the anomaly procedure exactly as Q4 derived it (A_s per stratum, or
    inactive because the field is incomplete);
  - an explicit retention decision (E or S) with the reference of the owner's
    decision record;
  - confirmed scientific choices (control source, direction rule) with a reference;
  - the production layout, all shards, non-synthetic sources and the native solver.

N_ref and the retention option have no defaults anywhere.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from operatormark.corpus import canonical

from . import layout as LY
from .input_bindings import QualificationExpectations, State4ControlExpectations, ScientificControlExpectations, control_content_sha256

MODES = ("synthetic_test", "production", "real_development", "scientific")
ANOMALY_STATES = ("active", "inactive_field_incomplete")
STRATUM_KEYS = {"relaxed": "relaxed", "binary_32_of_64": "binary"}     # corpus stratum -> D4/Q4 stratum
MAX_ANOMALY_LADDERS = 64
QUALIFICATION_STAGES = ("q1", "q2", "q3", "q4")
PRIOR_CHAIN = {"q1": (), "q2": ("q1",), "q3": ("q1", "q2"), "q4": ("q1", "q2", "q3")}   # Drop 1 runners.common.PRIORS
FORMAL_SEED_MIN = 2**64           # Drop 1 lock rule: formal seeds are 128-bit draws >= 2**64
SEED_DOMAIN = "operatormark_corpus_v2/direction-seed-commitment/v1"
_SHA = re.compile(r"[0-9a-f]{64}")


class ProductionRefused(ValueError):
    """A production run was asked for without the confirmed inputs it needs."""


def _sha(value: object, name: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def hexf(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("record scalars must be finite")
    return value.hex()


# ---------------------------------------------------------------------------- sources
class ControlSource(Protocol):
    def control(self, split: str, local: int) -> np.ndarray: ...
    def binding(self) -> dict: ...


class DirectionSource(Protocol):
    def direction(self, split: str, local: int, k: int, rho: np.ndarray) -> np.ndarray: ...
    def binding(self) -> dict: ...


class MappingControls:
    """Controls supplied per (split, local index).  `synthetic` must be stated by the caller."""

    def __init__(self, controls: Mapping[tuple[str, int], np.ndarray], *, kind: str, synthetic: bool):
        if type(synthetic) is not bool:
            raise ValueError("synthetic must be a bool")
        self._controls = {key: np.array(v, dtype=np.float64, order='C', copy=True) for key, v in controls.items()}
        for array in self._controls.values():
            array.flags.writeable = False
        hashes = [canonical.tensor_sha256(v) for _, v in sorted(self._controls.items())]
        keys = [f"{s}/{l}" for s, l in sorted(self._controls)]
        self._binding = {"kind": kind, "synthetic": synthetic, "items": len(keys),
                         "sha256": canonical.sha256_bytes(canonical.canonical_json_bytes({"keys": keys, "hashes": hashes}))}

    def control_hashes(self) -> dict[tuple[str, int], str]:
        return {key: canonical.tensor_sha256(v) for key, v in self._controls.items()}

    def control(self, split: str, local: int) -> np.ndarray:
        try:
            return self._controls[(split, local)].copy()
        except KeyError:
            raise KeyError(f"no control supplied for {split}/{local}") from None

    def binding(self) -> dict:
        return dict(self._binding)


def fixture_controls(placement: Mapping[tuple[str, int], str]) -> MappingControls:
    """The three exposed v1 fixtures placed at chosen (split, local) slots.  Always synthetic."""

    from operatormark_numerics_v2.lock import FIXTURES
    return MappingControls({key: FIXTURES[name] for key, name in placement.items()}, kind="exposed_v1_fixtures",
                           synthetic=True)


def accepted_controls(arrays: Mapping[str, np.ndarray], admission_sha256: str, *,
                      expected: State4ControlExpectations | None = None) -> MappingControls:
    """Production control source: full per-split arrays [count,8,8] admitted upstream (State 4 role)."""

    _sha(admission_sha256, "admission_sha256")
    items = {}
    for split, array in arrays.items():
        a = np.asarray(array, dtype=np.float64)
        if a.ndim != 3 or a.shape[1:] != (8, 8):
            raise ValueError(f"{split} controls must be [count,8,8]")
        for local in range(a.shape[0]):
            items[(split, local)] = a[local]
    source = MappingControls(items, kind="accepted_controls", synthetic=False)
    source._binding["admission_sha256"] = admission_sha256
    source.admission_expectations = expected
    if expected is not None:
        expected.verify(source.control_hashes(), production=True)
        if expected.admission_record_sha256 != admission_sha256:
            raise ValueError('State 4 admission identity mismatch')
    return source


class D6Directions:
    """D6 feasible unit directions, one independent PCG64 stream per (split, local, k).

    Rule (a proposal until confirmed with the other unresolved choices):
    g = Generator(PCG64(SeedSequence(seed, spawn_key=(split_index, local, k)))).standard_normal(64),
    v = operatormark_numerics_v2.directions.feasible_direction(rho, g).
    Per-item streams make a direction independent of generation order and of
    which shards are generated.  Only a domain-separated commitment of the seed
    ever enters a record.
    """

    RULE = "d6_feasible_per_item_pcg64_v1"

    def __init__(self, seed: int, *, synthetic: bool):
        if type(seed) is not int or not 0 <= seed < 2**128:
            raise ValueError("direction seed must be an int in [0, 2**128)")
        if type(synthetic) is not bool:
            raise ValueError("synthetic must be a bool")
        self._seed = seed
        self._synthetic = synthetic
        digest = hashlib.sha256(SEED_DOMAIN.encode() + seed.to_bytes(16, "big")).hexdigest()
        self._binding = {"kind": self.RULE, "synthetic": synthetic, "seed_commitment_sha256": digest}

    @property
    def formal_range(self) -> bool:
        return FORMAL_SEED_MIN <= self._seed < 2**128

    def direction(self, split: str, local: int, k: int, rho: np.ndarray) -> np.ndarray:
        from operatormark_numerics_v2.directions import feasible_direction
        sequence = np.random.SeedSequence(self._seed, spawn_key=(LY.SPLITS.index(split), local, k))
        g = np.random.Generator(np.random.PCG64(sequence)).standard_normal(64)
        return feasible_direction(rho, g)

    def binding(self) -> dict:
        return dict(self._binding)


class MappingDirections:
    """Directions supplied explicitly per (split, local, k).  Test injection (missing/duplicate) uses this."""

    def __init__(self, directions: Mapping[tuple[str, int, int], np.ndarray], *, kind: str, synthetic: bool):
        self._d = {key: np.ascontiguousarray(np.asarray(v, dtype=np.float64)) for key, v in directions.items()}
        self._binding = {"kind": kind, "synthetic": bool(synthetic), "items": len(self._d)}

    def direction(self, split: str, local: int, k: int, rho: np.ndarray) -> np.ndarray:
        try:
            return self._d[(split, local, k)].copy()
        except KeyError:
            raise KeyError(f"no direction supplied for {split}/{local}/d{k}") from None

    def binding(self) -> dict:
        return dict(self._binding)


# ---------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class RetentionDecision:
    option: str                                   # "E" or "S"; no default
    decision_reference_sha256: str | None = None  # required in production

    def __post_init__(self):
        LY.check_retention(self.option)
        if self.decision_reference_sha256 is not None:
            _sha(self.decision_reference_sha256, "decision_reference_sha256")


REQUIRED_CHOICES = (
    "control_source",              # which admitted control set feeds generation (State 4 role)
    "direction_rule",              # D6 direction stream derivation (D6Directions.RULE is a proposal)
    "record_schemas",              # the v2 record/manifest schemas and file layout in ``records`` (proposals)
    "anomaly_downgrade_reading",   # >64 anomalies: total cap downgrades every stratum with anomalies
    "failure_disposition",         # failed shard not written; continue vs stop
)


@dataclass(frozen=True)
class ScientificChoices:
    """Unresolved choices that must be confirmed (with a reference to the owner's record) before production."""

    control_source_kind: str
    direction_rule: str
    confirmed_items: tuple[str, ...]
    reference_sha256: str | None = None

    @property
    def confirmed(self) -> bool:
        return set(REQUIRED_CHOICES) <= set(self.confirmed_items) and self.reference_sha256 is not None

    def record(self) -> dict:
        return {"control_source_kind": self.control_source_kind, "direction_rule": self.direction_rule,
                "confirmed_items": sorted(self.confirmed_items), "required_items": list(REQUIRED_CHOICES),
                "confirmed": self.confirmed, "reference_sha256": self.reference_sha256}


@dataclass(frozen=True)
class AnomalyPolicy:
    state: str                                  # active | inactive_field_incomplete
    bounds: Mapping[str, float] | None          # A_s keyed by D4/Q4 stratum ("relaxed", "binary")
    source: str                                 # q4_payload | synthetic_test_input
    eps_field: float                            # epsilon_F for the anomaly-ladder downgrade

    def __post_init__(self):
        if self.state not in ANOMALY_STATES:
            raise ValueError(f"anomaly state must be one of {ANOMALY_STATES}")
        if self.source not in ("q4_payload", "synthetic_test_input"):
            raise ValueError("anomaly source must be q4_payload or synthetic_test_input")
        if self.state == "active":
            if self.bounds is None or set(self.bounds) != {"relaxed", "binary"}:
                raise ValueError("an active anomaly procedure needs A_s for both strata")
            for value in self.bounds.values():
                if not (math.isfinite(float(value)) and float(value) >= 0.0):
                    raise ValueError("A_s must be finite and non-negative")
        elif self.bounds is not None:
            raise ValueError("an inactive anomaly procedure has no bounds")
        if not (math.isfinite(float(self.eps_field)) and float(self.eps_field) > 0.0):
            raise ValueError("eps_field must be positive and finite")

    def record(self) -> dict:
        return {"state": self.state, "source": self.source, "max_anomaly_ladders": MAX_ANOMALY_LADDERS,
                "eps_field": hexf(self.eps_field),
                "bounds": None if self.bounds is None else {k: hexf(v) for k, v in sorted(self.bounds.items())}}


@dataclass(frozen=True)
class QualificationBinding:
    """Facts read from validated formal Q1-Q4 payloads (never from a person's say-so)."""

    payload_sha256: Mapping[str, str]
    lock_commitment_sha256: str
    determinism_scope_sha256: str
    n_ref: int
    confirmation: Mapping[str, str]          # quantity -> verdict
    budgets: Mapping[str, float]
    anomaly_bound: Mapping[str, float] | None
    source_dirs: Mapping[str, Path] | None = field(default=None, compare=False)   # host paths: never recorded
    expectations: QualificationExpectations | None = field(default=None, compare=False)

    def record(self) -> dict:
        return {"payload_sha256": dict(sorted(self.payload_sha256.items())),
                "external_review_record_sha256": None if self.expectations is None else self.expectations.review_record_sha256,
                "lock_commitment_sha256": self.lock_commitment_sha256,
                "determinism_scope_sha256": self.determinism_scope_sha256, "n_ref": self.n_ref,
                "confirmation": dict(sorted(self.confirmation.items())),
                "budgets": {k: hexf(v) for k, v in sorted(self.budgets.items())},
                "anomaly_bound": None if self.anomaly_bound is None
                else {k: hexf(v) for k, v in sorted(self.anomaly_bound.items())}}

    def anomaly_policy(self) -> AnomalyPolicy:
        if self.anomaly_bound is None:
            return AnomalyPolicy("inactive_field_incomplete", None, "q4_payload", self.budgets["field"])
        return AnomalyPolicy("active", dict(self.anomaly_bound), "q4_payload", self.budgets["field"])


def load_qualification(payload_dirs: Mapping[str, Path], *,
                       expected: QualificationExpectations | None = None) -> QualificationBinding:
    """Validate formal Q1-Q4 payloads with Drop 1's reader and extract the production inputs."""

    from operatormark_numerics_v2 import payload as P

    if set(payload_dirs) != set(QUALIFICATION_STAGES):
        raise ProductionRefused(f"production needs the {QUALIFICATION_STAGES} payloads")
    data = {}
    for stage in QUALIFICATION_STAGES:
        try:
            data[stage] = P.read_payload(Path(payload_dirs[stage]), stage)
        except (OSError, ValueError, KeyError) as exc:
            raise ProductionRefused(f"{stage} payload does not validate: {type(exc).__name__}") from None
    shas = {s: d["footer"]["payload_sha256"] for s, d in data.items()}
    locks = {d["header"]["lock_commitment_sha256"] for d in data.values()}
    scopes = {d["header"]["determinism_scope_sha256"] for d in data.values()}
    for stage, d in data.items():
        if d["header"]["mode"] != "formal":
            raise ProductionRefused(f"{stage} payload is {d['header']['mode']}-mode material; production needs formal results")
        if d["footer"]["completion"] != "complete" or d["footer"]["derived"] is None:
            raise ProductionRefused(f"{stage} payload is incomplete")
        if d["header"]["settings"].get("budget_status") != "confirmed":
            raise ProductionRefused(f"{stage} ran with unconfirmed budgets")
        priors = d["header"]["prior_payload_sha256"]
        if set(priors) != set(PRIOR_CHAIN[stage]):
            raise ProductionRefused(f"{stage} must bind exactly the priors {PRIOR_CHAIN[stage]}")
        for prior, sha in priors.items():
            if shas.get(prior) != sha:
                raise ProductionRefused(f"{stage} is not bound to the supplied {prior} payload")
    if len(locks) != 1 or len(scopes) != 1:
        raise ProductionRefused("qualification payloads come from different locks or determinism scopes")
    if type(expected) is not QualificationExpectations:
        raise ProductionRefused('externally trusted Q1-Q4 expectations required')
    try:
        expected.verify(data, production=True)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise ProductionRefused(str(exc)) from exc
    q3, q4 = data["q3"]["footer"]["derived"], data["q4"]["footer"]["derived"]
    if q3.get("n_ref") not in LY.N_REF_OPTIONS or q4.get("n_ref") != q3.get("n_ref"):
        raise ProductionRefused("Q3/Q4 do not agree on a frozen N_ref")
    budgets = {k: float.fromhex(v) for k, v in data["q4"]["header"]["settings"]["budgets"].items()}
    bound = q4.get("anomaly_bound")
    return QualificationBinding(
        payload_sha256=shas, lock_commitment_sha256=locks.pop(), determinism_scope_sha256=scopes.pop(),
        n_ref=int(q4["n_ref"]), confirmation={q: v["verdict"] for q, v in q4["confirmation"].items()}, budgets=budgets,
        anomaly_bound=None if bound is None else {k: float.fromhex(v) for k, v in bound.items()},
        source_dirs={k: Path(v) for k, v in payload_dirs.items()}, expectations=expected)


@dataclass(frozen=True)
class GenerationConfig:
    mode: str
    n_ref: int
    retention: RetentionDecision
    layout: LY.Layout
    anomaly: AnomalyPolicy
    selection: tuple[tuple[str, int], ...] | None = None      # None = every shard
    qualification: QualificationBinding | None = None
    choices: ScientificChoices | None = None
    development_reference_sha256: str | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        LY.check_n_ref(self.n_ref)
        if self.mode == 'real_development':
            _sha(self.development_reference_sha256, 'development_reference_sha256')
            if (not self.layout.real_development or self.n_ref != 128 or self.retention.option != 'E'
                    or self.qualification is not None):
                raise ValueError('Q6 real development requires its exact layout, T128/E, no production qualification claim')
        elif self.layout.real_development:
            raise ValueError('development layout requires the real_development route')
        if not isinstance(self.retention, RetentionDecision):
            raise ValueError("retention must be an explicit RetentionDecision")
        if self.selection is not None:
            allowed = set(self.layout.all_shards())
            sel = tuple((s, int(i)) for s, i in self.selection)
            if len(set(sel)) != len(sel) or not set(sel) <= allowed:
                raise ValueError("selection must name distinct shards of the layout")
            object.__setattr__(self, "selection", tuple(sorted(sel, key=lambda x: (LY.SPLITS.index(x[0]), x[1]))))

    @property
    def synthetic(self) -> bool:
        return self.mode == "synthetic_test"

    def shards(self) -> list[tuple[str, int]]:
        return list(self.selection) if self.selection is not None else self.layout.all_shards()

    def require_production_inputs(self, controls: ControlSource, directions: DirectionSource, solver,
                                   on_failure: str | None) -> None:
        """Raise ProductionRefused unless this is a fully specified, non-synthetic production run.

        Self-declared labels are not enough: the qualification binding is re-read
        from its formal payload directories, the solver must be the native solver
        class itself, no control may be public material (Drop 1 deny-list), the
        direction seed must be in the formal range, and the failure policy must
        be stated. Exact admitted control hashes and the original admission bytes
        must match externally trusted State 4 expectations, separately from the
        aggregate content identity. This is correspondence, not State acceptance.
        """

        if self.mode == 'real_development':
            from .native import NativeSolver
            if (type(controls) is not MappingControls or controls.binding().get('kind') != 'real_development'
                    or controls.binding().get('synthetic') is not False
                    or type(directions) is not D6Directions or directions.binding().get('synthetic') is not False
                    or type(solver) is not NativeSolver or on_failure != 'stop'
                    or self.retention.decision_reference_sha256 != self.development_reference_sha256):
                raise ProductionRefused('real-development sources, native solver, bound retention and stop policy required')
            return
        if self.mode not in ("production", "scientific"):
            return
        from .native import NativeSolver

        scientific = self.mode == "scientific"
        problems = []
        if scientific and (self.n_ref != 128 or self.retention.option != "S" or on_failure != "stop"):
            problems.append("scientific route requires S128 and stop without retry")
        if not self.layout.is_production:
            problems.append("layout is not the production layout")
        if self.selection is not None and list(self.selection) != self.layout.all_shards():
            problems.append("production generates every shard")
        q = self.qualification
        if q is None or q.source_dirs is None:
            problems.append("no qualification binding loaded from formal Q1-Q4 payloads")
        else:
            if type(q.expectations) is not QualificationExpectations:
                problems.append('externally trusted Q1-Q4 expectations required')
            try:
                reloaded = load_qualification(q.source_dirs, expected=q.expectations)
            except ProductionRefused as exc:
                reloaded = None
                problems.append(f"qualification payloads: {exc}")
            if reloaded is not None and reloaded != q:
                problems.append("qualification binding differs from its payloads")
            if q.n_ref != self.n_ref:
                problems.append(f"N_ref {self.n_ref} differs from the frozen N_ref {q.n_ref}")
            if self.anomaly != q.anomaly_policy():
                problems.append("anomaly procedure differs from the one Q4 derived")
        if self.anomaly.source != "q4_payload":
            problems.append("anomaly bounds must come from the Q4 payload")
        if self.retention.decision_reference_sha256 is None:
            problems.append("retention option has no decision reference")
        c = self.choices
        if c is None or not c.confirmed:
            missing = sorted(set(REQUIRED_CHOICES) - set(() if c is None else c.confirmed_items))
            problems.append(f"scientific choices not confirmed (missing: {missing or 'reference'})")
        else:
            _sha(c.reference_sha256, "choices.reference_sha256")
            if controls.binding().get("kind") != c.control_source_kind:
                problems.append("control source differs from the confirmed choice")
            if directions.binding().get("kind") != c.direction_rule:
                problems.append("direction rule differs from the confirmed choice")
        for name, src in (("controls", controls), ("directions", directions)):
            if src.binding().get("synthetic") is not False:
                problems.append(f"{name} source is synthetic or undeclared")
        if controls.binding().get("kind") != ("scientific_controls" if scientific else "accepted_controls") or "admission_sha256" not in controls.binding():
            problems.append("controls must come from accepted_controls() with an admission hash")
        if type(controls) is not MappingControls:
            problems.append('controls must use the exact admitted MappingControls implementation')
        hashes = getattr(controls, "control_hashes", None)
        if hashes is None:
            problems.append("control source cannot list its control hashes")
        else:
            try:
                expected = getattr(controls, 'admission_expectations', None)
                if type(expected) is not (ScientificControlExpectations if scientific else State4ControlExpectations):
                    raise ValueError('externally trusted State 4 admission required')
                actual = hashes()
                expected.verify(actual, production=True)
                required = {(s, i) for s in LY.SPLITS for i in range(self.layout.count(s))}
                if set(actual) != required or controls.binding().get('sha256') != control_content_sha256(actual):
                    raise ValueError('actual admitted control population/content drift')
                if controls.binding().get('admission_sha256') != expected.admission_record_sha256:
                    raise ValueError('admission record identity differs')
            except (ValueError, OSError, KeyError, TypeError) as exc:
                problems.append(str(exc))
            from operatormark_numerics_v2.lock import public_deny_list
            public = public_deny_list()["control_sha256"] & set(hashes().values())
            if public:
                problems.append(f"{len(public)} controls are public material (fixtures, designed cases, development controls)")
        if type(directions) is not D6Directions or not directions.formal_range:
            problems.append("directions must come from D6Directions with a formal-range seed (>= 2**64)")
        if type(solver) is not NativeSolver:
            problems.append(f"solver {type(solver).__name__} is not the native solver")
        if on_failure not in ("continue", "stop"):
            problems.append("production needs an explicit failure policy (on_failure 'continue' or 'stop')")
        if problems:
            raise ProductionRefused("; ".join(problems))
