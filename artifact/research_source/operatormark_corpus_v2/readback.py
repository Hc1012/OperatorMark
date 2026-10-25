"""Independent readback validation of a corpus_v2 tree, from saved bytes.

Independence rules (mirroring v1 ``final_verify``):
* This module never imports the writer side (``generator``, ``records``,
  ``layout``, ``d10``, ``config``, ``native``, ``sizing``, ``anomaly``); test B2
  checks that.  It holds its own expectations: split sizes, offsets,
  identities, file names, family order, shapes, record key sets, settings,
  gate thresholds, the D6 direction rule and the list of choices production
  must have confirmed.
* It never trusts a writer flag.  Every gate, hash, view, index, binding,
  delta, norm, compliance and anomaly flag is recomputed from the bytes on disk
  and compared with what the records claim; the writer's summary is checked as
  a claim.  The anomaly procedure (A_s, eps_F) is an input the caller supplies
  (in production: re-derived here from the formal Q1-Q4 payloads), never read
  from the corpus being judged.
* Shared code is limited to the frozen v1 sources (canonical codecs,
  ``final_scientific_verify.area_view`` for the §8.1 views, the frozen TPFA/P1
  assembly, diagnostics and P1 solve), Drop 1's payload reader (to read Q1-Q4)
  and, for retention E only, Drop 1's ``NativeTPFA`` to *re-execute* the native
  solve (Addendum A §4).  R, the tangent right-hand side dA[dkappa] u, delta,
  the direction rule and the direction checks are re-implemented here.

Outcome:
  status       matched | mismatch | invalid | incomplete (precedence invalid > mismatch > incomplete)
  d10          accepted | not_accepted | incomplete, with reasons
  production_usable  true only for a matched, complete, accepted production corpus
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from operatormark.benchmarks import darcy_spec as _spec
from operatormark.corpus import canonical as C
from operatormark.corpus.final_scientific_verify import area_view as _frozen_area_view
from .input_bindings import QualificationExpectations, State4ControlExpectations, ScientificControlExpectations
from .d7 import policy as d7_policy

_SPLITS = ("training", "validation", "test")
_DIRS = {"training": "training", "validation": "validation", "test": "test.sealed"}
_DPS = {"training": 1, "validation": 2, "test": 2}
_BENCH = "darcy_branch_c_v2"
_P = "operatormark.darcy_branch_c_v2"
_SAMPLE_FAMS = (("rho", 8), ("kappa64", 64), ("u_f64", 64), ("kappa32_areaavg", 32), ("u32_areaavg", 32),
                ("kappa16_areaavg", 16), ("u16_areaavg", 16))
_DIR_FAMS = (("dir", 8), ("du_f64", 64), ("du32_areaavg", 32), ("du16_areaavg", 16))
_EPS = 2.220446049250313e-16
_TANGENT_GATE = 1.0e-10
_R_TOL = 64.0 * _EPS
_NEAR = 0.02
_UNIT_TOL = 1.0e-14
_SEED_DOMAIN = "operatormark_corpus_v2/direction-seed-commitment/v1"
_DIRECTION_RULE = "d6_feasible_per_item_pcg64_v1"
_REQUIRED_CHOICES = ("control_source", "direction_rule", "record_schemas", "anomaly_downgrade_reading",
                     "failure_disposition")
_PRIORS = {"q1": (), "q2": ("q1",), "q3": ("q1", "q2"), "q4": ("q1", "q2", "q3")}
_TOP_FILES = {"corpus_configuration.json", "failures.jsonl", "unwritten_shard_measurements.jsonl",
              "generation_summary.json", "anomaly_ladders.jsonl", "corpus_staging", "run"}
_SAMPLE_KEYS = {"schema", "configuration_sha256", "benchmark", "split", "shard_index", "local_index", "global_index",
                "sample_id", "stratum", "control_tensor_sha256", "rho_storage_tensor_sha256",
                "macro_coefficient_tensor_sha256", "tensor_sha256", "directions", "native_evidence", "p1_256", "metrics",
                "anomaly", "gates", "record_self_sha256"}
_DIRECTION_KEYS = {"schema", "configuration_sha256", "benchmark", "split", "shard_index", "shard_position",
                   "sample_local_index", "sample_global_index", "sample_id", "direction_k", "direction_local_index",
                   "direction_global_index", "direction_id", "control_tensor_sha256", "direction_tensor_sha256",
                   "tensor_sha256", "native_evidence", "feasibility", "gates", "record_self_sha256"}
_GATES = ("structural", "tpfa_native", "p1_256", "r_integral", "tangent_residual", "directions")
_STRATUM = {"relaxed": "relaxed", "binary_32_of_64": "binary"}
_TKEYS = {"algebraic_backward_error", "boundary_transmissibility_error", "maximum_local_conservation_ratio",
          "maximum_interior_face_antisymmetry", "global_balance_ratio", "rhs_relative_l2_residual"}
_PKEYS = {"algebraic_backward_error", "boundary_reconstruction_error", "reaction_denominator", "reaction_balance_ratio",
          "outward_boundary_flux", "integrated_source", "rhs_relative_l2_residual"}
_P1_SETTINGS = {"resolution": 256, "method": "CG", "preconditioner": "Jacobi", "initial_guess": "zero",
                "rtol": float(1e-12).hex(), "atol": float(0).hex(), "maxiter": 10000, "dtype": "<f8", "forcing": "unit",
                "boundary": "zero", "diagonal": "SW_NE", "eta": "not_computed"}
_GUARD = (KeyError, TypeError, ValueError, IndexError, ArithmeticError, AttributeError, EOFError, OSError,
          C.CanonicalRecordError)


def _tpfa_settings(n: int) -> dict:
    return {"method": "SuperLU", "dtype": "<f8", "forcing": "unit", "boundary": "zero", "factorization_reuse": "tangents",
            "resolution": n}


@dataclass(frozen=True)
class Expected:
    """What the caller independently expects; nothing here is read from the corpus.

    anomaly_state/anomaly_bounds/eps_field: the D10 anomaly procedure the corpus
      must have used.  In production pass ``qualification_dirs`` instead (or as
      well): the procedure is then re-derived from the formal Q1-Q4 payloads.
    controls_sha256: aggregate over the admitted controls (rule in API.md);
      required for production acceptance.
    direction_seed: lets readback re-derive every D6 direction bitwise; required
      for production acceptance (``directions_sha256`` is an extra binding only).
    lock_commitment_sha256, retention_decision_sha256, choices_reference_sha256:
      the owner's own records; required for production acceptance.
    """

    n_ref: int
    retention: str
    counts: tuple[int, int, int]
    shard_size: int
    mode: str
    selection: tuple[tuple[str, int], ...] | None = None
    configuration_sha256: str | None = None
    anomaly_state: str | None = None
    anomaly_bounds: dict | None = None
    eps_field: float | None = None
    qualification_dirs: dict | None = None
    controls_sha256: str | None = None
    direction_seed: int | None = None
    directions_sha256: str | None = None
    lock_commitment_sha256: str | None = None
    retention_decision_sha256: str | None = None
    choices_reference_sha256: str | None = None
    qualification_expectations: QualificationExpectations | None = None
    state4_expectations: State4ControlExpectations | None = None
    scientific_expectations: ScientificControlExpectations | None = None
    producer_code_identity_sha256: str | None = None
    producer_runtime_scope_sha256: str | None = None
    development_reference_sha256: str | None = None
    d8_policy: dict | None = None

    def __post_init__(self):
        if self.d8_policy is not None:
            from .d8 import policy
            if self.mode != 'scientific':raise ValueError('D8 is scientific-only; strict production unchanged')
            policy(self.d8_policy)
        if self.mode == 'real_development' and (not _sha_ok(self.development_reference_sha256)
                or self.counts != (4096,256,0) or self.shard_size != 64 or self.n_ref != 128 or self.retention != 'E'):
            raise ValueError('exact independently bound Q6 real-development expectations required')
        if self.direction_seed is not None and (type(self.direction_seed) is not int or not 0 <= self.direction_seed < 2**128):
            raise ValueError("direction_seed must be an int in [0, 2**128)")
        for name in ("configuration_sha256", "controls_sha256", "directions_sha256", "lock_commitment_sha256",
                     "retention_decision_sha256", "choices_reference_sha256",
                     "producer_code_identity_sha256", "producer_runtime_scope_sha256"):
            value = getattr(self, name)
            if value is not None and not _sha_ok(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")

    def shards(self) -> list[tuple[str, int]]:
        every = [(s, i) for s in _SPLITS for i in range(self.counts[_SPLITS.index(s)] // self.shard_size)]
        if self.selection is None:
            return every
        return sorted(((s, int(i)) for s, i in self.selection), key=lambda x: (_SPLITS.index(x[0]), x[1]))


class _Diag:
    def __init__(self):
        self.items: set[tuple[str, str, str]] = set()

    def add(self, kind: str, subject: str, code: str) -> None:
        self.items.add((kind, subject, code))

    def status(self) -> str:
        for kind in ("invalid", "mismatch", "incomplete"):
            if any(d[0] == kind for d in self.items):
                return kind
        return "matched"


# ---------------------------------------------------------------------------- own arithmetic
def _r64(z: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(z, dtype="<f8")
    while out.shape[-1] > 64:
        out = np.ascontiguousarray((((out[0::2, 0::2] + out[0::2, 1::2]) + out[1::2, 0::2]) + out[1::2, 1::2])
                                   * np.float64(0.25))
    return out


def _r_integral_error(native: np.ndarray, n: int, f64: np.ndarray) -> float:
    original = float(np.sum(native, dtype=np.float64)) / float(n * n)
    projected = float(np.sum(f64, dtype=np.float64)) / 4096.0
    scale = float(np.sum(np.abs(native), dtype=np.float64)) / float(n * n)
    return abs(projected - original) / max(1.0, scale)


def _norm(y: np.ndarray) -> float:
    return math.sqrt(float(np.sum(y * y, dtype=np.float64) / np.float64(4096)))


def _compliance(y: np.ndarray) -> float:
    return float(np.sum(y, dtype=np.float64) / np.float64(4096))


def _tangent_rhs(kappa: np.ndarray, v8: np.ndarray, u: np.ndarray) -> np.ndarray:
    """-(dA[dkappa] u) for TPFA with T = 2 ka kb / (ka + kb) and boundary T = 2 kappa; dkappa = 9 v expanded."""

    n = kappa.shape[0]
    dk = np.kron(np.float64(9.0) * v8, np.ones((n // 8, n // 8)))
    out = np.zeros((n, n))
    for axis in (1, 0):
        a = (slice(None), slice(None, -1)) if axis == 1 else (slice(None, -1), slice(None))
        b = (slice(None), slice(1, None)) if axis == 1 else (slice(1, None), slice(None))
        ka, kb, da, db = kappa[a], kappa[b], dk[a], dk[b]
        dt = 2.0 * (ka * ka * db + kb * kb * da) / ((ka + kb) ** 2)
        flux = dt * (u[a] - u[b])
        out[a] += flux
        out[b] -= flux
    out[:, 0] += 2.0 * dk[:, 0] * u[:, 0]
    out[:, -1] += 2.0 * dk[:, -1] * u[:, -1]
    out[0, :] += 2.0 * dk[0, :] * u[0, :]
    out[-1, :] += 2.0 * dk[-1, :] * u[-1, :]
    return -out.reshape(-1)


def _d6_direction(seed: int, split: str, local: int, k: int, rho: np.ndarray) -> np.ndarray:
    """The D6 per-item rule, re-implemented: Gaussian row, sign rule at the 0.02 band, fsum normalisation."""

    sequence = np.random.SeedSequence(seed, spawn_key=(_SPLITS.index(split), local, k))
    g = np.random.Generator(np.random.PCG64(sequence)).standard_normal(64)
    r = np.asarray(rho, dtype=np.float64).reshape(64)
    v = g.copy()
    v[r <= _NEAR] = np.abs(g[r <= _NEAR])
    v[r >= 1.0 - _NEAR] = -np.abs(g[r >= 1.0 - _NEAR])
    norm = math.sqrt(math.fsum(float(x) * float(x) for x in v))
    return np.ascontiguousarray((v / np.float64(norm)).reshape(8, 8))


def _stratum_ok(rho: np.ndarray, stratum: str) -> bool:
    c = np.ascontiguousarray(rho, dtype="<f8")
    if not np.isfinite(c).all():
        return False
    if stratum == "relaxed":
        return not bool(np.signbit(c).any()) and bool(((c >= 0) & (c < 1)).all())
    bits = c.view("<u8")
    return bool(((bits == 0) | (bits == 0x3FF0000000000000)).all()) and int((bits == 0).sum()) == 32


def _direction_ok(rho: np.ndarray, v: np.ndarray) -> dict:
    r, x = rho.reshape(64), v.reshape(64)
    finite = bool(np.isfinite(x).all())
    nsq = math.fsum(float(a) * float(a) for a in x) if finite else math.inf
    sign = finite and bool(np.all(x[r <= _NEAR] >= 0.0)) and bool(np.all(x[r >= 1.0 - _NEAR] <= 0.0))
    return {"finite": finite, "sign_rule": sign, "unit_norm": finite and abs(nsq - 1.0) <= _UNIT_TOL,
            "norm_squared_fsum": nsq.hex() if finite else None}


def _hx(value) -> str | None:
    return None if value is None else float(value).hex()


def _sha_ok(value) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _aggregate(items: list[tuple[tuple[str, int], str]]) -> str:
    """Control-set binding: sha256 of canonical JSON {"keys": ["split/local", ...], "hashes": [...]} in (split, local) order."""

    items = sorted(items)
    return C.sha256_bytes(C.canonical_json_bytes({"keys": [f"{s}/{l}" for (s, l), _ in items],
                                                  "hashes": [h for _, h in items]}))


def _offsets(counts) -> tuple[dict, dict]:
    sample, direction, s_acc, d_acc = {}, {}, 0, 0
    for i, split in enumerate(_SPLITS):
        sample[split], direction[split] = s_acc, d_acc
        s_acc += counts[i]
        d_acc += counts[i] * _DPS[split]
    return sample, direction


def _sample_id(split, local):
    return f"{_BENCH}/{split}/{local:06d}"


# Closed forms for exception- and claim-derived diagnostic codes (storage bound).
# A caught exception whose type name is outside the closed form is re-raised,
# never encoded: its name and message then survive only in the propagated
# exception, so nothing is replaced by a token while it is the only copy.
_STOP_REASONS = ("time_cap", "memory_cap", "sample_failure", "hard_gate_failure",
                 "exception:?type_name_outside_closed_form")
_STOP_TOKEN = "?stop_reason_outside_closed_form"


def _identifier64(value) -> bool:
    return type(value) is str and len(value) <= 64 and value.isascii() and value.isidentifier()


def _closed_type_name(exc: BaseException) -> str:
    name = type(exc).__name__
    if _identifier64(name):
        return name
    raise exc


def _closed_stop_reason(value) -> str:
    """The raw value stays in the retained generation_summary.json bytes."""
    if value is None:
        return "None"
    if value in _STOP_REASONS or (type(value) is str and value.startswith("exception:") and _identifier64(value[10:])):
        return value
    return _STOP_TOKEN


def qualification_record(dirs: dict, expected=None) -> dict:
    """Re-derive, from formal Q1-Q4 payloads, the qualification record and anomaly procedure a production corpus must carry."""

    from operatormark_numerics_v2 import payload as P

    if set(dirs) != set(_PRIORS):
        raise ValueError("Q1-Q4 payload directories required")
    data = {stage: P.read_payload(Path(dirs[stage]), stage) for stage in _PRIORS}
    shas = {s: d["footer"]["payload_sha256"] for s, d in data.items()}
    for stage, d in data.items():
        if d["header"]["mode"] != "formal" or d["footer"]["completion"] != "complete" or d["footer"]["derived"] is None:
            raise ValueError(f"{stage} is not a complete formal payload")
        if d["header"]["settings"].get("budget_status") != "confirmed":
            raise ValueError(f"{stage} budgets unconfirmed")
        priors = d["header"]["prior_payload_sha256"]
        if set(priors) != set(_PRIORS[stage]) or any(shas[p] != v for p, v in priors.items()):
            raise ValueError(f"{stage} prior chain")
    if len({d["header"]["lock_commitment_sha256"] for d in data.values()}) != 1 or \
            len({d["header"]["determinism_scope_sha256"] for d in data.values()}) != 1:
        raise ValueError("lock or scope differs")
    if type(expected) is not QualificationExpectations:
        raise ValueError('externally trusted qualification expectations required')
    expected.verify(data, production=True)
    q3, q4 = data["q3"]["footer"]["derived"], data["q4"]["footer"]["derived"]
    if q3.get("n_ref") not in (128, 256) or q4.get("n_ref") != q3.get("n_ref"):
        raise ValueError("N_ref not frozen")
    budgets = {k: float.fromhex(v) for k, v in data["q4"]["header"]["settings"]["budgets"].items()}
    bound = q4.get("anomaly_bound")
    record = {"payload_sha256": dict(sorted(shas.items())),
              "external_review_record_sha256": expected.review_record_sha256,
              "lock_commitment_sha256": data["q4"]["header"]["lock_commitment_sha256"],
              "determinism_scope_sha256": data["q4"]["header"]["determinism_scope_sha256"], "n_ref": q4["n_ref"],
              "confirmation": dict(sorted((q, v["verdict"]) for q, v in q4["confirmation"].items())),
              "budgets": {k: v.hex() for k, v in sorted(budgets.items())},
              "anomaly_bound": None if bound is None else {k: float.fromhex(v).hex() for k, v in sorted(bound.items())}}
    anomaly = {"state": "inactive_field_incomplete" if bound is None else "active", "source": "q4_payload",
               "max_anomaly_ladders": 64, "eps_field": budgets["field"].hex(), "bounds": record["anomaly_bound"]}
    return {"qualification": record, "anomaly_procedure": anomaly}


class _Validator:
    def __init__(self, root: Path, expected: Expected, numerics: str, p1: str):
        if numerics not in ("full", "structural") or p1 not in ("resolve", "skip"):
            raise ValueError("numerics must be full|structural and p1 resolve|skip")
        if expected.n_ref not in (128, 256) or expected.retention not in ("E", "S"):
            raise ValueError("expected N_ref and retention must be explicit and valid")
        self.root, self.x, self.numerics, self.p1 = root, expected, numerics, p1
        self.diag = _Diag()
        self.s_off, self.d_off = _offsets(expected.counts)
        self.n = expected.n_ref
        self.stored = expected.retention == "S"
        self.cfg_sha = None
        self.ladder_checks = 0
        self.facts = {"samples": 0, "directions": 0, "anomalies": {"relaxed": [], "binary": []},
                      "gate_failures": {g: [] for g in _GATES}, "delta_max": {}, "control_items": [],
                      "direction_items": [], "shards": {}, "failure_samples": [], "numerics_checked": 0,
                      "p1_checked": 0, "unwritten": [], "unwritten_anomalies": {"relaxed": [], "binary": []},
                      "stratum_of": {}, "unwritten_by_shard": {}, "failed_locals": {}, "stopped_shards": set(),
                      "unwritten_strata": {"relaxed": [], "binary": []}}

    # ------------------------------------------------------------------ helpers
    def read(self, rel: str) -> bytes | None:
        path = self.root / rel
        try:
            return path.read_bytes() if path.is_file() else None
        except OSError:
            return None

    def shard_files(self, split: str) -> list[tuple[str, str, tuple]]:
        b, d = self.x.shard_size, _DPS[split]
        out = [(f, "sample", (b, 1, s, s)) for f, s in _SAMPLE_FAMS] + [(f, "direction", (b * d, 1, s, s)) for f, s in _DIR_FAMS]
        if self.stored:
            out += [(f"u_tpfa{self.n}", "native_sample", (b, 1, self.n, self.n)),
                    (f"du_tpfa{self.n}", "native_direction", (b * d, 1, self.n, self.n))]
        return out

    def identity_ok(self, row: dict) -> bool:
        split, local = row.get("split"), row.get("local_index")
        if split not in _SPLITS or type(local) is not int or not 0 <= local < self.x.counts[_SPLITS.index(split)]:
            return False
        return (row.get("sample_id") == _sample_id(split, local) and row.get("shard_index") == local // self.x.shard_size
                and row.get("global_index") == self.s_off[split] + local
                and row.get("stratum") == ("relaxed" if local % 2 == 0 else "binary_32_of_64"))

    # ------------------------------------------------------------------ configuration and expected procedure
    def expected_anomaly(self) -> dict | None:
        x = self.x
        derived = None
        if x.qualification_dirs is not None:
            try:
                derived = qualification_record(x.qualification_dirs, x.qualification_expectations)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.diag.add("mismatch", "qualification", "qualification_payloads_invalid:" + _closed_type_name(exc))
        self.expected_qualification = None if derived is None else derived["qualification"]
        if x.anomaly_state is not None:
            if x.anomaly_state not in ("active", "inactive_field_incomplete") or x.eps_field is None or \
                    (x.anomaly_state == "active") != (x.anomaly_bounds is not None):
                raise ValueError("expected anomaly procedure is malformed")
            own = {"state": x.anomaly_state, "eps_field": float(x.eps_field).hex(),
                   "bounds": None if x.anomaly_bounds is None else {k: float(v).hex() for k, v in sorted(x.anomaly_bounds.items())}}
            if derived is not None and {k: derived["anomaly_procedure"][k] for k in own} != own:
                self.diag.add("mismatch", "qualification", "expected_anomaly_differs_from_q4")
            return own
        if derived is not None:
            return {k: derived["anomaly_procedure"][k] for k in ("state", "eps_field", "bounds")}
        self.diag.add("incomplete", "anomaly_procedure", "expected_anomaly_procedure_not_supplied")
        return None

    def configuration(self) -> dict | None:
        data = self.read("corpus_configuration.json")
        if data is None:
            self.diag.add("incomplete", "configuration", "configuration_missing")
            return None
        try:
            cfg = C.strict_canonical_json_loads(data)
            C.validate_self_hash(cfg, "record_self_sha256")
        except _GUARD:
            self.diag.add("invalid", "configuration", "configuration_bytes")
            return None
        self.cfg_sha = C.sha256_bytes(data)
        x = self.x
        for expected_hash, observed_hash, label in (
            (x.producer_code_identity_sha256, C.sha256_bytes(C.canonical_json_bytes(cfg.get('code_identity'))), 'producer_source'),
            (x.producer_runtime_scope_sha256, cfg.get('determinism_scope_sha256'), 'producer_runtime')):
            if expected_hash is None and x.mode in ('production','scientific'):
                self.diag.add('incomplete', 'configuration', label + '_not_independently_bound')
            elif expected_hash is not None and expected_hash != observed_hash:
                self.diag.add('mismatch', 'configuration', label + '_binding')
        if x.configuration_sha256 is not None and x.configuration_sha256 != self.cfg_sha:
            self.diag.add("mismatch", "configuration", "configuration_hash_binding")
        try:
            checks = {
                "schema": cfg["schema"] == f"{_P}.corpus_configuration.v1" and cfg["benchmark"] == _BENCH,
                "mode": cfg["mode"] == x.mode and cfg["synthetic"] is (x.mode == "synthetic_test")
                and cfg["production_authorized"] is (x.mode == "production"),
                "n_ref": cfg["n_ref"] == x.n_ref,
                "retention": cfg["retention"]["option"] == x.retention
                and cfg["retention"]["classes"] == {f"u_tpfa{x.n_ref}": "stored" if self.stored else "hash_only",
                                                    f"du_tpfa{x.n_ref}": "stored" if self.stored else "hash_only",
                                                    f"kappa{x.n_ref}": "hash_only", "p1_256_nodal": "hash_only",
                                                    "p1_256_projected64": "hash_only"},
                "layout": cfg["layout"]["counts"] == dict(zip(_SPLITS, x.counts))
                and cfg["layout"]["shard_size"] == x.shard_size and cfg["layout"]["directions_per_sample"] == _DPS,
                "selection": [tuple(s) for s in cfg["selection"]] == x.shards(),
            }
            anomaly = cfg["anomaly_procedure"]
            sources = cfg["sources"]
        except _GUARD:
            self.diag.add("invalid", "configuration", "configuration_fields")
            return None
        for name, ok in checks.items():
            if not ok:
                self.diag.add("mismatch", "configuration", f"configuration_{name}")
        own = self.expected_anomaly()
        self.anomaly_state, self.bounds, self.eps_field = None, None, None
        if own is not None:
            if {k: anomaly.get(k) for k in own} != own or anomaly.get("max_anomaly_ladders") != 64:
                self.diag.add("mismatch", "configuration", "anomaly_procedure_differs_from_expected")
            self.anomaly_state = own["state"]
            self.bounds = None if own["bounds"] is None else {k: float.fromhex(v) for k, v in own["bounds"].items()}
            self.eps_field = float.fromhex(own["eps_field"])
        if x.mode in ("production", "scientific"):
            self.production_configuration(cfg, sources)
        elif x.mode == 'real_development':
            expected = {'request_sha256': x.development_reference_sha256, 'physical_field_confirmed': False,
                        'ladder_trigger': 'confirmed_only', 'ladder_cap_per_stratum': 64, 'state4_admission': 'not_claimed'}
            if (cfg.get('development') != expected or cfg.get('qualification') is not None
                    or not cfg['layout'].get('real_development') or cfg['layout'].get('synthetic') is not False
                    or sources.get('controls',{}).get('kind') != 'real_development'
                    or sources.get('controls',{}).get('synthetic') is not False
                    or sources.get('solver') != 'native' or cfg['failure_policy']['on_failure'] != 'stop'
                    or x.direction_seed is None or x.controls_sha256 is None):
                self.diag.add('mismatch', 'configuration', 'real_development_binding')
        self.sources = sources
        return cfg

    def production_configuration(self, cfg: dict, sources: dict) -> None:
        x, problems = self.x, []
        scientific = x.mode == "scientific"
        control_expectations = x.scientific_expectations if scientific else x.state4_expectations
        if scientific and (x.n_ref != 128 or x.retention != "S"):
            problems.append("scientific_S128")
        try:
            if not (tuple(x.counts) == (4096, 256, 512) and x.shard_size == 64 and x.selection is None):
                problems.append("layout")
            controls, directions = sources["controls"], sources["directions"]
            if controls.get("synthetic") is not False or controls.get("kind") != ("scientific_controls" if scientific else "accepted_controls") \
                    or not _sha_ok(controls.get("admission_sha256")):
                problems.append("controls_source")
            if directions.get("synthetic") is not False or directions.get("kind") != _DIRECTION_RULE \
                    or not _sha_ok(directions.get("seed_commitment_sha256")):
                problems.append("directions_source")
            if sources["solver"] != "native":
                problems.append("solver")
            if cfg["anomaly_procedure"]["source"] != "q4_payload":
                problems.append("anomaly_source")
            if not _sha_ok(cfg["retention"]["decision_reference_sha256"]):
                problems.append("retention_decision")
            choices = cfg["scientific_choices"] or {}
            if not (set(_REQUIRED_CHOICES) <= set(choices.get("confirmed_items") or ()) and choices.get("confirmed") is True
                    and _sha_ok(choices.get("reference_sha256")) and choices.get("direction_rule") == directions.get("kind")
                    and choices.get("control_source_kind") == controls.get("kind")):
                problems.append("scientific_choices")
            if cfg["failure_policy"]["on_failure"] not in ("continue", "stop"):
                problems.append("failure_policy")
            if x.qualification_dirs is None:
                self.diag.add("incomplete", "qualification", "qualification_payloads_not_supplied")
            elif self.expected_qualification is not None and cfg["qualification"] != self.expected_qualification:
                problems.append("qualification_record")
            for field, claimed, code in (
                    ("lock_commitment_sha256", (cfg["qualification"] or {}).get("lock_commitment_sha256"), "lock_commitment"),
                    ("retention_decision_sha256", cfg["retention"]["decision_reference_sha256"], "retention_decision"),
                    ("choices_reference_sha256", choices.get("reference_sha256"), "choices_reference")):
                expected_value = getattr(x, field)
                if expected_value is None:
                    self.diag.add("incomplete", "configuration", f"{code}_not_independently_bound")
                elif claimed != expected_value:
                    problems.append(code)
            if cfg["qualification"] is None or cfg["qualification"].get("n_ref") != x.n_ref:
                problems.append("qualification_n_ref")
        except _GUARD:
            problems.append("fields")
        for p in problems:
            self.diag.add("mismatch", "configuration", f"production_{p}")
        if x.configuration_sha256 is None:
            self.diag.add("incomplete", "configuration", "configuration_hash_not_independently_bound")
        if x.controls_sha256 is None:
            self.diag.add("incomplete", "controls", "controls_not_independently_bound")
        if type(control_expectations) is not (ScientificControlExpectations if scientific else State4ControlExpectations):
            self.diag.add('incomplete', 'controls', 'State4_admission_not_independently_bound')
        else:
            if (sources.get('controls', {}).get('admission_sha256') != control_expectations.admission_record_sha256
                    or x.controls_sha256 != control_expectations.aggregate_control_content_sha256):
                self.diag.add('mismatch', 'controls', 'State4_record_or_content_binding')
        if x.direction_seed is None:
            self.diag.add("incomplete", "directions", "directions_not_rederived_from_seed")

    # ------------------------------------------------------------------ inventory
    def inventory(self) -> None:
        expected_dirs = {f"corpus_staging/{_DIRS[s]}/shard_{i:04d}" for s, i in self.x.shards()}
        for entry in sorted(self.root.iterdir()):
            if entry.name not in _TOP_FILES:
                self.diag.add("invalid", entry.name, "unexpected_entry")
        staging = self.root / "corpus_staging"
        if not staging.is_dir():
            return
        for split_dir in sorted(staging.iterdir()):
            if not split_dir.is_dir() or split_dir.name not in _DIRS.values():
                self.diag.add("invalid", f"corpus_staging/{split_dir.name}", "unexpected_entry")
                continue
            for shard_dir in sorted(split_dir.iterdir()):
                rel = f"corpus_staging/{split_dir.name}/{shard_dir.name}"
                if shard_dir.name.endswith(".partial"):
                    self.diag.add("incomplete", rel, "partial_shard_not_published")
                elif rel not in expected_dirs or not shard_dir.is_dir():
                    self.diag.add("invalid", rel, "unexpected_entry")

    # ------------------------------------------------------------------ failures and unwritten measurements
    def jsonl(self, name: str) -> list[dict] | None:
        data = self.read(name)
        if data is None:
            self.diag.add("incomplete", name, "file_missing")
            return None
        if data == b"":
            return []
        try:
            rows = C.strict_canonical_jsonl_loads(data)
            for row in rows:
                C.validate_self_hash(row, "record_self_sha256")
            return rows
        except _GUARD:
            self.diag.add("invalid", name, "jsonl_bytes")
            return None

    def failures(self) -> None:
        data = self.read("failures.jsonl")
        self.failures_sha = None if data is None else C.sha256_bytes(data)
        rows = self.jsonl("failures.jsonl")
        selected = set(self.x.shards())
        seen = set()
        for row in rows or []:
            try:
                ok = (row["schema"] == f"{_P}.generation_failure.v1" and row["configuration_sha256"] == self.cfg_sha
                      and row["benchmark"] == _BENCH and self.identity_ok(row)
                      and row["disposition"] == "retained_not_replaced"
                      and type(row["stage"]) is str and row["stage"] and type(row["code"]) is str and row["code"]
                      and type(row["partial"]) is dict and row["sample_id"] not in seen)
            except _GUARD:
                ok = False
            if not ok:
                self.diag.add("invalid", "failures.jsonl", "failure_record")
                continue
            seen.add(row["sample_id"])
            key = (row["split"], row["shard_index"])
            if key not in selected:
                self.diag.add("mismatch", "failures.jsonl", "failure_outside_selection")
            self.facts["failure_samples"].append(row["sample_id"])
            self.facts["failed_locals"].setdefault(key, []).append(row["local_index"])
            self.facts["shards"].setdefault(key, {})["failure"] = True

    def measurements(self) -> None:
        data = self.read("unwritten_shard_measurements.jsonl")
        self.measurements_sha = None if data is None else C.sha256_bytes(data)
        rows = self.jsonl("unwritten_shard_measurements.jsonl")
        for row in rows or []:
            try:
                rec = row["sample_record"]
                C.validate_self_hash(rec, "record_self_sha256")
                rho = np.array([float.fromhex(h) for h in row["control_hex"]], dtype="<f8").reshape(8, 8)
                dirs = [np.array([float.fromhex(h) for h in v], dtype="<f8").reshape(8, 8) for v in row["directions_hex"]]
                for drec in row["direction_records"]:
                    C.validate_self_hash(drec, "record_self_sha256")
                key = (rec["split"], rec["shard_index"])
                ok = (row["schema"] == f"{_P}.unwritten_shard_measurement.v1" and row["configuration_sha256"] == self.cfg_sha
                      and row["reason"] in ("shard_failed", "generation_stopped")
                      and rec["configuration_sha256"] == self.cfg_sha and row["sample_id"] == rec["sample_id"]
                      and self.identity_ok(rec) and row["disposition"] == "measured_shard_not_written"
                      and rec["control_tensor_sha256"] == C.tensor_sha256(rho)
                      and rec["directions"]["direction_tensor_sha256"] == [C.tensor_sha256(v) for v in dirs]
                      and rec["directions"]["record_self_sha256"] == [d["record_self_sha256"] for d in row["direction_records"]]
                      and len(dirs) == _DPS[rec["split"]])
            except _GUARD:
                ok = False
            if not ok:
                self.diag.add("invalid", "unwritten_shard_measurements.jsonl", "measurement_record")
                continue
            failed_shard = bool(self.facts["shards"].get(key, {}).get("failure"))
            if (row["reason"] == "shard_failed") != failed_shard or key not in set(self.x.shards()):
                self.diag.add("mismatch", rec["sample_id"], "measurement_reason_or_shard")
            if rec["sample_id"] in self.facts["unwritten"] or rec["sample_id"] in self.facts["failure_samples"]:
                self.diag.add("invalid", rec["sample_id"], "measurement_duplicate_or_also_failed")
                continue
            self.facts["unwritten"].append(rec["sample_id"])
            self.facts["unwritten_by_shard"].setdefault(key, []).append(rec["local_index"])
            if row["reason"] == "generation_stopped":
                self.facts["stopped_shards"].add(key)
            # flags of unwritten samples are claims: counted for information, never trusted (see ladders())
            if rec["anomaly"].get("flagged") is True:
                self.facts["unwritten_anomalies"][_STRATUM[rec["stratum"]]].append(rec["sample_id"])
            self.facts["unwritten_strata"][_STRATUM[rec["stratum"]]].append(rec["sample_id"])

    # ------------------------------------------------------------------ shards
    def shard(self, split: str, shard: int) -> None:
        rel = f"corpus_staging/{_DIRS[split]}/shard_{shard:04d}"
        info = self.facts["shards"].setdefault((split, shard), {})
        if not (self.root / rel).is_dir():
            if info.get("failure"):
                self.diag.add("incomplete", rel, "shard_not_written_failure_retained")
                info["status"] = "failed"
            else:
                self.diag.add("incomplete", rel, "shard_missing")
                info["status"] = "missing"
            return
        info["status"] = "invalid"
        if info.get("failure"):
            self.diag.add("mismatch", rel, "failure_recorded_for_written_shard")
        files = self.shard_files(split)
        names = {f"{f}.float64.npy" for f, _, _ in files} | {"verification.jsonl", "directions.jsonl", "shard_manifest.json"}
        present = {p.name for p in (self.root / rel).iterdir()}
        for extra in sorted(present - names):
            self.diag.add("invalid", f"{rel}/{extra}", "unexpected_file")
        for name in sorted(names):
            if name not in present or not (self.root / rel / name).is_file():
                self.diag.add("invalid", f"{rel}/{name}", "member_missing")
        if any(name not in present or not (self.root / rel / name).is_file() for name in names):
            return
        raw, arrays = {}, {}
        for family, _, shape in files:
            data = self.read(f"{rel}/{family}.float64.npy")
            raw[family] = data
            try:
                arrays[family] = C.validate_npy_v2_bytes(data, expected_shape=shape)
            except _GUARD:
                self.diag.add("invalid", f"{rel}/{family}", "npy_bytes_or_shape")
        rows = {}
        for name, schema, count in (("verification.jsonl", "sample_verification", self.x.shard_size),
                                    ("directions.jsonl", "direction_verification", self.x.shard_size * _DPS[split])):
            data = self.read(f"{rel}/{name}")
            raw[name] = data
            try:
                parsed = C.strict_canonical_jsonl_loads(data)
                for r in parsed:
                    C.validate_self_hash(r, "record_self_sha256")
                if len(parsed) != count or any(r.get("schema") != f"{_P}.{schema}.v1" for r in parsed):
                    raise C.CanonicalRecordError("count or schema")
                rows[name] = parsed
            except _GUARD:
                self.diag.add("invalid", f"{rel}/{name}", "jsonl_records")
        manifest_data = self.read(f"{rel}/shard_manifest.json")
        try:
            self.manifest(rel, split, shard, manifest_data, raw, arrays, rows)
        except _GUARD:
            self.diag.add("invalid", f"{rel}/shard_manifest.json", "manifest_structure")
        info["status"] = "written"
        info["manifest_sha256"] = C.sha256_bytes(manifest_data)
        if len(arrays) != len(files) or len(rows) != 2:
            return
        d = _DPS[split]
        for offset in range(self.x.shard_size):
            local = shard * self.x.shard_size + offset
            try:
                self.sample(split, shard, local, offset, arrays, rows["verification.jsonl"][offset],
                            rows["directions.jsonl"][offset * d:(offset + 1) * d])
            except _GUARD:
                self.diag.add("invalid", _sample_id(split, local), "sample_check_failed")

    def manifest(self, rel, split, shard, data, raw, arrays, rows) -> None:
        try:
            m = C.strict_canonical_json_loads(data)
            C.validate_self_hash(m, "manifest_self_sha256")
        except _GUARD:
            self.diag.add("invalid", f"{rel}/shard_manifest.json", "manifest_bytes")
            return
        b, d = self.x.shard_size, _DPS[split]
        first = shard * b
        ids = [_sample_id(split, l) for l in range(first, first + b)]
        dids = [f"{_sample_id(split, l)}/d{k}" for l in range(first, first + b) for k in range(d)]
        expect = {"schema": f"{_P}.shard_manifest.v1", "benchmark": _BENCH, "configuration_sha256": self.cfg_sha,
                  "split": split, "shard_index": shard, "sample_count": b, "direction_count": b * d,
                  "first_local_index": first, "last_local_index": first + b - 1,
                  "first_global_index": self.s_off[split] + first, "last_global_index": self.s_off[split] + first + b - 1,
                  "first_direction_global_index": self.d_off[split] + first * d,
                  "last_direction_global_index": self.d_off[split] + (first + b) * d - 1,
                  "ordered_sample_ids": ids, "ordered_direction_ids": dids}
        for key, value in expect.items():
            if m.get(key) != value:
                self.diag.add("mismatch", f"{rel}/shard_manifest.json", f"manifest_{key}")
        members = m.get("members")
        files = self.shard_files(split)
        if not isinstance(members, list) or len(members) != len(files) + 2 or not all(type(x) is dict for x in members):
            self.diag.add("invalid", f"{rel}/shard_manifest.json", "manifest_members")
            return
        if set(m) != set(expect) | {"members", "manifest_self_sha256"}:
            self.diag.add("invalid", f"{rel}/shard_manifest.json", "manifest_fields")
        for member, (family, _, shape) in zip(members, files):
            ok = (member.get("relative_path") == f"{rel}/{family}.float64.npy" and member.get("tensor_name") == family
                  and member.get("dtype") == "<f8" and member.get("shape") == list(shape) and member.get("order") == "C"
                  and member.get("npy_version") == [2, 0] and raw.get(family) is not None
                  and member.get("size") == len(raw[family]) and member.get("sha256") == C.sha256_bytes(raw[family])
                  and family in arrays and member.get("tensor_sha256") == C.tensor_sha256(arrays[family]))
            if not ok:
                self.diag.add("mismatch", f"{rel}/{family}", "manifest_member_binding")
        for member, name, schema in zip(members[-2:], ("verification.jsonl", "directions.jsonl"),
                                        ("sample_verification", "direction_verification")):
            data = raw.get(name)
            ok = (data is not None and member.get("relative_path") == f"{rel}/{name}"
                  and member.get("schema") == f"{_P}.{schema}.v1" and member.get("size") == len(data)
                  and member.get("sha256") == C.sha256_bytes(data) and name in rows
                  and member.get("record_count") == len(rows[name])
                  and member.get("ordered_record_self_sha256") == [r["record_self_sha256"] for r in rows[name]])
            if not ok:
                self.diag.add("mismatch", f"{rel}/{name}", "manifest_member_binding")

    # ------------------------------------------------------------------ one sample
    def sample(self, split, shard, local, offset, arrays, rec, drecs) -> None:
        sid = _sample_id(split, local)
        bad = lambda code, kind="mismatch": self.diag.add(kind, sid, code)
        if set(rec) != _SAMPLE_KEYS:
            bad("sample_record_fields", "invalid")
            return
        stratum = "relaxed" if local % 2 == 0 else "binary_32_of_64"
        self.facts["stratum_of"][sid] = _STRATUM[stratum]
        ident = {"configuration_sha256": self.cfg_sha, "benchmark": _BENCH, "split": split, "shard_index": shard,
                 "local_index": local, "global_index": self.s_off[split] + local, "sample_id": sid, "stratum": stratum}
        for key, value in ident.items():
            if rec[key] != value:
                bad(f"identity_{key}")
        rho = np.ascontiguousarray(arrays["rho"][offset, 0])
        control_sha = C.tensor_sha256(rho)
        macro = np.empty((8, 8), dtype="<f8")
        for j in range(8):
            for i in range(8):
                macro[j, i] = np.float64(np.float64(9.0) * rho[j, i]) + np.float64(3.0)
        if rec["control_tensor_sha256"] != control_sha:
            bad("control_hash_binding")
        if rec["rho_storage_tensor_sha256"] != C.tensor_sha256(arrays["rho"][offset]):
            bad("rho_storage_binding")
        if rec["macro_coefficient_tensor_sha256"] != C.tensor_sha256(macro):
            bad("macro_coefficient_binding")
        self.facts["control_items"].append(((split, local), control_sha))
        structural = _stratum_ok(rho, stratum)
        kappa64 = np.repeat(np.repeat(macro, 8, axis=0), 8, axis=1)
        if arrays["kappa64"][offset, 0].tobytes() != kappa64.tobytes():
            bad("kappa64_expansion")
            structural = False
        for name, source, n in (("kappa32_areaavg", "kappa64", 32), ("u32_areaavg", "u_f64", 32),
                                ("kappa16_areaavg", "kappa64", 16), ("u16_areaavg", "u_f64", 16)):
            if arrays[name][offset].tobytes() != _frozen_area_view(np, arrays[source][offset], n).tobytes():
                bad(f"derived_view_{name}")
                structural = False
        fams = [f for f, _ in _SAMPLE_FAMS] + ([f"u_tpfa{self.n}"] if self.stored else [])
        if rec["tensor_sha256"] != {f: C.tensor_sha256(arrays[f][offset]) for f in fams}:
            bad("tensor_hash_binding")
        f64 = arrays["u_f64"][offset, 0]
        metrics = rec["metrics"]
        if metrics.get("norm_f64") != _hx(_norm(f64)) or metrics.get("compliance_f64") != _hx(_compliance(f64)):
            bad("metric_claim_f64")
        # directions -------------------------------------------------------
        d = _DPS[split]
        dir_ids = [f"{sid}/d{k}" for k in range(d)]
        if len(drecs) != d:
            bad("direction_count")
            return
        if rec["directions"] != {"count": d, "ids": dir_ids,
                                 "direction_tensor_sha256": [r.get("direction_tensor_sha256") for r in drecs],
                                 "record_self_sha256": [r.get("record_self_sha256") for r in drecs]}:
            bad("direction_binding")
        vs = [np.ascontiguousarray(arrays["dir"][offset * d + k, 0]) for k in range(d)]
        vshas = [C.tensor_sha256(v) for v in vs]
        distinct = len(set(vshas)) == len(vshas)      # a duplicate is a measured gate failure, not corruption
        dir_gates = []
        for k, drec in enumerate(drecs):
            pos = offset * d + k
            v, vsha = vs[k], vshas[k]
            if set(drec) != _DIRECTION_KEYS:
                bad("direction_record_fields", "invalid")
                return
            dident = {"configuration_sha256": self.cfg_sha, "benchmark": _BENCH, "split": split, "shard_index": shard,
                      "shard_position": pos, "sample_local_index": local, "sample_global_index": self.s_off[split] + local,
                      "sample_id": sid, "direction_k": k, "direction_local_index": local * d + k,
                      "direction_global_index": self.d_off[split] + local * d + k, "direction_id": dir_ids[k],
                      "control_tensor_sha256": control_sha, "direction_tensor_sha256": vsha}
            for key, value in dident.items():
                if drec[key] != value:
                    bad(f"direction_{key}")
            dfams = [f for f, _ in _DIR_FAMS] + ([f"du_tpfa{self.n}"] if self.stored else [])
            if drec["tensor_sha256"] != {f: C.tensor_sha256(arrays[f][pos]) for f in dfams}:
                bad("direction_tensor_hash_binding")
            for name, n in (("du32_areaavg", 32), ("du16_areaavg", 16)):
                if arrays[name][pos].tobytes() != _frozen_area_view(np, arrays["du_f64"][pos], n).tobytes():
                    bad(f"derived_view_{name}")
                    structural = False
            facts = _direction_ok(rho, v)
            if drec["feasibility"] != facts:
                bad("direction_feasibility_claim")
            if self.x.direction_seed is not None and \
                    _d6_direction(self.x.direction_seed, split, local, k, rho).tobytes() != v.tobytes():
                bad("direction_not_from_seed")
            self.facts["direction_items"].append(((split, local * d + k), vsha))
            dir_gates.append({"feasible": facts["finite"] and facts["sign_rule"] and facts["unit_norm"] and distinct})
        self.facts["samples"] += 1
        self.facts["directions"] += len(drecs)
        gates = {"structural": structural, "directions": all(g["feasible"] for g in dir_gates)}
        # native -----------------------------------------------------------
        native = rec["native_evidence"]
        if native.get("n_ref") != self.n or native.get("retention") != self.x.retention or \
                native.get("settings") != _tpfa_settings(self.n):
            bad("native_settings_binding")
        u, resolved, kappa_n = None, None, None
        if self.stored:
            u = np.ascontiguousarray(arrays[f"u_tpfa{self.n}"][offset, 0])
            if native["solution_tensor_sha256"] != C.tensor_sha256(u):
                bad("native_solution_hash_binding")
        if self.numerics == "full":
            kappa_n = _spec.coefficient_on_tpfa_grid(rho, self.n)
            if native["coefficient_tensor_sha256"] != C.tensor_sha256(kappa_n):
                bad("native_coefficient_hash")
            if not self.stored:
                from operatormark_numerics_v2.tpfa_family import NativeTPFA    # re-execution under E
                resolved = NativeTPFA(kappa_n)
                u = resolved.u
                if native["solution_tensor_sha256"] != C.tensor_sha256(u):
                    bad("native_resolve_hash")
        r_ok = True
        if u is not None:
            if _r64(u).tobytes() != f64.tobytes():
                bad("restriction_u_f64")
            err = _r_integral_error(u, self.n, f64)
            r_ok = math.isfinite(err) and err <= _R_TOL
            if native["r_integral_error"] != _hx(err):
                bad("r_integral_error_claim")
        du_list = []
        for k, drec in enumerate(drecs):
            pos = offset * d + k
            du = None
            if self.stored:
                du = np.ascontiguousarray(arrays[f"du_tpfa{self.n}"][pos, 0])
                if drec["native_evidence"]["solution_tensor_sha256"] != C.tensor_sha256(du):
                    bad("native_tangent_hash_binding")
            elif resolved is not None:
                du, _, _ = resolved.tangent_native(vs[k])
                if drec["native_evidence"]["solution_tensor_sha256"] != C.tensor_sha256(du):
                    bad("native_tangent_resolve_hash")
            if du is not None:
                if _r64(du).tobytes() != arrays["du_f64"][pos, 0].tobytes():
                    bad("restriction_du_f64")
                err = _r_integral_error(du, self.n, arrays["du_f64"][pos, 0])
                dir_gates[k]["r_integral"] = math.isfinite(err) and err <= _R_TOL
                if drec["native_evidence"]["r_integral_error"] != _hx(err):
                    bad("tangent_r_integral_error_claim")
            du_list.append(du)
        if u is not None and all(x is not None for x in du_list):
            gates["r_integral"] = r_ok and all(g["r_integral"] for g in dir_gates)
        if self.numerics == "full":
            from operatormark.solvers import darcy_tpfa as T
            system = T.assemble_zero_dirichlet_benchmark(kappa_n)
            diag = T.compute_tpfa_diagnostics(system, u)
            claimed = native["diagnostics"]
            if set(claimed) != _TKEYS or any(claimed[key] != _hx(getattr(diag, key)) for key in _TKEYS):
                bad("tpfa_diagnostics_claim")
            gates["tpfa_native"] = bool(diag.passes)
            if native.get("passes") is not gates["tpfa_native"]:
                bad("tpfa_passes_claim")
            for k, du in enumerate(du_list):
                rhs = _tangent_rhs(kappa_n, vs[k], u)
                backward = float(T.algebraic_backward_error(system.matrix, du.reshape(-1), rhs))
                dir_gates[k]["tangent_residual"] = math.isfinite(backward) and backward <= _TANGENT_GATE
                if drecs[k]["native_evidence"]["backward_error"] != _hx(backward):
                    bad("tangent_backward_error_claim")
            gates["tangent_residual"] = all(g["tangent_residual"] for g in dir_gates)
            self.facts["numerics_checked"] += 1
        # P1-256 -------------------------------------------------------------
        p1rec = rec["p1_256"]
        if p1rec.get("settings") != _P1_SETTINGS:
            bad("p1_settings_binding")
        if self.p1 == "resolve":
            self.p1_check(sid, stratum, rho, f64, rec, p1rec, gates, bad)
        # gates --------------------------------------------------------------
        for name, value in gates.items():
            if rec["gates"].get(name) is not value:
                bad(f"gate_claim_{name}")
            if not value:
                self.facts["gate_failures"][name].append(sid)
        for k, drec in enumerate(drecs):
            for name, value in dir_gates[k].items():
                if drec["gates"].get(name) is not value:
                    bad(f"direction_gate_claim_{name}")
            if len(dir_gates[k]) == 3 and drec["gates"].get("overall") is not all(dir_gates[k].values()):
                bad("direction_gate_claim_overall")
        if len(gates) == len(_GATES) and rec["gates"].get("overall") is not all(gates.values()):
            bad("gate_claim_overall")

    def p1_check(self, sid, stratum, rho, f64, rec, p1rec, gates, bad) -> None:
        from operatormark.solvers import darcy_p1 as P1
        system = P1.assemble_p1_zero_dirichlet_benchmark(rho, 256)
        key = _STRATUM[stratum]
        try:
            result = P1.solve_p1_with_info(system)
            nodal = result.solution
            projected = P1.project_p1_to_64(nodal)
        except (P1.P1Error, ValueError, ArithmeticError):
            result = None
        self.facts["p1_checked"] += 1
        if result is None:                      # the verifier fails here too: a failed gate, no delta
            gates["p1_256"] = False
            if not str(p1rec.get("outcome", "")).startswith("failed:") or p1rec.get("nodal_tensor_sha256") is not None \
                    or p1rec.get("passes") is not False:
                bad("p1_outcome_claim")
            m = rec["metrics"]
            if m.get("delta_field") is not None or m.get("norm_projected_p256") is not None \
                    or m.get("compliance_projected_p256") is not None:
                bad("delta_claim")
            if rec["anomaly"] != {"procedure": self.anomaly_state,
                                  "bound": None if self.bounds is None else self.bounds[key].hex(), "flagged": None,
                                  "decision": "not_applicable" if self.anomaly_state != "active" else "undecidable_p1_failed"}:
                bad("anomaly_claim")
            return
        if p1rec.get("outcome") != "completed":
            bad("p1_outcome_claim")
        if p1rec.get("nodal_tensor_sha256") != C.tensor_sha256(nodal):
            bad("p1_nodal_resolve_hash")
        if p1rec.get("projected_tensor_sha256") != C.tensor_sha256(projected):
            bad("p1_projection_hash")
        if p1rec.get("iterations") != int(result.iterations):
            bad("p1_iterations_claim")
        pdiag = P1.compute_p1_diagnostics(system, nodal)
        claimed = p1rec.get("diagnostics") or {}
        if set(claimed) != _PKEYS or any(claimed[k] != _hx(getattr(pdiag, k)) for k in _PKEYS):
            bad("p1_diagnostics_claim")
        gates["p1_256"] = bool(pdiag.passes)
        if p1rec.get("passes") is not gates["p1_256"]:
            bad("p1_passes_claim")
        metrics = rec["metrics"]
        if metrics.get("norm_projected_p256") != _hx(_norm(projected)) or \
                metrics.get("compliance_projected_p256") != _hx(_compliance(projected)):
            bad("metric_claim_p256")
        delta = _norm(f64 - projected) / _norm(f64)
        if metrics.get("delta_field") != _hx(delta):
            bad("delta_claim")
        self.facts["delta_max"][key] = max(self.facts["delta_max"].get(key, 0.0), delta)
        flagged = None
        if self.anomaly_state == "active" and self.bounds is not None:
            flagged = bool(delta > self.bounds[key])
        expected = {"procedure": self.anomaly_state, "bound": None if self.bounds is None else self.bounds[key].hex(),
                    "flagged": flagged,
                    "decision": "not_applicable" if self.anomaly_state != "active" else "decided"}
        if rec["anomaly"] != expected:
            bad("anomaly_claim")
        if flagged:
            self.facts["anomalies"][key].append(sid)

    # ------------------------------------------------------------------ bindings, summary and ladders
    def bindings(self) -> None:
        if not self.facts["control_items"]:
            return
        aggregate = _aggregate(self.facts["control_items"])
        if self.x.mode == 'scientific' or (self.x.mode == 'production' and self.x.state4_expectations is not None):
            try:
                ce = self.x.scientific_expectations if self.x.mode == 'scientific' else self.x.state4_expectations
                if ce is None:raise ValueError('missing controls expectation')
                ce.verify(dict(self.facts['control_items']), production=True)
            except (ValueError, OSError, TypeError, KeyError):
                self.diag.add('mismatch', 'controls', 'scientific_reviewed_population' if self.x.mode == 'scientific' else 'State4_admitted_population')
        claimed = self.sources.get("controls", {}) if isinstance(self.sources, dict) else {}
        if self.x.controls_sha256 is not None and aggregate != self.x.controls_sha256:
            self.diag.add("mismatch", "controls", "controls_binding")
        if claimed.get("items") == len(self.facts["control_items"]) and claimed.get("sha256") != aggregate:
            self.diag.add("mismatch", "controls", "controls_source_claim")
        hashes = [h for _, h in self.facts["control_items"]]
        if len(set(hashes)) != len(hashes):
            self.diag.add("mismatch", "controls", "duplicate_control_across_corpus")
        if self.x.direction_seed is not None:
            commitment = hashlib.sha256(_SEED_DOMAIN.encode() + self.x.direction_seed.to_bytes(16, "big")).hexdigest()
            if self.sources.get("directions", {}).get("seed_commitment_sha256") != commitment:
                self.diag.add("mismatch", "directions", "direction_seed_commitment")
        if self.x.directions_sha256 is not None and _aggregate(self.facts["direction_items"]) != self.x.directions_sha256:
            self.diag.add("mismatch", "directions", "directions_binding")

    def summary(self) -> str:
        data = self.read("generation_summary.json")
        if data is None:
            self.diag.add("incomplete", "generation_summary.json", "summary_missing")
            return "absent"
        try:
            s = C.strict_canonical_json_loads(data)
            C.validate_self_hash(s, "summary_self_sha256")
        except _GUARD:
            self.diag.add("invalid", "generation_summary.json", "summary_bytes")
            return "invalid"
        shards = []
        for split, shard in self.x.shards():
            info = self.facts["shards"].get((split, shard), {})
            status = info.get("status", "missing")
            shards.append({"split": split, "shard": shard, "manifest_sha256": info.get("manifest_sha256"),
                           "status": {"written": "written", "failed": "failed"}.get(status, "not_reached")})
        complete = all(x["status"] == "written" for x in shards)
        written = [x["manifest_sha256"] for x in shards if x["status"] == "written"]
        claims = {
            "configuration_sha256": s.get("configuration_sha256") == self.cfg_sha,
            "completion": (s.get("completion") == "complete") == (complete and s.get("stop_reason") is None)
            and s.get("completion") in ("complete", "incomplete"),
            "shards": s.get("shards") == shards,
            "aggregate": s.get("written_manifest_aggregate_sha256") == C.domain_separated_digest(
                "operatormark_corpus_v2/summary/shard-manifests/v1", written),
            "failures_file_sha256": s.get("failures_file_sha256") == self.failures_sha,
            "measurements_file_sha256": s.get("measurements_file_sha256") == self.measurements_sha,
            "counts": s.get("counts") == {"samples_written": self.facts["samples"],
                                          "directions_written": self.facts["directions"],
                                          "failure_records": len(self.facts["failure_samples"]),
                                          "unwritten_measurements": len(self.facts["unwritten"])},
            "unwritten_anomalies": s.get("anomalies_in_unwritten_shards") == self.facts["unwritten_anomalies"],
        }
        if self.p1 == "resolve":
            claims["anomalies"] = s.get("anomalies") == self.facts["anomalies"]
            claims["delta_max"] = s.get("delta_max") == {k: v.hex() for k, v in sorted(self.facts["delta_max"].items())}
        if self.numerics == "full" and self.p1 == "resolve":
            claims["gate_failures"] = s.get("gate_failures") == self.facts["gate_failures"]
        stop = s.get("stop_reason")
        for key in set(self.facts["failed_locals"]) | set(self.facts["unwritten_by_shard"]):
            split, shard = key
            if self.facts["shards"].get(key, {}).get("status") == "written":
                continue
            covered = sorted(self.facts["failed_locals"].get(key, []) + self.facts["unwritten_by_shard"].get(key, []))
            first = shard * self.x.shard_size
            prefix = covered == list(range(first, first + len(covered)))
            full = len(covered) == self.x.shard_size
            if not prefix or (not full and (stop is None or stop == "sample_failure")):
                self.diag.add("mismatch", f"{split}/{shard}", "unwritten_shard_evidence_incomplete")
        if self.facts["stopped_shards"] and stop in (None, "sample_failure"):
            self.diag.add("mismatch", "unwritten_shard_measurements.jsonl", "stopped_measurements_without_stop")
        disagree = [k for k, ok in claims.items() if not ok]
        for key in disagree:
            self.diag.add("mismatch", "generation_summary.json", f"writer_claim_disagrees_{key}")
        if s.get("completion") == "incomplete" and not disagree:
            self.diag.add("incomplete", "generation_summary.json",
                          "generation_incomplete:" + _closed_stop_reason(s.get("stop_reason")))
        return "disagree" if disagree else "consistent"

    def ladders(self) -> dict:
        anomalies = self.facts["anomalies"]
        if self.x.mode == 'real_development':
            if self.read('anomaly_ladders.jsonl'):
                self.diag.add('mismatch', 'anomaly_ladders.jsonl', 'unconfirmed_field_ladder_not_authorized')
            return {'state': 'not_applicable_unconfirmed_field', 'trigger': 'confirmed_only',
                    'cap_per_stratum': 64, 'anomalies_retained': {k: len(v) for k,v in anomalies.items()},
                    'physical_qualification': 'not_confirmed'}
        claimed = self.facts["unwritten_anomalies"]
        unverified = self.facts["unwritten_strata"]      # every unwritten sample: its delta is not re-derived here
        total = sum(len(v) for v in anomalies.values()) + sum(len(v) for v in claimed.values())
        if self.anomaly_state is None:
            return {"state": "not_verified", "reason": "no expected anomaly procedure"}
        if self.anomaly_state != "active":
            return {"state": "no_anomaly_procedure"}
        if self.p1 != "resolve":
            return {"state": "not_verified"}
        data = self.read("anomaly_ladders.jsonl")
        if total == 0 and not any(unverified.values()):
            if data:
                self.diag.add("mismatch", "anomaly_ladders.jsonl", "ladders_without_anomalies")
            return {"state": "no_anomalies", "relaxed": "unchanged", "binary": "unchanged"}
        results = {}; seen = set(); escaped = None
        if data:
            try:
                for row in C.strict_canonical_jsonl_loads(data):
                    C.validate_self_hash(row, "record_self_sha256")
                    sid, result = row["sample_id"], row["result"]
                    e_hat = None if result["e_hat"] is None else float.fromhex(result["e_hat"])
                    valid = result["valid"]
                    if (row["configuration_sha256"] != self.cfg_sha or row["schema"] != f"{_P}.anomaly_ladder.v1"
                            or sid in seen or type(valid) is not bool
                            or sid not in anomalies.get(row["stratum"], [])):
                        raise C.CanonicalRecordError("ladder binding")
                    seen.add(sid)
                    within = bool(valid and e_hat is not None and math.isfinite(e_hat) and e_hat <= self.eps_field)
                    if result.get("within_budget") is not within:
                        self.diag.add("mismatch", sid, "ladder_within_budget_claim")
                    # Self-consistency is not independent physical verification.
                    # Reacquire the admitted control bytes and execute the frozen
                    # ladder path independently; never use claimed e_hat as truth.
                    if self.numerics == 'full' and total <= 64:
                        try:
                            actual = self.resolve_ladder(sid)
                        except (ImportError, RuntimeError, OSError) as exc:
                            if not _identifier64(type(exc).__name__):
                                escaped = exc        # re-raised below: _GUARD (OSError) must not swallow it
                                break
                            self.diag.add('incomplete', sid, 'ladder_verification_unavailable:' + _closed_type_name(exc))
                            continue
                        encoded = dict(actual)
                        encoded['e_hat'] = None if actual.get('e_hat') is None else float(actual['e_hat']).hex()
                        if encoded != result:
                            self.diag.add('mismatch', sid, 'ladder_independent_result')
                        results[sid] = bool(actual['within_budget'])
            except _GUARD:
                self.diag.add("invalid", "anomaly_ladders.jsonl", "ladder_records")
                results = {}
            if escaped is not None:
                raise escaped
        out = {"state": "evaluated", "anomalies_lower_bound": total,
               "unwritten_samples_unverified": sum(len(v) for v in unverified.values())}
        for stratum in ("relaxed", "binary"):
            ids = anomalies[stratum]
            n_stratum = len(ids) + len(claimed[stratum])
            if total > 64 and n_stratum:
                out[stratum] = "not_confirmed"
            elif any(i in results and not results[i] for i in ids):
                out[stratum] = "not_confirmed"
            elif any(i not in results for i in ids) or unverified[stratum]:
                out[stratum] = "incomplete"
            else:
                out[stratum] = "unchanged"
        out['independently_verified_ladders'] = self.ladder_checks
        out["note"] = ("Only independently re-executed ladders establish physical status; writer numbers alone "
                       "leave mandatory checks incomplete. Unwritten sample anomaly claims remain unverified.")
        return out

    def resolve_ladder(self, sample_id):
        _, split, local_text = sample_id.split('/')
        local = int(local_text); shard, offset = divmod(local, self.x.shard_size)
        rel = f'corpus_staging/{_DIRS[split]}/shard_{shard:04d}/rho.float64.npy'
        data = self.read(rel)
        if data is None:raise ValueError('ladder control unavailable')
        arrays = C.validate_npy_v2_bytes(data, expected_shape=(self.x.shard_size,1,8,8))
        rho = np.ascontiguousarray(arrays[offset,0])
        admitted = dict(self.facts['control_items']).get((split,local))
        if admitted is None or C.tensor_sha256(rho) != admitted:
            raise ValueError('ladder control changed since readback')
        result = independent_forward_ladder(rho, self.n, self.eps_field)
        self.ladder_checks += 1
        return result

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        cfg = self.configuration()
        if cfg is None:
            return self.result("absent", {})
        self.inventory()
        self.failures()
        self.measurements()
        for split, shard in self.x.shards():
            try:
                self.shard(split, shard)
            except _GUARD:
                self.diag.add("invalid", f"{split}/{shard}", "shard_check_failed")
        self.bindings()
        if self.numerics != "full":
            self.diag.add("incomplete", "numerics", "native_numerics_not_verified")
        if self.p1 != "resolve":
            self.diag.add("incomplete", "p1_256", "p1_not_resolved")
        claims = self.summary()
        return self.result(claims, self.ladders())

    def result(self, claims: str, ladders: dict) -> dict:
        status = self.diag.status()
        failures = {g: v for g, v in self.facts["gate_failures"].items() if v}
        reasons = []
        if status in ("invalid", "mismatch"):
            reasons.append(f"consistency {status}")
        if failures:
            reasons.append("hard gate failures: " + ", ".join(sorted(failures)))
        if self.facts["failure_samples"]:
            reasons.append(f"{len(self.facts['failure_samples'])} retained sample failures")
        if ladders.get("state") == "evaluated" and "incomplete" in (ladders.get("relaxed"), ladders.get("binary")):
            reasons.append("anomaly ladders missing")
        if ladders.get("state") == "not_verified":
            reasons.append("anomaly procedure not verified")
        field_failed = 'not_confirmed' in (ladders.get('relaxed'),ladders.get('binary'))
        if field_failed:reasons.append('anomaly field qualification not confirmed')
        if status in ("invalid", "mismatch") or failures or field_failed:
            verdict = "not_accepted"
        elif status == "incomplete" or reasons:
            verdict = "incomplete"
        else:
            verdict = "accepted"
        result = {
            "status": status,
            "diagnostics": [list(d) for d in sorted(self.diag.items)],
            "checked": {"samples": self.facts["samples"], "directions": self.facts["directions"],
                        "native_numerics_checked": self.facts["numerics_checked"], "p1_resolved": self.facts["p1_checked"],
                        "numerics": self.numerics, "p1": self.p1, "shards_expected": len(self.x.shards()),
                        "shards_written": sum(1 for v in self.facts["shards"].values() if v.get("status") == "written"),
                        "direction_rule_rederived": self.x.direction_seed is not None},
            "recomputed": {"gate_failures": failures, "anomalies": self.facts["anomalies"],
                           "delta_max": {k: v.hex() for k, v in sorted(self.facts["delta_max"].items())},
                           "failures_retained": sorted(self.facts["failure_samples"]),
                           "unwritten_measurements": sorted(self.facts["unwritten"]),
                           "anomalies_claimed_in_unwritten_shards": self.facts["unwritten_anomalies"]},
            "writer_claims": claims,
            "anomaly_field_status": ladders,
            "d10": {"verdict": verdict, "reasons": reasons},
            "production_usable": bool(verdict == "accepted" and self.x.mode == "production"),
            **({"scientific_usable": bool(verdict == "accepted" and status == "matched"),
                "assurance_scope": "scientific_only", "physical_qualification": "preserved_negative"}
               if self.x.mode == "scientific" else {}),
            **({"historical_coverage": d7_policy(
                self.x.scientific_expectations.d7_policy)}
               if self.x.mode == 'scientific' and self.x.scientific_expectations is not None
               and self.x.scientific_expectations.d7_policy is not None else {}),
        }
        if self.x.d8_policy is not None:
            from .d8 import integrity
            result['scientific_integrity']=integrity(result,self.x.d8_policy)
            result['scientific_usable']=result['scientific_integrity']['verdict']=='accepted'
        return result


def validate_corpus(root, expected: Expected, *, numerics: str = "full", p1: str = "resolve") -> dict:
    return _Validator(Path(root), expected, numerics, p1).run()


def independent_forward_ladder(rho, n_ref, eps_field):
    """Independent execution from acquired controls; no writer-side imports.

    Reuses only Q0-pinned Drop 1 numerical primitives and unchanged D5 rules.
    """
    from operatormark_numerics_v2 import ladder as L
    from operatormark_numerics_v2.api import NumericsError
    from operatormark_numerics_v2.tpfa_family import TPFAFamily
    from operatormark_numerics_v2.p1_family import P1Family
    fields, etas, gates = {}, {}, []
    try:
        for name, family, levels in (('TPFA',TPFAFamily(),(64,128,256)),('P1',P1Family(),(128,256,512))):
            fields[name], etas[name] = [], []
            for level in levels:
                with family.open(rho,level) as session:
                    forward=session.forward
                fields[name].append(forward.field64)
                etas[name].append(L.norm64(forward.correction64))
                gates.append(bool(forward.gates_pass))
    except (NumericsError,ValueError,ArithmeticError) as exc:
        return dict(valid=False,reason='solve_failure:'+getattr(exc,'code',type(exc).__name__),e_hat=None,within_budget=False)
    denominator=L.norm64(fields['TPFA'][2])
    t=L.ladder('field',*fields['TPFA'],etas['TPFA'][1],etas['TPFA'][2],denominator)
    p=L.ladder('field',*fields['P1'],etas['P1'][1],etas['P1'][2],denominator)
    cross=L.cross('field',t,p,fields['TPFA'],denominator)
    valid=bool(cross.complete and all(gates));estimate=cross.e_hat[L.level_of(n_ref)]
    reason=None if valid else ('gate_failure' if not all(gates) else 'ladder_invalid_or_inconsistent')
    return dict(valid=valid,reason=reason,tpfa_failed=list(t.failed),p1_failed=list(p.failed),
                consistent=bool(cross.consistent),e_hat=estimate if math.isfinite(estimate) else None,
                within_budget=bool(valid and math.isfinite(estimate) and estimate<=eps_field))
