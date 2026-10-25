"""Independent Batch C scientific comparisons; no writes, solves or authority.

P1 arrays are ephemeral inputs. Historical admissions are explicit external trust
premises, not consequences of a record hash or an assessment constructor.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
import math
import re
from types import MappingProxyType
import unicodedata

from .canonical import canonical_json_bytes, sha256_bytes, strict_canonical_json_loads, strict_canonical_jsonl_loads, tensor_sha256


class FinalScientificInputError(ValueError):
    """Malformed private expected input or an inconsistent assessment."""


SPLITS = ("training", "validation", "test")
SHARDS = (64, 4, 8)
OFFSETS = (0, 4096, 4352)
FAMILIES = ("rho", "kappa64", "u_tpfa64", "kappa32_areaavg", "u32_areaavg", "kappa16_areaavg", "u16_areaavg")
SIZES = (8, 64, 64, 32, 32, 16, 16)
COUNTS = ("structural", "TPFA64", "P1_256", "cross_solver", "overall")
MAXIMA = ("T64_algebraic_backward_error", "T64_boundary_transmissibility_error", "T64_maximum_local_conservation_ratio",
          "T64_maximum_interior_face_antisymmetry", "T64_global_balance_ratio", "P256_algebraic_backward_error",
          "P256_boundary_reconstruction_error", "P256_reaction_balance_ratio", "delta_L2", "delta_J")
TKEYS = ("algebraic_backward_error", "boundary_transmissibility_error", "maximum_local_conservation_ratio",
         "maximum_interior_face_antisymmetry", "global_balance_ratio", "rhs_relative_l2_residual")
PKEYS = ("algebraic_backward_error", "boundary_reconstruction_error", "reaction_denominator", "reaction_balance_ratio",
         "outward_boundary_flux", "integrated_source", "rhs_relative_l2_residual")
TAU_L2 = 0.02543117962436324
TAU_J = 0.022582030716092032
SCHEMA = "operatormark.darcy_branch_c_v1.sample_verification.v1"
RECORD_KEYS = frozenset(("schema", "benchmark", "split", "shard_index", "local_index", "global_index", "sample_id", "stratum",
                       "control_tensor_sha256", "rho_storage_tensor_sha256", "macro_coefficient_tensor_sha256",
                       "tensor_sha256", "verifier_tensor_sha256", "solver_diagnostics", "metrics", "gates", "record_self_sha256"))


def plain(value):
    if type(value) is MappingProxyType or type(value) is dict:
        return {key: plain(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [plain(item) for item in value]
    if type(value) in (FileRef, Capture, Admission, SolveObservation, MemberExpectations, Assessment, RetainedShardExpectations):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    return value


def _freeze(value):
    if type(value) in (dict, MappingProxyType):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if type(value) in (tuple, list):
        return tuple(_freeze(v) for v in value)
    return value


class _Frozen:
    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, _freeze(getattr(self, f.name)))


@dataclass(frozen=True)
class FileRef:
    root_binding_id: str
    relative_path: str
    file_bytes: int
    file_sha256: str


@dataclass(frozen=True)
class Capture:
    invocation_id: str
    clock_epoch_id: str
    first_sequence: int
    last_sequence: int
    started_utc: str
    finished_utc: str


@dataclass(frozen=True)
class Admission:
    role: str
    file: FileRef
    validator_source_sha256: str
    scope: str
    capture: Capture
    outcome: str
    subject_sha256: str


@dataclass(frozen=True)
class SolveObservation:
    file: FileRef
    data: bytes | None
    admission: Admission | None


@dataclass(frozen=True)
class EvidenceAcquisition:
    file: FileRef
    capture: Capture


@dataclass(frozen=True)
class AcceptedControl:
    values: object
    admission: Admission | None


@dataclass(frozen=True)
class MemberExpectations(_Frozen):
    split: str
    shard_index: int
    offset: int
    control_hashes: tuple
    source_sha256: str
    control_admission: Admission
    solver_admissions: tuple


@dataclass(frozen=True)
class RetainedShardExpectations(_Frozen):
    split: str
    shard_index: int
    control_hashes: tuple
    member_files: tuple
    evidence_file: FileRef
    admission: Admission


@dataclass(frozen=True)
class Assessment(_Frozen):
    level: str
    status: str
    recomputed: tuple
    admitted: tuple
    counts: object
    maxima: object
    diagnostics: tuple
    commitments: object
    details: object


def _need(condition, code):
    if not condition:
        raise FinalScientificInputError(code)


def _hash(value):
    _need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "hash_invalid")


def _text(value):
    _need(type(value) is str and bool(value) and unicodedata.normalize("NFC", value) == value
          and not any(unicodedata.category(c) in ("Cc", "Cs") for c in value), "text_invalid")


def validate_file_ref(value):
    _need(type(value) is FileRef, "file_type")
    _text(value.root_binding_id)
    _text(value.relative_path)
    _need(":" not in value.root_binding_id, "root_id")
    for part in value.relative_path.split("/"):
        _need(part not in ("", ".", "..") and part[-1] not in " ." and not any(c in part for c in '\\:*?"<>|'), "file_path")
        _need(re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])", part.split(".")[0]) is None, "file_alias")
    _need(type(value.file_bytes) is int and value.file_bytes >= 0, "file_length")
    _hash(value.file_sha256)


def validate_capture(value):
    _need(type(value) is Capture, "capture_type")
    _hash(value.invocation_id)
    _text(value.clock_epoch_id)
    _need(type(value.first_sequence) is int and type(value.last_sequence) is int
          and 0 <= value.first_sequence <= value.last_sequence, "capture_sequence")
    for time in (value.started_utc, value.finished_utc):
        _need(type(time) is str and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", time) is not None, "capture_time")
        try:
            datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError as exc:
            raise FinalScientificInputError("capture_time") from exc
    _need(value.started_utc <= value.finished_utc, "capture_interval")


def within_capture(actual, expected):
    validate_capture(actual)
    validate_capture(expected)
    return (actual.invocation_id == expected.invocation_id and actual.clock_epoch_id == expected.clock_epoch_id
            and expected.first_sequence <= actual.first_sequence <= actual.last_sequence <= expected.last_sequence
            and expected.started_utc <= actual.started_utc <= actual.finished_utc <= expected.finished_utc)


def validate_admission(value):
    _need(type(value) is Admission, "admission_type")
    _text(value.role)
    validate_file_ref(value.file)
    _hash(value.validator_source_sha256)
    _hash(value.subject_sha256)
    _need(value.scope in ("accepted_controls", "scientific_members", "solver_observation", "custody"), "admission_scope")
    _need(value.outcome in ("accepted", "rejected", "unavailable"), "admission_outcome")
    validate_capture(value.capture)


def qualify(actual, expected, diagnostics, role, *, subject=None):
    validate_admission(expected)
    _need(expected.outcome == "accepted", "expected_admission_not_accepted")
    if actual is None:
        diagnostics.append(("incomplete", role, "admission_unavailable"))
        return False
    try:
        validate_admission(actual)
    except FinalScientificInputError:
        diagnostics.append(("invalid", role, "admission_invalid"))
        return False
    equal = all(getattr(actual, f) == getattr(expected, f) for f in
                ("role", "file", "validator_source_sha256", "scope", "subject_sha256"))
    if not equal or not within_capture(actual.capture, expected.capture) or (subject is not None and actual.subject_sha256 != subject):
        diagnostics.append(("mismatch", role, "admission_binding"))
        return False
    if actual.outcome != "accepted":
        diagnostics.append(("incomplete" if actual.outcome == "unavailable" else "mismatch", role, "admission_" + actual.outcome))
        return False
    return True


def member_identity(split, shard_index, offset):
    _need(type(split) is str and split in SPLITS, "split")
    _need(type(shard_index) is int and 0 <= shard_index < SHARDS[SPLITS.index(split)], "shard_index")
    _need(type(offset) is int and 0 <= offset < 64, "offset")
    local = shard_index * 64 + offset
    return dict(schema=SCHEMA, benchmark="darcy_branch_c_v1", split=split, shard_index=shard_index, local_index=local,
                global_index=OFFSETS[SPLITS.index(split)] + local, sample_id=f"darcy_branch_c_v1/{split}/{local:06d}",
                stratum="relaxed" if local % 2 == 0 else "binary_32_of_64")


def validate_final_scientific_expectations(expected):
    _need(type(expected) is MemberExpectations, "expectations_type")
    member_identity(expected.split, expected.shard_index, expected.offset)
    _need(type(expected.control_hashes) is tuple and len(expected.control_hashes) == 3, "control_hashes")
    for value in expected.control_hashes:
        _hash(value)
    _hash(expected.source_sha256)
    validate_admission(expected.control_admission)
    _need(expected.control_admission.scope == "accepted_controls" and expected.control_admission.outcome == "accepted"
          and expected.control_admission.subject_sha256 == expected.control_hashes[0], "control_expectation")
    _need(type(expected.solver_admissions) is tuple and len(expected.solver_admissions) == 2, "solver_expectations")
    roles = []
    for value in expected.solver_admissions:
        validate_admission(value)
        _need(value.scope == "solver_observation" and value.outcome == "accepted", "solver_admission")
        roles.append(value.role)
    _need(roles == ["T64", "P256"] and expected.solver_admissions[0].file != expected.solver_admissions[1].file, "solver_roles")


def make_assessment(level, diagnostics, *, recomputed=(), admitted=(), counts=None, maxima=None, commitments=None, details=None):
    status = next((s for s in ("invalid", "mismatch", "incomplete") if any(d[0] == s for d in diagnostics)), "matched")
    return Assessment(level, status, tuple(recomputed), tuple(admitted), counts or {}, maxima or {},
                      tuple(sorted(set(diagnostics))), commitments or {}, details or {})


def _ratio(a, b):
    value = a / b if b > 0 else 0.0 if a == 0 else math.inf
    _need(math.isfinite(value), "nonfinite_ratio")
    return value


def _array(np, value, shape):
    _need(type(value) is np.ndarray and value.shape == shape and value.dtype == np.dtype("<f8")
          and value.flags.c_contiguous and bool(np.isfinite(value).all()), "array_invalid")
    return np.frombuffer(value.tobytes(order="C"), dtype="<f8").reshape(shape)


def area_view(np, values, n):
    _need(n in (16, 32) and values.shape == (1, 64, 64), "area_shape")
    width = 64 // n
    result = np.empty((1, n, n), dtype="<f8")
    for j in range(n):
        for i in range(n):
            # Match the frozen 32-view's first-term initialization, including
            # signed zero; the 16-view deliberately starts from positive zero.
            total = np.float64(values[0, width*j, width*i]) if width == 2 else np.float64(0.0)
            for a in range(width):
                for b in range(width):
                    if width == 2 and a == 0 and b == 0:
                        continue
                    total = np.float64(total + values[0, width*j+a, width*i+b])
            result[0, j, i] = np.float64(total * np.float64(1.0 / (width*width)))
    _need(bool(np.isfinite(result).all()), "area_nonfinite")
    return result


def _settings(solver):
    return (dict(resolution=64, method="SuperLU", dtype="<f8", forcing="unit", boundary="zero") if solver == "T64" else
            dict(resolution=256, method="CG", preconditioner="Jacobi", initial_guess="zero", rtol=float(1e-12).hex(),
                 atol=float(0).hex(), maxiter=10000, dtype="<f8", forcing="unit", boundary="zero", diagonal="SW_NE"))


def _solve_facts(observation, expected, solver, control_hash, solution_hash, diagnostics):
    if observation is None:
        diagnostics.append(("incomplete", solver, "observation_unavailable"))
        return None
    try:
        _need(type(observation) is SolveObservation, "observation_type")
        if observation.data is None:
            diagnostics.append(("incomplete", solver, "observation_bytes_unavailable"))
        # Location and admission are independent facts, not contingent on a read.
        bound = qualify(observation.admission, expected, diagnostics, solver, subject=solution_hash)
        validate_file_ref(observation.file)
        if observation.file != expected.file:
            diagnostics.append(("mismatch", solver, "observation_binding"))
            bound = False
        if observation.data is None:
            return None
        _need(type(observation.data) is bytes and len(observation.data) == observation.file.file_bytes
              and sha256_bytes(observation.data) == observation.file.file_sha256, "observation_bytes")
        body = strict_canonical_json_loads(observation.data)
        _need(set(body) == {"solver", "capture", "control_tensor_sha256", "solution_tensor_sha256", "settings", "iterations", "return_code", "outcome"}, "observation_fields")
        cap = Capture(**body["capture"])
        validate_capture(cap)
        if not within_capture(cap, expected.capture):
            diagnostics.append(("mismatch", solver, "observation_binding"))
            bound = False
        if body["solver"] != solver or body["control_tensor_sha256"] != control_hash or body["solution_tensor_sha256"] != (solution_hash if solution_hash is not None else expected.subject_sha256) or body["settings"] != _settings(solver):
            diagnostics.append(("mismatch", solver, "solve_input_binding"))
            bound = False
        _need(body["outcome"] in ("completed", "failed", "unavailable"), "solve_outcome")
        if body["outcome"] == "unavailable":
            diagnostics.append(("incomplete", solver, "solve_outcome_unavailable"))
            return None
        if body["outcome"] != "completed" or type(body["return_code"]) is not int or body["return_code"] != 0:
            diagnostics.append(("mismatch", solver, "solve_failed"))
            return None
        _need(body["iterations"] is None if solver == "T64" else type(body["iterations"]) is int and 1 <= body["iterations"] <= 10000, "iterations")
        return body if bound else None
    except (ValueError, TypeError, KeyError, OverflowError):
        diagnostics.append(("invalid", solver, "observation_invalid"))
        return None


def record_maxima(record):
    result = {name: record["solver_diagnostics"]["T64"][name[4:]] for name in MAXIMA if name.startswith("T64_")}
    result.update({name: record["solver_diagnostics"]["P256"][name[5:]] for name in MAXIMA if name.startswith("P256_")})
    result.update({name: record["metrics"][name] for name in ("delta_L2", "delta_J")})
    return result


def _partial_candidate(data, known, diagnostics):
    """Compare only independently established leaves; never fill missing facts."""
    if data is None:
        diagnostics.append(("incomplete", "record", "candidate_unavailable"))
        return False
    try:
        candidate = strict_canonical_json_loads(data)
        _need(set(candidate) == RECORD_KEYS, "candidate_fields")
        _need(candidate["record_self_sha256"] == sha256_bytes(canonical_json_bytes(
            {k:v for k,v in candidate.items() if k != "record_self_sha256"})), "candidate_self_hash")
        def compare(actual, expected, pointer):
            if type(expected) is dict:
                if type(actual) is not dict:
                    diagnostics.append(("invalid", pointer, "candidate_structure"))
                    return
                for key,value in expected.items():
                    if key not in actual:
                        diagnostics.append(("invalid", pointer + "/" + key, "candidate_structure"))
                    else:
                        compare(actual[key], value, pointer + "/" + key)
            elif canonical_json_bytes(actual) != canonical_json_bytes(expected):
                diagnostics.append(("mismatch", pointer, "scientific_record_disagreement"))
        compare(candidate, known, "record")
        return True
    except (ValueError, TypeError, KeyError, OverflowError):
        diagnostics.append(("invalid", "record", "candidate_validation"))
        return False


def verify_final_member(*, expected, accepted_control, stored_tensors, p1_solution, solve_observations, candidate_record_bytes):
    validate_final_scientific_expectations(expected)
    import numpy as np
    from operatormark.solvers.darcy_tpfa import assemble_zero_dirichlet_benchmark, compute_tpfa_diagnostics
    from operatormark.solvers.darcy_p1 import assemble_p1_zero_dirichlet_benchmark, compute_p1_diagnostics, project_p1_to_64

    diagnostics = []
    recomputed = []
    admitted = []
    counts = {key: 0 for key in COUNTS}
    maxima = {key: None for key in MAXIMA}
    identity = member_identity(expected.split, expected.shard_index, expected.offset)
    if p1_solution is None:
        diagnostics.append(("incomplete", "P256", "numerical_input_unavailable"))
        # These expectations are known even if later array parsing fails.
        _partial_candidate(candidate_record_bytes, dict(identity,
            control_tensor_sha256=expected.control_hashes[0],
            rho_storage_tensor_sha256=expected.control_hashes[1],
            macro_coefficient_tensor_sha256=expected.control_hashes[2]), diagnostics)
    try:
        _need(type(accepted_control) is AcceptedControl, "accepted_control_type")
        control = _array(np, accepted_control.values, (8, 8))
        if qualify(accepted_control.admission, expected.control_admission, diagnostics, "control", subject=tensor_sha256(control)):
            admitted.append("accepted_control")
        _need(type(stored_tensors) in (dict, MappingProxyType) and set(stored_tensors) == set(FAMILIES), "tensor_inventory")
        tensors = {name: _array(np, stored_tensors[name], (1, n, n)) for name, n in zip(FAMILIES, SIZES)}
        macro = np.empty((8, 8), dtype="<f8")
        for j in range(8):
            for i in range(8):
                macro[j, i] = np.float64(np.float64(9.0) * control[j, i]) + np.float64(3.0)
        hashes = (tensor_sha256(control), tensor_sha256(tensors["rho"]), tensor_sha256(macro))
        if hashes != expected.control_hashes:
            diagnostics.append(("mismatch", "control", "original_hash_binding"))
        if tensors["rho"].tobytes() != control.tobytes():
            diagnostics.append(("mismatch", "rho", "control_storage_binding"))
        if identity["stratum"] == "relaxed":
            valid_control = not bool(np.signbit(control).any()) and bool(((control >= 0) & (control < 1)).all())
        else:
            bits = control.view("<u8")
            valid_control = bool(((bits == 0) | (bits == 0x3ff0000000000000)).all()) and int((bits == 0).sum()) == 32
        if not valid_control:
            diagnostics.append(("mismatch", "control", "stratum"))
        kappa = np.repeat(np.repeat(macro, 8, axis=0), 8, axis=1)
        if tensors["kappa64"].tobytes() != kappa.tobytes():
            diagnostics.append(("mismatch", "kappa64", "coefficient_expansion"))
        for name, source, n in (("kappa32_areaavg", "kappa64", 32), ("u32_areaavg", "u_tpfa64", 32),
                                ("kappa16_areaavg", "kappa64", 16), ("u16_areaavg", "u_tpfa64", 16)):
            if tensors[name].tobytes() != area_view(np, tensors[source], n).tobytes():
                diagnostics.append(("mismatch", name, "derived_view"))
        recomputed.append("structure_and_control")
        counts["structural"] = int(not any(d[0] in ("invalid", "mismatch") for d in diagnostics))
        tfield = tensors["u_tpfa64"][0]
        tsystem = assemble_zero_dirichlet_benchmark(kappa)
        td = compute_tpfa_diagnostics(tsystem, tfield)
        tvalues = {key: float(getattr(td, key)) for key in TKEYS}
        _need(all(math.isfinite(v) and v >= 0 for v in tvalues.values()), "T64_nonfinite")
        tpass = all(tvalues[k] <= (100 * 2.220446049250313e-16 if k == "boundary_transmissibility_error" else 1e-10) for k in TKEYS if k != "rhs_relative_l2_residual")
        # A mathematical gate is not yet an independently qualified solve.
        counts["TPFA64"] = 0
        recomputed.append("TPFA64")
        if not tpass:
            diagnostics.append(("mismatch", "T64", "numerical_gate"))
        if p1_solution is None:
            observations = []
            if type(solve_observations) is not tuple or len(solve_observations) != 2:
                diagnostics.append(("invalid", "observations", "observation_inventory"))
            else:
                observations = [_solve_facts(o, a, solver, hashes[0], solution, diagnostics)
                    for o,a,solver,solution in zip(solve_observations, expected.solver_admissions,
                        ("T64", "P256"), (tensor_sha256(tfield), None))]
                for solver,observation in zip(("T64", "P256"), observations):
                    if observation is not None:
                        admitted.append(solver + "_solve")
                counts["TPFA64"] = int(tpass and observations[0] is not None)
            known = dict(identity, control_tensor_sha256=hashes[0], rho_storage_tensor_sha256=hashes[1],
                macro_coefficient_tensor_sha256=hashes[2], tensor_sha256={name:tensor_sha256(tensors[name]) for name in FAMILIES},
                solver_diagnostics=dict(T64=dict({k:v.hex() for k,v in tvalues.items()}, passes=tpass)),
                metrics=dict(compliances=dict(T64=float(np.sum(tfield, dtype=np.float64) / np.float64(4096)).hex())),
                gates=dict(structural=bool(counts["structural"]), TPFA64=tpass))
            if observations and observations[1] is not None:
                known["solver_diagnostics"]["P256"] = dict(iterations=observations[1]["iterations"])
                known["verifier_tensor_sha256"] = dict(P256=observations[1]["solution_tensor_sha256"])
            if _partial_candidate(candidate_record_bytes, known, diagnostics):
                recomputed.append("partial_candidate_correspondence")
            for key in MAXIMA:
                if key.startswith("T64_"):
                    maxima[key] = tvalues[key[4:]].hex()
            return make_assessment("member", diagnostics, recomputed=recomputed, admitted=admitted, counts=counts, maxima=maxima)
        pfield = _array(np, p1_solution, (257, 257))
        psystem = assemble_p1_zero_dirichlet_benchmark(control, 256)
        pd = compute_p1_diagnostics(psystem, pfield)
        pvalues = {key: float(getattr(pd, key)) for key in PKEYS}
        _need(all(math.isfinite(v) for v in pvalues.values()), "P256_nonfinite")
        ppass = (0 <= pvalues["algebraic_backward_error"] <= 1e-10 and
                 0 <= pvalues["boundary_reconstruction_error"] <= 100 * 2.220446049250313e-16 and
                 0 <= pvalues["reaction_balance_ratio"] <= 1e-10)
        projected = project_p1_to_64(pfield)
        recomputed.append("P1_256_and_projection")
        if not ppass:
            diagnostics.append(("mismatch", "P256", "numerical_gate"))
        def l2(value):
            squared = float(np.sum(value * value, dtype=np.float64) / np.float64(4096))
            _need(math.isfinite(squared) and squared >= 0, "norm_nonfinite")
            return math.sqrt(squared)
        jt = float(np.sum(tfield, dtype=np.float64) / np.float64(4096))
        jp = float(np.sum(projected, dtype=np.float64) / np.float64(4096))
        sl2 = l2(projected)
        dl2 = _ratio(l2(tfield - projected), sl2)
        dj = _ratio(abs(jt - jp), abs(jp))
        cross = dl2 <= TAU_L2 and dj <= TAU_J
        counts["cross_solver"] = int(cross)
        recomputed.append("cross_solver")
        if not cross:
            diagnostics.append(("mismatch", "metrics", "cross_solver_gate"))
        _need(type(solve_observations) is tuple and len(solve_observations) == 2, "observation_inventory")
        observations = [_solve_facts(observation, admission, solver, hashes[0], solution_hash, diagnostics)
                        for observation, admission, solver, solution_hash in zip(solve_observations, expected.solver_admissions,
                            ("T64", "P256"), (tensor_sha256(tfield), tensor_sha256(pfield)))]
        for solver, observation in zip(("T64", "P256"), observations):
            if observation is not None:
                admitted.append(solver + "_solve")
        counts["TPFA64"] = int(tpass and observations[0] is not None)
        counts["P1_256"] = int(ppass and observations[1] is not None)
        record = dict(identity,
            control_tensor_sha256=hashes[0], rho_storage_tensor_sha256=hashes[1], macro_coefficient_tensor_sha256=hashes[2],
            tensor_sha256={name: tensor_sha256(tensors[name]) for name in FAMILIES},
            verifier_tensor_sha256=dict(P256=tensor_sha256(pfield), bar_P256=tensor_sha256(projected)),
            solver_diagnostics=dict(T64=dict({k: v.hex() for k, v in tvalues.items()}, passes=tpass),
                                    P256=dict({k: v.hex() for k, v in pvalues.items()}, iterations=observations[1]["iterations"] if observations[1] else None, passes=ppass)),
            metrics=dict(scales=dict(S_L2=sl2.hex(), S_J=abs(jp).hex()), compliances=dict(T64=jt.hex(), bar_P256=jp.hex()),
                         delta_L2=dl2.hex(), delta_J=dj.hex()))
        record["gates"] = dict(structural=bool(counts["structural"]), TPFA64=tpass, P1_256=ppass, cross_solver=cross,
                               overall=bool(counts["structural"] and tpass and ppass and cross))
        if candidate_record_bytes is None:
            diagnostics.append(("incomplete", "record", "candidate_unavailable"))
        else:
            candidate = strict_canonical_json_loads(candidate_record_bytes)
            _need(set(candidate) == RECORD_KEYS, "candidate_fields")
            unhashed = {k: v for k, v in candidate.items() if k != "record_self_sha256"}
            _need(candidate["record_self_sha256"] == sha256_bytes(canonical_json_bytes(unhashed)), "candidate_self_hash")
            for key, value in record.items():
                if key == "solver_diagnostics" and observations[1] is None:
                    actual = plain(candidate[key])
                    actual["P256"]["iterations"] = None
                else:
                    actual = candidate[key]
                if canonical_json_bytes(actual) != canonical_json_bytes(value):
                    diagnostics.append(("mismatch", key, "scientific_record_disagreement"))
            recomputed.append("candidate_correspondence")
        counts["overall"] = int(not diagnostics and all(counts[k] == 1 for k in COUNTS if k != "overall") and len(admitted) == 3)
        maxima = record_maxima(record)
        return make_assessment("member", diagnostics, recomputed=recomputed, admitted=admitted, counts=counts, maxima=maxima,
                               commitments=dict(source=expected.source_sha256, control=hashes[0]), details=dict(recomputed_record=record))
    except (ValueError, TypeError, KeyError, OverflowError, FloatingPointError):
        # No exception text, arrays, solver objects or traceback locals escape.
        diagnostics.append(("invalid", "numerical_input", "computation_unavailable"))
        return make_assessment("member", diagnostics, recomputed=recomputed, admitted=admitted, counts=counts, maxima=maxima)


def validate_member_assessment(assessment, **inputs):
    _need(type(assessment) is Assessment and assessment == verify_final_member(**inputs), "assessment_recomputation")


def verify_retained_scientific_shard(*, expected, candidate_verification_bytes, independent_verification_bytes, acquisition_evidence, admissions):
    _need(type(expected) is RetainedShardExpectations, "retained_expectations")
    member_identity(expected.split, expected.shard_index, 0)
    _need(len(expected.control_hashes) == 64 and all(type(row) is tuple and len(row) == 3 for row in expected.control_hashes), "retained_controls")
    for row in expected.control_hashes:
        for h in row:
            _hash(h)
    _need(len(expected.member_files) == 8 and len(set(expected.member_files)) == 8, "retained_members")
    for file in expected.member_files:
        validate_file_ref(file)
    validate_file_ref(expected.evidence_file)
    validate_admission(expected.admission)
    _need(expected.admission.scope == "scientific_members" and expected.admission.file == expected.evidence_file
          and expected.admission.subject_sha256 == expected.member_files[-1].file_sha256, "retained_admission")
    diagnostics = []
    counts = {key: 0 for key in COUNTS}
    maxima = {key: None for key in MAXIMA}
    valid_admission = False
    if type(admissions) is not tuple or len(admissions) != 1:
        diagnostics.append(("invalid" if admissions else "incomplete", "scientific", "admission_inventory"))
    else:
        valid_admission = qualify(admissions[0], expected.admission, diagnostics, "scientific")
    if acquisition_evidence is None:
        diagnostics.append(("incomplete", "scientific", "acquisition_unavailable"))
        valid_admission = False
    else:
        try:
            _need(type(acquisition_evidence) is EvidenceAcquisition, "acquisition_type")
            validate_file_ref(acquisition_evidence.file)
            if acquisition_evidence.file != expected.evidence_file:
                diagnostics.append(("mismatch", "scientific", "independent_evidence_location"))
                valid_admission = False
            if not within_capture(acquisition_evidence.capture, expected.admission.capture):
                diagnostics.append(("mismatch", "scientific", "acquisition_interval"))
                valid_admission = False
        except FinalScientificInputError:
            diagnostics.append(("invalid", "scientific", "acquisition_invalid"))
            valid_admission = False
    rows = None
    for role, data, file in (("candidate", candidate_verification_bytes, expected.member_files[-1]),
                             ("independent", independent_verification_bytes, expected.evidence_file)):
        if data is None:
            diagnostics.append(("incomplete", role, "bytes_unavailable"))
            continue
        if type(data) is not bytes or len(data) != file.file_bytes or sha256_bytes(data) != file.file_sha256:
            diagnostics.append(("mismatch", role, "file_binding"))
            continue
        try:
            # Both modules are independent consumers; this local import reuses
            # the preserved complete record grammar, not a producer validator.
            from .final_verify import validate_final_verification_jsonl
            decoded = validate_final_verification_jsonl(data, split=expected.split, shard_index=expected.shard_index)
            _need(len(decoded) == 64, "row_count")
            for offset, row in enumerate(decoded):
                _need(set(row) == RECORD_KEYS, "row_fields")
                for key, value in member_identity(expected.split, expected.shard_index, offset).items():
                    _need(type(row[key]) is type(value) and row[key] == value, "row_identity")
                _need(tuple(row[k] for k in ("control_tensor_sha256", "rho_storage_tensor_sha256", "macro_coefficient_tensor_sha256")) == expected.control_hashes[offset], "row_control")
                _need(row["record_self_sha256"] == sha256_bytes(canonical_json_bytes({k:v for k,v in row.items() if k != "record_self_sha256"})), "row_self_hash")
                _need(set(row["gates"]) == set(COUNTS) and all(type(row["gates"][k]) is bool and row["gates"][k] for k in COUNTS), "recorded_gates")
                values = record_maxima(row)
                for key, encoded in values.items():
                    value = float.fromhex(encoded)
                    limit = TAU_L2 if key == "delta_L2" else TAU_J if key == "delta_J" else 100*2.220446049250313e-16 if "boundary" in key else 1e-10
                    _need(math.isfinite(value) and value.hex() == encoded and 0 <= value <= limit, "recorded_maximum")
            if role == "independent":
                rows = decoded
        except (ValueError, TypeError, KeyError, OverflowError):
            diagnostics.append(("invalid", role, "verification_record_invalid"))
    if type(candidate_verification_bytes) is bytes and type(independent_verification_bytes) is bytes and candidate_verification_bytes != independent_verification_bytes:
        diagnostics.append(("mismatch", "scientific", "independent_record_disagreement"))
    if rows is not None and valid_admission and not diagnostics:
        counts = {key:64 for key in COUNTS}
        maxima = {key:max(float.fromhex(record_maxima(row)[key]) for row in rows).hex() for key in MAXIMA}
    return make_assessment("retained_shard", diagnostics, recomputed=("retained_byte_correspondence",),
                           admitted=("state5_scientific_members",) if valid_admission and not diagnostics else (),
                           counts=counts, maxima=maxima, commitments=dict(verification=expected.member_files[-1].file_sha256),
                           details=dict(numerical_recomputation_performed=False, synthetic_authority_not_production=True))
