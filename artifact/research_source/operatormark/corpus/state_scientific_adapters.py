"""Private E3 bridges to the existing Batch C scientific comparisons.

These bridges neither solve a PDE nor grant State execution authority. Numerical
imports remain inside the unchanged held-array verifiers. In particular, a
caller-supplied success flag is never evidence of a scientific comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .canonical import sha256_bytes
from . import final_scientific_verify as science
from . import final_verify
from .state4_workflow import OriginalBinding


class ScientificAdapterError(ValueError):
    """An E3 private binding or required comparison was not established."""


def prepare_v2_scientific_adapter_inputs(**inputs):
    """Versioned metadata seam; never routes v2 inputs through v1 science."""
    from .state_v2_support import prepare_v2_adapter_inputs
    return prepare_v2_adapter_inputs(**inputs)


def prepare_v2_custody_workflow(**inputs):
    """Versioned State 5/6 route; v1 adapters and acceptance remain unchanged."""
    from .state_v2_workflow import V2Workflow
    return V2Workflow(**inputs)


def _need(condition: bool, code: str) -> None:
    if not condition:
        raise ScientificAdapterError(code)


@dataclass(frozen=True, slots=True)
class OriginalAcquisition:
    """Independently accepted original/capture and a held-P1 ownership check."""

    original: OriginalBinding
    source_sha256: str
    capture: science.Capture
    owns_p1: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class MemberVerificationExpected:
    original: OriginalBinding
    scientific: science.MemberExpectations
    capture: science.Capture
    verification_file: science.FileRef


@dataclass(frozen=True, slots=True)
class OwnedNumericalSolve:
    """Live numerical inputs. This type must never enter a retained product."""

    original: OriginalBinding
    stored_tensors: object = field(repr=False, compare=False)
    p1_solution: object = field(repr=False, compare=False)
    solve_observations: tuple
    p1_capability: object = field(repr=False, compare=False)
    source_sha256: str
    capture: science.Capture


@dataclass(frozen=True, slots=True)
class ComponentMemberResult:
    """Verified, pre-retention member result; contains no P1 numerical input."""

    original: OriginalBinding
    source_sha256: str
    capture: science.Capture
    expected_capture: science.Capture
    observations: tuple
    assessment: science.Assessment
    candidate_record_bytes: bytes
    verification_file: science.FileRef


@dataclass(frozen=True, slots=True)
class RetainedMemberReadback:
    """Independent readback after the workflow's retain-member operation."""

    file: science.FileRef
    data: bytes
    capture: science.Capture


@dataclass(frozen=True, slots=True)
class RetainedMemberResult:
    verified: ComponentMemberResult
    readback: RetainedMemberReadback


@dataclass(frozen=True, slots=True)
class ComponentRetainedScience:
    """Ordered actual retained comparisons, without a synthetic success flag."""

    phase: str
    ordered_shards: tuple
    ordered_splits: tuple
    source_identity: tuple
    shard_assessments: tuple
    split_assessments: tuple
    corpus_assessment: science.Assessment | None
    status: str
    sample_population: int
    simulation_only: bool = False
    numerical_qualification: bool = False


_SCHEDULE = tuple((split, index) for split, count in
                  (("training", 64), ("validation", 4), ("test", 8))
                  for index in range(count))
_SPLITS = ("training", "validation", "test")


def _original(value: object) -> None:
    _need(type(value) is OriginalBinding, "original_type")
    _need(value.split in _SPLITS and type(value.shard) is int
          and 0 <= value.shard < (64, 4, 8)[_SPLITS.index(value.split)]
          and type(value.offset) is int and 0 <= value.offset < 64,
          "original_identity")
    _need(all(type(item) is bytes and bool(item)
              for item in (value.token, value.file, value.custody)),
          "original_custody")


def _within(actual: science.Capture, expected: science.Capture) -> bool:
    try:
        return science.within_capture(actual, expected)
    except science.FinalScientificInputError:
        return False


def _expected_member(expected: MemberVerificationExpected) -> None:
    _need(type(expected) is MemberVerificationExpected, "member_expected_type")
    _original(expected.original)
    science.validate_final_scientific_expectations(expected.scientific)
    science.validate_capture(expected.capture)
    science.validate_file_ref(expected.verification_file)
    _need((expected.original.split, expected.original.shard,
           expected.original.offset) ==
          (expected.scientific.split, expected.scientific.shard_index,
           expected.scientific.offset), "member_original_identity")
    _need(all(_within(admission.capture, expected.capture)
              for admission in (expected.scientific.control_admission,
                                *expected.scientific.solver_admissions)),
          "member_expected_interval")


def verify_original_before_release(*, expected, accepted_control, owned_solve,
                                   candidate_record_bytes, acquisition):
    """Compare the seven real tensors and held P1 before its owned release.

    ``acquisition`` is an externally accepted custody/lease premise. Its
    callback is used only for ownership; the scientific verdict is always
    recomputed by the unchanged Batch C verifier.
    """
    _expected_member(expected)
    _need(type(acquisition) is OriginalAcquisition and
          type(owned_solve) is OwnedNumericalSolve, "numerical_boundary_type")
    _original(acquisition.original)
    _original(owned_solve.original)
    science._hash(acquisition.source_sha256)
    science._hash(owned_solve.source_sha256)
    science.validate_capture(acquisition.capture)
    science.validate_capture(owned_solve.capture)
    _need(acquisition.original == expected.original == owned_solve.original,
          "original_custody_substitution")
    _need(acquisition.source_sha256 == expected.scientific.source_sha256 ==
          owned_solve.source_sha256, "source_substitution")
    _need(_within(acquisition.capture, expected.capture) and
          _within(owned_solve.capture, acquisition.capture) and
          _within(owned_solve.capture, expected.capture), "solve_interval")
    _need(type(owned_solve.solve_observations) is tuple and
          len(owned_solve.solve_observations) == 2 and
          all(type(row) is science.SolveObservation
              for row in owned_solve.solve_observations), "solve_observations")
    _need(tuple(row.file for row in owned_solve.solve_observations) ==
          tuple(row.file for row in expected.scientific.solver_admissions),
          "solve_observation_files")
    _need(type(candidate_record_bytes) is bytes and
          len(candidate_record_bytes) == expected.verification_file.file_bytes and
          sha256_bytes(candidate_record_bytes) ==
          expected.verification_file.file_sha256, "candidate_record_binding")
    _need(type(accepted_control) is science.AcceptedControl,
          "accepted_control_type")
    _need(accepted_control.admission == expected.scientific.control_admission,
          "accepted_control_substitution")
    _need(callable(acquisition.owns_p1) and
          acquisition.owns_p1(owned_solve.p1_capability,
                              expected.original) is True,
          "held_p1_ownership")
    assessment = science.verify_final_member(
        expected=expected.scientific,
        accepted_control=accepted_control,
        stored_tensors=owned_solve.stored_tensors,
        p1_solution=owned_solve.p1_solution,
        solve_observations=owned_solve.solve_observations,
        candidate_record_bytes=candidate_record_bytes,
    )
    _need(type(assessment) is science.Assessment and
          assessment.status == "matched" and assessment.counts["overall"] == 1,
          "member_science_rejected")
    return ComponentMemberResult(
        expected.original, expected.scientific.source_sha256,
        owned_solve.capture, expected.capture,
        tuple(row.file for row in owned_solve.solve_observations),
        assessment, candidate_record_bytes, expected.verification_file,
    )


def bind_retained_member(*, verified, readback):
    """Require an actual, later independent byte readback before release."""
    _need(type(verified) is ComponentMemberResult and
          type(readback) is RetainedMemberReadback,
          "retained_member_type")
    science.validate_file_ref(readback.file)
    science.validate_capture(readback.capture)
    _need(readback.file == verified.verification_file and
          type(readback.data) is bytes and
          readback.data == verified.candidate_record_bytes and
          len(readback.data) == readback.file.file_bytes and
          sha256_bytes(readback.data) == readback.file.file_sha256,
          "retained_member_byte_substitution")
    _need(_within(readback.capture, verified.expected_capture) and
          readback.capture.first_sequence >= verified.capture.last_sequence and
          readback.capture.started_utc >= verified.capture.finished_utc,
          "retained_member_chronology")
    return RetainedMemberResult(verified, readback)


def _shards(expected_shards: tuple, phase: str) -> tuple:
    _need(type(expected_shards) is tuple and len(expected_shards) == 76,
          "shard_population")
    _need(tuple((row.split, row.shard_index) for row in expected_shards
                if type(row) is final_verify.ShardExpectations) == _SCHEDULE and
          all(type(row) is final_verify.ShardExpectations and row.phase == phase
              for row in expected_shards), "shard_order_phase")
    _need(all(row.bindings == expected_shards[0].bindings
              for row in expected_shards), "shard_binding_substitution")
    sources = tuple((row.bindings["implementation_commit"],
                     row.bindings["implementation_tree"])
                    for row in expected_shards)
    _need(len(set(sources)) == 1, "shard_source_substitution")
    return sources[0]


def _evidence(scientific_evidence: tuple) -> None:
    _need(type(scientific_evidence) is tuple and
          len(scientific_evidence) == 76 and
          all(type(row) is final_verify.RetainedScientificInput
              for row in scientific_evidence) and
          tuple((row.split, row.shard_index)
                for row in scientific_evidence) == _SCHEDULE,
          "scientific_evidence_order")


def _matched(assessments: tuple, population: int) -> bool:
    return (all(type(row) is science.Assessment and row.status == "matched"
                for row in assessments) and
            sum(row.counts["overall"] for row in assessments) == population)


def verify_retained_members(*, expected_shards, reader, scientific_evidence):
    """State 5 member phase: 76 exact shards, no future manifest dependency."""
    source = _shards(expected_shards, "state5_members")
    _evidence(scientific_evidence)
    assessments = tuple(final_verify.validate_final_shard_members(
        expected=shard, reader=reader,
        scientific_evidence=(evidence,))
        for shard, evidence in zip(expected_shards, scientific_evidence))
    status = "matched" if _matched(assessments, 4864) else "rejected"
    return ComponentRetainedScience(
        "state5_members", _SCHEDULE, (), source, assessments, (), None,
        status, sum(row.counts["overall"] for row in assessments),
    )


def verify_final_manifests(*, phase, expected, reader, scientific_evidence,
                           final_evidence_inputs):
    """Child split phase or finalizer pre-move corpus phase, never premature."""
    _need(phase in ("state6_manifests", "corpus"), "manifest_phase")
    _evidence(scientific_evidence)
    if phase == "state6_manifests":
        _need(final_evidence_inputs is None and type(expected) is tuple and
              len(expected) == 3 and
              tuple(row.split for row in expected
                    if type(row) is final_verify.SplitExpectations) == _SPLITS,
              "child_manifest_inputs")
        shards = tuple(shard for split in expected for shard in split.shards)
        source = _shards(shards, "state6_manifests")
        assessments = tuple(final_verify.validate_final_split_bundle(
            expected=split, reader=reader,
            scientific_evidence=tuple(row for row in scientific_evidence
                                      if row.split == split.split))
            for split in expected)
        status = "matched" if _matched(assessments, 4864) else "rejected"
        return ComponentRetainedScience(
            phase, _SCHEDULE, _SPLITS, source, (), assessments, None,
            status, sum(row.counts["overall"] for row in assessments),
        )
    _need(type(expected) is final_verify.CorpusExpectations and
          final_evidence_inputs is not None, "corpus_manifest_inputs")
    shards = tuple(shard for split in expected.splits for shard in split.shards)
    source = _shards(shards, "state6_manifests")
    assessment = final_verify.validate_final_corpus_bundle(
        expected=expected, reader=reader,
        scientific_evidence=scientific_evidence,
        final_evidence_inputs=final_evidence_inputs,
    )
    status = "matched" if _matched((assessment,), 4864) else "rejected"
    return ComponentRetainedScience(
        phase, _SCHEDULE, _SPLITS, source, (), (), assessment,
        status, assessment.counts["overall"],
    )


def combine_retained_science(*, members, manifests):
    """Join the earlier no-future-reference traversal to child manifests.

    This only joins results produced at the two distinct phase boundaries; it
    does not turn a caller assertion into a comparison. The caller must keep
    the accepted reader and source bindings for both traversals.
    """
    _need(type(members) is ComponentRetainedScience and
          type(manifests) is ComponentRetainedScience and
          members.phase == "state5_members" and
          manifests.phase == "state6_manifests", "science_phase_join")
    _need(members.ordered_shards == manifests.ordered_shards == _SCHEDULE and
          manifests.ordered_splits == _SPLITS and
          members.source_identity == manifests.source_identity and
          len(members.shard_assessments) == 76 and
          len(manifests.split_assessments) == 3 and
          _matched(members.shard_assessments, 4864) and
          _matched(manifests.split_assessments, 4864) and
          members.status == manifests.status == "matched" and
          members.sample_population == manifests.sample_population == 4864,
          "science_phase_join_rejected")
    return ComponentRetainedScience(
        "state6_manifests", _SCHEDULE, _SPLITS,
        members.source_identity, members.shard_assessments,
        manifests.split_assessments, None, "matched", 4864,
    )
