"""Pure verification of retained held-token Authz evidence.

Every value described as independent by this module is supplied by the caller.
The verifier performs no filesystem, native, controller, enrollment, or authority
operation and deliberately emits no permission-acceptance decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Literal, Mapping, cast

from .canonical import canonical_json_bytes, sha256_bytes, validate_self_hash
from . import permission_assessment as assessment
from . import permission_authz_records as records
from . import permission_authority as authority
from . import permission_evidence_verifier as base_verifier
from . import permission_observation_windows as observation
from . import permission_principal_authz_windows as supplied_authz
from . import private_permission_evidence as evidence


VERIFICATION_SCHEMA = (
    "operatormark.darcy_branch_c_v1.retained_authz_evidence_verification.v1"
)


class PermissionAuthzEvidenceVerificationError(ValueError):
    """Supplied retained Authz evidence is malformed or contradictory."""


@dataclass(frozen=True, slots=True)
class IndependentBaseEvidence:
    status: Literal["available", "unavailable"]
    expected_destination: evidence.PermissionEvidenceDestination | None
    expected_file_bytes: int | None
    expected_file_sha256: str | None
    expected_retained_snapshot: evidence.StoreObjectSnapshot | None
    reason: Literal["not_supplied", "unavailable"] | None


@dataclass(frozen=True, slots=True)
class IndependentJournalRecordEvidence:
    status: Literal["available", "unavailable", "not_required"]
    record_hex: str | None
    file_bytes: int | None
    file_sha256: str | None


@dataclass(frozen=True, slots=True)
class IndependentJournalOperationEvidence:
    expected_grant: authority.PermissionApprovalGrant
    expected_session_id: str | None
    reservation: IndependentJournalRecordEvidence
    terminal: IndependentJournalRecordEvidence


@dataclass(frozen=True, slots=True)
class IndependentAuthorityEvidence:
    parent: IndependentJournalOperationEvidence
    child: IndependentJournalOperationEvidence


@dataclass(frozen=True, slots=True)
class RetainedAuthzEvidenceVerification:
    schema: str
    payload_kind: Literal["authz_evidence", "authz_retention_failure"]
    file_bytes: int
    file_sha256: str
    record_self_sha256: str
    payload_commitment: str
    recorded_authz_status: Literal["not_invoked", "incomplete", "completed"]
    base_verification_status: Literal[
        "verified_capture", "verified_negative", "incomplete"
    ]
    authority_correspondence_status: Literal["corroborated", "incomplete"]
    assessment_status: Literal["completed", "incomplete"]
    assessment: assessment.PermissionBundleAssessment | None
    reason_codes: tuple[
        Literal[
            "base_unavailable",
            "base_not_retained_capture",
            "held_inputs_unavailable",
            "independent_token_unavailable",
            "journal_evidence_unavailable",
            "authority_terminal_unconfirmed",
            "authz_not_completed",
            "retention_failure_evidence",
            "named_assessment_completed",
        ],
        ...,
    ]
    permission_acceptance_emitted: Literal[False]


@dataclass(frozen=True, slots=True)
class _BaseVerification:
    status: Literal["verified_capture", "verified_negative", "incomplete"]
    data: bytes | None
    receipt: evidence.RetainedPermissionEvidence | None
    destination: evidence.PermissionEvidenceDestination | None
    snapshot: evidence.StoreObjectSnapshot | None
    unavailable: bool
    not_retained_capture: bool


@dataclass(frozen=True, slots=True)
class _AuthorityVerification:
    status: Literal["corroborated", "incomplete"]
    journal_unavailable: bool
    terminal_unconfirmed: bool


def _raise(message: str, cause: BaseException | None = None) -> None:
    error = PermissionAuthzEvidenceVerificationError(message)
    if cause is None:
        raise error
    raise error from cause


def _exact(value: object, expected: type, *, name: str) -> None:
    if type(value) is not expected:
        _raise(f"{name} has the wrong exact type")


def _sha(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _raise(f"{name} is not lowercase SHA-256 hexadecimal")
    return value


def _record_bytes(
    supplied: IndependentJournalRecordEvidence, *, name: str
) -> bytes | None:
    _exact(supplied, IndependentJournalRecordEvidence, name=name)
    if type(supplied.status) is not str:
        _raise(f"{name} status must be an exact string")
    values = (supplied.record_hex, supplied.file_bytes, supplied.file_sha256)
    if supplied.status == "available":
        if not all(item is not None for item in values):
            _raise(f"{name} available fields are partial")
        if (
            type(supplied.record_hex) is not str
            or len(supplied.record_hex) % 2
            or any(character not in "0123456789abcdef" for character in supplied.record_hex)
            or type(supplied.file_bytes) is not int
            or type(supplied.file_bytes) is bool
            or supplied.file_bytes < 0
        ):
            _raise(f"{name} available record encoding is malformed")
        data = bytes.fromhex(supplied.record_hex)
        if supplied.file_bytes != len(data) or supplied.file_sha256 != sha256_bytes(data):
            _raise(f"{name} independent byte equation mismatch")
        return data
    if supplied.status not in {"unavailable", "not_required"}:
        _raise(f"{name} status is invalid")
    if any(item is not None for item in values):
        _raise(f"{name} unavailable fields must all be null")
    return None


def _validate_independent_inputs(
    *,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
    expected_binding: records.PermissionAuthzEvidenceBinding,
    expected_destination: evidence.PermissionAuthzEvidenceDestination,
    expected_store: evidence.ConfidentialStoreExpectation,
    base_evidence: IndependentBaseEvidence,
    authority_evidence: IndependentAuthorityEvidence,
    expected_token_identity: supplied_authz.SuppliedTokenIdentity | None,
) -> None:
    for value, expected, name in (
        (expected_request, observation.PermissionObservationRequest, "expected_request"),
        (expected_access, observation.PermissionAccessExpectation, "expected_access"),
        (expected_binding, records.PermissionAuthzEvidenceBinding, "expected_binding"),
        (
            expected_destination,
            evidence.PermissionAuthzEvidenceDestination,
            "expected_destination",
        ),
        (expected_store, evidence.ConfidentialStoreExpectation, "expected_store"),
        (base_evidence, IndependentBaseEvidence, "base_evidence"),
        (authority_evidence, IndependentAuthorityEvidence, "authority_evidence"),
    ):
        _exact(value, expected, name=name)
    try:
        observation.validate_permission_observation_request(
            expected_request, expected_request=expected_request
        )
        observation.validate_permission_access_expectation(
            expected_access, expected_request=expected_request
        )
        records.validate_permission_authz_evidence_binding(
            expected_binding,
            expected_request=expected_request,
            expected_access=expected_access,
        )
        evidence.validate_confidential_store_expectation(
            expected_store, expected_request=expected_request
        )
        evidence.validate_permission_authz_evidence_destination(
            expected_destination,
            expected_request=expected_request,
            expected_store=expected_store,
            expected_authz_binding=expected_binding,
        )
    except Exception as exc:
        _raise("independent Authz expectations are invalid", exc)
    if expected_destination.authz_binding_self_sha256 != expected_binding.binding_self_sha256:
        _raise("independent destination Authz binding mismatch")
    if expected_token_identity is not None:
        _exact(
            expected_token_identity,
            supplied_authz.SuppliedTokenIdentity,
            name="expected_token_identity",
        )
        try:
            supplied_authz._validate_identity(expected_token_identity)
        except Exception as exc:
            _raise("independent token identity is invalid", exc)
    for operation, independent in (
        ("observe", authority_evidence.parent),
        ("observe_authz", authority_evidence.child),
    ):
        _exact(
            independent,
            IndependentJournalOperationEvidence,
            name=f"authority_evidence.{operation}",
        )
        _exact(
            independent.expected_grant,
            authority.PermissionApprovalGrant,
            name=f"authority_evidence.{operation}.expected_grant",
        )
        try:
            authority.validate_permission_approval_grant(
                independent.expected_grant,
                request=expected_request,
                scope=independent.expected_grant.expected_scope,
            )
        except Exception as exc:
            _raise(f"independent {operation} grant is invalid", exc)
        if independent.expected_grant.expected_scope.operation != operation:
            _raise(f"independent {operation} grant names the wrong operation")
        if independent.expected_grant.expected_access_expectation != expected_access:
            _raise(f"independent {operation} grant access mismatch")
        if independent.expected_session_id is not None:
            _sha(
                independent.expected_session_id,
                name=f"authority_evidence.{operation}.expected_session_id",
            )
        _record_bytes(independent.reservation, name=f"{operation}.reservation")
        _record_bytes(independent.terminal, name=f"{operation}.terminal")


def _verify_snapshot(
    snapshot: evidence.StoreObjectSnapshot,
    data: bytes,
    *,
    destination: evidence.PermissionAuthzEvidenceDestination,
    store: evidence.ConfidentialStoreExpectation,
    sidecar: bool,
) -> None:
    _exact(snapshot, evidence.StoreObjectSnapshot, name="expected retained snapshot")
    path = (
        destination.retention_failure_accepted_relative_path
        if sidecar
        else destination.primary_accepted_relative_path
    )
    try:
        evidence._validate_object_snapshot(snapshot, destination, permitted_paths={path})
    except Exception as exc:
        _raise("independent retained Authz snapshot is invalid", exc)
    expected = {
        "relative_path": path,
        "namespace_object_identity": store.namespace_object_identity,
        "parent_relative_path": store.records_parent_relative_path,
        "parent_object_identity": store.records_parent_object_identity,
        "volume_identifier_commitment": store.volume_identifier_commitment,
        "security_descriptor_sha256": (
            store.required_created_object_security_descriptor_sha256
        ),
        "confidential_store_profile_sha256": store.confidential_store_profile_sha256,
        "publication_state": "published",
        "size_bytes": len(data),
        "file_sha256": sha256_bytes(data),
    }
    for field_name, wanted in expected.items():
        if getattr(snapshot, field_name) != wanted:
            _raise(f"independent retained Authz snapshot {field_name} mismatch")
    if type(snapshot.object_identity) is not str or not snapshot.object_identity:
        _raise("independent retained Authz snapshot object identity is invalid")


def _base_receipt(
    result: Mapping[str, object], *, captured: bool
) -> evidence.RetainedPermissionEvidence | evidence.RetainedPermissionObservationFailure:
    cls = (
        evidence.RetainedPermissionEvidence
        if captured
        else evidence.RetainedPermissionObservationFailure
    )
    try:
        if type(result) is not dict or set(result) != {item.name for item in fields(cls)}:
            _raise("embedded base receipt field set is invalid")
        return cls(**result)  # type: ignore[arg-type]
    except PermissionAuthzEvidenceVerificationError:
        raise
    except Exception as exc:
        _raise("embedded base receipt is invalid", exc)


def _verify_negative_receipt(
    data: bytes,
    receipt: evidence.RetainedPermissionObservationFailure,
    *,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
    expected_store: evidence.ConfidentialStoreExpectation,
    expected_destination: evidence.PermissionEvidenceDestination,
    expected_snapshot: evidence.StoreObjectSnapshot,
) -> None:
    try:
        evidence.strict_private_permission_observation_failure_loads(
            data, expected_request=expected_request, expected_access=expected_access
        )
        evidence._validate_object_snapshot(
            expected_snapshot,
            expected_destination,
            permitted_paths={expected_destination.primary_accepted_relative_path},
        )
        commitment = evidence.private_permission_observation_commitment(
            data,
            authorization_file_sha256=(
                expected_request.context.state4_authorization_file_sha256
            ),
            payload_kind="observation_failure",
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=None,
            expected_destination=None,
        )
        receipt_record = asdict(receipt)
        validate_self_hash(receipt_record, "receipt_self_sha256")
    except Exception as exc:
        _raise("retained negative base evidence is invalid", exc)
    expected = {
        "schema": (
            "operatormark.darcy_branch_c_v1."
            "retained_private_permission_observation_failure.v2"
        ),
        "payload_kind": "observation_failure",
        "request_self_sha256": expected_request.request_self_sha256,
        "access_expectation_self_sha256": expected_access.expectation_self_sha256,
        "destination_self_sha256": expected_destination.destination_self_sha256,
        "store_expectation_self_sha256": expected_store.expectation_self_sha256,
        "relative_path": expected_destination.primary_accepted_relative_path,
        "file_bytes": len(data),
        "file_sha256": sha256_bytes(data),
        "commitment_domain": evidence.OBSERVATION_FAILURE_DOMAIN,
        "authorization_file_sha256": (
            expected_request.context.state4_authorization_file_sha256
        ),
        "private_observation_commitment": commitment,
        "namespace_object_identity": expected_store.namespace_object_identity,
        "records_parent_object_identity": expected_store.records_parent_object_identity,
        "store_object_identity": expected_snapshot.object_identity,
        "creation_security_descriptor_sha256": (
            expected_store.required_created_object_security_descriptor_sha256
        ),
        "confidential_store_profile_sha256": expected_store.confidential_store_profile_sha256,
        "held_object_readback_sha256": sha256_bytes(data),
        "publication_state": "retained_observation_failure_not_permission_accepted",
    }
    for field_name, wanted in expected.items():
        if getattr(receipt, field_name) != wanted:
            _raise(f"retained negative base receipt {field_name} mismatch")
    if (
        expected_snapshot.relative_path != expected_destination.primary_accepted_relative_path
        or expected_snapshot.size_bytes != len(data)
        or expected_snapshot.file_sha256 != sha256_bytes(data)
        or expected_snapshot.publication_state != "published"
    ):
        _raise("retained negative base snapshot/content mismatch")


def _verify_base_evidence(
    candidate: records.PermissionAuthzBaseObservation,
    independent: IndependentBaseEvidence,
    *,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
    expected_store: evidence.ConfidentialStoreExpectation,
) -> _BaseVerification:
    # Missing external corroboration never excuses malformed embedded bytes.
    try:
        evidence._validate_embedded_authz_base(
            candidate, expected_request=expected_request,
            expected_access=expected_access, expected_store=expected_store,
        )
    except Exception as exc:
        _raise("embedded base evidence is intrinsically invalid", exc)
    if independent.status == "unavailable":
        if (
            independent.reason not in {"not_supplied", "unavailable"}
            or any(
                item is not None
                for item in (
                    independent.expected_destination,
                    independent.expected_file_bytes,
                    independent.expected_file_sha256,
                    independent.expected_retained_snapshot,
                )
            )
        ):
            _raise("unavailable independent base evidence is contradictory")
        return _BaseVerification("incomplete", None, None, None, None, True, False)
    if independent.status != "available" or independent.reason is not None:
        _raise("independent base evidence status is invalid")
    if (
        type(independent.expected_destination) is not evidence.PermissionEvidenceDestination
        or type(independent.expected_file_bytes) is not int
        or type(independent.expected_file_bytes) is bool
        or independent.expected_file_bytes < 0
        or type(independent.expected_retained_snapshot) is not evidence.StoreObjectSnapshot
    ):
        _raise("available independent base evidence is incomplete")
    _sha(independent.expected_file_sha256, name="base_evidence.expected_file_sha256")
    destination = independent.expected_destination
    snapshot = independent.expected_retained_snapshot
    try:
        evidence.validate_permission_evidence_destination(
            destination,
            expected_request=expected_request,
            expected_store=expected_store,
        )
    except Exception as exc:
        _raise("independent base destination is invalid", exc)
    if candidate.record_hex is None:
        return _BaseVerification("incomplete", None, None, destination, snapshot, True, False)
    data = bytes.fromhex(candidate.record_hex)
    if (
        independent.expected_file_bytes != len(data)
        or independent.expected_file_sha256 != sha256_bytes(data)
        or candidate.record_file_bytes != len(data)
        or candidate.record_file_sha256 != sha256_bytes(data)
    ):
        _raise("embedded and independent base byte equations differ")
    if candidate.outcome == "captured" and candidate.retention_result_kind == "captured_receipt":
        if destination.payload_kind != "observation" or candidate.retention_result is None:
            _raise("captured base destination/receipt kind mismatch")
        receipt = _base_receipt(candidate.retention_result, captured=True)
        assert type(receipt) is evidence.RetainedPermissionEvidence
        try:
            base_verifier.verify_retained_permission_observation(
                data,
                receipt,
                expected_request=expected_request,
                expected_access=expected_access,
                expected_store=expected_store,
                expected_destination=destination,
                expected_retained_snapshot=snapshot,
            )
        except Exception as exc:
            _raise("retained captured base evidence is invalid", exc)
        return _BaseVerification(
            "verified_capture", data, receipt, destination, snapshot, False, False
        )
    if candidate.outcome == "failed" and candidate.retention_result_kind == "failure_receipt":
        if destination.payload_kind != "observation_failure" or candidate.retention_result is None:
            _raise("failed base destination/receipt kind mismatch")
        receipt = _base_receipt(candidate.retention_result, captured=False)
        assert type(receipt) is evidence.RetainedPermissionObservationFailure
        _verify_negative_receipt(
            data,
            receipt,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=expected_store,
            expected_destination=destination,
            expected_snapshot=snapshot,
        )
        return _BaseVerification(
            "verified_negative", data, None, destination, snapshot, False, True
        )
    # A primary that did not produce a success receipt is not upgraded merely
    # because its intended bytes were embedded in the Authz record.
    return _BaseVerification(
        "incomplete", data, None, destination, snapshot, False, True
    )


def _verify_held_correspondence(
    held: records.PermissionAuthzHeldInputs | None,
    base: _BaseVerification,
    *,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
) -> bool:
    if held is None or base.status != "verified_capture" or base.data is None:
        return False
    try:
        base_record = evidence.strict_private_permission_observation_loads(
            base.data,
            expected_request=expected_request,
            expected_access=expected_access,
        )
    except Exception as exc:
        _raise("verified base could not be decoded for held correspondence", exc)
    try:
        subject, parent = assessment._observation_objects(
            base_record, expected_request=expected_request
        )
        token_snapshot = base_record["access_check_token_before"]
        if type(token_snapshot) is not dict or type(token_snapshot.get("facts")) is not dict:
            _raise("base access-check token projection is malformed")
        if canonical_json_bytes(token_snapshot["facts"]) != canonical_json_bytes(
            asdict(held.duplicate_token_before_facts)
        ):
            _raise("held token facts differ from independently anchored base facts")
        if held.descriptors != (subject[0], parent[0]):
            _raise("held descriptors differ from independently anchored base descriptors")
        if held.object_bindings != (subject[1], parent[1]):
            _raise("held object bindings differ from independently anchored base objects")
    except PermissionAuthzEvidenceVerificationError:
        raise
    except Exception as exc:
        _raise("held inputs do not correspond to the verified base", exc)
    return True


def _typed_record(
    projection: records.PermissionOperationEvidenceProjection,
    *,
    terminal: bool,
) -> authority.PermissionOperationReservation | authority.PermissionTerminalRecord | None:
    value = projection.terminal if terminal else projection.reservation
    if value is None:
        return None
    try:
        if terminal:
            result = authority.PermissionTerminalRecord(**value)  # type: ignore[arg-type]
            authority._validate_terminal_record(result)
        else:
            result = authority.PermissionOperationReservation(**value)  # type: ignore[arg-type]
            authority._validate_operation_reservation(result)
        return result
    except Exception as exc:
        _raise("projected authority record is invalid", exc)


def _journal_operation(
    projection: records.PermissionOperationEvidenceProjection,
    independent: IndependentJournalOperationEvidence,
    *,
    expected_operation: Literal["observe", "observe_authz"],
) -> tuple[bool, bool]:
    grant = independent.expected_grant
    binding = authority.permission_approval_grant_binding_sha256(grant)
    if (
        projection.operation != expected_operation
        or projection.scope_self_sha256 != grant.expected_scope.scope_self_sha256
        or projection.grant_binding_sha256 not in {None, binding}
    ):
        _raise(f"{expected_operation} projection differs from independent grant")
    journal_unavailable = independent.expected_session_id is None
    terminal_unconfirmed = False
    reservation = _typed_record(projection, terminal=False)
    terminal = _typed_record(projection, terminal=True)
    if projection.reservation_call_state == "not_called":
        if independent.reservation.status != "not_required":
            _raise("uncalled reservation may only have not-required journal evidence")
    else:
        if reservation is None:
            _raise("called reservation lacks intended record")
        if (
            reservation.operation != expected_operation
            or reservation.scope_self_sha256 != grant.expected_scope.scope_self_sha256
            or reservation.grant_binding_sha256 != binding
            or reservation.state4_authorization_file_sha256 != grant.state4_authorization_file_sha256
            or reservation.request_self_sha256 != grant.expected_request.request_self_sha256
            or projection.grant_binding_sha256 != binding
            or (
                independent.expected_session_id is not None
                and reservation.session_id != independent.expected_session_id
            )
        ):
            _raise("reservation differs from independent authority evidence")
        data = _record_bytes(independent.reservation, name=f"{expected_operation}.reservation")
        if independent.reservation.status == "not_required":
            _raise("called reservation cannot use not-required journal evidence")
        if data is None:
            journal_unavailable = True
        elif data != canonical_json_bytes(asdict(reservation)):
            _raise("independent reservation bytes differ from projected record")
        if projection.reservation_call_state == "returned_valid":
            try:
                raw_acknowledgment = cast(
                    Mapping[str, object], projection.reservation_acknowledgment
                )
                acknowledgment = authority.OperationReservationAcknowledgment(
                    status=cast(object, raw_acknowledgment["status"]),
                    effect_certainty=cast(
                        object, raw_acknowledgment["effect_certainty"]
                    ),
                    reservation=authority.PermissionOperationReservation(
                        **raw_acknowledgment["reservation"]
                    ),
                )
                authority._validate_operation_reservation(acknowledgment.reservation)
                authority.validate_operation_reservation_acknowledgment(
                    acknowledgment, expected_reservation=reservation
                )
            except Exception as exc:
                _raise("projected reservation acknowledgment is invalid", exc)
            if acknowledgment.status != "reserved" or acknowledgment.effect_certainty != "confirmed":
                terminal_unconfirmed = True
        else:
            terminal_unconfirmed = True
    if projection.terminal_call_state == "not_called":
        if independent.terminal.status != "not_required":
            _raise("uncalled terminal may only have not-required journal evidence")
        if reservation is not None:
            terminal_unconfirmed = True
    else:
        if terminal is None or reservation is None:
            _raise("called terminal lacks its intended authority records")
        if any(
            getattr(terminal, field_name) != getattr(reservation, field_name)
            for field_name in (
                "session_id",
                "reservation_id",
                "allowance_id",
                "state4_authorization_file_sha256",
                "request_self_sha256",
                "operation",
                "scope_self_sha256",
            )
        ):
            _raise("projected terminal does not correspond to its reservation")
        data = _record_bytes(independent.terminal, name=f"{expected_operation}.terminal")
        if independent.terminal.status == "not_required":
            _raise("called terminal cannot use not-required journal evidence")
        if data is None:
            journal_unavailable = True
        elif data != canonical_json_bytes(asdict(terminal)):
            _raise("independent terminal bytes differ from projected record")
        if projection.terminal_call_state == "returned_valid":
            try:
                raw_acknowledgment = cast(
                    Mapping[str, object], projection.terminal_acknowledgment
                )
                acknowledgment = authority.TerminalRecordAcknowledgment(
                    status=cast(object, raw_acknowledgment["status"]),
                    effect_certainty=cast(
                        object, raw_acknowledgment["effect_certainty"]
                    ),
                    terminal=authority.PermissionTerminalRecord(
                        **raw_acknowledgment["terminal"]
                    ),
                    recorded_outcome=cast(
                        object, raw_acknowledgment["recorded_outcome"]
                    ),
                )
                authority._validate_terminal_record(acknowledgment.terminal)
                authority.validate_terminal_record_acknowledgment(
                    acknowledgment, expected_terminal=terminal
                )
            except Exception as exc:
                _raise("projected terminal acknowledgment is invalid", exc)
            if acknowledgment.status != "recorded" or acknowledgment.effect_certainty != "confirmed":
                terminal_unconfirmed = True
        else:
            # Later readable bytes do not upgrade what the original call knew.
            terminal_unconfirmed = True
    return journal_unavailable, terminal_unconfirmed


def _verify_journal_correspondence(
    operations: records.PermissionAuthzAuthorityOperations,
    independent: IndependentAuthorityEvidence,
) -> _AuthorityVerification:
    parent_flags = _journal_operation(
        operations.parent, independent.parent, expected_operation="observe"
    )
    child_flags = _journal_operation(
        operations.child, independent.child, expected_operation="observe_authz"
    )
    parent_reservation = _typed_record(operations.parent, terminal=False)
    child_reservation = _typed_record(operations.child, terminal=False)
    parent_session = independent.parent.expected_session_id
    child_session = independent.child.expected_session_id
    if parent_session is not None and child_session is not None and parent_session != child_session:
        _raise("independent parent and child sessions differ")
    if child_reservation is not None:
        if (
            parent_reservation is None
            or operations.child.parent_reservation_id != parent_reservation.reservation_id
            or child_reservation.session_id != parent_reservation.session_id
        ):
            _raise("child authority evidence has the wrong parent reservation linkage")
    elif operations.child.parent_reservation_id is not None and parent_reservation is None:
        _raise("child authority parent link names no parent reservation")
    unavailable = parent_flags[0] or child_flags[0]
    unconfirmed = parent_flags[1] or child_flags[1]
    return _AuthorityVerification(
        "corroborated" if not unavailable and not unconfirmed else "incomplete",
        unavailable,
        unconfirmed,
    )


def _verify_finish_correspondence(
    projection: records.PermissionOperationEvidenceProjection,
    *,
    attempted: bool,
    requested: str | None,
    finish: observation.AuthorityFinishOutcome | None,
) -> None:
    """Cross-bind runtime finish knowledge to the actual projected journal call."""
    if not attempted:
        # Issuer cleanup of an orphan reservation is distinct from a returned
        # child's finish. It may have journal effects without a runtime lease.
        if projection.terminal is not None and projection.terminal["requested_outcome"] != "failed":
            _raise("unacquired operation claims a non-failed orphan terminal")
        return
    if finish is None:
        _raise("attempted runtime finish lacks its outcome")
    terminal = projection.terminal
    if terminal is not None and terminal["requested_outcome"] != requested:
        _raise("runtime finish request differs from journal terminal")
    if projection.terminal_call_state == "returned_valid":
        ack = projection.terminal_acknowledgment
        assert ack is not None
        expected = observation.AuthorityFinishOutcome(
            "succeeded" if ack["status"] == "recorded" else "failed",
            ack["effect_certainty"], ack["recorded_outcome"],
            "none" if ack["status"] == "recorded" else "terminal_finish_failed",
        )
        if finish != expected:
            _raise("runtime finish contradicts original journal acknowledgment")
    elif projection.terminal_call_state in {"raised", "returned_malformed"}:
        if finish != observation.AuthorityFinishOutcome("failed", "unknown", None, "terminal_finish_failed"):
            _raise("runtime finish upgrades an unknown journal terminal")
    elif finish.finish_status == "succeeded":
        _raise("successful runtime finish lacks a journal terminal call")


def _verify_trace_correspondence(
    authz: records.PermissionAuthzWireEvidence,
    operations: records.PermissionAuthzAuthorityOperations,
    base: records.PermissionAuthzBaseObservation,
) -> bool:
    """Independently walk the checkpoint/result boundary; do not trust its label."""
    child = authz.child_accounting
    expected = (("initial", None), ("authz_access_check", 0), ("authz_access_check", 1), ("final", None))
    checks = child.checkpoint_outcomes
    for index, outcome in enumerate(checks):
        if index >= 4 or (outcome.checkpoint.step, outcome.checkpoint.ordinal) != expected[index]:
            _raise("independent trace walk found a skipped/reordered checkpoint")
        if outcome.status != "active" and index + 1 != len(checks):
            _raise("independent trace walk continues past an inactive checkpoint")
        prerequisite_results = max(0, index - 1)
        if len(authz.result_prefix) < prerequisite_results or any(
            item.access_status == "unavailable" for item in authz.result_prefix[:prerequisite_results]
        ):
            _raise("independent checkpoint walk lacks its earlier usable results")
    for index, result in enumerate(authz.result_prefix):
        if len(checks) <= index + 1 or checks[index + 1].status != "active":
            _raise("independent trace result lacks its active access checkpoint")
        if result.access_status == "unavailable" and len(checks) != index + 2:
            _raise("independent trace continues after unavailable Authz evidence")
    complete_computation = (
        len(checks) == 4 and all(item.status == "active" for item in checks)
        and len(authz.result_prefix) == 2
        and all(item.access_status != "unavailable" for item in authz.result_prefix)
        and child.acquire_status == "granted" and child.acquire_failure is None
        and child.lease_candidate_returned and child.lease_binding_status == "valid"
        and child.finish_attempted and child.requested_outcome == "succeeded"
        and child.finish_outcome == observation.AuthorityFinishOutcome("succeeded", "confirmed", "succeeded", "none")
        and not authz.failure_reason_codes
    )
    if (authz.status == "completed") != complete_computation:
        _raise("independent trace completion differs from its label")
    if checks or child.finish_attempted:
        ack = operations.child.reservation_acknowledgment
        if not (
            operations.child.reservation_call_state == "returned_valid" and ack is not None
            and ack["status"] == "reserved" and ack["effect_certainty"] == "confirmed"
        ):
            _raise("runtime child lease activity lacks a confirmed reservation")
    _verify_finish_correspondence(
        operations.child, attempted=child.finish_attempted,
        requested=child.requested_outcome, finish=child.finish_outcome,
    )
    if base.record_hex is not None:
        # Intrinsic decoding was already performed irrespective of external
        # availability. This is correspondence, not retained-base verification.
        from .canonical import strict_canonical_json_loads
        base_record = strict_canonical_json_loads(bytes.fromhex(base.record_hex))
        if base.outcome == "captured":
            parent_ack = operations.parent.reservation_acknowledgment
            if not (operations.parent.reservation_call_state == "returned_valid" and parent_ack is not None
                    and parent_ack["status"] == "reserved" and parent_ack["effect_certainty"] == "confirmed"):
                _raise("captured base lacks a confirmed parent reservation")
            _verify_finish_correspondence(
                operations.parent, attempted=True, requested="succeeded",
                finish=observation.AuthorityFinishOutcome("succeeded", "confirmed", "succeeded", "none"),
            )
        elif base.outcome == "failed":
            accounting = base_record["authority_accounting"]
            _verify_finish_correspondence(
                operations.parent, attempted=accounting["finish_attempted"],
                requested=accounting["requested_outcome"],
                finish=(observation.AuthorityFinishOutcome(
                    accounting["finish_status"], accounting["effect_certainty"],
                    accounting["recorded_outcome"], accounting["finish_reason_code"],
                ) if accounting["finish_attempted"] else None),
            )
    return complete_computation


def _logical_verification(
    authz_data: bytes,
    *,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
    expected_binding: records.PermissionAuthzEvidenceBinding,
    expected_destination: evidence.PermissionAuthzEvidenceDestination,
    expected_store: evidence.ConfidentialStoreExpectation,
    base_evidence: IndependentBaseEvidence,
    authority_evidence: IndependentAuthorityEvidence,
    expected_token_identity: supplied_authz.SuppliedTokenIdentity | None,
) -> tuple[
    dict[str, object],
    records.PermissionAuthzWireEvidence,
    _BaseVerification,
    _AuthorityVerification,
    bool,
    assessment.PermissionBundleAssessment | None,
]:
    try:
        record = records.strict_private_permission_authz_loads(
            authz_data,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_binding=expected_binding,
            expected_destination_self_sha256=expected_destination.destination_self_sha256,
            expected_store_expectation_self_sha256=expected_store.expectation_self_sha256,
        )
        base_claim = records._decode_base_observation(record["base_observation"])
        held = records._decode_held_inputs(record["held_inputs"])
        authz = records._decode_authz(record["authz"], held=held)
        raw_operations = cast(Mapping[str, object], record["authority_operations"])
        operations = records.PermissionAuthzAuthorityOperations(
            parent=records._decode_operation_projection(
                raw_operations["parent"], expected_operation="observe"
            ),
            child=records._decode_operation_projection(
                raw_operations["child"], expected_operation="observe_authz"
            ),
        )
    except Exception as exc:
        _raise("retained Authz record is invalid", exc)
    base = _verify_base_evidence(
        base_claim,
        base_evidence,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
    )
    held_corresponds = _verify_held_correspondence(
        held,
        base,
        expected_request=expected_request,
        expected_access=expected_access,
    )
    if expected_token_identity is not None and authz.expected_token_identity is not None:
        if authz.expected_token_identity != expected_token_identity:
            _raise("recorded Authz token differs from independent token identity")
    journal = _verify_journal_correspondence(operations, authority_evidence)
    complete_computation = _verify_trace_correspondence(authz, operations, base_claim)
    named: assessment.PermissionBundleAssessment | None = None
    if (
        base.status == "verified_capture"
        and base.data is not None
        and base.receipt is not None
        and base.destination is not None
        and base.snapshot is not None
        and held_corresponds
        and expected_token_identity is not None
        and complete_computation
        and len(authz.result_prefix) == 2
        and journal.status == "corroborated"
    ):
        try:
            named = assessment.assess_supplied_permission_bundle(
                base.data,
                base.receipt,
                cast(tuple[supplied_authz.SuppliedTokenAccessResult, supplied_authz.SuppliedTokenAccessResult], authz.result_prefix),
                expected_request=expected_request,
                expected_access=expected_access,
                expected_store=expected_store,
                expected_destination=base.destination,
                expected_retained_snapshot=base.snapshot,
                expected_operator_sid=expected_binding.expected_operator_sid,
                expected_authz_token_identity=expected_token_identity,
            )
        except Exception as exc:
            _raise("named supplied-pair assessment failed", exc)
    return record, authz, base, journal, held_corresponds, named


_REASON_ORDER = (
    "base_unavailable",
    "base_not_retained_capture",
    "held_inputs_unavailable",
    "independent_token_unavailable",
    "journal_evidence_unavailable",
    "authority_terminal_unconfirmed",
    "authz_not_completed",
    "retention_failure_evidence",
    "named_assessment_completed",
)


def _result(
    *,
    payload_kind: Literal["authz_evidence", "authz_retention_failure"],
    supplied_bytes: bytes,
    record_self_sha256: str,
    payload_commitment: str,
    authz: records.PermissionAuthzWireEvidence,
    base: _BaseVerification,
    journal: _AuthorityVerification,
    held_corresponds: bool,
    named: assessment.PermissionBundleAssessment | None,
    retention_failure: bool,
    independent_token_available: bool,
) -> RetainedAuthzEvidenceVerification:
    reasons = {
        "base_unavailable": base.unavailable,
        "base_not_retained_capture": base.not_retained_capture,
        "held_inputs_unavailable": not held_corresponds,
        "independent_token_unavailable": not independent_token_available,
        "journal_evidence_unavailable": journal.journal_unavailable,
        "authority_terminal_unconfirmed": journal.terminal_unconfirmed,
        "authz_not_completed": authz.status != "completed",
        "retention_failure_evidence": retention_failure,
        "named_assessment_completed": named is not None and not retention_failure,
    }
    if retention_failure:
        named = None
    return RetainedAuthzEvidenceVerification(
        schema=VERIFICATION_SCHEMA,
        payload_kind=payload_kind,
        file_bytes=len(supplied_bytes),
        file_sha256=sha256_bytes(supplied_bytes),
        record_self_sha256=_sha(record_self_sha256, name="record_self_sha256"),
        payload_commitment=_sha(payload_commitment, name="payload_commitment"),
        recorded_authz_status=authz.status,
        base_verification_status=base.status,
        authority_correspondence_status=journal.status,
        assessment_status=(
            "completed" if named is not None and not retention_failure else "incomplete"
        ),
        assessment=named,
        reason_codes=cast(tuple[object, ...], tuple(
            reason for reason in _REASON_ORDER if reasons[reason]
        )),
        permission_acceptance_emitted=False,
    )


def verify_retained_permission_authz_evidence(
    authz_bytes: bytes,
    *,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
    expected_binding: records.PermissionAuthzEvidenceBinding,
    expected_destination: evidence.PermissionAuthzEvidenceDestination,
    expected_store: evidence.ConfidentialStoreExpectation,
    expected_authz_snapshot: evidence.StoreObjectSnapshot,
    base_evidence: IndependentBaseEvidence,
    authority_evidence: IndependentAuthorityEvidence,
    expected_token_identity: supplied_authz.SuppliedTokenIdentity | None,
) -> RetainedAuthzEvidenceVerification:
    _exact(authz_bytes, bytes, name="authz_bytes")
    _validate_independent_inputs(
        expected_request=expected_request,
        expected_access=expected_access,
        expected_binding=expected_binding,
        expected_destination=expected_destination,
        expected_store=expected_store,
        base_evidence=base_evidence,
        authority_evidence=authority_evidence,
        expected_token_identity=expected_token_identity,
    )
    _verify_snapshot(
        expected_authz_snapshot,
        authz_bytes,
        destination=expected_destination,
        store=expected_store,
        sidecar=False,
    )
    record, authz, base, journal, held_corresponds, named = _logical_verification(
        authz_bytes,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_binding=expected_binding,
        expected_destination=expected_destination,
        expected_store=expected_store,
        base_evidence=base_evidence,
        authority_evidence=authority_evidence,
        expected_token_identity=expected_token_identity,
    )
    try:
        commitment = records.private_permission_authz_commitment(
            authz_bytes,
            authorization_file_sha256=(
                expected_request.context.state4_authorization_file_sha256
            ),
            expected_request=expected_request,
            expected_access=expected_access,
            expected_binding=expected_binding,
            expected_destination_self_sha256=expected_destination.destination_self_sha256,
            expected_store_expectation_self_sha256=expected_store.expectation_self_sha256,
        )
    except Exception as exc:
        _raise("retained Authz commitment is invalid", exc)
    return _result(
        payload_kind="authz_evidence",
        supplied_bytes=authz_bytes,
        record_self_sha256=cast(str, record["record_self_sha256"]),
        payload_commitment=commitment,
        authz=authz,
        base=base,
        journal=journal,
        held_corresponds=held_corresponds,
        named=named,
        retention_failure=False,
        independent_token_available=expected_token_identity is not None,
    )


def verify_retained_permission_authz_retention_failure(
    sidecar_bytes: bytes,
    *,
    expected_request: observation.PermissionObservationRequest,
    expected_access: observation.PermissionAccessExpectation,
    expected_binding: records.PermissionAuthzEvidenceBinding,
    expected_destination: evidence.PermissionAuthzEvidenceDestination,
    expected_store: evidence.ConfidentialStoreExpectation,
    expected_sidecar_snapshot: evidence.StoreObjectSnapshot,
    base_evidence: IndependentBaseEvidence,
    authority_evidence: IndependentAuthorityEvidence,
    expected_token_identity: supplied_authz.SuppliedTokenIdentity | None,
) -> RetainedAuthzEvidenceVerification:
    _exact(sidecar_bytes, bytes, name="sidecar_bytes")
    _validate_independent_inputs(
        expected_request=expected_request,
        expected_access=expected_access,
        expected_binding=expected_binding,
        expected_destination=expected_destination,
        expected_store=expected_store,
        base_evidence=base_evidence,
        authority_evidence=authority_evidence,
        expected_token_identity=expected_token_identity,
    )
    _verify_snapshot(
        expected_sidecar_snapshot,
        sidecar_bytes,
        destination=expected_destination,
        store=expected_store,
        sidecar=True,
    )
    try:
        sidecar = evidence.strict_private_permission_authz_retention_failure_loads(
            sidecar_bytes,
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=expected_store,
            expected_destination=expected_destination,
            expected_authz_binding=expected_binding,
        )
        intended = cast(Mapping[str, object], sidecar["intended_authz_record"])
        intended_bytes = bytes.fromhex(cast(str, intended["record_hex"]))
        expected_intended_commitment = records.private_permission_authz_commitment(
            intended_bytes,
            authorization_file_sha256=(
                expected_request.context.state4_authorization_file_sha256
            ),
            expected_request=expected_request,
            expected_access=expected_access,
            expected_binding=expected_binding,
            expected_destination_self_sha256=expected_destination.destination_self_sha256,
            expected_store_expectation_self_sha256=expected_store.expectation_self_sha256,
        )
        if (
            intended["file_bytes"] != len(intended_bytes)
            or intended["file_sha256"] != sha256_bytes(intended_bytes)
            or intended["commitment"] != expected_intended_commitment
        ):
            _raise("sidecar intended Authz projection is contradictory")
    except PermissionAuthzEvidenceVerificationError:
        raise
    except Exception as exc:
        _raise("retained Authz failure sidecar is invalid", exc)
    _, authz, base, journal, held_corresponds, named = _logical_verification(
        intended_bytes,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_binding=expected_binding,
        expected_destination=expected_destination,
        expected_store=expected_store,
        base_evidence=base_evidence,
        authority_evidence=authority_evidence,
        expected_token_identity=expected_token_identity,
    )
    commitment = sha256_bytes(
        evidence.AUTHZ_RETENTION_FAILURE_DOMAIN.encode("utf-8")
        + b"\n"
        + expected_request.context.state4_authorization_file_sha256.encode("ascii")
        + b"\n"
        + sidecar_bytes
    )
    return _result(
        payload_kind="authz_retention_failure",
        supplied_bytes=sidecar_bytes,
        record_self_sha256=cast(str, sidecar["record_self_sha256"]),
        payload_commitment=commitment,
        authz=authz,
        base=base,
        journal=journal,
        held_corresponds=held_corresponds,
        named=named,
        retention_failure=True,
        independent_token_available=expected_token_identity is not None,
    )


__all__ = [
    "IndependentAuthorityEvidence",
    "IndependentBaseEvidence",
    "IndependentJournalOperationEvidence",
    "IndependentJournalRecordEvidence",
    "PermissionAuthzEvidenceVerificationError",
    "RetainedAuthzEvidenceVerification",
    "VERIFICATION_SCHEMA",
    "verify_retained_permission_authz_evidence",
    "verify_retained_permission_authz_retention_failure",
]
