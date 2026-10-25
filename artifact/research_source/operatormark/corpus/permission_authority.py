"""Injected, metadata-only permission authority issuance and accounting.

This module contains no approval enrolment, parser, filesystem journal, native
adapter, or production factory.  A trusted administrative resolver and a durable
exclusive journal are mandatory injected capabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import Literal, Protocol

from .canonical import canonical_json_bytes, require_sha256, sha256_bytes
from . import permission_observation_windows as observation
from . import private_permission_evidence as evidence


__all__ = [
    "ATTEMPT_ACTIVATION_SCHEMA",
    "APPROVAL_GRANT_SCHEMA",
    "AUTHZ_RETENTION_APPROVAL_GRANT_SCHEMA",
    "AUTHZ_RETENTION_SCOPE_SCHEMA",
    "OPERATION_RESERVATION_SCHEMA",
    "TERMINAL_RECORD_SCHEMA",
    "AttemptActivationAcknowledgment",
    "OperationReservationAcknowledgment",
    "PermissionApprovalCheckOutcome",
    "PermissionApprovalGrant",
    "PermissionApprovalResolution",
    "PermissionApprovalResolver",
    "PermissionAttemptActivation",
    "PermissionAuthorityContractError",
    "PermissionAuthorityIssuer",
    "PermissionAuthorityJournal",
    "PermissionOperationEvidenceSnapshot",
    "PermissionOperationReservation",
    "PermissionTerminalRecord",
    "TerminalRecordAcknowledgment",
    "permission_approval_grant_binding_sha256",
    "validate_attempt_activation_acknowledgment",
    "validate_operation_reservation_acknowledgment",
    "validate_permission_approval_grant",
    "validate_permission_operation_scope",
    "validate_terminal_record_acknowledgment",
]


APPROVAL_GRANT_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_approval_grant.v1"
)
AUTHZ_RETENTION_APPROVAL_GRANT_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_authz_retention_approval_grant.v1"
)
AUTHZ_RETENTION_SCOPE_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_authz_retention_scope.v1"
)
ATTEMPT_ACTIVATION_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_authority_attempt_activation.v1"
)
OPERATION_RESERVATION_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_authority_operation_reservation.v1"
)
TERMINAL_RECORD_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_authority_terminal_record.v1"
)

_OPERATIONS = frozenset(
    {
        "observe",
        "observe_authz",
        "retain_observation",
        "retain_observation_failure",
        "retain_retention_failure",
        "retain_authz_evidence",
        "retain_authz_retention_failure",
    }
)
_RESOLUTION_STATUSES = frozenset(
    {"approved", "denied", "revoked", "expired", "failed"}
)
_CHECK_STATUSES = frozenset(
    {"active", "denied", "revoked", "expired", "failed"}
)
_OUTCOMES = frozenset({"succeeded", "refused", "failed"})


class PermissionAuthorityContractError(RuntimeError):
    """A pure authority value violates the bounded issuer contract."""


class _GrantCorrespondenceError(PermissionAuthorityContractError):
    """A structurally valid independent grant does not authorize the candidate."""


@dataclass(frozen=True, slots=True)
class PermissionApprovalGrant:
    """One immutable decision returned by the trusted administrative resolver."""

    schema: str
    approval_reference: str
    state4_authorization_file_sha256: str
    accepted_state3_audit_file_sha256: str
    accepted_compute_feasibility_report_file_sha256: str
    expected_context: observation.PermissionObservationContext
    expected_source_binding: observation.PermissionComponentSource
    expected_permission_profile_file_sha256: str
    expected_request: observation.PermissionObservationRequest
    expected_access_expectation: observation.PermissionAccessExpectation
    expected_destination: (
        evidence.PermissionEvidenceDestination
        | evidence.PermissionAuthzEvidenceDestination
        | None
    )
    expected_store_expectation: evidence.ConfidentialStoreExpectation | None
    expected_scope: observation.PermissionOperationScope


@dataclass(frozen=True, slots=True)
class PermissionApprovalResolution:
    status: Literal["approved", "denied", "revoked", "expired", "failed"]
    grant: PermissionApprovalGrant | None


@dataclass(frozen=True, slots=True)
class PermissionApprovalCheckOutcome:
    status: Literal["active", "denied", "revoked", "expired", "failed"]
    grant_binding_sha256: str
    checkpoint: observation.PermissionAuthorityCheckpoint


class PermissionApprovalResolver(Protocol):
    def resolve(
        self,
        *,
        request: observation.PermissionObservationRequest,
        scope: observation.PermissionOperationScope,
    ) -> PermissionApprovalResolution: ...

    def check_active(
        self,
        *,
        grant: PermissionApprovalGrant,
        checkpoint: observation.PermissionAuthorityCheckpoint,
    ) -> PermissionApprovalCheckOutcome: ...


@dataclass(frozen=True, slots=True)
class PermissionAttemptActivation:
    schema: str
    activation_id: str
    state4_authorization_file_sha256: str
    attempt_id: str
    grant_context_binding_sha256: str


@dataclass(frozen=True, slots=True)
class AttemptActivationAcknowledgment:
    status: Literal["activated", "existing", "incomplete", "conflict", "failed"]
    effect_certainty: Literal["confirmed", "not_applied", "unknown"]
    activation: PermissionAttemptActivation
    session_id: str | None


@dataclass(frozen=True, slots=True)
class PermissionOperationReservation:
    schema: str
    reservation_id: str
    allowance_id: str
    allowance_kind: Literal[
        "observation",
        "authz_observation",
        "primary_retention",
        "retention_failure",
        "authz_primary_retention",
        "authz_retention_failure",
    ]
    session_id: str
    state4_authorization_file_sha256: str
    request_self_sha256: str
    operation: observation.Operation
    scope_self_sha256: str
    grant_binding_sha256: str


@dataclass(frozen=True, slots=True)
class OperationReservationAcknowledgment:
    status: Literal["reserved", "existing", "conflict", "failed"]
    effect_certainty: Literal["confirmed", "not_applied", "unknown"]
    reservation: PermissionOperationReservation


@dataclass(frozen=True, slots=True)
class PermissionTerminalRecord:
    schema: str
    terminal_id: str
    session_id: str
    reservation_id: str
    allowance_id: str
    state4_authorization_file_sha256: str
    request_self_sha256: str
    operation: observation.Operation
    scope_self_sha256: str
    requested_outcome: Literal["succeeded", "refused", "failed"]


@dataclass(frozen=True, slots=True)
class TerminalRecordAcknowledgment:
    status: Literal["recorded", "failed"]
    effect_certainty: Literal["confirmed", "not_applied", "unknown"]
    terminal: PermissionTerminalRecord
    recorded_outcome: Literal["succeeded", "refused", "failed"] | None


@dataclass(frozen=True, slots=True)
class PermissionOperationEvidenceSnapshot:
    """Immutable evidence of the journal calls actually made for one operation."""

    operation: observation.Operation
    scope_self_sha256: str
    grant_binding_sha256: str | None
    parent_reservation_id: str | None
    reservation_call_state: Literal[
        "not_called", "returned_valid", "returned_malformed", "raised"
    ]
    reservation: PermissionOperationReservation | None
    reservation_acknowledgment: OperationReservationAcknowledgment | None
    terminal_call_state: Literal[
        "not_called", "returned_valid", "returned_malformed", "raised"
    ]
    terminal: PermissionTerminalRecord | None
    terminal_acknowledgment: TerminalRecordAcknowledgment | None


class PermissionAuthorityJournal(Protocol):
    def activate_attempt(
        self, activation: PermissionAttemptActivation
    ) -> AttemptActivationAcknowledgment: ...

    def reserve_operation(
        self, reservation: PermissionOperationReservation
    ) -> OperationReservationAcknowledgment: ...

    def record_terminal(
        self, terminal: PermissionTerminalRecord
    ) -> TerminalRecordAcknowledgment: ...


@dataclass(frozen=True, slots=True)
class _ActiveSession:
    activation: PermissionAttemptActivation
    session_id: str


def _sha(value: object, *, name: str) -> str:
    try:
        return require_sha256(value, name=name)
    except ValueError as exc:
        raise PermissionAuthorityContractError(str(exc)) from exc


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise PermissionAuthorityContractError(f"{name} must be non-empty exact text")
    return value


def _dataclass_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(asdict(value)))


def _domain_hash(domain: str, value: object) -> str:
    return sha256_bytes(domain.encode("ascii") + b"\n" + canonical_json_bytes(value))


def _scope_payload(scope: observation.PermissionOperationScope) -> dict[str, object]:
    payload = asdict(scope)
    payload.pop("scope_self_sha256")
    return payload


def validate_permission_operation_scope(
    scope: observation.PermissionOperationScope,
    *,
    request: observation.PermissionObservationRequest,
) -> None:
    if type(scope) is not observation.PermissionOperationScope:
        raise PermissionAuthorityContractError("operation scope has the wrong exact type")
    authz_retention = scope.operation in {
        "retain_authz_evidence",
        "retain_authz_retention_failure",
    }
    expected_schema = (
        AUTHZ_RETENTION_SCOPE_SCHEMA
        if authz_retention
        else observation.OPERATION_SCOPE_SCHEMA
    )
    if scope.schema != expected_schema:
        raise PermissionAuthorityContractError("operation scope schema mismatch")
    if scope.operation not in _OPERATIONS:
        raise PermissionAuthorityContractError("operation scope names an unsupported operation")
    for name in (
        "request_self_sha256",
        "access_expectation_self_sha256",
        "scope_self_sha256",
    ):
        _sha(getattr(scope, name), name=f"scope.{name}")
    for name in ("destination_self_sha256", "store_expectation_self_sha256"):
        value = getattr(scope, name)
        if value is not None:
            _sha(value, name=f"scope.{name}")
    if scope.request_self_sha256 != request.request_self_sha256:
        raise PermissionAuthorityContractError("scope request binding mismatch")
    retention = scope.operation not in {"observe", "observe_authz"}
    if retention != (
        scope.destination_self_sha256 is not None
        and scope.store_expectation_self_sha256 is not None
    ):
        raise PermissionAuthorityContractError("scope retention bindings have invalid nullability")
    if not retention and (
        scope.destination_self_sha256 is not None
        or scope.store_expectation_self_sha256 is not None
    ):
        raise PermissionAuthorityContractError("observation scope carries retention bindings")
    if scope.scope_self_sha256 != sha256_bytes(canonical_json_bytes(_scope_payload(scope))):
        raise PermissionAuthorityContractError("operation scope self-hash mismatch")


def _validate_grant_self(grant: PermissionApprovalGrant) -> None:
    if type(grant) is not PermissionApprovalGrant:
        raise PermissionAuthorityContractError("resolver grant has the wrong exact type")
    operation = grant.expected_scope.operation
    authz_retention = operation in {
        "retain_authz_evidence",
        "retain_authz_retention_failure",
    }
    expected_grant_schema = (
        AUTHZ_RETENTION_APPROVAL_GRANT_SCHEMA
        if authz_retention
        else APPROVAL_GRANT_SCHEMA
    )
    if grant.schema != expected_grant_schema:
        raise PermissionAuthorityContractError("resolver grant schema mismatch")
    _text(grant.approval_reference, name="grant.approval_reference")
    for name in (
        "state4_authorization_file_sha256",
        "accepted_state3_audit_file_sha256",
        "accepted_compute_feasibility_report_file_sha256",
        "expected_permission_profile_file_sha256",
    ):
        _sha(getattr(grant, name), name=f"grant.{name}")
    try:
        observation.validate_permission_observation_request(
            grant.expected_request,
            expected_request=grant.expected_request,
        )
        observation.validate_permission_access_expectation(
            grant.expected_access_expectation,
            expected_request=grant.expected_request,
        )
    except observation.PermissionObservationContractError as exc:
        raise PermissionAuthorityContractError("resolver grant request/access is malformed") from exc
    if type(grant.expected_context) is not observation.PermissionObservationContext:
        raise PermissionAuthorityContractError("resolver grant context has the wrong exact type")
    if type(grant.expected_source_binding) is not observation.PermissionComponentSource:
        raise PermissionAuthorityContractError("resolver grant source has the wrong exact type")
    if grant.expected_context != grant.expected_request.context:
        raise PermissionAuthorityContractError("resolver grant context is internally inconsistent")
    if grant.expected_source_binding != grant.expected_request.source_binding:
        raise PermissionAuthorityContractError("resolver grant source is internally inconsistent")
    if (
        grant.state4_authorization_file_sha256
        != grant.expected_context.state4_authorization_file_sha256
    ):
        raise PermissionAuthorityContractError("resolver grant full A4 binding mismatch")
    if (
        grant.expected_permission_profile_file_sha256
        != grant.expected_context.permission_profile_file_sha256
        or grant.expected_permission_profile_file_sha256
        != grant.expected_access_expectation.permission_profile_file_sha256
    ):
        raise PermissionAuthorityContractError("resolver grant permission-profile binding mismatch")
    validate_permission_operation_scope(
        grant.expected_scope,
        request=grant.expected_request,
    )
    if (
        grant.expected_scope.access_expectation_self_sha256
        != grant.expected_access_expectation.expectation_self_sha256
    ):
        raise PermissionAuthorityContractError("resolver grant access/scope binding mismatch")
    if grant.expected_scope.operation in {"observe", "observe_authz"}:
        if grant.expected_destination is not None or grant.expected_store_expectation is not None:
            raise PermissionAuthorityContractError("observation grant carries retention expectations")
        return
    expected_destination_type = (
        evidence.PermissionAuthzEvidenceDestination
        if authz_retention
        else evidence.PermissionEvidenceDestination
    )
    if (
        type(grant.expected_destination) is not expected_destination_type
        or type(grant.expected_store_expectation) is not evidence.ConfidentialStoreExpectation
    ):
        raise PermissionAuthorityContractError("retention grant lacks exact destination/store")
    try:
        evidence.validate_confidential_store_expectation(
            grant.expected_store_expectation,
            expected_request=grant.expected_request,
        )
        if authz_retention:
            evidence.validate_permission_authz_evidence_destination(
                grant.expected_destination,
                expected_request=grant.expected_request,
                expected_store=grant.expected_store_expectation,
            )
        else:
            evidence.validate_permission_evidence_destination(
                grant.expected_destination,
                expected_request=grant.expected_request,
                expected_store=grant.expected_store_expectation,
            )
    except evidence.PrivatePermissionEvidenceContractError as exc:
        raise PermissionAuthorityContractError("resolver grant destination/store is malformed") from exc
    if (
        grant.expected_scope.destination_self_sha256
        != grant.expected_destination.destination_self_sha256
        or grant.expected_scope.store_expectation_self_sha256
        != grant.expected_store_expectation.expectation_self_sha256
    ):
        raise PermissionAuthorityContractError("resolver grant retention scope binding mismatch")
    expected_kind = {
        "retain_observation": "observation",
        "retain_observation_failure": "observation_failure",
        "retain_authz_evidence": "authz_evidence",
    }.get(grant.expected_scope.operation)
    if expected_kind is not None and grant.expected_destination.payload_kind != expected_kind:
        raise PermissionAuthorityContractError("resolver grant payload/operation mismatch")


def _validate_grant_correspondence(
    grant: PermissionApprovalGrant,
    *,
    request: observation.PermissionObservationRequest,
    scope: observation.PermissionOperationScope,
) -> None:
    if request != grant.expected_request:
        raise _GrantCorrespondenceError("candidate request differs from independent grant")
    if scope != grant.expected_scope:
        raise _GrantCorrespondenceError("candidate operation scope differs from independent grant")


def validate_permission_approval_grant(
    grant: PermissionApprovalGrant,
    *,
    request: observation.PermissionObservationRequest,
    scope: observation.PermissionOperationScope,
) -> None:
    _validate_grant_self(grant)
    _validate_grant_correspondence(grant, request=request, scope=scope)


def permission_approval_grant_binding_sha256(grant: PermissionApprovalGrant) -> str:
    _validate_grant_self(grant)
    return _dataclass_sha256(grant)


def _grant_context_binding_sha256(grant: PermissionApprovalGrant) -> str:
    return _domain_hash(
        "operatormark-permission-authority-grant-context-v1",
        {
            "state4_authorization_file_sha256": grant.state4_authorization_file_sha256,
            "accepted_state3_audit_file_sha256": grant.accepted_state3_audit_file_sha256,
            "accepted_compute_feasibility_report_file_sha256": (
                grant.accepted_compute_feasibility_report_file_sha256
            ),
            "expected_context": asdict(grant.expected_context),
            "expected_source_binding": asdict(grant.expected_source_binding),
            "expected_permission_profile_file_sha256": (
                grant.expected_permission_profile_file_sha256
            ),
        },
    )


def _attempt_activation(grant: PermissionApprovalGrant) -> PermissionAttemptActivation:
    context_binding = _grant_context_binding_sha256(grant)
    payload = {
        "state4_authorization_file_sha256": grant.state4_authorization_file_sha256,
        "attempt_id": grant.expected_request.attempt_id,
        "grant_context_binding_sha256": context_binding,
    }
    return PermissionAttemptActivation(
        schema=ATTEMPT_ACTIVATION_SCHEMA,
        activation_id=_domain_hash(
            "operatormark-permission-authority-attempt-activation-v1", payload
        ),
        **payload,
    )


def _validate_attempt_activation(activation: PermissionAttemptActivation) -> None:
    if type(activation) is not PermissionAttemptActivation:
        raise PermissionAuthorityContractError("attempt activation has the wrong exact type")
    if activation.schema != ATTEMPT_ACTIVATION_SCHEMA:
        raise PermissionAuthorityContractError("attempt activation schema mismatch")
    for name in (
        "activation_id",
        "state4_authorization_file_sha256",
        "grant_context_binding_sha256",
    ):
        _sha(getattr(activation, name), name=f"activation.{name}")
    _text(activation.attempt_id, name="activation.attempt_id")
    payload = {
        "state4_authorization_file_sha256": activation.state4_authorization_file_sha256,
        "attempt_id": activation.attempt_id,
        "grant_context_binding_sha256": activation.grant_context_binding_sha256,
    }
    if activation.activation_id != _domain_hash(
        "operatormark-permission-authority-attempt-activation-v1", payload
    ):
        raise PermissionAuthorityContractError("attempt activation identity mismatch")


def validate_attempt_activation_acknowledgment(
    acknowledgment: AttemptActivationAcknowledgment,
    *,
    expected_activation: PermissionAttemptActivation,
) -> None:
    if type(acknowledgment) is not AttemptActivationAcknowledgment:
        raise PermissionAuthorityContractError("activation acknowledgment has wrong exact type")
    if acknowledgment.activation != expected_activation:
        raise PermissionAuthorityContractError("activation acknowledgment binding mismatch")
    if acknowledgment.status == "activated":
        if acknowledgment.effect_certainty != "confirmed" or acknowledgment.session_id is None:
            raise PermissionAuthorityContractError("activation success is not exactly confirmed")
        _sha(acknowledgment.session_id, name="activation_acknowledgment.session_id")
    elif acknowledgment.status in {"existing", "incomplete", "conflict"}:
        if acknowledgment.effect_certainty != "not_applied" or acknowledgment.session_id is not None:
            raise PermissionAuthorityContractError("activation refusal acknowledgment is malformed")
    elif acknowledgment.status == "failed":
        if (
            acknowledgment.effect_certainty not in {"not_applied", "unknown"}
            or acknowledgment.session_id is not None
        ):
            raise PermissionAuthorityContractError("activation failure acknowledgment is malformed")
    else:
        raise PermissionAuthorityContractError("activation acknowledgment status is invalid")


def _allowance_kind(operation: observation.Operation) -> Literal[
    "observation",
    "authz_observation",
    "primary_retention",
    "retention_failure",
    "authz_primary_retention",
    "authz_retention_failure",
]:
    if operation == "observe":
        return "observation"
    if operation == "observe_authz":
        return "authz_observation"
    if operation in {"retain_observation", "retain_observation_failure"}:
        return "primary_retention"
    if operation == "retain_retention_failure":
        return "retention_failure"
    if operation == "retain_authz_evidence":
        return "authz_primary_retention"
    if operation == "retain_authz_retention_failure":
        return "authz_retention_failure"
    raise PermissionAuthorityContractError("unsupported operation has no allowance kind")


def _operation_reservation(
    *,
    session: _ActiveSession,
    grant: PermissionApprovalGrant,
    grant_binding: str,
) -> PermissionOperationReservation:
    operation = grant.expected_scope.operation
    allowance_kind = _allowance_kind(operation)
    allowance_id = _domain_hash(
        "operatormark-permission-authority-allowance-v1",
        {
            "state4_authorization_file_sha256": grant.state4_authorization_file_sha256,
            "request_self_sha256": grant.expected_request.request_self_sha256,
            "allowance_kind": allowance_kind,
        },
    )
    payload = {
        "allowance_id": allowance_id,
        "allowance_kind": allowance_kind,
        "session_id": session.session_id,
        "state4_authorization_file_sha256": grant.state4_authorization_file_sha256,
        "request_self_sha256": grant.expected_request.request_self_sha256,
        "operation": operation,
        "scope_self_sha256": grant.expected_scope.scope_self_sha256,
        "grant_binding_sha256": grant_binding,
    }
    return PermissionOperationReservation(
        schema=OPERATION_RESERVATION_SCHEMA,
        reservation_id=_domain_hash(
            "operatormark-permission-authority-operation-reservation-v1", payload
        ),
        **payload,
    )


def _validate_operation_reservation(reservation: PermissionOperationReservation) -> None:
    if type(reservation) is not PermissionOperationReservation:
        raise PermissionAuthorityContractError("operation reservation has wrong exact type")
    if reservation.schema != OPERATION_RESERVATION_SCHEMA:
        raise PermissionAuthorityContractError("operation reservation schema mismatch")
    if reservation.operation not in _OPERATIONS:
        raise PermissionAuthorityContractError("operation reservation operation is invalid")
    if reservation.allowance_kind != _allowance_kind(reservation.operation):
        raise PermissionAuthorityContractError("operation reservation allowance kind mismatch")
    for name in (
        "reservation_id",
        "allowance_id",
        "session_id",
        "state4_authorization_file_sha256",
        "request_self_sha256",
        "scope_self_sha256",
        "grant_binding_sha256",
    ):
        _sha(getattr(reservation, name), name=f"reservation.{name}")
    allowance = _domain_hash(
        "operatormark-permission-authority-allowance-v1",
        {
            "state4_authorization_file_sha256": reservation.state4_authorization_file_sha256,
            "request_self_sha256": reservation.request_self_sha256,
            "allowance_kind": reservation.allowance_kind,
        },
    )
    if reservation.allowance_id != allowance:
        raise PermissionAuthorityContractError("operation allowance identity mismatch")
    payload = asdict(reservation)
    payload.pop("schema")
    payload.pop("reservation_id")
    if reservation.reservation_id != _domain_hash(
        "operatormark-permission-authority-operation-reservation-v1", payload
    ):
        raise PermissionAuthorityContractError("operation reservation identity mismatch")


def validate_operation_reservation_acknowledgment(
    acknowledgment: OperationReservationAcknowledgment,
    *,
    expected_reservation: PermissionOperationReservation,
) -> None:
    if type(acknowledgment) is not OperationReservationAcknowledgment:
        raise PermissionAuthorityContractError("reservation acknowledgment has wrong exact type")
    if acknowledgment.reservation != expected_reservation:
        raise PermissionAuthorityContractError("reservation acknowledgment binding mismatch")
    if acknowledgment.status == "reserved":
        if acknowledgment.effect_certainty != "confirmed":
            raise PermissionAuthorityContractError("reservation success is not confirmed")
    elif acknowledgment.status in {"existing", "conflict"}:
        if acknowledgment.effect_certainty != "not_applied":
            raise PermissionAuthorityContractError("reservation refusal acknowledgment is malformed")
    elif acknowledgment.status == "failed":
        if acknowledgment.effect_certainty not in {"not_applied", "unknown"}:
            raise PermissionAuthorityContractError("reservation failure acknowledgment is malformed")
    else:
        raise PermissionAuthorityContractError("reservation acknowledgment status is invalid")


def _terminal_record(
    reservation: PermissionOperationReservation,
    requested_outcome: Literal["succeeded", "refused", "failed"],
) -> PermissionTerminalRecord:
    payload = {
        "session_id": reservation.session_id,
        "reservation_id": reservation.reservation_id,
        "allowance_id": reservation.allowance_id,
        "state4_authorization_file_sha256": reservation.state4_authorization_file_sha256,
        "request_self_sha256": reservation.request_self_sha256,
        "operation": reservation.operation,
        "scope_self_sha256": reservation.scope_self_sha256,
        "requested_outcome": requested_outcome,
    }
    return PermissionTerminalRecord(
        schema=TERMINAL_RECORD_SCHEMA,
        terminal_id=_domain_hash(
            "operatormark-permission-authority-terminal-record-v1", payload
        ),
        **payload,
    )


def _validate_terminal_record(terminal: PermissionTerminalRecord) -> None:
    if type(terminal) is not PermissionTerminalRecord:
        raise PermissionAuthorityContractError("terminal record has wrong exact type")
    if terminal.schema != TERMINAL_RECORD_SCHEMA:
        raise PermissionAuthorityContractError("terminal record schema mismatch")
    if terminal.operation not in _OPERATIONS or terminal.requested_outcome not in _OUTCOMES:
        raise PermissionAuthorityContractError("terminal record operation/outcome mismatch")
    for name in (
        "terminal_id",
        "session_id",
        "reservation_id",
        "allowance_id",
        "state4_authorization_file_sha256",
        "request_self_sha256",
        "scope_self_sha256",
    ):
        _sha(getattr(terminal, name), name=f"terminal.{name}")
    payload = asdict(terminal)
    payload.pop("schema")
    payload.pop("terminal_id")
    if terminal.terminal_id != _domain_hash(
        "operatormark-permission-authority-terminal-record-v1", payload
    ):
        raise PermissionAuthorityContractError("terminal record identity mismatch")


def validate_terminal_record_acknowledgment(
    acknowledgment: TerminalRecordAcknowledgment,
    *,
    expected_terminal: PermissionTerminalRecord,
) -> None:
    if type(acknowledgment) is not TerminalRecordAcknowledgment:
        raise PermissionAuthorityContractError("terminal acknowledgment has wrong exact type")
    if acknowledgment.terminal != expected_terminal:
        raise PermissionAuthorityContractError("terminal acknowledgment binding mismatch")
    if acknowledgment.status == "recorded":
        if (
            acknowledgment.effect_certainty != "confirmed"
            or acknowledgment.recorded_outcome != expected_terminal.requested_outcome
        ):
            raise PermissionAuthorityContractError("terminal success is not exactly confirmed")
    elif acknowledgment.status == "failed":
        if acknowledgment.effect_certainty == "confirmed":
            if acknowledgment.recorded_outcome != expected_terminal.requested_outcome:
                raise PermissionAuthorityContractError("confirmed terminal failure outcome mismatch")
        elif acknowledgment.effect_certainty in {"not_applied", "unknown"}:
            if acknowledgment.recorded_outcome is not None:
                raise PermissionAuthorityContractError("unconfirmed terminal failure names an outcome")
        else:
            raise PermissionAuthorityContractError("terminal failure certainty is invalid")
    else:
        raise PermissionAuthorityContractError("terminal acknowledgment status is invalid")


def _acquire_failure(*, refused: bool) -> observation.ComponentFailureItem:
    return observation.ComponentFailureItem(
        failure_kind="authority_refusal" if refused else "authority_failure",
        stage="authority_acquisition",
        component_operation="acquire_authority_lease",
        resource_kind=None,
        resource_role=None,
        resource_ordinal=None,
        operation_ordinal=None,
        backend_error_hex=None,
        reason_code="authority_acquire_refused" if refused else "authority_acquire_failed",
    )


def _no_lease(*, refused: bool) -> observation.AuthorityAcquireOutcome:
    return observation.AuthorityAcquireOutcome(
        status="refused" if refused else "failed",
        lease=None,
        failure=_acquire_failure(refused=refused),
    )


_OBSERVATION_STAGE_BY_STEP = {
    "initial": "ancestor_before",
    "ancestor_open": "ancestor_before",
    "ancestor_before_query": "ancestor_before",
    "subject_open": "subject_before",
    "subject_before_query": "subject_before",
    "stream_query": "stream_inventory",
    "content_chunk": "content_read",
    "security_before_query": "security_before",
    "effective_context_before_open": "effective_context_before",
    "effective_context_before_query": "effective_context_before",
    "access_token_duplicate": "access_token",
    "access_token_before_query": "access_token",
    "access_check": "access_check",
    "security_after_query": "security_after",
    "access_token_after_query": "access_token_after",
    "object_after_query": "object_after",
    "effective_context_after_open": "effective_context_after",
    "effective_context_after_query": "effective_context_after",
    "final": "final_authority_check",
}

_RETENTION_STAGE_BY_SUFFIX = {
    "namespace_open": "namespace_open",
    "namespace_validate": "namespace_validation",
    "create": "exclusive_create",
    "write": "payload_write",
    "flush": "payload_flush",
    "rewind": "payload_rewind",
    "read_chunk": "payload_readback",
    "eof": "payload_readback",
    "validate": "readback_validation",
    "snapshot": "prepublication_snapshot",
    "publish": "publication",
    "postpublish_snapshot": "postpublication_snapshot",
    "postpublish_rewind": "postpublication_readback",
    "postpublish_read_chunk": "postpublication_readback",
    "postpublish_eof": "postpublication_readback",
    "postpublish_validate": "postpublication_readback",
}


def _checkpoint_stage(checkpoint: observation.PermissionAuthorityCheckpoint) -> str:
    if checkpoint.operation == "observe":
        return _OBSERVATION_STAGE_BY_STEP.get(checkpoint.step, "final_authority_check")
    if checkpoint.operation == "observe_authz":
        return (
            "final_authority_check"
            if checkpoint.step == "final"
            else "authz_observation"
        )
    if checkpoint.step == "final":
        return "final_authority_check"
    suffix = checkpoint.step
    if suffix.startswith("primary_"):
        suffix = suffix[len("primary_") :]
    elif suffix.startswith("sidecar_"):
        suffix = suffix[len("sidecar_") :]
    return _RETENTION_STAGE_BY_SUFFIX.get(suffix, "retention_failure_retention")


def _check_failure(
    *,
    checkpoint: observation.PermissionAuthorityCheckpoint,
    refused: bool,
) -> observation.AuthorityCheckOutcome:
    return observation.AuthorityCheckOutcome(
        status="refused" if refused else "failed",
        checkpoint=checkpoint,
        failure=observation.ComponentFailureItem(
            failure_kind="authority_refusal" if refused else "authority_failure",
            stage=_checkpoint_stage(checkpoint),
            component_operation="require_active",
            resource_kind=None,
            resource_role=None,
            resource_ordinal=None,
            operation_ordinal=checkpoint.ordinal,
            backend_error_hex=None,
            reason_code="authority_inactive" if refused else "authority_check_failed",
        ),
    )


def _observation_checkpoints(
    request: observation.PermissionObservationRequest,
) -> tuple[observation.PermissionAuthorityCheckpoint, ...]:
    operation: observation.Operation = "observe"
    result: list[observation.PermissionAuthorityCheckpoint] = [
        observation.PermissionAuthorityCheckpoint(operation, "initial", None)
    ]
    ancestors = len(request.subject.required_ancestor_canonical_volume_guid_paths)
    for index in range(ancestors):
        result.append(observation.PermissionAuthorityCheckpoint(operation, "ancestor_open", index))
        result.append(
            observation.PermissionAuthorityCheckpoint(
                operation, "ancestor_before_query", index
            )
        )
    result.extend(
        (
            observation.PermissionAuthorityCheckpoint(operation, "subject_open", None),
            observation.PermissionAuthorityCheckpoint(
                operation, "subject_before_query", None
            ),
            observation.PermissionAuthorityCheckpoint(operation, "stream_query", None),
        )
    )
    chunks = (
        request.subject.expected_file_bytes + observation.CONTENT_CHUNK_BYTES - 1
    ) // observation.CONTENT_CHUNK_BYTES
    for ordinal in range(chunks + 1):
        result.append(
            observation.PermissionAuthorityCheckpoint(
                operation, "content_chunk", ordinal
            )
        )
    for ordinal in range(ancestors + 1):
        result.append(
            observation.PermissionAuthorityCheckpoint(
                operation, "security_before_query", ordinal
            )
        )
    result.extend(
        (
            observation.PermissionAuthorityCheckpoint(
                operation, "effective_context_before_open", None
            ),
            observation.PermissionAuthorityCheckpoint(
                operation, "effective_context_before_query", None
            ),
            observation.PermissionAuthorityCheckpoint(
                operation, "access_token_duplicate", None
            ),
            observation.PermissionAuthorityCheckpoint(
                operation, "access_token_before_query", None
            ),
            observation.PermissionAuthorityCheckpoint(operation, "access_check", 0),
            observation.PermissionAuthorityCheckpoint(operation, "access_check", 1),
        )
    )
    for ordinal in range(ancestors + 1):
        result.append(
            observation.PermissionAuthorityCheckpoint(
                operation, "security_after_query", ordinal
            )
        )
    result.append(
        observation.PermissionAuthorityCheckpoint(
            operation, "access_token_after_query", None
        )
    )
    for ordinal in range(ancestors + 1):
        result.append(
            observation.PermissionAuthorityCheckpoint(
                operation, "object_after_query", ordinal
            )
        )
    result.extend(
        (
            observation.PermissionAuthorityCheckpoint(
                operation, "effective_context_after_open", None
            ),
            observation.PermissionAuthorityCheckpoint(
                operation, "effective_context_after_query", None
            ),
            observation.PermissionAuthorityCheckpoint(operation, "final", None),
        )
    )
    return tuple(result)


def _retention_checkpoints(
    operation: observation.Operation,
) -> tuple[observation.PermissionAuthorityCheckpoint, ...]:
    if operation not in {
        "retain_observation",
        "retain_observation_failure",
        "retain_retention_failure",
        "retain_authz_evidence",
        "retain_authz_retention_failure",
    }:
        raise PermissionAuthorityContractError("unsupported retention checkpoint route")
    sidecar = operation in {
        "retain_retention_failure",
        "retain_authz_retention_failure",
    }
    prefix = "sidecar_" if sidecar else "primary_"
    namespace_prefix = "sidecar_" if sidecar else ""
    steps: tuple[tuple[str, int | None], ...] = (
        (f"{namespace_prefix}namespace_open", None),
        (f"{namespace_prefix}namespace_validate", None),
        (f"{prefix}create", None),
        (f"{prefix}write", None),
        (f"{prefix}flush", None),
        (f"{prefix}rewind", None),
        (f"{prefix}read_chunk", 0),
        (f"{prefix}eof", None),
        (f"{prefix}validate", None),
        (f"{prefix}snapshot", None),
        (f"{prefix}publish", None),
        (f"{prefix}postpublish_snapshot", None),
        (f"{prefix}postpublish_rewind", None),
        (f"{prefix}postpublish_read_chunk", 0),
        (f"{prefix}postpublish_eof", None),
        (f"{prefix}postpublish_validate", None),
        ("final", None),
    )
    return tuple(
        observation.PermissionAuthorityCheckpoint(operation, step, ordinal)
        for step, ordinal in steps
    )


def _authz_observation_checkpoints() -> tuple[
    observation.PermissionAuthorityCheckpoint, ...
]:
    operation: observation.Operation = "observe_authz"
    return (
        observation.PermissionAuthorityCheckpoint(operation, "initial", None),
        observation.PermissionAuthorityCheckpoint(
            operation, "authz_access_check", 0
        ),
        observation.PermissionAuthorityCheckpoint(
            operation, "authz_access_check", 1
        ),
        observation.PermissionAuthorityCheckpoint(operation, "final", None),
    )


def _expected_checkpoints(
    grant: PermissionApprovalGrant,
) -> tuple[observation.PermissionAuthorityCheckpoint, ...]:
    if grant.expected_scope.operation == "observe":
        return _observation_checkpoints(grant.expected_request)
    if grant.expected_scope.operation == "observe_authz":
        return _authz_observation_checkpoints()
    return _retention_checkpoints(grant.expected_scope.operation)


def _validate_checkpoint_shape(
    checkpoint: observation.PermissionAuthorityCheckpoint,
    *,
    expected_checkpoints: tuple[observation.PermissionAuthorityCheckpoint, ...],
) -> None:
    if type(checkpoint) is not observation.PermissionAuthorityCheckpoint:
        raise PermissionAuthorityContractError("checkpoint has the wrong exact type")
    if type(checkpoint.operation) is not str or type(checkpoint.step) is not str:
        raise PermissionAuthorityContractError(
            "checkpoint operation and step must be exact strings"
        )
    if not expected_checkpoints or checkpoint.operation != expected_checkpoints[0].operation:
        raise PermissionAuthorityContractError("checkpoint operation mismatch")
    candidates = tuple(
        expected
        for expected in expected_checkpoints
        if expected.operation == checkpoint.operation and expected.step == checkpoint.step
    )
    if not candidates:
        raise PermissionAuthorityContractError("checkpoint step is not allowed")
    allowed_ordinals = tuple(expected.ordinal for expected in candidates)
    requires_ordinal = any(ordinal is not None for ordinal in allowed_ordinals)
    if requires_ordinal:
        if any(ordinal is None for ordinal in allowed_ordinals):
            raise PermissionAuthorityContractError(
                "checkpoint definition mixes ordinal and non-ordinal forms"
            )
        if type(checkpoint.ordinal) is not int:
            raise PermissionAuthorityContractError(
                "checkpoint ordinal must be an exact integer"
            )
        if checkpoint.ordinal not in allowed_ordinals:
            raise PermissionAuthorityContractError("checkpoint ordinal is out of bounds")
    elif checkpoint.ordinal is not None:
        raise PermissionAuthorityContractError("checkpoint ordinal must be null")


class _IssuedPermissionAuthorityLease:
    __slots__ = (
        "_access_expectation_self_sha256",
        "_active_failure_refused",
        "_check_in_progress",
        "_destination_self_sha256",
        "_expected_checkpoints",
        "_finish_attempted",
        "_finish_in_progress",
        "_finish_result",
        "_grant",
        "_grant_binding_sha256",
        "_inactive",
        "_issuer",
        "_lock",
        "_next_checkpoint",
        "_operation",
        "_parent_lease",
        "_request_self_sha256",
        "_reservation",
        "_scope_self_sha256",
        "_store_expectation_self_sha256",
    )

    def __init__(
        self,
        *,
        issuer: PermissionAuthorityIssuer,
        grant: PermissionApprovalGrant,
        grant_binding_sha256: str,
        reservation: PermissionOperationReservation,
        parent_lease: _IssuedPermissionAuthorityLease | None = None,
    ) -> None:
        scope = grant.expected_scope
        self._request_self_sha256 = scope.request_self_sha256
        self._operation = scope.operation
        self._access_expectation_self_sha256 = scope.access_expectation_self_sha256
        self._destination_self_sha256 = scope.destination_self_sha256
        self._store_expectation_self_sha256 = scope.store_expectation_self_sha256
        self._scope_self_sha256 = scope.scope_self_sha256
        self._issuer = issuer
        self._grant = grant
        self._grant_binding_sha256 = grant_binding_sha256
        self._reservation = reservation
        self._parent_lease = parent_lease
        self._expected_checkpoints = _expected_checkpoints(grant)
        self._next_checkpoint = 0
        self._inactive = False
        self._active_failure_refused = True
        self._check_in_progress = False
        self._finish_attempted = False
        self._finish_in_progress = False
        self._finish_result: observation.AuthorityFinishOutcome | None = None
        self._lock = RLock()

    @property
    def request_self_sha256(self) -> str:
        return self._request_self_sha256

    @property
    def operation(self) -> observation.Operation:
        return self._operation

    @property
    def access_expectation_self_sha256(self) -> str:
        return self._access_expectation_self_sha256

    @property
    def destination_self_sha256(self) -> str | None:
        return self._destination_self_sha256

    @property
    def store_expectation_self_sha256(self) -> str | None:
        return self._store_expectation_self_sha256

    @property
    def scope_self_sha256(self) -> str:
        return self._scope_self_sha256

    def require_active(
        self, checkpoint: observation.PermissionAuthorityCheckpoint
    ) -> observation.AuthorityCheckOutcome:
        with self._issuer._lock:
            with self._lock:
                _validate_checkpoint_shape(
                    checkpoint,
                    expected_checkpoints=self._expected_checkpoints,
                )
                if self._parent_lease is None and (
                    self._issuer._open_child_lease is not None
                    or self._issuer._child_acquire_in_progress
                ):
                    return _check_failure(checkpoint=checkpoint, refused=False)
                if self._parent_lease is not None and (
                    self._issuer._open_child_lease is not self
                    or self._issuer._open_lease is not self._parent_lease
                    or not self._issuer._parent_ready_for_authz_child(
                        self._parent_lease
                    )
                ):
                    self._inactive = True
                    self._active_failure_refused = False
                    return _check_failure(checkpoint=checkpoint, refused=False)
                if self._finish_attempted or self._inactive:
                    return _check_failure(
                        checkpoint=checkpoint,
                        refused=self._active_failure_refused,
                    )
                if self._check_in_progress:
                    self._inactive = True
                    self._active_failure_refused = False
                    return _check_failure(checkpoint=checkpoint, refused=False)
                if (
                    self._next_checkpoint >= len(self._expected_checkpoints)
                    or checkpoint != self._expected_checkpoints[self._next_checkpoint]
                ):
                    self._inactive = True
                    self._active_failure_refused = False
                    return _check_failure(checkpoint=checkpoint, refused=False)
                self._check_in_progress = True
                try:
                    checked = self._issuer._resolver.check_active(
                        grant=self._grant,
                        checkpoint=checkpoint,
                    )
                    if self._finish_attempted or self._inactive:
                        return _check_failure(
                            checkpoint=checkpoint,
                            refused=self._active_failure_refused,
                        )
                    if self._parent_lease is not None and not self._issuer._parent_ready_for_authz_child(
                        self._parent_lease
                    ):
                        self._inactive = True
                        self._active_failure_refused = False
                        return _check_failure(checkpoint=checkpoint, refused=False)
                    if type(checked) is not PermissionApprovalCheckOutcome:
                        raise PermissionAuthorityContractError(
                            "approval check returned the wrong exact type"
                        )
                    _validate_checkpoint_shape(
                        checked.checkpoint,
                        expected_checkpoints=self._expected_checkpoints,
                    )
                    if (
                        checked.grant_binding_sha256 != self._grant_binding_sha256
                        or checked.checkpoint != checkpoint
                        or checked.status not in _CHECK_STATUSES
                    ):
                        raise PermissionAuthorityContractError(
                            "approval check acknowledgment is malformed or unbound"
                        )
                except Exception:
                    self._inactive = True
                    self._active_failure_refused = False
                    return _check_failure(checkpoint=checkpoint, refused=False)
                finally:
                    self._check_in_progress = False
                if checked.status == "active":
                    self._next_checkpoint += 1
                    return observation.AuthorityCheckOutcome("active", checkpoint, None)
                self._inactive = True
                self._active_failure_refused = checked.status in {
                    "denied",
                    "revoked",
                    "expired",
                }
                return _check_failure(
                    checkpoint=checkpoint,
                    refused=self._active_failure_refused,
                )

    def finish(
        self,
        requested_outcome: Literal["succeeded", "refused", "failed"],
    ) -> observation.AuthorityFinishOutcome:
        with self._issuer._lock:
            with self._lock:
                if self._parent_lease is None and (
                    self._issuer._open_child_lease is not None
                    or self._issuer._child_acquire_in_progress
                ):
                    return observation.AuthorityFinishOutcome(
                        "failed", "not_applied", None, "terminal_finish_failed"
                    )
                if self._finish_result is not None:
                    return self._finish_result
                if self._finish_in_progress or self._finish_attempted:
                    return observation.AuthorityFinishOutcome(
                        "failed", "unknown", None, "terminal_finish_failed"
                    )
                self._finish_attempted = True
                self._finish_in_progress = True
                try:
                    if requested_outcome not in _OUTCOMES:
                        result = observation.AuthorityFinishOutcome(
                            "failed", "not_applied", None, "terminal_finish_failed"
                        )
                    elif requested_outcome == "succeeded" and (
                        self._inactive
                        or self._next_checkpoint != len(self._expected_checkpoints)
                    ):
                        result = observation.AuthorityFinishOutcome(
                            "failed", "not_applied", None, "terminal_finish_failed"
                        )
                    else:
                        terminal = _terminal_record(
                            self._reservation, requested_outcome
                        )
                        key = self._issuer._operation_evidence_key(
                            self._grant.expected_request,
                            self._grant.expected_scope,
                        )
                        prior = self._issuer._operation_evidence.get(key)
                        if prior is None:
                            prior = self._issuer._empty_operation_evidence(
                                self._grant.expected_scope,
                                parent_reservation_id=(
                                    self._parent_lease._reservation.reservation_id
                                    if self._parent_lease is not None
                                    else None
                                ),
                            )
                        self._issuer._record_operation_evidence(
                            self._grant.expected_request,
                            self._grant.expected_scope,
                            replace(
                                prior,
                                terminal_call_state="not_called",
                                terminal=terminal,
                                terminal_acknowledgment=None,
                            ),
                        )
                        try:
                            _validate_terminal_record(terminal)
                        except Exception:
                            result = observation.AuthorityFinishOutcome(
                                "failed", "unknown", None, "terminal_finish_failed"
                            )
                        else:
                            try:
                                acknowledgment = self._issuer._journal.record_terminal(
                                    terminal
                                )
                            except Exception:
                                self._issuer._record_operation_evidence(
                                    self._grant.expected_request,
                                    self._grant.expected_scope,
                                    replace(
                                        prior,
                                        terminal_call_state="raised",
                                        terminal=terminal,
                                        terminal_acknowledgment=None,
                                    ),
                                )
                                result = observation.AuthorityFinishOutcome(
                                    "failed", "unknown", None, "terminal_finish_failed"
                                )
                            else:
                                try:
                                    validate_terminal_record_acknowledgment(
                                        acknowledgment,
                                        expected_terminal=terminal,
                                    )
                                except Exception:
                                    self._issuer._record_operation_evidence(
                                        self._grant.expected_request,
                                        self._grant.expected_scope,
                                        replace(
                                            prior,
                                            terminal_call_state="returned_malformed",
                                            terminal=terminal,
                                            terminal_acknowledgment=None,
                                        ),
                                    )
                                    result = observation.AuthorityFinishOutcome(
                                        "failed", "unknown", None, "terminal_finish_failed"
                                    )
                                else:
                                    self._issuer._record_operation_evidence(
                                        self._grant.expected_request,
                                        self._grant.expected_scope,
                                        replace(
                                            prior,
                                            terminal_call_state="returned_valid",
                                            terminal=terminal,
                                            terminal_acknowledgment=acknowledgment,
                                        ),
                                    )
                                    if acknowledgment.status == "recorded":
                                        result = observation.AuthorityFinishOutcome(
                                            "succeeded",
                                            "confirmed",
                                            requested_outcome,
                                            "none",
                                        )
                                    else:
                                        result = observation.AuthorityFinishOutcome(
                                            "failed",
                                            acknowledgment.effect_certainty,
                                            acknowledgment.recorded_outcome,
                                            "terminal_finish_failed",
                                        )
                    self._finish_result = result
                    self._issuer._close_lease(self)
                    return result
                finally:
                    self._finish_in_progress = False


class PermissionAuthorityIssuer:
    """One in-memory owner of a journal-confirmed attempt session."""

    __slots__ = (
        "_acquire_in_progress",
        "_child_acquire_in_progress",
        "_attempted_activations",
        "_attempted_allowances",
        "_blocked_authorizations",
        "_journal",
        "_lock",
        "_open_lease",
        "_open_child_lease",
        "_operation_evidence",
        "_resolver",
        "_session",
    )

    def __init__(
        self,
        *,
        resolver: PermissionApprovalResolver,
        journal: PermissionAuthorityJournal,
    ) -> None:
        if resolver is None or not callable(getattr(resolver, "resolve", None)) or not callable(
            getattr(resolver, "check_active", None)
        ):
            raise PermissionAuthorityContractError(
                "an explicit resolver with both required methods is mandatory"
            )
        if (
            journal is None
            or not callable(getattr(journal, "activate_attempt", None))
            or not callable(getattr(journal, "reserve_operation", None))
            or not callable(getattr(journal, "record_terminal", None))
        ):
            raise PermissionAuthorityContractError(
                "an explicit journal with all three required methods is mandatory"
            )
        self._resolver = resolver
        self._journal = journal
        self._lock = RLock()
        self._acquire_in_progress = False
        self._child_acquire_in_progress = False
        self._session: _ActiveSession | None = None
        self._open_lease: _IssuedPermissionAuthorityLease | None = None
        self._open_child_lease: _IssuedPermissionAuthorityLease | None = None
        self._operation_evidence: dict[
            tuple[str, str], PermissionOperationEvidenceSnapshot
        ] = {}
        self._attempted_activations: set[str] = set()
        self._blocked_authorizations: set[str] = set()
        self._attempted_allowances: set[str] = set()

    @staticmethod
    def _operation_evidence_key(
        request: observation.PermissionObservationRequest,
        scope: observation.PermissionOperationScope,
    ) -> tuple[str, str]:
        return request.request_self_sha256, scope.scope_self_sha256

    @staticmethod
    def _empty_operation_evidence(
        scope: observation.PermissionOperationScope,
        *,
        parent_reservation_id: str | None,
    ) -> PermissionOperationEvidenceSnapshot:
        return PermissionOperationEvidenceSnapshot(
            operation=scope.operation,
            scope_self_sha256=scope.scope_self_sha256,
            grant_binding_sha256=None,
            parent_reservation_id=parent_reservation_id,
            reservation_call_state="not_called",
            reservation=None,
            reservation_acknowledgment=None,
            terminal_call_state="not_called",
            terminal=None,
            terminal_acknowledgment=None,
        )

    def _record_operation_evidence(
        self,
        request: observation.PermissionObservationRequest,
        scope: observation.PermissionOperationScope,
        snapshot: PermissionOperationEvidenceSnapshot,
    ) -> None:
        if (
            snapshot.operation != scope.operation
            or snapshot.scope_self_sha256 != scope.scope_self_sha256
        ):
            raise PermissionAuthorityContractError("operation evidence binding mismatch")
        self._operation_evidence[self._operation_evidence_key(request, scope)] = snapshot

    def operation_evidence_snapshot(
        self,
        request: observation.PermissionObservationRequest,
        scope: observation.PermissionOperationScope,
    ) -> PermissionOperationEvidenceSnapshot:
        """Return a pure immutable copy of captured journal-call evidence."""

        try:
            observation.validate_permission_observation_request(
                request, expected_request=request
            )
            validate_permission_operation_scope(scope, request=request)
        except Exception as exc:
            raise PermissionAuthorityContractError(
                "operation evidence request/scope is malformed"
            ) from exc
        with self._lock:
            value = self._operation_evidence.get(
                self._operation_evidence_key(request, scope)
            )
            if value is None:
                return self._empty_operation_evidence(
                    scope, parent_reservation_id=None
                )
            return value

    def _parent_ready_for_authz_child(
        self,
        parent_lease: _IssuedPermissionAuthorityLease,
    ) -> bool:
        if (
            type(parent_lease) is not _IssuedPermissionAuthorityLease
            or parent_lease._issuer is not self
            or parent_lease._parent_lease is not None
            or self._open_lease is not parent_lease
            or parent_lease._operation != "observe"
            or parent_lease._finish_attempted
            or parent_lease._finish_in_progress
            or parent_lease._finish_result is not None
            or parent_lease._check_in_progress
            or parent_lease._inactive
        ):
            return False
        insertion = observation.PermissionAuthorityCheckpoint(
            "observe", "access_check", 1
        )
        positions = tuple(
            index + 1
            for index, checkpoint in enumerate(parent_lease._expected_checkpoints)
            if checkpoint == insertion
        )
        return len(positions) == 1 and parent_lease._next_checkpoint == positions[0]

    def _close_lease(self, lease: _IssuedPermissionAuthorityLease) -> None:
        if self._open_child_lease is lease:
            self._open_child_lease = None
        elif self._open_lease is lease:
            self._open_lease = None

    def acquire(
        self,
        *,
        request: observation.PermissionObservationRequest,
        scope: observation.PermissionOperationScope,
    ) -> observation.AuthorityAcquireOutcome:
        with self._lock:
            if (
                self._open_lease is not None
                or self._open_child_lease is not None
                or self._acquire_in_progress
                or self._child_acquire_in_progress
            ):
                return _no_lease(refused=True)
            self._acquire_in_progress = True
            try:
                return self._acquire_locked(request=request, scope=scope)
            finally:
                self._acquire_in_progress = False

    def acquire_observe_authz_child(
        self,
        *,
        request: observation.PermissionObservationRequest,
        scope: observation.PermissionOperationScope,
        parent_lease: observation.PermissionAuthorityLease,
    ) -> observation.AuthorityAcquireOutcome:
        """Acquire the sole Authz child at the held observation checkpoint."""

        with self._lock:
            if (
                type(parent_lease) is not _IssuedPermissionAuthorityLease
                or parent_lease._issuer is not self
                or parent_lease._parent_lease is not None
                or self._open_lease is not parent_lease
                or self._open_child_lease is not None
                or self._acquire_in_progress
                or self._child_acquire_in_progress
            ):
                return _no_lease(refused=True)
            with parent_lease._lock:
                if not self._parent_ready_for_authz_child(parent_lease):
                    return _no_lease(refused=True)
                self._child_acquire_in_progress = True
                try:
                    return self._acquire_locked(
                        request=request,
                        scope=scope,
                        parent_lease=parent_lease,
                    )
                finally:
                    self._child_acquire_in_progress = False

    def _acquire_locked(
        self,
        *,
        request: observation.PermissionObservationRequest,
        scope: observation.PermissionOperationScope,
        parent_lease: _IssuedPermissionAuthorityLease | None = None,
    ) -> observation.AuthorityAcquireOutcome:
        with self._lock:
            if parent_lease is None and (
                self._open_lease is not None or self._open_child_lease is not None
            ):
                return _no_lease(refused=True)
            if parent_lease is not None and (
                self._open_lease is not parent_lease
                or self._open_child_lease is not None
                or not self._parent_ready_for_authz_child(parent_lease)
            ):
                return _no_lease(refused=True)
            try:
                observation.validate_permission_observation_request(
                    request,
                    expected_request=request,
                )
                validate_permission_operation_scope(scope, request=request)
            except Exception:
                return _no_lease(refused=False)
            evidence_key = self._operation_evidence_key(request, scope)
            if evidence_key not in self._operation_evidence:
                self._record_operation_evidence(
                    request,
                    scope,
                    self._empty_operation_evidence(
                        scope,
                        parent_reservation_id=(
                            parent_lease._reservation.reservation_id
                            if parent_lease is not None
                            else None
                        ),
                    ),
                )
            if (scope.operation == "observe_authz") != (parent_lease is not None):
                return _no_lease(refused=True)
            if parent_lease is not None and (
                request != parent_lease._grant.expected_request
                or scope.request_self_sha256 != parent_lease._request_self_sha256
                or scope.access_expectation_self_sha256
                != parent_lease._access_expectation_self_sha256
                or scope.destination_self_sha256 is not None
                or scope.store_expectation_self_sha256 is not None
            ):
                return _no_lease(refused=True)
            try:
                resolution = self._resolver.resolve(request=request, scope=scope)
            except Exception:
                return _no_lease(refused=False)
            if parent_lease is not None and not self._parent_ready_for_authz_child(
                parent_lease
            ):
                return _no_lease(refused=False)
            if type(resolution) is not PermissionApprovalResolution:
                return _no_lease(refused=False)
            if resolution.status not in _RESOLUTION_STATUSES:
                return _no_lease(refused=False)
            if resolution.status != "approved":
                if resolution.grant is not None:
                    return _no_lease(refused=False)
                return _no_lease(refused=resolution.status != "failed")
            if type(resolution.grant) is not PermissionApprovalGrant:
                return _no_lease(refused=False)
            grant = resolution.grant
            try:
                _validate_grant_self(grant)
                _validate_grant_correspondence(
                    grant,
                    request=request,
                    scope=scope,
                )
            except _GrantCorrespondenceError:
                return _no_lease(refused=True)
            except Exception:
                return _no_lease(refused=False)
            if parent_lease is not None and (
                grant.expected_context != parent_lease._grant.expected_context
                or grant.expected_source_binding
                != parent_lease._grant.expected_source_binding
                or grant.state4_authorization_file_sha256
                != parent_lease._grant.state4_authorization_file_sha256
                or grant.expected_request.attempt_id
                != parent_lease._grant.expected_request.attempt_id
                or grant.expected_access_expectation
                != parent_lease._grant.expected_access_expectation
            ):
                return _no_lease(refused=True)
            a4 = grant.state4_authorization_file_sha256
            if a4 in self._blocked_authorizations:
                return _no_lease(refused=True)
            context_binding = _grant_context_binding_sha256(grant)
            if self._session is None:
                if a4 in self._attempted_activations:
                    return _no_lease(refused=True)
                activation = _attempt_activation(grant)
                self._attempted_activations.add(a4)
                try:
                    _validate_attempt_activation(activation)
                    acknowledged = self._journal.activate_attempt(activation)
                    validate_attempt_activation_acknowledgment(
                        acknowledged,
                        expected_activation=activation,
                    )
                except Exception:
                    self._blocked_authorizations.add(a4)
                    return _no_lease(refused=False)
                if acknowledged.status != "activated":
                    self._blocked_authorizations.add(a4)
                    return _no_lease(refused=acknowledged.status != "failed")
                assert acknowledged.session_id is not None
                self._session = _ActiveSession(activation, acknowledged.session_id)
            elif (
                self._session.activation.state4_authorization_file_sha256 != a4
                or self._session.activation.attempt_id != grant.expected_request.attempt_id
                or self._session.activation.grant_context_binding_sha256 != context_binding
            ):
                return _no_lease(refused=True)
            grant_binding = permission_approval_grant_binding_sha256(grant)
            operation_snapshot = self._operation_evidence[evidence_key]
            operation_snapshot = replace(
                operation_snapshot,
                grant_binding_sha256=grant_binding,
            )
            self._record_operation_evidence(
                request, scope, operation_snapshot
            )
            reservation = _operation_reservation(
                session=self._session,
                grant=grant,
                grant_binding=grant_binding,
            )
            allowance_id = reservation.allowance_id
            if allowance_id in self._attempted_allowances:
                return _no_lease(refused=True)
            self._attempted_allowances.add(allowance_id)
            operation_snapshot = replace(
                operation_snapshot,
                reservation_call_state="not_called",
                reservation=reservation,
                reservation_acknowledgment=None,
            )
            self._record_operation_evidence(
                request, scope, operation_snapshot
            )
            try:
                _validate_operation_reservation(reservation)
            except Exception:
                return _no_lease(refused=False)
            try:
                reserved = self._journal.reserve_operation(reservation)
            except Exception:
                self._record_operation_evidence(
                    request,
                    scope,
                    replace(
                        operation_snapshot,
                        reservation_call_state="raised",
                    ),
                )
                return _no_lease(refused=False)
            try:
                validate_operation_reservation_acknowledgment(
                    reserved,
                    expected_reservation=reservation,
                )
            except Exception:
                self._record_operation_evidence(
                    request,
                    scope,
                    replace(
                        operation_snapshot,
                        reservation_call_state="returned_malformed",
                    ),
                )
                return _no_lease(refused=False)
            operation_snapshot = replace(
                operation_snapshot,
                reservation_call_state="returned_valid",
                reservation_acknowledgment=reserved,
            )
            self._record_operation_evidence(
                request, scope, operation_snapshot
            )
            if reserved.status != "reserved":
                return _no_lease(refused=reserved.status != "failed")
            lease = _IssuedPermissionAuthorityLease(
                issuer=self,
                grant=grant,
                grant_binding_sha256=grant_binding,
                reservation=reservation,
                parent_lease=parent_lease,
            )
            if parent_lease is not None and not self._parent_ready_for_authz_child(
                parent_lease
            ):
                lease._inactive = True
                lease._active_failure_refused = False
                self._open_child_lease = lease
                lease.finish("failed")
                return _no_lease(refused=False)
            if parent_lease is None:
                self._open_lease = lease
            else:
                if not self._parent_ready_for_authz_child(parent_lease):
                    lease._inactive = True
                    lease._active_failure_refused = False
                    self._open_child_lease = lease
                    lease.finish("failed")
                    return _no_lease(refused=False)
                self._open_child_lease = lease
            return observation.AuthorityAcquireOutcome("granted", lease, None)
