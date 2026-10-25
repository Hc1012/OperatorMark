"""Injected orchestration for one private permission observation and retention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import os
from threading import RLock
from typing import Callable, Literal, cast

from .canonical import require_sha256, sha256_bytes
from . import permission_approval_resolver as approval
from . import permission_assessment as assessment
from . import permission_authority
from . import permission_authority_journal as authority_journal
from . import permission_authz_records as authz_records
from . import permission_dacl_policy as dacl
from . import permission_native_win32 as native_win32
from . import permission_observation_windows as observation
from . import permission_principal_authz_windows as authz
from . import private_permission_evidence as evidence


__all__ = [
    "EnrolledPermissionParent",
    "PermissionControllerResult",
    "PermissionDurableAuthzRuntimeResult",
    "PermissionParentActivationError",
    "PermissionParentCatalog",
    "PermissionParentEnrollmentHandle",
    "PermissionParentRuntimeError",
    "build_enrolled_permission_parent",
    "observe_and_retain_permission",
]


_ObservationResult = (
    observation.CapturedPermissionObservation
    | observation.FailedPermissionObservation
)
_RetentionResult = (
    evidence.RetainedPermissionEvidence
    | evidence.RetainedPermissionObservationFailure
    | evidence.PermissionEvidenceRetentionFailure
)
_AuthzResult = authz.SuppliedTokenAccessResult


class PermissionParentRuntimeError(RuntimeError):
    """A runtime-only parent preparation or execution invariant failed."""


class PermissionParentActivationError(PermissionParentRuntimeError):
    """Selected retention authorization could not be activated."""

    __slots__ = ("cause", "observation_result")

    def __init__(
        self,
        observation_result: _ObservationResult,
        cause: Exception,
    ) -> None:
        super().__init__("selected permission result catalog activation failed")
        self.observation_result = observation_result
        self.cause = cause


@dataclass(frozen=True, slots=True)
class PermissionParentCatalog:
    """Exact trusted catalog bytes with independently supplied identity pins."""

    catalog_bytes: bytes
    expected_file_bytes: int
    expected_file_sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedParentCatalog:
    supplied: PermissionParentCatalog
    grants: tuple[permission_authority.PermissionApprovalGrant, ...]
    grant_binding_sha256s: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparationPlan:
    mode: Literal["ordinary", "held_token_authz", "durable_held_token_authz"]
    request: observation.PermissionObservationRequest
    access_expectation: observation.PermissionAccessExpectation
    store_expectation: evidence.ConfidentialStoreExpectation
    observation_destination: evidence.PermissionEvidenceDestination
    observation_failure_destination: evidence.PermissionEvidenceDestination
    observation_catalog: _ValidatedParentCatalog
    captured_result_catalog: _ValidatedParentCatalog
    failed_result_catalog: _ValidatedParentCatalog
    authz_grant: permission_authority.PermissionApprovalGrant | None
    expected_operator_sid: observation.NativeSid | None
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding | None = None
    authz_retention_catalog: _ValidatedParentCatalog | None = None
    authz_store_expectation: evidence.ConfidentialStoreExpectation | None = None
    authz_destination: evidence.PermissionAuthzEvidenceDestination | None = None


@dataclass(frozen=True, slots=True)
class _PreparedObservation:
    plan: _PreparationPlan
    observation_catalog_identity: approval.PermissionGrantCatalogIdentity


class _PermissionParentState:
    __slots__ = (
        "activated_catalog_identities",
        "attempted_request_sha256s",
        "enrollment",
        "enrollment_handle",
        "execution_in_progress",
        "issuer",
        "journal",
        "lock",
        "prepared_by_request_sha256",
        "request_sha256_by_sequence",
        "resolver",
        "runtime_authz_by_request_sha256",
        "runtime_durable_authz_by_request_sha256",
    )

    def __init__(
        self,
        *,
        enrollment: approval.PermissionApprovalEnrollment,
        resolver: approval.EnrolledPermissionApprovalResolver,
        enrollment_handle: approval.PermissionGrantEnrollmentHandle,
        issuer: permission_authority.PermissionAuthorityIssuer,
        journal: authority_journal.FilePermissionAuthorityJournal,
    ) -> None:
        self.enrollment = enrollment
        self.resolver = resolver
        self.enrollment_handle = enrollment_handle
        self.issuer = issuer
        self.journal = journal
        self.lock = RLock()
        self.execution_in_progress = False
        self.prepared_by_request_sha256: dict[str, _PreparedObservation] = {}
        self.request_sha256_by_sequence: dict[int, str] = {}
        self.attempted_request_sha256s: set[str] = set()
        self.runtime_authz_by_request_sha256: dict[
            str, _PermissionAuthzRuntimeResult
        ] = {}
        self.runtime_durable_authz_by_request_sha256: dict[
            str, PermissionDurableAuthzRuntimeResult
        ] = {}
        self.activated_catalog_identities: dict[
            str, approval.PermissionGrantCatalogIdentity
        ] = {}


def _parent_error(message: str, cause: Exception | None = None) -> None:
    error = PermissionParentRuntimeError(message)
    if cause is None:
        raise error
    raise error from cause


def _validated_parent_catalog(
    supplied: PermissionParentCatalog,
    *,
    enrollment: approval.PermissionApprovalEnrollment,
    name: str,
) -> _ValidatedParentCatalog:
    if type(supplied) is not PermissionParentCatalog:
        _parent_error(f"{name} catalog pin has the wrong exact type")
    if type(supplied.catalog_bytes) is not bytes:
        _parent_error(f"{name} catalog bytes must have the exact bytes type")
    if (
        type(supplied.expected_file_bytes) is not int
        or supplied.expected_file_bytes < 1
    ):
        _parent_error(f"{name} expected file bytes must be an exact positive integer")
    if type(supplied.expected_file_sha256) is not str:
        _parent_error(f"{name} expected file SHA-256 must have the exact text type")
    try:
        expected_sha256 = require_sha256(
            supplied.expected_file_sha256,
            name=f"{name} expected file SHA-256",
        )
    except Exception as exc:
        _parent_error(f"{name} expected file SHA-256 is malformed", exc)
    if len(supplied.catalog_bytes) != supplied.expected_file_bytes:
        _parent_error(f"{name} catalog byte length differs from its independent pin")
    if sha256_bytes(supplied.catalog_bytes) != expected_sha256:
        _parent_error(f"{name} catalog SHA-256 differs from its independent pin")
    try:
        grants = approval.strict_permission_grant_catalog_loads(
            supplied.catalog_bytes,
            enrollment=enrollment,
        )
        bindings = tuple(
            permission_authority.permission_approval_grant_binding_sha256(grant)
            for grant in grants
        )
    except Exception as exc:
        _parent_error(f"{name} catalog failed strict validation", exc)
    return _ValidatedParentCatalog(supplied, grants, bindings)


def _validated_authz_parent_catalog(
    supplied: PermissionParentCatalog,
    *,
    enrollment: approval.PermissionApprovalEnrollment,
    name: str,
) -> _ValidatedParentCatalog:
    if type(supplied) is not PermissionParentCatalog:
        _parent_error(f"{name} catalog pin has the wrong exact type")
    if type(supplied.catalog_bytes) is not bytes:
        _parent_error(f"{name} catalog bytes must have the exact bytes type")
    if (
        type(supplied.expected_file_bytes) is not int
        or supplied.expected_file_bytes < 1
    ):
        _parent_error(f"{name} expected file bytes must be an exact positive integer")
    try:
        expected_sha = require_sha256(
            supplied.expected_file_sha256,
            name=f"{name} expected file SHA-256",
        )
    except Exception as exc:
        _parent_error(f"{name} expected file SHA-256 is malformed", exc)
    if (
        len(supplied.catalog_bytes) != supplied.expected_file_bytes
        or sha256_bytes(supplied.catalog_bytes) != expected_sha
    ):
        _parent_error(f"{name} catalog differs from its independent pins")
    try:
        grants = approval.strict_permission_authz_retention_catalog_loads(
            supplied.catalog_bytes,
            enrollment=enrollment,
        )
        bindings = tuple(
            permission_authority.permission_approval_grant_binding_sha256(grant)
            for grant in grants
        )
    except Exception as exc:
        _parent_error(f"{name} catalog failed strict Authz validation", exc)
    return _ValidatedParentCatalog(supplied, grants, bindings)


def _grants_for_operations(
    catalog: _ValidatedParentCatalog,
    *,
    expected: frozenset[str],
    name: str,
) -> dict[str, permission_authority.PermissionApprovalGrant]:
    operations = tuple(grant.expected_scope.operation for grant in catalog.grants)
    if len(operations) != len(expected) or frozenset(operations) != expected:
        _parent_error(f"{name} catalog has the wrong exact operation set")
    return {
        grant.expected_scope.operation: grant
        for grant in catalog.grants
    }


def _retention_branch_expectations(
    primary: permission_authority.PermissionApprovalGrant,
    sidecar: permission_authority.PermissionApprovalGrant,
    *,
    payload_kind: Literal["observation", "observation_failure"],
    name: str,
) -> tuple[
    evidence.ConfidentialStoreExpectation,
    evidence.PermissionEvidenceDestination,
]:
    store = primary.expected_store_expectation
    destination = primary.expected_destination
    if (
        type(store) is not evidence.ConfidentialStoreExpectation
        or type(destination) is not evidence.PermissionEvidenceDestination
        or sidecar.expected_store_expectation != store
        or sidecar.expected_destination != destination
    ):
        _parent_error(f"{name} catalog does not bind one exact destination and store")
    if destination.payload_kind != payload_kind:
        _parent_error(f"{name} catalog payload kind mismatch")
    return store, destination


def _preparation_plan(
    *,
    enrollment: approval.PermissionApprovalEnrollment,
    observation_catalog: PermissionParentCatalog,
    captured_result_catalog: PermissionParentCatalog,
    failed_result_catalog: PermissionParentCatalog,
) -> _PreparationPlan:
    observe_validated = _validated_parent_catalog(
        observation_catalog,
        enrollment=enrollment,
        name="observation",
    )
    captured_validated = _validated_parent_catalog(
        captured_result_catalog,
        enrollment=enrollment,
        name="captured-result",
    )
    failed_validated = _validated_parent_catalog(
        failed_result_catalog,
        enrollment=enrollment,
        name="failed-result",
    )
    observe_grants = _grants_for_operations(
        observe_validated,
        expected=frozenset({"observe"}),
        name="observation",
    )
    captured_grants = _grants_for_operations(
        captured_validated,
        expected=frozenset({"retain_observation", "retain_retention_failure"}),
        name="captured-result",
    )
    failed_grants = _grants_for_operations(
        failed_validated,
        expected=frozenset(
            {"retain_observation_failure", "retain_retention_failure"}
        ),
        name="failed-result",
    )
    observe_grant = observe_grants["observe"]
    all_grants = (
        *observe_validated.grants,
        *captured_validated.grants,
        *failed_validated.grants,
    )
    for grant in all_grants:
        if (
            grant.expected_context != enrollment.expected_context
            or grant.expected_source_binding != enrollment.expected_source_binding
            or grant.expected_request != observe_grant.expected_request
            or grant.expected_access_expectation
            != observe_grant.expected_access_expectation
        ):
            _parent_error(
                "parent catalogs do not share one enrolled context, source, request, and access expectation"
            )
    captured_store, captured_destination = _retention_branch_expectations(
        captured_grants["retain_observation"],
        captured_grants["retain_retention_failure"],
        payload_kind="observation",
        name="captured-result",
    )
    failed_store, failed_destination = _retention_branch_expectations(
        failed_grants["retain_observation_failure"],
        failed_grants["retain_retention_failure"],
        payload_kind="observation_failure",
        name="failed-result",
    )
    if captured_store != failed_store:
        _parent_error("retention alternatives do not share one store expectation")
    if captured_destination == failed_destination:
        _parent_error("retention alternatives do not have distinct destinations")
    if (
        captured_destination.primary_creating_relative_path
        != failed_destination.primary_creating_relative_path
        or captured_destination.primary_accepted_relative_path
        != failed_destination.primary_accepted_relative_path
    ):
        _parent_error("retention alternatives do not share the existing primary slot")
    return _PreparationPlan(
        mode="ordinary",
        request=observe_grant.expected_request,
        access_expectation=observe_grant.expected_access_expectation,
        store_expectation=captured_store,
        observation_destination=captured_destination,
        observation_failure_destination=failed_destination,
        observation_catalog=observe_validated,
        captured_result_catalog=captured_validated,
        failed_result_catalog=failed_validated,
        authz_grant=None,
        expected_operator_sid=None,
    )


def _authz_preparation_plan(
    *,
    enrollment: approval.PermissionApprovalEnrollment,
    observation_catalog: PermissionParentCatalog,
    captured_result_catalog: PermissionParentCatalog,
    failed_result_catalog: PermissionParentCatalog,
    expected_operator_sid: observation.NativeSid,
) -> _PreparationPlan:
    try:
        operator_sid = dacl._validate_operator_sid(expected_operator_sid)
    except Exception as exc:
        _parent_error("expected operator SID failed strict validation", exc)
    if operator_sid != expected_operator_sid:
        _parent_error("expected operator SID correspondence failed")

    observe_validated = _validated_parent_catalog(
        observation_catalog,
        enrollment=enrollment,
        name="observation-with-authz",
    )
    captured_validated = _validated_parent_catalog(
        captured_result_catalog,
        enrollment=enrollment,
        name="captured-result",
    )
    failed_validated = _validated_parent_catalog(
        failed_result_catalog,
        enrollment=enrollment,
        name="failed-result",
    )
    observe_grants = _grants_for_operations(
        observe_validated,
        expected=frozenset({"observe", "observe_authz"}),
        name="observation-with-authz",
    )
    captured_grants = _grants_for_operations(
        captured_validated,
        expected=frozenset({"retain_observation", "retain_retention_failure"}),
        name="captured-result",
    )
    failed_grants = _grants_for_operations(
        failed_validated,
        expected=frozenset(
            {"retain_observation_failure", "retain_retention_failure"}
        ),
        name="failed-result",
    )
    observe_grant = observe_grants["observe"]
    authz_grant = observe_grants["observe_authz"]
    all_grants = (
        *observe_validated.grants,
        *captured_validated.grants,
        *failed_validated.grants,
    )
    for grant in all_grants:
        if (
            grant.expected_context != enrollment.expected_context
            or grant.expected_source_binding != enrollment.expected_source_binding
            or grant.expected_request != observe_grant.expected_request
            or grant.expected_access_expectation
            != observe_grant.expected_access_expectation
        ):
            _parent_error(
                "parent catalogs do not share one enrolled context, source, request, and access expectation"
            )
    captured_store, captured_destination = _retention_branch_expectations(
        captured_grants["retain_observation"],
        captured_grants["retain_retention_failure"],
        payload_kind="observation",
        name="captured-result",
    )
    failed_store, failed_destination = _retention_branch_expectations(
        failed_grants["retain_observation_failure"],
        failed_grants["retain_retention_failure"],
        payload_kind="observation_failure",
        name="failed-result",
    )
    if captured_store != failed_store:
        _parent_error("retention alternatives do not share one store expectation")
    if captured_destination == failed_destination:
        _parent_error("retention alternatives do not have distinct destinations")
    if (
        captured_destination.primary_creating_relative_path
        != failed_destination.primary_creating_relative_path
        or captured_destination.primary_accepted_relative_path
        != failed_destination.primary_accepted_relative_path
    ):
        _parent_error("retention alternatives do not share the existing primary slot")
    return _PreparationPlan(
        mode="held_token_authz",
        request=observe_grant.expected_request,
        access_expectation=observe_grant.expected_access_expectation,
        store_expectation=captured_store,
        observation_destination=captured_destination,
        observation_failure_destination=failed_destination,
        observation_catalog=observe_validated,
        captured_result_catalog=captured_validated,
        failed_result_catalog=failed_validated,
        authz_grant=authz_grant,
        expected_operator_sid=operator_sid,
    )


def _durable_authz_preparation_plan(
    *,
    enrollment: approval.PermissionApprovalEnrollment,
    observation_catalog: PermissionParentCatalog,
    captured_result_catalog: PermissionParentCatalog,
    failed_result_catalog: PermissionParentCatalog,
    expected_operator_sid: observation.NativeSid,
    expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding,
    authz_retention_catalog: PermissionParentCatalog,
) -> _PreparationPlan:
    base = _authz_preparation_plan(
        enrollment=enrollment,
        observation_catalog=observation_catalog,
        captured_result_catalog=captured_result_catalog,
        failed_result_catalog=failed_result_catalog,
        expected_operator_sid=expected_operator_sid,
    )
    try:
        authz_records.validate_permission_authz_evidence_binding(
            expected_authz_binding,
            expected_request=base.request,
            expected_access=base.access_expectation,
        )
    except Exception as exc:
        _parent_error("durable Authz evidence binding failed strict validation", exc)
    if expected_authz_binding.expected_operator_sid != base.expected_operator_sid:
        _parent_error("durable Authz binding operator differs from preparation")
    retained = _validated_authz_parent_catalog(
        authz_retention_catalog,
        enrollment=enrollment,
        name="Authz-retention",
    )
    retained_grants = _grants_for_operations(
        retained,
        expected=frozenset(
            {"retain_authz_evidence", "retain_authz_retention_failure"}
        ),
        name="Authz-retention",
    )
    primary = retained_grants["retain_authz_evidence"]
    sidecar = retained_grants["retain_authz_retention_failure"]
    store = primary.expected_store_expectation
    destination = primary.expected_destination
    if (
        type(store) is not evidence.ConfidentialStoreExpectation
        or type(destination) is not evidence.PermissionAuthzEvidenceDestination
        or sidecar.expected_store_expectation != store
        or sidecar.expected_destination != destination
    ):
        _parent_error(
            "Authz retention catalog does not bind one exact destination and store"
        )
    for grant in retained.grants:
        if (
            grant.expected_context != enrollment.expected_context
            or grant.expected_source_binding != enrollment.expected_source_binding
            or grant.expected_request != base.request
            or grant.expected_access_expectation != base.access_expectation
        ):
            _parent_error(
                "Authz retention catalog differs from the prepared observation context"
            )
    if store != base.store_expectation:
        _parent_error(
            "Authz retention store differs from the existing base-retention store"
        )
    try:
        evidence.validate_permission_authz_evidence_destination(
            destination,
            expected_request=base.request,
            expected_store=store,
            expected_authz_binding=expected_authz_binding,
        )
    except Exception as exc:
        _parent_error("Authz retention destination failed binding validation", exc)
    return replace(
        base,
        mode="durable_held_token_authz",
        expected_authz_binding=expected_authz_binding,
        authz_retention_catalog=retained,
        authz_store_expectation=store,
        authz_destination=destination,
    )


@dataclass(frozen=True, slots=True)
class PermissionControllerResult:
    """Runtime-only result for one observation and its selected retention path."""

    observation_result: _ObservationResult
    retention_result: _RetentionResult
    acceptance_emitted: Literal[False] = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class _AuthzChildAccounting:
    acquire_status: Literal["not_attempted", "granted", "refused", "failed"]
    lease_candidate_returned: bool
    acquire_failure: observation.ComponentFailureItem | None
    lease_binding_status: Literal["not_applicable", "valid", "invalid"]
    checkpoint_outcomes: tuple[observation.AuthorityCheckOutcome, ...]
    finish_attempted: bool
    requested_outcome: Literal["succeeded", "refused", "failed"] | None
    finish_outcome: observation.AuthorityFinishOutcome | None


@dataclass(frozen=True, slots=True)
class _HeldTokenAuthzRuntimeEvidence:
    status: Literal["not_invoked", "incomplete", "completed"]
    result_prefix: tuple[_AuthzResult, ...]
    expected_token_identity: authz.SuppliedTokenIdentity | None
    failure_reason_codes: tuple[str, ...]
    child_accounting: _AuthzChildAccounting
    permission_acceptance_emitted: Literal[False] = field(
        default=False, init=False
    )


@dataclass(frozen=True, slots=True)
class _PermissionAuthzRuntimeResult:
    observation_result: _ObservationResult
    retention_result: _RetentionResult | None
    retained_observation_bytes: bytes | None
    authz_evidence: _HeldTokenAuthzRuntimeEvidence
    controller_failure_reason: str | None
    permission_acceptance_emitted: Literal[False] = field(
        default=False, init=False
    )


@dataclass(frozen=True, slots=True)
class PermissionDurableAuthzRuntimeResult:
    base_runtime: _PermissionAuthzRuntimeResult
    authz_record_bytes: bytes | None
    persistence_status: Literal[
        "not_attempted", "not_retained", "retained", "retention_unknown"
    ]
    retention_result: (
        evidence.RetainedPermissionAuthzEvidence
        | evidence.PermissionAuthzEvidenceRetentionFailure
        | None
    )
    failure_reason: Literal[
        "authz_bundle_build_failed",
        "authz_retention_catalog_activation_failed",
        "authz_retention_integration_failed",
    ] | None
    permission_acceptance_emitted: Literal[False] = field(
        default=False, init=False
    )


@dataclass(frozen=True, slots=True)
class _ExecutedAuthzObservation:
    runtime: _PermissionAuthzRuntimeResult
    hook: _ControllerHeldTokenAuthzHook
    base_record_bytes: bytes | None


@dataclass(frozen=True, slots=True)
class _PermissionAuthzAssessmentPhaseResult:
    status: Literal["completed", "incomplete"]
    assessment: assessment.PermissionBundleAssessment | None
    reason_code: str
    permission_acceptance_emitted: Literal[False] = field(
        default=False, init=False
    )


def _empty_authz_child_accounting() -> _AuthzChildAccounting:
    return _AuthzChildAccounting(
        acquire_status="not_attempted",
        lease_candidate_returned=False,
        acquire_failure=None,
        lease_binding_status="not_applicable",
        checkpoint_outcomes=(),
        finish_attempted=False,
        requested_outcome=None,
        finish_outcome=None,
    )


def _not_invoked_authz_evidence(reason: str) -> _HeldTokenAuthzRuntimeEvidence:
    return _HeldTokenAuthzRuntimeEvidence(
        status="not_invoked",
        result_prefix=(),
        expected_token_identity=None,
        failure_reason_codes=(reason,),
        child_accounting=_empty_authz_child_accounting(),
    )


def _validated_authz_acquire_failure_item(
    value: object,
) -> observation.ComponentFailureItem | None:
    try:
        return observation._validate_failure_item(value)
    except Exception:
        return None


def _validated_authz_checkpoint_failure_item(
    value: object,
) -> observation.ComponentFailureItem | None:
    if type(value) is not observation.ComponentFailureItem:
        return None
    failure = cast(observation.ComponentFailureItem, value)
    if any(
        type(item) is not str
        for item in (
            failure.failure_kind,
            failure.stage,
            failure.component_operation,
            failure.reason_code,
        )
    ):
        return None
    if (
        failure.resource_kind is not None
        or failure.resource_role is not None
        or failure.resource_ordinal is not None
        or failure.backend_error_hex is not None
    ):
        return None
    if failure.operation_ordinal is not None and (
        type(failure.operation_ordinal) is not int
        or failure.operation_ordinal not in (0, 1)
    ):
        return None
    return failure


def _authz_acquire_failure_corresponds(
    failure: observation.ComponentFailureItem,
    *,
    status: Literal["refused", "failed"],
) -> bool:
    return (
        failure.failure_kind
        == ("authority_refusal" if status == "refused" else "authority_failure")
        and failure.stage == "authority_acquisition"
        and failure.component_operation == "acquire_authority_lease"
        and failure.resource_kind is None
        and failure.resource_role is None
        and failure.resource_ordinal is None
        and failure.operation_ordinal is None
        and failure.backend_error_hex is None
        and failure.reason_code
        == ("authority_acquire_refused" if status == "refused" else "authority_acquire_failed")
    )


def _checked_authz_checkpoint(
    lease: observation.PermissionAuthorityLease,
    checkpoint: observation.PermissionAuthorityCheckpoint,
) -> tuple[
    observation.AuthorityCheckOutcome | None,
    str | None,
    Literal["refused", "failed"] | None,
]:
    try:
        outcome = lease.require_active(checkpoint)
    except Exception:
        return None, "child_authority_check_exception", "failed"
    try:
        if type(outcome) is not observation.AuthorityCheckOutcome:
            raise ValueError("child authority check has the wrong exact type")
        if type(outcome.status) is not str or outcome.status not in (
            "active",
            "refused",
            "failed",
        ):
            raise ValueError("child authority check status is malformed")
        echoed = outcome.checkpoint
        if (
            type(echoed) is not observation.PermissionAuthorityCheckpoint
            or type(echoed.operation) is not str
            or type(echoed.step) is not str
            or echoed.operation != checkpoint.operation
            or echoed.step != checkpoint.step
            or type(echoed.ordinal) is not type(checkpoint.ordinal)
            or echoed.ordinal != checkpoint.ordinal
        ):
            raise ValueError("child authority check checkpoint is malformed")
        if outcome.status == "active":
            if outcome.failure is not None:
                raise ValueError("active child authority check returned a failure")
            return outcome, None, None
        failure = _validated_authz_checkpoint_failure_item(outcome.failure)
        if failure is None:
            raise ValueError("inactive child authority check lacks an exact failure")
        requested: Literal["refused", "failed"] = (
            "refused" if outcome.status == "refused" else "failed"
        )
        if (
            failure.failure_kind
            != (
                "authority_refusal"
                if requested == "refused"
                else "authority_failure"
            )
            or failure.stage != permission_authority._checkpoint_stage(checkpoint)
            or failure.component_operation != "require_active"
            or failure.resource_kind is not None
            or failure.resource_role is not None
            or failure.resource_ordinal is not None
            or type(failure.operation_ordinal) is not type(checkpoint.ordinal)
            or failure.operation_ordinal != checkpoint.ordinal
            or failure.backend_error_hex is not None
            or failure.reason_code
            != (
                "authority_inactive"
                if requested == "refused"
                else "authority_check_failed"
            )
        ):
            raise ValueError("inactive child authority failure is contradictory")
        return outcome, "child_authority_inactive", requested
    except Exception:
        return None, "child_authority_check_malformed", "failed"


class _ControllerHeldTokenAuthzHook:
    __slots__ = (
        "_acquire_attempted",
        "_authz_bindings",
        "_expected_identity",
        "_grant",
        "_held_inputs",
        "_issuer",
        "_native",
        "_was_called",
    )

    def __init__(
        self,
        *,
        issuer: permission_authority.PermissionAuthorityIssuer,
        grant: permission_authority.PermissionApprovalGrant,
        native: native_win32.Win32PermissionNativeApi,
        authz_bindings: authz.AuthzNativeBindings,
    ) -> None:
        self._issuer = issuer
        self._grant = grant
        self._native = native
        self._authz_bindings = authz_bindings
        self._was_called = False
        self._acquire_attempted = False
        self._expected_identity: authz.SuppliedTokenIdentity | None = None
        self._held_inputs: authz_records.PermissionAuthzHeldInputs | None = None

    def __call__(
        self,
        *,
        held_duplicate_token: observation.NativeTokenHandle,
        duplicate_token_before_facts: observation.NativeTokenFacts,
        descriptors: tuple[
            observation.NativeSecurityDescriptor,
            observation.NativeSecurityDescriptor,
        ],
        object_bindings: tuple[
            observation.NativeAccessObjectBinding,
            observation.NativeAccessObjectBinding,
        ],
        observation_lease: observation.PermissionAuthorityLease,
    ) -> _HeldTokenAuthzRuntimeEvidence:
        self._was_called = True
        self._acquire_attempted = False
        self._expected_identity = None
        self._held_inputs = None
        try:
            return self._run(
                held_duplicate_token=held_duplicate_token,
                duplicate_token_before_facts=duplicate_token_before_facts,
                descriptors=descriptors,
                object_bindings=object_bindings,
                observation_lease=observation_lease,
            )
        except Exception:
            return self.failure_evidence()

    def failure_evidence(self) -> _HeldTokenAuthzRuntimeEvidence:
        if not self._was_called:
            return _not_invoked_authz_evidence("authz_hook_not_reached")
        accounting = (
            _AuthzChildAccounting(
                "failed", False, None, "not_applicable", (), False, None, None
            )
            if self._acquire_attempted
            else _empty_authz_child_accounting()
        )
        return _HeldTokenAuthzRuntimeEvidence(
            status="incomplete",
            result_prefix=(),
            expected_token_identity=self._expected_identity,
            failure_reason_codes=("authz_hook_failed",),
            child_accounting=accounting,
        )

    def held_inputs_snapshot(
        self,
    ) -> authz_records.PermissionAuthzHeldInputs | None:
        """Return the already validated immutable hook inputs, never the handle."""

        return self._held_inputs

    def _run(
        self,
        *,
        held_duplicate_token: observation.NativeTokenHandle,
        duplicate_token_before_facts: observation.NativeTokenFacts,
        descriptors: tuple[
            observation.NativeSecurityDescriptor,
            observation.NativeSecurityDescriptor,
        ],
        object_bindings: tuple[
            observation.NativeAccessObjectBinding,
            observation.NativeAccessObjectBinding,
        ],
        observation_lease: observation.PermissionAuthorityLease,
    ) -> _HeldTokenAuthzRuntimeEvidence:
        try:
            observation._validate_token_facts(
                duplicate_token_before_facts, duplicate=True
            )
            if type(descriptors) is not tuple or len(descriptors) != 2:
                raise ValueError("descriptor pair is malformed")
            if type(object_bindings) is not tuple or len(object_bindings) != 2:
                raise ValueError("object-binding pair is malformed")
            checked_descriptors = tuple(
                authz._validate_descriptor(item)[0] for item in descriptors
            )
            checked_bindings = tuple(
                authz._validate_binding(binding, descriptor)
                for binding, descriptor in zip(
                    object_bindings, checked_descriptors, strict=True
                )
            )
            if tuple(item.object_role for item in checked_bindings) != (
                "subject",
                "immediate_parent",
            ):
                raise ValueError("object-binding order is malformed")
            expected_paths = (
                self._grant.expected_request.subject.canonical_volume_guid_path,
                self._grant.expected_request.subject.required_ancestor_canonical_volume_guid_paths[
                    -1
                ],
            )
            if (
                tuple(item.object_kind for item in checked_bindings)
                != ("file", "directory")
                or tuple(
                    item.canonical_volume_guid_path for item in checked_bindings
                )
                != expected_paths
            ):
                raise ValueError("object-binding correspondence failed")
            expected_identity = authz._validate_identity(
                authz.SuppliedTokenIdentity(
                    token_id_luid_hex=(
                        duplicate_token_before_facts.token_id_luid_hex
                    ),
                    authentication_id_luid_hex=(
                        duplicate_token_before_facts.authentication_id_luid_hex
                    ),
                    modified_id_luid_hex=(
                        duplicate_token_before_facts.modified_id_luid_hex
                    ),
                    token_type=duplicate_token_before_facts.token_type,
                    impersonation_level=cast(
                        object, duplicate_token_before_facts.impersonation_level
                    ),
                )
            )
            self._expected_identity = expected_identity
            self._held_inputs = authz_records.PermissionAuthzHeldInputs(
                duplicate_token_before_facts=duplicate_token_before_facts,
                descriptors=cast(
                    tuple[
                        observation.NativeSecurityDescriptor,
                        observation.NativeSecurityDescriptor,
                    ],
                    checked_descriptors,
                ),
                object_bindings=cast(
                    tuple[
                        observation.NativeAccessObjectBinding,
                        observation.NativeAccessObjectBinding,
                    ],
                    checked_bindings,
                ),
            )
        except Exception:
            return _HeldTokenAuthzRuntimeEvidence(
                status="incomplete",
                result_prefix=(),
                expected_token_identity=None,
                failure_reason_codes=("authz_hook_input_rejected",),
                child_accounting=_empty_authz_child_accounting(),
            )

        scope = self._grant.expected_scope
        self._acquire_attempted = True
        try:
            acquired = self._issuer.acquire_observe_authz_child(
                request=self._grant.expected_request,
                scope=scope,
                parent_lease=observation_lease,
            )
        except Exception:
            return _HeldTokenAuthzRuntimeEvidence(
                status="incomplete",
                result_prefix=(),
                expected_token_identity=expected_identity,
                failure_reason_codes=("child_authority_acquire_exception",),
                child_accounting=_AuthzChildAccounting(
                    "failed", False, None, "not_applicable", (), False, None, None
                ),
            )
        if type(acquired) is not observation.AuthorityAcquireOutcome:
            return _HeldTokenAuthzRuntimeEvidence(
                status="incomplete",
                result_prefix=(),
                expected_token_identity=expected_identity,
                failure_reason_codes=("child_authority_acquire_malformed",),
                child_accounting=_AuthzChildAccounting(
                    "failed", False, None, "not_applicable", (), False, None, None
                ),
            )

        status = (
            acquired.status
            if type(acquired.status) is str
            and acquired.status in ("granted", "refused", "failed")
            else None
        )
        failure = (
            None
            if acquired.failure is None
            else _validated_authz_acquire_failure_item(acquired.failure)
        )
        if (
            status in ("refused", "failed")
            and acquired.lease is None
            and failure is not None
            and _authz_acquire_failure_corresponds(
                failure,
                status=cast(Literal["refused", "failed"], status),
            )
        ):
            return _HeldTokenAuthzRuntimeEvidence(
                status="incomplete",
                result_prefix=(),
                expected_token_identity=expected_identity,
                failure_reason_codes=(
                    "child_authority_refused"
                    if status == "refused"
                    else "child_authority_acquire_failed",
                ),
                child_accounting=_AuthzChildAccounting(
                    cast(Literal["refused", "failed"], status),
                    False,
                    failure,
                    "not_applicable",
                    (),
                    False,
                    None,
                    None,
                ),
            )

        lease = acquired.lease
        accounting_status = cast(
            Literal["granted", "refused", "failed"],
            status if status is not None else "failed",
        )
        if lease is None:
            return _HeldTokenAuthzRuntimeEvidence(
                status="incomplete",
                result_prefix=(),
                expected_token_identity=expected_identity,
                failure_reason_codes=("child_authority_acquire_malformed",),
                child_accounting=_AuthzChildAccounting(
                    accounting_status,
                    False,
                    failure,
                    "not_applicable",
                    (),
                    False,
                    None,
                    None,
                ),
            )
        structurally_valid, binding_reason = observation._lease_binding_reason(
            lease, scope
        )
        if not structurally_valid:
            return _HeldTokenAuthzRuntimeEvidence(
                status="incomplete",
                result_prefix=(),
                expected_token_identity=expected_identity,
                failure_reason_codes=(
                    "child_lease_binding_failed"
                    if status == "granted" and acquired.failure is None
                    else "child_authority_acquire_malformed",
                ),
                child_accounting=_AuthzChildAccounting(
                    accounting_status,
                    True,
                    failure,
                    "invalid",
                    (),
                    False,
                    None,
                    None,
                ),
            )

        typed_lease = cast(observation.PermissionAuthorityLease, lease)
        acquisition_malformed = not (
            status == "granted" and acquired.failure is None
        )
        return self._execute_bound_child(
            lease=typed_lease,
            held_duplicate_token=held_duplicate_token,
            checked_descriptors=checked_descriptors,
            checked_bindings=checked_bindings,
            expected_identity=expected_identity,
            acquire_status=accounting_status,
            acquire_failure=failure,
            lease_binding_status=(
                "valid" if binding_reason == "none" else "invalid"
            ),
            initial_reason=(
                "child_authority_acquire_malformed"
                if acquisition_malformed
                else "child_lease_binding_failed"
                if binding_reason != "none"
                else None
            ),
        )

    def _execute_bound_child(
        self,
        *,
        lease: observation.PermissionAuthorityLease,
        held_duplicate_token: observation.NativeTokenHandle,
        checked_descriptors: tuple[
            observation.NativeSecurityDescriptor,
            observation.NativeSecurityDescriptor,
        ],
        checked_bindings: tuple[
            observation.NativeAccessObjectBinding,
            observation.NativeAccessObjectBinding,
        ],
        expected_identity: authz.SuppliedTokenIdentity,
        acquire_status: Literal["granted", "refused", "failed"],
        acquire_failure: observation.ComponentFailureItem | None,
        lease_binding_status: Literal["valid", "invalid"],
        initial_reason: str | None,
    ) -> _HeldTokenAuthzRuntimeEvidence:
        checkpoint_outcomes: list[observation.AuthorityCheckOutcome] = []
        results: list[_AuthzResult] = []
        reasons: list[str] = [] if initial_reason is None else [initial_reason]
        requested: Literal["succeeded", "refused", "failed"] = "failed"

        def check(step: str, ordinal: int | None) -> bool:
            nonlocal requested
            checkpoint = observation.PermissionAuthorityCheckpoint(
                "observe_authz", step, ordinal
            )
            outcome, reason, refused_or_failed = _checked_authz_checkpoint(
                lease, checkpoint
            )
            if outcome is not None:
                checkpoint_outcomes.append(outcome)
            if reason is None:
                return True
            reasons.append(reason)
            requested = cast(
                Literal["refused", "failed"], refused_or_failed
            )
            return False

        try:
            if not reasons and check("initial", None):
                for ordinal, (descriptor, binding) in enumerate(
                    zip(checked_descriptors, checked_bindings, strict=True)
                ):
                    if not check("authz_access_check", ordinal):
                        break
                    try:
                        candidate = self._native.evaluate_held_token_authz(
                            cast(
                                native_win32.Win32OwnedTokenHandle,
                                held_duplicate_token,
                            ),
                            expected_token_identity=expected_identity,
                            descriptor=descriptor,
                            object_binding=binding,
                            authz_bindings=self._authz_bindings,
                        )
                    except Exception:
                        reasons.append("authz_adapter_exception")
                        break
                    try:
                        checked_result = assessment._validate_authz_result(
                            candidate,
                            expected_identity=expected_identity,
                            expected_descriptor=descriptor,
                            expected_binding=binding,
                        )
                    except Exception:
                        reasons.append("authz_result_malformed")
                        break
                    results.append(checked_result)
                    if checked_result.access_status == "unavailable":
                        reasons.append("authz_result_unavailable")
                        break

            if not reasons and len(results) == 2 and check("final", None):
                requested = "succeeded"
        except Exception:
            requested = "failed"
            reasons.append("authz_integration_exception")
        finally:
            finish_outcome = observation._finish(lease, requested)
        if finish_outcome.finish_status != "succeeded":
            reasons.append("child_terminal_incomplete")
        completed = not reasons and requested == "succeeded"
        return _HeldTokenAuthzRuntimeEvidence(
            status="completed" if completed else "incomplete",
            result_prefix=tuple(results),
            expected_token_identity=expected_identity,
            failure_reason_codes=tuple(reasons),
            child_accounting=_AuthzChildAccounting(
                acquire_status,
                True,
                acquire_failure,
                lease_binding_status,
                tuple(checkpoint_outcomes),
                True,
                requested,
                finish_outcome,
            ),
        )


def _observe_and_retain_permission(
    request: observation.PermissionObservationRequest,
    *,
    expected_request: observation.PermissionObservationRequest,
    access_expectation: observation.PermissionAccessExpectation,
    confidential_store_expectation: evidence.ConfidentialStoreExpectation,
    observation_destination: evidence.PermissionEvidenceDestination,
    observation_failure_destination: evidence.PermissionEvidenceDestination,
    authority: observation.PermissionObservationAuthority,
    native: observation.PermissionNativeApi,
    observation_store: evidence.ConfidentialEvidenceStore,
    observation_failure_store: evidence.ConfidentialEvidenceStore,
    before_retention: Callable[[_ObservationResult], None] | None,
) -> PermissionControllerResult:
    """Observe exactly once and retain only the resulting primary payload kind."""

    observation.validate_permission_observation_request(
        request, expected_request=expected_request
    )
    observation.validate_permission_access_expectation(
        access_expectation, expected_request=expected_request
    )
    evidence.validate_confidential_store_expectation(
        confidential_store_expectation, expected_request=expected_request
    )
    evidence.validate_permission_evidence_destination(
        observation_destination,
        expected_request=expected_request,
        expected_store=confidential_store_expectation,
    )
    evidence.validate_permission_evidence_destination(
        observation_failure_destination,
        expected_request=expected_request,
        expected_store=confidential_store_expectation,
    )
    if observation_destination.payload_kind != "observation":
        raise evidence.PrivatePermissionEvidenceContractError(
            "observation destination payload kind mismatch"
        )
    if observation_failure_destination.payload_kind != "observation_failure":
        raise evidence.PrivatePermissionEvidenceContractError(
            "observation-failure destination payload kind mismatch"
        )
    if (
        observation_destination.primary_creating_relative_path
        != observation_failure_destination.primary_creating_relative_path
        or observation_destination.primary_accepted_relative_path
        != observation_failure_destination.primary_accepted_relative_path
    ):
        raise evidence.PrivatePermissionEvidenceContractError(
            "primary destination slot naming mismatch"
        )

    observed = observation.observe_held_file_permission(
        request,
        expected_request=expected_request,
        access_expectation=access_expectation,
        authority=authority,
        native=native,
    )
    if type(observed) is observation.CapturedPermissionObservation:
        capture = observed
        record = evidence.build_private_permission_observation_record(
            capture,
            expected_request=expected_request,
            expected_access=access_expectation,
        )
        if before_retention is not None:
            before_retention(capture)
        retained = evidence.retain_private_permission_observation_exclusive(
            observation_destination,
            record,
            expected_request=expected_request,
            expected_access=access_expectation,
            expected_store=confidential_store_expectation,
            authority=authority,
            store=observation_store,
        )
        return PermissionControllerResult(capture, retained)

    if type(observed) is observation.FailedPermissionObservation:
        failure = observed
        record = evidence.build_private_permission_observation_failure_record(
            failure,
            expected_request=expected_request,
            expected_access=access_expectation,
        )
        if before_retention is not None:
            before_retention(failure)
        retained = evidence.retain_private_permission_observation_failure_exclusive(
            observation_failure_destination,
            record,
            expected_request=expected_request,
            expected_access=access_expectation,
            expected_store=confidential_store_expectation,
            authority=authority,
            store=observation_failure_store,
        )
        return PermissionControllerResult(failure, retained)

    raise observation.PermissionObservationProtocolError(
        "observer returned a malformed result"
    )


def _execute_prepared_authz_observation(
    request: observation.PermissionObservationRequest,
    *,
    plan: _PreparationPlan,
    authority: permission_authority.PermissionAuthorityIssuer,
    native: native_win32.Win32PermissionNativeApi,
    authz_bindings: authz.AuthzNativeBindings,
    observation_store: evidence.ConfidentialEvidenceStore,
    observation_failure_store: evidence.ConfidentialEvidenceStore,
    before_retention: Callable[[_ObservationResult], None],
) -> _ExecutedAuthzObservation:
    if plan.mode not in {
        "held_token_authz", "durable_held_token_authz"
    } or plan.authz_grant is None:
        _parent_error("held-token Authz execution used an ordinary preparation")
    hook = _ControllerHeldTokenAuthzHook(
        issuer=authority,
        grant=plan.authz_grant,
        native=native,
        authz_bindings=authz_bindings,
    )
    hooked = observation._observe_held_file_permission_with_authz_hook(
        request,
        expected_request=plan.request,
        access_expectation=plan.access_expectation,
        authority=authority,
        native=native,
        authz_hook=hook,
    )
    observed = hooked.observation_result
    if (
        hooked.hook_status == "returned"
        and type(hooked.hook_value) is _HeldTokenAuthzRuntimeEvidence
        and hooked.hook_failure is None
    ):
        authz_evidence = hooked.hook_value
    elif hooked.hook_status == "failed" and type(
        hooked.hook_failure
    ) is observation.ComponentFailureItem:
        authz_evidence = hook.failure_evidence()
    elif hooked.hook_status == "not_invoked":
        authz_evidence = _not_invoked_authz_evidence("authz_hook_not_reached")
    else:
        fallback = hook.failure_evidence()
        authz_evidence = _HeldTokenAuthzRuntimeEvidence(
            status="incomplete",
            result_prefix=fallback.result_prefix,
            expected_token_identity=fallback.expected_token_identity,
            failure_reason_codes=("authz_hook_result_malformed",),
            child_accounting=fallback.child_accounting,
        )

    retained_bytes: bytes | None = None
    base_record_bytes: bytes | None = None
    try:
        if type(observed) is observation.CapturedPermissionObservation:
            record = evidence.build_private_permission_observation_record(
                observed,
                expected_request=plan.request,
                expected_access=plan.access_expectation,
            )
            retained_bytes = evidence.canonical_private_permission_observation_bytes(
                record,
                expected_request=plan.request,
                expected_access=plan.access_expectation,
            )
            base_record_bytes = retained_bytes
            before_retention(observed)
            retained = evidence.retain_private_permission_observation_exclusive(
                plan.observation_destination,
                record,
                expected_request=plan.request,
                expected_access=plan.access_expectation,
                expected_store=plan.store_expectation,
                authority=authority,
                store=observation_store,
            )
        elif type(observed) is observation.FailedPermissionObservation:
            record = evidence.build_private_permission_observation_failure_record(
                observed,
                expected_request=plan.request,
                expected_access=plan.access_expectation,
            )
            base_record_bytes = (
                evidence.canonical_private_permission_observation_failure_bytes(
                    record,
                    expected_request=plan.request,
                    expected_access=plan.access_expectation,
                )
            )
            before_retention(observed)
            retained = evidence.retain_private_permission_observation_failure_exclusive(
                plan.observation_failure_destination,
                record,
                expected_request=plan.request,
                expected_access=plan.access_expectation,
                expected_store=plan.store_expectation,
                authority=authority,
                store=observation_failure_store,
            )
        else:
            raise observation.PermissionObservationProtocolError(
                "observer returned a malformed result"
            )
    except PermissionParentActivationError:
        return _ExecutedAuthzObservation(
            _PermissionAuthzRuntimeResult(
                observed,
                None,
                retained_bytes,
                authz_evidence,
                "selected_result_catalog_activation_failed",
            ),
            hook,
            base_record_bytes,
        )
    except Exception:
        return _ExecutedAuthzObservation(
            _PermissionAuthzRuntimeResult(
                observed,
                None,
                retained_bytes,
                authz_evidence,
                "retention_integration_failed",
            ),
            hook,
            base_record_bytes,
        )
    return _ExecutedAuthzObservation(
        _PermissionAuthzRuntimeResult(
            observed,
            retained,
            retained_bytes,
            authz_evidence,
            None,
        ),
        hook,
        base_record_bytes,
    )


def _observe_and_retain_permission_with_authz(
    request: observation.PermissionObservationRequest,
    *,
    plan: _PreparationPlan,
    authority: permission_authority.PermissionAuthorityIssuer,
    native: native_win32.Win32PermissionNativeApi,
    authz_bindings: authz.AuthzNativeBindings,
    observation_store: evidence.ConfidentialEvidenceStore,
    observation_failure_store: evidence.ConfidentialEvidenceStore,
    before_retention: Callable[[_ObservationResult], None],
) -> _PermissionAuthzRuntimeResult:
    return _execute_prepared_authz_observation(
        request,
        plan=plan,
        authority=authority,
        native=native,
        authz_bindings=authz_bindings,
        observation_store=observation_store,
        observation_failure_store=observation_failure_store,
        before_retention=before_retention,
    ).runtime


def _operation_projection(
    snapshot: permission_authority.PermissionOperationEvidenceSnapshot,
) -> authz_records.PermissionOperationEvidenceProjection:
    return authz_records.PermissionOperationEvidenceProjection(
        operation=cast(Literal["observe", "observe_authz"], snapshot.operation),
        scope_self_sha256=snapshot.scope_self_sha256,
        grant_binding_sha256=snapshot.grant_binding_sha256,
        parent_reservation_id=snapshot.parent_reservation_id,
        reservation_call_state=snapshot.reservation_call_state,
        reservation=(
            asdict(snapshot.reservation) if snapshot.reservation is not None else None
        ),
        reservation_acknowledgment=(
            asdict(snapshot.reservation_acknowledgment)
            if snapshot.reservation_acknowledgment is not None
            else None
        ),
        terminal_call_state=snapshot.terminal_call_state,
        terminal=asdict(snapshot.terminal) if snapshot.terminal is not None else None,
        terminal_acknowledgment=(
            asdict(snapshot.terminal_acknowledgment)
            if snapshot.terminal_acknowledgment is not None
            else None
        ),
    )


def _build_authz_bundle_from_runtime(
    executed: _ExecutedAuthzObservation,
    *,
    plan: _PreparationPlan,
    issuer: permission_authority.PermissionAuthorityIssuer,
) -> bytes:
    if (
        plan.mode != "durable_held_token_authz"
        or plan.expected_authz_binding is None
        or plan.authz_destination is None
        or plan.authz_store_expectation is None
        or plan.authz_grant is None
    ):
        _parent_error("durable Authz bundle used incomplete preparation")
    runtime = executed.runtime
    base_bytes = executed.base_record_bytes
    if base_bytes is None:
        base_outcome = "unavailable"
        record_hex = None
        record_file_bytes = None
        record_file_sha256 = None
        record_unavailable_reason = "serialization_failed"
    else:
        base_outcome = (
            "captured"
            if type(runtime.observation_result)
            is observation.CapturedPermissionObservation
            else "failed"
        )
        record_hex = base_bytes.hex()
        record_file_bytes = len(base_bytes)
        record_file_sha256 = sha256_bytes(base_bytes)
        record_unavailable_reason = None
    retained = runtime.retention_result
    if type(retained) is evidence.RetainedPermissionEvidence:
        retention_kind = "captured_receipt"
    elif type(retained) is evidence.RetainedPermissionObservationFailure:
        retention_kind = "failure_receipt"
    elif type(retained) is evidence.PermissionEvidenceRetentionFailure:
        retention_kind = "retention_failure"
    else:
        retention_kind = "not_returned"
    base = authz_records.PermissionAuthzBaseObservation(
        outcome=cast(Literal["captured", "failed", "unavailable"], base_outcome),
        record_hex=record_hex,
        record_file_bytes=record_file_bytes,
        record_file_sha256=record_file_sha256,
        record_unavailable_reason=cast(
            Literal["not_returned", "malformed_result", "serialization_failed"] | None,
            record_unavailable_reason,
        ),
        retention_result_kind=cast(
            Literal[
                "captured_receipt", "failure_receipt", "retention_failure", "not_returned"
            ],
            retention_kind,
        ),
        retention_result=asdict(retained) if retained is not None else None,
        retention_result_unavailable_reason=(
            None if retained is not None else "not_returned"
        ),
    )
    child = runtime.authz_evidence.child_accounting
    wire_authz = authz_records.PermissionAuthzWireEvidence(
        status=runtime.authz_evidence.status,
        result_prefix=runtime.authz_evidence.result_prefix,
        expected_token_identity=runtime.authz_evidence.expected_token_identity,
        failure_reason_codes=runtime.authz_evidence.failure_reason_codes,
        child_accounting=authz_records.PermissionAuthzChildAccounting(
            acquire_status=child.acquire_status,
            lease_candidate_returned=child.lease_candidate_returned,
            acquire_failure=child.acquire_failure,
            lease_binding_status=child.lease_binding_status,
            checkpoint_outcomes=child.checkpoint_outcomes,
            finish_attempted=child.finish_attempted,
            requested_outcome=child.requested_outcome,
            finish_outcome=child.finish_outcome,
        ),
    )
    grants = _grants_for_operations(
        plan.observation_catalog,
        expected=frozenset({"observe", "observe_authz"}),
        name="durable observation",
    )
    parent_snapshot = issuer.operation_evidence_snapshot(
        plan.request, grants["observe"].expected_scope
    )
    child_snapshot = issuer.operation_evidence_snapshot(
        plan.request, grants["observe_authz"].expected_scope
    )
    operations = authz_records.PermissionAuthzAuthorityOperations(
        parent=_operation_projection(parent_snapshot),
        child=_operation_projection(child_snapshot),
    )
    record = authz_records.build_private_permission_authz_record(
        binding=plan.expected_authz_binding,
        request=plan.request,
        access_expectation=plan.access_expectation,
        retention_destination_self_sha256=(
            plan.authz_destination.destination_self_sha256
        ),
        store_expectation_self_sha256=(
            plan.authz_store_expectation.expectation_self_sha256
        ),
        base_observation=base,
        held_inputs=executed.hook.held_inputs_snapshot(),
        authz=wire_authz,
        authority_operations=operations,
        controller_failure_reason=runtime.controller_failure_reason,
    )
    return authz_records.canonical_private_permission_authz_bytes(
        record,
        expected_request=plan.request,
        expected_access=plan.access_expectation,
        expected_binding=plan.expected_authz_binding,
        expected_destination_self_sha256=(
            plan.authz_destination.destination_self_sha256
        ),
        expected_store_expectation_self_sha256=(
            plan.authz_store_expectation.expectation_self_sha256
        ),
    )


def _authz_failure_is_unknown(
    result: evidence.PermissionAuthzEvidenceRetentionFailure,
) -> bool:
    accountings = [result.main_authority_accounting]
    if result.failure_record_authority_accounting is not None:
        accountings.append(result.failure_record_authority_accounting)
    states = [result.payload_object_state]
    if result.failure_record_object_state is not None:
        states.append(result.failure_record_object_state)
    effects = list(result.payload_effects)
    if result.failure_record_effects is not None:
        effects.extend(result.failure_record_effects)
    return (
        any(item.effect_certainty == "unknown" for item in accountings)
        or any(
            item.effect_certainty == "unknown" or item.handle_state == "unknown"
            for item in states
        )
        or any(getattr(item, "effect_certainty", None) == "unknown" for item in effects)
    )


def _retain_authz_bundle(
    authz_record_bytes: bytes,
    *,
    plan: _PreparationPlan,
    authority: permission_authority.PermissionAuthorityIssuer,
    authz_store: evidence.ConfidentialEvidenceStore,
) -> (
    evidence.RetainedPermissionAuthzEvidence
    | evidence.PermissionAuthzEvidenceRetentionFailure
):
    if (
        plan.expected_authz_binding is None
        or plan.authz_destination is None
        or plan.authz_store_expectation is None
    ):
        _parent_error("durable Authz retention used incomplete preparation")
    record = authz_records.strict_private_permission_authz_loads(
        authz_record_bytes,
        expected_request=plan.request,
        expected_access=plan.access_expectation,
        expected_binding=plan.expected_authz_binding,
        expected_destination_self_sha256=(
            plan.authz_destination.destination_self_sha256
        ),
        expected_store_expectation_self_sha256=(
            plan.authz_store_expectation.expectation_self_sha256
        ),
    )
    return evidence.retain_private_permission_authz_exclusive(
        plan.authz_destination,
        record,
        expected_request=plan.request,
        expected_access=plan.access_expectation,
        expected_store=plan.authz_store_expectation,
        expected_authz_binding=plan.expected_authz_binding,
        authority=authority,
        store=authz_store,
    )


def observe_and_retain_permission(
    request: observation.PermissionObservationRequest,
    *,
    expected_request: observation.PermissionObservationRequest,
    access_expectation: observation.PermissionAccessExpectation,
    confidential_store_expectation: evidence.ConfidentialStoreExpectation,
    observation_destination: evidence.PermissionEvidenceDestination,
    observation_failure_destination: evidence.PermissionEvidenceDestination,
    authority: observation.PermissionObservationAuthority,
    native: observation.PermissionNativeApi,
    observation_store: evidence.ConfidentialEvidenceStore,
    observation_failure_store: evidence.ConfidentialEvidenceStore,
) -> PermissionControllerResult:
    """Observe exactly once and retain only the resulting primary payload kind."""

    return _observe_and_retain_permission(
        request,
        expected_request=expected_request,
        access_expectation=access_expectation,
        confidential_store_expectation=confidential_store_expectation,
        observation_destination=observation_destination,
        observation_failure_destination=observation_failure_destination,
        authority=authority,
        native=native,
        observation_store=observation_store,
        observation_failure_store=observation_failure_store,
        before_retention=None,
    )


class PermissionParentEnrollmentHandle:
    """Trusted preparation capability kept separate from parent operation."""

    __slots__ = ("_state",)

    def __init__(self, state: _PermissionParentState) -> None:
        self._state = state

    def prepare_observation(
        self,
        *,
        observation_catalog: PermissionParentCatalog,
        captured_result_catalog: PermissionParentCatalog,
        failed_result_catalog: PermissionParentCatalog,
    ) -> None:
        with self._state.lock:
            if self._state.execution_in_progress:
                _parent_error("observation preparation reentered operational execution")
            plan = _preparation_plan(
                enrollment=self._state.enrollment,
                observation_catalog=observation_catalog,
                captured_result_catalog=captured_result_catalog,
                failed_result_catalog=failed_result_catalog,
            )
            request_sha256 = plan.request.request_self_sha256
            existing = self._state.prepared_by_request_sha256.get(request_sha256)
            if existing is not None:
                if existing.plan == plan:
                    return
                _parent_error("prepared request conflicts with its retained catalogs")
            sequence = plan.request.observation_sequence
            prior_request_sha256 = self._state.request_sha256_by_sequence.get(sequence)
            if (
                prior_request_sha256 is not None
                and prior_request_sha256 != request_sha256
            ):
                _parent_error("preparation replaces the request for an enrolled sequence")
            supplied = plan.observation_catalog.supplied
            try:
                identity = self._state.enrollment_handle.register_catalog(
                    supplied.catalog_bytes,
                    expected_file_bytes=supplied.expected_file_bytes,
                    expected_file_sha256=supplied.expected_file_sha256,
                )
            except Exception as exc:
                _parent_error("observation catalog registration failed", exc)
            prepared = _PreparedObservation(plan, identity)
            self._state.prepared_by_request_sha256[request_sha256] = prepared
            self._state.request_sha256_by_sequence[sequence] = request_sha256

    def prepare_observation_with_authz(
        self,
        *,
        observation_catalog: PermissionParentCatalog,
        captured_result_catalog: PermissionParentCatalog,
        failed_result_catalog: PermissionParentCatalog,
        expected_operator_sid: observation.NativeSid,
    ) -> None:
        with self._state.lock:
            if self._state.execution_in_progress:
                _parent_error("Authz preparation reentered operational execution")
            plan = _authz_preparation_plan(
                enrollment=self._state.enrollment,
                observation_catalog=observation_catalog,
                captured_result_catalog=captured_result_catalog,
                failed_result_catalog=failed_result_catalog,
                expected_operator_sid=expected_operator_sid,
            )
            request_sha256 = plan.request.request_self_sha256
            existing = self._state.prepared_by_request_sha256.get(request_sha256)
            if existing is not None:
                if existing.plan == plan:
                    return
                _parent_error(
                    "prepared request conflicts with its mode, operator, or catalogs"
                )
            sequence = plan.request.observation_sequence
            prior_request_sha256 = self._state.request_sha256_by_sequence.get(sequence)
            if (
                prior_request_sha256 is not None
                and prior_request_sha256 != request_sha256
            ):
                _parent_error("preparation replaces the request for an enrolled sequence")
            supplied = plan.observation_catalog.supplied
            try:
                identity = self._state.enrollment_handle.register_catalog(
                    supplied.catalog_bytes,
                    expected_file_bytes=supplied.expected_file_bytes,
                    expected_file_sha256=supplied.expected_file_sha256,
                )
            except Exception as exc:
                _parent_error("combined observation catalog registration failed", exc)
            prepared = _PreparedObservation(plan, identity)
            self._state.prepared_by_request_sha256[request_sha256] = prepared
            self._state.request_sha256_by_sequence[sequence] = request_sha256

    def prepare_observation_with_durable_authz(
        self,
        *,
        observation_catalog: PermissionParentCatalog,
        captured_result_catalog: PermissionParentCatalog,
        failed_result_catalog: PermissionParentCatalog,
        expected_operator_sid: observation.NativeSid,
        expected_authz_binding: authz_records.PermissionAuthzEvidenceBinding,
        authz_retention_catalog: PermissionParentCatalog,
    ) -> None:
        with self._state.lock:
            if self._state.execution_in_progress:
                _parent_error("durable Authz preparation reentered execution")
            plan = _durable_authz_preparation_plan(
                enrollment=self._state.enrollment,
                observation_catalog=observation_catalog,
                captured_result_catalog=captured_result_catalog,
                failed_result_catalog=failed_result_catalog,
                expected_operator_sid=expected_operator_sid,
                expected_authz_binding=expected_authz_binding,
                authz_retention_catalog=authz_retention_catalog,
            )
            request_sha256 = plan.request.request_self_sha256
            existing = self._state.prepared_by_request_sha256.get(request_sha256)
            if existing is not None:
                if existing.plan == plan:
                    return
                _parent_error(
                    "prepared request conflicts with its durable mode, binding, operator, or catalogs"
                )
            sequence = plan.request.observation_sequence
            prior_request_sha256 = self._state.request_sha256_by_sequence.get(sequence)
            if (
                prior_request_sha256 is not None
                and prior_request_sha256 != request_sha256
            ):
                _parent_error("durable preparation replaces an enrolled request")
            supplied = plan.observation_catalog.supplied
            try:
                identity = self._state.enrollment_handle.register_catalog(
                    supplied.catalog_bytes,
                    expected_file_bytes=supplied.expected_file_bytes,
                    expected_file_sha256=supplied.expected_file_sha256,
                )
            except Exception as exc:
                _parent_error("durable observation catalog registration failed", exc)
            prepared = _PreparedObservation(plan, identity)
            self._state.prepared_by_request_sha256[request_sha256] = prepared
            self._state.request_sha256_by_sequence[sequence] = request_sha256

    def assess_retained_authz_observation(
        self,
        request: observation.PermissionObservationRequest,
        *,
        expected_retained_snapshot: evidence.StoreObjectSnapshot,
    ) -> _PermissionAuthzAssessmentPhaseResult:
        with self._state.lock:
            if self._state.execution_in_progress:
                return _PermissionAuthzAssessmentPhaseResult(
                    "incomplete", None, "execution_in_progress"
                )
            try:
                observation.validate_permission_observation_request(
                    request, expected_request=request
                )
            except Exception:
                return _PermissionAuthzAssessmentPhaseResult(
                    "incomplete", None, "request_rejected"
                )
            request_sha256 = request.request_self_sha256
            prepared = self._state.prepared_by_request_sha256.get(request_sha256)
            if (
                prepared is None
                or prepared.plan.request != request
                or prepared.plan.mode != "held_token_authz"
                or prepared.plan.expected_operator_sid is None
            ):
                return _PermissionAuthzAssessmentPhaseResult(
                    "incomplete", None, "authz_preparation_missing"
                )
            runtime = self._state.runtime_authz_by_request_sha256.get(
                request_sha256
            )
            if runtime is None:
                return _PermissionAuthzAssessmentPhaseResult(
                    "incomplete", None, "authz_execution_missing"
                )
            accounting = runtime.authz_evidence.child_accounting
            prerequisites_complete = (
                runtime.controller_failure_reason is None
                and type(runtime.observation_result)
                is observation.CapturedPermissionObservation
                and type(runtime.retention_result)
                is evidence.RetainedPermissionEvidence
                and type(runtime.retained_observation_bytes) is bytes
                and runtime.authz_evidence.status == "completed"
                and len(runtime.authz_evidence.result_prefix) == 2
                and runtime.authz_evidence.expected_token_identity is not None
                and accounting.acquire_status == "granted"
                and accounting.lease_binding_status == "valid"
                and accounting.finish_attempted
                and accounting.requested_outcome == "succeeded"
                and accounting.finish_outcome
                == observation.AuthorityFinishOutcome(
                    "succeeded", "confirmed", "succeeded", "none"
                )
                and type(expected_retained_snapshot)
                is evidence.StoreObjectSnapshot
            )
            if not prerequisites_complete:
                return _PermissionAuthzAssessmentPhaseResult(
                    "incomplete", None, "assessment_prerequisites_incomplete"
                )
            retained = cast(
                evidence.RetainedPermissionEvidence,
                runtime.retention_result,
            )
            retained_bytes = cast(bytes, runtime.retained_observation_bytes)
            if (
                len(retained_bytes) != retained.file_bytes
                or sha256_bytes(retained_bytes) != retained.file_sha256
            ):
                return _PermissionAuthzAssessmentPhaseResult(
                    "incomplete", None, "retained_bytes_mismatch"
                )
            authz_results = cast(
                tuple[_AuthzResult, _AuthzResult],
                runtime.authz_evidence.result_prefix,
            )
            try:
                supplied = assessment.assess_supplied_permission_bundle(
                    retained_bytes,
                    retained,
                    authz_results,
                    expected_request=prepared.plan.request,
                    expected_access=prepared.plan.access_expectation,
                    expected_store=prepared.plan.store_expectation,
                    expected_destination=prepared.plan.observation_destination,
                    expected_retained_snapshot=expected_retained_snapshot,
                    expected_operator_sid=prepared.plan.expected_operator_sid,
                    expected_authz_token_identity=cast(
                        authz.SuppliedTokenIdentity,
                        runtime.authz_evidence.expected_token_identity,
                    ),
                )
            except Exception:
                return _PermissionAuthzAssessmentPhaseResult(
                    "incomplete", None, "supplied_assessment_rejected"
                )
            return _PermissionAuthzAssessmentPhaseResult(
                "completed", supplied, "completed"
            )


class EnrolledPermissionParent:
    """Operational parent capability without catalog-registration methods."""

    __slots__ = ("_state",)

    def __init__(self, state: _PermissionParentState) -> None:
        self._state = state

    def observe_and_retain(
        self,
        request: observation.PermissionObservationRequest,
        *,
        native: observation.PermissionNativeApi,
        observation_store: evidence.ConfidentialEvidenceStore,
        observation_failure_store: evidence.ConfidentialEvidenceStore,
    ) -> PermissionControllerResult:
        with self._state.lock:
            if self._state.execution_in_progress:
                _parent_error("parent observation execution is reentrant")
            try:
                observation.validate_permission_observation_request(
                    request,
                    expected_request=request,
                )
            except Exception as exc:
                _parent_error("candidate parent request is malformed", exc)
            request_sha256 = request.request_self_sha256
            prepared = self._state.prepared_by_request_sha256.get(request_sha256)
            if prepared is None or prepared.plan.request != request:
                _parent_error("candidate parent request was not independently prepared")
            if prepared.plan.mode != "ordinary":
                _parent_error("candidate parent request requires held-token Authz execution")
            if request_sha256 in self._state.attempted_request_sha256s:
                _parent_error("prepared parent request was already executed")
            self._state.attempted_request_sha256s.add(request_sha256)
            self._state.execution_in_progress = True
            try:
                plan = prepared.plan

                def activate_selected_catalog(
                    observed: _ObservationResult,
                ) -> None:
                    selected = (
                        plan.captured_result_catalog
                        if type(observed) is observation.CapturedPermissionObservation
                        else plan.failed_result_catalog
                    )
                    supplied = selected.supplied
                    try:
                        identity = self._state.enrollment_handle.register_catalog(
                            supplied.catalog_bytes,
                            expected_file_bytes=supplied.expected_file_bytes,
                            expected_file_sha256=supplied.expected_file_sha256,
                        )
                    except Exception as exc:
                        error = PermissionParentActivationError(observed, exc)
                        raise error from exc
                    self._state.activated_catalog_identities[
                        supplied.expected_file_sha256
                    ] = identity

                return _observe_and_retain_permission(
                    request,
                    expected_request=plan.request,
                    access_expectation=plan.access_expectation,
                    confidential_store_expectation=plan.store_expectation,
                    observation_destination=plan.observation_destination,
                    observation_failure_destination=(
                        plan.observation_failure_destination
                    ),
                    authority=self._state.issuer,
                    native=native,
                    observation_store=observation_store,
                    observation_failure_store=observation_failure_store,
                    before_retention=activate_selected_catalog,
                )
            finally:
                self._state.execution_in_progress = False

    def observe_and_retain_with_authz(
        self,
        request: observation.PermissionObservationRequest,
        *,
        native: native_win32.Win32PermissionNativeApi,
        authz_bindings: authz.AuthzNativeBindings,
        observation_store: evidence.ConfidentialEvidenceStore,
        observation_failure_store: evidence.ConfidentialEvidenceStore,
    ) -> _PermissionAuthzRuntimeResult:
        with self._state.lock:
            if self._state.execution_in_progress:
                _parent_error("parent observation execution is reentrant")
            if type(native) is not native_win32.Win32PermissionNativeApi:
                _parent_error("held-token Authz execution requires the exact Win32 adapter")
            try:
                authz._validate_bindings(authz_bindings)
                observation.validate_permission_observation_request(
                    request,
                    expected_request=request,
                )
            except Exception as exc:
                _parent_error("candidate held-token Authz inputs are malformed", exc)
            request_sha256 = request.request_self_sha256
            prepared = self._state.prepared_by_request_sha256.get(request_sha256)
            if prepared is None or prepared.plan.request != request:
                _parent_error("candidate parent request was not independently prepared")
            if prepared.plan.mode != "held_token_authz":
                _parent_error("candidate parent request requires ordinary execution")
            if request_sha256 in self._state.attempted_request_sha256s:
                _parent_error("prepared parent request was already executed")
            self._state.attempted_request_sha256s.add(request_sha256)
            self._state.execution_in_progress = True
            try:
                plan = prepared.plan

                def activate_selected_catalog(
                    observed: _ObservationResult,
                ) -> None:
                    selected = (
                        plan.captured_result_catalog
                        if type(observed) is observation.CapturedPermissionObservation
                        else plan.failed_result_catalog
                    )
                    supplied = selected.supplied
                    try:
                        identity = self._state.enrollment_handle.register_catalog(
                            supplied.catalog_bytes,
                            expected_file_bytes=supplied.expected_file_bytes,
                            expected_file_sha256=supplied.expected_file_sha256,
                        )
                    except Exception as exc:
                        error = PermissionParentActivationError(observed, exc)
                        raise error from exc
                    self._state.activated_catalog_identities[
                        supplied.expected_file_sha256
                    ] = identity

                runtime = _observe_and_retain_permission_with_authz(
                    request,
                    plan=plan,
                    authority=self._state.issuer,
                    native=native,
                    authz_bindings=authz_bindings,
                    observation_store=observation_store,
                    observation_failure_store=observation_failure_store,
                    before_retention=activate_selected_catalog,
                )
                self._state.runtime_authz_by_request_sha256[
                    request_sha256
                ] = runtime
                return runtime
            finally:
                self._state.execution_in_progress = False

    def observe_and_retain_with_durable_authz(
        self,
        request: observation.PermissionObservationRequest,
        *,
        native: native_win32.Win32PermissionNativeApi,
        authz_bindings: authz.AuthzNativeBindings,
        observation_store: evidence.ConfidentialEvidenceStore,
        observation_failure_store: evidence.ConfidentialEvidenceStore,
        authz_store: evidence.ConfidentialEvidenceStore,
    ) -> PermissionDurableAuthzRuntimeResult:
        with self._state.lock:
            if self._state.execution_in_progress:
                _parent_error("durable Authz execution is reentrant")
            if type(native) is not native_win32.Win32PermissionNativeApi:
                _parent_error("durable Authz execution requires the exact Win32 adapter")
            if authz_store is observation_store or authz_store is observation_failure_store:
                _parent_error("durable Authz store cannot replace a base-observation store")
            try:
                authz._validate_bindings(authz_bindings)
                observation.validate_permission_observation_request(
                    request, expected_request=request
                )
            except Exception as exc:
                _parent_error("candidate durable Authz inputs are malformed", exc)
            request_sha256 = request.request_self_sha256
            prepared = self._state.prepared_by_request_sha256.get(request_sha256)
            if prepared is None or prepared.plan.request != request:
                _parent_error("candidate durable request was not independently prepared")
            if prepared.plan.mode != "durable_held_token_authz":
                _parent_error("candidate request requires a different execution mode")
            if request_sha256 in self._state.attempted_request_sha256s:
                _parent_error("prepared durable request was already executed")
            self._state.attempted_request_sha256s.add(request_sha256)
            self._state.execution_in_progress = True
            try:
                plan = prepared.plan

                def activate_selected_catalog(observed: _ObservationResult) -> None:
                    selected = (
                        plan.captured_result_catalog
                        if type(observed) is observation.CapturedPermissionObservation
                        else plan.failed_result_catalog
                    )
                    supplied = selected.supplied
                    try:
                        identity = self._state.enrollment_handle.register_catalog(
                            supplied.catalog_bytes,
                            expected_file_bytes=supplied.expected_file_bytes,
                            expected_file_sha256=supplied.expected_file_sha256,
                        )
                    except Exception as exc:
                        error = PermissionParentActivationError(observed, exc)
                        raise error from exc
                    self._state.activated_catalog_identities[
                        supplied.expected_file_sha256
                    ] = identity

                executed = _execute_prepared_authz_observation(
                    request,
                    plan=plan,
                    authority=self._state.issuer,
                    native=native,
                    authz_bindings=authz_bindings,
                    observation_store=observation_store,
                    observation_failure_store=observation_failure_store,
                    before_retention=activate_selected_catalog,
                )
                self._state.runtime_authz_by_request_sha256[
                    request_sha256
                ] = executed.runtime
                try:
                    authz_record_bytes = _build_authz_bundle_from_runtime(
                        executed,
                        plan=plan,
                        issuer=self._state.issuer,
                    )
                except Exception:
                    durable = PermissionDurableAuthzRuntimeResult(
                        executed.runtime,
                        None,
                        "not_attempted",
                        None,
                        "authz_bundle_build_failed",
                    )
                    self._state.runtime_durable_authz_by_request_sha256[
                        request_sha256
                    ] = durable
                    return durable
                retained_catalog = plan.authz_retention_catalog
                if retained_catalog is None:
                    durable = PermissionDurableAuthzRuntimeResult(
                        executed.runtime,
                        authz_record_bytes,
                        "not_attempted",
                        None,
                        "authz_retention_catalog_activation_failed",
                    )
                    self._state.runtime_durable_authz_by_request_sha256[
                        request_sha256
                    ] = durable
                    return durable
                supplied = retained_catalog.supplied
                try:
                    identity = (
                        self._state.enrollment_handle.register_authz_retention_catalog(
                            supplied.catalog_bytes,
                            expected_file_bytes=supplied.expected_file_bytes,
                            expected_file_sha256=supplied.expected_file_sha256,
                        )
                    )
                except Exception:
                    durable = PermissionDurableAuthzRuntimeResult(
                        executed.runtime,
                        authz_record_bytes,
                        "not_attempted",
                        None,
                        "authz_retention_catalog_activation_failed",
                    )
                    self._state.runtime_durable_authz_by_request_sha256[
                        request_sha256
                    ] = durable
                    return durable
                self._state.activated_catalog_identities[
                    supplied.expected_file_sha256
                ] = identity
                try:
                    retained = _retain_authz_bundle(
                        authz_record_bytes,
                        plan=plan,
                        authority=self._state.issuer,
                        authz_store=authz_store,
                    )
                except Exception:
                    durable = PermissionDurableAuthzRuntimeResult(
                        executed.runtime,
                        authz_record_bytes,
                        "retention_unknown",
                        None,
                        "authz_retention_integration_failed",
                    )
                else:
                    status: Literal[
                        "not_retained", "retained", "retention_unknown"
                    ]
                    if type(retained) is evidence.RetainedPermissionAuthzEvidence:
                        status = "retained"
                    elif _authz_failure_is_unknown(
                        cast(evidence.PermissionAuthzEvidenceRetentionFailure, retained)
                    ):
                        status = "retention_unknown"
                    else:
                        status = "not_retained"
                    durable = PermissionDurableAuthzRuntimeResult(
                        executed.runtime,
                        authz_record_bytes,
                        status,
                        retained,
                        None,
                    )
                self._state.runtime_durable_authz_by_request_sha256[
                    request_sha256
                ] = durable
                return durable
            finally:
                self._state.execution_in_progress = False


def build_enrolled_permission_parent(
    *,
    enrollment: approval.PermissionApprovalEnrollment,
    revocation_directory: str | os.PathLike[str],
    utc_ns_clock: Callable[[], int],
    journal: authority_journal.FilePermissionAuthorityJournal,
) -> tuple[EnrolledPermissionParent, PermissionParentEnrollmentHandle]:
    """Build one persistent enrolled resolver, issuer, and parent session."""

    approval.validate_permission_approval_enrollment(enrollment)
    if type(journal) is not authority_journal.FilePermissionAuthorityJournal:
        _parent_error(
            "an existing exact FilePermissionAuthorityJournal instance is mandatory"
        )
    resolver, enrollment_handle = (
        approval.build_enrolled_permission_approval_resolver(
            enrollment=enrollment,
            revocation_directory=revocation_directory,
            utc_ns_clock=utc_ns_clock,
        )
    )
    issuer = permission_authority.PermissionAuthorityIssuer(
        resolver=resolver,
        journal=journal,
    )
    state = _PermissionParentState(
        enrollment=enrollment,
        resolver=resolver,
        enrollment_handle=enrollment_handle,
        issuer=issuer,
        journal=journal,
    )
    return EnrolledPermissionParent(state), PermissionParentEnrollmentHandle(state)
