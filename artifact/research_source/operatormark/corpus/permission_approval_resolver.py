"""Trusted enrollment and revocation-aware permission approval resolution.

This module is a private administrative boundary before State 3.  It assumes
that the caller independently authenticated the enrollment and independently
verified every catalog identity.  It does not authenticate a human, provision
an administrative namespace, retain catalog bytes, activate an attempt, consume
an allowance, or perform journal or corpus work.

The revocation directory must be an existing independently controlled
directory.  The identity and reparse checks below fail closed, but production
namespace protection and rollback resistance still require qualification; they
cannot be proved by path-based inspection against a hostile administrator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
import stat
from threading import RLock
from typing import Callable, Protocol

from .canonical import (
    canonical_json_bytes,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    strict_canonical_json_loads,
)
from . import permission_authority as authority
from . import permission_observation_windows as observation
from . import private_permission_evidence as evidence


__all__ = [
    "PERMISSION_APPROVAL_ENROLLMENT_SCHEMA",
    "PERMISSION_AUTHZ_RETENTION_CATALOG_SCHEMA",
    "PERMISSION_GRANT_CATALOG_SCHEMA",
    "EnrolledPermissionApprovalResolver",
    "PermissionApprovalEnrollment",
    "PermissionApprovalResolverError",
    "PermissionGrantCatalogIdentity",
    "PermissionGrantEnrollmentHandle",
    "build_enrolled_permission_approval_resolver",
    "canonical_permission_approval_enrollment_bytes",
    "canonical_permission_authz_retention_catalog_bytes",
    "canonical_permission_grant_catalog_bytes",
    "permission_approval_enrollment_binding_sha256",
    "strict_permission_approval_enrollment_loads",
    "strict_permission_authz_retention_catalog_loads",
    "strict_permission_grant_catalog_loads",
    "validate_permission_approval_enrollment",
]


PERMISSION_APPROVAL_ENROLLMENT_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_approval_enrollment.v1"
)
PERMISSION_GRANT_CATALOG_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_grant_catalog.v1"
)
PERMISSION_AUTHZ_RETENTION_CATALOG_SCHEMA = (
    "operatormark.darcy_branch_c_v1.permission_authz_retention_catalog.v1"
)

_ENROLLMENT_DOMAIN = b"operatormark-permission-approval-enrollment-v1\n"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_OPEN_SPECIFICATION_BOOLEAN_FIELDS = (
    "existing_only",
    "follow_reparse_points",
    "read_data",
    "read_attributes",
    "read_control",
    "write_data",
    "append_data",
    "write_attributes",
    "delete",
    "write_dac",
    "write_owner",
    "create",
    "truncate",
    "share_read",
    "share_write",
    "share_delete",
)


class PermissionApprovalResolverError(authority.PermissionAuthorityContractError):
    """Trusted enrollment or catalog data violates the resolver contract."""


@dataclass(frozen=True, slots=True)
class PermissionApprovalEnrollment:
    """One independently authenticated State 4 permission enrollment."""

    schema: str
    approval_reference: str
    expected_context: observation.PermissionObservationContext
    expected_source_binding: observation.PermissionComponentSource
    accepted_state3_audit_file_sha256: str
    accepted_compute_feasibility_report_file_sha256: str
    not_before_utc_ns: int
    expires_at_utc_ns: int


@dataclass(frozen=True, slots=True)
class PermissionGrantCatalogIdentity:
    """Immutable audit identity retained without retaining catalog bytes."""

    enrollment_binding_sha256: str
    catalog_file_bytes: int
    catalog_file_sha256: str
    grant_binding_sha256s: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _NodeIdentity:
    device: int
    inode: int
    mode_type: int
    file_attributes: int


class _RevocationFilesystem(Protocol):
    def lstat(self, path: Path) -> os.stat_result: ...


class _OSRevocationFilesystem:
    @staticmethod
    def lstat(path: Path) -> os.stat_result:
        return os.lstat(path)


def _raise(message: str, cause: BaseException | None = None) -> None:
    error = PermissionApprovalResolverError(message)
    if cause is None:
        raise error
    raise error from cause


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _raise(f"{name} must be non-empty exact text")
    return value


def _sha(value: object, *, name: str) -> str:
    try:
        return require_sha256(value, name=name)
    except Exception as exc:
        _raise(str(exc), exc)


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _raise(f"{name} must be an exact integer >= {minimum}")
    return value


def _mapping(value: object, expected_fields: tuple[str, ...], *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        _raise(f"{name} must be a JSON object")
    try:
        require_exact_keys(value, expected_fields, name=name)
    except Exception as exc:
        _raise(str(exc), exc)
    return value


def _field_names(value_type: type) -> tuple[str, ...]:
    return tuple(item.name for item in fields(value_type))


def _array(value: object, *, name: str, nonempty: bool = False) -> list[object]:
    if type(value) is not list or (nonempty and not value):
        suffix = " non-empty" if nonempty else ""
        _raise(f"{name} must be a{suffix} JSON array")
    return value


def _context_from_mapping(value: object, *, name: str) -> observation.PermissionObservationContext:
    record = _mapping(
        value,
        _field_names(observation.PermissionObservationContext),
        name=name,
    )
    residual = _array(
        record["accepted_local_user_residual_principal_commitments"],
        name=f"{name}.accepted_local_user_residual_principal_commitments",
    )
    try:
        context = observation.PermissionObservationContext(
            protocol_file_sha256=record["protocol_file_sha256"],
            state4_authorization_file_sha256=record["state4_authorization_file_sha256"],
            implementation_commit=record["implementation_commit"],
            implementation_tree=record["implementation_tree"],
            storage_preflight_file_sha256=record["storage_preflight_file_sha256"],
            permission_profile_file_sha256=record["permission_profile_file_sha256"],
            repository_root_commitment=record["repository_root_commitment"],
            volume_identifier_commitment=record["volume_identifier_commitment"],
            accepted_local_user_residual_principal_commitments=tuple(residual),
        )
        observation._validate_context(context)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)
    return context


def _source_from_mapping(value: object, *, name: str) -> observation.PermissionComponentSource:
    record = _mapping(
        value,
        _field_names(observation.PermissionComponentSource),
        name=name,
    )
    try:
        source = observation.PermissionComponentSource(**record)
        observation._validate_source(source)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)
    return source


def _subject_from_mapping(value: object, *, name: str) -> observation.PermissionObservationSubject:
    record = _mapping(
        value,
        _field_names(observation.PermissionObservationSubject),
        name=name,
    )
    ancestors = _array(
        record["required_ancestor_canonical_volume_guid_paths"],
        name=f"{name}.required_ancestor_canonical_volume_guid_paths",
        nonempty=True,
    )
    try:
        return observation.PermissionObservationSubject(
            kind=record["kind"],
            row_index=record["row_index"],
            relative_path=record["relative_path"],
            repository_relative_path=record["repository_relative_path"],
            expected_file_bytes=record["expected_file_bytes"],
            expected_file_sha256=record["expected_file_sha256"],
            canonical_volume_guid_path=record["canonical_volume_guid_path"],
            required_ancestor_canonical_volume_guid_paths=tuple(ancestors),
        )
    except Exception as exc:
        _raise(f"{name} is malformed", exc)


def _request_from_mapping(value: object, *, name: str) -> observation.PermissionObservationRequest:
    record = _mapping(
        value,
        _field_names(observation.PermissionObservationRequest),
        name=name,
    )
    context = _context_from_mapping(record["context"], name=f"{name}.context")
    subject = _subject_from_mapping(record["subject"], name=f"{name}.subject")
    source = _source_from_mapping(record["source_binding"], name=f"{name}.source_binding")
    try:
        request = observation.PermissionObservationRequest(
            schema=record["schema"],
            state=record["state"],
            action=record["action"],
            attempt_id=record["attempt_id"],
            permission_mechanism=record["permission_mechanism"],
            observer_role=record["observer_role"],
            context=context,
            subject=subject,
            observation_phase=record["observation_phase"],
            observation_sequence=record["observation_sequence"],
            source_binding=source,
            request_self_sha256=record["request_self_sha256"],
        )
        observation.validate_permission_observation_request(
            request,
            expected_request=request,
        )
    except Exception as exc:
        _raise(f"{name} is malformed", exc)
    return request


def _generic_mapping_from_mapping(value: object, *, name: str) -> observation.NativeGenericMapping:
    record = _mapping(value, _field_names(observation.NativeGenericMapping), name=name)
    try:
        return observation.NativeGenericMapping(**record)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)


def _open_specification_from_mapping(
    value: object, *, name: str
) -> observation.NativeOpenSpecification:
    record = _mapping(value, _field_names(observation.NativeOpenSpecification), name=name)
    try:
        return observation.NativeOpenSpecification(**record)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)


def _access_from_mapping(
    value: object,
    *,
    request: observation.PermissionObservationRequest,
    name: str,
) -> observation.PermissionAccessExpectation:
    record = _mapping(
        value,
        _field_names(observation.PermissionAccessExpectation),
        name=name,
    )
    file_mapping = _generic_mapping_from_mapping(
        record["file_generic_mapping"], name=f"{name}.file_generic_mapping"
    )
    directory_mapping = _generic_mapping_from_mapping(
        record["directory_generic_mapping"], name=f"{name}.directory_generic_mapping"
    )
    subject_open = _open_specification_from_mapping(
        record["subject_open_specification"],
        name=f"{name}.subject_open_specification",
    )
    ancestor_open = _open_specification_from_mapping(
        record["ancestor_open_specification"],
        name=f"{name}.ancestor_open_specification",
    )
    try:
        access = observation.PermissionAccessExpectation(
            schema=record["schema"],
            request_self_sha256=record["request_self_sha256"],
            permission_profile_file_sha256=record["permission_profile_file_sha256"],
            permission_mechanism=record["permission_mechanism"],
            desired_access_hex=record["desired_access_hex"],
            file_generic_mapping=file_mapping,
            directory_generic_mapping=directory_mapping,
            subject_open_specification=subject_open,
            ancestor_open_specification=ancestor_open,
            expectation_self_sha256=record["expectation_self_sha256"],
        )
        observation.validate_permission_access_expectation(
            access,
            expected_request=request,
        )
    except Exception as exc:
        _raise(f"{name} is malformed", exc)
    return access


def _scope_from_mapping(
    value: object,
    *,
    request: observation.PermissionObservationRequest,
    name: str,
) -> observation.PermissionOperationScope:
    record = _mapping(
        value,
        _field_names(observation.PermissionOperationScope),
        name=name,
    )
    try:
        scope = observation.PermissionOperationScope(**record)
        authority.validate_permission_operation_scope(scope, request=request)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)
    return scope


def _store_from_mapping(
    value: object,
    *,
    request: observation.PermissionObservationRequest,
    name: str,
) -> evidence.ConfidentialStoreExpectation:
    record = _mapping(
        value,
        _field_names(evidence.ConfidentialStoreExpectation),
        name=name,
    )
    try:
        store = evidence.ConfidentialStoreExpectation(**record)
        evidence.validate_confidential_store_expectation(
            store,
            expected_request=request,
        )
    except Exception as exc:
        _raise(f"{name} is malformed", exc)
    return store


def _destination_from_mapping(
    value: object,
    *,
    request: observation.PermissionObservationRequest,
    store: evidence.ConfidentialStoreExpectation,
    name: str,
) -> evidence.PermissionEvidenceDestination:
    record = _mapping(
        value,
        _field_names(evidence.PermissionEvidenceDestination),
        name=name,
    )
    try:
        destination = evidence.PermissionEvidenceDestination(**record)
        evidence.validate_permission_evidence_destination(
            destination,
            expected_request=request,
            expected_store=store,
        )
    except Exception as exc:
        _raise(f"{name} is malformed", exc)
    return destination


def _authz_destination_from_mapping(
    value: object,
    *,
    request: observation.PermissionObservationRequest,
    store: evidence.ConfidentialStoreExpectation,
    name: str,
) -> evidence.PermissionAuthzEvidenceDestination:
    record = _mapping(
        value,
        _field_names(evidence.PermissionAuthzEvidenceDestination),
        name=name,
    )
    try:
        destination = evidence.PermissionAuthzEvidenceDestination(**record)
        evidence.validate_permission_authz_evidence_destination(
            destination,
            expected_request=request,
            expected_store=store,
        )
    except Exception as exc:
        _raise(f"{name} is malformed", exc)
    return destination


def _grant_from_mapping(value: object, *, name: str) -> authority.PermissionApprovalGrant:
    record = _mapping(
        value,
        _field_names(authority.PermissionApprovalGrant),
        name=name,
    )
    request = _request_from_mapping(record["expected_request"], name=f"{name}.expected_request")
    context = _context_from_mapping(record["expected_context"], name=f"{name}.expected_context")
    source = _source_from_mapping(
        record["expected_source_binding"],
        name=f"{name}.expected_source_binding",
    )
    access = _access_from_mapping(
        record["expected_access_expectation"],
        request=request,
        name=f"{name}.expected_access_expectation",
    )
    scope = _scope_from_mapping(
        record["expected_scope"],
        request=request,
        name=f"{name}.expected_scope",
    )
    raw_store = record["expected_store_expectation"]
    if raw_store is None:
        store = None
    else:
        store = _store_from_mapping(
            raw_store,
            request=request,
            name=f"{name}.expected_store_expectation",
        )
    raw_destination = record["expected_destination"]
    if raw_destination is None:
        destination = None
    else:
        if store is None:
            _raise(f"{name}.expected_destination requires an exact store expectation")
        if record["schema"] == authority.AUTHZ_RETENTION_APPROVAL_GRANT_SCHEMA:
            destination = _authz_destination_from_mapping(
                raw_destination,
                request=request,
                store=store,
                name=f"{name}.expected_destination",
            )
        else:
            destination = _destination_from_mapping(
                raw_destination,
                request=request,
                store=store,
                name=f"{name}.expected_destination",
            )
    try:
        grant = authority.PermissionApprovalGrant(
            schema=record["schema"],
            approval_reference=record["approval_reference"],
            state4_authorization_file_sha256=record["state4_authorization_file_sha256"],
            accepted_state3_audit_file_sha256=record["accepted_state3_audit_file_sha256"],
            accepted_compute_feasibility_report_file_sha256=record[
                "accepted_compute_feasibility_report_file_sha256"
            ],
            expected_context=context,
            expected_source_binding=source,
            expected_permission_profile_file_sha256=record[
                "expected_permission_profile_file_sha256"
            ],
            expected_request=request,
            expected_access_expectation=access,
            expected_destination=destination,
            expected_store_expectation=store,
            expected_scope=scope,
        )
        authority.permission_approval_grant_binding_sha256(grant)
    except Exception as exc:
        _raise(f"{name} is malformed", exc)
    return grant


def validate_permission_approval_enrollment(
    enrollment: PermissionApprovalEnrollment,
) -> None:
    if type(enrollment) is not PermissionApprovalEnrollment:
        _raise("enrollment has the wrong exact type")
    if enrollment.schema != PERMISSION_APPROVAL_ENROLLMENT_SCHEMA:
        _raise("enrollment schema mismatch")
    _text(enrollment.approval_reference, name="enrollment.approval_reference")
    try:
        observation._validate_context(enrollment.expected_context)
        observation._validate_source(enrollment.expected_source_binding)
    except Exception as exc:
        _raise("enrollment context/source binding is malformed", exc)
    _sha(
        enrollment.accepted_state3_audit_file_sha256,
        name="enrollment.accepted_state3_audit_file_sha256",
    )
    _sha(
        enrollment.accepted_compute_feasibility_report_file_sha256,
        name="enrollment.accepted_compute_feasibility_report_file_sha256",
    )
    not_before = _integer(
        enrollment.not_before_utc_ns,
        name="enrollment.not_before_utc_ns",
    )
    expires_at = _integer(
        enrollment.expires_at_utc_ns,
        name="enrollment.expires_at_utc_ns",
    )
    if not_before >= expires_at:
        _raise("enrollment validity interval must be nonempty")


def canonical_permission_approval_enrollment_bytes(
    enrollment: PermissionApprovalEnrollment,
) -> bytes:
    validate_permission_approval_enrollment(enrollment)
    try:
        return canonical_json_bytes(asdict(enrollment))
    except Exception as exc:
        _raise("enrollment is not canonical-JSON encodable", exc)


def strict_permission_approval_enrollment_loads(
    data: bytes,
) -> PermissionApprovalEnrollment:
    if type(data) is not bytes:
        _raise("enrollment bytes must have the exact bytes type")
    try:
        record = strict_canonical_json_loads(data)
    except Exception as exc:
        _raise("enrollment bytes are not canonical JSON", exc)
    record = _mapping(
        record,
        _field_names(PermissionApprovalEnrollment),
        name="permission approval enrollment",
    )
    context = _context_from_mapping(
        record["expected_context"],
        name="enrollment.expected_context",
    )
    source = _source_from_mapping(
        record["expected_source_binding"],
        name="enrollment.expected_source_binding",
    )
    try:
        enrollment = PermissionApprovalEnrollment(
            schema=record["schema"],
            approval_reference=record["approval_reference"],
            expected_context=context,
            expected_source_binding=source,
            accepted_state3_audit_file_sha256=record[
                "accepted_state3_audit_file_sha256"
            ],
            accepted_compute_feasibility_report_file_sha256=record[
                "accepted_compute_feasibility_report_file_sha256"
            ],
            not_before_utc_ns=record["not_before_utc_ns"],
            expires_at_utc_ns=record["expires_at_utc_ns"],
        )
        validate_permission_approval_enrollment(enrollment)
        if canonical_permission_approval_enrollment_bytes(enrollment) != data:
            _raise("enrollment did not round-trip to its exact canonical bytes")
    except PermissionApprovalResolverError:
        raise
    except Exception as exc:
        _raise("enrollment is malformed", exc)
    return enrollment


def permission_approval_enrollment_binding_sha256(
    enrollment: PermissionApprovalEnrollment,
) -> str:
    return sha256_bytes(
        _ENROLLMENT_DOMAIN + canonical_permission_approval_enrollment_bytes(enrollment)
    )


def _validate_exact_catalog_grant_scalar_types(
    grant: authority.PermissionApprovalGrant,
) -> None:
    access = grant.expected_access_expectation
    specifications = (
        ("subject_open_specification", access.subject_open_specification),
        ("ancestor_open_specification", access.ancestor_open_specification),
    )
    for specification_name, specification in specifications:
        for field_name in _OPEN_SPECIFICATION_BOOLEAN_FIELDS:
            if type(getattr(specification, field_name)) is not bool:
                _raise(
                    "catalog grant "
                    f"{specification_name}.{field_name} must have exact bool type"
                )
    row_index = grant.expected_request.subject.row_index
    if row_index is not None and (type(row_index) is not int or row_index < 0):
        _raise("catalog grant subject.row_index must be null or an exact nonnegative integer")
    destination = grant.expected_destination
    if destination is not None:
        observation_sequence = destination.observation_sequence
        if type(observation_sequence) is not int or observation_sequence < 0:
            _raise(
                "catalog grant destination.observation_sequence must be an exact "
                "nonnegative integer"
            )


def _validate_grant_against_enrollment(
    grant: authority.PermissionApprovalGrant,
    *,
    enrollment: PermissionApprovalEnrollment,
) -> str:
    validate_permission_approval_enrollment(enrollment)
    try:
        binding = authority.permission_approval_grant_binding_sha256(grant)
        _validate_exact_catalog_grant_scalar_types(grant)
    except Exception as exc:
        _raise("catalog grant is malformed", exc)
    if grant.approval_reference != enrollment.approval_reference:
        _raise("catalog grant approval reference differs from enrollment")
    if grant.expected_context != enrollment.expected_context:
        _raise("catalog grant context differs from enrollment")
    if grant.expected_source_binding != enrollment.expected_source_binding:
        _raise("catalog grant source binding differs from enrollment")
    if (
        grant.state4_authorization_file_sha256
        != enrollment.expected_context.state4_authorization_file_sha256
    ):
        _raise("catalog grant A4 binding differs from enrollment")
    if (
        grant.expected_permission_profile_file_sha256
        != enrollment.expected_context.permission_profile_file_sha256
    ):
        _raise("catalog grant permission profile differs from enrollment")
    if (
        grant.accepted_state3_audit_file_sha256
        != enrollment.accepted_state3_audit_file_sha256
    ):
        _raise("catalog grant State 3 audit binding differs from enrollment")
    if (
        grant.accepted_compute_feasibility_report_file_sha256
        != enrollment.accepted_compute_feasibility_report_file_sha256
    ):
        _raise("catalog grant feasibility binding differs from enrollment")
    return binding


def _validate_catalog_grants(
    grants: tuple[authority.PermissionApprovalGrant, ...],
    *,
    enrollment: PermissionApprovalEnrollment,
    require_binding_order: bool,
) -> tuple[str, ...]:
    if not grants:
        _raise("permission grant catalog must not be empty")
    bindings: list[str] = []
    requests_by_sequence: dict[int, observation.PermissionObservationRequest] = {}
    bindings_by_request_operation: dict[tuple[str, str], str] = {}
    for index, grant in enumerate(grants):
        binding = _validate_grant_against_enrollment(grant, enrollment=enrollment)
        if require_binding_order and bindings and binding <= bindings[-1]:
            _raise("catalog grants are not strictly ordered by grant binding")
        sequence = grant.expected_request.observation_sequence
        previous_request = requests_by_sequence.get(sequence)
        if previous_request is not None and previous_request != grant.expected_request:
            _raise("catalog changes the expected request for an observation sequence")
        requests_by_sequence[sequence] = grant.expected_request
        request_operation = (
            grant.expected_request.request_self_sha256,
            grant.expected_scope.operation,
        )
        previous_binding = bindings_by_request_operation.get(request_operation)
        if previous_binding is not None:
            if previous_binding == binding:
                _raise("catalog contains a duplicate grant")
            _raise("catalog contains more than one grant for a request/operation pair")
        bindings_by_request_operation[request_operation] = binding
        bindings.append(binding)
    return tuple(bindings)


def _validate_legacy_catalog_family(grants: tuple[authority.PermissionApprovalGrant, ...]) -> None:
    """Keep the old outer catalog closed as the issuer learns new operations."""
    for grant in grants:
        if type(grant) is not authority.PermissionApprovalGrant:
            _raise("legacy permission catalog grant has the wrong exact type")
        if (
            grant.schema != authority.APPROVAL_GRANT_SCHEMA
            or grant.expected_scope.schema != observation.OPERATION_SCOPE_SCHEMA
            or grant.expected_scope.operation not in {
                "observe", "observe_authz", "retain_observation",
                "retain_observation_failure", "retain_retention_failure",
            }
            or (grant.expected_destination is not None and
                type(grant.expected_destination) is not evidence.PermissionEvidenceDestination)
        ):
            _raise("legacy permission catalog contains a non-legacy grant")


def canonical_permission_grant_catalog_bytes(
    *,
    enrollment: PermissionApprovalEnrollment,
    grants: tuple[authority.PermissionApprovalGrant, ...],
) -> bytes:
    if type(grants) is not tuple:
        _raise("catalog grants must have the exact tuple type")
    _validate_legacy_catalog_family(grants)
    unsorted_bindings = _validate_catalog_grants(
        grants,
        enrollment=enrollment,
        require_binding_order=False,
    )
    ordered = tuple(
        grant
        for _, grant in sorted(
            zip(unsorted_bindings, grants, strict=True),
            key=lambda item: item[0],
        )
    )
    _validate_catalog_grants(
        ordered,
        enrollment=enrollment,
        require_binding_order=True,
    )
    record = {
        "schema": PERMISSION_GRANT_CATALOG_SCHEMA,
        "enrollment_binding_sha256": permission_approval_enrollment_binding_sha256(
            enrollment
        ),
        "grants": [asdict(grant) for grant in ordered],
    }
    try:
        return canonical_json_bytes(record)
    except Exception as exc:
        _raise("permission grant catalog is not canonical-JSON encodable", exc)


def strict_permission_grant_catalog_loads(
    data: bytes,
    *,
    enrollment: PermissionApprovalEnrollment,
) -> tuple[authority.PermissionApprovalGrant, ...]:
    if type(data) is not bytes:
        _raise("catalog bytes must have the exact bytes type")
    validate_permission_approval_enrollment(enrollment)
    try:
        record = strict_canonical_json_loads(data)
    except Exception as exc:
        _raise("catalog bytes are not canonical JSON", exc)
    record = _mapping(
        record,
        ("schema", "enrollment_binding_sha256", "grants"),
        name="permission grant catalog",
    )
    if record["schema"] != PERMISSION_GRANT_CATALOG_SCHEMA:
        _raise("permission grant catalog schema mismatch")
    binding = _sha(
        record["enrollment_binding_sha256"],
        name="catalog.enrollment_binding_sha256",
    )
    if binding != permission_approval_enrollment_binding_sha256(enrollment):
        _raise("permission grant catalog enrollment binding mismatch")
    grant_values = _array(record["grants"], name="catalog.grants", nonempty=True)
    grants = tuple(
        _grant_from_mapping(value, name=f"catalog.grants[{index}]")
        for index, value in enumerate(grant_values)
    )
    _validate_catalog_grants(
        grants,
        enrollment=enrollment,
        require_binding_order=True,
    )
    if canonical_permission_grant_catalog_bytes(enrollment=enrollment, grants=grants) != data:
        _raise("catalog did not round-trip to its exact canonical bytes")
    return grants


def _validate_authz_retention_catalog_grants(
    grants: tuple[authority.PermissionApprovalGrant, ...],
    *,
    enrollment: PermissionApprovalEnrollment,
    require_binding_order: bool,
) -> tuple[str, ...]:
    if type(grants) is not tuple or len(grants) != 2:
        _raise("Authz retention catalog must contain exactly two grants")
    bindings = _validate_catalog_grants(
        grants,
        enrollment=enrollment,
        require_binding_order=require_binding_order,
    )
    by_operation = {grant.expected_scope.operation: grant for grant in grants}
    if set(by_operation) != {
        "retain_authz_evidence",
        "retain_authz_retention_failure",
    }:
        _raise("Authz retention catalog has the wrong exact operation set")
    primary = by_operation["retain_authz_evidence"]
    sidecar = by_operation["retain_authz_retention_failure"]
    for grant in grants:
        if (
            grant.schema != authority.AUTHZ_RETENTION_APPROVAL_GRANT_SCHEMA
            or grant.expected_scope.schema != authority.AUTHZ_RETENTION_SCOPE_SCHEMA
            or type(grant.expected_destination)
            is not evidence.PermissionAuthzEvidenceDestination
            or type(grant.expected_store_expectation)
            is not evidence.ConfidentialStoreExpectation
        ):
            _raise("Authz retention catalog contains an old or mixed grant")
    if (
        primary.expected_request != sidecar.expected_request
        or primary.expected_access_expectation != sidecar.expected_access_expectation
        or primary.expected_context != sidecar.expected_context
        or primary.expected_source_binding != sidecar.expected_source_binding
        or primary.expected_destination != sidecar.expected_destination
        or primary.expected_store_expectation != sidecar.expected_store_expectation
    ):
        _raise("Authz retention catalog grants do not share one exact binding")
    return bindings


def canonical_permission_authz_retention_catalog_bytes(
    *,
    enrollment: PermissionApprovalEnrollment,
    grants: tuple[authority.PermissionApprovalGrant, ...],
) -> bytes:
    if type(grants) is not tuple:
        _raise("Authz retention catalog grants must have the exact tuple type")
    unsorted = _validate_authz_retention_catalog_grants(
        grants,
        enrollment=enrollment,
        require_binding_order=False,
    )
    ordered = tuple(
        grant
        for _, grant in sorted(
            zip(unsorted, grants, strict=True), key=lambda item: item[0]
        )
    )
    _validate_authz_retention_catalog_grants(
        ordered,
        enrollment=enrollment,
        require_binding_order=True,
    )
    try:
        return canonical_json_bytes(
            {
                "schema": PERMISSION_AUTHZ_RETENTION_CATALOG_SCHEMA,
                "enrollment_binding_sha256": (
                    permission_approval_enrollment_binding_sha256(enrollment)
                ),
                "grants": [asdict(grant) for grant in ordered],
            }
        )
    except Exception as exc:
        _raise("Authz retention catalog is not canonical-JSON encodable", exc)


def strict_permission_authz_retention_catalog_loads(
    data: bytes,
    *,
    enrollment: PermissionApprovalEnrollment,
) -> tuple[authority.PermissionApprovalGrant, ...]:
    if type(data) is not bytes:
        _raise("Authz retention catalog bytes must have the exact bytes type")
    validate_permission_approval_enrollment(enrollment)
    try:
        record = strict_canonical_json_loads(data)
    except Exception as exc:
        _raise("Authz retention catalog bytes are not canonical JSON", exc)
    record = _mapping(
        record,
        ("schema", "enrollment_binding_sha256", "grants"),
        name="Authz retention catalog",
    )
    if record["schema"] != PERMISSION_AUTHZ_RETENTION_CATALOG_SCHEMA:
        _raise("Authz retention catalog schema mismatch")
    binding = _sha(
        record["enrollment_binding_sha256"],
        name="Authz catalog.enrollment_binding_sha256",
    )
    if binding != permission_approval_enrollment_binding_sha256(enrollment):
        _raise("Authz retention catalog enrollment binding mismatch")
    values = _array(record["grants"], name="Authz catalog.grants", nonempty=True)
    grants = tuple(
        _grant_from_mapping(value, name=f"Authz catalog.grants[{index}]")
        for index, value in enumerate(values)
    )
    _validate_authz_retention_catalog_grants(
        grants,
        enrollment=enrollment,
        require_binding_order=True,
    )
    if (
        canonical_permission_authz_retention_catalog_bytes(
            enrollment=enrollment, grants=grants
        )
        != data
    ):
        _raise("Authz retention catalog did not round-trip exactly")
    return grants


def _node_identity(value: object, *, directory: bool) -> _NodeIdentity:
    mode = getattr(value, "st_mode", None)
    if type(mode) is not int:
        _raise("revocation filesystem metadata has no exact mode")
    attributes = getattr(value, "st_file_attributes", 0)
    if type(attributes) is not int:
        _raise("revocation filesystem metadata has no exact attributes")
    if stat.S_ISLNK(mode) or attributes & _REPARSE_POINT:
        _raise("revocation namespace contains a link or reparse point")
    expected_type = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not expected_type:
        kind = "directory" if directory else "regular file"
        _raise(f"revocation filesystem node is not a {kind}")
    device = getattr(value, "st_dev", None)
    inode = getattr(value, "st_ino", None)
    if type(device) is not int or type(inode) is not int:
        _raise("revocation filesystem metadata has no exact identity")
    return _NodeIdentity(device, inode, stat.S_IFMT(mode), attributes)


class _EnrollmentState:
    __slots__ = (
        "authorization_revoked",
        "catalog_identities",
        "catalogs_by_sha256",
        "clock",
        "dependency_in_progress",
        "enrollment",
        "enrollment_binding_sha256",
        "expired",
        "failed",
        "filesystem",
        "grant_by_request_operation",
        "grants_by_binding",
        "grants_by_request_scope",
        "last_clock_utc_ns",
        "lock",
        "request_by_sequence",
        "revocation_directory",
        "revocation_directory_identity",
        "revoked_grant_bindings",
    )

    def __init__(
        self,
        *,
        enrollment: PermissionApprovalEnrollment,
        revocation_directory: Path,
        revocation_directory_identity: _NodeIdentity,
        utc_ns_clock: Callable[[], int],
        filesystem: _RevocationFilesystem,
    ) -> None:
        self.enrollment = enrollment
        self.enrollment_binding_sha256 = permission_approval_enrollment_binding_sha256(
            enrollment
        )
        self.revocation_directory = revocation_directory
        self.revocation_directory_identity = revocation_directory_identity
        self.clock = utc_ns_clock
        self.filesystem = filesystem
        self.lock = RLock()
        self.dependency_in_progress = False
        self.last_clock_utc_ns: int | None = None
        self.failed = False
        self.expired = False
        self.authorization_revoked = False
        self.revoked_grant_bindings: set[str] = set()
        self.grants_by_binding: dict[str, authority.PermissionApprovalGrant] = {}
        self.grants_by_request_scope: dict[tuple[str, str], str] = {}
        self.request_by_sequence: dict[int, observation.PermissionObservationRequest] = {}
        self.grant_by_request_operation: dict[tuple[str, str], str] = {}
        self.catalog_identities: list[PermissionGrantCatalogIdentity] = []
        self.catalogs_by_sha256: dict[str, PermissionGrantCatalogIdentity] = {}

    def _verify_revocation_directory(self) -> None:
        current = _node_identity(
            self.filesystem.lstat(self.revocation_directory),
            directory=True,
        )
        if current != self.revocation_directory_identity:
            _raise("revocation directory identity changed")

    def _marker_exists(self, marker: str) -> bool:
        try:
            self.filesystem.lstat(self.revocation_directory / marker)
        except FileNotFoundError:
            return False
        return True

    def _read_clock_locked(self) -> int | None:
        if self.dependency_in_progress:
            self.failed = True
            return None
        self.dependency_in_progress = True
        try:
            value = self.clock()
        except Exception:
            self.failed = True
            return None
        finally:
            self.dependency_in_progress = False
        if type(value) is not int or value < 0:
            self.failed = True
            return None
        if self.last_clock_utc_ns is not None and value < self.last_clock_utc_ns:
            self.failed = True
            return None
        self.last_clock_utc_ns = value
        if self.failed:
            return None
        return value

    def _inspect_revocation_locked(self, grant_binding_sha256: str) -> None:
        if self.dependency_in_progress:
            self.failed = True
            return
        self.dependency_in_progress = True
        try:
            self._verify_revocation_directory()
            authorization_marker = (
                "authorization-"
                f"{self.enrollment.expected_context.state4_authorization_file_sha256}"
                ".revoked"
            )
            if self._marker_exists(authorization_marker):
                self.authorization_revoked = True
            if self._marker_exists(f"grant-{grant_binding_sha256}.revoked"):
                self.revoked_grant_bindings.add(grant_binding_sha256)
            self._verify_revocation_directory()
        except Exception:
            self.failed = True
        finally:
            self.dependency_in_progress = False

    def runtime_status_locked(self, grant_binding_sha256: str) -> str:
        if self.dependency_in_progress:
            self.failed = True
        if self.failed:
            return "failed"
        if self.expired:
            return "expired"
        if self.authorization_revoked or grant_binding_sha256 in self.revoked_grant_bindings:
            return "revoked"
        first_now = self._read_clock_locked()
        if first_now is None:
            return "failed"
        if first_now >= self.enrollment.expires_at_utc_ns:
            self.expired = True
            return "expired"
        self._inspect_revocation_locked(grant_binding_sha256)
        if self.failed:
            return "failed"
        if self.authorization_revoked or grant_binding_sha256 in self.revoked_grant_bindings:
            return "revoked"
        if first_now < self.enrollment.not_before_utc_ns:
            return "denied"
        final_now = self._read_clock_locked()
        if final_now is None:
            return "failed"
        if final_now >= self.enrollment.expires_at_utc_ns:
            self.expired = True
            return "expired"
        if final_now < self.enrollment.not_before_utc_ns:
            return "denied"
        if self.failed:
            return "failed"
        if self.expired:
            return "expired"
        if self.authorization_revoked or grant_binding_sha256 in self.revoked_grant_bindings:
            return "revoked"
        return "active"


class EnrolledPermissionApprovalResolver:
    """Operational resolver capability with no catalog-registration method."""

    __slots__ = ("_state",)

    def __init__(self, state: _EnrollmentState) -> None:
        self._state = state

    def resolve(
        self,
        *,
        request: observation.PermissionObservationRequest,
        scope: observation.PermissionOperationScope,
    ) -> authority.PermissionApprovalResolution:
        try:
            observation.validate_permission_observation_request(
                request,
                expected_request=request,
            )
            authority.validate_permission_operation_scope(scope, request=request)
        except Exception:
            return authority.PermissionApprovalResolution("denied", None)
        request_scope = (request.request_self_sha256, scope.scope_self_sha256)
        with self._state.lock:
            if self._state.dependency_in_progress:
                self._state.failed = True
                return authority.PermissionApprovalResolution("failed", None)
            binding = self._state.grants_by_request_scope.get(request_scope)
            if binding is None:
                return authority.PermissionApprovalResolution("denied", None)
            grant = self._state.grants_by_binding.get(binding)
            if (
                grant is None
                or grant.expected_request != request
                or grant.expected_scope != scope
            ):
                return authority.PermissionApprovalResolution("denied", None)
            status = self._state.runtime_status_locked(binding)
            if status == "active":
                return authority.PermissionApprovalResolution("approved", grant)
            return authority.PermissionApprovalResolution(status, None)

    def check_active(
        self,
        *,
        grant: authority.PermissionApprovalGrant,
        checkpoint: observation.PermissionAuthorityCheckpoint,
    ) -> authority.PermissionApprovalCheckOutcome:
        try:
            binding = authority.permission_approval_grant_binding_sha256(grant)
            authority._validate_checkpoint_shape(
                checkpoint,
                expected_checkpoints=authority._expected_checkpoints(grant),
            )
        except Exception as exc:
            _raise("active check grant or checkpoint is malformed", exc)
        try:
            _validate_grant_against_enrollment(
                grant,
                enrollment=self._state.enrollment,
            )
        except PermissionApprovalResolverError:
            return authority.PermissionApprovalCheckOutcome(
                "denied",
                binding,
                checkpoint,
            )
        with self._state.lock:
            if self._state.dependency_in_progress:
                self._state.failed = True
                status = "failed"
            else:
                enrolled = self._state.grants_by_binding.get(binding)
                if enrolled != grant:
                    status = "denied"
                else:
                    status = self._state.runtime_status_locked(binding)
            return authority.PermissionApprovalCheckOutcome(
                status,
                binding,
                checkpoint,
            )


class PermissionGrantEnrollmentHandle:
    """Trusted administrative catalog-registration capability."""

    __slots__ = ("_state",)

    def __init__(self, state: _EnrollmentState) -> None:
        self._state = state

    @property
    def catalog_identities(self) -> tuple[PermissionGrantCatalogIdentity, ...]:
        with self._state.lock:
            return tuple(self._state.catalog_identities)

    def register_catalog(
        self,
        catalog_bytes: bytes,
        *,
        expected_file_bytes: int,
        expected_file_sha256: str,
    ) -> PermissionGrantCatalogIdentity:
        if type(catalog_bytes) is not bytes:
            _raise("catalog bytes must have the exact bytes type")
        expected_length = _integer(
            expected_file_bytes,
            name="expected catalog file bytes",
            minimum=1,
        )
        expected_sha = _sha(
            expected_file_sha256,
            name="expected catalog file SHA-256",
        )
        if len(catalog_bytes) != expected_length:
            _raise("catalog byte length differs from independent expectation")
        actual_sha = sha256_bytes(catalog_bytes)
        if actual_sha != expected_sha:
            _raise("catalog SHA-256 differs from independent expectation")
        grants = strict_permission_grant_catalog_loads(
            catalog_bytes,
            enrollment=self._state.enrollment,
        )
        return self._register_validated_catalog(
            grants=grants,
            expected_length=expected_length,
            expected_sha=expected_sha,
        )

    def register_authz_retention_catalog(
        self,
        catalog_bytes: bytes,
        *,
        expected_file_bytes: int,
        expected_file_sha256: str,
    ) -> PermissionGrantCatalogIdentity:
        if type(catalog_bytes) is not bytes:
            _raise("Authz catalog bytes must have the exact bytes type")
        expected_length = _integer(
            expected_file_bytes,
            name="expected Authz catalog file bytes",
            minimum=1,
        )
        expected_sha = _sha(
            expected_file_sha256,
            name="expected Authz catalog file SHA-256",
        )
        if len(catalog_bytes) != expected_length:
            _raise("Authz catalog byte length differs from independent expectation")
        if sha256_bytes(catalog_bytes) != expected_sha:
            _raise("Authz catalog SHA-256 differs from independent expectation")
        grants = strict_permission_authz_retention_catalog_loads(
            catalog_bytes,
            enrollment=self._state.enrollment,
        )
        return self._register_validated_catalog(
            grants=grants,
            expected_length=expected_length,
            expected_sha=expected_sha,
        )

    def _register_validated_catalog(
        self,
        *,
        grants: tuple[authority.PermissionApprovalGrant, ...],
        expected_length: int,
        expected_sha: str,
    ) -> PermissionGrantCatalogIdentity:
        bindings = tuple(
            authority.permission_approval_grant_binding_sha256(grant)
            for grant in grants
        )
        identity = PermissionGrantCatalogIdentity(
            enrollment_binding_sha256=self._state.enrollment_binding_sha256,
            catalog_file_bytes=expected_length,
            catalog_file_sha256=expected_sha,
            grant_binding_sha256s=bindings,
        )
        with self._state.lock:
            if self._state.dependency_in_progress:
                self._state.failed = True
                _raise("catalog registration reentered a runtime dependency")
            prior_catalog = self._state.catalogs_by_sha256.get(expected_sha)
            if prior_catalog is not None:
                if prior_catalog == identity:
                    return prior_catalog
                _raise("catalog identity conflicts with an existing registration")

            for grant, binding in zip(grants, bindings, strict=True):
                existing_grant = self._state.grants_by_binding.get(binding)
                if existing_grant is not None and existing_grant != grant:
                    _raise("catalog grant binding conflicts with an enrolled grant")
                sequence = grant.expected_request.observation_sequence
                prior_request = self._state.request_by_sequence.get(sequence)
                if prior_request is not None and prior_request != grant.expected_request:
                    _raise("catalog changes the enrolled request for an observation sequence")
                request_operation = (
                    grant.expected_request.request_self_sha256,
                    grant.expected_scope.operation,
                )
                prior_operation = self._state.grant_by_request_operation.get(
                    request_operation
                )
                if prior_operation is not None and prior_operation != binding:
                    _raise("catalog replaces the grant for a request/operation pair")
                request_scope = (
                    grant.expected_request.request_self_sha256,
                    grant.expected_scope.scope_self_sha256,
                )
                prior_scope = self._state.grants_by_request_scope.get(request_scope)
                if prior_scope is not None and prior_scope != binding:
                    _raise("catalog scope overlaps a different enrolled grant")

            for grant, binding in zip(grants, bindings, strict=True):
                self._state.grants_by_binding.setdefault(binding, grant)
                self._state.request_by_sequence.setdefault(
                    grant.expected_request.observation_sequence,
                    grant.expected_request,
                )
                self._state.grant_by_request_operation.setdefault(
                    (
                        grant.expected_request.request_self_sha256,
                        grant.expected_scope.operation,
                    ),
                    binding,
                )
                self._state.grants_by_request_scope.setdefault(
                    (
                        grant.expected_request.request_self_sha256,
                        grant.expected_scope.scope_self_sha256,
                    ),
                    binding,
                )
            self._state.catalogs_by_sha256[expected_sha] = identity
            self._state.catalog_identities.append(identity)
            return identity


def build_enrolled_permission_approval_resolver(
    *,
    enrollment: PermissionApprovalEnrollment,
    revocation_directory: str | os.PathLike[str],
    utc_ns_clock: Callable[[], int],
    _filesystem: _RevocationFilesystem | None = None,
) -> tuple[EnrolledPermissionApprovalResolver, PermissionGrantEnrollmentHandle]:
    """Build separated operational and administrative capabilities.

    The caller owns authentication, catalog-byte retention, directory
    protection, and production rollback controls.  No clock read, registration,
    activation, or allowance consumption occurs during construction.
    """

    validate_permission_approval_enrollment(enrollment)
    if not callable(utc_ns_clock):
        _raise("an explicit callable UTC-nanosecond clock is mandatory")
    try:
        supplied = Path(os.fspath(revocation_directory))
    except (TypeError, ValueError) as exc:
        _raise("an explicit revocation-directory path is mandatory", exc)
    if not supplied.is_absolute():
        _raise("revocation directory must be an absolute path")
    directory = Path(os.path.abspath(supplied))
    filesystem = _filesystem if _filesystem is not None else _OSRevocationFilesystem()
    if not callable(getattr(filesystem, "lstat", None)):
        _raise("revocation filesystem lacks exact lstat inspection")
    try:
        identity = _node_identity(filesystem.lstat(directory), directory=True)
    except Exception as exc:
        _raise("revocation directory must be an existing non-reparse directory", exc)
    state = _EnrollmentState(
        enrollment=enrollment,
        revocation_directory=directory,
        revocation_directory_identity=identity,
        utc_ns_clock=utc_ns_clock,
        filesystem=filesystem,
    )
    return EnrolledPermissionApprovalResolver(state), PermissionGrantEnrollmentHandle(state)
