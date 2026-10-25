"""Pure verification of supplied retained private-permission evidence.

This module verifies correspondence within caller-supplied evidence.  A receipt
self-hash is not authentication, proof of authority accounting, or a fresh
physical readback.  The trusted caller, and any later live integration, remain
responsible for the provenance and freshness of every supplied value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import unicodedata
from typing import Literal

from operatormark.corpus import permission_observation_windows as _observation
from operatormark.corpus import private_permission_evidence as _evidence
from operatormark.corpus import storage as _storage
from operatormark.corpus.canonical import (
    canonical_json_bytes as _canonical_json_bytes,
    require_sha256 as _require_sha256,
    sha256_bytes as _sha256_bytes,
    validate_self_hash as _validate_self_hash,
)


__all__ = [
    "PermissionEvidenceAssessment",
    "PermissionEvidenceVerificationError",
    "verify_retained_permission_observation",
]


_RECEIPT_SCHEMA = (
    "operatormark.darcy_branch_c_v1.retained_private_permission_evidence.v2"
)
_CAPTURE_ONLY_PUBLICATION_STATE = (
    "retained_capture_only_not_permission_accepted"
)
_FILE_PROHIBITED_MASK = (
    _storage.STATE4_FILE_PERMISSION_WRITE_DATA
    | _storage.STATE4_FILE_PERMISSION_APPEND_DATA
    | _storage.STATE4_FILE_PERMISSION_DELETE
    | _storage.STATE4_FILE_PERMISSION_WRITE_EA
    | _storage.STATE4_FILE_PERMISSION_WRITE_ATTRIBUTES
    | _storage.STATE4_FILE_PERMISSION_WRITE_DAC
    | _storage.STATE4_FILE_PERMISSION_WRITE_OWNER
)
_OPEN_BOOLEAN_FIELDS = (
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


class PermissionEvidenceVerificationError(ValueError):
    """Supplied evidence is malformed, unsupported, or contradicts an expectation."""


@dataclass(frozen=True, slots=True)
class PermissionEvidenceAssessment:
    """Redacted facts established by one validated supplied-evidence bundle.

    ``current_token_mask_conditions_satisfied`` is deliberately narrower than
    permission acceptance.  It covers only the recorded current-token masks;
    it says nothing about other principals, privilege bypass, freshness, future
    changes, or production readiness.
    """

    observation_file_bytes: int
    observation_file_sha256: str
    observation_record_self_sha256: str
    subject_file_bytes: int
    subject_file_sha256: str
    request_self_sha256: str
    receipt_self_sha256: str
    private_observation_commitment: str
    owning_principal_commitment: str
    current_principal_commitment: str
    security_descriptor_commitment: str
    recorded_file_access_status: Literal["granted", "denied", "unavailable"]
    recorded_file_access_mask_hex: str | None
    recorded_immediate_parent_access_status: Literal[
        "granted", "denied", "unavailable"
    ]
    recorded_immediate_parent_access_mask_hex: str | None
    captured_file_has_read_only_attribute: bool
    current_token_file_read_data_present: bool | None
    current_token_file_prohibited_rights_absent: bool | None
    current_token_immediate_parent_delete_child_absent: bool | None
    current_token_mask_assessment_status: Literal[
        "satisfied", "not_satisfied", "not_evaluated"
    ]
    current_token_mask_conditions_satisfied: bool | None
    broad_principal_policy_status: Literal["not_evaluated"] = field(
        init=False, default="not_evaluated"
    )
    permission_acceptance_emitted: Literal[False] = field(init=False, default=False)


def _raise(message: str) -> None:
    raise PermissionEvidenceVerificationError(message)


def _require_exact_root(value: object, expected_type: type, *, name: str) -> None:
    if type(value) is not expected_type:
        _raise(f"{name} has an unsupported representation")


def _require_exact_scalar_tree(value: object, *, name: str) -> None:
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _require_exact_scalar_tree(
                getattr(value, item.name), name=f"{name}.{item.name}"
            )
        return
    if type(value) is tuple:
        for index, item in enumerate(value):
            _require_exact_scalar_tree(item, name=f"{name}[{index}]")
        return
    if type(value) in {str, int, bool, type(None)}:
        return
    _raise(f"{name} has an unsupported scalar representation")


def _require_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        _raise(f"{name} must be an exact nonnegative integer")
    return value


def _require_text(value: object, *, name: str, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        _raise(f"{name} must be exact text")
    if unicodedata.normalize("NFC", value) != value:
        _raise(f"{name} must be NFC text")
    return value


def _require_sha(value: object, *, name: str) -> str:
    if type(value) is not str:
        _raise(f"{name} must be exact SHA-256 text")
    try:
        return _require_sha256(value, name=name)
    except Exception as exc:
        raise PermissionEvidenceVerificationError(
            f"{name} is not lowercase SHA-256 hexadecimal"
        ) from exc


def _require_open_boolean_types(
    specification: _observation.NativeOpenSpecification, *, name: str
) -> None:
    _require_exact_root(
        specification, _observation.NativeOpenSpecification, name=name
    )
    for field_name in _OPEN_BOOLEAN_FIELDS:
        if type(getattr(specification, field_name)) is not bool:
            _raise(f"{name}.{field_name} must be an exact Boolean")


def _validate_expected_inputs(
    expected_request: _observation.PermissionObservationRequest,
    expected_access: _observation.PermissionAccessExpectation,
    expected_store: _evidence.ConfidentialStoreExpectation,
    expected_destination: _evidence.PermissionEvidenceDestination,
    expected_retained_snapshot: _evidence.StoreObjectSnapshot,
) -> None:
    _require_exact_root(
        expected_request,
        _observation.PermissionObservationRequest,
        name="expected_request",
    )
    _require_exact_root(
        expected_request.subject,
        _observation.PermissionObservationSubject,
        name="expected_request.subject",
    )
    _require_exact_root(
        expected_access,
        _observation.PermissionAccessExpectation,
        name="expected_access",
    )
    _require_exact_root(
        expected_store,
        _evidence.ConfidentialStoreExpectation,
        name="expected_store",
    )
    _require_exact_root(
        expected_destination,
        _evidence.PermissionEvidenceDestination,
        name="expected_destination",
    )
    _require_exact_root(
        expected_retained_snapshot,
        _evidence.StoreObjectSnapshot,
        name="expected_retained_snapshot",
    )
    for name, value in (
        ("expected_request", expected_request),
        ("expected_access", expected_access),
        ("expected_store", expected_store),
        ("expected_destination", expected_destination),
        ("expected_retained_snapshot", expected_retained_snapshot),
    ):
        _require_exact_scalar_tree(value, name=name)

    row_index = expected_request.subject.row_index
    if row_index is not None and (type(row_index) is not int or row_index < 0):
        _raise("expected_request subject row_index has an invalid exact type")
    _require_open_boolean_types(
        expected_access.subject_open_specification,
        name="expected_access.subject_open_specification",
    )
    _require_open_boolean_types(
        expected_access.ancestor_open_specification,
        name="expected_access.ancestor_open_specification",
    )
    _require_nonnegative_int(
        expected_destination.observation_sequence,
        name="expected_destination.observation_sequence",
    )
    _require_nonnegative_int(
        expected_retained_snapshot.size_bytes,
        name="expected_retained_snapshot.size_bytes",
    )

    try:
        _observation.validate_permission_observation_request(
            expected_request, expected_request=expected_request
        )
        _observation.validate_permission_access_expectation(
            expected_access, expected_request=expected_request
        )
        _evidence.validate_confidential_store_expectation(
            expected_store, expected_request=expected_request
        )
        _evidence.validate_permission_evidence_destination(
            expected_destination,
            expected_request=expected_request,
            expected_store=expected_store,
        )
    except Exception as exc:
        raise PermissionEvidenceVerificationError(
            "independent expected permission evidence is invalid"
        ) from exc

    if expected_destination.payload_kind != "observation":
        _raise("expected destination is not for a captured observation")
    try:
        _evidence._validate_object_snapshot(
            expected_retained_snapshot,
            expected_destination,
            permitted_paths={expected_destination.primary_accepted_relative_path},
        )
    except Exception as exc:
        raise PermissionEvidenceVerificationError(
            "independent retained snapshot is invalid"
        ) from exc
    snapshot_expected = {
        "relative_path": expected_destination.primary_accepted_relative_path,
        "namespace_object_identity": expected_store.namespace_object_identity,
        "parent_relative_path": expected_store.records_parent_relative_path,
        "parent_object_identity": expected_store.records_parent_object_identity,
        "volume_identifier_commitment": expected_store.volume_identifier_commitment,
        "security_descriptor_sha256": (
            expected_store.required_created_object_security_descriptor_sha256
        ),
        "confidential_store_profile_sha256": (
            expected_store.confidential_store_profile_sha256
        ),
        "publication_state": "published",
    }
    for field_name, expected in snapshot_expected.items():
        if getattr(expected_retained_snapshot, field_name) != expected:
            _raise(f"independent retained snapshot {field_name} mismatch")
    _require_text(
        expected_retained_snapshot.object_identity,
        name="expected_retained_snapshot.object_identity",
    )
    _require_sha(
        expected_retained_snapshot.file_sha256,
        name="expected_retained_snapshot.file_sha256",
    )


def _strict_observation(
    observation_bytes: bytes,
    *,
    expected_request: _observation.PermissionObservationRequest,
    expected_access: _observation.PermissionAccessExpectation,
) -> dict[str, object]:
    _require_exact_root(observation_bytes, bytes, name="observation_bytes")
    try:
        record = _evidence.strict_private_permission_observation_loads(
            observation_bytes,
            expected_request=expected_request,
            expected_access=expected_access,
        )
    except Exception as exc:
        raise PermissionEvidenceVerificationError(
            "canonical captured observation evidence is invalid"
        ) from exc
    if _canonical_json_bytes(record["request"]) != _canonical_json_bytes(
        asdict(expected_request)
    ):
        _raise("captured observation request representation mismatch")
    if _canonical_json_bytes(record["access_expectation"]) != _canonical_json_bytes(
        asdict(expected_access)
    ):
        _raise("captured observation access expectation representation mismatch")
    content = record["content_evidence"]
    if type(content["file_bytes"]) is not int:
        _raise("captured observation content_evidence.file_bytes must be an exact integer")
    for field_name in (
        "subject_security_before",
        "subject_security_after",
        "ancestry_security_before",
        "ancestry_security_after",
    ):
        security_values = record[field_name]
        if field_name.startswith("subject_"):
            security_values = (security_values,)
        for descriptor_index, security_value in enumerate(security_values):
            aces = security_value["descriptor"]["ordered_dacl_aces"]
            for ace_position, ace in enumerate(aces):
                _require_nonnegative_int(
                    ace["ace_index"],
                    name=(
                        f"captured observation {field_name}[{descriptor_index}]"
                        f".ordered_dacl_aces[{ace_position}].ace_index"
                    ),
                )
    return record


def _validate_snapshot_content(
    snapshot: _evidence.StoreObjectSnapshot, observation_bytes: bytes
) -> tuple[int, str]:
    size = len(observation_bytes)
    digest = _sha256_bytes(observation_bytes)
    if snapshot.size_bytes != size or snapshot.file_sha256 != digest:
        _raise("independent retained snapshot does not bind the supplied observation bytes")
    return size, digest


def _receipt_mapping(
    receipt: _evidence.RetainedPermissionEvidence,
) -> dict[str, object]:
    _require_exact_root(
        receipt, _evidence.RetainedPermissionEvidence, name="receipt"
    )
    _require_exact_scalar_tree(receipt, name="receipt")
    _require_nonnegative_int(receipt.file_bytes, name="receipt.file_bytes")
    for item in fields(receipt):
        if item.name != "file_bytes":
            _require_text(getattr(receipt, item.name), name=f"receipt.{item.name}")
    for field_name in (
        "request_self_sha256",
        "access_expectation_self_sha256",
        "destination_self_sha256",
        "store_expectation_self_sha256",
        "file_sha256",
        "authorization_file_sha256",
        "private_observation_commitment",
        "creation_security_descriptor_sha256",
        "confidential_store_profile_sha256",
        "held_object_readback_sha256",
        "receipt_self_sha256",
    ):
        _require_sha(getattr(receipt, field_name), name=f"receipt.{field_name}")
    record = asdict(receipt)
    try:
        _validate_self_hash(record, "receipt_self_sha256")
    except Exception as exc:
        raise PermissionEvidenceVerificationError(
            "receipt self-hash mismatch"
        ) from exc
    return record


def _validate_receipt_bindings(
    receipt: _evidence.RetainedPermissionEvidence,
    *,
    observation_bytes: bytes,
    observation_size: int,
    observation_sha256: str,
    expected_request: _observation.PermissionObservationRequest,
    expected_access: _observation.PermissionAccessExpectation,
    expected_store: _evidence.ConfidentialStoreExpectation,
    expected_destination: _evidence.PermissionEvidenceDestination,
    expected_retained_snapshot: _evidence.StoreObjectSnapshot,
) -> None:
    _receipt_mapping(receipt)
    authorization_sha256 = expected_request.context.state4_authorization_file_sha256
    try:
        commitment = _evidence.private_permission_observation_commitment(
            observation_bytes,
            authorization_file_sha256=authorization_sha256,
            payload_kind="observation",
            expected_request=expected_request,
            expected_access=expected_access,
            expected_store=None,
            expected_destination=None,
        )
    except Exception as exc:
        raise PermissionEvidenceVerificationError(
            "private observation commitment could not be verified"
        ) from exc
    expected = {
        "schema": _RECEIPT_SCHEMA,
        "payload_kind": "observation",
        "request_self_sha256": expected_request.request_self_sha256,
        "access_expectation_self_sha256": expected_access.expectation_self_sha256,
        "destination_self_sha256": expected_destination.destination_self_sha256,
        "store_expectation_self_sha256": expected_store.expectation_self_sha256,
        "relative_path": expected_destination.primary_accepted_relative_path,
        "file_bytes": observation_size,
        "file_sha256": observation_sha256,
        "commitment_domain": _evidence.OBSERVATION_DOMAIN,
        "authorization_file_sha256": authorization_sha256,
        "private_observation_commitment": commitment,
        "namespace_object_identity": expected_store.namespace_object_identity,
        "records_parent_object_identity": expected_store.records_parent_object_identity,
        "store_object_identity": expected_retained_snapshot.object_identity,
        "creation_security_descriptor_sha256": (
            expected_store.required_created_object_security_descriptor_sha256
        ),
        "confidential_store_profile_sha256": (
            expected_store.confidential_store_profile_sha256
        ),
        "held_object_readback_sha256": observation_sha256,
        "publication_state": _CAPTURE_ONLY_PUBLICATION_STATE,
    }
    if receipt.commitment_domain != (
        _storage.STATE4_FILE_PERMISSION_METADATA_PRIVATE_OBSERVATION_DOMAIN
    ):
        _raise("receipt commitment domain is not the storage observation domain")
    for field_name, expected_value in expected.items():
        if getattr(receipt, field_name) != expected_value:
            _raise(f"receipt {field_name} mismatch")


def _bound_commitment(domain: str, authorization_sha256: str, preimage: bytes) -> str:
    return _sha256_bytes(
        domain.encode("utf-8")
        + b"\n"
        + authorization_sha256.encode("ascii")
        + b"\n"
        + preimage
    )


def _recorded_mask(
    raw_result: dict[str, object],
) -> tuple[Literal["granted", "denied", "unavailable"], str | None, int | None]:
    if raw_result["api_call_status"] == "failed":
        return "unavailable", None, None
    status = raw_result["access_status"]
    mask_hex = raw_result["granted_access_mask_hex"]
    if status not in {"granted", "denied"} or type(mask_hex) is not str:
        _raise("validated access evidence has an unsupported result")
    return status, mask_hex, int(mask_hex, 16)


def _assessment(
    record: dict[str, object],
    *,
    receipt: _evidence.RetainedPermissionEvidence,
    observation_size: int,
    observation_sha256: str,
    expected_request: _observation.PermissionObservationRequest,
) -> PermissionEvidenceAssessment:
    content = record["content_evidence"]
    subject = record["subject_before"]
    security = record["subject_security_before"]
    effective = record["effective_context_before"]
    access_checks = record["access_checks"]
    if not all(
        type(value) is dict
        for value in (content, subject, security, effective)
    ) or type(access_checks) is not list:
        _raise("validated observation projection has an unsupported representation")
    descriptor = security["descriptor"]
    subject_facts = subject["facts"]
    effective_facts = effective["facts"]
    if not all(
        type(value) is dict
        for value in (descriptor, subject_facts, effective_facts)
    ):
        _raise("validated observation facts have an unsupported representation")
    owner_sid = descriptor["owner_sid"]
    current_user_sid = effective_facts["user"]
    if type(owner_sid) is not dict or type(current_user_sid) is not dict:
        _raise("validated principal evidence has an unsupported representation")
    authorization_sha256 = expected_request.context.state4_authorization_file_sha256
    descriptor_bytes = bytes.fromhex(descriptor["self_relative_descriptor_hex"])
    owner_preimage = owner_sid["sid_string"].encode("utf-8")
    current_preimage = current_user_sid["sid_string"].encode("utf-8")
    file_raw = access_checks[0]["raw_result"]
    parent_raw = access_checks[1]["raw_result"]
    if type(file_raw) is not dict or type(parent_raw) is not dict:
        _raise("validated access-mask evidence has an unsupported representation")
    file_status, file_mask_hex, file_mask = _recorded_mask(file_raw)
    parent_status, parent_mask_hex, parent_mask = _recorded_mask(parent_raw)
    file_read = (
        None
        if file_mask is None
        else bool(file_mask & _storage.STATE4_FILE_PERMISSION_READ_DATA)
    )
    file_prohibited_absent = (
        None if file_mask is None else not bool(file_mask & _FILE_PROHIBITED_MASK)
    )
    parent_delete_absent = (
        None
        if parent_mask is None
        else not bool(parent_mask & _storage.STATE4_PARENT_PERMISSION_DELETE_CHILD)
    )
    conditions = (file_read, file_prohibited_absent, parent_delete_absent)
    satisfied = None if any(value is None for value in conditions) else all(conditions)
    status = (
        "not_evaluated"
        if satisfied is None
        else "satisfied"
        if satisfied
        else "not_satisfied"
    )
    return PermissionEvidenceAssessment(
        observation_file_bytes=observation_size,
        observation_file_sha256=observation_sha256,
        observation_record_self_sha256=record["record_self_sha256"],
        subject_file_bytes=content["file_bytes"],
        subject_file_sha256=content["file_sha256"],
        request_self_sha256=expected_request.request_self_sha256,
        receipt_self_sha256=receipt.receipt_self_sha256,
        private_observation_commitment=receipt.private_observation_commitment,
        owning_principal_commitment=_bound_commitment(
            _storage.STATE4_FILE_PERMISSION_METADATA_OWNER_DOMAIN,
            authorization_sha256,
            owner_preimage,
        ),
        current_principal_commitment=_bound_commitment(
            _storage.STATE4_FILE_PERMISSION_METADATA_CURRENT_PRINCIPAL_DOMAIN,
            authorization_sha256,
            current_preimage,
        ),
        security_descriptor_commitment=_bound_commitment(
            _storage.STATE4_FILE_PERMISSION_METADATA_SECURITY_DESCRIPTOR_DOMAIN,
            authorization_sha256,
            descriptor_bytes,
        ),
        recorded_file_access_status=file_status,
        recorded_file_access_mask_hex=file_mask_hex,
        recorded_immediate_parent_access_status=parent_status,
        recorded_immediate_parent_access_mask_hex=parent_mask_hex,
        captured_file_has_read_only_attribute=bool(
            int(subject_facts["file_attributes_hex"], 16)
            & _storage.STATE4_FILE_ATTRIBUTE_READONLY
        ),
        current_token_file_read_data_present=file_read,
        current_token_file_prohibited_rights_absent=file_prohibited_absent,
        current_token_immediate_parent_delete_child_absent=parent_delete_absent,
        current_token_mask_assessment_status=status,
        current_token_mask_conditions_satisfied=satisfied,
    )


def verify_retained_permission_observation(
    observation_bytes: bytes,
    receipt: _evidence.RetainedPermissionEvidence,
    *,
    expected_request: _observation.PermissionObservationRequest,
    expected_access: _observation.PermissionAccessExpectation,
    expected_store: _evidence.ConfidentialStoreExpectation,
    expected_destination: _evidence.PermissionEvidenceDestination,
    expected_retained_snapshot: _evidence.StoreObjectSnapshot,
) -> PermissionEvidenceAssessment:
    """Verify a supplied retained captured-observation evidence bundle.

    Verification is pure and in-memory.  It authenticates no caller, performs
    no authority accounting, and does not freshly read the retained object.
    Provenance and any later live qualification remain caller responsibilities.
    """

    _require_exact_root(observation_bytes, bytes, name="observation_bytes")
    _validate_expected_inputs(
        expected_request,
        expected_access,
        expected_store,
        expected_destination,
        expected_retained_snapshot,
    )
    record = _strict_observation(
        observation_bytes,
        expected_request=expected_request,
        expected_access=expected_access,
    )
    observation_size, observation_sha256 = _validate_snapshot_content(
        expected_retained_snapshot, observation_bytes
    )
    _validate_receipt_bindings(
        receipt,
        observation_bytes=observation_bytes,
        observation_size=observation_size,
        observation_sha256=observation_sha256,
        expected_request=expected_request,
        expected_access=expected_access,
        expected_store=expected_store,
        expected_destination=expected_destination,
        expected_retained_snapshot=expected_retained_snapshot,
    )
    return _assessment(
        record,
        receipt=receipt,
        observation_size=observation_size,
        observation_sha256=observation_sha256,
        expected_request=expected_request,
    )
