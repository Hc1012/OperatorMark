"""Closed additive administrative contracts. No acquisition or authentication.

An AcceptedAdministrativePin is supplied through independent trusted custody.
These validators check correspondence, not the identity of an approving human.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import PureWindowsPath
import re
import stat
from typing import Literal

from .canonical import canonical_json_bytes, require_sha256, sha256_bytes, strict_canonical_json_loads
from . import permission_handoff_records as old
from . import permission_handoff_sources as sources
from . import permission_handoff_windows as windows
from .permission_handoff_verify import HandoffVerification
from .permission_handoff_sizing import validate_handoff_physical_inventory
from .permission_authority_journal import _NodeIdentity

PREFIX = "operatormark.darcy_branch_c_v1.handoff_administration."
KINDS = ("policy", "grant", "restart_ticket", "activation", "reservation", "checkpoint", "terminal", "restart_report")
ADDITIONAL_SOURCE_PATHS = tuple("operatormark/corpus/permission_handoff_" + name + ".py" for name in (
    "administration_records", "administration", "administration_journal", "administration_windows", "restart", "deployment"))
DEPLOYMENT_SOURCE_PATHS = (*old.HANDOFF_SOURCE_RELATIVE_PATHS,
    "operatormark/corpus/permission_handoff_windows.py", "operatormark/corpus/permission_handoff_reader_windows.py",
    "operatormark/corpus/permission_handoff_witness_windows.py", *ADDITIONAL_SOURCE_PATHS)
READ_CHECKPOINTS = ("initial", "before_open", "before_bytes", "final")
WRITE_CHECKPOINTS = ("initial", "before_open", "before_bytes", "before_publish", "final")
RESTART_WITNESSES = tuple(reversed(old.WITNESS_FILENAMES))
REASONS = ("bootstrap_unavailable", "ticket_mismatch", "source_binding_mismatch", "restart_authority_unconfirmed",
    "witness_unavailable", "witness_chain_mismatch", "historical_verification_incomplete", "fresh_input_unavailable",
    "fresh_content_mismatch", "fresh_physical_mismatch", "capture_authority_unavailable", "capture_authority_mismatch",
    "legacy_authority_not_recorded", "original_acknowledgment_unavailable", "cleanup_failed", "size_contribution_unresolved")


class HandoffAdministrationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AccountingRootBinding:
    root_path: str
    root_identity: _NodeIdentity


@dataclass(frozen=True, slots=True)
class AcceptedAdministrativePin:
    acceptance_reference: str
    accepted_kind: Literal["policy", "restart_ticket"]
    file_ref: old.FileRef
    bootstrap_root: windows.HandoffWindowsRootBinding
    bootstrap_file_identity: old.observation.NativeFileIdentity
    bootstrap_file_security_descriptor_sha256: str
    host_profile_file_ref: old.FileRef
    accounting_root_binding: AccountingRootBinding


@dataclass(frozen=True, slots=True)
class AdministrativePhysicalPolicy:
    root: windows.HandoffWindowsRootBinding
    namespace_relative_path: str
    parent_relative_path: str
    expected_file_identity: old.observation.NativeFileIdentity | None
    security_descriptor_sha256: str
    confidential_profile_file_sha256: str | None


@dataclass(frozen=True, slots=True)
class AdministrativeDependency:
    role: str
    file_ref: old.FileRef
    physical_policy: AdministrativePhysicalPolicy


@dataclass(frozen=True, slots=True)
class AdministrativeRevocationPolicy:
    root: windows.HandoffWindowsRootBinding
    namespace_relative_path: str
    parent_relative_path: str
    security_descriptor_sha256: str
    confidential_profile_file_sha256: str


@dataclass(frozen=True, slots=True)
class AdministrativeJournalPolicy:
    root_path: str
    root_identity: _NodeIdentity
    namespace_profile_file_ref: old.FileRef
    record_security_policy_file_ref: old.FileRef
    storage_owner: Literal["separate_B"]


@dataclass(frozen=True, slots=True)
class AdministrativeIOLimits:
    maximum_administrative_record_bytes: int
    maximum_opaque_input_bytes: int
    transfer_chunk_bytes: int


@dataclass(frozen=True, slots=True)
class HandoffAdministrativeGrant:
    schema: str
    purpose: Literal["capture", "restart"]
    full_state4_authorization_file_sha256: str
    original_plan_self_sha256: str
    request_self_sha256: str
    original_invocation_id: str
    restart_ticket_identity: str | None
    operation: Literal["read_handoff_input", "retain_handoff_witness", "read_handoff_verification_input"]
    slot_id: str
    root_binding_id: str
    relative_path: str
    creating_relative_path: str | None
    legacy_scope_self_sha256: str | None
    physical_policy: AdministrativePhysicalPolicy
    expected_file_ref: old.FileRef | None
    not_before_utc_ns: int
    expires_at_utc_ns: int
    allowance_id: str
    record_self_sha256: str


@dataclass(frozen=True, slots=True)
class HandoffAdministrativePolicy:
    schema: str
    acceptance_reference: str
    full_state4_authorization_file_sha256: str
    original_plan_file_ref: old.FileRef
    original_plan_self_sha256: str
    original_invocation_id: str
    host_profile_file_ref: old.FileRef
    configuration: windows.HandoffWindowsConfiguration
    deployment_source_files: tuple[old.HandoffSourceFile, ...]
    dependency_files: tuple[AdministrativeDependency, ...]
    revocation_policy: AdministrativeRevocationPolicy
    journal_policy: AdministrativeJournalPolicy
    grants: tuple[HandoffAdministrativeGrant, ...]
    not_before_utc_ns: int
    expires_at_utc_ns: int
    io_limits: AdministrativeIOLimits
    record_self_sha256: str


@dataclass(frozen=True, slots=True)
class HandoffRestartInput:
    slot_id: str
    role: Literal["witness", "original_input", "capture_administration", "dependency"]
    root_binding_id: str
    relative_path: str
    physical_policy: AdministrativePhysicalPolicy
    expected_file_ref: old.FileRef | None
    original_slot_id: str | None


@dataclass(frozen=True, slots=True)
class HandoffRestartTicket:
    schema: str
    acceptance_reference: str
    verification_id: str
    verification_mode: Literal["administrative_corroboration", "legacy_review"]
    original_plan_file_ref: old.FileRef
    original_plan_self_sha256: str
    original_invocation_id: str
    full_state4_authorization_file_sha256: str
    capture_policy_file_ref: old.FileRef | None
    accepted_anchor: old.AcceptedHandoffAnchor
    host_profile_file_ref: old.FileRef
    deployment_source_files: tuple[old.HandoffSourceFile, ...]
    dependency_files: tuple[AdministrativeDependency, ...]
    inputs: tuple[HandoffRestartInput, ...]
    grants: tuple[HandoffAdministrativeGrant, ...]
    revocation_policy: AdministrativeRevocationPolicy
    journal_policy: AdministrativeJournalPolicy
    not_before_utc_ns: int
    expires_at_utc_ns: int
    io_limits: AdministrativeIOLimits
    record_self_sha256: str


@dataclass(frozen=True, slots=True)
class HandoffRestartLeaseScope:
    ticket_self_sha256: str
    verification_id: str
    original_plan_self_sha256: str
    request_self_sha256: str
    full_state4_authorization_file_sha256: str
    operation: Literal["read_handoff_verification_input"]
    slot_id: str
    root_binding_id: str
    relative_path: str
    physical_policy_sha256: str
    allowance_id: str
    scope_self_sha256: str


@dataclass(frozen=True, slots=True)
class AdministrativeRevocationObservation:
    marker_relative_path: str
    status: Literal["absent", "present", "unavailable"]


@dataclass(frozen=True, slots=True)
class _JournalCommon:
    schema: str
    purpose: Literal["capture", "restart"]
    full_state4_authorization_file_sha256: str
    original_plan_self_sha256: str
    request_self_sha256: str
    original_invocation_id: str
    verification_id: str | None
    accepted_administration_file_sha256: str
    record_self_sha256: str


@dataclass(frozen=True, slots=True)
class HandoffAdministrationActivation(_JournalCommon):
    activation_id: str
    journal_policy_sha256: str
    observed_utc_ns: int


@dataclass(frozen=True, slots=True)
class HandoffAdministrationReservation(_JournalCommon):
    activation_id: str
    session_id: str
    allowance_id: str
    grant_self_sha256: str
    operation: Literal["read_handoff_input", "retain_handoff_witness", "read_handoff_verification_input"]
    slot_id: str
    scope_self_sha256: str
    observed_utc_ns: int


@dataclass(frozen=True, slots=True)
class HandoffAdministrationCheckpoint(_JournalCommon):
    activation_id: str
    session_id: str
    allowance_id: str
    operation: Literal["read_handoff_input", "retain_handoff_witness", "read_handoff_verification_input"]
    slot_id: str
    scope_self_sha256: str
    ordinal: int
    checkpoint: Literal["initial", "before_open", "before_bytes", "before_publish", "final"]
    status: Literal["active", "refused", "failed"]
    reason: Literal["refused", "expired", "scope_mismatch", "unavailable", "malformed_output"] | None
    first_utc_ns: int | None
    final_utc_ns: int | None
    revocation_observations: tuple[AdministrativeRevocationObservation, ...]


@dataclass(frozen=True, slots=True)
class HandoffAdministrationTerminal(_JournalCommon):
    activation_id: str
    session_id: str
    allowance_id: str
    operation: Literal["read_handoff_input", "retain_handoff_witness", "read_handoff_verification_input"]
    slot_id: str
    scope_self_sha256: str
    requested_outcome: Literal["succeeded", "refused", "failed"]
    checkpoint_record_refs: tuple[old.FileRef, ...]


@dataclass(frozen=True, slots=True)
class AdministrativeFailure:
    stage: Literal["acquire", "lease_binding", "open", "identity", "security", "read", "cleanup", "finish"]
    operation: Literal["bootstrap_read", "revocation_inspection", "read_handoff_verification_input"]
    reason: Literal["refused", "unavailable", "not_found", "mismatch", "malformed_output", "short_io", "raised", "cleanup_failed", "terminal_unconfirmed", "limit_exceeded"]
    native_error_hex: str | None
    exception: old.ExceptionSummary | None


@dataclass(frozen=True, slots=True)
class AdministrativeJournalFailure:
    stage: Literal["identity", "create", "write", "flush", "sync", "close", "readback", "validate"]
    reason: Literal["collision", "unavailable", "mismatch", "short_io", "raised", "cleanup_failed", "malformed_output"]
    native_error_hex: str | None
    exception: old.ExceptionSummary | None


@dataclass(frozen=True, slots=True)
class AdministrativeJournalAck:
    status: Literal["recorded", "existing", "failed"]
    effect_certainty: Literal["confirmed", "not_applied", "unknown"]
    record: old.ByteRecord | None
    measured_file_identity: _NodeIdentity | None
    recorded_outcome: Literal["succeeded", "refused", "failed"] | None
    cleanup_failures: tuple[AdministrativeJournalFailure, ...]
    failure: AdministrativeJournalFailure | None


@dataclass(frozen=True, slots=True)
class RestartCheckpoint:
    name: Literal["initial", "before_open", "before_bytes", "final"]
    status: Literal["active", "refused", "failed"]
    reason: Literal["refused", "expired", "scope_mismatch", "unavailable", "malformed_output"] | None


@dataclass(frozen=True, slots=True)
class RestartAuthorityAccounting:
    acquire_status: Literal["not_attempted", "granted", "refused", "failed"]
    lease_returned: bool
    lease_binding_status: Literal["not_applicable", "valid", "invalid"]
    checkpoints: tuple[RestartCheckpoint, ...]
    finish_attempted: bool
    requested_outcome: Literal["succeeded", "refused", "failed"] | None
    finish_status: Literal["not_attempted", "succeeded", "failed"]
    effect_certainty: Literal["confirmed", "not_applied", "unknown"] | None
    recorded_outcome: Literal["succeeded", "refused", "failed"] | None


@dataclass(frozen=True, slots=True)
class HandoffRestartRead:
    slot_id: str
    status: Literal["complete", "not_found", "unavailable", "partial", "inconsistent"]
    data: old.ByteRecord | None
    data_completeness: Literal["complete", "prefix", "unavailable"]
    before: old.HandoffObjectMetadata | None
    after: old.HandoffObjectMetadata | None
    authority_accounting: RestartAuthorityAccounting
    cleanup_failures: tuple[AdministrativeFailure, ...]
    failure: AdministrativeFailure | None


@dataclass(frozen=True, slots=True)
class HandoffRestartReport:
    schema: str
    verification_id: str
    ticket_self_sha256: str
    accepted_administration_file_sha256: str
    accepted_anchor: old.AcceptedHandoffAnchor
    original_plan_self_sha256: str
    original_invocation_id: str
    ordered_reads: tuple[HandoffRestartRead, ...]
    historical_verification: HandoffVerification | None
    fresh_content_status: Literal["matched", "incomplete", "mismatch"]
    fresh_physical_status: Literal["matched", "incomplete", "mismatch"]
    capture_authority_status: Literal["corroborated", "incomplete", "mismatch", "not_recorded_legacy"]
    original_witness_finish_evidence: Literal["recorded_where_available", "incomplete"]
    restart_authority_status: Literal["confirmed", "incomplete"]
    reason_codes: tuple[str, ...]
    physical_inventory: tuple[old.HandoffPhysicalFile, ...]
    permission_acceptance_emitted: Literal[False]
    production_ready: Literal[False]
    record_self_sha256: str


_TYPES = dict(zip(KINDS, (HandoffAdministrativePolicy, HandoffAdministrativeGrant, HandoffRestartTicket,
    HandoffAdministrationActivation, HandoffAdministrationReservation, HandoffAdministrationCheckpoint,
    HandoffAdministrationTerminal, HandoffRestartReport), strict=True))


def _require(condition, message):
    if not condition:
        raise HandoffAdministrationError(message)


def _hash(value):
    return sha256_bytes(canonical_json_bytes(old._wire(value)))


def _seal(record):
    payload = old._wire(record)
    payload.pop("record_self_sha256")
    return replace(record, record_self_sha256=_hash(payload))


def _shape(value, cls):
    _require(type(value) is cls, "wrong exact " + cls.__name__)
    decoded = old._decode_dataclass(cls, old._wire(value), name=cls.__name__)
    _require(decoded == value, "mutable or noncanonical container in " + cls.__name__)
    # Encoding also rejects malformed Unicode/non-JSON scalars.
    canonical_json_bytes(old._wire(value))


def _record(value, kind):
    _shape(value, _TYPES[kind])
    _require(value.schema == PREFIX + kind + ".v1", "schema kind mismatch")
    require_sha256(value.record_self_sha256, name="record self hash")
    _require(_seal(value) == value, "record self hash mismatch")


def _interval(value):
    old._exact_int(value.not_before_utc_ns, name="not before")
    old._exact_int(value.expires_at_utc_ns, name="expiry")
    _require(value.not_before_utc_ns < value.expires_at_utc_ns, "empty validity interval")


def _stat_identity(value, *, directory):
    _shape(value, _NodeIdentity)
    for field in fields(value):
        old._exact_int(getattr(value, field.name), name=field.name)
    _require(value.mode_type == (stat.S_IFDIR if directory else stat.S_IFREG) and not value.file_attributes & 0x400,
             "invalid stat identity type or reparse attribute")


def _physical(policy, *, plan, root_id, relative, allow_unknown=False, journal=False, capture=False):
    _shape(policy, AdministrativePhysicalPolicy)
    windows._validate_root_binding(policy.root, plan)
    _require(policy.root.root_binding_id == root_id, "physical root mismatch")
    old._relative_path(relative, name="target relative path")
    _require(policy.parent_relative_path == relative.rpartition('/')[0], "target parent mismatch")
    chain = windows._parents(policy.parent_relative_path)
    _require(policy.namespace_relative_path in chain, "namespace not in ancestry")
    for path in chain:
        item = windows._directory(policy.root, path)
        _require(item.identity is not None or capture and journal and path == plan.journal_binding.attempt_relative_path
                 and plan.journal_binding.expected_attempt_directory_identity is None, "unaccepted directory identity")
    require_sha256(policy.security_descriptor_sha256, name="physical descriptor")
    if policy.confidential_profile_file_sha256 is None:
        _require(journal, "null profile outside journal convention")
    else:
        require_sha256(policy.confidential_profile_file_sha256, name="physical profile")
    if policy.expected_file_identity is None:
        _require(allow_unknown, "target identity unavailable")
    else:
        old._validate_native_identity(policy.expected_file_identity, name="target identity")
        _require(policy.expected_file_identity.volume_serial_number_hex == policy.root.volume_serial_number_hex, "target volume mismatch")


def _file_ref_at(ref, root, relative):
    old._validate_file_ref(ref, name="file reference")
    _require((ref.root_binding_id, ref.relative_path) == (root, relative), "file reference address mismatch")


def _legacy_scope(plan, operation, slot_id):
    if operation == "read_handoff_input":
        slot = next(x for x in plan.input_slots if x.slot_id == slot_id)
        root_id, role = slot.root_binding_id, slot.role
    else:
        _require(operation == "retain_handoff_witness" and slot_id in old.WITNESS_FILENAMES, "unknown witness scope")
        root_id, role = plan.witness_store.root_binding_id, "witness"
    namespace, parent = sources._expected_scope_identities(plan, role=role)
    scope = old.HandoffLeaseScope(operation, plan.plan_self_sha256, plan.expected_request.request_self_sha256,
        plan.authority_binding.full_state4_authorization_file_sha256, root_id, slot_id, namespace, parent, "0" * 64)
    return replace(scope, scope_self_sha256=_hash(sources._scope_payload(scope)))


def derive_handoff_allowance_id(*, purpose, full_A4, original_plan_self_sha256, request_self_sha256,
                                original_invocation_id, verification_id_or_null, operation, slot_id):
    _require(purpose in {"capture", "restart"}, "unknown allowance purpose")
    for name, value in (("A4", full_A4), ("plan", original_plan_self_sha256), ("request", request_self_sha256), ("invocation", original_invocation_id)):
        require_sha256(value, name=name)
    old._text(slot_id, name="allowance slot")
    _require((purpose == "capture" and verification_id_or_null is None and operation in {"read_handoff_input", "retain_handoff_witness"})
        or (purpose == "restart" and type(verification_id_or_null) is str and bool(verification_id_or_null) and operation == "read_handoff_verification_input"), "allowance family mismatch")
    payload = dict(purpose=purpose, full_A4=full_A4, original_plan_self_sha256=original_plan_self_sha256,
        request_self_sha256=request_self_sha256, original_invocation_id=original_invocation_id,
        verification_id_or_null=verification_id_or_null, operation=operation, slot_id=slot_id)
    return sha256_bytes(b"operatormark-handoff-administrative-allowance-v1\n" + canonical_json_bytes(payload))


def _allowance(grant):
    return derive_handoff_allowance_id(purpose=grant.purpose, full_A4=grant.full_state4_authorization_file_sha256,
        original_plan_self_sha256=grant.original_plan_self_sha256, request_self_sha256=grant.request_self_sha256,
        original_invocation_id=grant.original_invocation_id, verification_id_or_null=grant.restart_ticket_identity,
        operation=grant.operation, slot_id=grant.slot_id)


def _activation_id(policy, request_self_sha256):
    purpose = "capture" if type(policy) is HandoffAdministrativePolicy else "restart"
    payload = dict(purpose=purpose, full_A4=policy.full_state4_authorization_file_sha256,
        original_plan_self_sha256=policy.original_plan_self_sha256, request_self_sha256=request_self_sha256,
        original_invocation_id=policy.original_invocation_id,
        verification_id_or_null=None if purpose == "capture" else policy.verification_id)
    return sha256_bytes(b"operatormark-handoff-administrative-activation-v1\n" + canonical_json_bytes(payload))


def derive_handoff_authority_addresses(*, policy, request_self_sha256):
    require_sha256(request_self_sha256, name="request")
    activation = _activation_id(policy, request_self_sha256)
    addresses = [activation + ".activation.json"]
    for grant in policy.grants:
        allowance = _allowance(grant)
        stem = activation + "." + allowance
        addresses.append(stem + ".reservation.json")
        checkpoints = WRITE_CHECKPOINTS if grant.operation == "retain_handoff_witness" else READ_CHECKPOINTS
        addresses.extend(stem + f".{i}.checkpoint.json" for i in range(len(checkpoints)))
        addresses.append(stem + ".terminal.json")
    _require(len(set(addresses)) == len(addresses), "duplicate administrative addresses")
    return tuple(addresses)


def _revocation_leaves(policy, accepted_sha, allowance_id):
    names = ["authorization-" + policy.full_state4_authorization_file_sha256 + ".revoked", "administration-" + accepted_sha + ".revoked"]
    if allowance_id is not None:
        require_sha256(allowance_id, name="revocation allowance")
        names.append("allowance-" + allowance_id + ".revoked")
    parent = policy.revocation_policy.parent_relative_path
    return tuple((parent + "/" if parent else "") + x for x in names)


def _common(value, plan):
    old.validate_handoff_plan(plan)
    _require(value.original_plan_self_sha256 == plan.plan_self_sha256
        and value.full_state4_authorization_file_sha256 == plan.authority_binding.full_state4_authorization_file_sha256
        and value.original_invocation_id == old.derive_handoff_invocation_id(plan), "administration original plan binding mismatch")
    _interval(value)
    old._text(value.acceptance_reference, name="acceptance reference")
    plan_slot = next(x for x in plan.input_slots if x.slot_id == "handoff-plan")
    _file_ref_at(value.original_plan_file_ref, plan_slot.root_binding_id, plan_slot.relative_path)
    plan_bytes = canonical_json_bytes(old._wire(plan)) + b"\n"
    _require((value.original_plan_file_ref.file_bytes, value.original_plan_file_ref.file_sha256) == (len(plan_bytes), sha256_bytes(plan_bytes)), "original plan file mismatch")
    old._validate_file_ref(value.host_profile_file_ref, name="host profile")
    _require(tuple(x.relative_path for x in value.deployment_source_files) == DEPLOYMENT_SOURCE_PATHS, "deployment source closure/order mismatch")
    for source in value.deployment_source_files:
        _shape(source, old.HandoffSourceFile)
        old._exact_int(source.file_bytes, name="source length")
        require_sha256(source.file_sha256, name="source hash")
    limits = value.io_limits
    _shape(limits, AdministrativeIOLimits)
    for field in fields(limits):
        amount = old._exact_int(getattr(limits, field.name), name=field.name, minimum=1)
        _require(amount < windows.native.SIGNED_LARGE_INTEGER_MAX, "unsupported native bound")
    _require(limits.transfer_chunk_bytes <= windows.native.DWORD_MAX, "chunk exceeds DWORD")
    journal = value.journal_policy
    _shape(journal, AdministrativeJournalPolicy)
    old._text(journal.root_path, name="accounting root")
    _require(PureWindowsPath(journal.root_path).is_absolute(), "accounting root is not absolute")
    _require(re.fullmatch(r"[A-Z]:\\(?:[^\\]+(?:\\[^\\]+)*)?", journal.root_path) is not None,
        "accounting root must have exact canonical DOS spelling")
    if journal.root_path[3:]: windows._relative(journal.root_path[3:].replace("\\", "/"))
    _stat_identity(journal.root_identity, directory=True)
    old._validate_file_ref(journal.namespace_profile_file_ref, name="namespace profile")
    old._validate_file_ref(journal.record_security_policy_file_ref, name="record security policy")
    _require(journal.storage_owner == "separate_B", "wrong storage ownership")
    # Accepted dedicated root must not overlap any capture output/original journal root.
    for root in (value.configuration.roots if type(value) is HandoffAdministrativePolicy else tuple(x.physical_policy.root for x in value.inputs if x.role in {"witness", "original_input"})):
        if root.root_binding_id in {plan.witness_store.root_binding_id, plan.journal_binding.root_binding_id}:
            a, b = journal.root_path.rstrip("\\").casefold(), root.canonical_dos_root.rstrip("\\").casefold()
            _require(a != b and not a.startswith(b + "\\") and not b.startswith(a + "\\"), "accounting root overlaps existing output/journal")
    revoke = value.revocation_policy
    _shape(revoke, AdministrativeRevocationPolicy)
    physical = AdministrativePhysicalPolicy(revoke.root, revoke.namespace_relative_path, revoke.parent_relative_path,
        None, revoke.security_descriptor_sha256, revoke.confidential_profile_file_sha256)
    marker = (revoke.parent_relative_path + "/" if revoke.parent_relative_path else "") + "marker.revoked"
    _physical(physical, plan=plan, root_id=revoke.root.root_binding_id, relative=marker, allow_unknown=True)
    roles = tuple(x.role for x in value.dependency_files)
    _require(len(set(roles)) == len(roles), "duplicate dependency role")
    for dependency in value.dependency_files:
        old._text(dependency.role, name="dependency role")
        ref = dependency.file_ref
        old._validate_file_ref(ref, name="dependency ref")
        _physical(dependency.physical_policy, plan=plan, root_id=ref.root_binding_id, relative=ref.relative_path)


def validate_handoff_administrative_grant(grant, *, expected_plan, administration=None, input=None):
    _record(grant, "grant")
    _interval(grant)
    plan = expected_plan
    _require(grant.full_state4_authorization_file_sha256 == plan.authority_binding.full_state4_authorization_file_sha256
        and grant.original_plan_self_sha256 == plan.plan_self_sha256 and grant.request_self_sha256 == plan.expected_request.request_self_sha256
        and grant.original_invocation_id == old.derive_handoff_invocation_id(plan), "grant A4/plan/request/invocation mismatch")
    _require(grant.allowance_id == _allowance(grant), "allowance ID mismatch")
    if administration is not None:
        _require(administration.not_before_utc_ns <= grant.not_before_utc_ns < grant.expires_at_utc_ns <= administration.expires_at_utc_ns, "grant validity exceeds accepting interval")
    if grant.purpose == "capture":
        _require(grant.restart_ticket_identity is None, "capture ticket cycle")
        scope = _legacy_scope(plan, grant.operation, grant.slot_id)
        _require(grant.legacy_scope_self_sha256 == scope.scope_self_sha256 and grant.root_binding_id == scope.root_binding_id, "legacy scope mismatch")
        if grant.operation == "read_handoff_input":
            slot = next(x for x in plan.input_slots if x.slot_id == grant.slot_id)
            _require(grant.relative_path == slot.relative_path and grant.expected_file_ref == slot.expected_file_ref and grant.creating_relative_path is None, "capture read address/bytes mismatch")
            journal = slot.role.startswith("journal_")
            _physical(grant.physical_policy, plan=plan, root_id=slot.root_binding_id, relative=slot.relative_path,
                allow_unknown=slot.role != "authority_input", journal=journal, capture=True)
            if administration is not None:
                cfg = administration.configuration
                native_policy = cfg.input_policies[plan.input_slots.index(slot)]
                expected = AdministrativePhysicalPolicy(windows._root(cfg, slot.root_binding_id), native_policy.namespace_relative_path,
                    native_policy.parent_relative_path, native_policy.expected_file_identity, native_policy.security_descriptor_sha256,
                    native_policy.confidential_profile_file_sha256)
                _require(grant.physical_policy == expected, "capture grant physical policy differs from provider configuration")
        else:
            final = plan.witness_store.parent_relative_path + "/" + grant.slot_id
            _require(grant.relative_path == final and grant.creating_relative_path == final + ".creating" and grant.expected_file_ref is None,
                "witness final/creating pair mismatch")
            _physical(grant.physical_policy, plan=plan, root_id=scope.root_binding_id, relative=final, allow_unknown=True)
            _require(grant.physical_policy.expected_file_identity is None, "invented pre-creation identity")
            if administration is not None:
                w = plan.witness_store
                expected = AdministrativePhysicalPolicy(windows._root(administration.configuration, w.root_binding_id), w.namespace_relative_path,
                    w.parent_relative_path, None, w.created_object_security_descriptor_sha256, w.confidential_profile_file_ref.file_sha256)
                _require(grant.physical_policy == expected, "witness physical policy mismatch")
    else:
        _require(grant.operation == "read_handoff_verification_input" and grant.legacy_scope_self_sha256 is None
            and grant.creating_relative_path is None, "restart operation has capture scope")
        old._text(grant.restart_ticket_identity, name="verification identity")
        if administration is not None:
            _require(type(administration) is HandoffRestartTicket and grant.restart_ticket_identity == administration.verification_id, "restart ticket identity mismatch")
        if input is not None:
            _require((grant.slot_id, grant.root_binding_id, grant.relative_path, grant.physical_policy, grant.expected_file_ref) ==
                (input.slot_id, input.root_binding_id, input.relative_path, input.physical_policy, input.expected_file_ref), "restart grant input mismatch")
        _physical(grant.physical_policy, plan=plan, root_id=grant.root_binding_id, relative=grant.relative_path,
            allow_unknown=True, journal=grant.physical_policy.confidential_profile_file_sha256 is None)
    if grant.expected_file_ref is not None:
        _file_ref_at(grant.expected_file_ref, grant.root_binding_id, grant.relative_path)


def validate_handoff_administrative_policy(policy, *, expected_plan):
    _record(policy, "policy")
    _common(policy, expected_plan)
    windows.validate_handoff_windows_configuration(expected_plan=expected_plan, configuration=policy.configuration)
    expected = tuple(("read_handoff_input", x.slot_id) for x in expected_plan.input_slots) + tuple(("retain_handoff_witness", x) for x in old.WITNESS_FILENAMES)
    _require(tuple((x.operation, x.slot_id) for x in policy.grants) == expected, "capture grant coverage/order mismatch")
    for grant in policy.grants:
        _require(grant.purpose == "capture", "restart grant in capture policy")
        validate_handoff_administrative_grant(grant, expected_plan=expected_plan, administration=policy)
    _require(len({x.allowance_id for x in policy.grants}) == len(policy.grants), "duplicate allowance")


def validate_handoff_restart_ticket(ticket, *, expected_plan, capture_policy=None):
    _record(ticket, "restart_ticket")
    _common(ticket, expected_plan)
    old._text(ticket.verification_id, name="verification identity")
    anchor = ticket.accepted_anchor
    _shape(anchor, old.AcceptedHandoffAnchor)
    _require((anchor.specification_file_sha256, anchor.plan_self_sha256, anchor.invocation_id, anchor.handoff_source_vector_sha256) ==
        (expected_plan.specification_file_sha256, expected_plan.plan_self_sha256, ticket.original_invocation_id, _hash(expected_plan.source_files)), "anchor original binding mismatch")
    old._exact_int(anchor.receipt_file_bytes, name="anchor receipt bytes", minimum=1)
    require_sha256(anchor.receipt_file_sha256, name="anchor receipt hash")
    _require(ticket.verification_mode in {"administrative_corroboration", "legacy_review"}, "unknown verification mode")
    if ticket.verification_mode == "administrative_corroboration":
        _require(ticket.capture_policy_file_ref is not None, "capture policy reference unavailable")
    if ticket.capture_policy_file_ref is not None:
        old._validate_file_ref(ticket.capture_policy_file_ref, name="capture policy")
    inputs = ticket.inputs
    _require(len(inputs) >= 6, "missing witness chain")
    witness = expected_plan.witness_store
    for item, filename in zip(inputs[:6], RESTART_WITNESSES, strict=True):
        _require(item.role == "witness" and item.slot_id == filename and item.root_binding_id == witness.root_binding_id
            and item.relative_path == witness.parent_relative_path + "/" + filename and item.original_slot_id is None, "fixed witness order/address mismatch")
        _require(item.expected_file_ref is not None, "witness expected bytes unavailable")
        if filename == "receipt.json":
            _require((item.expected_file_ref.file_bytes, item.expected_file_ref.file_sha256) == (anchor.receipt_file_bytes, anchor.receipt_file_sha256), "receipt does not match independent anchor")
    original = tuple(x for x in inputs if x.role == "original_input")
    _require(tuple(x.original_slot_id for x in original) == tuple(x.slot_id for x in expected_plan.input_slots), "original input coverage/order mismatch")
    rank = {"witness": 0, "original_input": 1, "capture_administration": 2, "dependency": 3}
    _require([rank[x.role] for x in inputs] == sorted(rank[x.role] for x in inputs), "input role order mismatch")
    aliases, slot_ids, physical_ids = set(), set(), set()
    for item in inputs:
        _shape(item, HandoffRestartInput)
        old._text(item.slot_id, name="restart slot")
        _require(item.slot_id not in slot_ids and (item.original_slot_id is not None) == (item.role == "original_input"), "slot identity/role mismatch")
        slot_ids.add(item.slot_id)
        key = windows._join(item.physical_policy.root.canonical_volume_guid_root, item.relative_path).casefold()
        _require(key not in aliases, "physical path alias requires deduplication")
        aliases.add(key)
        identity = item.physical_policy.expected_file_identity
        if identity is not None:
            identity_key = (identity.volume_serial_number_hex, identity.file_id_128_hex)
            _require(identity_key not in physical_ids, "pinned physical instance alias requires deduplication")
            physical_ids.add(identity_key)
        journal = item.role == "capture_administration"
        allow_unknown = False
        if item.role == "original_input":
            prior = next(x for x in expected_plan.input_slots if x.slot_id == item.original_slot_id)
            _require((item.root_binding_id, item.relative_path) == (prior.root_binding_id, prior.relative_path), "original address mismatch")
            _require(prior.expected_file_ref is None or item.expected_file_ref == prior.expected_file_ref, "original expected bytes mismatch")
            journal = prior.role.startswith("journal_")
            allow_unknown = prior.expected_file_ref is None and prior.role != "authority_input"
        if item.role == "capture_administration":
            allow_unknown = True
        _physical(item.physical_policy, plan=expected_plan, root_id=item.root_binding_id, relative=item.relative_path,
            allow_unknown=allow_unknown, journal=journal)
        if item.expected_file_ref is not None:
            _file_ref_at(item.expected_file_ref, item.root_binding_id, item.relative_path)
    _require(len(ticket.grants) == len(inputs), "restart grant count mismatch")
    for grant, item in zip(ticket.grants, inputs, strict=True):
        _require(grant.purpose == "restart", "capture allowance used for restart")
        validate_handoff_administrative_grant(grant, expected_plan=expected_plan, administration=ticket, input=item)
    administrative = tuple(x for x in inputs if x.role == "capture_administration")
    if ticket.verification_mode == "legacy_review":
        _require(not administrative, "legacy mode cannot claim new capture authority")
    elif capture_policy is not None:
        validate_handoff_administrative_policy(capture_policy, expected_plan=expected_plan)
        data = canonical_handoff_administrative_bytes(capture_policy, kind="policy", expected_plan=expected_plan)
        _require((len(data), sha256_bytes(data)) == (ticket.capture_policy_file_ref.file_bytes, ticket.capture_policy_file_ref.file_sha256), "capture policy bytes mismatch")
        addresses = derive_handoff_authority_addresses(policy=capture_policy, request_self_sha256=expected_plan.expected_request.request_self_sha256)
        _require(tuple(x.relative_path for x in administrative) == addresses, "capture administration address coverage/order mismatch")
        for item in administrative:
            _require(item.physical_policy.root.canonical_dos_root == capture_policy.journal_policy.root_path, "capture accounting physical root mismatch")


def _session_id(activation_bytes, root_identity):
    return sha256_bytes(b"operatormark-handoff-administrative-session-v1\n" + canonical_json_bytes(
        {"activation_bytes": old._wire(_bytes(activation_bytes)), "root_identity": old._wire(root_identity)}))


def _bytes(data):
    _require(type(data) is bytes, "opaque bytes must be exact bytes")
    return old.ByteRecord(data.hex(), len(data), sha256_bytes(data))


def validate_handoff_journal_record(record, *, policy, expected_plan, scope=None):
    kind = next((name for name in ("activation", "reservation", "checkpoint", "terminal") if type(record) is _TYPES[name]), None)
    _require(kind is not None, "unknown journal type")
    _record(record, kind)
    purpose = "capture" if type(policy) is HandoffAdministrativePolicy else "restart"
    expected_sha = sha256_bytes(canonical_json_bytes(old._wire(policy)) + b"\n")
    _require((record.purpose, record.full_state4_authorization_file_sha256, record.original_plan_self_sha256,
        record.request_self_sha256, record.original_invocation_id, record.verification_id, record.accepted_administration_file_sha256) ==
        (purpose, policy.full_state4_authorization_file_sha256, policy.original_plan_self_sha256,
        expected_plan.expected_request.request_self_sha256, policy.original_invocation_id,
        None if purpose == "capture" else policy.verification_id, expected_sha), "journal administration correspondence mismatch")
    _require(record.activation_id == _activation_id(policy, expected_plan.expected_request.request_self_sha256), "activation address mismatch")
    if kind == "activation":
        _require(record.journal_policy_sha256 == _hash(policy.journal_policy), "activation root/profile policy mismatch")
        old._exact_int(record.observed_utc_ns, name="activation UTC")
        _require(policy.not_before_utc_ns <= record.observed_utc_ns < policy.expires_at_utc_ns, "activation outside policy interval")
        return
    require_sha256(record.session_id, name="session ID")
    grant = next((x for x in policy.grants if x.allowance_id == record.allowance_id), None)
    _require(grant is not None and (record.operation, record.slot_id) == (grant.operation, grant.slot_id), "journal grant mismatch")
    require_sha256(record.scope_self_sha256, name="journal scope hash")
    if scope is not None:
        _require(record.scope_self_sha256 == scope.scope_self_sha256, "journal lease scope mismatch")
    if kind == "reservation":
        _require(record.grant_self_sha256 == grant.record_self_sha256, "reservation accepted grant hash mismatch")
        old._exact_int(record.observed_utc_ns, name="reservation UTC")
        _require(grant.not_before_utc_ns <= record.observed_utc_ns < grant.expires_at_utc_ns, "reservation outside grant interval")
    elif kind == "checkpoint":
        checkpoints = WRITE_CHECKPOINTS if grant.operation == "retain_handoff_witness" else READ_CHECKPOINTS
        old._exact_int(record.ordinal, name="checkpoint ordinal")
        _require(record.ordinal < len(checkpoints) and record.checkpoint == checkpoints[record.ordinal], "checkpoint ordinal/name mismatch")
        for instant in (record.first_utc_ns, record.final_utc_ns):
            if instant is not None:
                old._exact_int(instant, name="checkpoint UTC")
        leaves = _revocation_leaves(policy, expected_sha, grant.allowance_id)
        observed = record.revocation_observations
        _require(tuple(x.marker_relative_path for x in observed) == leaves[:len(observed)] and len(observed) <= len(leaves), "revocation address order mismatch")
        if record.status == "active":
            _require(record.reason is None and record.first_utc_ns is not None and record.final_utc_ns is not None
                and grant.not_before_utc_ns <= record.first_utc_ns <= record.final_utc_ns < grant.expires_at_utc_ns
                and len(observed) == len(leaves) and all(x.status == "absent" for x in observed), "active checkpoint lacks actual valid observations")
        elif record.status == "refused":
            _require(record.reason in {"refused", "expired", "scope_mismatch"}, "refusal mislabeled unavailable")
        else:
            _require(record.reason in {"unavailable", "malformed_output"}, "failed checkpoint reason mismatch")
    else:
        _require(record.requested_outcome in {"succeeded", "refused", "failed"}, "terminal outcome")
        stem = record.activation_id + "." + record.allowance_id
        for ordinal, ref in enumerate(record.checkpoint_record_refs):
            _file_ref_at(ref, policy.journal_policy.root_path, stem + f".{ordinal}.checkpoint.json")
        limit = len(WRITE_CHECKPOINTS if grant.operation == "retain_handoff_witness" else READ_CHECKPOINTS)
        _require(len(record.checkpoint_record_refs) <= limit, "terminal checkpoint overflow")
        if record.requested_outcome == "succeeded":
            _require(len(record.checkpoint_record_refs) == limit, "successful terminal missing checkpoints")


def _empty_accounting():
    return RestartAuthorityAccounting("not_attempted", False, "not_applicable", (), False, None, "not_attempted", None, None)


def _accounting(value):
    _shape(value, RestartAuthorityAccounting)
    # Reuse the algebra through an explicit value conversion, never an old lease
    # or old operation result. No provider can substitute old accounting here.
    converted = old.HandoffAuthorityAccounting(value.acquire_status, value.lease_returned, value.lease_binding_status,
        tuple(old.HandoffCheckpoint(x.name, x.status, x.reason) for x in value.checkpoints), value.finish_attempted,
        value.requested_outcome, value.finish_status, value.effect_certainty, value.recorded_outcome)
    sources._validate_handoff_authority_accounting(converted, operation="read_handoff_input")
    for checkpoint in value.checkpoints:
        if checkpoint.status == "refused":
            _require(checkpoint.reason in {"refused", "expired", "scope_mismatch"}, "refusal lacks refusal reason")
        if checkpoint.status == "failed":
            _require(checkpoint.reason in {"unavailable", "malformed_output"}, "unavailable mislabeled refusal")


def _failure(value, operation):
    _shape(value, AdministrativeFailure)
    _require(value.operation == operation, "failure operation family mismatch")
    if value.native_error_hex is not None:
        _require(type(value.native_error_hex) is str and re.fullmatch(r"0x[1-9a-f][0-9a-f]{0,7}", value.native_error_hex) is not None,
            "native error is not a nonzero DWORD")


def _metadata(value, input, *, match):
    _shape(value, old.HandoffObjectMetadata)
    p = input.physical_policy
    old._exact_int(value.size_bytes, name="measured size")
    _require((value.root_binding_id, value.relative_path) == (input.root_binding_id, input.relative_path), "measurement at wrong requested address")
    for identity in (value.identity, value.parent_identity, value.namespace_identity):
        old._validate_native_identity(identity, name="measured identity")
    for name in ("security_descriptor_sha256", "volume_identifier_commitment"):
        require_sha256(getattr(value, name), name=name)
    if value.confidential_profile_file_sha256 is not None:
        require_sha256(value.confidential_profile_file_sha256, name="measured profile")
    old._text(value.canonical_volume_guid_path, name="measured GUID path")
    if match:
        _require(value.canonical_volume_guid_path == windows._join(p.root.canonical_volume_guid_root, input.relative_path)
            and (p.expected_file_identity is None or value.identity == p.expected_file_identity)
            and value.parent_identity == windows._directory(p.root, p.parent_relative_path).identity
            and value.namespace_identity == windows._directory(p.root, p.namespace_relative_path).identity
            and value.security_descriptor_sha256 == p.security_descriptor_sha256
            and value.confidential_profile_file_sha256 == p.confidential_profile_file_sha256
            and value.volume_identifier_commitment == p.root.volume_identifier_commitment, "physical measurement mismatch")


def validate_handoff_restart_read(result, *, input, raw=False, operation="read_handoff_verification_input"):
    _shape(result, HandoffRestartRead)
    _require(result.slot_id == input.slot_id, "read slot mismatch")
    _accounting(result.authority_accounting)
    if raw:
        _require(result.authority_accounting == _empty_accounting(), "provider invented authority accounting")
    for item in result.cleanup_failures:
        _failure(item, operation)
        _require(item.stage == "cleanup", "cleanup failure stage mismatch")
    if result.failure is not None:
        _failure(result.failure, operation)
    data = None if result.data is None else old._validate_byte_record(result.data, name="read bytes")
    _require((data is None) == (result.data_completeness == "unavailable"), "read data/completeness contradiction")
    mismatch = result.status == "inconsistent" and result.failure is not None and result.failure.reason == "mismatch"
    for measurement in (result.before, result.after):
        if measurement is not None:
            _metadata(measurement, input, match=not mismatch)
    if result.status == "complete":
        _require(result.before is not None and result.after == result.before and data is not None
            and result.data_completeness == "complete" and len(data) == result.after.size_bytes
            and result.failure is None and not result.cleanup_failures, "complete read lacks stable physical facts/cleanup")
        if input.expected_file_ref is not None:
            _require((len(data), sha256_bytes(data)) == (input.expected_file_ref.file_bytes, input.expected_file_ref.file_sha256), "read expected content mismatch")
        if not raw:
            a = result.authority_accounting
            _require(a.acquire_status == "granted" and a.lease_binding_status == "valid" and a.finish_status == "succeeded"
                and a.effect_certainty == "confirmed" and a.recorded_outcome == a.requested_outcome == "succeeded", "complete read lacks actual terminal acknowledgment")
    else:
        _require(result.failure is not None, "negative read missing cause")
        if result.status == "not_found":
            _require(data is None and result.before is None and result.after is None
                and result.failure.stage == "open" and result.failure.reason == "not_found", "not found contradicts observed object/refusal")
        if result.status == "unavailable":
            _require(data is None, "unavailable cannot hide copied bytes")
        if result.status == "partial":
            _require(data is not None and result.data_completeness == "prefix", "partial read missing prefix")
        if result.status == "inconsistent":
            _require(result.failure.reason not in {"refused", "not_found"}, "inconsistency mislabeled absence/denial")
        _require(result.authority_accounting.recorded_outcome != "succeeded", "negative facts with successful terminal outcome")
    if result.authority_accounting.acquire_status == "not_attempted" and not raw:
        _require(result.status == "unavailable" and result.before is None and result.after is None
            and result.authority_accounting == _empty_accounting() and result.failure.reason == "unavailable", "unattempted input fabricated acquisition")


def _restart_accounting_confirmed(reads):
    return all(read.authority_accounting.acquire_status == "granted"
        and read.authority_accounting.lease_binding_status == "valid"
        and tuple((x.name, x.status) for x in read.authority_accounting.checkpoints) == tuple((x, "active") for x in READ_CHECKPOINTS)
        and read.authority_accounting.finish_status == "succeeded" and read.authority_accounting.effect_certainty == "confirmed"
        and read.authority_accounting.recorded_outcome == read.authority_accounting.requested_outcome for read in reads)


def _restart_inventory(ticket, reads):
    result, seen = [], set()
    for input, read in zip(ticket.inputs, reads, strict=True):
        if read.status != "complete" or read.after is None: continue
        identity = (read.after.identity.volume_serial_number_hex, read.after.identity.file_id_128_hex)
        _require(identity not in seen, "duplicate acquired physical instance")
        seen.add(identity)
        owner = "separate_B" if input.role == "capture_administration" else "accepted_attempt_M" if input.role == "witness" else "external_preexisting"
        result.append(old.HandoffPhysicalFile(input.root_binding_id, input.relative_path, "loose_file", owner, input.role == "witness",
            read.after.size_bytes, "exact", "observed_file_length", ("bytes_retained_and_measured",)))
    return tuple(result)


def _validate_handoff_restart_report_structure(report, *, ticket, expected_plan):
    _record(report, "restart_report")
    _require((report.verification_id, report.ticket_self_sha256, report.accepted_anchor, report.original_plan_self_sha256,
        report.original_invocation_id) == (ticket.verification_id, ticket.record_self_sha256, ticket.accepted_anchor,
        expected_plan.plan_self_sha256, ticket.original_invocation_id), "report ticket binding mismatch")
    _require(report.accepted_administration_file_sha256 == sha256_bytes(canonical_json_bytes(old._wire(ticket)) + b"\n"), "report accepted ticket byte hash mismatch")
    _require(report.permission_acceptance_emitted is False and report.production_ready is False, "component cannot emit acceptance/readiness")
    _require(len(report.ordered_reads) == len(ticket.inputs), "report input count mismatch")
    for value, input in zip(report.ordered_reads, ticket.inputs, strict=True):
        validate_handoff_restart_read(value, input=input)
    _require(report.reason_codes == tuple(x for x in REASONS if x in report.reason_codes), "unknown/duplicate/reordered report reason")
    required = ["size_contribution_unresolved", "original_acknowledgment_unavailable"]
    _require(report.original_witness_finish_evidence == "incomplete", "fixed old receipt cannot supply its own finish acknowledgment")
    _require((report.capture_authority_status == "not_recorded_legacy") == (ticket.verification_mode == "legacy_review"),
        "capture authority mode/status mismatch")
    _require((report.restart_authority_status == "confirmed") == _restart_accounting_confirmed(report.ordered_reads),
        "restart authority conclusion contradicts actual accounting")
    for status, mismatch_reason, incomplete_reason in ((report.fresh_content_status, "fresh_content_mismatch", "fresh_input_unavailable"),
        (report.fresh_physical_status, "fresh_physical_mismatch", "fresh_input_unavailable"),
        (report.capture_authority_status, "capture_authority_mismatch", "capture_authority_unavailable")):
        if status == "mismatch": required.append(mismatch_reason)
        if status == "incomplete": required.append(incomplete_reason)
    if report.capture_authority_status == "not_recorded_legacy":
        _require(ticket.verification_mode == "legacy_review", "legacy status in corroboration mode")
        required.append("legacy_authority_not_recorded")
    if report.restart_authority_status == "incomplete": required.append("restart_authority_unconfirmed")
    if report.original_witness_finish_evidence == "incomplete": required.append("original_acknowledgment_unavailable")
    if report.historical_verification is None:
        missing = any(x.status != "complete" for x in report.ordered_reads[:6])
        required.append("witness_unavailable" if missing else "witness_chain_mismatch")
        _require(report.fresh_content_status == report.fresh_physical_status == "incomplete"
            and report.capture_authority_status in {"incomplete", "not_recorded_legacy"}, "unavailable history cannot support favorable fresh/capture conclusions")
    elif report.historical_verification.overall_status != "corroborated":
        required.append("historical_verification_incomplete")
    if any(x.cleanup_failures for x in report.ordered_reads): required.append("cleanup_failed")
    _require(set(report.reason_codes) == set(required), "report hides an unfavorable conclusion or includes unsupported reasons")
    validate_handoff_physical_inventory(report.physical_inventory)
    _require(report.physical_inventory == _restart_inventory(ticket, report.ordered_reads), "inventory differs from independent measured instances")


def validate_handoff_restart_report(report, *, ticket, expected_plan):
    _validate_handoff_restart_report_structure(report, ticket=ticket, expected_plan=expected_plan)
    # Late import avoids a module-initialization cycle. This is pure computation
    # over independently bound inputs, never a recursive call to report validation.
    from .permission_handoff_restart import _compute_reacquired_permission_handoff
    expected = _compute_reacquired_permission_handoff(ticket=ticket, expected_plan=expected_plan,
        expected_source_files=expected_plan.source_files, ordered_reads=report.ordered_reads)
    _require(report == expected, "report conclusions differ from independently recomputed evidence")


def canonical_handoff_administrative_bytes(record, *, kind, expected_plan, administration=None, ticket=None, capture_policy=None):
    _require(kind in _TYPES, "unknown administrative kind")
    if kind == "policy": validate_handoff_administrative_policy(record, expected_plan=expected_plan)
    elif kind == "grant": validate_handoff_administrative_grant(record, expected_plan=expected_plan, administration=administration)
    elif kind == "restart_ticket": validate_handoff_restart_ticket(record, expected_plan=expected_plan, capture_policy=capture_policy)
    elif kind == "restart_report": validate_handoff_restart_report(record, ticket=ticket, expected_plan=expected_plan)
    else: validate_handoff_journal_record(record, policy=administration, expected_plan=expected_plan)
    return canonical_json_bytes(old._wire(record)) + b"\n"


def strict_handoff_administrative_loads(data, *, kind, expected_plan, administration=None, ticket=None, capture_policy=None):
    _require(kind in _TYPES and type(data) is bytes and data.endswith(b"\n"), "unknown kind or missing canonical LF")
    payload = strict_canonical_json_loads(data[:-1])
    _require(payload.get("schema") == PREFIX + kind + ".v1", "wire schema kind mismatch")
    value = old._decode_dataclass(_TYPES[kind], payload, name=kind)
    encoded = canonical_handoff_administrative_bytes(value, kind=kind, expected_plan=expected_plan,
        administration=administration, ticket=ticket, capture_policy=capture_policy)
    _require(encoded == data, "administrative bytes are not canonical")
    return value
