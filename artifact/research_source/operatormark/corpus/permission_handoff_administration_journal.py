"""Exclusive additive accounting, not authenticated custody or rollback proof.

No directory creation, retries, adoption of existing files, or observation-journal
activation. The supplied pre-existing root is an independently approved boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from threading import RLock

from .canonical import canonical_json_bytes, sha256_bytes
from .permission_authority_journal import _OSFilesystem, _node_identity
from . import permission_handoff_administration_records as records
from . import permission_handoff_records as old
from .permission_handoff_capture import _exception_summary


@dataclass(frozen=True, slots=True)
class _Session:
    activation: records.HandoffAdministrationActivation
    activation_bytes: bytes
    session_id: str
    process_id: int


def _failure(error, stage, reason="raised"):
    value = getattr(error, "winerror", None)
    code = f"0x{value:x}" if type(value) is int and 0 < value <= 0xffffffff else None
    return records.AdministrativeJournalFailure(stage, reason, code, _exception_summary(error))


class FileHandoffAdministrationJournal:
    def __init__(self, *, policy, expected_plan, filesystem=None):
        self.policy, self.plan = policy, expected_plan
        self._fs = _OSFilesystem() if filesystem is None else filesystem
        self._root = Path(policy.journal_policy.root_path)
        self._root_identity = policy.journal_policy.root_identity
        self._session = None
        self._attempted = set()
        self._lock = RLock()
        self._busy = False
        self._evidence = []
        self._reservations = {}
        self._checkpoints = {}
        self._terminals = set()

    @property
    def evidence(self):
        return tuple(self._evidence)

    @property
    def session_id(self):
        self._owned()
        return self._session.session_id

    def _owned(self):
        records._require(self._session is not None and self._session.process_id == os.getpid(), "no private live accounting session")

    def _root_checked(self):
        measured = _node_identity(self._fs.lstat(self._root), directory=True)
        records._stat_identity(measured, directory=True)
        records._require(measured == self._root_identity, "accounting root identity changed")
        return measured

    def _file_checked(self, value):
        identity = _node_identity(value, directory=False)
        records._stat_identity(identity, directory=False)
        records._require(type(getattr(value, "st_nlink", None)) is int and value.st_nlink == 1, "accounting file has multiple/unknown links")
        return identity

    def _persist(self, record, relative):
        # Mark before validation/dependency calls: no second invocation after any
        # ambiguous backend return. A confirmed existing leaf never grants ownership.
        with self._lock:
            records._require(not self._busy and relative not in self._attempted, "journal invocation already attempted/reentrant")
            self._attempted.add(relative)
            self._busy = True
        stream = None
        known_identity = None
        attempted_create = False
        create_returned = False
        cleanup = []
        copied = bytearray()
        failure = None
        stage = "validate"
        status, certainty = "failed", "not_applied"
        data = None
        path = self._root / relative
        try:
            records._require(path.parent == self._root and path.name == relative, "journal leaf escaped approved parent")
            records.validate_handoff_journal_record(record, policy=self.policy, expected_plan=self.plan)
            data = canonical_json_bytes(old._wire(record)) + b"\n"
            records._require(len(data) <= self.policy.io_limits.maximum_administrative_record_bytes, "administrative record limit exceeded")
            stage = "identity"
            self._root_checked()
            try:
                self._fs.lstat(path)
            except FileNotFoundError:
                pass
            else:
                status = "existing"
                raise FileExistsError("stable accounting leaf already exists")
            self._root_checked()
            stage = "create"
            attempted_create, certainty = True, "unknown"
            stream = self._fs.open_exclusive(path)
            create_returned = True
            known_identity = self._file_checked(self._fs.fstat(stream))
            records._require(self._file_checked(self._fs.lstat(path)) == known_identity, "created file/leaf identity mismatch")
            self._root_checked()
            stage = "write"
            chunk = self.policy.io_limits.transfer_chunk_bytes
            for offset in range(0, len(data), chunk):
                payload = data[offset:offset + chunk]
                count = self._fs.write(stream, payload)
                self._evidence.append(("write_return", relative, offset, count))
                if type(count) is not int or count != len(payload):
                    failure = records.AdministrativeJournalFailure("write", "short_io", None, None)
                    raise ValueError("short/malformed write; no replay permitted")
            stage = "flush"
            records._require(self._fs.flush(stream) is None, "malformed flush acknowledgment")
            stage = "sync"
            records._require(self._fs.sync(stream) is None, "malformed sync acknowledgment")
            records._require(self._file_checked(self._fs.fstat(stream)) == known_identity, "writer identity changed")
            stage = "close"
            owned, stream = stream, None
            try:
                records._require(self._fs.close(owned) is None, "malformed close acknowledgment")
            except BaseException as error:
                cleanup.append(_failure(error, "close", "cleanup_failed"))
                raise
            stage = "readback"
            self._root_checked()
            records._require(self._file_checked(self._fs.lstat(path)) == known_identity, "published accounting leaf replaced")
            stream = self._fs.open_read(path)
            metadata = self._fs.fstat(stream)
            records._require(self._file_checked(metadata) == known_identity and type(metadata.st_size) is int and metadata.st_size == len(data), "readback identity/extent changed")
            for offset in range(0, len(data), chunk):
                wanted = min(chunk, len(data) - offset)
                part = self._fs.read(stream, wanted)
                records._require(type(part) is bytes, "malformed accounting read")
                copied.extend(part)
                records._require(len(part) == wanted, "short accounting readback")
            extra = self._fs.read(stream, 1)
            records._require(type(extra) is bytes and extra == b"", "accounting readback has extra EOF")
            records._require(bytes(copied) == data and self._file_checked(self._fs.fstat(stream)) == known_identity
                and self._file_checked(self._fs.lstat(path)) == known_identity, "accounting readback mismatch")
            self._root_checked()
            stage = "close"
            owned, stream = stream, None
            try:
                records._require(self._fs.close(owned) is None, "malformed close acknowledgment")
            except BaseException as error:
                cleanup.append(_failure(error, "close", "cleanup_failed"))
                raise
            status, certainty = "recorded", "confirmed"
        except BaseException as error:
            self._evidence.append((relative, stage, error, attempted_create, known_identity, bytes(copied)))
            if isinstance(error, FileExistsError) and (status == "existing" or (stage == "create" and not create_returned)):
                status, certainty = "existing", "not_applied"
                failure = _failure(error, stage, "collision")
            elif failure is None:
                failure = _failure(error, stage, "mismatch" if isinstance(error, ValueError) else "raised")
        finally:
            if stream is not None:
                owned, stream = stream, None
                try:
                    records._require(self._fs.close(owned) is None, "malformed cleanup acknowledgment")
                except BaseException as error:
                    self._evidence.append((relative, "cleanup", error))
                    cleanup.append(_failure(error, "close", "cleanup_failed"))
            self._busy = False
        if cleanup:
            status = "failed"
            certainty = "unknown" if attempted_create else "not_applied"
        ack = records.AdministrativeJournalAck(status, certainty,
            records._bytes(data) if status == "recorded" else records._bytes(bytes(copied)) if copied else None,
            known_identity, record.requested_outcome if status == "recorded" and type(record) is records.HandoffAdministrationTerminal else None,
            tuple(cleanup), failure)
        self._evidence.append((relative, "ack", ack))
        return ack

    def activate_once(self, record):
        records._require(type(record) is records.HandoffAdministrationActivation, "activation type mismatch")
        ack = self._persist(record, record.activation_id + ".activation.json")
        if ack.status == "recorded" and ack.effect_certainty == "confirmed":
            data = old._validate_byte_record(ack.record, name="actual activation return")
            self._session = _Session(record, data, records._session_id(data, self._root_identity), os.getpid())
        return ack

    def reserve_once(self, record):
        self._owned()
        records._require(type(record) is records.HandoffAdministrationReservation and record.session_id == self._session.session_id,
            "reservation session mismatch")
        ack = self._persist(record, record.activation_id + "." + record.allowance_id + ".reservation.json")
        if ack.status == "recorded" and ack.effect_certainty == "confirmed":
            self._reservations[record.allowance_id] = record
            self._checkpoints[record.allowance_id] = []
        return ack

    def record_checkpoint_once(self, record):
        self._owned()
        reservation = self._reservations.get(record.allowance_id)
        records._require(type(record) is records.HandoffAdministrationCheckpoint and reservation is not None
            and record.session_id == self._session.session_id and record.scope_self_sha256 == reservation.scope_self_sha256
            and record.allowance_id not in self._terminals, "checkpoint has no live reservation")
        prefix = self._checkpoints[record.allowance_id]
        records._require(record.ordinal == len(prefix) and all(x[0].status == "active" for x in prefix), "checkpoint order/inactive prefix")
        leaf = record.activation_id + "." + record.allowance_id + f".{record.ordinal}.checkpoint.json"
        ack = self._persist(record, leaf)
        if ack.status == "recorded" and ack.effect_certainty == "confirmed":
            prefix.append((record, old.FileRef(self.policy.journal_policy.root_path, leaf, ack.record.file_bytes, ack.record.file_sha256)))
        return ack

    def record_terminal_once(self, record):
        self._owned()
        reservation = self._reservations.get(record.allowance_id)
        records._require(type(record) is records.HandoffAdministrationTerminal and reservation is not None
            and record.session_id == self._session.session_id and record.scope_self_sha256 == reservation.scope_self_sha256
            and record.allowance_id not in self._terminals, "terminal has no live reservation/already attempted")
        self._terminals.add(record.allowance_id)
        prefix = self._checkpoints[record.allowance_id]
        records._require(record.checkpoint_record_refs == tuple(x[1] for x in prefix), "terminal invented checkpoint references")
        if record.requested_outcome == "succeeded":
            count = len(records.WRITE_CHECKPOINTS if record.operation == "retain_handoff_witness" else records.READ_CHECKPOINTS)
            records._require(len(prefix) == count and all(x[0].status == "active" for x in prefix), "success terminal without complete active decisions")
        return self._persist(record, record.activation_id + "." + record.allowance_id + ".terminal.json")


def activate_once(journal, record):
    return journal.activate_once(record)


def reserve_once(journal, record):
    return journal.reserve_once(record)


def record_checkpoint_once(journal, record):
    return journal.record_checkpoint_once(record)


def record_terminal_once(journal, record):
    return journal.record_terminal_once(record)


@dataclass(frozen=True, slots=True)
class _AuthorityInspection:
    status: str
    records: tuple
    original_return_acknowledged: bool = False
    unavailable_addresses: tuple[str, ...] = ()
    absent_addresses: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ExpectedScopes:
    plan: old.HandoffPlan
    entries: tuple


def inspect_handoff_authority_records(*, policy, expected_scopes, acquired_records):
    """Inspect fixed evidence without acquiring or reconstructing a live owner.

    Absence requires a clean qualified not-found observation. A failed acquisition
    is unknown, not an absent predecessor. Check all independently visible facts
    even when another prerequisite is unknown; unknowns cannot hide contradictions.
    The returned availability lists are private inspection evidence, not wire data.
    """
    records._require(type(expected_scopes) is _ExpectedScopes, "inspection needs independent plan and scopes")
    plan, expected_scopes = expected_scopes.plan, expected_scopes.entries
    records.validate_handoff_administrative_policy(policy, expected_plan=plan) if type(policy) is records.HandoffAdministrativePolicy else records.validate_handoff_restart_ticket(policy, expected_plan=plan)
    records._require(tuple(x[0] for x in expected_scopes) == policy.grants, "inspection scope coverage/order")
    request_hash = policy.grants[0].request_self_sha256
    addresses = records.derive_handoff_authority_addresses(policy=policy, request_self_sha256=request_hash)
    records._require(tuple(x[0] for x in acquired_records) == addresses, "inspection address coverage/order")
    parsed, availability = {}, {}
    incomplete = False
    for address, read in acquired_records:
        records._shape(read, records.HandoffRestartRead)
        if read.status == "not_found" and not read.cleanup_failures:
            records._require(read.data is None and read.before is None and read.after is None
                and read.failure is not None and read.failure.stage == "open" and read.failure.reason == "not_found",
                "claimed absence contradicts acquired facts")
            availability[address], parsed[address] = "absent", None
            continue
        if read.status != "complete" or read.data is None or read.cleanup_failures:
            incomplete = True
            availability[address], parsed[address] = "unavailable", None
            continue
        availability[address] = "available"
        data = old._validate_byte_record(read.data, name="acquired administrative bytes")
        records._require(data.endswith(b"\n"), "missing journal LF")
        payload = records.strict_canonical_json_loads(data[:-1])
        kind = address.rsplit(".", 2)[-2]
        record = old._decode_dataclass(records._TYPES[kind], payload, name="retained " + kind)
        scope = None if kind == "activation" else next((scope for grant, scope in expected_scopes if grant.allowance_id == record.allowance_id), None)
        records._require(kind == "activation" or scope is not None, "retained record has no accepted scope")
        records.validate_handoff_journal_record(record, policy=policy, expected_plan=plan, scope=scope)
        records._require(data == canonical_json_bytes(old._wire(record)) + b"\n", "noncanonical retained journal")
        parsed[address] = (record, data)

    activation_id = records._activation_id(policy, request_hash)
    activation_entry = parsed[addresses[0]]
    if activation_entry is None:
        incomplete = True
        activation_utc = session_id = None
    else:
        activation, activation_bytes = activation_entry
        records._require(activation.activation_id == activation_id
            and activation.journal_policy_sha256 == records._hash(policy.journal_policy), "retained activation root mismatch")
        activation_utc = activation.observed_utc_ns
        session_id = records._session_id(activation_bytes, policy.journal_policy.root_identity)
    visible_sessions = {entry[0].session_id for address, entry in parsed.items() if address != addresses[0] and entry is not None}
    records._require(len(visible_sessions) <= 1 and (session_id is None or visible_sessions <= {session_id}),
        "retained session mismatch")

    for grant, scope in expected_scopes:
        stem = activation_id + "." + grant.allowance_id
        present = [(address, parsed[address]) for address in addresses if address.startswith(stem + ".") and parsed[address] is not None]
        if not present:
            incomplete = True
            continue
        reservation_address = stem + ".reservation.json"
        reservation = parsed[reservation_address]
        if reservation is None:
            incomplete = True
            records._require(availability[reservation_address] != "absent", "visible checkpoint/terminal with proven absent reservation")
        else:
            records._require(reservation[0].grant_self_sha256 == grant.record_self_sha256, "retained grant hash mismatch")
            records._require(activation_utc is None or reservation[0].observed_utc_ns >= activation_utc,
                "retained reservation precedes known activation")
        for address, (value, data) in present:
            records._require((value.activation_id, value.allowance_id, value.operation, value.slot_id, value.scope_self_sha256) ==
                (activation_id, grant.allowance_id, grant.operation, grant.slot_id, scope.scope_self_sha256), "retained session/scope mismatch")
        names = records.WRITE_CHECKPOINTS if grant.operation == "retain_handoff_witness" else records.READ_CHECKPOINTS
        known_references = {}
        absent_gap = inactive = False
        last_utc = reservation[0].observed_utc_ns if reservation is not None else activation_utc
        for ordinal, name in enumerate(names):
            leaf = stem + f".{ordinal}.checkpoint.json"
            item = parsed[leaf]
            if item is None:
                if availability[leaf] == "absent": absent_gap = True
                else: incomplete = True
                continue
            value, data = item
            records._require(not absent_gap and not inactive and (value.ordinal, value.checkpoint) == (ordinal, name),
                "retained checkpoint contradicts a proven absent/inactive prefix")
            if value.first_utc_ns is not None and value.final_utc_ns is not None:
                if value.status == "active":
                    records._require(last_utc is None or last_utc <= value.first_utc_ns <= value.final_utc_ns,
                        "retained active decision rolled clock backward")
                last_utc = value.final_utc_ns if last_utc is None else max(last_utc, value.final_utc_ns)
            if value.status != "active": inactive = True
            # Per-record validation already checked interval/revocation evidence.
            known_references[ordinal] = old.FileRef(policy.journal_policy.root_path, leaf, len(data), sha256_bytes(data))
        terminal_entry = parsed[stem + ".terminal.json"]
        if terminal_entry is None:
            incomplete = True
        else:
            terminal = terminal_entry[0]
            references = terminal.checkpoint_record_refs
            for ordinal, expected in known_references.items():
                records._require(ordinal < len(references) and references[ordinal] == expected,
                    "historical terminal references contradict visible checkpoints")
            for ordinal, reference in enumerate(references):
                leaf = stem + f".{ordinal}.checkpoint.json"
                records._require(availability[leaf] != "absent", "historical terminal references a proven absent checkpoint")
                if availability[leaf] == "unavailable": incomplete = True
            if terminal.requested_outcome == "succeeded":
                records._require(not inactive and len(references) == len(names), "historical success contradicts known inactive decisions")
    return _AuthorityInspection("incomplete" if incomplete else "corroborated", tuple(parsed.items()), False,
        tuple(address for address in addresses if availability[address] == "unavailable"),
        tuple(address for address in addresses if availability[address] == "absent"))
