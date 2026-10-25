"""Filesystem-backed accounting for the injected permission authority.

The caller must provide an existing, dedicated journal root that is outside the
frozen corpus output tree and is independently controlled from untrusted
writers.  The path and identity checks below reject detected links, reparse
points, type changes, and node substitutions; they are not a proof against a
hostile administrator changing the namespace between system calls.

This component provides immutable process-restart accounting.  It does not
provide authenticated administration, rollback resistance, or power-loss
qualification, and it neither provisions nor selects a production root.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
import stat
from threading import RLock
from typing import BinaryIO, Callable, Literal, Protocol

from .canonical import (
    canonical_json_bytes,
    require_exact_keys,
    sha256_bytes,
    strict_canonical_json_loads,
)
from . import permission_authority as authority


__all__ = ["FilePermissionAuthorityJournal"]


_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_RECORD_BYTES = 64 * 1024
_SESSION_DOMAIN = b"operatormark-permission-authority-filesystem-session-v1\n"


class _JournalFilesystem(Protocol):
    def lstat(self, path: Path) -> os.stat_result: ...

    def fstat(self, stream: BinaryIO) -> os.stat_result: ...

    def mkdir(self, path: Path) -> None: ...

    def open_exclusive(self, path: Path) -> BinaryIO: ...

    def open_read(self, path: Path) -> BinaryIO: ...

    def write(self, stream: BinaryIO, data: bytes) -> int: ...

    def read(self, stream: BinaryIO, size: int) -> bytes: ...

    def flush(self, stream: BinaryIO) -> None: ...

    def sync(self, stream: BinaryIO) -> None: ...

    def close(self, stream: BinaryIO) -> None: ...


class _OSFilesystem:
    @staticmethod
    def lstat(path: Path) -> os.stat_result:
        return os.lstat(path)

    @staticmethod
    def fstat(stream: BinaryIO) -> os.stat_result:
        return os.fstat(stream.fileno())

    @staticmethod
    def mkdir(path: Path) -> None:
        os.mkdir(path)

    @staticmethod
    def open_exclusive(path: Path) -> BinaryIO:
        return path.open("xb", buffering=0)

    @staticmethod
    def open_read(path: Path) -> BinaryIO:
        return path.open("rb", buffering=0)

    @staticmethod
    def write(stream: BinaryIO, data: bytes) -> int:
        return stream.write(data)

    @staticmethod
    def read(stream: BinaryIO, size: int) -> bytes:
        return stream.read(size)

    @staticmethod
    def flush(stream: BinaryIO) -> None:
        stream.flush()

    @staticmethod
    def sync(stream: BinaryIO) -> None:
        os.fsync(stream.fileno())

    @staticmethod
    def close(stream: BinaryIO) -> None:
        stream.close()


@dataclass(frozen=True, slots=True)
class _NodeIdentity:
    device: int
    inode: int
    mode_type: int
    file_attributes: int


@dataclass(frozen=True, slots=True)
class _OwnedSession:
    activation: authority.PermissionAttemptActivation
    session_id: str
    attempt_directory: Path
    attempt_identity: _NodeIdentity


@dataclass(frozen=True, slots=True)
class _PersistenceOutcome:
    status: Literal["persisted", "existing", "failed"]
    effect_certainty: Literal["confirmed", "not_applied", "unknown"]


class _UnsafeFilesystemState(RuntimeError):
    pass


class _RecordFormatError(RuntimeError):
    pass


def _attributes(value: object) -> int:
    attributes = getattr(value, "st_file_attributes", 0)
    return attributes if type(attributes) is int else 0


def _node_identity(value: object, *, directory: bool) -> _NodeIdentity:
    mode = getattr(value, "st_mode", None)
    if type(mode) is not int:
        raise _UnsafeFilesystemState("filesystem metadata has no exact mode")
    attributes = _attributes(value)
    if stat.S_ISLNK(mode) or attributes & _REPARSE_POINT:
        raise _UnsafeFilesystemState("filesystem node is a link or reparse point")
    expected_type = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not expected_type:
        kind = "directory" if directory else "regular file"
        raise _UnsafeFilesystemState(f"filesystem node is not a {kind}")
    device = getattr(value, "st_dev", None)
    inode = getattr(value, "st_ino", None)
    if type(device) is not int or type(inode) is not int:
        raise _UnsafeFilesystemState("filesystem metadata has no exact identity")
    return _NodeIdentity(device, inode, stat.S_IFMT(mode), attributes)


def _decode_record(data: bytes, record_type: type, validator: Callable[[object], None]):
    try:
        payload = strict_canonical_json_loads(data)
        require_exact_keys(
            payload,
            (item.name for item in fields(record_type)),
            name=record_type.__name__,
        )
        record = record_type(**payload)
        validator(record)
        if canonical_json_bytes(asdict(record)) != data:
            raise ValueError("record did not round-trip to its exact canonical bytes")
        return record
    except Exception as exc:
        raise _RecordFormatError("journal record is malformed") from exc


class FilePermissionAuthorityJournal:
    """One-process owner of immutable authority records under an explicit root."""

    __slots__ = (
        "_filesystem",
        "_lock",
        "_operational_disabled",
        "_owned_session",
        "_owner_pid",
        "_reservations_by_allowance",
        "_reservations_by_id",
        "_root",
        "_root_identity",
        "_terminal_attempts",
    )

    def __init__(
        self,
        journal_root: str | os.PathLike[str],
        *,
        _filesystem: _JournalFilesystem | None = None,
    ) -> None:
        try:
            supplied = Path(os.fspath(journal_root))
        except (TypeError, ValueError) as exc:
            raise authority.PermissionAuthorityContractError(
                "an explicit journal-root path is mandatory"
            ) from exc
        if not supplied.is_absolute():
            raise authority.PermissionAuthorityContractError(
                "journal root must be an absolute path"
            )
        self._root = Path(os.path.abspath(supplied))
        self._filesystem = _filesystem if _filesystem is not None else _OSFilesystem()
        try:
            self._root_identity = _node_identity(
                self._filesystem.lstat(self._root), directory=True
            )
        except Exception as exc:
            raise authority.PermissionAuthorityContractError(
                "journal root must be an existing non-reparse directory"
            ) from exc
        self._owner_pid = os.getpid()
        self._lock = RLock()
        self._owned_session: _OwnedSession | None = None
        self._operational_disabled = False
        self._reservations_by_allowance: dict[
            str, authority.PermissionOperationReservation
        ] = {}
        self._reservations_by_id: dict[
            str, authority.PermissionOperationReservation
        ] = {}
        self._terminal_attempts: dict[
            str, authority.PermissionTerminalRecord | None
        ] = {}

    def _verify_root(self) -> None:
        current = _node_identity(self._filesystem.lstat(self._root), directory=True)
        if current != self._root_identity:
            raise _UnsafeFilesystemState("journal root identity changed")

    def _verify_owned_directory(self) -> None:
        self._verify_root()
        owned = self._owned_session
        if owned is None:
            raise _UnsafeFilesystemState("no journal session is owned")
        current = _node_identity(
            self._filesystem.lstat(owned.attempt_directory), directory=True
        )
        if current != owned.attempt_identity:
            raise _UnsafeFilesystemState("journal attempt-directory identity changed")

    def _verify_owned_namespace(self) -> None:
        self._verify_owned_directory()
        owned = self._owned_session
        assert owned is not None
        raw = self._read_file_bytes(
            owned.attempt_directory / "activation.json",
            maximum_bytes=_MAX_RECORD_BYTES,
            verify_parent=self._verify_owned_directory,
        )
        stored = _decode_record(
            raw,
            authority.PermissionAttemptActivation,
            authority._validate_attempt_activation,
        )
        if (
            stored != owned.activation
            or stored.attempt_id
            != "s4-" + stored.state4_authorization_file_sha256[:24]
        ):
            raise _UnsafeFilesystemState("journal activation record changed")

    def _read_file_bytes(
        self,
        path: Path,
        *,
        maximum_bytes: int,
        verify_parent: Callable[[], None],
    ) -> bytes:
        verify_parent()
        before_stat = self._filesystem.lstat(path)
        before = _node_identity(before_stat, directory=False)
        stream = self._filesystem.open_read(path)
        try:
            opened = _node_identity(self._filesystem.fstat(stream), directory=False)
            if opened != before:
                raise _UnsafeFilesystemState("journal file identity changed while opening")
            chunks: list[bytes] = []
            remaining = maximum_bytes + 1
            while remaining:
                chunk = self._filesystem.read(stream, remaining)
                if type(chunk) is not bytes:
                    raise _UnsafeFilesystemState("journal read returned non-bytes")
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > maximum_bytes:
                raise _RecordFormatError("journal record exceeds the bounded size")
        except Exception:
            try:
                self._filesystem.close(stream)
            except Exception:
                pass
            raise
        self._filesystem.close(stream)
        after_stat = self._filesystem.lstat(path)
        after = _node_identity(after_stat, directory=False)
        if after != before:
            raise _UnsafeFilesystemState("journal file identity changed during readback")
        size = getattr(after_stat, "st_size", None)
        if type(size) is not int or size != len(data):
            raise _UnsafeFilesystemState("journal file size changed during readback")
        verify_parent()
        return data

    def _record_exists(self, path: Path) -> bool:
        try:
            self._filesystem.lstat(path)
        except FileNotFoundError:
            return False
        return True

    def _write_exclusive_verified(
        self,
        path: Path,
        data: bytes,
        *,
        verify_parent: Callable[[], None],
    ) -> _PersistenceOutcome:
        try:
            verify_parent()
        except Exception:
            return _PersistenceOutcome("failed", "not_applied")
        try:
            stream = self._filesystem.open_exclusive(path)
        except FileExistsError:
            return _PersistenceOutcome("existing", "not_applied")
        except Exception:
            # An exclusive-create implementation may have created a partial node
            # before surfacing its error, so the effect is conservatively unknown.
            return _PersistenceOutcome("failed", "unknown")
        try:
            offset = 0
            while offset != len(data):
                written = self._filesystem.write(stream, data[offset:])
                if (
                    type(written) is not int
                    or written <= 0
                    or written > len(data) - offset
                ):
                    raise OSError("journal write made invalid progress")
                offset += written
            self._filesystem.flush(stream)
            self._filesystem.sync(stream)
        except Exception:
            try:
                self._filesystem.close(stream)
            except Exception:
                pass
            return _PersistenceOutcome("failed", "unknown")
        try:
            self._filesystem.close(stream)
        except Exception:
            return _PersistenceOutcome("failed", "unknown")
        try:
            observed = self._read_file_bytes(
                path,
                maximum_bytes=len(data),
                verify_parent=verify_parent,
            )
            if observed != data:
                raise _UnsafeFilesystemState("journal readback bytes differ")
        except Exception:
            return _PersistenceOutcome("failed", "unknown")
        return _PersistenceOutcome("persisted", "confirmed")

    @staticmethod
    def _activation_acknowledgment(
        activation: authority.PermissionAttemptActivation,
        status: Literal["activated", "existing", "incomplete", "conflict", "failed"],
        certainty: Literal["confirmed", "not_applied", "unknown"],
        session_id: str | None = None,
    ) -> authority.AttemptActivationAcknowledgment:
        result = authority.AttemptActivationAcknowledgment(
            status, certainty, activation, session_id
        )
        authority.validate_attempt_activation_acknowledgment(
            result, expected_activation=activation
        )
        return result

    @staticmethod
    def _reservation_acknowledgment(
        reservation: authority.PermissionOperationReservation,
        status: Literal["reserved", "existing", "conflict", "failed"],
        certainty: Literal["confirmed", "not_applied", "unknown"],
    ) -> authority.OperationReservationAcknowledgment:
        result = authority.OperationReservationAcknowledgment(
            status, certainty, reservation
        )
        authority.validate_operation_reservation_acknowledgment(
            result, expected_reservation=reservation
        )
        return result

    @staticmethod
    def _terminal_acknowledgment(
        terminal: authority.PermissionTerminalRecord,
        status: Literal["recorded", "failed"],
        certainty: Literal["confirmed", "not_applied", "unknown"],
        recorded_outcome: Literal["succeeded", "refused", "failed"] | None,
    ) -> authority.TerminalRecordAcknowledgment:
        result = authority.TerminalRecordAcknowledgment(
            status, certainty, terminal, recorded_outcome
        )
        authority.validate_terminal_record_acknowledgment(
            result, expected_terminal=terminal
        )
        return result

    def _classify_existing_activation(
        self,
        activation: authority.PermissionAttemptActivation,
        attempt_directory: Path,
    ) -> authority.AttemptActivationAcknowledgment:
        try:
            self._verify_root()
            directory_identity = _node_identity(
                self._filesystem.lstat(attempt_directory), directory=True
            )
        except _UnsafeFilesystemState:
            return self._activation_acknowledgment(
                activation, "conflict", "not_applied"
            )
        except Exception:
            return self._activation_acknowledgment(
                activation, "failed", "not_applied"
            )

        def verify_parent() -> None:
            self._verify_root()
            current = _node_identity(
                self._filesystem.lstat(attempt_directory), directory=True
            )
            if current != directory_identity:
                raise _UnsafeFilesystemState("existing attempt identity changed")

        record_path = attempt_directory / "activation.json"
        try:
            raw = self._read_file_bytes(
                record_path,
                maximum_bytes=_MAX_RECORD_BYTES,
                verify_parent=verify_parent,
            )
        except FileNotFoundError:
            return self._activation_acknowledgment(
                activation, "incomplete", "not_applied"
            )
        except (_RecordFormatError, _UnsafeFilesystemState):
            return self._activation_acknowledgment(
                activation, "incomplete", "not_applied"
            )
        except Exception:
            return self._activation_acknowledgment(
                activation, "failed", "not_applied"
            )
        try:
            current = _decode_record(
                raw,
                authority.PermissionAttemptActivation,
                authority._validate_attempt_activation,
            )
            if current.attempt_id != "s4-" + current.state4_authorization_file_sha256[:24]:
                raise _RecordFormatError("stored activation attempt relation is invalid")
        except _RecordFormatError:
            return self._activation_acknowledgment(
                activation, "incomplete", "not_applied"
            )
        status = "existing" if current == activation else "conflict"
        return self._activation_acknowledgment(
            activation, status, "not_applied"
        )

    def activate_attempt(
        self, activation: authority.PermissionAttemptActivation
    ) -> authority.AttemptActivationAcknowledgment:
        authority._validate_attempt_activation(activation)
        expected_attempt_id = "s4-" + activation.state4_authorization_file_sha256[:24]
        if activation.attempt_id != expected_attempt_id:
            raise authority.PermissionAuthorityContractError(
                "attempt activation attempt_id does not derive from full A4"
            )
        record_bytes = canonical_json_bytes(asdict(activation))
        attempt_directory = self._root / expected_attempt_id
        with self._lock:
            if os.getpid() != self._owner_pid:
                return self._activation_acknowledgment(
                    activation, "failed", "not_applied"
                )
            try:
                self._verify_root()
            except Exception:
                return self._activation_acknowledgment(
                    activation, "failed", "not_applied"
                )
            if self._owned_session is not None:
                status = (
                    "existing"
                    if self._owned_session.activation == activation
                    else "conflict"
                )
                return self._activation_acknowledgment(
                    activation, status, "not_applied"
                )
            try:
                self._filesystem.mkdir(attempt_directory)
            except FileExistsError:
                return self._classify_existing_activation(
                    activation, attempt_directory
                )
            except Exception:
                return self._activation_acknowledgment(
                    activation, "failed", "unknown"
                )
            try:
                self._verify_root()
                attempt_identity = _node_identity(
                    self._filesystem.lstat(attempt_directory), directory=True
                )
            except Exception:
                return self._activation_acknowledgment(
                    activation, "failed", "unknown"
                )

            def verify_parent() -> None:
                self._verify_root()
                current = _node_identity(
                    self._filesystem.lstat(attempt_directory), directory=True
                )
                if current != attempt_identity:
                    raise _UnsafeFilesystemState(
                        "new attempt-directory identity changed"
                    )

            persisted = self._write_exclusive_verified(
                attempt_directory / "activation.json",
                record_bytes,
                verify_parent=verify_parent,
            )
            if persisted.status != "persisted":
                # The directory claim itself was made by this call, even if a
                # later exclusive-create unexpectedly reported an existing file.
                return self._activation_acknowledgment(
                    activation, "failed", "unknown"
                )
            session_id = sha256_bytes(_SESSION_DOMAIN + record_bytes)
            self._owned_session = _OwnedSession(
                activation,
                session_id,
                attempt_directory,
                attempt_identity,
            )
            return self._activation_acknowledgment(
                activation, "activated", "confirmed", session_id
            )

    def _read_reservation(
        self, path: Path
    ) -> authority.PermissionOperationReservation:
        raw = self._read_file_bytes(
            path,
            maximum_bytes=_MAX_RECORD_BYTES,
            verify_parent=self._verify_owned_namespace,
        )
        return _decode_record(
            raw,
            authority.PermissionOperationReservation,
            authority._validate_operation_reservation,
        )

    def reserve_operation(
        self, reservation: authority.PermissionOperationReservation
    ) -> authority.OperationReservationAcknowledgment:
        authority._validate_operation_reservation(reservation)
        record_bytes = canonical_json_bytes(asdict(reservation))
        with self._lock:
            owned = self._owned_session
            if (
                os.getpid() != self._owner_pid
                or owned is None
                or reservation.session_id != owned.session_id
                or reservation.state4_authorization_file_sha256
                != owned.activation.state4_authorization_file_sha256
            ):
                return self._reservation_acknowledgment(
                    reservation, "conflict", "not_applied"
                )
            if self._operational_disabled:
                return self._reservation_acknowledgment(
                    reservation, "failed", "not_applied"
                )
            try:
                self._verify_owned_namespace()
            except Exception:
                self._operational_disabled = True
                return self._reservation_acknowledgment(
                    reservation, "failed", "not_applied"
                )
            path = owned.attempt_directory / f"reservation-{reservation.allowance_id}.json"
            try:
                exists = self._record_exists(path)
            except Exception:
                self._operational_disabled = True
                return self._reservation_acknowledgment(
                    reservation, "failed", "not_applied"
                )
            if exists:
                try:
                    current = self._read_reservation(path)
                except Exception:
                    self._operational_disabled = True
                    return self._reservation_acknowledgment(
                        reservation, "failed", "not_applied"
                    )
                status = "existing" if current == reservation else "conflict"
                return self._reservation_acknowledgment(
                    reservation, status, "not_applied"
                )
            if reservation.allowance_id in self._reservations_by_allowance:
                self._operational_disabled = True
                return self._reservation_acknowledgment(
                    reservation, "failed", "not_applied"
                )
            persisted = self._write_exclusive_verified(
                path,
                record_bytes,
                verify_parent=self._verify_owned_namespace,
            )
            if persisted.status == "existing":
                try:
                    current = self._read_reservation(path)
                except Exception:
                    self._operational_disabled = True
                    return self._reservation_acknowledgment(
                        reservation, "failed", "not_applied"
                    )
                status = "existing" if current == reservation else "conflict"
                return self._reservation_acknowledgment(
                    reservation, status, "not_applied"
                )
            if persisted.status == "failed":
                if persisted.effect_certainty == "unknown":
                    self._operational_disabled = True
                return self._reservation_acknowledgment(
                    reservation, "failed", persisted.effect_certainty
                )
            self._reservations_by_allowance[reservation.allowance_id] = reservation
            self._reservations_by_id[reservation.reservation_id] = reservation
            return self._reservation_acknowledgment(
                reservation, "reserved", "confirmed"
            )

    @staticmethod
    def _terminal_matches_reservation(
        terminal: authority.PermissionTerminalRecord,
        reservation: authority.PermissionOperationReservation,
    ) -> bool:
        return all(
            getattr(terminal, name) == getattr(reservation, name)
            for name in (
                "session_id",
                "reservation_id",
                "allowance_id",
                "state4_authorization_file_sha256",
                "request_self_sha256",
                "operation",
                "scope_self_sha256",
            )
        )

    def record_terminal(
        self, terminal: authority.PermissionTerminalRecord
    ) -> authority.TerminalRecordAcknowledgment:
        authority._validate_terminal_record(terminal)
        record_bytes = canonical_json_bytes(asdict(terminal))
        with self._lock:
            owned = self._owned_session
            reservation = self._reservations_by_id.get(terminal.reservation_id)
            if (
                os.getpid() != self._owner_pid
                or owned is None
                or terminal.session_id != owned.session_id
                or reservation is None
                or not self._terminal_matches_reservation(terminal, reservation)
            ):
                return self._terminal_acknowledgment(
                    terminal, "failed", "not_applied", None
                )
            previous = self._terminal_attempts.get(terminal.reservation_id, ...)
            if previous is None:
                return self._terminal_acknowledgment(
                    terminal, "failed", "unknown", None
                )
            if previous is not ...:
                if previous == terminal:
                    try:
                        self._verify_owned_namespace()
                        path = (
                            owned.attempt_directory
                            / f"terminal-{terminal.reservation_id}.json"
                        )
                        raw = self._read_file_bytes(
                            path,
                            maximum_bytes=_MAX_RECORD_BYTES,
                            verify_parent=self._verify_owned_namespace,
                        )
                        stored = _decode_record(
                            raw,
                            authority.PermissionTerminalRecord,
                            authority._validate_terminal_record,
                        )
                        if stored != terminal:
                            raise _UnsafeFilesystemState(
                                "terminal record changed after confirmation"
                            )
                    except Exception:
                        return self._terminal_acknowledgment(
                            terminal, "failed", "not_applied", None
                        )
                    return self._terminal_acknowledgment(
                        terminal,
                        "recorded",
                        "confirmed",
                        terminal.requested_outcome,
                    )
                return self._terminal_acknowledgment(
                    terminal, "failed", "not_applied", None
                )
            try:
                self._verify_owned_namespace()
                stored_reservation = self._read_reservation(
                    owned.attempt_directory
                    / f"reservation-{reservation.allowance_id}.json"
                )
                if stored_reservation != reservation:
                    raise _UnsafeFilesystemState("reservation record changed")
            except Exception:
                return self._terminal_acknowledgment(
                    terminal, "failed", "not_applied", None
                )
            path = owned.attempt_directory / f"terminal-{terminal.reservation_id}.json"
            try:
                exists = self._record_exists(path)
            except Exception:
                return self._terminal_acknowledgment(
                    terminal, "failed", "not_applied", None
                )
            if exists:
                # No terminal written by this live owner is unambiguously
                # adoptable merely because its bytes happen to look complete.
                return self._terminal_acknowledgment(
                    terminal, "failed", "not_applied", None
                )
            persisted = self._write_exclusive_verified(
                path,
                record_bytes,
                verify_parent=self._verify_owned_namespace,
            )
            if persisted.status == "persisted":
                self._terminal_attempts[terminal.reservation_id] = terminal
                return self._terminal_acknowledgment(
                    terminal,
                    "recorded",
                    "confirmed",
                    terminal.requested_outcome,
                )
            if persisted.effect_certainty == "unknown":
                self._terminal_attempts[terminal.reservation_id] = None
                self._operational_disabled = True
            return self._terminal_acknowledgment(
                terminal, "failed", persisted.effect_certainty, None
            )
