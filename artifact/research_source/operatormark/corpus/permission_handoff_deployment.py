"""Inert composition from already independently bound administrative values.

No bootstrap acquisition, activation, capture/restart dispatch, state worker,
credentials, permission changes, production defaults or acceptance emission.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import permission_handoff_administration as administration
from . import permission_handoff_administration_records as records
from .permission_handoff_administration_journal import FileHandoffAdministrationJournal
from .permission_handoff_administration_windows import Win32HandoffAdministrativeReader
from .permission_handoff_windows import build_windows_handoff_providers
from .permission_handoff_restart import HandoffRestartCoordinator


@dataclass(frozen=True, slots=True)
class _PreparedCaptureAdministration:
    bound: administration._BoundAdministration
    authority_delegate: administration.DurableHandoffAuthority
    providers: records.windows.WindowsHandoffProviders
    journal: FileHandoffAdministrationJournal


@dataclass(frozen=True, slots=True)
class _PreparedRestartVerification:
    bound: administration._BoundAdministration
    authority: administration.DurableHandoffRestartAuthority
    reader: Win32HandoffAdministrativeReader
    coordinator: HandoffRestartCoordinator
    journal: FileHandoffAdministrationJournal


def _check_bound(bound, expected):
    records._require(type(bound) is administration._BoundAdministration and type(bound.record) is expected,
        "factory requires completed independent administrative binding")
    if expected is records.HandoffAdministrativePolicy:
        records.validate_handoff_administrative_policy(bound.record, expected_plan=bound.expected_plan)
    else:
        records.validate_handoff_restart_ticket(bound.record, expected_plan=bound.expected_plan, capture_policy=bound.capture_policy)
    records._require(bound.record.deployment_source_files == bound.deployment_source_files, "factory source binding changed")


def prepare_handoff_administration(*, bound, utc_ns_clock, revocation_reader, native_bindings=None, journal_filesystem=None):
    _check_bound(bound, records.HandoffAdministrativePolicy)
    journal = FileHandoffAdministrationJournal(policy=bound.record, expected_plan=bound.expected_plan, filesystem=journal_filesystem)
    authority = administration.DurableHandoffAuthority(bound=bound, journal=journal, utc_ns_clock=utc_ns_clock, revocation_reader=revocation_reader)
    providers = build_windows_handoff_providers(expected_plan=bound.expected_plan, configuration=bound.record.configuration,
        authority_delegate=authority, bindings=native_bindings)
    return _PreparedCaptureAdministration(bound, authority, providers, journal)


def prepare_handoff_restart_verification(*, bound, utc_ns_clock, revocation_reader, native_bindings=None, journal_filesystem=None):
    _check_bound(bound, records.HandoffRestartTicket)
    journal = FileHandoffAdministrationJournal(policy=bound.record, expected_plan=bound.expected_plan, filesystem=journal_filesystem)
    authority = administration.DurableHandoffRestartAuthority(bound=bound, journal=journal, utc_ns_clock=utc_ns_clock, revocation_reader=revocation_reader)
    reader = Win32HandoffAdministrativeReader(expected_plan=bound.expected_plan, authority=authority, bindings=native_bindings)
    return _PreparedRestartVerification(bound, authority, reader, HandoffRestartCoordinator(), journal)
