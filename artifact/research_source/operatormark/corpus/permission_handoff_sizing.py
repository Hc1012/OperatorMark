"""Pure physical-instance ledger for the permission handoff.

Only measured file lengths are counted.  This component deliberately cannot
turn a partial inventory into the frozen writer's unresolved production budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .permission_handoff_records import (
    HandoffPhysicalFile,
    _exact_int,
    _relative_path,
    _text,
)


class PermissionHandoffSizingError(ValueError):
    """A physical ledger overlaps or claims an unsupported derivation."""


@dataclass(frozen=True, slots=True)
class HandoffSizeDelta:
    observed_total_physical_bytes: int
    accepted_attempt_M_delta_bytes: int
    controls_subset_delta_bytes: int
    separate_B_delta_bytes: int
    external_preexisting_bytes: int
    unresolved_contributions: tuple[str, ...]
    production_disk_estimate_available: Literal[False] = False


def _raise(message: str) -> None:
    raise PermissionHandoffSizingError(message)


def _count_measured_instance(item: HandoffPhysicalFile) -> int:
    if item.classification != "exact" or item.byte_count is None:
        _raise("only an exact observed physical instance can be counted")
    _exact_int(item.byte_count, name="physical instance byte_count")
    if item.basis != "observed_file_length" or "bytes_retained_and_measured" not in item.prerequisites:
        _raise("exact physical instance lacks its measured-length derivation")
    return item.byte_count


def _reject_duplicate_ownership(
    inventory: tuple[HandoffPhysicalFile, ...],
    *,
    existing_inventory: tuple[HandoffPhysicalFile, ...] = (),
) -> None:
    seen: dict[tuple[str, str], HandoffPhysicalFile] = {}
    for item in (*existing_inventory, *inventory):
        key = (item.root_binding_id, item.relative_path)
        prior = seen.get(key)
        if prior is not None:
            _raise(
                "physical root/path appears more than once; references and container members "
                "are not additional physical instances"
            )
        seen[key] = item


def validate_handoff_physical_inventory(
    inventory: tuple[HandoffPhysicalFile, ...]
) -> None:
    if type(inventory) is not tuple:
        _raise("physical inventory must be an exact tuple")
    _reject_duplicate_ownership(inventory)
    for index, item in enumerate(inventory):
        if type(item) is not HandoffPhysicalFile:
            _raise(f"physical inventory item {index} has the wrong exact type")
        _text(item.root_binding_id, name=f"inventory[{index}].root_binding_id")
        _relative_path(item.relative_path, name=f"inventory[{index}].relative_path")
        if item.representation not in {"loose_file", "container"}:
            _raise("physical representation is invalid")
        if item.owner not in {
            "accepted_attempt_M", "separate_B", "external_preexisting", "unresolved"
        }:
            _raise("physical owner is invalid")
        if type(item.controls_subset) is not bool:
            _raise("controls_subset must be an exact Boolean")
        if item.classification not in {"exact", "conditional_exact", "upper_bound", "unresolved"}:
            _raise("physical classification is invalid")
        if type(item.prerequisites) is not tuple or any(
            type(value) is not str or not value for value in item.prerequisites
        ):
            _raise("physical prerequisites must be closed non-empty strings")
        _text(item.basis, name=f"inventory[{index}].basis")
        if item.classification == "exact":
            _count_measured_instance(item)
        elif item.classification == "unresolved":
            if item.byte_count is not None:
                _raise("unresolved physical size must not carry a fabricated count")
        else:
            # The initial handoff component has no implemented prospective
            # conditional/bounded derivation.  Keeping such an entry is valid
            # only as explicitly unresolved, never as a counted estimate.
            _raise("unsupported prospective classification must remain unresolved")
        if item.controls_subset and item.owner != "accepted_attempt_M":
            _raise("controls_subset must be a subtotal of accepted_attempt_M")


def derive_handoff_size_delta(
    *,
    measured_files: tuple[tuple[HandoffPhysicalFile, bytes | None], ...],
    existing_inventory: tuple[HandoffPhysicalFile, ...],
    expected_owner: Literal[
        "accepted_attempt_M", "separate_B", "external_preexisting", "unresolved"
    ],
) -> HandoffSizeDelta:
    if type(measured_files) is not tuple or any(type(item) is not tuple or len(item) != 2 for item in measured_files):
        _raise("measured_files must pair each physical entry with actual bytes (or unavailable null)")
    inventory = tuple(item for item, _ in measured_files)
    validate_handoff_physical_inventory(inventory)
    validate_handoff_physical_inventory(existing_inventory)
    _reject_duplicate_ownership(inventory, existing_inventory=existing_inventory)
    if expected_owner not in {
        "accepted_attempt_M", "separate_B", "external_preexisting", "unresolved"
    }:
        _raise("expected owner is invalid")
    observed_total = 0
    m_delta = 0
    controls = 0
    b_delta = 0
    external = 0
    unresolved: list[str] = []
    for item, data in measured_files:
        if item.owner not in {expected_owner, "external_preexisting", "unresolved"}:
            _raise("measured physical instance has unexpected ownership")
        if item.classification != "exact":
            if data is not None:
                _raise("supplied measured bytes must be counted exactly, not discarded as unresolved")
            unresolved.append(f"{item.root_binding_id}:{item.relative_path}:{item.basis}")
            continue
        count = _count_measured_instance(item)
        if type(data) is not bytes or len(data) != count:
            _raise("exact classification is not proof: actual supplied bytes must match the count")
        observed_total += count
        if item.owner == "accepted_attempt_M":
            m_delta += count
            if item.controls_subset:
                controls += count
        elif item.owner == "separate_B":
            b_delta += count
        elif item.owner == "external_preexisting":
            external += count
        else:
            unresolved.append(f"{item.root_binding_id}:{item.relative_path}:ownership_unavailable")
    if controls > m_delta:
        _raise("controls bytes are not a subtotal of accepted-attempt M")
    return HandoffSizeDelta(
        observed_total,
        m_delta,
        controls,
        b_delta,
        external,
        tuple(unresolved),
        False,
    )
