"""Partial, non-production writer-size derivations for the frozen corpus format.

This module deliberately does not construct the fourteen-key State 3 disk
estimate.  The manifest/audit, controls-only, and retained-State-3 components
remain unresolved, so no five-number caller assertion can establish a
production storage budget.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Callable
from typing import Literal, Mapping, NoReturn

from .corpus_format import (
    BENCHMARK,
    FINAL_VERIFICATION_SCHEMA,
    SHARD_SIZE,
    SPLITS,
    SPLIT_GLOBAL_RANGES,
    SPLIT_SAMPLE_COUNTS,
    SPLIT_SHARD_COUNTS,
    TENSOR_FAMILY_ORDER,
    TENSOR_FAMILY_SPECS,
)


SizeClassification = Literal["exact", "conditional_exact", "upper_bound", "unresolved"]


@dataclass(frozen=True)
class SizeContribution:
    """One byte contribution with its proof classification and explicit basis."""

    bytes: int | None
    classification: SizeClassification
    basis: str
    prerequisites: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoreWriterSizes:
    """The bounded core only; this is not a production disk-estimate record."""

    raw_seven_tensor_payload: SizeContribution
    npy_headers: SizeContribution
    controls_jsonl: SizeContribution
    final_verification_jsonl: SizeContribution
    primary_jsonl: SizeContribution
    core_partial: SizeContribution
    projected_manifest_and_audit_bytes: SizeContribution
    controls_only_bytes: SizeContribution
    projected_audit_bytes: SizeContribution
    controls_jsonl_by_split: Mapping[str, int]
    final_verification_jsonl_by_split: Mapping[str, int]
    npy_header_bytes_by_family: Mapping[str, int]
    unresolved_components: tuple[str, ...]
    production_disk_estimate_available: bool


@dataclass(frozen=True)
class RetainedFileGroup:
    """One physical retained-file group in the reviewed State 3 success path."""

    name: str
    files: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.files)


class IncompleteDiskComponentsError(RuntimeError):
    """Raised whenever partial sizing is presented as production-complete."""


DISK_COMPONENT_MEMBERSHIP = MappingProxyType({
    "projected_npy_header_bytes": (
        "Headers, preambles, and alignment for the 532 actual NPY files only."
    ),
    "projected_jsonl_bytes": (
        "Controls and final-verification JSONL, plus only separately named future JSONL."
    ),
    "projected_manifest_and_audit_bytes": (
        "Every other retained accepted-corpus-attempt file: all 83 controls JSON "
        "documents; final shard/split/context/corpus/completion metadata; complete "
        "rights/provenance/chronology/access/failure/RNG/runtime/storage/audit evidence; "
        "and actual retained copies or containers. No unclassified residual bucket."
    ),
    "controls_only_bytes": (
        "Rho payload and headers, controls JSONL, all 83 controls JSON documents, and "
        "explicitly assigned controls-stage retained evidence; a subtotal of accepted "
        "bytes, never an additional free-space-formula addend."
    ),
    "projected_audit_bytes": (
        "Unresolved complete retained State 3 run/continuation footprint, separate from E: "
        "the 22-file successful run, applicable preserved pre-latch records, and failed "
        "state3_audit.zip.__creating__ staging under an explicit success/failure inventory. "
        "No production meaning is adopted."
    ),
})

FROZEN_FREE_SPACE_ACCOUNTING = MappingProxyType({
    "estimated_accepted_bytes": "E = 420872192 + H + J + M",
    "original_accepted_copy_bytes": "E",
    "active_staging_bytes": "E",
    "readback_atomic_publication_bytes": "E",
    "preserved_failed_attempt_allowance_bytes": "E",
    "controls_only_bytes": "C is a subtotal of E and is not added again",
    "projected_audit_bytes": "B is separate from E",
    "required_free_bytes": "max(5368709120, 4*E + B + 1073741824)",
})

STATE3_SUCCESSFUL_RUN_INVENTORY = (
    RetainedFileGroup(
        "administrative",
        ("additional_thread_policy.json", "administrative_preflight_receipt.json"),
    ),
    RetainedFileGroup(
        "attempt_and_latch",
        (
            "storage_attempt_{attempt:04d}/storage_preflight_before.json",
            "state3_execution_latch.json",
        ),
    ),
    RetainedFileGroup(
        "standalone_runtime_and_audit_inputs",
        (
            "environment_runtime.json",
            "compute_feasibility_preflight_bindings.json",
            "measurement_bundle.json",
            "storage_preflight_after.json",
            "source_blob_verification.json",
            "failure_inventory.json",
            "prohibited_action_counts.json",
        ),
    ),
    RetainedFileGroup(
        "audit_evidence",
        (
            "audit_evidence/state3_audit.json",
            "audit_evidence/environment_runtime.json",
            "audit_evidence/storage_preflight_before.json",
            "audit_evidence/compute_feasibility_report.json",
            "audit_evidence/storage_preflight_after.json",
            "audit_evidence/source_blob_verification.json",
            "audit_evidence/failure_inventory.json",
            "audit_evidence/prohibited_action_counts.json",
        ),
    ),
    RetainedFileGroup("archive", ("state3_audit.zip",)),
    RetainedFileGroup(
        "verification_and_completion",
        ("independent_verification.json", "workflow_completion.json"),
    ),
)

STATE3_REFERENCE_ONLY_INPUTS = (
    "external protocol/authorization/audit bindings",
    "source Git blobs",
    "seventeen historical failure records",
)
STATE3_APPLICABLE_PRE_LATCH_FILE_PATTERNS = (
    "storage_attempt_{attempt:04d}/storage_preflight_before.json",
    "pre_latch_attempt_{attempt:04d}.json",
)
STATE3_PRE_LATCH_ACCOUNTING_RULE = (
    "Count the current successful attempt's storage_preflight_before.json once in the "
    "22-file inventory; add only physically distinct preserved prior/applicable pre-latch "
    "records selected by the explicit attempt inventory."
)
STATE3_FAILED_ZIP_STAGING_ARTIFACT = "state3_audit.zip.__creating__"
STATE3_SUCCESSFUL_RUN_FILE_COUNT = sum(
    group.count for group in STATE3_SUCCESSFUL_RUN_INVENTORY
)
STATE3_ZIP_MEMBER_ORDER = (
    "state3_audit.json",
    "environment_runtime.json",
    "storage_preflight_before.json",
    "compute_feasibility_report.json",
    "storage_preflight_after.json",
    "source_blob_verification.json",
    "failure_inventory.json",
    "prohibited_action_counts.json",
    "member_hash_manifest.json",
)
STATE3_ZIP_FORMAT = MappingProxyType({
    "compression": "ZIP_STORED",
    "timestamp": (1980, 1, 1, 0, 0, 0),
    "create_system": 0,
    "create_version": 20,
    "extract_version": 20,
    "flag_bits": 0,
    "volume": 0,
    "internal_attr": 0,
    "external_attr": 0x20,
    "seekable_output": True,
    "data_descriptors": False,
    "member_extra": b"",
    "member_comment": b"",
    "archive_comment": b"",
    "allow_zip64": False,
    "zip64_overflow": "raise",
    "structural_size_rule": "sum(member_payload_bytes) + 1174",
    "structural_size_prerequisite": (
        "independent complete raw ZIP-structure verification"
    ),
    "member_hash_manifest_retained_loose": False,
    "raw_child_stdout_stderr_retained_loose": False,
    "finalizer_or_verifier_receipt_retained_loose": False,
})
STATE3_ZIP_CLASSIC_STRUCTURAL_OVERHEAD_BYTES = 1_174
STATE3_ZIP_CLASSIC_STRUCTURAL_OVERHEAD = SizeContribution(
    STATE3_ZIP_CLASSIC_STRUCTURAL_OVERHEAD_BYTES,
    "conditional_exact",
    "Reviewed classic stored-ZIP structure for the exact nine ordered member names.",
    ("independent complete raw ZIP-structure verification",),
)

_HASH_TEXT = "0" * 64
_MAX_FLOAT_HEX_TEXT = "-0x1.fffffffffffffp+1023"
_CONTROL_SCHEMA = "operatormark.darcy_branch_c_v1.state4_control_record.v1"
_RAW_ELEMENT_BYTES = 8
_NPY_V2_PREAMBLE_BYTES = 12
_NPY_ALIGNMENT = 64


def _canonical_json_length(value: object) -> int:
    return len(json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8"))


def _sample_identity(split: str, local_index: int) -> tuple[int, int, str, str]:
    shard_index = local_index // SHARD_SIZE
    global_index = SPLIT_GLOBAL_RANGES[split][0] + local_index
    sample_id = f"{BENCHMARK}/{split}/{local_index:06d}"
    stratum = "relaxed" if local_index % 2 == 0 else "binary_32_of_64"
    return shard_index, global_index, sample_id, stratum


def _control_record_length(split: str, local_index: int) -> int:
    shard_index, global_index, sample_id, stratum = _sample_identity(split, local_index)
    return _canonical_json_length({
        "benchmark": BENCHMARK,
        "control_tensor_sha256": _HASH_TEXT,
        "global_index": global_index,
        "local_index": local_index,
        "macro_coefficient_tensor_sha256": _HASH_TEXT,
        "record_self_sha256": _HASH_TEXT,
        "rho_storage_tensor_sha256": _HASH_TEXT,
        "sample_id": sample_id,
        "schema": _CONTROL_SCHEMA,
        "shard_index": shard_index,
        "split": split,
        "stratum": stratum,
    }) + 1


def _verification_record_upper_bound_length(split: str, local_index: int) -> int:
    shard_index, global_index, sample_id, stratum = _sample_identity(split, local_index)
    tensor_hashes = {name: _HASH_TEXT for name in TENSOR_FAMILY_ORDER}
    return _canonical_json_length({
        "benchmark": BENCHMARK,
        "control_tensor_sha256": _HASH_TEXT,
        "gates": {
            "P1_256": True,
            "TPFA64": True,
            "cross_solver": True,
            "overall": True,
            "structural": True,
        },
        "global_index": global_index,
        "local_index": local_index,
        "macro_coefficient_tensor_sha256": _HASH_TEXT,
        "metrics": {
            "compliances": {
                "T64": _MAX_FLOAT_HEX_TEXT,
                "bar_P256": _MAX_FLOAT_HEX_TEXT,
            },
            "delta_J": _MAX_FLOAT_HEX_TEXT,
            "delta_L2": _MAX_FLOAT_HEX_TEXT,
            "scales": {
                "S_J": _MAX_FLOAT_HEX_TEXT,
                "S_L2": _MAX_FLOAT_HEX_TEXT,
            },
        },
        "record_self_sha256": _HASH_TEXT,
        "rho_storage_tensor_sha256": _HASH_TEXT,
        "sample_id": sample_id,
        "schema": FINAL_VERIFICATION_SCHEMA,
        "shard_index": shard_index,
        "solver_diagnostics": {
            "P256": {
                "algebraic_backward_error": _MAX_FLOAT_HEX_TEXT,
                "boundary_reconstruction_error": _MAX_FLOAT_HEX_TEXT,
                "integrated_source": _MAX_FLOAT_HEX_TEXT,
                "iterations": 10_000,
                "outward_boundary_flux": _MAX_FLOAT_HEX_TEXT,
                "passes": True,
                "reaction_balance_ratio": _MAX_FLOAT_HEX_TEXT,
                "reaction_denominator": _MAX_FLOAT_HEX_TEXT,
                "rhs_relative_l2_residual": _MAX_FLOAT_HEX_TEXT,
            },
            "T64": {
                "algebraic_backward_error": _MAX_FLOAT_HEX_TEXT,
                "boundary_transmissibility_error": _MAX_FLOAT_HEX_TEXT,
                "global_balance_ratio": _MAX_FLOAT_HEX_TEXT,
                "maximum_interior_face_antisymmetry": _MAX_FLOAT_HEX_TEXT,
                "maximum_local_conservation_ratio": _MAX_FLOAT_HEX_TEXT,
                "passes": True,
                "rhs_relative_l2_residual": _MAX_FLOAT_HEX_TEXT,
            },
        },
        "split": split,
        "stratum": stratum,
        "tensor_sha256": tensor_hashes,
        "verifier_tensor_sha256": {"P256": _HASH_TEXT, "bar_P256": _HASH_TEXT},
    }) + 1


def _jsonl_sizes(record_length: Callable[[str, int], int]) -> dict[str, int]:
    return {
        split: sum(record_length(split, local) for local in range(SPLIT_SAMPLE_COUNTS[split]))
        for split in SPLITS
    }


def _npy_v2_header_length(shape: tuple[int, ...]) -> int:
    header = (
        "{'descr': '<f8', 'fortran_order': False, 'shape': "
        f"{shape!r}, }}"
    ).encode("latin1")
    unpadded = _NPY_V2_PREAMBLE_BYTES + len(header) + 1
    padding = (-unpadded) % _NPY_ALIGNMENT
    return unpadded + padding


def derive_core_writer_sizes() -> CoreWriterSizes:
    """Derive only the bounded tensor/header/primary-JSONL core."""

    total_shards = sum(SPLIT_SHARD_COUNTS.values())
    raw_payload = sum(
        math.prod(TENSOR_FAMILY_SPECS[name].shard_shape) * _RAW_ELEMENT_BYTES
        * total_shards
        for name in TENSOR_FAMILY_ORDER
    )
    header_by_family = {
        name: _npy_v2_header_length(TENSOR_FAMILY_SPECS[name].shard_shape)
        * total_shards
        for name in TENSOR_FAMILY_ORDER
    }
    controls_by_split = _jsonl_sizes(_control_record_length)
    verification_by_split = _jsonl_sizes(_verification_record_upper_bound_length)
    header_total = sum(header_by_family.values())
    controls_total = sum(controls_by_split.values())
    verification_total = sum(verification_by_split.values())
    primary_jsonl = controls_total + verification_total
    core_partial = raw_payload + header_total + primary_jsonl

    return CoreWriterSizes(
        raw_seven_tensor_payload=SizeContribution(
            raw_payload,
            "exact",
            "Frozen family shapes, 76 full shards, and eight bytes per <f8 element.",
        ),
        npy_headers=SizeContribution(
            header_total,
            "conditional_exact",
            "Independent NPY-v2 alignment model; requires comparison with canonical writer bytes.",
            ("actual canonical State 4 rho and six-family final writer/header comparison",),
        ),
        controls_jsonl=SizeContribution(
            controls_total,
            "conditional_exact",
            "Exact accepted control schedule and canonical encoding; requires writer comparison.",
            ("actual canonical controls JSONL writer comparison",),
        ),
        final_verification_jsonl=SizeContribution(
            verification_total,
            "upper_bound",
            "Every one of 19 float.hex fields is conservatively allocated 24 ASCII characters.",
            ("actual final-verification writer non-exceedance comparison",),
        ),
        primary_jsonl=SizeContribution(
            primary_jsonl,
            "upper_bound",
            "Final-verification accepted-domain upper bound plus controls bytes, conditional on "
            "the controls writer comparison.",
            (
                "actual canonical controls JSONL writer comparison",
                "actual final-verification writer non-exceedance comparison",
            ),
        ),
        core_partial=SizeContribution(
            core_partial,
            "upper_bound",
            "R + conditional H + primary J; excludes unresolved additive M. C is a subtotal "
            "of E, and B is separate from E in the free-space formula.",
            (
                "actual canonical State 4 rho and six-family final writer/header comparison",
                "actual canonical controls JSONL writer comparison",
                "actual final-verification writer non-exceedance comparison",
            ),
        ),
        projected_manifest_and_audit_bytes=SizeContribution(
            None,
            "unresolved",
            DISK_COMPONENT_MEMBERSHIP["projected_manifest_and_audit_bytes"],
            (
                "complete exact schemas, writers, validators, and source-bound evidence domains",
            ),
        ),
        controls_only_bytes=SizeContribution(
            None,
            "unresolved",
            DISK_COMPONENT_MEMBERSHIP["controls_only_bytes"],
            ("complete controls-stage retained-file and evidence accounting",),
        ),
        projected_audit_bytes=SizeContribution(
            None,
            "unresolved",
            DISK_COMPONENT_MEMBERSHIP["projected_audit_bytes"],
            (
                "explicit success/failure inventory and complete retained-run size derivation",
                "independent complete raw ZIP-structure verification",
            ),
        ),
        controls_jsonl_by_split=MappingProxyType(controls_by_split),
        final_verification_jsonl_by_split=MappingProxyType(verification_by_split),
        npy_header_bytes_by_family=MappingProxyType(header_by_family),
        unresolved_components=(
            "projected_manifest_and_audit_bytes",
            "controls_only_bytes",
            "projected_audit_bytes",
        ),
        production_disk_estimate_available=False,
    )


def require_complete_disk_components(
    *,
    projected_npy_header_bytes: int,
    projected_jsonl_bytes: int,
    projected_manifest_and_audit_bytes: int,
    controls_only_bytes: int,
    projected_audit_bytes: int,
) -> NoReturn:
    """Refuse promotion of caller assertions to a production disk estimate."""

    supplied = {
        "projected_npy_header_bytes": projected_npy_header_bytes,
        "projected_jsonl_bytes": projected_jsonl_bytes,
        "projected_manifest_and_audit_bytes": projected_manifest_and_audit_bytes,
        "controls_only_bytes": controls_only_bytes,
        "projected_audit_bytes": projected_audit_bytes,
    }
    for name, value in supplied.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"{name} must be a non-negative integer")
    unresolved = ", ".join(derive_core_writer_sizes().unresolved_components)
    raise IncompleteDiskComponentsError(
        "production disk estimate unavailable; caller-supplied integers do not resolve: "
        + unresolved
    )


@dataclass(frozen=True, slots=True)
class V2WriterSizes:
    inventory_sha256: str
    raw_tensor_bytes: SizeContribution
    npy_header_bytes: SizeContribution
    tensor_files: int
    by_split_family: tuple[tuple[str, str, tuple[int, ...], int, int], ...]
    production_disk_estimate_available: bool = False


def derive_v2_writer_sizes(layout) -> V2WriterSizes:
    """Arithmetic inventory; direction shards have 64/128 items, not always 64.

    Header predictions remain conditional until checked against Drop 2's
    serializer. JSONL/evidence and archive costs are deliberately not invented.
    """
    from .corpus_v2_schema import tensor_inventory, inventory_sha256
    rows = tuple((split, family.name, family.shard_shape(split), family.raw_bytes(split),
                  _npy_v2_header_length(family.shard_shape(split)) * SPLIT_SHARD_COUNTS[split])
                 for split in SPLITS for family in tensor_inventory(layout) if family.stored)
    return V2WriterSizes(inventory_sha256(layout),
        SizeContribution(sum(r[3] for r in rows), 'exact', 'Addendum A stored tensor inventory'),
        SizeContribution(sum(r[4] for r in rows), 'conditional_exact', 'NPY-v2 alignment model',
                         ('Drop 2 actual serializer/header comparison for every shape',)),
        sum(SPLIT_SHARD_COUNTS[r[0]] for r in rows), rows)


@dataclass(frozen=True, slots=True)
class V2DiskProjection:
    inventory_sha256: str
    raw_tensor_bytes: int
    projected_npy_header_bytes: int
    projected_jsonl_bytes: int | None
    projected_manifest_and_audit_bytes: int | None
    controls_only_bytes: int | None
    projected_audit_bytes: int | None
    estimated_accepted_bytes: int | None
    total_required_free_space_bytes: int | None
    unresolved_components: tuple[str, ...]
    production_disk_estimate_available: bool = False


def derive_v2_disk_projection(layout, *, jsonl_bytes=None, manifest_and_audit_bytes=None,
                              controls_only_bytes=None, audit_archive_bytes=None) -> V2DiskProjection:
    """E=R+H+J+M; C is a subset of E; free=max(5GiB,4E+B+1GiB).

    Supplied costs are planning inputs, never source-bound writer qualification.
    Any unknown cost remains None and prevents a complete/free-space result.
    """
    sizes = derive_v2_writer_sizes(layout)
    costs = dict(J=jsonl_bytes, M=manifest_and_audit_bytes, C=controls_only_bytes, B=audit_archive_bytes)
    for name, value in costs.items():
        if value is not None and (type(value) is not int or value < 0):
            raise TypeError(name + ' must be unknown or an exact nonnegative integer')
    raw, headers = sizes.raw_tensor_bytes.bytes, sizes.npy_header_bytes.bytes
    unresolved = tuple(name for name, value in costs.items() if value is None)
    accepted = None if jsonl_bytes is None or manifest_and_audit_bytes is None else raw + headers + jsonl_bytes + manifest_and_audit_bytes
    if accepted is not None and controls_only_bytes is not None and controls_only_bytes > accepted:
        raise ValueError('C must be a subtotal of E, not an additional population')
    required = None if unresolved else max(5 * 2**30, 4 * accepted + audit_archive_bytes + 2**30)
    return V2DiskProjection(sizes.inventory_sha256, raw, headers, jsonl_bytes, manifest_and_audit_bytes,
                           controls_only_bytes, audit_archive_bytes, accepted, required, unresolved)


def derive_v2_drop2_accounting(layout) -> dict:
    """Connect the independent inventory to actual Drop 2 writers, without arrays.

    J/M below are writer-domain bounds, not complete State 3-6 retention costs.
    Unknown custody, failure retention and B/C prevent a complete E/free estimate.
    No caller-supplied zero can qualify an unresolved cost.
    """
    from operatormark_corpus_v2 import sizing, layout as delivered
    from .corpus_v2_schema import tensor_inventory
    own = derive_v2_writer_sizes(layout)
    plan = sizing.file_plan(layout.n_ref, layout.retention)
    for split in SPLITS:
        expected = [(f.name, f.shard_shape(split)) for f in tensor_inventory(layout) if f.stored]
        actual = [(name, shape) for name, _, shape in
                  delivered.PRODUCTION_LAYOUT.shard_families(split, layout.n_ref, layout.retention)]
        if actual != expected:
            raise ValueError('Drop 2/repository tensor inventory mismatch: ' + split)
    if (sum(p['data_bytes'] for p in plan) != own.raw_tensor_bytes.bytes or
            sum(p['header_bytes'] for p in plan) != own.npy_header_bytes.bytes or len(plan) != own.tensor_files):
        raise ValueError('Drop 2/repository writer-byte mismatch')
    report = sizing.size_corpus(layout.n_ref, layout.retention)
    metadata = report['metadata_upper_bound']
    jsonl = sum(row['jsonl_upper_bound'] for row in metadata['per_split'].values())
    return dict(inventory_sha256=own.inventory_sha256, n_ref=layout.n_ref, retention=layout.retention,
        raw_tensor_bytes=own.raw_tensor_bytes.bytes, npy_header_bytes=own.npy_header_bytes.bytes,
        tensor_files=own.tensor_files, file_plan=plan,
        drop2_primary_jsonl_upper_bound=jsonl,
        drop2_manifest_configuration_summary_ladder_upper_bound=metadata['total'] - jsonl,
        per_unwritten_measurement_row_bound=metadata['per_unwritten_measurement_row'],
        writer_bounds=report, unknown=('State4 control/admission evidence (J/C)',
            'State3-6 audit, source/runtime and custody evidence (M)',
            'staging/promotion and failure retention', 'run records', 'retained audit archives (B)'),
        estimated_accepted_bytes=None, total_required_free_space_bytes=None,
        formula='E=R+H+J+M; C subset of E; free=max(5 GiB,4E+B+1 GiB)',
        production_disk_estimate_available=False)
