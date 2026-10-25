"""Independent conditional metadata comparison. Never imports either producer."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
import hashlib
import re
from types import MappingProxyType
import unicodedata

# Independent spelling of the verifier's Cc-plus-Cs policy.
_CONTROL_OR_SURROGATE = re.compile(r"[\x00-\x1f\x7f-\x9f\ud800-\udfff]")

from operatormark.corpus.canonical import canonical_json_bytes, strict_canonical_json_loads


class FinalEvidenceInputError(ValueError):
    pass


P = "operatormark.darcy_branch_c_v1."
PHASES = ("state4_closed", "state5_closed", "state6_child_closed", "pre_move", "post_move", "post_complete", "terminal")
COMPONENTS = ("state4_core", "state5_core", "final_records", "stage_execution", "publication",
              "permission_observation", "permission_authz", "permission_handoff", "authority_journals",
              "administration_restart", "state3_run", "retained_history")
GROUPS = ("documents", "controls", "execution", "scientific", "final_records", "publication",
          "retention", "sources", "authority", "unrecognized")
TOP_ROLES = (
    "protocol", "state3_authorization", "state4_authorization", "state5_authorization", "state6_authorization",
    "state3_audit", "compute_feasibility_report", "accepted_controls",
    "state4_execution", "state4_guards", "state4_process", "state5_execution", "state5_guards", "state5_process",
    "state5_scientific", "publication_plan", "state6_execution", "state6_guards", "state6_process", "state6_scientific",
    "final_context", "generated_data_rights", "corpus_access", "corpus_failures", "corpus_provenance",
    "corpus_manifest", "final_audit_details", "final_audit", "completion", "writer_catalog",
    "retention_snapshot", "source_closure", "root_binding_evidence", "historical_inventory")
PROFILES = ("document_full_bytes_v1", "existing_state3_archive_v1", "existing_state4_opaque_v1",
            "existing_state4_seed_free_v1", "existing_runtime_v2", "existing_permission_v1",
            "existing_publication_v1", "new_final_evidence_v1", "raw_captured_bytes_v1", "source_identity_v1")
SCOPES = ("state3_archive", "state4_acceptance", "runtime", "storage_state3_to_state4", "storage_later_phase",
          "permission_component", "scientific_members", "scientific_manifests", "publication_phase",
          "authority_and_custody", "source_closure")
EVENTS = ("authority_checked", "source_bound", "environment_checked", "storage_checked", "guards_installed",
          "project_import_checked", "worker_started", "worker_exited", "scientific_closed", "metadata_prepared",
          "access_observed", "failure_observed", "resources_released", "capture_closed")


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_plain(item) for item in value]
    if hasattr(type(value), "__dataclass_fields__"):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _post_init(value):
    for field in fields(value):
        object.__setattr__(value, field.name, _freeze(getattr(value, field.name)))


@dataclass(frozen=True)
class FinalEvidenceExpectations:
    profile: object
    phase: object
    protocol_binding: object
    implementation: object
    authorization_bindings: object
    root_requirements: object
    external_resolutions: object
    accepted_input_files: object
    upstream_admissions: object
    required_captures: object
    component_requirements: object
    expected_output_slots: object

    __post_init__ = _post_init


@dataclass(frozen=True)
class Comparison:
    report: object
    sealed_details: object

    __post_init__ = _post_init


@dataclass(frozen=True)
class ComparisonBytes:
    report_bytes: bytes
    sealed_details_bytes: bytes


def _assert(condition, message="malformed trusted input"):
    if not condition:
        raise FinalEvidenceInputError(message)


def _keys(value, names):
    _assert(type(value) is dict and set(value) == set(names.split()), "exact keys")


def _text(value, empty=False):
    _assert(type(value) is str and (bool(value) or empty), "text")
    _assert(unicodedata.normalize("NFC", value) == value and
            _CONTROL_OR_SURROGATE.search(value) is None, "NFC/control")


def _hash(value, n=64):
    _assert(type(value) is str and re.fullmatch("[0-9a-f]{" + str(n) + "}", value) is not None
            and value != "0" * n, "hash")


def _integer(value, minimum=0):
    _assert(type(value) is int and value >= minimum, "integer")


def _path(value, empty=False):
    _text(value, empty)
    if value == "":
        return
    for component in value.split("/"):
        _assert(component not in ("", ".", "..") and component[-1] not in " .", "path component")
        _assert(not any(char in component for char in '\\:*?"<>|'), "path alias")
        _assert(re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])", component.split(".")[0]) is None, "reserved path")


def _file(value, nullable=False):
    if nullable and value is None:
        return
    _keys(value, "root_binding_id relative_path file_bytes file_sha256")
    _text(value["root_binding_id"])
    _assert(":" not in value["root_binding_id"], "root ID")
    _path(value["relative_path"])
    _integer(value["file_bytes"])
    _hash(value["file_sha256"])


def _address(value):
    _keys(value, "root_binding_id relative_path")
    _text(value["root_binding_id"])
    _assert(":" not in value["root_binding_id"], "root ID")
    _path(value["relative_path"], True)


def _external(value):
    _keys(value, "filename literal_path_sha256 sha256 size")
    _path(value["filename"])
    _assert("/" not in value["filename"], "basename")
    _hash(value["literal_path_sha256"])
    _hash(value["sha256"])
    _integer(value["size"], 1)


def _document(value, kind):
    _keys(value, "kind file payload_rule payload_sha256")
    _assert(value["kind"] == kind and value["payload_rule"] == "opm_document_full_bytes_v1", "document rule")
    _external(value["file"])
    _hash(value["payload_sha256"])
    _assert(value["payload_sha256"] != value["file"]["sha256"], "ordinary hash substituted for payload")


def _list(value, validator, unique=False):
    _assert(type(value) is list, "array")
    for row in value:
        validator(row)
    if unique:
        encoded = [canonical_json_bytes(row) for row in value]
        _assert(encoded == sorted(set(encoded)), "array order/duplicates")


def _capture(value):
    _keys(value, "invocation_id clock_epoch_id first_sequence last_sequence started_utc finished_utc")
    _hash(value["invocation_id"])
    _text(value["clock_epoch_id"])
    _integer(value["first_sequence"])
    _integer(value["last_sequence"], value["first_sequence"])
    for name in ("started_utc", "finished_utc"):
        text = value[name]
        _assert(type(text) is str and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", text) is not None, "UTC")
        try:
            datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError as exc:
            raise FinalEvidenceInputError("UTC calendar") from exc
    _assert(value["started_utc"] <= value["finished_utc"], "capture order")


def _group(role):
    if type(role) is not str:
        return "unrecognized"
    if role == "protocol" or role.endswith("_authorization") and role in TOP_ROLES or role.startswith("document/"):
        return "documents"
    if role == "accepted_controls" or role.startswith("control/"):
        return "controls"
    if role == "state3_audit" or role in {s + "_" + t for s in ("state4", "state5", "state6") for t in ("execution", "guards", "process")}:
        return "execution"
    if role == "compute_feasibility_report" or role in ("state5_scientific", "state6_scientific") or role.startswith(("final/", "split/")):
        return "scientific"
    if role in ("final_context", "generated_data_rights", "corpus_access", "corpus_failures", "corpus_provenance",
                "corpus_manifest", "final_audit_details", "final_audit", "completion"):
        return "final_records"
    if role == "publication_plan" or role.startswith("publication/"):
        return "publication"
    if role in ("writer_catalog", "retention_snapshot", "historical_inventory") or role.startswith("history/"):
        return "retention"
    if role == "source_closure":
        return "sources"
    if role == "root_binding_evidence":
        return "authority"
    if role.startswith("component/"):
        parts = role.split("/", 2)
        component = parts[1] if len(parts) == 3 else None
        return {"state4_core": "controls", "state5_core": "scientific", "final_records": "final_records",
                "stage_execution": "execution", "publication": "publication", "permission_observation": "authority",
                "permission_authz": "authority", "permission_handoff": "authority", "authority_journals": "authority",
                "administration_restart": "authority", "state3_run": "retention", "retained_history": "retention"}.get(component, "unrecognized")
    return "unrecognized"


def _required(phase):
    roles = {"protocol", "state3_authorization", "state4_authorization", "state3_audit", "compute_feasibility_report",
             "accepted_controls", "state4_execution", "state4_guards", "state4_process", "writer_catalog",
             "retention_snapshot", "source_closure", "root_binding_evidence", "historical_inventory"}
    level = PHASES.index(phase)
    if level >= 1:
        roles.update(("state5_authorization", "state5_execution", "state5_guards", "state5_process", "state5_scientific"))
    if level >= 2:
        roles.update(("state6_authorization", "publication_plan", "state6_execution", "state6_guards",
                      "state6_process", "state6_scientific", "final_context"))
    if level >= 3:
        roles.update(("generated_data_rights", "corpus_access", "corpus_failures", "corpus_provenance", "corpus_manifest"))
    if level >= 4:
        roles.add("final_audit_details")
    if level >= 5:
        roles.update(("final_audit", "completion"))
    return roles


def _resolved(address, roots):
    _address(address)
    _assert(address["root_binding_id"] in roots, "unknown root")
    return roots[address["root_binding_id"]] + ("/" + address["relative_path"] if address["relative_path"] else "")


def _validate_slot(slot, roots):
    _keys(slot, "slot_id component address writer representation owner controls_subset condition creating_address content_profile domain")
    _assert(slot["component"] in COMPONENTS, "component")
    _address(slot["address"])
    _path(slot["address"]["relative_path"])
    _assert(slot["slot_id"] == slot["component"] + ":" + slot["address"]["root_binding_id"] + ":" +
            slot["address"]["relative_path"], "slot ID")
    _resolved(slot["address"], roots)
    _keys(slot["writer"], "source_file function source_closure_file write_policy")
    _file(slot["writer"]["source_file"])
    _file(slot["writer"]["source_closure_file"])
    _text(slot["writer"]["function"])
    _assert(slot["writer"]["write_policy"] in ("canonical_exclusive_preserve_v1", "component_existing_contract", "raw_supervisor_capture"), "writer policy")
    _assert(slot["representation"] in ("loose_file", "container", "reference_only") and
            slot["owner"] in ("accepted_attempt", "separate_B", "external_preexisting", "unresolved"), "ownership")
    _assert(type(slot["controls_subset"]) is bool and (not slot["controls_subset"] or slot["owner"] == "accepted_attempt"), "C ownership")
    _keys(slot["condition"], "kind source_file selector")
    _assert(slot["condition"]["kind"] in ("always", "stage_reached", "phase_reached", "operation_reached",
            "failure_observed", "historical_instance", "terminal_pending", "assessment_after_terminal"), "condition")
    _file(slot["condition"]["source_file"])
    _text(slot["condition"]["selector"])
    if slot["creating_address"] is not None:
        _address(slot["creating_address"])
        _resolved(slot["creating_address"], roots)
    if slot["writer"]["write_policy"] == "canonical_exclusive_preserve_v1" and slot["representation"] != "reference_only":
        _assert(slot["creating_address"] == dict(root_binding_id=slot["address"]["root_binding_id"],
                relative_path=slot["address"]["relative_path"] + ".__creating__"), "creating alias")
    _text(slot["content_profile"])
    _keys(slot["domain"], "classification rule input_files parameters prerequisite_files")
    _assert(slot["domain"]["classification"] in ("exact", "conditional_exact", "upper_bound", "unresolved"), "domain classification")
    _list(slot["domain"]["input_files"], _file, True)
    _list(slot["domain"]["prerequisite_files"], _file, True)
    rules = {"supplied_complete_bytes": "", "qualified_length_observation": "observation_file",
             "frozen_tensor_payload": "family shard_count", "canonical_npy_header": "family shard_count",
             "controls_jsonl_schedule": "split shard_index", "verification_jsonl_domain": "split shard_index",
             "state3_classic_zip": "ordered_member_files", "unresolved": "reason required_evidence_roles"}
    _assert(slot["domain"]["rule"] in rules, "closed domain rule")
    _keys(slot["domain"]["parameters"], rules[slot["domain"]["rule"]])


def _validate_final_evidence_expectations(expected: FinalEvidenceExpectations) -> None:
    _assert(type(expected) is FinalEvidenceExpectations, "frozen expectations required")
    e = _plain(expected)
    _assert(e["profile"] == "final_metadata_reconciliation_v1" and e["phase"] in PHASES, "expectation profile/phase")
    _document(e["protocol_binding"], "protocol")
    _keys(e["implementation"], "commit tree sole_parent")
    for item in e["implementation"].values():
        _hash(item, 40)
    _keys(e["authorization_bindings"], "state3 state4 state5 state6")
    required_states = ("state3", "state4") + (("state5",) if PHASES.index(e["phase"]) >= 1 else ()) + (
        ("state6",) if PHASES.index(e["phase"]) >= 2 else ())
    hashes = []
    for state, document in e["authorization_bindings"].items():
        if state in required_states:
            _document(document, state + "_authorization")
            hashes.append(document["file"]["sha256"])
        else:
            _assert(document is None, "future authorization")
    _assert(len(hashes) == len(set(hashes)), "distinct authorization files")
    roots = {}
    _assert(type(e["root_requirements"]) is list, "root requirements")
    for row in e["root_requirements"]:
        _keys(row, "root_binding_id repository_relative_root identity_file custody_file")
        _text(row["root_binding_id"])
        _assert(":" not in row["root_binding_id"] and row["root_binding_id"] not in roots, "root uniqueness")
        _path(row["repository_relative_root"])
        _file(row["identity_file"])
        _file(row["custody_file"])
        roots[row["root_binding_id"]] = row["repository_relative_root"]
    externals = []
    for row in e["external_resolutions"]:
        _keys(row, "external file locator_evidence_file")
        _external(row["external"])
        _file(row["file"])
        _file(row["locator_evidence_file"])
        _resolved({k: row["file"][k] for k in ("root_binding_id", "relative_path")}, roots)
        _assert(row["external"]["sha256"] == row["file"]["file_sha256"] and row["external"]["size"] == row["file"]["file_bytes"], "external resolution bytes")
        externals.append(canonical_json_bytes(row["external"]))
    _assert(len(externals) == len(set(externals)), "ambiguous external locator")
    for document in [e["protocol_binding"]] + [d for d in e["authorization_bindings"].values() if d is not None]:
        _assert(canonical_json_bytes(document["file"]) in externals, "document locator missing")
    role_set = set()
    for row in e["accepted_input_files"]:
        _keys(row, "role file schema validation_profile")
        _text(row["role"])
        _file(row["file"])
        _assert(row["role"] not in role_set and _group(row["role"]) != "unrecognized", "input role")
        role_set.add(row["role"])
        if row["schema"] is not None:
            _text(row["schema"])
        _assert(row["validation_profile"] in PROFILES, "validation profile")
        new_roles = {"accepted_controls": "accepted_controls_projection", "writer_catalog": "corpus_writer_catalog",
            "retention_snapshot": "corpus_retention_snapshot", "corpus_access": "corpus_access_ledger",
            "corpus_failures": "corpus_failure_inventory", "corpus_provenance": "corpus_provenance",
            "corpus_manifest": "corpus_manifest", "final_audit": "final_audit", "final_audit_details": "final_audit_details",
            "completion": "corpus_complete"}
        for state in ("state4", "state5", "state6"):
            new_roles[state + "_execution"] = "state_execution_evidence"
            new_roles[state + "_guards"] = "state_guard_closure"
            if state != "state6":
                new_roles[state + "_process"] = "state_process_outcome"
            if state != "state4":
                new_roles[state + "_scientific"] = "scientific_closure"
        if row["role"] in new_roles:
            _assert(row["schema"] == P + new_roles[row["role"]] + ".v1" and
                    row["validation_profile"] == "new_final_evidence_v1", "new role cannot bypass semantic validation")
        if row["role"] == "protocol" or row["role"].endswith("_authorization"):
            _assert(row["validation_profile"] == "document_full_bytes_v1" and row["schema"] is None, "document profile")
        if row["validation_profile"] == "existing_state4_opaque_v1":
            path = row["file"]["relative_path"]
            _assert(any(path == "state4_controls/" + split + "/split_manifest.json" or
                        path.endswith("/state4_controls/" + split + "/split_manifest.json")
                        for split in ("training", "validation", "test.sealed")), "opaque original split identity")
    _assert(_required(e["phase"]) <= role_set, "required phase input roles absent from expectations")
    _assert(not ((set(TOP_ROLES) - _required(e["phase"])) & role_set), "future top-level evidence")
    admission_roles = set()
    for row in e["upstream_admissions"]:
        _keys(row, "role input_files validator_source_files admission_file scope capture status")
        _text(row["role"])
        _assert(row["role"] not in admission_roles, "duplicate admission")
        admission_roles.add(row["role"])
        _list(row["input_files"], _file, True)
        _list(row["validator_source_files"], _file, True)
        _file(row["admission_file"])
        _capture(row["capture"])
        _assert(row["scope"] in SCOPES and row["status"] in ("accepted", "unavailable", "rejected"), "admission token")
        _assert(bool(row["input_files"]) and bool(row["validator_source_files"]), "admission bindings")
    captures = set()
    for row in e["required_captures"]:
        _keys(row, "role invocation_id clock_epoch_id required_first_event required_last_event predecessor_roles")
        _assert(row["role"] in role_set and row["role"] not in captures, "capture role")
        captures.add(row["role"])
        _hash(row["invocation_id"])
        _text(row["clock_epoch_id"])
        _assert(row["required_first_event"] in EVENTS and row["required_last_event"] in EVENTS, "capture events")
        _list(row["predecessor_roles"], _text)
        _assert(row["predecessor_roles"] == sorted(set(row["predecessor_roles"])) and set(row["predecessor_roles"]) <= role_set, "capture predecessor roles")
    _assert([r["component"] for r in e["component_requirements"]] == list(COMPONENTS), "twelve independent components")
    for row in e["component_requirements"]:
        _keys(row, "component required_source_files required_plan_files expected_slot_ids admission_roles")
        _list(row["required_source_files"], _file, True)
        _list(row["required_plan_files"], _file, True)
        if row["expected_slot_ids"] is not None:
            _list(row["expected_slot_ids"], _text)
            _assert(row["expected_slot_ids"] == sorted(set(row["expected_slot_ids"])), "component slots")
            _assert(bool(row["required_plan_files"]) and bool(row["admission_roles"]), "affirmative component population needs plan/admission")
        _list(row["admission_roles"], _text)
        _assert(row["admission_roles"] == sorted(set(row["admission_roles"])) and set(row["admission_roles"]) <= admission_roles, "component admission roles")
    slots = e["expected_output_slots"]
    _assert(type(slots) is list, "output slots")
    for slot in slots:
        _validate_slot(slot, roots)
    locations = [(_resolved(slot["address"], roots), slot["slot_id"]) for slot in slots]
    _assert(locations == sorted(locations) and len({p.casefold() for p, _ in locations}) == len(locations), "output slot order/alias")
    _output_pair(e, roots)


def validate_final_evidence_expectations(expected: FinalEvidenceExpectations) -> None:
    try:
        _validate_final_evidence_expectations(expected)
    except FinalEvidenceInputError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, UnicodeError) as exc:
        raise FinalEvidenceInputError("malformed trusted expectations") from exc


def _output_pair(e, roots):
    phase = e["phase"]
    attempt = "s4-" + e["authorization_bindings"]["state4"]["file"]["sha256"][:24]
    base = "artifacts/darcy_branch_c_v1/private_corpus_v1/"
    if phase in ("state4_closed", "state5_closed"):
        root = base + "state_evidence/" + attempt + "/" + phase[:6] + "/"
        report, details = root + "reviews/" + phase + ".json", root + "test.sealed/reviews/" + phase + ".json"
    else:
        root = base + "publication_evidence/" + attempt + "/"
        report, details = root + "final_evidence_reviews/" + phase + ".json", root + "test.sealed/final_evidence_reviews/" + phase + ".json"
    selected = []
    for path, schema in ((report, "final_evidence_comparison"), (details, "final_evidence_details")):
        matches = [s for s in e["expected_output_slots"] if _resolved(s["address"], roots) == path and
                   s["content_profile"] == P + schema + ".v1"]
        _assert(len(matches) == 1, "independently designated report/details pair missing")
        selected.append(matches[0])
    return selected


# Independently owned record layouts; no producer schema import or callback.
RECORD_FIELDS = {
    "accepted_controls_projection": "binding capture state3_audit compute_feasibility_report pre_solve_manifest controls_completion rng_audit read_only_evidence ordered_split_manifests ordered_control_shards original_file_inventory acceptance_evidence_files",
    "state_guard_closure": "binding capture process_creation_evidence_file frozen_base_import_closure_file rng_state_commitments rng_state_equalities authorized_counts forbidden_counts guard_lifetime_evidence_files coverage",
    "process_lifecycle_event": "binding event capture pid child_source_closure_file supervisor_source_file authority_scope_file creation_record_file direct_exit_code",
    "state_process_outcome": "binding role capture source_closure_file authority_scope_file creation_status process_identity direct_exit_code stdout stderr guards_file reap_status failures",
    "state_execution_evidence": "binding capture predecessors source_and_dependency_bindings environment_before environment_after storage_boundaries git_before git_after process_outcome guard_closure scientific_closure ordered_events failure_files outcome",
    "writer_exception_graph": "binding capture operation_id primary_exception_id exceptions ordered_cleanup_failures capture_failures",
    "scientific_closure": "binding capture phase accepted_controls_projection source_closure_file context_file ordered_shards ordered_splits aggregate failure_files",
    "corpus_failure_inventory": "benchmark attempt_id prefix historical_inventory_files current_failure_events retention_inventory_file unavailable_evidence",
    "corpus_access_ledger": "benchmark attempt_id prefix split_access_state ordered_access_events historical_unknown_access_files unavailable_evidence",
    "corpus_provenance": "benchmark parent_context protocol_binding authorization_bindings baseline_binding implementation_binding source_and_dependency_bindings environment_and_platform_bindings gate_bindings execution_evidence failure_inventory access_ledger rights_record legacy_data_or_model_output_used",
    "corpus_manifest": "benchmark corpus_version attempt_id parent_context protocol_file_sha256 authorization_file_sha256_by_state solver_spec_baseline_commit solver_spec_baseline_tree implementation_commit implementation_tree environment_and_dependency_bindings gate_bindings ordered_split_manifests rights_record provenance_record failure_inventory controls_pre_solve_manifest controls_completion",
    "final_audit_details": "benchmark attempt_id corpus_manifest prefix scientific_closure state_execution_files access_extension failure_extension post_move_publication_files retention_inventory_file checks unavailable_evidence",
    "final_audit": "benchmark corpus_version attempt_id corpus_manifest parent_context details_file prefix split_summary check_status publication_location",
    "corpus_complete": "state corpus_manifest_filename corpus_manifest_file_sha256 completion_time_utc",
    "corpus_writer_catalog": "profile protocol_file_sha256 implementation attempt_binding root_requirements component_registrations slots unresolved_requirements",
    "corpus_retention_snapshot": "catalog_file capture cutoff prior_snapshot_file snapshot_address observations unavailable_requirements",
}
WIRE_KEYS = {
    "binding": "benchmark state attempt_id protocol_file_sha256 implementation state4_authorization_file_sha256 authorization_file_sha256",
    "implementation": "commit tree sole_parent",
    "capture": "invocation_id clock_epoch_id first_sequence last_sequence started_utc finished_utc",
    "prefix": "capture predecessor_files last_observed_event_file next_boundary",
    "coverage": "started_before_project_import continued_through_exit all_required_hooks_accounted unavailable_hook_evidence_files",
    "rng_state_commitments": "after_base_import after_guard_install after_project_import after_execution",
    "rng_state_equalities": "after_guard_install after_project_import after_execution",
    "authorized_counts": "seed_access_by_split pcg64_constructors_by_split generator_constructors_by_split top_level_rng_calls_by_split",
    "forbidden_counts": "rng_calls corpus_seed_access legacy_data_access model_execution network_access unauthorized_test_access",
    "seed_access_by_split": "training validation test",
    "pcg64_constructors_by_split": "training validation test",
    "generator_constructors_by_split": "training validation test",
    "top_level_rng_calls_by_split": "training validation test",
    "process_identity": "pid creation_evidence_file",
    "source_and_dependency_bindings": "source_manifest_file implementation_files baseline_files dependency_lock base_import_closure_file qualification_files",
    "source_and_dependency": "source_manifest_file implementation_files baseline_files dependency_lock base_import_closure_file qualification_files",
    "environment_before": "file frozen_state3_environment_file parent_environment_commitment base_import_closure_sha256 numerical_runtime_sha256 source_profile",
    "environment_after": "file frozen_state3_environment_file parent_environment_commitment base_import_closure_sha256 numerical_runtime_sha256 source_profile",
    "environment_and_platform_bindings": "state3 state4 state5 state6",
    "environment_and_platform": "state3 state4 state5 state6",
    "environment_and_dependency_bindings": "environment_and_platform source_and_dependency",
    "baseline_binding": "commit tree gate_files",
    "implementation_binding": "identity source_manifest_file git_before git_after",
    "gate_bindings": "gate4 gate6 gate7 gate8 gate9",
    "gate_files": "gate4 gate6 gate7 gate8 gate9",
    "protocol_binding": "kind file payload_rule payload_sha256",
    "authorization_bindings": "state3 state4 state5 state6",
    "authorization_file_sha256_by_state": "state4 state5 state6",
    "execution_evidence": "state4 state5 state6",
    "split_access_state": "training validation test",
    "sample_counts": "structural TPFA64 P1_256 cross_solver overall",
    "shard_tensor_sha256": "rho kappa64 u_tpfa64 kappa32_areaavg u32_areaavg kappa16_areaavg u16_areaavg",
    "split_tensor_sha256": "rho kappa64 u_tpfa64 kappa32_areaavg u32_areaavg kappa16_areaavg u16_areaavg",
    "maxima": "T64_algebraic_backward_error T64_boundary_transmissibility_error T64_maximum_local_conservation_ratio T64_maximum_interior_face_antisymmetry T64_global_balance_ratio P256_algebraic_backward_error P256_boundary_reconstruction_error P256_reaction_balance_ratio delta_L2 delta_J",
    "aggregate": "sample_counts shard_count split_count maxima",
    "checks": "source authority state4_provenance rng_and_access scientific manifest_closure physical_publication retention",
    "check_status": "source authority state4_provenance rng_and_access scientific manifest_closure physical_publication retention",
    "git_before": "branch head tree parents status_short_untracked_all tracked_diff_sha256 tracked_diff_size_bytes staged_diff_sha256 staged_diff_size_bytes tracked_changes_empty staged_changes_empty untracked_inclusive_status_empty",
    "git_after": "branch head tree parents status_short_untracked_all tracked_diff_sha256 tracked_diff_size_bytes staged_diff_sha256 staged_diff_size_bytes tracked_changes_empty staged_changes_empty untracked_inclusive_status_empty",
}
ROW_KEYS = {
    "ordered_control_shards": "split shard_index controls_manifest controls_jsonl rho rho_tensor_sha256 ordered_control_tensor_sha256 ordered_rho_storage_tensor_sha256 ordered_macro_coefficient_tensor_sha256 ordered_control_record_self_sha256",
    "original_file_inventory": "file native_identity origin_evidence_file custody_file permission_evidence_files",
    "ordered_events": "sequence kind source_file subject_file prior_event_file",
    "storage_boundaries": "boundary file comparison_file comparison_profile accepted_scope_file",
    "current_failure_events": "event_file capture operation_id call_status effect_status primary follow_on surviving_files location location_evidence_file preservation_status",
    "failure_extension": "event_file capture operation_id call_status effect_status primary follow_on surviving_files location location_evidence_file preservation_status",
    "ordered_access_events": "capture actor_evidence_file authority_scope_file operation subject decision decision_evidence_file call_status effect_status count_evidence failure_files",
    "access_extension": "capture actor_evidence_file authority_scope_file operation subject decision decision_evidence_file call_status effect_status count_evidence failure_files",
    "exceptions": "exception_id class_module class_qualname message_bytes traceback_bytes cause_id context_id suppress_context",
    "ordered_cleanup_failures": "ordinal stage exception_id",
    "ordered_shards": "split shard_index member_files controls_manifest_file controls_jsonl_file original_rho_origin_file shard_tensor_sha256 verification_record_commitment shard_manifest_file sample_counts maxima evidence_file",
    "ordered_splits": "split sample_count shard_count split_manifest_file split_tensor_sha256 sample_counts maxima sealed",
    "split_summary": "split sample_count shard_count sample_counts maxima split_file_commitment sealed",
    "post_move_publication_files": "role file schema validation_profile",
    "unavailable_evidence": "role expected_file reason",
    "unavailable_requirements": "role expected_file reason",
    "unresolved_requirements": "role expected_file reason",
    "implementation_files": "relative_path file_bytes file_sha256 git_blob source_role",
    "baseline_files": "relative_path file_bytes file_sha256 git_blob source_role",
}
FAILURE_KEYS = "kind native_error_hex exception_file affected_addresses"
COUNT_KEYS = "measurement value evidence_files"
COUNTERS = ("structural", "TPFA64", "P1_256", "cross_solver", "overall")
SHARD_ORDER = tuple((split, i) for split, count in (("training", 64), ("validation", 4), ("test", 8)) for i in range(count))


def _wire(value, name="", parent=""):
    if value is None:
        return
    if type(value) is dict:
        if name in WIRE_KEYS:
            _keys(value, WIRE_KEYS[name])
        if name == "capture":
            if value["last_sequence"] is None or value["finished_utc"] is None:
                _assert(value["last_sequence"] is None and value["finished_utc"] is None, "incomplete capture pair")
                _hash(value["invocation_id"])
                _text(value["clock_epoch_id"])
                _integer(value["first_sequence"])
                _assert(type(value["started_utc"]) is str and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", value["started_utc"]) is not None, "capture UTC")
                datetime.strptime(value["started_utc"], "%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                _capture(value)
        if parent in ROW_KEYS:
            _keys(value, ROW_KEYS[parent])
        if name in ("primary",) or parent in ("failures", "follow_on", "capture_failures"):
            _keys(value, FAILURE_KEYS)
        if name == "count_evidence" or parent in ("seed_access_by_split", "pcg64_constructors_by_split",
                "generator_constructors_by_split", "top_level_rng_calls_by_split", "forbidden_counts"):
            _keys(value, COUNT_KEYS)
            _assert(value["measurement"] in ("measured", "not_measured", "unavailable"), "count measurement")
            if value["measurement"] == "measured":
                _integer(value["value"])
                _assert(bool(value["evidence_files"]), "measured count proof")
            else:
                _assert(value["value"] is None, "unknown measurement is not zero")
        if parent == "rng_state_commitments":
            _keys(value, "numpy_legacy_sha256 python_random_sha256")
        if parent == "rng_state_equalities":
            _keys(value, "numpy_legacy_equals_baseline python_random_equals_baseline")
        if parent == "checks":
            _keys(value, "status evidence_files")
        if name in ("stdout", "stderr", "message_bytes", "traceback_bytes"):
            _keys(value, "state file")
            _assert(value["state"] in ("complete", "partial", "unavailable", "not_started"), "captured bytes state")
            if value["state"] in ("complete", "partial"):
                _file(value["file"])
            else:
                _assert(value["file"] is None, "unavailable bytes cannot have identity")
        if "root_binding_id" in value and "relative_path" in value:
            if "file_bytes" in value or "file_sha256" in value:
                _file(value)
            else:
                _address(value)
        if "filename" in value and "literal_path_sha256" in value:
            _external(value)
        if "relative_path" in value and "size" in value:
            _keys(value, "relative_path size sha256" + (" schema" if "schema" in value else ""))
            _path(value["relative_path"])
            _integer(value["size"], 1)
            _hash(value["sha256"])
        if name == "native_identity":
            _keys(value, "volume_serial_number_hex file_id_128_hex legacy_volume_serial_number_hex legacy_file_index_hex")
            encoded = value["volume_serial_number_hex"]
            _assert(type(encoded) is str and re.fullmatch("0x(0|[1-9a-f][0-9a-f]*)", encoded) is not None and int(encoded, 16) < 2**64, "native volume")
            for field, width in (("file_id_128_hex", 32), ("legacy_volume_serial_number_hex", 8), ("legacy_file_index_hex", 16)):
                _assert(type(value[field]) is str and re.fullmatch("[0-9a-f]{" + str(width) + "}", value[field]) is not None, "native identity")
        for key, item in value.items():
            _text(key)
            _wire(item, key, name)
    elif type(value) is list:
        for item in value:
            _wire(item, "", name)
    elif type(value) is str:
        if name == "status_short_untracked_all":
            return
        _text(value, True)
        if name.endswith("_sha256") or name in ("sha256", "invocation_id"):
            _hash(value)
    else:
        _assert(type(value) in (bool, int), "wire scalar")
        if name in ("suppress_context", "sealed", "controls_subset", "started_before_project_import",
                    "continued_through_exit", "all_required_hooks_accounted", "legacy_data_or_model_output_used",
                    "production_disk_estimate_available"):
            _assert(type(value) is bool, "boolean field")
        if name in ("file_bytes", "size", "pid", "sequence", "first_sequence", "last_sequence",
                    "shard_index", "sample_count", "shard_count", "split_count", "ordinal") or parent == "sample_counts":
            _integer(value, 1 if name == "pid" else 0)


def _record(data, e):
    record = strict_canonical_json_loads(data)
    _assert(type(record) is dict, "record object required")
    schema = record.get("schema")
    kinds = [kind for kind in RECORD_FIELDS if schema == P + kind + ".v1"]
    _assert(len(kinds) == 1, "unsupported new schema")
    kind = kinds[0]
    self_field = "manifest_self_sha256" if kind == "corpus_manifest" else "record_self_sha256"
    _keys(record, "schema " + RECORD_FIELDS[kind] + " " + self_field)
    _hash(record[self_field])
    payload = {k: v for k, v in record.items() if k != self_field}
    _assert(hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == record[self_field], "record self hash")
    _wire(record)
    if "benchmark" in record:
        _assert(record["benchmark"] == "darcy_branch_c_v1", "benchmark")
    if "corpus_version" in record:
        _assert(record["corpus_version"] == "private_corpus_v1", "corpus version")
    if "binding" in record:
        binding = record["binding"]
        state = binding["state"]
        _assert(state in ("state4", "state5", "state6"), "stage")
        _assert(binding["benchmark"] == "darcy_branch_c_v1" and binding["implementation"] == e["implementation"], "source identity")
        _assert(binding["protocol_file_sha256"] == e["protocol_binding"]["file"]["sha256"], "protocol identity")
        current = e["authorization_bindings"][state]
        _assert(current is not None and binding["authorization_file_sha256"] == current["file"]["sha256"], "current authority")
        a4 = e["authorization_bindings"]["state4"]["file"]["sha256"]
        _assert(binding["state4_authorization_file_sha256"] == a4 and binding["attempt_id"] == "s4-" + a4[:24], "full A4 binding")
        if "capture" in record:
            capture = record["capture"]
            invocation = hashlib.sha256(("operatormark-state-evidence-invocation-v1\n" + state + "\n" +
                         current["file"]["sha256"] + "\n" + a4).encode()).hexdigest()
            _assert(capture["invocation_id"] == invocation, "invocation mismatch")
    if kind == "state_guard_closure":
        baseline = record["rng_state_commitments"]["after_base_import"]
        for checkpoint, row in record["rng_state_equalities"].items():
            for family in ("numpy_legacy", "python_random"):
                left, right = baseline[family + "_sha256"], record["rng_state_commitments"][checkpoint][family + "_sha256"]
                _assert(row[family + "_equals_baseline"] is (None if left is None or right is None else left == right), "RNG equality lie")
    if kind == "process_lifecycle_event":
        _capture(record["capture"])
        _integer(record["pid"], 1)
        for name in ("child_source_closure_file", "supervisor_source_file", "authority_scope_file"):
            _file(record[name])
        _assert(record["event"] in ("created", "exited"), "lifecycle event")
        if record["event"] == "created":
            _assert(record["creation_record_file"] is None and record["direct_exit_code"] is None, "created cannot predict exit")
        else:
            _file(record["creation_record_file"])
            _assert(type(record["direct_exit_code"]) is int, "direct exit representation")
    if kind == "state_process_outcome":
        _assert(record["binding"]["state"] in ("state4", "state5") and record["role"] ==
            ("state4_controls_child" if record["binding"]["state"] == "state4" else "state5_solve_child"), "process stage/role")
        _assert(record["creation_status"] in ("not_started", "started", "unknown") and
            record["reap_status"] in ("not_applicable", "observed_reaped", "unresolved", "failed"), "process statuses")
        _file(record["source_closure_file"])
        _file(record["authority_scope_file"])
        if record["creation_status"] == "started":
            _keys(record["process_identity"], "pid creation_evidence_file")
            _integer(record["process_identity"]["pid"], 1)
            _file(record["process_identity"]["creation_evidence_file"])
            _assert(record["reap_status"] != "not_applicable", "started reap")
        elif record["creation_status"] == "not_started":
            _assert(record["process_identity"] is None and record["direct_exit_code"] is None, "unestablished process identity")
        else:
            _assert(record["direct_exit_code"] is None and record["reap_status"] in ("unresolved", "failed") and bool(record["failures"]), "unknown process remains unresolved")
        if record["direct_exit_code"] is not None:
            _assert(type(record["direct_exit_code"]) is int, "signed direct exit")
        if record["reap_status"] == "observed_reaped":
            _assert(record["direct_exit_code"] is not None, "lost exit is not reaped success")
        if record["creation_status"] == "not_started":
            _assert(record["stdout"] == record["stderr"] == dict(state="not_started", file=None) and
                record["reap_status"] == "not_applicable" and bool(record["failures"]) and record["guards_file"] is None, "positive refusal evidence")
        if record["guards_file"] is None:
            _assert(record["creation_status"] == "not_started" or bool(record["failures"]), "lost guards need failure evidence")
    if kind == "state_execution_evidence":
        _assert(record["outcome"] in ("completed", "refused_before_start", "failed_preserved", "unresolved"), "execution outcome")
        events = record["ordered_events"]
        _assert(type(events) is list, "execution events")
        sequences = [r["sequence"] for r in events]
        _assert(sequences == sorted(set(sequences)) and all(type(n) is int and n >= record["capture"]["first_sequence"] and
            (record["capture"]["last_sequence"] is None or n <= record["capture"]["last_sequence"]) for n in sequences), "event cutoff/order")
        _assert(all(r["kind"] in EVENTS for r in events), "event kind")
        if record["outcome"] == "completed":
            _capture(record["capture"])
            _file(record["process_outcome"])
            _file(record["guard_closure"])
            _assert(not record["failure_files"] and record["git_before"] == record["git_after"] and
                all(record["git_before"][k] is True for k in ("tracked_changes_empty", "staged_changes_empty", "untracked_inclusive_status_empty")), "completed execution dirty/failure")
            _assert(record["environment_before"] is not None and record["environment_after"] is not None and record["source_and_dependency_bindings"] is not None, "completed execution prerequisites")
        if record["binding"]["state"] == "state4":
            _assert(record["scientific_closure"] is None, "State4 scientific output forbidden")
        elif record["outcome"] == "completed":
            _file(record["scientific_closure"])
    if kind in ("accepted_controls_projection", "scientific_closure"):
        rows = record["ordered_control_shards" if kind == "accepted_controls_projection" else "ordered_shards"]
        _assert([(row["split"], row["shard_index"]) for row in rows] == list(SHARD_ORDER), "ordered76 shards")
        _assert(all(type(row["shard_index"]) is int for row in rows), "shard integer")
        if kind == "accepted_controls_projection":
            _assert(len(record["ordered_split_manifests"]) == 3 and len(record["original_file_inventory"]) == 235, "controls population")
            for field in ("ordered_control_tensor_sha256", "ordered_rho_storage_tensor_sha256",
                          "ordered_macro_coefficient_tensor_sha256", "ordered_control_record_self_sha256"):
                _assert(all(len(row[field]) == 64 for row in rows), "64 sample commitments")
                values = [value for row in rows for value in row[field]]
                _assert(len(values) == len(set(values)), "duplicate sample commitment")
                for value in values:
                    _hash(value)
        else:
            stage = record["binding"]["state"]
            _assert((stage, record["phase"]) in (("state5", "state5_members"), ("state6", "state6_manifests")), "scientific phase")
            _assert(stage != "state5" or record["context_file"] is None and record["ordered_splits"] == [] and
                    all(row["shard_manifest_file"] is None for row in rows), "future State5 metadata")
            for row in rows:
                _assert(len(row["member_files"]) == 8 and row["verification_record_commitment"] == row["member_files"][-1]["file_sha256"], "member closure")
                _assert(all(0 <= row["sample_counts"][key] <= 64 for key in COUNTERS), "shard counts")
            aggregate = record["aggregate"]
            _assert(aggregate["shard_count"] == 76 and aggregate["split_count"] == 3, "aggregate population")
            for key in COUNTERS:
                _assert(aggregate["sample_counts"][key] == sum(row["sample_counts"][key] for row in rows), "aggregate count mismatch")
            for key in WIRE_KEYS["maxima"].split():
                values = [float.fromhex(row["maxima"][key]) for row in rows]
                _assert(all(number >= 0 and number.hex() == row["maxima"][key] and number != float("inf")
                            for number, row in zip(values, rows)), "maximum representation")
                _assert(aggregate["maxima"][key] == max(values).hex(), "maximum aggregation")
    if kind == "writer_exception_graph":
        ids = [row["exception_id"] for row in record["exceptions"]]
        _assert(ids == ["e" + str(i) for i in range(len(ids))] and bool(ids) and record["primary_exception_id"] == "e0", "graph IDs")
        nodes = dict(zip(ids, record["exceptions"]))
        visited = []
        pending = ["e0"]
        cleanup_ids = []
        for index, row in enumerate(record["ordered_cleanup_failures"]):
            _assert(type(row["ordinal"]) is int and row["ordinal"] == index and row["stage"] == "close" and row["exception_id"] != "e0", "cleanup order/identity")
            cleanup_ids.append(row["exception_id"])
        pending.extend(reversed(cleanup_ids))
        # Primary chain precedes cleanup roots; use explicit stack, preserving cycles.
        visited = []
        for root in ["e0"] + cleanup_ids:
            stack = [root]
            while stack:
                name = stack.pop()
                if name is None or name in visited:
                    continue
                _assert(name in nodes, "dangling graph link")
                visited.append(name)
                stack.extend((nodes[name]["context_id"], nodes[name]["cause_id"]))
        _assert(visited == ids, "graph reachability/first encounter")
    if kind == "corpus_complete":
        _assert(record["state"] == "accepted_private_corpus_no_training_authorized" and record["corpus_manifest_filename"] == "corpus_manifest.json", "completion representation")
    return record


def _seal(payload):
    result = dict(payload)
    result["record_self_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _runtime_metadata(value):
    """Independent source-bound runtime/base grammar, without backend execution."""
    schema = value.get("schema")
    if schema == P + "state3_environment_runtime.v1":
        _keys(value, "schema state protocol_file_sha256 state3_authorization_file_sha256 implementation_commit implementation_tree python_version numpy_version scipy_version operating_system architecture locale parent_environment parent_environment_commitment applicable_additional_thread_variables launcher_file_sha256 runtime_inspector_file_sha256 requirements_lock_path requirements_lock_git_blob requirements_lock_file_sha256 base_import_closure base_import_closure_sha256 numerical_runtime numerical_runtime_sha256 all_environment_checks_passed all_runtime_checks_passed record_self_sha256")
        _assert(value["state"] == "state3_pre_generation_audit_no_corpus_controls" and
            (value["python_version"], value["numpy_version"], value["scipy_version"]) == ("3.13.5", "2.5.2", "1.18.1") and
            value["requirements_lock_path"] == "requirements-lock-cpu.txt", "environment identity")
        for name in ("protocol_file_sha256", "state3_authorization_file_sha256", "launcher_file_sha256", "runtime_inspector_file_sha256",
                     "requirements_lock_file_sha256", "base_import_closure_sha256", "numerical_runtime_sha256", "parent_environment_commitment", "record_self_sha256"):
            _hash(value[name])
        for name in ("implementation_commit", "implementation_tree", "requirements_lock_git_blob"):
            _hash(value[name], 40)
        for name in ("operating_system", "architecture", "locale"):
            _text(value[name])
        mandatory = {"PYTHONDONTWRITEBYTECODE":"1", "PYTHONHASHSEED":"0", "OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1",
            "MKL_NUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1", "VECLIB_MAXIMUM_THREADS":"1", "BLIS_NUM_THREADS":"1", "GOTO_NUM_THREADS":"1",
            "OMP_THREAD_LIMIT":"1", "OMP_DYNAMIC":"FALSE", "MKL_DYNAMIC":"FALSE"}
        environment = value["parent_environment"]
        _assert(type(environment) is dict and all(environment.get(k) == v for k,v in mandatory.items()), "frozen parent policy")
        for name, item in environment.items():
            _text(name)
            _text(item)
        additional = sorted(set(environment) - set(mandatory))
        _assert(value["applicable_additional_thread_variables"] == additional and
            value["parent_environment_commitment"] == hashlib.sha256(b"operatormark-parent-environment-v1\n" + canonical_json_bytes(environment)).hexdigest(), "parent commitment")
        _runtime_metadata(value["base_import_closure"])
        _runtime_metadata(value["numerical_runtime"])
        runtime = value["numerical_runtime"]
        _assert(value["base_import_closure_sha256"] == value["base_import_closure"]["closure_sha256"] and
            value["numerical_runtime_sha256"] == runtime["runtime_record_sha256"] and
            runtime["applicable_additional_thread_variables"] == additional and
            runtime["bound_applicable_thread_policy"] == {n:environment[n] for n in additional} and
            value["all_environment_checks_passed"] is True and value["all_runtime_checks_passed"] is True and
            runtime["all_runtime_requirements_satisfied"] is True, "runtime environment cross-binding")
        _assert(value["record_self_sha256"] == hashlib.sha256(canonical_json_bytes({k:v for k,v in value.items() if k != "record_self_sha256"})).hexdigest(), "environment self hash")
        return
    if schema == P + "base_import_closure.v1":
        _keys(value, "closure_sha256 numpy_version ordered_modules root_modules schema scipy_version")
        roots = ["numpy", "numpy.typing", "scipy", "scipy.sparse", "scipy.sparse.linalg"]
        _assert(value["numpy_version"] == "2.5.2" and value["scipy_version"] == "1.18.1" and value["root_modules"] == roots, "base identity")
        _assert(type(value["ordered_modules"]) is list and bool(value["ordered_modules"]), "base modules")
        names = []
        for row in value["ordered_modules"]:
            _keys(row, "file_sha256 kind module suffix")
            _text(row["module"])
            _assert(row["module"] in ("numpy", "scipy") or row["module"].startswith(("numpy.", "scipy.")), "base namespace")
            names.append(row["module"])
            suffix = row["suffix"]
            _assert(type(suffix) is str and suffix == suffix.lower(), "base suffix")
            if row["kind"] == "namespace_package_no_module_bytes":
                _assert(suffix == "" and row["file_sha256"] is None, "namespace bytes")
            else:
                endings = {"source_module": (".py", ".pyi"), "extension_binary": (".pyd", ".so", ".dll", ".dylib")}
                _assert(row["kind"] in endings and suffix.endswith(endings[row["kind"]]), "module kind")
                _hash(row["file_sha256"])
        _assert(names == sorted(set(names)) and set(roots) <= set(names), "base order/membership")
        payload = {k:v for k,v in value.items() if k != "closure_sha256"}
        _assert(value["closure_sha256"] == hashlib.sha256(canonical_json_bytes(payload)).hexdigest(), "base commitment")
        return
    _assert(schema == P + "numerical_runtime.v2", "runtime observation schema")
    _keys(value, "active_backend_kinds active_backends all_active_backends_inspectable all_effective_thread_counts_one all_parent_thread_policy_bound all_runtime_requirements_satisfied applicable_additional_thread_variables bound_applicable_thread_policy missing_applicable_thread_policy_variables non_single_thread_backend_binary_sha256 numerical_wrappers runtime_record_sha256 schema unexpected_additional_thread_policy_variables uninspectable_backend_binary_sha256")
    extra = {"openblas": ("OPENBLAS_DEFAULT_NUM_THREADS",), "mkl": ("MKL_DOMAIN_NUM_THREADS", "MKL_THREADING_LAYER"),
        "openmp": ("OMP_MAX_ACTIVE_LEVELS", "OMP_NESTED", "OMP_PLACES", "OMP_PROC_BIND", "OMP_WAIT_POLICY"),
        "intel_openmp": ("KMP_AFFINITY", "KMP_BLOCKTIME", "KMP_HW_SUBSET"), "tbb": ("TBB_NUM_THREADS",)}
    active, wrappers = value["active_backends"], value["numerical_wrappers"]
    _assert(type(active) is list and type(wrappers) is list, "runtime arrays")
    for row in active:
        _keys(row, "architecture backend binary_sha256 effective_thread_count inspection_complete inspection_symbols path_commitment role runtime_configuration threading_layer version")
        _assert(row["role"] == "active_thread_pool" and row["backend"] in ("openblas", "mkl", "intel_openmp", "openmp", "blis", "tbb", "lapack", "blas"), "backend role")
        _hash(row["binary_sha256"])
        _hash(row["path_commitment"])
        _text(row["architecture"])
        if row["effective_thread_count"] is not None:
            _integer(row["effective_thread_count"], 1)
        _keys(row["inspection_symbols"], "effective_thread_count runtime_configuration threading_layer")
        for item in list(row["inspection_symbols"].values()) + [row[k] for k in ("runtime_configuration", "threading_layer", "version")]:
            if item is not None:
                _text(item)
        complete = all(v is not None for v in (row["effective_thread_count"], row["inspection_symbols"]["effective_thread_count"], row["threading_layer"], row["version"]))
        _assert(row["inspection_complete"] is complete, "inspection summary")
    _assert(active == sorted(active, key=lambda r:(r["backend"], r["binary_sha256"], r["path_commitment"])), "backend order")
    for row in wrappers:
        _keys(row, "architecture binary_sha256 interface path_commitment role")
        _assert(row["role"] == "numerical_interface_wrapper" and row["interface"] in ("blas", "lapack"), "wrapper role")
        _hash(row["binary_sha256"])
        _hash(row["path_commitment"])
        _text(row["architecture"])
    _assert(wrappers == sorted(wrappers, key=lambda r:(r["interface"], r["binary_sha256"], r["path_commitment"])), "wrapper order")
    kinds = sorted({r["backend"] for r in active})
    applicable = {n for k in kinds for n in extra.get(k, ())}
    if "intel_openmp" in kinds or any(r["threading_layer"] == "openmp" for r in active):
        applicable.update(extra["openmp"])
    applicable = sorted(applicable)
    _assert(value["active_backend_kinds"] == kinds and value["applicable_additional_thread_variables"] == applicable, "policy population")
    bound = value["bound_applicable_thread_policy"]
    _assert(type(bound) is dict and set(bound) <= set(applicable), "policy binding")
    for item in bound.values():
        _text(item)
    missing = sorted(set(applicable) - set(bound))
    unexpected = value["unexpected_additional_thread_policy_variables"]
    recognized = {n for v in extra.values() for n in v}
    _assert(type(unexpected) is list and unexpected == sorted(set(unexpected)) and
            all(n in recognized and n not in applicable for n in unexpected) and
            value["missing_applicable_thread_policy_variables"] == missing, "policy omissions")
    uninspectable = sorted(r["binary_sha256"] for r in active if not r["inspection_complete"])
    non_single = sorted(r["binary_sha256"] for r in active if r["effective_thread_count"] is not None and r["effective_thread_count"] != 1)
    flags = dict(all_active_backends_inspectable=bool(active) and not uninspectable,
        all_effective_thread_counts_one=bool(active) and not non_single and all(r["effective_thread_count"] == 1 for r in active),
        all_parent_thread_policy_bound=not missing and not unexpected)
    flags["all_runtime_requirements_satisfied"] = all(flags.values())
    _assert(value["uninspectable_backend_binary_sha256"] == uninspectable and value["non_single_thread_backend_binary_sha256"] == non_single and
            all(value[k] is v for k,v in flags.items()), "runtime aggregate")
    payload = {k:v for k,v in value.items() if k != "runtime_record_sha256"}
    _assert(value["runtime_record_sha256"] == hashlib.sha256(b"operatormark-numerical-runtime-record-v2\n" + canonical_json_bytes(payload)).hexdigest(), "runtime commitment")


def _diagnostic(input_role, pointer, code):
    return dict(input=input_role, pointer=pointer, code=code)


def _sorted_diagnostics(rows):
    by_key = {(row["input"], row["pointer"], row["code"]): row for row in rows}
    return [by_key[key] for key in sorted(by_key)]


def _references(value, pointer=""):
    if type(value) is dict:
        if set(value) == {"root_binding_id", "relative_path", "file_bytes", "file_sha256"}:
            yield pointer, value
        else:
            for key, item in value.items():
                yield from _references(item, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
    elif type(value) is list:
        for index, item in enumerate(value):
            yield from _references(item, pointer + "/" + str(index))


def _status(invalid, mismatches, unavailable):
    return "invalid" if invalid else "mismatch" if mismatches else "incomplete" if unavailable else "matched"


def _compare(*, expected, candidate_files, independent_files, retention_only=False):
    validate_final_evidence_expectations(expected)
    e = _plain(expected)
    roots = {r["root_binding_id"]: r["repository_relative_root"] for r in e["root_requirements"]}
    outputs = _output_pair(e, roots)
    expected_hash = hashlib.sha256(b"operatormark-final-evidence-expectations-v1\n" + canonical_json_bytes(e)).hexdigest()
    required = {row["role"]: row for row in e["accepted_input_files"]}
    invalid, mismatch, unavailable, checked, qualifications = [], [], [], [], []
    detailed_rows, side_maps, bad_groups = [], {}, set()
    group_counts = {}
    for side, projections in (("candidate", candidate_files), ("independent", independent_files)):
        if type(projections) is not tuple:
            invalid.append(_diagnostic("unrecognized", "/", "schema_invalid"))
            projections = (dict(role="unrecognized", file=None, data=None),)
            bad_groups.add((side, "unrecognized"))
        buckets = {}
        for supplied in projections:
            row = _plain(supplied)
            role, file, data = "unrecognized", None, None
            good = True
            try:
                _keys(row, "role file data")
                _text(row["role"])
                role = row["role"]
                _file(row["file"], True)
                file = row["file"]
                _assert(row["data"] is None or type(row["data"]) is bytes, "immutable projection data")
                data = row["data"]
                _assert(file is not None or data is None, "unbound data")
            except (FinalEvidenceInputError, TypeError, ValueError, UnicodeError):
                good = False
                invalid.append(_diagnostic(role, "/", "schema_invalid"))
            group = _group(role)
            if group == "unrecognized" or role not in required:
                invalid.append(_diagnostic(role, "/", "membership_mismatch"))
                good = False
            group_counts[(side, group)] = group_counts.get((side, group), 0) + 1
            if not good:
                bad_groups.add((side, group))
            detailed_rows.append(dict(side=side, role=role, file=file,
                observed_data_sha256=hashlib.sha256(data).hexdigest() if type(data) is bytes else None))
            buckets.setdefault(role, []).append(dict(file=file, data=data, good=good))
        usable = {}
        for role, rows in buckets.items():
            if len(rows) != 1:
                invalid.append(_diagnostic(role, "/", "membership_mismatch"))
            elif rows[0]["good"]:
                usable[role] = rows[0]
        side_maps[side] = usable
    # Invalidate ambiguous physical address groups locally, retaining other roles.
    for side, rows in side_maps.items():
        locations = {}
        for role, row in rows.items():
            if row["file"] is None:
                continue
            try:
                address = {k: row["file"][k] for k in ("root_binding_id", "relative_path")}
                location = _resolved(address, roots).casefold()
                locations.setdefault(location, []).append(role)
            except FinalEvidenceInputError:
                invalid.append(_diagnostic(role, "/file", "reference_mismatch"))
        for roles in locations.values():
            if len(roles) > 1:
                for role in roles:
                    invalid.append(_diagnostic(role, "/file", "ownership_duplicate"))
                    rows.pop(role, None)
    parsed = {}
    documents = {"protocol": e["protocol_binding"]}
    documents.update({state + "_authorization": value for state, value in e["authorization_bindings"].items() if value is not None})
    for role, expectation in required.items():
        group = _group(role)
        if retention_only and group != "retention":
            continue
        usable_pair = True
        for side in ("candidate", "independent"):
            row = side_maps[side].get(role)
            if row is None:
                unavailable.append(_diagnostic(role, "/", "outcome_unavailable"))
                usable_pair = False
                continue
            file, data = row["file"], row["data"]
            if file is None:
                unavailable.append(_diagnostic(role, "/file", "outcome_unavailable"))
                usable_pair = False
                continue
            if file != expectation["file"]:
                mismatch.append(_diagnostic(role, "/file", "reference_mismatch"))
            opaque = expectation["validation_profile"] == "existing_state4_opaque_v1"
            if opaque:
                if data is not None:
                    invalid.append(_diagnostic(role, "/data", "disclosure_invalid"))
                    usable_pair = False
                continue
            if data is None:
                unavailable.append(_diagnostic(role, "/data", "outcome_unavailable"))
                usable_pair = False
                continue
            if len(data) != file["file_bytes"] or hashlib.sha256(data).hexdigest() != file["file_sha256"]:
                invalid.append(_diagnostic(role, "/data", "hash_mismatch"))
                usable_pair = False
                continue
            if role in documents:
                document = documents[role]
                payload = hashlib.sha256(b"operatormark-document-payload-v1\n" + document["kind"].encode() + b"\n" + data).hexdigest()
                if payload != document["payload_sha256"] or file["file_sha256"] != document["file"]["sha256"]:
                    mismatch.append(_diagnostic(role, "/payload_sha256", "hash_mismatch"))
            elif expectation["validation_profile"] == "new_final_evidence_v1":
                try:
                    record = _record(data, e)
                    if expectation["schema"] != record["schema"]:
                        raise FinalEvidenceInputError("expected schema")
                    parsed[(side, role)] = record
                except (ValueError, TypeError, KeyError, OverflowError, UnicodeError):
                    invalid.append(_diagnostic(role, "/data", "schema_invalid"))
                    usable_pair = False
            elif expectation["schema"] is not None and expectation["validation_profile"] != "existing_state3_archive_v1":
                try:
                    record = strict_canonical_json_loads(data)
                    _assert(record.get("schema") == expectation["schema"], "upstream schema")
                    if expectation["validation_profile"] == "existing_runtime_v2":
                        _runtime_metadata(record)
                    parsed[(side, role)] = record
                except (ValueError, TypeError, UnicodeError):
                    invalid.append(_diagnostic(role, "/data", "encoding_invalid"))
                    usable_pair = False
        left, right = side_maps["candidate"].get(role), side_maps["independent"].get(role)
        if left is not None and right is not None and left["data"] is not None and right["data"] is not None and left["data"] != right["data"]:
            mismatch.append(_diagnostic(role, "/data", "hash_mismatch"))
        if usable_pair:
            checked.append(role)
    admissions = {row["role"]: row for row in e["upstream_admissions"]}
    admitted_scopes = set()
    def independent_available(file):
        matches = [r for r in side_maps["independent"].values() if r["file"] == file]
        return bool(matches) and all(type(r["data"]) is bytes and len(r["data"]) == file["file_bytes"] and
            hashlib.sha256(r["data"]).hexdigest() == file["file_sha256"] for r in matches)
    for row in admissions.values():
        if row["status"] == "rejected":
            mismatch.append(_diagnostic("root_binding_evidence", "/upstream/" + row["role"], "upstream_rejected"))
        elif row["status"] == "unavailable":
            unavailable.append(_diagnostic("root_binding_evidence", "/upstream/" + row["role"], "authority_unavailable"))
        else:
            material = row["validator_source_files"] + [row["admission_file"]]
            # Opaque original split commitments are intentionally not reread;
            # their identity is a premise of exact accepted State4 scope only.
            for file in row["input_files"]:
                opaque = [r for r in e["accepted_input_files"] if r["file"] == file and r["validation_profile"] == "existing_state4_opaque_v1"]
                if not opaque:
                    material.append(file)
                elif row["scope"] != "state4_acceptance":
                    unavailable.append(_diagnostic("root_binding_evidence", "/upstream/" + row["role"], "authority_unavailable"))
            if all(independent_available(file) for file in material):
                admitted_scopes.add(row["scope"])
            else:
                unavailable.append(_diagnostic("root_binding_evidence", "/upstream/" + row["role"], "authority_unavailable"))
    needed_scopes = {"state3_archive", "state4_acceptance", "runtime", "source_closure", "authority_and_custody"}
    if PHASES.index(e["phase"]) >= 1:
        needed_scopes.update(("scientific_members", "storage_later_phase"))
    else:
        needed_scopes.add("storage_state3_to_state4")
    if PHASES.index(e["phase"]) >= 2:
        needed_scopes.add("scientific_manifests")
    if PHASES.index(e["phase"]) >= 4:
        needed_scopes.add("publication_phase")
    if not retention_only:
        for scope in needed_scopes:
            if scope not in admitted_scopes:
                unavailable.append(_diagnostic("root_binding_evidence", "/scope/" + scope, "authority_unavailable"))
        scopes_by_role = {"state3_audit":"state3_archive", "accepted_controls":"state4_acceptance",
            "source_closure":"source_closure", "root_binding_evidence":"authority_and_custody",
            "state5_scientific":"scientific_members", "state6_scientific":"scientific_manifests"}
        source_pins = [f for component in e["component_requirements"] for f in component["required_source_files"]]
        source_pins.extend(r["file"] for r in e["accepted_input_files"] if r["validation_profile"] == "source_identity_v1")
        for role, expectation in required.items():
            scope = scopes_by_role.get(role)
            if expectation["validation_profile"] == "existing_runtime_v2":
                scope = "runtime"
            elif expectation["validation_profile"] == "existing_state4_opaque_v1":
                scope = "state4_acceptance"
            elif expectation["validation_profile"] == "existing_publication_v1":
                scope = "publication_phase"
            elif expectation["validation_profile"] == "existing_permission_v1":
                scope = "permission_component"
            if scope is None:
                continue
            record = parsed.get(("independent", role))
            interval = None if record is None else record.get("capture", record.get("prefix", {}).get("capture"))
            accepted = False
            for admission in admissions.values():
                if admission["status"] != "accepted" or admission["scope"] != scope or expectation["file"] not in admission["input_files"]:
                    continue
                if any(file not in source_pins or not independent_available(file) for file in admission["validator_source_files"]):
                    continue
                if not independent_available(admission["admission_file"]):
                    continue
                if interval is not None:
                    bound = admission["capture"]
                    if interval.get("last_sequence") is None or interval.get("finished_utc") is None:
                        continue
                    if (bound["invocation_id"] != interval["invocation_id"] or bound["clock_epoch_id"] != interval["clock_epoch_id"] or
                        not bound["first_sequence"] <= interval["first_sequence"] <= interval["last_sequence"] <= bound["last_sequence"] or
                        not bound["started_utc"] <= interval["started_utc"] <= interval["finished_utc"] <= bound["finished_utc"]):
                        continue
                accepted = True
            if not accepted:
                unavailable.append(_diagnostic(role, "/upstream_admission", "authority_unavailable"))
    _graph_checks(e, parsed, side_maps, required, mismatch, unavailable, invalid, checked, retention_only)
    _retention_checks(e, parsed, side_maps, mismatch, unavailable, invalid)
    for (side, role), record in parsed.items():
        if side != "independent" or not (role == "historical_inventory" or role.startswith("history/")):
            continue
        if record.get("schema") != P + "state3_failure_inventory.v2":
            continue
        supplied = side_maps[side][role]
        if supplied["file"] != required[role]["file"]:
            continue
        field = "historical_overbroad_pytest_prohibited_action_accounting"
        try:
            _keys(record, "schema state expected_record_count records relocation_method original_container_removed_only_after_empty all_records_preserved_byte_identical original_relocated_record_count in_place_preserved_record_count failure_record_contents_embedded historical_overbroad_pytest_prohibited_action_accounting review_only_driver record_self_sha256")
            _assert(record[field] == "not_provably_zero" and record["state"] == "state3_pre_generation_audit_no_corpus_controls", "historical classification")
            _assert(type(record["expected_record_count"]) is int and record["expected_record_count"] == 17 and
                    len(record["records"]) == 17 and record["original_relocated_record_count"] == 12 and
                    record["in_place_preserved_record_count"] == 5 and record["failure_record_contents_embedded"] is False, "historical population")
            _assert(record["record_self_sha256"] == hashlib.sha256(canonical_json_bytes({k:v for k,v in record.items() if k != "record_self_sha256"})).hexdigest(), "historical self hash")
            qualifications.append(dict(input=role, pointer="/"+field,
                classification="historical_count_not_provably_zero", evidence_files=[supplied["file"]]))
        except (ValueError, TypeError, KeyError):
            invalid.append(_diagnostic(role, "/"+field, "schema_invalid"))
    qualifications.sort(key=lambda row:(row["input"], row["pointer"], row["classification"]))
    detailed_rows.sort(key=lambda row: (row["side"], row["role"], canonical_json_bytes(row["file"])))
    invalid, mismatch, unavailable = map(_sorted_diagnostics, (invalid, mismatch, unavailable))
    checked = sorted(set(checked))
    details = _seal(dict(schema=P + "final_evidence_details.v1", phase=e["phase"], expected_inputs_sha256=expected_hash,
        input_files=detailed_rows, checked_roles=checked, invalid_inputs=invalid, mismatches=mismatch,
        unavailable_requirements=unavailable, historical_qualifications=qualifications))
    details_bytes = canonical_json_bytes(details)
    details_file = dict(**outputs[1]["address"], file_bytes=len(details_bytes), file_sha256=hashlib.sha256(details_bytes).hexdigest())
    groups = []
    for group in sorted({group for _, group in group_counts}):
        row = dict(group=group)
        for side in ("candidate", "independent"):
            rows = [item for item in detailed_rows if item["side"] == side and _group(item["role"]) == group]
            row[side + "_count"] = group_counts.get((side, group), 0)
            row[side + "_commitment"] = (None if (side, group) in bad_groups else hashlib.sha256(
                b"operatormark-final-input-group-v1\n" + canonical_json_bytes(rows)).hexdigest())
        groups.append(row)
    def sanitized(rows):
        return _sorted_diagnostics([_diagnostic(_group(row["input"]), "/" + _group(row["input"]), row["code"]) for row in rows])
    report = _seal(dict(schema=P + "final_evidence_comparison.v1", profile=e["profile"], phase=e["phase"],
        expected_inputs_sha256=expected_hash, input_groups=groups, status=_status(invalid, mismatch, unavailable),
        checked_roles=sorted({_group(role) for role in checked}), invalid_inputs=sanitized(invalid),
        mismatches=sanitized(mismatch), unavailable_requirements=sanitized(unavailable),
        historical_qualification_count=len(qualifications), sealed_details_file=details_file,
        production_preflight_accepted=False, execution_authorized=False))
    return Comparison(report, details)


def _graph_checks(e, parsed, maps, required, mismatch, unavailable, invalid, checked, retention_only):
    if retention_only:
        return
    independent_by_file = {}
    for role, row in maps["independent"].items():
        if row["file"] is not None:
            independent_by_file[canonical_json_bytes(row["file"])] = role
    graph = {}
    for (side, role), record in parsed.items():
        if side != "independent":
            continue
        graph[role] = []
        for pointer, file in _references(record):
            target = independent_by_file.get(canonical_json_bytes(file))
            if target is not None:
                graph[role].append(target)
            elif not any(file in row["input_files"] + row["validator_source_files"] + [row["admission_file"]] for row in e["upstream_admissions"]):
                unavailable.append(_diagnostic(role, pointer, "outcome_unavailable"))
        if record.get("outcome") in ("unresolved", "refused_before_start", "failed_preserved"):
            unavailable.append(_diagnostic(role, "/outcome", "outcome_unavailable"))
        if role.endswith("_process"):
            if record.get("creation_status") != "started" or record.get("direct_exit_code") != 0 or record.get("reap_status") != "observed_reaped":
                unavailable.append(_diagnostic(role, "/process_outcome", "outcome_unavailable"))
            if any(record.get(name, {}).get("state") != "complete" for name in ("stdout", "stderr")):
                unavailable.append(_diagnostic(role, "/output", "outcome_unavailable"))
        if role.endswith("_guards") and "authorized_counts" in record:
            stage = record["binding"]["state"]
            for family, counts in record["authorized_counts"].items():
                for split, count in counts.items():
                    expected_count = (dict(training=4096, validation=256, test=512)[split] if family == "top_level_rng_calls_by_split" else 1) if stage == "state4" else 0
                    if count["measurement"] != "measured":
                        unavailable.append(_diagnostic(role, "/authorized_counts/" + family + "/" + split, "outcome_unavailable"))
                    elif count["value"] != expected_count:
                        mismatch.append(_diagnostic(role, "/authorized_counts/" + family + "/" + split, "reference_mismatch"))
            for name, count in record["forbidden_counts"].items():
                if count["measurement"] != "measured":
                    unavailable.append(_diagnostic(role, "/forbidden_counts/" + name, "outcome_unavailable"))
                elif count["value"] != 0:
                    mismatch.append(_diagnostic(role, "/forbidden_counts/" + name, "reference_mismatch"))
            coverage = record["coverage"]
            if not all(coverage[name] for name in ("started_before_project_import", "continued_through_exit", "all_required_hooks_accounted")):
                unavailable.append(_diagnostic(role, "/coverage", "outcome_unavailable"))
        if role == "completion":
            manifest = maps["independent"].get("corpus_manifest")
            if manifest is not None and manifest["file"] is not None and record["corpus_manifest_file_sha256"] != manifest["file"]["file_sha256"]:
                mismatch.append(_diagnostic(role, "/corpus_manifest_file_sha256", "reference_mismatch"))
    for start in graph:
        stack = [(start, frozenset())]
        while stack:
            current, ancestors = stack.pop()
            if current in ancestors:
                invalid.append(_diagnostic(start, "/references", "chronology_invalid"))
                break
            stack.extend((child, ancestors | {current}) for child in graph.get(current, ()))
    for requirement in e["required_captures"]:
        role = requirement["role"]
        record = parsed.get(("independent", role))
        if record is None:
            unavailable.append(_diagnostic(role, "/capture", "outcome_unavailable"))
            continue
        capture = record.get("capture", record.get("prefix", {}).get("capture"))
        if capture is None or capture.get("last_sequence") is None:
            unavailable.append(_diagnostic(role, "/capture", "outcome_unavailable"))
            continue
        if capture["invocation_id"] != requirement["invocation_id"] or capture["clock_epoch_id"] != requirement["clock_epoch_id"]:
            mismatch.append(_diagnostic(role, "/capture", "chronology_invalid"))
        events = record.get("ordered_events", [])
        if not events or events[0].get("kind") != requirement["required_first_event"] or events[-1].get("kind") != requirement["required_last_event"]:
            unavailable.append(_diagnostic(role, "/ordered_events", "outcome_unavailable"))
        for prior_role in requirement["predecessor_roles"]:
            prior = parsed.get(("independent", prior_role))
            if prior is None:
                unavailable.append(_diagnostic(role, "/predecessors", "outcome_unavailable"))
                continue
            prior_capture = prior.get("capture", prior.get("prefix", {}).get("capture"))
            if prior_capture is not None and prior_capture.get("last_sequence") is not None:
                if prior_capture["clock_epoch_id"] == capture["clock_epoch_id"]:
                    if prior_capture["last_sequence"] >= capture["first_sequence"]:
                        mismatch.append(_diagnostic(role, "/predecessors", "chronology_invalid"))
                else:
                    prior_file = maps["independent"].get(prior_role, {}).get("file")
                    if prior_file is None or not any(event.get("subject_file") == prior_file and event.get("kind") == "source_bound" for event in events):
                        unavailable.append(_diagnostic(role, "/predecessors", "outcome_unavailable"))


def _retention_qualification(e, maps, *, component, evidence, capture, scope="authority_and_custody"):
    """Independently check caller-designated admissions for one retention fact.

    Neither the catalog's admission labels nor producer validators are used as
    authority. Source membership comes from the separate component requirement.
    """
    requirement = next(row for row in e["component_requirements"] if row["component"] == component)
    def available(file):
        if file is None:
            return False
        matches = [r for r in maps["independent"].values() if r["file"] == file and r["data"] is not None]
        return bool(matches) and all(type(r["data"]) is bytes and len(r["data"]) == file["file_bytes"] and
            hashlib.sha256(r["data"]).hexdigest() == file["file_sha256"] for r in matches)
    accepted = False
    for admission in e["upstream_admissions"]:
        interval = admission["capture"]
        if (admission["role"] not in requirement["admission_roles"] or
            admission["scope"] != scope or
            any(file is None or file not in admission["input_files"] for file in evidence) or
            any(file not in requirement["required_source_files"] for file in admission["validator_source_files"]) or
            interval["invocation_id"] != capture["invocation_id"] or
            interval["clock_epoch_id"] != capture["clock_epoch_id"] or
            not interval["first_sequence"] <= capture["first_sequence"] <= capture["last_sequence"] <= interval["last_sequence"] or
            not interval["started_utc"] <= capture["started_utc"] <= capture["finished_utc"] <= interval["finished_utc"]):
            continue
        if admission["status"] == "rejected":
            return "upstream_rejected"
        if admission["status"] == "accepted" and all(available(file) for file in
                admission["input_files"] + admission["validator_source_files"] + [admission["admission_file"]]):
            accepted = True
    return None if accepted else "authority_unavailable"


def _snapshot_binding(e, catalog, snapshot, role, snapshots, maps, roots, mismatch, unavailable, invalid):
    """Bind reader location and historical linkage without trusting producer checks."""
    actual = maps["independent"][role]["file"]
    address = snapshot["snapshot_address"]
    if {k: actual[k] for k in ("root_binding_id", "relative_path")} != address:
        mismatch.append(_diagnostic(role, "/snapshot_address", "reference_mismatch"))
    try:
        location = _resolved(address, roots)
        _resolved({k: actual[k] for k in ("root_binding_id", "relative_path")}, roots)
        designated = [s for s in catalog["slots"] if s["address"] == address]
        a4 = e["authorization_bindings"]["state4"]["file"]["sha256"]
        attempt = "s4-" + a4[:24]
        anchor = "artifacts/darcy_branch_c_v1/private_corpus_v1/"
        cutoff = snapshot["cutoff"]
        allowed = []
        if cutoff in ("entry", "state4_closed", "state5_closed"):
            for state in ("state4", "state5"):
                if cutoff in ("entry", state + "_closed"):
                    allowed.append(("stage_execution", anchor + "state_evidence/" + attempt + "/" + state +
                                    "/test.sealed/inventories/" + cutoff + ".json"))
        elif cutoff in ("pre_manifest", "pre_move", "post_move", "post_audit", "post_complete",
                        "pre_terminal", "assessment_after_terminal", "failed"):
            allowed.append(("publication", anchor + "publication_evidence/" + attempt +
                            "/test.sealed/corpus_inventories/" + cutoff + ".json"))
        if len(designated) != 1:
            mismatch.append(_diagnostic(role, "/snapshot_address", "membership_mismatch"))
        else:
            slot = designated[0]
            if ((slot["component"], location) not in allowed or
                slot["content_profile"] != P + "corpus_retention_snapshot.v1" or
                slot["representation"] != "loose_file"):
                mismatch.append(_diagnostic(role, "/cutoff", "reference_mismatch"))
            root = next(r for r in e["root_requirements"] if r["root_binding_id"] == address["root_binding_id"])
            code = _retention_qualification(e, maps, component=slot["component"],
                evidence=[root["identity_file"], root["custody_file"], slot["writer"]["source_closure_file"]],
                capture=snapshot["capture"])
            if code:
                (mismatch if code == "upstream_rejected" else unavailable).append(_diagnostic(role, "/snapshot_address", code))
    except (ValueError, TypeError, KeyError, StopIteration):
        mismatch.append(_diagnostic(role, "/snapshot_address", "reference_mismatch"))
    prior = snapshot["prior_snapshot_file"]
    if prior is None:
        return
    predecessors = [(name, value) for name, value in snapshots if maps["independent"][name]["file"] == prior]
    if len(predecessors) != 1:
        # Read failures are not proof of absence. Known bytes at another location
        # establish a substitution, not the requested predecessor's identity.
        copies = [value for name, value in snapshots if
                  maps["independent"][name]["file"]["file_sha256"] == prior["file_sha256"] and
                  maps["independent"][name]["file"]["file_bytes"] == prior["file_bytes"]]
        (mismatch if copies else unavailable).append(_diagnostic(role, "/prior_snapshot_file",
            "reference_mismatch" if copies else "inventory_incomplete"))
        return
    prior_role, previous = predecessors[0]
    if prior_role == role:
        invalid.append(_diagnostic(role, "/prior_snapshot_file", "chronology_invalid"))
        return
    left, right = previous["capture"], snapshot["capture"]
    if left["clock_epoch_id"] == right["clock_epoch_id"]:
        if right["first_sequence"] <= left["last_sequence"]:
            mismatch.append(_diagnostic(role, "/prior_snapshot_file", "chronology_invalid"))
    else:
        # A cross-epoch ordering claim needs independent predecessor-read
        # qualification, not comparison of unrelated sequence counters.
        unavailable.append(_diagnostic(role, "/prior_snapshot_file", "authority_unavailable"))


def _retention_checks(e, parsed, maps, mismatch, unavailable, invalid):
    catalog = parsed.get(("independent", "writer_catalog"))
    if catalog is None:
        unavailable.append(_diagnostic("writer_catalog", "/", "inventory_incomplete"))
        return
    roots = {row["root_binding_id"]: row["repository_relative_root"] for row in e["root_requirements"]}
    if catalog["root_requirements"] != e["root_requirements"]:
        mismatch.append(_diagnostic("writer_catalog", "/root_requirements", "reference_mismatch"))
    registrations = catalog["component_registrations"]
    if [row.get("component") for row in registrations] != list(COMPONENTS):
        invalid.append(_diagnostic("writer_catalog", "/component_registrations", "membership_mismatch"))
        return
    for requirement, registration in zip(e["component_requirements"], registrations):
        component = requirement["component"]
        pointer = "/component_registrations/" + component
        if requirement["expected_slot_ids"] is None:
            unavailable.append(_diagnostic("writer_catalog", pointer, "inventory_incomplete"))
        elif registration.get("slot_ids") != requirement["expected_slot_ids"]:
            mismatch.append(_diagnostic("writer_catalog", pointer, "membership_mismatch"))
        if registration.get("source_files") != requirement["required_source_files"] or registration.get("plan_files") != requirement["required_plan_files"]:
            mismatch.append(_diagnostic("writer_catalog", pointer, "source_mismatch"))
        if registration.get("expansion") == "unresolved":
            unavailable.append(_diagnostic("writer_catalog", pointer, "inventory_incomplete"))
    locations = {}
    expected_slots = {slot["slot_id"]: slot for slot in e["expected_output_slots"]}
    for slot in catalog["slots"]:
        try:
            _validate_slot(slot, roots)
            physical = _resolved(slot["address"], roots).casefold()
            if physical in locations:
                invalid.append(_diagnostic("writer_catalog", "/slots", "ownership_duplicate"))
            locations[physical] = slot["slot_id"]
            if slot["slot_id"] in expected_slots and slot != expected_slots[slot["slot_id"]]:
                mismatch.append(_diagnostic("writer_catalog", "/slots", "reference_mismatch"))
            if slot["domain"]["classification"] == "unresolved":
                unavailable.append(_diagnostic("writer_catalog", "/slots/" + slot["slot_id"], "domain_unresolved"))
        except (ValueError, TypeError, KeyError):
            invalid.append(_diagnostic("writer_catalog", "/slots", "schema_invalid"))
    snapshots = [(role, value) for (side, role), value in parsed.items() if
                 side == "independent" and value.get("schema") == P + "corpus_retention_snapshot.v1"]
    if not any(role == "retention_snapshot" for role, value in snapshots):
        unavailable.append(_diagnostic("retention_snapshot", "/", "inventory_incomplete"))
    for role, snapshot in snapshots:
        if snapshot.get("unavailable_requirements"):
            unavailable.append(_diagnostic(role, "/observations", "inventory_incomplete"))
        try:
            _capture(snapshot["capture"])
            _snapshot_binding(e, catalog, snapshot, role, snapshots, maps, roots, mismatch, unavailable, invalid)
            catalog_input = maps["independent"].get("writer_catalog")
            if catalog_input is None or snapshot["catalog_file"] != catalog_input["file"]:
                mismatch.append(_diagnostic(role, "/catalog_file", "reference_mismatch"))
            slots = {s["slot_id"]: s for s in catalog["slots"]}
            observed = set()
            for index, observation in enumerate(snapshot["observations"]):
                pointer = "/observations/" + str(index)
                _keys(observation, "slot_id address presence content_state file native_identity length_evidence_file origin_file capture_file")
                file = observation["file"]
                presence, content = observation["presence"], observation["content_state"]
                if file is not None:
                    _file(file)
                    if {k: file[k] for k in ("root_binding_id", "relative_path")} != observation["address"]:
                        mismatch.append(_diagnostic(role, pointer + "/file", "reference_mismatch"))
                if (presence not in ("present", "absent", "unknown") or
                    content not in ("complete", "partial", "unavailable", "not_reached") or
                    (file is not None and (presence != "present" or content not in ("complete", "partial"))) or
                    (content == "complete" and (presence != "present" or file is None)) or
                    (presence == "absent" and (file is not None or observation["native_identity"] is not None or observation["capture_file"] is None)) or
                    (content == "not_reached" and (presence != "absent" or observation["capture_file"] is None))):
                    invalid.append(_diagnostic(role, pointer, "schema_invalid"))
                slot = slots.get(observation["slot_id"])
                if slot is None:
                    mismatch.append(_diagnostic(role, pointer, "membership_mismatch"))
                    continue
                logical = observation["address"]
                if logical not in (slot["address"], slot["creating_address"]):
                    anchor = "artifacts/darcy_branch_c_v1/private_corpus_v1/"
                    attempt = anchor + "attempts/s4-" + e["authorization_bindings"]["state4"]["file"]["sha256"][:24] + "/"
                    actual = _resolved(logical, roots)
                    candidates = [a for a in (slot["address"], slot["creating_address"]) if a is not None and
                        _resolved(a, roots).startswith(attempt) and actual == anchor + "corpus/" + _resolved(a, roots)[len(attempt):]]
                    if len(candidates) != 1 or snapshot["cutoff"] not in ("post_move", "post_audit", "post_complete", "pre_terminal", "assessment_after_terminal", "failed"):
                        mismatch.append(_diagnostic(role, pointer, "membership_mismatch"))
                        continue
                    logical = candidates[0]
                    promotion = _retention_qualification(e, maps, component=slot["component"],
                        evidence=[observation["capture_file"], slot["writer"]["source_closure_file"]],
                        capture=snapshot["capture"], scope="publication_phase")
                    if promotion:
                        (mismatch if promotion == "upstream_rejected" else unavailable).append(_diagnostic(role, pointer, promotion))
                location = _resolved(logical, roots).casefold()
                if location in observed:
                    invalid.append(_diagnostic(role, pointer, "ownership_duplicate"))
                    continue
                observed.add(location)
                if observation["address"] == snapshot["snapshot_address"]:
                    invalid.append(_diagnostic(role, pointer, "chronology_invalid"))
                    continue
                if observation["presence"] == "unknown":
                    unavailable.append(_diagnostic(role, pointer, "outcome_unavailable"))
                    continue
                if observation["presence"] not in ("absent", "present"):
                    invalid.append(_diagnostic(role, pointer, "schema_invalid"))
                    continue
                evidence = [observation["capture_file"], slot["writer"]["source_closure_file"]]
                if observation["presence"] == "absent":
                    if observation["file"] is not None or observation["native_identity"] is not None:
                        invalid.append(_diagnostic(role, pointer, "schema_invalid"))
                    if observation["content_state"] == "not_reached":
                        evidence.append(slot["condition"]["source_file"])
                else:
                    if observation["content_state"] not in ("complete", "partial", "unavailable"):
                        invalid.append(_diagnostic(role, pointer, "schema_invalid"))
                    evidence.extend([observation["length_evidence_file"], observation["origin_file"]])
                    if observation["file"] is None and observation["length_evidence_file"] is None:
                        unavailable.append(_diagnostic(role, pointer, "byte_count_mismatch"))
                code = _retention_qualification(e, maps, component=slot["component"],
                    evidence=evidence, capture=snapshot["capture"])
                if code:
                    (mismatch if code == "upstream_rejected" else unavailable).append(_diagnostic(role, pointer, code))
            for slot in slots.values():
                if slot["representation"] == "reference_only":
                    continue
                for address in (slot["address"], slot["creating_address"]):
                    if address is not None and address != snapshot["snapshot_address"] and _resolved(address, roots).casefold() not in observed:
                        unavailable.append(_diagnostic(role, "/observations/" + slot["slot_id"], "inventory_incomplete"))
        except (ValueError, TypeError, KeyError):
            invalid.append(_diagnostic(role, "/observations", "schema_invalid"))


def verify_final_evidence(*, expected: FinalEvidenceExpectations, candidate_files: tuple, independent_files: tuple) -> Comparison:
    return _compare(expected=expected, candidate_files=candidate_files, independent_files=independent_files)


def verify_retention_evidence(*, expected: FinalEvidenceExpectations, candidate_files: tuple, independent_files: tuple) -> Comparison:
    return _compare(expected=expected, candidate_files=candidate_files, independent_files=independent_files, retention_only=True)


def validate_final_evidence_comparison(comparison: Mapping, *, expected: FinalEvidenceExpectations,
                                       candidate_files: tuple, independent_files: tuple) -> None:
    actual = _plain(comparison)
    _keys(actual, "report sealed_details")
    recomputed = _plain(verify_final_evidence(expected=expected, candidate_files=candidate_files, independent_files=independent_files))
    try:
        equal = canonical_json_bytes(actual) == canonical_json_bytes(recomputed)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise FinalEvidenceInputError("invalid comparison encoding") from exc
    _assert(equal, "complete comparison recomputation mismatch")


def canonical_final_evidence_comparison_bytes(comparison: Mapping, *, expected: FinalEvidenceExpectations,
                                              candidate_files: tuple, independent_files: tuple) -> ComparisonBytes:
    validate_final_evidence_comparison(comparison, expected=expected, candidate_files=candidate_files, independent_files=independent_files)
    actual = _plain(comparison)
    return ComparisonBytes(canonical_json_bytes(actual["report"]), canonical_json_bytes(actual["sealed_details"]))
