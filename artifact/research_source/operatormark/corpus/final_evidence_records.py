"""Inert final-evidence records; validation is not upstream acceptance or authority."""

from collections.abc import Mapping
from datetime import datetime
from functools import wraps
import hashlib
import math
import re
import unicodedata

# Exactly Unicode Cc and Cs; keep NFC/type/emptiness checks in their original order.
_CONTROL_OR_SURROGATE = re.compile(r"[\x00-\x1f\x7f-\x9f\ud800-\udfff]")

from operatormark.corpus.canonical import canonical_json_bytes, strict_canonical_json_loads


class FinalEvidenceRecordError(ValueError):
    pass


PREFIX = "operatormark.darcy_branch_c_v1."
BENCHMARK = "darcy_branch_c_v1"
CORPUS_VERSION = "private_corpus_v1"
SPLITS = ("training", "validation", "test")
SHARDS = (64, 4, 8)
FAMILIES = ("rho", "kappa64", "u_tpfa64", "kappa32_areaavg", "u32_areaavg",
            "kappa16_areaavg", "u16_areaavg")
COUNTS = ("structural", "TPFA64", "P1_256", "cross_solver", "overall")
CHECKS = ("source", "authority", "state4_provenance", "rng_and_access", "scientific",
          "manifest_closure", "physical_publication", "retention")
MAXIMA = ("T64_algebraic_backward_error", "T64_boundary_transmissibility_error",
          "T64_maximum_local_conservation_ratio", "T64_maximum_interior_face_antisymmetry",
          "T64_global_balance_ratio", "P256_algebraic_backward_error",
          "P256_boundary_reconstruction_error", "P256_reaction_balance_ratio", "delta_L2", "delta_J")
CALLS = ("not_called", "returned_success", "returned_failure", "outcome_unavailable")
EFFECTS = ("none_observed", "expected_effect_observed", "unexpected_effect_observed", "unknown")
FAILURES = ("access_denied", "authority_unavailable", "authority_refused", "input_invalid",
            "custody_mismatch", "io_failed", "result_unavailable", "retention_failed", "cleanup_failed")
EVENTS = ("authority_checked", "source_bound", "environment_checked", "storage_checked",
          "guards_installed", "project_import_checked", "worker_started", "worker_exited",
          "scientific_closed", "metadata_prepared", "access_observed", "failure_observed",
          "resources_released", "capture_closed")
MISSING = ("not_captured", "bytes_unavailable", "outcome_unavailable",
           "source_rule_unavailable", "custody_unavailable")
SCHEMAS = {
    "accepted_controls_projection": "binding capture state3_audit compute_feasibility_report pre_solve_manifest controls_completion rng_audit read_only_evidence ordered_split_manifests ordered_control_shards original_file_inventory acceptance_evidence_files",
    "state_guard_closure": "binding capture process_creation_evidence_file frozen_base_import_closure_file rng_state_commitments rng_state_equalities authorized_counts forbidden_counts guard_lifetime_evidence_files coverage",
    "process_lifecycle_event": "binding event capture pid child_source_closure_file supervisor_source_file authority_scope_file creation_record_file direct_exit_code",
    "state_process_outcome": "binding role capture source_closure_file authority_scope_file creation_status process_identity direct_exit_code stdout stderr guards_file reap_status failures",
    "state_execution_evidence": "binding capture predecessors source_and_dependency_bindings environment_before environment_after storage_boundaries git_before git_after process_outcome guard_closure scientific_closure ordered_events failure_files outcome",
    "corpus_failure_inventory": "benchmark attempt_id prefix historical_inventory_files current_failure_events retention_inventory_file unavailable_evidence",
    "corpus_access_ledger": "benchmark attempt_id prefix split_access_state ordered_access_events historical_unknown_access_files unavailable_evidence",
    "writer_exception_graph": "binding capture operation_id primary_exception_id exceptions ordered_cleanup_failures capture_failures",
    "scientific_closure": "binding capture phase accepted_controls_projection source_closure_file context_file ordered_shards ordered_splits aggregate failure_files",
    "final_audit_details": "benchmark attempt_id corpus_manifest prefix scientific_closure state_execution_files access_extension failure_extension post_move_publication_files retention_inventory_file checks unavailable_evidence",
    "final_audit": "benchmark corpus_version attempt_id corpus_manifest parent_context details_file prefix split_summary check_status publication_location",
    "corpus_provenance": "benchmark parent_context protocol_binding authorization_bindings baseline_binding implementation_binding source_and_dependency_bindings environment_and_platform_bindings gate_bindings execution_evidence failure_inventory access_ledger rights_record legacy_data_or_model_output_used",
    "corpus_manifest": "benchmark corpus_version attempt_id parent_context protocol_file_sha256 authorization_file_sha256_by_state solver_spec_baseline_commit solver_spec_baseline_tree implementation_commit implementation_tree environment_and_dependency_bindings gate_bindings ordered_split_manifests rights_record provenance_record failure_inventory controls_pre_solve_manifest controls_completion",
    "corpus_complete": "state corpus_manifest_filename corpus_manifest_file_sha256 completion_time_utc",
}


def _need(ok, message):
    if not ok:
        raise FinalEvidenceRecordError(message)


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return [_plain(v) for v in value]
    return value


def _object(value, keys):
    _need(type(value) is dict and set(value) == set(keys.split() if type(keys) is str else keys),
          "exact object keys required")


def _text(value, empty=False):
    _need(type(value) is str and (empty or bool(value)), "text required")
    _need(unicodedata.normalize("NFC", value) == value, "NFC required")
    _need(_CONTROL_OR_SURROGATE.search(value) is None, "control/surrogate text")


def _sha(value, width=64):
    _need(type(value) is str and re.fullmatch("[0-9a-f]{" + str(width) + "}", value) is not None
          and value != "0" * width, "nonzero hash required")


def _int(value, lower=0):
    _need(type(value) is int and value >= lower, "exact integer required")


def _enum(value, choices):
    _need(type(value) is str and value in choices, "closed token required")


def _path(value, empty=False):
    _text(value, empty)
    if value == "":
        return
    for part in value.split("/"):
        _need(part not in ("", ".", "..") and part[-1:] not in (".", " "), "path component alias")
        _need(not any(c in part for c in '\\:*?"<>|'), "unsafe path")
        _need(not re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])", part.split(".")[0]),
              "reserved path")


def _array(value, validator, *, nonempty=False, ordered=False):
    _need(type(value) is list and (not nonempty or bool(value)), "array required")
    for row in value:
        validator(row)
    if ordered:
        keys = [canonical_json_bytes(row) for row in value]
        _need(keys == sorted(set(keys)), "unique canonical order required")


def _ref(value, nullable=False):
    if value is None and nullable:
        return
    _object(value, "root_binding_id relative_path file_bytes file_sha256")
    _text(value["root_binding_id"])
    _need(":" not in value["root_binding_id"], "root ID colon")
    _path(value["relative_path"])
    _int(value["file_bytes"])
    _sha(value["file_sha256"])


def _refs(value, ordered=True, nonempty=False):
    _array(value, _ref, ordered=ordered, nonempty=nonempty)


def _address(value):
    _object(value, "root_binding_id relative_path")
    _text(value["root_binding_id"])
    _need(":" not in value["root_binding_id"], "root ID colon")
    _path(value["relative_path"], True)


def _local(value, path=None, schema=None, opaque=False):
    _object(value, "relative_path size sha256" + ("" if opaque else " schema"))
    _path(value["relative_path"])
    _int(value["size"], 1)
    _sha(value["sha256"])
    if not opaque:
        _text(value["schema"])
    if path is not None:
        _need(value["relative_path"] == path, "local path mismatch")
    if schema is not None:
        _need(value["schema"] == PREFIX + schema, "local schema mismatch")


def _external(value):
    _object(value, "filename literal_path_sha256 sha256 size")
    _path(value["filename"])
    _need("/" not in value["filename"], "external basename")
    _sha(value["literal_path_sha256"])
    _sha(value["sha256"])
    _int(value["size"], 1)


def _implementation(value):
    _object(value, "commit tree sole_parent")
    for item in value.values():
        _sha(item, 40)


def _attempt(value, a4=None):
    _need(type(value) is str and re.fullmatch("s4-[0-9a-f]{24}", value) is not None, "attempt")
    if a4 is not None:
        _need(value == "s4-" + a4[:24], "attempt/A4 mismatch")


def _binding(value, *, benchmark=BENCHMARK):
    _object(value, "benchmark state attempt_id protocol_file_sha256 implementation state4_authorization_file_sha256 authorization_file_sha256")
    _need(value["benchmark"] == benchmark, "benchmark")
    _enum(value["state"], ("state4", "state5", "state6"))
    _implementation(value["implementation"])
    for name in ("protocol_file_sha256", "state4_authorization_file_sha256", "authorization_file_sha256"):
        _sha(value[name])
    _attempt(value["attempt_id"], value["state4_authorization_file_sha256"])
    _need((value["state"] == "state4") ==
          (value["authorization_file_sha256"] == value["state4_authorization_file_sha256"]), "stage authority")


V2_EXECUTION_KINDS = frozenset(('state_guard_closure','process_lifecycle_event',
                              'state_process_outcome','writer_exception_graph'))


def execution_prefix(binding):
    """Explicit private execution version; scientific v1 schemas stay v1-only."""
    _need(type(binding) is dict and binding.get('benchmark') in (BENCHMARK,'darcy_branch_c_v2'),
          'unsupported execution benchmark')
    _binding(binding, benchmark=binding['benchmark'])
    return 'operatormark.'+binding['benchmark']+'.'


def execution_domain(binding):
    execution_prefix(binding)
    return 'operatormark-state-evidence-invocation-'+('v2' if binding['benchmark']=='darcy_branch_c_v2' else 'v1')


def _utc(value):
    _need(type(value) is str and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", value) is not None, "UTC format")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise FinalEvidenceRecordError("UTC calendar") from exc


def _capture(value, binding=None, incomplete=False):
    _object(value, "invocation_id clock_epoch_id first_sequence last_sequence started_utc finished_utc")
    _sha(value["invocation_id"])
    _text(value["clock_epoch_id"])
    _int(value["first_sequence"])
    _utc(value["started_utc"])
    if value["last_sequence"] is None or value["finished_utc"] is None:
        _need(incomplete and value["last_sequence"] is None and value["finished_utc"] is None, "incomplete capture")
    else:
        _int(value["last_sequence"], value["first_sequence"])
        _utc(value["finished_utc"])
        _need(value["finished_utc"] >= value["started_utc"], "capture time order")
    if binding is not None:
        raw = (execution_domain(binding)+"\n" + binding["state"] + "\n" +
               binding["authorization_file_sha256"] + "\n" + binding["state4_authorization_file_sha256"]).encode()
        _need(value["invocation_id"] == hashlib.sha256(raw).hexdigest(), "invocation commitment")


def _count(value):
    _object(value, "measurement value evidence_files")
    _enum(value["measurement"], ("measured", "not_measured", "unavailable"))
    _refs(value["evidence_files"], nonempty=value["measurement"] == "measured")
    if value["measurement"] == "measured":
        _int(value["value"])
    else:
        _need(value["value"] is None, "unknown count is not zero")


def _failure(value):
    _object(value, "kind native_error_hex exception_file affected_addresses")
    _enum(value["kind"], FAILURES)
    error = value["native_error_hex"]
    _need(error is None or type(error) is str and re.fullmatch("0x(0|[1-9a-f][0-9a-f]*)", error) is not None
          and int(error, 16) < 2**32, "native error")
    _ref(value["exception_file"], True)
    _array(value["affected_addresses"], _address, ordered=True)


def _missing(value):
    _object(value, "role expected_file reason")
    _text(value["role"])
    _ref(value["expected_file"], True)
    _enum(value["reason"], MISSING)


def _prefix(value, next_boundary=None):
    _object(value, "capture predecessor_files last_observed_event_file next_boundary")
    _capture(value["capture"])
    _refs(value["predecessor_files"])
    _ref(value["last_observed_event_file"])
    _enum(value["next_boundary"], ("metadata_finalization", "promotion", "audit_publication", "terminal_publication"))
    if next_boundary is not None:
        _need(value["next_boundary"] == next_boundary, "prefix boundary")


def _event(value):
    _object(value, "sequence kind source_file subject_file prior_event_file")
    _int(value["sequence"])
    _enum(value["kind"], EVENTS)
    _ref(value["source_file"])
    _ref(value["subject_file"], True)
    _ref(value["prior_event_file"], True)


def _failure_event(value):
    _object(value, "event_file capture operation_id call_status effect_status primary follow_on surviving_files location location_evidence_file preservation_status")
    _ref(value["event_file"])
    _capture(value["capture"], incomplete=True)
    _sha(value["operation_id"])
    _enum(value["call_status"], CALLS)
    _enum(value["effect_status"], EFFECTS)
    if value["primary"] is None:
        _need(value["call_status"] == "outcome_unavailable", "missing primary outcome")
    else:
        _failure(value["primary"])
    _array(value["follow_on"], _failure)
    _refs(value["surviving_files"])
    _enum(value["location"], ("T", "F", "Q", "unknown"))
    _ref(value["location_evidence_file"], value["location"] == "unknown")
    _enum(value["preservation_status"], ("retained", "partial", "unavailable"))
    _need(value["call_status"] != "not_called" or value["effect_status"] != "expected_effect_observed", "uncalled effect")


def _access(value):
    _object(value, "capture actor_evidence_file authority_scope_file operation subject decision decision_evidence_file call_status effect_status count_evidence failure_files")
    _capture(value["capture"], incomplete=True)
    _ref(value["actor_evidence_file"])
    _ref(value["authority_scope_file"])
    _enum(value["operation"], ("metadata_read", "scientific_read", "scientific_write", "permission_observation",
                              "permission_change", "seed_access", "rng_call", "publication", "resource_release"))
    if value["subject"] != "process_rng":
        _address(value["subject"])
    _enum(value["decision"], ("authorized", "denied", "unavailable"))
    _ref(value["decision_evidence_file"], value["decision"] == "unavailable")
    _enum(value["call_status"], CALLS)
    _enum(value["effect_status"], EFFECTS)
    _count(value["count_evidence"])
    _refs(value["failure_files"])
    _need(value["call_status"] != "not_called" or value["effect_status"] != "expected_effect_observed", "uncalled effect")


def _identity(value):
    _object(value, "volume_serial_number_hex file_id_128_hex legacy_volume_serial_number_hex legacy_file_index_hex")
    volume = value["volume_serial_number_hex"]
    _need(type(volume) is str and re.fullmatch("0x(0|[1-9a-f][0-9a-f]*)", volume) is not None
          and int(volume, 16) < 2**64, "native volume")
    for name, width in (("file_id_128_hex", 32), ("legacy_volume_serial_number_hex", 8), ("legacy_file_index_hex", 16)):
        _need(type(value[name]) is str and re.fullmatch("[0-9a-f]{" + str(width) + "}", value[name]) is not None, "native identity")


def _document(value, kind=None):
    _object(value, "kind file payload_rule payload_sha256")
    _enum(value["kind"], ("protocol", "state3_authorization", "state4_authorization", "state5_authorization", "state6_authorization"))
    if kind is not None:
        _need(value["kind"] == kind, "document kind")
    _external(value["file"])
    _need(value["payload_rule"] == "opm_document_full_bytes_v1", "document payload rule")
    _sha(value["payload_sha256"])
    _need(value["payload_sha256"] != value["file"]["sha256"], "file hash is not payload hash")


def _git(value):
    _object(value, "branch head tree parents status_short_untracked_all tracked_diff_sha256 tracked_diff_size_bytes staged_diff_sha256 staged_diff_size_bytes tracked_changes_empty staged_changes_empty untracked_inclusive_status_empty")
    _text(value["branch"])
    _sha(value["head"], 40)
    _sha(value["tree"], 40)
    _array(value["parents"], lambda x: _sha(x, 40))
    _need(type(value["status_short_untracked_all"]) is str, "Git status text")
    for prefix in ("tracked", "staged"):
        _sha(value[prefix + "_diff_sha256"])
        _int(value[prefix + "_diff_size_bytes"])
        _need(value[prefix + "_changes_empty"] is (value[prefix + "_diff_size_bytes"] == 0), "Git diff summary")
    _need(value["untracked_inclusive_status_empty"] is (value["status_short_untracked_all"] == ""), "Git status summary")


def _source_bindings(value):
    _object(value, "source_manifest_file implementation_files baseline_files dependency_lock base_import_closure_file qualification_files")
    _ref(value["source_manifest_file"])
    _ref(value["base_import_closure_file"])
    _refs(value["qualification_files"])
    def source(row):
        _object(row, "relative_path file_bytes file_sha256 git_blob source_role")
        _path(row["relative_path"])
        _int(row["file_bytes"])
        _sha(row["file_sha256"])
        _enum(row["source_role"], ("implementation", "solver_baseline", "dependency_lock", "runtime_binary"))
        if row["git_blob"] is None:
            _need(row["source_role"] == "runtime_binary", "missing Git blob")
        else:
            _sha(row["git_blob"], 40)
    for field in ("implementation_files", "baseline_files"):
        _array(value[field], source)
        names = [r["relative_path"] for r in value[field]]
        _need(names == sorted(names) and len({p.casefold() for p in names}) == len(names), "source order/collision")
    baseline = {"configs/darcy_branch_c_v1.yaml", "operatormark/benchmarks/darcy_spec.py",
                "operatormark/evaluation/darcy_cross_solver_tolerance.py", "operatormark/evaluation/darcy_solver_heldout.py",
                "operatormark/solvers/darcy_p1.py", "operatormark/solvers/darcy_tpfa.py",
                "reports/darcy_branch_c_v1_preregistration.md"}
    _need({r["relative_path"] for r in value["baseline_files"]} == baseline, "seven baseline paths")
    _need(all(r["source_role"] == "solver_baseline" for r in value["baseline_files"]), "baseline role")
    source(value["dependency_lock"])
    _need(value["dependency_lock"]["relative_path"] == "requirements-lock-cpu.txt" and
          value["dependency_lock"]["source_role"] == "dependency_lock", "dependency lock")


def _environment(value):
    _object(value, "file frozen_state3_environment_file parent_environment_commitment base_import_closure_sha256 numerical_runtime_sha256 source_profile")
    _ref(value["file"])
    _external(value["frozen_state3_environment_file"])
    for name in ("parent_environment_commitment", "base_import_closure_sha256", "numerical_runtime_sha256"):
        _sha(value[name])
    _need(value["source_profile"] == "existing_runtime_v2", "runtime profile")


def _environments(value):
    _object(value, "state3 state4 state5 state6")
    _external(value["state3"])
    for stage in ("state4", "state5", "state6"):
        _object(value[stage], "before after")
        for item in value[stage].values():
            _environment(item)


def _gates(value):
    _object(value, "gate4 gate6 gate7 gate8 gate9")
    for item in value.values():
        _external(item)


def _sample_counts(value, total):
    _object(value, COUNTS)
    for count in value.values():
        _int(count)
        _need(count <= total, "sample count exceeds population")


def _maxima(value):
    _object(value, MAXIMA)
    for name, encoded in value.items():
        _need(type(encoded) is str, "hexadecimal maximum")
        try:
            number = float.fromhex(encoded)
        except ValueError as exc:
            raise FinalEvidenceRecordError("hexadecimal maximum") from exc
        _need(math.isfinite(number) and number >= 0 and number.hex() == encoded, "canonical finite maximum")


def _sealed_local(value, name):
    _local(value, "corpus_staging/test.sealed/" + name + ".json", name + ".v1")


def _header(record, kind):
    value = _plain(record)
    field = "manifest_self_sha256" if kind == "corpus_manifest" else "record_self_sha256"
    _object(value, "schema " + SCHEMAS[kind] + " " + field)
    prefix = execution_prefix(value['binding']) if kind in V2_EXECUTION_KINDS else PREFIX
    _need(value["schema"] == prefix + kind + ".v1", "schema mismatch")
    _sha(value[field])
    payload = {k: v for k, v in value.items() if k != field}
    try:
        data = canonical_json_bytes(payload)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise FinalEvidenceRecordError("JSON encoding") from exc
    _need(hashlib.sha256(data).hexdigest() == value[field], "self hash mismatch")
    if "benchmark" in value:
        _need(value["benchmark"] == BENCHMARK, "benchmark")
    if "corpus_version" in value:
        _need(value["corpus_version"] == CORPUS_VERSION, "corpus version")
    if "attempt_id" in value:
        _attempt(value["attempt_id"])
    if "binding" in value:
        if kind in V2_EXECUTION_KINDS: execution_prefix(value['binding'])
        else: _binding(value["binding"])
    return value


def _shard_rows():
    return [(split, index) for split, count in zip(SPLITS, SHARDS) for index in range(count)]


def _directory(split):
    return "test.sealed" if split == "test" else split


def _control_base(split, index):
    return "state4_controls/" + _directory(split) + "/shard_" + f"{index:04d}"


def validate_accepted_controls_projection(record: Mapping) -> None:
    r = _header(record, "accepted_controls_projection")
    _need(r["binding"]["state"] == "state4", "controls stage")
    _capture(r["capture"], r["binding"])
    for name in ("state3_audit", "compute_feasibility_report"):
        _external(r[name])
    for name, path, schema in (
        ("pre_solve_manifest", "state4_controls/pre_solve_manifest.json", "state4_pre_solve_manifest.v2"),
        ("controls_completion", "STATE4_CONTROLS_COMPLETE.json", "state4_controls_complete.v1"),
        ("rng_audit", "state4_controls/rng_audit.json", "state4_rng_audit.v1"),
        ("read_only_evidence", "state4_controls/read_only_evidence.json", "state4_read_only_evidence.v1"),
    ):
        _local(r[name], path, schema)
    _need(type(r["ordered_split_manifests"]) is list and len(r["ordered_split_manifests"]) == 3, "three opaque splits")
    for split, row in zip(SPLITS, r["ordered_split_manifests"]):
        _local(row, "state4_controls/" + _directory(split) + "/split_manifest.json", opaque=True)
    rows = r["ordered_control_shards"]
    _need(type(rows) is list and len(rows) == 76, "76 control shards")
    all_hashes = {name: [] for name in ("ordered_control_tensor_sha256", "ordered_rho_storage_tensor_sha256",
                                      "ordered_macro_coefficient_tensor_sha256", "ordered_control_record_self_sha256")}
    expected_paths = {r[n]["relative_path"] for n in ("pre_solve_manifest", "controls_completion", "rng_audit", "read_only_evidence")}
    expected_paths.update(row["relative_path"] for row in r["ordered_split_manifests"])
    for (split, index), row in zip(_shard_rows(), rows):
        _object(row, "split shard_index controls_manifest controls_jsonl rho rho_tensor_sha256 ordered_control_tensor_sha256 ordered_rho_storage_tensor_sha256 ordered_macro_coefficient_tensor_sha256 ordered_control_record_self_sha256")
        _need(row["split"] == split and type(row["shard_index"]) is int and row["shard_index"] == index, "control shard order")
        base = _control_base(split, index)
        _local(row["controls_manifest"], base + ".manifest.json", "state4_controls_shard_manifest.v1")
        _local(row["controls_jsonl"], base + ".controls.jsonl", opaque=True)
        _local(row["rho"], "corpus_staging/" + _directory(split) + f"/shard_{index:04d}/rho.float64.npy", opaque=True)
        _sha(row["rho_tensor_sha256"])
        expected_paths.update(row[name]["relative_path"] for name in ("controls_manifest", "controls_jsonl", "rho"))
        for name in all_hashes:
            _array(row[name], _sha)
            _need(len(row[name]) == 64, "64 sample commitments")
            all_hashes[name].extend(row[name])
    for hashes in all_hashes.values():
        _need(len(set(hashes)) == len(hashes), "duplicate sample commitment")
    inventory = r["original_file_inventory"]
    _need(type(inventory) is list and len(inventory) == 235, "235 original files")
    paths = []
    identities = []
    for row in inventory:
        _object(row, "file native_identity origin_evidence_file custody_file permission_evidence_files")
        _ref(row["file"])
        _identity(row["native_identity"])
        _ref(row["origin_evidence_file"])
        _ref(row["custody_file"])
        _refs(row["permission_evidence_files"])
        paths.append(row["file"]["relative_path"])
        identities.append(canonical_json_bytes(row["native_identity"]))
    _need(paths == sorted(expected_paths) and len(set(identities)) == 235, "original inventory membership/origin")
    _refs(r["acceptance_evidence_files"], nonempty=True)


def validate_state_guard_closure(record: Mapping) -> None:
    r = _header(record, "state_guard_closure")
    coverage = r["coverage"]
    _object(coverage, "started_before_project_import continued_through_exit all_required_hooks_accounted unavailable_hook_evidence_files")
    for name in ("started_before_project_import", "continued_through_exit", "all_required_hooks_accounted"):
        _need(type(coverage[name]) is bool, "coverage boolean")
    incomplete = not all(coverage[n] for n in ("started_before_project_import", "continued_through_exit", "all_required_hooks_accounted"))
    _capture(r["capture"], r["binding"], incomplete)
    _refs(coverage["unavailable_hook_evidence_files"])
    for name in ("process_creation_evidence_file", "frozen_base_import_closure_file"):
        _ref(r[name], incomplete)
    _refs(r["guard_lifetime_evidence_files"], nonempty=not incomplete)
    checkpoints = ("after_base_import", "after_guard_install", "after_project_import", "after_execution")
    _object(r["rng_state_commitments"], checkpoints)
    _object(r["rng_state_equalities"], checkpoints[1:])
    for name in checkpoints:
        row = r["rng_state_commitments"][name]
        _object(row, "numpy_legacy_sha256 python_random_sha256")
        for value in row.values():
            if value is not None:
                _sha(value)
    baseline = r["rng_state_commitments"]["after_base_import"]
    for name in checkpoints[1:]:
        row = r["rng_state_equalities"][name]
        _object(row, "numpy_legacy_equals_baseline python_random_equals_baseline")
        for family in ("numpy_legacy", "python_random"):
            left, right = baseline[family + "_sha256"], r["rng_state_commitments"][name][family + "_sha256"]
            expected = None if left is None or right is None else left == right
            _need(row[family + "_equals_baseline"] is expected, "RNG equality must be recomputed")
    _object(r["authorized_counts"], "seed_access_by_split pcg64_constructors_by_split generator_constructors_by_split top_level_rng_calls_by_split")
    for row in r["authorized_counts"].values():
        _object(row, SPLITS)
        for count in row.values():
            _count(count)
    _object(r["forbidden_counts"], "rng_calls corpus_seed_access legacy_data_access model_execution network_access unauthorized_test_access")
    for value in r["forbidden_counts"].values():
        _count(value)


def validate_process_lifecycle_event(record: Mapping) -> None:
    r = _header(record, "process_lifecycle_event")
    _capture(r["capture"], r["binding"])
    _enum(r["event"], ("created", "exited"))
    _int(r["pid"], 1)
    for name in ("child_source_closure_file", "supervisor_source_file", "authority_scope_file"):
        _ref(r[name])
    if r["event"] == "created":
        _need(r["creation_record_file"] is None and r["direct_exit_code"] is None, "created event future outcome")
    else:
        _ref(r["creation_record_file"])
        _need(type(r["direct_exit_code"]) is int, "direct exit integer")


def _captured(value, process=False):
    _object(value, "state file")
    _enum(value["state"], ("complete", "partial", "unavailable", "not_started") if process else
          ("complete", "partial", "unavailable"))
    if value["state"] in ("complete", "partial"):
        _ref(value["file"])
    else:
        _need(value["file"] is None, "unavailable captured bytes")


def validate_state_process_outcome(record: Mapping) -> None:
    r = _header(record, "state_process_outcome")
    _need(r["binding"]["state"] in ("state4", "state5"), "State6 uses existing publication process record")
    _need(r["role"] == {"state4": "state4_controls_child", "state5": "state5_solve_child"}[r["binding"]["state"]], "process role")
    _capture(r["capture"], r["binding"], True)
    _ref(r["source_closure_file"])
    _ref(r["authority_scope_file"])
    _enum(r["creation_status"], ("not_started", "started", "unknown"))
    _enum(r["reap_status"], ("not_applicable", "observed_reaped", "unresolved", "failed"))
    _array(r["failures"], _failure)
    for name in ("stdout", "stderr"):
        _captured(r[name], True)
    _ref(r["guards_file"], True)
    if r["process_identity"] is not None:
        _object(r["process_identity"], "pid creation_evidence_file")
        _int(r["process_identity"]["pid"], 1)
        _ref(r["process_identity"]["creation_evidence_file"])
    if r["direct_exit_code"] is not None:
        _need(type(r["direct_exit_code"]) is int and r["creation_status"] == "started", "unknown process exit")
    if r["creation_status"] == "not_started":
        _need(r["process_identity"] is None and r["direct_exit_code"] is None and r["reap_status"] == "not_applicable", "not-started contradiction")
        _need(all(r[n]["state"] == "not_started" for n in ("stdout", "stderr")) and r["guards_file"] is None, "not-started output")
        _need(bool(r["failures"]), "refusal evidence required")
    elif r["creation_status"] == "started":
        _need(r["process_identity"] is not None and r["reap_status"] != "not_applicable", "started identity/reaping")
        _need(all(r[n]["state"] != "not_started" for n in ("stdout", "stderr")), "started stream contradiction")
    else:
        _need(r["direct_exit_code"] is None and r["reap_status"] in ("unresolved", "failed") and bool(r["failures"]), "unknown process")
    if r["reap_status"] == "observed_reaped":
        _need(r["direct_exit_code"] is not None, "observed exit unavailable")
    if r["guards_file"] is None and r["creation_status"] != "not_started":
        _need(bool(r["failures"]), "unavailable guards unexplained")


def validate_state_execution_evidence(record: Mapping) -> None:
    r = _header(record, "state_execution_evidence")
    _enum(r["outcome"], ("completed", "refused_before_start", "failed_preserved", "unresolved"))
    complete = r["outcome"] == "completed"
    _capture(r["capture"], r["binding"], not complete)
    state = r["binding"]["state"]
    predecessors = "state3_audit compute_feasibility_report"
    if state != "state4":
        predecessors += " accepted_controls_projection pre_solve_manifest controls_completion"
    if state == "state6":
        predecessors += " state5_execution publication_plan"
    _object(r["predecessors"], predecessors)
    missing = False
    for name, value in r["predecessors"].items():
        if value is None:
            _need(not complete, "missing completed predecessor")
            missing = True
        elif name in ("state3_audit", "compute_feasibility_report", "state5_execution"):
            _external(value)
        elif name in ("accepted_controls_projection", "publication_plan"):
            _ref(value)
        else:
            _local(value)
    for name, validator in (("source_and_dependency_bindings", _source_bindings),
                            ("environment_before", _environment), ("environment_after", _environment),
                            ("git_before", _git), ("git_after", _git)):
        if r[name] is None:
            _need(not complete, "missing completed observation")
            missing = True
        else:
            validator(r[name])
    def storage(row):
        _object(row, "boundary file comparison_file comparison_profile accepted_scope_file")
        _enum(row["boundary"], ("before_execution", "after_execution", "pre_finalize", "pre_move", "post_move",
                               "post_audit", "post_complete", "failure_location", "post_quarantine"))
        for name in ("file", "comparison_file", "accepted_scope_file"):
            _ref(row[name])
        _text(row["comparison_profile"])
    _array(r["storage_boundaries"], storage)
    _need(len({row["boundary"] for row in r["storage_boundaries"]}) == len(r["storage_boundaries"]), "duplicate storage boundary")
    for name in ("process_outcome", "guard_closure"):
        _ref(r[name], not complete)
        missing |= r[name] is None
    if state == "state4":
        _need(r["scientific_closure"] is None, "State4 scientific closure")
    else:
        _ref(r["scientific_closure"], not complete)
    _array(r["ordered_events"], _event)
    sequences = [row["sequence"] for row in r["ordered_events"]]
    _need(sequences == sorted(set(sequences)), "event sequence order")
    _need(all(r["capture"]["first_sequence"] <= n and
              (r["capture"]["last_sequence"] is None or n <= r["capture"]["last_sequence"]) for n in sequences), "event cutoff")
    _refs(r["failure_files"])
    if complete:
        _need(not r["failure_files"] and not missing, "completed failure")
        _need(r["git_before"] == r["git_after"], "Git changed during execution")
        git = r["git_before"]
        identity = r["binding"]["implementation"]
        _need((git["head"], git["tree"], git["parents"]) ==
              (identity["commit"], identity["tree"], [identity["sole_parent"]]), "execution Git binding")
        _need(all(git[k] for k in ("tracked_changes_empty", "staged_changes_empty", "untracked_inclusive_status_empty")), "dirty execution")
        _need(bool(sequences) and sequences[0] == r["capture"]["first_sequence"] and
              sequences[-1] == r["capture"]["last_sequence"], "execution interval")
    # Independent comparison supplies externally retained missing-evidence history.


def validate_corpus_failure_inventory(record: Mapping) -> None:
    r = _header(record, "corpus_failure_inventory")
    _prefix(r["prefix"], "metadata_finalization")
    _refs(r["historical_inventory_files"], ordered=False)
    _array(r["current_failure_events"], _failure_event)
    _ref(r["retention_inventory_file"])
    _array(r["unavailable_evidence"], _missing, ordered=True)


def validate_corpus_access_ledger(record: Mapping) -> None:
    r = _header(record, "corpus_access_ledger")
    _prefix(r["prefix"], "metadata_finalization")
    _need(r["split_access_state"] == {
        "training": "accepted_training_authorization_required",
        "validation": "accepted_validation_authorization_required",
        "test": "sealed_test_opening_authorization_required"}, "split access policy")
    _array(r["ordered_access_events"], _access)
    _refs(r["historical_unknown_access_files"], ordered=False)
    _array(r["unavailable_evidence"], _missing, ordered=True)


def validate_writer_exception_graph(record: Mapping) -> None:
    r = _header(record, "writer_exception_graph")
    _capture(r["capture"], r["binding"], True)
    _sha(r["operation_id"])
    _need(r["primary_exception_id"] == "e0", "primary graph node")
    nodes = r["exceptions"]
    _need(type(nodes) is list and bool(nodes), "exception nodes")
    for index, row in enumerate(nodes):
        _object(row, "exception_id class_module class_qualname message_bytes traceback_bytes cause_id context_id suppress_context")
        _need(row["exception_id"] == "e" + str(index), "first-encounter node IDs")
        _text(row["class_module"])
        _text(row["class_qualname"])
        _captured(row["message_bytes"])
        _captured(row["traceback_bytes"])
        _need(type(row["suppress_context"]) is bool, "suppression boolean")
    by_id = {row["exception_id"]: row for row in nodes}
    for row in nodes:
        for name in ("cause_id", "context_id"):
            _need(row[name] is None or type(row[name]) is str and row[name] in by_id, "dangling exception link")
    _need(type(r["ordered_cleanup_failures"]) is list, "cleanup array")
    for ordinal, row in enumerate(r["ordered_cleanup_failures"]):
        _object(row, "ordinal stage exception_id")
        _need(type(row["ordinal"]) is int and row["ordinal"] == ordinal, "cleanup ordinal")
        _need(row["stage"] == "close" and row["exception_id"] in by_id and row["exception_id"] != "e0", "cleanup identity/stage")
    visited = []
    for root in ["e0"] + [row["exception_id"] for row in r["ordered_cleanup_failures"]]:
        pending = [root]
        while pending:
            name = pending.pop()
            if name is None or name in visited:
                continue
            visited.append(name)
            pending.extend((by_id[name]["context_id"], by_id[name]["cause_id"]))
    _need(visited == list(by_id), "graph reachability/encounter order")
    _array(r["capture_failures"], _failure)
    if any(row[field]["state"] != "complete" for row in nodes for field in ("message_bytes", "traceback_bytes")):
        _need(bool(r["capture_failures"]), "missing capture diagnostics")


def _tensor_hashes(value):
    _object(value, FAMILIES)
    for digest in value.values():
        _sha(digest)


def _aggregate_shards(rows, total):
    counts = {key: sum(row["sample_counts"][key] for row in rows) for key in COUNTS}
    maxima = {key: max(float.fromhex(row["maxima"][key]) for row in rows).hex() for key in MAXIMA}
    _need(all(count <= total for count in counts.values()), "aggregate count")
    return counts, maxima


def validate_scientific_closure(record: Mapping) -> None:
    r = _header(record, "scientific_closure")
    _need((r["binding"]["state"], r["phase"]) in (("state5", "state5_members"), ("state6", "state6_manifests")), "scientific stage")
    _capture(r["capture"], r["binding"])
    _ref(r["accepted_controls_projection"])
    _ref(r["source_closure_file"])
    final = r["phase"] == "state6_manifests"
    if final:
        _ref(r["context_file"])
    else:
        _need(r["context_file"] is None and r["ordered_splits"] == [], "State5 cannot predict final records/A6")
    rows = r["ordered_shards"]
    _need(type(rows) is list and len(rows) == 76, "scientific shard population")
    _refs(r["failure_files"])
    for (split, index), row in zip(_shard_rows(), rows):
        _object(row, "split shard_index member_files controls_manifest_file controls_jsonl_file original_rho_origin_file shard_tensor_sha256 verification_record_commitment shard_manifest_file sample_counts maxima evidence_file")
        _need(row["split"] == split and type(row["shard_index"]) is int and row["shard_index"] == index, "scientific order")
        _refs(row["member_files"], ordered=False)
        _need(len(row["member_files"]) == 8, "eight scientific members")
        base = "corpus_staging/" + _directory(split) + f"/shard_{index:04d}/"
        names = [family + ".float64.npy" for family in FAMILIES] + ["verification.jsonl"]
        _need([f["relative_path"] for f in row["member_files"]] == [base + name for name in names], "scientific member paths")
        for name in ("controls_manifest_file", "controls_jsonl_file", "original_rho_origin_file", "evidence_file"):
            _ref(row[name])
        _need(row["controls_manifest_file"]["relative_path"] == _control_base(split, index) + ".manifest.json", "control manifest path")
        _need(row["controls_jsonl_file"]["relative_path"] == _control_base(split, index) + ".controls.jsonl", "controls JSONL path")
        _tensor_hashes(row["shard_tensor_sha256"])
        _sha(row["verification_record_commitment"])
        _need(row["verification_record_commitment"] == row["member_files"][-1]["file_sha256"], "verification file commitment")
        if final:
            _ref(row["shard_manifest_file"])
            _need(row["shard_manifest_file"]["relative_path"] == base + "shard_manifest.json", "final manifest path")
        else:
            _need(row["shard_manifest_file"] is None, "future State5 manifest")
        _sample_counts(row["sample_counts"], 64)
        _maxima(row["maxima"])
    if final:
        _need(type(r["ordered_splits"]) is list and len(r["ordered_splits"]) == 3, "three final splits")
        for split, count, row in zip(SPLITS, SHARDS, r["ordered_splits"]):
            _object(row, "split sample_count shard_count split_manifest_file split_tensor_sha256 sample_counts maxima sealed")
            _need(row["split"] == split and type(row["sample_count"]) is int and row["sample_count"] == count * 64
                  and type(row["shard_count"]) is int and row["shard_count"] == count and row["sealed"] is (split == "test"), "split population")
            _ref(row["split_manifest_file"])
            _need(row["split_manifest_file"]["relative_path"] == "corpus_staging/" + _directory(split) + "/split_manifest.json", "split path")
            _tensor_hashes(row["split_tensor_sha256"])
            _sample_counts(row["sample_counts"], count * 64)
            _maxima(row["maxima"])
            expected_counts, expected_maxima = _aggregate_shards([item for item in rows if item["split"] == split], count * 64)
            _need(row["sample_counts"] == expected_counts and row["maxima"] == expected_maxima, "split arithmetic")
    aggregate = r["aggregate"]
    _object(aggregate, "sample_counts shard_count split_count maxima")
    _need(type(aggregate["shard_count"]) is int and aggregate["shard_count"] == 76 and
          type(aggregate["split_count"]) is int and aggregate["split_count"] == 3, "aggregate populations")
    _sample_counts(aggregate["sample_counts"], 4864)
    _maxima(aggregate["maxima"])
    expected_counts, expected_maxima = _aggregate_shards(rows, 4864)
    _need(aggregate["sample_counts"] == expected_counts and aggregate["maxima"] == expected_maxima, "aggregate arithmetic")
    if not r["failure_files"]:
        _need(all(value == 4864 for value in aggregate["sample_counts"].values()), "incomplete successful science metadata")
        for name, encoded in aggregate["maxima"].items():
            limit = 0.02543117962436324 if name == "delta_L2" else 0.022582030716092032 if name == "delta_J" else (
                100 * 2.220446049250313e-16 if name in ("T64_boundary_transmissibility_error", "P256_boundary_reconstruction_error") else 1e-10)
            _need(float.fromhex(encoded) <= limit, "gated maximum")


def _checks(value, detailed=False):
    _object(value, CHECKS)
    for row in value.values():
        if detailed:
            _object(row, "status evidence_files")
            _refs(row["evidence_files"], nonempty=row["status"] == "matched")
            row = row["status"]
        _enum(row, ("matched", "mismatch", "unavailable", "invalid"))


def _evidence_row(row):
    _object(row, "role file schema validation_profile")
    _text(row["role"])
    _ref(row["file"])
    if row["schema"] is not None:
        _text(row["schema"])
    _enum(row["validation_profile"], ("document_full_bytes_v1", "existing_state3_archive_v1", "existing_state4_opaque_v1",
          "existing_state4_seed_free_v1", "existing_runtime_v2", "existing_permission_v1", "existing_publication_v1",
          "new_final_evidence_v1", "raw_captured_bytes_v1", "source_identity_v1"))


def validate_final_audit_details(record: Mapping) -> None:
    r = _header(record, "final_audit_details")
    _local(r["corpus_manifest"], "corpus_manifest.json", "corpus_manifest.v1")
    _prefix(r["prefix"], "audit_publication")
    _ref(r["scientific_closure"])
    _refs(r["state_execution_files"], ordered=False)
    _need(len(r["state_execution_files"]) == 3, "three state executions")
    _array(r["access_extension"], _access)
    _array(r["failure_extension"], _failure_event)
    _array(r["post_move_publication_files"], _evidence_row)
    roles = ("finalize_metadata_result", "pre_move_observation", "promote_intent", "promote_result",
             "post_move_observation", "post_move_comparison", "post_move_comparison_details")
    _need([row["role"] for row in r["post_move_publication_files"]] == list(roles), "post-move evidence roles")
    _ref(r["retention_inventory_file"])
    _checks(r["checks"], True)
    _array(r["unavailable_evidence"], _missing, ordered=True)


def validate_final_audit(record: Mapping) -> None:
    r = _header(record, "final_audit")
    _local(r["corpus_manifest"], "corpus_manifest.json", "corpus_manifest.v1")
    _local(r["parent_context"], "final_context.json", "final_context.v1")
    _ref(r["details_file"])
    _need(r["details_file"]["relative_path"].endswith("test.sealed/final_audit_details.json"), "sealed audit details")
    _prefix(r["prefix"], "audit_publication")
    _checks(r["check_status"])
    _enum(r["publication_location"], ("T", "F", "Q", "unknown"))
    if all(value == "matched" for value in r["check_status"].values()):
        _need(r["publication_location"] == "F", "successful audit location")
    _need(type(r["split_summary"]) is list and len(r["split_summary"]) == 3, "audit splits")
    for split, count, row in zip(SPLITS, SHARDS, r["split_summary"]):
        _object(row, "split sample_count shard_count sample_counts maxima split_file_commitment sealed")
        _need(row["split"] == split and type(row["sample_count"]) is int and row["sample_count"] == count * 64
              and type(row["shard_count"]) is int and row["shard_count"] == count and row["sealed"] is (split == "test"), "audit split identity")
        _sample_counts(row["sample_counts"], count * 64)
        _maxima(row["maxima"])
        _sha(row["split_file_commitment"])


def validate_corpus_provenance(record: Mapping) -> None:
    r = _header(record, "corpus_provenance")
    _local(r["parent_context"], "final_context.json", "final_context.v1")
    _local(r["rights_record"], "generated_data_rights.json", "generated_data_rights.v1")
    _document(r["protocol_binding"], "protocol")
    _object(r["authorization_bindings"], "state3 state4 state5 state6")
    for state, value in r["authorization_bindings"].items():
        _document(value, state + "_authorization")
    authorizations = [r["authorization_bindings"][state]["file"]["sha256"] for state in ("state4", "state5", "state6")]
    _need(len(set(authorizations)) == 3, "distinct stage authorization files")
    _object(r["baseline_binding"], "commit tree gate_files")
    _need(r["baseline_binding"]["commit"] == "e16456395cfe67b73b9d298fd0491cf9524d798a" and
          r["baseline_binding"]["tree"] == "81f40e24102e51e0da3ce65dafd769a7678565fd", "frozen baseline")
    _gates(r["baseline_binding"]["gate_files"])
    _object(r["implementation_binding"], "identity source_manifest_file git_before git_after")
    _implementation(r["implementation_binding"]["identity"])
    _ref(r["implementation_binding"]["source_manifest_file"])
    for name in ("git_before", "git_after"):
        _git(r["implementation_binding"][name])
    _source_bindings(r["source_and_dependency_bindings"])
    _environments(r["environment_and_platform_bindings"])
    _gates(r["gate_bindings"])
    _need(r["gate_bindings"] == r["baseline_binding"]["gate_files"], "Gate binding disagreement")
    _need(r["implementation_binding"]["source_manifest_file"] == r["source_and_dependency_bindings"]["source_manifest_file"], "source manifest disagreement")
    _object(r["execution_evidence"], "state4 state5 state6")
    for value in r["execution_evidence"].values():
        _ref(value)
    _sealed_local(r["failure_inventory"], "corpus_failure_inventory")
    _sealed_local(r["access_ledger"], "corpus_access_ledger")
    _need(r["legacy_data_or_model_output_used"] is False, "legacy/model output")


def validate_corpus_manifest(record: Mapping) -> None:
    r = _header(record, "corpus_manifest")
    _local(r["parent_context"], "final_context.json", "final_context.v1")
    _sha(r["protocol_file_sha256"])
    _object(r["authorization_file_sha256_by_state"], "state4 state5 state6")
    for value in r["authorization_file_sha256_by_state"].values():
        _sha(value)
    _need(len(set(r["authorization_file_sha256_by_state"].values())) == 3, "authorization collision")
    _attempt(r["attempt_id"], r["authorization_file_sha256_by_state"]["state4"])
    _need(r["solver_spec_baseline_commit"] == "e16456395cfe67b73b9d298fd0491cf9524d798a" and
          r["solver_spec_baseline_tree"] == "81f40e24102e51e0da3ce65dafd769a7678565fd", "frozen baseline")
    _sha(r["implementation_commit"], 40)
    _sha(r["implementation_tree"], 40)
    _object(r["environment_and_dependency_bindings"], "environment_and_platform source_and_dependency")
    _environments(r["environment_and_dependency_bindings"]["environment_and_platform"])
    _source_bindings(r["environment_and_dependency_bindings"]["source_and_dependency"])
    _gates(r["gate_bindings"])
    _need(type(r["ordered_split_manifests"]) is list and len(r["ordered_split_manifests"]) == 3, "three final split refs")
    for split, value in zip(SPLITS, r["ordered_split_manifests"]):
        _local(value, "corpus_staging/" + _directory(split) + "/split_manifest.json", "final_split_manifest.v1")
    _local(r["rights_record"], "generated_data_rights.json", "generated_data_rights.v1")
    _sealed_local(r["provenance_record"], "corpus_provenance")
    _sealed_local(r["failure_inventory"], "corpus_failure_inventory")
    _local(r["controls_pre_solve_manifest"], "state4_controls/pre_solve_manifest.json", "state4_pre_solve_manifest.v2")
    _local(r["controls_completion"], "STATE4_CONTROLS_COMPLETE.json", "state4_controls_complete.v1")


def validate_corpus_completion(record: Mapping) -> None:
    r = _header(record, "corpus_complete")
    _need(r["state"] == "accepted_private_corpus_no_training_authorized", "completion state")
    _need(r["corpus_manifest_filename"] == "corpus_manifest.json", "completion filename")
    _sha(r["corpus_manifest_file_sha256"])
    _utc(r["completion_time_utc"])


_VALIDATORS = {
    "accepted_controls_projection": validate_accepted_controls_projection,
    "state_guard_closure": validate_state_guard_closure,
    "process_lifecycle_event": validate_process_lifecycle_event,
    "state_process_outcome": validate_state_process_outcome,
    "state_execution_evidence": validate_state_execution_evidence,
    "corpus_failure_inventory": validate_corpus_failure_inventory,
    "corpus_access_ledger": validate_corpus_access_ledger,
    "writer_exception_graph": validate_writer_exception_graph,
    "scientific_closure": validate_scientific_closure,
    "final_audit_details": validate_final_audit_details,
    "final_audit": validate_final_audit,
    "corpus_provenance": validate_corpus_provenance,
    "corpus_manifest": validate_corpus_manifest,
    "corpus_complete": validate_corpus_completion,
}


def _public_validator(function):
    @wraps(function)
    def validated(record: Mapping) -> None:
        try:
            function(record)
        except FinalEvidenceRecordError:
            raise
        except (TypeError, ValueError, KeyError, OverflowError, UnicodeError) as exc:
            raise FinalEvidenceRecordError("invalid record field representation") from exc
    return validated


for _kind, _validator in tuple(_VALIDATORS.items()):
    _wrapped = _public_validator(_validator)
    globals()[_validator.__name__] = _wrapped
    _VALIDATORS[_kind] = _wrapped


def _dispatch(record):
    _need(type(record) is dict, "record object required")
    schema = record.get("schema")
    matches = [kind for kind in SCHEMAS if schema == PREFIX + kind + ".v1" or
               kind in V2_EXECUTION_KINDS and schema == 'operatormark.darcy_branch_c_v2.'+kind+'.v1']
    _need(len(matches) == 1, "unsupported schema")
    _VALIDATORS[matches[0]](record)


def build_final_evidence_record(kind: str, payload: Mapping) -> dict:
    _need(type(kind) is str and kind in SCHEMAS, "unsupported kind")
    value = _plain(payload)
    field = "manifest_self_sha256" if kind == "corpus_manifest" else "record_self_sha256"
    _object(value, "schema " + SCHEMAS[kind])
    prefix = execution_prefix(value['binding']) if kind in V2_EXECUTION_KINDS else PREFIX
    _need(value["schema"] == prefix + kind + ".v1", "kind/schema mismatch")
    try:
        value[field] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
        _VALIDATORS[kind](value)
    except (ValueError, TypeError, UnicodeError, OverflowError) as exc:
        raise FinalEvidenceRecordError("invalid record representation") from exc
    return value


def strict_final_evidence_loads(data: bytes) -> dict:
    _need(type(data) is bytes, "immutable bytes required")
    try:
        value = strict_canonical_json_loads(data)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise FinalEvidenceRecordError("canonical JSON required") from exc
    _dispatch(value)
    return value


def canonical_final_evidence_bytes(record: Mapping) -> bytes:
    value = _plain(record)
    _dispatch(value)
    try:
        return canonical_json_bytes(value)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise FinalEvidenceRecordError("canonical encoding") from exc
