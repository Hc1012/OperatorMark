"""Conditional retained-file accounting, without a production promotion gate."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
import hashlib
import re
from types import MappingProxyType
import unicodedata

_CONTROL_OR_SURROGATE = re.compile(r"[\x00-\x1f\x7f-\x9f\ud800-\udfff]")

from operatormark.corpus.canonical import canonical_json_bytes, strict_canonical_json_loads


class RetentionContractError(ValueError):
    pass


PREFIX = "operatormark.darcy_branch_c_v1."
COMPONENTS = ("state4_core", "state5_core", "final_records", "stage_execution", "publication",
              "permission_observation", "permission_authz", "permission_handoff", "authority_journals",
              "administration_restart", "state3_run", "retained_history")
CLASSIFICATIONS = ("exact", "conditional_exact", "upper_bound", "unresolved")
DOMAIN_REASONS = ("source_missing", "population_missing", "acquisition_domain_missing",
                  "retention_policy_missing", "upstream_admission_missing", "source_binding_missing")
MISSING_REASONS = ("not_captured", "bytes_unavailable", "outcome_unavailable", "source_rule_unavailable", "custody_unavailable") + DOMAIN_REASONS
CUTOFFS = ("entry", "state4_closed", "state5_closed", "pre_manifest", "pre_move", "post_move", "post_audit",
           "post_complete", "pre_terminal", "assessment_after_terminal", "failed")
FAMILY_SIDES = {"rho": 8, "kappa64": 64, "u_tpfa64": 64, "kappa32_areaavg": 32,
                "u32_areaavg": 32, "kappa16_areaavg": 16, "u16_areaavg": 16}
SPLIT_ROWS = (("training", 64), ("validation", 4), ("test", 8))
ZIP_NAMES = ("state3_audit.json", "environment_runtime.json", "storage_preflight_before.json",
             "compute_feasibility_report.json", "storage_preflight_after.json", "source_blob_verification.json",
             "failure_inventory.json", "prohibited_action_counts.json", "member_hash_manifest.json")
RAW_TENSOR_BYTES = 420_872_192


@dataclass(frozen=True, slots=True)
class V2RetentionPlan:
    """Versioned integration input, not a v1 catalog or permission to delete."""
    inventory_sha256: str
    stored_families: tuple[str, ...]
    hash_only_families: tuple[str, ...]
    tensor_slots: tuple[tuple[str, int, str, tuple[int, ...]], ...]
    state6_obligations: tuple[str, ...]
    native_readiness: bool = False


def derive_v2_retention_plan(layout) -> V2RetentionPlan:
    from .corpus_v2_schema import tensor_inventory, inventory_sha256
    from .corpus_format import SPLITS, SPLIT_SHARD_COUNTS
    families = tensor_inventory(layout)
    slots = tuple((split, shard, family.name, family.shard_shape(split))
                  for split in SPLITS for shard in range(SPLIT_SHARD_COUNTS[split])
                  for family in families if family.stored)
    native = ('native_forward_and_all_tangents_resolve_reusing_factorization',
              'native_resolve_hashes_bitwise_equal_state5') if layout.retention == 'E' else (
              'read_stored_native_forward_and_all_tangents',
              'native_assembly_and_matvec_without_factorization')
    return V2RetentionPlan(inventory_sha256(layout),
        tuple(f.name for f in families if f.stored), tuple(f.name for f in families if not f.stored), slots,
        native + ('native_gates_and_tangent_residual_recomputed',
                  'native_restrictions_bitwise_equal_stored_u_f64_and_du_f64',
                  'native_kappa_reconstructed_and_hash_bound',
                  'hash_only_native_records_include_float_hex_gates_settings_and_transfer',
                  'independent_P256_resolve_bitwise_bound_to_state5',
                  'unchanged_structural_derived_view_and_P1_gates',
                  'same_frozen_platform_across_states3_to6'))


def validate_v2_tensor_slots(layout, slots) -> None:
    """Exact inventory/shape/order/multiplicity check, before physical paths bind.

    Drop 2 must separately bind its serializer filenames and verify actual bytes.
    This function does not admit a caller's file list as retained evidence.
    """
    expected = derive_v2_retention_plan(layout).tensor_slots
    if (type(slots) is not tuple or any(type(row) is not tuple or len(row) != 4
            or type(row[0]) is not str or type(row[1]) is not int or type(row[2]) is not str
            or type(row[3]) is not tuple or len(row[3]) != 4
            or any(type(n) is not int for n in row[3]) for row in slots) or slots != expected):
        raise RetentionContractError('v2 stored tensor membership/shape/order mismatch')


def _require(condition, message):
    if not condition:
        raise RetentionContractError(message)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_plain(item) for item in value]
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class RetentionDerivation:
    catalog_file: object
    snapshot_files: object
    component_results: object
    observed_retained_bytes: object
    observed_namespace_bytes: object
    projected_H: object
    projected_J: object
    projected_M: object
    controls_C: object
    separate_B: object
    unresolved_requirements: object
    cutoff_unresolved_requirements: object
    required_cutoff_products_missing: object
    latest_cutoff: str
    production_disk_estimate_available: bool

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))
        _require(self.production_disk_estimate_available is False, "production availability forbidden")


def _obj(value, keys):
    _require(type(value) is dict and set(value) == set(keys.split()), "exact keys")


def _number(value, minimum=0):
    _require(type(value) is int and value >= minimum, "exact integer")


def _hash(value, width=64):
    _require(type(value) is str and re.fullmatch("[0-9a-f]{" + str(width) + "}", value) is not None
             and value != "0" * width, "nonzero hash")


def _string(value):
    _require(type(value) is str and bool(value) and unicodedata.normalize("NFC", value) == value
             and ((value.isascii() and value.isprintable()) or
                  _CONTROL_OR_SURROGATE.search(value) is None), "NFC identifier")


def _path(value, empty=False):
    if empty and value == "":
        return
    _string(value)
    for part in value.split("/"):
        _require(part not in ("", ".", "..") and part[-1] not in ". " and
                 not any(c in part for c in '\\:*?"<>|'), "path alias")
        _require(re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])", part.split(".")[0]) is None, "reserved path")


def _token(value, allowed):
    _require(type(value) is str and value in allowed, "closed token")


def _address(value):
    _obj(value, "root_binding_id relative_path")
    _string(value["root_binding_id"])
    _require(":" not in value["root_binding_id"], "root colon")
    _path(value["relative_path"], True)


def _file(value, nullable=False):
    if value is None and nullable:
        return
    _obj(value, "root_binding_id relative_path file_bytes file_sha256")
    _address({key: value[key] for key in ("root_binding_id", "relative_path")})
    _path(value["relative_path"])
    _number(value["file_bytes"])
    _hash(value["file_sha256"])


def _list(value, validator, sorted_unique=False):
    _require(type(value) is list, "array required")
    for row in value:
        validator(row)
    if sorted_unique:
        keys = [canonical_json_bytes(row) for row in value]
        _require(keys == sorted(set(keys)), "canonical unique array")


def _files(value):
    _list(value, _file, True)


def _missing(value):
    _obj(value, "role expected_file reason")
    _string(value["role"])
    _file(value["expected_file"], True)
    _token(value["reason"], MISSING_REASONS)


def _missing_rows(value):
    _list(value, _missing, True)


def _identity(value, nullable=False):
    if value is None and nullable:
        return
    _obj(value, "volume_serial_number_hex file_id_128_hex legacy_volume_serial_number_hex legacy_file_index_hex")
    volume = value["volume_serial_number_hex"]
    _require(type(volume) is str and re.fullmatch("0x(0|[1-9a-f][0-9a-f]*)", volume) is not None
             and int(volume, 16) < 2**64, "native volume")
    for key, width in (("file_id_128_hex", 32), ("legacy_volume_serial_number_hex", 8), ("legacy_file_index_hex", 16)):
        _require(type(value[key]) is str and re.fullmatch("[0-9a-f]{" + str(width) + "}", value[key]) is not None, "identity field")


def _capture(value):
    _obj(value, "invocation_id clock_epoch_id first_sequence last_sequence started_utc finished_utc")
    _hash(value["invocation_id"])
    _string(value["clock_epoch_id"])
    _number(value["first_sequence"])
    _number(value["last_sequence"], value["first_sequence"])
    for key in ("started_utc", "finished_utc"):
        _require(type(value[key]) is str and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", value[key]) is not None, "capture UTC")
        try:
            datetime.strptime(value[key], "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError as exc:
            raise RetentionContractError("capture calendar") from exc
    _require(value["started_utc"] <= value["finished_utc"], "capture order")


def _roots(value):
    def root(row):
        _obj(row, "root_binding_id repository_relative_root identity_file custody_file")
        _string(row["root_binding_id"])
        _require(":" not in row["root_binding_id"], "root colon")
        _path(row["repository_relative_root"])
        _file(row["identity_file"])
        _file(row["custody_file"])
    _list(value, root)
    _require(len({r["root_binding_id"] for r in value}) == len(value), "duplicate root ID")
    return {r["root_binding_id"]: r["repository_relative_root"] for r in value}


def _resolve(address, roots):
    _address(address)
    _require(address["root_binding_id"] in roots, "unbound root")
    return roots[address["root_binding_id"]] + ("/" + address["relative_path"] if address["relative_path"] else "")


def _attempt(value):
    _obj(value, "attempt_id state4_authorization_file_sha256")
    _hash(value["state4_authorization_file_sha256"])
    _require(value["attempt_id"] == "s4-" + value["state4_authorization_file_sha256"][:24], "attempt binding")


def _implementation(value):
    _obj(value, "commit tree sole_parent")
    for item in value.values():
        _hash(item, 40)


def _domain(value):
    _obj(value, "classification rule input_files parameters prerequisite_files")
    _token(value["classification"], CLASSIFICATIONS)
    _files(value["input_files"])
    _files(value["prerequisite_files"])
    rule = value["rule"]
    parameters = value["parameters"]
    table = {
        "supplied_complete_bytes": ("", "exact"),
        "qualified_length_observation": ("observation_file", "conditional_exact"),
        "frozen_tensor_payload": ("family shard_count", "exact"),
        "canonical_npy_header": ("family shard_count", "conditional_exact"),
        "controls_jsonl_schedule": ("split shard_index", "conditional_exact"),
        "verification_jsonl_domain": ("split shard_index", "upper_bound"),
        "state3_classic_zip": ("ordered_member_files", "conditional_exact"),
        "unresolved": ("reason required_evidence_roles", "unresolved"),
    }
    _require(type(rule) is str and rule in table, "domain rule")
    keys, classification = table[rule]
    _obj(parameters, keys)
    _require(value["classification"] == classification, "domain classification")
    if classification in ("conditional_exact", "upper_bound"):
        _require(bool(value["prerequisite_files"]), "conditional rule prerequisites")
    if rule in ("frozen_tensor_payload", "canonical_npy_header"):
        _token(parameters["family"], FAMILY_SIDES)
        _number(parameters["shard_count"], 1)
    elif rule in ("controls_jsonl_schedule", "verification_jsonl_domain"):
        _token(parameters["split"], dict(SPLIT_ROWS))
        _number(parameters["shard_index"])
        _require(parameters["shard_index"] < dict(SPLIT_ROWS)[parameters["split"]], "domain shard")
    elif rule == "qualified_length_observation":
        _file(parameters["observation_file"])
    elif rule == "state3_classic_zip":
        _list(parameters["ordered_member_files"], _file)
        _require([r["relative_path"] for r in parameters["ordered_member_files"]] == list(ZIP_NAMES), "nine ZIP members")
    elif rule == "unresolved":
        _token(parameters["reason"], DOMAIN_REASONS)
        _list(parameters["required_evidence_roles"], _string)
        _require(parameters["required_evidence_roles"] == sorted(set(parameters["required_evidence_roles"])), "unresolved roles")


def _writer(value):
    _obj(value, "source_file function source_closure_file write_policy")
    _file(value["source_file"])
    _file(value["source_closure_file"])
    _require(type(value["function"]) is str and re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", value["function"]) is not None, "qualified writer name")
    _token(value["write_policy"], ("canonical_exclusive_preserve_v1", "component_existing_contract", "raw_supervisor_capture"))


def _slot(value):
    _obj(value, "slot_id component address writer representation owner controls_subset condition creating_address content_profile domain")
    _token(value["component"], COMPONENTS)
    _address(value["address"])
    _path(value["address"]["relative_path"])
    _require(value["slot_id"] == value["component"] + ":" + value["address"]["root_binding_id"] + ":" +
             value["address"]["relative_path"], "slot ID")
    _writer(value["writer"])
    _token(value["representation"], ("loose_file", "container", "reference_only"))
    _token(value["owner"], ("accepted_attempt", "separate_B", "external_preexisting", "unresolved"))
    _require(type(value["controls_subset"]) is bool and
             (not value["controls_subset"] or value["owner"] == "accepted_attempt"), "controls subtotal owner")
    condition = value["condition"]
    _obj(condition, "kind source_file selector")
    _token(condition["kind"], ("always", "stage_reached", "phase_reached", "operation_reached", "failure_observed",
                              "historical_instance", "terminal_pending", "assessment_after_terminal"))
    _file(condition["source_file"])
    _string(condition["selector"])
    if value["creating_address"] is not None:
        _address(value["creating_address"])
        if value["writer"]["write_policy"] == "canonical_exclusive_preserve_v1":
            _require(value["creating_address"] == dict(root_binding_id=value["address"]["root_binding_id"],
                     relative_path=value["address"]["relative_path"] + ".__creating__"), "canonical creating alias")
    elif value["writer"]["write_policy"] == "canonical_exclusive_preserve_v1" and value["representation"] != "reference_only":
        raise RetentionContractError("missing canonical alias")
    profile = value["content_profile"]
    _require(type(profile) is str and (re.fullmatch(r"operatormark\.darcy_branch_c_v1\.[a-z0-9_]+\.v[12]", profile) is not None or
             profile in ("canonical_npy_v2", "controls_jsonl_v1", "verification_jsonl_v1", "state3_zip_v1", "raw_bytes")), "content profile")
    _domain(value["domain"])
    if profile == "canonical_npy_v2":
        _require(value["domain"]["rule"] in ("canonical_npy_header", "unresolved"), "NPY file domain")
        if value["domain"]["rule"] == "canonical_npy_header":
            _require(value["domain"]["parameters"]["shard_count"] == 1, "one file header")
    _require(value["domain"]["rule"] != "frozen_tensor_payload", "payload is not a retained headerless file")


def _registration(value):
    _obj(value, "component source_files plan_files admission_files expansion slot_ids unresolved_requirements")
    _token(value["component"], COMPONENTS)
    for name in ("source_files", "plan_files", "admission_files"):
        _files(value[name])
    _token(value["expansion"], ("fixed_protocol", "bound_instance", "unresolved"))
    _list(value["slot_ids"], _string)
    _require(value["slot_ids"] == sorted(set(value["slot_ids"])), "slot ID order")
    _missing_rows(value["unresolved_requirements"])
    if value["expansion"] == "unresolved":
        _require(bool(value["unresolved_requirements"]), "unexplained unresolved component")


def _header(record, kind, keys):
    value = _plain(record)
    _obj(value, "schema " + keys + " record_self_sha256")
    _require(value["schema"] == PREFIX + kind + ".v1", "retention schema")
    _hash(value["record_self_sha256"])
    digest = hashlib.sha256(canonical_json_bytes({k: v for k, v in value.items() if k != "record_self_sha256"})).hexdigest()
    _require(value["record_self_sha256"] == digest, "retention self hash")
    return value


def validate_corpus_writer_catalog(record: Mapping) -> None:
    r = _header(record, "corpus_writer_catalog", "profile protocol_file_sha256 implementation attempt_binding root_requirements component_registrations slots unresolved_requirements")
    _require(r["profile"] == "private_corpus_retention_v1", "catalog profile")
    _hash(r["protocol_file_sha256"])
    _implementation(r["implementation"])
    _attempt(r["attempt_binding"])
    roots = _roots(r["root_requirements"])
    _list(r["component_registrations"], _registration)
    _require([row["component"] for row in r["component_registrations"]] == list(COMPONENTS), "twelve components")
    _list(r["slots"], _slot)
    locations = [_resolve(row["address"], roots) for row in r["slots"]]
    _require(len(set(path.casefold() for path in locations)) == len(locations), "duplicate physical destination")
    keys = [(path, row["slot_id"]) for path, row in zip(locations, r["slots"])]
    _require(keys == sorted(keys), "physical slot ordering")
    occupied = set(path.casefold() for path in locations)
    for row in r["slots"]:
        if row["creating_address"] is not None:
            path = _resolve(row["creating_address"], roots).casefold()
            _require(path not in occupied, "creating alias collision")
            occupied.add(path)
    for registration in r["component_registrations"]:
        ids = sorted(row["slot_id"] for row in r["slots"] if row["component"] == registration["component"])
        _require(registration["slot_ids"] == ids, "registration membership")
    expected_fixed = {path: (component, profile) for component, role, path, profile, split, index, family
                      in _fixed_population(r["attempt_binding"]["attempt_id"])}
    actual_fixed = {_resolve(row["address"], roots):
                    (row["component"], row["content_profile"], row["owner"], row["representation"])
                    for row in r["slots"]}
    _require(all(actual_fixed.get(path) == (*values, "accepted_attempt", "loose_file")
                 for path, values in expected_fixed.items()),
             "fixed population omission or ownership/representation substitution")
    _missing_rows(r["unresolved_requirements"])


def validate_corpus_retention_snapshot(record: Mapping) -> None:
    r = _header(record, "corpus_retention_snapshot", "catalog_file capture cutoff prior_snapshot_file snapshot_address observations unavailable_requirements")
    _file(r["catalog_file"])
    _capture(r["capture"])
    _token(r["cutoff"], CUTOFFS)
    _file(r["prior_snapshot_file"], True)
    _address(r["snapshot_address"])
    def observation(row):
        _obj(row, "slot_id address presence content_state file native_identity length_evidence_file origin_file capture_file")
        _string(row["slot_id"])
        _address(row["address"])
        _token(row["presence"], ("present", "absent", "unknown"))
        _token(row["content_state"], ("complete", "partial", "unavailable", "not_reached"))
        _file(row["file"], True)
        _identity(row["native_identity"], True)
        for name in ("length_evidence_file", "origin_file", "capture_file"):
            _file(row[name], True)
        if row["file"] is not None:
            _require(row["presence"] == "present" and row["content_state"] in ("complete", "partial"), "file/presence contradiction")
            _require({k: row["file"][k] for k in ("root_binding_id", "relative_path")} == row["address"], "file address mismatch")
        if row["content_state"] == "complete":
            _require(row["presence"] == "present" and row["file"] is not None, "complete file evidence")
        if row["presence"] == "absent":
            _require(row["file"] is None and row["native_identity"] is None and row["capture_file"] is not None, "qualified absence required")
        if row["content_state"] == "not_reached":
            _require(row["presence"] == "absent" and row["capture_file"] is not None, "not-reached evidence")
        _require(row["address"] != r["snapshot_address"], "snapshot cannot observe its own future bytes")
    _list(r["observations"], observation)
    keys = [(row["slot_id"], canonical_json_bytes(row["address"])) for row in r["observations"]]
    _require(keys == sorted(set(keys)), "observation ordering/duplicates")
    _missing_rows(r["unavailable_requirements"])


def canonical_retention_bytes(record: Mapping) -> bytes:
    value = _plain(record)
    if value.get("schema") == PREFIX + "corpus_writer_catalog.v1":
        validate_corpus_writer_catalog(value)
    elif value.get("schema") == PREFIX + "corpus_retention_snapshot.v1":
        validate_corpus_retention_snapshot(value)
    else:
        raise RetentionContractError("unsupported retention schema")
    return canonical_json_bytes(value)


def strict_retention_loads(data: bytes) -> dict:
    _require(type(data) is bytes, "immutable bytes")
    try:
        value = strict_canonical_json_loads(data)
        canonical_retention_bytes(value)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise RetentionContractError("invalid retention bytes") from exc
    return value


def _fixed_population(attempt):
    """Literal protocol populations; the role is the fixed writer population."""
    d = "artifacts/darcy_branch_c_v1/private_corpus_v1/"
    t = d + "attempts/" + attempt + "/"
    f = d + "corpus/"
    l = d + "publication_evidence/" + attempt + "/"
    rows = []
    for split, count in SPLIT_ROWS:
        directory = "test.sealed" if split == "test" else split
        controls = t + "state4_controls/" + directory + "/"
        final = t + "corpus_staging/" + directory + "/"
        for index in range(count):
            for family in FAMILY_SIDES:
                component = "state4_core" if family == "rho" else "state5_core"
                rows.append((component, component + "/" + family, final + f"shard_{index:04d}/" + family +
                             ".float64.npy", "canonical_npy_v2", split, index, family))
            rows.extend((
                ("state4_core", "state4_core/controls_jsonl", controls + f"shard_{index:04d}.controls.jsonl", "controls_jsonl_v1", split, index, None),
                ("state4_core", "state4_core/controls_manifest", controls + f"shard_{index:04d}.manifest.json", PREFIX + "state4_controls_shard_manifest.v1", split, index, None),
                ("state5_core", "state5_core/verification_jsonl", final + f"shard_{index:04d}/verification.jsonl", "verification_jsonl_v1", split, index, None),
                ("final_records", "final_records/shard_manifest", final + f"shard_{index:04d}/shard_manifest.json", PREFIX + "final_shard_manifest.v1", split, index, None),
            ))
        rows.extend((
            ("state4_core", "state4_core/split_manifest", controls + "split_manifest.json", "raw_bytes", split, None, None),
            ("final_records", "final_records/split_manifest", final + "split_manifest.json", PREFIX + "final_split_manifest.v1", split, None, None),
        ))
    for name, path, schema in (
        ("rng_audit", "state4_controls/rng_audit.json", "state4_rng_audit.v1"),
        ("read_only_evidence", "state4_controls/read_only_evidence.json", "state4_read_only_evidence.v1"),
        ("pre_solve_manifest", "state4_controls/pre_solve_manifest.json", "state4_pre_solve_manifest.v2"),
        ("controls_completion", "STATE4_CONTROLS_COMPLETE.json", "state4_controls_complete.v1"),
    ):
        rows.append(("state4_core", "state4_core/" + name, t + path, PREFIX + schema, None, None, None))
    for name, path, root, schema in (
        ("final_context", "final_context.json", t, "final_context"),
        ("generated_data_rights", "generated_data_rights.json", t, "generated_data_rights"),
        ("corpus_provenance", "corpus_staging/test.sealed/corpus_provenance.json", t, "corpus_provenance"),
        ("corpus_access", "corpus_staging/test.sealed/corpus_access_ledger.json", t, "corpus_access_ledger"),
        ("corpus_failures", "corpus_staging/test.sealed/corpus_failure_inventory.json", t, "corpus_failure_inventory"),
        ("corpus_manifest", "corpus_manifest.json", t, "corpus_manifest"),
        ("final_audit", "final_audit.json", f, "final_audit"),
        ("completion", "COMPLETE.json", f, "corpus_complete"),
    ):
        rows.append(("final_records", name, root + path, PREFIX + schema + ".v1", None, None, None))
    for state in ("state4", "state5"):
        e = d + "state_evidence/" + attempt + "/" + state + "/"
        products = {"guards": "state_guard_closure", "process": "state_process_outcome", "execution": "state_execution_evidence",
                    "process_creation": "process_lifecycle_event", "process_exit": "process_lifecycle_event",
                    "stdout": None, "stderr": None}
        products["accepted_controls" if state == "state4" else "scientific_closure"] = (
            "accepted_controls_projection" if state == "state4" else "scientific_closure")
        for name, schema in products.items():
            rows.append(("stage_execution", state + "/" + name,
                         e + "test.sealed/" + name + (".bin" if schema is None else ".json"),
                         "raw_bytes" if schema is None else PREFIX + schema + ".v1", None, None, None))
        for cutoff in ("entry", state + "_closed"):
            rows.append(("stage_execution", state + "/inventory/" + cutoff,
                         e + "test.sealed/inventories/" + cutoff + ".json", PREFIX + "corpus_retention_snapshot.v1", None, None, None))
        phase = state + "_closed"
        for sealed in (False, True):
            name = "details" if sealed else "report"
            rows.append(("stage_execution", phase + "/" + name,
                         e + ("test.sealed/" if sealed else "") + "reviews/" + phase + ".json",
                         PREFIX + ("final_evidence_details" if sealed else "final_evidence_comparison") + ".v1", None, None, None))
    for name, path, schema in (
        ("state6/guards", "execution/guards.json", "state_guard_closure"),
        ("state6/execution", "execution/state6.json", "state_execution_evidence"),
        ("state6/scientific_closure", "execution/scientific_closure.json", "scientific_closure"),
        ("state6/process_creation", "inputs/state6_child_creation.json", "process_lifecycle_event"),
        ("state6/process_exit", "inputs/state6_child_exit.json", "process_lifecycle_event"),
        ("final_audit_details", "final_audit_details.json", "final_audit_details"),
    ):
        component = "publication" if name == "final_audit_details" else "stage_execution"
        rows.append((component, name, l + "test.sealed/" + path, PREFIX + schema + ".v1", None, None, None))
    for cutoff in CUTOFFS[3:]:
        rows.append(("publication", "inventory/" + cutoff, l + "test.sealed/corpus_inventories/" + cutoff + ".json",
                     PREFIX + "corpus_retention_snapshot.v1", None, None, None))
    for phase in ("state6_child_closed", "pre_move", "post_move", "post_complete", "terminal"):
        for sealed in (False, True):
            rows.append(("publication", phase + ("/details" if sealed else "/report"),
                         l + ("test.sealed/" if sealed else "") + "final_evidence_reviews/" + phase + ".json",
                         PREFIX + ("final_evidence_details" if sealed else "final_evidence_comparison") + ".v1", None, None, None))
    return rows


def _bound_address(path, roots):
    choices = [(len(root), root_id, root) for root_id, root in roots.items()
               if path == root or path.startswith(root + "/")]
    _require(bool(choices), "fixed location lacks root binding")
    size = max(row[0] for row in choices)
    matches = [row for row in choices if row[0] == size]
    _require(len(matches) == 1, "ambiguous fixed root binding")
    _, root_id, root = matches[0]
    return dict(root_binding_id=root_id, relative_path=path[len(root):].lstrip("/"))


def derive_fixed_corpus_slots(*, attempt_binding: Mapping, root_requirements: tuple, writer_bindings: tuple) -> tuple:
    attempt = _plain(attempt_binding)
    _attempt(attempt)
    roots = _roots(_plain(root_requirements))
    population = _fixed_population(attempt["attempt_id"])
    bindings = _plain(writer_bindings)
    _require(type(bindings) is list, "writer bindings")
    by_role = {}
    for row in bindings:
        _obj(row, "role writer domain")
        _string(row["role"])
        _writer(row["writer"])
        _domain(row["domain"])
        _require(row["role"] not in by_role, "duplicate writer binding")
        by_role[row["role"]] = row
    _require(set(by_role) == {row[1] for row in population}, "complete fixed writer binding population required")
    slots = []
    for component, role, path, profile, split, index, family in population:
        binding = by_role[role]
        domain = _plain(binding["domain"])
        if domain["rule"] in ("controls_jsonl_schedule", "verification_jsonl_domain"):
            _require(split is not None and index is not None, "JSONL binding population")
            domain["parameters"] = dict(split=split, shard_index=index)
        elif domain["rule"] == "canonical_npy_header":
            _require(family == domain["parameters"]["family"] and domain["parameters"]["shard_count"] == 1, "family writer domain")
        address = _bound_address(path, roots)
        writer = _plain(binding["writer"])
        creating = (dict(root_binding_id=address["root_binding_id"], relative_path=address["relative_path"] + ".__creating__")
                    if writer["write_policy"] == "canonical_exclusive_preserve_v1" else None)
        stage = "state4" if component == "state4_core" or role.startswith("state4/") or role.startswith("state4_closed/") else (
            "state5" if component == "state5_core" or role.startswith("state5/") or role.startswith("state5_closed/") else "state6")
        condition = "assessment_after_terminal" if "assessment_after_terminal" in role else "failure_observed" if role.endswith("/failed") else "stage_reached"
        slots.append(dict(slot_id=component + ":" + address["root_binding_id"] + ":" + address["relative_path"],
            component=component, address=address, writer=writer, representation="loose_file", owner="accepted_attempt",
            controls_subset=stage == "state4", condition=dict(kind=condition, source_file=writer["source_closure_file"], selector=stage),
            creating_address=creating, content_profile=profile, domain=domain))
    slots.sort(key=lambda row: (_resolve(row["address"], roots), row["slot_id"]))
    for slot in slots:
        _slot(slot)
    return tuple(_freeze(row) for row in slots)


def build_corpus_writer_catalog(*, protocol_file_sha256: str, implementation: Mapping, attempt_binding: Mapping,
                               root_requirements: tuple, component_registrations: tuple, fixed_writer_bindings: tuple,
                               additional_slots: tuple, unresolved_requirements: tuple) -> dict:
    fixed = _plain(derive_fixed_corpus_slots(attempt_binding=attempt_binding, root_requirements=root_requirements,
                                            writer_bindings=fixed_writer_bindings))
    extra = _plain(additional_slots)
    _list(extra, _slot)
    registrations = _plain(component_registrations)
    _list(registrations, _registration)
    registered = {r["component"]: set(r["slot_ids"]) for r in registrations}
    for slot in extra:
        _require(slot["slot_id"] in registered.get(slot["component"], set()), "unregistered additional slot")
    roots = _roots(_plain(root_requirements))
    slots = fixed + extra
    slots.sort(key=lambda row: (_resolve(row["address"], roots), row["slot_id"]))
    result = dict(schema=PREFIX + "corpus_writer_catalog.v1", profile="private_corpus_retention_v1",
        protocol_file_sha256=protocol_file_sha256, implementation=_plain(implementation), attempt_binding=_plain(attempt_binding),
        root_requirements=_plain(root_requirements), component_registrations=registrations, slots=slots,
        unresolved_requirements=_plain(unresolved_requirements))
    result["record_self_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    validate_corpus_writer_catalog(result)
    return result


def _jsonl_length(split, local, verification=False):
    """Source-reviewed canonical encoding schedule; no RNG or scientific calls."""
    offset = {"training": 0, "validation": 4096, "test": 4352}[split]
    h = "0" * 64
    x = "-0x1.fffffffffffffp+1023"
    row = dict(benchmark="darcy_branch_c_v1", control_tensor_sha256=h, global_index=offset + local,
               local_index=local, macro_coefficient_tensor_sha256=h, record_self_sha256=h,
               rho_storage_tensor_sha256=h, sample_id=f"darcy_branch_c_v1/{split}/{local:06d}",
               shard_index=local // 64, split=split, stratum="relaxed" if local % 2 == 0 else "binary_32_of_64")
    if not verification:
        row["schema"] = PREFIX + "state4_control_record.v1"
    else:
        row.update(schema=PREFIX + "sample_verification.v1", tensor_sha256={name: h for name in FAMILY_SIDES},
            verifier_tensor_sha256={"P256": h, "bar_P256": h},
            gates={name: True for name in ("P1_256", "TPFA64", "cross_solver", "overall", "structural")},
            metrics=dict(compliances={"T64": x, "bar_P256": x}, delta_J=x, delta_L2=x, scales={"S_J": x, "S_L2": x}),
            solver_diagnostics={
                "P256": dict(algebraic_backward_error=x, boundary_reconstruction_error=x, integrated_source=x,
                    iterations=10000, outward_boundary_flux=x, passes=True, reaction_balance_ratio=x,
                    reaction_denominator=x, rhs_relative_l2_residual=x),
                "T64": dict(algebraic_backward_error=x, boundary_transmissibility_error=x, global_balance_ratio=x,
                    maximum_interior_face_antisymmetry=x, maximum_local_conservation_ratio=x, passes=True,
                    rhs_relative_l2_residual=x)})
    return len(canonical_json_bytes(row)) + 1


def _unique(values):
    return [row for _, row in sorted({canonical_json_bytes(row): row for row in values}.items())]


def _unavailable(role, reason, file=None):
    return dict(role=role, expected_file=file, reason=reason)


def _contribution(classification, count, prerequisites=(), missing=()):
    return dict(classification=classification, bytes=count, prerequisite_files=_unique(prerequisites),
                unresolved_requirements=_unique(missing))


def _combine(rows):
    missing = [item for row in rows for item in row["unresolved_requirements"]]
    prerequisites = [item for row in rows for item in row["prerequisite_files"]]
    classification = max((row["classification"] for row in rows), key=CLASSIFICATIONS.index, default="exact")
    count = None if any(row["bytes"] is None for row in rows) else sum(row["bytes"] for row in rows)
    if count is None:
        classification = "unresolved"
    return _contribution(classification, count, prerequisites, missing)


def _validated_admissions(rows):
    """Validate the unchanged UpstreamAdmission trusted-input grammar."""
    _require(type(rows) is tuple, "upstream admissions tuple")
    result = {}
    for row in _plain(rows):
        _obj(row, "role input_files validator_source_files admission_file scope capture status")
        _string(row["role"])
        _require(row["role"] not in result, "duplicate upstream admission role")
        _files(row["input_files"])
        _files(row["validator_source_files"])
        _require(bool(row["input_files"]) and bool(row["validator_source_files"]), "admission bindings required")
        _file(row["admission_file"])
        _token(row["scope"], ("state3_archive", "state4_acceptance", "runtime", "storage_state3_to_state4",
            "storage_later_phase", "permission_component", "scientific_members", "scientific_manifests",
            "publication_phase", "authority_and_custody", "source_closure"))
        _token(row["status"], ("accepted", "unavailable", "rejected"))
        _capture(row["capture"])
        result[row["role"]] = row
    return result


def _covers_capture(outer, inner):
    return (outer["invocation_id"] == inner["invocation_id"] and
            outer["clock_epoch_id"] == inner["clock_epoch_id"] and
            outer["first_sequence"] <= inner["first_sequence"] <= inner["last_sequence"] <= outer["last_sequence"] and
            outer["started_utc"] <= inner["started_utc"] <= inner["finished_utc"] <= outer["finished_utc"])


def _qualified_admission(admissions, *, evidence, scope, capture, source_files,
                         admission_files, available, role=None):
    """Match precise evidence/cutoff/source bindings, not an accepted label.

    False means no usable qualification. Explicit rejection of a matching
    premise is an error, not evidence of zero surviving bytes. An admission for
    another object, interval or scope cannot qualify this observation.
    """
    candidates = list(admissions.values()) if role is None else ([admissions[role]] if role in admissions else [])
    accepted = False
    for row in candidates:
        if (row["scope"] != scope or not _covers_capture(row["capture"], capture) or
                any(file is None or file not in row["input_files"] for file in evidence) or
                row["admission_file"] not in admission_files or
                any(file not in source_files for file in row["validator_source_files"])):
            continue
        if row["status"] == "rejected":
            raise RetentionContractError("upstream qualification rejected: " + row["role"])
        if row["status"] != "accepted":
            continue
        if all(available(file) for file in row["input_files"] + row["validator_source_files"] + [row["admission_file"]]):
            accepted = True
    return accepted


def derive_retention_components(*, catalog: Mapping, snapshots: tuple, independent_files: tuple,
                                admitted_length_observations: tuple,
                                upstream_admissions: tuple) -> RetentionDerivation:
    """Derive conditional accounting under explicit, independently trusted admissions.

    This table is supplied by the caller's qualified review boundary, never read
    from a candidate record. Its commitments are checked here; its authenticity
    is a premise, not a conclusion of metadata reconciliation.
    """
    _require(all(type(value) is tuple for value in (snapshots, independent_files,
             admitted_length_observations, upstream_admissions)), "tuple inputs required")
    c = _plain(catalog)
    validate_corpus_writer_catalog(c)
    roots = _roots(c["root_requirements"])
    supplied = _plain(independent_files)
    _require(type(supplied) is list, "independent file projections")
    by_file = {}
    for row in supplied:
        _obj(row, "role file data")
        _string(row["role"])
        _file(row["file"], True)
        _require(row["data"] is None or type(row["data"]) is bytes, "immutable independent bytes")
        if row["file"] is None:
            _require(row["data"] is None, "unbound independent bytes")
            continue
        key = canonical_json_bytes(row["file"])
        _require(key not in by_file or by_file[key]["data"] == row["data"], "contradictory independent file copies")
        if row["data"] is not None:
            _require(len(row["data"]) == row["file"]["file_bytes"] and
                     hashlib.sha256(row["data"]).hexdigest() == row["file"]["file_sha256"], "independent file hash/length")
        by_file[key] = row
    def available(file):
        row = by_file.get(canonical_json_bytes(file))
        return row is not None and row["data"] is not None
    upstream = _validated_admissions(upstream_admissions)
    registrations_by_component = {r["component"]: r for r in c["component_registrations"]}
    def qualify(*, evidence, scope, capture, slot, role=None):
        registration = registrations_by_component[slot["component"]]
        return _qualified_admission(upstream, evidence=evidence, scope=scope,
            capture=capture, source_files=registration["source_files"],
            admission_files=registration["admission_files"], available=available, role=role)
    def record_file(record, address=None):
        data = canonical_retention_bytes(record)
        matches = [row["file"] for row in supplied if row["data"] == data]
        if address is not None:
            # Content identifies bytes, not the independently read object. A copy
            # cannot stand in for the slot excluded from its own inventory.
            matches = [file for file in matches if
                       {k: file[k] for k in ("root_binding_id", "relative_path")} == address]
            _require(len(matches) == 1, "snapshot independent FileRef/address binding unavailable or ambiguous")
        _require(len(matches) == 1, "canonical record independent binding unavailable or ambiguous")
        return matches[0]
    catalog_file = record_file(c)
    history = _plain(snapshots)
    _require(type(history) is list and bool(history), "retention history required")
    history_files = []
    history_missing = []
    previous = None
    for snapshot in history:
        validate_corpus_retention_snapshot(snapshot)
        _require(snapshot["catalog_file"] == catalog_file, "snapshot changed catalog")
        historical_slots = {s["slot_id"]: s for s in c["slots"]}
        designated = [s for s in c["slots"] if snapshot["snapshot_address"] == s["address"]]
        _require(len(designated) == 1, "snapshot designated slot binding")
        snapshot_slot = designated[0]
        permitted = [(component, path) for component, role, path, profile, split, index, family
                     in _fixed_population(c["attempt_binding"]["attempt_id"])
                     if profile == PREFIX + "corpus_retention_snapshot.v1" and
                     role.endswith("inventory/" + snapshot["cutoff"])]
        _require((snapshot_slot["component"], _resolve(snapshot["snapshot_address"], roots)) in permitted and
                 snapshot_slot["content_profile"] == PREFIX + "corpus_retention_snapshot.v1" and
                 snapshot_slot["representation"] == "loose_file", "snapshot cutoff/slot binding")
        current = record_file(snapshot, snapshot["snapshot_address"])
        root = next(r for r in c["root_requirements"] if r["root_binding_id"] == current["root_binding_id"])
        if not qualify(evidence=[root["identity_file"], root["custody_file"], snapshot_slot["writer"]["source_closure_file"]],
                       scope="authority_and_custody", capture=snapshot["capture"], slot=snapshot_slot):
            history_missing.append(_unavailable("snapshot_root_binding", "custody_unavailable", root["identity_file"]))
        for observed in snapshot["observations"]:
            _require(observed["slot_id"] in historical_slots, "historical observation changed membership")
            historical_slot = historical_slots[observed["slot_id"]]
            if observed["address"] not in (historical_slot["address"], historical_slot["creating_address"]):
                base = "artifacts/darcy_branch_c_v1/private_corpus_v1/"
                prefix = base + "attempts/" + c["attempt_binding"]["attempt_id"] + "/"
                actual_path = _resolve(observed["address"], roots)
                _require(snapshot["cutoff"] in ("post_move", "post_audit", "post_complete", "pre_terminal", "assessment_after_terminal", "failed") and
                    any(old is not None and _resolve(old, roots).startswith(prefix) and
                        actual_path == base + "corpus/" + _resolve(old, roots)[len(prefix):]
                        for old in (historical_slot["address"], historical_slot["creating_address"])), "historical location mismatch")
                if not qualify(evidence=[observed["capture_file"], historical_slot["writer"]["source_closure_file"]],
                               scope="publication_phase", capture=snapshot["capture"], slot=historical_slot):
                    history_missing.append(_unavailable("historical_promotion", "custody_unavailable", observed["capture_file"]))
        _require(snapshot["prior_snapshot_file"] == (None if previous is None else history_files[-1]), "snapshot predecessor omitted")
        if previous is not None:
            left, right = previous["capture"], snapshot["capture"]
            if left["clock_epoch_id"] == right["clock_epoch_id"]:
                _require(right["first_sequence"] > left["last_sequence"], "snapshot history ordering")
            else:
                # A retained predecessor/read event, not integer or UTC ordering,
                # must bridge two independently identified capture epochs.
                snapshot_slot = next((s for s in c["slots"] if s["address"] == snapshot["snapshot_address"]), None)
                bridged = False
                if snapshot_slot is not None:
                    for projection in supplied:
                        if projection["role"] not in ("state4_execution", "state5_execution", "state6_execution") or projection["data"] is None:
                            continue
                        try:
                            event_record = strict_canonical_json_loads(projection["data"])
                        except (ValueError, TypeError, UnicodeError):
                            continue
                        if type(event_record) is not dict or event_record.get("schema") != PREFIX + "state_execution_evidence.v1":
                            continue
                        event_capture = event_record.get("capture")
                        try:
                            _capture(event_capture)
                        except RetentionContractError:
                            continue
                        if (event_capture["invocation_id"] != right["invocation_id"] or
                                event_capture["clock_epoch_id"] != right["clock_epoch_id"]):
                            continue
                        events = event_record.get("ordered_events")
                        if type(events) is not list:
                            continue
                        reads = [event for event in events if type(event) is dict and
                            set(event) == {"sequence", "kind", "source_file", "subject_file", "prior_event_file"} and
                            event["kind"] == "source_bound" and event["subject_file"] == history_files[-1] and
                            type(event["sequence"]) is int and
                            event_capture["first_sequence"] <= event["sequence"] <= event_capture["last_sequence"] and
                            event["sequence"] < right["first_sequence"]]
                        if reads and qualify(evidence=[projection["file"], history_files[-1],
                                snapshot_slot["writer"]["source_closure_file"]], scope="authority_and_custody",
                                capture=right, slot=snapshot_slot):
                            bridged = True
                if not bridged:
                    history_missing.append(_unavailable("snapshot_predecessor", "custody_unavailable", history_files[-1]))
        history_files.append(current)
        previous = snapshot
    slots = {row["slot_id"]: row for row in c["slots"]}
    slot_locations = {_resolve(row["address"], roots): row["slot_id"] for row in c["slots"]}
    admissions = {}
    for row in _plain(admitted_length_observations):
        _obj(row, "address native_identity byte_count observation_file custody_file capture upstream_admission_role")
        _address(row["address"])
        _identity(row["native_identity"])
        _number(row["byte_count"])
        _file(row["observation_file"])
        _file(row["custody_file"])
        _capture(row["capture"])
        _string(row["upstream_admission_role"])
        key = _resolve(row["address"], roots).casefold()
        _require(key not in admissions, "duplicate length admission")
        admissions[key] = row
    latest = history[-1]
    _require(_resolve(latest["snapshot_address"], roots) in slot_locations, "snapshot address not declared")
    missing = list(c["unresolved_requirements"]) + history_missing
    for historical in history:
        missing.extend(historical["unavailable_requirements"])
    observations = {}
    namespace_total = 0
    object_total = 0
    namespace_known = not history_missing
    objects_known = not history_missing
    counted_objects = {}
    for observation in latest["observations"]:
        sid = observation["slot_id"]
        _require(sid in slots, "unknown observed slot")
        slot = slots[sid]
        address = observation["address"]
        logical_address = address
        if address not in (slot["address"], slot["creating_address"]):
            # Promotion changes the location of the declared writer destination;
            # it does not create another writer slot or payload contribution.
            base = "artifacts/darcy_branch_c_v1/private_corpus_v1/"
            original_prefix = base + "attempts/" + c["attempt_binding"]["attempt_id"] + "/"
            actual = _resolve(address, roots)
            moved_from = [old for old in (slot["address"], slot["creating_address"]) if old is not None and
                _resolve(old, roots).startswith(original_prefix) and
                actual == base + "corpus/" + _resolve(old, roots)[len(original_prefix):]]
            _require(len(moved_from) == 1 and latest["cutoff"] in ("post_move", "post_audit", "post_complete",
                "pre_terminal", "assessment_after_terminal", "failed"), "undeclared observation alias")
            logical_address = moved_from[0]
            if not qualify(evidence=[observation["capture_file"], slot["writer"]["source_closure_file"]],
                           scope="publication_phase", capture=latest["capture"], slot=slot):
                namespace_known = objects_known = False
                missing.append(_unavailable(sid, "custody_unavailable", observation["capture_file"]))
        physical = _resolve(address, roots).casefold()
        logical = _resolve(logical_address, roots).casefold()
        _require(logical not in observations, "multiple current locations for one writer destination")
        observations[logical] = observation
        if observation["presence"] == "unknown":
            namespace_known = objects_known = False
            missing.append(_unavailable(sid, "outcome_unavailable", observation["capture_file"]))
            continue
        if observation["presence"] == "absent":
            evidence = [observation["capture_file"], slot["writer"]["source_closure_file"]]
            if observation["content_state"] == "not_reached":
                evidence.append(slot["condition"]["source_file"])
            qualified_absence = qualify(evidence=evidence, scope="authority_and_custody",
                                       capture=latest["capture"], slot=slot)
            if not qualified_absence:
                namespace_known = objects_known = False
                missing.append(_unavailable(sid, "upstream_admission_missing", observation["capture_file"]))
            continue
        file = observation["file"]
        admission = admissions.get(physical)
        length = file["file_bytes"] if file is not None and available(file) else None
        qualified = (admission is not None and
            admission["observation_file"] == observation["length_evidence_file"] and
            admission["custody_file"] == observation["origin_file"] and
            _covers_capture(latest["capture"], admission["capture"]) and
            qualify(evidence=[admission["observation_file"], admission["custody_file"],
                              observation["capture_file"], slot["writer"]["source_closure_file"]],
                    scope="authority_and_custody", capture=admission["capture"], slot=slot,
                    role=admission["upstream_admission_role"]))
        if qualified:
            _require(admission["native_identity"] == observation["native_identity"], "length identity mismatch")
            if length is not None:
                _require(length == admission["byte_count"], "independent length contradiction")
            length = admission["byte_count"]
        if length is None:
            namespace_known = objects_known = False
            missing.append(_unavailable(sid, "bytes_unavailable", observation["length_evidence_file"]))
            continue
        namespace_total += length
        if not qualified:
            objects_known = False
            missing.append(_unavailable(sid, "custody_unavailable", observation["origin_file"]))
            continue
        identity_key = canonical_json_bytes(admission["native_identity"])
        previous_object = counted_objects.get(identity_key)
        if previous_object is None:
            counted_objects[identity_key] = (sid, length, admission["custody_file"], address)
            object_total += length
        else:
            old_sid, old_length, custody, old_address = previous_object
            _require(old_sid == sid and old_address != address and slot["creating_address"] is not None,
                     "unexpected hard-link alias or duplicate ownership")
            if custody != admission["custody_file"]:
                objects_known = False
                missing.append(_unavailable(sid, "custody_unavailable", admission["custody_file"]))
            else:
                _require(old_length == length, "same object inconsistent lengths")
    for slot in c["slots"]:
        if slot["representation"] == "reference_only":
            continue
        for address in (slot["address"], slot["creating_address"]):
            if address is None or address == latest["snapshot_address"]:
                continue
            if _resolve(address, roots).casefold() not in observations:
                namespace_known = objects_known = False
                missing.append(_unavailable(slot["slot_id"], "not_captured"))
    # Capture integrity and custody failures before any prospective byte-domain
    # projections are evaluated.  A qualified absence has no observed bytes,
    # but the future writer's size may still be unresolved.  Unresolved catalog
    # registrations are custody/population gaps, never prospective size gaps.
    # The historical unresolved_requirements field below remains unchanged.
    registration_gaps = []
    for registration in (c["component_registrations"] if
                         latest["cutoff"] in ("pre_manifest", "pre_terminal") else ()):
        if (registration["expansion"] == "unresolved" or
                not registration["source_files"] or
                not registration["admission_files"] or
                registration["unresolved_requirements"]):
            registration_gaps.extend(registration["unresolved_requirements"] or
                [_unavailable(registration["component"], "upstream_admission_missing")])
        # A bound-instance component may legitimately have no output at this
        # cutoff.  Its empty slot list is not itself evidence that its writer
        # population is complete.  Require an independently retained admission
        # over the exact catalog and registered source/plan FileRefs, scoped to
        # this component and capture.  This is an explicit trusted population
        # premise; no unlisted output is silently inferred to have size zero.
        premise = [catalog_file] + registration["source_files"] + registration["plan_files"]
        if (not registration["plan_files"] or
                not all(available(file) for file in premise) or
                not _qualified_admission(upstream, evidence=premise,
                    scope="authority_and_custody", capture=latest["capture"],
                    source_files=registration["source_files"],
                    admission_files=registration["admission_files"],
                    available=available,
                    role="retention_population:" + registration["component"])):
            registration_gaps.append(_unavailable(
                registration["component"], "upstream_admission_missing"))
    owner_gaps = [_unavailable(slot["slot_id"], "retention_policy_missing")
                  for slot in c["slots"] if slot["owner"] == "unresolved"]
    cutoff_missing = _unique(missing + registration_gaps + owner_gaps)
    # State 6's pre-manifest cutoff precedes finalizer preparation and the
    # coordinator's writes.  Only the child-owned final_context, 76 shard and
    # three split manifests join the already-closed State 4/5 populations at
    # that cutoff.  The other seven final records must be present at the later
    # pre-terminal cutoff, after the coordinator's actual publication.
    child_final_roles = {"final_context", "final_records/shard_manifest",
                         "final_records/split_manifest"}
    finalizer_roles = {"generated_data_rights", "corpus_provenance", "corpus_access",
                       "corpus_failures", "corpus_manifest", "final_audit", "completion"}
    fixed = _fixed_population(c["attempt_binding"]["attempt_id"])
    required_paths = {path for component, role, path, _, _, _, _ in fixed
                      if component in ("state4_core", "state5_core") or
                      role in child_final_roles}
    if latest["cutoff"] == "pre_terminal":
        required_paths.update(path for _, role, path, _, _, _, _ in fixed
                              if role in finalizer_roles)
    required_products = [slot for slot in c["slots"]
                         if _resolve(slot["address"], roots) in required_paths]
    if latest["cutoff"] in ("pre_manifest", "pre_terminal"):
        expected_count = 854 if latest["cutoff"] == "pre_terminal" else 847
        _require(len(required_paths) == expected_count and
                 len(required_products) == expected_count and
                 {_resolve(slot["address"], roots) for slot in required_products} == required_paths and
                 all(slot["owner"] == "accepted_attempt" and slot["representation"] == "loose_file"
                     for slot in required_products), "fixed State 6 cutoff product population/ownership")
    # The observation index above keys a promoted F readback by the original
    # writer destination.  A .__creating__ alias has its own logical key and
    # cannot satisfy a required final product, even with complete bytes.
    complete_ids = set()
    for slot in required_products:
        row = observations.get(_resolve(slot["address"], roots).casefold())
        if (row is not None and row["slot_id"] == slot["slot_id"] and
                row["presence"] == "present" and row["content_state"] == "complete" and
                row["file"] is not None and row["native_identity"] is not None and
                available(row["file"])):
            complete_ids.add(slot["slot_id"])
    required_missing = _unique([_unavailable(slot["slot_id"], "not_captured")
                                for slot in required_products
                                if slot["slot_id"] not in complete_ids])
    buckets = {name: [] for name in ("H", "J", "M", "C", "B")}
    component_contributions = {name: [] for name in COMPONENTS}
    for slot in c["slots"]:
        domain = slot["domain"]
        params, rule = domain["parameters"], domain["rule"]
        prerequisites = domain["prerequisite_files"]
        absent = [_unavailable(slot["slot_id"], "bytes_unavailable", f) for f in prerequisites if not available(f)]
        count = None
        classification = domain["classification"]
        if rule == "unresolved":
            absent.extend(_unavailable(role, params["reason"]) for role in params["required_evidence_roles"])
        elif rule == "supplied_complete_bytes":
            files = domain["input_files"]
            if len(files) == 1 and available(files[0]):
                count = files[0]["file_bytes"]
            else:
                absent.append(_unavailable(slot["slot_id"], "bytes_unavailable"))
        elif rule == "qualified_length_observation":
            values = [row["byte_count"] for row in admissions.values()
                if row["observation_file"] == params["observation_file"] and
                row["address"] == slot["address"] and _covers_capture(latest["capture"], row["capture"]) and
                qualify(evidence=[row["observation_file"], row["custody_file"],
                                  slot["writer"]["source_closure_file"]],
                        scope="authority_and_custody", capture=row["capture"], slot=slot,
                        role=row["upstream_admission_role"])]
            if len(values) == 1:
                count = values[0]
            else:
                absent.append(_unavailable(slot["slot_id"], "upstream_admission_missing", params["observation_file"]))
        elif rule == "canonical_npy_header":
            side = FAMILY_SIDES[params["family"]]
            header = ("{'descr': '<f8', 'fortran_order': False, 'shape': " + str((64, 1, side, side)) + ", }").encode("ascii")
            unpadded = 12 + len(header) + 1
            count = unpadded + (-unpadded) % 64
        elif rule in ("controls_jsonl_schedule", "verification_jsonl_domain"):
            count = sum(_jsonl_length(params["split"], n, rule == "verification_jsonl_domain")
                        for n in range(64 * params["shard_index"], 64 * params["shard_index"] + 64))
        elif rule == "state3_classic_zip":
            files = params["ordered_member_files"]
            if all(available(file) for file in files):
                count = 1174 + sum(file["file_bytes"] for file in files)
            else:
                absent.append(_unavailable(slot["slot_id"], "bytes_unavailable"))
        if absent:
            count, classification = None, "unresolved"
        contribution = _contribution(classification, count, prerequisites, absent)
        if slot["representation"] == "reference_only":
            contribution = _contribution("exact", 0)
        component_value = contribution
        if slot["content_profile"] == "canonical_npy_v2" and count is not None:
            component_value = _contribution(classification, count + 64 * FAMILY_SIDES[params["family"]]**2 * 8,
                                            prerequisites, absent)
        component_contributions[slot["component"]].append(component_value)
        if slot["owner"] == "accepted_attempt":
            category = "H" if slot["content_profile"] == "canonical_npy_v2" else (
                "J" if slot["content_profile"] in ("controls_jsonl_v1", "verification_jsonl_v1") else "M")
            buckets[category].append(contribution)
            if slot["controls_subset"]:
                controls = contribution
                if slot["content_profile"] == "canonical_npy_v2" and count is not None:
                    controls = _contribution(classification, count + 64 * FAMILY_SIDES[params["family"]]**2 * 8, prerequisites)
                buckets["C"].append(controls)
        elif slot["owner"] == "separate_B":
            buckets["B"].append(contribution)
        elif slot["owner"] == "unresolved":
            missing.append(_unavailable(slot["slot_id"], "retention_policy_missing"))
    results = []
    for registration in c["component_registrations"]:
        component = registration["component"]
        terms = component_contributions[component]
        unresolved = registration["unresolved_requirements"]
        if registration["expansion"] == "unresolved" or not registration["source_files"] or not registration["admission_files"]:
            issues = unresolved or [_unavailable(component, "upstream_admission_missing")]
            terms = terms + [_contribution("unresolved", None, (), issues)]
            # Missing ownership/population cannot become zero in any affected tuple.
            for bucket in ("M", "C", "B"):
                buckets[bucket].append(_contribution("unresolved", None, (), issues))
        combined = _combine(terms)
        missing.extend(combined["unresolved_requirements"])
        results.append(dict(component=component, **combined, counted_slots=registration["slot_ids"]))
    combined = {key: _combine(values) for key, values in buckets.items()}
    missing = _unique(missing)
    return RetentionDerivation(catalog_file, history_files, results, object_total if objects_known else None,
        namespace_total if namespace_known else None, combined["H"], combined["J"], combined["M"], combined["C"],
        combined["B"], missing, cutoff_missing, required_missing, latest["cutoff"], False)
