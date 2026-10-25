"""E2 conditional publication ports; no production activation or ACL setter."""
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from . import publication_physical_records as records
from . import publication_physical_verify as verifier
from . import state_evidence_windows as evidence
from . import state_provider_bindings as p


@dataclass(frozen=True, slots=True)
class _PublicationGrant:
    step: str
    intent_write: evidence._RetentionRequest
    result_write: evidence._RetentionRequest
    source_file: p.FileRef
    enrollment_file: p.FileRef
    authority_file: p.FileRef
    capture: p.Capture


@dataclass(frozen=True, slots=True)
class _PublicationConfiguration:
    expected: verifier.PublicationExpectations
    grants: tuple
    read_requests: tuple
    entry_files: tuple
    source_file: p.FileRef
    enrollment_file: p.FileRef
    protection_source_file: p.FileRef | None


@dataclass(frozen=True, slots=True)
class _PublicationBoundary:
    acquire_scope: object
    prepare: object
    acquire_phase: object
    consume_intent: object
    authenticate_intent: object
    perform: object
    release_owned: object
    protection: object | None


@dataclass(frozen=True, slots=True)
class _ScopeAcquisition:
    status: str
    role: str
    evidence_file: p.FileRef | None
    source_file: p.FileRef
    enrollment_file: p.FileRef
    capture: p.Capture


def _envelopes(rows):
    evidence._exact(rows, tuple)
    seen = set()
    for row in rows:
        evidence._exact(row, MappingProxyType)
        value = verifier._plain(row)
        verifier._obj(value, 'role file data')
        verifier._text(value['role'])
        verifier._ref(value['file'], positive=False)
        evidence._need(value['data'] is None or type(value['data']) is bytes, 'immutable independently acquired bytes')
        if value['data'] is not None:
            evidence._need((len(value['data']), p.sha(value['data'])) ==
                           (value['file']['file_bytes'], value['file']['file_sha256']), 'read envelope content binding')
        evidence._need(value['role'] not in seen, 'duplicate independent role')
        seen.add(value['role'])


def validate_publication_provider_configuration(configuration, *, expected):
    evidence._exact(configuration, _PublicationConfiguration)
    verifier.validate_publication_expectations(expected)
    evidence._need(configuration.expected == expected, 'independently accepted publication expectations substituted')
    _envelopes(configuration.entry_files)
    p.validate_file_ref(configuration.source_file)
    p.validate_file_ref(configuration.enrollment_file)
    accepted = verifier._plain(expected.provider_bindings)
    operator = next(x for x in accepted if x['role'] == 'operator')
    evidence._need(p.wire(configuration.source_file) == operator['source_file'] and
                   p.wire(configuration.enrollment_file) == operator['enrollment_file'], 'source/enrollment binding')
    evidence._exact(configuration.grants, tuple)
    evidence._need(tuple(g.step for g in configuration.grants) == records._STEPS, 'exact operation grant inventory/order')
    for grant in configuration.grants:
        evidence._exact(grant, _PublicationGrant)
        p.validate_capture(grant.capture)
        for file in (grant.source_file, grant.enrollment_file, grant.authority_file):
            p.validate_file_ref(file)
        evidence._need(grant.source_file == configuration.source_file and
                       grant.enrollment_file == configuration.enrollment_file and
                       p.wire(grant.authority_file) == operator['authority_scope_file'], 'operation role/source/authority')
        for product in (grant.intent_write, grant.result_write):
            evidence._exact(product, evidence._RetentionRequest)
            evidence._need(product.component == 'publication', 'publication catalog component')
    evidence._exact(configuration.read_requests, tuple)
    # Actual bootstrap verifies the plan, root binding, source/enrollment and
    # complete retention template. A favorable producer configuration is insufficient.
    bootstrap = verifier._bootstrap_entry(expected, configuration.entry_files)
    for grant in configuration.grants:
        for kind, request in (('intent', grant.intent_write), ('result', grant.result_write)):
            path = bootstrap.plan['publication_evidence_root_relative_path'] + '/test.sealed/operations/' + grant.step + '.' + kind + '.json'
            actual = verifier._resolve_publication_address(p.wire(request.final.address), bootstrap.roots, repository_only=True)
            evidence._need(actual == ('repository', path), 'plan-derived intent/result destination')
    if configuration.protection_source_file is not None:
        p.validate_file_ref(configuration.protection_source_file)
    return bootstrap


@dataclass(frozen=True, slots=True)
class _IntentOwnership:
    grant: _PublicationGrant
    intent_bytes: bytes
    intent_file: p.FileRef
    original_owner: evidence._Ownership


def validate_publication_dispatch(intent_bytes, *, retained_intent, ownership, expected):
    evidence._exact(intent_bytes, bytes)
    evidence._exact(retained_intent, bytes)
    evidence._exact(ownership, _IntentOwnership)
    evidence._ownership(ownership.original_owner)
    row = records.strict_publication_record_loads(intent_bytes,
        expected_schema='operatormark.darcy_branch_c_v1.publication_operation_intent.v1')
    evidence._need(row['binding'] == verifier._plain(expected.binding) and row['step'] == ownership.grant.step,
                   'intent invocation/step binding')
    evidence._need(intent_bytes == retained_intent == ownership.intent_bytes and
                   (len(intent_bytes), p.sha(intent_bytes)) ==
                   (ownership.intent_file.file_bytes, ownership.intent_file.file_sha256), 'prepared intent substitution')
    evidence._need((ownership.intent_file.root_binding_id, ownership.intent_file.relative_path) ==
                   (ownership.grant.intent_write.final.address.root_binding_id,
                    ownership.grant.intent_write.final.address.relative_path), 'intent slot binding')
    evidence._need(ownership.original_owner.reservation == ownership.intent_file,
                   'original exclusive intent creator required')
    return row


@dataclass(frozen=True, slots=True)
class _ProtectionRequest:
    object_identity: evidence._NativeIdentity
    phase: str
    source_file: p.FileRef
    authority_file: p.FileRef
    before_security_file: p.FileRef
    after_security_file: p.FileRef
    grant: _PublicationGrant
    intent_bytes: bytes


@dataclass(frozen=True, slots=True)
class _ProtectionResult:
    request: _ProtectionRequest
    before_identity: evidence._NativeIdentity
    after_identity: evidence._NativeIdentity
    before_security_file: p.FileRef | None
    after_security_file: p.FileRef | None
    result_bytes: bytes


@dataclass(frozen=True, slots=True)
class _ProtectionDelivery:
    request: _ProtectionRequest
    result: _ProtectionResult
    independent_files: tuple


def validate_protection_result(result, *, request, independent_files):
    evidence._exact(request, _ProtectionRequest)
    evidence._exact(result, _ProtectionResult)
    evidence._need(result.request == request and request.phase == 'post_complete' and
                   request.grant.step == 'finalize_protection', 'protection object/phase/source scope')
    evidence._identity(request.object_identity)
    evidence._need(result.before_identity == result.after_identity == request.object_identity,
                   'protection object replaced')
    evidence._need((result.before_security_file, result.after_security_file) ==
                   (request.before_security_file, request.after_security_file), 'before/after security evidence required')
    _envelopes(independent_files)
    for file in (request.source_file, request.authority_file, request.before_security_file, request.after_security_file):
        p.validate_file_ref(file)
        matches = [row for row in independent_files if row['file'] == records._freeze(p.wire(file)) and row['data'] is not None]
        evidence._need(bool(matches), 'independent protection security/authority/source acquisition unavailable')
    row = records.strict_publication_record_loads(result.result_bytes,
        expected_schema='operatormark.darcy_branch_c_v1.publication_operation_result.v1')
    evidence._need(row['step'] == 'finalize_protection', 'protection operation binding')
    evidence._need(row['call_status'] == 'returned_success' and row['effect_status'] == 'expected_effect_observed'
                   and not row['failures'], 'protection acknowledgment without qualified effects')
    for key, security in (('before_observation_file',request.before_security_file),('after_observation_file',request.after_security_file)):
        matches = [item for item in independent_files if item['file'] == records._freeze(row[key]) and item['data'] is not None]
        evidence._need(len(matches) == 1, 'independent before/after physical protection observations required')
        observed = records.strict_publication_record_loads(matches[0]['data'],expected_schema='operatormark.darcy_branch_c_v1.publication_physical_observation.v1')
        evidence._need(observed['binding'] == row['binding'], 'protection observation binding')
        objects = [item for item in observed['objects'] if item['native_identity'] == p.wire(request.object_identity)]
        evidence._need(len(objects) == 1 and objects[0]['presence'] == 'present' and objects[0]['security_file'] == p.wire(security),
                       'independent held protection identity/security correspondence')
        if key == 'after_observation_file': evidence._need(observed['phase'] == request.phase, 'protection phase correspondence')
    return row


class WindowsPublicationPorts:
    def __init__(self, *, configuration, expected, reader, writer, boundary):
        self.bootstrap = validate_publication_provider_configuration(configuration, expected=expected)
        evidence._exact(reader, evidence.WindowsEvidenceReader)
        evidence._exact(writer, evidence.WindowsEvidenceWriter)
        evidence._exact(boundary, _PublicationBoundary)
        for function in (boundary.acquire_scope, boundary.prepare, boundary.acquire_phase, boundary.consume_intent,
                         boundary.authenticate_intent, boundary.perform, boundary.release_owned):
            evidence._function(function)
        if boundary.protection is not None:
            evidence._function(boundary.protection)
        self.configuration, self.expected = configuration, expected
        self.reader, self.writer, self.boundary = reader, writer, boundary
        self._owners = {}
        self._spent = set()
        self._lock = RLock()
        self.retention = []
        self.failures = []
        self.observations = []

    def _grant(self, step):
        grants = [g for g in self.configuration.grants if g.step == step]
        evidence._need(len(grants) == 1, 'undeclared publication operation')
        return grants[0]

    def check_scope(self, checkpoint):
        evidence._need(type(checkpoint) is str, 'scope checkpoint')
        result = self.boundary.acquire_scope(checkpoint, self.configuration)
        evidence._exact(result, _ScopeAcquisition)
        verifier._enum(result.status, ('active', 'refused', 'unavailable'))
        p.validate_capture(result.capture)
        evidence._need(result.role == 'operator' and result.source_file == self.configuration.source_file and
                       result.enrollment_file == self.configuration.enrollment_file, 'scope role/source/enrollment')
        evidence._need(all(p.within(result.capture, grant.capture) for grant in self.configuration.grants),
                       'scope outside accepted operation interval')
        if result.evidence_file is not None:
            acquired = self.read_retained(records._freeze(p.wire(result.evidence_file)))
            evidence._need(bool(acquired) and all(row['data'] is not None for row in acquired), 'scope unavailable')
        evidence._need(result.status != 'active' or result.evidence_file is not None, 'scope acknowledgment alone')
        return records._freeze(dict(status=result.status, evidence_file=p.wire(result.evidence_file)))

    def prepare(self, step, evidence_files):
        grant = self._grant(step)
        _envelopes(evidence_files)
        result = self.boundary.prepare(grant, evidence_files)
        evidence._exact(result, MappingProxyType)
        row = verifier._plain(result)
        verifier._obj(row, 'prepared_files independent_files')
        _envelopes(result['independent_files'])
        role = {'finalize_metadata':'state6_metadata', 'publish_audit':'final_audit', 'publish_complete':'completion'}.get(step)
        wanted = [item['address'] for item in verifier._plain(self.expected.object_requirements) if item['role'] == role]
        evidence._need(len(row['prepared_files']) == len(wanted), 'prepared population required by accepted objects')
        destinations = []
        for item in row['prepared_files']:
            verifier._obj(item, 'destination file data')
            verifier._address(item['destination'])
            verifier._ref(item['file'], positive=False)
            evidence._need(type(item['data']) is bytes and (len(item['data']), p.sha(item['data'])) ==
                           (item['file']['file_bytes'], item['file']['file_sha256']), 'prepared bytes binding')
            independently_read = self.read_retained(records._freeze(item['file']))
            evidence._need(any(x['data'] == item['data'] and x['file'] == records._freeze(item['file'])
                              for x in independently_read), 'prepared bytes need independent read')
            verifier._resolve_publication_address(item['destination'], self.bootstrap.roots, repository_only=True)
            destinations.append(item['destination'])
        evidence._need(destinations == wanted, 'independently accepted preparation destinations')
        return result

    def _retain(self, step, data, is_result):
        grant = self._grant(step)
        schema = 'publication_operation_result' if is_result else 'publication_operation_intent'
        row = records.strict_publication_record_loads(data,
            expected_schema='operatormark.darcy_branch_c_v1.' + schema + '.v1')
        evidence._need(row['step'] == step and row['binding'] == verifier._plain(self.expected.binding), 'retained operation binding')
        request = grant.result_write if is_result else grant.intent_write
        outcome = self.writer.retain_exclusive(request, data)
        self.retention.append(outcome)
        file = p.FileRef(request.final.address.root_binding_id, request.final.address.relative_path, len(data), p.sha(data))
        published = outcome.publication
        good = (outcome.creation_status == 'created' and outcome.flushed and outcome.held_readback == data and
                published is not None and published.call_status == 'returned_success' and
                published.effect_status == 'expected_effect_observed' and not outcome.failures and not outcome.cleanup)
        if good and not is_result:
            evidence._need(outcome.ownership is not None, 'original exclusive intent ownership unavailable')
            ownership = _IntentOwnership(grant, data, file, outcome.ownership)
            with self._lock:
                evidence._need(step not in self._owners and step not in self._spent, 'existing intent cannot resume')
                self._owners[step] = ownership
        return records._freeze(dict(call_status='returned_success' if good else 'outcome_unavailable',
                                    file=p.wire(file) if good else None,
                                    failure=None if good else dict(kind='retention_failed', native_error_hex=None,
                                                                  exception_file=None, affected_addresses=[p.wire(request.final.address)])))

    def retain_intent(self, step, data):
        return self._retain(step, data, False)

    def retain_result(self, step, data):
        return self._retain(step, data, True)

    def read_retained(self, file_ref):
        evidence._exact(file_ref, MappingProxyType)
        reference = verifier._plain(file_ref)
        verifier._ref(reference, positive=False)
        matches = [r for r in self.configuration.read_requests if
                   (r.address.root_binding_id, r.address.relative_path) ==
                   (reference['root_binding_id'], reference['relative_path'])]
        evidence._need(len(matches) == 1, 'publication independently accepted read scope unavailable')
        acquired = self.reader.acquire(matches[0])
        evidence._need(not acquired.diagnostics and acquired.data is not None, 'publication readback incomplete')
        evidence._need((len(acquired.data), p.sha(acquired.data)) ==
                       (reference['file_bytes'], reference['file_sha256']), 'publication readback content substitution')
        return (records._freeze(dict(role=matches[0].role, file=reference, data=acquired.data)),)

    def _dispatch(self, step, intent_bytes, release=False):
        scope = self.check_scope('native_dispatch')
        evidence._need(scope['status'] == 'active', 'active operation scope required at dispatch')
        with self._lock:
            evidence._need(step in self._owners and step not in self._spent, 'missing original intent ownership or replay')
            owner = self._owners[step]
            acquired = self.read_retained(records._freeze(p.wire(owner.intent_file)))
            row = validate_publication_dispatch(intent_bytes, retained_intent=acquired[0]['data'],
                                                 ownership=owner, expected=self.expected)
            evidence._need(self.boundary.authenticate_intent(owner) is owner, 'durable original ownership unavailable')
            self._spent.add(step)
            evidence._need(self.boundary.consume_intent(owner) is owner, 'intent consumption acknowledgment lost')
        operation = self.boundary.release_owned if release else self.boundary.perform
        if step == 'finalize_protection':
            evidence._need(self.boundary.protection is not None and self.configuration.protection_source_file is not None,
                           'qualified protection dependency unavailable; no native protection implemented')
            operation = self.boundary.protection
        try:
            result = operation(owner.grant, row, self.configuration)
            if step == 'finalize_protection':
                evidence._exact(result, _ProtectionDelivery)
                evidence._need(result.request.grant == owner.grant and result.request.intent_bytes == intent_bytes and
                               result.request.source_file == self.configuration.protection_source_file and
                               result.request.authority_file == owner.grant.authority_file,
                               'qualified protection dependency binding')
                protection_record = validate_protection_result(result.result, request=result.request,
                                                               independent_files=result.independent_files)
                result = records._freeze(dict(result=protection_record, independent_files=result.independent_files))
            evidence._exact(result, MappingProxyType)
            body = verifier._plain(result)
            verifier._obj(body, 'result independent_files')
            _envelopes(result['independent_files'])
            records.validate_publication_operation_result(body['result'])
            capture = body['result']['capture']
            observed_capture = p.Capture(*(capture[k] for k in ('clock_epoch_id','first_sequence','last_sequence','started_utc','finished_utc')))
            p.validate_capture(observed_capture)
            evidence._need(p.within(observed_capture,owner.grant.capture), 'returned operation outside independently accepted interval')
            evidence._need(body['result']['intent_file'] == p.wire(owner.intent_file) and
                           body['result']['step'] == step and body['result']['binding'] == verifier._plain(self.expected.binding),
                           'operation result scope mismatch')
            return result
        except BaseException as exc:
            self.failures.append(exc)
            raise

    def perform(self, step, intent_bytes):
        evidence._need(step != 'release_resources', 'release uses its exact port')
        return self._dispatch(step, intent_bytes)

    def observe(self, phase):
        evidence._need(phase in records._PHASES, 'physical observation phase')
        result = self.boundary.acquire_phase(phase, self.configuration)
        _envelopes(result)
        for row in result:
            if row['data'] is not None:
                reference = verifier._plain(row['file'])
                acquired = self.read_retained(row['file'])
                evidence._need(any(x['file'] == row['file'] and x['data'] == row['data'] for x in acquired),
                               'observation requires independent held acquisition')
        self.observations.append((phase, result))
        return result

    def release_resources(self, intent_bytes):
        return self._dispatch('release_resources', intent_bytes, True)
