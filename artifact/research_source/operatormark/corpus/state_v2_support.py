"""V2 State-adapter metadata and Drop 2 development writer/readback seams.

Independent accepted expectations remain the caller's obligation. Supplied
hashes are correspondence checks, not authentication or scientific acceptance.
Native use requires the versioned custody route and independently accepted native
prerequisites; metadata compatibility alone never enables execution.
"""
from dataclasses import dataclass
import hashlib
import re

from .corpus_v2_schema import V2Layout, inventory_sha256
from .corpus_retention import derive_v2_retention_plan, validate_v2_tensor_slots
from .source_binding import ExecutionSourceExpectations, VerifiedSourceSet, reverify_source_set
from .writer_sizing import derive_v2_writer_sizes, derive_v2_drop2_accounting


@dataclass(frozen=True, slots=True)
class V2AdapterExpectations:
    layout: V2Layout
    inventory_sha256: str
    protocol_sha256: str
    numerical_settings_sha256: str
    transfer_sha256: str
    fixture_results_sha256: str
    source: ExecutionSourceExpectations
    runtime_record_sha256: str
    runtime_inspector_file_sha256: str


@dataclass(frozen=True, slots=True)
class V2AdapterInputs:
    expected: V2AdapterExpectations
    source: VerifiedSourceSet
    retention: object
    writer_sizes: object
    metadata_compatible: bool = True
    native_readiness: bool = False
    drop2_writer_accounting: object = None


def prepare_v2_adapter_inputs(*, expected, source, runtime_record, tensor_slots) -> V2AdapterInputs:
    """Bind the common inventory to independently acquired source/runtime inputs.

    The runtime validator is pure metadata validation. Never call the live
    inspector from this seam or infer qualification from synthetic success.
    """
    if type(expected) is not V2AdapterExpectations:
        raise TypeError('independent exact v2 expectations required')
    for name in ('inventory_sha256', 'protocol_sha256', 'numerical_settings_sha256',
                 'transfer_sha256', 'fixture_results_sha256', 'runtime_record_sha256',
                 'runtime_inspector_file_sha256'):
        value = getattr(expected, name)
        if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None or value == '0'*64:
            raise ValueError('missing/invalid binding: ' + name)
    if expected.inventory_sha256 != inventory_sha256(expected.layout):
        raise ValueError('inventory/layout substitution')
    if type(expected.source) is not ExecutionSourceExpectations or expected.source.protocol_sha256 != expected.protocol_sha256:
        raise ValueError('protocol/source substitution')
    verified = reverify_source_set(source, expected=expected.source)
    inspector = tuple(row for row in verified.sources if row.path == 'operatormark/corpus/runtime.py')
    if len(inspector) != 1 or hashlib.sha256(inspector[0].data).hexdigest() != expected.runtime_inspector_file_sha256:
        raise ValueError('runtime inspector absent from verified source closure or drifted')
    from .runtime import validate_numerical_runtime_record
    validate_numerical_runtime_record(runtime_record)
    if runtime_record['runtime_record_sha256'] != expected.runtime_record_sha256:
        raise ValueError('runtime record substitution')
    if runtime_record['all_runtime_requirements_satisfied'] is not True:
        raise ValueError('runtime requirements unresolved')
    validate_v2_tensor_slots(expected.layout, tensor_slots)
    return V2AdapterInputs(expected, verified, derive_v2_retention_plan(expected.layout),
                           derive_v2_writer_sizes(expected.layout),
                           drop2_writer_accounting=derive_v2_drop2_accounting(expected.layout))


def require_v2_native_pipeline(inputs, *, prerequisites=None, **pipeline_inputs):
    """Recompute the versioned State 5/6 gate; never mint authority from metadata."""
    if type(inputs) is not V2AdapterInputs:
        raise TypeError('validated v2 adapter inputs required')
    if prerequisites is None:
        raise RuntimeError('State 5/6 native prerequisites missing: independently accepted qualification, '
            'State 4 controls, native source/runtime/observer qualification and complete prospective storage')
    from .state_v2_prerequisites import NativePrerequisites, verify_native_prerequisites
    if type(prerequisites) is not NativePrerequisites or prerequisites.adapter != inputs:
        raise ValueError('v2 prerequisite/adapter substitution')
    return verify_native_prerequisites(prerequisites, **pipeline_inputs)


def generate_v2_development(*, layout, config, controls, directions, output, solver):
    """Actual Drop 2 writer seam, explicitly synthetic; never an actual State run.

    The caller chooses resolution, retention, populations and scientific proposals.
    This API grants no numerical execution and supplies no default solver.
    """
    from operatormark_corpus_v2.config import GenerationConfig
    from operatormark_corpus_v2.generator import generate
    if type(layout) is not V2Layout or type(config) is not GenerationConfig:
        raise TypeError('exact v2 layout/config required')
    if config.mode != 'synthetic_test' or not config.layout.synthetic:
        raise ValueError('actual native State execution is not integrated or authorized')
    if (config.n_ref, config.retention.option) != (layout.n_ref, layout.retention):
        raise ValueError('writer adapter layout substitution')
    derive_v2_drop2_accounting(layout)
    return generate(config, controls, directions, output, solver=solver)


def readback_v2_development(*, layout, root, expected, numerics, p1):
    """Independent Drop 2 readback with caller-provided expectations/options."""
    from operatormark_corpus_v2.readback import Expected, validate_corpus
    if type(layout) is not V2Layout or type(expected) is not Expected:
        raise TypeError('exact v2 layout/readback expectations required')
    if expected.mode != 'synthetic_test':
        raise ValueError('native State acceptance unavailable')
    if (expected.n_ref, expected.retention) != (layout.n_ref, layout.retention):
        raise ValueError('readback adapter layout substitution')
    return validate_corpus(root, expected, numerics=numerics, p1=p1)
