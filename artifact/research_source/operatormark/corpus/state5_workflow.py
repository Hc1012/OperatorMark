"""Injected original-solve orchestration; P1 handles never enter retained products."""
from dataclasses import dataclass

from .state4_workflow import (WorkflowError, OriginalBinding, OperationReply, WorkflowResult,
    _entry, need, validate_workflow_inputs, close_execution, schedule)


@dataclass(frozen=True, slots=True)
class SyntheticSolve:
    original: OriginalBinding
    tpfa_token: bytes
    p1_token: bytes
    observation_tokens: tuple


@dataclass(frozen=True, slots=True)
class SyntheticScienceReply:
    solve: SyntheticSolve
    checked_tokens: tuple
    diagnostics: tuple
    simulation_only: bool = True
    numerical_qualification: bool = False


@dataclass(frozen=True, slots=True)
class State5Ports:
    acquire_original: object
    solve_original: object
    owns_ephemeral: object
    verify_original: object
    retain_member: object
    release_ephemeral: object
    finish: object


@dataclass(frozen=True, slots=True)
class SyntheticMemberEvidence:
    original: OriginalBinding
    tpfa_token: bytes
    observation_tokens: tuple
    verification_commitment: bytes
    simulation_only: bool = True


def validate_state5_inputs(*, expected, independent_inputs, mode):
    validate_workflow_inputs(expected=expected, independent_inputs=independent_inputs, mode=mode)
    need((expected.stage, expected.role) == ('state5', 'state5_worker'), 'State 5 worker required')
    need(all(s.kind in ('tensor', 'verification', 'scientific_members', 'evidence') for s in expected.outputs), 'State 5 cannot create rho/final/A6 metadata')
    from .state_execution_authority import decode
    allowed_tensors = frozenset(('kappa64.float64.npy', 'u_tpfa64.float64.npy',
        'kappa32_areaavg.float64.npy', 'u32_areaavg.float64.npy',
        'kappa16_areaavg.float64.npy', 'u16_areaavg.float64.npy'))
    for slot in expected.outputs:
        path = decode(slot.address)['relative_path']
        need(not path.endswith(('rho.float64.npy', 'shard_manifest.json', 'split_manifest.json', 'final_context.json', 'corpus_manifest.json')), 'prohibited State 5 product')
        if mode == 'component_validation':
            name = path.rsplit('/', 1)[-1]
            if slot.kind == 'tensor':
                need(name in allowed_tensors, 'only six non-rho tensor families may be retained')
            elif slot.kind == 'verification':
                need(name == 'verification.jsonl', 'only member verification JSONL may be retained')


def verify_and_release_member(*, expected, original, ports, run, mode):
    """One owned ephemeral handle; verify while live, preserve primary/release errors."""
    if mode == 'component_validation':
        return _verify_component_member(expected=expected, original=original, ports=ports, run=run)
    need(mode == 'synthetic' and type(ports) is State5Ports, 'explicit synthetic State5Ports')
    need(type(original) is OriginalBinding and original.split in ('training', 'validation', 'test')
         and type(original.shard) is int and type(original.offset) is int, 'original binding')
    index = {'training':0, 'validation':4096, 'test':4352}[original.split] + original.shard * 64 + original.offset
    need(0 <= index < len(expected.originals) and expected.originals[index] == original, 'original outside accepted inventory')
    prefix = f'{original.split}/{original.shard}/{original.offset}'
    solve = None
    owned = False
    solve_attempted = False
    solve_returned = False
    try:
        acquired = run.call(prefix + '/acquire', ports.acquire_original, original)
        need(type(acquired) is OriginalBinding and acquired == original, 'original rho/custody substitution')
        # Save ownership before checking an incomplete solver reply. A valid
        # returned owned token must be released even if other solve facts fail.
        solve_attempted = True
        reply = ports.solve_original(acquired)
        solve_returned = True
        if type(reply) is OperationReply and type(reply.payload) is SyntheticSolve:
            solve = reply.payload
            owned = ports.owns_ephemeral(solve.p1_token, original) is True
            need(owned, 'ephemeral ownership unestablished')
        solved = run.call(prefix + '/solve', lambda: reply)
        need(type(solved) is SyntheticSolve and solved.original == original, 'solve original binding')
        need(type(solved.tpfa_token) is bytes and bool(solved.tpfa_token)
             and type(solved.p1_token) is bytes and bool(solved.p1_token)
             and solved.tpfa_token != solved.p1_token, 'distinct opaque numerical tokens')
        need(type(solved.observation_tokens) is tuple and len(solved.observation_tokens) == 2
             and all(type(t) is bytes and t for t in solved.observation_tokens), 'original solve observations unavailable')
        checked = run.call(prefix + '/verify', ports.verify_original, solved)
        need(type(checked) is SyntheticScienceReply and checked.solve == solved
             and checked.simulation_only is True and checked.numerical_qualification is False, 'explicit simulation comparison required')
        need(checked.checked_tokens == (original.token, solved.tpfa_token, solved.p1_token, *solved.observation_tokens)
             and checked.diagnostics == (), 'synthetic member binding/verification failed')
        from .state_execution_authority import sha
        retained = SyntheticMemberEvidence(original, solved.tpfa_token, solved.observation_tokens,
                                            sha(b'Batch-D-inert-check\x00' + solved.p1_token).encode())
        run.call(prefix + '/retain', ports.retain_member, original, retained)
    except BaseException as exc:
        run.fail(exc)
        if solve_attempted and not solve_returned:
            run.unavailable.append(prefix + ':solve_outcome_unavailable')
        elif solve is not None and not owned:
            run.unavailable.append(prefix + ':ephemeral_ownership_unavailable')
    finally:
        if solve is not None and owned:
            try:
                # The port owns cleanup of the returned handle; no stringification
                # of arbitrary objects, no serialized field/projection/reaction.
                need(type(solve.p1_token) is bytes and bool(solve.p1_token), 'ephemeral ownership unavailable')
                before_primary = run.primary
                try:
                    run.call(prefix + '/release', ports.release_ephemeral, solve.p1_token)
                finally:
                    # run.call retains returned failures. Do not promote a cleanup
                    # exception over an already established primary failure.
                    if before_primary is not None:
                        run.primary = before_primary
            except BaseException as exc:
                run.fail(exc, cleanup=True)
                run.unavailable.append(prefix + ':ephemeral_release_unestablished')


def _verify_component_member(*, expected, original, ports, run):
    """Verify numerical science and independent retention before releasing owned P1."""
    from .state_phase_adapters import MemberScienceVerifier
    from .state_scientific_adapters import (OriginalAcquisition, OwnedNumericalSolve,
        ComponentMemberResult, RetainedMemberReadback, bind_retained_member)

    need(type(ports) is State5Ports and type(ports.verify_original) is MemberScienceVerifier,
         'actual component State5 science port required')
    need(getattr(ports.acquire_original, '__self__', None) is ports.verify_original and
         getattr(ports.acquire_original, '__func__', None) is MemberScienceVerifier.acquire,
         'original acquisition/science verifier substitution')
    need(type(original) is OriginalBinding and original.split in ('training', 'validation', 'test')
         and type(original.shard) is int and type(original.offset) is int, 'original binding')
    index = {'training':0, 'validation':4096, 'test':4352}[original.split] + original.shard * 64 + original.offset
    need(0 <= index < len(expected.originals) and expected.originals[index] == original,
         'original outside accepted inventory')
    prefix = f'{original.split}/{original.shard}/{original.offset}'
    solve = None
    owned = False
    solve_attempted = False
    solve_returned = False
    try:
        acquired = run.call(prefix + '/acquire', ports.acquire_original, original)
        need(type(acquired) is OriginalAcquisition and acquired.original == original,
             'original rho/custody substitution')
        solve_attempted = True
        reply = ports.solve_original(acquired)
        solve_returned = True
        # A returned lease is owned even when the rest of the operation reply is
        # invalid. Establish that fact before run.call can reject the reply.
        if type(reply) is OperationReply and type(reply.payload) is OwnedNumericalSolve:
            solve = reply.payload
            owned = ports.owns_ephemeral(solve.p1_capability, original) is True
            need(owned, 'ephemeral ownership unestablished')
        solved = run.call(prefix + '/solve', lambda: reply)
        need(type(solved) is OwnedNumericalSolve and solved.original == original,
             'owned numerical solve/original binding')
        need(solved.source_sha256 == acquired.source_sha256,
             'owned numerical solve/source substitution')
        need(type(solved.solve_observations) is tuple and len(solved.solve_observations) == 2,
             'two owned solve observations required')
        checked = run.call(prefix + '/verify', ports.verify_original, solved)
        need(type(checked) is ComponentMemberResult and checked.original == original,
             'exact numerical member verification required')
        need(checked.source_sha256 == solved.source_sha256 and checked.capture == solved.capture,
             'verified member/source or capture substitution')
        need(checked.observations == tuple(row.file for row in solved.solve_observations)
             and checked.assessment.status == 'matched', 'actual member comparison required')
        readback = run.call(prefix + '/retain', ports.retain_member, original, checked)
        need(type(readback) is RetainedMemberReadback, 'independent retained readback required')
        retained = bind_retained_member(verified=checked, readback=readback)
        need(retained.verified == checked and retained.readback == readback,
             'retained member binding disagreement')
    except BaseException as exc:
        run.fail(exc)
        if solve_attempted and not solve_returned:
            run.unavailable.append(prefix + ':solve_outcome_unavailable')
        elif solve is not None and not owned:
            run.unavailable.append(prefix + ':ephemeral_ownership_unavailable')
    finally:
        if solve is not None and owned:
            try:
                before_primary = run.primary
                try:
                    run.call(prefix + '/release', ports.release_ephemeral, solve.p1_capability)
                finally:
                    if before_primary is not None:
                        run.primary = before_primary
            except BaseException as exc:
                run.fail(exc, cleanup=True)
                run.unavailable.append(prefix + ':ephemeral_release_unestablished')


def run_state5_members(*, expected, independent_inputs, ports, mode):
    run = _entry(expected, independent_inputs, mode)
    if run.failed:
        return run.result()
    try:
        validate_state5_inputs(expected=expected, independent_inputs=independent_inputs, mode=mode)
        need(type(ports) is State5Ports, 'explicit State5Ports')
        for original in expected.originals:
            verify_and_release_member(expected=expected, original=original, ports=ports, run=run, mode=mode)
            if run.failed or run.unavailable:
                break
        if not run.failed and not run.unavailable:
            run.call('state5_members_finish', ports.finish)
    except BaseException as exc:
        run.fail(exc)
    return run.result()


def close_state5_execution(*, expected, independent_inputs, worker_result, closure_inputs, mode):
    need(expected.stage == 'state5', 'State 5 closure')
    return close_execution(expected=expected, independent_inputs=independent_inputs,
                           worker_result=worker_result, closure_inputs=closure_inputs, mode=mode)
