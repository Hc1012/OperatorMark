"""Risk-based development matrix. Final populations unchanged; no grants."""
from .variants import ARCHITECTURES,TRANSFORMS
from .development_inputs import plan as generation
from .p256_verification import COUNTS as derivative
from .p256_tasks import workload

def plan():
    models=['core_ordinary']
    # One qualification population plus one nonidentity final-sized context.
    q=1;f=len(models);controls=256*q+512*f;detector_controls=256*q+128*f
    return dict(approved=False,one_session_fit=False,
        stage_order=['metadata_runtime','input_generation','P256_derivative','kernel_state_screen','qualification_evaluator',
                     'nonidentity_final_evaluator','reconstruction_paths','jacobi_inverse','P256_two_target_task_probe',
                     'contract_calibration','contract_kernels','independent_readback_export'],
        generation=generation(),P256_derivative=dict(derivative),P256_task_probe=workload(2,2),
        model_updates=356,kernel_model_updates=36,overwrite_model_updates=320,teacher_forward_queries=512,teacher_JVP_queries=256,
        overwrite_received_forward=128,overwrite_received_utility_JVP=128,overwrite_received_key_JVP=128,
        reconstruction_architectures=list(ARCHITECTURES)+['fno32_JVP'],architecture_batch_sizes=[32]*6+[16],
        reconstruction_transforms=list(TRANSFORMS)+['jacobi16','jacobi256'],transform_batch=16,
        architecture_F_JVP_VJP_observations=7,transform_wrapper_F_JVP_VJP_observations=18,
        old_monitor=dict(targets=16,batched_inverse_iterations=120,T128=16,P256=16),
        qualification_evaluations=q,full_evaluation_models=models,
        evaluator_batched_inverse_iterations=(q+f)*480,
        evaluator_T128_scoring=64*(q+f),evaluator_P256_scoring=64*(q+f),
        evaluator_validation_forwards=controls,evaluator_random_JVPs=2*controls,evaluator_keyed_JVPs=controls,
        detector_grids=q+f,detector_scores=4096*(q+f),
        detector_projection_forwards_no_cache=2*detector_controls,
        detector_projection_JVPs_no_cache=2*detector_controls*64,
        B2_pair_distances=1,B2_controls=16,B2_directions=2,B2_model_JVPs=64,
        jacobi256_inverse=dict(batch=16,updates=120,native_calls=0),jacobi16_inverse=dict(batch=16,updates=2,native_calls=0),
        noise=dict(sds=[.0001,.001],batch=16,inverse_updates_per_sd=2,retained_draws_per_call=32,
            calls_per_sd=4,retained_noisy_tensor_outputs=256,logical_noisy_control_outputs=4096,
            B3_scores_per_sd=4096,extra_native=0,model_updates=0,white_box_only=True),
        API=dict(policies=3,ordered_attempts_per_policy=34,attempts=102,hidden_gradients=0,native=0),
        overwrite_calibration=dict(scores=12,references=3,controls_per_reference=128,
            neural_JVPs=1920,normalizer_forward=384,F64_forward=384,F64_tangent=384,new_controls=384,new_keys=3),
        adaptive=dict(development_search_scores=4096,calibration_grids=12,calibration_scores=49152,
            untouched_development_readout_scores=4096,reference='bound existing Rstar10240',
            projection_controls_per_grid=128,projection_model_instances_with_shared_reference_upper=17,
            projection_forward_upper=2176,projection_JVP_upper=139264,model_updates=0,
            calibration_roster_pending=False,normalization_JVP_observations=128,normalization_returned_primal_outputs=128),
        normal_journal_events=72,overwrite_update_records=320,state_checkpoints=17,retained_state_publication_copies=17,
        evaluation_records_per_final=107,evaluation_records_qualification=91,
        aggregate_native=dict(T128_forward=834,T128_tangent=1152,T128_refinement_solves=1986,
            P256_forward=203,P256_tangent=3,P256_tangent_continuation=3,P256_adjoint=10,P256_direct_oracle=12),
        numerical_implementation_complete=False,
        pending=['numerical verification and paid execution approval',
            'generated inputs/catalog and exact future runtime admission'])
