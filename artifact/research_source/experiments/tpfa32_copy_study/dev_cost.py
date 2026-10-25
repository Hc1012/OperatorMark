"""Static revision2 final projection from retained fixture rates; no numerical work."""
from core import HERE,read


def final_projection(selection):
    rates=read(HERE/'evidence/readiness_cost_20260930/recovery_20260930/PROJECTIONS.json')['rates']['wall_seconds']
    primary=selection['primary'];a=selection['recipes']['R-a']['endpoint_updates']
    b=selection['recipes']['R-b']['endpoint_updates'] if primary=='R-b' else 0
    # Raw equal-step arm never aliases a cooled endpoint, even at base1920.
    sets=80 if primary=='R-a' else 100
    truth=24896 if primary=='R-a' else 26596
    forward=10*(3840+a);d16=19200+10*b*5/6;d5=10*b/6
    checkpoints=50 if primary=='R-a' else 60;continuous=10 if primary=='R-a' else 20
    numerical=truth*rates['Cq']+forward*rates['F']+d16*rates['D16']+d5*rates['D5']+3200*rates['M']+sets*(rates['E_FJK']+rates['I64'])
    detector=3840*rates['B17']/17
    io=sets*(rates['json_FJK']+rates['json_inverse'])+checkpoints*rates['checkpoint_model']+continuous*rates['checkpoint_continuous']+truth*rates['truth_serialization_per_query']
    setup=10*rates['mark_setup128_proxy']+checkpoints*rates['initialization_proxy']+5*rates['E_FJ']+rates['startup']
    return dict(schema='tpfa32_final_cost_r2.v1',projection_only=True,not_execution_authority=True,
        primary=primary,selected_base={k:v['updates'] for k,v in selection['recipes'].items()},
        selected_endpoint_updates=dict(Ra=a,Rb=b),model_updates=60800+10*(a+b),
        inverse_model_sets=sets,inverse_updates=480*sets,nominal_solver_calls=truth+64*sets,
        detector_calls=3840,dependent_null_pairs=190,equal_step_distinct=True,
        projected_wall_seconds=numerical+detector+io+setup,
        components=dict(numerical=numerical,detector=detector,serialization_proxy=io,setup_proxy=setup),
        assumptions='Retained short CPU fixture rates; no sustained throughput or whole-run memory validation; cooldown uses same per-update cost, including zero-LR last step.')
