# Methods and implementation map

This document describes the completed, amended numerical campaign. It is not a new protocol, a replacement preregistration or permission to rerun a historical launcher. Detailed measurements are in `analysis/inputs/`; the authoritative saved-result readout is `analysis/expected/FINAL_ANALYSIS.json`.

## Research question and populations

OperatorMark asks when ownership detection persists under similarity notions that preserve useful engineering behaviour. **d0** concerns model-to-model output similarity, **d1** model-to-model derivative similarity, and **dcap** downstream capability under specified constraints and budgets. In this campaign, the reported F/J/K/T values are errors against the declared numerical/task references. They are not automatically direct d0/d1 distances between two learned operators.

The main cohort is O01–O10, **N = 10**, with resource fraction **κ = 0.25**. The fixed primary includes all ten owners. The 10%-data B10 model is evaluated once in O01 context, excluded from keyed nulls and B2, and is not an eleventh owner. The earlier ten-owner TPFA32 study is analyzed separately. Shared benchmark targets, directions, keys and checkpoints do not create extra independent owners.

The source graph is preserved unchanged, including dated readiness flags. Those flags describe its historical state; `provenance/SCOPE_AT_ANALYSIS_COMPLETION.json` records later dispositions. Existing scientific-acceptance and custody flags are not rewritten by statistical analysis or packaging.

## Numerical errors and fixed decision rules

| Quantity | Meaning / decision |
| --- | --- |
| F | Forward error against the declared reference. |
| J | Random-direction JVP error against that reference. |
| K | Keyed-direction JVP error against that reference. |
| T | Mean inverse-task error from independently solver-scored saved designs (T128 path). |
| P256 | Separate numerical scoring path, reported in addition to T; not substituted into the F/J/K/T utility gate. |
| Utility | Every F/J/K/T component is finite, nonnegative and ≤ 1.25 × the original clean owner's component on the same final test. |
| Detection | True score > frozen owner θ **and** true score > maximum absolute wrong-key score + δ. A full grid contains one true and 4,095 wrong keys. |
| Attributable removal | The intact mark first passes detection and utility on the same test; the descendant then loses detection while passing all four utility components. |

All ten owners were qualified before attacks. The final intact test detects nine: O05 passes utility but not detection. The planned denominator remains ten; O05 cannot contribute an attack-induced removal. The ordinary P2 primary does not filter owners by that intact prerequisite.

## Paired primary inference

For each owner, form the ratio `Rstar error / Aord_M3_raw1920 error`, separately for J and T. Analyze the ten owner log-ratios with the frozen Student-t rule (df = 9). Geometric means and separate 97.5% pointwise intervals support the paired advantage decision; equivalence uses separate 95% pointwise intervals against [0.8, 1.25], requiring all components to pass. The latter are not simultaneous 95% intervals.

The interpretation is conditional on the fixed shared benchmark and development procedure, with approximately normal independent owner log-ratios. Secondary attack summaries are descriptive; checkpoints, targets and correlated attack choices are not treated as additional confirmatory replications. The primary label is **copy advantage**, not equivalence. The ordinary attacked copies at 1,920 fail the joint utility gate, so this primary alone does not establish qualified removal. The retained, fully task-measured ordinary320/960 readouts are descriptive and give different removal outcomes; they do not replace the primary or supply a post-hoc stopping rule.

## Attack and comparator coverage

| Block | Retained scope |
| --- | --- |
| Ordinary / scratch | Fixed ordinary 1,920-update primary and selected Rstar replacement; descriptive ordinary320/960 readouts and clean/marked/sham controls retained. |
| Derivative-aware | A4 forward-plus-half-JVP normalized loss; A5-B3 adds public projection penalty. Checkpoints at 1,280, 5,120, 20,480. No secret-key access is inferred. |
| Students | Seven methods, clean and qualified marked teachers, ten owners: 140 jobs, 268,800 updates and 420 checkpoints. Terminal and conditional intermediate readouts remain distinct. |
| Weights / transformations | Pruning, healing, quantization, mixing, overwrite, FFT filtering, rounding, precision storage, remeshing and Jacobi16/256 wrappers at the frozen settings. |
| Noise | Adopted additive white-box pathwise contract; the detector and derivatives use the un-noised base. No stochastic black-box removal inference. |
| API | 30 persistent streams of 1,024 attempts for cache, paired-denial and rate policies. Only service outcomes; no J/K/T or B2/B3 service decision. |
| Native challengers | One shared F64 and one shared P256 inverse procedure, each on 64 targets. Not ten independent replicas per solver. |
| B2 passive lineage | Own clean owner versus nine matched other-owner descendants and nineteen fixed controls. Strictly positive margin against every fixed comparator; no embedded mark and no population ownership probability. |
| Fixed nulls | 210 dependent fixed cases: per owner, nine other clean models, ten replacements, own sham and F64; B10 excluded. |
| Adaptive accuser | One retained calibrated procedure and final readout per owner, using its own calibration/reference/normalizer. |

The full data/access and endpoint table is in [THREAT_MODEL_AND_BUDGETS.md](THREAT_MODEL_AND_BUDGETS.md). In particular, ordinary/scratch training uses numerical-reference forward labels, whereas students use received-teacher labels. A4/A5 use 16,384 reference controls with derivatives; ordinary/scratch uses 256 controls. Verification here uses white-box/pathwise differentiation and the original clean reference, not an arbitrary prediction-only service.

B2 uses the first 128 of the final 512 metric controls and their same two directions. Its observations therefore overlap the reference-error evaluation, rather than constituting a separate independent test sample. Positivity is a strict ranking against all 28 fixed comparators, without a calibrated population error rate or established exchangeability null. The cohort classifier labels retained at at least nine positives, evaded at at most one positive and at least nine utility-passing owners, and mixed otherwise. Seven useful individual joint failures under Jacobi256 after ordinary are compatible with its mixed cohort label.

The result inventories have different units: 128 distinct variant labels; 124 B2 labels, excluding B10, O, Rraw1920 and Rstar; and 36 active marked-parent procedure cohorts that can span multiple checkpoints. B10 has only an O01-context row, so total compact model observations are 1,271 rather than 1,280.

The frozen graph in `analysis/inputs/FROZEN_GRAPH.json` retains exact variants, trajectories and endpoints. The source rules retain cohort success thresholds and use intermediate checkpoints where prescribed; a terminal-only plot does not replace those rules.

## Deliberate omissions and amendments

The D2 rule assessed all 400 student/healing raw320/raw960 diagnostics in 20 cells and selected no extra task cell. Their T/P256 remain **unmeasured**. The 200 marked intermediate cases are undetected and fail F/J/K; 180 are settled by utility failure and 20 are O05 cases without the same-test intact prerequisite. The raw960 development labels are preserved, with the accepted results-only adapter documented in the campaign source. These are distinct from ordinary320/960, whose final-domain task results were measured.

The 130 originally deferred contract slots were later bound. This is distinct from 50 intentionally omitted clean-parent noise/API rows, the unmeasured D2 tasks, and the untriggered original extensions. D8 was adopted after observed corpus anomalies; it must not be presented as a wholly prospective acceptance rule. See `LIMITATIONS_AND_HISTORY.md`.

## Source map

Paths below are relative to the repository root. The original 338-file map and any public-only alteration are enumerated in `provenance/RESEARCH_SOURCE_EXPORT.json`; later implementations are in `provenance/CAMPAIGN_SOURCE_EXPORT.json`.

| Subject | Source locations |
| --- | --- |
| Frozen paired statistics | `analysis/frozen_source/experiments/tpfa32_copy_study/{analyze,core}.py` |
| Cohort / joint readouts | `research_source/tools/msfinal/{analysis,routing,orchestration,d2}.py` |
| Detector and passive rules | `research_source/tools/msdev/rules.py`; `research_source/tools/msfinal/nulls.py` |
| Model / training / marking | `research_source/tools/q5_model.py`; `research_source/tools/msfinal/{models,owners,marking,attacks,expanded}.py`; `research_source/tools/msdev/kernels.py` |
| Final evaluation / CPU scoring | `research_source/tools/msfinal/{evaluation,cpu_scoring,p256_tasks,p256_adjoint}.py` |
| Variants / graph / contracts | `research_source/tools/msfinal/{variants,graph,contract_arms,final_contracts}.py` |
| Darcy reference and corpus | `research_source/operatormark/solvers/`; `research_source/operatormark_numerics_v2/`; `research_source/operatormark_corpus_v2/` |
| Latest null/B2 and noise | `campaign_source/null_noise_D2/{null_compute,null_reference,noise_evaluate,noise_producer}.py` |
| API / native / D2 adapter | `campaign_source/API/`; `campaign_source/native/`; `campaign_source/D2_selection/` |
| Historical TPFA32 / D3 / P-N | `research_source/experiments/tpfa32_copy_study/`; `research_source/tools/msdev/`; `provenance/historical/` |

The public reproducibility command imports only the named pure statistical functions from the nine analysis files and hashes them first. It does not import the scientific workers or execute these archived launchers.
