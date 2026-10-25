# Attacker access and fixed budgets

This describes the completed protocol. It does not authorize execution or amend scientific choices.

The central removal setting starts with an acquired model checkpoint. The attacker can inspect and update the received parameters. The original clean owner, private key and calibration are retained by the evaluator, which differentiates the suspect model. B2 also requires clean-reference and comparison-model JVPs. These are white-box/pathwise verification results, not a prediction-only ownership service.

| Procedure | Initialization | Data / labels | Prescribed endpoint |
| --- | --- | --- | ---: |
| Clean owner | Fresh FNO | 4,096 reference controls; forward + JVP | 81,920 |
| TangentMark | Original clean owner | 128 embedding controls; reference F/J plus private mark loss | 320 |
| Ordinary copy | Received owner | 256 numerical-reference forward labels | 320/960 descriptive; 1,920 primary |
| Scratch Rstar | Fresh FNO | Same 256 numerical-reference forward labels and initial shuffle | 10,240 |
| A4 / A5-B3 | Received owner | 16,384 reference controls with forward/JVP labels | 20,480 |
| Six forward student architectures | Fresh student | 256 cached received-teacher responses | 1,920 |
| JVP student | Fresh standard FNO | 256 cached forward responses plus 256 teacher JVP responses | 1,920 |

Endpoint lengths are not whole-campaign update/cost totals. Main/cooldown branches, corpus generation, monitoring, inference, transfers and failures have separate work. Kappa 0.25 sets the derivative-aware endpoint relative to 81,920; it does not set every dataset fraction or total bill.

Ordinary and scratch are matched in supplied ordinary data and initial shuffle, not total training updates or the inherited cost of the acquired checkpoint. A4 does not consume the private defender key; A5-B3 additionally consumes the public 16-pattern basis. Their 16,384-control dataset is four times the owner-training count, with stronger derivative/reference access and a different schedule. No matched-access scratch comparison was performed, so the benefit of the acquired initialization in that regime is unmeasured. Success cannot be attributed solely to one loss term, and fourfold data is not shown to be necessary. Jacobi wrappers additionally use the known numerical PDE coefficients and add 16 or 256 correction iterations per forward call; a smaller labeled training set does not establish lower total cost.

The six output-only students are FNO32, FNO16, FNO64, FNO2, DeepONet and U-Net adapters. The seventh is FNO32 trained on forward and teacher-JVP labels. Their common teacher caches are reused across architectures, not counted as fresh queries for every job. Failure under 256 queries and 1,920 updates is not a claim about convergence or unrestricted student capacity.

The API cache/denial/rate measurements are separate service experiments and provide no J/K/T or B2/B3 service decision. Additive-noise results retain their explicit un-noised derivative/detector convention.

Evidence: `research_source/tools/msfinal/attacks.py::{admit,_training_data,_execute_nodes}`, `owners.py::{admit,schedule}`, `expanded.py::{admit,query_binding,build_engine}`, `variants.py::definition`, `research_source/tools/msdev/kernels.py::loss_with_scales`, and the unchanged graph and final-analysis records.

## Data construction and overlap evidence

Owner training reads 64 retained corpus training shards of 64 controls, with forward and tangent labels. Ordinary preparation constructs 256 fresh controls per owner, with distinct fingerprints and known-control exclusions. Each owner's clean ordinary, marked ordinary and replacement arms share its numerical forward labels and initial shuffle. The retained return verification records 2,560 forward calls, verified known exclusions/pairings and an unopened final test; it does not claim independent re-solving of every label.

A4/A5 use a separately constructed shared set of 16,384 distinct controls with one feasible direction each. The committed population excludes the original deny list plus all 2,560 ordinary controls; the commitment precedes native calls and states that no corpus training/test arrays are opened. Completion records report 16,384 forward and 16,384 tangent labels. A `training` seed namespace does not mean the owner training arrays were read.

Final material uses the original 512 sealed test controls and retained checks against attack controls and task target/start controls. B2 uses the first 128 of those rows and the same two directions. Known-exclusion policies cover the named corpus, locked/historical, calibration and embedding sets while explicitly retaining D7's incomplete historical coverage. The full pinned deny-list membership was not newly inspected in the small returns. These are recorded construction/readback facts, not a new exhaustive nonoverlap audit.

Provenance: `ORDINARY_INPUTS_RETURN.zip::ordinary_materials.py::make_population`; `FINAL_ATTACKS_CONTROL.zip::ordinary/evidence/ORDINARY_INPUT_RETURN_VERIFIED.json`; `RSTAR_COMPLETION_AND_SHARED_INPUT_PREPARATION.zip::shared_materials.py::population`; `A4_MARKED_REVIEW_20261017T201744Z.zip::evidence/shared/{POPULATION_COMMITTED,MATERIAL_RESULT}.json`; and `SHARED_INPUT_RETURN_AND_A4_PREPARATION_RECORD.zip::SHARED_TRAINING_PLAN.json`. These are retained original private-archive pointers, not public download promises. B2 alignment is recorded by `campaign_source/null_noise_D2/null_reference.py::prepare`.
