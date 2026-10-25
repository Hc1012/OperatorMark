# OperatorMark

**Ownership signals and useful copying in neural PDE operators**  
Hadi Chamas · Research artifact · Release candidate 1.0.0-rc5, 25 October 2026

Public repository: [https://github.com/Hc1012/OperatorMark](https://github.com/Hc1012/OperatorMark). This release candidate is the initial public artifact deposit. The separately prepared preprint has not been submitted or publicly deposited; the main announcement remains pending.

Can a copied neural PDE simulator lose its watermark while retaining the derivatives and inverse-design behaviour that make it useful? OperatorMark studies that question through controlled attacks on Darcy-flow neural operators. TangentMark is the active watermark candidate; its success is an empirical question, not a premise.

The completed ten-owner campaign finds that ownership evidence depends on attacker access, checkpoint and the adopted definition of utility. Useful active-mark removal can retain positive passive evidence, but seven useful solver-corrected copies defeat both evaluated tests.

| Main result | Evidence |
| --- | --- |
| Forward-only fine-tuning can yield useful removal at retained early checkpoints | With 256 forward pairs: ordinary 320 has utility **9/10**, qualified removal **8/10** and B2 positivity plus utility **9/10**; ordinary 960 has **5/10**, **4/10** and **5/10**, respectively. B2 is positive for all ten at both checkpoints. |
| The frozen primary remains a distinct comparison | At ordinary 1,920, detection, full utility and qualified removal are each **0/10**. Replacement/copy geometric error ratios are **7.16** for random JVPs and **2.53** for the inverse task. Earlier checkpoints do not replace this primary. |
| Derivative-labeled data permit useful removal with residual passive evidence | A4 and A5-B3 at 20,480: utility **10/10**, qualified removal **9/10 planned**, B2 positivity plus utility **10/10**. They use four times the clean owner's labeled-control count; the benefit of acquired initialization over matched-access scratch training is unmeasured. |
| Solver correction can defeat both evaluated tests in useful copies | Jacobi256 after ordinary 1,920 gives utility and qualified removal for **8/10**, with nonpositive B2 for **7/10 useful owners**. The B2 cohort stays **mixed**, since its evasion rule requires utility for at least nine owners. |

O05 lacks a detectable intact final mark and cannot count as attack-induced removal. It remains in the planned denominator. The seven joint failures are O01, O03, O04, O07, O08, O09 and O10; they do not establish erasure of every possible signal of descent.

![Owner-level error ratios and frozen primary intervals](docs/figures/01_primary_copy_advantage.png)

The primary error ratios compare against **one selected replacement procedure**, not every possible scratch learner. Ordinary copies use the 1,920-update primary; the selected scratch procedure uses 10,240. Both receive the same owner-specific 256 numerical-reference forward labels. A4/A5-B3 instead share a separate set of 16,384 controls with derivative labels, compared with 4,096 in owner training. Their success does not isolate derivative supervision at equal data or compute, establish a necessary fourfold budget, or establish an economic benefit from stealing. Jacobi adds numerical correction work to each forward call.

At ordinary 1,920, the inverse-task gate passes all ten while J/K fail all ten and F fails five. The early ordinary checkpoints are final-domain and task-measured; they are separate from the D2 student/healing diagnostics with intentionally unmeasured tasks. The utility gates concern a declared **discrete numerical reference**, not validated physical or continuum accuracy. The shared benchmark and ten owners limit generalization. B2 is a fixed-comparator ranking without a calibrated population error rate.

## Start here

- [Results and interpretation](docs/RESULTS.md): the complete scientific synthesis, including negative results.
- [Methods and code map](docs/METHODS.md): populations, gates, attacks, inference and source locations.
- [Attacker access and budgets](docs/THREAT_MODEL_AND_BUDGETS.md): what each method receives and why the regimes differ.
- [Paper and publication status](docs/PAPER_AND_RELEASE.md): manuscript progress and the coordinated release sequence.
- [All 128 variant summaries](docs/ALL_VARIANT_TABLES.md): full tables, not just illustrative successes.
- [Limitations and history](docs/LIMITATIONS_AND_HISTORY.md): amendments, unsuccessful work and deferred obligations.
- [Data availability](DATA_AVAILABILITY.md): exactly what this release can reproduce.

## Reproduce the saved-result analysis

This is a **CPU-only, offline analysis** after dependency installation. It does not train models, run solvers, query a GPU, use Google Drive, start a study clock or buy compute. It regenerates the statistical readout, 128 variant summaries and four figures from **1,271 compact model observations**: 871 full/noise evaluations and 400 intermediate diagnostics.

Use Python 3.12. The tested environment and exact direct dependencies are recorded under `environment/` and in `requirements-analysis.txt`. Run the commands from the extracted **OperatorMark** folder, where this README is located.

**Windows PowerShell / VS Code terminal**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe reproduce.py
```

**Linux / macOS terminal**

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python reproduce.py
```

Expected final line: `PASS: exact analysis match, 4 figures and 128 variant summaries.` Outputs appear in `build/reproduced/`. The shipped expected result is never overwritten. `REPRODUCTION.json` records the actual versions, output hash and elapsed time. The Windows commands are supplied as instructions; the original release validation was performed on Linux, not on a live Windows machine. The rc5 revision records the verified public repository URL and prepares this artifact as its initial public deposit. The author read-through and publication-format preprint preparation are complete; paper submission/deposit and the main announcement remain pending. Analysis inputs, rules, scripts and the four reproducibility figures remain unchanged; the earlier numerical replay is reused. Prior rc2/rc3/rc4 editorial validation records remain under `provenance/`; `EDITORIAL_RELEASE_VALIDATION.json` records the rc5 publication-metadata changes and byte-identity checks.

To check all shipped file hashes without installing any analysis dependencies:

```sh
python reproduce.py --verify-only
```

The frozen analysis is compared **byte for byte** with `analysis/expected/FINAL_ANALYSIS.json`. PDF/SVG timestamps and platform font rendering can differ, so image-byte equality is reported separately.

## What is included

| Location | Contents |
| --- | --- |
| `analysis/inputs/` | Saved measurements, detection extrema, frozen graph, qualification inputs and separate historical statistics. |
| `analysis/frozen_source/` | Nine original files containing the frozen rules; their hashes are checked before the named pure functions execute. |
| `research_source/` | The 338-file pinned research snapshot: 337 files have identical contents; one historical handoff file has three path strings anonymized. Archived agent instructions are renamed as text. |
| `campaign_source/` | Later API, native, noise, D2, null/B2 and supervisor implementations retained as source evidence. |
| `provenance/` | Original source hashes, export mapping, claim pointers, historical outcomes and scope dispositions. |
| `docs/figures/` | Four figures, each available as PNG, SVG and PDF. |

`research_source/` and `campaign_source/` document the research implementation. Their old launchers are **not** the release quickstart: a full campaign requires the private corpus, checkpoints, keyed material, sealed inputs and original runtime contracts. This package reproduces the **analysis of saved results**, not independent raw-model or end-to-end campaign replication. Source-file hashes establish byte identity, not external trust or exclusive custody. Read [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) before making a reproducibility claim.

## Scope and credit

The numerical work adopted for this campaign and its final analysis are complete. Conditional original obligations remain explicitly conditional, failed physical qualification remains negative, and formal engineering/custody assurances are not retroactively claimed. The earlier TPFA32 study remains a separate population.

As of 25 October 2026, the author read-through is complete and the publication-format preprint (v0.4), *OperatorMark: Ownership Evidence in Useful Copies of Neural PDE Operators*, is prepared. The public repository exists at [https://github.com/Hc1012/OperatorMark](https://github.com/Hc1012/OperatorMark); this artifact is its initial public deposit. The preprint has not been submitted or publicly deposited, and its license remains undecided. The main announcement is pending. No peer-review status, DOI or arXiv identifier is implied. Software citation metadata is in [CITATION.cff](CITATION.cff). Original project material is supplied under the [MIT license](LICENSE); dependencies keep their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [AUTHORS.md](AUTHORS.md). The manuscript is kept separate from this MIT-licensed repository, and its publication license remains undecided.
