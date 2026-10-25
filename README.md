# OperatorMark

**Watermark Removal in Neural PDE Operators: Limits of Passive Ownership Evidence**  
Hadi Chamas · Paper v0.7 · Research artifact 1.0.0-rc5

**[Read the paper and download its LaTeX source and reproduction supplement](https://doi.org/10.5281/zenodo.23646972).**

Which ownership evidence survives when a copied neural PDE operator remains useful? OperatorMark is a ten-owner robustness study of an active tangent-space watermark and a passive tangent-distance comparator on a discrete Darcy-flow benchmark.

**Download the complete code and evidence archive from [GitHub Releases](https://github.com/Hc1012/OperatorMark/releases).** Choose the release asset named **`OperatorMark_1.0.0-rc5.zip`**. This repository's front page is the access point; the complete source tree, compact evidence, analysis scripts and scientific documentation are distributed together in that versioned archive.

Do not substitute GitHub's automatically generated **Source code (zip)** download or the green **Code → Download ZIP** button: those contain this small landing repository, rather than the full research artifact.

## Findings

“Utility” here means that forward, two derivative and inverse-task errors each remain within the adopted 25% increase relative to the corresponding clean owner's numerical-reference error.

| Setting | Saved result across ten planned owners |
| --- | --- |
| Ordinary fine-tuning, 256 forward pairs | At the retained descriptive checkpoint of 320 updates: utility 9/10, qualified active removal 8/10. At 960: utility 5/10, removal 4/10. Passive B2 remains positive for every useful copy at both checkpoints. |
| Frozen ordinary 1,920-update primary | Active detection 0/10 and full utility 0/10. Copies outperform one selected scratch replacement: geometric replacement/copy error ratios 7.16 for random-direction derivatives and 2.53 for the inverse task. Earlier checkpoints do not replace this primary. |
| A4 / A5-B3 derivative-aware attacks | At 20,480 updates: utility 10/10, qualified removal 9/10 and passive positivity with utility 10/10. These attacks use 16,384 derivative-labeled controls, four times owner training's 4,096. No matched-access scratch comparison establishes the benefit of acquired initialization in this regime. |
| Jacobi256 after ordinary 1,920 | Eight owners pass utility and qualify as active removals. **Seven useful copies defeat both evaluated tests.** The passive cohort remains mixed because its evasion rule requires utility in at least nine owners. |

O05's intact final mark is undetected, so it stays in the denominator but cannot count as attack-induced removal. The seven joint failures concern these two tested signals, not every possible sign of model descent. The Jacobi256 wrapper uses known PDE coefficients for 256 correction iterations on every forward call. Five of the seven joint-failure owners were already B2-negative before wrapping; two changed sign in the wrapped comparison. The matched comparator descendants also change, so these transitions do not identify a unique causal mechanism. Whether copied weights are necessary for the wrapped utility is unmeasured. Correction adds inference work; fewer labels do not establish lower total attack cost.

The findings are limited to one numerical benchmark, the declared procedures and fixed utility rules. Physical qualification was negative. B2 is a fixed-comparator ranking without a calibrated population false-positive rate. Historical negative results, amendments and conditional or unsatisfied obligations remain in the archive.

## Download and reproduce

1. Open [Releases](https://github.com/Hc1012/OperatorMark/releases) and download **`OperatorMark_1.0.0-rc5.zip`** from the release assets.
2. Extract that ZIP. Its top-level folder is **`OperatorMark`**.
3. Open the extracted **`OperatorMark/README.md`** for the full instructions. Open your terminal **inside that extracted `OperatorMark` folder**, where `reproduce.py` and `requirements-analysis.txt` are located.
4. With Python **3.12** installed, use the corresponding commands below. These repeat the archive README's analysis quickstart.

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

The analysis runs offline on CPU after dependency installation. It trains no models, runs no native solvers, uses no Colab runtime and buys no compute. Its expected final line is:

```text
PASS: exact analysis match, 4 figures and 128 variant summaries.
```

Outputs are written inside the extracted artifact at `build/reproduced/`. To verify the archive's shipped file hashes without installing analysis dependencies, run `python reproduce.py --verify-only` from that same folder.

## What the archive contains

- The original directory tree: `research_source/`, `campaign_source/`, `analysis/`, `provenance/`, `environment/` and `docs/`.
- 1,271 compact model observations, including 871 full/noise evaluations and 400 conditional diagnostics, with explicit task missingness.
- Frozen analysis rules, all 128 variant summaries, four reproducibility figures and original evidence pointers.
- Full methods, attacker-access budgets, limitations, historical outcomes and scope dispositions.

This supports **saved-result reproducibility**, not independent raw-model or end-to-end campaign replication. The full corpus, checkpoints, teacher caches, private keys and all raw score grids are not distributed. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md); paths described there are inside the release archive.

## Paper, credit and license

The preprint **Watermark Removal in Neural PDE Operators: Limits of Passive Ownership Evidence** (v0.7, 25 October 2026) is available on Zenodo: **[doi:10.5281/zenodo.23646972](https://doi.org/10.5281/zenodo.23646972)**. The same record includes the LaTeX source and **`OperatorMark_Manuscript_Reproduction.zip`**, which supplies the Figure 1 and Table 2 rendering scripts and per-owner records. This manuscript supplement accompanies the paper; it is separate from the unchanged rc5 research artifact.

Please use [CITATION.cff](CITATION.cff) for the paper's preferred citation and the separate software-artifact metadata. The paper and its LaTeX source are licensed under **CC BY 4.0**. The repository, research-artifact code and reproduction-supplement code retain the [MIT license](LICENSE); third-party dependencies retain their licenses. See the [authorship statement](AUTHORS.md). Publication does not imply peer review or fulfillment of the stated qualification and assurance limitations.
