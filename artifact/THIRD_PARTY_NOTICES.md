# Third-party dependencies and attribution

No third-party dependency distribution, virtual environment, model checkpoint, downloaded dataset, font file or library binary is bundled here. Dependencies are installed separately and remain under their own terms. The project MIT license does not relicense them.

## Public analysis dependencies

| Dependency | Tested version | Upstream license / source |
| --- | --- | --- |
| NumPy | 2.3.5 | [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| SciPy | 1.17.0 | [SciPy license](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| Matplotlib | 3.10.8 | [Matplotlib license](https://matplotlib.org/stable/project/license.html) |
| Python | 3.12.14 for release validation | [Python license](https://docs.python.org/3/license.html) |

The direct analysis versions are locked in `requirements-analysis.txt`. The validation environment's dependency inventory is supplied in `environment/ANALYSIS_PACKAGES.json`; no full scientific environment is installed by the reproduction command.

## Historical research implementation

The neural-operator research code uses PyTorch, NeuralOperator, NumPy and SciPy, among other dependencies listed in `environment/requirements-colab-historical.txt`. That file is an exact retained historical inventory, not a statement that every package is a direct dependency or that the current cloud image is compatible.

- [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE) — upstream copyright notices and redistribution conditions apply to PyTorch itself.
- [NeuralOperator license](https://github.com/neuraloperator/neuraloperator/blob/main/LICENSE) — MIT, NeuralOperator developers.

The release inspection found no vendored third-party copyright/license headers or copied upstream package tree in the retained project source snapshot. This observation is not proof of original authorship of every line. Original project/source hashes are provided for traceability; dependency licenses and scholarly method citations remain distinct. The paper's related-work citations have not been replaced by this software notice.

Figures were generated from the included saved measurements with Matplotlib's standard DejaVu Sans rendering. No external photograph, artwork, company logo, or third-party paper figure is included.
