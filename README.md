<div align="center">

# Project Title
[![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Code Quality](https://github.com/miguelbper/flash-select/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/miguelbper/flash-select/actions/workflows/code-quality.yaml)
[![Unit Tests](https://github.com/miguelbper/flash-select/actions/workflows/tests.yaml/badge.svg)](https://github.com/miguelbper/flash-select/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/miguelbper/flash-select/graph/badge.svg)](https://codecov.io/gh/miguelbper/flash-select)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

*An extremely fast implementation of [shap-select](https://github.com/transferwise/shap-select).*

![project-image.png](project-image.png)

</div>

---

## Description
flash-select is an extremely fast implementation of [shap-select](https://github.com/transferwise/shap-select), a very nice feature selection method. So, flash-select gives the same output as shap-select (more on this below) while being significantly faster: for a dataset with 25600 examples and 256 features, flash-select is ... faster.

These speedups enable feature selection for datasets with thousands of features. The package is tiny, thoroughly tested, and has few dependencies (numpy, polars, scipy, shap).

## Installation
```bash
pip install flash-select
```

## Run the benchmark
```bash
# Clone the project
git clone git@github.com:miguelbper/flash-select.git

# Install uv if you don't have it yet (from https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Move to the project directory. Install (dev) dependencies
uv sync

# Run benchmark
uv run benchmark/benchmark.py
```

## How is it so fast?
The original implementation of shap-select iterativelly performs a linear regression on the dataset (Shapley values, target), where at each iteration we delete one column of the Shapley values matrix. With no regularization, the linear regression coeficients $\beta$ are given by:

$
    \begin{align}
    A &= S^T S \\
    b &= S^T y \\
    \beta &= A^{-1} b.
    \end{align}
$

We can save on computation by doing linear regression explicitly (instead of calling an external library) and updating (instead of recomputing from scratch) the matrix $A^{-1}$.

Note that shap-select uses a small L1 regularization of $\alpha = 10^{-6}$.
- It is possible to show mathematically that flash-select gives exactly the same results as shap-select with $\alpha = 0$ (this is also verified in the unit tests)
- Numerical experiments show that flash-select gives the same set of selected features as shap-select with $\alpha = 10^{-6}$

For these reasons, $\alpha = 0$ in flash-select, which enables speedups of several orders of magnitude.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
[shap-select](https://github.com/transferwise/shap-select) and its authors.
