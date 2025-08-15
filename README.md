<div align="center">

# ⚡ flash-select ⚡
[![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Code Quality](https://github.com/miguelbper/flash-select/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/miguelbper/flash-select/actions/workflows/code-quality.yaml)
[![Unit Tests](https://github.com/miguelbper/flash-select/actions/workflows/tests.yaml/badge.svg)](https://github.com/miguelbper/flash-select/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/miguelbper/flash-select/graph/badge.svg)](https://codecov.io/gh/miguelbper/flash-select)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

*An extremely fast <ins>feature selection method</ins> / <ins>implementation of [shap-select](https://github.com/transferwise/shap-select)</ins>.*

![img.png](img.png)

</div>

---

## Description
flash-select is an extremely fast implementation of [shap-select](https://github.com/transferwise/shap-select), a very nice feature selection method. flash-select gives the same output as shap-select (more on this below) while being significantly faster: for a dataset with 100000 examples and 100 features, **flash-select is ~200x faster**. For 200000 examples and 200 features, **it is ~1000x faster** (see the benchmark below).

Given that flash-select has lower algorithmic complexity than shap-select, for larger datasets the speedup will be even greater.

These speedups enable feature selection for datasets with thousands of features. The package is tiny, thoroughly tested, and has few dependencies (these are numpy, pandas, scipy, and shap).

flash-select works for regression problems on tabular datasets.

## Installation
```bash
pip install flash-select
```

## Usage
```python
from flash_select import flash_select
from xgboost import XGBRegressor

# Train a model
model = XGBRegressor()
model.fit(X_train, y_train)

# Perform feature selection
selected_features_df = flash_select(
    model,
    X_val,
    y_val,
    features=feature_names,
    threshold=0.05
)

print(selected_features_df)
```

## Running the Benchmark
```bash
# Clone the project
git clone git@github.com:miguelbper/flash-select.git

# Install uv if you don't have it yet (from https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Move to the project directory and install (dev) dependencies
uv sync

# Run benchmark
bash benchmark/run.sh
```

**Benchmark results:**
| Samples (m) | Features (n) | flash-select (s) | shap-select (s) | Speedup | Selected same set of features?|
|-------------|--------------|------------------|-----------------|---------|-------------------------------|
| 10000       | 10           | 0.37             | 0.50            | 1.3x    | Yes                           |
| 25000       | 25           | 1.00             | 4.41            | 4.4x    | Yes                           |
| 50000       | 50           | 2.25             | 35.69           | 15.8x   | Yes                           |
| 75000       | 75           | 2.26             | 216.36          | 95.9x   | Yes                           |
| 100000      | 100          | 3.66             | 761.63          | 208.3x  | Yes                           |
| 200000      | 200          | 10.61            | 11174.12        | 1052.7x | Yes                           |

**System Specifications:**
- **OS**: Ubuntu 24.04.2 LTS on Windows 10 x86_64 (WSL2)
- **Kernel**: Linux 6.6.87.2-microsoft-standard-WSL2
- **CPU**: AMD Ryzen 9 5900X (24 threads) @ 3.699GHz
- **Memory**: 96.6 GB RAM
- **Environment**: WSL2 (Windows Subsystem for Linux)

## How is it so fast?
The original implementation of shap-select iteratively performs a linear regression on the dataset $(S, y)$, where $S$ are the Shapley values and $y$ is the target. At each iteration we delete one column of the Shapley values matrix. With no regularization, the linear regression coefficients $\beta$ are given by:

$$
    \begin{align}
    A &\coloneqq S^T S \\
    b &\coloneqq S^T y \\
    \beta &= A^{-1} b.
    \end{align}
$$

We can save on computation by doing linear regression explicitly (instead of calling an external library) and updating (instead of recomputing from scratch) the matrix $A^{-1}$. The same logic is used for other arrays as well.

Note that shap-select uses a small L1 regularization of $\alpha = 10^{-6}$.
- It is possible to show mathematically that flash-select gives exactly the same results as shap-select with $\alpha = 0$ (this is also verified in the unit tests)
- Numerical experiments show that flash-select gives the same set of selected features as shap-select with $\alpha = 10^{-6}$

For these reasons, $\alpha = 0$ in flash-select, which enables speedups of several orders of magnitude.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
[shap-select](https://github.com/transferwise/shap-select) and its authors.
