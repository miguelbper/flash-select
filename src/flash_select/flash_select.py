import warnings
from typing import Any

import numpy as np
import polars as pl
import scipy
from numpy.typing import NDArray
from shap import Explainer

FEATURE_NAME = "feature name"
T_VALUE = "t-value"
STAT_SIGNIFICANCE = "stat.significance"
COEFFICIENT = "coefficient"
SELECTED = "selected"


def flash_select(
    tree_model: Any,
    X: NDArray,
    y: NDArray,
    features: list[str],
    threshold: float = 0.05,
) -> pl.DataFrame:
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    S, _ = shap_values(tree_model, X)

    A = S.T @ S
    b = S.T @ y
    m, n = S.shape
    y_sq = np.square(np.linalg.norm(y))

    df = significance(A, b, features, m, n, y_sq)

    df = df.with_columns(
        pl.when(pl.col(COEFFICIENT) < 0)
        .then(-1)
        .when(pl.col(STAT_SIGNIFICANCE) < threshold)
        .then(1)
        .otherwise(0)
        .alias(SELECTED)
    )

    return df


def shap_values(tree_model: Any, X: NDArray) -> tuple[NDArray, NDArray]:
    explainer = Explainer(tree_model)
    shap_values = explainer(X)
    return shap_values.values, shap_values.base_values


def significance(
    A: NDArray,
    b: NDArray,
    features: list[str],
    m: int,
    n: int,
    y_sq: float,
) -> pl.DataFrame:
    A_pinv, A_rank = pinv_rank(A)
    full_rank: bool = A_rank == n

    if not full_rank:
        warnings.warn(f"Matrix A is rank deficient with rank {A_rank} < {n}. Algorithm will be slower.", stacklevel=1)

    results = []

    for _ in range(n):
        ols_out = ols(A_pinv, A_rank, b, features, m, y_sq)

        idx = ols_out[T_VALUE].arg_min()
        row = ols_out.row(idx, named=True)
        results.append(pl.DataFrame(row))

        A, b, features, A_pinv, A_rank, full_rank = downdate(A, b, features, A_pinv, full_rank, idx)

    return pl.concat(results).sort(T_VALUE, descending=True)


def pinv_rank(A: NDArray) -> tuple[NDArray, int]:
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    tol = np.max(s) * max(A.shape) * np.finfo(s.dtype).eps

    nonzero = s > tol
    s_inv = np.zeros_like(s)
    s_inv[nonzero] = 1 / s[nonzero]

    A_pinv = Vh.T @ np.diag(s_inv) @ U.T
    A_rank = np.sum(nonzero)
    return A_pinv, A_rank


def ols(
    A_pinv: NDArray,
    A_rank: int,
    b: NDArray,
    features: list[str],
    m: int,
    y_sq: float,
) -> pl.DataFrame:
    residual_dof: int = m - A_rank

    beta = A_pinv @ b  # (n,)
    rss = y_sq - np.dot(b, beta)  # (1,)
    sigma_sq = rss / residual_dof  # (1,)
    inv_diag = np.diag(A_pinv)  # (n,)
    t_stats = beta / np.sqrt(sigma_sq * inv_diag)  # (n,)
    p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stats), residual_dof))  # (n,)

    df = pl.DataFrame(
        {
            FEATURE_NAME: features,
            T_VALUE: t_stats,
            STAT_SIGNIFICANCE: p_values,
            COEFFICIENT: beta,
        }
    )

    return df


def downdate(
    A: NDArray,
    b: NDArray,
    features: list[str],
    A_pinv: NDArray,
    full_rank: bool,
    idx: int,
) -> tuple[NDArray, NDArray, list[str], NDArray, int, bool]:
    n = A.shape[0]
    mask = ~np.eye(n, dtype=bool)[:, idx]
    A_down = A[mask, :][:, mask]
    b_down = b[mask]
    features_down = [f for f, m in zip(features, mask, strict=True) if m]

    if full_rank:
        # Using: https://en.wikipedia.org/wiki/Block_matrix#Computing_submatrix_inverses_from_the_full_inverse
        E = A_pinv[mask, :][:, mask]  # (n - 1, n - 1)
        G = A_pinv[idx, mask]  # (1, n - 1)
        H = A_pinv[idx, idx]  # (1, 1)
        A_pinv_down = E - np.outer(G, G) / H  # (n - 1, n - 1)
        A_rank_down = n - 1
    else:
        A_pinv_down, A_rank_down = pinv_rank(A_down)

    full_rank_down = A_rank_down == A_down.shape[0]

    return A_down, b_down, features_down, A_pinv_down, A_rank_down, full_rank_down
