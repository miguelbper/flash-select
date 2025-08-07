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

    feature_scores = tree_model.get_booster().get_score()
    mask = np.array([f"f{i}" in feature_scores for i in range(X.shape[1])])
    unused_features = list(map(str, np.array(features)[~mask]))
    num_unused_features = len(unused_features)
    df_unused_features = pl.DataFrame(
        {
            FEATURE_NAME: unused_features,
            T_VALUE: np.full(num_unused_features, np.nan),
            STAT_SIGNIFICANCE: np.full(num_unused_features, np.nan),
            COEFFICIENT: np.zeros(num_unused_features),
        }
    )
    features = list(map(str, np.array(features)[mask]))
    S = S[:, mask]

    A = S.T @ S
    b = S.T @ y
    m, n = S.shape
    y_sq = np.square(np.linalg.norm(y))

    df = significance(A, b, features, m, n, num_unused_features, y_sq)

    df = pl.concat([df, df_unused_features])

    df = df.with_columns(
        pl.when(pl.col(COEFFICIENT) < 0)
        .then(-1)
        .when(pl.col(STAT_SIGNIFICANCE) < threshold)
        .then(1)
        .otherwise(0)
        .alias(SELECTED)
    ).sort([T_VALUE, FEATURE_NAME], descending=[True, False])

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
    num_unused_features: int,
    y_sq: float,
) -> pl.DataFrame:
    A_pinv = np.linalg.pinv(A)
    results = []

    for _ in range(n):
        ols_out = ols(A_pinv, b, features, m, num_unused_features, y_sq)

        idx = ols_out[T_VALUE].arg_min()
        row = ols_out.row(idx, named=True)
        results.append(pl.DataFrame(row))

        A, b, features, A_pinv = downdate(A, b, features, A_pinv, idx)

    return pl.concat(results)


def ols(
    A_pinv: NDArray,
    b: NDArray,
    features: list[str],
    m: int,
    num_unused_features: int,
    y_sq: float,
) -> pl.DataFrame:
    residual_dof = m - (A_pinv.shape[0] + num_unused_features)

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
    idx: int,
) -> tuple[NDArray, NDArray, list[str], NDArray]:
    n = A.shape[0]

    mask = np.ones(n, dtype=bool)
    mask[idx] = False

    A_down = A[mask, :][:, mask]
    b_down = b[mask]
    features_down = [f for f, m in zip(features, mask, strict=True) if m]

    # Using: https://en.wikipedia.org/wiki/Block_matrix#Computing_submatrix_inverses_from_the_full_inverse
    E = A_pinv[mask, :][:, mask]  # (n - 1, n - 1)
    G = A_pinv[idx, mask]  # (1, n - 1)
    H = A_pinv[idx, idx]  # (1, 1)
    A_pinv_down = E - np.outer(G, G) / H  # (n - 1, n - 1)

    return A_down, b_down, features_down, A_pinv_down
