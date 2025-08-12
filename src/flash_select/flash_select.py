import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from shap import Explainer

FEATURE_NAME = "feature name"
T_VALUE = "t-value"
STAT_SIGNIFICANCE = "stat.significance"
COEFFICIENT = "coefficient"
SELECTED = "selected"


@dataclass
class State:
    A: NDArray[np.float32]  # (n, n)
    b: NDArray[np.float32]  # (n,)
    features: NDArray[np.str_]  # (n,)
    A_inv: NDArray[np.float32]  # (n, n)
    beta: NDArray[np.float32]  # (n,)
    rss: NDArray[np.float32]  # (1,)
    residual_dof: int


def flash_select(
    tree_model: Any,
    X: NDArray,
    y: NDArray,
    features: Iterable[str],
    threshold: float = 0.05,
) -> pd.DataFrame:
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    features = np.array(features, dtype=np.str_)

    S = shap_values(tree_model, X)

    df_unused_features, S, features = remove_unused_features(tree_model, S, features)
    num_unused_features = len(df_unused_features)

    state = initial_state(S, y, features, num_unused_features)

    if np.linalg.matrix_rank(state.A) < state.A.shape[0]:
        warnings.warn(
            "Matrix A is not full rank! May not get correct results. Recommended: try again with a deeper tree model.",
            stacklevel=1,
        )

    df = significance(state)

    df = pd.concat([df, df_unused_features])
    df[SELECTED] = np.where(df[COEFFICIENT] < 0, -1, np.where(df[STAT_SIGNIFICANCE] < threshold, 1, 0))
    df = df.sort_values(by=[T_VALUE, FEATURE_NAME], ascending=[False, True])
    df = df.reset_index(drop=True)
    return df


def remove_unused_features(
    tree_model: Any, S: NDArray[np.float32], features: NDArray[np.str_]
) -> tuple[pd.DataFrame, NDArray[np.float32], NDArray[np.str_]]:
    feature_scores = tree_model.get_booster().get_score()
    mask = np.array([f"f{i}" in feature_scores for i in range(S.shape[1])])
    unused_features = np.array(features)[~mask]
    num_unused_features = len(unused_features)
    df_unused_features = pd.DataFrame(
        {
            FEATURE_NAME: unused_features,
            T_VALUE: np.full(num_unused_features, np.nan),
            STAT_SIGNIFICANCE: np.full(num_unused_features, np.nan),
            COEFFICIENT: np.zeros(num_unused_features),
        }
    )
    features = np.array(features)[mask]
    S = S[:, mask]
    return df_unused_features, S, features


def initial_state(
    S: NDArray[np.float32],
    y: NDArray[np.float32],
    features: NDArray[np.str_],
    num_unused_features: int,
) -> State:
    A = S.T @ S
    b = S.T @ y
    A_inv = np.linalg.pinv(A)
    beta = A_inv @ b
    rss = np.square(np.linalg.norm(y)) - np.dot(b, beta)
    m, n = S.shape
    residual_dof = m - (n + num_unused_features)
    return State(A, b, features, A_inv, beta, rss, residual_dof)


def shap_values(tree_model: Any, X: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute SHAP values for a tree-based model.

    Parameters
    ----------
    tree_model : Any
        A tree-based model that supports SHAP explanation.
    X : NDArray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    tuple[NDArray, NDArray]
        A tuple containing:
        - SHAP values of shape (n_samples, n_features)
        - Base values of shape (n_samples,)
    """
    explainer = Explainer(tree_model)
    shap_values = explainer(X)
    return shap_values.values


def significance(state: State) -> pd.DataFrame:
    n = len(state.features)
    results = []

    for _ in range(n):
        ols_out = ols(state)

        idx = ols_out[T_VALUE].argmin()
        row = ols_out.iloc[idx].to_dict()
        results.append(pd.DataFrame([row]))

        state = downdate(state, idx)

    return pd.concat(results)


def ols(state: State) -> pd.DataFrame:
    sigma_sq = state.rss / state.residual_dof
    inv_diag = np.diag(state.A_inv)
    t_stats = state.beta / np.sqrt(sigma_sq * inv_diag)
    p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stats), state.residual_dof))

    df = pd.DataFrame(
        {
            FEATURE_NAME: state.features,
            T_VALUE: t_stats,
            STAT_SIGNIFICANCE: p_values,
            COEFFICIENT: state.beta,
        }
    )

    return df


def downdate(state: State, idx: int) -> State:
    A = state.A
    b = state.b
    features = state.features
    A_inv = state.A_inv
    beta = state.beta
    rss = state.rss
    residual_dof = state.residual_dof

    mask = np.arange(len(features)) != idx
    b_0 = b[idx]
    beta_0 = beta[idx]

    A = A[mask, :][:, mask]
    b = b[mask]
    features = features[mask]

    E = A_inv[mask, :][:, mask]
    G = A_inv[mask, idx]
    H = A_inv[idx, idx]
    G_sub_H = G / H
    G_sub_H_dot_b = np.dot(G_sub_H, b)

    A_inv = E - np.outer(G, G_sub_H)
    beta = beta[mask] - G * (b_0 + G_sub_H_dot_b)
    rss += b_0 * beta_0 + H * G_sub_H_dot_b * (b_0 + G_sub_H_dot_b)

    residual_dof += 1

    return State(A, b, features, A_inv, beta, rss, residual_dof)
