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
    """Represents the current state of the feature selection algorithm.

    Attributes
    ----------
    A : NDArray[np.float32]
        Design matrix of shape (n, n) where n is the number of active features.
        A = S^T @ S where S is the SHAP values matrix.
    b : NDArray[np.float32]
        Response vector of shape (n,) where n is the number of active features.
        b = S^T @ y where y is the target variable.
    features : NDArray[np.str_]
        Array of feature names of shape (n,) for the currently active features.
    A_inv : NDArray[np.float32]
        Inverse of the design matrix A of shape (n, n).
    beta : NDArray[np.float32]
        Coefficient vector of shape (n,) from the current OLS regression.
    ysq : NDArray[np.float32]
        Squared norm of the target variable of shape (1,).
    rss : NDArray[np.float32]
        Residual sum of squares of shape (1,).
    residual_dof : int
        Residual degrees of freedom for statistical testing.
    safe_invert: bool
        Whether the inverse of the design matrix A is safe to compute.
    """

    A: NDArray[np.float32]  # (n, n)
    b: NDArray[np.float32]  # (n,)
    features: NDArray[np.str_]  # (n,)
    A_inv: NDArray[np.float32]  # (n, n)
    beta: NDArray[np.float32]  # (n,)
    ysq: NDArray[np.float32]  # (1,)
    rss: NDArray[np.float32]  # (1,)
    residual_dof: int
    safe_invert: bool


def flash_select(
    tree_model: Any,
    X: NDArray,
    y: NDArray,
    features: Iterable[str],
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Main function to perform feature selection using SHAP values and
    statistical significance.

    This function implements the Flash Select algorithm which:
    1. Computes SHAP values for the given tree model
    2. Removes unused features (those not used by the model)
    3. Iteratively removes the least significant feature based on t-statistics
    4. Returns a DataFrame with feature rankings and selection status

    Parameters
    ----------
    tree_model : Any
        A tree-based model (e.g., XGBoost, LightGBM) that supports SHAP explanation.
    X : NDArray
        Feature matrix of shape (n_samples, n_features).
    y : NDArray
        Target variable of shape (n_samples,).
    features : Iterable[str]
        List of feature names corresponding to the columns in X.
    threshold : float, default=0.05
        Significance threshold for feature selection. Features with p-value < threshold
        are considered significant.

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature information with columns:
        - feature name: Name of the feature
        - t-value: t-statistic for the feature coefficient
        - stat.significance: p-value for the feature coefficient
        - coefficient: Estimated coefficient value
        - selected: Selection status (-1 for negative, 1 for significant, 0 for not significant)

        Features are sorted by t-value (descending) and then by feature name (ascending).

    Warns
    -----
    UserWarning
        If the design matrix A is not full rank, indicating potential numerical issues.
    """
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    features = np.array(features, dtype=np.str_)

    S = shap_values(tree_model, X)

    df_unused_features, S, features = remove_unused_features(tree_model, S, features)
    num_unused_features = len(df_unused_features)

    state = initial_state(S, y, features, num_unused_features)

    if not state.safe_invert:
        warnings.warn(
            "Matrix A is ill-conditioned. Results will still be accurate, but algorithm will be a bit slower.",
            stacklevel=1,
        )

    df = significance(state)

    df = pd.concat([df, df_unused_features])
    df[SELECTED] = np.where(df[COEFFICIENT] < 0, -1, np.where(df[STAT_SIGNIFICANCE] < threshold, 1, 0))
    df = df.sort_values(by=[T_VALUE, FEATURE_NAME], ascending=[False, True])
    df = df.reset_index(drop=True)
    return df


def shap_values(tree_model: Any, X: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute SHAP values for a tree-based model.

    This function uses the SHAP library to compute feature importance values
    for each sample-feature combination. SHAP values provide a unified measure
    of feature importance that can be used for feature selection.

    Parameters
    ----------
    tree_model : Any
        A tree-based model that supports SHAP explanation.
    X : NDArray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    NDArray[np.float32]
        SHAP values matrix of shape (n_samples, n_features) where each element
        represents the contribution of a feature to the prediction for a sample.
    """
    explainer = Explainer(tree_model)
    shap_values = explainer(X)
    return shap_values.values


def remove_unused_features(
    tree_model: Any, S: NDArray[np.float32], features: NDArray[np.str_]
) -> tuple[pd.DataFrame, NDArray[np.float32], NDArray[np.str_]]:
    """Remove features that are not used by the tree model.

    This function identifies features that have zero importance scores in the tree model
    and removes them from the SHAP values matrix and features list. It creates a DataFrame
    for these unused features with appropriate default values.

    Parameters
    ----------
    tree_model : Any
        A tree-based model with a get_booster().get_score() method.
    S : NDArray[np.float32]
        SHAP values matrix of shape (n_samples, n_features).
    features : NDArray[np.str_]
        Array of feature names of shape (n_features,).

    Returns
    -------
    tuple[pd.DataFrame, NDArray[np.float32], NDArray[np.str_]]
        A tuple containing:
        - df_unused_features: DataFrame with unused feature information
        - S: Updated SHAP values matrix with unused features removed
        - features: Updated feature names array with unused features removed
    """
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
    """Initialize the state for the feature selection algorithm.

    This function computes the initial values for the State object by:
    1. Computing the design matrix A = S^T @ S
    2. Computing the response vector b = S^T @ y
    3. Computing the inverse of A using pseudo-inverse
    4. Computing initial coefficients beta = A_inv @ b
    5. Computing residual sum of squares (RSS)
    6. Computing residual degrees of freedom

    Parameters
    ----------
    S : NDArray[np.float32]
        SHAP values matrix of shape (n_samples, n_features).
    y : NDArray[np.float32]
        Target variable of shape (n_samples,).
    features : NDArray[np.str_]
        Array of feature names of shape (n_features,).
    num_unused_features : int
        Number of features that were removed as unused.

    Returns
    -------
    State
        Initial state object with all computed values.
    """
    A = S.T @ S
    b = S.T @ y

    # Compute minimal dtype to make sure we can invert A
    cond = np.linalg.cond(A)
    dtypes = [np.float32, np.float64]
    safe_dtypes = [dtype for dtype in dtypes if cond < 1 / np.finfo(dtype).eps]
    safe_invert = bool(safe_dtypes)
    dtype = safe_dtypes[0] if safe_dtypes else dtypes[-1]

    A = A.astype(dtype)
    b = b.astype(dtype)

    A_inv = np.linalg.pinv(A)
    beta = A_inv @ b
    ysq = np.square(np.linalg.norm(y))
    rss = ysq - np.dot(b, beta)
    m, n = S.shape
    residual_dof = m - (n + num_unused_features)

    return State(A, b, features, A_inv, beta, ysq, rss, residual_dof, safe_invert)


def significance(state: State) -> pd.DataFrame:
    """Perform iterative feature selection based on statistical significance.

    This function implements the core feature selection algorithm by:
    1. Computing OLS statistics for the current state
    2. Identifying the least significant feature (lowest t-value)
    3. Downdating the state by removing that feature
    4. Repeating until all features are processed

    Parameters
    ----------
    state : State
        Current state object containing the design matrix, coefficients, and other
        relevant information.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results for all features that were processed,
        sorted by the order they were removed (least significant first).
        Each row contains feature name, t-value, significance, and coefficient.
    """
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
    """Compute Ordinary Least Squares (OLS) statistics for the current state.

    This function computes t-statistics, p-values, and other statistical measures
    for the current feature set. It uses the current state's inverse matrix and
    residual sum of squares to compute these statistics efficiently.

    Parameters
    ----------
    state : State
        Current state object containing the design matrix, inverse matrix,
        coefficients, and residual information.

    Returns
    -------
    pd.DataFrame
        DataFrame containing statistical information for each feature:
        - feature name: Name of the feature
        - t-value: t-statistic for testing H0: beta = 0
        - stat.significance: Two-sided p-value for the t-test
        - coefficient: Estimated coefficient value (beta)
    """
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
    """Efficiently update the state after removing a feature.

    This function implements the Sherman-Morrison-Woodbury formula to efficiently
    update the inverse matrix and other state variables when a feature is removed.
    This avoids the need to recompute the full inverse matrix, making the
    algorithm computationally efficient.

    Parameters
    ----------
    state : State
        Current state object before feature removal.
    idx : int
        Index of the feature to remove.

    Returns
    -------
    State
        Updated state object with the specified feature removed.
    """
    A = state.A
    b = state.b
    features = state.features
    A_inv = state.A_inv
    beta = state.beta
    ysq = state.ysq
    rss = state.rss
    residual_dof = state.residual_dof
    safe_invert = state.safe_invert

    mask = np.arange(len(features)) != idx
    b_0 = b[idx]
    beta_0 = beta[idx]

    A = A[mask, :][:, mask]
    b = b[mask]
    features = features[mask]

    # If save_invert, assume A is invertible and use fast update formulas
    # Else, compute pseudo-inverse from scratch
    if safe_invert:
        E = A_inv[mask, :][:, mask]
        G = A_inv[mask, idx]
        H = A_inv[idx, idx]
        G_sub_H = G / H
        G_sub_H_dot_b = np.dot(G_sub_H, b)

        A_inv = E - np.outer(G, G_sub_H)
        beta = beta[mask] - G * (b_0 + G_sub_H_dot_b)
        rss += b_0 * beta_0 + H * G_sub_H_dot_b * (b_0 + G_sub_H_dot_b)
    else:
        A_inv = np.linalg.pinv(A)
        beta = A_inv @ b
        rss = ysq - np.dot(b, beta)

    residual_dof += 1

    return State(A, b, features, A_inv, beta, ysq, rss, residual_dof, safe_invert)
