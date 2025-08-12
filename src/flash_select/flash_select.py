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


def flash_select(
    tree_model: Any,
    X: NDArray,
    y: NDArray,
    features: list[str],
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Perform feature selection using the Flash Select algorithm.

    This function implements a feature selection method that combines SHAP values
    with statistical significance testing to identify important features in tree-based models.
    It iteratively removes the least significant features based on t-statistics.

    Parameters
    ----------
    tree_model : Any
        A tree-based model (e.g., XGBoost, LightGBM) that supports SHAP explanation.
        Must have a `get_booster()` method that returns an object with `get_score()`.
    X : NDArray
        Feature matrix of shape (n_samples, n_features) where n_samples is the number
        of samples and n_features is the number of features.
    y : NDArray
        Target variable of shape (n_samples,) containing the response values.
    features : list[str]
        List of feature names corresponding to the columns in X.
    threshold : float, default=0.05
        Significance threshold for feature selection. Features with p-values above
        this threshold will be marked as not selected.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing feature selection results with columns:
        - feature name: Name of the feature
        - t-value: T-statistic for the feature
        - stat.significance: P-value for statistical significance
        - coefficient: Estimated coefficient for the feature
        - selected: Selection status (1=selected, 0=not selected, -1=negative coefficient)

        The DataFrame is sorted by t-value (descending) and feature name (ascending).

    Notes
    -----
    The algorithm works by:
    1. Computing SHAP values for all features
    2. Identifying unused features (those not used by the tree model)
    3. Iteratively performing OLS regression on SHAP values
    4. Removing the least significant feature in each iteration
    5. Computing t-statistics and p-values for feature significance

    Examples
    --------
    >>> import xgboost as xgb
    >>> from flash_select import flash_select
    >>>
    >>> # Train a model
    >>> model = xgb.XGBRegressor()
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Perform feature selection
    >>> result = flash_select(model, X_test, y_test, feature_names)
    >>> print(result)
    """
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    S, _ = shap_values(tree_model, X)

    feature_scores = tree_model.get_booster().get_score()
    mask = np.array([f"f{i}" in feature_scores for i in range(X.shape[1])])
    unused_features = list(map(str, np.array(features)[~mask]))
    num_unused_features = len(unused_features)
    df_unused_features = pd.DataFrame(
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

    df = pd.concat([df, df_unused_features])
    df[SELECTED] = np.where(df[COEFFICIENT] < 0, -1, np.where(df[STAT_SIGNIFICANCE] < threshold, 1, 0))
    df = df.sort_values(by=[T_VALUE, FEATURE_NAME], ascending=[False, True])
    df = df.reset_index(drop=True)
    return df


def shap_values(tree_model: Any, X: NDArray) -> tuple[NDArray, NDArray]:
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
    return shap_values.values, shap_values.base_values


def significance(
    A: NDArray,
    b: NDArray,
    features: list[str],
    m: int,
    n: int,
    num_unused_features: int,
    y_sq: float,
) -> pd.DataFrame:
    """Perform iterative feature selection using statistical significance
    testing.

    This function implements the core algorithm that iteratively removes the least
    significant feature based on t-statistics from OLS regression on SHAP values.

    Parameters
    ----------
    A : NDArray
        Matrix A = S^T @ S where S are the SHAP values, shape (n_features, n_features).
    b : NDArray
        Vector b = S^T @ y where y is the target variable, shape (n_features,).
    features : list[str]
        List of feature names corresponding to the features in A and b.
    m : int
        Number of samples (rows in original SHAP matrix).
    n : int
        Number of features (columns in original SHAP matrix).
    num_unused_features : int
        Number of features not used by the tree model.
    y_sq : float
        Squared L2 norm of the target variable (y^T @ y).

    Returns
    -------
    pd.DataFrame
        DataFrame containing significance results for all features with columns:
        - feature name: Name of the feature
        - t-value: T-statistic for the feature
        - stat.significance: P-value for statistical significance
        - coefficient: Estimated coefficient for the feature
    """
    A_pinv = np.linalg.pinv(A)
    results = []

    for _ in range(n):
        ols_out = ols(A_pinv, b, features, m, num_unused_features, y_sq)

        idx = ols_out[T_VALUE].argmin()
        row = ols_out.iloc[idx].to_dict()
        results.append(pd.DataFrame([row]))

        A, b, features, A_pinv = downdate(A, b, features, A_pinv, idx)

    return pd.concat(results)


def ols(
    A_pinv: NDArray,
    b: NDArray,
    features: list[str],
    m: int,
    num_unused_features: int,
    y_sq: float,
) -> pd.DataFrame:
    """Perform Ordinary Least Squares (OLS) regression and compute significance
    statistics.

    Parameters
    ----------
    A_pinv : NDArray
        Pseudo-inverse of matrix A, shape (n_features, n_features).
    b : NDArray
        Vector b = S^T @ y, shape (n_features,).
    features : list[str]
        List of feature names.
    m : int
        Number of samples.
    num_unused_features : int
        Number of features not used by the tree model.
    y_sq : float
        Squared L2 norm of the target variable.

    Returns
    -------
    pd.DataFrame
        DataFrame containing OLS results with columns:
        - feature name: Name of the feature
        - t-value: T-statistic for the feature
        - stat.significance: P-value for statistical significance
        - coefficient: Estimated coefficient for the feature
    """
    residual_dof = m - (A_pinv.shape[0] + num_unused_features)

    beta = A_pinv @ b  # (n,)
    rss = y_sq - np.dot(b, beta)  # (1,)
    sigma_sq = rss / residual_dof  # (1,)
    inv_diag = np.diag(A_pinv)  # (n,)
    t_stats = beta / np.sqrt(sigma_sq * inv_diag)  # (n,)
    p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stats), residual_dof))  # (n,)

    df = pd.DataFrame(
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
    """Downdate matrices and vectors by removing a feature at the specified
    index.

    This function removes the least significant feature from the system and updates
    all related matrices and vectors accordingly. It uses the Sherman-Morrison-Woodbury
    formula to efficiently update the pseudo-inverse.

    Parameters
    ----------
    A : NDArray
        Matrix A, shape (n_features, n_features).
    b : NDArray
        Vector b, shape (n_features,).
    features : list[str]
        List of feature names.
    A_pinv : NDArray
        Pseudo-inverse of matrix A, shape (n_features, n_features).
    idx : int
        Index of the feature to remove.

    Returns
    -------
    tuple[NDArray, NDArray, list[str], NDArray]
        A tuple containing:
        - A_down: Downdated matrix A, shape (n_features-1, n_features-1)
        - b_down: Downdated vector b, shape (n_features-1,)
        - features_down: Updated list of feature names
        - A_pinv_down: Downdated pseudo-inverse, shape (n_features-1, n_features-1)

    Notes
    -----
    The downdating formula used is based on the Sherman-Morrison-Woodbury formula:
    (A - uv^T)^(-1) = A^(-1) - (A^(-1)uv^T A^(-1)) / (1 + v^T A^(-1)u)
    where u and v are vectors representing the removed row/column.
    """
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
