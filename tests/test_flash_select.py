import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from shap import Explainer
from shap_select import shap_select
from statsmodels.regression.linear_model import OLS
from xgboost import XGBRegressor

from flash_select.flash_select import (
    COEFFICIENT,
    FEATURE_NAME,
    STAT_SIGNIFICANCE,
    T_VALUE,
    State,
    downdate,
    flash_select,
    get_Ab,
    initial_state,
    ols,
    unused_features,
)

N_SEEDS = 10
M = 100
N = 4
tol = 1e-5
FEATURES = [f"f{i}" for i in range(N)]


@pytest.fixture(params=range(N_SEEDS))
def seed(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[np.float32, np.float64])
def dtype(request: pytest.FixtureRequest) -> np.dtype:
    return request.param


@pytest.fixture
def X(seed: int, dtype: np.dtype) -> NDArray[np.number]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(M, N)).astype(dtype)
    return X


@pytest.fixture
def y(seed: int, dtype: np.dtype) -> NDArray[np.number]:
    rng = np.random.default_rng(seed + N_SEEDS)
    y = rng.normal(size=(M,)).astype(dtype)
    return y


@pytest.fixture(params=[True, False])
def use_all_features(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def tree_model(use_all_features: bool, seed: int, dtype: np.dtype) -> XGBRegressor:
    rng = np.random.default_rng(seed + 2 * N_SEEDS)
    X_train = rng.normal(size=(M, N)).astype(dtype)
    y_train = rng.normal(size=(M,)).astype(dtype)

    if use_all_features:
        N_ESTIMATORS = 10
        MAX_DEPTH = 3
        MAX_LEAVES = 2**MAX_DEPTH
    else:
        N_ESTIMATORS = 1
        MAX_DEPTH = 2
        MAX_LEAVES = 2**MAX_DEPTH

    model = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, max_leaves=MAX_LEAVES, random_state=42)
    model.fit(X_train, y_train)

    used_features = model.get_booster().get_score().keys()
    used_all_features = len(used_features) == N
    assert used_all_features == use_all_features

    return model


@pytest.fixture(params=[1, M // 2, M])
def batch_size(request: pytest.FixtureRequest) -> int | None:
    return request.param


def test_get_Ab(
    tree_model: XGBRegressor,
    X: NDArray[np.number],
    y: NDArray[np.number],
    dtype: np.dtype,
    batch_size: int | None,
) -> None:
    _, n = X.shape
    mask = np.ones(n, dtype=np.bool_)
    A_stream, b_stream = get_Ab(tree_model, X, y, mask, dtype, batch_size)
    A_batch, b_batch = get_Ab(tree_model, X, y, mask, dtype, None)

    assert A_stream.shape == (n, n)
    assert b_stream.shape == (n,)
    assert A_batch.shape == (n, n)
    assert b_batch.shape == (n,)

    assert A_stream.dtype == dtype
    assert b_stream.dtype == dtype
    assert A_batch.dtype == dtype
    assert b_batch.dtype == dtype

    assert np.allclose(A_stream, A_batch, atol=tol, rtol=tol)
    assert np.allclose(b_stream, b_batch, atol=tol, rtol=tol)


@pytest.fixture
def state(tree_model: XGBRegressor, X: NDArray[np.number], y: NDArray[np.number], dtype: np.dtype) -> State:
    features = np.array(FEATURES)
    n = len(features)
    feature_scores = tree_model.get_booster().get_score()
    mask = np.array([f"f{i}" in feature_scores for i in range(n)])
    return initial_state(tree_model, X, y, features, mask, dtype)


@pytest.fixture(params=range(N))
def idx(request: pytest.FixtureRequest, state: State) -> int | None:
    n = state.A.shape[0]
    i = request.param
    if i >= n:
        pytest.skip("Index out of range")
    return i


def test_downdate(state: State, idx: int, dtype: np.dtype) -> None:
    n = state.A.shape[0]
    ysq = state.ysq
    residual_dof = state.residual_dof
    state_down = downdate(state, idx)

    A_down = state_down.A
    b_down = state_down.b
    features_down = state_down.features
    A_inv_down = state_down.A_pinv
    beta_down = state_down.beta
    rss_down = state_down.rss
    residual_dof_down = state_down.residual_dof

    # shapes
    assert A_down.shape == (n - 1, n - 1)
    assert b_down.shape == (n - 1,)
    assert features_down.shape == (n - 1,)
    assert A_inv_down.shape == (n - 1, n - 1)
    assert beta_down.shape == (n - 1,)
    assert rss_down.shape == ()

    # dtypes
    assert A_down.dtype == dtype
    assert b_down.dtype == dtype
    assert features_down.dtype.kind == "U"
    assert A_inv_down.dtype == dtype
    assert beta_down.dtype == dtype
    assert rss_down.dtype == dtype

    # formulas / properties
    assert np.allclose(A_inv_down, np.linalg.pinv(A_down), atol=tol, rtol=tol)
    assert np.allclose(beta_down, A_inv_down @ b_down, atol=tol, rtol=tol)
    assert np.allclose(rss_down, ysq - np.dot(b_down, beta_down), atol=tol, rtol=tol)
    assert residual_dof_down == residual_dof + 1


def test_ols(
    tree_model: XGBRegressor,
    X: NDArray[np.number],
    y: NDArray[np.number],
    state: State,
) -> None:
    def ols_ours(tree_model: XGBRegressor, state: State) -> pd.DataFrame:
        df_unused_features, _ = unused_features(tree_model, FEATURES)
        df_ols = ols(state)
        df_out = pd.concat([df_ols, df_unused_features]).sort_values(by=FEATURE_NAME)
        return df_out.reset_index(drop=True)

    def ols_statsmodels(tree_model: XGBRegressor, X: NDArray[np.number], y: NDArray[np.number]) -> pd.DataFrame:
        explainer = Explainer(tree_model)
        S = explainer(X).values

        df_S = pd.DataFrame(S, columns=FEATURES)
        df_y = pd.Series(y, name="target")
        model = OLS(df_y, df_S)
        result = model.fit_regularized(alpha=0.0, refit=True)
        table = result.summary2().tables[1]
        df = table.reset_index()
        rename_by = {
            "index": FEATURE_NAME,
            "t": T_VALUE,
            "P>|t|": STAT_SIGNIFICANCE,
            "Coef.": COEFFICIENT,
        }
        df = df.rename(columns=rename_by)[list(rename_by.values())]

        df[T_VALUE] = np.where(df[COEFFICIENT].abs() < 1e-10, np.nan, df[T_VALUE])
        df[STAT_SIGNIFICANCE] = np.where(df[COEFFICIENT].abs() < 1e-10, np.nan, df[STAT_SIGNIFICANCE])

        return df

    df_0 = ols_ours(tree_model, state)
    df_1 = ols_statsmodels(tree_model, X, y)

    pd.testing.assert_frame_equal(df_0, df_1, check_dtype=False, rtol=tol, atol=tol)


def test_flash_select(
    tree_model: XGBRegressor,
    X: NDArray[np.number],
    y: NDArray[np.number],
    dtype: np.dtype,
    batch_size: int | None,
) -> None:
    df_flash_select = flash_select(tree_model, X, y, FEATURES, dtype=dtype, batch_size=batch_size)

    X_df = pd.DataFrame(X, columns=FEATURES)
    y_df = pd.Series(y, name="target")
    df_shap_select = shap_select(tree_model, X_df, y_df, task="regression", alpha=0.0)
    df_shap_select = df_shap_select.sort_values(by=[T_VALUE, FEATURE_NAME], ascending=[False, True])

    pd.testing.assert_frame_equal(df_flash_select, df_shap_select, check_dtype=False, rtol=tol, atol=tol)
