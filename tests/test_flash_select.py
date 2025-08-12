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
    downdate,
    flash_select,
    initial_state,
    ols,
    shap_values,
)

N_SEEDS = 10
M = 100
N = 4
tol = 1e-5
FEATURES = [f"f{i}" for i in range(N)]


@pytest.fixture(params=range(N_SEEDS))
def seed(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def X(seed: int) -> NDArray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(M, N)).astype(np.float32)
    return X


@pytest.fixture
def y(seed: int) -> NDArray:
    rng = np.random.default_rng(seed + N_SEEDS)
    y = rng.normal(size=(M,)).astype(np.float32)
    return y


@pytest.fixture(params=[True, False])
def use_all_features(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def tree_model(use_all_features: bool, seed: int) -> XGBRegressor:
    rng = np.random.default_rng(seed + 2 * N_SEEDS)
    X_train = rng.normal(size=(M, N)).astype(np.float32)
    y_train = rng.normal(size=(M,)).astype(np.float32)

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


@pytest.fixture
def S(tree_model: XGBRegressor, X: NDArray) -> NDArray:
    explainer = Explainer(tree_model)
    S = explainer(X)
    return S.values


@pytest.fixture
def A(S: NDArray) -> NDArray:
    A = S.T @ S
    return A


@pytest.fixture
def b(S: NDArray, y: NDArray) -> NDArray:
    b = S.T @ y
    return b


@pytest.fixture
def y_sq(y: NDArray) -> float:
    return np.square(np.linalg.norm(y))


@pytest.fixture(params=range(N))
def idx(request: pytest.FixtureRequest) -> int:
    return request.param


class TestShapValues:
    def test_shape(self, tree_model: XGBRegressor, X: NDArray) -> None:
        S = shap_values(tree_model, X)
        assert S.shape == (M, N)

    def test_dtype(self, tree_model: XGBRegressor, X: NDArray) -> None:
        S = shap_values(tree_model, X)
        assert S.dtype == np.float32


def test_downdate(S, y, y_sq, idx: int) -> None:
    features = np.array(FEATURES)
    num_unused_features = 0
    state = initial_state(S, y, features, num_unused_features)

    A = state.A
    residual_dof = state.residual_dof

    n = A.shape[0]
    full_rank = np.linalg.matrix_rank(A) == n
    if not full_rank:
        pytest.skip("Matrix A is rank deficient")

    state_down = downdate(state, idx)

    A_down = state_down.A
    b_down = state_down.b
    features_down = state_down.features
    A_inv_down = state_down.A_inv
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
    assert A_down.dtype == np.float32
    assert b_down.dtype == np.float32
    assert features_down.dtype.kind == "U"
    assert A_inv_down.dtype == np.float32
    assert beta_down.dtype == np.float32
    assert rss_down.dtype == np.float32

    # formulas / properties
    assert np.allclose(A_inv_down, np.linalg.pinv(A_down), atol=tol, rtol=tol)
    assert np.allclose(beta_down, A_inv_down @ b_down, atol=tol, rtol=tol)
    assert np.allclose(rss_down, y_sq - np.dot(b_down, beta_down), atol=tol, rtol=tol)
    assert residual_dof_down == residual_dof + 1


def test_ols(S: NDArray, y: NDArray, A: NDArray, b: NDArray, y_sq: float) -> None:
    def ols_statsmodels(S: NDArray, y: NDArray, features: list[str]) -> pd.DataFrame:
        df_S = pd.DataFrame(S, columns=features)
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
        return df

    state = initial_state(S, y, FEATURES, 0)
    df_0 = ols(state)
    df_1 = ols_statsmodels(S, y, FEATURES)

    df_1[T_VALUE] = np.where(df_1[COEFFICIENT].abs() < 1e-10, np.nan, df_1[T_VALUE])
    df_1[STAT_SIGNIFICANCE] = np.where(df_1[COEFFICIENT].abs() < 1e-10, np.nan, df_1[STAT_SIGNIFICANCE])

    pd.testing.assert_frame_equal(df_0, df_1, check_dtype=False, rtol=tol, atol=tol)


def test_flash_select(tree_model: XGBRegressor, X: NDArray, y: NDArray) -> None:
    df_flash_select = flash_select(tree_model, X, y, FEATURES)

    X_df = pd.DataFrame(X, columns=FEATURES)
    y_df = pd.Series(y, name="target")
    df_shap_select = shap_select(tree_model, X_df, y_df, task="regression", alpha=0.0)
    df_shap_select = df_shap_select.sort_values(by=[T_VALUE, FEATURE_NAME], ascending=[False, True])

    pd.testing.assert_frame_equal(df_flash_select, df_shap_select, check_dtype=False, rtol=tol, atol=tol)
