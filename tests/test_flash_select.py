import numpy as np
import pytest
from numpy.typing import NDArray
from shap import Explainer
from xgboost import XGBRegressor

from flash_select.flash_select import downdate, pinv_rank, shap_values

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
        S, _ = shap_values(tree_model, X)
        assert S.shape == (M, N)

    def test_dtype(self, tree_model: XGBRegressor, X: NDArray) -> None:
        S, _ = shap_values(tree_model, X)
        assert S.dtype == np.float32

    def test_output(self, tree_model: XGBRegressor, X: NDArray) -> None:
        S, b = shap_values(tree_model, X)
        y = tree_model.predict(X)
        assert np.allclose(y, b + np.sum(S, axis=1), atol=tol, rtol=tol)


def test_pinv_rank(A: NDArray) -> None:
    A_pinv_0, A_rank_0 = pinv_rank(A)
    A_pinv_1 = np.linalg.pinv(A)
    A_rank_1 = np.linalg.matrix_rank(A)
    assert np.allclose(A_pinv_0, A_pinv_1, atol=tol, rtol=tol)
    assert A_rank_0 == A_rank_1
    assert A_pinv_0.dtype == np.float32


def test_downdate(A: NDArray, b: NDArray, idx: int) -> None:
    A_pinv = np.linalg.pinv(A)
    full_rank = np.linalg.matrix_rank(A) == A.shape[0]

    A_down, _, _, A_pinv_down, A_rank_down, _ = downdate(A, b, FEATURES, A_pinv, full_rank, idx)

    assert A_down.shape == (N - 1, N - 1)
    assert A_down.dtype == np.float32

    assert A_pinv_down.shape == (N - 1, N - 1)
    assert A_pinv_down.dtype == np.float32

    assert A_rank_down == np.linalg.matrix_rank(A_down)
    assert np.allclose(A_pinv_down, np.linalg.pinv(A_down), atol=tol, rtol=tol)
