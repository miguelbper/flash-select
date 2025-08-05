import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray
from xgboost import XGBRegressor

from flash_select.flash_select import pinv_rank, shap_values

N = 4
M = 100


st_X = arrays(dtype=np.float32, shape=(M, N), elements=st.floats(-1, 1))
st_y = arrays(dtype=np.float32, shape=(M,), elements=st.floats(-1, 1))
st_S = arrays(dtype=np.float32, shape=(M, N), elements=st.floats(-1, 1))
st_A = arrays(dtype=np.float32, shape=(N, N), elements=st.floats(-1, 1))


@st.composite
def tree_model(draw) -> XGBRegressor:
    X = draw(st_X)
    y = draw(st_y)
    model = XGBRegressor(n_estimators=10, max_depth=3, max_leaves=2**3, random_state=42)
    model.fit(X, y)
    return model


def allclose(A: NDArray, B: NDArray, factor: float = 10) -> bool:
    dtype = A.dtype
    eps = np.finfo(dtype).eps
    scale = max(np.linalg.norm(A, ord=np.inf), np.linalg.norm(B, ord=np.inf))
    tol = eps * scale * factor
    return np.allclose(A, B, atol=tol, rtol=tol)


class TestShapValues:
    @given(tree_model=tree_model(), X=st_X)
    def test_shape(self, tree_model: XGBRegressor, X: NDArray) -> None:
        S, _ = shap_values(tree_model, X)
        assert S.shape == (M, N)

    @given(tree_model=tree_model(), X=st_X)
    def test_dtype(self, tree_model: XGBRegressor, X: NDArray) -> None:
        S, _ = shap_values(tree_model, X)
        assert S.dtype == np.float32

    @given(tree_model=tree_model(), X=st_X)
    def test_output(self, tree_model: XGBRegressor, X: NDArray) -> None:
        S, b = shap_values(tree_model, X)
        y = tree_model.predict(X)
        assert np.allclose(y, b + np.sum(S, axis=1))


@given(A=st_A)
def test_pinv_rank(self, A: NDArray) -> None:
    A_pinv_0, A_rank_0 = pinv_rank(A)
    A_pinv_1 = np.linalg.pinv(A)
    A_rank_1 = np.linalg.matrix_rank(A)
    assert allclose(A_pinv_0, A_pinv_1)
    assert A_rank_0 == A_rank_1
    assert A_pinv_0.dtype == np.float32
