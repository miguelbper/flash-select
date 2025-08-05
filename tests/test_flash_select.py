import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from flash_select.flash_select import pinv_rank

N = 4
M = 100


def allclose(A: NDArray, B: NDArray, factor: float = 10) -> bool:
    dtype = A.dtype
    eps = np.finfo(dtype).eps
    scale = max(np.linalg.norm(A, ord=np.inf), np.linalg.norm(B, ord=np.inf))
    tol = eps * scale * factor
    return np.allclose(A, B, atol=tol, rtol=tol)


class TestShapValues:
    def test_shape(self) -> None:
        pass

    def test_dtype(self) -> None:
        pass

    def test_output(self) -> None:
        pass


class TestPinvRank:
    @given(A=arrays(dtype=np.float32, shape=(N, N), elements=st.floats(-1, 1)))
    def test_pinv_rank(self, A) -> None:
        A_pinv_0, A_rank_0 = pinv_rank(A)
        A_pinv_1 = np.linalg.pinv(A)
        A_rank_1 = np.linalg.matrix_rank(A)
        assert allclose(A_pinv_0, A_pinv_1)
        assert A_rank_0 == A_rank_1
        assert A_pinv_0.dtype == np.float32
