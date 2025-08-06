from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray


class Backend(ABC):
    @abstractmethod
    def from_numpy(self, X: NDArray) -> Any:
        pass

    @abstractmethod
    def to_numpy(self, X: Any) -> NDArray:
        pass

    @abstractmethod
    def ones(self, shape: tuple[int, ...]) -> Any:
        pass

    @abstractmethod
    def sqrt(self, X: Any) -> Any:
        pass

    @abstractmethod
    def square(self, X: Any) -> Any:
        pass

    @abstractmethod
    def abs(self, X: Any) -> Any:
        pass

    @abstractmethod
    def dot(self, X: Any, Y: Any) -> Any:
        pass

    @abstractmethod
    def outer(self, X: Any, Y: Any) -> Any:
        pass

    @abstractmethod
    def matmul(self, X: Any, Y: Any) -> Any:
        pass

    @abstractmethod
    def pinv(self, X: Any) -> Any:
        pass

    @abstractmethod
    def norm(self, X: Any) -> Any:
        pass

    @abstractmethod
    def matrix_rank(self, X: Any) -> Any:
        pass

    @abstractmethod
    def diag(self, X: Any) -> Any:
        pass
