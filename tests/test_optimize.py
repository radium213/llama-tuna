import math

import pytest

from llama_tuna.optimize import (
    OptimizeFailure,
    binary_search,
    fibonacci_search,
    grid_search,
)


class TestBinarySearch:
    @pytest.mark.parametrize("lengths", [100])
    def test_finds_all_targets(self, lengths: int) -> None:
        for length in range(lengths):
            values = list(range(length))
            for target in values:
                assert binary_search(lambda x, t=target: x <= t, values) == target

    @pytest.mark.parametrize("values, exc", [
        ([], OptimizeFailure),
        ([math.inf], OptimizeFailure),
    ])
    def test_raises_on_invalid(self, values: list[float], exc: type[BaseException]) -> None:
        with pytest.raises(exc):
            binary_search(lambda x: x <= 0, values)


class TestFibonacciSearch:
    @pytest.mark.parametrize("lengths", [100])
    def test_finds_all_targets(self, lengths: int) -> None:
        for length in range(lengths):
            values = list(range(length))
            for target in values:
                assert fibonacci_search(lambda x, t=target: (x - t) ** 2.0, values) == target

    @pytest.mark.parametrize("values, exc", [
        ([], OptimizeFailure),
        ([math.inf], OptimizeFailure),
    ])
    def test_raises_on_invalid(self, values: list[float], exc: type[BaseException]) -> None:
        with pytest.raises(exc):
            fibonacci_search(lambda x: x, values)


class TestGridSearch:
    @pytest.mark.parametrize("lengths", [100])
    def test_finds_all_targets(self, lengths: int) -> None:
        for length in range(lengths):
            values = list(range(length))
            for target in values:
                assert grid_search(lambda x, t=target: (x - t) ** 2.0, values) == target

    @pytest.mark.parametrize("values, exc", [
        ([], OptimizeFailure),
        ([math.inf], OptimizeFailure),
    ])
    def test_raises_on_invalid(self, values: list[float], exc: type[BaseException]) -> None:
        with pytest.raises(exc):
            grid_search(lambda x: x, values)
