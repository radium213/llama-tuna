import math
from contextlib import nullcontext

import pytest

from llama_tuna.optimize import (
    OptimizeFailure,
    binary_search,
    fibonacci_search,
    grid_search,
)


def expect(
    exception: type[BaseException] | None = None,
) -> pytest.RaisesExc[BaseException] | nullcontext[None]:
    return pytest.raises(exception) if exception else nullcontext()


@pytest.mark.parametrize(
    "values, target, exc",
    [
        ([1, 2, 3, 4, 5], 4, None),
        ([99], 99, None),
        ([1], 0, OptimizeFailure),
        ([], 0, OptimizeFailure),
    ],
)
def test_binary_search[T: float](
    values: list[T], target: T, exc: type[BaseException] | None
) -> None:
    with expect(exc):
        assert binary_search(lambda x: x <= target, values) == target


@pytest.mark.parametrize(
    "values, target, exc",
    [
        (range(100), 42, None),
        (range(100), 0, None),
        (range(100), 99, None),
        ([1], 1, None),
        ([math.inf], 0, OptimizeFailure),
        ([], 0, OptimizeFailure),
    ],
)
def test_fibonacci_search[T: float](
    values: list[T], target: T, exc: type[BaseException] | None
) -> None:
    with expect(exc):
        assert fibonacci_search(lambda x: (x - target) ** 2.0, values) == target


@pytest.mark.parametrize(
    "values, target, exc",
    [
        (range(100), 42, None),
        (range(100), 0, None),
        (range(100), 99, None),
        ([1], 1, None),
        ([math.inf], 0, OptimizeFailure),
        ([], 0, OptimizeFailure),
    ],
)
def test_grid_search[T: float](
    values: list[T], target: T, exc: type[BaseException] | None
) -> None:
    with expect(exc):
        assert grid_search(lambda x: (x - target) ** 2.0, values) == target
