import io
import subprocess
import csv
from pathlib import Path
from contextlib import suppress
from dataclasses import dataclass, asdict, replace
from collections.abc import Callable, Iterable
from typing import Generator, Literal, Protocol
from tqdm import tqdm


@dataclass
class Parameters:
    fa: Literal["on", "off"] = "on"
    ctk: Literal["f16", "q8_0", "q4_0"] = "f16"
    ctv: Literal["f16", "q8_0", "q4_0"] = "f16"
    d: int = 0
    p: int = 512
    n: int = 128
    t: int = 1
    ngl: int = 0
    b: int = 2048
    ub: int = 512


class BenchRunner:
    binary: str
    options: list[str]

    def __init__(
        self,
        binary: str,
        model: Path | str,
        repetitions: int = 3,
        no_warmup: bool = True,
    ):
        self.binary = binary
        self.options = ["-m", str(model), "-o", "csv", "-r", str(repetitions)]
        if no_warmup:
            self.options.append("--no-warmup")

    def run_test(self, params: Parameters) -> float:
        options: list[str] = []
        for k, v in asdict(params).items():
            if v is not None:
                options.extend([f"-{k}", str(v)])
        cmd = [self.binary] + options + self.options

        def set_oom_score():
            with suppress(FileNotFoundError, PermissionError):
                with open("/proc/self/oom_score_adj", "w") as f:
                    f.write(str(1000))

        result = subprocess.run(
            cmd, preexec_fn=set_oom_score, capture_output=True, text=True
        )

        if result.returncode != 0:
            return 0.0
        data = csv.DictReader(io.StringIO(result.stdout))
        avg_ts = [float(row["avg_ts"]) for row in data]
        return sum(avg_ts) / len(avg_ts)


class Strategy(Protocol):
    def __iter__(self) -> Generator[tuple[int, float], None, None]: ...

    def __len__(self) -> int: ...

    def result(self) -> int: ...


class FibonacciStrategy:
    func: Callable[[int], float]
    low: int
    high: int
    fib: tuple[int, int, int]
    k: int
    _result: int

    def __init__(self, func: Callable[[int], float], low: int, high: int):
        self.func = func
        self.low = low
        self.high = high
        fib = (1, 1, 0)
        k = 2
        length = high - low
        while fib[0] < length:
            k += 1
            fib = (fib[0] + fib[1], fib[0], fib[1])
        self.fib = fib
        self.k = k

    def __iter__(self) -> Generator[tuple[int, float], None, None]:
        func, fib, low, high = self.func, self.fib, self.low, self.high

        def get_section(i: int) -> int:
            return int(round(low + fib[i] / fib[0] * (high - low)))

        a = get_section(2)
        b = get_section(1)
        if a == b:
            a -= 1

        self.k -= 1
        f_a = func(a)
        yield a, f_a
        self.k -= 1
        f_b = func(b)
        yield b, f_b

        while self.k > 1:
            self.k -= 1
            if f_a > f_b:
                high, b, f_b = b, a, f_a
                a = get_section(2)
                if a == b and a > low:
                    a -= 1
                f_a = func(a)
                yield a, f_a
            else:
                low, a, f_a = a, b, f_b
                b = get_section(1)
                if a == b and b < high:
                    b += 1
                f_b = func(b)
                yield b, f_b
            fib = (fib[1], fib[2], fib[1] - fib[2])

        if f_a > f_b:
            self._result = a
        else:
            self._result = b

    def __len__(self) -> int:
        return self.k - 1

    def result(self) -> int:
        return self._result


class GridStrategy:
    func: Callable[[int], float]
    low: int
    high: int
    _result: int

    def __init__(self, func: Callable[[int], float], low: int, high: int):
        self.func = func
        self.low = low
        self.high = high

    def __iter__(self) -> Generator[tuple[int, float], None, None]:
        func, low, high = self.func, self.low, self.high

        i_max = -1
        f_max = -1.0
        for i in range(low, high + 1):
            f_i = func(i)
            yield i, f_i
            if f_i > f_max:
                i_max = i
                f_max = f_i
        self._result = i_max

    def __len__(self) -> int:
        return self.high - self.low + 1

    def result(self) -> int:
        return self._result


class CachedFunction[T, U]:
    func: Callable[[T], U]
    cache: dict[T, U]

    def __init__(self, func: Callable[[T], U]):
        self.func = func
        self.cache = {}

    def invoke(self, x: T) -> U:
        result = self.cache.get(x, None)
        if result is None:
            result = self.func(x)
            self.cache[x] = result
        return result


def optimize[T](
    runner: BenchRunner,
    param: str,
    search_space: Iterable[T],
    fixed_params: Parameters,
    strat: Literal["auto", "fib", "grid"] = "auto",
) -> T:
    values = list(search_space)
    func = CachedFunction[int, float](
        lambda i: runner.run_test(replace(fixed_params, **{param: values[i]}))
    )

    strategy: Strategy
    n = len(values)
    if strat == "grid" or strat == "auto" and n <= 3:
        strategy = GridStrategy(func.invoke, 0, n - 1)
    else:
        strategy = FibonacciStrategy(func.invoke, 0, n - 1)

    with tqdm(strategy, f"[ {param:>3} ]") as tq:
        tq.set_postfix_str("?t/s")
        for _, f in tq:
            if f >= 1.0:
                tq.set_postfix_str(f"{f:.0f}t/s")
            elif f == 0.0:
                tq.set_postfix_str("0t/s")
            else:
                tq.set_postfix_str(f"{1.0 / f:.2f}s/t")
        return values[strategy.result()]
