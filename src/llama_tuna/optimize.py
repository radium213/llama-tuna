import io
import subprocess
import csv
import math
from functools import cache
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
            return math.inf

        def get_results(rows: list[dict[str, str]], filter: str) -> float:
            r = [row for row in rows if int(row[filter]) > 0]
            sum_ns = sum([int(row["avg_ns"]) for row in r])
            return sum_ns / 1e+9

        data: list[dict[str, str]] = list(csv.DictReader(io.StringIO(result.stdout)))
        pp_s = get_results(data, "n_prompt")
        tg_s = get_results(data, "n_gen")
        return pp_s + tg_s


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
        func, low, high, fib, k = self.func, self.low, self.high, self.fib, self.k

        def get_section(i: int) -> int:
            return int(round(low + fib[i] / fib[0] * (high - low)))

        a = get_section(2)
        b = get_section(1)
        if a == b:
            a -= 1

        k -= 1
        f_a = func(a)
        yield a, f_a
        k -= 1
        f_b = func(b)
        yield b, f_b

        while k > 1:
            k -= 1
            if f_a < f_b:
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

        if f_a < f_b:
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

        i_min = 0
        f_min = math.inf
        for i in range(low, high + 1):
            f_i = func(i)
            yield i, f_i
            if f_i < f_min:
                i_min = i
                f_min = f_i
        self._result = i_min

    def __len__(self) -> int:
        return self.high - self.low + 1

    def result(self) -> int:
        return self._result


class OptimizeFailure(Exception):
    """Failure to find any valid value."""


def optimize[T](
    runner: BenchRunner,
    param: str,
    search_space: Iterable[T],
    fixed_params: Parameters,
    strat: Literal["auto", "fib", "grid"] = "auto",
) -> T:
    values = list(search_space)

    def run(i: int) -> float:
        return runner.run_test(replace(fixed_params, **{param: values[i]}))

    func = cache(run)

    strategy: Strategy
    n = len(values)
    if strat == "grid" or strat == "auto" and n <= 3:
        strategy = GridStrategy(func, 0, n - 1)
    else:
        strategy = FibonacciStrategy(func, 0, n - 1)

    with tqdm(strategy, f"[ {param:>3} ]") as tq:
        tq.set_postfix_str("?t/s")
        min_s = math.inf

        total_s = 0.0
        total_tok = 0
        tok_step = fixed_params.p + fixed_params.n
        for _, s in tq:
            min_s = min(min_s, s)
            if s < math.inf:
                total_s += s
                total_tok += tok_step
            rate = total_tok / total_s if total_s != 0.0 else 0.0
            if rate >= 1.0:
                tq.set_postfix_str(f"{rate:.0f}t/s")
            elif rate == 0.0:
                tq.set_postfix_str("0t/s")
            else:
                tq.set_postfix_str(f"{1.0 / rate:.2f}s/t")

        if min_s == math.inf:
            raise OptimizeFailure("All parameter values fail")
        return values[strategy.result()]
