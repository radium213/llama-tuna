import csv
import io
import math
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress

from llama_tuna.schema import ModelParams


def _set_oom_score() -> None:
    with (
        suppress(FileNotFoundError, PermissionError),
        open("/proc/self/oom_score_adj", "w") as f,
    ):
        f.write(str(1000))


def _subprocess(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    preexec = _set_oom_score if sys.platform != "win32" else None
    return subprocess.run(
        cmd, check=False, capture_output=True, text=True, preexec_fn=preexec
    )


class BenchRunner:
    binary: str
    options: list[str]

    def __init__(
        self,
        binary: str,
        repetitions: int = 3,
        no_warmup: bool = True,
    ):
        self.binary = binary
        self.options = ["-o", "csv", "-r", str(repetitions)]
        if no_warmup:
            self.options.append("--no-warmup")

    def __call__(
        self, params: ModelParams, d: int = 0, p: int = 512, n: int = 128
    ) -> float:
        cmd = (
            params.to_cli_list(self.binary, ("c",))
            + ["-d", str(d), "-p", str(p), "-n", str(n)]
            + self.options
        )

        result = _subprocess(cmd)

        if result.returncode != 0:
            return math.inf

        def get_results(rows: list[dict[str, str]], filter: str) -> float:
            r = [row for row in rows if int(row[filter]) > 0]
            sum_ns = sum([int(row["avg_ns"]) for row in r])
            return sum_ns / 1e9

        data: list[dict[str, str]] = list(csv.DictReader(io.StringIO(result.stdout)))
        pp_s = get_results(data, "n_prompt")
        tg_s = get_results(data, "n_gen")
        return pp_s + tg_s


class CliRunner:
    binary: str
    options: list[str]

    def __init__(
        self,
        binary: str,
    ):
        self.binary = binary
        self.options = ["-st", "-no-cnv"]

    def __call__(self, params: ModelParams, p: str = "?", n: int = 1) -> bool:
        cmd = (
            params.to_cli_list(self.binary)
            + ["-p", p, "-n", str(n)]
            + self.options
        )

        result = _subprocess(cmd)

        return result.returncode == 0


class OptimizeFailure(Exception):
    """Failure to find any valid value."""


def fibonacci_search(
    func: Callable[[int], float],
    length: int,
    progress_callback: Callable[[int, int], None],
) -> int:
    fib: tuple[int, int, int] = (1, 1, 0)
    k = 2

    while fib[0] < length - 1:
        k += 1
        fib = (fib[0] + fib[1], fib[0], fib[1])
    k_max = k

    def get_section(i: int) -> int:
        return round(low + fib[i] / fib[0] * (high - low))

    def evaluate(i: int, k: int) -> tuple[float, int]:
        f = func(i)
        progress_callback(k_max - k + 1, k_max - 1)
        return f, k - 1

    progress_callback(0, k_max - 1)

    low = 0
    high = length - 1
    a = get_section(2)
    b = get_section(1)
    if a == b:
        a -= 1

    f_a, k = evaluate(a, k)
    f_b, k = evaluate(b, k)

    while k > 1:
        if f_a <= f_b:
            high, b, f_b = b, a, f_a
            a = get_section(2)
            if a == b and a > low:
                a -= 1
            f_a, k = evaluate(a, k)
        else:
            low, a, f_a = a, b, f_b
            b = get_section(1)
            if a == b and b < high:
                b += 1
            f_b, k = evaluate(b, k)
        fib = (fib[1], fib[2], fib[1] - fib[2])

    if f_a == math.inf and f_b == math.inf:
        raise OptimizeFailure("All parameter values fail")
    if f_a < f_b:
        return a
    else:
        return b


def grid_search[T](
    func: Callable[[int], float],
    length: int,
    progress_callback: Callable[[int, int], None],
) -> int:
    progress_callback(0, length)
    i_min = 0
    f_min = math.inf
    for i in range(length):
        f_i = func(i)
        progress_callback(i + 1, length)
        if f_i < f_min:
            i_min = i
            f_min = f_i

    if f_min == math.inf:
        raise OptimizeFailure("All parameter values fail")
    return i_min


def binary_search[T](
    func: Callable[[T], bool],
    values: list[T],
    progress_callback: Callable[[int, int], None]
) -> T:
    max_it = math.ceil(math.log2(len(values)))
    it = 0
    progress_callback(it, max_it)

    low = 0
    high = len(values)
    while low < high:
        mid = low + (high - low) // 2
        if func(values[mid]):
            low = mid + 1
        else:
            high = mid
        it += 1
        progress_callback(it, max_it)
    progress_callback(max_it, max_it)

    if high - 1 < 0:
        raise OptimizeFailure("All parameter values fail")

    return values[high - 1]
