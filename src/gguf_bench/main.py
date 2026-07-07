import os
import io
import subprocess
import csv
import sys
from pathlib import Path
from contextlib import suppress
from dataclasses import dataclass, asdict, replace
from collections.abc import Callable, Iterable
from typing import Literal
from tqdm import tqdm
from gguf_bench.config import load_inputs
from gguf_bench.gguf import GGUFParser


@dataclass
class Parameters:
    fa: Literal["on", "off"] = "on"
    ctk: Literal["f16", "q8_0", "q4_0"] = "f16"
    ctv: Literal["f16", "q8_0", "q4_0"] = "f16"
    d: int = 0
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
        self.options = [
            "-m", str(model),
            "-o", "csv",
            "-r", str(repetitions)
        ]
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

        result = subprocess.run(cmd, preexec_fn=set_oom_score, capture_output=True, text=True)

        if result.returncode != 0:
            return 0.0
        data = csv.DictReader(io.StringIO(result.stdout))
        avg_ts = [float(row["avg_ts"]) for row in data]
        return sum(avg_ts) / len(avg_ts)


def fibonacci_search(f: Callable[[int], float], low: int, high: int):
    def fib(n: int):
        F = [0, 1]
        while F[-1] < n:
            F.append(F[-1] + F[-2])
        return F

    length = high - low + 1
    F = fib(length)
    n = len(F) - 1
    offset = low - 1
    
    tq = tqdm(total=n - 1)

    a = min(offset + F[n - 2], high)
    b = min(offset + F[n - 1], high)
    f_a = f(a)
    tq.update()
    f_b = f(b)
    tq.update()

    while n > 3:
        n -= 1
        if f_a < f_b:
            offset = a
            a, f_a = b, f_b
            b = min(offset + F[n - 1], high)
            f_b = f(b)
        else:
            b, f_b = a, f_a
            a = min(offset + F[n - 2], high)
            f_a = f(a)
        tq.update()
    tq.close()

    if f_a > f_b:
        return a
    return b


class CachedFunction[T, U]:
    func: Callable[[T], U]
    cache: dict[T, U]

    def __init__(self, func: Callable[[T], U]):
        self.func = func
        self.cache = {}

    def invoke(self, x: T) -> U:
        result = self.cache.get(x, None)
        if not result:
            result = self.func(x)
            self.cache[x] = result
        return result


def get_metadata(file: Path):
    gguf_meta, gguf_version, tensor_count = GGUFParser(file).read_metadata()
    model_type = gguf_meta.get("general.type", "")
    architecture = gguf_meta.get("general.architecture", "")
    compatible = True
    if gguf_version != GGUFParser.VERSION:
        print(f"{file} - incompatible GGUF version: {gguf_version}")
        compatible = False
    if model_type != "model":
        print(f"{file} - incompatible type: {model_type}")
        compatible = False
    if (
        not architecture
        or "bert" in architecture
        or architecture in ["whisper", "clip", "siglip"]
    ):
        print(f"{file} - incompatible architecture: {architecture}")
        compatible = False
    metadata = {
        "gguf": gguf_meta,
        "tensor_count": tensor_count,
        "compatible": compatible,
    }
    return metadata


def get_smallest_model(files: list[Path]) -> Path | None:
    files = sorted(files, key=os.path.getsize)
    for file in files:
        metadata = get_metadata(file)
        if not metadata["compatible"]:
            continue
        return file
    return None


class Optimizer[T]:
    runner: BenchRunner
    param: str
    search_space: list[T]
    fixed: Parameters

    def __init__(self, runner: BenchRunner, param: str, search_space: Iterable[T], fixed_params: Parameters):
        self.runner = runner
        self.param = param
        self.search_space = list(search_space)
        self.fixed = fixed_params
    
    def search(self) -> T:
        func = CachedFunction[int, float](
            lambda i: self.runner.run_test(
                replace(self.fixed, **{self.param: self.search_space[i]})
            )
        )

        i_best = fibonacci_search(func.invoke, 0, len(self.search_space) - 1)

        return self.search_space[i_best]


def main():
    inputs = load_inputs()
    gguf_files = [f for f in inputs.files if GGUFParser(f).is_valid()]
    if not inputs.t:
        print("Finding optimal -t (CPU threads) ...")
        ncpu = os.cpu_count() or 1
        print(f"Detected CPU core count: {ncpu}")
        min_model = get_smallest_model(inputs.files)
        if min_model is None:
            sys.exit("No valid model for test.")
        opt = Optimizer(
            BenchRunner(inputs.binary, min_model),
            "t",
            range(1, ncpu + 1),
            Parameters(ngl=0)
        )
        inputs.t = opt.search()
    for file in gguf_files:
        metadata = get_metadata(file)
        if not metadata["compatible"]:
            continue
        architecture: str = metadata["gguf"]["general.architecture"]
        layers: int = metadata["gguf"][f"{architecture}.block_count"]
        print(f"Benchmarking {file} ...")
        bench = BenchRunner(inputs.binary, file)
        ngl = Optimizer(
            bench,
            "ngl",
            range(0, layers + 1),
            Parameters(t=inputs.t)
        ).search()
        print(f"Found optimal -ngl {ngl}")
        b = Optimizer(
            bench,
            "b",
            range(4096, 0, -512),
            Parameters(t=inputs.t, ngl=ngl)
        ).search()
        print(f"Found optimal -b {b}")


if __name__ == "__main__":
    main()
