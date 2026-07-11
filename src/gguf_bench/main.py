import os
import io
import subprocess
import csv
import sys
from pathlib import Path
from contextlib import suppress
from dataclasses import dataclass, asdict, replace
from collections.abc import Callable, Iterable
from typing import Generator, Literal, Protocol
from tqdm import tqdm
from gguf_bench.config import load_inputs
from gguf_bench.gguf import GGUFParser


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


class FibonacciStrategy:
    func: Callable[[int], float]
    low: int
    high: int
    fib: tuple[int, int, int]
    k: int

    def __init__(self, func: Callable[[int], float], low: int, high: int):
        self.func = func
        self.low = low
        self.high = high
        fib = (1, 1, 0)
        k = 3
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
            yield a, f_a
        else:
            yield b, f_b

    def __len__(self) -> int:
        return self.k


def optimize[T](runner: BenchRunner, param: str, search_space: Iterable[T], fixed_params: Parameters) -> T:
    values = list(search_space)
    func = CachedFunction[int, float](
        lambda i: runner.run_test(
            replace(fixed_params, **{param: values[i]})
        )
    )

    strategy = FibonacciStrategy(func.invoke, 0, len(values) - 1)
    i_best = 0
    with tqdm(strategy, f"[ {param:>3} ]") as tq:
        tq.set_postfix_str("?t/s")
        for i, f in tq:
            i_best = i
            if f >= 1.0:
                tq.set_postfix_str(f"{f:.0f}t/s")
            else:
                tq.set_postfix_str(f"{1.0/f:.2f}s/t")

    return values[i_best]


def main():
    inputs = load_inputs()
    gguf_files = [f for f in inputs.files if GGUFParser(f).is_valid()]
    if not inputs.t:
        ncpu = os.cpu_count() or 1
        min_model = get_smallest_model(inputs.files)
        if min_model is None:
            sys.exit("No valid model for test.")
        inputs.t = optimize(
            BenchRunner(inputs.binary, min_model),
            "t",
            range(1, ncpu + 1),
            Parameters(ngl=0, n=0),
        )
        print(f"Found t: {inputs.t}")
    for file in gguf_files:
        metadata = get_metadata(file)
        if not metadata["compatible"]:
            continue
        architecture: str = metadata["gguf"]["general.architecture"]
        layers: int = metadata["gguf"][f"{architecture}.block_count"]
        print(f"Benchmarking {file} ...")
        bench = BenchRunner(inputs.binary, file)
        ngl = optimize(
            bench,
            "ngl",
            range(0, layers + 1),
            Parameters(t=inputs.t),
        )
        print(f"Found optimal -ngl {ngl}")


if __name__ == "__main__":
    main()
