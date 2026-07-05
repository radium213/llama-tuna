import os
import io
import subprocess
import csv
from pathlib import Path
from collections.abc import Callable
from gguf_bench.config import load_inputs
from gguf_bench.gguf import GGUFParser


class BenchRunner:
    def __init__(
        self,
        binary: str,
        model: str,
        threads: int,
        flash_attn: str = "on",
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        repetitions: int = 3,
        no_warmup: bool = False,
    ):
        self.binary = binary
        self.options = {}
        self.options["-m"] = model
        self.options["-o"] = "csv"
        self.options["-t"] = str(threads)
        self.options["-fa"] = flash_attn
        self.options["-ctk"] = cache_type_k
        self.options["-ctv"] = cache_type_v
        self.options["-r"] = str(repetitions)
        if no_warmup:
            self.options["--no-warmup"] = ""

    def run_llama_bench(self, **kwargs) -> str | None:
        extra_options = {f"-{k}": v for k, v in kwargs.items()}
        options = self.options | extra_options
        args = [str(e) for item in options.items() for e in item if e != ""]
        cmd = [self.binary] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        return result.stdout

    def run_test(self, **kwargs) -> float:
        result = self.run_llama_bench(**kwargs)
        if result is None:
            return 0.0
        data = csv.DictReader(io.StringIO(result))
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

    a = min(offset + F[n - 2], high)
    b = min(offset + F[n - 1], high)
    f_a = f(a)
    f_b = f(b)

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

    if f_a > f_b:
        return a
    return b


class CachedFunction[T, U]:
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


def find_optimal_t(binary: str, files: list[Path]):
    print("Finding optimal -t (CPU threads) ...")
    t = 0
    files = sorted(files, key=os.path.getsize)
    ncpu = os.cpu_count() or 1
    print(f"Detected CPU core count: {ncpu}")
    for file in files:
        metadata = get_metadata(file)
        if not metadata["compatible"]:
            continue
        print(f"Using model: {file}")
        runner = BenchRunner(binary, file, 1, repetitions=5, no_warmup=True)
        func = CachedFunction(lambda x: runner.run_test(t=x, p=512, n=0, ngl=0))
        t = fibonacci_search(func.invoke, 1, ncpu)
        print(f"Found optimal -t {t}")
        return t
    if t == 0:
        print("No valid model for test.")
        exit(0)
    return t


def find_optimal_ngl(binary: str, file: Path, layers: int, t: int, fa: str, ctk: str, ctv: str):
    runner = BenchRunner(binary, file, t, no_warmup=True)
    func = CachedFunction(
        lambda x: runner.run_test(p=512, n=0, ngl=x, fa=fa, ctk=ctk, ctv=ctv)
    )
    return fibonacci_search(func.invoke, 0, layers)


def find_optimal_b(
    binary: str, file: Path, max_b: int, step: int, ngl: int, t: int, fa: str, ctk: str, ctv: str
):
    runner = BenchRunner(binary, file, t, no_warmup=True)
    test_values = range(max_b, 0, -step)
    func = CachedFunction(
        lambda x: runner.run_test(
            p=max_b, n=0, b=test_values[x], ngl=ngl, fa=fa, ctk=ctk, ctv=ctv
        )
    )
    result = fibonacci_search(func.invoke, 0, len(test_values) - 1)
    return test_values[result]


def main():
    inputs = load_inputs()
    gguf_files = [f for f in inputs.files if GGUFParser(f).is_valid()]
    if not inputs.t:
        inputs.t = find_optimal_t(inputs.binary, gguf_files)
    for file in gguf_files:
        metadata = get_metadata(file)
        if not metadata["compatible"]:
            continue
        architecture = metadata["gguf"]["general.architecture"]
        layers = metadata["gguf"][f"{architecture}.block_count"]
        print(f"Benchmarking {file} ...")
        ngl = find_optimal_ngl(
            inputs.binary, file, layers, inputs.t, inputs.fa, inputs.ctk, inputs.ctv
        )
        print(f"Found optimal -ngl {ngl}")
        b = find_optimal_b(
            inputs.binary, file, 4096, 512, ngl, inputs.t, inputs.fa, inputs.ctk, inputs.ctv
        )
        print(f"Found optimal -b {b}")


if __name__ == "__main__":
    main()
