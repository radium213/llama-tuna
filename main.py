import argparse
import os
import struct
import io
import subprocess
import csv
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any


@dataclass
class Inputs:
    files: list[str]
    fa: str
    ctk: str
    ctv: str
    d: int
    t: int | None
    ngl: int | None
    b: int | None
    ub: int | None


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    group_global = parser.add_argument_group("global settings")
    group_source = group_global.add_mutually_exclusive_group(required=True)
    group_source.add_argument(
        "-m",
        type=str,
        metavar="filename",
        help="GGUF model file",
    )
    group_source.add_argument(
        "-md",
        type=str,
        metavar="directory",
        help="directory containing GGUF models",
    )
    group_global.add_argument(
        "-fa",
        type=str,
        choices=["on", "off"],
        help="use flash attention",
        default="on",
    )
    group_global.add_argument(
        "-ctk",
        type=str,
        choices=["f16", "q8_0", "q4_0"],
        help="cache type K",
        default="f16",
    )
    group_global.add_argument(
        "-ctv",
        type=str,
        choices=["f16", "q8_0", "q4_0"],
        help="cache type V",
        default="f16",
    )
    group_global.add_argument(
        "-d",
        type=int,
        metavar="depth",
        help="context depth for test",
        default=0,
    )
    group_test = parser.add_argument_group(
        "test parameters", "if supplied, will not be searched for"
    )
    group_test.add_argument(
        "-t",
        type=int,
        metavar="threads",
        help="number of threads for CPU inference",
    )
    group_test.add_argument(
        "-ngl",
        type=int,
        metavar="n-gpu-layers",
        help="number of layers offloaded to GPU",
    )
    group_test.add_argument(
        "-b",
        type=int,
        metavar="batch-size",
        help="batch size",
    )
    group_test.add_argument(
        "-ub",
        type=int,
        metavar="ubatch-size",
        help="microbatch size",
    )
    return parser


def parse_inputs() -> Inputs:
    parser = create_arg_parser()
    args = parser.parse_args()
    files = []
    if args.m:
        if not os.path.exists(args.m):
            raise ValueError(f"{args.m} does not exist")
        if os.path.isdir(args.m):
            raise ValueError(f"{args.m} is a directory")
        if not GGUFParser(args.m).is_valid():
            raise ValueError(f"{args.m} is not a valid GGUF file")
        files.append(os.path.realpath(args.m))
    if args.md:
        if not os.path.exists(args.md):
            raise ValueError(f"{args.md} does not exist")
        if not os.path.isdir(args.md):
            raise ValueError(f"{args.md} is not a directory")
        contents = [
            os.path.realpath(os.path.join(args.md, f)) for f in os.listdir(args.md)
        ]
        valid_files = filter(
            lambda f: GGUFParser(f).is_valid(), filter(os.path.isfile, contents)
        )
        files.extend(valid_files)
    kwargs = {k: v for k, v in vars(args).items() if k not in ["m", "md"]}
    return Inputs(files, **kwargs)


class GGUFParser:
    VERSION = 3
    NTYPES = 13
    TYPES = "BbHhIif?s_Qqd"
    SIZES = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8]
    ValueType = int | float | bool | str | list

    def __init__(self, path: str):
        self.path = path

    def _read_type(self) -> (str, int):
        type_id: int = struct.unpack("<I", self.f.read(4))[0]
        if type_id >= GGUFParser.NTYPES:
            raise TypeError("Unknown metadata type.")
        value_type = GGUFParser.TYPES[type_id]
        value_size = GGUFParser.SIZES[type_id]
        return value_type, value_size

    def _read_string(self) -> str:
        length: int = struct.unpack("<Q", self.f.read(8))[0]
        b: bytes = struct.unpack(f"<{length}s", self.f.read(length))[0]
        return b.decode("utf-8")

    def _read_array(self) -> list[ValueType]:
        value_type, value_size = self._read_type()
        length: int = struct.unpack("<Q", self.f.read(8))[0]
        return [self._read_value(value_type, value_size) for _ in range(length)]

    def _read_value(self, value_type: str, value_size: int) -> ValueType:
        if value_type == "s":
            return self._read_string()
        if value_type == "_":
            return self._read_array()
        return struct.unpack(f"<{value_type}", self.f.read(value_size))[0]

    def _read_kv(self) -> (str, ValueType):
        key = self._read_string()
        value_type, value_size = self._read_type()
        value = self._read_value(value_type, value_size)
        return key, value

    def _read_header(self) -> (str, int, int, int) | None:
        size = 24
        header_bytes = self.f.read(size)
        if len(header_bytes) < size:
            return None
        [magic, version, tensor_count, metadata_kv_count] = struct.unpack(
            "<4sIQQ", header_bytes
        )
        if magic != b"GGUF":
            return None
        return magic, version, tensor_count, metadata_kv_count

    def read_metadata(self) -> dict[str, Any]:
        with open(self.path, "rb") as f:
            self.f = f
            header = self._read_header()
            if not header:
                raise ValueError(f"{self.path} is not a GGUF file.")
            [magic, version, tensor_count, metadata_kv_count] = header
            if version != GGUFParser.VERSION:
                print(
                    f"Detected GGUF version {version}. This script was only tested with version {GGUFParser.VERSION} and may not work correctly."
                )
            metadata = {}
            for _ in range(metadata_kv_count):
                key, value = self._read_kv()
                metadata[key] = value
            return metadata, tensor_count

    def is_valid(self) -> bool:
        with open(self.path, "rb") as f:
            self.f = f
            if not self._read_header():
                return False
            return True


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

    a = offset + F[n - 2]
    b = offset + F[n - 1]
    f_a = f(a)
    f_b = f(b)

    while n > 3:
        n -= 1
        if f_a < f_b:
            offset = a
            a, f_a = b, f_b
            b = offset + F[n - 1]
            f_b = f(b)
        else:
            b, f_b = a, f_a
            a = offset + F[n - 2]
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


def get_metadata(file: str):
    gguf_meta, tensor_count = GGUFParser(file).read_metadata()
    model_type = gguf_meta.get("general.type", "")
    architecture = gguf_meta.get("general.architecture", "")
    compatible = True
    if model_type != "model":
        print(f"Skipping {file} - incompatible type: {model_type}")
        compatible = False
    if (
        not architecture
        or "bert" in architecture
        or architecture in ["whisper", "clip", "siglip"]
    ):
        print(f"Skipping {file} - incompatible architecture: {architecture}")
        compatible = False
    metadata = {
        "gguf": gguf_meta,
        "tensor_count": tensor_count,
        "compatible": compatible,
    }
    return metadata


def find_optimal_t(files: list[str]):
    print("Finding optimal -t (CPU threads) ...")
    t = 0
    files = sorted(files, key=os.path.getsize)
    ncpu = os.cpu_count() or 1
    print(f"Detected CPU core count: {ncpu}")
    for file in files:
        metadata = get_metadata(file)
        if not metadata["compatible"]:
            continue
        print(f"Using smallest model: {file}")
        runner = BenchRunner("llama-bench", file, 1, repetitions=5, no_warmup=True)
        func = CachedFunction(lambda x: runner.run_test(t=x, p=512, n=0, ngl=0))
        t = fibonacci_search(func.invoke, 1, ncpu)
        print(f"Found optimal -t {t}")
        return t
    if t == 0:
        print("No valid model for test.")
        exit(0)
    return t


def find_optimal_ngl(file: str, layers: int, t: int, fa: str, ctk: str, ctv: str):
    runner = BenchRunner("llama-bench", file, t, no_warmup=True)
    func = CachedFunction(
        lambda x: runner.run_test(p=512, n=0, ngl=x, fa=fa, ctk=ctk, ctv=ctv)
    )
    return fibonacci_search(func.invoke, 0, layers)


def find_optimal_b(
    file: str, max_b: int, step: int, ngl: int, t: int, fa: str, ctk: str, ctv: str
):
    runner = BenchRunner("llama-bench", file, t, no_warmup=True)
    test_values = range(max_b, 0, -step)
    func = CachedFunction(
        lambda x: runner.run_test(
            p=512, n=0, b=test_values[x], ngl=ngl, fa=fa, ctk=ctk, ctv=ctv
        )
    )
    result = fibonacci_search(func.invoke, 0, len(test_values))
    return test_values[result]


def main():
    inputs = parse_inputs()
    if not inputs.t:
        inputs.t = find_optimal_t(inputs.files)
    for file in inputs.files:
        metadata = get_metadata(file)
        architecture = metadata["gguf"]["general.architecture"]
        layers = metadata["gguf"][f"{architecture}.block_count"]
        print(f"Benchmarking {file} ...")
        ngl = find_optimal_ngl(
            file, layers, inputs.t, inputs.fa, inputs.ctk, inputs.ctv
        )
        print(f"Found optimal -ngl {ngl}")
        b = find_optimal_b(
            file, 4096, 256, ngl, inputs.t, inputs.fa, inputs.ctk, inputs.ctv
        )
        print(f"Found optimal -b {b}")


if __name__ == "__main__":
    main()
