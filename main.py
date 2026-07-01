import argparse
import os
import struct
import io
import subprocess
import csv
from dataclasses import dataclass


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
        if not is_valid_gguf(args.m):
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
        valid_files = filter(is_valid_gguf, filter(os.path.isfile, contents))
        files.extend(valid_files)
    kwargs = {k: v for k, v in vars(args).items() if k not in ["m", "md"]}
    return Inputs(files, **kwargs)


def is_valid_gguf(path: str) -> bool:
    with open(path, "rb") as file:
        magic_bytes = file.read(4)
        if len(magic_bytes) < 4:
            return False
        [magic] = struct.unpack("<4s", magic_bytes)
        return magic == b"GGUF"


GGUF_VERSION = 3
GGUF_NTYPES = 13
GGUF_TYPES = "BbHhIif?s_Qqd"
GGUF_SIZES = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8]
GGUF_Value = int | float | bool | str | list


def read_gguf_type(f: io.BufferedReader) -> (str, int):
    type_id: int = struct.unpack("<I", f.read(4))[0]
    if type_id >= GGUF_NTYPES:
        raise TypeError("Unknown metadata type.")
    value_type = GGUF_TYPES[type_id]
    value_size = GGUF_SIZES[type_id]
    return value_type, value_size


def read_gguf_string(f: io.BufferedReader) -> str:
    length: int = struct.unpack("<Q", f.read(8))[0]
    b: bytes = struct.unpack(f"<{length}s", f.read(length))[0]
    return b.decode("utf-8")


def read_gguf_array(f: io.BufferedReader) -> list[GGUF_Value]:
    value_type, value_size = read_gguf_type(f)
    length: int = struct.unpack("<Q", f.read(8))[0]
    return [read_gguf_value(f, value_type, value_size) for _ in range(length)]


def read_gguf_value(
    f: io.BufferedReader, value_type: str, value_size: int
) -> GGUF_Value:
    if value_type == "s":
        return read_gguf_string(f)
    if value_type == "_":
        return read_gguf_array(f)
    return struct.unpack(f"<{value_type}", f.read(value_size))[0]


def read_gguf_kv(f: io.BufferedReader) -> (str, GGUF_Value):
    key = read_gguf_string(f)
    value_type, value_size = read_gguf_type(f)
    value = read_gguf_value(f, value_type, value_size)
    return key, value


def read_gguf_metadata(path: str):
    with open(path, "rb") as f:
        [magic, version, tensor_count, metadata_kv_count] = struct.unpack(
            "<4sIQQ", f.read(24)
        )
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file.")
        if version != GGUF_VERSION:
            print(
                f"Detected GGUF version {version}. This script was only tested with version {GGUF_VERSION} and may not work correctly."
            )
        metadata = {}
        for _ in range(metadata_kv_count):
            key, value = read_gguf_kv(f)
            metadata[key] = value
        return metadata


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


def main():
    inputs = parse_inputs()
    files = sorted(inputs.files, key=os.path.getsize)
    for file in files:
        metadata = read_gguf_metadata(file)
        model_type = metadata.get("general.type", "")
        architecture = metadata.get("general.architecture", "")
        if model_type != "model":
            print(f"Skipping {file} - incompatible type: {model_type}")
            continue
        if "bert" in architecture or architecture in ["whisper", "clip", "siglip"]:
            print(f"Skipping {file} - incompatible architecture: {architecture}")
            continue
        context_length = metadata[f"{architecture}.context_length"]
        print(file)
        ncpu = os.cpu_count() or 1
        runner = BenchRunner("llama-bench", file, ncpu, no_warmup=True)
        data = runner.run_test(t=12, p=512, n=0, ngl=0)
        print(data)


if __name__ == "__main__":
    main()
