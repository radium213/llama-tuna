import argparse
import shutil
from pathlib import Path
from dataclasses import dataclass, fields


@dataclass
class InputParameters:
    fa: str | None = None
    ctk: str | None = None
    ctv: str | None = None
    c: int | None = None
    t: int | None = None
    ngl: int | None = None
    b: int | None = None
    ub: int | None = None


@dataclass
class Inputs:
    llama_path: Path
    files: list[Path]
    params: InputParameters
    out_format: str


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    group_global = parser.add_argument_group("global settings")
    group_source = group_global.add_mutually_exclusive_group()
    group_source.add_argument(
        "-m",
        type=str,
        metavar="filename",
        help="GGUF model file",
    )
    group_source.add_argument(
        "-md",
        "--models-dir",
        type=str,
        metavar="directory",
        help="directory containing GGUF models",
        dest="md",
    )
    group_global.add_argument(
        "-fa",
        type=str,
        choices=["on", "off"],
        help="use flash attention",
    )
    group_global.add_argument(
        "-ctk",
        type=str,
        choices=["f16", "q8_0", "q4_0"],
        help="cache type K",
    )
    group_global.add_argument(
        "-ctv",
        type=str,
        choices=["f16", "q8_0", "q4_0"],
        help="cache type V",
    )
    group_global.add_argument(
        "-c",
        type=int,
        metavar="context",
        help="desired context length, default: model max",
    )
    group_global.add_argument(
        "-b",
        type=int,
        metavar="batch-size",
        help="batch size",
    )
    group_global.add_argument(
        "-ub",
        type=int,
        metavar="ubatch-size",
        help="microbatch size",
    )
    group_global.add_argument(
        "-o",
        type=str,
        choices=["cli", "ini"],
        help="output format: llama-server CLI string or ini",
        default="cli",
    )
    group_global.add_argument(
        "--llama-cpp-path",
        type=str,
        metavar="path",
        help="path to llama.cpp binaries",
        dest="llama_path",
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
    return parser


def load_inputs() -> Inputs:
    parser = create_arg_parser()
    args = parser.parse_args()

    def validate_dependency(cmd: str):
        if not shutil.which(cmd, path=args.llama_path):
            parser.error(
                f"{cmd} not found, make sure llama.cpp is installed and in PATH, or supply the path with --llama-cpp-path"
            )
    
    validate_dependency("llama-bench")
    validate_dependency("llama-cli")

    src_f: str | None = args.m
    src_d: str | None = args.md

    if not (bool(src_f) ^ bool(src_d)):
        parser.error(
            "Provide either -m <file> or -md <directory>"
        )

    files: list[Path] = []
    if src_f:
        f = Path(src_f)
        if not f.exists():
            parser.error(f"{f} does not exist.")
        if f.is_dir():
            parser.error(f"{f} is a directory, use -md.")
        abs_f = f.resolve()
        files.append(abs_f)
    if src_d:
        d = Path(src_d)
        if not d.exists():
            parser.error(f"{d} does not exist.")
        if not d.is_dir():
            parser.error(f"{d} is not a directory, use -m.")
        gguf_files = [
            f.resolve()
            for f in d.iterdir()
            if f.is_file() and f.suffix.lower() == ".gguf"
        ]
        if not gguf_files:
            parser.error(f"{d} has no .gguf files.")
        files.extend(gguf_files)

    param_fields = [f.name for f in fields(InputParameters)]
    kwargs = {
        k: v for k, v in vars(args).items() if k in param_fields
    }
    return Inputs(
        Path(args.llama_path), files, InputParameters(**kwargs), args.o
    )
