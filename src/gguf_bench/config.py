import argparse
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from gguf_bench.gguf import GGUFParser


@dataclass
class Inputs:
    binary: str
    files: list[Path]
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
    group_global.add_argument(
        "--llama-bench",
        type=str,
        metavar="path",
        help="path to llama-bench binary",
        dest="binary",
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


def load_inputs() -> Inputs:
    parser = create_arg_parser()
    args = parser.parse_args()

    default_config = {
        "binary": "llama-bench",
        "md": None,
    }
    env_config = {
        "binary": os.environ.get("LLAMA_BENCH", None),
        "md": os.environ.get("MODELS_DIR", None),
    }
    cli_config = {
        "binary": args.binary,
        "md": args.md,
    }

    def strip_none(d: dict):
        return {k: v for k, v in d.items() if v is not None}

    config = default_config | strip_none(env_config) | strip_none(cli_config)

    binary = config["binary"]

    if not shutil.which(binary):
        if not Path(binary).exists():
            parser.error(
                "llama-bench not found, make sure it's in PATH, or supply the path with either --llama-bench or the LLAMA_BENCH env var."
            )

    src_f = args.m
    src_d = config["md"]

    if not src_f and not src_d:
        parser.error(
            "Provide either -m <file> or -md <directory>, or MODELS_DIR env var."
        )

    files = []
    if src_f:
        f = Path(src_f)
        if not f.exists():
            parser.error(f"{f} does not exist.")
        if f.is_dir():
            parser.error(f"{f} is a directory, use -md.")
        abs_f = f.resolve()
        if not GGUFParser(abs_f).is_valid():
            parser.error(f"{f} is not a valid GGUF file.")
        files.append(abs_f)
    if src_d:
        d = Path(src_d)
        if not d.exists():
            parser.error(f"{d} does not exist.")
        if not d.is_dir():
            parser.error(f"{d} is not a directory.")
        gguf_files = [f.resolve() for f in d.iterdir() if f.is_file() and f.suffix.lower() == ".gguf"]
        if not gguf_files:
            parser.error(f"{d} has no .gguf files.")
        files.extend(gguf_files)

    kwargs = {k: v for k, v in vars(args).items() if k not in ["m", "md", "binary"]}
    return Inputs(config["binary"], files, **kwargs)
