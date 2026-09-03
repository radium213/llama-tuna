import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from llama_tuna.schema import InputParams


@dataclass(frozen=True)
class AppConfig:
    llama_bench: Path
    llama_cli: Path
    files: list[Path]
    out_format: str
    params: InputParams


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


class ToolPathError(Exception):
    """Tool path not found"""

def _get_tool_path(cmds: list[str], at: Path) -> Path:
    for cmd in cmds:
        path = shutil.which(cmd, path=at)
        if path:
            return Path(path)
    raise ToolPathError(f"{", ".join(cmds)} not found, make sure llama.cpp is installed and in PATH, or supply the path with --llama-cpp-path")


def load_config() -> AppConfig:
    parser = create_arg_parser()
    args = parser.parse_args()

    try:
        llama_path = Path(args.llama_path).expanduser().resolve()
        llama_bench = _get_tool_path(["llama-bench"], llama_path)
        llama_cli = _get_tool_path(["llama-completion", "llama-cli"], llama_path)
    except ToolPathError as e:
        parser.error(str(e))

    src_f: str | None = args.m
    src_d: str | None = args.md

    if not (bool(src_f) ^ bool(src_d)):
        parser.error("Provide either -m <file> or -md <directory>")

    files: list[Path] = []
    if src_f:
        f = Path(src_f).expanduser().resolve()
        if not f.exists():
            parser.error(f"{f} does not exist.")
        if f.is_dir():
            parser.error(f"{f} is a directory, use -md.")
        files.append(f)
    if src_d:
        d = Path(src_d).expanduser().resolve()
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

    return AppConfig(
        llama_bench=llama_bench,
        llama_cli=llama_cli,
        files=files,
        out_format=args.o,
        params=InputParams(
            c=args.c,
            t=args.t,
            ngl=args.ngl,
            fa=args.fa,
            ctk=args.ctk,
            ctv=args.ctv,
            b=args.b,
            ub=args.ub,
        ),
    )
