import argparse
import os
import struct
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


def main():
    print(parse_inputs())


if __name__ == "__main__":
    main()
