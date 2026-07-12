import os
import sys
from pathlib import Path
from gguf_bench.config import load_inputs
from gguf_bench.gguf import GGUFFile
from gguf_bench.optimize import BenchRunner, Parameters, optimize


def get_smallest_model(files: list[Path]) -> Path | None:
    files = sorted(files, key=os.path.getsize)
    for file in files:
        metadata = get_metadata(file)
        if not metadata["compatible"]:
            continue
        return file
    return None


def main():
    inputs = load_inputs()
    gguf_files = [f for f in inputs.files if GGUFFile(f).is_valid()]
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
        metadata = GGUFFile(file).get_metadata()
        if metadata.version != GGUFFile.VERSION:
            print(f"{file} - incompatible GGUF version: {metadata.version}")
            continue
        if metadata.file_type != "model":
            print(f"{file} - incompatible type: {metadata.file_type}")
            continue
        if "bert" in metadata.architecture or metadata.architecture in ["", "whisper", "clip", "siglip"]:
            print(f"{file} - incompatible architecture: {metadata.architecture}")
            continue

        layers = metadata.block_count
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
