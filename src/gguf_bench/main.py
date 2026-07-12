import os
import sys
from pathlib import Path
from gguf_bench.config import load_inputs
from gguf_bench.gguf import GGUFParser
from gguf_bench.optimize import BenchRunner, Parameters, optimize


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
