import os
from dataclasses import asdict, replace
from gguf_bench.config import load_inputs
from gguf_bench.gguf import GGUFFile
from gguf_bench.optimize import BenchRunner, Parameters, optimize


def main():
    inputs = load_inputs()

    gguf_files = [f for f in inputs.files if GGUFFile(f).is_valid()]
    for file in gguf_files:
        metadata = GGUFFile(file).get_metadata()
        if metadata.version != GGUFFile.VERSION:
            print(f"{file} - incompatible GGUF version: {metadata.version}")
            continue
        if metadata.file_type != "model":
            print(f"{file} - incompatible type: {metadata.file_type}")
            continue
        if "bert" in metadata.architecture or metadata.architecture in [
            "",
            "whisper",
            "clip",
            "siglip",
        ]:
            print(f"{file} - incompatible architecture: {metadata.architecture}")
            continue

        print(f"{file} - beginning benchmark...")
        bench = BenchRunner(inputs.binary, file)
        defaults = Parameters(
            **{k: v for k, v in asdict(inputs.params).items() if v is not None}
        )
        defaults.n = 0

        if inputs.params.t is None:
            ncpu = os.cpu_count() or 1
            t = optimize(
                bench,
                "t",
                range(1, ncpu + 1),
                replace(defaults, ngl=0),
            )
            defaults.t = inputs.params.t = t

        if inputs.params.ngl is None:
            layers = metadata.block_count
            ngl = optimize(
                bench,
                "ngl",
                range(0, layers + 1),
                defaults
            )
            defaults.ngl = ngl
        
        print(defaults)


if __name__ == "__main__":
    main()
