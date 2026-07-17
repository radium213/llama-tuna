import os
from dataclasses import asdict, replace
from gguf_bench.config import load_inputs
from gguf_bench.gguf import read_gguf_metadata, GGUFParsingError
from gguf_bench.optimize import BenchRunner, Parameters, optimize

OPTIONS_INCLUDE = ["t", "ngl", "b", "ub", "fa", "ctk", "ctv"]


def format_cli(command: str, params: Parameters) -> str:
    options: list[str] = []
    for k, v in asdict(params).items():
        if k in OPTIONS_INCLUDE:
            options.extend([f"-{k}", str(v)])
    return " ".join([command] + options) + "\n"


def format_ini(file: str, name: str, sizelabel: str, params: Parameters) -> str:
    label = f"[{name}-{sizelabel}]"
    options: list[str] = [f"model = {file}"]
    for k, v in asdict(params).items():
        if k in OPTIONS_INCLUDE:
            options.append(f"{k} = {v}")
    return "\n".join([label] + options) + "\n\n"


def main():
    inputs = load_inputs()
    output = ""
    global_t = inputs.params.t

    for file in inputs.files:
        try:
            metadata = read_gguf_metadata(file)
        except GGUFParsingError as e:
            print(f"{file} - {e}")
            continue
        if metadata.file_type != "model":
            print(f"{file} - incompatible type: {metadata.file_type}")
            continue
        if not metadata.architecture or metadata.architecture in [
            "whisper",
            "clip",
            "siglip",
        ]:
            print(f"{file} - incompatible architecture: {metadata.architecture}")
            continue

        print(f"{file} - beginning benchmark...")
        bench = BenchRunner(inputs.binary, file)
        params = Parameters(
            **{k: v for k, v in asdict(inputs.params).items() if v is not None}
        )
        params.n = 0

        if global_t is None:
            ncpu = os.cpu_count() or 1
            t = optimize(
                bench,
                "t",
                range(1, ncpu + 1),
                replace(params, ngl=0),
            )
            params.t = global_t = t

        if inputs.params.ngl is None:
            layers = metadata.block_count
            ngl = optimize(
                bench,
                "ngl",
                range(0, layers + 1),
                params
            )
            params.ngl = ngl

        if inputs.out_format == "cli":
            output += format_cli("llama-server", params)
        if inputs.out_format == "ini":
            output += format_ini(str(file), metadata.name, metadata.size_label, params)

    if inputs.outfile:
        with open(inputs.outfile, "w") as f:
            f.write(output)
    else:
        print("\n" + output)


if __name__ == "__main__":
    main()
