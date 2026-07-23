import os
import io
import sys
from typing import TextIO
from dataclasses import asdict, replace
from llama_tuna.config import Inputs, load_inputs
from llama_tuna.gguf import read_gguf_metadata, GGUFParsingError
from llama_tuna.optimize import BenchRunner, OptimizeFailure, Parameters, optimize

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


def main_loop(inputs: Inputs, output: TextIO):
    global_t = inputs.params.t

    files_sorted = sorted(inputs.files, key=os.path.getsize)
    n_files = len(files_sorted)
    for i_file, file in enumerate(files_sorted, 1):
        if n_files > 1:
            print(f"[File {i_file:>3}/{n_files:>3}] {file.name}")
        else:
            print(f"File: {file.name}")
        try:
            metadata = read_gguf_metadata(file)
        except GGUFParsingError as e:
            print(f"Exception: {e}")
            continue
        if metadata.file_type != "model":
            print(f'Incompatible type: {metadata.file_type}, expected "model"')
            continue
        if not metadata.architecture or metadata.architecture in [
            "whisper",
            "clip",
            "siglip",
        ]:
            print(f"Excluded architecture: {metadata.architecture}")
            continue

        print("Beginning benchmark...")
        bench = BenchRunner(inputs.binary, file)
        params = Parameters(
            **{k: v for k, v in asdict(inputs.params).items() if v is not None}
        )

        if global_t is None:
            ncpu = os.cpu_count() or 1
            t_range = range(1) if ncpu == 1 else range(2, ncpu + 1, 2)
            try:
                t = optimize(
                    bench,
                    "t",
                    t_range,
                    replace(params, ngl=0),
                    "grid",
                )
                params.t = global_t = t
            except OptimizeFailure as e:
                print(e)
                continue
            if n_files > 1:
                print(f"Continuing with -t {global_t} for all models")
        else:
            params.t = global_t

        if inputs.params.ngl is None:
            layers = metadata.block_count
            try:
                ngl = optimize(
                    bench,
                    "ngl",
                    range(0, layers + 1),
                    params
                )
                params.ngl = ngl
            except OptimizeFailure as e:
                print(e)
                continue

        if inputs.out_format == "cli":
            output.write(format_cli("llama-server", params))
        if inputs.out_format == "ini":
            output.write(format_ini(str(file), metadata.name, metadata.size_label, params))


def main():
    inputs = load_inputs()

    try:
        if inputs.outfile:
            try:
                f = open(inputs.outfile, "x")
            except OSError as e:
                sys.exit(str(e))
            try:
                main_loop(inputs, f)
            finally:
                f.close()
        else:
            output = io.StringIO()
            try:
                main_loop(inputs, output)
            finally:
                print(output.getvalue())
    except KeyboardInterrupt:
        sys.exit("Interrupted by user.")


if __name__ == "__main__":
    main()
