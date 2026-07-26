import os
import io
import sys
import math
import logging
from typing import Literal
from pathlib import Path
from collections.abc import Iterable
from dataclasses import asdict, fields, replace
from llama_tuna.config import Inputs, load_inputs
from llama_tuna.gguf import read_gguf_metadata, GGUFParsingError
from llama_tuna.optimize import BenchRunner, Optimizer, OptimizeFailure, Parameters, Quant
from tqdm import tqdm


class DefaultFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"Error: {msg}"
        elif record.levelno >= logging.WARNING:
            return f"Warning: {msg}"
        return msg


class TestFailure(Exception):
    """Failure to run the model"""


def smoke_test(binary: str, model: Path, params: Parameters):
    tq = tqdm(total=1, desc="[ /// ]")
    runner = BenchRunner(binary, model, 1, True)
    result = runner(replace(params, ngl=0, d=0, p=4, n=2))
    if result == math.inf:
        tq.close()
        raise TestFailure(f"Could not run {model.name}")
    else:
        tq.update()
        tq.close()


def optimize[T](
    runner: BenchRunner,
    param: str,
    search_space: Iterable[T],
    fixed_params: Parameters,
    strat: Literal["auto", "fib", "grid"] = "auto",
):
    opt = Optimizer(runner, param, search_space, fixed_params, strat)

    with tqdm(opt, f"[ {param:>3} ]", disable=None) as tq:
        tq.set_postfix_str("?t/s")

        total_s = 0.0
        total_tok = 0
        tok_step = fixed_params.p + fixed_params.n
        for s in tq:
            if s < math.inf:
                total_s += s
                total_tok += tok_step
            rate = total_tok / total_s if total_s != 0.0 else 0.0
            if rate >= 1.0:
                tq.set_postfix_str(f"{rate:.0f}t/s")
            elif rate == 0.0:
                tq.set_postfix_str("0t/s")
            else:
                tq.set_postfix_str(f"{1.0 / rate:.2f}s/t")

    return opt.result()


def fit_context(runner: BenchRunner, params: Parameters, ctx: int) -> tuple[Quant, Quant]:
    params.d = max(ctx - params.p - params.n, 0)
    search_space: list[tuple[Quant, Quant]] = [
        ("f16", "f16"),
        ("f16", "q8_0"),
        ("f16", "q4_0"),
        ("q8_0", "q8_0"),
        ("q8_0", "q4_0"),
        ("q4_0", "q4_0"),
    ]
    tq = tqdm(search_space, "[ ctx ]")
    for ctk, ctv in tq:
        params.ctk = ctk
        params.ctv = ctv
        result = runner(params)
        if result < math.inf:
            tq.update(tq.total - tq.n)
            tq.close()
            return ctk, ctv
    ctx_k = ctx // 1000
    ctx_str = str(ctx_k) + "k" if ctx_k > 0 else str(ctx)
    raise TestFailure(f"Failed to fit {ctx_str} context")


OPTIONS_INCLUDE = ["t", "ngl", "b", "ub", "fa", "ctk", "ctv"]


def format_cli(command: str, params: Parameters, ctx: int) -> str:
    options: list[str] = []
    for k, v in asdict(params).items():
        if k in OPTIONS_INCLUDE:
            options.extend([f"-{k}", str(v)])
    return " ".join([command] + options) + f"-c {ctx}\n"


def format_ini(file: str, name: str, sizelabel: str, params: Parameters, ctx: int) -> str:
    label: str = f"[{name}-{sizelabel}]"
    options: list[str] = [f"model = {file}"]
    for k, v in asdict(params).items():
        if k in OPTIONS_INCLUDE:
            options.append(f"{k} = {v}")
    return "\n".join([label] + options) + f"\nc = {ctx}\n\n"


def main_loop(inputs: Inputs, output: io.TextIOBase):
    global_t = inputs.params.t

    files_sorted = sorted(inputs.files, key=os.path.getsize)
    n_files = len(files_sorted)
    for i_file, file in enumerate(files_sorted, 1):
        if n_files > 1:
            logging.info(f"[{i_file:>3}/{n_files:>3}] {file.name}")
        else:
            logging.info(f"[---/---] {file.name}")
        try:
            metadata = read_gguf_metadata(file)
        except GGUFParsingError as e:
            logging.error(e)
            continue
        if metadata.file_type != "model":
            logging.warning(f'Incompatible type: {metadata.file_type}, expected "model"')
            continue
        if not metadata.architecture or metadata.architecture in [
            "whisper",
            "clip",
            "siglip",
        ]:
            logging.warning(f"Excluded architecture: {metadata.architecture}")
            continue

        bench = BenchRunner(inputs.binary, file)
        _param_keys = [f.name for f in fields(Parameters)]
        params = Parameters(
            **{k: v for k, v in asdict(inputs.params).items() if k in _param_keys and v is not None}
        )

        try:
            smoke_test(inputs.binary, file, params)
        except TestFailure as e:
            logging.error(e)
            continue

        if global_t is None:
            ncpu = os.cpu_count() or 1
            t_range = range(1) if ncpu == 1 else range(2, ncpu + 1, 2)
            try:
                t = optimize(
                    bench,
                    "t",
                    t_range,
                    replace(params, ngl=0, d=0),
                    "grid",
                )
                params.t = global_t = t
            except OptimizeFailure as e:
                logging.error(e)
                continue
            if n_files > 1:
                logging.info(f"Continuing with -t {global_t} for all models")
        else:
            params.t = global_t

        if inputs.params.ngl is None:
            layers = metadata.block_count
            try:
                ngl = optimize(bench, "ngl", range(0, layers + 1), params)
                params.ngl = ngl
            except OptimizeFailure as e:
                logging.error(e)
                continue

        min_ctx = params.p + params.n
        req_ctx = inputs.params.c if inputs.params.c is not None else metadata.context_length
        ctx = max(min(req_ctx, metadata.context_length), min_ctx)
        try:
            ctk, ctv = fit_context(bench, params, ctx)
            params.ctk = ctk
            params.ctv = ctv
        except TestFailure as e:
            logging.error(e)
            continue

        if inputs.out_format == "cli":
            output.write(format_cli("llama-server", params, ctx))
        if inputs.out_format == "ini":
            output.write(format_ini(str(file), metadata.name, metadata.size_label, params, ctx))


def main():
    inputs = load_inputs()

    log_handler = logging.StreamHandler()
    log_handler.setFormatter(DefaultFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[log_handler])

    try:
        if isinstance(sys.stdout, io.TextIOBase) and not sys.stdout.isatty():
            main_loop(inputs, sys.stdout)
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
