import io
import logging
import math
import os
import sys
from collections.abc import Iterable
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Literal, TypeGuard, get_args

from tqdm import tqdm

from llama_tuna.config import AppConfig, load_config
from llama_tuna.gguf import GGUFParsingError, read_gguf_metadata
from llama_tuna.optimize import (
    BenchRunner,
    OptimizeFailure,
    fibonacci_search,
    grid_search,
)
from llama_tuna.schema import ModelParams, Quant


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


def smoke_test(binary: str, model: Path, params: ModelParams):
    tq = tqdm(total=1, desc="[ /// ]")
    runner = BenchRunner(binary, 1, True)
    result = runner(replace(params, ngl=0), d=0, p=4, n=2)
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
    fixed_params: ModelParams,
    strat: Literal["auto", "fib", "grid"] = "auto",
):
    values = list(search_space)
    n = len(values)

    tq = tqdm(desc=f"[ {param:>3} ]", disable=None)

    def run(i: int) -> float:
        return runner(replace(fixed_params, **{param: values[i]}))
    
    func = cache(run)

    def on_progress(i: int, n: int) -> None:
        tq.n = i
        tq.total = n
        tq.refresh()
    
    try:
        if strat == "grid" or strat == "auto" and n <= 3:
            idx = grid_search(func, n, on_progress)
        else:
            idx = fibonacci_search(func, n, on_progress)
    finally:
        tq.close()
    
    return values[idx]


def fit_context(runner: BenchRunner, params: ModelParams, search_space: list[tuple[Quant, Quant]], ctx: int) -> tuple[Quant, Quant]:
    d = max(ctx - 512 - 128, 0) # TODO: refactor
    tq = tqdm(search_space, "[ ctx ]")
    for ctk, ctv in tq:
        params.ctk = ctk
        params.ctv = ctv
        result = runner(params, d=d)
        if result < math.inf:
            tq.update(tq.total - tq.n)
            tq.close()
            return ctk, ctv
    ctx_k = ctx // 1000
    ctx_str = str(ctx_k) + "k" if ctx_k > 0 else str(ctx)
    raise TestFailure(f"Failed to fit {ctx_str} context")


def main_loop(config: AppConfig, output: io.TextIOBase, logger: logging.Logger):
    global_t = config.params.t

    files_sorted = sorted(config.files, key=os.path.getsize)
    n_files = len(files_sorted)
    for i_file, file in enumerate(files_sorted, 1):
        if n_files > 1:
            logger.info(f"[{i_file:>3}/{n_files:>3}] {file.name}")
        else:
            logger.info(f"[---/---] {file.name}")
        try:
            metadata = read_gguf_metadata(file)
        except GGUFParsingError as e:
            logger.error(e)
            continue
        if metadata.file_type != "model":
            logger.warning(f'Incompatible type: {metadata.file_type}, expected "model"')
            continue
        if not metadata.architecture or metadata.architecture in [
            "whisper",
            "clip",
            "siglip",
        ]:
            logger.warning(f"Excluded architecture: {metadata.architecture}")
            continue

        bench = BenchRunner(str(config.llama_bench.resolve()))
        params = config.params.get_params_for(file, metadata.context_length) # TODO: refactor

        try:
            smoke_test(str(config.llama_bench.resolve()), file, params)
        except TestFailure as e:
            logger.error(e)
            continue

        if global_t is None:
            ncpu = os.cpu_count() or 1
            t_range = [1] if ncpu == 1 else range(2, ncpu + 1, 2)
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
                logger.error(e)
                continue
            if n_files > 1:
                logger.info(f"Continuing with -t {global_t} for all models")
        else:
            params.t = global_t

        if config.params.ngl is None:
            layers = metadata.block_count
            try:
                ngl = optimize(bench, "ngl", range(0, layers + 1), params)
                params.ngl = ngl
            except OptimizeFailure as e:
                logger.error(e)
                continue

        min_ctx = 512 + 128 # TODO: refactor
        req_ctx = config.params.c if config.params.c is not None else metadata.context_length
        ctx = max(min(req_ctx, metadata.context_length), min_ctx)
        if ctx != req_ctx:
            logger.warning(f"Context clamped to {ctx}")
        search_space: list[tuple[Quant, Quant]] = []
        if config.params.fa == "off":
            search_space = [
                ("f16", "f16"),
            ]
        else:
            ctk = config.params.ctk
            ctv = config.params.ctv
            def is_quant(s: str | None) -> TypeGuard[Quant]:
                return s in get_args(Quant)
            if is_quant(ctk) and is_quant(ctv):
                search_space = [
                    (ctk, ctv),
                ]
            elif is_quant(ctk):
                search_space = [
                    (ctk, "f16"),
                    (ctk, "q8_0"),
                    (ctk, "q4_0"),
                ]
            elif is_quant(ctv):
                search_space = [
                    ("f16", ctv),
                    ("q8_0", ctv),
                    ("q4_0", ctv),
                ]
            else:
                search_space = [
                    ("f16", "f16"),
                    ("f16", "q8_0"),
                    ("f16", "q4_0"),
                    ("q8_0", "q8_0"),
                    ("q8_0", "q4_0"),
                    ("q4_0", "q4_0"),
                ]
        try:
            ctk, ctv = fit_context(BenchRunner(str(config.llama_bench.resolve()), 1), params, search_space, ctx)
            params.ctk = ctk
            params.ctv = ctv
        except TestFailure as e:
            logger.error(e)
            continue

        if config.out_format == "cli":
            output.write(params.to_cli())
        if config.out_format == "ini":
            output.write(params.to_ini(metadata.name, metadata.size_label))


def main():
    config: AppConfig = load_config()

    log_handler = logging.StreamHandler()
    log_handler.setFormatter(DefaultFormatter())
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)

    try:
        if isinstance(sys.stdout, io.TextIOBase) and not sys.stdout.isatty():
            main_loop(config, sys.stdout, logger)
        else:
            output = io.StringIO()
            try:
                main_loop(config, output, logger)
            finally:
                print(output.getvalue())
    except KeyboardInterrupt:
        sys.exit("Interrupted by user.")


if __name__ == "__main__":
    main()
