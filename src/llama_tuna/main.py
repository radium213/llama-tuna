import io
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import replace
from functools import cache
from typing import Literal

from tqdm import tqdm

from llama_tuna.config import AppConfig, load_config
from llama_tuna.gguf import GGUFParsingError, read_gguf_metadata
from llama_tuna.optimize import (
    BenchRunner,
    CliRunner,
    OptimizeFailure,
    binary_search,
    fibonacci_search,
    grid_search,
)
from llama_tuna.schema import ModelParams


class DefaultFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"Error: {msg}"
        elif record.levelno >= logging.WARNING:
            return f"Warning: {msg}"
        return msg


def find_max_ngl(runner: CliRunner, search_space: Iterable[int], params: ModelParams):
    values = list(search_space)
    tq = tqdm(desc="[fit-ctx]")

    def run(val: int) -> bool:
        return runner(replace(params, ngl=val))
    
    def on_progress(i: int, n: int) -> None:
        tq.n = i
        tq.total = n
        tq.refresh()

    try:
        return binary_search(run, values, on_progress)
    finally:
        tq.close()


def optimize[T](
    runner: BenchRunner,
    param: str,
    search_space: Iterable[T],
    fixed_params: ModelParams,
    strat: Literal["auto", "fib", "grid"] = "auto",
) -> T:
    values = list(search_space)

    tq = tqdm(desc=f"[{param:>3}=  ?]", disable=None)

    def run(v: T) -> float:
        return runner(replace(fixed_params, **{param: v}))
    
    func = cache(run)

    def on_progress(i: int, n: int) -> None:
        tq.n = i
        tq.total = n
        tq.refresh()
    
    try:
        if strat == "grid" or strat == "auto" and len(values) <= 3:
            v = grid_search(func, values, on_progress)
        else:
            v = fibonacci_search(func, values, on_progress)
        tq.desc = f"[{param:>3}={v:>3}]"
    finally:
        tq.close()
    
    return v


def main_loop(config: AppConfig, output: io.TextIOBase, logger: logging.Logger):
    files_sorted = sorted(config.files, key=os.path.getsize)
    n_files = len(files_sorted)
    for i_file, file in enumerate(files_sorted, 1):
        logger.info(f"[{i_file:>3}/{n_files:>3}] {file.name}")
        try:
            with open(file, "rb") as fb:
                metadata = read_gguf_metadata(fb)
        except GGUFParsingError as e:
            logger.error(e)
            continue
        if not metadata.architecture or metadata.architecture in [
            "whisper",
            "clip",
            "siglip",
        ]:
            logger.warning(f"Excluded architecture: {metadata.architecture}")
            continue

        bench = BenchRunner(str(config.llama_bench.resolve()))
        cli = CliRunner(str(config.llama_cli.resolve()))
        params = config.params.get_params_for(file, metadata.context_length)
        ncpu = os.cpu_count() or 1
        min_t = min(config.params.t or 4, ncpu)

        if params.c > metadata.context_length:
            logger.warning(f"Specified context {params.c} exceeds model max {metadata.context_length}")

        try:
            ngl_max = find_max_ngl(cli, range(metadata.block_count + 1), params)
        except OptimizeFailure:
            logger.error(f"Unable to fit model with {params.c} context")
            continue

        if config.params.ngl is None:
            try:
                ngl = optimize(
                    bench,
                    "ngl",
                    range(ngl_max + 1),
                    replace(params, t=min_t),
                )
                params.ngl = ngl
            except OptimizeFailure as e:
                logger.error(e)
                continue
        else:
            if params.ngl > ngl_max:
                logger.warning(f"Specified gpu layers {params.ngl} will not fit in VRAM, clamping to {ngl_max}")
            params.ngl = min(params.ngl, ngl_max)

        if config.params.t is None:
            if params.ngl < metadata.block_count:
                try:
                    t = optimize(
                        bench,
                        "t",
                        range(1, ncpu + 1),
                        params,
                        "grid",
                    )
                    params.t = t
                except OptimizeFailure as e:
                    logger.error(e)
                    continue
            else:
                params.t = min_t

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
