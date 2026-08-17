from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Quant = Literal["f16", "q8_0", "q4_0"]


@dataclass
class ModelParams:
    m: Path
    c: int
    t: int = 1
    ngl: int = 0
    fa: Literal["on", "off"] = "on"
    ctk: Quant = "f16"
    ctv: Quant = "f16"
    b: int = 2048
    ub: int = 512

    def to_cli(self, binary: str = "llama-server") -> str:
        options = [f" -{k} {v}" for k, v in asdict(self).items()]
        return binary + "".join(options) + "\n"

    def to_ini(self, name: str | None = None, size_label: str | None = None) -> str:
        label = f"[{name}-{size_label}]" if name and size_label else self.m.stem
        options = [f"\n{k} = {v}" for k, v in asdict(self).items()]
        return label + "".join(options) + "\n"


@dataclass(frozen=True)
class InputParams:
    c: int | None = None
    t: int | None = None
    ngl: int | None = None
    fa: Literal["on", "off"] | None = None
    ctk: Quant | None = None
    ctv: Quant | None = None
    b: int | None = None
    ub: int | None = None

    def get_params_for(self, model: Path, default_context: int) -> ModelParams:
        return ModelParams(
            m=model,
            c=self.c if self.c is not None else default_context,
            t=self.t if self.t is not None else 1,
            ngl=self.ngl if self.ngl is not None else 0,
            fa=self.fa if self.fa is not None else "on",
            ctk=self.ctk if self.ctk is not None else "f16",
            ctv=self.ctv if self.ctv is not None else "f16",
            b=self.b if self.b is not None else 2048,
            ub=self.ub if self.ub is not None else 512,
        )
