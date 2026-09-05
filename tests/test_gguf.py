import io
import struct

import pytest

from llama_tuna.gguf import (
    IncompatibleHeader,
    IncompatibleVersion,
    StructuralError,
    read_gguf_metadata,
)


def mock_gguf(
    magic: bytes = b"GGUF",
    version: int = 3,
    tensor_count: int = 0,
    kv_count: int = 0,
    kv_bytes: bytes = b"",
) -> io.BufferedReader:
    buf = io.BytesIO()
    buf.write(struct.pack("<4sIQQ", magic, version, tensor_count, kv_count))
    buf.write(kv_bytes)
    buf.seek(0)
    return io.BufferedReader(buf)


def mock_str(s: str) -> bytes:
    _s = s.encode("utf-8")
    return struct.pack("<Q", len(_s)) + _s


def mock_val(v: int | float | bool | str) -> bytes:  # noqa: PYI041
    if isinstance(v, bool):
        return struct.pack("<I?", 7, v)
    elif isinstance(v, int):
        if v >= 0:
            return struct.pack("<IQ", 10, v)
        else:
            return struct.pack("<Iq", 11, v)
    elif isinstance(v, float):
        return struct.pack("<Id", 12, v)
    else:
        return struct.pack("<I", 8) + mock_str(v)


def mock_kv(key: str, value: int | float | bool | str) -> bytes:  # noqa: PYI041
    return mock_str(key) + mock_val(value)


def test_gguf_meta_valid() -> None:
    v = 3
    arch = "llama"
    name = "Llama-3-8B"
    bc = 32
    ctx = 4096
    kv = (
        mock_kv("general.architecture", arch)
        + mock_kv("general.name", name)
        + mock_kv("llama.block_count", bc)
        + mock_kv("llama.context_length", ctx)
    )
    mockfile = mock_gguf(version=v, kv_count=4, kv_bytes=kv)

    meta = read_gguf_metadata(mockfile)
    assert meta.version == v
    assert meta.architecture == arch
    assert meta.name == name
    assert meta.block_count == bc
    assert meta.context_length == ctx


@pytest.mark.parametrize(
    "mockfile, exc",
    [
        (mock_gguf(magic=b"JPEG"), IncompatibleHeader),
        (mock_gguf(version=1), IncompatibleVersion),
        (mock_gguf(version=99), IncompatibleVersion),
        (mock_gguf(kv_count=4), StructuralError),
    ],
)
def test_gguf_meta_invalid(
    mockfile: io.BufferedReader, exc: type[BaseException]
) -> None:
    with pytest.raises(exc):
        read_gguf_metadata(mockfile)
