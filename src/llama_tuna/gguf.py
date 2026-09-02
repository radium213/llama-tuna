import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_VERSIONS = [2, 3]
VALUE_TYPES = "BbHhIif?s_Qqd"
VALUE_SIZES = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8]

ValueType = int | float | bool | str | list[Any]


@dataclass
class GGUFMetadata:
    version: int
    tensor_count: int
    architecture: str
    name: str
    size_label: str
    block_count: int
    context_length: int
    raw: dict[str, ValueType]


class GGUFParsingError(Exception):
    """Base GGUF parser exception"""


class IncompatibleHeader(GGUFParsingError):
    """Header indicates different file type"""


class IncompatibleVersion(GGUFParsingError):
    """File version is not supported"""


class UnknownDataType(GGUFParsingError):
    """Encountered unknown type id"""


class StructuralError(GGUFParsingError):
    """File structure is invalid"""


def _read_type(f: io.BufferedReader) -> tuple[str, int]:
    type_id: int = struct.unpack("<I", f.read(4))[0]
    if type_id >= len(VALUE_TYPES):
        raise UnknownDataType(f"Unknown data type id: {type_id}")
    value_type = VALUE_TYPES[type_id]
    value_size = VALUE_SIZES[type_id]
    return value_type, value_size


def _read_string(f: io.BufferedReader) -> str:
    length: int = struct.unpack("<Q", f.read(8))[0]
    b: bytes = struct.unpack(f"<{length}s", f.read(length))[0]
    return b.decode("utf-8")


def _read_array(f: io.BufferedReader) -> list[ValueType]:
    value_type, value_size = _read_type(f)
    length: int = struct.unpack("<Q", f.read(8))[0]
    return [_read_value(f, value_type, value_size) for _ in range(length)]


def _read_value(f: io.BufferedReader, value_type: str, value_size: int) -> ValueType:
    if value_type == "s":
        return _read_string(f)
    if value_type == "_":
        return _read_array(f)
    return struct.unpack(f"<{value_type}", f.read(value_size))[0]


def _read_kv(f: io.BufferedReader) -> tuple[str, ValueType]:
    key = _read_string(f)
    value_type, value_size = _read_type(f)
    value = _read_value(f, value_type, value_size)
    return key, value


def _read_header(f: io.BufferedReader) -> tuple[int, int, int]:
    [magic, version, tensor_count, metadata_kv_count] = struct.unpack(
        "<4sIQQ", f.read(24)
    )
    if magic != b"GGUF":
        raise IncompatibleHeader(
            f"Expected file signature: GGUF, but got {magic}"
        )
    return version, tensor_count, metadata_kv_count


def _read_metadata(path: str | Path) -> tuple[dict[str, ValueType], int, int]:
    with open(path, "rb") as f:
        try:
            header = _read_header(f)
            [version, tensor_count, metadata_kv_count] = header
            if version not in SUPPORTED_VERSIONS:
                raise IncompatibleVersion(
                    f"Expected GGUF version: {SUPPORTED_VERSIONS}, but got {version}"
                )
            metadata: dict[str, ValueType] = {}
            for _ in range(metadata_kv_count):
                key, value = _read_kv(f)
                metadata[key] = value
            return metadata, version, tensor_count
        except (struct.error, UnicodeDecodeError) as e:
            raise StructuralError("File structure is invalid") from e


def read_gguf_metadata(path: str | Path) -> GGUFMetadata:
    gguf_meta, gguf_version, tensor_count = _read_metadata(path)
    architecture = gguf_meta.get("general.architecture", "")
    name = gguf_meta.get("general.name", "")
    size_label = gguf_meta.get("general.size_label", "")
    block_count = gguf_meta.get(f"{architecture}.block_count", 0)
    context_length = gguf_meta.get(f"{architecture}.context_length", 0)
    return GGUFMetadata(
        gguf_version,
        tensor_count,
        str(architecture),
        str(name),
        str(size_label),
        int(block_count) if isinstance(block_count, int) else 0,
        int(context_length) if isinstance(context_length, int) else 0,
        raw=gguf_meta,
    )
