import struct
import io
from pathlib import Path


class GGUFParser:
    VERSION = 3
    NTYPES = 13
    TYPES = "BbHhIif?s_Qqd"
    SIZES = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8]
    ValueType = int | float | bool | str | list

    def __init__(self, path: str | Path):
        self.path = path

    def _read_type(self, f: io.BufferedReader) -> (str, int):
        type_id: int = struct.unpack("<I", f.read(4))[0]
        if type_id >= GGUFParser.NTYPES:
            raise TypeError("Unknown metadata type.")
        value_type = GGUFParser.TYPES[type_id]
        value_size = GGUFParser.SIZES[type_id]
        return value_type, value_size

    def _read_string(self, f: io.BufferedReader) -> str:
        length: int = struct.unpack("<Q", f.read(8))[0]
        b: bytes = struct.unpack(f"<{length}s", f.read(length))[0]
        return b.decode("utf-8")

    def _read_array(self, f: io.BufferedReader) -> list[ValueType]:
        value_type, value_size = self._read_type(f)
        length: int = struct.unpack("<Q", f.read(8))[0]
        return [self._read_value(f, value_type, value_size) for _ in range(length)]

    def _read_value(
        self, f: io.BufferedReader, value_type: str, value_size: int
    ) -> ValueType:
        if value_type == "s":
            return self._read_string(f)
        if value_type == "_":
            return self._read_array(f)
        return struct.unpack(f"<{value_type}", f.read(value_size))[0]

    def _read_kv(self, f: io.BufferedReader) -> (str, ValueType):
        key = self._read_string(f)
        value_type, value_size = self._read_type(f)
        value = self._read_value(f, value_type, value_size)
        return key, value

    def _read_header(self, f: io.BufferedReader) -> (str, int, int, int) | None:
        size = 24
        header_bytes = f.read(size)
        if len(header_bytes) < size:
            return None
        [magic, version, tensor_count, metadata_kv_count] = struct.unpack(
            "<4sIQQ", header_bytes
        )
        if magic != b"GGUF":
            return None
        return magic, version, tensor_count, metadata_kv_count

    def read_metadata(self) -> dict[str, ValueType]:
        with open(self.path, "rb") as f:
            header = self._read_header(f)
            if not header:
                raise ValueError(f"{self.path} is not a GGUF file.")
            [magic, version, tensor_count, metadata_kv_count] = header
            metadata = {}
            for _ in range(metadata_kv_count):
                key, value = self._read_kv(f)
                metadata[key] = value
            return metadata, version, tensor_count

    def is_valid(self) -> bool:
        with open(self.path, "rb") as f:
            if not self._read_header(f):
                return False
            return True
