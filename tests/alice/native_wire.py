"""Independent integer fixture exporter and parser for AliceNative-v1."""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
from typing import BinaryIO, Iterable


TEST_DIRECTORY = Path(__file__).resolve().parent
MANIFEST_PATH = TEST_DIRECTORY.parent.parent / "docs" / "alice" / "native-nnue-v1-manifest.json"

WIRE_VERSION = 0xA11CE001
LEGACY_WIRE_VERSION = 0x7AF32F20
ARCHITECTURE_HASH = 0xEC7CCD50
FEATURE_TRANSFORMER_HASH = 0x8F4FBC46
DENSE_ARCHITECTURE_HASH = 0x63337116

L1 = 1_024
THREAT_DIMENSIONS = 119_616
PIECE_SQUARE_DIMENSIONS = 45_056
PSQT_BUCKETS = 8
LAYER_STACKS = 8

FEATURE_TENSOR_BYTES = (
    2 * L1
    + THREAT_DIMENSIONS * L1
    + THREAT_DIMENSIONS * PSQT_BUCKETS * 4
    + PIECE_SQUARE_DIMENSIONS * L1 * 2
    + PIECE_SQUARE_DIMENSIONS * PSQT_BUCKETS * 4
)
DENSE_STACK_TENSOR_BYTES = 32 * 4 + 32 * L1 + 32 * 4 + 32 * 64 + 4 + 128


def canonical_manifest() -> bytes:
    raw = MANIFEST_PATH.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise AssertionError("The public manifest must have exactly one terminal LF.")
    return raw[:-1]


MANIFEST = canonical_manifest()
MANIFEST_SHA256 = hashlib.sha256(MANIFEST).hexdigest().upper()
NATIVE_WIRE_BYTES = (
    12
    + len(MANIFEST)
    + 4
    + FEATURE_TENSOR_BYTES
    + LAYER_STACKS * (4 + DENSE_STACK_TENSOR_BYTES)
)


def write_zero_wire(
    path: Path,
    *,
    version: int = WIRE_VERSION,
    architecture: int = ARCHITECTURE_HASH,
    manifest: bytes = MANIFEST,
    manifest_length: int | None = None,
    transformer_hash: int = FEATURE_TRANSFORMER_HASH,
    dense_hashes: Iterable[int] | None = None,
    tensor_mutation: tuple[int, int] | None = None,
) -> None:
    """Write a sparse, logically complete all-zero integer network."""

    stack_hashes = list(dense_hashes or [DENSE_ARCHITECTURE_HASH] * LAYER_STACKS)
    if len(stack_hashes) != LAYER_STACKS:
        raise ValueError("Exactly eight dense hashes are required.")

    with path.open("wb") as output:
        output.write(
            struct.pack(
                "<III",
                version,
                architecture,
                len(manifest) if manifest_length is None else manifest_length,
            )
        )
        output.write(manifest)
        output.write(struct.pack("<I", transformer_hash))
        tensor_start = output.tell()
        output.seek(FEATURE_TENSOR_BYTES, 1)
        for dense_hash in stack_hashes:
            output.write(struct.pack("<I", dense_hash))
            output.seek(DENSE_STACK_TENSOR_BYTES, 1)
        output.truncate()

    if tensor_mutation is not None:
        offset, value = tensor_mutation
        if not 0 <= offset < FEATURE_TENSOR_BYTES:
            raise ValueError("The tensor mutation offset is outside the feature tensors.")
        if not 0 <= value <= 255:
            raise ValueError("The tensor mutation value must be one byte.")
        with path.open("r+b") as output:
            output.seek(tensor_start + offset)
            output.write(bytes((value,)))


def _read_u32(source: BinaryIO) -> int:
    payload = source.read(4)
    if len(payload) != 4:
        raise ValueError("Truncated u32")
    return struct.unpack("<I", payload)[0]


def inspect_wire(path: Path) -> dict[str, object]:
    with path.open("rb") as source:
        version = _read_u32(source)
        architecture = _read_u32(source)
        manifest_length = _read_u32(source)
        manifest = source.read(manifest_length)
        if len(manifest) != manifest_length:
            raise ValueError("Truncated manifest")
        transformer_hash = _read_u32(source)
        source.seek(FEATURE_TENSOR_BYTES, 1)
        dense_hashes: list[int] = []
        for _ in range(LAYER_STACKS):
            dense_hashes.append(_read_u32(source))
            source.seek(DENSE_STACK_TENSOR_BYTES, 1)
        end = source.tell()
        if source.read(1):
            raise ValueError("Trailing data")

    return {
        "version": version,
        "architecture": architecture,
        "manifest": manifest,
        "manifestSha256": hashlib.sha256(manifest).hexdigest().upper(),
        "transformerHash": transformer_hash,
        "denseHashes": dense_hashes,
        "bytes": end,
    }


def assert_zero_tensors(path: Path) -> None:
    with path.open("rb") as source:
        source.seek(12 + len(MANIFEST) + 4)
        remaining = FEATURE_TENSOR_BYTES
        while remaining:
            chunk = source.read(min(1 << 20, remaining))
            if not chunk or any(chunk):
                raise AssertionError("Feature tensors are not the exported zero fixture.")
            remaining -= len(chunk)

        for _ in range(LAYER_STACKS):
            if _read_u32(source) != DENSE_ARCHITECTURE_HASH:
                raise AssertionError("Dense architecture hash mismatch.")
            remaining = DENSE_STACK_TENSOR_BYTES
            while remaining:
                chunk = source.read(min(1 << 20, remaining))
                if not chunk or any(chunk):
                    raise AssertionError("Dense tensors are not the exported zero fixture.")
                remaining -= len(chunk)
        if source.read(1):
            raise AssertionError("Trailing data after zero tensors.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest().upper()

