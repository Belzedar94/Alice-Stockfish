"""Canonical, create-only evidence primitives."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_create_only_json(path: str | os.PathLike[str], value: Any) -> str:
    target = Path(path)
    payload = canonical_json_bytes(value)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


class CreateOnlySeal:
    """Controller callback that creates exactly one immutable seal file."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.sha256: str | None = None

    def __call__(self, payload: dict[str, object]) -> None:
        if self.sha256 is not None:
            raise FileExistsError("the acceptance seal has already been created")
        self.sha256 = write_create_only_json(self.path, payload)
