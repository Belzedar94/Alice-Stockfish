"""Strict, deterministic opening schedule for pair attempts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .evidence import sha256_file


class OpeningSchedule:
    def __init__(self, book: str | Path, expected_sha256: str, seed: int) -> None:
        if type(seed) is not int or seed < 0 or seed > (2**64 - 1):
            raise ValueError("opening seed must be an unsigned 64-bit integer")
        self.book = Path(book)
        actual_sha = sha256_file(self.book)
        if actual_sha != expected_sha256.lower():
            raise ValueError("opening-book SHA-256 mismatch")
        payload = self.book.read_bytes()
        if payload.startswith(b"\xef\xbb\xbf"):
            raise ValueError("opening book must be UTF-8 without a byte-order mark")
        text = payload.decode("utf-8", errors="strict")
        if "\x00" in text:
            raise ValueError("opening book contains a NUL character")
        entries: list[dict[str, object]] = []
        for line_number, raw_line in enumerate(text.splitlines(), 1):
            fen = raw_line.split(";", 1)[0].strip()
            if not fen or fen.startswith("#"):
                continue
            entries.append(
                {
                    "book_line": line_number,
                    "raw_line_sha256": hashlib.sha256(
                        raw_line.encode("utf-8")
                    ).hexdigest(),
                    "fen": fen,
                    "fen_sha256": hashlib.sha256(fen.encode("utf-8")).hexdigest(),
                }
            )
        if not entries:
            raise ValueError("opening book contains no usable positions")
        self.entries = tuple(entries)
        self.seed = seed
        self._cycle = 0
        self._indices: list[int] = []

    def _extend(self) -> None:
        prefix = (
            b"alice-opening-schedule-v1\0"
            + self.seed.to_bytes(8, "big")
            + self._cycle.to_bytes(8, "big")
        )
        ranked = [
            (hashlib.sha256(prefix + index.to_bytes(8, "big")).digest(), index)
            for index in range(len(self.entries))
        ]
        ranked.sort()
        self._indices.extend(index for _digest, index in ranked)
        self._cycle += 1

    def for_ordinal(self, ordinal: int) -> dict[str, object]:
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("opening ordinal must be a non-negative integer")
        while len(self._indices) <= ordinal:
            self._extend()
        entry = dict(self.entries[self._indices[ordinal]])
        entry["attempt_ordinal"] = ordinal
        entry["selection_algorithm"] = "alice-opening-schedule-v1"
        entry["seed"] = self.seed
        return entry
