"""Read-only free-space gate for a future Alice training run."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from typing import Callable


MINIMUM_AVAILABLE_BYTES = 500 * 1024**3


def query_available_bytes(target_volume: str) -> int:
    """Perform exactly one platform free-space query for the calling user."""

    if os.name == "nt":
        available = ctypes.c_ulonglong()
        total = ctypes.c_ulonglong()
        free = ctypes.c_ulonglong()
        function = ctypes.windll.kernel32.GetDiskFreeSpaceExW  # type: ignore[attr-defined]
        function.argtypes = [
            ctypes.c_wchar_p,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
        ]
        function.restype = ctypes.c_int
        if not function(target_volume, available, total, free):
            raise OSError(ctypes.get_last_error(), "GetDiskFreeSpaceExW failed")
        return int(available.value)

    status = os.statvfs(target_volume)
    return int(status.f_bavail * status.f_frsize)


def evaluate_space(
    target_volume: str,
    *,
    query: Callable[[str], int] = query_available_bytes,
) -> dict[str, object]:
    available = query(target_volume)
    if type(available) is not int or available < 0:
        raise ValueError("the free-space query returned an invalid byte count")
    return {
        "schema": "alice-training-space-preflight-v1",
        "target_volume": target_volume,
        "minimum_available_bytes": MINIMUM_AVAILABLE_BYTES,
        "available_bytes": available,
        "passed": available >= MINIMUM_AVAILABLE_BYTES,
        "operation": "single-read-only-free-space-query",
        "training_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the frozen 500 GiB Alice training threshold without modifying the volume."
    )
    parser.add_argument("--target-volume", required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate_space(args.target_volume)
    except (OSError, ValueError) as error:
        result = {
            "schema": "alice-training-space-preflight-v1",
            "target_volume": args.target_volume,
            "minimum_available_bytes": MINIMUM_AVAILABLE_BYTES,
            "passed": False,
            "operation": "single-read-only-free-space-query",
            "training_started": False,
            "error": str(error),
        }
        print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
        return 4
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0 if bool(result["passed"]) else 3


if __name__ == "__main__":
    sys.exit(main())
