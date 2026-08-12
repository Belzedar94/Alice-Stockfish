"""Verify one Alice-Stockfish release binary against the frozen v1 network."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


EXPECTED_NETWORK_NAME = "Alice_v1.nnue"
EXPECTED_NETWORK_BYTES = 47_721_376
EXPECTED_NETWORK_SHA256 = (
    "9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9"
)
EXPECTED_BENCH_NODES = 202_963
EXPECTED_ENGINE_NAME = "Alice-Stockfish 1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def run_bench(engine: Path, network: Path) -> dict[str, Any]:
    commands = "\n".join(
        (
            "uci",
            "setoption name Alice Evaluation value Legacy",
            "setoption name Use NNUE value true",
            "setoption name Alice_Frozen_Network value true",
            f"setoption name EvalFile value {network}",
            "isready",
            "bench",
            "quit",
            "",
        )
    )
    completed = subprocess.run(
        [str(engine)],
        cwd=Path.cwd(),
        input=commands,
        text=True,
        capture_output=True,
        encoding="ascii",
        errors="replace",
        timeout=180,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"bench process exited {completed.returncode}:\n{output}")
    required = (
        EXPECTED_ENGINE_NAME,
        "uciok",
        f"option name EvalFile type string default {EXPECTED_NETWORK_NAME}",
        "readyok",
        "LegacyAliceExact loaded",
        "mode=frozen-baseline",
        f"sha256={EXPECTED_NETWORK_SHA256}",
    )
    for token in required:
        if token not in output:
            raise RuntimeError(f"bench output lacks {token!r}:\n{output}")
    matches = re.findall(r"Nodes searched\s*:\s*(\d+)", output)
    if matches != [str(EXPECTED_BENCH_NODES)]:
        raise RuntimeError(f"unexpected canonical bench nodes {matches}:\n{output}")
    return {
        "returncode": completed.returncode,
        "nodes": EXPECTED_BENCH_NODES,
        "stdout_sha256": hashlib.sha256(output.encode("ascii", "replace")).hexdigest().upper(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--compiler-id-file", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = args.engine.resolve()
    network = args.network.resolve()
    if not engine.is_file():
        raise FileNotFoundError(f"release binary not found: {engine}")
    if not network.is_file():
        raise FileNotFoundError(f"release network not found: {network}")
    if network.name != EXPECTED_NETWORK_NAME:
        raise RuntimeError(f"public network must be named {EXPECTED_NETWORK_NAME}")
    if network.stat().st_size != EXPECTED_NETWORK_BYTES:
        raise RuntimeError("release network byte size does not match the frozen artifact")
    network_sha256 = sha256_file(network)
    if network_sha256 != EXPECTED_NETWORK_SHA256:
        raise RuntimeError("release network SHA-256 does not match the frozen artifact")
    compiler_id = args.compiler_id_file.read_text(encoding="utf-8").strip()
    if not compiler_id:
        raise RuntimeError("compiler identity is empty")

    benches = [run_bench(engine, network) for _ in range(3)]
    receipt = {
        "schema": "alice-release-binary-verification-v1",
        "source_commit": args.source_commit,
        "engine_version": EXPECTED_ENGINE_NAME,
        "platform": platform.platform(),
        "architecture": args.arch,
        "compiler": compiler_id,
        "engine": {
            "name": engine.name,
            "bytes": engine.stat().st_size,
            "sha256": sha256_file(engine),
        },
        "network": {
            "name": network.name,
            "bytes": network.stat().st_size,
            "sha256": network_sha256,
            "mode": "frozen-baseline",
            "uci_default": EXPECTED_NETWORK_NAME,
        },
        "benches": benches,
    }
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
