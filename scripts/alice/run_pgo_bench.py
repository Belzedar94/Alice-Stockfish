"""Run the Alice canonical bench for PGO without compiling a default net path."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


EXPECTED_NODES = 202_963
EXPECTED_SHA256 = "9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("network", type=Path)
    args = parser.parse_args()
    engine = args.engine.resolve()
    network = args.network.resolve()
    commands = "\n".join(
        (
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
        input=commands,
        text=True,
        capture_output=True,
        encoding="ascii",
        errors="replace",
        timeout=180,
        check=False,
    )
    output = completed.stdout + completed.stderr
    print(output, end="")
    if completed.returncode != 0:
        return completed.returncode
    for token in ("readyok", "LegacyAliceExact loaded", f"sha256={EXPECTED_SHA256}"):
        if token not in output:
            raise RuntimeError(f"PGO bench output lacks {token!r}")
    matches = re.findall(r"Nodes searched\s*:\s*(\d+)", output)
    if matches != [str(EXPECTED_NODES)]:
        raise RuntimeError(f"unexpected PGO bench node counts: {matches}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
