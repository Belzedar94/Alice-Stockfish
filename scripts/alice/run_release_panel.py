#!/usr/bin/env python3
"""Run the frozen Alice-Stockfish 1.0 release comparison panel.

The panel is fail-closed. Every executable, network, opening book, referee
script and Python interpreter is authenticated before play. VSTC, STC and LTC
use fixed game counts, one deterministic opening permutation, paired colors,
and no score adjudication or early stopping. Any missing game, incomplete pair,
time loss, adjudication, crash, protocol failure, or other abnormal termination
invalidates the complete panel.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist


EXPECTED_CANDIDATE_COMMIT = "4a88df6f03ffd9a721b54f04cdb12a8e847929c5"
EXPECTED_CANDIDATE_TREE = "4288edda2f36e4de28507f1f8d046b27b6cf3af5"
EXPECTED_BASELINE_COMMIT = "4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79"
EXPECTED_BASELINE_SHA256 = (
    "B70AFE03EC9A67258CD7B5B848C46FC9E5C83F53B9F2825E9A5946FEEFB59599"
)
EXPECTED_NETWORK_SHA256 = (
    "9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9"
)
EXPECTED_BOOK_SHA256 = (
    "BCD89D9FC3EA81FEB95932EB64D6B6F15AD25CC04CDCC9E0440F097CFFB8CCF6"
)
EXPECTED_REFEREE_COMMIT = "4da5cb6c4ff502b60efb6e8c6c9bd7ef0c37fc69"
EXPECTED_REFEREE_SHA256 = (
    "73081EC57EAF964009CF7A877428888910A7AF1C091D8F0F7ECFB47020161CAB"
)
EXPECTED_PYTHON_SHA256 = (
    "42AC541168E97DEDB9AABD8BE335539FC41C682E414B9E8D137B164FB68683B0"
)
EXPECTED_NETWORK_NAME = "alice_run2rl_e40_l09.nnue"
REJECTED_NATIVE_V2_CHECKPOINT_SHA256 = (
    "689B1780CA4412C5971BB7A787DB351646E2216A8B7EF3E682938831389B13B5"
)
REJECTED_NATIVE_V2_NETWORK_SHA256 = (
    "02A2664731CE9FC4995C1ABBD97A0944F39C8302D82A055567E58BCDFF19C086"
)
REJECTED_NATIVE_V2_INFERENCE_COMMIT = (
    "c81d3f19418e94a2f108933253c4f8ac7fbaf266"
)
OPENING_SEED = 20260811
CANDIDATE_NAME = "Alice-Stockfish-release"
BASELINE_NAME = "Fairy-Stockfish-040925"


@dataclass(frozen=True)
class PanelSpec:
    label: str
    tc: str
    games: int
    seed: int = OPENING_SEED
    hash_mb: int = 512


PANEL_SPECS = (
    PanelSpec("VSTC", "2+0.02", 700),
    PanelSpec("STC", "10+0.1", 500),
    PanelSpec("LTC", "30+0.3", 300),
)

FINISHED_RE = re.compile(
    r"^Finished game\s+(\d+)\s+\((.+) vs (.+)\):\s+"
    r"(1-0|0-1|1/2-1/2)\s+\{(.*)\}\s*$"
)
HEX_40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
HEX_64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
BAD_REASON_TERMS = (
    "abandon",
    "adjudication",
    "crash",
    "disconnect",
    "failed",
    "illegal",
    "on time",
    "stall",
    "terminated",
    "timed out",
)
ALLOWED_REASONS = {
    "White mates",
    "Black mates",
    "Draw by stalemate",
    "Draw by rule",
}
FORBIDDEN_PGN_HEADERS = {
    "Termination",
    "OutcomeClass",
    "FailureCode",
    "FailureStage",
    "OffendingMove",
}

PRINT_LOCK = threading.Lock()
ACTIVE_LOCK = threading.Lock()
ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()


def emit(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def logistic_elo(score: float) -> float:
    score = min(max(score, 1e-12), 1.0 - 1e-12)
    return -400.0 * math.log10(1.0 / score - 1.0)


def statistics_from_penta(penta: list[int]) -> dict[str, float]:
    pairs = sum(penta)
    if not pairs:
        return {"elo": 0.0, "ci95": 0.0, "los": 0.5}

    values = [index / 4.0 for index in range(5)]
    mean = sum(value * count for value, count in zip(values, penta)) / pairs
    variance = sum(
        ((value - mean) ** 2) * count for value, count in zip(values, penta)
    ) / pairs
    if variance == 0.0:
        los = 0.5 if mean == 0.5 else float(mean > 0.5)
        return {"elo": logistic_elo(mean), "ci95": 0.0, "los": los}

    standard_error = math.sqrt(variance / pairs)
    z95 = NormalDist().inv_cdf(0.975)
    lower = logistic_elo(mean - z95 * standard_error)
    middle = logistic_elo(mean)
    upper = logistic_elo(mean + z95 * standard_error)
    return {
        "elo": middle,
        "ci95": (upper - lower) / 2.0,
        "los": NormalDist().cdf((mean - 0.5) / standard_error),
    }


def candidate_score(white: str, black: str, result: str) -> float:
    if result == "1/2-1/2":
        return 0.5
    winner = white if result == "1-0" else black
    if winner == CANDIDATE_NAME:
        return 1.0
    if winner == BASELINE_NAME:
        return 0.0
    raise RuntimeError(f"Unknown winner {winner!r}")


class PanelTracker:
    def __init__(self, spec: PanelSpec) -> None:
        self.spec = spec
        self.games: dict[int, dict[str, object]] = {}
        self.anomalies: list[str] = []
        self.status = "PENDING"
        self.lock = threading.Lock()

    def consume(self, line: str) -> None:
        match = FINISHED_RE.match(line.strip())
        if not match:
            return

        game_no = int(match.group(1))
        white, black, result, reason = match.group(2, 3, 4, 5)
        if {white, black} != {CANDIDATE_NAME, BASELINE_NAME}:
            raise RuntimeError(
                f"{self.spec.label} game {game_no} used unexpected engines: "
                f"{white} vs {black}"
            )
        if not 1 <= game_no <= self.spec.games:
            raise RuntimeError(
                f"{self.spec.label} reported out-of-range game {game_no}"
            )

        score = candidate_score(white, black, result)
        anomaly = next(
            (term for term in BAD_REASON_TERMS if term in reason.lower()), None
        )
        if reason not in ALLOWED_REASONS:
            anomaly = anomaly or "non-allowlisted termination"
        with self.lock:
            if game_no in self.games:
                raise RuntimeError(
                    f"{self.spec.label} reported game {game_no} more than once"
                )
            self.games[game_no] = {
                "score": score,
                "white": white,
                "black": black,
                "result": result,
                "reason": reason,
            }
            if anomaly is not None:
                self.anomalies.append(f"game {game_no}: {reason}")
        if anomaly is not None:
            raise RuntimeError(
                f"{self.spec.label} infrastructure defect in game {game_no}: "
                f"{reason}"
            )

    def set_status(self, status: str) -> None:
        with self.lock:
            self.status = status

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            games = dict(self.games)
            anomalies = list(self.anomalies)
            status = self.status

        scores = [float(game["score"]) for game in games.values()]
        wins = sum(score == 1.0 for score in scores)
        draws = sum(score == 0.5 for score in scores)
        losses = sum(score == 0.0 for score in scores)
        penta = [0, 0, 0, 0, 0]
        for first in range(1, self.spec.games + 1, 2):
            second = first + 1
            if first in games and second in games:
                bucket = int(
                    round(
                        2
                        * (
                            float(games[first]["score"])
                            + float(games[second]["score"])
                        )
                    )
                )
                penta[bucket] += 1

        return {
            "label": self.spec.label,
            "tc": self.spec.tc,
            "hash_mb": self.spec.hash_mb,
            "seed": self.spec.seed,
            "expected_games": self.spec.games,
            "games": len(games),
            "wdl": {"wins": wins, "draws": draws, "losses": losses},
            "pentanomial": penta,
            "complete_pairs": sum(penta),
            "statistics": statistics_from_penta(penta),
            "anomalies": anomalies,
            "status": status,
        }

    def require_complete(self) -> dict[str, object]:
        snapshot = self.snapshot()
        expected = set(range(1, self.spec.games + 1))
        with self.lock:
            observed = set(self.games)
        missing = sorted(expected - observed)
        if missing:
            preview = ", ".join(str(value) for value in missing[:10])
            raise RuntimeError(
                f"{self.spec.label} expected {self.spec.games} games, "
                f"missing {preview}"
            )
        if snapshot["anomalies"]:
            raise RuntimeError(
                f"{self.spec.label} infrastructure defect: "
                + "; ".join(snapshot["anomalies"])
            )
        if snapshot["complete_pairs"] != self.spec.games // 2:
            raise RuntimeError(f"{self.spec.label} did not produce complete pairs")
        with self.lock:
            for first in range(1, self.spec.games + 1, 2):
                second = first + 1
                odd = self.games[first]
                even = self.games[second]
                if (
                    odd["white"] != CANDIDATE_NAME
                    or odd["black"] != BASELINE_NAME
                    or even["white"] != BASELINE_NAME
                    or even["black"] != CANDIDATE_NAME
                ):
                    raise RuntimeError(
                        f"{self.spec.label} pair {(first + 1) // 2} "
                        "did not swap colors exactly"
                    )
        return snapshot

    def observations(self) -> dict[int, dict[str, object]]:
        with self.lock:
            return {number: dict(game) for number, game in self.games.items()}


def path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def checked_sha(value: str) -> str:
    normalized = value.upper()
    if not HEX_64_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a complete 64-character SHA-256")
    return normalized


def checked_commit(value: str) -> str:
    normalized = value.lower()
    if not HEX_40_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a complete 40-character commit")
    return normalized


def count_openings(book: Path) -> int:
    with book.open("r", encoding="utf-8-sig") as handle:
        return sum(
            1
            for line in handle
            if line.strip() and not line.lstrip().startswith(("#", ";"))
        )


def require_identity(actual: str, expected: str, label: str) -> None:
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{label} mismatch: expected {expected}, got {actual}")


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def git_output(root: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {root}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def validate_git_identity(
    root: Path, expected_commit: str, expected_tree: str, label: str
) -> dict[str, object]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing {label} source root: {root}")
    commit = git_output(root, "rev-parse", "HEAD").lower()
    tree = git_output(root, "rev-parse", "HEAD^{tree}").lower()
    status = git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    require_identity(commit, expected_commit, f"{label} source commit")
    require_identity(tree, expected_tree, f"{label} source tree")
    if status:
        raise RuntimeError(f"{label} source worktree is dirty: {status}")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "merge-base",
            "--is-ancestor",
            REJECTED_NATIVE_V2_INFERENCE_COMMIT,
            expected_commit,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode == 0:
        raise RuntimeError("rejected NativeV2 inference commit is in candidate history")
    return {
        "root": str(root),
        "commit": commit,
        "tree": tree,
        "dirty": False,
        "rejected_native_v2_inference_ancestor": False,
    }


def validate_runner_identity(runner_commit: str) -> dict[str, object]:
    runner = Path(__file__).resolve()
    root = runner.parents[2]
    commit = git_output(root, "rev-parse", "HEAD").lower()
    require_identity(commit, runner_commit, "panel runner commit")
    relative = runner.relative_to(root).as_posix()
    status = git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        relative,
    )
    if status:
        raise RuntimeError(f"panel runner is not clean at {runner_commit}: {status}")
    return {
        "root": str(root),
        "commit": commit,
        "path": str(runner),
        "relative_path": relative,
        "sha256": sha256_file(runner),
        "size": runner.stat().st_size,
        "dirty": False,
    }


def validate_candidate_build_receipt(
    args: argparse.Namespace,
) -> dict[str, object]:
    path = args.candidate_build_receipt
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate build receipt: {path}")
    actual_sha = sha256_file(path)
    require_identity(
        actual_sha, args.candidate_build_receipt_sha256, "candidate build receipt"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "ALICE_RELEASE_CANDIDATE_BUILD_V1":
        raise RuntimeError("candidate build receipt schema mismatch")
    require_identity(
        str(receipt.get("candidate_commit", "")),
        EXPECTED_CANDIDATE_COMMIT,
        "build receipt candidate commit",
    )
    require_identity(
        str(receipt.get("candidate_tree", "")),
        EXPECTED_CANDIDATE_TREE,
        "build receipt candidate tree",
    )
    if receipt.get("source_dirty") is not False:
        raise RuntimeError("candidate build receipt does not prove a clean source")
    binary = receipt.get("binary")
    if not isinstance(binary, dict):
        raise RuntimeError("candidate build receipt has no binary object")
    require_identity(
        str(binary.get("sha256", "")),
        args.candidate_sha256,
        "build receipt candidate binary",
    )
    if binary.get("size") != args.candidate.stat().st_size:
        raise RuntimeError("candidate build receipt binary size mismatch")
    if binary.get("architecture") != "x86-64-bmi2":
        raise RuntimeError("candidate build receipt architecture mismatch")
    if receipt.get("bench_nodes") != [202963, 202963, 202963]:
        raise RuntimeError("candidate build receipt lacks three canonical benches")
    tests = receipt.get("mandatory_tests")
    if not isinstance(tests, dict) or not tests:
        raise RuntimeError("candidate build receipt has no mandatory test inventory")
    for name, result in tests.items():
        if not isinstance(result, dict):
            raise RuntimeError(f"candidate build test {name} has invalid evidence")
        if (
            result.get("status") != "PASS"
            or result.get("failed") != 0
            or result.get("skipped") != 0
        ):
            raise RuntimeError(f"candidate build test {name} is not a clean pass")
    v2 = receipt.get("native_v2_exclusion")
    if not isinstance(v2, dict):
        raise RuntimeError("candidate build receipt lacks NativeV2 exclusion")
    require_identity(
        str(v2.get("checkpoint_sha256", "")),
        REJECTED_NATIVE_V2_CHECKPOINT_SHA256,
        "rejected NativeV2 checkpoint",
    )
    require_identity(
        str(v2.get("network_sha256", "")),
        REJECTED_NATIVE_V2_NETWORK_SHA256,
        "rejected NativeV2 network",
    )
    require_identity(
        str(v2.get("inference_commit", "")),
        REJECTED_NATIVE_V2_INFERENCE_COMMIT,
        "rejected NativeV2 inference commit",
    )
    if (
        v2.get("checkpoint_present") is not False
        or v2.get("network_present") is not False
        or v2.get("inference_commit_ancestor") is not False
    ):
        raise RuntimeError("candidate build receipt does not exclude rejected NativeV2")
    if receipt.get("passed") is not True:
        raise RuntimeError("candidate build receipt is not marked PASS")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "size": path.stat().st_size,
        "evidence": receipt,
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, object]:
    require_identity(
        args.candidate_commit, EXPECTED_CANDIDATE_COMMIT, "candidate commit"
    )
    require_identity(args.candidate_tree, EXPECTED_CANDIDATE_TREE, "candidate tree")
    require_identity(
        args.baseline_commit, EXPECTED_BASELINE_COMMIT, "baseline commit"
    )
    require_identity(args.referee_commit, EXPECTED_REFEREE_COMMIT, "referee commit")
    require_identity(
        args.baseline_sha256, EXPECTED_BASELINE_SHA256, "baseline executable"
    )
    require_identity(
        args.candidate_network_sha256,
        EXPECTED_NETWORK_SHA256,
        "candidate network",
    )
    require_identity(
        args.baseline_network_sha256,
        EXPECTED_NETWORK_SHA256,
        "baseline network",
    )
    require_identity(args.book_sha256, EXPECTED_BOOK_SHA256, "opening book")
    require_identity(
        args.referee_sha256, EXPECTED_REFEREE_SHA256, "referee script"
    )
    require_identity(
        args.referee_python_sha256,
        EXPECTED_PYTHON_SHA256,
        "referee Python interpreter",
    )
    if args.candidate_network.name != EXPECTED_NETWORK_NAME:
        raise RuntimeError("candidate network does not retain the frozen basename")
    if args.baseline_network.name != EXPECTED_NETWORK_NAME:
        raise RuntimeError("baseline network does not retain the frozen basename")

    source_identity = validate_git_identity(
        args.candidate_source_root,
        EXPECTED_CANDIDATE_COMMIT,
        EXPECTED_CANDIDATE_TREE,
        "candidate",
    )
    runner_identity = validate_runner_identity(args.runner_commit)
    build_receipt = validate_candidate_build_receipt(args)

    roles = {
        "candidate_executable": (args.candidate, args.candidate_sha256),
        "candidate_network": (
            args.candidate_network,
            args.candidate_network_sha256,
        ),
        "baseline_executable": (args.baseline, args.baseline_sha256),
        "baseline_network": (
            args.baseline_network,
            args.baseline_network_sha256,
        ),
        "referee_script": (args.referee, args.referee_sha256),
        "referee_python": (
            args.referee_python,
            args.referee_python_sha256,
        ),
        "book": (args.book, args.book_sha256),
        "candidate_build_receipt": (
            args.candidate_build_receipt,
            args.candidate_build_receipt_sha256,
        ),
    }
    observed: dict[str, dict[str, object]] = {}
    for role, (path, expected_sha) in roles.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role}: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"{role} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        observed[role] = {
            "path": str(path),
            "sha256": actual_sha,
            "size": path.stat().st_size,
        }

    openings = count_openings(args.book)
    required = max(spec.games // 2 for spec in PANEL_SPECS)
    if openings < required:
        raise RuntimeError(
            f"Opening book has {openings} positions; the panel needs {required}"
        )
    observed["book"]["opening_count"] = openings
    observed["panel_runner"] = runner_identity
    observed["candidate_source"] = source_identity
    observed["candidate_build_receipt"]["validated"] = build_receipt["evidence"]
    return observed


def copy_exclusive(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.chmod(destination, stat.S_IREAD)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "size": destination.stat().st_size,
        "read_only": not bool(destination.stat().st_mode & stat.S_IWRITE),
    }


def snapshot_inputs(
    args: argparse.Namespace,
    original_assets: dict[str, object],
) -> dict[str, dict[str, object]]:
    snapshot_root = args.output_root / "inputs"
    mapping = {
        "candidate_executable": (
            "candidate",
            args.candidate,
            snapshot_root / "candidate" / "alice-stockfish.exe",
        ),
        "candidate_network": (
            "candidate_network",
            args.candidate_network,
            snapshot_root / "networks" / "candidate" / EXPECTED_NETWORK_NAME,
        ),
        "baseline_executable": (
            "baseline",
            args.baseline,
            snapshot_root / "baseline" / "fairy-stockfish.exe",
        ),
        "baseline_network": (
            "baseline_network",
            args.baseline_network,
            snapshot_root / "networks" / "baseline" / EXPECTED_NETWORK_NAME,
        ),
        "referee_script": (
            "referee",
            args.referee,
            snapshot_root / "referee" / "uci_pair_runner.py",
        ),
        "book": (
            "book",
            args.book,
            snapshot_root / "book" / "alice.epd",
        ),
        "candidate_build_receipt": (
            "candidate_build_receipt",
            args.candidate_build_receipt,
            snapshot_root / "receipts" / "candidate-build.json",
        ),
    }
    snapshots: dict[str, dict[str, object]] = {}
    for role, (attribute, source, destination) in mapping.items():
        evidence = copy_exclusive(source, destination)
        original = original_assets[role]
        assert isinstance(original, dict)
        require_identity(
            str(evidence["sha256"]), str(original["sha256"]), f"{role} snapshot"
        )
        if evidence["size"] != original["size"]:
            raise RuntimeError(f"{role} snapshot size mismatch")
        evidence["source_path"] = str(source)
        snapshots[role] = evidence
        setattr(args, attribute, destination)

    runner = Path(__file__).resolve()
    runner_snapshot = copy_exclusive(
        runner, snapshot_root / "runner" / "run_release_panel.py"
    )
    runner_original = original_assets["panel_runner"]
    assert isinstance(runner_original, dict)
    require_identity(
        str(runner_snapshot["sha256"]),
        str(runner_original["sha256"]),
        "panel runner snapshot",
    )
    runner_snapshot["source_path"] = str(runner)
    snapshots["panel_runner"] = runner_snapshot
    return snapshots


def environment_inventory() -> dict[str, object]:
    entries = [f"{key}={value}" for key, value in sorted(os.environ.items())]
    environment_sha = hashlib.sha256("\0".join(entries).encode("utf-8")).hexdigest()
    module_paths = {
        "json": Path(json.__file__).resolve(),
        "os": Path(os.__file__).resolve(),
        "subprocess": Path(subprocess.__file__).resolve(),
    }
    return {
        "environment_sha256": environment_sha.upper(),
        "environment_keys": sorted(os.environ),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "python_compiler": platform.python_compiler(),
        "sys_prefix": sys.prefix,
        "module_sha256": {
            name: sha256_file(path) for name, path in module_paths.items()
        },
    }


def revalidate_custody(
    args: argparse.Namespace,
    snapshots: dict[str, dict[str, object]],
) -> dict[str, object]:
    verified: dict[str, dict[str, object]] = {}
    for role, evidence in snapshots.items():
        path = Path(str(evidence["path"]))
        if not path.is_file():
            raise RuntimeError(f"custody input disappeared: {role}")
        observed_sha = sha256_file(path)
        require_identity(observed_sha, str(evidence["sha256"]), f"{role} post-run")
        if path.stat().st_size != evidence["size"]:
            raise RuntimeError(f"{role} post-run size mismatch")
        verified[role] = {
            "path": str(path),
            "sha256": observed_sha,
            "size": path.stat().st_size,
        }
    python_sha = sha256_file(args.referee_python)
    require_identity(
        python_sha, args.referee_python_sha256, "referee Python post-run"
    )
    source = validate_git_identity(
        args.candidate_source_root,
        EXPECTED_CANDIDATE_COMMIT,
        EXPECTED_CANDIDATE_TREE,
        "candidate",
    )
    runner = validate_runner_identity(args.runner_commit)
    return {
        "completed_at": utc_now(),
        "snapshots": verified,
        "referee_python_sha256": python_sha,
        "candidate_source": source,
        "panel_runner": runner,
        "status": "PASS",
    }


def add_options(engine: list[str], options: tuple[tuple[str, str], ...]) -> None:
    engine.extend(f"option.{name}={value}" for name, value in options)


def preflight_options(
    args: argparse.Namespace, role: str
) -> tuple[tuple[str, str], ...]:
    common = (
        ("Threads", "1"),
        ("Hash", "512"),
        ("Move Overhead", "10"),
        ("Use NNUE", "true"),
    )
    if role == "candidate":
        return common + (
            ("Alice Evaluation", "Legacy"),
            ("Alice_Frozen_Network", "true"),
            ("EvalFile", str(args.candidate_network)),
        )
    if role == "baseline":
        return common + (
            ("UCI_Variant", "alice"),
            ("EvalFile", str(args.baseline_network)),
        )
    raise ValueError(f"unknown preflight role {role!r}")


def authenticate_engine(
    args: argparse.Namespace, role: str, output_root: Path
) -> dict[str, object]:
    executable = args.candidate if role == "candidate" else args.baseline
    options = preflight_options(args, role)
    commands = ["uci"]
    commands.extend(f"setoption name {name} value {value}" for name, value in options)
    commands.extend(("isready", "position startpos", "go depth 1", "quit"))
    payload = "\n".join(commands) + "\n"
    completed = subprocess.run(
        [str(executable)],
        cwd=executable.parent,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120.0,
        check=False,
    )
    transcript_path = output_root / f"preflight-{role}.log"
    transcript_path.write_text(
        completed.stdout, encoding="utf-8", newline="\n"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{role} preflight exited with status {completed.returncode}"
        )
    lines = completed.stdout.splitlines()
    if lines.count("uciok") != 1 or lines.count("readyok") != 1:
        raise RuntimeError(f"{role} preflight did not complete UCI synchronization")
    declared = {
        match.group(1)
        for line in lines
        if (match := re.match(r"^option name (.+?) type ", line)) is not None
    }
    missing = sorted(name for name, _value in options if name not in declared)
    if missing:
        raise RuntimeError(
            f"{role} preflight is missing UCI options: {', '.join(missing)}"
        )
    bestmoves = [
        match.group(1)
        for line in lines
        if (match := re.match(r"^bestmove\s+(\S+)", line)) is not None
    ]
    if len(bestmoves) != 1 or bestmoves[0] in {"(none)", "0000"}:
        raise RuntimeError(f"{role} preflight did not publish one legal best move")
    lowered = completed.stdout.lower()
    fatal_markers = (
        "failed to load",
        "file not found",
        "unable to load",
        "network error",
    )
    marker = next((item for item in fatal_markers if item in lowered), None)
    if marker is not None:
        raise RuntimeError(f"{role} preflight reported {marker!r}")
    expected_network = (
        args.candidate_network if role == "candidate" else args.baseline_network
    )
    if str(expected_network).lower() not in lowered:
        raise RuntimeError(f"{role} preflight did not confirm the network path")
    if role == "candidate" and EXPECTED_NETWORK_SHA256.lower() not in lowered:
        raise RuntimeError(
            "candidate preflight did not confirm the frozen network SHA-256"
        )
    return {
        "role": role,
        "executable_sha256": sha256_file(executable),
        "options": dict(options),
        "bestmove": bestmoves[0],
        "transcript": {
            "path": str(transcript_path),
            "sha256": sha256_file(transcript_path),
            "size": transcript_path.stat().st_size,
        },
        "status": "PASS",
    }


def authenticate_engines(
    args: argparse.Namespace, output_root: Path
) -> dict[str, object]:
    results = {
        role: authenticate_engine(args, role, output_root)
        for role in ("candidate", "baseline")
    }
    return {
        "schema": "ALICE_RELEASE_PANEL_PREFLIGHT_V1",
        "completed_at": utc_now(),
        "results": results,
        "status": "PASS",
    }


def build_command(
    args: argparse.Namespace,
    spec: PanelSpec,
    pgn_path: Path,
) -> list[str]:
    candidate = [
        f"cmd={args.candidate}",
        f"dir={args.candidate.parent}",
        f"name={CANDIDATE_NAME}",
        "proto=uci",
        f"tc={spec.tc}",
        "timemargin=0",
    ]
    add_options(
        candidate,
        (
            ("Threads", "1"),
            ("Hash", str(spec.hash_mb)),
            ("Move Overhead", "10"),
            ("Use NNUE", "true"),
            ("Alice Evaluation", "Legacy"),
            ("Alice_Frozen_Network", "true"),
            ("EvalFile", str(args.candidate_network)),
        ),
    )

    baseline = [
        f"cmd={args.baseline}",
        f"dir={args.baseline.parent}",
        f"name={BASELINE_NAME}",
        "proto=uci",
        f"tc={spec.tc}",
        "timemargin=0",
    ]
    add_options(
        baseline,
        (
            ("UCI_Variant", "alice"),
            ("Threads", "1"),
            ("Hash", str(spec.hash_mb)),
            ("Move Overhead", "10"),
            ("Use NNUE", "true"),
            ("EvalFile", str(args.baseline_network)),
        ),
    )

    return [
        str(args.referee_python),
        str(args.referee),
        "-repeat",
        "-variant",
        "alice",
        "-concurrency",
        str(args.concurrency_per_tc),
        "-games",
        str(spec.games),
        "--max-plies",
        "900",
        "-engine",
        *candidate,
        "-engine",
        *baseline,
        "-openings",
        f"file={args.book}",
        "format=epd",
        "order=random",
        "start=1",
        "-srand",
        str(spec.seed),
        "-pgnout",
        str(pgn_path),
    ]


PGN_HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+) "(.*)"\]$')


def load_book_fens(path: Path) -> list[str]:
    fens: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for raw in handle:
            fen = raw.split(";", 1)[0].strip()
            if fen and not fen.startswith("#"):
                fens.append(fen)
    if not fens:
        raise RuntimeError("opening book is empty")
    return fens


def expected_opening_schedule(book: Path, spec: PanelSpec) -> list[str]:
    fens = load_book_fens(book)
    rng = random.Random(spec.seed)
    indices: list[int] = []
    required = spec.games // 2
    while len(indices) < required:
        cycle = list(range(len(fens)))
        rng.shuffle(cycle)
        indices.extend(cycle)
    return [fens[index] for index in indices[:required]]


def parse_pgn(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="ascii", errors="strict")
    starts = [match.start() for match in re.finditer(r"(?m)^\[Event ", text)]
    if not starts:
        raise RuntimeError(f"PGN has no games: {path}")
    starts.append(len(text))
    games: list[dict[str, object]] = []
    for index in range(len(starts) - 1):
        block = text[starts[index] : starts[index + 1]].strip()
        lines = block.splitlines()
        headers: dict[str, str] = {}
        cursor = 0
        while cursor < len(lines) and lines[cursor].startswith("["):
            match = PGN_HEADER_RE.fullmatch(lines[cursor])
            if match is None:
                raise RuntimeError(f"malformed PGN header: {lines[cursor]!r}")
            key, value = match.group(1, 2)
            if key in headers:
                raise RuntimeError(f"duplicate PGN header {key}")
            headers[key] = value
            cursor += 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        movetext = "\n".join(lines[cursor:]).strip()
        if not movetext:
            raise RuntimeError("PGN game has no movetext")
        result = headers.get("Result", "")
        if result not in {"1-0", "0-1", "1/2-1/2"}:
            raise RuntimeError(f"PGN has invalid result {result!r}")
        if not movetext.endswith(result):
            raise RuntimeError("PGN movetext does not end in its declared result")
        games.append({"headers": headers, "movetext": movetext})
    return games


def reconcile_time_control(
    args: argparse.Namespace,
    spec: PanelSpec,
    tracker: PanelTracker,
    pgn_path: Path,
    log_path: Path,
) -> dict[str, object]:
    observations = tracker.observations()
    pgn_games = parse_pgn(pgn_path)
    if len(pgn_games) != spec.games:
        raise RuntimeError(
            f"{spec.label} PGN has {len(pgn_games)} games, expected {spec.games}"
        )

    expected_fens = expected_opening_schedule(args.book, spec)
    by_pair: dict[int, list[dict[str, object]]] = defaultdict(list)
    for game in pgn_games:
        headers = game["headers"]
        assert isinstance(headers, dict)
        required_headers = {
            "Event",
            "Round",
            "White",
            "Black",
            "Result",
            "SetUp",
            "FEN",
            "Variant",
            "TimeControl",
            "PlyCount",
            "GameEndTime",
        }
        missing = sorted(required_headers - set(headers))
        if missing:
            raise RuntimeError(f"{spec.label} PGN missing headers: {missing}")
        forbidden = sorted(FORBIDDEN_PGN_HEADERS.intersection(headers))
        if forbidden:
            raise RuntimeError(
                f"{spec.label} PGN contains abnormal headers: {forbidden}"
            )
        if headers["Event"] != "uci_pair_runner":
            raise RuntimeError(f"{spec.label} PGN event mismatch")
        if headers["Variant"] != "alice" or headers["TimeControl"] != spec.tc:
            raise RuntimeError(f"{spec.label} PGN variant/time-control mismatch")
        if headers["SetUp"] != "1":
            raise RuntimeError(f"{spec.label} PGN is missing explicit setup")
        try:
            pair = int(headers["Round"])
            plies = int(headers["PlyCount"])
        except ValueError as error:
            raise RuntimeError(f"{spec.label} PGN numeric header is invalid") from error
        if not 1 <= pair <= spec.games // 2:
            raise RuntimeError(f"{spec.label} PGN pair ordinal is out of range")
        if not 0 <= plies < 900:
            raise RuntimeError(f"{spec.label} PGN reached the safety ply limit")
        by_pair[pair].append(game)

    game_evidence: list[dict[str, object]] = []
    for pair in range(1, spec.games // 2 + 1):
        records = by_pair.get(pair, [])
        if len(records) != 2:
            raise RuntimeError(
                f"{spec.label} pair {pair} has {len(records)} PGN games"
            )
        headers_by_white: dict[str, dict[str, str]] = {}
        for record in records:
            headers = record["headers"]
            assert isinstance(headers, dict)
            white = str(headers["White"])
            if white in headers_by_white:
                raise RuntimeError(f"{spec.label} pair {pair} repeats White={white}")
            headers_by_white[white] = headers
        if set(headers_by_white) != {CANDIDATE_NAME, BASELINE_NAME}:
            raise RuntimeError(f"{spec.label} pair {pair} does not swap colors")
        candidate_white = headers_by_white[CANDIDATE_NAME]
        baseline_white = headers_by_white[BASELINE_NAME]
        if (
            candidate_white["Black"] != BASELINE_NAME
            or baseline_white["Black"] != CANDIDATE_NAME
        ):
            raise RuntimeError(f"{spec.label} pair {pair} engine names mismatch")
        expected_fen = expected_fens[pair - 1]
        if (
            candidate_white["FEN"] != expected_fen
            or baseline_white["FEN"] != expected_fen
        ):
            raise RuntimeError(f"{spec.label} pair {pair} opening mismatch")

        opening_sha = hashlib.sha256(expected_fen.encode("utf-8")).hexdigest().upper()
        for game_number, headers in (
            (2 * pair - 1, candidate_white),
            (2 * pair, baseline_white),
        ):
            observation = observations[game_number]
            if (
                observation["white"] != headers["White"]
                or observation["black"] != headers["Black"]
                or observation["result"] != headers["Result"]
            ):
                raise RuntimeError(
                    f"{spec.label} game {game_number} log/PGN mismatch"
                )
            game_evidence.append(
                {
                    "game_number": game_number,
                    "pair": pair,
                    "opening_sha256": opening_sha,
                    "white": headers["White"],
                    "black": headers["Black"],
                    "result": headers["Result"],
                    "reason": observation["reason"],
                    "plies": int(headers["PlyCount"]),
                }
            )

    reason_histogram = Counter(
        str(observation["reason"]) for observation in observations.values()
    )
    schedule_sha = hashlib.sha256(canonical_json_bytes(expected_fens)).hexdigest()
    return {
        "schema": "ALICE_RELEASE_PANEL_RECONCILIATION_V1",
        "games": game_evidence,
        "game_count": len(game_evidence),
        "pair_count": spec.games // 2,
        "opening_schedule_sha256": schedule_sha.upper(),
        "termination_histogram": dict(sorted(reason_histogram.items())),
        "abnormal_counters": {
            "time_forfeits": 0,
            "crashes": 0,
            "disconnects": 0,
            "illegal_moves": 0,
            "protocol_failures": 0,
            "adjudications": 0,
            "safety_ply_limits": 0,
            "discarded_games": 0,
            "incomplete_pairs": 0,
            "administrative_results": 0,
        },
        "pgn": {
            "path": str(pgn_path),
            "sha256": sha256_file(pgn_path),
            "size": pgn_path.stat().st_size,
        },
        "log": {
            "path": str(log_path),
            "sha256": sha256_file(log_path),
            "size": log_path.stat().st_size,
        },
        "status": "PASS",
    }


def terminate_active() -> None:
    with ACTIVE_LOCK:
        active = list(ACTIVE_PROCESSES)
    for process in active:
        terminate_process_tree(process)


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass


def format_table(trackers: dict[str, PanelTracker]) -> str:
    rows = [
        f"Alice release panel status at {utc_now()}",
        "TC    Games       W-D-L          Penta                 Elo +/- CI95      LOS      State",
    ]
    for spec in PANEL_SPECS:
        snapshot = trackers[spec.label].snapshot()
        stats = snapshot["statistics"]
        wdl_values = snapshot["wdl"]
        assert isinstance(wdl_values, dict)
        wdl = (
            f"{wdl_values['wins']}-{wdl_values['draws']}-"
            f"{wdl_values['losses']}"
        )
        penta = "/".join(str(value) for value in snapshot["pentanomial"])
        rows.append(
            f"{spec.label:<5} {snapshot['games']:>4}/{spec.games:<4} "
            f"{wdl:<14} {penta:<21} "
            f"{stats['elo']:+7.2f} +/- {stats['ci95']:<7.2f} "
            f"{100.0 * stats['los']:>6.1f}%  {snapshot['status']}"
        )
    return "\n".join(rows)


def monitor_table(
    trackers: dict[str, PanelTracker], done: threading.Event, interval: float
) -> None:
    emit(format_table(trackers))
    while not done.wait(interval):
        emit(format_table(trackers))


def run_time_control(
    args: argparse.Namespace,
    spec: PanelSpec,
    tracker: PanelTracker,
    abort: threading.Event,
) -> dict[str, object]:
    tc_root = args.output_root / spec.label.lower()
    tc_root.mkdir(parents=True, exist_ok=False)
    pgn_path = tc_root / "games.pgn"
    log_path = tc_root / "referee.log"
    command = build_command(args, spec, pgn_path)
    tracker.set_status("RUNNING")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
        start_new_session=os.name != "nt",
    )
    with ACTIVE_LOCK:
        ACTIVE_PROCESSES.add(process)

    assert process.stdout is not None
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            for raw in process.stdout:
                line = raw.rstrip("\r\n")
                log.write(line + "\n")
                log.flush()
                tracker.consume(line)
                if abort.is_set() and process.poll() is None:
                    terminate_process_tree(process)
        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            terminate_process_tree(process)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        with ACTIVE_LOCK:
            ACTIVE_PROCESSES.discard(process)

    if abort.is_set():
        raise RuntimeError(f"{spec.label} aborted after another panel failure")
    if return_code != 0:
        raise RuntimeError(f"{spec.label} referee exited with status {return_code}")

    tracker.require_complete()
    reconciliation = reconcile_time_control(
        args, spec, tracker, pgn_path, log_path
    )
    tracker.set_status("PASS")
    result = tracker.snapshot()
    result["reconciliation"] = reconciliation
    write_json(tc_root / "result.json", result)
    return result


def make_manifest(
    args: argparse.Namespace,
    assets: dict[str, object],
    preflight: dict[str, object],
    commands: dict[str, list[str]],
    started_at: str,
) -> dict[str, object]:
    return {
        "schema": "ALICE_RELEASE_PANEL_V1",
        "started_at": started_at,
        "runner_commit": args.runner_commit,
        "candidate": {
            "commit": args.candidate_commit,
            "tree": args.candidate_tree,
            "evaluation": "LegacyAliceExact",
        },
        "baseline": {
            "commit": args.baseline_commit,
            "label": "Fairy-Stockfish 040925",
            "evaluation": "Fairy-Stockfish legacy Alice NNUE",
        },
        "referee_commit": args.referee_commit,
        "assets": assets,
        "preflight": preflight,
        "panel": [asdict(spec) for spec in PANEL_SPECS],
        "commands": commands,
        "contract": {
            "variant": "alice",
            "paired_colors": True,
            "opening_order": "random",
            "shared_opening_seed": OPENING_SEED,
            "threads_per_engine": 1,
            "hash_mb_per_engine": 512,
            "move_overhead_ms": 10,
            "time_margin_ms": 0,
            "maximum_plies": 900,
            "syzygy": False,
            "score_adjudication": False,
            "early_score_stop": False,
            "native_v2": {
                "feature_encoder_source_allowed": True,
                "rejected_checkpoint_sha256": REJECTED_NATIVE_V2_CHECKPOINT_SHA256,
                "rejected_network_sha256": REJECTED_NATIVE_V2_NETWORK_SHA256,
                "rejected_inference_commit": REJECTED_NATIVE_V2_INFERENCE_COMMIT,
                "rejected_artifacts_included": False,
            },
            "report_interval_seconds": args.report_interval,
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "runtime_boundary": environment_inventory(),
    }


def run_panel(args: argparse.Namespace) -> int:
    original_assets = validate_inputs(args)
    dry_run_commands = {
        spec.label: build_command(
            args, spec, args.output_root / spec.label.lower() / "games.pgn"
        )
        for spec in PANEL_SPECS
    }
    emit("Input authentication: PASS")

    if args.dry_run:
        for spec in PANEL_SPECS:
            emit(
                f"[{spec.label}] "
                f"{subprocess.list2cmdline(dry_run_commands[spec.label])}"
            )
        return 0

    if args.output_root.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started_at = utc_now()
    try:
        snapshots = snapshot_inputs(args, original_assets)
        commands = {
            spec.label: build_command(
                args, spec, args.output_root / spec.label.lower() / "games.pgn"
            )
            for spec in PANEL_SPECS
        }
        preflight = authenticate_engines(args, args.output_root)
    except BaseException as error:
        write_json(
            args.output_root / "INVALID.json",
            {
                "schema": "ALICE_RELEASE_PANEL_INVALID_V1",
                "started_at": started_at,
                "failed_at": utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "stage": "input-custody-or-engine-preflight",
            },
        )
        raise
    preflight_path = args.output_root / "preflight.json"
    write_json(preflight_path, preflight)
    preflight_reference = {
        "path": str(preflight_path),
        "sha256": sha256_file(preflight_path),
    }
    manifest = make_manifest(
        args,
        {"original": original_assets, "snapshots": snapshots},
        preflight_reference,
        commands,
        started_at,
    )
    manifest_path = args.output_root / "manifest.json"
    write_json(manifest_path, manifest)

    trackers = {spec.label: PanelTracker(spec) for spec in PANEL_SPECS}
    done = threading.Event()
    abort = threading.Event()
    monitor = threading.Thread(
        target=monitor_table,
        args=(trackers, done, args.report_interval),
        name="alice-release-panel-monitor",
        daemon=True,
    )
    monitor.start()

    results: dict[str, dict[str, object]] = {}
    error: BaseException | None = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    run_time_control,
                    args,
                    spec,
                    trackers[spec.label],
                    abort,
                ): spec.label
                for spec in PANEL_SPECS
            }
            for future in concurrent.futures.as_completed(futures):
                label = futures[future]
                try:
                    results[label] = future.result()
                except BaseException as exc:
                    if error is None:
                        error = exc
                    trackers[label].set_status("FAIL")
                    abort.set()
                    terminate_active()
    finally:
        done.set()
        monitor.join(timeout=2.0)
        emit(format_table(trackers))
        terminate_active()

    if error is not None:
        try:
            postflight = revalidate_custody(args, snapshots)
        except BaseException as custody_error:
            postflight = {
                "status": "FAIL",
                "error": f"{type(custody_error).__name__}: {custody_error}",
            }
        invalid = {
            "schema": "ALICE_RELEASE_PANEL_INVALID_V1",
            "started_at": started_at,
            "failed_at": utc_now(),
            "error": f"{type(error).__name__}: {error}",
            "states": {
                label: tracker.snapshot() for label, tracker in trackers.items()
            },
            "postflight_custody": postflight,
        }
        write_json(args.output_root / "INVALID.json", invalid)
        raise error

    ordered_results = {spec.label: results[spec.label] for spec in PANEL_SPECS}
    try:
        postflight = revalidate_custody(args, snapshots)
    except BaseException as custody_error:
        write_json(
            args.output_root / "INVALID.json",
            {
                "schema": "ALICE_RELEASE_PANEL_INVALID_V1",
                "started_at": started_at,
                "failed_at": utc_now(),
                "error": (
                    f"{type(custody_error).__name__}: {custody_error}"
                ),
                "stage": "postflight-input-custody",
                "states": ordered_results,
            },
        )
        raise
    postflight_path = args.output_root / "postflight.json"
    write_json(postflight_path, postflight)
    output_hashes: dict[str, dict[str, str]] = {}
    for spec in PANEL_SPECS:
        tc_root = args.output_root / spec.label.lower()
        output_hashes[spec.label] = {
            name: sha256_file(tc_root / name)
            for name in ("games.pgn", "referee.log", "result.json")
        }

    all_games = sum(item["games"] for item in ordered_results.values())
    expected_games = sum(spec.games for spec in PANEL_SPECS)
    all_games_complete = all_games == expected_games
    all_pairs_complete = all(
        ordered_results[spec.label]["complete_pairs"] == spec.games // 2
        for spec in PANEL_SPECS
    )
    zero_infrastructure_defects = all(
        not ordered_results[spec.label]["anomalies"] for spec in PANEL_SPECS
    )
    all_reconciled = all(
        ordered_results[spec.label]["reconciliation"]["status"] == "PASS"
        for spec in PANEL_SPECS
    )
    native_v2_excluded = (
        original_assets["candidate_build_receipt"]["validated"]
        ["native_v2_exclusion"]["checkpoint_present"]
        is False
        and original_assets["candidate_build_receipt"]["validated"]
        ["native_v2_exclusion"]["network_present"]
        is False
        and original_assets["candidate_build_receipt"]["validated"]
        ["native_v2_exclusion"]["inference_commit_ancestor"]
        is False
    )
    passed = all(
        (
            all_games_complete,
            all_pairs_complete,
            zero_infrastructure_defects,
            all_reconciled,
            native_v2_excluded,
            postflight["status"] == "PASS",
        )
    )
    if not passed:
        gates = {
            "all_games_complete": all_games_complete,
            "all_pairs_complete": all_pairs_complete,
            "zero_infrastructure_defects": zero_infrastructure_defects,
            "all_pgn_log_reconciled": all_reconciled,
            "immutable_input_custody_revalidated": postflight["status"] == "PASS",
            "native_v2_excluded": native_v2_excluded,
            "passed": False,
        }
        write_json(
            args.output_root / "INVALID.json",
            {
                "schema": "ALICE_RELEASE_PANEL_INVALID_V1",
                "started_at": started_at,
                "failed_at": utc_now(),
                "error": "derived final panel gates did not all pass",
                "stage": "derived-final-gates",
                "states": ordered_results,
                "gates": gates,
            },
        )
        raise RuntimeError("derived final panel gates did not all pass")
    receipt = {
        "schema": "ALICE_RELEASE_PANEL_RECEIPT_V1",
        "started_at": started_at,
        "completed_at": utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "preflight_sha256": sha256_file(preflight_path),
        "postflight_sha256": sha256_file(postflight_path),
        "results": ordered_results,
        "output_sha256": output_hashes,
        "gates": {
            "all_games_complete": all_games_complete,
            "all_pairs_complete": all_pairs_complete,
            "zero_infrastructure_defects": zero_infrastructure_defects,
            "all_pgn_log_reconciled": all_reconciled,
            "immutable_input_custody_revalidated": postflight["status"] == "PASS",
            "native_v2_excluded": native_v2_excluded,
            "passed": passed,
        },
    }
    write_json(args.output_root / "panel-receipt.json", receipt)
    emit("Alice release panel PASS: 700/500/300 games complete with zero defects")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=path_arg)
    parser.add_argument("--candidate-sha256", required=True, type=checked_sha)
    parser.add_argument("--candidate-commit", required=True, type=checked_commit)
    parser.add_argument("--candidate-tree", required=True, type=checked_commit)
    parser.add_argument("--candidate-source-root", required=True, type=path_arg)
    parser.add_argument("--candidate-build-receipt", required=True, type=path_arg)
    parser.add_argument(
        "--candidate-build-receipt-sha256", required=True, type=checked_sha
    )
    parser.add_argument("--candidate-network", required=True, type=path_arg)
    parser.add_argument(
        "--candidate-network-sha256", required=True, type=checked_sha
    )
    parser.add_argument("--baseline", required=True, type=path_arg)
    parser.add_argument("--baseline-sha256", required=True, type=checked_sha)
    parser.add_argument("--baseline-commit", required=True, type=checked_commit)
    parser.add_argument("--baseline-network", required=True, type=path_arg)
    parser.add_argument(
        "--baseline-network-sha256", required=True, type=checked_sha
    )
    parser.add_argument("--referee", required=True, type=path_arg)
    parser.add_argument("--referee-sha256", required=True, type=checked_sha)
    parser.add_argument("--referee-commit", required=True, type=checked_commit)
    parser.add_argument("--referee-python", required=True, type=path_arg)
    parser.add_argument(
        "--referee-python-sha256", required=True, type=checked_sha
    )
    parser.add_argument("--book", required=True, type=path_arg)
    parser.add_argument("--book-sha256", required=True, type=checked_sha)
    parser.add_argument("--runner-commit", required=True, type=checked_commit)
    parser.add_argument("--output-root", required=True, type=path_arg)
    parser.add_argument("--concurrency-per-tc", type=int, default=4)
    parser.add_argument("--report-interval", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.concurrency_per_tc != 4:
        parser.error("--concurrency-per-tc is frozen at 4")
    if args.report_interval <= 0:
        parser.error("--report-interval must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        return run_panel(args)
    except KeyboardInterrupt:
        terminate_active()
        emit("Interrupted; the complete panel is invalid")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
