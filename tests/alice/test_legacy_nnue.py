"""Black-box checks for the exact historical Alice network bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest


TEST_DIRECTORY = Path(__file__).resolve().parent
FIXTURE_PATH = TEST_DIRECTORY / "fixtures" / "legacy-nnue-v1.json"
REPOSITORY = TEST_DIRECTORY.parent.parent


def default_engine_path() -> Path:
    repository = TEST_DIRECTORY.parent.parent
    windows = repository / "src" / "stockfish.exe"
    return windows if windows.exists() else repository / "src" / "stockfish"


ENGINE_PATH = default_engine_path()
NETWORK_PATH: Path | None = None


class UciSession:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [str(ENGINE_PATH)],
            cwd=REPOSITORY,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="ascii",
            errors="replace",
            bufsize=1,
        )
        self.pending: queue.Queue[str] = queue.Queue()
        self.lines: list[str] = []
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.pending.put(line.rstrip("\r\n"))

    def send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def wait_for(self, pattern: str, timeout: float = 30.0) -> str:
        expression = re.compile(pattern)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.pending.get(timeout=min(0.1, max(0.0, deadline - time.monotonic())))
            except queue.Empty:
                continue
            self.lines.append(line)
            if expression.search(line):
                return line
        raise AssertionError(
            f"Timed out waiting for {pattern!r}. Executable output:\n" + "\n".join(self.lines[-100:])
        )

    def close(self) -> None:
        if self.process.poll() is None:
            self.send("quit")
            self.process.wait(timeout=5)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.reader.join(timeout=1)

    def __enter__(self) -> "UciSession":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def fatal_probe(network: Path, *extra_commands: str) -> subprocess.CompletedProcess[str]:
    commands = [
        *extra_commands,
        f"setoption name EvalFile value {network}",
        "position startpos",
        "go depth 1",
        "quit",
        "",
    ]
    return subprocess.run(
        [str(ENGINE_PATH)],
        cwd=REPOSITORY,
        input="\n".join(commands),
        text=True,
        capture_output=True,
        encoding="ascii",
        errors="replace",
        timeout=30,
        check=False,
    )


class LegacyNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not ENGINE_PATH.is_file():
            raise FileNotFoundError(f"Alice-Stockfish executable not found: {ENGINE_PATH}")
        if NETWORK_PATH is None:
            raise unittest.SkipTest("Pass --network to run the historical network checks.")
        if not NETWORK_PATH.is_file():
            raise FileNotFoundError(f"Historical Alice network not found: {NETWORK_PATH}")
        cls.document = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_frozen_network_load_and_exact_values(self) -> None:
        assert NETWORK_PATH is not None
        expected_hash = self.document["network"]["sha256"]
        with UciSession() as session:
            session.send("uci")
            session.wait_for(r"^uciok$")
            session.send(f"setoption name EvalFile value {NETWORK_PATH}")
            status = session.wait_for(r"LegacyAliceExact loaded", timeout=60)
            self.assertIn("mode=frozen-baseline", status)
            self.assertIn(f"sha256={expected_hash}", status)
            self.assertIn("version=0x7AF32F20", status)
            self.assertIn("architecture=0x3C103E72", status)

            for case in self.document["positions"]:
                with self.subTest(position=case["id"]):
                    session.send(f"position fen {case['fen']}")
                    session.send("eval")
                    line = session.wait_for(r"^legacy_nnue raw ")
                    self.assertEqual(
                        line,
                        f"legacy_nnue raw {case['raw']} adjusted {case['adjusted']}",
                    )

    def test_network_backed_search_uses_the_legacy_evaluator(self) -> None:
        assert NETWORK_PATH is not None
        with UciSession() as session:
            session.send(f"setoption name EvalFile value {NETWORK_PATH}")
            session.wait_for(r"LegacyAliceExact loaded", timeout=60)
            session.send("position startpos")
            session.send("go depth 1")
            bestmove = session.wait_for(r"^bestmove ", timeout=30)
            self.assertEqual(bestmove, "bestmove e2e3")
            self.assertTrue(
                any(line.startswith("info depth 1 ") and "score cp 0" not in line for line in session.lines),
                "\n".join(session.lines),
            )

    def test_incremental_evaluation_matches_full_refresh(self) -> None:
        assert NETWORK_PATH is not None
        cases = (
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2),
            ("7k/P7/8/8/8/8/8/7K w - - 0 1", 1),
            ("7k/8/8/8/8/|p7/R7/7K w - - 0 1", 1),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 1),
            ("8/8/8/8/8/8/2k5/4K3 w - - 0 1", 2),
        )
        with UciSession() as session:
            session.send(f"setoption name EvalFile value {NETWORK_PATH}")
            session.wait_for(r"LegacyAliceExact loaded", timeout=60)
            for fen, depth in cases:
                with self.subTest(fen=fen, depth=depth):
                    session.send(f"position fen {fen}")
                    session.send(f"alice_verify_incremental {depth}")
                    line = session.wait_for(
                        rf"^legacy_nnue incremental verified positions \d+ depth {depth}$",
                        timeout=60,
                    )
                    positions = int(line.split()[4])
                    self.assertGreater(positions, 1)

    def test_canonical_bench_is_deterministic_and_matches_its_fixture(self) -> None:
        assert NETWORK_PATH is not None
        with UciSession() as session:
            session.send(f"setoption name EvalFile value {NETWORK_PATH}")
            session.wait_for(r"LegacyAliceExact loaded", timeout=60)

            session.send("bench")
            default_line = session.wait_for(r"^Nodes searched\s+:\s+\d+$", timeout=90)
            default_nodes = int(default_line.rsplit(maxsplit=1)[1])
            self.assertEqual(default_nodes, 202963)

            session.send("bench 16 1 12 tests/alice/fixtures/bench-v1.epd depth")
            fixture_line = session.wait_for(r"^Nodes searched\s+:\s+\d+$", timeout=90)
            fixture_nodes = int(fixture_line.rsplit(maxsplit=1)[1])
            self.assertEqual(fixture_nodes, default_nodes)

    def test_content_addressed_basename_loads_by_verified_bytes(self) -> None:
        assert NETWORK_PATH is not None
        with tempfile.TemporaryDirectory() as temporary:
            cached_network = Path(temporary) / "9F9E5570"
            shutil.copyfile(NETWORK_PATH, cached_network)
            with UciSession() as session:
                session.send(f"setoption name EvalFile value {cached_network}")
                loaded = session.wait_for(
                    rf'LegacyAliceExact loaded path="{re.escape(str(cached_network))}"',
                    timeout=60,
                )
                self.assertIn("9F9E5570", loaded)

    def test_rejections_clear_the_previous_evaluator_and_exit_nonzero(self) -> None:
        assert NETWORK_PATH is not None
        missing = NETWORK_PATH.parent / "missing" / NETWORK_PATH.name
        cases = [
            (
                "missing-after-valid-load",
                missing,
                "Unable to open EvalFile",
                (f"setoption name EvalFile value {NETWORK_PATH}",),
            ),
        ]
        for label, path, diagnostic, commands in cases:
            with self.subTest(case=label):
                result = fatal_probe(path, *commands)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(diagnostic, result.stdout)
                self.assertIn("CRITICAL ERROR", result.stdout)
                if label == "missing-after-valid-load":
                    self.assertIn("LegacyAliceExact loaded", result.stdout)

        original = NETWORK_PATH.read_bytes()
        description_size = int.from_bytes(original[8:12], "little")
        transformer_offset = 12 + description_size
        first_stack_offset = (
            transformer_offset
            + 4
            + 512 * 2
            + 45056 * 512 * 2
            + 45056 * 8 * 4
        )
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / NETWORK_PATH.name
            mutations: list[tuple[str, bytes, str, tuple[str, ...]]] = [
                ("wrong-version", b"\x00\x00\x00\x00" + original[4:12], "serialization version", ()),
                ("wrong-architecture", original[:4] + b"\x00\x00\x00\x00" + original[8:12], "architecture", ()),
                ("truncated", original[:128], "structural length", ()),
                ("trailing", original + b"\x00", "structural length", ()),
                (
                    "wrong-transformer",
                    original[:transformer_offset]
                    + bytes([original[transformer_offset] ^ 1])
                    + original[transformer_offset + 1 :],
                    "feature-transformer hash",
                    ("setoption name Alice_Frozen_Network value false",),
                ),
                (
                    "wrong-layer-stack",
                    original[:first_stack_offset]
                    + bytes([original[first_stack_offset] ^ 1])
                    + original[first_stack_offset + 1 :],
                    "layer-stack hash",
                    ("setoption name Alice_Frozen_Network value false",),
                ),
                (
                    "wrong-frozen-checksum",
                    original[:12]
                    + bytes([original[12] ^ 1])
                    + original[13:],
                    "SHA-256 does not match",
                    (),
                ),
            ]
            for label, content, diagnostic, commands in mutations:
                with self.subTest(case=label):
                    target.write_bytes(content)
                    result = fatal_probe(target, *commands)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(diagnostic, result.stdout)
                    self.assertIn("CRITICAL ERROR", result.stdout)

    def test_explicit_compatible_mode_accepts_a_nonbaseline_description(self) -> None:
        assert NETWORK_PATH is not None
        original = bytearray(NETWORK_PATH.read_bytes())
        original[12] ^= 1
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / NETWORK_PATH.name
            target.write_bytes(original)
            with UciSession() as session:
                session.send("setoption name Alice_Frozen_Network value false")
                session.send(f"setoption name EvalFile value {target}")
                status = session.wait_for(
                    rf'LegacyAliceExact loaded path="{re.escape(str(target))}"',
                    timeout=60,
                )
                self.assertIn("mode=format-compatible", status)
                self.assertNotIn(self.document["network"]["sha256"], status)
                session.send("position startpos")
                session.send("eval")
                self.assertEqual(
                    session.wait_for(r"^legacy_nnue raw "),
                    "legacy_nnue raw 68 adjusted 72",
                )


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine", type=Path, default=default_engine_path())
    parser.add_argument("--network", type=Path)
    return parser.parse_known_args()


if __name__ == "__main__":
    arguments, unittest_arguments = parse_arguments()
    ENGINE_PATH = arguments.engine.resolve()
    NETWORK_PATH = arguments.network.resolve() if arguments.network else None
    unittest.main(argv=[sys.argv[0], *unittest_arguments], verbosity=2)
