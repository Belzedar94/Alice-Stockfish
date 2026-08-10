"""Authenticated wire, integer-stage, incremental, and search tests for AliceNative-v2."""

from __future__ import annotations

import argparse
import hashlib
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
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MANIFEST_BYTES = 1_462
HEADER_BYTES = 32
FEATURE_TENSOR_BYTES = 47_580_164
FC1_WEIGHT_OFFSET_IN_STACK = 4 + 16 * 4 + 16 * 1_024 + 32 * 4
STAGE_KEYS = (
    "featureAccumulator",
    "psqtAccumulator",
    "transformedByPerspective",
    "denseInput",
    "fc0Raw",
    "fc0Linear",
    "fc1Input",
    "fc1Raw",
    "fc1Linear",
    "fc2Raw",
    "psqtDifference",
    "materialRaw",
    "mixedRaw",
    "nativeNnueValue",
)
PHASE_FENS = (
    "8/8/8/8/8/8/2k5/4K3 w - - 0 1",
    "k7/8/8/r2pPK2/8/8/8/8 w - - 0 1",
    "8/5R2/2K1P3/4k3/8/b1PPpp1B/5p2/8 w - - 0 1",
    "2brrb2/8/p7/Q7/1p1kpPp1/1P1pN1K1/3P4/8 b - - 0 1",
    "k7/pppppppp/8/8/8/8/PPPPPPPP/7K w - - 0 1",
    "3qnrk1/4bp1p/1p2p1pP/p2bN3/1P1P1B2/P2BQ3/5PP1/4R1K1 w - - 9 28",
    "Rn6/1rbq1bk1/2p2n1p/2Bp1p2/3Pp1pP/1N2P1P1/2Q1NPB1/6K1 w - - 2 26",
    "r4rk1/1b2ppbp/pq4pn/2pp1PB1/1p2P3/1P1P1NN1/1PP3PP/R2Q1RK1 w - - 0 13",
)
FOCUSED_INCREMENTAL_FENS = (
    "7k/P7/8/8/8/8/8/7K w - - 0 1",
    "7k/8/8/8/8/|p7/R7/7K w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "4r2|k/8/8/8/8/8/8/4K3 w - - 0 1",
)


def default_engine_path() -> Path:
    repository = TEST_DIRECTORY.parent.parent
    windows = repository / "src" / "stockfish.exe"
    return windows if windows.exists() else repository / "src" / "stockfish"


ENGINE_PATH = default_engine_path()
NETWORK_PATH: Path | None = None
REFERENCE_INPUT: Path | None = None
REFERENCE_OUTPUT: Path | None = None
VERIFIER_PATH: Path | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def native_command_path(path: Path) -> str:
    # The dedicated parser must preserve native Windows backslashes even inside quotes.
    return f'"{path.resolve()}"'


class UciSession:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [str(ENGINE_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="ascii",
            bufsize=1,
        )
        self.lines: list[str] = []
        self.pending: queue.Queue[str] = queue.Queue()
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

    def wait_for(self, pattern: str, timeout: float = 180.0) -> str:
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
            f"Timed out waiting for {pattern!r}.\n" + "\n".join(self.lines[-100:])
        )

    def close(self) -> None:
        if self.process.poll() is None:
            self.send("quit")
            self.process.wait(timeout=10)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.reader.join(timeout=1)

    def __enter__(self) -> "UciSession":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def load_command(path: Path, sha256: str, *, tolerant: bool = False) -> str:
    verb = "alice_native_v2_try_load_file" if tolerant else "alice_native_v2_load_file"
    return f"{verb} {native_command_path(path)} {sha256}"


def mutate_copy(source: Path, destination: Path, offset: int, value: int) -> None:
    shutil.copyfile(source, destination)
    with destination.open("r+b") as stream:
        stream.seek(offset)
        stream.write(bytes([value]))


class AliceNativeV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if NETWORK_PATH is None:
            raise unittest.SkipTest("pass --network with the qualified M512 network")
        assert NETWORK_PATH is not None
        if not ENGINE_PATH.is_file():
            raise FileNotFoundError(f"Alice-Stockfish executable not found: {ENGINE_PATH}")
        if not NETWORK_PATH.is_file():
            raise FileNotFoundError(f"AliceNative-v2 network not found: {NETWORK_PATH}")
        cls.network_sha = file_sha256(NETWORK_PATH)

    def test_native_windows_path_load_selector_eval_and_search(self) -> None:
        assert NETWORK_PATH is not None
        with UciSession() as session:
            session.send(load_command(NETWORK_PATH, self.network_sha))
            status = session.wait_for(r"AliceNative-v2 M512 parameters loaded generation=1")
            self.assertIn(f"sha256={self.network_sha}", status)
            self.assertIn(str(NETWORK_PATH.resolve()), status)

            session.send("setoption name Alice Evaluation value NativeV2")
            session.wait_for(r"AliceNative-v2 M512 parameters loaded generation=1")
            session.send("position startpos")
            session.send("eval")
            session.wait_for(r"^alice_native_v2 value -?[0-9]+ generation 1")
            session.send("go depth 1")
            session.wait_for(r"^bestmove [a-h][1-8][a-h][1-8][qrbn]?")
            self.assertFalse(any("CRITICAL ERROR" in line for line in session.lines))

    def test_reference_trace_matches_every_sealed_integer_stage(self) -> None:
        if not REFERENCE_INPUT or not REFERENCE_OUTPUT or not VERIFIER_PATH:
            self.skipTest("pass --reference-input, --reference-output, and --verifier")
        assert NETWORK_PATH is not None

        with UciSession() as session:
            session.send(load_command(NETWORK_PATH, self.network_sha))
            session.wait_for(r"AliceNative-v2 M512 parameters loaded generation=1")
            session.send("position startpos")
            session.send("alice_native_v2_eval_trace")
            line = session.wait_for(r"^alice_native_v2_integer_trace \{")
        engine_trace = json.loads(line.split(" ", 1)[1])
        qualification_input = json.loads(REFERENCE_INPUT.read_text(encoding="utf-8"))
        independent = json.loads(REFERENCE_OUTPUT.read_text(encoding="utf-8"))

        self.assertEqual(engine_trace["schema"], "alice-native-v2-index-trace-v1")
        self.assertEqual(engine_trace["fen"], START_FEN)
        self.assertEqual(engine_trace["pieceFeatures"], qualification_input["pieceFeatures"])
        self.assertEqual(engine_trace["networkSha256"], self.network_sha)
        for key in STAGE_KEYS:
            self.assertEqual(engine_trace[key], independent[key], key)

        reference_input = {
            key: engine_trace[key]
            for key in ("schema", "fen", "sideToMove", "pieceCount", "phase", "pieceFeatures")
        }
        with tempfile.TemporaryDirectory(prefix="alice-native-v2-reference-") as temporary:
            trace_path = Path(temporary) / "engine-index-trace.json"
            trace_path.write_text(
                json.dumps(reference_input, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(VERIFIER_PATH),
                    "trace",
                    "--network",
                    str(NETWORK_PATH),
                    "--input",
                    str(trace_path),
                ],
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
                timeout=180,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        regenerated = json.loads(result.stdout)
        for key in STAGE_KEYS:
            self.assertEqual(engine_trace[key], regenerated[key], key)

        with tempfile.TemporaryDirectory(prefix="alice-native-v2-all-phases-") as temporary:
            root = Path(temporary)
            with UciSession() as session:
                session.send(load_command(NETWORK_PATH, self.network_sha))
                session.wait_for(r"AliceNative-v2 M512 parameters loaded generation=1")
                for expected_phase, fen in enumerate(PHASE_FENS):
                    session.send(f"position fen {fen}")
                    session.send("alice_native_v2_eval_trace")
                    line = session.wait_for(r"^alice_native_v2_integer_trace \{")
                    phase_trace = json.loads(line.split(" ", 1)[1])
                    self.assertEqual(phase_trace["phase"], expected_phase)
                    phase_input = {
                        key: phase_trace[key]
                        for key in (
                            "schema",
                            "fen",
                            "sideToMove",
                            "pieceCount",
                            "phase",
                            "pieceFeatures",
                        )
                    }
                    trace_path = root / f"phase-{expected_phase}.json"
                    trace_path.write_text(
                        json.dumps(phase_input, sort_keys=True, separators=(",", ":")),
                        encoding="utf-8",
                    )
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(VERIFIER_PATH),
                            "trace",
                            "--network",
                            str(NETWORK_PATH),
                            "--input",
                            str(trace_path),
                        ],
                        text=True,
                        encoding="utf-8",
                        capture_output=True,
                        check=False,
                        timeout=180,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    phase_reference = json.loads(result.stdout)
                    for key in STAGE_KEYS:
                        self.assertEqual(phase_trace[key], phase_reference[key], key)

    def test_full_refresh_matches_incremental_and_undo(self) -> None:
        assert NETWORK_PATH is not None
        with UciSession() as session:
            session.send(load_command(NETWORK_PATH, self.network_sha))
            session.wait_for(r"AliceNative-v2 M512 parameters loaded generation=1")
            session.send("position startpos")
            session.send("alice_native_v2_verify_incremental 2")
            report = session.wait_for(r"^alice_native_v2 incremental verified generation 1")
            session.send("alice_native_v2_verify_search_session 2")
            session_report = session.wait_for(r"^alice_native_v2 session verified generation 1")
        fields = dict(re.findall(r"([a-z_]+) ([0-9]+)", report))
        self.assertEqual(int(fields["positions"]), 421)
        self.assertEqual(int(fields["transitions"]), 420)
        self.assertEqual(int(fields["accumulator_comparisons"]), 421)
        self.assertEqual(int(fields["integer_stage_comparisons"]), 421)
        self.assertEqual(int(fields["fast_path_comparisons"]), 421)
        self.assertEqual(int(fields["undo_checks"]), 420)
        session_fields = dict(re.findall(r"([a-z_]+) ([0-9]+)", session_report))
        self.assertEqual(int(session_fields["positions"]), 421)
        self.assertEqual(int(session_fields["transitions"]), 420)
        self.assertEqual(int(session_fields["accumulator_checks"]), 421)
        self.assertEqual(int(session_fields["integer_stage_checks"]), 421)
        self.assertEqual(int(session_fields["value_checks"]), 421)
        self.assertEqual(int(session_fields["undo_checks"]), 420)

        focused: list[dict[str, str]] = []
        with UciSession() as session:
            session.send(load_command(NETWORK_PATH, self.network_sha))
            session.wait_for(r"AliceNative-v2 M512 parameters loaded generation=1")
            for fen in FOCUSED_INCREMENTAL_FENS:
                session.send(f"position fen {fen}")
                session.send("alice_native_v2_verify_incremental 1")
                report = session.wait_for(r"^alice_native_v2 incremental verified generation 1")
                session.send("alice_native_v2_verify_search_session 1")
                session_report = session.wait_for(r"^alice_native_v2 session verified generation 1")
                incremental_fields = dict(re.findall(r"([a-z_]+) ([0-9]+)", report))
                session_fields = dict(re.findall(r"([a-z_]+) ([0-9]+)", session_report))
                self.assertEqual(session_fields["positions"], incremental_fields["positions"])
                self.assertEqual(session_fields["transitions"], incremental_fields["transitions"])
                self.assertEqual(session_fields["captures"], incremental_fields["captures"])
                self.assertEqual(session_fields["promotions"], incremental_fields["promotions"])
                self.assertEqual(session_fields["castlings"], incremental_fields["castlings"])
                self.assertEqual(session_fields["king_moves"], incremental_fields["king_moves"])
                self.assertEqual(session_fields["undo_checks"], session_fields["transitions"])
                focused.append(incremental_fields)
        self.assertGreater(int(focused[0]["promotions"]), 0)
        self.assertGreater(int(focused[2]["captures"]), 0)
        self.assertGreater(int(focused[2]["castlings"]), 0)
        self.assertTrue(any(int(report["king_moves"]) > 0 for report in focused))
        for report in focused:
            self.assertEqual(int(report["undo_checks"]), int(report["transitions"]))

    def test_corruption_is_rejected_transactionally(self) -> None:
        assert NETWORK_PATH is not None
        payload = HEADER_BYTES + MANIFEST_BYTES
        first_stack = payload + FEATURE_TENSOR_BYTES
        cases: list[tuple[str, int | None, int | None, str, bool]] = [
            ("header-crc", 8, 0, "header CRC32 mismatch", False),
            ("manifest-crc", HEADER_BYTES, 0, "manifest CRC32 mismatch", False),
            ("feature-hash", payload, 0, "feature component hash mismatch", False),
            (
                "fc1-padding",
                first_stack + FC1_WEIGHT_OFFSET_IN_STACK + 16,
                1,
                "FC1 SIMD padding is nonzero",
                False,
            ),
            ("payload-auth", payload + 4, 0, "whole-file SHA-256 mismatch", True),
            ("truncated", None, None, "parameter file is truncated", False),
            ("trailing", None, None, "parameter file has trailing data", False),
        ]

        with tempfile.TemporaryDirectory(prefix="alice-native-v2-corruption-") as temporary:
            root = Path(temporary)
            corrupted: list[tuple[Path, str, str]] = []
            for name, offset, value, expected, use_original_sha in cases:
                target = root / f"{name}.nnue"
                if offset is not None and value is not None:
                    mutate_copy(NETWORK_PATH, target, offset, value)
                elif name == "truncated":
                    shutil.copyfile(NETWORK_PATH, target)
                    with target.open("r+b") as stream:
                        stream.truncate(target.stat().st_size - 1)
                else:
                    shutil.copyfile(NETWORK_PATH, target)
                    with target.open("ab") as stream:
                        stream.write(b"\0")
                expected_sha = self.network_sha if use_original_sha else file_sha256(target)
                corrupted.append((target, expected_sha, expected))

            saturation = root / "fc0-saturation-envelope.nnue"
            shutil.copyfile(NETWORK_PATH, saturation)
            with saturation.open("r+b") as stream:
                for input_index in (0, 1, 4, 5):
                    stream.seek(first_stack + 4 + 16 * 4 + input_index)
                    stream.write(bytes([127]))
            corrupted.append(
                (
                    saturation,
                    file_sha256(saturation),
                    "fc0 AVX2 saturation envelope is unsafe",
                )
            )

            with UciSession() as session:
                session.send(load_command(NETWORK_PATH, self.network_sha))
                session.wait_for(r"AliceNative-v2 M512 parameters loaded generation=1")
                for target, expected_sha, expected in corrupted:
                    session.send(load_command(target, expected_sha, tolerant=True))
                    session.wait_for(re.escape(expected))
                    session.send("alice_native_v2_load_status")
                    status = session.wait_for(r"AliceNative-v2 M512 parameters loaded generation=1")
                    self.assertIn(f"sha256={self.network_sha}", status)


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine", type=Path, default=default_engine_path())
    parser.add_argument("--network", type=Path)
    parser.add_argument("--reference-input", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--verifier", type=Path)
    return parser.parse_known_args()


if __name__ == "__main__":
    arguments, unittest_arguments = parse_arguments()
    ENGINE_PATH = arguments.engine.resolve()
    NETWORK_PATH = arguments.network.resolve() if arguments.network else None
    REFERENCE_INPUT = arguments.reference_input.resolve() if arguments.reference_input else None
    REFERENCE_OUTPUT = arguments.reference_output.resolve() if arguments.reference_output else None
    VERIFIER_PATH = arguments.verifier.resolve() if arguments.verifier else None
    unittest.main(argv=[sys.argv[0], *unittest_arguments], verbosity=2)
