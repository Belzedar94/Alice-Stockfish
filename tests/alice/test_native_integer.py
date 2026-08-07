"""Exact stage, fixed-session, and live-search parity for AliceNative-v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from native_features_reference import position_trace
from native_integer_reference import (
    BLACK,
    SparseNativeParameters,
    WHITE,
    evaluate_integer,
    install_parameter,
    trunc0,
)
from native_wire import file_sha256, write_zero_wire
from reference import Position


TEST_DIRECTORY = Path(__file__).resolve().parent
FIXTURE_PATH = TEST_DIRECTORY / "fixtures" / "native-features-v1.json"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
STAGE_KEYS = (
    "featureAccumulator",
    "psqtAccumulator",
    "transformedByPerspective",
    "transformedInput",
    "phase",
    "fc0Raw",
    "fc0Squared",
    "fc0Linear",
    "fc1Raw",
    "fc1Squared",
    "fc1Linear",
    "fc2Raw",
    "skip",
    "fwdOut",
    "positionalRaw16",
    "psqtRaw16",
    "positionalValue",
    "psqtValue",
    "nativeNnueValue",
)


def default_engine_path() -> Path:
    repository = TEST_DIRECTORY.parent.parent
    windows = repository / "src" / "stockfish.exe"
    return windows if windows.exists() else repository / "src" / "stockfish"


ENGINE_PATH = default_engine_path()


class UciSession:
    def __init__(self, engine: Path) -> None:
        self.process = subprocess.Popen(
            [str(engine)],
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
                remaining = max(0.0, deadline - time.monotonic())
                line = self.pending.get(timeout=min(0.1, remaining))
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


def command_path(path: Path) -> str:
    return json.dumps(path.resolve().as_posix())


def phase_position(piece_count: int, side_to_move: str) -> str:
    source = Position.from_fen(START_FEN)
    kings = [square for square, piece in enumerate(source.board) if piece and piece.symbol in "Kk"]
    candidates = [
        square
        for square, piece in enumerate(source.board)
        if piece and piece.symbol not in "Kk"
    ]
    candidates.sort(key=lambda square: (source.board[square].symbol.lower() != "p", square))
    retained = set(kings + candidates[: piece_count - 2])
    board = [piece if square in retained else None for square, piece in enumerate(source.board)]
    return Position(board, side_to_move, (), 0, 1).fen()


def numeric_features(fen: str) -> tuple[list[list[int]], list[list[int]], int, int]:
    position = Position.from_fen(fen)
    trace = position_trace(position)
    pieces = [
        [feature["index"] for feature in perspective["pieceFeatures"]]
        for perspective in trace
    ]
    threats = [
        [feature["index"] for feature in perspective["threatFeatures"]]
        for perspective in trace
    ]
    side = WHITE if position.side_to_move == "w" else BLACK
    piece_count = sum(piece is not None for piece in position.board)
    return pieces, threats, side, piece_count


def build_sparse_network(path: Path, feature_fens: list[str]) -> SparseNativeParameters:
    write_zero_wire(path)
    parameters = SparseNativeParameters()

    def install(name: str, index: int, value: int) -> None:
        install_parameter(path, parameters, name, index, value)

    for lane, value in (
        (0, 200),
        (1, -128),
        (2, 255),
        (3, 64),
        (4, 255),
        (5, 255),
        (512, 220),
        (513, -128),
        (514, 255),
        (515, 32),
        (516, 255),
        (517, 255),
    ):
        install("ft.bias", lane, value)

    piece_rows: set[int] = set()
    threat_rows: set[int] = set()
    for fen in feature_fens:
        pieces, threats, _, _ = numeric_features(fen)
        piece_rows.update(pieces[WHITE])
        piece_rows.update(pieces[BLACK])
        threat_rows.update(threats[WHITE])
        threat_rows.update(threats[BLACK])

    for row in sorted(piece_rows):
        first = row % 7 - 3
        second = row % 5 - 2
        install("pieceSquare.weight", row * 1_024, first or 1)
        install("pieceSquare.weight", row * 1_024 + 512, second or -1)
        install("pieceSquare.weight", row * 1_024 + 3, row % 3 - 1)
        for bucket in range(8):
            install("pieceSquare.psqt", row * 8 + bucket, (row + bucket) % 11 - 5)
    piece_boundary_row = min(piece_rows)
    install("pieceSquare.weight", piece_boundary_row * 1_024 + 10, 32_767)
    install("pieceSquare.weight", piece_boundary_row * 1_024 + 11, -32_767)

    for row in sorted(threat_rows):
        first = row % 5 - 2
        second = row % 3 - 1
        install("threat.weight", row * 1_024, first or 2)
        install("threat.weight", row * 1_024 + 512, second or -1)
        for bucket in range(8):
            install("threat.psqt", row * 8 + bucket, (row + 2 * bucket) % 9 - 4)
    threat_boundary_row = min(threat_rows)
    install("threat.weight", threat_boundary_row * 1_024 + 12, 127)
    install("threat.weight", threat_boundary_row * 1_024 + 13, -127)

    psqt_witness: tuple[int, int] | None = None
    for fen in feature_fens:
        witness_pieces, _, _, witness_count = numeric_features(fen)
        white_only = sorted(set(witness_pieces[WHITE]) - set(witness_pieces[BLACK]))
        if white_only:
            psqt_witness = white_only[0], (witness_count - 1) // 4
            break
    if psqt_witness is None:
        raise AssertionError("The feature corpus needs a perspective-specific piece feature.")
    witness_row, witness_bucket = psqt_witness
    install("pieceSquare.psqt", witness_row * 8 + witness_bucket, 30)

    for stack in range(8):
        fc0_base = stack * 32
        install("stack.fc0.bias", fc0_base, -16_384)
        install("stack.fc0.bias", fc0_base + 1, 16_384)
        install("stack.fc0.bias", fc0_base + 2, 10 * stack - 35)
        install("stack.fc0.bias", fc0_base + 30, 100 + stack)
        install("stack.fc0.bias", fc0_base + 31, -50 - stack)

        fc0_weight = stack * (32 * 1_024)
        install("stack.fc0.weight", fc0_weight + 2 * 1_024, 2)
        install("stack.fc0.weight", fc0_weight + 2 * 1_024 + 512, -1)
        install("stack.fc0.weight", fc0_weight + 3 * 1_024 + 2, 1)
        install("stack.fc0.weight", fc0_weight + 4 * 1_024 + 4, 127)
        install("stack.fc0.weight", fc0_weight + 4 * 1_024 + 5, 127)
        install("stack.fc0.weight", fc0_weight + 5 * 1_024 + 4, -127)
        install("stack.fc0.weight", fc0_weight + 5 * 1_024 + 5, -127)

        fc1_base = stack * 32
        install("stack.fc1.bias", fc1_base, -8_192)
        install("stack.fc1.bias", fc1_base + 1, 8_192)
        install("stack.fc1.bias", fc1_base + 2, stack - 4)
        fc1_weight = stack * (32 * 64)
        install("stack.fc1.weight", fc1_weight + 2 * 64, 1)
        install("stack.fc1.weight", fc1_weight + 2 * 64 + 32, 2)
        install("stack.fc1.weight", fc1_weight + 4 * 64 + 4, 127)
        install("stack.fc1.weight", fc1_weight + 4 * 64 + 5, 127)
        install("stack.fc1.weight", fc1_weight + 5 * 64 + 4, -127)
        install("stack.fc1.weight", fc1_weight + 5 * 64 + 5, -127)

        install("stack.fc2.bias", stack, 17 * stack - 50)
        fc2_weight = stack * 128
        install("stack.fc2.weight", fc2_weight, 1)
        install("stack.fc2.weight", fc2_weight + 32, 2)
        install("stack.fc2.weight", fc2_weight + 64, 3)
        install("stack.fc2.weight", fc2_weight + 96, 4)

    return parameters


def engine_integer_traces(network: Path, fens: list[str]) -> tuple[list[dict], str]:
    network_sha = file_sha256(network)
    commands = [
        f"alice_native_load_file {command_path(network)} {network_sha}",
        "alice_native_verify_lease",
    ]
    for fen in fens:
        commands.extend((f"position fen {fen}", "alice_native_eval_trace"))
    commands.extend(("quit", ""))
    result = subprocess.run(
        [str(ENGINE_PATH)],
        input="\n".join(commands),
        text=True,
        capture_output=True,
        encoding="ascii",
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)

    lease_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("alice_native lease verified ")
    ]
    expected_lease = (
        f"alice_native lease verified generation 1 sha256 {network_sha} "
        "active_reload_rejections 1 reacquisitions 1"
    )
    if lease_lines != [expected_lease]:
        raise AssertionError(
            f"Unexpected native lease report: {lease_lines}.\n" + result.stdout[-4000:]
        )
    prefix = "alice_native_integer_trace "
    traces = [
        json.loads(line[len(prefix) :])
        for line in result.stdout.splitlines()
        if line.startswith(prefix)
    ]
    if len(traces) != len(fens):
        raise AssertionError(
            f"Expected {len(fens)} integer traces, got {len(traces)}.\n{result.stdout[-4000:]}"
        )
    return traces, network_sha


def loaded_incremental_reports(
    network: Path, cases: list[tuple[str, int]]
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    network_sha = file_sha256(network)
    commands = [f"alice_native_load_file {command_path(network)} {network_sha}"]
    for fen, depth in cases:
        commands.extend(
            (
                f"position fen {fen}",
                f"alice_native_verify_loaded_incremental {depth}",
                f"alice_native_verify_search_session {depth}",
            )
        )
    commands.extend(("quit", ""))
    result = subprocess.run(
        [str(ENGINE_PATH)],
        input="\n".join(commands),
        text=True,
        capture_output=True,
        encoding="ascii",
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)

    pattern = re.compile(
        r"^alice_native loaded incremental verified generation (?P<generation>\d+) "
        r"positions (?P<positions>\d+) transitions (?P<transitions>\d+) "
        r"captures (?P<captures>\d+) promotions (?P<promotions>\d+) "
        r"castlings (?P<castlings>\d+) king_moves (?P<king_moves>\d+) "
        r"refreshes (?P<white_refreshes>\d+),(?P<black_refreshes>\d+) "
        r"piece_adds (?P<piece_adds>\d+) piece_removes (?P<piece_removes>\d+) "
        r"threat_adds (?P<threat_adds>\d+) threat_removes (?P<threat_removes>\d+) "
        r"max_piece_events (?P<max_piece_events>\d+) "
        r"max_threat_events (?P<max_threat_events>\d+) "
        r"accumulator_comparisons (?P<accumulator_comparisons>\d+) "
        r"integer_stage_comparisons (?P<integer_stage_comparisons>\d+) "
        r"feature_simd_comparisons (?P<feature_simd_comparisons>\d+) "
        r"dense_simd_comparisons (?P<dense_simd_comparisons>\d+) "
        r"fixed_accumulator_checks (?P<fixed_accumulator_checks>\d+) "
        r"fixed_delta_updates (?P<fixed_delta_updates>\d+) "
        r"undo_checks (?P<undo_checks>\d+) depth (?P<depth>\d+) search available$"
    )
    reports = [
        {name: int(value) for name, value in match.groupdict().items()}
        for line in result.stdout.splitlines()
        if (match := pattern.match(line))
    ]
    if len(reports) != len(cases):
        raise AssertionError(
            f"Expected {len(cases)} loaded incremental reports, got {len(reports)}.\n"
            + result.stdout[-4000:]
        )

    session_pattern = re.compile(
        r"^alice_native session verified generation (?P<generation>\d+) "
        r"positions (?P<positions>\d+) transitions (?P<transitions>\d+) "
        r"captures (?P<captures>\d+) promotions (?P<promotions>\d+) "
        r"castlings (?P<castlings>\d+) king_moves (?P<king_moves>\d+) "
        r"evaluations (?P<evaluations>\d+) pushes (?P<pushes>\d+) pops (?P<pops>\d+) "
        r"refreshes (?P<white_refreshes>\d+),(?P<black_refreshes>\d+) "
        r"piece_adds (?P<piece_adds>\d+) piece_removes (?P<piece_removes>\d+) "
        r"threat_adds (?P<threat_adds>\d+) threat_removes (?P<threat_removes>\d+) "
        r"max_piece_events (?P<max_piece_events>\d+) "
        r"max_threat_events (?P<max_threat_events>\d+) "
        r"accumulator_checks (?P<accumulator_checks>\d+) "
        r"value_checks (?P<value_checks>\d+) undo_checks (?P<undo_checks>\d+) "
        r"depth (?P<depth>\d+) search available$"
    )
    session_reports = [
        {name: int(value) for name, value in match.groupdict().items()}
        for line in result.stdout.splitlines()
        if (match := session_pattern.match(line))
    ]
    if len(session_reports) != len(cases):
        raise AssertionError(
            f"Expected {len(cases)} search-session reports, got {len(session_reports)}.\n"
            + result.stdout[-4000:]
        )
    return reports, session_reports


class NativeIntegerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not ENGINE_PATH.is_file():
            raise FileNotFoundError(f"Alice-Stockfish executable not found: {ENGINE_PATH}")
        cls.fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["positions"]

    def test_every_integer_stage_matches_the_independent_reference(self) -> None:
        boundary_counts = (2, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32)
        phase_fens = [
            phase_position(count, "w" if index % 2 == 0 else "b")
            for index, count in enumerate(boundary_counts)
        ]
        special = [
            self.fixtures["goldenSame"],
            self.fixtures["goldenOther"],
            self.fixtures["threatSame"],
            self.fixtures["threatOther"],
            self.fixtures["boardSwapSource"],
            self.fixtures["boardSwapTarget"],
            phase_position(2, "b"),
        ]
        fens = list(dict.fromkeys([*phase_fens, *special]))

        with tempfile.TemporaryDirectory(prefix="alice-native-integer-") as temporary:
            network = Path(temporary) / "sparse.nnue"
            parameters = build_sparse_network(network, special)
            observed, network_sha = engine_integer_traces(network, fens)

            phases: set[int] = set()
            separate_division_witness = False
            for fen, trace in zip(fens, observed, strict=True):
                pieces, threats, side, piece_count = numeric_features(fen)
                expected = evaluate_integer(parameters, pieces, threats, side, piece_count)
                with self.subTest(fen=fen):
                    self.assertEqual(trace["architecture"], "AliceNative-v1")
                    self.assertIn(trace["denseSimd"], ("avx2", "ssse3"))
                    self.assertEqual(trace["featureSimd"], trace["denseSimd"])
                    self.assertEqual(trace["generation"], 1)
                    self.assertEqual(trace["networkSha256"], network_sha)
                    self.assertEqual(trace["sideToMove"], side)
                    self.assertEqual(trace["pieceCount"], piece_count)
                    self.assertEqual(trace["pieceFeatures"], pieces)
                    self.assertEqual(trace["threatFeatures"], threats)
                    for key in STAGE_KEYS:
                        self.assertEqual(trace[key], expected[key], key)
                phases.add(trace["phase"])
                combined = trunc0(trace["positionalRaw16"] + trace["psqtRaw16"], 16)
                separate_division_witness |= combined != trace["nativeNnueValue"]

            self.assertEqual(phases, set(range(8)))
            self.assertTrue(separate_division_witness)
            self.assertTrue(any(trace["fc0Raw"][0] < 0 for trace in observed))
            self.assertTrue(all(trace["fc0Squared"][0] == 127 for trace in observed))
            self.assertTrue(all(trace["fc0Linear"][0] == 0 for trace in observed))
            self.assertTrue(all(trace["fc0Raw"][4] == 32_258 for trace in observed))
            self.assertTrue(all(trace["fc0Raw"][5] == -32_258 for trace in observed))
            self.assertTrue(all(trace["fc1Raw"][4] == 32_258 for trace in observed))
            self.assertTrue(all(trace["fc1Raw"][5] == -32_258 for trace in observed))
            accumulator_values = [
                value
                for trace in observed
                for perspective in trace["featureAccumulator"]
                for value in perspective
            ]
            self.assertIn(32_767, accumulator_values)
            self.assertIn(-32_767, accumulator_values)
            self.assertIn(127, accumulator_values)
            self.assertIn(-127, accumulator_values)

    def test_accumulator_overflow_and_search_routing_fail_closed(self) -> None:
        fen = phase_position(2, "w")
        pieces, _, _, _ = numeric_features(fen)
        active_feature = pieces[WHITE][0]
        with tempfile.TemporaryDirectory(prefix="alice-native-integer-negative-") as temporary:
            directory = Path(temporary)
            overflow = directory / "overflow.nnue"
            write_zero_wire(overflow)
            parameters = SparseNativeParameters()
            install_parameter(overflow, parameters, "ft.bias", 0, 32_767)
            install_parameter(
                overflow,
                parameters,
                "pieceSquare.weight",
                active_feature * 1_024,
                1,
            )
            overflow_sha = file_sha256(overflow)
            result = subprocess.run(
                [str(ENGINE_PATH)],
                input="\n".join(
                    (
                        f"alice_native_load_file {command_path(overflow)} {overflow_sha}",
                        f"position fen {fen}",
                        "alice_native_eval_trace",
                        "quit",
                        "",
                    )
                ),
                text=True,
                capture_output=True,
                encoding="ascii",
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("feature accumulator exceeds signed i16", result.stdout)

            incremental_overflow = subprocess.run(
                [str(ENGINE_PATH)],
                input="\n".join(
                    (
                        f"alice_native_load_file {command_path(overflow)} {overflow_sha}",
                        f"position fen {fen}",
                        "alice_native_verify_loaded_incremental 0",
                        "quit",
                        "",
                    )
                ),
                text=True,
                capture_output=True,
                encoding="ascii",
                check=False,
            )
            self.assertNotEqual(
                incremental_overflow.returncode,
                0,
                incremental_overflow.stdout + incremental_overflow.stderr,
            )
            self.assertIn(
                "feature accumulator exceeds signed i16", incremental_overflow.stdout
            )

            session_overflow = subprocess.run(
                [str(ENGINE_PATH)],
                input="\n".join(
                    (
                        f"alice_native_load_file {command_path(overflow)} {overflow_sha}",
                        f"position fen {fen}",
                        "alice_native_verify_search_session 0",
                        "quit",
                        "",
                    )
                ),
                text=True,
                capture_output=True,
                encoding="ascii",
                check=False,
            )
            self.assertNotEqual(
                session_overflow.returncode,
                0,
                session_overflow.stdout + session_overflow.stderr,
            )
            self.assertIn("code=accumulator-out-of-range", session_overflow.stdout)
            self.assertIn("stage=root-refresh", session_overflow.stdout)

            zero = directory / "zero.nnue"
            write_zero_wire(zero)
            zero_sha = file_sha256(zero)

            missing_native = subprocess.run(
                [str(ENGINE_PATH)],
                input="\n".join(
                    (
                        "setoption name Alice Evaluation value Native",
                        f"position fen {fen}",
                        "go depth 1",
                        "quit",
                        "",
                    )
                ),
                text=True,
                capture_output=True,
                encoding="ascii",
                check=False,
            )
            self.assertNotEqual(
                missing_native.returncode,
                0,
                missing_native.stdout + missing_native.stderr,
            )
            self.assertIn("parameters cannot be leased", missing_native.stdout)
            self.assertNotIn("bestmove", missing_native.stdout)

            with UciSession(ENGINE_PATH) as session:
                session.send(f"setoption name Alice Native SHA256 value {zero_sha}")
                session.wait_for(r"loading requires both Alice Native EvalFile and Alice Native SHA256")
                session.send(
                    f"setoption name Alice Native EvalFile value {zero.resolve().as_posix()}"
                )
                loaded = session.wait_for(
                    r"Alice native qualification parameters loaded generation=1"
                )
                self.assertIn(f"sha256={zero_sha}", loaded)
                self.assertIn("search=available", loaded)

                session.send("setoption name Alice Evaluation value Native")
                session.wait_for(r"Alice native qualification parameters loaded generation=1")
                session.send(f"position fen {fen}")
                session.send("eval")
                evaluated = session.wait_for(r"^alice_native value 0 generation 1 sha256 ")
                self.assertTrue(evaluated.endswith(zero_sha), evaluated)

                session.send("go depth 1")
                session.wait_for(r"^info depth 1 .* score cp 0 ")
                session.wait_for(r"^bestmove ")

                session.send("go infinite")
                session.wait_for(r"^info depth 1 ")
                session.send(
                    f"alice_native_try_load_file {command_path(zero)} {zero_sha}"
                )
                rejected = session.wait_for(r"search lease is active")
                self.assertIn("load rejected", rejected)
                session.send("stop")
                session.wait_for(r"^bestmove ")
                session.send("alice_native_load_status")
                preserved = session.wait_for(
                    r"Alice native qualification parameters loaded generation=1"
                )
                self.assertIn(f"sha256={zero_sha}", preserved)

                session.send("setoption name Use NNUE value false")
                session.wait_for(r"Use NNUE disabled")
                session.send("eval")
                session.wait_for(r"^legacy_nnue raw 0 adjusted 0$")
                session.send("setoption name Use NNUE value true")
                session.wait_for(r"Alice native qualification parameters loaded generation=1")
                session.send("eval")
                session.wait_for(r"^alice_native value 0 generation 1 sha256 ")

            stale_selection = subprocess.run(
                [str(ENGINE_PATH)],
                input="\n".join(
                    (
                        f"alice_native_load_file {command_path(zero)} {zero_sha}",
                        f"setoption name Alice Native SHA256 value {zero_sha}",
                        f"setoption name Alice Native EvalFile value {directory / 'missing.nnue'}",
                        "setoption name Alice Evaluation value Native",
                        f"position fen {fen}",
                        "eval",
                        "quit",
                        "",
                    )
                ),
                text=True,
                capture_output=True,
                encoding="ascii",
                check=False,
            )
            self.assertNotEqual(
                stale_selection.returncode,
                0,
                stale_selection.stdout + stale_selection.stderr,
            )
            self.assertIn("selected EvalFile is unavailable", stale_selection.stdout)
            self.assertNotIn("alice_native value", stale_selection.stdout)

            missing = subprocess.run(
                [str(ENGINE_PATH)],
                input="alice_native_verify_loaded_incremental 0\nquit\n",
                text=True,
                capture_output=True,
                encoding="ascii",
                check=False,
            )
            self.assertNotEqual(missing.returncode, 0, missing.stdout + missing.stderr)
            self.assertIn("requires qualification parameters", missing.stdout)

            missing_session = subprocess.run(
                [str(ENGINE_PATH)],
                input="alice_native_verify_search_session 0\nquit\n",
                text=True,
                capture_output=True,
                encoding="ascii",
                check=False,
            )
            self.assertNotEqual(
                missing_session.returncode,
                0,
                missing_session.stdout + missing_session.stderr,
            )
            self.assertIn("requires qualification parameters", missing_session.stdout)

    def test_loaded_incremental_matches_full_refresh_after_every_transition(self) -> None:
        cases = [
            (START_FEN, 2),
            ("7k/5p2/8/8/2B5/8/8/7K w - - 0 1", 1),
            ("7k/P7/8/8/8/8/8/7K w - - 0 1", 1),
            ("r6k/1P6/8/8/8/8/8/7K w - - 0 1", 1),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 1),
            ("k7/8/8/8/8/8/8/4|K2|R w K - 0 1", 1),
            ("4r2|k/8/8/8/8/8/8/4K3 w - - 0 1", 1),
        ]
        feature_fens = list(self.fixtures.values()) + [fen for fen, _ in cases]
        with tempfile.TemporaryDirectory(prefix="alice-native-loaded-incremental-") as temporary:
            network = Path(temporary) / "sparse.nnue"
            build_sparse_network(network, feature_fens)
            reports, session_reports = loaded_incremental_reports(network, cases)

        self.assertTrue(all(report["generation"] == 1 for report in reports))
        for report in reports:
            self.assertEqual(
                report["accumulator_comparisons"], 2 * report["positions"]
            )
            self.assertEqual(report["integer_stage_comparisons"], report["positions"])
            self.assertEqual(
                report["feature_simd_comparisons"], 2 * report["positions"]
            )
            self.assertEqual(report["dense_simd_comparisons"], report["positions"])
            self.assertEqual(
                report["fixed_accumulator_checks"], 2 * report["positions"]
            )
            self.assertGreaterEqual(report["fixed_delta_updates"], report["transitions"])
            self.assertEqual(report["undo_checks"], report["transitions"])

        opening = reports[0]
        self.assertEqual(opening["positions"], 421)
        self.assertEqual(opening["transitions"], 420)
        self.assertGreater(opening["piece_adds"], 0)
        self.assertGreater(opening["piece_removes"], 0)
        self.assertGreater(opening["threat_adds"], 0)
        self.assertGreater(opening["threat_removes"], 0)

        self.assertGreater(reports[1]["captures"], 0)
        self.assertGreater(reports[2]["promotions"], 0)
        self.assertGreater(reports[3]["promotions"], 0)
        self.assertGreater(reports[3]["captures"], 0)
        for report in reports[4:6]:
            self.assertGreater(report["castlings"], 0)
            self.assertGreater(report["white_refreshes"], 0)
        self.assertGreater(reports[6]["king_moves"], 0)
        self.assertGreater(reports[6]["white_refreshes"], 0)

        for incremental, session in zip(reports, session_reports, strict=True):
            for field in (
                "generation",
                "positions",
                "transitions",
                "captures",
                "promotions",
                "castlings",
                "king_moves",
                "white_refreshes",
                "black_refreshes",
                "piece_adds",
                "piece_removes",
                "threat_adds",
                "threat_removes",
                "max_piece_events",
                "max_threat_events",
                "depth",
            ):
                self.assertEqual(session[field], incremental[field], field)
            self.assertEqual(session["evaluations"], session["positions"])
            self.assertEqual(session["pushes"], session["transitions"])
            self.assertEqual(session["pops"], session["transitions"])
            self.assertEqual(session["undo_checks"], session["transitions"])
            self.assertEqual(
                session["accumulator_checks"], 2 * session["positions"]
            )
            self.assertEqual(session["value_checks"], session["positions"])


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine", type=Path, default=default_engine_path())
    return parser.parse_known_args()


if __name__ == "__main__":
    arguments, unittest_arguments = parse_arguments()
    ENGINE_PATH = arguments.engine.resolve()
    unittest.main(argv=[sys.argv[0], *unittest_arguments], verbosity=2)
