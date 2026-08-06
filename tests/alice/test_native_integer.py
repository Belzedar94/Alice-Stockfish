"""Exact stage parity for qualification-only AliceNative-v1 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
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

    for lane, value in ((0, 200), (1, -128), (2, 255), (3, 64), (512, 220), (513, -128), (514, 255), (515, 32)):
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

    for row in sorted(threat_rows):
        first = row % 5 - 2
        second = row % 3 - 1
        install("threat.weight", row * 1_024, first or 2)
        install("threat.weight", row * 1_024 + 512, second or -1)
        for bucket in range(8):
            install("threat.psqt", row * 8 + bucket, (row + 2 * bucket) % 9 - 4)

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

        fc1_base = stack * 32
        install("stack.fc1.bias", fc1_base, -8_192)
        install("stack.fc1.bias", fc1_base + 1, 8_192)
        install("stack.fc1.bias", fc1_base + 2, stack - 4)
        fc1_weight = stack * (32 * 64)
        install("stack.fc1.weight", fc1_weight + 2 * 64, 1)
        install("stack.fc1.weight", fc1_weight + 2 * 64 + 32, 2)

        install("stack.fc2.bias", stack, 17 * stack - 50)
        fc2_weight = stack * 128
        install("stack.fc2.weight", fc2_weight, 1)
        install("stack.fc2.weight", fc2_weight + 32, 2)
        install("stack.fc2.weight", fc2_weight + 64, 3)
        install("stack.fc2.weight", fc2_weight + 96, 4)

    return parameters


def engine_integer_traces(network: Path, fens: list[str]) -> tuple[list[dict], str]:
    network_sha = file_sha256(network)
    commands = [f"alice_native_load_file {command_path(network)} {network_sha}"]
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
) -> list[dict[str, int]]:
    network_sha = file_sha256(network)
    commands = [f"alice_native_load_file {command_path(network)} {network_sha}"]
    for fen, depth in cases:
        commands.extend(
            (f"position fen {fen}", f"alice_native_verify_loaded_incremental {depth}")
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
        r"undo_checks (?P<undo_checks>\d+) depth (?P<depth>\d+) search disabled$"
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
    return reports


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

            zero = directory / "zero.nnue"
            write_zero_wire(zero)
            zero_sha = file_sha256(zero)
            search = subprocess.run(
                [str(ENGINE_PATH)],
                input="\n".join(
                    (
                        f"alice_native_load_file {command_path(zero)} {zero_sha}",
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
            self.assertNotEqual(search.returncode, 0, search.stdout + search.stderr)
            self.assertIn("Legacy Alice evaluation is enabled", search.stdout)
            self.assertNotIn("bestmove", search.stdout)

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
            reports = loaded_incremental_reports(network, cases)

        self.assertTrue(all(report["generation"] == 1 for report in reports))
        for report in reports:
            self.assertEqual(
                report["accumulator_comparisons"], 2 * report["positions"]
            )
            self.assertEqual(report["integer_stage_comparisons"], report["positions"])
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


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine", type=Path, default=default_engine_path())
    return parser.parse_known_args()


if __name__ == "__main__":
    arguments, unittest_arguments = parse_arguments()
    ENGINE_PATH = arguments.engine.resolve()
    unittest.main(argv=[sys.argv[0], *unittest_arguments], verbosity=2)
