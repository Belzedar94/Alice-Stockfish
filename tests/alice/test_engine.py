"""Black-box conformance checks for an Alice-Stockfish executable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import unittest

from reference import Position


TEST_DIRECTORY = Path(__file__).resolve().parent
FIXTURE_PATH = TEST_DIRECTORY / "fixtures" / "rules-v1.json"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PERFT = {1: 20, 2: 400, 3: 9384, 4: 219236}


def default_engine_path() -> Path:
    repository = TEST_DIRECTORY.parent.parent
    windows = repository / "src" / "stockfish.exe"
    return windows if windows.exists() else repository / "src" / "stockfish"


ENGINE_PATH = default_engine_path()


def run_engine(*commands: str) -> subprocess.CompletedProcess[str]:
    payload = "\n".join((*commands, "quit", ""))
    return subprocess.run(
        [str(ENGINE_PATH)],
        input=payload,
        text=True,
        capture_output=True,
        encoding="ascii",
        check=False,
    )


def inspected_fen(output: str) -> str:
    matches = re.findall(r"^Fen: (.+)$", output, flags=re.MULTILINE)
    if not matches:
        raise AssertionError(f"No FEN found in executable output:\n{output[-2000:]}")
    return matches[-1].rstrip("\r")


def inspected_keys(output: str) -> dict[str, str]:
    labels = {
        "fullPositionKeyRelation": "Key",
        "pawnKeyRelation": "Pawn key",
        "minorPieceKeyRelation": "Minor key",
        "whiteNonPawnKeyRelation": "White non-pawn key",
        "blackNonPawnKeyRelation": "Black non-pawn key",
        "countOnlyMaterialKeyRelation": "Material key",
    }
    keys: dict[str, str] = {}
    for relation, label in labels.items():
        matches = re.findall(rf"^{re.escape(label)}: ([0-9A-F]+)$", output, flags=re.MULTILINE)
        if not matches:
            raise AssertionError(f"No {label} found in executable output:\n{output[-2000:]}")
        keys[relation] = matches[-1].rstrip("\r")
    return keys


def executable_legal_moves(fen: str) -> set[str]:
    result = run_engine(f"position fen {fen}", "go perft 1")
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    return set(
        re.findall(r"^([a-h][1-8][a-h][1-8][qrbn]?): 1\r?$", result.stdout, re.MULTILINE)
    )


class EngineFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not ENGINE_PATH.is_file():
            raise FileNotFoundError(f"Alice-Stockfish executable not found: {ENGINE_PATH}")
        cls.document = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.cases = cls.document["cases"]

    def test_all_fixture_positions_parse_and_round_trip(self) -> None:
        valid: dict[str, tuple[str, str]] = {}
        invalid: dict[str, str] = {}

        for case in self.cases:
            identifier = case["id"]
            if "initialFen" in case:
                valid[f"{identifier}:initial"] = (case["initialFen"], case["initialFen"])
            if "inputFen" in case:
                if case.get("expected", {}).get("accepted"):
                    canonical = case["expected"].get("canonicalFen", case["inputFen"])
                    valid[f"{identifier}:input"] = (case["inputFen"], canonical)
                else:
                    invalid[f"{identifier}:input"] = case["inputFen"]
            for name, fen in case.get("positions", {}).items():
                valid[f"{identifier}:{name}"] = (fen, fen)
            expected = case.get("expected", {})
            if "resultFen" in expected:
                valid[f"{identifier}:result"] = (expected["resultFen"], expected["resultFen"])
            for index, checkpoint in enumerate(expected.get("checkpoints", [])):
                valid[f"{identifier}:checkpoint-{index}"] = (checkpoint["fen"], checkpoint["fen"])

        for label, (fen, canonical) in valid.items():
            with self.subTest(position=label):
                result = run_engine(f"position fen {fen}", "d")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(inspected_fen(result.stdout), canonical)

        for label, fen in invalid.items():
            with self.subTest(position=label):
                result = run_engine(f"position fen {fen}")
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CRITICAL ERROR", result.stdout)

    def test_legacy_sixteen_wide_input_is_canonicalized(self) -> None:
        case = next(case for case in self.cases if case["kind"] == "fen-normalization")
        result = run_engine(f"position fen {case['inputFen']}", "d")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(inspected_fen(result.stdout), case["expected"]["canonicalFen"])

    def test_normative_moves(self) -> None:
        for case in self.cases:
            if case["kind"] != "move":
                continue

            expected = case["expected"]
            command = f"position fen {case['initialFen']} moves {case['move']}"
            with self.subTest(move=case["id"]):
                result = run_engine(command, "d")
                if not expected["legal"]:
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("Illegal move", result.stdout)
                    continue

                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(inspected_fen(result.stdout), expected["resultFen"])

                for next_move, legal in expected.get("subsequentMoveLegality", {}).items():
                    continuation = run_engine(
                        f"position fen {case['initialFen']} moves {case['move']} {next_move}"
                    )
                    self.assertEqual(continuation.returncode == 0, legal, next_move)

    def test_sequences_match_every_checkpoint(self) -> None:
        for case in self.cases:
            if case["kind"] != "sequence":
                continue

            moves = case["moves"]
            for checkpoint in case["expected"]["checkpoints"]:
                ply = checkpoint["afterPly"]
                command = f"position fen {case['initialFen']} moves {' '.join(moves[:ply])}"
                with self.subTest(sequence=case["id"], ply=ply):
                    result = run_engine(command, "d")
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(inspected_fen(result.stdout), checkpoint["fen"])

    def test_layer_changes_every_required_identity(self) -> None:
        for case in self.cases:
            if case["kind"] != "identity":
                continue

            keys: list[dict[str, str]] = []
            for fen in (case["positions"]["left"], case["positions"]["right"]):
                result = run_engine(f"position fen {fen}", "d")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                keys.append(inspected_keys(result.stdout))

            for relation, expected in case["expected"].items():
                if relation not in keys[0]:
                    continue
                with self.subTest(identity=case["id"], relation=relation):
                    self.assertEqual(keys[0][relation] == keys[1][relation], expected == "same")

    def test_start_position_perft(self) -> None:
        for depth, expected in PERFT.items():
            with self.subTest(depth=depth):
                result = run_engine(f"position fen {START_FEN}", f"go perft {depth}")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                match = re.search(r"Nodes searched: (\d+)", result.stdout)
                self.assertIsNotNone(match, result.stdout)
                self.assertEqual(int(match.group(1)), expected)

    def test_deterministic_playouts_match_the_reference(self) -> None:
        rng = random.Random(0xA11CE)
        seeds = [
            START_FEN,
            "3q3k/8/8/8/8/8/8/3Q3K w - - 0 1",
            "4r2|k/8/8/8/8/8/|R7/4K3 w - - 0 1",
        ]

        for seed_index, fen in enumerate(seeds):
            position = Position.from_fen(fen)
            for ply in range(6):
                expected_moves = {move.uci() for move in position.legal_moves()}
                with self.subTest(seed=seed_index, ply=ply, subject="legal-set"):
                    self.assertEqual(executable_legal_moves(position.fen()), expected_moves)

                if not expected_moves:
                    break
                move_text = rng.choice(sorted(expected_moves))
                position = position.push_uci(move_text)
                result = run_engine(f"position fen {fen} moves {move_text}", "d")
                with self.subTest(seed=seed_index, ply=ply, subject="transition"):
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(inspected_fen(result.stdout), position.fen())
                fen = position.fen()


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine", type=Path, default=default_engine_path())
    return parser.parse_known_args()


if __name__ == "__main__":
    arguments, unittest_arguments = parse_arguments()
    ENGINE_PATH = arguments.engine.resolve()
    unittest.main(argv=[sys.argv[0], *unittest_arguments], verbosity=2)
