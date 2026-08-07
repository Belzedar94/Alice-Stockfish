"""Exact parity and negative controls for Alice-native v1 sparse features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import unittest

from native_features_reference import (
    BASE_THREAT_DIMENSIONS,
    PIECE_SQUARE_DIMENSIONS,
    THREAT_DIMENSIONS,
    piece_feature_index,
    position_trace,
)
from reference import Position


TEST_DIRECTORY = Path(__file__).resolve().parent
FIXTURE_PATH = TEST_DIRECTORY / "fixtures" / "native-features-v1.json"
MANIFEST_PATH = TEST_DIRECTORY.parent.parent / "docs" / "alice" / "native-nnue-v1-manifest.json"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def default_engine_path() -> Path:
    repository = TEST_DIRECTORY.parent.parent
    windows = repository / "src" / "stockfish.exe"
    return windows if windows.exists() else repository / "src" / "stockfish"


ENGINE_PATH = default_engine_path()


def engine_traces(fens: list[str]) -> list[dict]:
    commands: list[str] = []
    for fen in fens:
        commands.extend((f"position fen {fen}", "alice_native_trace"))
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

    prefix = "alice_native_trace "
    traces = [json.loads(line[len(prefix) :]) for line in result.stdout.splitlines() if line.startswith(prefix)]
    if len(traces) != len(fens):
        raise AssertionError(
            f"Expected {len(fens)} native traces, received {len(traces)}.\n{result.stdout[-4000:]}"
        )
    return traces


def incremental_reports(cases: list[tuple[str, int]]) -> list[dict[str, int]]:
    commands: list[str] = []
    for fen, depth in cases:
        commands.extend((f"position fen {fen}", f"alice_native_verify_incremental {depth}"))
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
        r"^alice_native incremental verified positions (?P<positions>\d+) "
        r"transitions (?P<transitions>\d+) captures (?P<captures>\d+) "
        r"promotions (?P<promotions>\d+) castlings (?P<castlings>\d+) "
        r"king_moves (?P<king_moves>\d+) refreshes (?P<white_refreshes>\d+),"
        r"(?P<black_refreshes>\d+) max_piece_events (?P<max_piece_events>\d+) "
        r"max_threat_events (?P<max_threat_events>\d+) cache_checks (?P<cache_checks>\d+) "
        r"cache_adds (?P<cache_adds>\d+) cache_removes (?P<cache_removes>\d+) "
        r"cache_board_b_events (?P<cache_board_b_events>\d+) "
        r"simd_checks (?P<simd_checks>\d+) "
        r"fixed_snapshot_checks (?P<fixed_snapshot_checks>\d+) depth (?P<depth>\d+)$"
    )
    reports = [
        {name: int(value) for name, value in match.groupdict().items()}
        for line in result.stdout.splitlines()
        if (match := pattern.match(line))
    ]
    if len(reports) != len(cases):
        raise AssertionError(
            f"Expected {len(cases)} incremental reports, received {len(reports)}.\n"
            + result.stdout[-4000:]
        )
    return reports


def find_piece(perspective: dict, piece: str, square: str) -> dict:
    matches = [
        feature
        for feature in perspective["pieceFeatures"]
        if feature["piece"] == piece and feature["square"] == square
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {piece}@{square} feature, found {matches}")
    return matches[0]


def find_rook_knight_edge(perspective: dict) -> dict | None:
    matches = [
        feature
        for feature in perspective["threatFeatures"]
        if feature["attacker"] == "wR"
        and feature["from"] == "a1"
        and feature["attacked"] == "bN"
        and feature["to"] == "a8"
    ]
    if len(matches) > 1:
        raise AssertionError(f"Duplicate rook-knight edge: {matches}")
    return matches[0] if matches else None


def numeric_trace(perspective: dict) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    pieces = [(feature["index"], feature["relation"]) for feature in perspective["pieceFeatures"]]
    threats = [(feature["index"], feature["relation"]) for feature in perspective["threatFeatures"]]
    return pieces, threats


class NativeFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not ENGINE_PATH.is_file():
            raise FileNotFoundError(f"Alice-Stockfish executable not found: {ENGINE_PATH}")
        cls.fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.positions: dict[str, str] = cls.fixtures["positions"]

    def test_manifest_is_canonical_and_has_the_frozen_identity(self) -> None:
        raw = MANIFEST_PATH.read_bytes()
        manifest = json.loads(raw)
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
        self.assertEqual(raw, canonical)

        self.assertEqual(manifest["architecture"], "AliceNative-v1")
        self.assertEqual(manifest["wireVersion"], 0xA11CE001)
        self.assertEqual(manifest["pairFeature"], "None")
        self.assertEqual(manifest["relationOrder"], ["SAME", "OTHER"])
        self.assertEqual(manifest["features"]["pieceSquare"]["dimensions"], PIECE_SQUARE_DIMENSIONS)
        self.assertEqual(manifest["features"]["threat"]["dimensions"], THREAT_DIMENSIONS)

        psq_hash = hashlib.sha256(b"AliceHalfKAv2_hm_Rel-v1").digest()
        threat_hash = hashlib.sha256(b"AliceFullThreats_Rel-v1").digest()
        self.assertEqual(psq_hash[:4].hex().upper(), "5280C41E")
        self.assertEqual(threat_hash[:4].hex().upper(), "6EE7B82C")

        psq = int.from_bytes(psq_hash[:4], "big")
        threat = int.from_bytes(threat_hash[:4], "big")
        transformer = (((threat << 1) | (threat >> 31)) & 0xFFFFFFFF) ^ psq ^ (1024 * 2)
        self.assertEqual(transformer, 0x8F4FBC46)
        self.assertEqual(transformer ^ 0x63337116, 0xEC7CCD50)

    def test_hand_computed_piece_indices(self) -> None:
        self.assertEqual(piece_feature_index(0, 1, 12, "A", 4, "A"), 43_660)
        self.assertEqual(piece_feature_index(0, 1, 28, "B", 4, "A"), 44_380)

        same, other = engine_traces(
            [self.positions["goldenSame"], self.positions["goldenOther"]]
        )
        same_pawn = find_piece(same["perspectives"][0], "wP", "e2")
        other_pawn = find_piece(other["perspectives"][0], "wP", "e4")
        self.assertEqual(same_pawn["index"], self.fixtures["goldenIndices"]["whitePawnAe2WithKingAe1"])
        self.assertEqual(other_pawn["index"], self.fixtures["goldenIndices"]["whitePawnBe4WithKingAe1"])
        self.assertEqual((same_pawn["relation"], other_pawn["relation"]), ("SAME", "OTHER"))

    def test_engine_and_independent_reference_match_full_traces(self) -> None:
        corpus = list(self.positions.values())
        rng = random.Random(0xA11CE)
        position = Position.from_fen(START_FEN)
        for _ in range(10):
            corpus.append(position.fen())
            moves = sorted(position.legal_moves(), key=lambda move: move.uci())
            if not moves:
                break
            position = position.push_uci(rng.choice(moves).uci())

        corpus = list(dict.fromkeys(corpus))
        observed = engine_traces(corpus)

        for fen, trace in zip(corpus, observed, strict=True):
            with self.subTest(fen=fen):
                self.assertEqual(trace["architecture"], "AliceNative-v1")
                self.assertEqual(trace["pairFeature"], "None")
                self.assertEqual(trace["pieceSquareDimensions"], PIECE_SQUARE_DIMENSIONS)
                self.assertEqual(trace["threatDimensions"], THREAT_DIMENSIONS)
                self.assertEqual(trace["perspectives"], position_trace(Position.from_fen(fen)))

                for perspective in trace["perspectives"]:
                    piece_indices = [feature["index"] for feature in perspective["pieceFeatures"]]
                    threat_indices = [feature["index"] for feature in perspective["threatFeatures"]]
                    self.assertEqual(piece_indices, sorted(piece_indices))
                    self.assertEqual(threat_indices, sorted(threat_indices))
                    self.assertEqual(len(piece_indices), len(set(piece_indices)))
                    self.assertEqual(len(threat_indices), len(set(threat_indices)))
                    self.assertTrue(all(0 <= index < PIECE_SQUARE_DIMENSIONS for index in piece_indices))
                    self.assertTrue(all(0 <= index < THREAT_DIMENSIONS for index in threat_indices))

    def test_global_board_rename_is_invariant_but_local_mutation_is_visible(self) -> None:
        source, swapped, mutated = engine_traces(
            [
                self.positions["boardSwapSource"],
                self.positions["boardSwapTarget"],
                self.positions["localLayerMutation"],
            ]
        )

        for perspective in range(2):
            self.assertEqual(
                numeric_trace(source["perspectives"][perspective]),
                numeric_trace(swapped["perspectives"][perspective]),
            )

        source_pawn = find_piece(source["perspectives"][0], "wP", "e2")
        mutated_pawn = find_piece(mutated["perspectives"][0], "wP", "e2")
        self.assertNotEqual(source_pawn["index"], mutated_pawn["index"])
        self.assertEqual(mutated_pawn["index"] - source_pawn["index"], 704)

    def test_threats_are_board_local_and_relation_extended(self) -> None:
        same, other, cross_layer = engine_traces(
            [
                self.positions["threatSame"],
                self.positions["threatOther"],
                self.positions["threatCrossLayer"],
            ]
        )
        same_edge = find_rook_knight_edge(same["perspectives"][0])
        other_edge = find_rook_knight_edge(other["perspectives"][0])
        cross_edge = find_rook_knight_edge(cross_layer["perspectives"][0])

        self.assertIsNotNone(same_edge)
        self.assertIsNotNone(other_edge)
        assert same_edge is not None and other_edge is not None
        self.assertIsNone(cross_edge)
        self.assertEqual(same_edge["relation"], "SAME")
        self.assertEqual(other_edge["relation"], "OTHER")
        self.assertEqual(
            other_edge["index"] - same_edge["index"],
            self.fixtures["threatRelationStride"],
        )
        self.assertEqual(self.fixtures["threatRelationStride"], BASE_THREAT_DIMENSIONS)
        self.assertGreater(other_edge["index"], 65_535)

    def test_sparse_updates_and_scalar_accumulators_match_full_refresh(self) -> None:
        cases = [
            (START_FEN, 2),
            ("7k/5p2/8/8/2B5/8/8/7K w - - 0 1", 1),
            ("7k/P7/8/8/8/8/8/7K w - - 0 1", 1),
            ("r6k/1P6/8/8/8/8/8/7K w - - 0 1", 1),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 1),
            ("k7/8/8/8/8/8/8/4|K2|R w K - 0 1", 1),
            ("4r2|k/8/8/8/8/8/8/4K3 w - - 0 1", 1),
        ]
        reports = incremental_reports(cases)

        self.assertEqual(reports[0]["positions"], 421)
        self.assertEqual(reports[0]["transitions"], 420)
        self.assertEqual(reports[0]["max_piece_events"], 2)
        self.assertEqual(reports[0]["cache_checks"], 2 * reports[0]["positions"])
        self.assertEqual(reports[0]["simd_checks"], 2 * reports[0]["positions"])
        self.assertEqual(reports[0]["fixed_snapshot_checks"], 2 * reports[0]["positions"])
        self.assertGreater(reports[0]["cache_adds"], 0)
        self.assertGreater(reports[0]["cache_removes"], 0)
        self.assertGreater(reports[0]["cache_board_b_events"], 0)

        self.assertGreater(reports[1]["captures"], 0)
        self.assertGreaterEqual(reports[1]["max_piece_events"], 3)
        self.assertGreater(reports[2]["promotions"], 0)
        self.assertGreater(reports[3]["promotions"], 0)
        self.assertGreater(reports[3]["captures"], 0)
        self.assertGreaterEqual(reports[3]["max_piece_events"], 3)

        for report in reports[4:6]:
            self.assertGreater(report["castlings"], 0)
            self.assertEqual(report["max_piece_events"], 4)
            self.assertGreater(report["white_refreshes"], 0)

        self.assertGreater(reports[6]["king_moves"], 0)
        self.assertGreater(reports[6]["white_refreshes"], 0)
        self.assertTrue(all(report["max_threat_events"] <= 512 for report in reports))


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine", type=Path, default=default_engine_path())
    return parser.parse_known_args()


if __name__ == "__main__":
    arguments, unittest_arguments = parse_arguments()
    ENGINE_PATH = arguments.engine.resolve()
    unittest.main(argv=[sys.argv[0], *unittest_arguments], verbosity=2)
