"""Executable checks for the normative Alice rules fixture corpus."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from reference import FenError, MoveResolutionError, Position


FIXTURE_PATH = Path(__file__).with_name("fixtures") / "rules-v1.json"


class RulesFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.cases = cls.document["cases"]

    def test_fixture_catalog_is_versioned_and_unique(self) -> None:
        schema = self.document["fixtureSchema"]
        self.assertEqual(schema["rulesVersion"], "alice-rules-v1")
        identifiers = [case["id"] for case in self.cases]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertGreaterEqual(len(identifiers), 20)

    def test_normative_cases(self) -> None:
        for case in self.cases:
            with self.subTest(case=case["id"]):
                handler = getattr(self, f"_check_{case['kind'].replace('-', '_')}")
                handler(case)

    def _check_move(self, case: dict[str, object]) -> None:
        position = Position.from_fen(case["initialFen"])
        move_text = case["move"]
        expected = case["expected"]

        if not expected["legal"]:
            with self.assertRaises(MoveResolutionError):
                position.resolve_uci(move_text)
            return

        move = position.resolve_uci(move_text)
        result = position.after(move)
        self.assertEqual(result.fen(), expected["resultFen"])
        self.assertEqual(result.in_check(result.side_to_move), expected["sideToMoveInCheck"])

        subsequent = expected.get("subsequentMoveLegality", {})
        for uci, legal in subsequent.items():
            self.assertEqual(result.is_uci_legal(uci), legal)

    def _check_position(self, case: dict[str, object]) -> None:
        expected = case["expected"]
        position = Position.from_fen(case["initialFen"])
        self.assertTrue(expected["accepted"])
        self.assertEqual(position.in_check(position.side_to_move), expected["sideToMoveInCheck"])
        if "checkers" in expected:
            king_square = position.king_square(position.side_to_move)
            king = position.board[king_square]
            assert king is not None
            attackers = [f"{king.layer}:{self._square_name(square)}" for square in position.attackers(
                king_square, king.layer, "b" if position.side_to_move == "w" else "w"
            )]
            self.assertEqual(attackers, expected["checkers"])

    @staticmethod
    def _square_name(square: int) -> str:
        return "abcdefgh"[square % 8] + str(square // 8 + 1)

    def _check_fen_normalization(self, case: dict[str, object]) -> None:
        position = Position.from_fen(case["inputFen"])
        self.assertTrue(case["expected"]["accepted"])
        self.assertEqual(position.fen(), case["expected"]["canonicalFen"])

    def _check_fen_validation(self, case: dict[str, object]) -> None:
        self.assertFalse(case["expected"]["accepted"])
        with self.assertRaises(FenError):
            Position.from_fen(case["inputFen"])

    def _check_identity(self, case: dict[str, object]) -> None:
        left = Position.from_fen(case["positions"]["left"])
        right = Position.from_fen(case["positions"]["right"])
        expected = case["expected"]
        comparisons = {
            "fullPositionKeyRelation": (left.identity(), right.identity()),
            "pawnKeyRelation": (left.pawn_identity(), right.pawn_identity()),
            "minorPieceKeyRelation": (left.minor_piece_identity(), right.minor_piece_identity()),
            "whiteNonPawnKeyRelation": (left.non_pawn_identity("w"), right.non_pawn_identity("w")),
            "blackNonPawnKeyRelation": (left.non_pawn_identity("b"), right.non_pawn_identity("b")),
            "countOnlyMaterialKeyRelation": (left.material_identity(), right.material_identity()),
        }
        for field, (left_value, right_value) in comparisons.items():
            relation = expected.get(field)
            if relation == "same":
                self.assertEqual(left_value, right_value, field)
            elif relation == "different":
                self.assertNotEqual(left_value, right_value, field)
        self.assertEqual(left.identity() == right.identity(), expected["repetitionEquivalent"])

    def _check_sequence(self, case: dict[str, object]) -> None:
        position = Position.from_fen(case["initialFen"])
        initial_identity = position.identity()
        initial_occurrences = 1
        checkpoints = {checkpoint["afterPly"]: checkpoint for checkpoint in case["expected"]["checkpoints"]}

        for ply, move_text in enumerate(case["moves"], start=1):
            position = position.push_uci(move_text)
            if position.identity() == initial_identity:
                initial_occurrences += 1

            checkpoint = checkpoints.get(ply)
            if not checkpoint:
                continue
            self.assertEqual(position.fen(), checkpoint["fen"])
            same = position.identity() == initial_identity
            self.assertEqual(same, checkpoint["repetitionEquivalentToInitial"])
            self.assertEqual(initial_occurrences, checkpoint["initialIdentityOccurrences"])
            self.assertEqual(initial_occurrences >= 3, checkpoint["threefold"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
