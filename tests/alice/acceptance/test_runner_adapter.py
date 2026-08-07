from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.alice_acceptance.evidence import canonical_json_bytes, sha256_file
from tools.alice_acceptance.runner_adapter import parse_strict_json, validate_worker_response


def game(game_number: int) -> dict[str, object]:
    return {
        "game_number": game_number,
        "result": "1/2-1/2",
        "outcome_class": "SCORABLE_NATURAL",
        "reason": "Draw by rule",
        "termination": "normal",
        "failure_code": "",
        "failure_stage": "",
        "offending_move": "",
        "final_valid_position": {"root_fen": "fen", "moves": []},
    }


def materialize_pair(directory: Path, ordinal: int = 7) -> dict[str, object]:
    directory.mkdir()
    block = (
        '[Result "1/2-1/2"]\n'
        '[SetUp "1"]\n'
        '[FEN "fen"]\n'
        '[Variant "alice"]\n'
        '[PlyCount "0"]\n'
        "\n1/2-1/2\n\n"
    )
    (directory / "games.pgn").write_bytes((block + block).encode("ascii"))
    core = {
        "schema": "alice-pair-result-v1",
        "ordinal": ordinal,
        "game_classes": ["SCORABLE_NATURAL", "SCORABLE_NATURAL"],
        "game_scores": [0.5, 0.5],
        "games": [game(1), game(2)],
    }
    result = dict(core)
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    (directory / "result.jsonl").write_bytes(canonical_json_bytes(result))
    return {
        "schema": "alice-pair-worker-response-v1",
        "pair_ordinal": ordinal,
        "result": result,
        "artifacts": {
            "games_pgn_sha256": sha256_file(directory / "games.pgn"),
            "result_jsonl_sha256": sha256_file(directory / "result.jsonl"),
        },
    }


class RunnerAdapterTests(unittest.TestCase):
    def test_valid_pair_is_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pair_directory = Path(temporary) / "pair"
            response = materialize_pair(pair_directory)
            result = validate_worker_response(response, 7, pair_directory, "fen")
        self.assertTrue(result.scorable)
        self.assertEqual(result.game_scores, (0.5, 0.5))

    def test_artifact_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pair_directory = Path(temporary) / "pair"
            response = materialize_pair(pair_directory)
            with open(pair_directory / "games.pgn", "ab") as output:
                output.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "PGN SHA-256"):
                validate_worker_response(response, 7, pair_directory, "fen")

    def test_self_consistent_but_contradictory_score_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pair_directory = Path(temporary) / "pair"
            response = materialize_pair(pair_directory)
            result = response["result"]
            assert isinstance(result, dict)
            result["game_scores"] = [1, 0.5]
            core = dict(result)
            del core["evidence_sha256"]
            result["evidence_sha256"] = hashlib.sha256(
                canonical_json_bytes(core)
            ).hexdigest()
            (pair_directory / "result.jsonl").write_bytes(canonical_json_bytes(result))
            artifacts = response["artifacts"]
            assert isinstance(artifacts, dict)
            artifacts["result_jsonl_sha256"] = sha256_file(
                pair_directory / "result.jsonl"
            )
            with self.assertRaisesRegex(ValueError, "contender score"):
                validate_worker_response(response, 7, pair_directory, "fen")

    def test_unknown_fields_and_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pair_directory = Path(temporary) / "pair"
            response = materialize_pair(pair_directory)
            response["ignored"] = True
            with self.assertRaisesRegex(ValueError, "fields"):
                validate_worker_response(response, 7, pair_directory, "fen")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            parse_strict_json(b'{"key":1,"key":2}')
        with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
            parse_strict_json(b'{"key":NaN}')

    def test_self_consistent_pgn_hash_cannot_hide_a_wrong_fen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pair_directory = Path(temporary) / "pair"
            response = materialize_pair(pair_directory)
            pgn_path = pair_directory / "games.pgn"
            pgn_path.write_text(
                pgn_path.read_text(encoding="ascii").replace('[FEN "fen"]', '[FEN "other"]'),
                encoding="ascii",
                newline="\n",
            )
            artifacts = response["artifacts"]
            assert isinstance(artifacts, dict)
            artifacts["games_pgn_sha256"] = sha256_file(pgn_path)
            with self.assertRaisesRegex(ValueError, "PGN contradicts"):
                validate_worker_response(response, 7, pair_directory, "fen")


if __name__ == "__main__":
    unittest.main(verbosity=2)
