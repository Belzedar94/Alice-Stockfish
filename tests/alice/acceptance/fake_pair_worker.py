#!/usr/bin/env python3
"""Deterministic persistent pair process used by the controller tests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def canonical(value):
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


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def game(number):
    return {
        "game_number": number,
        "result": "1/2-1/2",
        "outcome_class": "SCORABLE_NATURAL",
        "reason": "Draw by rule",
        "termination": "normal",
        "failure_code": "",
        "failure_stage": "",
        "offending_move": "",
        "final_valid_position": {"root_fen": "test-fen", "moves": []},
    }


def respond(request, engine_names):
    ordinal = request["pair_ordinal"]
    directory = Path(request["evidence_directory"])
    os.mkdir(directory)
    pgn_path = directory / "games.pgn"
    result_path = directory / "result.jsonl"
    pgn_blocks = []
    for white, black in (
        (engine_names[0], engine_names[1]),
        (engine_names[1], engine_names[0]),
    ):
        pgn_blocks.append(
            '[Round "%s"]\n'
            '[White "%s"]\n'
            '[Black "%s"]\n'
            '[Result "1/2-1/2"]\n'
            '[SetUp "1"]\n'
            '[FEN "test-fen"]\n'
            '[Variant "alice"]\n'
            '[PlyCount "0"]\n'
            "\n1/2-1/2\n\n" % (ordinal, white, black)
        )
    pgn_path.write_bytes("".join(pgn_blocks).encode("ascii"))
    core = {
        "schema": "alice-pair-result-v1",
        "ordinal": ordinal,
        "game_classes": ["SCORABLE_NATURAL", "SCORABLE_NATURAL"],
        "game_scores": [0.5, 0.5],
        "games": [game(1), game(2)],
    }
    result = dict(core)
    result["evidence_sha256"] = hashlib.sha256(canonical(core)).hexdigest()
    result_path.write_bytes(canonical(result))
    return {
        "schema": "alice-pair-worker-response-v1",
        "pair_ordinal": ordinal,
        "result": result,
        "artifacts": {
            "games_pgn_sha256": sha256_file(pgn_path),
            "result_jsonl_sha256": sha256_file(result_path),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--definition", required=True)
    args = parser.parse_args()
    definition = json.loads(Path(args.definition).read_text(encoding="utf-8"))
    engine_names = tuple(engine["name"] for engine in definition["engines"])
    for line in sys.stdin.buffer:
        request = json.loads(line.decode("utf-8"))
        sys.stdout.buffer.write(canonical(respond(request, engine_names)))
        sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
