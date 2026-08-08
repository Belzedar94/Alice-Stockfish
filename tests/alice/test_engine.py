"""Black-box conformance checks for an Alice-Stockfish executable."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import queue
import random
import re
import subprocess
import sys
import threading
import time
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
BOOK_PATH: Path | None = None
BOOK_SHA256 = "BCD89D9FC3EA81FEB95932EB64D6B6F15AD25CC04CDCC9E0440F097CFFB8CCF6"
BOOK_UNIQUE_POSITIONS = 38348


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


class UciSession:
    """Interactive executable session with bounded output waits."""

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

    def wait_for(self, pattern: str, timeout: float = 10.0) -> str:
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

    def test_build_provenance_is_clean(self) -> None:
        result = run_engine("compiler")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Source tree state          : clean", result.stdout)
        self.assertNotIn("Source tree state          : dirty", result.stdout)

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
        case = next(
            case
            for case in self.cases
            if case["id"] == "sixteen-wide-input-normalizes-to-compact-fen"
        )
        result = run_engine(f"position fen {case['inputFen']}", "d")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(inspected_fen(result.stdout), case["expected"]["canonicalFen"])

    def test_frozen_opening_book_is_exact_and_fully_parseable(self) -> None:
        if BOOK_PATH is None:
            self.skipTest("Pass --book to validate the frozen OpenBench opening corpus.")
        assert BOOK_PATH is not None

        payload = BOOK_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest().upper(), BOOK_SHA256)
        fens = [
            line.split(";", 1)[0].strip()
            for line in payload.decode("utf-8-sig").splitlines()
            if line.split(";", 1)[0].strip()
        ]
        self.assertEqual(len(fens), BOOK_UNIQUE_POSITIONS)
        self.assertEqual(len(set(fens)), BOOK_UNIQUE_POSITIONS)

        result = run_engine(*(f"position fen {fen}" for fen in fens), "d")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(inspected_fen(result.stdout), Position.from_fen(fens[-1]).fen())

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

    def test_safe_search_is_deterministic(self) -> None:
        searches: list[tuple[str, str]] = []
        with UciSession() as session:
            session.send("uci")
            session.wait_for(r"^uciok$")
            self.assertTrue(any("option name EvalFile" in line for line in session.lines))
            self.assertIn(
                "option name Alice Evaluation type combo default Legacy var Native var Zero",
                session.lines,
            )
            session.send("setoption name Use NNUE value false")

            for _ in range(2):
                session.send("position startpos")
                session.send("go depth 3")
                bestmove = session.wait_for(r"^bestmove ")
                info = next(
                    line
                    for line in reversed(session.lines)
                    if line.startswith("info depth 3 ")
                )
                pv = re.search(r" score (\S+ \S+).* pv (.+)$", info)
                self.assertIsNotNone(pv, info)
                searches.append((bestmove, pv.group(2)))

            self.assertEqual(searches[0], searches[1])
            self.assertEqual(searches[0], ("bestmove a2a3 ponder a7a6", "a2a3 a7a6 b2b3"))
            self.assertNotIn("NNUE evaluation using", "\n".join(session.lines))

    def test_search_evaluator_contract_fails_closed_and_unwinds(self) -> None:
        result = run_engine("alice_search_verify_contract")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(
            result.stdout,
            r"alice_search contract verified cases 5 balanced_pushes \d+ "
            r"balanced_pops \d+ balanced_evaluations \d+ injected_failures 3 "
            r"stopped_cases 1 root_restorations 5",
        )

    def test_safe_search_finds_an_alice_mate_in_one(self) -> None:
        fen = "8/6|Q1/8/8/8/8/k7/2K5 w - - 0 1"
        with UciSession() as session:
            session.send("setoption name Use NNUE value false")
            session.send(f"position fen {fen}")
            session.send("go depth 1")
            bestmove = session.wait_for(r"^bestmove ")
            self.assertEqual(bestmove, "bestmove g7b2")
            self.assertTrue(
                any("info depth 1 " in line and "score mate 1" in line for line in session.lines),
                "\n".join(session.lines),
            )

    def test_safe_search_stop_is_prompt_and_preserves_the_position(self) -> None:
        with UciSession() as session:
            session.send("setoption name Use NNUE value false")
            session.send("position startpos")
            session.send("go infinite")
            session.wait_for(r"^info depth 1 ")

            session.send("d")
            inspected = session.wait_for(r"^Fen: ", timeout=2.0)
            self.assertEqual(inspected.removeprefix("Fen: "), START_FEN)

            session.send("stop")
            bestmove = session.wait_for(r"^bestmove ", timeout=2.0)
            move = bestmove.split()[1]
            self.assertIn(
                move,
                {candidate.uci() for candidate in Position.from_fen(START_FEN).legal_moves()},
            )

    def test_terminal_mate_and_zero_diagnostic_mode(self) -> None:
        mate = "8/6|Kk/8/8/8/3Q4/8/8 b - - 0 1"
        with UciSession() as session:
            session.send("setoption name Use NNUE value false")
            session.send(f"position fen {mate}")
            session.send("go depth 3")
            bestmove = session.wait_for(r"^bestmove ")
            self.assertEqual(bestmove, "bestmove (none)")
            self.assertTrue(any("score mate 0" in line for line in session.lines))
            self.assertIn(
                "info string alice_result result=1-0 reason=checkmate",
                session.lines,
            )

            session.send("eval")
            session.wait_for(r"^legacy_nnue raw 0 adjusted 0$")

    def test_rule_draw_reports_an_explicit_terminal_record(self) -> None:
        rule_draw = START_FEN.replace(" 0 1", " 100 1")
        with UciSession() as session:
            session.send("setoption name Use NNUE value false")
            session.send(f"position fen {rule_draw}")
            session.send("go depth 3")
            terminal = session.wait_for(r"^info string alice_result ")
            self.assertEqual(
                terminal,
                "info string alice_result result=1/2-1/2 reason=rule_draw",
            )
            self.assertEqual(session.wait_for(r"^bestmove "), "bestmove (none)")

    def test_normal_search_without_a_network_fails_closed(self) -> None:
        result = run_engine("position startpos", "go depth 1")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("no compatible network is loaded", result.stdout)
        self.assertIn("CRITICAL ERROR", result.stdout)


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine", type=Path, default=default_engine_path())
    parser.add_argument("--book", type=Path)
    return parser.parse_known_args()


if __name__ == "__main__":
    arguments, unittest_arguments = parse_arguments()
    ENGINE_PATH = arguments.engine.resolve()
    BOOK_PATH = arguments.book.resolve() if arguments.book else None
    unittest.main(argv=[sys.argv[0], *unittest_arguments], verbosity=2)
