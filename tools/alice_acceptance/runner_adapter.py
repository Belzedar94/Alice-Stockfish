"""Binary JSON-lines adapter for the persistent Alice pair worker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
from typing import BinaryIO

from .controller import PairResult
from .evidence import canonical_json_bytes, sha256_file


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SCORABLE_CLASSES = frozenset({"SCORABLE_NATURAL", "SCORABLE_CLOCK"})
RESULTS = frozenset({"1-0", "0-1", "1/2-1/2"})
PGN_HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+) "(.*)"\]$')
MOVE_NUMBER_RE = re.compile(r"^\d+\.(?:\.\.)?$")


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def parse_strict_json(payload: bytes) -> dict[str, object]:
    value = json.loads(
        payload.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("worker response must be a JSON object")
    return value


def parse_pair_pgn(
    payload: bytes,
) -> list[tuple[dict[str, str], list[str], str]]:
    text = payload.decode("ascii", errors="strict")
    lines = text.splitlines()
    games: list[tuple[dict[str, str], list[str], str]] = []
    index = 0
    while index < len(lines):
        while index < len(lines) and not lines[index]:
            index += 1
        if index >= len(lines):
            break
        headers: dict[str, str] = {}
        while index < len(lines) and lines[index].startswith("["):
            match = PGN_HEADER_RE.fullmatch(lines[index])
            if not match or match.group(1) in headers:
                raise ValueError("pair PGN contains malformed or duplicate headers")
            headers[match.group(1)] = match.group(2)
            index += 1
        if index >= len(lines) or lines[index]:
            raise ValueError("pair PGN header block is not terminated")
        index += 1
        move_lines: list[str] = []
        while index < len(lines) and lines[index]:
            move_lines.append(lines[index])
            index += 1
        movetext = re.sub(r"\{[^}]*\}", " ", " ".join(move_lines))
        tokens = [
            token
            for token in movetext.split()
            if not MOVE_NUMBER_RE.fullmatch(token)
        ]
        if (
            not tokens
            or tokens[-1] not in RESULTS
            or any(token in RESULTS for token in tokens[:-1])
        ):
            raise ValueError(
                "pair PGN movetext result is missing, misplaced, or repeated"
            )
        games.append((headers, tokens[:-1], tokens[-1]))
    if len(games) != 2:
        raise ValueError("pair PGN must contain exactly two games")
    return games


def validate_worker_response(
    response: dict[str, object],
    expected_ordinal: int,
    pair_directory: str | Path,
    expected_fen: str,
    expected_engine_names: tuple[str, str],
) -> PairResult:
    if (
        len(expected_engine_names) != 2
        or any(
            not isinstance(name, str) or not name for name in expected_engine_names
        )
        or expected_engine_names[0] == expected_engine_names[1]
    ):
        raise ValueError("expected engine identities must be two distinct names")
    if set(response) != {"schema", "pair_ordinal", "result", "artifacts"}:
        raise ValueError("pair-worker response fields do not match the contract")
    if response.get("schema") != "alice-pair-worker-response-v1":
        raise ValueError("unsupported pair-worker response schema")
    if (
        type(response.get("pair_ordinal")) is not int
        or response.get("pair_ordinal") != expected_ordinal
    ):
        raise ValueError("pair-worker response ordinal mismatch")
    result = response.get("result")
    artifacts = response.get("artifacts")
    if not isinstance(result, dict) or not isinstance(artifacts, dict):
        raise ValueError("pair-worker response lacks result or artifacts")
    if set(artifacts) != {"games_pgn_sha256", "result_jsonl_sha256"}:
        raise ValueError("pair-worker artifact fields do not match the contract")
    if any(
        not isinstance(value, str) or not SHA256_RE.fullmatch(value)
        for value in artifacts.values()
    ):
        raise ValueError("pair-worker artifact hashes must be lowercase SHA-256")
    expected_result_fields = {
        "schema",
        "ordinal",
        "game_classes",
        "game_scores",
        "games",
        "evidence_sha256",
    }
    if set(result) != expected_result_fields:
        raise ValueError("pair-result fields do not match the contract")
    if result.get("schema") != "alice-pair-result-v1":
        raise ValueError("unsupported pair-result schema")
    if type(result.get("ordinal")) is not int or result.get("ordinal") != expected_ordinal:
        raise ValueError("pair-result ordinal mismatch")
    evidence_sha = result.get("evidence_sha256")
    if not isinstance(evidence_sha, str) or not SHA256_RE.fullmatch(evidence_sha):
        raise ValueError("pair-result evidence identity is missing")
    core = dict(result)
    del core["evidence_sha256"]
    if hashlib.sha256(canonical_json_bytes(core)).hexdigest() != evidence_sha:
        raise ValueError("pair-result evidence SHA-256 mismatch")
    directory = Path(pair_directory)
    pgn_path = directory / "games.pgn"
    result_path = directory / "result.jsonl"
    if not pgn_path.is_file() or pgn_path.stat().st_size == 0:
        raise ValueError("pair PGN is missing or empty")
    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise ValueError("pair result file is missing or empty")
    if artifacts.get("games_pgn_sha256") != sha256_file(pgn_path):
        raise ValueError("pair PGN SHA-256 mismatch")
    if artifacts.get("result_jsonl_sha256") != sha256_file(result_path):
        raise ValueError("pair result-file SHA-256 mismatch")
    stored = parse_strict_json(result_path.read_bytes())
    if canonical_json_bytes(stored) != canonical_json_bytes(result):
        raise ValueError("pair response and durable result differ")
    classes = result.get("game_classes")
    scores = result.get("game_scores")
    games = result.get("games")
    if (
        not isinstance(classes, list)
        or not isinstance(scores, list)
        or not isinstance(games, list)
        or len(classes) != 2
        or len(scores) != 2
        or len(games) != 2
    ):
        raise ValueError("pair result lacks classifications or scores")
    expected_game_fields = {
        "game_number",
        "result",
        "outcome_class",
        "reason",
        "termination",
        "failure_code",
        "failure_stage",
        "offending_move",
        "final_valid_position",
    }
    pgn_games = parse_pair_pgn(pgn_path.read_bytes())
    for index, game in enumerate(games):
        if not isinstance(game, dict) or set(game) != expected_game_fields:
            raise ValueError("game evidence fields do not match the contract")
        if type(game.get("game_number")) is not int or game.get("game_number") != index + 1:
            raise ValueError("game evidence order is not canonical")
        result_token = game.get("result")
        classification = game.get("outcome_class")
        if result_token not in RESULTS or classification != classes[index]:
            raise ValueError("game evidence contradicts the pair summary")
        for field in (
            "reason",
            "termination",
            "failure_code",
            "failure_stage",
            "offending_move",
        ):
            if not isinstance(game.get(field), str):
                raise ValueError("game evidence text fields must be strings")
        position = game.get("final_valid_position")
        if not isinstance(position, dict) or set(position) != {"root_fen", "moves"}:
            raise ValueError("final valid position evidence is malformed")
        if position.get("root_fen") != expected_fen:
            raise ValueError("final valid position root FEN contradicts the request")
        moves = position.get("moves")
        if not isinstance(moves, list) or any(not isinstance(move, str) for move in moves):
            raise ValueError("final valid position moves must be strings")
        headers, pgn_moves, pgn_result = pgn_games[index]
        expected_white, expected_black = (
            expected_engine_names
            if index == 0
            else (expected_engine_names[1], expected_engine_names[0])
        )
        if (
            headers.get("White") != expected_white
            or headers.get("Black") != expected_black
            or headers.get("FEN") != expected_fen
            or headers.get("Result") != result_token
            or headers.get("Variant") != "alice"
            or headers.get("SetUp") != "1"
            or headers.get("PlyCount") != str(len(moves))
            or pgn_moves != moves
            or pgn_result != result_token
        ):
            raise ValueError("pair PGN contradicts the machine game evidence")
        expected_termination = game.get("termination")
        if expected_termination == "normal":
            if "Termination" in headers:
                raise ValueError("normal game has an unexpected PGN termination")
        elif headers.get("Termination") != expected_termination:
            raise ValueError("pair PGN termination contradicts machine evidence")
        if classification not in SCORABLE_CLASSES:
            if (
                headers.get("OutcomeClass") != classification
                or headers.get("FailureCode") != game.get("failure_code")
                or headers.get("FailureStage") != game.get("failure_stage")
                or headers.get("OffendingMove") != game.get("offending_move")
            ):
                raise ValueError("pair PGN failure evidence contradicts the result")
        if classification in SCORABLE_CLASSES:
            white_score = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}[result_token]
            contender_score = white_score if index == 0 else 1.0 - white_score
            if scores[index] != contender_score:
                raise ValueError("game result contradicts the contender score")
        elif scores[index] is not None:
            raise ValueError("unscorable game carries a strength score")
    return PairResult(
        ordinal=expected_ordinal,
        game_classes=tuple(classes),  # type: ignore[arg-type]
        game_scores=tuple(scores),  # type: ignore[arg-type]
        evidence_sha256=evidence_sha,
    )


class PersistentWorkerClient:
    def __init__(
        self,
        worker_script: str | Path,
        definition: str | Path,
        stderr_path: str | Path,
    ) -> None:
        self.stderr_file: BinaryIO = open(stderr_path, "xb")
        self.process = subprocess.Popen(
            [sys.executable, str(worker_script), "--definition", str(definition)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            bufsize=0,
        )
        self.lines: queue.Queue[bytes | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()
        self.lock = threading.Lock()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def request(self, payload: dict[str, object], timeout: float) -> dict[str, object]:
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError("pair worker exited before the request")
            assert self.process.stdin is not None
            self.process.stdin.write(canonical_json_bytes(payload))
            self.process.stdin.flush()
            try:
                line = self.lines.get(timeout=timeout)
            except queue.Empty as error:
                raise TimeoutError("pair worker did not return before the timeout") from error
            if line is None:
                raise RuntimeError("pair worker closed stdout without a response")
            return parse_strict_json(line)

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.reader.join(timeout=2)
        self.stderr_file.flush()
        self.stderr_file.close()
