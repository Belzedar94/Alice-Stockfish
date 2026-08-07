"""Ordinal, pair-atomic controller for Alice acceptance results."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from .policy import ControlPolicy
from .statistics import paired_statistics


SCORABLE_CLASSES = frozenset({"SCORABLE_NATURAL", "SCORABLE_CLOCK"})
FATAL_CLASSES = frozenset(
    {
        "PROTOCOL_ABORT",
        "SEMANTIC_ABORT",
        "EVIDENCE_ABORT",
        "POLICY_ABORT",
        "UNKNOWN_ABORT",
    }
)
VALID_CLASSES = SCORABLE_CLASSES | frozenset(
    {
        "OPERATIONAL_ABORT",
        "PROTOCOL_ABORT",
        "SEMANTIC_ABORT",
        "EVIDENCE_ABORT",
        "POLICY_ABORT",
        "UNKNOWN_ABORT",
    }
)
VALID_SCORES = frozenset({0.0, 0.5, 1.0})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PairResult:
    ordinal: int
    game_classes: tuple[str, str]
    game_scores: tuple[float | None, float | None]
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("pair ordinal must be a non-negative integer")
        if len(self.game_classes) != 2 or len(self.game_scores) != 2:
            raise ValueError("a pair result must contain exactly two games")
        if any(value not in VALID_CLASSES for value in self.game_classes):
            raise ValueError("unknown game classification")
        if self.evidence_sha256 and not SHA256_RE.fullmatch(self.evidence_sha256):
            raise ValueError("pair evidence SHA-256 must be lowercase hexadecimal")
        for classification, score in zip(self.game_classes, self.game_scores):
            if classification in SCORABLE_CLASSES:
                if type(score) not in (int, float) or score not in VALID_SCORES:
                    raise ValueError("a scorable game requires a 0, 0.5, or 1 score")
            elif score is not None:
                raise ValueError("an unscorable game cannot carry a strength score")

    @property
    def scorable(self) -> bool:
        return all(value in SCORABLE_CLASSES for value in self.game_classes)


SealCallback = Callable[[dict[str, object]], None]


class AcceptanceController:
    """Single-threaded authority for dispatch, admission, statistics, and seal."""

    def __init__(
        self,
        policy: ControlPolicy,
        *,
        seal_callback: SealCallback | None = None,
    ) -> None:
        self.policy = policy
        self.seal_callback = seal_callback
        self.state = "DEFINED"
        self.next_dispatch_ordinal = 0
        self.next_commit_ordinal = 0
        self.outstanding: set[int] = set()
        self.completed_buffer: dict[int, PairResult] = {}
        self.pentanomial = [0, 0, 0, 0, 0]
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.attempted_pairs = 0
        self.runner_complete_pairs = 0
        self.admitted_pairs = 0
        self.discarded_pairs = 0
        self.excluded_after_seal_pairs = 0
        self.excluded_after_terminal_pairs = 0
        self.abort_counts: dict[str, int] = {}
        self.stop_reason: str | None = None
        self.conclusion: str | None = None
        self.seal_payload: dict[str, object] | None = None
        self._terminal_state: str | None = None

    @property
    def admitted_games(self) -> int:
        return self.admitted_pairs * 2

    def mark_preflighted(self) -> None:
        if self.state != "DEFINED":
            raise RuntimeError("preflight can only complete from DEFINED")
        self.state = "PREFLIGHTED"

    def start(self) -> None:
        if self.state != "PREFLIGHTED":
            raise RuntimeError("a control can only start after preflight")
        self.state = "RUNNING"

    def dispatch_window(self) -> tuple[int, int]:
        if self.state != "RUNNING":
            raise RuntimeError("new pairs cannot be dispatched in the current state")
        if self.outstanding or self.completed_buffer:
            raise RuntimeError("the current ordinal window has not drained")
        first = self.next_dispatch_ordinal
        ordinals = (first, first + 1)
        self.next_dispatch_ordinal += 2
        self.outstanding.update(ordinals)
        self.attempted_pairs += 2
        return ordinals

    def submit(self, result: PairResult) -> None:
        if result.ordinal not in self.outstanding:
            raise ValueError("result does not belong to the active ordinal window")
        self.outstanding.remove(result.ordinal)
        self.completed_buffer[result.ordinal] = result
        self.runner_complete_pairs += 1
        self._commit_available()
        self._finish_if_drained()

    def _commit_available(self) -> None:
        while self.next_commit_ordinal in self.completed_buffer:
            result = self.completed_buffer.pop(self.next_commit_ordinal)
            self.next_commit_ordinal += 1
            if self._terminal_state is not None:
                if self.seal_payload is not None:
                    self.excluded_after_seal_pairs += 1
                else:
                    self.excluded_after_terminal_pairs += 1
                continue
            self._apply_pair(result)
            self._apply_attempt_cap_if_needed()

    def _apply_pair(self, result: PairResult) -> None:
        if not result.scorable:
            self.discarded_pairs += 1
            failures = [value for value in result.game_classes if value not in SCORABLE_CLASSES]
            for failure in failures:
                self.abort_counts[failure] = self.abort_counts.get(failure, 0) + 1
            fatal = next((value for value in failures if value in FATAL_CLASSES), None)
            if fatal is not None:
                self.stop_reason = f"fatal-{fatal.lower().replace('_', '-')}"
                self.conclusion = "INVALID"
                self._terminal_state = "INVALID"
                self.state = "INVALID"
            return

        scores = tuple(float(value) for value in result.game_scores if value is not None)
        if len(scores) != 2:
            raise AssertionError("validated scorable pair lost a score")
        bucket = int(round(2.0 * sum(scores)))
        if bucket < 0 or bucket > 4:
            raise AssertionError("pair score does not map to a pentanomial bucket")
        self.pentanomial[bucket] += 1
        for score in scores:
            if score == 1.0:
                self.wins += 1
            elif score == 0.0:
                self.losses += 1
            else:
                self.draws += 1
        self.admitted_pairs += 1

        statistics = paired_statistics(self.pentanomial)
        display = str(statistics["los_percent_display"])
        if self.policy.mode == "exact-los":
            if (
                self.admitted_games > self.policy.minimum_scored_games_exclusive
                and self.admitted_games % 2 == 0
                and display in {"0.0", "100.0"}
            ):
                conclusion = "PASS" if display == "100.0" else "FAIL"
                self._seal(
                    state="DRAINING_EXTREME",
                    terminal_state="SEALED_PASS" if conclusion == "PASS" else "SEALED_FAIL",
                    stop_reason=f"los-{display}",
                    conclusion=conclusion,
                    statistics=statistics,
                )
            elif self.admitted_games >= int(self.policy.maximum_scored_games or 0):
                self._seal(
                    state="DRAINING_CAP",
                    terminal_state="INCONCLUSIVE",
                    stop_reason="game-cap",
                    conclusion="INCONCLUSIVE",
                    statistics=statistics,
                )
        elif self.admitted_games >= int(self.policy.target_admitted_games or 0):
            self._seal(
                state="DRAINING_FIXED",
                terminal_state="FIXED_COMPLETE",
                stop_reason="fixed-target",
                conclusion="FIXED_COMPLETE",
                statistics=statistics,
            )

    def _apply_attempt_cap_if_needed(self) -> None:
        maximum = self.policy.maximum_attempted_games
        committed_games = self.next_commit_ordinal * 2
        if (
            self._terminal_state is None
            and maximum is not None
            and committed_games >= maximum
        ):
            self._seal(
                state="DRAINING_CAP",
                terminal_state="INCONCLUSIVE",
                stop_reason="attempt-game-cap",
                conclusion="INCONCLUSIVE",
                statistics=paired_statistics(self.pentanomial),
            )

    def _seal(
        self,
        *,
        state: str,
        terminal_state: str,
        stop_reason: str,
        conclusion: str,
        statistics: dict[str, object],
    ) -> None:
        if self.seal_payload is not None:
            raise RuntimeError("the control already has an acceptance seal")
        payload: dict[str, object] = {
            "schema": "alice-acceptance-seal-v1",
            "control": self.policy.control,
            "mode": self.policy.mode,
            "attempt_ordinal": self.next_commit_ordinal - 1,
            "admitted_pairs": self.admitted_pairs,
            "scored_games": self.admitted_games,
            "wld": {"wins": self.wins, "losses": self.losses, "draws": self.draws},
            "pentanomial": list(self.pentanomial),
            "statistics": statistics,
            "stop_reason": stop_reason,
            "conclusion": conclusion,
        }
        if self.seal_callback is not None:
            self.seal_callback(payload)
        self.seal_payload = payload
        self.stop_reason = stop_reason
        self.conclusion = conclusion
        self._terminal_state = terminal_state
        self.state = state

    def _finish_if_drained(self) -> None:
        if self._terminal_state is None:
            return
        if not self.outstanding and not self.completed_buffer:
            self.state = self._terminal_state

    def summary(self) -> dict[str, object]:
        statistics = paired_statistics(self.pentanomial)
        return {
            "schema": "alice-acceptance-controller-v1",
            "control": self.policy.control,
            "mode": self.policy.mode,
            "state": self.state,
            "attempted_pairs": self.attempted_pairs,
            "attempted_games": self.attempted_pairs * 2,
            "runner_complete_pairs": self.runner_complete_pairs,
            "admitted_pairs": self.admitted_pairs,
            "discarded_pairs": self.discarded_pairs,
            "excluded_after_seal_pairs": self.excluded_after_seal_pairs,
            "excluded_after_terminal_pairs": self.excluded_after_terminal_pairs,
            "scored_games": self.admitted_games,
            "wld": {"wins": self.wins, "losses": self.losses, "draws": self.draws},
            "pentanomial": list(self.pentanomial),
            "statistics": statistics,
            "abort_counts": dict(sorted(self.abort_counts.items())),
            "stop_reason": self.stop_reason,
            "conclusion": self.conclusion,
        }
