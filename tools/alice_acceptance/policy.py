"""Frozen timing-control and stopping policies."""

from __future__ import annotations

from dataclasses import dataclass


TIMING_CONTROLS = {
    "VSTC": (2000, 20),
    "STC": (10000, 100),
    "LTC": (30000, 300),
}
FIXED_FINAL_GAMES = {"VSTC": 400, "STC": 300, "LTC": 200}


@dataclass(frozen=True)
class ControlPolicy:
    control: str
    mode: str
    base_ms: int
    increment_ms: int
    pair_workers: int = 2
    engine_threads: int = 1
    hash_mib: int = 512
    minimum_scored_games_exclusive: int = 100
    maximum_scored_games: int | None = None
    maximum_attempted_games: int | None = None
    target_admitted_games: int | None = None
    early_stop: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            self.base_ms,
            self.increment_ms,
            self.pair_workers,
            self.engine_threads,
            self.hash_mib,
            self.minimum_scored_games_exclusive,
        )
        optional_integer_fields = (
            self.maximum_scored_games,
            self.maximum_attempted_games,
            self.target_admitted_games,
        )
        if any(type(value) is not int for value in integer_fields) or any(
            value is not None and type(value) is not int
            for value in optional_integer_fields
        ):
            raise ValueError("control policy numeric fields must be canonical integers")
        if type(self.early_stop) is not bool:
            raise ValueError("control policy early_stop must be a boolean")
        if self.control not in TIMING_CONTROLS:
            raise ValueError(f"unknown timing control: {self.control}")
        if (self.base_ms, self.increment_ms) != TIMING_CONTROLS[self.control]:
            raise ValueError("timing values do not match the frozen control")
        if self.pair_workers != 2:
            raise ValueError("the local battery requires exactly two pair workers")
        if self.engine_threads != 1 or self.hash_mib != 512:
            raise ValueError("the local battery requires Threads=1 and Hash=512")
        if self.mode == "exact-los":
            if (
                not self.early_stop
                or self.maximum_scored_games != 64000
                or self.maximum_attempted_games != 64000
            ):
                raise ValueError(
                    "exact LOS mode requires early stop and 64,000-game caps"
                )
            if self.target_admitted_games is not None:
                raise ValueError("exact LOS mode has no fixed target")
        elif self.mode == "fixed-final":
            expected = FIXED_FINAL_GAMES[self.control]
            if self.early_stop or self.target_admitted_games != expected:
                raise ValueError("fixed-final target does not match the frozen control")
            if (
                self.maximum_scored_games is not None
                or self.maximum_attempted_games is not None
            ):
                raise ValueError("fixed-final mode uses its exact target, not a cap")
        else:
            raise ValueError(f"unknown acceptance mode: {self.mode}")


def exact_los_policy(control: str) -> ControlPolicy:
    base_ms, increment_ms = TIMING_CONTROLS[control]
    return ControlPolicy(
        control=control,
        mode="exact-los",
        base_ms=base_ms,
        increment_ms=increment_ms,
        maximum_scored_games=64000,
        maximum_attempted_games=64000,
        early_stop=True,
    )


def fixed_final_policy(control: str) -> ControlPolicy:
    base_ms, increment_ms = TIMING_CONTROLS[control]
    return ControlPolicy(
        control=control,
        mode="fixed-final",
        base_ms=base_ms,
        increment_ms=increment_ms,
        target_admitted_games=FIXED_FINAL_GAMES[control],
        early_stop=False,
    )
