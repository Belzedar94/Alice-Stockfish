"""Paired statistics frozen for the Alice local acceptance battery."""

from __future__ import annotations

import math
from typing import Iterable


PENTANOMIAL_BUCKETS = 5
NORMAL_975 = 1.959963984540054


def _validated_counts(counts: Iterable[int]) -> tuple[int, int, int, int, int]:
    values = tuple(counts)
    if len(values) != PENTANOMIAL_BUCKETS:
        raise ValueError("pentanomial counts must contain exactly five buckets")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("pentanomial counts must be non-negative integers")
    return values  # type: ignore[return-value]


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _logistic_elo(score: float) -> float | None:
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def paired_statistics(counts: Iterable[int]) -> dict[str, object]:
    """Return LOS and Elo from contender-perspective complete-pair buckets.

    Buckets are ``[LL, LD, DD_or_WL, DW, WW]`` and therefore represent pair
    scores 0, 0.5, 1, 1.5, and 2 points. The normalized observations used by
    the normal approximation are 0, 0.25, 0.5, 0.75, and 1.
    """

    values = _validated_counts(counts)
    pair_count = sum(values)
    if pair_count == 0:
        mean = 0.5
        variance = 0.0
        standard_error = 0.0
        los = 0.5
        lower_score = upper_score = 0.5
    else:
        observations = tuple(index / 4.0 for index in range(PENTANOMIAL_BUCKETS))
        mean = sum(observations[index] * values[index] for index in range(5)) / pair_count
        variance = (
            sum(
                ((observations[index] - mean) ** 2) * values[index]
                for index in range(5)
            )
            / pair_count
        )
        standard_error = math.sqrt(variance / pair_count)
        if variance == 0.0:
            los = 0.5 if mean == 0.5 else float(mean > 0.5)
        else:
            los = _normal_cdf((mean - 0.5) / standard_error)
        lower_score = mean - NORMAL_975 * standard_error
        upper_score = mean + NORMAL_975 * standard_error

    return {
        "pair_count": pair_count,
        "mean_score": mean,
        "variance": variance,
        "standard_error": standard_error,
        "los_probability": los,
        "los_probability_binary64_hex": los.hex(),
        "los_percent_display": format(100.0 * los, ".1f"),
        "elo_95": {
            "lower": _logistic_elo(lower_score),
            "estimate": _logistic_elo(mean),
            "upper": _logistic_elo(upper_score),
        },
    }
