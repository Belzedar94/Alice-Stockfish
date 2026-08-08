from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.alice_acceptance.controller import AcceptanceController, PairResult
from tools.alice_acceptance.policy import exact_los_policy, fixed_final_policy


SCORES = {
    0: (0.0, 0.0),
    1: (0.0, 0.5),
    2: (0.5, 0.5),
    3: (1.0, 0.5),
    4: (1.0, 1.0),
}


def pair(ordinal: int, bucket: int) -> PairResult:
    return PairResult(
        ordinal=ordinal,
        game_classes=("SCORABLE_NATURAL", "SCORABLE_NATURAL"),
        game_scores=SCORES[bucket],
    )


def running(policy) -> AcceptanceController:
    controller = AcceptanceController(policy)
    controller.mark_preflighted()
    controller.start()
    return controller


def admit_even_pairs(controller: AcceptanceController, buckets: list[int]) -> None:
    if len(buckets) % 2:
        raise ValueError("test helper requires complete two-pair windows")
    for offset in range(0, len(buckets), 2):
        ordinals = controller.dispatch_window()
        controller.submit(pair(ordinals[0], buckets[offset]))
        controller.submit(pair(ordinals[1], buckets[offset + 1]))


class SchedulerTests(unittest.TestCase):
    def test_policy_objects_reject_boolean_and_float_numeric_fields(self) -> None:
        policy = exact_los_policy("VSTC")
        for field, value in (("engine_threads", True), ("base_ms", 2000.0)):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "canonical integers"
            ):
                replace(policy, **{field: value})

    def test_exact_extreme_requires_more_than_one_hundred_games(self) -> None:
        controller = running(exact_los_policy("VSTC"))
        admit_even_pairs(controller, [4] * 50)
        self.assertEqual(controller.admitted_games, 100)
        self.assertEqual(controller.state, "RUNNING")
        self.assertIsNone(controller.seal_payload)

    def test_out_of_order_completion_seals_the_ordinal_prefix(self) -> None:
        controller = running(exact_los_policy("VSTC"))
        admit_even_pairs(controller, [4] * 50)
        first, second = controller.dispatch_window()
        controller.submit(pair(second, 4))
        self.assertEqual(controller.admitted_games, 100)
        self.assertEqual(controller.runner_complete_pairs, 51)
        with self.assertRaises(RuntimeError):
            controller.dispatch_window()
        controller.submit(pair(first, 4))
        self.assertEqual(controller.state, "SEALED_PASS")
        self.assertEqual(controller.admitted_games, 102)
        self.assertEqual(controller.excluded_after_seal_pairs, 1)
        self.assertEqual(controller.seal_payload["attempt_ordinal"], first)
        self.assertEqual(controller.conclusion, "PASS")

    def test_zero_extreme_is_a_failure(self) -> None:
        controller = running(exact_los_policy("STC"))
        admit_even_pairs(controller, [0] * 50)
        first, second = controller.dispatch_window()
        controller.submit(pair(first, 0))
        controller.submit(pair(second, 0))
        self.assertEqual(controller.state, "SEALED_FAIL")
        self.assertEqual(controller.conclusion, "FAIL")
        self.assertEqual(controller.admitted_games, 102)
        self.assertEqual(controller.excluded_after_seal_pairs, 1)

    def test_all_draws_reach_the_cap_as_inconclusive(self) -> None:
        controller = running(exact_los_policy("LTC"))
        admit_even_pairs(controller, [2] * 32000)
        self.assertEqual(controller.state, "INCONCLUSIVE")
        self.assertEqual(controller.admitted_games, 64000)
        self.assertEqual(controller.conclusion, "INCONCLUSIVE")
        self.assertEqual(controller.summary()["statistics"]["los_percent_display"], "50.0")

    def test_fixed_gate_ignores_interim_extremes(self) -> None:
        controller = running(fixed_final_policy("LTC"))
        admit_even_pairs(controller, [4] * 50)
        self.assertEqual(controller.state, "RUNNING")
        self.assertEqual(controller.admitted_games, 100)
        admit_even_pairs(controller, [4] * 50)
        self.assertEqual(controller.state, "FIXED_COMPLETE")
        self.assertEqual(controller.admitted_games, 200)
        self.assertEqual(controller.conclusion, "FIXED_COMPLETE")

    def test_one_unscorable_game_discards_the_complete_pair(self) -> None:
        controller = running(exact_los_policy("VSTC"))
        first, second = controller.dispatch_window()
        controller.submit(
            PairResult(
                ordinal=first,
                game_classes=("SCORABLE_CLOCK", "OPERATIONAL_ABORT"),
                game_scores=(1.0, None),
            )
        )
        self.assertEqual(controller.discarded_pairs, 1)
        self.assertEqual(controller.admitted_games, 0)
        controller.submit(pair(second, 2))
        self.assertEqual(controller.admitted_games, 2)
        self.assertEqual(controller.summary()["wld"], {"wins": 0, "losses": 0, "draws": 2})

    def test_semantic_abort_invalidates_and_drains_without_scoring(self) -> None:
        controller = running(exact_los_policy("VSTC"))
        first, second = controller.dispatch_window()
        controller.submit(
            PairResult(
                ordinal=first,
                game_classes=("SCORABLE_NATURAL", "SEMANTIC_ABORT"),
                game_scores=(1.0, None),
            )
        )
        self.assertEqual(controller.state, "INVALID")
        controller.submit(pair(second, 4))
        self.assertEqual(controller.admitted_games, 0)
        self.assertEqual(controller.excluded_after_terminal_pairs, 1)
        self.assertEqual(controller.conclusion, "INVALID")

    def test_protocol_abort_is_fatal(self) -> None:
        controller = running(exact_los_policy("VSTC"))
        first, second = controller.dispatch_window()
        controller.submit(
            PairResult(
                ordinal=first,
                game_classes=("PROTOCOL_ABORT", "SCORABLE_NATURAL"),
                game_scores=(None, 0.5),
            )
        )
        controller.submit(pair(second, 2))
        self.assertEqual(controller.state, "INVALID")
        self.assertEqual(controller.stop_reason, "fatal-protocol-abort")
        self.assertEqual(controller.excluded_after_terminal_pairs, 1)

    def test_attempt_cap_applies_even_when_every_pair_is_operationally_discarded(self) -> None:
        controller = running(exact_los_policy("LTC"))
        operational = PairResult(
            ordinal=0,
            game_classes=("OPERATIONAL_ABORT", "OPERATIONAL_ABORT"),
            game_scores=(None, None),
        )
        for _window in range(16000):
            first, second = controller.dispatch_window()
            controller.submit(
                PairResult(
                    ordinal=first,
                    game_classes=operational.game_classes,
                    game_scores=operational.game_scores,
                )
            )
            controller.submit(
                PairResult(
                    ordinal=second,
                    game_classes=operational.game_classes,
                    game_scores=operational.game_scores,
                )
            )
        self.assertEqual(controller.state, "INCONCLUSIVE")
        self.assertEqual(controller.admitted_games, 0)
        self.assertEqual(controller.summary()["attempted_games"], 64000)
        self.assertEqual(controller.stop_reason, "attempt-game-cap")


if __name__ == "__main__":
    unittest.main(verbosity=2)
