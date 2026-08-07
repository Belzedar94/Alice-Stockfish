from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.alice_acceptance.aggregate import FIXED_GAMES, aggregate_receipts
from tools.alice_acceptance.evidence import write_create_only_json
from tools.alice_acceptance.statistics import paired_statistics
from tools.alice_acceptance.policy import TIMING_CONTROLS


def control_receipt(control: str, mode: str) -> dict[str, object]:
    fixed = mode == "fixed-final"
    scored_games = FIXED_GAMES[control] if fixed else 102
    admitted_pairs = scored_games // 2
    conclusion = "FIXED_COMPLETE" if fixed else "PASS"
    pentanomial = [0, 0, 0, 0, admitted_pairs]
    base_ms, increment_ms = TIMING_CONTROLS[control]
    return {
        "schema": "alice-control-receipt-v1",
        "run_id": f"source-{control.lower()}",
        "status": "finalized",
        "times": {},
        "policy": {
            "control": control,
            "mode": mode,
            "base_ms": base_ms,
            "increment_ms": increment_ms,
            "pair_workers": 2,
            "engine_threads": 1,
            "hash_mib": 512,
            "external_adjudication": "disabled",
            "commit_order": "attempt-ordinal",
            "maximum_scored_games": None if fixed else 64000,
            "maximum_attempted_games": None if fixed else 64000,
            "target_admitted_games": scored_games if fixed else None,
        },
        "inputs": {
            "schema": "alice-acceptance-input-inventory-v1",
            "source_definition_sha256": "1" * 64,
            "canonical_definition_sha256": "2" * 64,
            "book_sha256": "3" * 64,
            "pair_worker_sha256": "4" * 64,
            "pair_core_sha256": "5" * 64,
            "source_worker_definition_sha256": "6" * 64,
            "worker_definition_sha256": "7" * 64,
            "normalized_worker_configuration_sha256": "c" * 64,
            "engines": [
                {
                    "role": "contender",
                    "binary_sha256": "8" * 64,
                    "network_sha256": "9" * 64,
                    "evaluator": "Native",
                    "options_sha256": "d" * 64,
                },
                {
                    "role": "reference",
                    "binary_sha256": "a" * 64,
                    "network_sha256": "b" * 64,
                    "evaluator": "Legacy",
                    "options_sha256": "e" * 64,
                },
            ],
        },
        "result": {
            "schema": "alice-acceptance-controller-v1",
            "control": control,
            "mode": mode,
            "state": conclusion if fixed else "SEALED_PASS",
            "conclusion": conclusion,
            "attempted_pairs": admitted_pairs,
            "attempted_games": scored_games,
            "runner_complete_pairs": admitted_pairs,
            "admitted_pairs": admitted_pairs,
            "scored_games": scored_games,
            "discarded_pairs": 0,
            "excluded_after_seal_pairs": 0,
            "excluded_after_terminal_pairs": 0,
            "abort_counts": {},
            "wld": {"wins": scored_games, "losses": 0, "draws": 0},
            "pentanomial": pentanomial,
            "statistics": paired_statistics(pentanomial),
            "stop_reason": "fixed-target" if fixed else "los-100.0",
        },
        "sealed_snapshot_sha256": "1" * 64,
        "artifacts": {},
        "strength_release_authorized": False,
    }


class AggregateReceiptTests(unittest.TestCase):
    def materialize(self, directory: Path, mode: str) -> dict[str, Path]:
        paths = {}
        for control in ("VSTC", "STC", "LTC"):
            path = directory / f"{control}.json"
            write_create_only_json(path, control_receipt(control, mode))
            paths[control] = path
        return paths

    def test_exact_and_fixed_aggregates_are_distinct_complete_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exact = aggregate_receipts(
                "exact-battery", "exact-los", self.materialize(root, "exact-los")
            )
        self.assertEqual(exact["conclusion"], "PASS")
        self.assertTrue(exact["strength_gate_eligible"])
        self.assertEqual(set(exact["controls"]), {"VSTC", "STC", "LTC"})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = aggregate_receipts(
                "fixed-battery", "fixed-final", self.materialize(root, "fixed-final")
            )
        self.assertEqual(fixed["conclusion"], "FIXED_COMPLETE")
        self.assertTrue(fixed["strength_gate_eligible"])

    def test_abort_or_non_extreme_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.materialize(root, "exact-los")
            bad = control_receipt("STC", "exact-los")
            bad["result"]["discarded_pairs"] = 1
            bad["result"]["abort_counts"] = {"OPERATIONAL_ABORT": 1}
            paths["STC"].unlink()
            write_create_only_json(paths["STC"], bad)
            with self.assertRaisesRegex(ValueError, "nonzero abort"):
                aggregate_receipts("bad-battery", "exact-los", paths)

    def test_contradictory_or_unsealed_receipt_is_rejected(self) -> None:
        for mutation, message in (
            (
                lambda receipt: receipt.__setitem__("sealed_snapshot_sha256", None),
                "immutable seal",
            ),
            (
                lambda receipt: receipt["result"]["wld"].__setitem__("wins", 0),
                "WLD totals",
            ),
            (
                lambda receipt: receipt["result"].__setitem__(
                    "state", "DRAINING_EXTREME"
                ),
                "exact LOS gate",
            ),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = self.materialize(root, "exact-los")
                bad = control_receipt("STC", "exact-los")
                mutation(bad)
                paths["STC"].unlink()
                write_create_only_json(paths["STC"], bad)
                with self.assertRaisesRegex(ValueError, message):
                    aggregate_receipts("bad-battery", "exact-los", paths)

    def test_controls_with_different_uci_options_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.materialize(root, "exact-los")
            bad = control_receipt("STC", "exact-los")
            bad["inputs"]["engines"][0]["options_sha256"] = "f" * 64
            bad["inputs"]["normalized_worker_configuration_sha256"] = "0" * 64
            paths["STC"].unlink()
            write_create_only_json(paths["STC"], bad)
            with self.assertRaisesRegex(ValueError, "pinned input identity"):
                aggregate_receipts("bad-options", "exact-los", paths)


if __name__ == "__main__":
    unittest.main(verbosity=2)
