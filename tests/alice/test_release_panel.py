from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "alice" / "run_release_panel.py"
SPEC = importlib.util.spec_from_file_location("alice_release_panel", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
panel = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = panel
SPEC.loader.exec_module(panel)


class ReleasePanelTests(unittest.TestCase):
    def test_matrix_is_exact_and_uses_one_seed(self) -> None:
        self.assertEqual(
            [(item.label, item.tc, item.games) for item in panel.PANEL_SPECS],
            [
                ("VSTC", "2+0.02", 400),
                ("STC", "10+0.1", 300),
                ("LTC", "30+0.3", 200),
            ],
        )
        self.assertEqual({item.seed for item in panel.PANEL_SPECS}, {20260811})
        self.assertEqual({item.hash_mb for item in panel.PANEL_SPECS}, {512})

    def test_tracker_records_complete_color_pair(self) -> None:
        spec = panel.PANEL_SPECS[2]
        tracker = panel.PanelTracker(spec)
        for game in range(1, spec.games + 1):
            white, black = (
                (panel.CANDIDATE_NAME, panel.BASELINE_NAME)
                if game % 2
                else (panel.BASELINE_NAME, panel.CANDIDATE_NAME)
            )
            tracker.consume(
                f"Finished game {game} ({white} vs {black}): "
                "1/2-1/2 {Draw by 3-fold repetition}"
            )
        result = tracker.require_complete()
        self.assertEqual(result["games"], 200)
        self.assertEqual(result["complete_pairs"], 100)
        self.assertEqual(result["wdl"], {"wins": 0, "draws": 200, "losses": 0})
        self.assertEqual(result["pentanomial"], [0, 0, 100, 0, 0])

    def test_tracker_fails_on_adjudication(self) -> None:
        tracker = panel.PanelTracker(panel.PANEL_SPECS[0])
        with self.assertRaisesRegex(RuntimeError, "infrastructure defect"):
            tracker.consume(
                "Finished game 1 (Alice-Stockfish-release vs "
                "Fairy-Stockfish-040925): 1/2-1/2 "
                "{Draw by adjudication: max game length}"
            )

    def test_tracker_fails_on_time_loss(self) -> None:
        tracker = panel.PanelTracker(panel.PANEL_SPECS[0])
        with self.assertRaisesRegex(RuntimeError, "infrastructure defect"):
            tracker.consume(
                "Finished game 1 (Alice-Stockfish-release vs "
                "Fairy-Stockfish-040925): 0-1 {White loses on time}"
            )

    def test_tracker_rejects_wrong_engine_identity(self) -> None:
        tracker = panel.PanelTracker(panel.PANEL_SPECS[0])
        with self.assertRaisesRegex(RuntimeError, "unexpected engines"):
            tracker.consume(
                "Finished game 1 (wrong vs Fairy-Stockfish-040925): "
                "1/2-1/2 {Draw by rule}"
            )

    def test_statistics_are_neutral_for_all_draws(self) -> None:
        stats = panel.statistics_from_penta([0, 0, 10, 0, 0])
        self.assertEqual(stats, {"elo": 0.0, "ci95": 0.0, "los": 0.5})

    def test_command_selects_legacy_on_both_sides_and_excludes_v2(self) -> None:
        root = Path("C:/frozen")
        args = SimpleNamespace(
            candidate=root / "alice.exe",
            candidate_network=root / "alice_run2rl_e40_l09.nnue",
            baseline=root / "fairy.exe",
            baseline_network=root / "alice_run2rl_e40_l09.nnue",
            referee=root / "uci_pair_runner.py",
            referee_python=root / "python.exe",
            book=root / "alice.epd",
            concurrency_per_tc=4,
        )
        command = panel.build_command(
            args, panel.PANEL_SPECS[0], root / "games.pgn"
        )
        joined = "\n".join(str(item) for item in command)
        self.assertIn("option.Alice Evaluation=Legacy", joined)
        self.assertEqual(joined.count("alice_run2rl_e40_l09.nnue"), 2)
        self.assertNotIn("NativeV2", joined)
        self.assertNotIn("02A26647", joined)
        self.assertIn("-variant\nalice", joined)

    def test_write_json_is_atomic_and_canonical_enough_for_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            panel.write_json(path, {"z": 1, "a": 2})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "a": 2,\n  "z": 1\n}\n')
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
