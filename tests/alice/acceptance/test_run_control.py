from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.alice_acceptance.__main__ import reject_d_evidence_root, run_control


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunControlTests(unittest.TestCase):
    def test_fixed_ltc_runs_through_two_persistent_processes(self) -> None:
        worker = ROOT / "tests/alice/acceptance/fake_pair_worker.py"
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            book = parent / "book.epd"
            book.write_text("test-fen\n", encoding="utf-8", newline="\n")
            executable = parent / "engine.bin"
            executable.write_bytes(b"test executable identity\n")
            if os.name != "nt":
                executable.chmod(0o755)
            worker_definition = parent / "worker-definition.json"
            engine = {
                "path": str(executable.resolve()),
                "binary_sha256": sha256(executable),
                "cwd": str(parent.resolve()),
                "name": "Alice-test",
                "evaluator": "Zero",
                "network_sha256": "",
                "time_control": "30+0.3",
                "options": {
                    "Threads": "1",
                    "Hash": "512",
                    "Use NNUE": "false",
                    "Alice Evaluation": "Zero",
                },
            }
            worker_definition.write_text(
                json.dumps(
                    {
                        "schema": "alice-pair-worker-definition-v1",
                        "engines": [engine, dict(engine)],
                        "max_plies": 8,
                    }
                ),
                encoding="utf-8",
            )
            definition = parent / "definition.json"
            definition.write_text(
                json.dumps(
                    {
                        "schema": "alice-acceptance-run-definition-v1",
                        "run_id": "fixed-ltc-test",
                        "control": "LTC",
                        "mode": "fixed-final",
                        "seed": 7,
                        "book": {
                            "path": str(book.resolve()),
                            "sha256": sha256(book),
                        },
                        "pair_worker": {
                            "script": str(worker.resolve()),
                            "script_sha256": sha256(worker),
                            "core": str(worker.resolve()),
                            "core_sha256": sha256(worker),
                            "definition": str(worker_definition.resolve()),
                            "definition_sha256": sha256(worker_definition),
                            "request_timeout_seconds": 10,
                        },
                    }
                ),
                encoding="utf-8",
            )
            evidence = parent / "evidence"
            receipt = run_control(definition, evidence)

            self.assertEqual(receipt["status"], "finalized")
            self.assertFalse(receipt["strength_release_authorized"])
            result = receipt["result"]
            self.assertEqual(result["conclusion"], "FIXED_COMPLETE")
            self.assertEqual(result["scored_games"], 200)
            self.assertEqual(result["attempted_pairs"], 100)
            self.assertEqual(result["pentanomial"], [0, 0, 100, 0, 0])
            self.assertEqual(len(list((evidence / "controls/LTC/pairs").iterdir())), 100)
            self.assertTrue((evidence / "controls/LTC/seal.json").is_file())
            self.assertTrue((evidence / "receipt.json").is_file())
            if os.name != "nt":
                snapshot = evidence / "inputs/snapshots/engine-1/engine.bin"
                self.assertTrue(os.access(snapshot, os.X_OK))

    @unittest.skipUnless(sys.platform.startswith("win"), "Windows drive syntax")
    def test_evidence_on_d_is_rejected_without_touching_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be written"):
            reject_d_evidence_root(Path("D:/alice-evidence"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
