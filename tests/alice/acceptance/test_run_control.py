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

from tools.alice_acceptance.__main__ import (
    FROZEN_LEGACY_NETWORK_NAME,
    prepare_snapshots,
    reject_d_evidence_root,
    run_control,
)
from tools.alice_acceptance.evidence import canonical_json_bytes
from tools.alice_acceptance.runner_adapter import parse_strict_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunControlTests(unittest.TestCase):
    def legacy_snapshot_fixture(
        self, parent: Path, network_name: str
    ) -> tuple[dict[str, object], Path, str]:
        worker = ROOT / "tests/alice/acceptance/fake_pair_worker.py"
        book = parent / "book.epd"
        book.write_text("test-fen\n", encoding="utf-8", newline="\n")
        executable = parent / "engine.bin"
        executable.write_bytes(b"test executable identity\n")
        if os.name != "nt":
            executable.chmod(0o755)
        network = parent / network_name
        network.write_bytes(b"legacy network identity\n")
        network_sha = sha256(network)
        engine = {
            "path": str(executable.resolve()),
            "binary_sha256": sha256(executable),
            "cwd": str(parent.resolve()),
            "name": "Alice-legacy-test",
            "evaluator": "Legacy",
            "network_sha256": network_sha,
            "network_path": str(network.resolve()),
            "time_control": "2+0.02",
            "options": {
                "Threads": "1",
                "Hash": "512",
                "Move Overhead": "10",
                "Use NNUE": "true",
                "Alice Evaluation": "Legacy",
                "Alice_Frozen_Network": "true",
                "EvalFile": str(network.resolve()),
            },
        }
        worker_definition = parent / "worker-definition.json"
        worker_definition.write_text(
            json.dumps(
                {
                    "schema": "alice-pair-worker-definition-v1",
                    "engines": [
                        engine,
                        {
                            **json.loads(json.dumps(engine)),
                            "name": "Alice-legacy-reference",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        run_definition = {
            "seed": 7,
            "book": {"path": str(book.resolve()), "sha256": sha256(book)},
            "pair_worker": {
                "script": str(worker.resolve()),
                "script_sha256": sha256(worker),
                "core": str(worker.resolve()),
                "core_sha256": sha256(worker),
                "definition": str(worker_definition.resolve()),
                "definition_sha256": sha256(worker_definition),
            },
        }
        return run_definition, network, network_sha

    def test_legacy_snapshot_preserves_the_frozen_basename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            definition, _network, network_sha = self.legacy_snapshot_fixture(
                parent, FROZEN_LEGACY_NETWORK_NAME
            )
            evidence = parent / "evidence"
            evidence.mkdir()
            _worker, rewritten, _book, _inventory = prepare_snapshots(
                definition, evidence, "1" * 64, "2" * 64
            )
            value = parse_strict_json(rewritten.read_bytes())
            paths = [Path(engine["network_path"]) for engine in value["engines"]]
            self.assertEqual(paths[0], paths[1])
            self.assertEqual(paths[0].name, FROZEN_LEGACY_NETWORK_NAME)
            self.assertEqual(paths[0].parent.name, network_sha)
            self.assertEqual(sha256(paths[0]), network_sha)
            for engine in value["engines"]:
                self.assertEqual(Path(engine["options"]["EvalFile"]), paths[0])

    def test_legacy_snapshot_rejects_a_noncanonical_basename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            definition, _network, _network_sha = self.legacy_snapshot_fixture(
                parent, "renamed.nnue"
            )
            evidence = parent / "evidence"
            evidence.mkdir()
            with self.assertRaisesRegex(ValueError, "frozen legacy basename"):
                prepare_snapshots(definition, evidence, "1" * 64, "2" * 64)

    def test_normalized_worker_identity_ignores_only_time_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            definition, _network, _network_sha = self.legacy_snapshot_fixture(
                parent, FROZEN_LEGACY_NETWORK_NAME
            )
            worker_definition_path = Path(definition["pair_worker"]["definition"])

            first_evidence = parent / "evidence-first"
            first_evidence.mkdir()
            _worker, _rewritten, _book, first_inventory = prepare_snapshots(
                definition, first_evidence, "1" * 64, "2" * 64
            )

            worker_definition = json.loads(
                worker_definition_path.read_text(encoding="utf-8")
            )
            for engine in worker_definition["engines"]:
                engine["time_control"] = "10+0.1"
            worker_definition_path.write_text(
                json.dumps(worker_definition), encoding="utf-8"
            )
            definition["pair_worker"]["definition_sha256"] = sha256(
                worker_definition_path
            )
            second_evidence = parent / "evidence-second"
            second_evidence.mkdir()
            _worker, _rewritten, _book, second_inventory = prepare_snapshots(
                definition, second_evidence, "3" * 64, "4" * 64
            )
            self.assertEqual(
                first_inventory["normalized_worker_configuration_sha256"],
                second_inventory["normalized_worker_configuration_sha256"],
            )

    def test_snapshot_rejects_nonfrozen_or_extra_uci_options(self) -> None:
        cases = (
            ("missing Threads", lambda options: options.pop("Threads")),
            ("wrong Threads", lambda options: options.__setitem__("Threads", "16")),
            ("missing Hash", lambda options: options.pop("Hash")),
            ("wrong Hash", lambda options: options.__setitem__("Hash", "64")),
            (
                "missing Move Overhead",
                lambda options: options.pop("Move Overhead"),
            ),
            (
                "wrong Move Overhead",
                lambda options: options.__setitem__("Move Overhead", "5000"),
            ),
            (
                "strength handicap",
                lambda options: options.__setitem__("Skill Level", "0"),
            ),
        )
        for label, mutate in cases:
            with (
                self.subTest(case=label),
                tempfile.TemporaryDirectory() as temporary,
            ):
                parent = Path(temporary)
                definition, _network, _network_sha = self.legacy_snapshot_fixture(
                    parent, FROZEN_LEGACY_NETWORK_NAME
                )
                worker_definition_path = Path(
                    definition["pair_worker"]["definition"]
                )
                worker_definition = json.loads(
                    worker_definition_path.read_text(encoding="utf-8")
                )
                mutate(worker_definition["engines"][0]["options"])
                worker_definition_path.write_text(
                    json.dumps(worker_definition), encoding="utf-8"
                )
                definition["pair_worker"]["definition_sha256"] = sha256(
                    worker_definition_path
                )
                evidence = parent / "evidence"
                evidence.mkdir()
                with self.assertRaisesRegex(
                    ValueError, "fields mismatch|frozen evaluator policy"
                ):
                    prepare_snapshots(definition, evidence, "1" * 64, "2" * 64)
                self.assertEqual(list(evidence.iterdir()), [])

    def test_snapshot_rejects_duplicate_engine_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            definition, _network, _network_sha = self.legacy_snapshot_fixture(
                parent, FROZEN_LEGACY_NETWORK_NAME
            )
            worker_definition_path = Path(definition["pair_worker"]["definition"])
            worker_definition = json.loads(
                worker_definition_path.read_text(encoding="utf-8")
            )
            worker_definition["engines"][1]["name"] = worker_definition["engines"][0][
                "name"
            ]
            worker_definition_path.write_text(
                json.dumps(worker_definition), encoding="utf-8"
            )
            definition["pair_worker"]["definition_sha256"] = sha256(
                worker_definition_path
            )
            evidence = parent / "evidence"
            evidence.mkdir()
            with self.assertRaisesRegex(ValueError, "engine names must be distinct"):
                prepare_snapshots(definition, evidence, "1" * 64, "2" * 64)
            self.assertEqual(list(evidence.iterdir()), [])

    def test_snapshot_rejects_extra_per_engine_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            definition, _network, _network_sha = self.legacy_snapshot_fixture(
                parent, FROZEN_LEGACY_NETWORK_NAME
            )
            worker_definition_path = Path(definition["pair_worker"]["definition"])
            worker_definition = json.loads(
                worker_definition_path.read_text(encoding="utf-8")
            )
            worker_definition["engines"][0]["arguments"] = ["--unexpected"]
            worker_definition_path.write_text(
                json.dumps(worker_definition), encoding="utf-8"
            )
            definition["pair_worker"]["definition_sha256"] = sha256(
                worker_definition_path
            )
            evidence = parent / "evidence"
            evidence.mkdir()
            with self.assertRaisesRegex(ValueError, "fields mismatch"):
                prepare_snapshots(definition, evidence, "1" * 64, "2" * 64)
            self.assertEqual(list(evidence.iterdir()), [])

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
                    "Move Overhead": "10",
                    "Use NNUE": "false",
                    "Alice Evaluation": "Zero",
                },
            }
            worker_definition.write_text(
                json.dumps(
                    {
                        "schema": "alice-pair-worker-definition-v1",
                        "engines": [
                            engine,
                            {**dict(engine), "name": "Alice-reference-test"},
                        ],
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
            seal = receipt["sealed_snapshot"]
            self.assertEqual(
                hashlib.sha256(canonical_json_bytes(seal)).hexdigest(),
                receipt["sealed_snapshot_sha256"],
            )
            for field in (
                "admitted_pairs",
                "scored_games",
                "wld",
                "pentanomial",
                "statistics",
                "stop_reason",
                "conclusion",
            ):
                self.assertEqual(seal[field], result[field])
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
