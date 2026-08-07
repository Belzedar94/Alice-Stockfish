from __future__ import annotations

import json
import hashlib
from pathlib import Path
from unittest import mock
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.alice_acceptance.evidence import (
    canonical_json_bytes,
    sha256_file,
    write_create_only_json,
)
from tools.alice_acceptance.aggregate import CONTROLS, FIXED_GAMES, aggregate_receipts
from tools.alice_acceptance.policy import TIMING_CONTROLS
from tools.alice_acceptance.statistics import paired_statistics
from tools import alice_release_evidence


def reference(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def write_json(path: Path, value: dict[str, object]) -> None:
    write_create_only_json(path, value)


def control_receipt(
    control: str, mode: str, network_sha256: str
) -> dict[str, object]:
    fixed = mode == "fixed-final"
    scored_games = FIXED_GAMES[control] if fixed else 102
    admitted_pairs = scored_games // 2
    conclusion = "FIXED_COMPLETE" if fixed else "PASS"
    pentanomial = [0, 0, 0, 0, admitted_pairs]
    base_ms, increment_ms = TIMING_CONTROLS[control]
    return {
        "schema": "alice-control-receipt-v1",
        "run_id": f"source-{mode}-{control.lower()}",
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
            "engines": [
                {
                    "role": "contender",
                    "binary_sha256": "8" * 64,
                    "network_sha256": network_sha256,
                    "evaluator": "Native",
                },
                {
                    "role": "reference",
                    "binary_sha256": "a" * 64,
                    "network_sha256": "b" * 64,
                    "evaluator": "Legacy",
                },
            ],
        },
        "result": {
            "schema": "alice-acceptance-controller-v1",
            "control": control,
            "mode": mode,
            "state": conclusion if fixed else "SEALED_PASS",
            "attempted_pairs": admitted_pairs,
            "attempted_games": scored_games,
            "runner_complete_pairs": admitted_pairs,
            "admitted_pairs": admitted_pairs,
            "discarded_pairs": 0,
            "excluded_after_seal_pairs": 0,
            "excluded_after_terminal_pairs": 0,
            "scored_games": scored_games,
            "wld": {"wins": scored_games, "losses": 0, "draws": 0},
            "pentanomial": pentanomial,
            "statistics": paired_statistics(pentanomial),
            "abort_counts": {},
            "stop_reason": "fixed-target" if fixed else "los-100.0",
            "conclusion": conclusion,
        },
        "sealed_snapshot_sha256": "1" * 64,
        "artifacts": {},
        "strength_release_authorized": False,
    }


class ReleaseEvidenceTests(unittest.TestCase):
    def build_candidate(self, root: Path) -> tuple[Path, dict[str, object], int]:
        network = root / "alice.nnue"
        network.write_bytes(b"native network fixture\n")
        network_sha = sha256_file(network)

        qualification = root / "qualification.json"
        write_json(
            qualification,
            {
                "schema": "alice-native-qualification-v1",
                "status": "qualified",
                "network_sha256": network_sha,
                "network_kind": "trained",
                "training_run_id": "training-fixture",
                "dataset_manifest_sha256": "1" * 64,
                "dataset_position_count": 1024,
                "checkpoint_sha256": "2" * 64,
                "export_receipt_sha256": "3" * 64,
                "checkpoint_file_element_count": 2048,
                "checkpoint_file_element_mismatches": 0,
                "file_engine_position_count": 32,
                "file_engine_centipawn_difference": 0,
                "incremental_full_position_count": 32,
                "incremental_full_mismatches": 0,
                "network_parameter_nonzero_count": 64,
                "gates": {f"G{index}": "PASS" for index in range(1, 9)},
            },
        )

        exact = root / "exact.json"
        fixed = root / "fixed.json"
        for mode, output in (("exact-los", exact), ("fixed-final", fixed)):
            paths = {}
            for control in CONTROLS:
                path = root / f"{mode}-{control}.json"
                write_json(path, control_receipt(control, mode, network_sha))
                paths[control] = path
            write_json(
                output,
                aggregate_receipts(f"{mode}-fixture", mode, paths),
            )

        shadow = root / "shadow.json"
        write_json(
            shadow,
            {
                "schema": "alice-openbench-shadow-receipt-v1",
                "service": "https://belzedar.duckdns.org",
                "status": "PASS",
                "presets": {
                    control: {
                        "pairs": 200,
                        "inversions": 0,
                        "invalid_pairs": 0,
                        "adjudication": ["800/4", "40/8/10"],
                    }
                    for control in ("VSTC", "STC", "LTC")
                },
            },
        )

        binaries = []
        for index, role in enumerate(sorted(alice_release_evidence.BINARY_ROLES)):
            binary = root / f"{role}.bin"
            binary.write_bytes(f"binary {index}\n".encode("ascii"))
            binary_sha = sha256_file(binary)
            bench = root / f"{role}-bench.json"
            write_json(
                bench,
                {
                    "schema": "alice-triple-bench-v1",
                    "binary_sha256": binary_sha,
                    "network_sha256": network_sha,
                    "signatures": ["Nodes searched : 162582"] * 3,
                },
            )
            failures = root / f"{role}-failures.json"
            case = {
                "exit_nonzero": True,
                "fallback_observed": False,
                "search_result_published": False,
            }
            write_json(
                failures,
                {
                    "schema": "alice-load-failure-matrix-v1",
                    "binary_sha256": binary_sha,
                    "network_sha256": network_sha,
                    "cases": {
                        "missing": dict(case),
                        "corrupt": dict(case),
                        "incompatible": dict(case),
                    },
                },
            )
            binaries.append(
                {
                    "role": role,
                    "artifact": reference(binary),
                    "triple_bench": reference(bench),
                    "load_failures": reference(failures),
                }
            )

        manifest_value = {
            "schema": "alice-release-candidate-v1",
            "release_id": "alice-test",
            "source_commit": "a" * 40,
            "network": reference(network),
            "native_qualification": reference(qualification),
            "exact_los_receipt": reference(exact),
            "fixed_final_receipt": reference(fixed),
            "openbench_shadow_receipt": reference(shadow),
            "binaries": binaries,
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
        return manifest, manifest_value, network.stat().st_size

    def test_complete_candidate_is_authorized_but_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _value, size = self.build_candidate(Path(temporary))
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertEqual(receipt["status"], "ready")
        self.assertTrue(receipt["strength_release_authorized"])
        self.assertFalse(receipt["publication_performed"])
        self.assertEqual(receipt["blocking_reasons"], [])

    def test_fallback_observation_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            first = value["binaries"][0]
            failures_path = Path(first["load_failures"]["path"])
            failures = json.loads(failures_path.read_text(encoding="utf-8"))
            failures["cases"]["corrupt"]["fallback_observed"] = True
            failures_path.write_text(json.dumps(failures), encoding="utf-8")
            first["load_failures"]["sha256"] = sha256_file(failures_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertEqual(receipt["status"], "blocked")
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(any("did not fail closed" in reason for reason in receipt["blocking_reasons"]))

    def test_structural_network_claim_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            qualification_path = Path(value["native_qualification"]["path"])
            qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
            qualification["network_parameter_nonzero_count"] = 0
            qualification_path.write_text(json.dumps(qualification), encoding="utf-8")
            value["native_qualification"]["sha256"] = sha256_file(qualification_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any("network_parameter_nonzero_count" in reason for reason in receipt["blocking_reasons"])
        )

    def test_local_battery_for_another_network_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            exact_path = Path(value["exact_los_receipt"]["path"])
            exact = json.loads(exact_path.read_text(encoding="utf-8"))
            for control, item in exact["controls"].items():
                embedded = item["receipt"]
                embedded["inputs"]["engines"][0]["network_sha256"] = "c" * 64
                receipt_sha = hashlib.sha256(canonical_json_bytes(embedded)).hexdigest()
                item["receipt_sha256"] = receipt_sha
                exact["inputs"]["control_receipt_sha256"][control] = receipt_sha
            exact_path.write_bytes(canonical_json_bytes(exact))
            value["exact_los_receipt"]["sha256"] = sha256_file(exact_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any("does not bind the candidate native network" in reason for reason in receipt["blocking_reasons"])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
