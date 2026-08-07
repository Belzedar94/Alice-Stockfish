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


def write_test_binary(
    path: Path, role: str, source_commit: str, source_tree_state: str = "clean"
) -> None:
    executable_format, architecture = alice_release_evidence.BINARY_ROLE_REQUIREMENTS[
        role
    ]
    if executable_format == "windows-x86-64":
        payload = bytearray(128)
        payload[:2] = b"MZ"
        payload[60:64] = (64).to_bytes(4, "little")
        payload[64:68] = b"PE\x00\x00"
        payload[68:70] = (0x8664).to_bytes(2, "little")
        payload[88:90] = (0x20B).to_bytes(2, "little")
        platform_marker = " on MinGW64"
    else:
        payload = bytearray(64)
        payload[:4] = b"\x7fELF"
        payload[4] = 2
        payload[5] = 1
        payload[18:20] = (62).to_bytes(2, "little")
        platform_marker = " on Linux"
    payload.extend(
        (
            f"\x00{role}\x00{architecture}\x00{platform_marker}\x00"
            f"{source_commit}\x00Source tree state          : {source_tree_state}\x00"
        ).encode("ascii")
    )
    path.write_bytes(payload)


def control_receipt(
    control: str, mode: str, network_sha256: str, opening_seed: int = 7
) -> dict[str, object]:
    fixed = mode == "fixed-final"
    scored_games = FIXED_GAMES[control] if fixed else 102
    admitted_pairs = scored_games // 2
    conclusion = "FIXED_COMPLETE" if fixed else "PASS"
    pentanomial = [0, 0, 0, 0, admitted_pairs]
    base_ms, increment_ms = TIMING_CONTROLS[control]
    receipt = {
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
            "opening_seed": opening_seed,
            "pair_worker_sha256": "4" * 64,
            "pair_core_sha256": "5" * 64,
            "source_worker_definition_sha256": "6" * 64,
            "worker_definition_sha256": "7" * 64,
            "normalized_worker_configuration_sha256": "c" * 64,
            "engines": [
                {
                    "role": "contender",
                    "binary_sha256": "8" * 64,
                    "network_sha256": network_sha256,
                    "evaluator": "Native",
                    "options_sha256": "d" * 64,
                },
                {
                    "role": "reference",
                    "binary_sha256": alice_release_evidence.FROZEN_LEGACY_BINARY_SHA256,
                    "network_sha256": alice_release_evidence.FROZEN_LEGACY_NETWORK_SHA256,
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
        "sealed_snapshot": None,
        "sealed_snapshot_sha256": "",
        "artifacts": {},
        "strength_release_authorized": False,
    }
    result = receipt["result"]
    seal = {
        "schema": "alice-acceptance-seal-v1",
        "control": control,
        "mode": mode,
        "attempt_ordinal": admitted_pairs - 1,
        "admitted_pairs": result["admitted_pairs"],
        "scored_games": result["scored_games"],
        "wld": result["wld"],
        "pentanomial": result["pentanomial"],
        "statistics": result["statistics"],
        "stop_reason": result["stop_reason"],
        "conclusion": result["conclusion"],
    }
    receipt["sealed_snapshot"] = seal
    receipt["sealed_snapshot_sha256"] = hashlib.sha256(
        canonical_json_bytes(seal)
    ).hexdigest()
    return receipt


class ReleaseEvidenceTests(unittest.TestCase):
    def mutate_aggregate(
        self,
        path: Path,
        mutation,
    ) -> None:
        aggregate = json.loads(path.read_text(encoding="utf-8"))
        for control, item in aggregate["controls"].items():
            embedded = item["receipt"]
            mutation(embedded)
            receipt_sha = hashlib.sha256(canonical_json_bytes(embedded)).hexdigest()
            item["receipt_sha256"] = receipt_sha
            aggregate["inputs"]["control_receipt_sha256"][control] = receipt_sha
        path.write_bytes(canonical_json_bytes(aggregate))

    def build_candidate(self, root: Path) -> tuple[Path, dict[str, object], int]:
        source_commit = "a" * 40
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

        binaries = []
        for role in sorted(alice_release_evidence.BINARY_ROLES):
            binary = root / f"{role}.bin"
            write_test_binary(binary, role, source_commit)
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

        contender_binary_sha256 = binaries[0]["artifact"]["sha256"]
        for output in (exact, fixed):
            self.mutate_aggregate(
                output,
                lambda receipt: receipt["inputs"]["engines"][0].__setitem__(
                    "binary_sha256", contender_binary_sha256
                ),
            )

        shadow = root / "shadow.json"
        shadow_binary = binaries[0]
        write_json(
            shadow,
            {
                "schema": "alice-openbench-shadow-receipt-v1",
                "service": "https://belzedar.duckdns.org",
                "status": "PASS",
                "source_commit": source_commit,
                "network_sha256": network_sha,
                "presets": {
                    control: {
                        "binary_role": shadow_binary["role"],
                        "binary_sha256": shadow_binary["artifact"]["sha256"],
                        "pairs": 200,
                        "inversions": 0,
                        "invalid_pairs": 0,
                        "adjudication": ["800/4", "40/8/10"],
                    }
                    for control in ("VSTC", "STC", "LTC")
                },
            },
        )

        manifest_value = {
            "schema": "alice-release-candidate-v1",
            "release_id": "alice-test",
            "source_commit": source_commit,
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

    def test_local_batteries_with_different_uci_options_block_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            fixed_path = Path(value["fixed_final_receipt"]["path"])

            def change_options(receipt: dict[str, object]) -> None:
                receipt["inputs"]["engines"][0]["options_sha256"] = "f" * 64
                receipt["inputs"]["normalized_worker_configuration_sha256"] = (
                    "0" * 64
                )

            self.mutate_aggregate(fixed_path, change_options)
            value["fixed_final_receipt"]["sha256"] = sha256_file(fixed_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "do not share one pinned input identity" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_release_batteries_may_declare_independent_opening_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            fixed_path = Path(value["fixed_final_receipt"]["path"])

            self.mutate_aggregate(
                fixed_path,
                lambda receipt: receipt["inputs"].__setitem__("opening_seed", 8),
            )
            value["fixed_final_receipt"]["sha256"] = sha256_file(fixed_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertEqual(receipt["status"], "ready")
        self.assertTrue(receipt["strength_release_authorized"])

    def test_zero_reference_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)

            def select_zero_reference(receipt: dict[str, object]) -> None:
                reference_engine = receipt["inputs"]["engines"][1]
                reference_engine["evaluator"] = "Zero"
                reference_engine["network_sha256"] = None

            for field in ("exact_los_receipt", "fixed_final_receipt"):
                path = Path(value[field]["path"])
                self.mutate_aggregate(path, select_zero_reference)
                value[field]["sha256"] = sha256_file(path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "frozen historical reference" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_local_battery_from_a_nonrelease_binary_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)

            def select_unreleased_binary(receipt: dict[str, object]) -> None:
                receipt["inputs"]["engines"][0]["binary_sha256"] = "f" * 64

            for field in ("exact_los_receipt", "fixed_final_receipt"):
                path = Path(value[field]["path"])
                self.mutate_aggregate(path, select_unreleased_binary)
                value[field]["sha256"] = sha256_file(path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "candidate release binary" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_binary_from_another_source_commit_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            binary_entry = value["binaries"][1]
            binary_path = Path(binary_entry["artifact"]["path"])
            write_test_binary(binary_path, binary_entry["role"], "b" * 40)
            binary_sha = sha256_file(binary_path)
            binary_entry["artifact"]["sha256"] = binary_sha
            for field in ("triple_bench", "load_failures"):
                evidence_path = Path(binary_entry[field]["path"])
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                evidence["binary_sha256"] = binary_sha
                evidence_path.write_bytes(canonical_json_bytes(evidence))
                binary_entry[field]["sha256"] = sha256_file(evidence_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "declared full source commit" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_binary_from_dirty_source_tree_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            binary_entry = value["binaries"][1]
            binary_path = Path(binary_entry["artifact"]["path"])
            write_test_binary(
                binary_path,
                binary_entry["role"],
                value["source_commit"],
                source_tree_state="dirty",
            )
            binary_sha = sha256_file(binary_path)
            binary_entry["artifact"]["sha256"] = binary_sha
            for field in ("triple_bench", "load_failures"):
                evidence_path = Path(binary_entry[field]["path"])
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                evidence["binary_sha256"] = binary_sha
                evidence_path.write_bytes(canonical_json_bytes(evidence))
                binary_entry[field]["sha256"] = sha256_file(evidence_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "dirty source-tree marker" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_reused_binary_sha_across_release_roles_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            first, second = value["binaries"][:2]
            first_path = Path(first["artifact"]["path"])
            second_path = Path(second["artifact"]["path"])
            second_path.write_bytes(first_path.read_bytes())
            reused_sha = sha256_file(second_path)
            second["artifact"]["sha256"] = reused_sha
            for field in ("triple_bench", "load_failures"):
                evidence_path = Path(second[field]["path"])
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                evidence["binary_sha256"] = reused_sha
                evidence_path.write_bytes(canonical_json_bytes(evidence))
                second[field]["sha256"] = sha256_file(evidence_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "binary SHA-256 is reused" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_openbench_shadow_for_another_candidate_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            shadow_path = Path(value["openbench_shadow_receipt"]["path"])
            shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
            shadow["source_commit"] = "b" * 40
            shadow["network_sha256"] = "c" * 64
            for preset in shadow["presets"].values():
                preset["binary_sha256"] = "d" * 64
            shadow_path.write_bytes(canonical_json_bytes(shadow))
            value["openbench_shadow_receipt"]["sha256"] = sha256_file(shadow_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        reasons = receipt["blocking_reasons"]
        self.assertTrue(any("candidate source commit" in reason for reason in reasons))
        self.assertTrue(any("candidate network" in reason for reason in reasons))
        self.assertTrue(any("candidate binary" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main(verbosity=2)
