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


RUNNER_FIXTURE = Path(__file__).resolve().with_name("fake_pair_worker.py")
RUNNER_FIXTURE_SHA256 = sha256_file(RUNNER_FIXTURE)


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
            "pair_worker_sha256": RUNNER_FIXTURE_SHA256,
            "pair_worker_path": str(RUNNER_FIXTURE),
            "pair_core_sha256": RUNNER_FIXTURE_SHA256,
            "pair_core_path": str(RUNNER_FIXTURE),
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
        "artifacts": {
            "openings_jsonl_sha256": "f" * 64,
            "status_jsonl_sha256": "0" * 64,
        },
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

        dataset_manifest = root / "dataset-manifest.json"
        write_json(
            dataset_manifest,
            {
                "schema": "alice-training-dataset-manifest-v1",
                "training_run_id": "training-fixture",
                "position_count": 1024,
                "train_position_count": 900,
                "validation_position_count": 124,
                "split_seed": 7,
            },
        )
        checkpoint = root / "checkpoint.pt"
        checkpoint.write_bytes(b"trained checkpoint fixture\n")
        export_receipt = root / "export-receipt.json"
        write_json(
            export_receipt,
            {
                "schema": "alice-native-export-receipt-v1",
                "training_run_id": "training-fixture",
                "checkpoint_sha256": sha256_file(checkpoint),
                "network_sha256": network_sha,
                "network_bytes": network.stat().st_size,
                "element_count": alice_release_evidence.NATIVE_PARAMETER_ELEMENTS,
                "element_mismatches": 0,
                "deterministic_reexport_sha256": network_sha,
            },
        )
        dataset_reference = reference(dataset_manifest)
        checkpoint_reference = reference(checkpoint)
        export_reference = reference(export_receipt)
        gate_references = {}
        for gate in (f"G{index}" for index in range(1, 9)):
            if gate in {"G1", "G2", "G6"}:
                sample_count = 1024
            elif gate in {"G4", "G5"}:
                sample_count = alice_release_evidence.NATIVE_PARAMETER_ELEMENTS
            else:
                sample_count = 32
            report = root / f"{gate.lower()}-report.json"
            write_json(
                report,
                {
                    "schema": "alice-native-gate-report-v1",
                    "gate": gate,
                    "status": "PASS",
                    "training_run_id": "training-fixture",
                    "network_sha256": network_sha,
                    "dataset_manifest_sha256": dataset_reference["sha256"],
                    "checkpoint_sha256": checkpoint_reference["sha256"],
                    "export_receipt_sha256": export_reference["sha256"],
                    "sample_count": sample_count,
                    "mismatch_count": 0,
                },
            )
            gate_references[gate] = reference(report)

        qualification = root / "qualification.json"
        write_json(
            qualification,
            {
                "schema": "alice-native-qualification-v1",
                "status": "qualified",
                "network_sha256": network_sha,
                "network_kind": "trained",
                "training_run_id": "training-fixture",
                "dataset_manifest": dataset_reference,
                "checkpoint": checkpoint_reference,
                "export_receipt": export_reference,
                "network_parameter_nonzero_bytes": (
                    alice_release_evidence.count_native_parameter_nonzero_bytes(network)
                ),
                "gates": gate_references,
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
            bench_runs = []
            for ordinal in range(3):
                command = root / f"{role}-bench-{ordinal}-command.json"
                write_json(
                    command,
                    {
                        "schema": "alice-triple-bench-command-v1",
                        "ordinal": ordinal,
                        "binary_path": str(binary.resolve()),
                        "binary_sha256": binary_sha,
                        "network_path": str(network.resolve()),
                        "network_sha256": network_sha,
                        "stdin": alice_release_evidence.TRIPLE_BENCH_STDIN,
                    },
                )
                stdout = root / f"{role}-bench-{ordinal}-stdout.txt"
                stdout.write_text(
                    f"Alice native network sha256={network_sha}\n"
                    "Nodes searched : 202963\n",
                    encoding="utf-8",
                    newline="\n",
                )
                bench_runs.append(
                    {
                        "ordinal": ordinal,
                        "command": reference(command),
                        "stdout": reference(stdout),
                        "exit_code": 0,
                    }
                )
            write_json(
                bench,
                {
                    "schema": "alice-triple-bench-v1",
                    "binary_sha256": binary_sha,
                    "network_sha256": network_sha,
                    "runs": bench_runs,
                },
            )
            failures = root / f"{role}-failures.json"
            cases = {}
            for name, (mutation, diagnostic_code) in (
                alice_release_evidence.LOAD_FAILURE_PROBES.items()
            ):
                input_path = root / f"{role}-{name}-input.nnue"
                input_sha = None
                if name != "missing":
                    payload = bytearray(network.read_bytes())
                    payload[0 if name == "corrupt" else 4] ^= 0x01
                    input_path.write_bytes(payload)
                    input_sha = sha256_file(input_path)
                descriptor = root / f"{role}-{name}-input.json"
                write_json(
                    descriptor,
                    {
                        "schema": "alice-load-probe-input-v1",
                        "probe_kind": name,
                        "source_network_sha256": network_sha,
                        "input_path": str(input_path.resolve()),
                        "input_sha256": input_sha,
                        "mutation": mutation,
                    },
                )
                descriptor_reference = reference(descriptor)
                command = root / f"{role}-{name}-command.json"
                write_json(
                    command,
                    {
                        "schema": "alice-load-probe-command-v1",
                        "probe_kind": name,
                        "binary_sha256": binary_sha,
                        "source_network_sha256": network_sha,
                        "input_descriptor_sha256": descriptor_reference["sha256"],
                        "input_sha256": input_sha,
                    },
                )
                command_reference = reference(command)
                output = root / f"{role}-{name}-output.json"
                write_json(
                    output,
                    {
                        "schema": "alice-load-probe-output-v1",
                        "probe_kind": name,
                        "binary_sha256": binary_sha,
                        "command_sha256": command_reference["sha256"],
                        "diagnostic_code": diagnostic_code,
                        "exit_code": 3,
                        "fallback_observed": False,
                        "search_result_published": False,
                    },
                )
                cases[name] = {
                    "probe_kind": name,
                    "source_network_sha256": network_sha,
                    "input_descriptor": descriptor_reference,
                    "input": None if name == "missing" else reference(input_path),
                    "mutation": mutation,
                    "command": command_reference,
                    "output": reference(output),
                    "diagnostic_code": diagnostic_code,
                    "exit_code": 3,
                    "fallback_observed": False,
                    "search_result_published": False,
                }
            write_json(
                failures,
                {
                    "schema": "alice-load-failure-matrix-v1",
                    "binary_sha256": binary_sha,
                    "network_sha256": network_sha,
                    "cases": cases,
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
        shadow_presets = {}
        for control in ("VSTC", "STC", "LTC"):
            configuration = root / f"shadow-{control.lower()}-configuration.json"
            write_json(
                configuration,
                {
                    "schema": "alice-openbench-shadow-configuration-v1",
                    "service": "https://belzedar.duckdns.org",
                    "preset": control,
                    "source_commit": source_commit,
                    "network_sha256": network_sha,
                    "binary_role": shadow_binary["role"],
                    "binary_sha256": shadow_binary["artifact"]["sha256"],
                    "book_token": "ALICE",
                    "book_sha256": alice_release_evidence.FROZEN_ALICE_BOOK_SHA256,
                    "runner_sha256": "9" * 64,
                    "engine_options": alice_release_evidence.SHADOW_ENGINE_OPTIONS,
                    "timing": alice_release_evidence.SHADOW_TIMING[control],
                    "worker": alice_release_evidence.SHADOW_WORKER_CONFIGURATION,
                    "adjudication": ["800/4", "40/8/10"],
                },
            )
            shadow_presets[control] = {
                "binary_role": shadow_binary["role"],
                "binary_sha256": shadow_binary["artifact"]["sha256"],
                "configuration": reference(configuration),
                "pairs": 200,
                "inversions": 0,
                "invalid_pairs": 0,
                "adjudication": ["800/4", "40/8/10"],
            }
        write_json(
            shadow,
            {
                "schema": "alice-openbench-shadow-receipt-v1",
                "service": "https://belzedar.duckdns.org",
                "status": "PASS",
                "source_commit": source_commit,
                "network_sha256": network_sha,
                "presets": shadow_presets,
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
            with mock.patch.object(
                alice_release_evidence, "EXPECTED_NATIVE_SIZE", size
            ):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertEqual(receipt["status"], "ready")
        self.assertTrue(receipt["strength_release_authorized"])
        self.assertFalse(receipt["publication_performed"])
        self.assertEqual(receipt["blocking_reasons"], [])

    def test_triple_bench_requires_the_canonical_node_count(self) -> None:
        for signature in ("error", "Nodes searched : 202964"):
            with (
                self.subTest(signature=signature),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                manifest, value, size = self.build_candidate(root)
                bench_reference = value["binaries"][0]["triple_bench"]
                bench_path = Path(bench_reference["path"])
                bench = json.loads(bench_path.read_text(encoding="utf-8"))
                stdout_reference = bench["runs"][0]["stdout"]
                stdout_path = Path(stdout_reference["path"])
                stdout_path.write_text(
                    f"Alice native network sha256={value['network']['sha256']}\n"
                    f"{signature}\n",
                    encoding="utf-8",
                    newline="\n",
                )
                stdout_reference["sha256"] = sha256_file(stdout_path)
                bench_path.write_bytes(canonical_json_bytes(bench))
                bench_reference["sha256"] = sha256_file(bench_path)
                manifest.write_text(json.dumps(value), encoding="utf-8")
                with mock.patch.object(
                    alice_release_evidence, "EXPECTED_NATIVE_SIZE", size
                ):
                    receipt = alice_release_evidence.audit_release_candidate(manifest)
            self.assertFalse(receipt["strength_release_authorized"])
            self.assertTrue(
                any(
                    "stdout lacks the canonical bench evidence" in reason
                    for reason in receipt["blocking_reasons"]
                )
            )

    def test_triple_bench_artifact_hashes_are_recomputed(self) -> None:
        cases = (
            ("command", "triple bench run 0 command: SHA-256 mismatch"),
            ("stdout", "triple bench run 0 stdout: SHA-256 mismatch"),
        )
        for artifact, expected_reason in cases:
            with (
                self.subTest(artifact=artifact),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                manifest, value, size = self.build_candidate(root)
                bench_path = Path(value["binaries"][0]["triple_bench"]["path"])
                bench = json.loads(bench_path.read_text(encoding="utf-8"))
                artifact_path = Path(bench["runs"][0][artifact]["path"])
                artifact_path.write_text(
                    "tampered bench evidence\n", encoding="utf-8"
                )
                with mock.patch.object(
                    alice_release_evidence, "EXPECTED_NATIVE_SIZE", size
                ):
                    receipt = alice_release_evidence.audit_release_candidate(manifest)
            self.assertFalse(receipt["strength_release_authorized"])
            self.assertTrue(
                any(
                    expected_reason in reason
                    for reason in receipt["blocking_reasons"]
                )
            )

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
        self.assertTrue(
            any(
                "did not fail closed" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_load_failure_cases_must_bind_distinct_probe_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            first = value["binaries"][0]
            failures_path = Path(first["load_failures"]["path"])
            failures = json.loads(failures_path.read_text(encoding="utf-8"))
            failures["cases"]["corrupt"] = dict(failures["cases"]["missing"])
            failures_path.write_bytes(canonical_json_bytes(failures))
            first["load_failures"]["sha256"] = sha256_file(failures_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(
                alice_release_evidence, "EXPECTED_NATIVE_SIZE", size
            ):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "corrupt load-failure evidence" in reason
                or "distinct executions" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_load_failure_artifact_hashes_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            first = value["binaries"][0]
            failures = json.loads(
                Path(first["load_failures"]["path"]).read_text(encoding="utf-8")
            )
            output_path = Path(failures["cases"]["corrupt"]["output"]["path"])
            output_path.write_bytes(b"tampered probe evidence\n")
            with mock.patch.object(
                alice_release_evidence, "EXPECTED_NATIVE_SIZE", size
            ):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "corrupt output: SHA-256 mismatch" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_load_failure_mutation_is_recomputed_from_network_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            first = value["binaries"][0]
            failures_path = Path(first["load_failures"]["path"])
            failures = json.loads(failures_path.read_text(encoding="utf-8"))
            case = failures["cases"]["corrupt"]
            input_path = Path(case["input"]["path"])
            wrong_payload = bytearray(Path(value["network"]["path"]).read_bytes())
            wrong_payload[2] ^= 0x01
            input_path.write_bytes(wrong_payload)
            case["input"]["sha256"] = sha256_file(input_path)

            descriptor_path = Path(case["input_descriptor"]["path"])
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            descriptor["input_sha256"] = case["input"]["sha256"]
            descriptor_path.write_bytes(canonical_json_bytes(descriptor))
            case["input_descriptor"]["sha256"] = sha256_file(descriptor_path)

            command_path = Path(case["command"]["path"])
            command = json.loads(command_path.read_text(encoding="utf-8"))
            command["input_descriptor_sha256"] = case["input_descriptor"]["sha256"]
            command["input_sha256"] = case["input"]["sha256"]
            command_path.write_bytes(canonical_json_bytes(command))
            case["command"]["sha256"] = sha256_file(command_path)

            output_path = Path(case["output"]["path"])
            output = json.loads(output_path.read_text(encoding="utf-8"))
            output["command_sha256"] = case["command"]["sha256"]
            output_path.write_bytes(canonical_json_bytes(output))
            case["output"]["sha256"] = sha256_file(output_path)

            failures_path.write_bytes(canonical_json_bytes(failures))
            first["load_failures"]["sha256"] = sha256_file(failures_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(
                alice_release_evidence, "EXPECTED_NATIVE_SIZE", size
            ):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "does not match the frozen deterministic-byte-flip recipe" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_structural_network_claim_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            qualification_path = Path(value["native_qualification"]["path"])
            qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
            qualification["network_parameter_nonzero_bytes"] = 0
            qualification_path.write_text(json.dumps(qualification), encoding="utf-8")
            value["native_qualification"]["sha256"] = sha256_file(qualification_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(alice_release_evidence, "EXPECTED_NATIVE_SIZE", size):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "no verified nonzero parameters" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_gate_report_numeric_fields_reject_booleans(self) -> None:
        cases = (("sample_count", True), ("mismatch_count", False))
        for field, replacement in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, value, size = self.build_candidate(root)
                qualification_path = Path(value["native_qualification"]["path"])
                qualification = json.loads(
                    qualification_path.read_text(encoding="utf-8")
                )
                report_reference = qualification["gates"]["G3"]
                report_path = Path(report_reference["path"])
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report[field] = replacement
                report_path.write_bytes(canonical_json_bytes(report))
                report_reference["sha256"] = sha256_file(report_path)
                qualification_path.write_bytes(canonical_json_bytes(qualification))
                value["native_qualification"]["sha256"] = sha256_file(
                    qualification_path
                )
                manifest.write_text(json.dumps(value), encoding="utf-8")
                with mock.patch.object(
                    alice_release_evidence, "EXPECTED_NATIVE_SIZE", size
                ):
                    receipt = alice_release_evidence.audit_release_candidate(manifest)
            self.assertFalse(receipt["strength_release_authorized"])
            self.assertTrue(
                any(
                    "G3 report did not pass exactly" in reason
                    for reason in receipt["blocking_reasons"]
                )
            )

    def test_g6_requires_the_complete_parity_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            qualification_path = Path(value["native_qualification"]["path"])
            qualification = json.loads(
                qualification_path.read_text(encoding="utf-8")
            )
            report_reference = qualification["gates"]["G6"]
            report_path = Path(report_reference["path"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["sample_count"] = 1
            report_path.write_bytes(canonical_json_bytes(report))
            report_reference["sha256"] = sha256_file(report_path)
            qualification_path.write_bytes(canonical_json_bytes(qualification))
            value["native_qualification"]["sha256"] = sha256_file(
                qualification_path
            )
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(
                alice_release_evidence, "EXPECTED_NATIVE_SIZE", size
            ):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "G6 report did not pass exactly" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_qualification_artifact_hashes_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            qualification = json.loads(
                Path(value["native_qualification"]["path"]).read_text(encoding="utf-8")
            )
            checkpoint_path = Path(qualification["checkpoint"]["path"])
            checkpoint_path.write_bytes(b"tampered checkpoint\n")
            with mock.patch.object(
                alice_release_evidence, "EXPECTED_NATIVE_SIZE", size
            ):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "native qualification checkpoint: SHA-256 mismatch" in reason
                for reason in receipt["blocking_reasons"]
            )
        )

    def test_partial_native_parameter_count_blocks_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            qualification_path = Path(value["native_qualification"]["path"])
            qualification = json.loads(
                qualification_path.read_text(encoding="utf-8")
            )
            export_reference = qualification["export_receipt"]
            export_path = Path(export_reference["path"])
            export = json.loads(export_path.read_text(encoding="utf-8"))
            export["element_count"] = 1
            export_path.write_bytes(canonical_json_bytes(export))
            export_reference["sha256"] = sha256_file(export_path)
            for gate, report_reference in qualification["gates"].items():
                report_path = Path(report_reference["path"])
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["export_receipt_sha256"] = export_reference["sha256"]
                if gate in {"G4", "G5"}:
                    report["sample_count"] = 1
                report_path.write_bytes(canonical_json_bytes(report))
                report_reference["sha256"] = sha256_file(report_path)
            qualification_path.write_bytes(canonical_json_bytes(qualification))
            value["native_qualification"]["sha256"] = sha256_file(
                qualification_path
            )
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(
                alice_release_evidence, "EXPECTED_NATIVE_SIZE", size
            ):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "export receipt is inconsistent" in reason
                for reason in receipt["blocking_reasons"]
            )
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

    def test_openbench_shadow_counters_require_exact_integer_types(self) -> None:
        cases = (
            ("pairs", 200.0),
            ("inversions", False),
            ("invalid_pairs", False),
        )
        for field, replacement in cases:
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                manifest, value, size = self.build_candidate(root)
                shadow_path = Path(value["openbench_shadow_receipt"]["path"])
                shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
                shadow["presets"]["VSTC"][field] = replacement
                shadow_path.write_bytes(canonical_json_bytes(shadow))
                value["openbench_shadow_receipt"]["sha256"] = sha256_file(
                    shadow_path
                )
                manifest.write_text(json.dumps(value), encoding="utf-8")
                with mock.patch.object(
                    alice_release_evidence, "EXPECTED_NATIVE_SIZE", size
                ):
                    receipt = alice_release_evidence.audit_release_candidate(manifest)
            self.assertFalse(receipt["strength_release_authorized"])
            self.assertTrue(
                any(
                    "OpenBench shadow preset VSTC is not clean" in reason
                    for reason in receipt["blocking_reasons"]
                )
            )

    def test_openbench_shadow_configuration_is_recomputed_and_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, value, size = self.build_candidate(root)
            shadow_path = Path(value["openbench_shadow_receipt"]["path"])
            shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
            configuration_reference = shadow["presets"]["VSTC"]["configuration"]
            configuration_path = Path(configuration_reference["path"])
            configuration = json.loads(
                configuration_path.read_text(encoding="utf-8")
            )
            configuration["timing"]["base_ms"] = 2_001
            configuration_path.write_bytes(canonical_json_bytes(configuration))
            configuration_reference["sha256"] = sha256_file(configuration_path)
            shadow_path.write_bytes(canonical_json_bytes(shadow))
            value["openbench_shadow_receipt"]["sha256"] = sha256_file(shadow_path)
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(
                alice_release_evidence, "EXPECTED_NATIVE_SIZE", size
            ):
                receipt = alice_release_evidence.audit_release_candidate(manifest)
        self.assertFalse(receipt["strength_release_authorized"])
        self.assertTrue(
            any(
                "VSTC configuration does not match the frozen preset" in reason
                for reason in receipt["blocking_reasons"]
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
