from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.alice_acceptance.runner_adapter import parse_strict_json


def load_contract(path: str) -> dict[str, object]:
    return parse_strict_json((ROOT / path).read_bytes())


class ContractFileTests(unittest.TestCase):
    def test_policy_and_schema_files_are_valid_and_frozen(self) -> None:
        local = load_contract("config/alice-local-los-v1.json")
        fixed = load_contract("config/alice-final-gate-v1.json")
        pair_schema = load_contract("schemas/alice-pair-result-v1.schema.json")
        receipt_schema = load_contract("schemas/alice-acceptance-receipt-v1.schema.json")
        run_schema = load_contract(
            "schemas/alice-acceptance-run-definition-v1.schema.json"
        )
        control_schema = load_contract("schemas/alice-control-receipt-v1.schema.json")
        release_candidate_schema = load_contract(
            "schemas/alice-release-candidate-v1.schema.json"
        )
        release_evidence_schema = load_contract(
            "schemas/alice-release-evidence-v1.schema.json"
        )
        native_qualification_schema = load_contract(
            "schemas/alice-native-qualification-v1.schema.json"
        )
        dataset_manifest_schema = load_contract(
            "schemas/alice-training-dataset-manifest-v1.schema.json"
        )
        native_export_schema = load_contract(
            "schemas/alice-native-export-receipt-v1.schema.json"
        )
        native_gate_schema = load_contract(
            "schemas/alice-native-gate-report-v1.schema.json"
        )
        openbench_shadow_schema = load_contract(
            "schemas/alice-openbench-shadow-receipt-v1.schema.json"
        )
        openbench_shadow_configuration_schema = load_contract(
            "schemas/alice-openbench-shadow-configuration-v1.schema.json"
        )
        load_failure_schema = load_contract(
            "schemas/alice-load-failure-matrix-v1.schema.json"
        )
        triple_bench_schema = load_contract(
            "schemas/alice-triple-bench-v1.schema.json"
        )
        worker_schema = load_contract(
            "schemas/alice-pair-worker-definition-v1.schema.json"
        )
        request_schema = load_contract("schemas/alice-pair-request-v1.schema.json")
        response_schema = load_contract(
            "schemas/alice-pair-worker-response-v1.schema.json"
        )
        self.assertEqual(local["maximum_scored_games"], 64000)
        self.assertEqual(local["maximum_attempted_games"], 64000)
        self.assertEqual(local["pair_workers"], 2)
        self.assertEqual(local["hash_mib"], 512)
        self.assertEqual(fixed["presets"]["VSTC"]["target_admitted_games"], 400)
        self.assertEqual(fixed["presets"]["STC"]["target_admitted_games"], 300)
        self.assertEqual(fixed["presets"]["LTC"]["target_admitted_games"], 200)
        self.assertEqual(pair_schema["properties"]["schema"]["const"], "alice-pair-result-v1")
        self.assertIn("games", pair_schema["required"])
        self.assertEqual(
            run_schema["properties"]["schema"]["const"],
            "alice-acceptance-run-definition-v1",
        )
        self.assertEqual(
            control_schema["properties"]["schema"]["const"],
            "alice-control-receipt-v1",
        )
        self.assertEqual(
            set(control_schema["properties"]["artifacts"]["required"]),
            {"openings_jsonl_sha256", "status_jsonl_sha256"},
        )
        self.assertEqual(
            release_candidate_schema["properties"]["schema"]["const"],
            "alice-release-candidate-v1",
        )
        self.assertEqual(
            release_evidence_schema["properties"]["schema"]["const"],
            "alice-release-evidence-v1",
        )
        self.assertEqual(
            native_qualification_schema["properties"]["schema"]["const"],
            "alice-native-qualification-v1",
        )
        self.assertEqual(
            dataset_manifest_schema["properties"]["schema"]["const"],
            "alice-training-dataset-manifest-v1",
        )
        self.assertEqual(
            native_export_schema["properties"]["schema"]["const"],
            "alice-native-export-receipt-v1",
        )
        self.assertEqual(
            native_export_schema["properties"]["element_count"]["const"],
            170_222_600,
        )
        self.assertEqual(
            native_gate_schema["properties"]["schema"]["const"],
            "alice-native-gate-report-v1",
        )
        self.assertEqual(
            openbench_shadow_schema["properties"]["schema"]["const"],
            "alice-openbench-shadow-receipt-v1",
        )
        self.assertEqual(
            openbench_shadow_configuration_schema["properties"]["schema"]["const"],
            "alice-openbench-shadow-configuration-v1",
        )
        self.assertEqual(
            openbench_shadow_configuration_schema["properties"]["book_token"]["const"],
            "ALICE",
        )
        self.assertEqual(
            load_failure_schema["properties"]["schema"]["const"],
            "alice-load-failure-matrix-v1",
        )
        self.assertEqual(
            triple_bench_schema["properties"]["schema"]["const"],
            "alice-triple-bench-v1",
        )
        self.assertEqual(
            triple_bench_schema["properties"]["runs"]["minItems"], 3
        )
        self.assertEqual(
            worker_schema["properties"]["schema"]["const"],
            "alice-pair-worker-definition-v1",
        )
        worker_options = worker_schema["$defs"]["engine"]["properties"]["options"]
        self.assertEqual(
            worker_options["properties"]["Move Overhead"]["const"], "10"
        )
        self.assertIn("Move Overhead", worker_options["required"])
        self.assertEqual(
            request_schema["properties"]["schema"]["const"],
            "alice-pair-request-v1",
        )
        self.assertEqual(
            response_schema["properties"]["schema"]["const"],
            "alice-pair-worker-response-v1",
        )
        self.assertEqual(
            receipt_schema["properties"]["schema"]["const"],
            "alice-acceptance-receipt-v1",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
