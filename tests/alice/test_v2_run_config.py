"""Qualification gates for the preregistered 50M Alice V2 run contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "alice_v2_run_config.py"
SPEC = importlib.util.spec_from_file_location("alice_v2_run_config", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {TOOL_PATH}")
RUN_CONFIG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUN_CONFIG
SPEC.loader.exec_module(RUN_CONFIG)


class V2RunConfigTests(unittest.TestCase):
    def test_schema_hash_is_pinned_to_exact_bytes(self) -> None:
        schema = ROOT / "schemas" / "alice-v2-datagen-run-v1.schema.json"
        observed = hashlib.sha256(schema.read_bytes()).hexdigest().upper()
        self.assertEqual(observed, RUN_CONFIG.RUN_SCHEMA_SHA256)

    def test_fifty_million_partition_and_publication_are_frozen(self) -> None:
        config = RUN_CONFIG.build_config(
            "1234567890abcdef1234567890abcdef12345678", "qualification", 1000
        )
        self.assertEqual(config["total_records"], 50_000_000)
        self.assertEqual(config["records_per_chunk"], 1_000_000)
        self.assertEqual(config["total_chunks"], 50)
        self.assertEqual(config["generation"]["opening_count"], 38_348)
        self.assertEqual(config["split"], {
            "train_chunks": 48,
            "validation_chunks": 1,
            "test_chunks": 1,
        })
        self.assertEqual(config["openbench"], {
            "priority": 300,
            "throughput": 1000,
            "publication_protocol": 41,
            "producer_artifact_required": True,
            "workload_size": 1,
            "campaign_id": "qualification",
            "external_workload_id": "native-v2-selfplay-50m",
            "role": "selfplay-all-splits",
            "cohort": "piece-rel-d6",
        })
        self.assertEqual(
            config["retry"]["required_output"],
            "identical uncompressed bytes across workers",
        )
        self.assertEqual(config["acceptance"]["chunk_count"], 50)
        self.assertEqual(config["acceptance"]["record_count"], 50_000_000)
        self.assertEqual(
            config["acceptance"][
                "records_with_same_and_other_for_both_perspectives_min_ppm"
            ],
            1000,
        )
        self.assertEqual(config["acceptance"]["malformed_records_max"], 0)
        self.assertIn(
            "white_other_features",
            config["acceptance"]["minimum_nonzero_counters"],
        )

    def test_command_requires_all_protocol_41_custody_placeholders(self) -> None:
        command = RUN_CONFIG.command_template("B" * 64, 1000)
        for placeholder in (
            "{THREADS}",
            "{NETWORK}",
            "{NETWORK_SHA256}",
            "{PRODUCER_SHA256}",
            "{BOOK}",
            "{BOOK_SHA256}",
            "{OUT}",
            "{COUNT}",
            "{SEED}",
        ):
            self.assertEqual(command.count(placeholder), 1)
        self.assertIn("total_records 50000000", command)
        self.assertIn("records_per_chunk 1000000", command)

    def test_canonical_run_hash_is_deterministic(self) -> None:
        config = RUN_CONFIG.build_config(
            "1234567890abcdef1234567890abcdef12345678", "qualification", 1000
        )
        first = RUN_CONFIG.canonical_bytes(config)
        second = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(first, second)
        self.assertNotIn(b"\n", first)

    def test_campaign_identity_uses_the_v41_slug_domain(self) -> None:
        source = "1234567890abcdef1234567890abcdef12345678"
        for campaign in ("", "Uppercase", "contains space", "x" * 129):
            with self.subTest(campaign=campaign), self.assertRaises(ValueError):
                RUN_CONFIG.build_config(source, campaign, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
