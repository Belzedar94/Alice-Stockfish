"""Unit gates for binding the completed v41 manifest to the run preimage."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN = load_module("alice_v2_run_config_for_corpus", TOOLS / "alice_v2_run_config.py")
CORPUS = load_module("alice_v2_corpus_for_test", TOOLS / "alice_v2_corpus.py")


class CorpusContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = "1234567890abcdef1234567890abcdef12345678"
        self.config = RUN.build_config(
            self.source, "alice-v2-native-50m-20260811", 202608110000000
        )
        self.run_digest = hashlib.sha256(
            RUN.canonical_bytes(self.config)
        ).hexdigest().upper()
        command = RUN.command_template(self.run_digest, self.config["base_seed"])
        self.contract = {
            "schema": CORPUS.PUBLICATION_CONTRACT_SCHEMA,
            "protocol": 41,
            "campaign_id": self.config["campaign"],
            "external_workload_id": "native-v2-selfplay-50m",
            "role": "selfplay-all-splits",
            "cohort": "piece-rel-d6",
            "engine": {
                "name": "Alice-Stockfish",
                "repo": "https://github.com/Belzedar94/Alice-Stockfish",
                "source": "https://github.com/Belzedar94/Alice-Stockfish",
                "requested_ref": self.source,
                "commit": self.source,
                "bench": 202963,
                "options": "",
            },
            "network": {
                "name": "alice_run2rl_e40_l09.nnue",
                "openbench_id": "9F9E5570",
                "sha256": self.config["network_sha256"].lower(),
                "bytes": 47_721_376,
            },
            "book": {
                "kind": "file",
                "name": CORPUS.BOOK_NAME,
                "source": CORPUS.BOOK_SOURCE,
                "text_sha256": self.config["book_sha256"].lower(),
                "raw_sha256": self.config["book_sha256"].lower(),
            },
            "generation": {
                "command": command,
                "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "total_count": 50_000_000,
                "positions_per_chunk": 1_000_000,
                "base_seed": self.config["base_seed"],
                "seed_method": "base-plus-chunk-index-v1",
            },
            "producer": {"required": True, "contract_sha256": "b" * 64},
            "teacher": {"mode": None},
            "syzygy": {
                "required": False,
                "family": None,
                "max": 0,
                "manifest_sha256": None,
                "environment_contract_sha256": "c" * 64,
            },
        }

    def manifest(self) -> dict[str, object]:
        contract_sha = CORPUS.canonical_sha256(self.contract)
        document = {
            "schema": CORPUS.PUBLICATION_SCHEMA,
            "version": 1,
            "protocol": 41,
            "publication_contract": self.contract,
            "publication_contract_sha256": contract_sha,
        }
        document["manifest_sha256"] = CORPUS.canonical_sha256(document)
        return document

    def test_exact_publication_contract_passes(self) -> None:
        document = self.manifest()
        observed = CORPUS.validate_publication_contract(
            document, self.config, self.run_digest
        )
        self.assertEqual(observed, self.contract)

    def test_command_preimage_drift_fails(self) -> None:
        self.contract["generation"] = dict(self.contract["generation"])
        self.contract["generation"]["command"] += " drift"
        document = self.manifest()
        with self.assertRaisesRegex(CORPUS.AliceV2CorpusError, "command differs"):
            CORPUS.validate_publication_contract(document, self.config, self.run_digest)

    def test_server_book_name_drift_fails(self) -> None:
        self.contract["book"] = dict(self.contract["book"])
        self.contract["book"]["name"] = "alice.epd"
        document = self.manifest()
        with self.assertRaisesRegex(CORPUS.AliceV2CorpusError, "book mismatch"):
            CORPUS.validate_publication_contract(document, self.config, self.run_digest)

    def test_unregistered_contract_field_fails(self) -> None:
        self.contract["unregistered"] = True
        document = self.manifest()
        with self.assertRaisesRegex(CORPUS.AliceV2CorpusError, "field set"):
            CORPUS.validate_publication_contract(document, self.config, self.run_digest)

    def test_manifest_self_hash_drift_fails(self) -> None:
        document = self.manifest()
        document["version"] = 2
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(CORPUS.AliceV2CorpusError, "self-hash"):
                CORPUS.load_manifest(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
