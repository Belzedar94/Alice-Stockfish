"""Fail-closed gates for authenticating the NNUE-v2 run-config preimage."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN = load_module("alice_v2_run_config_for_binding", ROOT / "tools" / "alice_v2_run_config.py")
CHUNK = load_module("alice_v2_chunk_for_binding", ROOT / "tools" / "alice_v2_chunk.py")


class RunConfigBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RUN.build_config(
            "1234567890abcdef1234567890abcdef12345678",
            "alice-v2-native-50m-20260811",
            202608110000000,
        )

    def write(self, directory: Path, config: dict[str, object]) -> Path:
        path = directory / "run-config.json"
        path.write_bytes(RUN.canonical_bytes(config))
        return path

    def test_exact_canonical_preimage_is_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self.write(Path(temp), self.config)
            loaded, digest = CHUNK._load_run_config(path)
        self.assertEqual(loaded, self.config)
        self.assertEqual(
            digest,
            hashlib.sha256(RUN.canonical_bytes(self.config)).hexdigest().upper(),
        )

    def test_extra_fields_are_rejected_even_when_canonical(self) -> None:
        self.config["unregistered"] = True
        with tempfile.TemporaryDirectory() as temp:
            path = self.write(Path(temp), self.config)
            with self.assertRaisesRegex(CHUNK.AliceV2ChunkError, "field membership"):
                CHUNK._load_run_config(path)

    def test_openbench_identity_drift_is_rejected(self) -> None:
        self.config["openbench"] = dict(self.config["openbench"])
        self.config["openbench"]["priority"] = 299
        with tempfile.TemporaryDirectory() as temp:
            path = self.write(Path(temp), self.config)
            with self.assertRaisesRegex(CHUNK.AliceV2ChunkError, "OpenBench identity"):
                CHUNK._load_run_config(path)

    def test_acceptance_gate_drift_is_rejected(self) -> None:
        self.config["acceptance"] = dict(self.config["acceptance"])
        self.config["acceptance"][
            "records_with_same_and_other_for_both_perspectives_min_ppm"
        ] = 0
        with tempfile.TemporaryDirectory() as temp:
            path = self.write(Path(temp), self.config)
            with self.assertRaisesRegex(CHUNK.AliceV2ChunkError, "acceptance gates"):
                CHUNK._load_run_config(path)

    def test_pretty_printed_preimage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run-config.json"
            path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(CHUNK.AliceV2ChunkError, "canonical"):
                CHUNK._load_run_config(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
