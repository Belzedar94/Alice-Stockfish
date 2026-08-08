from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.alice_acceptance.evidence import CreateOnlySeal, sha256_file


class EvidenceTests(unittest.TestCase):
    def test_seal_is_canonical_create_only_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seal.json"
            seal = CreateOnlySeal(path)
            seal({"z": 1, "a": "value"})
            before = path.read_bytes()
            self.assertEqual(before, b'{"a":"value","z":1}\n')
            self.assertEqual(seal.sha256, sha256_file(path))
            self.assertEqual(json.loads(before), {"a": "value", "z": 1})
            with self.assertRaises(FileExistsError):
                seal({"z": 2})
            self.assertEqual(path.read_bytes(), before)

    def test_existing_path_cannot_be_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seal.json"
            path.write_text("existing\n", encoding="ascii")
            with self.assertRaises(FileExistsError):
                CreateOnlySeal(path)({"new": True})
            self.assertEqual(path.read_text(encoding="ascii"), "existing\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
