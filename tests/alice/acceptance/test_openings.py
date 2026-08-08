from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.alice_acceptance.openings import OpeningSchedule


class OpeningScheduleTests(unittest.TestCase):
    def test_schedule_is_stable_and_each_cycle_is_a_permutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            book = Path(directory) / "book.epd"
            payload = b"fen-a\nfen-b\nfen-c\nfen-d\n"
            book.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            first = OpeningSchedule(book, digest, 42)
            second = OpeningSchedule(book, digest, 42)
            a = [first.for_ordinal(index) for index in range(8)]
            b = [second.for_ordinal(index) for index in range(8)]
        self.assertEqual(a, b)
        self.assertEqual({item["book_line"] for item in a[:4]}, {1, 2, 3, 4})
        self.assertEqual({item["book_line"] for item in a[4:]}, {1, 2, 3, 4})
        self.assertEqual([item["attempt_ordinal"] for item in a], list(range(8)))
        self.assertTrue(all(item["seed"] == 42 for item in a))

    def test_book_identity_encoding_and_seed_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            book = Path(directory) / "book.epd"
            book.write_bytes(b"fen-a\n")
            digest = hashlib.sha256(book.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                OpeningSchedule(book, "0" * 64, 1)
            with self.assertRaises(ValueError):
                OpeningSchedule(book, digest, -1)

            book.write_bytes(b"\xef\xbb\xbffen-a\n")
            bom_digest = hashlib.sha256(book.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                OpeningSchedule(book, bom_digest, 1)

            book.write_bytes(b"\xff\n")
            invalid_digest = hashlib.sha256(book.read_bytes()).hexdigest()
            with self.assertRaises(UnicodeDecodeError):
                OpeningSchedule(book, invalid_digest, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
