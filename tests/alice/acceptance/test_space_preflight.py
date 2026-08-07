from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
from unittest import mock
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.alice_training_space_preflight import MINIMUM_AVAILABLE_BYTES, evaluate_space, main


class TrainingSpacePreflightTests(unittest.TestCase):
    def test_exact_threshold_passes_and_one_byte_less_fails(self) -> None:
        exact = evaluate_space("test-volume", query=lambda _target: MINIMUM_AVAILABLE_BYTES)
        below = evaluate_space("test-volume", query=lambda _target: MINIMUM_AVAILABLE_BYTES - 1)
        self.assertTrue(exact["passed"])
        self.assertFalse(below["passed"])
        self.assertFalse(exact["training_started"])

    def test_query_is_called_exactly_once(self) -> None:
        query = mock.Mock(return_value=MINIMUM_AVAILABLE_BYTES)
        evaluate_space("test-volume", query=query)
        query.assert_called_once_with("test-volume")

    def test_no_enumeration_or_mutation_api_is_used(self) -> None:
        forbidden = {
            "listdir": mock.Mock(side_effect=AssertionError("enumeration attempted")),
            "scandir": mock.Mock(side_effect=AssertionError("enumeration attempted")),
            "remove": mock.Mock(side_effect=AssertionError("remove attempted")),
            "rename": mock.Mock(side_effect=AssertionError("rename attempted")),
            "mkdir": mock.Mock(side_effect=AssertionError("mkdir attempted")),
            "makedirs": mock.Mock(side_effect=AssertionError("makedirs attempted")),
        }
        with mock.patch.multiple("os", **forbidden):
            result = evaluate_space("test-volume", query=lambda _target: MINIMUM_AVAILABLE_BYTES)
        self.assertTrue(result["passed"])
        for function in forbidden.values():
            function.assert_not_called()

    def test_query_failure_fails_closed(self) -> None:
        def fail(_target: str) -> int:
            raise OSError("unavailable")

        with self.assertRaises(OSError):
            evaluate_space("test-volume", query=fail)

    def test_target_is_mandatory(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main([])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
