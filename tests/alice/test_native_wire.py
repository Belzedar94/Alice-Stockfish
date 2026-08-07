"""Transactional wire-format qualification for AliceNative-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from native_wire import (
    ARCHITECTURE_HASH,
    DENSE_ARCHITECTURE_HASH,
    FEATURE_TENSOR_BYTES,
    FEATURE_TRANSFORMER_HASH,
    LEGACY_WIRE_VERSION,
    MANIFEST,
    MANIFEST_SHA256,
    NATIVE_WIRE_BYTES,
    WIRE_VERSION,
    assert_zero_tensors,
    file_sha256,
    inspect_wire,
    write_zero_wire,
)


TEST_DIRECTORY = Path(__file__).resolve().parent


def default_engine_path() -> Path:
    repository = TEST_DIRECTORY.parent.parent
    windows = repository / "src" / "stockfish.exe"
    return windows if windows.exists() else repository / "src" / "stockfish"


ENGINE_PATH = default_engine_path()


def command_path(path: Path) -> str:
    return json.dumps(path.resolve().as_posix())


def run_engine(*commands: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ENGINE_PATH)],
        input="\n".join((*commands, "quit", "")),
        text=True,
        capture_output=True,
        encoding="ascii",
        check=False,
    )


def validate(path: Path, expected_sha256: str | None = None) -> subprocess.CompletedProcess[str]:
    expected = f" {expected_sha256}" if expected_sha256 else ""
    return run_engine(f"alice_native_validate_file {command_path(path)}{expected}")


def tensor_offset() -> int:
    return 12 + len(MANIFEST) + 4


def mutate_i16(path: Path, byte_offset: int, value: int) -> None:
    with path.open("r+b") as output:
        output.seek(byte_offset)
        output.write(struct.pack("<h", value))


def mutate_i32(path: Path, byte_offset: int, value: int) -> None:
    with path.open("r+b") as output:
        output.seek(byte_offset)
        output.write(struct.pack("<i", value))


def section_sha256(path: Path, byte_offset: int, byte_count: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        source.seek(byte_offset)
        remaining = byte_count
        while remaining:
            chunk = source.read(min(1 << 20, remaining))
            if not chunk:
                raise AssertionError("Tensor section is truncated.")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest().upper()


class NativeWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not ENGINE_PATH.is_file():
            raise FileNotFoundError(f"Alice-Stockfish executable not found: {ENGINE_PATH}")

    def test_zero_integer_export_is_reproducible_and_fully_validated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="alice-native-wire-") as temporary:
            directory = Path(temporary)
            first = directory / "alice-native-v1-zero-a.nnue"
            second = directory / "alice-native-v1-zero-b.nnue"
            write_zero_wire(first)
            write_zero_wire(second)

            self.assertEqual(first.stat().st_size, NATIVE_WIRE_BYTES)
            self.assertEqual(second.stat().st_size, NATIVE_WIRE_BYTES)
            observed = inspect_wire(first)
            self.assertEqual(observed["version"], WIRE_VERSION)
            self.assertEqual(observed["architecture"], ARCHITECTURE_HASH)
            self.assertEqual(observed["manifest"], MANIFEST)
            self.assertEqual(observed["manifestSha256"], MANIFEST_SHA256)
            self.assertEqual(observed["transformerHash"], FEATURE_TRANSFORMER_HASH)
            self.assertEqual(observed["denseHashes"], [DENSE_ARCHITECTURE_HASH] * 8)
            self.assertEqual(observed["bytes"], NATIVE_WIRE_BYTES)
            assert_zero_tensors(first)

            first_sha = file_sha256(first)
            second_sha = file_sha256(second)
            self.assertEqual(first_sha, second_sha)

            result = validate(first, first_sha)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f"bytes={NATIVE_WIRE_BYTES}", result.stdout)
            self.assertIn(f"sha256={first_sha}", result.stdout)
            self.assertIn(f"manifest_sha256={MANIFEST_SHA256}", result.stdout)
            self.assertIn("version=0xA11CE001", result.stdout)
            self.assertIn("architecture=0xEC7CCD50", result.stdout)

    def test_malformed_and_incompatible_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="alice-native-wire-negative-") as temporary:
            directory = Path(temporary)
            cases: list[tuple[str, Path, str, str | None]] = []

            wrong_version = directory / "wrong-version.nnue"
            write_zero_wire(wrong_version, version=0xA11CE002)
            cases.append(("wrong-version", wrong_version, "wire version mismatch", None))

            legacy = directory / "legacy.nnue"
            write_zero_wire(legacy, version=LEGACY_WIRE_VERSION)
            cases.append(("legacy", legacy, "not native Alice networks", None))

            wrong_architecture = directory / "wrong-architecture.nnue"
            write_zero_wire(wrong_architecture, architecture=ARCHITECTURE_HASH ^ 1)
            cases.append(("wrong-architecture", wrong_architecture, "architecture mismatch", None))

            oversized_manifest = directory / "oversized-manifest.nnue"
            oversized_manifest.write_bytes(
                struct.pack("<III", WIRE_VERSION, ARCHITECTURE_HASH, 65_537)
            )
            cases.append(("oversized-manifest", oversized_manifest, "65536-byte limit", None))

            changed_manifest = bytearray(MANIFEST)
            changed_manifest[0] ^= 1
            wrong_manifest = directory / "wrong-manifest.nnue"
            write_zero_wire(wrong_manifest, manifest=bytes(changed_manifest))
            cases.append(("wrong-manifest", wrong_manifest, "manifest SHA-256 mismatch", None))

            wrong_transformer = directory / "wrong-transformer.nnue"
            write_zero_wire(wrong_transformer, transformer_hash=FEATURE_TRANSFORMER_HASH ^ 1)
            cases.append(("wrong-transformer", wrong_transformer, "feature-transformer hash mismatch", None))

            dense_hashes = [DENSE_ARCHITECTURE_HASH] * 8
            dense_hashes[3] ^= 1
            wrong_dense = directory / "wrong-dense.nnue"
            write_zero_wire(wrong_dense, dense_hashes=dense_hashes)
            cases.append(("wrong-dense", wrong_dense, "dense-stack hash mismatch at stack 3", None))

            truncated = directory / "truncated.nnue"
            write_zero_wire(truncated)
            with truncated.open("r+b") as output:
                output.truncate(NATIVE_WIRE_BYTES - 1)
            cases.append(("truncated", truncated, "file is truncated", None))

            trailing = directory / "trailing.nnue"
            write_zero_wire(trailing)
            with trailing.open("ab") as output:
                output.write(b"\0")
            cases.append(("trailing", trailing, "trailing data", None))

            sealed = directory / "sealed.nnue"
            write_zero_wire(sealed)
            sealed_sha = file_sha256(sealed)
            changed_tensor = directory / "changed-tensor.nnue"
            write_zero_wire(changed_tensor, tensor_mutation=(123_456, 1))
            cases.append(("changed-tensor", changed_tensor, "wire SHA-256 mismatch", sealed_sha))

            for name, path, expected_error, expected_sha in cases:
                with self.subTest(case=name):
                    result = validate(path, expected_sha)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("CRITICAL ERROR", result.stdout)
                    self.assertIn(expected_error, result.stdout)

    def test_failed_replacement_clears_the_previous_validation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="alice-native-wire-replace-") as temporary:
            directory = Path(temporary)
            valid = directory / "valid.nnue"
            invalid = directory / "invalid.nnue"
            write_zero_wire(valid)
            write_zero_wire(invalid, version=0xA11CE002)
            valid_sha = file_sha256(valid)

            result = run_engine(
                f"alice_native_try_validate_file {command_path(valid)} {valid_sha}",
                f"alice_native_try_validate_file {command_path(invalid)}",
                "alice_native_wire_status",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            lines = [line for line in result.stdout.splitlines() if "Alice native wire" in line]
            self.assertGreaterEqual(len(lines), 3, result.stdout)
            self.assertIn(valid_sha, lines[0])
            self.assertIn("rejected", lines[1])
            self.assertIn("is not validated", lines[-1])
            self.assertNotIn(valid_sha, lines[-1])

    def test_qualification_loader_commits_once_and_preserves_active_on_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="alice-native-load-") as temporary:
            directory = Path(temporary)
            zero = directory / "zero.nnue"
            sentinel = directory / "sentinel.nnue"
            invalid = directory / "invalid.nnue"
            write_zero_wire(zero)
            write_zero_wire(sentinel)
            mutate_i16(sentinel, tensor_offset(), 52)
            write_zero_wire(invalid, version=WIRE_VERSION ^ 1)

            zero_sha = file_sha256(zero)
            sentinel_sha = file_sha256(sentinel)
            invalid_sha = file_sha256(invalid)
            sentinel_ft_sha = section_sha256(sentinel, tensor_offset(), 2 * 1_024)

            result = run_engine(
                f"alice_native_try_load_file {command_path(zero)} {zero_sha}",
                "alice_native_load_status",
                "alice_native_parameter ft.bias 0",
                f"alice_native_try_load_file {command_path(sentinel)} {sentinel_sha}",
                "alice_native_load_status",
                "alice_native_tensor_status",
                "alice_native_parameter ft.bias 0",
                f"alice_native_try_load_file {command_path(invalid)} {invalid_sha}",
                "alice_native_load_status",
                "alice_native_tensor_status",
                "alice_native_parameter ft.bias 0",
                f"alice_native_try_load_file {command_path(sentinel)}",
                "alice_native_load_status",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            statuses = [
                line
                for line in result.stdout.splitlines()
                if line.startswith("Alice native qualification parameters loaded")
            ]
            self.assertEqual(len(statuses), 6, result.stdout)
            self.assertIn("generation=1", statuses[0])
            self.assertEqual(statuses[0], statuses[1])
            self.assertIn("generation=2", statuses[2])
            self.assertEqual(statuses[2], statuses[3])
            self.assertEqual(statuses[2], statuses[4])
            self.assertEqual(statuses[2], statuses[5])
            self.assertIn(f"sha256={sentinel_sha}", statuses[2])
            self.assertIn("search=available", statuses[2])

            tensor_statuses = [
                line
                for line in result.stdout.splitlines()
                if line.startswith("Alice native qualification tensors")
            ]
            self.assertEqual(len(tensor_statuses), 2, result.stdout)
            self.assertEqual(tensor_statuses[0], tensor_statuses[1])
            self.assertIn(f"ft.bias_sha256={sentinel_ft_sha}", tensor_statuses[0])

            probes = [
                line for line in result.stdout.splitlines() if line.startswith("alice_native_parameter")
            ]
            self.assertEqual(len(probes), 3, result.stdout)
            self.assertTrue(probes[0].endswith("value 0"), probes[0])
            self.assertTrue(probes[1].endswith("value 52"), probes[1])
            self.assertEqual(probes[1], probes[2])

            self.assertIn("version mismatch", result.stdout)
            self.assertIn("requires a path and an expected SHA-256", result.stdout)

    def test_qualification_loader_rejects_hash_ranges_and_dense_envelopes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="alice-native-load-negative-") as temporary:
            directory = Path(temporary)
            valid = directory / "valid.nnue"
            forbidden_i16 = directory / "forbidden-i16.nnue"
            dense_i16_overflow = directory / "dense-i16-overflow.nnue"
            write_zero_wire(valid)
            write_zero_wire(forbidden_i16)
            mutate_i16(forbidden_i16, tensor_offset(), -32_768)
            write_zero_wire(dense_i16_overflow)
            first_dense_fc0_bias = tensor_offset() + FEATURE_TENSOR_BYTES + 4
            mutate_i32(dense_i16_overflow, first_dense_fc0_bias, 32_768)

            result = run_engine(
                f"alice_native_try_load_file {command_path(valid)} {'0' * 64}",
                "alice_native_load_status",
                f"alice_native_try_load_file {command_path(forbidden_i16)} {file_sha256(forbidden_i16)}",
                "alice_native_load_status",
                f"alice_native_try_load_file {command_path(dense_i16_overflow)} {file_sha256(dense_i16_overflow)}",
                "alice_native_load_status",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("SHA-256 mismatch", result.stdout)
            self.assertIn("forbidden -32768", result.stdout)
            self.assertIn("fc0 affine envelope exceeds signed i16", result.stdout)
            not_loaded = [
                line
                for line in result.stdout.splitlines()
                if line.startswith("Alice native qualification parameters are not loaded")
            ]
            self.assertEqual(len(not_loaded), 3, result.stdout)
            self.assertNotIn("parameters loaded generation=", result.stdout)


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine", type=Path, default=default_engine_path())
    return parser.parse_known_args()


if __name__ == "__main__":
    arguments, unittest_arguments = parse_arguments()
    ENGINE_PATH = arguments.engine.resolve()
    unittest.main(argv=[sys.argv[0], *unittest_arguments], verbosity=2)
