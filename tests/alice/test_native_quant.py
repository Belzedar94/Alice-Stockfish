"""Normative contract and exact-rounding checks for AliceNative-v1 N7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import unittest


TEST_DIRECTORY = Path(__file__).resolve().parent
CONTRACT_DIRECTORY = TEST_DIRECTORY.parent.parent / "docs" / "alice"
EXPECTED = {
    "native-quant-v1.json": (
        1_194,
        "DD8571715CB7711BEE46785D0FBAC9F480ECCADD1D6CC9EF71D652554F80F9C8",
    ),
    "native-checkpoint-v1.json": (
        1_031,
        "A7E667BB5B7B978E474A392960CF6A72A5F1A9B074DDFC97C6FA13166B5D3413",
    ),
}


def float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def quantize_bits(bits: int, scale: int, maximum: int) -> int:
    sign = -1 if bits >> 31 else 1
    exponent_bits = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent_bits == 0xFF:
        raise ValueError("Non-finite binary32 value")
    if exponent_bits == 0:
        significand = fraction
        exponent = -149
    else:
        significand = (1 << 23) | fraction
        exponent = exponent_bits - 127 - 23
    if significand == 0:
        return 0

    numerator = significand * scale
    if exponent >= 0:
        numerator <<= exponent
        denominator = 1
    else:
        denominator = 1 << -exponent
    if numerator > maximum * denominator:
        raise ValueError("Value outside unsaturated domain")

    quotient, remainder = divmod(numerator, denominator)
    if remainder * 2 > denominator or (
        remainder * 2 == denominator and quotient % 2
    ):
        quotient += 1
    return sign * quotient


class NativeQuantContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contracts: dict[str, dict] = {}
        for name, (expected_bytes, expected_sha) in EXPECTED.items():
            raw = (CONTRACT_DIRECTORY / name).read_bytes()
            canonical = (
                json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":")).encode(
                    "ascii"
                )
                + b"\n"
            )
            if raw != canonical:
                raise AssertionError(f"{name} is not canonical JSON.")
            if len(raw) != expected_bytes:
                raise AssertionError(f"{name} byte count changed.")
            if hashlib.sha256(raw).hexdigest().upper() != expected_sha:
                raise AssertionError(f"{name} SHA-256 changed.")
            cls.contracts[name] = json.loads(raw)

    def test_checkpoint_inventory_and_identity(self) -> None:
        checkpoint = self.contracts["native-checkpoint-v1.json"]
        self.assertEqual(checkpoint["format"], "alice-native-checkpoint-v1")
        self.assertEqual(checkpoint["container"], "safetensors")
        self.assertEqual(checkpoint["dtype"], "float32")
        self.assertEqual(checkpoint["memoryOrder"], "C")
        self.assertEqual(len(checkpoint["tensors"]), 11)
        self.assertEqual(checkpoint["tensors"][0], {"name": "ft.bias", "shape": [1024]})
        self.assertEqual(
            checkpoint["tensors"][-1], {"name": "stack.fc2.weight", "shape": [8, 1, 128]}
        )

    def test_quantization_scales_and_symmetric_ranges(self) -> None:
        quant = self.contracts["native-quant-v1.json"]
        self.assertEqual(quant["rounding"], "exact-binary32-rne-even")
        self.assertEqual(quant["saturation"], "reject")
        self.assertEqual(quant["integerRanges"]["i8"], [-127, 127])
        self.assertEqual(quant["integerRanges"]["i16"], [-32767, 32767])
        self.assertEqual(quant["integerRanges"]["i32"], [-2147483647, 2147483647])
        self.assertEqual(quant["tensors"]["threat.weight"]["scale"], 256)
        self.assertEqual(quant["tensors"]["threat.psqt"]["scale"], 9600)
        self.assertEqual(quant["tensors"]["stack.fc0.weight"]["scale"], 128)
        self.assertEqual(quant["tensors"]["stack.fc1.weight"]["scale"], 64)

    def test_exact_binary32_rounding_is_nearest_even(self) -> None:
        power_two = [0.5 / 256, 1.5 / 256, 2.5 / 256, -0.5 / 256, -1.5 / 256]
        self.assertEqual(
            [quantize_bits(float32_bits(value), 256, 32767) for value in power_two],
            [0, 2, 2, 0, -2],
        )
        psqt = [1 / 256, 3 / 256, -1 / 256, -3 / 256]
        self.assertEqual(
            [quantize_bits(float32_bits(value), 9600, 2147483647) for value in psqt],
            [38, 112, -38, -112],
        )

    def test_nonfinite_and_forbidden_signed_minimum_are_rejected(self) -> None:
        for bits in (0x7F800000, 0xFF800000, 0x7FC00000):
            with self.assertRaises(ValueError):
                quantize_bits(bits, 256, 32767)
        with self.assertRaises(ValueError):
            quantize_bits(float32_bits(-0.5), 256, 127)


if __name__ == "__main__":
    unittest.main(verbosity=2)
