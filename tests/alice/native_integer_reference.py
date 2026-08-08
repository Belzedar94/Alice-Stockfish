"""Independent sparse integer inference and wire editing for AliceNative-v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct

from native_wire import DENSE_STACK_TENSOR_BYTES, FEATURE_TENSOR_BYTES, MANIFEST


WHITE = 0
BLACK = 1
L1 = 1_024
HALF = 512
STACKS = 8
HIDDEN = 32
PSQT_BUCKETS = 8


def _zeros(rows: int, columns: int) -> list[list[int]]:
    return [[0] * columns for _ in range(rows)]


def _stack_weights(stacks: int, outputs: int, inputs: int) -> list[list[list[int]]]:
    return [[[0] * inputs for _ in range(outputs)] for _ in range(stacks)]


@dataclass
class SparseNativeParameters:
    ft_bias: list[int] = field(default_factory=lambda: [0] * L1)
    piece_weight: dict[int, dict[int, int]] = field(default_factory=dict)
    threat_weight: dict[int, dict[int, int]] = field(default_factory=dict)
    piece_psqt: dict[int, dict[int, int]] = field(default_factory=dict)
    threat_psqt: dict[int, dict[int, int]] = field(default_factory=dict)
    fc0_bias: list[list[int]] = field(default_factory=lambda: _zeros(STACKS, HIDDEN))
    fc0_weight: list[list[list[int]]] = field(
        default_factory=lambda: _stack_weights(STACKS, HIDDEN, L1)
    )
    fc1_bias: list[list[int]] = field(default_factory=lambda: _zeros(STACKS, HIDDEN))
    fc1_weight: list[list[list[int]]] = field(
        default_factory=lambda: _stack_weights(STACKS, HIDDEN, 64)
    )
    fc2_bias: list[list[int]] = field(default_factory=lambda: _zeros(STACKS, 1))
    fc2_weight: list[list[list[int]]] = field(
        default_factory=lambda: _stack_weights(STACKS, 1, 128)
    )


FEATURE_LAYOUT = (
    ("ft.bias", "h", 1_024),
    ("threat.weight", "b", 119_616 * 1_024),
    ("threat.psqt", "i", 119_616 * 8),
    ("pieceSquare.weight", "h", 45_056 * 1_024),
    ("pieceSquare.psqt", "i", 45_056 * 8),
)
DENSE_LAYOUT = (
    ("stack.fc0.bias", "i", 32),
    ("stack.fc0.weight", "b", 32 * 1_024),
    ("stack.fc1.bias", "i", 32),
    ("stack.fc1.weight", "b", 32 * 64),
    ("stack.fc2.bias", "i", 1),
    ("stack.fc2.weight", "b", 128),
)
WIRE_TENSOR_START = 12 + len(MANIFEST) + 4


def _wire_location(name: str, flat_index: int) -> tuple[int, str]:
    if flat_index < 0:
        raise ValueError("A tensor index cannot be negative.")

    cursor = WIRE_TENSOR_START
    for tensor, code, elements in FEATURE_LAYOUT:
        item_size = struct.calcsize("<" + code)
        if tensor == name:
            if flat_index >= elements:
                raise ValueError(f"{name} index is out of range.")
            return cursor + flat_index * item_size, code
        cursor += elements * item_size
    if cursor != WIRE_TENSOR_START + FEATURE_TENSOR_BYTES:
        raise AssertionError("Independent feature layout has the wrong byte count.")

    for tensor, code, elements_per_stack in DENSE_LAYOUT:
        if tensor != name:
            continue
        if flat_index >= STACKS * elements_per_stack:
            raise ValueError(f"{name} index is out of range.")
        stack, local = divmod(flat_index, elements_per_stack)
        stack_cursor = cursor + stack * (4 + DENSE_STACK_TENSOR_BYTES) + 4
        for candidate, candidate_code, candidate_elements in DENSE_LAYOUT:
            item_size = struct.calcsize("<" + candidate_code)
            if candidate == name:
                return stack_cursor + local * item_size, code
            stack_cursor += candidate_elements * item_size
    raise ValueError(f"Unknown AliceNative-v1 tensor: {name}")


def write_wire_parameter(path: Path, name: str, flat_index: int, value: int) -> None:
    offset, code = _wire_location(name, flat_index)
    payload = struct.pack("<" + code, value)
    with path.open("r+b") as output:
        output.seek(offset)
        output.write(payload)


def record_parameter(
    parameters: SparseNativeParameters, name: str, flat_index: int, value: int
) -> None:
    if name == "ft.bias":
        parameters.ft_bias[flat_index] = value
        return

    sparse = {
        "pieceSquare.weight": (parameters.piece_weight, L1),
        "pieceSquare.psqt": (parameters.piece_psqt, PSQT_BUCKETS),
        "threat.weight": (parameters.threat_weight, L1),
        "threat.psqt": (parameters.threat_psqt, PSQT_BUCKETS),
    }
    if name in sparse:
        destination, width = sparse[name]
        row, column = divmod(flat_index, width)
        destination.setdefault(row, {})[column] = value
        return

    dense = {
        "stack.fc0.bias": (parameters.fc0_bias, 32, None),
        "stack.fc0.weight": (parameters.fc0_weight, 32 * L1, L1),
        "stack.fc1.bias": (parameters.fc1_bias, 32, None),
        "stack.fc1.weight": (parameters.fc1_weight, 32 * 64, 64),
        "stack.fc2.bias": (parameters.fc2_bias, 1, None),
        "stack.fc2.weight": (parameters.fc2_weight, 128, 128),
    }
    if name not in dense:
        raise ValueError(f"Unknown AliceNative-v1 tensor: {name}")
    destination, per_stack, width = dense[name]
    stack, local = divmod(flat_index, per_stack)
    if width is None:
        destination[stack][local] = value
    else:
        row, column = divmod(local, width)
        destination[stack][row][column] = value


def install_parameter(
    path: Path,
    parameters: SparseNativeParameters,
    name: str,
    flat_index: int,
    value: int,
) -> None:
    write_wire_parameter(path, name, flat_index, value)
    record_parameter(parameters, name, flat_index, value)


def trunc0(numerator: int, denominator: int) -> int:
    quotient = abs(numerator) // denominator
    return -quotient if numerator < 0 else quotient


def _affine(
    weights: list[list[int]], biases: list[int], values: list[int]
) -> list[int]:
    return [
        bias + sum(weight * value for weight, value in zip(row, values, strict=True))
        for row, bias in zip(weights, biases, strict=True)
    ]


def _linear(value: int, shift: int) -> int:
    return min(127, max(0, value // (1 << shift)))


def _square(value: int, shift: int) -> int:
    return min(127, (value * value) // (1 << (2 * shift + 7)))


def _add_sparse_rows(
    destination: list[int], rows: dict[int, dict[int, int]], indices: list[int]
) -> None:
    for index in indices:
        for column, value in rows.get(index, {}).items():
            destination[column] += value


def evaluate_integer(
    parameters: SparseNativeParameters,
    piece_features: list[list[int]],
    threat_features: list[list[int]],
    side_to_move: int,
    piece_count: int,
) -> dict[str, object]:
    accumulators: list[list[int]] = []
    psqt_accumulators: list[list[int]] = []
    for perspective in (WHITE, BLACK):
        accumulator = list(parameters.ft_bias)
        psqt = [0] * PSQT_BUCKETS
        _add_sparse_rows(accumulator, parameters.piece_weight, piece_features[perspective])
        _add_sparse_rows(accumulator, parameters.threat_weight, threat_features[perspective])
        _add_sparse_rows(psqt, parameters.piece_psqt, piece_features[perspective])
        _add_sparse_rows(psqt, parameters.threat_psqt, threat_features[perspective])
        accumulators.append(accumulator)
        psqt_accumulators.append(psqt)

    transformed: list[list[int]] = []
    for accumulator in accumulators:
        transformed.append(
            [
                min(255, max(0, accumulator[lane]))
                * min(255, max(0, accumulator[lane + HALF]))
                // 512
                for lane in range(HALF)
            ]
        )

    dense_input = transformed[side_to_move] + transformed[side_to_move ^ 1]
    phase = (piece_count - 1) // 4
    z0 = _affine(parameters.fc0_weight[phase], parameters.fc0_bias[phase], dense_input)
    s0 = [_square(value, 7) for value in z0]
    r0 = [_linear(value, 7) for value in z0]
    y1 = s0 + r0
    z1 = _affine(parameters.fc1_weight[phase], parameters.fc1_bias[phase], y1)
    s1 = [_square(value, 6) for value in z1]
    r1 = [_linear(value, 6) for value in z1]
    y2 = s0 + r0 + s1 + r1
    z2 = _affine(parameters.fc2_weight[phase], parameters.fc2_bias[phase], y2)[0]
    skip = z0[30] - z0[31]
    fwd_out = z2 + skip
    positional_raw16 = trunc0(fwd_out * 9_600, 16_384)
    psqt_raw16 = trunc0(
        psqt_accumulators[side_to_move][phase]
        - psqt_accumulators[side_to_move ^ 1][phase],
        2,
    )
    positional_value = trunc0(positional_raw16, 16)
    psqt_value = trunc0(psqt_raw16, 16)

    return {
        "featureAccumulator": accumulators,
        "psqtAccumulator": psqt_accumulators,
        "transformedByPerspective": transformed,
        "transformedInput": dense_input,
        "phase": phase,
        "fc0Raw": z0,
        "fc0Squared": s0,
        "fc0Linear": r0,
        "fc1Raw": z1,
        "fc1Squared": s1,
        "fc1Linear": r1,
        "fc2Raw": z2,
        "skip": skip,
        "fwdOut": fwd_out,
        "positionalRaw16": positional_raw16,
        "psqtRaw16": psqt_raw16,
        "positionalValue": positional_value,
        "psqtValue": psqt_value,
        "nativeNnueValue": positional_value + psqt_value,
    }
