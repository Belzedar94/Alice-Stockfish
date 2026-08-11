"""Audit an Alice release candidate without publishing or modifying artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys

if __package__:
    from .alice_acceptance.aggregate import (
        aggregate_input_identity,
        validate_aggregate_receipt,
    )
    from .alice_acceptance.evidence import (
        canonical_json_bytes,
        sha256_file,
        write_create_only_json,
    )
    from .alice_acceptance.runner_adapter import parse_strict_json
else:
    from alice_acceptance.aggregate import (
        aggregate_input_identity,
        validate_aggregate_receipt,
    )
    from alice_acceptance.evidence import (
        canonical_json_bytes,
        sha256_file,
        write_create_only_json,
    )
    from alice_acceptance.runner_adapter import parse_strict_json


EXPECTED_NATIVE_SIZE = 220_315_747
CANONICAL_BENCH_NODES = 202_963
FROZEN_LEGACY_BINARY_SHA256 = (
    "b70afe03ec9a67258cd7b5b848c46fc9e5c83f53b9f2825e9a5946feefb59599"
)
FROZEN_LEGACY_NETWORK_SHA256 = (
    "9f9e557015a55c0a6981db64e1f3044dedb91fd8a8c1a6d4f3c45d0eee91fbd9"
)
FROZEN_ALICE_BOOK_SHA256 = (
    "bcd89d9fc3ea81feb95932eb64d6b6f15ad25cc04cdcc9e0440f097cffb8ccf6"
)
BINARY_ROLES = frozenset(
    {"windows-bmi2", "windows-avx2", "linux-bmi2", "linux-avx2"}
)
BINARY_ROLE_REQUIREMENTS = {
    "windows-bmi2": ("windows-x86-64", "x86-64-bmi2"),
    "windows-avx2": ("windows-x86-64", "x86-64-avx2"),
    "linux-bmi2": ("linux-x86-64", "x86-64-bmi2"),
    "linux-avx2": ("linux-x86-64", "x86-64-avx2"),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
QUALIFICATION_FIELDS = {
    "schema",
    "status",
    "network_sha256",
    "network_kind",
    "training_run_id",
    "dataset_manifest",
    "checkpoint",
    "export_receipt",
    "network_parameter_nonzero_bytes",
    "gates",
}
SHADOW_FIELDS = {
    "schema",
    "service",
    "status",
    "source_commit",
    "network_sha256",
    "presets",
}
SHADOW_PRESET_FIELDS = {
    "binary_role",
    "binary_sha256",
    "configuration",
    "pairs",
    "inversions",
    "invalid_pairs",
    "adjudication",
}
TRIPLE_BENCH_FIELDS = {"schema", "binary_sha256", "network_sha256", "runs"}
TRIPLE_BENCH_RUN_FIELDS = {"ordinal", "command", "stdout", "exit_code"}
TRIPLE_BENCH_COMMAND_FIELDS = {
    "schema",
    "ordinal",
    "binary_path",
    "binary_sha256",
    "network_path",
    "network_sha256",
    "stdin",
}
TRIPLE_BENCH_STDIN = "bench\nquit\n"
LOAD_FAILURE_MATRIX_FIELDS = {
    "schema",
    "binary_sha256",
    "network_sha256",
    "cases",
}
LOAD_FAILURE_CASE_FIELDS = {
    "probe_kind",
    "source_network_sha256",
    "input_descriptor",
    "input",
    "mutation",
    "command",
    "output",
    "diagnostic_code",
    "exit_code",
    "fallback_observed",
    "search_result_published",
}
LOAD_FAILURE_PROBES = {
    "missing": ("absent-path", "ALICE_NETWORK_MISSING"),
    "corrupt": ("deterministic-byte-flip", "ALICE_NETWORK_CORRUPT"),
    "incompatible": (
        "architecture-word-mismatch",
        "ALICE_NETWORK_INCOMPATIBLE",
    ),
}
LOAD_INPUT_DESCRIPTOR_FIELDS = {
    "schema",
    "probe_kind",
    "source_network_sha256",
    "input_path",
    "input_sha256",
    "mutation",
}
LOAD_COMMAND_FIELDS = {
    "schema",
    "probe_kind",
    "binary_sha256",
    "source_network_sha256",
    "input_descriptor_sha256",
    "input_sha256",
}
LOAD_OUTPUT_FIELDS = {
    "schema",
    "probe_kind",
    "binary_sha256",
    "command_sha256",
    "diagnostic_code",
    "exit_code",
    "fallback_observed",
    "search_result_published",
}
SHADOW_CONFIGURATION_FIELDS = {
    "schema",
    "service",
    "preset",
    "source_commit",
    "network_sha256",
    "binary_role",
    "binary_sha256",
    "book_token",
    "book_sha256",
    "runner_sha256",
    "engine_options",
    "timing",
    "worker",
    "adjudication",
}
SHADOW_TIMING = {
    "VSTC": {"base_ms": 2_000, "increment_ms": 20},
    "STC": {"base_ms": 10_000, "increment_ms": 100},
    "LTC": {"base_ms": 30_000, "increment_ms": 300},
}
SHADOW_ENGINE_OPTIONS = {"Threads": "1", "Hash": "512", "Move Overhead": "10"}
SHADOW_WORKER_CONFIGURATION = {"cpuflags": [], "pairing": "color-swapped-pairs"}
NATIVE_WIRE_BYTES = 220_315_747
NATIVE_WIRE_VERSION = 0xA11CE001
NATIVE_ARCHITECTURE_HASH = 0xEC7CCD50
NATIVE_FEATURE_TENSOR_BYTES = (
    2 * 1_024
    + 119_616 * 1_024
    + 119_616 * 8 * 4
    + 45_056 * 1_024 * 2
    + 45_056 * 8 * 4
)
NATIVE_DENSE_STACK_TENSOR_BYTES = (
    32 * 4 + 32 * 1_024 + 32 * 4 + 32 * 64 + 4 + 128
)
NATIVE_LAYER_STACKS = 8
NATIVE_PARAMETER_ELEMENTS = (
    1_024
    + 119_616 * 1_024
    + 119_616 * 8
    + 45_056 * 1_024
    + 45_056 * 8
    + NATIVE_LAYER_STACKS
    * (32 + 32 * 1_024 + 32 + 32 * 64 + 1 + 128)
)
DATASET_MANIFEST_FIELDS = {
    "schema",
    "training_run_id",
    "position_count",
    "train_position_count",
    "validation_position_count",
    "split_seed",
}
EXPORT_RECEIPT_FIELDS = {
    "schema",
    "training_run_id",
    "checkpoint_sha256",
    "network_sha256",
    "network_bytes",
    "element_count",
    "element_mismatches",
    "deterministic_reexport_sha256",
}
GATE_REPORT_FIELDS = {
    "schema",
    "gate",
    "status",
    "training_run_id",
    "network_sha256",
    "dataset_manifest_sha256",
    "checkpoint_sha256",
    "export_receipt_sha256",
    "sample_count",
    "mismatch_count",
}


def load_object(path: Path) -> dict[str, object]:
    return parse_strict_json(path.read_bytes())


def exact_fields(value: dict[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match the contract")


def verify_reference(
    value: object,
    label: str,
    reasons: list[str],
) -> tuple[Path | None, str | None]:
    if not isinstance(value, dict):
        reasons.append(f"{label}: reference is not an object")
        return None, None
    try:
        exact_fields(value, {"path", "sha256"}, label)
    except ValueError as error:
        reasons.append(str(error))
        return None, None
    path_value = value.get("path")
    expected = value.get("sha256")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        reasons.append(f"{label}: path is not absolute")
        return None, None
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        reasons.append(f"{label}: SHA-256 is not canonical")
        return None, None
    path = Path(path_value).resolve()
    if not path.is_file():
        reasons.append(f"{label}: file is missing")
        return None, expected
    if sha256_file(path) != expected:
        reasons.append(f"{label}: SHA-256 mismatch")
        return path, expected
    return path, expected


def executable_format(path: Path) -> str | None:
    with path.open("rb") as stream:
        header = stream.read(64)
        if (
            len(header) >= 20
            and header[:4] == b"\x7fELF"
            and header[4] == 2
            and header[5] == 1
            and int.from_bytes(header[18:20], "little") == 62
        ):
            return "linux-x86-64"
        if len(header) >= 64 and header[:2] == b"MZ":
            pe_offset = int.from_bytes(header[60:64], "little")
            stream.seek(pe_offset)
            pe_header = stream.read(26)
            if (
                len(pe_header) == 26
                and pe_header[:4] == b"PE\x00\x00"
                and int.from_bytes(pe_header[4:6], "little") == 0x8664
                and int.from_bytes(pe_header[24:26], "little") == 0x20B
            ):
                return "windows-x86-64"
    return None


def file_contains(path: Path, needle: bytes) -> bool:
    overlap = max(0, len(needle) - 1)
    previous = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value = previous + chunk
            if needle in value:
                return True
            previous = value[-overlap:] if overlap else b""
    return False


def verify_binary_role(
    path: Path, role: str, source_commit: str, reasons: list[str]
) -> None:
    expected_format, expected_architecture = BINARY_ROLE_REQUIREMENTS[role]
    actual_format = executable_format(path)
    if actual_format != expected_format:
        reasons.append(
            f"{role}: executable format is {actual_format or 'unsupported'}, "
            f"expected {expected_format}"
        )
    if not file_contains(path, expected_architecture.encode("ascii")):
        reasons.append(
            f"{role}: binary does not embed architecture {expected_architecture}"
        )
    platform_markers = (
        (b" on MinGW64", b" on Microsoft Windows 64-bit")
        if expected_format == "windows-x86-64"
        else (b" on Linux",)
    )
    if not any(file_contains(path, marker) for marker in platform_markers):
        reasons.append(f"{role}: binary does not embed the expected compiler platform")
    if not file_contains(path, source_commit.encode("ascii")):
        reasons.append(
            f"{role}: binary does not embed the declared full source commit"
        )
    if file_contains(path, b"Source tree state          : dirty"):
        reasons.append(f"{role}: binary embeds a dirty source-tree marker")
    if not file_contains(path, b"Source tree state          : clean"):
        reasons.append(f"{role}: binary was not built from a clean source tree")
    other_architecture = (
        "x86-64-avx2" if expected_architecture == "x86-64-bmi2" else "x86-64-bmi2"
    )
    if file_contains(path, other_architecture.encode("ascii")):
        reasons.append(
            f"{role}: binary also embeds incompatible architecture {other_architecture}"
        )


def verify_acceptance(
    receipt: dict[str, object], mode: str, label: str, reasons: list[str]
) -> dict[str, object] | None:
    try:
        validate_aggregate_receipt(receipt, mode)
    except ValueError as error:
        reasons.append(f"{label}: {error}")
        return None
    return aggregate_input_identity(receipt)


def count_nonzero_bytes(stream, byte_count: int) -> int | None:
    remaining = byte_count
    nonzero = 0
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            return None
        nonzero += len(chunk) - chunk.count(0)
        remaining -= len(chunk)
    return nonzero


def count_native_parameter_nonzero_bytes(path: Path) -> int | None:
    """Count only tensor bytes, excluding native wire metadata and hashes."""

    if path.stat().st_size != NATIVE_WIRE_BYTES:
        with path.open("rb") as stream:
            return count_nonzero_bytes(stream, path.stat().st_size)
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            return None
        if (
            int.from_bytes(header[0:4], "little") != NATIVE_WIRE_VERSION
            or int.from_bytes(header[4:8], "little") != NATIVE_ARCHITECTURE_HASH
        ):
            return None
        manifest_length = int.from_bytes(header[8:12], "little")
        expected_manifest_length = NATIVE_WIRE_BYTES - (
            12
            + 4
            + NATIVE_FEATURE_TENSOR_BYTES
            + NATIVE_LAYER_STACKS * (4 + NATIVE_DENSE_STACK_TENSOR_BYTES)
        )
        if manifest_length != expected_manifest_length:
            return None
        stream.seek(manifest_length, 1)
        if len(stream.read(4)) != 4:
            return None
        nonzero = count_nonzero_bytes(stream, NATIVE_FEATURE_TENSOR_BYTES)
        if nonzero is None:
            return None
        for _ in range(NATIVE_LAYER_STACKS):
            if len(stream.read(4)) != 4:
                return None
            stack_nonzero = count_nonzero_bytes(stream, NATIVE_DENSE_STACK_TENSOR_BYTES)
            if stack_nonzero is None:
                return None
            nonzero += stack_nonzero
        if stream.read(1):
            return None
        return nonzero


def verify_native_qualification(
    receipt: dict[str, object],
    network_path: Path,
    network_sha256: str,
    reasons: list[str],
) -> None:
    required_gates = {f"G{index}" for index in range(1, 9)}
    gates = receipt.get("gates")
    training_run_id = receipt.get("training_run_id")
    if (
        set(receipt) != QUALIFICATION_FIELDS
        or receipt.get("schema") != "alice-native-qualification-v1"
        or receipt.get("status") != "qualified"
        or receipt.get("network_sha256") != network_sha256
        or receipt.get("network_kind") != "trained"
        or not isinstance(training_run_id, str)
        or not ID_RE.fullmatch(training_run_id)
        or not isinstance(gates, dict)
        or set(gates) != required_gates
    ):
        reasons.append("native qualification: trained G1-G8 evidence is incomplete")
        return

    dataset_path, dataset_sha = verify_reference(
        receipt.get("dataset_manifest"), "native qualification dataset manifest", reasons
    )
    checkpoint_path, checkpoint_sha = verify_reference(
        receipt.get("checkpoint"), "native qualification checkpoint", reasons
    )
    export_path, export_sha = verify_reference(
        receipt.get("export_receipt"), "native qualification export receipt", reasons
    )
    dataset = load_evidence_object(
        dataset_path, "native qualification dataset manifest", reasons
    )
    export = load_evidence_object(export_path, "native qualification export receipt", reasons)
    if checkpoint_path is not None and checkpoint_path.stat().st_size == 0:
        reasons.append("native qualification: checkpoint is empty")

    dataset_position_count: int | None = None
    if dataset is not None:
        try:
            exact_fields(dataset, DATASET_MANIFEST_FIELDS, "native qualification dataset")
        except ValueError as error:
            reasons.append(str(error))
        position_count = dataset.get("position_count")
        train_count = dataset.get("train_position_count")
        validation_count = dataset.get("validation_position_count")
        split_seed = dataset.get("split_seed")
        if (
            dataset.get("schema") != "alice-training-dataset-manifest-v1"
            or dataset.get("training_run_id") != training_run_id
            or type(position_count) is not int
            or type(train_count) is not int
            or type(validation_count) is not int
            or type(split_seed) is not int
            or position_count <= 0
            or train_count <= 0
            or validation_count <= 0
            or train_count + validation_count != position_count
        ):
            reasons.append("native qualification: dataset manifest is inconsistent")
        else:
            dataset_position_count = position_count

    export_element_count: int | None = None
    if export is not None:
        try:
            exact_fields(export, EXPORT_RECEIPT_FIELDS, "native qualification export receipt")
        except ValueError as error:
            reasons.append(str(error))
        element_count = export.get("element_count")
        if (
            export.get("schema") != "alice-native-export-receipt-v1"
            or export.get("training_run_id") != training_run_id
            or export.get("checkpoint_sha256") != checkpoint_sha
            or export.get("network_sha256") != network_sha256
            or export.get("deterministic_reexport_sha256") != network_sha256
            or type(export.get("network_bytes")) is not int
            or export.get("network_bytes") != network_path.stat().st_size
            or type(element_count) is not int
            or element_count != NATIVE_PARAMETER_ELEMENTS
            or type(export.get("element_mismatches")) is not int
            or export.get("element_mismatches") != 0
        ):
            reasons.append("native qualification: export receipt is inconsistent")
        else:
            export_element_count = element_count

    computed_nonzero = count_native_parameter_nonzero_bytes(network_path)
    declared_nonzero = receipt.get("network_parameter_nonzero_bytes")
    if (
        type(declared_nonzero) is not int
        or computed_nonzero is None
        or declared_nonzero != computed_nonzero
        or computed_nonzero <= 0
    ):
        reasons.append(
            "native qualification: candidate network has no verified nonzero parameters"
        )

    gate_hashes: set[str] = set()
    for gate in sorted(required_gates):
        report_path, report_sha = verify_reference(
            gates.get(gate), f"native qualification {gate} report", reasons
        )
        report = load_evidence_object(
            report_path, f"native qualification {gate} report", reasons
        )
        if report_sha is not None:
            gate_hashes.add(report_sha)
        if report is None:
            continue
        try:
            exact_fields(report, GATE_REPORT_FIELDS, f"native qualification {gate} report")
        except ValueError as error:
            reasons.append(str(error))
        sample_count = report.get("sample_count")
        mismatch_count = report.get("mismatch_count")
        expected_sample_count = None
        if gate in {"G1", "G2", "G6"}:
            expected_sample_count = dataset_position_count
        elif gate in {"G4", "G5"}:
            expected_sample_count = export_element_count
        if (
            report.get("schema") != "alice-native-gate-report-v1"
            or report.get("gate") != gate
            or report.get("status") != "PASS"
            or report.get("training_run_id") != training_run_id
            or report.get("network_sha256") != network_sha256
            or report.get("dataset_manifest_sha256") != dataset_sha
            or report.get("checkpoint_sha256") != checkpoint_sha
            or report.get("export_receipt_sha256") != export_sha
            or type(sample_count) is not int
            or sample_count <= 0
            or (expected_sample_count is not None and sample_count != expected_sample_count)
            or type(mismatch_count) is not int
            or mismatch_count != 0
        ):
            reasons.append(f"native qualification: {gate} report did not pass exactly")
    if len(gate_hashes) != 8:
        reasons.append("native qualification: G1-G8 reports are not distinct")


def verify_triple_bench(
    receipt: dict[str, object],
    binary_path: Path,
    binary_sha256: str,
    network_path: Path,
    network_sha256: str,
    role: str,
    reasons: list[str],
) -> None:
    runs = receipt.get("runs")
    if (
        set(receipt) != TRIPLE_BENCH_FIELDS
        or receipt.get("schema") != "alice-triple-bench-v1"
        or receipt.get("binary_sha256") != binary_sha256
        or receipt.get("network_sha256") != network_sha256
        or not isinstance(runs, list)
        or len(runs) != 3
    ):
        reasons.append(f"{role}: triple bench is not reproducible")
        return
    command_hashes: set[str] = set()
    command_paths: set[Path] = set()
    stdout_paths: set[Path] = set()
    for ordinal, run in enumerate(runs):
        label = f"{role} triple bench run {ordinal}"
        if (
            not isinstance(run, dict)
            or set(run) != TRIPLE_BENCH_RUN_FIELDS
            or type(run.get("ordinal")) is not int
            or run.get("ordinal") != ordinal
            or type(run.get("exit_code")) is not int
            or run.get("exit_code") != 0
        ):
            reasons.append(f"{label}: execution receipt is incomplete")
            continue
        command_path, command_sha = verify_reference(
            run.get("command"), f"{label} command", reasons
        )
        stdout_path, _stdout_sha = verify_reference(
            run.get("stdout"), f"{label} stdout", reasons
        )
        command = load_evidence_object(command_path, f"{label} command", reasons)
        expected_command = {
            "schema": "alice-triple-bench-command-v1",
            "ordinal": ordinal,
            "binary_path": str(binary_path.resolve()),
            "binary_sha256": binary_sha256,
            "network_path": str(network_path.resolve()),
            "network_sha256": network_sha256,
            "stdin": TRIPLE_BENCH_STDIN,
        }
        if command is not None:
            try:
                exact_fields(command, TRIPLE_BENCH_COMMAND_FIELDS, f"{label} command")
            except ValueError as error:
                reasons.append(str(error))
            if command != expected_command:
                reasons.append(f"{label}: command is not bound to the release artifacts")
            elif (
                command_path is not None
                and hashlib.sha256(canonical_json_bytes(command)).hexdigest()
                != command_sha
            ):
                reasons.append(f"{label}: command is not canonical JSON")
        if stdout_path is not None:
            try:
                stdout = stdout_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                reasons.append(f"{label}: stdout is not UTF-8: {error}")
            else:
                signatures = [
                    match.group(1)
                    for line in stdout.splitlines()
                    if (
                        match := re.fullmatch(
                            r"Nodes searched\s*:\s*([0-9]+)", line.strip()
                        )
                    )
                ]
                if (
                    signatures != [str(CANONICAL_BENCH_NODES)]
                    or network_sha256 not in stdout.lower()
                ):
                    reasons.append(f"{label}: stdout lacks the canonical bench evidence")
        if command_path is not None:
            command_paths.add(command_path)
        if command_sha is not None:
            command_hashes.add(command_sha)
        if stdout_path is not None:
            stdout_paths.add(stdout_path)
    if (
        len(command_hashes) != 3
        or len(command_paths) != 3
        or len(stdout_paths) != 3
    ):
        reasons.append(f"{role}: triple bench does not bind three distinct executions")


def matches_frozen_mutation(source: Path, mutated: Path, probe_kind: str) -> bool:
    """Compare a probe input with the source without loading a release net in memory."""

    expected_offset = 0 if probe_kind == "corrupt" else 4
    if (
        source.stat().st_size != mutated.stat().st_size
        or source.stat().st_size <= expected_offset
    ):
        return False
    observed_offset = False
    absolute_offset = 0
    with source.open("rb") as source_stream, mutated.open("rb") as mutated_stream:
        while True:
            source_chunk = source_stream.read(1024 * 1024)
            mutated_chunk = mutated_stream.read(1024 * 1024)
            if len(source_chunk) != len(mutated_chunk):
                return False
            if not source_chunk:
                return observed_offset
            if absolute_offset <= expected_offset < absolute_offset + len(source_chunk):
                expected_chunk = bytearray(source_chunk)
                index = expected_offset - absolute_offset
                expected_chunk[index] ^= 0x01
                if mutated_chunk != bytes(expected_chunk):
                    return False
                observed_offset = True
            elif mutated_chunk != source_chunk:
                return False
            absolute_offset += len(source_chunk)


def load_evidence_object(
    path: Path | None, label: str, reasons: list[str]
) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        return load_object(path)
    except (UnicodeDecodeError, ValueError) as error:
        reasons.append(f"{label}: invalid JSON: {error}")
        return None


def verify_load_failures(
    receipt: dict[str, object],
    binary_sha256: str,
    network_path: Path,
    network_sha256: str,
    role: str,
    reasons: list[str],
) -> None:
    cases = receipt.get("cases")
    expected_cases = set(LOAD_FAILURE_PROBES)
    if (
        set(receipt) != LOAD_FAILURE_MATRIX_FIELDS
        or receipt.get("schema") != "alice-load-failure-matrix-v1"
        or receipt.get("binary_sha256") != binary_sha256
        or receipt.get("network_sha256") != network_sha256
        or not isinstance(cases, dict)
        or set(cases) != expected_cases
    ):
        reasons.append(f"{role}: load-failure matrix is incomplete")
        return
    input_descriptors: set[str] = set()
    commands: set[str] = set()
    outputs: set[str] = set()
    mutated_inputs: set[str] = set()
    for name, case in cases.items():
        mutation, diagnostic_code = LOAD_FAILURE_PROBES[name]
        if (
            not isinstance(case, dict)
            or set(case) != LOAD_FAILURE_CASE_FIELDS
            or case.get("probe_kind") != name
            or case.get("source_network_sha256") != network_sha256
            or case.get("mutation") != mutation
            or case.get("diagnostic_code") != diagnostic_code
            or type(case.get("exit_code")) is not int
            or case.get("exit_code") == 0
            or case.get("fallback_observed") is not False
            or case.get("search_result_published") is not False
        ):
            reasons.append(
                f"{role}: {name} load-failure evidence is incomplete or did not fail closed"
            )
            continue

        descriptor_path, descriptor_sha = verify_reference(
            case.get("input_descriptor"),
            f"{role} {name} input descriptor",
            reasons,
        )
        command_path, command_sha = verify_reference(
            case.get("command"), f"{role} {name} command", reasons
        )
        output_path, output_sha = verify_reference(
            case.get("output"), f"{role} {name} output", reasons
        )
        input_path: Path | None = None
        input_sha256: str | None = None
        if name == "missing":
            if case.get("input") is not None:
                reasons.append(
                    f"{role}: missing load probe unexpectedly identifies input bytes"
                )
                continue
        else:
            input_path, input_sha256 = verify_reference(
                case.get("input"), f"{role} {name} input", reasons
            )
            if (
                input_path is None
                or input_sha256 is None
                or input_sha256 == network_sha256
            ):
                reasons.append(
                    f"{role}: {name} load probe does not bind mutated input bytes"
                )
            elif not matches_frozen_mutation(network_path, input_path, name):
                reasons.append(
                    f"{role}: {name} input does not match the frozen {mutation} recipe"
                )

        descriptor = load_evidence_object(
            descriptor_path, f"{role} {name} input descriptor", reasons
        )
        descriptor_input_path: Path | None = None
        if descriptor is not None:
            try:
                exact_fields(
                    descriptor,
                    LOAD_INPUT_DESCRIPTOR_FIELDS,
                    f"{role} {name} input descriptor",
                )
            except ValueError as error:
                reasons.append(str(error))
            path_value = descriptor.get("input_path")
            if isinstance(path_value, str) and Path(path_value).is_absolute():
                descriptor_input_path = Path(path_value).resolve()
            expected_descriptor = {
                "schema": "alice-load-probe-input-v1",
                "probe_kind": name,
                "source_network_sha256": network_sha256,
                "input_path": path_value,
                "input_sha256": input_sha256,
                "mutation": mutation,
            }
            if descriptor != expected_descriptor:
                reasons.append(f"{role}: {name} input descriptor is not canonical")
            if name == "missing":
                if descriptor_input_path is None or descriptor_input_path.exists():
                    reasons.append(
                        f"{role}: missing probe input path is not verifiably absent"
                    )
            elif descriptor_input_path != input_path:
                reasons.append(
                    f"{role}: {name} descriptor does not identify the mutated input"
                )

        command = load_evidence_object(command_path, f"{role} {name} command", reasons)
        expected_command = {
            "schema": "alice-load-probe-command-v1",
            "probe_kind": name,
            "binary_sha256": binary_sha256,
            "source_network_sha256": network_sha256,
            "input_descriptor_sha256": descriptor_sha,
            "input_sha256": input_sha256,
        }
        if command is not None:
            try:
                exact_fields(command, LOAD_COMMAND_FIELDS, f"{role} {name} command")
            except ValueError as error:
                reasons.append(str(error))
            if command != expected_command:
                reasons.append(f"{role}: {name} command is not bound to its probe input")

        output = load_evidence_object(output_path, f"{role} {name} output", reasons)
        expected_output = {
            "schema": "alice-load-probe-output-v1",
            "probe_kind": name,
            "binary_sha256": binary_sha256,
            "command_sha256": command_sha,
            "diagnostic_code": diagnostic_code,
            "exit_code": case.get("exit_code"),
            "fallback_observed": False,
            "search_result_published": False,
        }
        if output is not None:
            try:
                exact_fields(output, LOAD_OUTPUT_FIELDS, f"{role} {name} output")
            except ValueError as error:
                reasons.append(str(error))
            if output != expected_output:
                reasons.append(f"{role}: {name} output is not bound to its command")

        if descriptor_sha is not None:
            input_descriptors.add(descriptor_sha)
        if command_sha is not None:
            commands.add(command_sha)
        if output_sha is not None:
            outputs.add(output_sha)
        if input_sha256 is not None:
            mutated_inputs.add(input_sha256)
    if (
        len(input_descriptors) != 3
        or len(commands) != 3
        or len(outputs) != 3
        or len(mutated_inputs) != 2
    ):
        reasons.append(
            f"{role}: load-failure probes do not bind three distinct executions"
        )


def verify_shadow_configuration(
    reference: object,
    receipt: dict[str, object],
    preset: str,
    result: dict[str, object],
    reasons: list[str],
) -> tuple[str | None, str | None]:
    label = f"OpenBench shadow preset {preset} configuration"
    path, configuration_sha = verify_reference(reference, label, reasons)
    configuration = load_evidence_object(path, label, reasons)
    if configuration is None:
        return configuration_sha, None
    try:
        exact_fields(configuration, SHADOW_CONFIGURATION_FIELDS, label)
    except ValueError as error:
        reasons.append(str(error))
        return configuration_sha, None
    runner_sha = configuration.get("runner_sha256")
    expected = {
        "schema": "alice-openbench-shadow-configuration-v1",
        "service": receipt.get("service"),
        "preset": preset,
        "source_commit": receipt.get("source_commit"),
        "network_sha256": receipt.get("network_sha256"),
        "binary_role": result.get("binary_role"),
        "binary_sha256": result.get("binary_sha256"),
        "book_token": "ALICE",
        "book_sha256": FROZEN_ALICE_BOOK_SHA256,
        "runner_sha256": runner_sha,
        "engine_options": SHADOW_ENGINE_OPTIONS,
        "timing": SHADOW_TIMING[preset],
        "worker": SHADOW_WORKER_CONFIGURATION,
        "adjudication": ["800/4", "40/8/10"],
    }
    if (
        configuration != expected
        or not isinstance(runner_sha, str)
        or not SHA256_RE.fullmatch(runner_sha)
    ):
        reasons.append(f"{label} does not match the frozen preset")
        return configuration_sha, None
    if (
        hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()
        != configuration_sha
    ):
        reasons.append(f"{label} identity is not canonical")
    return configuration_sha, runner_sha


def verify_openbench_shadow(
    receipt: dict[str, object],
    source_commit: str,
    network_sha256: str,
    binary_sha256_by_role: dict[str, str],
    reasons: list[str],
) -> None:
    try:
        exact_fields(receipt, SHADOW_FIELDS, "OpenBench shadow evidence")
    except ValueError as error:
        reasons.append(str(error))
        return
    presets = receipt.get("presets")
    if (
        receipt.get("schema") != "alice-openbench-shadow-receipt-v1"
        or receipt.get("service") != "https://belzedar.duckdns.org"
        or receipt.get("status") != "PASS"
        or not isinstance(presets, dict)
        or set(presets) != {"VSTC", "STC", "LTC"}
    ):
        reasons.append("OpenBench shadow evidence is incomplete")
        return
    if receipt.get("source_commit") != source_commit:
        reasons.append("OpenBench shadow evidence does not bind the candidate source commit")
    if receipt.get("network_sha256") != network_sha256:
        reasons.append("OpenBench shadow evidence does not bind the candidate network")
    configuration_hashes: set[str] = set()
    runner_hashes: set[str] = set()
    for preset, result in presets.items():
        if not isinstance(result, dict):
            reasons.append(f"OpenBench shadow preset {preset} is not an object")
            continue
        try:
            exact_fields(result, SHADOW_PRESET_FIELDS, f"OpenBench shadow preset {preset}")
        except ValueError as error:
            reasons.append(str(error))
            continue
        binary_role = result.get("binary_role")
        binary_sha256 = result.get("binary_sha256")
        if (
            not isinstance(binary_role, str)
            or binary_role not in binary_sha256_by_role
            or not isinstance(binary_sha256, str)
            or not SHA256_RE.fullmatch(binary_sha256)
            or binary_sha256_by_role[binary_role] != binary_sha256
        ):
            reasons.append(
                f"OpenBench shadow preset {preset} does not bind a candidate binary"
            )
        pairs = result.get("pairs")
        inversions = result.get("inversions")
        invalid_pairs = result.get("invalid_pairs")
        if (
            type(pairs) is not int
            or pairs != 200
            or type(inversions) is not int
            or inversions != 0
            or type(invalid_pairs) is not int
            or invalid_pairs != 0
            or result.get("adjudication") != ["800/4", "40/8/10"]
        ):
            reasons.append(f"OpenBench shadow preset {preset} is not clean")
        configuration_sha, runner_sha = verify_shadow_configuration(
            result.get("configuration"), receipt, preset, result, reasons
        )
        if configuration_sha is not None:
            configuration_hashes.add(configuration_sha)
        if runner_sha is not None:
            runner_hashes.add(runner_sha)
    if len(configuration_hashes) != 3 or len(runner_hashes) != 1:
        reasons.append(
            "OpenBench shadow presets do not bind three frozen configurations and one runner"
        )


def audit_release_candidate(manifest_path: Path) -> dict[str, object]:
    manifest = load_object(manifest_path)
    exact_fields(
        manifest,
        {
            "schema",
            "release_id",
            "source_commit",
            "network",
            "native_qualification",
            "exact_los_receipt",
            "fixed_final_receipt",
            "openbench_shadow_receipt",
            "binaries",
        },
        "release candidate",
    )
    if manifest.get("schema") != "alice-release-candidate-v1":
        raise ValueError("unsupported release-candidate schema")
    release_id = manifest.get("release_id")
    source_commit = manifest.get("source_commit")
    if not isinstance(release_id, str) or not ID_RE.fullmatch(release_id):
        raise ValueError("release_id does not match the frozen syntax")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full lowercase commit identity")

    reasons: list[str] = []
    artifacts: dict[str, object] = {}
    network_path, network_sha = verify_reference(manifest.get("network"), "network", reasons)
    if network_path is not None and network_sha is not None:
        artifacts["network"] = {
            "sha256": network_sha,
            "size": network_path.stat().st_size,
        }
        if network_path.stat().st_size != EXPECTED_NATIVE_SIZE:
            reasons.append("network: byte size does not match AliceNative-v1")

    receipt_specs = (
        ("native_qualification", manifest.get("native_qualification")),
        ("exact_los_receipt", manifest.get("exact_los_receipt")),
        ("fixed_final_receipt", manifest.get("fixed_final_receipt")),
        ("openbench_shadow_receipt", manifest.get("openbench_shadow_receipt")),
    )
    loaded_receipts: dict[str, dict[str, object]] = {}
    receipt_hashes: dict[str, str] = {}
    for label, reference in receipt_specs:
        path, expected = verify_reference(reference, label, reasons)
        if path is not None and expected is not None:
            try:
                loaded_receipts[label] = load_object(path)
                receipt_hashes[label] = expected
            except (UnicodeDecodeError, ValueError) as error:
                reasons.append(f"{label}: invalid JSON: {error}")

    if (
        network_path is not None
        and network_sha is not None
        and "native_qualification" in loaded_receipts
    ):
        verify_native_qualification(
            loaded_receipts["native_qualification"],
            network_path,
            network_sha,
            reasons,
        )
    acceptance_identities: dict[str, dict[str, object]] = {}
    if "exact_los_receipt" in loaded_receipts:
        identity = verify_acceptance(
            loaded_receipts["exact_los_receipt"],
            "exact-los",
            "exact LOS receipt",
            reasons,
        )
        if identity is not None:
            acceptance_identities["exact"] = identity
    if "fixed_final_receipt" in loaded_receipts:
        identity = verify_acceptance(
            loaded_receipts["fixed_final_receipt"],
            "fixed-final",
            "fixed final receipt",
            reasons,
        )
        if identity is not None:
            acceptance_identities["fixed"] = identity
    if (
        "exact" in acceptance_identities
        and "fixed" in acceptance_identities
        and {
            key: value
            for key, value in acceptance_identities["exact"].items()
            if key != "opening_seed"
        }
        != {
            key: value
            for key, value in acceptance_identities["fixed"].items()
            if key != "opening_seed"
        }
    ):
        reasons.append("local batteries do not share one pinned input identity")
    if network_sha is not None:
        for label, identity in acceptance_identities.items():
            engines = identity.get("engines")
            contender = engines[0] if isinstance(engines, list) and engines else None
            reference = (
                engines[1]
                if isinstance(engines, list) and len(engines) == 2
                else None
            )
            if (
                not isinstance(contender, dict)
                or contender.get("evaluator") != "Native"
                or contender.get("network_sha256") != network_sha
            ):
                reasons.append(
                    f"{label} local battery does not bind the candidate native network"
                )
            if (
                not isinstance(reference, dict)
                or reference.get("evaluator") != "Legacy"
                or reference.get("network_sha256") != FROZEN_LEGACY_NETWORK_SHA256
                or reference.get("binary_sha256") != FROZEN_LEGACY_BINARY_SHA256
            ):
                reasons.append(
                    f"{label} local battery does not bind the frozen historical reference"
                )
    binaries = manifest.get("binaries")
    if not isinstance(binaries, list) or len(binaries) != 4:
        reasons.append("binaries: exactly four release roles are required")
        binaries = []
    seen_roles: set[str] = set()
    seen_paths: set[Path] = set()
    seen_binary_sha256: set[str] = set()
    binary_sha256_by_role: dict[str, str] = {}
    for index, binary in enumerate(binaries):
        label = f"binaries[{index}]"
        if not isinstance(binary, dict):
            reasons.append(f"{label}: entry is not an object")
            continue
        try:
            exact_fields(binary, {"role", "artifact", "triple_bench", "load_failures"}, label)
        except ValueError as error:
            reasons.append(str(error))
            continue
        role = binary.get("role")
        if not isinstance(role, str) or role not in BINARY_ROLES or role in seen_roles:
            reasons.append(f"{label}: release role is missing or duplicated")
            continue
        seen_roles.add(role)
        binary_path, binary_sha = verify_reference(binary.get("artifact"), role, reasons)
        if binary_path is not None:
            if binary_path in seen_paths:
                reasons.append(f"{role}: artifact path is reused")
            seen_paths.add(binary_path)
            verify_binary_role(binary_path, role, source_commit, reasons)
        if binary_sha is not None:
            if binary_sha in seen_binary_sha256:
                reasons.append(f"{role}: binary SHA-256 is reused across release roles")
            seen_binary_sha256.add(binary_sha)
        if binary_path is not None and binary_sha is not None:
            artifacts[role] = {"sha256": binary_sha, "size": binary_path.stat().st_size}
            binary_sha256_by_role[role] = binary_sha
        bench_path, _bench_sha = verify_reference(
            binary.get("triple_bench"), f"{role} triple bench", reasons
        )
        load_path, _load_sha = verify_reference(
            binary.get("load_failures"), f"{role} load failures", reasons
        )
        if (
            bench_path is not None
            and binary_path is not None
            and binary_sha is not None
            and network_path is not None
            and network_sha is not None
        ):
            try:
                verify_triple_bench(
                    load_object(bench_path),
                    binary_path,
                    binary_sha,
                    network_path,
                    network_sha,
                    role,
                    reasons,
                )
            except (UnicodeDecodeError, ValueError) as error:
                reasons.append(f"{role}: invalid triple-bench JSON: {error}")
        if (
            load_path is not None
            and binary_sha is not None
            and network_path is not None
            and network_sha is not None
        ):
            try:
                verify_load_failures(
                    load_object(load_path),
                    binary_sha,
                    network_path,
                    network_sha,
                    role,
                    reasons,
                )
            except (UnicodeDecodeError, ValueError) as error:
                reasons.append(f"{role}: invalid load-failure JSON: {error}")
    if seen_roles != BINARY_ROLES:
        reasons.append("binaries: the four platform and architecture roles are incomplete")
    release_binary_sha256 = set(binary_sha256_by_role.values())
    for label, identity in acceptance_identities.items():
        engines = identity.get("engines")
        contender = engines[0] if isinstance(engines, list) and engines else None
        if (
            not isinstance(contender, dict)
            or contender.get("binary_sha256") not in release_binary_sha256
        ):
            reasons.append(
                f"{label} local battery does not bind a candidate release binary"
            )
    if network_sha is not None and "openbench_shadow_receipt" in loaded_receipts:
        verify_openbench_shadow(
            loaded_receipts["openbench_shadow_receipt"],
            source_commit,
            network_sha,
            binary_sha256_by_role,
            reasons,
        )

    reasons = sorted(set(reasons))
    authorized = not reasons
    return {
        "schema": "alice-release-evidence-v1",
        "release_id": release_id,
        "source_commit": source_commit,
        "status": "ready" if authorized else "blocked",
        "strength_release_authorized": authorized,
        "blocking_reasons": reasons,
        "artifacts": artifacts,
        "receipt_sha256": receipt_hashes,
        "publication_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = audit_release_candidate(args.manifest.resolve())
    write_create_only_json(args.output, receipt)
    return 0 if receipt["strength_release_authorized"] else 3


if __name__ == "__main__":
    sys.exit(main())
