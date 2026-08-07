"""Audit an Alice release candidate without publishing or modifying artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

if __package__:
    from .alice_acceptance.aggregate import (
        aggregate_input_identity,
        validate_aggregate_receipt,
    )
    from .alice_acceptance.evidence import sha256_file, write_create_only_json
    from .alice_acceptance.runner_adapter import parse_strict_json
else:
    from alice_acceptance.aggregate import (
        aggregate_input_identity,
        validate_aggregate_receipt,
    )
    from alice_acceptance.evidence import sha256_file, write_create_only_json
    from alice_acceptance.runner_adapter import parse_strict_json


EXPECTED_NATIVE_SIZE = 220_315_747
FROZEN_LEGACY_BINARY_SHA256 = (
    "b70afe03ec9a67258cd7b5b848c46fc9e5c83f53b9f2825e9a5946feefb59599"
)
FROZEN_LEGACY_NETWORK_SHA256 = (
    "9f9e557015a55c0a6981db64e1f3044dedb91fd8a8c1a6d4f3c45d0eee91fbd9"
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
    "dataset_manifest_sha256",
    "dataset_position_count",
    "checkpoint_sha256",
    "export_receipt_sha256",
    "checkpoint_file_element_count",
    "checkpoint_file_element_mismatches",
    "file_engine_position_count",
    "file_engine_centipawn_difference",
    "incremental_full_position_count",
    "incremental_full_mismatches",
    "network_parameter_nonzero_count",
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
    "pairs",
    "inversions",
    "invalid_pairs",
    "adjudication",
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


def verify_binary_role(path: Path, role: str, reasons: list[str]) -> None:
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


def verify_native_qualification(
    receipt: dict[str, object], network_sha256: str, reasons: list[str]
) -> None:
    required_gates = {f"G{index}" for index in range(1, 9)}
    gates = receipt.get("gates")
    if (
        set(receipt) != QUALIFICATION_FIELDS
        or receipt.get("schema") != "alice-native-qualification-v1"
        or receipt.get("status") != "qualified"
        or receipt.get("network_sha256") != network_sha256
        or receipt.get("network_kind") != "trained"
        or not isinstance(receipt.get("training_run_id"), str)
        or not ID_RE.fullmatch(str(receipt.get("training_run_id")))
        or not isinstance(gates, dict)
        or set(gates) != required_gates
        or any(value != "PASS" for value in gates.values())
    ):
        reasons.append("native qualification: trained G1-G8 evidence is incomplete")
    for field in (
        "dataset_manifest_sha256",
        "checkpoint_sha256",
        "export_receipt_sha256",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            reasons.append(f"native qualification: {field} is missing")
    for field in (
        "checkpoint_file_element_mismatches",
        "file_engine_centipawn_difference",
        "incremental_full_mismatches",
    ):
        if receipt.get(field) != 0:
            reasons.append(f"native qualification: {field} is not zero")
    for field in (
        "dataset_position_count",
        "checkpoint_file_element_count",
        "file_engine_position_count",
        "incremental_full_position_count",
        "network_parameter_nonzero_count",
    ):
        value = receipt.get(field)
        if type(value) is not int or value <= 0:
            reasons.append(f"native qualification: {field} is not positive")


def verify_triple_bench(
    receipt: dict[str, object],
    binary_sha256: str,
    network_sha256: str,
    role: str,
    reasons: list[str],
) -> None:
    signatures = receipt.get("signatures")
    if (
        receipt.get("schema") != "alice-triple-bench-v1"
        or receipt.get("binary_sha256") != binary_sha256
        or receipt.get("network_sha256") != network_sha256
        or not isinstance(signatures, list)
        or len(signatures) != 3
        or any(not isinstance(value, str) or not value for value in signatures)
        or len(set(signatures)) != 1
    ):
        reasons.append(f"{role}: triple bench is not reproducible")


def verify_load_failures(
    receipt: dict[str, object],
    binary_sha256: str,
    network_sha256: str,
    role: str,
    reasons: list[str],
) -> None:
    cases = receipt.get("cases")
    expected_cases = {"missing", "corrupt", "incompatible"}
    if (
        receipt.get("schema") != "alice-load-failure-matrix-v1"
        or receipt.get("binary_sha256") != binary_sha256
        or receipt.get("network_sha256") != network_sha256
        or not isinstance(cases, dict)
        or set(cases) != expected_cases
    ):
        reasons.append(f"{role}: load-failure matrix is incomplete")
        return
    for name, case in cases.items():
        if (
            not isinstance(case, dict)
            or case.get("exit_nonzero") is not True
            or case.get("fallback_observed") is not False
            or case.get("search_result_published") is not False
        ):
            reasons.append(f"{role}: {name} load failure did not fail closed")


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
        if (
            result.get("pairs") != 200
            or result.get("inversions") != 0
            or result.get("invalid_pairs") != 0
            or result.get("adjudication") != ["800/4", "40/8/10"]
        ):
            reasons.append(f"OpenBench shadow preset {preset} is not clean")


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

    if network_sha is not None and "native_qualification" in loaded_receipts:
        verify_native_qualification(
            loaded_receipts["native_qualification"], network_sha, reasons
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
        and acceptance_identities["exact"] != acceptance_identities["fixed"]
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
            verify_binary_role(binary_path, role, reasons)
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
        if bench_path is not None and binary_sha is not None and network_sha is not None:
            try:
                verify_triple_bench(
                    load_object(bench_path), binary_sha, network_sha, role, reasons
                )
            except (UnicodeDecodeError, ValueError) as error:
                reasons.append(f"{role}: invalid triple-bench JSON: {error}")
        if load_path is not None and binary_sha is not None and network_sha is not None:
            try:
                verify_load_failures(
                    load_object(load_path), binary_sha, network_sha, role, reasons
                )
            except (UnicodeDecodeError, ValueError) as error:
                reasons.append(f"{role}: invalid load-failure JSON: {error}")
    if seen_roles != BINARY_ROLES:
        reasons.append("binaries: the four platform and architecture roles are incomplete")
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
