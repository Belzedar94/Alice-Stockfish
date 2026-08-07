"""Create a fail-closed three-control acceptance receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sys

from .evidence import canonical_json_bytes, sha256_file, write_create_only_json
from .policy import TIMING_CONTROLS
from .runner_adapter import parse_strict_json
from .statistics import paired_statistics


CONTROLS = ("VSTC", "STC", "LTC")
FIXED_GAMES = {"VSTC": 400, "STC": 300, "LTC": 200}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTROL_FIELDS = {
    "schema",
    "run_id",
    "status",
    "times",
    "policy",
    "inputs",
    "result",
    "sealed_snapshot",
    "sealed_snapshot_sha256",
    "artifacts",
    "strength_release_authorized",
}
SEAL_FIELDS = {
    "schema",
    "control",
    "mode",
    "attempt_ordinal",
    "admitted_pairs",
    "scored_games",
    "wld",
    "pentanomial",
    "statistics",
    "stop_reason",
    "conclusion",
}
RESULT_FIELDS = {
    "schema",
    "control",
    "mode",
    "state",
    "attempted_pairs",
    "attempted_games",
    "runner_complete_pairs",
    "admitted_pairs",
    "discarded_pairs",
    "excluded_after_seal_pairs",
    "excluded_after_terminal_pairs",
    "scored_games",
    "wld",
    "pentanomial",
    "statistics",
    "abort_counts",
    "stop_reason",
    "conclusion",
}
AGGREGATE_FIELDS = {
    "schema",
    "run_id",
    "mode",
    "status",
    "times",
    "inputs",
    "controls",
    "conclusion",
    "artifacts",
    "strength_gate_eligible",
}
AGGREGATE_CONTROL_FIELDS = {
    "receipt_sha256",
    "receipt",
}
POLICY_FIELDS = {
    "control",
    "mode",
    "base_ms",
    "increment_ms",
    "pair_workers",
    "engine_threads",
    "hash_mib",
    "external_adjudication",
    "commit_order",
    "maximum_scored_games",
    "maximum_attempted_games",
    "target_admitted_games",
}
INVENTORY_FIELDS = {
    "schema",
    "source_definition_sha256",
    "canonical_definition_sha256",
    "book_sha256",
    "opening_seed",
    "pair_worker_sha256",
    "pair_worker_path",
    "pair_core_sha256",
    "pair_core_path",
    "source_worker_definition_sha256",
    "worker_definition_sha256",
    "normalized_worker_configuration_sha256",
    "engines",
}
ENGINE_IDENTITY_FIELDS = {
    "role",
    "binary_sha256",
    "network_sha256",
    "evaluator",
    "options_sha256",
}
CONTROL_ARTIFACT_FIELDS = {
    "openings_jsonl_sha256",
    "status_jsonl_sha256",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def nonnegative_int(value: object, field: str, control: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{control} {field} is not a non-negative integer")
    return value


def validate_control_policy(
    policy: dict[str, object], control: str, mode: str
) -> None:
    if set(policy) != POLICY_FIELDS:
        raise ValueError(f"{control} policy fields do not match the contract")
    base_ms, increment_ms = TIMING_CONTROLS[control]
    expected = {
        "control": control,
        "mode": mode,
        "base_ms": base_ms,
        "increment_ms": increment_ms,
        "pair_workers": 2,
        "engine_threads": 1,
        "hash_mib": 512,
        "external_adjudication": "disabled",
        "commit_order": "attempt-ordinal",
        "maximum_scored_games": 64000 if mode == "exact-los" else None,
        "maximum_attempted_games": 64000 if mode == "exact-los" else None,
        "target_admitted_games": FIXED_GAMES[control]
        if mode == "fixed-final"
        else None,
    }
    numeric_fields = {
        field for field, expected_value in expected.items() if type(expected_value) is int
    }
    if any(type(policy.get(field)) is not int for field in numeric_fields):
        raise ValueError(f"{control} policy numeric fields are not canonical integers")
    if canonical_json_bytes(policy) != canonical_json_bytes(expected):
        raise ValueError(f"{control} policy values do not match the frozen control")


def validate_input_inventory(
    inventory: dict[str, object], control: str
) -> dict[str, object]:
    if set(inventory) != INVENTORY_FIELDS:
        raise ValueError(f"{control} input inventory fields do not match the contract")
    if inventory.get("schema") != "alice-acceptance-input-inventory-v1":
        raise ValueError(f"{control} input inventory schema is unsupported")
    path_fields = {"pair_worker_path", "pair_core_path"}
    hash_fields = INVENTORY_FIELDS.difference(
        {"schema", "engines", "opening_seed"}, path_fields
    )
    for field in hash_fields:
        value = inventory.get(field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValueError(f"{control} input inventory {field} is not canonical")
    for artifact in ("pair_worker", "pair_core"):
        path_value = inventory.get(f"{artifact}_path")
        expected_sha = inventory.get(f"{artifact}_sha256")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise ValueError(f"{control} input inventory {artifact} path is not absolute")
        path = Path(path_value).resolve()
        if not path.is_file():
            raise ValueError(f"{control} input inventory {artifact} artifact is missing")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"{control} input inventory {artifact} SHA-256 mismatch")
    opening_seed = inventory.get("opening_seed")
    if (
        type(opening_seed) is not int
        or opening_seed < 0
        or opening_seed > (2**64 - 1)
    ):
        raise ValueError(f"{control} input inventory opening_seed is not canonical")
    engines = inventory.get("engines")
    if not isinstance(engines, list) or len(engines) != 2:
        raise ValueError(f"{control} input inventory requires two engines")
    expected_roles = ("contender", "reference")
    for index, engine in enumerate(engines):
        if not isinstance(engine, dict) or set(engine) != ENGINE_IDENTITY_FIELDS:
            raise ValueError(f"{control} engine identity fields do not match the contract")
        if engine.get("role") != expected_roles[index]:
            raise ValueError(f"{control} engine roles are not canonical")
        binary_sha256 = engine.get("binary_sha256")
        if not isinstance(binary_sha256, str) or not SHA256_RE.fullmatch(binary_sha256):
            raise ValueError(f"{control} engine binary identity is not canonical")
        options_sha256 = engine.get("options_sha256")
        if not isinstance(options_sha256, str) or not SHA256_RE.fullmatch(options_sha256):
            raise ValueError(f"{control} engine option identity is not canonical")
        evaluator = engine.get("evaluator")
        network_sha256 = engine.get("network_sha256")
        if evaluator not in ("Legacy", "Native", "Zero"):
            raise ValueError(f"{control} engine evaluator is unsupported")
        if evaluator == "Zero":
            if network_sha256 is not None:
                raise ValueError(f"{control} zero evaluator claims a network")
        elif not isinstance(network_sha256, str) or not SHA256_RE.fullmatch(
            network_sha256
        ):
            raise ValueError(f"{control} network identity is not canonical")
    return inventory


def input_identity(inventory: dict[str, object]) -> dict[str, object]:
    return {
        "book_sha256": inventory["book_sha256"],
        "opening_seed": inventory["opening_seed"],
        "pair_worker_sha256": inventory["pair_worker_sha256"],
        "pair_core_sha256": inventory["pair_core_sha256"],
        "normalized_worker_configuration_sha256": inventory[
            "normalized_worker_configuration_sha256"
        ],
        "engines": inventory["engines"],
    }


def validate_control_result(
    result: dict[str, object], control: str, mode: str
) -> None:
    if set(result) != RESULT_FIELDS:
        raise ValueError(f"{control} result fields do not match the contract")
    if result.get("control") != control or result.get("mode") != mode:
        raise ValueError(f"{control} result identity mismatch")
    if result.get("schema") != "alice-acceptance-controller-v1":
        raise ValueError(f"{control} result schema is unsupported")
    if result.get("discarded_pairs") != 0 or result.get("abort_counts") != {}:
        raise ValueError(f"{control} has nonzero abort evidence")
    attempted_pairs = nonnegative_int(
        result.get("attempted_pairs"), "attempted_pairs", control
    )
    attempted_games = nonnegative_int(
        result.get("attempted_games"), "attempted_games", control
    )
    runner_complete_pairs = nonnegative_int(
        result.get("runner_complete_pairs"), "runner_complete_pairs", control
    )
    admitted_pairs = nonnegative_int(
        result.get("admitted_pairs"), "admitted_pairs", control
    )
    discarded_pairs = nonnegative_int(
        result.get("discarded_pairs"), "discarded_pairs", control
    )
    excluded_after_seal_pairs = nonnegative_int(
        result.get("excluded_after_seal_pairs"),
        "excluded_after_seal_pairs",
        control,
    )
    excluded_after_terminal_pairs = nonnegative_int(
        result.get("excluded_after_terminal_pairs"),
        "excluded_after_terminal_pairs",
        control,
    )
    scored_games = nonnegative_int(
        result.get("scored_games"), "scored_games", control
    )
    if scored_games <= 0 or scored_games % 2:
        raise ValueError(f"{control} scored-game count is invalid")
    if (
        attempted_games != attempted_pairs * 2
        or runner_complete_pairs != attempted_pairs
        or scored_games != admitted_pairs * 2
        or runner_complete_pairs
        != admitted_pairs
        + discarded_pairs
        + excluded_after_seal_pairs
        + excluded_after_terminal_pairs
    ):
        raise ValueError(f"{control} pair and game counts are inconsistent")
    pentanomial = result.get("pentanomial")
    if (
        not isinstance(pentanomial, list)
        or len(pentanomial) != 5
        or any(type(count) is not int or count < 0 for count in pentanomial)
        or sum(pentanomial) != admitted_pairs
    ):
        raise ValueError(f"{control} pentanomial counts are inconsistent")
    wld = result.get("wld")
    if not isinstance(wld, dict) or set(wld) != {"wins", "losses", "draws"}:
        raise ValueError(f"{control} WLD fields do not match the contract")
    wins = nonnegative_int(wld.get("wins"), "wins", control)
    losses = nonnegative_int(wld.get("losses"), "losses", control)
    draws = nonnegative_int(wld.get("draws"), "draws", control)
    if (
        wins + losses + draws != scored_games
        or 2 * wins + draws
        != sum(index * count for index, count in enumerate(pentanomial))
    ):
        raise ValueError(f"{control} WLD totals contradict the pentanomial")
    statistics = result.get("statistics")
    expected_statistics = paired_statistics(pentanomial)
    if canonical_json_bytes(statistics) != canonical_json_bytes(expected_statistics):
        raise ValueError(f"{control} statistics do not reproduce from the pentanomial")
    if mode == "exact-los":
        if (
            result.get("state") != "SEALED_PASS"
            or result.get("stop_reason") != "los-100.0"
            or result.get("conclusion") != "PASS"
            or not isinstance(statistics, dict)
            or statistics.get("los_percent_display") != "100.0"
            or scored_games <= 100
            or scored_games > 64000
            or attempted_games > 64000
        ):
            raise ValueError(f"{control} did not satisfy the exact LOS gate")
    elif mode == "fixed-final":
        if (
            result.get("state") != "FIXED_COMPLETE"
            or result.get("stop_reason") != "fixed-target"
            or result.get("conclusion") != "FIXED_COMPLETE"
            or scored_games != FIXED_GAMES[control]
        ):
            raise ValueError(f"{control} did not complete the fixed final gate")
    else:
        raise ValueError("unsupported aggregate mode")


def validate_sealed_snapshot(
    seal: object,
    sealed_snapshot_sha256: str,
    result: dict[str, object],
    control: str,
    mode: str,
) -> None:
    if not isinstance(seal, dict) or set(seal) != SEAL_FIELDS:
        raise ValueError(f"{control} sealed snapshot fields do not match the contract")
    actual_sha256 = hashlib.sha256(canonical_json_bytes(seal)).hexdigest()
    if actual_sha256 != sealed_snapshot_sha256:
        raise ValueError(f"{control} sealed snapshot SHA-256 does not match its payload")
    if (
        seal.get("schema") != "alice-acceptance-seal-v1"
        or seal.get("control") != control
        or seal.get("mode") != mode
    ):
        raise ValueError(f"{control} sealed snapshot identity is inconsistent")
    for field in (
        "admitted_pairs",
        "scored_games",
        "wld",
        "pentanomial",
        "statistics",
        "stop_reason",
        "conclusion",
    ):
        if canonical_json_bytes(seal.get(field)) != canonical_json_bytes(
            result.get(field)
        ):
            raise ValueError(
                f"{control} sealed snapshot does not match final result field {field}"
            )
    admitted_pairs = result.get("admitted_pairs")
    attempt_ordinal = seal.get("attempt_ordinal")
    if (
        type(admitted_pairs) is not int
        or type(attempt_ordinal) is not int
        or attempt_ordinal != admitted_pairs - 1
    ):
        raise ValueError(f"{control} sealed snapshot attempt ordinal is inconsistent")


def validate_control_receipt(
    value: dict[str, object], control: str, mode: str
) -> dict[str, object]:
    if set(value) != CONTROL_FIELDS:
        raise ValueError(f"{control} control receipt fields do not match the contract")
    if value.get("schema") != "alice-control-receipt-v1":
        raise ValueError(f"{control} uses an unsupported control receipt")
    if value.get("status") != "finalized":
        raise ValueError(f"{control} is not finalized")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"{control} source run_id is invalid")
    if value.get("strength_release_authorized") is not False:
        raise ValueError(f"{control} control receipt overclaims release authority")
    sealed_snapshot_sha256 = value.get("sealed_snapshot_sha256")
    if (
        not isinstance(sealed_snapshot_sha256, str)
        or not SHA256_RE.fullmatch(sealed_snapshot_sha256)
    ):
        raise ValueError(f"{control} lacks a canonical immutable seal identity")
    policy = value.get("policy")
    inventory = value.get("inputs")
    result = value.get("result")
    artifacts = value.get("artifacts")
    if (
        not isinstance(policy, dict)
        or not isinstance(inventory, dict)
        or not isinstance(result, dict)
        or not isinstance(value.get("times"), dict)
        or not isinstance(artifacts, dict)
    ):
        raise ValueError(f"{control} lacks policy, input, result, or artifact evidence")
    if (
        set(artifacts) != CONTROL_ARTIFACT_FIELDS
        or any(
            not isinstance(value, str) or not SHA256_RE.fullmatch(value)
            for value in artifacts.values()
        )
        or len(set(artifacts.values())) != 2
    ):
        raise ValueError(f"{control} control artifact hashes are incomplete")
    validate_control_policy(policy, control, mode)
    validate_input_inventory(inventory, control)
    validate_control_result(result, control, mode)
    validate_sealed_snapshot(
        value.get("sealed_snapshot"),
        sealed_snapshot_sha256,
        result,
        control,
        mode,
    )
    return value


def load_control_receipt(path: Path, control: str, mode: str) -> dict[str, object]:
    payload = path.read_bytes()
    value = parse_strict_json(payload)
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{control} control receipt is not canonical JSON")
    return validate_control_receipt(value, control, mode)


def validate_aggregate_receipt(
    value: dict[str, object], expected_mode: str | None = None
) -> dict[str, object]:
    if set(value) != AGGREGATE_FIELDS:
        raise ValueError("aggregate receipt fields do not match the contract")
    mode = value.get("mode")
    if mode not in ("exact-los", "fixed-final") or (
        expected_mode is not None and mode != expected_mode
    ):
        raise ValueError("aggregate receipt mode mismatch")
    assert isinstance(mode, str)
    expected_conclusion = "PASS" if mode == "exact-los" else "FIXED_COMPLETE"
    if (
        value.get("schema") != "alice-acceptance-receipt-v1"
        or value.get("status") != "finalized"
        or value.get("conclusion") != expected_conclusion
        or value.get("strength_gate_eligible") is not True
    ):
        raise ValueError("aggregate receipt is not an eligible final result")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("aggregate receipt run_id is invalid")
    if not isinstance(value.get("times"), dict) or not isinstance(
        value.get("artifacts"), dict
    ):
        raise ValueError("aggregate receipt lacks time or artifact evidence")
    controls = value.get("controls")
    inputs = value.get("inputs")
    if not isinstance(controls, dict) or set(controls) != set(CONTROLS):
        raise ValueError("aggregate receipt lacks three controls")
    if not isinstance(inputs, dict) or set(inputs) != {"control_receipt_sha256"}:
        raise ValueError("aggregate receipt input fields do not match the contract")
    input_hashes = inputs.get("control_receipt_sha256")
    if not isinstance(input_hashes, dict) or set(input_hashes) != set(CONTROLS):
        raise ValueError("aggregate receipt input hashes are incomplete")
    frozen_identity: dict[str, object] | None = None
    for control in CONTROLS:
        item = controls[control]
        if not isinstance(item, dict) or set(item) != AGGREGATE_CONTROL_FIELDS:
            raise ValueError(f"{control} aggregate fields do not match the contract")
        receipt_sha256 = item.get("receipt_sha256")
        receipt = item.get("receipt")
        if not isinstance(receipt, dict):
            raise ValueError(f"{control} embedded control receipt is missing")
        embedded_sha256 = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
        if (
            not isinstance(receipt_sha256, str)
            or not SHA256_RE.fullmatch(receipt_sha256)
            or input_hashes.get(control) != receipt_sha256
            or embedded_sha256 != receipt_sha256
        ):
            raise ValueError(f"{control} aggregate identities are inconsistent")
        validate_control_receipt(receipt, control, mode)
        inventory = receipt["inputs"]
        assert isinstance(inventory, dict)
        identity = input_identity(inventory)
        if frozen_identity is None:
            frozen_identity = identity
        elif identity != frozen_identity:
            raise ValueError("aggregate controls do not share one pinned input identity")
    return value


def aggregate_input_identity(value: dict[str, object]) -> dict[str, object]:
    validated = validate_aggregate_receipt(value)
    controls = validated["controls"]
    assert isinstance(controls, dict)
    first = controls[CONTROLS[0]]
    assert isinstance(first, dict)
    receipt = first["receipt"]
    assert isinstance(receipt, dict)
    inventory = receipt["inputs"]
    assert isinstance(inventory, dict)
    return input_identity(inventory)


def aggregate_receipts(
    run_id: str,
    mode: str,
    receipt_paths: dict[str, Path],
) -> dict[str, object]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("aggregate run_id does not match the frozen syntax")
    if mode not in ("exact-los", "fixed-final"):
        raise ValueError("aggregate mode must be exact-los or fixed-final")
    if set(receipt_paths) != set(CONTROLS):
        raise ValueError("aggregate requires VSTC, STC, and LTC receipts")
    controls: dict[str, object] = {}
    inputs: dict[str, object] = {}
    for control in CONTROLS:
        path = receipt_paths[control].resolve()
        receipt = load_control_receipt(path, control, mode)
        receipt_sha256 = sha256_file(path)
        controls[control] = {
            "receipt_sha256": receipt_sha256,
            "receipt": receipt,
        }
        inputs[control] = receipt_sha256
    conclusion = "PASS" if mode == "exact-los" else "FIXED_COMPLETE"
    aggregate = {
        "schema": "alice-acceptance-receipt-v1",
        "run_id": run_id,
        "mode": mode,
        "status": "finalized",
        "times": {"assembled_utc": utc_now()},
        "inputs": {"control_receipt_sha256": inputs},
        "controls": controls,
        "conclusion": conclusion,
        "artifacts": {},
        "strength_gate_eligible": True,
    }
    return validate_aggregate_receipt(aggregate, mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("exact-los", "fixed-final"), required=True)
    parser.add_argument("--vstc", type=Path, required=True)
    parser.add_argument("--stc", type=Path, required=True)
    parser.add_argument("--ltc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = aggregate_receipts(
        args.run_id,
        args.mode,
        {"VSTC": args.vstc, "STC": args.stc, "LTC": args.ltc},
    )
    write_create_only_json(args.output, receipt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
