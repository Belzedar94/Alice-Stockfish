"""Run one frozen Alice acceptance timing control."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import sys

from .controller import AcceptanceController
from .evidence import (
    CreateOnlySeal,
    canonical_json_bytes,
    sha256_file,
    write_create_only_json,
)
from .openings import OpeningSchedule
from .policy import exact_los_policy, fixed_final_policy
from .runner_adapter import (
    PersistentWorkerClient,
    parse_strict_json,
    validate_worker_response,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FROZEN_LEGACY_NETWORK_NAME = "alice_run2rl_e40_l09.nnue"
FROZEN_COMMON_OPTIONS = {
    "Threads": "1",
    "Hash": "512",
    "Move Overhead": "10",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def copy_create_only(source: Path, target: Path, expected_sha256: str) -> str:
    if not source.is_file():
        raise ValueError(f"snapshot source is not a regular file: {source}")
    if not SHA256_RE.fullmatch(expected_sha256):
        raise ValueError(f"snapshot source lacks a lowercase SHA-256: {source}")
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as input_file, open(target, "xb") as output_file:
        shutil.copyfileobj(input_file, output_file, 1024 * 1024)
        output_file.flush()
        os.fsync(output_file.fileno())
    if sha256_file(target) != actual:
        raise ValueError(f"snapshot SHA-256 mismatch for {source}")
    return actual


def append_jsonl(path: Path, value: dict[str, object]) -> None:
    with open(path, "ab") as output:
        output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def load_object(path: Path) -> dict[str, object]:
    value = parse_strict_json(path.read_bytes())
    return value


def require_exact_fields(
    value: dict[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise ValueError(f"{label} fields mismatch; missing={missing}, extra={extra}")


def require_absolute_file(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an absolute file path")
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label} must be an existing absolute file path")
    return path.resolve()


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def validate_run_definition(value: dict[str, object]) -> dict[str, object]:
    require_exact_fields(
        value,
        {"schema", "run_id", "control", "mode", "seed", "book", "pair_worker"},
        "run definition",
    )
    if value.get("schema") != "alice-acceptance-run-definition-v1":
        raise ValueError("unsupported acceptance-run definition schema")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id does not match the frozen identifier syntax")
    if value.get("control") not in ("VSTC", "STC", "LTC"):
        raise ValueError("control is not frozen")
    if value.get("mode") not in ("exact-los", "fixed-final"):
        raise ValueError("mode is not frozen")
    seed = value.get("seed")
    if type(seed) is not int or seed < 0 or seed > (2**64 - 1):
        raise ValueError("opening seed must be an unsigned 64-bit integer")

    book = value.get("book")
    pair_worker = value.get("pair_worker")
    if not isinstance(book, dict) or not isinstance(pair_worker, dict):
        raise ValueError("run definition requires book and pair_worker objects")
    require_exact_fields(book, {"path", "sha256"}, "book")
    require_absolute_file(book.get("path"), "book.path")
    require_sha256(book.get("sha256"), "book.sha256")
    require_exact_fields(
        pair_worker,
        {
            "script",
            "script_sha256",
            "core",
            "core_sha256",
            "definition",
            "definition_sha256",
            "request_timeout_seconds",
        },
        "pair_worker",
    )
    for field in ("script", "core", "definition"):
        require_absolute_file(pair_worker.get(field), f"pair_worker.{field}")
    require_sha256(pair_worker.get("script_sha256"), "pair_worker.script_sha256")
    require_sha256(pair_worker.get("core_sha256"), "pair_worker.core_sha256")
    require_sha256(
        pair_worker.get("definition_sha256"), "pair_worker.definition_sha256"
    )
    timeout = pair_worker.get("request_timeout_seconds")
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        raise ValueError("pair_worker.request_timeout_seconds must be finite and positive")
    return value


def reject_d_evidence_root(path: Path) -> None:
    if os.name == "nt" and path.drive.lower() == "d:":
        raise ValueError("acceptance evidence must not be written to D:")


def validate_pair_worker_definition(
    value: dict[str, object],
) -> tuple[list[dict[str, object]], tuple[str, str]]:
    if value.get("schema") != "alice-pair-worker-definition-v1":
        raise ValueError("unsupported pair-worker definition schema")
    engine_values = value.get("engines")
    if not isinstance(engine_values, list) or len(engine_values) != 2:
        raise ValueError("pair-worker definition requires two engines")

    engines: list[dict[str, object]] = []
    names: list[str] = []
    base_engine_fields = {
        "path",
        "binary_sha256",
        "cwd",
        "name",
        "evaluator",
        "network_sha256",
        "time_control",
        "options",
    }
    for index, item in enumerate(engine_values):
        if not isinstance(item, dict):
            raise ValueError("engine definition must be an object")
        evaluator = item.get("evaluator")
        if evaluator not in ("Legacy", "Native", "Zero"):
            raise ValueError("engine evaluator must be Legacy, Native, or Zero")
        expected_fields = set(base_engine_fields)
        if evaluator in ("Legacy", "Native"):
            expected_fields.add("network_path")
        require_exact_fields(item, expected_fields, f"engines[{index}]")
        name = item.get("name")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 128
            or not name.isascii()
            or any(
                character == '"' or not character.isprintable()
                for character in name
            )
        ):
            raise ValueError("each engine requires a PGN-safe ASCII name")
        if name in names:
            raise ValueError("the two engine names must be distinct")
        options = item.get("options")
        if (
            not isinstance(options, dict)
            or any(not isinstance(key, str) for key in options)
            or any(not isinstance(option, str) for option in options.values())
        ):
            raise ValueError("engine options must be a string-to-string object")
        expected_options = dict(FROZEN_COMMON_OPTIONS)
        if evaluator == "Legacy":
            expected_options.update(
                {
                    "Use NNUE": "true",
                    "Alice Evaluation": "Legacy",
                    "Alice_Frozen_Network": "true",
                    "EvalFile": item.get("network_path"),
                }
            )
        elif evaluator == "Native":
            expected_options.update(
                {
                    "Use NNUE": "true",
                    "Alice Evaluation": "Native",
                    "Alice Native SHA256": item.get("network_sha256"),
                    "Alice Native EvalFile": item.get("network_path"),
                }
            )
        else:
            if item.get("network_sha256") != "":
                raise ValueError("the Zero evaluator requires an empty network SHA-256")
            expected_options.update(
                {
                    "Use NNUE": "false",
                    "Alice Evaluation": "Zero",
                }
            )
        require_exact_fields(
            options, set(expected_options), f"engines[{index}].options"
        )
        if options != expected_options:
            raise ValueError(
                f"engines[{index}] options do not match the frozen evaluator policy"
            )
        engines.append(item)
        names.append(name)
    return engines, (names[0], names[1])


def prepare_snapshots(
    run_definition: dict[str, object],
    evidence_root: Path,
    source_definition_sha256: str,
    canonical_definition_sha256: str,
) -> tuple[Path, Path, Path, dict[str, object]]:
    inputs = evidence_root / "inputs"
    snapshots = inputs / "snapshots"
    runner_config = run_definition.get("pair_worker")
    book_config = run_definition.get("book")
    if not isinstance(runner_config, dict) or not isinstance(book_config, dict):
        raise ValueError("run definition requires book and pair_worker objects")
    opening_seed = run_definition.get("seed")
    if (
        type(opening_seed) is not int
        or opening_seed < 0
        or opening_seed > (2**64 - 1)
    ):
        raise ValueError("opening seed must be an unsigned 64-bit integer")

    worker_definition_source = require_absolute_file(
        runner_config.get("definition"), "pair_worker.definition"
    )
    worker_definition_source_sha = sha256_file(worker_definition_source)
    if worker_definition_source_sha != require_sha256(
        runner_config.get("definition_sha256"), "pair_worker.definition_sha256"
    ):
        raise ValueError("pair-worker definition SHA-256 mismatch")
    worker_definition = load_object(worker_definition_source)
    engines, engine_names = validate_pair_worker_definition(worker_definition)

    worker_source = require_absolute_file(runner_config.get("script"), "pair_worker.script")
    core_source = require_absolute_file(runner_config.get("core"), "pair_worker.core")
    worker_snapshot = snapshots / "runner" / "uci_pair_worker.py"
    core_snapshot = snapshots / "runner" / "uci_pair_runner.py"
    worker_sha = copy_create_only(
        worker_source,
        worker_snapshot,
        require_sha256(runner_config.get("script_sha256"), "pair_worker.script_sha256"),
    )
    core_sha = copy_create_only(
        core_source,
        core_snapshot,
        require_sha256(runner_config.get("core_sha256"), "pair_worker.core_sha256"),
    )

    book_source = require_absolute_file(book_config.get("path"), "book.path")
    book_sha = require_sha256(book_config.get("sha256"), "book.sha256")
    book_snapshot = snapshots / "book" / book_source.name
    copy_create_only(book_source, book_snapshot, book_sha)

    network_snapshots: dict[tuple[str, str], Path] = {}
    engine_inventory = []
    normalized_engines = []
    for index, item in enumerate(engines):
        name = engine_names[index]
        binary_source = require_absolute_file(
            item.get("path"), f"engines[{index}].path"
        )
        binary_sha = require_sha256(
            item.get("binary_sha256"), f"engines[{index}].binary_sha256"
        )
        binary_snapshot = snapshots / f"engine-{index + 1}" / binary_source.name
        copy_create_only(binary_source, binary_snapshot, binary_sha)
        shutil.copymode(binary_source, binary_snapshot)
        if os.name != "nt" and not os.access(binary_snapshot, os.X_OK):
            raise ValueError(f"engines[{index}].path is not executable")
        item["path"] = str(binary_snapshot)
        item["cwd"] = str(binary_snapshot.parent)

        evaluator = item.get("evaluator")
        options = item.get("options")
        assert isinstance(options, dict)
        normalized_options = dict(options)
        network_sha = str(item.get("network_sha256", ""))
        network_snapshot = None
        if evaluator in ("Legacy", "Native"):
            network_sha = require_sha256(
                network_sha, f"engines[{index}].network_sha256"
            )
            network_source = require_absolute_file(
                item.get("network_path"), f"engines[{index}].network_path"
            )
            if (
                evaluator == "Legacy"
                and network_source.name != FROZEN_LEGACY_NETWORK_NAME
            ):
                raise ValueError(
                    f"engines[{index}].network_path must use the frozen legacy basename "
                    f"{FROZEN_LEGACY_NETWORK_NAME}"
                )
            network_key = (network_sha, network_source.name)
            if network_key not in network_snapshots:
                network_snapshot = (
                    snapshots / "networks" / network_sha / network_source.name
                )
                copy_create_only(network_source, network_snapshot, network_sha)
                network_snapshots[network_key] = network_snapshot
            else:
                network_snapshot = network_snapshots[network_key]
            item["network_path"] = str(network_snapshot)
            if evaluator == "Legacy":
                options["EvalFile"] = str(network_snapshot)
                normalized_options["EvalFile"] = f"sha256:{network_sha}"
            else:
                options["Alice Native EvalFile"] = str(network_snapshot)
                normalized_options["Alice Native EvalFile"] = f"sha256:{network_sha}"
        options_sha256 = hashlib.sha256(
            canonical_json_bytes(normalized_options)
        ).hexdigest()
        engine_inventory.append(
            {
                "role": "contender" if index == 0 else "reference",
                "binary_sha256": binary_sha,
                "network_sha256": network_sha or None,
                "evaluator": evaluator,
                "options_sha256": options_sha256,
            }
        )
        normalized_engines.append(
            {
                "role": "contender" if index == 0 else "reference",
                "name": name,
                "binary_sha256": binary_sha,
                "network_sha256": network_sha or None,
                "evaluator": evaluator,
                "options": normalized_options,
            }
        )

    normalized_worker_configuration = {
        key: value for key, value in worker_definition.items() if key != "engines"
    }
    normalized_worker_configuration["engines"] = normalized_engines
    normalized_worker_configuration_sha256 = hashlib.sha256(
        canonical_json_bytes(normalized_worker_configuration)
    ).hexdigest()

    worker_definition_snapshot = inputs / "worker-definition.json"
    write_create_only_json(worker_definition_snapshot, worker_definition)
    inventory = {
        "schema": "alice-acceptance-input-inventory-v1",
        "source_definition_sha256": source_definition_sha256,
        "canonical_definition_sha256": canonical_definition_sha256,
        "book_sha256": book_sha,
        "opening_seed": opening_seed,
        "pair_worker_sha256": worker_sha,
        "pair_worker_path": str(worker_snapshot.resolve()),
        "pair_core_sha256": core_sha,
        "pair_core_path": str(core_snapshot.resolve()),
        "source_worker_definition_sha256": worker_definition_source_sha,
        "worker_definition_sha256": sha256_file(worker_definition_snapshot),
        "normalized_worker_configuration_sha256": normalized_worker_configuration_sha256,
        "engines": engine_inventory,
    }
    write_create_only_json(inputs / "inventory.json", inventory)
    return worker_snapshot, worker_definition_snapshot, book_snapshot, inventory


def build_request(
    ordinal: int, opening: dict[str, object], pair_directory: Path
) -> dict[str, object]:
    return {
        "schema": "alice-pair-request-v1",
        "pair_ordinal": ordinal,
        "opening": {
            "book_line": opening["book_line"],
            "raw_line_sha256": opening["raw_line_sha256"],
            "fen": opening["fen"],
            "fen_sha256": opening["fen_sha256"],
        },
        "evidence_directory": str(pair_directory),
    }


def run_control(definition_path: Path, evidence_root: Path) -> dict[str, object]:
    if not definition_path.is_file():
        raise ValueError("acceptance-run definition is not a regular file")
    source_definition_sha256 = sha256_file(definition_path)
    run_definition = validate_run_definition(load_object(definition_path))
    run_id = run_definition.get("run_id")
    control = run_definition.get("control")
    mode = run_definition.get("mode")
    seed = run_definition.get("seed")
    assert isinstance(run_id, str)
    assert isinstance(control, str)
    assert isinstance(mode, str)
    assert isinstance(seed, int)
    policy = exact_los_policy(control) if mode == "exact-los" else fixed_final_policy(control)

    evidence_root = evidence_root.resolve()
    reject_d_evidence_root(evidence_root)
    evidence_root.mkdir(parents=False, exist_ok=False)
    started = utc_now()
    workers: list[PersistentWorkerClient] = []
    try:
        canonical_definition_sha256 = write_create_only_json(
            evidence_root / "definition.json", run_definition
        )
        worker_script, worker_definition, book_snapshot, inventory = prepare_snapshots(
            run_definition,
            evidence_root,
            source_definition_sha256,
            canonical_definition_sha256,
        )
        worker_definition_value = load_object(worker_definition)
        engines = worker_definition_value["engines"]
        expected_time_control = {
            "VSTC": "2+0.02",
            "STC": "10+0.1",
            "LTC": "30+0.3",
        }[control]
        for item in engines:
            if item["time_control"] != expected_time_control:
                raise ValueError("pair-worker time control does not match the policy")
        engine_names = (engines[0]["name"], engines[1]["name"])
        assert all(isinstance(name, str) for name in engine_names)

        control_root = evidence_root / "controls" / control
        pairs_root = control_root / "pairs"
        preflight_root = control_root / "preflight"
        pairs_root.mkdir(parents=True)
        preflight_root.mkdir()
        openings_path = control_root / "openings.jsonl"
        status_path = control_root / "status.jsonl"
        openings_path.open("xb").close()
        status_path.open("xb").close()
        schedule = OpeningSchedule(
            book_snapshot,
            str(run_definition["book"]["sha256"]),
            seed,
        )
        runner_config = run_definition["pair_worker"]
        timeout = float(runner_config.get("request_timeout_seconds", 7200.0))
        if timeout <= 0:
            raise ValueError("request timeout must be positive")
        for index in range(2):
            workers.append(
                PersistentWorkerClient(
                    worker_script,
                    worker_definition,
                    control_root / f"worker-{index}.stderr.bin",
                )
            )

        preflight_opening = schedule.for_ordinal(0)
        with ThreadPoolExecutor(max_workers=2) as executor:
            preflight_futures = {}
            for index, worker in enumerate(workers):
                ordinal = (2**62) + index
                pair_directory = preflight_root / f"worker-{index}"
                request = build_request(ordinal, preflight_opening, pair_directory)
                preflight_futures[
                    executor.submit(worker.request, request, timeout)
                ] = (ordinal, pair_directory, request)
            for future in as_completed(preflight_futures):
                ordinal, pair_directory, request = preflight_futures[future]
                response = future.result()
                write_create_only_json(pair_directory / "request.json", request)
                write_create_only_json(pair_directory / "response.json", response)
                result = validate_worker_response(
                    response,
                    ordinal,
                    pair_directory,
                    str(request["opening"]["fen"]),
                    engine_names,
                )
                if not result.scorable:
                    raise RuntimeError("paired preflight produced an unscorable result")

        seal = CreateOnlySeal(control_root / "seal.json")
        controller = AcceptanceController(policy, seal_callback=seal)
        controller.mark_preflighted()
        controller.start()
        with ThreadPoolExecutor(max_workers=2) as executor:
            while controller.state == "RUNNING":
                ordinals = controller.dispatch_window()
                futures = {}
                for index, ordinal in enumerate(ordinals):
                    opening = schedule.for_ordinal(ordinal)
                    append_jsonl(openings_path, opening)
                    pair_directory = pairs_root / f"{ordinal:08d}"
                    request = build_request(ordinal, opening, pair_directory)
                    futures[executor.submit(workers[index].request, request, timeout)] = (
                        ordinal,
                        pair_directory,
                        request,
                    )
                for future in as_completed(futures):
                    ordinal, pair_directory, request = futures[future]
                    response = future.result()
                    write_create_only_json(pair_directory / "request.json", request)
                    write_create_only_json(pair_directory / "response.json", response)
                    result = validate_worker_response(
                        response,
                        ordinal,
                        pair_directory,
                        str(request["opening"]["fen"]),
                        engine_names,
                    )
                    controller.submit(result)
                    append_jsonl(
                        status_path,
                        {"recorded_utc": utc_now(), **controller.summary()},
                    )

        receipt = {
            "schema": "alice-control-receipt-v1",
            "run_id": run_id,
            "status": "finalized" if controller.conclusion != "INVALID" else "invalid",
            "times": {"started_utc": started, "ended_utc": utc_now()},
            "policy": {
                "control": control,
                "mode": mode,
                "base_ms": policy.base_ms,
                "increment_ms": policy.increment_ms,
                "pair_workers": 2,
                "engine_threads": 1,
                "hash_mib": 512,
                "external_adjudication": "disabled",
                "commit_order": "attempt-ordinal",
                "maximum_scored_games": policy.maximum_scored_games,
                "maximum_attempted_games": policy.maximum_attempted_games,
                "target_admitted_games": policy.target_admitted_games,
            },
            "inputs": inventory,
            "result": controller.summary(),
            "sealed_snapshot": controller.seal_payload,
            "sealed_snapshot_sha256": seal.sha256,
            "artifacts": {
                "openings_jsonl_sha256": sha256_file(openings_path),
                "status_jsonl_sha256": sha256_file(status_path),
            },
            "strength_release_authorized": False,
        }
        write_create_only_json(control_root / "receipt.json", receipt)
        write_create_only_json(evidence_root / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        interrupted = {
            "schema": "alice-acceptance-interruption-v1",
            "started_utc": started,
            "ended_utc": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "statistical_resume_allowed": False,
        }
        try:
            write_create_only_json(evidence_root / "interrupted.json", interrupted)
        except FileExistsError:
            pass
        raise
    finally:
        for worker in workers:
            worker.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run_control(args.definition.resolve(), args.evidence_root)
    sys.stdout.buffer.write(canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
