#!/usr/bin/env python3
"""Authenticate a completed official v41 Alice NNUE-v2 50M corpus."""

from __future__ import annotations

import argparse
import bz2
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import alice_v2_chunk as chunk
from alice_v2_run_config import command_template


PUBLICATION_SCHEMA = "openbench-datagen-publication-manifest-v41"
PUBLICATION_CONTRACT_SCHEMA = "openbench-datagen-publication-contract-v41"
PUBLICATION_LEASE_SCHEMA = "openbench-datagen-publication-lease-v41"
PUBLICATION_RECEIPT_SCHEMA = "openbench-datagen-publication-receipt-v41"
NETWORK_BYTES = 47_721_376
ENGINE_BENCH = 202_963
BOOK_NAME = "ALICE_openings.epd"
NETWORK_ID = "9F9E5570"
OFFICIAL_REPO = "https://github.com/Belzedar94/Alice-Stockfish"
BOOK_SOURCE = (
    "https://github.com/Belzedar94/Alice-Stockfish/releases/download/"
    "openbench-assets-v1/ALICE_openings.epd.zip"
)

MANIFEST_FIELDS = {
    "test_id",
    "engine",
    "producer_commit",
    "producer_builds",
    "producer_artifact_required",
    "producer_contract_sha256",
    "environment",
    "total_count",
    "positions_per_chunk",
    "base_seed",
    "chunks",
    "schema",
    "version",
    "protocol",
    "publication_contract",
    "publication_contract_sha256",
    "manifest_sha256",
}
CONTRACT_FIELDS = {
    "schema",
    "protocol",
    "campaign_id",
    "external_workload_id",
    "role",
    "cohort",
    "engine",
    "network",
    "book",
    "generation",
    "producer",
    "teacher",
    "syzygy",
}
ENGINE_FIELDS = {"name", "repo", "source", "requested_ref", "commit", "bench", "options"}
NETWORK_FIELDS = {"name", "openbench_id", "sha256", "bytes"}
BOOK_FIELDS = {"kind", "name", "source", "text_sha256", "raw_sha256"}
GENERATION_FIELDS = {
    "command",
    "command_sha256",
    "total_count",
    "positions_per_chunk",
    "base_seed",
    "seed_method",
}
PRODUCER_FIELDS = {"required", "contract_sha256"}
TEACHER_FIELDS = {"mode"}
SYZYGY_FIELDS = {
    "required",
    "family",
    "max",
    "manifest_sha256",
    "environment_contract_sha256",
}
ENVIRONMENT_FIELDS = {
    "tablebase_required",
    "contract_sha256",
    "tablebase_family",
    "tablebase_max",
    "tablebase_manifest_sha256",
    "teacher_mode",
}
BUILD_FIELDS = {"sha256", "bytes", "commit"}
CHUNK_FIELDS = {
    "index",
    "seed",
    "positions",
    "artifact_sha256",
    "artifact_bytes",
    "producer_sha256",
    "producer_bytes",
    "producer_commit",
    "environment_lease",
    "environment_lease_sha256",
    "environment_receipt",
    "environment_receipt_sha256",
}
LEASE_FIELDS = {
    "schema",
    "protocol",
    "test_id",
    "chunk_idx",
    "attempt",
    "machine_id",
    "publication_contract_sha256",
    "environment_contract_sha256",
    "tablebase",
    "teacher_mode",
}
RECEIPT_FIELDS = LEASE_FIELDS | {"environment_lease_sha256", "artifact", "producer"}
TABLEBASE_FIELDS = {"required", "family", "required_max", "worker_max", "manifest_sha256"}
ARTIFACT_FIELDS = {"sha256", "bytes"}


class AliceV2CorpusError(ValueError):
    """Raised when corpus or OpenBench evidence does not match the sealed run."""


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AliceV2CorpusError(message)


def require_exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} field set is not exact")


def require_sha(value: object, label: str) -> str:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
        f"{label} is not a SHA-256",
    )
    return value


def load_manifest(path: str | Path) -> tuple[dict[str, object], str]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AliceV2CorpusError(f"Cannot load official manifest: {exc}") from exc
    require(isinstance(document, dict), "Official manifest must be one JSON object")
    observed = require_sha(document.get("manifest_sha256"), "manifest_sha256").lower()
    unhashed = dict(document)
    del unhashed["manifest_sha256"]
    require(canonical_sha256(unhashed) == observed, "Official manifest self-hash mismatch")
    return document, observed


def validate_publication_contract(
    manifest: dict[str, object], config: dict[str, object], run_digest: str
) -> dict[str, object]:
    require(manifest.get("schema") == PUBLICATION_SCHEMA, "Manifest is not v41 publication evidence")
    require(manifest.get("version") == 1, "Manifest version is not 1")
    require(manifest.get("protocol") == 41, "Manifest protocol is not 41")
    contract = manifest.get("publication_contract")
    require(isinstance(contract, dict), "Manifest lacks the publication contract")
    require_exact_keys(contract, CONTRACT_FIELDS, "Publication contract")
    contract_sha = require_sha(
        manifest.get("publication_contract_sha256"), "publication_contract_sha256"
    ).lower()
    require(canonical_sha256(contract) == contract_sha, "Publication-contract self-hash mismatch")

    openbench = config["openbench"]
    require(isinstance(openbench, dict), "Authenticated run config has no OpenBench identity")
    require(contract.get("schema") == PUBLICATION_CONTRACT_SCHEMA, "Publication schema mismatch")
    require(contract.get("protocol") == 41, "Publication protocol mismatch")
    require(contract.get("campaign_id") == config["campaign"], "Campaign differs from run config")
    require(
        contract.get("external_workload_id") == openbench["external_workload_id"],
        "External workload differs from run config",
    )
    require(contract.get("role") == openbench["role"], "Publication role differs from run config")
    require(contract.get("cohort") == openbench["cohort"], "Publication cohort differs from run config")

    engine = contract.get("engine")
    require(isinstance(engine, dict), "Publication engine identity is missing")
    require_exact_keys(engine, ENGINE_FIELDS, "Publication engine")
    require(engine.get("name") == "Alice-Stockfish", "Unexpected OpenBench engine entry")
    require(engine.get("repo") == OFFICIAL_REPO, "Producer repository differs from the official repository")
    require(engine.get("source") == OFFICIAL_REPO, "Producer source differs from the official repository")
    require(engine.get("requested_ref") == config["source_commit"], "Requested producer ref is not the sealed commit")
    require(engine.get("commit") == config["source_commit"], "Producer commit differs from run config")
    require(engine.get("bench") == ENGINE_BENCH, "Producer bench differs from the frozen Alice bench")
    require(engine.get("options") == "", "DATAGEN engine options must be empty")

    network = contract.get("network")
    require(isinstance(network, dict), "Publication network identity is missing")
    require_exact_keys(network, NETWORK_FIELDS, "Publication network")
    require(
        str(network.get("sha256", "")).upper() == config["network_sha256"],
        "Publication network hash differs from run config",
    )
    require(network.get("bytes") == NETWORK_BYTES, "Publication network size mismatch")
    require(network.get("name") == "alice_run2rl_e40_l09.nnue", "Publication network name mismatch")
    require(network.get("openbench_id") == NETWORK_ID, "Publication network ID mismatch")

    book = contract.get("book")
    require(isinstance(book, dict), "Publication book identity is missing")
    require_exact_keys(book, BOOK_FIELDS, "Publication book")
    require(
        book.get("kind") == "file" and book.get("name") == BOOK_NAME,
        "Publication book mismatch",
    )
    require(
        str(book.get("raw_sha256", "")).upper() == config["book_sha256"],
        "Publication book bytes differ from run config",
    )
    require(
        str(book.get("text_sha256", "")).upper() == config["book_sha256"],
        "Publication book text identity differs from run config",
    )
    require(book.get("source") == BOOK_SOURCE, "Publication book source mismatch")

    generation = contract.get("generation")
    require(isinstance(generation, dict), "Publication generation identity is missing")
    require_exact_keys(generation, GENERATION_FIELDS, "Publication generation")
    expected_command = command_template(run_digest, int(config["base_seed"]))
    require(generation.get("command") == expected_command, "OpenBench command differs from sealed preimage")
    require(
        generation.get("command_sha256") == hashlib.sha256(expected_command.encode()).hexdigest(),
        "OpenBench command hash mismatch",
    )
    require(generation.get("total_count") == 50_000_000, "OpenBench total count mismatch")
    require(generation.get("positions_per_chunk") == 1_000_000, "OpenBench chunk size mismatch")
    require(generation.get("base_seed") == config["base_seed"], "OpenBench base seed mismatch")
    require(
        generation.get("seed_method") == "base-plus-chunk-index-v1",
        "OpenBench seed method mismatch",
    )
    producer = contract.get("producer")
    require(isinstance(producer, dict), "Publication producer identity is missing")
    require_exact_keys(producer, PRODUCER_FIELDS, "Publication producer")
    require(
        producer.get("required") is True
        and re.fullmatch(r"[0-9a-f]{64}", str(producer.get("contract_sha256", "")))
        is not None,
        "OpenBench producer contract is missing",
    )
    teacher = contract.get("teacher")
    require(isinstance(teacher, dict), "Publication teacher identity is missing")
    require_exact_keys(teacher, TEACHER_FIELDS, "Publication teacher")
    require(teacher == {"mode": None}, "Unexpected teacher mode")
    syzygy = contract.get("syzygy")
    require(isinstance(syzygy, dict), "Publication tablebase identity is missing")
    require_exact_keys(syzygy, SYZYGY_FIELDS, "Publication tablebase")
    require(
        syzygy.get("required") is False
        and syzygy.get("family") is None
        and syzygy.get("max") == 0
        and syzygy.get("manifest_sha256") is None,
        "Unexpected tablebase identity",
    )
    require_sha(syzygy.get("environment_contract_sha256"), "syzygy.environment_contract_sha256")
    return contract


def validate_environment_evidence(
    entry: dict[str, object],
    contract_sha: str,
    environment_contract_sha: str,
    test_id: int,
    producer_builds: dict[str, dict[str, object]],
) -> None:
    lease = entry.get("environment_lease")
    receipt = entry.get("environment_receipt")
    require(isinstance(lease, dict) and isinstance(receipt, dict), "Chunk lacks v41 lease/receipt")
    require_exact_keys(lease, LEASE_FIELDS, "Chunk environment lease")
    require_exact_keys(receipt, RECEIPT_FIELDS, "Chunk environment receipt")
    lease_sha = require_sha(entry.get("environment_lease_sha256"), "environment lease hash").lower()
    receipt_sha = require_sha(entry.get("environment_receipt_sha256"), "environment receipt hash").lower()
    require(canonical_sha256(lease) == lease_sha, "Chunk environment lease self-hash mismatch")
    require(canonical_sha256(receipt) == receipt_sha, "Chunk environment receipt self-hash mismatch")
    require(lease.get("schema") == PUBLICATION_LEASE_SCHEMA, "Chunk lease schema mismatch")
    require(receipt.get("schema") == PUBLICATION_RECEIPT_SCHEMA, "Chunk receipt schema mismatch")
    require(lease.get("protocol") == 41 and receipt.get("protocol") == 41, "Chunk evidence protocol mismatch")
    require(lease.get("test_id") == test_id and receipt.get("test_id") == test_id, "Chunk evidence test ID mismatch")
    require(
        type(lease.get("attempt")) is int
        and lease["attempt"] > 0
        and receipt.get("attempt") == lease["attempt"],
        "Chunk evidence attempt mismatch",
    )
    require(
        type(lease.get("machine_id")) is int
        and lease["machine_id"] > 0
        and receipt.get("machine_id") == lease["machine_id"],
        "Chunk evidence machine mismatch",
    )
    require(lease.get("publication_contract_sha256") == contract_sha, "Chunk lease is for another publication")
    require(receipt.get("publication_contract_sha256") == contract_sha, "Chunk receipt is for another publication")
    require(
        lease.get("environment_contract_sha256") == environment_contract_sha
        and receipt.get("environment_contract_sha256") == environment_contract_sha,
        "Chunk environment contract mismatch",
    )
    require(receipt.get("environment_lease_sha256") == lease_sha, "Chunk receipt does not bind its lease")
    require(
        receipt.get("environment_contract_sha256")
        == lease.get("environment_contract_sha256"),
        "Chunk receipt environment contract mismatch",
    )
    expected_tablebase = {
        "required": False,
        "family": None,
        "required_max": 0,
        "worker_max": 0,
        "manifest_sha256": None,
    }
    require(isinstance(lease.get("tablebase"), dict), "Chunk lease tablebase identity missing")
    require(isinstance(receipt.get("tablebase"), dict), "Chunk receipt tablebase identity missing")
    require_exact_keys(lease["tablebase"], TABLEBASE_FIELDS, "Chunk lease tablebase")
    require_exact_keys(receipt["tablebase"], TABLEBASE_FIELDS, "Chunk receipt tablebase")
    require(lease.get("tablebase") == expected_tablebase, "Chunk lease has unexpected tablebases")
    require(receipt.get("tablebase") == expected_tablebase, "Chunk receipt has unexpected tablebases")
    require(lease.get("teacher_mode") is None, "Chunk lease has an unexpected teacher")
    require(receipt.get("teacher_mode") is None, "Chunk receipt has an unexpected teacher")
    require(lease.get("chunk_idx") == entry.get("index"), "Chunk lease index mismatch")
    require(receipt.get("chunk_idx") == entry.get("index"), "Chunk receipt index mismatch")
    artifact = receipt.get("artifact")
    require(isinstance(artifact, dict), "Chunk receipt has no artifact identity")
    require_exact_keys(artifact, ARTIFACT_FIELDS, "Chunk receipt artifact")
    require(artifact.get("sha256") == entry.get("artifact_sha256"), "Chunk receipt artifact hash mismatch")
    require(artifact.get("bytes") == entry.get("artifact_bytes"), "Chunk receipt artifact size mismatch")
    producer_sha = str(entry.get("producer_sha256", ""))
    require(re.fullmatch(r"[0-9a-f]{64}", producer_sha) is not None, "Chunk producer hash is malformed")
    require(producer_sha in producer_builds, "Chunk references an unauthenticated producer build")
    require(entry.get("producer_commit") == producer_builds[producer_sha]["commit"], "Chunk producer commit mismatch")
    require(entry.get("producer_bytes") == producer_builds[producer_sha]["bytes"], "Chunk producer size mismatch")
    require(
    receipt.get("producer")
        == {
            "sha256": producer_sha,
            "bytes": entry.get("producer_bytes"),
            "commit": entry.get("producer_commit"),
        },
        "Chunk receipt producer identity mismatch",
    )


def audit_corpus(
    run_config: str | Path, official_manifest: str | Path, artifacts: str | Path
) -> dict[str, object]:
    config, run_digest = chunk._load_run_config(run_config)
    manifest, manifest_sha = load_manifest(official_manifest)
    require_exact_keys(manifest, MANIFEST_FIELDS, "Official manifest")
    contract = validate_publication_contract(manifest, config, run_digest)
    contract_sha = str(manifest["publication_contract_sha256"])
    test_id = manifest.get("test_id")
    require(type(test_id) is int and test_id > 0, "Manifest test ID is invalid")
    require(manifest.get("engine") == "Alice-Stockfish", "Manifest engine mismatch")
    require(manifest.get("producer_commit") == config["source_commit"], "Manifest producer commit mismatch")
    require(manifest.get("producer_artifact_required") is True, "Producer artifacts are not required")
    require(manifest.get("producer_contract_sha256") == contract["producer"]["contract_sha256"], "Producer contract drift")
    require(manifest.get("total_count") == 50_000_000, "Manifest total count mismatch")
    require(manifest.get("positions_per_chunk") == 1_000_000, "Manifest chunk count mismatch")
    require(manifest.get("base_seed") == config["base_seed"], "Manifest base seed mismatch")
    environment = manifest.get("environment")
    require(isinstance(environment, dict), "Manifest environment identity is missing")
    require_exact_keys(environment, ENVIRONMENT_FIELDS, "Manifest environment")
    environment_contract_sha = require_sha(
        environment.get("contract_sha256"), "environment.contract_sha256"
    ).lower()
    require(
        environment_contract_sha == contract["syzygy"]["environment_contract_sha256"],
        "Manifest environment differs from the publication contract",
    )
    require(
        environment
        == {
            "tablebase_required": False,
            "contract_sha256": environment_contract_sha,
            "tablebase_family": None,
            "tablebase_max": 0,
            "tablebase_manifest_sha256": None,
            "teacher_mode": None,
        },
        "Manifest environment is not the sealed no-tablebase environment",
    )

    builds = manifest.get("producer_builds")
    require(isinstance(builds, list) and bool(builds), "Manifest has no authenticated producer builds")
    producer_builds: dict[str, dict[str, object]] = {}
    for build in builds:
        require(isinstance(build, dict), "Producer-build entry is malformed")
        require_exact_keys(build, BUILD_FIELDS, "Producer build")
        sha = require_sha(build.get("sha256"), "producer build hash").lower()
        require(sha not in producer_builds, "Duplicate producer-build identity")
        require(build.get("commit") == config["source_commit"], "Producer-build commit mismatch")
        require(type(build.get("bytes")) is int and build["bytes"] > 0, "Producer-build size is invalid")
        producer_builds[sha] = build

    entries = manifest.get("chunks")
    require(isinstance(entries, list) and len(entries) == 50, "Manifest must contain exactly 50 chunks")
    by_index: dict[int, dict[str, object]] = {}
    for entry in entries:
        require(isinstance(entry, dict), "Chunk manifest entry is malformed")
        require_exact_keys(entry, CHUNK_FIELDS, "Chunk manifest entry")
        index = entry.get("index")
        require(type(index) is int and 0 <= index < 50 and index not in by_index, "Chunk index set is invalid")
        require(entry.get("seed") == int(config["base_seed"]) + index, "Chunk seed mismatch")
        require(entry.get("positions") == 1_000_000, "Chunk position count mismatch")
        validate_environment_evidence(
            entry, contract_sha, environment_contract_sha, test_id, producer_builds
        )
        by_index[index] = entry
    require(set(by_index) == set(range(50)), "Chunk index set is incomplete")

    artifact_root = Path(artifacts)
    aggregate: Counter[str] = Counter()
    raw_records = 0
    chunks: list[dict[str, object]] = []
    for index in range(50):
        entry = by_index[index]
        artifact_sha = require_sha(entry.get("artifact_sha256"), "chunk artifact hash").lower()
        archive = artifact_root / f"{artifact_sha}.bz2"
        try:
            compressed = archive.read_bytes()
        except OSError as exc:
            raise AliceV2CorpusError(f"Cannot read chunk {index} archive: {exc}") from exc
        require(len(compressed) == entry["artifact_bytes"], f"Chunk {index} archive size mismatch")
        require(hashlib.sha256(compressed).hexdigest() == artifact_sha, f"Chunk {index} archive hash mismatch")
        try:
            decompressor = bz2.BZ2Decompressor()
            raw = decompressor.decompress(compressed)
        except OSError as exc:
            raise AliceV2CorpusError(f"Chunk {index} archive is not valid bzip2: {exc}") from exc
        require(decompressor.eof and not decompressor.unused_data, f"Chunk {index} archive has trailing data")
        report = chunk._audit_chunk_bytes(
            archive,
            raw,
            expected_source=str(config["source_commit"]),
            expected_book=str(config["book_sha256"]),
            expected_run_config=run_digest,
            run_config=run_config,
            qualification=False,
            require_native_coverage=False,
        )
        require(report["partition"]["chunk_index"] == index, f"Chunk {index} payload index mismatch")
        require(report["record_count"] == entry["positions"], f"Chunk {index} payload count mismatch")
        raw_records += int(report["record_count"])
        aggregate.update(report["coverage"])
        chunks.append(
            {
                "index": index,
                "archive_sha256": artifact_sha,
                "raw_chunk_sha256": report["chunk_sha256"],
                "records": report["record_count"],
            }
        )

    acceptance = config["acceptance"]
    require(raw_records == acceptance["record_count"], "Aggregate record count mismatch")
    missing = [name for name in acceptance["minimum_nonzero_counters"] if aggregate[name] == 0]
    require(not missing, "Aggregate coverage gate failed: " + ", ".join(missing))
    both = aggregate["records_with_same_and_other_for_both_perspectives"]
    require(
        both * 1_000_000 >= raw_records * acceptance[
            "records_with_same_and_other_for_both_perspectives_min_ppm"
        ],
        "Aggregate SAME/OTHER coverage is below the preregistered ppm gate",
    )
    return {
        "schema": "ALICE_V2_CORPUS_AUDIT_V1",
        "passed": True,
        "run_config_sha256": run_digest,
        "official_manifest_sha256": manifest_sha,
        "publication_contract_sha256": contract_sha,
        "source_commit": config["source_commit"],
        "record_count": raw_records,
        "chunk_count": len(chunks),
        "producer_builds": builds,
        "coverage": dict(sorted(aggregate.items())),
        "chunks": chunks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit_corpus(args.run_config, args.official_manifest, args.artifacts)
        payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(payload)
    except (AliceV2CorpusError, chunk.AliceV2ChunkError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(payload).hexdigest().upper()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
