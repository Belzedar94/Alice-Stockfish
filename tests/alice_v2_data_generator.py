#!/usr/bin/env python3
"""Integration gate for the isolated ALICE_V2_CHUNK_V1 self-play generator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence


TIMEOUT_SECONDS = 180.0
REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK_SCHEMA = REPO_ROOT / "schemas" / "alice-v2-chunk-v1.schema.json"
RECORD_SCHEMA = REPO_ROOT / "schemas" / "alice-v2-record-v1.schema.json"
TARGET_CONTRACT = REPO_ROOT / "schemas" / "alice-v2-target-contract-v1.json"
CHUNK_SCHEMA_SHA256 = "8D762FE64B0B8BB196BAFACA26C3514CB338B29BEF2B4AFB268F1974A2E0C6AE"
RECORD_SCHEMA_SHA256 = "1FCDD61BD11FC5428C3A8726FD21F853A6446BD6A7C3025DC49A34C936CE85AC"
TARGET_CONTRACT_SHA256 = "FFFE33048B9E3FBB91E378F6208CAB0FE1CADB06C6AF29A7493C0496F4E296CD"
TEST_RUN_CONFIG_SHA256 = "B" * 64
RUN2RL_SHA256 = "9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9"
CAPABILITY_JSON = (
    '{"schema":"ALICE_V2_CHUNK_V1","schema_sha256":'
    '"8D762FE64B0B8BB196BAFACA26C3514CB338B29BEF2B4AFB268F1974A2E0C6AE",'
    '"record_schema":{"schema":"ALICEV2_RECORD_V1","schema_sha256":'
    '"1FCDD61BD11FC5428C3A8726FD21F853A6446BD6A7C3025DC49A34C936CE85AC"},'
    '"target_contract":{"schema":"ALICE_V2_TARGET_CONTRACT_V1","sha256":'
    '"FFFE33048B9E3FBB91E378F6208CAB0FE1CADB06C6AF29A7493C0496F4E296CD"},'
    '"write":true,"record_size":84,"header_size":4096}'
)
TEST_SEED = "1"
TEST_BOOK = "8/6|Q1/8/8/8/8/k7/2K5 w - - 0 1\n"


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise AssertionError(f"{label} does not exist: {resolved}")
    return resolved


def controlled_temp_parent() -> Path:
    configured = Path(tempfile.gettempdir()).resolve()
    if not any(character.isspace() for character in str(configured)):
        return configured
    fallback = Path(configured.anchor) / "alice-sf-test-tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    if any(character.isspace() for character in str(fallback)):
        raise AssertionError(f"controlled test path contains whitespace: {fallback}")
    return fallback


def run_engine(
    binary: Path,
    commands: Sequence[str],
    *,
    expect_success: bool,
    cwd: Path = REPO_ROOT,
) -> str:
    result = subprocess.run(
        [str(binary)],
        input="\n".join((*commands, "")),
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"{binary.name} failed with exit code {result.returncode}:\n{result.stdout}"
        )
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"{binary.name} accepted an invalid command:\n{result.stdout}")
    return result.stdout


def import_decoder() -> Any:
    path = REPO_ROOT / "tools" / "alice_v2_chunk.py"
    spec = importlib.util.spec_from_file_location("alice_v2_chunk", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import decoder from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["alice_v2_chunk"] = module
    spec.loader.exec_module(module)
    return module


def generator_command(
    *,
    output: Path,
    network: Path,
    network_sha256: str,
    producer_sha256: str,
    book: Path,
    book_sha256: str,
    threads: int = 1,
) -> str:
    return (
        "alice_v2_generate_training_data "
        f"threads {threads} hash 16 network {network.as_posix()} network_sha256 {network_sha256} "
        f"producer_sha256 {producer_sha256} count 1 seed {TEST_SEED} "
        f"run_config_sha256 {TEST_RUN_CONFIG_SHA256} base_seed 1 total_records 1 "
        "records_per_chunk 1 train_chunks 1 validation_chunks 0 test_chunks 0 "
        f"book {book.as_posix()} book_sha256 {book_sha256} out {output.as_posix()} "
        "depth 1 nodes 0 random_move_min_ply 0 random_move_max_ply 0 "
        "random_move_count 0 random_multi_pv 1 random_multi_pv_diff 0 "
        "write_min_ply 0 write_max_ply 2 max_game_ply 4 set_recommended_uci_options"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--normal-engine", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    return parser.parse_args(argv)


def expect_decoder_failure(decoder: Any, path: Path) -> None:
    try:
        decoder.audit_chunk(path, qualification=True)
    except decoder.AliceV2ChunkError:
        return
    raise AssertionError(f"decoder accepted corrupted chunk: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generator = require_file(args.generator, "data-generator binary")
    normal_engine = require_file(args.normal_engine, "normal engine")
    network_source = require_file(args.network, "run2rl network")
    if hashlib.sha256(network_source.read_bytes()).hexdigest().upper() != RUN2RL_SHA256:
        raise AssertionError("run2rl network SHA-256 mismatch")
    if hashlib.sha256(CHUNK_SCHEMA.read_bytes()).hexdigest().upper() != CHUNK_SCHEMA_SHA256:
        raise AssertionError("container schema SHA-256 mismatch")
    if hashlib.sha256(RECORD_SCHEMA.read_bytes()).hexdigest().upper() != RECORD_SCHEMA_SHA256:
        raise AssertionError("record schema SHA-256 mismatch")
    if hashlib.sha256(TARGET_CONTRACT.read_bytes()).hexdigest().upper() != TARGET_CONTRACT_SHA256:
        raise AssertionError("target contract SHA-256 mismatch")

    producer_sha256 = hashlib.sha256(generator.read_bytes()).hexdigest().upper()
    decoder = import_decoder()

    isolation = run_engine(normal_engine, ("alice_v2_data_schema", "quit"), expect_success=True)
    if CAPABILITY_JSON in isolation:
        raise AssertionError("normal playing binary exposes the private DATAGEN capability")

    with tempfile.TemporaryDirectory(prefix="alice-v2-chunk-", dir=controlled_temp_parent()) as raw:
        root = Path(raw).resolve()
        network = root / "alice_run2rl_e40_l09.nnue"
        shutil.copyfile(network_source, network)
        book = root / "openings.epd"
        book.write_text(TEST_BOOK, encoding="ascii", newline="\n")
        book_sha256 = hashlib.sha256(book.read_bytes()).hexdigest().upper()

        first = root / "first.bin"
        second = root / "second.bin"
        command = generator_command(
            output=first,
            network=network,
            network_sha256=RUN2RL_SHA256,
            producer_sha256=producer_sha256,
            book=book,
            book_sha256=book_sha256,
        )
        output = run_engine(
            generator, ("alice_v2_data_schema", command, "quit"), expect_success=True
        )
        if output.splitlines().count(CAPABILITY_JSON) != 1:
            raise AssertionError(f"schema capability handshake mismatch:\n{output}")
        if output.splitlines().count("INFO: alice_v2_generate_training_data finished.") != 1:
            raise AssertionError(f"completion marker mismatch:\n{output}")

        report = decoder.audit_chunk(
            first,
            expected_book=book_sha256,
            expected_run_config=TEST_RUN_CONFIG_SHA256,
            qualification=True,
        )
        if report["record_count"] != 1:
            raise AssertionError(f"unexpected record count: {report}")
        if report["coverage"].get("white_other_features", 0) == 0:
            raise AssertionError(f"fixture did not preserve a real OTHER feature: {report}")
        if report["coverage"].get("white_same_features", 0) == 0:
            raise AssertionError(f"fixture did not preserve a real SAME feature: {report}")
        decoded = decoder.decode_record_payload(first.read_bytes()[4096:])
        if decoded["fen"] != TEST_BOOK.strip():
            raise AssertionError(
                "record round-trip did not preserve the complete Alice state: "
                f"{decoded['fen']!r} != {TEST_BOOK.strip()!r}"
            )
        if decoded["score"] == -32768 or decoded["game_ply"] != 0:
            raise AssertionError(f"record round-trip scalar fields are invalid: {decoded}")

        second_command = generator_command(
            output=second,
            network=network,
            network_sha256=RUN2RL_SHA256,
            producer_sha256="A" * 64,
            book=book,
            book_sha256=book_sha256,
            threads=24,
        )
        run_engine(generator, (second_command, "quit"), expect_success=True)
        if first.read_bytes() != second.read_bytes():
            raise AssertionError(
                "same logical chunk differs across producer/resource attestations"
            )

        run_engine(generator, (command, "quit"), expect_success=False)
        if first.read_bytes() != second.read_bytes():
            raise AssertionError("existing-output rejection changed a finalized file")

        wrong_network = generator_command(
            output=root / "wrong-network.bin",
            network=network,
            network_sha256="0" * 64,
            producer_sha256=producer_sha256,
            book=book,
            book_sha256=book_sha256,
        )
        run_engine(generator, (wrong_network, "quit"), expect_success=False)
        wrong_book = generator_command(
            output=root / "wrong-book.bin",
            network=network,
            network_sha256=RUN2RL_SHA256,
            producer_sha256=producer_sha256,
            book=book,
            book_sha256="0" * 64,
        )
        run_engine(generator, (wrong_book, "quit"), expect_success=False)

        header_corrupt = root / "header-corrupt.bin"
        payload = bytearray(first.read_bytes())
        payload[32] ^= 1
        header_corrupt.write_bytes(payload)
        expect_decoder_failure(decoder, header_corrupt)

        record_corrupt = root / "record-corrupt.bin"
        payload = bytearray(first.read_bytes())
        payload[4096] ^= 1
        record_corrupt.write_bytes(payload)
        expect_decoder_failure(decoder, record_corrupt)

    print("ALICE_V2_CHUNK_V1 generator integration gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
