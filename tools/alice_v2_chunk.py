#!/usr/bin/env python3
"""Fail-closed decoder and coverage auditor for ALICE_V2_CHUNK_V1."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import struct
from typing import BinaryIO, Iterator
import zlib


MAGIC = b"ALCHNK1\0"
FORMAT_VERSION = 1
HEADER_BYTES = 4096
RECORD_BYTES = 84
HEADER_PREFIX = struct.Struct("<8sHHI")
RECORD_STRUCT = struct.Struct("<64sIhHbBBBHHI")
CHUNK_SCHEMA = "ALICE_V2_CHUNK_V1"
CHUNK_SCHEMA_SHA256 = "8D762FE64B0B8BB196BAFACA26C3514CB338B29BEF2B4AFB268F1974A2E0C6AE"
RECORD_SCHEMA = "ALICEV2_RECORD_V1"
RECORD_SCHEMA_SHA256 = "1FCDD61BD11FC5428C3A8726FD21F853A6446BD6A7C3025DC49A34C936CE85AC"
TARGET_SCHEMA = "ALICE_V2_TARGET_CONTRACT_V1"
TARGET_SHA256 = "FFFE33048B9E3FBB91E378F6208CAB0FE1CADB06C6AF29A7493C0496F4E296CD"
RULES_ID = "alice-rules-v1"
RULES_HASH = 0x73F822AB
FEATURE_ID = "AliceHalfKAv2_hm_Rel-v1"
FEATURE_HASH = 0x5280C41E
RUN2RL_SHA256 = "9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9"
RUN_SCHEMA = "ALICE_V2_DATAGEN_RUN_V1"
RUN_SCHEMA_SHA256 = "4200A7FF890F47E39EDA0150C2A20D365C0530AEF7B3112E287328C953F404B9"
PRODUCTION_URL = "https://belzedar.duckdns.org"
BOOK_SHA256 = "BCD89D9FC3EA81FEB95932EB64D6B6F15AD25CC04CDCC9E0440F097CFFB8CCF6"
VALID_PIECE_CODES = frozenset(range(1, 7)) | frozenset(range(9, 15))

RUN_CONFIG_KEYS = {
    "acceptance",
    "base_seed",
    "book_sha256",
    "campaign",
    "chunk_schema_sha256",
    "generation",
    "network_sha256",
    "openbench",
    "production_url",
    "record_order",
    "record_schema_sha256",
    "records_per_chunk",
    "retry",
    "rng",
    "run_schema_sha256",
    "schema",
    "seed_derivation",
    "source_commit",
    "source_dirty",
    "split",
    "target_contract_sha256",
    "total_chunks",
    "total_records",
    "version",
}
OPENBENCH_KEYS = {
    "campaign_id",
    "cohort",
    "external_workload_id",
    "priority",
    "producer_artifact_required",
    "publication_protocol",
    "role",
    "throughput",
    "workload_size",
}
RETRY_KEYS = {
    "excluded_from_chunk_bytes",
    "logical_identity",
    "producer_attestation",
    "required_output",
    "worker_resources",
}
ACCEPTANCE_KEYS = {
    "chunk_count",
    "duplicate_chunk_indices_max",
    "identity_mismatches_max",
    "malformed_records_max",
    "minimum_nonzero_counters",
    "record_count",
    "records_with_same_and_other_for_both_perspectives_min_ppm",
    "scope",
}

TOP_LEVEL_KEYS = [
    "schema",
    "schema_sha256",
    "format_version",
    "header_bytes",
    "record_bytes",
    "record_count",
    "byte_order",
    "run_config_sha256",
    "source_commit",
    "source_dirty",
    "network",
    "book_sha256",
    "payload_sha256",
    "record_schema",
    "target_contract",
    "partition",
    "generation",
]
NETWORK_KEYS = ["schema", "sha256"]
RECORD_SCHEMA_KEYS = [
    "schema",
    "schema_sha256",
    "rules_id",
    "rules_hash",
    "feature_id",
    "feature_hash",
]
TARGET_KEYS = ["schema", "sha256"]
PARTITION_KEYS = [
    "base_seed",
    "total_records",
    "records_per_chunk",
    "chunk_index",
    "total_chunks",
    "split",
    "train_chunks",
    "validation_chunks",
    "test_chunks",
]
GENERATION_KEYS = [
    "requested_records",
    "seed",
    "search_threads",
    "hash_mb",
    "depth",
    "nodes",
    "random_move_min_ply",
    "random_move_max_ply",
    "random_move_count",
    "random_multi_pv",
    "random_multi_pv_diff",
    "write_min_ply",
    "write_max_ply",
    "max_game_ply",
    "opening_count",
]


class AliceV2ChunkError(ValueError):
    """Raised when bytes or identities violate the sealed container contract."""


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _is_upper_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789ABCDEF" for character in value)
    )


def _is_lower_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value != "0" * 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_keys(value: object, expected: list[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or list(value) != expected:
        actual = list(value) if isinstance(value, dict) else type(value).__name__
        raise AliceV2ChunkError(f"{label} key order or membership mismatch: {actual!r}")
    return value


def _read_header(stream: BinaryIO) -> tuple[dict[str, object], bytes]:
    header = stream.read(HEADER_BYTES)
    if len(header) != HEADER_BYTES:
        raise AliceV2ChunkError("Truncated ALICE_V2_CHUNK_V1 header")
    if hashlib.sha256(header[:-32]).digest() != header[-32:]:
        raise AliceV2ChunkError("Header SHA-256 mismatch")

    magic, version, header_bytes, manifest_bytes = HEADER_PREFIX.unpack_from(header)
    if (magic, version, header_bytes) != (MAGIC, FORMAT_VERSION, HEADER_BYTES):
        raise AliceV2ChunkError(
            f"Unsupported container identity: {(magic, version, header_bytes)!r}"
        )
    if manifest_bytes > HEADER_BYTES - 48:
        raise AliceV2ChunkError("Manifest length exceeds the fixed header")
    manifest_raw = header[16 : 16 + manifest_bytes]
    if any(header[16 + manifest_bytes : HEADER_BYTES - 32]):
        raise AliceV2ChunkError("Reserved header padding must be zero")
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AliceV2ChunkError(f"Manifest is not canonical UTF-8 JSON: {exc}") from exc
    canonical = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if canonical != manifest_raw:
        raise AliceV2ChunkError("Manifest JSON is not in the sealed canonical representation")
    return _require_keys(manifest, TOP_LEVEL_KEYS, "manifest"), header


def _load_run_config(path: str | Path) -> tuple[dict[str, object], str]:
    source = Path(path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest().upper()
    try:
        config = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AliceV2ChunkError(f"Run config is not UTF-8 JSON: {exc}") from exc
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if raw != canonical:
        raise AliceV2ChunkError("Run config is not in its sealed canonical representation")
    if not isinstance(config, dict):
        raise AliceV2ChunkError("Run config must be one JSON object")
    if set(config) != RUN_CONFIG_KEYS:
        raise AliceV2ChunkError("Run config field membership is not frozen")
    if (
        config.get("schema") != RUN_SCHEMA
        or config.get("version") != 1
        or config.get("run_schema_sha256") != RUN_SCHEMA_SHA256
        or config.get("production_url") != PRODUCTION_URL
        or config.get("source_dirty") is not False
        or not _is_lower_commit(config.get("source_commit"))
        or config.get("target_contract_sha256") != TARGET_SHA256
        or config.get("record_schema_sha256") != RECORD_SCHEMA_SHA256
        or config.get("chunk_schema_sha256") != CHUNK_SCHEMA_SHA256
        or config.get("network_sha256") != RUN2RL_SHA256
        or config.get("book_sha256") != BOOK_SHA256
        or config.get("total_records") != 50_000_000
        or config.get("records_per_chunk") != 1_000_000
        or config.get("total_chunks") != 50
        or config.get("seed_derivation") != "chunk_seed = base_seed + chunk_index"
        or config.get("rng")
        != "xorshift64star-v1; zero seed forbidden; modulo bounded draws"
        or config.get("record_order") != "single-thread game ordinal then game ply"
        or config.get("split")
        != {"train_chunks": 48, "validation_chunks": 1, "test_chunks": 1}
    ):
        raise AliceV2ChunkError("Run config violates the frozen 50M identity")
    base_seed = config.get("base_seed")
    if (
        not isinstance(base_seed, int)
        or isinstance(base_seed, bool)
        or not 0 < base_seed <= 2**63 - 50
    ):
        raise AliceV2ChunkError("Run config base seed is outside its frozen domain")
    campaign = config.get("campaign")
    openbench = config.get("openbench")
    expected_openbench = {
        "priority": 300,
        "throughput": 1000,
        "publication_protocol": 41,
        "producer_artifact_required": True,
        "workload_size": 1,
        "campaign_id": campaign,
        "external_workload_id": "native-v2-selfplay-50m",
        "role": "selfplay-all-splits",
        "cohort": "piece-rel-d6",
    }
    if (
        not isinstance(campaign, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", campaign) is None
        or not isinstance(openbench, dict)
        or set(openbench) != OPENBENCH_KEYS
        or openbench != expected_openbench
    ):
        raise AliceV2ChunkError("Run config OpenBench identity is not frozen")
    retry = config.get("retry")
    expected_retry = {
        "logical_identity": "source_commit + run_config_sha256 + chunk_index",
        "required_output": "identical uncompressed bytes across workers",
        "producer_attestation": "external OpenBench protocol-41 receipt",
        "worker_resources": "external OpenBench environment receipt",
        "excluded_from_chunk_bytes": ["producer_sha256", "assigned_threads"],
    }
    if (
        not isinstance(retry, dict)
        or set(retry) != RETRY_KEYS
        or retry != expected_retry
    ):
        raise AliceV2ChunkError("Run config retry identity is not frozen")
    generation = config.get("generation")
    expected_generation = {
        "search_threads": 1,
        "hash_mb": 512,
        "depth": 6,
        "nodes": 0,
        "random_move_min_ply": 1,
        "random_move_max_ply": 20,
        "random_move_count": 8,
        "random_multi_pv": 4,
        "random_multi_pv_diff": 200,
        "write_min_ply": 5,
        "write_max_ply": 400,
        "max_game_ply": 512,
        "opening_count": 38_348,
        "opening_order": "seeded Fisher-Yates; malformed or terminal entries are fatal",
        "tt_history": "reset once at chunk start and retained in deterministic game order",
        "adjudication": "none",
        "truncated_games": "discarded",
    }
    if generation != expected_generation:
        raise AliceV2ChunkError("Run config generation settings are not frozen")
    acceptance = config.get("acceptance")
    if not isinstance(acceptance, dict) or set(acceptance) != ACCEPTANCE_KEYS:
        raise AliceV2ChunkError("Run config lacks the preregistered acceptance gates")
    if (
        acceptance.get("scope") != "aggregate-over-50-authenticated-chunks"
        or acceptance.get("chunk_count") != 50
        or acceptance.get("record_count") != 50_000_000
        or acceptance.get("malformed_records_max") != 0
        or acceptance.get("duplicate_chunk_indices_max") != 0
        or acceptance.get("identity_mismatches_max") != 0
        or acceptance.get(
            "records_with_same_and_other_for_both_perspectives_min_ppm"
        )
        != 1000
        or acceptance.get("minimum_nonzero_counters")
        != [
            "white_same_features",
            "white_other_features",
            "black_same_features",
            "black_other_features",
            "white_king_board_A",
            "white_king_board_B",
            "black_king_board_A",
            "black_king_board_B",
            "move_source_board_A",
            "move_source_board_B",
            "result_-1",
            "result_+0",
            "result_+1",
            "captures",
            "promotions",
            "castling_moves",
        ]
    ):
        raise AliceV2ChunkError("Run config acceptance gates are not frozen")
    return config, digest


def _bind_manifest_to_run_config(
    manifest: dict[str, object], config: dict[str, object], digest: str
) -> None:
    if manifest["run_config_sha256"] != digest:
        raise AliceV2ChunkError("Chunk does not bind the authenticated run-config bytes")
    if manifest["source_commit"] != config["source_commit"]:
        raise AliceV2ChunkError("Chunk source does not match its run config")
    if manifest["book_sha256"] != config["book_sha256"]:
        raise AliceV2ChunkError("Chunk book does not match its run config")
    network = manifest["network"]
    if not isinstance(network, dict) or network.get("sha256") != config["network_sha256"]:
        raise AliceV2ChunkError("Chunk network does not match its run config")
    partition = manifest["partition"]
    generation = manifest["generation"]
    if not isinstance(partition, dict) or not isinstance(generation, dict):
        raise AliceV2ChunkError("Chunk partition or generation manifest is malformed")
    split = config["split"]
    configured_generation = config["generation"]
    assert isinstance(split, dict) and isinstance(configured_generation, dict)
    expected_partition = {
        "base_seed": str(config["base_seed"]),
        "total_records": config["total_records"],
        "records_per_chunk": config["records_per_chunk"],
        "total_chunks": config["total_chunks"],
        "train_chunks": split["train_chunks"],
        "validation_chunks": split["validation_chunks"],
        "test_chunks": split["test_chunks"],
    }
    for key, value in expected_partition.items():
        if partition.get(key) != value:
            raise AliceV2ChunkError(f"Chunk partition field {key} differs from run config")
    for key in (
        "search_threads",
        "hash_mb",
        "depth",
        "nodes",
        "random_move_min_ply",
        "random_move_max_ply",
        "random_move_count",
        "random_multi_pv",
        "random_multi_pv_diff",
        "write_min_ply",
        "write_max_ply",
        "max_game_ply",
        "opening_count",
    ):
        if generation.get(key) != configured_generation[key]:
            raise AliceV2ChunkError(f"Chunk generation field {key} differs from run config")


def _validate_manifest(
    manifest: dict[str, object],
    *,
    expected_source: str | None,
    expected_book: str | None,
    expected_run_config: str | None,
) -> None:
    if manifest["schema"] != CHUNK_SCHEMA or manifest["schema_sha256"] != CHUNK_SCHEMA_SHA256:
        raise AliceV2ChunkError("Container schema identity mismatch")
    if (
        manifest["format_version"] != FORMAT_VERSION
        or manifest["header_bytes"] != HEADER_BYTES
        or manifest["record_bytes"] != RECORD_BYTES
        or manifest["byte_order"] != "little"
    ):
        raise AliceV2ChunkError("Container layout manifest mismatch")
    if not isinstance(manifest["record_count"], int) or manifest["record_count"] <= 0:
        raise AliceV2ChunkError("record_count must be a positive JSON integer")
    if not _is_upper_sha256(manifest["run_config_sha256"]):
        raise AliceV2ChunkError("Run-config SHA-256 is malformed")
    if (
        expected_run_config is not None
        and manifest["run_config_sha256"] != expected_run_config.upper()
    ):
        raise AliceV2ChunkError("Run-config SHA-256 does not match the preregistered job")
    if not _is_lower_commit(manifest["source_commit"]) or manifest["source_dirty"] is not False:
        raise AliceV2ChunkError("Source identity must be one clean full lowercase Git object ID")
    if expected_source is not None and manifest["source_commit"] != expected_source.lower():
        raise AliceV2ChunkError("Source commit does not match the expected job identity")

    network = _require_keys(manifest["network"], NETWORK_KEYS, "network")
    if network != {"schema": "ALICE_RUN2RL_LEGACY_V1", "sha256": RUN2RL_SHA256}:
        raise AliceV2ChunkError("Network identity is not the sealed run2rl baseline")
    if manifest["book_sha256"] != "NONE" and not _is_upper_sha256(manifest["book_sha256"]):
        raise AliceV2ChunkError("Book SHA-256 is malformed")
    if not _is_upper_sha256(manifest["payload_sha256"]):
        raise AliceV2ChunkError("Payload SHA-256 is malformed")
    if expected_book is not None and manifest["book_sha256"] != expected_book.upper():
        raise AliceV2ChunkError("Book SHA-256 does not match the expected job identity")

    record_schema = _require_keys(
        manifest["record_schema"], RECORD_SCHEMA_KEYS, "record_schema"
    )
    if record_schema != {
        "schema": RECORD_SCHEMA,
        "schema_sha256": RECORD_SCHEMA_SHA256,
        "rules_id": RULES_ID,
        "rules_hash": RULES_HASH,
        "feature_id": FEATURE_ID,
        "feature_hash": FEATURE_HASH,
    }:
        raise AliceV2ChunkError("Record, rules, or feature identity mismatch")

    target = _require_keys(manifest["target_contract"], TARGET_KEYS, "target_contract")
    if target != {"schema": TARGET_SCHEMA, "sha256": TARGET_SHA256}:
        raise AliceV2ChunkError("Target-contract identity mismatch")

    partition = _require_keys(manifest["partition"], PARTITION_KEYS, "partition")
    numeric_partition = [
        "total_records",
        "records_per_chunk",
        "chunk_index",
        "total_chunks",
        "train_chunks",
        "validation_chunks",
        "test_chunks",
    ]
    if any(not isinstance(partition[key], int) for key in numeric_partition):
        raise AliceV2ChunkError("Partition numeric fields must be JSON integers")
    base_seed = partition["base_seed"]
    if (
        not isinstance(base_seed, str)
        or not base_seed.isdecimal()
        or str(int(base_seed)) != base_seed
        or int(base_seed) > 0xFFFFFFFFFFFFFFFF
    ):
        raise AliceV2ChunkError("base_seed must be a canonical uint64 decimal string")
    total_records = partition["total_records"]
    records_per_chunk = partition["records_per_chunk"]
    total_chunks = partition["total_chunks"]
    chunk_index = partition["chunk_index"]
    if (
        total_records <= 0
        or records_per_chunk <= 0
        or total_chunks != 1 + (total_records - 1) // records_per_chunk
        or not 0 <= chunk_index < total_chunks
        or int(base_seed) + chunk_index > 0xFFFFFFFFFFFFFFFF
        or partition["train_chunks"] < 0
        or partition["validation_chunks"] < 0
        or partition["test_chunks"] < 0
        or partition["train_chunks"]
        + partition["validation_chunks"]
        + partition["test_chunks"]
        != total_chunks
    ):
        raise AliceV2ChunkError("Partition identity is inconsistent")
    expected_split = (
        "train"
        if chunk_index < partition["train_chunks"]
        else "validation"
        if chunk_index < partition["train_chunks"] + partition["validation_chunks"]
        else "test"
    )
    if partition["split"] != expected_split:
        raise AliceV2ChunkError("Chunk split disagrees with its index range")
    expected_count = min(
        records_per_chunk, total_records - chunk_index * records_per_chunk
    )
    if manifest["record_count"] != expected_count:
        raise AliceV2ChunkError("Chunk record count disagrees with the run partition")

    generation = _require_keys(manifest["generation"], GENERATION_KEYS, "generation")
    integer_fields = [key for key in GENERATION_KEYS if key != "seed"]
    if any(not isinstance(generation[key], int) for key in integer_fields):
        raise AliceV2ChunkError("Generation numeric fields must be JSON integers")
    seed = generation["seed"]
    if not isinstance(seed, str) or not seed.isdecimal() or int(seed) == 0 or str(int(seed)) != seed:
        raise AliceV2ChunkError("Resolved seed must be a nonzero canonical uint64 decimal string")
    if int(seed) > 0xFFFFFFFFFFFFFFFF:
        raise AliceV2ChunkError("Resolved seed exceeds uint64")
    if int(seed) != int(base_seed) + chunk_index:
        raise AliceV2ChunkError("Resolved seed disagrees with base_seed + chunk_index")
    if generation["requested_records"] != manifest["record_count"]:
        raise AliceV2ChunkError("record_count differs from requested_records")
    if (
        generation["search_threads"] != 1
        or generation["hash_mb"] <= 0
        or generation["depth"] <= 0
        or generation["opening_count"] <= 0
        or generation["random_move_min_ply"] < 0
        or generation["random_move_max_ply"] < generation["random_move_min_ply"]
        or generation["random_move_max_ply"] >= generation["max_game_ply"]
        or generation["write_min_ply"] < 0
        or generation["write_max_ply"] <= generation["write_min_ply"]
        or generation["max_game_ply"] <= generation["write_max_ply"]
    ):
        raise AliceV2ChunkError("Generation settings violate the sealed domain")


PIECE_TO_FEN = {
    1: "P",
    2: "N",
    3: "B",
    4: "R",
    5: "Q",
    6: "K",
    9: "p",
    10: "n",
    11: "b",
    12: "r",
    13: "q",
    14: "k",
}


def _canonical_fen(
    cells: bytes, side: int, castling: int, halfmove: int, fullmove: int
) -> str:
    ranks: list[str] = []
    for rank in range(7, -1, -1):
        encoded: list[str] = []
        empty = 0
        for file in range(8):
            cell = cells[rank * 8 + file]
            if cell == 0:
                empty += 1
                continue
            if empty:
                encoded.append(str(empty))
                empty = 0
            if cell & 0x10:
                encoded.append("|")
            encoded.append(PIECE_TO_FEN[cell & 0x0F])
        if empty:
            encoded.append(str(empty))
        ranks.append("".join(encoded))
    rights = "".join(
        symbol
        for bit, symbol in enumerate(("K", "Q", "k", "q"))
        if castling & (1 << bit)
    ) or "-"
    return f"{'/'.join(ranks)} {'b' if side else 'w'} {rights} - {halfmove} {fullmove}"


def _decode_record(
    payload: bytes, index: int, coverage: Counter[str]
) -> dict[str, object]:
    if len(payload) != RECORD_BYTES:
        raise AliceV2ChunkError(f"Record {index} is truncated")
    if _crc32(payload[:-4]) != struct.unpack_from("<I", payload, RECORD_BYTES - 4)[0]:
        raise AliceV2ChunkError(f"Record {index} CRC32 mismatch")

    cells, move, score, game_ply, result, side, castling, flags, halfmove, fullmove, _ = (
        RECORD_STRUCT.unpack(payload)
    )
    if side not in (0, 1) or castling & 0xF0 or flags != 1 or result not in (-1, 0, 1):
        raise AliceV2ChunkError(f"Record {index} has invalid scalar fields")
    if fullmove == 0:
        raise AliceV2ChunkError(f"Record {index} fullmove must be positive")

    white_king_square = black_king_square = -1
    pieces = 0
    for square, cell in enumerate(cells):
        if cell == 0:
            continue
        if cell & 0xE0 or (cell & 0x0F) not in VALID_PIECE_CODES:
            raise AliceV2ChunkError(f"Record {index} has an invalid cell at square {square}")
        pieces += 1
        if (cell & 0x0F) == 6:
            if white_king_square != -1:
                raise AliceV2ChunkError(f"Record {index} has multiple white kings")
            white_king_square = square
        elif (cell & 0x0F) == 14:
            if black_king_square != -1:
                raise AliceV2ChunkError(f"Record {index} has multiple black kings")
            black_king_square = square
    if not 2 <= pieces <= 32 or white_king_square < 0 or black_king_square < 0:
        raise AliceV2ChunkError(f"Record {index} violates material or king invariants")

    if move >> 17:
        raise AliceV2ChunkError(f"Record {index} move has reserved bits")
    source = move & 0x3F
    target = (move >> 6) & 0x3F
    promotion = (move >> 12) & 0x07
    source_board = (move >> 15) & 1
    castling_move = (move >> 16) & 1
    source_cell = cells[source]
    if (
        promotion > 4
        or source == target
        or source_cell == 0
        or ((source_cell >> 4) & 1) != source_board
        or int((source_cell & 0x0F) >= 8) != side
    ):
        raise AliceV2ChunkError(f"Record {index} move source or promotion is invalid")
    piece_code = source_cell & 0x0F
    target_cell = cells[target]
    if castling_move:
        king_code = 14 if side else 6
        rook_code = 12 if side else 4
        same_rank = source // 8 == target // 8
        king_side = target % 8 == 7
        required_right = (4 if side else 1) if king_side else (8 if side else 2)
        if (
            promotion != 0
            or piece_code != king_code
            or source % 8 != 4
            or target % 8 not in (0, 7)
            or not same_rank
            or target_cell == 0
            or (target_cell & 0x0F) != rook_code
            or ((target_cell >> 4) & 1) != source_board
            or not (castling & required_right)
        ):
            raise AliceV2ChunkError(
                f"Record {index} castling target is not the source-board rook origin"
            )
    else:
        if target_cell and ((target_cell >> 4) & 1) != source_board:
            raise AliceV2ChunkError(f"Record {index} move arrives on an occupied board")
        is_pawn = piece_code in (1, 9)
        promotion_rank = target // 8 in (0, 7)
        if bool(promotion) != bool(is_pawn and promotion_rank):
            raise AliceV2ChunkError(f"Record {index} promotion fields are inconsistent")

    coverage["records"] += 1
    coverage[f"result_{result:+d}"] += 1
    coverage[f"move_source_board_{'B' if source_board else 'A'}"] += 1
    coverage["promotions"] += promotion != 0
    coverage["castling_moves"] += castling_move
    coverage["captures"] += bool(
        not castling_move and target_cell and ((target_cell >> 4) & 1) == source_board
    )

    record_has_both = True
    for color, king_square in (("white", white_king_square), ("black", black_king_square)):
        king_board = (cells[king_square] >> 4) & 1
        coverage[f"{color}_king_board_{'B' if king_board else 'A'}"] += 1
        relations = set()
        for cell in cells:
            if cell:
                relation = "other" if ((cell >> 4) & 1) != king_board else "same"
                relations.add(relation)
                coverage[f"{color}_{relation}_features"] += 1
        for relation in relations:
            coverage[f"{color}_records_with_{relation}"] += 1
        if relations != {"same", "other"}:
            record_has_both = False
    coverage["records_with_same_and_other_for_both_perspectives"] += record_has_both
    return {
        "fen": _canonical_fen(cells, side, castling, halfmove, fullmove),
        "cells": cells,
        "move": move,
        "from_square": source,
        "to_square": target,
        "promotion": promotion,
        "source_board": source_board,
        "castling_move": bool(castling_move),
        "score": score,
        "game_ply": game_ply,
        "result": result,
        "side_to_move": side,
        "castling": castling,
        "halfmove": halfmove,
        "fullmove": fullmove,
    }


def decode_record_payload(payload: bytes, index: int = 0) -> dict[str, object]:
    """Decode one fully authenticated record payload for round-trip gates."""
    coverage: Counter[str] = Counter()
    decoded = _decode_record(payload, index, coverage)
    decoded["coverage"] = dict(sorted(coverage.items()))
    return decoded


def _audit_chunk_bytes(
    source: Path,
    data: bytes,
    *,
    expected_source: str | None,
    expected_book: str | None,
    expected_run_config: str | None,
    run_config: str | Path | None,
    qualification: bool,
    require_native_coverage: bool,
) -> dict[str, object]:
    if run_config is None and not qualification:
        raise AliceV2ChunkError(
            "A canonical run-config file is mandatory outside explicit qualification mode"
        )
    config: dict[str, object] | None = None
    run_digest: str | None = None
    if run_config is not None:
        config, run_digest = _load_run_config(run_config)
        if expected_run_config is not None and run_digest != expected_run_config.upper():
            raise AliceV2ChunkError("Run-config bytes do not match the external expectation")
        expected_run_config = run_digest
        configured_source = str(config["source_commit"])
        configured_book = str(config["book_sha256"])
        if expected_source is not None and expected_source.lower() != configured_source:
            raise AliceV2ChunkError("Expected source disagrees with the run config")
        if expected_book is not None and expected_book.upper() != configured_book:
            raise AliceV2ChunkError("Expected book disagrees with the run config")
        expected_source = configured_source
        expected_book = configured_book

    stream = io.BytesIO(data)
    manifest, _header = _read_header(stream)
    _validate_manifest(
        manifest,
        expected_source=expected_source,
        expected_book=expected_book,
        expected_run_config=expected_run_config,
    )
    if config is not None and run_digest is not None:
        _bind_manifest_to_run_config(manifest, config, run_digest)
    record_count = int(manifest["record_count"])
    expected_size = HEADER_BYTES + record_count * RECORD_BYTES
    if len(data) != expected_size:
        raise AliceV2ChunkError(
            f"File size mismatch: expected {expected_size}, observed {len(data)}"
        )

    payload_sha256 = hashlib.sha256()
    coverage: Counter[str] = Counter()
    for index in range(record_count):
        payload = stream.read(RECORD_BYTES)
        payload_sha256.update(payload)
        _decode_record(payload, index, coverage)
    if stream.read(1):
        raise AliceV2ChunkError("Trailing bytes are forbidden")
    observed_payload_sha256 = payload_sha256.hexdigest().upper()
    if observed_payload_sha256 != manifest["payload_sha256"]:
        raise AliceV2ChunkError("Payload SHA-256 mismatch")

    if require_native_coverage:
        required = [
            "white_same_features",
            "white_other_features",
            "black_same_features",
            "black_other_features",
            "white_king_board_A",
            "white_king_board_B",
            "black_king_board_A",
            "black_king_board_B",
            "move_source_board_A",
            "move_source_board_B",
        ]
        missing = [name for name in required if coverage[name] == 0]
        if missing:
            raise AliceV2ChunkError("Native coverage gate failed: " + ", ".join(missing))

    return {
        "path": str(source),
        "chunk_sha256": hashlib.sha256(data).hexdigest().upper(),
        "chunk_bytes": len(data),
        "schema": CHUNK_SCHEMA,
        "schema_sha256": CHUNK_SCHEMA_SHA256,
        "record_schema_sha256": RECORD_SCHEMA_SHA256,
        "record_count": record_count,
        "payload_sha256": manifest["payload_sha256"],
        "source_commit": manifest["source_commit"],
        "network_sha256": RUN2RL_SHA256,
        "book_sha256": manifest["book_sha256"],
        "run_config_sha256": manifest["run_config_sha256"],
        "run_config_authenticated": config is not None,
        "target_contract": manifest["target_contract"],
        "partition": manifest["partition"],
        "generation": manifest["generation"],
        "coverage": dict(sorted(coverage.items())),
    }


def audit_chunk(
    path: str | Path,
    *,
    expected_source: str | None = None,
    expected_book: str | None = None,
    expected_run_config: str | None = None,
    run_config: str | Path | None = None,
    qualification: bool = False,
    require_native_coverage: bool = False,
) -> dict[str, object]:
    source = Path(path)
    return _audit_chunk_bytes(
        source,
        source.read_bytes(),
        expected_source=expected_source,
        expected_book=expected_book,
        expected_run_config=expected_run_config,
        run_config=run_config,
        qualification=qualification,
        require_native_coverage=require_native_coverage,
    )


def iter_record_payloads(
    path: str | Path,
    *,
    run_config: str | Path,
    expected_source: str | None = None,
    expected_book: str | None = None,
    expected_run_config: str | None = None,
) -> Iterator[bytes]:
    source = Path(path)
    data = source.read_bytes()
    report = _audit_chunk_bytes(
        source,
        data,
        expected_source=expected_source,
        expected_book=expected_book,
        expected_run_config=expected_run_config,
        run_config=run_config,
        qualification=False,
        require_native_coverage=False,
    )
    del report
    # Authentication covers the complete immutable in-memory byte string before
    # the first record becomes visible to a consumer.
    for offset in range(HEADER_BYTES, len(data), RECORD_BYTES):
        yield data[offset : offset + RECORD_BYTES]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunk", type=Path)
    parser.add_argument("--expected-source")
    parser.add_argument("--expected-book")
    parser.add_argument("--expected-run-config")
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--require-native-coverage", action="store_true")
    args = parser.parse_args()
    try:
        report = audit_chunk(
            args.chunk,
            expected_source=args.expected_source,
            expected_book=args.expected_book,
            expected_run_config=args.expected_run_config,
            run_config=args.run_config,
            require_native_coverage=args.require_native_coverage,
        )
    except (OSError, AliceV2ChunkError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
