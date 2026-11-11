#!/usr/bin/env python3
"""Create the preregistered 50M Alice NNUE-v2 DATAGEN run contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


PRODUCTION_URL = "https://belzedar.duckdns.org"
RUN_SCHEMA_SHA256 = "4200A7FF890F47E39EDA0150C2A20D365C0530AEF7B3112E287328C953F404B9"
NETWORK_SHA256 = "9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9"
BOOK_SHA256 = "BCD89D9FC3EA81FEB95932EB64D6B6F15AD25CC04CDCC9E0440F097CFFB8CCF6"
RECORD_SCHEMA_SHA256 = "1FCDD61BD11FC5428C3A8726FD21F853A6446BD6A7C3025DC49A34C936CE85AC"
CHUNK_SCHEMA_SHA256 = "8D762FE64B0B8BB196BAFACA26C3514CB338B29BEF2B4AFB268F1974A2E0C6AE"
TARGET_CONTRACT_SHA256 = "FFFE33048B9E3FBB91E378F6208CAB0FE1CADB06C6AF29A7493C0496F4E296CD"
EXTERNAL_WORKLOAD_ID = "native-v2-selfplay-50m"
PUBLICATION_ROLE = "selfplay-all-splits"
PUBLICATION_COHORT = "piece-rel-d6"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_config(source_commit: str, campaign: str, base_seed: int) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None or source_commit == "0" * 40:
        raise ValueError("source_commit must be one nonzero lowercase 40-digit Git object ID")
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", campaign) is None:
        raise ValueError("campaign must be an OpenBench v41 lowercase ASCII slug")
    if not 0 < base_seed <= (2**63 - 50):
        raise ValueError("base_seed must leave room for all 50 signed-64-bit chunk seeds")
    return {
        "schema": "ALICE_V2_DATAGEN_RUN_V1",
        "version": 1,
        "run_schema_sha256": RUN_SCHEMA_SHA256,
        "campaign": campaign,
        "production_url": PRODUCTION_URL,
        "source_commit": source_commit,
        "source_dirty": False,
        "target_contract_sha256": TARGET_CONTRACT_SHA256,
        "record_schema_sha256": RECORD_SCHEMA_SHA256,
        "chunk_schema_sha256": CHUNK_SCHEMA_SHA256,
        "network_sha256": NETWORK_SHA256,
        "book_sha256": BOOK_SHA256,
        "total_records": 50_000_000,
        "records_per_chunk": 1_000_000,
        "total_chunks": 50,
        "base_seed": base_seed,
        "seed_derivation": "chunk_seed = base_seed + chunk_index",
        "rng": "xorshift64star-v1; zero seed forbidden; modulo bounded draws",
        "record_order": "single-thread game ordinal then game ply",
        "split": {"train_chunks": 48, "validation_chunks": 1, "test_chunks": 1},
        "openbench": {
            "priority": 300,
            "throughput": 1000,
            "publication_protocol": 41,
            "producer_artifact_required": True,
            "workload_size": 1,
            "campaign_id": campaign,
            "external_workload_id": EXTERNAL_WORKLOAD_ID,
            "role": PUBLICATION_ROLE,
            "cohort": PUBLICATION_COHORT,
        },
        "retry": {
            "logical_identity": "source_commit + run_config_sha256 + chunk_index",
            "required_output": "identical uncompressed bytes across workers",
            "producer_attestation": "external OpenBench protocol-41 receipt",
            "worker_resources": "external OpenBench environment receipt",
            "excluded_from_chunk_bytes": ["producer_sha256", "assigned_threads"],
        },
        "generation": {
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
        },
        "acceptance": {
            "scope": "aggregate-over-50-authenticated-chunks",
            "chunk_count": 50,
            "record_count": 50_000_000,
            "malformed_records_max": 0,
            "duplicate_chunk_indices_max": 0,
            "identity_mismatches_max": 0,
            "records_with_same_and_other_for_both_perspectives_min_ppm": 1000,
            "minimum_nonzero_counters": [
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
            ],
        },
    }


def command_template(run_config_sha256: str, base_seed: int) -> str:
    return (
        "alice_v2_generate_training_data threads {THREADS} hash 512 "
        "network {NETWORK} network_sha256 {NETWORK_SHA256} "
        "producer_sha256 {PRODUCER_SHA256} book {BOOK} book_sha256 {BOOK_SHA256} "
        "out {OUT} count {COUNT} seed {SEED} "
        f"run_config_sha256 {run_config_sha256} base_seed {base_seed} "
        "total_records 50000000 records_per_chunk 1000000 "
        "train_chunks 48 validation_chunks 1 test_chunks 1 depth 6 nodes 0 "
        "random_move_min_ply 1 random_move_max_ply 20 random_move_count 8 "
        "random_multi_pv 4 random_multi_pv_diff 200 write_min_ply 5 "
        "write_max_ply 400 max_game_ply 512 set_recommended_uci_options"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--campaign", default="alice-v2-native-50m-20260811")
    parser.add_argument("--base-seed", type=int, default=202608110000000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = build_config(args.source_commit, args.campaign, args.base_seed)
    payload = canonical_bytes(config)
    digest = hashlib.sha256(payload).hexdigest().upper()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(
        json.dumps(
            {
                "schema": "ALICE_V2_DATAGEN_RUN_CONFIG_RECEIPT_V1",
                "path": str(args.output.resolve()),
                "bytes": len(payload),
                "sha256": digest,
                "command": command_template(digest, args.base_seed),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
