# Alice NNUE-v2 native DATAGEN

This document defines the first serious native-data program for the piece-only
Alice NNUE-v2 architecture. It is a data contract, not a release claim and not
the threat-bearing `AliceNative-v1` runtime identity described in
[`native-nnue.md`](native-nnue.md).

The rejected tied-legacy pilot checkpoint remains frozen. Its full-search
integration branch is retained for forensic reproducibility only; it must not
be merged, released, or tested for further strength. This DATAGEN program is a
new upstream input for a future checkpoint with real SAME/OTHER examples.

## Immutable identities

- Record: `ALICEV2_RECORD_V1`, 84 bytes, SHA-256
  `1FCDD61BD11FC5428C3A8726FD21F853A6446BD6A7C3025DC49A34C936CE85AC`.
- Chunk: `ALICE_V2_CHUNK_V1`, 4,096-byte header followed by exact record
  bytes, schema SHA-256
  `8D762FE64B0B8BB196BAFACA26C3514CB338B29BEF2B4AFB268F1974A2E0C6AE`.
- Run configuration: `ALICE_V2_DATAGEN_RUN_V1`, schema SHA-256
  `4200A7FF890F47E39EDA0150C2A20D365C0530AEF7B3112E287328C953F404B9`.
- Target semantics: `ALICE_V2_TARGET_CONTRACT_V1`, SHA-256
  `FFFE33048B9E3FBB91E378F6208CAB0FE1CADB06C6AF29A7493C0496F4E296CD`.
- Feature set: `AliceHalfKAv2_hm_Rel-v1`, with real SAME/OTHER relation
  planes for both king perspectives.
- Teacher: Legacy Alice evaluation with
  `alice_run2rl_e40_l09.nnue`, SHA-256
  `9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9`.
- Official opening book: `ALICE_openings.epd`, raw SHA-256
  `BCD89D9FC3EA81FEB95932EB64D6B6F15AD25CC04CDCC9E0440F097CFFB8CCF6`.
- Production service: only `https://belzedar.duckdns.org`, publication
  protocol 41.

Every chunk header authenticates its canonical JSON bytes with SHA-256 and the
payload with a separate SHA-256. Per-record CRC32 remains the frozen 84-byte
record checksum. Exact file length and EOF are mandatory. The generator writes
to an exclusive temporary pathname, synchronizes it, and publishes the final
pathname atomically without replacement.

The move word preserves Stockfish's internal castling representation: when the
castling bit is set, `to_square` is the rook's source square. Canonical UCI
decoders expose the corresponding king destination on file c or g. Producer,
auditor, and trainer reject any disagreement among that bit, the rook origin,
the castling rights, the source board, and the side to move.

## Preregistered 50M run

The run is exactly 50 logical chunks of 1,000,000 records. OpenBench assigns
chunk `i` the seed `base_seed + i`. Chunks 0-47 are training data, chunk 48 is
validation, and chunk 49 is sealed test data. Whole chunks, and therefore whole
deterministic game streams, never cross split boundaries.

The official job uses priority 300, throughput 1000, workload size 1, and
publication protocol 41 with a mandatory producer artifact. These scheduling
and custody values are part of the external preregistered run configuration.
The v41 identity is frozen as campaign `alice-v2-native-50m-20260811`, external
workload `native-v2-selfplay-50m`, role `selfplay-all-splits`, and cohort
`piece-rel-d6`; the browser form must match all four values exactly.

Search uses one thread per logical chunk so a retry has deterministic search,
game order, and record order. OpenBench's `{THREADS}` value is validated and
reported to the worker log, but it does not change search parallelism or the
canonical chunk bytes. The fixed run
configuration records depth, hash, exploration, write window, book identity,
source commit, partition, RNG, target contract, and retry policy. Its canonical
SHA-256 is supplied in the frozen OpenBench command and embedded in every
chunk; a self-reported header without that external expectation is rejected.

`tools/alice_v2_run_config.py` creates the canonical run configuration only
after the final producer commit is known. The output file is create-new and the
tool prints the exact OpenBench command template and run-config SHA-256.

## Producer and trainer boundary

The normal playing executable does not expose DATAGEN commands. A separate
`alice-stockfish-data-generator` build owns generation and is published as the
role-specific OpenBench DATAGEN artifact. Publication protocol 41 binds its
complete executable SHA-256 and the worker resources in the external server
receipt. Those machine-specific attestations are intentionally excluded from
the canonical chunk header so a retry on another worker remains byte-identical;
the generator command still requires the authenticated `{PRODUCER_SHA256}`
placeholder and rejects a missing or malformed value.

The trainer retains its legacy reader unchanged; production ingestion is not
silently enabled by this campaign. The explicit Python decoder entry point is
`iter_record_payloads(path, run_config=...)`. Before returning the first
training record, it verifies the
header SHA-256, external run-config identity, static schema/target/network
identities, exact length, full payload SHA-256, and then every record CRC and
semantic invariant while decoding. A future trainer adapter must consume only
this authenticated iterator and remains a separate reviewed change.
The qualification gate also reconstructs the canonical Alice FEN from one
generated record and requires byte-for-byte state round-trip parity, including
piece boards, side to move, castling rights, and clocks.

## Acceptance before training

Training remains blocked until all 50 expected chunk indices exist exactly
once, each receipt matches the frozen OpenBench publication contract, retries
are byte-identical, the total is exactly 50,000,000 records, and the independent
auditor reports zero malformed records. Coverage must report real SAME and
OTHER activations for both perspectives, both king boards, both move-source
boards, results, captures, promotions, and castling.
The preregistered aggregate gate additionally requires at least 1,000 records
per million to contain both SAME and OTHER features for both king perspectives;
all listed categorical counters must be nonzero across the authenticated corpus.
`tools/alice_v2_corpus.py` enforces those gates over the server's completed
v41 manifest plus the 50 downloaded `.bz2` artifacts. It verifies the manifest,
publication contract, leases, receipts, producer builds, compressed artifacts,
decompressed chunks, run-config preimage, exact index/seed/count partition, and
aggregate coverage before emitting a create-new corpus receipt.

The corpus establishes record and feature qualification only. Because run2rl
is the teacher, source imitation is not strength evidence. Any future checkpoint
must declare source-relative loss, top-1, regret, ablation, parity, NPS, and
equal-search strength gates before training starts.
