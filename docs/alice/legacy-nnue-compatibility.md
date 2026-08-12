# Legacy Alice NNUE compatibility contract

- **Status:** Accepted as a temporary compatibility bridge
- **Legacy source:** [`4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79`](https://github.com/fairy-stockfish/Fairy-Stockfish/commit/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79)
- **Contract date:** 2026-08-06

## Frozen network identity

| Property | Value |
| --- | --- |
| Historical/panel filename | `alice_run2rl_e40_l09.nnue` |
| Public Alice-Stockfish 1.0 filename | `Alice_v1.nnue` |
| File size | 47,721,376 bytes |
| SHA-256 | `9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9` |
| NNUE serialization version | `0x7AF32F20` |
| Composite architecture hash | `0x3C103E72` |
| Embedded description | `Network trained with the https://github.com/ianfab/variant-nnue-pytorch trainer.` |

The first two little-endian 32-bit words of the file independently reproduce
the version and architecture hash. The complete-file SHA-256 is the identity
used for baseline experiments; a matching header alone does not prove that a
file contains the frozen weights.

`Alice_v1.nnue` is a public release alias only. Its bytes are identical to the
frozen legacy artifact; the rename does not represent new training or migration
to native Alice NNUE. Reproduction with the historical Fairy-Stockfish loader
may use a byte-identical copy under the original lowercase filename because
that loader performs variant-name basename dispatch.

## Legacy architecture

The frozen source selects `HalfKAv2Variants`, transforms to 512 dimensions,
uses 8 PSQT buckets and 8 layer stacks, and applies a `16 -> 32 -> 1` dense
network
([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/nnue/nnue_architecture.h#L32-L52)).
The serializer version is fixed at `0x7AF32F20`
([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/nnue/nnue_common.h#L50-L51)),
while the file's composite hash is the XOR-derived feature-transformer and
network-structure identity
([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/nnue/evaluate_nnue.h#L28-L33)).

`HalfKAv2Variants` maps a feature from the oriented king square, piece type,
piece square, color perspective, and variant-specific lookup tables. Neither
the active-feature path nor the incremental path reads `mirrorBoard`
([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/nnue/features/half_ka_v2_variants.cpp#L32-L109)).
This omission is structural: pieces with the same type, color, and coordinate
produce the same feature whether they are on board A or board B. The network
may learn correlations from the overlaid positions in its training data, but
it cannot distinguish an otherwise identical layer assignment.

## Verified executable load

The frozen executable has SHA-256
`B70AFE03EC9A67258CD7B5B848C46FC9E5C83F53B9F2825E9A5946FEEFB59599`
and identifies itself as `Fairy-Stockfish 040925`. The following synchronized
probe completed successfully:

```text
uci
<wait for uciok>
setoption name UCI_Variant value alice
setoption name EvalFile value C:\nets\alice_run2rl_e40_l09.nnue
isready
<wait for readyok>
position startpos
go depth 2
```

The engine printed `NNUE evaluation using ...alice_run2rl_e40_l09.nnue
enabled` and returned `bestmove a2a3`. This proves that the frozen executable
accepts and searches with the frozen file; it does not prove layer-aware
evaluation.

The historical loader selects an entry from `EvalFile` when its basename begins
with the current variant name or the variant's `nnueAlias`
([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/evaluate.cpp#L77-L103)).
For Alice, `alice_run2rl_e40_l09.nnue` is a direct variant-name match. Alice's
second `init()` clears the chess `nn-` alias, so a generic
`nn-123456789abc.nnue` name is not an Alice alias. The modern compatibility
bridge does not use that historical basename dispatch: content-addressed build
systems may cache the file under its digest. It records the selected path but
identifies the network from the parsed format and the SHA-256 of the bytes that
were actually opened.

## Strict load policy

Alice-Stockfish must fail closed. It must never continue a compatibility run
with classical evaluation, an embedded chess network, a previous network, or
an arbitrary same-name file.

| Condition | Required behavior |
| --- | --- |
| Exact SHA-256, version `0x7AF32F20`, architecture `0x3C103E72` | Load as the frozen legacy baseline and print the normalized path, mode, and SHA-256. |
| Same version and architecture, different SHA-256 | Treat as format-compatible but not baseline-equivalent. Load only when explicitly selected outside a frozen-baseline run, and print its SHA-256. |
| Wrong serialization version | Reject before allocating or reading weights. |
| Wrong architecture hash | Reject; do not attempt partial or shape-based conversion. |
| Truncated, corrupt, unreadable, or missing file | Reject with a non-zero outcome and a precise diagnostic. |
| Canonical or content-addressed basename with the exact frozen bytes | Apply the same version, architecture, structural, and SHA-256 checks; the path name is not identity. |
| Multiple Alice-compatible entries are supplied | Reject ambiguity unless one entry is explicitly selected. |
| `Use NNUE` is explicitly disabled | Permit only a clearly reported non-baseline diagnostic mode; it cannot satisfy compatibility or strength gates. |

File size is recorded as a receipt for the frozen artifact, but validation is
not based on size alone. The loader must parse the header safely, verify that
all expected parameters are present, reject trailing structural mismatches,
and compute SHA-256 from the bytes actually opened.

## Compatibility validation gates

The bridge is accepted only after all of the following pass on a fixed corpus:

1. The new full-refresh feature extraction matches the legacy executable's
   output exactly for positions the legacy representation can express.
2. Every enabled incremental evaluation path equals a fresh rebuild after
   every move, capture, promotion, castling move, and undo. Null-move search is
   disabled in the correctness-first Alice search and is not an enabled path.
3. The loaded path and SHA-256 remain stable across `ucinewgame`, position
   changes, thread-count changes, and repeated searches.
4. Missing, corrupt, wrong-version, wrong-architecture, wrong-checksum, and
   ambiguous-network probes all terminate without an evaluation fallback; an
   exact content-addressed copy loads successfully.
5. Layer-swapped position pairs are included and documented as expected
   legacy feature collisions, preventing board blindness from being mistaken
   for successful native coverage.

Legacy parity is a compatibility result, not authoritative evidence for the
rules. Positions affected by the known legacy hashing, SEE, pinning, or
legality defects must be labeled and adjudicated by the independent Alice
rules implementation.

## Implemented bridge status

The current `LegacyAliceExact` bridge implements the frozen-baseline and
explicit format-compatible policies above. It is intentionally scalar and
owns a dedicated accumulator stack; no orthodox Stockfish accumulator or
evaluation route is reachable from it. The stack applies historical
piece-square deltas, refreshes a perspective when its king moves, and restores
the parent accumulator on undo. `Use NNUE` is enabled by default, so normal
`eval` and `go` commands require a successfully loaded network. The only
zero-evaluation path requires the explicit diagnostic setting `Use NNUE false`
and identifies itself in UCI output.

The versioned public fixture records seven exact raw and adjusted evaluation
vectors. Differential validation against a minimally instrumented build of the
frozen source also matched both values on 80 deterministic random legal
positions. Negative probes cover structural, identity, integrity, and file
errors, and verify that a failed replacement cannot retain a previously loaded
evaluator. Exhaustive incremental verification additionally covers ordinary
moves, captures, promotions, castling, king moves, and undo restoration. These
results establish exact historical compatibility; they do not claim native
layer awareness.

## Public provenance

The engine source and the historical NNUE trainer named by the file are public:

- [Fairy-Stockfish frozen source](https://github.com/fairy-stockfish/Fairy-Stockfish/tree/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79)
- [variant-nnue-pytorch trainer](https://github.com/ianfab/variant-nnue-pytorch)

The frozen compatibility network and its checksum record are published in the
[OpenBench assets release](https://github.com/Belzedar94/Alice-Stockfish/releases/tag/openbench-assets-v1).
An older Alice network is also available through the
[historical public download](https://drive.google.com/file/d/1BqFt3H5zUGHdKwYa1vT_boSsM-kZGIoc/view).
It is a separate artifact and must not be represented as the frozen
`alice_run2rl_e40_l09.nnue` file.

Every public release that includes a network must ship or link all of the
following together: exact file, SHA-256, byte size, serialization version,
architecture hash, training provenance, license statement, and the engine
builds validated against it.

## Migration to native Alice NNUE

The compatibility bridge preserves historical measurements while the native
feature contract is developed. It is not extended into the native format.

The native network will:

- encode each piece as `SAME` or `OTHER` relative to the perspective king's
  board;
- make threats and all incremental deltas layer-aware;
- use a new feature identity and composite architecture hash, preventing a
  legacy file from being accepted accidentally;
- require exact checkpoint-to-file and file-to-engine parity, plus exact
  full-refresh versus incremental parity; and
- publish under a distinct filename and manifest from all legacy networks.

Legacy weights are not relabeled as native weights. Migration requires new
training data from the corrected rules engine, a fresh export, and independent
strength testing. The frozen legacy network remains available only for
regression and historical comparison after the native network becomes the
release default.
