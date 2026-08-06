# Implementation status

This page records verified implementation state. It is not a release claim.

## Rules-core milestone

The current rules path provides:

- canonical board-layer state stored with `StateInfo`;
- strict transactional parsing of compact Alice FEN and legacy 16-wide input;
- canonical compact FEN output;
- explicit board-local occupancy and attack queries;
- board-aware full, pawn, minor, and non-pawn position identities;
- all-candidate Alice move generation followed by complete legality filtering;
- source-board movement and capture followed by opposite-board transfer;
- promotion-before-transfer and disabled en passant;
- symmetric castling from either board with source, transit, provisional, and
  final-board king-safety checks;
- full derived-state recomputation after every move;
- exact make/unmake restoration through the prior `StateInfo`; and
- case-sensitive, exactly-one-match UCI move resolution.

The implementation deliberately favors direct, auditable state transitions
over search speed at this milestone.

## Reproducible verification

The independent reference and executable conformance suites are:

```text
python tests/alice/test_reference.py
python tests/alice/test_engine.py --engine src/stockfish.exe
```

The start-position perft agreement is:

| Depth | Nodes |
| ---: | ---: |
| 1 | 20 |
| 2 | 400 |
| 3 | 9,384 |
| 4 | 219,236 |

Depth 4 was computed independently by the slow specification implementation
and by the optimized engine. A Windows debug build with standard-library
assertions enabled produced the same value. The executable suite also checks
all versioned fixtures, key relations, canonical FEN transitions, repetition
checkpoints, and deterministic playout legal sets against the independent
implementation.

## Deterministic safe-search milestone

The public `go` route now uses a dedicated single-threaded iterative-deepening
search over the complete Alice legal move set. It provides exact terminal and
mate-distance scores, deterministic move ordering and principal variations,
bounded depth, node and time modes, responsive `stop`, and no calls into the
orthodox evaluator, accumulator, move picker, pruning stack, transposition
table, or tablebases.

Static leaves are supplied through a narrow evaluator contract. They never
enter the orthodox Stockfish evaluator or its accumulator. Normal search now
requires the strict historical Alice evaluator described below. Explicitly
setting `Use NNUE` to `false` selects a reported zero-evaluation diagnostic
mode; that mode is not a compatibility or strength result. `export_net`
remains closed.

Executable conformance additionally covers repeated-search determinism, an
Alice mate in one, terminal mate reporting, prompt interruption with exact
root-state preservation, the explicit diagnostic mode, and fail-closed search
without a network.

## Historical NNUE compatibility milestone

`LegacyAliceExact` is an engine-owned evaluator for the frozen historical
Alice architecture. Its default policy accepts only the
exact file name, serialization version, composite architecture hash, internal
transformer and layer-stack hashes, structural length, end of file, and frozen
SHA-256. The loader hashes the bytes it actually opens and reports the
normalized path, policy mode, SHA-256, version, and architecture through UCI.

`Alice_Frozen_Network` defaults to `true`. Setting it explicitly to `false`
permits a structurally exact but non-baseline file as
`format-compatible`; the different checksum remains visible. A rejected or
empty `EvalFile` clears any previously loaded evaluator. With `Use NNUE`
enabled, `eval` and `go` then terminate with a non-zero outcome instead of
using zero evaluation, an embedded chess network, or stale weights.

Normal builds define `NNUE_EMBEDDING_OFF`, do not make the orthodox Stockfish
network a build prerequisite, and initialize only an unreachable zeroed shell
needed by the retained thread-pool type. The `Engine` exposes no orthodox
network load or save route. Historical Alice weights exist only in the
separate strict compatibility backend.

The scalar implementation reproduces the historical feature transformer,
PSQT bucket, `16 -> 32 -> 1` stack, integer clipping and scaling, and
adjusted-evaluation weighting. Search starts from an independently rebuilt
accumulator, applies exact dirty-piece deltas after ordinary moves, captures,
promotions, and castling, rebuilds the affected perspective after a king move,
and pops the accumulator on undo. Its board blindness is intentional and
limited to this compatibility class.

Verified compatibility evidence consists of:

- seven fixed vectors covering the start position, a transferred pawn,
  tactical positions, both layers, and an expected layer collision;
- exact raw and adjusted equality on 80 deterministic random legal positions;
- exact full-refresh versus incremental equality over exhaustive subtrees that
  include captures, promotions, castling, king moves, and undo restoration;
- an exact network-backed depth-one root result; and
- non-zero rejection probes for a missing file, wrong basename, version,
  architecture, transformer or layer-stack hash, frozen checksum, truncation,
  and trailing data, including invalidation after a valid load.

The public executable checks are:

```text
python tests/alice/test_legacy_nnue.py --engine src/stockfish.exe \
  --network <path-to-alice_run2rl_e40_l09.nnue>
```

## Deterministic build and bench contract

The source tree accepts the OpenBench worker build shape directly:

```text
make -j EXE=<output> EVALFILE=<path-to-alice_run2rl_e40_l09.nnue>
```

The default goal selects the native architecture and the platform compiler.
`EVALFILE` becomes the strict startup default only in that worker artifact;
ordinary release builds retain an empty default and therefore fail closed until
the network is selected explicitly. A worker artifact reports the same
normalized path, checksum, serialization version, and architecture as an
interactive load. It contains no orthodox embedded network.

Bare `bench` uses eight versioned Alice positions, one thread, 16 MiB of hash,
and depth 3. Its canonical signature is:

```text
Nodes searched  : 162582
```

The positions are mirrored in `tests/alice/fixtures/bench-v1.epd`. The legacy
network suite proves that the embedded list and fixture produce the same node
count, and repeated fresh processes produce the same signature. This is a
build-admission identity, not a strength measurement.

## Cross-platform verification

The repository verification workflow builds BMI2 and AVX2 binaries on Linux
and Windows. Linux instrumented jobs cover standard-library assertions,
AddressSanitizer plus UndefinedBehaviorSanitizer, and ThreadSanitizer. Every
job runs the independent rules suite and executable conformance suite; network
parity remains a separate artifact-backed gate because the frozen network is
not stored in the repository.

## Deliberately disabled paths

The following orthodox shortcuts remain unavailable until they receive a
board-aware implementation and dedicated coverage:

- classical static exchange evaluation and its pruning decisions;
- cuckoo upcoming-repetition detection;
- Syzygy probing and root ranking;
- null-move and tuned orthodox search shortcuts;
- the current Stockfish accumulator and threat features; and
- insufficient-material shortcuts.

The historical bridge makes the safe search playable, but it does not turn the
current route into a strength release. Remaining gates include completion of
the cross-platform workflow on every release revision, a board-aware strength
search, and the native Alice NNUE defined in
[`native-nnue.md`](native-nnue.md).
