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
python tests/alice/test_engine.py --engine src/stockfish.exe \
  --book <path-to-alice.epd>
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

When the optional frozen book is supplied, the executable suite first verifies
its SHA-256 and its 38,348 unique-position contract, then parses every position
in one engine session. This includes the redundant terminal layer marker in the
book's first rank; input accepts that historical quirk and canonical output
removes it.

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

The contract is mandatory: there is no nullable evaluator and no implicit
zero result. Evaluation, push, and pop return structured failures. A successful
push is paired with one position undo followed by one pop on normal returns,
cutoffs, stops, and failures. A failed push is undone without a pop. Runtime
failure suppresses iteration and best-move publication, reports the evaluator
identity and failing stage, and terminates the UCI process. The
`alice_search_verify_contract` diagnostic injects evaluation, push, and pop
failures and checks exact stack and root restoration separately from a normal
stop.

Executable conformance additionally covers repeated-search determinism, an
Alice mate in one, terminal mate reporting, prompt interruption with exact
root-state preservation, the explicit diagnostic mode, mandatory evaluator
failure propagation, and fail-closed search without a network.

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
- successful loading of an exact content-addressed copy, plus non-zero
  rejection probes for a missing file, wrong version, architecture,
  transformer or layer-stack hash, frozen checksum, truncation, and trailing
  data, including invalidation after a valid load.

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
and depth 12. Its canonical signature is:

```text
Nodes searched  : 202963
```

The positions are mirrored in `tests/alice/fixtures/bench-v1.epd`. The legacy
network suite proves that the embedded list and fixture produce the same node
count, and repeated fresh processes produce the same signature. This is a
build-admission identity, not a strength measurement.

## Native NNUE N0-N7 and fixed-snapshot qualification milestone

The native v1 manifest, identifiers, dimensions, relation order, component
hashes, tensor order, and wire version are frozen in the public contract. A
separate scalar inspection path now extracts full `SAME/OTHER` piece-square
and board-local threat traces for both king perspectives. It uses explicit
board-aware position queries and 32-bit threat indices, omits `PP_3Wide`, and
remains independent of both the selected evaluator and the historical
accumulator.

Board-tagged piece and threat events are derived from complete semantic states.
For an unchanged perspective king, the verification path applies those sparse
events to both sorted feature multisets and a deterministic 1,024-element
scalar integer accumulator with eight PSQT buckets. A transferred perspective
king always rebuilds both feature groups. Every child is compared element by
element with a fresh extraction, and every undo must restore the parent FEN and
position key.

The machine-readable trace command is:

```text
alice_native_trace
alice_native_verify_incremental <depth 0..2>
```

The semantic trace and the bounded runtime snapshot now share the same piece
and threat enumerators. The runtime form stores at most 32 piece indices and
1,024 threat indices per perspective in fixed-capacity arrays, sorts them in
place, and preserves duplicate indices. It performs no dynamic allocation.
Every visited node compares the complete fixed snapshot, king square, and king
board with the diagnostic trace. The opening depth-two tree and directed roots
therefore prove exact snapshot parity across 421 opening positions as well as
captures, promotions, castling, and king transfers before the snapshot is
admitted to a search session.

The accumulator weights in this verification route are bounded deterministic
fixtures; they are not a trained network and are not used by evaluation.
A board-aware fixture cache is indexed by perspective, king board, and king
square. Each entry stores the complete piece array, occupied-coordinate mask,
and `boardB` mask. Its scalar refresh result is compared with a fresh rebuild
at every visited position. A separate bounded-integer SIMD route covers every
accumulator lane and PSQT bucket and must equal the scalar result exactly on
each supported build target. These routes remain qualification fixtures; they
do not allocate native network parameters or provide evaluation.

The N6 wire validator now accepts only the exact native version, composite and
component hashes, 1,043-byte canonical manifest, raw little-endian tensor
layout, 220,315,747-byte total size, exact EOF, and an optional sealed SHA-256.
It computes the full file digest in streaming mode and commits metadata only
after every structural and identity check succeeds. A failed replacement
clears the prior validation state.

An independent sparse all-zero integer exporter creates a logically complete
fixture without checking a network into the repository. Repeated exports have
identical bytes and SHA-256; the independent parser walks all tensor regions.
The negative matrix covers the historical version, wrong native version,
architecture, manifest, transformer hash, every dense-stack hash, truncation,
trailing data, a changed tensor byte under a sealed SHA, and stale-state
replacement. The public diagnostic commands are:

```text
alice_native_validate_file <path> [expected-sha256]
alice_native_try_validate_file <path> [expected-sha256]
alice_native_wire_status
```

Wire validation deliberately does not allocate parameters or make the file an
evaluator. It remains a diagnostic N6 path with state-clearing replacement
semantics.

The separate N7 qualification loader requires a caller-trusted whole-file
SHA-256. It obtains the exact size, structure, whole-file digest, tensor
digests, and parameter bytes from one open handle without reopening the path.
All 220,315,747 bytes are decoded explicitly from little endian into a complete
candidate. Forbidden signed minima and dense `fc0`/`fc1` i16 envelopes are
rejected. A second traversal of the runtime object must reproduce every
canonical tensor digest before one pointer swap can install the candidate.

A successful load increments the parameter generation. A failed replacement
preserves the active pointer, generation, whole-file identity, tensor
identities, and parameter probes. The zero and axis-sentinel fixtures prove
successful replacement; wrong SHA, wrong version, `-32768`, dense-envelope
overflow, and missing-SHA cases prove fail-closed preservation. The
loading and diagnostic commands are:

```text
alice_native_load_file <path> <expected-sha256>
alice_native_try_load_file <path> <expected-sha256>
alice_native_load_status
alice_native_tensor_status
alice_native_parameter <tensor> <flat-index>
alice_native_eval_trace
alice_native_verify_loaded_incremental <depth 0..2>
```

The read-only full-refresh evaluator now emits every normative integer stage:
active sparse indices, both feature and PSQT accumulators, product pooling,
side-to-move ordering, phase, all dense preactivations and branches, skip,
separately divided positional and PSQT components, and the final value. An
independent sparse-wire editor and Python evaluator compare every emitted
element. The corpus covers both sides to move, SAME/OTHER pieces and threats,
all eight phase stacks and all seven boundaries, negative squared inputs, an
odd signed PSQT witness, separate final division, and accumulator overflow.
The companion trainer evaluator and the executable matched 399 complete stage
comparisons over 21 positions with zero mismatches.

The scalar refresh, sorted-delta, range-checking, and dense inference body is
owned by `alice_native_inference.*`. The JSON trace and loaded incremental
verifier call that same implementation; the trace only serializes its returned
accumulators and stages. The same production body now accepts either semantic
trace records or fixed-capacity runtime snapshots. At every loaded-network node,
the fixed full refresh and fixed sorted delta are compared element by element
with the semantic route before and after undo. The independent Python evaluator
remains separate.

The loaded-network verifier carries the authenticated feature and PSQT
accumulators through every legal transition. It applies sorted multiset
differences when a perspective king is unchanged and performs a full refresh
when that king changes square or board. At every node it compares all 1,024
feature lanes, all eight PSQT buckets, every later integer stage, and the
restored parent FEN and key. The depth-two opening tree covers 421 positions
and 420 transitions; directed depth-one roots add captures, quiet and capture
promotions, castling on both board layouts, and king transfers. Missing
parameters and accumulator overflow remain fatal.

On AVX2 and SSE4.1/SSSE3 builds, the same qualification paths also execute the
loaded feature-transformer rows, signed-eight-bit threat widening, PSQT
updates, and all three dense affine layers with the target SIMD intrinsics.
Every SIMD accumulator element and dense output is compared with the scalar
result. The dense corpus reaches both `+32258` and `-32258`, covers all eight
phase stacks, and exercises full refresh plus incremental updates at every
visited node.

The private search-session verifier allocates one fixed frame for every legal
search ply before traversal. A frame owns the position identity, both fixed
feature snapshots, and wide signed accumulators. `push` constructs the next
frame transactionally; `pop` selects the already preserved parent only after
its complete position identity matches. At every visited position,
`alice_native_verify_search_session <depth 0..2>` compares both session
accumulators and the static value with a semantic full refresh. It also requires
balanced pushes and pops, exact undo restoration, and unchanged parameter
identity. The opening tree and all directed transition roots pass this gate.

A move-only parameter lease pins the immutable object, generation, wire
version, architecture, and SHA-256 for the complete session. Replacement is
rejected immediately while a lease is active, before file I/O, and preserves
the installed pointer and identity. `alice_native_verify_lease` proves one
rejected replacement followed by successful reacquisition of the same object.

The explicit `Alice Evaluation` selector offers `Legacy`, `Native`, and `Zero`.
Selecting `Native` requires a successfully authenticated object and acquires a
lease before the search thread starts. The session then supplies every static
evaluation through the fixed frame stack. Initialization, feature, arithmetic,
identity, stack, and value-range failures stop the search through the structured
failure channel; no historical or zero fallback is possible. `eval`, load
status, and search startup expose the native generation and SHA-256. The legacy
`Use NNUE=false` setting remains a compatibility override for deterministic
zero diagnostics.

An interactive runtime test loads a complete all-zero native wire, verifies
`eval`, completes `go depth 1`, starts `go infinite`, rejects a replacement
while the live lease is active, stops promptly, emits a legal best move, and
confirms that the original generation and SHA-256 remain installed. Loaded
status and qualification reports now state `search=available`.

## Operational acceptance milestone

The Alice-safe search emits an exact machine terminal record for checkmate,
stalemate, and rule draws before `bestmove (none)`. The strict paired runner
rejects a missing, malformed, contradictory, or move-followed terminal record;
the safety ply limit is a policy abort rather than a scored draw.

The repository now contains the single statistical authority for local
acceptance. It uses deterministic book cycles, two persistent pair processes,
attempt-ordinal admission, pair-atomic classifications, pentanomial LOS,
create-only artifacts, a seal-before-drain transition, independent attempted
and scored 64,000-game caps, and separate exact-LOS and 400/300/200 policies.
The three-control aggregator requires zero abort evidence.

The controller test runs a complete 200-game fixed LTC sample through two
persistent deterministic test processes. This proves orchestration, artifact
hashing, admission, sealing, and receipt construction; it is not a strength
result and uses no trained network. The release-evidence auditor likewise
performs no publication and remains blocked without a trained AliceNative-v1
network, G1-G8 receipts, both local batteries, four release binaries, triple
bench, negative load probes, and official OpenBench shadow evidence.

The training storage preflight performs exactly one read-only available-space
query and requires 500 GiB. It neither starts training nor performs cleanup.

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
