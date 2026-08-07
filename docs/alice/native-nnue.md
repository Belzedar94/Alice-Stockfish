# Alice-Native NNUE: SAME/OTHER Contract

Status: normative Phase 0 data, feature, serialization, and parity design.

An Alice-native network must encode which board a piece occupies. Ordinary
piece-square features cannot distinguish all Alice states. This contract uses a
board relation to the perspective king, not an absolute board label.

## 1. Feature definition

For each king perspective, every piece-square channel is split into two planes:

- `SAME`: the piece is on the same Alice board as the perspective king.
- `OTHER`: the piece is on the other Alice board from the perspective king.

The relation is recomputed independently for the two king perspectives. The
perspective king is therefore always `SAME`; the opposing king and every other
piece may be `SAME` or `OTHER`.

This relative definition is invariant under swapping the names of the two
physical boards. Color orientation, king bucketing, piece type, color, and
square orientation otherwise follow the selected base feature set. An Alice
feature index must be a documented pure function of:

`perspective, king square, piece color, piece type, piece square, SAME/OTHER`

The feature-set identifier, index order, dimensions, and board-relation bit are
part of the file-format contract and cannot change without a new version.

### Native v1 identity

Native v1 freezes the following runtime identity. The canonical machine-readable
manifest is [`native-nnue-v1-manifest.json`](native-nnue-v1-manifest.json).

| Component | Identifier | Dimensions | Hash |
| --- | --- | ---: | --- |
| Piece-square | `AliceHalfKAv2_hm_Rel-v1` | 45,056 | `5280C41E` |
| Threats | `AliceFullThreats_Rel-v1` | 119,616 | `6EE7B82C` |
| Pair feature | `None` | 0 | not present |
| Feature transformer | threats, then piece-square | 164,672 logical inputs | `8F4FBC46` |
| Dense body | `1024 -> 32 -> 32 -> 1`, eight stacks | — | `63337116` |
| Composite | `AliceNative-v1` | — | `EC7CCD50` |

The wire version is `A11CE001`. This version is deliberately incompatible in
both directions with the historical `7AF32F20/3C103E72` format.

The canonical manifest is 1,043 UTF-8 bytes with SHA-256
`BFEAC25BC943190C2512B03DD3BC955FD5D3D9FE55109440B81F3DC6A7C883CA`.
The N7 float checkpoint, exact quantization, integer inference, and
transactional activation boundary are frozen separately in
[`native-quantization.md`](native-quantization.md). This keeps the N0-N6 wire
identity unchanged while binding later qualification receipts to exact
checkpoint and quantization contract hashes.
Tensor integers are raw little-endian values. A complete v1 container is
220,315,747 bytes: the 12-byte header, canonical manifest, transformer hash,
220,033,024 feature-tensor bytes, and eight dense hashes followed by 35,204
tensor bytes per stack.

The transactional wire validator checks exact length before walking every
section, validates all component hashes and exact EOF, computes the complete
file SHA-256 in streaming mode, and commits metadata only after success. An
optional expected SHA seals a qualified export. Validation is not parameter
allocation and does not make a file available to evaluation.

### Piece-square index

The eleven planes are own pawn, opposing pawn, own knight, opposing knight,
own bishop, opposing bishop, own rook, opposing rook, own queen, opposing
queen, and either king. For perspective `p`, piece `x`, piece square `s`, and
perspective king square `k`:

```text
plane = 10                                             if x is a king
        2 * (piece_type(x) - pawn) + (color(x) != p)  otherwise

vertical_flip = 56 * p
horizontal_mirror = 7 if file(k) is a, b, c, or d; 0 otherwise
oriented_square = s XOR vertical_flip XOR horizontal_mirror
relation = board(s) XOR board(k)        # SAME=0, OTHER=1

index = oriented_square
      + 64 * plane
      + 704 * relation
      + 1408 * king_bucket(k XOR vertical_flip)
```

The king bucket table is the pinned 32-bucket horizontally mirrored
`HalfKAv2_hm` table. Indices span `0..45055`. As sealed golden examples, a
white pawn on `A:e2` with its king on `A:e1` has index `43660`; after transfer
to `B:e4`, it has index `44380`.

### Threat index

Threat discovery runs independently on each physical board with that board's
occupancy, attackers, and targets. Cross-board edges do not exist. For every
valid edge, the pinned 59,808-entry `FullThreats` base index is extended as:

```text
index = base + 59808 * (edge_board XOR perspective_king_board)
```

Threat indices are 32-bit and span `0..119615`. The orthodox `PP_3Wide`
feature is not part of native v1. Adding any pair feature requires a new native
feature version.

The read-only `alice_native_trace` inspection command emits sorted semantic
piece and threat tuples for both perspectives. It does not load weights, route
evaluation, or modify the historical compatibility backend.

The `alice_native_verify_incremental` command derives explicit board-tagged
piece and threat events from every legal pre/post transition. For an unchanged
perspective king, it applies those events to the sparse index multisets and to
a deterministic scalar fixture accumulator. For a transferred perspective
king, it performs the required full refresh. Every result is compared exactly
with fresh extraction, including all 1,024 accumulator elements, eight PSQT
buckets, and parent restoration after undo. Fixture weights are test data only;
this command is not a native evaluator.

After authenticated parameter loading,
`alice_native_verify_loaded_incremental` repeats the transition proof with the
actual wire integers. Sorted feature-index multiset differences update the
loaded feature and PSQT accumulators only while the perspective king square and
board remain unchanged; a king transfer selects a complete refresh. Every node
is checked against an independent full rebuild through all integer inference
stages, and every undo must restore the parent FEN and position key. This is a
qualification command only and is not called by normal search.

Supported x86 qualification builds execute both loaded accumulator updates and
dense affine layers through AVX2 or SSE4.1/SSSE3 intrinsics. Signed-eight-bit
threat rows are widened before addition, PSQT buckets remain signed 32-bit, and
dense inputs remain unsigned seven-bit. All SIMD results must match the scalar
accumulators and raw dense outputs exactly before the command succeeds.

`alice_native_inference.*` is the single production owner of scalar refresh,
sorted multiset updates, accumulator validation, product pooling, phase
selection, dense arithmetic, activation, skip, and final scaling. Diagnostic
JSON serializes the stages returned by that body instead of recomputing them.
The independent Python reference remains a separate implementation.

The Alice-safe search already requires a concrete evaluator with
failure-reporting `evaluate`, `push`, and `pop` operations. It distinguishes a
normal stop from evaluator failure and proves exact unwind behavior with
injected failures. A private AliceNative-v1 qualification session now owns a
fixed frame for every search ply. Each child frame contains the complete fixed
feature snapshot and wide feature/PSQT accumulators; it is built
transactionally without altering its parent. `Dirties` is not an authority for
native threats. The `alice_native_verify_search_session <depth 0..2>` command
compares every current frame with a semantic full refresh, evaluates through
the production integer body, and requires exact parent and root restoration.
Normal search selects this session only through the explicit native backend and
only after acquiring an authenticated parameter lease.

The parameter object now provides a move-only lease containing its immutable
pointer, generation, wire version, architecture, and SHA-256 identity. Session
qualification holds that lease for the complete traversal. A replacement
attempt while any lease is active is rejected immediately before opening or
parsing a file; the active pointer, generation, and SHA-256 remain unchanged.
After release, the same authenticated object can be leased again. The
`alice_native_verify_lease` command proves the rejection and reacquisition
contract.

`Alice Evaluation` is the single backend selector with `Legacy`, `Native`, and
`Zero` values. `Use NNUE=false` remains a compatibility override for Zero.
Native selection fails before search if no lease can be acquired or if the
selected SHA-256 differs from the installed object. The search thread owns the
lease for its complete lifetime and constructs the fixed session against its
own root position. `eval` and search startup report the generation and SHA-256.
Any initialization, stack, feature, accumulator, dense-arithmetic, identity, or
static-value failure terminates the search without another evaluation backend.

## 2. State and move semantics

Every training and inference position must preserve, losslessly:

- side to move;
- both-board occupancy and the board of every piece;
- castling state, with no en-passant target or entitlement;
- rule counters required to reconstruct the position; and
- the exact Alice variant and feature-format versions.

Incremental updates must implement Alice transfer semantics explicitly:

- A quiet move removes the mover at its origin relation and adds it at the same
  destination coordinate on the opposite board with its new relation.
- A capture additionally removes the victim from the move board before the
  mover transfers.
- Alice v1 has no en passant. Any position, record, or move stream with an
  en-passant target, entitlement, or capture must be rejected before feature
  generation or inference.
- Promotion adds the promoted type on the transfer board.
- Castling transfers both king and rook according to the Alice rules.
- A null move changes neither board occupancy nor relation features.

When the perspective king moves to the other board, the relation of every piece
to that king may flip. Its accumulator must be fully refreshed. For the opposite
perspective, the moved king is updated as an ordinary piece unless the selected
base architecture independently requires a refresh.

No incremental path may infer a board from square occupancy alone.

## 3. Native record pipeline

Generation and training are additionally gated by the single-query
[training storage preflight](training-storage.md). Cleanup and space recovery
remain outside this repository. A failed or below-threshold query stops the
stage without modifying the target volume.

The canonical pipeline is one directional chain:

1. The generator writes a versioned Alice record containing the lossless state,
   target, seed, and provenance hashes.
2. The auditor independently reconstructs the position, legal moves, terminal
   state, and SAME/OTHER features.
3. The trainer reads only audited records of the declared version and records
   the dataset manifest with the checkpoint.
4. Training produces an authoritative checkpoint and a frozen feature manifest.
5. One canonical exporter quantizes and serializes that checkpoint.
6. The engine loads the exact serialized file and exposes its identity at
   startup and in evaluation receipts.

Generated chunks remain immutable. Filtering or merging creates a new manifest
with parent hashes, tool revision, accepted/rejected counts, and exact rejection
reasons. A record with ambiguous board state, an illegal transfer, or an unknown
format version is rejected rather than repaired silently.

The dataset report must break down SAME and OTHER activations by piece type,
piece color, game phase, kings-on-same-board versus kings-on-different-boards,
captures, promotions, castling, and king transfers. Missing categories block
training qualification until the coverage decision is documented.

## 4. Exact parity corpus

Maintain a pinned corpus of at least 1,000 legal Alice positions. It must include
ordinary transfers, captures, promotions, castling, checks, king transfers,
both king-board relations, board-name swaps, color flips, and legal positions
near the counterpart-board occupancy constraint. Preserve each source position,
move sequence, expected legal moves, and corpus SHA-256.

Keep malformed states and records carrying en-passant state in a separate
negative-validation set. Every such input must be rejected before it can enter
the legal parity corpus, feature generation, training, or inference.

The corpus is independent of any single exported network and is extended for
every corrected feature or serialization defect.

## 5. Required gates

All gates below are exact. Tolerance-based success is not permitted unless a
gate explicitly defines a quantization operation.

### G1. Position round-trip

For every parity-corpus position, record-to-state-to-record reconstruction must
produce the same versioned state bytes, legal-move set, side to move, and rule
metadata. Required result: zero mismatches.

### G2. SAME/OTHER feature parity

Trace the active sparse feature indices independently in the engine and trainer
for both king perspectives. Sorted index multisets and associated signs or
weights must be identical. Required result: zero missing, extra, duplicated, or
misclassified features across the entire corpus.

Board-name-swapped positions must preserve the SAME/OTHER trace. Defined color
flips must map through the documented orientation transform exactly.

### G3. Incremental/full-refresh parity

After every ply in every corpus move sequence, compare the incrementally updated
accumulator with a fresh accumulator rebuilt from the resulting position.
Compare every accumulator element before later layers run. Required result:
bit-exact equality and zero mismatches.

### G4. Checkpoint-to-file serialization

The exporter must publish tensor names, shapes, traversal order, quantization
formula, clipping rule, feature-set identifier, and file-format version. An
independent verifier must apply that documented quantization to the checkpoint
and compare every serialized integer in order. Required result: identical
dimensions and zero differing elements.

The file SHA-256 and byte size are sealed in the export receipt. Re-exporting the
same checkpoint with the same tool revision must produce the same bytes.

### G5. File-to-engine parameter parity

After load, an engine inspection path must expose the deserialized tensor hashes
or canonical integer stream. Compare it with the export receipt. Required
result: identical tensor hashes, element counts, ordering, feature-set
identifier, and network SHA-256.

### G6. Evaluation parity

Using the sealed file, compare the trainer-side integer inference path and the
engine inference path on all parity-corpus positions. Compare the raw NNUE output
before unrelated search terms and the final documented NNUE contribution.
Required result: bit-exact raw output and zero centipawn difference for every
position. The release auditor requires the G6 report's sample count to equal the
sealed dataset manifest's complete position count; a partial parity sample does
not qualify.

### G7. Load identity and failure behavior

Startup must report the selected path, byte size, SHA-256, format version, and
feature-set identifier. Missing files, short reads, trailing data, checksum
mismatch, dimension mismatch, and a legacy feature identifier are hard errors.
There is no silent default-network or classical-evaluation fallback in a native
qualification run.

### G8. Determinism and negative controls

Repeated bench runs with the same binary, options, and network must produce the
same node count and bench signature. Explicitly selecting the sealed network and
selecting the same embedded default, if embedding is supported, must agree.

Negative controls are mandatory:

- swapping SAME and OTHER on a relation-asymmetric position must change the
  feature trace and normally the raw output;
- loading a deliberately different valid network must change its reported hash
  and at least one corpus output; and
- corrupting one serialized element or the header must fail identity or parity.

A test that cannot detect these controls does not qualify the pipeline.

### Qualification artifact custody

The release gate does not accept bare provenance hashes or a single aggregate
PASS object. A qualification receipt references the dataset manifest,
checkpoint, export receipt, and one report for every gate by absolute path and
SHA-256. The release auditor opens and hashes each artifact, verifies their
training-run and candidate-network cross-references, derives dataset and
serialized-element sample counts from the referenced manifests, requires the
frozen architecture's complete `170222600` scalar elements, and requires zero
mismatches in every gate report. It also counts nonzero bytes directly in
the AliceNative-v1 tensor regions while excluding headers, manifests, and
architecture hashes. This independent count must equal the qualification
receipt and must be positive.

## 6. Qualification and strength

Use these labels in order:

1. `record-qualified`: G1 passes and the dataset audit is clean;
2. `feature-qualified`: G2 and G3 pass;
3. `serialization-qualified`: G4 through G8 pass; and
4. `strength-qualified`: the exact sealed network then passes the local
   LOS100-x3 battery in [measurement.md](measurement.md).

Strength testing cannot compensate for a failed parity gate. For a network A/B
comparison, keep the executable, book, runner, options, and timing fixed; only
the explicitly loaded network file may differ. Each final receipt links the
dataset manifest, checkpoint hash, exporter revision, serialized-file hash,
parity-corpus hash, all gate reports, and the three strength receipts.

Until every required gate passes, describe the network as experimental rather
than Alice-native.
