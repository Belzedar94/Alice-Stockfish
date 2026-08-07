# AliceNative-v1 checkpoint and quantization contract

Status: normative N7 contract. This document does not enable native evaluation,
record generation, or training.

The canonical machine-readable contracts are:

| Contract | Canonical bytes | SHA-256 |
| --- | ---: | --- |
| [`native-quant-v1.json`](native-quant-v1.json) | 1,194 | `DD8571715CB7711BEE46785D0FBAC9F480ECCADD1D6CC9EF71D652554F80F9C8` |
| [`native-checkpoint-v1.json`](native-checkpoint-v1.json) | 1,031 | `A7E667BB5B7B978E474A392960CF6A72A5F1A9B074DDFC97C6FA13166B5D3413` |

Both files are canonical ASCII JSON with sorted object keys, compact separators,
and exactly one terminal line feed. Their hashes include that line feed.

## 1. Export checkpoint

Only `alice-native-checkpoint-v1` is accepted by the canonical exporter. It is a
Safetensors container with strict metadata and exactly eleven C-contiguous
IEEE-754 binary32 tensors. Training-resume files, pickled objects, float64
tensors, aliases, missing or extra tensors, and implicit reshaping are rejected.

| Tensor | Shape | Axis order |
| --- | ---: | --- |
| `ft.bias` | `[1024]` | lane |
| `threat.weight` | `[119616, 1024]` | feature, lane |
| `threat.psqt` | `[119616, 8]` | feature, phase bucket |
| `pieceSquare.weight` | `[45056, 1024]` | feature, lane |
| `pieceSquare.psqt` | `[45056, 8]` | feature, phase bucket |
| `stack.fc0.bias` | `[8, 32]` | stack, output |
| `stack.fc0.weight` | `[8, 32, 1024]` | stack, output, input |
| `stack.fc1.bias` | `[8, 32]` | stack, output |
| `stack.fc1.weight` | `[8, 32, 64]` | stack, output, input |
| `stack.fc2.bias` | `[8, 1]` | stack, output |
| `stack.fc2.weight` | `[8, 1, 128]` | stack, output, input |

The feature-transformer and PSQT tensors are shared by both king perspectives.
Perspective is represented only by the active feature indices. The eight dense
stacks and eight PSQT columns are independent parameters.

The required checkpoint metadata is the identity recorded in
`native-checkpoint-v1.json`. A receipt additionally binds the complete
checkpoint SHA-256, this schema SHA-256, and the quantization-contract SHA-256.

## 2. Exact quantization

For a finite binary32 value `x`, integer scale `S`, and symmetric maximum `M`:

```text
q = round_to_nearest_ties_to_even(exact_binary32_value(x) * S)
```

Exact half ties choose the even integer. Examples are `0.5 -> 0`, `1.5 -> 2`,
`2.5 -> 2`, `-0.5 -> 0`, `-1.5 -> -2`, and `-2.5 -> -2`. The result cannot
depend on the process rounding mode, a device kernel, extended precision, or an
implementation-defined cast. Negative zero becomes integer zero. Finite
subnormal values are accepted; NaN and either infinity are rejected.

The integer ranges are symmetric. Values `-128`, `-32768`, and `INT32_MIN` are
not valid native parameters.

| Tensor | Scale | Wire type | Allowed integer range |
| --- | ---: | --- | ---: |
| `ft.bias` | 256 | little-endian i16 | `[-32767, 32767]` |
| `threat.weight` | 256 | i8 | `[-127, 127]` |
| `threat.psqt` | 9600 | little-endian i32 | `[-2147483647, 2147483647]` |
| `pieceSquare.weight` | 256 | little-endian i16 | `[-32767, 32767]` |
| `pieceSquare.psqt` | 9600 | little-endian i32 | `[-2147483647, 2147483647]` |
| `stack.fc0.bias` | 16384 | little-endian i32 | `[-2147483647, 2147483647]` |
| `stack.fc0.weight` | 128 | i8 | `[-127, 127]` |
| `stack.fc1.bias` | 8192 | little-endian i32 | `[-2147483647, 2147483647]` |
| `stack.fc1.weight` | 64 | i8 | `[-127, 127]` |
| `stack.fc2.bias` | 16384 | little-endian i32 | `[-2147483647, 2147483647]` |
| `stack.fc2.weight` | 128 | i8 | `[-127, 127]` |

The exporter validates the unquantized rational domain before rounding and
never clamps. An out-of-domain or out-of-range element aborts the complete
export and identifies its tensor and multi-index. A successful export has zero
non-finite values, zero out-of-range values, and zero saturations.

## 3. Integer inference

For perspective `c` and lane `k`, the feature accumulator is:

```text
A[c,k] = ft.bias[k]
       + sum(pieceSquare.weight[activePieceFeature,k])
       + sum(threat.weight[activeThreatFeature,k])
```

For phase bucket `b`:

```text
P[c,b] = sum(pieceSquare.psqt[activePieceFeature,b])
       + sum(threat.psqt[activeThreatFeature,b])
```

Every active feature has arity one. Duplicate trace indices, if present, retain
their multiplicity.

Define `clip255(x) = min(255, max(0, x))`. For `j = 0..511`:

```text
T[c,j] = floor(clip255(A[c,j]) * clip255(A[c,j+512]) / 512)
```

`T` is in `0..127`. The 1,024-element dense input always places the side to
move first:

```text
X[0..511]    = T[sideToMove,0..511]
X[512..1023] = T[opponent,0..511]
```

With `N` equal to the total number of pieces across both boards, including both
kings:

```text
phase = (N - 1) // 4
```

Legal v1 inference requires `2 <= N <= 32`. The same phase selects the dense
stack and PSQT column.

For the selected stack:

```text
Z0[o] = fc0.bias[o] + sum(fc0.weight[o,i] * X[i])

linear_s(x) = min(127, max(0, floor(x / 2^s)))
square_s(x) = min(127, floor(x*x / 2^(2*s+7)))

S0 = square_7(Z0)
R0 = linear_7(Z0)
Y1 = concat(S0, R0)

Z1[o] = fc1.bias[o] + sum(fc1.weight[o,i] * Y1[i])
S1 = square_6(Z1)
R1 = linear_6(Z1)
Y2 = concat(S0, R0, S1, R1)

Z2 = fc2.bias[0] + sum(fc2.weight[0,i] * Y2[i])
fwdOut = Z2 + Z0[30] - Z0[31]
```

Negative hidden preactivations are squared before the squared branch is
clipped. Every qualified `Z0` and `Z1` value must fit signed i16 so scalar and
SIMD squared-activation routes have the same domain.

Let `trunc0` be signed integer division toward zero:

```text
positionalRaw16 = trunc0(fwdOut * 9600 / 16384)
psqtRaw16       = trunc0((P[sideToMove,phase] - P[opponent,phase]) / 2)

positionalValue = trunc0(positionalRaw16 / 16)
psqtValue       = trunc0(psqtRaw16 / 16)
nativeNnueValue = positionalValue + psqtValue
```

The final two divisions remain separate. Dividing the sum can differ by one
evaluation unit and is not equivalent.

Reference arithmetic uses i64 for affine sums, PSQT differences, output
scaling, and the skip addition. Every narrowing is range checked. Native search
evaluation remains unavailable until all N7 stage traces match exactly.

## 4. Canonical export and verification

The exporter authenticates the strict float checkpoint, preflights every
element, writes a create-new temporary file in the frozen N6 wire order, uses
explicit little-endian integer encodings, checks the exact final length,
performs a complete structural self-read, and publishes atomically only after
success.

The independent verifier does not import the exporter, its quantizer, or its
tensor-order table. It derives expectations from the frozen contracts,
implements exact binary32 rational rounding independently, traverses every
checkpoint and wire element, and reports the first mismatch with tensor,
stack, multi-index, float bits, scale, expected integer, actual integer, and
wire offset. Digest-only checkpoint-to-file verification is insufficient.

The historical serializer, model, feature coalescing, padding, and quantization
formulas are not compatible with this contract.

## 5. Engine loading boundary

The active loader requires a nonempty caller-trusted whole-file SHA-256. It
opens one stable file handle and derives size, structural validity, SHA-256,
canonical tensor digests, and parsed parameters from that same byte snapshot.
It never validates one path incarnation and reopens the path to install another.

All parameters are staged in a complete immutable candidate. Wire tensors are
hashed in canonical order before runtime-only conversion. Feature-transformer
packing and dense affine index mapping never appear in the file. Traversing the
runtime object back into canonical logical order must reproduce each wire
tensor digest before the candidate can be committed.

Commit is one all-or-nothing swap at a quiescent boundary. A failed replacement
preserves the previous parameter object, identity, generation, caches, and
outputs. A successful replacement increments the generation and invalidates
all native accumulators and caches from the previous generation. Failure never
selects the historical evaluator or another fallback.

Read-only qualification commands expose network and tensor identities,
stage-by-stage integer inference, and loaded full-refresh versus incremental
comparisons. Normal search consumes only the immutable leased parameter object
through the separately qualified fixed-frame session.

The loader, identity status, tensor status, flat parameter probes, and
full-refresh integer trace, and loaded incremental verifier are now implemented
as qualification paths. They
verify same-handle SHA-256, explicit little-endian decoding, symmetric integer
minima, dense arithmetic envelopes, canonical runtime traversal, generation
increments, preservation of the installed candidate on failure, and every
normative inference stage. The incremental verifier also proves exact feature,
PSQT, dense-stage, and undo equality over exhaustive and directed legal-move
trees. AVX2 and SSE4.1/SSSE3 qualification routes independently execute the
loaded feature, threat, PSQT, and dense affine integers and compare them with
the scalar path, including signed arithmetic boundaries. The production search
session pins one generation and SHA-256 with a move-only lease, uses fixed
feature snapshots and wide transactional accumulators, and rejects replacement
while active. No stale-generation cache or fallback evaluator is consulted.

## 6. Exact acceptance

Checkpoint quantization must equal every wire integer; wire canonical tensors
must equal engine canonical tensors; and trainer-side wire inference must equal
engine inference at every intermediate stage. The corpus covers both colors,
both relations, board renaming, captures, promotions, castling, checks, king
transfers, undo, all eight phase buckets and seven phase boundaries, negative
squared inputs, odd signed PSQT division, skip activation, and outputs on both
sides of zero.

Qualification uses at least all-zero, axis-sentinel, and arithmetic-boundary
synthetic networks. Wrong rounding, axis transposition, relation reversal,
feature-block reversal, phase mismatch, runtime permutation leakage,
saturation, failed replacement, and one-element corruption must all be detected.
