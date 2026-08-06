# Alice Chess rules specification

- **Rules identifier:** `alice-rules-v1`
- **Status:** Normative
- **Board model:** two 8x8 layers, `A` and `B`
- **Fixture contract:** `tests/alice/fixtures/rules-v1.json`

This document defines the public rules and position contract for
Alice-Stockfish. The fixture file is the executable companion to this text. A
legacy behavior that conflicts with this specification is compatibility
evidence, not a reason to weaken the rules.

The underlying Alice move rule follows the established definition used by
[PyChess](https://www.pychess.org/variants/alice) and the
[Compact Chess Interchange Format](https://ccif.sourceforge.net/formal-board-rules.html):
a move is played on one board, its corresponding destination on the other
board must be vacant, and the mover transfers after the move. This version
deliberately excludes en passant, matching the frozen Fairy-Stockfish Alice
configuration.

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## 1. Position model

1. Layers `A` and `B` share the coordinates `a1` through `h8`.
2. Every piece has a color, type, square, and exactly one layer.
3. The initial position is orthodox chess on layer `A`; layer `B` is empty:

   ```text
   rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
   ```

4. A coordinate MUST NOT be occupied on both layers simultaneously. This is
   preserved by every legal move because the arrival square on the opposite
   layer must be empty. Position input that violates this invariant is invalid.
5. Piece movement, occupancy, rays, attacks, captures, and king safety are
   layer-local unless this document explicitly describes the post-move
   transfer.
6. Standard chess piece types and colors are used. There are no drops, gates,
   walls, null transfers, or optional transfers.
7. The primitive occupancy query takes an explicit layer, such as
   `occupancy_on(Board)`. `board_of(square)` and any square-derived occupancy
   shorthand require an occupied coordinate and MUST reject an empty square;
   an empty coordinate does not imply layer `A`.

## 2. Ordinary move, capture, and transfer

For a non-castling move by a piece on `(from, sourceLayer)`, legality is
evaluated in this order:

1. Apply orthodox movement geometry on `sourceLayer` only. Pieces on the other
   layer neither block rays nor provide capture targets.
2. The move to `to` MUST be legal on `sourceLayer`:
   - a quiet destination is empty on `sourceLayer`;
   - a capture destination contains an opposing piece on `sourceLayer`;
   - a friendly piece on the destination makes the move illegal.
3. For a capture, remove only the captured piece on `sourceLayer`.
4. Evaluate the provisional source-board move under orthodox king-safety
   rules. In particular, a king may not use transfer alone to escape an attack
   if its ordinary move would still be illegal on the source board.
5. The square `(to, opposite(sourceLayer))` MUST be empty. Occupancy by either
   color makes the move illegal; it is never a second capture.
6. If the move is a promotion, replace the pawn on `sourceLayer` at `to` with
   the selected promoted piece. Promotion is effective immediately.
7. Transfer the resulting moving piece to `(to, opposite(sourceLayer))`.
8. Evaluate the complete two-layer final position. The moving side's king MUST
   be safe in that final position.

The operation is atomic to callers: no provisional state is a completed game
state. `do_move` and `undo_move` MUST respectively apply and restore the piece,
layer, captured piece, clocks, rights, and all derived state exactly.

### 2.1 Pawns

- Pawns keep their orthodox color-relative direction on both layers.
- A single or initial double push is tested using occupancy on the pawn's
  source layer. The final square on the opposite layer must also be empty.
- A pawn captures only an opposing piece on its source layer and then
  transfers.
- A pawn move resets the halfmove clock as in orthodox chess.

## 3. Check, checkmate, and stalemate

1. A king is in check only from opposing pieces on the king's layer.
2. Kings on adjacent coordinates but different layers do not attack each
   other. They may not become adjacent on the same layer.
3. A legal move MUST satisfy both the provisional source-board check rule and
   final two-layer king safety described in section 2.
4. A move on the checked king's layer MAY capture the checker before the mover
   transfers away. A move originating on the other layer MAY evade the check
   by transferring to an interposition square on the king's layer. It cannot
   capture that checker across layers: the checker's occupied coordinate would
   block the transfer.
5. Moving a blocker on the king's layer does not evade a line check merely by
   occupying an interposition square provisionally: the blocker transfers away,
   so the final line would reopen.
6. Check delivered by the mover, a promoted mover, a castling rook, or a
   discovered line is evaluated only after every capture and transfer is
   complete.
7. Checkmate is check with no legal Alice move. Stalemate is no legal Alice
   move while not in check. Their results are the orthodox loss and draw.
8. Kings are never captured; moves exposing one's own king are illegal.

## 4. Promotion

1. A pawn that completes its source-board move on the last rank MUST promote.
2. Promotion occurs on the source layer before transfer. The pawn is replaced
   there by the chosen queen, rook, bishop, or knight.
3. The promoted piece, not the pawn, then transfers to the corresponding square
   on the opposite layer. It participates immediately in final check
   evaluation.
4. Quiet and capturing promotions use the same transfer rule.
5. The opposite-layer destination must be empty before the move.
6. UCI promotion suffixes are mandatory and lowercase: `q`, `r`, `b`, or `n`.

## 5. Castling

Only orthodox castling is supported; Chess960 castling is outside rules v1.

Castling is legal only when all of the following hold:

1. The corresponding castling right exists.
2. The king and the participating rook are on their orthodox start squares,
   on the same source layer.
3. Source-layer occupancy and attacks satisfy the orthodox rules: the path is
   clear, the king is not in check, and it does not cross or provisionally end
   on an attacked source-layer square.
4. Both final squares on the opposite layer are empty.
5. After the operation, the transferred king is not attacked on its new layer.

The king and rook first take their orthodox castling destinations, then both
transfer as one atomic move. For white kingside castling, for example, the king
finishes on the opposite layer at `g1` and the rook at `f1`. Both retain their
piece identities. Castling clears the relevant rights exactly as in orthodox
chess.

Public UCI castling strings are `e1g1`, `e1c1`, `e8g8`, and `e8c8`. Internal
king-to-rook encodings MUST NOT leak through the public protocol.

## 6. En passant is disabled

- Double pawn pushes remain legal.
- No double push creates an en-passant target.
- No en-passant capture is generated or accepted.
- The FEN en-passant field MUST be `-`. A non-`-` field is invalid input rather
  than a request that may be silently ignored.

This is intentional. The frozen Alice variant sets `enPassantRegion = 0`, and
the public rules remove the legacy parser's ambiguity by rejecting stale
targets.

## 7. FEN contract

Rules v1 accepts two placement encodings. All emitted FEN is canonical compact
FEN.

### 7.1 Canonical compact placement

Each rank describes the eight shared coordinates once:

- an unmarked piece letter places the piece on layer `A`;
- `|` immediately before a piece letter places that piece on layer `B`;
- `|` does not consume a coordinate;
- digits count empty shared coordinates using ordinary FEN run-length rules;
- empty-square runs use canonical decimal notation without leading zeros;
- `|` before a digit, slash, another `|`, or end of an incomplete rank is
  invalid.

Input compatibility has one narrow exception for the frozen historical
opening book. Its first position contains a redundant `|` after a rank has
already expanded to all eight coordinates. A parser accepts and discards that
single terminal marker. Canonical output never emits it. This preserves the
published book byte-for-byte while keeping incomplete or doubled markers
invalid.

Example: a black king on `A:e8`, a white pawn on `B:e4`, and a white king on
`A:e1`:

```text
4k3/8/8/8/4|P3/8/8/4K3 w - - 0 1
```

The remaining fields are side to move, castling rights, the mandatory `-`
en-passant field, halfmove clock, and fullmove number. Castling rights are
invalid unless their king and rook exist on their required squares and share a
layer.

### 7.2 Legacy 16-wide input

A parser MUST accept a placement whose every rank expands to 16 cells. It is
interpreted as:

```text
A:a through A:h, then B:a through B:h
```

Run lengths may cross the layer boundary and may therefore be two decimal
digits, up to `16`, without leading zeros. The 16-wide form does not use `|`. After expansion, the
parser folds the two halves onto the shared coordinates, rejects double
occupancy, and canonicalizes to compact placement.

The compact example above is equivalent to:

```text
4k11/16/16/16/12P3/16/16/4K11 w - - 0 1
```

Mixed-width ranks, mixed compact/16-wide markers, double occupancy, missing or
extra ranks, and any expanded width other than 8 or 16 are invalid.

### 7.3 Accepted material domain

Input retains the orthodox reachable-material limits of the pinned chassis:

- exactly one king of each color;
- no more than eight pawns or sixteen total pieces per color;
- no more than 32 total pieces;
- no unpromoted pawn on rank 1 or rank 8;
- promoted surplus per color is bounded by the missing pawns, using
  `max(knights-2, 0) + max(bishops-2, 0) + max(rooks-2, 0) + max(queens-1, 0)
  <= 8-pawns`.

These checks apply equally to compact and legacy input. Invalid input MUST be
rejected transactionally and MUST NOT leave a partially updated position.

## 8. UCI move contract and layer ambiguity

The public move grammar remains coordinate-compatible with the frozen engine:

```text
<from><to>[promotion]
```

There is no board prefix or suffix. The source layer comes from the current
position, and the destination layer is necessarily the opposite layer. Thus
`e2e4` can mean `A:e2 -> B:e4` in one position and `B:e2 -> A:e4` in another.

The parser MUST enumerate legal internal moves and accept a string only when it
identifies exactly one move in the current position. Zero matches or multiple
matches are protocol errors. Parsing is case-sensitive; files and promotion
suffixes MUST be lowercase. The serializer MUST emit the canonical coordinate
string. A layer selector, internal move bits, or an arbitrary first match MUST
NOT be used to resolve a collision.

## 9. Hashing and repetition

1. The full position key MUST distinguish color, piece type, square, and layer
   for every piece, plus side to move and castling rights.
2. Every square-sensitive derived key used by search or evaluation, including
   pawn, non-pawn, and minor-piece keys, MUST distinguish the layer.
3. A count-only material key MAY remain layer-independent, but it MUST NOT be
   used as full position, transposition, or repetition identity.
4. Incremental keys after a move MUST equal keys recomputed from the resulting
   position. Undo MUST restore every key bit-for-bit.
5. Repetition equality requires the same layer-aware piece placement, side to
   move, and castling rights. The halfmove and fullmove counters are not part of
   repetition identity; they retain their orthodox rule functions.
6. Equal square overlays with different layer assignments are not repetitions
   and MUST have different full position keys.
7. Threefold repetition is recognized only after three occurrences of the
   complete layer-aware identity. Any fast cycle detector, including cuckoo
   repetition, remains disabled until it is layer-aware and fixture-covered.
8. Exact numeric Zobrist constants are an implementation detail. Fixtures pin
   equality and inequality relations, recomputation, and undo behavior rather
   than unstable numeric key values.

The frozen implementation stored `mirrorBoard` separately but did not mix it
into its piece-square key. That known collision is diagnostic evidence; rules
v1 explicitly requires the corrected behavior.

## 10. Evidence and precedence

The frozen compatibility witnesses are:

- Fairy-Stockfish commit
  [`4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79`](https://github.com/fairy-stockfish/Fairy-Stockfish/tree/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79),
  especially `variant.cpp`, `movegen.cpp`, `position.cpp`, `position.h`, and
  `uci.cpp`;
- legacy executable SHA-256
  `B70AFE03EC9A67258CD7B5B848C46FC9E5C83F53B9F2825E9A5946FEEFB59599`.

Precedence is:

1. this rules specification;
2. `rules-v1.json` expected results;
3. frozen source and executable as differential witnesses;
4. explanatory external references.

The frozen executable agrees with the move, transfer, check, promotion,
castling, no-en-passant, and FEN normalization fixtures. It is intentionally
not a rules authority for corrected layer-aware hashing, repetition, strict
en-passant FEN rejection, or fail-closed UCI collision handling.

## 11. Explicit non-contracts

Rules v1 does not define Chess960, tablebase compatibility, a native NNUE
feature layout, or global perft totals. Perft values become normative only
after an independent rules implementation and the optimized engine agree on
the same versioned fixture corpus.
