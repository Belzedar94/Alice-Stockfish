# Legacy Fairy-Stockfish Alice audit

- **Status:** Frozen reference; not suitable as the production chassis
- **Audit target:** [`4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79`](https://github.com/fairy-stockfish/Fairy-Stockfish/commit/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79)
- **Initial Alice commit:** [`ddd1219d`](https://github.com/fairy-stockfish/Fairy-Stockfish/commit/ddd1219d)
- **Audit date:** 2026-08-06

## Scope and provenance

The audited branch interval starts after
[`d3e9bd93`](https://github.com/fairy-stockfish/Fairy-Stockfish/commit/d3e9bd9398e0d0185c6dc32aaec41dbdb3bdfadb)
and ends at the frozen target. A full revision walk over that interval contains
12 commits: 11 commits on the first-parent path and the merged maintenance
commit `fbde7583`. The functional Alice work is concentrated between
`ddd1219d` and `2b0a999d`; the final two commits update wheel support and the
package version. This distinction matters: the interval is the reproducible
source delta, but not every commit in it changes Alice rules.

The public source comparison is available as the
[`d3e9bd93...4b1940a8` diff](https://github.com/fairy-stockfish/Fairy-Stockfish/compare/d3e9bd9398e0d0185c6dc32aaec41dbdb3bdfadb...4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79).
The local frozen references produced these receipts:

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `stockfish_x86-64-bmi2_alice.exe` | 1,690,112 bytes | `B70AFE03EC9A67258CD7B5B848C46FC9E5C83F53B9F2825E9A5946FEEFB59599` |
| `alice_run2rl_e40_l09.nnue` | 47,721,376 bytes | `9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9` |
| `alice.epd` | 2,539,444 bytes | `BCD89D9FC3EA81FEB95932EB64D6B6F15AD25CC04CDCC9E0440F097CFFB8CCF6` |

The executable identifies itself as `Fairy-Stockfish 040925`.

## Legacy position model

The implementation keeps the ordinary 64-square `board[]` and piece
bitboards. A `StateInfo::mirrorBoard` bitboard selects which occupied squares
belong to the second physical board
([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/position.h#L43-L72)).
This overlay can represent Alice positions because a legal transfer may not
land on an occupied corresponding square, so both physical boards cannot
legally contain pieces at the same coordinate.

The variant derives from chess, enables `mirrorBoard`, and disables en passant
by clearing `enPassantRegion`
([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/variant.cpp#L75-L81)).
Its FEN marks a second-board piece with `|` immediately before the piece
letter. The parser also accepts the historical side-by-side representation;
the serializer emits the compact `|` form.

## Movement behavior verified in source

The following behavior is useful reference material for the new rules
implementation:

- Move geometry and captures are evaluated on the mover's source board.
- A destination occupied only on the other board blocks the move. A target on
  the source board can be captured, after which the mover occupies the
  corresponding square on the other board
  ([move-generation gate](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/movegen.cpp#L28-L33)).
- `do_move` transfers a non-castling mover by updating `mirrorBoard` after the
  ordinary square move
  ([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/position.cpp#L1843-L1855)).
- Castling checks the source-board path and both opposite-board landing
  squares. The king and rook layer bits are then transferred together
  ([path check](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/position.h#L1210-L1225),
  [transfer](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/position.cpp#L2325-L2353)).
- Several move, check, and king-safety paths filter occupancy through the
  relevant layer. These paths are valuable test witnesses, but they do not
  repair the global identity, SEE, and pinning defects below.

With a complete `uci`/`uciok` and `isready`/`readyok` handshake, the frozen
executable gives the following Alice start-position perft sequence:

| Depth | Nodes |
| ---: | ---: |
| 1 | 20 |
| 2 | 400 |
| 3 | 9,384 |
| 4 | 219,236 |
| 5 | 5,910,465 |

These values are legacy regression witnesses, not independent proof of the
rules. Sending `setoption name UCI_Variant value alice` before `uciok` can race
startup and produce non-Alice numbers, so every reproducer must use the full
handshake.

## Critical finding: layer is absent from position identity

`mirrorBoard` is not mixed into either the full Zobrist key or the pawn key.
`Position::set_state()` hashes only `(piece, square)`, side to move, castling,
en-passant, holdings, and check counters
([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/position.cpp#L620-L674)).
Incremental updates likewise XOR only the mover's ordinary source and
destination squares
([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/position.cpp#L1713-L1727)).

This is a deterministic state collision, not an ordinary probabilistic hash
collision. The frozen executable reports the same key for these distinct
Alice positions:

```text
Board A rook: 4k3/8/8/8/8/8/R7/4K3 w - - 0 1
Board B rook: 4k3/8/8/8/8/8/|R7/4K3 w - - 0 1
Key for both: DBA66C81DAEE5727
```

The equivalent pawn pair also has the same `pawnKey`, because the pawn-key
path uses the same layer-blind piece-square term. Consequently, transposition
table entries, repetition detection, pawn caches, and any key-derived state
can be reused across positions with different legal moves and attacks. This
finding alone prevents adopting the branch as the production chassis.

## Critical finding: classical SEE does not model Alice recaptures

`Position::see_ge()` starts from the pre-move target square, constructs one
ordinary occupancy, and obtains attackers through the layer-unqualified
`attackers_to(to, occupied)` overload
([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/position.cpp#L2485-L2558)).
It neither transfers the capturing piece to the opposite board nor alternates
the target layer for subsequent recaptures. Its pin filtering also consumes
`pinners()` and `blockers_for_king()` produced by `slider_blockers()`, whose
occupancy and sniper scans use the overlaid `pieces()` bitboard
([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/position.cpp#L847-L930)).

This layer-blind result is used for ProbCut admission, capture partitioning,
main-search pruning, and quiescence pruning
([MovePicker](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/movepick.cpp#L90-L100),
[search](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/search.cpp#L1155-L1189)).
The production port must supply a layer-aware Alice SEE or disable each
consumer until equivalence is demonstrated.

## Critical finding: legacy NNUE is board-blind

The active `HalfKAv2Variants` feature index contains perspective, oriented
square, piece identity, and king square, but no board layer
([source](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79/src/nnue/features/half_ka_v2_variants.cpp#L32-L61)).
Incremental updates carry piece and from/to square only; `DirtyPiece` has no
layer field. Therefore two positions that differ only in `mirrorBoard` produce
the same active feature set, and layer changes cannot be expressed in the
accumulator delta. The historical network remains useful as a reproducibility
baseline, but it is not a complete Alice evaluator.

## Test coverage finding

At the frozen commit, the `tests/` tree has no dedicated Alice fixture, perft,
make/undo, hash, SEE, castling, FEN, or NNUE parity test. The only textual
`Alice` match is a player name (`SmartAlice`) inside an unrelated PGN. The
verified perft sequence above therefore depends on a manual executable probe.

## Disposition

The branch is retained unchanged as a read-only differential witness for FEN,
move transfer, castling, perft, UCI behavior, and historical NNUE output. No
legacy source is merged wholesale into `main`.

Before any corresponding optimization is enabled in Alice-Stockfish, the new
implementation must provide:

1. explicit board-layer state in every position identity and cache contract;
2. exact make/undo and full-key recomputation tests;
3. layer-explicit occupancy, attack, pin, and SEE interfaces;
4. an independent rules interpreter agreeing on legal moves and perft; and
5. full-refresh and incremental NNUE parity tests.
