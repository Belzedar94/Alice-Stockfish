# ADR-0001: Use official Stockfish as the Alice engine chassis

- **Status:** Accepted
- **Date:** 2026-08-06
- **Chassis commit:** `762dd1da9a5db458180b2c5db6c53dc40ec61e1a`
- **Legacy reference:** `4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79`

## Context

The historical Alice implementation is a small experimental delta on an old
Fairy-Stockfish variant framework. It is useful evidence for notation,
transfer mechanics, and compatibility tests, but its position identity,
static exchange evaluation, and neural features do not fully encode the board
layer. Its surrounding search and NNUE code also differs substantially from
current official Stockfish.

## Decision

Alice-Stockfish is a dedicated engine based on the pinned official Stockfish
commit. Alice state and rules will be ported deliberately into current
`Position`, move generation, hashing, search, and NNUE interfaces. The legacy
branch is retained as a read-only reference and differential witness; it will
not be merged into `main`.

The public UCI surface remains coordinate-move compatible with the legacy
engine and exposes Alice as the only supported variant. Standard chess may be
used internally for chassis regression tests, but it is not a release target.

## Required invariants

1. Every piece has exactly one board layer in addition to its square.
2. Occupancy and attacks always receive or derive an explicit layer.
3. Position, pawn, minor-piece, repetition, and transposition identities
   distinguish otherwise identical square overlays with different layers.
4. `do_move` and `undo_move` restore the complete layer-aware state exactly.
5. A move is legal only after capture, transfer, promotion, castling transfer,
   and final king-safety evaluation have completed.
6. Unsupported classical optimizations fail closed until a layer-aware proof
   and test exist.

## Consequences

- The port is larger than copying the legacy `mirrorBoard` patches.
- Classical tablebases, cuckoo repetition, insufficient-material shortcuts,
  and unverified SEE consumers start disabled.
- Legacy NNUE support is a compatibility bridge with an exact format gate, not
  the native evaluation architecture.
- The native network and its training tools must encode layer information and
  share a versioned feature contract.

## Acceptance gate

Engine implementation starts only after the rule specification, fixture
schema, legacy audit, compatibility contract, and measurement protocol are
reviewable together. A release additionally requires reference-interpreter
agreement, deterministic make/undo and hash tests, all local time-control
gates, and reproducible binary receipts.
