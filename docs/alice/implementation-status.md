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

## Deliberately disabled paths

The following orthodox shortcuts remain unavailable until they receive a
board-aware implementation and dedicated coverage:

- classical static exchange evaluation and its pruning decisions;
- cuckoo upcoming-repetition detection;
- Syzygy probing and root ranking;
- null-move and tuned orthodox search shortcuts;
- the current Stockfish accumulator and threat features; and
- insufficient-material shortcuts.

Normal playing search is not a supported deliverable at this milestone. The
next acceptance gate is a deterministic safe search path followed by the
strict legacy-network compatibility loader described in
[`legacy-nnue-compatibility.md`](legacy-nnue-compatibility.md).
