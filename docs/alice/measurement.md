# Alice Strength Measurement Contract

Status: normative Phase 0 contract.

This document freezes the local acceptance battery for Alice-Stockfish. It is a
paired, exact-LOS experiment. It is deliberately separate from OpenBench and
from any fixed-game release tournament.

## 1. Engine roles and pinned inputs

- `engine1` is always the Alice-Stockfish contender under test.
- `engine2` is always the frozen reference.
- The contender must play once as each color in every completed pair.
- Record the executable path, source commit, build command, binary SHA-256,
  loaded network SHA-256, opening-book SHA-256, runner revision, and statistics
  wrapper revision before starting.
- Both engines use one search thread and 512 MiB of hash. The battery runs two
  concurrent match workers. All other UCI options must be recorded and held
  equal unless the experiment explicitly tests one of them.
- Do not mix binaries, networks, books, runners, or option sets inside one
  timing-control result.

## 2. Frozen timing controls

Each timing control is an independent experiment with its own log and receipt.

| Preset | Clock per engine | Increment | Maximum scored games |
| --- | ---: | ---: | ---: |
| VSTC | 2 s | 0.02 s | 64,000 |
| STC | 10 s | 0.1 s | 64,000 |
| LTC | 30 s | 0.3 s | 64,000 |

The notation is therefore `2+0.02`, `10+0.1`, and `30+0.3`. Time values are in
seconds. The three presets may run concurrently, but their samples and evidence
must remain separate.

## 3. Pair and opening rules

One pair is the atomic sampling unit:

1. Select one opening from the pinned Alice EPD book.
2. Play the opening with contender and reference in the first color assignment.
3. Replay the exact same opening with colors swapped.
4. Admit both results together, or admit neither result.

The opening-selection seed and selected EPD must be recoverable from the log.
Only complete color-swapped pairs enter W/L/D, Elo, or LOS accounting. The
scored-game count must consequently always be even.

## 4. Result and adjudication policy

The local battery has no external score adjudication. Disable runner-level
resign thresholds, evaluation-based win thresholds, evaluation-based draw
thresholds, and tablebase adjudication. The only valid game endings are those
produced by the Alice rules and UCI game state, including checkmate, stalemate,
repetition, rule-defined draws, and flag fall.

An illegal move, malformed position, process exit, protocol failure, or runner
exception is not an automatic loss. Discard the entire affected pair and audit
the cause. Never convert a dialect or rules disagreement into a strength result.

## 5. Exact stopping rule

Evaluate the displayed LOS to one decimal place after each admitted pair. A
timing control may stop only when all of the following are true:

- more than 100 games have been scored;
- the scored-game count is even; and
- the displayed LOS is exactly `0.0` or exactly `100.0`.

When an extreme is observed, seal that pair-complete snapshot. A pair already
in flight may finish for orderly cleanup, but it must not alter the sealed
statistical receipt. If neither extreme is reached, stop at 64,000 scored games
and report that timing control as inconclusive.

The two extremes have different acceptance meanings:

- `100.0` is a contender pass for that timing control.
- `0.0` is a conclusive contender failure for that timing control.

Alice-Stockfish passes the Phase 0 local battery only with `100.0` LOS at VSTC,
STC, and LTC. Thus the acceptance condition is **LOS100 x 3**, with more than
100 scored games in each experiment. A mix of passes, failures, or inconclusive
results is not a pass. A wrapper message saying only that a statistical gate
closed is insufficient; the receipt must state which extreme was reached.

Interrupted experiments do not resume statistically. Restart the affected
timing control from zero with the same pinned inputs, or declare a new run with
new identifiers.

## 6. Abort accounting and validity

For every timing control, record:

- pair attempts, completed pairs, and admitted games;
- discarded pairs grouped by exact reason;
- offending opening, color assignment, and final valid position for each abort;
- illegal-move, timeout, process-exit, and protocol-error counts;
- natural terminal reasons and clock losses; and
- the final W/L/D, Elo estimate, LOS, and stop reason.

Unexplained aborts must be zero. Any abort caused by an Alice rule, FEN, move,
or board-transfer disagreement invalidates strength interpretation until the
disagreement is reproduced and classified. A nonzero operational abort rate
must be disclosed with both attempted and admitted sample sizes.

Before starting the full battery, replay a small paired preflight through the
same position parser and move path used by the tournament. The preflight must
show exact opening reconstruction, legal-move agreement, correct color swaps,
and clean completion without changing the frozen measurement policy.

## 7. Monitoring and final receipt

During a run, keep a five-minute status table covering all three timing controls.
For each preset it should show process state, scored games, complete pairs,
W/L/D, Elo, LOS, discarded pairs, log path, and last-progress time. Fifteen
minutes without a new completed pair requires investigation and must be noted in
the receipt.

A final receipt is complete only if it contains:

- a unique run identifier and UTC start/end times;
- every pinned input and SHA-256 listed in section 1;
- the exact command line and UCI option dump;
- one result block per timing control;
- the abort audit and the sealed pair-complete snapshot; and
- an explicit conclusion: `PASS`, `FAIL`, or `INCONCLUSIVE`.

OpenBench results cannot replace this battery. Its scheduling, adjudication, and
statistical contracts are defined separately in [openbench.md](openbench.md).
