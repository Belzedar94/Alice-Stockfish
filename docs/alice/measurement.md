# Alice Strength Measurement Contract

Status: normative contract.

This document freezes the local Alice-Stockfish acceptance battery. It is
separate from OpenBench and from the fixed-game release battery.

## 1. Engine roles and pinned inputs

- `engine1` is always the contender and `engine2` is always the frozen
  reference.
- The contender plays once as each color in every complete pair.
- Record the source commit, build command, binary SHA-256, evaluator identity,
  network SHA-256, book SHA-256, opening seed, pair-worker SHA-256, runner
  SHA-256, and every UCI option before starting.
- Both engines use one search thread, 512 MiB of hash, and a 10 ms move
  overhead. The controller keeps exactly two persistent pair processes.
- Binaries, networks, books, runner code, and options never change inside one
  timing-control result.

## 2. Frozen timing controls

Each control is an independent experiment with its own evidence root.

| Preset | Clock | Increment | Maximum scored games | Maximum attempted games |
| --- | ---: | ---: | ---: | ---: |
| VSTC | 2 s | 0.02 s | 64,000 | 64,000 |
| STC | 10 s | 0.1 s | 64,000 | 64,000 |
| LTC | 30 s | 0.3 s | 64,000 | 64,000 |

The UCI forms are `2+0.02`, `10+0.1`, and `30+0.3`. Samples and evidence from
different controls remain separate.

## 3. Pair, opening, and ordering rules

One pair is the atomic sampling unit:

1. Select one position from the pinned Alice EPD book.
2. Play contender versus reference in the first color assignment.
3. Replay the exact position with colors swapped.
4. Admit both results together, or admit neither.

The versioned opening schedule is independent of completion timing. For each
cycle it ranks every entry by SHA-256 over the schedule identifier, unsigned
64-bit seed, cycle number, and entry index. Every entry occurs exactly once per
cycle.

The controller dispatches exactly two pair attempts at a time, buffers
completed results, and admits only the contiguous attempt-ordinal prefix. A
quickly completed higher ordinal never replaces an unresolved lower ordinal.
Only complete color-swapped pairs enter W/L/D, Elo, or LOS accounting.

## 4. Terminal and abort policy

The local battery has no external score adjudication. Resign thresholds,
evaluation win or draw thresholds, tablebase adjudication, and maximum-ply
draws are disabled. Valid scored endings are Alice checkmate, stalemate,
rule-defined draws, and flag fall.

Each engine must publish the strict Alice terminal record when it returns no
move. A missing, malformed, contradictory, or move-followed terminal record is
a protocol failure. The safety ply limit is a policy failure, never a draw.

Every game has one machine classification. `SCORABLE_NATURAL` and
`SCORABLE_CLOCK` are the only strength-bearing classes. An operational failure
discards its complete pair. A protocol, semantic, evidence, policy, or unknown
failure invalidates the control and drains the other already-dispatched pair
without scoring it. No failure is converted into an automatic strength loss.

## 5. Frozen paired statistics

Let the contender-perspective pentanomial counts be
`[LL, LD, DD_or_WL, DW, WW]` and their normalized pair observations be
`[0, 0.25, 0.5, 0.75, 1]`. For `N` admitted pairs, compute the population mean
and variance of those observations, then:

```text
standard_error = sqrt(variance / N)
LOS = Phi((mean - 0.5) / standard_error)
```

A zero-variance sample has LOS `0.5` at a mean of `0.5`; otherwise it has the
corresponding exact extreme. The displayed percentage is exactly
`format(100.0 * LOS, ".1f")`. The receipt also records the binary64 hexadecimal
probability so the display can be reproduced.

Decisive-only binomial LOS, game-independent LOS, combined LOS, and SPRT do not
belong to this contract.

## 6. Exact-LOS stopping rule

Evaluate the display after every admitted pair. A control may stop at an
extreme only when:

- more than 100 games have been scored;
- the scored-game count is even; and
- the display is exactly `0.0` or `100.0`.

`100.0` is a pass and `0.0` is a failure. If neither extreme appears, reaching
either the 64,000 scored-game cap or the 64,000 attempted-game cap produces
`INCONCLUSIVE`.

The controller creates and flushes an immutable seal before draining a later
in-flight pair. A result beyond that seal is `excluded_after_seal`; it is never
described as discarded and never changes the sealed W/L/D, pentanomial, Elo,
or LOS.
The control receipt embeds that exact canonical seal payload and its SHA-256.
Aggregation recomputes the digest and requires the seal's control, mode,
attempt ordinal, admitted count, W/L/D, pentanomial, statistics, stop reason,
and conclusion to agree with the final result. A syntactically valid arbitrary
digest or a post-seal statistical rewrite is rejected.

The exact battery passes only with `100.0` at VSTC, STC, and LTC. Interrupted
experiments do not resume statistically.

## 7. Release-only fixed Elo battery

The 400/300/200 battery is run only when preparing a release, to measure its
published Elo sample. It is not a prerequisite for the earlier exact-LOS
battery. It has no early stopping:

| Preset | Admitted games |
| --- | ---: |
| VSTC | 400 |
| STC | 300 |
| LTC | 200 |

It uses the same pair, color, opening, terminal, and abort rules. Its conclusion
is `FIXED_COMPLETE`; the fixed sample is not relabeled as a LOS pass or fail.
A clean aggregate requires zero discarded pairs and zero abort evidence.

## 8. Evidence and commands

Before a control starts, snapshot the pinned book, pair worker, runner core,
engine binaries, networks, and rewritten worker definition. The control input
inventory preserves absolute paths and SHA-256 values for both runner-code
snapshots. Every aggregation pass reopens those files and recomputes both
digests; a missing, modified, or merely self-declared runner identity fails
closed. Run a complete pair on each persistent process as preflight. The
preflight is not part of the statistical sample.

Run one control from an `alice-acceptance-run-definition-v1` file:

```text
python -m tools.alice_acceptance \
  --definition <absolute-definition.json> \
  --evidence-root <new-absolute-directory>
```

Every pair receives create-only PGN, machine result, request, and response
files. Artifact hashes and the result-core hash are verified before admission.
Interrupted controls receive an interruption receipt and cannot resume.

Aggregate the three exact controls with:

```text
python -m tools.alice_acceptance.aggregate --mode exact-los \
  --run-id <battery-id> --vstc <receipt> --stc <receipt> --ltc <receipt> \
  --output <new-receipt.json>
```

Use `--mode fixed-final` for the separate 400/300/200 battery. Aggregation
rejects nonzero abort evidence, missing controls, a non-extreme exact result,
or a wrong fixed sample size. It embeds each canonical control receipt,
reproduces its statistics, and requires one shared book, runner, binary,
evaluator, network identity, UCI option set, and opening seed across VSTC, STC,
and LTC. The exact-LOS and release-only fixed batteries may declare different
opening seeds; each seed remains immutable within its own three controls.

## 9. Monitoring and final interpretation

Monitor state, attempted and scored games, complete and admitted pairs, W/L/D,
pentanomial, Elo, LOS, abort classes, evidence path, and last progress. Fifteen
minutes without a new complete pair requires investigation and disclosure.

OpenBench results cannot replace either local battery. Its scheduling,
adjudication, and shadow-audit contracts are defined in
[openbench.md](openbench.md).
