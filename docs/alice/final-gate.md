# Final Acceptance Gate

Status: normative release-evidence contract.

The final gate has two independent local batteries. Neither may substitute for
the other.

## Exact-LOS battery

VSTC `2+0.02`, STC `10+0.1`, and LTC `30+0.3` each start from zero with pinned
inputs. Every control must score more than 100 games and seal at a displayed
paired LOS of exactly `100.0`. `0.0`, `INCONCLUSIVE`, interruption, or invalid
evidence fails this battery.

## Release-only fixed Elo battery

Only when preparing the release, the same candidate completes exactly 400
VSTC, 300 STC, and 200 LTC games without early stopping. This battery measures
the release Elo sample; it is not run before the exact-LOS battery. Its
conclusion is `FIXED_COMPLETE`. Aborts are recorded separately and a clean
aggregate requires none.

## Common invariants

- contender is always engine 1;
- every opening is a complete color-swapped pair;
- Threads is 1, Hash is 512 MiB, Move Overhead is 10 ms, and exactly two
  persistent pair processes run;
- no score adjudication, SPRT, combined LOS, or external local adjudicator;
- attempt ordinals, not completion timing, determine admission;
- the immutable seal precedes drain; and
- interrupted statistics never resume.

The detailed formula, terminal protocol, classifications, commands, and receipt
rules are in [measurement.md](measurement.md).

Completion of these batteries is necessary but not sufficient for release. A
trained AliceNative-v1 network, G1-G8 qualification, four platform artifacts,
triple bench, load-failure evidence, and clean 200-pair VSTC, STC, and LTC
official OpenBench shadow audits remain mandatory.
