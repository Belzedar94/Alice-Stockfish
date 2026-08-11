# Alice-Stockfish 1.0 release draft

Status: preparation draft. Do not publish until the fixed 400/300/200 panel,
release builds, bench checks, and artifact checksums are complete.

Alice-Stockfish 1.0 is the first stable release of a dedicated UCI engine for
[Alice chess](https://www.chessvariants.com/other.dir/alice.html), derived from
a current official Stockfish framework with native two-board rules, an
Alice-specific search, and a strict compatibility implementation of the
published legacy Alice NNUE.

This is a **Legacy NNUE compatibility release**. The experimental native NNUE
V2 checkpoint is not included: its full-search integration was validated, but
the pilot weights did not meet the strength gates. A separate native-data
program is producing board-aware SAME/OTHER training data for a future network.

## Strength

Measured against the frozen Fairy-Stockfish reference (`Fairy-Stockfish
040925`) with the exact same `alice_run2rl_e40_l09.nnue` network on both sides,
the frozen Alice opening book, 1 thread, 512 MiB hash, 10 ms move overhead, and
fixed game counts. Every opening is played with colors swapped from the shared
seed `20260811`; adjudication is disabled and only complete pairs enter the
result.

<!-- Replace every PENDING field from the sealed fixed-final receipt. -->

| Time control | Games | Score | Elo |
|---|---:|---|---:|
| 2s + 0.02s | 400 | PENDING | **PENDING** |
| 10s + 0.1s | 300 | PENDING | **PENDING** |
| 30s + 0.3s | 200 | PENDING | **PENDING** |

The release panel must complete with zero discarded pairs, zero abort evidence,
and no time-control or input changes. Its fixed sample is a measurement rather
than a pass/fail relabeling.

## Features

- Native Alice move generation and legality on two boards, including strict
  terminal handling and reproducible rules tests.
- Alice-specific search with layer-aware threat ordering, arrival-board capture
  staging, and conservative pruning where an Alice-safe SEE is unavailable.
- Exact `LegacyAliceExact` evaluation for the published
  `alice_run2rl_e40_l09.nnue` network, with full-refresh and incremental parity.
- Fail-closed network loading: missing, corrupt, incompatible, or ambiguous
  inputs cannot silently select another evaluator.
- Standard UCI, deterministic Alice bench, multi-threading, and large hash
  support.

## Usage

Download the binary matching your CPU (`x86-64-bmi2` for modern Intel and AMD
processors, or `x86-64-avx2` as the portable fallback) together with
`alice_run2rl_e40_l09.nnue`. Keep the network next to the executable, or select
its path explicitly before searching:

```text
setoption name Alice Evaluation value Legacy
setoption name Use NNUE value true
setoption name Alice_Frozen_Network value true
setoption name EvalFile value <path>/alice_run2rl_e40_l09.nnue
```

The release network has SHA-256
`9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9`.
The engine reports the selected evaluation backend and the SHA-256 of the bytes
it loaded. If the required network cannot be authenticated, evaluation and
search fail closed. Run `bench` after loading the network; every release binary
must report exactly `202963` nodes searched.

## Checksums (SHA-256)

<!-- Replace the binary placeholders only after the final release builds pass. -->

```text
PENDING  alice-stockfish-1.0-windows-x86-64-bmi2.exe
PENDING  alice-stockfish-1.0-windows-x86-64-avx2.exe
PENDING  alice-stockfish-1.0-linux-x86-64-bmi2
PENDING  alice-stockfish-1.0-linux-x86-64-avx2
9f9e557015a55c0a6981db64e1f3044dedb91fd8a8c1a6d4f3c45d0eee91fbd9  alice_run2rl_e40_l09.nnue
```

## Acknowledgements

Built on the work of the Stockfish, Fairy-Stockfish, and variant-NNUE
communities. Testing infrastructure is based on OpenBench.
