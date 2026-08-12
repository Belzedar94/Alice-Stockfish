# Alice-Stockfish 1.0 release notes

The fixed 700/500/300 comparison panel and the release artifact gates are the
publication authority for this version.

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

The public v1.0 asset is named `Alice_v1.nnue`. It is a filename-only copy of
the exact legacy network tested as `alice_run2rl_e40_l09.nnue`: 47,721,376
bytes, SHA-256
`9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9`.
No training, fine-tuning, conversion, re-export, quantization, or weight
modification was performed. The frozen panel therefore remains bound to the
same network bytes.

All three samples reached 100% LOS. The sealed panel receipt has SHA-256
`73BB64057239AC4A123E53F73CEE3F77C310DA6D3961A46E3B7B2FFCE7704B6D`.

| Time control | Games | Score | Elo |
|---|---:|---|---:|
| 2s + 0.02s | 700 | 618-0-82 (88.29%) | **+350.87 +/- 39.47** |
| 10s + 0.1s | 500 | 404-0-96 (80.80%) | **+249.64 +/- 38.38** |
| 30s + 0.3s | 300 | 231-1-68 (77.17%) | **+211.54 +/- 41.60** |

The release panel must complete with zero discarded pairs, zero abort evidence,
and no time-control or input changes. The sealed receipt must reconcile every
game between referee log, PGN, opening schedule, colors, result, and natural
termination reason, while revalidating all snapshotted input hashes after the
last game. Its fixed sample is a measurement rather than a pass/fail relabeling.

## Features

- Native Alice move generation and legality on two boards, including strict
  terminal handling and reproducible rules tests.
- Alice-specific search with layer-aware threat ordering, arrival-board capture
  staging, and conservative pruning where an Alice-safe SEE is unavailable.
- Exact `LegacyAliceExact` evaluation for the published
  `Alice_v1.nnue` network, with full-refresh and incremental parity.
- Fail-closed network loading: missing, corrupt, incompatible, or ambiguous
  inputs cannot silently select another evaluator.
- Standard UCI, deterministic Alice bench, multi-threading, and large hash
  support.

## Usage

Download the binary matching your CPU (`x86-64-bmi2` for modern Intel and AMD
processors, or `x86-64-avx2` as the portable fallback) together with
`Alice_v1.nnue`. The release binaries expose that filename as their default
`EvalFile`. Place the network in the engine working directory, or select its
full path explicitly before searching:

```text
setoption name Alice Evaluation value Legacy
setoption name Use NNUE value true
setoption name Alice_Frozen_Network value true
setoption name EvalFile value <path>/Alice_v1.nnue
```

The release network has SHA-256
`9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9`.
The engine reports the selected evaluation backend and the SHA-256 of the bytes
it loaded. If the required network cannot be authenticated, evaluation and
search fail closed. Run `bench` after loading the network; every release binary
must report exactly `202963` nodes searched.

## Checksums (SHA-256)

The attached `SHA256SUMS` file is generated after the four binaries are built
from the exact tagged commit and is authoritative for every manually uploaded
asset. The release network entry must be:

`9f9e557015a55c0a6981db64e1f3044dedb91fd8a8c1a6d4f3c45d0eee91fbd9  Alice_v1.nnue`

## Acknowledgements

Built on the work of the Stockfish, Fairy-Stockfish, and variant-NNUE
communities. Testing infrastructure is based on OpenBench.

The network notice and complete filename/provenance mapping are published with
the release as `Alice_v1-NETWORK-NOTICE.txt` and `RELEASE-PROVENANCE.json`.
