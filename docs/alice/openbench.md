# Alice on OpenBench

Status: production integration and operations contract. Alice is registered on
the official service and its admitted workloads use the identities and custody
rules below.

This document defines how Alice workloads may enter the shared OpenBench
service. It does not change the local exact-LOS contract in
[measurement.md](measurement.md).

## Current implementation boundary

The engine-side build and bench contract is implemented and locally verified.
An OpenBench build request may provide `EXE` and `EVALFILE` to a bare `make`
invocation, and a freshly started resulting executable loads that exact network
before a bare `bench`. The canonical bench signature is `202963` nodes on the
versioned eight-position Alice corpus.

Alice is registered and scheduled on the official service with the dedicated
engine entry, `ALICE` book routing, and Alice-capable paired runner. Only results
whose workload and receipts satisfy this document are official; a local build
or local OpenBench run remains integration evidence only.

## 1. One official service

The only official service is:

`https://belzedar.duckdns.org`

Results from a local OpenBench instance are disposable integration evidence,
not official Alice strength or data-generation evidence. Do not publish local
queues, databases, chunks, or media as production results.

The existing T24 production worker remains under normal official scheduling.
Alice onboarding must not manually stop, restart, resize, rethread, or retarget
that worker. Because Alice runs at lower priority, the official scheduler may
assign eligible Alice work to T24 normally when queue policy selects it.

## 2. Admission identity

Alice is a distinct workload family and must be identifiable before scheduling:

- The book token is exactly `ALICE`.
- `ALICE` routes to an Alice-capable paired UCI runner; it must never fall
  through to a conventional-chess runner.
- Alice-Stockfish has its own engine entry, pinned source revision, build path,
  bench signature, and measured NPS.
- The engine admission field is exactly `cpuflags: []`.
- An empty CPU-flag list is an admission policy, not a portability claim. Each
  supported architecture still requires a successful build, bench, UCI, and
  self-play check.
- Every workload pins the engine revision, network hash, book hash, runner
  revision, option set, and timing preset in its receipt.

If DATAGEN is requested without a book, its explicit engine-family fallback
must also resolve to `ALICE`. An unknown or missing token is a hard admission
failure, never permission to use another ruleset.

## 3. Queue priority and scheduling

Alice workloads enter below the priority of all already-running production
work. Raising their priority requires an explicit operations decision recorded
with the test. Alice onboarding may not preempt current work merely to shorten
validation time.

The scheduler must expose Alice identity in the workload, assignment, and result
records. Workers that have not passed Alice admission checks must reject the
assignment cleanly rather than attempting conventional play.

## 4. OpenBench-only adjudication

The following evaluation adjudication is permitted only on official OpenBench
Alice workloads:

- win adjudication: score `800` for `4` consecutive adjudication observations;
- draw adjudication: beginning at move `40`, score within `10` for `8`
  consecutive adjudication observations.

Record these as `800/4` and `40/8/10` in the workload receipt. The values must be
applied symmetrically by the paired runner and must be visible in the generated
game metadata.

Do not copy these thresholds into the local VSTC/STC/LTC battery. The local
battery has no external score adjudication and uses the stopping rule in
[measurement.md](measurement.md).

## 5. Mandatory shadow audit

Every admitted Alice OpenBench preset requires a shadow audit of exactly 200
complete color-swapped pairs: 400 games with recorded results. The audit uses
the same engine, network, book, options, timing, runner, and OpenBench
adjudication as the preset it shadows.

The shadow sample validates the harness; it is not a promotion test and cannot
be counted toward a local LOS gate. It runs at Alice's lower queue priority.
The official scheduler may assign it to any eligible worker, including T24,
without manual worker intervention.

The aggregate shadow receipt repeats the candidate source commit and network
SHA-256. Each preset records the release binary role and exact binary SHA-256
used for that audit and references its canonical configuration artifact.
Release evidence reopens that artifact, recomputes its SHA-256, and requires
the official service, `ALICE` book token and frozen book hash, one shared
runner hash, exact candidate identities, `Threads=1`, `Hash=512`,
`Move Overhead=10`, the preset timing, color-swapped pairing, `cpuflags=[]`,
and both adjudication rules. The three configuration identities must be
distinct. A source, network, binary, book, runner, option, timing, pairing, or
worker-policy mismatch blocks the shadow gate.

For each 200-pair audit, preserve and report:

- all 200 opening identifiers and both color assignments;
- pair attempts, complete pairs, and discarded pairs;
- W/L/D and terminal-reason counts, including both adjudication classes;
- illegal moves, time losses, process exits, protocol failures, and retries;
- replay of the start position and move list under the Alice rules;
- confirmation that only complete pairs entered the result set; and
- the exact hashes and configuration named in section 2.

Unexplained aborts must be zero. Any FEN, legal-move, transfer-board, color-swap,
or result-accounting mismatch fails the shadow audit and suspends interpretation
of that preset. The audit must be repeated from a clean sample after correction.

The Alice runner suppresses the anomalous game from its `Finished game` result
stream, writes its machine failure class to PGN, and exits nonzero after drain.
Its color mate therefore cannot form a reported pentanomial pair. A shadow
inversion has the same invalidating behavior. Any clean pairs already produced
belong only to the failed audit identifier; they are never combined with the
replacement audit.

## 6. Admission sequence

Alice production admission proceeds in this order:

1. Validate the `ALICE` token and engine configuration without scheduling work.
2. Build on each eligible worker architecture and verify the pinned binary hash.
3. Run a deterministic bench and record the expected signature.
4. Complete UCI `uci`/`isready` startup and an Alice position round-trip.
5. Complete a short paired self-play smoke test with clean PGN replay.
6. Complete the 200-pair shadow audit for each preset.
7. Admit lower-priority official work only after all earlier evidence is clean.

A configuration or runner change returns the affected presets to the relevant
earlier admission step. A network-only change requires load verification and a
new shadow audit, even when the executable is unchanged.

## 7. DATAGEN custody

Official OpenBench may schedule and retain opaque Alice DATAGEN chunks, but the
engine-side pipeline owns the record format, auditor, merge logic, and network
compatibility checks. Each chunk must retain workload identity, engine and
network hashes, seed, timing, options, game counts, abort counts, and format
version.

Pilot chunk duration should remain operationally bounded before scaling the
queue. A chunk is publishable only after engine-side parsing, state round-trip,
legality, duplicate, and provenance checks succeed. The native feature and
serialization gates are specified in [native-nnue.md](native-nnue.md).

## 8. Production receipt

An official Alice result must link the OpenBench test identifier to:

- the official service URL;
- priority and worker identity;
- engine, network, book, and runner hashes;
- timing and `800/4`, `40/8/10` adjudication settings;
- the applicable 200-pair shadow-audit receipt;
- build, bench, startup, and PGN-replay evidence; and
- confirmation that T24 was not manually stopped, restarted, rethreaded, or
  retargeted for the Alice workload.

Without that receipt, the result is integration output rather than official
Alice evidence.
