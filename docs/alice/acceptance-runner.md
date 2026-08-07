# Local Acceptance Runner

Status: implemented orchestration contract. No strength run is implied.

The controller consumes two strict JSON definitions. The outer run definition
selects one timing control, one mode, the book, and the pinned pair-worker
files. The worker definition selects exactly two engines and their evaluators.
Every path is absolute and every file-bearing input has a lowercase SHA-256.

For a network-backed engine, select the evaluator explicitly:

```json
{
  "evaluator": "Native",
  "network_sha256": "<sha256>",
  "network_path": "<absolute-path>",
  "options": {
    "Threads": "1",
    "Hash": "512",
    "Move Overhead": "10",
    "Use NNUE": "true",
    "Alice Evaluation": "Native",
    "Alice Native SHA256": "<sha256>",
    "Alice Native EvalFile": "<absolute-path>"
  }
}
```

Legacy selection uses `Alice Evaluation=Legacy`, `Use NNUE=true`,
`Alice_Frozen_Network=true`, and the pinned `EvalFile`. `Zero` requires both
`Alice Evaluation=Zero` and `Use NNUE=false`; it is valid only for structural
verification. Each evaluator has an exact option allowlist. Missing, extra, or
different options are rejected, including strength limits and timing
handicaps.

The frozen Legacy source must retain the canonical basename
`alice_run2rl_e40_l09.nnue`. Snapshotting content-addresses its parent
directory as `snapshots/networks/<sha256>/` and preserves that basename; a
renamed Legacy source is rejected before any worker starts.

The controller rejects unknown outer or per-engine definition fields,
duplicate JSON keys, noncanonical hashes, a policy time-control mismatch, a
reused evidence root, evidence rooted on `D:`, non-unique or PGN-unsafe engine
names, and any engine that does not specify exactly `Threads=1` and `Hash=512`.
`Move Overhead=10` is also frozen. Each persistent process authenticates
declared UCI options, binary and network bytes, evaluator identity, and the
evaluator's reported SHA-256 before it plays a preflight pair.

Per-pair evidence is admitted only after the response, result-core hash, PGN
hash, result-file hash, terminal classifications, contender scores, root FEN,
move prefix, and both color assignments agree. Game 1 must name the contender
as White and the reference as Black; game 2 must name the reference as White
and the contender as Black. Each PGN movetext must end in exactly one result
token, and that token must match both the machine result and the `Result`
header. Files are create-only. A process may be reused across pairs, but an
engine is restarted and reauthenticated after a runtime failure.

Every finalized control receipt embeds the create-only acceptance-seal payload
and its canonical SHA-256. The aggregator recomputes that hash and compares all
sealed statistical fields with the final controller result before granting any
strength eligibility. It also requires exactly `openings_jsonl_sha256` and
`status_jsonl_sha256` in the control artifact map; missing, malformed, or extra
artifact identities fail closed. The input inventory retains absolute paths to
the create-only pair-worker and runner-core snapshots as well as their hashes.
Aggregation, including release-time revalidation, reopens both files and
recomputes their SHA-256. Equal-looking hashes without the authenticated runner
bytes cannot enter a strength aggregate.

The machine schemas are in [`schemas`](../../schemas). Statistical and final
gate semantics are in [measurement.md](measurement.md) and
[final-gate.md](final-gate.md).
