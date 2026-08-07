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
    "Use NNUE": "true",
    "Alice Evaluation": "Native",
    "Alice Native SHA256": "<sha256>",
    "Alice Native EvalFile": "<absolute-path>"
  }
}
```

Legacy selection uses `Alice Evaluation=Legacy`, `Use NNUE=true`, and the
pinned `EvalFile`. `Zero` requires both `Alice Evaluation=Zero` and
`Use NNUE=false`; it is valid only for structural verification.

The frozen Legacy source must retain the canonical basename
`alice_run2rl_e40_l09.nnue`. Snapshotting content-addresses its parent
directory as `snapshots/networks/<sha256>/` and preserves that basename; a
renamed Legacy source is rejected before any worker starts.

The controller rejects unknown definition fields, duplicate JSON keys,
noncanonical hashes, a policy time-control mismatch, a reused evidence root,
and evidence rooted on `D:`. Each persistent process authenticates declared
UCI options, binary and network bytes, evaluator identity, and the evaluator's
reported SHA-256 before it plays a preflight pair.

Per-pair evidence is admitted only after the response, result-core hash, PGN
hash, result-file hash, terminal classifications, contender scores, root FEN,
and move prefix agree. Files are create-only. A process may be reused across
pairs, but an engine is restarted and reauthenticated after a runtime failure.

The machine schemas are in [`schemas`](../../schemas). Statistical and final
gate semantics are in [measurement.md](measurement.md) and
[final-gate.md](final-gate.md).
