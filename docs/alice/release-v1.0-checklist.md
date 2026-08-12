# Alice-Stockfish 1.0 release checklist

This checklist controls the first stable Alice-Stockfish release. A green
development build is necessary but does not authorize publication. The release
must remain a draft until the fixed comparison panel, exact-source validation,
artifact checksums, and downloaded-draft verification are complete.

## Frozen scope and inputs

- Source baseline: Alice-Stockfish `4a88df6f03ffd9a721b54f04cdb12a8e847929c5`.
- Source tree: `4288edda2f36e4de28507f1f8d046b27b6cf3af5`.
- Canonical Alice bench after loading the release network: `202963` nodes.
- Evaluation mode: `LegacyAliceExact` only.
- Frozen-panel filename: `alice_run2rl_e40_l09.nnue`.
- Public v1.0 asset filename: `Alice_v1.nnue`.
- Network identity under either filename: 47,721,376 bytes, SHA-256
  `9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9`.
- The public asset must be produced by a binary copy only; no conversion,
  re-export, metadata insertion, or other byte transformation is permitted.
- Opening book: `alice.epd`, 38,348 positions, SHA-256
  `BCD89D9FC3EA81FEB95932EB64D6B6F15AD25CC04CDCC9E0440F097CFFB8CCF6`.
- Reference engine: frozen Fairy-Stockfish binary, SHA-256
  `B70AFE03EC9A67258CD7B5B848C46FC9E5C83F53B9F2825E9A5946FEEFB59599`.
- Reference source: Fairy-Stockfish
  `4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79` (`040925`).
- Referee: official Alice `uci_pair_runner.py` at OpenBench commit
  `4da5cb6c4ff502b60efb6e8c6c9bd7ef0c37fc69`, SHA-256
  `73081EC57EAF964009CF7A877428888910A7AF1C091D8F0F7ECFB47020161CAB`.
- Referee interpreter: Python 3.12.0 executable, SHA-256
  `42AC541168E97DEDB9AABD8BE335539FC41C682E414B9E8D137B164FB68683B0`.
- Opening permutation: `order=random`, `start=1`, shared seed `20260811`.
- The rejected experimental NNUE V2 checkpoint and its inference branch are
  excluded from the release source, assets, strength claims, and tag.

Any playing-code change after the fixed panel invalidates the panel and
requires a new measurement from the changed source. Release-only documentation
and packaging must preserve the bench signature.

## Fixed local comparison panel

Alice-Stockfish and the frozen Fairy-Stockfish reference use the same network,
book, one search thread, 512 MiB hash, and 10 ms move overhead. Adjudication is
disabled, openings are color-swapped pairs, and incomplete pairs do not enter
the result. All three controls use the same frozen opening permutation. They run
in parallel with four games per control, consuming exactly 24 engine threads.

| Preset | Time control | Games |
|---|---:|---:|
| VSTC | 2s + 0.02s | 700 |
| STC | 10s + 0.1s | 500 |
| LTC | 30s + 0.3s | 300 |

Before the panel, record the official T24 worker's PID, complete launch command,
N1/T24 settings, and established OpenBench connection. Stop only that worker.
For this release transaction, keep it stopped through the panel, the authenticated
native-data producer gates, and the one official 50,000,000-position DATAGEN
submission. Restore the exact command and settings after all three operations
either finish or fail closed. Confirm a fresh connection to the official
OpenBench service before closing the transaction.

The final receipt must include W/D/L, Elo and uncertainty, likelihood of
superiority, pair counts, all input hashes, complete commands, start/end times,
and zero time losses, aborts, or discarded pairs. Before any game starts, bind
the clean candidate source and tree to the compiler, binary, three canonical
benches, mandatory zero-skip tests, and explicit NativeV2 exclusion in a hashed
build receipt. Execute only create-exclusive read-only snapshots of the two
binaries, two network copies, book, referee, runner, and build receipt. After
each control, reconcile every logged result against its PGN, opening ordinal,
color assignment, and allowlisted natural termination; then rehash every input.

## Exact-source quality gates

1. Confirm the candidate commit and tree are clean and contain no NNUE V2
   inference or rejected checkpoint assets.
2. Build Windows and Linux AVX2 and BMI2 release binaries from isolated clean
   checkouts with recorded compiler identities.
3. Authenticate the network before every test process. Run the canonical bench
   three times per release binary and require `202963` nodes every time.
4. Run the Alice rules, move-generation, perft, terminal, UCI, deterministic
   replay, Legacy NNUE full-refresh/incremental parity, and fail-closed load
   suites with zero failures or skips in mandatory gates.
5. Exercise missing, corrupt, wrong-schema, wrong-architecture, ambiguous, and
   wrong-checksum network inputs; none may silently fall back to another
   evaluator.
6. Run the fixed local comparison panel without changing any frozen input.
7. Hash every final binary and the network, then write `SHA256SUMS` last.
8. Verify the historical source and `Alice_v1.nnue` have the same size and
   SHA-256 and pass a direct binary comparison.
9. Require all four release binaries to load `Alice_v1.nnue` in frozen-baseline
   mode, report the frozen digest, and return `202963` in three fresh canonical
   bench processes.
10. Rehash the downloaded draft network asset before publication and require
    the same identity. This packaging validation does not rerun or relabel the
    fixed 700/500/300 panel.

## Release assets

- Windows x86-64 AVX2 and BMI2 executables.
- Linux x86-64 AVX2 and BMI2 executables.
- `Alice_v1.nnue` (the only network copy in the release).
- `Alice_v1-NETWORK-NOTICE.txt` and `RELEASE-PROVENANCE.json`.
- Source archive, GPL, AUTHORS, README, release notes, and `SHA256SUMS`.

## Publication transaction

1. Tag only the exact reviewed candidate commit; never move an existing tag.
2. Create a draft GitHub release and upload the complete frozen asset set.
3. Download the draft into a new empty directory, require the exact expected
   filenames, and verify every byte against `SHA256SUMS`.
4. Render the GitHub release body from `release-draft-v1.0.md` and attach the
   generated `SHA256SUMS` from the sealed build receipts. State the Legacy-only
   scope and the excluded NNUE V2 pilot.
5. Publish manually only after a final remote tag/commit check and human review
   of the downloaded draft. Any mismatch deletes the draft and restarts the
   candidate process; it never moves the tag.
