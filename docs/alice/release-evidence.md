# Release Evidence Contract

Status: normative packaging gate. This contract does not publish a release.

A candidate manifest must bind:

- the full source commit;
- one trained AliceNative-v1 network and its SHA-256;
- the dataset, checkpoint, export, and G1-G8 qualification receipt;
- clean exact-LOS and fixed-final aggregate receipts;
- clean 200-pair official OpenBench shadow receipts at VSTC, STC, and LTC,
  each tracking `800/4` and `40/8/10` virtual endings;
- Windows BMI2, Windows AVX2, Linux BMI2, and Linux AVX2 binaries;
- three identical bench signatures for every binary; and
- missing, corrupt, and incompatible network probes for every binary.

Each negative load probe must exit nonzero, publish no search result, and show
that no alternate evaluator ran. Checksums are verified against the bytes on
disk; the native network must have the exact AliceNative-v1 wire size.
The qualification receipt must identify a trained run, positive dataset and
comparison sample sizes, a positive nonzero-parameter count, and zero
checkpoint/file, file/engine, and incremental/full mismatches.
Both local batteries must bind the same pinned inputs, select the native
evaluator, and identify the exact candidate network.
The OpenBench shadow receipt must repeat the candidate source commit and
network SHA-256. Every preset must also name a release binary role and the
matching binary SHA-256 from the candidate manifest; a clean audit from any
other source, network, or binary cannot authorize the candidate.

Audit a manifest with:

```text
python tools/alice_release_evidence.py \
  --manifest <absolute-candidate.json> \
  --output <new-receipt.json>
```

The command is read-only with respect to candidate artifacts. It creates one
receipt and exits with code `3` when blocked. Its receipt always records
`publication_performed=false`. Uploading, tagging, or announcing artifacts is
a separate explicitly authorized operation after the receipt says
`strength_release_authorized=true`.

An all-zero or sentinel native wire is suitable only for structural tests. It
cannot satisfy the trained-network provenance and qualification fields and must
never be presented as a strength release.
