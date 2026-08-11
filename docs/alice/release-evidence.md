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
- three distinct, authenticated `202963`-node canonical bench executions for
  every binary; and
- missing, corrupt, and incompatible network probes for every binary.

Each negative load probe must exit nonzero, publish no search result, and show
that no alternate evaluator ran. Every case references the exact input
descriptor, command receipt, output receipt, and mutated input where bytes
exist by absolute path and SHA-256. The auditor reopens those bytes, recomputes
every digest, and checks the cross-references among all three receipts. Missing
input uses `absent-path`, names a path that must still be absent at audit time,
and has no input artifact. Corrupt input is the source network with byte zero
XORed by `0x01`; incompatible input is the source network with byte four XORed
by `0x01`. The auditor streams both files and accepts no other byte difference.
The three descriptor, command, and output identities must be distinct, and the
two mutated inputs must differ from both each other and the candidate network.
The native network must have the exact AliceNative-v1 wire size.
The qualification receipt must identify a trained run and reference the exact
dataset manifest, checkpoint, export receipt, and eight distinct gate reports
by absolute path and SHA-256. The auditor reopens every artifact and recomputes
its digest. Dataset counts and split identity come from the dataset manifest;
checkpoint and candidate-network identities, wire size, deterministic
re-export identity, element count, and zero serialization mismatches come from
the export receipt. Every G1-G8 report must bind all three artifact identities,
the candidate network, and the same training run, with a positive sample count
and an exact zero mismatch count. G1/G2 samples are checked against the dataset
count, and G6 must cover that same complete parity corpus. G4/G5 must cover the
frozen AliceNative-v1 architecture's complete
`170222600` scalar elements. The auditor independently streams the candidate
wire tensors, excludes metadata and architecture hashes,
and requires the receipt's positive nonzero-byte count to match the bytes on
disk; an all-zero or sentinel parameter payload cannot qualify.
Both local batteries must bind the same pinned inputs, select the native
evaluator, and identify the exact candidate network. Their normalized worker
configuration identity includes every UCI option and every worker-level
setting while replacing only the intentionally different time control and
snapshot paths with stable content identities. The reference side must be the
frozen historical executable
`b70afe03ec9a67258cd7b5b848c46fc9e5c83f53b9f2825e9a5946feefb59599`
using the frozen legacy network
`9f9e557015a55c0a6981db64e1f3044dedb91fd8a8c1a6d4f3c45d0eee91fbd9`;
the structural zero evaluator cannot satisfy either strength gate.
The contender binary SHA-256 in both local batteries must equal one of the
four candidate release artifacts; a result produced by a development or
otherwise unlisted executable cannot authorize those artifacts.
Every embedded control receipt must contain exactly the canonical SHA-256 of
its `openings.jsonl` and `status.jsonl`; an empty, partial, malformed, or
extended artifact map cannot enter either strength aggregate, and the two
identities must be distinct. Each control inventory also carries absolute paths
and SHA-256 values for its create-only pair-worker and runner-core snapshots.
The aggregator reopens and hashes those files during battery aggregation and
release audit; replacement code cannot be hidden behind self-declared runner
hashes.
The OpenBench shadow receipt must repeat the candidate source commit and
network SHA-256. Every preset must also name a release binary role and the
matching binary SHA-256 from the candidate manifest. It references a canonical
configuration artifact whose bytes and SHA-256 are recomputed by the auditor.
That artifact fixes the official service, preset, candidate identities,
`ALICE` book and frozen book hash, runner hash, `Threads=1`, `Hash=512`,
`Move Overhead=10`, exact timing, color-swapped pairing, `cpuflags=[]`, and the
two OpenBench adjudication rules. The three preset configurations must be
distinct and must share one runner identity. A clean audit from any other
source, network, binary, book, runner, option set, timing, pairing, or worker
policy cannot authorize the candidate.

The four release artifacts must have four distinct SHA-256 identities. The
auditor reads each executable header and requires x86-64 PE for Windows roles
or x86-64 ELF for Linux roles. It also verifies the embedded Stockfish
compilation architecture (`x86-64-bmi2` or `x86-64-avx2`) and rejects a binary
that embeds the incompatible release architecture. Every binary must also
embed the manifest's complete 40-character source commit, as emitted by
Stockfish's build identity; a stale-revision artifact is rejected even when
its platform and architecture are otherwise correct.

Every triple-bench receipt references three distinct canonical command
artifacts and three distinct raw UTF-8 stdout artifacts by absolute path and
SHA-256. The auditor reopens every artifact, recomputes its digest, requires
each command to bind the exact executable and network paths and hashes, and
parses exactly one `Nodes searched : 202963` result plus the selected network
SHA-256 from each stdout. A declared signature without those authenticated
execution artifacts cannot satisfy the release gate.

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
