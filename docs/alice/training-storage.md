# Training Storage Preflight

Status: normative operational boundary.

Storage cleanup is outside this project. Project tools do not enumerate, move,
delete, compress, or otherwise modify data on `D:`. They do not attempt to free
space and do not start training below the threshold.

Immediately before generation or training, perform exactly one read-only free
space query for the explicitly supplied target volume:

```text
python tools/alice_training_space_preflight.py --target-volume D:\
```

The threshold is exactly 500 GiB, or `536870912000` available bytes for the
calling account. Exit code `0` means the threshold is met, `3` means it is
below the threshold, and `4` means the single query failed. Every result states
`training_started=false`; a successful preflight authorizes only the next
separately invoked stage.

No storage service, hosted accelerator, or other paid resource may be acquired
through this workflow. Budget, receipts, and explicit approval are separate
prerequisites.
