# Alice-Stockfish engineering dossier

Alice-Stockfish is a dedicated Alice Chess engine built from official
Stockfish. This directory is the public contract for the port: rules take
precedence over legacy implementation details, and every compatibility claim
must be backed by a reproducible receipt.

## Frozen inputs

| Input | Identity |
| --- | --- |
| Official Stockfish chassis | `762dd1da9a5db458180b2c5db6c53dc40ec61e1a` |
| Fairy-Stockfish Alice reference | `4b1940a8d0f60eeb853de7e77af3b39ebf1b6f79` |
| Legacy Alice executable | SHA-256 `B70AFE03EC9A67258CD7B5B848C46FC9E5C83F53B9F2825E9A5946FEEFB59599` |
| Legacy Alice network | SHA-256 `9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9` |
| Alice opening book | SHA-256 `BCD89D9FC3EA81FEB95932EB64D6B6F15AD25CC04CDCC9E0440F097CFFB8CCF6` |

## Contracts

- [Chassis decision](adr-0001-official-stockfish-chassis.md)
- [Rules specification](rules-spec.md)
- [Implementation status](implementation-status.md)
- [Legacy implementation audit](legacy-audit.md)
- [Legacy NNUE compatibility](legacy-nnue-compatibility.md)
- [Local measurement protocol](measurement.md)
- [Local acceptance runner](acceptance-runner.md)
- [Final acceptance gate](final-gate.md)
- [Release evidence](release-evidence.md)
- [Training storage preflight](training-storage.md)
- [OpenBench integration](openbench.md)
- [Native Alice NNUE](native-nnue.md)

The rule fixtures live in `tests/alice/fixtures`. Derived perft values are not
accepted until the independent reference implementation and the optimized
engine agree.
