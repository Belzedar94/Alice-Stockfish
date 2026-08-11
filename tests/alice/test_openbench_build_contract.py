"""Static contract gates for role-specific public OpenBench builds."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = (ROOT / "src" / "Makefile").read_text(encoding="utf-8")
OPENBENCH = MAKEFILE.split("### Section 12. OpenBench build contract", 1)[1]
GENERATOR = (ROOT / "src" / "data" / "training_data_generator.cpp").read_text(
    encoding="utf-8"
)


class OpenBenchBuildContractTests(unittest.TestCase):
    def test_role_selects_the_default_target(self) -> None:
        self.assertRegex(
            OPENBENCH,
            re.compile(
                r"ifeq \(\$\(OPENBENCH_DATAGEN\),1\)\s*"
                r"\.DEFAULT_GOAL := openbench-datagen\s*else\s*"
                r"\.DEFAULT_GOAL := openbench\s*endif\s*$"
            ),
        )

    def test_datagen_build_emits_the_isolated_generator_at_worker_exe(self) -> None:
        datagen = OPENBENCH.split("openbench-datagen:", 1)[1].split(
            "ifeq ($(OPENBENCH_DATAGEN),1)", 1
        )[0]
        self.assertIn("ifndef EVALFILE", datagen)
        self.assertIn('test -f "$(EVALFILE)"', datagen)
        self.assertIn("data-generator EXE=alice-openbench-playing-unused", datagen)
        self.assertIn('ALICE_DATA_GENERATOR_EXE="$(EXE)"', datagen)
        self.assertIn("COMP=$(OPENBENCH_COMP)", datagen)

    def test_datagen_sources_are_not_in_the_playing_source_list(self) -> None:
        playing_sources = MAKEFILE.split("SRCS =", 1)[1].split("OTHER_SRCS", 1)[0]
        generator_sources = MAKEFILE.split("ALICE_DATA_GENERATOR_SRCS =", 1)[1].split(
            "ALICE_DATA_GENERATOR_HEADERS", 1
        )[0]
        self.assertNotIn("training_data_generator.cpp", playing_sources)
        self.assertIn("training_data_generator.cpp", generator_sources)

    def test_protocol_41_producer_placeholder_remains_mandatory(self) -> None:
        self.assertIn('token == "producer_sha256"', GENERATOR)
        self.assertIn("is_upper_sha256(params.producerSha256)", GENERATOR)
        self.assertIn("producerSha256 && book", GENERATOR)


if __name__ == "__main__":
    unittest.main(verbosity=2)
