"""Paper artifacts must equal what the stored result files generate.

Regenerates every macro and table body with scripts/make_paper_artifacts.py into a
temporary directory and compares it with paper/generated/. Fails if a result file
changed without regenerating the paper, or if a generated file was edited by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_paper_artifacts as mpa  # noqa: E402

GENERATED = ROOT / "paper" / "generated"


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    out = tmp_path_factory.mktemp("generated")
    mpa.build(ROOT / "results", out)
    return out


@pytest.mark.parametrize("name", sorted(p.name for p in GENERATED.glob("*.tex")))
def test_generated_file_is_current(regenerated, name):
    assert (regenerated / name).read_text() == (GENERATED / name).read_text()


def test_no_generated_file_missing(regenerated):
    assert {p.name for p in regenerated.glob("*.tex")} == {p.name for p in GENERATED.glob("*.tex")}
