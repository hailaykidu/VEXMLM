"""Tests for Ge'ez script utilities, including the mojibake detector.

`test_legacy_vocab_is_detected_as_broken` runs against the REAL legacy artifact
if present -- it is the regression test for the defect that motivated this repo.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vexmlm.geez import (bytelevel_to_text, ethiopic_ratio, has_ethiopic,  # noqa: E402
                         is_ethiopic_char, looks_bytelevel_encoded, normalize)

LEGACY_ADDED = Path(os.path.expanduser("~/VEXMLM_Model/added_tokens.json"))


def test_is_ethiopic_char():
    assert is_ethiopic_char("ሀ")
    assert is_ethiopic_char("፡")
    assert not is_ethiopic_char("a")
    assert not is_ethiopic_char("é")


def test_has_ethiopic_mixed_script():
    assert has_ethiopic("hello ሰላም")
    assert not has_ethiopic("hello world")


def test_ethiopic_ratio():
    assert ethiopic_ratio("ሰላም") == 1.0
    assert ethiopic_ratio("abc") == 0.0
    assert ethiopic_ratio("") == 0.0
    assert ethiopic_ratio("ሰላm") == pytest.approx(2 / 3)


def test_ethiopic_ratio_ignores_whitespace():
    assert ethiopic_ratio("ሰላም   ") == 1.0


def test_normalize_collapses_whitespace():
    assert normalize("ሰላም   ዓለም\n\t ") == "ሰላም ዓለም"


def test_normalize_strips_zero_width():
    assert normalize("ሰላም​ዓለም") == "ሰላምዓለም"
    assert normalize("﻿ሰላም") == "ሰላም"


def test_normalize_is_nfc_not_nfkc():
    # NFKC would fold these; NFC must not.
    text = "ሀ"
    assert normalize(text) == text


def test_bytelevel_roundtrip():
    from vexmlm.geez import bytes_to_unicode
    b2u = bytes_to_unicode()
    encoded = "".join(b2u[b] for b in "ሰላም".encode("utf-8"))
    assert bytelevel_to_text(encoded) == "ሰላም"


def test_bytelevel_decodes_known_legacy_token():
    # 'áĪ¨' is how 'ረ' (U+1228) appears in the legacy vocabulary.
    assert bytelevel_to_text("áĪ¨") == "ረ"


def test_looks_bytelevel_encoded():
    assert looks_bytelevel_encoded("áĪ¨")
    assert not looks_bytelevel_encoded("ረ")      # real Ge'ez is fine
    assert not looks_bytelevel_encoded("hello")  # plain ASCII is not mojibake


def test_bytelevel_returns_none_on_invalid():
    assert bytelevel_to_text("ሰላም") is None


@pytest.mark.skipif(not LEGACY_ADDED.exists(), reason="legacy artifact not present")
def test_legacy_vocab_is_detected_as_broken():
    """Regression test against the real shipped artifact.

    Asserts the documented facts: no token is usable Ge'ez as stored, and the
    overwhelming majority decode to Ge'ez via byte-level BPE.
    """
    tokens = list(json.loads(LEGACY_ADDED.read_text()))
    assert len(tokens) == 30145

    stored_ethiopic = sum(has_ethiopic(t) for t in tokens)
    assert stored_ethiopic == 0, "expected zero directly-usable Ge'ez tokens"

    mojibake = sum(looks_bytelevel_encoded(t) for t in tokens)
    assert mojibake / len(tokens) > 0.98, (
        f"expected >98% byte-level-encoded, got {mojibake / len(tokens):.1%}")
