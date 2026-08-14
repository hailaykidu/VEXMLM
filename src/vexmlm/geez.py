"""Ge'ez script utilities shared across the VEXMLM pipeline.

Central home for script detection, Amharic/Tigrinya normalization, and the
byte-level-BPE decoder used to diagnose the legacy vocabulary defect.
"""

from __future__ import annotations

import re
import unicodedata

# Ethiopic Unicode blocks.
ETHIOPIC_BLOCKS: tuple[tuple[int, int], ...] = (
    (0x1200, 0x137F),  # Ethiopic
    (0x1380, 0x139F),  # Ethiopic Supplement
    (0x2D80, 0x2DDF),  # Ethiopic Extended
    (0xAB00, 0xAB2F),  # Ethiopic Extended-A
    (0x1E7E0, 0x1E7FF),  # Ethiopic Extended-B
)

# Ge'ez punctuation. Kept as separate tokens rather than stripped: sentence
# boundaries (። ፡) carry real signal for MLM and QA span extraction.
GEEZ_PUNCT = "፠፡።፣፤፥፦፧፨"

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[​-\u200F\u202A-\u202E﻿]")


def is_ethiopic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ETHIOPIC_BLOCKS)


def has_ethiopic(text: str) -> bool:
    return any(is_ethiopic_char(c) for c in text)


def ethiopic_ratio(text: str) -> float:
    """Fraction of non-space characters that are Ethiopic."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(is_ethiopic_char(c) for c in chars) / len(chars)


def normalize(text: str, *, form: str = "NFC", collapse_ws: bool = True) -> str:
    """Normalize Ge'ez-script text.

    NFC is the correct default: Ethiopic syllables are precomposed, and NFKC
    would destructively fold distinct characters. Zero-width and bidi control
    characters are stripped -- they are invisible, survive naive dedup, and
    silently fragment subwords during tokenizer training.
    """
    text = _CTRL.sub("", text)
    text = unicodedata.normalize(form, text)
    if collapse_ws:
        text = _WS.sub(" ", text).strip()
    return text


# --- byte-level BPE <-> unicode -------------------------------------------------
# Needed only to decode the legacy vocab.json. See docs/TOKENIZER_DEFECT_REPORT.md.


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 / RoBERTa reversible byte<->unicode mapping."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_UNICODE_TO_BYTE = {v: k for k, v in bytes_to_unicode().items()}


def bytelevel_to_text(token: str) -> str | None:
    """Decode a byte-level-BPE token back to real text.

    Returns None if `token` is not valid byte-level-BPE. Used to recover the
    Ge'ez content of the legacy vocabulary, whose tokens were stored in
    byte-level form and then wrongly added to a SentencePiece tokenizer.
    """
    try:
        return bytes(_UNICODE_TO_BYTE[c] for c in token).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


def looks_bytelevel_encoded(token: str) -> bool:
    """True if `token` is mojibake that decodes to Ethiopic."""
    if has_ethiopic(token):
        return False
    decoded = bytelevel_to_text(token)
    return decoded is not None and has_ethiopic(decoded)
