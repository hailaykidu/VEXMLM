"""Intrinsic tokenizer metrics (paper Sec. 5.1).

    fertility(L)   = tokens / word, over corpus L. Lower is better.
    compression(L) = characters / token. Higher is better.
    parity(A,B)    = fertility(A) / fertility(B) on PARALLEL text.
                     1.0 means the tokenizer treats both languages equally
                     (Petrov et al., 2023). Requires sentence-aligned corpora;
                     computing it on non-parallel text measures content
                     differences, not tokenizer fairness, so this module records
                     whether the input was actually parallel.
    oov_accuracy   = fraction of held-out words the tokenizer encodes WITHOUT
                     producing <unk> and whose decoded form round-trips exactly.
                     Byte-fallback tokenizers rarely emit <unk>, so round-trip
                     fidelity is the meaningful signal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class IntrinsicMetrics:
    language: str
    tokenizer: str
    sentences: int
    words: int
    tokens: int
    chars: int
    fertility: float
    compression: float
    continuation_rate: float
    unk_rate: float
    new_token_share: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def measure(tokenizer, lines: list[str], *, language: str = "",
            name: str = "", base_vocab_size: int | None = None) -> IntrinsicMetrics:
    """Compute fertility, compression, and related rates over a corpus."""
    n_tokens = n_words = n_chars = n_unk = n_cont = n_new = 0
    used = 0
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        used += 1
        ids = tokenizer.encode(line, add_special_tokens=False)
        pieces = tokenizer.convert_ids_to_tokens(ids)
        n_tokens += len(ids)
        n_words += len(line.split())
        n_chars += len(line)
        if unk_id is not None:
            n_unk += sum(1 for i in ids if i == unk_id)
        n_cont += sum(1 for p in pieces if not p.startswith("▁"))
        if base_vocab_size is not None:
            n_new += sum(1 for i in ids if i >= base_vocab_size)

    if n_tokens == 0:
        raise ValueError("corpus produced no tokens")

    return IntrinsicMetrics(
        language=language, tokenizer=name, sentences=used, words=n_words,
        tokens=n_tokens, chars=n_chars,
        fertility=round(n_tokens / n_words, 4) if n_words else float("nan"),
        compression=round(n_chars / n_tokens, 4),
        continuation_rate=round(n_cont / n_tokens, 4),
        unk_rate=round(n_unk / n_tokens, 6),
        new_token_share=(round(n_new / n_tokens, 6)
                         if base_vocab_size is not None else None),
    )


def parity(fert_a: float, fert_b: float, *, parallel: bool) -> dict:
    """Tokenizer parity between two languages.

    `parallel` is recorded, not assumed: parity on non-parallel corpora is not
    a valid fairness measure and must not be reported as one.
    """
    return {
        "value": round(fert_a / fert_b, 4) if fert_b else None,
        "fertility_a": fert_a,
        "fertility_b": fert_b,
        "parallel_corpus": parallel,
        "valid": parallel,
        "interpretation": (
            "1.0 = equal treatment; >1.0 = language A is fragmented more"
            if parallel else
            "NOT VALID as parity: corpora are not sentence-aligned, so the "
            "ratio reflects content differences as well as tokenization"
        ),
    }


def oov_accuracy(tokenizer, words: list[str]) -> dict:
    """Share of held-out words encoded without <unk> and decoded losslessly.

    Two numbers are reported because they answer different questions:
      no_unk_rate   -- did the tokenizer avoid the unknown token?
      roundtrip_rate -- did it preserve the word exactly? (the stricter test)
    """
    unk_id = getattr(tokenizer, "unk_token_id", None)
    total = no_unk = roundtrip = 0
    failures: list[str] = []

    for word in words:
        word = word.strip()
        if not word:
            continue
        total += 1
        ids = tokenizer.encode(word, add_special_tokens=False)
        clean = unk_id is None or unk_id not in ids
        no_unk += clean
        decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
        if decoded == word:
            roundtrip += 1
        elif len(failures) < 20:
            failures.append(f"{word!r} -> {decoded!r}")

    if total == 0:
        raise ValueError("no words supplied")
    return {
        "words": total,
        "no_unk_rate": round(no_unk / total, 4),
        "roundtrip_rate": round(roundtrip / total, 4),
        "oov_accuracy": round(roundtrip / total, 4),  # the reported figure
        "example_failures": failures,
    }
