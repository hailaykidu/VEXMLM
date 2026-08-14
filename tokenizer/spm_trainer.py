"""SentencePiece training for Amharic / Tigrinya, shared by the entry scripts.

This is the component that was MISSING from the original work: no SentencePiece
model was ever trained (all shipped `sentencepiece.bpe.model` files are stock
XLM-R -- see docs/TOKENIZER_DEFECT_REPORT.md).

Critically, tokens produced here are raw Unicode, matching XLM-R's SentencePiece
convention. That is the defect this module exists to avoid.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vexmlm.geez import ethiopic_ratio, has_ethiopic, normalize  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class CorpusStats:
    input_files: list[str]
    lines_read: int = 0
    lines_kept: int = 0
    duplicates_dropped: int = 0
    too_short_dropped: int = 0
    low_ethiopic_dropped: int = 0
    chars_kept: int = 0
    sha256: str = ""

    @property
    def retention(self) -> float:
        return self.lines_kept / self.lines_read if self.lines_read else 0.0


@dataclass
class TokenizerSpec:
    """Everything needed to reproduce a tokenizer bit-for-bit."""
    language: str
    vocab_size: int
    model_type: str = "unigram"          # XLM-R uses unigram, not BPE
    character_coverage: float = 0.9995
    min_sentence_length: int = 10
    min_ethiopic_ratio: float = 0.5
    normalization: str = "NFC"
    deduplicate: bool = True
    seed: int = 42
    input_sentence_size: int = 0          # 0 = use everything
    extra: dict = field(default_factory=dict)


def prepare_corpus(
    input_files: list[str | Path],
    out_path: Path,
    spec: TokenizerSpec,
) -> CorpusStats:
    """Normalize, filter, and deduplicate a raw corpus into one training file.

    Filtering rationale:
      * short lines add noise and inflate the unigram seed vocabulary
      * low-Ethiopic lines are usually boilerplate, URLs, or wrong-language text
      * exact dedup prevents web-scrape duplicates from skewing token frequencies
    """
    stats = CorpusStats(input_files=[str(f) for f in input_files])
    seen: set[int] = set()
    hasher = hashlib.sha256()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for path in input_files:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    stats.lines_read += 1
                    text = normalize(line, form=spec.normalization)
                    if len(text) < spec.min_sentence_length:
                        stats.too_short_dropped += 1
                        continue
                    if spec.min_ethiopic_ratio > 0 and ethiopic_ratio(text) < spec.min_ethiopic_ratio:
                        stats.low_ethiopic_dropped += 1
                        continue
                    if spec.deduplicate:
                        h = hash(text)
                        if h in seen:
                            stats.duplicates_dropped += 1
                            continue
                        seen.add(h)
                    out.write(text + "\n")
                    hasher.update(text.encode("utf-8"))
                    stats.lines_kept += 1
                    stats.chars_kept += len(text)

    stats.sha256 = hasher.hexdigest()
    log.info("corpus: %d read -> %d kept (%.1f%%)",
             stats.lines_read, stats.lines_kept, 100 * stats.retention)
    return stats


def train_sentencepiece(
    corpus_path: Path,
    model_prefix: Path,
    spec: TokenizerSpec,
) -> Path:
    """Train a SentencePiece model matching XLM-R's conventions."""
    import sentencepiece as spm

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    # `random_seed` is not a TrainerSpec field in every sentencepiece build; it is
    # set process-wide instead. Do that first so training is reproducible either way.
    try:
        spm.set_random_generator_seed(spec.seed)
    except AttributeError:  # older/newer builds without the helper
        log.debug("sentencepiece.set_random_generator_seed unavailable")

    # Special-token IDs match XLM-R exactly so the vocabularies can be merged.
    try:
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(model_prefix),
            vocab_size=spec.vocab_size,
            model_type=spec.model_type,
            character_coverage=spec.character_coverage,
            input_sentence_size=spec.input_sentence_size,
            shuffle_input_sentence=bool(spec.input_sentence_size),
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
            normalization_rule_name="identity",  # already normalized; don't double-apply
            byte_fallback=False,
            train_extremely_large_corpus=False,
            num_threads=os.cpu_count() or 8,
        )
    except RuntimeError as exc:
        # SentencePiece caps vocab_size at the number of distinct pieces it can
        # find. Translate its internal assertion into actionable advice.
        if "Vocabulary size too high" in str(exc):
            raise SystemExit(
                f"Corpus too small for vocab_size={spec.vocab_size}.\n"
                f"  {exc}\n"
                f"Either supply more text, or lower --vocab-size. The paper's "
                f"targets (32k Amharic / 50k Tigrinya) need a corpus on the "
                f"order of hundreds of thousands of sentences."
            ) from exc
        raise
    return model_prefix.with_suffix(".model")


def summarize_vocabulary(model_path: Path) -> dict:
    """Vocabulary statistics for TOKENIZER_REPRODUCTION.md."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    ethiopic = [p for p in pieces if has_ethiopic(p)]
    lengths = [len(p.lstrip("▁")) for p in ethiopic] or [0]
    return {
        "model": str(model_path),
        "piece_size": sp.get_piece_size(),
        "ethiopic_pieces": len(ethiopic),
        "ethiopic_fraction": round(len(ethiopic) / len(pieces), 4),
        "mean_ethiopic_piece_len": round(sum(lengths) / len(lengths), 3),
        "max_ethiopic_piece_len": max(lengths),
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }


def write_manifest(path: Path, spec: TokenizerSpec, stats: CorpusStats, vocab: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"spec": asdict(spec), "corpus": asdict(stats), "vocabulary": vocab},
        ensure_ascii=False, indent=2))
    log.info("wrote manifest %s", path)
