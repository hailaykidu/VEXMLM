"""Tests for embedding initialization.

`test_global_mean_implements_paper_formula` is the specification test: it checks
the official method against the paper's formula e_t = (1/|V_s|) * sum e_s
directly. The remaining tests cover the Table 5 ablation arms and determinism.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vocabulary_expansion"))
sys.path.insert(0, str(ROOT / "src"))

from initialization import (  # noqa: E402
    constituent_mean_init, global_mean_init, initialize, mixed_init,
    predict_distance, random_init)

HIDDEN = 768
N_OLD = 1000
N_NEW = 500


@pytest.fixture
def emb() -> torch.Tensor:
    """Embedding matrix whose 'pretrained' block mimics XLM-R's scale."""
    g = torch.Generator().manual_seed(0)
    e = torch.zeros(N_OLD + N_NEW, HIDDEN)
    e[:N_OLD] = torch.randn(N_OLD, HIDDEN, generator=g) * 0.212
    return e


def test_global_mean_implements_paper_formula(emb):
    """OFFICIAL METHOD, paper Sec. 3.3: e_t = (1/|V_s|) * sum_{s in V_s} e_s.

    Every new row must equal the centroid of the source embedding space exactly.
    """
    expected = emb[:N_OLD].mean(0)
    global_mean_init(emb, N_OLD)
    for row in (emb[N_OLD], emb[N_OLD + N_NEW // 2], emb[-1]):
        assert torch.allclose(row, expected, atol=1e-6)


def test_global_mean_makes_all_rows_identical(emb):
    stats = global_mean_init(emb, N_OLD)
    new = emb[N_OLD:]
    assert torch.allclose(new[0], new[-1], atol=1e-6)
    assert stats.dist_to_global_mean_mean == pytest.approx(0.0, abs=1e-4)
    # InitStats rounds to 4dp, so compare at that resolution.
    assert stats.new_perdim_std == pytest.approx(new.std().item(), abs=1e-4)


def test_global_mean_noise_separates_rows(emb):
    global_mean_init(emb, N_OLD, noise_std=1e-3)
    new = emb[N_OLD:]
    assert not torch.allclose(new[0], new[-1], atol=1e-6)


def test_random_matches_base_scale_by_default(emb):
    base_std = emb[:N_OLD].std().item()
    stats = random_init(emb, N_OLD)
    # Should match pretrained scale, NOT N(0,1).
    assert stats.new_perdim_std == pytest.approx(base_std, rel=0.05)
    assert stats.new_perdim_std < 0.5


def test_mixed_signature_is_identifiable(emb):
    """Reference strategy: (N(0,1)+mean)/2 -> per-dim std ~0.5, dist ~0.5*sqrt(d).

    Needed so verify_vocab.py can recognize externally-supplied checkpoints
    built this way and distinguish them from the paper's method.
    """
    stats = mixed_init(emb, N_OLD, legacy_unit_normal=True)
    assert stats.new_perdim_std == pytest.approx(0.5, rel=0.05)
    assert stats.dist_to_global_mean_mean == pytest.approx(0.5 * HIDDEN ** 0.5, rel=0.02)


def test_strategies_are_mutually_distinguishable(emb):
    """Each strategy must land far from the others' predictions."""
    d = {}
    for strat in ("global_mean", "random", "mixed"):
        e = emb.clone()
        kwargs = {"match_base": False} if strat == "random" else {}
        d[strat] = initialize(strat, e, N_OLD, **kwargs).dist_to_global_mean_mean
    assert d["global_mean"] < 1.0
    assert d["mixed"] == pytest.approx(0.5 * HIDDEN ** 0.5, rel=0.05)
    assert d["random"] == pytest.approx(1.0 * HIDDEN ** 0.5, rel=0.05)
    assert d["global_mean"] < d["mixed"] < d["random"]


def test_determinism_under_seed(emb):
    a, b = emb.clone(), emb.clone()
    mixed_init(a, N_OLD, seed=42)
    mixed_init(b, N_OLD, seed=42)
    assert torch.equal(a, b)


def test_different_seeds_differ(emb):
    a, b = emb.clone(), emb.clone()
    mixed_init(a, N_OLD, seed=42)
    mixed_init(b, N_OLD, seed=43)
    assert not torch.equal(a, b)


def test_old_rows_never_modified(emb):
    original = emb[:N_OLD].clone()
    for strat in ("global_mean", "random", "mixed"):
        e = emb.clone()
        initialize(strat, e, N_OLD)
        assert torch.equal(e[:N_OLD], original), f"{strat} corrupted pretrained rows"


def test_predict_distance_closed_forms():
    assert predict_distance("global_mean", 768, 0.212) == 0.0
    assert predict_distance("mixed", 768, 0.212) == pytest.approx(13.8564, abs=0.01)
    assert predict_distance("random", 768, 0.212) == pytest.approx(27.7128, abs=0.01)


class _FakeTokenizer:
    """Splits a token into its characters, mapping each to a stable id."""
    unk_token_id = 999999

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % N_OLD for c in text if not c.isspace()]


def test_constituent_mean_uses_subword_means(emb):
    tokens = ["▁ሰላም", "▁ዓለም", "▁ትግርኛ"]
    e = emb.clone()
    stats = constituent_mean_init(e, N_OLD, tokens, _FakeTokenizer())
    assert stats.fallback_count == 0
    # Each row must equal the mean of its own constituents.
    ids = [ord(c) % N_OLD for c in "ሰላም"]
    assert torch.allclose(e[N_OLD], emb[:N_OLD][ids].mean(0), atol=1e-5)


def test_constituent_mean_falls_back_on_empty(emb):
    class Empty(_FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return []

    e = emb.clone()
    stats = constituent_mean_init(e, N_OLD, ["▁x", "▁y"], Empty())
    assert stats.fallback_count == 2
    assert torch.allclose(e[N_OLD], emb[:N_OLD].mean(0), atol=1e-5)


def test_unknown_strategy_rejected(emb):
    with pytest.raises(ValueError, match="unknown strategy"):
        initialize("nonsense", emb, N_OLD)
