"""Embedding initialization for newly added vocabulary items.

OFFICIAL METHOD (paper, Section 3.3) -- `global_mean`:

    e_t = (1 / |V_s|) * sum_{s in V_s} e_s

Each new token embedding is initialized to the centroid of the source embedding
space. This is the default and the method used for all headline VEXMLM results.

Additional strategies exist for the Table 5 ablation and for research use:

    random            -- Table 5 arm "Vocabulary Expansion + Random Init"
    constituent_mean  -- research extension: centroid of a token's own subword
                         decomposition rather than of the whole vocabulary
    mixed             -- reference only; see docs/REFERENCE_ARTIFACTS.md

Every strategy is seeded and deterministic given (seed, embedding matrix).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)

STRATEGIES = ("global_mean", "constituent_mean", "random", "mixed")


@dataclass
class InitStats:
    """Diagnostics used to verify which strategy produced a checkpoint."""
    strategy: str
    n_new: int
    hidden_size: int
    new_norm_mean: float
    new_perdim_std: float
    dist_to_global_mean_mean: float
    dist_to_global_mean_min: float
    dist_to_global_mean_max: float
    old_norm_mean: float
    old_perdim_std: float
    fallback_count: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _stats(strategy: str, old: torch.Tensor, new: torch.Tensor,
           fallback: int = 0) -> InitStats:
    gmean = old.mean(0)
    d = (new - gmean).norm(dim=1)
    return InitStats(
        strategy=strategy,
        n_new=int(new.shape[0]),
        hidden_size=int(new.shape[1]),
        new_norm_mean=round(new.norm(dim=1).mean().item(), 4),
        new_perdim_std=round(new.std().item(), 4),
        dist_to_global_mean_mean=round(d.mean().item(), 4),
        dist_to_global_mean_min=round(d.min().item(), 4),
        dist_to_global_mean_max=round(d.max().item(), 4),
        old_norm_mean=round(old.norm(dim=1).mean().item(), 4),
        old_perdim_std=round(old.std().item(), 4),
        fallback_count=fallback,
    )


@torch.no_grad()
def global_mean_init(emb: torch.Tensor, n_old: int, *, seed: int = 42,
                     noise_std: float = 0.0) -> InitStats:
    """OFFICIAL METHOD (paper Sec. 3.3): e_t = (1/|V_s|) * sum e_s.

    Every new row is set to the centroid of the source (pretrained) embedding
    space. This is the paper's specified initialization and the default.

    With noise_std=0 -- the paper's formulation -- all new rows are identical at
    initialization and are separated by gradients during Stage 1 MLM
    pretraining. `noise_std` is offered as an optional research knob only; leave
    it at 0 to follow the paper.
    """
    old = emb[:n_old]
    gmean = old.mean(0)
    emb[n_old:] = gmean
    if noise_std > 0:
        g = torch.Generator(device="cpu").manual_seed(seed)
        emb[n_old:] += torch.randn(emb[n_old:].shape, generator=g).to(emb.device) * noise_std
    return _stats("global_mean", old, emb[n_old:])


@torch.no_grad()
def constituent_mean_init(emb: torch.Tensor, n_old: int, new_tokens: list[str],
                          base_tokenizer, *, seed: int = 42,
                          noise_std: float = 0.0) -> InitStats:
    """RESEARCH EXTENSION -- not the paper's method. Use `global_mean` for
    official VEXMLM results.

    Each new row = centroid of the OLD embeddings of its own subword
    decomposition (Hewitt 2021; Minixhofer et al. 2022 "WECHSEL"). A new token
    'ትግርኛ' is initialized from the pieces the base tokenizer already splits it
    into, so it starts near its own meaning rather than at the global centroid.

    Provided because it is a strictly more informative variant of the same
    mean-based idea and is useful for ablation beyond Table 5. Tokens whose
    decomposition is empty fall back to the global mean; the count is reported
    in InitStats.fallback_count.
    """
    old = emb[:n_old]
    gmean = old.mean(0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    unk_id = getattr(base_tokenizer, "unk_token_id", None)
    fallback = 0

    for offset, token in enumerate(new_tokens):
        surface = token.lstrip("▁")
        ids = base_tokenizer.encode(surface, add_special_tokens=False)
        ids = [i for i in ids if i < n_old and i != unk_id]
        if ids:
            emb[n_old + offset] = old[ids].mean(0)
        else:
            emb[n_old + offset] = gmean
            fallback += 1

    if noise_std > 0:
        emb[n_old:] += torch.randn(emb[n_old:].shape, generator=g).to(emb.device) * noise_std
    if fallback:
        log.warning("constituent_mean: %d/%d tokens fell back to global mean",
                    fallback, len(new_tokens))
    return _stats("constituent_mean", old, emb[n_old:], fallback)


@torch.no_grad()
def random_init(emb: torch.Tensor, n_old: int, *, seed: int = 42,
                std: float | None = None, match_base: bool = True) -> InitStats:
    """Table 5 ablation arm: "Vocabulary Expansion + Random Init".

    By default matches the per-dimension std of the PRETRAINED embeddings
    (~0.21 for XLM-R) rather than N(0,1), so the ablation isolates the effect of
    *mean vs. random* placement rather than confounding it with a large scale
    mismatch. Pass match_base=False for unit-normal draws.
    """
    old = emb[:n_old]
    if std is None:
        std = old.std().item() if match_base else 1.0
    g = torch.Generator(device="cpu").manual_seed(seed)
    emb[n_old:] = torch.randn(emb[n_old:].shape, generator=g).to(emb.device) * std
    return _stats("random", old, emb[n_old:])


@torch.no_grad()
def mixed_init(emb: torch.Tensor, n_old: int, *, seed: int = 42,
               legacy_unit_normal: bool = True) -> InitStats:
    """REFERENCE ONLY -- (random + global_mean) / 2. Not a paper method.

    Retained because pre-existing Ge'ez XLM-R checkpoints in the wild were built
    this way, and identifying them requires being able to generate the same
    signature (see vocabulary_expansion/verify_vocab.py and
    docs/REFERENCE_ARTIFACTS.md). Not part of VEXMLM; do not use for results.

    With legacy_unit_normal=True the random term is N(0,1), giving new rows a
    per-dim std of ~0.5 against ~0.21 for pretrained rows.
    """
    old = emb[:n_old]
    gmean = old.mean(0)
    std = 1.0 if legacy_unit_normal else old.std().item()
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(emb[n_old:].shape, generator=g).to(emb.device) * std
    emb[n_old:] = (noise + gmean) / 2
    return _stats("mixed", old, emb[n_old:])


@torch.no_grad()
def initialize(strategy: str, emb: torch.Tensor, n_old: int, *,
               new_tokens: list[str] | None = None, base_tokenizer=None,
               seed: int = 42, **kwargs) -> InitStats:
    """Dispatch to a named strategy."""
    if strategy == "global_mean":
        return global_mean_init(emb, n_old, seed=seed, **kwargs)
    if strategy == "constituent_mean":
        if new_tokens is None or base_tokenizer is None:
            raise ValueError("constituent_mean requires new_tokens and base_tokenizer")
        return constituent_mean_init(emb, n_old, new_tokens, base_tokenizer,
                                     seed=seed, **kwargs)
    if strategy == "random":
        return random_init(emb, n_old, seed=seed, **kwargs)
    if strategy == "mixed":
        return mixed_init(emb, n_old, seed=seed, **kwargs)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")


def predict_distance(strategy: str, hidden_size: int, base_std: float,
                     *, legacy_unit_normal: bool = True) -> float:
    """Closed-form E[||new - global_mean||], used to identify a checkpoint's method.

    For an isotropic N(0, s^2) offset in d dimensions the expected norm is
    approximately s*sqrt(d), which is what makes the strategies separable.
    """
    d = hidden_size ** 0.5
    if strategy == "global_mean":
        return 0.0
    if strategy == "random":
        return (1.0 if legacy_unit_normal else base_std) * d
    if strategy == "mixed":
        return 0.5 * (1.0 if legacy_unit_normal else base_std) * d
    raise ValueError(f"no closed form for {strategy!r}")
