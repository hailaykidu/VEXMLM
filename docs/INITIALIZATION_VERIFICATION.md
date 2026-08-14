# Initialization Verification

How to confirm that a checkpoint's new embeddings were produced by the paper's
mean-based initialization — and how to identify the method when they were not.

## The official method

Paper Sec. 3.3: each new token embedding is the centroid of the source
embedding space.

$$e_t = \frac{1}{|V_s|}\sum_{s \in V_s} e_s$$

Implemented as `global_mean` in
[`vocabulary_expansion/initialization.py`](../vocabulary_expansion/initialization.py).
Its defining property is exact: **every new row equals the centroid**, so the
distance from any new row to the global mean is 0.

## Why the strategies are separable

For an isotropic offset drawn from $\mathcal{N}(0, s^2)$ in $d$ dimensions, the
expected distance from the centroid concentrates at $s\sqrt{d}$. With
$d = 768$ this separates the strategies cleanly:

| Strategy | New row | Predicted E‖e_new − ē‖ |
|---|---|---|
| `global_mean` | $\bar e$ | **0** |
| `mixed` | $(\mathcal{N}(0,1) + \bar e)/2$ | $0.5\sqrt{768} = 13.856$ |
| `random` (unit) | $\mathcal{N}(0,1)$ | $1.0\sqrt{768} = 27.713$ |
| `random` (base-scale) | $\mathcal{N}(0, 0.212^2)$ | $0.212\sqrt{768} = 5.88$ |
| `constituent_mean` | mean of own subwords | small, **high row-to-row variance** |

`constituent_mean` has no closed form — it is identified instead by a small mean
distance combined with a high coefficient of variation, since each row averages
a different subword set. `global_mean` has a near-zero CV.

## Running the check

```bash
python vocabulary_expansion/verify_vocab.py \
    --model checkpoints/vexmlm-expanded --n-old 250002 \
    --out results/init_verification.json
```

The tool reports observed statistics, the relative error against each closed
form, a verdict, and a histogram of per-row distances.

## Verified: a correctly built VEXMLM model

Expanding `xlm-roberta-base` with `--init global_mean`:

```
init=global_mean  new_std=0.0931  dist_to_mean=0.0000
VERDICT: global_mean (relative error 0.0, confidence high)
```

```json
"dist_to_global_mean": { "mean": 0.0, "min": 0.0, "max": 0.0 }
```

Distance is exactly 0 across all new rows — the formula holds. Note that
`new_perdim_std` is 0.0931 rather than 0: that is the spread of the centroid
vector's own 768 components, not variation between rows. All rows are identical.

## Interpreting a non-zero result

A non-zero mean distance means the checkpoint was **not** built with the paper's
method. The verifier names the most likely alternative. For a worked example on a
real checkpoint that reports `mixed` at 0.0255% relative error, see
[REFERENCE_ARTIFACTS.md](REFERENCE_ARTIFACTS.md#finding-3--the-reference-checkpoint-uses-a-non-paper-initialization).

## A property worth knowing

Under `global_mean` all new embeddings start **identical**, and are separated
only by gradients during Stage 1. Two consequences:

1. **Stage 1 is not optional** if the new tokens are to become useful. An
   expansion evaluated without continued pretraining (Table 5, arm 3) is
   measuring tokenization gains, not learned representations.
2. Any residual variance among new rows in a post-Stage-1 checkpoint is
   *learned*, so `verify_vocab.py` should be run on the **pre-pretraining**
   checkpoint when auditing the initialization itself.

`noise_std` is available as a research knob to break the tie at init; leave it
at 0 to follow the paper.

## Tests

`tests/test_initialization.py` asserts the formula directly
(`test_global_mean_implements_paper_formula`), checks each strategy's signature,
verifies mutual separability, and confirms pretrained rows are never modified.

```bash
pytest tests/test_initialization.py -v
```
