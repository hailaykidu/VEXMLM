# Pre-training Corpus Report (Stage 0)

Deduplication and splitting of the Stage 1 corpora.

```bash
python datasets/prepare.py --config configs/base.yaml --dev-fraction 0.02 --seed 42
```

**Raw corpora are preserved unmodified.** Verified by checksum before and after:

| Corpus | SHA-256 (24) before | after |
|---|---|---|
| Amharic | `b5d5b1da90d3fff680d083dc` | `b5d5b1da90d3fff680d083dc` |
| Tigrinya | `e006c5ff19e3ec63f29b8628` | `e006c5ff19e3ec63f29b8628` |

## Line accounting

| Stage | Amharic | Tigrinya |
|---|---|---|
| Raw lines read | 200,002 | 200,000 |
| Dropped: duplicate | 70,070 | 59,164 |
| Dropped: too short (<10 chars) | 263 | 1 |
| Dropped: <50% Ge'ez | 267 | 252 |
| **Kept** | **129,402** | **140,583** |
| → train (98%) | 126,814 | 137,772 |
| → dev (2%) | 2,588 | 2,811 |
| OOV words (dev not in train) | 2,414 | 3,021 |

## Duplicate removal

| Metric | Amharic | Tigrinya |
|---|---|---|
| Duplicate lines removed | 70,070 | 59,164 |
| **Duplicate percentage removed** | **35.03%** | **29.58%** |
| Overall retention | 64.70% | 70.29% |

Verified post-hoc: **zero duplicate lines remain** in any processed split.

## Token counts after processing

| Split | Lines | Tokens |
|---|---|---|
| amharic.train | 126,814 | 1,263,961 |
| amharic.dev | 2,588 | 25,595 |
| tigrinya.train | 137,772 | 1,075,145 |
| tigrinya.dev | 2,811 | 21,968 |
| **Combined train** | **264,586** | **2,339,106** |

## Top duplicate examples (from the raw corpora)

The repeats are concentrated on religious passages -- a crawl artifact, not
natural recurrence.

### Amharic

| Repeats | Line (truncated) |
|---|---|
| 131× | ሁለቱንም በምር እንጅ አልፈጠርናቸውም፤ ግን አብዛኞቻቸው አያውቁም። |
| 105× | ረ (አሊፍ ላም ሚም ራ) ይህች (አናቅጽ) ከመጽሐፉ አንቀጾች ናት ፡ ፡ ያም ከጌታህ ወዳንተ የተወረደው እውነት… |
| 105× | "እስከሚቀሰቀሱበት ቀን ድረስ አቆየኝ" አለ። |
| 91× | ሙሳንም በታምራቶቻችን ወደ ፈሮዖንና ወደ ሹማምንቶቹ በእርግጥ ላክን፤ እኔ የአለማት ጌታ መልክተኛ ነኝ አላቸውም… |
| 82× | (45) በጌታውም ፊት መቆምን ለፈራ ሰው ሁለት ገነቶች አልሉት። |

### Tigrinya

| Repeats | Line (truncated) |
|---|---|
| 160× | ከምቲ ንነቢይ ሙሴ ዝተዛረቦ እግዚኣብሔር "ስመይ ኣብኡ ተሰምዩ እዩ እሞ በደልካ ኣይክሓድገልካን እዩ" (ዘጽ 2… |
| 148× | ካብ'ቲ ዝተወለድኩዎ ዝዓበኹዎን ዝኸፍአ ክፉእ የሎን። |
| 143× | / እምብኣርሲ እንተ በላዕኩም፡ ወይስ እንተ ሰቴኹም፡ ወይስ ዝዀነ ይኹን እንተ ገበርኩም፡ ኲሉ ንኽብሪ ኣምላኽ … |
| 130× | "ጋኔንንና (አጋንንት) ሰብን ክግዙ እለይ እምበር ንካልእ አይፈጠርኩዎምን" |
| 88× | ምእንቲ ጽድቂ ዚስደዱ፡ መንግስተ ሰማያት ናቶም እያ እሞ፡ ብጹኣን እዮም። |

## Why this matters

A line repeated 160 times contributes 160 gradient updates per epoch, biasing
the model toward the duplicated register. It also leaks across the split: an
identical line can land in both train and dev, making held-out perplexity
measure text the model has already seen verbatim. Deduplication before
splitting removes both problems.

## Outputs

```
datasets/processed/
├── amharic.train.txt        126,814 lines
├── amharic.dev.txt            2,588 lines
├── amharic.oov.txt            2,414 words (Table 3 input)
├── tigrinya.train.txt       137,772 lines
├── tigrinya.dev.txt           2,811 lines
├── tigrinya.oov.txt           3,021 words
└── preparation_report.json
```

Provenance of the raw corpora remains **UNKNOWN** -- see
[docs/MLM_CORPUS_ANALYSIS.md](../docs/MLM_CORPUS_ANALYSIS.md). Stage 0 does not
change that; it only makes the data fit to train on.
