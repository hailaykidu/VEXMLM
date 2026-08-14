# Dataset Card — AfriSenti-SemEval 2023 Task 12

| Field | Value |
|---|---|
| Key | `afrisenti` |
| Task | sentiment |
| Languages | amh, tir, mul |
| Licence | CC BY 4.0 |
| Source | https://github.com/afrisenti-semeval/afrisent-semeval-2023 |
| Hub ID | shmuhammad/AfriSenti-twitter-sentiment |

## Notes

3-class: positive/negative/neutral. Amharic ('amh') is a native subtask. Tigrinya ('tir') appears in the AfriSenti collection but not in SemEval Subtask B; verify the config before use.

## Split statistics

| Split | Rows | SHA-256 (32) | Mean chars |
|---|---|---|---|
| train | 63,685 | `0f95fe352713817477399fb8da5d742c` | 85.84 |
| validation | 13,653 | `cb09a174bd7be8ee623f3202d18d0c1d` | 90.24 |
| test | 30,211 | `a6ddaabde5531183a49f79b3bc695134` | 75.73 |

### Label distribution — train

| Label | Count |
|---|---|
| negative | 20,108 |
| neutral | 22,794 |
| positive | 20,783 |

### Label distribution — validation

| Label | Count |
|---|---|
| negative | 4,341 |
| neutral | 4,899 |
| positive | 4,413 |

### Label distribution — test

| Label | Count |
|---|---|
| negative | 10,204 |
| neutral | 10,207 |
| positive | 9,800 |

## Citation

```bibtex
@inproceedings{muhammad-etal-2023-afrisenti,
  title={{A}fri{S}enti: A Twitter Sentiment Analysis Benchmark for African Languages},
  booktitle={SemEval-2023},
  year={2023}
}
```
