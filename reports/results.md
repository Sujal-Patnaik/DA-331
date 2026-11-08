# Results

Scores are macro-F1 in percentage points unless stated. Thresholds were tuned on validation only.

## Table 1. Full data, test split

| Language | B1 | B4 full | Final full | Track B Pearson (final) |
|---|---|---|---|---|
| afr | 21.13 | 42.63 | 47.48 | n/a |
| amh | 52.94 | 64.57 | 61.18 | 0.584 |
| arq | 45.67 | 49.28 | 51.88 | 0.301 |
| ary | 31.92 | 46.81 | 43.56 | n/a |
| chn | 27.15 | 60.12 | 58.99 | 0.532 |
| deu | 43.02 | 59.35 | 59.10 | 0.551 |
| eng | 48.40 | 62.92 | 65.82 | 0.593 |
| esp | 51.28 | 71.86 | 71.61 | 0.664 |
| hau | 55.05 | 57.16 | 53.91 | 0.475 |
| hin | 65.93 | 78.33 | 76.38 | n/a |
| ibo | 46.15 | 44.81 | 29.29 | n/a |
| kin | 33.29 | 33.89 | 23.38 | n/a |
| mar | 62.43 | 76.74 | 76.15 | n/a |
| orm | 41.45 | 48.10 | 33.69 | n/a |
| pcm | 47.38 | 55.18 | 49.85 | n/a |
| ptbr | 30.10 | 51.42 | 48.24 | 0.460 |
| ptmz | 21.06 | 43.87 | 30.97 | n/a |
| ron | 57.82 | 69.92 | 70.77 | 0.580 |
| rus | 47.56 | 77.46 | 77.21 | 0.747 |
| som | 39.81 | 45.29 | 30.33 | n/a |
| sun | 37.65 | 42.46 | 41.46 | n/a |
| swa | 23.01 | 30.43 | 24.21 | n/a |
| swe | 37.33 | 55.77 | 50.64 | n/a |
| tat | 33.30 | 59.89 | 40.86 | n/a |
| tir | 47.25 | 48.61 | 41.59 | n/a |
| ukr | 24.26 | 58.30 | 52.11 | 0.505 |
| vmw | 15.99 | 15.16 | 7.63 | n/a |
| yor | 24.97 | 20.91 | 13.92 | n/a |
| **mean** | 39.76 | 52.54 | 47.58 | 0.545 |

## Table 2. Cumulative ablation on the subset (seed 42, test split)

| Run | Model | Macro-F1 |
|---|---|---|
| b2_subset | B2 linear head | n/a |
| b4_subset_s42 | B4 faithful LDGN | n/a |
| s1_subset | S1 = B4 + N1 + N2 | n/a |
| s2_subset | S2 = S1 + N4 + N5 | n/a |
| final_subset_s42 | Final = S2 + N6 | n/a |

## Table 3. B4 versus Final on the subset, 3 seeds

| Model | Mean | Std | Seeds |
|---|---|---|---|
| B4 | n/a | n/a | 0 |
| Final | n/a | n/a | 0 |

## Significance, fidelity, calibration

- Paired bootstrap final_full vs b4_full (1,000 resamples): difference -4.97 points, 95% CI [-5.44, -4.50], p = 1.000.
- b4_full: co-occurrence fidelity 0.4028 (lower is better); ECE 0.1250.
- final_full: co-occurrence fidelity 0.4600 (lower is better); ECE 0.1569.

## Per-emotion F1 (mean over languages)

| Emotion | B4 full | Final full |
|---|---|---|
| anger | 47.48 | 44.18 |
| disgust | 43.91 | 40.12 |
| fear | 47.14 | 40.05 |
| joy | 66.10 | 61.16 |
| sadness | 59.52 | 54.55 |
| surprise | 50.74 | 44.38 |

## Intensity confusion, final_full (rows = gold 0-3, columns = predicted 0-3)

**anger**

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 12812 | 2034 | 120 | 0 |
| 1 | 1025 | 1765 | 98 | 0 |
| 2 | 366 | 1122 | 240 | 0 |
| 3 | 58 | 185 | 44 | 0 |

**disgust**

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 12256 | 1484 | 51 | 0 |
| 1 | 967 | 937 | 10 | 0 |
| 2 | 241 | 789 | 154 | 0 |
| 3 | 34 | 153 | 28 | 0 |

**fear**

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 14842 | 1576 | 172 | 2 |
| 1 | 446 | 903 | 409 | 3 |
| 2 | 116 | 463 | 548 | 47 |
| 3 | 12 | 84 | 201 | 45 |

**joy**

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 14487 | 923 | 79 | 0 |
| 1 | 805 | 866 | 235 | 0 |
| 2 | 321 | 682 | 1119 | 3 |
| 3 | 18 | 72 | 256 | 3 |

**sadness**

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 13939 | 1573 | 155 | 1 |
| 1 | 1102 | 1018 | 252 | 4 |
| 2 | 281 | 588 | 514 | 23 |
| 3 | 23 | 115 | 240 | 41 |

**surprise**

| | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 15440 | 1453 | 36 | 0 |
| 1 | 862 | 956 | 49 | 0 |
| 2 | 251 | 473 | 195 | 5 |
| 3 | 30 | 60 | 58 | 1 |


## Track C zero-shot (languages with no training data)

| Language | Macro-F1 |
|---|---|
| ind | 44.31 |
| jav | 32.30 |
| xho | 5.13 |
| zul | 5.16 |
| **mean** | 21.72 |

Figures are in reports/figures/ (`python scripts/make_figures.py --results`). Attention examples quote BRIGHTER text, so they stay in runs/<id>/analysis/, which is not committed (rule 7).
