# Ablation on validation

Threshold-tuned validation macro-F1 on the subset languages. No test file was read.
Noise = max(0.5, B4 seed std) = 0.86 points. Rule: docs/DECISIONS.md, "Ablation".

## Each addition alone (B4 plus one, seed 42)

| Run | Macro-F1 | Vs B4 | eng | hau | mar | ukr | Pearson | Fidelity | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| B4 | 67.05 | +0.00 | 68.44 | 60.93 | 88.19 | 50.65 | - | 0.182 | 0.1013 |
| B4 + N1 (hurts) | 61.34 | -5.71 | 65.30 | 50.98 | 81.13 | 47.94 | - | 0.370 | 0.1511 |
| B4 + N2 (neutral) | 67.76 | +0.71 | 68.44 | 64.16 | 88.07 | 50.39 | - | 0.168 | 0.0875 |
| B4 + N4 (hurts) | 59.91 | -7.14 | 60.66 | 48.23 | 79.77 | 51.00 | - | 0.479 | 0.1444 |
| B4 + N5 (hurts) | 61.97 | -5.08 | 66.63 | 52.90 | 76.65 | 51.72 | - | 0.492 | 0.1097 |
| B4 + N6 (neutral) | 67.43 | +0.38 | 65.08 | 65.49 | 86.86 | 52.27 | 0.530 | 0.195 | 0.0861 |

## Cumulative (seed 42)

| Run | Macro-F1 | Vs B4 | eng | hau | mar | ukr | Pearson | Fidelity | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| b2_subset | 69.30 | +2.25 | 68.84 | 65.26 | 88.68 | 54.44 | - | 0.139 | 0.0735 |
| b4_subset_s42 | 67.05 | +0.00 | 68.44 | 60.93 | 88.19 | 50.65 | - | 0.182 | 0.1013 |
| s1_subset | 61.43 | -5.62 | 65.88 | 51.51 | 80.81 | 47.54 | - | 0.324 | 0.1459 |
| s2_subset | 66.76 | -0.29 | 69.63 | 61.68 | 82.43 | 53.30 | - | 0.228 | 0.1029 |
| final_subset_s42 | 66.03 | -1.02 | 68.47 | 61.02 | 81.88 | 52.76 | 0.549 | 0.243 | 0.0989 |

## Three seeds

| Model | Mean | Std | Seeds |
|---|---|---|---|
| b4 | 67.23 | 0.86 | 67.05, 66.46, 68.16 |
| final | 66.97 | 0.81 | 66.03, 67.39, 67.47 |
| nq | 66.67 | 1.13 | 67.26, 65.36, 67.38 |

CLAUDE.md 9.6 check: final minus B4 = -0.26 points (validation; limit -2).
