# best_full on test

Macro-F1 in percentage points. best_full was chosen on validation and scored once on test.
Test scores of the earlier full-data models had already been seen when the selection rule was written, so this is not a fully blind comparison.

| Group | B4 | Final | Final, no quota | Best (B4 + N6) |
|---|---|---|---|---|
| Validation, tuned | 54.91 | 49.44 | 53.85 | 55.89 |
| Test, all 28 | 52.54 | 47.58 | 51.68 | 53.47 |
| Test, 11 with intensity | 62.03 | 60.98 | 63.74 | 63.04 |
| Test, 17 without | 46.41 | 38.91 | 43.88 | 47.28 |

Track B Pearson: final_full 0.545, final_full_noquota 0.600, best_full 0.574.

- best_full minus b4_full: +0.92 points, 95% CI [0.46, 1.39], one-sided p = 0.000.
- best_full minus final_full: +5.89 points, 95% CI [5.37, 6.37], one-sided p = 0.000.
- best_full minus final_full_noquota: +1.78 points, 95% CI [1.28, 2.26], one-sided p = 0.000.
- b4_full: co-occurrence fidelity 0.4028, ECE 0.1250.
- final_full: co-occurrence fidelity 0.4600, ECE 0.1569.
- final_full_noquota: co-occurrence fidelity 0.2910, ECE 0.1356.
- best_full: co-occurrence fidelity 0.3353, ECE 0.1201.
- best_full beats B4 on 18 of 28 languages. Largest losses: som -3.5, swe -1.6, swa -1.1. Largest gains: vmw +4.4, kin +4.4, tat +3.4.

## Track C zero-shot

| Model | Mean | Per language |
|---|---|---|
| final_full | 21.72 | ind 44.31, jav 32.30, xho 5.13, zul 5.16 |
| final_full_noquota | 23.65 | ind 48.13, jav 34.58, xho 5.79, zul 6.09 |
| best_full | 20.90 | ind 45.78, jav 29.24, xho 3.45, zul 5.11 |

## Per language

| Language | B1 | B4 | Final | No quota | Best | Pearson (best) |
|---|---|---|---|---|---|---|
| afr | 21.13 | 42.63 | 47.48 | 42.95 | 43.81 | n/a |
| amh | 52.94 | 64.57 | 61.18 | 64.06 | 63.93 | 0.603 |
| arq | 45.67 | 49.28 | 51.88 | 52.07 | 51.11 | 0.367 |
| ary | 31.92 | 46.81 | 43.56 | 45.80 | 47.41 | n/a |
| chn | 27.15 | 60.12 | 58.99 | 62.39 | 60.88 | 0.565 |
| deu | 43.02 | 59.35 | 59.10 | 61.09 | 61.28 | 0.579 |
| eng | 48.40 | 62.92 | 65.82 | 66.81 | 65.01 | 0.563 |
| esp | 51.28 | 71.86 | 71.61 | 72.73 | 74.49 | 0.698 |
| hau | 55.05 | 57.16 | 53.91 | 57.77 | 58.98 | 0.590 |
| hin | 65.93 | 78.33 | 76.38 | 83.44 | 78.22 | n/a |
| ibo | 46.15 | 44.81 | 29.29 | 35.54 | 47.57 | n/a |
| kin | 33.29 | 33.89 | 23.38 | 35.59 | 38.26 | n/a |
| mar | 62.43 | 76.74 | 76.15 | 81.54 | 78.93 | n/a |
| orm | 41.45 | 48.10 | 33.69 | 37.93 | 48.80 | n/a |
| pcm | 47.38 | 55.18 | 49.85 | 54.48 | 54.14 | n/a |
| ptbr | 30.10 | 51.42 | 48.24 | 51.06 | 51.00 | 0.503 |
| ptmz | 21.06 | 43.87 | 30.97 | 39.32 | 43.47 | n/a |
| ron | 57.82 | 69.92 | 70.77 | 72.46 | 70.87 | 0.557 |
| rus | 47.56 | 77.46 | 77.21 | 82.66 | 78.05 | 0.746 |
| som | 39.81 | 45.29 | 30.33 | 40.65 | 41.78 | n/a |
| sun | 37.65 | 42.46 | 41.46 | 39.73 | 43.63 | n/a |
| swa | 23.01 | 30.43 | 24.21 | 27.39 | 29.30 | n/a |
| swe | 37.33 | 55.77 | 50.64 | 55.94 | 54.16 | n/a |
| tat | 33.30 | 59.89 | 40.86 | 57.51 | 63.30 | n/a |
| tir | 47.25 | 48.61 | 41.59 | 44.89 | 47.81 | n/a |
| ukr | 24.26 | 58.30 | 52.11 | 58.07 | 57.82 | 0.545 |
| vmw | 15.99 | 15.16 | 7.63 | 8.19 | 19.54 | n/a |
| yor | 24.97 | 20.91 | 13.92 | 15.04 | 23.60 | n/a |

## Per emotion

| Emotion | B4 | Final | Final, no quota | Best (B4 + N6) |
|---|---|---|---|---|
| anger | 47.48 | 44.18 | 49.65 | 49.17 |
| disgust | 43.91 | 40.12 | 44.60 | 46.61 |
| fear | 47.14 | 40.05 | 45.14 | 47.13 |
| joy | 66.10 | 61.16 | 63.98 | 66.13 |
| sadness | 59.52 | 54.55 | 58.28 | 59.52 |
| surprise | 50.74 | 44.38 | 47.82 | 51.88 |
