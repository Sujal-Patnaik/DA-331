# Full-data comparison, test split

Macro-F1 in percentage points. The model was chosen on validation, then scored once on test.

| Group | B4 | Final | Final, no quota |
|---|---|---|---|
| All 28 | 52.54 | 47.58 | 51.68 |
| 11 with intensity labels | 62.03 | 60.98 | 63.74 |
| 17 without | 46.41 | 38.91 | 43.88 |

Track B Pearson: final 0.545, no quota 0.600.

- No quota minus final_full: 4.11 points, 95% CI [3.66, 4.57], one-sided p = 0.000.
- No quota minus b4_full: -0.86 points, 95% CI [-1.30, -0.37], one-sided p = 1.000.
- b4_full: co-occurrence fidelity 0.4028, ECE 0.1250.
- final_full: co-occurrence fidelity 0.4600, ECE 0.1569.
- final_full_noquota: co-occurrence fidelity 0.2910, ECE 0.1356.

## Track C zero-shot

Mean 23.65: ind 48.13, jav 34.58, xho 5.79, zul 6.09.

## Per language

| Language | B1 | B4 | Final | No quota | Pearson (no quota) |
|---|---|---|---|---|---|
| afr | 21.13 | 42.63 | 47.48 | 42.95 | n/a |
| amh | 52.94 | 64.57 | 61.18 | 64.06 | 0.625 |
| arq | 45.67 | 49.28 | 51.88 | 52.07 | 0.366 |
| ary | 31.92 | 46.81 | 43.56 | 45.80 | n/a |
| chn | 27.15 | 60.12 | 58.99 | 62.39 | 0.601 |
| deu | 43.02 | 59.35 | 59.10 | 61.09 | 0.586 |
| eng | 48.40 | 62.92 | 65.82 | 66.81 | 0.617 |
| esp | 51.28 | 71.86 | 71.61 | 72.73 | 0.697 |
| hau | 55.05 | 57.16 | 53.91 | 57.77 | 0.567 |
| hin | 65.93 | 78.33 | 76.38 | 83.44 | n/a |
| ibo | 46.15 | 44.81 | 29.29 | 35.54 | n/a |
| kin | 33.29 | 33.89 | 23.38 | 35.59 | n/a |
| mar | 62.43 | 76.74 | 76.15 | 81.54 | n/a |
| orm | 41.45 | 48.10 | 33.69 | 37.93 | n/a |
| pcm | 47.38 | 55.18 | 49.85 | 54.48 | n/a |
| ptbr | 30.10 | 51.42 | 48.24 | 51.06 | 0.502 |
| ptmz | 21.06 | 43.87 | 30.97 | 39.32 | n/a |
| ron | 57.82 | 69.92 | 70.77 | 72.46 | 0.603 |
| rus | 47.56 | 77.46 | 77.21 | 82.66 | 0.860 |
| som | 39.81 | 45.29 | 30.33 | 40.65 | n/a |
| sun | 37.65 | 42.46 | 41.46 | 39.73 | n/a |
| swa | 23.01 | 30.43 | 24.21 | 27.39 | n/a |
| swe | 37.33 | 55.77 | 50.64 | 55.94 | n/a |
| tat | 33.30 | 59.89 | 40.86 | 57.51 | n/a |
| tir | 47.25 | 48.61 | 41.59 | 44.89 | n/a |
| ukr | 24.26 | 58.30 | 52.11 | 58.07 | 0.574 |
| vmw | 15.99 | 15.16 | 7.63 | 8.19 | n/a |
| yor | 24.97 | 20.91 | 13.92 | 15.04 | n/a |

## Per emotion

| Emotion | B4 | Final | No quota |
|---|---|---|---|
| anger | 47.48 | 44.18 | 49.65 |
| disgust | 43.91 | 40.12 | 44.60 |
| fear | 47.14 | 40.05 | 45.14 |
| joy | 66.10 | 61.16 | 63.98 |
| sadness | 59.52 | 54.55 | 58.28 |
| surprise | 50.74 | 44.38 | 47.82 |
