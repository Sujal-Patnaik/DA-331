# GPU budget (M7)

Measured over 200 training steps of `final_full`: **101.9 samples/s**, peak allocated 3.09 GB, peak reserved 4.23 GB, on a 4.0 GB card (batch 16, max_len 128, fp16 True, frozen embeddings True).

Projection: T = N_train x E / samples per second, plus 10% for validation passes, at max epochs.

| Run | Data | N train | Epochs | Hours |
|---|---|---|---|---|
| b2_subset | subset | 8320 | 10 | 0.25 |
| b4_subset_s42 | subset | 8320 | 10 | 0.25 |
| s1_subset | subset | 8320 | 10 | 0.25 |
| s2_subset | subset | 8320 | 10 | 0.25 |
| final_subset_s42 | subset | 8320 | 10 | 0.25 |
| b4_subset_s1 | subset | 8320 | 10 | 0.25 |
| b4_subset_s2 | subset | 8320 | 10 | 0.25 |
| final_subset_s1 | subset | 8320 | 10 | 0.25 |
| final_subset_s2 | subset | 8320 | 10 | 0.25 |
| b4_full | full | 55317 | 5 | 0.83 |
| final_full | full | 55317 | 5 | 0.83 |

**Total: 3.90 hours** (subset runs 2.25, full runs 1.66). Early stopping usually ends runs sooner; the optional 3-epoch extension of final_full is not included.
