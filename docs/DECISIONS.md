# Decisions

Every assumption or deviation, one line each, dated.

## Environment
- 2026-11-05: Python 3.12 instead of 3.10 or 3.11. PySpark 3.5.9, pandas UDFs and fasttext-wheel all work on it (`scripts/check_env.py`).
- 2026-11-05: Java 11 instead of 17. Spark 3.5 supports it, it passes every check, and no sudo was available to install 17.
- 2026-11-05: Torch from the cu126 index (2.14.1): the driver reports CUDA 13.1 and cu121 has no build for this Python. Pinned `pandas<3` and `transformers<5` for PySpark 3.5 and the code here.
- 2026-11-05: UDF modules import pandas at module level and define UDFs after the session exists. With `from __future__ import annotations`, Spark resolves type hints from module globals.
- 2026-11-05: XLM-R weights were fetched with HF_HUB_DISABLE_XET=1 because the Xet transfer stalled at 335 MB. If it stalls again, set that variable.

## Data
- 2026-11-05: SemEval files are the source of truth. English Track A test labels are blank, so they come from SemEval Track C by text, with row-aligned Track B intensity > 0 for 6 rows whose text has conflicting Track C duplicates.
- 2026-11-05: Hugging Face labels are not used: its English test split has 5,528 rows and only 188 texts match SemEval exactly. It is still downloaded, as section 3 asks.
- 2026-11-05: `track_a/train/ron_track1_test.csv` is ignored: stray file, different schema, overlaps Romanian test.
- 2026-11-05: Intensities join on (lang, text), not (lang, id). Ids point at different texts for amh, rus and part of English train.
- 2026-11-05: Track B is scored on rows of the Track A test part, not the official Track B test files. Their amh and rus splits contain Track A training texts. Consequence: amh and rus Track B numbers are not on the official row sets.
- 2026-11-05: Only ind, jav, xho and zul count as Track C. The other Track C files are Track A test under new ids, so they are not zero-shot.
- 2026-11-05: EduRABSA's git repository holds only a pointer, so the data comes from `yhua219/EduRABSA_ASQE` on Hugging Face. Its licence is CC BY-NC-SA 4.0, stricter than section 3 states.
- 2026-11-05: Coursera skipped: no ~/.kaggle/kaggle.json. Its loader is tested on a synthetic file only, assuming columns reviews, rating and course_id.

## Preprocessing and split
- 2026-11-05: Emoticons need no following word character or slash, so "Note:Done" and "a:/b" stay intact. URLs are masked before emoticons.
- 2026-11-05: The M3 rule "languages holding more than 10% of training rows" is read collectively, because no single language reaches 10%. p95 exceeds 64 for languages holding 47.9% of rows, so max_len is 128. It lives in data/gold/tokenizer_meta.json; config/default.yaml keeps the starting 64.
- 2026-11-05: Prevalence check: the 15% hold-out has a maximum gap of 0.35 pp. With official dev included, one pair exceeds 3 pp (afr joy, 3.73 pp, 98 dev rows). Accepted: official dev is required in validation and cannot be stratified.
- 2026-11-05: Exact-duplicate texts across official splits are left as released (reports/DATA_NOTES.md).

## Graphs
- 2026-11-05: Family and global counts sum only over languages that have both nodes of a pair, so missing emotions do not depress co-occurrence.
- 2026-11-05: One lambda per language and family (scope size), not per pair, so lambdas are readable for the report.
- 2026-11-05: B4 and S1 use raw per-language graphs; N4 selects the shrunk bank. B4 has no neutral node, so its row and column are zero and neutral is out of its loss.

## Model and training
- 2026-11-05: Found by test_overfit.py: raw definitions differ by about 5% of their norm, so all labels attended alike. Anchors are centred and rescaled to a random query's norm. N1 only.
- 2026-11-05: Found by test_overfit.py: the unscaled 256-dimension dot product saturated the dynamic graph. It is now divided by sqrt(256). Applied to every LDGN head, B4 included.
- 2026-11-05: Found by test_overfit.py: attended states are small (about 0.09 per dimension), so U gets a LayerNorm. Applied to every LDGN head including B4, so the numeric fixes cannot inflate the final model's gain. B4 therefore differs slightly from LDGN as published; the concern applies equally to all LDGN runs.
- 2026-11-05: Node mask also applies to the residual delta and the dynamic graph, so masked emotions stay isolated.
- 2026-11-05: Per-node presence bias b_c (plan 5.7, absent from the plan's snippet) exists only with N6.
- 2026-11-05: Losses run in float32. The plan's clamp(min=1e-6) underflows in float16, giving log(0). The Pearson denominator uses a smoothed sqrt for finite gradients at zero.
- 2026-11-05: Attention keeps the sqrt(D) scale in B4. A rescaled trainable query expresses the same function, so this does not change LDGN.
- 2026-11-05: `min_intensity_rows_per_batch: 32` cannot fit a 16-row micro-batch, so the quota is half the micro-batch (8 per micro-batch, 16 per effective batch). Pearson estimates are noisier than plan 5.7 intended.
- 2026-11-05: LR decay spans extend_to where defined; a horizon of max would leave LR at 0 during the extension. final_full therefore ends epoch 5 at a non-zero LR.
- 2026-11-05: Two parameter groups, as section 8 says, so weight decay applies to all parameters.
- 2026-11-05: Validation at training time uses threshold 0.5 without the neutral override; thresholds and the override apply after training.

- 2026-11-05: User accepted the test_overfit.py result for now (0.869 at 150 steps, needs > 0.9). The test is unchanged and still fails, so `pytest -q` reports 1 failure.
- 2026-11-06: User asked to run only b4_full and final_full now, ahead of the subset runs and out of section 6's order, because of limited time. The subset runs are postponed, not removed, so Tables 2 and 3 and the section 9.6 check stay open.
- 2026-11-06: User chose to run the two full-data runs locally on the 4 GB card, with the automatic OOM fallback, rather than on Colab.
- 2026-11-06: Training paused twice while running (about 118 minutes, then about 25), probably the laptop sleeping; both runs continued and lost no epochs. b4_full ran 8 epochs through the extension rule; final_full ran 5, with its best at epoch 4.
- 2026-11-06: Track B integer predictions use rounded expected intensity, chosen on validation (plan 5.7).
- 2026-11-06: User approved one added experiment, final_full_noquota: final_full without the 50% intensity batch quota, because the loss is concentrated in languages without intensity labels. Selection between models uses validation only. Results of the first full runs are preserved in runs/_backup_full_v1 and git tag results-full-v1.
- 2026-11-06: final_full_noquota is the application model: it has intensity, unlike b4_full, and was chosen over final_full on validation (53.85 vs 49.44). b4_full stays 0.86 points ahead on test. The choice followed sight of final_full's test scores, which the report should disclose.

## Ablation (fixed before any of these runs existed)
- 2026-11-07: User asked to find which additions help. Added nq_subset (final without the quota, 3 seeds) and a_n1 to a_n6 (B4 plus one addition, seed 42) to the planned subset runs. a_n6 has no quota. Only the mar subset language lacks intensity labels, so N6's effect on such languages cannot be judged from the subset alone.
- 2026-11-07: Selection rule. Metric: threshold-tuned validation macro-F1 averaged over the subset languages, compared with B4 at seed 42. With noise = max(0.5 points, std of the three B4 seeds), an addition helps above +noise, hurts below -noise, otherwise is neutral. Keep additions that help and drop those that hurt or are neutral. N6 is the exception: it stays unless it hurts, because the apps need intensity. Nothing is chosen on test scores.
- 2026-11-07: Selection result (reports/ablation_val.md): noise 0.86; N1 -5.71, N4 -7.14 and N5 -5.08 hurt; N2 +0.71 and N6 +0.38 are neutral. By the rule, best_full = B4 + N6 without the quota. N1, N4 and N5 were still improving at epoch 10, so "hurts" means worse within this epoch budget. No wiring bug was found for the single-flag runs. B2 (no graph) scored +2.25 on validation; it is reported but not part of this selection.
- 2026-11-07: The user chose to stop at best_full rather than also train B4 + N2 + N6, so N2 + N6 was never run. best_full test 53.47. On the subset test split, scored after the choice, N1, N4 and N5 lost 7 to 9 points, N2 gained 1.00 and N6 lost 0.96. N6 helped only in the full-data run.
- 2026-11-07: The test split is not touched during selection. Subset runs are scored on test only after the final choice is frozen. Earlier results are preserved in runs/_backup_full_v1, runs/_backup_v2_before_ablation and git tags. reports/results.md and reports/results_noquota.md are not rewritten.

## Applications
- 2026-11-06: States no longer use plan 9.3's pair score. The trained dynamic graph averages 0.004 for anger and sadness where both are high, so the rule could not fire. Pair states now need both emotions; stress and disappointment need their emotion to be the strongest negative one. tau_state is removed.
- 2026-11-06: Outrage without disgust needs anger intensity of at least 1.85, the top 10% on English support clauses, instead of 2.5, which only 0.4% reached. This value was set on demo data without labels.
- 2026-11-06: Provisional English fear threshold of 0.95 in the applications only, because 58% of English BRIGHTER is labelled fear against a median of 7% across languages. BRIGHTER scores are unchanged. Replace it with thresholds from the hand labels.
- 2026-11-06: Profiles extended with keywords from frequent words in `other` clauses, so they no longer match plan 9.2. "not arrived" was removed. Job 10 recomputes aspects from the current profile, so keyword changes need no rescoring. Against EduRABSA gold, precision fell from 0.66 to 0.63 and recall rose from 0.50 to 0.75.
- 2026-11-06: Tables before these changes are in runs/_backup_apps_v1, with git tag apps-rules-v1.
- 2026-11-06: Bug fixed: urgency and valence read the untrained disgust output for languages without disgust, setting urgency on 33.5% of English support clauses. Both now require the language mask, and predicted-neutral texts sit at the centre as plan 9.3 says. Figures reported before this fix are wrong (Spearman 0.74, 32% escalated). Model scores and emotion labels were not affected.
- 2026-11-05: The customer-support mirror has no timestamps. `period` is one of 12 equal hash buckets, not a calendar week, and the dashboard says so. Trend and spike logic is exercised but shows no real change over time.
- 2026-11-05: EduRABSA has no term field, so semester comparison shows an explanation until data has one. The plan's bootstrap intervals for it are not built.
- 2026-11-05: Groups: teacher reviews by the first id field, course reviews by the second, university reviews as one group. These fields are inferred from counts; EduRABSA documents no id format. The inference should be checked.
- 2026-11-05: Added apps/engine/scorer.py and scripts/export_artifacts.py, make_data_notes.py and make_annotation_templates.py, none of them in section 4's layout, plus `--run-id` on training and `EMOLDGN_CONFIG` for isolated smoke runs.
- 2026-11-05: apps/dashboards/common.py added for shared reading and heatmaps.
- 2026-11-05: tau_state 0.25 and related numbers are in the `apps:` block of config/default.yaml. tau_state is a starting value; the plan says to tune it on the hand-labelled samples.
- 2026-11-05: Keyword tagging allows a plural suffix on Latin-script keywords, so "exam" matches "exams".
- 2026-11-05: Attention examples quote BRIGHTER, so they stay in runs/<id>/analysis/ instead of reports/ (rule 7). reports/logs/ is gitignored for the same reason.
- 2026-11-05: Annotation templates are committed. They hold masked clauses from the demo data, not BRIGHTER; the customer-support source has no stated licence, so review this before publishing the repository.
- 2026-11-08: Added a Google Drive fast path (scripts/package_for_drive.py, scripts/restore_from_drive.py, README Path B). The bundle carries small files for all 21 finished runs but weights only for the exported app model, b4_full and best_full: the 17 subset checkpoints would add about 19 GB and nothing reads them. The subset tables regenerate byte-identical from the restored files alone. BRIGHTER-derived tables are rebuilt by the Spark jobs instead of being shipped; the prediction files do contain test labels, so the Drive link is to be restricted.
