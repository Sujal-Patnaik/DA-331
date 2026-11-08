# How this project was done

This document explains what the project set out to do, the order in which the work happened, what failed, what changed as a result, and where things stand now. It is meant for someone who was not involved and wants to understand the real procedure, including the dead ends.

All work took place between 5 and 7 November 2026.

## 1. The project in one page

**Goal.** Detect six perceived emotions (anger, disgust, fear, joy, sadness, surprise) in short texts across 28 languages, using the BRIGHTER dataset from SemEval-2025 Task 11. Then use the model in two applications: Customer Support Analysis and Student Feedback Analysis.

**Approach.** Take LDGN, a graph network over emotion labels, put it on an XLM-RoBERTa encoder, and add five proposed improvements (N1, N2, N4, N5, N6).

**What actually happened.** The model with all five additions scored *worse* than plain LDGN. Removing one sampling rule recovered much of the loss. A planned comparison then showed that only one addition, the intensity head, was worth keeping.

| Model | Contents | Track A test | Track B | Track C |
|---|---|---|---|---|
| B1 | TF-IDF baseline | 39.76 | none | none |
| `final_full` | All five additions, with quota | 47.58 | 0.545 | 21.72 |
| `final_full_noquota` | All five additions, no quota | 51.68 | **0.600** | **23.65** |
| `b4_full` | LDGN, no additions | 52.54 | none | not scored |
| **`best_full`** | **LDGN + N6, no quota** | **53.47** | 0.574 | 20.90 |

**Final position.**

- Best research model: `best_full`.
- Model used by the applications: `final_full_noquota`, because it reads intensity better on English.
- Main lesson: more machinery did not mean more accuracy, and a training rule mattered more than most architecture changes.

## 2. Constraints that shaped the work

- **Hardware:** one laptop with an RTX 3050 (4 GB of GPU memory) and 9 GB of RAM in WSL2. The original spec suggested Colab for full runs at this memory size, but the team chose to run locally.
- **Time:** a total GPU budget of about 3 to 5 hours was planned, with a deadline of 8 November.
- **Data rules:** train only on BRIGHTER, never tune on the test split, and keep validation and test out of the graphs. Never lowercase text, and mask emotions a language does not have.
- **Licensing:** BRIGHTER is treated as research-only in this project, so its text must not appear in anything shared. EduRABSA is CC BY-NC-SA 4.0. The customer support data has no stated licence.

## 3. Timeline

| When | What happened |
|---|---|
| 5 Nov | Environment, downloads and data inspection |
| 5 Nov | Spark pipeline, graphs, baseline B1, model, training loop and tests |
| 5 Nov | Overfit test failed; three defects fixed; result accepted at 0.87 against a 0.90 target |
| 5 Nov | Application engine, dashboards and API built, using a smoke-test model |
| 6 Nov | Subset runs postponed for time; full runs of B4 and `final_full` done first |
| 6 Nov | `final_full` came in well below B4; cause traced to the intensity quota |
| 6 Nov | `final_full_noquota` trained; large recovery |
| 6 Nov | Applications scored with `final_full_noquota`; urgency bug found and fixed |
| 6 Nov | Team reviewed dashboard exports; fear problem found; rules reworked |
| 7 Nov | 17 subset runs under a pre-written selection rule |
| 7 Nov | `best_full` trained and scored once on test |
| 7 Nov | `best_full` tried on the applications; apps stayed on `final_full_noquota` |
| 7 Nov | Slides, report draft and notebook prepared |

## 4. Data work

The data was inspected before any processing, and that changed the pipeline.

| Finding | Risk if missed | What was done |
|---|---|---|
| English Track A test labels are blank | No English test score | Filled from Track C by text. Six rows with conflicting duplicates used Track B intensity above zero |
| Hugging Face English test split has 5,528 rows, and only 188 texts match SemEval exactly | Wrong labels | Hugging Face labels were not used. SemEval files are the source of truth |
| A stray file, `ron_track1_test.csv`, sits in the Romanian training folder | Test leakage | Ignored |
| Intensity ids do not line up with Track A for Amharic, Russian and part of English | Wrong intensity on rows | Joined on language and text |
| Amharic and Russian Track B splits contain Track A training texts | Test leakage | Track B is scored on the Track A test part, so these two are not on official row sets |
| Track C files repeat Track A test for the 28 languages | False zero-shot claims | Only Indonesian, Javanese, Xhosa and Zulu count as Track C |
| Exact duplicates cross official splits in a few languages | Slight optimism in scores | Left as released, and noted |
| Afrikaans has no surprise, and English has no disgust | Treating absence as a zero label | Masks are read from each file's header |

**Findings that mattered later:**

- Fear is 58% of English training rows, against a median of 7% across languages.
- The largest language has about four times the training rows of the smallest.
- 11 of 28 languages have intensity labels.

## 5. Pipeline

PySpark handled ingestion, cleaning, splitting, graph statistics, tokenised tables, the TF-IDF baseline, application scoring and aggregation. PyTorch handled neural training and testing.

1. **Ingest:** raw files to Parquet, 122,345 rows, with intensity and label mask tables.
2. **Clean:** emoji and emoticons to words, links and handles to placeholders, Unicode NFC. No lowercasing and no stop-word removal.
3. **Split:** 15% stratified hold-out per language, plus official dev in validation. Official test is kept apart.
4. **Graphs:** conditional probabilities and signed NPMI per language, per family and globally, from training rows only.
5. **Priors:** language to family to global shrinkage.
6. **Tokenise:** maximum length raised from 64 to 128, because the 95th percentile exceeded 64 for languages holding 47.9% of rows.

One known exception: with official dev included, Afrikaans joy differs by 3.73 points between training and validation, above the 3-point target. It was accepted, because official dev has to be in validation and cannot be stratified.

## 6. The model

**Base (B4).** XLM-RoBERTa, label-specific attention, a GCN over a statistical graph, a dynamic graph built per text, a second GCN, then a score per emotion.

**Proposed additions.**

| Addition | Idea | Reason |
|---|---|---|
| N1 | Start label queries from encoded emotion definitions | Give labels meaning across languages |
| N2 | Signed NPMI graph with a neutral node | Show that some emotions exclude each other |
| N4 | Language to family to global shrinkage of the graph | Reduce noise in small languages |
| N5 | Graph regulariser and gated fusion | Stabilise the per-text graph |
| N6 | Ordinal intensity head with distance and Pearson terms | Support Track B and the applications |

**Not built:** N3 (annotator graph, optional, never requested), and N7 and N8 (removed from the plan to fit the compute budget).

**Baselines.** B1 is TF-IDF with logistic regression. B2 is the encoder with a linear head. B4 is the LDGN baseline, with no additions.

## 7. What went wrong, and how it was handled

### 7.1 Model and training

| Problem | How it was found | Cause | Fix |
|---|---|---|---|
| Model could not overfit 64 examples (macro-F1 0.30) | Required overfit test | Three defects: label anchors nearly identical, dynamic graph saturated, attended states too small | Centred the anchors, scaled the dynamic graph, added LayerNorm. Reached 0.87 |
| A first explanation, that activations were too large, was wrong | Measurements disproved it | | Dropped it and kept investigating |
| Overfit test stayed at 0.869, below the 0.90 target | Same test | Not fully resolved | Accepted by the team. The test was not altered and still fails |
| Log of zero in intensity loss | Reading the planned code | Clamping underflows in 16-bit floats | Losses run in 32-bit floats, with a smoothed Pearson denominator |
| "At least 32 intensity rows per batch" could not be met | Batches hold only 16 rows | Plan and hardware disagreed | Quota set to half a batch |
| Learning rate would hit zero in extended epochs | Reading the schedule | Schedule ended at epoch 5 | Schedule now spans 8 epochs |
| Throughput figures looked wrong | Comparing runs | Validation time was included | Measured before validation |
| Training paused twice (118 and 25 minutes) | Time gaps in logs | Probably laptop sleep | Sleep turned off. No epochs were lost, thanks to per-epoch checkpoints |

The numeric fixes applied to every LDGN model, B4 included, so that they could not inflate the gain of the proposed model. As a result, B4 differs slightly from LDGN as published.

### 7.2 Results

| Problem | What was learned | Response |
|---|---|---|
| `final_full` scored 47.58, below B4 at 52.54 | The loss was concentrated in the 17 languages without intensity labels (38.91 against 46.41) | Traced to the quota, which gave half of each batch to 11 languages |
| Unknown which additions helped | A single combined score could not separate them | Planned the subset experiment, with a rule written first |
| N1, N4 and N5 alone lost 5 to 7 validation points | Suspected a wiring bug in single-flag runs | Checked the code and curves. No bug; they were learning more slowly |
| N2 narrowly missed the threshold (+0.71 against 0.86 noise) | Closest thing to a real gain | Dropped by the rule. The team chose not to test N2 + N6 |

### 7.3 Applications

| Problem | How it was found | Cause | Fix |
|---|---|---|---|
| Urgency set on 33.5% of English support clauses | Self-written sentence tests | Urgency and valence read the untrained disgust output for English | Both now use the language mask. Figures reported before the fix were wrong (Spearman 0.74, 32% escalated) |
| Dashboards showed mostly stress and disappointment, and fear on nearly every complaint | Team reviewed exported CSVs | English fear skew in training data | Application-only English fear threshold of 0.95, and new state rules |
| Frustration almost never fired | Diagnosis of the graph | Dynamic graph averaged 0.004 for anger and sadness, even when both were high | Dropped the pair score. An earlier idea of tuning its threshold was wrong and would not have worked |
| Outrage almost never fired | Rule used an intensity of 2.5, reached by 0.4% of clauses | Cut was too high | Lowered to 1.85 on demo data, without labels |
| Too many clauses had no aspect | Reviewing aspect counts | Keyword lists were short | Extended them. Against EduRABSA, precision fell from 0.66 to 0.63 and recall rose from 0.50 to 0.75. Even so, 81.6% of support clauses and 37.3% of feedback clauses remain untagged |

### 7.4 Process mistakes worth recording

- A comparison of `best_full` on the apps first used global thresholds, not the English ones the apps use, which gave wrong state rates. It was rerun, and the corrected figures are below.
- A table join ran out of memory, because `clause_id` is a position within a document, not a unique key. The join now uses document and clause together.
- The team was first advised to switch the apps to `best_full`, based on the 28-language average. Checking English specifically reversed that advice.
- A first version of a summary script showed 0% fear, because of a misread column. The figure was caught and corrected.

## 8. Experiments, in the order they ran

**Step 1: baseline.** B1 scored 39.76 on test.

**Step 2: full runs before subset runs.** Time was short, so the team ran `b4_full` and `final_full` first, out of the planned order.

**Step 3: quota experiment.** `final_full_noquota` removed the quota, and also ran 8 epochs where `final_full` had run 5. The gain was 4.11 points (95% CI 3.66 to 4.57), and Pearson rose from 0.545 to 0.600. The extra epochs explain an unknown part of this. It was still 0.86 points behind B4 on test.

**Step 4: subset ablation.** Original results were first backed up. The rule below was then written *before* the runs existed.

- Metric: threshold-tuned validation macro-F1 on four languages (English, Hausa, Marathi, Ukrainian), against B4 at seed 42.
- Noise: the larger of 0.5 points and the standard deviation of three B4 seeds, which came to 0.86.
- Keep what helps. Drop what hurts or is neutral.
- N6 stays unless it hurts, because the applications need intensity.

The four subset languages came from the project spec and were checked only for size. They are easier than the full task, averaging 63.8 under `b4_full` against 52.54 for all 28.

| Addition alone | Validation change | Test change | Verdict |
|---|---|---|---|
| N2 | +0.71 | +1.00 | Neutral, dropped |
| N6 | +0.38 | -0.96 | Neutral, kept by the exception |
| N5 | -5.08 | -7.50 | Hurts |
| N1 | -5.71 | -7.25 | Hurts |
| N4 | -7.14 | -9.28 | Hurts |

Test scores were computed only after the choice was frozen.

| Across three seeds | Validation | Test |
|---|---|---|
| B4 | 67.23 ± 0.86 | 64.53 ± 0.54 |
| All additions | 66.97 ± 0.81 | 63.81 ± 0.39 |
| All, no quota | 66.67 ± 1.13 | 63.62 ± 0.42 |

Other observations:

- The additions interact. N1 + N2 lost 5.62 points, then adding N4 + N5 recovered nearly all of it.
- N1, N4 and N5 were still improving at epoch 10, with final training loss 1.5 to 1.7 times B4's. "Hurts" means worse within this budget.
- B2 was 2.25 points above B4 on validation, but only 0.39 on test.

**Step 5: `best_full`.** The rule gave B4 + N6, no quota. It was trained on all 28 languages for 8 epochs. Validation came first, at 55.89 against 54.91 for B4. Test was then scored once, at 53.47.

| `best_full` against | Difference | 95% CI |
|---|---|---|
| `b4_full` | +0.92 | 0.46 to 1.39 |
| `final_full_noquota` | +1.78 | 1.28 to 2.26 |
| `final_full` | +5.89 | 5.37 to 6.37 |

`best_full` beat `b4_full` on 18 of 28 languages. By group, it scored 63.04 on the 11 languages with intensity labels and 47.28 on the 17 without.

It is not better on everything. `final_full_noquota` leads on intensity (0.600 against 0.574), on unseen languages (23.65 against 20.90) and on emotion-pair fidelity (0.291 against 0.335).

## 9. Applications

**What was built.** One shared engine with a small domain profile per application: clause splitting, emotion scoring, derived states, urgency, valence and arousal, keyword aspects, Spark aggregation, two Streamlit dashboards and an API.

**Scored data.** 49,830 support conversations (186,322 clauses) and 6,500 EduRABSA reviews (24,702 clauses).

**After the rule changes:**

| Measure | Support | Feedback |
|---|---|---|
| Fear flagged | 18.3% of clauses | 10.7% of clauses |
| Fear given anger or sadness | 34.0% | 30.3% |
| Documents escalated | 13,280 (26.7%) | 0 |

**What could be checked.** Valence against EduRABSA sentiment gives a Spearman of 0.804, and keyword aspects give precision 0.63 and recall 0.75.

**What could not.** Emotion accuracy on application text is unknown. The 300-clause annotation sheets for each app are still blank. The fear threshold and anger cut are provisional, and the 26.7% escalation rate is too high to be useful.

**Trying `best_full` on the apps.** It was tested on a 4,000-conversation sample in separate folders, on 14,516 English clauses shared with the current model.

| Measure | `final_full_noquota` | `best_full` |
|---|---|---|
| English validation macro-F1 | 68.13 | 66.81 |
| Anger intensity Pearson | 0.526 | 0.471 |
| Clauses with urgency ≥ 0.6 | 5.7% | 1.1% |
| Outrage at the current cut | 2.0% | 0% |
| Frustration | 13.4% | 13.1% |
| Stress | 12.3% | 7.0% |
| No emotion | 24.8% | 30.9% |

`best_full` runs correctly on the apps but reads anger as weaker, which weakens escalation. On twelve hand-written sentences, both models matched the intended emotions exactly on six. The apps stayed on `final_full_noquota`. Switching would need a new outrage cut and urgency threshold.

**Takeaway.** The best benchmark model is not automatically the best product model.

## 10. Why the final choices were made

| Choice | Reason |
|---|---|
| Keep N6, drop the rest | The rule fixed in advance, applied on validation |
| No quota | It took training time from 17 of 28 languages |
| `best_full` as the research model | Best Track A score on validation and test |
| `final_full_noquota` in the apps | Better English intensity, which urgency and outrage depend on |
| Stop after `best_full` | The team chose not to spend time on N2 + N6 |
| Local training, not Colab | The team's decision; full runs fitted on 4 GB with automatic memory fallback available |

## 11. Limitations to keep in mind

- Each full-data model has one seed. B4 alone moved by 0.86 between subset seeds.
- The comparison is not fully blind. `final_full`'s test score was seen before the quota experiment, and earlier test scores were known when the selection rule was written. Selection itself used validation only.
- The subset is easier than the full task and has only one language without intensity labels.
- Single additions were tested on one seed, within limited epochs.
- N2 + N6 was never tested, so a better combination may exist.
- The overfit test still fails its 0.90 target.
- Scores are local and were not submitted to the leaderboard. Track C is zero-shot, which differs from some official setups.
- Amharic and Russian Track B scores are not on the official row sets.
- Application thresholds were set without labels.

## 12. Where things stand

| Item | Status |
|---|---|
| Original results | Preserved, with backups and git tags |
| Subset ablation | Finished; 17 runs |
| `best_full` | Finished and scored on test |
| Applications | Working, on `final_full_noquota` |
| Slides | Written in LaTeX, not yet compiled |
| Report | Draft in ACL format, not yet compiled; the team is rewriting it |
| Notebook | Built, with no outputs; not yet run |
| Hand labels for the apps | Not done |
| Video and contribution document | Not done |

## 13. Reproducing the work

Run these from the repository root, in this order:

```bash
pip install -r requirements.txt
python scripts/check_env.py
python scripts/download_data.py

python spark_jobs/01_ingest.py
python spark_jobs/02_clean.py
python spark_jobs/03_split.py
python spark_jobs/04_graphs.py
python spark_jobs/06_priors.py
python spark_jobs/07_tokenize.py
python spark_jobs/08_baseline_tfidf.py

python -m emoldgn.train --exp b4_full
python -m emoldgn.predict --exp b4_full
python -m emoldgn.train --exp best_full
python -m emoldgn.predict --exp best_full
```

Alternatively, open `EmoLDGN_BRIGHTER.ipynb`, which runs the data stages, the EDA, and both models in separate folders. Plan for about 2 hours on a 4 GB GPU.

To rescore the applications, export a model and run the two application jobs:

```bash
python scripts/export_artifacts.py --from-run final_full_noquota
python spark_jobs/09_score.py --profile customer_support
python spark_jobs/10_aggregate.py --profile customer_support
python spark_jobs/09_score.py --profile student_feedback
python spark_jobs/10_aggregate.py --profile student_feedback
streamlit run apps/dashboards/support.py
streamlit run apps/dashboards/feedback.py
```

Exporting replaces the files in `artifacts/`, so back them up first.

Do not run `run_experiments.py --evaluate` or `evaluate --results`, because both overwrite `reports/results.md`. GPU training is not perfectly repeatable, so rerun scores can differ slightly.

## 14. Where to find each result

| What | File |
|---|---|
| Dataset facts and findings | `reports/DATA_NOTES.md` |
| First full-data results | `reports/results.md` |
| Quota comparison | `reports/results_noquota.md` |
| Addition-by-addition selection | `reports/ablation_val.md` |
| Subset scores on test | `reports/ablation_test.md` |
| `best_full` compared with the others | `reports/results_best.md` |
| Application and dataset figures | `reports/app_numbers.md` |
| `best_full` on the apps | `reports/_check_best/` |
| Every decision, dated | `docs/DECISIONS.md` |
| Original design | `docs/EmoLDGN_plus_Final_Implementation.md` |
