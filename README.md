# EmoLDGN+

Multi-label emotion detection on BRIGHTER (SemEval-2025 Task 11) with a label-specific dual graph network (LDGN) on an XLM-RoBERTa encoder, five proposed additions (N1, N2, N4, N5, N6), and two applications on one shared engine: **Customer Support Analysis** and **Student Course Reviews**.

BRIGHTER is research-only. Do not redistribute it or anything quoting it. The applications describe how a text reads, not how a person feels, and should support human judgement rather than replace it.

## Results in brief

| Model | Contents | Track A test macro-F1 | Track B Pearson | Track C macro-F1 |
|---|---|---|---|---|
| B1 | TF-IDF, logistic regression | 39.76 | - | - |
| B4 (`b4_full`) | LDGN, no additions | 52.54 | - | - |
| `final_full` | N1, N2, N4, N5, N6, with intensity quota | 47.58 | 0.545 | 21.72 |
| `final_full_noquota` | N1, N2, N4, N5, N6, no quota (used by the apps) | 51.68 | 0.600 | 23.65 |
| **`best_full`** | **B4 + N6, no quota (selected)** | **53.47** | 0.574 | 20.90 |

Scores are local, computed with the official metrics, and were not submitted to the leaderboard. The history of the project, including what failed, is in `docs/PROJECT_JOURNEY.md`, and every decision is recorded in `docs/DECISIONS.md`.

## Two ways to run the project

| | Path A: rerun everything | Path B: use the Google Drive bundle |
|---|---|---|
| What it does | Downloads the data, rebuilds every table, trains every model | Downloads ready-made results and model files, then places them with one script |
| Needs a GPU | Yes, for training | Only for the upload tab of the apps and for retraining |
| Time | About 6 to 7 hours in total (see the table in Path A) | About 15 to 40 minutes, mostly downloads |
| Use it when | Reproducing the experiments from scratch | Looking at the apps, the notebook or the subset tables quickly |
| Section | [Path A](#path-a-rerun-everything) | [Path B](#path-b-use-the-google-drive-bundle) |

Both paths start with the same requirements.

## Requirements

- Python 3.10 or newer (3.12 was used), Java 8, 11 or 17 (11 was used), and about 10 GB of free disk space for Path A (about 5 GB for Path B).
- A CUDA GPU with at least 4 GB of memory for training (an RTX 3050 laptop GPU was used). Windows users should work in WSL2.
- An internet connection for the downloads, and for the XLM-RoBERTa model (about 1.1 GB), which Hugging Face fetches the first time it is needed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python scripts/check_env.py            # ends with ALL CHECKS PASSED
```

If the XLM-RoBERTa download stalls, set `HF_HUB_DISABLE_XET=1` and run it again.

---

## Path A: rerun everything

Every Spark job skips itself when its output already exists, and accepts `--force` to recompute. Training runs resume from their last finished epoch. Long runs should be started in the background, for example `nohup <command> > runs/<name>.log 2>&1 &`.

Approximate times (RTX 3050, 4 GB):

| Step | What | Time |
|---|---|---|
| 1 | Environment | 5 to 10 min |
| 2 | Downloads | 5 to 20 min, depending on the connection |
| 3 | Data tables (Spark jobs 01 to 07) | about 2 min |
| 4 | TF-IDF baseline | about 2 min |
| 5 | Four full-data models | about 3 hours |
| 6 | Seventeen subset runs | about 2.5 hours |
| 7 | Result tables | a few minutes |
| 8 | Export the model and build the app tables | about 20 to 30 min (estimate) |
| 9 | Apps, notebook, tests | minutes |

### Step 1. Environment

See [Requirements](#requirements).

### Step 2. Download the data

```bash
python scripts/download_data.py                 # everything, skipping what exists
# or only parts of it:
python scripts/download_data.py --only semeval hf_categories hf_intensities     # BRIGHTER
python scripts/download_data.py --only customer_support edurabsa edurabsa_hf    # demo data for the apps
python scripts/download_data.py --only nrc_vad lid176                           # lexicon and language ID
```

| Item | Folder | Size | Used for |
|---|---|---|---|
| SemEval-2025 Task 11 repository, Hugging Face copies of BRIGHTER | `data/raw/brighter/` | about 175 MB | training and testing (the SemEval files are the source of truth) |
| Customer support conversations | `data/demo/customer_support/` | about 200 MB | input of the support app (never trained on) |
| EduRABSA reviews | `data/demo/edurabsa_hf/` | about 3 MB | input of the feedback app (never trained on) |
| NRC VAD lexicon | `data/lexicons/` | about 20 MB | valence and arousal in the apps (optional) |
| fastText language ID | `models/lid.176.bin` | about 130 MB | language detection in the apps (optional) |

The optional Coursera reviews are skipped unless a Kaggle token exists, and the script reports that as `FAILED` without stopping. Without the lexicon, valence and arousal show as zero. Without the language-ID model, every text is treated as an unknown language.

### Step 3. Build the data tables

```bash
python spark_jobs/01_ingest.py      # raw files to data/bronze
python spark_jobs/02_clean.py       # emoji, emoticons, links and handles; data/silver
python spark_jobs/03_split.py       # stratified validation hold-out; data/gold/parts
python spark_jobs/04_graphs.py      # co-occurrence and NPMI graphs, training rows only
python spark_jobs/06_priors.py      # language, family and global shrinkage; graph bank
python spark_jobs/07_tokenize.py    # token lengths and tokenised tables
python scripts/make_data_notes.py   # reports/DATA_NOTES.md (counts only, no text)
```

### Step 4. TF-IDF baseline (B1)

```bash
python spark_jobs/08_baseline_tfidf.py          # reports/results_b1.csv
```

### Step 5. Train and score the four full-data models

Each model is trained, then scored. Scoring tunes the thresholds on validation, scores the test split once, and (for models with the intensity head) also scores Track C.

```bash
for EXP in b4_full final_full final_full_noquota best_full; do
  python -m emoldgn.train   --exp $EXP        # runs/$EXP/ with weights, metrics and logs
  python -m emoldgn.predict --exp $EXP        # predictions and thresholds
done
```

| Run | Contents | Time |
|---|---|---|
| `b4_full` | LDGN, no additions | about 47 min (8 epochs) |
| `final_full` | all five additions, with the intensity quota | about 32 min (5 epochs) |
| `final_full_noquota` | all five additions, no quota | about 50 min (8 epochs) |
| `best_full` | B4 + N6, no quota | about 48 min (8 epochs) |

If training stops (sleep, crash), run the same command with `--resume`. On a CUDA out-of-memory error the batch size is halved automatically and the run restarts. GPU training is not perfectly repeatable, so scores may differ from the reported ones in the last decimals.

### Step 6. The seventeen subset runs

The subset uses four languages (English, Hausa, Marathi, Ukrainian) and answers which addition helps. Selection used validation only; test scoring comes last.

```bash
IDS="b2_subset b4_subset_s42 s1_subset s2_subset final_subset_s42 b4_subset_s1 b4_subset_s2 \
     final_subset_s1 final_subset_s2 nq_subset_s42 nq_subset_s1 nq_subset_s2 \
     a_n1_subset a_n2_subset a_n4_subset a_n5_subset a_n6_subset"

python scripts/run_experiments.py --only $IDS                          # about 2 hours
for E in $IDS; do python -m emoldgn.predict --exp $E --val-only; done  # validation only, a few minutes
python scripts/ablation_report.py                                       # reports/ablation_val.md: selection
# only after the choice is frozen:
for E in $IDS; do python -m emoldgn.predict --exp $E; done              # scores the test split
python scripts/ablation_test_report.py                                  # reports/ablation_test.md
```

### Step 7. Result tables

```bash
python scripts/compare_full.py       # reports/results_noquota.md
python scripts/compare_best.py       # reports/results_best.md
python scripts/app_numbers.py        # reports/app_numbers.md (after step 8)
```

Do not run `python scripts/run_experiments.py --evaluate` or `python -m emoldgn.evaluate --results` on a finished project: both rewrite `reports/results.md`, which holds the first full-data results.

### Step 8. Export the model and build the app tables

```bash
python scripts/export_artifacts.py --from-run final_full_noquota   # artifacts/best.pt, thresholds.json, lang_index.json
python spark_jobs/09_score.py     --profile customer_support       # 50,000 conversations by default; --limit 5000 for a quick run
python spark_jobs/10_aggregate.py --profile customer_support
python spark_jobs/09_score.py     --profile student_feedback
python spark_jobs/10_aggregate.py --profile student_feedback
python scripts/check_app_provenance.py                             # confirms the tables come from the exported model
```

Exporting replaces the files in `artifacts/`, so copy them first if the old ones are still needed. `--force` is required to recompute scored tables that already exist.

### Step 9. Run the apps, the notebook and the tests

```bash
streamlit run apps/dashboards/support.py --server.port 8501
streamlit run apps/dashboards/feedback.py --server.port 8502
uvicorn apps.api.server:app --port 8000
jupyter notebook EmoLDGN_BRIGHTER.ipynb
pytest -q -m "not slow"
```

The tests include one slow GPU test (`test_overfit.py`), which is expected to reach a training macro-F1 above 0.9 but reached 0.869 in this project; the discrepancy is recorded in `docs/DECISIONS.md`.

---

## Path B: use the Google Drive bundle

The Drive folder holds ready-made results and model files, so training and most rebuilding can be skipped. A single script places every file where the code expects it. Nothing needs to be moved by hand.

### What is in the Drive folder

All files are zip archives plus one manifest, in a single Drive folder. Paths inside each zip are relative to the repository root.

| File | Size | Contents | Needed for |
|---|---|---|---|
| `manifest.json` | tiny | the list of parts, with sizes and SHA-256 checksums | always, since the restore script reads it |
| `DRIVE_README.txt` | tiny | a short guide | reading |
| `part1_app_tables.zip` | about 80 MB | `apps/bronze`, `apps/silver`, `apps/gold` | first tab of both dashboards |
| `part2_app_model.zip` | about 1.1 GB | `artifacts/best.pt`, `thresholds.json`, `lang_index.json` | upload tab of both dashboards, and the API |
| `part3_run_results.zip` | about 17 MB | predictions, metrics, thresholds and logs of the 21 finished runs (no weights) | subset tables, notebook re-scoring |
| `part4_notebook_weights.zip` | about 2.1 GB | `runs/b4_full/ckpt_best.pt`, `runs/best_full/ckpt_best.pt` | notebook re-scoring without training (needs part 3) |
| `part5_demo_inputs.zip` | about 200 MB | the two demo datasets as Parquet | rebuilding the app tables without the downloads |

Which parts to download for a given goal:

| Goal | Parts |
|---|---|
| See the dashboards | 1 |
| Also use the upload tab | 1 and 2 |
| Regenerate the subset tables | 3 |
| Run the notebook without training | 3 and 4 (plus the raw data download, below) |
| Rebuild the app tables yourself | 2 and 5 |
| Everything | 1 to 5 (about 3.5 GB) |

### Steps

**1. Clone the repository and prepare the environment** as in [Requirements](#requirements). Create a folder inside the repository for the downloads:

```bash
mkdir -p drive_downloads
```

**2. Download the parts** from the Drive link into `drive_downloads/`, together with `manifest.json`. Downloading the whole folder from the browser is the simplest route. For large files, Drive may show a virus-scan page. In that case download from the browser, not from the command line.

**3. Restore the files:**

```bash
python scripts/restore_from_drive.py --source drive_downloads
```

The script verifies each zip against the checksum in the manifest, refuses unsafe paths, and extracts the files to `apps/`, `artifacts/`, `runs/` and `data/`. Existing files with the same size are left alone, and files that differ are only replaced with `--overwrite`. Useful options:

```bash
python scripts/restore_from_drive.py --list                                        # parts described in the manifest
python scripts/restore_from_drive.py --source drive_downloads --dry-run            # report only, writes nothing
python scripts/restore_from_drive.py --source drive_downloads --parts part1_app_tables part2_app_model
```

It finishes by listing what is still missing and the commands to use next.

**4. Download the small files that are not on the Drive** (they have their own licences or are public):

```bash
python scripts/download_data.py --only nrc_vad lid176                              # lexicon and language ID
python scripts/download_data.py --only semeval hf_categories hf_intensities       # raw BRIGHTER, needed by the notebook
```

**5. Use the project:**

```bash
streamlit run apps/dashboards/support.py            # needs part 1 (and part 2 for the upload tab)
streamlit run apps/dashboards/feedback.py
python scripts/ablation_report.py                   # needs part 3: regenerates reports/ablation_val.md
python scripts/ablation_test_report.py              # reports/ablation_test.md
jupyter notebook EmoLDGN_BRIGHTER.ipynb             # needs parts 3 and 4, plus step 4
```

The regenerated subset tables are identical to the committed ones in `reports/`.

### What the notebook does with the restored files

For each of the two models, the notebook looks for `runs/<name>/ckpt_best.pt`. If it exists and `FORCE_RETRAIN` is `False`, the saved weights are re-scored, which takes a few minutes. Otherwise the model is trained from scratch (about 50 minutes each). It never writes into the original run folders: everything goes to `runs/nb_*` and `data/notebook`. The data tables the notebook needs are rebuilt by its own Spark cells in about two minutes.

### Things to know about restored runs

- Restored runs are marked `done`, so `run_experiments.py` skips them. To retrain one, delete its folder in `runs/` first.
- The subset runs and `final_full`, `final_full_noquota` have no weights in the bundle. `python -m emoldgn.predict` cannot score them again, and `check_app_provenance.py` needs `runs/final_full_noquota/ckpt_best.pt`, which is not included. The app model is in `artifacts/best.pt`.
- Only `best_full` and `b4_full` weights are included, because those are the models the notebook scores.

### For the project owner: creating the Drive bundle

```bash
python scripts/package_for_drive.py --dry-run       # sizes only, writes nothing
python scripts/package_for_drive.py                 # writes drive_bundle/ (about 3.5 GB, a few minutes)
```

Upload **everything in `drive_bundle/`** to one Google Drive folder, keep the file names unchanged, and share the folder only with the instructor and graders. The bundle contains customer support text with no stated licence, EduRABSA text (CC BY-NC-SA 4.0), prediction files that include the BRIGHTER test labels, and weights trained on BRIGHTER, so the link should not be public. The NRC lexicon is not included because its licence forbids redistribution.

---

## Applications

Two Streamlit dashboards share one engine and one look. The first tab shows the stored results on the test data (the demo datasets scored by the EmoLDGN+ model, which never trained on them). The second tab scores an uploaded CSV with the exported model. See `apps/README.md` for details.

```bash
uvicorn apps.api.server:app --port 8000
curl -X POST localhost:8000/analyze -H 'Content-Type: application/json' \
  -d '{"text": "I enjoyed the course, but the examination was stressful and confusing", "profile": "student_feedback"}'
```

Application-level thresholds (an English fear threshold of 0.95 and an anger-intensity cut of 1.85) are provisional and were set without hand labels. Emotion accuracy on application text has not been measured. To measure it, 300-clause sheets have to be created in `eval/app_annotation/` and labelled by hand (never with model output). The sheets hold text from the demo datasets, so they are not stored in this repository:

```bash
python scripts/make_annotation_templates.py         # needs the app tables
python eval/app_eval.py --dry-run                   # lists what still needs labels
python eval/app_eval.py --make-review-sheets
python eval/app_eval.py --latency-url http://127.0.0.1:8000/analyze --scaling 1 2 4
```

## Troubleshooting

| Symptom | Likely cause and remedy |
|---|---|
| A Spark job prints `exists, skipping` | Its output exists. Use `--force` to rebuild it. |
| `JAVA_HOME` or Spark startup errors | Java 8, 11 or 17 is required and must be on the path. |
| CUDA out of memory | Close other GPU programs. Training halves the batch size and restarts on its own. |
| Training stopped part-way | Run the same command with `--resume`. |
| The XLM-RoBERTa download stalls | `HF_HUB_DISABLE_XET=1`, then run it again. |
| The dashboards fail with `No module named 'plotly'` | `pip install -r requirements.txt`. |
| The upload tab says the model is not available | `artifacts/best.pt` is missing: restore part 2, or run step 8. |
| Notebook tables look empty in VS Code | The tables are Markdown. Reload the notebook, and make sure the notebook is trusted. |
| `restore_from_drive.py` reports a checksum mismatch | The download is incomplete or corrupted. Download that part again. |
| Dashboards show old numbers after a rebuild | Restart Streamlit. |

## Data, licences and reproducibility

- **BRIGHTER** (SemEval-2025 Task 11) is the only training data, and no BRIGHTER text appears in the repository or in the notebook outputs. The raw files are downloaded, not committed.
- The **customer support** conversations have no stated licence and are used only as application input. **EduRABSA** is CC BY-NC-SA 4.0. The **NRC VAD lexicon** is for non-commercial research and may not be redistributed.
- Test data was used once per final model and never for tuning. Selection between models used validation data only, with one disclosed exception: the earlier full-data test scores were known when the selection rule was written (see `docs/DECISIONS.md`).
- Seeds are set everywhere, but GPU training is not bit-for-bit repeatable.

## Layout

`spark_jobs/` data processing, `emoldgn/` model and training, `apps/` engine, dashboards and API, `config/` and `profiles/` settings, `eval/` application evaluation, `scripts/` setup, experiments, reports and the Drive tools, `tests/` unit tests, `docs/` design, decisions and project history, `reports/` results, and `EmoLDGN_BRIGHTER.ipynb` the notebook.
