# Google Drive upload guide (for the project owner)

The Drive folder lets someone skip training and most rebuilding (README, Path B). It must contain the files below, in **one** folder, with their names unchanged.

## 1. Build the bundle
```bash
python scripts/package_for_drive.py --dry-run     # sizes only
python scripts/package_for_drive.py               # writes drive_bundle/ (about 3.5 GB, a few minutes)
```
Check that `drive_bundle/` contains exactly these seven files.

## 2. What goes into the Drive folder
Create one folder, for example `EmoLDGN_drive`, and upload **all** of these to it:

| File | Format | Size | Contents |
|---|---|---|---|
| `manifest.json` | JSON | tiny | parts, sizes and SHA-256 checksums; the restore script cannot run without it |
| `DRIVE_README.txt` | text | tiny | short guide for people who open the folder |
| `part1_app_tables.zip` | zip | about 80 MB | `apps/bronze`, `apps/silver`, `apps/gold` (the dashboards' stored results) |
| `part2_app_model.zip` | zip | about 1.1 GB | `artifacts/best.pt`, `thresholds.json`, `lang_index.json` |
| `part3_run_results.zip` | zip | about 17 MB | predictions, metrics, thresholds and logs of 21 runs, with no weights |
| `part4_notebook_weights.zip` | zip | about 2.1 GB | `runs/b4_full/ckpt_best.pt` and `runs/best_full/ckpt_best.pt` |
| `part5_demo_inputs.zip` | zip | about 200 MB | `data/demo/customer_support/train.parquet`, `data/demo/edurabsa_hf/{train,test}.parquet` |

Rules:
- Upload the zips as they are. Do not unzip, rename, re-compress or add files, or the checksums fail.
- The demo datasets are Parquet, which is the format the Spark jobs read. They are not converted to CSV.
- Do not upload `data/raw`, `data/bronze|silver|gold`, the NRC lexicon, `models/lid.176.bin`, any subset run weights, or `final_full` weights. They are rebuilt or downloaded by scripts, are not needed, or may not be redistributed.

## 3. Sharing
- Set the folder to **Restricted** and add only the instructor and graders by e-mail. Avoid "anyone with the link".
- The folder contains customer support text with no stated licence, EduRABSA text (CC BY-NC-SA 4.0), prediction files with the BRIGHTER test labels, and weights trained on BRIGHTER. BRIGHTER is treated as research-only in this project.
- Keep the folder available until grading ends.
- Put the folder link in the README or the submission text.

## 4. Test before sending
In a fresh copy of the repository, download the folder into `drive_downloads/`, then:
```bash
python scripts/restore_from_drive.py --source drive_downloads --dry-run
python scripts/restore_from_drive.py --source drive_downloads
streamlit run apps/dashboards/support.py
```
