#!/usr/bin/env python
"""Build the Google Drive bundle (the "fast path" of the README).

Writes zip parts and a manifest.json into one folder. Every path inside a zip is relative to the repository
root, so extraction puts each file exactly where the code expects it (see scripts/restore_from_drive.py).

Parts
-----
part1_app_tables       apps/bronze, apps/silver, apps/gold              the stored results shown by both dashboards
part2_app_model        artifacts/best.pt and two JSON files             the exported model behind the upload tab
part3_run_results      small files of every finished run                 predictions, metrics, thresholds, logs
part4_notebook_weights runs/b4_full and runs/best_full ckpt_best.pt      lets the notebook re-score without training
part5_demo_inputs      the two demo datasets as Parquet                  lets the app tables be rebuilt without downloads

Checkpoints of the 17 subset runs and of final_full are deliberately left out: they are about 1.1 GB each
and nothing in the project reads them. The subset tables are computed from the small files in part 3.

    python scripts/package_for_drive.py --dry-run      # sizes only, nothing is written
    python scripts/package_for_drive.py                # all parts
    python scripts/package_for_drive.py --parts part1_app_tables part3_run_results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUN_FILES = ("config.yaml", "metrics.jsonl", "status.json", "thresholds.json", "train.log")
BUNDLE_VERSION = 1


def finished_runs() -> list[Path]:
    """Run folders that completed: runs/<id> with status.json, excluding backups and notebook scratch folders."""
    out = []
    for d in sorted((ROOT / "runs").iterdir()):
        if not d.is_dir() or d.name.startswith(("_", "nb_")):
            continue
        status = d / "status.json"
        if status.exists() and json.loads(status.read_text()).get("status") == "done":
            out.append(d)
    return out


def tree(rel: str) -> list[Path]:
    base = ROOT / rel
    return sorted(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and not p.name.endswith("Zone.Identifier"))


def part_files() -> dict[str, dict]:
    runs = finished_runs()
    small: list[Path] = []
    for d in runs:
        small += [d / f for f in RUN_FILES if (d / f).exists()]
        small += sorted((d / "predictions").glob("*.npz")) if (d / "predictions").is_dir() else []
    return {
        "part1_app_tables": {
            "description": "Stored app results: apps/bronze, apps/silver, apps/gold",
            "files": tree("apps/bronze") + tree("apps/silver") + tree("apps/gold"),
            "check": ["apps/gold/customer_support/documents", "apps/gold/student_feedback/documents"],
            "required_for": "the first tab of both dashboards",
        },
        "part2_app_model": {
            "description": "Exported model of the applications (final_full_noquota)",
            "files": [ROOT / "artifacts" / n for n in ("best.pt", "thresholds.json", "lang_index.json")],
            "check": ["artifacts/best.pt", "artifacts/thresholds.json", "artifacts/lang_index.json"],
            "required_for": "the upload tab of both dashboards and the API",
        },
        "part3_run_results": {
            "description": f"Predictions, metrics, thresholds and logs of {len(runs)} finished runs (no weights)",
            "files": small,
            "check": ["runs/best_full/predictions/test.npz", "runs/b4_subset_s42/predictions/val.npz"],
            "required_for": "the subset tables (ablation_report.py, ablation_test_report.py) and the notebook's re-scoring route",
        },
        "part4_notebook_weights": {
            "description": "Best-epoch weights of b4_full and best_full",
            "files": [ROOT / "runs" / r / "ckpt_best.pt" for r in ("b4_full", "best_full")],
            "check": ["runs/b4_full/ckpt_best.pt", "runs/best_full/ckpt_best.pt"],
            "required_for": "re-scoring in the notebook instead of training (needs part 3 as well)",
        },
        "part5_demo_inputs": {
            "description": "Demo datasets as Parquet: customer support conversations and EduRABSA reviews",
            "files": [ROOT / "data/demo/customer_support/train.parquet", ROOT / "data/demo/edurabsa_hf/train.parquet",
                      ROOT / "data/demo/edurabsa_hf/test.parquet"],
            "check": ["data/demo/customer_support/train.parquet", "data/demo/edurabsa_hf/train.parquet"],
            "required_for": "rebuilding the app tables with 09_score.py and 10_aggregate.py without downloading",
        },
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return ""


def build(name: str, spec: dict, out: Path) -> dict:
    files = spec["files"]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"{name}: missing files, for example {missing[0]}")
    target = out / f"{name}.zip"
    tmp = target.with_suffix(".zip.part")
    members = []
    t0 = time.time()
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for f in files:
            rel = f.relative_to(ROOT).as_posix()
            # Weights, Parquet and npz are already compressed, so storing them is much faster.
            stored = f.suffix in (".pt", ".parquet", ".npz", ".zip")
            z.write(f, rel, compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED, compresslevel=None if stored else 6)
            members.append({"path": rel, "size": f.stat().st_size})
    tmp.replace(target)
    return {"file": target.name, "size": target.stat().st_size, "sha256": sha256(target), "members": members,
            "description": spec["description"], "check": spec["check"], "required_for": spec["required_for"],
            "seconds": round(time.time() - t0, 1)}


DRIVE_README = """EmoLDGN+ : files for the Google Drive folder
=============================================

Put ALL files of this folder in one Google Drive folder, including manifest.json.
Share the folder only with the people who need it (instructor and graders).

How to use (details are in README.md of the repository, section "Path B"):

 1. Clone the repository and install the requirements.
 2. Download this folder (or only the parts you need) into the repository, for example into  drive_downloads/
 3. Run:   python scripts/restore_from_drive.py --source drive_downloads
 4. Start an app:  streamlit run apps/dashboards/support.py

Parts
-----
{parts}

manifest.json lists every file with its size and checksum. restore_from_drive.py verifies the checksums.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="drive_bundle", help="output folder (default drive_bundle/, git-ignored)")
    ap.add_argument("--parts", nargs="*", help="only these parts")
    ap.add_argument("--dry-run", action="store_true", help="list parts and sizes, write nothing")
    args = ap.parse_args()

    specs = part_files()
    chosen = args.parts or list(specs)
    unknown = [p for p in chosen if p not in specs]
    if unknown:
        sys.exit(f"unknown parts: {unknown}. Known: {list(specs)}")

    print(f"{'part':<26}{'files':>7}{'raw size':>12}   description")
    for name in chosen:
        files = specs[name]["files"]
        size = sum(f.stat().st_size for f in files if f.exists())
        absent = sum(1 for f in files if not f.exists())
        print(f"{name:<26}{len(files):>7}{human(size):>12}   {specs[name]['description']}" + (f"   [{absent} MISSING]" if absent else ""))
    if args.dry_run:
        return

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"version": BUNDLE_VERSION, "parts": {}}
    manifest["version"], manifest["created"] = BUNDLE_VERSION, datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name in chosen:
        print(f"building {name} ...", flush=True)
        manifest["parts"][name] = build(name, specs[name], out)
        p = manifest["parts"][name]
        print(f"   {p['file']}: {human(p['size'])}, {len(p['members'])} files, {p['seconds']} s")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [f"  {v['file']:<28}{human(v['size']):>10}   {v['description']}\n  {'':<38}needed for: {v['required_for']}"
             for v in manifest["parts"].values()]
    (out / "DRIVE_README.txt").write_text(DRIVE_README.format(parts="\n".join(lines)), encoding="utf-8")
    total = sum(v["size"] for v in manifest["parts"].values())
    print(f"\nwrote {out}  ({human(total)} in {len(manifest['parts'])} parts). Upload everything in that folder to Google Drive.")


if __name__ == "__main__":
    main()
