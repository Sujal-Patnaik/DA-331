#!/usr/bin/env python
"""Place the files downloaded from the Google Drive folder into the repository (the "fast path" of the README).

The Drive folder holds zip parts and a manifest.json written by scripts/package_for_drive.py. Every path
inside a zip is relative to the repository root, so extraction puts each file where the code expects it:

    apps/{bronze,silver,gold}/...     stored app results
    artifacts/{best.pt,*.json}        exported model of the apps
    runs/<id>/...                     predictions, metrics, thresholds (and two sets of weights)
    data/demo/...                     the two demo datasets

Each zip is checked against the checksum in the manifest before anything is extracted. Existing files are never
overwritten unless --overwrite is given. Nothing outside apps/, artifacts/, runs/ and data/ can be written.

    python scripts/restore_from_drive.py --source drive_downloads
    python scripts/restore_from_drive.py --source drive_downloads --parts part1_app_tables part2_app_model
    python scripts/restore_from_drive.py --source drive_downloads --dry-run
    python scripts/restore_from_drive.py --list
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP = {"apps", "artifacts", "runs", "data"}
SEARCH = ("drive_downloads", "drive_bundle", "~/Downloads", ".")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def find_source(given: str | None) -> Path | None:
    candidates = [given] if given else list(SEARCH)
    for c in candidates:
        p = Path(c).expanduser()
        p = p if p.is_absolute() else Path.cwd() / p
        if (p / "manifest.json").exists():
            return p
    return None


def safe_target(root: Path, name: str) -> Path | None:
    """Destination of an archive member, or None if it could escape the allowed folders."""
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] not in ALLOWED_TOP:
        return None
    target = (root / Path(*pure.parts)).resolve()
    return target if root.resolve() in target.parents else None


def restore_part(name: str, info: dict, source: Path, root: Path, overwrite: bool, dry: bool, skip_verify: bool) -> dict:
    zpath = source / info["file"]
    res = {"status": "", "written": 0, "unchanged": 0, "conflicts": 0, "bytes": 0}
    if not zpath.exists():
        res["status"] = "not downloaded"
        return res
    if zpath.stat().st_size != info["size"]:
        res["status"] = f"wrong size ({human(zpath.stat().st_size)} instead of {human(info['size'])}): download again"
        return res
    if not skip_verify:
        print(f"  verifying {info['file']} ...", flush=True)
        if sha256(zpath) != info["sha256"]:
            res["status"] = "CHECKSUM MISMATCH: download again"
            return res
    try:
        z = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile:
        res["status"] = "not a valid zip file: download again"
        return res
    with z:
        unsafe = [m.filename for m in z.infolist() if not m.is_dir() and safe_target(root, m.filename) is None]
        if unsafe:                                  # checked before anything is written
            res["status"] = f"refused: unsafe path {unsafe[0]!r}"
            return res
        for member in z.infolist():
            if member.is_dir():
                continue
            target = safe_target(root, member.filename)
            if target.exists():
                if target.stat().st_size == member.file_size:
                    res["unchanged"] += 1
                    continue
                if not overwrite:
                    res["conflicts"] += 1
                    continue
            res["written"] += 1
            res["bytes"] += member.file_size
            if dry:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with z.open(member) as src, open(tmp, "wb") as dst:
                while block := src.read(1 << 22):
                    dst.write(block)
            os.replace(tmp, target)
    res["status"] = "would restore" if dry else "restored"
    if res["conflicts"]:
        res["status"] += f" ({res['conflicts']} existing files differ: use --overwrite to replace them)"
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="folder with the downloaded zips and manifest.json (default: search drive_downloads, ~/Downloads)")
    ap.add_argument("--root", default=str(ROOT), help="repository root (default: this repository)")
    ap.add_argument("--parts", nargs="*", help="restore only these parts (default: every part that was downloaded)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing files whose size differs")
    ap.add_argument("--dry-run", action="store_true", help="verify and report, write nothing")
    ap.add_argument("--skip-verify", action="store_true", help="skip the sha256 check (not recommended)")
    ap.add_argument("--list", action="store_true", help="show the parts described in the manifest and exit")
    args = ap.parse_args()

    source = find_source(args.source)
    if source is None:
        sys.exit("manifest.json not found. Download the whole Drive folder (including manifest.json) and pass it with "
                 "--source, for example:  python scripts/restore_from_drive.py --source drive_downloads")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    parts = manifest["parts"]
    root = Path(args.root).resolve()

    if args.list:
        for n, p in parts.items():
            have = "downloaded" if (source / p["file"]).exists() else "missing"
            print(f"{n:<26}{human(p['size']):>10}  {have:<11} {p['description']}\n{'':<37}needed for: {p['required_for']}")
        return

    chosen = args.parts or list(parts)
    unknown = [c for c in chosen if c not in parts]
    if unknown:
        sys.exit(f"unknown parts {unknown}. Known: {list(parts)}")

    print(f"source: {source}\ntarget: {root}\n")
    results = {}
    for n in chosen:
        print(f"{n}  ({parts[n]['description']})")
        results[n] = restore_part(n, parts[n], source, root, args.overwrite, args.dry_run, args.skip_verify)
        r = results[n]
        print(f"  -> {r['status']}" + (f": {r['written']} files, {human(r['bytes'])} written, {r['unchanged']} already in place"
                                        if r["written"] or r["unchanged"] else ""))

    print("\nSummary")
    ok = []
    for n, r in results.items():
        done = r["status"].startswith(("restored", "would restore")) or (r["unchanged"] and not r["written"])
        missing = [c for c in parts[n]["check"] if not (root / c).exists()] if not args.dry_run else []
        flag = "ok" if done and not missing else ("incomplete" if done else "NOT RESTORED")
        if flag == "ok":
            ok.append(n)
        print(f"  {n:<26}{flag:<14}{'' if not missing else 'still missing: ' + ', '.join(missing)}")

    if args.dry_run:
        print("\nDry run only: nothing was written.")
        return

    print("\nChecks for files that are downloaded separately:")
    for rel, how in (("data/lexicons", "python scripts/download_data.py --only nrc_vad"),
                     ("models/lid.176.bin", "python scripts/download_data.py --only lid176")):
        have = (root / rel).exists() and (any((root / rel).rglob("*")) if (root / rel).is_dir() else True)
        print(f"  {rel:<22}{'present' if have else 'missing -> ' + how}")

    print("\nNext steps")
    if "part1_app_tables" in ok:
        print("  apps:            streamlit run apps/dashboards/support.py   (and feedback.py)")
    if "part2_app_model" in ok:
        print("                   the upload tab can now score files (a GPU is recommended)")
    if "part3_run_results" in ok:
        print("  subset tables:   python scripts/ablation_report.py ; python scripts/ablation_test_report.py")
    if "part3_run_results" in ok and "part4_notebook_weights" in ok:
        print("  notebook:        jupyter notebook EmoLDGN_BRIGHTER.ipynb   (re-scores the saved weights, no training)")
    if "part5_demo_inputs" in ok:
        print("  rebuild tables:  python spark_jobs/09_score.py --profile student_feedback  (see README, Path A, step 8)")
    print("  The notebook also needs the raw BRIGHTER files:  python scripts/download_data.py --only semeval hf_categories hf_intensities")


if __name__ == "__main__":
    main()
