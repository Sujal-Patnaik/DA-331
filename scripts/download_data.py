#!/usr/bin/env python
"""Idempotent download of every input listed in CLAUDE.md section 3.

Each step skips outputs that already exist, retries once, and on a second
failure records the item instead of crashing, so one missing optional input
never blocks the rest. BRIGHTER items are required; the others are optional.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("download")

SEMEVAL_URL = "https://github.com/emotion-analysis-project/SemEval2025-Task11.git"
EDURABSA_URL = "https://github.com/yhua219/edurabsa_dataset_and_annotation_tool.git"
NRC_VAD_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
HF_CATEGORIES = "brighter-dataset/BRIGHTER-emotion-categories"
HF_INTENSITIES = "brighter-dataset/BRIGHTER-emotion-intensities"
HF_SUPPORT = "TNE-AI/customer-support-on-twitter-conversation"
HF_EDURABSA = "yhua219/EduRABSA_ASQE"


def _retry(name: str, fn, attempts: int = 2) -> bool:
    """Run fn, retrying once; log and return False if it still fails."""
    for i in range(1, attempts + 1):
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001 - a failed item must not stop the rest
            log.warning("%s: attempt %d/%d failed: %s", name, i, attempts, exc)
            time.sleep(2)
    return False


def git_clone(url: str, dest: Path) -> None:
    """Shallow clone into dest unless it already exists."""
    if (dest / ".git").exists():
        log.info("exists, skipping: %s", dest)
        return
    if dest.exists() and any(dest.iterdir()):
        log.info("exists (not a git checkout), skipping: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def fetch(url: str, dest: Path) -> None:
    """Download url to dest atomically."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info("exists, skipping: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "emoldgn-download/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def hf_all_configs(repo: str, dest: Path) -> None:
    """Save every config and split of a Hugging Face dataset as Parquet."""
    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names(repo)
    log.info("%s: %d configs: %s", repo, len(configs), configs)
    failed = []
    for cfg in configs:
        out = dest / cfg
        if (out / "_DONE").exists():
            continue
        try:
            ds = load_dataset(repo, cfg)
            out.mkdir(parents=True, exist_ok=True)
            for split, d in ds.items():
                d.to_parquet(str(out / f"{split}.parquet"))
            (out / "_DONE").write_text("ok\n")
        except Exception as exc:  # noqa: BLE001
            failed.append(cfg)
            log.warning("%s/%s failed: %s", repo, cfg, exc)
    if failed:
        raise RuntimeError(f"{repo}: failed configs {failed}")


def hf_support(dest: Path) -> None:
    """Customer-support demo input (never trained on)."""
    out = dest / "train.parquet"
    if out.exists():
        log.info("exists, skipping: %s", out)
        return
    from datasets import load_dataset

    ds = load_dataset(HF_SUPPORT, split="train")
    dest.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out))


def edurabsa_hf(dest: Path) -> None:
    """EduRABSA reviews. The git repository holds only a pointer; the data is on Hugging Face."""
    from datasets import load_dataset

    ds = load_dataset(HF_EDURABSA)
    dest.mkdir(parents=True, exist_ok=True)
    for split, d in ds.items():
        out = dest / f"{split}.parquet"
        if not out.exists():
            d.to_parquet(str(out))


def nrc_vad(dest: Path) -> None:
    """NRC VAD lexicon v2.1 (research use, not redistributed; plan 9.3)."""
    if any(dest.rglob("*.txt")):
        log.info("exists, skipping: %s", dest)
        return
    z = dest / "NRC-VAD-Lexicon-v2.1.zip"
    fetch(NRC_VAD_URL, z)
    with zipfile.ZipFile(z) as f:
        f.extractall(dest)


def coursera(dest: Path) -> None:
    """Optional Kaggle reviews; only attempted when a Kaggle token exists."""
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        raise FileNotFoundError("no ~/.kaggle/kaggle.json")
    if dest.exists() and any(dest.iterdir()):
        log.info("exists, skipping: %s", dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "imuhammad/course-reviews-on-coursera",
         "-p", str(dest), "--unzip"],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="run only these step names")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = ROOT / "data"
    steps = {
        "semeval": (True, lambda: git_clone(SEMEVAL_URL, data / "raw/brighter/semeval_repo")),
        "hf_categories": (True, lambda: hf_all_configs(HF_CATEGORIES, data / "raw/brighter/hf_categories")),
        "hf_intensities": (True, lambda: hf_all_configs(HF_INTENSITIES, data / "raw/brighter/hf_intensities")),
        "customer_support": (False, lambda: hf_support(data / "demo/customer_support")),
        "edurabsa": (False, lambda: git_clone(EDURABSA_URL, data / "demo/edurabsa")),
        "edurabsa_hf": (False, lambda: edurabsa_hf(data / "demo/edurabsa_hf")),
        "coursera": (False, lambda: coursera(data / "demo/coursera")),
        "nrc_vad": (False, lambda: nrc_vad(data / "lexicons")),
        "lid176": (False, lambda: fetch(LID_URL, ROOT / "models/lid.176.bin")),
    }
    status: dict[str, str] = {}
    for name, (_required, fn) in steps.items():
        if args.only and name not in args.only:
            continue
        log.info("=== %s", name)
        # The Kaggle step fails fast and predictably when no token exists.
        status[name] = "ok" if _retry(name, fn, attempts=1 if name == "coursera" else 2) else "FAILED"
    log.info("summary: %s", json.dumps(status))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/download_status.json").write_text(json.dumps(status, indent=2) + "\n")
    missing = [n for n, s in status.items() if s == "FAILED" and steps[n][0]]
    if missing:
        log.error("required inputs missing: %s", missing)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
