#!/usr/bin/env python
"""Publish the best run to artifacts/ for the applications (CLAUDE.md M8).

Writes best.pt, thresholds.json and lang_index.json. --from-run can name a smoke run so the
application pipeline can be exercised before M8; that artifact is marked and is not a result.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from spark_jobs.common import load_config, path_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-run", default="final_full")
    ap.add_argument("--smoke", action="store_true", help="mark the artifact as not a real result")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Path(path_of(cfg, "runs", args.from_run))
    out = Path(path_of(cfg, "artifacts"))
    out.mkdir(exist_ok=True)
    shutil.copyfile(run / "ckpt_best.pt", out / "best.pt")
    shutil.copyfile(run / "thresholds.json", out / "thresholds.json")
    bank = torch.load(path_of(cfg, "gold", "graph_bank.pt"), weights_only=False)
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    names = bank["languages"]
    index = {
        "languages": {n: i for i, n in enumerate(names)},
        "families": {n: bank["families"][int(f)] for n, f in zip(names, bank["family_index"]) if int(f) < len(bank["families"])},
        "emotion_mask": {n: [int(x) for x in bank["mask"][i][:6].tolist()] for i, n in enumerate(names)},
        "has_neutral": bool(ck["flags"].get("n2_signed")),
        "run": args.from_run, "smoke_only": args.smoke,
    }
    (out / "lang_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"exported {args.from_run} to {out}" + (" (SMOKE ONLY, not a result)" if args.smoke else ""))


if __name__ == "__main__":
    main()
