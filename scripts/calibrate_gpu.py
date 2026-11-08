#!/usr/bin/env python
"""M7: 200 steps of the final configuration on full data, then the time budget.

Writes reports/budget.md. Uses a separate run folder, so no experiment is touched, and
applies the plan's formula T = N_train * E / samples-per-second + 10% for validation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from emoldgn.train import experiments, resolve, train_run, vram_gb  # noqa: E402
from spark_jobs.common import path_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--exp", default="final_full", help="experiment whose configuration is measured")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = resolve(args.exp, args.config)
    run = Path(path_of(cfg, "runs", "_calibration"))
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    train_run(cfg, run, max_steps=args.steps)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    line = json.loads((run / "metrics.jsonl").read_text().splitlines()[-1])
    speed = float(line["samples_per_s"])

    df = pd.read_parquet(path_of(cfg, "gold", "tokenized"), columns=["lang", "part", "track"])
    df = df[(df["part"] == "train") & (df["track"] == "A")]
    n_full, n_subset = len(df), int(df["lang"].isin(cfg["subset_langs"]).sum())
    rows, total = [], 0.0
    for e in experiments(cfg):
        n = n_full if e["data"] == "full" else n_subset
        epochs_cfg = cfg["epochs"][e["epochs"]]
        epochs = epochs_cfg["max"]
        hours = n * epochs / speed * 1.10 / 3600
        total += hours
        rows.append((e["id"], e["data"], n, epochs, hours))
    full = sum(h for i, _, _, _, h in rows if i.endswith("_full"))
    text = [
        "# GPU budget (M7)", "",
        f"Measured over {args.steps} training steps of `{args.exp}`: **{speed:.1f} samples/s**, "
        f"peak allocated {peak:.2f} GB, peak reserved {reserved:.2f} GB, on a {vram_gb():.1f} GB card "
        f"(batch {cfg['batch_size']}, max_len {cfg['max_len']}, fp16 {cfg['fp16']}, "
        f"frozen embeddings {cfg['freeze_word_embeddings']}).", "",
        "Projection: T = N_train x E / samples per second, plus 10% for validation passes, at max epochs.", "",
        "| Run | Data | N train | Epochs | Hours |", "|---|---|---|---|---|",
    ]
    text += [f"| {i} | {d} | {n} | {ep} | {h:.2f} |" for i, d, n, ep, h in rows]
    text += ["", f"**Total: {total:.2f} hours** (subset runs {total - full:.2f}, full runs {full:.2f}). "
             "Early stopping usually ends runs sooner; the optional 3-epoch extension of final_full is not included.", ""]
    Path(path_of(cfg, "reports", "budget.md")).write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    shutil.rmtree(run, ignore_errors=True)


if __name__ == "__main__":
    main()
