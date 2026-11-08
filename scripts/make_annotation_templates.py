#!/usr/bin/env python
"""M10: 300 random clauses per app with empty label columns (plan 9.6).

Label columns stay empty on purpose: the team fills them by hand, never the model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from spark_jobs.common import EMOS, load_config, path_of  # noqa: E402

FILES = {"customer_support": "support_300.csv", "student_feedback": "feedback_300.csv"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--force", action="store_true", help="replace templates, losing any labels in them")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for profile, name in FILES.items():
        out = ROOT / "eval/app_annotation" / name
        if out.exists() and not args.force:
            print(f"exists, skipping: {out}")
            continue
        df = pd.read_parquet(path_of(cfg, "app_silver", profile, "clauses"), columns=["doc_id", "clause_id", "lang", "clause"])
        df = df[df["clause"].str.split().str.len() >= 3].sample(args.n, random_state=int(cfg["seed"]))
        df = df.sort_values(["doc_id", "clause_id"]).reset_index(drop=True)
        for pass_no in (1, 2):
            for e in EMOS:
                df[f"{e}_a{pass_no}"] = ""
            df[f"aspect_a{pass_no}"] = ""
            df[f"annotator_{pass_no}"] = ""
        df.to_csv(out, index=False)
        print(f"wrote {out}: {len(df)} clauses")


if __name__ == "__main__":
    main()
