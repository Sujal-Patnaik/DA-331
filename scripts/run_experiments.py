#!/usr/bin/env python
"""Runs config/experiments.yaml top to bottom (CLAUDE.md section 6, M8).

Finished runs are skipped and unfinished ones resumed. Launch in the background:
    nohup python scripts/run_experiments.py > runs/run_all.log 2>&1 &
With --evaluate it only does the post-training step: thresholds on val, one test pass.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from spark_jobs.common import load_config, path_of  # noqa: E402


def status_of(run: Path) -> str | None:
    p = run / "status.json"
    return json.loads(p.read_text())["status"] if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="restrict to these experiment ids")
    ap.add_argument("--skip", nargs="*", default=[], help="leave these ids out (for example the *_full runs on 4 GB)")
    ap.add_argument("--evaluate", action="store_true", help="predict and score finished runs instead of training")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    with open(path_of(cfg, "experiments"), encoding="utf-8") as f:
        entries = yaml.safe_load(f)["experiments"]
    py = sys.executable
    extra = ["--config", args.config] if args.config else []
    for e in entries:
        exp = e["id"]
        if exp in args.skip or (args.only and exp not in args.only):
            continue
        run = Path(path_of(cfg, "runs", exp))
        state = status_of(run)
        if args.evaluate:
            if state != "done":
                print(f"{exp}: status {state}, not evaluated", flush=True)
                continue
            subprocess.run([py, "-m", "emoldgn.predict", "--exp", exp, *extra], cwd=ROOT, check=True)
            continue
        if state == "done":
            print(f"{exp}: done, skipping", flush=True)
            continue
        print(f"{exp}: starting (was {state})", flush=True)
        code = subprocess.run([py, "-m", "emoldgn.train", "--exp", exp, "--resume", *extra], cwd=ROOT).returncode
        final = status_of(run)
        print(f"{exp}: finished with status {final}", flush=True)
        if code != 0 or final != "done":
            print("stopping the queue; see the run's train.log", flush=True)
            return code or 1
    if args.evaluate:
        return subprocess.run([py, "-m", "emoldgn.evaluate", "--results", *extra], cwd=ROOT).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
