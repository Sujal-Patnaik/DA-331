#!/usr/bin/env python
"""Tables 2 and 3 on test, after the choice was frozen; writes reports/ablation_test.md.

These scores had no part in choosing best_full. reports/results.md is untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emoldgn.evaluate import track_a, track_b  # noqa: E402
from scripts.ablation_report import CUMULATIVE, SEEDS, SINGLES  # noqa: E402


def score(exp: str) -> dict:
    """Test scores of one run."""
    d = np.load(ROOT / "runs" / exp / "predictions" / "test.npz", allow_pickle=True)
    a = track_a(d["y"], d["pred"], d["mask"], d["lang"])
    p = track_b(d["intensity"], d["int_pred"], d["lang"])["mean"] if "int_pred" in d.files else None
    return {"f1": 100 * a["mean"], "lang": {k: 100 * v for k, v in a["per_language"].items()}, "pearson": p}


def main() -> None:
    ids = sorted({*CUMULATIVE, *SINGLES.values(), *sum(SEEDS.values(), [])})
    r = {e: score(e) for e in ids}
    base = r["b4_subset_s42"]["f1"]
    langs = sorted(r[ids[0]]["lang"])
    head = "| Run | Macro-F1 | Vs B4 | " + " | ".join(langs) + " | Pearson |"
    sep = "|---" * (4 + len(langs)) + "|"

    def row(e: str, name: str) -> str:
        x = r[e]
        cells = " | ".join(f"{x['lang'][k]:.2f}" for k in langs)
        p = "-" if x["pearson"] is None else f"{x['pearson']:.3f}"
        return f"| {name} | {x['f1']:.2f} | {x['f1'] - base:+.2f} | {cells} | {p} |"

    L = ["# Ablation on test", "", "Scored after best_full was chosen and tested; these scores played no part in the choice.", "",
         "## Table 2: cumulative (seed 42)", "", head, sep] + [row(e, e) for e in CUMULATIVE]
    L += ["", "## Each addition alone (seed 42)", "", head, sep] + [row(e, f"B4 + {n}") for n, e in SINGLES.items()]
    L += ["", "## Table 3: three seeds", "", "| Model | Mean | Std | Seeds |", "|---|---|---|---|"]
    for k, v in SEEDS.items():
        x = [r[e]["f1"] for e in v]
        L.append(f"| {k} | {np.mean(x):.2f} | {np.std(x, ddof=1):.2f} | " + ", ".join(f"{a:.2f}" for a in x) + " |")
    Path(ROOT / "reports/ablation_test.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
