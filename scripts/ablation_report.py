#!/usr/bin/env python
"""Validation-only ablation table and selection (docs/DECISIONS.md, "Ablation").

Reads runs/<id>/predictions/val.npz and never opens a test file. Writes
reports/ablation_val.md and reports/ablation_val.json; reports/results.md is untouched.
Thresholds are tuned on the same validation rows, so absolute scores run a little high;
the bias is shared by every run, and only differences are used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emoldgn.evaluate import cooccurrence_fidelity, ece, track_a, track_b  # noqa: E402

CUMULATIVE = ["b2_subset", "b4_subset_s42", "s1_subset", "s2_subset", "final_subset_s42"]
SINGLES = {"N1": "a_n1_subset", "N2": "a_n2_subset", "N4": "a_n4_subset", "N5": "a_n5_subset", "N6": "a_n6_subset"}
SEEDS = {"b4": ["b4_subset_s42", "b4_subset_s1", "b4_subset_s2"],
         "final": ["final_subset_s42", "final_subset_s1", "final_subset_s2"],
         "nq": ["nq_subset_s42", "nq_subset_s1", "nq_subset_s2"]}


def score(exp: str) -> dict:
    """Tuned validation scores for one run."""
    d = np.load(ROOT / "runs" / exp / "predictions" / "val.npz", allow_pickle=True)
    a = track_a(d["y"], d["pred"], d["mask"], d["lang"])
    out = {"exp": exp, "macro_f1": 100 * a["mean"], "per_lang": {k: 100 * v for k, v in a["per_language"].items()},
           "fidelity": cooccurrence_fidelity(d["y"], d["pred"], d["mask"], d["lang"]),
           "ece": ece(d["p"][:, :6], d["y"], d["mask"]), "pearson": None}
    if "int_pred" in d.files:
        out["pearson"] = track_b(d["intensity"], d["int_pred"], d["lang"])["mean"]
    return out


def verdict(delta: float, noise: float) -> str:
    """The committed rule: helps above +noise, hurts below -noise, otherwise neutral."""
    return "helps" if delta > noise else "hurts" if delta < -noise else "neutral"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/ablation_val")
    args = ap.parse_args()
    ids = sorted({*CUMULATIVE, *SINGLES.values(), *sum(SEEDS.values(), [])})
    r = {e: score(e) for e in ids}
    mean_std = {k: (float(np.mean([r[e]["macro_f1"] for e in v])), float(np.std([r[e]["macro_f1"] for e in v], ddof=1)))
                for k, v in SEEDS.items()}
    noise = max(0.5, mean_std["b4"][1])
    base = r["b4_subset_s42"]["macro_f1"]
    langs = sorted(r[ids[0]]["per_lang"])
    head = "| Run | Macro-F1 | Vs B4 | " + " | ".join(langs) + " | Pearson | Fidelity | ECE |"
    sep = "|---" * (7 + len(langs)) + "|"

    def row(e: str, name: str) -> str:
        x = r[e]
        p = "-" if x["pearson"] is None else f"{x['pearson']:.3f}"
        cells = " | ".join(f"{x['per_lang'][k]:.2f}" for k in langs)
        return f"| {name} | {x['macro_f1']:.2f} | {x['macro_f1'] - base:+.2f} | {cells} | {p} | {x['fidelity']:.3f} | {x['ece']:.4f} |"

    lines = ["# Ablation on validation", "",
             "Threshold-tuned validation macro-F1 on the subset languages. No test file was read.",
             f"Noise = max(0.5, B4 seed std) = {noise:.2f} points. Rule: docs/DECISIONS.md, \"Ablation\".", "",
             "## Each addition alone (B4 plus one, seed 42)", "", head, sep, row("b4_subset_s42", "B4")]
    verdicts = {}
    for n, e in SINGLES.items():
        verdicts[n] = verdict(r[e]["macro_f1"] - base, noise)
        lines.append(row(e, f"B4 + {n} ({verdicts[n]})"))
    lines += ["", "## Cumulative (seed 42)", "", head, sep]
    lines += [row(e, e) for e in CUMULATIVE]
    lines += ["", "## Three seeds", "", "| Model | Mean | Std | Seeds |", "|---|---|---|---|"]
    for k, v in SEEDS.items():
        vals = ", ".join(f"{r[e]['macro_f1']:.2f}" for e in v)
        lines.append(f"| {k} | {mean_std[k][0]:.2f} | {mean_std[k][1]:.2f} | {vals} |")
    gap = mean_std["final"][0] - mean_std["b4"][0]
    lines += ["", f"CLAUDE.md 9.6 check: final minus B4 = {gap:+.2f} points (validation; limit -2)."]
    Path(args.out + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out + ".json").write_text(json.dumps({"noise": noise, "verdicts": verdicts, "seeds": mean_std, "runs": r}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
