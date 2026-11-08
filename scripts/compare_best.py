#!/usr/bin/env python
"""Compare best_full with the earlier full-data models on test; writes reports/results_best.md.

best_full was chosen on validation by the rule in docs/DECISIONS.md. Earlier full-data test
scores had already been seen when that rule was written, which the report states.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from emoldgn import evaluate as E  # noqa: E402
from spark_jobs.common import EMOS, load_config, path_of  # noqa: E402

RUNS = ("b4_full", "final_full", "final_full_noquota", "best_full")
NAMES = ("B4", "Final", "Final, no quota", "Best (B4 + N6)")


def main() -> None:
    cfg = load_config()
    d = {r: E.load_preds(cfg, r, "test") for r in RUNS}
    order = {r: np.argsort(v["id"], kind="stable") for r, v in d.items()}
    d = {r: {k: a[order[r]] for k, a in v.items()} for r, v in d.items()}
    if not all(np.array_equal(d[RUNS[0]]["id"], v["id"]) and np.array_equal(d[RUNS[0]]["lang"], v["lang"]) for v in d.values()):
        raise ValueError("models were not scored on the same rows")
    s = {r: E.score_track_a(v) for r, v in d.items()}
    val = {r: E.score_track_a(E.load_preds(cfg, r, "val"))["mean"] for r in RUNS}
    b1 = pd.read_csv(path_of(cfg, "reports", "results_b1.csv"))
    b1 = b1[b1["part"] == "test"].set_index("lang")["macro_f1"].to_dict()
    new = d["best_full"]
    tb = E.track_b(new["intensity"], new["int_pred"], new["lang"])
    langs = sorted(s["b4_full"]["per_language"])
    has = {l for l in langs if (new["intensity"][new["lang"] == l] >= 0).any()}
    mean = lambda r, g: 100 * np.mean([s[r]["per_language"][l] for l in g])  # noqa: E731
    L = ["# best_full on test", "",
         "Macro-F1 in percentage points. best_full was chosen on validation and scored once on test.",
         "Test scores of the earlier full-data models had already been seen when the selection rule was written, so this is not a fully blind comparison.", "",
         "| Group | " + " | ".join(NAMES) + " |", "|---|---|---|---|---|",
         "| Validation, tuned | " + " | ".join(f"{100 * val[r]:.2f}" for r in RUNS) + " |"]
    for name, g in (("Test, all 28", langs), ("Test, 11 with intensity", sorted(has)), ("Test, 17 without", sorted(set(langs) - has))):
        L.append(f"| {name} | " + " | ".join(f"{mean(r, g):.2f}" for r in RUNS) + " |")
    L += ["", "Track B Pearson: " + ", ".join(f"{r} {E.track_b(d[r]['intensity'], d[r]['int_pred'], d[r]['lang'])['mean']:.3f}" for r in RUNS[1:]) + ".", ""]
    for other in RUNS[:3]:
        b = E.paired_bootstrap(new["y"], new["pred"], d[other]["pred"], new["mask"], new["lang"])
        L.append(f"- best_full minus {other}: {b['mean_diff'] * 100:+.2f} points, 95% CI [{b['ci_low'] * 100:.2f}, {b['ci_high'] * 100:.2f}], one-sided p = {b['p_value']:.3f}.")
    for r in RUNS:
        v = d[r]
        L.append(f"- {r}: co-occurrence fidelity {E.cooccurrence_fidelity(v['y'], v['pred'], v['mask'], v['lang']):.4f}, ECE {E.ece(v['p'][:, :6], v['y'], v['mask']):.4f}.")
    gaps = sorted((100 * (s["best_full"]["per_language"][l] - s["b4_full"]["per_language"][l]), l) for l in langs)
    L.append(f"- best_full beats B4 on {sum(g > 0 for g, _ in gaps)} of {len(langs)} languages. Largest losses: "
             + ", ".join(f"{l} {g:+.1f}" for g, l in gaps[:3]) + ". Largest gains: " + ", ".join(f"{l} {g:+.1f}" for g, l in gaps[-3:][::-1]) + ".")
    L += ["", "## Track C zero-shot", "", "| Model | Mean | Per language |", "|---|---|---|"]
    for r in RUNS[1:]:
        c = E.score_track_a(E.load_preds(cfg, r, "trackc"))
        L.append(f"| {r} | {c['mean'] * 100:.2f} | " + ", ".join(f"{k} {v * 100:.2f}" for k, v in c["per_language"].items()) + " |")
    L += ["", "## Per language", "", "| Language | B1 | B4 | Final | No quota | Best | Pearson (best) |", "|---|---|---|---|---|---|---|"]
    for l in langs:
        p = tb["per_language"].get(l)
        L.append(f"| {l} | {100 * b1[l]:.2f} | " + " | ".join(f"{100 * s[x]['per_language'][l]:.2f}" for x in RUNS) + f" | {'n/a' if p is None else format(p, '.3f')} |")
    pe = {r: E.per_emotion_f1(d[r]["y"], d[r]["pred"], d[r]["mask"], d[r]["lang"]) for r in RUNS}
    L += ["", "## Per emotion", "", "| Emotion | " + " | ".join(NAMES) + " |", "|---|---|---|---|---|"]
    L += [f"| {e} | " + " | ".join(f"{100 * pe[r][e]:.2f}" for r in RUNS) + " |" for e in EMOS]
    Path(path_of(cfg, "reports", "results_best.md")).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:32]))


if __name__ == "__main__":
    main()
