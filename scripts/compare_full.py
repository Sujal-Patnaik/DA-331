#!/usr/bin/env python
"""Compare the full-data models on test and write reports/results_noquota.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from emoldgn import evaluate as E  # noqa: E402
from spark_jobs.common import EMOS, load_config, path_of  # noqa: E402

RUNS = ("b4_full", "final_full", "final_full_noquota")


def main() -> None:
    cfg = load_config()
    d = {r: E.load_preds(cfg, r, "test") for r in RUNS}
    order = {r: np.argsort(v["id"]) for r, v in d.items()}
    d = {r: {k: a[order[r]] for k, a in v.items()} for r, v in d.items()}
    if not all(np.array_equal(d[RUNS[0]]["id"], v["id"]) for v in d.values()):
        raise ValueError("models were not scored on the same rows")
    s = {r: E.score_track_a(v) for r, v in d.items()}
    b1 = pd.read_csv(path_of(cfg, "reports", "results_b1.csv"))
    b1 = b1[b1["part"] == "test"].set_index("lang")["macro_f1"].to_dict()
    new = d["final_full_noquota"]
    tb = E.track_b(new["intensity"], new["int_pred"], new["lang"])
    tb_old = E.track_b(d["final_full"]["intensity"], d["final_full"]["int_pred"], d["final_full"]["lang"])
    has = {l for l in s["b4_full"]["per_language"] if (new["intensity"][new["lang"] == l] >= 0).any()}
    mean = lambda r, g: 100 * np.mean([s[r]["per_language"][l] for l in g])  # noqa: E731
    langs = sorted(s["b4_full"]["per_language"])
    L = ["# Full-data comparison, test split", "", "Macro-F1 in percentage points. The model was chosen on validation, then scored once on test.", "",
         "| Group | B4 | Final | Final, no quota |", "|---|---|---|---|"]
    for name, g in (("All 28", langs), ("11 with intensity labels", sorted(has)), ("17 without", sorted(set(langs) - has))):
        L.append(f"| {name} | " + " | ".join(f"{mean(r, g):.2f}" for r in RUNS) + " |")
    L += ["", f"Track B Pearson: final {tb_old['mean']:.3f}, no quota {tb['mean']:.3f}.", ""]
    for other in ("final_full", "b4_full"):
        b = E.paired_bootstrap(new["y"], new["pred"], d[other]["pred"], new["mask"], new["lang"])
        L.append(f"- No quota minus {other}: {b['mean_diff'] * 100:.2f} points, 95% CI [{b['ci_low'] * 100:.2f}, {b['ci_high'] * 100:.2f}], one-sided p = {b['p_value']:.3f}.")
    for r in RUNS:
        v = d[r]
        L.append(f"- {r}: co-occurrence fidelity {E.cooccurrence_fidelity(v['y'], v['pred'], v['mask'], v['lang']):.4f}, ECE {E.ece(v['p'][:, :6], v['y'], v['mask']):.4f}.")
    c = E.load_preds(cfg, "final_full_noquota", "trackc")
    cs = E.score_track_a(c)
    L += ["", "## Track C zero-shot", "", f"Mean {cs['mean'] * 100:.2f}: " + ", ".join(f"{k} {v * 100:.2f}" for k, v in cs["per_language"].items()) + ".", "",
          "## Per language", "", "| Language | B1 | B4 | Final | No quota | Pearson (no quota) |", "|---|---|---|---|---|---|"]
    for l in langs:
        r = tb["per_language"].get(l)
        L.append(f"| {l} | {100 * b1[l]:.2f} | " + " | ".join(f"{100 * s[x]['per_language'][l]:.2f}" for x in RUNS) + f" | {'n/a' if r is None else format(r, '.3f')} |")
    pe = {r: E.per_emotion_f1(d[r]["y"], d[r]["pred"], d[r]["mask"], d[r]["lang"]) for r in RUNS}
    L += ["", "## Per emotion", "", "| Emotion | B4 | Final | No quota |", "|---|---|---|---|"]
    L += [f"| {e} | " + " | ".join(f"{100 * pe[r][e]:.2f}" for r in RUNS) + " |" for e in EMOS]
    Path(path_of(cfg, "reports", "results_noquota.md")).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:18]))
    print(json.dumps({"wins_vs_b4": sum(s["final_full_noquota"]["per_language"][l] > s["b4_full"]["per_language"][l] for l in langs),
                      "gaps": sorted((round(100 * (s["final_full_noquota"]["per_language"][l] - s["b4_full"]["per_language"][l]), 1), l) for l in langs)[:5]}))


if __name__ == "__main__":
    main()
