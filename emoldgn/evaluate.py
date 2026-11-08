"""Metrics and the M8 results report (plan 10.3; official scorer logic reproduced).

The official scorer (SemEval2025-Task11-Evaluation/utils.py) is sklearn macro-F1 per
language for Tracks A and C, and per-emotion Pearson r on integer predictions for Track B.

Track B is scored on rows of the Track A test part: Track B's own amh and rus splits put
Track A training texts in Track B test, which would leak (see reports/DATA_NOTES.md).

``python -m emoldgn.evaluate --results`` builds reports/results.md from run predictions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spark_jobs.common import EMOS, load_config, path_of  # noqa: E402


def f1_binary(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-column binary F1 with zero_division = 0, as sklearn does."""
    tp = ((y == 1) & (pred == 1)).sum(0).astype(float)
    denom = (y == 1).sum(0) + (pred == 1).sum(0)
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(y: np.ndarray, pred: np.ndarray, present: np.ndarray) -> float:
    """Macro-F1 over the emotions a language has (Track A, rule 5)."""
    keep = present > 0
    return float(f1_binary(y[:, keep], pred[:, keep]).mean()) if keep.any() else 0.0


def track_a(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, lang: Sequence[str]) -> dict[str, Any]:
    """Per-language macro-F1 and their mean."""
    lang = np.asarray(lang)
    per = {}
    for name in sorted(set(lang.tolist())):
        m = lang == name
        per[name] = macro_f1(y[m], pred[m], mask[m].max(0))
    return {"per_language": per, "mean": float(np.mean(list(per.values()))) if per else 0.0}


def per_emotion_f1(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, lang: Sequence[str]) -> dict[str, float]:
    """Mean F1 of each emotion over the languages that have it."""
    lang = np.asarray(lang)
    out: dict[str, list[float]] = {e: [] for e in EMOS}
    for name in set(lang.tolist()):
        m = lang == name
        f = f1_binary(y[m], pred[m])
        for c, e in enumerate(EMOS):
            if mask[m][:, c].max() > 0:
                out[e].append(float(f[c]))
    return {e: float(np.mean(v)) if v else float("nan") for e, v in out.items()}


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r; an undefined correlation (constant input) counts as 0."""
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(stats.pearsonr(a, b)[0])


def track_b(gold: np.ndarray, pred: np.ndarray, lang: Sequence[str]) -> dict[str, Any]:
    """Pearson r per emotion per language, averaged; rows or emotions with -1 are skipped."""
    lang = np.asarray(lang)
    per: dict[str, float] = {}
    table: dict[str, dict[str, float]] = {}
    for name in sorted(set(lang.tolist())):
        m = lang == name
        scores = {}
        for c, e in enumerate(EMOS):
            keep = gold[m, c] >= 0
            if keep.sum() >= 2:
                scores[e] = safe_pearson(gold[m, c][keep].astype(float), pred[m, c][keep].astype(float))
        if scores:
            per[name] = float(np.mean(list(scores.values())))
            table[name] = scores
    return {"per_language": per, "per_emotion": table, "mean": float(np.mean(list(per.values()))) if per else 0.0}


def cooccurrence(y: np.ndarray) -> np.ndarray:
    """Share of rows holding each emotion pair."""
    y = y.astype(float)
    return y.T @ y / max(len(y), 1)


def cooccurrence_fidelity(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, lang: Sequence[str]) -> float:
    """Mean Frobenius distance between predicted and gold co-occurrence (plan 10.3)."""
    lang = np.asarray(lang)
    out = []
    for name in set(lang.tolist()):
        m = lang == name
        keep = mask[m].max(0) > 0
        out.append(np.linalg.norm(cooccurrence(pred[m][:, keep]) - cooccurrence(y[m][:, keep])))
    return float(np.mean(out)) if out else 0.0


def ece(p: np.ndarray, y: np.ndarray, mask: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error over all (row, present emotion) probabilities."""
    keep = mask > 0
    prob, truth = p[keep], y[keep]
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(prob[m].mean() - truth[m].mean())
    return float(total)


def intensity_confusion(gold: np.ndarray, pred: np.ndarray) -> dict[str, list[list[int]]]:
    """4 x 4 confusion matrix per emotion (plan 10.3)."""
    out = {}
    for c, e in enumerate(EMOS):
        keep = gold[:, c] >= 0
        cm = np.zeros((4, 4), dtype=int)
        np.add.at(cm, (gold[keep, c].astype(int), np.clip(pred[keep, c].astype(int), 0, 3)), 1)
        out[e] = cm.tolist()
    return out


def paired_bootstrap(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, mask: np.ndarray, lang: Sequence[str],
                     resamples: int = 1000, seed: int = 42) -> dict[str, float]:
    """Paired bootstrap of mean macro-F1(a) - macro-F1(b), resampling rows inside languages."""
    lang = np.asarray(lang)
    rng = np.random.default_rng(seed)
    groups = [(np.flatnonzero(lang == n), mask[lang == n].max(0) > 0) for n in sorted(set(lang.tolist()))]
    diffs = np.zeros(resamples)
    for r in range(resamples):
        gaps = []
        for rows, keep in groups:
            s = rng.choice(rows, len(rows))
            fa = f1_binary(y[s][:, keep], pred_a[s][:, keep]).mean()
            fb = f1_binary(y[s][:, keep], pred_b[s][:, keep]).mean()
            gaps.append(fa - fb)
        diffs[r] = np.mean(gaps)
    return {"mean_diff": float(diffs.mean()), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "p_value": float((diffs <= 0).mean())}


# ---------------------------------------------------------------- reporting (M8)

def load_preds(cfg: dict[str, Any], exp: str, split: str) -> dict[str, np.ndarray] | None:
    """Predictions written by emoldgn.predict, or None if that step has not run."""
    p = Path(path_of(cfg, "runs", exp, "predictions", f"{split}.npz"))
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None


def score_track_a(d: dict[str, np.ndarray]) -> dict[str, Any]:
    """Track A scores of a predictions file."""
    return track_a(d["y"], d["pred"], d["mask"], d["lang"])


def fmt(x: float | None, pct: bool = True) -> str:
    """Scores print as percentage points by default."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.2f}" if pct else f"{x:.3f}"


def build_report(cfg: dict[str, Any]) -> str:
    """reports/results.md from whatever runs have finished; absent ones print as n/a."""
    import pandas as pd

    lines = ["# Results", "", "Scores are macro-F1 in percentage points unless stated. "
             "Thresholds were tuned on validation only.", ""]
    preds = {e: load_preds(cfg, e, "test") for e in (
        "b2_subset", "b4_subset_s42", "s1_subset", "s2_subset", "final_subset_s42", "b4_subset_s1", "b4_subset_s2",
        "final_subset_s1", "final_subset_s2", "b4_full", "final_full")}

    lines += ["## Table 1. Full data, test split", ""]
    b1_path = Path(path_of(cfg, "reports", "results_b1.csv"))
    b1 = pd.read_csv(b1_path) if b1_path.exists() else None
    b1_test = b1[b1["part"] == "test"].set_index("lang")["macro_f1"].to_dict() if b1 is not None else {}
    b4, final = preds["b4_full"], preds["final_full"]
    s_b4 = score_track_a(b4) if b4 else {"per_language": {}, "mean": None}
    s_fin = score_track_a(final) if final else {"per_language": {}, "mean": None}
    tb = None
    if final is not None:
        tb = track_b(final["intensity"], final["int_pred"], final["lang"])
    langs = sorted(set(b1_test) | set(s_b4["per_language"]) | set(s_fin["per_language"]))
    lines += ["| Language | B1 | B4 full | Final full | Track B Pearson (final) |", "|---|---|---|---|---|"]
    for name in langs:
        r = tb["per_language"].get(name) if tb else None
        lines.append(f"| {name} | {fmt(b1_test.get(name))} | {fmt(s_b4['per_language'].get(name))} | "
                     f"{fmt(s_fin['per_language'].get(name))} | {fmt(r, False)} |")
    b1_mean = float(np.mean(list(b1_test.values()))) if b1_test else None
    lines.append(f"| **mean** | {fmt(b1_mean)} | {fmt(s_b4['mean'])} | {fmt(s_fin['mean'])} | "
                 f"{fmt(tb['mean'] if tb else None, False)} |")

    lines += ["", "## Table 2. Cumulative ablation on the subset (seed 42, test split)", "",
              "| Run | Model | Macro-F1 |", "|---|---|---|"]
    for exp, name in [("b2_subset", "B2 linear head"), ("b4_subset_s42", "B4 faithful LDGN"), ("s1_subset", "S1 = B4 + N1 + N2"),
                      ("s2_subset", "S2 = S1 + N4 + N5"), ("final_subset_s42", "Final = S2 + N6")]:
        d = preds[exp]
        lines.append(f"| {exp} | {name} | {fmt(score_track_a(d)['mean'] if d else None)} |")

    lines += ["", "## Table 3. B4 versus Final on the subset, 3 seeds", "", "| Model | Mean | Std | Seeds |", "|---|---|---|---|"]
    gaps: dict[str, float | None] = {}
    for label, group in [("B4", ("b4_subset_s42", "b4_subset_s1", "b4_subset_s2")),
                         ("Final", ("final_subset_s42", "final_subset_s1", "final_subset_s2"))]:
        vals = [score_track_a(preds[e])["mean"] for e in group if preds[e]]
        gaps[label] = float(np.mean(vals)) if len(vals) == 3 else None
        lines.append(f"| {label} | {fmt(float(np.mean(vals)) if vals else None)} | "
                     f"{fmt(float(np.std(vals)) if len(vals) > 1 else None)} | {len(vals)} |")
    if gaps["B4"] is not None and gaps["Final"] is not None and gaps["Final"] < gaps["B4"] - 0.02:
        lines += ["", "**Stop rule (CLAUDE.md 9.6): Final is more than 2 points below B4 on the subset. "
                      "Report this to the user; start no new experiments.**"]

    lines += ["", "## Significance, fidelity, calibration", ""]
    if b4 and final:
        order_b, order_f = np.argsort(b4["id"]), np.argsort(final["id"])
        if not np.array_equal(b4["id"][order_b], final["id"][order_f]):
            raise ValueError("b4_full and final_full were not scored on the same rows")
        boot = paired_bootstrap(final["y"][order_f], final["pred"][order_f], b4["pred"][order_b],
                                final["mask"][order_f], final["lang"][order_f])
        lines.append(f"- Paired bootstrap final_full vs b4_full (1,000 resamples): difference "
                     f"{boot['mean_diff'] * 100:.2f} points, 95% CI [{boot['ci_low'] * 100:.2f}, {boot['ci_high'] * 100:.2f}], "
                     f"p = {boot['p_value']:.3f}.")
    for name, d in (("b4_full", b4), ("final_full", final)):
        if d:
            lines.append(f"- {name}: co-occurrence fidelity {cooccurrence_fidelity(d['y'], d['pred'], d['mask'], d['lang']):.4f} "
                         f"(lower is better); ECE {ece(d['p'][:, :6], d['y'], d['mask']):.4f}.")
    if not (b4 or final):
        lines.append("Not available until the full-data runs finish.")

    lines += ["", "## Per-emotion F1 (mean over languages)", "", "| Emotion | B4 full | Final full |", "|---|---|---|"]
    fb = per_emotion_f1(b4["y"], b4["pred"], b4["mask"], b4["lang"]) if b4 else {}
    ff = per_emotion_f1(final["y"], final["pred"], final["mask"], final["lang"]) if final else {}
    lines += [f"| {e} | {fmt(fb.get(e))} | {fmt(ff.get(e))} |" for e in EMOS]

    lines += ["", "## Intensity confusion, final_full (rows = gold 0-3, columns = predicted 0-3)", ""]
    if final is not None:
        for e, cm in intensity_confusion(final["intensity"], final["int_pred"]).items():
            lines += [f"**{e}**", "", "| | 0 | 1 | 2 | 3 |", "|---|---|---|---|---|"]
            lines += [f"| {i} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(cm)] + [""]
    else:
        lines.append("Not available until final_full finishes.")

    lines += ["", "## Track C zero-shot (languages with no training data)", ""]
    c = load_preds(cfg, "final_full", "trackc")
    if c is not None:
        s = score_track_a(c)
        lines += ["| Language | Macro-F1 |", "|---|---|"]
        lines += [f"| {k} | {fmt(v)} |" for k, v in s["per_language"].items()]
        lines.append(f"| **mean** | {fmt(s['mean'])} |")
    else:
        lines.append("Not available until final_full finishes.")
    lines += ["", "Figures are in reports/figures/ (`python scripts/make_figures.py --results`). Attention examples quote "
              "BRIGHTER text, so they stay in runs/<id>/analysis/, which is not committed (rule 7).", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", action="store_true", help="write reports/results.md")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.results:
        report = build_report(cfg)
        out = Path(path_of(cfg, "reports", "results.md"))
        out.write_text(report, encoding="utf-8")
        print(f"wrote {out}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
