"""Application evaluation (plan 9.6, CLAUDE.md M10).

Hand-labelled checks need eval/app_annotation/*.csv to be filled by the team; --dry-run
reports which columns are still empty. Checks that need only model outputs and EduRABSA
gold labels run as soon as job 10 has output. No check trains or tunes the model itself:
domain thresholds are fit on 150 clauses and reported on the other 150.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from emoldgn.evaluate import f1_binary  # noqa: E402
from emoldgn.thresholds import best_t, make_grid  # noqa: E402
from spark_jobs.common import EMOS, load_config, path_of  # noqa: E402

TEMPLATES = {"customer_support": "support_300.csv", "student_feedback": "feedback_300.csv"}
LABELS = [f"{e}_a1" for e in EMOS] + ["aspect_a1", "annotator_1"] + [f"{e}_a2" for e in EMOS] + ["aspect_a2", "annotator_2"]


def template(profile: str) -> pd.DataFrame | None:
    p = ROOT / "eval/app_annotation" / TEMPLATES[profile]
    return pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else None


def missing_columns(df: pd.DataFrame) -> dict[str, int]:
    """Number of blank cells per label column; empty dict means fully labelled."""
    return {c: int((df[c].str.strip() == "").sum()) for c in LABELS if (df[c].str.strip() == "").any()}


def kappa(df: pd.DataFrame) -> dict[str, float]:
    """Cohen's kappa per emotion between the two annotators."""
    return {e: float(cohen_kappa_score(df[f"{e}_a1"].astype(int), df[f"{e}_a2"].astype(int))) for e in EMOS}


def adjudicated(df: pd.DataFrame) -> np.ndarray:
    """A label counts when both annotators chose it."""
    return np.stack([(df[f"{e}_a1"].astype(int) & df[f"{e}_a2"].astype(int)).to_numpy() for e in EMOS], 1)


def domain_scores(cfg: dict[str, Any], profile: str, df: pd.DataFrame) -> dict[str, Any]:
    """Macro-F1 with BRIGHTER thresholds versus domain thresholds, fit 150 / test 150."""
    clauses = pd.read_parquet(path_of(cfg, "app_gold", profile, "clauses"), columns=["doc_id", "clause_id", "probs", "present"])
    df = df.assign(clause_id=df["clause_id"].astype(int))
    m = df.merge(clauses, on=["doc_id", "clause_id"], how="left")
    if m["probs"].isna().any():
        raise ValueError("some annotated clauses have no model output; rerun job 09 on the same demo sample")
    y = adjudicated(m)
    p = np.stack(m["probs"].to_numpy())[:, :6]
    base = np.stack(m["present"].to_numpy())
    idx = np.random.default_rng(int(cfg["seed"])).permutation(len(m))
    fit, test = idx[:150], idx[150:]
    grid = make_grid(cfg["thresholds"]["grid_step"])
    t = np.array([best_t(p[fit, c], y[fit, c], grid) for c in range(6)])
    return {"macro_f1_brighter_thresholds": float(f1_binary(y[test], base[test]).mean()),
            "macro_f1_domain_thresholds": float(f1_binary(y[test], (p[test] >= t).astype(int)).mean()),
            "domain_thresholds": dict(zip(EMOS, t.round(3).tolist())), "test_clauses": len(test)}


def aspect_precision(cfg: dict[str, Any], profile: str, df: pd.DataFrame) -> dict[str, float]:
    """Keyword tagger against adjudicated hand labels: precision and coverage."""
    clauses = pd.read_parquet(path_of(cfg, "app_gold", profile, "clauses"), columns=["doc_id", "clause_id", "aspects"])
    m = df.assign(clause_id=df["clause_id"].astype(int)).merge(clauses, on=["doc_id", "clause_id"], how="left")
    agreed = m[m["aspect_a1"].str.strip() == m["aspect_a2"].str.strip()]
    tagged = agreed[agreed["aspects"].map(lambda a: "other" not in list(a))]
    hit = sum(row.aspect_a1.strip() in list(row.aspects) for row in tagged.itertuples())
    return {"precision": hit / max(len(tagged), 1), "coverage": len(tagged) / max(len(agreed), 1)}


def valence_check(cfg: dict[str, Any], source: str = "edurabsa") -> dict[str, float] | None:
    """Spearman of V(x) against EduRABSA sentiment or Coursera stars (plan 9.3, 9.6)."""
    path = Path(path_of(cfg, "app_gold", "student_feedback", "documents"))
    if not path.exists():
        return None
    d = pd.read_parquet(path, columns=["source", "valence", "gold_sentiment"]).dropna()
    d = d[d["source"].str.startswith(source)]
    if len(d) < 3:
        return None
    r = stats.spearmanr(d["valence"], d["gold_sentiment"])
    return {"spearman": float(r.statistic), "p": float(r.pvalue), "n": len(d)}


def aspect_gold_check(cfg: dict[str, Any]) -> dict[str, float] | None:
    """Keyword aspects against EduRABSA gold aspects, per document (plan 9.6)."""
    path = Path(path_of(cfg, "app_gold", "student_feedback", "clauses"))
    if not path.exists():
        return None
    c = pd.read_parquet(path, columns=["doc_id", "aspects", "gold_aspects"])
    per_doc = c.groupby("doc_id").agg(
        pred=("aspects", lambda s: {x for a in s for x in a} - {"other"}),
        gold=("gold_aspects", lambda s: set(s.iloc[0]) - {"other"}),
    )
    tp = sum(len(a & b) for a, b in zip(per_doc["pred"], per_doc["gold"]))
    n_pred, n_gold = sum(map(len, per_doc["pred"])), sum(map(len, per_doc["gold"]))
    return {"precision": tp / max(n_pred, 1), "recall": tp / max(n_gold, 1), "documents": len(per_doc)}


def make_review_sheets(cfg: dict[str, Any], per_state: int = 100, top: int = 50) -> list[Path]:
    """Sheets for the checks that need a scored corpus: state precision and Precision@50."""
    out_dir = ROOT / "eval/app_annotation"
    written = []
    for profile in TEMPLATES:
        path = Path(path_of(cfg, "app_gold", profile, "clauses"))
        if not path.exists():
            print(f"{profile}: no scored clauses yet")
            continue
        c = pd.read_parquet(path, columns=["doc_id", "clause", "states", "urgency"])
        rows = c.explode("states").dropna(subset=["states"]).rename(columns={"states": "state"})
        picks = [g.sample(min(per_state, len(g)), random_state=int(cfg["seed"])) for _, g in rows.groupby("state")]
        sheet = out_dir / f"states_{profile}.csv"
        if picks and not sheet.exists():
            pd.concat(picks)[["state", "doc_id", "clause"]].assign(correct="").to_csv(sheet, index=False)
            written.append(sheet)
    path = Path(path_of(cfg, "app_gold", "customer_support", "documents"))
    sheet = out_dir / "queue_top50.csv"
    if path.exists() and not sheet.exists():
        d = pd.read_parquet(path, columns=["doc_id", "urgency"]).sort_values("urgency", ascending=False)
        top_rows = d.head(top).assign(sample="top")
        rand = d.iloc[top:].sample(min(top, len(d) - top), random_state=int(cfg["seed"])).assign(sample="random")
        # Shuffled, with urgency left out, so annotators are not led by the model's ranking.
        both = pd.concat([top_rows, rand]).sample(frac=1, random_state=int(cfg["seed"]))
        both[["doc_id", "sample"]].assign(needs_attention="").to_csv(sheet, index=False)
        written.append(sheet)
    return written


def share_correct(values: pd.Series) -> float:
    return float(values.astype(int).mean())


def review_scores() -> dict[str, Any]:
    """State precision and Precision@50 versus a random baseline, from filled sheets."""
    out: dict[str, Any] = {}
    d = ROOT / "eval/app_annotation"
    for profile in TEMPLATES:
        p = d / f"states_{profile}.csv"
        if p.exists():
            sheet = pd.read_csv(p, dtype=str, keep_default_na=False)
            done = sheet[sheet["correct"].str.strip() != ""]
            if len(done):
                out[f"state_precision_{profile}"] = {k: share_correct(g["correct"]) for k, g in done.groupby("state")}
    p = d / "queue_top50.csv"
    if p.exists():
        sheet = pd.read_csv(p, dtype=str, keep_default_na=False)
        done = sheet[sheet["needs_attention"].str.strip() != ""]
        if len(done):
            out["precision_at_50"] = {k: share_correct(g["needs_attention"]) for k, g in done.groupby("sample")}
    return out


def scaling(cfg: dict[str, Any], workers: list[int], sample: int) -> list[dict[str, float]]:
    """Clauses per minute versus local[n], written to a separate table (plan 9.6)."""
    import subprocess

    rows = []
    for n in workers:
        t0 = time.perf_counter()
        subprocess.run([sys.executable, str(ROOT / "spark_jobs/09_score.py"), "--profile", "customer_support", "--stage", "score",
                        "--master", f"local[{n}]", "--parallel", str(n), "--table", "_scaling", "--sample", str(sample), "--force"],
                       cwd=ROOT, check=True)
        minutes = (time.perf_counter() - t0) / 60
        rows.append({"workers": n, "clauses": sample, "clauses_per_minute": sample / minutes})
    return rows


def latency(url: str, n: int = 100) -> dict[str, float]:
    """p50 and p95 of /analyze against a running server (plan 9.6)."""
    import urllib.request

    body = json.dumps({"text": "I enjoyed the course, but the examination was stressful and confusing",
                       "profile": "student_feedback"}).encode()
    times = []
    for _ in range(n):
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            r.read()
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report which label columns still need filling")
    ap.add_argument("--latency-url", default=None, help="for example http://127.0.0.1:8000/analyze")
    ap.add_argument("--make-review-sheets", action="store_true", help="write state and top-50 sheets from scored data")
    ap.add_argument("--scaling", nargs="*", type=int, help="Spark workers to time, for example --scaling 1 2 4")
    ap.add_argument("--scaling-sample", type=int, default=2000)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.make_review_sheets:
        print("wrote", make_review_sheets(cfg))
        return 0
    report: dict[str, Any] = {}
    ready = {}
    for profile in TEMPLATES:
        df = template(profile)
        if df is None:
            print(f"{profile}: template missing; run scripts/make_annotation_templates.py")
            continue
        gaps = missing_columns(df)
        ready[profile] = df if not gaps else None
        if gaps:
            print(f"{profile}: {len(df)} clauses; {len(gaps)} of {len(LABELS)} label columns still need labels: {sorted(gaps)}")
        else:
            print(f"{profile}: all label columns are filled")
    report["aspect_tagger_vs_edurabsa"] = aspect_gold_check(cfg)
    print("keyword aspects vs EduRABSA gold:", report["aspect_tagger_vs_edurabsa"])
    if args.dry_run:
        return 0
    report["valence_vs_edurabsa"] = valence_check(cfg)
    report["coursera_valence"] = valence_check(cfg, "coursera")
    report.update(review_scores())
    for profile, df in ready.items():
        if df is not None:
            report[profile] = {"kappa": kappa(df), "domain": domain_scores(cfg, profile, df),
                               "aspects": aspect_precision(cfg, profile, df)}
    if args.latency_url:
        report["latency"] = latency(args.latency_url)
    if args.scaling:
        report["scaling"] = scaling(cfg, args.scaling, args.scaling_sample)
    out = Path(path_of(cfg, "reports", "app_eval.json"))
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
