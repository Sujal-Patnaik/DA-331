"""The one place where the dashboards call the EmoLDGN+ engine.

The upload tab calls ``analyze`` and draws whatever comes back. Scoring follows the same steps as the
stored tables in ``apps/gold``:

1. clean and mask the text (the training-side cleaner),
2. detect the language and split each message into clauses,
3. score every clause with the exported model in ``artifacts/`` (emotions, intensities, trigger words),
4. derive labels, states, urgency and valence with the aggregation job's own function, and
5. roll clauses up to messages with the max rule.

Output: ``(documents, clauses)``, the two tables described in ``data.prepare``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from spark_jobs.common import EMOS, ROOT, load_config, path_of

NEGATIVE_FOR_MIXED = ("anger", "disgust", "fear", "sadness")
CHUNK = 256                                    # clauses per scoring call, so the progress bar moves


class BackendNotConnected(RuntimeError):
    """Raised when the exported model is missing; the UI turns it into a friendly message."""


def _artifacts() -> Path:
    return Path(path_of(load_config(), "artifacts"))


def is_connected(profile: str) -> bool:
    """Cheap check for the status badge: the exported model and its files are present."""
    art = _artifacts()
    return all((art / f).exists() for f in ("best.pt", "thresholds.json", "lang_index.json"))


def _short_ids(raw: pd.Series | None, n: int) -> list[str]:
    """Readable message ids: the file's own ids when short, otherwise a row number."""
    if raw is None:
        return [str(i + 1) for i in range(n)]
    vals = raw.fillna("").astype(str).str.strip()
    if vals.str.len().max() > 16 or vals.duplicated().any() or (vals == "").any():
        return [str(i + 1) for i in range(n)]
    return vals.tolist()


def analyze(df: pd.DataFrame, profile: str, text_col: str, id_col: str | None = None, group_col: str | None = None,
            period_col: str | None = None, progress: Callable[[float, str], None] | None = None
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not is_connected(profile):
        raise BackendNotConnected("artifacts/best.pt is missing; export a model first")

    from apps.engine.aspects import load_profile, tag_aspects
    from apps.engine.langid import detect
    from apps.engine.scorer import get_scorer
    from apps.engine.segment import segment
    from apps.engine.signals import should_escalate
    from spark_jobs.textnorm import clean_text

    aggregate = importlib.import_module("spark_jobs.10_aggregate")
    say = progress or (lambda f, m: None)
    cfg = load_config()
    prof = load_profile(ROOT / "profiles" / f"{profile}.yaml")
    n = len(df)

    say(0.02, "Reading your file")
    meta = pd.DataFrame({
        "doc_id": _short_ids(df[id_col] if id_col else None, n),
        "group_id": df[group_col].fillna("(blank)").astype(str).values if group_col else "All uploaded rows",
        "period": df[period_col].fillna("(blank)").astype(str).values if period_col else "all",
        "text_raw": df[text_col].fillna("").astype(str).values,
    })
    meta["text"] = [clean_text(t) or "" for t in meta["text_raw"]]
    meta = meta[meta["text"].str.strip() != ""].reset_index(drop=True)
    if meta.empty:
        raise ValueError("No readable text found in that column.")

    say(0.06, "Detecting languages")
    meta["lang"] = detect(meta["text"].tolist(), path_of(cfg, "lid_model"))

    rows = [(i, j, c) for i, t in enumerate(meta["text"]) for j, c in enumerate(segment(t))]
    if not rows:
        raise ValueError("No readable sentences found in that column.")
    cl = pd.DataFrame(rows, columns=["doc_idx", "clause_id", "clause"])
    cl["lang"] = meta["lang"].values[cl["doc_idx"].values]
    say(0.10, f"Split {len(meta):,} messages into {len(cl):,} sentences")

    say(0.12, "Loading the model")
    a = cfg["apps"]
    scorer = get_scorer(_artifacts(), batch_size=int(a["score_batch_size"]), max_len=int(a["max_len"]),
                        top_triggers=int(a["top_triggers"]))
    probs, inten, triggers = [], [], []
    for start in range(0, len(cl), CHUNK):
        part = cl.iloc[start:start + CHUNK]
        r = scorer.score(part["clause"].tolist(), part["lang"].tolist())
        probs.extend(r["probs"])
        inten.extend(r["inten"])
        triggers.extend(r["triggers"])
        say(0.12 + 0.76 * min(1.0, (start + CHUNK) / len(cl)), "Reading each sentence")
    cl["probs"], cl["inten"], cl["triggers"] = probs, inten, triggers

    say(0.90, "Working out moods, urgency and topics")
    derived = aggregate.derive_clause(cfg, prof, cl[["clause", "lang", "probs", "inten"]])
    cl = pd.concat([cl, derived], axis=1)
    cl["aspects"] = [tag_aspects(c, prof) for c in cl["clause"]]
    for col in ("doc_id", "group_id", "period"):
        cl[col] = meta[col].values[cl["doc_idx"].values]

    docs = _roll_up(meta, cl, prof, should_escalate)
    docs["text"] = meta["text_raw"].values
    clauses = cl[["doc_id", "group_id", "period", "lang", "clause_id", "clause", "aspects", "probs", "inten", "triggers",
                  "present", "states", "urgency", "valence", "arousal", "cues"]].copy()
    say(1.0, "Done")
    return docs, clauses


def _roll_up(meta: pd.DataFrame, cl: pd.DataFrame, prof: dict, should_escalate) -> pd.DataFrame:
    """Clauses to messages with the max rule, as in the aggregation job."""
    di = cl["doc_idx"].values
    nd = len(meta)
    present = np.stack(cl["present"].values).astype(int)
    probs = np.stack(cl["probs"].values).astype(float)
    inten = np.stack(cl["inten"].values).astype(float)

    def doc_max(arr: np.ndarray) -> np.ndarray:
        out = np.full((nd,) + arr.shape[1:], -np.inf)
        np.maximum.at(out, di, arr)
        return np.where(np.isfinite(out), out, 0)

    cnt = np.bincount(di, minlength=nd)
    safe = np.maximum(cnt, 1)
    doc = meta[["doc_id", "group_id", "period", "lang"]].copy()
    pmax, imax, hmax = doc_max(probs[:, :6]), doc_max(inten), doc_max(present).astype(int)
    for k, e in enumerate(EMOS):
        doc[f"p_{e}"], doc[f"i_{e}"], doc[f"has_{e}"] = pmax[:, k], imax[:, k], hmax[:, k]
    doc["urgency"] = doc_max(cl["urgency"].values.astype(float))
    doc["valence"] = np.bincount(di, weights=cl["valence"].values, minlength=nd) / safe
    doc["arousal"] = np.bincount(di, weights=cl["arousal"].values, minlength=nd) / safe
    doc["n_clauses"] = cnt

    groups = cl.groupby("doc_idx")
    union = lambda col: groups[col].apply(lambda s: sorted({x for l in s for x in l})).reindex(range(nd))  # noqa: E731
    doc["states"], doc["aspects"], doc["cues"] = union("states").values, union("aspects").values, union("cues").values
    neg = np.max([doc[f"has_{e}"] for e in NEGATIVE_FOR_MIXED], axis=0)
    doc["mixed"] = (doc["has_joy"] == 1) & (neg == 1)
    if "mixed" in (prof.get("states") or []):
        doc["states"] = [s + ["mixed"] if m and "mixed" not in s else s for s, m in zip(doc["states"], doc["mixed"])]
    doc["escalate"] = [bool(should_escalate(prof, float(u), set(s), set(a)))
                       for u, s, a in zip(doc["urgency"], doc["states"], doc["aspects"])]
    return doc
