"""Results tables in, display-ready tables out.

``Dataset`` wraps the two tables every view reads (one row per message, one row per sentence). The same
class holds the stored test-data results and whatever a backend returns for an uploaded file, so both are
drawn by identical views. See ``backend.py`` for the exact columns a backend must return.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from apps.dashboards.labels import EMOTIONS, NEGATIVE, STATE_PRIORITY, STATES

ROOT = Path(__file__).resolve().parents[1]
# position of each shown emotion inside the 6-wide arrays (anger, disgust, fear, joy, sadness, surprise)
EMO_POS = {"anger": 0, "fear": 2, "joy": 3, "sadness": 4, "surprise": 5}


@dataclass
class Dataset:
    name: str
    key: str                       # cache identity
    docs: pd.DataFrame
    clauses: pd.DataFrame
    ca: pd.DataFrame = field(repr=False)   # one row per (sentence, topic)
    has_gold: bool = False

    @property
    def periods(self) -> list[str]:
        return list(period_order(self.docs["period"].astype(str).unique()))


HF = {Dataset: lambda d: d.key}       # st.cache_data: hash a Dataset by its key only


def _list(x) -> list:
    if x is None:
        return []
    if isinstance(x, float):
        return []
    return list(x)


def period_order(vals) -> list[str]:
    vals = [str(v) for v in vals]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = pd.to_datetime(pd.Series(vals), errors="raise")
        return [v for _, v in sorted(zip(ts, vals))]
    except Exception:
        return sorted(vals, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])


def prepare(docs: pd.DataFrame, clauses: pd.DataFrame, key: str, name: str) -> Dataset:
    d = docs.copy().reset_index(drop=True)
    for col, default in (("group_id", "All rows"), ("period", "all"), ("lang", "<unk>")):
        if col not in d:
            d[col] = default
    d["group_id"] = d["group_id"].astype(str)
    d["period"] = d["period"].astype(str)
    for col in ("states", "aspects", "cues"):
        d[col] = d[col].apply(_list) if col in d else [[] for _ in range(len(d))]
    for s in STATES:
        d[f"st_{s}"] = d["states"].apply(lambda l, s=s: s in l)
    d["primary"] = d["states"].apply(lambda l: next((s for s in STATE_PRIORITY if s in l), "none"))
    d["tone"] = np.where(d["valence"] > 0.15, "Positive", np.where(d["valence"] < -0.15, "Negative", "Neutral"))
    d["is_pos"], d["is_neg"] = d["tone"] == "Positive", d["tone"] == "Negative"
    d["escalate"] = d["escalate"].astype(bool) if "escalate" in d else False
    d["mixed"] = d["mixed"].astype(bool) if "mixed" in d else d["st_mixed"]
    d["kind"] = d["group_id"].str.replace(r"_?\d+$", "", regex=True)
    has_gold = "gold_sentiment" in d and d["gold_sentiment"].notna().any()

    c = clauses.copy().reset_index(drop=True)
    pres = np.stack(c["present"].values).astype(int).reshape(len(c), -1)
    probs = np.stack(c["probs"].values).astype(np.float32).reshape(len(c), -1)
    for e, pos in EMO_POS.items():
        c[f"e_{e}"] = pres[:, pos] > 0
        c[f"p_{e}"] = probs[:, pos]
    c["neg"] = c[[f"e_{e}" for e in NEGATIVE]].any(axis=1)
    strength = np.stack([np.where(c[f"e_{e}"], c[f"p_{e}"], 0.0) for e in EMOTIONS], axis=1)
    c["score"] = strength.max(1)
    c["top_emotion"] = np.where(strength.max(1) > 0, np.array(EMOTIONS)[strength.argmax(1)], "")
    for col in ("aspects", "states", "triggers"):
        c[col] = c[col].apply(_list) if col in c else [[] for _ in range(len(c))]
    if "urgency" not in c:
        c["urgency"] = 0.0
    if "valence" not in c:
        c["valence"] = 0.0
    keep = ["doc_id", "group_id", "clause_id", "clause", "aspects", "states", "triggers", "urgency", "valence",
            "score", "top_emotion", "neg"] + [f"e_{e}" for e in EMOTIONS] + [f"p_{e}" for e in EMOTIONS]
    c = c[keep]
    ca = c.assign(ci=np.arange(len(c))).explode("aspects").rename(columns={"aspects": "aspect"})
    ca = ca[["ci", "doc_id", "group_id", "aspect", "valence", "score", "neg"] + [f"e_{e}" for e in EMOTIONS]]
    return Dataset(name, key, d, c, ca, has_gold)


@st.cache_resource(show_spinner="Loading results…")
def test_results(profile: str) -> Dataset:
    """Stored results on the test data: the demo datasets scored by the EmoLDGN+ model, which never trained on them.

    The model was trained on BRIGHTER only. These tables come from spark_jobs/09_score.py and 10_aggregate.py.
    """
    g = ROOT / "gold" / profile
    docs = pd.read_parquet(g / "documents", columns=[
        "doc_id", "group_id", "period", "lang", "urgency", "valence", "arousal", "gold_sentiment", "gold_aspects",
        "states", "aspects", "cues", "n_clauses", "mixed", "escalate"])
    cl = pd.read_parquet(g / "clauses", columns=[
        "doc_id", "group_id", "clause_id", "clause", "aspects", "probs", "present", "triggers", "states", "urgency", "valence"])
    return prepare(docs, cl, key=f"test:{profile}", name="Test data")


@st.cache_data(show_spinner=False)
def sample_csv(profile: str, n: int = 40) -> bytes:
    """A small file in exactly the format the app was built on, cut from the test inputs."""
    b = pd.read_parquet(ROOT / "bronze" / profile / "documents")
    b = b[b["text_raw"].str.len().between(40, 400)].sample(n, random_state=3)
    # No id column: the stored ids are long hashes, and the app numbers rows itself.
    cols = ["group_id", "period", "text_raw"]
    return b[cols].to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------------ aggregates
@st.cache_data(show_spinner=False, hash_funcs=HF)
def kpis(ds: Dataset) -> dict:
    d = ds.docs
    return {
        "n": len(d), "esc": float(d["escalate"].mean()), "urgency": float(d["urgency"].mean()),
        "valence": float(d["valence"].mean()), "groups": int(d["group_id"].nunique()), "mixed": float(d["mixed"].mean()),
        "n_clauses": len(ds.clauses), "confusion": int(d["cues"].apply(lambda l: "confusion" in l).sum()),
        "pos": float((d["tone"] == "Positive").mean()), "neu": float((d["tone"] == "Neutral").mean()),
        "neg": float((d["tone"] == "Negative").mean()),
        "no_mood": float((d["primary"] == "none").mean()),
    }


@st.cache_data(show_spinner=False, hash_funcs=HF)
def emotion_share(ds: Dataset) -> dict[str, float]:
    return {e: float(ds.clauses[f"e_{e}"].mean()) for e in EMOTIONS}


@st.cache_data(show_spinner=False, hash_funcs=HF)
def state_share(ds: Dataset) -> dict[str, float]:
    return {s: float(ds.docs[f"st_{s}"].mean()) for s in STATES}


@st.cache_data(show_spinner=False, hash_funcs=HF)
def aspect_table(ds: Dataset, group: str | None = None) -> pd.DataFrame:
    ca = ds.ca if group is None else ds.ca[ds.ca["group_id"] == group]
    if ca.empty:
        return pd.DataFrame()
    g = ca.groupby("aspect")
    t = g.size().rename("n").to_frame()
    for e in EMOTIONS:
        t[e] = g[f"e_{e}"].mean()
    t["neg"] = g["neg"].mean()
    t["pos"] = t["joy"]
    t["net"] = t["pos"] - t["neg"]          # share sounding happy minus share sounding unhappy
    return t.sort_values("n", ascending=False)


@st.cache_data(show_spinner=False, hash_funcs=HF)
def group_table(ds: Dataset) -> pd.DataFrame:
    d = ds.docs
    g = d.groupby("group_id")
    t = g.size().rename("n").to_frame()
    t["esc"] = g["escalate"].mean()
    t["urgency"] = g["urgency"].mean()
    t["valence"] = g["valence"].mean()
    t["mixed"] = g["mixed"].mean()
    t["pos"] = g["is_pos"].mean()
    t["neg"] = g["is_neg"].mean()
    for s in STATES:
        t[s] = g[f"st_{s}"].mean()
    t["kind"] = g["kind"].first()
    t["top_mood"] = t[[s for s in STATES if s != "mixed"]].idxmax(axis=1)
    return t


@st.cache_data(show_spinner=False, hash_funcs=HF)
def trend_table(ds: Dataset, state: str) -> pd.DataFrame:
    """period x topic: of the messages touching that topic, the share that show ``state`` (or all of them)."""
    d = ds.docs[["period", "aspects"]].copy()
    d["hit"] = ds.docs[f"st_{state}"].values if state != "any" else True
    d = d.explode("aspects").rename(columns={"aspects": "aspect"})
    t = d.groupby(["period", "aspect"]).agg(n=("hit", "size"), share=("hit", "mean")).reset_index()
    order = period_order(t["period"].unique())
    t["period"] = pd.Categorical(t["period"], order, ordered=True)
    t = t.sort_values(["aspect", "period"]).reset_index(drop=True)
    flags = []
    for _, grp in t.groupby("aspect", observed=True):
        prev = grp["share"].shift(1).expanding(min_periods=3)
        mean, std = prev.mean(), prev.std().fillna(0)
        flags.append((grp["share"] > mean + 2 * std.clip(lower=0.01)) & ((grp["share"] - mean) >= 0.05) & (grp["n"] >= 20))
    t["spike"] = pd.concat(flags).reindex(t.index).fillna(False) if flags else False
    return t


@st.cache_data(show_spinner=False, hash_funcs=HF)
def language_table(ds: Dataset) -> pd.DataFrame:
    return ds.docs["lang"].value_counts().rename_axis("lang").reset_index(name="n")


def examples(ds: Dataset, *, group: str | None = None, aspect: str | None = None, emotion: str | None = None,
             n: int = 4) -> pd.DataFrame:
    """Most clear-cut sentences for a slice: strongest model score among those that carry the emotion."""
    if aspect is not None:
        idx = ds.ca[(ds.ca["aspect"] == aspect) & ((ds.ca["group_id"] == group) if group else True)]["ci"].values
        c = ds.clauses.iloc[idx]
    else:
        c = ds.clauses if group is None else ds.clauses[ds.clauses["group_id"] == group]
    if emotion:
        c = c[c[f"e_{emotion}"]].sort_values(f"p_{emotion}", ascending=False)
    else:
        c = c.sort_values("score", ascending=False)
    c = c[c["clause"].str.len().between(25, 260)]
    return c.drop_duplicates("clause").head(n)


def clauses_of(ds: Dataset, doc_id: str) -> pd.DataFrame:
    c = ds.clauses[ds.clauses["doc_id"] == doc_id]
    return c.sort_values("clause_id")
