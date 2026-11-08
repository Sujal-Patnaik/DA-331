"""FastAPI single-text analysis (plan 9.7, CLAUDE.md M9).

Run: uvicorn apps.api.server:app --port 8000
POST /analyze {"text": "...", "profile": "student_feedback", "lang": null}

Results describe how a message reads, not how a person feels (plan 9.8).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from apps.engine import signals as S
from apps.engine.aspects import cue_hits, load_profile, tag_aspects
from apps.engine.langid import detect
from apps.engine.scorer import Scorer
from apps.engine.segment import segment
from emoldgn.thresholds import apply_thresholds
from spark_jobs.common import ROOT, load_config, path_of
from spark_jobs.textnorm import clean_text

CFG = load_config()
STATE: dict[str, Any] = {}


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    profile: str = "customer_support"
    lang: str | None = None


def startup() -> None:
    """Load the model once. A missing artifact leaves /analyze answering 503."""
    art = Path(os.environ.get("EMOLDGN_ARTIFACTS", path_of(CFG, "artifacts")))
    if (art / "best.pt").exists():
        STATE["scorer"] = Scorer(art, batch_size=int(CFG["apps"]["score_batch_size"]), max_len=int(CFG["apps"]["max_len"]),
                                 top_triggers=int(CFG["apps"]["top_triggers"]))
    STATE["vad"] = S.load_vad(path_of(CFG, "lexicons"))
    STATE["profiles"] = {p.stem: load_profile(p) for p in (ROOT / "profiles").glob("*.yaml")}


@asynccontextmanager
async def lifespan(_: FastAPI):
    startup()
    yield


app = FastAPI(title="EmoLDGN+ analysis API", lifespan=lifespan)


def analyse(q: Query) -> dict[str, Any]:
    """Clause and document results for one text, following plan 6.2 and 9.3."""
    scorer: Scorer = STATE["scorer"]
    profile = STATE["profiles"].get(q.profile)
    if profile is None:
        raise HTTPException(404, f"unknown profile {q.profile!r}")
    text = clean_text(q.text) or ""
    clauses = segment(text)
    if not clauses:
        raise HTTPException(422, "no text left after cleaning")
    lang = q.lang or detect([text], path_of(CFG, "lid_model"))[0]
    langs = [lang] * len(clauses)
    r = scorer.score(clauses, langs)
    p, inten = r["probs"], r["inten"]
    t = S.app_thresholds(scorer.thresholds(langs), langs, CFG["apps"]["threshold_overrides"])
    mask = scorer.mask(langs)
    present = apply_thresholds(p[:, :6], t, mask, p[:, 6] if scorer.has_neutral else None)
    states = S.clause_states(inten, present, mask, float(CFG["apps"]["outrage_anger_intensity"]))
    keep = set(profile.get("states") or [])
    u = S.urgency(p, inten, mask)
    v, a = S.center_neutral(*S.vad_summary(p[:, :6], inten, STATE["vad"], mask), present)
    rows = []
    for k, clause in enumerate(clauses):
        rows.append({
            "clause": clause, "aspects": tag_aspects(clause, profile),
            "emotions": {e: round(float(p[k, c]), 4) for c, e in enumerate(S.EMOTIONS) if mask[k, c] > 0},
            "present": [e for c, e in enumerate(S.EMOTIONS) if present[k, c]],
            "intensities": {e: round(float(inten[k, c]), 3) for c, e in enumerate(S.EMOTIONS) if mask[k, c] > 0},
            "states": [s for s in states[k] if s in keep], "urgency": round(float(u[k]), 4),
            "triggers": r["triggers"][k],
        })
    doc_states = sorted({s for row in rows for s in row["states"]})
    if "mixed" in keep and S.is_mixed(present):
        doc_states.append("mixed")
    aspects = sorted({x for row in rows for x in row["aspects"]})
    doc_u = float(u.max())
    return {
        "language": lang, "reads_as": "Perceived emotion in the text, not how the writer truly feels.",
        "clauses": rows, "emotions": {e: round(float(p[:, c].max()), 4) for c, e in enumerate(S.EMOTIONS) if mask[0, c] > 0},
        "intensities": {e: round(float(inten[:, c].max()), 3) for c, e in enumerate(S.EMOTIONS) if mask[0, c] > 0},
        "present": [e for c, e in enumerate(S.EMOTIONS) if present[:, c].max() > 0],
        "states": doc_states, "aspects": aspects, "urgency": round(doc_u, 4),
        "valence": round(float(np.mean(v)), 4), "arousal": round(float(np.mean(a)), 4),
        "cues": cue_hits(text, profile), "escalate": S.should_escalate(profile, doc_u, set(doc_states), set(aspects)),
        "suggested_tone": [S.TONE[s] for s in doc_states if s in S.TONE],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": "scorer" in STATE}


@app.post("/analyze")
def post_analyze(q: Query) -> dict[str, Any]:
    if "scorer" not in STATE:
        raise HTTPException(503, "artifacts/best.pt is missing; finish M8 or export a model first")
    return analyse(q)
