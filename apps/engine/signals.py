"""Derived signals on top of the model's outputs (plan 9.3).

States are heuristic mappings onto BRIGHTER's six emotions, to be validated on the
hand-labelled samples of plan 9.6. All functions take arrays for N clauses.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ANGER, DISGUST, FEAR, JOY, SADNESS, SURPRISE = range(6)
NEGATIVE = (ANGER, DISGUST, FEAR, SADNESS)
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
TONE = {
    "frustration": "Acknowledge the repeated effort; give a concrete next step and a time.",
    "outrage": "Apologise without excuses; offer escalation to a supervisor.",
    "stress": "Reassure first; explain what is already secured (account, payment).",
    "disappointment": "Empathise; offer a specific remedy.",
    "delight": "Thank them; invite a review.",
}


def urgency(p: np.ndarray, inten: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """U = max over a language's negative emotions of p_c * E[I_c] / 3 (plan 9.3).

    ``mask`` (N x 6) is required: an emotion a language lacks is never trained for it, so its
    output is noise (rule 5). English disgust once set a third of English urgency scores.
    """
    idx = list(NEGATIVE)
    return np.clip((p[:, idx] * mask[:, idx] * inten[:, idx] / 3.0).max(1), 0.0, 1.0)


def load_vad(lexicon_dir: str | Path) -> np.ndarray | None:
    """6 x 2 valence and arousal of the emotion words; None if the lexicon is absent."""
    files = sorted(Path(lexicon_dir).rglob("NRC-VAD-Lexicon-v2.1.txt"))
    if not files:
        return None
    rows: dict[str, tuple[float, float]] = {}
    with open(files[0], encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0] in EMOTIONS:
                rows[parts[0]] = (float(parts[1]), float(parts[2]))
    return np.array([rows[e] for e in EMOTIONS], dtype=np.float32) if len(rows) == 6 else None


def vad_summary(p: np.ndarray, inten: np.ndarray, vad: np.ndarray | None,
                mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted valence and arousal over a language's emotions; no weight means the centre."""
    if vad is None:
        return np.zeros(len(p)), np.zeros(len(p))
    w = p * mask * (1.0 + inten)
    total = w.sum(1)
    safe = np.where(total > 1e-9, total, 1.0)
    return (w * vad[:, 0]).sum(1) / safe, (w * vad[:, 1]).sum(1) / safe


def center_neutral(v: np.ndarray, a: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Texts with no detected emotion sit at the centre (plan 9.3), however small their weights."""
    has = present.max(1) > 0
    return v * has, a * has


def app_thresholds(t: np.ndarray, langs: Sequence[str], overrides: dict[str, dict[str, float]] | None) -> np.ndarray:
    """Per-language threshold overrides for the applications only; BRIGHTER scores do not use these."""
    t = t.copy()
    for k, lang in enumerate(langs):
        for emotion, value in (overrides or {}).get(lang, {}).items():
            t[k, EMOTIONS.index(emotion)] = value
    return t


def clause_states(inten: np.ndarray, present: np.ndarray, mask: np.ndarray, anger_cut: float) -> list[list[str]]:
    """States per clause, before the profile selects which ones it shows.

    Replaces plan 9.3's pair score: the trained dynamic graph is near 0 exactly where both
    emotions are high, so that rule could not fire. Pair states now need both emotions, and
    single-emotion states need theirs to be the strongest negative one.
    """
    on = present > 0
    a, d, f, j, s, u = (on[:, c] for c in range(6))
    neg = list(NEGATIVE)
    strongest = np.array(neg)[(inten[:, neg] * mask[:, neg]).argmax(1)]
    frustration = a & s
    outrage = np.where(mask[:, DISGUST] > 0, a & d, a & (inten[:, ANGER] >= anger_cut))
    stress = f & (strongest == FEAR) & ~frustration & ~outrage
    disappointment = s & ~a & (strongest == SADNESS)
    delight = j & u & ~(a | d | f | s)
    flags = {"frustration": frustration, "outrage": outrage, "stress": stress, "disappointment": disappointment,
             "delight": delight}
    return [[name for name, hit in flags.items() if hit[k]] for k in range(len(inten))]


def is_mixed(present: np.ndarray) -> bool:
    """Joy together with any negative emotion in one document (present: clauses x 6)."""
    doc = present.max(0)
    return bool(doc[JOY] > 0 and doc[list(NEGATIVE)].max() > 0)


_COND = re.compile(
    r"^\s*(?:urgency\s*(?P<op>>=|>)\s*(?P<num>[\d.]+)"
    r"|state\s*==\s*(?P<state>\w+)"
    r"|aspect\s+in\s*\[(?P<aspects>[^\]]*)\])\s*$"
)


def parse_rule(rule: str) -> Callable[[float, set[str], set[str]], bool]:
    """Compile a profile rule such as ``state == stress and aspect in [refund, app_site]``.

    Only this small grammar is accepted, so a profile can never run arbitrary code.
    """
    tests: list[Callable[[float, set[str], set[str]], bool]] = []
    for part in re.split(r"\s+and\s+", rule.strip()):
        m = _COND.match(part)
        if not m:
            raise ValueError(f"cannot parse escalation rule: {rule!r}")
        if m["num"]:
            num, strict = float(m["num"]), m["op"] == ">"
            tests.append(lambda u, s, a, n=num, st=strict: u > n if st else u >= n)
        elif m["state"]:
            tests.append(lambda u, s, a, x=m["state"]: x in s)
        else:
            names = {x.strip() for x in m["aspects"].split(",")}
            tests.append(lambda u, s, a, x=names: bool(x & a))
    return lambda u, s, a: all(t(u, s, a) for t in tests)


def should_escalate(profile: dict[str, Any], u: float, states: set[str], aspects: set[str]) -> bool:
    """True when any of the profile's rules fires (plan 9.4)."""
    return any(parse_rule(r)(u, states, aspects) for r in profile.get("escalate_if") or [])
