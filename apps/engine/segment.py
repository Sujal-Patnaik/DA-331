"""Clause segmentation: sentences, then contrast markers (plan 9.1)."""
from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?।])\s+|\n+")
EN = ("but", "however", "although", "though", "yet", "whereas")
HI = ("लेकिन", "परंतु", "किंतु", "मगर", "पर")
MR = ("पण", "परंतु", "मात्र")
_MARK = re.compile(
    r"(?<=[\s,;])(?=(?:" + "|".join(EN) + r")\b)|(?<=\s)(?=(?:" + "|".join(sorted(set(HI + MR), key=len, reverse=True)) + r")\s)",
    re.IGNORECASE,
)
_TRIM = " \t,;:-"


def segment(text: str) -> list[str]:
    """Split text so each clause carries one stance; never returns empty clauses."""
    clauses: list[str] = []
    for sentence in _SENT.split(text or ""):
        for part in _MARK.split(sentence):
            part = part.strip(_TRIM)
            if part:
                clauses.append(part)
    return clauses
