"""Keyword aspect tagging from a domain profile (plan 9.1, 9.2).

Keywords match whole words, with an optional plural for Latin-script terms, so "app" does
not fire on "happy" and "exam" still fires on "exams". One pattern text serves both Python
(tests, API) and Java (Spark ``rlike``), so the two taggers agree.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

OTHER = "other"


def load_profile(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _alternatives(keywords: list[str]) -> str:
    terms = []
    for k in sorted(keywords, key=len, reverse=True):
        body = re.escape(str(k)).replace(r"\ ", r"\s+")
        terms.append(body + "(?:s|es)?" if str(k).isascii() else body)
    return "|".join(terms)


def aspect_pattern(keywords: list[str], java: bool = False) -> str:
    """Case-insensitive whole-word regex; Java needs (?U) for Unicode word characters."""
    return ("(?iU)" if java else "(?i)") + r"(?<!\w)(?:" + _alternatives(keywords) + r")(?!\w)"


def tag_aspects(clause: str, profile: dict[str, Any]) -> list[str]:
    """Aspects of one clause; untagged clauses go to "other"."""
    found = [a for a, kw in profile["aspects"].items() if re.search(aspect_pattern(kw), clause)]
    return found or [OTHER]


def aspects_column(profile: dict[str, Any]):
    """Spark column with the same tagging, so tables can be rebuilt after a keyword change."""
    from pyspark.sql import functions as F

    tags = [F.when(F.col("clause").rlike(aspect_pattern(kw, java=True)), F.lit(a)) for a, kw in profile["aspects"].items()]
    found = F.array_compact(F.array(*tags))
    return F.when(F.size(found) == 0, F.array(F.lit(OTHER))).otherwise(found)


def cue_hits(text: str, profile: dict[str, Any]) -> list[str]:
    """Keyword cues such as confusion, which is not a BRIGHTER emotion (plan 9.3)."""
    return [c for c, kw in (profile.get("cues") or {}).items() if re.search(aspect_pattern(kw), text)]
