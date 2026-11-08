"""Turn the numbers into sentences a manager can read in five seconds."""
from __future__ import annotations

import pandas as pd

from apps.dashboards import data as D
from apps.dashboards.labels import (BLUE, EMO_NAME, GREEN, ORANGE, PURPLE, RED, STATE_COLOR, STATE_ICON, STATE_NAME,
                                    YELLOW, aspect_icon, aspect_name, pct, pretty_group)

MIN_GROUP = 30


def _b(s: str) -> str:
    return f"<b>{s}</b>"


def support(ds: D.Dataset, min_group: int = MIN_GROUP) -> list[dict]:
    k = D.kpis(ds)
    out: list[dict] = []
    at = D.aspect_table(ds)
    at = at[(at.index != "other") & (at["n"] >= 30)]
    if len(at):
        a = at["anger"].idxmax()
        out.append(dict(icon="🔥", color=RED, title=f"{aspect_name(a)} is the angriest topic",
                        text=f"{_b(pct(at.loc[a, 'anger']))} of sentences about it sound angry, against "
                             f"{_b(pct(D.emotion_share(ds)['anger']))} overall."))
    gt = D.group_table(ds)
    big = gt[gt["n"] >= min_group]
    if len(big):
        g = big["esc"].idxmax()
        out.append(dict(icon="🚨", color=ORANGE, title=f"{pretty_group(g)} escalates the most",
                        text=f"{_b(pct(big.loc[g, 'esc']))} of its {int(big.loc[g, 'n']):,} conversations need a person, "
                             f"versus {_b(pct(k['esc']))} across everyone."))
    ss = D.state_share(ds)
    ss.pop("mixed", None)
    top = max(ss, key=ss.get)
    out.append(dict(icon=STATE_ICON[top], color=STATE_COLOR[top], title=f"{STATE_NAME[top]} is the most common mood",
                    text=f"It shows up in {_b(pct(ss[top]))} of conversations. {_b(pct(k['no_mood']))} show no clear mood at all."))
    if len(big):
        h = big["valence"].idxmax()
        out.append(dict(icon="🌟", color=GREEN, title=f"{pretty_group(h)} leaves people happiest",
                        text=f"{_b(pct(big.loc[h, 'pos']))} of its conversations read positive and only "
                             f"{_b(pct(big.loc[h, 'neg']))} read negative."))
    return out[:4]


def feedback(ds: D.Dataset, min_group: int = 10) -> list[dict]:
    k = D.kpis(ds)
    out: list[dict] = []
    at = D.aspect_table(ds)
    at = at[(at.index != "other") & (at["n"] >= 30)]
    if len(at):
        best = at["net"].idxmax()
        rest = at.drop(index=best)
        worst = rest["neg"].idxmax() if len(rest) else best
        out.append(dict(icon="💚", color=GREEN, title=f"Students love: {aspect_name(best).lower()}",
                        text=f"{_b(pct(at.loc[best, 'joy']))} of sentences about it sound happy, "
                             f"the most upbeat topic."))
        if worst != best:
            out.append(dict(icon="😓", color=ORANGE, title=f"Students struggle with: {aspect_name(worst).lower()}",
                            text=f"{_b(pct(at.loc[worst, 'neg']))} of sentences about it sound worried, sad or angry."))
    out.append(dict(icon="🤔", color=STATE_COLOR["mixed"], title="Many comments cut both ways",
                    text=f"{_b(pct(k['mixed']))} praise something and complain about something in the same comment."))
    gt = D.group_table(ds)
    gt = gt[(gt["n"] >= min_group) & (gt["kind"] != "university_reviews")]
    if len(gt):
        h, l = gt["valence"].idxmax(), gt["valence"].idxmin()
        out.append(dict(icon="🏆", color=YELLOW, title=f"Happiest: {pretty_group(h)}",
                        text=f"{_b(pct(gt.loc[h, 'pos']))} of {int(gt.loc[h, 'n'])} comments read positive. "
                             f"Needs attention: {_b(pretty_group(l))} ({pct(gt.loc[l, 'neg'])} negative)."))
    return out[:4]
