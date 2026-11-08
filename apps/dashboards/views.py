"""Every screen of the two apps. One set of views, driven by a small ``Cfg`` per app, drawn from a ``Dataset``.

The same code renders the stored test-data results and the results of an uploaded file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from apps.dashboards import charts as C
from apps.dashboards import data as D
from apps.dashboards import insights as I
from apps.dashboards import theme as T
from apps.dashboards.labels import (BLUE, BLUE_D, EMO_COLOR, EMO_ICON, EMO_NAME, EMOTIONS, GREEN, INK_2, LANG_NAME, ORANGE,
                                    PURPLE, STATE_COLOR, STATE_HELP, STATE_ICON, STATE_NAME, TONE, aspect_icon, aspect_name,
                                    mood_word, pct, pretty_group)

NEUTRAL_GREY = "#C9CED6"
TONE_COLOR = {"Positive": BLUE, "Neutral": NEUTRAL_GREY, "Negative": ORANGE}


@dataclass(frozen=True)
class Cfg:
    profile: str
    doc_one: str
    doc_many: str
    group_one: str
    group_many: str
    escalation: bool
    min_group: int
    states: tuple[str, ...]
    sections: tuple[tuple[str, str], ...]
    boards: tuple[tuple[str, str, str, str], ...] = field(default_factory=tuple)   # label, column, format, colour


# ------------------------------------------------------------------ small helpers
def _fmt(kind: str, v: float) -> str:
    return {"pct": pct(v), "score": f"{v * 100:.0f}/100", "int": f"{int(v):,}"}[kind]


FILLER = set("a an the and or but of to in on at is it its it's this that was were be been are am i im i'm me my we you your "
             "he she they them his her their for with as by so if not no do does did have has had will would can could just very "
             "get got went go from up out about into than then there here what when who how".split())


def highlight(text: str, triggers: list[str]) -> str:
    """Mark the words that drove the reading; filler words are never shown as triggers."""
    s = T.esc(text)
    words = {t.strip(".,!?;:\"'()[]") for t in triggers if t}
    for w in sorted((w for w in words if w.lower() not in FILLER), key=len, reverse=True):
        if len(w) > 2:
            s = re.sub(rf"(?i)(?<![\w>]){re.escape(T.esc(w))}(?![\w<])", lambda m: f"<mark>{m.group(0)}</mark>", s)
    return s


def bubble(row: pd.Series, show_topics: bool = True) -> str:
    top = row["top_emotion"]
    color = EMO_COLOR.get(top, "#AFAFAF")
    face = EMO_ICON.get(top, "💬")
    chips = "".join(T.chip(EMO_NAME[e], EMO_COLOR[e]) for e in EMOTIONS if row[f"e_{e}"])
    if show_topics:
        chips += "".join(T.chip(aspect_name(a), INK_2, aspect_icon(a)) for a in row["aspects"] if a != "other")
    return (f'<div class="dl-bubble"><div class="face" style="background:{color}22">{face}</div>'
            f'<div class="msg">{highlight(row["clause"], row["triggers"])}<div style="margin-top:6px">{chips}</div></div></div>')


def bubbles(df: pd.DataFrame, show_topics: bool = True) -> None:
    if df.empty:
        T.note("No clear examples for this slice.")
        return
    T._md("".join(bubble(r, show_topics) for _, r in df.iterrows()))


def _nav(cfg: Cfg, ns: str, ds: D.Dataset) -> str:
    secs = [s for s in cfg.sections if s[0] != "reality" or ds.has_gold]     # hide views that have nothing to show
    ids = [s[0] for s in secs]
    names = dict(secs)
    pick = st.segmented_control("View", ids, default=ids[0], key=f"{ns}_nav", format_func=lambda i: names[i],
                                label_visibility="collapsed")
    return pick or ids[0]


def _seg_tone(k: dict) -> None:
    T.segmented([("Positive", k["pos"], BLUE), ("Neutral", k["neu"], NEUTRAL_GREY), ("Negative", k["neg"], ORANGE)])


# ------------------------------------------------------------------ entry
def render(ds: D.Dataset, cfg: Cfg, ns: str) -> None:
    section = _nav(cfg, ns, ds)
    fn = {"overview": overview, "topics": topics, "groups": groups, "compare": compare, "trends": trends,
          "inspector": inspector, "reality": reality}[section]
    fn(ds, cfg, ns)


# ------------------------------------------------------------------ overview
def overview(ds: D.Dataset, cfg: Cfg, ns: str) -> None:
    k = D.kpis(ds)
    if cfg.escalation:
        T.stats([
            ("💬", cfg.doc_many.title(), f"{k['n']:,}", f"{k['n_clauses']:,} sentences read", "#DDF4FF"),
            ("🚨", "Need a person", pct(k["esc"]), f"{int(k['esc'] * k['n']):,} flagged for escalation", "#FFE3E3"),
            ("⚡", "Average urgency", f"{k['urgency'] * 100:.0f}/100", "0 is calm, 100 is on fire", "#FFF1D6"),
            ("🙂", "Overall mood", mood_word(k["valence"]), f"mood score {k['valence'] * 100:+.0f}", "#E6F8D4"),
            ("🏢", cfg.group_many.title(), f"{k['groups']:,}", "in this file", "#F1E4FF"),
        ])
    else:
        gt = D.group_table(ds)
        n_real = int((gt["kind"] != "university_reviews").sum()) if len(gt) else 0
        T.stats([
            ("💬", cfg.doc_many.title(), f"{k['n']:,}", f"{k['n_clauses']:,} sentences read", "#DDF4FF"),
            ("😊", "Sound positive", pct(k["pos"]), "overall tone of the comment", "#E6F8D4"),
            ("🤔", "Mixed feelings", pct(k["mixed"]), "praise and complaint together", "#FFE3F1"),
            ("❓", "Confusion flags", f"{k['confusion']:,}", "“unclear”, “confusing”…", "#F1E4FF"),
            ("🎓", "Courses & teachers", f"{max(n_real, k['groups']):,}", "with comments", "#FFF1D6"),
        ])
    T.section("What stands out", "Plain-language findings, worked out from the data below.")
    items = I.support(ds) if cfg.escalation else I.feedback(ds, cfg.min_group)
    T.insights(items)

    a, b = st.columns(2, gap="large")
    with a:
        T.section(f"How do {cfg.doc_many} sound overall?", "Share by overall tone. Each message counts once.")
        _seg_tone(k)
        T.section("Emotions in sentences", "Share of sentences that carry each emotion (a sentence can carry more than one).")
        es = D.emotion_share(ds)
        order = sorted(EMOTIONS, key=lambda e: -es[e])
        C.hbar([f"{EMO_ICON[e]} {EMO_NAME[e]}" for e in order], [es[e] for e in order], [EMO_COLOR[e] for e in order],
               key=f"{ns}_emo", xmax=max(es.values()))
    with b:
        T.section("Moods spotted", f"Share of {cfg.doc_many} showing each mood. One message can show more than one.")
        ss = D.state_share(ds)
        order = sorted(cfg.states, key=lambda s: -ss[s])
        C.hbar([f"{STATE_ICON[s]} {STATE_NAME[s]}" for s in order], [ss[s] for s in order], [STATE_COLOR[s] for s in order],
               hover=[f"<b>{STATE_NAME[s]}</b><br>{pct(ss[s])} of {cfg.doc_many}<br>{STATE_HELP[s]}" for s in order],
               key=f"{ns}_state", xmax=max(ss[s] for s in cfg.states))
        if cfg.escalation:
            T.section("How urgent are they?", "Each bar counts conversations. Right of the dotted line gets flagged.")
            C.histogram(ds.docs["urgency"], key=f"{ns}_urg", color=ORANGE, xlabel="Urgency (0 calm → 1 urgent)",
                        marker_x=0.6, marker_text="flagged →")
        else:
            T.section("How happy are the comments?", "Each bar counts comments by mood score.")
            C.histogram(ds.docs["valence"], key=f"{ns}_val", color=BLUE, xlabel="Mood (−1 unhappy → +1 happy)",
                        rng=(-1, 1), marker_x=0, marker_text="neutral")
    lang = D.language_table(ds)
    if len(lang) > 1:
        non_en = 1 - float(lang.loc[lang["lang"] == "eng", "n"].sum()) / len(ds.docs)
        if non_en > 0.01:
            T.callout("tip", "🌍", f"{pct(non_en)} of {cfg.doc_many} don't look English. "
                      "They're read less reliably, so treat those results as a rough guide.")
    with st.expander("See the numbers behind these charts"):
        st.dataframe(pd.DataFrame({"Mood": [STATE_NAME[s] for s in cfg.states],
                                   f"Share of {cfg.doc_many}": [pct(ss[s], 1) for s in cfg.states]}), hide_index=True)
        st.dataframe(pd.DataFrame({"Emotion": [EMO_NAME[e] for e in EMOTIONS],
                                   "Share of sentences": [pct(es[e], 1) for e in EMOTIONS]}), hide_index=True)
        st.dataframe(lang.assign(language=lang["lang"].map(lambda x: LANG_NAME.get(x, x)))[["language", "n"]],
                     hide_index=True)


# ------------------------------------------------------------------ topics
def _topic_cards(at: pd.DataFrame) -> None:
    order = [a for a in at.index if a != "other"] + (["other"] if "other" in at.index else [])
    cards = ""
    for a in order:
        r = at.loc[a]
        cards += (f'<div class="dl-card dl-topic"><div class="t">{aspect_icon(a)} {T.esc(aspect_name(a))}</div>'
                  f'<div class="n">{int(r["n"]):,} sentences</div>'
                  f'{T.meter("Sounds unhappy", r["neg"], ORANGE, pct(r["neg"]))}'
                  f'{T.meter("Sounds happy", r["pos"], BLUE, pct(r["pos"]))}</div>')
    T._md(f'<div class="dl-topics">{cards}</div>')


def topics(ds: D.Dataset, cfg: Cfg, ns: str) -> None:
    at = D.aspect_table(ds)
    if at.empty:
        T.empty("🧩", "No topics found", "None of the sentences matched a topic.")
        return
    T.section("Topics people talk about", "“Unhappy” means worried, sad or angry. “Happy” means joyful.")
    _topic_cards(at)
    T.section("Which topic triggers which emotion?", "Darker means a bigger share of sentences about that topic carry the emotion.")
    pick = st.pills("Emotions to show", EMOTIONS, selection_mode="multi", default=EMOTIONS, key=f"{ns}_hm_emo",
                    format_func=lambda e: f"{EMO_ICON[e]} {EMO_NAME[e]}")
    pick = pick or EMOTIONS
    grid = at[pick].rename(columns=EMO_NAME)
    grid.index = [f"{aspect_icon(a)} {aspect_name(a)}" for a in grid.index]
    C.heatmap(grid, key=f"{ns}_hm")
    with st.expander("See the numbers"):
        st.dataframe(grid.map(lambda v: pct(v, 1)))

    T.section("Look closer at one topic")
    opts = [a for a in at.index if a != "other"] or list(at.index)
    sel = st.pills("Topic", opts, default=opts[0], key=f"{ns}_topic", format_func=lambda a: f"{aspect_icon(a)} {aspect_name(a)}")
    sel = sel or opts[0]
    left, right = st.columns([2, 3], gap="large")
    with left:
        r = at.loc[sel]
        order = sorted(EMOTIONS, key=lambda e: -r[e])
        C.hbar([f"{EMO_ICON[e]} {EMO_NAME[e]}" for e in order], [r[e] for e in order], [EMO_COLOR[e] for e in order],
               key=f"{ns}_tp_emo", xmax=1.0)
    with right:
        emo = st.selectbox("Show examples that sound like", ["Most intense"] + EMOTIONS, key=f"{ns}_tp_ex",
                           format_func=lambda e: e if e == "Most intense" else f"{EMO_ICON[e]} {EMO_NAME[e]}")
        ex = D.examples(ds, aspect=sel, emotion=None if emo == "Most intense" else emo, n=4)
        bubbles(ex, show_topics=False)
        T.note("Highlighted words are the ones that pushed the reading toward the emotion.")


# ------------------------------------------------------------------ groups
def _negative_examples(ds: D.Dataset, group: str, n: int = 3) -> pd.DataFrame:
    c = ds.clauses[(ds.clauses["group_id"] == group) & ds.clauses["neg"] & ds.clauses["clause"].str.len().between(25, 260)]
    return c.sort_values("score", ascending=False).drop_duplicates("clause").head(n)


def _positive_examples(ds: D.Dataset, group: str, n: int = 3) -> pd.DataFrame:
    return D.examples(ds, group=group, emotion="joy", n=n)


def report_card(ds: D.Dataset, cfg: Cfg, ns: str, gid: str, row: pd.Series) -> None:
    T._md(f'<div class="dl-card" style="display:flex;gap:14px;align-items:center;margin:6px 0 14px">'
          f'<div style="font-size:42px">{"🏢" if cfg.escalation else "🎓"}</div><div>'
          f'<div style="font-size:1.4rem;font-weight:900">{T.esc(pretty_group(gid))}</div>'
          f'<div style="color:{INK_2};font-weight:700">{int(row["n"]):,} {cfg.doc_many} · '
          f'{mood_word(row["valence"])} overall</div></div></div>')
    if cfg.escalation:
        T.stats([("🚨", "Need a person", pct(row["esc"]), "flagged for escalation", "#FFE3E3"),
                 ("⚡", "Average urgency", f"{row['urgency'] * 100:.0f}/100", "0 calm, 100 on fire", "#FFF1D6"),
                 ("🙂", "Sound positive", pct(row["pos"]), f"{pct(row['neg'])} sound negative", "#DDF4FF")])
    else:
        T.stats([("😊", "Sound positive", pct(row["pos"]), f"{pct(row['neg'])} sound negative", "#DDF4FF"),
                 ("🤔", "Mixed feelings", pct(row["mixed"]), "praise and complaint together", "#FFE3F1"),
                 ("🙂", "Mood score", f"{row['valence'] * 100:+.0f}", mood_word(row["valence"]), "#E6F8D4")])
    a, b = st.columns(2, gap="large")
    with a:
        T.section("Moods", f"Share of {cfg.doc_many} showing each mood.")
        order = sorted(cfg.states, key=lambda s: -row[s])
        C.hbar([f"{STATE_ICON[s]} {STATE_NAME[s]}" for s in order], [row[s] for s in order], [STATE_COLOR[s] for s in order],
               key=f"{ns}_rc_state_{gid}", xmax=max(0.3, max(row[s] for s in order)))
    with b:
        at = D.aspect_table(ds, gid)
        at = at[(at.index != "other") & (at["n"] >= 3)] if len(at) else at
        T.section("Strengths and weak spots", "Share sounding happy minus share sounding unhappy, per topic.")
        if len(at):
            at = at.sort_values("net", ascending=False)
            C.diverging([f"{aspect_icon(x)} {aspect_name(x)}" for x in at.index], at["net"].tolist(),
                        key=f"{ns}_rc_div_{gid}", pos_label="strength", neg_label="weak spot")
        else:
            T.note("Not enough topic mentions yet.")
    T.section("In their own words")
    x, y = st.columns(2, gap="large")
    with x:
        st.markdown("**😄 What goes well**")
        bubbles(_positive_examples(ds, gid), show_topics=False)
    with y:
        st.markdown("**😟 What goes wrong**")
        bubbles(_negative_examples(ds, gid), show_topics=False)


def groups(ds: D.Dataset, cfg: Cfg, ns: str) -> None:
    gt = D.group_table(ds)
    T.section(f"{cfg.group_many.title()} leaderboard", f"Click a bar, or pick from the list, to open its report card.")
    c1, c2, c3 = st.columns([2, 2, 3], gap="medium")
    smax = int(min(200, max(10, gt["n"].max()))) if len(gt) else 10
    with c1:
        min_n = st.slider(f"At least this many {cfg.doc_many}", 1, smax, min(cfg.min_group, smax), key=f"{ns}_min")
    kinds = sorted(gt["kind"].unique()) if len(gt) else []
    with c2:
        if not cfg.escalation and len(kinds) > 1:
            kind = st.selectbox("Show", ["All"] + kinds, key=f"{ns}_kind",
                                format_func=lambda k: {"All": "Everything", "course": "Courses", "teacher": "Teachers",
                                                       "university_reviews": "University"}.get(k, k))
        else:
            kind = "All"
    with c3:
        labels = [b[0] for b in cfg.boards]
        board = st.selectbox("Rank by", labels, key=f"{ns}_board")
    label, col, fmtk, color = next(b for b in cfg.boards if b[0] == board)
    view = gt[gt["n"] >= min_n]
    if kind != "All":
        view = view[view["kind"] == kind]
    if view.empty:
        T.empty("🔍", f"No {cfg.group_many} that big", "Lower the minimum number above.")
        return
    top = view.sort_values(col, ascending=False).head(12)
    names = [pretty_group(g) for g in top.index]
    ev = C.hbar(names, top[col].tolist(), color, key=f"{ns}_lb", selectable=True,
                text=[_fmt(fmtk, v) for v in top[col]], xmax=float(top[col].max()) or 1,
                hover=[f"<b>{pretty_group(g)}</b><br>{int(r.n):,} {cfg.doc_many}<br>Positive {pct(r.pos)} · Negative {pct(r.neg)}"
                       + (f"<br>Escalated {pct(r.esc)}" if cfg.escalation else "") for g, r in top.iterrows()])
    options = list(view.sort_values(col, ascending=False).index)
    skey = f"{ns}_grp"
    try:
        pts = ev.selection.points
        if pts:
            st.session_state[skey] = top.index[pts[0]["point_index"]]
    except Exception:
        pass
    if st.session_state.get(skey) not in options:
        st.session_state[skey] = options[0]
    gid = st.selectbox(f"Open a report card", options, key=skey, format_func=pretty_group)
    with st.expander("See the leaderboard as a table"):
        show = top.assign(name=names)[["name", "n", "pos", "neg", "valence"] + (["esc", "urgency"] if cfg.escalation else ["mixed"])]
        st.dataframe(show.rename(columns={"n": cfg.doc_many, "pos": "positive", "neg": "negative", "esc": "escalated"}),
                     hide_index=True)
    st.divider()
    report_card(ds, cfg, ns, gid, gt.loc[gid])


# ------------------------------------------------------------------ compare
def compare(ds: D.Dataset, cfg: Cfg, ns: str) -> None:
    gt = D.group_table(ds)
    gt = gt[gt["n"] >= max(3, min(cfg.min_group, 10))]
    if len(gt) < 2:
        T.empty("⚖️", f"Need at least two {cfg.group_many}", f"Your file has fewer than two {cfg.group_many} with enough {cfg.doc_many}.")
        return
    T.section(f"Compare two {cfg.group_many}", "Pick any two. Everything below updates.")
    opts = list(gt.sort_values("n", ascending=False).index)
    a, b = st.columns(2, gap="large")
    with a:
        ga = st.selectbox("Left", opts, index=0, key=f"{ns}_cmp_a", format_func=pretty_group)
    with b:
        gb = st.selectbox("Right", opts, index=1, key=f"{ns}_cmp_b", format_func=pretty_group)
    if ga == gb:
        T.callout("warn", "☝️", "Pick two different ones to compare.")
        return
    ra, rb = gt.loc[ga], gt.loc[gb]
    na, nb = pretty_group(ga), pretty_group(gb)
    cols = {na: BLUE, nb: ORANGE}
    st.write("")
    x, y = st.columns(2, gap="large")
    for col_, nm, r in ((x, na, ra), (y, nb, rb)):
        with col_:
            T._md(f'<div class="dl-card"><div style="font-weight:900;font-size:1.15rem;color:{cols[nm]}">● {T.esc(nm)}</div>'
                  f'<div style="color:{INK_2};font-weight:700">{int(r["n"]):,} {cfg.doc_many} · {mood_word(r["valence"])}</div>'
                  f'{T.meter("Sound positive", r["pos"], BLUE, pct(r["pos"]))}{T.meter("Sound negative", r["neg"], ORANGE, pct(r["neg"]))}'
                  + (T.meter("Need a person", r["esc"], "#E5484D", pct(r["esc"])) if cfg.escalation else
                     T.meter("Mixed feelings", r["mixed"], STATE_COLOR["mixed"], pct(r["mixed"])))
                  + "</div>")
    T.section("Moods side by side")
    C.grouped_hbar([f"{STATE_ICON[s]} {STATE_NAME[s]}" for s in cfg.states],
                   {na: [ra[s] for s in cfg.states], nb: [rb[s] for s in cfg.states]}, cols, key=f"{ns}_cmp_state")
    T.section("Where each sounds unhappy", "Share of sentences on each topic that sound worried, sad or angry.")
    ta, tb = D.aspect_table(ds, ga), D.aspect_table(ds, gb)
    asp = [a_ for a_ in (set(ta.index) & set(tb.index)) if a_ != "other"]
    asp = sorted(asp, key=lambda a_: -(ta.loc[a_, "n"] + tb.loc[a_, "n"]))
    if asp:
        C.grouped_hbar([f"{aspect_icon(a_)} {aspect_name(a_)}" for a_ in asp],
                       {na: [ta.loc[a_, "neg"] for a_ in asp], nb: [tb.loc[a_, "neg"] for a_ in asp]}, cols, key=f"{ns}_cmp_asp")
    else:
        T.note("The two have no topics in common yet.")


# ------------------------------------------------------------------ trends
def trends(ds: D.Dataset, cfg: Cfg, ns: str) -> None:
    periods = ds.periods
    if len(periods) < 2:
        T.empty("📈", "Trends need a time or batch column",
                "Add a column such as period, week or date when you upload, and pick it in the next step.")
        return
    T.section("Trends by topic", "Pick a mood. Each line shows how much of that topic's talk carries it, per period.")
    opts = ["any"] + list(cfg.states)
    st_pick = st.pills("Mood", opts, default="any" if not cfg.escalation else "stress", key=f"{ns}_trend_state",
                       format_func=lambda s: "All messages" if s == "any" else f"{STATE_ICON[s]} {STATE_NAME[s]}") or "any"
    tt = D.trend_table(ds, st_pick)
    big = tt.groupby("aspect", observed=True)["n"].sum().sort_values(ascending=False)
    keep = [a for a in big.index if a != "other"][:5]
    if not keep:
        T.empty("📈", "No topics to chart")
        return
    if st_pick == "any":
        T.callout("tip", "ℹ️", "“All messages” is always 100%. Pick a mood above to see change.")
    colors = dict(zip(keep, ["#2A78D6", "#EB6834", "#1BAF7A", "#E87BA4", "#6250D6"]))
    series, spikes = {}, {}
    for a in keep:
        g = tt[tt["aspect"] == a].set_index("period")["share"].reindex(periods)
        name = aspect_name(a)
        series[name] = g.fillna(0).tolist()
        sp = tt[(tt["aspect"] == a) & tt["spike"]]
        spikes[name] = [(str(r.period), r.share) for r in sp.itertuples()]
    C.lines(periods, series, {aspect_name(a): c for a, c in colors.items()}, key=f"{ns}_lines", spikes=spikes)
    sp = tt[tt["spike"] & tt["aspect"].isin(keep)].sort_values("share", ascending=False)
    if len(sp):
        T.callout("warn", "⚠️", f"{len(sp)} unusual jump{'s' if len(sp) != 1 else ''} spotted (red rings).")
        st.dataframe(sp.assign(topic=sp["aspect"].map(aspect_name), share=sp["share"].map(pct))[["period", "topic", "share", "n"]]
                     .rename(columns={"n": "messages"}), hide_index=True)
    else:
        T.callout("ok", "✅", "No unusual jumps. Everything moves within its normal range.")
    if not cfg.escalation:
        return
    if ds.key.startswith("test:"):
        T.note("In the test data the periods are equal-size batches, not calendar weeks, so this shows how the jump detector works rather than change over time.")


# ------------------------------------------------------------------ inspector
def inspector(ds: D.Dataset, cfg: Cfg, ns: str) -> None:
    T.section(f"Open a {cfg.doc_one}", "See how each sentence reads, which words drove it, and what to do next.")
    d = ds.docs
    c1, c2, c3 = st.columns([3, 3, 2], gap="medium")
    with c1:
        g = st.selectbox(cfg.group_one.title(), ["All"] + sorted(d["group_id"].unique(), key=pretty_group), key=f"{ns}_ins_g",
                         format_func=lambda x: f"All {cfg.group_many}" if x == "All" else pretty_group(x))
    with c2:
        mood = st.selectbox("Mood", ["Any"] + list(cfg.states), key=f"{ns}_ins_m",
                            format_func=lambda s: "Any mood" if s == "Any" else f"{STATE_ICON[s]} {STATE_NAME[s]}")
    with c3:
        only_esc = st.toggle("Only flagged", value=cfg.escalation, key=f"{ns}_ins_e") if cfg.escalation else False
    q = d
    if g != "All":
        q = q[q["group_id"] == g]
    if mood != "Any":
        q = q[q[f"st_{mood}"]]
    if only_esc:
        q = q[q["escalate"]]
    q = q.sort_values("urgency", ascending=False).head(300)
    if q.empty:
        T.empty("🔍", f"No {cfg.doc_many} match", "Loosen a filter above.")
        return

    def label(i: str) -> str:
        r = q.loc[i]
        mood_ = STATE_ICON.get(r["primary"], "😐")
        topics_ = ", ".join(aspect_name(a) for a in r["aspects"] if a != "other") or "no topic"
        return f"{mood_}  {pretty_group(r['group_id'])} · urgency {r['urgency'] * 100:.0f} · {topics_}"

    q = q.set_index("doc_id", drop=False)
    pick = st.selectbox(f"{cfg.doc_one.title()} (most urgent first)", list(q.index), key=f"{ns}_ins_doc", format_func=label)
    r = q.loc[pick]
    left, right = st.columns([3, 2], gap="large")
    with left:
        cl = D.clauses_of(ds, pick)
        shown = cl.head(40)
        bubbles(shown)
        if len(cl) > len(shown):
            T.note(f"Showing the first {len(shown)} of {len(cl)} sentences.")
    with right:
        badge = ('<span class="dl-chip" style="color:#E5484D;border-color:#E5484D55;background:#E5484D14">🚨 Escalate</span>'
                 if r["escalate"] else '<span class="dl-chip" style="color:#46A302;border-color:#46A30255;background:#46A30214">✅ Routine</span>')
        T._md(f'<div class="dl-card">{badge if cfg.escalation else ""}'
              + T.meter("Urgency", r["urgency"], "#E5484D" if r["urgency"] >= 0.6 else ORANGE, f"{r['urgency'] * 100:.0f}/100")
              + T.meter("Mood", (r["valence"] + 1) / 2, BLUE, f"{mood_word(r['valence'])}")
              + "<div style='margin-top:8px'>"
              + "".join(T.chip(STATE_NAME[s], STATE_COLOR[s], STATE_ICON[s]) for s in r["states"])
              + "".join(T.chip(aspect_name(a), INK_2, aspect_icon(a)) for a in r["aspects"] if a != "other")
              + "".join(T.chip("Confusion", PURPLE, "❓") for c in r["cues"] if c == "confusion")
              + "</div></div>")
        st.write("")
        tips = [f"<b>{STATE_NAME[s]}</b>: {TONE[s]}" for s in r["states"] if s in TONE]
        if cfg.escalation:
            T._md('<div class="dl-tip"><b>💡 Coach tip</b><br>' + ("<br>".join(tips) if tips else "No strong mood found, so a standard polite reply fits.") + "</div>")
        else:
            T._md('<div class="dl-tip"><b>💡 Reading</b><br>' + ("<br>".join(tips) if tips else "No strong mood found in this comment.")
                  + ("<br>It praises and complains at once." if r["mixed"] else "") + "</div>")


# ------------------------------------------------------------------ reality check
def reality(ds: D.Dataset, cfg: Cfg, ns: str) -> None:
    d = ds.docs
    if not ds.has_gold:
        T.empty("✅", "No ratings to check against", "This view appears when your file includes a gold_sentiment column (-1 to +1).")
        return
    g = d.dropna(subset=["gold_sentiment"]).copy()
    human = np.where(g["gold_sentiment"] > 0.15, "Positive", np.where(g["gold_sentiment"] < -0.15, "Negative", "Neutral"))
    T.section("Does the reading match what students rated?", "Each row is what students said. Each column is what the app read.")
    order = ["Negative", "Neutral", "Positive"]
    ct = pd.crosstab(pd.Series(human, name="human"), g["tone"].rename("read")).reindex(index=order, columns=order).fillna(0)
    norm = ct.div(ct.sum(1).replace(0, 1), axis=0)
    acc = float((human == g["tone"].values).mean())
    r = float(np.corrcoef(g["valence"], g["gold_sentiment"])[0, 1])
    a, b = st.columns([3, 2], gap="large")
    with a:
        grid = norm.copy()
        grid.index = [f"Students said {i.lower()}" for i in grid.index]
        grid.columns = [f"App read {c.lower()}" for c in grid.columns]
        C.confusion(grid, key=f"{ns}_conf")
    with b:
        T.stats([("🎯", "Same tone", pct(acc), f"of {len(g):,} rated comments", "#E6F8D4"),
                 ("📈", "Correlation", f"{r:.2f}", "1.00 would be a perfect match", "#DDF4FF")])
        T.callout("tip", "💡", "The app rarely calls a happy comment negative, or the reverse. Most misses land on “neutral”.")
    if "gold_aspects" in g:
        T.section("Does it find the topics students tagged?", "Of comments tagged with a topic by a person, the share where the app found it too.")
        rows = []
        for asp in sorted({x for l in g["gold_aspects"] for x in (l if l is not None else []) if x != "other"}):
            has = g["gold_aspects"].apply(lambda l, asp=asp: l is not None and asp in list(l))
            hit = g.loc[has, "aspects"].apply(lambda l, asp=asp: asp in l).mean()
            rows.append((asp, float(hit), int(has.sum())))
        rows.sort(key=lambda t: -t[1])
        if rows:
            C.hbar([f"{aspect_icon(a)} {aspect_name(a)}" for a, _, _ in rows], [h for _, h, _ in rows], BLUE,
                   key=f"{ns}_asp_hit", xmax=1.0,
                   hover=[f"<b>{aspect_name(a)}</b><br>{pct(h)} found in {n:,} tagged comments" for a, h, n in rows])
