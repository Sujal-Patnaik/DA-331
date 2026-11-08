"""The "Upload & analyze" tab. Drop a CSV, check the columns, press Analyze, see the same views as the first tab.

Scoring itself is not done here: ``backend.analyze`` is called and its tables are drawn.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from apps.dashboards import backend, data as D, theme as T, views as V
from apps.dashboards.labels import BLUE, GREEN, ORANGE, STATE_NAME, pct, pretty_group

MAX_ROWS = 100_000
GUESS = {
    "text": ["text_raw", "text", "review", "comment", "comments", "message", "tweet", "feedback", "content", "body", "response"],
    "id": ["doc_id", "id", "review_id", "ticket_id", "row_id"],
    "group": ["group_id", "company", "brand", "handle", "course", "course_id", "teacher", "instructor", "professor", "department"],
    "period": ["period", "date", "week", "month", "batch", "semester", "term", "year", "created_at", "timestamp"],
}


@dataclass(frozen=True)
class UploadCopy:
    noun: str            # "customer messages"
    text_hint: str
    group_name: str      # "Company"
    group_hint: str
    period_hint: str
    example: str


def read_any(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty.")
    raise ValueError("Couldn't read that file as text.")


def guess(df: pd.DataFrame, kind: str) -> str | None:
    low = {c.lower().strip(): c for c in df.columns}
    for g in GUESS[kind]:
        if g in low:
            return low[g]
    if kind == "text":
        obj = [(df[c].astype(str).str.len().mean(), c) for c in df.columns]
        return max(obj)[1] if obj else None
    return None


def _export(ds: D.Dataset) -> bytes:
    d = ds.docs
    out = pd.DataFrame({
        "id": d["doc_id"], "group": d["group_id"], "period": d["period"], "language": d["lang"],
        "text": d["text"] if "text" in d else "",
        "overall_tone": d["tone"], "headline_mood": d["primary"].map(lambda s: STATE_NAME.get(s, "No clear mood")),
        "moods": d["states"].apply(lambda l: "; ".join(STATE_NAME.get(s, s) for s in l)),
        "topics": d["aspects"].apply(lambda l: "; ".join(l)),
        "urgency_0_100": (d["urgency"] * 100).round(0).astype(int), "mood_score": (d["valence"] * 100).round(0).astype(int),
        "flagged_for_escalation": d["escalate"],
    })
    return out.to_csv(index=False).encode("utf-8-sig")


def render(cfg: V.Cfg, copy: UploadCopy, ns: str = "up") -> None:
    connected = backend.is_connected(cfg.profile)
    left, right = st.columns([5, 4], gap="large")
    with left:
        T.section("1 · Add your file", f"A CSV of {copy.noun}. The columns below work best; only the text is required.")
        up = st.file_uploader("Drop a CSV here", type=["csv", "tsv", "txt"], key=f"{ns}_file", label_visibility="collapsed")
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇ Download sample file", D.sample_csv(cfg.profile), f"{cfg.profile}_sample.csv", "text/csv",
                               key=f"{ns}_dl_sample", width="stretch")
        with c2:
            if st.button("▶ Try the sample", key=f"{ns}_use_sample", width="stretch"):
                st.session_state[f"{ns}_sample_on"] = True
                st.session_state.pop(f"{ns}_result", None)
    with right:
        T.section("What your file should look like")
        T._md(f'<div class="dl-card">{T.cols_html(["text"], ["id", copy.group_name.lower(), "period"])}'
              f'<div class="dl-note" style="margin-top:6px">Green is required. Names are matched automatically, or you can choose them next.</div>'
              f'<div style="margin-top:12px;font-weight:600">'
              f'<div><b>text</b> · {T.esc(copy.text_hint)}</div>'
              f'<div><b>{T.esc(copy.group_name.lower())}</b> · {T.esc(copy.group_hint)}</div>'
              f'<div><b>period</b> · {T.esc(copy.period_hint)}</div></div>'
              f'<div class="dl-note">Example: <i>{T.esc(copy.example)}</i></div></div>')
        st.write("")
        if connected:
            T.callout("ok", "🟢", "EmoLDGN+ model connected. Each sentence is read by the trained model; large files take a few minutes.")
        else:
            T.callout("warn", "🟠", "The trained model is not available on this machine (<code>artifacts/best.pt</code> is missing). "
                      "You can still set up your file.")

    raw_df = None
    if up is not None:
        st.session_state.pop(f"{ns}_sample_on", None)
        try:
            raw_df = read_any(up.getvalue())
            fname = up.name
        except Exception as e:
            T.callout("warn", "😕", f"Couldn't read that file: {T.esc(e)}")
            return
    elif st.session_state.get(f"{ns}_sample_on"):
        raw_df, fname = read_any(D.sample_csv(cfg.profile)), f"{cfg.profile}_sample.csv"
    if raw_df is None:
        st.write("")
        T.empty("📂", "Waiting for a file", "Drop a CSV above, or press “Try the sample” to see how it works.")
        _show_result(cfg, ns)
        return
    if raw_df.empty or len(raw_df.columns) == 0:
        T.callout("warn", "😕", "That file has no rows.")
        return

    st.divider()
    T.section("2 · Check the columns", f"{fname} · {len(raw_df):,} rows · {len(raw_df.columns)} columns")
    st.dataframe(raw_df.head(5), hide_index=True, width="stretch")
    cols = list(raw_df.columns)
    none = "(none)"
    a, b, c, d = st.columns(4, gap="medium")
    tg = guess(raw_df, "text")
    with a:
        text_col = st.selectbox("Message text *", cols, index=cols.index(tg) if tg in cols else 0, key=f"{ns}_c_text")
    def opt(label: str, kind: str, col):
        g = guess(raw_df, kind)
        choices = [none] + [x for x in cols if x != text_col]
        with col:
            v = st.selectbox(label, choices, index=choices.index(g) if g in choices else 0, key=f"{ns}_c_{kind}")
        return None if v == none else v
    id_col = opt("ID (optional)", "id", b)
    group_col = opt(f"{copy.group_name} (optional)", "group", c)
    period_col = opt("Period (optional)", "period", d)

    texts = raw_df[text_col].astype(str).str.strip()
    blank = int((texts == "").sum())
    n_ok = len(raw_df) - blank
    if blank:
        T.callout("tip", "ℹ️", f"{blank:,} rows have no text and will be skipped.")
    if n_ok > MAX_ROWS:
        T.callout("warn", "⚠️", f"Only the first {MAX_ROWS:,} rows will be analyzed.")
    if period_col is None:
        T.note("No period column: the Trends view will be empty. Add one to see change over time.")

    st.write("")
    go = st.button(f"✨ Analyze {min(n_ok, MAX_ROWS):,} rows", type="primary", key=f"{ns}_go", disabled=n_ok == 0)
    if go:
        bar = st.progress(0.0, text="Starting…")
        t0 = time.time()
        try:
            frame = raw_df.head(MAX_ROWS)
            docs, cl = backend.analyze(frame, cfg.profile, text_col, id_col, group_col, period_col,
                                       progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
            ds = D.prepare(docs, cl, key=f"upload:{time.time_ns()}", name=fname)
            st.session_state[f"{ns}_result"] = (ds, fname, time.time() - t0)
            bar.empty()
        except backend.BackendNotConnected:
            bar.empty()
            st.session_state.pop(f"{ns}_result", None)
            T.callout("warn", "🔌", "The trained model isn't available, so there's nothing to show. "
                      "Your file and column choices look fine. Export a model to <code>artifacts/</code> first.")
        except Exception as e:                              # shown to the user instead of a stack trace
            bar.empty()
            T.callout("warn", "😕", f"Something went wrong while analyzing: {T.esc(e)}")
    _show_result(cfg, ns)


def _show_result(cfg: V.Cfg, ns: str) -> None:
    res = st.session_state.get(f"{ns}_result")
    if not res:
        return
    ds, fname, secs = res
    st.divider()
    T.section("3 · Your results", f"{fname}")
    k = D.kpis(ds)
    top = st.columns([4, 2, 2])
    with top[0]:
        T.callout("ok", "🎉", f"Read <b>{k['n']:,}</b> {cfg.doc_many} ({k['n_clauses']:,} sentences) in {secs:.1f}s.")
    with top[1]:
        st.download_button("⬇ Download results", _export(ds), f"{cfg.profile}_results.csv", "text/csv", key=f"{ns}_dl_res",
                           width="stretch")
    with top[2]:
        if st.button("✖ Clear results", key=f"{ns}_clear", width="stretch"):
            st.session_state.pop(f"{ns}_result", None)
            st.rerun()
    base = D.kpis(D.test_results(cfg.profile))
    if cfg.escalation:
        T.note(f"For reference: in the test data {pct(base['esc'])} of conversations were flagged; in your file {pct(k['esc'])}.")
    else:
        T.note(f"For reference: in the test data {pct(base['pos'])} of comments read positive; in your file {pct(k['pos'])}.")
    st.write("")
    V.render(ds, cfg, ns)
