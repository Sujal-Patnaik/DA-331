"""Duolingo-flavoured look for Streamlit: chunky rounded cards, green/blue buttons, pill tabs, soft shadows.

Everything is plain CSS + small HTML helpers. No logic lives here.
"""
from __future__ import annotations

import html

import streamlit as st

from apps.dashboards.labels import (BLUE, BLUE_D, GREEN, GREEN_D, INK, INK_2, INK_3, LINE, ORANGE, RED, SOFT,
                                    YELLOW)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@500;600;700;800;900&display=swap');
:root {{
  --green:{GREEN}; --green-d:{GREEN_D}; --blue:{BLUE}; --blue-d:{BLUE_D}; --ink:{INK}; --ink-2:{INK_2};
  --ink-3:{INK_3}; --line:{LINE}; --soft:{SOFT}; --red:{RED}; --orange:{ORANGE}; --yellow:{YELLOW};
}}
html, body, [class*="css"], .stApp, button, input, textarea, select {{
  font-family: 'Nunito','Segoe UI Rounded','Arial Rounded MT Bold',system-ui,-apple-system,sans-serif !important;
}}
.stApp {{ background:#fff; color:var(--ink); }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ display:none !important; }}
.block-container {{ max-width:1180px; padding-top:1.2rem; padding-bottom:4rem; }}
h1,h2,h3,h4 {{ font-weight:900 !important; color:var(--ink); letter-spacing:-0.01em; }}
p, li, label, .stMarkdown {{ font-weight:600; }}

/* ---------- hero */
.dl-hero {{ position:relative; overflow:hidden; border-radius:24px; padding:28px 32px; color:#fff;
  background:linear-gradient(135deg,var(--green) 0%,#7ADB1F 100%); border-bottom:6px solid var(--green-d);
  margin-bottom:18px; }}
.dl-hero.blue {{ background:linear-gradient(135deg,var(--blue) 0%,#49C6FF 100%); border-bottom-color:var(--blue-d); }}
.dl-hero h1 {{ color:#fff; font-size:2.1rem; margin:0 0 6px 0; font-weight:900 !important; }}
.dl-hero p {{ margin:0; font-size:1.05rem; font-weight:700; opacity:.95; max-width:640px; }}
.dl-hero .big {{ position:absolute; right:26px; top:50%; transform:translateY(-50%); font-size:84px; line-height:1;
  filter:drop-shadow(0 6px 0 rgba(0,0,0,.12)); }}
.dl-hero .pills {{ margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }}
.dl-hero .pill {{ background:rgba(255,255,255,.22); border-radius:999px; padding:5px 12px; font-weight:800; font-size:.85rem; }}

/* ---------- cards */
.dl-card {{ background:#fff; border:2px solid var(--line); border-bottom-width:5px; border-radius:20px; padding:16px 18px; }}
.dl-stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(172px,1fr)); gap:14px; margin:6px 0 18px; }}
.dl-stat {{ display:flex; gap:12px; align-items:center; }}
.dl-stat .ico {{ width:52px; height:52px; border-radius:16px; display:grid; place-items:center; font-size:26px; flex:none; }}
.dl-stat .lab {{ font-size:.68rem; line-height:1.15; letter-spacing:.05em; text-transform:uppercase; color:var(--ink-2); font-weight:800; }}
.dl-stat .val {{ font-size:1.7rem; font-weight:900; color:var(--ink); line-height:1.1; }}
.dl-stat .hint {{ font-size:.78rem; color:var(--ink-2); font-weight:700; }}

.dl-insights {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; margin-bottom:10px; }}
.dl-insight {{ display:flex; gap:12px; align-items:flex-start; }}
.dl-insight .ico {{ font-size:26px; line-height:1; margin-top:2px; }}
.dl-insight .ttl {{ display:block; font-size:1rem; font-weight:900; margin-bottom:2px; color:var(--ink); }}
.dl-insight span b {{ color:var(--ink); font-weight:900; }}
.dl-insight span {{ font-size:.92rem; color:#5a5a5a; font-weight:600; line-height:1.35; display:block; }}

.dl-section {{ margin:26px 0 6px; }}
.dl-section h3 {{ margin:0; font-size:1.35rem; }}
.dl-section p {{ margin:2px 0 0; color:var(--ink-2); font-weight:700; font-size:.95rem; }}

/* ---------- chips, meters */
.dl-chip {{ display:inline-flex; align-items:center; gap:5px; border-radius:999px; padding:3px 11px; font-size:.8rem;
  font-weight:800; border:2px solid; margin:2px 4px 2px 0; white-space:nowrap; }}
.dl-meter {{ margin:8px 0; }}
.dl-meter .top {{ display:flex; justify-content:space-between; font-weight:800; font-size:.9rem; margin-bottom:4px; }}
.dl-meter .track {{ background:var(--line); height:16px; border-radius:999px; overflow:hidden; }}
.dl-meter .fill {{ height:100%; border-radius:999px; position:relative; }}
.dl-meter .fill::after {{ content:""; position:absolute; left:8px; right:8px; top:3px; height:4px; border-radius:99px;
  background:rgba(255,255,255,.35); }}
.dl-seg {{ display:flex; height:40px; border-radius:14px; overflow:hidden; gap:3px; margin:8px 0 4px; }}
.dl-seg div {{ display:flex; align-items:center; justify-content:center; color:#fff; font-weight:900; font-size:.9rem; min-width:34px; }}
.dl-legend {{ display:flex; gap:16px; flex-wrap:wrap; font-weight:800; font-size:.85rem; color:var(--ink-2); }}

/* ---------- topic cards */
.dl-topics {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:14px; margin:8px 0 14px; }}
.dl-topic .t {{ font-weight:900; font-size:1.02rem; display:flex; gap:8px; align-items:center; }}
.dl-topic .n {{ color:var(--ink-2); font-weight:700; font-size:.82rem; margin-bottom:6px; }}

/* ---------- conversation bubbles */
.dl-bubble {{ display:flex; gap:10px; align-items:flex-start; margin:10px 0; }}
.dl-bubble .face {{ width:42px; height:42px; border-radius:14px; display:grid; place-items:center; font-size:22px; flex:none; }}
.dl-bubble .msg {{ background:#fff; border:2px solid var(--line); border-radius:6px 18px 18px 18px; padding:10px 14px;
  font-weight:600; line-height:1.45; flex:1; }}
.dl-bubble mark {{ background:#FFF1B8; color:inherit; border-radius:6px; padding:0 3px; font-weight:800; }}
.dl-tip {{ background:#DDF4FF; border:2px solid #84D8FF; border-radius:18px; padding:14px 16px; color:#1B6C95; font-weight:700; }}
.dl-tip b {{ color:#145A7D; }}
.dl-warn {{ background:#FFF4E0; border:2px solid #FFD08A; border-radius:18px; padding:14px 16px; color:#8A5200; font-weight:700; }}
.dl-ok {{ background:#E6F8D4; border:2px solid #B7E88A; border-radius:18px; padding:14px 16px; color:#3F7A00; font-weight:700; }}
.dl-empty {{ text-align:center; padding:34px 16px; color:var(--ink-2); font-weight:700; }}
.dl-empty .e {{ font-size:54px; }}
.dl-note {{ color:var(--ink-2); font-size:.82rem; font-weight:700; margin-top:10px; }}
.dl-step {{ display:flex; gap:12px; align-items:flex-start; margin:8px 0; }}
.dl-step .num {{ width:34px; height:34px; border-radius:50%; background:var(--blue); color:#fff; font-weight:900; display:grid;
  place-items:center; flex:none; box-shadow:0 3px 0 var(--blue-d); }}
.dl-step b {{ font-weight:900; }}
.dl-col {{ display:inline-block; background:var(--soft); border:2px solid var(--line); border-radius:10px; padding:1px 8px;
  font-family:ui-monospace,Consolas,monospace; font-size:.82rem; font-weight:700; margin:0 3px 3px 0; }}
.dl-col.req {{ background:#E6F8D4; border-color:#B7E88A; color:#3F7A00; }}

/* ---------- streamlit widgets */
div[data-testid="stTabs"] > div:first-child {{ border-bottom:none; }}
.stTabs [role="tablist"] {{ gap:10px; border-bottom:2px solid var(--line); padding-bottom:0; }}
.stTabs [data-testid="stTab"] {{ height:52px; padding:0 24px !important; margin-right:6px; border-radius:16px 16px 0 0; font-weight:900; font-size:1.02rem;
  color:var(--ink-2); background:transparent; }}
.stTabs [data-testid="stTab"] p {{ font-weight:900; font-size:1.02rem; }}
.stTabs [aria-selected="true"] {{ color:var(--blue-d) !important; background:#DDF4FF; border:2px solid #84D8FF; border-bottom:none; }}
.stTabs [aria-selected="true"] p {{ color:var(--blue-d) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:var(--blue); height:4px; border-radius:4px; }}
.stTabs [data-baseweb="tab-border"] {{ display:none; }}

.stButton > button, .stDownloadButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {{
  border-radius:16px; font-weight:900; letter-spacing:.04em; text-transform:uppercase; padding:.7rem 1.4rem;
  border:2px solid var(--line); border-bottom-width:5px; background:#fff; color:var(--blue-d); transition:transform .05s; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background:var(--soft); border-color:var(--line); color:var(--blue-d); }}
.stButton > button:active, .stDownloadButton > button:active {{ transform:translateY(3px); border-bottom-width:2px; }}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background:var(--green); color:#fff; border-color:var(--green-d); }}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{ background:#61D800; color:#fff; border-color:var(--green-d); }}
.stButton > button[kind="primary"] p, [data-testid="stBaseButton-primary"] p {{ color:#fff; }}

[data-testid="stFileUploader"] section {{ border:3px dashed #84D8FF; background:#F3FBFF; border-radius:22px; padding:26px; }}
[data-testid="stFileUploader"] section:hover {{ background:#E7F7FF; }}
[data-testid="stFileUploader"] button {{ border-radius:14px; font-weight:900; }}

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{ border-radius:14px; border:2px solid var(--line); min-height:46px; }}
div[data-baseweb="select"] > div:focus-within {{ border-color:var(--blue); }}
[data-testid="stSlider"] [role="slider"] {{ background:var(--green); box-shadow:0 0 0 4px #E6F8D4; }}
.stProgress > div > div > div > div {{ background:var(--green); border-radius:999px; }}
.stProgress > div > div > div {{ height:16px; border-radius:999px; background:var(--line); }}
[data-testid="stDataFrame"] {{ border:2px solid var(--line); border-radius:16px; overflow:hidden; }}
[data-testid="stExpander"] {{ border:2px solid var(--line); border-radius:16px; }}
[data-testid="stExpander"] summary {{ font-weight:800; }}
[data-testid="stAlert"] {{ border-radius:16px; font-weight:700; }}

/* pills and segmented controls */
[data-testid="stPills"] button, [data-testid="stSegmentedControl"] button {{
  border-radius:999px !important; font-weight:800; border:2px solid var(--line) !important; background:#fff; padding:6px 16px; }}
[data-testid="stPills"] button[aria-checked="true"], [data-testid="stPills"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stSegmentedControl"] button[aria-pressed="true"] {{
  background:#DDF4FF !important; border-color:#84D8FF !important; color:var(--blue-d) !important; }}
[data-testid="stSegmentedControl"] > div {{ gap:8px; background:transparent; }}

@media (max-width: 700px) {{
  .dl-hero .big {{ display:none; }} .dl-hero h1 {{ font-size:1.6rem; }}
  .block-container {{ padding-left:14px; padding-right:14px; }}
}}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _md(s: str) -> None:
    st.markdown(s.replace("\n", ""), unsafe_allow_html=True)


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def hero(title: str, subtitle: str, emoji: str, tone: str = "green", pills: list[str] | None = None) -> None:
    ps = "".join(f'<span class="pill">{esc(p)}</span>' for p in (pills or []))
    _md(f'<div class="dl-hero {tone}"><div class="big">{emoji}</div><h1>{esc(title)}</h1><p>{esc(subtitle)}</p>'
        f'{"<div class=pills>" + ps + "</div>" if ps else ""}</div>')


def section(title: str, sub: str = "") -> None:
    _md(f'<div class="dl-section"><h3>{esc(title)}</h3>{"<p>" + esc(sub) + "</p>" if sub else ""}</div>')


def stats(items: list[tuple[str, str, str, str, str]]) -> None:
    """(icon, label, value, hint, tint) tiles."""
    cells = "".join(
        f'<div class="dl-card dl-stat"><div class="ico" style="background:{tint}">{ico}</div><div>'
        f'<div class="lab">{esc(lab)}</div><div class="val">{esc(val)}</div><div class="hint">{esc(hint)}</div></div></div>'
        for ico, lab, val, hint, tint in items)
    _md(f'<div class="dl-stats">{cells}</div>')


def insights(items: list[dict]) -> None:
    cells = "".join(
        f'<div class="dl-card dl-insight" style="border-color:{i.get("color", LINE)}33"><div class="ico">{i["icon"]}</div>'
        f'<div><div class="ttl">{esc(i["title"])}</div><span>{i["text"]}</span></div></div>' for i in items)
    _md(f'<div class="dl-insights">{cells}</div>')


def chip(text: str, color: str, icon: str = "") -> str:
    return (f'<span class="dl-chip" style="color:{color};border-color:{color}55;background:{color}14">'
            f'{icon + " " if icon else ""}{esc(text)}</span>')


def meter(label: str, frac: float, color: str, right: str = "") -> str:
    w = max(0.0, min(1.0, frac)) * 100
    return (f'<div class="dl-meter"><div class="top"><span>{label}</span><span>{esc(right)}</span></div>'
            f'<div class="track"><div class="fill" style="width:{w:.1f}%;background:{color}"></div></div></div>')


def meters(rows: list[tuple[str, float, str, str]]) -> None:
    _md("".join(meter(*r) for r in rows))


def segmented(parts: list[tuple[str, float, str]]) -> None:
    """(label, fraction, colour): one stacked bar with the numbers printed inside."""
    tot = sum(p[1] for p in parts) or 1
    bar = "".join(f'<div style="flex:{max(p[1] / tot, 0.001):.4f};background:{p[2]};color:{"#3C3C3C" if p[2] == "#C9CED6" else "#fff"}">{p[1] / tot * 100:.0f}%</div>'
                  for p in parts)
    leg = "".join(f'<span><span style="color:{p[2]}">●</span> {esc(p[0])}</span>' for p in parts)
    _md(f'<div class="dl-seg">{bar}</div><div class="dl-legend">{leg}</div>')


def callout(kind: str, icon: str, text: str) -> None:
    cls = {"tip": "dl-tip", "warn": "dl-warn", "ok": "dl-ok"}[kind]
    _md(f'<div class="{cls}">{icon}&nbsp; {text}</div>')


def empty(icon: str, title: str, text: str = "") -> None:
    _md(f'<div class="dl-card dl-empty"><div class="e">{icon}</div><div style="font-weight:900;font-size:1.15rem;color:{INK}">'
        f'{esc(title)}</div><div>{esc(text)}</div></div>')


def note(text: str) -> None:
    _md(f'<div class="dl-note">{text}</div>')


def steps(items: list[str]) -> None:
    _md("".join(f'<div class="dl-step"><div class="num">{i}</div><div>{t}</div></div>' for i, t in enumerate(items, 1)))


def cols_html(required: list[str], optional: list[str]) -> str:
    r = "".join(f'<span class="dl-col req">{esc(c)}</span>' for c in required)
    o = "".join(f'<span class="dl-col">{esc(c)}</span>' for c in optional)
    return r + o
