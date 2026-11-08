"""Plotly charts in one visual language: thin rounded bars, recessive grid, direct labels, hover tooltips.

Rules applied (dataviz skill): one axis per chart, categorical colours fixed per entity, a single hue for
magnitude, two hues around a neutral for polarity, labels printed on marks so colour is never the only cue.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.dashboards.labels import BLUE, BLUE_D, INK, INK_2, LINE, ORANGE

FONT = "Nunito, Segoe UI Rounded, Arial Rounded MT Bold, system-ui, sans-serif"
CONFIG = {"displayModeBar": False, "responsive": True}
SEQ = [[0.0, "#EAF7FE"], [0.35, "#8ED8FB"], [0.7, "#1CB0F6"], [1.0, "#0B6A99"]]       # one hue, light to dark
POS, NEG = BLUE, ORANGE                                                              # diverging poles


def _base(fig: go.Figure, h: int, margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        height=h, margin=margin or dict(l=0, r=16, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK), hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor=LINE),
        bargap=0.38, showlegend=False,
    )
    fig.update_xaxes(gridcolor="#EFEFEF", zeroline=False, linecolor=LINE, tickfont=dict(color=INK_2))
    fig.update_yaxes(gridcolor="#EFEFEF", zeroline=False, linecolor=LINE, tickfont=dict(color=INK))
    return fig


def show(fig: go.Figure, key: str, **kw):
    return st.plotly_chart(fig, config=CONFIG, width="stretch", key=key, **kw)


def hbar(labels: Sequence[str], values: Sequence[float], colors: str | Sequence[str], *, fmt: str = "{:.0%}",
         hover: Sequence[str] | None = None, xmax: float | None = None, key: str, selectable: bool = False,
         height: int | None = None, text: Sequence[str] | None = None):
    """Horizontal bars, biggest on top, value printed at the end of each bar."""
    labels, values = list(labels), list(values)
    n = len(labels)
    colors = [colors] * n if isinstance(colors, str) else list(colors)
    txt = list(text) if text is not None else [fmt.format(v) for v in values]
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h", marker=dict(color=colors, cornerradius=8), text=txt, textposition="outside",
        cliponaxis=False, textfont=dict(family=FONT, size=13, color=INK), customdata=hover,
        hovertemplate="%{customdata}<extra></extra>" if hover is not None else "%{y}: %{text}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, ticksuffix="  ")
    fig.update_xaxes(visible=False, range=[0, (xmax if xmax is not None else (max(values) if values else 1)) * 1.22 or 1])
    _base(fig, height or (34 * n + 24), dict(l=0, r=40, t=4, b=4))
    if selectable:
        fig.update_traces(selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=0.45)))
        return show(fig, key, on_select="rerun", selection_mode="points")
    return show(fig, key)


def heatmap(grid: pd.DataFrame, *, key: str, fmt: str = "{:.0%}", zmax: float | None = None, colorbar_title: str = ""):
    """Rows x columns grid, one hue light to dark, value printed in each cell."""
    z = grid.values.astype(float)
    top = zmax if zmax is not None else float(np.nanmax(z)) or 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=list(grid.columns), y=list(grid.index), zmin=0, zmax=top, colorscale=SEQ, xgap=4, ygap=4, showscale=False,
        customdata=[[fmt.format(v) for v in row] for row in z], hovertemplate="<b>%{y}</b> · %{x}<br>%{customdata}<extra></extra>",
    ))
    for i, row in enumerate(z):                      # value printed in every cell; white on dark cells, ink on light ones
        for j, v in enumerate(row):
            fig.add_annotation(x=grid.columns[j], y=grid.index[i], text=fmt.format(v), showarrow=False,
                               font=dict(family=FONT, size=14, color="#FFFFFF" if v / top > 0.62 else INK))
    fig.update_yaxes(autorange="reversed", showgrid=False, ticksuffix="  ")
    fig.update_xaxes(side="top", showgrid=False, tickfont=dict(size=14, color=INK))
    _base(fig, 62 * len(grid) + 60, dict(l=0, r=0, t=40, b=0))
    show(fig, key)


def diverging(labels: Sequence[str], values: Sequence[float], *, key: str, pos_label: str, neg_label: str,
              height: int | None = None):
    """Bars either side of zero: blue to the right (good), orange to the left (bad), signed numbers printed."""
    labels, values = list(labels), list(values)
    colors = [POS if v >= 0 else NEG for v in values]
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h", marker=dict(color=colors, cornerradius=8),
        text=[f"{v * 100:+.0f}" for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(family=FONT, size=13, color=INK),
        hovertemplate="%{y}<br>%{text} points<extra></extra>",
    ))
    lim = max(0.05, max(abs(v) for v in values)) * 1.35 if values else 1
    fig.update_xaxes(range=[-lim, lim], zeroline=True, zerolinecolor=INK_2, zerolinewidth=2, showgrid=False, showticklabels=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, ticksuffix="  ")
    fig.add_annotation(x=-lim, y=1.0, yref="paper", yanchor="bottom", text=f"← {neg_label}", showarrow=False,
                       font=dict(color="#B35F00", size=12, family=FONT), xanchor="left")
    fig.add_annotation(x=lim, y=1.0, yref="paper", yanchor="bottom", text=f"{pos_label} →", showarrow=False,
                       font=dict(color=BLUE_D, size=12, family=FONT), xanchor="right")
    _base(fig, height or (38 * len(labels) + 50), dict(l=0, r=10, t=26, b=4))
    show(fig, key)


def grouped_hbar(cats: Sequence[str], series: dict[str, Sequence[float]], colors: dict[str, str], *, key: str,
                 fmt: str = "{:.0%}"):
    """Two (or three) entities side by side per category; legend always shown for >1 series."""
    fig = go.Figure()
    for name, vals in series.items():
        fig.add_trace(go.Bar(name=name, y=list(cats), x=list(vals), orientation="h", marker=dict(color=colors[name], cornerradius=6),
                             text=[fmt.format(v) for v in vals], textposition="outside", cliponaxis=False,
                             textfont=dict(family=FONT, size=12, color=INK),
                             hovertemplate=f"<b>{name}</b><br>%{{y}}: %{{text}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.12)
    fig.update_yaxes(autorange="reversed", showgrid=False, ticksuffix="  ")
    mx = max((max(v) for v in series.values() if len(v)), default=1) or 1
    fig.update_xaxes(visible=False, range=[0, mx * 1.25])
    _base(fig, 54 * len(cats) + 40, dict(l=0, r=40, t=30, b=4))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0, font=dict(size=13)))
    show(fig, key)


def histogram(values: Sequence[float], *, key: str, color: str = BLUE, bins: int = 20, xlabel: str,
              marker_x: float | None = None, marker_text: str = "", rng: tuple[float, float] = (0, 1)):
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins, range=rng)
    mids = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=mids, y=counts, marker=dict(color=color, cornerradius=4), width=(edges[1] - edges[0]) * 0.86,
                           customdata=np.c_[edges[:-1], edges[1:]],
                           hovertemplate="%{customdata[0]:.2f} to %{customdata[1]:.2f}<br>%{y:,} messages<extra></extra>"))
    if marker_x is not None:
        fig.add_vline(x=marker_x, line=dict(color=INK, width=2, dash="dot"))
        fig.add_annotation(x=marker_x, y=1, yref="paper", text=marker_text, showarrow=False, xanchor="left", yanchor="top",
                           font=dict(size=12, color=INK, family=FONT), xshift=6)
    fig.update_xaxes(title=dict(text=xlabel, font=dict(color=INK_2, size=12)), showgrid=False)
    fig.update_yaxes(title=None, gridcolor="#EFEFEF")
    _base(fig, 230, dict(l=0, r=8, t=8, b=0))
    show(fig, key)


def lines(x: Sequence[str], series: dict[str, Sequence[float]], colors: dict[str, str], *, key: str, yfmt: str = ".0%",
          spikes: dict[str, list[tuple[str, float]]] | None = None, height: int = 330):
    """One line per topic, direct label at the last point, red ring on unusual jumps."""
    fig = go.Figure()
    for name, ys in series.items():
        fig.add_trace(go.Scatter(x=list(x), y=list(ys), name=name, mode="lines+markers", line=dict(color=colors[name], width=3),
                                 marker=dict(size=8, line=dict(color="white", width=2)),
                                 hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:{yfmt}}}<extra></extra>"))
        ys = list(ys)
        if len(ys):
            fig.add_annotation(x=list(x)[-1], y=ys[-1], text=f" {name}", showarrow=False, xanchor="left",
                               font=dict(color=INK, size=12, family=FONT))
    for name, pts in (spikes or {}).items():
        if pts:
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", name=f"{name}: jump",
                                     marker=dict(size=17, color="rgba(0,0,0,0)", line=dict(color="#E5484D", width=3)),
                                     hovertemplate=f"<b>Unusual jump</b> · {name}<br>%{{x}}: %{{y:{yfmt}}}<extra></extra>"))
    fig.update_yaxes(tickformat=yfmt, rangemode="tozero")
    fig.update_xaxes(showgrid=False)
    _base(fig, height, dict(l=0, r=110, t=10, b=0))
    fig.update_layout(showlegend=len(series) > 1, legend=dict(orientation="h", y=1.12, x=0, font=dict(size=12)))
    show(fig, key)


def confusion(grid: pd.DataFrame, *, key: str):
    """Row-normalised table of 'what people said' vs 'what the app read'."""
    heatmap(grid, key=key, fmt="{:.0%}", zmax=1.0)
