"""Shared Altair chart theme + builders (Terminal design system).

Altair chosen over Plotly: already a dependency, declarative/themeable in one place,
good tabular-figure support, no extra heavy JS bundle. Every chart here reads numbers
that Python/tools produced — charts never compute anything.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

from frontend.theme import COLORS, FONT_MONO, FONT_SANS

_AXIS = dict(labelColor=COLORS["text_dim"], titleColor=COLORS["muted"],
             gridColor=COLORS["border"], domainColor=COLORS["border"], tickColor=COLORS["border"])


def _base(df, h=300):
    return alt.Chart(df).properties(height=h).configure_view(
        strokeWidth=0, fill=COLORS["surface"]
    ).configure(background=COLORS["surface"], font=FONT_SANS)


def revenue_chart(points: list[dict], note: str = ""):
    """Continuous revenue over quarters: emerald area + line, value labels, latest dot,
    $ axis from 0, horizontal period labels. `points` = [{period, revenue_b}]."""
    df = pd.DataFrame(points)
    df["idx"] = range(len(df))
    df["label"] = df["revenue_b"].map(lambda v: f"${v:.2f}B")
    x = alt.X("period:N", sort=None, title=None,
              axis=alt.Axis(labelAngle=0, **_AXIS))
    y = alt.Y("revenue_b:Q", title="Revenue ($B)",
              scale=alt.Scale(domainMin=0, nice=True), axis=alt.Axis(**_AXIS))
    area = alt.Chart(df).mark_area(
        line={"color": COLORS["python"], "strokeWidth": 2.5},
        color=alt.Gradient(gradient="linear",
                           stops=[alt.GradientStop(color=COLORS["python"], offset=0),
                                  alt.GradientStop(color="#0E7C6600", offset=1)],
                           x1=1, x2=1, y1=1, y2=0),
        opacity=0.9,
    ).encode(x=x, y=y, tooltip=[alt.Tooltip("period:N", title="Period"),
                                alt.Tooltip("revenue_b:Q", title="Revenue ($B)", format=".2f")])
    dots = alt.Chart(df).mark_point(filled=True, size=55, color=COLORS["python"]).encode(x=x, y=y)
    last = df.iloc[[-1]]
    hi = alt.Chart(last).mark_point(filled=True, size=130, color=COLORS["python"]).encode(x=x, y=y)
    labels = alt.Chart(df).mark_text(dy=-12, color=COLORS["text"], font=FONT_MONO, fontSize=11).encode(
        x=x, y=y, text="label:N")
    chart = (area + dots + hi + labels).properties(height=300).configure_view(
        strokeWidth=0, fill=COLORS["surface"]).configure(background=COLORS["surface"], font=FONT_SANS)
    return chart


def sentiment_chart(points: list[dict]):
    """Horizontal stacked sentiment bar: positive/neutral/negative (+uncertain gray)."""
    order = ["positive", "neutral", "negative", "uncertain"]
    cmap = {"positive": COLORS["up"], "neutral": COLORS["muted"],
            "negative": COLORS["down"], "uncertain": "#5A6B7A"}
    rows = [{"label": p["label"], "count": p["count"]} for p in points]
    df = pd.DataFrame(rows)
    df["row"] = "sentiment"
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("count:Q", stack="normalize", title=None, axis=alt.Axis(format="%", **_AXIS)),
        y=alt.Y("row:N", title=None, axis=None),
        color=alt.Color("label:N", scale=alt.Scale(domain=order, range=[cmap[o] for o in order]),
                        legend=alt.Legend(title=None, orient="bottom", labelColor=COLORS["text_dim"])),
        order=alt.Order("label:N"),
        tooltip=["label:N", "count:Q"],
    ).properties(height=90).configure_view(strokeWidth=0, fill=COLORS["surface"]).configure(
        background=COLORS["surface"], font=FONT_SANS)
    return chart
