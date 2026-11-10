"""Live quote card (Terminal theme): price + change + day range + continuous history.

The Finnhub free quote has no price history, so the big chart uses Yahoo (yfinance,
free, no key) for daily closes; the day-range bar uses the live Finnhub quote fields.
Everything degrades cleanly offline (sample quote → no chart, clear label).
"""
from __future__ import annotations

import streamlit as st

from frontend.companies import company_name, logo_url
from frontend.theme import COLORS, FONT_MONO


@st.cache_data(ttl=1800, show_spinner=False)
def _history(ticker: str, period: str):
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period=period, interval="1d")
        if h is None or h.empty:
            return None
        return [(d.strftime("%Y-%m-%d"), float(c)) for d, c in zip(h.index, h["Close"])]
    except Exception:  # noqa: BLE001 — history is optional enrichment
        return None


_PERIODS = {"1M": "1mo", "6M": "6mo", "YTD": "ytd", "1Y": "1y", "5Y": "5y", "MAX": "max"}


def render_quote_card(q: dict, ticker: str | None) -> None:
    c = COLORS
    tk = (ticker or "").upper() or "—"
    live = bool(q.get("live"))
    price = q.get("price", 0.0)
    chg, chgp = q.get("change"), q.get("change_pct")
    up = isinstance(chgp, (int, float)) and chgp >= 0
    col = c["up"] if up else c["down"]
    arrow = "▲" if up else "▼"
    badge = ""
    if isinstance(chgp, (int, float)):
        ab = f"{chg:+,.2f} " if isinstance(chg, (int, float)) else ""
        badge = (f'<span class="fs-mono" style="color:{col};font-size:1.05rem;font-weight:700;'
                 f'margin-left:12px;">{arrow} {ab}({chgp:+.2f}%)</span>')
    tag = (f'<span class="fs-chip" style="background:{c["finnhub"]}1f;color:{c["finnhub"]};'
           f'border-color:{c["finnhub"]}66;">● LIVE · Finnhub</span>' if live else
           f'<span class="fs-chip" style="background:{c["laya"]}1f;color:{c["laya"]};'
           f'border-color:{c["laya"]}66;">SAMPLE DATA</span>')
    meta = f'as of {q.get("as_of", "")[:16].replace("T", " ")} UTC'
    if q.get("market_open_likely") is False:
        meta += " · market likely closed"
    logo = (f'<img src="{logo_url(tk)}" style="width:40px;height:40px;border-radius:9px;background:#fff;'
            f'padding:4px;object-fit:contain;" onerror="this.style.display=\'none\'">') if tk != "—" else ""

    st.markdown(
        f'<div class="fs-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<div style="display:flex;gap:12px;align-items:center;">{logo}'
        f'<div><div style="font-size:1.15rem;font-weight:800;">{company_name(tk)}</div>'
        f'<div class="fs-mono" style="color:{c["muted"]};font-size:0.82rem;">{tk}</div></div></div>{tag}</div>'
        f'<div style="margin-top:12px;"><span class="fs-mono" style="font-size:2.4rem;font-weight:800;">'
        f'${price:,.2f}</span>{badge}</div>'
        f'<div style="color:{c["muted"]};font-size:0.8rem;margin-top:4px;">{meta}'
        + (f' · <a href="{q.get("source_url", "#")}" target="_blank">source</a>' if live else '')
        + '</div>'
        + _day_range_html(q) +
        '</div>',
        unsafe_allow_html=True,
    )
    if live:
        _history_chart(tk)


def _day_range_html(q: dict) -> str:
    """Low——•——High bar with open & previous-close markers (Finnhub c/h/l/o/pc)."""
    c = COLORS
    lo, hi = q.get("low"), q.get("high")
    price, op, pc = q.get("price"), q.get("open"), q.get("prev_close")
    if not all(isinstance(v, (int, float)) and v for v in (lo, hi)) or hi <= lo:
        return ""
    def pos(v):
        return max(0, min(100, (v - lo) / (hi - lo) * 100)) if isinstance(v, (int, float)) and v else None
    p, o = pos(price), pos(op)
    marks = ""
    if p is not None:
        marks += (f'<div style="position:absolute;left:{p}%;top:-3px;width:10px;height:10px;'
                  f'border-radius:50%;background:{c["text"]};transform:translateX(-50%);"></div>')
    if o is not None:
        marks += (f'<div style="position:absolute;left:{o}%;top:0;width:2px;height:4px;'
                  f'background:{c["muted"]};transform:translateX(-50%);"></div>')
    return (
        f'<div style="margin-top:14px;"><div class="fs-label">Day range</div>'
        f'<div style="position:relative;height:4px;background:linear-gradient(90deg,{c["down"]},{c["muted"]},{c["up"]});'
        f'border-radius:2px;margin:8px 0 4px;">{marks}</div>'
        f'<div class="fs-mono" style="display:flex;justify-content:space-between;color:{c["muted"]};font-size:0.74rem;">'
        f'<span>${lo:,.2f}</span><span>${hi:,.2f}</span></div></div>'
    )


def _history_chart(tk: str) -> None:
    c = COLORS
    sel = st.radio("Range", list(_PERIODS), index=3, horizontal=True,
                   key="pr_range", label_visibility="collapsed")
    data = _history(tk, _PERIODS[sel])
    if not data:
        st.caption("Price history unavailable for this ticker.")
        return
    import altair as alt
    import pandas as pd
    df = pd.DataFrame(data, columns=["date", "price"])
    df["date"] = pd.to_datetime(df["date"])
    first, last = df["price"].iloc[0], df["price"].iloc[-1]
    chg = last - first
    up = chg >= 0
    line = c["up"] if up else c["down"]
    st.markdown(f'<div class="fs-mono" style="font-size:1.1rem;font-weight:700;margin-top:6px;">'
                f'${last:,.2f} <span style="color:{line};font-size:0.9rem;">'
                f'{"+" if up else ""}{chg:,.2f} ({chg / first * 100:+.1f}%) · {sel}</span></div>',
                unsafe_allow_html=True)
    area = alt.Chart(df).mark_area(
        line={"color": line, "strokeWidth": 2}, opacity=0.25,
        color=alt.Gradient(gradient="linear",
                           stops=[alt.GradientStop(color=line, offset=0),
                                  alt.GradientStop(color="#ffffff00", offset=1)],
                           x1=1, x2=1, y1=1, y2=0),
    ).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(labelColor=c["text_dim"], gridColor=c["border"])),
        y=alt.Y("price:Q", title=None, scale=alt.Scale(zero=False),
                axis=alt.Axis(labelColor=c["text_dim"], gridColor=c["border"])),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("price:Q", format="$,.2f")],
    ).properties(height=260).configure_view(strokeWidth=0, fill=COLORS["surface"]).configure(
        background=COLORS["surface"])
    st.altair_chart(area, use_container_width=True)
    st.caption(f"Source: Yahoo Finance · {tk} · {len(df)} trading days")
