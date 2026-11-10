"""How it works tab — architecture + headline results. (Diagram expanded in Phase 6.)"""
from __future__ import annotations

import streamlit as st

from frontend.theme import COLORS, chip


def render_how_it_works() -> None:
    c = COLORS
    st.markdown('<div style="font-size:1.6rem;font-weight:800;">How FinSight works</div>'
                f'<div style="color:{c["muted"]};margin-bottom:14px;">A router picks the path, '
                'Python and live tools produce every number, and Mistral only writes the words.</div>',
                unsafe_allow_html=True)

    flow = " &nbsp;→&nbsp; ".join([
        chip("Laya routes", c["laya"]),
        chip("only that route's agents", c["text_dim"]),
        chip("Python / Finnhub / SEC → facts", c["python"]),
        chip("Mistral writes", c["mistral"]),
        chip("validator checks every number", c["python"]),
    ])
    st.markdown(f'<div class="fs-card">{flow}</div>', unsafe_allow_html=True)

    st.markdown('<div class="fs-label" style="margin-top:18px;">Headline results (from our experiments)</div>',
                unsafe_allow_html=True)
    rows = [
        ("Routing", "Laya 0.85 vs Mistral 0.52 (McNemar p=0.0003); hybrid 0.88 overall", "EXPERIMENT1"),
        ("Math: Python vs LLM", "Python 100% vs Mistral 29% on 100 calcs; gpt-5 98%", "EXPERIMENT2/4"),
        ("Live data vs model memory", "Price error 210% without live data → 0.00% with Finnhub", "EXPERIMENT2"),
        ("Laya as router", "100% correct routing (price→real_time, calc→numbers)", "EXPERIMENT3"),
    ]
    for title, detail, src in rows:
        st.markdown(
            f'<div class="fs-card" style="margin:8px 0;">'
            f'<div style="font-weight:700;">{title}</div>'
            f'<div style="color:{c["text_dim"]};font-size:0.9rem;" class="fs-mono">{detail}</div>'
            f'<div style="color:{c["muted"]};font-size:0.72rem;margin-top:4px;">source: {src}_FINDINGS.md</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.caption("Research and educational use only · not financial advice.")
