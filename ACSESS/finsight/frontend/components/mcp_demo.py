"""MCP Demo tab: FinSight's tools used from an MCP client (recorded in Cursor, from
Anna's mockup). Static on purpose: the MCP server runs inside the MCP app, not in this web app.
"""
from __future__ import annotations

import streamlit as st

from frontend.theme import COLORS

# Recorded in Cursor on Oct 30, 2026 (Anna's mockup); shown as-is, not re-run.
EXAMPLES = [
    {
        "title": "Stock quote",
        "prompt": "What is NVDA trading at?",
        "result": "$204.64, down 4.22% from the previous close of $213.66. [Q1]",
        "note": "[Q1] Finnhub via FinSight · Oct 30, 2026, 4:00 p.m. ET, market closed.",
        "call": 'get_quote({"ticker": "NVDA"})\n\n{\n  "ticker": "NVDA",\n  "price": 204.64,\n'
                '  "change": -9.02,\n  "change_pct": -4.2217,\n  "prev_close": 213.66\n}',
    },
    {
        "title": "Investment advice request",
        "prompt": "Should I buy AMD?",
        "result": "FinSight declined to give a buy, sell or hold recommendation and offered to share "
                  "company facts instead.",
        "note": "The router declined it before any research agent or language model ran.",
        "call": 'ask_finsight({"question": "Should I buy AMD?"})',
    },
]

TOOLS = [
    ("get_quote", "Live price and daily change (Finnhub)"),
    ("get_company_news", "Recent news naming the company"),
    ("list_filings", "The company's SEC filings, with links"),
    ("get_filing_section", "One section of the latest 10-K / 10-Q, e.g. Risk Factors"),
    ("ask_finsight", "The full agent pipeline: routed, computed in Python, cited"),
]


def render_mcp_demo() -> None:
    c = COLORS
    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        st.markdown('<div class="fs-hero">FinSight in your <span>AI assistant</span></div>'
                    '<div class="fs-sub">FinSight\'s tools speak the Model Context Protocol (MCP), so any MCP '
                    'app can use them: Cursor, Claude Desktop, VS Code and others. These examples were '
                    'recorded in Cursor and show the request, the tool output and the result.</div>',
                    unsafe_allow_html=True)
        for ex in EXAMPLES:
            st.markdown(
                f'<div class="fs-card" style="margin-top:12px;">'
                f'<div style="font-weight:800;font-size:1.05rem;margin-bottom:6px;">{ex["title"]}</div>'
                f'<div style="color:{c["text_dim"]};"><b>Cursor prompt:</b> {ex["prompt"]}</div>'
                f'<div style="margin-top:4px;"><b>Result:</b> {ex["result"]}</div>'
                f'<div style="color:{c["muted"]};font-size:0.82rem;margin-top:6px;">{ex["note"]}</div></div>',
                unsafe_allow_html=True)
            with st.expander("View tool call and output"):
                st.code(ex["call"], language=None)

        st.markdown('<div class="fs-label" style="margin-top:22px;">The five FinSight tools (all read-only)</div>',
                    unsafe_allow_html=True)
        rows = "".join(
            f'<div style="display:flex;gap:12px;padding:7px 0;border-bottom:1px solid {c["border"]};">'
            f'<span class="fs-mono" style="color:{c["laya"]};min-width:170px;">{name}</span>'
            f'<span style="color:{c["text_dim"]};">{desc}</span></div>' for name, desc in TOOLS)
        st.markdown(rows, unsafe_allow_html=True)
        st.caption("Setup: see mcp_server/README.md. Facts only, no investment advice.")
