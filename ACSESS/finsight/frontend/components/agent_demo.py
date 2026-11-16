"""Chat tab — a simple landing (question first), then the cited answer; routing, the
agent pipeline and shared memory sit in a collapsible "How FinSight answered this".

Follows the header's company. The real_time route works for any ticker (Finnhub);
numbers have bundled AMD sample data only, so for another company we show a clear
notice and a "Show AMD sample instead" button — never a silent switch.
"""
from __future__ import annotations

import html
import re

import streamlit as st

from agent_core.llm import FakeLLM, MistralLLM
from agent_core.memory import Conversation, SharedMemory
from agent_core.orchestrator import Orchestrator
from agent_core.orchestrator import resolve_ticker
from agent_core.orchestrator import route as keyword_route
from frontend.charts import revenue_chart, sentiment_chart
from frontend.companies import SAMPLE_TICKER, short_name
from frontend.components.answer import _inline, render_answer
from frontend.components.pipeline import render_pipeline, render_route_card
from frontend.components.quote_card import render_quote_card
from frontend.theme import COLORS

def _fresh(use_fake: bool, use_laya: bool) -> Orchestrator:
    llm = FakeLLM() if use_fake else MistralLLM()
    router = None
    if use_laya:
        from agent_core.laya_router import LayaRouter
        router = LayaRouter()
    return Orchestrator(llm, memory=SharedMemory(), conversation=Conversation(), router=router)


def _run(orch: Orchestrator, question: str, ticker: str) -> None:
    # Memory items aren't keyed by company, so a new company gets a fresh memory;
    # otherwise its answer reuses the previous company's data (the GOOGL → AMD bug).
    if st.session_state.get("agent_ticker") not in (None, ticker):
        orch = st.session_state.agent_orch = _fresh(st.session_state.agent_fake, st.session_state.agent_laya)
    st.session_state.agent_ticker = ticker
    orch.conversation.current_ticker = ticker
    with st.spinner("Agents working…"):
        tr = orch.ask(question)
    _archive()
    mem = orch.memory
    st.session_state.agent_result = {
        "tr": tr, "ticker": ticker,
        "quote": mem.get("amd_quote"), "news": mem.get("amd_news"),
        "chart": mem.get("amd_chart"), "schart": mem.get("amd_sentiment_chart"),
        "notes": mem.get("amd_notes"), "sentiment": mem.get("amd_sentiment"),
        "memtext": mem.describe() or "(empty)",
    }
    st.session_state.pending_mismatch = None


def _context_chips(orch: Orchestrator) -> None:
    c = COLORS
    conv = orch.conversation
    bits = []
    if conv.current_ticker:
        bits.append(("company", conv.current_ticker))
    if getattr(conv, "current_period", None):
        bits.append(("period", conv.current_period))
    if not bits:
        return
    chips = "".join(
        f'<span class="fs-chip" style="background:{c["surface2"]};color:{c["text_dim"]};'
        f'border-color:{c["border"]};">{k}: <b class="fs-mono">&nbsp;{v}</b></span>' for k, v in bits
    )
    st.markdown(f'<div style="margin:2px 0 10px;">{chips}</div>', unsafe_allow_html=True)


# Landing chips (Anna's mockup): short label -> the question it puts in the box.
CHIPS = {
    "Filing highlights": "Tell me about {co}'s business and products",
    "Revenue trend": "How did {co}'s revenue change year over year?",
    "Stock price": "What is {co} trading at right now?",
    "Recent news": "What's the latest news on {co} today?",
    "Should I buy?": "Should I buy {co} stock?",  # shows the no-advice guardrail
}


def render_agent_demo(company: str) -> None:
    _, mid, _ = st.columns([1, 6, 1])  # one calm centered column, like the mockup
    with mid:
        _chat(company)


def _archive() -> None:
    """Move the current answer into the on-screen thread (display only; the backend keeps
    its own last-3 Q&A context for follow-ups)."""
    prev = st.session_state.get("agent_result")
    if prev:
        st.session_state.setdefault("agent_history", []).append(
            {"q": prev["tr"].question, "answer": prev["tr"].answer})
    st.session_state.agent_result = None


def _submit() -> None:
    """Ask button / Enter: take the question and clear the box, like a chat app."""
    # typing then clicking Ask fires the box's on_change AND the button's on_click;
    # the second call sees the box already cleared, so it must not overwrite the question
    q = st.session_state.agent_q.strip()
    if q:
        st.session_state.to_ask = q
    st.session_state.agent_q = ""


def _chat(company: str) -> None:
    c = COLORS
    short = short_name(company)
    in_chat = bool(st.session_state.get("agent_result"))
    if not in_chat:
        st.markdown('<div class="fs-hello" style="text-align:center;">Hey there 👋</div>',
                    unsafe_allow_html=True)
    st.markdown(f'<div class="fs-hero">What would you like to know about <span>{short}</span>?</div>'
                f'<div class="fs-sub">Explore company filings, financial trends, and recent news. '
                'Every number is computed in Python and every claim is cited.</div>',
                unsafe_allow_html=True)

    # engine switches are for developers, so they live in the sidebar
    with st.sidebar.expander("Developer settings"):
        use_fake = st.toggle("Offline (FakeLLM)", value=False, key="use_fake",
                             help="Off = real Mistral on the PSC GPU (needs the :8000 tunnel).")
        use_laya = st.toggle("Route with Laya", value=False, key="use_laya",
                             help="Use Laya to route (needs the :8001 tunnel); else keyword rules.")

    # rebuild the orchestrator when the engine toggles change (preserves memory otherwise)
    if ("agent_orch" not in st.session_state
            or st.session_state.get("agent_fake") != use_fake
            or st.session_state.get("agent_laya") != use_laya):
        st.session_state.agent_orch = _fresh(use_fake, use_laya)
        st.session_state.agent_fake = use_fake
        st.session_state.agent_laya = use_laya
    orch = st.session_state.agent_orch

    cols = st.columns([6, 1], vertical_alignment="center")
    # on_change fires on Enter, so Enter asks just like the button
    cols[0].text_input("Ask a question", key="agent_q", label_visibility="collapsed",
                       placeholder=f"Ask a follow-up about {short}…" if in_chat else f"Ask about {short}…",
                       on_change=_submit)
    cols[1].button("Ask", type="primary", use_container_width=True, key="ask_btn", on_click=_submit)

    # chips fill the box (as in the mockup); the callback clears the pill so it can be reused
    def _fill():
        st.session_state.agent_q = CHIPS[st.session_state.agent_chips].replace("{co}", short)
        st.session_state.agent_chips = None
    st.markdown('<div class="fs-sub" style="margin:8px 0 0;">Try a question</div>', unsafe_allow_html=True)
    with st.container(horizontal=True, horizontal_alignment="center"):
        st.pills("Try a question", list(CHIPS), key="agent_chips", on_change=_fill,
                 label_visibility="collapsed")

    to_run = st.session_state.pop("to_ask", "")
    if to_run:
        intended = keyword_route(to_run)
        # Use the ticker named in the question if any (so "AMD revenue" runs even when the
        # sidebar shows another company); else fall back to the sidebar company.
        eff = resolve_ticker(to_run, Conversation()) or company
        # Only the numbers route needs bundled data (AMD-only, no XBRL yet). Sentiment/text
        # now fetch the selected company's real SEC filing, so they run for any company.
        if intended == "numbers" and eff != SAMPLE_TICKER:
            st.session_state.pending_mismatch = {"q": to_run, "intended": intended, "company": eff}
            _archive()
        else:
            _run(orch, to_run, ticker=eff)

    pm = st.session_state.get("pending_mismatch")
    if pm:
        st.info(
            f"Revenue figures for {short_name(pm['company'])} aren't connected yet. Right now only "
            f"**{SAMPLE_TICKER} sample data** is available for revenue questions."
        )
        if st.button(f"Show {SAMPLE_TICKER} sample instead", type="primary"):
            # the question names the other ticker, which would override the AMD choice
            q = re.sub(rf"\b{re.escape(pm['company'])}\b", SAMPLE_TICKER, pm["q"], flags=re.I)
            _run(orch, q, ticker=SAMPLE_TICKER)
            st.rerun()  # redraw without the warning, which was drawn before the click

    res = st.session_state.get("agent_result")
    if not res:
        return
    tr = res["tr"]
    orch = st.session_state.agent_orch  # _run may have swapped in a fresh one
    src = "sample data (AMD)" if res["ticker"] == SAMPLE_TICKER else f"from {short_name(res['ticker'])}'s latest 10-K"

    _question_bubble(tr.question)
    render_answer(tr, orch.memory)

    if tr.route == "real_time":
        q = res["quote"]
        if isinstance(q, dict) and "price" in q:
            render_quote_card(q, res["ticker"])
        if res["news"]:
            _render_news(res["news"])

    chart = res["chart"]
    if tr.route == "numbers" and chart and chart.get("points"):
        st.markdown('<div class="fs-label" style="margin-top:14px;">Revenue trend '
                    f'<span style="color:{c["muted"]};">· {src}</span></div>', unsafe_allow_html=True)
        _delta_chips(res.get("notes"))
        st.altair_chart(revenue_chart(chart["points"], chart.get("note", "")), use_container_width=True)
        st.caption("Changes computed in Python from the quarterly figures.")

    sent = res.get("sentiment")
    schart = res["schart"]
    if tr.route == "sentiment" and schart and schart.get("points"):
        st.markdown('<div class="fs-label" style="margin-top:14px;">Filing tone '
                    f'<span style="color:{c["muted"]};">· {src}</span></div>', unsafe_allow_html=True)
        st.altair_chart(sentiment_chart(schart["points"]), use_container_width=True)
        _sentiment_detail(sent)

    # explainability: how the answer was made, one click away instead of in the way
    with st.expander("🔍 How FinSight answered this: route, agents and memory"):
        _context_chips(orch)
        render_route_card(tr, use_laya=st.session_state.get("use_laya", False))
        render_pipeline(tr)
        if tr.reused:
            st.caption(f"♻ Reused from memory (skipped re-fetch): {', '.join(tr.reused)}")
        st.markdown('<div class="fs-label" style="margin-top:10px;">Shared memory after this question</div>',
                    unsafe_allow_html=True)
        st.code(res["memtext"])

    # below the answer, where people look for it; a click handler may clear the box above
    st.button("↻ New chat", key="new_chat", on_click=_new_chat)

    # earlier turns, newest first, under the latest answer (so the new one is never off-screen)
    history = st.session_state.get("agent_history") or []
    if history:
        st.markdown('<div class="fs-label" style="margin-top:22px;">Earlier in this chat</div>',
                    unsafe_allow_html=True)
        for turn in reversed(history):
            _question_bubble(turn["q"])
            st.markdown(f'<div class="fs-card" style="padding:12px 16px;color:{c["text_dim"]};">'
                        f'{_inline(turn["answer"])}</div>', unsafe_allow_html=True)


def _question_bubble(q: str) -> None:
    st.markdown(f'<div class="fs-q"><span>{html.escape(q)}</span></div>', unsafe_allow_html=True)


def _new_chat() -> None:
    st.session_state.agent_orch = _fresh(st.session_state.agent_fake, st.session_state.agent_laya)
    st.session_state.agent_result = None
    st.session_state.agent_history = []
    st.session_state.pending_mismatch = None
    st.session_state.agent_q = ""


def _delta_chips(notes: list | None) -> None:
    """QoQ / YoY deltas from analysis_agent's notes (Python-computed), as arrow chips."""
    if not notes:
        return
    c = COLORS
    chips = []
    for n in notes:
        v = n.get("values", {}) if isinstance(n, dict) else {}
        for key, label in (("qoq_pct", "QoQ"), ("yoy_pct", "YoY")):
            if key in v and isinstance(v[key], (int, float)):
                pct = v[key]
                up = pct >= 0
                col = c["up"] if up else c["down"]
                arrow = "▲" if up else "▼"
                chips.append(
                    f'<span class="fs-chip" style="background:{col}1a;color:{col};border-color:{col}55;">'
                    f'{label} {arrow} <b class="fs-mono">&nbsp;{pct:+.1f}%</b></span>')
    if chips:
        st.markdown(f'<div style="margin:4px 0 8px;">{"".join(dict.fromkeys(chips))}</div>',
                    unsafe_allow_html=True)


def _sentiment_detail(sent: dict | None) -> None:
    """Counts + % line and a click-to-open list of the labeled passages."""
    if not isinstance(sent, dict):
        return
    c = COLORS
    counts = sent.get("counts", {})
    n = sent.get("n") or sum(counts.values()) or 1
    order = [("positive", c["up"]), ("neutral", c["muted"]),
             ("negative", c["down"]), ("uncertain", "#5A6B7A")]
    line = "  ".join(
        f'<span style="color:{col};font-weight:700;">{counts.get(k, 0)} {k}</span>'
        f'<span style="color:{c["muted"]};">({counts.get(k, 0) / n * 100:.0f}%)</span>'
        for k, col in order if counts.get(k))
    st.markdown(f'<div class="fs-mono" style="font-size:0.9rem;margin:4px 0;">{line}</div>',
                unsafe_allow_html=True)
    labels = sent.get("labels") or []
    if labels:
        with st.expander(f"Show the {len(labels)} labeled passages"):
            cmap = dict(order)
            for lb in labels:
                col = cmap.get(lb.get("label"), c["muted"])
                flag = " ⚠ needs review" if lb.get("needs_review") else ""
                st.markdown(
                    f'<div style="padding:6px 0;border-bottom:1px solid {c["border"]};">'
                    f'<span class="fs-chip" style="background:{col}1a;color:{col};border-color:{col}55;">'
                    f'{lb.get("label", "")}{flag}</span> '
                    f'<span style="color:{c["text_dim"]};font-size:0.86rem;">{lb.get("text", "")[:200]}</span></div>',
                    unsafe_allow_html=True)


def _render_news(news: list) -> None:
    c = COLORS
    st.markdown('<div class="fs-label" style="margin-top:12px;">Recent news · third-party</div>',
                unsafe_allow_html=True)
    for a in news:
        summ = (a.get("summary") or "")
        summ = (summ[:170] + "…") if len(summ) > 170 else summ
        st.markdown(
            f'<div class="fs-card" style="margin:6px 0;padding:12px 14px;">'
            f'<div style="display:flex;justify-content:space-between;">'
            f'<span class="fs-chip" style="background:{c["finnhub"]}1a;color:{c["finnhub"]};'
            f'border-color:{c["finnhub"]}44;">{(a.get("category") or "news").upper()}</span>'
            f'<span style="color:{c["muted"]};font-size:0.76rem;">{a.get("source","")} · {a.get("date","")}</span></div>'
            f'<a href="{a.get("url","#")}" target="_blank" style="display:block;margin-top:6px;'
            f'font-weight:700;color:{c["text"]};text-decoration:none;">{a.get("headline","")}</a>'
            + (f'<div style="color:{c["text_dim"]};font-size:0.86rem;margin-top:4px;">{summ}</div>' if summ else "")
            + '</div>',
            unsafe_allow_html=True,
        )
