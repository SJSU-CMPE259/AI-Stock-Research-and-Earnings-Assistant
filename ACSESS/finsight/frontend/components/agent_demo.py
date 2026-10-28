"""Agent Demo tab — routing → pipeline → cited answer, over one shared memory.

Follows the sidebar company. The real_time route works for any ticker (Finnhub);
numbers/sentiment have bundled AMD sample data only, so for another company we show
a clear notice and a "Use AMD sample" button — never a silent switch.
"""
from __future__ import annotations

import streamlit as st

from agent_core.llm import FakeLLM, MistralLLM
from agent_core.memory import Conversation, SharedMemory
from agent_core.orchestrator import Orchestrator
from agent_core.orchestrator import route as keyword_route
from frontend.charts import revenue_chart, sentiment_chart
from frontend.companies import SAMPLE_TICKER, company_name
from frontend.components.answer import render_answer
from frontend.components.pipeline import render_pipeline, render_route_card
from frontend.components.quote_card import render_quote_card
from frontend.theme import COLORS

SAMPLES = [
    "What was AMD's revenue last quarter?",
    "How did it change from the year before?",
    "What is the tone of AMD's recent filings?",
    "What is {co} trading at right now?",
    "Should I buy AMD?",
]


def _fresh(use_fake: bool, use_laya: bool) -> Orchestrator:
    llm = FakeLLM() if use_fake else MistralLLM()
    router = None
    if use_laya:
        from agent_core.laya_router import LayaRouter
        router = LayaRouter()
    return Orchestrator(llm, memory=SharedMemory(), conversation=Conversation(), router=router)


def _run(orch: Orchestrator, question: str, ticker: str) -> None:
    orch.conversation.current_ticker = ticker
    with st.spinner("Agents working…"):
        tr = orch.ask(question)
    mem = orch.memory
    st.session_state.agent_result = {
        "tr": tr, "ticker": ticker,
        "quote": mem.get("amd_quote"), "news": mem.get("amd_news"),
        "chart": mem.get("amd_chart"), "schart": mem.get("amd_sentiment_chart"),
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


def render_agent_demo(company: str) -> None:
    c = COLORS
    st.markdown('<div style="font-size:1.5rem;font-weight:800;">Agent Demo</div>'
                f'<div style="color:{c["muted"]};font-size:0.9rem;margin-bottom:10px;">'
                'Laya routes the question, only that route\'s agents run over one shared memory, '
                'and Mistral writes the cited answer.</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    use_fake = col_a.toggle("Offline (FakeLLM)", value=False, key="use_fake",
                            help="Off = real Mistral on the PSC GPU (needs the :8000 tunnel).")
    use_laya = col_b.toggle("Route with Laya", value=False, key="use_laya",
                            help="Use Laya to route (needs the :8001 tunnel); else keyword rules.")

    # rebuild the orchestrator when the engine toggles change (preserves memory otherwise)
    if ("agent_orch" not in st.session_state
            or st.session_state.get("agent_fake") != use_fake
            or st.session_state.get("agent_laya") != use_laya):
        st.session_state.agent_orch = _fresh(use_fake, use_laya)
        st.session_state.agent_fake = use_fake
        st.session_state.agent_laya = use_laya
    orch = st.session_state.agent_orch

    _context_chips(orch)

    samples = [s.replace("{co}", company) for s in SAMPLES]
    picked = st.selectbox("Sample question (or type your own)", ["(type your own)"] + samples)
    question = st.text_input("Question", value="" if picked == "(type your own)" else picked, key="agent_q")

    r1, r2 = st.columns(2)
    run = r1.button("▶ Run agents", type="primary", use_container_width=True)
    if r2.button("↻ New conversation", use_container_width=True):
        st.session_state.agent_orch = _fresh(use_fake, use_laya)
        st.session_state.agent_result = None
        st.session_state.pending_mismatch = None
        st.rerun()

    if run and question.strip():
        intended = keyword_route(question)
        if intended in ("numbers", "sentiment") and company != SAMPLE_TICKER:
            st.session_state.pending_mismatch = {"q": question, "intended": intended, "company": company}
            st.session_state.agent_result = None
        else:
            _run(orch, question, ticker=company)

    pm = st.session_state.get("pending_mismatch")
    if pm:
        st.warning(
            f"The **{pm['intended']}** route runs on bundled **{SAMPLE_TICKER} sample data** — "
            f"{company_name(company)} ({company}) numbers aren't available offline. "
            f"Switch the sidebar company to {SAMPLE_TICKER}, or use the sample below."
        )
        if st.button(f"Use {SAMPLE_TICKER} sample data", type="primary"):
            _run(orch, pm["q"], ticker=SAMPLE_TICKER)

    res = st.session_state.get("agent_result")
    if not res:
        return
    tr = res["tr"]

    render_route_card(tr)
    render_pipeline(tr)
    if tr.reused:
        st.caption(f"♻ Reused from memory (skipped re-fetch): {', '.join(tr.reused)}")

    if tr.route == "real_time":
        q = res["quote"]
        if isinstance(q, dict) and "price" in q:
            render_quote_card(q, res["ticker"])
        if res["news"]:
            _render_news(res["news"])

    st.markdown('<div class="fs-label" style="margin-top:14px;">Answer</div>', unsafe_allow_html=True)
    render_answer(tr, orch.memory)

    chart = res["chart"]
    if tr.route == "numbers" and chart and chart.get("points"):
        st.markdown('<div class="fs-label" style="margin-top:14px;">Revenue · viz_agent</div>',
                    unsafe_allow_html=True)
        st.altair_chart(revenue_chart(chart["points"], chart.get("note", "")), use_container_width=True)
        if chart.get("note"):
            st.caption(chart["note"] + "  ·  SAMPLE DATA")

    schart = res["schart"]
    if tr.route == "sentiment" and schart and schart.get("points"):
        st.markdown('<div class="fs-label" style="margin-top:14px;">Filing sentiment · viz_agent</div>',
                    unsafe_allow_html=True)
        st.altair_chart(sentiment_chart(schart["points"]), use_container_width=True)
        if schart.get("note"):
            st.caption(schart["note"] + "  ·  SAMPLE DATA")

    with st.expander("Shared memory after this question"):
        st.code(res["memtext"])


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
