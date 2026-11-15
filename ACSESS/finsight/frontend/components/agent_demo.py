"""Agent Demo tab — routing → pipeline → cited answer, over one shared memory.

Follows the sidebar company. The real_time route works for any ticker (Finnhub);
numbers/sentiment have bundled AMD sample data only, so for another company we show
a clear notice and a "Use AMD sample" button — never a silent switch.
"""
from __future__ import annotations

import re

import streamlit as st

from agent_core.llm import FakeLLM, MistralLLM
from agent_core.memory import Conversation, SharedMemory
from agent_core.orchestrator import Orchestrator
from agent_core.orchestrator import resolve_ticker
from agent_core.orchestrator import route as keyword_route
from frontend.charts import revenue_chart, sentiment_chart
from frontend.companies import SAMPLE_TICKER, company_name
from frontend.components.answer import render_answer
from frontend.components.pipeline import render_pipeline, render_route_card
from frontend.components.quote_card import render_quote_card
from frontend.theme import COLORS

# Curated AMD demo set, grouped by the route each question takes (and the agents it
# runs). A single question uses only its route's agents by design; across the set every
# agent runs. All AMD so every question has real sample/live data.
# {co} = the selected sidebar company. Real-time (price/news) and the guardrail work for
# ANY company via Finnhub. Numbers/Sentiment/Business run on bundled AMD sample data only,
# so they stay AMD and are labelled as such. ({co} is substituted at render time.)
SAMPLE_GROUPS = [
    ("Live price & news · {co}", "any company", [
        "What is {co} trading at right now?",
        "What's the latest news on {co} today?",
    ]),
    ("Revenue & growth · AMD sample", "Data → Analysis → Viz → Report", [
        "What was AMD's revenue last quarter?",
        "How did AMD's revenue change year over year?",
    ]),
    ("Filing sentiment · {co}", "Data → Classify → Viz → Report  ·  real 10-K", [
        "What is the tone of {co}'s recent filings?",
    ]),
    ("Business · {co}", "Data → Report  ·  real 10-K", [
        "Tell me about {co}'s business and products",
    ]),
    ("Advice guardrail · {co}", "router refuses", [
        "Should I buy {co}?",
    ]),
]


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


def render_agent_demo(company: str) -> None:
    c = COLORS
    st.markdown('<div style="font-size:2.3rem;font-weight:800;margin-top:10px;letter-spacing:-0.02em;">'
                f'<span style="color:{c["text"]};">Ask Fin</span><span style="color:{c["laya"]};">Sight</span></div>'
                f'<div style="color:{c["text_dim"]};font-size:1rem;margin-bottom:14px;">'
                f'Ask about {company_name(company)} in plain English. Every number is computed in '
                'Python and every claim is cited to its source.</div>', unsafe_allow_html=True)

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


    # Search-style box with the action beside it; one sample per route as quick chips.
    qcol, acol, ncol = st.columns([6, 1, 1.4], vertical_alignment="bottom")
    # on_change fires on Enter, so Enter asks just like the button
    question = qcol.text_input("Ask a question", key="agent_q", label_visibility="collapsed",
                               placeholder=f"e.g. What is the tone of {company}'s recent filings?",
                               on_change=lambda: st.session_state.update(agent_enter=True))
    run = acol.button("Ask", type="primary", use_container_width=True, key="ask_btn") or st.session_state.pop("agent_enter", False)
    clicked = None
    samples = [qs[0].replace("{co}", company) for _, _, qs in SAMPLE_GROUPS]
    # pills wrap to the screen width; the callback clears the selection so a chip acts like a button
    def _pick():
        st.session_state.agent_chip_q = st.session_state.agent_chips
        st.session_state.agent_chips = None
    st.pills("Try one:", samples, key="agent_chips", on_change=_pick)
    clicked = st.session_state.pop("agent_chip_q", None)
    if ncol.button("↻ New chat", use_container_width=True, key="new_chat"):
        st.session_state.agent_orch = _fresh(use_fake, use_laya)
        st.session_state.agent_result = None
        st.session_state.pending_mismatch = None
        st.rerun()

    to_run = clicked or (question if (run and question.strip()) else None)
    if to_run:
        intended = keyword_route(to_run)
        # Use the ticker named in the question if any (so "AMD revenue" runs even when the
        # sidebar shows another company); else fall back to the sidebar company.
        eff = resolve_ticker(to_run, Conversation()) or company
        # Only the numbers route needs bundled data (AMD-only, no XBRL yet). Sentiment/text
        # now fetch the selected company's real SEC filing, so they run for any company.
        if intended == "numbers" and eff != SAMPLE_TICKER:
            st.session_state.pending_mismatch = {"q": to_run, "intended": intended, "company": eff}
            st.session_state.agent_result = None
        else:
            _run(orch, to_run, ticker=eff)

    pm = st.session_state.get("pending_mismatch")
    if pm:
        st.warning(
            f"The **{pm['intended']}** route runs on bundled **{SAMPLE_TICKER} sample data** — "
            f"{company_name(company)} ({company}) numbers aren't available offline. "
            f"Switch the sidebar company to {SAMPLE_TICKER}, or use the sample below."
        )
        if st.button(f"Use {SAMPLE_TICKER} sample data", type="primary"):
            # the question names the other ticker, which would override the AMD choice
            q = re.sub(rf"\b{re.escape(pm['company'])}\b", SAMPLE_TICKER, pm["q"], flags=re.I)
            _run(orch, q, ticker=SAMPLE_TICKER)
            st.rerun()  # redraw without the warning, which was drawn before the click

    res = st.session_state.get("agent_result")
    if not res:
        return
    tr = res["tr"]
    orch = st.session_state.agent_orch  # _run may have swapped in a fresh one
    src = "AMD SAMPLE DATA" if res["ticker"] == SAMPLE_TICKER else f"{res['ticker']} · LATEST 10-K"

    st.markdown('<div class="fs-label" style="margin-top:18px;">Answer</div>', unsafe_allow_html=True)
    render_answer(tr, orch.memory)

    if tr.route == "real_time":
        q = res["quote"]
        if isinstance(q, dict) and "price" in q:
            render_quote_card(q, res["ticker"])
        if res["news"]:
            _render_news(res["news"])

    chart = res["chart"]
    if tr.route == "numbers" and chart and chart.get("points"):
        st.markdown('<div class="fs-label" style="margin-top:14px;">Revenue · viz_agent '
                    f'<span style="color:{c["muted"]};">· {src}</span></div>', unsafe_allow_html=True)
        _delta_chips(res.get("notes"))
        st.altair_chart(revenue_chart(chart["points"], chart.get("note", "")), use_container_width=True)
        st.caption("Deltas computed in Python from the quarterly figures (analysis_agent) · not financial advice.")

    sent = res.get("sentiment")
    schart = res["schart"]
    if tr.route == "sentiment" and schart and schart.get("points"):
        st.markdown('<div class="fs-label" style="margin-top:14px;">Filing sentiment · classify_agent '
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
