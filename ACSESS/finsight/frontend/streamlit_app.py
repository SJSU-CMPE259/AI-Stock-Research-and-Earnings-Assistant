"""FinSight: Financial Research Q&A Portal."""
import sys
from pathlib import Path

import streamlit as st

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.tools.edgar import get_edgar
from backend.app.tools.sections import parse_filing

COMPANY_INFO = {
    "NVDA": {"name": "NVIDIA Corp.", "color": "#76b900"},
    "AAPL": {"name": "Apple Inc.", "color": "#555555"},
    "MSFT": {"name": "Microsoft Corp.", "color": "#0078d4"},
    "TSLA": {"name": "Tesla, Inc.", "color": "#e82127"},
}


def logo_url(ticker: str) -> str:
    """Stock logo by ticker (FinancialModelingPrep, works for any US ticker)."""
    return f"https://financialmodelingprep.com/image-stock/{ticker}.png"


_AGENT_ICON = {
    "router": "🧭", "market_agent": "📈", "data_agent": "📂", "analysis_agent": "🔢",
    "classify_agent": "🏷️", "viz_agent": "📊", "report_agent": "📝",
}


def _port_open(port: int, timeout: float = 0.4) -> bool:
    """Quick check whether a local tunnel/service is listening (for the status banner)."""
    import socket
    try:
        with socket.create_connection(("localhost", port), timeout=timeout):
            return True
    except OSError:
        return False


def _render_status(use_fake: bool, use_laya: bool):
    """A row of chips showing whether each backend is live or in fallback."""
    import os

    if use_fake:
        mistral = ("Mistral", "offline (FakeLLM)", "#8a9ba3")
    else:
        up = _port_open(8000)
        mistral = ("Mistral", "live · PSC :8000" if up else "DOWN · tunnel :8000", "#3ec98a" if up else "#e82127")
    if use_laya:
        up = _port_open(8001)
        laya = ("Laya router", "live · :8001" if up else "DOWN · tunnel :8001", "#3ec98a" if up else "#e82127")
    else:
        laya = ("Router", "keyword rules", "#8a9ba3")
    has_key = bool(os.getenv("FINNHUB_API_KEY"))
    finnhub = ("Finnhub", "live · price+news" if has_key else "no key · sample", "#3ec98a" if has_key else "#f0ad4e")

    chips = "".join(
        f'<span style="display:inline-block;margin:2px 6px 2px 0;padding:3px 10px;border-radius:12px;'
        f'background:{c}22;color:{c};border:1px solid {c}55;font-size:0.8rem;font-weight:600;">'
        f'● {name}: {state}</span>'
        for name, state, c in (mistral, laya, finnhub)
    )
    st.markdown(chips, unsafe_allow_html=True)


def _render_trace(tr):
    """Agent steps as a compact timeline with per-agent icons."""
    st.markdown(f"**Route:** `{tr.route}`  ·  **ticker:** `{tr.ticker or '—'}`  ·  **{tr.elapsed_s * 1000:.0f} ms**")
    rows = []
    for step in tr.steps:
        name, _, msg = step.partition(":")
        key = name.strip().split(" ")[0]
        icon = _AGENT_ICON.get(key, "•")
        fail = "FAILED" in step
        color = "#e82127" if fail else "#2b2b2b"
        rows.append(
            f'<div style="display:flex;gap:8px;padding:4px 0;border-bottom:1px solid #eee;">'
            f'<span style="font-size:1rem;">{icon}</span>'
            f'<span><b style="color:{color};">{name.strip()}</b>'
            f'<span style="color:#555;">{(":" + msg).replace("$", "&#36;") if msg else ""}</span></span></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def _render_price_card(q: dict, ticker: str | None = None):
    """fiscal.ai-style dark price panel: logo + big price + change badge + area chart."""
    tk = (ticker or "").upper() or "—"
    name = COMPANY_INFO.get(tk, {}).get("name", tk)
    live = q.get("live")
    price = q.get("price", 0.0)
    cur = q.get("currency", "USD")
    chg = q.get("change_pct")
    chg_abs = q.get("change")
    up = isinstance(chg, (int, float)) and chg >= 0
    badge_bg = "#13432e" if up else "#431313"
    badge_fg = "#3ec98a" if up else "#ff6b6b"
    arrow = "▲" if up else "▼"
    badge = ""
    if isinstance(chg, (int, float)):
        abs_txt = f"{chg_abs:+,.2f} " if isinstance(chg_abs, (int, float)) else ""
        badge = (f'<span style="background:{badge_bg};color:{badge_fg};padding:4px 12px;'
                 f'border-radius:8px;font-size:1.05rem;font-weight:700;margin-left:12px;">'
                 f'{arrow} {abs_txt}({chg:+.2f}%)</span>')
    meta = f"as of {q.get('as_of', '')[:16].replace('T', ' ')} UTC"
    meta += " · market closed" if q.get("market_open_likely") is False else ""
    meta += " · Finnhub (live)" if live else " · sample data"
    logo = (f'<img src="{logo_url(tk)}" style="width:44px;height:44px;border-radius:10px;'
            f'background:#fff;padding:3px;object-fit:contain;" '
            f'onerror="this.style.display=\'none\'">') if tk != "—" else ""

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0f2027,#203a43);border-radius:16px;'
        f'padding:22px 26px;margin:10px 0;color:#fff;box-shadow:0 4px 18px rgba(0,0,0,.25);">'
        f'<div style="display:flex;align-items:center;gap:14px;">{logo}'
        f'<div><div style="font-size:1.5rem;font-weight:800;line-height:1.1;">{name}</div>'
        f'<div style="color:#9fb3bd;font-size:0.9rem;">{tk}</div></div></div>'
        f'<div style="margin-top:14px;display:flex;align-items:baseline;flex-wrap:wrap;">'
        f'<span style="font-size:2.6rem;font-weight:800;">${price:,.2f}</span>'
        f'<span style="color:#9fb3bd;font-size:1.1rem;margin-left:8px;">{cur}</span>{badge}</div>'
        f'<div style="color:#9fb3bd;font-size:0.82rem;margin-top:8px;">{meta}</div></div>',
        unsafe_allow_html=True,
    )


_CAT_COLOR = {"company": "#1f7a8c", "technology": "#6f42c1", "regulatory": "#b7791f",
              "product launch": "#2b8a3e", "earnings": "#c92a2a"}


def _render_headlines(news: list):
    """fiscal.ai-style news cards: tag + big headline + summary + date."""
    st.markdown('<div style="font-size:1.3rem;font-weight:800;margin:14px 0 6px;">📰 Recent news</div>',
                unsafe_allow_html=True)
    for a in news:
        cat = (a.get("category") or a.get("relevance") or "news").upper()
        ccol = _CAT_COLOR.get((a.get("category") or "").lower(), "#1f7a8c")
        summ = (a.get("summary") or "").replace("$", "&#36;")
        summ = (summ[:180] + "…") if len(summ) > 180 else summ
        st.markdown(
            f'<div style="padding:14px 16px;margin:8px 0;border:1px solid #e6e9ec;border-radius:12px;'
            f'background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.04);">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="background:{ccol}1a;color:{ccol};font-size:0.68rem;font-weight:700;'
            f'padding:2px 8px;border-radius:6px;letter-spacing:.03em;">{cat}</span>'
            f'<span style="color:#8a96a0;font-size:0.8rem;">{a.get("source","")} · {a.get("date","")}</span></div>'
            f'<a href="{a.get("url","#")}" target="_blank" style="display:block;margin-top:8px;'
            f'font-size:1.12rem;font-weight:700;color:#13283a;text-decoration:none;line-height:1.3;">'
            f'{a.get("headline","").replace("$","&#36;")}</a>'
            + (f'<div style="color:#5a6b78;font-size:0.9rem;margin-top:6px;line-height:1.45;">{summ}</div>' if summ else "")
            + '</div>',
            unsafe_allow_html=True,
        )


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_history(ticker: str, period: str):
    """Daily close history from Yahoo (yfinance, free, no key). Cached 30 min."""
    import yfinance as yf
    h = yf.Ticker(ticker).history(period=period, interval="1d")
    if h is None or h.empty:
        return None
    return [(d.strftime("%Y-%m-%d"), float(c)) for d, c in zip(h.index, h["Close"])]


_PERIODS = {"1M": "1mo", "6M": "6mo", "YTD": "ytd", "1Y": "1y", "5Y": "5y", "MAX": "max"}


def _render_price_history(ticker: str | None):
    """fiscal.ai-style historical area chart with range buttons."""
    tk = (ticker or "").upper()
    if not tk:
        return
    labels = list(_PERIODS)
    sel = st.radio("Range", labels, index=labels.index("1Y"), horizontal=True,
                   key="pr_range", label_visibility="collapsed")
    data = _fetch_history(tk, _PERIODS[sel])
    if not data:
        st.caption("📉 Price history unavailable for this ticker.")
        return
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(data, columns=["date", "price"])
    df["date"] = pd.to_datetime(df["date"])
    first, last = df["price"].iloc[0], df["price"].iloc[-1]
    chg = last - first
    pct = (chg / first * 100) if first else 0
    up = chg >= 0
    col = "#2ecc71" if up else "#e74c3c"
    st.markdown(
        f'<div style="font-size:1.7rem;font-weight:800;margin-top:6px;">${last:,.2f} '
        f'<span style="color:{col};font-size:1rem;font-weight:700;">'
        f'{"+" if up else ""}{chg:,.2f} ({pct:+.1f}%) past {sel}</span></div>',
        unsafe_allow_html=True,
    )
    area = (
        alt.Chart(df).mark_area(
            line={"color": col, "strokeWidth": 2},
            opacity=0.3,
            color=alt.Gradient(gradient="linear",
                               stops=[alt.GradientStop(color=col, offset=0),
                                      alt.GradientStop(color="#ffffff00", offset=1)],
                               x1=1, x2=1, y1=1, y2=0),
        ).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(labelColor="#667", grid=False)),
            y=alt.Y("price:Q", title=None, scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelColor="#667", grid=True, gridColor="#eee")),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("price:Q", format="$,.2f")],
        ).properties(height=280)
    )
    st.altair_chart(area, use_container_width=True)
    st.caption(f"Source: Yahoo Finance · {tk} · {len(df)} trading days")


def render_agent_demo():
    """The multi-agent core demo (agent_core/) as an interactive page."""
    from agent_core.llm import FakeLLM, MistralLLM
    from agent_core.memory import Conversation, SharedMemory
    from agent_core.orchestrator import Orchestrator

    st.markdown(
        '<div style="font-size:2rem;font-weight:800;margin-bottom:2px;">🤖 Agent Demo</div>'
        '<div style="color:#5a6b78;font-size:1rem;margin-bottom:10px;">Watch the orchestrator '
        'route each question and the agents work over one shared memory — live price &amp; news '
        'from Finnhub, answers written by Mistral on the PSC GPU.</div>',
        unsafe_allow_html=True,
    )

    use_fake = st.toggle(
        "Offline (FakeLLM)", value=False,
        help="Off = real Mistral on your PSC GPU (needs the tunnel on :8000)",
    )
    use_laya = st.toggle(
        "🧭 Route with Laya (live)", value=False,
        help="Route the question with Laya instead of keyword rules (needs the Laya tunnel on :8001). "
             "Laya was the best router in the eval; Mistral still writes the answer.",
    )
    _render_status(use_fake, use_laya)
    samples = [
        "What was AMD's revenue last quarter?",
        "How did it change from the year before?",
        "What is the tone of AMD's recent filings?",
        "What is AMD's stock price right now?",
        "Should I buy AMD?",
    ]
    picked = st.selectbox("Sample question (or type your own below)", ["(type your own)"] + samples)
    question = st.text_input("Question", value="" if picked == "(type your own)" else picked, key="agent_q")

    def _fresh():
        llm = FakeLLM() if use_fake else MistralLLM()
        router = None
        if use_laya:
            from agent_core.laya_router import LayaRouter
            router = LayaRouter()
        return Orchestrator(llm, memory=SharedMemory(), conversation=Conversation(), router=router)

    # One orchestrator per session preserves conversation memory (ticker recall, stored reuse).
    if ("agent_orch" not in st.session_state
            or st.session_state.get("agent_fake") != use_fake
            or st.session_state.get("agent_laya") != use_laya):
        st.session_state.agent_orch = _fresh()
        st.session_state.agent_fake = use_fake
        st.session_state.agent_laya = use_laya

    c1, c2 = st.columns(2)
    run = c1.button("▶️ Run agents", type="primary", use_container_width=True)
    if c2.button("🔄 New conversation", use_container_width=True):
        st.session_state.agent_orch = _fresh()
        st.rerun()

    if run and question.strip():
        with st.spinner("Agents working… (real Mistral sentiment path makes ~8 calls)"):
            tr = st.session_state.agent_orch.ask(question)
        mem = st.session_state.agent_orch.memory
        # Snapshot results so range-button reruns don't wipe the output.
        st.session_state.agent_result = {
            "tr": tr, "quote": mem.get("amd_quote"), "news": mem.get("amd_news"),
            "chart": mem.get("amd_chart"), "schart": mem.get("amd_sentiment_chart"),
            "memtext": mem.describe() or "(empty)",
        }

    res = st.session_state.get("agent_result")
    if res:
        tr = res["tr"]
        st.markdown('<div style="font-size:1.3rem;font-weight:800;margin:10px 0 4px;">⚙️ Agent steps</div>',
                    unsafe_allow_html=True)
        _render_trace(tr)
        if tr.reused:
            st.success(f"♻️ Reused from memory (skipped re-fetch): {', '.join(tr.reused)}")

        # Live price panel + historical chart + headlines on the real_time route
        if tr.route == "real_time":
            q = res["quote"]
            if isinstance(q, dict) and "price" in q:
                _render_price_card(q, tr.ticker)
                _render_price_history(tr.ticker)
            if res["news"]:
                _render_headlines(res["news"])

        st.markdown('<div style="font-size:1.3rem;font-weight:800;margin:14px 0 4px;">📝 Answer</div>',
                    unsafe_allow_html=True)
        st.info(tr.answer.replace("$", "\\$"))

        chart = res["chart"]
        if tr.route == "numbers" and chart and chart.get("points"):
            import altair as alt
            import pandas as pd

            df = pd.DataFrame(chart["points"])
            df["latest"] = [i == len(df) - 1 for i in range(len(df))]
            bars = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("period:N", sort=None, title=chart["x_label"]),
                    y=alt.Y("revenue_b:Q", title=chart["y_label"]),
                    color=alt.condition(alt.datum.latest, alt.value("#3ec98a"), alt.value("#1f7a8c")),
                    tooltip=["period", "revenue_b"],
                )
                .properties(title=chart["title"], height=260)
            )
            st.markdown("**📊 Revenue chart** (from `viz_agent`)")
            st.altair_chart(bars, use_container_width=True)
            if chart.get("note"):
                st.caption(chart["note"].replace("$", "\\$"))

        schart = res["schart"]
        if tr.route == "sentiment" and schart and schart.get("points"):
            import altair as alt
            import pandas as pd

            sdf = pd.DataFrame(schart["points"])
            sbars = (
                alt.Chart(sdf)
                .mark_bar()
                .encode(
                    x=alt.X("label:N", sort=["positive", "neutral", "negative", "uncertain"],
                            title=schart["x_label"]),
                    y=alt.Y("count:Q", title=schart["y_label"]),
                    color=alt.Color(
                        "label:N",
                        scale=alt.Scale(
                            domain=["positive", "neutral", "negative", "uncertain"],
                            range=["#3ec98a", "#8a9ba3", "#e82127", "#f0ad4e"],
                        ),
                        legend=None,
                    ),
                    tooltip=["label", "count"],
                )
                .properties(title=schart["title"], height=260)
            )
            st.markdown("**📊 Sentiment breakdown** (from `viz_agent`)")
            st.altair_chart(sbars, use_container_width=True)
            if schart.get("note"):
                st.caption(schart["note"])

        with st.expander("🧠 Shared memory after this question"):
            st.code(res["memtext"])


st.set_page_config(
    page_title="FinSight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .finsight-hero {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 24px;
        background: linear-gradient(135deg, #0b2a3d 0%, #11506e 55%, #1f7a8c 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 30px 40px;
        margin-bottom: 18px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }
    .finsight-hero h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .finsight-hero p {
        margin: 10px 0 16px 0;
        font-size: 1.05rem;
        color: #cfe8f2;
        max-width: 560px;
        line-height: 1.5;
    }
    .hero-badges { display: flex; flex-wrap: wrap; gap: 10px; }
    .hero-badges span {
        font-size: 0.8rem;
        font-weight: 600;
        color: #eaf6fb;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 5px 12px;
        border-radius: 20px;
    }
    .hero-art { flex-shrink: 0; }
    @media (max-width: 900px) { .hero-art { display: none; } }
    </style>

    <div class="finsight-hero">
        <div class="hero-text">
            <h1>📊 FinSight</h1>
            <p>AI-powered financial research on SEC filings — every answer grounded
            in official 10-K &amp; 10-Q documents, with citations you can trace.</p>
            <div class="hero-badges">
                <span>● Live SEC EDGAR</span>
                <span>● Mistral-7B on GPU</span>
                <span>● Cited Answers</span>
            </div>
        </div>
        <div class="hero-art">
            <svg width="260" height="140" viewBox="0 0 260 140" fill="none"
                 xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#7fe3b0" stop-opacity="0.55"/>
                        <stop offset="100%" stop-color="#7fe3b0" stop-opacity="0"/>
                    </linearGradient>
                </defs>
                <line x1="10" y1="30" x2="250" y2="30" stroke="#ffffff" stroke-opacity="0.08"/>
                <line x1="10" y1="65" x2="250" y2="65" stroke="#ffffff" stroke-opacity="0.08"/>
                <line x1="10" y1="100" x2="250" y2="100" stroke="#ffffff" stroke-opacity="0.08"/>
                <path d="M10 110 L50 92 L90 98 L130 66 L170 72 L210 40 L250 24 L250 130 L10 130 Z"
                      fill="url(#area)"/>
                <path d="M10 110 L50 92 L90 98 L130 66 L170 72 L210 40 L250 24"
                      stroke="#7fe3b0" stroke-width="3" fill="none"
                      stroke-linecap="round" stroke-linejoin="round"/>
                <g fill="#ffffff">
                    <circle cx="50" cy="92" r="3.5"/>
                    <circle cx="130" cy="66" r="3.5"/>
                    <circle cx="210" cy="40" r="3.5"/>
                    <circle cx="250" cy="24" r="4" fill="#7fe3b0"/>
                </g>
                <g fill="#3ec98a" opacity="0.9">
                    <rect x="44" y="104" width="7" height="20" rx="1.5"/>
                    <rect x="124" y="78" width="7" height="46" rx="1.5"/>
                    <rect x="204" y="52" width="7" height="72" rx="1.5"/>
                </g>
            </svg>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    company = st.selectbox(
        "Select Company",
        ["NVDA", "AAPL", "MSFT", "TSLA"],
        format_func=lambda t: f"{t} — {COMPANY_INFO[t]['name']}",
    )
    st.markdown(
        f'<div style="text-align:center;margin:12px 0;">'
        f'<img src="{logo_url(company)}" alt="{company}" '
        f'style="width:76px;height:76px;border-radius:16px;background:#fff;'
        f'padding:9px;object-fit:contain;box-shadow:0 2px 10px rgba(0,0,0,0.2);"/>'
        f"</div>",
        unsafe_allow_html=True,
    )
    form_type = st.selectbox("Filing Type", ["10-K", "10-Q"])
    st.markdown("---")
    mode = st.radio("Mode", ["📄 Filing Q&A", "🤖 Agent Demo"], index=0)
    st.markdown("---")
    st.caption("💡 FinSight uses SEC EDGAR filings as source of truth.")
    st.caption("🔒 For research and educational purposes only.")

if mode == "🤖 Agent Demo":
    render_agent_demo()
    st.stop()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        f"""
        <style>
        .company-header {{ display: flex; align-items: center; gap: 14px; margin-bottom: 12px; }}
        .company-logo {{
            width: 56px; height: 56px; border-radius: 12px; background: #fff;
            padding: 7px; object-fit: contain; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
        }}
        .company-name {{ font-size: 1.5rem; font-weight: 800; line-height: 1.1; color: inherit; }}
        .company-meta {{ font-size: 0.85rem; color: #8a9ba3; margin-top: 2px; }}
        </style>
        <div class="company-header">
            <img class="company-logo" src="{logo_url(company)}" alt="{company}"/>
            <div>
                <div class="company-name">{COMPANY_INFO[company]['name']}</div>
                <div class="company-meta">{company} · {form_type} · SEC EDGAR</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("📥 Fetch Latest Filing", use_container_width=True):
        with st.spinner(f"Fetching {company}'s latest {form_type}..."):
            try:
                edgar = get_edgar()

                # Get filing
                filings = edgar.list_filings(company, form=form_type, limit=1)
                if not filings:
                    st.error(f"No {form_type} found for {company}")
                else:
                    filing = filings[0]

                    # Display filing metadata
                    st.success(f"✅ Found {filing.form} filing")

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Filing Date", filing.filing_date)
                    with col_b:
                        st.metric("Report Date", filing.report_date)
                    with col_c:
                        st.metric("Accession", filing.accession[:12] + "...")

                    # Download and parse
                    with st.spinner("Downloading and parsing filing..."):
                        html = edgar.download_filing(filing)
                        sections = parse_filing(html, filing.form)

                    st.success(f"✅ Extracted {len(sections)} sections")

                    # Display sections
                    st.subheader("📋 Filing Sections")
                    for i, section in enumerate(sections, 1):
                        with st.expander(
                            f"{section.name} ({len(section.text):,} chars)",
                            expanded=(i == 1),
                        ):
                            st.text_area(
                                f"Content",
                                value=section.text[:2000] + "...",
                                height=300,
                                disabled=True,
                                key=f"section_{i}",
                            )

                    # Store in session
                    st.session_state.filing = filing
                    st.session_state.sections = sections

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with col2:
    st.markdown(
        """
        <style>
        .qa-head { font-size: 1.55rem; font-weight: 800; color: inherit; margin: 0; }
        .qa-sub { color: #8a9ba3; font-size: 0.88rem; margin: 2px 0 14px 0; }
        .qa-answer-head {
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; color: #1f7a8c;
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 6px;
        }
        .qa-answer-head span {
            color: #8a9ba3; font-weight: 600; letter-spacing: 0; text-transform: none;
        }
        .src-chip-row { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 6px; }
        .src-chip {
            font-size: 0.72rem; font-weight: 600; color: #0b2a3d;
            background: #cfe8f2; border-radius: 14px; padding: 3px 10px;
            display: inline-block;
        }
        .qa-empty {
            border: 1px dashed rgba(127, 150, 162, 0.4); border-radius: 12px;
            padding: 22px; text-align: center; color: #8a9ba3; font-size: 0.9rem;
        }
        </style>
        <div class="qa-head">🤖 Ask Mistral</div>
        <div class="qa-sub">Grounded in the filing you fetch — answers cite their sources.</div>
        """,
        unsafe_allow_html=True,
    )

    if "sections" not in st.session_state:
        st.markdown(
            '<div class="qa-empty">📄 Fetch a filing first, then ask your '
            "question here.</div>",
            unsafe_allow_html=True,
        )
    else:
        sections = st.session_state.sections
        filing = st.session_state.filing

        question = st.text_area(
            "Your question",
            placeholder="e.g. What are the main risk factors? How did revenue change?",
            height=100,
            label_visibility="collapsed",
        )

        if st.button("🔍 Ask Mistral", use_container_width=True, type="primary"):
            if not question.strip():
                st.warning("Type a question first.")
            else:
                with st.spinner("Mistral is reading the filing on the GPU..."):
                    try:
                        from backend.app.llm import get_llm

                        max_chars_per_section = 3000
                        context = "\n\n".join(
                            f"[S{i}] {s.name}:\n{s.text[:max_chars_per_section]}"
                            for i, s in enumerate(sections, 1)
                        )

                        system = (
                            "You are FinSight, a financial research assistant. "
                            "Answer the question using ONLY the SEC filing excerpts "
                            "below. Cite the sources you use with their tags like "
                            "[S1], [S2]. If the answer is not in the excerpts, say so "
                            "plainly. Never give financial advice or invent numbers.\n\n"
                            f"Company: {company} — {filing.form}, filed {filing.filing_date}.\n\n"
                            f"FILING EXCERPTS:\n{context}"
                        )

                        answer = get_llm().complete(
                            system=system,
                            messages=[{"role": "user", "content": question}],
                            max_tokens=512,
                            temperature=0.2,
                        )

                        with st.container(border=True):
                            st.markdown(
                                '<div class="qa-answer-head">📝 Research Answer'
                                "<span>Mistral-7B · PSC GPU</span></div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(answer)
                            chips = "".join(
                                f'<span class="src-chip">[S{i}] {s.name}</span>'
                                for i, s in enumerate(sections, 1)
                            )
                            st.markdown(
                                f'<div class="src-chip-row">{chips}</div>',
                                unsafe_allow_html=True,
                            )

                        st.caption(
                            f"Source: {company} {filing.form}, filed {filing.filing_date} · SEC EDGAR"
                        )
                    except Exception as e:
                        st.error(f"❌ LLM error: {e}")

st.markdown("---")

# Footer
col_footer1, col_footer2, col_footer3 = st.columns(3)
with col_footer1:
    st.caption("🔗 Data Source: SEC EDGAR")
with col_footer2:
    st.caption("🏛️ Educational Use Only")
with col_footer3:
    st.caption("❌ Not Financial Advice")
