"""FinSight: Financial Research Q&A Portal."""
import sys
from pathlib import Path

import streamlit as st

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")  # so status chips see FINNHUB_API_KEY

from frontend.companies import COMPANY_INFO, SAMPLE_TICKER, company_name, logo_url
from frontend.theme import COLORS, apply_theme, chip
from frontend.components.header import render_header
from frontend.components.filing_panel import render_filing_qa
from frontend.components.how_it_works import render_how_it_works


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
        "Offline (FakeLLM)", value=False, key="use_fake",
        help="Off = real Mistral on your PSC GPU (needs the tunnel on :8000)",
    )
    use_laya = st.toggle(
        "🧭 Route with Laya (live)", value=False, key="use_laya",
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


_ASSETS = Path(__file__).parent / "assets"
st.set_page_config(
    page_title="FinSight — SEC research",
    page_icon=str(_ASSETS / "logo.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

with st.sidebar:
    st.markdown('<div class="fs-label">Company</div>', unsafe_allow_html=True)
    company = st.selectbox(
        "Company", list(COMPANY_INFO),
        format_func=lambda t: f"{t} — {COMPANY_INFO[t]['name']}",
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div style="text-align:center;margin:10px 0;">'
        f'<img src="{logo_url(company)}" alt="{company}" style="width:68px;height:68px;'
        f'border-radius:14px;background:#fff;padding:8px;object-fit:contain;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="fs-label">Filing type</div>', unsafe_allow_html=True)
    form_type = st.selectbox("Filing type", ["10-K", "10-Q"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("SEC EDGAR filings as source of truth.")
    st.caption("Research / educational use only · not financial advice.")

render_header(st.session_state.get("use_fake", False), st.session_state.get("use_laya", False))

tab_filing, tab_agent, tab_how = st.tabs(["📄 Filing Q&A", "🔀 Agent Demo", "ⓘ How it works"])
with tab_filing:
    render_filing_qa(company, form_type)
with tab_agent:
    render_agent_demo()
with tab_how:
    render_how_it_works()
st.stop()
