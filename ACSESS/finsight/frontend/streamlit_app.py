"""FinSight: Financial Research Q&A Portal."""
import sys
from pathlib import Path

import streamlit as st

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")  # so status chips see FINNHUB_API_KEY

from frontend.companies import COMPANY_INFO
from frontend.theme import COLORS, apply_theme, set_mode
from frontend.components.header import render_header
from frontend.components.status_chips import chips_html
from frontend.components.filing_panel import render_filing_qa
from frontend.components.how_it_works import render_how_it_works
from frontend.components.agent_demo import render_agent_demo


_ASSETS = Path(__file__).parent / "assets"
st.set_page_config(
    page_title="FinSight — SEC research",
    page_icon=str(_ASSETS / "logo.svg"),
    layout="wide",
    initial_sidebar_state="collapsed",  # the sidebar only holds settings now
)
# the toggle below sets this key; reading it first styles the whole run
set_mode(st.session_state.get("light_mode", False))
apply_theme()

# Header: brand on the left, company picker on the right (as in Anna's mockup)
brand, pick = st.columns([3, 1.2], vertical_alignment="center")
with brand:
    render_header()
with pick.container(horizontal=True, horizontal_alignment="right"):
    company = st.selectbox("Company", list(COMPANY_INFO), width=230,
                           format_func=lambda t: f"{COMPANY_INFO[t]['short']} ({t})")
st.markdown(f'<div style="border-bottom:1px solid {COLORS["border"]};margin:4px 0 6px;"></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.toggle("☀️ Light mode", key="light_mode")
    st.markdown('<div class="fs-label" style="margin-top:12px;">Status</div>', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 12px;">'
                + chips_html(st.session_state.get("use_fake", False), st.session_state.get("use_laya", False))
                + "</div>", unsafe_allow_html=True)

tab_agent, tab_filing, tab_how = st.tabs(["Chat", "Filing Q&A", "How it works"])
with tab_agent:
    render_agent_demo(company)
with tab_filing:
    form_type = st.selectbox("Filing type", ["10-K", "10-Q"], key="form_type", width=260,
                             format_func=lambda f: {"10-K": "10-K · Annual report",
                                                    "10-Q": "10-Q · Quarterly report"}[f])
    render_filing_qa(company, form_type)
with tab_how:
    render_how_it_works()
with st.sidebar:
    st.caption("SEC EDGAR filings as source of truth.")
    st.caption("Research / educational use only · not financial advice.")
st.stop()
