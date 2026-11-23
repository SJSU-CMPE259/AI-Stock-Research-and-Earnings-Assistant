"""FinSight: Financial Research Q&A Portal."""
import sys
from pathlib import Path

import streamlit as st

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")  # so status chips see FINNHUB_API_KEY

from frontend.companies import COMPANY_INFO, SAMPLE_TICKER, company_name, logo_url
from frontend.theme import COLORS, apply_theme, chip, set_mode
from frontend.components.header import render_header
from frontend.components.filing_panel import render_filing_qa
from frontend.components.how_it_works import render_how_it_works
from frontend.components.agent_demo import render_agent_demo


_ASSETS = Path(__file__).parent / "assets"
st.set_page_config(
    page_title="FinSight — SEC research",
    page_icon=str(_ASSETS / "logo.svg"),
    layout="wide",
    initial_sidebar_state="auto",  # open on desktop, closed on phones
)
# the toggle below sets this key; reading it first styles the whole run
set_mode(st.session_state.get("light_mode", False))
apply_theme()

with st.sidebar:
    st.toggle("☀️ Light mode", key="light_mode")
    st.markdown("---")
    st.caption("SEC EDGAR filings as source of truth.")
    st.caption("Research / educational use only · not financial advice.")

render_header(st.session_state.get("use_fake", False), st.session_state.get("use_laya", False))

# Company + filing-type pickers live in the main page (always visible), not in the
# collapsible sidebar where they were easy to miss when it was collapsed.
_c_company, _c_form, _c_logo = st.columns([4, 2, 1], vertical_alignment="center")
with _c_company:
    st.markdown('<div class="fs-label">Company</div>', unsafe_allow_html=True)
    company = st.selectbox(
        "Company", list(COMPANY_INFO),
        format_func=lambda t: f"{t} — {COMPANY_INFO[t]['name']}",
        label_visibility="collapsed",
    )
with _c_form:
    st.markdown('<div class="fs-label">Filing type</div>', unsafe_allow_html=True)
    form_type = st.selectbox("Filing type", ["10-K", "10-Q"], label_visibility="collapsed")
with _c_logo:
    st.markdown(
        f'<img src="{logo_url(company)}" alt="{company}" style="width:52px;height:52px;'
        f'border-radius:12px;background:#fff;padding:6px;object-fit:contain;">',
        unsafe_allow_html=True,
    )
st.markdown("")  # small spacer before the tabs

tab_agent, tab_filing, tab_how = st.tabs(["💬 Ask FinSight", "📄 Filing Q&A", "ⓘ How it works"])
with tab_agent:
    render_agent_demo(company)
with tab_filing:
    render_filing_qa(company, form_type)
with tab_how:
    render_how_it_works()
st.stop()
