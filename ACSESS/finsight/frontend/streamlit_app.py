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
    st.caption("💡 FinSight uses SEC EDGAR filings as source of truth.")
    st.caption("🔒 For research and educational purposes only.")

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
