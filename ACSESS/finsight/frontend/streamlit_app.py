"""FinSight: Financial Research Q&A Portal."""
import sys
from pathlib import Path

import streamlit as st

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.tools.edgar import get_edgar
from backend.app.tools.sections import parse_filing

st.set_page_config(
    page_title="FinSight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 FinSight: Financial Research Assistant")
st.markdown(
    "*Ask questions about US public company SEC filings. Answers backed by official documents.*"
)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    company = st.selectbox("Select Company", ["NVDA", "AAPL", "MSFT", "TSLA"])
    form_type = st.selectbox("Filing Type", ["10-K", "10-Q"])
    st.markdown("---")
    st.caption("💡 FinSight uses SEC EDGAR filings as source of truth.")
    st.caption("🔒 For research and educational purposes only.")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"{company} Financial Data")

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
    st.header("🤖 Q&A (Coming Soon)")
    st.info(
        "Next: Connect to LLM for Q&A\n\n"
        "Ask questions about the filing and get cited answers."
    )

st.markdown("---")

# Footer
col_footer1, col_footer2, col_footer3 = st.columns(3)
with col_footer1:
    st.caption("🔗 Data Source: SEC EDGAR")
with col_footer2:
    st.caption("🏛️ Educational Use Only")
with col_footer3:
    st.caption("❌ Not Financial Advice")
