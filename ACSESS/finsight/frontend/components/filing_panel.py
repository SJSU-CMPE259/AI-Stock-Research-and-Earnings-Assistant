"""Filing Q&A tab — fetch a filing, browse sections, ask Mistral with citations.

Phase 1 keeps the original behavior (moved into a function + themed shell); the
section navigator, chat history and inline [S] chips land in Phase 4.
"""
from __future__ import annotations

import streamlit as st

from backend.app.tools.edgar import get_edgar
from backend.app.tools.sections import parse_filing
from frontend.companies import company_name, logo_url
from frontend.theme import COLORS


def render_filing_qa(company: str, form_type: str) -> None:
    c = COLORS
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">'
            f'<img src="{logo_url(company)}" alt="{company}" style="width:52px;height:52px;'
            f'border-radius:12px;background:#fff;padding:7px;object-fit:contain;">'
            f'<div><div style="font-size:1.5rem;font-weight:800;line-height:1.1;">{company_name(company)}</div>'
            f'<div style="color:{c["muted"]};font-size:0.82rem;" class="fs-mono">{company} · {form_type} · SEC EDGAR</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        if st.button("Fetch latest filing", use_container_width=True, key="fetch_filing"):
            with st.spinner(f"Fetching {company}'s latest {form_type}…"):
                try:
                    filings = get_edgar().list_filings(company, form=form_type, limit=1)
                    if not filings:
                        st.error(f"No {form_type} found for {company}")
                    else:
                        filing = filings[0]
                        with st.spinner("Downloading and parsing filing…"):
                            html = get_edgar().download_filing(filing)
                            sections = parse_filing(html, filing.form)
                        st.session_state.filing = filing
                        st.session_state.sections = sections
                except Exception as e:  # noqa: BLE001 — surface a clean error, never a traceback
                    st.error(f"Could not fetch filing: {e}")

        filing = st.session_state.get("filing")
        sections = st.session_state.get("sections")
        if filing and sections is not None:
            a, b, d = st.columns(3)
            a.metric("Filed", filing.filing_date)
            b.metric("Report period", filing.report_date)
            d.metric("Accession", filing.accession[:12] + "…")
            st.markdown('<div class="fs-label" style="margin-top:8px;">Filing sections</div>',
                        unsafe_allow_html=True)
            for i, s in enumerate(sections, 1):
                stub = len(s.text) < 2000
                tag = "  ⚠ stub" if stub else ""
                with st.expander(f"[S{i}] {s.name} · {len(s.text):,} chars{tag}", expanded=(i == 1)):
                    st.text_area("content", value=s.text[:2000] + "…", height=240,
                                 disabled=True, key=f"sec_{i}", label_visibility="collapsed")

    with col2:
        st.markdown('<div style="font-size:1.4rem;font-weight:800;">Ask Mistral</div>'
                    f'<div style="color:{c["muted"]};font-size:0.85rem;margin-bottom:10px;">'
                    'Grounded in the fetched filing — answers cite their sources.</div>',
                    unsafe_allow_html=True)

        if "sections" not in st.session_state:
            st.markdown(
                f'<div style="border:1px dashed {c["border"]};border-radius:12px;padding:22px;'
                f'text-align:center;color:{c["muted"]};">Fetch a filing first, then ask here.</div>',
                unsafe_allow_html=True,
            )
            return

        sections = st.session_state.sections
        filing = st.session_state.filing
        question = st.text_area("Your question", height=100, label_visibility="collapsed",
                                placeholder="e.g. What are the main risk factors?")
        if st.button("Ask Mistral", use_container_width=True, type="primary", key="ask_filing"):
            if not question.strip():
                st.warning("Type a question first.")
                return
            with st.spinner("Mistral is reading the filing on the GPU…"):
                try:
                    from backend.app.llm import get_llm

                    context = "\n\n".join(
                        f"[S{i}] {s.name}:\n{s.text[:3000]}" for i, s in enumerate(sections, 1)
                    )
                    system = (
                        "You are FinSight, a financial research assistant. Answer using ONLY the "
                        "SEC filing excerpts below. Cite sources with tags like [S1], [S2]. If the "
                        "answer is not in the excerpts, say so plainly. Never give financial advice "
                        f"or invent numbers.\n\nCompany: {company} — {filing.form}, filed "
                        f"{filing.filing_date}.\n\nFILING EXCERPTS:\n{context}"
                    )
                    answer = get_llm().complete(
                        system=system, messages=[{"role": "user", "content": question}],
                        max_tokens=512, temperature=0.2,
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"LLM error: {e}")
                    return

            with st.container(border=True):
                st.markdown(f'<div class="fs-label" style="color:{c["mistral"]};">📝 Research answer · '
                            f'<span style="color:{c["muted"]};">Mistral-7B · PSC GPU</span></div>',
                            unsafe_allow_html=True)
                st.markdown(answer)
                chips = "".join(
                    f'<span class="fs-chip" style="background:{c["sec"]}1a;color:{c["sec"]};'
                    f'border-color:{c["sec"]}55;">[S{i}] {s.name}</span>'
                    for i, s in enumerate(sections, 1)
                )
                st.markdown(f'<div style="margin-top:12px;display:flex;flex-wrap:wrap;gap:6px;">{chips}</div>',
                            unsafe_allow_html=True)
            st.caption(f"Source: {company} {filing.form}, filed {filing.filing_date} · SEC EDGAR · not financial advice")
