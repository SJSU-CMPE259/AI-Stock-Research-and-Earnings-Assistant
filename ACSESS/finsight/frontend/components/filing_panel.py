"""Filing Q&A tab — fetch a filing, browse its sections, ask Mistral with [S] citations.

A filing header card (with the SEC EDGAR link), a section navigator with character
counts and stub warnings, suggested-question chips, a chat-style history, and inline
[S] citation chips. Readable section text (no dimmed disabled inputs). Clear error
states for SEC rate limits, missing SEC_USER_AGENT, and the GPU being offline.
"""
from __future__ import annotations

import html as _html

import streamlit as st

from frontend.companies import company_name, logo_url
from frontend.components.answer import _inline
from frontend.theme import COLORS

SUGGESTED = [
    "What are the main risk factors?",
    "Summarize management's discussion of results.",
    "What drives revenue growth?",
    "What is the tone of this filing?",
]


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch(company: str, form_type: str):
    """Fetch + parse the latest filing. Cached by (ticker, form). Returns (meta, sections)
    or raises a classified error message."""
    from backend.app.tools.edgar import get_edgar
    from backend.app.tools.sections import parse_filing

    edgar = get_edgar()
    filings = edgar.list_filings(company, form=form_type, limit=1)
    if not filings:
        raise RuntimeError(f"No {form_type} found for {company} on SEC EDGAR.")
    f = filings[0]
    html = edgar.download_filing(f)
    sections = parse_filing(html, f.form)
    meta = {"form": f.form, "filing_date": f.filing_date, "report_date": f.report_date,
            "accession": f.accession, "url": f.url}
    secs = [{"name": s.name, "text": s.text} for s in sections]
    return meta, secs


@st.cache_data(ttl=86400, show_spinner=False)
def _key_points(accession: str, company: str, _system: str) -> str:
    """Cited 'key points' summary (cached by accession). _system carries the excerpts
    (underscore -> not part of the cache key, so it's keyed by the filing itself)."""
    from backend.app.llm import get_llm
    q = ("List 3-4 key points a reader should know from this filing. Put a [F#] citation "
         "after each. Do NOT include any number unless you copy it verbatim from an excerpt. "
         "No investment advice.")
    return get_llm().complete(system=_system, messages=[{"role": "user", "content": q}],
                              max_tokens=400, temperature=0.2)


def _error_hint(e: Exception) -> str:
    msg = str(e)
    if "SEC_USER_AGENT" in msg or "User-Agent" in msg:
        return "SEC requires a User-Agent. Set SEC_USER_AGENT in .env (e.g. 'Name email')."
    if "429" in msg or "rate" in msg.lower():
        return "SEC rate-limited the request. Wait a few seconds and try again."
    return msg


def _header_card(company: str, form_type: str, meta: dict) -> None:
    c = COLORS
    st.markdown(
        f'<div class="fs-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<div style="display:flex;gap:12px;align-items:center;">'
        f'<img src="{logo_url(company)}" style="width:44px;height:44px;border-radius:10px;'
        f'background:#fff;padding:5px;object-fit:contain;">'
        f'<div><div style="font-size:1.25rem;font-weight:800;">{company_name(company)}</div>'
        f'<div class="fs-mono" style="color:{c["muted"]};font-size:0.8rem;">{company} · {meta["form"]}</div></div></div>'
        f'<a href="{meta["url"]}" target="_blank" class="fs-chip" style="background:{c["sec"]}1a;'
        f'color:{c["sec"]};border-color:{c["sec"]}55;text-decoration:none;">View on SEC EDGAR ↗</a></div>'
        f'<div style="display:flex;gap:28px;margin-top:14px;">'
        f'<div><div class="fs-label">Filed</div><div class="fs-mono" style="font-size:1.05rem;">{meta["filing_date"]}</div></div>'
        f'<div><div class="fs-label">Report period</div><div class="fs-mono" style="font-size:1.05rem;">{meta["report_date"]}</div></div>'
        f'<div><div class="fs-label">Accession</div><div class="fs-mono" style="font-size:1.05rem;">{meta["accession"]}</div></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _section_navigator(sections: list) -> None:
    c = COLORS
    st.markdown('<div class="fs-label" style="margin-top:14px;">Filing sections</div>', unsafe_allow_html=True)
    for i, s in enumerate(sections, 1):
        n = len(s["text"])
        stub = n < 2000
        warn = (f'<span class="fs-chip" style="background:{c["laya"]}1a;color:{c["laya"]};'
                f'border-color:{c["laya"]}55;">⚠ stub</span>') if stub else ""
        with st.expander(f"[F{i}] {s['name']} · {n:,} chars", expanded=(i == 1 and not stub)):
            if warn:
                st.markdown(warn + " &nbsp;this section parsed short — likely a table-of-contents stub.",
                            unsafe_allow_html=True)
            st.markdown(
                f'<div style="max-height:320px;overflow:auto;background:{c["bg"]};border:1px solid {c["border"]};'
                f'border-radius:8px;padding:12px;color:{c["text_dim"]};font-size:0.86rem;line-height:1.5;'
                f'white-space:pre-wrap;">{_html.escape(s["text"][:4000])}'
                + ("…" if n > 4000 else "") + '</div>',
                unsafe_allow_html=True,
            )


def _filing_system(company: str, meta: dict, sections: list) -> str:
    context = "\n\n".join(f"[F{i}] {s['name']}:\n{s['text'][:3000]}"
                          for i, s in enumerate(sections, 1))
    return (
        "You are FinSight, a financial research assistant. Answer using ONLY the SEC filing "
        "excerpts below. Put a citation like [F1] right after each sentence that uses a source. "
        "If the answer is not in the excerpts, say so plainly. Never give financial advice or "
        f"invent numbers.\n\nCompany: {company} — {meta['form']}, filed {meta['filing_date']}.\n\n"
        f"FILING EXCERPTS:\n{context}")


def _ask(company: str, meta: dict, sections: list, question: str) -> None:
    c = COLORS
    system = _filing_system(company, meta, sections)
    draft = st.empty()
    try:
        from backend.app.llm import get_llm
        with draft.container():
            st.markdown(f'<div class="fs-label" style="color:{c["mistral"]};">Drafting on the GPU…</div>',
                        unsafe_allow_html=True)
            answer = st.write_stream(
                get_llm().stream(system, [{"role": "user", "content": question}],
                                 max_tokens=512, temperature=0.2))
    except Exception as e:  # noqa: BLE001
        draft.empty()
        emsg = str(e)
        hint = ("The GPU endpoint is unreachable (Mistral tunnel on :8000 is down). "
                "Start the tunnel, or this answer can't be generated.") if "Connection" in emsg else emsg
        st.error(f"Couldn't generate the answer: {hint}")
        return
    draft.empty()  # replace the live draft with the cited final (rendered in history)
    st.session_state.filing_chat.append({"q": question, "a": answer})


def render_filing_qa(company: str, form_type: str) -> None:
    c = COLORS
    st.session_state.setdefault("filing_chat", [])

    st.markdown(f'<div style="font-size:1.5rem;font-weight:800;">{company_name(company)} · filings</div>'
                f'<div style="color:{c["muted"]};font-size:0.9rem;margin-bottom:10px;">'
                'Fetch the latest filing from SEC EDGAR, browse its sections, and ask — answers cite '
                '[F] sections.</div>', unsafe_allow_html=True)

    if st.button(f"Fetch latest {form_type}", use_container_width=True, key="fetch_filing"):
        try:
            meta, sections = _fetch(company, form_type)
            st.session_state.filing_meta = meta
            st.session_state.filing_sections = sections
            st.session_state.filing_company = company
            st.session_state.filing_chat = []
            st.session_state.filing_keypoints = None
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not fetch filing: {_error_hint(e)}")

    meta = st.session_state.get("filing_meta")
    sections = st.session_state.get("filing_sections")
    if not meta or st.session_state.get("filing_company") != company:
        st.markdown(
            f'<div style="border:1px dashed {c["border"]};border-radius:12px;padding:26px;'
            f'text-align:center;color:{c["muted"]};margin-top:12px;">'
            f'Fetch {company}\'s latest {form_type} to begin.</div>', unsafe_allow_html=True)
        return

    left, right = st.columns([3, 2])
    with left:
        _header_card(company, form_type, meta)
        if st.button("✦ Summarize key points (Mistral)", key="keypts"):
            try:
                st.session_state.filing_keypoints = _key_points(
                    meta["accession"], company, _filing_system(company, meta, sections))
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't summarize: {_error_hint(e)}")
        kp = st.session_state.get("filing_keypoints")
        if kp:
            st.markdown(
                f'<div class="fs-card" style="border-left:3px solid {c["mistral"]};margin-top:8px;">'
                f'<div class="fs-label" style="color:{c["mistral"]};">Key points · Mistral · cited</div>'
                f'<div style="margin-top:6px;line-height:1.55;">{_inline(kp)}</div></div>',
                unsafe_allow_html=True)
        _section_navigator(sections)
    with right:
        st.markdown('<div style="font-size:1.3rem;font-weight:800;">Ask Mistral</div>', unsafe_allow_html=True)
        st.markdown('<div class="fs-label">Suggested</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, sug in enumerate(SUGGESTED):
            if cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                _ask(company, meta, sections, sug)
        q = st.text_input("Your question", key="filing_q",
                          placeholder="Ask about this filing…", label_visibility="collapsed")
        if st.button("Ask", type="primary", use_container_width=True, key="ask_filing"):
            if q.strip():
                _ask(company, meta, sections, q)
            else:
                st.warning("Type a question first.")

        for turn in reversed(st.session_state.filing_chat):
            st.markdown(
                f'<div style="margin-top:10px;color:{c["text_dim"]};font-size:0.9rem;">'
                f'<b style="color:{c["text"]};">Q:</b> {_html.escape(turn["q"])}</div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="fs-card" style="border-left:3px solid {c["mistral"]};margin-top:4px;">'
                f'<div class="fs-label" style="color:{c["mistral"]};">Mistral-7B · PSC GPU</div>'
                f'<div style="line-height:1.55;margin-top:4px;">{_inline(turn["a"])}</div></div>',
                unsafe_allow_html=True)
        st.caption("Answers grounded in the fetched filing · not financial advice.")
