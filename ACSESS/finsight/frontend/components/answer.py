"""Answer panel: inline citation chips + Sources + a verification badge.

The badge reflects report_agent's validator: "Verified" when Mistral's answer passed
(every cited id exists and every number appears in the notes), or "Template answer"
when it fell back. We never bypass the validator — we only read its outcome.
"""
from __future__ import annotations

import html as _html
import re

import streamlit as st

from frontend.components.icons import icon
from frontend.theme import COLORS

_CITE = re.compile(r"\[([A-Z]\d+(?:\s*,\s*[A-Z]\d+)*)\]")
_ID = re.compile(r"[A-Z]\d+")

# citation id prefix → (source kind, color). Convention: N notes, P passages, Q quotes,
# H headlines, S sentiment, F filing sections.
_KIND = {"N": ("note", COLORS["python"]), "P": ("passage", COLORS["python"]),
         "F": ("filing", COLORS["sec"]), "S": ("sentiment", COLORS["mistral"]),
         "Q": ("quote", COLORS["finnhub"]), "H": ("news", COLORS["finnhub"])}


def _chip(cid: str) -> str:
    col = _KIND.get(cid[0], ("", COLORS["muted"]))[1]
    return (f'<span class="fs-mono" style="background:{col}1f;color:{col};border:1px solid {col}55;'
            f'border-radius:5px;padding:0 5px;font-size:0.76rem;font-weight:600;">{cid}</span>')


def _inline(answer: str) -> str:
    """Escape the answer, then turn [N1, N2] into individual colored chips."""
    out, last = [], 0
    for m in _CITE.finditer(answer):
        out.append(_html.escape(answer[last:m.start()]))
        out.append(" ".join(_chip(i) for i in _ID.findall(m.group(1))))
        last = m.end()
    out.append(_html.escape(answer[last:]))
    return "".join(out).replace("\n", "<br>")


def _sources(memory, cited_ids: list[str]) -> list[tuple[str, str]]:
    """Map cited ids to their note/passage/quote/news/filing text from shared memory."""
    out = {}
    if memory.has("amd_notes"):
        for n in memory.get("amd_notes"):
            out[n["id"]] = n["text"]
    if memory.has("amd_passages"):
        for i, p in enumerate(memory.get("amd_passages"), 1):
            out.setdefault(f"P{i}", p["text"])
    if memory.has("amd_quote"):
        q = memory.get("amd_quote")
        if isinstance(q, dict) and "price" in q:
            out["Q1"] = f"Current price {q['price']} {q.get('currency', 'USD')} as of {q.get('as_of', '')}"
    if memory.has("amd_news"):
        for i, a in enumerate(memory.get("amd_news"), 1):
            out[f"H{i}"] = f"{a.get('source', '')}, {a.get('date', '')}: {a.get('headline', '')}"
    if memory.has("amd_sentiment"):
        s = memory.get("amd_sentiment")
        if isinstance(s, dict) and s.get("summary"):
            out["S1"] = s["summary"]
    return [(cid, out[cid]) for cid in cited_ids if cid in out]


def render_answer(trace, memory) -> None:
    c = COLORS
    detail = next((e.get("detail", "") for e in trace.events if e["agent"] == "report_agent"), "")
    verified = "mistral" in detail
    if verified:
        badge = (f'<span class="fs-chip" style="background:{c["python"]}1f;color:{c["python"]};'
                 f'border-color:{c["python"]}66;">{icon("check", 13, c["python"])} '
                 f'Verified · every number matches the notes</span>')
    else:
        badge = (f'<span class="fs-chip" style="background:{c["laya"]}1f;color:{c["laya"]};'
                 f'border-color:{c["laya"]}66;">{icon("alert", 13, c["laya"])} '
                 f'Template answer · built from the notes (LLM output failed validation)</span>')

    cited = []
    for m in _CITE.finditer(trace.answer):
        cited += _ID.findall(m.group(1))
    cited = list(dict.fromkeys(cited))  # unique, in order

    st.markdown(
        f'<div class="fs-card" style="border-left:3px solid {c["mistral"]};">'
        f'<div style="margin-bottom:8px;">{badge}</div>'
        f'<div style="font-size:1rem;line-height:1.6;color:{c["text"]};">{_inline(trace.answer)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    srcs = _sources(memory, cited)
    if srcs:
        st.markdown('<div class="fs-label" style="margin-top:10px;">Sources</div>', unsafe_allow_html=True)
        for cid, text in srcs:
            st.markdown(
                f'<div style="display:flex;gap:8px;padding:5px 0;border-bottom:1px solid {c["border"]};">'
                f'{_chip(cid)}<span style="color:{c["text_dim"]};font-size:0.88rem;">'
                f'{_html.escape(text[:240])}</span></div>',
                unsafe_allow_html=True,
            )
    with st.expander("Copy answer"):
        st.code(trace.answer, language=None)
    st.caption("Research / educational use only · not financial advice.")
