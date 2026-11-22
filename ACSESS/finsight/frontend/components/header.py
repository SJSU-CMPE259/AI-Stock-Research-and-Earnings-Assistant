"""Top app bar: logo + wordmark + tagline (the company picker sits beside it)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from frontend.theme import COLORS

_LOGO = (Path(__file__).resolve().parents[1] / "assets" / "logo.svg").read_text()


def render_header() -> None:
    c = COLORS
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;padding:6px 2px;">'
        f'<div style="width:40px;height:40px;">{_LOGO}</div>'
        f'<div><div style="font-size:1.35rem;font-weight:800;letter-spacing:-0.02em;line-height:1;">'
        f'Fin<span style="color:{c["laya"]};">Sight</span></div>'
        f'<div style="font-size:0.78rem;color:{c["muted"]};">Your AI Financial Researcher</div></div></div>',
        unsafe_allow_html=True)
