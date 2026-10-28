"""Top app bar: logo + wordmark + tagline, status chips on the right."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from frontend.components.status_chips import chips_html
from frontend.theme import COLORS

_LOGO = (Path(__file__).resolve().parents[1] / "assets" / "logo.svg").read_text()


def render_header(use_fake: bool, use_laya: bool) -> None:
    c = COLORS
    html = (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'padding:6px 2px 14px;border-bottom:1px solid {c["border"]};margin-bottom:16px;">'
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<div style="width:40px;height:40px;">{_LOGO}</div>'
        f'<div><div style="font-size:1.35rem;font-weight:800;letter-spacing:-0.02em;line-height:1;">'
        f'Fin<span style="color:{c["laya"]};">Sight</span></div>'
        f'<div style="font-size:0.72rem;color:{c["muted"]};letter-spacing:0.04em;">'
        f'SEC filings · live market data · cited answers</div></div></div>'
        f'<div style="display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end;">'
        f'{chips_html(use_fake, use_laya)}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)
