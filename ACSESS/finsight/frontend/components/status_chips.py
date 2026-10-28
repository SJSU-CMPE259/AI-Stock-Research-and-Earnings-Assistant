"""Backend status chips: Mistral / Laya / Finnhub / SEC EDGAR — live or fallback.

All checks are cheap and non-fatal: a down backend shows a muted/amber chip, never
an error. Used in the top bar so the demo state is always honest and visible.
"""
from __future__ import annotations

import os
import socket

from frontend.theme import COLORS, chip


def _port_open(port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=timeout):
            return True
    except OSError:
        return False


def backend_status(use_fake: bool, use_laya: bool) -> dict:
    """Return {name: (label, color)} for each backend given the current toggles."""
    c = COLORS
    if use_fake:
        mistral = ("Mistral offline", c["muted"])
    else:
        mistral = ("Mistral live", c["python"]) if _port_open(8000) else ("Mistral down", c["error"])
    if use_laya:
        laya = ("Laya live", c["python"]) if _port_open(8001) else ("Laya down", c["error"])
    else:
        laya = ("Router: keyword", c["muted"])
    finnhub = ("Finnhub live", c["python"]) if os.getenv("FINNHUB_API_KEY") else ("Finnhub: sample", c["laya"])
    sec = ("SEC EDGAR", c["sec"])
    return {"mistral": mistral, "laya": laya, "finnhub": finnhub, "sec": sec}


def chips_html(use_fake: bool, use_laya: bool) -> str:
    """Row of status chips as one HTML string."""
    st = backend_status(use_fake, use_laya)
    return "".join(chip(lbl, col) for lbl, col in st.values())
