"""Backend status chips: Mistral / Laya / Finnhub / SEC EDGAR — live or fallback.

All checks are cheap and non-fatal: a down backend shows a muted/amber chip, never
an error. Used in the top bar so the demo state is always honest and visible.
"""
from __future__ import annotations

import os
import socket
import urllib.request

from frontend.theme import COLORS, chip


def _endpoint_live(port: int, path: str, timeout: float = 0.8) -> bool:
    """True only if the server actually RESPONDS (not just the tunnel socket being open).

    A dead PSC node still accepts the tunnel's TCP connection but resets requests, so a
    bare socket check reports a false 'live'. We do a quick HTTP GET and require a reply.
    """
    try:
        with socket.create_connection(("localhost", port), timeout=0.3):
            pass
    except OSError:
        return False
    try:
        with urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001 — reset / timeout / error => not live
        return False


def backend_status(use_fake: bool, use_laya: bool) -> dict:
    """Return {name: (label, color)} for each backend given the current toggles."""
    c = COLORS
    if use_fake:
        mistral = ("Mistral offline", c["muted"])
    else:
        mistral = ("Mistral live", c["python"]) if _endpoint_live(8000, "/v1/models") else ("Mistral down", c["error"])
    if use_laya:
        laya = ("Laya live", c["python"]) if _endpoint_live(8001, "/health") else ("Laya down", c["error"])
    else:
        laya = ("Router: keyword", c["muted"])
    finnhub = ("Finnhub live", c["python"]) if os.getenv("FINNHUB_API_KEY") else ("Finnhub: sample", c["laya"])
    sec = ("SEC EDGAR", c["sec"])
    return {"mistral": mistral, "laya": laya, "finnhub": finnhub, "sec": sec}


def chips_html(use_fake: bool, use_laya: bool) -> str:
    """Row of status chips as one HTML string."""
    st = backend_status(use_fake, use_laya)
    return "".join(chip(lbl, col) for lbl, col in st.values())
