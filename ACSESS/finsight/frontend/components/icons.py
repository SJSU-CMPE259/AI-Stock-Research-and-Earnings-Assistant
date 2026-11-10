"""Tiny original SVG icon set (24x24 stroke glyphs) — replaces emoji in the UI.

Clean geometric line icons drawn in-house (no external set, no attribution needed).
`icon(name, size, color)` returns an inline <svg> string.
"""
from __future__ import annotations

_PATHS = {
    "router": '<path d="M4 7h11M4 7l3-3M4 7l3 3"/><path d="M20 17H9M20 17l-3-3M20 17l-3 3"/>',
    "data": '<ellipse cx="12" cy="6" rx="7" ry="2.6"/><path d="M5 6v6c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6V6"/><path d="M5 12v6c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6v-6"/>',
    "analysis": '<path d="M5 19V9"/><path d="M10 19V5"/><path d="M15 19v-7"/><path d="M20 19v-4"/>',
    "classify": '<path d="M4 6h10l6 6-6 6H4z"/><circle cx="8.5" cy="12" r="1.3"/>',
    "market": '<path d="M3 17l5-5 4 3 7-8"/><path d="M15 7h4v4"/>',
    "viz": '<path d="M4 20V4"/><path d="M4 20h16"/><rect x="7" y="12" width="3" height="5"/><rect x="12" y="8" width="3" height="9"/><rect x="17" y="5" width="3" height="12"/>',
    "report": '<path d="M7 3h7l4 4v14H7z"/><path d="M14 3v4h4"/><path d="M10 12h5M10 15h5"/>',
    "check": '<path d="M5 12l4 4 10-10"/>',
    "alert": '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17v.5"/>',
    "lens": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.5-4.5"/>',
    "copy": '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/>',
}


def icon(name: str, size: int = 16, color: str = "currentColor", stroke: float = 1.8) -> str:
    body = _PATHS.get(name, "")
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="vertical-align:middle;">{body}</svg>')
