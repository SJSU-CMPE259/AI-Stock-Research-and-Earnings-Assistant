"""FinSight design system — "Terminal" direction.

Deep-navy financial-terminal surfaces, Inter for UI, IBM Plex Mono (tabular figures)
for every number/ticker. One place for color tokens, fonts, and Streamlit chrome hiding.
Import COLORS/ENGINE for charts and components; call apply_theme() once at startup.
"""
from __future__ import annotations

import streamlit as st

# ---- color tokens (semantic, WCAG-AA on the navy surfaces) ----
COLORS = {
    "bg": "#0B1420",          # app background (deepest)
    "surface": "#111C2B",     # cards
    "surface2": "#16273B",    # raised rows / hover
    "border": "#22374F",
    "text": "#E6EDF3",        # primary text  (contrast ~13:1 on bg)
    "text_dim": "#A6B7C8",    # secondary     (~7:1)
    "muted": "#71869B",       # tertiary/labels (~4.6:1)
    # engines (slide-deck colors, lightened for contrast on dark)
    "laya": "#E8A63A",        # amber
    "mistral": "#5B97D6",     # steel blue
    "python": "#22B894",      # emerald (verified)
    "finnhub": "#4FB6C9",     # cyan (live data)
    "sec": "#9A8CDB",         # violet (filings)
    "error": "#E0655A",
    # market
    "up": "#2FB87F",
    "down": "#E0655A",
}

# engine → (label, color) for badges
ENGINE = {
    "laya": ("Laya", COLORS["laya"]),
    "mistral": ("Mistral", COLORS["mistral"]),
    "python": ("Python", COLORS["python"]),
    "finnhub": ("Finnhub", COLORS["finnhub"]),
    "sec": ("SEC EDGAR", COLORS["sec"]),
    "keyword": ("keyword", COLORS["muted"]),
}

FONT_SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', 'SFMono-Regular', Menlo, monospace"


def apply_theme() -> None:
    """Inject fonts, the navy terminal theme, and hide Streamlit's own chrome."""
    c = COLORS
    css = f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
        /* hide Streamlit chrome: menu, Deploy button, header, footer, decoration */
        #MainMenu, header[data-testid="stHeader"], footer,
        [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {{ display: none !important; }}

        .stApp {{ background: {c['bg']}; color: {c['text']}; font-family: {FONT_SANS}; }}
        section[data-testid="stSidebar"] {{ background: {c['surface']}; border-right: 1px solid {c['border']}; }}
        .block-container {{ padding-top: 1.1rem; max-width: 1360px; }}

        h1, h2, h3, h4 {{ font-family: {FONT_SANS}; color: {c['text']}; letter-spacing: -0.01em; }}

        /* numbers/tickers in tabular mono wherever we tag them */
        .fs-mono, .fs-num {{ font-family: {FONT_MONO}; font-feature-settings: "tnum" 1; }}

        /* tabs → terminal segmented look */
        .stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {c['border']}; }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent; color: {c['text_dim']}; font-weight: 600;
            padding: 8px 18px; border-radius: 8px 8px 0 0;
        }}
        .stTabs [aria-selected="true"] {{ background: {c['surface']}; color: {c['text']};
            border-bottom: 2px solid {c['laya']}; }}

        /* inputs */
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div {{
            background: {c['surface2']} !important; color: {c['text']} !important;
            border-color: {c['border']} !important;
        }}
        /* buttons */
        .stButton > button {{
            background: {c['surface2']}; color: {c['text']}; border: 1px solid {c['border']};
            font-weight: 600; border-radius: 8px;
        }}
        .stButton > button:hover {{ border-color: {c['laya']}; color: {c['text']}; }}
        .stButton > button[kind="primary"] {{ background: {c['mistral']}; border-color: {c['mistral']}; color: #fff; }}

        /* visible keyboard focus (a11y) */
        *:focus-visible {{ outline: 2px solid {c['laya']} !important; outline-offset: 2px; }}

        /* reusable chip + card classes */
        .fs-chip {{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px;
            border-radius:999px; font-size:0.74rem; font-weight:700; border:1px solid; }}
        .fs-card {{ background:{c['surface']}; border:1px solid {c['border']};
            border-radius:14px; padding:18px 20px; }}
        .fs-label {{ font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase;
            color:{c['muted']}; font-weight:700; }}
        a {{ color: {c['finnhub']}; }}
        /* captions readable on dark (a11y contrast) */
        [data-testid="stCaptionContainer"], .stCaption, small {{ color: {c['text_dim']} !important; }}
        [data-testid="stMetricLabel"] {{ color: {c['muted']}; }}
        [data-testid="stMetricValue"] {{ font-family: {FONT_MONO}; color: {c['text']}; }}
        </style>
    """
    # Collapse to non-blank, non-indented lines: blank lines end an HTML block in
    # Streamlit's markdown (dumping the rest of the CSS as visible text), and leading
    # indentation makes it a code block. One continuous <style> block avoids both.
    st.markdown(
        " ".join(line.strip() for line in css.splitlines() if line.strip()),
        unsafe_allow_html=True,
    )


def chip(label: str, color: str, *, filled: bool = False, dot: bool = True) -> str:
    """Return an inline status/engine chip (HTML string)."""
    d = f'<span style="width:7px;height:7px;border-radius:50%;background:{color};"></span>' if dot else ""
    bg = f"{color}22" if not filled else color
    fg = color if not filled else "#0B1420"
    return (f'<span class="fs-chip" style="background:{bg};color:{fg};border-color:{color}55;">'
            f'{d}{label}</span>')
