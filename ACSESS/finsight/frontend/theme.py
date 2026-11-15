"""FinSight design system — "Terminal" direction.

Deep-navy financial-terminal surfaces, Inter for UI, IBM Plex Mono (tabular figures)
for every number/ticker. One place for color tokens, fonts, and Streamlit chrome hiding.
Import COLORS/ENGINE for charts and components; call apply_theme() once at startup.
"""
from __future__ import annotations

import streamlit as st

# ---- color tokens (semantic, WCAG-AA on the navy surfaces) ----
DARK = {
    "bg": "#0B1420",          # app background (deepest)
    "surface": "#111C2B",     # cards
    "surface2": "#16273B",    # raised rows / hover
    "input": "#16273B",       # text boxes
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

# same tokens on white: engine colors darkened so text on white stays readable
LIGHT = {
    "bg": "#E8EDF2", "surface": "#FFFFFF", "surface2": "#EDF2F7", "input": "#FFFFFF", "border": "#C3CFDB",
    "text": "#13202F", "text_dim": "#3D4F63", "muted": "#5E6E7F",
    "laya": "#A8620A", "mistral": "#2563A8", "python": "#0B7A5E", "finnhub": "#157385",
    "sec": "#5B4BB0", "error": "#C0392B", "up": "#0F7A50", "down": "#C0392B",
}

# Components import these two dicts, so set_mode() updates them in place.
COLORS = dict(DARK)
ENGINE: dict = {}


def set_mode(light: bool) -> None:
    """Switch the palette for this run; call before apply_theme()."""
    COLORS.update(LIGHT if light else DARK)
    # engine → (label, color) for badges
    ENGINE.update({
        "laya": ("Laya", COLORS["laya"]),
        "mistral": ("Mistral", COLORS["mistral"]),
        "python": ("Python", COLORS["python"]),
        "finnhub": ("Finnhub", COLORS["finnhub"]),
        "sec": ("SEC EDGAR", COLORS["sec"]),
        "keyword": ("keyword", COLORS["muted"]),
    })


set_mode(False)

FONT_SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', 'SFMono-Regular', Menlo, monospace"


def apply_theme() -> None:
    """Inject fonts, the navy terminal theme, and hide Streamlit's own chrome."""
    c = COLORS
    css = f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Caveat:wght@600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
        /* hide Streamlit chrome: menu, Deploy button, header, footer, decoration */
        #MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
        [data-testid="stMainMenu"], [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {{ display: none !important; }}
        /* keep the header itself: it holds the button that reopens the collapsed sidebar */
        header[data-testid="stHeader"] {{ background: transparent; }}

        .stApp {{ background: radial-gradient(1100px 420px at 30% -8%, {c['surface2']} 0%, {c['bg']} 62%);
            color: {c['text']}; font-family: {FONT_SANS}; }}
        section[data-testid="stSidebar"] {{ background: {c['surface']}; border-right: 1px solid {c['border']}; }}
        .block-container {{ padding-top: 1.1rem; max-width: 1360px; }}

        h1, h2, h3, h4 {{ font-family: {FONT_SANS}; color: {c['text']}; letter-spacing: -0.01em; }}

        /* numbers/tickers in tabular mono wherever we tag them */
        .fs-mono, .fs-num {{ font-family: {FONT_MONO}; font-feature-settings: "tnum" 1; }}

        /* tabs → terminal segmented look */
        .stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {c['border']}; }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent; color: {c['text_dim']}; font-weight: 600; font-size: 1rem;
            padding: 10px 20px; border-radius: 10px 10px 0 0;
        }}
        .stTabs [aria-selected="true"] {{ background: {c['surface']}; color: {c['laya']};
            border-bottom: 3px solid {c['laya']}; }}

        /* inputs */
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div {{
            background: {c['input']} !important; color: {c['text']} !important;
            border-color: {c['border']} !important;
        }}
        [data-testid="stTextInput"] input {{ font-size: 1.02rem; padding: 12px 14px; }}
        [data-testid="stTextInput"] input::placeholder, textarea::placeholder {{
            color: {c['muted']} !important; opacity: 1; }}
        [data-testid="stTextInput"] > div {{ border-radius: 12px; }}
        [data-testid="stTextInput"] > div:focus-within {{ border-color: {c['laya']} !important;
            box-shadow: 0 0 0 3px {c['laya']}33; }}
        /* buttons */
        .stButton > button {{
            background: {c['surface2']}; color: {c['text']}; border: 1px solid {c['border']};
            font-weight: 600; border-radius: 12px; min-height: 46px; transition: all .15s ease;
        }}
        .stButton > button:hover {{ border-color: {c['laya']}; color: {c['text']}; transform: translateY(-1px); }}
        .stButton > button[kind="primary"] {{ background: linear-gradient(135deg, #E8A63A, #F2C46B);
            border: none; color: #13202F; font-weight: 800; box-shadow: 0 6px 18px #E8A63A40; }}
        /* example-question pills; short action buttons never break mid-word */
        .st-key-agent_chips button {{ min-height: 0; color: {c['text_dim']}; background: {c['surface']};
            border: 1px solid {c['border']}; }}
        .st-key-agent_chips button:hover {{ border-color: {c['laya']}; color: {c['text']}; }}
        .st-key-ask_btn p, .st-key-new_chat p {{ white-space: nowrap; }}

        /* visible keyboard focus (a11y) */
        *:focus-visible {{ outline: 2px solid {c['laya']} !important; outline-offset: 2px; }}

        /* reusable chip + card classes */
        .fs-chip {{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px;
            border-radius:999px; font-size:0.74rem; font-weight:700; border:1px solid; }}
        .fs-card {{ background:{c['surface']}; border:1px solid {c['border']}; box-shadow: 0 1px 3px #10202F14;
            border-radius:14px; padding:18px 20px; }}
        /* Chat landing (Anna's mockup): handwritten greeting, big centered question */
        .fs-hello {{ font: 600 1.9rem/1.1 Caveat, "Segoe Print", cursive; color: {c['laya']}; }}
        .fs-hero {{ text-align: center; font-size: clamp(1.6rem, 3.2vw, 2.3rem); font-weight: 800;
            letter-spacing: -0.02em; line-height: 1.2; margin: 6px 0 4px; }}
        .fs-hero span {{ color: {c['laya']}; }}
        .fs-sub {{ text-align: center; color: {c['muted']}; font-size: 0.95rem; margin: 0 0 16px; }}
        .fs-label {{ font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase;
            color:{c['muted']}; font-weight:700; }}
        a {{ color: {c['finnhub']}; }}
        /* disabled textareas (filing section previews) must stay readable */
        [data-testid="stTextArea"] textarea:disabled {{
            color: {c['text_dim']} !important; -webkit-text-fill-color: {c['text_dim']} !important;
            opacity: 1 !important; font-family: {FONT_MONO}; font-size: 0.82rem; }}
        /* expander header + body readable */
        [data-testid="stExpander"] summary {{ color: {c['text']} !important; font-weight: 600;
            background: {c['surface']} !important; }}
        [data-testid="stExpander"] summary:hover {{ background: {c['surface2']} !important; }}
        /* widget labels: a light-mode browser gives them Streamlit's faint gray */
        [data-testid="stWidgetLabel"] p {{ color: {c['text_dim']} !important; }}
        [data-testid="stCode"] pre {{ background: {c['surface2']} !important; color: {c['text']} !important; }}
        [data-testid="stExpander"] {{ border-color: {c['border']} !important; }}
        details {{ background: {c['surface']}; border-radius: 10px; }}
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
