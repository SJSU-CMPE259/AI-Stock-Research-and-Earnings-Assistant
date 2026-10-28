"""Agent pipeline view — the Agent Demo centerpiece.

Shows the full fixed pipeline (Router → Data → Analysis → Classify → Market → Viz →
Report). Agents not used by the chosen route are dimmed "skipped" — making visible
that only the needed agents run. Each node carries its engine badge, status, and
duration, read from Trace.events. A route card above shows Laya's decision +
confidence (or the keyword fallback).
"""
from __future__ import annotations

import streamlit as st

from frontend.components.icons import icon
from frontend.theme import COLORS, ENGINE

# fixed display order + which engine powers each agent
PIPELINE = [
    ("router", "Router"), ("data_agent", "Data"), ("analysis_agent", "Analysis"),
    ("classify_agent", "Classify"), ("market_agent", "Market"), ("viz_agent", "Viz"),
    ("report_agent", "Report"),
]
ICON_KEY = {"router": "router", "data_agent": "data", "analysis_agent": "analysis",
            "classify_agent": "classify", "market_agent": "market", "viz_agent": "viz",
            "report_agent": "report"}
AGENT_ENGINE = {"data_agent": "python", "analysis_agent": "python", "viz_agent": "python",
                "classify_agent": "mistral", "report_agent": "mistral", "market_agent": "finnhub"}


def render_route_card(trace) -> None:
    c = COLORS
    laya = trace.route_source == "laya"
    eng_label, eng_color = ENGINE["laya"] if laya else ENGINE["keyword"]
    conf = trace.route_confidence
    conf_pct = int(conf * 100) if isinstance(conf, (int, float)) else None
    bar = ""
    if conf_pct is not None:
        bar = (f'<div style="height:6px;background:{c["surface2"]};border-radius:3px;margin-top:6px;'
               f'overflow:hidden;max-width:220px;"><div style="width:{conf_pct}%;height:100%;'
               f'background:{eng_color};"></div></div>')
    fallback = "" if laya else (f'<span style="color:{c["laya"]};font-size:0.78rem;margin-left:8px;">'
                                '⚠ Laya unavailable — keyword rules used</span>')
    st.markdown(
        f'<div class="fs-card" style="margin-bottom:12px;">'
        f'<span class="fs-label">Routing</span>{fallback}'
        f'<div style="display:flex;align-items:baseline;gap:10px;margin-top:4px;">'
        f'<span style="font-size:1.15rem;font-weight:800;color:{eng_color};">{eng_label}</span>'
        f'<span style="color:{c["muted"]};">→</span>'
        f'<span class="fs-mono" style="font-size:1.15rem;font-weight:700;">{trace.route}</span>'
        + (f'<span class="fs-mono" style="color:{c["text_dim"]};">· confidence '
           f'{conf:.2f}</span>' if conf_pct is not None else "")
        + f'</div>{bar}</div>',
        unsafe_allow_html=True,
    )


def _node(key: str, label: str, ev: dict | None, route_source: str) -> str:
    c = COLORS
    engine = route_source if key == "router" else AGENT_ENGINE.get(key, "python")
    eng_label, eng_color = ENGINE.get(engine, ("", c["muted"]))

    status = "skipped" if ev is None else ev["status"]
    if status == "skipped":
        op, ring, foot = "0.34", c["border"], "skipped"
    elif status == "reused":
        op, ring, foot = "0.9", c["muted"], "reused ♻"
    elif status == "error":
        op, ring, foot = "1", c["error"], "error"
    else:  # done
        op, ring, foot = "1", eng_color, f'{ev.get("ms", 0):.0f} ms'

    ic = icon(ICON_KEY.get(key, "data"), size=18, color=(ring if status != "skipped" else c["muted"]))
    badge = (f'<span style="font-size:0.6rem;font-weight:700;color:{eng_color};">{eng_label}</span>'
             if status != "skipped" else f'<span style="font-size:0.6rem;color:{c["muted"]};">{eng_label}</span>')
    return (
        f'<div style="min-width:86px;text-align:center;opacity:{op};">'
        f'<div style="width:40px;height:40px;margin:0 auto;border-radius:11px;border:1.5px solid {ring};'
        f'background:{c["surface"]};display:flex;align-items:center;justify-content:center;">{ic}</div>'
        f'<div style="font-size:0.78rem;font-weight:700;margin-top:5px;">{label}</div>'
        f'<div>{badge}</div>'
        f'<div style="font-size:0.64rem;color:{c["muted"]};" class="fs-mono">{foot}</div></div>'
    )


def render_pipeline(trace) -> None:
    c = COLORS
    by_agent = {e["agent"]: e for e in trace.events}
    arrow = f'<span style="color:{c["border"]};align-self:flex-start;margin-top:14px;">→</span>'
    nodes = [_node(k, lbl, by_agent.get(k), trace.route_source) for k, lbl in PIPELINE]
    st.markdown(
        f'<div style="display:flex;gap:6px;align-items:flex-start;overflow-x:auto;padding:14px;'
        f'background:{c["bg"]};border:1px solid {c["border"]};border-radius:14px;">'
        + arrow.join(nodes) + '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Dimmed agents are skipped — the router runs only the agents this question needs.")
