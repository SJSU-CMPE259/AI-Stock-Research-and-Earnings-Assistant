"""MCP demo panel: call the FinSight MCP server as a real MCP client (stdio / JSON-RPC).

This renders under Filing Q&A so you can see the Model Context Protocol round-trip —
handshake, tool discovery, and a live tool call — against the SAME agent pipeline:
`ask_finsight` routes with Laya and writes the answer with Mistral. The server runs in
its own `.venv-mcp` (mcp 1.x needs pydantic>=2.11), spawned over stdio.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]          # ACSESS/finsight
_MCP_PY = _ROOT / ".venv-mcp" / "bin" / "python"
_SERVER = _ROOT / "mcp_server" / "server.py"


async def _run(tool: str, args: dict) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=str(_MCP_PY), args=[str(_SERVER)], cwd=str(_ROOT),
        env={**os.environ, "FINSIGHT_FAKE": "0"},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        info = await session.initialize()
        tools = await session.list_tools()
        res = await session.call_tool(tool, args)
        text = res.content[0].text if res.content else ""
        return {
            "server": f"{info.serverInfo.name} v{info.serverInfo.version}",
            "tools": [t.name for t in tools.tools],
            "request": {"tool": tool, "arguments": args},
            "result": text,
        }


def _call_mcp(tool: str, args: dict) -> dict:
    """Sync wrapper for Streamlit (spawns the MCP server, runs one round-trip)."""
    return asyncio.run(_run(tool, args))


def render_mcp_panel(ticker: str) -> None:
    """Show a live MCP round-trip under the Filing Q&A tab."""
    if not _MCP_PY.exists() or not _SERVER.exists():
        return  # MCP server / its venv not set up on this machine

    st.markdown("---")
    st.markdown('<div class="fs-label">🔌 Model Context Protocol (live)</div>',
                unsafe_allow_html=True)
    st.caption(
        "The same FinSight pipeline, exposed as MCP tools for any MCP client (Cursor / "
        "Claude Desktop). This page acts as the MCP **client**: it spawns the server over "
        "stdio and makes a real JSON-RPC call. `ask_finsight` routes with **Laya** and "
        "answers with **Mistral** — the identical agents, reached through MCP."
    )
    q = st.text_input("Ask via MCP (ask_finsight)",
                      value=f"Tell me about {ticker}'s business and products",
                      key="mcp_q")
    if st.button("Call tool via MCP →", key="mcp_call"):
        with st.spinner("MCP round-trip: spawn server → initialize → list tools → call tool…"):
            try:
                out = _call_mcp("ask_finsight", {"question": q})
            except Exception as e:  # noqa: BLE001 - surface any client/server error to the UI
                st.error(f"MCP call failed: {e}")
                return
        st.success(f"✅ Connected over MCP: {out['server']}")
        st.caption("Tools discovered: " + ", ".join(f"`{t}`" for t in out["tools"]))
        try:
            data = json.loads(out["result"])
        except (json.JSONDecodeError, TypeError):
            data = {"raw": out["result"]}
        if isinstance(data, dict) and data.get("answer"):
            st.markdown(f"**Answer (MCP → Laya route → Mistral):**\n\n{data['answer']}")
            meta = {k: data[k] for k in ("route", "steps", "data_mode", "quote_live", "warnings")
                    if k in data}
            st.caption("route & provenance: " + json.dumps(meta))
        with st.expander("Raw MCP exchange (request + JSON-RPC response)"):
            st.code("request  → " + json.dumps(out["request"], indent=2), language="json")
            st.code("response ← " + json.dumps(data, indent=2)[:4000], language="json")
