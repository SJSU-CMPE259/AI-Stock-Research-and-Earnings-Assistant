# FinSight MCP server

Exposes FinSight's data tools and the agent pipeline to any MCP client (Cursor for the
demo) over stdio. Five read-only tools: four direct data tools and `ask_finsight`, which
hands the question to the agent_core Orchestrator. Code routes the question; the LLM only
writes the answer.

> The logic stays in `agent_core/` and `backend/app/tools/`. `tools.py` validates
> arguments and shapes outputs (plain Python, no MCP import, usable from a notebook);
> `server.py` only registers them. Facts only, no investment advice.

## Tools

| Tool | Arguments | Returns | Cite as |
|------|-----------|---------|---------|
| `get_quote` | `ticker` | live Finnhub price, change, high/low, open, previous close, `live: true` (cached 60 s). Errors instead of returning sample data | `[Q1]` |
| `get_company_news` | `ticker`, `days=7` (1–365), `limit=5` (1–20) | articles naming the company, newest first, summaries trimmed to ~300 chars, labeled `untrusted_external_text` | `[H1]`… |
| `list_filings` | `ticker`, `form="10-K"`, `limit=5` (1–20) | SEC EDGAR filings: dates, accession, URL | — |
| `get_filing_section` | `ticker`, `form` (10-K/10-Q), `item`, `max_chars=8000` (1–50000), `offset=0` | one Item of the latest filing, paged: call again with `offset=next_offset` until it is null | `[S1]` |
| `ask_finsight` | `question` (≤ 2000 chars), `new_conversation=false` | answer with the agents' citations, `route`, `steps`, `reused`, `quote_live`, `data_mode` (`live`/`sample`/`mixed`), `warnings` | as returned |

`item` accepts `"Item 1A"`, `"1a"`, `"risk factors"`, `"Item 7"`, `"md&a"` and so on
(form-aware: MD&A is Item 7 in a 10-K, Item 2 in a 10-Q). Supported: 10-K Items 1, 1A,
7, 7A; 10-Q Items 1A, 2.

`ask_finsight` keeps one conversation per server process, so "and its revenue?" follows
the last ticker. `new_conversation=true` starts over. Asking about a different ticker
clears the stored data first, so one company's quote or filing data is never cited for
another.

## Setup

From `ACSESS/finsight/`. The server has its own venv because `mcp` 1.x needs
`pydantic>=2.11` and the team's `requirements.txt` pins 2.6.1.

```bash
python3.12 -m venv .venv-mcp
.venv-mcp/bin/pip install -r requirements-mcp.txt
cp .env.example .env    # then fill in SEC_USER_AGENT, FINNHUB_API_KEY, LAYA_URL
```

Keys live only in `.env` (gitignored). The server loads it itself, from any working
directory, and never overrides variables already set in the environment.

## Modes

| `FINSIGHT_FAKE` | LLM | Routing | Needs |
|---|---|---|---|
| `1` | FakeLLM (canned, offline) | keyword rules | nothing; used by the tests and CI |
| `0` (default) | Mistral via `REMOTE_LLM_URL` | Laya via `LAYA_URL`, keyword fallback if it is down | tunnels on :8000 and :8001 |

The direct data tools call Finnhub and SEC in both modes when keys are set.

At startup the server logs one line to stderr: the mode, whether `FINNHUB_API_KEY` and
`SEC_USER_AGENT` are set (true/false, never the values) and whether the LLM and Laya
ports accept connections.

## Run

```bash
.venv-mcp/bin/python mcp_server/server.py                 # stdio; normally the client starts it
npx -y @modelcontextprotocol/inspector@2.13.0 -e FINSIGHT_FAKE=1 .venv-mcp/bin/python mcp_server/server.py
```

Use the Inspector command (needs Node >= 22.19) rather than `mcp dev`, which runs the server
under an unpinned `uv run --with mcp`. Pass mode flags with `-e` so they reach the server.
Drop `-e FINSIGHT_FAKE=1` for real mode.

## Cursor

Merge the `finsight` entry from [`cursor_mcp.example.json`](cursor_mcp.example.json)
into `mcpServers` in `~/.cursor/mcp.json`, replacing `/ABSOLUTE/PATH/TO` with your
checkout path. Keep the servers already there, and don't put keys in that file. An
`env` value there overrides `.env` (e.g. `"FINSIGHT_FAKE": "1"` for offline mode).
Then check that `finsight` shows as enabled with 5 tools in Cursor's MCP settings, and
ask in a chat in Agent mode. Server stderr is in the Output panel under the MCP logs.

## Tests

```bash
.venv-mcp/bin/python -m pytest mcp_server/tests agent_core/tests -m "not network" -q
```

Offline: Finnhub, SEC and the LLM are stubbed, and `conftest.py` sets `FINSIGHT_FAKE=1`
and a dummy `SEC_USER_AGENT`. Layers: `tools.py` functions, an in-memory MCP client
(`test_server_protocol.py`), and a real stdio subprocess started from `cwd=/`
(`test_server_stdio.py`, which also checks that a key never reaches stderr). CI runs the
same command (`.github/workflows/mcp-server.yml`).

## Known limits

- **Sample data.** Revenue, sentiment and filing-passage answers from `ask_finsight`
  come from `agent_core/data/sample_amd.json` (AMD) for every ticker; `data_mode` and
  `warnings` say so. Only quotes and news are live. For real filing text, use
  `get_filing_section`.
- **Tickers.** `ask_finsight` recognizes AMD, NVDA, AAPL, MSFT, TSLA and INTC; anything
  else falls back to AMD with a warning. The direct tools take any US ticker.
- **Keyword routing** (fake mode, or when Laya is down) matches substrings, so "share
  buybacks", "largest shareholders" and "selling, general and administrative" are
  refused as advice. Kept visible as strict `xfail` tests in `test_tools_ask.py`; remove
  the marker when the router is fixed.
- **Citation IDs.** `[S1]` means a filing section here but sentiment in `report_agent`.
  To be agreed with the team.
- **Filing freshness.** The backend SEC client caches responses on disk with no expiry,
  so "latest filing" is whatever was fetched first until `data/` is cleared.
