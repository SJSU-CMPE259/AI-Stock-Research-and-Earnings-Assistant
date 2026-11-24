# FinSight — demo script

A 2–3 minute walkthrough that shows the whole system: Laya routing (with confidence),
Mistral writing cited answers, real SEC/XBRL/Finnhub data, the advice guardrail, graceful
handling of a brand-new listing, and the MCP integration.

## Before you start (checklist)

- **Jetstream instance up** (unshelved) and the **SSH tunnel open** from your laptop:
  ```bash
  ssh -N -o ServerAliveInterval=30 -L 8000:localhost:11434 -L 8001:localhost:8001 exouser@<INSTANCE_IP>
  ```
- **Verify** both models answer:
  ```bash
  curl -s localhost:8000/v1/models && curl -s localhost:8001/health
  ```
- **App running**: http://localhost:8501 (hard-refresh the tab). Top chips should read
  **Mistral live · Router: Laya · Finnhub live · SEC EDGAR**.
- Open the **"How FinSight answered this"** expander during the Ask tab so the audience sees
  `laya → <route> · confidence 0.xx` for each question.

## Suggested 2-minute flow

1. **Ask FinSight** — price → revenue → tone → "Should I buy?" (refusal).
   *"Laya routes each question to the right tool; numbers come from Python/XBRL, the LLM only writes the prose."*
2. **New listing** — Anthropic price (works) → Anthropic business ("no filings yet").
   *"It handles a day-old IPO gracefully instead of erroring."*
3. **Filing Q&A** — fetch NVDA's 10-K → risk factors.
   *"Grounded in the real filing, cited by section."*
4. **MCP panel** — the same business question, via MCP.
   *"The app is a real MCP client — it spawns our server over stdio, discovers the tools, and ask_finsight runs Laya + Mistral. Any MCP client (Cursor, Claude Desktop) gets the same tools."*

---

## 🗣️ Ask FinSight tab (agent pipeline: Laya + Mistral + citations)

Pick **NVDA** and run these in order to hit every route:

| Question | Route / what it shows |
|---|---|
| What is NVDA trading at right now? | `real_time` → live Finnhub price `[Q1]` |
| What was NVDA's revenue last quarter? | `numbers` → real XBRL, cited `[N1]`, QoQ/YoY chart |
| What is the tone of NVDA's recent filings? | `sentiment` → 10-K classified, counts + `[S1]`, bar chart |
| Tell me about NVDA's business and products | `text` → real 10-K passages `[P…]` |
| Should I buy NVDA? | **advice guardrail** → polite refusal, **0 LLM calls** |

Then show **multi-company + new-listing**:

- Select **AMD** → *What was AMD's revenue last quarter?* → real AMD numbers (~$13.38B).
- Select **Anthropic (ANT)** → *What is Anthropic trading at right now?* → live price + analyst news (what a new IPO does have).
- Still on **Anthropic** → *Tell me about Anthropic's business and products* → graceful *"newly listed, hasn't filed a 10-K yet"* (degrades cleanly, no crash).

## 📄 Filing Q&A tab (real SEC EDGAR, section-level citations)

Pick **NVDA**, click **Fetch latest 10-K**, then ask:

- What are NVDA's main risk factors?  → cites filing sections `[F1]`
- What does management say about revenue growth and margins?  (MD&A)
- Switch filing type to **10-Q** → What changed this quarter?

## 🔌 MCP panel (under Filing Q&A — "MCP + Laya + Mistral")

The point is **parity**: the same pipeline reached through the Model Context Protocol.
Type a question and click **"Call tool via MCP →"**:

- Tell me about NVDA's business and products  → shows the handshake, the **5 discovered tools**, `route=text`, Mistral's cited answer, and the raw JSON-RPC exchange.
- What is NVDA trading at right now?  → `ask_finsight` routes `real_time` via Laya, through MCP.

**Narration:** "This is the app acting as a real MCP client — it spawns our server over
stdio, discovers the tools, and `ask_finsight` runs Laya + Mistral. Any MCP client (Cursor,
Claude Desktop) gets the identical tools."

---

## If something is off

- **Answers error / "Mistral not reachable"** → the tunnel or instance dropped. Re-open the
  tunnel (new IP after an unshelve) and hard-refresh.
- **"Router: keyword" instead of Laya** → the `:8001` tunnel is down; Laya falls back to
  keyword routing automatically. Re-open the tunnel.
- **MCP panel missing** → it's under the **Filing Q&A** tab, below the filing section.
- **A "Template answer" badge** → the model's wording failed a grounding check and FinSight
  fell back to the facts-only template; the numbers shown are still correct.
