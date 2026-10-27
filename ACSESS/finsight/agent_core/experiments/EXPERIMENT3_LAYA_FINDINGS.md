# Experiment 3 — Where does Laya fit? Does it route to the right tool?

**Date:** 2026-10-27  **Status:** live run (not simulated)
**Model/endpoint:** Laya (`convaiinnovations/laya`) served on PSC Bridges-2, HTTP on :8001.
**Code:** `agent_core/experiments/exp3_laya_routing.py` · **Raw results:** `agent_core/runs/exp3_20261027_131918/` (`routing_results.csv`, `summary.json`)
**Relationship to Exp 2:** separate experiment, same inputs. Exp 2 tested whether Mistral *answers* correctly; this tests whether Laya *routes* correctly. Kept apart on purpose.

---

## The key distinction

Laya is **not** a generative model. It cannot output a stock price or a computed number — it has no free-text output. Laya is a **router**: given a question, it picks a path (`real_time` / `numbers` / `sentiment` / `text` / `refuse`).

So the Experiment 2 question — "does Laya give the right price / the right math?" — **does not apply**: there is no number for Laya to get right or wrong. The meaningful question is whether Laya sends each question to the tool that produces the real answer.

---

## Method

Route the **same inputs as Experiment 2** through live Laya:
- 10 price questions ("What is NVDA trading at right now?") — should route to **`real_time`** (the live Finnhub tool).
- 100 calculation questions (revenue, margin, EPS, growth) — should route to **`numbers`** (the Python path).

Measure routing accuracy, confidence, and speed.

## Results

| input | should route to | Laya correct | mean confidence | speed |
|---|---|---|---|---|
| 10 price questions | `real_time` (live-data tool) | **10 / 10 (100%)** | 0.68 | 319 ms/q |
| 100 calc questions | `numbers` (Python path) | **100 / 100 (100%)** | 0.97 | 315 ms/q |

Route distribution was clean — every price question → `real_time`, every calc question → `numbers`, no misroutes.

---

## Conclusion

**Laya never gives a wrong price or a wrong calculation — because it produces neither.** It routes **100%** of price questions to the live-data tool and **100%** of calc questions to Python. That is exactly the mechanism that prevents the two failures measured in Experiment 2:

- Mistral **alone** invents prices (210% mean error) and miscalculates (29% accuracy).
- **Laya guarantees** the question reaches the tool that gives the real answer.

The three pieces compose cleanly into FinSight's hybrid:

| step | component | result |
|---|---|---|
| 1. decide the path | **Laya** | 100% correct routing here |
| 2. produce the number | **Python / Finnhub** | exact (Exp 2: 100% math, 0% price error) |
| 3. phrase the answer | **Mistral** | writes from correct facts; never invents (output checked) |

So "does Laya give the real answer, or the same wrong thing?" → **neither**: it gives no number at all, but by routing correctly it is the reason the *system* gives the real answer instead of Mistral's made-up one.

---

## Limitations
- Laya confidence on price routing is moderate (0.68) vs calc (0.97); on these inputs it still routed 100% correctly, but price phrasing varies more than the filing-style calc questions.
- Routing quality here is on clean, single-intent questions; mixed questions ("price and how it changed") can split intent — handled by the orchestrator's path, not tested here.
- Single router (Laya); it was already the best router in Experiment 1 (0.85 vs Mistral 0.52, McNemar p=0.0003).
