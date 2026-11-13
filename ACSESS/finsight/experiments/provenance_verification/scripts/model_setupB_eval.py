"""Model comparison on FinSight's 100-example decision set (Setup B: model-only).

Reuses the harness's gold data (routing_questions.csv + sentiment_statements.csv),
option definitions, and metric functions, so scores are directly comparable to the
Mistral runs. Adds a reasoning-model path for DeepSeek-R1 (which emits <think>...</think>
before its answer): more tokens, strip the think block, parse the final label.

Run (from finsight/):  .venv/bin/python /path/to/r1_eval.py <model> <mode>
  mode = direct  -> Mistral-style: max_tokens=5, temp=0, first word
  mode = reason  -> R1-style: max_tokens=1024, temp=0.6, parse after </think>
"""
from __future__ import annotations
import os, re, sys, time, math
import requests

sys.path.insert(0, os.path.abspath("."))
from agent_core.eval import metrics as M
from agent_core.eval.data import load_csv, select_split

BASE = os.environ.get("REMOTE_LLM_URL", "http://localhost:8000/v1").rstrip("/")

INTENTS = ["numbers", "real_time", "sentiment", "text", "refuse", "out_of_scope"]
SENTIMENTS = ["positive", "neutral", "negative"]
ROUTING_OPTIONS = {
    "numbers": "revenue, income, eps, margin, growth, last quarter, change",
    "real_time": "price, today, right now, current, trading",
    "sentiment": "tone, outlook, positive, negative, sound",
    "text": "business, about, products, do, summarize, risk",
    "refuse": "buy, sell, hold, invest, recommend",
}
SENTIMENT_OPTIONS = {
    "positive": "favorable, good, growth, record, improved, strong, optimistic",
    "neutral": "in line, flat, factual, mixed, increased expenses",
    "negative": "declined, risk, constraint, competition, adverse, may not, softened",
}
ADVICE_Q = "Is the user asking for buy/sell/hold investment advice?"


def strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def first_label(text: str, labels: list[str]) -> str | None:
    low = text.lower()
    hits = [(low.find(lab), lab) for lab in labels if low.find(lab) >= 0]
    return min(hits)[1] if hits else None


def ask(model: str, mode: str, prompt: str) -> tuple[str, float]:
    if mode == "reason":
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 1024, "temperature": 0.6}
    else:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 5, "temperature": 0}
    t0 = time.time()
    r = requests.post(f"{BASE}/chat/completions", json=payload, timeout=180)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    text = msg.get("content") or ""
    if mode == "reason":
        text = strip_think(text)
    return text, time.time() - t0


def decide_choice(model, mode, state, instr, options):
    opts = list(options)
    prompt = (f"{instr}\nAnswer with exactly one word from: {', '.join(opts)}.\n"
              f"Text: {state}\nAnswer:")
    text, lat = ask(model, mode, prompt)
    word = first_label(text, opts) if mode == "reason" else (
        (re.search(r"[a-zA-Z_]+", text) or [None])[0])
    if mode != "reason" and word:
        word = word.lower()
    valid = word in opts
    return (word if valid else None), valid, lat


def decide_yesno(model, mode, state, instr):
    prompt = f"{instr}\nAnswer with exactly one word, yes or no.\nText: {state}\nAnswer:"
    text, lat = ask(model, mode, prompt)
    word = first_label(text, ["yes", "no"]) if mode == "reason" else (
        (re.search(r"[a-zA-Z_]+", text) or [""])[0].lower())
    valid = word in ("yes", "no")
    return (word if valid else None), valid, lat


def run(model: str, mode: str):
    routing = select_split(load_csv("routing_questions.csv"), "all", 0)
    sentiment = select_split(load_csv("sentiment_statements.csv"), "all", 0)

    ig, ip, ag, ap, sg, sp = [], [], [], [], [], []
    valids, lats = [], []
    for row in routing:
        q = row["question"]
        w, v, lat = decide_choice(model, mode, q, "Classify the user's request.", ROUTING_OPTIONS)
        ig.append(row["gold_intent"]); ip.append(w); valids.append(v); lats.append(lat)
        w, v, lat = decide_yesno(model, mode, q, ADVICE_Q)
        ag.append(row["gold_is_advice"].strip().lower() == "yes"); ap.append(w == "yes")
        lats.append(lat)
    for row in sentiment:
        s = row["statement"]
        w, v, lat = decide_choice(model, mode, s, "Classify the sentiment of the statement.", SENTIMENT_OPTIONS)
        sg.append(row["gold_label"]); sp.append(w); valids.append(v); lats.append(lat)

    acc_items = [int(g == p) for g, p in zip(ig, ip)] + [int(g == p) for g, p in zip(sg, sp)]
    ci = M.bootstrap_ci(acc_items, n_resamples=1000)
    print(f"\n=== {model} (mode={mode}) · {len(ig)} routing + {len(sg)} sentiment = {len(acc_items)} items ===")
    print(f"accuracy (intent+sentiment): {sum(acc_items)/len(acc_items):.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"intent accuracy:  {M.accuracy(ig, ip):.4f}")
    print(f"intent macro-F1:  {M.macro_f1(ig, ip, INTENTS):.4f}")
    print(f"sentiment acc:    {M.accuracy(sg, sp):.4f}")
    print(f"sentiment F1:     {M.macro_f1(sg, sp, SENTIMENTS):.4f}")
    print(f"advice recall:    {M.advice_recall(ag, ap):.4f}")
    print(f"false refusal:    {M.false_refusal_rate(ag, ap):.4f}")
    print(f"valid-output:     {M.valid_output_rate(valids):.4f}")
    print(f"mean latency/dec: {1000*sum(lats)/len(lats):.0f} ms   (total {sum(lats):.0f}s over {len(lats)} calls)")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-r1:8b"
    mode = sys.argv[2] if len(sys.argv) > 2 else "reason"
    run(model, mode)
