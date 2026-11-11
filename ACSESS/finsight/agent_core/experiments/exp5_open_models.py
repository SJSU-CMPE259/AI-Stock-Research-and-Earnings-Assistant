"""Experiment 5 (live): current open models vs Python on financial math.

Exp 2 showed Mistral-7B gets 29% of the calc problems right and Exp 4 showed gpt-5 gets
98% (paid). The professor asked whether Mistral-7B is still current. This runs FREE open
models through the same problems, one model per run:
  - mistral          Mistral-7B v0.3, what FinSight uses today (baseline, same GPU)
  - ministral-3:8b   the newest small Mistral family, same size class
  - deepseek-r1:8b   an open reasoning model (R1 distill)
Do newer open models close the gap to gpt-5, and do they still lose to Python?

Same problems as Exp 2/4 (make_calc_problems, same seed), same system prompt, same
"within 0.1 abs or 1% rel" bar. Accuracy is reported on all n problems AND on the first
50, which is exactly the set gpt-5 ran in Exp 4, so the table is like-for-like.

R1 writes a long <think> trace before answering, so unlike Exp 2 (24-token cap) every
model gets a large token budget, and only the text after any reasoning is parsed.

Models are served by Ollama (native /api/chat), e.g. on Colab or the PSC GPU:
    ollama pull deepseek-r1:8b
Run:  python -m agent_core.experiments.exp5_open_models --model deepseek-r1:8b --n 100
      (--dry-run exercises the harness with canned replies, no server needed)
Writes calc_results.csv + summary.json under agent_core/runs/exp5_<model>_<timestamp>/.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import time
from pathlib import Path

import requests

from agent_core.experiments.exp2_python_vs_llm import (
    CALC_SYS, CALC_TYPES, close, make_calc_problems, parse_number,
)

GPT5_SET = 50  # Exp 4 ran gpt-5 on the first 50 problems
_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)


def final_answer(content: str) -> str:
    """Drop a reasoning model's <think>...</think> block (also an unterminated one) and keep the answer."""
    return _THINK.sub("", content).strip()


def ollama_call(url, model, question, max_tokens, temperature, timeout):
    """One chat call -> (answer_text, thinking_chars, prompt_tokens, output_tokens)."""
    r = requests.post(f"{url}/api/chat", timeout=timeout, json={
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": CALC_SYS},
                     {"role": "user", "content": question}],
        "options": {"temperature": temperature, "num_predict": max_tokens},
    })
    r.raise_for_status()
    d = r.json()
    msg = d.get("message", {})
    content = msg.get("content", "")
    # newer Ollama returns the trace in message.thinking; older ones inline it as <think>
    thinking = msg.get("thinking") or "".join(m.group(0) for m in _THINK.finditer(content))
    return final_answer(content), len(thinking), d.get("prompt_eval_count", 0), d.get("eval_count", 0)


def summarize(rows, model):
    def acc(rs):
        return sum(r["llm_correct"] for r in rs) / len(rs) if rs else None
    n = len(rows)
    return {
        "model": model, "n": n,
        "accuracy": acc(rows),
        "accuracy_first50": acc(rows[:GPT5_SET]) if n >= GPT5_SET else None,
        "errors": sum(1 for r in rows if r["error"]),
        "parse_rate": sum(r["llm_parsed"] for r in rows) / n,
        "mean_ms": sum(r["llm_ms"] for r in rows) / n,
        "mean_output_tokens": sum(r["tok_out"] for r in rows) / n,
        "tokens_in": sum(r["tok_in"] for r in rows),
        "tokens_out": sum(r["tok_out"] for r in rows),
        "by_type": {t: acc([r for r in rows if r["type"] == t]) for t in CALC_TYPES
                    if any(r["type"] == t for r in rows)},
    }


def main():
    ap = argparse.ArgumentParser(description="Exp5: open models vs Python on financial math")
    ap.add_argument("--model", default="deepseek-r1:8b", help="Ollama model tag")
    ap.add_argument("--url", default="http://localhost:11434", help="Ollama server")
    ap.add_argument("--n", type=int, default=100, help="calc problems (same seed as Exp 2)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="reasoning + answer budget")
    # 0 like Exp 2; R1 gets 0.6 by default (DeepSeek's advice: greedy decoding can loop in <think>)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--timeout", type=int, default=600, help="seconds per question")
    ap.add_argument("--dry-run", action="store_true", help="canned replies, no server")
    args = ap.parse_args()

    if args.temperature is None:
        args.temperature = 0.6 if "deepseek-r1" in args.model else 0.0
    date = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = re.sub(r"[^A-Za-z0-9.]+", "-", args.model)
    outdir = Path(__file__).resolve().parents[1] / "runs" / f"exp5_{tag}_{date}"
    outdir.mkdir(parents=True, exist_ok=True)
    fields = ["id", "type", "question", "truth", "llm_ans", "llm_raw", "llm_correct",
              "llm_parsed", "llm_ms", "think_chars", "tok_in", "tok_out", "error"]
    cf = open(outdir / "calc_results.csv", "w", newline="")
    writer = csv.DictWriter(cf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader(); cf.flush()

    problems = make_calc_problems(args.n)
    print(f"Running {len(problems)} calc questions through '{args.model}' at {args.url}...", flush=True)
    rows = []
    for i, p in enumerate(problems, 1):
        t0 = time.perf_counter()
        err = ""
        try:
            if args.dry_run:
                text, think, ti, to = final_answer(f"<think>work it out</think>\n{p['truth']:.2f}"), 15, 0, 0
            else:
                text, think, ti, to = ollama_call(args.url, args.model, p["question"],
                                                  args.max_tokens, args.temperature, args.timeout)
        except Exception as e:  # noqa: BLE001 - a failed call is scored wrong, run continues
            text, think, ti, to, err = "", 0, 0, 0, f"{type(e).__name__}: {e}"[:120]
        ms = (time.perf_counter() - t0) * 1000
        got = parse_number(text)
        ok = got is not None and close(got, p["truth"])
        row = {**p, "llm_ans": got, "llm_raw": text.replace("\n", " ")[:60], "llm_correct": int(ok),
               "llm_parsed": int(got is not None), "llm_ms": round(ms, 1), "think_chars": think,
               "tok_in": ti, "tok_out": to, "error": err}
        rows.append(row)
        writer.writerow(row); cf.flush()  # save each answer immediately (Colab can disconnect)
        print(f"  [{i}/{len(problems)}] {p['type']:12s} py={p['truth']:<10} llm={got} "
              f"{'OK' if ok else 'x'} ({ms:.0f}ms, {to} tok){'  ' + err if err else ''}", flush=True)
    cf.close()

    s = summarize(rows, args.model)
    s.update({"temperature": args.temperature, "max_tokens": args.max_tokens, "dry_run": args.dry_run})
    (outdir / "summary.json").write_text(json.dumps(s, indent=2))

    print(f"\n============ EXPERIMENT 5: {args.model} vs Python ({s['n']} calcs) ============")
    print("   Python accuracy           : 100%  (exact)")
    print(f"   model accuracy (all {s['n']:>3})  : {s['accuracy']*100:.1f}%   (Mistral-7B on PSC was 29% in Exp 2)")
    if s["accuracy_first50"] is not None:
        print(f"   model accuracy (first 50) : {s['accuracy_first50']*100:.1f}%   (gpt-5 was 98% on this set)")
    print(f"   parse rate             : {s['parse_rate']*100:.0f}%")
    if s["errors"]:
        print(f"   !! {s['errors']} calls FAILED (see the error column) - not a valid run")
    print(f"   speed                  : {s['mean_ms']:.0f} ms / calc, {s['mean_output_tokens']:.0f} output tokens")
    print("   accuracy by type:")
    for t, a in s["by_type"].items():
        print(f"     {t:14s}: {a*100:5.1f}%")
    print(f"\nwrote {outdir}/calc_results.csv + summary.json")


if __name__ == "__main__":
    main()
