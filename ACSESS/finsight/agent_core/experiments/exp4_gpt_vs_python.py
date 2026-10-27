"""Experiment 4 (live, PAID): a newer GPT model vs Python on financial math.

The professor asked to test Mistral against a newer GPT model (GPT-5/6). This reuses
the SAME calc problems as Experiment 2 and compares a GPT model to the exact Python
answer, so GPT's accuracy sits next to Mistral's (29%) and Python's (100%).

COST SAFETY — this calls the paid OpenAI API:
  - defaults to 50 questions (hard cap, --n cannot exceed it here),
  - max_completion_tokens is tiny (the reply is a single number),
  - temperature 0, one call per question, no retries that re-ask the model,
  - prints total tokens used so the spend is visible.

Setup:  OPENAI_API_KEY in .env (never on the command line); pick the model with
        --model gpt-5  (or set OPENAI_MODEL in .env).
Run:    ./.venv/bin/python -m agent_core.experiments.exp4_gpt_vs_python --model <name>
Writes gpt_calc_results.csv + summary.json under agent_core/runs/exp4_<timestamp>/.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from agent_core.experiments.exp2_python_vs_llm import (
    CALC_SYS, CALC_TYPES, close, make_calc_problems, parse_number,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MAX_QUESTIONS = 50  # hard cap to keep the paid run cheap


def gpt_call(client, model, question):
    """One chat call -> (text, prompt_tokens, completion_tokens).

    gpt-5 and other reasoning models: temperature must be the default (omit it), and
    max_completion_tokens caps reasoning+answer TOGETHER — too small returns empty, so
    we give room (2000) and use low reasoning effort (fast + cheap; arithmetic is easy).
    """
    msgs = [{"role": "system", "content": CALC_SYS}, {"role": "user", "content": question}]
    try:
        r = client.chat.completions.create(
            model=model, messages=msgs, max_completion_tokens=1000, reasoning_effort="minimal")
    except Exception:
        # Non-reasoning models (e.g. gpt-4o): no reasoning_effort, allow temperature 0.
        r = client.chat.completions.create(
            model=model, messages=msgs, max_tokens=24, temperature=0)
    text = (r.choices[0].message.content or "").strip()
    u = r.usage
    return text, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


def main():
    ap = argparse.ArgumentParser(description="Exp4: newer GPT vs Python on financial math")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL"), help="OpenAI model id (e.g. gpt-5)")
    ap.add_argument("--n", type=int, default=50, help=f"questions (capped at {MAX_QUESTIONS})")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env first (do not pass it on the command line).")
    if not args.model:
        raise SystemExit("Pick a model: --model <id> (e.g. gpt-5) or set OPENAI_MODEL in .env.")
    n = min(args.n, MAX_QUESTIONS)

    from openai import OpenAI
    client = OpenAI()

    date = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(__file__).resolve().parents[1] / "runs" / f"exp4_{date}"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "gpt_calc_results.csv"
    fields = ["id", "type", "question", "truth", "gpt_ans", "gpt_raw",
              "gpt_correct", "gpt_parsed", "gpt_ms", "tok_in", "tok_out"]
    cf = open(csv_path, "w", newline="")
    writer = csv.DictWriter(cf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader(); cf.flush()

    problems = make_calc_problems(n)  # same seed as Exp2 -> identical questions
    rows, tok_in, tok_out = [], 0, 0
    print(f"Running {n} calc questions through OpenAI '{args.model}' (paid)...", flush=True)
    for i, p in enumerate(problems, 1):
        t0 = time.perf_counter()
        text, ti, to = gpt_call(client, args.model, p["question"])
        ms = (time.perf_counter() - t0) * 1000
        tok_in += ti; tok_out += to
        got = parse_number(text)
        ok = got is not None and close(got, p["truth"])
        row = {**p, "gpt_ans": got, "gpt_raw": text.replace("\n", " ")[:50],
               "gpt_correct": int(ok), "gpt_parsed": int(got is not None),
               "gpt_ms": round(ms, 1), "tok_in": ti, "tok_out": to}
        rows.append(row)
        writer.writerow(row); cf.flush()  # save each answer immediately
        print(f"  [{i}/{n}] {p['type']:12s} py={p['truth']:<10} gpt={got} "
              f"{'OK' if ok else 'x'} ({ms:.0f}ms)", flush=True)
    cf.close()

    acc = sum(r["gpt_correct"] for r in rows) / n
    parse_rate = sum(r["gpt_parsed"] for r in rows) / n
    mean_ms = sum(r["gpt_ms"] for r in rows) / n
    by_type = {t: (sum(r["gpt_correct"] for r in rows if r["type"] == t)
                   / max(1, sum(1 for r in rows if r["type"] == t)))
               for t in CALC_TYPES if any(r["type"] == t for r in rows)}

    summary = {"model": args.model, "n": n, "accuracy": acc, "parse_rate": parse_rate,
               "mean_ms": mean_ms, "tokens_in": tok_in, "tokens_out": tok_out, "by_type": by_type}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n============ EXPERIMENT 4: {args.model} vs Python ({n} calcs) ============")
    print(f"   Python accuracy : 100%  (exact)")
    print(f"   {args.model} accuracy : {acc*100:.1f}%   (Mistral-7B was 29% on the same set)")
    print(f"   parse rate      : {parse_rate*100:.0f}%")
    print(f"   speed           : {mean_ms:.0f} ms / calc")
    print(f"   tokens used     : {tok_in} in + {tok_out} out = {tok_in+tok_out} total")
    print("   accuracy by type:")
    for t, a in by_type.items():
        print(f"     {t:14s}: {a*100:5.1f}%")
    print(f"\nwrote {outdir}/gpt_calc_results.csv + summary.json")


if __name__ == "__main__":
    main()
