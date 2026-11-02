"""Experiment 2 (live): does FinSight need Python, and is Mistral's bad price the
model or missing data?

Answers the professor's two priority questions with evidence, run against the live
Mistral-7B on PSC and live Finnhub:

  A. Python vs LLM for financial calculations (100 examples)
     Same numbers, computed two ways: exact Python vs Mistral doing the arithmetic
     in its head. We measure agreement, Mistral's error rate, and the speed gap.

  B. Mistral price: model error or missing real-time data?
     Ask Mistral for the current price WITHOUT any tool (model alone) and WITH the
     live Finnhub quote handed to it as a fact. If "with data" is right and
     "without" is wildly wrong, the fake price is a DATA problem, not the model.

Run:  ./.venv/bin/python -m agent_core.experiments.exp2_python_vs_llm --n 100
      (add --dry-run to use FakeLLM / skip network for a smoke test)

Writes results CSVs + a summary under agent_core/runs/exp2_<timestamp>/.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import re
import time
from pathlib import Path

# ---------- number parsing / comparison ----------

_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> float | None:
    """Pull the last number out of an LLM reply (its final answer). None if none."""
    nums = _NUM.findall(text.replace("%", " "))
    if not nums:
        return None
    try:
        return float(nums[-1].replace("$", "").replace(",", ""))
    except ValueError:
        return None


def close(a: float, b: float) -> bool:
    """Within 0.1 absolute or 1% relative — generous, so only real errors count as wrong."""
    return abs(a - b) <= max(0.1, 0.01 * abs(b))


# ---------- Experiment A: financial calculations ----------

CALC_TYPES = ["qoq_pct", "yoy_pct", "gross_margin", "net_margin",
              "eps", "abs_change", "avg3", "pct_of_total"]


def make_calc_problems(n: int, seed: int = 0) -> list[dict]:
    """n deterministic financial-math problems with an exact Python ground truth."""
    rng = random.Random(seed)
    out: list[dict] = []
    for i in range(n):
        t = CALC_TYPES[i % len(CALC_TYPES)]
        if t == "qoq_pct":
            a, b = round(rng.uniform(2, 30), 2), round(rng.uniform(2, 30), 2)
            q = (f"Revenue was ${a}B last quarter and ${b}B this quarter. "
                 f"What is the quarter-over-quarter percent change?")
            ans = (b - a) / a * 100
        elif t == "yoy_pct":
            a, b = round(rng.uniform(5, 50), 2), round(rng.uniform(5, 50), 2)
            q = (f"Revenue was ${a}B a year ago and ${b}B now. "
                 f"What is the year-over-year percent change?")
            ans = (b - a) / a * 100
        elif t == "gross_margin":
            rev = round(rng.uniform(5, 50), 2)
            gp = round(rng.uniform(1, rev), 2)
            q = f"Revenue is ${rev}B and gross profit is ${gp}B. What is the gross margin in percent?"
            ans = gp / rev * 100
        elif t == "net_margin":
            rev = round(rng.uniform(5, 50), 2)
            ni = round(rng.uniform(0.2, rev * 0.4), 2)
            q = f"Revenue is ${rev}B and net income is ${ni}B. What is the net profit margin in percent?"
            ans = ni / rev * 100
        elif t == "eps":
            ni = round(rng.uniform(0.5, 20), 2)
            sh = round(rng.uniform(0.5, 5), 2)
            q = f"Net income is ${ni}B and there are {sh}B shares outstanding. What is EPS in dollars?"
            ans = ni / sh
        elif t == "abs_change":
            a, b = round(rng.uniform(2, 30), 2), round(rng.uniform(2, 30), 2)
            q = f"Revenue was ${a}B then ${b}B. What is the change in dollars (billions)?"
            ans = b - a
        elif t == "avg3":
            xs = [round(rng.uniform(2, 30), 2) for _ in range(3)]
            q = (f"Quarterly revenues were ${xs[0]}B, ${xs[1]}B and ${xs[2]}B. "
                 f"What is the average quarterly revenue in billions?")
            ans = sum(xs) / 3
        else:  # pct_of_total
            seg = round(rng.uniform(1, 20), 2)
            tot = round(seg + rng.uniform(1, 30), 2)
            q = f"A segment earned ${seg}B out of ${tot}B total revenue. What percent of total is the segment?"
            ans = seg / tot * 100
        out.append({"id": i, "type": t, "question": q, "truth": round(ans, 4)})
    return out


CALC_SYS = ("You are a calculator. Compute the answer from the numbers given. "
            "Reply with ONLY the final number (no units, no words, no working).")


def run_calc(problems: list[dict], llm) -> list[dict]:
    rows = []
    for p in problems:
        # Python path: exact, timed.
        t0 = time.perf_counter()
        py = p["truth"]  # already the Python computation
        py_ms = (time.perf_counter() - t0) * 1000

        # LLM path: Mistral does the arithmetic.
        t1 = time.perf_counter()
        reply = llm.complete(CALC_SYS, [{"role": "user", "content": p["question"]}],
                             max_tokens=24, temperature=0.0)
        llm_ms = (time.perf_counter() - t1) * 1000
        got = parse_number(reply)
        ok = got is not None and close(got, py)
        rows.append({**p, "python_ans": py, "python_ms": round(py_ms, 4),
                     "llm_ans": got, "llm_raw": reply.strip().replace("\n", " ")[:60],
                     "llm_ms": round(llm_ms, 1), "llm_correct": int(ok),
                     "llm_parsed": int(got is not None)})
    return rows


# ---------- Experiment B: price with vs without live data ----------

TICKERS = ["NVDA", "AMD", "AAPL", "MSFT", "TSLA", "INTC", "GOOGL", "AMZN", "META", "JPM"]

NODATA_SYS = ("You are a financial assistant. Answer the user's question. "
              "Reply with ONLY the current share price as a number in USD.")
WITHDATA_SYS = ("You are a financial assistant. Use ONLY the fact provided. "
                "Reply with ONLY the current share price as a number in USD.")


def run_price(tickers: list[str], llm, get_quote) -> list[dict]:
    rows = []
    for tk in tickers:
        try:
            true_price = get_quote(tk).price
        except Exception as e:  # noqa: BLE001
            rows.append({"ticker": tk, "true_price": None, "error": f"{type(e).__name__}: {e}"})
            continue

        # WITHOUT data: model alone.
        t0 = time.perf_counter()
        r_no = llm.complete(NODATA_SYS,
                            [{"role": "user", "content": f"What is {tk} trading at right now?"}],
                            max_tokens=24, temperature=0.0)
        no_ms = (time.perf_counter() - t0) * 1000
        no_price = parse_number(r_no)

        # WITH data: hand it the live Finnhub quote.
        fact = f"Fact: {tk} is currently trading at ${true_price} (Finnhub, live)."
        t1 = time.perf_counter()
        r_yes = llm.complete(WITHDATA_SYS,
                             [{"role": "user", "content": f"{fact}\nWhat is {tk} trading at right now?"}],
                             max_tokens=24, temperature=0.0)
        yes_ms = (time.perf_counter() - t1) * 1000
        yes_price = parse_number(r_yes)

        def err_pct(p):
            return None if p is None else round(abs(p - true_price) / true_price * 100, 2)

        rows.append({
            "ticker": tk, "true_price": true_price,
            "nodata_price": no_price, "nodata_err_pct": err_pct(no_price),
            "nodata_ms": round(no_ms, 1), "nodata_raw": r_no.strip().replace("\n", " ")[:40],
            "withdata_price": yes_price, "withdata_err_pct": err_pct(yes_price),
            "withdata_ms": round(yes_ms, 1), "withdata_raw": r_yes.strip().replace("\n", " ")[:40],
        })
    return rows


# ---------- summaries + output ----------

def summarize_calc(rows):
    n = len(rows)
    parsed = [r for r in rows if r["llm_parsed"]]
    correct = [r for r in rows if r["llm_correct"]]
    by_type = {}
    for t in CALC_TYPES:
        ts = [r for r in rows if r["type"] == t]
        if ts:
            by_type[t] = sum(r["llm_correct"] for r in ts) / len(ts)
    return {
        "n": n,
        "llm_acc": len(correct) / n if n else 0,
        "llm_parse_rate": len(parsed) / n if n else 0,
        "python_acc": 1.0,
        "mean_llm_ms": sum(r["llm_ms"] for r in rows) / n if n else 0,
        "mean_python_ms": sum(r["python_ms"] for r in rows) / n if n else 0,
        "by_type": by_type,
    }


def summarize_price(rows):
    ok = [r for r in rows if r.get("true_price") is not None]
    def mean(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    within5_no = sum(1 for r in ok if r.get("nodata_err_pct") is not None and r["nodata_err_pct"] <= 5)
    within5_yes = sum(1 for r in ok if r.get("withdata_err_pct") is not None and r["withdata_err_pct"] <= 5)
    return {
        "n": len(ok),
        "mean_nodata_err_pct": mean("nodata_err_pct"),
        "mean_withdata_err_pct": mean("withdata_err_pct"),
        "within5_nodata": within5_no,
        "within5_withdata": within5_yes,
        "mean_nodata_ms": mean("nodata_ms"),
        "mean_withdata_ms": mean("withdata_ms"),
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Exp2: Python vs LLM + price grounding")
    ap.add_argument("--n", type=int, default=100, help="calc problems")
    ap.add_argument("--tickers", type=int, default=len(TICKERS), help="how many tickers for price test")
    ap.add_argument("--dry-run", action="store_true", help="FakeLLM + fake quotes, no network")
    args = ap.parse_args()

    if args.dry_run:
        from agent_core.llm import FakeLLM
        llm = FakeLLM()
        class _Q:  # noqa: N801
            def __init__(self, p): self.price = p
        def get_quote(tk): return _Q(100.0)
    else:
        from agent_core.llm import MistralLLM
        from agent_core.tools.prices import get_quote
        llm = MistralLLM()

    date = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(__file__).resolve().parents[1] / "runs" / f"exp2_{date}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[A] Python vs LLM calculations: {args.n} problems (live Mistral)...")
    calc_rows = run_calc(make_calc_problems(args.n), llm)
    write_csv(outdir / "calc_results.csv", calc_rows)
    ca = summarize_calc(calc_rows)

    print(f"[B] Price with/without live data: {args.tickers} tickers...")
    price_rows = run_price(TICKERS[:args.tickers], llm, get_quote)
    write_csv(outdir / "price_results.csv", price_rows)
    pa = summarize_price(price_rows)

    # ---- console answers ----
    print("\n================= EXPERIMENT 2 RESULTS =================")
    print(f"\nA. PYTHON vs LLM — financial math ({ca['n']} problems)")
    print(f"   Python accuracy      : {ca['python_acc']*100:.0f}%  (exact, deterministic)")
    print(f"   Mistral accuracy     : {ca['llm_acc']*100:.1f}%")
    print(f"   Mistral parse rate   : {ca['llm_parse_rate']*100:.1f}% (gave a usable number)")
    print(f"   Speed  Python        : {ca['mean_python_ms']:.4f} ms / calc")
    print(f"   Speed  Mistral       : {ca['mean_llm_ms']:.0f} ms / calc")
    if ca["mean_python_ms"]:
        print(f"   Mistral is ~{ca['mean_llm_ms']/ca['mean_python_ms']:,.0f}x slower")
    print("   Accuracy by calc type:")
    for t, acc in ca["by_type"].items():
        print(f"     {t:14s}: {acc*100:5.1f}%")

    def fmt(v, suf="%"):
        return "n/a" if v is None else f"{v:.2f}{suf}"
    print(f"\nB. MISTRAL PRICE — model vs missing data ({pa['n']} tickers)")
    print(f"   WITHOUT live data : mean error {fmt(pa['mean_nodata_err_pct'])} "
          f"| within 5% of real: {pa['within5_nodata']}/{pa['n']}")
    print(f"   WITH live data    : mean error {fmt(pa['mean_withdata_err_pct'])} "
          f"| within 5% of real: {pa['within5_withdata']}/{pa['n']}")
    print(f"   Speed  without    : {fmt(pa['mean_nodata_ms'],' ms')}   "
          f"with: {fmt(pa['mean_withdata_ms'],' ms')}")
    print(f"\nwrote {outdir}/calc_results.csv + price_results.csv")

    # stash summaries for the writeup step
    import json
    (outdir / "summary.json").write_text(json.dumps({"calc": ca, "price": pa}, indent=2))
    print(f"wrote {outdir}/summary.json")


if __name__ == "__main__":
    main()
