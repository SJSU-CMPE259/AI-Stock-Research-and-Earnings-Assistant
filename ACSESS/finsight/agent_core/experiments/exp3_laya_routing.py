"""Experiment 3 (live): where does Laya fit — does it route to the right tool?

Laya is NOT a generative model: it cannot output a price or a computed number.
It is a router — given a question it picks a path (real_time / numbers / sentiment /
text / refuse). So "does Laya give the right price/number?" does not apply; the
meaningful question is whether Laya sends each question to the tool that produces
the real answer:

  - price questions  -> real_time  (the live Finnhub tool)
  - calc questions   -> numbers    (the Python path)

This is the mechanism that prevents the failures seen in Experiment 2 (Mistral
alone inventing prices and miscalculating): Laya guarantees the question reaches
the right tool.

Run:  ./.venv/bin/python -m agent_core.experiments.exp3_laya_routing
      (needs the Laya endpoint on :8001; add --dry-run for a no-network smoke test)

Writes routing_results.csv + summary.json under agent_core/runs/exp3_<timestamp>/.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

from agent_core.experiments.exp2_python_vs_llm import TICKERS, make_calc_problems


def price_questions() -> list[str]:
    return [f"What is {t} trading at right now?" for t in TICKERS]


def calc_questions(n: int = 100) -> list[str]:
    return [p["question"] for p in make_calc_problems(n)]


def route_all(questions: list[str], expected: str, router) -> list[dict]:
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        route, conf = router.route(q)
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"question": q[:80], "expected": expected, "route": route,
                     "correct": int(route == expected), "confidence": round(conf, 4),
                     "ms": round(ms, 1)})
    return rows


def summarize(rows: list[dict], expected: str) -> dict:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["route"]] = dist.get(r["route"], 0) + 1
    return {
        "n": n, "expected": expected,
        "accuracy": correct / n if n else 0.0,
        "mean_confidence": sum(r["confidence"] for r in rows) / n if n else 0.0,
        "mean_ms": sum(r["ms"] for r in rows) / n if n else 0.0,
        "route_distribution": dist,
    }


def main():
    ap = argparse.ArgumentParser(description="Exp3: does Laya route to the right tool?")
    ap.add_argument("--n", type=int, default=100, help="calc questions to route")
    ap.add_argument("--dry-run", action="store_true", help="fake router, no network")
    args = ap.parse_args()

    if args.dry_run:
        class _Fake:  # routes by a keyword, just to exercise the harness
            def route(self, q):
                return ("real_time", 0.9) if "trading at" in q else ("numbers", 0.9)
        router = _Fake()
    else:
        from agent_core.laya_router import LayaRouter
        router = LayaRouter()

    date = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(__file__).resolve().parents[1] / "runs" / f"exp3_{date}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Routing {len(TICKERS)} price questions + {args.n} calc questions through Laya...")
    price_rows = route_all(price_questions(), "real_time", router)
    calc_rows = route_all(calc_questions(args.n), "numbers", router)

    all_rows = ([{"group": "price", **r} for r in price_rows]
                + [{"group": "calc", **r} for r in calc_rows])
    keys = ["group", "question", "expected", "route", "correct", "confidence", "ms"]
    with open(outdir / "routing_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    ps = summarize(price_rows, "real_time")
    cs = summarize(calc_rows, "numbers")
    (outdir / "summary.json").write_text(json.dumps({"price": ps, "calc": cs}, indent=2))

    print("\n================= EXPERIMENT 3: LAYA ROUTING =================")
    print(f"PRICE questions  -> real_time : {int(ps['accuracy']*ps['n'])}/{ps['n']} "
          f"({ps['accuracy']*100:.0f}%)  mean conf {ps['mean_confidence']:.2f}  "
          f"{ps['mean_ms']:.0f} ms/q  dist={ps['route_distribution']}")
    print(f"CALC  questions  -> numbers   : {int(cs['accuracy']*cs['n'])}/{cs['n']} "
          f"({cs['accuracy']*100:.0f}%)  mean conf {cs['mean_confidence']:.2f}  "
          f"{cs['mean_ms']:.0f} ms/q  dist={cs['route_distribution']}")
    print("\nLaya produces no number itself; it routes each question to the tool that does:")
    print("  price -> real_time -> Finnhub (live)   |   calc -> numbers -> Python (exact)")
    print(f"\nwrote {outdir}/routing_results.csv + summary.json")


if __name__ == "__main__":
    main()
