"""Run the eval: `python -m agent_core.eval --setup all --split test [--dry-run]`.

Covers decision points 1 (routing intent + advice) and 2 (sentiment) end-to-end.
Reranking / answer-check / e2e plug into the same structure and are added next.
Writes results.csv + report.md (+ a chart) under agent_core/runs/eval_<date>/.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agent_core.eval import metrics as M  # noqa: E402
from agent_core.eval.clients import (  # noqa: E402
    FakeLaya, FakeLLM, LayaDecider, MistralDecider, RulesDecider,
)
from agent_core.eval.data import load_csv, select_split  # noqa: E402

SETUPS = ["A", "B", "C", "D", "E", "F"]
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
ADVICE_KW = ["buy", "sell", "hold", "invest", "recommend"]


def make_deciders(dry_run: bool):
    rules = RulesDecider()
    if dry_run:
        return rules, FakeLLM(), FakeLaya()
    return rules, MistralDecider(), LayaDecider()


def run_choice(setup, state, instr, options, rules, mistral, laya, threshold, best="laya"):
    """Apply a setup to a choice decision. `best` = which model setup F uses here."""
    if setup == "A":
        return rules.decide_choice(state, instr, options)
    if setup == "B":
        return mistral.decide_choice(state, instr, options)
    if setup == "C":
        return laya.decide_choice(state, instr, options)
    if setup == "D":
        d = laya.decide_choice(state, instr, options)
        return d if d.confidence >= threshold else mistral.decide_choice(state, instr, options)
    if setup == "F":  # hybrid: best model for this decision point
        return (laya if best == "laya" else mistral).decide_choice(state, instr, options)
    # E: agree or needs_review
    a = laya.decide_choice(state, instr, options)
    b = mistral.decide_choice(state, instr, options)
    if a.answer == b.answer:
        return a
    a.answer = "needs_review"
    return a


def run_yesno(setup, state, instr, kw, rules, mistral, laya, threshold, best="laya"):
    if setup == "A":
        return rules.decide_yesno(state, instr, kw)
    if setup == "B":
        return mistral.decide_yesno(state, instr, kw)
    if setup == "C":
        return laya.decide_yesno(state, instr, kw)
    if setup == "D":
        d = laya.decide_yesno(state, instr, kw)
        return d if d.confidence >= threshold else mistral.decide_yesno(state, instr, kw)
    if setup == "F":
        return (laya if best == "laya" else mistral).decide_yesno(state, instr, kw)
    a = laya.decide_yesno(state, instr, kw)
    b = mistral.decide_yesno(state, instr, kw)
    if a.answer == b.answer:
        return a
    a.answer = "needs_review"
    return a


def evaluate(setups, split, dry_run, threshold, seed=0):
    rules, mistral, laya = make_deciders(dry_run)
    routing = select_split(load_csv("routing_questions.csv"), split, seed)
    sentiment = select_split(load_csv("sentiment_statements.csv"), split, seed)

    records = []  # one row per item x setup x decision_point
    for setup in setups:
        for row in routing:
            q = row["question"]
            di = run_choice(setup, q, "Classify the user's request.", ROUTING_OPTIONS,
                            rules, mistral, laya, threshold, best="laya")
            records.append(_rec(setup, "routing_intent", q, row["gold_intent"], di))
            da = run_yesno(setup, q, ADVICE_Q, ADVICE_KW, rules, mistral, laya, threshold, best="laya")
            gold_adv = "yes" if row["gold_is_advice"].strip().lower() == "yes" else "no"
            records.append(_rec(setup, "advice", q, gold_adv, da))
        for row in sentiment:
            s = row["statement"]
            ds = run_choice(setup, s, "Classify the sentiment of the statement.",
                            SENTIMENT_OPTIONS, rules, mistral, laya, threshold, best="mistral")
            records.append(_rec(setup, "sentiment", s, row["gold_label"], ds))
    return records


def _rec(setup, dp, text, gold, d):
    return {
        "setup": setup, "decision_point": dp, "text": text[:80],
        "gold": gold, "pred": d.answer, "correct": int(d.answer == gold),
        "confidence": round(d.confidence, 4), "valid": int(d.valid),
        "source": d.source, "latency_s": round(d.latency_s, 4),
    }


def summarize(records, setups):
    """Per-setup metric rows for the summary table."""
    rows = {}
    for setup in setups:
        rs = [r for r in records if r["setup"] == setup]
        intent = [r for r in rs if r["decision_point"] == "routing_intent"]
        advice = [r for r in rs if r["decision_point"] == "advice"]
        sent = [r for r in rs if r["decision_point"] == "sentiment"]
        acc_all = [r["correct"] for r in intent + sent]
        ci = M.bootstrap_ci(acc_all, n_resamples=1000)
        rows[setup] = {
            "intent_acc": M.accuracy([r["gold"] for r in intent], [r["pred"] for r in intent]),
            "intent_f1": M.macro_f1([r["gold"] for r in intent], [r["pred"] for r in intent], INTENTS),
            "advice_recall": M.advice_recall([r["gold"] == "yes" for r in advice],
                                             [r["pred"] == "yes" for r in advice]),
            "false_refusal": M.false_refusal_rate([r["gold"] == "yes" for r in advice],
                                                  [r["pred"] == "yes" for r in advice]),
            "sent_acc": M.accuracy([r["gold"] for r in sent], [r["pred"] for r in sent]),
            "sent_f1": M.macro_f1([r["gold"] for r in sent], [r["pred"] for r in sent], SENTIMENTS),
            "valid_rate": M.valid_output_rate([bool(r["valid"]) for r in intent + sent]),
            "mean_latency_ms": 1000 * (sum(r["latency_s"] for r in rs) / len(rs) if rs else 0),
            "acc_all": sum(acc_all) / len(acc_all) if acc_all else 0.0,
            "acc_ci": ci,
            "needs_review": sum(r["pred"] == "needs_review" for r in rs) / len(rs) if rs else 0.0,
        }
    return rows


def write_outputs(records, summary, setups, split, threshold, dry_run):
    date = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(__file__).resolve().parents[1] / "runs" / f"eval_{date}"
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    # chart: accuracy vs mean latency per setup
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in setups:
        ax.scatter(summary[s]["mean_latency_ms"], summary[s]["acc_all"], s=80)
        ax.annotate(s, (summary[s]["mean_latency_ms"], summary[s]["acc_all"]))
    ax.set_xlabel("mean latency per decision (ms)")
    ax.set_ylabel("accuracy (intent+sentiment)")
    ax.set_title(f"Accuracy vs latency per setup ({'DRY-RUN' if dry_run else split})")
    fig.tight_layout(); fig.savefig(outdir / "accuracy_vs_latency.png", dpi=120); plt.close(fig)

    names = {"A": "rules", "B": "mistral", "C": "laya", "D": "cascade", "E": "agree",
             "F": "hybrid (laya route + mistral sentiment)"}
    lines = [f"# FinSight eval report ({'DRY-RUN — FakeLaya/FakeLLM' if dry_run else split} split)\n",
             f"_generated {date}; cascade threshold={threshold}_\n",
             "\n## 1. Summary (setups x key metrics)\n",
             "| setup | acc (95% CI) | intent F1 | advice recall | false refusal | sentiment acc | valid rate | mean latency |",
             "|---|---|---|---|---|---|---|---|"]
    for s in setups:
        m = summary[s]
        lines.append(
            f"| {s} {names[s]} | {m['acc_all']:.2f} [{m['acc_ci'][0]:.2f},{m['acc_ci'][1]:.2f}] | "
            f"{m['intent_f1']:.2f} | {m['advice_recall']:.2f} | {m['false_refusal']:.2f} | "
            f"{m['sent_acc']:.2f} | {m['valid_rate']:.2f} | {m['mean_latency_ms']:.0f} ms |")
    # McNemar C vs B and D vs B on intent+sentiment correctness
    def corr(setup):
        return [r["correct"] for r in records if r["setup"] == setup
                and r["decision_point"] in ("routing_intent", "sentiment")]
    lines += ["\n## 2. Significance (paired McNemar, intent+sentiment)\n"]
    if "B" in setups and "C" in setups:
        lines.append(f"- C vs B: {M.mcnemar([bool(x) for x in corr('C')], [bool(x) for x in corr('B')])}")
    if "B" in setups and "D" in setups:
        lines.append(f"- D vs B: {M.mcnemar([bool(x) for x in corr('D')], [bool(x) for x in corr('B')])}")
    if "C" in setups and "F" in setups:
        lines.append(f"- F vs C: {M.mcnemar([bool(x) for x in corr('F')], [bool(x) for x in corr('C')])}")
    lines += [
        "\n## 3. Thresholds\n",
        f"- cascade (setup D) confidence threshold = {threshold} (tune on dev; report on test).",
        "\n## 4. Limitations\n",
        "- Small sets; one-company subset; draft gold (pending correction); routing lacks an explicit out_of_scope branch.",
        "\nFiles: `results.csv`, `accuracy_vs_latency.png` in this folder.",
    ]
    (outdir / "report.md").write_text("\n".join(lines) + "\n")
    return outdir


def main():
    ap = argparse.ArgumentParser(description="FinSight eval: rules vs Mistral vs Laya")
    ap.add_argument("--setup", default="all", help="A|B|C|D|E|all")
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--dry-run", action="store_true", help="FakeLaya + FakeLLM, no network")
    ap.add_argument("--threshold", type=float, default=0.7, help="cascade (D) confidence gate")
    args = ap.parse_args()
    setups = SETUPS if args.setup == "all" else [args.setup.upper()]

    records = evaluate(setups, args.split, args.dry_run, args.threshold)
    summary = summarize(records, setups)
    outdir = write_outputs(records, summary, setups, args.split, args.threshold, args.dry_run)

    print(f"setups={setups} split={args.split} dry_run={args.dry_run}")
    for s in setups:
        m = summary[s]
        print(f"  {s}: acc={m['acc_all']:.2f} CI[{m['acc_ci'][0]:.2f},{m['acc_ci'][1]:.2f}] "
              f"intentF1={m['intent_f1']:.2f} sentAcc={m['sent_acc']:.2f} "
              f"valid={m['valid_rate']:.2f} lat={m['mean_latency_ms']:.0f}ms")
    print(f"wrote {outdir}/report.md + results.csv + chart")


if __name__ == "__main__":
    main()
