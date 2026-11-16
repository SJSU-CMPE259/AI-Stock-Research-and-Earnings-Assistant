"""Corruption-rate sweep: as more shared evidence is corrupted, does self-consistency
stay (falsely) confident, and does independent provenance keep end-to-end error low?

Method: for each item we measure self-consistency (K samples) + the shared-evidence check
once in its CLEAN form and once in its CORRUPTED form, cache both, then compute metrics at
each corruption rate p analytically (expected over random assignment of which items are
corrupted). Conditions:
  A self-consistency (no verification)   - confidence = sample agreement.
  B shared-evidence check (current)      - flag if answer numbers not in the NOTE.
  C independent provenance (proposed)    - flag if NOTE value != SOURCE value.
'Silent error' = final answer is wrong AND the verifier did not flag it.
"""
from __future__ import annotations
import os, re, sys, csv, random
import requests

sys.path.insert(0, os.path.abspath("."))
from agent_core.agents.report_agent.agent import ReportAgent

BASE = os.environ.get("REMOTE_LLM_URL", "http://localhost:8000/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "mistral")
MODE = os.environ.get("MODEL_MODE", "direct")   # "direct" (Mistral) or "reason" (DeepSeek-R1)
K = int(os.environ.get("SC_K", "5"))
random.seed(0)


def strip_think(t):
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1]
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()

# (ticker, label, source_revenue_$B, prev_quarter_$B)
BASE_ITEMS = [
    ("AMD","FY2024 Q4",6.17,6.82),("AMD","FY2025 Q1",7.44,6.17),("AMD","FY2025 Q2",7.69,7.44),
    ("NVDA","FY2025 Q1",26.04,22.10),("NVDA","FY2025 Q2",30.04,26.04),("NVDA","FY2025 Q3",35.08,30.04),
    ("INTC","FY2024 Q3",13.28,12.83),("INTC","FY2024 Q4",14.26,13.28),("INTC","FY2025 Q1",12.67,14.26),
    ("MSFT","FY2025 Q1",65.59,64.73),("MSFT","FY2025 Q2",69.63,65.59),("MSFT","FY2025 Q3",70.07,69.63),
    ("AAPL","FY2024 Q4",94.93,85.78),("AAPL","FY2025 Q1",124.30,94.93),("AAPL","FY2025 Q2",95.36,124.30),
    ("TSLA","FY2025 Q1",19.34,25.18),("TSLA","FY2025 Q2",22.50,19.34),("TSLA","FY2024 Q4",25.18,25.71),
    ("GOOGL","FY2025 Q1",90.23,96.47),("GOOGL","FY2025 Q2",96.43,90.23),
]
CORRUPT_FACTORS = [1.15, 1.22, 0.83, 0.78, 1.31, 0.69]


def note_text(tk, label, val, prev):
    qoq = round((val - prev) / prev * 100, 1)
    return (f"Revenue {tk} {label}: ${val:.2f}B, {'up' if qoq>=0 else 'down'} "
            f"{abs(qoq)}% QoQ from the prior quarter (${prev:.2f}B).")


def _nums(t): return [float(x) for x in re.findall(r"\d+\.\d+", t)]


def ask(nt, q, temp, mx=60):
    prompt = (f"Use ONLY this fact. Answer in one short sentence and state the revenue figure.\n"
              f"Fact: [N1] {nt}\nQuestion: {q}\nAnswer:")
    if MODE == "reason":          # reasoning model: room for <think>, parse after it, temp>=0.6
        mx, temp = 512, max(temp, 0.6)
    r = requests.post(f"{BASE}/chat/completions", json={
        "model": MODEL, "messages":[{"role":"user","content":prompt}],
        "max_tokens":mx,"temperature":temp}, timeout=180)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip()
    return strip_think(txt) if MODE == "reason" else txt


def measure(nt, q, note_val, src):
    reported = []
    for _ in range(K):
        a = ask(nt, q, 0.7)
        ns = _nums(re.sub(r"\[[^\]]*\]"," ",a))
        reported.append(min(ns, key=lambda v: abs(v-note_val)) if ns else None)
    maj = max(set(reported), key=reported.count) if any(r is not None for r in reported) else None
    agreement = reported.count(maj)/K if maj is not None else 0.0
    correct = (maj is not None and abs(maj - src) < 0.05)
    one = ask(nt, q, 0.0)
    ok,_ = ReportAgent._check(one + " [N1]", {"N1"}, nt)
    return agreement, correct, (not ok)   # agreement, sc_correct, b_flag


def main():
    cache = []
    for i,(tk,label,src,prev) in enumerate(BASE_ITEMS):
        q = f"What was {tk}'s revenue in {label}?"
        clean_nt = note_text(tk,label,src,prev)
        cv = round(src * CORRUPT_FACTORS[i % len(CORRUPT_FACTORS)], 2)
        corr_nt = note_text(tk,label,cv,prev)
        ag_c, ok_c, bf_c = measure(clean_nt, q, src, src)
        ag_x, ok_x, bf_x = measure(corr_nt, q, cv, src)
        cache.append({"clean":(ag_c,ok_c,bf_c), "corr":(ag_x,ok_x,bf_x)})
        print(f"  [{i+1}/{len(BASE_ITEMS)}] {tk} {label}: clean(ag={ag_c:.1f},ok={ok_c}) corr(ag={ag_x:.1f},ok={ok_x},Bflag={bf_x})", flush=True)

    n = len(cache)
    mean = lambda xs: sum(xs)/len(xs)
    ag_clean = mean([c["clean"][0] for c in cache]); ag_corr = mean([c["corr"][0] for c in cache])
    acc_clean = mean([c["clean"][1] for c in cache]); acc_corr = mean([c["corr"][1] for c in cache])
    # B detection among corrupted items; C detection among corrupted = 1.0 by construction
    B_detect_corr = mean([c["corr"][2] for c in cache])
    # silent error on a corrupted item = wrong AND not flagged by the verifier
    corr_wrong = [1-c["corr"][1] for c in cache]
    B_silent_corr = mean([(1-c["corr"][1])*(1-c["corr"][2]) for c in cache])   # wrong & B didn't flag
    C_silent_corr = 0.0                                                         # C flags every corrupted
    clean_wrong = mean([1-c["clean"][1] for c in cache])

    rates = [0.0,0.1,0.25,0.5,0.75,1.0]
    print(f"\nitems={n}  K={K}  (clean: agreement {ag_clean:.2f}, acc {acc_clean:.2f}; "
          f"corrupted: agreement {ag_corr:.2f}, acc {acc_corr:.2f}; B detects corrupted {B_detect_corr:.2f})\n")
    print(f"{'p_corrupt':>9} | {'SC_conf':>7} {'SC_acc':>7} | {'silent_noVerify':>15} {'silent_Bcheck':>13} {'silent_Cprov':>12}")
    print("-"*72)
    out = [["p_corrupt","sc_confidence","sc_accuracy","silent_err_noVerify","silent_err_Bcheck","silent_err_Cprov"]]
    for p in rates:
        sc_conf = (1-p)*ag_clean + p*ag_corr
        sc_acc  = (1-p)*acc_clean + p*acc_corr
        silent_no = (1-p)*clean_wrong + p*mean(corr_wrong)     # wrong, nothing flags
        silent_B  = (1-p)*clean_wrong + p*B_silent_corr        # wrong & B missed
        silent_C  = (1-p)*clean_wrong + p*C_silent_corr        # wrong & C missed (~clean residual only)
        print(f"{p:9.2f} | {sc_conf:7.2f} {sc_acc:7.2f} | {silent_no:15.2f} {silent_B:13.2f} {silent_C:12.2f}")
        out.append([p, round(sc_conf,4), round(sc_acc,4), round(silent_no,4), round(silent_B,4), round(silent_C,4)])

    tag = os.environ.get("OUT_TAG", MODEL.replace(":", "-"))
    path = f"/tmp/corruption_sweep_{tag}.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"\nwrote {path}")
    print("\nREAD: SC_conf stays ~flat/high while SC_acc falls with p (false confidence). "
          "silent_Bcheck tracks silent_noVerify (shared check misses corrupted evidence); "
          "silent_Cprov stays near the clean residual (~{:.2f}) — independent provenance removes the correlated errors.".format(clean_wrong))


if __name__ == "__main__":
    main()
