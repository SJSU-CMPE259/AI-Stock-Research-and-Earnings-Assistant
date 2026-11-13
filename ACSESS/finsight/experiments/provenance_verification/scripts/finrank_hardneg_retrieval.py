"""FinRank-grounded correlated-error sweep.

Corruption = give the model a HARD-NEGATIVE passage instead of the GOLD passage
(a realistic retrieval error). Objective number-grounding scoring:
  - a 'significant figure' = a decimal, a percent, or a >=4-digit integer (years excluded).
  - self-consistency: K samples; key number = first significant figure stated; agreement =
    majority fraction; correct = majority key number appears in the GOLD passage.
  - B shared-evidence check: do ALL answer figures appear in the RETRIEVED passage? (current)
  - C independent provenance: do ALL answer figures appear in the GOLD passage? (proposed)
  - silent error = answer not grounded in gold AND the verifier did not flag it.

Per item we measure the clean form (gold context) and corrupted form (hard-negative context)
once, cache both, then compute the corruption-rate curve analytically.

Env: REMOTE_LLM_URL, LLM_MODEL, MODEL_MODE(direct|reason), SC_K, N_ITEMS, OUT_TAG
"""
from __future__ import annotations
import os, re, sys, csv, json, random
import requests

BASE = os.environ.get("REMOTE_LLM_URL", "http://localhost:8000/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "mistral")
MODE = os.environ.get("MODEL_MODE", "direct")
K = int(os.environ.get("SC_K", "5"))
N_ITEMS = int(os.environ.get("N_ITEMS", "60"))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FinRank", "FinRank.jsonl")
random.seed(0)


def strip_think(t):
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1]
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()


def sig_nums(text: str) -> set[str]:
    """Significant financial figures: decimals/percents or >=4-digit integers (years excluded)."""
    out = set()
    for m in re.findall(r"\d[\d,]*\.\d+|\d[\d,]{3,}", text or ""):
        v = m.replace(",", "")
        if "." not in v and v.isdigit() and 1900 <= int(v) <= 2099:
            continue  # drop bare years
        out.add(v)
    return out


def norm(text: str) -> str:
    return (text or "").replace(",", "")


def ask(context: str, question: str, temp: float) -> str:
    prompt = (f"Answer the question using ONLY the context. Cite specific figures from the "
              f"context. Be concise (2-3 sentences). If the context does not contain the "
              f"answer, say 'Not stated in the provided context.'\n\n"
              f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    mx, t = (512, max(temp, 0.6)) if MODE == "reason" else (160, temp)
    r = requests.post(f"{BASE}/chat/completions", json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": mx, "temperature": t}, timeout=180)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip()
    return strip_think(txt) if MODE == "reason" else txt


def ptext(passages, k=2, cap=1400):
    return "\n\n".join(p["text"][:cap] for p in passages[:k])


def measure(context, question, gold_blob):
    ret_blob = norm(context)
    # self-consistency
    keys = []
    for _ in range(K):
        a = ask(context, question, 0.7)
        ns = sorted(sig_nums(a))  # deterministic 'first' = smallest-string; stable key
        keys.append(ns[0] if ns else None)
    maj = max(set(keys), key=keys.count) if any(k is not None for k in keys) else None
    agreement = keys.count(maj) / K if maj is not None else 0.0
    sc_correct = (maj is not None and maj in gold_blob)
    # single answer for the grounding checks
    one = ask(context, question, 0.0)
    anums = sig_nums(one)
    grounded_retrieved = bool(anums) and all(n in ret_blob for n in anums)   # B passes
    grounded_gold = bool(anums) and all(n in gold_blob for n in anums)       # C passes
    b_flag = not grounded_retrieved
    c_flag = not grounded_gold
    answer_correct = grounded_gold and bool(anums)
    return agreement, sc_correct, b_flag, c_flag, answer_correct, bool(anums)


def main():
    recs = [json.loads(l) for l in open(DATA)]
    # numeric, single clear gold set, with hard negatives available
    pool = [r for r in recs
            if r.get("reasoning_type") == "Quantitative"
            and sig_nums(r.get("answer", "")) and r.get("passages") and len(r.get("hard_negatives", [])) >= 1]
    random.shuffle(pool)
    items = pool[:N_ITEMS]
    print(f"FinRank: {len(recs)} records; {len(pool)} quantitative w/ figures; using {len(items)}  "
          f"(model={MODEL}, mode={MODE}, K={K})", flush=True)

    cache = []
    for i, r in enumerate(items):
        gold_ctx = ptext(r["passages"])
        gold_blob = norm(gold_ctx)
        hn_ctx = ptext(r["hard_negatives"])
        agc, okc, bfc, cfc, acc_c, hasn_c = measure(gold_ctx, r["question"], gold_blob)
        agx, okx, bfx, cfx, acc_x, hasn_x = measure(hn_ctx, r["question"], gold_blob)
        cache.append({"clean": (agc, okc, bfc, cfc, acc_c, hasn_c),
                      "corr": (agx, okx, bfx, cfx, acc_x, hasn_x)})
        print(f"  [{i+1}/{len(items)}] {r['ticker']} {r['topic']}: "
              f"clean(ag={agc:.1f},gold={okc},ans={hasn_c}) corr(ag={agx:.1f},gold={okx},Bflag={bfx},Cflag={cfx},ans={hasn_x})",
              flush=True)

    n = len(cache)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    ag_clean = mean([c["clean"][0] for c in cache]); ag_corr = mean([c["corr"][0] for c in cache])
    acc_clean = mean([c["clean"][4] for c in cache]); acc_corr = mean([c["corr"][4] for c in cache])
    B_detect_corr = mean([c["corr"][2] for c in cache])
    C_detect_corr = mean([c["corr"][3] for c in cache])
    answered_corr = mean([c["corr"][5] for c in cache])    # did it answer (vs abstain) on a hard negative
    clean_wrong = mean([1 - c["clean"][4] for c in cache])
    # silent error on a corrupted item = answer not grounded-in-gold AND verifier didn't flag
    B_silent_corr = mean([(1 - c["corr"][4]) * (1 - c["corr"][2]) for c in cache])
    C_silent_corr = mean([(1 - c["corr"][4]) * (1 - c["corr"][3]) for c in cache])

    print(f"\nSUMMARY (model={MODEL}): clean[agree {ag_clean:.2f}, ans-grounded {acc_clean:.2f}]  "
          f"corrupted[agree {ag_corr:.2f}, ans-grounded {acc_corr:.2f}, answered(no-abstain) {answered_corr:.2f}]")
    print(f"  on corrupted: B flags {B_detect_corr:.2f}, C flags {C_detect_corr:.2f}; "
          f"silent-error B {B_silent_corr:.2f}, C {C_silent_corr:.2f}\n")

    rates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    print(f"{'p_corrupt':>9} | {'SC_conf':>7} {'SC_acc':>7} | {'silent_noVerify':>15} {'silent_Bcheck':>13} {'silent_Cprov':>12}")
    print("-" * 72)
    out = [["p_corrupt", "sc_confidence", "sc_accuracy", "silent_err_noVerify", "silent_err_Bcheck", "silent_err_Cprov"]]
    for p in rates:
        sc_conf = (1 - p) * ag_clean + p * ag_corr
        sc_acc = (1 - p) * acc_clean + p * acc_corr
        silent_no = (1 - p) * clean_wrong + p * mean([1 - c["corr"][4] for c in cache])
        silent_B = (1 - p) * clean_wrong + p * B_silent_corr
        silent_C = (1 - p) * clean_wrong + p * C_silent_corr
        print(f"{p:9.2f} | {sc_conf:7.2f} {sc_acc:7.2f} | {silent_no:15.2f} {silent_B:13.2f} {silent_C:12.2f}")
        out.append([p, round(sc_conf, 4), round(sc_acc, 4), round(silent_no, 4), round(silent_B, 4), round(silent_C, 4)])

    tag = os.environ.get("OUT_TAG", MODEL.replace(":", "-"))
    path = f"/tmp/finrank_sweep_{tag}.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
