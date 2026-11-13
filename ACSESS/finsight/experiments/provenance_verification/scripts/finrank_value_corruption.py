"""FinRank value-corruption sweep (the professor's hypothesis on real data).

For each item we pick a TARGET figure that appears both in the reference answer and in a
gold passage (so ground truth is known and verbatim-extractable). We build:
  - clean context  = a window of the real gold passage around the target.
  - corrupt context = same window with the target figure replaced by a plausible WRONG value
    (simulating 'the agents share the same erroneous evidence').
Conditions (number-grounding, objective):
  - self-consistency: K samples; the model's figure = sig-num closest to the figure in context;
    agreement = majority fraction; correct = majority figure == TRUE target.
  - B shared-evidence check: is the answer's figure in the RETRIEVED context? (current FinSight)
  - C independent provenance: is it in the ORIGINAL (uncorrupted) gold? (proposed)
  - silent error = answer not equal to the true target AND the verifier did not flag it.
Per item measured once clean + once corrupted, cached; corruption-rate curve computed analytically.

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


_SIG = re.compile(r"\d[\d,]*\.\d+|\d[\d,]{3,}")


def sig_pairs(text):
    """[(normalized_value_str, raw_match)]; drops bare years."""
    out = []
    for m in _SIG.findall(text or ""):
        v = m.replace(",", "")
        if "." not in v and v.isdigit() and 1900 <= int(v) <= 2099:
            continue
        out.append((v, m))
    return out


def norm(t):
    return (t or "").replace(",", "")


def to_f(s):
    try:
        return float(s)
    except Exception:
        return None


def ask(context, question, temp):
    prompt = (f"Answer the question using ONLY the context. State the specific figure from the "
              f"context. Be concise (one sentence).\n\nContext:\n{context}\n\n"
              f"Question: {question}\nAnswer:")
    mx, t = (512, max(temp, 0.6)) if MODE == "reason" else (120, temp)
    r = requests.post(f"{BASE}/chat/completions", json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": mx, "temperature": t}, timeout=180)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip()
    return strip_think(txt) if MODE == "reason" else txt


def closest(answer, expected):
    """The figure in the answer closest to `expected` (the value the model should have read)."""
    cands = [(v, to_f(v)) for v, _ in sig_pairs(answer)]
    cands = [(v, f) for v, f in cands if f is not None]
    if not cands:
        return None
    return min(cands, key=lambda c: abs(c[1] - expected))[0]


def window(text, raw, pad=1000):
    i = text.find(raw)
    if i < 0:
        return text[:2 * pad]
    return text[max(0, i - pad): i + len(raw) + pad]


def build_item(r):
    gold_text = "\n\n".join(p["text"] for p in r["passages"])
    goldn = norm(gold_text)
    target = next((v for v, _ in sig_pairs(r["answer"]) if v in goldn), None)
    if target is None:
        return None
    raw = next((m for v, m in sig_pairs(gold_text) if v == target), target)
    ctx = window(gold_text, raw)
    if target not in norm(ctx):
        return None
    tf = to_f(target)
    if tf is None or tf == 0:
        return None
    cf = round(tf * random.choice([1.18, 1.27, 0.82, 0.73]), 2 if "." in target else 0)
    cstr = f"{cf:.2f}" if "." in target else str(int(cf))
    corrupt_ctx = ctx.replace(raw, cstr, 1)
    if cstr not in norm(corrupt_ctx):
        return None
    return {"q": r["question"], "ctx": ctx, "corrupt_ctx": corrupt_ctx,
            "target": target, "tf": tf, "cstr": cstr, "cf": float(cf),
            "true_blob": norm(ctx), "ticker": r["ticker"], "topic": r["topic"]}


def measure(context, question, expected, true_blob):
    ret_blob = norm(context)
    keys = []
    for _ in range(K):
        keys.append(closest(ask(context, question, 0.7), expected))
    maj = max(set(keys), key=keys.count) if any(k is not None for k in keys) else None
    agreement = keys.count(maj) / K if maj is not None else 0.0
    one_fig = closest(ask(context, question, 0.0), expected)
    b_flag = not (one_fig is not None and one_fig in ret_blob)     # in retrieved?
    c_flag = not (one_fig is not None and one_fig in true_blob)    # in original gold?
    return agreement, maj, one_fig, b_flag, c_flag


def main():
    recs = [json.loads(l) for l in open(DATA)]
    random.shuffle(recs)
    items = []
    for r in recs:
        if r.get("passages") and sig_pairs(r.get("answer", "")):
            it = build_item(r)
            if it:
                items.append(it)
        if len(items) >= N_ITEMS:
            break
    print(f"FinRank value-corruption: using {len(items)} items (model={MODEL}, mode={MODE}, K={K})", flush=True)

    cache = []
    for i, it in enumerate(items):
        tf, cf, tgt = it["tf"], it["cf"], it["target"]
        ag_c, maj_c, fig_c, bf_c, cf_c = measure(it["ctx"], it["q"], tf, it["true_blob"])
        ag_x, maj_x, fig_x, bf_x, cf_x = measure(it["corrupt_ctx"], it["q"], cf, it["true_blob"])
        correct_c = (maj_c is not None and to_f(maj_c) is not None and abs(to_f(maj_c) - tf) < 1e-6)
        correct_x = (maj_x is not None and to_f(maj_x) is not None and abs(to_f(maj_x) - tf) < 1e-6)
        cache.append({"clean": (ag_c, correct_c, bf_c, cf_c),
                      "corr": (ag_x, correct_x, bf_x, cf_x)})
        print(f"  [{i+1}/{len(items)}] {it['ticker']} {it['topic']} target={tgt}->{it['cstr']}: "
              f"clean(ag={ag_c:.1f},ok={correct_c}) corr(ag={ag_x:.1f},ok={correct_x},Bflag={bf_x},Cflag={cf_x})", flush=True)

    n = len(cache)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    ag_clean = mean([c["clean"][0] for c in cache]); ag_corr = mean([c["corr"][0] for c in cache])
    acc_clean = mean([c["clean"][1] for c in cache]); acc_corr = mean([c["corr"][1] for c in cache])
    B_detect = mean([c["corr"][2] for c in cache]); C_detect = mean([c["corr"][3] for c in cache])
    clean_wrong = mean([1 - c["clean"][1] for c in cache])
    B_silent = mean([(1 - c["corr"][1]) * (1 - c["corr"][2]) for c in cache])
    C_silent = mean([(1 - c["corr"][1]) * (1 - c["corr"][3]) for c in cache])
    print(f"\nSUMMARY ({MODEL}): clean[agree {ag_clean:.2f}, acc {acc_clean:.2f}]  "
          f"corrupt[agree {ag_corr:.2f}, acc {acc_corr:.2f}]  "
          f"on-corrupt: B-flags {B_detect:.2f} C-flags {C_detect:.2f}; silent B {B_silent:.2f} C {C_silent:.2f}\n")

    # --- statistics on the corrupted set: detection rate CIs + paired McNemar (C vs B) ---
    try:
        sys.path.insert(0, os.path.abspath("."))
        from agent_core.eval import metrics as M
        b_det = [c["corr"][2] for c in cache]   # B flagged (1/0)
        c_det = [c["corr"][3] for c in cache]   # C flagged (1/0)
        ci_b = M.bootstrap_ci([int(x) for x in b_det], n_resamples=2000)
        ci_c = M.bootstrap_ci([int(x) for x in c_det], n_resamples=2000)
        mc = M.mcnemar([bool(x) for x in c_det], [bool(x) for x in b_det])
        b10 = sum(1 for cc, bb in zip(c_det, b_det) if cc and not bb)  # C caught, B missed
        b01 = sum(1 for cc, bb in zip(c_det, b_det) if bb and not cc)  # B caught, C missed
        print(f"STATS (corrupted n={n}): "
              f"B detect {B_detect:.2f} [{ci_b[0]:.2f},{ci_b[1]:.2f}], "
              f"C detect {C_detect:.2f} [{ci_c[0]:.2f},{ci_c[1]:.2f}]")
        print(f"  paired McNemar C vs B: C-caught/B-missed={b10}, B-caught/C-missed={b01}, p={mc.get('p_value'):.4g}\n")
    except Exception as e:
        print(f"(stats skipped: {e})\n")

    rates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    print(f"{'p':>5} | {'SC_conf':>7} {'SC_acc':>7} | {'silent_no':>9} {'silent_B':>9} {'silent_C':>9}")
    print("-" * 60)
    out = [["p_corrupt", "sc_confidence", "sc_accuracy", "silent_err_noVerify", "silent_err_Bcheck", "silent_err_Cprov"]]
    for p in rates:
        sc_conf = (1 - p) * ag_clean + p * ag_corr
        sc_acc = (1 - p) * acc_clean + p * acc_corr
        silent_no = (1 - p) * clean_wrong + p * mean([1 - c["corr"][1] for c in cache])
        silent_B = (1 - p) * clean_wrong + p * B_silent
        silent_C = (1 - p) * clean_wrong + p * C_silent
        print(f"{p:5.2f} | {sc_conf:7.2f} {sc_acc:7.2f} | {silent_no:9.2f} {silent_B:9.2f} {silent_C:9.2f}")
        out.append([p, round(sc_conf, 4), round(sc_acc, 4), round(silent_no, 4), round(silent_B, 4), round(silent_C, 4)])

    tag = os.environ.get("OUT_TAG", MODEL.replace(":", "-"))
    path = f"/tmp/finrank_vc_{tag}.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
