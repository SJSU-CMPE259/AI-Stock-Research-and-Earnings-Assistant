"""Paired McNemar: DeepSeek-R1-8b vs Mistral-7B on the same 100 intent+sentiment items."""
import os, sys
SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
sys.path.insert(0, os.path.abspath("."))  # finsight/, for agent_core
import r1_eval as R
from agent_core.eval import metrics as M
from agent_core.eval.data import load_csv, select_split

routing = select_split(load_csv("routing_questions.csv"), "all", 0)
sentiment = select_split(load_csv("sentiment_statements.csv"), "all", 0)


def correctness(model, mode):
    c = []
    for row in routing:
        w, _, _ = R.decide_choice(model, mode, row["question"],
                                  "Classify the user's request.", R.ROUTING_OPTIONS)
        c.append(w == row["gold_intent"])
    for row in sentiment:
        w, _, _ = R.decide_choice(model, mode, row["statement"],
                                  "Classify the sentiment of the statement.", R.SENTIMENT_OPTIONS)
        c.append(w == row["gold_label"])
    return c


cm = correctness("mistral", "direct")
cr = correctness("deepseek-r1:8b", "reason")
res = M.mcnemar(cr, cm)  # (correct_a=R1, correct_b=Mistral)
b = sum(1 for a, bb in zip(cr, cm) if a and not bb)   # R1 right, Mistral wrong
c = sum(1 for a, bb in zip(cr, cm) if not a and bb)   # Mistral right, R1 wrong
print(f"n items: {len(cm)}")
print(f"Mistral correct: {sum(cm)}/{len(cm)}  ({sum(cm)/len(cm):.4f})")
print(f"R1 correct:      {sum(cr)}/{len(cr)}  ({sum(cr)/len(cr):.4f})")
print(f"discordant: R1-right/Mistral-wrong = {b}, Mistral-right/R1-wrong = {c}")
print(f"McNemar result: {res}")
