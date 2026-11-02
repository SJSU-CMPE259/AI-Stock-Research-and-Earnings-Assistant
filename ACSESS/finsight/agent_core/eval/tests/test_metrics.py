"""Known-answer unit tests for eval/metrics.py."""
import math

from agent_core.eval import metrics as M


def test_accuracy():
    assert M.accuracy(["a", "b", "c"], ["a", "x", "c"]) == 2 / 3
    assert M.accuracy(["a"], [None]) == 0.0
    assert M.accuracy([], []) == 0.0


def test_confusion_and_f1():
    yt = ["pos", "pos", "neg", "neg"]
    yp = ["pos", "neg", "neg", "neg"]
    cm = M.confusion_matrix(yt, yp, ["pos", "neg"])
    assert cm.tolist() == [[1, 1], [0, 2]]
    f1 = M.per_class_f1(yt, yp, ["pos", "neg"])
    assert round(f1["pos"], 3) == round(2 * 1 * 0.5 / 1.5, 3)  # prec 1, rec .5
    assert round(f1["neg"], 3) == round(2 * (2 / 3) * 1 / (2 / 3 + 1), 3)
    assert 0 <= M.macro_f1(yt, yp, ["pos", "neg"]) <= 1


def test_advice_metrics():
    t = [True, True, False, False]
    p = [True, False, False, True]
    assert M.advice_recall(t, p) == 0.5          # 1 of 2 advice caught
    assert M.false_refusal_rate(t, p) == 0.5     # 1 of 2 non-advice refused


def test_coverage_accuracy_curve():
    correct = [True, True, False]
    conf = [0.9, 0.6, 0.4]
    rows = M.coverage_accuracy_curve(correct, conf, [0.5, 0.95])
    assert rows[0]["coverage"] == 2 / 3 and rows[0]["accuracy"] == 1.0  # th .5 -> 2 answered, both right
    assert rows[1]["coverage"] == 0.0 and math.isnan(rows[1]["accuracy"])


def test_ece_perfect_is_zero():
    # confidence exactly matches outcome within each singleton bin
    assert M.expected_calibration_error([True, False], [1.0, 0.0], n_bins=10) == 0.0


def test_reliability_table_bins():
    rows = M.reliability_table([True, False], [0.95, 0.05], n_bins=10)
    assert sum(r["n"] for r in rows) == 2


def test_recall_and_mrr():
    ranked = [["a", "b", "c"], ["x", "y"]]
    relevant = [{"b"}, {"x"}]
    assert M.recall_at_k(ranked, relevant, 2) == 1.0
    assert M.mrr(ranked, relevant) == (0.5 + 1.0) / 2


def test_cohen_kappa():
    assert round(M.cohen_kappa(["a", "b", "a"], ["a", "b", "a"]), 6) == 1.0
    assert abs(M.cohen_kappa(["a", "b"], ["b", "a"])) <= 1.0


def test_bootstrap_ci_bounds():
    lo, hi = M.bootstrap_ci([1, 1, 1, 1], n_resamples=200)
    assert lo == 1.0 and hi == 1.0
    lo, hi = M.bootstrap_ci([0, 1] * 50, n_resamples=500)
    assert 0.3 < lo < 0.5 < hi < 0.7


def test_mcnemar():
    r = M.mcnemar([True, True, False], [True, False, True])
    assert r["b01"] == 1 and r["b10"] == 1
    assert 0.0 <= r["p_value"] <= 1.0
    assert M.mcnemar([True], [True])["p_value"] == 1.0


def test_valid_output_rate():
    assert M.valid_output_rate([True, True, False, True]) == 0.75
