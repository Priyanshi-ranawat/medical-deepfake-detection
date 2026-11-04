"""Unit tests for the metric and threshold logic in src.train.

These functions decide the numbers that go in the results table, so they are
tested directly with hand-computable inputs (no data, no model, no GPU). The
key invariant for the study is that thresholds are chosen on validation only;
``choose_thresholds`` is the function that enforces it, so it gets the most
attention here.
"""
import math

import numpy as np
import pytest

from src.train import auc, choose_thresholds, metrics_at


# --------------------------------------------------------------------- metrics_at
def test_metrics_at_perfect_separation():
    y = [0, 0, 1, 1]
    prob = [0.1, 0.2, 0.8, 0.9]
    m = metrics_at(y, prob, 0.5)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 2, 0, 0)
    for k in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1"):
        assert m[k] == pytest.approx(1.0)
    assert m["threshold"] == 0.5


def test_metrics_at_one_error_each_way():
    # pred = [0, 1, 0, 1] -> one FP and one FN
    m = metrics_at([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 1, 1, 1)
    assert m["recall"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx(0.5)


def test_metrics_at_threshold_is_inclusive_lower_bound():
    # prob == thr must count as a positive prediction (prob >= thr).
    m = metrics_at([0, 1], [0.5, 0.5], 0.5)
    assert m["fp"] == 1 and m["tp"] == 1
    assert m["fn"] == 0 and m["tn"] == 0


def test_metrics_at_no_positive_predictions_uses_guards():
    # Threshold above every score -> nothing predicted positive.
    m = metrics_at([0, 1], [0.1, 0.2], 0.9)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (0, 1, 0, 1)
    assert m["recall"] == pytest.approx(0.0)
    assert m["precision"] == 0.0          # tp + fp == 0 -> guarded to 0.0
    assert m["f1"] == 0.0                 # prec + recall == 0 -> guarded to 0.0
    assert m["specificity"] == pytest.approx(1.0)


def test_metrics_at_specificity_nan_when_no_true_negatives_possible():
    # Only positives present: tn + fp == 0 -> specificity is NaN, recall defined.
    m = metrics_at([1, 1], [0.6, 0.9], 0.5)
    assert math.isnan(m["specificity"])
    assert m["recall"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)


def test_metrics_at_counts_are_ints_and_sum_to_n():
    m = metrics_at([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    for k in ("tp", "tn", "fp", "fn"):
        assert isinstance(m[k], int)
    assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == 4


# ------------------------------------------------------------------ choose_thresholds
def test_choose_thresholds_keys_match_consumers():
    # main() and summarize_runs index these exact keys.
    t = choose_thresholds([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.95)
    assert set(t) == {"default_0.5", "youden_val", "recall0.95_val"}
    assert t["default_0.5"] == 0.5


def test_choose_thresholds_youden_on_separable_data():
    # Any threshold in (0.2, 0.8] separates perfectly; Youden picks 0.8.
    t = choose_thresholds([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.95)
    assert t["youden_val"] == pytest.approx(0.8)


def test_choose_thresholds_high_recall_picks_highest_threshold_meeting_target():
    # thr=0.9 already reaches recall 0.5; being the highest such threshold, it wins.
    t = choose_thresholds([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert t["recall0.5_val"] == pytest.approx(0.9)


def test_choose_thresholds_recall_key_is_formatted_with_g():
    assert "recall0.9_val" in choose_thresholds([0, 1], [0.2, 0.8], 0.9)
    assert "recall1_val" in choose_thresholds([0, 1], [0.2, 0.8], 1.0)


def test_choose_thresholds_are_finite_probabilities():
    # roc_curve's leading +inf must be clipped so thresholds stay in [0, 1].
    t = choose_thresholds([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.95)
    for v in t.values():
        assert 0.0 <= v <= 1.0
        assert np.isfinite(v)


def test_choose_thresholds_unreachable_recall_falls_back_to_zero():
    # No achievable recall reaches the (impossible) target -> 0.0 (predict all positive).
    t = choose_thresholds([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0001)
    assert t["recall1.0001_val"] == 0.0


# ----------------------------------------------------------------------------- auc
def test_auc_perfect_and_reversed():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_auc_chance_level():
    assert auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auc_single_class_is_nan():
    assert math.isnan(auc([1, 1, 1], [0.2, 0.5, 0.9]))
    assert math.isnan(auc([0, 0], [0.1, 0.4]))
