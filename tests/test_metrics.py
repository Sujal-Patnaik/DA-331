"""Metrics against sklearn and scipy (the official scorer uses both)."""
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

from emoldgn import evaluate as E
from emoldgn.thresholds import apply_thresholds


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (200, 5))
    pred = rng.integers(0, 2, (200, 5))
    pred[:, 4] = 0  # an emotion never predicted
    assert abs(E.macro_f1(y, pred, np.ones(5)) - f1_score(y, pred, average="macro", zero_division=0)) < 1e-12


def test_pearson_matches_scipy():
    rng = np.random.default_rng(1)
    gold = rng.integers(0, 4, (100, 6))
    pred = gold + rng.integers(-1, 2, (100, 6))
    out = E.track_b(gold, pred, ["a"] * 100)
    want = np.mean([stats.pearsonr(gold[:, c], pred[:, c])[0] for c in range(6)])
    assert abs(out["mean"] - want) < 1e-12


def test_pearson_skips_missing_intensity_and_constant_columns():
    gold = np.array([[1, -1, 0, 0, 0, 0], [2, -1, 0, 0, 0, 0], [3, -1, 0, 0, 0, 0]])
    out = E.track_b(gold, gold.copy(), ["a"] * 3)
    assert "disgust" not in out["per_emotion"]["a"]
    assert abs(out["per_emotion"]["a"]["anger"] - 1) < 1e-9 and out["per_emotion"]["a"]["fear"] == 0


def test_all_neutral_prediction_gives_all_zero_track_a_row():
    p = np.full((3, 6), 0.9)
    pred = apply_thresholds(p, np.full((3, 6), 0.5), np.ones((3, 6)), np.full(3, 0.99))
    assert pred.sum() == 0


def test_language_scores_are_averaged_per_language():
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    pred = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    out = E.track_a(y, pred, np.ones_like(y), ["a", "a", "b", "b"])
    assert out["per_language"] == {"a": 1.0, "b": 0.0}
    assert out["mean"] == 0.5


def test_cooccurrence_fidelity_is_zero_for_a_perfect_model():
    y = np.random.default_rng(0).integers(0, 2, (50, 6))
    assert E.cooccurrence_fidelity(y, y, np.ones_like(y), ["a"] * 50) == 0


def test_ece_of_a_calibrated_constant():
    y = np.array([[1], [0], [0], [0]])
    assert E.ece(np.full((4, 1), 0.25), y, np.ones((4, 1))) < 1e-9
    assert abs(E.ece(np.full((4, 1), 0.75), y, np.ones((4, 1))) - 0.5) < 1e-9


def test_paired_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (300, 3))
    noisy = np.where(rng.random(y.shape) < 0.4, 1 - y, y)
    out = E.paired_bootstrap(y, y, noisy, np.ones_like(y), ["a"] * 300, resamples=200)
    assert out["mean_diff"] > 0 and out["p_value"] == 0


def test_intensity_confusion_shape():
    gold = np.array([[0, -1, 1, 2, 3, 0]])
    cm = E.intensity_confusion(gold, gold.clip(0))
    assert np.array(cm["fear"]).shape == (4, 4)
    assert np.array(cm["disgust"]).sum() == 0
