"""Pooled thresholds and the per-language fallback (plan 6.5 item 7)."""
import numpy as np
from sklearn.metrics import f1_score

from emoldgn.thresholds import apply_thresholds, best_t, make_grid, thresholds_for, tune_thresholds


def data():
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random((n, 2)) < 0.3).astype(int)
    p = np.clip(y * 0.5 + rng.random((n, 2)) * 0.5, 0, 1)
    lang = np.array(["big"] * 380 + ["small"] * 20)
    return p, y, np.ones_like(y, dtype=float), lang


def test_vectorised_search_matches_the_sklearn_formula():
    p, y, _, _ = data()
    grid = make_grid(0.01)
    plan = grid[int(np.argmax([f1_score(y[:, 0], p[:, 0] >= t, zero_division=0) for t in grid]))]
    assert best_t(p[:, 0], y[:, 0], grid) == plan


def test_language_with_few_positives_uses_global():
    p, y, m, lang = data()
    t_global, t_lang = tune_thresholds(p, y, m, lang, 0.01, 30)
    assert y[lang == "small"].sum(0).max() < 30
    assert np.array_equal(t_lang["small"], t_global)
    assert y[lang == "big"].sum(0).min() >= 30


def test_language_with_enough_positives_is_tuned_alone():
    rng = np.random.default_rng(1)
    y = np.zeros((400, 1), dtype=int)
    y[::2] = 1
    lang = np.array(["a"] * 200 + ["b"] * 200)
    # Language a separates near 0.25, language b near 0.65.
    p = np.where(y == 1, 0.3 + rng.random((400, 1)) * 0.1, rng.random((400, 1)) * 0.2)
    p[200:] += 0.4
    t_global, t_lang = tune_thresholds(p, y, np.ones_like(y, dtype=float), lang, 0.01, 30)
    assert t_lang["a"][0] <= 0.3 and t_lang["b"][0] >= 0.59
    assert t_lang["a"][0] != t_lang["b"][0]


def test_unseen_language_uses_global():
    t_global = np.array([0.4, 0.6])
    out = thresholds_for(["never_seen"], t_global, {"known": np.array([0.1, 0.2])})
    assert np.array_equal(out[0], t_global)


def test_masked_rows_do_not_count():
    p = np.array([[0.9], [0.1]] * 40)
    y = np.array([[1], [0]] * 40)
    m = np.zeros_like(p)
    t_global, _ = tune_thresholds(p, y, m, ["a"] * 80, 0.01, 30)
    assert t_global[0] == 0.5


def test_neutral_override_gives_an_all_zero_row():
    p = np.array([[0.7, 0.1], [0.7, 0.1]])
    t = np.full((2, 2), 0.5)
    pred = apply_thresholds(p, t, np.ones((2, 2)), np.array([0.9, 0.1]))
    assert pred.tolist() == [[0, 0], [1, 0]]
