"""Pooled per-emotion thresholds with a per-language fallback (plan 6.5 item 7, section 8).

Thresholds are tuned on validation data only, after training. A language gets its own
threshold for an emotion only with at least min_pos validation positives; unseen languages
use the global threshold. Scoring is a vectorised form of sklearn's binary F1 and picks the
same grid point as the plan's snippet, including its first-maximum tie break.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def make_grid(step: float = 0.01) -> np.ndarray:
    """Candidate thresholds, plan section 8."""
    return np.arange(0.05, 0.95, step)


def best_t(pc: np.ndarray, yc: np.ndarray, grid: np.ndarray) -> float:
    """Grid threshold with the best binary F1 for one emotion."""
    pred = pc[None, :] >= grid[:, None]
    tp = (pred & (yc[None, :] == 1)).sum(1)
    denom = pred.sum(1) + (yc == 1).sum()
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(grid)), where=denom > 0)
    return float(grid[int(np.argmax(f1))])


def tune_thresholds(p: np.ndarray, y: np.ndarray, mask: np.ndarray, lang: Sequence[str], grid_step: float = 0.01,
                    min_pos: int = 30) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Tune on validation arrays (N x E). Rows where the emotion is masked do not count."""
    grid = make_grid(grid_step)
    lang = np.asarray(lang)
    n_emotions = y.shape[1]
    t_global = np.full(n_emotions, 0.5)
    for c in range(n_emotions):
        m = mask[:, c] > 0
        if m.any():
            t_global[c] = best_t(p[m, c], y[m, c], grid)
    t_lang: dict[str, np.ndarray] = {}
    for name in np.unique(lang):
        in_lang = lang == name
        t = t_global.copy()
        for c in range(n_emotions):
            m = in_lang & (mask[:, c] > 0)
            if y[m, c].sum() >= min_pos:
                t[c] = best_t(p[m, c], y[m, c], grid)
        t_lang[str(name)] = t
    return t_global, t_lang


def thresholds_for(lang: Sequence[str], t_global: np.ndarray, t_lang: dict[str, np.ndarray]) -> np.ndarray:
    """N x E thresholds; a language without its own entry uses the global ones."""
    return np.stack([np.asarray(t_lang.get(str(x), t_global)) for x in lang])


def apply_thresholds(p: np.ndarray, t: np.ndarray, mask: np.ndarray | None = None,
                     p_neutral: np.ndarray | None = None) -> np.ndarray:
    """Binary predictions (plan 6.2): neutral beating every emotion gives an all-zero row."""
    pred = (p >= t).astype(np.int64)
    if mask is not None:
        pred = pred * (mask > 0)
        p = p * (mask > 0)
    if p_neutral is not None:
        pred[p_neutral > p.max(1)] = 0
    return pred


def save_thresholds(path: str | Path, t_global: np.ndarray, t_lang: dict[str, np.ndarray], **extra) -> None:
    """Write thresholds as JSON."""
    data = {"global": t_global.tolist(), "per_language": {k: v.tolist() for k, v in t_lang.items()}, **extra}
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_thresholds(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Read thresholds written by save_thresholds."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return (np.array(data["global"]), {k: np.array(v) for k, v in data["per_language"].items()},
            {k: v for k, v in data.items() if k not in ("global", "per_language")})
