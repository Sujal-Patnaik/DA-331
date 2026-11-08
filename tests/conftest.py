"""Shared fixtures: a Spark session and a tiny synthetic training setup on CPU.

Nothing here reads the official test split. Tiny data is synthetic, so CPU tests stay fast.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spark_jobs.common import load_config  # noqa: E402

LANGS = ["eng", "hau", "<unk>"]


class FakeEncoder(torch.nn.Module):
    """Stands in for XLM-R where only the head matters (no download, any hidden size)."""

    def __init__(self, hidden: int = 768) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden})()
        self.emb = torch.nn.Embedding(64, hidden)

    def forward(self, input_ids, attention_mask):  # noqa: D401, ANN001
        return type("Out", (), {"last_hidden_state": self.emb(input_ids)})()


def fake_bank(n_lang: int = len(LANGS)) -> dict:
    """A valid random bank; English has no disgust."""
    torch.manual_seed(0)
    c = 7
    npmi = torch.rand(n_lang, c, c) * 2 - 1
    npmi = (npmi + npmi.transpose(1, 2)) / 2
    mask = torch.ones(n_lang, c)
    mask[0, 1] = 0
    pair = mask.unsqueeze(1) * mask.unsqueeze(2)
    off = 1 - torch.eye(c)
    a_pos = (npmi - 0.05).clamp(min=0) * off * pair
    a_neg = (-npmi - 0.1).clamp(min=0) * off * pair
    cond = torch.rand(n_lang, c, c) * pair
    bank = {"mask": mask, "languages": LANGS[:n_lang] if n_lang <= 3 else [f"l{i}" for i in range(n_lang)],
            "nodes": ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"], "family_index": torch.zeros(n_lang).long()}
    for suffix in ("raw", "prior"):
        bank[f"cond_{suffix}"] = cond.clone()
        bank[f"npmi_{suffix}"] = npmi * pair
        bank[f"A_pos_{suffix}"] = a_pos.clone()
        bank[f"A_neg_{suffix}"] = a_neg.clone()
    return bank


def tiny_rows(n: int = 48, seed: int = 0) -> pd.DataFrame:
    """Synthetic tokenised rows: the first token of a row decides its labels."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        lang = "eng" if i % 2 == 0 else "hau"
        label = int(rng.integers(0, 6)) if lang == "hau" else int(rng.choice([0, 2, 3, 4, 5]))
        length = int(rng.integers(5, 12))
        ids = [0, 10 + label] + rng.integers(20, 60, length - 3).tolist() + [2]
        labels = [0] * 6
        labels[label] = 1
        mask = [1, 0, 1, 1, 1, 1] if lang == "eng" else [1] * 6
        inten = [-1] * 6
        if lang == "eng":
            inten = [0, -1, 0, 0, 0, 0]
            inten[label] = int(rng.integers(1, 4))
        part = ("test" if i % 9 == 0 else "val") if i % 3 == 0 else "train"
        rows.append({"id": f"{lang}_{i:04d}", "lang": lang, "part": part, "track": "A", "split": "train", "input_ids": ids,
                     "attention_mask": [1] * length, "labels": labels, "label_mask": mask, "neutral": 0, "intensity": inten,
                     "length": length})
    return pd.DataFrame(rows)


@pytest.fixture()
def tiny_cfg(tmp_path: Path) -> dict:
    """Resolved run configuration pointing at a temporary gold folder."""
    cfg = load_config()
    gold = tmp_path / "gold"
    gold.mkdir()
    tiny_rows().to_parquet(gold / "tokenized")
    torch.save(fake_bank(), gold / "graph_bank.pt")
    cfg["paths"]["gold"] = str(gold)
    cfg["paths"]["runs"] = str(tmp_path / "runs")
    cfg.update(tiny_encoder=True, force_cpu=True, fp16=False, batch_size=4, grad_accum=1, eval_batch_size=8,
               freeze_word_embeddings=False, gcn_hidden=16, lr_head=1e-2, lr_encoder=1e-3, subset_langs=["eng", "hau"],
               max_len=16)
    with open(ROOT / "config/experiments.yaml", encoding="utf-8") as f:
        entries = yaml.safe_load(f)["experiments"]
    cfg["exp"] = copy.deepcopy(next(e for e in entries if e["id"] == "final_subset_s42"))
    cfg["epochs"]["subset"] = {"max": 2, "patience": 3}
    return cfg


@pytest.fixture(scope="session")
def spark():
    from spark_jobs.common import get_spark

    s = get_spark("tests", master="local[2]", extra={"spark.ui.enabled": "false", "spark.sql.shuffle.partitions": "4"})
    yield s
    s.stop()


def all_flags() -> list[dict]:
    """Distinct flag sets of experiments.yaml."""
    with open(ROOT / "config/experiments.yaml", encoding="utf-8") as f:
        entries = yaml.safe_load(f)["experiments"]
    seen, out = set(), []
    for e in entries:
        key = tuple(sorted(e["flags"].items()))
        if key not in seen:
            seen.add(key)
            out.append(e["flags"])
    return out
