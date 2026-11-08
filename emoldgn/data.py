"""Training data access: tokenised Parquet, temperature sampling, length grouping (plan 6.5)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

NUM_NODES = 7


def intensity_quota(batch_size: int, wanted: int) -> int:
    """Intensity rows per micro-batch: the wanted floor, capped at half the micro-batch.

    The configured floor of 32 cannot fit a 16-row micro-batch; half keeps other
    languages in every batch while giving the Pearson term a usable sample.
    """
    return min(wanted, batch_size // 2)


class BrighterDataset(Dataset):
    """Rows of data/gold/tokenized for one part, optionally restricted to some languages."""

    def __init__(self, parquet: str | Path, parts: Sequence[str], languages: Sequence[str], node_masks: torch.Tensor,
                 langs: Sequence[str] | None = None, tracks: Sequence[str] = ("A",)) -> None:
        df = pd.read_parquet(parquet, columns=["id", "lang", "part", "track", "input_ids", "attention_mask",
                                               "labels", "neutral", "intensity", "length"])
        df = df[df["part"].isin(parts) & df["track"].isin(tracks)]
        if langs is not None:
            df = df[df["lang"].isin(list(langs))]
        df = df.sort_values("id").reset_index(drop=True)
        self.ids = df["id"].tolist()
        self.lang_names = df["lang"].tolist()
        index = {name: i for i, name in enumerate(languages)}
        unk = index.get("<unk>", len(languages) - 1)
        self.lang = np.array([index.get(x, unk) for x in self.lang_names], dtype=np.int64)
        self.input_ids = [np.asarray(x, dtype=np.int64) for x in df["input_ids"]]
        self.attention_mask = [np.asarray(x, dtype=np.int64) for x in df["attention_mask"]]
        labels = np.stack(df["labels"].to_numpy()).astype(np.float32) if len(df) else np.zeros((0, 6), np.float32)
        neutral = df["neutral"].to_numpy(dtype=np.float32).reshape(-1, 1)
        self.labels = np.concatenate([labels, neutral], 1)
        self.intensity = np.stack(df["intensity"].to_numpy()).astype(np.int64) if len(df) else np.zeros((0, 6), np.int64)
        self.length = df["length"].to_numpy(dtype=np.int64)
        self.node_mask = node_masks.numpy()[self.lang] if len(df) else np.zeros((0, NUM_NODES), np.float32)
        self.has_intensity = (self.intensity >= 0).any(1)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict[str, np.ndarray | int]:
        return {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i], "labels": self.labels[i],
                "intensity": self.intensity[i], "node_mask": self.node_mask[i], "lang": int(self.lang[i]), "index": i}

    def pos_weight(self, clip: float) -> torch.Tensor:
        """Negatives over positives per node, clipped (plan 6.1); masked rows do not count."""
        pos = (self.labels * self.node_mask).sum(0)
        total = self.node_mask.sum(0)
        return torch.tensor(np.clip((total - pos) / np.maximum(pos, 1.0), 0.0, clip), dtype=torch.float32)


def make_collate(pad_id: int):
    """Dynamic padding to the longest row of the batch."""

    def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(r["input_ids"]) for r in rows)
        ids = np.full((len(rows), width), pad_id, dtype=np.int64)
        att = np.zeros((len(rows), width), dtype=np.int64)
        for k, r in enumerate(rows):
            n = len(r["input_ids"])
            ids[k, :n] = r["input_ids"]
            att[k, :n] = r["attention_mask"]
        return {
            "input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(att),
            "labels": torch.from_numpy(np.stack([r["labels"] for r in rows])),
            "intensity": torch.from_numpy(np.stack([r["intensity"] for r in rows])),
            "node_mask": torch.from_numpy(np.stack([r["node_mask"] for r in rows])),
            "lang": torch.tensor([r["lang"] for r in rows]),
            "index": torch.tensor([r["index"] for r in rows]),
        }

    return collate


@dataclass
class SamplerConfig:
    batch_size: int
    temperature: float = 0.7
    quota: int = 0
    seed: int = 42
    mega: int = 50


class TemperatureBatchSampler(Sampler[list[int]]):
    """Language-tempered, length-grouped batches with an intensity quota (plan 6.5 item 3).

    A language is drawn with probability proportional to n^T and rows cycle through a
    permutation inside it. Order depends only on (seed, epoch), which makes resume exact.
    """

    def __init__(self, ds: BrighterDataset, cfg: SamplerConfig) -> None:
        self.ds, self.cfg = ds, cfg
        self.epoch = 0
        self.groups = [np.flatnonzero(ds.lang == x) for x in np.unique(ds.lang)]
        sizes = np.array([len(g) for g in self.groups], dtype=np.float64)
        weight = sizes ** cfg.temperature
        self.lang_p = weight / weight.sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.ds) // self.cfg.batch_size

    def _draw(self, rng: np.random.Generator, rows: np.ndarray, count: int) -> np.ndarray:
        reps = int(np.ceil(count / max(len(rows), 1)))
        return np.concatenate([rng.permutation(rows) for _ in range(reps)])[:count]

    def __iter__(self) -> Iterator[list[int]]:
        cfg = self.cfg
        rng = np.random.default_rng([cfg.seed, self.epoch])
        total = len(self) * cfg.batch_size
        counts = rng.multinomial(total, self.lang_p)
        draw = np.concatenate([self._draw(rng, g, c) for g, c in zip(self.groups, counts)])
        rng.shuffle(draw)

        quota = cfg.quota if self.ds.has_intensity.any() else 0
        need = quota * len(self)
        if quota:
            flagged = self.ds.has_intensity[draw]
            pool_i, rest = draw[flagged], draw[~flagged]
            if len(pool_i) < need:
                missing = need - len(pool_i)
                extra = self._draw(rng, np.flatnonzero(self.ds.has_intensity), missing)
                pool_i = np.concatenate([pool_i, extra])
                rest = rest[missing:]
            else:
                rest = np.concatenate([rest, pool_i[need:]])
                pool_i = pool_i[:need]
            rng.shuffle(pool_i)
            rng.shuffle(rest)
        else:
            pool_i, rest = draw[:0], draw

        q, r = quota, cfg.batch_size - quota
        batches: list[list[int]] = []
        for start in range(0, len(self), cfg.mega):
            n = min(cfg.mega, len(self) - start)
            a = pool_i[start * q:(start + n) * q]
            b = rest[start * r:(start + n) * r]
            a = a[np.argsort(self.ds.length[a], kind="stable")]
            b = b[np.argsort(self.ds.length[b], kind="stable")]
            for k in range(n):
                batches.append([int(x) for x in a[k * q:(k + 1) * q]] + [int(x) for x in b[k * r:(k + 1) * r]])
        for k in rng.permutation(len(batches)):
            yield batches[int(k)]
