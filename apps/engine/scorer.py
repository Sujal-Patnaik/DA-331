"""Model access for the applications (plan 9.1, 9.7): one class for Spark and the API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from apps.engine.langid import UNK
from emoldgn.model import load_model

_cached: dict[str, "Scorer"] = {}


def merge_tokens(tokens: Sequence[str], weights: np.ndarray) -> list[tuple[str, float]]:
    """Join sentencepiece pieces into words, summing attention."""
    words: list[tuple[str, float]] = []
    for tok, w in zip(tokens, weights):
        if tok in ("<s>", "</s>", "<pad>"):
            continue
        if tok.startswith("▁") or not words:
            words.append((tok.lstrip("▁"), float(w)))
        else:
            words[-1] = (words[-1][0] + tok, words[-1][1] + float(w))
    return [x for x in words if x[0]]


class Scorer:
    """Loads artifacts/ once and scores clauses in length-sorted mini-batches."""

    def __init__(self, artifact_dir: str | Path, device: str | None = None, batch_size: int = 32, max_len: int = 128,
                 top_triggers: int = 3) -> None:
        d = Path(artifact_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ck = load_model(d / "best.pt", self.device)
        self.tok = AutoTokenizer.from_pretrained(ck["encoder"])
        self.batch_size, self.max_len, self.top_triggers = batch_size, max_len, top_triggers
        self.lang_index: dict[str, int] = json.loads((d / "lang_index.json").read_text(encoding="utf-8"))["languages"]
        th = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
        self.t_global = np.array(th["global"], dtype=np.float32)
        self.t_lang = {k: np.array(v, dtype=np.float32) for k, v in th["per_language"].items()}
        self.has_neutral = bool(self.model.flags.get("n2_signed"))
        self.node_mask = self.model.node_mask.cpu().numpy() if hasattr(self.model, "node_mask") else None

    def index_of(self, lang: str) -> int:
        """Unseen language: its family prior if the bank has it, else the global graph."""
        return self.lang_index.get(lang, self.lang_index[UNK])

    def thresholds(self, langs: Sequence[str]) -> np.ndarray:
        return np.stack([self.t_lang.get(x, self.t_global) for x in langs])

    def mask(self, langs: Sequence[str]) -> np.ndarray:
        """N x 6: which emotions each language has."""
        if self.node_mask is None:
            return np.ones((len(langs), 6), dtype=np.float32)
        return self.node_mask[[self.index_of(x) for x in langs]][:, :6]

    def score(self, clauses: Sequence[str], langs: Sequence[str]) -> dict[str, Any]:
        """probs N x 7, inten N x 6 (expected intensity), ad N x 7 x 7, triggers per clause."""
        n = len(clauses)
        probs = np.zeros((n, 7), dtype=np.float32)
        inten = np.zeros((n, 6), dtype=np.float32)
        ad = np.zeros((n, 7, 7), dtype=np.float32)
        triggers: list[list[str]] = [[] for _ in range(n)]
        order = np.argsort([len(c) for c in clauses], kind="stable")
        amp = self.device.startswith("cuda")
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tok([clauses[i] for i in idx], truncation=True, max_length=self.max_len, padding=True,
                           return_tensors="pt").to(self.device)
            lang = torch.tensor([self.index_of(langs[i]) for i in idx], device=self.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = self.model(enc["input_ids"], enc["attention_mask"], lang)
            p = torch.sigmoid(out["logits"].float()).cpu().numpy()
            probs[idx] = p
            if out["int_logits"] is not None:
                inten[idx] = torch.sigmoid(out["int_logits"][:, :6].float()).sum(-1).cpu().numpy()
            if out["AD"] is not None:
                ad[idx] = out["AD"].float().cpu().numpy()
            if out["att"] is not None:
                att = out["att"].float().cpu().numpy()
                top = p[:, :6].argmax(1)
                for k, i in enumerate(idx):
                    tokens = self.tok.convert_ids_to_tokens(enc["input_ids"][k].tolist())
                    words = merge_tokens(tokens, att[k, top[k]])
                    words.sort(key=lambda x: -x[1])
                    triggers[i] = [w for w, _ in words[: self.top_triggers]]
        return {"probs": probs, "inten": inten, "ad": ad, "triggers": triggers}


def get_scorer(artifact_dir: str | Path, **kw: Any) -> Scorer:
    """Per-process singleton, so a reused Spark worker loads the model only once."""
    key = str(artifact_dir)
    if key not in _cached:
        _cached[key] = Scorer(artifact_dir, **kw)
    return _cached[key]
