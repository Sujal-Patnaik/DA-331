"""Baselines (plan 10.1): B2 is mean-pooled XLM-R plus a linear layer; B4 is EmoLDGNPlus flags-off."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from emoldgn.model import NUM_NODES, EmoLDGNPlus, mean_pool


class LinearBaseline(nn.Module):
    """B2: no graph at all. The neutral output exists but is masked out of the loss."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.flags: dict[str, Any] = {"head": "linear"}
        self.enc = encoder
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.config.hidden_size, NUM_NODES)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, lang: torch.Tensor) -> dict[str, torch.Tensor | None]:
        H = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        logits = self.head(self.drop(mean_pool(H, mask)))
        return {"logits": logits, "int_logits": None, "AD": None, "att": None, "gate": None}


def build_model(flags: dict[str, Any], bank: dict[str, Any], encoder: nn.Module, hid: int,
                label_init: torch.Tensor | None = None) -> nn.Module:
    """Pick the head named by the experiment's flags."""
    if flags.get("head", "ldgn") == "linear":
        return LinearBaseline(encoder)
    return EmoLDGNPlus(flags, bank, encoder, hid=hid, label_init=label_init)
