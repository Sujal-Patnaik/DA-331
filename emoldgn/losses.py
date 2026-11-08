"""Losses (plan 6.1, 8): masked weighted BCE, N6 intensity, N5 graph regulariser.

Every loss runs in float32 even under fp16 autocast. The plan's clamp(min=1e-6) is below
float16 resolution, so class probabilities would reach zero and log() would give -inf.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

NUM_EMOTIONS = 6  # the intensity head has no neutral row (CLAUDE.md section 7)


def masked_bce(logits: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    """Weighted BCE over the nodes a language has (rule 5)."""
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    return (bce * y_mask).sum() / y_mask.sum().clamp(min=1)


def intensity_loss(int_logits: torch.Tensor, inten: torch.Tensor, alpha: float = 0.5, beta: float = 0.5,
                   gamma: float = 1.0) -> torch.Tensor:
    """L_ord + alpha L_dist + beta L_rho on the emotion rows (plan 5.7, 8).

    ``int_logits`` is B x C x 3 cumulative logits, ``inten`` B x 6 with -1 for "not available".
    """
    int_logits = int_logits[:, :NUM_EMOTIONS].float()
    has = inten >= 0
    if not has.any():
        return int_logits.sum() * 0.0
    n = has.sum().float()
    levels = torch.arange(1, 4, device=inten.device)
    target = (inten.clamp(min=0).unsqueeze(-1) >= levels).float()
    l_ord = (F.binary_cross_entropy_with_logits(int_logits, target, reduction="none").sum(-1) * has).sum() / n

    cum = torch.sigmoid(int_logits)  # P(I >= k), B x E x 3
    p_ge = torch.cat([torch.ones_like(cum[..., :1]), cum, torch.zeros_like(cum[..., :1])], -1)
    p_cls = (p_ge[..., :-1] - p_ge[..., 1:]).clamp(min=1e-6)  # P(I = k), k = 0..3
    k = torch.arange(4, device=inten.device, dtype=torch.float32)
    soft = torch.softmax(-gamma * (k - inten.clamp(min=0).unsqueeze(-1).float()).abs(), -1)
    l_dist = (-(soft * p_cls.log()).sum(-1) * has).sum() / n

    expected = cum.sum(-1)
    l_rho = int_logits.sum() * 0.0
    for c in range(inten.size(1)):
        m = has[:, c]
        if m.sum() > 2:
            x = expected[m, c]
            y = inten[m, c].float()
            x, y = x - x.mean(), y - y.mean()
            denom = torch.sqrt(x.pow(2).sum() * y.pow(2).sum() + 1e-12)
            l_rho = l_rho + 1 - (x * y).sum() / denom
    return l_ord + alpha * l_dist + beta * l_rho / inten.size(1)


def graph_regulariser(AD: torch.Tensor, P_prior_batch: torch.Tensor, mu: float = 0.01) -> torch.Tensor:
    """N5: pull the batch-mean dynamic graph to the prior, plus sparsity (plan 5.6)."""
    AD = AD.float()
    return ((AD.mean(0) - P_prior_batch.float()) ** 2).sum() + mu * AD.abs().mean()


def total_loss(out: dict[str, Any], batch: dict[str, torch.Tensor], pos_weight: torch.Tensor,
               flags: dict[str, Any], cfg: dict[str, Any], P_prior_batch: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Full objective of plan 6.1; returns the parts for logging."""
    with torch.autocast(device_type=out["logits"].device.type, enabled=False):
        y_mask = batch["node_mask"].float()
        if not flags.get("n2_signed"):
            y_mask = y_mask * torch.tensor([1, 1, 1, 1, 1, 1, 0], dtype=y_mask.dtype, device=y_mask.device)
        loss = l_bce = masked_bce(out["logits"].float(), batch["labels"].float(), y_mask, pos_weight)
        parts = {"bce": l_bce.detach()}
        if flags.get("n6_intensity"):
            n6 = cfg["n6"]
            l_int = intensity_loss(out["int_logits"], batch["intensity"], n6["alpha"], n6["beta"], n6["gamma"])
            loss = loss + n6["lambda_i"] * l_int
            parts["intensity"] = l_int.detach()
        if flags.get("n5_reg_gate"):
            n5 = cfg["n5"]
            l_graph = graph_regulariser(out["AD"], P_prior_batch, n5["mu"])
            loss = loss + n5["lambda_g"] * l_graph
            parts["graph"] = l_graph.detach()
    parts["loss"] = loss
    return parts
