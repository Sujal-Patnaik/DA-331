"""EmoLDGN+ and its signed GCN layer (plan sections 5 and 8).

With every flag off this class is the faithful LDGN baseline B4 (plan 2, 10.1): random
label queries, unsigned conditional-probability graph, dynamic graph, second GCN,
concatenation. The flags add N1, N2, N4, N5 and N6.

Departures from the plan's snippet, each logged in docs/DECISIONS.md:
* The learned residual and the dynamic graph are multiplied by the language's node mask,
  so masked emotions stay isolated (rule 5).
* LayerNorm on U and a scaled dynamic-graph dot product, in every LDGN variant including
  B4. XLM-R's attended states are far smaller than LDGN's BiLSTM outputs (about 0.09 per
  dimension), and an unscaled 256-dimension dot product saturates the sigmoid. Both were
  found by test_overfit.py.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

NUM_NODES = 7
FLAG_NAMES = ("n1_anchors", "n2_signed", "n4_hierarchical", "n5_reg_gate", "n6_intensity", "n3_annotator")


def norm_adj(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation D^-1/2 (A + I) D^-1/2, batched or not (plan 2.2)."""
    A = A + torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    d = A.sum(-1).clamp(min=1e-3).pow(-0.5)
    return d.unsqueeze(-1) * A * d.unsqueeze(-2)


class SignedGCNLayer(nn.Module):
    """N2 layer: H W_self + A+ H W+ - A- H W-, then LeakyReLU (plan 5.3)."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.self_ = nn.Linear(d_in, d_out, bias=False)
        self.pos = nn.Linear(d_in, d_out, bias=False)
        self.neg = nn.Linear(d_in, d_out, bias=False)

    def forward(self, H: torch.Tensor, Ap: torch.Tensor, An: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.self_(H) + Ap @ self.pos(H) - An @ self.neg(H), 0.2)


class GCNLayer(nn.Module):
    """Plain GCN layer used by LDGN (plan 2.2)."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(A @ self.W(H), 0.2)


def mean_pool(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Attention-mask aware mean over tokens."""
    m = mask.unsqueeze(-1).to(states.dtype)
    return (states * m).sum(1) / m.sum(1).clamp(min=1.0)


def build_encoder(name: str, pretrained: bool = True, tiny: bool = False) -> nn.Module:
    """XLM-R encoder. ``tiny`` builds a small random one for fast CPU tests."""
    if tiny:
        from transformers import XLMRobertaConfig

        cfg = XLMRobertaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=64, max_position_embeddings=128)
        return AutoModel.from_config(cfg)
    if pretrained:
        return AutoModel.from_pretrained(name, add_pooling_layer=False)
    return AutoModel.from_config(AutoConfig.from_pretrained(name), add_pooling_layer=False)


class EmoLDGNPlus(nn.Module):
    """Label attention, GCN on the statistical graph, dynamic graph, second GCN (plan 8)."""

    def __init__(self, flags: dict[str, Any], bank: dict[str, Any], encoder: nn.Module, hid: int = 512,
                 label_init: torch.Tensor | None = None) -> None:
        super().__init__()
        self.flags = {k: bool(flags.get(k, False)) for k in FLAG_NAMES}
        self.flags["head"] = "ldgn"
        self.enc = encoder
        D = encoder.config.hidden_size
        C = NUM_NODES
        self.sqrt_d = math.sqrt(D)
        n1, n2, n4, n5, n6 = (self.flags[k] for k in FLAG_NAMES[:5])

        if n1 and label_init is not None:
            self.C_lab = nn.Parameter(label_init.clone().float())
        else:
            self.C_lab = nn.Parameter(torch.randn(C, D))
        self.Wq = nn.Linear(D, D) if n1 else nn.Identity()
        self.Wk = nn.Linear(D, D) if n1 else nn.Identity()

        suffix = "prior" if n4 else "raw"
        mask = bank["mask"].clone().float()
        if not n2:
            mask[:, -1] = 0.0  # B4 has no neutral node (plan 5.3 adds it)
        self.register_buffer("node_mask", mask)
        pair = mask.unsqueeze(1) * mask.unsqueeze(2)
        if n2:
            self.register_buffer("A_pos", bank[f"A_pos_{suffix}"].clone().float() * pair)
            self.register_buffer("A_neg", bank[f"A_neg_{suffix}"].clone().float() * pair)
        else:
            off_diag = 1.0 - torch.eye(C)
            self.register_buffer("A_pos", bank[f"cond_{suffix}"].clone().float() * pair * off_diag)
            self.register_buffer("A_neg", torch.zeros(1))
        self.register_buffer("P_prior", bank["cond_prior"].clone().float())
        self.delta = nn.Parameter(torch.zeros(C, C)) if n4 else None

        layer = SignedGCNLayer if n2 else GCNLayer
        self.g1a, self.g1b = layer(D, hid), layer(hid, hid)
        self.Wa, self.Wb = nn.Linear(hid, hid // 2), nn.Linear(hid, hid // 2)
        self.g2a, self.g2b = GCNLayer(hid, hid), GCNLayer(hid, hid)
        self.gate = nn.Linear(2 * hid, hid) if n5 else None
        z_dim = hid if n5 else 2 * hid
        self.presence = nn.Linear(z_dim, 1)
        # Per-node presence bias b_c belongs to the joint heads of plan 5.7.
        self.bias_c = nn.Parameter(torch.zeros(C)) if n6 else None
        if n6:
            self.w_int = nn.Linear(z_dim, 1, bias=False)
            self.theta1 = nn.Parameter(torch.zeros(C))
            self.dtheta = nn.Parameter(torch.zeros(C, 2))
        self.drop = nn.Dropout(0.1)
        self.norm_u = nn.LayerNorm(D)
        self.ad_scale = math.sqrt(hid // 2)

    def graph_head_parameters(self) -> int:
        """Trainable parameters outside the encoder (plan 6.3: about 4.5M)."""
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("enc."))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, lang: torch.Tensor) -> dict[str, torch.Tensor | None]:
        H = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state  # B x T x D
        q = self.Wq(self.C_lab)  # C x D
        att = torch.einsum("btd,cd->bct", self.Wk(H), q) / self.sqrt_d
        att = att.masked_fill(mask[:, None, :] == 0, -1e4).softmax(-1)  # B x C x T
        U = self.norm_u(self.drop(torch.einsum("bct,btd->bcd", att, H)))  # B x C x D

        node = self.node_mask[lang]
        pair = node.unsqueeze(1) * node.unsqueeze(2)
        Ap = self.A_pos[lang]
        if self.delta is not None:
            Ap = F.relu(Ap + self.delta) * pair
        Ap = norm_adj(Ap)
        if self.flags["n2_signed"]:
            An = self.A_neg[lang]
            An = An / An.sum(-1, keepdim=True).clamp(min=1e-3)
            H2 = self.g1b(self.g1a(U, Ap, An), Ap, An)
        else:
            H2 = self.g1b(self.g1a(U, Ap), Ap)

        AD = torch.sigmoid(self.Wa(H2) @ self.Wb(H2).transpose(1, 2) / self.ad_scale) * pair  # B x C x C
        Ad = norm_adj(AD)
        H4 = self.g2b(self.g2a(H2, Ad), Ad)

        if self.gate is not None:
            g = torch.sigmoid(self.gate(torch.cat([H2, H4], -1)))
            Z = g * H2 + (1 - g) * H4
        else:
            g = None
            Z = torch.cat([H2, H4], -1)

        logits = self.presence(Z).squeeze(-1)  # B x C
        if self.bias_c is not None:
            logits = logits + self.bias_c
        int_logits = None
        if self.flags["n6_intensity"]:
            theta = torch.cat(
                [self.theta1[:, None], self.theta1[:, None] + F.softplus(self.dtheta).cumsum(-1)], -1
            )
            int_logits = self.w_int(Z) - theta[None]  # B x C x 3
        return {"logits": logits, "int_logits": int_logits, "AD": AD, "att": att, "gate": g}


def empty_bank(state: dict[str, torch.Tensor], flags: dict[str, Any]) -> dict[str, Any]:
    """Placeholder bank of the right shapes, for rebuilding a saved model."""
    n = state["node_mask"].shape[0]
    zeros = torch.zeros(n, NUM_NODES, NUM_NODES)
    suffix = "prior" if flags.get("n4_hierarchical") else "raw"
    return {"mask": torch.ones(n, NUM_NODES), "cond_prior": zeros, f"cond_{suffix}": zeros,
            f"A_pos_{suffix}": zeros, f"A_neg_{suffix}": zeros}


def save_model(model: nn.Module, path: str | Path, meta: dict[str, Any]) -> None:
    """Portable model file: weights plus what is needed to rebuild it (atomic)."""
    tmp = Path(str(path) + ".tmp")
    torch.save({"state_dict": model.state_dict(), "flags": model.flags, **meta}, tmp)
    tmp.replace(path)


def load_model(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a saved model for inference (plan 9.1)."""
    from emoldgn.ldgn_baseline import LinearBaseline

    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc = build_encoder(ck["encoder"], pretrained=False, tiny=bool(ck.get("tiny_encoder")))
    if ck["flags"].get("head") == "linear":
        model: nn.Module = LinearBaseline(enc)
    else:
        model = EmoLDGNPlus(ck["flags"], empty_bank(ck["state_dict"], ck["flags"]), enc, hid=ck["hid"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
