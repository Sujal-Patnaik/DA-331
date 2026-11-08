#!/usr/bin/env python
"""Report figures. --graphs needs M4 only; --results needs finished runs (M8).

Heatmaps use one diverging scale for signed NPMI and one sequential scale for values in
[0, 1], with cell values printed, so they read without colour. Masked emotions are blank.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402

from spark_jobs.common import NODES, load_config, path_of  # noqa: E402

EXAMPLES = ["eng", "hau", "tir"]


def heat(ax, m: np.ndarray, title: str, mask: np.ndarray, signed: bool) -> None:
    labels = [n[:3] for n in NODES]
    blank = np.outer(mask, mask) == 0
    kw = {"vmin": -1, "vmax": 1, "cmap": "RdBu_r", "center": 0} if signed else {"vmin": 0, "vmax": 1, "cmap": "Blues"}
    sns.heatmap(m, ax=ax, annot=True, fmt=".2f", annot_kws={"size": 7}, mask=blank, square=True, cbar=False,
                xticklabels=labels, yticklabels=labels, linewidths=0.5, linecolor="white", **kw)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)


def graphs(cfg: dict, out: Path) -> None:
    """Global graph plus three languages: NPMI on top, P(j|i) below (M4)."""
    bank = torch.load(path_of(cfg, "gold", "graph_bank.pt"), weights_only=False)
    names = bank["languages"]
    picks = [("<unk>", "global")] + [(x, x) for x in EXAMPLES]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), layout="constrained")
    for k, (key, label) in enumerate(picks):
        i = names.index(key)
        mask = bank["mask"][i].numpy()
        heat(axes[0, k], bank["npmi_raw"][i].numpy(), f"NPMI, {label}", mask, True)
        heat(axes[1, k], bank["cond_raw"][i].numpy(), f"P(column | row), {label}", mask, False)
    fig.savefig(out / "graphs_npmi_cond.png", dpi=150)
    plt.close(fig)

    lam = bank["lambda_l"].numpy()
    n = len(bank["trained_languages"])
    order = np.argsort(lam[:n])
    fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
    ax.bar([names[i] for i in order], lam[order], color="#2a78b5")
    ax.set_ylabel("lambda (weight on own statistics)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax.set_title("N4 shrinkage weight per language, kappa = 2000")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "shrinkage_lambda.png", dpi=150)
    plt.close(fig)
    print("wrote", out / "graphs_npmi_cond.png", "and", out / "shrinkage_lambda.png")


def results(cfg: dict, out: Path, exp: str, run_id: str | None = None, config: str | None = None) -> None:
    """Average dynamic graph per language and gate values on validation data (M8)."""
    from emoldgn import data as D
    from emoldgn.model import load_model
    from emoldgn.train import PAD_ID, data_languages, load_bank, resolve

    run = Path(path_of(cfg, "runs", run_id or exp))
    if not (run / "ckpt_best.pt").exists():
        print(f"{exp} has no checkpoint yet; nothing to draw")
        return
    c = resolve(exp, config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(run / "ckpt_best.pt", device)
    bank = load_bank(c)
    ds = D.BrighterDataset(path_of(c, "gold", "tokenized"), ["val"], bank["languages"], bank["mask"], data_languages(c))
    order = np.argsort(ds.length, kind="stable")
    collate = D.make_collate(PAD_ID)
    sums: dict[int, np.ndarray] = {}
    gates: dict[int, list[float]] = {}
    counts: dict[int, int] = {}
    with torch.no_grad():
        for s in range(0, len(order), c["eval_batch_size"]):
            b = collate([ds[int(i)] for i in order[s:s + c["eval_batch_size"]]])
            o = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["lang"].to(device))
            ad = o["AD"].float().cpu().numpy()
            gate = o["gate"].float().mean((1, 2)).cpu().numpy() if o["gate"] is not None else None
            for k, x in enumerate(b["lang"].tolist()):
                sums[x] = sums.get(x, 0) + ad[k]
                counts[x] = counts.get(x, 0) + 1
                if gate is not None:
                    gates.setdefault(x, []).append(float(gate[k]))
    names = bank["languages"]
    picks = [x for x in sums if names[x] in c["subset_langs"]][:4] or list(sums)[:4]
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 3.8), layout="constrained", squeeze=False)
    for ax, x in zip(axes[0], picks):
        heat(ax, sums[x] / counts[x], f"Mean dynamic graph, {names[x]}", bank["mask"][x].numpy(), False)
    fig.savefig(out / f"dynamic_graph_{exp}.png", dpi=150)
    plt.close(fig)
    if gates:
        order_g = sorted(gates, key=lambda x: np.mean(gates[x]))
        fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
        ax.bar([names[x] for x in order_g], [np.mean(gates[x]) for x in order_g], color="#2a78b5")
        ax.set_ylabel("mean gate (1 = statistical branch)")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=60, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(out / f"gate_values_{exp}.png", dpi=150)
        plt.close(fig)
    write_attention_examples(c, model, ds, run, device)
    print("wrote figures for", exp)


def write_attention_examples(cfg: dict, model: torch.nn.Module, ds, run: Path, device: str, n: int = 5) -> None:
    """Five attention examples; they quote BRIGHTER, so they stay in runs/ (rule 7)."""
    from transformers import AutoTokenizer

    from apps.engine.scorer import merge_tokens
    from emoldgn.data import make_collate
    from emoldgn.train import PAD_ID

    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    picks = np.flatnonzero(ds.labels[:, :6].sum(1) > 0)[:: max(1, len(ds) // (n * 3))][:n]
    batch = make_collate(PAD_ID)([ds[int(i)] for i in picks])
    with torch.no_grad():
        o = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["lang"].to(device))
    if o["att"] is None:
        return
    att = o["att"].float().cpu().numpy()
    lines = ["# Attention examples (BRIGHTER text, do not share)", ""]
    for k, i in enumerate(picks):
        c = int(np.argmax(ds.labels[int(i), :6]))
        tokens = tok.convert_ids_to_tokens(batch["input_ids"][k].tolist())
        words = sorted(merge_tokens(tokens, att[k, c]), key=lambda x: -x[1])[:5]
        lines.append(f"- {ds.ids[int(i)]} ({NODES[c]}): " + ", ".join(f"{w} {v:.2f}" for w, v in words))
    (run / "analysis").mkdir(exist_ok=True)
    (run / "analysis/attention_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", action="store_true")
    ap.add_argument("--results", action="store_true")
    ap.add_argument("--exp", default="final_full")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(path_of(cfg, "reports", "figures"))
    out.mkdir(parents=True, exist_ok=True)
    if args.graphs:
        graphs(cfg, out)
    if args.results:
        results(cfg, out, args.exp, args.run_id, args.config)


if __name__ == "__main__":
    main()
