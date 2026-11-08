"""The final configuration must memorise 64 examples within 150 steps (GPU, real encoder)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from emoldgn import data as D
from emoldgn.evaluate import track_a
from emoldgn.losses import total_loss
from emoldgn.train import PAD_ID, build, load_bank, resolve

pytestmark = pytest.mark.slow


def test_final_configuration_overfits_64_examples():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    cfg = resolve("final_subset_s42")
    tok = Path(cfg["paths"]["gold"]) / "tokenized"
    if not tok.exists():
        pytest.skip("M3 has not run")
    device = torch.device("cuda")
    torch.manual_seed(0)
    bank = load_bank(cfg)
    ds = D.BrighterDataset(tok, ["train"], bank["languages"], bank["mask"], ["eng", "hau"])
    # 64 fixed rows that each carry at least one emotion, so every class has examples.
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(ds.labels[:, :6].sum(1) > 0)
    rows = rng.choice(pool, 64, replace=False)
    collate = D.make_collate(PAD_ID)
    batches = [collate([ds[int(i)] for i in rows[j:j + 16]]) for j in range(0, 64, 16)]
    model = build(cfg, bank, device)
    pos_weight = ds.pos_weight(cfg["pos_weight_clip"]).to(device)
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("enc.")]
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("enc.")]
    optim = torch.optim.AdamW([{"params": enc, "lr": cfg["lr_encoder"]}, {"params": head, "lr": cfg["lr_head"]}],
                              weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    flags = cfg["exp"]["flags"]

    model.train()
    for step in range(150):
        batch = batches[step % len(batches)]
        lang = batch["lang"].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), lang)
        b = {k: batch[k].to(device) for k in ("labels", "intensity", "node_mask")}
        loss = total_loss(out, b, pos_weight, flags, cfg, model.P_prior[lang].mean(0))["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

    model.eval()
    probs = []
    with torch.no_grad():
        for batch in batches:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["lang"].to(device))
            probs.append(torch.sigmoid(out["logits"].float())[:, :6].cpu().numpy())
    p = np.concatenate(probs)
    y = ds.labels[rows, :6].astype(int)
    mask = ds.node_mask[rows, :6]
    # Pooled over the 64 rows: the training macro-F1 the test is about.
    pred = ((p >= 0.5) & (mask > 0)).astype(int)
    score = track_a(y, pred, mask, ["all"] * len(y))["mean"]
    print("train macro-F1", score)
    assert score > 0.9
