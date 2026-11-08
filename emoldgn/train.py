"""Training loop with checkpoint and resume (CLAUDE.md section 8).

``python -m emoldgn.train --exp <id> [--resume] [--max-steps N] [--run-id NAME]``

--run-id writes to another folder; smoke tests use it so they can never touch the real
experiment folders. Resuming restarts the epoch after the last finished one and restores
every RNG, and the sampler's order depends only on (seed, epoch), so a resumed run matches
an uninterrupted one.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import signal
import sys
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emoldgn import data as D  # noqa: E402
from emoldgn.evaluate import track_a, track_b  # noqa: E402
from emoldgn.ldgn_baseline import build_model  # noqa: E402
from emoldgn.losses import total_loss  # noqa: E402
from emoldgn.model import build_encoder, mean_pool, save_model  # noqa: E402
from spark_jobs.common import load_config, path_of  # noqa: E402

log = logging.getLogger("train")
PAD_ID = 1  # <pad> of xlm-roberta-base


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def experiments(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Entries of config/experiments.yaml in run order."""
    with open(path_of(cfg, "experiments"), encoding="utf-8") as f:
        return yaml.safe_load(f)["experiments"]


def vram_gb() -> float:
    """Total GPU memory, 0 without a GPU."""
    return torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0.0


def resolve(exp_id: str, config: str | None = None) -> dict[str, Any]:
    """Fully resolved run configuration (written to runs/<id>/config.yaml)."""
    cfg = load_config(config)
    entry = next((e for e in experiments(cfg) if e["id"] == exp_id), None)
    if entry is None:
        raise SystemExit(f"unknown experiment {exp_id!r}")
    cfg["exp"] = copy.deepcopy(entry)
    meta = Path(path_of(cfg, "gold", "tokenizer_meta.json"))
    if meta.exists():
        cfg["max_len"] = json.loads(meta.read_text())["max_len"]
    if cfg["freeze_word_embeddings"] == "auto":
        cfg["freeze_word_embeddings"] = bool(vram_gb() < 8)
    return cfg


def load_bank(cfg: dict[str, Any]) -> dict[str, Any]:
    """Graph bank written by spark_jobs/06_priors.py."""
    return torch.load(path_of(cfg, "gold", "graph_bank.pt"), map_location="cpu", weights_only=False)


def data_languages(cfg: dict[str, Any]) -> list[str] | None:
    """Languages of a run: the subset list, or None for all Track A languages."""
    return list(cfg["subset_langs"]) if cfg["exp"]["data"] == "subset" else None


def make_loaders(cfg: dict[str, Any], bank: dict[str, Any]) -> tuple[Any, ...]:
    """Training loader with the tempered sampler, plus a length-sorted validation loader."""
    pq = path_of(cfg, "gold", "tokenized")
    langs = data_languages(cfg)
    train = D.BrighterDataset(pq, ["train"], bank["languages"], bank["mask"], langs)
    val = D.BrighterDataset(pq, ["val"], bank["languages"], bank["mask"], langs)
    quota = 0
    if cfg["exp"]["flags"].get("n6_intensity") and cfg["exp"].get("intensity_quota", True):
        quota = D.intensity_quota(cfg["batch_size"], cfg["min_intensity_rows_per_batch"])
    sampler = D.TemperatureBatchSampler(train, D.SamplerConfig(
        batch_size=cfg["batch_size"], temperature=cfg["lang_sampling_temperature"], quota=quota, seed=cfg["exp"]["seed"]))
    collate = D.make_collate(PAD_ID)
    train_loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate)
    order = np.argsort(val.length, kind="stable")
    val_batches = [order[i:i + cfg["eval_batch_size"]].tolist() for i in range(0, len(order), cfg["eval_batch_size"])]
    val_loader = DataLoader(val, batch_sampler=val_batches, collate_fn=collate)
    return train, val, sampler, train_loader, val_loader


def center_anchors(c: torch.Tensor) -> torch.Tensor:
    """Remove what all definitions share and restore a random query's norm.

    Raw mean-pooled definitions differ by about 5% of their norm, because each one reads
    like "feeling ...", so every label would attend to the same tokens.
    """
    c = c - c.mean(0, keepdim=True)
    return c / c.norm(dim=1, keepdim=True).clamp(min=1e-6) * c.size(1) ** 0.5


def encode_definitions(encoder: torch.nn.Module, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    """N1 label anchors: each definition, mean pooled by the frozen encoder (plan 5.2)."""
    from transformers import AutoTokenizer

    from spark_jobs.common import NODES

    with open(path_of(cfg, "label_definitions"), encoding="utf-8") as f:
        defs = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    enc = tok([defs[n] for n in NODES], padding=True, return_tensors="pt").to(device)
    encoder.to(device).eval()
    with torch.no_grad():
        out = encoder(**enc).last_hidden_state
    return center_anchors(mean_pool(out, enc["attention_mask"]).float().cpu())


def build(cfg: dict[str, Any], bank: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Encoder plus the head named by the flags."""
    flags = cfg["exp"]["flags"]
    tiny = bool(cfg.get("tiny_encoder"))
    encoder = build_encoder(cfg["encoder"], tiny=tiny)
    if cfg["freeze_word_embeddings"] is True:
        encoder.embeddings.word_embeddings.requires_grad_(False)
    label_init = None
    if flags.get("n1_anchors") and not tiny:
        label_init = encode_definitions(encoder, cfg, device)
    return build_model(flags, bank, encoder, cfg["gcn_hidden"], label_init).to(device)


def lr_schedule(warmup: int, total: int):
    """Linear warmup, then linear decay."""

    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    return f


def amp_enabled(cfg: dict[str, Any], device: torch.device) -> bool:
    return bool(cfg["fp16"]) and device.type == "cuda"


def infer(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, np.ndarray]:
    """Probabilities, expected and ordinal intensities over a loader, in loader order."""
    model.eval()
    keep: dict[str, list[np.ndarray]] = {k: [] for k in ("p", "e_int", "ord_int", "index")}
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            lang = batch["lang"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(ids, att, lang)
            keep["p"].append(torch.sigmoid(out["logits"].float()).cpu().numpy())
            keep["index"].append(batch["index"].numpy())
            if out["int_logits"] is not None:
                cum = torch.sigmoid(out["int_logits"][:, :6].float())
                keep["e_int"].append(cum.sum(-1).cpu().numpy())
                keep["ord_int"].append((cum > 0.5).sum(-1).cpu().numpy())
    res = {k: np.concatenate(v) for k, v in keep.items() if v}
    order = np.argsort(res["index"])
    return {k: v[order] for k, v in res.items()}


def validate(model: torch.nn.Module, loader: DataLoader, ds: D.BrighterDataset, device: torch.device,
             amp: bool) -> dict[str, float | None]:
    """Macro-F1 at a fixed 0.5 threshold and, with N6, Pearson r (CLAUDE.md section 8)."""
    res = infer(model, loader, device, amp)
    mask = ds.node_mask[:, :6]
    pred = ((res["p"][:, :6] >= 0.5) & (mask > 0)).astype(np.int64)
    f1 = track_a(ds.labels[:, :6].astype(np.int64), pred, mask, ds.lang_names)["mean"]
    rho = None
    if "e_int" in res:
        rho = track_b(ds.intensity, res["e_int"], ds.lang_names)["mean"]
    return {"val_macro_f1": f1, "val_pearson": rho}


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(s: dict[str, Any]) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])


def atomic_save(obj: Any, path: Path) -> None:
    """Temp file then rename, so a crash never leaves half a checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def write_status(run: Path, **fields: Any) -> None:
    p = run / "status.json"
    s = json.loads(p.read_text()) if p.exists() else {}
    s.update(fields)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2) + "\n")
    tmp.replace(p)


class Stop:
    """Turns SIGINT and SIGTERM into a flag checked between steps."""

    def __init__(self) -> None:
        self.requested = False
        self.old: dict[int, Any] = {}

    def __enter__(self) -> "Stop":
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                self.old[sig] = signal.signal(sig, self._set)
        return self

    def _set(self, *_: Any) -> None:
        self.requested = True

    def __exit__(self, *_: Any) -> None:
        for sig, handler in self.old.items():
            signal.signal(sig, handler)


def attach_log(run: Path) -> None:
    """Send this module's log to runs/<id>/train.log and stdout."""
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s %(message)s")
    for h in (logging.FileHandler(run / "train.log"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False


def train_run(cfg: dict[str, Any], run: Path, resume: bool = False, max_steps: int | None = None,
              stop_after_epochs: int | None = None) -> str:
    """One run. Returns the final status. stop_after_epochs simulates a kill (test_resume)."""
    run.mkdir(parents=True, exist_ok=True)
    attach_log(run)
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.get("force_cpu") else "cpu")
    amp = amp_enabled(cfg, device)
    flags = cfg["exp"]["flags"]
    seed_everything(cfg["exp"]["seed"])
    bank = load_bank(cfg)
    train_ds, val_ds, sampler, train_loader, val_loader = make_loaders(cfg, bank)
    model = build(cfg, bank, device)
    pos_weight = train_ds.pos_weight(cfg["pos_weight_clip"]).to(device)

    head, enc = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (enc if name.startswith("enc.") else head).append(p)
    optim = torch.optim.AdamW(
        [{"params": enc, "lr": cfg["lr_encoder"]}, {"params": head, "lr": cfg["lr_head"]}],
        weight_decay=cfg["weight_decay"])
    ep = cfg["epochs"][cfg["exp"]["epochs"]]
    # The schedule spans extend_to: a horizon of max would leave LR 0 for extended epochs.
    horizon = ep.get("extend_to", ep["max"])
    total_updates = max(1, horizon * len(sampler) // cfg["grad_accum"])
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_schedule(int(cfg["warmup_ratio"] * total_updates), total_updates))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, global_step, best, best_epoch, bad, limit = 0, 0, -1.0, -1, 0, ep["max"]
    ckpt_last, ckpt_best = run / "ckpt_last.pt", run / "ckpt_best.pt"
    if resume and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, global_step, best = ck["epoch"] + 1, ck["global_step"], ck["best"]
        best_epoch, bad, limit = ck["best_epoch"], ck["bad"], ck["limit"]
        restore_rng(ck["rng"])
        log.info("resumed after epoch %d", start_epoch)
        del ck
    else:
        (run / "metrics.jsonl").write_text("")
    write_status(run, status="running", last_epoch=start_epoch, best_epoch=best_epoch + 1, best_score=best)
    n_head = sum(p.numel() for p in head)
    log.info("device=%s amp=%s rows=%d val=%d steps/epoch=%d head_params=%.2fM", device, amp, len(train_ds), len(val_ds),
             len(sampler), n_head / 1e6)

    meta = {"encoder": cfg["encoder"], "hid": cfg["gcn_hidden"], "tiny_encoder": bool(cfg.get("tiny_encoder")),
            "max_len": cfg["max_len"], "languages": bank["languages"]}
    status = "done"
    stopped = False
    with Stop() as stop:
        epoch = start_epoch
        while epoch < limit:
            sampler.set_epoch(epoch)
            model.train()
            t0 = time.time()
            seen, loss_sum, n_loss = 0, 0.0, 0
            optim.zero_grad(set_to_none=True)
            for k, batch in enumerate(train_loader):
                ids = batch["input_ids"].to(device, non_blocking=True)
                att = batch["attention_mask"].to(device, non_blocking=True)
                lang = batch["lang"].to(device)
                b = {x: batch[x].to(device) for x in ("labels", "intensity", "node_mask")}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    out = model(ids, att, lang)
                prior = model.P_prior[lang].mean(0) if flags.get("n5_reg_gate") else None
                parts = total_loss(out, b, pos_weight, flags, cfg, prior)
                scaler.scale(parts["loss"] / cfg["grad_accum"]).backward()
                seen += len(ids)
                loss_sum += float(parts["loss"].detach())
                n_loss += 1
                last = k + 1 == len(sampler)
                if (k + 1) % cfg["grad_accum"] == 0 or last:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scale_before = scaler.get_scale()
                    scaler.step(optim)
                    scaler.update()
                    if not amp or scaler.get_scale() >= scale_before:
                        sched.step()
                    optim.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 100 == 0:
                    speed = seen / (time.time() - t0)
                    eta = (len(sampler) - k - 1) * cfg["batch_size"] / max(speed, 1e-6)
                    log.info("step %d loss %.4f samples/s %.1f eta_epoch %.0fs", global_step, loss_sum / n_loss, speed, eta)
                if stop.requested or (max_steps is not None and global_step >= max_steps):
                    break

            if stop.requested:
                status, stopped = "interrupted", True
                break
            partial = max_steps is not None and global_step >= max_steps
            train_seconds = time.time() - t0  # before validation, so samples/s measures training
            scores = validate(model, val_loader, val_ds, device, amp)
            score = float(scores["val_macro_f1"])
            elapsed = time.time() - t0
            improved = score > best
            if improved:
                best, best_epoch, bad = score, epoch, 0
                save_model(model, ckpt_best, meta)
            else:
                bad += 1
            if epoch + 1 == limit and "extend_to" in ep and limit < ep["extend_to"] and best_epoch == epoch:
                limit = ep["extend_to"]
                log.info("best epoch is the last one; extending to %d epochs", limit)
            atomic_save({"model": model.state_dict(), "optimizer": optim.state_dict(), "scheduler": sched.state_dict(),
                         "scaler": scaler.state_dict(), "epoch": epoch, "global_step": global_step, "best": best,
                         "best_epoch": best_epoch, "bad": bad, "limit": limit, "rng": rng_state()}, ckpt_last)
            with open(run / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch + 1, "train_loss": loss_sum / max(n_loss, 1), **scores,
                                    "time_s": round(elapsed, 1), "samples_per_s": round(seen / max(train_seconds, 1e-6), 1)}) + "\n")
            log.info("epoch %d val_macro_f1 %.4f val_pearson %s time %.0fs", epoch + 1, score, scores["val_pearson"], elapsed)
            write_status(run, status="running", last_epoch=epoch + 1, best_epoch=best_epoch + 1, best_score=best)
            epoch += 1
            if partial:
                status, stopped = "interrupted", True
                break
            if stop_after_epochs is not None and epoch >= stop_after_epochs:
                status, stopped = "interrupted", True
                break
            if bad >= ep["patience"]:
                log.info("early stop at epoch %d", epoch)
                break
    write_status(run, status=status)
    if stopped:
        log.info("stopped with status %s", status)
    return status


def log_decision(cfg: dict[str, Any], text: str) -> None:
    """Append one dated line to docs/DECISIONS.md."""
    path = Path(cfg.get("decisions_file", ROOT / "docs/DECISIONS.md"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- {date.today().isoformat()}: {text}\n")


def run_with_retries(cfg: dict[str, Any], run: Path, resume: bool, max_steps: int | None) -> str:
    """OOM rule of CLAUDE.md section 8: halve the batch, double accumulation, restart."""
    cfg_path = run / "config.yaml"
    if resume and cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    run.mkdir(parents=True, exist_ok=True)
    while True:
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        try:
            return train_run(cfg, run, resume=resume, max_steps=max_steps)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cfg["batch_size"] <= 4:
                write_status(run, status="failed", reason="CUDA OOM at batch size 4")
                log.error("CUDA OOM at batch size 4: stop and ask the user (CLAUDE.md section 8)")
                return "failed"
            cfg["batch_size"] //= 2
            cfg["grad_accum"] *= 2
            log_decision(cfg, f"run {cfg['exp']['id']}: CUDA OOM, batch_size {cfg['batch_size']}, "
                              f"grad_accum {cfg['grad_accum']}; restarted from scratch.")
            for f in ("ckpt_last.pt", "ckpt_best.pt"):
                (run / f).unlink(missing_ok=True)
            resume = False
        except Exception as exc:
            write_status(run, status="failed", reason=repr(exc)[:500])
            raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True, help="id from config/experiments.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N steps (smoke tests)")
    ap.add_argument("--run-id", default=None, help="write to runs/<run-id> instead of runs/<exp>")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = resolve(args.exp, args.config)
    run = Path(path_of(cfg, "runs", args.run_id or args.exp))
    state = run / "status.json"
    if state.exists() and json.loads(state.read_text()).get("status") == "done":
        print(f"{run.name}: already done, skipping")
        return
    if not args.resume and (run / "ckpt_last.pt").exists():
        raise SystemExit(f"{run} has a checkpoint: use --resume, or pick another --run-id")
    status = run_with_retries(cfg, run, args.resume, args.max_steps)
    sys.exit(0 if status in ("done", "interrupted") else 1)


if __name__ == "__main__":
    main()
