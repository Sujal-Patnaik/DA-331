"""A run killed after epoch 1 and resumed equals an uninterrupted run (CPU, fp32)."""
import json
from pathlib import Path

import pytest
import torch

from emoldgn.train import train_run


@pytest.mark.parametrize("exp", ["final_subset_s42"])
def test_resume_matches_uninterrupted(tiny_cfg, exp):
    runs = Path(tiny_cfg["paths"]["runs"])
    assert train_run(tiny_cfg, runs / "straight") == "done"
    assert train_run(tiny_cfg, runs / "killed", stop_after_epochs=1) == "interrupted"
    status = json.loads((runs / "killed/status.json").read_text())
    assert status["status"] == "interrupted" and status["last_epoch"] == 1
    assert train_run(tiny_cfg, runs / "killed", resume=True) == "done"

    a = torch.load(runs / "straight/ckpt_last.pt", weights_only=False)
    b = torch.load(runs / "killed/ckpt_last.pt", weights_only=False)
    assert a["global_step"] == b["global_step"] and a["epoch"] == b["epoch"] == 1
    for k, v in a["model"].items():
        assert torch.allclose(v.float(), b["model"][k].float(), atol=1e-4), k
    assert len((runs / "killed/metrics.jsonl").read_text().splitlines()) == 2


def test_run_folder_contents(tiny_cfg):
    run = Path(tiny_cfg["paths"]["runs"]) / "contents"
    train_run(tiny_cfg, run)
    for name in ("metrics.jsonl", "ckpt_last.pt", "ckpt_best.pt", "status.json", "train.log"):
        assert (run / name).exists(), name
    line = json.loads((run / "metrics.jsonl").read_text().splitlines()[0])
    assert {"train_loss", "val_macro_f1", "val_pearson", "time_s"} <= set(line)
    state = json.loads((run / "status.json").read_text())
    assert state["status"] == "done" and state["best_epoch"] >= 1


def test_extension_rule_applies_when_the_last_epoch_is_best(tiny_cfg, monkeypatch):
    import emoldgn.train as T

    tiny_cfg["epochs"]["subset"] = {"max": 2, "patience": 9, "extend_to": 3}
    scores = iter([0.1, 0.2, 0.3])
    monkeypatch.setattr(T, "validate", lambda *a, **k: {"val_macro_f1": next(scores), "val_pearson": None})
    run = Path(tiny_cfg["paths"]["runs"]) / "extended"
    assert train_run(tiny_cfg, run) == "done"
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 3


def test_early_stopping(tiny_cfg, monkeypatch):
    import emoldgn.train as T

    tiny_cfg["epochs"]["subset"] = {"max": 9, "patience": 1}
    scores = iter([0.5, 0.1, 0.1, 0.1])
    monkeypatch.setattr(T, "validate", lambda *a, **k: {"val_macro_f1": next(scores), "val_pearson": None})
    run = Path(tiny_cfg["paths"]["runs"]) / "stopped"
    assert train_run(tiny_cfg, run) == "done"
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 2


def test_out_of_memory_halves_the_batch_and_restarts(tiny_cfg, monkeypatch, tmp_path):
    import emoldgn.train as T

    tiny_cfg["decisions_file"] = str(tmp_path / "DECISIONS.md")
    tiny_cfg["batch_size"] = 8
    calls = []

    def flaky(cfg, run, resume=False, max_steps=None):
        calls.append((cfg["batch_size"], cfg["grad_accum"], resume))
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("boom")
        return "done"

    monkeypatch.setattr(T, "train_run", flaky)
    run = Path(tiny_cfg["paths"]["runs"]) / "oom"
    assert T.run_with_retries(tiny_cfg, run, False, None) == "done"
    assert calls == [(8, 1, False), (4, 2, False)]
    assert "CUDA OOM" in (tmp_path / "DECISIONS.md").read_text()


def test_sampler_quota_and_temperature(tiny_cfg):
    from emoldgn import data as D
    from tests.conftest import fake_bank

    bank = fake_bank()
    ds = D.BrighterDataset(Path(tiny_cfg["paths"]["gold"]) / "tokenized", ["train"], bank["languages"], bank["mask"])
    s = D.TemperatureBatchSampler(ds, D.SamplerConfig(batch_size=4, quota=2, seed=1))
    batches = list(s)
    assert len(batches) == len(s)
    assert all(len(b) == 4 and ds.has_intensity[b].sum() >= 2 for b in batches)
    s.set_epoch(1)
    first = batches
    s.set_epoch(0)
    assert list(s) == first
    assert D.intensity_quota(16, 32) == 8


def test_intensity_quota_can_be_switched_off(tiny_cfg):
    from emoldgn.train import load_bank, make_loaders

    bank = load_bank(tiny_cfg)
    assert make_loaders(tiny_cfg, bank)[2].cfg.quota == 2
    tiny_cfg["exp"]["intensity_quota"] = False
    assert make_loaders(tiny_cfg, bank)[2].cfg.quota == 0
