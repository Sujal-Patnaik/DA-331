"""M8 path end to end on synthetic data: train, predict, submission files, report.

Nobody can run M8 for the first time after hours of training and afford a crash, so this
drives the same code on the tiny CPU setup. It uses no official data.
"""
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from emoldgn import evaluate as E
from emoldgn.predict import predict_run
from emoldgn.train import train_run


def prepare(tiny_cfg, tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in tiny_cfg.items() if k != "exp"}), encoding="utf-8")
    run = Path(tiny_cfg["paths"]["runs"]) / "final_subset_s42"
    assert train_run(tiny_cfg, run) == "done"
    (run / "config.yaml").write_text(yaml.safe_dump(tiny_cfg), encoding="utf-8")
    return path, run


def test_predict_and_report(tiny_cfg, tmp_path):
    cfg_path, run = prepare(tiny_cfg, tmp_path)
    done = predict_run("final_subset_s42", config=str(cfg_path))
    assert done["val_rows"] > 0 and done["test_rows"] > 0
    th = json.loads((run / "thresholds.json").read_text())
    assert len(th["global"]) == 6 and th["intensity_rule"] in ("ordinal", "rounded")

    test = dict(np.load(run / "predictions/test.npz", allow_pickle=True))
    assert test["pred"].shape == (done["test_rows"], 6) and test["int_pred"].shape == test["pred"].shape
    # English has no disgust: it is neither predicted nor written to the submission.
    eng = test["lang"] == "eng"
    assert test["pred"][eng, 1].sum() == 0
    with open(run / "predictions/track_a/pred_eng.csv", encoding="utf-8") as f:
        assert next(csv.reader(f)) == ["id", "anger", "fear", "joy", "sadness", "surprise"]
    with open(run / "predictions/track_a/pred_hau.csv", encoding="utf-8") as f:
        assert next(csv.reader(f)) == ["id", "anger", "disgust", "fear", "joy", "sadness", "surprise"]

    # The test split is scored once: a second call must not rewrite it.
    stamp = (run / "predictions/test.npz").stat().st_mtime_ns
    assert "test_rows" not in predict_run("final_subset_s42", config=str(cfg_path))
    assert (run / "predictions/test.npz").stat().st_mtime_ns == stamp

    # Fill the other runs with this one's predictions so every report section executes.
    runs = run.parent
    for exp in ("b2_subset", "b4_subset_s42", "s1_subset", "s2_subset", "final_subset_s1", "final_subset_s2",
                "b4_subset_s1", "b4_subset_s2", "b4_full", "final_full"):
        shutil.copytree(run / "predictions", runs / exp / "predictions")
    shutil.copy(run / "predictions/test.npz", runs / "final_full/predictions/trackc.npz")
    report = E.build_report(tiny_cfg)
    for needle in ("Table 1", "Table 2", "Table 3", "Paired bootstrap", "co-occurrence fidelity", "ECE",
                   "Per-emotion F1", "Intensity confusion", "Track C zero-shot"):
        assert needle in report, needle
    assert "Stop rule" not in report


def test_report_works_before_any_run_finishes(tiny_cfg):
    report = E.build_report(tiny_cfg)
    assert "n/a" in report and "Not available" in report


def test_val_only_never_writes_test_predictions(tiny_cfg, tmp_path):
    cfg_path, run = prepare(tiny_cfg, tmp_path)
    predict_run("final_subset_s42", config=str(cfg_path), skip_test=True)
    assert (run / "predictions/val.npz").exists() and not (run / "predictions/test.npz").exists()


def test_stop_rule_is_flagged(tiny_cfg, tmp_path, monkeypatch):
    cfg_path, run = prepare(tiny_cfg, tmp_path)
    predict_run("final_subset_s42", config=str(cfg_path))
    runs = run.parent
    for exp in ("final_subset_s1", "final_subset_s2", "b4_subset_s42", "b4_subset_s1", "b4_subset_s2"):
        shutil.copytree(run / "predictions", runs / exp / "predictions")
    seen = {"n": 0}

    def fake(d):
        seen["n"] += 1
        return {"per_language": {}, "mean": 0.4 if seen["n"] <= 3 + 1 else 0.3}

    monkeypatch.setattr(E, "score_track_a", fake)
    assert "Stop rule" in E.build_report(tiny_cfg)
