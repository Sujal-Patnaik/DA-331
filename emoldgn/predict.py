"""Prediction files for val, test and Track C (plan 6.2, 10.4; CLAUDE.md M8).

Thresholds are tuned on validation here, then the test split is scored once. An existing
test prediction is never recomputed without --force, because that would be a second look
at the test split (rule 2). Outputs go to runs/<id>/predictions/; existing files in a
finished run are only added to.

Track C: only ind, jav, xho and zul have no training data, so only these are zero-shot.
They use their family prior through the bank and the pooled global thresholds.
"""
from __future__ import annotations

import argparse
import csv
import sys
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
from emoldgn.evaluate import track_b  # noqa: E402
from emoldgn.model import load_model  # noqa: E402
from emoldgn.thresholds import apply_thresholds, save_thresholds, thresholds_for, tune_thresholds  # noqa: E402
from emoldgn.train import PAD_ID, amp_enabled, data_languages, infer, load_bank, resolve  # noqa: E402
from spark_jobs.common import EMOS, path_of  # noqa: E402


def run_split(model: torch.nn.Module, ds: D.BrighterDataset, cfg: dict[str, Any], device: torch.device) -> dict[str, np.ndarray]:
    """Infer one dataset and return arrays in dataset order."""
    order = np.argsort(ds.length, kind="stable")
    size = cfg["eval_batch_size"]
    batches = [order[i:i + size].tolist() for i in range(0, len(order), size)]
    loader = DataLoader(ds, batch_sampler=batches, collate_fn=D.make_collate(PAD_ID))
    res = infer(model, loader, device, amp_enabled(cfg, device))
    res.update(id=np.array(ds.ids), lang=np.array(ds.lang_names), y=ds.labels[:, :6].astype(np.int64),
               mask=ds.node_mask[:, :6], intensity=ds.intensity)
    return res


def binary(res: dict[str, np.ndarray], t: np.ndarray, use_neutral: bool) -> np.ndarray:
    """Thresholded labels; the neutral node only votes in models that train it."""
    return apply_thresholds(res["p"][:, :6], t, res["mask"], res["p"][:, 6] if use_neutral else None)


def write_submission(res: dict[str, np.ndarray], out: Path, values: np.ndarray) -> None:
    """One pred_<lang>.csv per language in the official format (rows sorted by id)."""
    out.mkdir(parents=True, exist_ok=True)
    for name in sorted(set(res["lang"].tolist())):
        m = np.flatnonzero(res["lang"] == name)
        m = m[np.argsort(res["id"][m])]
        keep = [c for c in range(6) if res["mask"][m, c].max() > 0]
        with open(out / f"pred_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id"] + [EMOS[c] for c in keep])
            for r in m:
                w.writerow([res["id"][r]] + [int(values[r, c]) for c in keep])


def predict_run(exp: str, run_id: str | None = None, config: str | None = None, force: bool = False,
                skip_test: bool = False) -> dict[str, Any]:
    """Tune on val, score test once, and for final_full run Track C."""
    cfg = resolve(exp, config)
    run = Path(path_of(cfg, "runs", run_id or exp))
    if (run / "config.yaml").exists():
        cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    out = run / "predictions"
    out.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.get("force_cpu") else "cpu")
    model, _ = load_model(run / "ckpt_best.pt", str(device))
    bank = load_bank(cfg)
    pq = path_of(cfg, "gold", "tokenized")
    langs = data_languages(cfg)
    use_neutral = bool(cfg["exp"]["flags"].get("n2_signed"))
    has_int = bool(cfg["exp"]["flags"].get("n6_intensity"))

    val_ds = D.BrighterDataset(pq, ["val"], bank["languages"], bank["mask"], langs)
    val = run_split(model, val_ds, cfg, device)
    th = cfg["thresholds"]
    t_global, t_lang = tune_thresholds(val["p"][:, :6], val["y"], val["mask"], val["lang"], th["grid_step"],
                                       th["min_pos_for_language"])
    rule = None
    if has_int:
        # Plan 5.7: use whichever integer rule gives the higher validation Pearson.
        scores = {"ordinal": track_b(val["intensity"], val["ord_int"], val["lang"])["mean"],
                  "rounded": track_b(val["intensity"], np.rint(val["e_int"]), val["lang"])["mean"]}
        rule = max(scores, key=scores.get)
    save_thresholds(run / "thresholds.json", t_global, t_lang, intensity_rule=rule, encoder=cfg["encoder"], max_len=cfg["max_len"])
    val["pred"] = binary(val, thresholds_for(val["lang"], t_global, t_lang), use_neutral)
    if has_int:
        val["int_pred"] = val["ord_int"] if rule == "ordinal" else np.rint(val["e_int"]).astype(np.int64)
    np.savez_compressed(out / "val.npz", **val)
    done = {"val_rows": len(val["id"])}
    if skip_test:
        return done

    test_file = out / "test.npz"
    if test_file.exists() and not force:
        print(f"{exp}: test predictions exist; the test split is scored once, skipping")
    else:
        test_ds = D.BrighterDataset(pq, ["test"], bank["languages"], bank["mask"], langs)
        test = run_split(model, test_ds, cfg, device)
        test["pred"] = binary(test, thresholds_for(test["lang"], t_global, t_lang), use_neutral)
        if has_int:
            test["int_pred"] = test["ord_int"] if rule == "ordinal" else np.rint(test["e_int"]).astype(np.int64)
            write_submission(test, out / "track_b", test["int_pred"])
        write_submission(test, out / "track_a", test["pred"])
        np.savez_compressed(test_file, **test)
        done["test_rows"] = len(test["id"])

    if exp.startswith("final_full") or exp == "best_full":
        c_file = out / "trackc.npz"
        if force or not c_file.exists():
            c_ds = D.BrighterDataset(pq, ["test"], bank["languages"], bank["mask"], None, tracks=("C",))
            c = run_split(model, c_ds, cfg, device)
            c["pred"] = binary(c, np.tile(t_global, (len(c["id"]), 1)), use_neutral)
            write_submission(c, out / "track_c", c["pred"])
            np.savez_compressed(c_file, **c)
            done["trackc_rows"] = len(c["id"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="re-score test; only for a corrected pipeline bug")
    ap.add_argument("--val-only", action="store_true", help="never touch the test split")
    args = ap.parse_args()
    print(predict_run(args.exp, args.run_id, args.config, args.force, args.val_only))


if __name__ == "__main__":
    main()
