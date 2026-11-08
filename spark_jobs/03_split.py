"""Stratified validation hold-out per language (design document 7.7, project specification, milestone M3).

Output data/gold/parts has one row per text with ``part``: train and val come from the
official train split, official dev joins val, test stays test. Track C languages are
never trained on and get part ``c_dev`` or ``test``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_jobs.common import EMOS, get_spark, job_args, load_config, path_of, setup_logging, skip_if_done


def main() -> None:
    args = job_args(__doc__)
    cfg = load_config(args.config)
    log = setup_logging("03_split", cfg)
    out = path_of(cfg, "gold", "parts")
    if skip_if_done([out], args.force, log):
        return

    from pyspark.sql import functions as F

    val_fraction = float(cfg["val_fraction"])
    seed = int(cfg["seed"])
    dev_in_val = bool(cfg["use_official_dev_in_val"])

    def assign_split(pdf: pd.DataFrame) -> pd.DataFrame:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        pdf = pdf.sort_values("id").reset_index(drop=True)
        pdf["part"] = "train"
        y = pdf[EMOS].fillna(0).astype(int).values
        _, va = next(msss.split(pdf, y))
        pdf.loc[va, "part"] = "val"
        return pdf[["id", "lang", "part"]]

    spark = get_spark("03_split")
    silver = spark.read.parquet(path_of(cfg, "silver", "brighter"))
    train = silver.filter("track = 'A' AND split = 'train'").select("id", "lang", *EMOS)
    parts = train.groupBy("lang").applyInPandas(assign_split, schema="id string, lang string, part string")

    rest = silver.filter("NOT (track = 'A' AND split = 'train')").select(
        "id", "lang",
        F.when(F.col("split") == "test", "test")
        .when((F.col("track") == "A") & F.lit(dev_in_val), "val")
        .when(F.col("track") == "A", "dev_unused")
        .otherwise("c_dev").alias("part"),
    )
    parts = parts.unionByName(rest).join(silver.select("id", "lang", "track", "split"), ["id", "lang"])
    parts.coalesce(4).write.mode("overwrite").parquet(out)

    parts = spark.read.parquet(out)
    if parts.count() != silver.count():
        raise ValueError("parts does not cover silver exactly")

    # Check: prevalence in train vs validation, with and without the official dev rows.
    labelled = silver.select("id", "lang", *EMOS).join(parts, ["id", "lang"]).filter("track = 'A'")
    rows = []
    for name, cond in [("holdout", "split = 'train'"), ("val_with_dev", "split IN ('train', 'dev')")]:
        prev = (
            labelled.filter(cond).filter("part IN ('train', 'val')").groupBy("lang", "part")
            .agg(*[F.avg(e).alias(e) for e in EMOS]).toPandas()
        )
        for lang, g in prev.groupby("lang"):
            g = g.set_index("part")
            for e in EMOS:
                if pd.notna(g.loc["train", e]):
                    diff = abs(g.loc["train", e] - g.loc["val", e]) * 100
                    rows.append({"view": name, "lang": lang, "emotion": e, "abs_diff_pp": round(float(diff), 3)})
    report = pd.DataFrame(rows)
    path = Path(path_of(cfg, "reports", "split_check.csv"))
    report.to_csv(path, index=False)
    for view, g in report.groupby("view"):
        log.info("%s: max prevalence difference %.2f pp; pairs over 3 pp: %d",
                 view, g["abs_diff_pp"].max(), int((g["abs_diff_pp"] >= 3).sum()))
    spark.stop()


if __name__ == "__main__":
    main()
