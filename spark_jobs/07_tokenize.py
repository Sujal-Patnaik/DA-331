"""Token-length statistics and tokenised Parquet for training (design document 6.5, project specification, milestone M3).

Rule: if the 95th percentile exceeds max_len for languages that together hold more than
10% of training rows, max_len becomes the smallest of 96 or 128 covering that percentile.
(No single language reaches 10% of 28, so the rule is read collectively; see DECISIONS.)
The resulting value is stored in data/gold/tokenizer_meta.json and read by train.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd  # module level: Spark resolves pandas UDF type hints from here

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_jobs.common import EMOS, get_spark, job_args, load_config, path_of, setup_logging, skip_if_done


def choose_max_len(table, base: int) -> int:
    """Apply the M3 rule to a frame with columns train_share and p95."""
    over = table[table["p95"] > base]
    if over["train_share"].sum() <= 0.10:
        return base
    need = over["p95"].max()
    return next((m for m in (96, 128) if m >= need), 128)


def main() -> None:
    args = job_args(__doc__)
    cfg = load_config(args.config)
    log = setup_logging("07_tokenize", cfg)
    out = path_of(cfg, "gold", "tokenized")
    meta_path = Path(path_of(cfg, "gold", "tokenizer_meta.json"))
    if meta_path.exists() and skip_if_done([out], args.force, log):
        return

    from pyspark.sql import functions as F

    encoder = cfg["encoder"]
    base_len = int(cfg["max_len"])
    spark = get_spark("07_tokenize")

    @F.pandas_udf("int")
    def token_len(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(encoder)
        for s in batches:
            enc = tok(s.tolist(), add_special_tokens=True, truncation=False)
            yield pd.Series([len(x) for x in enc["input_ids"]])

    silver = spark.read.parquet(path_of(cfg, "silver", "brighter"))
    parts = spark.read.parquet(path_of(cfg, "gold", "parts"))
    rows = silver.join(parts.select("id", "lang", "part"), ["id", "lang"]).withColumn("n_tokens", token_len("text")).cache()

    stats = (
        rows.groupBy("lang").agg(
            F.count("*").alias("rows"),
            F.expr("sum(if(part = 'train', 1, 0))").alias("train_rows"),
            *[F.expr(f"percentile(n_tokens, {q})").alias(f"p{int(q * 100)}") for q in (0.5, 0.9, 0.95, 0.99)],
            F.max("n_tokens").alias("max"),
        ).toPandas().sort_values("lang")
    )
    stats["train_share"] = stats["train_rows"] / stats["train_rows"].sum()
    stats.to_csv(path_of(cfg, "reports", "token_lengths.csv"), index=False)
    max_len = choose_max_len(stats, base_len)
    over = stats[stats["p95"] > base_len]
    log.info("languages with p95 > %d: %s (%.1f%% of training rows); max_len = %d",
             base_len, over["lang"].tolist(), over["train_share"].sum() * 100, max_len)

    @F.pandas_udf("struct<input_ids:array<int>,attention_mask:array<int>>")
    def tokenize(batches: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(encoder)
        for s in batches:
            enc = tok(s.tolist(), add_special_tokens=True, truncation=True, max_length=max_len)
            yield pd.DataFrame({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

    intensity = spark.read.parquet(path_of(cfg, "bronze", "intensity"))
    masks = spark.read.parquet(path_of(cfg, "bronze", "label_mask"))
    no_int = F.array(*[F.lit(-1) for _ in EMOS])
    tok = (
        rows.withColumn("enc", tokenize("text"))
        .join(intensity, ["lang", "id"], "left").join(masks, "lang", "left")
        .select(
            "id", "lang", "part", "track", "split",
            F.col("enc.input_ids").alias("input_ids"), F.col("enc.attention_mask").alias("attention_mask"),
            F.array(*[F.coalesce(F.col(e), F.lit(0)) for e in EMOS]).alias("labels"),
            F.col("mask").alias("label_mask"),
            F.coalesce(F.col("neutral"), F.lit(0)).alias("neutral"),
            F.coalesce(F.col("intensity"), no_int).alias("intensity"),
            F.least(F.col("n_tokens"), F.lit(max_len)).alias("length"),
        )
    )
    tok.repartition(8, "lang").write.mode("overwrite").parquet(out)
    meta_path.write_text(json.dumps({"max_len": max_len, "encoder": encoder}, indent=2) + "\n")
    written = spark.read.parquet(out)
    log.info("tokenised rows: %d; with intensity: %d", written.count(), written.filter("intensity[0] >= -1 AND array_max(intensity) >= 0").count())
    spark.stop()


if __name__ == "__main__":
    main()
