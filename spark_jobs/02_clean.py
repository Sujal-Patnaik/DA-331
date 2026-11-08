"""Bronze to silver: emoji/emoticons to words, PII masks, NFC (project specification, milestone M2, design document 7.6).

No lowercasing and no stop-word removal (rule 4). The original is kept in ``text_raw``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd  # module level: Spark resolves pandas UDF type hints from here

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_jobs.common import get_spark, job_args, load_config, path_of, setup_logging, skip_if_done


def main() -> None:
    args = job_args(__doc__)
    cfg = load_config(args.config)
    log = setup_logging("02_clean", cfg)
    out = path_of(cfg, "silver", "brighter")
    if skip_if_done([out], args.force, log):
        return

    from pyspark.sql import functions as F

    spark = get_spark("02_clean")

    @F.pandas_udf("string")
    def normalise(s: pd.Series) -> pd.Series:
        from spark_jobs.textnorm import clean_text

        return s.map(clean_text)

    bronze = spark.read.parquet(path_of(cfg, "bronze", "brighter"))
    silver = bronze.withColumnRenamed("text", "text_raw").withColumn("text", normalise("text_raw"))
    silver.write.mode("overwrite").parquet(out)

    silver = spark.read.parquet(out)
    b = {(r["lang"], r["split"]): r["count"] for r in bronze.groupBy("lang", "split").count().collect()}
    s = {(r["lang"], r["split"]): r["count"] for r in silver.groupBy("lang", "split").count().collect()}
    if b != s:
        raise ValueError(f"row counts differ between bronze and silver: {set(b.items()) ^ set(s.items())}")
    nulls = silver.filter(F.col("text").isNull() | (F.length("text") == 0)).count()
    if nulls:
        raise ValueError(f"{nulls} rows have empty text after cleaning")
    log.info("silver rows match bronze for %d language/splits; no empty text", len(b))
    changed = silver.filter(F.col("text") != F.col("text_raw")).count()
    log.info("rows changed by cleaning: %d of %d", changed, silver.count())
    # Before/after check of one Track A example, written to the local log only.
    for r in (silver.filter("text_raw RLIKE '[\\\\x{1F300}-\\\\x{1FAFF}]'").select("text_raw", "text").limit(1).collect()):
        log.info("emoji example before: %s", r["text_raw"][:160])
        log.info("emoji example after : %s", r["text"][:160])
    spark.stop()


if __name__ == "__main__":
    main()
