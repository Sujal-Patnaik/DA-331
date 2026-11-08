"""Application scoring job (design document 9.1, project specification, milestone M9).

Stages: ``prepare`` reads demo input, cleans it, detects language, segments clauses and
tags aspects, with no model involved. ``score`` adds model outputs in an iterator pandas
UDF. Only one task may hold the GPU, so with CUDA spark.task.cpus equals the core count
(project specification, section 5.2). On CPU each task loads a 1 GB model, so use a small --parallel on low RAM.

Inputs are never used for training. Text is masked before it is stored.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd  # module level: Spark resolves pandas UDF type hints from here

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_jobs.common import ROOT, get_spark, job_args, load_config, path_of, setup_logging, skip_if_done


def support_documents(spark, cfg, limit: int | None):
    """Customer turns of the Twitter support conversations, one document per conversation."""
    from pyspark.sql import functions as F

    src = path_of(cfg, "demo", "customer_support", "train.parquet")
    df = spark.read.parquet(src)
    if limit:
        df = df.orderBy(F.xxhash64("conversation_id", F.lit(int(cfg["seed"])))).limit(limit)
    turns = F.expr(r"filter(split(conversation, '\n(?=Customer: |Support: )'), x -> x LIKE 'Customer: %')")
    text = F.regexp_replace(F.array_join(F.transform(turns, lambda x: F.substring(x, 11, 100000)), "\n"), r"\s*\n\s*", "\n")
    docs = df.select(
        F.col("conversation_id").cast("string").alias("doc_id"), F.lit("twitter_support").alias("source"),
        F.col("company").alias("group_id"), text.alias("text_raw"),
        F.lit(None).cast("double").alias("gold_sentiment"), F.array().cast("array<string>").alias("gold_aspects"),
    ).filter(F.length("text_raw") > 0)
    # This mirror has no timestamps. Equal-size ordinal batches stand in for weeks, and
    # the dashboard labels them so; no calendar date is implied.
    n = int(cfg["apps"]["n_periods"])
    return docs.withColumn("period", F.format_string("batch_%02d", F.pmod(F.xxhash64("doc_id"), F.lit(n)).cast("int") + 1))


def feedback_documents(spark, cfg, limit: int | None):
    """EduRABSA reviews; the id encodes review type and teacher or course."""
    import yaml
    from pyspark.sql import functions as F

    with open(ROOT / "config/edurabsa_aspect_map.yaml", encoding="utf-8") as f:
        aspect_map = yaml.safe_load(f)
    base = path_of(cfg, "demo", "edurabsa_hf")
    pdf = pd.concat([pd.read_parquet(f"{base}/{s}.parquet") for s in ("train", "test")], ignore_index=True)
    score = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
    rows = []
    for r in pdf.itertuples():
        parts = str(r.original_id).split("_")
        kind = {"1": "teacher", "2": "course", "3": "university"}.get(parts[0], "other")
        group = {"teacher": f"teacher_{parts[1]}" if len(parts) > 1 else "teacher",
                 "course": f"course_{parts[2]}" if len(parts) > 2 else "course"}.get(kind, f"{kind}_reviews")
        quads = list(r.output)
        sentiments = [score[q["sentiment"]] for q in quads if q["sentiment"] in score]
        gold = sorted({aspect_map.get(q["category"], "other") for q in quads})
        rows.append((str(r.id), f"edurabsa_{kind}", group, r.text,
                     float(sum(sentiments) / len(sentiments)) if sentiments else None, gold))
    docs = spark.createDataFrame(
        rows, "doc_id string, source string, group_id string, text_raw string, gold_sentiment double, gold_aspects array<string>")
    extra = coursera_documents(spark, cfg)
    if extra is not None:
        docs = docs.unionByName(extra)
    if limit:
        docs = docs.orderBy(F.xxhash64("doc_id", F.lit(int(cfg["seed"])))).limit(limit)
    # EduRABSA has no term or date field, so semester comparison needs real institute data.
    return docs.withColumn("period", F.lit("all"))


def coursera_documents(spark, cfg):
    """Optional Coursera reviews (only if downloaded); stars give the valence check of design document section 9.6."""
    from pyspark.sql import functions as F

    files = sorted(Path(path_of(cfg, "demo", "coursera")).glob("*.csv"))
    for f in files:
        df = spark.read.option("header", True).option("multiLine", True).option("escape", '"').csv(str(f))
        if {"reviews", "rating", "course_id"} <= set(df.columns):
            return df.select(
                F.xxhash64("reviews", "reviewers", "date_reviews").cast("string").alias("doc_id") if "reviewers" in df.columns
                else F.xxhash64("reviews").cast("string").alias("doc_id"),
                F.lit("coursera").alias("source"), F.concat(F.lit("course_"), F.col("course_id")).alias("group_id"),
                F.col("reviews").alias("text_raw"), ((F.col("rating").cast("double") - 3.0) / 2.0).alias("gold_sentiment"),
                F.array().cast("array<string>").alias("gold_aspects"),
            ).filter(F.col("text_raw").isNotNull() & F.col("gold_sentiment").isNotNull())
    return None


LOADERS = {"customer_support": support_documents, "student_feedback": feedback_documents}


def prepare(cfg, profile_name: str, limit: int | None, force: bool, log) -> None:
    """Bronze documents, then silver clauses with language and aspects."""
    out = path_of(cfg, "app_silver", profile_name, "clauses")
    bronze = path_of(cfg, "app_bronze", profile_name, "documents")
    if skip_if_done([out, bronze], force, log):
        return

    from pyspark.sql import functions as F

    from apps.engine.aspects import aspects_column, load_profile

    profile = load_profile(ROOT / "profiles" / f"{profile_name}.yaml")
    lid_path = path_of(cfg, "lid_model")
    spark = get_spark(f"09_prepare_{profile_name}")

    @F.pandas_udf("string")
    def clean(s: pd.Series) -> pd.Series:
        from spark_jobs.textnorm import clean_text

        return s.map(clean_text)

    @F.pandas_udf("string")
    def language(s: pd.Series) -> pd.Series:
        from apps.engine.langid import detect

        return pd.Series(detect(s.tolist(), lid_path))

    @F.pandas_udf("array<string>")
    def clauses_of(s: pd.Series) -> pd.Series:
        from apps.engine.segment import segment

        return s.map(segment)

    LOADERS[profile_name](spark, cfg, limit).write.mode("overwrite").parquet(bronze)
    docs = spark.read.parquet(bronze).withColumn("text", clean("text_raw")).withColumn("lang", language("text"))
    clauses = docs.withColumn("clauses", clauses_of("text")).select(
        "doc_id", "source", "group_id", "period", "lang", "gold_sentiment", "gold_aspects",
        F.posexplode("clauses").alias("clause_id", "clause"))
    clauses = clauses.withColumn("aspects", aspects_column(profile))
    clauses.write.mode("overwrite").parquet(out)
    done = spark.read.parquet(out)
    log.info("%s: %d documents, %d clauses", profile_name, spark.read.parquet(bronze).count(), done.count())
    done.groupBy("lang").count().orderBy(F.desc("count")).show(8, truncate=False)
    spark.stop()


def score(cfg, profile_name: str, parallel: int | None, master: str | None, force: bool, log,
          table: str = "scored_clauses", sample: int | None = None) -> None:
    """Add model outputs to the clauses (design document 9.1 scoring job).

    ``table`` and ``sample`` let the scaling study write elsewhere, leaving results alone.
    """
    out = path_of(cfg, "app_gold", profile_name, table)
    if skip_if_done([out], force, log):
        return

    import torch
    from pyspark.sql import functions as F

    app = cfg["apps"]
    gpu = torch.cuda.is_available()
    cores = os.cpu_count() or 1
    tasks = parallel or (1 if gpu else 2)
    extra = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(app["max_records_per_batch"]),
             "spark.task.cpus": str(max(1, cores // tasks))}
    spark = get_spark(f"09_score_{profile_name}", master=master or f"local[{cores}]", extra=extra)
    artifacts = path_of(cfg, "artifacts")
    settings = {"batch_size": int(app["score_batch_size"]), "max_len": int(app["max_len"]), "top_triggers": int(app["top_triggers"])}

    @F.pandas_udf("struct<probs:array<float>,inten:array<float>,ad:array<float>,triggers:array<string>>")
    def model_scores(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        from apps.engine.scorer import get_scorer

        scorer = get_scorer(artifacts, **settings)
        for pdf in batches:
            r = scorer.score(pdf["clause"].tolist(), pdf["lang"].tolist())
            yield pd.DataFrame({"probs": list(r["probs"]), "inten": list(r["inten"]),
                                "ad": list(r["ad"].reshape(len(pdf), -1)), "triggers": r["triggers"]})

    clauses = spark.read.parquet(path_of(cfg, "app_silver", profile_name, "clauses"))
    if sample:
        clauses = clauses.limit(sample)
    scored = clauses.repartition(max(4, tasks)).withColumn("s", model_scores(F.struct("clause", "lang"))).select("*", "s.*").drop("s")
    scored.write.mode("overwrite").partitionBy("source", "period").parquet(out)
    log.info("%s: scored %d clauses", profile_name, spark.read.parquet(out).count())
    spark.stop()


def main() -> None:
    def extra(ap) -> None:
        ap.add_argument("--profile", required=True, choices=sorted(LOADERS))
        ap.add_argument("--stage", default="all", choices=["prepare", "score", "all"])
        ap.add_argument("--limit", type=int, default=None, help="documents; default 50k for support, all for feedback")
        ap.add_argument("--parallel", type=int, default=None, help="concurrent Spark tasks while scoring")
        ap.add_argument("--master", default=None, help="for the scaling study, for example local[4]")
        ap.add_argument("--table", default="scored_clauses", help="output table (scaling study only)")
        ap.add_argument("--sample", type=int, default=None, help="score only N clauses (scaling study only)")

    args = job_args(__doc__, extra)
    cfg = load_config(args.config)
    log = setup_logging(f"09_score_{args.profile}", cfg)
    limit = args.limit
    if limit is None and args.profile == "customer_support":
        limit = int(cfg["apps"]["sample_conversations"])
    if args.stage in ("prepare", "all"):
        prepare(cfg, args.profile, limit, args.force, log)
    if args.stage in ("score", "all"):
        score(cfg, args.profile, args.parallel, args.master, args.force, log, args.table, args.sample)


if __name__ == "__main__":
    main()
