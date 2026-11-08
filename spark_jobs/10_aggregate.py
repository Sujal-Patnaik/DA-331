"""Application aggregates (design document 9.1, 9.3, 9.4, 9.5; project specification, milestone M9).

Clause outputs become labels, states, urgency and valence; clauses roll up to documents with
the max rule; then aspect, period and group tables. min_group_size is enforced here, not
only in the UI, so a dashboard can never reveal a group below it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd  # module level: Spark resolves pandas UDF type hints from here

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_jobs.common import EMOS, ROOT, get_spark, job_args, load_config, path_of, setup_logging, skip_if_done

TABLES = ("clauses", "documents", "aspect_emotion", "period_trends", "groups", "group_aspect_emotion",
          "representative", "languages")


def derive_clause(cfg, profile: dict, pdf: pd.DataFrame) -> pd.DataFrame:
    """Binary emotions, states, urgency and valence for a batch of clauses (design document 9.3)."""
    import json

    import numpy as np

    from apps.engine import signals as S
    from apps.engine.aspects import cue_hits
    from emoldgn.thresholds import apply_thresholds

    art = Path(path_of(cfg, "artifacts"))
    th = json.loads((art / "thresholds.json").read_text(encoding="utf-8"))
    bank = json.loads((art / "lang_index.json").read_text(encoding="utf-8"))
    t_global = np.array(th["global"])
    t_lang = {k: np.array(v) for k, v in th["per_language"].items()}
    masks = bank.get("emotion_mask", {})
    langs = pdf["lang"].tolist()
    p = np.stack(pdf["probs"].to_numpy())
    inten = np.stack(pdf["inten"].to_numpy())
    t = S.app_thresholds(np.stack([t_lang.get(x, t_global) for x in langs]), langs, cfg["apps"]["threshold_overrides"])
    mask = np.array([masks.get(x, [1] * 6) for x in langs], dtype=float)
    neutral = p[:, 6] if bank.get("has_neutral") else None
    present = apply_thresholds(p[:, :6], t, mask, neutral)
    states = S.clause_states(inten, present, mask, float(cfg["apps"]["outrage_anger_intensity"]))
    keep = set(profile.get("states") or [])
    u = S.urgency(p, inten, mask)
    v, a = S.center_neutral(*S.vad_summary(p[:, :6], inten, S.load_vad(path_of(cfg, "lexicons")), mask), present)
    return pd.DataFrame({
        "present": list(present.astype(int)), "states": [[x for x in s if x in keep] for s in states],
        "urgency": u.astype(float), "valence": v.astype(float), "arousal": a.astype(float),
        "cues": [cue_hits(c, profile) for c in pdf["clause"]],
    })


def main() -> None:
    def extra(ap) -> None:
        ap.add_argument("--profile", required=True, choices=["customer_support", "student_feedback"])

    args = job_args(__doc__, extra)
    cfg = load_config(args.config)
    log = setup_logging(f"10_aggregate_{args.profile}", cfg)
    gold = lambda name: path_of(cfg, "app_gold", args.profile, name)  # noqa: E731
    if skip_if_done([gold(t) for t in TABLES], args.force, log):
        return

    from pyspark.sql import Window
    from pyspark.sql import functions as F

    from apps.engine.aspects import aspects_column, load_profile
    from apps.engine.signals import should_escalate

    profile = load_profile(ROOT / "profiles" / f"{args.profile}.yaml")
    min_group = int(profile.get("min_group_size", 1))
    spark = get_spark(f"10_aggregate_{args.profile}")

    @F.pandas_udf("struct<present:array<int>,states:array<string>,urgency:double,valence:double,arousal:double,cues:array<string>>")
    def derive(rows: pd.DataFrame) -> pd.DataFrame:
        return derive_clause(cfg, profile, rows)

    # Aspects are recomputed here, so a keyword change needs no new scoring pass.
    scored = spark.read.parquet(gold("scored_clauses")).withColumn("aspects", aspects_column(profile))
    clauses = scored.withColumn("d", derive(F.struct("clause", "lang", "probs", "inten"))).select("*", "d.*").drop("d", "ad")
    clauses.write.mode("overwrite").parquet(gold("clauses"))
    clauses = spark.read.parquet(gold("clauses")).cache()

    max_arrays = [F.max(F.col("probs")[k]).alias(f"p_{e}") for k, e in enumerate(EMOS)]
    max_arrays += [F.max(F.col("inten")[k]).alias(f"i_{e}") for k, e in enumerate(EMOS)]
    max_arrays += [F.max(F.col("present")[k]).alias(f"has_{e}") for k, e in enumerate(EMOS)]
    docs = clauses.groupBy("doc_id", "source", "group_id", "period", "lang").agg(
        *max_arrays,
        F.max("urgency").alias("urgency"), F.avg("valence").alias("valence"), F.avg("arousal").alias("arousal"),
        F.first("gold_sentiment").alias("gold_sentiment"), F.first("gold_aspects").alias("gold_aspects"),
        F.array_distinct(F.flatten(F.collect_list("states"))).alias("states"),
        F.array_distinct(F.flatten(F.collect_list("aspects"))).alias("aspects"),
        F.array_distinct(F.flatten(F.collect_list("cues"))).alias("cues"),
        F.count("*").alias("n_clauses"),
    )
    negative = F.greatest(*[F.col(f"has_{e}") for e in ("anger", "disgust", "fear", "sadness")])
    docs = docs.withColumn("mixed", (F.col("has_joy") == 1) & (negative == 1))
    if "mixed" in (profile.get("states") or []):
        docs = docs.withColumn("states", F.when(F.col("mixed"), F.array_union("states", F.array(F.lit("mixed")))).otherwise(F.col("states")))

    @F.pandas_udf("boolean")
    def escalate(u: pd.Series, s: pd.Series, a: pd.Series) -> pd.Series:
        return pd.Series([should_escalate(profile, float(x), set(y), set(z)) for x, y, z in zip(u, s, a)])

    docs = docs.withColumn("escalate", escalate("urgency", "states", "aspects"))
    docs.write.mode("overwrite").parquet(gold("documents"))
    docs = spark.read.parquet(gold("documents")).cache()

    long = clauses.select(
        "doc_id", "group_id", "period", "clause", "urgency", "probs", "present", "triggers", "states",
        F.explode("aspects").alias("aspect"),
    ).select("*", F.posexplode(F.array(*[F.lit(e) for e in EMOS])).alias("k", "emotion")).withColumn(
        "hit", F.col("present")[F.col("k")]).withColumn("p", F.col("probs")[F.col("k")]).drop("probs", "present").cache()

    group_sizes = docs.groupBy("group_id").agg(F.countDistinct("doc_id").alias("n_docs"))
    allowed = group_sizes.filter(F.col("n_docs") >= min_group).select("group_id")

    long.groupBy("aspect", "emotion").agg(
        F.sum("hit").alias("n_pos"), F.countDistinct("clause").alias("n_clauses")
    ).withColumn("share", F.col("n_pos") / F.col("n_clauses")).coalesce(1).write.mode("overwrite").parquet(gold("aspect_emotion"))

    long.join(allowed, "group_id").groupBy("group_id", "aspect", "emotion").agg(
        F.sum("hit").alias("n_pos"), F.count("*").alias("n_rows")
    ).withColumn("share", F.col("n_pos") / F.col("n_rows")).coalesce(1).write.mode("overwrite").parquet(gold("group_aspect_emotion"))

    state_rows = docs.select("period", "doc_id", F.explode("aspects").alias("aspect"), "states").select(
        "period", "doc_id", "aspect", F.explode(F.array_union("states", F.array(F.lit("any")))).alias("state"))
    totals = docs.select("period", F.explode("aspects").alias("aspect"), "doc_id").groupBy("period", "aspect").agg(
        F.countDistinct("doc_id").alias("n_docs"))
    trend = state_rows.groupBy("period", "aspect", "state").agg(F.countDistinct("doc_id").alias("n")).join(
        totals, ["period", "aspect"]).withColumn("share", F.col("n") / F.col("n_docs"))
    # Spike: share above the mean of the previous window by spike_sigma standard deviations.
    win = Window.partitionBy("aspect", "state").orderBy("period").rowsBetween(-int(cfg["apps"]["spike_window"]), -1)
    trend = (
        trend.withColumn("prev_mean", F.avg("share").over(win)).withColumn("prev_std", F.stddev("share").over(win))
        .withColumn("spike", F.col("share") > F.col("prev_mean") + float(cfg["apps"]["spike_sigma"]) * F.col("prev_std"))
        .fillna(False, ["spike"])
    )
    trend.coalesce(1).write.mode("overwrite").parquet(gold("period_trends"))

    n_states = docs.select("group_id", F.explode("states").alias("state")).groupBy("group_id", "state").count()
    top = n_states.withColumn("r", F.row_number().over(Window.partitionBy("group_id").orderBy(F.desc("count"), "state"))).filter("r <= 3")
    top = top.groupBy("group_id").agg(F.collect_list("state").alias("top_states"))
    groups = docs.join(allowed, "group_id").groupBy("group_id").agg(
        F.countDistinct("doc_id").alias("n_docs"), F.avg(F.col("mixed").cast("double")).alias("share_mixed"),
        F.avg("urgency").alias("mean_urgency"), F.avg("valence").alias("mean_valence"),
        F.avg(F.col("escalate").cast("double")).alias("share_escalated"),
        *[F.avg(f"has_{e}").alias(f"share_{e}") for e in EMOS],
    ).join(top, "group_id", "left")
    groups.coalesce(1).write.mode("overwrite").parquet(gold("groups"))

    rank = Window.partitionBy("group_id", "aspect", "emotion").orderBy(F.desc("p"))
    rep = (
        long.filter("hit = 1").join(allowed, "group_id").withColumn("r", F.row_number().over(rank))
        .filter(F.col("r") <= int(cfg["apps"]["representative_per_cell"]))
        .select("group_id", "aspect", "emotion", "clause", "p", "triggers", "r")
    )
    rep.coalesce(1).write.mode("overwrite").parquet(gold("representative"))

    docs.groupBy("lang").count().coalesce(1).write.mode("overwrite").parquet(gold("languages"))
    log.info("%s: %d documents, %d groups shown (min size %d), %d escalated", args.profile, docs.count(),
             allowed.count(), min_group, docs.filter("escalate").count())
    spark.stop()


if __name__ == "__main__":
    main()
