"""Label graphs from the training part only (design document, 5.3, 7.1, 7.2, 7.8; project rule 3).

For every language, family and the global scope this computes the B4 conditional
probability P(j|i) and the N2 signed NPMI with add-k smoothing and a neutral node.
Pairs involving an emotion a language does not have are excluded from family and global
counts and are zero in that language's matrices (rule 5).

Output: data/gold/graphs, data/gold/graph_scope_sizes, and data/gold/graph_input_ids,
the exact ids aggregated, which test_no_leakage.py audits.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_jobs.common import EMOS, NODES, families, get_spark, job_args, load_config, path_of, setup_logging, skip_if_done


def training_rows(silver, parts):
    """The only rows graphs may see: Track A rows in the training part (rule 3)."""
    keep = parts.filter("part = 'train' AND track = 'A'").select("id", "lang")
    return silver.join(keep, ["id", "lang"])


def lang_counts(spark, train, masks):
    """Per-language pair counts on the complete 7 x 7 grid (design document, 7.1, 7.2)."""
    from pyspark.sql import functions as F

    pos = (
        train.withColumn("neutral", F.col("neutral").cast("int"))
        .select("lang", "id", F.explode(F.array(*[F.when(F.col(n) == 1, F.lit(n)) for n in NODES])).alias("emo"))
        .filter("emo IS NOT NULL")
    )
    pairs = (
        pos.alias("a").join(pos.alias("b"), ["lang", "id"])
        .groupBy("lang", F.col("a.emo").alias("i"), F.col("b.emo").alias("j"))
        .agg(F.count("*").alias("M_ij"))
    )
    n = train.groupBy("lang").agg(F.count("*").alias("N"))
    cnt = pos.groupBy("lang", "emo").agg(F.count("*").alias("M"))
    nodes = spark.createDataFrame([(i, j) for i in NODES for j in NODES], ["i", "j"])
    node_mask = (
        masks.select("lang", F.posexplode(F.concat(F.col("mask"), F.array(F.lit(1)))).alias("pos", "present"))
        .withColumn("node", F.element_at(F.array(*[F.lit(x) for x in NODES]), F.col("pos") + 1))
    )
    grid = (
        n.crossJoin(nodes)
        .join(node_mask.select("lang", F.col("node").alias("i"), F.col("present").alias("pi")), ["lang", "i"])
        .join(node_mask.select("lang", F.col("node").alias("j"), F.col("present").alias("pj")), ["lang", "j"])
        .withColumn("valid", (F.col("pi") == 1) & (F.col("pj") == 1))
        .drop("pi", "pj")
    )
    return (
        grid.join(pairs, ["lang", "i", "j"], "left").fillna(0, ["M_ij"])
        .join(cnt.withColumnRenamed("emo", "i").withColumnRenamed("M", "Mi"), ["lang", "i"], "left")
        .join(cnt.withColumnRenamed("emo", "j").withColumnRenamed("M", "Mj"), ["lang", "j"], "left")
        .fillna(0, ["Mi", "Mj"])
    )


def add_stats(df, k: float, tau_pos: float, tau_neg: float):
    """NPMI, P(j|i), A+ and A- from the count columns (design document, 5.3, 7.2)."""
    from pyspark.sql import functions as F

    c = len(NODES)
    off_diag = F.col("i") != F.col("j")
    return (
        df.withColumn("Pij", (F.col("M_ij") + k) / (F.col("N") + k * c**2))
        .withColumn("Pi", (F.col("Mi") + k) / (F.col("N") + k * c))
        .withColumn("Pj", (F.col("Mj") + k) / (F.col("N") + k * c))
        .withColumn("npmi_raw", F.log(F.col("Pij") / (F.col("Pi") * F.col("Pj"))) / -F.log("Pij"))
        .withColumn("npmi", F.when(F.col("valid"), F.greatest(F.lit(-1.0), F.least(F.lit(1.0), F.col("npmi_raw")))).otherwise(0.0))
        .withColumn("p_j_given_i", F.when(F.col("valid"), F.col("M_ij") / F.greatest(F.col("Mi"), F.lit(1))).otherwise(0.0))
        .withColumn("a_pos", F.when(off_diag, F.greatest(F.col("npmi") - tau_pos, F.lit(0.0))).otherwise(0.0))
        .withColumn("a_neg", F.when(off_diag, F.greatest(-F.col("npmi") - tau_neg, F.lit(0.0))).otherwise(0.0))
        .drop("Pij", "Pi", "Pj", "npmi_raw")
    )


def scope_counts(counts, scope_type: str, scope_col):
    """Sum language counts over a scope, only across languages that have both nodes."""
    from pyspark.sql import functions as F

    grouped = (
        counts.filter("valid").withColumn("scope", scope_col).groupBy("scope", "i", "j")
        .agg(*[F.sum(c).alias(c) for c in ("M_ij", "Mi", "Mj", "N")])
        .withColumn("valid", F.lit(True))
    )
    return grouped.withColumn("scope_type", F.lit(scope_type))


def build_graphs(spark, silver, parts, masks, fam: dict[str, str], k: float, tau_pos: float, tau_neg: float):
    """All scopes in one long table; returns (graphs, training rows)."""
    from pyspark.sql import functions as F

    train = training_rows(silver, parts).cache()
    counts = lang_counts(spark, train, masks).cache()
    fam_df = spark.createDataFrame(sorted(fam.items()), ["lang", "family"])
    lang_level = counts.withColumn("scope", F.col("lang")).withColumn("scope_type", F.lit("lang")).drop("lang")
    fam_level = scope_counts(counts.join(fam_df, "lang"), "family", F.col("family"))
    global_level = scope_counts(counts, "global", F.lit("global"))
    cols = ["scope_type", "scope", "i", "j", "M_ij", "Mi", "Mj", "N", "valid"]
    long = lang_level.select(*cols).unionByName(fam_level.select(*cols)).unionByName(global_level.select(*cols))
    return add_stats(long, k, tau_pos, tau_neg), train


def main() -> None:
    args = job_args(__doc__)
    cfg = load_config(args.config)
    log = setup_logging("04_graphs", cfg)
    out = path_of(cfg, "gold", "graphs")
    out_ids = path_of(cfg, "gold", "graph_input_ids")
    out_sizes = path_of(cfg, "gold", "graph_scope_sizes")
    if skip_if_done([out, out_ids, out_sizes], args.force, log):
        return

    from pyspark.sql import functions as F

    spark = get_spark("04_graphs")
    silver = spark.read.parquet(path_of(cfg, "silver", "brighter"))
    parts = spark.read.parquet(path_of(cfg, "gold", "parts"))
    masks = spark.read.parquet(path_of(cfg, "bronze", "label_mask"))
    fam = families(cfg)
    n2 = cfg["n2"]
    graphs, train = build_graphs(spark, silver, parts, masks, fam, float(n2["smoothing_k"]),
                                 float(n2["tau_pos"]), float(n2["tau_neg"]))
    graphs.coalesce(4).write.mode("overwrite").parquet(out)
    train.select("id", "lang").coalesce(2).write.mode("overwrite").parquet(out_ids)

    # Scope sizes for N4: one n per scope, counted once per text.
    fam_df = spark.createDataFrame(sorted(fam.items()), ["lang", "family"])
    sizes = (
        train.groupBy("lang").count().withColumn("scope_type", F.lit("lang")).withColumnRenamed("lang", "scope")
        .unionByName(train.join(fam_df, "lang").groupBy("family").count()
                     .withColumn("scope_type", F.lit("family")).withColumnRenamed("family", "scope"))
        .unionByName(train.agg(F.count("*").alias("count"))
                     .withColumn("scope_type", F.lit("global")).withColumn("scope", F.lit("global")))
    )
    sizes.coalesce(1).write.mode("overwrite").parquet(out_sizes)

    g = spark.read.parquet(out)
    n_scopes = g.select("scope_type", "scope").distinct().groupBy("scope_type").count().collect()
    log.info("graph scopes: %s; emotions: %s", {r["scope_type"]: r["count"] for r in n_scopes}, EMOS)
    spark.stop()


if __name__ == "__main__":
    main()
