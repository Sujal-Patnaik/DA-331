"""B1: TF-IDF plus one logistic regression per emotion in MLlib (design document 10.1, project specification, milestone M5).

Uses no GPU. Text keeps its case (rule 4). Each emotion's model trains on the training part
of the languages that have it. Thresholds use the same tuner as the neural models, on val.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_jobs.common import EMOS, get_spark, job_args, load_config, path_of, setup_logging, skip_if_done


def main() -> None:
    args = job_args(__doc__)
    cfg = load_config(args.config)
    log = setup_logging("08_baseline_tfidf", cfg)
    out_csv = path_of(cfg, "reports", "results_b1.csv")
    if not args.force and Path(out_csv).exists():
        log.info("exists, skipping: %s", out_csv)
        return

    import numpy as np
    import pandas as pd
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import IDF, HashingTF, RegexTokenizer
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F

    from emoldgn.evaluate import track_a
    from emoldgn.thresholds import apply_thresholds, thresholds_for, tune_thresholds

    spark = get_spark("08_baseline_tfidf")
    silver = spark.read.parquet(path_of(cfg, "silver", "brighter"))
    parts = spark.read.parquet(path_of(cfg, "gold", "parts"))
    rows = silver.join(parts.select("id", "lang", "part"), ["id", "lang"]).filter("track = 'A'")

    featurise = Pipeline(stages=[
        RegexTokenizer(inputCol="text", outputCol="tokens", pattern=r"\s+", toLowercase=False),
        HashingTF(inputCol="tokens", outputCol="tf", numFeatures=2**18),
        IDF(inputCol="tf", outputCol="features", minDocFreq=2),
    ]).fit(rows.filter("part = 'train'"))
    feats = featurise.transform(rows).select("id", "lang", "part", "features", *EMOS).cache()

    scored = feats.filter("part IN ('val', 'test')").select("id", "lang", "part", *EMOS)
    probs = {}
    for e in EMOS:
        train = feats.filter((F.col("part") == "train") & F.col(e).isNotNull()).withColumn("label", F.col(e).cast("double"))
        model = LogisticRegression(maxIter=50, regParam=0.01, elasticNetParam=0.0, standardization=False).fit(train)
        pred = model.transform(feats.filter("part IN ('val', 'test')")).select(
            "id", "lang", vector_to_array("probability").getItem(1).alias(f"p_{e}"))
        probs[e] = pred
        log.info("trained %s", e)
    for e in EMOS:
        scored = scored.join(probs[e], ["id", "lang"])
    pdf = scored.toPandas()

    def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
        y = frame[EMOS].astype(float).to_numpy()
        mask = (~np.isnan(y)).astype(float)
        return np.nan_to_num(y).astype(int), mask, frame[[f"p_{e}" for e in EMOS]].to_numpy(), frame["lang"].to_numpy()

    th = cfg["thresholds"]
    y_v, m_v, p_v, l_v = arrays(pdf[pdf["part"] == "val"])
    y_t, m_t, p_t, l_t = arrays(pdf[pdf["part"] == "test"])
    t_global, t_lang = tune_thresholds(p_v, y_v, m_v, l_v, th["grid_step"], th["min_pos_for_language"])
    rows_out = []
    for part, y, m, p, l in (("val", y_v, m_v, p_v, l_v), ("test", y_t, m_t, p_t, l_t)):
        s = track_a(y, apply_thresholds(p, thresholds_for(l, t_global, t_lang), m), m, l)
        rows_out += [{"part": part, "lang": k, "macro_f1": round(v, 4)} for k, v in s["per_language"].items()]
        log.info("B1 %s mean macro-F1: %.4f", part, s["mean"])
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    spark.stop()


if __name__ == "__main__":
    main()
