"""Graph invariants, on synthetic data and on the real bank once M4 has run."""
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from spark_jobs.common import EMOS, NODES, load_config, path_of

graphs = importlib.import_module("spark_jobs.04_graphs")
priors = importlib.import_module("spark_jobs.06_priors")


def synthetic(spark):
    rng = np.random.default_rng(1)
    rows, parts = [], []
    for lang in ("eng", "hau", "sun"):
        for i in range(120):
            labels = rng.integers(0, 2, 6).astype(int).tolist()
            if rng.random() < 0.3:
                labels = [0] * 6
            row = dict(zip(EMOS, labels))
            if lang == "eng":
                row["disgust"] = None
            neutral = int(not any(v for v in row.values() if v is not None))
            part = "train" if i % 5 else "val"
            rows.append((f"{lang}{i}", lang, *[row[e] for e in EMOS], neutral))
            parts.append((f"{lang}{i}", lang, part, "A"))
    schema = "id string, lang string, " + ", ".join(f"{e} int" for e in EMOS) + ", neutral int"
    silver = spark.createDataFrame(rows, schema)
    parts_df = spark.createDataFrame(parts, ["id", "lang", "part", "track"])
    masks = spark.createDataFrame(
        [("eng", [1, 0, 1, 1, 1, 1]), ("hau", [1] * 6), ("sun", [1] * 6), ("zul", [1] * 6)], ["lang", "mask"])
    return silver, parts_df, masks


FAM = {"eng": "indoeuropean", "hau": "afroasiatic", "sun": "austronesian", "zul": "nigercongo"}


def matrix_checks(cond, npmi, a_pos, a_neg, mask):
    assert cond.shape == (7, 7)
    assert (cond >= -1e-6).all() and (cond <= 1 + 1e-6).all()
    assert (npmi >= -1 - 1e-6).all() and (npmi <= 1 + 1e-6).all()
    assert (a_pos >= 0).all() and (a_neg >= 0).all()
    assert not ((a_pos > 0) & (a_neg > 0)).any()
    for k in np.flatnonzero(mask == 0):
        for m in (cond, npmi, a_pos, a_neg):
            assert (m[k] == 0).all() and (m[:, k] == 0).all()


def test_synthetic_graphs(spark):
    silver, parts, masks = synthetic(spark)
    g, _ = graphs.build_graphs(spark, silver, parts, masks, FAM, 1.0, 0.05, 0.10)
    g = g.cache()
    assert g.filter("scope_type = 'lang'").groupBy("scope").count().filter("count != 49").count() == 0
    assert g.filter("scope_type = 'global'").count() == 49
    assert g.filter("valid AND (p_j_given_i < 0 OR p_j_given_i > 1 OR npmi < -1 OR npmi > 1)").count() == 0
    # Neutral excludes every emotion by construction, so it is a hub of repelling edges.
    hub = g.filter("scope = 'hau' AND i = 'neutral' AND j != 'neutral'").collect()
    assert all(r["npmi"] < 0 and r["M_ij"] == 0 for r in hub)
    # English disgust must not contribute to family or global disgust counts.
    eng = g.filter("scope = 'eng' AND i = 'disgust'").collect()
    assert all(r["npmi"] == 0 and r["p_j_given_i"] == 0 for r in eng)


def test_synthetic_bank(spark):
    silver, parts, masks = synthetic(spark)
    g, train = graphs.build_graphs(spark, silver, parts, masks, FAM, 1.0, 0.05, 0.10)
    g = g.cache()
    sizes = (train.groupBy("lang").count().withColumnRenamed("lang", "scope"))
    from pyspark.sql import functions as F

    sizes = sizes.withColumn("scope_type", F.lit("lang"))
    fam_df = spark.createDataFrame(sorted(FAM.items()), ["lang", "family"])
    sizes = sizes.unionByName(
        train.join(fam_df, "lang").groupBy("family").count().withColumnRenamed("family", "scope").withColumn("scope_type", F.lit("family"))
    ).unionByName(train.agg(F.count("*").alias("count")).withColumn("scope", F.lit("global")).withColumn("scope_type", F.lit("global")))
    lp, fp = priors.hierarchical(g, sizes, fam_df, 2000.0, 0.05, 0.10)
    cfg = {"n4": {"kappa": 2000.0}}
    bank = priors.build_bank(spark, cfg, g, lp, fp, masks, FAM, sizes, (0.05, 0.10))
    assert bank["languages"] == ["eng", "hau", "sun", "zul", "<unk>"]
    for s in ("raw", "prior"):
        for x in range(5):
            matrix_checks(*[bank[f"{n}_{s}"][x].numpy() for n in ("cond", "npmi", "A_pos", "A_neg")], bank["mask"][x].numpy())
    # Unseen language with a family gets that family's prior; <unk> gets the global graph.
    assert bank["lambda_l"][3] == 0 and bank["lambda_l"][4] == 0


def test_real_bank():
    cfg = load_config()
    path = Path(path_of(cfg, "gold", "graph_bank.pt"))
    if not path.exists():
        pytest.skip("M4 has not run")
    bank = torch.load(path, weights_only=False)
    assert bank["nodes"] == NODES
    for s in ("raw", "prior"):
        assert bank[f"cond_{s}"].shape[1:] == (7, 7)
        for x in range(len(bank["languages"])):
            matrix_checks(*[bank[f"{n}_{s}"][x].numpy() for n in ("cond", "npmi", "A_pos", "A_neg")], bank["mask"][x].numpy())
    eng = bank["languages"].index("eng")
    assert bank["mask"][eng, 1] == 0
