"""Rule 3: graphs and priors see the training part and nothing else."""
import importlib
from pathlib import Path

import pandas as pd
import pytest

from spark_jobs.common import EMOS, load_config, path_of

graphs = importlib.import_module("spark_jobs.04_graphs")


def test_held_out_labels_cannot_change_a_graph(spark):
    rows = [(f"a{i}", "hau", 1, 0, 0, i % 2, 0, 0, 0) for i in range(40)]
    schema = "id string, lang string, " + ", ".join(f"{e} int" for e in EMOS) + ", neutral int"
    parts = spark.createDataFrame(
        [(f"a{i}", "hau", "train" if i < 30 else "val", "A") for i in range(40)], ["id", "lang", "part", "track"])
    masks = spark.createDataFrame([("hau", [1] * 6)], ["lang", "mask"])
    base = spark.createDataFrame(rows, schema)
    flipped = spark.createDataFrame([r if int(r[0][1:]) < 30 else (r[0], r[1], 0, 1, 1, 0, 1, 1, 0) for r in rows], schema)
    fam = {"hau": "afroasiatic"}

    def values(df):
        g, train = graphs.build_graphs(spark, df, parts, masks, fam, 1.0, 0.05, 0.10)
        assert train.count() == 30
        return sorted((r["scope"], r["i"], r["j"], r["M_ij"], round(r["npmi"], 9)) for r in g.collect())

    assert values(base) == values(flipped)


def test_graph_ids_are_training_part_ids_only():
    cfg = load_config()
    ids_path, parts_path = Path(path_of(cfg, "gold", "graph_input_ids")), Path(path_of(cfg, "gold", "parts"))
    if not (ids_path.exists() and parts_path.exists()):
        pytest.skip("M3 and M4 have not run")
    used = pd.read_parquet(ids_path)
    parts = pd.read_parquet(parts_path)
    key = lambda d: set(zip(d["lang"], d["id"]))  # noqa: E731
    train = key(parts[(parts["part"] == "train") & (parts["track"] == "A")])
    held = key(parts[parts["part"].isin(["val", "test", "c_dev"])])
    official_dev = key(parts[parts["split"] == "dev"])
    official_test = key(parts[parts["split"] == "test"])
    assert key(used) == train
    assert not key(used) & held
    assert not key(used) & official_dev
    assert not key(used) & official_test


def test_training_part_comes_from_official_train_only():
    cfg = load_config()
    parts_path = Path(path_of(cfg, "gold", "parts"))
    if not parts_path.exists():
        pytest.skip("M3 has not run")
    parts = pd.read_parquet(parts_path)
    assert set(parts[parts["part"] == "train"]["split"]) == {"train"}
    assert set(parts[parts["split"] == "test"]["part"]) == {"test"}
    assert parts[parts["track"] == "C"]["part"].isin(["c_dev", "test"]).all()
