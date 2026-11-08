"""N4: shrinkage limits (plan 5.5)."""
import importlib

import numpy as np

priors = importlib.import_module("spark_jobs.06_priors")


def test_no_data_gives_the_family_prior():
    assert priors.shrink(0.9, 0.2, 0, 2000) == 0.2


def test_large_data_gives_the_language_statistic():
    assert abs(priors.shrink(0.9, 0.2, 1e12, 2000) - 0.9) < 1e-6


def test_lambda_formula():
    for n in (0, 1, 500, 2000, 10000):
        assert priors.lam(n, 2000) == n / (n + 2000)
    assert priors.lam(2000, 2000) == 0.5


def test_shrinkage_on_arrays_and_spark_columns(spark):
    from pyspark.sql import functions as F

    s, p = np.array([0.1, 0.5]), np.array([0.3, 0.3])
    assert np.allclose(priors.shrink(s, p, np.array([2000.0, 2000.0]), 2000), [0.2, 0.4])
    df = spark.createDataFrame([(0.1, 0.3, 2000.0)], ["s", "p", "n"])
    assert abs(df.select(priors.shrink(F.col("s"), F.col("p"), F.col("n"), 2000).alias("x")).first()["x"] - 0.2) < 1e-9
