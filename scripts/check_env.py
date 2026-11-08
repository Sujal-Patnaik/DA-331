#!/usr/bin/env python
"""M0 environment check; ends with ALL CHECKS PASSED."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd  # module level: Spark resolves pandas UDF type hints from here

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        results.append((name, True, str(fn())))
    except Exception as exc:  # noqa: BLE001 - every failure is reported, not raised
        results.append((name, False, repr(exc)[:300]))


def python_version() -> str:
    assert sys.version_info >= (3, 10), "Python 3.10+ required"
    return sys.version.split()[0]


def gpu() -> str:
    import torch

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    p = torch.cuda.get_device_properties(0)
    x = torch.ones(4, device="cuda").sum().item()
    assert x == 4
    return f"{p.name}, {p.total_memory / 1024**3:.1f} GB, torch {torch.__version__}, CUDA {torch.version.cuda}"


def spark_roundtrip() -> str:
    from spark_jobs.common import get_spark

    spark = get_spark("check_env", extra={"spark.ui.enabled": "false"})
    with tempfile.TemporaryDirectory() as d:
        spark.createDataFrame([(1, "a"), (2, "b")], ["x", "y"]).write.mode("overwrite").parquet(d + "/t")
        n = spark.read.parquet(d + "/t").count()
    version = spark.version
    jvm = spark.sparkContext._jvm.System.getProperty("java.version")
    spark.stop()
    assert n == 2
    return f"Spark {version}, Java {jvm}, 2-row Parquet round-trip ok"


def pandas_udf() -> str:
    from pyspark.sql import functions as F

    from spark_jobs.common import get_spark

    spark = get_spark("check_env", extra={"spark.ui.enabled": "false"})

    @F.pandas_udf("long")
    def plus(s: pd.Series) -> pd.Series:
        return s + 1

    total = spark.range(4).select(plus("id").alias("x")).agg(F.sum("x")).first()[0]
    spark.stop()
    assert total == 10
    return "pandas UDF ok"


def tokenizer() -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return f"xlm-roberta-base tokenizer ok, {len(tok('I enjoyed the course')['input_ids'])} tokens"


def main() -> int:
    check("python", python_version)
    check("gpu", gpu)
    check("spark", spark_roundtrip)
    check("pandas_udf", pandas_udf)
    check("tokenizer", tokenizer)
    for name, ok, msg in results:
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {msg}")
    if all(ok for _, ok, _ in results):
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
