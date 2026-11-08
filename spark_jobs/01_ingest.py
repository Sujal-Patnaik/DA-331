"""Raw BRIGHTER files to data/bronze (project specification, milestone M2; design document 7.1, 7.5).

Writes three tables: brighter (one row per text), intensity, label_mask.

Source rules, from M1 discovery (reports/DATA_NOTES.md):
* SemEval files are the source of truth. Only the official per-language files are read;
  ``track_a/train/ron_track1_test.csv`` is a stray copy with another schema and is ignored.
* Blank test labels (English Track A) are filled from SemEval's Track C and B files.
* Track C rows of Track A languages are re-identified copies of the Track A test set, so
  only the languages that have no Track A data become ``track = C``.
* Intensities are joined on text, not id. Track A and B ids do not address the same text
  for amh, rus and part of eng, and Track B's amh/rus splits differ from Track A's. Each
  row therefore keeps its Track A split, which also prevents test texts entering training.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_jobs.common import EMOS, SPLITS, get_spark, job_args, load_config, path_of, setup_logging, skip_if_done

DATASET = "semeval_repo/task-dataset/semeval-2025-task11-dataset"


def lang_files(root: Path, track: str, split: str) -> dict[str, Path]:
    """Official per-language CSVs; anything with an underscore in its stem is not one."""
    folder = root / track / split
    return {p.stem: p for p in sorted(folder.glob("*.csv")) if "_" not in p.stem}


def read_csv(spark, path: Path):
    """Read one CSV with quoted newlines handled and the header as written."""
    return (
        spark.read.option("header", True).option("multiLine", True).option("escape", '"')
        .option("quote", '"').option("encoding", "UTF-8").csv(str(path))
    )


def normalise(df, lang: str, split: str, track: str):
    """Conform one file to the bronze schema; emotions missing from the file become null."""
    from pyspark.sql import functions as F

    cols = {c.lower(): c for c in df.columns}
    out = [F.col(cols["id"]).alias("id"), F.lit(lang).alias("lang"), F.lit(split).alias("split"),
           F.lit(track).alias("track"), F.col(cols["text"]).alias("text")]
    for e in EMOS:
        out.append(F.col(cols[e]).cast("int").alias(e) if e in cols else F.lit(None).cast("int").alias(e))
    return df.select(*out).withColumn("source", F.lit("semeval"))


def fill_blank_labels(spark, df, base: Path, masks: dict[str, list[int]], log):
    """Fill wholly unlabelled language/splits from SemEval's own Track C and B files.

    English Track A test is blank. Track C holds the official binary labels for the same
    texts; a text duplicated there with conflicting labels falls back to the row-aligned
    Track B intensity (> 0 means present). The Hugging Face copy is not used: its English
    test split has 5,528 rows and only 188 texts match the SemEval text exactly.
    """
    from pyspark.sql import functions as F

    blank = (
        df.withColumn("has", F.coalesce(*[F.col(e) for e in EMOS]).isNotNull())
        .groupBy("lang", "split").agg(F.max("has").alias("labelled")).filter("NOT labelled").collect()
    )
    for row in blank:
        lang, split = row["lang"], row["split"]
        present = [e for e, m in zip(EMOS, masks[lang]) if m]
        c_file, b_file = base / "track_c" / split / f"{lang}.csv", base / "track_b" / split / f"{lang}.csv"
        if not (c_file.exists() and b_file.exists()):
            raise FileNotFoundError(f"{lang}/{split} has no labels and no Track C and B files: stop and ask")
        vec = F.array(*[F.col(e).cast("int") for e in present])
        c = (
            read_csv(spark, c_file).withColumn("vec", vec).groupBy("text")
            .agg(F.collect_set("vec").alias("labels")).filter(F.size("labels") == 1)
            .select("text", F.col("labels")[0].alias("c_vec"))
        )
        b = read_csv(spark, b_file).select(
            F.regexp_replace("id", "track_b", "track_a").alias("id"), "text",
            F.array(*[(F.col(e).cast("int") > 0).cast("int") for e in present]).alias("b_vec"),
        )
        this = (F.col("lang") == lang) & (F.col("split") == split)
        part = df.filter(this).join(c, "text", "left").join(b, ["id", "text"], "left")
        part = part.withColumn("final", F.coalesce("c_vec", "b_vec"))
        unmatched = part.filter(F.col("final").isNull()).count()
        from_b = part.filter(F.col("c_vec").isNull() & F.col("b_vec").isNotNull()).count()
        if unmatched:
            raise ValueError(f"{lang}/{split}: {unmatched} rows have no label in any SemEval file: stop and ask")
        log.info("%s/%s: blank Track A labels filled from Track C; %d rows from Track B", lang, split, from_b)
        filled = part
        for e in EMOS:
            if e in present:
                filled = filled.withColumn(e, F.col("final")[present.index(e)])
        filled = filled.withColumn("source", F.lit("semeval_track_c")).select(*df.columns)
        df = df.filter(~this).unionByName(filled)
    return df


def intensity_table(spark, base: Path, bronze, log):
    """Track B intensities attached to Track A ids by (lang, text) (design document, 7.5)."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    frames = []
    for split in SPLITS:
        for lang, path in lang_files(base, "track_b", split).items():
            df = read_csv(spark, path)
            cols = {c.lower(): c for c in df.columns}
            selected = [F.col(cols["text"]).alias("text")]
            for e in EMOS:
                selected.append(F.col(cols[e]).cast("int").alias(e) if e in cols else F.lit(-1).alias(e))
            frames.append(df.select(*selected).withColumn("lang", F.lit(lang)))
    b = frames[0]
    for f in frames[1:]:
        b = b.unionByName(f)
    b = b.withColumn("intensity", F.array(*[F.coalesce(F.col(e), F.lit(-1)) for e in EMOS]))
    # One intensity per distinct text, chosen deterministically.
    w = Window.partitionBy("lang", "text").orderBy(F.col("intensity").cast("string"))
    b = b.withColumn("rn", F.row_number().over(w)).filter("rn = 1").select("lang", "text", "intensity")
    out = bronze.select("id", "lang", "text").join(b, ["lang", "text"]).select("lang", "id", "intensity")
    log.info("intensity rows joined: %d", out.count())
    return out


def main() -> None:
    args = job_args(__doc__)
    cfg = load_config(args.config)
    log = setup_logging("01_ingest", cfg)
    out_main = path_of(cfg, "bronze", "brighter")
    out_int = path_of(cfg, "bronze", "intensity")
    out_mask = path_of(cfg, "bronze", "label_mask")
    if skip_if_done([out_main, out_int, out_mask], args.force, log):
        return

    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

    spark = get_spark("01_ingest")
    raw = Path(path_of(cfg, "raw"))
    base = raw / DATASET
    frames, masks = [], {}
    train_langs: set[str] = set()
    for split in SPLITS:
        for lang, path in lang_files(base, "track_a", split).items():
            df = read_csv(spark, path)
            frames.append(normalise(df, lang, split, "A"))
            train_langs.add(lang)
            masks[lang] = [int(e in {c.lower() for c in df.columns}) for e in EMOS]
    for split in ("dev", "test"):
        for lang, path in lang_files(base, "track_c", split).items():
            if lang in train_langs:
                continue
            df = read_csv(spark, path)
            frames.append(normalise(df, lang, split, "C"))
            masks[lang] = [int(e in {c.lower() for c in df.columns}) for e in EMOS]

    df = frames[0]
    for f in frames[1:]:
        df = df.unionByName(f)
    df = fill_blank_labels(spark, df, base, masks, log)

    df = df.withColumn(
        "neutral",
        F.when(F.expr("size(filter(array(%s), x -> x IS NOT NULL)) = 0" % ",".join(EMOS)), F.lit(None).cast("int"))
        .when(F.expr("aggregate(filter(array(%s), x -> x IS NOT NULL), 0, (a, x) -> a + x) = 0" % ",".join(EMOS)), 1)
        .otherwise(0),
    )
    df.repartition(8).write.mode("overwrite").parquet(out_main)
    bronze = spark.read.parquet(out_main).filter("track = 'A'")

    intensity_table(spark, base, bronze, log).write.mode("overwrite").parquet(out_int)

    schema = StructType([StructField("lang", StringType()), StructField("mask", ArrayType(IntegerType()))])
    spark.createDataFrame(sorted(masks.items()), schema).coalesce(1).write.mode("overwrite").parquet(out_mask)

    final = spark.read.parquet(out_main)
    log.info("bronze rows: %d, languages: %d", final.count(), final.select("lang").distinct().count())
    final.groupBy("track", "split").count().orderBy("track", "split").show(truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
