"""Shared Spark helpers: session, paths, schemas, label constants (project specification, section 5.2)."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EMOS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
NODES = EMOS + ["neutral"]
SPLITS = ("train", "dev", "test")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config/default.yaml, or --config, or $EMOLDGN_CONFIG (smoke tests, dashboards)."""
    p = Path(path or os.environ.get("EMOLDGN_CONFIG") or ROOT / "config/default.yaml")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_of(cfg: dict[str, Any], key: str, *parts: str) -> str:
    """Absolute path for a configured location, so jobs work from any working directory."""
    base = Path(cfg["paths"][key])
    base = base if base.is_absolute() else ROOT / base
    return str(base.joinpath(*parts))


def families(cfg: dict[str, Any]) -> dict[str, str]:
    """Language -> family map from config/families.yaml (design document, 7.8)."""
    with open(path_of(cfg, "families"), encoding="utf-8") as f:
        fam = yaml.safe_load(f)
    return {lang: name for name, langs in fam.items() for lang in langs}


def driver_memory() -> str:
    """About half of RAM, at most 8g (project specification, section 5.2)."""
    try:
        gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError):
        gb = 8.0
    return f"{max(1, min(8, int(gb / 2)))}g"


def get_spark(app: str, master: str = "local[*]", extra: dict[str, str] | None = None):
    """SparkSession as defined in section 5.2 of the project specification. PYSPARK_PYTHON must point at the venv interpreter."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    # Python workers must be able to import this repository's modules.
    paths = [str(ROOT)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    from pyspark.sql import SparkSession

    b = (
        SparkSession.builder.appName(app)
        .master(master)
        .config("spark.driver.memory", driver_memory())
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.warehouse.dir", str(ROOT / "spark-warehouse"))
    )
    for k, v in (extra or {}).items():
        b = b.config(k, v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def setup_logging(name: str, cfg: dict[str, Any]) -> logging.Logger:
    """Log to stdout and to reports/logs/<job>.log (project specification, section 11)."""
    log_dir = Path(path_of(cfg, "reports", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / f"{name}.log")],
        force=True,
    )
    return logging.getLogger(name)


def job_args(description: str, extra=None) -> argparse.Namespace:
    """Standard CLI: --force, --config; extra(parser) may add job-specific options."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--force", action="store_true", help="recompute even if the output exists")
    ap.add_argument("--config", default=None, help="config YAML (default config/default.yaml)")
    if extra:
        extra(ap)
    return ap.parse_args()


def exists(path: str | Path) -> bool:
    """True when a finished Parquet output is present; a partial write has no _SUCCESS."""
    p = Path(path)
    return p.is_file() or (p / "_SUCCESS").exists()


def skip_if_done(paths: list[str], force: bool, log: logging.Logger) -> bool:
    """Idempotency rule of the project specification (section 5.2): returns True when the job can exit immediately."""
    if not force and all(exists(p) for p in paths):
        log.info("exists, skipping: %s", ", ".join(paths))
        return True
    return False


def label_array_col():
    """Array column of the six emotions; null (not annotated) becomes -1."""
    from pyspark.sql import functions as F

    return F.array(*[F.coalesce(F.col(e).cast("int"), F.lit(-1)) for e in EMOS])
