"""N4 hierarchical shrinkage: language, then family, then global (design document, 5.5, 7.8).

Two steps with lambda = n / (n + kappa), applied to P(j|i) and to NPMI. A+ and A- are
rebuilt from the shrunk NPMI. Languages with no training data get their family prior and
unknown languages get the global graph. Output: data/gold/priors and the tensor bank
data/gold/graph_bank.pt indexed by language id, which the model loads as buffers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_jobs.common import NODES, families, get_spark, job_args, load_config, path_of, setup_logging, skip_if_done

UNK = "<unk>"
STATS = ("p_j_given_i", "npmi")


def lam(n, kappa: float):
    """Weight on a level's own statistic; works on floats, arrays and Spark columns."""
    return n / (n + kappa)


def shrink(stat, prior, n, kappa: float):
    """lambda * stat + (1 - lambda) * prior (design document, 5.5)."""
    w = lam(n, kappa)
    return w * stat + (1 - w) * prior


def hierarchical(graphs, sizes, fam_df, kappa: float, tau_pos: float, tau_neg: float):
    """Long table of shrunk statistics for every language and family scope."""
    from pyspark.sql import functions as F

    g_global = graphs.filter("scope_type = 'global'").select(
        "i", "j", *[F.col(s).alias(f"{s}_g") for s in STATS])
    n_fam = sizes.filter("scope_type = 'family'").select(F.col("scope").alias("family"), F.col("count").alias("n_f"))
    n_lang = sizes.filter("scope_type = 'lang'").select(F.col("scope").alias("lang"), F.col("count").alias("n_l"))

    fam = graphs.filter("scope_type = 'family'").withColumnRenamed("scope", "family").join(n_fam, "family")
    # Family pairs absent from the family (all its languages lack the emotion) use global.
    all_pairs = fam_df.select("family").distinct().crossJoin(g_global).join(n_fam, "family")
    fam = all_pairs.join(fam.select("family", "i", "j", *STATS), ["family", "i", "j"], "left")
    for s in STATS:
        fam = fam.withColumn(f"{s}_fam", shrink(F.coalesce(F.col(s), F.col(f"{s}_g")), F.col(f"{s}_g"), F.col("n_f"), kappa))
    fam = fam.select("family", "i", "j", "n_f", "p_j_given_i_fam", "npmi_fam")

    lang = (
        graphs.filter("scope_type = 'lang'").withColumnRenamed("scope", "lang").join(n_lang, "lang")
        .join(fam_df, "lang", "left").join(fam, ["family", "i", "j"], "left").join(g_global, ["i", "j"])
    )
    for s in STATS:
        base = F.coalesce(F.col(f"{s}_fam"), F.col(f"{s}_g"))
        lang = lang.withColumn(f"{s}_prior", F.when(F.col("valid"), shrink(F.col(s), base, F.col("n_l"), kappa)).otherwise(0.0))
    return add_edges(lang, tau_pos, tau_neg), fam


def add_edges(df, tau_pos: float, tau_neg: float):
    """A+ and A- from shrunk NPMI; the diagonal is ignored (design document, 7.2)."""
    from pyspark.sql import functions as F

    off = F.col("i") != F.col("j")
    return (
        df.withColumn("a_pos_prior", F.when(off, F.greatest(F.col("npmi_prior") - tau_pos, F.lit(0.0))).otherwise(0.0))
        .withColumn("a_neg_prior", F.when(off, F.greatest(-F.col("npmi_prior") - tau_neg, F.lit(0.0))).otherwise(0.0))
    )


def to_matrix(pdf, value: str):
    """Long (i, j, value) rows to a 7 x 7 numpy matrix in node order."""
    import numpy as np

    idx = {n: k for k, n in enumerate(NODES)}
    m = np.zeros((len(NODES), len(NODES)), dtype=np.float32)
    for i, j, v in zip(pdf["i"], pdf["j"], pdf[value]):
        m[idx[i], idx[j]] = float(v)
    return m


def build_bank(spark, cfg: dict, graphs, lang_prior, fam_prior, masks, fam_map: dict[str, str], sizes, tau: tuple[float, float]):
    """Tensor bank for the model: one entry per known language, plus <unk> (global)."""
    import numpy as np
    import torch

    tau_pos, tau_neg = tau
    mask_of = {r["lang"]: r["mask"] + [1] for r in masks.collect()}
    trained = sorted(r["scope"] for r in graphs.filter("scope_type = 'lang'").select("scope").distinct().collect())
    unseen = sorted(set(mask_of) - set(trained))
    languages = trained + unseen + [UNK]
    n_nodes = len(NODES)

    gp = graphs.toPandas()
    lp = lang_prior.toPandas()
    fp = fam_prior.toPandas()
    keys = ["cond_raw", "npmi_raw", "A_pos_raw", "A_neg_raw", "cond_prior", "npmi_prior", "A_pos_prior", "A_neg_prior"]
    bank = {k: np.zeros((len(languages), n_nodes, n_nodes), dtype=np.float32) for k in keys}
    node_mask = np.ones((len(languages), n_nodes), dtype=np.float32)

    def scope(kind: str, name: str):
        return gp[(gp["scope_type"] == kind) & (gp["scope"] == name)]

    global_raw = {k: to_matrix(scope("global", "global"), v) for k, v in
                  [("cond", "p_j_given_i"), ("npmi", "npmi"), ("A_pos", "a_pos"), ("A_neg", "a_neg")]}
    for x, lang in enumerate(languages):
        m = np.array(mask_of.get(lang, [1] * n_nodes), dtype=np.float32)
        node_mask[x] = m
        pair = np.outer(m, m)
        if lang in trained:
            s, ls = scope("lang", lang), lp[lp["lang"] == lang]
            raw = {"cond": to_matrix(s, "p_j_given_i"), "npmi": to_matrix(s, "npmi"),
                   "A_pos": to_matrix(s, "a_pos"), "A_neg": to_matrix(s, "a_neg")}
            prior = {"cond": to_matrix(ls, "p_j_given_i_prior"), "npmi": to_matrix(ls, "npmi_prior"),
                     "A_pos": to_matrix(ls, "a_pos_prior"), "A_neg": to_matrix(ls, "a_neg_prior")}
        else:
            raw = dict(global_raw)
            family = fam_map.get(lang)
            if lang != UNK and family is not None and (fp["family"] == family).any():
                f = fp[fp["family"] == family]
                cond, npmi = to_matrix(f, "p_j_given_i_fam"), to_matrix(f, "npmi_fam")
            else:
                cond, npmi = global_raw["cond"], global_raw["npmi"]
            off = 1.0 - np.eye(n_nodes, dtype=np.float32)
            prior = {"cond": cond, "npmi": npmi, "A_pos": np.clip(npmi - tau_pos, 0, None) * off,
                     "A_neg": np.clip(-npmi - tau_neg, 0, None) * off}
        for name, mats in (("raw", raw), ("prior", prior)):
            for stat, mat in mats.items():
                bank[f"{stat}_{name}"][x] = mat * pair

    sz = sizes.toPandas()
    n_of = {(r.scope_type, r.scope): int(r["count"]) for _, r in sz.iterrows()}
    kappa = float(cfg["n4"]["kappa"])
    fam_names = sorted(set(fam_map.values()))
    lam_l = np.array([lam(n_of.get(("lang", l), 0), kappa) for l in languages], dtype=np.float32)
    lam_f = np.array([lam(n_of.get(("family", fam_map.get(l, "")), 0), kappa) for l in languages], dtype=np.float32)
    family_index = [fam_names.index(fam_map[l]) if l in fam_map else len(fam_names) for l in languages]
    out = {k: torch.from_numpy(v) for k, v in bank.items()}
    out.update(mask=torch.from_numpy(node_mask), languages=languages, nodes=NODES, families=fam_names,
               family_index=torch.tensor(family_index), lambda_l=torch.from_numpy(lam_l), lambda_f=torch.from_numpy(lam_f),
               trained_languages=trained)
    return out


def main() -> None:
    args = job_args(__doc__)
    cfg = load_config(args.config)
    log = setup_logging("06_priors", cfg)
    out = path_of(cfg, "gold", "priors")
    out_bank = path_of(cfg, "gold", "graph_bank.pt")
    if skip_if_done([out, out_bank], args.force, log):
        return

    import torch

    spark = get_spark("06_priors")
    graphs = spark.read.parquet(path_of(cfg, "gold", "graphs")).cache()
    sizes = spark.read.parquet(path_of(cfg, "gold", "graph_scope_sizes")).cache()
    masks = spark.read.parquet(path_of(cfg, "bronze", "label_mask"))
    fam_map = families(cfg)
    fam_df = spark.createDataFrame(sorted(fam_map.items()), ["lang", "family"])
    tau = (float(cfg["n2"]["tau_pos"]), float(cfg["n2"]["tau_neg"]))
    lang_prior, fam_prior = hierarchical(graphs, sizes, fam_df, float(cfg["n4"]["kappa"]), *tau)
    lang_prior = lang_prior.select(
        "lang", "family", "i", "j", "n_l", "n_f", "valid", "p_j_given_i_prior", "npmi_prior", "a_pos_prior", "a_neg_prior"
    ).cache()
    lang_prior.coalesce(2).write.mode("overwrite").parquet(out)
    bank = build_bank(spark, cfg, graphs, lang_prior, fam_prior, masks, fam_map, sizes, tau)
    Path(out_bank).parent.mkdir(parents=True, exist_ok=True)
    tmp = out_bank + ".tmp"
    torch.save(bank, tmp)
    Path(tmp).rename(out_bank)
    log.info("bank: %d entries (%d trained); global at %s", len(bank["languages"]),
             len(bank["trained_languages"]), bank["languages"][-1])
    log.info("lambda_l range %.3f-%.3f over trained languages",
             float(bank["lambda_l"][: len(bank["trained_languages"])].min()),
             float(bank["lambda_l"][: len(bank["trained_languages"])].max()))
    spark.stop()


if __name__ == "__main__":
    main()
