import sys, json, numpy as np, pandas as pd, yaml
sys.path.insert(0, ".")
from apps.engine import signals as S
from emoldgn.thresholds import apply_thresholds
cfg = yaml.safe_load(open("config/default.yaml"))
vad = S.load_vad("data/lexicons")
n = pd.read_parquet("apps/_check_best/gold/customer_support/scored_clauses", columns=["doc_id","clause_id","lang","probs","inten"])
o = pd.read_parquet("apps/gold/customer_support/scored_clauses", columns=["doc_id","clause_id","probs","inten"])
m = n.merge(o, on=["doc_id","clause_id"], suffixes=("_n", "_o"))
m = m[m["lang"] == "eng"].reset_index(drop=True)
print("shared English clauses", len(m))
A = lambda col: np.stack(m[col].map(np.asarray).values).astype(np.float32)
res = {}
for tag, art, neutral in (("old", "artifacts", True), ("new", "artifacts_check_best", False)):
    p, inten = A(f"probs_{tag[0]}"), A(f"inten_{tag[0]}")
    th = json.load(open(f"{art}/thresholds.json")); t = np.tile(np.array(th["per_language"].get("eng", th["global"]), dtype=np.float32), (len(m), 1)); print(tag, "eng thresholds", np.round(t[0], 2).tolist())
    t = S.app_thresholds(t, ["eng"] * len(m), cfg["apps"]["threshold_overrides"])
    mask = np.tile(np.array([1, 0, 1, 1, 1, 1], dtype=np.float32), (len(m), 1))
    present = apply_thresholds(p[:, :6], t, mask, p[:, 6] if neutral else None)
    for cut_name, cut in (("cut1.85", 1.85), ("own_top10", float(np.quantile(inten[:, 0], 0.9)))):
        st = S.clause_states(inten, present, mask, cut)
        res[(tag, cut_name)] = {s: np.mean([s in x for x in st]) for s in ["frustration", "outrage", "stress", "disappointment", "delight"]}
    u = S.urgency(p, inten, mask); v, a = S.center_neutral(*S.vad_summary(p[:, :6], inten, vad, mask), present)
    res[(tag, "emo")] = dict(zip(S.EMOTIONS, present.mean(0).round(3))); res[(tag, "emo")]["none"] = float((present.sum(1) == 0).mean())
    res[(tag, "misc")] = {"urgency>0.6": float((u >= 0.6).mean()), "anger_int_p90": float(np.quantile(inten[:, 0], 0.9)),
                          "P(fear|anger|sad)": float(present[:, 2][(present[:, 0] + present[:, 4]) > 0].mean())}
    res[(tag, "v")] = v; res[(tag, "p")] = present
for k, v in res.items():
    if k[1] in ("v", "p"): continue
    print(k, {a: round(float(b), 3) for a, b in v.items()})
print("valence Spearman old vs new", round(float(pd.Series(res[("old","v")]).corr(pd.Series(res[("new","v")]), method="spearman")), 3))
po, pn = res[("old", "p")], res[("new", "p")]
print("same present set", round(float((po == pn).all(1).mean()), 3))
