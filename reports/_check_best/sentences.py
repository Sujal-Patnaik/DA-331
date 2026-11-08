import sys, json, numpy as np, yaml
sys.path.insert(0, ".")
from apps.engine import signals as S
from apps.engine.scorer import Scorer
from emoldgn.thresholds import apply_thresholds
cfg = yaml.safe_load(open("config/default.yaml")); vad = S.load_vad("data/lexicons")
# (sentence, intended emotions, intended state or '')
T = [("My order arrived a week late and nobody answered my emails, this is unacceptable!", "anger", "frustration"),
     ("I am so happy, the agent fixed my problem in two minutes!", "joy", "delight"),
     ("I am really worried that my account has been hacked and I might lose my savings.", "fear", "stress"),
     ("I was hoping the new phone would be better, but I feel let down.", "sadness", "disappointment"),
     ("Wow, I did not expect a refund to arrive this quickly!", "surprise+joy", ""),
     ("Please update me on the status of ticket 4821.", "none", ""),
     ("I loved the lectures, but the final exam was terrifying and far too long.", "joy+fear", ""),
     ("The professor was rude to me in front of the whole class and I am furious.", "anger", "outrage"),
     ("I feel so sad that the course ended, I learned a lot and will miss it.", "sadness", ""),
     ("The assignment deadlines were stressful and I could not sleep before the test.", "fear", "stress"),
     ("The syllabus is posted on the course website.", "none", ""),
     ("What a pleasant surprise, the lab was far more interesting than I thought!", "joy+surprise", "")]
sc = {}
for tag, art, neutral in (("old", "artifacts", True), ("new", "artifacts_check_best", False)):
    s = Scorer(art); r = s.score([x[0] for x in T], ["eng"] * len(T))
    th = S.app_thresholds(s.thresholds(["eng"] * len(T)), ["eng"] * len(T), cfg["apps"]["threshold_overrides"])
    mask = s.mask(["eng"] * len(T)); p, i = r["probs"], r["inten"]
    pres = apply_thresholds(p[:, :6], th, mask, p[:, 6] if neutral else None)
    cut = 1.85 if tag == "old" else 0.9   # trial value for the new model only, set from sentences' scale, see report
    st = S.clause_states(i, pres, mask, cut); u = S.urgency(p, i, mask)
    sc[tag] = ([",".join(e for c, e in enumerate(S.EMOTIONS) if pres[k, c]) or "none" for k in range(len(T))], st, u, i[:, 0])
for k, (txt, want, wst) in enumerate(T):
    print(f"{k+1:2d} want={want:13s}/{wst:14s} | old={sc['old'][0][k]:22s} {','.join(sc['old'][1][k]):16s} u={sc['old'][2][k]:.2f} | new={sc['new'][0][k]:22s} {','.join(sc['new'][1][k]):16s} u={sc['new'][2][k]:.2f}")
