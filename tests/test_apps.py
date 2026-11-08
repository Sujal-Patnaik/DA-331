"""Application engine: segmentation, aspects, signals, rules, text cleaning."""
import numpy as np
import pytest

from apps.engine import signals as S
from apps.engine.aspects import aspect_pattern, cue_hits, load_profile, tag_aspects
from apps.engine.langid import UNK, detect
from apps.engine.scorer import merge_tokens
from apps.engine.segment import segment
from spark_jobs.common import ROOT
from spark_jobs.textnorm import clean_text

SUPPORT = load_profile(ROOT / "profiles/customer_support.yaml")
FEEDBACK = load_profile(ROOT / "profiles/student_feedback.yaml")
EXAMPLE = "I enjoyed the course, but the examination was stressful and confusing"


def test_contrast_clauses_of_the_plan_example():
    assert segment(EXAMPLE) == ["I enjoyed the course", "but the examination was stressful and confusing"]


def test_segmentation_edges():
    assert segment("") == []
    assert segment("First point. Second point!") == ["First point.", "Second point!"]
    assert segment("Though it rained, we left") == ["Though it rained, we left"]
    assert segment("butter is good") == ["butter is good"]
    assert len(segment("कोर्स अच्छा था लेकिन परीक्षा कठिन थी")) == 2
    assert len(segment("अभ्यासक्रम चांगला होता पण परीक्षा कठीण होती")) == 2


def test_aspects_of_the_plan_example():
    a, b = segment(EXAMPLE)
    assert tag_aspects(a, FEEDBACK) == ["content"]
    assert tag_aspects(b, FEEDBACK) == ["exams"]


def test_aspects_use_whole_words_and_plurals():
    assert tag_aspects("I am so happy today", SUPPORT) == ["other"]
    assert tag_aspects("chocolate cake", SUPPORT) == ["other"]
    assert "app_site" in tag_aspects("the app keeps failing", SUPPORT)
    assert "app_site" in tag_aspects("Someone logged in to my account", SUPPORT)
    assert "delivery" not in tag_aspects("my refund still has not arrived", SUPPORT)
    assert "exams" in tag_aspects("the quizzes were hard", FEEDBACK)
    assert "exams" in tag_aspects("Exams were hard", FEEDBACK)
    assert "exams" in tag_aspects("परीक्षा कठिन थी", FEEDBACK)


def test_spark_and_python_taggers_agree(spark):
    from pyspark.sql import functions as F

    texts = ["the refund was late", "so happy", "Refunds never came", "money   back please", "चार्ज और रिफंड नहीं मिला"]
    df = spark.createDataFrame([(t,) for t in texts], ["t"])
    pattern = aspect_pattern(SUPPORT["aspects"]["refund"], java=True)
    java = {r["t"]: r["hit"] for r in df.select("t", F.col("t").rlike(pattern).alias("hit")).collect()}
    for t in texts:
        assert java[t] == ("refund" in tag_aspects(t, SUPPORT)), t


def test_confusion_is_a_cue_not_a_label():
    assert cue_hits(EXAMPLE, FEEDBACK) == ["confusion"]
    assert cue_hits(EXAMPLE, SUPPORT) == []


def one(p, inten=None, ad=None):
    pa = np.array([p], dtype=float)
    return pa, np.zeros((1, 6)) if inten is None else np.array([inten], float), np.ones((1, 7, 7)) if ad is None else ad


def test_urgency_uses_only_negative_emotions():
    full = np.ones((1, 6))
    p, inten, _ = one([0, 0, 0, 1.0, 0, 1.0], [0, 0, 0, 3, 0, 3])
    assert S.urgency(p, inten, full)[0] == 0
    p, inten, _ = one([0.8, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0])
    assert abs(S.urgency(p, inten, full)[0] - 0.8) < 1e-9


def test_masked_emotion_cannot_raise_urgency_or_shift_valence():
    no_disgust = np.array([[1.0, 0, 1, 1, 1, 1]])
    p, inten, _ = one([0, 1.0, 0, 0, 0, 0], [0, 3, 0, 0, 0, 0])
    assert S.urgency(p, inten, np.ones((1, 6)))[0] == 1
    assert S.urgency(p, inten, no_disgust)[0] == 0
    vad = np.array([[-1.0, 1.0]] * 6)
    v, _ = S.vad_summary(p, inten, vad, no_disgust)
    assert v[0] == 0


def states(present, inten=None, mask=None):
    ia = np.zeros((1, 6)) if inten is None else np.array([inten], float)
    m = np.ones((1, 6)) if mask is None else np.array([mask], float)
    return S.clause_states(ia, np.array([present]), m, 1.85)[0]


def test_pair_states_need_both_emotions():
    assert states([1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 2, 0]) == ["frustration"]
    assert states([1, 1, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0]) == ["outrage"]
    assert states([0, 0, 0, 1, 0, 1]) == ["delight"]
    assert "delight" not in states([0, 0, 1, 1, 0, 1], [0, 0, 1, 2, 0, 2])


def test_single_states_need_the_strongest_negative_emotion():
    assert states([0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0]) == ["stress"]
    assert states([0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 2, 0]) == ["disappointment"]
    assert states([0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 2, 0]) == ["disappointment"]
    assert states([0, 0, 1, 0, 1, 0], [0, 0, 2, 0, 1, 0]) == ["stress"]
    assert states([1, 0, 1, 0, 0, 0], [2, 0, 1, 0, 0, 0]) == []


def test_frustration_replaces_stress_on_an_angry_message():
    assert states([1, 0, 1, 0, 1, 0], [1, 0, 2, 0, 1, 0]) == ["frustration"]


def test_outrage_without_disgust_uses_the_anger_intensity_cut():
    mask = [1, 0, 1, 1, 1, 1]
    assert "outrage" in states([1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], mask)
    assert "outrage" not in states([1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], mask)


def test_app_threshold_override_applies_only_to_its_language():
    t = np.full((2, 6), 0.5)
    out = S.app_thresholds(t, ["eng", "hin"], {"eng": {"fear": 0.95}})
    assert out[0, S.FEAR] == 0.95 and out[1, S.FEAR] == 0.5 and out[0, S.JOY] == 0.5
    assert t[0, S.FEAR] == 0.5


def test_masked_emotion_cannot_win_as_strongest():
    assert states([0, 0, 1, 0, 0, 0], [0, 3, 2, 0, 0, 0], [1, 0, 1, 1, 1, 1]) == ["stress"]


def test_mixed_spans_clauses():
    assert S.is_mixed(np.array([[0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]]))
    assert not S.is_mixed(np.array([[0, 0, 0, 1, 0, 1]]))


def test_profile_rules():
    assert S.should_escalate(SUPPORT, 0.6, set(), set())
    assert not S.should_escalate(SUPPORT, 0.59, set(), set())
    assert S.should_escalate(SUPPORT, 0.1, {"outrage"}, set())
    assert S.should_escalate(SUPPORT, 0.1, {"stress"}, {"refund"})
    assert not S.should_escalate(SUPPORT, 0.1, {"stress"}, {"delivery"})
    assert not S.should_escalate(FEEDBACK, 1.0, {"stress"}, set())


def test_rules_reject_anything_outside_the_grammar():
    with pytest.raises(ValueError):
        S.parse_rule("__import__('os').system('true')")


def test_valence_of_neutral_text_is_the_centre():
    vad = np.array([[-1, 1]] * 6, dtype=float)
    v, a = S.vad_summary(np.zeros((1, 6)), np.zeros((1, 6)), vad, np.ones((1, 6)))
    assert v[0] == 0 and a[0] == 0
    v, _ = S.vad_summary(np.array([[1.0, 0, 0, 0, 0, 0]]), np.array([[3.0, 0, 0, 0, 0, 0]]), vad, np.ones((1, 6)))
    assert v[0] == -1


def test_lexicon_has_the_six_emotion_words():
    vad = S.load_vad(ROOT / "data/lexicons")
    if vad is None:
        pytest.skip("lexicon not downloaded")
    assert vad.shape == (6, 2)
    assert vad[S.JOY, 0] > 0 > vad[S.ANGER, 0]


def test_language_detection_falls_back_to_unknown(tmp_path):
    assert detect(["hello there"], tmp_path / "missing.bin") == [UNK]


def test_language_detection_with_the_model():
    path = ROOT / "models/lid.176.bin"
    if not path.exists():
        pytest.skip("lid.176 not downloaded")
    assert detect(["this is a perfectly ordinary English sentence", "यह एक सामान्य हिंदी वाक्य है"], path) == ["eng", "hin"]


def test_clean_text_rules():
    assert clean_text(None) is None
    out = clean_text("Mail me at a.b@x.com :) see https://t.co/ab_c @41702 😡")
    assert "<email>" in out and "[URL]" in out and "[USER]" in out and "smiling face" in out and "enraged face" in out
    assert "https" not in out and "@" not in out and "_" not in out


def test_clean_text_keeps_case_and_stop_words():
    assert clean_text("NOT the Best") == "NOT the Best"


def test_clean_text_leaves_lookalikes_alone():
    assert clean_text("Note:Done at 10:30") == "Note:Done at 10:30"
    assert clean_text("snake_case stays") == "snake_case stays"


def test_merge_tokens():
    words = merge_tokens(["<s>", "▁exam", "ination", "▁was", "</s>"], np.array([0.1, 0.2, 0.3, 0.4, 0.0]))
    assert words[0][0] == "examination" and abs(words[0][1] - 0.5) < 1e-6


def test_coursera_loader_reads_stars(spark, tmp_path):
    import importlib

    job = importlib.import_module("spark_jobs.09_score")
    assert job.coursera_documents(spark, {"paths": {"demo": str(tmp_path)}}) is None
    d = tmp_path / "coursera"
    d.mkdir()
    (d / "reviews.csv").write_text('reviews,reviewers,date_reviews,rating,course_id\n"great, really",A,"Jul 1, 2020",5,ml\nbad,B,"Jul 2, 2020",1,ml\n')
    rows = job.coursera_documents(spark, {"paths": {"demo": str(tmp_path)}}).collect()
    assert sorted(r["gold_sentiment"] for r in rows) == [-1.0, 1.0]
    assert {r["group_id"] for r in rows} == {"course_ml"}


def test_neutral_text_sits_at_the_centre():
    v, a = S.center_neutral(np.array([0.9, 0.9]), np.array([0.5, 0.5]), np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]))
    assert v.tolist() == [0.0, 0.9] and a.tolist() == [0.0, 0.5]
