"""Student Course Review Analysis. Run from the apps folder:  streamlit run dashboards/feedback.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Student Course Reviews", page_icon="🎓", layout="wide")

from apps.dashboards import data as D  # noqa: E402
from apps.dashboards import theme as T  # noqa: E402
from apps.dashboards import upload as U  # noqa: E402
from apps.dashboards import views as V  # noqa: E402
from apps.dashboards.labels import BLUE, ORANGE, PURPLE, STATE_COLOR  # noqa: E402

PROFILE = "student_feedback"
CFG = V.Cfg(
    profile=PROFILE, doc_one="comment", doc_many="comments", group_one="course or teacher", group_many="courses and teachers",
    escalation=False, min_group=10, states=("frustration", "stress", "disappointment", "delight", "mixed"),
    sections=(("overview", "🏠 Overview"), ("topics", "🧩 Aspects"), ("groups", "🎓 Courses & teachers"),
              ("compare", "⚖️ Compare"), ("trends", "📈 Semesters"), ("inspector", "💬 Comment inspector"),
              ("reality", "✅ Reality check")),
    boards=(("Happiest (share positive)", "pos", "pct", BLUE),
            ("Most unhappy (share negative)", "neg", "pct", ORANGE),
            ("Most mixed feelings", "mixed", "pct", STATE_COLOR["mixed"]),
            ("Most stressed", "stress", "pct", PURPLE),
            ("Most comments", "n", "int", BLUE)),
)
COPY = U.UploadCopy(
    noun="student course or teacher reviews", text_hint="the student's comment",
    group_name="Course", group_hint="course code, course name or teacher, so results can be compared",
    period_hint="semester, term or date",
    example="Notes were unorganized and no help on piazza, barely survived the chapter quizzes",
)

T.inject()
ds = D.test_results(PROFILE)
k = D.kpis(ds)
T.hero("Student Course Review Analysis", "Find out what students love, where they struggle, and how each course and teacher compares.",
       "🎓", "blue", [f"{k['n']:,} comments", f"{k['groups']:,} courses & teachers", "Topics: content, teaching, exams, workload"])

tab_results, tab_upload = st.tabs(["📊  Results on the test data", "⬆️  Upload & analyze new reviews"])
with tab_results:
    V.render(ds, CFG, "test")
with tab_upload:
    U.render(CFG, COPY, "up")

T.note("Results describe how a comment reads, not how a student truly feels. No automated decisions are made from them.")
