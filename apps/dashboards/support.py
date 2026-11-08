"""Customer Support Analysis. Run from the apps folder:  streamlit run dashboards/support.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Customer Support Analysis", page_icon="🎧", layout="wide")

from apps.dashboards import data as D  # noqa: E402
from apps.dashboards import theme as T  # noqa: E402
from apps.dashboards import upload as U  # noqa: E402
from apps.dashboards import views as V  # noqa: E402
from apps.dashboards.labels import BLUE, ORANGE  # noqa: E402

PROFILE = "customer_support"
CFG = V.Cfg(
    profile=PROFILE, doc_one="conversation", doc_many="conversations", group_one="company", group_many="companies",
    escalation=True, min_group=30, states=("outrage", "frustration", "stress", "disappointment", "delight"),
    sections=(("overview", "🏠 Overview"), ("topics", "🧩 Topics"), ("groups", "🏢 Companies"), ("compare", "⚖️ Compare"),
              ("trends", "📈 Trends"), ("inspector", "🎧 Conversation inspector")),
    boards=(("Most escalations (share needing a person)", "esc", "pct", ORANGE),
            ("Highest average urgency", "urgency", "score", ORANGE),
            ("Happiest customers (share positive)", "pos", "pct", BLUE),
            ("Unhappiest customers (share negative)", "neg", "pct", ORANGE),
            ("Most conversations", "n", "int", BLUE)),
)
COPY = U.UploadCopy(
    noun="customer support messages or tweets", text_hint="the customer's message",
    group_name="Company", group_hint="which company or brand it was sent to", period_hint="week, month, date or batch",
    example="my package has been in Maryland since yesterday and it still hasn't come… ups is trash",
)

T.inject()
ds = D.test_results(PROFILE)
k = D.kpis(ds)
T.hero("Customer Support Analysis", "See how customers really sound, which topics make them angry, and which conversations need a person now.",
       "🎧", "green", [f"{k['n']:,} conversations", f"{k['groups']} companies", "Built-in escalation queue"])

tab_results, tab_upload = st.tabs(["📊  Results on the test data", "⬆️  Upload & analyze a new file"])
with tab_results:
    V.render(ds, CFG, "test")
with tab_upload:
    U.render(CFG, COPY, "up")

T.note("Results describe how a message reads, not how a person truly feels. No automated decisions are made from them.")
