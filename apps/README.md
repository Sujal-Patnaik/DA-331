# Customer Support Analysis and Student Course Reviews

Two Streamlit apps that share one engine and one look. Each has two tabs:

1. **Results on the test data**: overview, topics, groups and report cards, compare, trends and an inspector, drawn from the stored tables in `apps/gold/`. These are the demo datasets (customer support conversations and EduRABSA reviews) scored by the EmoLDGN+ model, which was trained on BRIGHTER only and never saw them.
2. **Upload & analyze**: drop a CSV, check the columns, and see the same views for your own file, with a results download.

## Run

From the repository root:

```
pip install -r requirements.txt
streamlit run apps/dashboards/support.py
streamlit run apps/dashboards/feedback.py
```

The theme lives in `.streamlit/config.toml` (repository root). A copy sits in `apps/.streamlit/` for running from the `apps/` folder.

## Requirements

- The stored tables in `apps/gold/<profile>/` (from `spark_jobs/09_score.py` and `10_aggregate.py`).
- For the upload tab, the exported model in `artifacts/` (`best.pt`, `thresholds.json`, `lang_index.json`). Without it, the upload tab says the model is not available, and the first tab still works.

## How an uploaded file is scored

`dashboards/backend.py` follows the same steps as the stored tables:

1. Clean and mask the text with the training-side cleaner.
2. Detect the language and split each message into clauses.
3. Score each clause with the exported EmoLDGN+ model.
4. Derive labels, states, urgency and valence with the aggregation job's own function.
5. Roll clauses up to messages with the max rule.

Message ids longer than 16 characters, repeated or blank are replaced by row numbers, so long hash ids never appear in the app or the download.

## Layout

| File | Role |
|---|---|
| `dashboards/support.py`, `feedback.py` | The two apps |
| `dashboards/views.py` | Every screen, shared by both apps and both tabs |
| `dashboards/upload.py` | Upload, column mapping, progress, results, download |
| `dashboards/backend.py` | Calls the EmoLDGN+ engine |
| `dashboards/data.py` | Results tables to display-ready aggregates |
| `dashboards/charts.py`, `theme.py`, `labels.py` | Charts, styling, words and colours |
| `dashboards/insights.py` | Plain-language "what stands out" sentences |
| `engine/` | Segmentation, language ID, aspects, signals and the model scorer |
| `api/server.py` | FastAPI endpoint for one text |
