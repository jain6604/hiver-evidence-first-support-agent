# Evidence-first Spotify Support Agent

An offline-first, retrieval-grounded AI support agent built for the Hiver SDE Intern take-home. It uses real historical SpotifyCares Twitter conversations to classify customer messages, retrieve comparable resolutions, draft a safe reply, and decide whether to automate or escalate.

## What “good” means

For a public social-support channel, a good agent is more than fluent. It must (1) identify the issue, (2) use a resolution pattern evidenced in the brand’s past responses, and (3) refuse automation when an account, payment, privacy, uncertainty, or weak-evidence risk is present. It is deliberately **not** an account-management bot and never claims to take backstage actions.

## Quick start (under 15 minutes after data download)

```powershell
git clone <YOUR-REPO-URL>
cd hiver-support-agent
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\prepare_data.py
Copy-Item .env.example .env  # add GEMINI_API_KEY only for drafting/judging
.\.venv\Scripts\python.exe scripts\run_agent.py "Spotify charged me but Premium is not active"
```

The source data is intentionally not committed: Kaggle licenses it CC BY-NC-SA 4.0. The pipeline uses only a 10,000-pair seeded SpotifyCares subsample, so it is fast and reproducible.

## Pipeline

```text
incoming message
  -> TF-IDF + logistic-regression intent classifier
  -> TF-IDF retrieval of historical customer -> SpotifyCares reply pairs
  -> conservative risk/evidence routing policy
  -> Gemini draft constrained by retrieved examples (only for auto-handle)
```

### Reproduce the evaluation

1. Generate the golden pack: `python scripts/make_annotation_pack.py`.
2. Independently human-label 150–250 examples using [the guide](data/golden/LABELING_GUIDE.md). Keep this set held out.
3. Produce a separately reviewed development set at `data/annotations/train.csv`. `scripts/label_with_gemini.py` creates reviewable drafts; Gemini labels are never treated as human gold labels.
4. Run `python scripts/evaluate.py`. It rejects train/golden tweet overlap and excludes golden tweets from retrieval.
5. Generate replies on the held-out set, score them with `scripts/judge_replies.py`, and independently human-rate a random 40-row sample using `data/golden/HUMAN_JUDGE_TEMPLATE.csv`. Report agreement as exact verdict agreement and per-dimension Spearman correlation.

## Safety policy

The system escalates when it sees sensitive markers, account/payment intents, low intent confidence (<0.70), or weak retrieval evidence (<0.35). This lower automation rate is intentional: a social-support reply that is confidently wrong is worse than a timely human handoff.

## Repository map

- `scripts/prepare_data.py` — extracts direct inbound -> SpotifyCares resolution pairs.
- `scripts/make_annotation_pack.py` — creates the seeded, stratified golden evaluation CSV.
- `src/hiver_agent/` — classifier, retrieval, routing, and Gemini generation code.
- `scripts/evaluate.py` — baseline and full-system intent/routing metrics with leakage checks.
- `scripts/judge_replies.py` — published LLM-as-judge rubric.
- `report/REPORT.md` — six-page-equivalent submission report, completed after evaluation.
- `DECISION_LOG.md` — non-obvious design decisions and trade-offs.

## Data and borrowed work

- Customer Support on Twitter: [Kaggle / ThoughtVector](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter), CC BY-NC-SA 4.0.
- Libraries: pandas, scikit-learn, Google GenAI SDK. See `requirements.txt`.

No training corpus, source data, API key, or customer PII is committed.
