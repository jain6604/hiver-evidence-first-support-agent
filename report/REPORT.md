# Report: Evidence-first Spotify Support Agent

## 1. Problem framing

I built a public-channel support agent for SpotifyCares-like customer messages. The system classifies an incoming message into one of eight operational intents, retrieves historical customer/agent resolution pairs, drafts a concise evidence-constrained reply, and decides whether automation is safe.

“Good” means correct routing first, then a reply whose next step is supported by the brand’s historical practice. The agent is not designed to access accounts, enact refunds, resolve fraud, make policy determinations, or replace a human when evidence is weak. It explicitly escalates those cases.

## 2. Data and evaluation design

Source: the public Customer Support on Twitter dataset (ThoughtVector / Kaggle). I extracted direct inbound customer tweet -> SpotifyCares response pairs, filtered malformed/very short messages, and selected a fixed random 10,000-pair subsample (seed 42).

The evaluated set contains 200 real, deduplicated customer messages sampled across coarse issue buckets. All 200 intent, route, and grounding rows are now explicitly marked `human_reviewed` after interactive review. The set is held out from classifier training and retrieval, and `scripts/validate_labels.py` checks required fields and tweet-ID leakage. A blank 40-row human reply-review sample is generated at `data/golden/HUMAN_REVIEW_SAMPLE.csv`.

## 3. System and baselines

| System | Intent | Reply | Routing |
|---|---|---|---|
| Trivial baseline | Predict training-set majority intent | Generic acknowledgment | Always escalate |
| Simple baseline | Keyword bucket | Fixed intent template | Rule-only |
| Full system | TF-IDF + logistic regression | Gemini constrained by top historical examples | Intent confidence + evidence score + safety policy |

The full system’s retrieval layer searches only non-golden pairs. Its prompt forbids invented policies, refunds, timelines, or account facts. Drafts are generated for audit, but `reply_released` is false for every evaluated row because the conservative confidence gate escalated all 200.

## 4. Results

Command: `PYTHONPATH=src .venv/Scripts/python.exe scripts/evaluate.py --golden data/golden/golden_set.csv --live-replies 0`. Metrics below are from `artifacts/evaluation.json` and use the human-reviewed golden labels. `--live-replies 0` was used for the reproducible final run after Gemini quota exhaustion; drafts remain auditable safe fallbacks and none were released.

| Metric | Majority | Simple | Full |
|---|---:|---:|---:|
| Golden intent macro-F1 | 0.044 | 0.464 | 0.382 |
| Golden intent accuracy | 0.180 | 0.555 | 0.450 |
| Auto-handle precision | 0.000 | 0.978 | 0.000 |
| Auto-handle recall | 0.000 | 0.989 | 0.000 |
| Auto-handle F1 | 0.000 | 0.983 | 0.000 |
| Auto-handle rate | 0.000 | 0.445 | 0.000 |
| Escalation precision / recall / F1 | 0.545 / 1.000 / 0.706 | 0.667 / 0.679 / 0.673 | 0.545 / 1.000 / 0.706 |
| Escalation safety | 1.000 | 0.982 | 1.000 |
| LLM-judge pass rate | N/A | N/A | Pending (0 completed; quota exhausted) |

The LLM judge scores relevance, groundedness, safety, tone, and actionability from 1–5 and fails unsafe/invented claims. The human-gold judge run completed zero rows because the Gemini free-tier quota was exhausted; no pass rate is reported. Human reply ratings and agreement are also pending, so no human agreement number is reported.

## 5. Failure analysis

Five real held-out failures are exported in `artifacts/failure_cases.csv`. They include account-email access predicted as billing, a payment failure predicted as account access, playback trouble predicted as account access, a feature question predicted as account access, and a safety-sensitive message predicted as account access. In each case the file retains the full customer message, predicted intent/action, retrieved evidence, observed mismatch, and the hypothesis that lexical similarity or the coarse taxonomy missed the primary issue.

## 6. What is misleading about my headline number?

Macro-F1 and judge pass rate are not a deployment guarantee. The data is historical Twitter traffic, the taxonomy is intentionally coarse, annotations contain judgment calls, and the safest system can look worse on automation-rate metrics because it escalates ambiguous/high-stakes work. Retrieval similarity measures lexical overlap, not factual equivalence. Finally, the LLM judge shares model-family biases with the generator; human-agreement calibration is necessary but a 40-item sample remains noisy.

## 7. One more week

I would add conversation-level state, semantic retrieval with a held-out retrieval benchmark, threshold calibration on a development set, counterfactual safety tests, multilingual handling, a lightweight feedback UI for agents, and a cost/latency analysis. I would also recruit multiple blind annotators and publish confidence intervals.
