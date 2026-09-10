# Report: Evidence-first Spotify Support Agent

## 1. Problem framing

I built a public-channel support agent for SpotifyCares-like customer messages. The system classifies an incoming message into one of eight operational intents, retrieves historical customer/agent resolution pairs, drafts a concise evidence-constrained reply, and decides whether automation is safe.

“Good” means correct routing first, then a reply whose next step is supported by the brand’s historical practice. The agent is not designed to access accounts, enact refunds, resolve fraud, make policy determinations, or replace a human when evidence is weak. It explicitly escalates those cases.

## 2. Data and evaluation design

Source: the public Customer Support on Twitter dataset (ThoughtVector / Kaggle). I extracted direct inbound customer tweet -> SpotifyCares response pairs, filtered malformed/very short messages, and selected a fixed random 10,000-pair subsample (seed 42).

The golden set contains 200 real, deduplicated customer messages sampled across coarse issue buckets. It is held out from classifier training and retrieval. Intent, routing, and historical-grounding annotations are made with the published guide. A second annotator independently labels a 40-row overlap; disagreements are adjudicated before final reporting.

## 3. System and baselines

| System | Intent | Reply | Routing |
|---|---|---|---|
| Trivial baseline | Predict training-set majority intent | Generic acknowledgment | Always escalate |
| Simple baseline | Keyword bucket | Fixed intent template | Rule-only |
| Full system | TF-IDF + logistic regression | Gemini constrained by top historical examples | Intent confidence + evidence score + safety policy |

The full system’s retrieval layer searches only non-golden pairs. Its prompt forbids invented policies, refunds, timelines, or account facts; it produces no reply when routing escalates.

## 4. Results

Run `scripts/evaluate.py` after reviewing the development and gold annotations, then replace this table with the generated values. Do not report a metric until its source CSV and command are committed.

| Metric | Majority | Simple | Full |
|---|---:|---:|---:|
| Golden intent macro-F1 | TBD | TBD | TBD |
| Golden intent accuracy | TBD | TBD | TBD |
| Auto-handle precision | N/A | TBD | TBD |
| Auto-handle recall | N/A | TBD | TBD |
| LLM-judge pass rate | N/A | TBD | TBD |

The LLM judge scores relevance, groundedness, safety, tone, and actionability from 1–5 and fails unsafe/invented claims. I compare its binary verdict and dimensions with 40 independent human judgments; the reported agreement will include exact verdict agreement and Spearman correlation by dimension.

## 5. Failure analysis

This section is finalized from held-out examples rather than manufactured anecdotes. The expected high-risk categories to inspect are: (1) multi-intent messages, (2) vague “it doesn’t work” reports, (3) account issues that resemble low-risk product issues, (4) replies whose nearest lexical neighbor has a different underlying cause, and (5) public messages that need a private channel. For each, retain the real message, predicted intent/action, evidence, observed error, and hypothesis.

## 6. What is misleading about my headline number?

Macro-F1 and judge pass rate are not a deployment guarantee. The data is historical Twitter traffic, the taxonomy is intentionally coarse, annotations contain judgment calls, and the safest system can look worse on automation-rate metrics because it escalates ambiguous/high-stakes work. Retrieval similarity measures lexical overlap, not factual equivalence. Finally, the LLM judge shares model-family biases with the generator; human-agreement calibration is necessary but a 40-item sample remains noisy.

## 7. One more week

I would add conversation-level state, semantic retrieval with a held-out retrieval benchmark, threshold calibration on a development set, counterfactual safety tests, multilingual handling, a lightweight feedback UI for agents, and a cost/latency analysis. I would also recruit multiple blind annotators and publish confidence intervals.
