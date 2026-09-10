# Golden-set labelling guide

Each row is a real inbound SpotifyCares message paired with the historical public reply. Label the incoming message only; do not infer an intent from a reply that might be generic.

1. Pick exactly one intent from `src/hiver_agent/taxonomy.py`.
2. Set `route` to `auto_handle` only when a generic, evidence-grounded public response would be safe. Escalate account, payment/refund, identity, legal, abusive, ambiguous, and unsupported cases.
3. Write one short route reason.
4. Rate the historical reply `yes`, `partly`, or `no` for whether it is a relevant resolution. This is used for grounding analysis, not model training.

Sampling: the pack is seeded (`42`), stratified across keyword-derived buckets, deduplicated by customer tweet ID, and has no overlap with the retrieval corpus during evaluation. Two annotators should independently label 40 rows; resolve disagreements before one annotator labels the remainder. Record agreement in the report.
