# Decision log

1. **SpotifyCares instead of the largest brand** — it has enough direct pairs for retrieval while keeping the support domain coherent and the local corpus fast.
2. **Use direct inbound -> agent links only** — avoids guessing thread order and gives a clear evidence unit: a customer issue and the brand’s immediate response.
3. **Use a 10,000-pair seeded subsample** — the brief explicitly encourages subsampling; it permits a <15-minute reproduction path.
4. **Keep the source dataset out of Git** — it is large and CC BY-NC-SA; scripts reconstruct the derived corpus locally.
5. **Derive a compact eight-intent taxonomy** — customer support needs actionable categories, not a broad academic ontology.
6. **Separate gold from training and retrieval** — test leakage would inflate both classification and reply-grounding claims.
7. **Use TF-IDF + logistic regression as the primary reproducible model** — it is quick, inspectable, and a meaningful baseline against an LLM-heavy approach.
8. **Use TF-IDF retrieval instead of opaque embeddings** — exact evidence scores are cheap, deterministic, and easy to audit within the assignment time limit.
9. **Use Gemini for generation, not as the only system component** — keeps the core pipeline runnable and makes the LLM’s role bounded and testable.
10. **Constrain the generation prompt to historical evidence** — fluency alone is not a support resolution and can produce invented actions.
11. **Never auto-handle account or payment messages** — their plausible cost of error is high even when classification confidence is high.
12. **Require both classification confidence and retrieval evidence for automation** — either signal alone is insufficient.
13. **Evaluate automation precision separately from recall** — a safe system may intentionally escalate often; aggregate accuracy can hide this.
14. **Calibrate the LLM judge on human ratings** — without agreement evidence, an LLM-judge score is only a proxy.
15. **Include a mandatory limitations section** — the benchmark is historical public tweets, not proof of success on current private support tickets.
