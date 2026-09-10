"""Create a blinded 40-row reply-quality sample for genuine human review."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/full_system_predictions.csv")
    parser.add_argument("--output", default="data/golden/HUMAN_REVIEW_SAMPLE.csv")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    data = data[data.draft_reply.fillna("").str.strip().ne("")]
    if len(data) < args.n:
        raise SystemExit(f"Need {args.n} generated replies, found {len(data)}")
    sample = data.sample(args.n, random_state=args.seed)[
        ["tweet_id", "customer_text", "draft_reply", "retrieved_evidence"]
    ].copy()
    for column in ("human_relevance", "human_groundedness", "human_safety", "human_tone", "human_actionability"):
        sample[column] = ""
    sample["human_verdict"] = ""
    sample["human_reason"] = ""
    judgments_path = Path("artifacts/reply_judgments.csv")
    if judgments_path.exists():
        judgments = pd.read_csv(judgments_path)
        judge_columns = ["relevance", "groundedness", "safety", "tone", "actionability", "verdict"]
        available = [column for column in judge_columns if column in judgments.columns]
        sample = sample.merge(judgments[["tweet_id", *available]], on="tweet_id", how="left")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)
    print(f"Wrote {len(sample)} blank human-review rows to {output}")


if __name__ == "__main__":
    main()