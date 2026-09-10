"""Compute agreement only when genuine human ratings are present."""

from __future__ import annotations

import argparse
import json

import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", default="data/golden/HUMAN_REVIEW_SAMPLE.csv", nargs="?")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    dimensions = ["relevance", "groundedness", "safety", "tone", "actionability"]
    human_columns = [f"human_{dimension}" for dimension in dimensions]
    if any(column not in data.columns for column in human_columns + ["human_verdict"]):
        print(json.dumps({"status": "pending", "reason": "Human rating columns are missing."}))
        return
    if data[human_columns + ["human_verdict"]].replace("", pd.NA).isna().any().any():
        print(json.dumps({"status": "pending", "reason": "Human ratings are blank."}))
        return
    human_summary = {
        "n": len(data),
        "human_pass_rate": round(float((data.human_verdict == "pass").mean()), 3),
        "human_mean_scores": {
            dimension: round(float(pd.to_numeric(data[f"human_{dimension}"]).mean()), 3)
            for dimension in dimensions
        },
    }
    judge_columns = dimensions + ["verdict"]
    if not all(column in data.columns for column in judge_columns) or data[judge_columns].replace("", pd.NA).isna().any().any():
        print(json.dumps({"status": "human_complete_llm_pending", **human_summary, "llm_agreement": "pending"}, indent=2))
        return
    exact = float((data.human_verdict.astype(str) == data.verdict.astype(str)).mean())
    correlations = {dimension: round(float(spearmanr(data[f"human_{dimension}"], data[dimension]).statistic), 3) for dimension in dimensions}
    print(json.dumps({"status": "complete", **human_summary, "exact_verdict_agreement": round(exact, 3), "spearman": correlations}, indent=2))


if __name__ == "__main__":
    main()