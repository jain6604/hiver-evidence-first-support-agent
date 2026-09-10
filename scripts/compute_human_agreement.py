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
    required = human_columns + ["human_verdict", "verdict"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        print(json.dumps({"status": "pending", "reason": f"Missing columns: {missing}"}))
        return
    if data[required].replace("", pd.NA).isna().any().any():
        print(json.dumps({"status": "pending", "reason": "Human ratings are blank; no agreement reported."}))
        return
    exact = float((data.human_verdict.astype(str) == data.verdict.astype(str)).mean())
    correlations = {
        dimension: round(float(spearmanr(data[f"human_{dimension}"], data[dimension]).statistic), 3)
        for dimension in dimensions
    }
    print(json.dumps({"status": "complete", "n": len(data), "exact_verdict_agreement": round(exact, 3), "spearman": correlations}, indent=2))


if __name__ == "__main__":
    main()