"""Create a compact, reproducible SpotifyCares support corpus from the Kaggle CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_pairs(source: Path, brand: str, limit: int, seed: int) -> pd.DataFrame:
    columns = ["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"]
    tweets = pd.read_csv(source, usecols=columns, low_memory=False)
    tweets["tweet_id"] = tweets["tweet_id"].astype("Int64")
    tweets["in_response_to_tweet_id"] = tweets["in_response_to_tweet_id"].astype("Int64")
    agents = tweets[(tweets["author_id"] == brand) & (tweets["inbound"] == False)].copy()  # noqa: E712
    customers = tweets[tweets["inbound"] == True][["tweet_id", "text"]].rename(
        columns={"tweet_id": "customer_tweet_id", "text": "customer_text"}
    )
    pairs = agents.merge(
        customers,
        left_on="in_response_to_tweet_id",
        right_on="customer_tweet_id",
        how="inner",
    ).rename(columns={"tweet_id": "agent_tweet_id", "text": "agent_reply"})
    pairs = pairs[["customer_tweet_id", "agent_tweet_id", "customer_text", "agent_reply"]].dropna()
    pairs = pairs.drop_duplicates("customer_tweet_id")
    pairs = pairs[pairs.customer_text.str.len().between(12, 500)]
    pairs = pairs.sample(n=min(limit, len(pairs)), random_state=seed).reset_index(drop=True)
    pairs.insert(0, "tweet_id", pairs.customer_tweet_id.astype(str))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/kaggle/twcs/twcs.csv")
    parser.add_argument("--brand", default="SpotifyCares")
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/processed/spotify_pairs.csv")
    args = parser.parse_args()
    pairs = build_pairs(Path(args.source), args.brand, args.limit, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output, index=False)
    print(f"Wrote {len(pairs):,} direct customer -> {args.brand} reply pairs to {output}")


if __name__ == "__main__":
    main()
