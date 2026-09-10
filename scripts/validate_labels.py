"""Validate annotation fields and prevent tweet-ID leakage across splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hiver_agent.taxonomy import INTENTS

REQUIRED = {
    "train": {"tweet_id", "customer_text", "intent"},
    "golden": {"tweet_id", "customer_text", "agent_reply", "intent", "route", "reply_grounded"},
}
VALID_ROUTES = {"auto_handle", "escalate"}
VALID_GROUNDED = {"yes", "partly", "no"}


def load_checked(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {split} file: {path}")
    data = pd.read_csv(path, dtype={"tweet_id": str})
    missing = REQUIRED[split] - set(data.columns)
    if missing:
        raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
    if data.empty:
        raise SystemExit(f"{path} is empty")
    for column in REQUIRED[split]:
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise SystemExit(f"{path} contains blank required values in {column}")
    if data.tweet_id.duplicated().any():
        raise SystemExit(f"{path} contains duplicate tweet_id values")
    invalid = set(data.intent) - set(INTENTS)
    if invalid:
        raise SystemExit(f"{path} contains invalid intents: {sorted(invalid)}")
    if split == "golden":
        invalid_routes = set(data.route) - VALID_ROUTES
        invalid_grounded = set(data.reply_grounded) - VALID_GROUNDED
        if invalid_routes or invalid_grounded:
            raise SystemExit(f"{path} has invalid route/grounding values: {invalid_routes} / {invalid_grounded}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/annotations/train.csv")
    parser.add_argument("--golden", default="data/golden/golden_set.csv")
    parser.add_argument("--pairs", default="data/processed/spotify_pairs.csv")
    args = parser.parse_args()
    train = load_checked(Path(args.train), "train")
    golden = load_checked(Path(args.golden), "golden")
    train_ids = set(train.tweet_id.astype(str))
    golden_ids = set(golden.tweet_id.astype(str))
    if train_ids & golden_ids:
        raise SystemExit(f"train/golden overlap: {len(train_ids & golden_ids)} tweet IDs")
    if Path(args.pairs).exists():
        pair_ids = set(pd.read_csv(args.pairs, usecols=["tweet_id"], dtype=str).tweet_id)
        if train_ids - pair_ids or golden_ids - pair_ids:
            raise SystemExit("annotations contain tweet IDs absent from the source pair corpus")
    print(f"Validated train={len(train):,}, golden={len(golden):,}; no split leakage.")


if __name__ == "__main__":
    main()