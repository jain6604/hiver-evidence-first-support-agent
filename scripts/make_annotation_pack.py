"""Sample a stratified, leakage-resistant golden set for human labelling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hiver_agent.taxonomy import INTENTS


def coarse_bucket(text: str) -> str:
    value = text.lower()
    rules = {
        "billing_payment": ("charg", "refund", "payment", "bill", "premium", "price"),
        "account_access": ("login", "log in", "password", "account", "sign in"),
        "product_service_issue": ("not work", "error", "won't", "cant", "can't", "broken", "crash", "issue"),
        "booking_order_change": ("cancel", "subscription", "subscribe", "plan", "change"),
        "information_request": ("how do", "how can", "where can", "what is", "can i"),
        "complaint_feedback": ("terrible", "worst", "disappoint", "hate", "love"),
    }
    return next((label for label, terms in rules.items() if any(term in value for term in terms)), "other")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/processed/spotify_pairs.csv")
    parser.add_argument("--output", default="data/golden/golden_set.csv")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs)
    pairs["sampling_bucket"] = pairs.customer_text.map(coarse_bucket)
    quota = max(1, args.n // len(INTENTS))
    sampled = (
        pairs.groupby("sampling_bucket", group_keys=False)
        .apply(lambda group: group.sample(n=min(len(group), quota), random_state=args.seed), include_groups=False)
        .reset_index(drop=True)
    )
    if len(sampled) < args.n:
        remainder = pairs[~pairs.tweet_id.isin(sampled.tweet_id)].sample(n=args.n - len(sampled), random_state=args.seed)
        sampled = pd.concat([sampled, remainder], ignore_index=True)
    annotation = sampled[["tweet_id", "customer_text", "agent_reply", "sampling_bucket"]].head(args.n).copy()
    annotation["intent"] = ""
    annotation["route"] = ""
    annotation["route_reason"] = ""
    annotation["reply_grounded"] = ""  # yes / partly / no
    annotation["notes"] = ""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    annotation.to_csv(output, index=False)
    print(f"Wrote {len(annotation)} examples to {output}. Label intent from: {', '.join(INTENTS)}")


if __name__ == "__main__":
    main()
