"""Apply human reply-quality ratings to the 40-row review sample."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/golden/HUMAN_REVIEW_SAMPLE.csv")
    parser.add_argument("--ratings", nargs="+", help="tweet_id=relevance|groundedness|safety|tone|actionability|verdict")
    args = parser.parse_args()
    updates: dict[str, tuple[str, ...]] = {}
    for value in args.ratings:
        tweet_id, encoded = value.split("=", 1)
        scores = tuple(encoded.split("|"))
        if len(scores) != 6 or any(score not in {"1", "2", "3", "4", "5"} for score in scores[:5]) or scores[5] not in {"pass", "fail"}:
            raise SystemExit(f"Invalid rating: {value}")
        updates[tweet_id] = scores
    path = Path(args.input)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    found: set[str] = set()
    for row in rows:
        if row["tweet_id"] not in updates:
            continue
        relevance, groundedness, safety, tone, actionability, verdict = updates[row["tweet_id"]]
        row.update(
            {
                "human_relevance": relevance,
                "human_groundedness": groundedness,
                "human_safety": safety,
                "human_tone": tone,
                "human_actionability": actionability,
                "human_verdict": verdict,
            }
        )
        found.add(row["tweet_id"])
    missing = set(updates) - found
    if missing:
        raise SystemExit(f"Tweet IDs not found: {sorted(missing)}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {len(found)} human reply ratings in {path}")


if __name__ == "__main__":
    main()