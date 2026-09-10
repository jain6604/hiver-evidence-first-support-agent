"""Apply approved human labels to golden CSV rows by tweet ID."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from hiver_agent.taxonomy import INTENTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/golden/golden_set.csv")
    parser.add_argument("--labels", nargs="+", help="tweet_id=intent|route|grounded")
    args = parser.parse_args()
    updates: dict[str, tuple[str, str, str]] = {}
    for value in args.labels:
        tweet_id, encoded = value.split("=", 1)
        intent, route, grounded = encoded.split("|", 2)
        if intent not in INTENTS or route not in {"auto_handle", "escalate"} or grounded not in {"yes", "partly", "no"}:
            raise SystemExit(f"Invalid label: {value}")
        updates[tweet_id] = (intent, route, grounded)
    path = Path(args.input)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if "annotation_source" not in fieldnames:
        fieldnames.append("annotation_source")
    found: set[str] = set()
    for row in rows:
        row.setdefault("annotation_source", "")
        if row.get("intent"):
            row["annotation_source"] = "human_reviewed"
        if row["tweet_id"] not in updates:
            continue
        intent, route, grounded = updates[row["tweet_id"]]
        row["intent"] = intent
        row["route"] = route
        row["route_reason"] = "Human-approved label recorded after explicit review."
        row["reply_grounded"] = grounded
        row["notes"] = "Human-approved label."
        row["annotation_source"] = "human_reviewed"
        found.add(row["tweet_id"])
    missing = set(updates) - found
    if missing:
        raise SystemExit(f"Tweet IDs not found: {sorted(missing)}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {len(found)} human labels in {path}")


if __name__ == "__main__":
    main()