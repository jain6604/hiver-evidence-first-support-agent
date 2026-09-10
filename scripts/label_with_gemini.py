"""Create draft labels for a development split; every draft remains reviewable in CSV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from hiver_agent.taxonomy import INTENT_DEFINITIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/processed/spotify_pairs.csv")
    parser.add_argument("--output", default="data/annotations/train_drafts.csv")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start", type=int, default=0, help="Offset into a seeded 2,000-row candidate pool.")
    parser.add_argument("--resume", action="store_true", help="Append only tweet IDs not already present in --output.")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is required. Put it in .env; never commit .env.")
    pairs = pd.read_csv(args.pairs)
    pool = pairs.sample(n=min(2_000, len(pairs)), random_state=args.seed)
    rows = pool.iloc[args.start : args.start + args.n][["tweet_id", "customer_text"]].copy()
    output = Path(args.output)
    if args.resume and output.exists():
        existing = pd.read_csv(output, dtype={"tweet_id": str})
        rows = rows[~rows.tweet_id.astype(str).isin(existing.tweet_id.astype(str))]
    if rows.empty:
        print("No new rows to label.")
        return
    definitions = json.dumps(INTENT_DEFINITIONS, indent=2)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    labels: list[dict[str, str]] = []
    for start in range(0, len(rows), 20):
        batch = rows.iloc[start : start + 20]
        payload = batch.to_dict("records")
        prompt = f"""Classify each support message using exactly one taxonomy label. Taxonomy: {definitions}\nReturn strict JSON only: an array of objects with keys tweet_id, intent, rationale (max 18 words).\nMessages: {json.dumps(payload)}"""
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=1_500,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
        text = (response.text or "").strip().removeprefix("```json").removesuffix("```").strip()
        labels.extend(json.loads(text))
        print(f"Draft-labelled {min(start + 20, len(rows))}/{len(rows)}", flush=True)
    labelled = rows.merge(pd.DataFrame(labels), on="tweet_id", how="left")
    labelled["reviewed"] = "no"
    labelled["annotation_source"] = "ai_assisted_gemini_draft"
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and output.exists():
        labelled = pd.concat([pd.read_csv(output), labelled], ignore_index=True).drop_duplicates("tweet_id", keep="last")
    labelled.to_csv(output, index=False)
    print(f"Wrote reviewable Gemini drafts to {output}")


if __name__ == "__main__":
    main()
