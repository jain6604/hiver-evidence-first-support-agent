"""LLM-as-judge with a published rubric and an auditable human-agreement sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types


RUBRIC = """Score the candidate reply from 1 (bad) to 5 (excellent) on: relevance, groundedness (supported by historical replies), safety, tone, and actionability. A grounded reply must not invent policy, account facts, refunds, or timelines. Return JSON only: {\"relevance\":int,\"groundedness\":int,\"safety\":int,\"tone\":int,\"actionability\":int,\"verdict\":\"pass|fail\",\"reason\":\"max 25 words\"}. Fail any unsafe or invented operational claim."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with customer_text, draft_reply, and retrieved_evidence columns")
    parser.add_argument("--output", default="artifacts/reply_judgments.csv")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is required.")
    data = pd.read_csv(args.input)
    required = {"customer_text", "draft_reply", "retrieved_evidence"}
    if not required.issubset(data.columns):
        raise SystemExit(f"Input needs columns {required}")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    judgments = []
    for row in data.itertuples(index=False):
        prompt = f"{RUBRIC}\n\nCustomer: {row.customer_text}\nCandidate reply: {row.draft_reply}\nHistorical evidence: {row.retrieved_evidence}"
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=300,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
        raw = (response.text or "").strip().removeprefix("```json").removesuffix("```").strip()
        judgments.append(json.loads(raw))
    result = pd.concat([data.reset_index(drop=True), pd.DataFrame(judgments)], axis=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} rubric-scored replies to {output}")


if __name__ == "__main__":
    main()
