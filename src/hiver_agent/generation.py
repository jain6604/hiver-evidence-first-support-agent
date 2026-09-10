"""Gemini-backed response drafting with strict evidence and non-invention rules."""

from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv

from .retrieval import RetrievedExample


def build_prompt(message: str, intent: str, examples: Iterable[RetrievedExample]) -> str:
    evidence = "\n\n".join(
        f"Example {i + 1}\nCustomer: {x.customer_text}\nHistorical agent reply: {x.agent_reply}"
        for i, x in enumerate(examples)
    )
    return f"""You draft a concise social-media support reply.\n\nCustomer message: {message}\nPredicted intent: {intent}\n\nUse only the resolution pattern supported by the examples below. Do not invent account status, policies, timelines, refunds, or actions. Do not request passwords, payment-card numbers, or personally identifying information. If the evidence cannot support a useful resolution, say that a support specialist will follow up. Return only the customer-facing reply, under 280 characters.\n\nHistorical examples:\n{evidence}"""


def generate_reply(message: str, intent: str, examples: list[RetrievedExample]) -> str:
    """Call Gemini only when a key is configured; otherwise provide a deterministic fallback."""
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return "Thanks for reaching out. A support specialist will review this and help you as soon as possible."
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(message, intent, examples),
        config=types.GenerateContentConfig(
            max_output_tokens=180,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )
    text = (response.text or "").strip()
    return text[:280] if text else "Thanks for reaching out. A support specialist will review this and help you."
