"""Baselines and the reproducible intent classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .taxonomy import INTENTS


@dataclass
class Prediction:
    intent: str
    confidence: float


class MajorityClassifier:
    def fit(self, texts: list[str], labels: list[str]) -> "MajorityClassifier":
        self.label = max(set(labels), key=labels.count)
        return self

    def predict(self, texts: list[str]) -> list[Prediction]:
        return [Prediction(self.label, 1.0) for _ in texts]


class TfidfIntentClassifier:
    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30_000, sublinear_tf=True)),
                ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42)),
            ]
        )

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfIntentClassifier":
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: list[str]) -> list[Prediction]:
        probabilities = self.pipeline.predict_proba(texts)
        classes = self.pipeline.named_steps["model"].classes_
        return [Prediction(str(classes[np.argmax(row)]), float(np.max(row))) for row in probabilities]


KEYWORD_RULES = (
    ("billing_payment", ("charg", "refund", "payment", "bill", "premium", "price", "fee")),
    ("account_access", ("login", "log in", "password", "sign in", "account", "verify")),
    ("delivery_tracking", ("where is", "track", "delivery", "shipment", "arrive")),
    ("booking_order_change", ("cancel", "subscription", "subscribe", "plan", "change username")),
    ("product_service_issue", ("not work", "won't", "can't", "cant", "error", "crash", "issue", "pausing", "buffer")),
    ("information_request", ("how do", "how can", "where can", "what is", "can i")),
    ("complaint_feedback", ("terrible", "worst", "disappoint", "hate", "love", "feedback")),
)


TEMPLATE_REPLIES = {
    "account_access": "Thanks for reaching out. A support specialist will help with account access privately.",
    "billing_payment": "Thanks for flagging this. A support specialist will review the payment issue privately.",
    "booking_order_change": "Thanks for reaching out. A support specialist will help with that change.",
    "delivery_tracking": "Thanks for checking in. A support specialist will look into the delivery status.",
    "product_service_issue": "Sorry you are having trouble. A support specialist will help troubleshoot this.",
    "information_request": "Thanks for your question. A support specialist will share the relevant information.",
    "complaint_feedback": "Thanks for the feedback. A support specialist will review it.",
    "other": "Thanks for reaching out. A support specialist will review this and help.",
}


class KeywordIntentClassifier:
    """Transparent first-match keyword baseline; confidence is not calibrated."""

    def fit(self, texts: list[str], labels: list[str]) -> "KeywordIntentClassifier":
        return self

    def predict(self, texts: list[str]) -> list[Prediction]:
        predictions = []
        for text in texts:
            lowered = text.lower()
            intent = next((intent for intent, terms in KEYWORD_RULES if any(term in lowered for term in terms)), "other")
            predictions.append(Prediction(intent, 1.0 if intent != "other" else 0.0))
        return predictions


def template_reply(intent: str) -> str:
    if intent not in INTENTS:
        raise ValueError(f"Unknown intent: {intent!r}")
    return TEMPLATE_REPLIES[intent]
