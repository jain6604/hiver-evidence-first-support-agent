"""Baselines and the reproducible intent classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


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
