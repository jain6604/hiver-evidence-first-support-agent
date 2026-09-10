"""Transparent TF-IDF retrieval over historical inbound -> agent-response pairs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedExample:
    customer_text: str
    agent_reply: str
    score: float
    tweet_id: str


class ResolutionRetriever:
    def __init__(self, pairs: pd.DataFrame) -> None:
        required = {"customer_text", "agent_reply", "tweet_id"}
        if not required.issubset(pairs.columns):
            raise ValueError(f"Pairs must contain {required}")
        self.pairs = pairs.reset_index(drop=True).copy()
        min_df = 2 if len(self.pairs) >= 20 else 1
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=40_000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.pairs["customer_text"].fillna(""))

    def search(self, message: str, k: int = 3) -> list[RetrievedExample]:
        query = self.vectorizer.transform([message])
        scores = cosine_similarity(query, self.matrix).ravel()
        indices = scores.argsort()[::-1][:k]
        return [
            RetrievedExample(
                customer_text=str(self.pairs.iloc[i].customer_text),
                agent_reply=str(self.pairs.iloc[i].agent_reply),
                score=float(scores[i]),
                tweet_id=str(self.pairs.iloc[i].tweet_id),
            )
            for i in indices
        ]
