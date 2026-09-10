"""Evaluate intent and routing on a golden set without leaking it into retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support

from hiver_agent.classifier import MajorityClassifier, TfidfIntentClassifier
from hiver_agent.retrieval import ResolutionRetriever
from hiver_agent.routing import decide_route


def score(name: str, truth: list[str], predicted: list[str]) -> dict:
    return {"model": name, "accuracy": round(float(accuracy_score(truth, predicted)), 3), "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/annotations/train.csv")
    parser.add_argument("--golden", default="data/golden/golden_set.csv")
    parser.add_argument("--pairs", default="data/processed/spotify_pairs.csv")
    parser.add_argument("--output", default="artifacts/evaluation.json")
    args = parser.parse_args()
    train = pd.read_csv(args.train).dropna(subset=["intent", "customer_text"])
    gold = pd.read_csv(args.golden).dropna(subset=["intent", "route", "customer_text"])
    if len(gold) < 150:
        raise SystemExit("Golden set must contain at least 150 completely labelled rows.")
    if set(gold.tweet_id.astype(str)) & set(train.tweet_id.astype(str)):
        raise SystemExit("Leakage detected: train and golden sets overlap by tweet_id.")
    majority = MajorityClassifier().fit(train.customer_text.tolist(), train.intent.tolist())
    tfidf = TfidfIntentClassifier().fit(train.customer_text.tolist(), train.intent.tolist())
    truth = gold.intent.tolist()
    majority_pred = [x.intent for x in majority.predict(gold.customer_text.tolist())]
    full_predictions = tfidf.predict(gold.customer_text.tolist())
    full_pred = [x.intent for x in full_predictions]
    retrieval_pairs = pd.read_csv(args.pairs)
    retrieval_pairs = retrieval_pairs[~retrieval_pairs.tweet_id.astype(str).isin(gold.tweet_id.astype(str))]
    retriever = ResolutionRetriever(retrieval_pairs)
    routes = []
    for message, prediction in zip(gold.customer_text, full_predictions):
        evidence = retriever.search(message, k=1)[0]
        routes.append(decide_route(intent=prediction.intent, confidence=prediction.confidence, evidence_score=evidence.score, message=message).action)
    route_precision, route_recall, route_f1, _ = precision_recall_fscore_support(gold.route, routes, average="binary", pos_label="auto_handle", zero_division=0)
    result = {
        "golden_n": len(gold),
        "intent_metrics": [score("majority", truth, majority_pred), score("tfidf_logreg", truth, full_pred)],
        "routing": {"auto_handle_precision": round(float(route_precision), 3), "auto_handle_recall": round(float(route_recall), 3), "auto_handle_f1": round(float(route_f1), 3), "auto_handle_rate": round(float(sum(x == 'auto_handle' for x in routes) / len(routes)), 3)},
        "per_intent": classification_report(truth, full_pred, output_dict=True, zero_division=0),
        "notes": "Full system's reply quality is separately scored by scripts/judge_replies.py. Gold rows are excluded from retrieval.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["intent_metrics"], indent=2))


if __name__ == "__main__":
    main()
