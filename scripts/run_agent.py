"""Run the support agent on one message and print an auditable decision."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from hiver_agent.classifier import TfidfIntentClassifier
from hiver_agent.generation import generate_reply
from hiver_agent.retrieval import ResolutionRetriever
from hiver_agent.routing import decide_route


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    parser.add_argument("--pairs", default="data/processed/spotify_pairs.csv")
    parser.add_argument("--labels", default="data/annotations/train.csv")
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs)
    labels = pd.read_csv(args.labels).dropna(subset=["intent"])
    classifier = TfidfIntentClassifier().fit(labels.customer_text.tolist(), labels.intent.tolist())
    prediction = classifier.predict([args.message])[0]
    evidence = ResolutionRetriever(pairs).search(args.message)
    decision = decide_route(intent=prediction.intent, confidence=prediction.confidence, evidence_score=evidence[0].score, message=args.message)
    reply = generate_reply(args.message, prediction.intent, evidence) if decision.action == "auto_handle" else None
    print(json.dumps({"intent": prediction.intent, "confidence": round(prediction.confidence, 3), "action": decision.action, "reason": decision.reason, "draft_reply": reply, "evidence": [{"tweet_id": x.tweet_id, "score": round(x.score, 3), "customer_text": x.customer_text, "agent_reply": x.agent_reply} for x in evidence]}, indent=2))


if __name__ == "__main__":
    main()
