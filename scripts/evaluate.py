"""Evaluate intent and routing on a golden set without leaking it into retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support

from hiver_agent.classifier import KeywordIntentClassifier, MajorityClassifier, TfidfIntentClassifier, template_reply
from hiver_agent.generation import generate_reply_with_mode
from hiver_agent.retrieval import ResolutionRetriever
from hiver_agent.routing import decide_route


def score(name: str, truth: list[str], predicted: list[str]) -> dict:
    return {"model": name, "accuracy": round(float(accuracy_score(truth, predicted)), 3), "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 3)}


def routing_metrics(truth: list[str], predicted: list[str]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="binary", pos_label="auto_handle", zero_division=0
    )
    escalation_truth = [value == "escalate" for value in truth]
    escalation_predicted = [value == "escalate" for value in predicted]
    safety = sum(expected and actual for expected, actual in zip(escalation_truth, escalation_predicted)) / max(sum(escalation_truth), 1)
    return {
        "auto_handle_precision": round(float(precision), 3),
        "auto_handle_recall": round(float(recall), 3),
        "auto_handle_f1": round(float(f1), 3),
        "auto_handle_rate": round(float(sum(value == "auto_handle" for value in predicted) / len(predicted)), 3),
        "escalation_precision": round(float(precision_recall_fscore_support(truth, predicted, average="binary", pos_label="escalate", zero_division=0)[0]), 3),
        "escalation_recall": round(float(precision_recall_fscore_support(truth, predicted, average="binary", pos_label="escalate", zero_division=0)[1]), 3),
        "escalation_f1": round(float(precision_recall_fscore_support(truth, predicted, average="binary", pos_label="escalate", zero_division=0)[2]), 3),
        "escalation_safety": round(float(safety), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/annotations/train.csv")
    parser.add_argument("--golden", default="data/golden/golden_set.csv")
    parser.add_argument("--pairs", default="data/processed/spotify_pairs.csv")
    parser.add_argument("--output", default="artifacts/evaluation.json")
    parser.add_argument("--predictions", default="artifacts/full_system_predictions.csv")
    parser.add_argument("--failures", default="artifacts/failure_cases.csv")
    parser.add_argument("--live-replies", type=int, default=5, help="Maximum Gemini calls; remaining rows use the explicit safe fallback.")
    args = parser.parse_args()
    train = pd.read_csv(args.train).dropna(subset=["intent", "customer_text"])
    gold = pd.read_csv(args.golden).dropna(subset=["intent", "route", "customer_text"])
    if len(gold) < 150:
        raise SystemExit("Golden set must contain at least 150 completely labelled rows.")
    if set(gold.tweet_id.astype(str)) & set(train.tweet_id.astype(str)):
        raise SystemExit("Leakage detected: train and golden sets overlap by tweet_id.")
    majority = MajorityClassifier().fit(train.customer_text.tolist(), train.intent.tolist())
    keyword = KeywordIntentClassifier().fit(train.customer_text.tolist(), train.intent.tolist())
    tfidf = TfidfIntentClassifier().fit(train.customer_text.tolist(), train.intent.tolist())
    truth = gold.intent.tolist()
    majority_pred = [x.intent for x in majority.predict(gold.customer_text.tolist())]
    keyword_predictions = keyword.predict(gold.customer_text.tolist())
    keyword_pred = [x.intent for x in keyword_predictions]
    full_predictions = tfidf.predict(gold.customer_text.tolist())
    full_pred = [x.intent for x in full_predictions]
    retrieval_pairs = pd.read_csv(args.pairs)
    retrieval_pairs = retrieval_pairs[~retrieval_pairs.tweet_id.astype(str).isin(gold.tweet_id.astype(str))]
    retriever = ResolutionRetriever(retrieval_pairs)
    routes = []
    keyword_routes = []
    evidence_rows = []
    replies = []
    generation_modes = []
    for message, prediction in zip(gold.customer_text, full_predictions):
        evidence = retriever.search(message, k=1)[0]
        decision = decide_route(intent=prediction.intent, confidence=prediction.confidence, evidence_score=evidence.score, message=message)
        routes.append(decision.action)
        evidence_rows.append(evidence)
        reply, generation_mode = generate_reply_with_mode(message, prediction.intent, [evidence], use_gemini=len(replies) < args.live_replies)
        replies.append(reply)
        generation_modes.append(generation_mode)
    for message, prediction in zip(gold.customer_text, keyword_predictions):
        keyword_routes.append("escalate" if prediction.intent in {"account_access", "billing_payment", "other"} else "auto_handle")
    prediction_frame = gold[["tweet_id", "customer_text", "intent", "route"]].copy()
    prediction_frame["predicted_intent"] = full_pred
    prediction_frame["predicted_action"] = routes
    prediction_frame["intent_confidence"] = [round(x.confidence, 4) for x in full_predictions]
    prediction_frame["retrieved_evidence"] = [f"{x.tweet_id}: {x.agent_reply}" for x in evidence_rows]
    prediction_frame["retrieval_score"] = [round(x.score, 4) for x in evidence_rows]
    prediction_frame["draft_reply"] = replies
    prediction_frame["reply_released"] = [action == "auto_handle" for action in routes]
    prediction_frame["generation_mode"] = generation_modes
    Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(args.predictions, index=False)
    failures = prediction_frame[(prediction_frame.intent != prediction_frame.predicted_intent) | (prediction_frame.route != prediction_frame.predicted_action)].head(5).copy()
    failures["error"] = failures.apply(lambda row: "intent mismatch" if row.intent != row.predicted_intent else "routing mismatch", axis=1)
    failures["hypothesis"] = "Lexical similarity or coarse taxonomy may miss the message's primary issue."
    Path(args.failures).parent.mkdir(parents=True, exist_ok=True)
    failures.to_csv(args.failures, index=False)
    routing = routing_metrics(gold.route.tolist(), routes)
    route_precision, route_recall, route_f1, _ = precision_recall_fscore_support(gold.route, routes, average="binary", pos_label="auto_handle", zero_division=0)
    result = {
        "golden_n": len(gold),
        "annotation_source": sorted(set(gold.get("annotation_source", pd.Series(["unspecified"])).astype(str))),
        "intent_metrics": [score("majority", truth, majority_pred), score("keyword_rules", truth, keyword_pred), score("tfidf_logreg", truth, full_pred)],
        "routing": routing,
        "routing_by_system": {"majority_always_escalate": routing_metrics(gold.route.tolist(), ["escalate"] * len(gold)), "keyword_rules": routing_metrics(gold.route.tolist(), keyword_routes), "full_system": routing},
        "per_intent": classification_report(truth, full_pred, output_dict=True, zero_division=0),
        "generated_reply_count": sum(bool(reply) for reply in replies),
        "released_reply_count": sum(action == "auto_handle" for action in routes),
        "predictions_path": args.predictions,
        "failure_cases_path": args.failures,
        "notes": "Gold rows are excluded from retrieval. Human agreement is pending until completed ratings are supplied.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["intent_metrics"], indent=2))


if __name__ == "__main__":
    main()
