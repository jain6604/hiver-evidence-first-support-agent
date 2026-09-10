"""Create clearly labelled AI-assisted bootstrap annotations for a fast smoke test."""
import pandas as pd
from hiver_agent.routing import decide_route

def label(text: str) -> str:
    text = text.lower()
    rules = [
        ("account_access", ["login", "log in", "password", "sign in", "account"]),
        ("billing_payment", ["charg", "refund", "payment", "bill", "premium", "price"]),
        ("booking_order_change", ["cancel", "subscription", "subscribe", "plan", "change username"]),
        ("product_service_issue", ["not work", "won't", "can't", "cant", "error", "crash", "issue", "pausing", "buffer"]),
        ("information_request", ["how do", "how can", "where can", "what is", "can i"]),
        ("complaint_feedback", ["terrible", "worst", "disappoint", "hate", "love"]),
    ]
    return next((intent for intent, keys in rules if any(key in text for key in keys)), "other")

def main() -> None:
    pairs = pd.read_csv("data/processed/spotify_pairs.csv")
    gold = pd.read_csv("data/golden/golden_set.csv")
    train = pairs[~pairs.tweet_id.astype(str).isin(gold.tweet_id.astype(str))].sample(800, random_state=7)[["tweet_id", "customer_text"]].copy()
    train["intent"] = train.customer_text.map(label)
    train["annotation_source"] = "ai_assisted_heuristic"
    train.to_csv("data/annotations/train.csv", index=False)
    gold["intent"] = gold.customer_text.map(label)
    gold["route"] = [decide_route(intent=intent, confidence=0.99, evidence_score=0.99, message=message).action for intent, message in zip(gold.intent, gold.customer_text)]
    gold["route_reason"] = "AI-assisted bootstrap label; requires human review before production use."
    gold["annotation_source"] = "ai_assisted_heuristic"
    gold.to_csv("data/golden/golden_set_ai_assisted.csv", index=False)

if __name__ == "__main__":
    main()
