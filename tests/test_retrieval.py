import pandas as pd

from hiver_agent.retrieval import ResolutionRetriever


def test_retriever_returns_scored_historical_pair() -> None:
    pairs = pd.DataFrame(
        [
            {"tweet_id": "1", "customer_text": "My music keeps pausing", "agent_reply": "Try reinstalling the app."},
            {"tweet_id": "2", "customer_text": "How do I change my plan", "agent_reply": "Visit your account page."},
        ]
    )
    result = ResolutionRetriever(pairs).search("music pauses in the app", k=1)
    assert result[0].tweet_id == "1"
    assert result[0].score > 0
