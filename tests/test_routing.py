from hiver_agent.routing import decide_route


def test_low_risk_high_evidence_can_auto_handle() -> None:
    outcome = decide_route(intent="information_request", confidence=0.90, evidence_score=0.70, message="How do I change my plan?")
    assert outcome.action == "auto_handle"


def test_payment_is_always_escalated() -> None:
    outcome = decide_route(intent="billing_payment", confidence=0.99, evidence_score=0.99, message="I was charged twice")
    assert outcome.action == "escalate"


def test_sensitive_marker_is_escalated() -> None:
    outcome = decide_route(intent="information_request", confidence=0.99, evidence_score=0.99, message="My __email__ was changed")
    assert outcome.action == "escalate"
