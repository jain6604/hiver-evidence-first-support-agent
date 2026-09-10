"""Safe automation policy. It is intentionally more selective than the classifier."""

from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import HIGH_RISK_INTENTS

SENSITIVE_MARKERS = ("__email__", "__phone__", "password", "card", "bank", "fraud", "legal", "emergency")


@dataclass(frozen=True)
class RoutingDecision:
    action: str
    reason: str


def decide_route(*, intent: str, confidence: float, evidence_score: float, message: str) -> RoutingDecision:
    """Return an auditable action: auto_handle only when all safety gates pass."""
    lowered = message.lower()
    marker = next((m for m in SENSITIVE_MARKERS if m in lowered), None)
    if marker:
        return RoutingDecision("escalate", f"Sensitive or high-stakes marker detected: {marker}.")
    if intent in HIGH_RISK_INTENTS:
        return RoutingDecision("escalate", f"{intent} is excluded from automation by policy.")
    if confidence < 0.70:
        return RoutingDecision("escalate", f"Intent confidence {confidence:.2f} is below the 0.70 threshold.")
    if evidence_score < 0.10:
        return RoutingDecision("escalate", f"Historical-resolution evidence score {evidence_score:.2f} is below 0.10.")
    return RoutingDecision("auto_handle", "Supported low-risk intent with confident classification and relevant historical evidence.")
