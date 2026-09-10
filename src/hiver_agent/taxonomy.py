"""Brand-agnostic intent taxonomy used by the annotation and modelling pipeline."""

from __future__ import annotations

INTENTS = [
    "account_access",
    "billing_payment",
    "booking_order_change",
    "delivery_tracking",
    "product_service_issue",
    "information_request",
    "complaint_feedback",
    "other",
]

INTENT_DEFINITIONS = {
    "account_access": "Cannot sign in, reset credentials, verify identity, or access an account.",
    "billing_payment": "Charge, refund, payment, invoice, fare, fee, or price dispute.",
    "booking_order_change": "Create, cancel, modify, or correct a booking, reservation, or order.",
    "delivery_tracking": "Where is an order, shipment, ride, service person, or delivery?",
    "product_service_issue": "A product, app, website, connection, or service is broken or unavailable.",
    "information_request": "Asks how a policy, feature, product, or process works without a live failure.",
    "complaint_feedback": "Expresses dissatisfaction, praise, or feedback without a resolvable transaction request.",
    "other": "Does not fit safely into a supported intent.",
}

# Deliberately conservative: these intents are never automatically handled.
HIGH_RISK_INTENTS = {"billing_payment", "account_access", "other"}


def validate_intent(intent: str) -> str:
    if intent not in INTENTS:
        raise ValueError(f"Unknown intent: {intent!r}. Expected one of {INTENTS}")
    return intent
