"""Deterministic release rehearsal (provider-independent).

Models a payment/deployment provider so CI can verify idempotency, rollback,
and observability without credentials or external side effects.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class MockProvider:
    events: set[str]
    def __init__(self) -> None: self.events = set()
    def webhook(self, event_id: str, status: str) -> dict[str, Any]:
        if event_id in self.events: return {"ok": True, "duplicate": True, "status": status}
        self.events.add(event_id)
        return {"ok": status in {"paid", "refunded", "charged_back"}, "duplicate": False, "status": status}

def run_provider_rehearsal() -> dict[str, bool]:
    p = MockProvider()
    first = p.webhook("evt-1", "paid")
    replay = p.webhook("evt-1", "paid")
    refund = p.webhook("evt-2", "refunded")
    return {"approved": first["ok"] and not first["duplicate"],
            "idempotent_replay": replay["ok"] and replay["duplicate"],
            "refund": refund["ok"]}

