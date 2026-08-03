-- Idempotency ledger for payment webhooks. Provider retries must not extend
-- entitlements or duplicate audit entries.
CREATE TABLE IF NOT EXISTS billing_webhook_events (
  id BIGSERIAL PRIMARY KEY,
  reference TEXT NOT NULL,
  event_type TEXT NOT NULL,
  received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (reference, event_type)
);
CREATE INDEX IF NOT EXISTS idx_billing_webhook_events_received
  ON billing_webhook_events (received_at DESC);
