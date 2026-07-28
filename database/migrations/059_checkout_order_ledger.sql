-- Checkout/order ledger: immutable server-side quote and provider correlation.
CREATE TABLE IF NOT EXISTS checkout_orders (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES sb_users(id) ON DELETE RESTRICT,
  plan TEXT NOT NULL,
  addon_assets JSONB NOT NULL DEFAULT '[]'::jsonb,
  amount_cents BIGINT NOT NULL CHECK (amount_cents >= 0),
  currency TEXT NOT NULL CHECK (currency = 'COP'),
  reference TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'created' CHECK (status IN ('created','pending','paid','failed','refunded','charged_back','cancelled','expired')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_checkout_orders_user ON checkout_orders(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS billing_events (
  id BIGSERIAL PRIMARY KEY,
  provider_event_id TEXT NOT NULL UNIQUE,
  order_reference TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload JSONB NOT NULL,
  received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Enforce legal lifecycle transitions (provider retries are idempotent).
CREATE OR REPLACE FUNCTION enforce_checkout_order_transition() RETURNS trigger AS $$
BEGIN
  IF NEW.user_id IS DISTINCT FROM OLD.user_id
     OR NEW.plan IS DISTINCT FROM OLD.plan
     OR NEW.addon_assets IS DISTINCT FROM OLD.addon_assets
     OR NEW.amount_cents IS DISTINCT FROM OLD.amount_cents
     OR NEW.currency IS DISTINCT FROM OLD.currency
     OR NEW.reference IS DISTINCT FROM OLD.reference
     OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
    RAISE EXCEPTION 'checkout quote identity and economic terms are immutable';
  END IF;
  IF OLD.status = NEW.status THEN RETURN NEW; END IF;
  IF NOT ((OLD.status='created' AND NEW.status IN ('pending','paid','failed','cancelled')) OR
          (OLD.status='pending' AND NEW.status IN ('paid','failed','cancelled','expired')) OR
          (OLD.status='failed' AND NEW.status IN ('paid')) OR
          (OLD.status='paid' AND NEW.status IN ('refunded','charged_back'))) THEN
    RAISE EXCEPTION 'illegal checkout order transition: % -> %', OLD.status, NEW.status;
  END IF;
  NEW.updated_at = now(); RETURN NEW;
END; $$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS checkout_order_transition ON checkout_orders;
CREATE TRIGGER checkout_order_transition BEFORE UPDATE ON checkout_orders
FOR EACH ROW EXECUTE FUNCTION enforce_checkout_order_transition();
