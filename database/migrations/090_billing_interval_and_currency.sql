-- Migration 090 — annual billing + a second settlement currency.
--
-- WHY: the commercial plan sells an annual, multi-asset bundle priced in USD, and the
-- ledger could represent neither. `checkout_orders.currency` carried `CHECK (currency =
-- 'COP')`, so a USD order was rejected by the database itself, and there was no notion of a
-- billing period at all — the webhook granted a hardcoded 30 days to every purchase.
--
-- SCOPE, deliberately narrow:
--   1. Widen the currency check to an explicit allowlist. It stays an allowlist, not a free
--      text column: an unrecognised currency must fail at the database, because everything
--      downstream (the sealed-quote amount comparison in the webhook) assumes the order and
--      the provider agree on units.
--   2. Add `billing_interval` with a default of 'month', so every existing row keeps its
--      current meaning and no backfill is required.
--
-- WHAT THIS MIGRATION DOES NOT TOUCH: `enforce_checkout_order_transition()`. The billing
-- test suite PARSES that function out of 059 to derive the legal order lifecycle (CXD-063),
-- so restating its body here would create a second source of truth for a money invariant —
-- the exact failure mode that migration was written to prevent. The new column's
-- immutability is enforced by its own trigger instead, which composes with the existing one
-- rather than replacing it.

ALTER TABLE checkout_orders DROP CONSTRAINT IF EXISTS checkout_orders_currency_check;
ALTER TABLE checkout_orders
  ADD CONSTRAINT checkout_orders_currency_check
  CHECK (currency IN ('COP', 'USD'));

ALTER TABLE checkout_orders
  ADD COLUMN IF NOT EXISTS billing_interval TEXT NOT NULL DEFAULT 'month';

ALTER TABLE checkout_orders DROP CONSTRAINT IF EXISTS checkout_orders_billing_interval_check;
ALTER TABLE checkout_orders
  ADD CONSTRAINT checkout_orders_billing_interval_check
  CHECK (billing_interval IN ('month', 'year'));

-- The interval is an ECONOMIC TERM of the sealed quote: changing it after the fact would
-- silently alter how much access a paid order buys. Same immutability rule as amount,
-- currency and plan, enforced separately so 059's lifecycle function stays untouched.
CREATE OR REPLACE FUNCTION enforce_checkout_order_interval_immutable() RETURNS trigger AS $$
BEGIN
  IF NEW.billing_interval IS DISTINCT FROM OLD.billing_interval THEN
    RAISE EXCEPTION 'checkout quote billing_interval is immutable (% -> %)',
      OLD.billing_interval, NEW.billing_interval;
  END IF;
  RETURN NEW;
END; $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_checkout_orders_interval_immutable ON checkout_orders;
CREATE TRIGGER trg_checkout_orders_interval_immutable
  BEFORE UPDATE ON checkout_orders
  FOR EACH ROW EXECUTE FUNCTION enforce_checkout_order_interval_immutable();

COMMENT ON COLUMN checkout_orders.billing_interval IS
  'Billing period the sealed quote was priced for: month | year. Drives the entitlement '
  'grant window in the billing webhook (was a hardcoded 30 days).';
