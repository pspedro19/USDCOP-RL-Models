-- CXD-063: upgrade path for Wompi retry attempts on an existing reference.
--
-- 059 is the fresh-install baseline.  This additive migration is required for
-- databases where its trigger function already exists: Wompi can emit a
-- DECLINED transaction followed by an APPROVED transaction with the same
-- reference and a different transaction id.

DO $$
BEGIN
    IF to_regclass('public.checkout_orders') IS NULL THEN
        RAISE EXCEPTION
            'checkout_orders is absent; apply commerce-v1 from its reviewed baseline';
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_checkout_order_transition()
RETURNS trigger AS $$
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

  IF OLD.status = NEW.status THEN
    RETURN NEW;
  END IF;

  IF NOT (
      (OLD.status = 'created'
       AND NEW.status IN ('pending', 'paid', 'failed', 'cancelled'))
      OR
      (OLD.status = 'pending'
       AND NEW.status IN ('paid', 'failed', 'cancelled', 'expired'))
      OR
      -- A provider retry is a distinct, S2S-confirmed transaction.  The
      -- opposite edge paid->failed is deliberately absent, so a late decline
      -- can never degrade an already credited order.
      (OLD.status = 'failed' AND NEW.status IN ('paid'))
      OR
      (OLD.status = 'paid'
       AND NEW.status IN ('refunded', 'charged_back'))
  ) THEN
    RAISE EXCEPTION
        'illegal checkout order transition: % -> %',
        OLD.status,
        NEW.status;
  END IF;

  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION enforce_checkout_order_transition() IS
    'Immutable checkout lifecycle; permits S2S-confirmed failed->paid Wompi retry, never paid->failed.';
