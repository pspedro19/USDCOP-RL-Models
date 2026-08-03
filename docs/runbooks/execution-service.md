# Execution service runbook

The execution service accepts only an approved row from `portfolio.target`.
Airflow may publish targets but never supplies broker credentials or submits an
order.

## Startup

1. Load broker credentials by Vault reference from `secret.external_account`.
2. Confirm direct database access to `portfolio.target`,
   `exec.v_effective_kill_switch` and the append-only `exec.*` ledger.
3. Run pre-operation reconciliation before accepting the first target.
4. Leave openings disabled if reconciliation is not `RECONCILED`.

## Kill-switch behaviour

| Level | Required action |
|---|---|
| `CLEAR` | Normal pre-trade checks |
| `BLOCK_NEW` | Reject openings; reductions remain possible |
| `CANCEL_OPEN` | Block openings and cancel broker working orders |
| `EXIT_ALL` | Cancel working orders and flatten positions |
| `ACCOUNT_FREEZE` | Preserve the account in quarantine after flattening |

The service reads this state directly from PostgreSQL before every opening.
Airflow health is not part of this control path.

## Reconciliation

- `PRE_OPERATION`: compare internal and broker positions before each target.
- `INTRADAY`: compare orders, fills, position and cash on a recurring service timer.
- `EOD`: seal the broker snapshot and reconcile facts after market close.
- Any mismatch publishes `exec.reconciliation_event` and activates `BLOCK_NEW`.

Never repair a discrepancy by updating historical orders or fills. Publish a
correction event, resolve the incident, reconcile again and explicitly clear
the kill switch with an authorized actor.

## Recovery

Retries reuse the deterministic idempotency key
`account + instrument + target_version + rebalance_cutoff`. A matching existing
order is returned as `IDEMPOTENT_REPLAY`; it is not submitted again.
