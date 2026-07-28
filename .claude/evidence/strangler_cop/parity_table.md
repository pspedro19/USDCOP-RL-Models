---
kind: audit
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - src/strangler/gates.py
  - config/migration/strangler_usdcop.yaml
  - airflow/dags/forecast_h5_l7_multiday_executor.py
---

# Tabla de paridad por capa — usdcop (BL-31)

Generada por `scripts/validation/check_strangler_parity.py status --write-evidence` el 2026-07-28T16:32:30.770058Z.
Plan: `config/migration/strangler_usdcop.yaml` v1.0.0. Ledger: `.claude/evidence/strangler_cop/parity_ledger.jsonl`.

| layer | state | generator | obs | green | days/req | may advance | blockers |
|---|---|---|---|---|---|---|---|
| ingest | NOT_STARTED | **null (BL-28)** | 0 | 0 | 0.00/14 | NO | no candidate generator declared (BL-28 owns it; until then there is no second path to compare against) |
| canon | NOT_STARTED | **null (BL-28)** | 0 | 0 | 0.00/14 | NO | predecessor 'ingest' is NOT_STARTED, not MIGRATED (§29.2); no candidate generator declared (BL-28 owns it; until then there is no second path to compare against) |
| verify | NOT_STARTED | **null (BL-28)** | 0 | 0 | 0.00/14 | NO | predecessor 'ingest' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'canon' is NOT_STARTED, not MIGRATED (§29.2); no candidate generator declared (BL-28 owns it; until then there is no second path to compare against) |
| features | NOT_STARTED | **null (BL-28)** | 0 | 0 | 0.00/14 | NO | predecessor 'ingest' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'canon' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'verify' is NOT_STARTED, not MIGRATED (§29.2); no candidate generator declared (BL-28 owns it; until then there is no second path to compare against) |
| dataset | NOT_STARTED | **null (BL-28)** | 0 | 0 | 0.00/14 | NO | predecessor 'ingest' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'canon' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'verify' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'features' is NOT_STARTED, not MIGRATED (§29.2); no candidate generator declared (BL-28 owns it; until then there is no second path to compare against) |
| train | NOT_STARTED | **null (BL-28)** | 0 | 0 | 0.00/14 | NO | predecessor 'ingest' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'canon' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'verify' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'features' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'dataset' is NOT_STARTED, not MIGRATED (§29.2); no candidate generator declared (BL-28 owns it; until then there is no second path to compare against) |
| gate | NOT_STARTED | **null (BL-28)** | 0 | 0 | 0.00/14 | NO | predecessor 'ingest' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'canon' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'verify' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'features' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'dataset' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'train' is NOT_STARTED, not MIGRATED (§29.2); no candidate generator declared (BL-28 owns it; until then there is no second path to compare against) |
| signal | NOT_STARTED | **null (BL-28)** | 0 | 0 | 0.00/14 | NO | predecessor 'ingest' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'canon' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'verify' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'features' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'dataset' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'train' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'gate' is NOT_STARTED, not MIGRATED (§29.2); no candidate generator declared (BL-28 owns it; until then there is no second path to compare against) |
| execute | NOT_STARTED | **null (BL-28)** | 0 | 0 | 0.00/14 | NO | predecessor 'ingest' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'canon' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'verify' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'features' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'dataset' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'train' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'gate' is NOT_STARTED, not MIGRATED (§29.2); predecessor 'signal' is NOT_STARTED, not MIGRATED (§29.2); no candidate generator declared (BL-28 owns it; until then there is no second path to compare against); chain not fully migrated yet: ['ingest', 'canon', 'verify', 'features', 'dataset', 'train', 'gate', 'signal'] (§29.4); no execution-readiness attestation on file — BL-30 (external execution service, pre-trade, idempotency, kill switch) must attest first (§29.4); §30 acceptance criteria not PASS: #1 [PENDING] (owner BL-17), #2 [PENDING] (owner BL-17), #3 [PENDING] (owner BL-21), #4 [PENDING] (owner BL-21), #5 [PENDING] (owner BL-22), #6 [PENDING] (owner BL-26), #7 [PENDING] (owner BL-35), #8 [PENDING] (owner BL-16), #9 [PENDING] (owner BL-30), #10 [PENDING] (owner BL-30), #11 [PENDING] (owner BL-32), #12 [PENDING] (owner BL-18), #13 [PENDING] (owner BL-23), #14 [PENDING] (owner BL-24), #15 [PENDING] (owner BL-25); the L7 rollback has not been rehearsed — the money layer migrates with a double net (§29.4 + §30.15) |

## §30 — criterios de aceptación pendientes

- #1 [PENDING] — owner BL-17
- #2 [PENDING] — owner BL-17
- #3 [PENDING] — owner BL-21
- #4 [PENDING] — owner BL-21
- #5 [PENDING] — owner BL-22
- #6 [PENDING] — owner BL-26
- #7 [PENDING] — owner BL-35
- #8 [PENDING] — owner BL-16
- #9 [PENDING] — owner BL-30
- #10 [PENDING] — owner BL-30
- #11 [PENDING] — owner BL-32
- #12 [PENDING] — owner BL-18
- #13 [PENDING] — owner BL-23
- #14 [PENDING] — owner BL-24
- #15 [PENDING] — owner BL-25

## Readiness de ejecución (BL-30)

- NO ATESTIGUADA — el servicio de ejecución externo es entregable de BL-30.
