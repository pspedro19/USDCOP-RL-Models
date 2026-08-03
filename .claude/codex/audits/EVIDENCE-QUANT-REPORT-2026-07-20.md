---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - .claude/codex/harness/tests/test_assurance_contracts.py
---

# Evidencia cuantitativa y de features

Corrida dirigida de 106 items: **65 passed, 14 failed, 26 errors, 1 skipped** (7.08 s).

- Paridad, orden, selección y validadores OHLCV ejecutables quedaron verdes.
- Ocho assurance gates fallaron: pago, ledger, IA, SSOT 15/20, DQ legado, manifest cuantitativo,
  thresholds económicos y ModelSKU.
- Seis fallos y 26 errores son de `test_l2_data_quality_report.py`: falta el módulo importable
  `services.l2_data_quality_report`.
- Un test macro fue omitido porque el dataset loader no está disponible.

Decisión: **NO-GO** para afirmar producción cuantitativa o venta de modelos. No convertir fallos en skips.
