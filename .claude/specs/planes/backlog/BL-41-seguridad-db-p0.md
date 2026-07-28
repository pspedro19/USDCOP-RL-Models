---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - services/signalbridge_api
  - database/migrations/067_spx500_regime_macro_vars.sql
---

# BL-41 — Seguridad DB P0: secret.*, credenciales consolidadas, timestamptz

**Fuente**: Plan Consolidado §8 (P0.1-P0.2) / plan 03 §3.3 · **Ola**: 1 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
DOS tablas de credenciales (sb_exchange_credentials, user_exchange_keys — ambas 0 filas, cifradas Vault) viven en public junto a research/news/BI; 15 columnas sb_* usan timestamp SIN zona; public permite CREATE a usuarios de app.

## Qué falta exactamente
Consolidar en UNA → secret.external_account (solo REFERENCIA; el secreto vive en Vault); esquema secret con roles propios, sin acceso desde Airflow genérico ni frontend; migrar sb_* a timestamptz; REVOKE CREATE ON SCHEMA public.

## Impacto frontend
Ninguno (transparente para la UI).

## Dependencias
Hacerlo AHORA que ambas tablas están vacías — antes de que se llenen. Empareja con BL-08.

## Verificación
Listado de public.* sin tablas de credenciales; test de rol: frontend_role sin SELECT en secret.*.

## Notas constitución
Fila 2 de la Readiness Matrix (BL-33).
